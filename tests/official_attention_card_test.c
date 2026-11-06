/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Closed-form independent core controls: zero queries make every valid key's
 * probability one. Constant values and a zero sink therefore yield v*n/(n+1),
 * independent of any reference implementation. Cases cross K64 updates and
 * the window/pool boundary. This checks guards/repetition, not model admission. */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "anemone/official_attention.h"
#include "linear_card_fixture.h"

#define WORK_BYTES (8ull << 20u)
#define WEIGHT_BYTES (1ull << 16u)
#define CONTEXT_TOKENS 512u
#define TIME_LIMIT_SECONDS 120u
#define CONSTANT_KEY_VALUE 4.0f
#define CONSTANT_KEY_BFLOAT16 0x4080u
#define CONSTANT_KEY_HALF 0x4400u
#define COMPLEX_COMPONENTS 2u
#define HALF_WORD_BYTES 2u
#define SMALL_CASES 2u
#define ARGUMENT_LIMIT 2

static bool fill_words(struct test_linear_card *card, struct linear_range range,
		       uint16_t value)
{
	for (uint64_t offset = 0; offset < range.byte_count;
	     offset += sizeof(value)) {
		if (!test_linear_card_put(card, range, offset, &value,
					  sizeof(value)))
			return false;
	}
	return true;
}

static bool check_case(struct test_linear_card *card, uint32_t kernel,
		       uint32_t position, uint32_t ratio, uint32_t types)
{
	test_linear_card_clear(card);
	test_linear_context(
		"official sparse core position %u ratio %u types %u", position,
		ratio, types);
	const uint64_t pool_bytes = ratio == 0 ?
					    MODEL_STATE_ROW_BYTES :
					    (uint64_t)CONTEXT_TOKENS / ratio *
						    MODEL_STATE_ROW_BYTES;
	struct attention_core_description description = {
		.step = test_linear_card_work(
			card, STEP_BLOCK_BYTES_VALUE + STEP_ROW_BYTES_VALUE),
		.queries = test_linear_card_work(card, MODEL_HEADS_ROW_BYTES),
		.ring = test_linear_card_work(card,
					      (uint64_t)MODEL_RING_SLOTS *
						      MODEL_STATE_ROW_BYTES),
		.pool = test_linear_card_work(card, pool_bytes),
		.selection =
			test_linear_card_work(card, MODEL_SELECTION_ROW_BYTES),
		.sinks = test_linear_card_work(
			card, (uint64_t)MODEL_HEADS_VALUE * sizeof(float)),
		.rotation =
			test_linear_card_work(card, MODEL_ROTATION_ROW_BYTES),
		.output = test_linear_card_work(card, MODEL_HEADS_ROW_BYTES),
		.row_count = 1,
		.session_capacity = 1,
		.context_token_limit = CONTEXT_TOKENS,
		.pool_ratio = ratio,
		.head_count = MODEL_HEADS_VALUE,
		.order = LINEAR_ORDER_REFERENCE,
		.arithmetic_class = ARITHMETIC_CLASS_SM_80
	};
	if (!test_linear_card_has_room(card))
		return false;
	const struct step_block step = { .sequence = 1, .row_count = 1 };
	const struct step_row_record row = { .position = (int32_t)position,
					     .flags = STEP_ROW_LOGITS };
	bool passed =
		test_linear_card_put(card, description.step, 0, &step,
				     sizeof(step)) &&
		test_linear_card_put(card, description.step, sizeof(step), &row,
				     sizeof(row)) &&
		fill_words(card, description.ring,
			   (types & OFFICIAL_ATTENTION_RING_BFLOAT16) != 0 ?
				   CONSTANT_KEY_BFLOAT16 :
				   CONSTANT_KEY_HALF) &&
		fill_words(card, description.pool,
			   (types & OFFICIAL_ATTENTION_POOL_BFLOAT16) != 0 ?
				   CONSTANT_KEY_BFLOAT16 :
				   CONSTANT_KEY_HALF);
	const uint32_t visible = ratio == 0 ? 0 : (position + 1u) / ratio;
	for (uint32_t slot = 0; passed && slot < MODEL_SELECTED; slot++) {
		const int32_t selected = slot < visible ? (int32_t)slot : -1;
		passed = test_linear_card_put(card, description.selection,
					      (uint64_t)slot * sizeof(selected),
					      &selected, sizeof(selected));
	}
	for (uint64_t offset = 0;
	     passed && offset < description.rotation.byte_count;
	     offset += COMPLEX_COMPONENTS * sizeof(float)) {
		const float cosine = 1.0f;
		passed = test_linear_card_put(card, description.rotation,
					      offset, &cosine, sizeof(cosine));
	}
	struct kernel_launch launch;
	passed = passed &&
		 test_linear_expect_status(
			 official_attention_prepare(&description, types,
						    card->arena_byte_count,
						    kernel, &launch),
			 STATUS_OK, "prepare independent sparse core");
	const uint32_t window = position + 1u < MODEL_WINDOW ? position + 1u :
							       MODEL_WINDOW;
	const uint32_t key_count = window + visible;
	const float result = CONSTANT_KEY_VALUE * (float)key_count /
			     ((float)key_count + 1.0f);
	const uint16_t expected = test_linear_bfloat16_nearest((double)result);
	for (uint32_t repeat = 0; passed && repeat < COMPLEX_COMPONENTS;
	     repeat++) {
		passed = test_linear_card_load(card) &&
			 test_linear_card_run(card, &launch, 1) &&
			 test_linear_card_fetch(card);
		const struct linear_range inputs[] = {
			description.step,      description.queries,
			description.ring,      description.pool,
			description.selection, description.sinks,
			description.rotation
		};
		for (uint32_t input = 0;
		     passed && input < sizeof(inputs) / sizeof(inputs[0]);
		     input++)
			passed = test_linear_card_expect_untouched(
				card, inputs[input], "core input preserved");
		for (uint64_t offset = 0;
		     passed && offset < description.output.byte_count;
		     offset += HALF_WORD_BYTES)
			passed = test_linear_card_expect_bytes(
				card, description.output, offset, &expected,
				sizeof(expected), "closed-form BF16 quotient");
	}
	return passed;
}

int main(int count, char **arguments)
{
	(void)alarm(TIME_LIMIT_SECONDS);
	if (count > ARGUMENT_LIMIT ||
	    (count == ARGUMENT_LIMIT && strcmp(arguments[1], "--small") != 0))
		return EXIT_FAILURE;
	static const uint32_t positions[] = { 0, 1, 63, 64, 127, 128, 255 };
	static const uint32_t ratios[] = { 0, MODEL_DIRECT_RATIO,
					   MODEL_POOLED_RATIO };
	static const uint32_t types[] = {
		OFFICIAL_ATTENTION_QUERY_BFLOAT16,
		OFFICIAL_ATTENTION_QUERY_BFLOAT16 |
			OFFICIAL_ATTENTION_RING_BFLOAT16,
		OFFICIAL_ATTENTION_OPERAND_TYPES
	};
	const uint32_t cases =
		count == ARGUMENT_LIMIT ?
			SMALL_CASES :
			(uint32_t)(sizeof(positions) / sizeof(positions[0]));
	struct test_linear_card card = { 0 };
	bool passed = test_linear_card_open(&card, 0, WORK_BYTES, WEIGHT_BYTES);
	uint32_t kernel = BACKEND_NO_KERNEL;
	passed = passed &&
		 test_linear_expect_status(
			 official_attention_kernel_prepare(&backend_cuda_value,
							   card.card, &kernel),
			 STATUS_OK, "prepare release core kernel");
	for (uint32_t position = 0; passed && position < cases; position++)
		for (uint32_t ratio = 0;
		     passed && ratio < sizeof(ratios) / sizeof(ratios[0]);
		     ratio++)
			for (uint32_t type = 0;
			     passed && type < sizeof(types) / sizeof(types[0]);
			     type++)
				passed = check_case(&card, kernel,
						    positions[position],
						    ratios[ratio], types[type]);
	test_linear_card_close(&card);
	(void)printf("official_attention_card_test: %s; %d failures\n",
		     passed ? "passed" : "failed", test_linear_failures());
	return passed && test_linear_failures() == 0 ? EXIT_SUCCESS :
						       EXIT_FAILURE;
}
