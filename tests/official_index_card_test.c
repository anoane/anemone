/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Full-capacity and empty-step all-visible selections. Score/key/query inputs
 * contain no usable values; this pass depends only on checked step metadata.
 * Every guard and input byte is protected; candidate readers never write the
 * owner's candidate table. */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "anemone/official_index.h"
#include "linear_card_fixture.h"

#define WORK_BYTES (1ull << 20u)
#define WEIGHT_BYTES (1ull << 16u)
#define TIME_LIMIT_SECONDS 120u
#define REPEATS 2u

static bool check_case(struct test_linear_card *card, uint32_t kernel,
		       uint32_t position, uint32_t ratio, uint32_t role,
		       bool empty)
{
	test_linear_card_clear(card);
	test_linear_context("all-visible position %u ratio %u role %u empty %u",
			    position, ratio, role, empty ? 1u : 0u);
	const uint32_t context = MODEL_SELECTED;
	struct index_scan_description description = {
		.step = test_linear_card_work(
			card, STEP_BLOCK_BYTES_VALUE + STEP_ROW_BYTES_VALUE),
		.keys = test_linear_card_work(card, (uint64_t)context / ratio *
							    MODEL_KEY_BYTES),
		.queries = test_linear_card_work(card,
						 MODEL_INDEX_QUERIES_ROW_BYTES),
		.weights = test_linear_card_work(card,
						 MODEL_INDEX_WEIGHTS_ROW_BYTES),
		.scores = test_linear_card_work(
			card, (uint64_t)MODEL_CONTEXT_WITHOUT_CANDIDATES *
				      INDEX_SCORE_BYTES),
		.selection =
			test_linear_card_work(card, MODEL_SELECTION_ROW_BYTES),
		.candidates =
			test_linear_card_work(card, MODEL_CANDIDATES_ROW_BYTES),
		.context_token_limit = context,
		.session_capacity = 1,
		.ratio = ratio,
		.row_count = 1,
		.candidate_role = role
	};
	if (!test_linear_card_has_room(card))
		return false;
	const struct step_block step = { .sequence = 1,
					 .row_count = empty ? 0 : 1 };
	const struct step_row_record row = { .position = (int32_t)position,
					     .flags = STEP_ROW_LOGITS };
	bool passed = test_linear_card_put(card, description.step, 0, &step,
					   sizeof(step)) &&
		      test_linear_card_put(card, description.step, sizeof(step),
					   &row, sizeof(row));
	struct kernel_launch launch;
	passed = passed &&
		 test_linear_expect_status(
			 official_index_prepare(&description,
						card->arena_byte_count, kernel,
						&launch),
			 STATUS_OK,
			 "prepare score-free chronological selection");
	int32_t selected[MODEL_SELECTED];
	uint32_t candidates[MODEL_CANDIDATE_BLOCKS];
	const uint32_t visible = (position + 1u) / ratio;
	const uint32_t blocks =
		(visible + MODEL_KEYS_PER_TILE - 1u) / MODEL_KEYS_PER_TILE;
	for (uint32_t entry = 0; entry < MODEL_SELECTED; entry++)
		selected[entry] = entry < visible ? (int32_t)entry : -1;
	for (uint32_t block = 0; block < MODEL_CANDIDATE_BLOCKS; block++)
		candidates[block] = block < blocks ? block : MODEL_NO_CANDIDATE;
	for (uint32_t repeat = 0; passed && repeat < REPEATS; repeat++) {
		passed = test_linear_card_load(card) &&
			 test_linear_card_run(card, &launch, 1) &&
			 test_linear_card_fetch(card);
		const struct linear_range inputs[] = {
			description.step, description.keys, description.queries,
			description.weights, description.scores
		};
		for (uint32_t input = 0;
		     passed && input < sizeof(inputs) / sizeof(inputs[0]);
		     input++)
			passed = test_linear_card_expect_untouched(
				card, inputs[input],
				"selection input preserved");
		if (empty)
			passed = passed &&
				 test_linear_card_expect_untouched(
					 card, description.selection,
					 "empty selection remains untouched");
		else
			passed = passed &&
				 test_linear_card_expect_bytes(
					 card, description.selection, 0,
					 selected, sizeof(selected),
					 "every visible entry in source order");
		if (!empty && role == INDEX_CANDIDATES_SOURCE)
			passed = passed &&
				 test_linear_card_expect_bytes(
					 card, description.candidates, 0,
					 candidates, sizeof(candidates),
					 "all visible candidate blocks");
		else
			passed =
				passed &&
				test_linear_card_expect_untouched(
					card, description.candidates,
					"reader or empty step preserves owner candidates");
	}
	return passed;
}

int main(void)
{
	(void)alarm(TIME_LIMIT_SECONDS);
	static const uint32_t positions[] = { 0,   1,	127,
					      128, 255, MODEL_SELECTED - 1u };
	struct test_linear_card card = { 0 };
	bool passed = test_linear_card_open(&card, 0, WORK_BYTES, WEIGHT_BYTES);
	uint32_t kernel = BACKEND_NO_KERNEL;
	passed = passed &&
		 test_linear_expect_status(
			 official_index_kernel_prepare(&backend_cuda_value,
						       card.card, &kernel),
			 STATUS_OK, "prepare all-visible kernel");
	for (uint32_t role = INDEX_CANDIDATES_NONE;
	     passed && role <= INDEX_CANDIDATES_READER; role++) {
		for (uint32_t ratio = MODEL_DIRECT_RATIO;
		     passed && ratio <= MODEL_POOLED_RATIO; ratio++) {
			if (role != INDEX_CANDIDATES_NONE &&
			    ratio != MODEL_DIRECT_RATIO)
				continue;
			for (uint32_t index = 0;
			     passed &&
			     index < sizeof(positions) / sizeof(positions[0]);
			     index++)
				passed = check_case(&card, kernel,
						    positions[index], ratio,
						    role, false) &&
					 check_case(&card, kernel,
						    positions[index], ratio,
						    role, true);
		}
	}
	test_linear_card_close(&card);
	(void)printf("official_index_card_test: %s; %d failures\n",
		     passed ? "passed" : "failed", test_linear_failures());
	return passed && test_linear_failures() == 0 ? EXIT_SUCCESS :
						       EXIT_FAILURE;
}
