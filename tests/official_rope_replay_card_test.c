/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Original operands and coefficients are replayed at both supported strides,
 * all native row prefixes, and both alias modes. The unrotated prefix is a
 * synthetic preservation control. No GEMM, norm or frequency generator runs.
 */
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "anemone/official_rope.h"
#include "capture_reader.h"
#include "capture_writer.h"
#include "card_timing.h"
#include "linear_card_fixture.h"

#define WORK_CAPACITY (2ull << 20u)
#define WEIGHTS_CAPACITY (1ull << 16u)
#define SITES 3u
#define WIDTHS 2u
#define ALIAS_MODES 2u
#define ROW_SUM 16u
#define NAME_BYTES TEST_CAPTURE_WRITER_NAME_CAPACITY
#define TIME_LIMIT_SECONDS 300u
#define TIME_SAMPLES 9u
#define TIME_REPLAYS 128u
#define TIME_WARM 8u
#define VECTOR_DIMENSIONS 2u

enum argument {
	PROGRAM_ARGUMENT,
	MODE_ARGUMENT,
	REFERENCE_ARGUMENT,
	OUTPUT_ARGUMENT,
	ARGUMENTS
};

static const uint32_t row_counts[] = { 1, 3, 4, 8 };
static const uint32_t widths[] = { OFFICIAL_ROPE_WIDTH, MODEL_HEAD_DIMS };

struct replay {
	struct test_capture *reference;
	struct test_capture_writer *writer;
	struct test_linear_card card;
	uint32_t kernel;
	uint32_t records[SITES][WIDTHS][LINEAR_MAXIMUM_ROWS + 1u][ALIAS_MODES]
			[LINEAR_MAXIMUM_ROWS];
};

static bool plan_output(struct replay *replay, const char *folder)
{
	if (!test_linear_expect_status(
		    test_capture_writer_create(SITES * WIDTHS * ALIAS_MODES *
						       ROW_SUM,
					       &replay->writer),
		    STATUS_OK, "create rotation writer"))
		return false;
	for (uint32_t call = 0; call < SITES; call++) {
		const uint32_t heads = call == 1 ? 1 : MODEL_HEADS_VALUE;
		for (uint32_t index = 0; index < WIDTHS; index++) {
			for (uint32_t count = 0;
			     count < sizeof(row_counts) / sizeof(row_counts[0]);
			     count++) {
				const uint32_t rows = row_counts[count];
				for (uint32_t alias = 0; alias < ALIAS_MODES;
				     alias++) {
					for (uint32_t row = 0; row < rows;
					     row++) {
						char name[NAME_BYTES];
						(void)snprintf(
							name, sizeof(name),
							"rope_%u.width_%u.rows_%u.in_place_%u.position_%u.output",
							call, widths[index],
							rows, alias, row);
						const uint64_t dimensions[] = {
							heads, widths[index]
						};
						if (!test_linear_expect_status(
							    test_capture_writer_add(
								    replay->writer,
								    name,
								    TEST_CAPTURE_BFLOAT16,
								    dimensions,
								    VECTOR_DIMENSIONS,
								    heads, 0,
								    row,
								    "Original RoPE replay",
								    &replay->records
									     [call]
									     [index]
									     [rows]
									     [alias]
									     [row]),
							    STATUS_OK,
							    "add rotation result"))
							return false;
					}
				}
			}
		}
	}
	return test_linear_expect_status(
		test_capture_writer_open(replay->writer, folder), STATUS_OK,
		"open rotation output");
}

static bool read_float(struct replay *replay, const char *name, uint64_t offset,
		       uint64_t count, float *values)
{
	const struct test_capture_tensor *const tensor =
		test_capture_find(replay->reference, name);
	return test_linear_expect(
		       tensor != NULL && tensor->type == TEST_CAPTURE_FLOAT32 &&
			       offset <= tensor->element_count &&
			       count <= tensor->element_count - offset,
		       name) &&
	       test_linear_expect_status(
		       test_capture_read(
			       replay->reference, name, TEST_CAPTURE_FLOAT32,
			       tensor->dimensions, tensor->dimension_count,
			       offset, count, values, count * sizeof(float)),
		       STATUS_OK, name);
}

static bool read_vectors(struct replay *replay, uint32_t call, uint32_t row,
			 uint32_t heads, uint32_t width, const char *part,
			 uint16_t *values)
{
	char name[NAME_BYTES];
	(void)snprintf(name, sizeof(name),
		       "layer_0.position_%" PRIu32 ".rope_%" PRIu32 ".%s", row,
		       call, part);
	uint16_t stored[MODEL_HEADS_VALUE * OFFICIAL_ROPE_WIDTH];
	const uint32_t rotary_values = heads * OFFICIAL_ROPE_WIDTH;
	const struct test_capture_tensor *const tensor =
		test_capture_find(replay->reference, name);
	if (!test_linear_expect(tensor != NULL &&
					tensor->type == TEST_CAPTURE_BFLOAT16 &&
					tensor->element_count == rotary_values,
				name) ||
	    !test_linear_expect_status(
		    test_capture_read(
			    replay->reference, name, TEST_CAPTURE_BFLOAT16,
			    tensor->dimensions, tensor->dimension_count, 0,
			    rotary_values, stored,
			    (uint64_t)rotary_values * sizeof(uint16_t)),
		    STATUS_OK, name))
		return false;
	for (uint32_t head = 0; head < heads; head++) {
		for (uint32_t column = 0; column < width; column++) {
			if (column < width - OFFICIAL_ROPE_WIDTH) {
				values[(size_t)head * width + column] =
					test_linear_bfloat16_nearest(
						(double)(column + head));
				continue;
			}
			const uint16_t bits =
				stored[head * OFFICIAL_ROPE_WIDTH + column -
				       width + OFFICIAL_ROPE_WIDTH];
			if (!test_linear_expect(
				    test_linear_bfloat16_is_finite(bits),
				    "captured rotary value is finite BF16"))
				return false;
			values[(size_t)head * width + column] = bits;
		}
	}
	return true;
}

static bool time_rotation(struct replay *replay,
			  const struct kernel_launch *launch, uint32_t call,
			  uint32_t width, uint32_t rows)
{
	uint32_t graph = 0;
	if (!test_linear_expect_status(
		    backend_cuda_value.graph_capture(replay->card.card, launch,
						     1, &graph),
		    STATUS_OK, "capture rotation timing graph"))
		return false;
	const struct test_card_timing_form forms[TEST_CARD_TIMING_FORMS] = {
		{ &graph, 1 }, { &graph, 1 }
	};
	double samples[TEST_CARD_TIMING_FORMS * TIME_SAMPLES],
		deltas[TIME_SAMPLES];
	if (!test_card_timing_pair(replay->card.card, forms, TIME_WARM,
				   TIME_REPLAYS, TIME_SAMPLES, samples, deltas))
		return false;
	for (uint32_t batch = 0; batch < TEST_CARD_TIMING_FORMS; batch++) {
		for (uint32_t sample = 0; sample < TIME_SAMPLES; sample++)
			(void)printf(
				"ROPE_TIME,call=%u,width=%u,rows=%u,batch=%u,sample=%u,microseconds=%.9g\n",
				call, width, rows, batch, sample,
				samples[batch * TIME_SAMPLES + sample]);
	}
	return test_linear_card_fetch(&replay->card);
}

static bool run_case(struct replay *replay, uint32_t call, uint32_t rows,
		     uint32_t width, bool in_place, bool timed)
{
	struct test_linear_card *const card = &replay->card;
	test_linear_card_clear(card);
	const uint32_t heads = call == 1 ? 1 : MODEL_HEADS_VALUE;
	const uint64_t row_bytes = (uint64_t)heads * width * sizeof(uint16_t),
		       rotation_bytes = (uint64_t)rows * OFFICIAL_ROPE_WIDTH *
					sizeof(float);
	struct official_rope_description description = {
		.input = test_linear_card_work(card, row_bytes * rows),
		.rotation = test_linear_card_work(card, rotation_bytes),
		.width = width,
		.heads = heads,
		.rows = rows,
		.inverse = call == SITES - 1u ? 1u : 0u
	};
	description.output =
		in_place ? description.input :
			   test_linear_card_work(card, row_bytes * rows);
	float rotation[LINEAR_MAXIMUM_ROWS * OFFICIAL_ROPE_WIDTH];
	char name[NAME_BYTES];
	(void)snprintf(name, sizeof(name), "layer_0.rope_%u.frequencies", call);
	bool passed = test_linear_card_has_room(card) &&
		      read_float(replay, name, 0, rows * OFFICIAL_ROPE_WIDTH,
				 rotation) &&
		      test_linear_card_put(card, description.rotation, 0,
					   rotation, rotation_bytes);
	for (uint32_t row = 0; passed && row < rows; row++) {
		uint16_t input[MODEL_HEADS_VALUE * MODEL_HEAD_DIMS];
		passed = read_vectors(replay, call, row, heads, width, "input",
				      input) &&
			 test_linear_card_put(card, description.input,
					      row * row_bytes, input,
					      row_bytes);
	}
	struct kernel_launch launch;
	passed = passed &&
		 test_linear_expect_status(
			 official_rope_prepare(&description,
					       card->arena_byte_count,
					       replay->kernel, &launch),
			 STATUS_OK, "prepare original rotation") &&
		 test_linear_card_load(card) &&
		 test_linear_card_run(card, &launch, 1) &&
		 test_linear_card_fetch(card) &&
		 (in_place ||
		  test_linear_card_expect_untouched(card, description.input,
						    "input preserved")) &&
		 test_linear_card_expect_untouched(card, description.rotation,
						   "coefficients preserved") &&
		 test_linear_card_expect_weights_untouched(card);
	if (passed && timed && !in_place)
		passed = time_rotation(replay, &launch, call, width, rows);
	for (uint32_t row = 0; passed && row < rows; row++) {
		uint16_t expected[MODEL_HEADS_VALUE * MODEL_HEAD_DIMS],
			actual[MODEL_HEADS_VALUE * MODEL_HEAD_DIMS];
		passed =
			read_vectors(replay, call, row, heads, width, "output",
				     expected) &&
			test_linear_card_get(card, description.output,
					     row * row_bytes, actual,
					     row_bytes) &&
			test_linear_expect(
				memcmp(expected, actual, (size_t)row_bytes) ==
					0,
				"all original rotation and copied prefix bits equal");
		const uint32_t record =
			replay->records[call]
				       [width == OFFICIAL_ROPE_WIDTH ? 0u : 1u]
				       [rows][in_place ? 1u : 0u][row];
		for (uint32_t head = 0; passed && head < heads; head++)
			passed = test_linear_expect_status(
				test_capture_writer_put(
					replay->writer, record, head,
					actual + (size_t)head * width),
				STATUS_OK, "write rotation result");
	}
	return passed;
}

int main(int argument_count, char **arguments)
{
	(void)alarm(TIME_LIMIT_SECONDS);
	if (argument_count == 1) {
		(void)puts(
			"official_rope_replay_card_test: no fixture; no original comparison executed");
		return EXIT_SUCCESS;
	}
	if (argument_count != ARGUMENTS ||
	    (strcmp(arguments[MODE_ARGUMENT], "check") != 0 &&
	     strcmp(arguments[MODE_ARGUMENT], "time") != 0))
		return EXIT_FAILURE;
	struct replay replay = { 0 };
	bool passed = test_linear_expect_status(
			      test_capture_open(arguments[REFERENCE_ARGUMENT],
						&replay.reference),
			      STATUS_OK, "open source capture") &&
		      test_linear_card_open(&replay.card, 0, WORK_CAPACITY,
					    WEIGHTS_CAPACITY) &&
		      test_linear_expect_status(
			      official_rope_kernel_prepare(&backend_cuda_value,
							   replay.card.card,
							   &replay.kernel),
			      STATUS_OK, "prepare rotation kernel") &&
		      plan_output(&replay, arguments[OUTPUT_ARGUMENT]);
	for (uint32_t call = 0; passed && call < SITES; call++) {
		for (uint32_t index = 0; passed && index < WIDTHS; index++) {
			for (uint32_t count = 0;
			     passed &&
			     count < sizeof(row_counts) / sizeof(row_counts[0]);
			     count++) {
				for (uint32_t alias = 0;
				     passed && alias < ALIAS_MODES; alias++)
					passed = run_case(
						&replay, call,
						row_counts[count],
						widths[index], alias != 0,
						strcmp(arguments[MODE_ARGUMENT],
						       "time") == 0);
			}
		}
	}
	const result_status closed =
		test_capture_writer_close(replay.writer, passed, "{}", "{}");
	passed = passed && test_linear_expect_status(closed, STATUS_OK,
						     "close rotation capture");
	test_capture_close(replay.reference);
	test_linear_card_close(&replay.card);
	return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
