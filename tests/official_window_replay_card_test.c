/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Replay the actual original window composition and compare the retained
 * three-pass form with the optional one-pass form. All source fields are
 * strict; complete work canaries and read-only inputs are checked after timing.
 */
#include <inttypes.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "anemone/official_rope.h"
#include "anemone/official_roundtrip.h"
#include "anemone/official_window.h"
#include "capture_reader.h"
#include "capture_writer.h"
#include "card_timing.h"
#include "linear_card_fixture.h"

#define WORK_CAPACITY (1ull << 20u)
#define WEIGHTS_CAPACITY (1ull << 16u)
#define MOST_ROWS LINEAR_MAXIMUM_ROWS
#define WIDTH MODEL_HEAD_DIMS
#define NAME_BYTES TEST_CAPTURE_WRITER_NAME_CAPACITY
#define TIME_LIMIT_SECONDS 300u
#define TIME_SAMPLES 9u
#define TIME_REPLAYS 128u
#define TIME_WARM 8u
#define STAGED_LAUNCHES 3u
#define ROTATION_LAUNCH 1u
#define ROUNDTRIP_LAUNCH 2u
#define FIELDS 8u
#define FLOAT_WORD_HALVES 2u

enum argument {
	PROGRAM_ARGUMENT,
	MODE_ARGUMENT,
	SHAPE_ARGUMENT,
	REFERENCE_ARGUMENT,
	OUTPUT_ARGUMENT,
	ARGUMENTS
};

static bool read_part(struct test_capture *capture, const char *name,
		      enum test_capture_type type, uint64_t elements,
		      void *destination)
{
	const struct test_capture_tensor *const value =
		test_capture_find(capture, name);
	if (value == NULL || value->type != (uint32_t)type ||
	    value->element_count != elements)
		return test_linear_expect(false, name);
	return test_linear_expect_status(
		test_capture_read(capture, name, type, value->dimensions,
				  value->dimension_count, 0, elements,
				  destination,
				  elements * test_capture_element_bytes(type)),
		STATUS_OK, name);
}

static bool name_part(char *name, uint32_t row, const char *part)
{
	const int count =
		snprintf(name, NAME_BYTES, "window.position_%u.%s", row, part);
	return test_linear_expect(count > 0 && (uint32_t)count < NAME_BYTES,
				  "window name fits");
}

static bool time_forms(struct test_linear_card *card,
		       const struct kernel_launch *staged,
		       const struct kernel_launch *fused, uint32_t rows,
		       uint32_t shape)
{
	uint32_t graphs[TEST_CARD_TIMING_FORMS];
	if (!test_linear_expect_status(
		    backend_cuda_value.graph_capture(
			    card->card, staged, STAGED_LAUNCHES, &graphs[0]),
		    STATUS_OK, "capture three-pass window graph") ||
	    !test_linear_expect_status(
		    backend_cuda_value.graph_capture(card->card, fused, 1,
						     &graphs[1]),
		    STATUS_OK, "capture fused window graph"))
		return false;
	const struct test_card_timing_form forms[TEST_CARD_TIMING_FORMS] = {
		{ &graphs[0], 1 }, { &graphs[1], 1 }
	};
	double samples[TEST_CARD_TIMING_FORMS * TIME_SAMPLES],
		deltas[TIME_SAMPLES];
	if (!test_card_timing_pair(card->card, forms, TIME_WARM, TIME_REPLAYS,
				   TIME_SAMPLES, samples, deltas))
		return false;
	for (uint32_t form = 0; form < TEST_CARD_TIMING_FORMS; form++) {
		for (uint32_t sample = 0; sample < TIME_SAMPLES; sample++)
			(void)printf(
				"WINDOW_TIME,rows=%u,shape=%u,fused=%u,sample=%u,microseconds=%.9g\n",
				rows, shape, form, sample,
				samples[form * TIME_SAMPLES + sample]);
	}
	return true;
}

int main(int count, char **arguments)
{
	(void)alarm(TIME_LIMIT_SECONDS);
	if (count == 1) {
		(void)puts(
			"official_window_replay_card_test: no fixture; no comparison executed");
		return EXIT_SUCCESS;
	}
	if (count != ARGUMENTS ||
	    (strcmp(arguments[MODE_ARGUMENT], "check") != 0 &&
	     strcmp(arguments[MODE_ARGUMENT], "time") != 0))
		return EXIT_FAILURE;
	static const char *const row_names[] = { "1", "3", "4", "8" };
	static const uint32_t row_counts[] = { 1, 3, 4, 8 };
	uint32_t shape = 0;
	for (uint32_t index = 0;
	     index < sizeof(row_counts) / sizeof(row_counts[0]); index++) {
		if (strcmp(arguments[SHAPE_ARGUMENT], row_names[index]) == 0)
			shape = row_counts[index];
	}
	if (shape == 0)
		return EXIT_FAILURE;
	struct test_capture *reference = NULL;
	struct test_capture_writer *writer = NULL;
	struct test_linear_card card = { 0 };
	bool passed = test_linear_expect_status(
		test_capture_open(arguments[REFERENCE_ARGUMENT], &reference),
		STATUS_OK, "open original window capture");
	const struct test_capture_tensor *const tokens =
		test_capture_find(reference, "input.token_ids");
	const uint32_t rows =
		passed && tokens != NULL && tokens->element_count <= MOST_ROWS ?
			(uint32_t)tokens->element_count :
			0;
	passed = passed && rows != 0 &&
		 test_linear_card_open(&card, 0, WORK_CAPACITY,
				       WEIGHTS_CAPACITY);
	if (!passed) {
		test_capture_close(reference);
		test_linear_card_close(&card);
		return EXIT_FAILURE;
	}
	uint32_t norm_kernel = BACKEND_NO_KERNEL,
		 rope_kernel = BACKEND_NO_KERNEL,
		 roundtrip_kernel = BACKEND_NO_KERNEL,
		 fused_kernel = BACKEND_NO_KERNEL;
	passed = passed &&
		 test_linear_expect_status(
			 official_norm_kernel_prepare(&backend_cuda_value,
						      card.card, &norm_kernel),
			 STATUS_OK, "prepare independent norm kernel") &&
		 test_linear_expect_status(
			 official_rope_kernel_prepare(&backend_cuda_value,
						      card.card, &rope_kernel),
			 STATUS_OK, "prepare independent rotation kernel") &&
		 test_linear_expect_status(
			 official_roundtrip_kernel_prepare(&backend_cuda_value,
							   card.card,
							   &roundtrip_kernel),
			 STATUS_OK, "prepare independent roundtrip kernel") &&
		 test_linear_expect_status(
			 official_window_kernel_prepare(
				 &backend_cuda_value, card.card, &fused_kernel),
			 STATUS_OK, "prepare optional composition kernel");
	const uint64_t row_bytes = (uint64_t)WIDTH * sizeof(uint16_t);
	struct official_norm_description norm = {
		.input = test_linear_card_work(&card, rows * row_bytes),
		.weight = test_linear_card_weights(&card, row_bytes),
		.output = test_linear_card_work(&card, rows * row_bytes),
		.statistics = test_linear_card_work(
			&card, (uint64_t)rows * OFFICIAL_NORM_STATISTICS *
				       sizeof(float)),
		.width = WIDTH,
		.row_count = rows
	};
	struct official_rope_description rope = {
		.input = norm.output,
		.rotation = test_linear_card_work(
			&card,
			(uint64_t)rows * OFFICIAL_ROPE_WIDTH * sizeof(float)),
		.output = test_linear_card_work(&card, rows * row_bytes),
		.width = WIDTH,
		.heads = 1,
		.rows = rows
	};
	struct official_roundtrip_description roundtrip = {
		.input = rope.output,
		.output = test_linear_card_work(&card, rows * row_bytes),
		.width = WIDTH,
		.row_count = rows
	};
	struct official_window_description fused = { norm, rope.rotation };
	fused.norm.output = test_linear_card_work(&card, rows * row_bytes);
	fused.norm.statistics = test_linear_card_work(
		&card,
		(uint64_t)rows * OFFICIAL_NORM_STATISTICS * sizeof(float));
	uint16_t weight[WIDTH];
	passed = passed && test_linear_card_has_room(&card) &&
		 read_part(reference, "window.weight", TEST_CAPTURE_BFLOAT16,
			   WIDTH, weight) &&
		 test_linear_card_put(&card, norm.weight, 0, weight, row_bytes);
	for (uint32_t row = 0; passed && row < rows; row++) {
		char name[NAME_BYTES];
		uint16_t values[WIDTH];
		float frequency[OFFICIAL_ROPE_WIDTH];
		passed = name_part(name, row, "input") &&
			 read_part(reference, name, TEST_CAPTURE_BFLOAT16,
				   WIDTH, values) &&
			 test_linear_card_put(&card, norm.input,
					      row * row_bytes, values,
					      row_bytes) &&
			 name_part(name, row, "rotation") &&
			 read_part(reference, name, TEST_CAPTURE_FLOAT32,
				   OFFICIAL_ROPE_WIDTH, frequency) &&
			 test_linear_card_put(&card, rope.rotation,
					      row * sizeof(frequency),
					      frequency, sizeof(frequency));
	}
	struct kernel_launch staged[STAGED_LAUNCHES], combined;
	passed = passed &&
		 test_linear_expect_status(
			 official_norm_shape_prepare(&norm, shape,
						     card.arena_byte_count,
						     norm_kernel, &staged[0]),
			 STATUS_OK, "prepare norm") &&
		 test_linear_expect_status(
			 official_rope_prepare(&rope, card.arena_byte_count,
					       rope_kernel,
					       &staged[ROTATION_LAUNCH]),
			 STATUS_OK, "prepare rotation") &&
		 test_linear_expect_status(
			 official_roundtrip_prepare(
				 &roundtrip, card.arena_byte_count,
				 roundtrip_kernel, &staged[ROUNDTRIP_LAUNCH]),
			 STATUS_OK, "prepare roundtrip") &&
		 test_linear_expect_status(
			 official_window_prepare(&fused, shape,
						 card.arena_byte_count,
						 fused_kernel, &combined),
			 STATUS_OK, "prepare fused composition") &&
		 test_linear_card_load(&card) &&
		 test_linear_card_run(&card, staged, STAGED_LAUNCHES) &&
		 test_linear_card_run(&card, &combined, 1) &&
		 (strcmp(arguments[MODE_ARGUMENT], "time") != 0 ||
		  time_forms(&card, staged, &combined, rows, shape)) &&
		 test_linear_card_run(&card, staged, STAGED_LAUNCHES) &&
		 test_linear_card_run(&card, &combined, 1) &&
		 test_linear_card_fetch(&card) &&
		 test_linear_card_expect_untouched(&card, norm.input,
						   "WKV input preserved") &&
		 test_linear_card_expect_untouched(&card, rope.rotation,
						   "coefficients preserved") &&
		 test_linear_card_expect_weights_untouched(&card) &&
		 test_capture_writer_create(rows * FIELDS, &writer) ==
			 STATUS_OK;
	static const char *const names[FIELDS] = {
		"normalized", "rotated",      "output",	    "mean",
		"inverse",    "fused_output", "fused_mean", "fused_inverse"
	};
	static const char *const source_names[FIELDS] = {
		"normalized", "rotated", "output", "mean",
		"inverse",    "output",	 "mean",   "inverse"
	};
	const struct linear_range regions[FIELDS] = {
		norm.output,	       rope.output,	     roundtrip.output,
		norm.statistics,       norm.statistics,	     fused.norm.output,
		fused.norm.statistics, fused.norm.statistics
	};
	static const uint32_t statistic_indices[FIELDS] = { 0, 0, 0, 0,
							    1, 0, 0, 1 };
	static const bool statistics[FIELDS] = { false, false, false, true,
						 true,	false, true,  true };
	uint32_t records[MOST_ROWS][FIELDS];
	for (uint32_t row = 0; passed && row < rows; row++) {
		for (uint32_t field = 0; passed && field < FIELDS; field++) {
			char name[NAME_BYTES];
			const uint64_t dimensions[] = { statistics[field] ?
								1u :
								WIDTH };
			passed = name_part(name, row, names[field]) &&
				 test_linear_expect_status(
					 test_capture_writer_add(
						 writer, name,
						 statistics[field] ?
							 TEST_CAPTURE_FLOAT32 :
							 TEST_CAPTURE_BFLOAT16,
						 dimensions, 1, 1, 0, row,
						 "Original window replay",
						 &records[row][field]),
					 STATUS_OK, "add window field");
		}
	}
	passed = passed && test_linear_expect_status(
				   test_capture_writer_open(
					   writer, arguments[OUTPUT_ARGUMENT]),
				   STATUS_OK, "open output capture");
	for (uint32_t row = 0; passed && row < rows; row++) {
		for (uint32_t field = 0; passed && field < FIELDS; field++) {
			char name[NAME_BYTES];
			union {
				uint16_t bfloat16[WIDTH];
				float float32[WIDTH / FLOAT_WORD_HALVES];
			} actual, expected;
			const uint64_t bytes =
				statistics[field] ? sizeof(float) : row_bytes;
			const uint64_t offset =
				statistics[field] ?
					((uint64_t)row *
						 OFFICIAL_NORM_STATISTICS +
					 statistic_indices[field]) *
						sizeof(float) :
					row * row_bytes;
			passed =
				name_part(name, row, source_names[field]) &&
				read_part(reference, name,
					  statistics[field] ?
						  TEST_CAPTURE_FLOAT32 :
						  TEST_CAPTURE_BFLOAT16,
					  statistics[field] ? 1u : WIDTH,
					  &expected) &&
				test_linear_card_get(&card, regions[field],
						     offset, &actual, bytes) &&
				test_linear_expect(
					memcmp(&actual, &expected,
					       (size_t)bytes) == 0,
					"all original window field bits equal") &&
				test_linear_expect_status(
					test_capture_writer_put(
						writer, records[row][field], 0,
						&actual),
					STATUS_OK, "write actual field");
		}
	}
	char conditions[NAME_BYTES];
	(void)snprintf(
		conditions, sizeof(conditions),
		"{\"actual_rows\":%u,\"reference_rows\":%u,\"scope\":\"window composition\"}",
		rows, shape);
	const result_status closed =
		test_capture_writer_close(writer, passed, conditions, "{}");
	passed = passed && test_linear_expect_status(closed, STATUS_OK,
						     "close window capture");
	test_capture_close(reference);
	test_linear_card_close(&card);
	return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
