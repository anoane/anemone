/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Original captured window/output quantize-dequantize. Strict byte equality is
 * checked inside this replay and independently in the retained result captures.
 */
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "anemone/official_roundtrip.h"
#include "capture_reader.h"
#include "capture_writer.h"
#include "card_timing.h"
#include "linear_card_fixture.h"

#define WORK_CAPACITY (2ull << 20u)
#define WEIGHTS_CAPACITY (1ull << 16u)
#define SITES 2u
#define TIME_LIMIT_SECONDS 120u
#define NAME_BYTES TEST_CAPTURE_WRITER_NAME_CAPACITY
#define TIME_SAMPLES 9u
#define TIME_REPLAYS 128u
#define TIME_WARM 8u
#define OUTPUT_DIMENSIONS 2u
#define OUTPUT_GROUP_WIDTH 4096u
#define OUTPUT_GROUPS (OFFICIAL_ROUNDTRIP_MOST_WIDTH / OUTPUT_GROUP_WIDTH)

enum argument {
	PROGRAM_ARGUMENT,
	ROWS_ARGUMENT,
	MODE_ARGUMENT,
	REFERENCE_ARGUMENT,
	OUTPUT_ARGUMENT,
	ARGUMENTS
};

static bool read_row(const struct test_capture *reference, const char *name,
		     uint32_t width, uint32_t row, uint16_t *values)
{
	const struct test_capture_tensor *const tensor =
		test_capture_find(reference, name);
	return tensor != NULL &&
	       tensor->element_count >= (uint64_t)(row + 1u) * width &&
	       test_linear_expect_status(
		       test_capture_read(reference, name, TEST_CAPTURE_BFLOAT16,
					 tensor->dimensions,
					 tensor->dimension_count,
					 (uint64_t)row * width, width, values,
					 (uint64_t)width * sizeof(uint16_t)),
		       STATUS_OK, name);
}

/* Both timing batches run the same graph. Report absolute component latency;
 * there is no omitted-operation baseline or speedup inferred from their delta. */
static bool time_component(struct test_linear_card *card,
			   const struct kernel_launch *launch, uint32_t site,
			   uint32_t rows, bool in_place)
{
	uint32_t graph = 0;
	bool passed = test_linear_expect_status(
		backend_cuda_value.graph_capture(card->card, launch, 1, &graph),
		STATUS_OK, "capture round trip timing graph");
	unsigned char *const before =
		passed ? malloc((size_t)card->work_byte_count) : NULL;
	passed = passed &&
		 test_linear_expect(before != NULL,
				    "allocate full readback snapshot");
	if (passed) {
		memcpy(before, card->received, (size_t)card->work_byte_count);
		const struct test_card_timing_form
			forms[TEST_CARD_TIMING_FORMS] = { { &graph, 1 },
							  { &graph, 1 } };
		double samples[TEST_CARD_TIMING_FORMS * TIME_SAMPLES],
			deltas[TIME_SAMPLES];
		passed =
			test_card_timing_pair(card->card, forms, TIME_WARM,
					      TIME_REPLAYS, TIME_SAMPLES,
					      samples, deltas) &&
			test_linear_card_fetch(card) &&
			test_linear_expect(
				memcmp(before, card->received,
				       (size_t)card->work_byte_count) == 0,
				"timed round trip preserves entire already-quantized work image and canaries");
		for (uint32_t batch = 0;
		     passed && batch < TEST_CARD_TIMING_FORMS; batch++) {
			for (uint32_t sample = 0; sample < TIME_SAMPLES;
			     sample++)
				(void)printf(
					"RT_TIME,site=%u,rows=%u,in_place=%u,batch=%u,sample=%u,microseconds=%.9g\n",
					site, rows, in_place ? 1u : 0u, batch,
					sample,
					samples[batch * TIME_SAMPLES + sample]);
		}
	}
	free(before);
	return passed;
}

int main(int argument_count, char **arguments)
{
	(void)alarm(TIME_LIMIT_SECONDS);
	if (argument_count == 1) {
		(void)puts(
			"official_roundtrip_replay_card_test: no fixture; no original comparison executed");
		return EXIT_SUCCESS;
	}
	if (argument_count != ARGUMENTS ||
	    (strcmp(arguments[MODE_ARGUMENT], "in_place") != 0 &&
	     strcmp(arguments[MODE_ARGUMENT], "separate") != 0))
		return EXIT_FAILURE;
	static const char *const row_names[] = { "1", "3", "4", "8" };
	static const uint32_t row_counts[] = { 1, 3, 4, 8 };
	uint32_t rows = 0;
	for (uint32_t index = 0;
	     index < sizeof(row_counts) / sizeof(row_counts[0]); index++) {
		if (strcmp(arguments[ROWS_ARGUMENT], row_names[index]) == 0)
			rows = row_counts[index];
	}
	if (rows == 0)
		return EXIT_FAILURE;
	struct test_capture *reference = NULL;
	struct test_capture_writer *writer = NULL;
	struct test_linear_card card = { 0 };
	uint32_t kernel = BACKEND_NO_KERNEL;
	bool passed = test_linear_expect_status(
			      test_capture_open(arguments[REFERENCE_ARGUMENT],
						&reference),
			      STATUS_OK, "open original capture") &&
		      test_linear_card_open(&card, 0, WORK_CAPACITY,
					    WEIGHTS_CAPACITY) &&
		      test_linear_expect_status(
			      official_roundtrip_kernel_prepare(
				      &backend_cuda_value, card.card, &kernel),
			      STATUS_OK, "prepare round trip kernel") &&
		      test_linear_expect_status(
			      test_capture_writer_create(SITES * rows, &writer),
			      STATUS_OK, "create output capture");
	static const uint32_t calls[] = { 3, 4 };
	static const uint32_t widths[] = { MODEL_HEAD_DIMS,
					   OFFICIAL_ROUNDTRIP_MOST_WIDTH };
	uint32_t records[SITES][LINEAR_MAXIMUM_ROWS];
	for (uint32_t site = 0; passed && site < SITES; site++) {
		const uint64_t dimensions[] = { site == 0 ? widths[site] :
							    OUTPUT_GROUPS,
						OUTPUT_GROUP_WIDTH };
		for (uint32_t row = 0; passed && row < rows; row++) {
			char name[NAME_BYTES];
			const int count = snprintf(
				name, sizeof(name),
				"roundtrip.quant_%u.position_%u.output",
				calls[site], row);
			passed =
				count > 0 && (size_t)count < sizeof(name) &&
				test_linear_expect_status(
					test_capture_writer_add(
						writer, name,
						TEST_CAPTURE_BFLOAT16,
						dimensions,
						site == 0 ? 1u :
							    OUTPUT_DIMENSIONS,
						site == 0 ? 1u : OUTPUT_GROUPS,
						0, row,
						"Actual original-input round trip",
						&records[site][row]),
					STATUS_OK, "add output record");
		}
	}
	passed = passed && test_linear_expect_status(
				   test_capture_writer_open(
					   writer, arguments[OUTPUT_ARGUMENT]),
				   STATUS_OK, "open output capture");
	const bool in_place = strcmp(arguments[MODE_ARGUMENT], "in_place") == 0;
	for (uint32_t site = 0; passed && site < SITES; site++) {
		const uint32_t width = widths[site];
		const uint64_t row_bytes = (uint64_t)width * sizeof(uint16_t);
		test_linear_card_clear(&card);
		struct official_roundtrip_description description = {
			.input = test_linear_card_work(&card, row_bytes * rows),
			.width = width,
			.row_count = rows
		};
		description.output =
			in_place ?
				description.input :
				test_linear_card_work(&card, row_bytes * rows);
		char input_name[NAME_BYTES], output_name[NAME_BYTES];
		(void)snprintf(input_name, sizeof(input_name),
			       "layer_0.quant_%u.input", calls[site]);
		(void)snprintf(output_name, sizeof(output_name),
			       "layer_0.quant_%u.output", calls[site]);
		for (uint32_t row = 0; passed && row < rows; row++) {
			uint16_t values[OFFICIAL_ROUNDTRIP_MOST_WIDTH];
			passed = read_row(reference, input_name, width, row,
					  values) &&
				 test_linear_card_put(&card, description.input,
						      row * row_bytes, values,
						      row_bytes);
		}
		struct kernel_launch launch;
		passed = passed && test_linear_card_has_room(&card) &&
			 test_linear_expect_status(
				 official_roundtrip_prepare(
					 &description, card.arena_byte_count,
					 kernel, &launch),
				 STATUS_OK, "prepare original round trip") &&
			 test_linear_card_load(&card) &&
			 test_linear_card_run(&card, &launch, 1) &&
			 test_linear_card_fetch(&card) &&
			 (in_place || test_linear_card_expect_untouched(
					      &card, description.input,
					      "read-only input preserved")) &&
			 test_linear_card_expect_weights_untouched(&card);
		for (uint32_t row = 0; passed && row < rows; row++) {
			uint16_t expected[OFFICIAL_ROUNDTRIP_MOST_WIDTH],
				actual[OFFICIAL_ROUNDTRIP_MOST_WIDTH];
			passed =
				read_row(reference, output_name, width, row,
					 expected) &&
				test_linear_card_get(&card, description.output,
						     row * row_bytes, actual,
						     row_bytes) &&
				test_linear_expect(
					memcmp(actual, expected,
					       (size_t)row_bytes) == 0,
					"all original round trip output bits equal");
			const uint32_t slabs = site == 0 ? 1u : OUTPUT_GROUPS;
			for (uint32_t slab = 0; passed && slab < slabs; slab++)
				passed = test_linear_expect_status(
					test_capture_writer_put(
						writer, records[site][row],
						slab,
						actual + (uint64_t)slab *
								 width / slabs),
					STATUS_OK, "write actual output");
		}
		if (passed && getenv("TRACK_R_SKIP_TIMING") == NULL)
			passed = time_component(&card, &launch, calls[site],
						rows, in_place);
	}
	const result_status closed = test_capture_writer_close(
		writer, passed,
		in_place ? "{\"roundtrip\":\"in_place\"}" :
			   "{\"roundtrip\":\"separate\"}",
		"{}");
	passed = passed && test_linear_expect_status(closed, STATUS_OK,
						     "close output capture");
	test_capture_close(reference);
	test_linear_card_close(&card);
	return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
