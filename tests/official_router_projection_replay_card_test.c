/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Saved cached-M1 router operands and raw F32 logits. A generic F32-weight
 * dense product is a paired component control; no serving selector is timed.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "anemone/dense_linear.h"
#include "anemone/official_router_projection.h"
#include "capture_reader.h"
#include "capture_writer.h"
#include "card_timing.h"
#include "linear_card_fixture.h"

#define WORK_BYTES (1ull << 20u)
#define WEIGHT_BYTES (16ull << 20u)
#define CASES 4u
#define NAME_BYTES TEST_CAPTURE_WRITER_NAME_CAPACITY
#define MATRIX_DIMENSIONS 2u
#define TIME_SAMPLES 9u
#define TIME_REPLAYS 128u
#define TIME_WARM 8u
#define BFLOAT_SHIFT 16u
#define SOURCE_ROWS LINEAR_MAXIMUM_ROWS

enum argument {
	PROGRAM_ARGUMENT,
	MODE_ARGUMENT,
	REFERENCE_ARGUMENT,
	OUTPUT_ARGUMENT,
	ARGUMENTS
};

struct replay {
	struct test_linear_card card;
	struct test_capture *reference;
	struct test_capture_writer *writer;
	struct dense_linear_kernels dense;
	struct linear_range weights, old_weights, input, output, legacy;
	uint32_t kernel, records[CASES], old_records[CASES];
	bool timed;
};

static uint16_t inputs[SOURCE_ROWS * MODEL_HIDDEN];
static float expected[SOURCE_ROWS * MODEL_EXPERTS];
static float received[SOURCE_ROWS * MODEL_EXPERTS];

static bool read_matrix(struct test_capture *capture, const char *name,
			enum test_capture_type type, uint64_t rows,
			uint64_t width, void *output, uint64_t bytes)
{
	const uint64_t dimensions[] = { rows, width };
	return test_linear_expect_status(
		test_capture_read(capture, name, type, dimensions,
				  MATRIX_DIMENSIONS, 0, rows * width, output,
				  bytes),
		STATUS_OK, name);
}

static bool put_output(struct replay *self, uint32_t record,
		       struct linear_range range, uint32_t rows, bool exact)
{
	const uint64_t row_bytes = (uint64_t)MODEL_EXPERTS * sizeof(float);
	bool passed = test_linear_card_get(&self->card, range, 0, received,
					   row_bytes * rows);
	if (passed && exact)
		passed = test_linear_expect(
			memcmp(received, expected,
			       (size_t)(row_bytes * rows)) == 0,
			"every cached-M1 raw router logit matches");
	for (uint32_t row = 0; passed && row < rows; row++)
		passed = test_linear_expect_status(
			test_capture_writer_put(
				self->writer, record, row,
				received + (size_t)row * MODEL_EXPERTS),
			STATUS_OK, "write raw router output");
	return passed;
}

static bool timings(struct replay *self, const struct kernel_launch *old,
		    const struct kernel_launch *candidate, uint32_t rows)
{
	uint32_t graph[TEST_CARD_TIMING_FORMS] = { 0, 0 };
	bool passed = test_linear_expect_status(
			      backend_cuda_value.graph_capture(
				      self->card.card, old, 1, &graph[0]),
			      STATUS_OK, "capture old router product") &&
		      test_linear_expect_status(
			      backend_cuda_value.graph_capture(
				      self->card.card, candidate, 1, &graph[1]),
			      STATUS_OK, "capture M1 router product");
	const struct test_card_timing_form forms[TEST_CARD_TIMING_FORMS] = {
		{ &graph[0], 1 }, { &graph[1], 1 }
	};
	double samples[TEST_CARD_TIMING_FORMS * TIME_SAMPLES],
		deltas[TIME_SAMPLES];
	passed = passed && test_card_timing_pair(self->card.card, forms,
						 TIME_WARM, TIME_REPLAYS,
						 TIME_SAMPLES, samples, deltas);
	for (uint32_t form = 0; passed && form < TEST_CARD_TIMING_FORMS; form++)
		for (uint32_t sample = 0; sample < TIME_SAMPLES; sample++)
			(void)printf(
				"ROUTER_TIME,rows=%u,legacy=%u,sample=%u,microseconds=%.9g\n",
				rows, form == 0 ? 1u : 0u, sample,
				samples[form * TIME_SAMPLES + sample]);
	return passed;
}

int main(int argument_count, char **arguments)
{
	if (argument_count == 1) {
		(void)puts(
			"official_router_projection_replay_card_test: no fixture; no comparison executed");
		return EXIT_SUCCESS;
	}
	if (argument_count != ARGUMENTS ||
	    (strcmp(arguments[MODE_ARGUMENT], "check") != 0 &&
	     strcmp(arguments[MODE_ARGUMENT], "time") != 0))
		return EXIT_FAILURE;
	struct replay self = { .timed = strcmp(arguments[MODE_ARGUMENT],
					       "time") == 0 };
	bool passed =
		test_linear_expect_status(
			test_capture_open(arguments[REFERENCE_ARGUMENT],
					  &self.reference),
			STATUS_OK, "open original router fixture") &&
		test_linear_card_open(&self.card, 0, WORK_BYTES,
				      WEIGHT_BYTES) &&
		test_linear_expect_status(
			official_router_projection_kernel_prepare(
				&backend_cuda_value, self.card.card,
				&self.kernel),
			STATUS_OK, "prepare M1 router kernel") &&
		test_linear_expect_status(dense_linear_kernels_prepare(
						  &backend_cuda_value,
						  self.card.card, &self.dense),
					  STATUS_OK,
					  "prepare old dense control") &&
		test_linear_expect_status(
			test_capture_writer_create(
				TEST_CARD_TIMING_FORMS * CASES, &self.writer),
			STATUS_OK, "create router capture");
	static const uint32_t counts[] = { 1, 3, 4, 8 };
	for (uint32_t i = 0; passed && i < CASES; i++) {
		const uint64_t dimensions[] = { counts[i], MODEL_EXPERTS };
		for (uint32_t old = 0; passed && old < TEST_CARD_TIMING_FORMS;
		     old++) {
			char name[NAME_BYTES];
			(void)snprintf(name, sizeof(name), "router.rows_%u.%s",
				       counts[i],
				       old == 0 ? "candidate" : "legacy");
			passed = test_linear_expect_status(
				test_capture_writer_add(
					self.writer, name, TEST_CAPTURE_FLOAT32,
					dimensions, MATRIX_DIMENSIONS,
					counts[i], 0,
					TEST_CAPTURE_WRITER_NO_POSITION,
					old == 0 ?
						"Actual canonical M1 raw router logits" :
						"Generic dense F32-weight component control",
					old == 0 ? &self.records[i] :
						   &self.old_records[i]),
				STATUS_OK, "plan router record");
		}
	}
	passed = passed &&
		 test_linear_expect_status(
			 test_capture_writer_open(self.writer,
						  arguments[OUTPUT_ARGUMENT]),
			 STATUS_OK, "open router capture");
	const uint64_t weight_values = (uint64_t)MODEL_EXPERTS * MODEL_HIDDEN;
	uint16_t *const weight =
		malloc((size_t)(weight_values * sizeof(uint16_t)));
	float *const old_weight =
		malloc((size_t)(weight_values * sizeof(float)));
	passed = passed &&
		 test_linear_expect(weight != NULL && old_weight != NULL,
				    "allocate source router weights");
	if (passed) {
		test_linear_card_clear(&self.card);
		self.weights = test_linear_card_weights(
			&self.card, weight_values * sizeof(uint16_t));
		self.old_weights = test_linear_card_weights(
			&self.card, weight_values * sizeof(float));
		self.input = test_linear_card_work(&self.card, sizeof(inputs));
		self.output =
			test_linear_card_work(&self.card, sizeof(expected));
		self.legacy =
			test_linear_card_work(&self.card, sizeof(expected));
		passed = test_linear_card_has_room(&self.card) &&
			 read_matrix(self.reference, "router.weight",
				     TEST_CAPTURE_BFLOAT16, MODEL_EXPERTS,
				     MODEL_HIDDEN, weight,
				     weight_values * sizeof(uint16_t)) &&
			 read_matrix(self.reference, "router.input",
				     TEST_CAPTURE_BFLOAT16, SOURCE_ROWS,
				     MODEL_HIDDEN, inputs, sizeof(inputs)) &&
			 read_matrix(self.reference, "router.output",
				     TEST_CAPTURE_FLOAT32, SOURCE_ROWS,
				     MODEL_EXPERTS, expected, sizeof(expected));
		for (uint64_t i = 0; passed && i < weight_values; i++) {
			const uint32_t bits = (uint32_t)weight[i]
					      << BFLOAT_SHIFT;
			memcpy(old_weight + i, &bits, sizeof(bits));
		}
		passed = passed &&
			 test_linear_card_put(
				 &self.card, self.weights, 0, weight,
				 weight_values * sizeof(uint16_t)) &&
			 test_linear_card_put(&self.card, self.old_weights, 0,
					      old_weight,
					      weight_values * sizeof(float)) &&
			 test_linear_card_put(&self.card, self.input, 0, inputs,
					      sizeof(inputs)) &&
			 test_linear_card_load(&self.card);
	}
	for (uint32_t i = 0; passed && i < CASES; i++) {
		const uint32_t rows = counts[i];
		const struct official_router_projection_description
			description = { .weights = self.weights,
					.rows = self.input,
					.output = self.output,
					.row_count = rows };
		const struct dense_linear_description old = {
			.weights = self.old_weights,
			.rows = self.input,
			.output = self.legacy,
			.output_size = MODEL_EXPERTS,
			.input_size = MODEL_HIDDEN,
			.row_count = rows,
			.weight_type = DENSE_FLOAT32,
			.output_type = DENSE_FLOAT32,
			.output_scale = 1.0f
		};
		struct kernel_launch candidate, legacy;
		passed = test_linear_expect_status(
				 official_router_projection_prepare(
					 &description,
					 self.card.arena_byte_count,
					 self.kernel, &candidate),
				 STATUS_OK, "prepare saved-input router") &&
			 test_linear_expect_status(
				 dense_linear_prepare(
					 &old, self.card.arena_byte_count,
					 &self.dense, &legacy),
				 STATUS_OK,
				 "prepare generic F32 dense control") &&
			 test_linear_card_run(&self.card, &legacy, 1) &&
			 test_linear_card_fetch(&self.card) &&
			 put_output(&self, self.old_records[i], self.legacy,
				    rows, false) &&
			 test_linear_card_run(&self.card, &candidate, 1);
		if (passed && self.timed && (i == 0 || i + 1u == CASES))
			passed = timings(&self, &legacy, &candidate, rows);
		passed = passed && test_linear_card_fetch(&self.card) &&
			 put_output(&self, self.records[i], self.output, rows,
				    true) &&
			 test_linear_card_expect_untouched(
				 &self.card, self.input,
				 "router input preserved") &&
			 test_linear_card_expect_weights_untouched(&self.card);
	}
	const bool complete = passed && test_linear_failures() == 0;
	const result_status closed = test_capture_writer_close(
		self.writer, complete,
		"{\"scope\":\"cached-M1 raw router; original BF16 weights and rows; fixed per-row target\"}",
		"{}");
	test_capture_close(self.reference);
	test_linear_card_close(&self.card);
	free(weight);
	free(old_weight);
	return complete && closed == STATUS_OK ? EXIT_SUCCESS : EXIT_FAILURE;
}
