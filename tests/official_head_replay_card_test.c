/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Actual nine M1 heads at fixed per-output arithmetic, row prefixes and four
 * vocabulary cuts. The generic dense component is a paired timing/deviation
 * control, not a claim about a selected serving head. No model norm runs here.
 */
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "anemone/dense_linear.h"
#include "anemone/official_head.h"
#include "capture_reader.h"
#include "capture_writer.h"
#include "card_timing.h"
#include "linear_card_fixture.h"

#define WORK_CAPACITY (16ull << 20u)
#define WEIGHT_BYTES \
	((uint64_t)MODEL_VOCABULARY_VALUE * MODEL_HIDDEN * sizeof(uint16_t))
#define WEIGHT_CAPACITY (3ull << 29u)
#define SMALL_WEIGHT_CAPACITY (1ull << 16u)
#define SMALL_WEIGHT_BYTES \
	((uint64_t)OFFICIAL_HEAD_OUTPUTS * MODEL_HIDDEN * sizeof(uint16_t))
#define CASES 10u
#define FULL_CASES 5u
#define FORMS 2u
#define RECORDS (CASES * FORMS + FULL_CASES)
#define NAME_BYTES 96u
#define TIME_LIMIT_SECONDS 300u
#define TIME_SAMPLES 9u
#define TIME_REPLAYS 8u
#define TIME_WARM 8u
#define QUARTERS 4u
#define MATRIX_DIMENSIONS 2u
#define CACHED_BATCH_CASE 3u

enum argument {
	PROGRAM_ARGUMENT,
	MODE_ARGUMENT,
	REFERENCE_ARGUMENT,
	WEIGHTS_ARGUMENT,
	OUTPUT_ARGUMENT,
	ARGUMENTS
};

struct replay {
	struct test_linear_card card;
	struct test_capture *reference;
	struct test_capture_writer *writer;
	struct dense_linear_kernels dense;
	struct linear_range weights, input, output, legacy;
	uint32_t kernels[FORMS];
	uint32_t records[CASES][FORMS], legacy_records[FULL_CASES];
	float *expected, *received;
	bool timed, small;
};

static uint32_t rows_of(uint32_t index)
{
	static const uint32_t prefix[] = { 1, 3, 4, 8, 1 };
	return index < FULL_CASES ? prefix[index] : LINEAR_MAXIMUM_ROWS;
}

static uint32_t outputs_of(uint32_t index)
{
	return index < FULL_CASES  ? MODEL_VOCABULARY_VALUE :
	       index + 1u == CASES ? OFFICIAL_HEAD_OUTPUTS :
				     MODEL_VOCABULARY_VALUE / QUARTERS;
}

static uint32_t first_of(uint32_t index)
{
	return index < FULL_CASES ?
		       0u :
	       index + 1u == CASES ?
		       MODEL_VOCABULARY_VALUE - OFFICIAL_HEAD_OUTPUTS :
		       (index - FULL_CASES) * outputs_of(index);
}

static bool selected(const struct replay *replay, uint32_t index)
{
	return !replay->small || index + 1u == CASES;
}

static bool read_values(const struct test_capture *capture, const char *name,
			enum test_capture_type type, uint32_t rows,
			uint32_t width, void *destination, uint64_t bytes)
{
	const struct test_capture_tensor *const tensor =
		test_capture_find(capture, name);
	return tensor != NULL && tensor->type == (uint32_t)type &&
	       tensor->dimension_count == MATRIX_DIMENSIONS &&
	       tensor->dimensions[0] == rows &&
	       tensor->dimensions[1] == width &&
	       test_linear_expect_status(
		       test_capture_read(capture, name, type,
					 tensor->dimensions, MATRIX_DIMENSIONS,
					 0, (uint64_t)rows * width, destination,
					 bytes),
		       STATUS_OK, "read unchanged original head values");
}

static bool plan_output(struct replay *replay, const char *folder)
{
	bool passed = test_linear_expect_status(
		test_capture_writer_create(RECORDS, &replay->writer), STATUS_OK,
		"create head capture");
	for (uint32_t index = 0; passed && index < CASES; index++) {
		if (!selected(replay, index))
			continue;
		const uint64_t dimensions[] = { rows_of(index),
						outputs_of(index) };
		for (uint32_t form = 0; passed && form < FORMS; form++) {
			char name[NAME_BYTES];
			(void)snprintf(name, sizeof(name), "head.case_%u.%s",
				       index, form == 0 ? "direct" : "reused");
			passed = test_linear_expect_status(
				test_capture_writer_add(
					replay->writer, name,
					TEST_CAPTURE_FLOAT32, dimensions,
					MATRIX_DIMENSIONS, rows_of(index),
					TEST_CAPTURE_NO_LAYER,
					TEST_CAPTURE_WRITER_NO_POSITION,
					"Actual canonical M1 component output; no norm or sampling",
					&replay->records[index][form]),
				STATUS_OK, "plan candidate head record");
		}
		if (passed && index < FULL_CASES) {
			char name[NAME_BYTES];
			(void)snprintf(name, sizeof(name),
				       "head.case_%u.legacy_dense", index);
			passed = test_linear_expect_status(
				test_capture_writer_add(
					replay->writer, name,
					TEST_CAPTURE_FLOAT32, dimensions,
					MATRIX_DIMENSIONS, rows_of(index),
					TEST_CAPTURE_NO_LAYER,
					TEST_CAPTURE_WRITER_NO_POSITION,
					"Generic legacy dense component on the same inputs; no serving selector claim",
					&replay->legacy_records[index]),
				STATUS_OK, "plan dense control");
		}
	}
	return passed &&
	       test_linear_expect_status(
		       test_capture_writer_open(replay->writer, folder),
		       STATUS_OK, "open head capture");
}

static bool timing(struct replay *replay, const struct kernel_launch *legacy,
		   const struct kernel_launch *candidate, uint32_t rows,
		   uint32_t form)
{
	uint32_t graphs[FORMS] = { 0, 0 };
	bool passed =
		test_linear_expect_status(
			backend_cuda_value.graph_capture(replay->card.card,
							 legacy, 1, &graphs[0]),
			STATUS_OK, "capture generic dense timing graph") &&
		test_linear_expect_status(
			backend_cuda_value.graph_capture(
				replay->card.card, candidate, 1, &graphs[1]),
			STATUS_OK, "capture candidate timing graph");
	const struct test_card_timing_form forms[TEST_CARD_TIMING_FORMS] = {
		{ &graphs[0], 1 }, { &graphs[1], 1 }
	};
	double samples[TEST_CARD_TIMING_FORMS * TIME_SAMPLES],
		deltas[TIME_SAMPLES];
	passed = passed && test_card_timing_pair(replay->card.card, forms,
						 TIME_WARM, TIME_REPLAYS,
						 TIME_SAMPLES, samples, deltas);
	if (passed) {
		for (uint32_t batch = 0; batch < TEST_CARD_TIMING_FORMS;
		     batch++)
			for (uint32_t sample = 0; sample < TIME_SAMPLES;
			     sample++)
				(void)printf(
					"HEAD_TIME,rows=%u,reused=%u,legacy=%u,sample=%u,microseconds=%.9g\n",
					rows, form, batch == 0 ? 1u : 0u,
					sample,
					samples[batch * TIME_SAMPLES + sample]);
	}
	return passed;
}

static bool put_output(struct replay *replay, uint32_t index, uint32_t record,
		       struct linear_range range, bool require_original)
{
	const uint32_t rows = rows_of(index), outputs = outputs_of(index),
		       first = first_of(index);
	const uint64_t bytes = (uint64_t)rows * outputs * sizeof(float);
	bool passed = test_linear_card_get(&replay->card, range, 0,
					   replay->received, bytes);
	for (uint32_t row = 0; passed && row < rows; row++) {
		const float *const actual =
			replay->received + (uint64_t)row * outputs;
		const float *const expected =
			replay->expected +
			(uint64_t)row * MODEL_VOCABULARY_VALUE + first;
		for (uint32_t n = 0; passed && n < outputs; n++)
			passed = test_linear_expect(isfinite(actual[n]),
						    "finite head output");
		if (passed && require_original)
			passed = test_linear_expect(
				memcmp(actual, expected,
				       (size_t)outputs * sizeof(float)) == 0,
				"all original M1 head bits match");
		if (passed)
			passed = test_linear_expect_status(
				test_capture_writer_put(replay->writer, record,
							row, actual),
				STATUS_OK, "store complete head row");
	}
	return passed;
}

static bool run_case(struct replay *replay, uint32_t index)
{
	const bool prompt = index + 1u == FULL_CASES;
	const uint32_t rows = rows_of(index), outputs = outputs_of(index),
		       first = first_of(index);
	const uint32_t source_rows = prompt ? 1u : LINEAR_MAXIMUM_ROWS;
	uint16_t values[LINEAR_MAXIMUM_ROWS * MODEL_HIDDEN];
	bool passed =
		read_values(replay->reference,
			    prompt ? "head.prompt.input" : "head.cached.input",
			    TEST_CAPTURE_BFLOAT16, source_rows, MODEL_HIDDEN,
			    values,
			    (uint64_t)source_rows * MODEL_HIDDEN *
				    sizeof(uint16_t)) &&
		read_values(replay->reference,
			    prompt ? "head.prompt.output" :
				     "head.cached.output",
			    TEST_CAPTURE_FLOAT32, source_rows,
			    MODEL_VOCABULARY_VALUE, replay->expected,
			    (uint64_t)source_rows * MODEL_VOCABULARY_VALUE *
				    sizeof(float)) &&
		test_linear_card_put(&replay->card, replay->input, 0, values,
				     (uint64_t)rows * MODEL_HIDDEN *
					     sizeof(uint16_t)) &&
		test_linear_card_upload(&replay->card, replay->input, 0,
					(uint64_t)rows * MODEL_HIDDEN *
						sizeof(uint16_t));
	const uint64_t skipped =
		replay->small ?
			0u :
			(uint64_t)first * MODEL_HIDDEN * sizeof(uint16_t);
	const struct official_head_description description = {
		.weights = { replay->weights.byte_offset + skipped,
			     (uint64_t)outputs * MODEL_HIDDEN *
				     sizeof(uint16_t) },
		.rows = replay->input,
		.output = replay->output,
		.output_size = outputs,
		.row_count = rows
	};
	struct kernel_launch legacy = { 0 };
	if (passed && index < FULL_CASES) {
		const struct dense_linear_description old = {
			.weights = description.weights,
			.rows = replay->input,
			.output = replay->legacy,
			.output_size = outputs,
			.input_size = MODEL_HIDDEN,
			.row_count = rows,
			.weight_type = DENSE_BFLOAT16,
			.output_type = DENSE_FLOAT32,
			.output_scale = 1.0f
		};
		passed = test_linear_expect_status(
				 dense_linear_prepare(
					 &old, replay->card.arena_byte_count,
					 &replay->dense, &legacy),
				 STATUS_OK, "prepare generic dense control") &&
			 test_linear_card_run(&replay->card, &legacy, 1) &&
			 test_linear_card_fetch(&replay->card) &&
			 put_output(replay, index,
				    replay->legacy_records[index],
				    replay->legacy, false);
	}
	for (uint32_t form = 0; passed && form < FORMS; form++) {
		struct kernel_launch candidate;
		const result_status status =
			form == 0 ? official_head_prepare(
					    &description,
					    replay->card.arena_byte_count,
					    replay->kernels[form], &candidate) :
				    official_head_reused_prepare(
					    &description,
					    replay->card.arena_byte_count,
					    replay->kernels[form], &candidate);
		passed = test_linear_expect_status(
				 status, STATUS_OK,
				 "prepare canonical M1 head") &&
			 test_linear_card_run(&replay->card, &candidate, 1);
		if (passed && replay->timed &&
		    (index == 0 || index == CACHED_BATCH_CASE))
			passed =
				timing(replay, &legacy, &candidate, rows, form);
		passed = passed && test_linear_card_fetch(&replay->card) &&
			 put_output(replay, index, replay->records[index][form],
				    replay->output, true) &&
			 test_linear_card_expect_untouched(&replay->card,
							   replay->input,
							   "input unchanged");
	}
	(void)printf(
		"HEAD_CASE,index=%u,rows=%u,first=%u,outputs=%u,passed=%u\n",
		index, rows, first, outputs, passed ? 1u : 0u);
	return passed;
}

int main(int argument_count, char **arguments)
{
	(void)alarm(TIME_LIMIT_SECONDS);
	if (argument_count == 1) {
		(void)puts(
			"official_head_replay_card_test: no fixture; no comparison executed");
		return EXIT_SUCCESS;
	}
	if (argument_count != ARGUMENTS ||
	    (strcmp(arguments[MODE_ARGUMENT], "check") != 0 &&
	     strcmp(arguments[MODE_ARGUMENT], "time") != 0 &&
	     strcmp(arguments[MODE_ARGUMENT], "small") != 0))
		return EXIT_FAILURE;
	struct replay replay = {
		.timed = strcmp(arguments[MODE_ARGUMENT], "time") == 0,
		.small = strcmp(arguments[MODE_ARGUMENT], "small") == 0
	};
	const uint64_t vector_bytes = (uint64_t)LINEAR_MAXIMUM_ROWS *
				      MODEL_VOCABULARY_VALUE * sizeof(float);
	const uint64_t weight_bytes = replay.small ? SMALL_WEIGHT_BYTES :
						     WEIGHT_BYTES;
	replay.expected = malloc((size_t)vector_bytes);
	replay.received = malloc((size_t)vector_bytes);
	bool passed = replay.expected != NULL && replay.received != NULL &&
		      test_linear_expect_status(
			      test_capture_open(arguments[REFERENCE_ARGUMENT],
						&replay.reference),
			      STATUS_OK, "open original inputs") &&
		      test_linear_card_open(&replay.card, 0, WORK_CAPACITY,
					    replay.small ?
						    SMALL_WEIGHT_CAPACITY :
						    WEIGHT_CAPACITY);
	if (passed) {
		test_linear_card_clear(&replay.card);
		replay.weights =
			test_linear_card_weights(&replay.card, weight_bytes);
		replay.input = test_linear_card_work(
			&replay.card, (uint64_t)LINEAR_MAXIMUM_ROWS *
					      MODEL_HIDDEN * sizeof(uint16_t));
		replay.output =
			test_linear_card_work(&replay.card, vector_bytes);
		replay.legacy =
			test_linear_card_work(&replay.card, vector_bytes);
		FILE *const weights = fopen(arguments[WEIGHTS_ARGUMENT], "rb");
		unsigned char *const stored =
			test_linear_card_writable(&replay.card, replay.weights);
		passed = test_linear_card_has_room(&replay.card) &&
			 weights != NULL && stored != NULL;
		if (passed)
			passed = fread(stored, 1, (size_t)weight_bytes,
				       weights) == (size_t)weight_bytes &&
				 fgetc(weights) == EOF && ferror(weights) == 0;
		if (weights != NULL)
			passed = fclose(weights) == 0 && passed;
		passed = passed && test_linear_card_load(&replay.card) &&
			 test_linear_expect_status(
				 official_head_kernel_prepare(
					 &backend_cuda_value, replay.card.card,
					 &replay.kernels[0]),
				 STATUS_OK, "prepare direct kernel") &&
			 test_linear_expect_status(
				 official_head_reused_kernel_prepare(
					 &backend_cuda_value, replay.card.card,
					 &replay.kernels[1]),
				 STATUS_OK, "prepare reused kernel") &&
			 test_linear_expect_status(
				 dense_linear_kernels_prepare(
					 &backend_cuda_value, replay.card.card,
					 &replay.dense),
				 STATUS_OK, "prepare dense control") &&
			 plan_output(&replay, arguments[OUTPUT_ARGUMENT]);
	}
	for (uint32_t index = 0; passed && index < CASES; index++)
		if (selected(&replay, index))
			passed = run_case(&replay, index);
	passed = passed &&
		 test_linear_card_expect_weights_untouched(&replay.card);
	const result_status closed = test_capture_writer_close(
		replay.writer, passed,
		"{\"scope\":\"independent canonical M1 head; BF16 weights/input; no norm/sampling/runtime\"}",
		passed ? "{\"passed\":true}" : "{\"passed\":false}");
	passed = passed && test_linear_expect_status(closed, STATUS_OK,
						     "close head capture");
	test_linear_card_close(&replay.card);
	test_capture_close(replay.reference);
	free(replay.expected);
	free(replay.received);
	return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
