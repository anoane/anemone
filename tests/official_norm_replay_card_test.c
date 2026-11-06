/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Isolated original-input norm replay, with actual mean/inverse output and an
 * independent FP64 sanity bound. Strict original comparison is external and has
 * no tolerance. Legacy mode observes only stored 5120-wide norm outputs.
 */
#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "anemone/hyper_connection.h"
#include "anemone/official_norm.h"
#include "capture_reader.h"
#include "capture_writer.h"
#include "card_timing.h"
#include "linear_card_fixture.h"

#define WORK_CAPACITY (1ull << 20u)
#define WEIGHTS_CAPACITY (1ull << 16u)
#define NORM_SITES 4u
#define FIELDS 3u
#define LEGACY_SITES 2u
#define MOST_ROWS LINEAR_MAXIMUM_ROWS
#define MOST_WIDTH MODEL_HIDDEN
#define NAME_BYTES TEST_CAPTURE_WRITER_NAME_CAPACITY
#define TIME_LIMIT_SECONDS 300u
#define MEAN_BOUND (64.0 * (double)FLT_EPSILON)
#define OUTPUT_BOUND (1.0 / 64.0)
#define TIME_SAMPLES 9u
#define TIME_REPLAYS 128u
#define TIME_WARM 8u

enum field { OUTPUT_FIELD, MEAN_FIELD, INVERSE_FIELD };
enum argument {
	PROGRAM_ARGUMENT,
	MODE_ARGUMENT,
	REFERENCE_ARGUMENT,
	OUTPUT_ARGUMENT,
	ARGUMENTS
};
static const char *const sites[NORM_SITES] = { "attn", "ffn", "qr", "kv" };
static const char *const fields[FIELDS] = { "output", "mean", "inverse" };
static const uint32_t widths[NORM_SITES] = { MODEL_HIDDEN, MODEL_HIDDEN,
					     MODEL_QUERY_RANK,
					     MODEL_HEAD_DIMS };

struct replay {
	struct test_linear_card card;
	struct test_capture *reference;
	struct test_capture_writer *writer;
	struct hyper_kernels_record hyper;
	uint32_t kernel, rows;
	uint32_t reference_rows;
	uint32_t records[NORM_SITES][MOST_ROWS][FIELDS];
	bool legacy;
	bool timed;
};

struct mode {
	const char *name;
	bool legacy;
	bool timed;
	uint32_t reference_rows;
};

static bool time_norm(struct replay *replay, const struct kernel_launch *launch,
		      uint32_t site)
{
	uint32_t graph = 0;
	if (!test_linear_expect_status(
		    backend_cuda_value.graph_capture(replay->card.card, launch,
						     1, &graph),
		    STATUS_OK, "capture norm timing graph"))
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
				"NORM_TIME,site=%s,rows=%u,shape=%u,legacy=%u,batch=%u,sample=%u,microseconds=%.9g\n",
				sites[site], replay->rows,
				replay->reference_rows,
				replay->legacy ? 1u : 0u, batch, sample,
				samples[batch * TIME_SAMPLES + sample]);
	}
	return true;
}

static bool read_values(struct replay *replay, const char *name,
			enum test_capture_type type, uint64_t count,
			void *destination)
{
	const struct test_capture_tensor *const tensor =
		test_capture_find(replay->reference, name);
	if (tensor == NULL || tensor->type != (uint32_t)type ||
	    tensor->element_count != count)
		return test_linear_expect(false, name);
	return test_linear_expect_status(
		test_capture_read(replay->reference, name, type,
				  tensor->dimensions, tensor->dimension_count,
				  0, count, destination,
				  count * test_capture_element_bytes(type)),
		STATUS_OK, name);
}

static bool record_name(char *name, uint32_t site, uint32_t row,
			const char *part)
{
	const int count = snprintf(name, NAME_BYTES,
				   "norm.%s.position_%" PRIu32 ".%s",
				   sites[site], row, part);
	return test_linear_expect(count > 0 && (uint32_t)count < NAME_BYTES,
				  "norm record name fits");
}

static bool plan_output(struct replay *replay, const char *folder)
{
	if (!test_linear_expect_status(
		    test_capture_writer_create(NORM_SITES * replay->rows *
						       FIELDS,
					       &replay->writer),
		    STATUS_OK, "create norm writer"))
		return false;
	for (uint32_t site = 0;
	     site < (replay->legacy ? LEGACY_SITES : NORM_SITES); site++) {
		for (uint32_t row = 0; row < replay->rows; row++) {
			for (uint32_t field = 0;
			     field < (replay->legacy ? 1u : FIELDS); field++) {
				char name[NAME_BYTES];
				const uint64_t dimensions[] = {
					field == OUTPUT_FIELD ? widths[site] :
								1u
				};
				if (!record_name(name, site, row,
						 fields[field]) ||
				    !test_linear_expect_status(
					    test_capture_writer_add(
						    replay->writer, name,
						    field == OUTPUT_FIELD ?
							    TEST_CAPTURE_BFLOAT16 :
							    TEST_CAPTURE_FLOAT32,
						    dimensions, 1, 1, 0, row,
						    "Independent norm on exact original input",
						    &replay->records[site][row]
								    [field]),
					    STATUS_OK, "add norm result"))
					return false;
			}
		}
	}
	return test_linear_expect_status(
		test_capture_writer_open(replay->writer, folder), STATUS_OK,
		"open norm output");
}

static bool independent_check(const uint16_t *input, const uint16_t *weight,
			      const uint16_t *output, uint32_t width,
			      const float *statistics, bool legacy)
{
	double sum = 0.0;
	for (uint32_t column = 0; column < width; column++) {
		const double value = test_linear_bfloat16_value(input[column]);
		sum += value * value;
	}
	const double mean = sum / (double)width;
	const double inverse = 1.0 / sqrt(mean + (double)MODEL_NORM_EPSILON);
	bool passed =
		legacy ||
		(isfinite(statistics[0]) && isfinite(statistics[1]) &&
		 fabs((double)statistics[0] - mean) <= mean * MEAN_BOUND &&
		 fabs((double)statistics[1] - inverse) <= inverse * MEAN_BOUND);
	for (uint32_t column = 0; column < width; column++) {
		const double expected =
			test_linear_bfloat16_value(weight[column]) *
			(test_linear_bfloat16_value(input[column]) * inverse);
		const double actual =
			test_linear_bfloat16_value(output[column]);
		passed = passed && isfinite(expected) && isfinite(actual) &&
			 fabs(actual - expected) <=
				 fabs(expected) * OUTPUT_BOUND + DBL_MIN;
	}
	return test_linear_expect(
		passed,
		"independent FP64 norm sanity bound; strict source bits checked separately");
}

static bool run_site(struct replay *replay, uint32_t site)
{
	struct test_linear_card *const card = &replay->card;
	const uint32_t width = widths[site];
	const uint64_t row_bytes = (uint64_t)width * sizeof(uint16_t);
	test_linear_card_clear(card);
	struct official_norm_description description = {
		.input = test_linear_card_work(card, row_bytes * replay->rows),
		.weight = test_linear_card_weights(card, row_bytes),
		.output = test_linear_card_work(card, row_bytes * replay->rows),
		.statistics = test_linear_card_work(
			card, (uint64_t)replay->rows *
				      OFFICIAL_NORM_STATISTICS * sizeof(float)),
		.width = width,
		.row_count = replay->rows
	};
	uint16_t weight[MOST_WIDTH];
	char name[NAME_BYTES];
	const int count =
		snprintf(name, sizeof(name), "norm.%s.weight", sites[site]);
	bool passed = test_linear_card_has_room(card) && count > 0 &&
		      (size_t)count < sizeof(name) &&
		      read_values(replay, name, TEST_CAPTURE_BFLOAT16, width,
				  weight) &&
		      test_linear_card_put(card, description.weight, 0, weight,
					   row_bytes);
	for (uint32_t row = 0; passed && row < replay->rows; row++) {
		uint16_t values[MOST_WIDTH];
		passed = record_name(name, site, row, "input") &&
			 read_values(replay, name, TEST_CAPTURE_BFLOAT16, width,
				     values) &&
			 test_linear_card_put(card, description.input,
					      row * row_bytes, values,
					      row_bytes);
	}
	struct kernel_launch launch;
	passed = passed &&
		 test_linear_expect_status(
			 official_norm_shape_prepare(&description,
						     replay->reference_rows,
						     card->arena_byte_count,
						     replay->kernel, &launch),
			 STATUS_OK, "prepare independent norm");
	if (passed && replay->legacy) {
		const struct hyper_draft_inputs_description old = {
			.input = description.input,
			.weight = description.weight,
			.output = description.output,
			.row_count = replay->rows,
			.mode = HYPER_ROW_NORM
		};
		passed = test_linear_expect_status(
			hyper_draft_inputs_prepare(&old, card->arena_byte_count,
						   &replay->hyper, &launch),
			STATUS_OK, "prepare legacy norm");
	}
	passed = passed && test_linear_card_load(card) &&
		 test_linear_card_run(card, &launch, 1) &&
		 (!replay->timed || time_norm(replay, &launch, site)) &&
		 test_linear_card_fetch(card) &&
		 test_linear_card_expect_untouched(card, description.input,
						   "norm input unchanged") &&
		 test_linear_card_expect_weights_untouched(card);
	for (uint32_t row = 0; passed && row < replay->rows; row++) {
		uint16_t input[MOST_WIDTH], output[MOST_WIDTH];
		float statistics[OFFICIAL_NORM_STATISTICS];
		passed =
			test_linear_card_get(card, description.input,
					     row * row_bytes, input,
					     row_bytes) &&
			test_linear_card_get(card, description.output,
					     row * row_bytes, output,
					     row_bytes) &&
			test_linear_card_get(card, description.statistics,
					     (uint64_t)row * sizeof(statistics),
					     statistics, sizeof(statistics)) &&
			independent_check(input, weight, output, width,
					  statistics, replay->legacy);
		for (uint32_t field = 0;
		     passed && field < (replay->legacy ? 1u : FIELDS); field++)
			passed = test_linear_expect_status(
				test_capture_writer_put(
					replay->writer,
					replay->records[site][row][field], 0,
					field == OUTPUT_FIELD ?
						(const void *)output :
						(const void *)&statistics[field -
									  1u]),
				STATUS_OK, "write norm result");
	}
	return passed;
}

int main(int count, char **arguments)
{
	(void)alarm(TIME_LIMIT_SECONDS);
	if (count == 1) {
		(void)puts(
			"official_norm_replay_card_test: no fixture; no comparison executed");
		return EXIT_SUCCESS;
	}
	if (count != ARGUMENTS)
		return EXIT_FAILURE;
	static const struct mode modes[] = { { "source", false, false, 0 },
					     { "legacy", true, false, 0 },
					     { "time_source", false, true, 0 },
					     { "time_legacy", true, true, 0 },
					     { "shape1", false, false, 1 },
					     { "shape3", false, false, 3 },
					     { "shape4", false, false, 4 },
					     { "shape8", false, false, 8 },
					     { "time_shape1", false, true, 1 },
					     { "time_shape3", false, true, 3 },
					     { "time_shape4", false, true, 4 },
					     { "time_shape8", false, true, 8 } };
	const struct mode *mode = NULL;
	for (uint32_t index = 0; index < sizeof(modes) / sizeof(modes[0]);
	     index++) {
		if (strcmp(arguments[MODE_ARGUMENT], modes[index].name) == 0)
			mode = modes + index;
	}
	if (mode == NULL)
		return EXIT_FAILURE;
	struct replay replay = { .legacy = mode->legacy,
				 .timed = mode->timed,
				 .reference_rows = mode->reference_rows };
	bool passed = test_linear_expect_status(
		test_capture_open(arguments[REFERENCE_ARGUMENT],
				  &replay.reference),
		STATUS_OK, "open norm reference");
	const struct test_capture_tensor *const tokens =
		test_capture_find(replay.reference, "input.token_ids");
	passed = passed && tokens != NULL &&
		 tokens->element_count <= MOST_ROWS &&
		 tokens->element_count > 0;
	replay.rows = passed ? (uint32_t)tokens->element_count : 0;
	if (replay.reference_rows == 0)
		replay.reference_rows = replay.rows;
	passed =
		passed &&
		test_linear_card_open(&replay.card, 0, WORK_CAPACITY,
				      WEIGHTS_CAPACITY) &&
		test_linear_expect_status(
			official_norm_kernel_prepare(&backend_cuda_value,
						     replay.card.card,
						     &replay.kernel),
			STATUS_OK, "prepare norm kernel") &&
		test_linear_expect_status(
			hyper_kernels_prepare(&backend_cuda_value,
					      replay.card.card, &replay.hyper),
			STATUS_OK, "prepare hyper kernels") &&
		(!replay.legacy ||
		 test_linear_expect_status(
			 backend_cuda_value.kernel_prepare(
				 replay.card.card, KERNEL_FILE_HYPER_CONNECTION,
				 KERNEL_HYPER_DRAFT_INPUTS,
				 HYPER_DRAFT_INPUTS_ARGUMENT_BYTES,
				 &replay.hyper.draft_inputs),
			 STATUS_OK, "prepare optional legacy norm kernel")) &&
		plan_output(&replay, arguments[OUTPUT_ARGUMENT]);
	for (uint32_t site = 0;
	     passed && site < (replay.legacy ? LEGACY_SITES : NORM_SITES);
	     site++)
		passed = run_site(&replay, site);
	char conditions[NAME_BYTES];
	(void)snprintf(
		conditions, sizeof(conditions),
		"{\"rows\":%u,\"reference_rows\":%u,\"legacy\":%s,\"timed\":%s}",
		replay.rows, replay.reference_rows,
		replay.legacy ? "true" : "false",
		replay.timed ? "true" : "false");
	const result_status closed = test_capture_writer_close(
		replay.writer, passed, conditions, "{}");
	passed = passed && test_linear_expect_status(closed, STATUS_OK,
						     "close norm output");
	test_linear_card_close(&replay.card);
	test_capture_close(replay.reference);
	return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
