/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Replay actual official layer-zero inputs through the existing dense and hyper operators.
 * Each site is independently injected, including the preceding site's carried pre.
 * The carried_ mode prefix requires explicit incoming carried_pre records.
 * Other modes retain the historical generated-attention-pre convention, which
 * is not the original injected FFN argument for every crossed shape.
 * The output is a strict-comparison capture, never a tolerance-based pass claim.
 */
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "anemone/dense_linear.h"
#include "anemone/hyper_connection.h"
#include "anemone/official_hc.h"
#include "anemone/official_hc_partitioned.h"
#include "capture_reader.h"
#include "capture_writer.h"
#include "card_timing.h"
#include "linear_card_fixture.h"

#define WORK_CAPACITY (1ull << 22u)
#define WEIGHTS_CAPACITY (1ull << 22u)
#define NAME_CAPACITY 96u
#define HIDDEN MODEL_HIDDEN
#define STREAMS MODEL_STREAMS
#define MIXES MODEL_MIXES
#define FLAT_INPUTS (HIDDEN * STREAMS)
#define COMB_VALUES (STREAMS * STREAMS)
#define MOST_POSITIONS LINEAR_MAXIMUM_ROWS
#define SITES 2u
#define MATRIX_DIMENSIONS 2u
#define FIELDS 5u
#define LEAST_CAPTURED_PROMPT_ROWS 3u
#define TIME_LIMIT_SECONDS 300u
#define WARM_RUNS 3u
#define TIME_REPLAYS 128u
#define TIME_SAMPLES 9u

enum field { MIX_FIELD, PRE_FIELD, POST_FIELD, COMB_FIELD, NORM_FIELD };
enum argument {
	PROGRAM_ARGUMENT,
	MODE_ARGUMENT,
	REFERENCE_ARGUMENT,
	OUTPUT_ARGUMENT,
	ARGUMENTS
};
static const char *const field_names[FIELDS] = { "mixes", "pre", "post", "comb",
						 "normed" };
static const uint32_t field_values[FIELDS] = { MIXES, STREAMS, STREAMS,
					       COMB_VALUES, HIDDEN };

struct replay {
	struct test_linear_card card;
	struct test_capture *capture;
	struct test_capture_writer *writer;
	struct dense_linear_kernels dense;
	struct hyper_kernels_record hyper;
	struct linear_range input, carried, mixes, post, comb, hidden, output;
	struct linear_range partials;
	struct linear_range weights, scale, base, norm;
	uint32_t positions;
	uint32_t records[SITES][MOST_POSITIONS][FIELDS];
	bool injected_mixes;
	bool source_order;
	bool staged;
	bool timed;
	bool raw_projection;
	bool explicit_carried;
	bool partitioned;
	uint32_t reference_rows;
	struct official_hc_kernels official;
	struct official_hc_partitioned_kernels partition_kernels;
};

static bool read_values(struct replay *replay, const char *name,
			enum test_capture_type type, uint64_t elements,
			void *output)
{
	const struct test_capture_tensor *const tensor =
		test_capture_find(replay->capture, name);
	if (tensor == NULL || tensor->element_count != elements ||
	    tensor->type != (uint32_t)type)
		return test_linear_expect(false, name);
	return test_linear_expect_status(
		test_capture_read(replay->capture, name, type,
				  tensor->dimensions, tensor->dimension_count,
				  0, elements, output,
				  elements * test_capture_element_bytes(type)),
		STATUS_OK, name);
}

static bool plan_output(struct replay *replay, const char *folder)
{
	if (!test_linear_expect_status(
		    test_capture_writer_create(
			    SITES * replay->positions *
				    (replay->raw_projection ? 1u : FIELDS),
			    &replay->writer),
		    STATUS_OK, "create capture writer"))
		return false;
	for (uint32_t site = 0; site < SITES; site++) {
		for (uint32_t position = 0; position < replay->positions;
		     position++) {
			for (uint32_t field = 0;
			     field < (replay->raw_projection ? 1u : FIELDS);
			     field++) {
				char name[NAME_CAPACITY];
				const int length = snprintf(
					name, sizeof(name),
					"layer_0.position_%" PRIu32
					".%s_site.%s",
					position, site == 0 ? "attn" : "ffn",
					replay->raw_projection ?
						"raw_projection" :
						field_names[field]);
				const uint64_t dimensions[] = {
					field == COMB_FIELD ?
						STREAMS :
						field_values[field],
					STREAMS
				};
				if (length < 0 ||
				    (size_t)length >= sizeof(name) ||
				    !test_linear_expect_status(
					    test_capture_writer_add(
						    replay->writer, name,
						    field == NORM_FIELD ?
							    TEST_CAPTURE_BFLOAT16 :
							    TEST_CAPTURE_FLOAT32,
						    dimensions,
						    field == COMB_FIELD ?
							    MATRIX_DIMENSIONS :
							    1,
						    1, 0, position,
						    "Existing engine operator on official inputs",
						    &replay->records[site]
								    [position]
								    [field]),
					    STATUS_OK, "add output record"))
					return false;
			}
		}
	}
	return test_linear_expect_status(
		test_capture_writer_open(replay->writer, folder), STATUS_OK,
		"open output capture");
}

static bool site_inputs(struct replay *replay, uint32_t site)
{
	struct test_linear_card *const card = &replay->card;
	char name[NAME_CAPACITY];
	const char *const prefix = site == 0 ? "hc_attn" : "hc_ffn";
	const struct linear_range ranges[] = { replay->weights, replay->scale,
					       replay->base, replay->norm };
	const char *const suffixes[] = { "fn", "scale", "base", "norm" };
	const uint64_t counts[] = { (uint64_t)MIXES * FLAT_INPUTS, HYPER_SCALES,
				    MIXES, HIDDEN };
	for (uint32_t parameter = 0;
	     parameter < sizeof(ranges) / sizeof(ranges[0]); parameter++) {
		if (parameter == sizeof(ranges) / sizeof(ranges[0]) - 1u)
			(void)snprintf(name, sizeof(name),
				       "layer_0.parameter.%s_norm",
				       site == 0 ? "attn" : "ffn");
		else
			(void)snprintf(name, sizeof(name),
				       "layer_0.parameter.%s_%s", prefix,
				       suffixes[parameter]);
		unsigned char *const destination =
			test_linear_card_writable(card, ranges[parameter]);
		if (destination == NULL ||
		    !read_values(
			    replay, name,
			    parameter == sizeof(ranges) / sizeof(ranges[0]) -
						    1u ?
				    TEST_CAPTURE_BFLOAT16 :
				    TEST_CAPTURE_FLOAT32,
			    counts[parameter], destination))
			return false;
	}
	for (uint32_t position = 0; position < replay->positions; position++) {
		uint16_t streams[FLAT_INPUTS];
		float carried[STREAMS] = { 1.0f };
		(void)snprintf(name, sizeof(name),
			       "layer_0.position_%" PRIu32 ".%s_site.streams",
			       position, site == 0 ? "attn" : "ffn");
		if (!read_values(replay, name, TEST_CAPTURE_BFLOAT16,
				 FLAT_INPUTS, streams))
			return false;
		if (replay->explicit_carried) {
			(void)snprintf(name, sizeof(name),
				       "layer_0.position_%" PRIu32
				       ".%s_site.carried_pre",
				       position, site == 0 ? "attn" : "ffn");
			if (!read_values(replay, name, TEST_CAPTURE_FLOAT32,
					 STREAMS, carried))
				return false;
		} else if (site != 0) {
			(void)snprintf(name, sizeof(name),
				       "layer_0.position_%" PRIu32
				       ".attn_site.pre",
				       position);
			if (!read_values(replay, name, TEST_CAPTURE_FLOAT32,
					 STREAMS, carried))
				return false;
		}
		if (!test_linear_card_put(card, replay->input,
					  (uint64_t)position * sizeof(streams),
					  streams, sizeof(streams)) ||
		    !test_linear_card_put(card, replay->carried,
					  (uint64_t)position * sizeof(carried),
					  carried, sizeof(carried)))
			return false;
		if (replay->injected_mixes) {
			float mixes[MIXES];
			(void)snprintf(name, sizeof(name),
				       "layer_0.position_%" PRIu32
				       ".%s_site.mixes",
				       position, site == 0 ? "attn" : "ffn");
			if (!read_values(replay, name, TEST_CAPTURE_FLOAT32,
					 MIXES, mixes) ||
			    !test_linear_card_put(card, replay->mixes,
						  (uint64_t)position *
							  sizeof(mixes),
						  mixes, sizeof(mixes)))
				return false;
		}
	}
	return test_linear_card_load(card);
}

static bool run_site(struct replay *replay, uint32_t site)
{
	struct test_linear_card *const card = &replay->card;
	struct kernel_launch dense, hyper[HYPER_SITE_LAUNCH_COUNT];
	const struct dense_linear_description product = {
		.weights = replay->weights,
		.rows = replay->input,
		.output = replay->mixes,
		.output_size = MIXES,
		.input_size = FLAT_INPUTS,
		.row_count = replay->positions,
		.weight_type = DENSE_FLOAT32,
		.output_type = DENSE_FLOAT32,
		.flags = replay->raw_projection ? 0u : DENSE_ROW_SCALE,
		.output_scale = 1.0f,
		.epsilon = MODEL_NORM_EPSILON
	};
	const struct hyper_site_description finish = {
		.streams = replay->input,
		.carried_pre = replay->carried,
		.mixes = replay->mixes,
		.scale = replay->scale,
		.base = replay->base,
		.norm_weight = replay->norm,
		.post = replay->post,
		.comb = replay->comb,
		.hidden = replay->hidden,
		.output = replay->output,
		.row_count = replay->positions
	};
	bool passed =
		site_inputs(replay, site) &&
		test_linear_expect_status(
			dense_linear_prepare(&product, card->arena_byte_count,
					     &replay->dense, &dense),
			STATUS_OK, "prepare dense") &&
		test_linear_expect_status(
			hyper_site_prepare(&finish, card->arena_byte_count,
					   &replay->hyper, hyper),
			STATUS_OK, "prepare hyper");
	if (passed && replay->source_order) {
		passed = test_linear_expect_status(
			official_hc_shape_prepare(
				&product,
				replay->reference_rows != 0 ?
					replay->reference_rows :
					replay->positions,
				card->arena_byte_count, &replay->dense,
				&replay->official, &dense),
			STATUS_OK, "prepare source-order launch");
	}
	struct kernel_launch projections[OFFICIAL_HC_PARTITIONED_LAUNCHES];
	uint32_t projection_count = 1;
	if (passed) {
		projections[0] = dense;
		if (replay->partitioned) {
			passed = test_linear_expect_status(
				official_hc_partitioned_prepare(
					&product, &replay->partials,
					card->arena_byte_count, &replay->dense,
					&replay->official,
					&replay->partition_kernels,
					projections),
				STATUS_OK,
				"prepare two-pass source F1 projection");
			projection_count = OFFICIAL_HC_PARTITIONED_LAUNCHES;
		}
	}
	if (passed && !replay->injected_mixes)
		passed = test_linear_card_run(card, projections,
					      projection_count);
	passed = passed &&
		 (replay->raw_projection ||
		  test_linear_card_run(card, &hyper[HYPER_SITE_MIX_FINISH],
				       1)) &&
		 test_linear_card_fetch(card);
	const struct linear_range fields[] = { replay->mixes, replay->carried,
					       replay->post, replay->comb,
					       replay->hidden };
	for (uint32_t position = 0; passed && position < replay->positions;
	     position++) {
		for (uint32_t field = 0;
		     passed && field < (replay->raw_projection ? 1u : FIELDS);
		     field++) {
			float values[HIDDEN];
			const uint64_t bytes = field_values[field] *
					       (field == NORM_FIELD ?
							sizeof(uint16_t) :
							sizeof(float));
			passed =
				test_linear_card_get(card, fields[field],
						     (uint64_t)position * bytes,
						     values, bytes) &&
				test_linear_expect_status(
					test_capture_writer_put(
						replay->writer,
						replay->records[site][position]
							       [field],
						0, values),
					STATUS_OK, "write result");
		}
	}
	if (passed && replay->timed) {
		struct kernel_launch baseline = { 0 };
		passed = test_linear_expect_status(
			dense_linear_prepare(&product, card->arena_byte_count,
					     &replay->dense, &baseline),
			STATUS_OK, "prepare timing baseline");
		uint32_t graphs[TEST_CARD_TIMING_FORMS];
		for (uint32_t form = 0; passed && form < TEST_CARD_TIMING_FORMS;
		     form++)
			passed = test_linear_expect_status(
				backend_cuda_value.graph_capture(
					card->card,
					form == 0 ? &baseline : projections,
					form == 0 ? 1u : projection_count,
					&graphs[form]),
				STATUS_OK, "capture timed projection");
		unsigned char *const before =
			passed ? malloc((size_t)card->work_byte_count) : NULL;
		passed = passed &&
			 test_linear_expect(
				 before != NULL,
				 "timing readback snapshot allocated");
		if (passed) {
			memcpy(before, card->received,
			       (size_t)card->work_byte_count);
			const struct test_card_timing_form
				forms[TEST_CARD_TIMING_FORMS] = {
					{ &graphs[0], 1 }, { &graphs[1], 1 }
				};
			double samples[TEST_CARD_TIMING_FORMS * TIME_SAMPLES],
				savings[TIME_SAMPLES];
			passed = test_card_timing_pair(card->card, forms,
						       WARM_RUNS, TIME_REPLAYS,
						       TIME_SAMPLES, samples,
						       savings);
			if (passed) {
				(void)printf(
					"TIME %s rows=%u reference_rows=%u staged=%u partitioned=%u",
					site == 0 ? "attention" : "ffn",
					replay->positions,
					replay->reference_rows != 0 ?
						replay->reference_rows :
						replay->positions,
					replay->staged ? 1u : 0u,
					replay->partitioned ? 1u : 0u);
				test_card_timing_print(samples, TIME_SAMPLES,
						       1.0, "legacy");
				test_card_timing_print(samples + TIME_SAMPLES,
						       TIME_SAMPLES, 1.0,
						       "candidate");
				(void)printf("\n");
			}
			passed =
				passed &&
				test_linear_card_run(card, projections,
						     projection_count) &&
				test_linear_card_fetch(card) &&
				test_linear_expect(
					memcmp(before, card->received,
					       (size_t)card->work_byte_count) ==
						0,
					"full work readback unchanged after timed replay") &&
				test_linear_card_expect_weights_untouched(card);
		}
		free(before);
	}
	return passed;
}

int main(int argument_count, char **arguments)
{
	(void)alarm(TIME_LIMIT_SECONDS);
	if (argument_count == 1) {
		(void)puts(
			"official_site_replay_card_test: no fixture supplied; no official comparison executed");
		return EXIT_SUCCESS;
	}
	const char *const supplied_method =
		argument_count > 1 ? arguments[MODE_ARGUMENT] : "";
	const bool explicit_carried =
		strncmp(supplied_method, "carried_", strlen("carried_")) == 0;
	const char *const method =
		supplied_method + (explicit_carried ? strlen("carried_") : 0u);
	const bool timed_partitioned =
		strncmp(method, "time_partitioned_shape",
			strlen("time_partitioned_shape")) == 0;
	const bool partitioned = strncmp(method, "partitioned_shape",
					 strlen("partitioned_shape")) == 0;
	const bool timed_staged_shape = strncmp(method, "time_staged_shape",
						strlen("time_staged_shape")) ==
					0;
	const bool timed_shape =
		strncmp(method, "time_shape", strlen("time_shape")) == 0;
	const bool staged_shape =
		strncmp(method, "staged_shape", strlen("staged_shape")) == 0;
	const char *const shape_method =
		method + (timed_partitioned  ? strlen("time_partitioned_") :
			  partitioned	     ? strlen("partitioned_") :
			  timed_staged_shape ? strlen("time_staged_") :
			  timed_shape	     ? strlen("time_") :
			  staged_shape	     ? strlen("staged_") :
					       0u);
	static const char *const shape_names[] = { "shape1", "shape3", "shape4",
						   "shape8" };
	static const uint32_t shape_rows[] = { 1, LEAST_CAPTURED_PROMPT_ROWS,
					       STREAMS, MOST_POSITIONS };
	uint32_t reference_rows = 0;
	for (uint32_t shape = 0;
	     argument_count > 1 &&
	     shape < sizeof(shape_rows) / sizeof(shape_rows[0]);
	     shape++) {
		if (strcmp(shape_method, shape_names[shape]) == 0)
			reference_rows = shape_rows[shape];
	}
	if (argument_count != ARGUMENTS ||
	    (reference_rows == 0 && strcmp(method, "computed") != 0 &&
	     strcmp(method, "injected") != 0 && strcmp(method, "source") != 0 &&
	     strcmp(method, "raw") != 0 && strcmp(method, "staged") != 0 &&
	     strcmp(method, "time_direct") != 0 &&
	     strcmp(method, "time_staged") != 0))
		return EXIT_FAILURE;
	if ((partitioned || timed_partitioned) && reference_rows != 1)
		return EXIT_FAILURE;
	struct replay replay = {
		.injected_mixes = strcmp(method, "injected") == 0,
		.explicit_carried = explicit_carried,
		.partitioned = partitioned || timed_partitioned,
		.reference_rows = reference_rows,
		.source_order = reference_rows != 0 ||
				strcmp(method, "source") == 0 ||
				strcmp(method, "staged") == 0 ||
				strcmp(method, "time_direct") == 0 ||
				strcmp(method, "time_staged") == 0,
		.staged = staged_shape || timed_staged_shape ||
			  strcmp(method, "staged") == 0 ||
			  strcmp(method, "time_staged") == 0,
		.timed = timed_shape || timed_staged_shape ||
			 timed_partitioned ||
			 strcmp(method, "time_direct") == 0 ||
			 strcmp(method, "time_staged") == 0,
		.raw_projection = strcmp(method, "raw") == 0
	};
	bool passed = test_linear_expect_status(
		test_capture_open(arguments[REFERENCE_ARGUMENT],
				  &replay.capture),
		STATUS_OK, "open reference");
	const struct test_capture_tensor *const tokens =
		test_capture_find(replay.capture, "input.token_ids");
	passed = passed && tokens != NULL && tokens->element_count > 0 &&
		 tokens->element_count <= MOST_POSITIONS;
	replay.positions = passed ? (uint32_t)tokens->element_count : 0;
	if (replay.source_order && replay.positions != 1 &&
	    replay.positions != LEAST_CAPTURED_PROMPT_ROWS &&
	    replay.positions != STREAMS && replay.positions != MOST_POSITIONS)
		passed = test_linear_expect(
			false,
			"source-order shape is not qualified for probing");
	passed = passed && test_linear_card_open(&replay.card, 0, WORK_CAPACITY,
						 WEIGHTS_CAPACITY);
	if (passed) {
		struct test_linear_card *const card = &replay.card;
		test_linear_card_clear(card);
		replay.input = test_linear_card_work(
			card, (uint64_t)replay.positions * FLAT_INPUTS *
				      sizeof(uint16_t));
		replay.carried = test_linear_card_work(
			card,
			(uint64_t)replay.positions * STREAMS * sizeof(float));
		replay.mixes = test_linear_card_work(
			card,
			(uint64_t)replay.positions * MIXES * sizeof(float));
		replay.post =
			test_linear_card_work(card, replay.carried.byte_count);
		replay.comb = test_linear_card_work(
			card, (uint64_t)replay.positions * COMB_VALUES *
				      sizeof(float));
		replay.hidden = test_linear_card_work(
			card,
			(uint64_t)replay.positions * HIDDEN * sizeof(uint16_t));
		if (replay.partitioned)
			replay.partials = test_linear_card_work(
				card, (uint64_t)replay.positions * MIXES *
					      OFFICIAL_HC_PARTITIONS *
					      sizeof(float));
		replay.output =
			test_linear_card_work(card, replay.hidden.byte_count);
		replay.weights = test_linear_card_weights(
			card, (uint64_t)MIXES * FLAT_INPUTS * sizeof(float));
		replay.scale = test_linear_card_weights(
			card, HYPER_SCALES * sizeof(float));
		replay.base =
			test_linear_card_weights(card, MIXES * sizeof(float));
		replay.norm = test_linear_card_weights(
			card, HIDDEN * sizeof(uint16_t));
		passed = test_linear_card_has_room(card) &&
			 test_linear_expect_status(
				 dense_linear_kernels_prepare(
					 &backend_cuda_value, card->card,
					 &replay.dense),
				 STATUS_OK, "prepare dense kernels") &&
			 test_linear_expect_status(
				 hyper_kernels_prepare(&backend_cuda_value,
						       card->card,
						       &replay.hyper),
				 STATUS_OK, "prepare hyper kernels") &&
			 plan_output(&replay, arguments[OUTPUT_ARGUMENT]);
		if (passed && replay.source_order)
			passed = test_linear_expect_status(
				official_hc_kernels_prepare(&backend_cuda_value,
							    card->card,
							    &replay.official),
				STATUS_OK, "prepare source-order kernel");
		if (passed && replay.staged)
			passed = test_linear_expect_status(
				official_hc_staged_prepare(&backend_cuda_value,
							   card->card,
							   &replay.official),
				STATUS_OK,
				"prepare staged source-order kernel");
		if (passed && replay.partitioned)
			passed = test_linear_expect_status(
				official_hc_partitioned_kernels_prepare(
					&backend_cuda_value, card->card,
					&replay.partition_kernels),
				STATUS_OK,
				"prepare two-pass source projection kernels");
	}
	for (uint32_t site = 0; passed && site < SITES; site++)
		passed = run_site(&replay, site);
	const result_status closed = test_capture_writer_close(
		replay.writer, passed,
		replay.explicit_carried ?
			"{\"mixes\":\"candidate\",\"carried\":\"explicit fixture input\"}" :
		replay.injected_mixes ?
			"{\"mixes\":\"official injected\"}" :
		replay.raw_projection ?
			"{\"mixes\":\"legacy raw diagnostic\"}" :
		replay.source_order ? "{\"mixes\":\"source order candidate\"}" :
				      "{\"mixes\":\"engine computed\"}",
		"{}");
	passed = passed &&
		 test_linear_expect_status(closed, STATUS_OK, "close output");
	test_linear_card_close(&replay.card);
	test_capture_close(replay.capture);
	(void)printf(
		"official_site_replay_card_test: capture %s; compare it separately for strict identity\n",
		passed ? "complete" : "FAILED");
	return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
