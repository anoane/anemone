/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * draft_round_card_test: a synthetic draft segment without a host turnaround between verify, its device-written
 * step/views, window, block core, kept collapse, sequential Markov/sampler pairs and confidence. The large embedding,
 * expert and vocabulary products are covered by program data-flow tests and their separate product card tests.
 * Every round byte and ring slot is checked against independent host references. Runs in both PRO target classes;
 * the draft itself always prepares native arithmetic. --break changes a confidence weight on the card alone.
 */
#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "anemone/draft.h"
#include "anemone/sampler.h"
#include "attention_core_reference.h"
#include "attention_state_reference.h"
#include "draft_reference.h"
#include "hyper_connection_reference.h"
#include "linear_card_fixture.h"
#include "sampler_reference.h"

#define WORK_CAPACITY (16ull << 20u)
#define WEIGHTS_CAPACITY (1ull << 20u)
#define TEST_SECONDS 180u
#define OPTION_ARGUMENTS 2
#define POSITIONS STEP_DRAFT_MOST
#define TARGET_ROWS (POSITIONS + 1u)
#define VOCABULARY 256u
#define SESSIONS 2u
#define SESSION 1u
#define CONTEXT 1024u
#define FIRST_POSITION 128
#define FIRST_TOKEN 17
#define OTHER_TOKEN 63
#define END_TOKEN 1u
#define HIDDEN MODEL_HIDDEN
#define RANK MODEL_MARKOV_RANK
#define RING_VALUES (SESSIONS * MODEL_RING_SLOTS * MODEL_HEAD_DIMS)
#define ROTATION_VALUES (MODEL_ROTATION_ROW_BYTES / sizeof(float))
#define HEAD_VALUES (MODEL_HEADS_ROW_BYTES / sizeof(uint16_t))
#define BF16_ONE 0x3F80u
#define HALF_CONTEXT 0x3800u
#define STREAM_SEED 0x524F554E44ull
#define LOGIT_PERIOD 19u
#define LOGIT_SCALE 0.125f
#define TEMPERATURE 0.6f
#define TARGET_LOGIT 2.0f
#define CONFIDENCE_SCALE 0.25
#define BROKEN_WEIGHT 0x4200u
#define LAUNCH_CAPACITY (7u + POSITIONS * 2u)
#define COUNT_OF(table) ((uint32_t)(sizeof(table) / sizeof((table)[0])))

static const uint64_t seeds[] = { 0, 1, 0x243F6A8885A308D3ull };
static uint16_t embedding[VOCABULARY * RANK], head[VOCABULARY * RANK];
static uint16_t streams[POSITIONS][MODEL_STREAMS * HIDDEN];
static uint16_t collapse[POSITIONS][HIDDEN], markov[POSITIONS][RANK];
static uint16_t confidence_weight[MODEL_DRAFT_CONFIDENCE_INPUTS];
static uint16_t norm_weight[HIDDEN], window_weight[MODEL_HEAD_DIMS];
static uint16_t window_input[POSITIONS][MODEL_HEAD_DIMS],
	ring_expected[RING_VALUES];
static uint16_t core_expected[POSITIONS][HEAD_VALUES];
static float pre[POSITIONS][MODEL_STREAMS],
	rotation[POSITIONS][ROTATION_VALUES];
static float logits[POSITIONS][VOCABULARY],
	expected_logits[POSITIONS][VOCABULARY];

struct ranges {
	struct linear_range step, result, round, draft_step, views, sessions;
	struct linear_range streams, pre, norm, hidden, collapse, logits,
		sampling, results, markov;
	struct linear_range embedding, head, confidence, rotation, window_input,
		window_weight, ring;
	struct linear_range queries, sinks, core;
};

static bool status_holds(result_status status)
{
	return test_linear_expect_status(status, STATUS_OK,
					 "the segment launch is prepared");
}

static void inputs_make(void)
{
	for (uint32_t token = 0; token < VOCABULARY; token++) {
		test_linear_values(embedding + (size_t)token * RANK, RANK,
				   STREAM_SEED + token,
				   TEST_LINEAR_VALUES_PLAIN);
		test_linear_values(head + (size_t)token * RANK, RANK,
				   STREAM_SEED + VOCABULARY + token,
				   TEST_LINEAR_VALUES_PLAIN);
	}
	for (uint32_t index = 0; index < HIDDEN; index++) {
		norm_weight[index] = BF16_ONE;
	}
	for (uint32_t index = 0; index < MODEL_HEAD_DIMS; index++) {
		window_weight[index] = BF16_ONE;
	}
	for (uint32_t index = 0; index < MODEL_DRAFT_CONFIDENCE_INPUTS;
	     index++) {
		confidence_weight[index] =
			test_linear_bfloat16_nearest(CONFIDENCE_SCALE);
	}
	for (uint32_t position = 0; position < POSITIONS; position++) {
		test_linear_values(streams[position], MODEL_STREAMS * HIDDEN,
				   STREAM_SEED + position,
				   TEST_LINEAR_VALUES_PLAIN);
		for (uint32_t stream = 0; stream < MODEL_STREAMS; stream++) {
			pre[position][stream] =
				(float)(stream + 1u) / (float)MODEL_STREAMS;
		}
		test_hyper_collapse(streams[position], pre[position],
				    collapse[position]);
		for (uint32_t index = 0; index < MODEL_HEAD_DIMS; index++) {
			window_input[position][index] = BF16_ONE;
		}
		for (uint32_t angle = 0; angle < ROTATION_VALUES;
		     angle += MODEL_PAIR_VALUES) {
			rotation[position][angle] = 1.0f;
		}
		for (uint32_t token = 0; token < VOCABULARY; token++) {
			logits[position][token] =
				(float)((token + position) % LOGIT_PERIOD) *
				LOGIT_SCALE;
		}
	}
}

static void ranges_make(struct test_linear_card *card, struct ranges *ranges)
{
#define WORK(name, bytes) ranges->name = test_linear_card_work(card, (bytes))
#define WEIGHT(name, values) \
	ranges->name = test_linear_card_weights(card, sizeof(values))
	WORK(step, STEP_BLOCK_BYTES_VALUE + TARGET_ROWS * STEP_ROW_BYTES_VALUE);
	WORK(result, STEP_RESULT_BYTES + TARGET_ROWS * STEP_ROW_RESULT_BYTES);
	WORK(round, STEP_ROUND_BYTES);
	WORK(draft_step,
	     STEP_BLOCK_BYTES_VALUE + POSITIONS * STEP_ROW_BYTES_VALUE);
	WORK(views, (uint64_t)POSITIONS * STEP_DRAFT_VIEW_BYTES);
	WORK(sessions, (uint64_t)SESSIONS * STEP_SESSION_BYTES);
	WORK(streams, sizeof(streams));
	WORK(pre, sizeof(pre));
	WORK(hidden, sizeof(collapse));
	WORK(collapse, sizeof(collapse));
	WORK(logits, sizeof(logits));
	WORK(sampling, sizeof(logits[0]));
	WORK(results, (uint64_t)POSITIONS * STEP_DRAFT_RESULT_BYTES);
	WORK(markov, sizeof(markov));
	WORK(rotation, sizeof(rotation));
	WORK(window_input, sizeof(window_input));
	WORK(ring, sizeof(ring_expected));
	WORK(queries, (uint64_t)POSITIONS * MODEL_HEADS_ROW_BYTES);
	WORK(core, sizeof(core_expected));
	WORK(sinks, MODEL_HEADS_VALUE * sizeof(float));
	WEIGHT(norm, norm_weight);
	WEIGHT(window_weight, window_weight);
	WEIGHT(embedding, embedding);
	WEIGHT(head, head);
	WEIGHT(confidence, confidence_weight);
#undef WORK
#undef WEIGHT
}

static uint32_t draw_reference(const float *row,
			       const struct step_session *session,
			       int32_t position)
{
	const uint32_t largest = test_sampler_greedy(row, VOCABULARY);
	if (session->temperature == 0.0f) {
		return largest;
	}
	float weights[VOCABULARY];
	uint32_t kept[VOCABULARY];
	for (uint32_t token = 0; token < VOCABULARY; token++) {
		weights[token] = test_sampler_weight(row[token], row[largest],
						     session->temperature);
		kept[token] = token;
	}
	struct test_sampler_race race;
	test_sampler_race(weights, kept, VOCABULARY, session->seed, position,
			  &race);
	return race.winner;
}

static bool segment_prepare(struct test_linear_card *card,
			    const struct ranges *ranges,
			    const struct hyper_kernels_record *hyper,
			    const struct draft_kernels_record *draft,
			    const struct sampler_kernels_record *sampler,
			    const struct attention_state_kernels *state,
			    const struct attention_core_kernels *core,
			    struct kernel_launch launches[LAUNCH_CAPACITY],
			    uint32_t *count_out)
{
	uint32_t count = 0;
	const struct verify_matches_description verify = {
		ranges->step,
		ranges->result,
		ranges->round,
		ranges->draft_step,
		ranges->views,
		TARGET_ROWS,
		POSITIONS,
		END_TOKEN,
		MODEL_VOCABULARY_VALUE,
		0,
		0,
		0,
		0
	};
	const struct attention_window_description window = {
		.step = ranges->draft_step,
		.rotation = ranges->rotation,
		.window_input = ranges->window_input,
		.window_norm = ranges->window_weight,
		.ring = ranges->ring,
		.row_count = POSITIONS,
		.session_capacity = SESSIONS,
		.context_token_limit = CONTEXT
	};
	const struct attention_core_description attention = {
		.step = ranges->draft_step,
		.queries = ranges->queries,
		.ring = ranges->ring,
		.sinks = ranges->sinks,
		.rotation = ranges->rotation,
		.output = ranges->core,
		.row_count = POSITIONS,
		.session_capacity = SESSIONS,
		.context_token_limit = CONTEXT,
		.head_count = MODEL_HEADS_VALUE,
		.order = LINEAR_ORDER_FIXED,
		.arithmetic_class = card->arithmetic_class ==
						    ARITHMETIC_CLASS_SM_80 ?
					    ARITHMETIC_CLASS_SM_80 :
					    ARITHMETIC_CLASS_SM_120_NATIVE,
		.window_mode = ATTENTION_CORE_WINDOW_BLOCK
	};
	const struct hyper_final_description final = { ranges->streams,
						       ranges->pre,
						       ranges->norm,
						       ranges->hidden,
						       ranges->collapse,
						       POSITIONS,
						       0 };
	uint32_t core_count = 0;
	bool ready = status_holds(verify_matches_prepare(
		&verify, card->arena_byte_count, draft, &launches[count++]));
	ready = ready && status_holds(attention_window_prepare(
				 &window, card->arena_byte_count, state,
				 &launches[count++]));
	struct kernel_launch core_launches[ATTENTION_CORE_MOST_LAUNCHES];
	ready = ready &&
		status_holds(attention_core_prepare(
			&attention, card->arena_byte_count, core, core_launches,
			&core_count)) &&
		test_linear_expect(core_count == 1,
				   "the block core is one native launch");
	if (!ready) {
		return false;
	}
	launches[count++] = core_launches[0];
	ready = status_holds(hyper_final_prepare(&final, card->arena_byte_count,
						 hyper, &launches[count++]));
	for (uint32_t position = 0; ready && position < POSITIONS; position++) {
		const struct linear_range row_logits = {
			ranges->logits.byte_offset +
				position * sizeof(logits[0]),
			sizeof(logits[0])
		};
		const struct draft_markov_bias_description bias = {
			ranges->draft_step,
			ranges->results,
			ranges->embedding,
			ranges->head,
			ranges->logits,
			ranges->markov,
			position,
			POSITIONS,
			VOCABULARY,
			0,
			0
		};
		const struct sampler_description draw = {
			.step = { ranges->views.byte_offset +
					  (uint64_t)position *
						  STEP_DRAFT_VIEW_BYTES,
				  STEP_DRAFT_VIEW_BYTES },
			.sessions = ranges->sessions,
			.logits = row_logits,
			.weights = ranges->sampling,
			.result = { ranges->results.byte_offset +
					    (uint64_t)position *
						    STEP_DRAFT_RESULT_BYTES,
				    STEP_DRAFT_RESULT_BYTES },
			.vocabulary_size = VOCABULARY,
			.session_capacity = SESSIONS,
			.row_count = 1
		};
		ready = status_holds(
			draft_markov_bias_prepare(&bias, card->arena_byte_count,
						  draft, &launches[count++]));
		ready = ready && status_holds(sampler_prepare_entry(
					 &draw, card->arena_byte_count, sampler,
					 &launches[count++]));
	}
	const struct draft_confidence_description confidence = {
		ranges->draft_step,
		ranges->collapse,
		ranges->markov,
		ranges->confidence,
		ranges->results,
		ranges->round,
		POSITIONS,
		0,
		0
	};
	ready = ready && status_holds(draft_confidence_prepare(
				 &confidence, card->arena_byte_count, draft,
				 &launches[count++]));
	*count_out = count;
	return ready;
}

static void round_run(struct test_linear_card *card,
		      const struct ranges *ranges,
		      const struct kernel_launch *launches,
		      uint32_t launch_count, uint32_t accepted,
		      uint64_t sequence, uint64_t seed, bool broken)
{
	const struct step_block block = { sequence, TARGET_ROWS, 0 };
	const struct step_session session = { seed,
					      seed == 0 ? 0.0f : TEMPERATURE,
					      1.0f };
	struct step_row_record rows[TARGET_ROWS];
	struct step_row_result results[TARGET_ROWS];
	for (uint32_t row = 0; row < TARGET_ROWS; row++) {
		rows[row] = (struct step_row_record){
			FIRST_POSITION + (int32_t)row,
			FIRST_TOKEN + (int32_t)row, SESSION,
			STEP_ROW_LOGITS |
				(row == 0 ? STEP_ROW_DRAFTS : STEP_ROW_PROPOSED)
		};
		results[row] = (struct step_row_result){
			row < accepted ? FIRST_TOKEN + (int32_t)row + 1 :
					 OTHER_TOKEN,
			TARGET_LOGIT
		};
	}
	struct step_round expected = { 0 };
	struct step_block draft_block;
	struct step_row_record draft_rows[POSITIONS];
	unsigned char views[POSITIONS * STEP_DRAFT_VIEW_BYTES];
	test_draft_verify(&block, rows, results, TARGET_ROWS, POSITIONS,
			  END_TOKEN, MODEL_DRAFT_NOISE_TOKEN, &expected,
			  &draft_block, draft_rows, views);
	for (uint32_t index = 0; index < RING_VALUES; index++) {
		ring_expected[index] = HALF_CONTEXT;
	}
	bool ready = true;
#define PUT(name, offset, data, bytes)                                      \
	ready = ready && test_linear_card_put(card, ranges->name, (offset), \
					      (data), (bytes))
	PUT(step, 0, &block, sizeof(block));
	PUT(step, STEP_BLOCK_BYTES_VALUE, rows, sizeof(rows));
	PUT(result, STEP_RESULT_BYTES, results, sizeof(results));
	PUT(sessions, (uint64_t)SESSION * STEP_SESSION_BYTES, &session,
	    sizeof(session));
	PUT(streams, 0, streams, sizeof(streams));
	PUT(pre, 0, pre, sizeof(pre));
	PUT(norm, 0, norm_weight, sizeof(norm_weight));
	PUT(window_input, 0, window_input, sizeof(window_input));
	PUT(window_weight, 0, window_weight, sizeof(window_weight));
	PUT(rotation, 0, rotation, sizeof(rotation));
	PUT(ring, 0, ring_expected, sizeof(ring_expected));
	PUT(embedding, 0, embedding, sizeof(embedding));
	PUT(head, 0, head, sizeof(head));
	PUT(confidence, 0, confidence_weight, sizeof(confidence_weight));
	PUT(logits, 0, logits, sizeof(logits));
	if (broken) {
		const uint16_t weight = BROKEN_WEIGHT;
		PUT(confidence, 0, &weight, sizeof(weight));
	}
#undef PUT
	memcpy(expected_logits, logits, sizeof(logits));
	uint32_t input_token = (uint32_t)expected.anchor_token;
	for (uint32_t position = 0; position < POSITIONS; position++) {
		test_draft_markov_bias(embedding, head, VOCABULARY, input_token,
				       expected_logits[position],
				       markov[position]);
		input_token = draw_reference(expected_logits[position],
					     &session,
					     draft_rows[position].position);
		expected.proposals[position] = (int32_t)input_token;
		expected.confidence_logits[position] = test_draft_confidence(
			collapse[position], markov[position],
			confidence_weight);
		const uint32_t slot = (uint32_t)draft_rows[position].position %
				      MODEL_RING_SLOTS;
		test_attention_window_row(
			window_input[position], window_weight, 1.0f,
			rotation[position],
			ring_expected + ((size_t)SESSION * MODEL_RING_SLOTS +
					 slot) * MODEL_HEAD_DIMS);
	}
	const double core_value =
		((double)MODEL_WINDOW *
			 test_attention_half_value(HALF_CONTEXT) +
		 POSITIONS) /
		(double)(MODEL_WINDOW + POSITIONS + 1u);
	for (uint32_t position = 0; position < POSITIONS; position++) {
		for (uint32_t index = 0; index < HEAD_VALUES; index++) {
			core_expected[position][index] =
				test_linear_bfloat16_nearest(core_value);
		}
	}
	if (ready && test_linear_card_load(card) &&
	    test_linear_card_run(card, launches, launch_count) &&
	    test_linear_card_fetch(card)) {
		(void)test_linear_card_expect_bytes(
			card, ranges->round, 0, &expected, sizeof(expected),
			"the complete round is the host reference's");
		(void)test_linear_card_expect_bytes(
			card, ranges->draft_step, 0, &draft_block,
			sizeof(draft_block),
			"the device step carries the round sequence");
		(void)test_linear_card_expect_bytes(
			card, ranges->draft_step, STEP_BLOCK_BYTES_VALUE,
			draft_rows, sizeof(draft_rows),
			"the device step starts at the accepted anchor");
		(void)test_linear_card_expect_bytes(
			card, ranges->views, 0, views, sizeof(views),
			"sampler views carry the draft positions and session");
		(void)test_linear_card_expect_bytes(
			card, ranges->ring, 0, ring_expected,
			sizeof(ring_expected),
			"the device step addresses only the draft window slots");
		(void)test_linear_card_expect_bytes(
			card, ranges->core, 0, core_expected,
			sizeof(core_expected),
			"the block core includes all five draft window rows");
		(void)test_linear_card_expect_bytes(
			card, ranges->collapse, 0, collapse, sizeof(collapse),
			"confidence consumes the collapse before normalization");
		(void)test_linear_card_expect_bytes(
			card, ranges->markov, 0, markov, sizeof(markov),
			"each Markov launch consumes the preceding sampled token");
		(void)test_linear_card_expect_bytes(
			card, ranges->logits, 0, expected_logits,
			sizeof(expected_logits),
			"the sequential Markov biases match the host reference");
	}
}

int main(int argument_count, char **arguments)
{
	const bool broken = argument_count == OPTION_ARGUMENTS &&
			    strcmp(arguments[1], "--break") == 0;
	const bool emulated = argument_count == OPTION_ARGUMENTS &&
			      strcmp(arguments[1], "--pro_emulation=1") == 0;
	if (argument_count != 1 && !broken && !emulated) {
		(void)fprintf(
			stderr,
			"usage: draft_round_card_test [--pro_emulation=1 | --break]\n");
		return EXIT_FAILURE;
	}
	(void)alarm(TEST_SECONDS);
	inputs_make();
	uint32_t card_count = 0;
	if (!status_holds(backend_cuda_value.card_count(&card_count)) ||
	    !test_linear_expect(card_count != 0,
				"a selected card is visible")) {
		return EXIT_FAILURE;
	}
	for (uint32_t card_index = 0; card_index < card_count; card_index++) {
		static struct test_linear_card card;
		struct hyper_kernels_record hyper = { 0 };
		struct router_kernels_record router = { 0 };
		struct attention_state_kernels state = { 0 };
		struct attention_core_kernels target_core = { 0 }, core = { 0 };
		struct linear_kernels_record target_linear = { 0 },
					     draft_linear = { 0 };
		struct draft_kernels_record draft = { 0 };
		struct sampler_kernels_record sampler = { 0 };
		if (!test_linear_card_open(&card, card_index, WORK_CAPACITY,
					   WEIGHTS_CAPACITY)) {
			continue;
		}
		const uint32_t draft_class =
			card.arithmetic_class == ARITHMETIC_CLASS_SM_80 ?
				ARITHMETIC_CLASS_SM_80 :
				ARITHMETIC_CLASS_SM_120_NATIVE;
		const uint32_t target_class =
			draft_class != ARITHMETIC_CLASS_SM_80 && emulated ?
				ARITHMETIC_CLASS_SM_120 :
				draft_class;
		/* Both target and draft tables coexist on the same real card, including the emulated/native pair. */
		bool ready =
			status_holds(linear_kernels_prepare(
				&backend_cuda_value, card.card, target_class,
				&target_linear)) &&
			status_holds(hyper_kernels_prepare(
				&backend_cuda_value, card.card, &hyper)) &&
			status_holds(router_kernels_prepare(
				&backend_cuda_value, card.card, &router)) &&
			status_holds(attention_state_kernels_prepare(
				&backend_cuda_value, card.card, &state)) &&
			status_holds(attention_core_kernels_prepare(
				&backend_cuda_value, card.card, target_class,
				&target_core)) &&
			status_holds(linear_kernels_prepare(
				&backend_cuda_value, card.card, draft_class,
				&draft_linear)) &&
			status_holds(attention_core_kernels_prepare(
				&backend_cuda_value, card.card, draft_class,
				&core)) &&
			status_holds(draft_kernels_prepare(
				&backend_cuda_value, card.card, draft_class,
				&hyper, &router, &state, &core, &draft)) &&
			status_holds(sampler_kernels_prepare(
				&backend_cuda_value, card.card, &sampler));
		ready = ready &&
			test_linear_expect(
				target_core.attention_core_block ==
						BACKEND_NO_KERNEL &&
					core.attention_core_block !=
						BACKEND_NO_KERNEL &&
					(target_class == draft_class ||
					 (target_core.attention_core !=
						  core.attention_core &&
					  target_linear.trunk_linear_quantized[0] !=
						  draft_linear
							  .trunk_linear_quantized
								  [0])),
				"emulated target and native draft use distinct product/core handles");
		struct ranges ranges;
		ranges_make(&card, &ranges);
		struct kernel_launch launches[LAUNCH_CAPACITY];
		uint32_t launch_count = 0;
		ready = ready && test_linear_card_has_room(&card) &&
			segment_prepare(&card, &ranges, &hyper, &draft,
					&sampler, &state, &core, launches,
					&launch_count);
		for (uint32_t seed = 0; ready && seed < COUNT_OF(seeds);
		     seed++) {
			for (uint32_t accepted = 0; accepted <= POSITIONS;
			     accepted++) {
				const uint64_t sequence =
					1u + seed * TARGET_ROWS + accepted;
				test_linear_context("card %" PRIu32
						    ", seed %" PRIu32
						    ", accepted %" PRIu32,
						    card_index, seed, accepted);
				round_run(&card, &ranges, launches,
					  launch_count, accepted, sequence,
					  seeds[seed], broken);
			}
		}
		(void)printf("draft segment: target class %" PRIu32
			     ", draft class %" PRIu32 ", %u cases\n",
			     target_class, draft_class,
			     COUNT_OF(seeds) * TARGET_ROWS);
		test_linear_card_close(&card);
	}
	return test_linear_failures() == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
