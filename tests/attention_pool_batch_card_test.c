/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Compare a step-wide compressor plus deferred keeps with the ordered eight-row slab walk. Includes an odd
 * first position, forty rows of one session, slot wraps, eight sessions, stale rows and a restarted session. */
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "anemone/attention_state.h"
#include "attention_state_card_fixture.h"
#include "linear_card_fixture.h"

#define WORK_BYTES (16ull << 20u)
#define WEIGHT_BYTES (1ull << 20u)
#define TIME_LIMIT_SECONDS 900u
#define ROWS 48u
#define SESSIONS 8u
#define CONTEXT 128u
#define LONG_OWNER_ROWS 40u
#define SHORT_OWNER_ROWS 8u
#define REGULAR_OWNER_ROWS 6u
#define RAGGED_OWNER_ROWS 7u
#define CASES 5u
#define SINGLE_WRAP_CASE 4u
#define REPEATS 16u
#define RESTART_CASE 3u
#define RAGGED_CASE 2u
#define REGULAR_CASE 1u
#define INPUT_SCALE 0.001f
#define SCORE_SCALE 0.125f
#define DIM_PERIOD 17u
#define POSITION_STRIDE 3u
#define SESSION_STRIDE 97u
#define BFLOAT16_ONE 0x3f80u
#define ROTATION_FLOATS (MODEL_ROTATION_ROW_BYTES / sizeof(float))
#define DIMS MODEL_HEAD_DIMS
#define SLOT_VALUES \
	(SESSIONS * ATTENTION_COMPRESSOR_SLOTS * ATTENTION_SLOT_VECTORS * DIMS)
#define BREAK_ARGUMENTS 2
#define TEST_WRONG_USAGE 2
#define DEFERRED_LAUNCHES 2u

static struct test_attention_row step_rows[ROWS];
static float inputs[ROWS][DIMS], scores[ROWS][DIMS];
static float rotations[ROWS][ROTATION_FLOATS];
static float initial_slots[SLOT_VALUES];
static uint16_t norm[DIMS];
static bool break_keep, old_inline;

static struct attention_state_description
pool_set_aside(struct test_linear_card *card, uint32_t rows)
{
	const uint64_t entries =
		(uint64_t)SESSIONS * CONTEXT / MODEL_POOLED_RATIO;
	return (struct attention_state_description){
		.step = test_linear_card_work(
			card, STEP_BLOCK_BYTES_VALUE +
				      (uint64_t)rows * STEP_ROW_BYTES_VALUE),
		.rotation = test_linear_card_work(
			card, (uint64_t)rows * MODEL_ROTATION_ROW_BYTES),
		.pool_norm = test_linear_card_weights(card, sizeof(norm)),
		.key_norm = test_linear_card_weights(card, MODEL_KEY_BYTES),
		.latent = test_linear_card_work(card, (uint64_t)rows * DIMS *
							      sizeof(uint16_t)),
		.pool = test_linear_card_work(card, entries * DIMS *
							    sizeof(uint16_t)),
		.key_input = test_linear_card_work(
			card, (uint64_t)rows * MODEL_KEY_BYTES),
		.keys = test_linear_card_work(card, entries * MODEL_KEY_BYTES),
		.pool_input = test_linear_card_work(
			card, (uint64_t)rows * DIMS * sizeof(float)),
		.pool_score = test_linear_card_work(
			card, (uint64_t)rows * DIMS * sizeof(float)),
		.slots = test_linear_card_work(card, sizeof(initial_slots)),
		.row_count = rows,
		.session_capacity = SESSIONS,
		.context_token_limit = CONTEXT,
		.theta = 1,
		.has_index = 1,
		.pool_ratio = MODEL_POOLED_RATIO,
	};
}

static bool inputs_put(struct test_linear_card *card,
		       const struct attention_state_description *pool,
		       uint32_t first, uint32_t rows, uint64_t sequence)
{
	struct test_attention_row metadata[ROWS];
	for (uint32_t row = 0; row < pool->row_count; row++)
		metadata[row] =
			row < rows ?
				step_rows[first + row] :
				(struct test_attention_row){ -1, UINT32_MAX };
	const uint64_t values_bytes = (uint64_t)rows * sizeof(inputs[0]);
	const uint64_t angle_bytes = (uint64_t)rows * sizeof(rotations[0]);
	return test_attention_step_put(card, pool->step, metadata,
				       pool->row_count, rows, sequence) &&
	       test_linear_card_put(card, pool->pool_input, 0, inputs[first],
				    values_bytes) &&
	       test_linear_card_put(card, pool->pool_score, 0, scores[first],
				    values_bytes) &&
	       test_linear_card_put(card, pool->rotation, 0, rotations,
				    angle_bytes) &&
	       test_linear_card_upload(card, pool->pool_input, 0,
				       values_bytes) &&
	       test_linear_card_upload(card, pool->pool_score, 0,
				       values_bytes) &&
	       test_linear_card_upload(card, pool->rotation, 0, angle_bytes);
}

static uint32_t case_inputs(uint32_t trial, uint32_t *positions)
{
	uint32_t rows = 0;
	if (trial == RESTART_CASE)
		positions[0] = 0;
	if (trial == SINGLE_WRAP_CASE)
		positions[0] =
			1; /* preserve the restarted position0, whose slot the final even row will wrap */
	for (uint32_t session = 0; session < SESSIONS; session++) {
		const uint32_t count =
			trial == SINGLE_WRAP_CASE ?
				(session == 0 ? ROWS : 0u) :
			trial == REGULAR_CASE ?
				REGULAR_OWNER_ROWS :
			trial == RAGGED_CASE ?
				(session == 0 ? RAGGED_OWNER_ROWS :
				 session == 1 ? REGULAR_OWNER_ROWS :
						0u) :
			trial == RESTART_CASE ?
				(session == 0 ? SHORT_OWNER_ROWS : 0u) :
				(session == 0 ? LONG_OWNER_ROWS :
				 session == 1 ? SHORT_OWNER_ROWS :
						0u);
		for (uint32_t index = 0; index < count; index++, rows++) {
			const uint32_t position = positions[session]++;
			step_rows[rows] =
				(struct test_attention_row){ (int32_t)position,
							     session };
			for (uint32_t dim = 0; dim < DIMS; dim++) {
				inputs[rows][dim] =
					(float)(1u + session * SESSION_STRIDE +
						position * POSITION_STRIDE +
						dim % DIM_PERIOD) *
					INPUT_SCALE;
				scores[rows][dim] =
					(float)((position + dim) % DIM_PERIOD) *
					SCORE_SCALE;
			}
		}
	}
	return rows;
}

static void slot_mode_set(struct kernel_launch *launch, uint32_t mode)
{
	struct attention_compressor_arguments record;
	memcpy(&record, launch->argument, sizeof(record));
	record.slot_mode = mode;
	memcpy(launch->argument, &record, sizeof(record));
}

static void card_run(uint32_t card_index)
{
	struct test_linear_card card;
	if (!test_linear_card_open(&card, card_index, WORK_BYTES, WEIGHT_BYTES))
		return;
	struct attention_state_kernels kernels;
	struct kernel_launch wide_launches[ATTENTION_STATE_LAUNCH_COUNT] = { 0 };
	struct kernel_launch slab_launches[ATTENTION_STATE_LAUNCH_COUNT] = { 0 };
	const struct attention_state_description wide =
		pool_set_aside(&card, ROWS);
	const struct attention_state_description slab =
		pool_set_aside(&card, LINEAR_MAXIMUM_ROWS);
	bool passed =
		test_linear_card_has_room(&card) &&
		test_linear_expect_status(
			attention_state_kernels_prepare(&backend_cuda_value,
							card.card, &kernels),
			STATUS_OK, "pool kernels prepared") &&
		test_linear_expect_status(
			attention_pool_prepare(&wide, card.arena_byte_count,
					       &kernels, wide_launches),
			STATUS_OK, "step-wide pool prepared") &&
		test_linear_expect_status(
			attention_pool_prepare(&slab, card.arena_byte_count,
					       &kernels, slab_launches),
			STATUS_OK, "slab pool prepared");
	const struct attention_state_description *const pools[] = { &wide,
								    &slab };
	for (uint32_t pool = 0;
	     pool < sizeof(pools) / sizeof(pools[0]) && passed; pool++) {
		passed = test_linear_card_put(&card, pools[pool]->pool_norm, 0,
					      norm, sizeof(norm)) &&
			 test_linear_card_put(&card, pools[pool]->key_norm, 0,
					      norm, MODEL_KEY_BYTES) &&
			 test_linear_card_put(&card, pools[pool]->slots, 0,
					      initial_slots,
					      sizeof(initial_slots));
	}
	struct kernel_launch deferred[DEFERRED_LAUNCHES] = {
		wide_launches[ATTENTION_STATE_COMPRESSOR],
		wide_launches[ATTENTION_STATE_COMPRESSOR]
	};
	slot_mode_set(&deferred[0], old_inline ?
					    ATTENTION_COMPRESSOR_SLOTS_INLINE :
					    ATTENTION_COMPRESSOR_SLOTS_DEFER);
	slot_mode_set(&deferred[1], break_keep ?
					    ATTENTION_COMPRESSOR_SLOTS_DEFER :
					    ATTENTION_COMPRESSOR_SLOTS_KEEP);
	uint32_t checked = 0;
	for (uint32_t repeat = 0; repeat < REPEATS && passed; repeat++) {
		uint32_t positions[SESSIONS];
		for (uint32_t session = 0; session < SESSIONS; session++)
			positions[session] = 1;
		passed = test_linear_card_load(&card);
		for (uint32_t trial = 0; trial < CASES && passed; trial++) {
			test_linear_context("card %" PRIu32 " repeat %" PRIu32
					    " pool case %" PRIu32,
					    card_index, repeat, trial);
			const uint32_t rows = case_inputs(trial, positions);
			passed = inputs_put(&card, &wide, 0, rows, trial) &&
				 test_linear_card_run(
					 &card, deferred,
					 old_inline ? 1u : DEFERRED_LAUNCHES);
			for (uint32_t first = 0; first < rows && passed;
			     first += LINEAR_MAXIMUM_ROWS) {
				const uint32_t count =
					rows - first < LINEAR_MAXIMUM_ROWS ?
						rows - first :
						LINEAR_MAXIMUM_ROWS;
				passed =
					inputs_put(&card, &slab, first, count,
						   trial) &&
					test_linear_card_run(
						&card,
						&slab_launches
							[ATTENTION_STATE_COMPRESSOR],
						1) &&
					test_linear_card_fetch(&card);
				for (uint32_t row = 0; row < count && passed;
				     row++)
					if (((uint32_t)step_rows[first + row]
						     .position +
					     1u) % MODEL_POOLED_RATIO ==
					    0)
						passed = test_linear_card_expect_same(
							&card, wide.latent,
							(uint64_t)(first +
								   row) *
								sizeof(norm),
							slab.latent,
							(uint64_t)row *
								sizeof(norm),
							sizeof(norm),
							"emitting latents match slabs");
			}
			passed =
				passed &&
				test_linear_card_expect_same(
					&card, wide.pool, 0, slab.pool, 0,
					wide.pool.byte_count,
					"every pool byte matches slabs") &&
				test_linear_card_expect_same(
					&card, wide.slots, 0, slab.slots, 0,
					wide.slots.byte_count,
					"every retained slot byte matches slabs");
			checked += passed ? 1u : 0u;
		}
	}
	(void)test_linear_expect(passed && checked == REPEATS * CASES,
				 "all mixed-session pool cases completed");
	(void)printf("card %" PRIu32 ": %" PRIu32 " mixed-session pool cases\n",
		     card_index, checked);
	test_linear_card_close(&card);
}

int main(int argument_count, char **arguments)
{
	break_keep = argument_count == BREAK_ARGUMENTS &&
		     strcmp(arguments[1], "--break-keep") == 0;
	old_inline = argument_count == BREAK_ARGUMENTS &&
		     strcmp(arguments[1], "--old-inline") == 0;
	if (argument_count != 1 && !break_keep && !old_inline)
		return TEST_WRONG_USAGE;
	(void)alarm(TIME_LIMIT_SECONDS);
	for (uint32_t dim = 0; dim < DIMS; dim++)
		norm[dim] = BFLOAT16_ONE;
	for (uint32_t row = 0; row < ROWS; row++)
		for (uint32_t angle = 0; angle < ROTATION_FLOATS;
		     angle += MODEL_PAIR_VALUES)
			rotations[row][angle] = 1.0f;
	for (uint32_t value = 0; value < SLOT_VALUES; value++)
		initial_slots[value] =
			(float)(1u + value % DIM_PERIOD) * INPUT_SCALE;
	uint32_t cards = 0;
	if (test_linear_expect_status(backend_cuda_value.card_count(&cards),
				      STATUS_OK, "cards counted") &&
	    test_linear_expect(cards != 0 && cards <= BACKEND_MAXIMUM_CARDS,
			       "visible test cards present"))
		for (uint32_t card = 0; card < cards; card++)
			card_run(card);
	(void)printf("attention_pool_batch_card_test: %d failures\n",
		     test_linear_failures());
	return test_linear_failures() == 0 ? 0 : 1;
}
