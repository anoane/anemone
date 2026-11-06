/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * attention_state_window_card_test: the window alone of kernel_attention_state.cu, the draft's context
 * (dspark_v1/DESIGN.md C2, K3), on the real cards.
 *
 *     attention_state_window_card_test [--break window | --trap <card>]
 *
 * It needs the cards free (`make card_tests`): a card without room FAILS the test. On every card:
 *   - PREPARE: steps of 1 to 8 rows of two sessions (positions from 0, across the ring's wrap) through the rotation
 *     step and the prepare of a layer of theta 0 (rotation use 0) and of theta 1 (use 1), then again through the
 *     rotation step and attention_window of the same rows and use: the ring byte for byte the prepare's.
 *   - CHUNK: CHUNK_ROWS rows of one session in one window launch, their positions across the ring's wrap, angles laid
 *     out by the host: every ring row bit for bit the reference's (test_attention_window_row) for a factor within
 *     RSQRT_PLACES float32 places of the reciprocal root of its operand (rsqrtf is approximate), every other slot
 *     untouched; the launch of the first 8 rows alone gives their rows the same bits.
 * Shown failing (--break, the test then passes only if that check alone failed): window, one word of one row's wkv
 * output changed on the card alone (CHUNK).
 * --trap <card>: in the builds that link the checked kernel modules, a row whose normed value no FP16 holds (a wkv row
 * of a single 1 under a kv_norm weight of 2^12: 2^12 sqrt(512) > 65,504) must end the window's launch with
 * STATUS_DEVICE_ERROR (DESIGN K3). A trap ends what a process can do with a card: a batch runs it as a step
 * of its own; with the release modules it runs nothing.
 */
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "anemone/attention_state.h"
#include "anemone/backend_seam.h"
#include "anemone/draft.h"
#include "anemone/status.h"
#include "anemone/step_records.h"
#include "attention_state_reference.h"
#include "linear_card_fixture.h"
#include "linear_reference.h"

#define TEST_TIME_LIMIT_SECONDS 1800
#define WORK_CAPACITY (24ull << 20u)
#define WEIGHTS_CAPACITY (1ull << 20u)
#define ROWS LINEAR_MAXIMUM_ROWS
#define CHUNK_ROWS MODEL_WIDE_MOST_ROWS
#define DIMS MODEL_HEAD_DIMS
#define QUERY_RANK MODEL_QUERY_RANK
#define RING_SLOTS MODEL_RING_SLOTS
#define ROTATION_FLOATS (MODEL_ROTATION_ROW_BYTES / sizeof(float))
#define USE_FLOATS (ROTATION_FLOATS / MODEL_ROTATION_USES)
#define WORDS (MODEL_THETAS * MODEL_FREQUENCIES)
#define SESSIONS 2u
#define PREPARE_CONTEXT 8192u
#define CHUNK_FIRST_POSITION \
	3000 /* the chunk's slots 3,000 .. 4,095, then 0 .. 2,999 */
#define CHUNK_CONTEXT 8192u
#define RSQRT_PLACES \
	3u /* rsqrtf: within 2 places of 1 / sqrt, 3 across a power of two */
#define NO_FACTOR (RSQRT_PLACES + 1u)
#define INPUT_STREAM 0x57494E44u
#define WEIGHT_STREAM 0x4B564E4Fu
#define FREQUENCY_STREAM 0x46524551u
#define ANGLE_STREAM 0x414E474Cu
#define HASH_FRACTION_SHIFT 40u
#define HASH_FRACTION_UNIT 0x1p-24
#define FREQUENCY_SPAN 0.5 /* frequencies from 0 to 0.5 radian a position */
#define ANGLE_SPAN 2.0 /* host angles: cos and sin from -1 to 1 */
#define HALF 0.5
#define POISON 0x7E00u /* an FP16 not-a-number: a slot no row writes */
#define BROKEN_ROW 5u
#define FLAG_ARGUMENTS \
	2 /* a flag and its value: --break <check>, --trap <card> */
#define TRAP_INPUT 0x3F80u /* bfloat16 1, the trap's row's one value */
#define TRAP_WEIGHT 0x4580u /* bfloat16 2^12: times sqrt(512) beyond FP16 */
#define DECIMAL 10
#define ROTATION_AND_PREPARE \
	2u /* the rotation step, then the prepare (or the window alone) */

enum test_status { TEST_PASSED = 0, TEST_FAILED = 1, TEST_WRONG_USAGE = 2 };
enum check { PREPARE_CHECK, CHUNK_CHECK, CHECK_COUNT };
static const char *const check_names[CHECK_COUNT] = { "PREPARE", "CHUNK" };
static const char *const break_names[CHECK_COUNT] = { "", "window" };
static const uint32_t step_row_counts[] = { 1, 2, 3, 5, 8 };
static uint32_t check_failures[CHECK_COUNT];
static int break_check = -1;

static uint16_t inputs[CHUNK_ROWS][DIMS]; /* wkv's output of every row */
static uint16_t query_ranks[ROWS][QUERY_RANK];
static uint16_t kv_norm[DIMS];
static uint16_t q_norm[QUERY_RANK];
static float frequencies[WORDS];
static float angles[CHUNK_ROWS][ROTATION_FLOATS];
static struct step_row_record step_rows[CHUNK_ROWS];
static uint16_t ring_back[SESSIONS * RING_SLOTS * DIMS];
static uint16_t ring_prepared[SESSIONS * RING_SLOTS * DIMS];
static uint16_t ring_poison[SESSIONS * RING_SLOTS * DIMS];
static uint16_t expected_row[DIMS];

static void check_fail(enum check check, const char *what, uint32_t row)
{
	if (check_failures[check] < ROWS) {
		test_linear_fail("%s: %s (row %" PRIu32 ")", check_names[check],
				 what, row);
	}
	check_failures[check]++;
}

static double hash_fraction(uint64_t stream, uint64_t position)
{
	return (double)(test_linear_hash(stream, position) >>
			HASH_FRACTION_SHIFT) *
	       HASH_FRACTION_UNIT;
}

static uint32_t bits_of(float value)
{
	uint32_t bits = 0;
	memcpy(&bits, &value, sizeof(bits));
	return bits;
}

static float float_of(uint32_t bits)
{
	float value = 0.0f;
	memcpy(&value, &bits, sizeof(value));
	return value;
}

static void inputs_make(void)
{
	for (uint32_t row = 0; row < CHUNK_ROWS; row++) {
		test_linear_values(inputs[row], DIMS, INPUT_STREAM + row,
				   TEST_LINEAR_VALUES_PLAIN);
		for (uint32_t index = 0; index < ROTATION_FLOATS; index++) {
			angles[row][index] =
				(float)((hash_fraction(ANGLE_STREAM + row,
						       index) -
					 HALF) *
					ANGLE_SPAN);
		}
	}
	for (uint32_t row = 0; row < ROWS; row++) {
		test_linear_values(query_ranks[row], QUERY_RANK,
				   INPUT_STREAM + CHUNK_ROWS + row,
				   TEST_LINEAR_VALUES_PLAIN);
	}
	test_linear_values(kv_norm, DIMS, WEIGHT_STREAM,
			   TEST_LINEAR_VALUES_PLAIN);
	test_linear_values(q_norm, QUERY_RANK, WEIGHT_STREAM + 1u,
			   TEST_LINEAR_VALUES_PLAIN);
	for (uint32_t word = 0; word < WORDS; word++) {
		frequencies[word] =
			(float)(hash_fraction(FREQUENCY_STREAM, word) *
				FREQUENCY_SPAN);
	}
	for (size_t index = 0;
	     index < sizeof(ring_poison) / sizeof(ring_poison[0]); index++) {
		ring_poison[index] = POISON;
	}
}

/* The places from the nearest factor (tried nearest first) of the reciprocal root of a row's operand for which the
 * reference's window row is ring_row, or NO_FACTOR. */
static uint32_t factor_places(const uint16_t *input, const float *row_angles,
			      const uint16_t *ring_row)
{
	const uint32_t root_bits = bits_of(
		(float)(1.0 /
			test_attention_square_root(
				(double)test_attention_window_operand(input))));
	for (uint32_t step = 0; step <= RSQRT_PLACES + RSQRT_PLACES; step++) {
		const uint32_t places =
			(step + 1u) >>
			1u; /* 0, then 1 above, 1 below, 2 above, ... */
		test_attention_window_row(input, kv_norm,
					  float_of((step & 1u) != 0 ?
							   root_bits + places :
							   root_bits - places),
					  row_angles, expected_row);
		if (memcmp(expected_row, ring_row, sizeof(expected_row)) == 0) {
			return places;
		}
	}
	return NO_FACTOR;
}

/* The ranges of one launch's arrays: step area, rotation, wkv's output and the ring of `sessions`. */
struct window_ranges {
	struct linear_range step, rotation, input, ring, kv_norm;
};

static bool window_ranges_set_aside(struct test_linear_card *card,
				    uint32_t rows, uint32_t sessions,
				    struct window_ranges *ranges)
{
	ranges->step = test_linear_card_work(
		card,
		STEP_BLOCK_BYTES_VALUE + (uint64_t)rows * STEP_ROW_BYTES_VALUE);
	ranges->rotation = test_linear_card_work(
		card, (uint64_t)rows * MODEL_ROTATION_ROW_BYTES);
	ranges->input = test_linear_card_work(
		card, (uint64_t)rows * MODEL_STATE_ROW_BYTES);
	ranges->ring = test_linear_card_work(
		card, (uint64_t)sessions * RING_SLOTS * MODEL_STATE_ROW_BYTES);
	ranges->kv_norm = test_linear_card_weights(card, sizeof(kv_norm));
	const struct step_block block = { 1, rows, 0 };
	return test_linear_card_has_room(card) &&
	       test_linear_card_put(card, ranges->step, 0, &block,
				    sizeof(block)) &&
	       test_linear_card_put(card, ranges->step, STEP_BLOCK_BYTES_VALUE,
				    step_rows,
				    (uint64_t)rows * STEP_ROW_BYTES_VALUE) &&
	       test_linear_card_put(card, ranges->input, 0, inputs,
				    (uint64_t)rows * MODEL_STATE_ROW_BYTES) &&
	       test_linear_card_put(card, ranges->ring, 0, ring_poison,
				    ranges->ring.byte_count) &&
	       test_linear_card_put(card, ranges->kv_norm, 0, kv_norm,
				    sizeof(kv_norm));
}

static bool window_prepare(struct test_linear_card *card,
			   const struct attention_state_kernels *kernels,
			   const struct window_ranges *ranges, uint32_t rows,
			   uint32_t sessions, uint32_t context, uint32_t use,
			   struct kernel_launch *launch_out)
{
	const struct attention_window_description description = {
		.step = ranges->step,
		.rotation = ranges->rotation,
		.window_input = ranges->input,
		.window_norm = ranges->kv_norm,
		.ring = ranges->ring,
		.row_count = rows,
		.session_capacity = sessions,
		.context_token_limit = context,
		.rotation_use = use
	};
	return test_linear_expect_status(
		attention_window_prepare(&description, card->arena_byte_count,
					 kernels, launch_out),
		STATUS_OK, "the window alone is prepared");
}

/* PREPARE: of each step and use, the ring of the rotation step and the prepare, then of the rotation and the window. */
static void prepare_runs(struct test_linear_card *card,
			 const struct attention_state_kernels *kernels)
{
	for (size_t step_index = 0;
	     step_index < sizeof(step_row_counts) / sizeof(step_row_counts[0]);
	     step_index++) {
		const uint32_t rows = step_row_counts[step_index];
		for (uint32_t theta = 0; theta < MODEL_THETAS; theta++) {
			struct window_ranges ranges;
			struct kernel_launch
				launches[ATTENTION_STATE_LAUNCH_COUNT + 1u];
			for (uint32_t row = 0; row < rows;
			     row++) { /* sessions alternating: from 0, across the wrap */
				const uint32_t place = row / SESSIONS;
				step_rows[row] = (struct step_row_record){
					(int32_t)(row % SESSIONS == 0 ?
							  place :
							  RING_SLOTS - 1u +
								  place),
					(int32_t)row, row % SESSIONS,
					STEP_ROW_LOGITS
				};
			}
			test_linear_context("card %" PRIu32
					    ", PREPARE, %" PRIu32
					    " rows, theta %" PRIu32,
					    card->card_index, rows, theta);
			test_linear_card_clear(card);
			bool is_run = window_ranges_set_aside(
				card, rows, SESSIONS, &ranges);
			const struct linear_range frequencies_range =
				test_linear_card_work(card,
						      sizeof(frequencies));
			const struct attention_rotation_description rotation = {
				ranges.step, frequencies_range, ranges.rotation,
				rows, 0
			};
			struct attention_state_description layer = {
				.step = ranges.step,
				.rotation = ranges.rotation,
				.query_rank = test_linear_card_work(
					card, (uint64_t)rows * QUERY_RANK *
						      sizeof(uint16_t)),
				.query_norm = test_linear_card_weights(
					card, sizeof(q_norm)),
				.query_normed = test_linear_card_work(
					card, (uint64_t)rows * QUERY_RANK *
						      sizeof(uint16_t)),
				.window_input = ranges.input,
				.window_norm = ranges.kv_norm,
				.ring = ranges.ring,
				.heads = test_linear_card_work(
					card,
					(uint64_t)rows * MODEL_HEADS_ROW_BYTES),
				.queries = test_linear_card_work(
					card,
					(uint64_t)rows * MODEL_HEADS_ROW_BYTES),
				.row_count = rows,
				.session_capacity = SESSIONS,
				.context_token_limit = PREPARE_CONTEXT,
				.theta = theta,
				.first_head = 0,
				.head_count = MODEL_HEADS_VALUE
			};
			is_run =
				is_run && test_linear_card_has_room(card) &&
				test_linear_card_put(card, frequencies_range, 0,
						     frequencies,
						     sizeof(frequencies)) &&
				test_linear_card_put(
					card, layer.query_rank, 0, query_ranks,
					(uint64_t)rows * QUERY_RANK *
						sizeof(uint16_t)) &&
				test_linear_card_put(card, layer.query_norm, 0,
						     q_norm, sizeof(q_norm)) &&
				test_linear_expect_status(
					attention_rotation_prepare(
						&rotation,
						card->arena_byte_count, kernels,
						&launches[0]),
					STATUS_OK,
					"the rotation step is prepared") &&
				test_linear_expect_status(
					attention_state_prepare(
						&layer, card->arena_byte_count,
						kernels, &launches[1]),
					STATUS_OK, "the layer is prepared") &&
				test_linear_card_load(card) &&
				test_linear_card_run(card, launches,
						     ROTATION_AND_PREPARE) &&
				test_linear_card_fetch(card) &&
				test_linear_card_get(card, ranges.ring, 0,
						     ring_prepared,
						     ranges.ring.byte_count) &&
				window_prepare(
					card, kernels, &ranges, rows, SESSIONS,
					PREPARE_CONTEXT,
					theta == 0 ?
						ATTENTION_ROTATION_WINDOW_ONLY :
						ATTENTION_ROTATION_POOLED,
					&launches[1]) &&
				test_linear_card_load(card) &&
				test_linear_card_run(card, launches,
						     ROTATION_AND_PREPARE) &&
				test_linear_card_fetch(card) &&
				test_linear_card_get(card, ranges.ring, 0,
						     ring_back,
						     ranges.ring.byte_count) &&
				test_linear_card_expect_untouched(
					card, ranges.input,
					"wkv's output is only read") &&
				test_linear_card_expect_weights_untouched(card);
			if (!is_run) {
				check_fail(PREPARE_CHECK,
					   "the case did not run", 0);
			} else if (memcmp(ring_back, ring_prepared,
					  ranges.ring.byte_count) != 0) {
				check_fail(
					PREPARE_CHECK,
					"the window alone wrote a ring other than the prepare's",
					rows);
			}
		}
	}
}

/* Every slot of one session's ring: the rows' slots the reference's (rows below `rows`), every other poison. */
static void ring_check(enum check check, uint32_t rows,
		       uint32_t *largest_places)
{
	static bool is_written[RING_SLOTS];
	memset(is_written, 0, sizeof(is_written));
	for (uint32_t row = 0; row < rows; row++) {
		const uint32_t slot =
			(uint32_t)step_rows[row].position % RING_SLOTS;
		const uint32_t places =
			factor_places(inputs[row], angles[row],
				      &ring_back[(size_t)slot * DIMS]);
		is_written[slot] = true;
		if (places == NO_FACTOR) {
			check_fail(
				check,
				"no factor near the reciprocal root gives the reference's ring row",
				row);
		} else if (places > *largest_places) {
			*largest_places = places;
		}
	}
	for (uint32_t slot = 0; slot < RING_SLOTS; slot++) {
		if (!is_written[slot] &&
		    memcmp(&ring_back[(size_t)slot * DIMS], ring_poison,
			   DIMS * sizeof(uint16_t)) != 0) {
			check_fail(check, "a slot no row writes was written",
				   slot);
		}
	}
}

/* CHUNK: rows of one session, the host's angles, in one launch of the window; then its first 8 rows alone. */
static void chunk_runs(struct test_linear_card *card,
		       const struct attention_state_kernels *kernels)
{
	static uint16_t first_rows[ROWS][DIMS];
	for (uint32_t row = 0; row < CHUNK_ROWS; row++) {
		step_rows[row] = (struct step_row_record){
			CHUNK_FIRST_POSITION + (int32_t)row, (int32_t)row, 0,
			STEP_ROW_LOGITS
		};
	}
	struct window_ranges ranges;
	struct kernel_launch launch;
	uint32_t largest_places = 0;
	const uint32_t failures_before = check_failures[CHUNK_CHECK];
	const uint16_t broken = (uint16_t)(inputs[BROKEN_ROW][0] ^ 1u);
	test_linear_context("card %" PRIu32 ", %s, %" PRIu32 " rows",
			    card->card_index, check_names[CHUNK_CHECK],
			    CHUNK_ROWS);
	test_linear_card_clear(card);
	const bool is_run =
		window_ranges_set_aside(card, CHUNK_ROWS, 1, &ranges) &&
		test_linear_card_put(card, ranges.rotation, 0, angles,
				     (uint64_t)CHUNK_ROWS *
					     MODEL_ROTATION_ROW_BYTES) &&
		(break_check != CHUNK_CHECK ||
		 test_linear_card_put(card, ranges.input,
				      (uint64_t)BROKEN_ROW *
					      MODEL_STATE_ROW_BYTES,
				      &broken, sizeof(broken))) &&
		window_prepare(card, kernels, &ranges, CHUNK_ROWS, 1,
			       CHUNK_CONTEXT, ATTENTION_ROTATION_WINDOW_ONLY,
			       &launch) &&
		test_linear_card_load(card) &&
		test_linear_card_run(card, &launch, 1) &&
		test_linear_card_fetch(card) &&
		test_linear_card_get(card, ranges.ring, 0, ring_back,
				     ranges.ring.byte_count) &&
		test_linear_card_expect_untouched(
			card, ranges.step, "the step area is only read") &&
		test_linear_card_expect_untouched(
			card, ranges.rotation, "the rotation is only read") &&
		test_linear_card_expect_weights_untouched(card);
	if (!is_run) {
		check_fail(CHUNK_CHECK, "the case did not run", 0);
		return;
	}
	ring_check(CHUNK_CHECK, CHUNK_ROWS, &largest_places);
	for (uint32_t row = 0; row < ROWS; row++) {
		memcpy(first_rows[row],
		       &ring_back[(size_t)((uint32_t)step_rows[row].position %
					   RING_SLOTS) *
				  DIMS],
		       sizeof(first_rows[row]));
	}
	struct window_ranges alone;
	test_linear_card_clear(card);
	const bool is_alone =
		window_ranges_set_aside(card, ROWS, 1, &alone) &&
		test_linear_card_put(card, alone.rotation, 0, angles,
				     (uint64_t)ROWS *
					     MODEL_ROTATION_ROW_BYTES) &&
		window_prepare(card, kernels, &alone, ROWS, 1, CHUNK_CONTEXT,
			       ATTENTION_ROTATION_WINDOW_ONLY, &launch) &&
		test_linear_card_load(card) &&
		test_linear_card_run(card, &launch, 1) &&
		test_linear_card_fetch(card) &&
		test_linear_card_get(card, alone.ring, 0, ring_back,
				     alone.ring.byte_count);
	for (uint32_t row = 0; is_alone && row < ROWS; row++) {
		const size_t slot =
			(uint32_t)step_rows[row].position % RING_SLOTS;
		if (memcmp(first_rows[row], &ring_back[slot * DIMS],
			   sizeof(first_rows[row])) != 0) {
			check_fail(
				CHUNK_CHECK,
				"a row of a launch of 8 is not its row of the chunk's launch",
				row);
		}
	}
	if (!is_alone) {
		check_fail(CHUNK_CHECK, "the launch of 8 rows did not run", 0);
	}
	(void)printf("card %" PRIu32 " (class %" PRIu32 "): %s: %" PRIu32
		     " rows %s, with a factor %" PRIu32
		     " places from the reciprocal root at most\n",
		     card->card_index, card->arithmetic_class,
		     check_names[CHUNK_CHECK], CHUNK_ROWS,
		     check_failures[CHUNK_CHECK] == failures_before ?
			     "the reference's bits" :
			     "NOT the reference's",
		     largest_places);
}

/* The kernels of the attention state with the draft's members, on an open card. */
static bool kernels_prepare(struct test_linear_card *card,
			    struct attention_state_kernels *state_kernels)
{
	struct hyper_kernels_record
		hyper_kernels; /* the families draft.h fills beside the attention state's */
	struct router_kernels_record router_kernels;
	struct attention_core_kernels core_kernels;
	struct draft_kernels_record draft_kernels;
	return test_linear_expect_status(
		       attention_state_kernels_prepare(
			       &backend_cuda_value, card->card, state_kernels),
		       STATUS_OK,
		       "the kernels of the attention state are prepared") &&
	       test_linear_expect_status(
		       draft_kernels_prepare(&backend_cuda_value, card->card,
					     card->arithmetic_class,
					     &hyper_kernels, &router_kernels,
					     state_kernels, &core_kernels,
					     &draft_kernels),
		       STATUS_OK, "the draft's kernels are prepared");
}

static void card_run(uint32_t card_index)
{
	static struct test_linear_card card;
	struct attention_state_kernels state_kernels;
	if (test_linear_card_open(&card, card_index, WORK_CAPACITY,
				  WEIGHTS_CAPACITY) &&
	    kernels_prepare(&card, &state_kernels)) {
		prepare_runs(&card, &state_kernels);
		chunk_runs(&card, &state_kernels);
	}
	test_linear_card_close(&card);
}

/* --trap: one row of position 0 whose normed first value no FP16 holds, in the window alone. */
static void trap_run(uint32_t card_index)
{
	static struct test_linear_card card;
	struct attention_state_kernels state_kernels;
	struct window_ranges ranges;
	struct kernel_launch launch;
	memset(inputs[0], 0, sizeof(inputs[0]));
	inputs[0][0] = TRAP_INPUT;
	kv_norm[0] = TRAP_WEIGHT;
	step_rows[0] = (struct step_row_record){ 0, 0, 0, STEP_ROW_LOGITS };
	test_linear_context("card %" PRIu32 ", the trap", card_index);
	if (test_linear_card_open(&card, card_index, WORK_CAPACITY,
				  WEIGHTS_CAPACITY) &&
	    kernels_prepare(&card, &state_kernels) &&
	    window_ranges_set_aside(&card, 1, 1, &ranges) &&
	    window_prepare(&card, &state_kernels, &ranges, 1, 1, CHUNK_CONTEXT,
			   ATTENTION_ROTATION_WINDOW_ONLY, &launch) &&
	    test_linear_card_load(&card)) {
		(void)test_linear_expect_status(
			backend_cuda_value.launch_run(card.card, &launch, 1),
			STATUS_OK, "the launch is queued");
		(void)test_linear_expect_status(
			backend_cuda_value.launch_wait(
				card.card, TEST_LINEAR_WAIT_LIMIT_NANOSECONDS),
			STATUS_DEVICE_ERROR, "the checked module traps");
	}
	test_linear_card_close(&card);
}

int main(int argument_count, char **arguments)
{
	uint32_t card_count = 0;
	char *card_end = NULL;
	const bool shows_a_trap = argument_count == FLAG_ARGUMENTS + 1 &&
				  strcmp(arguments[1], "--trap") == 0;
	const unsigned long trap_card =
		shows_a_trap ?
			strtoul(arguments[FLAG_ARGUMENTS], &card_end, DECIMAL) :
			0;
	if (argument_count == FLAG_ARGUMENTS + 1 &&
	    strcmp(arguments[1], "--break") == 0) {
		for (uint32_t check = 0; check < CHECK_COUNT; check++) {
			break_check =
				break_names[check][0] != '\0' &&
						strcmp(arguments[FLAG_ARGUMENTS],
						       break_names[check]) ==
							0 ?
					(int)check :
					break_check;
		}
	}
	if ((argument_count != 1 && break_check < 0 && !shows_a_trap) ||
	    (shows_a_trap &&
	     (card_end == arguments[FLAG_ARGUMENTS] || *card_end != '\0'))) {
		(void)fprintf(
			stderr,
			"usage: attention_state_window_card_test [--break window | --trap <card>]\n");
		return TEST_WRONG_USAGE;
	}
	(void)alarm(TEST_TIME_LIMIT_SECONDS);
	inputs_make();
	if (test_linear_expect_status(backend_cuda_value.card_count(&card_count),
				      STATUS_OK, "the cards are counted") &&
	    test_linear_expect(card_count > 0 &&
				       card_count <= BACKEND_MAXIMUM_CARDS,
			       "the machine has cards for the test")) {
		for (uint32_t card_index = 0;
		     !shows_a_trap && card_index < card_count; card_index++) {
			card_run(card_index);
		}
		if (shows_a_trap &&
		    test_linear_expect(trap_card < card_count,
				       "the card asked for is there") &&
		    CHECKING_HOOKS_ENABLED) {
			trap_run((uint32_t)trap_card);
		}
	}
	if (break_check >= 0) {
		bool only_the_break = check_failures[break_check] != 0;
		for (uint32_t check = 0; check < CHECK_COUNT; check++) {
			only_the_break = only_the_break &&
					 ((int)check == break_check ||
					  check_failures[check] == 0);
		}
		(void)printf(
			"attention_state_window_card_test %s: the break of %s\n",
			only_the_break ? "passed" : "FAILED",
			break_names[break_check]);
		return only_the_break ? TEST_PASSED : TEST_FAILED;
	}
	if (test_linear_failures() != 0) {
		(void)fprintf(
			stderr,
			"attention_state_window_card_test FAILED: %d failure lines\n",
			test_linear_failures());
		return TEST_FAILED;
	}
	(void)printf("attention_state_window_card_test passed on %" PRIu32
		     " cards%s\n",
		     card_count,
		     !shows_a_trap ?
			     "" :
		     CHECKING_HOOKS_ENABLED ?
			     ": the checked module trapped" :
			     ": the trap (the release modules check nothing)");
	return TEST_PASSED;
}
