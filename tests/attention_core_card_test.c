/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * attention_core_card_test: on the real cards, the two kernels of kernel_attention_core.cu, the fixed order's and
 * the reference order's, against the host reference of the core (attention_core_reference.h) on synthetic rows;
 * and, given a capture folder of the oracle, replayed in both orders on the reference's own inputs (plan 3.2, the
 * core's replay).
 *
 *     attention_core_card_test [--capture FOLDER] [--break window|word|order]
 *
 * It needs the cards free (`make card_tests`, or a batch): a card without room for its arena FAILS the test, it is
 * never skipped. On every card, each case runs in both orders, each held to the kernel form of its order, every row
 * of its step within TEST_ATTENTION_TOLERANCE of the row's rms and every value a finite number; FORM BITS, at least
 * SYNTHETIC_EQUAL_PER_THOUSAND of a thousand of a launch's values the bits of its order's form (the order's rounding
 * of P: the other order's kernel holds far fewer):
 *   - corners: 8 rows of 8 sessions, a context of 16,384 tokens and a pool of ratio 2, at positions 0, 1, 5, 126,
 *     127, 300, 1,025 and 16,383 (no pool entry yet, a window before position 0, fewer than 512 entries and all of
 *     them, 512 of 8,192 picked by a hash); queries of 8 magnitudes by head, 2^-6 to 2^8 times keys up to 2 (flat
 *     softmaxes, peaked ones that rescale, p below FP16's subnormals); keys holding zeros of both signs and FP16
 *     subnormals; sinks from -4 to 4.
 *   - consecutive: 8 rows of one session at positions 1,000 .. 1,007 with a pool of ratio 1: one step of M1c.
 *   - window: 8 rows of 8 sessions, the window alone (ratio 0), turned back by theta 0.
 *   - turn: window as at position 0 (one key, so the output is that key), sinks far below the scores, each pair
 *     turned back by an angle whose bfloat16 bits tell the kernel's contraction from another form
 *     (test_attention_turn_case): every value the reference's bits, in both orders.
 *   - partial: corners in a step of 3 rows: rows 3 .. 7 untouched, though their selections name entries far beyond
 *     every pool.
 *   - heads 24 .. 39 alone: the bits of the launch of every head; the other heads untouched.
 *   - ROW EXACTNESS: each row of corners alone in a launch of one row has its bits of the 8-row launch.
 *   - ORDERS APART: in every case but turn (a key a row, p = 1 in both) the two orders' outputs differ: the order
 *     picks the kernel.
 *   - BLOCK (dspark_v1/DESIGN.md K4): the draft's block core (attention_core_block_entry) of one session's rows at
 *     first positions b in {1, 5, 127, 128, 129, 4,095, 4,096, 4,100} and 1 to 8 rows, on the card's own kernels and,
 *     on a card of class 120, the native ones too: every row within TEST_ATTENTION_TOLERANCE of the host model of
 *     its keys (positions b - 128 to the block's last, none below 0) and FORM BITS; where b >= the rows, BLOCK TWIN:
 *     bit for bit the fixed order's causal core of a twin whose rows all stand at b - 1 and read the block's rows as
 *     pool entries of ratio 1 (the same keys in the same groups). Its outputs are not in the card's digest.
 *   Ring slots and pool entries no row reads hold not-a-numbers, so a key read out of place shows. The guards behind
 *   every range, the ranges the launch only reads (back as they went) and the sinks (in the weights region, by the
 *   card's digest) show that no launch writes what is not its own. Cards of one class give the same bits, and in
 *   the exact profile every card gives the first card's.
 * Shown failing (--break; the test must then fail): window, each step row's position one later on the card than
 * the reference's where the context allows (the window and the pool's reach move by one: plan 3.5's window break);
 * word, one FP16 word of the newest window row of each case's row 0 set to 65504 on the card alone; order, the
 * reference order's launches prepared in the fixed order (FORM BITS and ORDERS APART, and the reference order's
 * replay gate).
 * Replay (--capture FOLDER: a stepwise capture of tools/oracle/stepwise_capture.py with --frequencies): for each
 * layer of the capture points and each snapshot position the folder holds, the kernels fed the reference's own
 * inputs of that step: attn.q (bfloat16, widened to FP16 as the queries kernel widens it), the pool part of
 * attn.topk_idxs (ascending, -1 behind), snapshot.ring (slot q mod 128 into slot q mod 256), snapshot.pool of the
 * layer's source, attn.sink (the layer's sinks, float32 [64]: a record the capture must add), and cos and sin of the
 * layer's theta at the position. First, without a card: the reference's own turning back, attn.output_rotated to
 * attn.output, is the kernel's form (attention_core_records.h) on every value of dims 448 .. 511 and keeps the
 * others (TURN prints how many each form of test_attention_turn_back gives). The gate of each order: the engine's
 * row within twice the distance of the oracle's stand-in (attn.output) from the exact form on the same inputs, each
 * the relative rms over the row (ratio= the first over the second), and within TEST_ATTENTION_TOLERANCE of its
 * order's kernel form. The reference order computes as the stand-in does, so it is also held bit-close to it: its
 * row within REFERENCE_FROM_STAND_IN of the stand-in's rms, and REFERENCE_EQUAL_PER_THOUSAND of a thousand of its
 * values the stand-in's bits (from_stand_in= and equal=, printed for both orders; stand_in_form= is the stand-in's
 * own distance from the order's kernel form, kernel_form= the engine's). The window part of
 * attn.topk_idxs must list the positions the kernel derives. A record the replay needs that the folder lacks, or a
 * folder with nothing to replay, is a failure. Without --capture it says that it replayed nothing.
 */
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "anemone/attention_core.h"
#include "anemone/backend_seam.h"
#include "anemone/draft.h"
#include "anemone/pack_digest.h"
#include "attention_core_card.h"
#include "attention_core_reference.h"
#include "capture_reader.h"
#include "linear_card_fixture.h"
#include "linear_reference.h"

#define COUNT_OF(table) (sizeof(table) / sizeof((table)[0]))
#define TEST_TIME_LIMIT_SECONDS 3600u
#define MEBIBYTE 1048576ull
#define WORK_CAPACITY (160ull * MEBIBYTE)
#define WEIGHTS_CAPACITY 4096ull
#define ROWS LINEAR_MAXIMUM_ROWS
#define HEADS MODEL_HEADS_VALUE
#define DIMS MODEL_HEAD_DIMS
#define ROW_VALUES TEST_ATTENTION_ROW_VALUES
#define RING_SLOTS MODEL_RING_SLOTS
#define WINDOW MODEL_WINDOW
#define SELECTED MODEL_SELECTED
#define ROTATION_FLOATS TEST_ATTENTION_ROTATION_FLOATS
#define USE_FLOATS TEST_ATTENTION_USE_FLOATS
#define ORDERS TEST_ATTENTION_ORDERS
#define FP16_SUBNORMAL_MASK 0x03FFu
#define FP16_SIGN 0x8000u
#define VALUE_STEPS 2048u                       /* a key's value: (k - 1024) / 512, -2 .. 2 */
#define VALUE_CENTRE 1024
#define VALUE_SCALE 512.0
#define CLASS_COUNT 16u                         /* of a key's values: one class zero, one subnormal */
#define CLASS_ZERO 0u
#define CLASS_SUBNORMAL 1u
#define QUERY_MAGNITUDES 8u                     /* head h: query_scales[h % 8] */
#define SINK_SCALE 2.0                          /* sinks from -4 to 4 */
#define ROTATION_STEPS 2097152u                 /* cos and sin: (k - 2^20) / 2^20, -1 .. 1, exact in float32 */
#define ROTATION_CENTRE 1048576
#define ROTATION_SCALE 1048576.0
#define HASH_CLASS_SHIFT 32u                    /* the bits of a hash that choose a value's class */
#define STREAM_KEYS 0x11u
#define STREAM_QUERIES 0x12u
#define STREAM_SINKS 0x13u
#define STREAM_ROTATION 0x14u
#define STREAM_SELECTION 0x15u
#define STREAM_POOL 0x10000u                    /* added to a session's stream for its pool */
#define PARTIAL_ROWS 3u
#define SOME_FIRST_HEAD 24u
#define SOME_HEAD_COUNT 16u
#define STAND_IN_FACTOR 2.0                     /* plan 3.2: twice the stand-in's distance */
/* The reference order bit-close to the stand-in: of the stand-in's rms, and the share of its bits. Batch
 * core_reference (2026-10-18, 42 points on each of four cards): the reference order 9e-6 to 2.4e-4 and 99.63 to
 * 99.99% (its scores, sums and exponentials are float32 in another order than the stand-in's, which moves a p across a
 * rounding of bfloat16 now and then); the fixed order 1.14e-3 to 1.72e-3 and 75 to 87%. */
#define REFERENCE_FROM_STAND_IN 5.0e-4
#define REFERENCE_EQUAL_PER_THOUSAND 990u
#define PER_THOUSAND 1000u
/* FORM BITS. Batch core_reference (2026-10-18, both classes): 98.86 to 99.35% in either order; the fixed kernel in
 * the reference order's place 75.8 to 88.1%. */
#define SYNTHETIC_EQUAL_PER_THOUSAND 970u
#define REPLAY_CONTEXT MODEL_CONTEXT_WITHOUT_CANDIDATES
#define REFERENCE_RING_SLOTS \
	MODEL_WINDOW /* the reference's ring: slot q mod 128 (REF:718) */
#if defined(LINEAR_CLASS_FORMS)
#define CLASSES_DIFFER true                     /* each class its own forms: the bits of the first card of its class */
#else
#define CLASSES_DIFFER false                    /* the exact profile: every card the bits of the first */
#endif
#define NAME_CAPACITY 128u
#define EXIT_USAGE 2
#define FIRST_POOLED_LAYER 2u                   /* CFG compress_ratios: 0 for layers 0 and 1, 2 to layer 19 */
#define FIRST_DIRECT_LAYER 20u                  /* then 1 */
#define THETA_0_TABLE "frequencies.theta_10000.cos_sin_below_16384"
#define THETA_1_TABLE "frequencies.theta_160000.cos_sin_below_16384"
#define STREAM_TURN 0x16u
#define TURN_STREAM_SHIFT 32u                   /* a pair's stream: STREAM_TURN above, its row's pair below */
#define TURN_SINK (-200.0f)                     /* exp(-200 - 0) is no float32: the denominator is 1 */
#define SQUARE_ROOT_STEPS 64u                   /* Newton's steps: from 1 down to 1e-9 and to the last place */
#define HALF 0.5
#define BLOCK_CONTEXT 8192u                     /* holds the last block, 4,100 + 7 */

enum case_kind { CASE_CORNERS, CASE_CONSECUTIVE, CASE_WINDOW, CASE_TURN, CASE_COUNT };

struct case_spec {
    const char *name;
    uint32_t session_count;
    uint32_t context_token_limit;
    uint32_t pool_ratio;
    bool turn;                                  /* the turn case: rows of one key, every value the reference's bits */
    int32_t positions[ROWS];
    uint32_t sessions[ROWS];
};

static const struct case_spec case_specs[CASE_COUNT] = {
	[CASE_CORNERS] = { "corners",
			   8,
			   16384,
			   MODEL_POOLED_RATIO,
			   false,
			   { 0, 1, 5, 126, 127, 300, 1025, 16383 },
			   { 0, 1, 2, 3, 4, 5, 6, 7 } },
	[CASE_CONSECUTIVE] = { "consecutive",
			       1,
			       2048,
			       MODEL_DIRECT_RATIO,
			       false,
			       { 1000, 1001, 1002, 1003, 1004, 1005, 1006,
				 1007 },
			       { 0, 0, 0, 0, 0, 0, 0, 0 } },
	[CASE_WINDOW] = { "window",
			  8,
			  2048,
			  0,
			  false,
			  { 0, 3, 64, 127, 128, 200, 255, 2047 },
			  { 0, 1, 2, 3, 4, 5, 6, 7 } },
	[CASE_TURN] = { "turn",
			8,
			2048,
			0,
			true,
			{ 0, 0, 0, 0, 0, 0, 0, 0 },
			{ 0, 1, 2, 3, 4, 5, 6, 7 } },
};

static const int32_t block_firsts[] = {1, 5, 127, 128, 129, 4095, 4096, 4100};
static bool break_window = false;
static bool break_word = false;
static bool break_order = false;
static const char *const order_names[ORDERS] = {"fixed", "reference"};

/* The order a launch of order `order` is prepared in: the fixed one for both under --break order. */
static enum linear_order_code prepared_order(uint32_t order)
{
	return break_order ? LINEAR_ORDER_FIXED : (enum linear_order_code)order;
}

/* A plan of rows (test_attention_plan_of_rows) with the breaks the test was given. */
static void plan_rows(struct test_attention_plan *plan, const char *name, uint32_t first_row, uint32_t row_count)
{
    test_attention_plan_of_rows(plan, name, first_row, row_count);
    plan->window_one_later = break_window;
    plan->largest_word = break_word;
}

/* ---- the synthetic rows ---- */

static uint16_t key_value(uint64_t stream, uint64_t index, uint32_t dim)
{
    const uint64_t hash = test_linear_hash(stream, index * DIMS + dim);
    const uint64_t value_class = (hash >> HASH_CLASS_SHIFT) % CLASS_COUNT;
    if (value_class == CLASS_ZERO) {
        return (uint16_t)(hash & FP16_SIGN);
    }
    if (value_class == CLASS_SUBNORMAL) {
        return (uint16_t)((hash & FP16_SIGN) | ((hash % FP16_SUBNORMAL_MASK) + 1u));
    }
    return test_attention_half_bits(((double)(hash % VALUE_STEPS) - VALUE_CENTRE) / VALUE_SCALE);
}

static double scaled_hash(uint64_t stream, uint64_t position, double scale)
{
    return ((double)(test_linear_hash(stream, position) % VALUE_STEPS) - VALUE_CENTRE) / VALUE_SCALE * scale;
}

static const double query_scales[QUERY_MAGNITUDES] = {0x1p-6, 0x1p-4, 0x1p-2, 1.0, 0x1p2, 0x1p4, 0x1p6, 0x1p8};

static int compare_entries(const void *first, const void *second)
{
    const int32_t left = *(const int32_t *)first, right = *(const int32_t *)second;
    return (left > right) - (left < right);
}

/* min(512, visible) distinct entries below visible, by a hash, ascending; -1 behind. */
static bool selection_make(int32_t *selection, uint32_t visible, uint64_t stream)
{
    int32_t *const entries = malloc((visible == 0 ? 1u : visible) * sizeof(int32_t));
    const uint32_t selected = visible < SELECTED ? visible : SELECTED;
    if (entries == NULL) {
        test_linear_fail("the entries of a selection are not held");
        return false;
    }
    for (uint32_t entry = 0; entry < visible; entry++) {
        entries[entry] = (int32_t)entry;
    }
    for (uint32_t place = 0; place < selected && visible > SELECTED; place++) {
        const uint32_t other = place + (uint32_t)(test_linear_hash(stream, place) % (visible - place));
        const int32_t kept = entries[place];
        entries[place] = entries[other];
        entries[other] = kept;
    }
    qsort(entries, selected, sizeof(int32_t), compare_entries);
    for (uint32_t place = 0; place < SELECTED; place++) {
        selection[place] = place < selected ? entries[place] : -1;
    }
    free(entries);
    return true;
}

/* The turn case over the rows case_make made: zero queries and sinks far below, so a row's one key is its output;
 * pair i of row r (dims 448 + 2 i, 449 + 2 i of the key, its angle of use 0) tells the kernel's form from the other
 * contraction (i / 2 even) or from products rounded (odd) in part i % 2. */
static bool turn_case_make(struct test_attention_case *data)
{
    memset(data->queries, 0, (size_t)ROWS * ROW_VALUES * sizeof(uint16_t));
    for (uint32_t head = 0; head < HEADS; head++) {
        data->sinks[head] = TURN_SINK;
    }
    for (uint32_t row = 0; row < ROWS; row++) {
	    const struct step_row_record *const step_row =
		    &data->step_rows[row];
	    uint16_t *const key =
		    data->ring + ((size_t)step_row->session_slot * RING_SLOTS +
				  (uint32_t)step_row->position % RING_SLOTS) *
					 DIMS;
	    for (uint32_t pair = 0; pair < MODEL_FREQUENCIES; pair++) {
		    const enum test_attention_turn_form form =
			    pair / MODEL_PAIR_VALUES % MODEL_PAIR_VALUES == 0 ?
				    TEST_ATTENTION_TURN_OTHER :
				    TEST_ATTENTION_TURN_PRODUCTS;
		    float values[MODEL_PAIR_VALUES];
		    float *const angle = data->rotation +
					 (size_t)row * ROTATION_FLOATS +
					 data->rotation_use * USE_FLOATS +
					 (size_t)pair * MODEL_PAIR_VALUES;
		    if (!test_attention_turn_case(
				((uint64_t)STREAM_TURN << TURN_STREAM_SHIFT) +
					(uint64_t)row * MODEL_FREQUENCIES +
					pair,
				pair % MODEL_PAIR_VALUES, form, values,
				angle)) {
			    test_linear_fail("row %" PRIu32 ", pair %" PRIu32
					     ": no angle tells the forms apart",
					     row, pair);
			    return false;
		    }
		    for (uint32_t value = 0; value < MODEL_PAIR_VALUES;
			 value++) {
			    key[MODEL_FIRST_ROTATED_DIM +
				pair * MODEL_PAIR_VALUES + value] =
				    test_attention_half_bits(
					    (double)values[value]);
		    }
	    }
    }
    return true;
}

/* A case of case_specs: its rows, the keys each row reads, and the reference of each. */
static bool case_make(const struct case_spec *spec, struct test_attention_case *data)
{
    memset(data, 0, sizeof(*data));
    data->name = spec->name;
    data->session_count = spec->session_count;
    data->context_token_limit = spec->context_token_limit;
    data->pool_ratio = spec->pool_ratio;
    data->rotation_use = spec->pool_ratio != 0 ? 1u : 0u;
    data->exact = spec->turn;
    data->pool_entries = spec->pool_ratio == 0 ? 0 : spec->context_token_limit / spec->pool_ratio;
    if (!test_attention_case_hold(data)) {
        return false;
    }
    for (uint32_t head = 0; head < HEADS; head++) {
        data->sinks[head] = (float)scaled_hash(STREAM_SINKS, head, SINK_SCALE);
    }
    for (uint32_t row = 0; row < ROWS; row++) {
        const int32_t position = spec->positions[row];
        const uint32_t session = spec->sessions[row];
        const uint32_t visible = spec->pool_ratio == 0 ? 0 : (uint32_t)(position + 1) / spec->pool_ratio;
	data->step_rows[row] = (struct step_row_record){ position, 0, session,
							 STEP_ROW_LOGITS };
	for (uint32_t index = 0; index < ROW_VALUES; index++) {
            data->queries[(size_t)row * ROW_VALUES + index] = test_attention_half_bits(scaled_hash(
                STREAM_QUERIES, (uint64_t)row * ROW_VALUES + index, query_scales[index / DIMS % QUERY_MAGNITUDES]));
        }
        for (uint32_t index = 0; index < ROTATION_FLOATS; index++) {
            const uint64_t hash = test_linear_hash(STREAM_ROTATION, (uint64_t)row * ROTATION_FLOATS + index);
            data->rotation[(size_t)row * ROTATION_FLOATS + index] =
                (float)(((double)(hash % ROTATION_STEPS) - ROTATION_CENTRE) / ROTATION_SCALE);
        }
        for (int32_t key_position = position - (int32_t)WINDOW + 1; key_position <= position; key_position++) {
            for (uint32_t dim = 0; key_position >= 0 && dim < DIMS; dim++) {
                data->ring[((uint64_t)session * RING_SLOTS + (uint32_t)key_position % RING_SLOTS) * DIMS + dim] =
                    key_value(STREAM_KEYS + session, (uint64_t)key_position, dim);
            }
        }
        for (uint64_t entry = 0; data->pool != NULL && entry < visible; entry++) {
            for (uint32_t dim = 0; dim < DIMS; dim++) {
                data->pool[((uint64_t)session * data->pool_entries + entry) * DIMS + dim] =
                    key_value(STREAM_KEYS + STREAM_POOL + session, entry, dim);
            }
        }
        if (data->pool != NULL &&
            !selection_make(data->selection + (size_t)row * SELECTED, visible, STREAM_SELECTION + row)) {
            return false;
        }
    }
    return (!spec->turn || turn_case_make(data)) && test_attention_case_references(data);
}

/* ---- what came back ---- */

static double square_root(double value)
{
    double root = value > 1.0 ? value : 1.0;
    for (uint32_t step = 0; step < SQUARE_ROOT_STEPS; step++) {
        root = (root + value / root) * HALF;
    }
    return root;
}

/* Rows of the step within the tolerance of the order's reference in the plan's heads, the other heads and rows
 * untouched. */
static void expect_rows(uint32_t card_index, const struct test_attention_case *data,
                        const struct test_attention_plan *plan, uint32_t order, const uint16_t *output)
{
    double worst = 0.0;
    uint64_t equal_count = 0;
    uint64_t value_count = 0;
    for (uint32_t row = 0; row < plan->program_rows; row++) {
        const uint16_t *const row_output = output + (size_t)row * ROW_VALUES;
        const uint64_t first = (uint64_t)plan->first_head * DIMS;
        const uint64_t count = (uint64_t)plan->head_count * DIMS;
        struct test_attention_distance distance;
        memset(&distance, 0, sizeof(distance));
        if (row >= plan->step_rows) {
            (void)test_linear_expect(test_attention_is_untouched(row_output, ROW_VALUES),
                                     "a row behind the step is untouched");
            continue;
        }
        test_attention_distance_add(&distance, row_output + first,
                                    data->reference[order] + (size_t)plan->source_rows[row] * ROW_VALUES + first,
                                    count);
        if (!test_attention_core_within(&distance, TEST_ATTENTION_TOLERANCE)) {
            test_linear_fail("row %" PRIu32 ": %.3e of its rms from the reference, %" PRIu64 " values not finite",
                             row, square_root(test_attention_relative_squared(&distance)),
                             distance.not_finite_count);
        }
        if (data->exact && distance.equal_count != distance.value_count) {
            test_linear_fail("row %" PRIu32 ": %" PRIu64 " values not the reference's bits", row,
                             distance.value_count - distance.equal_count);
        }
        (void)test_linear_expect(test_attention_is_untouched(row_output, first) &&
                                     test_attention_is_untouched(row_output + first + count,
                                                                 ROW_VALUES - first - count),
                                 "the heads outside the launch are untouched");
        const double relative = test_attention_relative_squared(&distance);
        worst = relative > worst ? relative : worst;
        equal_count += distance.equal_count;
        value_count += distance.value_count;
    }
    (void)printf("card %" PRIu32 " %s, %s, %s order: worst row %.2e of its rms, %.4f%% of values the reference's "
                 "rounded\n", card_index, data->name, plan->name, order_names[order], square_root(worst),
                 test_linear_percent(equal_count, value_count));
    (void)test_linear_expect(fflush(stdout) == 0, "the line is written before any failure line");
    if (!data->exact && equal_count * PER_THOUSAND < value_count * SYNTHETIC_EQUAL_PER_THOUSAND) {
        test_linear_fail("FORM BITS: fewer than %u of a thousand values the bits of the %s order's form",
                         SYNTHETIC_EQUAL_PER_THOUSAND, order_names[order]);
    }
}

static void expect_same_bits(const uint16_t *values, const uint16_t *expected, uint64_t count, const char *what)
{
    (void)test_linear_expect(memcmp(values, expected, count * sizeof(uint16_t)) == 0, what);
}

/* ---- the synthetic cases ---- */

struct test_card {
    struct test_linear_card card;
    struct attention_core_kernels kernels;
    bool is_open;
    uint64_t digest;                            /* of every output of the synthetic cases, in both orders */
};

static void digest_add(struct test_card *card, const uint16_t *output, uint64_t value_count)
{
    uint64_t digest = 0;
    if (test_linear_expect_status(
		pack_digest_of_host_bytes(
			output, value_count * sizeof(uint16_t), &digest),
		STATUS_OK, "the output is digested")) {
	    card->digest += digest;
    }
}

/* Every launch of one case on one card in one order. output and alone: [ROWS][64][512] each. False where the launch
 * of every row did not run. */
static bool case_run(struct test_card *card, const struct test_attention_case *data, uint32_t order,
                     uint16_t *output, uint16_t *alone)
{
    struct test_attention_plan plan;
    plan_rows(&plan, "every row", 0, ROWS);
    if (!test_attention_plan_run(&card->card, &card->kernels, data, &plan, prepared_order(order), output)) {
        return false;
    }
    expect_rows(card->card.card_index, data, &plan, order, output);
    digest_add(card, output, (uint64_t)ROWS * ROW_VALUES);
    if (data->pool_ratio != MODEL_POOLED_RATIO || data->session_count != ROWS) {
	    return true; /* what follows is of corners */
    }
    plan_rows(&plan, "a step of 3 rows", 0, ROWS);
    plan.step_rows = PARTIAL_ROWS;
    plan.far_selections_behind = true;
    if (test_attention_plan_run(&card->card, &card->kernels, data, &plan, prepared_order(order), alone)) {
        expect_rows(card->card.card_index, data, &plan, order, alone);
        expect_same_bits(alone, output, (uint64_t)PARTIAL_ROWS * ROW_VALUES, "the rows of the step keep their bits");
    }
    plan_rows(&plan, "heads 24 .. 39", 0, ROWS);
    plan.first_head = SOME_FIRST_HEAD;
    plan.head_count = SOME_HEAD_COUNT;
    if (test_attention_plan_run(&card->card, &card->kernels, data, &plan, prepared_order(order), alone)) {
        expect_rows(card->card.card_index, data, &plan, order, alone);
        for (uint32_t row = 0; row < ROWS; row++) {
            const size_t first = (size_t)row * ROW_VALUES + (size_t)SOME_FIRST_HEAD * DIMS;
            expect_same_bits(alone + first, output + first, (uint64_t)SOME_HEAD_COUNT * DIMS,
                             "heads launched alone keep their bits");
        }
    }
    for (uint32_t row = 0; row < ROWS; row++) {
        plan_rows(&plan, "a row alone", row, 1u);
        if (test_attention_plan_run(&card->card, &card->kernels, data, &plan, prepared_order(order), alone)) {
            expect_same_bits(alone, output + (size_t)row * ROW_VALUES, ROW_VALUES,
                             "ROW EXACTNESS: a row alone has its bits of the launch of 8");
        }
    }
    return true;
}

/* ORDERS APART: the outputs of every row in the two orders, of a case whose rows read more than one key. */
static void expect_orders_apart(uint32_t card_index, const struct test_attention_case *data,
                                uint16_t *const outputs[ORDERS])
{
    uint64_t differing = 0;
    for (size_t index = 0; index < (size_t)ROWS * ROW_VALUES; index++) {
	    const bool apart = outputs[LINEAR_ORDER_FIXED][index] !=
			       outputs[LINEAR_ORDER_REFERENCE][index];
	    differing += apart ? 1u : 0u;
    }
    test_linear_context("card %" PRIu32 ", %s, both orders", card_index, data->name);
    (void)printf("card %" PRIu32 " %s: ORDERS APART in %" PRIu64 " of %" PRIu64 " values\n", card_index, data->name,
                 differing, (uint64_t)ROWS * ROW_VALUES);
    (void)test_linear_expect(fflush(stdout) == 0, "the line is written before any failure line");
    (void)test_linear_expect(differing != 0, "ORDERS APART: the reference order's kernel gives other bits");
}

/* ---- the block core ---- */

/* The block case of first position `first`: one session, rows `first` .. on, the keys of positions first - 128 to the
 * eighth row's in the ring, queries, rotation (use 0) and sinks as case_make makes them; the reference of each of the
 * first `rows` rows in the fixed order's form, over the block's keys. */
static bool block_case_make(int32_t first, uint32_t rows, struct test_attention_case *data)
{
    memset(data, 0, sizeof(*data));
    data->name = "block";
    data->session_count = 1;
    data->context_token_limit = BLOCK_CONTEXT;
    if (!test_attention_case_hold(data)) {
        return false;
    }
    for (uint32_t head = 0; head < HEADS; head++) {
        data->sinks[head] = (float)scaled_hash(STREAM_SINKS, head, SINK_SCALE);
    }
    for (uint32_t row = 0; row < ROWS; row++) {
	    data->step_rows[row] =
		    (struct step_row_record){ first + (int32_t)row, 0, 0,
					      STEP_ROW_LOGITS };
	    for (uint32_t index = 0; index < ROW_VALUES; index++) {
		    data->queries[(size_t)row * ROW_VALUES + index] =
			    test_attention_half_bits(scaled_hash(
				    STREAM_QUERIES,
				    (uint64_t)row * ROW_VALUES + index,
				    query_scales[index / DIMS %
						 QUERY_MAGNITUDES]));
	    }
        for (uint32_t index = 0; index < ROTATION_FLOATS; index++) {
            const uint64_t hash = test_linear_hash(STREAM_ROTATION, (uint64_t)row * ROTATION_FLOATS + index);
            data->rotation[(size_t)row * ROTATION_FLOATS + index] =
                (float)(((double)(hash % ROTATION_STEPS) - ROTATION_CENTRE) / ROTATION_SCALE);
        }
    }
    const int32_t last = first + (int32_t)ROWS - 1;
    for (int32_t position = first - (int32_t)WINDOW; position <= last; position++) {
        for (uint32_t dim = 0; position >= 0 && dim < DIMS; dim++) {
            data->ring[(size_t)((uint32_t)position % RING_SLOTS) * DIMS + dim] =
                key_value(STREAM_KEYS, (uint64_t)position, dim);
        }
    }
    for (uint32_t row = 0; row < rows; row++) {
        struct test_attention_row reference;
        memset(&reference, 0, sizeof(reference));
        reference.queries = data->queries + (size_t)row * ROW_VALUES;
        for (uint32_t key = 0; key <= WINDOW + rows - 1u; key++) {
            const int32_t position = first - (int32_t)WINDOW + (int32_t)key;
            reference.keys[key] = position < 0 ? NULL : data->ring + (size_t)((uint32_t)position % RING_SLOTS) * DIMS;
        }
        reference.sinks = data->sinks;
        reference.rotation = data->rotation + (size_t)row * ROTATION_FLOATS;
        reference.head_count = HEADS;
        reference.form = TEST_ATTENTION_KERNEL;
	if (!test_linear_expect_status(
		    test_attention_core_row(
			    &reference, data->reference[LINEAR_ORDER_FIXED] +
						(size_t)row * ROW_VALUES),
		    STATUS_OK, "the reference computes the block's row")) {
		return false;
	}
    }
    return true;
}

/* The causal twin of a block case: every row at first - 1, reading the block's rows as pool entries 0 .. rows - 1 of
 * ratio 1, its rotation's use 1 the block's use 0: the same keys in the same groups of the fixed order. */
static bool block_twin_make(const struct test_attention_case *block, int32_t first, uint32_t rows,
                            struct test_attention_case *twin)
{
    memset(twin, 0, sizeof(*twin));
    twin->name = "block's twin";
    twin->session_count = 1;
    twin->context_token_limit = BLOCK_CONTEXT;
    twin->pool_ratio = MODEL_DIRECT_RATIO;
    twin->rotation_use = 1;
    twin->pool_entries = BLOCK_CONTEXT;
    if (!test_attention_case_hold(twin)) {
        return false;
    }
    memcpy(twin->sinks, block->sinks, sizeof(twin->sinks));
    memcpy(twin->queries, block->queries, (size_t)ROWS * ROW_VALUES * sizeof(uint16_t));
    memcpy(twin->ring, block->ring, (size_t)RING_SLOTS * DIMS * sizeof(uint16_t));
    memcpy(twin->rotation, block->rotation, (size_t)ROWS * ROTATION_FLOATS * sizeof(float));
    for (uint32_t row = 0; row < ROWS; row++) {
	    twin->step_rows[row] = (struct step_row_record){ first - 1, 0, 0,
							     STEP_ROW_LOGITS };
	    memcpy(twin->rotation + (size_t)row * ROTATION_FLOATS + USE_FLOATS,
		   block->rotation + (size_t)row * ROTATION_FLOATS,
		   USE_FLOATS * sizeof(float));
	    for (uint32_t place = 0; place < rows; place++) {
		    twin->selection[(size_t)row * SELECTED + place] =
			    (int32_t)place;
	    }
    }
    for (uint32_t entry = 0; entry < rows; entry++) {
        memcpy(twin->pool + (size_t)entry * DIMS,
               block->ring + (size_t)((uint32_t)(first + (int32_t)entry) % RING_SLOTS) * DIMS, DIMS * sizeof(uint16_t));
    }
    return true;
}

/* Every block case on every card, on the card's own kernels and, on a card of class 120, the native ones. */
static void block_cases(struct test_card *cards, uint32_t card_count)
{
	static struct attention_core_kernels kernel_sets[BACKEND_MAXIMUM_CARDS]
							[ORDERS];
	static uint32_t set_classes[BACKEND_MAXIMUM_CARDS][ORDERS];
	static uint32_t set_counts[BACKEND_MAXIMUM_CARDS];
	uint16_t *const output =
		malloc((size_t)ROWS * ROW_VALUES * sizeof(uint16_t));
	uint16_t *const twin_output =
		malloc((size_t)ROWS * ROW_VALUES * sizeof(uint16_t));
	uint32_t block_count = 0, twin_count = 0;
	for (uint32_t card_index = 0; card_index < card_count; card_index++) {
		struct test_card *const card = &cards[card_index];
		struct hyper_kernels_record
			hyper_kernels; /* the families draft.h fills beside the core's */
		struct router_kernels_record router_kernels;
		struct attention_state_kernels state_kernels;
		struct draft_kernels_record draft_kernels;
		const bool is_class_120 = card->card.arithmetic_class ==
					  ARITHMETIC_CLASS_SM_120;
		set_counts[card_index] = 0;
		set_classes[card_index][0] = card->card.arithmetic_class;
		set_classes[card_index][1] = ARITHMETIC_CLASS_SM_120_NATIVE;
		kernel_sets[card_index][0] = card->kernels;
		test_linear_context("card %" PRIu32
				    ", the block core's kernels",
				    card_index);
		for (uint32_t set = 0;
		     card->is_open && set < (is_class_120 ? ORDERS : 1u);
		     set++) {
			struct attention_core_kernels *const set_kernels =
				&kernel_sets[card_index][set];
			if ((set == 0 ||
			     test_linear_expect_status(
				     attention_core_kernels_prepare(
					     &backend_cuda_value,
					     card->card.card,
					     set_classes[card_index][set],
					     set_kernels),
				     STATUS_OK,
				     "the native kernels are prepared")) &&
			    test_linear_expect_status(
				    draft_kernels_prepare(
					    &backend_cuda_value,
					    card->card.card,
					    set_classes[card_index][set],
					    &hyper_kernels, &router_kernels,
					    &state_kernels, set_kernels,
					    &draft_kernels),
				    STATUS_OK,
				    "the draft's kernels are prepared")) {
				set_counts[card_index]++;
			}
		}
    }
    for (size_t first_index = 0; output != NULL && twin_output != NULL && first_index < COUNT_OF(block_firsts);
         first_index++) {
        for (uint32_t rows = 1; rows <= ROWS; rows++) {
            const int32_t first = block_firsts[first_index];
            const bool has_twin = (uint32_t)first >= rows;
            struct test_attention_case data, twin;
            const bool is_made = block_case_make(first, rows, &data) && (!has_twin || block_twin_make(&data, first,
                                                                                                    rows, &twin));
            for (uint32_t card_index = 0; is_made && card_index < card_count; card_index++) {
                for (uint32_t set = 0; set < set_counts[card_index]; set++) {
                    struct test_card *const card = &cards[card_index];
                    struct test_attention_plan plan;
                    plan_rows(&plan, set == 0 ? "the block" : "the block, class 120 native", 0, rows);
                    plan.block_window = true;
		    if (test_attention_plan_run(
				&card->card, &kernel_sets[card_index][set],
				&data, &plan, LINEAR_ORDER_FIXED, output)) {
			    expect_rows(card_index, &data, &plan,
					LINEAR_ORDER_FIXED, output);
			    block_count++;
			    plan_rows(&plan, "the block's twin", 0, rows);
			    if (has_twin &&
				test_attention_plan_run(
					&card->card,
					&kernel_sets[card_index][set], &twin,
					&plan, LINEAR_ORDER_FIXED,
					twin_output)) {
				    expect_same_bits(
					    output, twin_output,
					    (uint64_t)rows * ROW_VALUES,
					    "BLOCK TWIN: the block core is the causal core of the same keys");
				    twin_count++;
			    }
		    }
		}
            }
            if (has_twin) {
                test_attention_case_free(&twin);
            }
            test_attention_case_free(&data);
        }
    }
    (void)test_linear_expect(output != NULL && twin_output != NULL, "the block's outputs are held");
    free(output);
    free(twin_output);
    (void)printf("block core: %" PRIu32 " launches against the host model, %" PRIu32 " bit for bit their causal "
                 "twins\n", block_count, twin_count);
}

/* ---- the replay ---- */

static const uint32_t replay_layers[] = {0, 1, 2, 3, 8, 14, 15, 16, 20, 21, 24, 31, 32, 39};  /* the capture points */
static const uint32_t replay_positions[] = {129, 1026, 2047};                             /* the snapshot positions */
static const uint32_t kv_source_layers[] = {2, 8, 14, 20};                               /* CFG kv_source_layers */

/* The pool ratio of a layer (CFG compress_ratios) and the layer whose pool it reads. */
static uint32_t ratio_of_layer(uint32_t layer)
{
	return layer < FIRST_POOLED_LAYER ? 0u :
	       layer < FIRST_DIRECT_LAYER ? MODEL_POOLED_RATIO :
					    MODEL_DIRECT_RATIO;
}

static uint32_t source_of_layer(uint32_t layer)
{
    uint32_t source = kv_source_layers[0];
    for (size_t index = 0; index < COUNT_OF(kv_source_layers); index++) {
        source = kv_source_layers[index] <= layer ? kv_source_layers[index] : source;
    }
    return source;
}

static bool capture_read(const struct test_capture *capture, const char *name, enum test_capture_type type,
                         const uint64_t *dimensions, uint32_t dimension_count, uint64_t first_element,
                         uint64_t element_count, void *destination)
{
	return test_linear_expect_status(
		test_capture_read(
			capture, name, type, dimensions, dimension_count,
			first_element, element_count, destination,
			element_count * test_capture_element_bytes(type)),
		STATUS_OK, name);
}

/* FP16 values held in float32 (the u16 arm's state) into FP16 bits: false where a value is no FP16 number. */
static bool halves_of_floats(const float *values, uint64_t count, uint16_t *halves)
{
    for (uint64_t index = 0; index < count; index++) {
        halves[index] = test_attention_half_bits((double)values[index]);
        if (test_attention_core_half_value(halves[index]) != (double)values[index]) {
            test_linear_fail("value %" PRIu64 " of a state, %.9g, is no FP16 number", index, (double)values[index]);
            return false;
        }
    }
    return true;
}

/* The selection of the replayed row from attn.topk_idxs: its window part must list slots q mod 128 of positions
 * p - 127 .. p, its pool part the window's length plus entries below visible, or -1. */
static bool selection_of_indices(const int32_t *indices, uint64_t index_count, int32_t position, uint32_t visible,
                                 int32_t *selection)
{
    uint32_t selected = 0;
    bool held = index_count >= WINDOW && index_count <= TEST_ATTENTION_KEYS;
    for (uint32_t key = 0; held && key < WINDOW; key++) {
        const int32_t key_position = position - (int32_t)(WINDOW - 1u) + (int32_t)key;
        held = indices[key] == (key_position < 0 ? -1 : key_position % (int32_t)REFERENCE_RING_SLOTS);
    }
    for (uint64_t key = WINDOW; held && key < index_count; key++) {
        const int32_t entry = indices[key] < 0 ? -1 : indices[key] - (int32_t)WINDOW;
        held = indices[key] == -1 || (entry >= 0 && (uint32_t)entry < visible);
        if (held && entry >= 0) {
            selection[selected] = entry;
            selected++;
        }
    }
    qsort(selection, selected, sizeof(int32_t), compare_entries);
    for (uint32_t place = 1; held && place < selected; place++) {
        held = selection[place] > selection[place - 1u];
    }
    for (uint32_t place = selected; place < SELECTED; place++) {
        selection[place] = -1;
    }
    return test_linear_expect(held, "attn.topk_idxs: the window's positions, then distinct visible entries");
}

/* What one replayed step needs beside its case: the stand-in's output (its bits, and their values), the exact form,
 * the engine's, and room for the float32 state and the bfloat16 records as the capture holds them. */
struct replay_buffers {
    float *floats;                              /* [entries or 128][512] */
    uint16_t *bfloat16;                         /* [64][512] */
    uint16_t *stand_in;                         /* [64][512] */
    double *stand_in_values;                    /* [64][512] */
    uint16_t *rotated;                          /* [64][512]: the stand-in's output before it is turned back */
    uint16_t *engine;                           /* [64][512] */
    double *exact;                              /* [64][512] */
    int32_t indices[TEST_ATTENTION_KEYS];
};

/* The name of a capture point's record: layer_L.position_P.<record>. */
static bool replay_name(char *name, uint32_t layer, uint32_t position, const char *record)
{
    const int length = snprintf(name, NAME_CAPACITY, "layer_%" PRIu32 ".position_%" PRIu32 ".%s", layer, position,
                                record);
    return test_linear_expect(length > 0 && (uint32_t)length < NAME_CAPACITY, "a record's name fits");
}

/* The step's inputs from the capture into the case, the stand-in's output into the buffers. */
static bool replay_read(const struct test_capture *capture, uint32_t layer, uint32_t position,
                        struct test_attention_case *data, struct replay_buffers *buffers)
{
    char name[NAME_CAPACITY];
    const uint64_t heads_dims[] = {HEADS, DIMS};
    const uint64_t ring_dims[] = {REFERENCE_RING_SLOTS, DIMS};
    const uint64_t pool_dims[] = {data->pool_ratio == 0 ? 0u : (position + 1u) / data->pool_ratio, DIMS};
    const uint64_t sink_dims[] = {HEADS};
    const uint64_t table_dims[] = { REPLAY_CONTEXT, MODEL_FREQUENCIES,
				    MODEL_PAIR_VALUES };
    uint64_t inexact_count = 0;
    bool held = replay_name(name, layer, position, "attn.q") &&
                capture_read(capture, name, TEST_CAPTURE_BFLOAT16, heads_dims, COUNT_OF(heads_dims), 0, ROW_VALUES,
                             buffers->bfloat16);
    for (uint32_t index = 0; held && index < ROW_VALUES; index++) {
        const double value = test_linear_bfloat16_value(buffers->bfloat16[index]);
        data->queries[index] = test_attention_half_bits(value);     /* as the queries kernel widens them */
        inexact_count += test_attention_core_half_value(data->queries[index]) != value ? 1u : 0u;
    }
    held = held && replay_name(name, layer, position, "attn.topk_idxs");
    const struct test_capture_tensor *const indices = held ? test_capture_find(capture, name) : NULL;
    held = held && indices != NULL && indices->dimension_count == 1u && indices->dimensions[0] <= TEST_ATTENTION_KEYS &&
           capture_read(capture, name, TEST_CAPTURE_INT32, indices->dimensions, 1u, 0, indices->dimensions[0],
                        buffers->indices) &&
           selection_of_indices(buffers->indices, indices->dimensions[0], (int32_t)position, (uint32_t)pool_dims[0],
                                data->selection);
    held = held && replay_name(name, layer, position, "snapshot.ring") &&
           capture_read(capture, name, TEST_CAPTURE_FLOAT32, ring_dims, COUNT_OF(ring_dims), 0,
                        (uint64_t)REFERENCE_RING_SLOTS * DIMS, buffers->floats);
    for (uint32_t key = 0; held && key < WINDOW; key++) {
        const int32_t key_position = (int32_t)position - (int32_t)(WINDOW - 1u) + (int32_t)key;
        held = key_position < 0 ||
               halves_of_floats(buffers->floats + (size_t)((uint32_t)key_position % REFERENCE_RING_SLOTS) * DIMS,
                                DIMS, data->ring + (size_t)((uint32_t)key_position % RING_SLOTS) * DIMS);
    }
    held = held && (data->pool == NULL ||
                    (replay_name(name, source_of_layer(layer), position, "snapshot.pool") &&
                     capture_read(capture, name, TEST_CAPTURE_FLOAT32, pool_dims, COUNT_OF(pool_dims), 0,
                                  pool_dims[0] * DIMS, buffers->floats) &&
                     halves_of_floats(buffers->floats, pool_dims[0] * DIMS, data->pool)));
    held = held && replay_name(name, layer, position, "attn.sink") &&
           capture_read(capture, name, TEST_CAPTURE_FLOAT32, sink_dims, COUNT_OF(sink_dims), 0, HEADS, data->sinks);
    held = held && capture_read(capture, data->rotation_use == 0 ? THETA_0_TABLE : THETA_1_TABLE,
                                TEST_CAPTURE_FLOAT32, table_dims, COUNT_OF(table_dims), (uint64_t)position * USE_FLOATS,
                                USE_FLOATS, data->rotation + data->rotation_use * USE_FLOATS);
    held = held && replay_name(name, layer, position, "attn.output") &&
           capture_read(capture, name, TEST_CAPTURE_BFLOAT16, heads_dims, COUNT_OF(heads_dims), 0, ROW_VALUES,
                        buffers->stand_in) &&
           replay_name(name, layer, position, "attn.output_rotated") &&
           capture_read(capture, name, TEST_CAPTURE_BFLOAT16, heads_dims, COUNT_OF(heads_dims), 0, ROW_VALUES,
                        buffers->rotated);
    if (held && inexact_count != 0) {
        (void)printf("replay layer %" PRIu32 " position %" PRIu32 ": %" PRIu64
                     " queries FP16 does not hold (risk 8 of the plan)\n", layer, position, inexact_count);
    }
    return held;
}

/* The reference's own turning back (REF:781) of the stand-in's output, from attn.output_rotated to attn.output:
 * dims below 448 kept, each value of 448 .. 511 the kernel's form bit for bit; how many each form of
 * test_attention_turn_back gives (the kernel's, the other product fused, products rounded) is printed. */
static bool turn_back_holds(const uint16_t *rotated, const uint16_t *turned, const float *angles)
{
    uint64_t kept_count = 0, form_counts[TEST_ATTENTION_TURN_FORMS] = {0};
    for (uint32_t head = 0; head < HEADS; head++) {
        const uint16_t *const before = rotated + (size_t)head * DIMS, *const after = turned + (size_t)head * DIMS;
	for (uint32_t dim = 0; dim < MODEL_FIRST_ROTATED_DIM; dim++) {
		kept_count += before[dim] == after[dim] ? 1u : 0u;
	}
	for (uint32_t dim = MODEL_FIRST_ROTATED_DIM; dim < DIMS;
	     dim += MODEL_PAIR_VALUES) {
		const float first =
			(float)test_linear_bfloat16_value(before[dim]);
		const float second =
			(float)test_linear_bfloat16_value(before[dim + 1u]);
		const float *const angle =
			angles + (dim - MODEL_FIRST_ROTATED_DIM);
		for (uint32_t form = 0; form < TEST_ATTENTION_TURN_FORMS;
		     form++) {
			for (uint32_t part = 0; part < MODEL_PAIR_VALUES;
			     part++) {
				const float value = test_attention_turn_back(
					first, second, angle[0], angle[1], part,
					(enum test_attention_turn_form)form);
				form_counts[form] +=
					test_linear_bfloat16_nearest(
						(double)value) ==
							after[dim + part] ?
						1u :
						0u;
			}
		}
	}
    }
    const uint64_t rotated_count = (uint64_t)HEADS * MODEL_ROTATED_DIMS;
    (void)printf("TURN kept=%" PRIu64 " of %" PRIu64 " kernel_form=%" PRIu64
		 " other_fused=%" PRIu64 " products_rounded=%" PRIu64
		 " of %" PRIu64 "\n",
		 kept_count, (uint64_t)HEADS * MODEL_FIRST_ROTATED_DIM,
		 form_counts[TEST_ATTENTION_TURN_KERNEL],
		 form_counts[TEST_ATTENTION_TURN_OTHER],
		 form_counts[TEST_ATTENTION_TURN_PRODUCTS], rotated_count);
    return kept_count == (uint64_t)HEADS * MODEL_FIRST_ROTATED_DIM &&
	   form_counts[TEST_ATTENTION_TURN_KERNEL] == rotated_count;
}

/* The gate of one order on one card (see the top), and its REPLAY line. kernel_form: of the order. */
static void replay_verdict(uint32_t card_index, uint32_t layer, uint32_t position, uint32_t order,
                           const struct replay_buffers *buffers, const double *kernel_form, double stand_in_squared)
{
    struct test_attention_distance engine, from_kernel_form, from_stand_in, stand_in_form;
    memset(&engine, 0, sizeof(engine));
    memset(&from_kernel_form, 0, sizeof(from_kernel_form));
    memset(&from_stand_in, 0, sizeof(from_stand_in));
    memset(&stand_in_form, 0, sizeof(stand_in_form));
    test_attention_distance_add(&engine, buffers->engine, buffers->exact, ROW_VALUES);
    test_attention_distance_add(&from_kernel_form, buffers->engine, kernel_form, ROW_VALUES);
    test_attention_distance_add(&from_stand_in, buffers->engine, buffers->stand_in_values, ROW_VALUES);
    test_attention_distance_add(&stand_in_form, buffers->stand_in, kernel_form, ROW_VALUES);
    const double engine_squared = test_attention_relative_squared(&engine);
    const bool is_close = test_attention_core_within(&from_stand_in, REFERENCE_FROM_STAND_IN) &&
                          from_stand_in.equal_count * PER_THOUSAND >=
                              from_stand_in.value_count * REFERENCE_EQUAL_PER_THOUSAND;
    const bool held = engine.not_finite_count == 0 &&
		      engine_squared <= STAND_IN_FACTOR * STAND_IN_FACTOR *
						stand_in_squared &&
		      test_attention_core_within(&from_kernel_form,
						 TEST_ATTENTION_TOLERANCE) &&
		      (order != LINEAR_ORDER_REFERENCE || is_close);
    (void)printf("REPLAY card=%" PRIu32 " layer=%" PRIu32 " position=%" PRIu32 " order=%s engine=%.3e stand_in=%.3e "
                 "ratio=%.3f kernel_form=%.3e stand_in_form=%.3e from_stand_in=%.3e equal=%.4f%% verdict=%s\n",
                 card_index, layer, position, order_names[order], square_root(engine_squared),
                 square_root(stand_in_squared),
                 stand_in_squared > 0.0 ? square_root(engine_squared / stand_in_squared) : 0.0,
                 square_root(test_attention_relative_squared(&from_kernel_form)),
                 square_root(test_attention_relative_squared(&stand_in_form)),
                 square_root(test_attention_relative_squared(&from_stand_in)),
                 test_linear_percent(from_stand_in.equal_count, from_stand_in.value_count), held ? "PASS" : "FAIL");
    (void)test_linear_expect(fflush(stdout) == 0, "the line is written before any failure line");
    (void)test_linear_expect(
	    held,
	    order == LINEAR_ORDER_REFERENCE ?
		    "the reference order bit-close to the stand-in and within twice its distance" :
		    "the engine within twice the stand-in's distance from the exact result");
}

/* One step of one layer replayed on every card in both orders. */
static void replay_point(const struct test_capture *capture, struct test_card *cards, uint32_t card_count,
                         uint32_t layer, uint32_t position, struct replay_buffers *buffers)
{
    struct test_attention_case data;
    struct test_attention_row row;
    struct test_attention_plan plan;
    memset(&data, 0, sizeof(data));
    test_linear_context("replay of layer %" PRIu32 " at position %" PRIu32, layer, position);
    data.name = "replay";
    data.session_count = 1u;
    data.context_token_limit = REPLAY_CONTEXT;
    data.pool_ratio = ratio_of_layer(layer);
    data.rotation_use = data.pool_ratio != 0 ? 1u : 0u;
    data.pool_entries = data.pool_ratio == 0 ? 0u : REPLAY_CONTEXT / data.pool_ratio;
    const bool is_held = test_attention_case_hold(&data);
    if (is_held) {
        const float poison = (float)test_attention_core_half_value(TEST_ATTENTION_POISON);
	data.step_rows[0] = (struct step_row_record){ (int32_t)position, 0, 0,
						      STEP_ROW_LOGITS };
	for (size_t index = 0; index < (size_t)ROWS * ROTATION_FLOATS; index++) {
            data.rotation[index] = poison;      /* the uses the launch must not read */
        }
    }
    if (!is_held || !replay_read(capture, layer, position, &data, buffers)) {
        test_linear_fail("the capture lacks a record the replay needs, or holds one it cannot take");
        test_attention_case_free(&data);
        return;
    }
    (void)test_linear_expect(turn_back_holds(buffers->rotated, buffers->stand_in,
                                             data.rotation + data.rotation_use * USE_FLOATS),
                             "the reference turns its output back as the kernel does");
    bool computed = true;
    for (uint32_t order = 0; order < ORDERS; order++) {
	    test_attention_reference_row(&data, 0,
					 (enum linear_order_code)order, &row);
	    computed = computed &&
		       test_attention_core_row(&row, data.reference[order]) ==
			       STATUS_OK;
    }
    row.form = TEST_ATTENTION_EXACT;
    if (!test_linear_expect(
		computed && test_attention_core_row(&row, buffers->exact) ==
				    STATUS_OK,
		"the reference computes the replayed row")) {
	    test_attention_case_free(&data);
	    return;
    }
    struct test_attention_distance stand_in;
    memset(&stand_in, 0, sizeof(stand_in));
    test_attention_distance_add(&stand_in, buffers->stand_in, buffers->exact, ROW_VALUES);
    for (uint32_t index = 0; index < ROW_VALUES; index++) {
        buffers->stand_in_values[index] = test_linear_bfloat16_value(buffers->stand_in[index]);
    }
    plan_rows(&plan, "replayed", 0, 1u);
    for (uint32_t card_index = 0; card_index < card_count; card_index++) {
        for (uint32_t order = 0; cards[card_index].is_open && order < ORDERS; order++) {
            if (test_attention_plan_run(&cards[card_index].card, &cards[card_index].kernels, &data, &plan,
                                        prepared_order(order), buffers->engine)) {
                replay_verdict(card_index, layer, position, order, buffers, data.reference[order],
                               test_attention_relative_squared(&stand_in));
            }
        }
    }
    test_attention_case_free(&data);
}

static void replay(const char *folder, struct test_card *cards, uint32_t card_count)
{
    struct test_capture *capture = NULL;
    struct replay_buffers buffers;
    uint32_t replayed_count = 0;
    char name[NAME_CAPACITY];
    test_linear_context("the replay of %s", folder);
    buffers.floats = malloc((size_t)REPLAY_CONTEXT * DIMS * sizeof(float));
    buffers.bfloat16 = malloc(ROW_VALUES * sizeof(uint16_t));
    buffers.stand_in = malloc(ROW_VALUES * sizeof(uint16_t));
    buffers.stand_in_values = malloc(ROW_VALUES * sizeof(double));
    buffers.rotated = malloc(ROW_VALUES * sizeof(uint16_t));
    buffers.engine = malloc(ROW_VALUES * sizeof(uint16_t));
    buffers.exact = malloc(ROW_VALUES * sizeof(double));
    if (buffers.floats == NULL || buffers.bfloat16 == NULL || buffers.stand_in == NULL ||
        buffers.stand_in_values == NULL || buffers.rotated == NULL || buffers.engine == NULL || buffers.exact == NULL) {
        test_linear_fail("the replay's buffers are not held");
    } else if (test_linear_expect_status(test_capture_open(folder, &capture),
					 STATUS_OK, "the capture opens")) {
	    for (size_t layer = 0; layer < COUNT_OF(replay_layers); layer++) {
		    for (size_t position = 0;
			 position < COUNT_OF(replay_positions); position++) {
			    if (replay_name(name, replay_layers[layer],
					    replay_positions[position],
					    "snapshot.ring") &&
				test_capture_find(capture, name) != NULL) {
				    replay_point(capture, cards, card_count,
						 replay_layers[layer],
						 replay_positions[position],
						 &buffers);
				    replayed_count++;
			    }
		    }
	    }
	    test_linear_context("the replay of %s", folder);
	    (void)test_linear_expect(replayed_count != 0,
				     "the capture holds a step to replay");
	    (void)printf("replay: %" PRIu32 " steps of %s\n", replayed_count,
			 folder);
    }
    test_capture_close(capture);
    free(buffers.floats);
    free(buffers.bfloat16);
    free(buffers.stand_in);
    free(buffers.stand_in_values);
    free(buffers.rotated);
    free(buffers.engine);
    free(buffers.exact);
}

/* ---- the program ---- */

static bool arguments_read(int argument_count, char **arguments, const char **capture_folder_out)
{
    for (int index = 1; index < argument_count; index++) {
        const bool has_value = index + 1 < argument_count;
        if (has_value && strcmp(arguments[index], "--capture") == 0) {
            *capture_folder_out = arguments[index + 1];
        } else if (has_value && strcmp(arguments[index], "--break") == 0 &&
                   (strcmp(arguments[index + 1], "window") == 0 || strcmp(arguments[index + 1], "word") == 0 ||
                    strcmp(arguments[index + 1], "order") == 0)) {
            break_window = break_window || strcmp(arguments[index + 1], "window") == 0;
            break_word = break_word || strcmp(arguments[index + 1], "word") == 0;
            break_order = break_order || strcmp(arguments[index + 1], "order") == 0;
        } else {
            (void)fprintf(stderr, "usage: attention_core_card_test [--capture FOLDER] [--break window|word|order]\n");
            return false;
        }
        index++;
    }
    return true;
}

int main(int argument_count, char **arguments)
{
	static struct test_card cards[BACKEND_MAXIMUM_CARDS];
	const char *capture_folder = NULL;
	uint32_t card_count = 0;
	uint32_t agreeing_count = 0;
	if (!arguments_read(argument_count, arguments, &capture_folder)) {
		return EXIT_USAGE;
	}
    (void)alarm(TEST_TIME_LIMIT_SECONDS);
    (void)test_linear_expect_status(backend_cuda_value.card_count(&card_count),
				    STATUS_OK, "the cards are counted");
    if (!test_linear_expect(card_count > 0 &&
				    card_count <= BACKEND_MAXIMUM_CARDS,
			    "the machine has cards for the test")) {
	    return EXIT_FAILURE;
    }
    for (uint32_t card_index = 0; card_index < card_count; card_index++) {
	    cards[card_index].is_open =
		    test_linear_card_open(&cards[card_index].card, card_index,
					  WORK_CAPACITY, WEIGHTS_CAPACITY) &&
		    test_linear_expect_status(
			    attention_core_kernels_prepare(
				    &backend_cuda_value,
				    cards[card_index].card.card,
				    cards[card_index].card.arithmetic_class,
				    &cards[card_index].kernels),
			    STATUS_OK,
			    "the kernels of the attention core are prepared");
    }
    uint16_t *const outputs[ORDERS] = {malloc((size_t)ROWS * ROW_VALUES * sizeof(uint16_t)),
                                       malloc((size_t)ROWS * ROW_VALUES * sizeof(uint16_t))};
    uint16_t *const alone = malloc((size_t)ROWS * ROW_VALUES * sizeof(uint16_t));
    const bool are_held = outputs[0] != NULL && outputs[1u] != NULL && alone != NULL;
    for (uint32_t kind = 0; are_held && kind < CASE_COUNT; kind++) {
        struct test_attention_case data;
        if (case_make(&case_specs[kind], &data)) {
            for (uint32_t card_index = 0; card_index < card_count; card_index++) {
                bool ran = cards[card_index].is_open;
                for (uint32_t order = 0; ran && order < ORDERS; order++) {
                    ran = case_run(&cards[card_index], &data, order, outputs[order], alone);
                }
                if (ran && !data.exact) {
                    expect_orders_apart(card_index, &data, outputs);
                }
            }
        }
        test_attention_case_free(&data);
    }
    (void)test_linear_expect(are_held, "the outputs are held");
    for (uint32_t order = 0; order < ORDERS; order++) {
        free(outputs[order]);
    }
    free(alone);
    block_cases(cards, card_count);
    for (uint32_t card_index = 0; card_index < card_count; card_index++) {
        uint32_t first = 0;     /* the first open card (of its class where CLASSES_DIFFER) */
        while (first < card_index &&
               (!cards[first].is_open ||
                (CLASSES_DIFFER && cards[first].card.arithmetic_class != cards[card_index].card.arithmetic_class))) {
            first++;
        }
        test_linear_context("card %" PRIu32, card_index);
        (void)printf("card %" PRIu32 " digest %016" PRIx64 "\n", card_index, cards[card_index].digest);
        agreeing_count += cards[card_index].is_open &&
                          test_linear_expect(cards[card_index].digest == cards[first].digest,
                                             "the bits of the first card of its class") ? 1u : 0u;
    }
    if (capture_folder != NULL) {
        replay(capture_folder, cards, card_count);
    } else {
        (void)printf("replay: no capture folder given: replayed nothing\n");
    }
    for (uint32_t card_index = 0; card_index < card_count; card_index++) {
        test_linear_card_close(&cards[card_index].card);
    }
    (void)printf("%" PRIu32 " of %" PRIu32 " cards computed the bits of their class's first card\n", agreeing_count,
                 card_count);
    if (test_linear_failures() != 0) {
        (void)fprintf(stderr, "attention_core_card_test FAILED: %d failures%s\n", test_linear_failures(),
                      break_window || break_word || break_order ? " (a break was asked for)" : "");
        return EXIT_FAILURE;
    }
    (void)printf("attention_core_card_test passed on %" PRIu32 " cards\n", card_count);
    return EXIT_SUCCESS;
}
