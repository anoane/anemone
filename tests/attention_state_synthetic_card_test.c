/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * attention_state_synthetic_card_test: the kernels of the attention state on the real cards against the host
 * reference (attention_state_reference.h), and a row's bits whatever its companions (class R).
 *
 *     attention_state_synthetic_card_test [--break weight|frequency|rounding]
 *     attention_state_synthetic_card_test --trap <card> slot|negative|beyond|empty|more|overflow
 *
 * It needs the cards free (`make card_tests`): a card without room FAILS the test, it is never skipped. On every
 * card, three layers (theta 0 without a pool; pools of ratio 1 and of ratio 2, both with an indexer; the ratio-1
 * layer runs heads 32 to 63 only, as a rank of two, and must leave the others untouched) through the planned steps
 * of two sessions: steps of 1 to 8 rows, both sessions interleaved, a 2-to-1 group cut by a step, a stale row behind
 * a step's rows, a session begun again. Every output against the host reference: the rotation within two float32
 * places (cosf and sinf are within two of the true value), every other value the nearest of its format to the
 * reference give or take 2^-20 of the magnitude of its terms (test_attention_within), the main queries in bfloat16;
 * the compressor's slots bit for bit. Then every row again alone, in a launch of one row with state of its own: its
 * outputs, and at the end the ring, the slots, the pool and the keys, bit for bit those of the steps. Last, the
 * rotation step at positions up to 2^20 - 1. A pooled value is kept off the ties of bfloat16 by its inputs (a value
 * a tie away would round either way on the card and in double precision alike).
 *   - --break weight: the card gets one kv_norm weight changed, the reference keeps it; --break frequency: one
 *     frequency word one place off; --break rounding: every rounded value is held to the reference moved one place
 *     of its format toward zero, as a kernel that cut its values would be held to the true one. Each must make the
 *     test FAIL: shown failing.
 *   - --trap <card> <case>: in the builds that link the checked kernel modules, a step the checked modules refuse
 *     must end the launch with STATUS_DEVICE_ERROR, and the card then begins no new work: a session slot
 *     past the capacity, a position below 0, one at the context, a step of no row, one of more rows than the
 *     launch, a main query no FP16 value holds. A trap ends what a process can do with a card: a batch runs each
 *     case as a step of its own; with the release modules no case is run.
 */
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "anemone/attention_state.h"
#include "anemone/attention_state_records.h"
#include "anemone/backend_seam.h"
#include "anemone/status.h"
#include "attention_state_card_fixture.h"
#include "attention_state_reference.h"
#include "linear_card_fixture.h"
#include "linear_reference.h"

#define TEST_TIME_LIMIT_SECONDS 1800
#define WORK_CAPACITY (16ull << 20u)
#define WEIGHTS_CAPACITY (1ull << 20u)
#define ROWS LINEAR_MAXIMUM_ROWS
#define SESSIONS 2u
#define CONTEXT 64u
#define QUERY_RANK MODEL_QUERY_RANK
#define DIMS MODEL_HEAD_DIMS
#define INDEX_DIMS MODEL_INDEX_HEAD_DIMS
#define INDEX_VALUES ((uint64_t)MODEL_INDEX_HEADS * MODEL_INDEX_HEAD_DIMS)
#define HEAD_VALUES ((uint64_t)MODEL_HEADS_VALUE * MODEL_HEAD_DIMS)
#define FREQUENCIES MODEL_FREQUENCIES
#define WORDS (MODEL_THETAS * MODEL_FREQUENCIES)
#define PAIR MODEL_PAIR_VALUES
#define FIRST_ROTATED MODEL_FIRST_ROTATED_DIM
#define INDEX_FIRST_ROTATED (MODEL_INDEX_HEAD_DIMS - MODEL_ROTATED_DIMS)
#define RATIO_TWO MODEL_POOLED_RATIO
#define SLOT_FLOATS ((uint64_t)ATTENTION_SLOT_VECTORS * MODEL_HEAD_DIMS)
#define COSINE_PLACES 2.0
#define INPUT_STREAM 0x0A77E000000ull
#define SESSION_STREAMS 0x100000ull
#define POSITION_STREAMS 16u
#define KV_STREAM 7u
#define SCORE_STREAM 8u
#define SIGN_STREAM 9u
#define WEIGHT_STREAM 0x0A77E100000ull
#define FLOAT_EXPONENT_LOWEST 123u              /* 2^-4 .. 4: six binades */
#define FLOAT_EXPONENT_SPREAD 6u
#define FLOAT_EXPONENT_SHIFT 23u
#define FLOAT_MANTISSA_MASK 0x7FFFFFull
#define FLOAT_SIGN_BIT 0x80000000u
#define HASH_MANTISSA_SHIFT 8u
#define SCORE_SPAN 16.0                         /* scores from -8 to 8 */
#define SCORE_UNITS 16777216u                   /* 2^24 */
#define TIE_MARGIN 2.44140625e-4                /* 2^-12: a pooled value this far from a bfloat16 tie is safe */
#define NUDGE 1.00390625f                       /* 1 + 2^-8 */
#define MOST_NUDGES 64u
#define LOG_THETA_0 9.21034037197618273607      /* ln 10,000 (CFG:30) */
#define LOG_THETA_1 11.98292909421596317090     /* ln 160,000 (CFG:51) */
#define BROKEN_WEIGHT 7u                        /* the kv_norm weight --break weight changes */
#define BROKEN_WORD FREQUENCIES                 /* --break frequency: the first word of theta 1 */
#define HALF_HEADS 32u                          /* the ratio-1 layer runs heads 32 to 63 */
#define TRAP_LARGE_BFLOAT16 0x4800u             /* 2^17: beyond FP16 */
#define MORE_ROWS 2u                            /* a step's, in a launch of one */
#define BREAK_FLAG "--break"
#define TRAP_FLAG "--trap"
#define FLAG_ARGUMENTS 3                        /* the program's name, a flag and its value */
#define TRAP_ARGUMENTS 4                        /* the program's name, the flag, a card and a case */
#define DECIMAL 10

/* The planned steps: rows (position, session); those from row_count on are stale rows of no step. */
struct planned_step {
    uint32_t row_count;
    struct test_attention_row rows[ROWS];
};
static const struct planned_step planned_steps[] = {
    {8, {{0, 0}, {1, 0}, {2, 0}, {3, 0}, {0, 1}, {1, 1}, {2, 1}, {3, 1}}},
    {8, {{4, 1}, {4, 0}, {5, 1}, {5, 0}, {6, 1}, {6, 0}, {7, 1}, {7, 0}}},
    {8, {{8, 0}, {9, 0}, {10, 0}, {11, 0}, {12, 0}, {13, 0}, {14, 0}, {15, 0}}},
    {7, {{16, 0}, {17, 0}, {18, 0}, {19, 0}, {20, 0}, {21, 0}, {22, 0}, {61, 1}}},
    {3, {{23, 0}, {24, 0}, {25, 0}, {63, 1}, {62, 1}, {61, 1}, {60, 1}, {59, 1}}},
    {1, {{8, 1}, {1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}, {6, 0}, {7, 0}}},
    {2, {{9, 1}, {10, 1}, {33, 0}, {34, 0}, {35, 0}, {36, 0}, {37, 0}, {38, 0}}},
    {2, {{0, 0}, {1, 0}, {57, 1}, {56, 1}, {55, 1}, {54, 1}, {53, 1}, {52, 1}}},
};
static const struct test_attention_row large_positions[ROWS] = {
    {1000, 0}, {4095, 0}, {8191, 0}, {12345, 0}, {16383, 0}, {65535, 0}, {524287, 0}, {1048575, 0}};
static const struct test_attention_kind synthetic_kinds[] = {{0, 0, 0, 0}, {1, 1, 1, HALF_HEADS}, {1, 1, 2, 0}};
static const struct test_attention_kind trap_kind = {1, 1, 2, 0};

/* --trap: a step row and the step's row count that the checked modules refuse, in a layer of one row. */
struct trap_case {
    const char *name;
    struct test_attention_row row;
    uint32_t row_count;
    uint16_t first_main_query;                  /* bfloat16 bits of the first value of wq_b's output */
};
static const struct trap_case trap_cases[] = {
    {"slot", {1, SESSIONS}, 1, 0}, {"negative", {-1, 0}, 1, 0},    {"beyond", {CONTEXT, 0}, 1, 0},
    {"empty", {1, 0}, 0, 0},       {"more", {1, 0}, MORE_ROWS, 0}, {"overflow", {1, 0}, 1, TRAP_LARGE_BFLOAT16}};

/* What a row of a step feeds the kernels: a function of its session and position alone. */
struct row_inputs {
    uint16_t query_rank[QUERY_RANK];
    uint16_t window[DIMS];
    uint16_t heads[HEAD_VALUES];
    uint16_t index_heads[INDEX_VALUES];
    uint16_t pooled_bfloat16[DIMS];             /* ratio 1: wkv's bfloat16 product */
    float kv[DIMS];                             /* ratio 2: wkv's and wgate's float32 products */
    float score[DIMS];
    uint16_t key[INDEX_DIMS];                   /* wk's output */
};

/* One case: the layer of the steps (8 rows) and that of the rows alone (1 row), on one card. */
struct synthetic_case {
    struct test_linear_card *card;
    struct test_attention_layer steps;
    struct test_attention_layer alone;
    struct test_attention_kind kind;
};

static struct row_inputs row_inputs;
static struct row_inputs partner_inputs;
static uint16_t norm_weights[TEST_ATTENTION_NORMS][QUERY_RANK];
static float frequencies[WORDS];
static uint32_t index_source_of[INDEX_VALUES];  /* the head-major value at each placed half of the index queries */
static uint32_t key_source_of[INDEX_DIMS];
static double reference[HEAD_VALUES];
static double magnitudes[HEAD_VALUES];
static double values[HEAD_VALUES];
static struct test_attention_tally tallies[TEST_ATTENTION_CHECK_COUNT];
static const char *broken = NULL;

static const uint32_t norm_widths[TEST_ATTENTION_NORMS] = {QUERY_RANK, DIMS, DIMS, INDEX_DIMS};

static float synthetic_float(uint64_t hash, bool is_negative)
{
    const uint32_t bits = (is_negative ? FLOAT_SIGN_BIT : 0) |
                          ((FLOAT_EXPONENT_LOWEST + (uint32_t)(hash % FLOAT_EXPONENT_SPREAD)) << FLOAT_EXPONENT_SHIFT) |
                          (uint32_t)((hash >> HASH_MANTISSA_SHIFT) & FLOAT_MANTISSA_MASK);
    float value = 0.0f;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static double magnitude_of(double value)
{
    return value < 0.0 ? -value : value;
}

static bool breaks(const char *what)
{
    return broken != NULL && strcmp(broken, what) == 0;
}

/* The value a rounded output is held to: the reference's, or under --break rounding one place toward zero. */
static double expected_of(double reference_value, int32_t fraction_bits, int32_t lowest)
{
    const double place = test_attention_last_place(reference_value, fraction_bits, lowest);
    return !breaks("rounding") ? reference_value
           : reference_value > 0.0 ? reference_value - place
           : reference_value < 0.0 ? reference_value + place
                                   : reference_value;
}

static void kv_score_of(uint32_t session, int32_t position, float *kv, float *score)
{
    const uint64_t stream = INPUT_STREAM + session * SESSION_STREAMS + (uint64_t)position * POSITION_STREAMS;
    for (uint32_t dim = 0; dim < DIMS; dim++) {
        const bool is_negative = (test_linear_hash(INPUT_STREAM + SIGN_STREAM, dim) & 1u) != 0;   /* a sign a dim */
        kv[dim] = synthetic_float(test_linear_hash(stream + KV_STREAM, dim), is_negative);
        score[dim] = (float)((double)(test_linear_hash(stream + SCORE_STREAM, dim) % SCORE_UNITS) / SCORE_UNITS *
                                 SCORE_SPAN - SCORE_SPAN / PAIR);
    }
}

static bool is_near_a_tie(double pooled)
{
    return test_linear_bfloat16_nearest(pooled * (1.0 - TIE_MARGIN)) !=
           test_linear_bfloat16_nearest(pooled * (1.0 + TIE_MARGIN));
}

/* A row's inputs; at an emitting position of ratio 2 its kv moved off the ties of the value it pools. */
static void row_inputs_of(uint32_t session, int32_t position, struct row_inputs *inputs)
{
    uint64_t stream = INPUT_STREAM + session * SESSION_STREAMS + (uint64_t)position * POSITION_STREAMS;
    test_linear_values(inputs->query_rank, QUERY_RANK, stream++, TEST_LINEAR_VALUES_PLAIN);
    test_linear_values(inputs->window, DIMS, stream++, TEST_LINEAR_VALUES_PLAIN);
    test_linear_values(inputs->heads, HEAD_VALUES, stream++, TEST_LINEAR_VALUES_PLAIN);
    test_linear_values(inputs->index_heads, INDEX_VALUES, stream++, TEST_LINEAR_VALUES_PLAIN);
    test_linear_values(inputs->pooled_bfloat16, DIMS, stream++, TEST_LINEAR_VALUES_PLAIN);
    test_linear_values(inputs->key, INDEX_DIMS, stream, TEST_LINEAR_VALUES_PLAIN);
    kv_score_of(session, position, inputs->kv, inputs->score);
    if ((uint32_t)position % RATIO_TWO == 1) {
        kv_score_of(session, position - 1, partner_inputs.kv, partner_inputs.score);
        for (uint32_t dim = 0; dim < DIMS; dim++) {
            for (uint32_t nudge = 0;
                 nudge < MOST_NUDGES && is_near_a_tie(test_attention_pooled(partner_inputs.kv[dim],
                                                                            partner_inputs.score[dim],
                                                                            inputs->kv[dim], inputs->score[dim]));
                 nudge++) {
                inputs->kv[dim] *= NUDGE;
            }
        }
    }
}

/* The model's words, as the reference's expression forms them (REF:376), without YaRN: what matters here is that
 * they are fixed. */
static void frequencies_make(void)
{
    for (uint32_t word = 0; word < FREQUENCIES; word++) {
	    const double exponent =
		    -(double)(PAIR * word) / (double)MODEL_ROTATED_DIMS;
	    frequencies[word] =
		    (float)test_attention_exponential(exponent * LOG_THETA_0);
	    frequencies[FREQUENCIES + word] =
		    (float)test_attention_exponential(exponent * LOG_THETA_1);
    }
}

/* The placements, walked once: the source of each placed half. */
static void placements_make(void)
{
    static uint16_t identity[INDEX_VALUES], placed[INDEX_VALUES];
    for (uint32_t index = 0; index < INDEX_VALUES; index++) {
        identity[index] = (uint16_t)index;
    }
    test_attention_layout_index_queries(identity, placed);
    for (uint32_t index = 0; index < INDEX_VALUES; index++) {
        index_source_of[index] = placed[index];
    }
    test_attention_layout_key(identity, placed);
    for (uint32_t index = 0; index < INDEX_DIMS; index++) {
        key_source_of[index] = placed[index];
    }
}

static void bfloat16_values(const uint16_t *bits, uint32_t count, double *values_out)
{
    for (uint32_t index = 0; index < count; index++) {
        values_out[index] = test_linear_bfloat16_value(bits[index]);
    }
}

/* reference_out: the values rotated from first_rotated on by the card's angles of a row and use; magnitudes_out:
 * the magnitude of each one's terms. */
static void rotated(const struct test_linear_card *card,
		    struct linear_range rotation, uint32_t row, uint32_t use,
		    uint32_t width, uint32_t first_rotated,
		    double *reference_out)
{
    for (uint32_t dim = 0; dim < width; dim += PAIR) {
        double cosine = 1.0, sine = 0.0, real = reference_out[dim], imaginary = reference_out[dim + 1u];
        if (dim >= first_rotated) {
            test_attention_angle_of(card, rotation, row, use, (dim - first_rotated) / PAIR, &cosine, &sine);
        }
        magnitudes[dim] = magnitude_of(real * cosine) + magnitude_of(imaginary * sine);
        magnitudes[dim + 1u] = magnitude_of(real * sine) + magnitude_of(imaginary * cosine);
        test_attention_rotate(&real, &imaginary, cosine, sine);
        reference_out[dim] = real;
        reference_out[dim + 1u] = imaginary;
    }
}

static void check_halves(const struct test_linear_card *card,
			 struct linear_range range, uint64_t first,
			 const uint32_t *source_of, uint32_t count,
			 enum test_attention_check check, const char *what)
{
    for (uint32_t index = 0; index < count; index++) {
        const uint32_t source = source_of == NULL ? index : source_of[index];
        const uint16_t bits = test_attention_half_at(card, range, first + index);
        test_attention_tally(tallies, check, bits == test_attention_half_nearest(reference[source]),
                             test_attention_within(test_attention_half_value(bits),
                                                   expected_of(reference[source], TEST_ATTENTION_HALF_BITS,
                                                               TEST_ATTENTION_HALF_LOWEST),
                                                   magnitudes[source], TEST_ATTENTION_HALF_BITS,
                                                   TEST_ATTENTION_HALF_LOWEST),
                             what, index);
    }
}

static void check_bfloat16(const struct test_linear_card *card,
			   struct linear_range range, uint64_t first,
			   uint32_t count, enum test_attention_check check,
			   const char *what)
{
    for (uint32_t index = 0; index < count; index++) {
        const uint16_t bits = test_attention_half_at(card, range, first + index);
        test_attention_tally(tallies, check, bits == test_linear_bfloat16_nearest(reference[index]),
                             test_attention_within(test_linear_bfloat16_value(bits),
                                                   expected_of(reference[index], TEST_ATTENTION_BFLOAT16_BITS,
                                                               TEST_ATTENTION_BFLOAT16_LOWEST),
                                                   magnitudes[index], TEST_ATTENTION_BFLOAT16_BITS,
                                                   TEST_ATTENTION_BFLOAT16_LOWEST),
                             what, index);
    }
}

/* The norm of a row by a weight, its magnitudes those of the values (a norm only scales). */
static void normed(const double *row_values, enum test_attention_norm norm, uint32_t width)
{
    test_attention_norm(row_values, norm_weights[norm], width, reference);
    for (uint32_t index = 0; index < width; index++) {
        magnitudes[index] = magnitude_of(reference[index]);
    }
}

/* Every row of a step of the steps' layer against the reference. */
static void step_check(const struct synthetic_case *test_case, const struct planned_step *step)
{
    const struct test_linear_card *const card = test_case->card;
    const struct attention_state_description *const layer =
	    &test_case->steps.layer;
    const uint32_t ratio = layer->pool_ratio;
    for (uint32_t row = 0; row < ROWS; row++) {             /* the rotation of every row, stale ones included */
	    for (uint32_t use = 0; use < MODEL_ROTATION_USES; use++) {
		    const float turned_position =
			    (float)(step->rows[row].position -
				    (use == ATTENTION_ROTATION_GROUP_START ?
					     1 :
					     0));
		    for (uint32_t pair = 0; pair < FREQUENCIES; pair++) {
			    double cosine = 0.0, sine = 0.0, card_cosine = 0.0,
				   card_sine = 0.0;
			    const float angle =
				    turned_position *
				    frequencies[(use == 0 ? 0 : FREQUENCIES) +
						pair];
			    test_attention_cosine_sine((double)angle, &cosine,
						       &sine);
			    test_attention_angle_of(card, layer->rotation, row,
						    use, pair, &card_cosine,
						    &card_sine);
			    const double cosine_place =
				    test_attention_last_place(
					    cosine, TEST_ATTENTION_FLOAT_BITS,
					    TEST_ATTENTION_FLOAT_LOWEST);
			    const double sine_place = test_attention_last_place(
				    sine, TEST_ATTENTION_FLOAT_BITS,
				    TEST_ATTENTION_FLOAT_LOWEST);
			    test_attention_tally(
				    tallies, TEST_ATTENTION_ROTATION,
				    (float)card_cosine == (float)cosine &&
					    (float)card_sine == (float)sine,
				    magnitude_of(card_cosine - cosine) <=
						    COSINE_PLACES *
							    cosine_place &&
					    magnitude_of(card_sine - sine) <=
						    COSINE_PLACES * sine_place,
				    "cos and sin of a row", pair);
		    }
	    }
    }
    for (uint32_t row = 0; row < step->row_count; row++) {
        const uint32_t session = step->rows[row].session;
        const int32_t position = step->rows[row].position;
        const bool emits = ratio != 0 && ((uint32_t)position + 1u) % ratio == 0;
	const uint32_t pool_use = ratio == 1 ? ATTENTION_ROTATION_POOLED :
					       ATTENTION_ROTATION_GROUP_START;
	row_inputs_of(session, position, &row_inputs);
        bfloat16_values(row_inputs.query_rank, QUERY_RANK, values);
        normed(values, TEST_ATTENTION_Q_NORM, QUERY_RANK);
        check_bfloat16(card, layer->query_normed, (uint64_t)row * QUERY_RANK, QUERY_RANK, TEST_ATTENTION_QR, "qr");
        bfloat16_values(row_inputs.window, DIMS, values);
        normed(values, TEST_ATTENTION_KV_NORM, DIMS);
        rotated(card, layer->rotation, row, layer->theta, DIMS, FIRST_ROTATED, reference);
	const uint64_t ring_row = (uint64_t)session * MODEL_RING_SLOTS +
				  (uint32_t)position % MODEL_RING_SLOTS;
	check_halves(card, layer->ring, ring_row * DIMS, NULL, DIMS, TEST_ATTENTION_WINDOW, "a window row");
	for (uint32_t head = 0; head < MODEL_HEADS_VALUE; head++) {
		const uint64_t first =
			((uint64_t)row * MODEL_HEADS_VALUE + head) * DIMS;
		if (head <
		    layer->first_head) { /* a head of another rank: untouched */
			for (uint32_t dim = 0; dim < DIMS; dim++) {
				const bool is_untouched =
					test_attention_half_at(card,
							       layer->queries,
							       first + dim) ==
					0;
				test_attention_tally(
					tallies, TEST_ATTENTION_QUERIES,
					is_untouched, is_untouched,
					"a query of a head the layer does not run",
					dim);
			}
			continue;
		}
		bfloat16_values(row_inputs.heads + (uint64_t)head * DIMS, DIMS,
				reference);
		rotated(card, layer->rotation, row, layer->theta, DIMS,
			FIRST_ROTATED, reference);
		for (uint32_t dim = 0; dim < DIMS;
		     dim++) { /* a value of q in bfloat16, widened to FP16 exactly */
			const double card_value = test_attention_half_value(
				test_attention_half_at(card, layer->queries,
						       first + dim));
			test_attention_tally(
				tallies, TEST_ATTENTION_QUERIES,
				test_linear_bfloat16_nearest(card_value) ==
					test_linear_bfloat16_nearest(
						reference[dim]),
				test_linear_bfloat16_value(
					test_linear_bfloat16_nearest(
						card_value)) == card_value &&
					test_attention_within(
						card_value,
						expected_of(
							reference[dim],
							TEST_ATTENTION_BFLOAT16_BITS,
							TEST_ATTENTION_BFLOAT16_LOWEST),
						magnitudes[dim],
						TEST_ATTENTION_BFLOAT16_BITS,
						TEST_ATTENTION_BFLOAT16_LOWEST),
				"a main query", dim);
		}
	}
	if (layer->has_index != 0) {
		for (uint32_t head = 0; head < MODEL_INDEX_HEADS; head++) {
			bfloat16_values(row_inputs.index_heads +
						(uint64_t)head * INDEX_DIMS,
					INDEX_DIMS,
					reference +
						(uint64_t)head * INDEX_DIMS);
			rotated(card, layer->rotation, row, layer->theta,
				INDEX_DIMS, INDEX_FIRST_ROTATED,
				reference + (uint64_t)head * INDEX_DIMS);
			memcpy(values + (uint64_t)head * INDEX_DIMS, magnitudes,
			       INDEX_DIMS * sizeof(double));
		}
	    memcpy(magnitudes, values, INDEX_VALUES * sizeof(double));
            check_halves(card, layer->index_queries, (uint64_t)row * INDEX_VALUES, index_source_of, INDEX_VALUES,
                         TEST_ATTENTION_INDEX_QUERIES, "an index query");
        }
        if (ratio == RATIO_TWO && !emits) {
		const uint64_t slot =
			((uint64_t)session * ATTENTION_COMPRESSOR_SLOTS +
			 (uint32_t)position % ATTENTION_COMPRESSOR_SLOTS) *
			SLOT_FLOATS;
		for (uint32_t dim = 0; dim < DIMS; dim++) {
			const bool is_kept =
				test_attention_float_at(card, layer->slots,
							slot + dim) ==
					row_inputs.kv[dim] &&
				test_attention_float_at(card, layer->slots,
							slot + DIMS + dim) ==
					row_inputs.score[dim];
			test_attention_tally(tallies, TEST_ATTENTION_SLOTS,
					     is_kept, is_kept, "a slot", dim);
		}
        }
        if (!emits) {
            continue;
        }
        for (uint32_t dim = 0; dim < DIMS; dim++) {
            values[dim] = ratio == 1 ? test_linear_bfloat16_value(row_inputs.pooled_bfloat16[dim])
                                     : test_linear_bfloat16_value(test_linear_bfloat16_nearest(test_attention_pooled(
                                           partner_inputs.kv[dim], partner_inputs.score[dim], row_inputs.kv[dim],
                                           row_inputs.score[dim])));
        }
        normed(values, TEST_ATTENTION_POOL_NORM, DIMS);
        check_bfloat16(card, layer->latent, (uint64_t)row * DIMS, DIMS, TEST_ATTENTION_LATENT, "a latent");
        rotated(card, layer->rotation, row, pool_use, DIMS, FIRST_ROTATED, reference);
        const uint64_t entry = (uint64_t)session * (CONTEXT / ratio) + (uint32_t)position / ratio;
        check_halves(card, layer->pool, entry * DIMS, NULL, DIMS, TEST_ATTENTION_POOL, "a pool entry");
        bfloat16_values(row_inputs.key, INDEX_DIMS, values);
        normed(values, TEST_ATTENTION_K_NORM, INDEX_DIMS);
        rotated(card, layer->rotation, row, pool_use, INDEX_DIMS, INDEX_FIRST_ROTATED, reference);
        check_halves(card, layer->keys, entry * INDEX_DIMS, key_source_of, INDEX_DIMS, TEST_ATTENTION_KEYS, "a key");
    }
}

/* A row's inputs into a layer's input arrays at row, uploaded. */
static bool inputs_put(struct test_linear_card *card,
		       const struct attention_state_description *layer,
		       uint32_t row, const struct row_inputs *inputs)
{
    const uint32_t ratio = layer->pool_ratio;
    const struct {
	    const struct linear_range *range;
	    const void *values;
	    uint64_t byte_count; /* of one row; 0 where the layer has not the array */
    } parts[] = {
        {&layer->query_rank, inputs->query_rank, sizeof(inputs->query_rank)},
        {&layer->window_input, inputs->window, sizeof(inputs->window)},
        {&layer->heads, inputs->heads, sizeof(inputs->heads)},
        {&layer->index_heads, inputs->index_heads, layer->has_index != 0 ? sizeof(inputs->index_heads) : 0},
        {&layer->pool_input, ratio == 1 ? (const void *)inputs->pooled_bfloat16 : (const void *)inputs->kv,
         ratio == 0 ? 0 : ratio == 1 ? sizeof(inputs->pooled_bfloat16) : sizeof(inputs->kv)},
        {&layer->pool_score, inputs->score, ratio == RATIO_TWO ? sizeof(inputs->score) : 0},
        {&layer->key_input, inputs->key, ratio != 0 ? sizeof(inputs->key) : 0},
    };
    bool is_put = true;
    for (size_t part = 0; part < sizeof(parts) / sizeof(parts[0]); part++) {
        const uint64_t byte_count = parts[part].byte_count;
        is_put = is_put && (byte_count == 0 ||
                            (test_linear_card_put(card, *parts[part].range, row * byte_count, parts[part].values,
                                                  byte_count) &&
                             test_linear_card_upload(card, *parts[part].range, row * byte_count, byte_count)));
    }
    return is_put;
}

/* The norms in the weights region (shared by both layers of a case) and the frequencies, with the --break. */
static bool shared_set_aside(struct test_linear_card *card,
			     struct linear_range norms[TEST_ATTENTION_NORMS],
			     struct linear_range *frequencies_range)
{
    bool is_put = true;
    for (uint32_t norm = 0; norm < TEST_ATTENTION_NORMS; norm++) {
        uint16_t weights[QUERY_RANK];
        test_linear_values(norm_weights[norm], norm_widths[norm], WEIGHT_STREAM + norm, TEST_LINEAR_VALUES_PLAIN);
        memcpy(weights, norm_weights[norm], sizeof(uint16_t) * norm_widths[norm]);
        if (breaks("weight") && norm == TEST_ATTENTION_KV_NORM) {
            weights[BROKEN_WEIGHT] = test_linear_bfloat16_at(test_linear_bfloat16_ordinal(weights[BROKEN_WEIGHT]) + 1);
        }
        norms[norm] = test_linear_card_weights(card, sizeof(uint16_t) * norm_widths[norm]);
        is_put = is_put && test_linear_card_put(card, norms[norm], 0, weights, norms[norm].byte_count);
    }
    float words[WORDS];
    memcpy(words, frequencies, sizeof(words));
    if (breaks("frequency")) {
        uint32_t bits = 0;
        memcpy(&bits, &words[BROKEN_WORD], sizeof(bits));
        bits++;
        memcpy(&words[BROKEN_WORD], &bits, sizeof(bits));
    }
    *frequencies_range = test_linear_card_work(card, sizeof(words));
    return is_put && test_linear_card_put(card, *frequencies_range, 0, words, sizeof(words));
}

/* The planned steps through the steps' layer, each checked; then each row alone, bit for bit the same. */
static void case_run(struct synthetic_case *test_case)
{
    struct test_linear_card *const card = test_case->card;
    const struct attention_state_description *const steps =
	    &test_case->steps.layer;
    const struct attention_state_description *const alone =
	    &test_case->alone.layer;
    uint64_t sequence = 1;
    for (size_t step_index = 0; step_index < sizeof(planned_steps) / sizeof(planned_steps[0]); step_index++) {
        const struct planned_step *const step = &planned_steps[step_index];
        bool is_run = test_attention_step_put(card, steps->step, step->rows, ROWS, step->row_count, sequence++);
        for (uint32_t row = 0; row < ROWS && is_run; row++) {
            row_inputs_of(step->rows[row].session, step->rows[row].position, &row_inputs);
            is_run = inputs_put(card, steps, row, &row_inputs);
        }
        if (!is_run || !test_linear_card_run(card, test_case->steps.launches, test_case->steps.launch_count) ||
            !test_linear_card_fetch(card)) {
            return;
        }
        step_check(test_case, step);
        for (uint32_t row = 0; row < step->row_count; row++) {
            const struct test_attention_row *const own = &step->rows[row];
            const bool emits = steps->pool_ratio != 0 && ((uint32_t)own->position + 1u) % steps->pool_ratio == 0;
            row_inputs_of(own->session, own->position, &row_inputs);
            if (!test_attention_step_put(card, alone->step, own, 1, 1, sequence++) ||
                !inputs_put(card, alone, 0, &row_inputs) ||
                !test_linear_card_run(card, test_case->alone.launches, test_case->alone.launch_count) ||
                !test_linear_card_fetch(card)) {
                return;
            }
            const struct {
		    struct linear_range of_steps, of_alone;
		    uint64_t row_bytes;
	    } outputs[] = {
		    { steps->rotation, alone->rotation,
		      MODEL_ROTATION_ROW_BYTES },
		    { steps->query_normed, alone->query_normed,
		      sizeof(uint16_t) * QUERY_RANK },
		    { steps->queries, alone->queries,
		      sizeof(uint16_t) * HEAD_VALUES },
		    { steps->index_queries, alone->index_queries,
		      steps->has_index != 0 ? MODEL_INDEX_QUERIES_ROW_BYTES :
					      0 },
		    { steps->latent, alone->latent,
		      emits ? MODEL_STATE_ROW_BYTES : 0 },
	    };
	    for (size_t output = 0; output < sizeof(outputs) / sizeof(outputs[0]); output++) {
                const bool is_same =
                    outputs[output].row_bytes == 0 ||
                    test_linear_card_expect_same(card, outputs[output].of_steps, row * outputs[output].row_bytes,
                                                 outputs[output].of_alone, 0, outputs[output].row_bytes,
                                                 "a row alone gives the output it gave in its step");
                test_attention_tally(tallies, TEST_ATTENTION_ALONE, is_same, is_same, "an output of a row", output);
            }
        }
    }
    const struct {
	    struct linear_range of_steps, of_alone;
    } states[] = {{steps->ring, alone->ring}, {steps->slots, alone->slots}, {steps->pool, alone->pool},
                  {steps->keys, alone->keys}};
    for (size_t state = 0; state < sizeof(states) / sizeof(states[0]); state++) {
        const bool is_same = states[state].of_steps.byte_count == 0 ||
                             test_linear_card_expect_same(card, states[state].of_steps, 0, states[state].of_alone, 0,
                                                          states[state].of_steps.byte_count,
                                                          "the state of the rows alone is that of the steps");
        test_attention_tally(tallies, TEST_ATTENTION_ALONE, is_same, is_same, "a state array", state);
    }
}

/* The rotation step alone at large positions, against the reference: a step of no row checks the rotation only. */
static void large_positions_run(struct test_linear_card *card, const struct test_attention_layer *layer)
{
    struct planned_step step = {0, {{0, 0}}};
    const struct synthetic_case rotation_only = {card, *layer, *layer, {0, 0, 0, 0}};
    memcpy(step.rows, large_positions, sizeof(step.rows));
    if (test_attention_step_put(card, layer->layer.step, step.rows, ROWS, 0, 0) &&
        test_linear_card_run(card, layer->launches, 1) && test_linear_card_fetch(card)) {
        step_check(&rotation_only, &step);
    }
}

static void card_run(uint32_t card_index)
{
    struct test_linear_card card;
    struct attention_state_kernels kernels;
    struct linear_range norms[TEST_ATTENTION_NORMS];
    struct linear_range frequencies_range = { 0, 0 };
    if (!test_linear_card_open(&card, card_index, WORK_CAPACITY,
			       WEIGHTS_CAPACITY) ||
	!test_linear_expect_status(
		attention_state_kernels_prepare(&backend_cuda_value, card.card,
						&kernels),
		STATUS_OK, "the kernels of the attention state are prepared")) {
	    test_linear_card_close(&card);
	    return;
    }
    for (size_t kind = 0; kind < sizeof(synthetic_kinds) / sizeof(synthetic_kinds[0]); kind++) {
        struct synthetic_case test_case;
        memset(&test_case, 0, sizeof(test_case));
        test_case.card = &card;
        test_case.kind = synthetic_kinds[kind];
        test_linear_context("card %" PRIu32 ", layer of theta %" PRIu32 ", indexer %" PRIu32 ", pool ratio %" PRIu32
                            ", heads from %" PRIu32, card_index, test_case.kind.theta, test_case.kind.has_index,
                            test_case.kind.pool_ratio, test_case.kind.first_head);
        test_linear_card_clear(&card);
        if (shared_set_aside(&card, norms, &frequencies_range) &&
            test_attention_layer_set_aside(&card, &kernels, &test_case.steps, test_case.kind, ROWS, SESSIONS,
                                           CONTEXT, norms, frequencies_range) &&
            test_attention_layer_set_aside(&card, &kernels, &test_case.alone, test_case.kind, 1, SESSIONS, CONTEXT,
                                           norms, frequencies_range) &&
            test_linear_card_load(&card)) {
            case_run(&test_case);
            if (kind == 0) {
                large_positions_run(&card, &test_case.steps);
            }
        }
    }
    test_linear_card_close(&card);
}

/* --trap: one case, in a layer of ratio 2 and one row, on one card. */
static void trap_run(uint32_t card_index, const struct trap_case *trap)
{
    struct test_linear_card card;
    struct attention_state_kernels kernels;
    struct linear_range norms[TEST_ATTENTION_NORMS];
    struct linear_range frequencies_range = { 0, 0 };
    struct test_attention_layer layer;
    test_linear_context("card %" PRIu32 ", the trap case %s", card_index, trap->name);
    if (test_linear_card_open(&card, card_index, WORK_CAPACITY,
			      WEIGHTS_CAPACITY) &&
	test_linear_expect_status(
		attention_state_kernels_prepare(&backend_cuda_value, card.card,
						&kernels),
		STATUS_OK, "the kernels of the attention state are prepared") &&
	shared_set_aside(&card, norms, &frequencies_range) &&
	test_attention_layer_set_aside(&card, &kernels, &layer, trap_kind, 1,
				       SESSIONS, CONTEXT, norms,
				       frequencies_range) &&
	test_linear_card_put(&card, layer.layer.heads, 0,
			     &trap->first_main_query,
			     sizeof(trap->first_main_query)) &&
	test_linear_card_load(&card) &&
	test_attention_step_put(&card, layer.layer.step, &trap->row, 1,
				trap->row_count, 1)) {
	    (void)test_linear_expect_status(
		    backend_cuda_value.launch_run(card.card, layer.launches,
						  layer.launch_count),
		    STATUS_OK, "the launches are queued");
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
    size_t trap = 0;
    const bool shows_a_trap = argument_count == TRAP_ARGUMENTS && strcmp(arguments[1], TRAP_FLAG) == 0;
    const unsigned long trap_card = shows_a_trap ? strtoul(arguments[FLAG_ARGUMENTS - 1], &card_end, DECIMAL) : 0;
    while (shows_a_trap && trap < sizeof(trap_cases) / sizeof(trap_cases[0]) &&
           strcmp(arguments[TRAP_ARGUMENTS - 1], trap_cases[trap].name) != 0) {
        trap++;
    }
    broken = argument_count == FLAG_ARGUMENTS && strcmp(arguments[1], BREAK_FLAG) == 0 ? arguments[FLAG_ARGUMENTS - 1]
                                                                                        : NULL;
    if ((argument_count != 1 && !shows_a_trap && broken == NULL) ||
        (shows_a_trap && (card_end == arguments[FLAG_ARGUMENTS - 1] || *card_end != '\0' ||
                          trap == sizeof(trap_cases) / sizeof(trap_cases[0]))) ||
        (broken != NULL && !breaks("weight") && !breaks("frequency") && !breaks("rounding"))) {
        (void)fprintf(stderr, "usage: attention_state_synthetic_card_test [%s weight|frequency|rounding | %s <card> "
                              "slot|negative|beyond|empty|more|overflow]\n", BREAK_FLAG, TRAP_FLAG);
        return EXIT_FAILURE;
    }
    (void)alarm(TEST_TIME_LIMIT_SECONDS);
    frequencies_make();
    placements_make();
    if (broken != NULL) {
        (void)printf("BREAK %s: the test must fail\n", broken);
    }
    if (!test_linear_expect_status(backend_cuda_value.card_count(&card_count),
				   STATUS_OK, "the cards are counted") ||
	!test_linear_expect(
		card_count > 0 && card_count <= BACKEND_MAXIMUM_CARDS &&
			trap_card < card_count,
		"the machine has cards for the test, and the card asked for")) {
	    return EXIT_FAILURE;
    }
    for (uint32_t card_index = 0; !shows_a_trap && card_index < card_count; card_index++) {
        card_run(card_index);
    }
    if (shows_a_trap && CHECKING_HOOKS_ENABLED) {
	    trap_run((uint32_t)trap_card, &trap_cases[trap]);
    }
    test_attention_tallies_print(tallies, "attention_state_synthetic_card_test");
    if (test_linear_failures() != 0) {
        (void)fprintf(stderr, "attention_state_synthetic_card_test FAILED: %d failures\n", test_linear_failures());
        return EXIT_FAILURE;
    }
    (void)printf("attention_state_synthetic_card_test passed on %" PRIu32
		 " cards%s\n",
		 card_count,
		 !shows_a_trap ? "" :
		 CHECKING_HOOKS_ENABLED ?
				 " (a trap shown)" :
				 " (release modules: no device check to show)");
    return EXIT_SUCCESS;
}
