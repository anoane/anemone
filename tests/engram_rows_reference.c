/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * engram_rows_reference.c: the host's references of the n-gram layer's kernels. Contract: engram_rows_reference.h.
 */
#include "engram_rows_reference.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

#include "linear_card_fixture.h"
#include "linear_reference.h"
#include "weight_format_reference.h"

#define HIDDEN MODEL_HIDDEN
#define STREAMS MODEL_STREAMS
#define STREAM_VALUES 20480u                    /* of a row: 4 streams of 5,120 */
#define CASES TEST_ENGRAM_ROWS_GATE_CASE_COUNT
#define CODE_COUNT 256u
#define CODE_STEP 167u                          /* odd: the places of a row give every byte once */
#define SCALE_STEP 37u                          /* odd: at a block position 256 consecutive rows give every code */
#define SCALES_PER_ROW (ENGRAM_ROWS_VALUES / ENGRAM_ROWS_VALUES_PER_SCALE)
#define BLOCK_CODE_STEP (CODE_COUNT / SCALES_PER_ROW)   /* the eight codes of a row: 32 apart */
#define BYTE_PER_CODE 2u                        /* even: a row still holds every byte once */
#define BFLOAT16_SIGN 0x8000u
#define BFLOAT16_NOT_A_NUMBER 0x7FC0u
#define BFLOAT16_INFINITY 0x7F80u
#define BFLOAT16_EXPONENT_SHIFT 7u
#define BFLOAT16_FRACTIONS 128u
#define BFLOAT16_ONE_FIELD 127u                 /* the exponent field of 1 */
#define FRACTION_SPAN_BINADES 8u                /* the value's upper half: every fraction in eight binades */
#define FRACTION_SPAN_LOWEST 4u                 /* from 2^-4 */
#define HASH_FRACTION_SHIFT 8u                  /* the bits of a hash that give a fraction, and a sign */
#define HASH_SIGN_SHIFT 16u
#define PLAIN_LOWEST (-2)                       /* exponents of plain numbers: 2^-2 to below 2^1 */
#define PLAIN_HIGHEST 0
#define VALUE_LOWEST (-3)
#define VALUE_HIGHEST (-1)
#define WEIGHT_LOWEST (-1)
#define WEIGHT_HIGHEST 0
#define TINY_SHIFT 40                           /* keys of 2^-40: their mean square, 2^-80, is far below epsilon */
#define LARGE_SHIFT 20                          /* 2^20: squares that add up to 2^52, far from an infinity */
#define QUERY_WEIGHT_STREAM 1u                  /* of the hash */
#define KEY_WEIGHT_STREAM 2u
#define VALUE_STREAM 16u
#define STREAM_STREAM 64u
#define UNIT_ROUNDOFF 5.9604644775390625e-8     /* 2^-24: a float32 rounding's relative error */
#define SUBNORMAL_ROUNDOFF 7.006492321624085e-46   /* 2^-150: its absolute error below the normal numbers */
#define LONGEST_CHAIN 32u                       /* additions a term of a gate sum meets: 9 + 5 + 15, and room */
#define STEP_ROUNDINGS 3u                       /* of a term (q k, h w, then key), of a mean (its factor, product and
                                                   epsilon), of the dot (D rstd, the scale, their product) and of the
                                                   gate (1 + e, the division, and room) */
#define FUNCTION_LAST_PLACES 2.0                /* the card's rsqrtf and expf */
#define UNITS_PER_LAST_PLACE 2.0                /* a last place is at most twice the unit roundoff */
#define CLAMP 1.0e-6                            /* model.py:341: the card's is its float32 */
#define SQUARE_ROOT_STEPS 8u                    /* Newton's steps from within a factor of 2: far past 2^-52 */
#define ONE_HALF 0.5
#define TWO 2.0

_Static_assert(STREAM_VALUES == STREAMS * HIDDEN, "a row's streams");

uint16_t test_engram_rows_value(uint32_t code, uint32_t scale_code)
{
    /* the product of an E4M3 number (four significant bits) and a power of two is a float32 or beyond the largest */
    const double product = test_weight_format_e4m3_value(code) * test_weight_format_e8m0_value(scale_code);
    return test_linear_bfloat16_nearest((double)(float)product);
}

double test_engram_rows_square_root(double value)
{
    if (value == 0.0 || isinf(value)) {
        return value;                           /* where the steps below would not end, or not hold */
    }
    double root = 1.0;
    while (root * root > value * TWO) {
        root *= ONE_HALF;
    }
    while (root * root * TWO < value) {
        root *= TWO;
    }
    for (uint32_t step = 0; step < SQUARE_ROOT_STEPS; step++) {
        root = ONE_HALF * (root + value / root);
    }
    return root;
}

/* The linear's reference of an expert's hidden step is s / (1 + e^-s) with an up of 1, no limit and a weight of 1. */
double test_engram_rows_sigmoid(double value)
{
    return test_linear_hidden_value(value, 1.0, (double)INFINITY, 1.0) / value;
}

/* Of a mean square: its relative error on the card, from the sum's chain, the subnormal floor and three roundings. */
static double mean_relative_error(double sum, double absolute_floor)
{
	const double mean = sum / (double)HIDDEN + (double)MODEL_NORM_EPSILON;
	return ((double)(LONGEST_CHAIN + 1u) * UNIT_ROUNDOFF * sum /
			(double)HIDDEN +
		absolute_floor) /
		       mean +
	       (double)STEP_ROUNDINGS * UNIT_ROUNDOFF;
}

struct test_engram_rows_gate test_engram_rows_gate_of(const uint16_t *stream, const uint16_t *key,
                                                      const uint16_t *query_weight, const uint16_t *key_weight)
{
    double squares = 0.0;
    double key_squares = 0.0;
    double dot_sum = 0.0;
    double dot_magnitude = 0.0;
    struct test_engram_rows_gate gate = {0.0, 0.0, 0.0, false};
    for (uint32_t place = 0; place < HIDDEN; place++) {
        const double h = test_linear_bfloat16_value(stream[place]);
        const double key_value = test_linear_bfloat16_value(key[place]);
        const double term = h * (test_linear_bfloat16_value(query_weight[place]) *
                                 test_linear_bfloat16_value(key_weight[place])) * key_value;
        squares += h * h;
        key_squares += key_value * key_value;
        dot_sum += term;
        dot_magnitude += term < 0.0 ? -term : term;
    }
    if (isnan(squares) || isnan(key_squares) || isnan(dot_sum)) {
        gate.is_not_a_number = true;
        return gate;
    }
    const double absolute_floor = (double)(HIDDEN * (LONGEST_CHAIN + STEP_ROUNDINGS)) * SUBNORMAL_ROUNDOFF;
    const double epsilon = (double)MODEL_NORM_EPSILON;
    const double rstd = 1.0 / test_engram_rows_square_root(squares / (double)HIDDEN + epsilon) /
                        test_engram_rows_square_root(key_squares / (double)HIDDEN + epsilon);
    const double rstd_relative =
        (mean_relative_error(squares, absolute_floor) + mean_relative_error(key_squares, absolute_floor)) / TWO +
        (TWO * FUNCTION_LAST_PLACES * UNITS_PER_LAST_PLACE + 1.0) * UNIT_ROUNDOFF;
    const double scale = 1.0 / test_engram_rows_square_root((double)HIDDEN);
    const double dot = dot_sum * rstd * scale;
    const double magnitude = dot < 0.0 ? -dot : dot;
    /* every term 0: the card's sums are +0 exactly, and so its dot */
    const double dot_error =
        dot_magnitude == 0.0
            ? 0.0
            : ((double)(LONGEST_CHAIN + STEP_ROUNDINGS) * UNIT_ROUNDOFF * dot_magnitude + absolute_floor) * rstd *
                      scale * (1.0 + rstd_relative) +
                  magnitude * (rstd_relative + (double)STEP_ROUNDINGS * UNIT_ROUNDOFF);
    const double clamp = (double)(float)CLAMP;
    const double root = test_engram_rows_square_root(magnitude < clamp ? clamp : magnitude);
    const double lowest_magnitude = magnitude - dot_error < clamp ? clamp : magnitude - dot_error;
    const double root_error =
        dot_error / (TWO * test_engram_rows_square_root(lowest_magnitude)) + root * UNIT_ROUNDOFF;
    const double signed_root = dot < 0.0 ? -root : root;
    const bool sign_is_known = magnitude > dot_error || dot_magnitude == 0.0;
    const double gate_relative = (FUNCTION_LAST_PLACES * UNITS_PER_LAST_PLACE + STEP_ROUNDINGS) * UNIT_ROUNDOFF;
    const double highest = test_engram_rows_sigmoid(sign_is_known ? signed_root + root_error : root + root_error) *
                           (1.0 + gate_relative);
    gate.central = test_engram_rows_sigmoid(signed_root);
    gate.lowest = test_engram_rows_sigmoid(sign_is_known ? signed_root - root_error : -(root + root_error)) *
                  (1.0 - gate_relative);
    gate.highest = highest > 1.0 ? 1.0 : highest;
    return gate;
}

/* The float32 nearest a gate from 0 to 1 on the side asked: at or below it, or at or above it. */
static float float_toward(double value, bool is_upward)
{
    float rounded = (float)value;
    uint32_t bits = 0;
    memcpy(&bits, &rounded, sizeof(bits));
    bits = is_upward ? bits + ((double)rounded < value ? 1u : 0u) : bits - ((double)rounded > value ? 1u : 0u);
    memcpy(&rounded, &bits, sizeof(bits));
    return rounded;
}

static float float_of(uint16_t bits)
{
    return (float)test_linear_bfloat16_value(bits);
}

/* h + gate value as the kernel computes it: a float32 product, a float32 sum, rounded to bfloat16. */
static uint16_t output_of(float h, float gate, float value)
{
    const float product = gate * value;
    return test_linear_bfloat16_nearest((double)(h + product));
}

uint64_t test_engram_rows_gate_outside(const struct test_engram_rows_gate *gate, const uint16_t *stream,
                                       const uint16_t *value, const uint16_t *output, uint64_t *equal_count,
                                       uint32_t *first_outside)
{
    const float lowest = float_toward(gate->lowest, false);
    const float highest = float_toward(gate->highest, true);
    const float central = (float)gate->central;
    uint64_t outside = 0;
    for (uint32_t place = 0; place < HIDDEN; place++) {
        const float h = float_of(stream[place]);
        const float v = float_of(value[place]);
        const uint16_t at_lowest = output_of(h, lowest, v);
        const uint16_t at_highest = output_of(h, highest, v);
        const int32_t first = test_linear_bfloat16_ordinal(at_lowest);
        const int32_t second = test_linear_bfloat16_ordinal(at_highest);
        const int32_t card = test_linear_bfloat16_ordinal(output[place]);
        const bool is_missing = gate->is_not_a_number || isnan(float_of(at_lowest)) || isnan(float_of(at_highest));
        const bool is_inside = is_missing ? output[place] == BFLOAT16_NOT_A_NUMBER
                                          : !isnan(float_of(output[place])) &&
                                                card >= (first < second ? first : second) &&
                                                card <= (first < second ? second : first);
        const uint16_t central_bits = gate->is_not_a_number ? BFLOAT16_NOT_A_NUMBER : output_of(h, central, v);
        *equal_count += output[place] == central_bits ? 1u : 0u;
        if (!is_inside && outside == 0) {
            *first_outside = place;
        }
        outside += is_inside ? 0u : 1u;
    }
    return outside;
}

/* Block b of table row t has the code c = 37 t + 32 b, and place p of the row the byte 167 p + 2 c (modulo 256). A
 * row holds every byte once. In 256 consecutive rows a code has one block at each of the eight positions, together
 * the 256 places of a row, so their bytes are every byte once: each pair of a byte and a code appears once. */
void test_engram_rows_rows_fill(uint8_t *rows_out, uint32_t table_row_count)
{
    for (uint32_t table_row = 0; table_row < table_row_count; table_row++) {
	    uint8_t *const row =
		    &rows_out[(size_t)table_row * ENGRAM_ROWS_ROW_BYTES];
	    for (uint32_t block = 0; block < SCALES_PER_ROW; block++) {
		    row[ENGRAM_ROWS_VALUES + block] =
			    (uint8_t)((table_row * SCALE_STEP +
				       block * BLOCK_CODE_STEP) %
				      CODE_COUNT);
	    }
	    for (uint32_t place = 0; place < ENGRAM_ROWS_VALUES; place++) {
		    const uint32_t code =
			    row[ENGRAM_ROWS_VALUES +
				place / ENGRAM_ROWS_VALUES_PER_SCALE];
		    row[place] = (uint8_t)((place * CODE_STEP +
					    code * BYTE_PER_CODE) %
					   CODE_COUNT);
	    }
    }
}

/* A bfloat16 of either sign and any fraction from the hash, its exponent from lowest to highest. */
static uint16_t bfloat16_of(uint64_t hash, int32_t lowest_exponent, int32_t highest_exponent)
{
    const uint32_t span = (uint32_t)(highest_exponent - lowest_exponent + 1);
    const uint32_t field = (uint32_t)((int32_t)BFLOAT16_ONE_FIELD + lowest_exponent + (int32_t)(hash % span));
    const uint32_t fraction = (uint32_t)(hash >> HASH_FRACTION_SHIFT) % BFLOAT16_FRACTIONS;
    const uint32_t sign = (hash >> HASH_SIGN_SHIFT & 1u) != 0 ? BFLOAT16_SIGN : 0;
    return (uint16_t)(sign | field << BFLOAT16_EXPONENT_SHIFT | fraction);
}

/* The bfloat16 of the same number times 2^shift: its field moved (no case leaves the normal numbers); 0 stays. */
static uint16_t bfloat16_shifted(uint16_t bits, int32_t shift)
{
    const uint32_t field = ((uint32_t)bits >> BFLOAT16_EXPONENT_SHIFT) % (BFLOAT16_SIGN >> BFLOAT16_EXPONENT_SHIFT);
    return (uint16_t)(field == 0 ? bits : (int32_t)bits + shift * (int32_t)BFLOAT16_FRACTIONS);
}

/* The key of a stream in a case, from the stream's plain number, another, and the sign of q_weight k_weight there:
 * along, every term h w key of the dot is positive; against, every one negative. */
static uint16_t key_of(uint32_t gate_case, uint32_t stream, uint16_t plain, uint16_t other, uint16_t weight_sign)
{
    const uint16_t along = (uint16_t)(bfloat16_shifted(plain, -(int32_t)stream) ^ weight_sign);
    switch (gate_case) {
    case TEST_ENGRAM_ROWS_ALONG:
        return along;
    case TEST_ENGRAM_ROWS_AGAINST:
        return (uint16_t)(along ^ BFLOAT16_SIGN);
    case TEST_ENGRAM_ROWS_ZERO_KEY:
        return 0;
    case TEST_ENGRAM_ROWS_TINY_KEY:
        return bfloat16_shifted(other, -TINY_SHIFT);
    case TEST_ENGRAM_ROWS_LARGE:
        return bfloat16_shifted(other, LARGE_SHIFT);
    default:
        return other;
    }
}

void test_engram_rows_gate_fill(uint16_t *streams_out, uint16_t *projection_out, uint16_t *query_weight_out,
                                uint16_t *key_weight_out)
{
    for (uint32_t place = 0; place < STREAM_VALUES; place++) {
        query_weight_out[place] = bfloat16_of(test_linear_hash(QUERY_WEIGHT_STREAM, place), WEIGHT_LOWEST,
                                              WEIGHT_HIGHEST);
        key_weight_out[place] = bfloat16_of(test_linear_hash(KEY_WEIGHT_STREAM, place), WEIGHT_LOWEST, WEIGHT_HIGHEST);
    }
    for (uint32_t row = 0; row < CASES; row++) {
	    uint16_t *const value =
		    &projection_out[(size_t)row * ENGRAM_ROWS_PROJECTION +
				    STREAM_VALUES];
	    for (uint32_t place = 0; place < HIDDEN; place++) {
		    const uint32_t binade =
			    place / BFLOAT16_FRACTIONS % FRACTION_SPAN_BINADES;
		    value[place] =
			    place < TEST_ENGRAM_ROWS_HALF ?
				    bfloat16_of(
					    test_linear_hash(VALUE_STREAM + row,
							     place),
					    VALUE_LOWEST, VALUE_HIGHEST) :
				    (uint16_t)((BFLOAT16_ONE_FIELD + binade -
						FRACTION_SPAN_LOWEST)
						       << BFLOAT16_EXPONENT_SHIFT |
					       place % BFLOAT16_FRACTIONS);
	    }
        for (uint32_t stream = 0; stream < STREAMS; stream++) {
            uint16_t *const h = &streams_out[((size_t)row * STREAMS + stream) * HIDDEN];
	    uint16_t *const key =
		    &projection_out[(size_t)row * ENGRAM_ROWS_PROJECTION +
				    (size_t)stream * HIDDEN];
	    const uint64_t seed = STREAM_STREAM + (uint64_t)row * STREAMS + stream;
            for (uint32_t place = 0; place < HIDDEN; place++) {
                const size_t weight_place = (size_t)stream * HIDDEN + place;
                const uint16_t plain = bfloat16_of(test_linear_hash(seed, place), PLAIN_LOWEST, PLAIN_HIGHEST);
                const uint16_t other =
                    bfloat16_of(test_linear_hash(seed + STREAM_VALUES, place), PLAIN_LOWEST, PLAIN_HIGHEST);
                const bool is_zero = place >= TEST_ENGRAM_ROWS_HALF || row == TEST_ENGRAM_ROWS_ZERO_STREAM;
                h[place] = is_zero ? 0 : bfloat16_shifted(plain, row == TEST_ENGRAM_ROWS_LARGE ? LARGE_SHIFT : 0);
                key[place] = key_of(row, stream, plain, other,
                                    (uint16_t)((query_weight_out[weight_place] ^ key_weight_out[weight_place]) &
                                               BFLOAT16_SIGN));
            }
        }
    }
    streams_out[(size_t)TEST_ENGRAM_ROWS_NOT_A_NUMBER * STREAM_VALUES + TEST_ENGRAM_ROWS_NOT_A_NUMBER_PLACE] =
        BFLOAT16_NOT_A_NUMBER;
    projection_out[(size_t)TEST_ENGRAM_ROWS_NOT_A_NUMBER *
			   ENGRAM_ROWS_PROJECTION +
		   STREAM_VALUES + TEST_ENGRAM_ROWS_INFINITY_PLACE] =
	    BFLOAT16_INFINITY;
}
