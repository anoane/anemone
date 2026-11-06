/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * index_scan_reference.c: the host's index scan and selection. Contract: index_scan_reference.h. Test support: no part
 * of the engine.
 */
#include "index_scan_reference.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "anemone/model_layout.h"

#define HALF_SIGN 0x8000u
#define HALF_FRACTION_BITS 10u
#define HALF_FRACTION_MASK 0x3FFu
#define HALF_EXPONENT_MASK 0x1Fu
#define HALF_INFINITY 0x7C00u
#define HALF_NOT_A_NUMBER 0x7E00u
#define HALF_IMPLICIT_ONE 1024.0                /* 2^10 */
#define HALF_OVERFLOW 65520.0                   /* half a place above the largest finite FP16 */
#define HALF_A_PLACE 0.5
#define HALF_SMALLEST_NORMAL_EXPONENT (-14)
#define HALF_PLACE_OF_ONE (-10)                 /* the last place of a normal number, against its leading bit */
#define HALF_SUBNORMAL_PLACE (-24)
#define HALF_FIELD_OF_PLACE 25                  /* exponent field = place + 25 */
#define HALF_EXPONENT_BIAS 15
#define ORDER_FLIP_NEGATIVE 0x7FFFu
#define BFLOAT16_SHIFT 16u
#define DIMS MODEL_INDEX_HEAD_DIMS
#define HEADS MODEL_INDEX_HEADS
#define LANES 32u
#define LANE_PLACES 4u
#define STEPS 8u                                /* of 16 dims */
#define DIMS_PER_STEP 16u
#define HALF_STEP_DIMS 8u
#define HEADS_PER_TILE 16u
#define HALF_TILE_HEADS 8u
#define REGISTERS 4u                            /* of operand A in a lane */
#define PAIR 2u                                 /* values of a register; dims of a pair of a run */
#define RUN_BYTES 64u
#define HALF_BYTES 2u
#define HALF_FIELD(bits) (((bits) >> HALF_FRACTION_BITS) & HALF_EXPONENT_MASK)

double test_index_half_value(uint16_t bits)
{
    const double sign = (bits & HALF_SIGN) != 0 ? -1.0 : 1.0;
    const uint32_t field = HALF_FIELD(bits), fraction = bits & HALF_FRACTION_MASK;
    if (field == HALF_EXPONENT_MASK) {
        return fraction == 0 ? sign * (double)INFINITY : (double)NAN;
    }
    return field == 0 ? sign * ldexp((double)fraction, HALF_SUBNORMAL_PLACE)
                      : sign * ldexp(HALF_IMPLICIT_ONE + fraction, (int)field - HALF_FIELD_OF_PLACE);
}

uint16_t test_index_half_nearest(double value)
{
    const uint16_t sign = signbit(value) ? (uint16_t)HALF_SIGN : 0;
    const double magnitude = fabs(value);
    int exponent = 0;
    if (isnan(value)) {
        return HALF_NOT_A_NUMBER;
    }
    if (magnitude >= HALF_OVERFLOW || magnitude == 0.0) {
        return (uint16_t)(sign | (magnitude == 0.0 ? 0u : HALF_INFINITY));
    }
    (void)frexp(magnitude, &exponent);           /* magnitude in [2^(exponent - 1), 2^exponent) */
    const int place = exponent - 1 >= HALF_SMALLEST_NORMAL_EXPONENT ? exponent - 1 + HALF_PLACE_OF_ONE
                                                                     : HALF_SUBNORMAL_PLACE;
    const double scaled = ldexp(magnitude, -place);         /* exact: below 2^11 */
    uint32_t units = (uint32_t)scaled;
    const double remainder = scaled - units;
    units += remainder > HALF_A_PLACE || (remainder == HALF_A_PLACE && (units & 1u) != 0) ? 1u : 0u;
    /* a normal number is (1,024 + fraction) places of 2^(field - 25): units of 2,048 carry into the next field */
    const uint32_t bits = place == HALF_SUBNORMAL_PLACE
                              ? units
                              : ((uint32_t)(place + HALF_FIELD_OF_PLACE) << HALF_FRACTION_BITS) + units -
                                    (uint32_t)HALF_IMPLICIT_ONE;
    return (uint16_t)(sign | bits);
}

uint32_t test_index_order_key(uint16_t score)
{
    return (score & HALF_SIGN) != 0 ? (uint32_t)(score ^ (HALF_SIGN | ORDER_FLIP_NEGATIVE))
                                    : (uint32_t)(score ^ HALF_SIGN);
}

void test_index_place_key(const uint16_t *natural, unsigned char *tile, uint32_t key)
{
	unsigned char *const place = tile + (size_t)key * MODEL_KEY_BYTES;
	for (uint32_t run = 0; run < LANE_PLACES; run++) {
		for (uint32_t step = 0; step < STEPS; step++) {
			const uint32_t first =
				step * DIMS_PER_STEP + run * PAIR;
			const uint16_t values[REGISTERS] = {
				natural[first], natural[first + 1],
				natural[first + HALF_STEP_DIMS],
				natural[first + HALF_STEP_DIMS + 1]
			};
			memcpy(place + (size_t)run * RUN_BYTES +
				       step * sizeof(values),
			       values, sizeof(values));
		}
    }
}

void test_index_place_queries(const uint16_t *natural, unsigned char *placed)
{
    for (uint32_t head_tile = 0; head_tile < HEADS / HEADS_PER_TILE; head_tile++) {
        for (uint32_t step = 0; step < STEPS; step++) {
            for (uint32_t lane = 0; lane < LANES; lane++) {
                for (uint32_t operand_register = 0; operand_register < REGISTERS; operand_register++) {
                    const uint32_t head = head_tile * HEADS_PER_TILE + lane / LANE_PLACES +
                                          operand_register % PAIR * HALF_TILE_HEADS;
                    const uint32_t dim = step * DIMS_PER_STEP + lane % LANE_PLACES * PAIR +
                                         operand_register / PAIR * HALF_STEP_DIMS;
                    const size_t at = ((size_t)(head_tile * STEPS + step) * LANES + lane) * REGISTERS +
                                      operand_register;
                    memcpy(placed + at * PAIR * HALF_BYTES, &natural[(size_t)head * DIMS + dim],
                           (size_t)PAIR * HALF_BYTES);
                }
            }
        }
    }
}

/* The float32 sum of a key's 32 terms (relu times weight): the records' tree, or the heads one after another. */
static float float32_sum(const float terms[HEADS], bool heads_in_order)
{
    float groups[HALF_TILE_HEADS];             /* v_g of index_scan_records.h: heads g, g + 8, g + 16, g + 24 */
    float sum = terms[0];
    for (uint32_t head = 1; head < HEADS; head++) {
        sum += terms[head];
    }
    for (uint32_t group = 0; group < HALF_TILE_HEADS; group++) {
        groups[group] = (terms[group] + terms[group + HALF_TILE_HEADS]) +
                        (terms[group + HEADS_PER_TILE] + terms[group + HEADS_PER_TILE + HALF_TILE_HEADS]);
    }
    for (uint32_t width = HALF_TILE_HEADS / PAIR; width != 0; width /= PAIR) {   /* ((v_0 + v_1) + (v_2 + v_3)) + .. */
        for (uint32_t group = 0; group < width; group++) {
            groups[group] = groups[(size_t)PAIR * group] + groups[(size_t)PAIR * group + 1u];
        }
    }
    return heads_in_order ? sum : groups[0];
}

void test_index_scores(const uint16_t *query, const uint16_t *weights, const uint16_t *keys, uint32_t key_count,
                       enum test_index_sum sum, double *scores_out)
{
    static double query_values[HEADS * DIMS];
    float weight_values[HEADS];
    for (uint32_t value = 0; value < HEADS * DIMS; value++) {
        query_values[value] = test_index_half_value(query[value]);
    }
    for (uint32_t head = 0; head < HEADS; head++) {
        const uint32_t weight_bits = (uint32_t)weights[head] << BFLOAT16_SHIFT;
        memcpy(&weight_values[head], &weight_bits, sizeof(weight_values[head]));
    }
    for (uint32_t key = 0; key < key_count; key++) {
        double key_values[DIMS];
        float terms[HEADS];
        double score = 0.0;
        for (uint32_t dim = 0; dim < DIMS; dim++) {
            key_values[dim] = test_index_half_value(keys[(size_t)key * DIMS + dim]);
        }
        for (uint32_t head = 0; head < HEADS; head++) {
            double dot = 0.0;
            for (uint32_t dim = 0; dim < DIMS; dim++) {
                dot += query_values[(size_t)head * DIMS + dim] * key_values[dim];
            }
            score += dot > 0.0 ? dot * (double)weight_values[head] : 0.0;
            terms[head] = (dot > 0.0 ? (float)dot : 0.0f) * weight_values[head];
        }
        scores_out[key] = sum == TEST_INDEX_SUM_DOUBLE
                              ? score
                              : (double)float32_sum(terms, sum == TEST_INDEX_SUM_HEADS_IN_ORDER);
    }
}

struct ranked {
    uint32_t key;
    uint32_t position;
};
static bool highest_wins_ties;

static int by_rank(const void *first, const void *second)
{
    const struct ranked *const a = first, *const b = second;
    if (a->key != b->key) {
        return a->key > b->key ? -1 : 1;
    }
    if (a->position == b->position) {
        return 0;
    }
    return (a->position < b->position) != highest_wins_ties ? -1 : 1;
}

static int by_position(const void *first, const void *second)
{
    const int32_t a = *(const int32_t *)first, b = *(const int32_t *)second;
    return a < b ? -1 : a > b ? 1 : 0;
}

void test_index_select(const uint16_t *scores, uint32_t count, bool highest_on_tie, int32_t *selected_out)
{
    struct ranked *const ranks = calloc(count != 0 ? count : 1u, sizeof(*ranks));
    const uint32_t need = count < MODEL_SELECTED ? count : MODEL_SELECTED;
    for (uint32_t output = 0; output < MODEL_SELECTED; output++) {
	    selected_out[output] = -1;
    }
    if (ranks == NULL) {
        return;                                  /* the caller's comparison then fails */
    }
    for (uint32_t position = 0; position < count; position++) {
        ranks[position] = (struct ranked){test_index_order_key(scores[position]), position};
    }
    highest_wins_ties = highest_on_tie;
    qsort(ranks, count, sizeof(*ranks), by_rank);
    for (uint32_t output = 0; output < need; output++) {
        selected_out[output] = (int32_t)ranks[output].position;
    }
    qsort(selected_out, need, sizeof(*selected_out), by_position);
    free(ranks);
}
