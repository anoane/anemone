/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * attention_state_reference.c: the host reference of attention_state_reference.h. Test support: no part of the engine.
 *
 * Contract: attention_state_reference.h. What it leaves open is decided here:
 *   - A double's fields are read from its bits; a power of two is made in them. Rounding a double to a whole number
 *     adds and takes away 2^52, which the default rounding of the host does to nearest, a tie to even.
 *   - The cosine and sine: the angle less the nearest whole number of turns, 2 pi in two parts, the first of 31 bits
 *     so that turns times it is exact below 2^22 turns: what the turns taken away cost is then under 2^-60 of a
 *     radian, behind the one rounding of the reduced angle; then 40 terms of each series.
 */
#include "attention_state_reference.h"

#include <math.h>
#include <string.h>

#define DOUBLE_EXPONENT_SHIFT 52u
#define DOUBLE_EXPONENT_MASK 0x7FFull
#define DOUBLE_EXPONENT_BIAS 1023
#define DOUBLE_LOWEST_EXPONENT (-1022)
#define TWO_TO_52 4503599627370496.0
#define NEWTON_STEPS 8u                         /* from within a factor of 2: far past the double's 53 bits */
#define EXPONENTIAL_FLOOR (-700.0)
#define EXPONENTIAL_TERMS 24u                   /* of an argument within half of ln 2 */
#define NATURAL_LOG_2 0.69314718055994530942
#define HALF 0.5
#define TWO_PI 6.28318530717958647693
#define TWO_PI_HIGH 6.2831853069365024566650390625  /* 2 pi to 31 bits */
#define TWO_PI_LOW 2.43084020260247689973e-10       /* 2 pi less it, to the nearest double */
#define SERIES_TERMS 40u
#define SERIES_PHASES 4u                        /* the powers of the angle go to cos +, sin +, cos -, sin - */
#define COSINE_MINUS_PHASE 2u
#define ROOT_OF_A_POWER 2                       /* the root of 2^e is 2^(e / 2) */
#define DOUBLE_INFINITY_BITS 0x7FF0000000000000ull
#define DOUBLE_NOT_A_NUMBER_BITS 0x7FF8000000000000ull
#define NORM_EPSILON 1e-20
#define HALF_SIGN_BIT 0x8000u
#define HALF_MAGNITUDE_MASK 0x7FFFu
#define HALF_INFINITY 0x7C00u
#define HALF_FRACTION_BITS 10
#define HALF_EXPONENT_BIAS 15
#define HALF_SMALLEST_EXPONENT (-14)
#define HALF_HIDDEN_ONE 0x400u
#define HALF_FRACTION_MASK 0x3FFu
#define HALF_EXPONENT_SHIFT 10u
#define HALF_ROUNDS_TO_INFINITY 65520.0
#define FLOAT32_MAGNITUDE_MASK 0x7FFFFFFFu
#define FLOAT32_SIGN_BIT 0x80000000u
#define BFLOAT16_SHIFT 16u
#define INDEX_DIMS 128u
#define INDEX_HEAD_TILES 2u
#define HEADS_PER_TILE 16u
#define STEPS 8u                                /* of 16 dims */
#define STEP_DIMS 16u
#define LANE_GROUPS 8u
#define LANE_PLACES 4u
#define QUERY_REGISTERS 4u
#define REGISTER_HALVES 2u
#define VALUES_PER_STEP_RUN 4u                  /* of a key: dims 16 j + 2 t + {0, 1, 8, 9} */

static const uint32_t key_dim_of_value[VALUES_PER_STEP_RUN] = {0, 1, 8, 9};

/* The window part's numbers (attention_state_records.h): a row of 512 dims as 256 pairs, 128 threads of 4 warps. */
#define WINDOW_DIMS 512u
#define WINDOW_PAIRS 256u
#define STATE_THREADS 128u
#define WARP_LANES 32u
#define STATE_WARPS (STATE_THREADS / WARP_LANES)
#define FIRST_ROTATED_PAIR 224u                 /* dim 448 */
#define PAIR_VALUES 2u

static double two_to(int32_t exponent)          /* for a normal double's exponent */
{
    const uint64_t bits = (uint64_t)(exponent + DOUBLE_EXPONENT_BIAS) << DOUBLE_EXPONENT_SHIFT;
    double value = 0.0;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static int32_t exponent_of(double value)       /* of a normal double; the lowest for a subnormal one */
{
    uint64_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    const int32_t exponent = (int32_t)((bits >> DOUBLE_EXPONENT_SHIFT) & DOUBLE_EXPONENT_MASK) - DOUBLE_EXPONENT_BIAS;
    return exponent < DOUBLE_LOWEST_EXPONENT ? DOUBLE_LOWEST_EXPONENT : exponent;
}

static double whole_nearest(double value)      /* of a magnitude below 2^52; the sign kept */
{
    const double magnitude = value < 0.0 ? -value : value;
    const double whole = (magnitude + TWO_TO_52) - TWO_TO_52;
    return value < 0.0 ? -whole : whole;
}

double test_attention_square_root(double value)
{
    double root = two_to(exponent_of(value) / ROOT_OF_A_POWER);
    for (uint32_t step = 0; step < NEWTON_STEPS; step++) {
        root = (root + value / root) * HALF;
    }
    return root;
}

double test_attention_exponential(double value)
{
    if (value < EXPONENTIAL_FLOOR) {
        return 0.0;
    }
    const double doublings = whole_nearest(value / NATURAL_LOG_2);
    const double rest = value - doublings * NATURAL_LOG_2;
    double term = 1.0, sum = 0.0;
    for (uint32_t power = 0; power < EXPONENTIAL_TERMS; power++) {
        sum += term;
        term = term * rest / (double)(power + 1u);
    }
    return sum * two_to((int32_t)doublings);
}

void test_attention_cosine_sine(double angle, double *cosine_out, double *sine_out)
{
    const double turns = whole_nearest(angle / TWO_PI);
    const double reduced = (angle - turns * TWO_PI_HIGH) - turns * TWO_PI_LOW;
    double term = 1.0, cosine = 0.0, sine = 0.0;
    for (uint32_t power = 0; power < SERIES_TERMS; power++) {
        const uint32_t phase = power % SERIES_PHASES;
        if (phase == 0 || phase == COSINE_MINUS_PHASE) {
            cosine += phase == 0 ? term : -term;
        } else {
            sine += phase == 1 ? term : -term;
        }
        term = term * reduced / (double)(power + 1u);
    }
    *cosine_out = cosine;
    *sine_out = sine;
}

double test_attention_half_value(uint16_t bits)
{
    const uint32_t fraction = bits & HALF_FRACTION_MASK;
    const uint32_t exponent_field = (bits & HALF_MAGNITUDE_MASK) >> HALF_EXPONENT_SHIFT;
    double magnitude = 0.0;
    if (exponent_field == (HALF_INFINITY >> HALF_EXPONENT_SHIFT)) {
        const uint64_t special_bits = fraction == 0 ? DOUBLE_INFINITY_BITS : DOUBLE_NOT_A_NUMBER_BITS;
        memcpy(&magnitude, &special_bits, sizeof(magnitude));
    } else if (exponent_field == 0) {
        magnitude = (double)fraction * two_to(HALF_SMALLEST_EXPONENT - HALF_FRACTION_BITS);
    } else {
        magnitude = (double)(fraction | HALF_HIDDEN_ONE) *
                    two_to((int32_t)exponent_field - HALF_EXPONENT_BIAS - HALF_FRACTION_BITS);
    }
    return (bits & HALF_SIGN_BIT) != 0 ? -magnitude : magnitude;
}

uint16_t test_attention_half_nearest(double value)
{
    if (value != value) {
        return TEST_ATTENTION_HALF_NOT_A_NUMBER;
    }
    const bool is_negative = value < 0.0 || (value == 0.0 && 1.0 / value < 0.0);
    const double magnitude = value < 0.0 ? -value : value;
    const uint32_t sign = is_negative ? HALF_SIGN_BIT : 0;
    if (magnitude >= HALF_ROUNDS_TO_INFINITY) {
        return (uint16_t)(sign | HALF_INFINITY);
    }
    const int32_t exponent = exponent_of(magnitude);
    const bool is_normal = exponent >= HALF_SMALLEST_EXPONENT;
    const int32_t place = (is_normal ? exponent : HALF_SMALLEST_EXPONENT) - HALF_FRACTION_BITS;
    /* in places: 1,024 to 2,048 for a normal number, 0 to 1,024 below; a carry is the next binade's first */
    const uint32_t places = (uint32_t)whole_nearest(magnitude * two_to(-place));
    const uint32_t magnitude_bits =
        is_normal ? ((uint32_t)(exponent + HALF_EXPONENT_BIAS) << HALF_EXPONENT_SHIFT) + places - HALF_HIDDEN_ONE
                  : places;
    return (uint16_t)(sign | magnitude_bits);
}

int32_t test_attention_half_ordinal(uint16_t bits)
{
    const int32_t magnitude = (int32_t)(bits & HALF_MAGNITUDE_MASK);
    return (bits & HALF_SIGN_BIT) != 0 ? -magnitude : magnitude;
}

int32_t test_attention_float_ordinal(float value)
{
    uint32_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    const int32_t magnitude = (int32_t)(bits & FLOAT32_MAGNITUDE_MASK);
    return (bits & FLOAT32_SIGN_BIT) != 0 ? -magnitude : magnitude;
}

double test_attention_last_place(double value, int32_t fraction_bits, int32_t smallest_exponent)
{
    const int32_t exponent = exponent_of(value < 0.0 ? -value : value);
    return two_to((exponent > smallest_exponent ? exponent : smallest_exponent) - fraction_bits);
}

void test_attention_norm(const double *values, const uint16_t *weights, uint32_t width, double *normed_out)
{
    double squares = 0.0;
    for (uint32_t index = 0; index < width; index++) {
        squares += values[index] * values[index];
    }
    const double factor = 1.0 / test_attention_square_root(squares / (double)width + NORM_EPSILON);
    for (uint32_t index = 0; index < width; index++) {
        const uint32_t weight_bits = (uint32_t)weights[index] << BFLOAT16_SHIFT;
        float weight = 0.0f;
        memcpy(&weight, &weight_bits, sizeof(weight));
        normed_out[index] = (double)weight * values[index] * factor;
    }
}

void test_attention_rotate(double *real, double *imaginary, double cosine, double sine)
{
    const double turned_real = *real * cosine - *imaginary * sine;
    *imaginary = *real * sine + *imaginary * cosine;
    *real = turned_real;
}

double test_attention_pooled(double first_kv, double first_score, double kv, double score)
{
    const double largest = first_score > score ? first_score : score;
    const double first_weight = test_attention_exponential(first_score - largest);
    const double weight = test_attention_exponential(score - largest);
    return (first_kv * first_weight + kv * weight) / (first_weight + weight);
}

void test_attention_layout_index_queries(const uint16_t *halves, uint16_t *placed_out)
{
    uint32_t placed = 0;
    for (uint32_t tile = 0; tile < INDEX_HEAD_TILES; tile++) {
        for (uint32_t step = 0; step < STEPS; step++) {
            for (uint32_t lane_group = 0; lane_group < LANE_GROUPS; lane_group++) {
                for (uint32_t lane_place = 0; lane_place < LANE_PLACES; lane_place++) {
                    for (uint32_t operand_register = 0; operand_register < QUERY_REGISTERS; operand_register++) {
                        for (uint32_t half = 0; half < REGISTER_HALVES; half++, placed++) {
                            const uint32_t head = tile * HEADS_PER_TILE + lane_group +
                                                  LANE_GROUPS * (operand_register % REGISTER_HALVES);
                            const uint32_t dim = step * STEP_DIMS + REGISTER_HALVES * lane_place +
                                                 LANE_GROUPS * (operand_register / REGISTER_HALVES) + half;
                            placed_out[placed] = halves[head * INDEX_DIMS + dim];
                        }
                    }
                }
            }
        }
    }
}

static float float_of_bits(uint32_t bits)
{
    float value = 0.0f;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

float test_attention_window_operand(const uint16_t *window_input)
{
    float warp_sums[STATE_WARPS];
    for (uint32_t warp = 0; warp < STATE_WARPS; warp++) {
        float lanes[WARP_LANES];
        for (uint32_t lane = 0; lane < WARP_LANES; lane++) {
            float partial = 0.0f;
            for (uint32_t pair = warp * WARP_LANES + lane; pair < WINDOW_PAIRS; pair += STATE_THREADS) {
                for (uint32_t half = 0; half < PAIR_VALUES; half++) {
                    const float value = float_of_bits((uint32_t)window_input[(size_t)pair * PAIR_VALUES + half]
                                                      << BFLOAT16_SHIFT);
                    partial = partial + value * value;
                }
            }
            lanes[lane] = partial;
        }
        for (uint32_t distance = WARP_LANES >> 1u; distance != 0; distance >>= 1u) {
            float next[WARP_LANES];
            for (uint32_t lane = 0; lane < WARP_LANES; lane++) {
                next[lane] = lanes[lane] + lanes[lane ^ distance];
            }
            memcpy(lanes, next, sizeof(next));
        }
        warp_sums[warp] = lanes[0];
    }
    float sum = 0.0f;
    for (uint32_t warp = 0; warp < STATE_WARPS; warp++) {
        sum = sum + warp_sums[warp];
    }
    return sum * (1.0f / (float)WINDOW_DIMS) + (float)NORM_EPSILON;
}

void test_attention_window_row(const uint16_t *window_input, const uint16_t *weights, float factor,
                               const float *angles, uint16_t *ring_row_out)
{
    float row[WINDOW_DIMS];
    for (uint32_t dim = 0; dim < WINDOW_DIMS; dim++) {
        const float value = float_of_bits((uint32_t)window_input[dim] << BFLOAT16_SHIFT);
        const float weight = float_of_bits((uint32_t)weights[dim] << BFLOAT16_SHIFT);
        row[dim] = weight * (value * factor);
    }
    for (uint32_t pair = FIRST_ROTATED_PAIR; pair < WINDOW_PAIRS; pair++) {
        const size_t place = (size_t)pair * PAIR_VALUES, angle = (size_t)(pair - FIRST_ROTATED_PAIR) * PAIR_VALUES;
        const float first = row[place], second = row[place + 1u];
        const float cosine = angles[angle], sine = angles[angle + 1u];
        row[place] = fmaf(first, cosine, -(second * sine));
        row[place + 1u] = fmaf(second, cosine, first * sine);
    }
    for (uint32_t dim = 0; dim < WINDOW_DIMS; dim++) {
        ring_row_out[dim] = test_attention_half_nearest((double)row[dim]);
    }
}

void test_attention_layout_key(const uint16_t *halves, uint16_t *placed_out)
{
    uint32_t placed = 0;
    for (uint32_t run = 0; run < LANE_PLACES; run++) {
        for (uint32_t step = 0; step < STEPS; step++) {
            for (uint32_t value = 0; value < VALUES_PER_STEP_RUN; value++, placed++) {
                placed_out[placed] = halves[step * STEP_DIMS + REGISTER_HALVES * run + key_dim_of_value[value]];
            }
        }
    }
}
