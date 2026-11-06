/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * hyper_connection_reference.c: the host reference of the hyper-connection kernels. Contract:
 * hyper_connection_reference.h.
 */
#include "hyper_connection_reference.h"

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>

#include "anemone/hyper_connection_records.h"
#include "anemone/model_layout.h"
#include "linear_reference.h"

#define HIDDEN MODEL_HIDDEN
#define STREAMS MODEL_STREAMS
#define COMB_VALUES (STREAMS * STREAMS)
#define COMB_FIRST_MIX (MODEL_SITE_VECTORS * STREAMS)
#define EPSILON MODEL_HYPER_EPSILON
#define POST_FACTOR 2.0f
#define NATURAL_LOGARITHM_OF_TWO 0.69314718055994530942
#define ONE_HALF 0.5
#define EXPONENT_LIMIT 700.0                        /* e^x of a larger magnitude is no float32 but 0 or infinity */
#define SERIES_TERMS 24u                            /* of e^r, |r| <= ln 2 / 2: far below a double's last place */
#define DOUBLE_EXPONENT_BIAS 1023
#define DOUBLE_FRACTION_BITS 52u
#define DOUBLE_EXPONENT_MASK 0x7FFu
#define NEWTON_STEPS 8u                             /* from within a factor of two: past a double's precision */
#define SQUARE_ROOT_OF_A_POWER 2                    /* halves its exponent */
#define FLOAT32_SIGN_BIT 0x80000000u
#define VALUES_PER_THREAD HYPER_VALUES_PER_THREAD
#define WARP_LANES LINEAR_THREADS_PER_WARP
#define ROW_WARPS (HYPER_ROW_THREADS / WARP_LANES)
#define MEAN_FACTOR (1.0f / (float)HIDDEN)
enum scale_index { SCALE_PRE, SCALE_POST, SCALE_COMB };
enum line_place { FIRST_PLACE, SECOND_PLACE, THIRD_PLACE, FOURTH_PLACE };

/* 2^exponent for an exponent of a normal double. */
static double two_to(int32_t exponent)
{
    const uint64_t bits = (uint64_t)(exponent + DOUBLE_EXPONENT_BIAS) << DOUBLE_FRACTION_BITS;
    double power = 0.0;
    memcpy(&power, &bits, sizeof(power));
    return power;
}

/* e^x: x less a whole number of ln 2, by its series, times that power of two. */
static double exponential(double x)
{
    if (x != x || x > EXPONENT_LIMIT || x < -EXPONENT_LIMIT) {
        return x != x ? x : (x > 0.0 ? (double)INFINITY : 0.0);
    }
    const double doublings_wanted = x / NATURAL_LOGARITHM_OF_TWO;
    const int32_t doublings = (int32_t)(doublings_wanted < 0.0 ? doublings_wanted - ONE_HALF
                                                               : doublings_wanted + ONE_HALF);
    const double rest = x - (double)doublings * NATURAL_LOGARITHM_OF_TWO;
    double term = 1.0;
    double sum = 1.0;
    for (uint32_t term_index = 1; term_index <= SERIES_TERMS; term_index++) {
        term *= rest / (double)term_index;
        sum += term;
    }
    return sum * two_to(doublings);
}

/* The square root of a positive normal double: Newton's steps from the power of two of half its exponent. */
static double square_root(double value)
{
    uint64_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    const int32_t exponent = (int32_t)((bits >> DOUBLE_FRACTION_BITS) & DOUBLE_EXPONENT_MASK) - DOUBLE_EXPONENT_BIAS;
    double root = two_to(exponent / SQUARE_ROOT_OF_A_POWER);
    for (uint32_t step = 0; step < NEWTON_STEPS; step++) {
        root = (root + value / root) * ONE_HALF;
    }
    return root;
}

static float value_of(uint16_t bits)
{
    return (float)test_linear_bfloat16_value(bits);
}

static uint16_t bfloat16_of(float value)
{
    return test_linear_bfloat16_nearest((double)value);
}

static float sigmoid(float value)
{
    return 1.0f / (1.0f + (float)exponential(-(double)value));
}

/* (x_0 + x_2) + (x_1 + x_3) of four values stride apart. */
static float line_sum(const float *values, uint32_t stride)
{
    return (values[(size_t)FIRST_PLACE * stride] + values[(size_t)THIRD_PLACE * stride]) +
           (values[(size_t)SECOND_PLACE * stride] + values[(size_t)FOURTH_PLACE * stride]);
}

/* Each value divided by its row's sum (or its column's) plus sum_epsilon, plus quotient_epsilon: the sums of the
 * values before any of them changes. */
static void divide_by_sums(float *comb, bool is_rows, float sum_epsilon, float quotient_epsilon)
{
    float sums[STREAMS];
    for (uint32_t line = 0; line < STREAMS; line++) {
        sums[line] = is_rows ? line_sum(&comb[(size_t)line * STREAMS], 1) : line_sum(&comb[line], STREAMS);
    }
    for (uint32_t element = 0; element < COMB_VALUES; element++) {
        comb[element] = comb[element] / (sums[is_rows ? element / STREAMS : element % STREAMS] + sum_epsilon) +
                        quotient_epsilon;
    }
}

void test_hyper_sinkhorn(const float *mixes, const float *scale, const float *base, float *pre_out, float *post_out,
                         float *comb_out)
{
    float comb[COMB_VALUES];
    for (uint32_t stream = 0; stream < STREAMS; stream++) {
        pre_out[stream] = sigmoid(fmaf(mixes[stream], scale[SCALE_PRE], base[stream])) + EPSILON;
        post_out[stream] =
            POST_FACTOR * sigmoid(fmaf(mixes[STREAMS + stream], scale[SCALE_POST], base[STREAMS + stream]));
    }
    for (uint32_t row = 0; row < STREAMS; row++) {
        float largest = -INFINITY;
        for (uint32_t column = 0; column < STREAMS; column++) {
            const uint32_t element = row * STREAMS + column;
            comb[element] = fmaf(mixes[COMB_FIRST_MIX + element], scale[SCALE_COMB], base[COMB_FIRST_MIX + element]);
            largest = comb[element] > largest ? comb[element] : largest;
        }
        for (uint32_t column = 0; column < STREAMS; column++) {
            comb[row * STREAMS + column] = (float)exponential((double)(comb[row * STREAMS + column] - largest));
        }
    }
    divide_by_sums(comb, true, 0.0f, EPSILON);
    divide_by_sums(comb, false, EPSILON, 0.0f);
    for (uint32_t iteration = 1; iteration < MODEL_SINKHORN_ITERATIONS;
	 iteration++) {
	    divide_by_sums(comb, true, EPSILON, 0.0f);
	    divide_by_sums(comb, false, EPSILON, 0.0f);
    }
    memcpy(comb_out, comb, sizeof(comb));
}

void test_hyper_collapse(const uint16_t *streams, const float *pre, uint16_t *collapsed_out)
{
    for (uint32_t dimension = 0; dimension < HIDDEN; dimension++) {
        float sum = 0.0f;
        for (uint32_t stream = 0; stream < STREAMS; stream++) {
            sum = sum + pre[stream] * value_of(streams[(size_t)stream * HIDDEN + dimension]);
        }
        collapsed_out[dimension] = bfloat16_of(sum);
    }
}

void test_hyper_norm(const uint16_t *collapsed, const uint16_t *weight, double *normed_out)
{
    double square_sum = 0.0;
    for (uint32_t dimension = 0; dimension < HIDDEN; dimension++) {
        const double value = test_linear_bfloat16_value(collapsed[dimension]);
        square_sum += value * value;
    }
    const double factor =
	    1.0 / square_root(square_sum / HIDDEN + (double)MODEL_NORM_EPSILON);
    for (uint32_t dimension = 0; dimension < HIDDEN; dimension++) {
        normed_out[dimension] =
            test_linear_bfloat16_value(weight[dimension]) * (test_linear_bfloat16_value(collapsed[dimension]) * factor);
    }
}

/* The value of lane 0 after lane l adds lane l ^ d's at d = 16, 8, 4, 2 and 1 (every lane ends with the same). */
static float warp_sum(float lanes[WARP_LANES])
{
    for (uint32_t distance = WARP_LANES >> 1u; distance != 0; distance >>= 1u) {
        float next[WARP_LANES];
        for (uint32_t lane = 0; lane < WARP_LANES; lane++) {
            next[lane] = lanes[lane] + lanes[lane ^ distance];
        }
        memcpy(lanes, next, sizeof(next));
    }
    return lanes[0];
}

float test_hyper_norm_operand(const uint16_t *collapsed)
{
    float warp_sums[WARP_LANES] = {0.0f};           /* zeros behind the row's warps */
    for (uint32_t warp = 0; warp < ROW_WARPS; warp++) {
        float lanes[WARP_LANES];
        for (uint32_t lane = 0; lane < WARP_LANES; lane++) {
            const size_t first = ((size_t)warp * WARP_LANES + lane) * VALUES_PER_THREAD;
            float sum = 0.0f;
            for (uint32_t value = 0; value < VALUES_PER_THREAD; value++) {
                const float collapsed_value = value_of(collapsed[first + value]);
                sum = sum + collapsed_value * collapsed_value;
            }
            lanes[lane] = sum;
        }
        warp_sums[warp] = warp_sum(lanes);
    }
    return warp_sum(warp_sums) * MEAN_FACTOR + MODEL_NORM_EPSILON;
}

float test_hyper_reciprocal_root(float operand)
{
    return (float)(1.0 / square_root((double)operand));
}

void test_hyper_norm_with(const uint16_t *collapsed, const uint16_t *weight, float factor, uint16_t *normed_out)
{
    for (uint32_t dimension = 0; dimension < HIDDEN; dimension++) {
        const float scaled = value_of(collapsed[dimension]) * factor;
        normed_out[dimension] = bfloat16_of(value_of(weight[dimension]) * scaled);
    }
}

void test_hyper_post(const uint16_t *streams, const float *post, const float *comb, const uint16_t *output,
                     const float *routed_sums, uint16_t *streams_out)
{
    for (uint32_t dimension = 0; dimension < HIDDEN; dimension++) {
        float sublayer = value_of(output[dimension]);
        if (routed_sums != NULL) {
            sublayer = value_of(bfloat16_of(routed_sums[dimension] + sublayer));
        }
        for (uint32_t target = 0; target < STREAMS; target++) {
            float sum = 0.0f;
            for (uint32_t source = 0; source < STREAMS; source++) {
                sum = sum + comb[source * STREAMS + target] * value_of(streams[(size_t)source * HIDDEN + dimension]);
            }
            streams_out[(size_t)target * HIDDEN + dimension] = bfloat16_of(post[target] * sublayer + sum);
        }
    }
}

/* The place of a finite float32 on a line where neighbours are one apart and both zeros 0. */
static int64_t float_ordinal(float value)
{
    uint32_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    const int64_t magnitude = (int64_t)(bits & ~FLOAT32_SIGN_BIT);
    return (bits & FLOAT32_SIGN_BIT) != 0 ? -magnitude : magnitude;
}

uint32_t test_hyper_float_places(float first, float second)
{
    const int64_t distance = float_ordinal(first) - float_ordinal(second);
    const int64_t magnitude = distance < 0 ? -distance : distance;
    return magnitude > (int64_t)UINT32_MAX ? UINT32_MAX : (uint32_t)magnitude;
}
