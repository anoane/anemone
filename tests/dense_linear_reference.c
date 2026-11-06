/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * dense_linear_reference.c: the host reference of the dense products. Contract: dense_linear_reference.h.
 */
#include "dense_linear_reference.h"

#include <stddef.h>
#include <string.h>

#include "anemone/dense_linear_records.h"
#include "linear_reference.h"

#define DOUBLE_EXPONENT_SHIFT 52u
#define DOUBLE_EXPONENT_MASK 0x7FFull
#define DOUBLE_EXPONENT_BIAS 1023
#define DOUBLE_FRACTION_MASK 0xFFFFFFFFFFFFFull
#define HALVES 2                                /* of an exponent */
#define NEWTON_STEPS 10u                        /* from half the root's reciprocal: past the 53 bits of a double */
#define ONE_AND_A_HALF 1.5
#define ONE_HALF 0.5

double test_dense_magnitude(double value)
{
    return value < 0.0 ? -value : value;
}

/* value = w 4^h with w from 1 to below 4, and Newton's steps for 1 / sqrt(w) from 1/2, which they approach from
 * below. */
double test_dense_reciprocal_square_root(double value)
{
    uint64_t bits = 0;
    double reduced = 0.0;
    double power = 0.0;
    memcpy(&bits, &value, sizeof(bits));
    const int64_t exponent = (int64_t)((bits >> DOUBLE_EXPONENT_SHIFT) & DOUBLE_EXPONENT_MASK) - DOUBLE_EXPONENT_BIAS;
    const int64_t half = exponent >= 0 ? exponent / HALVES : -((HALVES - 1 - exponent) / HALVES);    /* rounded down */
    const uint64_t reduced_bits = (bits & DOUBLE_FRACTION_MASK) |
                                  (uint64_t)(DOUBLE_EXPONENT_BIAS + exponent - HALVES * half) << DOUBLE_EXPONENT_SHIFT;
    const uint64_t power_bits = (uint64_t)(DOUBLE_EXPONENT_BIAS - half) << DOUBLE_EXPONENT_SHIFT;
    memcpy(&reduced, &reduced_bits, sizeof(reduced));
    memcpy(&power, &power_bits, sizeof(power));
    double guess = ONE_HALF;
    for (uint32_t step = 0; step < NEWTON_STEPS; step++) {
        guess = guess * (ONE_AND_A_HALF - ONE_HALF * reduced * guess * guess);
    }
    return guess * power;
}

static double weight_at(uint32_t weight_type, const unsigned char *weights, uint64_t index)
{
	if (weight_type == DENSE_FLOAT32) {
		float value = 0.0f;
		memcpy(&value, weights + index * sizeof(float), sizeof(value));
		return (double)value;
	}
    uint16_t bits = 0;
    memcpy(&bits, weights + index * sizeof(uint16_t), sizeof(bits));
    return test_linear_bfloat16_value(bits);
}

void test_dense_reference(uint32_t weight_type, const unsigned char *weights, uint32_t input_size, uint32_t flags,
                          float epsilon, const uint16_t *rows, uint32_t row_count, uint32_t first_output,
                          uint32_t output_count, double *sums_out, double *magnitudes_out)
{
    for (uint32_t row = 0; row < row_count; row++) {
        const uint16_t *const values = rows + (size_t)row * input_size;
        double scale = 1.0;
	if ((flags & DENSE_ROW_SCALE) != 0) {
		double squares = 0.0;
		for (uint32_t input = 0; input < input_size; input++) {
			squares += test_linear_bfloat16_value(values[input]) *
				   test_linear_bfloat16_value(values[input]);
		}
		scale = test_dense_reciprocal_square_root(squares / input_size +
							  (double)epsilon);
	}
	for (uint32_t output = 0; output < output_count; output++) {
            const uint64_t first_weight = (uint64_t)(first_output + output) * input_size;
            double sum = 0.0, magnitude_sum = 0.0;
            for (uint32_t input = 0; input < input_size; input++) {
                const double product =
                    weight_at(weight_type, weights, first_weight + input) * test_linear_bfloat16_value(values[input]);
                sum += product;
                magnitude_sum += test_dense_magnitude(product);
            }
            sums_out[(size_t)row * output_count + output] = sum * scale;
            magnitudes_out[(size_t)row * output_count + output] = magnitude_sum * scale;
        }
    }
}

/* The bfloat16 the kernel makes of a float32 sum: rounded, multiplied by the scale in float32, rounded again. */
static uint16_t bfloat16_scaled(double sum, float output_scale)
{
    const float rounded = (float)test_linear_bfloat16_value(test_linear_bfloat16_nearest(sum));
    return test_linear_bfloat16_nearest((double)(rounded * output_scale));
}

bool test_dense_output_holds(uint32_t output_type, float output_scale, const unsigned char *outputs, uint64_t index,
                             double sum, double magnitude_sum)
{
    const double bound = TEST_DENSE_TOLERANCE * magnitude_sum + TEST_DENSE_SUBNORMAL_ALLOWANCE;
    if (output_type == DENSE_FLOAT32) {
	    float value = 0.0f;
	    memcpy(&value, outputs + index * sizeof(float), sizeof(value));
	    return test_dense_magnitude((double)value -
					sum * (double)output_scale) <=
		   bound * (double)output_scale;
    }
    uint16_t bits = 0;
    memcpy(&bits, outputs + index * sizeof(uint16_t), sizeof(bits));
    const int32_t ordinal = test_linear_bfloat16_ordinal(bits);
    return test_linear_bfloat16_is_finite(bits) &&
           ordinal >= test_linear_bfloat16_ordinal(bfloat16_scaled(sum - bound, output_scale)) &&
           ordinal <= test_linear_bfloat16_ordinal(bfloat16_scaled(sum + bound, output_scale));
}
