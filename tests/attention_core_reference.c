/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * attention_core_reference.c: the host reference of the attention core. Contract: attention_core_reference.h.
 *
 *   - A double is taken apart by its bits (sign, exponent field, fraction): that tells a not-a-number, an infinity
 *     and the sign of a zero without the mathematics library.
 *   - The exponential: x = k ln 2 + r with |r| at most half ln 2 (ln 2 in two parts, so that k ln 2 is exact to far
 *     below a last place), then the series of exp(r) to SERIES_TERMS terms (the last under 2^-80 of the sum), times
 *     2^k. Its error is a few last places of a double: far below every tolerance it serves.
 */
#include "attention_core_reference.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "linear_card_fixture.h"
#include "linear_reference.h"

#define DOUBLE_EXPONENT_SHIFT 52u
#define DOUBLE_EXPONENT_MASK 0x7FFull
#define DOUBLE_FRACTION_MASK 0xFFFFFFFFFFFFFull
#define DOUBLE_SIGN_SHIFT 63u
#define DOUBLE_EXPONENT_BIAS 1023
#define DOUBLE_LEAST_NORMAL_EXPONENT (-1022)
#define DOUBLE_QUIET_NOT_A_NUMBER 0x7FF8000000000000ull
#define LN2_HIGH 6.93147180369123816490e-01     /* ln 2 to 33 bits: whole multiples of it below 2^20 are exact */
#define LN2_LOW 1.90821492927058770002e-10      /* ln 2 less LN2_HIGH */
#define INVERSE_LN2 1.44269504088896338700e+00
#define EXP_LARGEST 709.782712893383973096      /* above: an infinity */
#define EXP_SMALLEST (-745.133219101941108420)  /* below: zero */
#define SERIES_TERMS 24u
#define ROUNDING_HALF 0.5
#define HALF_FRACTION_BITS 10
#define HALF_EXPONENT_BIAS 15
#define HALF_LEAST_NORMAL_EXPONENT (-14)
#define HALF_LARGEST_EXPONENT 15
#define HALF_EXPONENT_MASK 0x1Fu
#define HALF_FRACTION_MASK 0x3FFu
#define HALF_SIGN 0x8000u
#define HALF_INFINITY 0x7C00u
#define HALF_OVERFLOW 65520.0                   /* half a last place above the largest FP16, 65504 */
#define FIRST_ROTATED_DIM MODEL_FIRST_ROTATED_DIM
#define HEAD_DIMS MODEL_HEAD_DIMS
#define GROUP_KEYS 16u                          /* a softmax step of the kernel */
#define TURN_ATTEMPTS 16777216u                 /* about 1 pair in 45,000 tells two forms apart */
#define PAIR_CODE_BITS 9u
#define PAIR_CODE_MASK 0x1FFu
#define PAIR_CENTRE 256.0
#define PAIR_SCALE 128.0
#define ANGLE_CODE_BITS 21u
#define ANGLE_CODE_MASK 0x1FFFFFu
#define ANGLE_CENTRE 1048576.0
#define ANGLE_SCALE 1048576.0

static uint64_t bits_of(double value)
{
    uint64_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static double double_of(uint64_t bits)
{
    double value = 0.0;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static bool is_not_a_number(double value)
{
    const uint64_t bits = bits_of(value);
    return ((bits >> DOUBLE_EXPONENT_SHIFT) & DOUBLE_EXPONENT_MASK) == DOUBLE_EXPONENT_MASK &&
           (bits & DOUBLE_FRACTION_MASK) != 0;
}

static bool is_finite(double value)
{
    return ((bits_of(value) >> DOUBLE_EXPONENT_SHIFT) & DOUBLE_EXPONENT_MASK) != DOUBLE_EXPONENT_MASK;
}

/* 2^exponent for an exponent from -1075 to 1024: built in the exponent field, twice where it lies outside. */
static double power_of_two(int32_t exponent)
{
    if (exponent < DOUBLE_LEAST_NORMAL_EXPONENT) {
        return power_of_two(DOUBLE_LEAST_NORMAL_EXPONENT) * power_of_two(exponent - DOUBLE_LEAST_NORMAL_EXPONENT);
    }
    if (exponent > DOUBLE_EXPONENT_BIAS) {
        return power_of_two(exponent - 1) * power_of_two(1);
    }
    return double_of((uint64_t)(exponent + DOUBLE_EXPONENT_BIAS) << DOUBLE_EXPONENT_SHIFT);
}

static double infinity(void)
{
    return double_of(DOUBLE_EXPONENT_MASK << DOUBLE_EXPONENT_SHIFT);
}

double test_attention_exp(double exponent)
{
    if (is_not_a_number(exponent)) {
        return exponent;
    }
    if (exponent > EXP_LARGEST) {
        return infinity();
    }
    if (exponent < EXP_SMALLEST) {
        return 0.0;
    }
    const int32_t whole = (int32_t)(exponent * INVERSE_LN2 + (exponent < 0.0 ? -ROUNDING_HALF : ROUNDING_HALF));
    const double reduced = (exponent - (double)whole * LN2_HIGH) - (double)whole * LN2_LOW;
    double term = 1.0;
    double sum = 1.0;
    for (uint32_t index = 1; index <= SERIES_TERMS; index++) {
        term = term * reduced / (double)index;
        sum += term;
    }
    /* below the normal exponents in two products, the first exact: the result is rounded once */
    return whole < DOUBLE_LEAST_NORMAL_EXPONENT
               ? sum * power_of_two(whole - DOUBLE_LEAST_NORMAL_EXPONENT) * power_of_two(DOUBLE_LEAST_NORMAL_EXPONENT)
               : sum * power_of_two(whole);
}

double test_attention_core_half_value(uint16_t bits)
{
    const uint32_t exponent_field = ((uint32_t)bits >> HALF_FRACTION_BITS) & HALF_EXPONENT_MASK;
    const double fraction = (double)(bits & HALF_FRACTION_MASK);
    const double sign = (bits & HALF_SIGN) != 0 ? -1.0 : 1.0;
    if (exponent_field == HALF_EXPONENT_MASK) {
        return (bits & HALF_FRACTION_MASK) != 0 ? double_of(DOUBLE_QUIET_NOT_A_NUMBER) : sign * infinity();
    }
    if (exponent_field == 0) {
        return sign * fraction * power_of_two(HALF_LEAST_NORMAL_EXPONENT - HALF_FRACTION_BITS);
    }
    return sign * (fraction + power_of_two(HALF_FRACTION_BITS)) *
           power_of_two((int32_t)exponent_field - HALF_EXPONENT_BIAS - HALF_FRACTION_BITS);
}

uint16_t test_attention_half_bits(double value)
{
    if (is_not_a_number(value)) {
        return TEST_ATTENTION_HALF_NOT_A_NUMBER;
    }
    const uint32_t sign = (bits_of(value) >> DOUBLE_SIGN_SHIFT) != 0 ? HALF_SIGN : 0u;
    const double magnitude = sign != 0 ? -value : value;
    if (magnitude >= HALF_OVERFLOW) {
        return (uint16_t)(sign | HALF_INFINITY);
    }
    int32_t exponent = HALF_LEAST_NORMAL_EXPONENT;
    while (exponent < HALF_LARGEST_EXPONENT && magnitude >= power_of_two(exponent + 1)) {
        exponent++;
    }
    const bool is_normal = magnitude >= power_of_two(HALF_LEAST_NORMAL_EXPONENT);
    const double places = magnitude * power_of_two(HALF_FRACTION_BITS - exponent);   /* below 2^11: exact */
    uint32_t rounded = (uint32_t)places;
    const double left = places - (double)rounded;
    rounded += left > ROUNDING_HALF || (left == ROUNDING_HALF && (rounded & 1u) != 0) ? 1u : 0u;
    /* a subnormal is its places; a normal number its exponent field and fraction, a carry moving into the field */
    const uint32_t magnitude_bits =
        is_normal ? ((uint32_t)(exponent + HALF_EXPONENT_BIAS) << HALF_FRACTION_BITS) + rounded -
                        (1u << HALF_FRACTION_BITS)
                  : rounded;
    return (uint16_t)(sign | magnitude_bits);
}

float test_attention_turn_back(float first, float second, float cosine, float sine, uint32_t part,
                               enum test_attention_turn_form form)
{
    if (form == TEST_ATTENTION_TURN_PRODUCTS) {
        return part == 0 ? first * cosine + second * sine : second * cosine - first * sine;
    }
    if (form == TEST_ATTENTION_TURN_OTHER) {
        return part == 0 ? fmaf(second, sine, first * cosine) : fmaf(first, -sine, second * cosine);
    }
    return part == 0 ? fmaf(first, cosine, second * sine) : fmaf(second, cosine, -(first * sine));
}

/* A hash gives the pair's two values (k - 256) / 128, k of 9 bits (8 significant bits at most: bfloat16 and FP16
 * hold them), and cos and sin (k - 2^20) / 2^20, k of 21 bits (float32 holds them). */
bool test_attention_turn_case(uint64_t stream, uint32_t part,
			      enum test_attention_turn_form form,
			      float pair[MODEL_PAIR_VALUES],
			      float angle[MODEL_PAIR_VALUES])
{
    for (uint64_t attempt = 0; attempt < TURN_ATTEMPTS; attempt++) {
        uint64_t hash = test_linear_hash(stream, attempt);
	for (uint32_t value = 0; value < MODEL_PAIR_VALUES; value++) {
		pair[value] = (float)(((double)(hash & PAIR_CODE_MASK) -
				       PAIR_CENTRE) /
				      PAIR_SCALE);
		hash >>= PAIR_CODE_BITS;
	}
	for (uint32_t value = 0; value < MODEL_PAIR_VALUES; value++) {
		angle[value] = (float)(((double)(hash & ANGLE_CODE_MASK) -
					ANGLE_CENTRE) /
				       ANGLE_SCALE);
		hash >>= ANGLE_CODE_BITS;
	}
	const float kernel = test_attention_turn_back(pair[0], pair[1], angle[0], angle[1], part,
                                                      TEST_ATTENTION_TURN_KERNEL);
        const float other = test_attention_turn_back(pair[0], pair[1], angle[0], angle[1], part, form);
        if (test_linear_bfloat16_nearest((double)kernel) != test_linear_bfloat16_nearest((double)other)) {
            return true;
        }
    }
    return false;
}

/* The weight of a key in P V, of its p, in a form. */
static double weight_of(double probability, enum test_attention_form form)
{
	const double scale = (double)ATTENTION_CORE_WEIGHT_SCALE;
	if (form == TEST_ATTENTION_KERNEL) {
		return test_attention_core_half_value(
			test_attention_half_bits(probability));
	}
    if (form == TEST_ATTENTION_REFERENCE_ORDER) {
        const double rounded = test_linear_bfloat16_value(test_linear_bfloat16_nearest(probability));
        return test_attention_core_half_value(test_attention_half_bits(rounded * scale)) / scale;
    }
    return probability;
}

/* One head of one row: its 512 values into values, in the row's form. keys: the decoded keys, NULL rows absent. */
static void head_of_row(const struct test_attention_row *row, const double *const *keys, uint32_t head,
                        double *values)
{
    double query[HEAD_DIMS];
    double scores[TEST_ATTENTION_KEYS];
    const double floor = (double)ATTENTION_CORE_SCORE_FLOOR;
    double maximum = floor;
    double sum = 0.0;
    for (uint32_t dim = 0; dim < HEAD_DIMS; dim++) {
        query[dim] = test_attention_core_half_value(row->queries[(size_t)head * HEAD_DIMS + dim]);
        values[dim] = 0.0;
    }
    for (uint32_t key = 0; key < TEST_ATTENTION_KEYS; key++) {
        double dot = 0.0;
        for (uint32_t dim = 0; keys[key] != NULL && dim < HEAD_DIMS; dim++) {
            dot += query[dim] * keys[key][dim];
        }
	scores[key] = dot * (double)ATTENTION_CORE_SCALE;
	if (row->form != TEST_ATTENTION_KERNEL && keys[key] != NULL && scores[key] > maximum) {
            maximum = scores[key];
        }
    }
    for (uint32_t key = 0; key < TEST_ATTENTION_KEYS; key++) {
        if (row->form == TEST_ATTENTION_KERNEL && key % GROUP_KEYS == 0) {
            double largest = floor;
            for (uint32_t other = key; other < key + GROUP_KEYS; other++) {
                largest = keys[other] != NULL && scores[other] > largest ? scores[other] : largest;
            }
	    if (largest > maximum + (double)ATTENTION_CORE_RESCALE_STEP) {
		    const double factor = test_attention_exp(maximum - largest);
		    for (uint32_t dim = 0; dim < HEAD_DIMS; dim++) {
			    values[dim] *= factor;
		    }
		    sum *= factor;
		    maximum = largest;
	    }
	}
        if (keys[key] == NULL) {
            continue;
        }
        const double probability = test_attention_exp(scores[key] - maximum);
        const double weight = weight_of(probability, row->form);
        sum += probability;
        for (uint32_t dim = 0; dim < HEAD_DIMS; dim++) {
            values[dim] += weight * keys[key][dim];
        }
    }
    const double denominator = sum + test_attention_exp((double)row->sinks[head] - maximum);
    for (uint32_t dim = 0; dim < HEAD_DIMS; dim++) {
        values[dim] /= denominator;
        if (row->form != TEST_ATTENTION_EXACT) {
            values[dim] = test_linear_bfloat16_value(test_linear_bfloat16_nearest(values[dim]));
        }
    }
    for (uint32_t dim = FIRST_ROTATED_DIM; dim < HEAD_DIMS;
	 dim += MODEL_PAIR_VALUES) {
	    const float *angle = row->rotation + (dim - FIRST_ROTATED_DIM);
	    if (row->form !=
		TEST_ATTENTION_EXACT) { /* float32, the kernels' fused multiply-adds */
		    const float first = (float)values[dim];
		    const float second = (float)values[dim + 1u];
		    const enum test_attention_turn_form form =
			    TEST_ATTENTION_TURN_KERNEL;
		    const float real = test_attention_turn_back(
			    first, second, angle[0], angle[1], 0, form);
		    const float imaginary = test_attention_turn_back(
			    first, second, angle[0], angle[1], 1u, form);
		    values[dim] = test_linear_bfloat16_value(
			    test_linear_bfloat16_nearest((double)real));
		    values[dim + 1u] = test_linear_bfloat16_value(
			    test_linear_bfloat16_nearest((double)imaginary));
	    } else {
		    const double first = values[dim];
		    values[dim] = first * (double)angle[0] +
				  values[dim + 1u] * (double)angle[1];
		    values[dim + 1u] = values[dim + 1u] * (double)angle[0] -
				       first * (double)angle[1];
	    }
    }
}

result_status test_attention_core_row(const struct test_attention_row *row,
				      double *output)
{
	if (row == NULL || output == NULL || row->queries == NULL ||
	    row->sinks == NULL || row->rotation == NULL ||
	    row->head_count > MODEL_HEADS_VALUE ||
	    row->first_head > MODEL_HEADS_VALUE - row->head_count) {
		return STATUS_INVALID_ARGUMENT;
	}
    const double *keys[TEST_ATTENTION_KEYS];
    double *decoded = malloc((size_t)TEST_ATTENTION_KEYS * HEAD_DIMS * sizeof(double));
    if (decoded == NULL) {
	    return STATUS_OUT_OF_MEMORY;
    }
    for (uint32_t key = 0; key < TEST_ATTENTION_KEYS; key++) {
        double *values = decoded + (size_t)key * HEAD_DIMS;
        keys[key] = row->keys[key] == NULL ? NULL : values;
        for (uint32_t dim = 0; row->keys[key] != NULL && dim < HEAD_DIMS; dim++) {
            values[dim] = test_attention_core_half_value(row->keys[key][dim]);
        }
    }
    for (uint32_t head = row->first_head; head < row->first_head + row->head_count; head++) {
        head_of_row(row, keys, head, output + (size_t)head * HEAD_DIMS);
    }
    free(decoded);
    return STATUS_OK;
}

void test_attention_distance_add(struct test_attention_distance *distance, const uint16_t *values,
                                 const double *reference, uint64_t count)
{
    for (uint64_t index = 0; index < count; index++) {
        const double value = test_linear_bfloat16_value(values[index]);
        const uint16_t nearest = test_linear_bfloat16_nearest(reference[index]);
        distance->value_count++;
        if (!is_finite(value) && is_finite(reference[index])) {
            distance->not_finite_count++;
            continue;
        }
        const double difference = value - reference[index];
        distance->error_squares += is_finite(difference) ? difference * difference : 0.0;
        distance->squares += is_finite(reference[index]) ? reference[index] * reference[index] : 0.0;
        distance->equal_count += values[index] == nearest ? 1u : 0u;
        if (test_linear_bfloat16_is_finite(values[index]) && test_linear_bfloat16_is_finite(nearest)) {
            const int32_t places = test_linear_bfloat16_ordinal(values[index]) - test_linear_bfloat16_ordinal(nearest);
            const int32_t magnitude = places < 0 ? -places : places;
            distance->largest_places = magnitude > distance->largest_places ? magnitude : distance->largest_places;
        }
    }
}

double test_attention_relative_squared(const struct test_attention_distance *distance)
{
    return distance->squares > 0.0 ? distance->error_squares / distance->squares : 0.0;
}

bool test_attention_core_within(const struct test_attention_distance *distance, double relative)
{
    return distance->not_finite_count == 0 &&
           (distance->squares > 0.0 ? distance->error_squares <= relative * relative * distance->squares
                                    : distance->error_squares == 0.0);
}
