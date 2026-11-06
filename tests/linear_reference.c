/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * linear_reference.c: the host reference of linear_reference.h. Test support: no part of the engine.
 *
 * Contract: linear_reference.h. What it leaves open is decided here:
 *   - A double is taken apart by its bits (sign, exponent field, fraction): that tells a not-a-number and an infinity
 *     from a number and gives the leading bit of a value without a function of the mathematics library.
 *   - A value below 2^52 is rounded to a whole number, a tie to even, by adding and subtracting 2^52: the addition
 *     rounds as the machine does, and the build never reorders floating-point arithmetic.
 *   - A power of two beyond what one E8M0 byte stands for is a product of several, each product exact.
 *   - The working memory of test_linear_products (one row of weights, the scales of the fed vectors) is obtained for
 *     the call and returned before it ends.
 */
#include "linear_reference.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "weight_format_reference.h"

#define DOUBLE_SIGN_SHIFT 63u
#define DOUBLE_EXPONENT_SHIFT 52u
#define DOUBLE_EXPONENT_MASK 0x7FFu             /* all of it set: an infinity or a not-a-number */
#define DOUBLE_EXPONENT_BIAS 1023
#define DOUBLE_FRACTION_MASK 0xFFFFFFFFFFFFFull
#define WHOLE_NUMBERS_FROM 4503599627370496.0   /* 2^52: a double at or above it has no fraction */
#define ONE_HALF 0.5

#define E8M0_BIAS 127                           /* an E8M0 byte c stands for 2^(c - 127) */
#define CARRY_EXPONENT 100                      /* an operand is a value times 2^100 */
#define E8M0_LARGEST_CODE 254u
#define E8M0_CODE_COUNT 255u                    /* the bytes that are a number */

#define BFLOAT16_SIGN_BIT 0x8000u
#define BFLOAT16_MAGNITUDE_MASK 0x7FFFu
#define BFLOAT16_FRACTION_BITS 7
#define BFLOAT16_FRACTION_SHIFT 7u
#define BFLOAT16_FRACTION_MASK 0x7Fu
#define BFLOAT16_HIDDEN_ONE 128u                /* the leading bit of a normal number, in units of its last place */
#define BFLOAT16_EXPONENT_MASK 0xFFu
#define BFLOAT16_EXPONENT_BIAS 127
#define BFLOAT16_SMALLEST_EXPONENT (-126)       /* of a normal number: below it the last place stays 2^-133 */
#define BFLOAT16_INFINITY 0x7F80u
#define BFLOAT16_BITS 16u                       /* a bfloat16 is the upper half of a float32 */
#define FLOAT32_OVERFLOW_EXPONENT 128           /* 2^128 and above: beyond every float32 and bfloat16 */

#define HALF_OVERFLOW 65520.0                   /* halfway from the largest 16-bit float to 2^16: infinite from here */
#define HALF_SMALLEST_NORMAL_EXPONENT (-14)     /* below 2^-14 a 16-bit float is a multiple of 2^-24 */
#define HALF_SUBNORMAL_PLACE_EXPONENT (-24)

#define E4M3_SIGN_BIT 0x80u
#define E4M3_MAGNITUDE_COUNT 127u               /* the codes 0x00 to 0x7E, in ascending order of their values */
#define CODE_PARITIES 2u                        /* a code is even or odd: a tie goes to the even one */
#define E4M3_LARGEST 448.0
#define E4M3_LARGEST_INVERSE (1.0f / 448.0f)    /* float32(1 / 448): the reference multiplies by it */
#define MAGNITUDE_FLOOR 1.0e-4f                 /* float32(1e-4): the least a block's largest magnitude counts as */
#define FLOAT32_ROUNDING_PLACE_EXPONENT (-24)   /* a float32 operation rounds by at most 2^-24 of its result */
#define FLOAT32_SMALLEST_PLACE_EXPONENT (-149)  /* or by this, where the result is subnormal */

#define VALUES_PER_LOAD 64u                     /* the lane order of a quantized vector: linear_records.h */
#define VALUES_PER_STEP 16u
#define VALUES_PER_HALF_STEP 8u
#define VALUES_PER_LANE_PLACE 2u
#define LANE_PLACES 4u
#define STEPS_PER_LOAD 4u
#define BYTES_PER_WORD 4u
#define REGISTERS_PER_WORD 2u                   /* j % 2: the register of the low and of the high inputs */

#define NATURAL_LOGARITHM_OF_TWO 0.6931471805599453
#define EXPONENTIAL_SERIES_TERMS 18u            /* of an argument within half of ln 2: the rest is below 2^-70 */
#define FLOAT32_EXPONENTIAL_LIMIT (128.0 * NATURAL_LOGARITHM_OF_TWO)
#define TOLERANCE_STEPS_LIMIT 4096              /* neighbours tried on each side of a product: far above any need */

struct double_fields {
    bool is_negative;
    uint32_t exponent_field;
    uint64_t fraction;
};

static struct double_fields fields_of(double value)
{
    uint64_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    return (struct double_fields){.is_negative = (bits >> DOUBLE_SIGN_SHIFT) != 0,
                                  .exponent_field = (uint32_t)(bits >> DOUBLE_EXPONENT_SHIFT) & DOUBLE_EXPONENT_MASK,
                                  .fraction = bits & DOUBLE_FRACTION_MASK};
}

static bool is_not_a_number(double value)
{
    const struct double_fields fields = fields_of(value);
    return fields.exponent_field == DOUBLE_EXPONENT_MASK && fields.fraction != 0;
}

static bool is_finite(double value)
{
    return fields_of(value).exponent_field != DOUBLE_EXPONENT_MASK;
}

static double magnitude_of(double value)
{
    return fields_of(value).is_negative ? -value : value;
}

/* 2 to a whole power from -1000 to 1000: every factor and every product is a power of two a double holds. */
static double two_to(int32_t exponent)
{
    double power = 1.0;
    int32_t exponent_left = exponent;
    while (exponent_left > E8M0_BIAS) {
        power *= test_weight_format_e8m0_value(E8M0_LARGEST_CODE);
        exponent_left -= E8M0_BIAS;
    }
    while (exponent_left < -E8M0_BIAS) {
        power *= test_weight_format_e8m0_value(0);
        exponent_left += E8M0_BIAS;
    }
    return power * test_weight_format_e8m0_value((uint32_t)(exponent_left + E8M0_BIAS));
}

/* A value from 0 to below 2^52 as the nearest whole number, a tie to the even one. */
static double rounded_to_whole(double value)
{
    const double shifted = value + WHOLE_NUMBERS_FROM;
    return shifted - WHOLE_NUMBERS_FROM;
}

/* The float32 a bfloat16 is the upper half of. */
static float float_of_bfloat16(uint16_t bits)
{
    const uint32_t float_bits = (uint32_t)bits << BFLOAT16_BITS;
    float value = 0.0f;
    memcpy(&value, &float_bits, sizeof(value));
    return value;
}

double test_linear_bfloat16_value(uint16_t bits)
{
    const uint32_t exponent_field = ((uint32_t)bits >> BFLOAT16_FRACTION_SHIFT) & BFLOAT16_EXPONENT_MASK;
    const uint32_t fraction = bits & BFLOAT16_FRACTION_MASK;
    double magnitude = 0.0;
    if (exponent_field == BFLOAT16_EXPONENT_MASK) {
        magnitude = fraction != 0 ? (double)NAN : (double)INFINITY;
    } else if (exponent_field == 0) {
        magnitude = (double)fraction * two_to(BFLOAT16_SMALLEST_EXPONENT - BFLOAT16_FRACTION_BITS);
    } else {
        magnitude = (double)(BFLOAT16_HIDDEN_ONE + fraction) *
                    two_to((int32_t)exponent_field - BFLOAT16_EXPONENT_BIAS - BFLOAT16_FRACTION_BITS);
    }
    return (bits & BFLOAT16_SIGN_BIT) != 0 ? -magnitude : magnitude;
}

bool test_linear_bfloat16_is_finite(uint16_t bits)
{
    return (((uint32_t)bits >> BFLOAT16_FRACTION_SHIFT) & BFLOAT16_EXPONENT_MASK) != BFLOAT16_EXPONENT_MASK;
}

uint16_t test_linear_bfloat16_nearest(double value)
{
    const struct double_fields fields = fields_of(value);
    const uint32_t sign = fields.is_negative ? BFLOAT16_SIGN_BIT : 0;
    /* a subnormal double counts as far below every bfloat16: its exponent here is -1023 */
    const int32_t exponent = (int32_t)fields.exponent_field - DOUBLE_EXPONENT_BIAS;
    if (is_not_a_number(value)) {
        return TEST_LINEAR_BFLOAT16_NOT_A_NUMBER;
    }
    if (exponent >= FLOAT32_OVERFLOW_EXPONENT) {
        return (uint16_t)(sign | BFLOAT16_INFINITY);
    }
    const bool is_normal = exponent >= BFLOAT16_SMALLEST_EXPONENT;
    const int32_t place_exponent = (is_normal ? exponent : BFLOAT16_SMALLEST_EXPONENT) - BFLOAT16_FRACTION_BITS;
    /* in units of the last place: 128 to 256 for a normal number, 0 to 128 below; the scaling is exact */
    const uint32_t places = (uint32_t)rounded_to_whole(magnitude_of(value) * two_to(-place_exponent));
    /* a fraction that rounds up to 256 places carries into the exponent, and from the largest into the infinity */
    const uint32_t magnitude_bits =
        is_normal ? ((uint32_t)(exponent + BFLOAT16_EXPONENT_BIAS) << BFLOAT16_FRACTION_SHIFT) + places -
                        BFLOAT16_HIDDEN_ONE
                  : places;
    return (uint16_t)(sign | (magnitude_bits < BFLOAT16_INFINITY ? magnitude_bits : BFLOAT16_INFINITY));
}

int32_t test_linear_bfloat16_ordinal(uint16_t bits)
{
    const int32_t magnitude = (int32_t)(bits & BFLOAT16_MAGNITUDE_MASK);
    return (bits & BFLOAT16_SIGN_BIT) != 0 ? -magnitude : magnitude;
}

uint16_t test_linear_bfloat16_at(int32_t ordinal)
{
    const uint32_t magnitude = (uint32_t)(ordinal < 0 ? -(int64_t)ordinal : (int64_t)ordinal);
    const uint32_t magnitude_bits = magnitude < BFLOAT16_INFINITY ? magnitude : BFLOAT16_INFINITY;
    return (uint16_t)((ordinal < 0 ? BFLOAT16_SIGN_BIT : 0) | magnitude_bits);
}

double test_linear_bfloat16_last_place(uint16_t bits)
{
    const uint32_t exponent_field = ((uint32_t)bits >> BFLOAT16_FRACTION_SHIFT) & BFLOAT16_EXPONENT_MASK;
    const int32_t exponent =
        exponent_field <= 1 ? BFLOAT16_SMALLEST_EXPONENT : (int32_t)exponent_field - BFLOAT16_EXPONENT_BIAS;
    return two_to(exponent - BFLOAT16_FRACTION_BITS);
}

double test_linear_half_value(uint16_t bfloat16_bits)
{
    const double value = test_linear_bfloat16_value(bfloat16_bits);
    const bool is_negative = (bfloat16_bits & BFLOAT16_SIGN_BIT) != 0;
    const double magnitude = is_negative ? -value : value;
    double half_magnitude = magnitude;          /* 8 significant bits: a normal 16-bit float holds them all */
    if (is_not_a_number(value)) {
        return value;
    }
    if (magnitude >= HALF_OVERFLOW) {
        half_magnitude = (double)INFINITY;
    } else if (magnitude < two_to(HALF_SMALLEST_NORMAL_EXPONENT)) {
        half_magnitude = rounded_to_whole(magnitude * two_to(-HALF_SUBNORMAL_PLACE_EXPONENT)) *
                         two_to(HALF_SUBNORMAL_PLACE_EXPONENT);
    }
    return is_negative ? -half_magnitude : half_magnitude;
}

/* The E8M0 code of the smallest power of two that is not below ratio, or E8M0_CODE_COUNT if there is none. */
static uint32_t scale_code_of(float ratio)
{
    uint32_t code = 0;
    while (code < E8M0_CODE_COUNT && test_weight_format_e8m0_value(code) < (double)ratio) {
        code++;
    }
    return code;
}

/* The code 0x00 to 0x7E whose value is nearest to a magnitude from 0 to 448; of two equally near, the even one.
 * magnitudes holds the value of each code, in ascending order. */
static uint32_t nearest_magnitude_code(const double *magnitudes, double magnitude)
{
    uint32_t nearest = 0;
    double nearest_distance = magnitude;        /* from the value of code 0, which is 0 */
    for (uint32_t code = 1; code < E4M3_MAGNITUDE_COUNT; code++) {
        const double distance = magnitude_of(magnitudes[code] - magnitude);
        if (distance < nearest_distance || (distance == nearest_distance && code % CODE_PARITIES == 0)) {
            nearest = code;
            nearest_distance = distance;
        }
    }
    return nearest;
}

result_status test_linear_quantize(const uint16_t *values, uint32_t width,
				   uint8_t *codes_out, uint8_t *exponents_out)
{
    double magnitudes[E4M3_MAGNITUDE_COUNT] = {0.0};
    if (values == NULL || codes_out == NULL || exponents_out == NULL || width == 0 ||
        width % TEST_LINEAR_VALUES_PER_BLOCK != 0) {
	    return STATUS_INVALID_ARGUMENT;
    }
    for (uint32_t code = 0; code < E4M3_MAGNITUDE_COUNT; code++) {
        magnitudes[code] = test_weight_format_e4m3_value(code);
    }
    for (uint32_t block_index = 0; block_index < width / TEST_LINEAR_VALUES_PER_BLOCK; block_index++) {
        const uint16_t *const block_values = values + (size_t)block_index * TEST_LINEAR_VALUES_PER_BLOCK;
        float largest = MAGNITUDE_FLOOR;
        for (uint32_t value_index = 0; value_index < TEST_LINEAR_VALUES_PER_BLOCK; value_index++) {
            /* a bfloat16 is a float32: the magnitude is one exactly */
            const float magnitude =
                float_of_bfloat16((uint16_t)(block_values[value_index] & BFLOAT16_MAGNITUDE_MASK));
            if (!test_linear_bfloat16_is_finite(block_values[value_index])) {
		    return STATUS_INVALID_ARGUMENT;
	    }
            largest = magnitude > largest ? magnitude : largest;
        }
        const uint32_t scale_code = scale_code_of(largest * E4M3_LARGEST_INVERSE);
        if (scale_code >= E8M0_CODE_COUNT) {
		return STATUS_INVALID_ARGUMENT;
	}
        const double inverse_scale = two_to(E8M0_BIAS - (int32_t)scale_code);
        exponents_out[block_index] = (uint8_t)scale_code;
        for (uint32_t value_index = 0; value_index < TEST_LINEAR_VALUES_PER_BLOCK; value_index++) {
            const uint16_t bits = block_values[value_index];
            const double scaled =
                test_linear_bfloat16_value((uint16_t)(bits & BFLOAT16_MAGNITUDE_MASK)) * inverse_scale;
            const uint32_t code = nearest_magnitude_code(magnitudes, scaled > E4M3_LARGEST ? E4M3_LARGEST : scaled);
            codes_out[(size_t)block_index * TEST_LINEAR_VALUES_PER_BLOCK + value_index] =
                (uint8_t)(code | ((bits & BFLOAT16_SIGN_BIT) != 0 ? E4M3_SIGN_BIT : 0));
        }
    }
    return STATUS_OK;
}

result_status test_linear_codes_lay_out(const uint8_t *codes, uint32_t width,
					uint8_t *stored_out)
{
    if (codes == NULL || stored_out == NULL || width == 0 || width % VALUES_PER_LOAD != 0) {
	    return STATUS_INVALID_ARGUMENT;
    }
    /* byte ((4 l + t) 4 + s) 4 + j holds the code of value 64 l + 16 s + 2 t + j / 2 + 8 (j % 2) */
    for (uint32_t byte_index = 0; byte_index < width; byte_index++) {
        const uint32_t byte_in_word = byte_index % BYTES_PER_WORD;                      /* j */
        const uint32_t step = byte_index / BYTES_PER_WORD % STEPS_PER_LOAD;             /* s */
        const uint32_t lane_place = byte_index / VALUES_PER_STEP % LANE_PLACES;         /* t */
        const uint32_t load = byte_index / VALUES_PER_LOAD;                             /* l */
        stored_out[byte_index] =
            codes[VALUES_PER_LOAD * load + VALUES_PER_STEP * step + VALUES_PER_LANE_PLACE * lane_place +
                  byte_in_word / REGISTERS_PER_WORD + VALUES_PER_HALF_STEP * (byte_in_word % REGISTERS_PER_WORD)];
    }
    return STATUS_OK;
}

result_status test_linear_operands_lay_out(const uint8_t *codes,
					   const uint8_t *exponents,
					   uint32_t width,
					   uint16_t *operands_out)
{
    if (codes == NULL || exponents == NULL || operands_out == NULL || width == 0 || width % VALUES_PER_LOAD != 0) {
	    return STATUS_INVALID_ARGUMENT;
    }
    /* half ((4 l + t) 4 + s) 4 + 2 r + h holds the operand of value 64 l + 16 s + 2 t + h + 8 r */
    for (uint32_t half_index = 0; half_index < width; half_index++) {
        const uint32_t half_of_register = half_index % VALUES_PER_LANE_PLACE;                        /* h */
        const uint32_t input_register = half_index / VALUES_PER_LANE_PLACE % REGISTERS_PER_WORD;     /* r */
        const uint32_t step = half_index / BYTES_PER_WORD % STEPS_PER_LOAD;                          /* s */
        const uint32_t lane_place = half_index / VALUES_PER_STEP % LANE_PLACES;                      /* t */
        const uint32_t value_index = VALUES_PER_LOAD * (half_index / VALUES_PER_LOAD) + VALUES_PER_STEP * step +
                                     VALUES_PER_LANE_PLACE * lane_place + half_of_register +
                                     VALUES_PER_HALF_STEP * input_register;
        const uint32_t exponent = exponents[value_index / TEST_LINEAR_VALUES_PER_BLOCK];
        const double value = test_weight_format_e4m3_value(codes[value_index]);
        const uint16_t sign = (codes[value_index] & E4M3_SIGN_BIT) != 0 ? BFLOAT16_SIGN_BIT : 0;
        if (is_not_a_number(value)) {
		return STATUS_INVALID_ARGUMENT;
	}
        operands_out[half_index] =
            exponent > TEST_LINEAR_CARRIED_EXPONENT_LARGEST
                ? (uint16_t)(BFLOAT16_INFINITY | sign)
                : test_linear_bfloat16_nearest(value * two_to((int32_t)exponent - E8M0_BIAS + CARRY_EXPONENT));
    }
    return STATUS_OK;
}

uint32_t test_linear_code_place(uint32_t value_index)
{
    /* from a value to its byte: the load and the instruction it is fed to, the lane place that holds it, and in that
     * place's word the register (low or high inputs) and the half of the register */
    const uint32_t load = value_index / VALUES_PER_LOAD;
    const uint32_t step = value_index % VALUES_PER_LOAD / VALUES_PER_STEP;
    const uint32_t value_in_step = value_index % VALUES_PER_STEP;
    const uint32_t input_register = value_in_step / VALUES_PER_HALF_STEP;
    const uint32_t lane_place = value_in_step % VALUES_PER_HALF_STEP / VALUES_PER_LANE_PLACE;
    const uint32_t half_of_register = value_in_step % VALUES_PER_LANE_PLACE;
    return ((load * LANE_PLACES + lane_place) * STEPS_PER_LOAD + step) * BYTES_PER_WORD +
           half_of_register * REGISTERS_PER_WORD + input_register;
}

result_status test_linear_fed_values_of_codes(const uint8_t *codes,
					      uint32_t width,
					      double *fed_values_out)
{
    if (codes == NULL || fed_values_out == NULL || width == 0 || width % TEST_LINEAR_VALUES_PER_BLOCK != 0) {
	    return STATUS_INVALID_ARGUMENT;
    }
    for (uint32_t value_index = 0; value_index < width; value_index++) {
        const double value = test_weight_format_e4m3_value(codes[value_index]);
        if (is_not_a_number(value)) {
		return STATUS_INVALID_ARGUMENT;
	}
        fed_values_out[value_index] = value;
    }
    return STATUS_OK;
}

result_status test_linear_fed_values_of_bfloat16(const uint16_t *values,
						 uint32_t width,
						 double *fed_values_out)
{
    if (values == NULL || fed_values_out == NULL || width == 0 || width % TEST_LINEAR_VALUES_PER_BLOCK != 0) {
	    return STATUS_INVALID_ARGUMENT;
    }
    for (uint32_t value_index = 0; value_index < width; value_index++) {
        fed_values_out[value_index] = test_linear_half_value(values[value_index]);
    }
    return STATUS_OK;
}

/* One row of a matrix: the value of each weight's code, and the value of the scale of each block of 32 inputs. */
MUST_CHECK static result_status
weight_row(enum weight_format_code format,
	   const struct matrix_shape_record *shape, const uint8_t *stored,
	   uint64_t stored_byte_count, uint32_t output_index,
	   double *code_values_out, double *scales_out)
{
    uint32_t block_scale_code = 0;
    for (uint32_t input_index = 0; input_index < shape->input_size; input_index++) {
        struct test_weight_format_weight weight = {0};
	const result_status status = test_weight_format_read_weight(
		format, shape, stored, stored_byte_count, output_index,
		input_index, &weight);
	if (status != STATUS_OK) {
		return status;
	}
	const double code_value =
		format == WEIGHT_FORMAT_E2M1_SCALED ?
			test_weight_format_e2m1_value(weight.code) :
			test_weight_format_e4m3_value(weight.code);
	if (input_index % TEST_LINEAR_VALUES_PER_BLOCK == 0) {
            block_scale_code = weight.scale_code;
            scales_out[input_index / TEST_LINEAR_VALUES_PER_BLOCK] = test_weight_format_e8m0_value(block_scale_code);
        }
        if (is_not_a_number(code_value) || weight.scale_code != block_scale_code ||
            weight.scale_code > E8M0_LARGEST_CODE) {
		return STATUS_INVALID_ARGUMENT;
	}
        code_values_out[input_index] = code_value;
    }
    return STATUS_OK;
}

/* One output for one fed vector: fed_values [input_size]; fed_scales [input_size / 32], or NULL. */
static struct test_linear_product product_of(const double *code_values, const double *weight_scales,
                                             const double *fed_values, const double *fed_scales, uint32_t input_size)
{
    struct test_linear_product product = {0.0, 0.0};
    for (uint32_t block_index = 0; block_index < input_size / TEST_LINEAR_VALUES_PER_BLOCK; block_index++) {
        const size_t first_input = (size_t)block_index * TEST_LINEAR_VALUES_PER_BLOCK;
        const double fed_scale = fed_scales != NULL ? fed_scales[block_index] : 1.0;
        double block_sum = 0.0;
        double block_magnitude_sum = 0.0;
        for (size_t input_index = first_input; input_index < first_input + TEST_LINEAR_VALUES_PER_BLOCK;
             input_index++) {
            const double term = code_values[input_index] * fed_values[input_index];
            block_sum += term;
            block_magnitude_sum += magnitude_of(term);
        }
        product.sum += block_sum * weight_scales[block_index] * fed_scale;
        product.magnitude_sum += block_magnitude_sum * weight_scales[block_index] * fed_scale;
    }
    return product;
}

result_status test_linear_products(
	enum weight_format_code format, const struct matrix_shape_record *shape,
	const uint8_t *stored, uint64_t stored_byte_count,
	uint32_t first_output, uint32_t output_count, const double *fed_values,
	const uint8_t *fed_exponents, uint32_t vector_count,
	uint32_t segment_count, struct test_linear_product *products_out)
{
    struct test_weight_format_sizes sizes = {0};
    result_status status = test_weight_format_sizes_of(format, shape, &sizes);
    if (status != STATUS_OK || stored == NULL || fed_values == NULL ||
	products_out == NULL || stored_byte_count != sizes.stored_byte_count ||
	first_output > shape->output_size ||
	output_count > shape->output_size - first_output || vector_count == 0 ||
	segment_count == 0 || shape->output_size % segment_count != 0 ||
	shape->input_size % TEST_LINEAR_VALUES_PER_BLOCK != 0) {
	    return STATUS_INVALID_ARGUMENT;
    }
    const size_t input_size = shape->input_size;
    const size_t blocks_per_segment = input_size / TEST_LINEAR_VALUES_PER_BLOCK;
    const size_t blocks_per_vector = blocks_per_segment * segment_count;
    const uint32_t outputs_per_segment = shape->output_size / segment_count;
    double *const code_values = malloc(input_size * sizeof(*code_values));
    double *const weight_scales = malloc(blocks_per_segment * sizeof(*weight_scales));
    double *const fed_scales =
        fed_exponents != NULL ? malloc(blocks_per_vector * vector_count * sizeof(*fed_scales)) : NULL;
    if (code_values == NULL || weight_scales == NULL || (fed_exponents != NULL && fed_scales == NULL)) {
        free(fed_scales);
        free(weight_scales);
        free(code_values);
	return STATUS_OUT_OF_MEMORY;
    }
    for (size_t scale_index = 0;
	 status == STATUS_OK && fed_exponents != NULL && fed_scales != NULL &&
	 scale_index < blocks_per_vector * vector_count;
	 scale_index++) {
	    fed_scales[scale_index] =
		    test_weight_format_e8m0_value(fed_exponents[scale_index]);
	    status = fed_exponents[scale_index] > E8M0_LARGEST_CODE ?
			     STATUS_INVALID_ARGUMENT :
			     status;
    }
    for (uint32_t output_offset = 0;
	 status == STATUS_OK && output_offset < output_count; output_offset++) {
	    const uint32_t output_index = first_output + output_offset;
	    const size_t segment = output_index / outputs_per_segment;
	    status = weight_row(format, shape, stored, stored_byte_count,
				output_index, code_values, weight_scales);
	    for (size_t vector = 0;
		 status == STATUS_OK && vector < vector_count; vector++) {
		    products_out[vector * output_count + output_offset] =
			    product_of(
				    code_values, weight_scales,
				    fed_values +
					    (vector * segment_count + segment) *
						    input_size,
				    fed_scales != NULL ?
					    fed_scales +
						    vector * blocks_per_vector +
						    segment *
							    blocks_per_segment :
					    NULL,
				    shape->input_size);
	    }
    }
    free(fed_scales);
    free(weight_scales);
    free(code_values);
    return status;
}

/* What the float32 sum of a card may differ from the sum of a product by: the bound of linear_reference.h. */
static double float32_bound(const struct test_linear_product *product, uint32_t input_size, uint32_t slice_count)
{
    const uint32_t block_count = input_size / TEST_LINEAR_VALUES_PER_BLOCK;
    return (TEST_LINEAR_TENSOR_CORE_ALLOWANCE + (double)block_count + (double)slice_count) *
               two_to(FLOAT32_ROUNDING_PLACE_EXPONENT) * product->magnitude_sum +
           (double)block_count * two_to(FLOAT32_SMALLEST_PLACE_EXPONENT);
}

bool test_linear_output_is_within_tolerance(uint16_t output_bits, const struct test_linear_product *product,
                                            uint32_t input_size, uint32_t slice_count)
{
    const double bound = float32_bound(product, input_size, slice_count);
    const double output = test_linear_bfloat16_value(output_bits);
    if (!is_finite(product->sum) || !is_finite(product->magnitude_sum)) {
        return !test_linear_bfloat16_is_finite(output_bits);
    }
    if (is_not_a_number(output)) {
        return false;
    }
    if (!test_linear_bfloat16_is_finite(output_bits)) {
        /* the sum moved by the bound towards that infinity must round to it */
        return test_linear_bfloat16_nearest(output > 0.0 ? product->sum + bound : product->sum - bound) == output_bits;
    }
    return magnitude_of(output - product->sum) <=
           test_linear_bfloat16_last_place(output_bits) * ONE_HALF + bound;
}

uint64_t test_linear_compare(const uint16_t *outputs, const struct test_linear_product *products,
                             uint64_t output_count, uint32_t input_size, uint32_t slice_count,
                             struct test_linear_comparison *comparison)
{
    uint64_t first_outside = TEST_LINEAR_NO_OUTPUT;
    for (uint64_t output_index = 0; output_index < output_count; output_index++) {
        const uint16_t rounded_once = test_linear_bfloat16_nearest(products[output_index].sum);
        const int32_t distance =
            test_linear_bfloat16_ordinal(outputs[output_index]) - test_linear_bfloat16_ordinal(rounded_once);
        const uint32_t last_places = (uint32_t)(distance < 0 ? -distance : distance);
        if (is_finite(products[output_index].sum)) {
            comparison->output_count++;
            comparison->equal_count += outputs[output_index] == rounded_once ? 1u : 0u;
        }
        if (test_linear_bfloat16_is_finite(outputs[output_index]) && test_linear_bfloat16_is_finite(rounded_once) &&
            last_places > comparison->largest_distance) {
            comparison->largest_distance = last_places;
        }
        if (!test_linear_output_is_within_tolerance(outputs[output_index], &products[output_index], input_size,
                                                    slice_count)) {
            comparison->outside_count++;
            first_outside = first_outside == TEST_LINEAR_NO_OUTPUT ? output_index : first_outside;
        }
    }
    return first_outside;
}

/* e to a power within 128 ln 2 of zero: the argument less a whole number of ln 2, by its series, times that power
 * of two. */
static double exponential_of(double argument)
{
    const double doublings_wanted = argument / NATURAL_LOGARITHM_OF_TWO;
    const int32_t doublings =
        (int32_t)(doublings_wanted < 0.0 ? doublings_wanted - ONE_HALF : doublings_wanted + ONE_HALF);
    const double rest = argument - (double)doublings * NATURAL_LOGARITHM_OF_TWO;
    double term = 1.0;
    double sum = 1.0;
    for (uint32_t term_index = 1; term_index <= EXPONENTIAL_SERIES_TERMS; term_index++) {
        term *= rest / (double)term_index;
        sum += term;
    }
    return sum * two_to(doublings);
}

double test_linear_hidden_value(double gate, double up, double limit, double route_weight)
{
    const double bounded_gate = gate > limit ? limit : gate;
    const double bounded_up = up > limit ? limit : (up < -limit ? -limit : up);
    double gated = bounded_gate;                /* where exp(-g) is below 2^-128: g / (1 + exp(-g)) is g */
    if (is_not_a_number(bounded_gate)) {
        return bounded_gate;
    }
    if (-bounded_gate > FLOAT32_EXPONENTIAL_LIMIT) {
        gated = -0.0;                           /* a negative gate over the infinity of a float32 exponential */
    } else if (-bounded_gate >= -FLOAT32_EXPONENTIAL_LIMIT) {
        gated = bounded_gate / (1.0 + exponential_of(-bounded_gate));
    }
    return gated * bounded_up * route_weight;
}

/* The ordinals of the lowest and of the highest bfloat16 within the tolerance of a product, around the ordinal of
 * the product rounded once. */
static void tolerated_ordinals(const struct test_linear_product *product, uint32_t input_size, uint32_t slice_count,
                               int32_t central, int32_t *lowest_out, int32_t *highest_out)
{
    int32_t lowest = central;
    int32_t highest = central;
    while (central - lowest < TOLERANCE_STEPS_LIMIT &&
           test_linear_output_is_within_tolerance(test_linear_bfloat16_at(lowest - 1), product, input_size,
                                                  slice_count)) {
        lowest--;
    }
    while (highest - central < TOLERANCE_STEPS_LIMIT &&
           test_linear_output_is_within_tolerance(test_linear_bfloat16_at(highest + 1), product, input_size,
                                                  slice_count)) {
        highest++;
    }
    *lowest_out = lowest;
    *highest_out = highest;
}

void test_linear_hidden_range(const struct test_linear_product *gate, const struct test_linear_product *up,
                              uint32_t input_size, uint32_t slice_count, double limit, double route_weight,
                              struct test_linear_hidden_range *range_out)
{
    enum { LOWEST, CENTRAL, HIGHEST, CANDIDATE_COUNT };
    int32_t gate_ordinals[CANDIDATE_COUNT] = {0};
    int32_t up_ordinals[CANDIDATE_COUNT] = {0};
    int32_t lowest = INT32_MAX;
    int32_t highest = INT32_MIN;
    gate_ordinals[CENTRAL] = test_linear_bfloat16_ordinal(test_linear_bfloat16_nearest(gate->sum));
    up_ordinals[CENTRAL] = test_linear_bfloat16_ordinal(test_linear_bfloat16_nearest(up->sum));
    tolerated_ordinals(gate, input_size, slice_count, gate_ordinals[CENTRAL], &gate_ordinals[LOWEST],
                       &gate_ordinals[HIGHEST]);
    tolerated_ordinals(up, input_size, slice_count, up_ordinals[CENTRAL], &up_ordinals[LOWEST],
                       &up_ordinals[HIGHEST]);
    for (uint32_t gate_candidate = 0; gate_candidate < CANDIDATE_COUNT; gate_candidate++) {
        for (uint32_t up_candidate = 0; up_candidate < CANDIDATE_COUNT; up_candidate++) {
            const double hidden = test_linear_hidden_value(
                test_linear_bfloat16_value(test_linear_bfloat16_at(gate_ordinals[gate_candidate])),
                test_linear_bfloat16_value(test_linear_bfloat16_at(up_ordinals[up_candidate])), limit, route_weight);
            const uint16_t hidden_bits = test_linear_bfloat16_nearest(hidden);
            const int32_t ordinal = test_linear_bfloat16_ordinal(hidden_bits);
            lowest = ordinal < lowest ? ordinal : lowest;
            highest = ordinal > highest ? ordinal : highest;
            if (gate_candidate == CENTRAL && up_candidate == CENTRAL) {
                range_out->central_bits = hidden_bits;
            }
        }
    }
    range_out->lowest_ordinal = lowest - 1;
    range_out->highest_ordinal = highest + 1;
    range_out->reserved_zero = 0;
}

float test_linear_row_sum(const uint16_t *pair_outputs, uint32_t pair_count, uint32_t width, uint32_t column)
{
    float sum = 0.0f;
    for (uint32_t slot = 0; slot < pair_count; slot++) {
        sum += float_of_bfloat16(pair_outputs[(size_t)slot * width + column]);
    }
    return sum;
}
