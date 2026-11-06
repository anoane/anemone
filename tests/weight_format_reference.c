/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * weight_format_reference.c: the host reference of weight_format_reference.h. Test support: no part of the engine.
 *
 * Contract: weight_format_reference.h. What it leaves open is decided here:
 *   - The stored form is bytes: a 32-bit word of the seam is four of them, the lowest first, as a pack is
 *     little-endian. The lay-out builds each word of 4-bit codes from its eight nibbles and writes its four bytes;
 *     the read takes the one byte that holds the nibble it wants. Neither needs an aligned buffer.
 *   - The lay-out walks the stored form by index: word after word (E2M1) or byte after byte (E4M3) of the weights,
 *     then byte after byte of the scales, and takes each index apart into the tile, load or group, lane and place
 *     the seam names. The read goes the other way: from an output and an input to the one index.
 *   - No function of the mathematics library is called (the tests are linked without it): a power of two is made
 *     by multiplications that are all exact.
 */
#include "weight_format_reference.h"

#include <math.h>
#include <stdbool.h>
#include <stddef.h>

/* What both layouts are made of: the 16-by-16 weight operand of the tensor-core instruction, held by 32 lanes. */
#define OUTPUTS_PER_TILE 16
#define OUTPUTS_PER_HALF_TILE 8                 /* a lane holds an output of each half of the tile */
#define LANES_PER_WARP 32
#define LANES_PER_LANE_GROUP 4                  /* lane = 4 g + t: the lanes of one pair of outputs */
#define LANE_GROUPS_PER_TILE 8
#define INPUTS_PER_SCALE 32
#define INPUTS_PER_STEP 16                      /* the inputs one instruction takes */
#define INPUTS_PER_HALF_STEP 8
#define INPUTS_PER_LANE_PLACE 2                 /* a lane holds the inputs 2 t and 2 t + 1 of each half step */
#define BITS_PER_BYTE 8
#define BYTE_MASK 0xFFu

/* WEIGHT_FORMAT_E2M1_SCALED */
#define E2M1_BITS_PER_WEIGHT 4
#define E2M1_WEIGHTS_PER_BYTE 2
#define E2M1_SIGN_BIT 8u
#define E2M1_MAGNITUDE_MASK 7u
#define E2M1_MAGNITUDE_COUNT 8
#define E2M1_CODE_MASK 15u
#define E2M1_INPUTS_PER_LOAD 64
#define E2M1_STEPS_PER_LOAD 4
#define E2M1_REGISTERS_PER_WORD 4               /* r of the seam: a word holds two nibbles of each */
#define E2M1_HALVES_PER_REGISTER 2              /* h of the seam */
#define E2M1_REGISTERS_PER_OUTPUT_PAIR 2        /* r % 2 chooses the half of the tile, r / 2 the half of the step */
#define E2M1_BITS_PER_HALF 16
#define E2M1_BYTES_PER_WORD 4
#define E2M1_WORDS_PER_LOAD 128                 /* 32 lanes by 4 steps */
#define E2M1_WEIGHTS_PER_WORD 8
#define E2M1_SCALE_BYTES_PER_LOAD 32            /* 8 lane groups by the 4 bytes of a word */
#define E2M1_SCALES_PER_LOAD 2                  /* q of the seam: the two groups of 32 inputs of a load */
#define E2M1_SCALE_HALVES 2                     /* r of the seam's scales: the half of the tile */

/* WEIGHT_FORMAT_E4M3_SCALED */
#define E4M3_BITS_PER_WEIGHT 8
#define E4M3_SIGN_BIT 0x80u
#define E4M3_EXPONENT_SHIFT 3u
#define E4M3_EXPONENT_MASK 15u
#define E4M3_MANTISSA_MASK 7u
#define E4M3_MANTISSA_STEPS 8.0                 /* a mantissa counts eighths */
#define E4M3_EXPONENT_BIAS 7
#define E4M3_BYTES_PER_LANE 16
#define E4M3_BYTES_PER_GROUP 512                /* 32 lanes by 16 bytes */
#define E4M3_BYTES_PER_QUARTER 4                /* b / 4 of the seam: a quarter of a group is 8 inputs */
#define E4M3_BYTES_PER_INPUT 2                  /* b / 2 chooses the input of a pair, b % 2 the half of the tile */
#define E4M3_TILE_HALVES 2
#define E4M3_INPUTS_PER_PAIR 2
#define E4M3_OUTPUTS_PER_SCALE 32

#define E8M0_EXPONENT_BIAS 127
#define BINARY_RADIX 2.0

static const double e2m1_magnitudes[E2M1_MAGNITUDE_COUNT] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};

/* 2 to a whole power from -255 to 255. Exact: every factor and every product is a power of two that a double
 * holds, the largest of them 2^256. */
static double power_of_two(int32_t exponent)
{
    uint32_t doublings_left = (uint32_t)(exponent < 0 ? -exponent : exponent);
    double factor = BINARY_RADIX;               /* 2 to the power the lowest bit of doublings_left counts */
    double power = 1.0;
    while (doublings_left != 0) {
        power = (doublings_left & 1u) != 0 ? power * factor : power;
        factor *= factor;
        doublings_left >>= 1u;
    }
    return exponent < 0 ? 1.0 / power : power;
}

double test_weight_format_e2m1_value(uint32_t code)
{
    if (code >= TEST_WEIGHT_FORMAT_E2M1_CODE_COUNT) {
        return (double)NAN;
    }
    return (code & E2M1_SIGN_BIT) != 0 ? -e2m1_magnitudes[code & E2M1_MAGNITUDE_MASK]
                                       : e2m1_magnitudes[code & E2M1_MAGNITUDE_MASK];
}

double test_weight_format_e4m3_value(uint32_t code)
{
    const uint32_t exponent = (code >> E4M3_EXPONENT_SHIFT) & E4M3_EXPONENT_MASK;
    const double fraction = (double)(code & E4M3_MANTISSA_MASK) / E4M3_MANTISSA_STEPS;
    /* an exponent of 0 has no hidden one and counts as an exponent of 1 */
    const double magnitude = exponent == 0 ? fraction * power_of_two(1 - E4M3_EXPONENT_BIAS)
                                           : (1.0 + fraction) * power_of_two((int32_t)exponent - E4M3_EXPONENT_BIAS);
    if (code >= TEST_WEIGHT_FORMAT_BYTE_CODE_COUNT ||
        (code & TEST_WEIGHT_FORMAT_E4M3_NOT_A_NUMBER_MAGNITUDE) == TEST_WEIGHT_FORMAT_E4M3_NOT_A_NUMBER_MAGNITUDE) {
        return (double)NAN;
    }
    return (code & E4M3_SIGN_BIT) != 0 ? -magnitude : magnitude;
}

double test_weight_format_e8m0_value(uint32_t code)
{
    if (code >= TEST_WEIGHT_FORMAT_E8M0_NOT_A_NUMBER) {
        return (double)NAN;
    }
    return power_of_two((int32_t)code - E8M0_EXPONENT_BIAS);
}

result_status
test_weight_format_sizes_of(enum weight_format_code format,
			    const struct matrix_shape_record *shape,
			    struct test_weight_format_sizes *sizes_out)
{
	const bool is_e2m1 = format == WEIGHT_FORMAT_E2M1_SCALED;
	if (shape == NULL || sizes_out == NULL ||
	    (!is_e2m1 && format != WEIGHT_FORMAT_E4M3_SCALED)) {
		return STATUS_INVALID_ARGUMENT;
	}
    const uint64_t output_size = shape->output_size;
    const uint64_t input_size = shape->input_size;
    const uint64_t inputs_per_block = is_e2m1 ? E2M1_INPUTS_PER_LOAD : INPUTS_PER_SCALE;
    const uint32_t bits_per_weight = is_e2m1 ? E2M1_BITS_PER_WEIGHT : E4M3_BITS_PER_WEIGHT;
    if (output_size == 0 || output_size % OUTPUTS_PER_TILE != 0 || input_size == 0 ||
        input_size % inputs_per_block != 0 || shape->bits_per_weight != bits_per_weight ||
        shape->reserved_must_be_zero != 0) {
	    return STATUS_INVALID_ARGUMENT;
    }
    /* two sizes below 2^32 each: their product is below 2^64, and so is every count but the last sum */
    const uint64_t weight_count = output_size * input_size;
    const uint64_t weight_byte_count = is_e2m1 ? weight_count / E2M1_WEIGHTS_PER_BYTE : weight_count;
    const uint64_t scale_count =
        is_e2m1 ? output_size * (input_size / INPUTS_PER_SCALE)
                : (output_size + E4M3_OUTPUTS_PER_SCALE - 1) / E4M3_OUTPUTS_PER_SCALE * (input_size / INPUTS_PER_SCALE);
    if (scale_count > UINT64_MAX - weight_byte_count) {
	    return STATUS_INVALID_ARGUMENT;
    }
    *sizes_out = (struct test_weight_format_sizes){.weight_count = weight_count, .scale_count = scale_count,
                                                   .stored_byte_count = weight_byte_count + scale_count};
    return STATUS_OK;
}

/* Walks the words of the weights, [tile][load][lane][step], and then the bytes of the scale words, [tile][load][g]
 * with four bytes each, and fills each from the weight or the scale the seam puts there. */
static void lay_out_e2m1(const struct matrix_shape_record *shape,
			 const struct test_weight_format_sizes *sizes,
			 const uint8_t *codes, const uint8_t *scale_codes,
			 uint8_t *stored)
{
    const uint64_t input_size = shape->input_size;
    const uint64_t load_count = input_size / E2M1_INPUTS_PER_LOAD;
    const uint64_t scales_per_output = input_size / INPUTS_PER_SCALE;
    const uint64_t word_count = sizes->weight_count / E2M1_WEIGHTS_PER_WORD;
    uint8_t *const stored_scales = stored + word_count * E2M1_BYTES_PER_WORD;
    for (uint64_t word_index = 0; word_index < word_count; word_index++) {
        const uint64_t step = word_index % E2M1_STEPS_PER_LOAD;
        const uint64_t lane = word_index / E2M1_STEPS_PER_LOAD % LANES_PER_WARP;
        const uint64_t load = word_index / E2M1_WORDS_PER_LOAD % load_count;
        const uint64_t tile = word_index / E2M1_WORDS_PER_LOAD / load_count;
        const uint64_t lane_group = lane / LANES_PER_LANE_GROUP;                   /* g */
        const uint64_t lane_place = lane % LANES_PER_LANE_GROUP;                   /* t */
        uint32_t word = 0;
        for (uint64_t operand_register = 0; operand_register < E2M1_REGISTERS_PER_WORD; operand_register++) {
            for (uint64_t half = 0; half < E2M1_HALVES_PER_REGISTER; half++) {
                const uint64_t output_index =
                    OUTPUTS_PER_TILE * tile + lane_group +
                    OUTPUTS_PER_HALF_TILE * (operand_register % E2M1_REGISTERS_PER_OUTPUT_PAIR);
                const uint64_t input_index =
                    E2M1_INPUTS_PER_LOAD * load + INPUTS_PER_STEP * step + INPUTS_PER_LANE_PLACE * lane_place +
                    INPUTS_PER_HALF_STEP * (operand_register / E2M1_REGISTERS_PER_OUTPUT_PAIR) + half;
                const uint32_t code = codes[output_index * input_size + input_index];
                word |= code << (E2M1_BITS_PER_WEIGHT * operand_register + E2M1_BITS_PER_HALF * half);
            }
        }
        for (uint32_t byte_in_word = 0; byte_in_word < E2M1_BYTES_PER_WORD; byte_in_word++) {
            stored[word_index * E2M1_BYTES_PER_WORD + byte_in_word] =
                (uint8_t)((word >> (BITS_PER_BYTE * byte_in_word)) & BYTE_MASK);
        }
    }
    for (uint64_t scale_index = 0; scale_index < sizes->scale_count; scale_index++) {
        const uint64_t byte_in_word = scale_index % E2M1_BYTES_PER_WORD;            /* 2 q + r */
        const uint64_t lane_group = scale_index / E2M1_BYTES_PER_WORD % LANE_GROUPS_PER_TILE;
        const uint64_t load = scale_index / E2M1_SCALE_BYTES_PER_LOAD % load_count;
        const uint64_t tile = scale_index / E2M1_SCALE_BYTES_PER_LOAD / load_count;
        const uint64_t output_index =
            OUTPUTS_PER_TILE * tile + lane_group + OUTPUTS_PER_HALF_TILE * (byte_in_word % E2M1_SCALE_HALVES);
        const uint64_t scale_of_output = E2M1_SCALES_PER_LOAD * load + byte_in_word / E2M1_SCALE_HALVES;
        stored_scales[scale_index] = scale_codes[output_index * scales_per_output + scale_of_output];
    }
}

/* Walks the bytes of the weights, [tile][group][lane][16], and fills each from the weight the seam puts there; the
 * scales, [block][group], are stored in the arrangement they come in. */
static void lay_out_e4m3(const struct matrix_shape_record *shape,
			 const struct test_weight_format_sizes *sizes,
			 const uint8_t *codes, const uint8_t *scale_codes,
			 uint8_t *stored)
{
    const uint64_t input_size = shape->input_size;
    const uint64_t group_count = input_size / INPUTS_PER_SCALE;
    for (uint64_t byte_index = 0; byte_index < sizes->weight_count; byte_index++) {
        const uint64_t byte_in_lane = byte_index % E4M3_BYTES_PER_LANE;             /* b */
        const uint64_t lane = byte_index / E4M3_BYTES_PER_LANE % LANES_PER_WARP;
        const uint64_t group = byte_index / E4M3_BYTES_PER_GROUP % group_count;
        const uint64_t tile = byte_index / E4M3_BYTES_PER_GROUP / group_count;
        const uint64_t output_index = OUTPUTS_PER_TILE * tile + lane / LANES_PER_LANE_GROUP +
                                      OUTPUTS_PER_HALF_TILE * (byte_in_lane % E4M3_TILE_HALVES);
        const uint64_t input_index =
            INPUTS_PER_SCALE * group + INPUTS_PER_HALF_STEP * (byte_in_lane / E4M3_BYTES_PER_QUARTER) +
            INPUTS_PER_LANE_PLACE * (lane % LANES_PER_LANE_GROUP) +
            byte_in_lane / E4M3_BYTES_PER_INPUT % E4M3_INPUTS_PER_PAIR;
        stored[byte_index] = codes[output_index * input_size + input_index];
    }
    for (uint64_t scale_index = 0; scale_index < sizes->scale_count; scale_index++) {
        stored[sizes->weight_count + scale_index] = scale_codes[scale_index];
    }
}

result_status test_weight_format_lay_out(
	enum weight_format_code format, const struct matrix_shape_record *shape,
	const uint8_t *codes, uint64_t code_count, const uint8_t *scale_codes,
	uint64_t scale_code_count, uint8_t *stored, uint64_t stored_byte_count)
{
    struct test_weight_format_sizes sizes = {0};
    const result_status status =
	    test_weight_format_sizes_of(format, shape, &sizes);
    if (status != STATUS_OK || codes == NULL || scale_codes == NULL ||
	stored == NULL || code_count != sizes.weight_count ||
	scale_code_count != sizes.scale_count ||
	stored_byte_count != sizes.stored_byte_count) {
	    return STATUS_INVALID_ARGUMENT;
    }
    if (format == WEIGHT_FORMAT_E4M3_SCALED) {
	    lay_out_e4m3(shape, &sizes, codes, scale_codes, stored);
	    return STATUS_OK;
    }
    for (uint64_t code_index = 0; code_index < code_count; code_index++) {
        if (codes[code_index] > E2M1_CODE_MASK) {
		return STATUS_INVALID_ARGUMENT;
	}
    }
    lay_out_e2m1(shape, &sizes, codes, scale_codes, stored);
    return STATUS_OK;
}

/* From an output and an input to the nibble of the stored words and the byte of the stored scale words. */
static void read_e2m1(const struct matrix_shape_record *shape,
		      const uint8_t *stored, uint32_t output_index,
		      uint32_t input_index,
		      struct test_weight_format_weight *weight_out)
{
    const uint64_t output_size = shape->output_size;
    const uint64_t input_size = shape->input_size;
    const uint64_t tile_load = (uint64_t)(output_index / OUTPUTS_PER_TILE) * (input_size / E2M1_INPUTS_PER_LOAD) +
                               input_index / E2M1_INPUTS_PER_LOAD;
    const uint32_t tile_half = output_index % OUTPUTS_PER_TILE / OUTPUTS_PER_HALF_TILE;
    const uint32_t lane_group = output_index % OUTPUTS_PER_HALF_TILE;               /* g */
    const uint32_t input_in_step = input_index % INPUTS_PER_STEP;
    const uint32_t step = input_index % E2M1_INPUTS_PER_LOAD / INPUTS_PER_STEP;
    const uint32_t lane =
        LANES_PER_LANE_GROUP * lane_group + input_in_step % INPUTS_PER_HALF_STEP / INPUTS_PER_LANE_PLACE;
    const uint32_t operand_register =
        tile_half + E2M1_REGISTERS_PER_OUTPUT_PAIR * (input_in_step / INPUTS_PER_HALF_STEP);
    const uint32_t nibble_bit = E2M1_BITS_PER_WEIGHT * operand_register +
                                E2M1_BITS_PER_HALF * (input_in_step % INPUTS_PER_LANE_PLACE);
    const uint64_t word_index = (tile_load * LANES_PER_WARP + lane) * E2M1_STEPS_PER_LOAD + step;
    const uint32_t scale_byte_in_word =
        E2M1_SCALE_HALVES * (input_index % E2M1_INPUTS_PER_LOAD / INPUTS_PER_SCALE) + tile_half;    /* 2 q + r */
    const uint64_t scale_byte_index = output_size * input_size / E2M1_WEIGHTS_PER_BYTE +
                                      (tile_load * LANE_GROUPS_PER_TILE + lane_group) * E2M1_BYTES_PER_WORD +
                                      scale_byte_in_word;
    const uint32_t nibble_byte = stored[word_index * E2M1_BYTES_PER_WORD + nibble_bit / BITS_PER_BYTE];
    const uint32_t code = (nibble_byte >> (nibble_bit % BITS_PER_BYTE)) & E2M1_CODE_MASK;
    const uint32_t scale_code = stored[scale_byte_index];
    *weight_out = (struct test_weight_format_weight){
        .code = code, .scale_code = scale_code,
        .value = test_weight_format_e2m1_value(code) * test_weight_format_e8m0_value(scale_code)};
}

/* From an output and an input to the byte of the stored weights and the byte of the stored scales. */
static void read_e4m3(const struct matrix_shape_record *shape,
		      const uint8_t *stored, uint32_t output_index,
		      uint32_t input_index,
		      struct test_weight_format_weight *weight_out)
{
    const uint64_t output_size = shape->output_size;
    const uint64_t group_count = shape->input_size / INPUTS_PER_SCALE;
    const uint64_t tile_group =
        (uint64_t)(output_index / OUTPUTS_PER_TILE) * group_count + input_index / INPUTS_PER_SCALE;
    const uint32_t input_in_quarter = input_index % INPUTS_PER_HALF_STEP;
    const uint32_t lane = LANES_PER_LANE_GROUP * (output_index % OUTPUTS_PER_HALF_TILE) +
                          input_in_quarter / INPUTS_PER_LANE_PLACE;
    const uint32_t byte_in_lane =
        E4M3_BYTES_PER_QUARTER * (input_index % INPUTS_PER_SCALE / INPUTS_PER_HALF_STEP) +
        E4M3_BYTES_PER_INPUT * (input_in_quarter % E4M3_INPUTS_PER_PAIR) +
        output_index % OUTPUTS_PER_TILE / OUTPUTS_PER_HALF_TILE;
    const uint64_t byte_index = (tile_group * LANES_PER_WARP + lane) * E4M3_BYTES_PER_LANE + byte_in_lane;
    const uint64_t scale_byte_index = output_size * shape->input_size +
                                      (uint64_t)(output_index / E4M3_OUTPUTS_PER_SCALE) * group_count +
                                      input_index / INPUTS_PER_SCALE;
    const uint32_t code = stored[byte_index];
    const uint32_t scale_code = stored[scale_byte_index];
    *weight_out = (struct test_weight_format_weight){
        .code = code, .scale_code = scale_code,
        .value = test_weight_format_e4m3_value(code) * test_weight_format_e8m0_value(scale_code)};
}

result_status test_weight_format_read_weight(
	enum weight_format_code format, const struct matrix_shape_record *shape,
	const uint8_t *stored, uint64_t stored_byte_count,
	uint32_t output_index, uint32_t input_index,
	struct test_weight_format_weight *weight_out)
{
    struct test_weight_format_sizes sizes = {0};
    const result_status status =
	    test_weight_format_sizes_of(format, shape, &sizes);
    if (status != STATUS_OK || stored == NULL || weight_out == NULL ||
	stored_byte_count != sizes.stored_byte_count ||
	output_index >= shape->output_size ||
	input_index >= shape->input_size) {
	    return STATUS_INVALID_ARGUMENT;
    }
    if (format == WEIGHT_FORMAT_E4M3_SCALED) {
	    read_e4m3(shape, stored, output_index, input_index, weight_out);
    } else {
	    read_e2m1(shape, stored, output_index, input_index, weight_out);
    }
    return STATUS_OK;
}
