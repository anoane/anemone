/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * kernel_activation_quantize.cuh: the quantization of one block of 32 bfloat16 values, which
 * kernel_activation_quantize.cu and the fused site front (kernel_site_front.cu) share: the exponent of a block from
 * its largest magnitude, and the code and the operand of each value, stored in lane order and operand order. Device
 * code only.
 *
 * Contract: include/anemone/linear_records.h defines the quantization and the arrays. Here:
 *   - The exponent is read from the bits of float32(max(largest, 1e-4) * (1 / 448)), as the reference reads it.
 *   - Dividing by the scale is a multiplication by its inverse, a power of two: the same float32. The code is found
 *     in the exponent field of the quotient: E4M3 numbers are 2^(f - 130) apart for a float32 exponent field f,
 *     and 2^-9 apart below 2^-6, so the magnitude in units of that distance, rounded to the nearest whole number
 *     with a tie to even, is added to the code of the first number of that stretch. The whole numbers of a stretch
 *     alternate with the parity of their codes: a tie goes to the even code, and 448 is the code 0x7E.
 *   - The operand of a value is its code's units times the power of two of the code's stretch, times 2^(c - 127)
 *     and 2^100: units times 2^(f + c - 30 - 127) for a float32 exponent field f. Up to
 *     ACTIVATION_CARRIED_EXPONENT_LARGEST that field is that of a normal number and the product of at most
 *     16 units is exact, and so its upper half; above it the operand is an infinity, of the code's sign.
 *   - The largest magnitude of a block is a maximum, which has no order: any tree of fmaxf over the block's fabsf
 *     values gives the same float32 (a not-a-number drops out unless every value is one), so the exponent is the one
 *     the definition gives whoever forms it.
 */
#ifndef KERNEL_ACTIVATION_QUANTIZE_CUH
#define KERNEL_ACTIVATION_QUANTIZE_CUH

#include "kernel_linear_common.cuh"

#define QUANTIZE_MAGNITUDE_FLOOR 1.0e-4f        /* the reference's lower bound of a block's largest magnitude */
#define QUANTIZE_E4M3_LARGEST 448.0f
#define QUANTIZE_E4M3_LARGEST_INVERSE (1.0f / 448.0f) /* float32(1 / 448): the reference multiplies by it */
#define QUANTIZE_E4M3_SMALLEST_NORMAL 0.015625f /* 2^-6 */
#define QUANTIZE_E4M3_SIGN_BIT 0x80u
#define QUANTIZE_SIGN_FROM_FLOAT32_SHIFT 24u    /* moves the sign of a float32 to the sign of a code */
#define QUANTIZE_E4M3_MANTISSA_BITS 3u
#define QUANTIZE_UNITS_EXPONENT_BASE 257u       /* 2 * 127 + 3: a field f has units of 2^(f - 130) */
#define QUANTIZE_CODE_BASE 968u                 /* ((127 - 7) << 3) + 8: code = 8 f + units - this */
#define QUANTIZE_INVERSE_SCALE_EXPONENT_BASE 254u /* 127 + 127: the inverse of the scale of code c has field 254 - c */
#define QUANTIZE_FLOAT32_EXPONENT_MASK 0xFFu
#define QUANTIZE_FLOAT32_MANTISSA_MASK 0x007FFFFFu
#define QUANTIZE_VALUES_PER_STEP 16u            /* the inputs of one instruction */
#define QUANTIZE_VALUES_PER_HALF_STEP 8u
#define QUANTIZE_VALUES_PER_LANE_PLACE 2u       /* a lane place holds inputs 2 t and 2 t + 1 of each half */
#define QUANTIZE_REGISTERS_PER_OPERAND_B 2u
#define QUANTIZE_VALUES_PER_WORD 4u             /* a lane place's codes, or operands, of one instruction */
#define QUANTIZE_CARRIED_EXPONENT_BASE 30u      /* 130 - 100: see the top */
#define QUANTIZE_BFLOAT16_INFINITY 0x7F80u
#define QUANTIZE_CODE_SIGN_TO_BFLOAT16_SHIFT 8u
#define QUANTIZE_EVERY_LANE 0xFFFFFFFFu

/* The E4M3 code nearest to scaled, its magnitude limited to 448, a tie to the even code; the sign is scaled's. Its
 * operand, of a block of exponent code exponent_code, into operand_out. */
__device__ static __forceinline__ uint32_t quantize_nearest_e4m3(float scaled, uint32_t exponent_code,
                                                                 uint32_t *operand_out)
{
    const float magnitude = fminf(fabsf(scaled), QUANTIZE_E4M3_LARGEST);
    const uint32_t exponent_field =
        __float_as_uint(fmaxf(magnitude, QUANTIZE_E4M3_SMALLEST_NORMAL)) >> FLOAT32_EXPONENT_SHIFT;
    const float units_per_one =
        __uint_as_float((QUANTIZE_UNITS_EXPONENT_BASE - exponent_field) << FLOAT32_EXPONENT_SHIFT);
    const uint32_t units = (uint32_t)__float2int_rn(magnitude * units_per_one);          /* 0 .. 16 */
    const uint32_t sign = (__float_as_uint(scaled) >> QUANTIZE_SIGN_FROM_FLOAT32_SHIFT) & QUANTIZE_E4M3_SIGN_BIT;
    const uint32_t carried_field = exponent_field + exponent_code - QUANTIZE_CARRIED_EXPONENT_BASE;
    const float carried = (float)units * __uint_as_float(carried_field << FLOAT32_EXPONENT_SHIFT);
    const bool is_carried = exponent_code <=
			    ACTIVATION_CARRIED_EXPONENT_LARGEST;
    *operand_out = (is_carried ? __float_as_uint(carried) >> BITS_PER_HALF : QUANTIZE_BFLOAT16_INFINITY) |
                   (sign << QUANTIZE_CODE_SIGN_TO_BFLOAT16_SHIFT);
    return ((exponent_field << QUANTIZE_E4M3_MANTISSA_BITS) + units - QUANTIZE_CODE_BASE) | sign;
}

/* Where value value_in_block of block block_index lies in lane order and in operand order: the word of the lane
 * place and instruction that take it (four codes, or four operands), and its place in that word: of a code half h
 * by 2 plus register r, of an operand r by 2 plus h (linear_records.h). */
__device__ static __forceinline__ size_t quantize_word(uint32_t block_index, uint32_t value_in_block,
                                                       uint32_t *code_in_word_out, uint32_t *operand_in_word_out)
{
    const uint32_t step = block_index % BLOCKS_PER_LOAD * STEPS_PER_BLOCK + value_in_block / QUANTIZE_VALUES_PER_STEP;
    const uint32_t value_in_step = value_in_block % QUANTIZE_VALUES_PER_STEP;
    const uint32_t lane_place = value_in_step % QUANTIZE_VALUES_PER_HALF_STEP / QUANTIZE_VALUES_PER_LANE_PLACE;
    const uint32_t half = value_in_step % QUANTIZE_VALUES_PER_LANE_PLACE;
    const uint32_t operand_register = value_in_step / QUANTIZE_VALUES_PER_HALF_STEP;
    *code_in_word_out = half * QUANTIZE_REGISTERS_PER_OPERAND_B + operand_register;
    *operand_in_word_out = operand_register * QUANTIZE_VALUES_PER_LANE_PLACE + half;
    return ((size_t)(block_index / BLOCKS_PER_LOAD) * LANES_PER_LANE_GROUP + lane_place) * STEPS_PER_LOAD + step;
}

/* The exponent code of a block of largest magnitude largest, and the inverse of its scale. */
__device__ static __forceinline__ uint32_t quantize_exponent_code(float largest)
{
    const uint32_t ratio_bits =
        __float_as_uint(fmaxf(largest, QUANTIZE_MAGNITUDE_FLOOR) * QUANTIZE_E4M3_LARGEST_INVERSE);
    return ((ratio_bits >> FLOAT32_EXPONENT_SHIFT) & QUANTIZE_FLOAT32_EXPONENT_MASK) +
           ((ratio_bits & QUANTIZE_FLOAT32_MANTISSA_MASK) != 0 ? 1u : 0u);
}
__device__ static __forceinline__ float quantize_inverse_scale(uint32_t exponent_code)
{
    return __uint_as_float((QUANTIZE_INVERSE_SCALE_EXPONENT_BASE - exponent_code) << FLOAT32_EXPONENT_SHIFT);
}

/* Value value_in_block of block block_index of one vector, of the block's exponent code and inverse scale: its code
 * into the vector's codes (lane order) and its operand into the vector's operands (operand order). */
__device__ static __forceinline__ void quantize_value_store(unsigned char *vector_codes, uint16_t *vector_operands,
                                                            uint32_t block_index, uint32_t value_in_block,
                                                            float value, uint32_t exponent_code, float inverse_scale)
{
    uint32_t code_in_word = 0, operand_in_word = 0, operand = 0;
    const size_t word = quantize_word(block_index, value_in_block, &code_in_word, &operand_in_word);
    const uint32_t code = quantize_nearest_e4m3(value * inverse_scale, exponent_code, &operand);
    vector_codes[word * QUANTIZE_VALUES_PER_WORD + code_in_word] = (unsigned char)code;
    vector_operands[word * QUANTIZE_VALUES_PER_WORD + operand_in_word] = (uint16_t)operand;
}

/* One block of 32 values of a vector, by one warp, lane i value i of the block (its bfloat16 bits value_bits): the
 * largest magnitude over the warp, the block's exponent (stored by lane 0 at block_index of the vector's exponents),
 * each lane's code and operand where lane order and operand order put them: what activation_quantize_entry does
 * for a block, the one function every kernel that quantizes a warp's block shares. Every lane of the warp calls it. */
__device__ static __forceinline__ void activation_quantize_block(uint32_t value_bits, uint32_t block_index,
                                                                 uint32_t lane_index, unsigned char *vector_codes,
                                                                 unsigned char *vector_exponents,
                                                                 uint16_t *vector_operands)
{
    const float value = linear_bfloat16_value(value_bits);
    float largest = fabsf(value);
    for (uint32_t distance = THREADS_PER_WARP >> 1u; distance != 0; distance >>= 1u) {
        largest = fmaxf(largest, __shfl_xor_sync(QUANTIZE_EVERY_LANE, largest, (int)distance));
    }
    const uint32_t exponent_code = quantize_exponent_code(largest);
    quantize_value_store(vector_codes, vector_operands, block_index, lane_index, value, exponent_code,
                         quantize_inverse_scale(exponent_code));
    if (lane_index == 0) {
        vector_exponents[block_index] = (unsigned char)exponent_code;
    }
}

#endif
