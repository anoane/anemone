/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * kernel_activation_quantize.cu: vectors of bfloat16 activations into the E4M3 bytes and E8M0 scale exponents that
 * the reference feeds to every quantized matrix (model.py linear(): act_quant with scales rounded to a power of
 * two, kernel.py act_quant_kernel). Device code only, built once per architecture into the module
 * "kernel_activation_quantize".
 *
 * Contract: include/anemone/linear_records.h defines the quantization, the three arrays and the launch. Here:
 *   - One warp takes one block of 32 consecutive values of one vector, a lane one value: warp w of block
 *     blockIdx.x of the grid takes block blockIdx.x * (blockDim.x / 32) + w of vector blockIdx.y, and does nothing
 *     where the vector has no such block. blockDim.x is a multiple of 32: every lane of a warp is there.
 *   - The largest magnitude of the block is a maximum over the warp; the exponent, the code and the operand of each
 *     value are kernel_activation_quantize.cuh's, which says how, and the lane of value 0 stores the exponent.
 *   - Each lane stores the one byte of its value where lane order puts it, and its operand where operand order puts
 *     it: no two lanes store to one address.
 *   - Nothing but the values of the vectors is read; nothing but their codes, exponents and operands is written.
 */
#include "kernel_activation_quantize.cuh"

#define EVERY_LANE 0xFFFFFFFFu
#define ACTIVATION_CARRY_SCALE 0x1p100f /* the operands' 2^100 (linear_records.h) */

extern "C" __global__ void
activation_quantize_entry(struct activation_quantize_arguments arguments,
			  unsigned char *arena)
{
    const uint32_t lane_index = threadIdx.x % THREADS_PER_WARP;
    const uint32_t block_index = blockIdx.x * (blockDim.x / THREADS_PER_WARP) + threadIdx.x / THREADS_PER_WARP;
    const uint32_t block_count = arguments.width / VALUES_PER_BLOCK;
    const size_t vector = blockIdx.y;
    if (block_index >= block_count) {
        return;
    }
    const uint16_t *block_values = (const uint16_t *)(arena + arguments.values_offset) + vector * arguments.width +
                                   (size_t)block_index * VALUES_PER_BLOCK;
    const float value = linear_bfloat16_value(block_values[lane_index]);
    float largest = fabsf(value);
    for (uint32_t distance = THREADS_PER_WARP >> 1u; distance != 0; distance >>= 1u) {
        largest = fmaxf(largest, __shfl_xor_sync(EVERY_LANE, largest, (int)distance));
    }
    const uint32_t exponent_code = quantize_exponent_code(largest);
    quantize_value_store(arena + arguments.codes_offset + vector * arguments.width,
                         (uint16_t *)(arena + arguments.operands_offset) + vector * arguments.width, block_index,
                         lane_index, value, exponent_code, quantize_inverse_scale(exponent_code));
    if (lane_index == 0) {
        arena[arguments.exponents_offset + vector * block_count + block_index] = (unsigned char)exponent_code;
    }
}

/* draft_bf16_operands (track L): the codes and exponents of activation_quantize_entry, each operand the value's own
 * bfloat16 times 2^100: exact (a power of two times a bfloat16, normal for every finite bfloat16 below 2^28 in
 * magnitude; an infinity of its sign from 2^28 on, beyond any activation), so a carried product (class 80:
 * linear_records.h) reads the rows unrounded in place of their E4M3 codes. */
extern "C" __global__ void
activation_carry(struct activation_quantize_arguments arguments,
		 unsigned char *arena)
{
    const uint32_t lane_index = threadIdx.x % THREADS_PER_WARP;
    const uint32_t block_index = blockIdx.x * (blockDim.x / THREADS_PER_WARP) + threadIdx.x / THREADS_PER_WARP;
    const uint32_t block_count = arguments.width / VALUES_PER_BLOCK;
    const size_t vector = blockIdx.y;
    if (block_index >= block_count) {
        return;
    }
    const uint16_t *block_values = (const uint16_t *)(arena + arguments.values_offset) + vector * arguments.width +
                                   (size_t)block_index * VALUES_PER_BLOCK;
    const float value = linear_bfloat16_value(block_values[lane_index]);
    float largest = fabsf(value);
    for (uint32_t distance = THREADS_PER_WARP >> 1u; distance != 0; distance >>= 1u) {
        largest = fmaxf(largest, __shfl_xor_sync(EVERY_LANE, largest, (int)distance));
    }
    const uint32_t exponent_code = quantize_exponent_code(largest);
    uint32_t code_in_word = 0, operand_in_word = 0, rounded = 0;
    const size_t word = quantize_word(block_index, lane_index, &code_in_word, &operand_in_word);
    const uint32_t code = quantize_nearest_e4m3(value * quantize_inverse_scale(exponent_code), exponent_code, &rounded);
    (arena + arguments.codes_offset + vector * arguments.width)[word * QUANTIZE_VALUES_PER_WORD + code_in_word] =
        (unsigned char)code;
    ((uint16_t *)(arena + arguments.operands_offset) + vector * arguments.width)[word * QUANTIZE_VALUES_PER_WORD +
                                                                                 operand_in_word] =
        (uint16_t)(__float_as_uint(value * ACTIVATION_CARRY_SCALE) >> BITS_PER_HALF);
    if (lane_index == 0) {
        arena[arguments.exponents_offset + vector * block_count + block_index] = (unsigned char)exponent_code;
    }
}
