/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * native_expert_forms_kernels.cuh: the routed experts of one layer in the checkpoint's own format (E2M1 weights,
 * E4M3 activations) as five launches a step: the rows' inputs quantized, gate with up and the hidden step, the
 * hidden vectors quantized, down, a row's six experts added. The step, the stored experts and the synthetic data
 * are shared_rows_expert_probe.cu's; the product kernel is written anew as ONE body, forms_product, of which every
 * lever of round 1, package 2 of the speed campaign (research/speed/ROUND_1.md) is a compile-time choice:
 *   operand     what a lane takes as operand B. CODES: 16 bytes of E4M3 codes a load, placed in the loop as fp16 of
 *               the value times 2^-8 (the earlier probe). PLACED16: those fp16 words, written by the quantizer in
 *               lane order: two loads, no placement. CARRIED16: fp16 of the dequantized value times 2^7, so that a
 *               block's power of two comes from the weights' scale code alone and no exponent is read; exact for a
 *               block whose scale is at most 1 (largest magnitude at most 448). A block above it keeps the placed
 *               words: the quantizer says per vector whether it has one (unfit), a lane then reads its two columns'
 *               exponents and takes, per block and column, the scale that belongs to what the block holds: no
 *               result depends on another column's vector. CARRIED_BF16: bfloat16 of the dequantized value times
 *               2^100 with the bfloat16 instruction, the weights placed as their value times 2^-126; exact up to a
 *               block scale of 2^19 (a dequantized 1.75 * 2^27). Above it the quantizer writes infinities, and a
 *               sum they reach is infinite or not-a-number, never a wrong finite number.
 *   ahead       loads asked that many loads before their use and held in registers: weights, scale word, operand
 *               words, exponents. A slice never asks beyond its own last load.
 *   persistent  the grid is as many blocks as the card holds at once and a block walks the units (a tile of a job)
 *               blockIdx.x, blockIdx.x + gridDim.x, .. A unit's chains lie in one of two halves of the block's
 *               shared memory in turn, so that one barrier a unit is enough.
 *   slices      warps that share the input width of one matrix of one tile, each a chain of its own: slice i holds
 *               loads floor(i L / S) to floor((i + 1) L / S) - 1 of the width's L. Bits depend on S.
 *   columns     2: every sum is scaled. 1: only the even column's two, for a launch whose jobs all have one row.
 *   by_rows     1 only in the kernel that shows the row-exactness check failing: never timed.
 * The order of additions is the engine's (engine/include/anemone/linear_records.h): inside an instruction the 16
 * products of one output and one column; the two instructions of a block of 32 inputs, the second on the sum of
 * the first; one fused multiply-add per block in ascending order from +0 within a slice; the slices in ascending
 * order from +0. Only slices and by_rows touch that order, and a carried operand changes no bit if the tensor cores
 * multiply exactly and add as a floating-point unit does: the probe compares, it does not assume.
 *
 * What a launch must satisfy (a kernel checks nothing; native_expert_forms_probe.cu validates every launch on the
 * host): the block and grid stated at FORMS_PRODUCT_KERNELS and at the quantizer; every job's expert inside the
 * stored experts, its rows 1 .. FORMS_MOST_ROWS, each of its eight pair indices a pair of the step; the arrays of
 * struct forms_vectors and forms_step as large as the step's rows and pairs need. No atomic operation. A scale
 * code is taken to keep every power of two a normal fp32, as the probe's stored codes do. At the end, as in
 * card_work_kernels.cuh, the host's references. Measurement tool: not part of the engine.
 */
#ifndef NATIVE_EXPERT_FORMS_KERNELS_CUH
#define NATIVE_EXPERT_FORMS_KERNELS_CUH
#include <cuda_fp16.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

enum {
    FORMS_HIDDEN_SIZE = 5120, FORMS_INNER_SIZE = 2304, FORMS_EXPERTS_PER_ROW = 6,
    FORMS_MOST_ROWS = 8,                         /* rows of a batch: the eight columns of the instruction */
    FORMS_MOST_PAIRS = FORMS_MOST_ROWS * FORMS_EXPERTS_PER_ROW,       /* a pair: a row at one of its experts */
    FORMS_GATE_MATRIX = 0, FORMS_UP_MATRIX = 1, FORMS_DOWN_MATRIX = 2, FORMS_MATRIX_KINDS = 3,
    FORMS_GATE_UP_MATRICES = 2, FORMS_DOWN_MATRICES = 1,              /* the matrices of a block of each product */
    FORMS_LANES_PER_WARP = 32, FORMS_LANES_PER_LANE_GROUP = 4, FORMS_LANE_GROUPS = 8,
    FORMS_OUTPUTS_PER_TILE = 16, FORMS_HALF_TILE = 8,                 /* the instruction's M; a lane's two outputs */
    FORMS_INPUTS_PER_STEP = 16, FORMS_HALF_INPUTS = 8,                /* its K; a step is one instruction */
    FORMS_VALUES_PER_BLOCK = 32,                 /* inputs that share one scale: of the weights, of a vector */
    FORMS_STEPS_PER_BLOCK = 2, FORMS_BLOCKS_PER_LOAD = 2, FORMS_STEPS_PER_LOAD = 4, FORMS_INPUTS_PER_LOAD = 64,
    FORMS_WORDS_PER_BLOCK = FORMS_LANES_PER_LANE_GROUP * FORMS_STEPS_PER_BLOCK,       /* of a vector's codes */
    FORMS_ELEMENTS_PER_LANE = 4, FORMS_COLUMNS_PER_LANE = 2, FORMS_OUTPUTS_PER_LANE = 2,
    FORMS_REGISTERS_PER_OPERAND_A = 4, FORMS_REGISTERS_PER_OPERAND_B = 2, FORMS_HALVES_PER_REGISTER = 2,
    FORMS_FED_LOADS = 2,                         /* 16-byte loads of operand words a load of 64 inputs */
    FORMS_FED_WORDS = FORMS_STEPS_PER_LOAD * FORMS_REGISTERS_PER_OPERAND_B,
    FORMS_MOST_SCALE_BLOCKS = FORMS_HIDDEN_SIZE / FORMS_VALUES_PER_BLOCK,
    FORMS_ELEMENT_THREADS = 256,                 /* the block of a kernel whose threads take one value each */
    FORMS_CHAIN_TURNS = 2,                       /* persistent blocks: the halves of shared memory units alternate in */
    /* a stored matrix: per (tile of 16 outputs, load of 64 inputs) 32 lanes of four words, and 8 scale words */
    FORMS_TILE_LOADS_PER_MATRIX =
        FORMS_HIDDEN_SIZE * FORMS_INNER_SIZE / (FORMS_OUTPUTS_PER_TILE * FORMS_INPUTS_PER_LOAD),
    FORMS_PACKED_WORDS_PER_MATRIX = FORMS_TILE_LOADS_PER_MATRIX * FORMS_LANES_PER_WARP * FORMS_STEPS_PER_LOAD,
    FORMS_SCALE_WORDS_PER_MATRIX = FORMS_TILE_LOADS_PER_MATRIX * FORMS_LANE_GROUPS,
    FORMS_BYTE_BITS = 8, FORMS_BYTE_MASK = 0xFF, FORMS_HALF_BITS = 16, FORMS_HALF_MASK = 0xFFFF, FORMS_WORD_BYTES = 4,
    FORMS_NIBBLE_BITS = 4, FORMS_NIBBLE_MASK = 15,
    /* E2M1: sign, two exponent bits, one mantissa bit. Its three magnitude bits at bits 9 .. 11 of an fp16 are the
     * value times 2^-14, at bits 6 .. 8 of a bfloat16 the value times 2^-126 (0.5 a subnormal in both). */
    FORMS_E2M1_MAGNITUDES = 0x00070007, FORMS_FP16_MAGNITUDE_BIT = 9, FORMS_BF16_MAGNITUDE_BIT = 6,
    FORMS_E2M1_SIGN_SHIFT = 12, FORMS_FP16_WEIGHT_EXPONENT = 14, FORMS_BF16_WEIGHT_EXPONENT = 126,
    /* E4M3: sign, four exponent bits (bias 7), three mantissa bits; the largest value is 448. Its seven magnitude
     * bits at bits 7 .. 13 of an fp16 are the value times 2^-8, subnormal codes included. */
    FORMS_E4M3_PAIR_SIGNS = 0x00800080, FORMS_E4M3_PAIR_MAGNITUDES = 0x007F007F, FORMS_E4M3_MAGNITUDE_SHIFT = 7,
    FORMS_E4M3_SIGN_BIT = 0x80, FORMS_E4M3_EXPONENT_SHIFT = 3, FORMS_E4M3_EXPONENT_BIAS = 7,
    FORMS_E4M3_IMPLICIT_ONE = 8, FORMS_E4M3_SIGN_FROM_FP32_SHIFT = 24, FORMS_CODES_EXPONENT = 8,
    FORMS_E8M0_BIAS = 127,                       /* a scale is 2^(code - 127) */
    FORMS_FP32_EXPONENT_SHIFT = 23, FORMS_FP32_EXPONENT_BIAS = 127, FORMS_FP32_MANTISSA_MASK = 0x007FFFFF,
    FORMS_FP32_INFINITY_FIELD = 255,
    /* quantizing a value of fp32 exponent field f: codes are 2^(f - 130) apart, the code is 8 (f - 120) + steps - 8 */
    FORMS_E4M3_STEPS_EXPONENT_BASE = 2 * FORMS_FP32_EXPONENT_BIAS + FORMS_E4M3_EXPONENT_SHIFT,
    FORMS_E4M3_CODE_BASE = ((FORMS_FP32_EXPONENT_BIAS - FORMS_E4M3_EXPONENT_BIAS) << FORMS_E4M3_EXPONENT_SHIFT)
                           + FORMS_E4M3_IMPLICIT_ONE,
    /* what a carried operand is multiplied by, as a power of two; and the exponent field of a block's power of two
     * less the weights' scale code: with placed codes it is also the vector's exponent less FORMS_PLACED_BIAS */
    FORMS_CARRIED16_EXPONENT = 7, FORMS_CARRIED_BF16_EXPONENT = 100,
    FORMS_PLACED_BIAS = FORMS_E8M0_BIAS - FORMS_FP16_WEIGHT_EXPONENT - FORMS_CODES_EXPONENT,
    FORMS_CARRIED16_FIELD = FORMS_FP16_WEIGHT_EXPONENT - FORMS_CARRIED16_EXPONENT,
    FORMS_CARRIED_BF16_FIELD = FORMS_BF16_WEIGHT_EXPONENT - FORMS_CARRIED_BF16_EXPONENT,
    FORMS_OPERAND_CODES = 0, FORMS_OPERAND_PLACED16, FORMS_OPERAND_CARRIED16, FORMS_OPERAND_CARRIED_BF16,
    /* synthetic inputs: multiples of 1/64 in -2 .. 2, divided per block of 32 by 2^0 .. 2^6 or by 2^16 (under the
     * reference's floor); and the kinds of block the checked vectors hold in turn (forms_input_value) */
    FORMS_INPUT_VALUE_STEPS = 255, FORMS_INPUT_BLOCK_KINDS = 8, FORMS_INPUT_TINY_SHIFT = 16, FORMS_STEP_STRIDE = 977,
    FORMS_MIX_SHIFT_OUTER = 16, FORMS_MIX_SHIFT_INNER = 15,
    FORMS_CORNER_PLAIN = 0, FORMS_CORNER_SUBNORMAL_CODES, FORMS_CORNER_ZEROS, FORMS_CORNER_FLOOR,
    FORMS_CORNER_LARGEST_CODE, FORMS_CORNER_ABOVE_448, FORMS_CORNER_KINDS,
    FORMS_CORNER_STEPS = 15, FORMS_CORNER_TIE_BASE = 17, FORMS_CORNER_TIE_STEPS = 7, FORMS_CORNER_SHIFTS = 3,
    /* for the host's references. binary16: sign, five exponent bits, ten mantissa bits: (1024 + mantissa) *
     * 2^(field - 25), or with field 0 mantissa * 2^-24 */
    FORMS_E4M3_NOT_A_NUMBER = 0x7F, FORMS_E4M3_EXPONENT_MASK = 15, FORMS_E4M3_MANTISSA_MASK = 7,
    FORMS_BF16_SIGNIFICANT_BITS = 8, FORMS_FP16_SIGN = 0x8000, FORMS_FP16_MANTISSA_BITS = 10,
    FORMS_FP16_MANTISSA_MASK = 1023, FORMS_FP16_EXPONENT_MASK = 31, FORMS_FP16_IMPLICIT_ONE = 1024,
    FORMS_FP16_EXPONENT_BIAS = 25, FORMS_FP16_SUBNORMAL_EXPONENT = 24
};
#define FORMS_BLOCK_THREADS(matrices, slices) ((matrices) * (slices) * FORMS_LANES_PER_WARP)   /* of a product */
#define FORMS_PAIR_SIGN_MASK 0x80008000u         /* bit 15 of both halves of a register */
#define FORMS_BF16_ROUNDING_BIAS 0x7FFFu         /* plus the kept half's lowest bit: to nearest, a tie to even */
#define FORMS_BF16_KEPT_MASK 0xFFFF0000u
#define FORMS_PACKED_SEED 0x243F6A88u
#define FORMS_SCALE_SEED 0x85A308D3u
#define FORMS_SCALE_WORD_BASE 0x76767676u        /* synthetic weight scales: E8M0 codes 118 .. 121 */
#define FORMS_SCALE_WORD_SPREAD 0x03030303u
#define FORMS_INPUT_SEED 0x1234567u
#define FORMS_MIX_MULTIPLIER_FIRST 0x7feb352du
#define FORMS_MIX_MULTIPLIER_SECOND 0x846ca68bu
#define FORMS_SIGN_BIT 0x80000000u
#define FORMS_SWIGLU_LIMIT 10.0f
#define FORMS_E4M3_VALUE_DIVISOR 1024.0          /* 2^(bias + mantissa bits) */
#define FORMS_E4M3_MAXIMUM 448.0f
#define FORMS_E4M3_MAXIMUM_INVERSE (1.0f / FORMS_E4M3_MAXIMUM)
#define FORMS_E4M3_SMALLEST_NORMAL 0.015625f     /* 2^-6: codes below it are 2^-9 apart, as those just above it */
#define FORMS_AMAX_FLOOR 1.0e-4f                 /* the reference's lower bound of a block's largest magnitude */
#define FORMS_INPUT_VALUE_OFFSET 128.0f
#define FORMS_INPUT_VALUE_DIVISOR 64.0f
#define FORMS_CORNER_SUBNORMAL_DIVISOR 262144.0f /* 2^18: beside a 1.0, steps of half a subnormal code */
#define FORMS_CORNER_FLOOR_DIVISOR 32768.0f      /* 2^15: every magnitude under the floor */
#define FORMS_CORNER_LARGEST 1.6875f             /* 432 * 2^-8: a tie between the two largest codes at its scale */
#define FORMS_CORNER_TIE_DIVISOR 4096.0f         /* odd multiples of 2^-12: ties between codes at that scale */
#define FORMS_CORNER_ABOVE 1000.0f               /* a block scale of 4: no fp16 holds its values times 2^7 */

struct forms_job {                               /* one expert with the rows routed to it: the columns of a block */
    uint32_t expert_index;
    uint32_t row_count;                          /* columns in use, 1 .. FORMS_MOST_ROWS */
    uint32_t pair_index[FORMS_MOST_ROWS];        /* of each column; a column not in use repeats column 0 */
    float route_weight[FORMS_MOST_ROWS];
};
struct forms_vectors {                           /* the vectors in front of a product and what the quantizer writes */
    float *values;                               /* [vector][size]: the rows' inputs; the pairs' hidden vectors */
    uint32_t *codes;                             /* [vector][load][lane place][step]: four E4M3 bytes a word */
    uint8_t *exponents;                          /* [vector][block of 32]: E8M0 codes */
    uint32_t *operands;                          /* [vector][load][lane place][step][register of operand B] */
    uint32_t *unfit;                             /* [vector], CARRIED16: 1 where a block's scale is above 1 */
    uint32_t size, pairs_per_vector;             /* six pairs read a row's input, one its own hidden vector */
};
struct forms_step {                              /* what the launches of one step read and write */
    const uint4 *packed;                         /* [expert][matrix][tile][load][lane]: eight E2M1 codes a word */
    const uint32_t *scales;                      /* [expert][matrix][tile][load][lane group]: four E8M0 codes */
    const struct forms_job *jobs;
    struct forms_vectors input, hidden;          /* of the rows; of the pairs */
    float *pair_output, *row_output;             /* [pair][hidden size]; [row][hidden size] */
    uint32_t job_count;
};

/* ---- synthetic data and layouts: the same functions on the host and on the card ---- */

__host__ __device__ inline uint32_t forms_mix_bits(uint32_t value)
{
    value ^= value >> FORMS_MIX_SHIFT_OUTER;
    value *= FORMS_MIX_MULTIPLIER_FIRST;
    value ^= value >> FORMS_MIX_SHIFT_INNER;
    value *= FORMS_MIX_MULTIPLIER_SECOND;
    value ^= value >> FORMS_MIX_SHIFT_OUTER;
    return value;
}

/* A stored word is a hash of its place in its array: eight weight nibbles, or the E8M0 codes of four scales. */
__host__ __device__ inline uint32_t forms_stored_word(uint64_t word_index, uint32_t is_scale)
{
    const uint32_t seed = is_scale != 0u ? FORMS_SCALE_SEED : FORMS_PACKED_SEED;
    const uint32_t bits = forms_mix_bits(seed + (uint32_t)word_index);
    return is_scale != 0u ? FORMS_SCALE_WORD_BASE + (bits & FORMS_SCALE_WORD_SPREAD) : bits;
}

/* Value `column` of vector `vector` of a step. Timed steps (corners 0) have the earlier probe's values. The checked
 * step (corners 1) gives block b of vector v the kind (b + v) % FORMS_CORNER_KINDS, so that every kind lies beside
 * every other in a vector and in the neighbouring column: values that become subnormal codes and ties beside a
 * 1.0 (magnitudes spread over 2^18); zeros of both signs; a block under the floor of 1e-4; a largest value that is a
 * tie up to the largest code, the others on ties; and, in odd vectors only (a lane must also be seen without it),
 * a block above 448. Every value is exact in fp32. */
__host__ __device__ inline float forms_input_value(uint32_t vector, uint32_t column, uint32_t step, uint32_t corners)
{
    const uint32_t seed = FORMS_INPUT_SEED + step * FORMS_STEP_STRIDE + vector * FORMS_HIDDEN_SIZE;
    const uint32_t block = column / FORMS_VALUES_PER_BLOCK, bits = forms_mix_bits(seed + column);
    const uint32_t kind = (block + vector) % FORMS_CORNER_KINDS, steps = (bits >> FORMS_BYTE_BITS) & FORMS_CORNER_STEPS;
    const int is_first = column % FORMS_VALUES_PER_BLOCK == 0;
    const float value =
        ((float)(bits & FORMS_INPUT_VALUE_STEPS) - FORMS_INPUT_VALUE_OFFSET) / FORMS_INPUT_VALUE_DIVISOR;
    const float sign = (bits & FORMS_SIGN_BIT) != 0u ? -1.0f : 1.0f;
    if (corners == 0u) {
        const uint32_t shift = forms_mix_bits(~seed + block) % FORMS_INPUT_BLOCK_KINDS;      /* the last kind: tiny */
        return value / (float)(1u << (shift < FORMS_INPUT_BLOCK_KINDS - 1u ? shift : (uint32_t)FORMS_INPUT_TINY_SHIFT));
    }
    if (kind == FORMS_CORNER_SUBNORMAL_CODES) {
        return is_first ? 1.0f : sign * (float)steps / FORMS_CORNER_SUBNORMAL_DIVISOR;
    }
    if (kind == FORMS_CORNER_ZEROS) {
        return (steps & 1u) != 0u ? value : sign * 0.0f;
    }
    if (kind == FORMS_CORNER_FLOOR) {
        return value / FORMS_CORNER_FLOOR_DIVISOR;
    }
    if (kind == FORMS_CORNER_LARGEST_CODE) {
        const uint32_t tie = FORMS_CORNER_TIE_BASE + ((steps & FORMS_CORNER_TIE_STEPS) << 1);      /* odd */
        return is_first ? FORMS_CORNER_LARGEST : sign * (float)tie / FORMS_CORNER_TIE_DIVISOR;
    }
    if (kind == FORMS_CORNER_ABOVE_448 && (vector & 1u) != 0u && is_first) {
        return FORMS_CORNER_ABOVE;
    }
    return value / (float)(1u << (steps & FORMS_CORNER_SHIFTS));
}

/* The order of a quantized vector's bytes. Lane place t of the column that carries the vector needs, for each step,
 * inputs 2t and 2t + 1 (register 0 of operand B, low half first) and 2t + 8 and 2t + 9 (register 1). Its four bytes
 * are one word: bytes 0 and 2 are register 0, bytes 1 and 3 register 1. forms_activation_offset is the place of a
 * byte's value in its block of 32; the words are stored as [load][lane place][step of the load]. */
__host__ __device__ inline uint32_t forms_activation_offset(uint32_t step_in_block, uint32_t lane_place, uint32_t byte)
{
    return step_in_block * FORMS_INPUTS_PER_STEP + lane_place * FORMS_REGISTERS_PER_OPERAND_B
           + byte / FORMS_REGISTERS_PER_OPERAND_B + byte % FORMS_REGISTERS_PER_OPERAND_B * FORMS_HALF_INPUTS;
}

__host__ __device__ inline uint32_t forms_activation_word(uint32_t block, uint32_t lane_place, uint32_t step_in_block)
{
    return (block / FORMS_BLOCKS_PER_LOAD * FORMS_LANES_PER_LANE_GROUP + lane_place) * FORMS_STEPS_PER_LOAD
           + block % FORMS_BLOCKS_PER_LOAD * FORMS_STEPS_PER_BLOCK + step_in_block;
}

/* Any grid, any block: words 0 .. word_count - 1 of a stored array get the stored words of their places. */
__global__ void forms_fill_stored_kernel(uint32_t *words, uint32_t is_scale, uint64_t word_count)
{
    const uint64_t thread_total = (uint64_t)gridDim.x * blockDim.x;
    for (uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; index < word_count; index += thread_total) {
        words[index] = forms_stored_word(index, is_scale);
    }
}

/* Grid (vectors), any block: slot blockIdx.x of values gets vector first_vector + blockIdx.x of the step. */
__global__ void forms_fill_values_kernel(float *values, uint32_t size, uint32_t first_vector, uint32_t step,
                                         uint32_t corners)
{
    for (uint32_t column = threadIdx.x; column < size; column += blockDim.x) {
        values[(size_t)blockIdx.x * size + column] =
            forms_input_value(first_vector + blockIdx.x, column, step, corners);
    }
}

/* ---- the quantizer ---- */

/* An fp32 rounded to bfloat16, its upper half, to nearest with ties to even: the value the cast to bf16 keeps. */
__device__ __forceinline__ float forms_bf16_value(float value)
{
    const uint32_t bits = __float_as_uint(value);
    return __uint_as_float((bits + FORMS_BF16_ROUNDING_BIAS + ((bits >> FORMS_HALF_BITS) & 1u)) & FORMS_BF16_KEPT_MASK);
}

/* The E4M3 code nearest to value * inverse_scale, clamped to +-448, ties to the even code. */
__device__ __forceinline__ uint32_t forms_quantize_to_e4m3(float value, float inverse_scale)
{
    const float scaled = value * inverse_scale;                  /* by a power of two: exact */
    const float magnitude = fminf(fabsf(scaled), FORMS_E4M3_MAXIMUM);
    const uint32_t exponent_field =
        __float_as_uint(fmaxf(magnitude, FORMS_E4M3_SMALLEST_NORMAL)) >> FORMS_FP32_EXPONENT_SHIFT;
    const float steps_per_unit =
        __uint_as_float((FORMS_E4M3_STEPS_EXPONENT_BASE - exponent_field) << FORMS_FP32_EXPONENT_SHIFT);
    const uint32_t steps = (uint32_t)__float2int_rn(magnitude * steps_per_unit);        /* 0 .. 16, ties to even */
    return ((exponent_field << FORMS_E4M3_EXPONENT_SHIFT) + steps - FORMS_E4M3_CODE_BASE)
           | ((__float_as_uint(scaled) >> FORMS_E4M3_SIGN_FROM_FP32_SHIFT) & FORMS_E4M3_SIGN_BIT);
}

/* Four E4M3 bytes of one word as the two registers of operand B of a step, each byte the fp16 of its value times
 * 2^-8: bytes 0 and 2 are the halves of the first register, bytes 1 and 3 those of the second. */
__device__ __forceinline__ void forms_place_codes(uint32_t word, uint32_t *low_inputs, uint32_t *high_inputs)
{
    const uint32_t odd_bytes = word >> FORMS_BYTE_BITS;
    *low_inputs = ((word & FORMS_E4M3_PAIR_SIGNS) << FORMS_BYTE_BITS)
                  | ((word & FORMS_E4M3_PAIR_MAGNITUDES) << FORMS_E4M3_MAGNITUDE_SHIFT);
    *high_inputs = ((odd_bytes & FORMS_E4M3_PAIR_SIGNS) << FORMS_BYTE_BITS)
                   | ((odd_bytes & FORMS_E4M3_PAIR_MAGNITUDES) << FORMS_E4M3_MAGNITUDE_SHIFT);
}

/* A register of placed codes as the operand register of a form. carry is the power of two that takes a placed
 * value to the carried one; it multiplies exactly, and so does the conversion back, wherever the form holds the
 * value (the header says where). carried 0: the block keeps its placed words. */
template <int operand>
__device__ __forceinline__ uint32_t forms_operand_register(uint32_t placed, float carry, int carried)
{
    if (operand == FORMS_OPERAND_PLACED16 || !carried) {
        return placed;
    }
    const float low = __half2float(__ushort_as_half((unsigned short)(placed & FORMS_HALF_MASK))) * carry;
    const float high = __half2float(__ushort_as_half((unsigned short)(placed >> FORMS_HALF_BITS))) * carry;
    if (operand == FORMS_OPERAND_CARRIED_BF16) {                 /* a bfloat16 is the upper half of its fp32 */
        return (__float_as_uint(low) >> FORMS_HALF_BITS) | (__float_as_uint(high) & FORMS_BF16_KEPT_MASK);
    }
    return (uint32_t)__half_as_ushort(__float2half(low))
           | ((uint32_t)__half_as_ushort(__float2half(high)) << FORMS_HALF_BITS);
}

/* The reference's act_quant for blocks of 32: grid (vectors), a block of exactly size / 32 threads, one thread per
 * block of values. The largest magnitude of the block (at least 1e-4) times 1 / 448 in fp32, rounded up to a power
 * of two read from its fp32 fields, is the scale; a value over the scale, clamped to +-448, becomes the nearest
 * E4M3 code. Every form writes codes and exponents; all but CODES also the operand words; CARRIED16 the vector's
 * unfit word, which thread 0 gathers behind the block's one barrier. */
template <int operand>
__global__ void forms_quantize_kernel(struct forms_vectors vectors)
{
    __shared__ uint32_t block_unfit[FORMS_MOST_SCALE_BLOCKS];
    const uint32_t block = threadIdx.x, block_count = vectors.size / FORMS_VALUES_PER_BLOCK;
    const float *block_values = vectors.values + (size_t)blockIdx.x * vectors.size + block * FORMS_VALUES_PER_BLOCK;
    uint32_t *vector_codes = vectors.codes + (size_t)blockIdx.x * (vectors.size / FORMS_WORD_BYTES);
    uint32_t *vector_operands = vectors.operands + (size_t)blockIdx.x * (vectors.size / FORMS_HALVES_PER_REGISTER);
    float largest = 0.0f;
    for (uint32_t offset = 0; offset < FORMS_VALUES_PER_BLOCK; offset++) {
        largest = fmaxf(largest, fabsf(block_values[offset]));
    }
    const uint32_t ratio_bits = __float_as_uint(fmaxf(largest, FORMS_AMAX_FLOOR) * FORMS_E4M3_MAXIMUM_INVERSE);
    const uint32_t scale_exponent = (ratio_bits >> FORMS_FP32_EXPONENT_SHIFT)
                                    + ((ratio_bits & FORMS_FP32_MANTISSA_MASK) != 0u ? 1u : 0u);
    const float inverse_scale = __uint_as_float((FORMS_FP32_EXPONENT_BIAS + FORMS_E8M0_BIAS - scale_exponent)
                                                << FORMS_FP32_EXPONENT_SHIFT);
    /* placed values are the codes' times 2^-8; a field of 255 is infinity: beyond what CARRIED_BF16 holds */
    const uint32_t carry_field = scale_exponent + FORMS_CODES_EXPONENT
        + (operand == FORMS_OPERAND_CARRIED16 ? FORMS_CARRIED16_EXPONENT : FORMS_CARRIED_BF16_EXPONENT);
    const float carry = __uint_as_float((carry_field < FORMS_FP32_INFINITY_FIELD ? carry_field
                                                                                  : (uint32_t)FORMS_FP32_INFINITY_FIELD)
                                        << FORMS_FP32_EXPONENT_SHIFT);
    const int carried = operand != FORMS_OPERAND_CARRIED16 || scale_exponent <= FORMS_E8M0_BIAS;
    for (uint32_t word_in_block = 0; word_in_block < FORMS_WORDS_PER_BLOCK; word_in_block++) {
        const uint32_t lane_place = word_in_block / FORMS_STEPS_PER_BLOCK;
        const uint32_t step_in_block = word_in_block % FORMS_STEPS_PER_BLOCK;
        const uint32_t word_index = forms_activation_word(block, lane_place, step_in_block);
        uint32_t word = 0;
        for (uint32_t byte = 0; byte < FORMS_WORD_BYTES; byte++) {
            const float value = block_values[forms_activation_offset(step_in_block, lane_place, byte)];
            word |= forms_quantize_to_e4m3(value, inverse_scale) << (byte * FORMS_BYTE_BITS);
        }
        vector_codes[word_index] = word;
        if (operand != FORMS_OPERAND_CODES) {
            uint32_t low_inputs, high_inputs;
            forms_place_codes(word, &low_inputs, &high_inputs);
            vector_operands[word_index * FORMS_REGISTERS_PER_OPERAND_B] =
                forms_operand_register<operand>(low_inputs, carry, carried);
            vector_operands[word_index * FORMS_REGISTERS_PER_OPERAND_B + 1u] =
                forms_operand_register<operand>(high_inputs, carry, carried);
        }
    }
    vectors.exponents[(size_t)blockIdx.x * block_count + block] = (uint8_t)scale_exponent;
    if (operand == FORMS_OPERAND_CARRIED16) {
        block_unfit[block] = carried ? 0u : 1u;
        __syncthreads();
        if (block == 0u) {
            uint32_t unfit = 0;
            for (uint32_t other = 0; other < block_count; other++) {
                unfit |= block_unfit[other];
            }
            vectors.unfit[blockIdx.x] = unfit;
        }
    }
}

/* ---- the product ---- */

#define FORMS_MMA(kind)                                                                                             \
    "mma.sync.aligned.m16n8k16.row.col.f32." kind "." kind ".f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "          \
    "{%10,%11,%12,%13};\n"
#define FORMS_MMA_OPERANDS                                                                                          \
    "=f"(accumulator[0]), "=f"(accumulator[1]), "=f"(accumulator[2]), "=f"(accumulator[3])                          \
        : "r"(weights[0]), "r"(weights[1]), "r"(weights[2]), "r"(weights[3]), "r"(low_inputs), "r"(high_inputs),    \
          "f"(accumulator[0]), "f"(accumulator[1]), "f"(accumulator[2]), "f"(accumulator[3])

/* accumulator = A B + accumulator for the lane's part of each. Lane 4 g + t of a warp holds, of the 16-by-16
 * operand A (the weights), outputs g and g + 8 at inputs 2t, 2t + 1, 2t + 8, 2t + 9; of the 16-by-8 operand B column
 * g (one vector) at the same four inputs; of the result, outputs g and g + 8 for columns 2t and 2t + 1. */
template <int operand>
__device__ __forceinline__ void forms_tensor_core(float accumulator[FORMS_ELEMENTS_PER_LANE],
                                                  const uint32_t weights[FORMS_REGISTERS_PER_OPERAND_A],
                                                  uint32_t low_inputs, uint32_t high_inputs)
{
    if (operand == FORMS_OPERAND_CARRIED_BF16) {
        asm volatile(FORMS_MMA("bf16") : FORMS_MMA_OPERANDS);
    } else {
        asm volatile(FORMS_MMA("f16") : FORMS_MMA_OPERANDS);
    }
}

/* The eight E2M1 codes of one stored word as the four registers of operand A: the code at bit 4 r + 16 h is half h
 * of register r; its magnitude bits go where the operand's kind has them, its sign to bit 15. */
template <int operand>
__device__ __forceinline__ void forms_place_weights(uint32_t word, uint32_t weights[FORMS_REGISTERS_PER_OPERAND_A])
{
    const uint32_t magnitude_bit =
        operand == FORMS_OPERAND_CARRIED_BF16 ? FORMS_BF16_MAGNITUDE_BIT : FORMS_FP16_MAGNITUDE_BIT;
    for (int operand_register = 0; operand_register < FORMS_REGISTERS_PER_OPERAND_A; operand_register++) {
        const uint32_t codes = word >> (operand_register * FORMS_NIBBLE_BITS);
        weights[operand_register] = ((codes << magnitude_bit) & ((uint32_t)FORMS_E2M1_MAGNITUDES << magnitude_bit))
                                    | ((codes << FORMS_E2M1_SIGN_SHIFT) & FORMS_PAIR_SIGN_MASK);
    }
}

struct forms_lane {                              /* what one lane of one slice reads: set from a validated step */
    const uint4 *weights;                        /* its 16 bytes of the first load of its tile */
    const uint32_t *weight_scales;               /* the word of its lane group at that load */
    const uint4 *fed;                            /* of the vector its column carries, at the vector's first load:
                                                    its lane place's codes, or its two loads of operand words */
    const uint16_t *exponents[FORMS_COLUMNS_PER_LANE];      /* of the vectors of its two result columns */
    uint32_t falls_back;                         /* CARRIED16: one of those two vectors is unfit */
};
struct forms_load {                              /* what a lane holds of one load of 64 inputs */
    uint4 packed, fed[FORMS_FED_LOADS];
    uint32_t weight_codes;                       /* byte 2 q + r: block q of the load, output half r */
    uint32_t column_codes[FORMS_COLUMNS_PER_LANE];           /* byte q: block q; 0 where the form reads none */
};

template <int operand, int columns>
__device__ __forceinline__ struct forms_load forms_fetch(const struct forms_lane *lane, uint32_t load)
{
    const int has_exponents = operand <= FORMS_OPERAND_PLACED16 || lane->falls_back != 0u;
    const size_t fed_at =
        (size_t)load * FORMS_LANES_PER_LANE_GROUP * (operand == FORMS_OPERAND_CODES ? 1 : FORMS_FED_LOADS);
    struct forms_load fetched;
    fetched.packed = lane->weights[(size_t)load * FORMS_LANES_PER_WARP];
    fetched.weight_codes = lane->weight_scales[(size_t)load * FORMS_LANE_GROUPS];
    fetched.fed[0] = lane->fed[fed_at];
    fetched.fed[1] = operand == FORMS_OPERAND_CODES ? fetched.fed[0] : lane->fed[fed_at + 1u];
    for (int column = 0; column < FORMS_COLUMNS_PER_LANE; column++) {
        fetched.column_codes[column] = column < columns && has_exponents ? lane->exponents[column][load] : 0u;
    }
    return fetched;
}

/* chain = fma(sum, 2^e, chain) for the lane's sums of one block of 32 inputs. weight_codes: byte r is the scale
 * code of output half r. column_fields: per result column, e's exponent field less that code. */
template <int columns>
__device__ __forceinline__ void forms_scale_sums(float chains[FORMS_ELEMENTS_PER_LANE],
                                                 const float accumulator[FORMS_ELEMENTS_PER_LANE],
                                                 uint32_t weight_codes,
                                                 const uint32_t column_fields[FORMS_COLUMNS_PER_LANE])
{
    for (int element = 0; element < FORMS_ELEMENTS_PER_LANE; element += FORMS_COLUMNS_PER_LANE / columns) {
        const uint32_t weight_code =
            (weight_codes >> (element / FORMS_COLUMNS_PER_LANE * FORMS_BYTE_BITS)) & FORMS_BYTE_MASK;
        const float scale = __uint_as_float((weight_code + column_fields[element % FORMS_COLUMNS_PER_LANE])
                                            << FORMS_FP32_EXPONENT_SHIFT);
        chains[element] = fmaf(accumulator[element], scale, chains[element]);
    }
}

/* One slice of one tile for one lane: the chains of its four results over loads first_load .. end_load - 1 (at
 * least one), each from +0. The chains are the lane's own until the slice ends: chains_out is written once. */
template <int operand, int ahead, int columns>
__device__ __forceinline__ void forms_slice_chains(const struct forms_lane *lane, uint32_t first_load,
                                                   uint32_t end_load, float chains_out[FORMS_ELEMENTS_PER_LANE])
{
    const uint32_t carried_field =
        operand == FORMS_OPERAND_CARRIED_BF16 ? FORMS_CARRIED_BF16_FIELD : FORMS_CARRIED16_FIELD;
    float chains[FORMS_ELEMENTS_PER_LANE] = {0.0f, 0.0f, 0.0f, 0.0f};
    struct forms_load held[ahead + 1] = {};      /* held[k]: the load k loads ahead of the one in use */
    for (int stage = 0; stage < ahead; stage++) {
        const uint32_t load = first_load + (uint32_t)stage;
        held[stage] = forms_fetch<operand, columns>(lane, load < end_load ? load : end_load - 1u);
    }
    for (uint32_t load = first_load; load < end_load; load++) {
        if (ahead == 0 || load + (uint32_t)ahead < end_load) {   /* never beyond the slice's last load */
            held[ahead] = forms_fetch<operand, columns>(lane, load + (uint32_t)ahead);
        }
        const struct forms_load now = held[0];
        for (int stage = 0; stage < ahead; stage++) {
            held[stage] = held[stage + 1];
        }
        const uint32_t weight_words[FORMS_STEPS_PER_LOAD] = {now.packed.x, now.packed.y, now.packed.z, now.packed.w};
        const uint32_t fed_words[FORMS_FED_WORDS] = {now.fed[0].x, now.fed[0].y, now.fed[0].z, now.fed[0].w,
                                                     now.fed[1].x, now.fed[1].y, now.fed[1].z, now.fed[1].w};
        for (uint32_t block = 0; block < FORMS_BLOCKS_PER_LOAD; block++) {
            const uint32_t weight_codes = now.weight_codes >> (block * FORMS_HALF_BITS);
            float accumulator[FORMS_ELEMENTS_PER_LANE] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (uint32_t step = block * FORMS_STEPS_PER_BLOCK; step < (block + 1u) * FORMS_STEPS_PER_BLOCK; step++) {
                uint32_t weights[FORMS_REGISTERS_PER_OPERAND_A], low_inputs, high_inputs;
                forms_place_weights<operand>(weight_words[step], weights);
                if (operand == FORMS_OPERAND_CODES) {
                    forms_place_codes(fed_words[step], &low_inputs, &high_inputs);
                } else {
                    low_inputs = fed_words[step * FORMS_REGISTERS_PER_OPERAND_B];
                    high_inputs = fed_words[step * FORMS_REGISTERS_PER_OPERAND_B + 1u];
                }
                forms_tensor_core<operand>(accumulator, weights, low_inputs, high_inputs);
            }
            if (operand <= FORMS_OPERAND_PLACED16 || (operand == FORMS_OPERAND_CARRIED16 && lane->falls_back != 0u)) {
                uint32_t column_fields[FORMS_COLUMNS_PER_LANE] = {0u, 0u};
                for (int column = 0; column < columns; column++) {       /* per column: what its block holds */
                    const uint32_t code = (now.column_codes[column] >> (block * FORMS_BYTE_BITS)) & FORMS_BYTE_MASK;
                    column_fields[column] = operand == FORMS_OPERAND_CARRIED16 && code <= FORMS_E8M0_BIAS
                                                ? (uint32_t)FORMS_CARRIED16_FIELD : code - FORMS_PLACED_BIAS;
                }
                forms_scale_sums<columns>(chains, accumulator, weight_codes, column_fields);
            } else {                             /* the weights' scale code alone: no exponent was read */
                const uint32_t column_fields[FORMS_COLUMNS_PER_LANE] = {carried_field, carried_field};
                forms_scale_sums<columns>(chains, accumulator, weight_codes, column_fields);
            }
        }
    }
    for (int element = 0; element < FORMS_ELEMENTS_PER_LANE; element++) {
        chains_out[element] = chains[element];
    }
}

/* One result of one matrix: the chains its slices left as [warp][lane][element], warps first_warp .. , added from
 * +0 in ascending slice order; by_rows 1 starts at the second slice where the job has several rows. */
template <int slices, int by_rows>
__device__ __forceinline__ float forms_sum_of_slices(const float *chains, int first_warp, uint32_t lane_index,
                                                     uint32_t element, uint32_t row_count)
{
    const int first_slice = by_rows != 0 && row_count > 1u ? 1 : 0;
    float sum = 0.0f;
    for (int position = 0; position < slices; position++) {
        const int warp = first_warp + (first_slice + position) % slices;
        sum += chains[((size_t)warp * FORMS_LANES_PER_WARP + lane_index) * FORMS_ELEMENTS_PER_LANE + element];
    }
    return sum;
}

/* One block of a product: `matrices` matrices of one expert for one tile of 16 outputs, `slices` warps a matrix.
 * Two matrices: gate (the first warps) and up from the rows' quantized inputs, each rounded to bf16 as the
 * reference's product returns it; h = silu(min(gate, L)) * clamp(up, -L, L) * route weight, cast to bf16, goes to
 * hidden.values, [pair][inner size]. One matrix: down from the pairs' quantized hidden vectors; the fp32 sum goes to
 * pair_output, [pair][hidden size]. The rows of a job are the instruction's columns: column c carries the vector
 * of pair_index[c], and a column from row_count on is stored nowhere. Not persistent: grid (tiles, jobs).
 * Persistent: grid (blocks, 1), at most tiles * jobs blocks. block_chains: the block's shared memory. */
template <int operand, int ahead, int persistent, int slices, int columns, int matrices, int by_rows>
__device__ __forceinline__ void forms_product(struct forms_step step, float *block_chains)
{
    const struct forms_vectors vectors = matrices == FORMS_GATE_UP_MATRICES ? step.input : step.hidden;
    const uint32_t output_size = matrices == FORMS_GATE_UP_MATRICES ? step.hidden.size : step.input.size;
    const uint32_t tiles = output_size / FORMS_OUTPUTS_PER_TILE, loads = vectors.size / FORMS_INPUTS_PER_LOAD;
    const uint32_t warp = threadIdx.x / FORMS_LANES_PER_WARP, lane_index = threadIdx.x % FORMS_LANES_PER_WARP;
    const uint32_t lane_group = lane_index / FORMS_LANES_PER_LANE_GROUP;
    const uint32_t lane_place = lane_index % FORMS_LANES_PER_LANE_GROUP;
    const uint32_t slice = warp % (uint32_t)slices, even_column = lane_place * FORMS_COLUMNS_PER_LANE;
    uint32_t unit = persistent ? blockIdx.x : 0u, turn = 0;
    do {
        const struct forms_job *job = step.jobs + (persistent ? unit / tiles : blockIdx.y);
        const uint32_t tile = persistent ? unit % tiles : blockIdx.x;
        const uint32_t matrix_kind =
            matrices == FORMS_DOWN_MATRICES ? (uint32_t)FORMS_DOWN_MATRIX : warp / (uint32_t)slices;
        const size_t matrix = (size_t)job->expert_index * FORMS_MATRIX_KINDS + matrix_kind;
        const size_t tile_load = matrix * FORMS_TILE_LOADS_PER_MATRIX + (size_t)tile * loads;
        const size_t fed_vector = job->pair_index[lane_group] / vectors.pairs_per_vector;
        const size_t even_vector = job->pair_index[even_column] / vectors.pairs_per_vector;
        const size_t odd_vector = job->pair_index[even_column + 1u] / vectors.pairs_per_vector;
        const size_t fed_place = fed_vector * loads * FORMS_LANES_PER_LANE_GROUP + lane_place;
        float *chains = block_chains + (size_t)turn * FORMS_BLOCK_THREADS(matrices, slices) * FORMS_ELEMENTS_PER_LANE;
        struct forms_lane lane;
        lane.weights = step.packed + tile_load * FORMS_LANES_PER_WARP + lane_index;
        lane.weight_scales = step.scales + tile_load * FORMS_LANE_GROUPS + lane_group;
        lane.fed = operand == FORMS_OPERAND_CODES ? (const uint4 *)vectors.codes + fed_place
                                                  : (const uint4 *)vectors.operands + fed_place * FORMS_FED_LOADS;
        lane.exponents[0] = (const uint16_t *)vectors.exponents + even_vector * loads;
        lane.exponents[1] = (const uint16_t *)vectors.exponents + odd_vector * loads;
        lane.falls_back = 0u;
        if (operand == FORMS_OPERAND_CARRIED16) {
            lane.falls_back = vectors.unfit[even_vector];
            lane.falls_back |= columns == FORMS_COLUMNS_PER_LANE ? vectors.unfit[odd_vector] : 0u;
        }
        forms_slice_chains<operand, ahead, columns>(&lane, slice * loads / (uint32_t)slices,
                                                    (slice + 1u) * loads / (uint32_t)slices,
                                                    chains + (size_t)threadIdx.x * FORMS_ELEMENTS_PER_LANE);
        if (matrices * slices > 1) {
            __syncthreads();
        }
        for (uint32_t element = 0; warp == 0u && element < FORMS_ELEMENTS_PER_LANE; element++) {
            const uint32_t column = even_column + element % FORMS_COLUMNS_PER_LANE;
            const size_t output = (size_t)tile * FORMS_OUTPUTS_PER_TILE + lane_group
                                  + element / FORMS_COLUMNS_PER_LANE * FORMS_HALF_TILE;
            if (column >= job->row_count) {
                continue;
            }
            const size_t stored_at = (size_t)job->pair_index[column] * output_size + output;
            const float sum = forms_sum_of_slices<slices, by_rows>(chains, 0, lane_index, element, job->row_count);
            if (matrices == FORMS_DOWN_MATRICES) {
                step.pair_output[stored_at] = sum;
            } else {
                const float gate = fminf(forms_bf16_value(sum), FORMS_SWIGLU_LIMIT);
                const float up = forms_bf16_value(
                    forms_sum_of_slices<slices, by_rows>(chains, slices, lane_index, element, job->row_count));
                const float silu = gate / (1.0f + expf(-gate));
                const float bounded_up = fminf(fmaxf(up, -FORMS_SWIGLU_LIMIT), FORMS_SWIGLU_LIMIT);
                step.hidden.values[stored_at] = forms_bf16_value(silu * bounded_up * job->route_weight[column]);
            }
        }
        unit += gridDim.x;
        turn ^= 1u;
    } while (persistent && unit < step.job_count * tiles);
}

/* A row's six experts, each rounded to bf16 as the reference's product returns it, added in slot order, which is
 * ascending expert order, the order of the reference's loop: grid (rows), any block. */
__global__ void forms_row_sum_kernel(const float *pair_output, float *row_output)
{
    for (uint32_t column = threadIdx.x; column < FORMS_HIDDEN_SIZE; column += blockDim.x) {
        float sum = 0.0f;
        for (uint32_t slot = 0; slot < FORMS_EXPERTS_PER_ROW; slot++) {
            sum += forms_bf16_value(
                pair_output[((size_t)blockIdx.x * FORMS_EXPERTS_PER_ROW + slot) * FORMS_HIDDEN_SIZE + column]);
        }
        row_output[(size_t)blockIdx.x * FORMS_HIDDEN_SIZE + column] = sum;
    }
}

/* ---- the kernels of one variant: forms_gate_up_<id> and forms_down_<id> ----
 * Blocks of FORMS_BLOCK_THREADS threads. bound is OPEN (no launch bound: the baseline has none, as no kernel of
 * the engine) or STATED: `blocks` blocks of gate with up to an SM, and twice as many of down's half-size block. */
#define FORMS_BOUND_OPEN(threads, blocks)
#define FORMS_BOUND_STATED(threads, blocks) __launch_bounds__(threads, blocks)
#define FORMS_SHARED_FLOATS(matrices, slices, persistent)                                                           \
    (((persistent) != 0 ? FORMS_CHAIN_TURNS : 1) * FORMS_BLOCK_THREADS(matrices, slices) * FORMS_ELEMENTS_PER_LANE)
#define FORMS_PRODUCT_KERNEL(name, matrices, operand, ahead, bound, blocks, persistent, slices, columns, by_rows)   \
    __global__ void FORMS_BOUND_##bound(FORMS_BLOCK_THREADS(matrices, slices), blocks) name(struct forms_step step) \
    {                                                                                                               \
        __shared__ float block_chains[FORMS_SHARED_FLOATS(matrices, slices, persistent)];                           \
        forms_product<operand, ahead, persistent, slices, columns, matrices, by_rows>(step, block_chains);          \
    }
#define FORMS_PRODUCT_KERNELS(id, operand, ahead, bound, blocks, persistent, slices, columns, by_rows)              \
    FORMS_PRODUCT_KERNEL(forms_gate_up_##id, FORMS_GATE_UP_MATRICES, operand, ahead, bound, blocks, persistent,     \
                         slices, columns, by_rows)                                                                  \
    FORMS_PRODUCT_KERNEL(forms_down_##id, FORMS_DOWN_MATRICES, operand, ahead, bound,                               \
                         FORMS_GATE_UP_MATRICES * (blocks), persistent, slices, columns, by_rows)

/* ---- host side: the formats decoded from their fields and the references in double precision; they share with
 * the kernels above the stored words (forms_stored_word) and the constants of the formats ---- */
static const double forms_e2m1_values[] = {0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
                                           -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0};

/* E4M3 from its fields: (8 + mantissa) * 2^(exponent - 10), or mantissa * 2^-9 where the exponent is 0. */
inline double forms_host_e4m3(uint32_t code)
{
    const uint32_t exponent = (code >> FORMS_E4M3_EXPONENT_SHIFT) & FORMS_E4M3_EXPONENT_MASK;
    const uint32_t mantissa = code & FORMS_E4M3_MANTISSA_MASK;
    const uint32_t scaled = exponent == 0 ? mantissa << 1 : (FORMS_E4M3_IMPLICIT_ONE + mantissa) << exponent;
    return ((code & FORMS_E4M3_SIGN_BIT) != 0u ? -1.0 : 1.0) * scaled / FORMS_E4M3_VALUE_DIVISOR;
}

/* What half of an operand register holds, from its fields: a binary16, or for CARRIED_BF16 the upper half of an
 * fp32. A zero keeps its sign. */
inline double forms_host_half(uint32_t operand, uint32_t half)
{
    const int field = (int)(half >> FORMS_FP16_MANTISSA_BITS) & FORMS_FP16_EXPONENT_MASK;
    const double mantissa = (double)(half & FORMS_FP16_MANTISSA_MASK), implicit_one = FORMS_FP16_IMPLICIT_ONE;
    const double magnitude = field == 0 ? ldexp(mantissa, -FORMS_FP16_SUBNORMAL_EXPONENT)
                                        : ldexp(implicit_one + mantissa, field - FORMS_FP16_EXPONENT_BIAS);
    const uint32_t wide_bits = half << FORMS_HALF_BITS;
    float wide = 0.0f;
    memcpy(&wide, &wide_bits, sizeof(wide));
    if (operand == FORMS_OPERAND_CARRIED_BF16) {
        return (double)wide;
    }
    return (half & FORMS_FP16_SIGN) != 0u ? -magnitude : magnitude;
}

/* A double rounded to bf16's eight significant bits, to nearest with ties to even (rint in the default mode). */
inline double forms_host_bf16(double value)
{
    int binary_exponent = 0;
    const double mantissa = frexp(value, &binary_exponent);
    return ldexp(rint(ldexp(mantissa, FORMS_BF16_SIGNIFICANT_BITS)), binary_exponent - FORMS_BF16_SIGNIFICANT_BITS);
}

/* The reference's act_quant for one vector, from its definition: per block of 32 the scale is 2^ceil(log2(ratio)),
 * ratio being the largest magnitude (at least 1e-4) times 1 / 448 in fp32; each value over the scale, clamped to
 * +-448, becomes the nearest E4M3 value, ties to the even code. quantized: the codes in the order of the values,
 * then the blocks' E8M0 codes. least_largest: the reference's FORMS_AMAX_FLOOR. */
inline void forms_host_quantize(const float *values, uint32_t size, float least_largest, uint8_t *quantized)
{
    for (uint32_t block = 0; block < size / FORMS_VALUES_PER_BLOCK; block++) {
        const float *block_values = values + block * FORMS_VALUES_PER_BLOCK;
        float largest = least_largest;
        for (uint32_t offset = 0; offset < FORMS_VALUES_PER_BLOCK; offset++) {
            largest = fmaxf(largest, fabsf(block_values[offset]));
        }
        const float ratio = largest * FORMS_E4M3_MAXIMUM_INVERSE;
        const int log2_ceiling = (int)ceil(log2((double)ratio));     /* no float's log2 is within 1e-8 of a whole */
        quantized[size + block] = (uint8_t)(log2_ceiling + FORMS_E8M0_BIAS);
        for (uint32_t offset = 0; offset < FORMS_VALUES_PER_BLOCK; offset++) {
            const double scaled = ldexp((double)block_values[offset], -log2_ceiling);
            const double magnitude = fmin(fabs(scaled), (double)FORMS_E4M3_MAXIMUM);
            uint32_t nearest = 0;
            for (uint32_t code = 1; code < FORMS_E4M3_NOT_A_NUMBER; code++) {
                const double distance = fabs(forms_host_e4m3(code) - magnitude);
                const double least = fabs(forms_host_e4m3(nearest) - magnitude);
                nearest = distance < least || (distance == least && (code & 1u) == 0u) ? code : nearest;
            }
            quantized[block * FORMS_VALUES_PER_BLOCK + offset] =
                (uint8_t)(nearest | (signbit(scaled) ? (uint32_t)FORMS_E4M3_SIGN_BIT : 0u));
        }
    }
}

/* One output of one matrix of one expert in double, from the stored words: per block of 32 inputs the sum of weight
 * times activation, times both scales. It says where the kernels find a weight and a scale: packed words are
 * [expert][matrix][tile of 16 outputs][load of 64 inputs][lane][step of the load], eight nibbles each, the nibble
 * of register r and half h of operand A at bit 4 r + 16 h; scale words are [expert][matrix][tile][load][lane group
 * g]: byte 2 q + r is the E8M0 code of block q of the load for output g + 8 r. */
inline double forms_host_product(uint32_t matrix_kind, uint32_t expert, uint32_t output_row, const uint8_t *quantized)
{
    const uint32_t input_size = matrix_kind == FORMS_DOWN_MATRIX ? FORMS_INNER_SIZE : FORMS_HIDDEN_SIZE;
    const uint32_t lane_group = output_row % FORMS_HALF_TILE, tile = output_row / FORMS_OUTPUTS_PER_TILE;
    const uint32_t output_half = output_row % FORMS_OUTPUTS_PER_TILE / FORMS_HALF_TILE;
    const uint64_t matrix = (uint64_t)expert * FORMS_MATRIX_KINDS + matrix_kind;
    const uint64_t first_tile_load =
        matrix * FORMS_TILE_LOADS_PER_MATRIX + (uint64_t)tile * (input_size / FORMS_INPUTS_PER_LOAD);
    double sum = 0.0;
    for (uint32_t block = 0; block < input_size / FORMS_VALUES_PER_BLOCK; block++) {
        const uint64_t tile_load = first_tile_load + block / FORMS_BLOCKS_PER_LOAD;
        const uint32_t scale_byte = block % FORMS_BLOCKS_PER_LOAD * FORMS_OUTPUTS_PER_LANE + output_half;
        const uint32_t scale_word = forms_stored_word(tile_load * FORMS_LANE_GROUPS + lane_group, 1);
        double block_sum = 0.0;
        for (uint32_t offset = 0; offset < FORMS_VALUES_PER_BLOCK; offset++) {
            const uint32_t input = offset % FORMS_INPUTS_PER_STEP, upper_inputs = input / FORMS_HALF_INPUTS;
            const uint32_t step =
                block % FORMS_BLOCKS_PER_LOAD * FORMS_STEPS_PER_BLOCK + offset / FORMS_INPUTS_PER_STEP;
            const uint64_t lane = tile_load * FORMS_LANES_PER_WARP + lane_group * FORMS_LANES_PER_LANE_GROUP
                                  + input % FORMS_HALF_INPUTS / FORMS_HALVES_PER_REGISTER;
            const uint32_t nibble = output_half + upper_inputs * FORMS_OUTPUTS_PER_LANE
                                    + input % FORMS_HALVES_PER_REGISTER * FORMS_REGISTERS_PER_OPERAND_A;
            const uint32_t word = forms_stored_word(lane * FORMS_STEPS_PER_LOAD + step, 0);
            block_sum += forms_e2m1_values[(word >> (nibble * FORMS_NIBBLE_BITS)) & FORMS_NIBBLE_MASK]
                         * forms_host_e4m3(quantized[block * FORMS_VALUES_PER_BLOCK + offset]);
        }
        sum += ldexp(block_sum, (int)((scale_word >> (scale_byte * FORMS_BYTE_BITS)) & FORMS_BYTE_MASK)
                                    + (int)quantized[input_size + block] - FORMS_E8M0_BIAS - FORMS_E8M0_BIAS);
    }
    return sum;
}

/* The hidden step of one inner output in double: gate and up rounded to bf16 as the reference's product returns
 * them, then silu(min(gate, L)) * clamp(up, -L, L) * route weight, cast to bf16. */
inline double forms_host_hidden(uint32_t expert, uint32_t inner, const uint8_t *quantized, double route_weight)
{
    const double limit = FORMS_SWIGLU_LIMIT;
    const double gate = fmin(forms_host_bf16(forms_host_product(FORMS_GATE_MATRIX, expert, inner, quantized)), limit);
    const double up = forms_host_bf16(forms_host_product(FORMS_UP_MATRIX, expert, inner, quantized));
    return forms_host_bf16(gate / (1.0 + exp(-gate)) * fmin(fmax(up, -limit), limit) * route_weight);
}

#endif /* NATIVE_EXPERT_FORMS_KERNELS_CUH */
