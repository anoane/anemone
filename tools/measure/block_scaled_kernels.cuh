/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * block_scaled_kernels.cuh: the routed experts of one layer in the checkpoint's own format (E2M1 weights, E4M3
 * activations, a power of two per 32 inputs on each side) as five launches a step: the rows' inputs quantized, gate
 * with up and the hidden step, the hidden vectors quantized, down, a row's six experts added. Round 1, unit 3b of
 * the speed campaign (research/speed/ROUND_1.md): the product through the instruction of sm_120a that is the
 * format's own. ONE product body, scaled_product, whose choices are compile-time:
 *   product        F16: today's kernel: per block of 32 inputs two m16n8k16 on codes placed as fp16 (a weight's
 *                  value times 2^-14, an activation's times 2^-8), then chain = fma(sum, 2^e, chain).
 *                  MIXED: one m16n8k32 of kind f8f6f4 on the stored bytes, the same multiply-add outside.
 *                  FRESH: the block-scaled m16n8k32, both scales inside, on a zero accumulator, then chain += sum:
 *                  today's rounding points. CHAINED: the same with the chain as its accumulator: no addition
 *                  outside, and the instruction's own closing of an inexact sum, which cuts toward zero.
 *   column_groups  1: the rows of a job are the eight columns of one instruction. 8: a job has up to 64 rows; a
 *                  lane makes the weights' operand once a block for its eight instructions.
 *   matrices       2: gate (the first warps) with up, and the hidden step. 1: one matrix, its fp32 sums stored.
 *   fault          0 in every timed kernel; the two others are what a check is shown failing with.
 * and one quantizer whose two code steps are software or, on sm_120a, the card's conversions.
 * Operands of the byte forms, held by the layout and the selector map of measurements/i03a_mma_unit/unit_card_1.log.
 * Of a stored word of eight weight codes the even nibbles are output g and the odd ones output g + 8 at inputs 2t,
 * 2t + 8, 2t + 1, 2t + 9 of its step: the inputs of bytes 0 to 3 of the activation word of the same step. So
 * (word << 2) & 0x3C3C3C3C and (word >> 2) & 0x3C3C3C3C of a block's two words are operand A (a code in the central
 * four bits of its byte, the spare bits zero: the form reads them) and the two activation words are operand B as
 * stored. The 32 inputs meet in another order than today's, the same on both sides: a sum has no order to show
 * (same log). Scales: outputs g and g + 8 take byte b of the scale words of lane places 0 and 1 of lane group g,
 * column g byte b of lane place 0's: the weights' stored word (byte 2 q + r: block q, output g + 8 r), a byte down
 * in odd lanes, with b = 2 q; the two exponent bytes of the fed vector's load with b = q. A code c is 2^(c - 127).
 * The order of additions is the engine's (engine/include/anemone/linear_records.h): a block of 32 inputs a step in
 * ascending order from +0 within a slice, the SCALED_SLICES slices in ascending order from +0. A kernel checks
 * nothing and uses no atomic operation: block_scaled_probe.cu validates on the host the grid and block stated at
 * each kernel, every job's expert, its rows (1 .. 8 * column_groups) and each of its 64 pair indices against the
 * launch's arrays, whose sizes are whole loads and tiles of a stored matrix. A scale code is taken to keep every
 * power of two a normal fp32 (F16, MIXED) or to be under 255, which is no number (FRESH, CHAINED). At the end the
 * synthetic experts. Measurement tool: not part of the engine.
 */
#ifndef BLOCK_SCALED_KERNELS_CUH
#define BLOCK_SCALED_KERNELS_CUH
#include <cuda_runtime.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
enum {
    SCALED_HIDDEN_SIZE = 5120, SCALED_INNER_SIZE = 2304, SCALED_EXPERTS_PER_ROW = 6,
    SCALED_GATE_MATRIX = 0, SCALED_UP_MATRIX = 1, SCALED_DOWN_MATRIX = 2, SCALED_MATRIX_KINDS = 3,
    SCALED_ONE_MATRIX = 1, SCALED_GATE_UP_MATRICES = 2,           /* the matrices of a block of a product */
    SCALED_SLICES = 4,                           /* warps that share the input width of a matrix of a tile */
    SCALED_NARROW = 1, SCALED_WIDE = 8,          /* column groups of a job: eight columns, one instruction, each */
    SCALED_COLUMNS_PER_GROUP = 8, SCALED_MOST_COLUMNS = SCALED_WIDE * SCALED_COLUMNS_PER_GROUP,
    SCALED_LANES_PER_WARP = 32, SCALED_LANES_PER_LANE_GROUP = 4, SCALED_LANE_GROUPS = 8,
    SCALED_OUTPUTS_PER_TILE = 16, SCALED_HALF_TILE = 8,           /* the instruction's M; a lane's two outputs */
    SCALED_VALUES_PER_BLOCK = 32,                /* inputs that share one scale: of the weights, of a vector */
    SCALED_INPUTS_PER_STEP = 16, SCALED_HALF_INPUTS = 8,          /* a step: one stored word a lane, one m16n8k16 */
    SCALED_STEPS_PER_BLOCK = 2, SCALED_BLOCKS_PER_LOAD = 2, SCALED_STEPS_PER_LOAD = 4, SCALED_INPUTS_PER_LOAD = 64,
    SCALED_ELEMENTS_PER_LANE = 4, SCALED_COLUMNS_PER_LANE = 2,    /* a lane's results: two outputs by two columns */
    SCALED_REGISTERS_PER_OPERAND_A = 4, SCALED_REGISTERS_PER_OPERAND_B = 2, SCALED_CODES_PER_CONVERSION = 2,
    /* a lane's results: output g or g + 8 by column 2t or 2t + 1; its registers of A: output by half of the inputs */
    SCALED_LOW_EVEN = 0, SCALED_LOW_ODD, SCALED_HIGH_EVEN, SCALED_HIGH_ODD,
    SCALED_LOW_FIRST = 0, SCALED_HIGH_FIRST, SCALED_LOW_SECOND, SCALED_HIGH_SECOND,
    SCALED_ELEMENT_THREADS = 256,                /* the block of a kernel whose threads take one value each */
    /* a stored matrix: per (tile of 16 outputs, load of 64 inputs) 32 lanes of four words, and 8 scale words */
    SCALED_TILE_LOADS_PER_MATRIX =
        SCALED_HIDDEN_SIZE * SCALED_INNER_SIZE / (SCALED_OUTPUTS_PER_TILE * SCALED_INPUTS_PER_LOAD),
    SCALED_PACKED_WORDS_PER_MATRIX = SCALED_TILE_LOADS_PER_MATRIX * SCALED_LANES_PER_WARP * SCALED_STEPS_PER_LOAD,
    SCALED_SCALE_WORDS_PER_MATRIX = SCALED_TILE_LOADS_PER_MATRIX * SCALED_LANE_GROUPS,
    SCALED_BYTE_BITS = 8, SCALED_BYTE_MASK = 0xFF, SCALED_HALF_BITS = 16, SCALED_WORD_BYTES = 4, SCALED_NIBBLE_BITS = 4,
    /* E2M1: sign, two exponent bits, one mantissa bit. As an fp16 its three magnitude bits lie at bits 9 .. 11: the
     * value times 2^-14 (0.5 a subnormal). In a byte of kind f8f6f4 the whole code lies at bits 2 .. 5. */
    SCALED_FP16_MAGNITUDE_BIT = 9, SCALED_E2M1_SIGN_SHIFT = 12, SCALED_FP16_WEIGHT_EXPONENT = 14,
    SCALED_E2M1_BYTE_SHIFT = 2,
    /* E4M3: sign, four exponent bits (bias 7), three mantissa bits; the largest value is 448. As an fp16 its seven
     * magnitude bits lie at bits 7 .. 13: the value times 2^-8, subnormal codes included. */
    SCALED_E4M3_MAGNITUDE_SHIFT = 7, SCALED_CODES_EXPONENT = 8, SCALED_E4M3_SIGN_BIT = 0x80,
    SCALED_E4M3_EXPONENT_SHIFT = 3, SCALED_E4M3_EXPONENT_BIAS = 7, SCALED_E4M3_IMPLICIT_ONE = 8,
    SCALED_E4M3_SIGN_FROM_FP32_SHIFT = 24,
    SCALED_E8M0_BIAS = 127,                      /* a scale is 2^(code - 127) */
    SCALED_FP32_EXPONENT_SHIFT = 23, SCALED_FP32_EXPONENT_BIAS = 127, SCALED_FP32_MANTISSA_MASK = 0x007FFFFF,
    /* the exponent field of a block's power of two is the sum of its two scale codes less this: placed codes; bytes */
    SCALED_PLACED_BIAS = SCALED_E8M0_BIAS - SCALED_FP16_WEIGHT_EXPONENT - SCALED_CODES_EXPONENT,
    SCALED_BYTES_BIAS = SCALED_E8M0_BIAS,
    /* quantizing a value of fp32 exponent field f: codes are 2^(f - 130) apart, the code is 8 (f - 120) + steps - 8 */
    SCALED_E4M3_STEPS_EXPONENT_BASE = 2 * SCALED_FP32_EXPONENT_BIAS + SCALED_E4M3_EXPONENT_SHIFT,
    SCALED_E4M3_CODE_BASE = ((SCALED_FP32_EXPONENT_BIAS - SCALED_E4M3_EXPONENT_BIAS) << SCALED_E4M3_EXPONENT_SHIFT)
                            + SCALED_E4M3_IMPLICIT_ONE,
    SCALED_PRODUCT_F16 = 0, SCALED_PRODUCT_MIXED, SCALED_PRODUCT_FRESH, SCALED_PRODUCT_CHAINED,
    SCALED_QUANTIZER_SOFTWARE = 0, SCALED_QUANTIZER_CONVERSION,
    /* faults: the slices of a job of several rows added from the second on; the scale lanes of g and g + 8 exchanged */
    SCALED_SOUND = 0, SCALED_FAULT_SLICE_ORDER, SCALED_FAULT_SCALE_LANES,
    SCALED_MIX_SHIFT_OUTER = 16, SCALED_MIX_SHIFT_INNER = 15
};
#define SCALED_SM_120A 120                       /* -DPROBE_SM of the sm_120a build, for the preprocessor */
#define SCALED_BLOCK_THREADS(matrices) ((matrices) * SCALED_SLICES * SCALED_LANES_PER_WARP)     /* of a product */
#define SCALED_E2M1_PAIR_MAGNITUDES 0x0E000E00u  /* bits 9 .. 11 of both halves of a register */
#define SCALED_PAIR_SIGN_MASK 0x80008000u        /* bit 15 of both halves */
#define SCALED_E2M1_BYTE_MASK 0x3C3C3C3Cu        /* the central four bits of every byte */
#define SCALED_E4M3_PAIR_SIGNS 0x00800080u       /* of bytes 0 and 2 of a word: the halves of one register */
#define SCALED_E4M3_PAIR_MAGNITUDES 0x007F007Fu
#define SCALED_BF16_ROUNDING_BIAS 0x7FFFu        /* plus the kept half's lowest bit: to nearest, a tie to even */
#define SCALED_BF16_KEPT_MASK 0xFFFF0000u
#define SCALED_SWIGLU_LIMIT 10.0f
#define SCALED_E4M3_MAXIMUM 448.0f
#define SCALED_E4M3_MAXIMUM_INVERSE (1.0f / SCALED_E4M3_MAXIMUM)
#define SCALED_E4M3_SMALLEST_NORMAL 0.015625f    /* 2^-6: codes below it are 2^-9 apart, as those just above it */
#define SCALED_AMAX_FLOOR 1.0e-4f                /* the reference's lower bound of a block's largest magnitude */
#define SCALED_PACKED_SEED 0x243F6A88u
#define SCALED_SCALE_SEED 0x85A308D3u
#define SCALED_SCALE_WORD_BASE 0x76767676u       /* synthetic weight scales: E8M0 codes 118 .. 121 */
#define SCALED_SCALE_WORD_SPREAD 0x03030303u
#define SCALED_MIX_MULTIPLIER_FIRST 0x7feb352du
#define SCALED_MIX_MULTIPLIER_SECOND 0x846ca68bu
struct scaled_job {                              /* one expert with the rows routed to it: the columns of a block */
    uint32_t expert_index;
    uint32_t row_count;                          /* columns in use, from column 0 on */
    uint32_t pair_index[SCALED_MOST_COLUMNS];    /* a pair: a row at one of its experts. Unused: column 0's */
    float route_weight[SCALED_MOST_COLUMNS];
};
struct scaled_vectors {                          /* the vectors in front of a product and what the quantizer writes */
    float *values;                               /* [vector][size] */
    uint32_t *codes;                             /* [vector][load][lane place][step]: four E4M3 bytes a word */
    uint8_t *exponents;                          /* [vector][block of 32]: E8M0 codes */
    uint32_t size, pairs_per_vector;             /* six pairs read a row's input, one its own hidden vector: 6, 1 */
};
struct scaled_launch {                           /* what one launch of a product reads and writes */
    const uint4 *packed;                         /* [expert][matrix][tile][load][lane]: eight E2M1 codes a word */
    const uint32_t *scales;                      /* [expert][matrix][tile][load][lane group]: four E8M0 codes */
    const struct scaled_job *jobs;               /* [grid y] */
    struct scaled_vectors fed;                   /* what the columns carry */
    float *output;                               /* [pair][output size]: fp32 sums; gate with up: hidden values */
    uint32_t output_size, matrix_kind;           /* of the block's first matrix */
};
/* ---- the quantizer ---- */
/* The order of a quantized vector's bytes: lane place t of the column that carries it holds, of each step, inputs
 * 2t, 2t + 8, 2t + 1, 2t + 9 as bytes 0 to 3 of one word. The place of a byte's value in its block of 32; its word. */
__host__ __device__ inline uint32_t scaled_activation_offset(uint32_t step_in_block, uint32_t lane_place, uint32_t byte)
{
    return step_in_block * SCALED_INPUTS_PER_STEP + lane_place * SCALED_REGISTERS_PER_OPERAND_B
           + byte / SCALED_REGISTERS_PER_OPERAND_B + byte % SCALED_REGISTERS_PER_OPERAND_B * SCALED_HALF_INPUTS;
}
__host__ __device__ inline uint32_t scaled_activation_word(uint32_t block, uint32_t lane_place, uint32_t step_in_block)
{
    return (block / SCALED_BLOCKS_PER_LOAD * SCALED_LANES_PER_LANE_GROUP + lane_place) * SCALED_STEPS_PER_LOAD
           + block % SCALED_BLOCKS_PER_LOAD * SCALED_STEPS_PER_BLOCK + step_in_block;
}
/* An fp32 rounded to bfloat16, its upper half, to nearest with ties to even: the value the cast to bf16 keeps. */
__device__ __forceinline__ float scaled_bf16_value(float value)
{
    const uint32_t bits = __float_as_uint(value);
    return __uint_as_float((bits + SCALED_BF16_ROUNDING_BIAS + ((bits >> SCALED_HALF_BITS) & 1u))
                           & SCALED_BF16_KEPT_MASK);
}
/* Two fp32 as two 8-bit codes by a conversion of sm_120a: the first operand's code is the upper byte (unit 3a). */
#define SCALED_CONVERT_PAIR(instruction, upper, lower, pair)                                                        \
    asm volatile("{ .reg .b16 packed;\n" instruction " packed, %1, %2;\n" "cvt.u32.u16 %0, packed; }\n"             \
                 : "=r"(pair) : "f"(upper), "f"(lower))
/* The E8M0 code of the smallest power of two not under ratio (normal, positive): from its bits, or by cvt.rp. */
template <int quantizer>
__device__ __forceinline__ uint32_t scaled_scale_code(float ratio)
{
    const uint32_t bits = __float_as_uint(ratio);
    uint32_t code = (bits >> SCALED_FP32_EXPONENT_SHIFT) + ((bits & SCALED_FP32_MANTISSA_MASK) != 0u ? 1u : 0u);
#if PROBE_SM == SCALED_SM_120A
    if (quantizer == SCALED_QUANTIZER_CONVERSION) {
        SCALED_CONVERT_PAIR("cvt.rp.satfinite.ue8m0x2.f32", ratio, ratio, code);
        code &= SCALED_BYTE_MASK;
    }
#endif
    return code;
}
/* Four values, already divided by their block's scale, as the four E4M3 bytes of one word: each the nearest code,
 * its magnitude at most 448, a tie to the even code, the sign kept also where the code is zero. In software from
 * the fp32 fields, or two cvt.rn.satfinite.e4m3x2.f32, which unit 3a held to the same rule on every fp32. */
template <int quantizer>
__device__ __forceinline__ uint32_t scaled_codes_word(const float scaled[SCALED_WORD_BYTES])
{
    uint32_t word = 0;
    for (uint32_t byte = 0; quantizer == SCALED_QUANTIZER_SOFTWARE && byte < SCALED_WORD_BYTES; byte++) {
        const float magnitude = fminf(fabsf(scaled[byte]), SCALED_E4M3_MAXIMUM);
        const uint32_t exponent_field =
            __float_as_uint(fmaxf(magnitude, SCALED_E4M3_SMALLEST_NORMAL)) >> SCALED_FP32_EXPONENT_SHIFT;
        const float steps_per_unit =
            __uint_as_float((SCALED_E4M3_STEPS_EXPONENT_BASE - exponent_field) << SCALED_FP32_EXPONENT_SHIFT);
        const uint32_t steps = (uint32_t)__float2int_rn(magnitude * steps_per_unit);        /* 0 .. 16, ties to even */
        const uint32_t sign =
            (__float_as_uint(scaled[byte]) >> SCALED_E4M3_SIGN_FROM_FP32_SHIFT) & SCALED_E4M3_SIGN_BIT;
        word |= (((exponent_field << SCALED_E4M3_EXPONENT_SHIFT) + steps - SCALED_E4M3_CODE_BASE) | sign)
                << (byte * SCALED_BYTE_BITS);
    }
#if PROBE_SM == SCALED_SM_120A
    for (uint32_t byte = 0; quantizer == SCALED_QUANTIZER_CONVERSION && byte < SCALED_WORD_BYTES;
         byte += SCALED_CODES_PER_CONVERSION) {
        uint32_t pair;
        SCALED_CONVERT_PAIR("cvt.rn.satfinite.e4m3x2.f32", scaled[byte + 1u], scaled[byte], pair);
        word |= pair << (byte * SCALED_BYTE_BITS);
    }
#endif
    return word;
}
/* The reference's act_quant for blocks of 32: grid (vectors), a block of exactly size / 32 threads, one thread per
 * block of values of vector first_vector + blockIdx.x. The largest magnitude of the block (at least 1e-4: the floor
 * comes before either scale step; fmaxf passes over a not-a-number) times 1 / 448 in fp32, rounded up to a power of
 * two, is the scale; a value over the scale becomes its code. A block that holds an infinity or a not-a-number gets
 * unspecified codes and exponent, as linear_records.h says, and nothing else changes. */
template <int quantizer>
__device__ __forceinline__ void scaled_quantize(struct scaled_vectors vectors, uint32_t first_vector)
{
    const size_t vector = (size_t)first_vector + blockIdx.x;
    const uint32_t block = threadIdx.x, block_count = vectors.size / SCALED_VALUES_PER_BLOCK;
    const float *block_values = vectors.values + vector * vectors.size + block * SCALED_VALUES_PER_BLOCK;
    uint32_t *vector_codes = vectors.codes + vector * (vectors.size / SCALED_WORD_BYTES);
    float largest = 0.0f;
    for (uint32_t offset = 0; offset < SCALED_VALUES_PER_BLOCK; offset++) {
        largest = fmaxf(largest, fabsf(block_values[offset]));
    }
    const uint32_t scale_code =
        scaled_scale_code<quantizer>(fmaxf(largest, SCALED_AMAX_FLOOR) * SCALED_E4M3_MAXIMUM_INVERSE);
    const float inverse_scale = __uint_as_float((SCALED_FP32_EXPONENT_BIAS + SCALED_E8M0_BIAS - scale_code)
                                                << SCALED_FP32_EXPONENT_SHIFT);
    for (uint32_t word_in_block = 0; word_in_block < SCALED_VALUES_PER_BLOCK / SCALED_WORD_BYTES; word_in_block++) {
        const uint32_t lane_place = word_in_block / SCALED_STEPS_PER_BLOCK;
        const uint32_t step_in_block = word_in_block % SCALED_STEPS_PER_BLOCK;
        float scaled[SCALED_WORD_BYTES];
        for (uint32_t byte = 0; byte < SCALED_WORD_BYTES; byte++) {      /* by a power of two: exact */
            scaled[byte] = block_values[scaled_activation_offset(step_in_block, lane_place, byte)] * inverse_scale;
        }
        vector_codes[scaled_activation_word(block, lane_place, step_in_block)] = scaled_codes_word<quantizer>(scaled);
    }
    vectors.exponents[vector * block_count + block] = (uint8_t)scale_code;
}
#define SCALED_QUANTIZE_KERNEL(name, quantizer) \
    __global__ void name(struct scaled_vectors vectors, uint32_t first) { scaled_quantize<quantizer>(vectors, first); }
SCALED_QUANTIZE_KERNEL(scaled_quantize_software, SCALED_QUANTIZER_SOFTWARE)
#if PROBE_SM == SCALED_SM_120A
SCALED_QUANTIZE_KERNEL(scaled_quantize_conversion, SCALED_QUANTIZER_CONVERSION)
#endif
/* ---- the product ---- */
/* sums = A B + sums for the lane's part of each, by one instruction: its form, then what follows operand C. */
#define SCALED_MMA(form, scales)                                                                                    \
    "mma.sync.aligned." form " {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}" scales ";\n"
#define SCALED_MMA_OPERANDS(sums, a, first_b, second_b)                                                             \
    "=f"((sums)[SCALED_LOW_EVEN]), "=f"((sums)[SCALED_LOW_ODD]), "=f"((sums)[SCALED_HIGH_EVEN]),                    \
    "=f"((sums)[SCALED_HIGH_ODD])                                                                                   \
        : "r"((a)[SCALED_LOW_FIRST]), "r"((a)[SCALED_HIGH_FIRST]), "r"((a)[SCALED_LOW_SECOND]),                     \
          "r"((a)[SCALED_HIGH_SECOND]), "r"(first_b), "r"(second_b), "f"((sums)[SCALED_LOW_EVEN]),                  \
          "f"((sums)[SCALED_LOW_ODD]), "f"((sums)[SCALED_HIGH_EVEN]), "f"((sums)[SCALED_HIGH_ODD])
#define SCALED_FORM_F16 "m16n8k16.row.col.f32.f16.f16.f32"
#define SCALED_FORM_MIXED "kind::f8f6f4.m16n8k32.row.col.f32.e2m1.e4m3.f32"
#define SCALED_FORM_SCALED "kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e2m1.e4m3.f32.ue8m0"
/* scale word of A {byte, lane pair of the lane group}, scale word of B {byte, lane place}: of a load's two blocks */
#define SCALED_SELECTORS_FIRST_BLOCK ", %14, {0, 0}, %15, {0, 0}"
#define SCALED_SELECTORS_SECOND_BLOCK ", %14, {2, 0}, %15, {1, 0}"
/* The eight E2M1 codes of one stored word as the four registers of operand A of an m16n8k16: the code at bit
 * 4 r + 16 h is half h of register r; its magnitude bits go to bits 9 .. 11 of the half, its sign to bit 15. */
__device__ __forceinline__ void scaled_place_weights(uint32_t word, uint32_t weights[SCALED_REGISTERS_PER_OPERAND_A])
{
    for (int operand_register = 0; operand_register < SCALED_REGISTERS_PER_OPERAND_A; operand_register++) {
        const uint32_t codes = word >> (operand_register * SCALED_NIBBLE_BITS);
        weights[operand_register] = ((codes << SCALED_FP16_MAGNITUDE_BIT) & SCALED_E2M1_PAIR_MAGNITUDES)
                                    | ((codes << SCALED_E2M1_SIGN_SHIFT) & SCALED_PAIR_SIGN_MASK);
    }
}
/* Four E4M3 bytes of one word as the two registers of operand B of an m16n8k16, each byte the fp16 of its value
 * times 2^-8: bytes 0 and 2 are the halves of the first register, bytes 1 and 3 those of the second. */
__device__ __forceinline__ void scaled_place_codes(uint32_t word, uint32_t *low_inputs, uint32_t *high_inputs)
{
    const uint32_t odd_bytes = word >> SCALED_BYTE_BITS;
    *low_inputs = ((word & SCALED_E4M3_PAIR_SIGNS) << SCALED_BYTE_BITS)
                  | ((word & SCALED_E4M3_PAIR_MAGNITUDES) << SCALED_E4M3_MAGNITUDE_SHIFT);
    *high_inputs = ((odd_bytes & SCALED_E4M3_PAIR_SIGNS) << SCALED_BYTE_BITS)
                   | ((odd_bytes & SCALED_E4M3_PAIR_MAGNITUDES) << SCALED_E4M3_MAGNITUDE_SHIFT);
}
/* One block of 32 inputs of one column group for one lane: its four sums into its chains. placed: operand A, per
 * m16n8k16 (F16) or in placed[0] (the byte forms). first_fed, second_fed: the block's two activation words of the
 * lane's column. weight_codes: the load's scale word, byte 2 q + r: block q, output half r. column_codes: byte q of
 * [c] is the exponent of block q of the vector of the lane's result column c (FRESH, CHAINED: [0], of its fed
 * vector). scale_shift: what brings the scale byte of the lane's output half to byte 2 q of weight_codes. */
template <int product, int block>
__device__ __forceinline__ void scaled_block_sums(float chains[SCALED_ELEMENTS_PER_LANE],
                                                  const uint32_t placed[][SCALED_REGISTERS_PER_OPERAND_A],
                                                  uint32_t first_fed, uint32_t second_fed, uint32_t weight_codes,
                                                  const uint32_t column_codes[SCALED_COLUMNS_PER_LANE],
                                                  uint32_t scale_shift)
{
    float sums[SCALED_ELEMENTS_PER_LANE] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (product == SCALED_PRODUCT_F16) {
        uint32_t low_inputs, high_inputs;
        scaled_place_codes(first_fed, &low_inputs, &high_inputs);
        asm volatile(SCALED_MMA(SCALED_FORM_F16, "") : SCALED_MMA_OPERANDS(sums, placed[0], low_inputs, high_inputs));
        scaled_place_codes(second_fed, &low_inputs, &high_inputs);
        asm volatile(SCALED_MMA(SCALED_FORM_F16, "") : SCALED_MMA_OPERANDS(sums, placed[1], low_inputs, high_inputs));
    }
#if PROBE_SM == SCALED_SM_120A
    if (product == SCALED_PRODUCT_MIXED) {
        asm volatile(SCALED_MMA(SCALED_FORM_MIXED, "") : SCALED_MMA_OPERANDS(sums, placed[0], first_fed, second_fed));
    } else if (product >= SCALED_PRODUCT_FRESH) {
        float *into = product == SCALED_PRODUCT_CHAINED ? chains : sums;
        const uint32_t weight_scales = weight_codes >> scale_shift, column_scales = column_codes[0];
        if (block == 0) {
            asm volatile(SCALED_MMA(SCALED_FORM_SCALED, SCALED_SELECTORS_FIRST_BLOCK)
                         : SCALED_MMA_OPERANDS(into, placed[0], first_fed, second_fed), "r"(weight_scales),
                           "r"(column_scales));
        } else {
            asm volatile(SCALED_MMA(SCALED_FORM_SCALED, SCALED_SELECTORS_SECOND_BLOCK)
                         : SCALED_MMA_OPERANDS(into, placed[0], first_fed, second_fed), "r"(weight_scales),
                           "r"(column_scales));
        }
    }
#else
    (void)scale_shift;                           /* sm_80 has the first form alone */
#endif
    for (int element = 0; element < SCALED_ELEMENTS_PER_LANE; element++) {
        if (product <= SCALED_PRODUCT_MIXED) {   /* the block's power of two outside: both scales, the placements */
            const uint32_t weight_code = (weight_codes >> ((block * SCALED_BLOCKS_PER_LOAD
                                                            + element / SCALED_COLUMNS_PER_LANE) * SCALED_BYTE_BITS))
                                         & SCALED_BYTE_MASK;
            const uint32_t column_code =
                (column_codes[element % SCALED_COLUMNS_PER_LANE] >> (block * SCALED_BYTE_BITS)) & SCALED_BYTE_MASK;
            const uint32_t bias = product == SCALED_PRODUCT_F16 ? SCALED_PLACED_BIAS : SCALED_BYTES_BIAS;
            chains[element] = fmaf(sums[element], __uint_as_float((weight_code + column_code - bias)
                                                                  << SCALED_FP32_EXPONENT_SHIFT), chains[element]);
        } else if (product == SCALED_PRODUCT_FRESH) {
            chains[element] += sums[element];
        }
    }
}
/* The vector a pair reads: its row's input, six pairs a row, or its own. No division by a number of the launch. */
__device__ __forceinline__ size_t scaled_vector_of(const struct scaled_launch *launch, uint32_t pair)
{
    return launch->fed.pairs_per_vector == 1u ? pair : pair / SCALED_EXPERTS_PER_ROW;
}
/* What follows is written once per column group with the group's number a constant (the eight numbers below), so
 * that every chain and every word a lane holds is a register of its own: a loop would leave them in its stack. */
#define SCALED_EVERY_GROUP(statement)                                                                               \
    statement(0) statement(1) statement(2) statement(3) statement(4) statement(5) statement(6) statement(7)
#define SCALED_GROUP_SOURCES(group)                                                                                 \
    if (group < column_groups) {                                                                                    \
        const uint32_t *pairs = job->pair_index + group * SCALED_COLUMNS_PER_GROUP;                                 \
        const size_t fed_vector = scaled_vector_of(launch, pairs[lane_group]);                                      \
        fed[group] = (const uint4 *)launch->fed.codes + fed_vector * loads * SCALED_LANES_PER_LANE_GROUP + lane_place; \
        for (int column = 0; column < SCALED_COLUMNS_PER_LANE; column++) {                                          \
            const size_t vector = scales_inside                                                                     \
                ? fed_vector : scaled_vector_of(launch, pairs[lane_place * SCALED_COLUMNS_PER_LANE + column]);      \
            exponents[group][column] = (const uint16_t *)launch->fed.exponents + vector * loads;                    \
            chains[group][column] = chains[group][column + SCALED_COLUMNS_PER_LANE] = 0.0f;                         \
        }                                                                                                           \
    }
#define SCALED_GROUP_FETCH(group)                                                                                   \
    if (group < column_groups) {                                                                                    \
        fed_now[group] = fed[group][(size_t)load * SCALED_LANES_PER_LANE_GROUP];                                    \
        column_codes[group][0] = exponents[group][0][load];                                                         \
        column_codes[group][1] = scales_inside ? 0u : exponents[group][1][load];                                    \
    }
#define SCALED_GROUP_BLOCK(group, block, first, second)                                                             \
    if (group < column_groups) {                                                                                    \
        scaled_block_sums<product, block>(chains[group], placed, fed_now[group].first, fed_now[group].second,       \
                                          weight_codes, column_codes[group], scale_shift);                          \
    }
#define SCALED_GROUP_FIRST_BLOCK(group) SCALED_GROUP_BLOCK(group, 0, x, y)
#define SCALED_GROUP_SECOND_BLOCK(group) SCALED_GROUP_BLOCK(group, 1, z, w)
#define SCALED_GROUP_STORE(group)                                                                                   \
    for (int element = 0; group < column_groups && element < SCALED_ELEMENTS_PER_LANE; element++) {                 \
        chains_out[group * SCALED_ELEMENTS_PER_LANE + element] = chains[group][element];                            \
    }
/* Operand A of one block of 32 inputs from its two stored words: per m16n8k16, or the four registers of an
 * m16n8k32 of bytes: outputs g and g + 8 at the first word's inputs, then at the second word's. */
template <int product>
__device__ __forceinline__ void scaled_place_block(uint32_t first_word, uint32_t second_word,
                                                   uint32_t placed[][SCALED_REGISTERS_PER_OPERAND_A])
{
    if (product == SCALED_PRODUCT_F16) {
        scaled_place_weights(first_word, placed[0]);
        scaled_place_weights(second_word, placed[1]);
    } else {
        placed[0][SCALED_LOW_FIRST] = (first_word << SCALED_E2M1_BYTE_SHIFT) & SCALED_E2M1_BYTE_MASK;
        placed[0][SCALED_HIGH_FIRST] = (first_word >> SCALED_E2M1_BYTE_SHIFT) & SCALED_E2M1_BYTE_MASK;
        placed[0][SCALED_LOW_SECOND] = (second_word << SCALED_E2M1_BYTE_SHIFT) & SCALED_E2M1_BYTE_MASK;
        placed[0][SCALED_HIGH_SECOND] = (second_word >> SCALED_E2M1_BYTE_SHIFT) & SCALED_E2M1_BYTE_MASK;
    }
}
/* One slice of one tile for one lane: the chains of its four results of every column group over loads first_load ..
 * end_load - 1 (at least one) of the tile, each from +0; chains_out, [column group][element], is written once. */
template <int product, int column_groups, int fault>
__device__ __forceinline__ void scaled_slice_chains(const struct scaled_launch *launch, const struct scaled_job *job,
                                                    size_t tile_load, uint32_t lane_index, uint32_t first_load,
                                                    uint32_t end_load, float *chains_out)
{
    const int scales_inside = product >= SCALED_PRODUCT_FRESH;
    const uint32_t lane_group = lane_index / SCALED_LANES_PER_LANE_GROUP;
    const uint32_t lane_place = lane_index % SCALED_LANES_PER_LANE_GROUP;
    const uint32_t loads = launch->fed.size / SCALED_INPUTS_PER_LOAD;
    const uint32_t scale_shift =
        ((lane_index ^ (fault == SCALED_FAULT_SCALE_LANES ? 1u : 0u)) & 1u) * SCALED_BYTE_BITS;
    const uint4 *weights = launch->packed + tile_load * SCALED_LANES_PER_WARP + lane_index;
    const uint32_t *weight_scales = launch->scales + tile_load * SCALED_LANE_GROUPS + lane_group;
    /* per column group, of which a narrow kernel has the first alone: its column's vector at the lane's place */
    const uint4 *fed[SCALED_WIDE];
    const uint16_t *exponents[SCALED_WIDE][SCALED_COLUMNS_PER_LANE];
    float chains[SCALED_WIDE][SCALED_ELEMENTS_PER_LANE];
    SCALED_EVERY_GROUP(SCALED_GROUP_SOURCES)
    for (uint32_t load = first_load; load < end_load; load++) {
        const uint4 packed = weights[(size_t)load * SCALED_LANES_PER_WARP];
        const uint32_t weight_codes = weight_scales[(size_t)load * SCALED_LANE_GROUPS];
        uint4 fed_now[SCALED_WIDE];
        uint32_t column_codes[SCALED_WIDE][SCALED_COLUMNS_PER_LANE];
        uint32_t placed[SCALED_STEPS_PER_BLOCK][SCALED_REGISTERS_PER_OPERAND_A];
        SCALED_EVERY_GROUP(SCALED_GROUP_FETCH)
        scaled_place_block<product>(packed.x, packed.y, placed);
        SCALED_EVERY_GROUP(SCALED_GROUP_FIRST_BLOCK)
        scaled_place_block<product>(packed.z, packed.w, placed);
        SCALED_EVERY_GROUP(SCALED_GROUP_SECOND_BLOCK)
    }
    SCALED_EVERY_GROUP(SCALED_GROUP_STORE)
}
/* One result: the chains its slices left, slice_stride floats apart, added from +0 in ascending slice order. */
template <int fault>
__device__ __forceinline__ float scaled_sum_of_slices(const float *first_chain, uint32_t slice_stride,
                                                      uint32_t row_count)
{
    const uint32_t first_slice = fault == SCALED_FAULT_SLICE_ORDER && row_count > 1u ? 1u : 0u;
    float sum = 0.0f;
    for (uint32_t position = 0; position < SCALED_SLICES; position++) {
        sum += first_chain[(size_t)((first_slice + position) % SCALED_SLICES) * slice_stride];
    }
    return sum;
}
/* One block of a product: `matrices` matrices of one expert for one tile of 16 outputs, SCALED_SLICES warps a
 * matrix; grid (tiles, jobs); block_chains: the block's shared memory, [warp][lane][column group][element]. One
 * matrix: its fp32 sums go to output. Two: gate and up, each rounded to bf16 as the reference's product returns it;
 * h = silu(min(gate, L)) * clamp(up, -L, L) * route weight, cast to bf16, goes to output. Column c of a job carries
 * the vector of pair_index[c] and stores at that pair; a column from row_count on is stored nowhere. After the
 * block's one barrier every warp finishes its share of the lanes' results. */
template <int product, int column_groups, int matrices, int fault>
__device__ __forceinline__ void scaled_product(const struct scaled_launch *launch, float *block_chains)
{
    const uint32_t loads = launch->fed.size / SCALED_INPUTS_PER_LOAD;
    const uint32_t warp = threadIdx.x / SCALED_LANES_PER_WARP, lane_index = threadIdx.x % SCALED_LANES_PER_WARP;
    const uint32_t slice = warp % SCALED_SLICES, lane_floats = column_groups * SCALED_ELEMENTS_PER_LANE;
    const uint32_t slice_stride = SCALED_LANES_PER_WARP * lane_floats, warps = matrices * SCALED_SLICES;
    const struct scaled_job *job = launch->jobs + blockIdx.y;
    const size_t matrix = (size_t)job->expert_index * SCALED_MATRIX_KINDS + launch->matrix_kind + warp / SCALED_SLICES;
    const size_t tile_load = matrix * SCALED_TILE_LOADS_PER_MATRIX + (size_t)blockIdx.x * loads;
    scaled_slice_chains<product, column_groups, fault>(launch, job, tile_load, lane_index,
                                                       slice * loads / SCALED_SLICES,
                                                       (slice + 1u) * loads / SCALED_SLICES,
                                                       block_chains + (size_t)threadIdx.x * lane_floats);
    __syncthreads();
    for (uint32_t result = warp; result < lane_floats; result += warps) {     /* warp w: results w, w + warps, .. */
        const uint32_t element = result % SCALED_ELEMENTS_PER_LANE;
        const uint32_t column = result / SCALED_ELEMENTS_PER_LANE * SCALED_COLUMNS_PER_GROUP
                                + lane_index % SCALED_LANES_PER_LANE_GROUP * SCALED_COLUMNS_PER_LANE
                                + element % SCALED_COLUMNS_PER_LANE;
        const size_t output = (size_t)blockIdx.x * SCALED_OUTPUTS_PER_TILE + lane_index / SCALED_LANES_PER_LANE_GROUP
                              + element / SCALED_COLUMNS_PER_LANE * SCALED_HALF_TILE;
        const float *chain = block_chains + (size_t)lane_index * lane_floats + result;
        if (column >= job->row_count) { continue; }
        float *stored = launch->output + (size_t)job->pair_index[column] * launch->output_size + output;
        const float sum = scaled_sum_of_slices<fault>(chain, slice_stride, job->row_count);
        if (matrices == SCALED_ONE_MATRIX) {
            *stored = sum;
        } else {
            const float gate = fminf(scaled_bf16_value(sum), SCALED_SWIGLU_LIMIT);
            const float up = scaled_bf16_value(
                scaled_sum_of_slices<fault>(chain + (size_t)SCALED_SLICES * slice_stride, slice_stride,
                                            job->row_count));
            const float bounded_up = fminf(fmaxf(up, -SCALED_SWIGLU_LIMIT), SCALED_SWIGLU_LIMIT);
            *stored = scaled_bf16_value(gate / (1.0f + expf(-gate)) * bounded_up * job->route_weight[column]);
        }
    }
}
/* A row's six experts, each rounded to bf16 as the reference's product returns it, added in slot order, which is
 * ascending expert order, the order of the reference's loop: grid (rows), any block; row first_row + blockIdx.x. */
__global__ void scaled_row_sum_kernel(const float *pair_output, float *row_output, uint32_t first_row)
{
    const size_t row = (size_t)first_row + blockIdx.x;
    for (uint32_t column = threadIdx.x; column < SCALED_HIDDEN_SIZE; column += blockDim.x) {
        float sum = 0.0f;
        for (uint32_t slot = 0; slot < SCALED_EXPERTS_PER_ROW; slot++) {
            sum += scaled_bf16_value(pair_output[(row * SCALED_EXPERTS_PER_ROW + slot) * SCALED_HIDDEN_SIZE + column]);
        }
        row_output[row * SCALED_HIDDEN_SIZE + column] = sum;
    }
}
/* ---- the kernels of one variant: scaled_gate_up_<id> and scaled_matrix_<id> for jobs of up to eight rows,
 * scaled_wide_* for jobs of up to 64; blocks of SCALED_BLOCK_THREADS(matrices) threads. bound is OPEN (none: the
 * baseline's, as no kernel of the engine states one) or STATED: `blocks` blocks of gate with up to an SM, and twice
 * as many of the half-size block of one matrix. ---- */
#define SCALED_BOUND_OPEN(threads, blocks)
#define SCALED_BOUND_STATED(threads, blocks) __launch_bounds__(threads, blocks)
#define SCALED_PRODUCT_KERNEL(name, product, column_groups, matrices, bound, blocks, fault)                          \
    __global__ void SCALED_BOUND_##bound(SCALED_BLOCK_THREADS(matrices), blocks) name(struct scaled_launch launch)  \
    {                                                                                                               \
        __shared__ float block_chains[SCALED_BLOCK_THREADS(matrices) * (column_groups) * SCALED_ELEMENTS_PER_LANE]; \
        scaled_product<product, column_groups, matrices, fault>(&launch, block_chains);                             \
    }
#define SCALED_PRODUCT_KERNELS(id, product, bound, blocks, fault)                                                   \
    SCALED_PRODUCT_KERNEL(scaled_gate_up_##id, product, SCALED_NARROW, SCALED_GATE_UP_MATRICES, bound, blocks, fault) \
    SCALED_PRODUCT_KERNEL(scaled_matrix_##id, product, SCALED_NARROW, SCALED_ONE_MATRIX, bound,                     \
                          SCALED_GATE_UP_MATRICES * (blocks), fault)                                                \
    SCALED_PRODUCT_KERNEL(scaled_wide_gate_up_##id, product, SCALED_WIDE, SCALED_GATE_UP_MATRICES, bound, blocks,   \
                          fault)                                                                                    \
    SCALED_PRODUCT_KERNEL(scaled_wide_matrix_##id, product, SCALED_WIDE, SCALED_ONE_MATRIX, bound,                  \
                          SCALED_GATE_UP_MATRICES * (blocks), fault)
/* ---- the synthetic experts: the same words on the host and on the card ---- */
__host__ __device__ inline uint32_t scaled_mix_bits(uint32_t value)
{
    value ^= value >> SCALED_MIX_SHIFT_OUTER;
    value *= SCALED_MIX_MULTIPLIER_FIRST;
    value ^= value >> SCALED_MIX_SHIFT_INNER;
    value *= SCALED_MIX_MULTIPLIER_SECOND;
    value ^= value >> SCALED_MIX_SHIFT_OUTER;
    return value;
}
/* A stored word is a hash of its place in its array: eight weight nibbles, or the E8M0 codes of four scales. */
__host__ __device__ inline uint32_t scaled_stored_word(uint64_t word_index, uint32_t is_scale)
{
    const uint32_t seed = is_scale != 0u ? SCALED_SCALE_SEED : SCALED_PACKED_SEED;
    const uint32_t bits = scaled_mix_bits(seed + (uint32_t)word_index);
    return is_scale != 0u ? SCALED_SCALE_WORD_BASE + (bits & SCALED_SCALE_WORD_SPREAD) : bits;
}
/* Any grid, any block: words 0 .. word_count - 1 of a stored array get the stored words of their places. */
__global__ void scaled_fill_stored_kernel(uint32_t *words, uint32_t is_scale, uint64_t word_count)
{
    const uint64_t thread_total = (uint64_t)gridDim.x * blockDim.x;
    for (uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; index < word_count; index += thread_total) {
        words[index] = scaled_stored_word(index, is_scale);
    }
}
#endif /* BLOCK_SCALED_KERNELS_CUH */
