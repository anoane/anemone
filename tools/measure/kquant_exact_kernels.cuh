/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * kquant_exact_kernels.cuh: the routed experts of one MoE layer with K-quant weights (Q2_K, Q3_K of the GGUF format)
 * under contract C1, written for the engine: ONE body for both block types and both stages (gate with up, down),
 * every lever a compile-time constant, a kernel stamped per choice (KQ_STAMP). Round 1, package 5 of the speed
 * campaign (research/speed/ROUND_1.md); kquant_forms_probe.cu times the stamps.
 *
 * THE FILE'S BLOCK. 256 consecutive inputs of one output are a block, 16 a group.
 *   Q2_K, 84 bytes: weight = d * scale * code - dmin * minimum; 2-bit codes, a 4-bit scale and minimum a group.
 *   Q3_K, 110 bytes: weight = d * (scale - 32) * code; 3-bit codes -4 .. 3, a 6-bit scale a group. d, dmin: fp16.
 * THE RECORD: the 16 blocks a tile of 16 outputs has for the same 256 inputs, in the order the instruction's lanes
 * take them. Lane L = 4 g + t holds, of operand A, outputs g and g + 8 at inputs 2t, 2t + 1, 2t + 8, 2t + 9 of a
 * group: register r is output g + 8 (r % 2) at inputs 2t + 8 (r / 2) (its half 0) and the next (half 1).
 *   codes         [half block][lane], 16 bytes: word p is pair p of the half's four pairs of groups; the stored code
 *                 of step s of the pair, register r, half h is at bit 8 s + 2 r + 16 h
 *   signs         Q3_K, [half block][lane], 8 bytes: word q is instructions 4 q .. 4 q + 3; a negative code is
 *                 stored as its magnitude less one with its flag at bit 4 j + r + 16 h of instruction 4 q + j
 *   group scales  [half block][lane group], 16 bytes: byte 2 i + o is instruction i, output g + 8 o (Q2_K: the file's
 *                 byte, scale in the low half; Q3_K: the 6-bit scale in a byte: 114 bytes a block against 110)
 *   block scales  [output]: d, then Q2_K's dmin, as the file's fp16
 * CONTRACT C1, as kquant_expert_probe.cu defined it (a is an input's E4M3 value, 2^E the scale of its block of 32):
 *   1. G = the sum over a group of code * a: one instruction m16n8k16 from a zero accumulator, exact if the unit
 *      multiplies subnormal operands exactly (a code is a whole number in a subnormal's mantissa);
 *   2. per pair of groups: R = fma(G1, s1, fma(G0, s0, 0)), then P = fma(R, 2^E, P), P the block's;
 *   3. Q2_K, per group: M = fma(X, m, M), X = 2^E * the sum of the group's 16 a, made once per vector;
 *   4. per block: T = fma(P, d, T), then Q2_K T = fma(M, -dmin, T), T a slice's; 5. slices added in slice order.
 * THE LEVERS (template arguments of kq_slice; each leaves the bits of C1 where the unit computes exactly):
 *   hold      registers: a half block's codes, scales and signs and a load's inputs are held (the earlier probe);
 *             fetch: every instruction reads its own words, 4 bytes each: fewer live values; shared: as fetch, the
 *             stored words from the lane's slot of shared memory, brought there by asynchronous copies (ledger K10)
 *   ahead     1: the stored words of the next half block are asked before the present one is worked on (fetch: its
 *             first word of each part, which brings the lines), never beyond the slice's last half block
 *   operand   codes: operand B decoded from E4M3 bytes per instruction; placed16: the same fp16 pairs written once
 *             by kq_prepare_kernel; carried: bf16 pairs of a * 2^(E + 110), codes in a bf16 subnormal: G comes out
 *             times 2^(E - 23), step 2's multiply is an addition and d takes the 2^23 (exact while every dequantized
 *             input is 0 or of magnitude 2^-31 .. 2^17: the engine's quantizer has to hold that; the probe's are);
 *             fp8 (sm_120a): codes as E4M3 bytes in m16n8k16 e4m3 by e4m3, the vector's bytes untouched
 * A LAUNCH. Grid (tiles of 16 outputs, jobs); a block is KQ_STAGE_THREADS threads: per matrix of the stage a warp
 * per slice of the input width. The kernels check nothing: the host holds every job, grid and array to its range
 * before a launch. No atomics. Measurement tool: not part of the engine.
 */
#ifndef KQUANT_EXACT_KERNELS_CUH
#define KQUANT_EXACT_KERNELS_CUH
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

#ifndef KQ_ASYNC_COPIES                          /* 0: this compiler refused the asynchronous copy: no hold shared */
#define KQ_ASYNC_COPIES 1
#endif
/* sm_120a has the FP8 instruction, sm_80 has not (h01). The build's own switch says which, as CONVENTIONS 1.1 has
 * it: the host's pass over this file never has the compiler's symbol and must see the kernels it launches (i03a). */
#define KQ_SM_120A 120                           /* -DPROBE_SM of the sm_120a build */
#define KQ_HAS_FP8 (PROBE_SM == KQ_SM_120A)
enum {
    /* the model and a launch: a pair is a row at one of its experts (row * 6 + slot); a stage is named by its
     * number of matrices: gate with up, which share the input, or down */
    KQ_HIDDEN_SIZE = 5120, KQ_INNER_SIZE = 2304, KQ_EXPERTS_PER_ROW = 6, KQ_MOST_ROWS = 8, KQ_EXPERT_COUNT = 128,
    KQ_MOST_PAIRS = KQ_MOST_ROWS * KQ_EXPERTS_PER_ROW, KQ_MATRIX_KINDS = 3, KQ_UP_MATRIX = 1, KQ_DOWN_MATRIX = 2,
    KQ_GATE_UP = 2, KQ_DOWN = 1, KQ_STAGES = 2, KQ_GATE_UP_SLICES = 4, KQ_DOWN_SLICES = 3, KQ_ROW_SUM_THREADS = 256,
    /* the instruction: a lane holds four registers of weights, two of inputs, four sums (outputs g, g + 8 by the
     * columns 2t, 2t + 1) */
    KQ_LANES_PER_WARP = 32, KQ_LANES_PER_GROUP = 4, KQ_TILE_OUTPUTS = 16, KQ_HALF_OUTPUTS = 8, KQ_HALF_INPUTS = 8,
    KQ_WEIGHT_REGISTERS = 4, KQ_REGISTER_HALVES = 2, KQ_LANE_SUMS = 4, KQ_LANE_COLUMNS = 2, KQ_LANE_OUTPUTS = 2,
    /* the block and its record */
    KQ_Q2_K = 0, KQ_Q3_K = 1, KQ_TYPES = 2, KQ_BLOCK_WEIGHTS = 256, KQ_GROUP_WEIGHTS = 16, KQ_BLOCK_HALVES = 2,
    KQ_HALF_GROUPS = 8, KQ_HALF_PAIRS = 4, KQ_PAIR_GROUPS = 2, KQ_LOAD_GROUPS = 4, KQ_LOAD_PAIRS = 2,
    KQ_WORD_BYTES = 4, KQ_LOAD_WORDS = 4, KQ_SIGN_WORDS_PER_HALF = 2, KQ_SCALE_WORDS_AT = 4, KQ_SIGN_WORDS_AT = 8,
    KQ_HALF_WORDS = 10, KQ_SIGNS_AT = 1024, KQ_Q2_K_GROUP_SCALES_AT = 1024, KQ_Q3_K_GROUP_SCALES_AT = 1536,
    KQ_GROUP_SCALE_BYTES = 256, KQ_Q2_K_RECORD_BYTES = 1344, KQ_Q3_K_RECORD_BYTES = 1824, KQ_Q2_K_BLOCK_SCALES = 2,
    KQ_Q3_K_BLOCK_SCALES = 1, KQ_MATRIX_RECORDS = KQ_HIDDEN_SIZE * KQ_INNER_SIZE / (KQ_TILE_OUTPUTS * KQ_BLOCK_WEIGHTS),
    KQ_CODE_BITS = 2, KQ_STEP_CODE_BITS = 8, KQ_STEP_SIGN_BITS = 4, KQ_CODE_PAIR_MASK = 0x00030003,
    KQ_SIGN_PAIR_MASK = 0x00010001, KQ_Q3_K_SCALE_BIAS = 32, KQ_FP16_SIGN_BIT = 15, KQ_BYTE_BITS = 8,
    KQ_BYTE_MASK = 0xFF, KQ_HALF_BITS = 16, KQ_NIBBLE_BITS = 4, KQ_NIBBLE_MASK = 15, KQ_FP16_BYTES = 2,
    /* one output's four codes of an instruction as four bytes (kq_code_bytes), its four sign flags likewise */
    KQ_OUTPUT_CODES_MASK = 0x00330033, KQ_BYTE_CODES_MASK = 0x03030303, KQ_OUTPUT_SIGNS_MASK = 0x00050005,
    KQ_BYTE_SIGNS_MASK = 0x01010101, KQ_SIGN_SPREAD_SHIFT = 6, KQ_FP8_SIGN_AND_ONE = 0x81,
    /* a vector made ready: a block of 32 under one scale 2^(E8M0 code - 127) is a pair of groups; a 16-byte load is
     * four instructions; ((b & 0x80) << 8) | ((b & 0x7F) << 7) is the fp16 of E4M3 byte b times 2^-8 */
    KQ_ACTIVATION_BLOCK = 32, KQ_E4M3_PAIR_SIGN_MASK = 0x00800080, KQ_E4M3_PAIR_MAGNITUDE_MASK = 0x007F007F,
    KQ_E4M3_MAGNITUDE_SHIFT = 7, KQ_E4M3_EXPONENT_SHIFT = 3, KQ_E4M3_EXPONENT_BIAS = 7, KQ_E4M3_IMPLICIT_ONE = 8,
    KQ_E4M3_SIGN_FROM_FP32_SHIFT = 24, KQ_E8M0_BIAS = 127, KQ_FP32_EXPONENT_SHIFT = 23, KQ_FP32_EXPONENT_BIAS = 127,
    KQ_FP32_MANTISSA_MASK = 0x007FFFFF, KQ_BF16_ROUNDING_BIAS = 0x7FFF,
    /* quantizing a value of fp32 exponent field f: codes are 2^(f - 130) apart; the code is 8 (f - 120) + steps - 8 */
    KQ_E4M3_STEPS_EXPONENT_BASE = 2 * KQ_FP32_EXPONENT_BIAS + KQ_E4M3_EXPONENT_SHIFT,
    KQ_E4M3_CODE_BASE = ((KQ_FP32_EXPONENT_BIAS - KQ_E4M3_EXPONENT_BIAS) << KQ_E4M3_EXPONENT_SHIFT)
                        + KQ_E4M3_IMPLICIT_ONE,
    /* an instruction's sum is the true one times 2^-(24 + 8) with fp16 operands, times 2^-9 with FP8 ones: what
     * is added to E's code for the fp32 field of step 2's power; a carried operand is a * 2^(E + 110) */
    KQ_FP16_POWER_OFFSET = 32, KQ_FP8_POWER_OFFSET = 9, KQ_CARRIED_EXPONENT = 110,
    /* the levers, and what a prepare kernel writes (kq_prepare_kernel the first four) */
    KQ_HOLD_REGISTERS = 0, KQ_HOLD_FETCH, KQ_HOLD_SHARED,
    KQ_OPERAND_CODES = 0, KQ_OPERAND_PLACED16, KQ_OPERAND_CARRIED, KQ_OPERAND_FP8,
    KQ_INPUTS_E4M3 = 0, KQ_INPUTS_PLACED16, KQ_INPUTS_CARRIED, KQ_INPUTS_RAW, KQ_INPUTS_INT8, KQ_INPUT_KINDS,
    KQ_SLOT_LOADS = 3, KQ_SLOT_WORDS = KQ_SLOT_LOADS * KQ_LOAD_WORDS, KQ_SLOT_STAGES = 2
};
#define KQ_FP32_SIGN_MASK 0x80000000u
#define KQ_BF16_KEPT_MASK 0xFFFF0000u            /* the upper half; the bias and its lowest bit round ties to even */
#define KQ_SWIGLU_LIMIT 10.0f
#define KQ_E4M3_MAXIMUM 448.0f
#define KQ_E4M3_MAXIMUM_INVERSE (1.0f / KQ_E4M3_MAXIMUM)
#define KQ_E4M3_SMALLEST_NORMAL 0.015625f        /* 2^-6: codes below it are 2^-9 apart, as those just above it */
#define KQ_AMAX_FLOOR 1.0e-4f                    /* the reference's lower bound of a block's largest magnitude */
#define KQ_CARRIED_UNDO 8388608.0f               /* 2^23: what d takes where the operand carried 2^(110 - 133) */
#define KQ_STAGE_SLICES(matrices) ((matrices) == KQ_DOWN ? KQ_DOWN_SLICES : KQ_GATE_UP_SLICES)
#define KQ_STAGE_INPUT_SIZE(matrices) ((matrices) == KQ_DOWN ? KQ_INNER_SIZE : KQ_HIDDEN_SIZE)
#define KQ_STAGE_TILES(matrices) (((matrices) == KQ_DOWN ? KQ_HIDDEN_SIZE : KQ_INNER_SIZE) / KQ_TILE_OUTPUTS)
#define KQ_STAGE_THREADS(matrices) ((matrices) * KQ_STAGE_SLICES(matrices) * KQ_LANES_PER_WARP)
#define KQ_STAGE_BLOCKS(matrices) (KQ_STAGE_INPUT_SIZE(matrices) / KQ_BLOCK_WEIGHTS)
#define KQ_SLICE_BLOCKS(matrices) (KQ_STAGE_BLOCKS(matrices) / KQ_STAGE_SLICES(matrices))
#define KQ_RECORD_BYTES(kquant) ((kquant) == KQ_Q2_K ? KQ_Q2_K_RECORD_BYTES : KQ_Q3_K_RECORD_BYTES)
#define KQ_GROUP_SCALES_AT(kquant) ((kquant) == KQ_Q2_K ? KQ_Q2_K_GROUP_SCALES_AT : KQ_Q3_K_GROUP_SCALES_AT)
#define KQ_BLOCK_SCALES(kquant) ((kquant) == KQ_Q2_K ? KQ_Q2_K_BLOCK_SCALES : KQ_Q3_K_BLOCK_SCALES)
#define KQ_BLOCK_SCALES_AT(kquant) (KQ_GROUP_SCALES_AT(kquant) + KQ_GROUP_SCALE_BYTES)
#define KQ_EXPERT_BYTES(kquant) ((uint64_t)KQ_MATRIX_KINDS * KQ_MATRIX_RECORDS * KQ_RECORD_BYTES(kquant))
#define KQ_OPERAND_IS_PAIRS(operand) ((operand) == KQ_OPERAND_PLACED16 || (operand) == KQ_OPERAND_CARRIED)
#define KQ_SHARED_SUMS(matrices) (KQ_STAGE_THREADS(matrices) * KQ_LANE_SUMS)
#define KQ_SHARED_SLOT_LOADS(hold, matrices) \
    ((hold) == KQ_HOLD_SHARED ? KQ_SLOT_STAGES * KQ_STAGE_THREADS(matrices) * KQ_SLOT_LOADS : 1)
#define KQ_RECORD_END(kquant) (KQ_BLOCK_SCALES_AT(kquant) + KQ_TILE_OUTPUTS * KQ_BLOCK_SCALES(kquant) * KQ_FP16_BYTES)
static_assert(KQ_RECORD_BYTES(KQ_Q2_K) == KQ_RECORD_END(KQ_Q2_K) && KQ_RECORD_BYTES(KQ_Q3_K) == KQ_RECORD_END(KQ_Q3_K)
              && KQ_STAGE_BLOCKS(KQ_GATE_UP) % KQ_GATE_UP_SLICES + KQ_STAGE_BLOCKS(KQ_DOWN) % KQ_DOWN_SLICES == 0
              && KQ_HIDDEN_SIZE % KQ_BLOCK_WEIGHTS + KQ_INNER_SIZE % KQ_BLOCK_WEIGHTS == 0,
              "a record that is not the sum of its parts, or a width that is not whole blocks for every slice");

struct kq_job {                                  /* one expert with the rows routed to it: a block's columns */
    uint32_t expert, row_count;                  /* row_count: the columns in use, 1 .. KQ_MOST_ROWS */
    uint32_t pair[KQ_MOST_ROWS];                 /* of each column; a column not in use repeats column 0 */
    float route_weight[KQ_MOST_ROWS];
};
struct kq_vectors {                              /* the vectors in front of a product, made ready by a prepare kernel */
    const float *values;                         /* [vector][size]: what is made ready */
    uint32_t *words;                             /* [vector][load of 64][lane position]: 16 bytes, E4M3 or integers */
    uint32_t *pairs;                             /* [vector][load][lane position][instruction]: B's two registers */
    uint8_t *exponents;                          /* [vector][block of 32]: E8M0 codes */
    float *group_sums;                           /* Q2_K: [vector][group of 16]: X of step 3 */
    float *scales;                               /* 8-bit integer inputs: [vector][block of 256] */
    uint32_t *minimum_operand;                   /* Q2_K, integer inputs: [vector][block][lane position]: two words */
    uint32_t size, pairs_per_vector;             /* 5,120 and 6: a row's input; 2,304 and 1: a pair's hidden vector */
};
struct kq_launch {
    const uint8_t *stored;                       /* [expert][matrix kind][tile][block of 256 inputs]: records */
    struct kq_vectors vectors;
    const struct kq_job *jobs;
    float *hidden_values, *pair_output;          /* gate with up: [pair][inner size]; down: [pair][hidden size] */
};
struct kq_lane {                                 /* what a lane reads in its slice: set once, from its job */
    const uint8_t *records;                      /* its tile's records of its matrix, block 0 first */
    const uint32_t *fed;                         /* the column it feeds: the vector's words or pairs at its position */
    const uint8_t *exponents[KQ_LANE_COLUMNS];   /* of its two result columns */
    const float *group_sums[KQ_LANE_COLUMNS], *scales[KQ_LANE_COLUMNS];
    const uint32_t *minimum_operand;             /* of the column it feeds */
    uint32_t *slots;                             /* hold shared: its KQ_SLOT_WORDS of stage 0; stage 1 a block on */
    uint32_t lane, slot_stage_words;
};
struct kq_half {                                 /* a lane's stored words of one half block */
    uint32_t held[KQ_HALF_WORDS];                /* hold registers: codes 0 .. 3, scales 4 .. 7, signs 8, 9 */
    const uint32_t *codes, *scales, *signs;      /* hold fetch: in the record; hold shared: in the lane's slot */
};
struct kq_held {                                 /* hold registers: what a lane holds of its vectors between loads */
    uint32_t fed[KQ_LOAD_WORDS];                 /* a load's four instructions, or a pair's four registers */
    float group_sums[KQ_LANE_COLUMNS][KQ_LOAD_GROUPS];
    uint32_t exponents[KQ_LANE_COLUMNS];         /* a half block's four E8M0 codes */
};
typedef void kq_launcher(const struct kq_launch *launch, uint32_t job_count);
struct kq_kernels {                              /* a stamped set: [type][gate with up, down], and its kernels' names */
    kq_launcher *launch[KQ_TYPES][KQ_STAGES];
    const char *names[KQ_TYPES];
};

/* ---- the vectors made ready ---- */
/* A vector's order. Lane position t of the column that carries a vector needs, per instruction, inputs 2t, 2t + 1
 * (register 0 of operand B) and 2t + 8, 2t + 9 (register 1): one word, bytes 0 and 2 register 0, 1 and 3 register 1.
 * kq_value_offset is a byte's place in its block of 32; words are [load of 64 values][t][instruction]. */
__host__ __device__ inline uint32_t kq_value_offset(uint32_t step, uint32_t position, uint32_t byte) {
    return step * KQ_GROUP_WEIGHTS + position * KQ_REGISTER_HALVES + byte / KQ_REGISTER_HALVES
           + byte % KQ_REGISTER_HALVES * KQ_HALF_INPUTS;
}
__host__ __device__ inline uint32_t kq_word_index(uint32_t block, uint32_t position, uint32_t step) {
    return ((block / KQ_LOAD_PAIRS) * KQ_LANES_PER_GROUP + position) * KQ_LOAD_GROUPS
           + block % KQ_LOAD_PAIRS * KQ_PAIR_GROUPS + step;
}
/* An fp32 rounded to bf16, its upper half, to nearest with ties to even: the value the cast to bf16 keeps. */
__device__ __forceinline__ float kq_bf16_value(float value) {
    const uint32_t bits = __float_as_uint(value);
    return __uint_as_float((bits + KQ_BF16_ROUNDING_BIAS + ((bits >> KQ_HALF_BITS) & 1u)) & KQ_BF16_KEPT_MASK);
}
/* Bytes 0 and 2 of a word of E4M3 codes as one register of operand B: each the fp16 of its value times 2^-8. */
__device__ __forceinline__ uint32_t kq_place_e4m3_pair(uint32_t word) {
    return ((word & KQ_E4M3_PAIR_SIGN_MASK) << KQ_BYTE_BITS)
           | ((word & KQ_E4M3_PAIR_MAGNITUDE_MASK) << KQ_E4M3_MAGNITUDE_SHIFT);
}
/* 2^(field - 127) for an fp32 exponent field of 1 .. 254. */
__device__ __forceinline__ float kq_power_of_two(uint32_t field) {
    return __uint_as_float(field << KQ_FP32_EXPONENT_SHIFT);
}
/* The E4M3 code nearest to value * inverse_scale (clamped to +-448, ties to the even code) and, exact, its value. */
__device__ __forceinline__ uint32_t kq_quantize_to_e4m3(float value, float inverse_scale, float *quantized) {
    const float scaled = value * inverse_scale;                   /* by a power of two: exact */
    const float magnitude = fminf(fabsf(scaled), KQ_E4M3_MAXIMUM);
    const uint32_t field = __float_as_uint(fmaxf(magnitude, KQ_E4M3_SMALLEST_NORMAL)) >> KQ_FP32_EXPONENT_SHIFT;
    const float step = kq_power_of_two(field - KQ_E4M3_EXPONENT_SHIFT);
    const uint32_t steps = (uint32_t)__float2int_rn(magnitude * kq_power_of_two(KQ_E4M3_STEPS_EXPONENT_BASE - field));
    const uint32_t sign = __float_as_uint(scaled) & KQ_FP32_SIGN_MASK;               /* steps: 0 .. 16, ties to even */
    *quantized = __uint_as_float(__float_as_uint((float)steps * step) | sign);
    return ((field << KQ_E4M3_EXPONENT_SHIFT) + steps - KQ_E4M3_CODE_BASE) | (sign >> KQ_E4M3_SIGN_FROM_FP32_SHIFT);
}
/* The reference's act_quant for blocks of 32 and what a product kernel wants beside it: grid (vectors), a thread per
 * block. The scale is the largest magnitude (at least 1e-4) times 1 / 448 in fp32, rounded up to a power of two as
 * the reference reads it from the fp32 fields. inputs: KQ_INPUTS_E4M3 the codes and E8M0 codes; _PLACED16 also the
 * fp16 pairs of operand codes; _CARRIED also bf16 pairs of the dequantized value times 2^110; _RAW (the 16-bit
 * inputs unquantized, ledger K8) only such pairs, of the values themselves, which are bfloat16. Q2_K also gets its
 * groups' X: exact for quantized inputs; for raw ones the fp32 sum in the order of the vector's words. */
template <int kquant, int inputs>
__global__ void kq_prepare_kernel(struct kq_vectors vectors) {
    const uint32_t block_count = vectors.size / KQ_ACTIVATION_BLOCK;
    uint32_t *vector_words = vectors.words + (size_t)blockIdx.x * (vectors.size / KQ_WORD_BYTES);
    uint32_t *vector_pairs = vectors.pairs + (size_t)blockIdx.x * (vectors.size / KQ_REGISTER_HALVES);
    for (uint32_t block = threadIdx.x; block < block_count; block += blockDim.x) {
        const float *values = vectors.values + (size_t)blockIdx.x * vectors.size + block * KQ_ACTIVATION_BLOCK;
        const size_t vector_block = (size_t)blockIdx.x * block_count + block;
        float largest = 0.0f;
        for (uint32_t offset = 0; offset < KQ_ACTIVATION_BLOCK; offset++) {
            largest = fmaxf(largest, fabsf(values[offset]));
        }
        const uint32_t ratio_bits = __float_as_uint(fmaxf(largest, KQ_AMAX_FLOOR) * KQ_E4M3_MAXIMUM_INVERSE);
        const uint32_t rounded_up = (ratio_bits & KQ_FP32_MANTISSA_MASK) != 0u ? 1u : 0u;
        const uint32_t scale_field = inputs == KQ_INPUTS_RAW ? (uint32_t)KQ_FP32_EXPONENT_BIAS
                                     : (ratio_bits >> KQ_FP32_EXPONENT_SHIFT) + rounded_up;
        const float inverse_scale = kq_power_of_two(KQ_FP32_EXPONENT_BIAS + KQ_E8M0_BIAS - scale_field);
        const float carried_scale = kq_power_of_two(scale_field + KQ_CARRIED_EXPONENT);
        for (uint32_t step = 0; step < KQ_PAIR_GROUPS; step++) {
            float group_sum = 0.0f;
            for (uint32_t position = 0; position < KQ_LANES_PER_GROUP; position++) {
                const uint32_t word_index = kq_word_index(block, position, step);
                uint32_t word = 0, carried[KQ_REGISTER_HALVES] = {0, 0};
                for (uint32_t byte = 0; byte < KQ_WORD_BYTES; byte++) {
                    float quantized = values[kq_value_offset(step, position, byte)];
                    if (inputs != KQ_INPUTS_RAW) {
                        word |= kq_quantize_to_e4m3(quantized, inverse_scale, &quantized) << (byte * KQ_BYTE_BITS);
                    }
                    group_sum += quantized;
                    carried[byte % KQ_REGISTER_HALVES] |= (__float_as_uint(quantized * carried_scale) >> KQ_HALF_BITS)
                                                          << (byte / KQ_REGISTER_HALVES * KQ_HALF_BITS);
                }
                if (inputs != KQ_INPUTS_RAW) { vector_words[word_index] = word; }
                if (inputs != KQ_INPUTS_E4M3) {
                    const int placed = inputs == KQ_INPUTS_PLACED16;
                    uint32_t *pair = &vector_pairs[word_index * KQ_REGISTER_HALVES];
                    pair[0] = placed ? kq_place_e4m3_pair(word) : carried[0];
                    pair[1] = placed ? kq_place_e4m3_pair(word >> KQ_BYTE_BITS) : carried[1];
                }
            }
            if (kquant == KQ_Q2_K) {
                vectors.group_sums[vector_block * KQ_PAIR_GROUPS + step] = group_sum * kq_power_of_two(scale_field);
            }
        }
        if (inputs != KQ_INPUTS_RAW) { vectors.exponents[vector_block] = (uint8_t)scale_field; }
    }
}

/* ---- the instruction and its operands ---- */
#define KQ_K16_FLOAT_INSTRUCTION(types)                                                                              \
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32." types ".f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "      \
                 "{%10,%11,%12,%13};\n"                                                                              \
                 : "=f"(sums[0]), "=f"(sums[1]), "=f"(sums[2]), "=f"(sums[3])                                        \
                 : "r"(weights[0]), "r"(weights[1]), "r"(weights[2]), "r"(weights[3]), "r"(low), "r"(high),          \
                   "f"(sums[0]), "f"(sums[1]), "f"(sums[2]), "f"(sums[3]))
/* sums: outputs g (elements 0, 1) and g + 8 for the lane's two columns; weights: the header's four registers. */
template <int operand>
__device__ __forceinline__ void kq_tensor_core(float *sums, const uint32_t *weights, uint32_t low, uint32_t high) {
    if (operand == KQ_OPERAND_CARRIED) {
        KQ_K16_FLOAT_INSTRUCTION("bf16.bf16");
    } else {
        KQ_K16_FLOAT_INSTRUCTION("f16.f16");
    }
}
#if KQ_HAS_FP8
/* The same sums from bytes: output g's and output g + 8's four codes and the fed column's four, all as E4M3. */
__device__ __forceinline__ void kq_tensor_core_fp8(float *sums, uint32_t first_output, uint32_t second_output,
                                                   uint32_t fed) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
                 : "=f"(sums[0]), "=f"(sums[1]), "=f"(sums[2]), "=f"(sums[3])
                 : "r"(first_output), "r"(second_output), "r"(fed), "f"(sums[0]), "f"(sums[1]), "f"(sums[2]),
                   "f"(sums[3]));
}
#endif
/* The four codes one output has in an instruction's bits of a code word (registers r and r + 2, both halves: bits
 * 0, 4, 16, 20) as four bytes, inputs 2t, 2t + 8, 2t + 1, 2t + 9: the order of a vector's word. kq_sign_bytes: its
 * four sign flags (bits 0, 2, 16, 18) likewise, a 1 per negative code. */
__device__ __forceinline__ uint32_t kq_code_bytes(uint32_t codes) {
    const uint32_t fields = codes & KQ_OUTPUT_CODES_MASK;
    return (fields | (fields << KQ_NIBBLE_BITS)) & KQ_BYTE_CODES_MASK;
}
__device__ __forceinline__ uint32_t kq_sign_bytes(uint32_t signs) {
    const uint32_t flags = signs & KQ_OUTPUT_SIGNS_MASK;
    return (flags | (flags << KQ_SIGN_SPREAD_SHIFT)) & KQ_BYTE_SIGNS_MASK;
}

/* ---- a lane's stored words of a half block: held, fetched, or copied to its slot ---- */
#if KQ_ASYNC_COPIES
#define KQ_ASK_COPY(byte_count, slot, source)                                                                        \
    asm volatile("{ .reg .u64 slot_address;\ncvta.to.shared.u64 slot_address, %0;\n"                                \
                 "cp.async.ca.shared.global [slot_address], [%1], " byte_count "; }\n"                              \
                 :: "l"((uint64_t)(slot)), "l"((uint64_t)(source)))
/* A half block's codes and scales (16 bytes each) and, with_signs, its signs (8) asked into a lane's slot as one
 * group of copies; kq_wait_for_copies returns when every group but the `left` (0 or 1) last ones has arrived. */
__device__ __forceinline__ void kq_copy_ahead(uint32_t *slot, const struct kq_half *from, int with_signs) {
    KQ_ASK_COPY("16", slot, from->codes);
    KQ_ASK_COPY("16", slot + KQ_SCALE_WORDS_AT, from->scales);
    if (with_signs) { KQ_ASK_COPY("8", slot + KQ_SIGN_WORDS_AT, from->signs); }
    asm volatile("cp.async.commit_group;\n" ::);
}
__device__ __forceinline__ void kq_wait_for_copies(int left) {
    if (left) {
        asm volatile("cp.async.wait_group 1;\n" ::);
    } else {
        asm volatile("cp.async.wait_group 0;\n" ::);
    }
}
#else
__device__ __forceinline__ void kq_copy_ahead(uint32_t *, const struct kq_half *, int) {}
__device__ __forceinline__ void kq_wait_for_copies(int) {}
#endif
/* Half block `vector_half` of the lane's slice is opened: its words loaded (registers), asked into stage `stage` of
 * the lane's slot (shared), or only located (fetch; with ahead its first word of each part is read now). */
template <int kquant, int hold, int ahead>
__device__ __forceinline__ void kq_open_half(const struct kq_lane *lane, uint32_t vector_half, uint32_t stage,
                                             struct kq_half *opened) {
    const uint8_t *record = lane->records + (size_t)(vector_half / KQ_BLOCK_HALVES) * KQ_RECORD_BYTES(kquant);
    const uint32_t half_lane = vector_half % KQ_BLOCK_HALVES * KQ_LANES_PER_WARP + lane->lane;
    const uint32_t half_group = vector_half % KQ_BLOCK_HALVES * KQ_HALF_OUTPUTS + lane->lane / KQ_LANES_PER_GROUP;
    uint32_t *slot = lane->slots + (size_t)stage * lane->slot_stage_words;
    opened->codes = (const uint32_t *)record + (size_t)half_lane * KQ_HALF_PAIRS;
    opened->scales = (const uint32_t *)(record + KQ_GROUP_SCALES_AT(kquant)) + (size_t)half_group * KQ_LOAD_WORDS;
    opened->signs = (const uint32_t *)(record + KQ_SIGNS_AT) + (size_t)half_lane * KQ_SIGN_WORDS_PER_HALF;
    if (hold == KQ_HOLD_REGISTERS) {
        const uint4 codes = *(const uint4 *)opened->codes, scales = *(const uint4 *)opened->scales;
        const uint2 signs = kquant == KQ_Q3_K ? *(const uint2 *)opened->signs : make_uint2(0, 0);
        const uint32_t held[KQ_HALF_WORDS] = {codes.x,  codes.y,  codes.z,  codes.w, scales.x,
                                              scales.y, scales.z, scales.w, signs.x, signs.y};
        for (uint32_t word = 0; word < KQ_HALF_WORDS; word++) { opened->held[word] = held[word]; }
    } else if (hold == KQ_HOLD_SHARED) {
        kq_copy_ahead(slot, opened, kquant == KQ_Q3_K);
        opened->codes = slot;
        opened->scales = slot + KQ_SCALE_WORDS_AT;
        opened->signs = slot + KQ_SIGN_WORDS_AT;
    } else if (ahead != 0) {
        opened->held[0] = opened->codes[0];
        opened->held[KQ_SCALE_WORDS_AT] = opened->scales[0];
        opened->held[KQ_SIGN_WORDS_AT] = kquant == KQ_Q3_K ? opened->signs[0] : 0u;
    }
}
/* Word `index` (a constant where this is called) of an opened half block: codes 0 .. 3, scales 4 .. 7, signs 8, 9. */
template <int hold, int ahead>
__device__ __forceinline__ uint32_t kq_half_word(const struct kq_half *opened, uint32_t index) {
    if (hold == KQ_HOLD_REGISTERS || (hold == KQ_HOLD_FETCH && ahead != 0 && index % KQ_LOAD_WORDS == 0)) {
        return opened->held[index];
    }
    if (index >= KQ_SIGN_WORDS_AT) { return opened->signs[index - KQ_SIGN_WORDS_AT]; }
    return index < KQ_SCALE_WORDS_AT ? opened->codes[index] : opened->scales[index - KQ_SCALE_WORDS_AT];
}
/* Where the fed column's words of instruction `in_half` of a half block are: one word of codes, or two of pairs. */
template <int operand>
__device__ __forceinline__ const uint32_t *kq_fed_at(const struct kq_lane *lane, uint32_t vector_half,
                                                     uint32_t in_half) {
    const size_t instruction = (size_t)(vector_half * KQ_LOAD_PAIRS + in_half / KQ_LOAD_GROUPS) * KQ_LANES_PER_GROUP
                               * KQ_LOAD_GROUPS + in_half % KQ_LOAD_GROUPS;
    return lane->fed + instruction * (KQ_OPERAND_IS_PAIRS(operand) ? KQ_REGISTER_HALVES : 1);
}
/* Hold registers: before pair `pair` of a half block the lane takes what it holds of its vectors: per load of four
 * instructions its fed column's codes and (Q2_K) its result columns' X, per pair its fed column's pairs, per half
 * block its result columns' E8M0 codes. */
template <int kquant, int operand>
__device__ __forceinline__ void kq_hold_vectors(const struct kq_lane *lane, struct kq_held *held, uint32_t vector_half,
                                                uint32_t pair) {
    if (KQ_OPERAND_IS_PAIRS(operand) || pair % KQ_LOAD_PAIRS == 0) {
        const uint4 fed = *(const uint4 *)kq_fed_at<operand>(lane, vector_half, pair * KQ_PAIR_GROUPS);
        const uint32_t words[KQ_LOAD_WORDS] = {fed.x, fed.y, fed.z, fed.w};
        for (uint32_t word = 0; word < KQ_LOAD_WORDS; word++) { held->fed[word] = words[word]; }
    }
    for (uint32_t column = 0; column < KQ_LANE_COLUMNS; column++) {
        if (kquant == KQ_Q2_K && pair % KQ_LOAD_PAIRS == 0) {
            const float4 sums = *(const float4 *)(lane->group_sums[column] + (size_t)vector_half * KQ_HALF_GROUPS
                                                  + pair * KQ_PAIR_GROUPS);
            const float groups[KQ_LOAD_GROUPS] = {sums.x, sums.y, sums.z, sums.w};
            for (uint32_t group = 0; group < KQ_LOAD_GROUPS; group++) {
                held->group_sums[column][group] = groups[group];
            }
        }
        if (pair == 0 && operand != KQ_OPERAND_CARRIED) {
            const uint8_t *codes = lane->exponents[column] + (size_t)vector_half * KQ_HALF_PAIRS;
            held->exponents[column] = *(const uint32_t *)codes;
        }
    }
}

/* ---- the slice ---- */
/* One instruction of a lane: instruction `in_half` (a constant) of half block `vector_half`, a group of 16 inputs.
 * A weight is its code, 0 .. 3 (Q2_K) or -4 .. 3 (Q3_K: the stored code plus its sign flag is the magnitude), a
 * whole number in the mantissa of a 16-bit subnormal; FP8: the same number as an E4M3 byte. The instruction's sums,
 * exact, go times their output's scale into pair_sums; Q2_K's minimum term into minimum_sums. */
template <int kquant, int hold, int ahead, int operand>
__device__ __forceinline__ void kq_instruction(const struct kq_lane *lane, const struct kq_half *stored,
                                               const struct kq_held *held, uint32_t vector_half, uint32_t in_half,
                                               float *pair_sums, float *minimum_sums) {
    const uint32_t pair = in_half / KQ_PAIR_GROUPS, step = in_half % KQ_PAIR_GROUPS, in_load = in_half % KQ_LOAD_GROUPS;
    const uint32_t codes = kq_half_word<hold, ahead>(stored, pair) >> (step * KQ_STEP_CODE_BITS);
    const uint32_t scales = kq_half_word<hold, ahead>(stored, KQ_SCALE_WORDS_AT + pair) >> (step * KQ_HALF_BITS);
    const uint32_t signs = kquant != KQ_Q3_K ? 0u
                           : kq_half_word<hold, ahead>(stored, KQ_SIGN_WORDS_AT + in_half / KQ_LOAD_GROUPS)
                             >> (in_load * KQ_STEP_SIGN_BITS);
    const uint32_t *fed = hold != KQ_HOLD_REGISTERS ? kq_fed_at<operand>(lane, vector_half, in_half)
                          : held->fed + (KQ_OPERAND_IS_PAIRS(operand) ? step * KQ_REGISTER_HALVES : in_load);
    float sums[KQ_LANE_SUMS] = {0.0f, 0.0f, 0.0f, 0.0f};
    uint32_t weights[KQ_WEIGHT_REGISTERS], low = fed[0];
    uint32_t high = KQ_OPERAND_IS_PAIRS(operand) ? fed[1] : low >> KQ_BYTE_BITS;
    if (operand == KQ_OPERAND_CODES) {           /* four E4M3 bytes become the two fp16 pairs of operand B */
        low = kq_place_e4m3_pair(low);
        high = kq_place_e4m3_pair(high);
    }
    for (uint32_t weight_register = 0; weight_register < KQ_WEIGHT_REGISTERS; weight_register++) {
        const uint32_t magnitudes = (codes >> (weight_register * KQ_CODE_BITS)) & KQ_CODE_PAIR_MASK;
        const uint32_t negative = (signs >> weight_register) & KQ_SIGN_PAIR_MASK;
        weights[weight_register] = (magnitudes + negative) | (negative << KQ_FP16_SIGN_BIT);
    }
#if KQ_HAS_FP8
    if (operand == KQ_OPERAND_FP8) {
        kq_tensor_core_fp8(sums, kq_code_bytes(codes) + kq_sign_bytes(signs) * KQ_FP8_SIGN_AND_ONE,
                           kq_code_bytes(codes >> KQ_CODE_BITS) + kq_sign_bytes(signs >> 1) * KQ_FP8_SIGN_AND_ONE,
                           fed[0]);
    } else
#endif
    {
        kq_tensor_core<operand>(sums, weights, low, high);
    }
    for (uint32_t element = 0; element < KQ_LANE_SUMS; element++) {
        const uint32_t scale_byte = (scales >> (element / KQ_LANE_COLUMNS * KQ_BYTE_BITS)) & KQ_BYTE_MASK;
        const uint32_t column = element % KQ_LANE_COLUMNS;
        const int scale = kquant == KQ_Q2_K ? (int)(scale_byte & KQ_NIBBLE_MASK) : (int)scale_byte - KQ_Q3_K_SCALE_BIAS;
        pair_sums[element] = fmaf(sums[element], (float)scale, pair_sums[element]);
        if (kquant == KQ_Q2_K) {
            const float group_sum = hold == KQ_HOLD_REGISTERS ? held->group_sums[column][in_load]
                                    : lane->group_sums[column][(size_t)vector_half * KQ_HALF_GROUPS + in_half];
            minimum_sums[element] = fmaf(group_sum, (float)(scale_byte >> KQ_NIBBLE_BITS), minimum_sums[element]);
        }
    }
}
/* One pair of groups of a lane: 32 inputs under one activation scale of each column. The pair's sums go times
 * 2^E (one fp32 exponent field with what the operands' placement took) into scale_sums; a carried operand has
 * brought 2^E, and they are added. `pair` is a constant where this is called. */
template <int kquant, int hold, int ahead, int operand>
__device__ __forceinline__ void kq_pair(const struct kq_lane *lane, const struct kq_half *stored, struct kq_held *held,
                                        uint32_t vector_half, uint32_t pair, float *scale_sums, float *minimum_sums) {
    float pair_sums[KQ_LANE_SUMS] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (hold == KQ_HOLD_REGISTERS) { kq_hold_vectors<kquant, operand>(lane, held, vector_half, pair); }
    kq_instruction<kquant, hold, ahead, operand>(lane, stored, held, vector_half, pair * KQ_PAIR_GROUPS, pair_sums,
                                                 minimum_sums);
    kq_instruction<kquant, hold, ahead, operand>(lane, stored, held, vector_half, pair * KQ_PAIR_GROUPS + 1u, pair_sums,
                                                 minimum_sums);
    for (uint32_t element = 0; element < KQ_LANE_SUMS; element++) {
        const uint32_t column = element % KQ_LANE_COLUMNS;
        if (operand == KQ_OPERAND_CARRIED) {
            scale_sums[element] += pair_sums[element];
        } else {
            const uint32_t fetched = lane->exponents[column][(size_t)vector_half * KQ_HALF_PAIRS + pair];
            const uint32_t code = hold != KQ_HOLD_REGISTERS ? fetched
                                  : (held->exponents[column] >> (pair * KQ_BYTE_BITS)) & KQ_BYTE_MASK;
            const uint32_t offset = operand == KQ_OPERAND_FP8 ? KQ_FP8_POWER_OFFSET : KQ_FP16_POWER_OFFSET;
            scale_sums[element] = fmaf(pair_sums[element], kq_power_of_two(code + offset), scale_sums[element]);
        }
    }
}
/* One warp, one tile of 16 outputs of one job: blocks first_block .. first_block + block_count - 1 of the input
 * width, each lane's four sums to partial_sums [thread][element]. */
template <int kquant, int hold, int ahead, int operand>
__device__ __forceinline__ void kq_slice(const struct kq_lane *lane, uint32_t first_block, uint32_t block_count,
                                         float *partial_sums) {
    const uint32_t half_count = block_count * KQ_BLOCK_HALVES, first_half = first_block * KQ_BLOCK_HALVES;
    float totals[KQ_LANE_SUMS] = {0.0f, 0.0f, 0.0f, 0.0f};
    struct kq_half stored, next;
    struct kq_held held;
    if (ahead != 0) { kq_open_half<kquant, hold, ahead>(lane, first_half, 0, &next); }
    for (uint32_t block = 0; block < block_count; block++) {
        const uint16_t *block_scales = (const uint16_t *)(lane->records + KQ_BLOCK_SCALES_AT(kquant)
                                                          + (size_t)(first_block + block) * KQ_RECORD_BYTES(kquant));
        float scale_sums[KQ_LANE_SUMS] = {0.0f, 0.0f, 0.0f, 0.0f};
        float minimum_sums[KQ_LANE_SUMS] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (uint32_t half = 0; half < KQ_BLOCK_HALVES; half++) {
            const uint32_t slice_half = block * KQ_BLOCK_HALVES + half, vector_half = first_half + slice_half;
            const int more = slice_half + 1u < half_count;
            if (ahead == 0) {
                kq_open_half<kquant, hold, ahead>(lane, vector_half, 0, &stored);
            } else {
                stored = next;
                if (more) {
                    kq_open_half<kquant, hold, ahead>(lane, vector_half + 1u, (slice_half + 1u) % KQ_SLOT_STAGES,
                                                      &next);
                }
            }
            if (hold == KQ_HOLD_SHARED) { kq_wait_for_copies(ahead != 0 && more); }
            kq_pair<kquant, hold, ahead, operand>(lane, &stored, &held, vector_half, 0, scale_sums, minimum_sums);
            kq_pair<kquant, hold, ahead, operand>(lane, &stored, &held, vector_half, 1, scale_sums, minimum_sums);
            kq_pair<kquant, hold, ahead, operand>(lane, &stored, &held, vector_half, KQ_LOAD_PAIRS, scale_sums,
                                                  minimum_sums);
            kq_pair<kquant, hold, ahead, operand>(lane, &stored, &held, vector_half, KQ_LOAD_PAIRS + 1, scale_sums,
                                                  minimum_sums);
        }
        for (uint32_t element = 0; element < KQ_LANE_SUMS; element++) {
            const uint32_t first_scale = (lane->lane / KQ_LANES_PER_GROUP + element / KQ_LANE_COLUMNS * KQ_HALF_OUTPUTS)
                                         * KQ_BLOCK_SCALES(kquant);
            const float d = __half2float(__ushort_as_half(block_scales[first_scale]))
                            * (operand == KQ_OPERAND_CARRIED ? KQ_CARRIED_UNDO : 1.0f);
            totals[element] = fmaf(scale_sums[element], d, totals[element]);
            if (kquant == KQ_Q2_K) {
                const float dmin = __half2float(__ushort_as_half(block_scales[first_scale + 1u]));
                totals[element] = fmaf(minimum_sums[element], -dmin, totals[element]);
            }
        }
    }
    for (uint32_t element = 0; element < KQ_LANE_SUMS; element++) {
        partial_sums[threadIdx.x * KQ_LANE_SUMS + element] = totals[element];
    }
}

/* ---- a block of threads: its lanes' places, its slices, and what it stores ---- */
/* The places a lane reads in its slice, from its job: its matrix is the stage's own (down) or its warp's (gate in
 * the first warps, up in the others). */
template <int kquant, int matrices, int operand>
__device__ __forceinline__ void kq_set_lane(const struct kq_launch *launch, const struct kq_job *job, uint32_t *slots,
                                            struct kq_lane *lane) {
    const struct kq_vectors *vectors = &launch->vectors;
    const uint32_t position = threadIdx.x % KQ_LANES_PER_GROUP, size = vectors->size;
    const uint32_t kind = matrices == KQ_DOWN ? (uint32_t)KQ_DOWN_MATRIX
                          : threadIdx.x / KQ_LANES_PER_WARP / KQ_STAGE_SLICES(matrices);
    const size_t matrix = (size_t)job->expert * KQ_MATRIX_KINDS + kind;
    const size_t record = matrix * KQ_MATRIX_RECORDS + (size_t)blockIdx.x * KQ_STAGE_BLOCKS(matrices);
    const size_t fed = job->pair[threadIdx.x % KQ_LANES_PER_WARP / KQ_LANES_PER_GROUP] / vectors->pairs_per_vector;
    lane->lane = threadIdx.x % KQ_LANES_PER_WARP;
    lane->records = launch->stored + record * KQ_RECORD_BYTES(kquant);
    lane->fed = vectors->words + fed * (size / KQ_WORD_BYTES) + position * KQ_LOAD_GROUPS;
    if (KQ_OPERAND_IS_PAIRS(operand)) {
        lane->fed = vectors->pairs + fed * (size / KQ_REGISTER_HALVES) + position * KQ_LOAD_GROUPS * KQ_REGISTER_HALVES;
    }
    lane->minimum_operand = vectors->minimum_operand
                            + (fed * (size / KQ_BLOCK_WEIGHTS) * KQ_LANES_PER_GROUP + position) * KQ_REGISTER_HALVES;
    lane->slots = slots + (size_t)threadIdx.x * KQ_SLOT_WORDS;
    lane->slot_stage_words = KQ_STAGE_THREADS(matrices) * KQ_SLOT_WORDS;
    for (uint32_t column = 0; column < KQ_LANE_COLUMNS; column++) {
        const size_t vector = job->pair[position * KQ_LANE_COLUMNS + column] / vectors->pairs_per_vector;
        lane->exponents[column] = vectors->exponents + vector * (size / KQ_ACTIVATION_BLOCK);
        lane->group_sums[column] = vectors->group_sums + vector * (size / KQ_GROUP_WEIGHTS);
        lane->scales[column] = vectors->scales + vector * (size / KQ_BLOCK_WEIGHTS);
    }
}
/* One element of one matrix: the partial sums of its slices (warps first_warp ..), added in slice order from +0.
 * broken_order (the stamp of --break exactness, never timed): the sum of a job of several rows starts at another
 * slice, which depends on their number. */
template <int matrices, int broken_order>
__device__ __forceinline__ float kq_sum_slices(const float *partial_sums, uint32_t first_warp, uint32_t lane,
                                               uint32_t element, uint32_t rows) {
    const uint32_t slices = KQ_STAGE_SLICES(matrices);
    const uint32_t first_slice = broken_order != 0 && rows > 1u ? 1u + rows % (slices - 1u) : 0u;
    float sum = 0.0f;
    for (uint32_t position = 0; position < slices; position++) {
        const uint32_t warp = first_warp + (first_slice + position) % slices;
        sum += partial_sums[(warp * KQ_LANES_PER_WARP + lane) * KQ_LANE_SUMS + element];
    }
    return sum;
}
/* After the block's barrier the lanes of warp 0 add the slices. Two matrices: gate and up, each rounded to bf16 as
 * the reference's product returns it; h = silu(min(gate, L)) * clamp(up, -L, L) * route weight, cast to bf16, goes
 * to hidden_values. One matrix: down, as fp32, to pair_output. */
template <int matrices, int broken_order>
__device__ __forceinline__ void kq_close_block(const struct kq_launch *launch, const struct kq_job *job,
                                               const float *partial_sums) {
    const uint32_t lane = threadIdx.x % KQ_LANES_PER_WARP, rows = job->row_count;
    for (uint32_t element = 0; element < KQ_LANE_SUMS && threadIdx.x < KQ_LANES_PER_WARP; element++) {
        const uint32_t column = lane % KQ_LANES_PER_GROUP * KQ_LANE_COLUMNS + element % KQ_LANE_COLUMNS;
        const uint32_t output = blockIdx.x * KQ_TILE_OUTPUTS + lane / KQ_LANES_PER_GROUP
                                + element / KQ_LANE_COLUMNS * KQ_HALF_OUTPUTS;
        const float value = kq_sum_slices<matrices, broken_order>(partial_sums, 0, lane, element, rows);
        if (column < rows && matrices == KQ_DOWN) {
            launch->pair_output[(size_t)job->pair[column] * KQ_HIDDEN_SIZE + output] = value;
        } else if (column < rows) {
            const float gate_value = fminf(kq_bf16_value(value), KQ_SWIGLU_LIMIT);
            const float up_value = kq_bf16_value(kq_sum_slices<matrices, broken_order>(
                partial_sums, KQ_UP_MATRIX * KQ_STAGE_SLICES(matrices), lane, element, rows));
            const float silu = gate_value / (1.0f + expf(-gate_value));
            const float clamped = fminf(fmaxf(up_value, -KQ_SWIGLU_LIMIT), KQ_SWIGLU_LIMIT);
            launch->hidden_values[(size_t)job->pair[column] * KQ_INNER_SIZE + output] =
                kq_bf16_value(silu * clamped * job->route_weight[column]);
        }
    }
}
/* One product stage of every job, for one thread. */
template <int kquant, int matrices, int hold, int ahead, int operand, int broken_order>
__device__ __forceinline__ void kq_block(const struct kq_launch *launch, float *partial_sums, uint32_t *slots) {
    const struct kq_job *job = &launch->jobs[blockIdx.y];
    struct kq_lane lane;
    kq_set_lane<kquant, matrices, operand>(launch, job, slots, &lane);
    kq_slice<kquant, hold, ahead, operand>(
        &lane, threadIdx.x / KQ_LANES_PER_WARP % KQ_STAGE_SLICES(matrices) * KQ_SLICE_BLOCKS(matrices),
        KQ_SLICE_BLOCKS(matrices), partial_sums);
    __syncthreads();
    kq_close_block<matrices, broken_order>(launch, job, partial_sums);
}
/* A row's six experts, each rounded to bf16 as the reference's product returns it, added in slot order: grid (rows). */
__global__ void kq_row_sum_kernel(const float *pair_output, float *row_output) {
    const float *row_pairs = pair_output + (size_t)blockIdx.x * KQ_EXPERTS_PER_ROW * KQ_HIDDEN_SIZE;
    for (uint32_t column = threadIdx.x; column < KQ_HIDDEN_SIZE; column += blockDim.x) {
        float sum = 0.0f;
        for (uint32_t slot = 0; slot < KQ_EXPERTS_PER_ROW; slot++) {
            sum += kq_bf16_value(row_pairs[(size_t)slot * KQ_HIDDEN_SIZE + column]);
        }
        row_output[(size_t)blockIdx.x * KQ_HIDDEN_SIZE + column] = sum;
    }
}

/* ---- the stamps: a kernel and its launch per choice. A launch bound cannot come from template arguments (i00), so
 * it is a macro: KQ_BOUND(blocks) asks an SM for `blocks` blocks of gate with up, 256 threads each, and for as many
 * threads of down's blocks of 96. The baseline states none, as the earlier probe; the four bounds are the sweep. */
#define KQ_UNBOUND(matrices)
#define KQ_BOUND(blocks, matrices) \
    __launch_bounds__(KQ_STAGE_THREADS(matrices), (blocks) * KQ_STAGE_THREADS(KQ_GATE_UP) / KQ_STAGE_THREADS(matrices))
#define KQ_BOUND_3(matrices) KQ_BOUND(3, matrices)
#define KQ_BOUND_4(matrices) KQ_BOUND(4, matrices)
#define KQ_BOUND_5(matrices) KQ_BOUND(5, matrices)
#define KQ_BOUND_6(matrices) KQ_BOUND(6, matrices)
#define KQ_STAMP(name, bound, matrices, hold, body)                                                               \
    __global__ void bound(matrices) name(struct kq_launch launch) {                                               \
        __shared__ float partial_sums[KQ_SHARED_SUMS(matrices)];                                                  \
        __shared__ uint4 slots[KQ_SHARED_SLOT_LOADS(hold, matrices)];                                             \
        body(&launch, partial_sums, (uint32_t *)slots);                                                           \
    }                                                                                                             \
    inline void launch_##name(const struct kq_launch *launch, uint32_t job_count) {                               \
        name<<<dim3(KQ_STAGE_TILES(matrices), job_count), KQ_STAGE_THREADS(matrices)>>>(*launch);                 \
    }
#define KQ_COMMA ,
#define KQ_EXACT(type, kquant, tag, bound, hold, ahead, operand, order)                                           \
    KQ_STAMP(kq_##type##_gate_up_##tag, bound, KQ_GATE_UP, hold,                                                  \
             kq_block<kquant KQ_COMMA KQ_GATE_UP KQ_COMMA hold KQ_COMMA ahead KQ_COMMA operand KQ_COMMA order>)   \
    KQ_STAMP(kq_##type##_down_##tag, bound, KQ_DOWN, hold,                                                        \
             kq_block<kquant KQ_COMMA KQ_DOWN KQ_COMMA hold KQ_COMMA ahead KQ_COMMA operand KQ_COMMA order>)
#define KQ_EXACT_BOTH(tag, bound, hold, ahead, operand)          \
    KQ_EXACT(q2_k, KQ_Q2_K, tag, bound, hold, ahead, operand, 0) \
    KQ_EXACT(q3_k, KQ_Q3_K, tag, bound, hold, ahead, operand, 0)
#define KQ_SET_OF(type, tag) {launch_kq_##type##_gate_up_##tag, launch_kq_##type##_down_##tag}
#define KQ_NAMES_OF(type, tag) "kq_" #type "_gate_up_" #tag "+kq_" #type "_down_" #tag
#define KQ_BOTH_TYPES(tag) \
    {{KQ_SET_OF(q2_k, tag), KQ_SET_OF(q3_k, tag)}, {KQ_NAMES_OF(q2_k, tag), KQ_NAMES_OF(q3_k, tag)}}
#define KQ_NO_KERNELS {{{NULL, NULL}, {NULL, NULL}}, {NULL, NULL}}
/* The eight combinations of round 1 are compiled at one bound, with fetch as the other hold: a second batch takes
 * the bound its first showed best from -DKQ_COMBINED_BLOCKS=4 (ROUND_FLAGS_<architecture>); another hold is an edit. */
#ifndef KQ_COMBINED_BLOCKS
#define KQ_COMBINED_BLOCKS 5
#endif
#define KQ_BOUND_COMBINED(matrices) KQ_BOUND(KQ_COMBINED_BLOCKS, matrices)
KQ_EXACT_BOTH(baseline, KQ_UNBOUND, KQ_HOLD_REGISTERS, 0, KQ_OPERAND_CODES)
KQ_EXACT(q2_k, KQ_Q2_K, broken_order, KQ_UNBOUND, KQ_HOLD_REGISTERS, 0, KQ_OPERAND_CODES, 1)
KQ_EXACT(q3_k, KQ_Q3_K, broken_order, KQ_UNBOUND, KQ_HOLD_REGISTERS, 0, KQ_OPERAND_CODES, 1)
KQ_EXACT_BOTH(bound_3, KQ_BOUND_3, KQ_HOLD_REGISTERS, 0, KQ_OPERAND_CODES)
KQ_EXACT_BOTH(bound_4, KQ_BOUND_4, KQ_HOLD_REGISTERS, 0, KQ_OPERAND_CODES)
KQ_EXACT_BOTH(bound_5, KQ_BOUND_5, KQ_HOLD_REGISTERS, 0, KQ_OPERAND_CODES)
KQ_EXACT_BOTH(bound_6, KQ_BOUND_6, KQ_HOLD_REGISTERS, 0, KQ_OPERAND_CODES)
KQ_EXACT_BOTH(fetch, KQ_UNBOUND, KQ_HOLD_FETCH, 0, KQ_OPERAND_CODES)
KQ_EXACT_BOTH(ahead, KQ_UNBOUND, KQ_HOLD_REGISTERS, 1, KQ_OPERAND_CODES)
KQ_EXACT_BOTH(placed16, KQ_UNBOUND, KQ_HOLD_REGISTERS, 0, KQ_OPERAND_PLACED16)
KQ_EXACT_BOTH(carried, KQ_UNBOUND, KQ_HOLD_REGISTERS, 0, KQ_OPERAND_CARRIED)
KQ_EXACT_BOTH(held_ahead, KQ_BOUND_COMBINED, KQ_HOLD_REGISTERS, 1, KQ_OPERAND_CODES)
KQ_EXACT_BOTH(held_carried, KQ_BOUND_COMBINED, KQ_HOLD_REGISTERS, 0, KQ_OPERAND_CARRIED)
KQ_EXACT_BOTH(held_ahead_carried, KQ_BOUND_COMBINED, KQ_HOLD_REGISTERS, 1, KQ_OPERAND_CARRIED)
KQ_EXACT_BOTH(fetched, KQ_BOUND_COMBINED, KQ_HOLD_FETCH, 0, KQ_OPERAND_CODES)
KQ_EXACT_BOTH(fetched_ahead, KQ_BOUND_COMBINED, KQ_HOLD_FETCH, 1, KQ_OPERAND_CODES)
KQ_EXACT_BOTH(fetched_carried, KQ_BOUND_COMBINED, KQ_HOLD_FETCH, 0, KQ_OPERAND_CARRIED)
KQ_EXACT_BOTH(fetched_ahead_carried, KQ_BOUND_COMBINED, KQ_HOLD_FETCH, 1, KQ_OPERAND_CARRIED)
#if KQ_ASYNC_COPIES
KQ_EXACT_BOTH(shared, KQ_UNBOUND, KQ_HOLD_SHARED, 0, KQ_OPERAND_CODES)
#define KQ_SHARED_KERNELS KQ_BOTH_TYPES(shared)
#else
#define KQ_SHARED_KERNELS KQ_NO_KERNELS
#endif
#if KQ_HAS_FP8
KQ_EXACT_BOTH(fp8, KQ_UNBOUND, KQ_HOLD_REGISTERS, 0, KQ_OPERAND_FP8)
#define KQ_FP8_KERNELS KQ_BOTH_TYPES(fp8)
#else
#define KQ_FP8_KERNELS KQ_NO_KERNELS
#endif
#endif /* KQUANT_EXACT_KERNELS_CUH */
