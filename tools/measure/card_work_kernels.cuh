/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * card_work_kernels.cuh: the `work` kernel of the speed campaign: ONE launch with the instruction mix of the 4-bit
 * expert kernel's loop and nothing else of an expert step, for questions that are about a card and not about a
 * weight format (the noise of a timing, a launch bound, the clock under a duty cycle). Read by pilot_probe.cu and
 * by the probe of package 9; include probe_harness.h before it. Measurement tool: not part of the engine.
 *
 * What one launch does. A grid of blocks of WORK_BLOCK_THREADS threads: 8 warps, two "matrices" of 4 slices each.
 * A warp makes loads_per_warp loads: per lane 16 bytes of 4-bit weights from a cold ring (packed: [warp load][lane],
 * four words, a word an instruction), 16 bytes of E4M3 activation codes of its column's row (inputs: [row][vector
 * load][lane position], warm), the E8M0 codes of its two outputs' scales (scales: [warp load][lane group]) and of
 * its two columns' (exponents: [row][vector load], two bytes). Per load: the weights' nibbles placed as fp16 pairs
 * (the value times 2^-14), the activation bytes placed as fp16 pairs (the value times 2^-8), four tensor-core
 * instructions m16n8k16 with fp32 sums, the second of a group of 32 inputs on the sum of the first, and per group
 * one fused multiply-add by 2^(both scales + 22) into the lane's four running sums: the loop of
 * shared_rows_expert_probe.cu, 156 instructions a load there [h02:sass/probe_shared_rows_expert_probe_sm_80], with
 * one fold of the five words a lane read on top. After the block's barrier the lanes of warp 0 add the four slices
 * of each matrix in slice order from +0 and store, per lane, eight sums (sums: [block][matrix][lane][element]) and
 * one checksum of what the block's eight warps read at that lane (checksums: [block][lane]). With WORK_STEP_BLOCKS
 * blocks and WORK_MOST_LOADS_PER_WARP loads a launch is 207,360 warp loads, 829,440 instructions and 112.8 MB:
 * the counts of one layer's six routed experts at one row.
 *
 * Why the host can hold it to its bits. Every operand is exact: a weight is a whole number of 2^-15 of magnitude at
 * most 12, an activation code a whole number of 2^-17 of at most 960 (work_stored_word keeps the E4M3 exponent
 * field under 8), so the sum of a group's 32 products is a whole number of 2^-32 under 2^19: exact in fp32 in
 * whatever order and precision the unit adds, provided it multiplies subnormal fp16 operands exactly (assumed, as
 * in kquant_expert_probe.cu; the first run of the pilot is its first test on these cards). Everything after it is
 * a correctly rounded fp32 operation in a fixed order (fmaf, then additions): work_host_block does the same.
 *
 * What a launch must satisfy (a kernel checks nothing: work_launch_is_valid does, on the host, before every launch):
 * blocks of exactly WORK_BLOCK_THREADS threads; 1 .. WORK_MOST_LOADS_PER_WARP loads a warp; 1 .. WORK_MOST_ROWS
 * rows; warp loads first_warp_load .. first_warp_load + blocks * 8 * loads_per_warp - 1 inside both ring arrays (a
 * launch does not wrap around the ring); row_count rows of inputs and exponents; a block's 256 sums and 32
 * checksums for every block. No atomics. In the debug build (PROBE_DIAGNOSTICS) a kernel takes a second argument,
 * so its symbol is another: it checks the same bounds itself, stores (what << 32 | block) of a block that failed to
 * one status word and returns before it reads anything, and every thread counts its loads as it makes them and
 * stores the count to a slot of its own when its loop is over (work_diagnostics_are_valid: the record's two arrays).
 *
 * Kernels: work_kernel (no launch bound), work_kernel_bound_256x1 and work_kernel_bound_256x5 (WORK_KERNEL stamps
 * one per bound: the body is one function); work_fill_kernel (the stored words of a ring array). At the end, under
 * ROUND_FORM_<NAME>, the forms build_round_1.sh compiles alone: nothing launches them.
 */
#ifndef CARD_WORK_KERNELS_CUH
#define CARD_WORK_KERNELS_CUH
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

enum {
    WORK_LANES_PER_WARP = 32, WORK_WARPS_PER_BLOCK = 8, WORK_BLOCK_THREADS = WORK_LANES_PER_WARP * WORK_WARPS_PER_BLOCK,
    WORK_SLICES = 4,                             /* warps that share the width of one vector: a quarter each */
    WORK_MATRICES = WORK_WARPS_PER_BLOCK / WORK_SLICES,
    WORK_LANES_PER_LANE_GROUP = 4,               /* lanes that hold one pair of outputs */
    WORK_LANE_GROUPS = WORK_LANES_PER_WARP / WORK_LANES_PER_LANE_GROUP,      /* also the instruction's eight columns */
    WORK_MOST_ROWS = WORK_LANE_GROUPS,
    WORK_STEPS_PER_LOAD = 4,                     /* instructions a 16-byte load feeds: 64 inputs */
    WORK_STEPS_PER_SCALE_GROUP = 2, WORK_SCALE_GROUPS_PER_LOAD = WORK_STEPS_PER_LOAD / WORK_STEPS_PER_SCALE_GROUP,
    WORK_INPUTS_PER_STEP = 16, WORK_HALF_INPUTS = WORK_INPUTS_PER_STEP / 2, WORK_HALF_OUTPUTS = 8,
    WORK_REGISTERS_PER_OPERAND_A = 4, WORK_REGISTERS_PER_OPERAND_B = 2, WORK_HALVES_PER_REGISTER = 2,
    WORK_ELEMENTS_PER_LANE = 4, WORK_COLUMNS_PER_LANE = 2, WORK_OUTPUTS_PER_LANE = 2,
    WORK_VECTOR_LOADS = 80,                      /* a vector of 5,120 inputs */
    WORK_MOST_LOADS_PER_WARP = WORK_VECTOR_LOADS / WORK_SLICES,
    WORK_STEP_BLOCKS = 1296,                     /* with 20 loads a warp: 207,360 warp loads, six experts' at one row */
    WORK_STEP_WARP_LOADS = WORK_STEP_BLOCKS * WORK_WARPS_PER_BLOCK * WORK_MOST_LOADS_PER_WARP,
    WORK_WORDS_PER_LOAD = 4,                     /* of a lane's 16 bytes */
    WORK_WORD_BYTES = 4,
    WORK_PACKED_BYTES_PER_WARP_LOAD = WORK_LANES_PER_WARP * WORK_WORDS_PER_LOAD * WORK_WORD_BYTES,
    WORK_SCALE_BYTES_PER_WARP_LOAD = WORK_LANE_GROUPS * WORK_WORD_BYTES,
    WORK_BYTES_PER_WARP_LOAD = WORK_PACKED_BYTES_PER_WARP_LOAD + WORK_SCALE_BYTES_PER_WARP_LOAD,
    WORK_INPUT_WORDS_PER_ROW = WORK_VECTOR_LOADS * WORK_LANES_PER_LANE_GROUP * WORK_WORDS_PER_LOAD,
    WORK_INPUT_WORDS = WORK_MOST_ROWS * WORK_INPUT_WORDS_PER_ROW,
    WORK_INPUT_EXPONENTS = WORK_MOST_ROWS * WORK_VECTOR_LOADS,
    WORK_SUMS_PER_BLOCK = WORK_MATRICES * WORK_LANES_PER_WARP * WORK_ELEMENTS_PER_LANE,
    WORK_SHARED_BYTES = (WORK_BLOCK_THREADS * WORK_ELEMENTS_PER_LANE + WORK_BLOCK_THREADS) * WORK_WORD_BYTES,
    WORK_LANE_SUMS_PER_LOAD = WORK_LANES_PER_WARP * WORK_ELEMENTS_PER_LANE * WORK_SCALE_GROUPS_PER_LOAD,
    WORK_BOUND_FIVE_BLOCKS = 5,                  /* what the 48 registers of the 4-bit kernel allow on sm_80 */
    WORK_BYTE_BITS = 8, WORK_HALF_BITS = 16, WORK_NIBBLE_BITS = 4, WORK_NIBBLE_MASK = 15, WORK_BYTE_MASK = 0xFF,
    /* a nibble's three magnitude bits at fp16 bits 9 .. 11 are the E2M1 value times 2^-14; its sign goes to bit 15.
     * The nibbles of register r of operand A lie at bits 4r and 4r + 16 of a stored word. */
    WORK_MAGNITUDE_BIT = 9, WORK_SIGN_BIT = 12, WORK_MAGNITUDE_MASK = 0x0E000E00,
    WORK_REGISTER_1_BIT = WORK_NIBBLE_BITS, WORK_REGISTER_2_BIT = 2 * WORK_NIBBLE_BITS,
    WORK_REGISTER_3_BIT = 3 * WORK_NIBBLE_BITS,
    WORK_E2M1_SIGN_BIT = 8, WORK_E2M1_EXPONENT_MASK = 3, WORK_E2M1_IMPLICIT_ONE = 2,     /* sign, two bits, one bit */
    /* ((b & 0x80) << 8) | ((b & 0x7F) << 7) is the fp16 of E4M3 byte b times 2^-8: bytes 0 and 2 of a word at once */
    WORK_E4M3_PAIR_SIGN_MASK = 0x00800080, WORK_E4M3_PAIR_MAGNITUDE_MASK = 0x007F007F, WORK_E4M3_MAGNITUDE_SHIFT = 7,
    WORK_E4M3_EXPONENT_SHIFT = 3, WORK_E4M3_EXPONENT_MASK = 15, WORK_E4M3_MANTISSA_MASK = 7, WORK_E4M3_IMPLICIT_ONE = 8,
    WORK_E4M3_SIGN_BIT = 0x80,
    /* an instruction's sum is the true one times 2^-(14 + 8): the fp32 field of both E8M0 scales and that factor */
    WORK_E8M0_BIAS = 127, WORK_FP32_EXPONENT_BIAS = 127, WORK_FP32_EXPONENT_SHIFT = 23,
    WORK_WEIGHT_PLACEMENT_EXPONENT = 14, WORK_INPUT_PLACEMENT_EXPONENT = 8,
    WORK_PRODUCT_EXPONENT_BIAS = WORK_E8M0_BIAS - WORK_WEIGHT_PLACEMENT_EXPONENT - WORK_INPUT_PLACEMENT_EXPONENT,
    WORK_PRODUCT_UNIT_EXPONENT = 32,             /* the host's units: weights in 2^-15, activations in 2^-17 */
    WORK_PACKED = 0, WORK_SCALES = 1, WORK_INPUTS = 2, WORK_EXPONENTS = 3,       /* the kinds of stored words */
    WORK_MIX_SHIFT_OUTER = 16, WORK_MIX_SHIFT_INNER = 15, WORK_WORD_BITS = 32
};
#define WORK_SIGN_MASK 0x80008000u
#define WORK_MIX_MULTIPLIER_FIRST 0x7feb352du
#define WORK_MIX_MULTIPLIER_SECOND 0x846ca68bu
#define WORK_SEED 0x243F6A88u
#define WORK_SEED_STRIDE 0x61C88647u             /* a seed per kind of stored word */
#define WORK_FOLD_PRIME 0x01000193u
#define WORK_SCALE_WORD_BASE 0x76767676u         /* weight scales: E8M0 codes 118 .. 121 */
#define WORK_SCALE_WORD_SPREAD 0x03030303u
#define WORK_INPUT_WORD_MASK 0xBFBFBFBFu         /* E4M3 codes with exponent field 0 .. 7: magnitudes up to 1.875 */
#define WORK_EXPONENT_WORD_BASE 0x75757575u      /* activation scales: E8M0 codes 117 .. 124 */
#define WORK_EXPONENT_WORD_SPREAD 0x07070707u
#define WORK_FAILED_BLOCK_THREADS 1u             /* what a debug kernel stores of a launch it must not run */
#define WORK_FAILED_BLOCKS 2u
#define WORK_FAILED_LOADS 3u
#define WORK_FAILED_ROWS 4u
#define WORK_FAILED_RING 5u

struct work_arguments {
    const uint4 *ring_packed;                    /* [warp load][lane]: the weights */
    const uint32_t *ring_scales;                 /* [warp load][lane group]: byte 2q + r, group q, output g + 8r */
    const uint4 *input_words;                    /* [row][vector load][lane position]: E4M3 codes */
    const uint16_t *input_exponents;             /* [row][vector load]: byte q is the E8M0 code of group q */
    float *sums;                                 /* [block][matrix][lane][element] */
    uint32_t *checksums;                         /* [block][lane] */
    uint64_t first_warp_load;                    /* of this launch in the ring */
    uint32_t loads_per_warp, row_count;
};
struct work_diagnostics {                        /* debug build only */
    uint64_t *status_word;                       /* 0, or (what << 32 | block) of a block that must not have run */
    uint32_t *thread_loads;                      /* [block][thread]: the loads each thread made */
    uint64_t ring_warp_loads;
    uint32_t block_count;
};
struct work_array_bytes {                        /* what the arrays of a launch hold, as allocated */
    uint64_t ring_packed, ring_scales, input_words, input_exponents, sums, checksums;
};
#if PROBE_DIAGNOSTICS
#define WORK_DIAGNOSTICS_PARAMETER , struct work_diagnostics diagnostics
#define WORK_DIAGNOSTICS_ARGUMENT , diagnostics
#else
#define WORK_DIAGNOSTICS_PARAMETER
#define WORK_DIAGNOSTICS_ARGUMENT
#endif

/* ---- stored words and the checksum: the same definitions on the host and on the card ---- */

__host__ __device__ inline uint32_t work_mix_bits(uint32_t value)
{
    value ^= value >> WORK_MIX_SHIFT_OUTER;
    value *= WORK_MIX_MULTIPLIER_FIRST;
    value ^= value >> WORK_MIX_SHIFT_INNER;
    value *= WORK_MIX_MULTIPLIER_SECOND;
    value ^= value >> WORK_MIX_SHIFT_OUTER;
    return value;
}

/* The word at a place of an array of one kind: eight weight nibbles (every nibble is an E2M1 code), four E8M0 scale
 * codes, four E4M3 activation codes, or two columns' pairs of E8M0 codes (an exponent pair is the low half). */
__host__ __device__ inline uint32_t work_stored_word(uint64_t word_index, uint32_t kind)
{
    uint32_t upper = work_mix_bits((uint32_t)(word_index >> WORK_WORD_BITS));
    uint32_t bits = work_mix_bits(WORK_SEED + kind * WORK_SEED_STRIDE + (uint32_t)word_index + upper);
    if (kind == WORK_SCALES) {
        return WORK_SCALE_WORD_BASE + (bits & WORK_SCALE_WORD_SPREAD);
    }
    if (kind == WORK_INPUTS) {
        return bits & WORK_INPUT_WORD_MASK;
    }
    return kind == WORK_EXPONENTS ? WORK_EXPONENT_WORD_BASE + (bits & WORK_EXPONENT_WORD_SPREAD) : bits;
}

__host__ __device__ inline uint32_t work_rotated(uint32_t word, uint32_t bits)
{
    return (word << bits) | (word >> (WORK_WORD_BITS - bits));
}

/* What a lane read in one load, as one word, and a running checksum with one more word folded in. */
__host__ __device__ inline uint32_t work_load_word(const uint32_t packed[WORK_WORDS_PER_LOAD], uint32_t scale_word)
{
    return packed[0] ^ work_rotated(packed[1], WORK_BYTE_BITS) ^ work_rotated(packed[2], WORK_HALF_BITS)
           ^ work_rotated(packed[3], WORK_HALF_BITS + WORK_BYTE_BITS) ^ scale_word;
}

__host__ __device__ inline uint32_t work_fold(uint32_t checksum, uint32_t word)
{
    return (checksum ^ word) * WORK_FOLD_PRIME;
}

/* Words 0 .. word_count - 1 of a ring array get the stored words of their places: any grid, any block. */
__global__ void work_fill_kernel(uint32_t *words, uint64_t word_count, uint32_t kind)
{
    uint64_t thread_total = (uint64_t)gridDim.x * blockDim.x;
    for (uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; index < word_count; index += thread_total) {
        words[index] = work_stored_word(index, kind);
    }
}

/* ---- the kernel ---- */

__device__ __forceinline__ void work_tensor_core(float accumulator[WORK_ELEMENTS_PER_LANE],
                                                 const uint32_t weights[WORK_REGISTERS_PER_OPERAND_A],
                                                 uint32_t input_low, uint32_t input_high)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(accumulator[0]), "=f"(accumulator[1]), "=f"(accumulator[2]), "=f"(accumulator[3])
        : "r"(weights[0]), "r"(weights[1]), "r"(weights[2]), "r"(weights[3]), "r"(input_low), "r"(input_high),
          "f"(accumulator[0]), "f"(accumulator[1]), "f"(accumulator[2]), "f"(accumulator[3]));
}

/* The eight nibbles of a stored word as the four registers of operand A: register r holds the nibbles at bits 4r
 * (low half) and 4r + 16 (high half), each as the fp16 of its E2M1 value times 2^-14. */
__device__ __forceinline__ void work_place_weights(uint32_t word, uint32_t weights[WORK_REGISTERS_PER_OPERAND_A])
{
    weights[0] = ((word << WORK_MAGNITUDE_BIT) & WORK_MAGNITUDE_MASK) | ((word << WORK_SIGN_BIT) & WORK_SIGN_MASK);
    weights[1] = ((word << (WORK_MAGNITUDE_BIT - WORK_REGISTER_1_BIT)) & WORK_MAGNITUDE_MASK)
                 | ((word << (WORK_SIGN_BIT - WORK_REGISTER_1_BIT)) & WORK_SIGN_MASK);
    weights[2] = ((word << (WORK_MAGNITUDE_BIT - WORK_REGISTER_2_BIT)) & WORK_MAGNITUDE_MASK)
                 | ((word << (WORK_SIGN_BIT - WORK_REGISTER_2_BIT)) & WORK_SIGN_MASK);
    weights[3] = ((word >> (WORK_REGISTER_3_BIT - WORK_MAGNITUDE_BIT)) & WORK_MAGNITUDE_MASK)
                 | ((word << (WORK_SIGN_BIT - WORK_REGISTER_3_BIT)) & WORK_SIGN_MASK);
}

/* One thread of one block. Lane L (g = L / 4, t = L % 4) of a warp holds, of operand A, outputs g and g + 8 at
 * inputs 2t, 2t + 1, 2t + 8, 2t + 9 of the instruction's 16; of operand B column g at the same four inputs; of the
 * result, outputs g and g + 8 for columns 2t and 2t + 1. A column past the rows of the launch repeats row 0, as the
 * unused columns of an expert's job do. The warp's sums go to the block's shared memory as [thread][element]. */
__device__ __forceinline__ void work_block(struct work_arguments arguments, float *partial_sums,
                                           uint32_t *partial_checksums WORK_DIAGNOSTICS_PARAMETER)
{
    float sums[WORK_ELEMENTS_PER_LANE] = {0.0f, 0.0f, 0.0f, 0.0f};
    uint32_t lane = threadIdx.x % WORK_LANES_PER_WARP, warp = threadIdx.x / WORK_LANES_PER_WARP;
    uint32_t lane_group = lane / WORK_LANES_PER_LANE_GROUP, lane_position = lane % WORK_LANES_PER_LANE_GROUP;
    uint32_t even_column = lane_position * WORK_COLUMNS_PER_LANE, odd_column = even_column + 1u, checksum = 0;
    uint64_t fed_row = lane_group < arguments.row_count ? lane_group : 0u;
    uint64_t even_row = even_column < arguments.row_count ? even_column : 0u;
    uint64_t odd_row = odd_column < arguments.row_count ? odd_column : 0u;
    uint64_t warp_load = arguments.first_warp_load
                         + ((uint64_t)blockIdx.x * WORK_WARPS_PER_BLOCK + warp) * arguments.loads_per_warp;
    uint64_t vector_load = (uint64_t)(warp % WORK_SLICES) * arguments.loads_per_warp;
#if PROBE_DIAGNOSTICS
    uint64_t launch_end = arguments.first_warp_load
                          + (uint64_t)gridDim.x * WORK_WARPS_PER_BLOCK * arguments.loads_per_warp;
    uint32_t failed = 0, loads_made = 0;         /* failed: the same in every thread: all of a launch return, or none */
    failed = launch_end > diagnostics.ring_warp_loads ? WORK_FAILED_RING : failed;
    failed = arguments.row_count - 1u >= WORK_MOST_ROWS ? WORK_FAILED_ROWS : failed;
    failed = arguments.loads_per_warp - 1u >= WORK_MOST_LOADS_PER_WARP ? WORK_FAILED_LOADS : failed;
    failed = gridDim.x > diagnostics.block_count ? WORK_FAILED_BLOCKS : failed;
    failed = blockDim.x != WORK_BLOCK_THREADS ? WORK_FAILED_BLOCK_THREADS : failed;
    if (failed != 0u) {
        *diagnostics.status_word = ((uint64_t)failed << WORK_WORD_BITS) | blockIdx.x;      /* one plain store */
        return;
    }
#endif
    const uint4 *lane_weights = arguments.ring_packed + warp_load * WORK_LANES_PER_WARP + lane;
    const uint32_t *lane_scales = arguments.ring_scales + warp_load * WORK_LANE_GROUPS + lane_group;
    const uint4 *lane_inputs =
        arguments.input_words + (fed_row * WORK_VECTOR_LOADS + vector_load) * WORK_LANES_PER_LANE_GROUP + lane_position;
    const uint16_t *even_exponents = arguments.input_exponents + even_row * WORK_VECTOR_LOADS + vector_load;
    const uint16_t *odd_exponents = arguments.input_exponents + odd_row * WORK_VECTOR_LOADS + vector_load;
    for (uint32_t load = 0; load < arguments.loads_per_warp; load++) {
        uint4 packed = lane_weights[(uint64_t)load * WORK_LANES_PER_WARP];
        uint4 fed = lane_inputs[(uint64_t)load * WORK_LANES_PER_LANE_GROUP];
        uint32_t weight_words[WORK_STEPS_PER_LOAD] = {packed.x, packed.y, packed.z, packed.w};
        uint32_t input_words[WORK_STEPS_PER_LOAD] = {fed.x, fed.y, fed.z, fed.w};
        uint32_t weight_exponents = lane_scales[(uint64_t)load * WORK_LANE_GROUPS];
        uint32_t column_exponents[WORK_COLUMNS_PER_LANE] = {even_exponents[load], odd_exponents[load]};
        checksum = work_fold(checksum, work_load_word(weight_words, weight_exponents));
        for (uint32_t group = 0; group < WORK_SCALE_GROUPS_PER_LOAD; group++) {
            float accumulator[WORK_ELEMENTS_PER_LANE] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (uint32_t step = group * WORK_STEPS_PER_SCALE_GROUP; step < (group + 1u) * WORK_STEPS_PER_SCALE_GROUP;
                 step++) {
                uint32_t weights[WORK_REGISTERS_PER_OPERAND_A];
                uint32_t word = input_words[step], odd_bytes = word >> WORK_BYTE_BITS;
                uint32_t input_low = ((word & WORK_E4M3_PAIR_SIGN_MASK) << WORK_BYTE_BITS)
                                     | ((word & WORK_E4M3_PAIR_MAGNITUDE_MASK) << WORK_E4M3_MAGNITUDE_SHIFT);
                uint32_t input_high = ((odd_bytes & WORK_E4M3_PAIR_SIGN_MASK) << WORK_BYTE_BITS)
                                      | ((odd_bytes & WORK_E4M3_PAIR_MAGNITUDE_MASK) << WORK_E4M3_MAGNITUDE_SHIFT);
                work_place_weights(weight_words[step], weights);
                work_tensor_core(accumulator, weights, input_low, input_high);
            }
            for (uint32_t element = 0; element < WORK_ELEMENTS_PER_LANE; element++) {
                uint32_t weight_byte = group * WORK_OUTPUTS_PER_LANE + element / WORK_COLUMNS_PER_LANE;
                uint32_t input_exponents = column_exponents[element % WORK_COLUMNS_PER_LANE];
                uint32_t exponent_sum = ((weight_exponents >> (weight_byte * WORK_BYTE_BITS)) & WORK_BYTE_MASK)
                                        + ((input_exponents >> (group * WORK_BYTE_BITS)) & WORK_BYTE_MASK);
                /* both scales and the placements' 2^22 as one power of two: the stored codes keep it normal */
                float scale = __uint_as_float((exponent_sum - WORK_PRODUCT_EXPONENT_BIAS) << WORK_FP32_EXPONENT_SHIFT);
                sums[element] = fmaf(accumulator[element], scale, sums[element]);
            }
        }
#if PROBE_DIAGNOSTICS
        loads_made++;                            /* counted where a load is made: the host holds it to loads_per_warp */
#endif
    }
#if PROBE_DIAGNOSTICS
    diagnostics.thread_loads[(uint64_t)blockIdx.x * WORK_BLOCK_THREADS + threadIdx.x] = loads_made;
#endif
    for (uint32_t element = 0; element < WORK_ELEMENTS_PER_LANE; element++) {
        partial_sums[threadIdx.x * WORK_ELEMENTS_PER_LANE + element] = sums[element];
    }
    partial_checksums[threadIdx.x] = checksum;
    __syncthreads();
    if (warp == 0u) {                            /* the lanes of warp 0 close the block: slices in order, from +0 */
        float *block_sums = arguments.sums + (uint64_t)blockIdx.x * WORK_SUMS_PER_BLOCK;
        uint32_t folded = 0;
        for (uint32_t value = 0; value < WORK_MATRICES * WORK_ELEMENTS_PER_LANE; value++) {
            uint32_t matrix = value / WORK_ELEMENTS_PER_LANE, element = value % WORK_ELEMENTS_PER_LANE;
            float sum = 0.0f;
            for (uint32_t slice = 0; slice < WORK_SLICES; slice++) {
                uint32_t thread = (matrix * WORK_SLICES + slice) * WORK_LANES_PER_WARP + lane;
                sum += partial_sums[thread * WORK_ELEMENTS_PER_LANE + element];
            }
            block_sums[(matrix * WORK_LANES_PER_WARP + lane) * WORK_ELEMENTS_PER_LANE + element] = sum;
        }
        for (uint32_t source_warp = 0; source_warp < WORK_WARPS_PER_BLOCK; source_warp++) {
            folded = work_fold(folded, partial_checksums[source_warp * WORK_LANES_PER_WARP + lane]);
        }
        arguments.checksums[(uint64_t)blockIdx.x * WORK_LANES_PER_WARP + lane] = folded;
    }
}

/* One kernel per launch bound, the body the same function. A bound is stated for the block the kernel is launched
 * with and is a variant: the baseline states none, as no kernel of this repository does (CONVENTIONS 1.1). */
#define WORK_NO_BOUND
#define WORK_KERNEL(kernel_name, launch_bound)                                                              \
    __global__ void launch_bound kernel_name(struct work_arguments arguments WORK_DIAGNOSTICS_PARAMETER)    \
    {                                                                                                       \
        __shared__ float partial_sums[WORK_BLOCK_THREADS * WORK_ELEMENTS_PER_LANE];                         \
        __shared__ uint32_t partial_checksums[WORK_BLOCK_THREADS];                                          \
        work_block(arguments, partial_sums, partial_checksums WORK_DIAGNOSTICS_ARGUMENT);                   \
    }
WORK_KERNEL(work_kernel, WORK_NO_BOUND)
WORK_KERNEL(work_kernel_bound_256x1, __launch_bounds__(WORK_BLOCK_THREADS, 1))
WORK_KERNEL(work_kernel_bound_256x5, __launch_bounds__(WORK_BLOCK_THREADS, WORK_BOUND_FIVE_BLOCKS))

/* ---- host side: the contract of a launch, and the reference ---- */

/* 1: a launch of block_count blocks of block_threads threads with these arguments stays inside arrays of these
 * sizes and inside the card's limits and the instance's bound (check 10 of CONVENTIONS 5); 0: it printed why not.
 * Every product is formed in 64 bits from values already held to their ranges: nothing wraps. */
inline int work_launch_is_valid(const struct probe_instance *instance, const struct work_arguments *arguments,
                                const struct work_array_bytes *bytes, uint64_t block_count, uint64_t block_threads)
{
    const char *kernel = "work_kernel";
    int valid = 1;
    probe_launch_geometry(&valid, kernel, instance, block_count, 1, 1, block_threads, WORK_SHARED_BYTES);
    probe_launch_value(&valid, kernel, "threads of a block of this kernel", block_threads, WORK_BLOCK_THREADS,
                       WORK_BLOCK_THREADS);
    probe_launch_value(&valid, kernel, "loads of a warp", arguments->loads_per_warp, 1, WORK_MOST_LOADS_PER_WARP);
    probe_launch_value(&valid, kernel, "rows", arguments->row_count, 1, WORK_MOST_ROWS);
    if (valid) {                                 /* the counts below are products of what was just held to a range */
        uint64_t warp_loads = block_count * WORK_WARPS_PER_BLOCK * arguments->loads_per_warp;
        probe_launch_range(&valid, kernel, "ring_packed, in warp loads", arguments->first_warp_load, warp_loads,
                           WORK_PACKED_BYTES_PER_WARP_LOAD, bytes->ring_packed);
        probe_launch_range(&valid, kernel, "ring_scales, in warp loads", arguments->first_warp_load, warp_loads,
                           WORK_SCALE_BYTES_PER_WARP_LOAD, bytes->ring_scales);
        probe_launch_range(&valid, kernel, "input_words, in rows", 0, arguments->row_count,
                           WORK_INPUT_WORDS_PER_ROW * sizeof(uint32_t), bytes->input_words);
        probe_launch_range(&valid, kernel, "input_exponents, in rows", 0, arguments->row_count,
                           WORK_VECTOR_LOADS * sizeof(uint16_t), bytes->input_exponents);
        probe_launch_range(&valid, kernel, "sums, in blocks", 0, block_count, WORK_SUMS_PER_BLOCK * sizeof(float),
                           bytes->sums);
        probe_launch_range(&valid, kernel, "checksums, in blocks", 0, block_count,
                           WORK_LANES_PER_WARP * sizeof(uint32_t), bytes->checksums);
    }
    return valid;
}

#if PROBE_DIAGNOSTICS
/* The record of a debug launch against what its two arrays were allocated with: the status word, and a slot for
 * every thread of the blocks the record admits (the kernel stores to no slot of a launch with more blocks, and to
 * none where it refuses the launch: a record of no blocks needs no slot). 1: it holds; 0: it printed why not. */
inline int work_diagnostics_are_valid(const struct work_diagnostics *diagnostics, uint64_t status_bytes,
                                      uint64_t thread_load_bytes)
{
    const char *kernel = "work_kernel";
    int valid = 1;
    probe_launch_range(&valid, kernel, "status_word", 0, 1, sizeof(uint64_t), status_bytes);
    if (diagnostics->block_count != 0u) {
        probe_launch_range(&valid, kernel, "thread_loads, in blocks", 0, diagnostics->block_count,
                           WORK_BLOCK_THREADS * sizeof(uint32_t), thread_load_bytes);
    }
    return valid;
}
#endif

/* A weight nibble in units of 2^-15 and an activation byte in units of 2^-17, from the formats' fields and not from
 * the card's placement: E2M1 is a sign, two exponent bits and one mantissa bit, E4M3 a sign, four and three; an
 * exponent field of 0 is subnormal. */
inline int64_t work_host_weight_units(uint32_t nibble)
{
    int64_t exponent = (nibble >> 1) & WORK_E2M1_EXPONENT_MASK, mantissa = nibble & 1;
    int64_t magnitude = exponent == 0 ? mantissa : (WORK_E2M1_IMPLICIT_ONE + mantissa) << (exponent - 1);
    return (nibble & WORK_E2M1_SIGN_BIT) != 0 ? -magnitude : magnitude;
}

inline int64_t work_host_input_units(uint32_t byte)
{
    int64_t exponent = (byte >> WORK_E4M3_EXPONENT_SHIFT) & WORK_E4M3_EXPONENT_MASK;
    int64_t mantissa = byte & WORK_E4M3_MANTISSA_MASK;
    int64_t magnitude = exponent == 0 ? mantissa : (WORK_E4M3_IMPLICIT_ONE + mantissa) << (exponent - 1);
    return (byte & WORK_E4M3_SIGN_BIT) != 0 ? -magnitude : magnitude;
}

/* The sum, as a whole number of 2^-32, of the 32 products of group `group` of one load for one output (the output
 * of lane group lane_group in half output_half of the tile) and one column: packed is the warp's load, row_words
 * the column's row at this vector load as [lane position][instruction]. Input k of an instruction is held by lane
 * position (k % 8) / 2: of operand A in register output_half + 2 (k / 8), of operand B in register k / 8, in
 * either as half k % 2 (a nibble at bit 4 register + 16 half of the weight word; byte register + 2 half of the
 * activation word). */
inline int64_t work_host_group_sum(const uint32_t packed[WORK_LANES_PER_WARP][WORK_WORDS_PER_LOAD],
                                   const uint32_t *row_words, uint32_t lane_group, uint32_t output_half, uint32_t group)
{
    int64_t group_sum = 0;
    for (uint32_t product = 0; product < WORK_STEPS_PER_SCALE_GROUP * WORK_INPUTS_PER_STEP; product++) {
        uint32_t step = group * WORK_STEPS_PER_SCALE_GROUP + product / WORK_INPUTS_PER_STEP;
        uint32_t input = product % WORK_INPUTS_PER_STEP, upper_inputs = input / WORK_HALF_INPUTS;
        uint32_t holder = input % WORK_HALF_INPUTS / WORK_HALVES_PER_REGISTER, half = input % WORK_HALVES_PER_REGISTER;
        uint32_t weight_register = output_half + upper_inputs * WORK_OUTPUTS_PER_LANE;
        uint32_t nibble_at = weight_register * WORK_NIBBLE_BITS + half * WORK_HALF_BITS;
        uint32_t byte_at = (upper_inputs + half * WORK_REGISTERS_PER_OPERAND_B) * WORK_BYTE_BITS;
        uint32_t weight_word = packed[lane_group * WORK_LANES_PER_LANE_GROUP + holder][step];
        uint32_t input_word = row_words[holder * WORK_WORDS_PER_LOAD + step];
        group_sum += work_host_weight_units((weight_word >> nibble_at) & WORK_NIBBLE_MASK)
                     * work_host_input_units((input_word >> byte_at) & WORK_BYTE_MASK);
    }
    return group_sum;
}

/* What one block of a launch must store, from the definitions: block_checksums [lane], and if block_sums is not
 * NULL block_sums [matrix][lane][element]. The ring's words come from work_stored_word, the inputs and exponents
 * from the host's copies of what was uploaded ([row][vector load][lane position][instruction] words;
 * [row][vector load] pairs). A group's 32 products are summed as whole numbers; from there the card's operations
 * in the card's order: one fmaf per group, the slices in order from +0. */
inline void work_host_block(const uint32_t *input_words, const uint16_t *input_exponents,
                            const struct work_arguments *arguments, uint64_t block, float *block_sums,
                            uint32_t *block_checksums)
{
    static float partial_sums[WORK_BLOCK_THREADS][WORK_ELEMENTS_PER_LANE];
    static uint32_t partial_checksums[WORK_BLOCK_THREADS];
    memset(partial_sums, 0, sizeof(partial_sums));
    memset(partial_checksums, 0, sizeof(partial_checksums));
    for (uint64_t warp_step = 0; warp_step < (uint64_t)WORK_WARPS_PER_BLOCK * arguments->loads_per_warp; warp_step++) {
        uint32_t warp = (uint32_t)(warp_step / arguments->loads_per_warp);
        uint64_t load = warp_step % arguments->loads_per_warp;
        uint64_t warp_load = arguments->first_warp_load
                             + (block * WORK_WARPS_PER_BLOCK + warp) * arguments->loads_per_warp + load;
        uint64_t vector_load = (uint64_t)(warp % WORK_SLICES) * arguments->loads_per_warp + load;
        uint32_t packed[WORK_LANES_PER_WARP][WORK_WORDS_PER_LOAD], scales[WORK_LANE_GROUPS];
        for (uint32_t word = 0; word < WORK_LANES_PER_WARP * WORK_WORDS_PER_LOAD; word++) {
            packed[word / WORK_WORDS_PER_LOAD][word % WORK_WORDS_PER_LOAD] =
                work_stored_word(warp_load * WORK_LANES_PER_WARP * WORK_WORDS_PER_LOAD + word, WORK_PACKED);
        }
        for (uint32_t lane_group = 0; lane_group < WORK_LANE_GROUPS; lane_group++) {
            scales[lane_group] = work_stored_word(warp_load * WORK_LANE_GROUPS + lane_group, WORK_SCALES);
        }
        for (uint32_t lane = 0; lane < WORK_LANES_PER_WARP; lane++) {
            uint32_t thread = warp * WORK_LANES_PER_WARP + lane;
            uint32_t load_word = work_load_word(packed[lane], scales[lane / WORK_LANES_PER_LANE_GROUP]);
            partial_checksums[thread] = work_fold(partial_checksums[thread], load_word);
        }
        for (uint32_t value = 0; block_sums != NULL && value < WORK_LANE_SUMS_PER_LOAD; value++) {
            uint32_t group = value % WORK_SCALE_GROUPS_PER_LOAD;         /* every lane, element and group of the load */
            uint32_t element = value / WORK_SCALE_GROUPS_PER_LOAD % WORK_ELEMENTS_PER_LANE;
            uint32_t lane = value / (WORK_SCALE_GROUPS_PER_LOAD * WORK_ELEMENTS_PER_LANE);
            uint32_t lane_group = lane / WORK_LANES_PER_LANE_GROUP, output_half = element / WORK_COLUMNS_PER_LANE;
            uint32_t lane_position = lane % WORK_LANES_PER_LANE_GROUP;
            uint32_t column = lane_position * WORK_COLUMNS_PER_LANE + element % WORK_COLUMNS_PER_LANE;
            uint64_t row = column < arguments->row_count ? column : 0u;
            uint64_t row_load = row * WORK_VECTOR_LOADS + vector_load;
            uint32_t weight_byte = group * WORK_OUTPUTS_PER_LANE + output_half;
            uint32_t weight_code = scales[lane_group] >> (weight_byte * WORK_BYTE_BITS);
            uint32_t input_code = (uint32_t)input_exponents[row_load] >> (group * WORK_BYTE_BITS);
            int power = (int)((weight_code & WORK_BYTE_MASK) + (input_code & WORK_BYTE_MASK));
            int64_t group_sum =
                work_host_group_sum(packed, input_words + row_load * WORK_LANES_PER_LANE_GROUP * WORK_WORDS_PER_LOAD,
                                    lane_group, output_half, group);
            float accumulator = (float)ldexp((double)group_sum, -WORK_PRODUCT_UNIT_EXPONENT);
            float scale = (float)ldexp(1.0, power - WORK_PRODUCT_EXPONENT_BIAS - WORK_FP32_EXPONENT_BIAS);
            float *sum = &partial_sums[warp * WORK_LANES_PER_WARP + lane][element];
            *sum = fmaf(accumulator, scale, *sum);
        }
    }
    for (uint32_t lane = 0; lane < WORK_LANES_PER_WARP; lane++) {
        block_checksums[lane] = 0;
        for (uint32_t warp = 0; warp < WORK_WARPS_PER_BLOCK; warp++) {
            uint32_t warp_checksum = partial_checksums[warp * WORK_LANES_PER_WARP + lane];
            block_checksums[lane] = work_fold(block_checksums[lane], warp_checksum);
        }
        for (uint32_t value = 0; block_sums != NULL && value < WORK_MATRICES * WORK_ELEMENTS_PER_LANE; value++) {
            uint32_t matrix = value / WORK_ELEMENTS_PER_LANE, element = value % WORK_ELEMENTS_PER_LANE;
            float sum = 0.0f;
            for (uint32_t slice = 0; slice < WORK_SLICES; slice++) {
                sum += partial_sums[(matrix * WORK_SLICES + slice) * WORK_LANES_PER_WARP + lane][element];
            }
            block_sums[(matrix * WORK_LANES_PER_WARP + lane) * WORK_ELEMENTS_PER_LANE + element] = sum;
        }
    }
}

/* ---- forms that build_round_1.sh compiles alone, each for both architectures (-cubin -DROUND_FORM_<NAME>):
 * whether the compiler accepts them is the answer; nothing launches them. A lane's operands are words of an array.
 */
#if defined(ROUND_FORM_UNSIGNED_DOT_PRODUCT)     /* the dot product of four unsigned bytes: the trellis' byte sum */
__global__ void round_form_kernel(const uint32_t *operands, uint32_t *results)
{
    const uint32_t *lane = operands + threadIdx.x * WORK_WORDS_PER_LOAD;
    uint32_t result = 0;
    asm volatile("dp4a.u32.u32 %0, %1, %2, %3;\n" : "=r"(result) : "r"(lane[0]), "r"(lane[1]), "r"(lane[2]));
    results[threadIdx.x] = result;
}
#elif defined(ROUND_FORM_FP16_PAIR_TO_E4M3)      /* an fp16 pair rounded to two E4M3 codes: exlr8's rounding */
__global__ void round_form_kernel(const uint32_t *operands, uint32_t *results)
{
    uint32_t result = 0;
    asm volatile("{ .reg .b16 packed;\n"
                 "cvt.rn.satfinite.e4m3x2.f16x2 packed, %1;\n"
                 "cvt.u32.u16 %0, packed; }\n" : "=r"(result) : "r"(operands[threadIdx.x]));
    results[threadIdx.x] = result;
}
#elif defined(ROUND_FORM_TEMPLATE_LAUNCH_BOUND)  /* a launch bound taken from template arguments */
template <int bound_threads, int bound_blocks>
__global__ void __launch_bounds__(bound_threads, bound_blocks) round_form_kernel(const uint32_t *operands,
                                                                                 uint32_t *results)
{
    results[threadIdx.x] = operands[threadIdx.x] + (uint32_t)(bound_threads + bound_blocks);
}
template void round_form_kernel<WORK_BLOCK_THREADS, WORK_BOUND_FIVE_BLOCKS>(const uint32_t *, uint32_t *);
#elif defined(ROUND_FORM_BOUNDED_WORK)           /* the work kernel under a bound given on the command line */
WORK_KERNEL(round_form_kernel, __launch_bounds__(WORK_BLOCK_THREADS, ROUND_FORM_BOUND_BLOCKS))
#endif

#endif /* CARD_WORK_KERNELS_CUH */
