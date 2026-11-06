/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * kquant_integer_kernels.cuh: the K-quant experts under the format's own integer contract, which is what its
 * reference implementation computes on a processor: inputs as 8-bit integers with one fp32 scale per block of 256,
 * integer sums that are exact in any order, floats only where a block ends. Unit 5c priced these inputs on logits
 * at nothing (measurements/i05c_input_precision), so this is a first-class path. It reads the records of
 * kquant_exact_kernels.cuh and shares that file's lanes, its loads one half block ahead, its epilogue and stamps.
 *
 * THE CONTRACT (q an input's integer, x its block's scale, c a code, s a group's scale, less 32 for Q3_K):
 *   0. per block of a vector (kq_prepare_integers_kernel): with m the value of largest magnitude (of equal ones the
 *      first), x = 1 / (-127 / m) and q = min(127, the nearest whole number to (-127 / m) * value, ties to even), all
 *      in fp32; a block of zeros has q = 0 and x = 0. b is a group's sum of q;
 *   1. per block, output and column: I = the sum over its groups of s * (the sum over a group of c * q), an integer:
 *      Q2_K by ONE instruction m16n8k32 u8 by s8 per pair of groups with c * s as the unsigned operand (at most 45 a
 *      byte), the sum running through the block untouched; Q3_K by one instruction m16n8k16 s8 by s8 per group from
 *      zero and one integer multiply-add by its scale;
 *   2. Q2_K: J = the sum over the block's groups of minimum * b, by one instruction: b as its limbs b >> 4 and
 *      b & 15, the minimums once and times 16;
 *   3. per block: T = fma((float)I, d * x, T), then Q2_K T = fma((float)J, -(dmin * x), T), d * x and dmin * x one
 *      fp32 product each, T a slice's; slices added in slice order, as contract C1.
 * Every I and J is below 2^24 in magnitude (256 * 45 * 127, 256 * 128 * 127, 16 * 15 * 2,032): its float is exact.
 * A scale per 32 inputs is not built: unit 5c found it no better (0.0919 and 0.0436 nats against 0.0932 and 0.0433)
 * and it can only cost more, a float step per instruction. Measurement tool: not part of the engine.
 */
#ifndef KQUANT_INTEGER_KERNELS_CUH
#define KQUANT_INTEGER_KERNELS_CUH
#include "kquant_exact_kernels.cuh"

enum {
    KQ_SPAN_BLOCKS = KQ_BLOCK_WEIGHTS / KQ_ACTIVATION_BLOCK,     /* the threads that quantize one block of 256 */
    KQ_INT8_MOST = 127, KQ_LIMB_BITS = 4, KQ_LIMB_MASK = 15,     /* a group's sum b as b >> 4 (signed) and b & 15 */
    KQ_LANE_MINIMUMS = 4,                        /* the groups whose minimums one lane position holds: 4t .. 4t + 3 */
    KQ_LIMB_BYTES = KQ_REGISTER_HALVES * KQ_WORD_BYTES,          /* of a lane position: low limbs, then high limbs */
    KQ_MINIMUM_PAIR_MASK = 0x000F000F, KQ_MINIMUM_BYTES_MASK = 0x0F0F
};
#define KQ_INT8_LARGEST_TO -127.0f               /* the value of largest magnitude becomes -127 whatever its sign */

/* Step 0 of the contract: grid (vectors), exactly size / 32 threads, one per 32 values (the eight threads of a block
 * of 256 are neighbours in one warp and agree on its largest value by three exchanges, whose mask names those eight
 * lanes alone: 2,304 values end in a warp of eight). Writes the integers in the order of a vector's words, the block's
 * scale, and for Q2_K step 2's operand: [block][lane position]: the low limbs of groups 4t .. 4t + 3, then the high. */
template <int kquant>
__global__ void kq_prepare_integers_kernel(struct kq_vectors vectors) {
    const uint32_t part = threadIdx.x, in_block = part % KQ_SPAN_BLOCKS;
    const uint32_t exchanging = ((1u << KQ_SPAN_BLOCKS) - 1u) << (part % KQ_LANES_PER_WARP - in_block);
    const size_t block = (size_t)blockIdx.x * (vectors.size / KQ_BLOCK_WEIGHTS) + part / KQ_SPAN_BLOCKS;
    const float *values = vectors.values + (size_t)blockIdx.x * vectors.size + part * KQ_ACTIVATION_BLOCK;
    uint32_t *vector_words = vectors.words + (size_t)blockIdx.x * (vectors.size / KQ_WORD_BYTES);
    uint8_t *limbs = (uint8_t *)(vectors.minimum_operand + block * KQ_LANES_PER_GROUP * KQ_REGISTER_HALVES);
    float largest = 0.0f;
    for (uint32_t offset = 0; offset < KQ_ACTIVATION_BLOCK; offset++) {
        largest = fabsf(values[offset]) > fabsf(largest) ? values[offset] : largest;
    }
    for (uint32_t distance = 1; distance < KQ_SPAN_BLOCKS; distance += distance) {
        const float other = __shfl_xor_sync(exchanging, largest, (int)distance);
        const int other_is_first = (part & distance) != 0u;                 /* of equal magnitudes the first */
        largest = fabsf(other) > fabsf(largest) || (fabsf(other) == fabsf(largest) && other_is_first) ? other : largest;
    }
    const float inverse_scale = largest == 0.0f ? 0.0f : KQ_INT8_LARGEST_TO / largest;
    for (uint32_t step = 0; step < KQ_PAIR_GROUPS; step++) {
        const uint32_t group = in_block * KQ_PAIR_GROUPS + step;            /* of the block */
        uint8_t *group_limbs = limbs + group / KQ_LANE_MINIMUMS * KQ_LIMB_BYTES + group % KQ_LANE_MINIMUMS;
        int32_t group_sum = 0;
        for (uint32_t position = 0; position < KQ_LANES_PER_GROUP; position++) {
            uint32_t word = 0;
            for (uint32_t byte = 0; byte < KQ_WORD_BYTES; byte++) {
                const int32_t nearest = __float2int_rn(inverse_scale * values[kq_value_offset(step, position, byte)]);
                const int32_t integer = nearest > KQ_INT8_MOST ? (int32_t)KQ_INT8_MOST : nearest;
                word |= ((uint32_t)integer & KQ_BYTE_MASK) << (byte * KQ_BYTE_BITS);
                group_sum += integer;
            }
            vector_words[kq_word_index(part, position, step)] = word;
        }
        if (kquant == KQ_Q2_K) {
            group_limbs[0] = (uint8_t)(group_sum & KQ_LIMB_MASK);
            group_limbs[KQ_WORD_BYTES] = (uint8_t)(group_sum >> KQ_LIMB_BITS);
        }
    }
    if (in_block == 0u) { vectors.scales[block] = largest == 0.0f ? 0.0f : 1.0f / inverse_scale; }
}

/* The two integer instructions. kq_integer_core_32: weights 0 and 1 are outputs g and g + 8 at the first sixteen
 * inputs (four bytes each, the order of a vector's word), 2 and 3 at the second sixteen, unsigned; first_fed and
 * second_fed the fed column's words of the two; the sums run on. kq_integer_core_16: one group, signed weights. */
__device__ __forceinline__ void kq_integer_core_32(int32_t *sums, const uint32_t *weights, uint32_t first_fed,
                                                   uint32_t second_fed) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%10,%11,%12,%13};\n"
                 : "=r"(sums[0]), "=r"(sums[1]), "=r"(sums[2]), "=r"(sums[3])
                 : "r"(weights[0]), "r"(weights[1]), "r"(weights[2]), "r"(weights[3]), "r"(first_fed), "r"(second_fed),
                   "r"(sums[0]), "r"(sums[1]), "r"(sums[2]), "r"(sums[3]));
}
__device__ __forceinline__ void kq_integer_core_16(int32_t *sums, uint32_t first_output, uint32_t second_output,
                                                   uint32_t fed) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
                 : "=r"(sums[0]), "=r"(sums[1]), "=r"(sums[2]), "=r"(sums[3])
                 : "r"(first_output), "r"(second_output), "r"(fed), "r"(sums[0]), "r"(sums[1]), "r"(sums[2]),
                   "r"(sums[3]));
}
/* One pair of groups of a lane (`pair` is a constant where this is called): step 1 of the contract, I running on in
 * `sums`. Register r of the weights is output g + 8 (r % 2) in group r / 2 of the pair: Q2_K's code times scale;
 * Q3_K's code, a negative one c stored as -c - 1 and here its complement. `fed`: the load's four words of the fed
 * column, taken at its first pair. Q2_K: after the lane's own four groups of the block (4t .. 4t + 3) their
 * minimums become operand A of step 2: bytes for outputs g and g + 8, then both times 16. */
template <int kquant>
__device__ __forceinline__ void kq_integer_pair(const struct kq_lane *lane, const struct kq_half *stored, uint32_t *fed,
                                                uint32_t vector_half, uint32_t pair, int32_t *sums,
                                                uint32_t *minimum_weights) {
    const uint32_t in_load = pair % KQ_LOAD_PAIRS * KQ_PAIR_GROUPS;
    const uint32_t lane_groups = (vector_half % KQ_BLOCK_HALVES * KQ_HALF_PAIRS + pair) / KQ_LOAD_PAIRS;
    const uint32_t codes = kq_half_word<KQ_HOLD_REGISTERS, 1>(stored, pair);
    const uint32_t scales = kq_half_word<KQ_HOLD_REGISTERS, 1>(stored, KQ_SCALE_WORDS_AT + pair);
    const uint32_t signs = kquant != KQ_Q3_K ? 0u
                           : kq_half_word<KQ_HOLD_REGISTERS, 1>(stored, KQ_SIGN_WORDS_AT + pair / KQ_LOAD_PAIRS)
                             >> (in_load * KQ_STEP_SIGN_BITS);
    uint32_t weights[KQ_WEIGHT_REGISTERS];
    if (pair % KQ_LOAD_PAIRS == 0) {
        const uint4 load = *(const uint4 *)kq_fed_at<KQ_OPERAND_CODES>(lane, vector_half, pair * KQ_PAIR_GROUPS);
        const uint32_t words[KQ_LOAD_WORDS] = {load.x, load.y, load.z, load.w};
        for (uint32_t word = 0; word < KQ_LOAD_WORDS; word++) { fed[word] = words[word]; }
    }
    for (uint32_t weight_register = 0; weight_register < KQ_WEIGHT_REGISTERS; weight_register++) {
        const uint32_t step = weight_register / KQ_LANE_OUTPUTS, output = weight_register % KQ_LANE_OUTPUTS;
        const uint32_t bytes = kq_code_bytes(codes >> (step * KQ_STEP_CODE_BITS + output * KQ_CODE_BITS));
        const uint32_t scale = (scales >> (step * KQ_HALF_BITS + output * KQ_BYTE_BITS)) & KQ_NIBBLE_MASK;
        const uint32_t negative = kq_sign_bytes(signs >> (step * KQ_STEP_SIGN_BITS + output));
        weights[weight_register] = kquant == KQ_Q2_K ? bytes * scale : bytes ^ (negative * KQ_BYTE_MASK);
    }
    if (kquant == KQ_Q2_K) { kq_integer_core_32(sums, weights, fed[in_load], fed[in_load + 1u]); }
    for (uint32_t step = 0; step < KQ_PAIR_GROUPS && kquant == KQ_Q3_K; step++) {
        int32_t products[KQ_LANE_SUMS] = {0, 0, 0, 0};
        kq_integer_core_16(products, weights[step * KQ_LANE_OUTPUTS], weights[step * KQ_LANE_OUTPUTS + 1u],
                           fed[in_load + step]);
        for (uint32_t element = 0; element < KQ_LANE_SUMS; element++) {
            const uint32_t scale_at = step * KQ_HALF_BITS + element / KQ_LANE_COLUMNS * KQ_BYTE_BITS;
            sums[element] += products[element] * ((int32_t)((scales >> scale_at) & KQ_BYTE_MASK) - KQ_Q3_K_SCALE_BIAS);
        }
    }
    if (kquant == KQ_Q2_K && pair % KQ_LOAD_PAIRS != 0u && lane_groups == lane->lane % KQ_LANES_PER_GROUP) {
        const uint32_t first_pair = kq_half_word<KQ_HOLD_REGISTERS, 1>(stored, KQ_SCALE_WORDS_AT + pair - 1u);
        for (uint32_t output = 0; output < KQ_LANE_OUTPUTS; output++) {
            const uint32_t shift = output * KQ_BYTE_BITS + KQ_NIBBLE_BITS;
            const uint32_t low = (first_pair >> shift) & KQ_MINIMUM_PAIR_MASK;
            const uint32_t high = (scales >> shift) & KQ_MINIMUM_PAIR_MASK;
            minimum_weights[output] = ((low | (low >> KQ_BYTE_BITS)) & KQ_MINIMUM_BYTES_MASK)
                                      | (((high | (high >> KQ_BYTE_BITS)) & KQ_MINIMUM_BYTES_MASK) << KQ_HALF_BITS);
            minimum_weights[output + KQ_LANE_OUTPUTS] = minimum_weights[output] << KQ_LIMB_BITS;
        }
    }
}
/* One warp, one tile of one job: blocks first_block .. first_block + block_count - 1 of the input width, the
 * stored words asked one half block ahead as in kq_slice. */
template <int kquant>
__device__ __forceinline__ void kq_integer_slice(const struct kq_lane *lane, uint32_t first_block, uint32_t block_count,
                                                 float *partial_sums) {
    const uint32_t half_count = block_count * KQ_BLOCK_HALVES, first_half = first_block * KQ_BLOCK_HALVES;
    float totals[KQ_LANE_SUMS] = {0.0f, 0.0f, 0.0f, 0.0f};
    struct kq_half stored, next;
    kq_open_half<kquant, KQ_HOLD_REGISTERS, 1>(lane, first_half, 0, &next);
    for (uint32_t block = first_block; block < first_block + block_count; block++) {
        const uint16_t *block_scales = (const uint16_t *)(lane->records + KQ_BLOCK_SCALES_AT(kquant)
                                                          + (size_t)block * KQ_RECORD_BYTES(kquant));
        int32_t sums[KQ_LANE_SUMS] = {0, 0, 0, 0}, minimums[KQ_LANE_SUMS] = {0, 0, 0, 0};
        uint32_t minimum_weights[KQ_WEIGHT_REGISTERS] = {0, 0, 0, 0}, fed[KQ_LOAD_WORDS] = {0, 0, 0, 0};
        for (uint32_t half = 0; half < KQ_BLOCK_HALVES; half++) {
            const uint32_t vector_half = block * KQ_BLOCK_HALVES + half;
            stored = next;
            if (vector_half + 1u < first_half + half_count) {
                kq_open_half<kquant, KQ_HOLD_REGISTERS, 1>(lane, vector_half + 1u, 0, &next);
            }
            kq_integer_pair<kquant>(lane, &stored, fed, vector_half, 0, sums, minimum_weights);
            kq_integer_pair<kquant>(lane, &stored, fed, vector_half, 1, sums, minimum_weights);
            kq_integer_pair<kquant>(lane, &stored, fed, vector_half, KQ_LOAD_PAIRS, sums, minimum_weights);
            kq_integer_pair<kquant>(lane, &stored, fed, vector_half, KQ_LOAD_PAIRS + 1, sums, minimum_weights);
        }
        if (kquant == KQ_Q2_K) {
            const uint32_t *limbs = lane->minimum_operand + (size_t)block * KQ_LANES_PER_GROUP * KQ_REGISTER_HALVES;
            kq_integer_core_32(minimums, minimum_weights, limbs[0], limbs[1]);
        }
        for (uint32_t element = 0; element < KQ_LANE_SUMS; element++) {
            const uint32_t first_scale = (lane->lane / KQ_LANES_PER_GROUP + element / KQ_LANE_COLUMNS * KQ_HALF_OUTPUTS)
                                         * KQ_BLOCK_SCALES(kquant);
            const float scale = lane->scales[element % KQ_LANE_COLUMNS][block];
            const float d = __half2float(__ushort_as_half(block_scales[first_scale]));
            totals[element] = fmaf((float)sums[element], d * scale, totals[element]);
            if (kquant == KQ_Q2_K) {
                const float dmin = __half2float(__ushort_as_half(block_scales[first_scale + 1u]));
                totals[element] = fmaf((float)minimums[element], -(dmin * scale), totals[element]);
            }
        }
    }
    for (uint32_t element = 0; element < KQ_LANE_SUMS; element++) {
        partial_sums[threadIdx.x * KQ_LANE_SUMS + element] = totals[element];
    }
}
/* One product stage of every job, for one thread. */
template <int kquant, int matrices>
__device__ __forceinline__ void kq_integer_block(const struct kq_launch *launch, float *partial_sums, uint32_t *slots) {
    const struct kq_job *job = &launch->jobs[blockIdx.y];
    struct kq_lane lane;
    kq_set_lane<kquant, matrices, KQ_OPERAND_CODES>(launch, job, slots, &lane);
    kq_integer_slice<kquant>(
        &lane, threadIdx.x / KQ_LANES_PER_WARP % KQ_STAGE_SLICES(matrices) * KQ_SLICE_BLOCKS(matrices),
        KQ_SLICE_BLOCKS(matrices), partial_sums);
    __syncthreads();
    kq_close_block<matrices, 0>(launch, job, partial_sums);
}
#define KQ_INTEGER(type, kquant)                                                        \
    KQ_STAMP(kq_##type##_gate_up_int8, KQ_UNBOUND, KQ_GATE_UP, KQ_HOLD_REGISTERS,       \
             kq_integer_block<kquant KQ_COMMA KQ_GATE_UP>)                              \
    KQ_STAMP(kq_##type##_down_int8, KQ_UNBOUND, KQ_DOWN, KQ_HOLD_REGISTERS, kq_integer_block<kquant KQ_COMMA KQ_DOWN>)
KQ_INTEGER(q2_k, KQ_Q2_K)
KQ_INTEGER(q3_k, KQ_Q3_K)
#endif /* KQUANT_INTEGER_KERNELS_CUH */
