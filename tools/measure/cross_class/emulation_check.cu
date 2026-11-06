/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * emulation_check.cu: the engine's linear_tensor_core (kernel_linear_common.cuh) over random operands, chained: on a
 * card of class 80 the tensor cores, on class 120 in the exact profile the emulation of class 80's sums. The host
 * (cross_class_math.c, CROSS_CLASS_MMA_ONLY with this module) compares the warps' hashes of the two cards.
 */
#define CHECKING_HOOKS_ENABLED 0
#include "kernel_linear_common.cuh"

#define CHECK_ITERATIONS 64u
#define CHECK_MIXING 0x9E3779B97F4A7C15ull
#define CHECK_HASH_PRIME 1099511628211ull
#define CHECK_HASH_OFFSET 14695981039346656037ull

/* A random word of two 16-bit operands: exponents over 16 binades (form 0 bf16 around 2^-120 for A, 2^100 for B, as
 * the carried form; form 1 f16 around 1), one half in 16 zero, one in 16 deep (subnormal). */
__device__ static __forceinline__ uint32_t check_operand_pair(unsigned long long seed, uint32_t index, uint32_t form,
                                                              bool is_a)
{
    unsigned long long mixed = (seed + index) * CHECK_MIXING;
    mixed ^= mixed >> 29;
    mixed *= CHECK_MIXING;
    uint32_t word = 0;
    for (uint32_t half = 0; half < 2u; half++) {
        const uint32_t bits = (uint32_t)(mixed >> (half * 32u));
        const uint32_t sign = bits & 0x8000u, kind = (bits >> 16) & 15u;
        uint32_t value = 0;
        if (form == 0u) {
            const uint32_t field = kind == 1u ? 0u : (is_a ? 120u : 220u) + ((bits >> 20) & 15u) - 8u;
            value = kind == 0u ? 0u : sign | field << 7 | (bits & 0x7Fu);
        } else {
            const uint32_t field = kind == 1u ? 0u : 15u + ((bits >> 20) & 15u) - 8u;
            value = kind == 0u ? 0u : sign | field << 10 | (bits & 0x3FFu);
        }
        word |= value << (half * 16u);
    }
    return word;
}

extern "C" __global__ void emulation_check(uint32_t form, unsigned long long seed, unsigned long long *hashes)
{
    const uint32_t lane = threadIdx.x % 32u, warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32u;
    float accumulator[ELEMENTS_PER_LANE] = {0.0f, 0.0f, 0.0f, 0.0f};
    unsigned long long hash = CHECK_HASH_OFFSET;
    for (uint32_t iteration = 0; iteration < CHECK_ITERATIONS; iteration++) {
        const uint32_t index = ((warp * CHECK_ITERATIONS + iteration) * 32u + lane) * 8u;
        uint32_t weights[REGISTERS_PER_OPERAND_A];
        for (uint32_t word = 0; word < REGISTERS_PER_OPERAND_A; word++) {
            weights[word] = check_operand_pair(seed, index + word, form, true);
        }
        const uint32_t fed_low = check_operand_pair(seed, index + 4u, form, false);
        const uint32_t fed_high = check_operand_pair(seed, index + 5u, form, false);
        if (form == 0u) {
            linear_tensor_core<LINEAR_FORM_CARRIED>(accumulator, weights, fed_low, fed_high);
        } else {
            linear_tensor_core<LINEAR_FORM_CODES>(accumulator, weights, fed_low, fed_high);
        }
        for (uint32_t element = 0; element < ELEMENTS_PER_LANE; element++) {
            hash = (hash ^ __float_as_uint(accumulator[element])) * CHECK_HASH_PRIME;
        }
        if ((iteration & 7u) == 7u) {
            for (uint32_t element = 0; element < ELEMENTS_PER_LANE; element++) {
                accumulator[element] = 0.0f;
            }
        }
    }
    hash ^= (unsigned long long)lane << 56;
    for (uint32_t distance = 16; distance != 0; distance >>= 1) {
        hash ^= __shfl_xor_sync(0xFFFFFFFFu, hash, (int)distance) * CHECK_HASH_PRIME;
    }
    if (lane == 0) {
        hashes[warp] = hash;
    }
}

/* As emulation_check with one column in use, as a product's decode makes it: only column 0's outputs are hashed. */
extern "C" __global__ void emulation_check_one(uint32_t form, unsigned long long seed, unsigned long long *hashes)
{
    const uint32_t lane = threadIdx.x % 32u, warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32u;
    float accumulator[ELEMENTS_PER_LANE] = {0.0f, 0.0f, 0.0f, 0.0f};
    unsigned long long hash = CHECK_HASH_OFFSET;
    for (uint32_t iteration = 0; iteration < CHECK_ITERATIONS; iteration++) {
        const uint32_t index = ((warp * CHECK_ITERATIONS + iteration) * 32u + lane) * 8u;
        uint32_t weights[REGISTERS_PER_OPERAND_A];
        for (uint32_t word = 0; word < REGISTERS_PER_OPERAND_A; word++) {
            weights[word] = check_operand_pair(seed, index + word, form, true);
        }
        const uint32_t fed_low = check_operand_pair(seed, index + 4u, form, false);
        const uint32_t fed_high = check_operand_pair(seed, index + 5u, form, false);
        if (form == 0u) {
            linear_tensor_core_of_columns<LINEAR_FORM_CARRIED>(accumulator, weights, fed_low, fed_high, 1u);
        } else {
            linear_tensor_core_of_columns<LINEAR_FORM_CODES>(accumulator, weights, fed_low, fed_high, 1u);
        }
        if (lane % 4u == 0u) {                    /* lane place 0: column 0 of rows g and g + 8 */
            hash = (hash ^ __float_as_uint(accumulator[LOW_OUTPUT_EVEN_COLUMN])) * CHECK_HASH_PRIME;
            hash = (hash ^ __float_as_uint(accumulator[HIGH_OUTPUT_EVEN_COLUMN])) * CHECK_HASH_PRIME;
        }
        if ((iteration & 7u) == 7u) {
            for (uint32_t element = 0; element < ELEMENTS_PER_LANE; element++) {
                accumulator[element] = 0.0f;
            }
        }
    }
    hash ^= (unsigned long long)lane << 56;
    for (uint32_t distance = 16; distance != 0; distance >>= 1) {
        hash ^= __shfl_xor_sync(0xFFFFFFFFu, hash, (int)distance) * CHECK_HASH_PRIME;
    }
    if (lane == 0) {
        hashes[warp] = hash;
    }
}
