/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * cross_class_math.cu: the float32 functions the engine's kernels call, over every input bit pattern, hashed per
 * chunk, so that a card of class 80 and one of class 120 can be compared bit for bit (cross_class_math.c). Device
 * code only, built once per architecture with the engine's kernel flags (--fmad=false, no fast math).
 */
#include <stdint.h>

#define FUNCTION_EXP 0u
#define FUNCTION_LOG 1u
#define FUNCTION_LOG1P 2u
#define FUNCTION_SIN 3u
#define FUNCTION_COS 4u
#define FUNCTION_RSQRT 5u
#define FUNCTION_SQRT 6u
#define FUNCTION_RECIPROCAL 7u
#define HASH_PRIME 1099511628211ull
#define HASH_OFFSET 14695981039346656037ull
#define THREADS 256u

__device__ static __forceinline__ float cross_value(uint32_t function, float input)
{
    switch (function) {
    case FUNCTION_EXP: return expf(input);
    case FUNCTION_LOG: return logf(input);
    case FUNCTION_LOG1P: return log1pf(input);
    case FUNCTION_SIN: return sinf(input);
    case FUNCTION_COS: return cosf(input);
    case FUNCTION_RSQRT: return rsqrtf(input);
    case FUNCTION_SQRT: return sqrtf(input);
    default: return 1.0f / input;
    }
}

/* grid (chunk_inputs / THREADS / inputs_per_thread), THREADS threads: hashes[block] covers its inputs from first. */
extern "C" __global__ void cross_class_hash(uint32_t function, uint32_t first, uint32_t inputs_per_thread,
                                            unsigned long long *hashes)
{
    __shared__ unsigned long long thread_hashes[THREADS];
    const uint32_t base = first + (blockIdx.x * THREADS + threadIdx.x) * inputs_per_thread;
    unsigned long long hash = HASH_OFFSET;
    for (uint32_t index = 0; index < inputs_per_thread; index++) {
        const uint32_t bits = __float_as_uint(cross_value(function, __uint_as_float(base + index)));
        hash = (hash ^ bits) * HASH_PRIME;
    }
    thread_hashes[threadIdx.x] = hash;
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned long long block_hash = HASH_OFFSET;
        for (uint32_t thread = 0; thread < THREADS; thread++) {
            block_hash = (block_hash ^ thread_hashes[thread]) * HASH_PRIME;
        }
        hashes[blockIdx.x] = block_hash;
    }
}

/* grid (count / THREADS), THREADS threads: the values of count inputs from first, for the chunks that differ. */
extern "C" __global__ void cross_class_values(uint32_t function, uint32_t first, uint32_t *values)
{
    const uint32_t index = blockIdx.x * THREADS + threadIdx.x;
    values[index] = __float_as_uint(cross_value(function, __uint_as_float(first + index)));
}

/* ---- the tensor-core instructions of the engine's products and attention ---- */

#define MMA_ITERATIONS 64u
#define MIXING_MULTIPLIER 0x9E3779B97F4A7C15ull
#define EXPONENT_SPREAD_MASK 0x0F00u            /* of a 16-bit operand: spread exponents, so sums round */
#define FORM_BFLOAT16 0u

/* A pseudo-random 32-bit word of (seed, index): two 16-bit operands with exponents spread over 16 binades. */
__device__ static __forceinline__ uint32_t mma_operand_pair(unsigned long long seed, uint32_t index, uint32_t form)
{
    unsigned long long mixed = (seed + index) * MIXING_MULTIPLIER;
    mixed ^= mixed >> 29;
    mixed *= MIXING_MULTIPLIER;
    const uint32_t word = (uint32_t)(mixed >> 32);
    /* bfloat16: exponent field 120 .. 135 (0x3C00 + spread); FP16: exponent field 8 .. 23 (0x2000 + spread) */
    const uint32_t base = form == FORM_BFLOAT16 ? 0x3C003C00u : 0x20002000u;
    const uint32_t keep = form == FORM_BFLOAT16 ? 0x807F807Fu : 0x83FF83FFu;
    return (word & keep) | base | ((word >> 4) & ((uint32_t)EXPONENT_SPREAD_MASK | ((uint32_t)EXPONENT_SPREAD_MASK << 16)));
}

/* grid any, 32 threads a warp: each warp chains MMA_ITERATIONS instructions on its own operands; hashes[warp]. */
extern "C" __global__ void cross_class_mma(uint32_t form, unsigned long long seed, unsigned long long *hashes)
{
    const uint32_t lane = threadIdx.x % 32u, warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32u;
    float accumulator[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    unsigned long long hash = HASH_OFFSET;
    for (uint32_t iteration = 0; iteration < MMA_ITERATIONS; iteration++) {
        const uint32_t index = ((warp * MMA_ITERATIONS + iteration) * 32u + lane) * 8u;
        uint32_t a[4], b[2];
        for (uint32_t word = 0; word < 4u; word++) {
            a[word] = mma_operand_pair(seed, index + word, form);
        }
        b[0] = mma_operand_pair(seed, index + 4u, form);
        b[1] = mma_operand_pair(seed, index + 5u, form);
        if (form == FORM_BFLOAT16) {
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
                         "{%8, %9}, {%0, %1, %2, %3};"
                         : "+f"(accumulator[0]), "+f"(accumulator[1]), "+f"(accumulator[2]), "+f"(accumulator[3])
                         : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
        } else {
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
                         "{%8, %9}, {%0, %1, %2, %3};"
                         : "+f"(accumulator[0]), "+f"(accumulator[1]), "+f"(accumulator[2]), "+f"(accumulator[3])
                         : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
        }
        for (uint32_t element = 0; element < 4u; element++) {
            hash = (hash ^ __float_as_uint(accumulator[element])) * HASH_PRIME;
        }
        if ((iteration & 7u) == 7u) {           /* restart the chain now and then: single sums are compared too */
            accumulator[0] = accumulator[1] = accumulator[2] = accumulator[3] = 0.0f;
        }
    }
    hash ^= (unsigned long long)lane << 56;
    for (uint32_t distance = 16; distance != 0; distance >>= 1) {
        hash ^= __shfl_xor_sync(0xFFFFFFFFu, hash, (int)distance) * HASH_PRIME;
    }
    if (lane == 0) {
        hashes[warp] = hash;
    }
}
