/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* kernel_sampler.cuh: what kernel_sampler.cu (the sampler) and kernel_sampler_shard.cu (its vocabulary-sharded form,
 * tp4_v3_claude) share word for word: a candidate's key, the keyed draws' mixer, e_t and the block maximum. */
#ifndef KERNEL_SAMPLER_CUH
#define KERNEL_SAMPLER_CUH

#include "kernel_linear_common.cuh"

#include "anemone/sampler_records.h"

#define PACK_DIGEST_POSITION_MULTIPLIER \
	0x9E3779B97F4A7C15ull /* odd: 2^64 divided by the golden ratio */
#define PACK_DIGEST_FIRST_MULTIPLIER \
	0xBB67AE8584CAA73Bull /* odd: the fraction of the square root of 3 */
#define PACK_DIGEST_SECOND_MULTIPLIER \
	0xA54FF53A5F1D36F1ull /* odd: the fraction of the square root of 7 */
#define PACK_DIGEST_FIRST_SHIFT 32u
#define PACK_DIGEST_SECOND_SHIFT 29u
#define PACK_DIGEST_THIRD_SHIFT 32u
#define SAMPLER_WARPS (SAMPLER_THREADS / THREADS_PER_WARP)
#define SAMPLER_TOKENS_PER_LOAD SAMPLER_TOKENS_PER_LOAD_VALUE
#define SAMPLER_TURN (SAMPLER_THREADS * SAMPLER_TOKENS_PER_LOAD)
#define SAMPLER_DIGIT_BITS 10u
#define SAMPLER_BINS 1024u                      /* 2^SAMPLER_DIGIT_BITS: a bin a thread */
#define SAMPLER_BINS_PER_LANE (SAMPLER_BINS / THREADS_PER_WARP)
#define SAMPLER_DIGITS 3u
#define SAMPLER_EVERY_LANE 0xFFFFFFFFu
#define SAMPLER_WORD_BITS 32u                   /* of a key's token, and of a bin's words */
#define SAMPLER_HALF 0.5f

static_assert(SAMPLER_BINS == SAMPLER_THREADS, "each thread clears one bin");
static_assert(sizeof(float4) == SAMPLER_TOKENS_PER_LOAD * sizeof(float) &&
		      LINEAR_OFFSET_ALIGNMENT_BYTES % sizeof(float4) == 0,
	      "a row of a multiple of four tokens begins on 16 bytes");

__device__ static __forceinline__ unsigned long long sampler_mix(unsigned long long value)
{
	value = (value ^ (value >> PACK_DIGEST_FIRST_SHIFT)) *
		PACK_DIGEST_FIRST_MULTIPLIER;
	value = (value ^ (value >> PACK_DIGEST_SECOND_SHIFT)) *
		PACK_DIGEST_SECOND_MULTIPLIER;
	return value ^ (value >> PACK_DIGEST_THIRD_SHIFT);
}

__device__ static __forceinline__ unsigned long long sampler_key(float value, uint32_t token)
{
    const uint32_t bits = __float_as_uint(value == 0.0f ? 0.0f : value);
    const uint32_t order = value != value ? 0u : bits > FLOAT32_MAGNITUDE_MASK ? ~bits : bits | ~FLOAT32_MAGNITUDE_MASK;
    return (unsigned long long)order << SAMPLER_WORD_BITS | (UINT32_MAX - token);
}

/* e_t of a logit, given x_max and the inverse of the temperature. */
__device__ static __forceinline__ float sampler_weight(float logit, float scaled_largest, float inverse)
{
    return fmaxf(expf(logit * inverse - scaled_largest), 0.0f);
}

/* The largest key of the block, in every thread. */
__device__ static unsigned long long sampler_block_maximum(unsigned long long key, unsigned long long *warp_keys)
{
    for (uint32_t distance = THREADS_PER_WARP >> 1u; distance != 0; distance >>= 1u) {
        key = max(key, __shfl_xor_sync(SAMPLER_EVERY_LANE, key, (int)distance));
    }
    __syncthreads();                            /* nobody reads warp_keys of an earlier call any more */
    if (threadIdx.x % THREADS_PER_WARP == 0) {
        warp_keys[threadIdx.x / THREADS_PER_WARP] = key;
    }
    __syncthreads();
    for (uint32_t warp = 0; warp < SAMPLER_WARPS; warp++) {
        key = max(key, warp_keys[warp]);
    }
    return key;
}

__device__ static __forceinline__ unsigned long long sampler_bin(const uint32_t *low, const uint32_t *high,
                                                                  uint32_t bin)
{
    return (unsigned long long)high[bin] << SAMPLER_WORD_BITS | low[bin];
}

#endif
