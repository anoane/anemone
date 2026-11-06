/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * kernel_sampler.cu: the token of each row of a step, on the card that samples. Device code only, built once per
 * architecture into the module "kernel_sampler". Contract: include/anemone/sampler_records.h. Here:
 *   - A candidate is one key, the order of its value high (0 for a not-a-number, -0 as +0) and UINT32_MAX less its
 *     token low: the largest key is the first largest value.
 *   - Thread i loads tokens 4 i to 4 i + 3 of each SAMPLER_TURN tokens of a row, in 16 bytes.
 *   - tau in SAMPLER_DIGITS passes of SAMPLER_DIGIT_BITS bits, highest first (e_t <= 1 has bits 30 and 31 clear): a
 *     token whose higher bits agree with tau so far adds its mass to the bin of its digit, two words, the carry of
 *     the low one added to the high one, both by 32-bit integer atomics (64-bit ones would spin). Warp 0 finds the
 *     highest lane whose runs of bins from it up reach the target, which walks its run down.
 *   - The first pass writes e_t to weights; the later passes and the race read it back in the thread that wrote it.
 *   - Every thread of a block reaches each of its barriers: a block leaves early only as a whole.
 *   - The mixer is that of include/anemone/pack_digest.h, its constants repeated word for word.
 */
#include "kernel_sampler.cuh"

/* The mass of a weight, at most 2^32, added into a bin: the carry of each addition is counted once, in any order. */
__device__ static __forceinline__ void sampler_bin_add(uint32_t *low, uint32_t *high, float weight)
{
	const unsigned long long mass =
		__float2ull_rn(weight * SAMPLER_MASS_SCALE);
	const uint32_t before = atomicAdd(low, (uint32_t)mass);
	const uint32_t carried = (uint32_t)(mass >> SAMPLER_WORD_BITS) +
				 (before + (uint32_t)mass < before ? 1u : 0u);
	if (carried != 0) {
		atomicAdd(high, carried);
	}
}

extern "C" __global__ void __launch_bounds__(SAMPLER_THREADS, 1)
	token_sampler(struct sampler_arguments arguments, unsigned char *arena)
{
    __shared__ unsigned long long warp_keys[SAMPLER_WARPS], target, above;
    __shared__ uint32_t low_bins[SAMPLER_BINS], high_bins[SAMPLER_BINS], prefix;
    const struct step_block *const step =
	    (const struct step_block *)(arena + arguments.step_offset);
    const struct step_row_record row =
	    ((const struct step_row_record *)(step + 1))[blockIdx.y];
    struct step_result *const result =
	    (struct step_result *)(arena + arguments.result_offset);
    struct step_row_result *const answer =
	    (struct step_row_result *)(result + 1) + blockIdx.y;
    const uint32_t vocabulary = arguments.vocabulary_size, row_count = min(step->row_count, gridDim.y);
    const float *const logits = (const float *)(arena + arguments.logits_offset) + (size_t)blockIdx.y * vocabulary;
    const float4 *const logit_loads = (const float4 *)logits;
    float4 *const weights = (float4 *)(arena + arguments.weights_offset) +
                            (size_t)blockIdx.y * (vocabulary / SAMPLER_TOKENS_PER_LOAD);
    LINEAR_DEVICE_CHECK(step->row_count <= gridDim.y);
    if (blockIdx.y == 0 && threadIdx.x == 0) {
	    const struct step_result answered = { step->sequence, row_count,
						  0 };
	    *result = answered;
    }
    if (blockIdx.y >= row_count || (row.flags & STEP_ROW_LOGITS) == 0) {
	    if (blockIdx.y < row_count && threadIdx.x == 0) {
		    answer->token_id = STEP_NO_TOKEN;
		    answer->logit = 0.0f;
	    }
	    return;
    }
    LINEAR_DEVICE_CHECK(row.session_slot < arguments.session_capacity && row.position >= 0);
    const struct step_session session =
	    ((const struct step_session
		      *)(arena + arguments.session_offset))[row.session_slot];
    LINEAR_DEVICE_CHECK(isfinite(session.temperature) && session.temperature >= 0.0f && session.top_p > 0.0f &&
                        session.top_p <= 1.0f);
    unsigned long long best = 0;
    for (uint32_t token = threadIdx.x * SAMPLER_TOKENS_PER_LOAD; token < vocabulary; token += SAMPLER_TURN) {
        const float4 four = logit_loads[token / SAMPLER_TOKENS_PER_LOAD];
        const float values[SAMPLER_TOKENS_PER_LOAD] = {four.x, four.y, four.z, four.w};
#pragma unroll
        for (uint32_t place = 0; place < SAMPLER_TOKENS_PER_LOAD; place++) {
            best = max(best, sampler_key(values[place], token + place));
        }
    }
    best = sampler_block_maximum(best, warp_keys);
    const float inverse =
	    1.0f / fmaxf(session.temperature, SAMPLER_TEMPERATURE_FLOOR);
    const float scaled_largest = logits[UINT32_MAX - (uint32_t)best] * inverse;
    if (session.temperature > 0.0f && isfinite(scaled_largest)) {
        const bool cuts = session.top_p < 1.0f;
        if (threadIdx.x == 0) {
            prefix = 0;
        }
        for (uint32_t digit = 0; digit < SAMPLER_DIGITS && (digit == 0 || cuts); digit++) {
            const uint32_t shift = (SAMPLER_DIGITS - 1u - digit) * SAMPLER_DIGIT_BITS;
            low_bins[threadIdx.x] = 0;
            high_bins[threadIdx.x] = 0;
            __syncthreads();
            const uint32_t agreed = prefix;
            for (uint32_t token = threadIdx.x * SAMPLER_TOKENS_PER_LOAD; token < vocabulary; token += SAMPLER_TURN) {
                float4 four = (digit == 0 ? logit_loads : weights)[token / SAMPLER_TOKENS_PER_LOAD];
                if (digit == 0) {
                    four = make_float4(sampler_weight(four.x, scaled_largest, inverse),
                                       sampler_weight(four.y, scaled_largest, inverse),
                                       sampler_weight(four.z, scaled_largest, inverse),
                                       sampler_weight(four.w, scaled_largest, inverse));
                    weights[token / SAMPLER_TOKENS_PER_LOAD] = four;
                }
                const float values[SAMPLER_TOKENS_PER_LOAD] = {four.x, four.y, four.z, four.w};
#pragma unroll
                for (uint32_t place = 0; place < SAMPLER_TOKENS_PER_LOAD; place++) {
                    const uint32_t key = __float_as_uint(values[place]);
		    if (cuts &&
			(key >> shift >> SAMPLER_DIGIT_BITS) == agreed &&
			values[place] * SAMPLER_MASS_SCALE >
				SAMPLER_HALF) { /* a mass of 1 or more */
			    const uint32_t bin = (key >> shift) % SAMPLER_BINS;
			    sampler_bin_add(&low_bins[bin], &high_bins[bin],
					    values[place]);
		    }
		}
            }
            __syncthreads();
            if (cuts && threadIdx.x < THREADS_PER_WARP) {
                const uint32_t lane = threadIdx.x;
                unsigned long long run = 0;
                for (uint32_t bin = 0; bin < SAMPLER_BINS_PER_LANE; bin++) {     /* turned: 32 banks at once */
                    run += sampler_bin(low_bins, high_bins,
                                       lane * SAMPLER_BINS_PER_LANE + (bin + lane) % SAMPLER_BINS_PER_LANE);
                }
                unsigned long long from_lane = run;
                for (uint32_t distance = 1; distance < THREADS_PER_WARP; distance <<= 1u) {
                    const unsigned long long higher = __shfl_down_sync(SAMPLER_EVERY_LANE, from_lane, distance);
                    from_lane += lane + distance < THREADS_PER_WARP ? higher : 0ull;
                }
                const double total = (double)__shfl_sync(SAMPLER_EVERY_LANE, from_lane, 0);
                const unsigned long long goal =
                    digit == 0 ? (unsigned long long)ceil((double)session.top_p * total) : target;
                const unsigned long long before = digit == 0 ? 0ull : above;
                const uint32_t reaching = __ballot_sync(SAMPLER_EVERY_LANE, before + from_lane >= goal);
                LINEAR_DEVICE_CHECK(reaching != 0);
                __syncwarp(SAMPLER_EVERY_LANE);     /* a vote orders no memory: every lane has read target and above */
                if (lane + 1u == (uint32_t)__popc(reaching)) {     /* the lanes that reach are 0 up to this one */
                    uint32_t bin = (lane + 1u) * SAMPLER_BINS_PER_LANE - 1u;
                    unsigned long long mass = before + from_lane - run;
                    for (; mass + sampler_bin(low_bins, high_bins, bin) < goal; bin--) {
                        mass += sampler_bin(low_bins, high_bins, bin);
                    }
                    target = goal;
                    above = mass;
                    prefix = (prefix << SAMPLER_DIGIT_BITS) | bin;
                }
            }
            __syncthreads();
        }
        const uint32_t threshold = cuts ? prefix : 0u;
        const unsigned long long seed_key = sampler_mix(session.seed);
	const unsigned long long position_key =
		(unsigned long long)(uint32_t)row.position
		<< SAMPLER_POSITION_SHIFT;
	best = 0;
        for (uint32_t token = threadIdx.x * SAMPLER_TOKENS_PER_LOAD; token < vocabulary; token += SAMPLER_TURN) {
            const float4 four = weights[token / SAMPLER_TOKENS_PER_LOAD];
            const float values[SAMPLER_TOKENS_PER_LOAD] = {four.x, four.y, four.z, four.w};
#pragma unroll
            for (uint32_t place = 0; place < SAMPLER_TOKENS_PER_LOAD; place++) {
                if (__float_as_uint(values[place]) >= threshold) {
			const unsigned long long hash = sampler_mix(
				seed_key ^ ((position_key | (token + place)) *
					    PACK_DIGEST_POSITION_MULTIPLIER));
			const float uniform =
				((float)(uint32_t)(hash >>
						   SAMPLER_UNIFORM_SHIFT) +
				 SAMPLER_HALF) *
				SAMPLER_UNIFORM_SCALE;
			best = max(best,
				   sampler_key(values[place] / -logf(uniform),
					       token + place));
                }
            }
        }
        best = sampler_block_maximum(best, warp_keys);
    }
    if (threadIdx.x == 0) {
        const uint32_t token = UINT32_MAX - (uint32_t)best;
        answer->token_id = (int32_t)token;
        answer->logit = logits[token];
    }
}
