/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * kernel_sampler_shard.cu (tp4_v3_claude): the sampler over a vocabulary cut into SAMPLER_SHARD_RANKS shards,
 * one on each card of a TP group (include/anemone/sampler_shard_records.h). Device code only, built once per
 * architecture into the module "kernel_sampler_shard". Here:
 *   - Block r answers row r as the sampler's block does, over its shard, with kernel_sampler.cuh's functions and the
 *     global token ids; threads 0 to 3 store this rank's record into card t's slot and then read card t's record of
 *     the row from the local area, each with one 16-byte access; every thread takes the largest of the four keys.
 *   - A block reads its row's session settings from this card's copy, once its word is the step's sequence (the
 *     holder's sampler_settings_push, below, writes every copy at the start of the step).
 *   - Every card writes the same answers; nothing depends on timing but the waits, which end or trap.
 */
#include "kernel_sampler.cuh"

#include "anemone/sampler_shard_records.h"

#define SHARD_RANKS SAMPLER_SHARD_RANKS
#define SHARD_PHASES SAMPLER_SHARD_PHASES
#define SHARD_TAG_LOW_BIT 1u

static_assert(SHARD_RANKS <= THREADS_PER_WARP &&
		      SAMPLER_SHARD_RECORD_BYTES == sizeof(uint4),
	      "a lane a rank; a record one 16-byte access");

__device__ static __forceinline__ void shard_record_store(uint4 *slot, uint4 record)
{
    asm volatile("st.volatile.global.v4.u32 [%0], {%1, %2, %3, %4};\n" ::"l"(slot), "r"(record.x), "r"(record.y),
                 "r"(record.z), "r"(record.w)
                 : "memory");
}

__device__ static __forceinline__ uint4 shard_record_load(const uint4 *slot)
{
    uint4 record;
    asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(record.x), "=r"(record.y), "=r"(record.z), "=r"(record.w)
                 : "l"(slot)
                 : "memory");
    return record;
}

/* The record slot of (parity, phase, source rank, row) in the area at base, for use u. */
__device__ static __forceinline__ uint4 *shard_slot(unsigned char *base, uint64_t use, uint32_t phase, uint32_t source,
                                                     uint32_t row)
{
	const uint32_t parity = (uint32_t)(use % SAMPLER_SHARD_PARITIES);
	return (uint4 *)base +
	       ((parity * SHARD_PHASES + phase) * SHARD_RANKS + source) *
		       LINEAR_MAXIMUM_ROWS +
	       row;
}

/* This rank's (key, logit) of a phase to every card, then the largest key of the four ranks with its logit, in every
 * thread. keys and logits are the block's shared slots of the four ranks. */
__device__ static void
shard_exchange(const struct sampler_shard_arguments *arguments,
	       unsigned char *arena, uint64_t use, uint32_t phase, uint32_t row,
	       unsigned long long key, float logit, unsigned long long *keys,
	       float *logits, unsigned long long *best, float *best_logit)
{
    const uint32_t tag = (uint32_t)(use * SHARD_PHASES + phase) * 2u + SHARD_TAG_LOW_BIT;
    const uint32_t thread = threadIdx.x;
    if (thread < SHARD_RANKS) {
        const uint64_t *const peers = (const uint64_t *)(arena + arguments->peer_table_offset);
        unsigned char *const destination = (unsigned char *)peers[thread];
        shard_record_store(shard_slot(destination, use, phase, arguments->rank, row),
                           make_uint4((uint32_t)key, (uint32_t)(key >> SAMPLER_WORD_BITS), __float_as_uint(logit),
                                      tag));
        const uint4 *const own = shard_slot(arena + arguments->exchange_offset, use, phase, thread, row);
        const long long start = clock64();
        uint4 record = shard_record_load(own);
        while (record.w != tag) {
            if ((unsigned long long)(clock64() - start) > arguments->deadline_cycles)
                asm volatile("trap;");
            record = shard_record_load(own);
        }
        keys[thread] = (unsigned long long)record.y << SAMPLER_WORD_BITS | record.x;
        logits[thread] = __uint_as_float(record.z);
    }
    __syncthreads();
    unsigned long long largest = keys[0];
    float largest_logit = logits[0];
    for (uint32_t rank = 1; rank < SHARD_RANKS; rank++) {
        if (keys[rank] > largest) {
            largest = keys[rank];
            largest_logit = logits[rank];
        }
    }
    *best = largest;
    *best_logit = largest_logit;
    __syncthreads(); /* nobody rewrites the slots before every thread read them */
}

extern "C" __global__ void __launch_bounds__(SAMPLER_THREADS, 1)
	sampler_shard(struct sampler_shard_arguments arguments,
		      unsigned char *arena)
{
    __shared__ unsigned long long warp_keys[SAMPLER_WARPS], rank_keys[SHARD_RANKS];
    __shared__ float rank_logits[SHARD_RANKS];
    const struct step_block *const step =
	    (const struct step_block *)(arena + arguments.step_offset);
    const struct step_row_record row =
	    ((const struct step_row_record *)(step + 1))[blockIdx.y];
    struct step_result *const result =
	    (struct step_result *)(arena + arguments.result_offset);
    struct step_row_result *const answer =
	    (struct step_row_result *)(result + 1) + blockIdx.y;
    const uint32_t shard = arguments.shard_size, first = arguments.first_token;
    const uint32_t row_count = min(step->row_count, gridDim.y);
    const float *const logits = (const float *)(arena + arguments.logits_offset) + (size_t)blockIdx.y * shard;
    const float4 *const logit_loads = (const float4 *)logits;
    LINEAR_DEVICE_CHECK(step->row_count <= gridDim.y &&
			arguments.rank < SHARD_RANKS &&
			arguments.call < SAMPLER_SHARD_CALLS &&
			shard % SAMPLER_TOKENS_PER_LOAD == 0 &&
			first + shard <= arguments.vocabulary_size);
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
    /* the settings: this card's copy, once its word is the step's sequence; read before this rank publishes anything */
    const unsigned char *const copy = arena + arguments.session_offset;
    const long long settings_start = clock64();
    unsigned long long pushed;
    do {
        asm volatile("ld.acquire.sys.global.u64 %0, [%1];\n" : "=l"(pushed) : "l"(copy) : "memory");
        if (pushed != step->sequence && (unsigned long long)(clock64() - settings_start) > arguments.deadline_cycles)
            asm volatile("trap;");
    } while (pushed != step->sequence);
    const uint4 settings = shard_record_load(
	    (const uint4 *)(copy + SAMPLER_SETTINGS_SLOTS_OFFSET) +
	    row.session_slot);
    const struct step_session session = {
	    (uint64_t)settings.y << SAMPLER_WORD_BITS | settings.x,
	    __uint_as_float(settings.z), __uint_as_float(settings.w)
    };
    LINEAR_DEVICE_CHECK(isfinite(session.temperature) && session.temperature >= 0.0f && session.top_p > 0.0f &&
                        session.top_p <= 1.0f);
    unsigned long long best = 0;
    for (uint32_t local = threadIdx.x * SAMPLER_TOKENS_PER_LOAD; local < shard; local += SAMPLER_TURN) {
        const float4 four = logit_loads[local / SAMPLER_TOKENS_PER_LOAD];
        const float values[SAMPLER_TOKENS_PER_LOAD] = {four.x, four.y, four.z, four.w};
#pragma unroll
        for (uint32_t place = 0; place < SAMPLER_TOKENS_PER_LOAD; place++) {
            best = max(best, sampler_key(values[place], first + local + place));
        }
    }
    best = sampler_block_maximum(best, warp_keys);
    const uint64_t use = step->sequence * SAMPLER_SHARD_CALLS + arguments.call;
    unsigned long long global_best = 0;
    float global_logit = 0.0f;
    shard_exchange(&arguments, arena, use, 0, blockIdx.y, best, logits[(UINT32_MAX - (uint32_t)best) - first],
                   rank_keys, rank_logits, &global_best, &global_logit);
    const float inverse =
	    1.0f / fmaxf(session.temperature, SAMPLER_TEMPERATURE_FLOOR);
    const float scaled_largest = global_logit * inverse;
    if (session.temperature > 0.0f && isfinite(scaled_largest)) {
        if (session.top_p < 1.0f)
            asm volatile("trap;"); /* refused: the host prepares no such launch (sampler_shard_records.h) */
        const unsigned long long seed_key = sampler_mix(session.seed);
	const unsigned long long position_key =
		(unsigned long long)(uint32_t)row.position
		<< SAMPLER_POSITION_SHIFT;
	best = 0;
        for (uint32_t local = threadIdx.x * SAMPLER_TOKENS_PER_LOAD; local < shard; local += SAMPLER_TURN) {
            const float4 four = logit_loads[local / SAMPLER_TOKENS_PER_LOAD];
            const float values[SAMPLER_TOKENS_PER_LOAD] = {sampler_weight(four.x, scaled_largest, inverse),
                                                           sampler_weight(four.y, scaled_largest, inverse),
                                                           sampler_weight(four.z, scaled_largest, inverse),
                                                           sampler_weight(four.w, scaled_largest, inverse)};
#pragma unroll
            for (uint32_t place = 0; place < SAMPLER_TOKENS_PER_LOAD; place++) {
                const uint32_t token = first + local + place;
		const unsigned long long hash = sampler_mix(
			seed_key ^ ((position_key | token) *
				    PACK_DIGEST_POSITION_MULTIPLIER));
		const float uniform =
			((float)(uint32_t)(hash >> SAMPLER_UNIFORM_SHIFT) +
			 SAMPLER_HALF) *
			SAMPLER_UNIFORM_SCALE;
		best = max(best, sampler_key(values[place] / -logf(uniform), token));
            }
        }
        best = sampler_block_maximum(best, warp_keys);
        shard_exchange(&arguments, arena, use, 1, blockIdx.y, best,
                       logits[(UINT32_MAX - (uint32_t)best) - first], rank_keys, rank_logits, &global_best,
                       &global_logit);
    }
    if (threadIdx.x == 0) {
        answer->token_id = (int32_t)(UINT32_MAX - (uint32_t)global_best);
        answer->logit = global_logit;
    }
}

extern "C" __global__ void __launch_bounds__(SAMPLER_SETTINGS_PUSH_THREADS)
	sampler_settings_push(const struct sampler_settings_push_arguments
				      __grid_constant__ arguments,
			      unsigned char *arena)
{
    const uint64_t *const copies = (const uint64_t *)(arena + arguments.peer_table_offset);
    const uint4 *const sessions = (const uint4 *)(arena + arguments.sessions_offset);
    for (uint32_t slot = threadIdx.x; slot < arguments.session_capacity; slot += blockDim.x) {
        const uint4 settings = sessions[slot];
        for (uint32_t rank = 0; rank < SHARD_RANKS; rank++)
		shard_record_store((uint4 *)((unsigned char *)copies[rank] +
					     SAMPLER_SETTINGS_SLOTS_OFFSET) +
					   slot,
				   settings);
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x < SHARD_RANKS) {
	    const uint64_t sequence =
		    ((const struct step_block *)(arena + arguments.step_offset))
			    ->sequence;
	    asm volatile("st.release.sys.global.u64 [%0], %1;\n" ::"l"(
				 (unsigned char *)copies[threadIdx.x]),
			 "l"(sequence)
			 : "memory");
    }
}
