/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * kernel_router.cuh: what the router's kernels (kernel_router.cu) share with the FFN front (kernel_site_front.cu):
 * the key of an expert, the six experts of one row by one warp, and the select's tail, the picks of the live rows and
 * their job table. Device code only. Contract: include/anemone/router_records.h (router_select_entry).
 */
#ifndef KERNEL_ROUTER_CUH
#define KERNEL_ROUTER_CUH

#include "kernel_linear_common.cuh"

#include "anemone/router_records.h"

#define EXPERTS MODEL_EXPERTS
#define PICKS MODEL_EXPERTS_PER_ROW
#define EXPERTS_PER_LANE (EXPERTS / THREADS_PER_WARP)
#define MOST_PAIRS (LINEAR_MAXIMUM_ROWS * PICKS)
#define JOB_WORDS (EXPERT_JOB_BYTES / sizeof(uint32_t))
#define ALL_BITS \
	0xFFFFFFFFu /* every lane of a warp; the key of a not-a-number */
#define SIGN_BIT 0x80000000u
#define KEY_SHIFT 32u
#define TAKEN \
	0ull /* below every key: an ordered float32 is at least 0x007FFFFF */

enum router_place {
	FIRST,
	SECOND,
	THIRD,
	FOURTH,
	FIFTH,
	SIXTH
}; /* of the six in the order of the top-k */

__device__ static __forceinline__ uint64_t router_key(float biased,
						      uint32_t expert)
{
	const uint32_t bits = __float_as_uint(biased);
	const uint32_t ordered =
		biased != biased ?
			ALL_BITS :
			bits ^ ((bits & SIGN_BIT) != 0 ? ALL_BITS : SIGN_BIT);
	return (uint64_t)ordered << KEY_SHIFT | (uint64_t)(uint32_t)~expert;
}

/* One row of `experts` experts, by one warp: its `picks` experts by slot, and their weights. The reference order's
 * sum exists for the model's six picks alone. */
template <uint32_t experts, uint32_t picks>
__device__ static __forceinline__ void
router_row(const float *affinities, const float *bias, uint32_t lane,
	   uint32_t reference_order, uint32_t *chosen, float *weights)
{
	uint64_t keys[experts / THREADS_PER_WARP];
	uint32_t picked_experts[picks];
	float picked[picks], sum = 0.0f;
#pragma unroll
	for (uint32_t place = 0, expert = lane;
	     place < experts / THREADS_PER_WARP;
	     place++, expert += THREADS_PER_WARP) {
		keys[place] =
			router_key(affinities[expert] + bias[expert], expert);
	}
#pragma unroll
	for (uint32_t pick = 0; pick < picks; pick++) {
		uint64_t best = TAKEN;
#pragma unroll
		for (uint32_t place = 0; place < experts / THREADS_PER_WARP;
		     place++) {
			best = keys[place] > best ? keys[place] : best;
		}
		for (uint32_t distance = THREADS_PER_WARP >> 1u; distance != 0;
		     distance >>= 1u) {
			const uint64_t other =
				__shfl_xor_sync(ALL_BITS, best, (int)distance);
			best = other > best ? other : best;
		}
#pragma unroll
		for (uint32_t place = 0; place < experts / THREADS_PER_WARP;
		     place++) {
			keys[place] = keys[place] == best ? TAKEN : keys[place];
		}
		picked_experts[pick] = ~(uint32_t)best;
		picked[pick] = affinities[picked_experts[pick]];
	}
#pragma unroll
	for (uint32_t pick = 0; pick < picks; pick++) {
		uint32_t slot = 0;
#pragma unroll
		for (uint32_t other = 0; other < picks; other++) {
			slot += picked_experts[other] < picked_experts[pick] ?
					1u :
					0u;
		}
		if (lane == 0) {
			chosen[slot] = picked_experts[pick];
			weights[slot] = picked[pick];
		}
	}
	__syncwarp();
#pragma unroll
	for (uint32_t slot = 0; slot < picks; slot++) {
		sum += weights[slot];
	}
	if constexpr (picks == PICKS) {
		if (reference_order != 0) {
			sum = ((picked[FIRST] + picked[FIFTH]) +
			       picked[THIRD]) +
			      ((picked[SECOND] + picked[SIXTH]) +
			       picked[FOURTH]);
		}
	}
	__syncwarp();
	if (lane < picks) {
		weights[lane] = weights[lane] / (sum + ROUTER_WEIGHT_EPSILON) *
				MODEL_ROUTE_SCALE;
	}
}

/* The select behind the affinities of the live rows in shared memory (affinities: [row][384]): each live row's six
 * experts by its warp, then the job table (router_records.h): router_select_entry's own statements, a macro so that
 * its module keeps the instructions it had in their order (a function of the same statements, inlined, compiles to
 * them in another schedule on sm_120a, 26 October 2026). arguments is the select's record (an expression of struct
 * router_select_arguments), bias the experts' bias (the arena's or a copy). The caller's names: arena,
 * live_rows, warp (the thread's), pair (the thread's index in the block of ROUTER_THREADS), pair_count (6
 * times the live rows), affinities, pair_weights, pair_experts and pair_is_first (shared memory: the select's arrays).
 * Every thread of the block runs it and passes its barriers. */
#define ROUTER_SELECT_JOBS(arguments, bias, expert_count, pick_count)          \
	do {                                                                   \
		if (warp < live_rows) {                                        \
			router_row<expert_count, pick_count>(                  \
				affinities + warp * expert_count, (bias),      \
				threadIdx.x % THREADS_PER_WARP,                \
				arguments.reference_order,                     \
				pair_experts + warp * pick_count,              \
				pair_weights + warp * pick_count);             \
		}                                                              \
		__syncthreads();                                               \
		const uint32_t expert =                                        \
			pair < pair_count ? pair_experts[pair] : 0u;           \
		uint32_t column = 0, job = 0, rows = 0;                        \
		for (uint32_t earlier = 0;                                     \
		     earlier < pair && pair < pair_count; earlier++) {         \
			column += pair_experts[earlier] == expert ? 1u : 0u;   \
		}                                                              \
		if (pair < pair_count) {                                       \
			pair_is_first[pair] = column == 0 ? 1u : 0u;           \
		}                                                              \
		const uint32_t used_jobs = (uint32_t)__syncthreads_count(      \
			pair < pair_count && column == 0);                     \
		struct expert_job_record *const jobs =                         \
			(struct expert_job_record *)(arena +                   \
						     arguments.jobs_offset);   \
		for (uint32_t other = 0;                                       \
		     other < pair_count && pair < pair_count; other++) {       \
			job += pair_is_first[other] != 0 &&                    \
					       pair_experts[other] < expert ?  \
				       1u :                                    \
				       0u;                                     \
			rows += pair_experts[other] == expert ? 1u : 0u;       \
		}                                                              \
		if (pair < pair_count) {                                       \
			jobs[job].pair_index[column] = pair;                   \
			jobs[job].route_weight[column] = pair_weights[pair];   \
		}                                                              \
		if (pair < pair_count && column == 0) {                        \
			const uint64_t *const offsets =                        \
				(const uint64_t                                \
					 *)(arena +                            \
					    arguments.expert_offsets_offset) + \
				(size_t)expert * MODEL_EXPERT_MATRICES;        \
			const uint64_t gate = offsets[ROUTER_GATE],            \
				       up = offsets[ROUTER_UP];                \
			const uint64_t down = offsets[ROUTER_DOWN];            \
			LINEAR_DEVICE_CHECK(                                   \
				(gate | up |                                   \
				 down) % LINEAR_OFFSET_ALIGNMENT_BYTES ==      \
					0 &&                                   \
				gate >= arguments.weights_offset &&            \
				gate <= arguments.gate_up_last_offset &&       \
				up >= arguments.weights_offset &&              \
				up <= arguments.gate_up_last_offset &&         \
				down >= arguments.weights_offset &&            \
				down <= arguments.down_last_offset);           \
			jobs[job].gate_offset = gate;                          \
			jobs[job].up_offset = up;                              \
			jobs[job].down_offset = down;                          \
			jobs[job].row_count = rows;                            \
			jobs[job].expert = expert;                             \
			for (uint32_t unused = rows;                           \
			     unused < LINEAR_MAXIMUM_ROWS; unused++) {         \
				jobs[job].pair_index[unused] = 0;              \
				jobs[job].route_weight[unused] = 0.0f;         \
			}                                                      \
		}                                                              \
		uint32_t *const empty_words = (uint32_t *)(jobs + used_jobs);  \
		for (size_t word = threadIdx.x;                                \
		     word < (arguments.row_count * pick_count - used_jobs) *   \
				    JOB_WORDS;                                 \
		     word += blockDim.x) {                                     \
			empty_words[word] = 0;                                 \
		}                                                              \
	} while (0)

#endif
