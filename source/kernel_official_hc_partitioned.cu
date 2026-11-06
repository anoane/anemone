/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* The direct dot47 form's exact partials, now distributed over CTAs. The
 * second independent pass retains its fold and the source F1 row-RMS tree.
 */
#include "kernel_linear_common.cuh"
#include "anemone/official_hc_partitioned_records.h"

#define DOT_VECTOR 2u
#define OWNER_GROUPS (OFFICIAL_HC_PARTIAL_THREADS / THREADS_PER_WARP)
#define FINAL_GROUPS 4u
#define STATISTIC_VECTOR 4u
#define EVERY_LANE 0xFFFFFFFFu

extern "C" __global__ void
official_hc_partials(struct official_hc_partitioned_arguments arguments,
		     unsigned char *arena)
{
	__shared__ float owners[OFFICIAL_HC_PARTIAL_THREADS];
	LINEAR_DEVICE_CHECK(blockDim.x == OFFICIAL_HC_PARTIAL_THREADS &&
			    gridDim.x == MODEL_MIXES &&
			    gridDim.y == OFFICIAL_HC_PARTITIONS &&
			    gridDim.z == arguments.actual_rows &&
			    arguments.reserved_zero == 0 &&
			    arguments.reserved_tail == 0);
	const uint32_t thread = threadIdx.x, partition = blockIdx.y;
	const uint16_t *const input =
		(const uint16_t *)(arena + arguments.rows_offset) +
		(size_t)blockIdx.z * OFFICIAL_HC_INPUTS;
	const float *const weights =
		(const float *)(arena + arguments.weights_offset) +
		(size_t)blockIdx.x * OFFICIAL_HC_INPUTS;
	float chains[DOT_VECTOR] = {};
	for (uint32_t first =
		     (partition * OFFICIAL_HC_PARTIAL_THREADS + thread) *
		     DOT_VECTOR;
	     first < OFFICIAL_HC_INPUTS;
	     first += OFFICIAL_HC_PARTITIONS * OFFICIAL_HC_PARTIAL_THREADS *
		      DOT_VECTOR) {
		for (uint32_t slot = 0; slot < DOT_VECTOR; slot++)
			chains[slot] = __fmaf_rn(
				linear_bfloat16_value(input[first + slot]),
				weights[first + slot], chains[slot]);
	}
	owners[thread] = __fadd_rn(__fadd_rn(0.0f, chains[0]), chains[1]);
	__syncthreads();
	if (thread < THREADS_PER_WARP) {
		float sum = __fadd_rn(
			__fadd_rn(
				owners[thread],
				owners[thread + DOT_VECTOR * THREADS_PER_WARP]),
			__fadd_rn(owners[thread + THREADS_PER_WARP],
				  owners[thread + (OWNER_GROUPS - 1u) *
							  THREADS_PER_WARP]));
		for (uint32_t distance = THREADS_PER_WARP / DOT_VECTOR;
		     distance != 0; distance /= DOT_VECTOR)
			sum = __fadd_rn(sum, __shfl_down_sync(EVERY_LANE, sum,
							      distance));
		if (thread == 0)
			((float *)(arena + arguments.partials_offset))
				[((size_t)blockIdx.z * MODEL_MIXES + blockIdx.x) *
					 OFFICIAL_HC_PARTITIONS +
				 partition] = sum;
	}
}

extern "C" __global__ void
official_hc_reduce(struct official_hc_partitioned_arguments arguments,
		   unsigned char *arena)
{
	__shared__ float sums[OFFICIAL_HC_THREADS];
	LINEAR_DEVICE_CHECK(
		blockDim.x == OFFICIAL_HC_THREADS && gridDim.x == 1 &&
		gridDim.y == arguments.actual_rows && gridDim.z == 1 &&
		arguments.reserved_zero == 0 && arguments.reserved_tail == 0);
	const uint32_t thread = threadIdx.x;
	const uint16_t *const input =
		(const uint16_t *)(arena + arguments.rows_offset) +
		(size_t)blockIdx.y * OFFICIAL_HC_INPUTS;
	float chains[STATISTIC_VECTOR] = {};
	for (uint32_t first = thread * STATISTIC_VECTOR;
	     first < OFFICIAL_HC_INPUTS;
	     first += OFFICIAL_HC_THREADS * STATISTIC_VECTOR) {
		for (uint32_t slot = 0; slot < STATISTIC_VECTOR; slot++) {
			const float value =
				linear_bfloat16_value(input[first + slot]);
			chains[slot] = __fadd_rn(chains[slot],
						 __fmul_rn(value, value));
		}
	}
	sums[thread] = __fadd_rn(__fadd_rn(__fadd_rn(chains[0], chains[1]),
					   chains[DOT_VECTOR]),
				 chains[STATISTIC_VECTOR - 1u]);
	__syncthreads();
	for (uint32_t distance = OFFICIAL_HC_THREADS / DOT_VECTOR;
	     distance >= THREADS_PER_WARP; distance /= DOT_VECTOR) {
		if (thread < distance)
			sums[thread] = __fadd_rn(sums[thread],
						 sums[thread + distance]);
		__syncthreads();
	}
	if (thread < THREADS_PER_WARP) {
		float sum = sums[thread];
		for (uint32_t distance = THREADS_PER_WARP / DOT_VECTOR;
		     distance != 0; distance /= DOT_VECTOR)
			sum = __fadd_rn(sum, __shfl_down_sync(EVERY_LANE, sum,
							      distance));
		if (thread == 0)
			sums[0] = rsqrtf(__fadd_rn(
				__fmul_rn(sum,
					  1.0f / (float)OFFICIAL_HC_INPUTS),
				arguments.epsilon));
	}
	__syncthreads();
	if (thread >= MODEL_MIXES)
		return;
	const float *const partial =
		(const float *)(arena + arguments.partials_offset) +
		((size_t)blockIdx.y * MODEL_MIXES + thread) *
			OFFICIAL_HC_PARTITIONS;
	float result = 0.0f;
	for (uint32_t group = 0; group < FINAL_GROUPS; group++) {
		float sum = 0.0f;
		for (uint32_t index = group; index < THREADS_PER_WARP;
		     index += FINAL_GROUPS) {
			const uint32_t upper = index + THREADS_PER_WARP;
			const float folded = __fadd_rn(
				partial[index], upper < OFFICIAL_HC_PARTITIONS ?
							partial[upper] :
							0.0f);
			sum = __fadd_rn(folded, sum);
		}
		result = __fadd_rn(result, sum);
	}
	((float *)(arena +
		   arguments.output_offset))[(size_t)blockIdx.y * MODEL_MIXES +
					     thread] =
		__fmul_rn(result, sums[0]);
}
