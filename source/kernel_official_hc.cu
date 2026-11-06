/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Experimental mHC projection orders, retained independently of the selected dense kernels.
 * These mirror the observed pinned-PRO library arithmetic, using our building-block
 * derivation (bb_hc_dot47 / bb_hc_split128), rather than a vendor implementation.
 * Admission belongs to the caller: FP32 weights, BF16 streams, 24 by 20480,
 * rows 1 or 3/4/8, positive norm epsilon, disjoint read/write extents.
 */
#include "kernel_linear_common.cuh"

#include "anemone/official_hc_records.h"

#define OFFICIAL_THREADS OFFICIAL_HC_THREADS
#define OFFICIAL_STREAMS MODEL_STREAMS
#define OFFICIAL_INPUTS OFFICIAL_HC_INPUTS
#define OFFICIAL_MIXES MODEL_MIXES
#define STATISTIC_VECTOR 4u
#define STATISTIC_GROUPS (OFFICIAL_THREADS / THREADS_PER_WARP)
#define DOT_PARTITIONS 47u
#define DOT_OWNERS 128u
#define DOT_VECTOR 2u
#define DOT_WARP_GROUPS (DOT_OWNERS / THREADS_PER_WARP)
#define DOT_FINAL_WIDTH 32u
#define DOT_FINAL_GROUPS 4u
#define DOT_PARTIALS (DOT_PARTITIONS + DOT_FINAL_GROUPS)
#define SPLIT_INNER 128u
#define SPLIT_PARTITIONS (OFFICIAL_INPUTS / SPLIT_INNER)
#define STAGE_FEATURES 32u
#define STAGE_PITCH (SPLIT_PARTITIONS + 1u)
#define STAGE_VALUES (STAGE_FEATURES * STAGE_PITCH)
#define STAGE_PLANES 2u
#define EVERY_LANE 0xFFFFFFFFu

/* The pinned contiguous mean kernel: four independent RN square/add chains,
 * then descending X and Y trees. Its logical X/Y changes with row count.
 */
__device__ static float official_hc_inverse(const uint16_t *input,
					    uint32_t rows, float epsilon,
					    float *values, float *groups)
{
	const uint32_t thread = threadIdx.x;
	uint32_t group_count = 1;
	while (group_count * DOT_VECTOR <= rows)
		group_count *= DOT_VECTOR;
	const uint32_t width = OFFICIAL_THREADS / group_count;
	const uint32_t lane = thread % width, group = thread / width;
	float chains[STATISTIC_VECTOR] = {};
	for (uint32_t first = thread * STATISTIC_VECTOR;
	     first < OFFICIAL_INPUTS;
	     first += OFFICIAL_THREADS * STATISTIC_VECTOR) {
		for (uint32_t slot = 0; slot < STATISTIC_VECTOR; slot++) {
			const float value =
				linear_bfloat16_value(input[first + slot]);
			chains[slot] = __fadd_rn(chains[slot],
						 __fmul_rn(value, value));
		}
	}
	float sum = __fadd_rn(__fadd_rn(__fadd_rn(chains[0], chains[1]),
					chains[DOT_VECTOR]),
			      chains[STATISTIC_VECTOR - 1u]);
	values[thread] = sum;
	__syncthreads();
	for (uint32_t distance = width / DOT_VECTOR;
	     distance >= THREADS_PER_WARP; distance /= DOT_VECTOR) {
		if (lane < distance)
			values[thread] = __fadd_rn(values[thread],
						   values[thread + distance]);
		__syncthreads();
	}
	if (lane < THREADS_PER_WARP) {
		sum = values[thread];
		for (uint32_t distance = THREADS_PER_WARP / DOT_VECTOR;
		     distance != 0; distance /= DOT_VECTOR)
			sum = __fadd_rn(sum, __shfl_down_sync(EVERY_LANE, sum,
							      distance));
		if (lane == 0)
			groups[group] = sum;
	}
	__syncthreads();
	for (uint32_t distance = group_count / DOT_VECTOR; distance != 0;
	     distance /= DOT_VECTOR) {
		if (thread < distance)
			groups[thread] = __fadd_rn(groups[thread],
						   groups[thread + distance]);
		__syncthreads();
	}
	return rsqrtf(__fadd_rn(
		__fmul_rn(groups[0], 1.0f / (float)OFFICIAL_INPUTS), epsilon));
}

__device__ static float official_hc_dot_partition(const uint16_t *input,
						  const float *weights,
						  uint32_t partition)
{
	const uint32_t lane = threadIdx.x % THREADS_PER_WARP;
	float owners[DOT_WARP_GROUPS];
	for (uint32_t group = 0; group < DOT_WARP_GROUPS; group++) {
		float chains[DOT_VECTOR] = {};
		for (uint32_t first = (partition * DOT_OWNERS + lane +
				       group * THREADS_PER_WARP) *
				      DOT_VECTOR;
		     first < OFFICIAL_INPUTS;
		     first += DOT_PARTITIONS * DOT_OWNERS * DOT_VECTOR) {
			for (uint32_t slot = 0; slot < DOT_VECTOR; slot++)
				chains[slot] = __fmaf_rn(
					linear_bfloat16_value(
						input[first + slot]),
					weights[first + slot], chains[slot]);
		}
		owners[group] =
			__fadd_rn(__fadd_rn(0.0f, chains[0]), chains[1]);
	}
	float sum =
		__fadd_rn(__fadd_rn(owners[0], owners[DOT_VECTOR]),
			  __fadd_rn(owners[1], owners[DOT_WARP_GROUPS - 1u]));
	for (uint32_t distance = THREADS_PER_WARP / DOT_VECTOR; distance != 0;
	     distance /= DOT_VECTOR)
		sum = __fadd_rn(sum,
				__shfl_down_sync(EVERY_LANE, sum, distance));
	return sum;
}

extern "C" __global__ void
official_hc_dot47(struct dense_linear_arguments arguments, unsigned char *arena)
{
	__shared__ float partial[DOT_PARTIALS], values[OFFICIAL_THREADS],
		groups[STATISTIC_GROUPS];
	const uint16_t *const input =
		(const uint16_t *)(arena + arguments.rows_offset) +
		(size_t)blockIdx.y * OFFICIAL_INPUTS;
	const float *const weights =
		(const float *)(arena + arguments.weights_offset) +
		(size_t)blockIdx.x * OFFICIAL_INPUTS;
	const uint32_t warp = threadIdx.x / THREADS_PER_WARP;
	for (uint32_t partition = warp; partition < DOT_PARTITIONS;
	     partition += OFFICIAL_THREADS / THREADS_PER_WARP) {
		const float sum =
			official_hc_dot_partition(input, weights, partition);
		if (threadIdx.x % THREADS_PER_WARP == 0)
			partial[partition] = sum;
	}
	const float inverse = official_hc_inverse(input, 1, arguments.epsilon,
						  values, groups);
	if (threadIdx.x < DOT_FINAL_GROUPS) {
		float sum = 0.0f;
		for (uint32_t index = threadIdx.x; index < DOT_FINAL_WIDTH;
		     index += DOT_FINAL_GROUPS) {
			const uint32_t upper = index + DOT_FINAL_WIDTH;
			const float folded = __fadd_rn(
				partial[index],
				upper < DOT_PARTITIONS ? partial[upper] : 0.0f);
			sum = __fadd_rn(folded, sum);
		}
		partial[DOT_PARTITIONS + threadIdx.x] = sum;
	}
	__syncthreads();
	if (threadIdx.x == 0) {
		float sum = 0.0f;
		for (uint32_t group = 0; group < DOT_FINAL_GROUPS; group++)
			sum = __fadd_rn(sum, partial[DOT_PARTITIONS + group]);
		((float *)(arena + arguments.output_offset))
			[(size_t)blockIdx.y * OFFICIAL_MIXES + blockIdx.x] =
				__fmul_rn(sum, inverse);
	}
}

extern "C" __global__ void
official_hc_split128(struct dense_linear_arguments arguments,
		     unsigned char *arena)
{
	__shared__ float partial[SPLIT_PARTITIONS], values[OFFICIAL_THREADS],
		groups[STATISTIC_GROUPS];
	const uint16_t *const input =
		(const uint16_t *)(arena + arguments.rows_offset) +
		(size_t)blockIdx.y * OFFICIAL_INPUTS;
	const float *const weights =
		(const float *)(arena + arguments.weights_offset) +
		(size_t)blockIdx.x * OFFICIAL_INPUTS;
	if (threadIdx.x < SPLIT_PARTITIONS) {
		float sum = 0.0f;
		const uint32_t first = threadIdx.x * SPLIT_INNER;
		for (uint32_t feature = first; feature < first + SPLIT_INNER;
		     feature++)
			sum = __fmaf_rn(linear_bfloat16_value(input[feature]),
					weights[feature], sum);
		partial[threadIdx.x] = sum;
	}
	const float inverse = official_hc_inverse(
		input, arguments.row_count, arguments.epsilon, values, groups);
	if (threadIdx.x == 0) {
		float sum = 0.0f;
		for (uint32_t split = 0; split < SPLIT_PARTITIONS; split++)
			sum = __fadd_rn(sum, partial[split]);
		((float *)(arena + arguments.output_offset))
			[(size_t)blockIdx.y * OFFICIAL_MIXES + blockIdx.x] =
				__fmul_rn(sum, inverse);
	}
}

/* Coalesced global loads and a padded shared transpose change no FMA owner or
 * feature order. Every 128-term chain remains contiguous and independently RN.
 */
extern "C" __global__ void
official_hc_split128_staged(struct dense_linear_arguments arguments,
			    unsigned char *arena)
{
	__shared__ float partial[SPLIT_PARTITIONS], values[OFFICIAL_THREADS],
		groups[STATISTIC_GROUPS];
	__shared__ float stage[STAGE_PLANES * STAGE_VALUES];
	const uint16_t *const input =
		(const uint16_t *)(arena + arguments.rows_offset) +
		(size_t)blockIdx.y * OFFICIAL_INPUTS;
	const float *const weights =
		(const float *)(arena + arguments.weights_offset) +
		(size_t)blockIdx.x * OFFICIAL_INPUTS;
	float sum = 0.0f;
	for (uint32_t chunk = 0; chunk < SPLIT_INNER; chunk += STAGE_FEATURES) {
		for (uint32_t element = threadIdx.x;
		     element < STAGE_FEATURES * SPLIT_PARTITIONS;
		     element += OFFICIAL_THREADS) {
			const uint32_t split = element / STAGE_FEATURES,
				       feature = element % STAGE_FEATURES;
			const uint32_t inner =
				split * SPLIT_INNER + chunk + feature;
			const uint32_t stored = feature * STAGE_PITCH + split;
			stage[stored] = weights[inner];
			stage[STAGE_VALUES + stored] =
				linear_bfloat16_value(input[inner]);
		}
		__syncthreads();
		if (threadIdx.x < SPLIT_PARTITIONS) {
			for (uint32_t feature = 0; feature < STAGE_FEATURES;
			     feature++) {
				const uint32_t stored =
					feature * STAGE_PITCH + threadIdx.x;
				sum = __fmaf_rn(stage[STAGE_VALUES + stored],
						stage[stored], sum);
			}
		}
		__syncthreads();
	}
	if (threadIdx.x < SPLIT_PARTITIONS)
		partial[threadIdx.x] = sum;
	const float inverse = official_hc_inverse(
		input, arguments.row_count, arguments.epsilon, values, groups);
	if (threadIdx.x == 0) {
		float total = 0.0f;
		for (uint32_t split = 0; split < SPLIT_PARTITIONS; split++)
			total = __fadd_rn(total, partial[split]);
		((float *)(arena + arguments.output_offset))
			[(size_t)blockIdx.y * OFFICIAL_MIXES + blockIdx.x] =
				__fmul_rn(total, inverse);
	}
}
