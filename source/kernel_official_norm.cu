/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Source-derived contiguous vector4 square/mean order, kept independent of selected norms.
 * Logical block Y assigns different rows for these widths; separate CTAs retain each row's
 * exact input ownership and X tree. No library implementation is copied here.
 */
#include "kernel_linear_common.cuh"

#include "anemone/official_norm_records.h"

#define VECTOR_VALUES 4u
#define TWO 2u
#define EVERY_LANE 0xFFFFFFFFu

extern "C" __global__ void
official_norm_entry(struct official_norm_arguments arguments,
		    unsigned char *arena)
{
	__shared__ float sums[OFFICIAL_NORM_MOST_THREADS];
	const uint32_t thread = threadIdx.x;
	const size_t row = blockIdx.y;
	LINEAR_DEVICE_CHECK(blockDim.x >= THREADS_PER_WARP &&
			    blockDim.x <= OFFICIAL_NORM_MOST_THREADS &&
			    (blockDim.x & (blockDim.x - 1u)) == 0 &&
			    gridDim.x == 1 && arguments.reserved_zero == 0 &&
			    (arguments.width == MODEL_HEAD_DIMS ||
			     arguments.width == MODEL_QUERY_RANK ||
			     arguments.width == MODEL_HIDDEN));
	const uint16_t *const input =
		(const uint16_t *)(arena + arguments.input_offset) +
		row * arguments.width;
	const uint16_t *const weight =
		(const uint16_t *)(arena + arguments.weight_offset);
	uint16_t *const output = (uint16_t *)(arena + arguments.output_offset) +
				 row * arguments.width;
	float chains[VECTOR_VALUES] = {};
	for (uint32_t first = thread * VECTOR_VALUES; first < arguments.width;
	     first += blockDim.x * VECTOR_VALUES) {
		for (uint32_t slot = 0; slot < VECTOR_VALUES; slot++) {
			const float value =
				linear_bfloat16_value(input[first + slot]);
			chains[slot] = __fadd_rn(chains[slot],
						 __fmul_rn(value, value));
		}
	}
	sums[thread] = __fadd_rn(__fadd_rn(__fadd_rn(chains[0], chains[1]),
					   chains[TWO]),
				 chains[VECTOR_VALUES - 1u]);
	__syncthreads();
	for (uint32_t distance = blockDim.x / TWO; distance >= THREADS_PER_WARP;
	     distance /= TWO) {
		if (thread < distance)
			sums[thread] = __fadd_rn(sums[thread],
						 sums[thread + distance]);
		__syncthreads();
	}
	if (thread < THREADS_PER_WARP) {
		float sum = sums[thread];
		for (uint32_t distance = THREADS_PER_WARP / TWO; distance != 0;
		     distance /= TWO)
			sum = __fadd_rn(sum, __shfl_down_sync(EVERY_LANE, sum,
							      distance));
		if (thread == 0) {
			const float mean =
				__fmul_rn(sum, arguments.mean_factor);
			const float inverse =
				rsqrtf(__fadd_rn(mean, arguments.epsilon));
			float *const statistics =
				(float *)(arena + arguments.statistics_offset) +
				row * OFFICIAL_NORM_STATISTICS;
			statistics[0] = mean;
			statistics[1] = inverse;
			sums[0] = inverse;
		}
	}
	__syncthreads();
	const float inverse = sums[0];
	for (uint32_t column = thread; column < arguments.width;
	     column += blockDim.x)
		output[column] = (uint16_t)linear_bfloat16_bits(__fmul_rn(
			linear_bfloat16_value(weight[column]),
			__fmul_rn(linear_bfloat16_value(input[column]),
				  inverse)));
}
