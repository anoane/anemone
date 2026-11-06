/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Optional composition; independent norm, rotation and roundtrip remain
 * available. BF16 barriers are actual conversions in registers. Whole warps
 * own the norm's output groups, including the adjacent rotary pairs.
 */
#include "kernel_activation_quantize.cuh"
#include "anemone/official_window_records.h"

#define VECTOR_VALUES 4u
#define TWO 2u
#define EVERY_LANE 0xFFFFFFFFu
#define ROUNDTRIP_UNCARRY 0x1p-100f
#define ROUNDTRIP_DOMAIN_LIMIT 65536.0f

extern "C" __global__ void
official_window(struct official_window_arguments arguments,
		unsigned char *arena)
{
	__shared__ float sums[OFFICIAL_NORM_MOST_THREADS];
	const uint32_t thread = threadIdx.x;
	const size_t row = blockIdx.y;
	LINEAR_DEVICE_CHECK(
		(blockDim.x == MODEL_HEAD_DIMS / VECTOR_VALUES ||
		 blockDim.x == MODEL_HEAD_DIMS / (VECTOR_VALUES * TWO)) &&
		gridDim.x == 1 && gridDim.y == arguments.rows &&
		arguments.reserved_zero == 0);
	const uint16_t *const input =
		(const uint16_t *)(arena + arguments.input_offset) +
		row * MODEL_HEAD_DIMS;
	const uint16_t *const gain =
		(const uint16_t *)(arena + arguments.weight_offset);
	float chains[VECTOR_VALUES] = {};
	for (uint32_t first = thread * VECTOR_VALUES; first < MODEL_HEAD_DIMS;
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
	for (uint32_t column = thread; column < MODEL_HEAD_DIMS;
	     column += blockDim.x) {
		float value =
			linear_bfloat16_value(linear_bfloat16_bits(__fmul_rn(
				linear_bfloat16_value(gain[column]),
				__fmul_rn(linear_bfloat16_value(input[column]),
					  inverse))));
		if (column >= MODEL_FIRST_ROTATED_DIM) {
			const float partner =
				__shfl_xor_sync(EVERY_LANE, value, 1);
			const uint32_t pair =
				(column - MODEL_FIRST_ROTATED_DIM) /
				MODEL_PAIR_VALUES;
			const float2 rotation =
				((const float2 *)(arena +
						  arguments.rotation_offset))
					[row * MODEL_FREQUENCIES + pair];
			const float product = __fmul_rn(partner, rotation.y);
			value = linear_bfloat16_value(linear_bfloat16_bits(fmaf(
				value, rotation.x,
				column % MODEL_PAIR_VALUES == 0 ? -product :
								  product)));
		}
		LINEAR_DEVICE_CHECK(isfinite(value) &&
				    fabsf(value) <= ROUNDTRIP_DOMAIN_LIMIT);
		float largest = fabsf(value);
		for (uint32_t distance = THREADS_PER_WARP / TWO; distance != 0;
		     distance /= TWO)
			largest = fmaxf(largest,
					__shfl_xor_sync(EVERY_LANE, largest,
							distance));
		const uint32_t exponent = quantize_exponent_code(largest);
		uint32_t operand = 0;
		(void)quantize_nearest_e4m3(
			value * quantize_inverse_scale(exponent), exponent,
			&operand);
		((uint16_t *)(arena +
			      arguments.output_offset))[row * MODEL_HEAD_DIMS +
							column] =
			linear_bfloat16_bits(linear_bfloat16_value(operand) *
					     ROUNDTRIP_UNCARRY);
	}
}
