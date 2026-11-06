/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* The per-output chain and fold found by the independent captured-input CPU
 * search. The native full-head geometry gives four outputs per block with16
 * reduction owners each. Actual rows change only grid-y; no row changes another
 * row's owner, instruction chain or rounding. No reference library code is used.
 */
#include "kernel_linear_common.cuh"
#include "anemone/official_head_records.h"

#define NORMAL_FLOAT32_MINIMUM 0x1p-126f
#define HEAD_EVERY_LANE 0xffffffffu
#define HEAD_TREE_RADIX 2u

extern "C" __global__ void
official_head(struct official_head_arguments arguments, unsigned char *arena)
{
	LINEAR_DEVICE_CHECK(
		blockDim.x == OFFICIAL_HEAD_OWNERS &&
		blockDim.y == OFFICIAL_HEAD_OUTPUTS && blockDim.z == 1 &&
		gridDim.x * OFFICIAL_HEAD_OUTPUTS == arguments.output_size &&
		gridDim.y == arguments.row_count && gridDim.z == 1);
	const uint32_t output =
		blockIdx.x * OFFICIAL_HEAD_OUTPUTS + threadIdx.y;
	const uint16_t *const weight =
		(const uint16_t *)(arena + arguments.weights_offset) +
		(size_t)output * MODEL_HIDDEN;
	const uint16_t *const input =
		(const uint16_t *)(arena + arguments.rows_offset) +
		(size_t)blockIdx.y * MODEL_HIDDEN;
	float sum = 0.0f;
	for (uint32_t k = threadIdx.x; k < MODEL_HIDDEN;
	     k += OFFICIAL_HEAD_OWNERS) {
		const float w = linear_bfloat16_value(weight[k]);
		const float x = linear_bfloat16_value(input[k]);
		LINEAR_DEVICE_CHECK(
			isfinite(w) && isfinite(x) &&
			isfinite(__fmul_rn(w, x)) &&
			(__fmul_rn(w, x) == 0.0f ||
			 fabsf(__fmul_rn(w, x)) >= NORMAL_FLOAT32_MINIMUM));
		sum = __fmaf_rn(w, x, sum);
	}
	sum = __fadd_rn(0.0f, sum);
	for (uint32_t distance = OFFICIAL_HEAD_OWNERS / HEAD_TREE_RADIX;
	     distance != 0; distance >>= 1u)
		sum = __fadd_rn(sum,
				__shfl_down_sync(HEAD_EVERY_LANE, sum, distance,
						 OFFICIAL_HEAD_OWNERS));
	if (threadIdx.x == 0) {
		LINEAR_DEVICE_CHECK(isfinite(sum));
		((float *)(arena + arguments.output_offset))
			[(size_t)blockIdx.y * arguments.output_size + output] =
				__fadd_rn(0.0f, sum);
	}
}

/* Independent row chains interleave only reads and FMAs. A weight load serves
 * all actual rows before the next K position; no accumulator is shared by rows.
 */
extern "C" __global__ void
official_head_reused(struct official_head_arguments arguments,
		     unsigned char *arena)
{
	LINEAR_DEVICE_CHECK(
		blockDim.x == OFFICIAL_HEAD_OWNERS &&
		blockDim.y == OFFICIAL_HEAD_OUTPUTS && blockDim.z == 1 &&
		gridDim.x * OFFICIAL_HEAD_OUTPUTS == arguments.output_size &&
		gridDim.y == 1 && gridDim.z == 1 && arguments.row_count != 0 &&
		arguments.row_count <= LINEAR_MAXIMUM_ROWS);
	const uint32_t output =
		blockIdx.x * OFFICIAL_HEAD_OUTPUTS + threadIdx.y;
	const uint16_t *const weight =
		(const uint16_t *)(arena + arguments.weights_offset) +
		(size_t)output * MODEL_HIDDEN;
	const uint16_t *const input =
		(const uint16_t *)(arena + arguments.rows_offset);
	float sums[LINEAR_MAXIMUM_ROWS] = { 0.0f };
	for (uint32_t k = threadIdx.x; k < MODEL_HIDDEN;
	     k += OFFICIAL_HEAD_OWNERS) {
		const float w = linear_bfloat16_value(weight[k]);
#pragma unroll
		for (uint32_t row = 0; row < LINEAR_MAXIMUM_ROWS; row++) {
			if (row < arguments.row_count) {
				const float x = linear_bfloat16_value(
					input[(size_t)row * MODEL_HIDDEN + k]);
				LINEAR_DEVICE_CHECK(
					isfinite(w) && isfinite(x) &&
					isfinite(__fmul_rn(w, x)) &&
					(__fmul_rn(w, x) == 0.0f ||
					 fabsf(__fmul_rn(w, x)) >=
						 NORMAL_FLOAT32_MINIMUM));
				sums[row] = __fmaf_rn(w, x, sums[row]);
			}
		}
	}
#pragma unroll
	for (uint32_t row = 0; row < LINEAR_MAXIMUM_ROWS; row++) {
		if (row < arguments.row_count) {
			float sum = __fadd_rn(0.0f, sums[row]);
			for (uint32_t distance =
				     OFFICIAL_HEAD_OWNERS / HEAD_TREE_RADIX;
			     distance != 0; distance >>= 1u)
				sum = __fadd_rn(
					sum,
					__shfl_down_sync(HEAD_EVERY_LANE, sum,
							 distance,
							 OFFICIAL_HEAD_OWNERS));
			if (threadIdx.x == 0) {
				LINEAR_DEVICE_CHECK(isfinite(sum));
				((float *)(arena + arguments.output_offset))
					[(size_t)row * arguments.output_size +
					 output] = __fadd_rn(0.0f, sum);
			}
		}
	}
}
