/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* An independent deterministic equivalent of the observed cached-M1 product.
 * No vendor implementation is used. Each output's32 FMA chains own k modulo32;
 * the descending halving fold is independent of actual rows and CTA scheduling.
 */
#include "kernel_linear_common.cuh"
#include "anemone/official_router_projection_records.h"

#define NORMAL_FLOAT32_MINIMUM 0x1p-126f
#define ROUTER_EVERY_LANE 0xffffffffu
#define ROUTER_TREE_RADIX 2u

extern "C" __global__ void official_router_projection(
	struct official_router_projection_arguments arguments,
	unsigned char *arena)
{
	LINEAR_DEVICE_CHECK(blockDim.x == OFFICIAL_ROUTER_PROJECTION_OWNERS &&
			    blockDim.y == OFFICIAL_ROUTER_PROJECTION_OUTPUTS &&
			    blockDim.z == 1 &&
			    gridDim.y == arguments.row_count &&
			    gridDim.x * OFFICIAL_ROUTER_PROJECTION_OUTPUTS ==
				    MODEL_EXPERTS &&
			    gridDim.z == 1 && arguments.reserved_zero == 0);
	const uint32_t output =
		blockIdx.x * OFFICIAL_ROUTER_PROJECTION_OUTPUTS + threadIdx.y;
	const uint16_t *const weight =
		(const uint16_t *)(arena + arguments.weights_offset) +
		(size_t)output * MODEL_HIDDEN;
	const uint16_t *const input =
		(const uint16_t *)(arena + arguments.rows_offset) +
		(size_t)blockIdx.y * MODEL_HIDDEN;
	float sum = 0.0f;
	for (uint32_t k = threadIdx.x; k < MODEL_HIDDEN;
	     k += OFFICIAL_ROUTER_PROJECTION_OWNERS) {
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
	for (uint32_t distance =
		     OFFICIAL_ROUTER_PROJECTION_OWNERS / ROUTER_TREE_RADIX;
	     distance != 0; distance >>= 1u)
		sum = __fadd_rn(
			sum,
			__shfl_down_sync(ROUTER_EVERY_LANE, sum, distance,
					 OFFICIAL_ROUTER_PROJECTION_OWNERS));
	if (threadIdx.x == 0) {
		LINEAR_DEVICE_CHECK(isfinite(sum));
		((float *)(arena + arguments.output_offset))
			[(size_t)blockIdx.y * MODEL_EXPERTS + output] =
				__fadd_rn(0.0f, sum);
	}
}
