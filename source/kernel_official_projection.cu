/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Independent F32-weight/BF16-input compressor projection; fixed32 RN FMA
 * owners and16/8/4/2/1 fold. The host admits only5120x512, scale1, no flags. */
#include "anemone/dense_linear_records.h"
#include <cuda_bf16.h>
#define OFFICIAL_PROJECTION_OWNERS 32u
#define OFFICIAL_PROJECTION_OUTPUTS 4u
#define OFFICIAL_PROJECTION_PAIR 2u
#define OFFICIAL_PROJECTION_BFLOAT16_SHIFT 16u
#define OFFICIAL_PROJECTION_EVERY_LANE 0xffffffffu
extern "C" __global__ void
official_projection_float32(struct dense_linear_arguments arguments,
			    unsigned char *arena)
{
	const uint32_t row = blockIdx.y,
		       output = blockIdx.x * OFFICIAL_PROJECTION_OUTPUTS +
				threadIdx.y;
	const uint16_t *input =
		(const uint16_t *)(arena + arguments.rows_offset) +
		(uint64_t)row * arguments.input_size;
	const float *weight =
		(const float *)(arena + arguments.weights_offset) +
		(uint64_t)output * arguments.input_size;
	float sum = 0.0f;
	for (uint32_t inner_offset = threadIdx.x;
	     inner_offset < arguments.input_size;
	     inner_offset += OFFICIAL_PROJECTION_OWNERS)
		sum = __fmaf_rn(
			weight[inner_offset],
			__uint_as_float((uint32_t)input[inner_offset]
					<< OFFICIAL_PROJECTION_BFLOAT16_SHIFT),
			sum);
	sum = __fadd_rn(0.0f, sum);
	for (uint32_t distance =
		     OFFICIAL_PROJECTION_OWNERS / OFFICIAL_PROJECTION_PAIR;
	     distance != 0; distance /= OFFICIAL_PROJECTION_PAIR)
		sum = __fadd_rn(sum,
				__shfl_down_sync(OFFICIAL_PROJECTION_EVERY_LANE,
						 sum, distance,
						 OFFICIAL_PROJECTION_OWNERS));
	if (threadIdx.x == 0)
		((float *)(arena + arguments.output_offset))
			[(uint64_t)row * arguments.output_size + output] =
				__fadd_rn(0.0f, sum);
}

extern "C" __global__ void
official_projection_bfloat16(struct dense_linear_arguments arguments,
			     unsigned char *arena)
{
	const uint32_t row = blockIdx.y,
		       output = blockIdx.x * OFFICIAL_PROJECTION_OUTPUTS +
				threadIdx.y;
	const uint16_t *input =
		(const uint16_t *)(arena + arguments.rows_offset) +
		(uint64_t)row * arguments.input_size;
	const uint16_t *weight =
		(const uint16_t *)(arena + arguments.weights_offset) +
		(uint64_t)output * arguments.input_size;
	float sum = 0.0f;
	for (uint32_t inner_offset = threadIdx.x;
	     inner_offset < arguments.input_size;
	     inner_offset += OFFICIAL_PROJECTION_OWNERS)
		sum = __fmaf_rn(
			__uint_as_float((uint32_t)weight[inner_offset]
					<< OFFICIAL_PROJECTION_BFLOAT16_SHIFT),
			__uint_as_float((uint32_t)input[inner_offset]
					<< OFFICIAL_PROJECTION_BFLOAT16_SHIFT),
			sum);
	sum = __fadd_rn(0.0f, sum);
	for (uint32_t distance =
		     OFFICIAL_PROJECTION_OWNERS / OFFICIAL_PROJECTION_PAIR;
	     distance != 0; distance /= OFFICIAL_PROJECTION_PAIR)
		sum = __fadd_rn(sum,
				__shfl_down_sync(OFFICIAL_PROJECTION_EVERY_LANE,
						 sum, distance,
						 OFFICIAL_PROJECTION_OWNERS));
	if (threadIdx.x == 0) {
		const uint64_t index =
			(uint64_t)row * arguments.output_size + output;
		const float result = __fadd_rn(0.0f, sum);
		if (arguments.output_type == DENSE_BFLOAT16)
			((uint16_t *)(arena + arguments.output_offset))[index] =
				__bfloat16_as_ushort(
					__float2bfloat16_rn(result));
		else
			((float *)(arena + arguments.output_offset))[index] =
				result;
	}
}
