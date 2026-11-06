/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* The existing producer's compatible complex multiply, isolated with BF16
 * assignment instead of an FP16 cache carrier. One thread owns both adjacent
 * values; identical input/output therefore needs no cross-thread barrier.
 */
#include "kernel_linear_common.cuh"
#include "anemone/official_rope_records.h"

extern "C" __global__ void
official_rope(struct official_rope_arguments arguments, unsigned char *arena)
{
	LINEAR_DEVICE_CHECK(
		blockDim.x == OFFICIAL_ROPE_THREADS &&
		gridDim.x == arguments.heads && gridDim.y == arguments.rows &&
		arguments.reserved_zero == 0 && arguments.inverse <= 1 &&
		(arguments.width == OFFICIAL_ROPE_WIDTH ||
		 arguments.width == MODEL_HEAD_DIMS));
	const uint32_t pair = threadIdx.x;
	if (pair >= arguments.width / MODEL_PAIR_VALUES)
		return;
	const size_t first =
		((size_t)blockIdx.y * arguments.heads + blockIdx.x) *
			arguments.width +
		pair * MODEL_PAIR_VALUES;
	const uint16_t *const input =
		(const uint16_t *)(arena + arguments.input_offset);
	uint16_t *const output = (uint16_t *)(arena + arguments.output_offset);
	const uint32_t rotary_start =
		(arguments.width - OFFICIAL_ROPE_WIDTH) / MODEL_PAIR_VALUES;
	const uint16_t real_bits = input[first],
		       imaginary_bits = input[first + 1u];
	if (pair < rotary_start) {
		output[first] = real_bits;
		output[first + 1u] = imaginary_bits;
		return;
	}
	const float2 rotation = ((const float2 *)(arena +
						  arguments.rotation_offset))
		[(size_t)blockIdx.y * MODEL_FREQUENCIES + pair - rotary_start];
	const float real = linear_bfloat16_value(real_bits),
		    imaginary = linear_bfloat16_value(imaginary_bits);
	const float sine = arguments.inverse != 0 ? -rotation.y : rotation.y;
	output[first] = (uint16_t)linear_bfloat16_bits(
		fmaf(real, rotation.x, -__fmul_rn(imaginary, sine)));
	output[first + 1u] = (uint16_t)linear_bfloat16_bits(
		fmaf(imaginary, rotation.x, __fmul_rn(real, sine)));
}
