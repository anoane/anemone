/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Independent in-place-capable source round trip. Each warp owns one complete
 * group32, reads all values before its collective maximum, then stores only its
 * own values. Reuses the existing activation encoder, preserving its tie order.
 */
#include "kernel_activation_quantize.cuh"
#include "anemone/official_roundtrip_records.h"

#define ROUNDTRIP_UNCARRY 0x1p-100f
#define ROUNDTRIP_DOMAIN_LIMIT 65536.0f
#define ROUNDTRIP_EVERY_LANE 0xFFFFFFFFu

extern "C" __global__ void
official_roundtrip_entry(struct official_roundtrip_arguments arguments,
			 unsigned char *arena)
{
	const uint32_t feature = blockIdx.x * blockDim.x + threadIdx.x;
	if (feature >= arguments.width)
		return;
	const size_t index = (size_t)blockIdx.y * arguments.width + feature;
	const float value = linear_bfloat16_value(
		((const uint16_t *)(arena + arguments.input_offset))[index]);
	LINEAR_DEVICE_CHECK(isfinite(value) &&
			    fabsf(value) <= ROUNDTRIP_DOMAIN_LIMIT);
	float largest = fabsf(value);
	for (uint32_t distance = THREADS_PER_WARP >> 1u; distance != 0;
	     distance >>= 1u)
		largest = fmaxf(largest, __shfl_xor_sync(ROUNDTRIP_EVERY_LANE,
							 largest, distance));
	const uint32_t exponent = quantize_exponent_code(largest);
	uint32_t operand = 0;
	(void)quantize_nearest_e4m3(value * quantize_inverse_scale(exponent),
				    exponent, &operand);
	const float decoded =
		linear_bfloat16_value(operand) * ROUNDTRIP_UNCARRY;
	((uint16_t *)(arena + arguments.output_offset))[index] =
		linear_bfloat16_bits(decoded);
}
