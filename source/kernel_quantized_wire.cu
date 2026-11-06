/* SPDX-License-Identifier: AGPL-3.0-or-later */
#include "anemone/quantized_wire_records.h"
#include "kernel_activation_quantize.cuh"
#define WIRE_CODE_EXPONENT_SHIFT 3u
#define WIRE_CODE_FRACTION_MASK 7u
#define WIRE_CODE_EXPONENT_MASK 15u
#define WIRE_NORMAL_CARRIED_BIAS 93u
#define WIRE_SUBNORMAL_CARRIED_BIAS 91u
#define WIRE_BFLOAT_EXPONENT_SHIFT 7u
#define WIRE_NORMAL_MANTISSA_SHIFT 4u
#define WIRE_FLOAT_MOST_BIT 31u
#define WIRE_BFLOAT_MANTISSA_MASK 127u

extern "C" __global__ void quantized_wire(
	const struct quantized_wire_arguments __grid_constant__ arguments,
	unsigned char *arena)
{
	const uint32_t channel = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t width = arguments.local_width * QUANTIZED_WIRE_OWNERS;
	if (channel >= width || blockIdx.y >= arguments.pairs)
		return;
	const uint32_t owner = channel / QUANTIZED_WIRE_LOCAL_WIDTH;
	const uint32_t local = channel % QUANTIZED_WIRE_LOCAL_WIDTH;
	const uint32_t lane = local % VALUES_PER_BLOCK;
	uint32_t code_place = 0, operand_place = 0;
	const size_t local_word = quantize_word(local / VALUES_PER_BLOCK, lane,
						&code_place, &operand_place);
	const unsigned char *const wire =
		arena + arguments.input_offset +
		((size_t)blockIdx.y * QUANTIZED_WIRE_OWNERS + owner) *
			arguments.wire_pitch;
	const uint32_t code =
		wire[local_word * QUANTIZE_VALUES_PER_WORD + code_place];
	const uint32_t exponent =
		wire[QUANTIZED_WIRE_LOCAL_WIDTH + local / VALUES_PER_BLOCK];
	const uint32_t code_exponent = (code >> WIRE_CODE_EXPONENT_SHIFT) &
				       WIRE_CODE_EXPONENT_MASK;
	const uint32_t fraction = code & WIRE_CODE_FRACTION_MASK;
	uint32_t magnitude = 0;
	if (code_exponent != 0) {
		magnitude =
			((exponent + WIRE_NORMAL_CARRIED_BIAS + code_exponent)
			 << WIRE_BFLOAT_EXPONENT_SHIFT) |
			(fraction << WIRE_NORMAL_MANTISSA_SHIFT);
	} else if (fraction != 0) {
		const uint32_t most_bit =
			WIRE_FLOAT_MOST_BIT - (uint32_t)__clz(fraction);
		magnitude =
			((exponent + WIRE_SUBNORMAL_CARRIED_BIAS + most_bit)
			 << WIRE_BFLOAT_EXPONENT_SHIFT) |
			((fraction << (WIRE_BFLOAT_EXPONENT_SHIFT - most_bit)) &
			 WIRE_BFLOAT_MANTISSA_MASK);
	}
	const uint32_t sign = (code & QUANTIZE_E4M3_SIGN_BIT)
			      << QUANTIZE_CODE_SIGN_TO_BFLOAT16_SHIFT;
	const uint16_t operand =
		(uint16_t)((exponent <= ACTIVATION_CARRIED_EXPONENT_LARGEST ?
				    magnitude :
				    QUANTIZE_BFLOAT16_INFINITY) |
			   sign);
	const size_t global_word = quantize_word(
		channel / VALUES_PER_BLOCK, lane, &code_place, &operand_place);
	arena[arguments.codes_offset + (size_t)blockIdx.y * width +
	      global_word * QUANTIZE_VALUES_PER_WORD + code_place] =
		(unsigned char)code;
	((uint16_t *)(arena + arguments.operands_offset))
		[(size_t)blockIdx.y * width +
		 global_word * QUANTIZE_VALUES_PER_WORD + operand_place] =
			operand;
	if (lane == 0)
		arena[arguments.exponents_offset +
		      (size_t)blockIdx.y * width / VALUES_PER_BLOCK +
		      channel / VALUES_PER_BLOCK] = (unsigned char)exponent;
}

/* Independent producer control, retaining the original quantizer helper without cross-CTA arrival atomics. */
extern "C" __global__ void
quantize_wire(const struct quantize_wire_arguments __grid_constant__ arguments,
	      unsigned char *arena)
{
	const uint32_t lane = threadIdx.x % THREADS_PER_WARP;
	const uint32_t group = blockIdx.x * (blockDim.x / THREADS_PER_WARP) +
			       threadIdx.x / THREADS_PER_WARP;
	const uint32_t width = QUANTIZED_WIRE_LOCAL_WIDTH;
	const uint32_t groups = width / VALUES_PER_BLOCK;
	if (group >= groups || blockIdx.y >= arguments.pairs)
		return;
	const uint16_t value = ((const uint16_t *)(arena +
						   arguments.values_offset))
		[(size_t)blockIdx.y * width + group * VALUES_PER_BLOCK + lane];
	unsigned char *const wire = arena + arguments.wire_offset +
				    (size_t)blockIdx.y * QUANTIZED_WIRE_PITCH;
	const float input = linear_bfloat16_value(value);
	float largest = fabsf(input);
	for (uint32_t distance = THREADS_PER_WARP >> 1u; distance != 0;
	     distance >>= 1u)
		largest =
			fmaxf(largest, __shfl_xor_sync(QUANTIZE_EVERY_LANE,
						       largest, (int)distance));
	const uint32_t exponent = quantize_exponent_code(largest);
	uint32_t unused_operand = 0, code_place = 0, operand_place = 0;
	const uint32_t code =
		quantize_nearest_e4m3(input * quantize_inverse_scale(exponent),
				      exponent, &unused_operand);
	const size_t word =
		quantize_word(group, lane, &code_place, &operand_place);
	wire[word * QUANTIZE_VALUES_PER_WORD + code_place] =
		(unsigned char)code;
	if (lane == 0)
		wire[width + group] = (unsigned char)exponent;
	if (group + 1u == groups)
		for (uint32_t padding = groups + lane;
		     padding < QUANTIZED_WIRE_PITCH - width;
		     padding += THREADS_PER_WARP)
			wire[width + padding] = 0;
}
