/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Packed original E4M3 K32 projections. Independent portable and native
 * entries retain the same record and separate source scale operations. */
#include "anemone/official_linear_records.h"
#include "kernel_official_linear_common.cuh"
#define OFFICIAL_LINEAR_NATIVE_WARP_OUTPUTS 8u
#define OFFICIAL_LINEAR_NATIVE_HALF_ROWS 8u
#define OFFICIAL_LINEAR_UPPER_FIRST 2u
#define OFFICIAL_LINEAR_UPPER_SECOND 3u
#define OFFICIAL_LINEAR_NATIVE_ARCHITECTURE 1200
extern "C" __global__ void
official_linear(struct trunk_linear_quantized_arguments arguments,
		unsigned char *arena)
{
	const uint32_t lane_index = threadIdx.x % OFFICIAL_LINEAR_WARP,
		       output_column =
			       blockIdx.x * OFFICIAL_LINEAR_PORTABLE_OUTPUTS +
			       threadIdx.x / OFFICIAL_LINEAR_WARP;
	const uint32_t matrix_index =
		output_column < arguments.output_size[0] ? 0u : 1u;
	const uint32_t matrix_column = output_column -
				       (matrix_index == 0 ?
						0u :
						arguments.output_size[0]),
		       row_index = blockIdx.y;
	if (row_index >= arguments.row_count ||
	    matrix_column >= (matrix_index == 0 ? arguments.output_size[0] :
						  arguments.output_size[1]))
		return;
	const uint32_t scale_groups =
		arguments.input_size / OFFICIAL_LINEAR_WARP;
	const uint8_t *weights = arena + (matrix_index == 0 ?
						  arguments.weights_offset[0] :
						  arguments.weights_offset[1]);
	const uint8_t *input_codes = arena + arguments.codes_offset +
				     (uint64_t)row_index * arguments.input_size;
	const uint8_t *input_scales = arena + arguments.exponents_offset +
				      (uint64_t)row_index * scale_groups;
	const uint8_t *weight_scales =
		weights + (uint64_t)(matrix_index == 0 ?
					     arguments.output_size[0] :
					     arguments.output_size[1]) *
				  arguments.input_size;
	float accumulator = 0.0f;
	for (uint32_t group = 0; group < scale_groups; group++) {
		const uint32_t inner_index =
			group * OFFICIAL_LINEAR_WARP + lane_index;
		const float dot = official_linear_dot32(
			input_codes[official_linear_input_place(inner_index)],
			weights[official_linear_weight_place(
				matrix_column, inner_index, scale_groups)]);
		if (lane_index == 0)
			accumulator = __fadd_rn(
				accumulator,
				__fmul_rn(
					__fmul_rn(dot,
						  official_linear_scale(
							  input_scales[group])),
					official_linear_scale(
						weight_scales
							[(uint64_t)(matrix_column /
								    OFFICIAL_LINEAR_WARP) *
								 scale_groups +
							 group])));
	}
	if (lane_index == 0)
		((uint16_t *)(arena + (matrix_index == 0 ?
					       arguments.output_offset[0] :
					       arguments.output_offset[1])))
			[(uint64_t)row_index *
				 (matrix_index == 0 ? arguments.output_size[0] :
						      arguments.output_size[1]) +
			 matrix_column] =
				__bfloat16_as_ushort(
					__float2bfloat16_rn(accumulator));
}

#if defined(__CUDA_ARCH__) && \
	__CUDA_ARCH__ >= OFFICIAL_LINEAR_NATIVE_ARCHITECTURE
__device__ static __forceinline__ uint32_t
official_linear_input_word(const uint8_t *input, uint32_t inner_index)
{
	uint32_t word = 0;
#pragma unroll
	for (uint32_t byte_index = 0;
	     byte_index < OFFICIAL_LINEAR_PACK_WORD_BYTES; byte_index++)
		word |= (uint32_t)input[official_linear_input_place(
				inner_index + byte_index)]
			<< (OFFICIAL_LINEAR_PACK_BYTE_BITS * byte_index);
	return word;
}
__device__ static __forceinline__ uint32_t
official_linear_weight_word(const uint8_t *weights, uint32_t output_column,
			    uint32_t inner_index, uint32_t scale_groups)
{
	uint32_t word = 0;
#pragma unroll
	for (uint32_t byte_index = 0;
	     byte_index < OFFICIAL_LINEAR_PACK_WORD_BYTES; byte_index++)
		word |= (uint32_t)weights[official_linear_weight_place(
				output_column, inner_index + byte_index,
				scale_groups)]
			<< (OFFICIAL_LINEAR_PACK_BYTE_BITS * byte_index);
	return word;
}
extern "C" __global__ void
official_linear_native(struct trunk_linear_quantized_arguments arguments,
		       unsigned char *arena)
{
	const uint32_t lane_index = threadIdx.x % OFFICIAL_LINEAR_WARP,
		       row_owner = lane_index / OFFICIAL_LINEAR_PACK_WORD_BYTES,
		       column_owner =
			       lane_index % OFFICIAL_LINEAR_PACK_WORD_BYTES;
	const uint32_t output_column =
		blockIdx.x * OFFICIAL_LINEAR_NATIVE_OUTPUTS +
		(threadIdx.x / OFFICIAL_LINEAR_WARP) *
			OFFICIAL_LINEAR_NATIVE_WARP_OUTPUTS;
	const uint32_t matrix_index =
		output_column < arguments.output_size[0] ? 0u : 1u;
	const uint32_t output_start =
		output_column -
		(matrix_index == 0 ? 0u : arguments.output_size[0]);
	const uint32_t output_count = arguments.output_size[matrix_index],
		       first_row = blockIdx.y * OFFICIAL_LINEAR_NATIVE_ROWS;
	const uint32_t scale_groups =
		arguments.input_size / OFFICIAL_LINEAR_WARP;
	const uint8_t *weights = arena + arguments.weights_offset[matrix_index];
	const uint8_t *scales =
		weights + (uint64_t)output_count * arguments.input_size;
	const uint8_t *codes = arena + arguments.codes_offset;
	const uint8_t *exponents = arena + arguments.exponents_offset;
	float accumulated[OFFICIAL_LINEAR_PACK_WORD_BYTES] = {};
	for (uint32_t group = 0; group < scale_groups; group++) {
		const uint32_t inner_index =
			group * OFFICIAL_LINEAR_WARP +
			column_owner * OFFICIAL_LINEAR_PACK_WORD_BYTES;
		uint32_t input_fragment[OFFICIAL_LINEAR_PACK_WORD_BYTES] = {},
			 weight_fragment[OFFICIAL_LINEAR_PAIR] = {};
		if (first_row + row_owner < arguments.row_count) {
			const uint8_t *row =
				codes + (uint64_t)(first_row + row_owner) *
						arguments.input_size;
			input_fragment[0] =
				official_linear_input_word(row, inner_index);
			input_fragment[OFFICIAL_LINEAR_UPPER_FIRST] =
				official_linear_input_word(
					row,
					inner_index +
						OFFICIAL_LINEAR_NATIVE_ROWS);
		}
		if (first_row + row_owner + OFFICIAL_LINEAR_NATIVE_HALF_ROWS <
		    arguments.row_count) {
			const uint8_t *row =
				codes +
				(uint64_t)(first_row + row_owner +
					   OFFICIAL_LINEAR_NATIVE_HALF_ROWS) *
					arguments.input_size;
			input_fragment[1] =
				official_linear_input_word(row, inner_index);
			input_fragment[OFFICIAL_LINEAR_UPPER_SECOND] =
				official_linear_input_word(
					row,
					inner_index +
						OFFICIAL_LINEAR_NATIVE_ROWS);
		}
		if (output_start + row_owner < output_count) {
			weight_fragment[0] = official_linear_weight_word(
				weights, output_start + row_owner, inner_index,
				scale_groups);
			weight_fragment[1] = official_linear_weight_word(
				weights, output_start + row_owner,
				inner_index + OFFICIAL_LINEAR_NATIVE_ROWS,
				scale_groups);
		}
		float dot[OFFICIAL_LINEAR_PACK_WORD_BYTES] = {};
		asm volatile(
			"mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3},"
			"{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
			: "+f"(dot[0]), "+f"(dot[1]),
			  "+f"(dot[OFFICIAL_LINEAR_UPPER_FIRST]),
			  "+f"(dot[OFFICIAL_LINEAR_UPPER_SECOND])
			: "r"(input_fragment[0]), "r"(input_fragment[1]),
			  "r"(input_fragment[OFFICIAL_LINEAR_UPPER_FIRST]),
			  "r"(input_fragment[OFFICIAL_LINEAR_UPPER_SECOND]),
			  "r"(weight_fragment[0]), "r"(weight_fragment[1]));
#pragma unroll
		for (uint32_t fragment_index = 0;
		     fragment_index < OFFICIAL_LINEAR_PACK_WORD_BYTES;
		     fragment_index++) {
			const uint32_t row =
				first_row + row_owner +
				fragment_index / OFFICIAL_LINEAR_PAIR *
					OFFICIAL_LINEAR_NATIVE_HALF_ROWS;
			const uint32_t output =
				output_start +
				column_owner * OFFICIAL_LINEAR_PAIR +
				fragment_index % OFFICIAL_LINEAR_PAIR;
			if (row < arguments.row_count &&
			    output < output_count) {
				const float scaled = __fmul_rn(
					__fmul_rn(
						dot[fragment_index],
						official_linear_scale(
							exponents[(uint64_t)row *
									  scale_groups +
								  group])),
					official_linear_scale(
						scales[(uint64_t)(output /
								  OFFICIAL_LINEAR_WARP) *
							       scale_groups +
						       group]));
				accumulated[fragment_index] = __fadd_rn(
					accumulated[fragment_index], scaled);
			}
		}
	}
#pragma unroll
	for (uint32_t fragment_index = 0;
	     fragment_index < OFFICIAL_LINEAR_PACK_WORD_BYTES;
	     fragment_index++) {
		const uint32_t row = first_row + row_owner +
				     fragment_index / OFFICIAL_LINEAR_PAIR *
					     OFFICIAL_LINEAR_NATIVE_HALF_ROWS;
		const uint32_t output = output_start +
					column_owner * OFFICIAL_LINEAR_PAIR +
					fragment_index % OFFICIAL_LINEAR_PAIR;
		if (row < arguments.row_count && output < output_count)
			((uint16_t *)(arena +
				      arguments.output_offset[matrix_index]))
				[(uint64_t)row * output_count + output] =
					__bfloat16_as_ushort(__float2bfloat16_rn(
						accumulated[fragment_index]));
	}
}

#endif
