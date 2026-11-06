/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Grouped WoA: packed E4M3 plus scale becomes BF16 before the K16
 * operand-anchor update. F32 carry participates at every update. */
#include "anemone/official_linear_records.h"
#include "kernel_official_linear_common.cuh"
#define OFFICIAL_LINEAR_WOA_LANES 16u
#define OFFICIAL_LINEAR_WOA_SCALE_GROUPS \
	(OFFICIAL_LINEAR_WOA_INNER / OFFICIAL_LINEAR_WARP)
#define OFFICIAL_LINEAR_WOA_WEIGHT_VALUES                                     \
	((uint64_t)OFFICIAL_LINEAR_WOA_GROUPS * OFFICIAL_LINEAR_WOA_COLUMNS * \
	 OFFICIAL_LINEAR_WOA_INNER)
__device__ static __forceinline__ int32_t
official_linear_bfloat16_exponent(uint16_t value)
{
	return official_linear_exponent(__uint_as_float(
		(uint32_t)value << OFFICIAL_LINEAR_BF16_TO_F32_SHIFT));
}
__device__ static __forceinline__ uint16_t official_linear_woa_weight(
	const uint8_t *weights, uint32_t output_column, uint32_t inner_index)
{
	const uint64_t place = official_linear_weight_place(
		output_column, inner_index, OFFICIAL_LINEAR_WOA_SCALE_GROUPS);
	const uint8_t scale =
		weights[OFFICIAL_LINEAR_WOA_WEIGHT_VALUES +
			(uint64_t)(output_column / OFFICIAL_LINEAR_WARP) *
				OFFICIAL_LINEAR_WOA_SCALE_GROUPS +
			inner_index / OFFICIAL_LINEAR_WARP];
	return __bfloat16_as_ushort(__float2bfloat16_rn(
		__fmul_rn(official_linear_fp8_value(weights[place]),
			  official_linear_scale(scale))));
}
extern "C" __global__ void
official_linear_woa(struct trunk_linear_bfloat16_arguments arguments,
		    unsigned char *arena)
{
	const uint32_t lane_index = threadIdx.x % OFFICIAL_LINEAR_WOA_LANES;
	const uint32_t output_column =
		blockIdx.x * OFFICIAL_LINEAR_WOA_OUTPUTS +
		threadIdx.x / OFFICIAL_LINEAR_WOA_LANES;
	const uint32_t row_index = blockIdx.y, group_index = blockIdx.z;
	const uint32_t weight_row =
		group_index * OFFICIAL_LINEAR_WOA_COLUMNS + output_column;
	const uint16_t *rows =
		(const uint16_t *)(arena + arguments.values_offset);
	const uint8_t *weights = arena + arguments.weights_offset;
	float accumulator = 0.0f;
	for (uint32_t first = 0; first < OFFICIAL_LINEAR_WOA_INNER;
	     first += OFFICIAL_LINEAR_WOA_LANES) {
		const uint32_t inner_index = first + lane_index;
		const uint16_t input_bits =
			rows[((uint64_t)row_index * OFFICIAL_LINEAR_WOA_GROUPS +
			      group_index) *
				     OFFICIAL_LINEAR_WOA_INNER +
			     inner_index];
		const uint16_t weight_bits = official_linear_woa_weight(
			weights, weight_row, inner_index);
		const float product = __fmul_rn(
			__uint_as_float((uint32_t)input_bits
					<< OFFICIAL_LINEAR_BF16_TO_F32_SHIFT),
			__uint_as_float((uint32_t)weight_bits
					<< OFFICIAL_LINEAR_BF16_TO_F32_SHIFT));
		int32_t anchor =
			(input_bits & OFFICIAL_LINEAR_BF16_MAGNITUDE) != 0 &&
					(weight_bits &
					 OFFICIAL_LINEAR_BF16_MAGNITUDE) != 0 ?
				official_linear_bfloat16_exponent(input_bits) +
					official_linear_bfloat16_exponent(
						weight_bits) :
				INT32_MIN;
		for (uint32_t distance =
			     OFFICIAL_LINEAR_WOA_LANES / OFFICIAL_LINEAR_PAIR;
		     distance != 0; distance /= OFFICIAL_LINEAR_PAIR) {
			const int32_t other = __shfl_xor_sync(
				OFFICIAL_LINEAR_ALL_LANES, anchor,
				(int)distance, OFFICIAL_LINEAR_WOA_LANES);
			anchor = anchor > other ? anchor : other;
		}
		const int32_t carry =
			accumulator != 0.0f ?
				official_linear_exponent(accumulator) :
				INT32_MIN;
		anchor = anchor > carry ? anchor : carry;
		if (anchor != INT32_MIN) {
			int64_t sum = official_linear_units(product, anchor);
			for (uint32_t distance = OFFICIAL_LINEAR_WOA_LANES /
						 OFFICIAL_LINEAR_PAIR;
			     distance != 0; distance /= OFFICIAL_LINEAR_PAIR)
				sum += __shfl_xor_sync(
					OFFICIAL_LINEAR_ALL_LANES, sum,
					(int)distance,
					OFFICIAL_LINEAR_WOA_LANES);
			sum += official_linear_units(accumulator, anchor);
			accumulator = ldexpf(
				__ll2float_rz((long long)sum),
				anchor - OFFICIAL_LINEAR_ANCHOR_KEPT_BITS);
		}
	}
	if (lane_index == 0)
		((uint16_t *)(arena + arguments.output_offset))
			[((uint64_t)row_index * OFFICIAL_LINEAR_WOA_GROUPS +
			  group_index) *
				 OFFICIAL_LINEAR_WOA_COLUMNS +
			 output_column] =
				__bfloat16_as_ushort(
					__float2bfloat16_rn(accumulator));
}
