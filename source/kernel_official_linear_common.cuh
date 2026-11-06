/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Exact packed layouts and the observed finite instruction rounding. */
#ifndef KERNEL_OFFICIAL_LINEAR_COMMON_CUH
#define KERNEL_OFFICIAL_LINEAR_COMMON_CUH

#include <cuda_bf16.h>
#include <limits.h>
#include <stdint.h>

#define OFFICIAL_LINEAR_WARP 32u
#define OFFICIAL_LINEAR_ALL_LANES 0xffffffffu
#define OFFICIAL_LINEAR_PAIR 2u
#define OFFICIAL_LINEAR_PACK_WORD_BYTES 4u
#define OFFICIAL_LINEAR_PACK_BYTE_BITS 8u
#define OFFICIAL_LINEAR_PACK_HALF_WORD 16u
#define OFFICIAL_LINEAR_PACK_LOAD 64u
#define OFFICIAL_LINEAR_FP8_FRACTION_BITS 3u
#define OFFICIAL_LINEAR_FP8_FRACTION_MASK 7u
#define OFFICIAL_LINEAR_FP8_MAGNITUDE 127u
#define OFFICIAL_LINEAR_FP8_SIGN 128u
#define OFFICIAL_LINEAR_FP8_SUBNORMAL_STEP 0x1p-9f
#define OFFICIAL_LINEAR_FP8_TO_F32_BIAS 120u
#define OFFICIAL_LINEAR_FP8_TO_F32_FRACTION_SHIFT 20u
#define OFFICIAL_LINEAR_FP8_TO_F32_SIGN_SHIFT 24u
#define OFFICIAL_LINEAR_EXPONENT_MASK 255u
#define OFFICIAL_LINEAR_EXPONENT_BIAS 127
#define OFFICIAL_LINEAR_F32_FRACTION_BITS 23
#define OFFICIAL_LINEAR_F32_FRACTION_MASK 0x7fffffu
#define OFFICIAL_LINEAR_F32_HIDDEN 0x800000u
#define OFFICIAL_LINEAR_F32_EXPONENT_MASK 0x7f800000u
#define OFFICIAL_LINEAR_F32_SIGN 0x80000000u
#define OFFICIAL_LINEAR_F32_CANONICAL_NAN 0x7fc00000u
#define OFFICIAL_LINEAR_SCALE_ZERO_BITS 0x00400000u
#define OFFICIAL_LINEAR_BF16_FRACTION_BITS 7u
#define OFFICIAL_LINEAR_BF16_MAGNITUDE 0x7fffu
#define OFFICIAL_LINEAR_BF16_TO_F32_SHIFT 16u
#define OFFICIAL_LINEAR_ANCHOR_KEPT_BITS 25
#define OFFICIAL_LINEAR_INTEGER_BITS 64

__device__ static __forceinline__ float official_linear_fp8_value(uint8_t code)
{
	const uint32_t magnitude = code & OFFICIAL_LINEAR_FP8_MAGNITUDE;
	const uint32_t exponent = magnitude >>
				  OFFICIAL_LINEAR_FP8_FRACTION_BITS;
	const uint32_t fraction = magnitude & OFFICIAL_LINEAR_FP8_FRACTION_MASK;
	const float value =
		exponent == 0 ?
			__fmul_rn((float)fraction,
				  OFFICIAL_LINEAR_FP8_SUBNORMAL_STEP) :
			__uint_as_float(
				((exponent + OFFICIAL_LINEAR_FP8_TO_F32_BIAS)
				 << OFFICIAL_LINEAR_F32_FRACTION_BITS) |
				(fraction
				 << OFFICIAL_LINEAR_FP8_TO_F32_FRACTION_SHIFT));
	return __uint_as_float(__float_as_uint(value) |
			       ((uint32_t)(code & OFFICIAL_LINEAR_FP8_SIGN)
				<< OFFICIAL_LINEAR_FP8_TO_F32_SIGN_SHIFT));
}

__device__ static __forceinline__ float official_linear_scale(uint8_t code)
{
	return __uint_as_float(
		code == 0 ?
			OFFICIAL_LINEAR_SCALE_ZERO_BITS :
		code == OFFICIAL_LINEAR_EXPONENT_MASK ?
			OFFICIAL_LINEAR_F32_CANONICAL_NAN :
			(uint32_t)code << OFFICIAL_LINEAR_F32_FRACTION_BITS);
}

__device__ static __forceinline__ int32_t official_linear_exponent(float value)
{
	const uint32_t field =
		(__float_as_uint(value) >> OFFICIAL_LINEAR_F32_FRACTION_BITS) &
		OFFICIAL_LINEAR_EXPONENT_MASK;
	return (int32_t)(field != 0 ? field : 1u) -
	       OFFICIAL_LINEAR_EXPONENT_BIAS;
}

__device__ static __forceinline__ int64_t official_linear_units(float value,
								int32_t anchor)
{
	const uint32_t bits = __float_as_uint(value);
	const uint64_t significand =
		(bits & OFFICIAL_LINEAR_F32_FRACTION_MASK) |
		((bits & OFFICIAL_LINEAR_F32_EXPONENT_MASK) != 0 ?
			 OFFICIAL_LINEAR_F32_HIDDEN :
			 0u);
	const int32_t shift = official_linear_exponent(value) -
			      OFFICIAL_LINEAR_F32_FRACTION_BITS - anchor +
			      OFFICIAL_LINEAR_ANCHOR_KEPT_BITS;
	const uint64_t magnitude = shift >= 0 ? significand << (uint32_t)shift :
				   -shift >= OFFICIAL_LINEAR_INTEGER_BITS ?
						0u :
						significand >> (uint32_t)-shift;
	return (bits & OFFICIAL_LINEAR_F32_SIGN) != 0 ? -(int64_t)magnitude :
							(int64_t)magnitude;
}

__device__ static __forceinline__ uint32_t
official_linear_input_place(uint32_t inner_index)
{
	const uint32_t load = inner_index / OFFICIAL_LINEAR_PACK_LOAD;
	const uint32_t step = inner_index % OFFICIAL_LINEAR_PACK_LOAD /
			      OFFICIAL_LINEAR_PACK_HALF_WORD;
	const uint32_t lane = inner_index % OFFICIAL_LINEAR_PACK_BYTE_BITS /
			      OFFICIAL_LINEAR_PAIR;
	const uint32_t register_index = inner_index %
					OFFICIAL_LINEAR_PACK_HALF_WORD /
					OFFICIAL_LINEAR_PACK_BYTE_BITS;
	const uint32_t half = inner_index % OFFICIAL_LINEAR_PAIR;
	return ((load * OFFICIAL_LINEAR_PACK_WORD_BYTES + lane) *
			OFFICIAL_LINEAR_PACK_WORD_BYTES +
		step) * OFFICIAL_LINEAR_PACK_WORD_BYTES +
	       half * OFFICIAL_LINEAR_PAIR + register_index;
}

__device__ static __forceinline__ uint64_t official_linear_weight_place(
	uint32_t output_column, uint32_t inner_index, uint32_t scale_groups)
{
	const uint32_t lane = output_column % OFFICIAL_LINEAR_PACK_BYTE_BITS *
				      OFFICIAL_LINEAR_PACK_WORD_BYTES +
			      inner_index % OFFICIAL_LINEAR_PACK_BYTE_BITS /
				      OFFICIAL_LINEAR_PAIR;
	const uint32_t byte =
		inner_index % OFFICIAL_LINEAR_WARP /
			OFFICIAL_LINEAR_PACK_BYTE_BITS *
			OFFICIAL_LINEAR_PACK_WORD_BYTES +
		inner_index % OFFICIAL_LINEAR_PAIR * OFFICIAL_LINEAR_PAIR +
		output_column % OFFICIAL_LINEAR_PACK_HALF_WORD /
			OFFICIAL_LINEAR_PACK_BYTE_BITS;
	return (((uint64_t)(output_column / OFFICIAL_LINEAR_PACK_HALF_WORD) *
			 scale_groups +
		 inner_index / OFFICIAL_LINEAR_WARP) *
			OFFICIAL_LINEAR_WARP +
		lane) * OFFICIAL_LINEAR_PACK_HALF_WORD +
	       byte;
}

__device__ static __forceinline__ float
official_linear_dot32(uint8_t input_code, uint8_t weight_code)
{
	if (!__all_sync(OFFICIAL_LINEAR_ALL_LANES,
			(input_code & OFFICIAL_LINEAR_FP8_MAGNITUDE) !=
					OFFICIAL_LINEAR_FP8_MAGNITUDE &&
				(weight_code & OFFICIAL_LINEAR_FP8_MAGNITUDE) !=
					OFFICIAL_LINEAR_FP8_MAGNITUDE))
		return __uint_as_float(OFFICIAL_LINEAR_F32_CANONICAL_NAN);
	const float input = official_linear_fp8_value(input_code),
		    weight = official_linear_fp8_value(weight_code);
	int32_t anchor = (input_code & OFFICIAL_LINEAR_FP8_MAGNITUDE) != 0 &&
					 (weight_code &
					  OFFICIAL_LINEAR_FP8_MAGNITUDE) != 0 ?
				 official_linear_exponent(input) +
					 official_linear_exponent(weight) :
				 INT32_MIN;
	for (uint32_t distance = OFFICIAL_LINEAR_WARP / OFFICIAL_LINEAR_PAIR;
	     distance != 0; distance /= OFFICIAL_LINEAR_PAIR) {
		const int32_t other = __shfl_xor_sync(OFFICIAL_LINEAR_ALL_LANES,
						      anchor, (int)distance,
						      OFFICIAL_LINEAR_WARP);
		anchor = anchor > other ? anchor : other;
	}
	if (anchor == INT32_MIN)
		return 0.0f;
	int64_t sum = official_linear_units(__fmul_rn(input, weight), anchor);
	for (uint32_t distance = OFFICIAL_LINEAR_WARP / OFFICIAL_LINEAR_PAIR;
	     distance != 0; distance /= OFFICIAL_LINEAR_PAIR)
		sum += __shfl_xor_sync(OFFICIAL_LINEAR_ALL_LANES, sum,
				       (int)distance, OFFICIAL_LINEAR_WARP);
	return ldexpf(__ll2float_rz((long long)sum),
		      anchor - OFFICIAL_LINEAR_ANCHOR_KEPT_BITS);
}

#endif
