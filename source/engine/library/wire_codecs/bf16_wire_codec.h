/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Lossless integer leaves for a BF16 wire codec. Eight values share a warp-wide top exponent. A lane whose
 * exponents differ from that top by more than fifteen sends its four original words instead. The compressed
 * lane holds eight sign/mantissa bytes and eight exponent nibbles: three words instead of four.
 * These leaves preserve every BF16 bit, including signed zeros and NaN payloads. They perform no arithmetic.
 * Caller proves top >= both exponents and deltas <= fifteen before encoding; decoding uses that same top. */
#ifndef BF16_WIRE_CODEC_H
#define BF16_WIRE_CODEC_H

#include <stdint.h>

#define BF16_WIRE_EXPONENT_MASK 255u
#define BF16_WIRE_MANTISSA_MASK 127u
#define BF16_WIRE_BYTE_SIGN 128u
#define BF16_WIRE_EXPONENT_SHIFT 7u
#define BF16_WIRE_HALF_BITS 16u
#define BF16_WIRE_BYTE_BITS 8u
#define BF16_WIRE_DELTA_BITS 4u
#define BF16_WIRE_DELTA_MASK 15u

#if defined(__CUDACC__)
#define BF16_WIRE_INLINE static __host__ __device__ __forceinline__
#else
#define BF16_WIRE_INLINE static inline
#endif

BF16_WIRE_INLINE uint32_t bf16_wire_low_exponent(uint32_t word)
{
	return (word >> BF16_WIRE_EXPONENT_SHIFT) & BF16_WIRE_EXPONENT_MASK;
}

BF16_WIRE_INLINE uint32_t bf16_wire_high_exponent(uint32_t word)
{
	return bf16_wire_low_exponent(word >> BF16_WIRE_HALF_BITS);
}

/* Two sign/mantissa bytes in the low half, two exponent deltas in the next byte. */
BF16_WIRE_INLINE uint32_t bf16_wire_pair_pack(uint32_t word, uint32_t top)
{
	const uint32_t low =
		(word & BF16_WIRE_MANTISSA_MASK) |
		((word >> BF16_WIRE_BYTE_BITS) & BF16_WIRE_BYTE_SIGN);
	const uint32_t high_word = word >> BF16_WIRE_HALF_BITS;
	const uint32_t high =
		(high_word & BF16_WIRE_MANTISSA_MASK) |
		((high_word >> BF16_WIRE_BYTE_BITS) & BF16_WIRE_BYTE_SIGN);
	const uint32_t deltas =
		(top - bf16_wire_low_exponent(word)) |
		((top - bf16_wire_high_exponent(word)) << BF16_WIRE_DELTA_BITS);
	return low | (high << BF16_WIRE_BYTE_BITS) |
	       (deltas << BF16_WIRE_HALF_BITS);
}

BF16_WIRE_INLINE uint32_t bf16_wire_pair_unpack(uint32_t mantissas,
						uint32_t deltas, uint32_t top)
{
	const uint32_t low =
		(mantissas & BF16_WIRE_MANTISSA_MASK) |
		((mantissas & BF16_WIRE_BYTE_SIGN) << BF16_WIRE_BYTE_BITS) |
		((top - (deltas & BF16_WIRE_DELTA_MASK))
		 << BF16_WIRE_EXPONENT_SHIFT);
	mantissas >>= BF16_WIRE_BYTE_BITS;
	deltas >>= BF16_WIRE_DELTA_BITS;
	const uint32_t high =
		(mantissas & BF16_WIRE_MANTISSA_MASK) |
		((mantissas & BF16_WIRE_BYTE_SIGN) << BF16_WIRE_BYTE_BITS) |
		((top - (deltas & BF16_WIRE_DELTA_MASK))
		 << BF16_WIRE_EXPONENT_SHIFT);
	return low | (high << BF16_WIRE_HALF_BITS);
}

#undef BF16_WIRE_INLINE
#endif
