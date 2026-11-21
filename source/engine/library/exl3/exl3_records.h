/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* EXL3 mul1. Trellis [input/16][output/16][16*K] uint16, then FP16 suh[input], svh[output]. Matrices have whole
 * 128-wide transforms. The ring of a 16x16 tile is 256 K bits, MSB-first in little-endian uint32 words; the 16-bit
 * window of ring position i ends at ring bit S(i) (cyclic). No native matrix uses this layout.
 * Widths: a whole K steps K bits a position, S(i) = (i + 1) K. A half K = KA + 1/2 (exllamav3 1.6's fractional
 * trellis, the only pattern it writes: MASK 0xAAAA) steps KA bits at even positions and KA + 1 at odd ones, so
 * S(i) = floor((i + 1) bits_x2 / 2) for both kinds, bits_x2 = 2 K. A stored width (the pack's bits_per_weight and
 * every record's) is K for a whole K and 16 K for a half K (sixteenths of a bit, the tile's uint16 count), which no
 * whole K reaches. */
#ifndef EXL3_RECORDS_H
#define EXL3_RECORDS_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __CUDACC__
#define EXL3_WIDTH_FUNCTION __host__ __device__ static constexpr inline
#else
#define EXL3_WIDTH_FUNCTION static inline
#endif

#define EXL3_TILE 16u
#define EXL3_TILE_VALUES 256u
#define EXL3_TRANSFORM 128u
#define EXL3_WHOLE_WIDTH_MOST 8u /* the widest whole K the format defines */
#define EXL3_HALF_WIDTH_SCALE 16u /* a half K's stored width is 16 K */
#define EXL3_HALF_BITS_PER_BIT 2u /* bits_x2 = 2 K */
#define EXL3_HALF_WIDTH_RESIDUE  \
	(EXL3_HALF_WIDTH_SCALE / \
	 EXL3_HALF_BITS_PER_BIT) /* 16 K mod 16 of a half K */
#define EXL3_K2 2u /* stored widths of the kernel entries */
#define EXL3_K2_5 40u
#define EXL3_K3 3u
#define EXL3_K4 4u
#define EXL3_BITS_X2_K2 4u /* their bits_x2 */
#define EXL3_BITS_X2_K2_5 5u
#define EXL3_BITS_X2_K3 6u
#define EXL3_BITS_X2_K4 8u
/* The widths this build decodes, a row each, in the order every kernel family instantiates them:
 * ROW(the entries' width part, bits_x2, TC or NO_TC, ...), TC where the tensor-core MoE (kernel_exl3_tc,
 * kernel_exl3_tcp: SM_80 only, the TP route) has entries for the width: no K4 checkpoint fits the CMPs, so K4 takes the
 * other families. ROW takes the arguments after the table's as its own (at least one, empty allowed): EXL3_EVERY_WIDTH
 * and EXL3_TC_WIDTH call X(width part, bits_x2, ...) for each row or each TC row. A width is one row here and its two
 * named constants; one outside the table is refused at the pack's open (pack.h). */
#define EXL3_WIDTHS(ROW, ...)                         \
	ROW(k2, EXL3_BITS_X2_K2, TC, __VA_ARGS__)     \
	ROW(k2_5, EXL3_BITS_X2_K2_5, TC, __VA_ARGS__) \
	ROW(k3, EXL3_BITS_X2_K3, TC, __VA_ARGS__)     \
	ROW(k4, EXL3_BITS_X2_K4, NO_TC, __VA_ARGS__)
#define EXL3_EVERY_WIDTH(width, bits_x2, tc, X, ...) \
	X(width, bits_x2, __VA_ARGS__)
#define EXL3_TC_WIDTH(width, bits_x2, tc, X, ...) \
	EXL3_TC_WIDTH_##tc(X, width, bits_x2, __VA_ARGS__)
#define EXL3_TC_WIDTH_TC(X, width, bits_x2, ...) X(width, bits_x2, __VA_ARGS__)
#define EXL3_TC_WIDTH_NO_TC(X, width, bits_x2, ...)
#define EXL3_WIDTH_BIT(width, bits_x2, ...) | (1u << (bits_x2))
/* The widths this build decodes, a bit per bits_x2 */
#define EXL3_SUPPORTED_BITS_X2 \
	(0u EXL3_WIDTHS(EXL3_EVERY_WIDTH, EXL3_WIDTH_BIT, ))
/* The tensor-core MoE's widths */
#define EXL3_TC_SUPPORTED_BITS_X2 \
	(0u EXL3_WIDTHS(EXL3_TC_WIDTH, EXL3_WIDTH_BIT, ))
#define EXL3_BITS_PER_BYTE 8u
#define EXL3_HALF_BYTES 2u
#define EXL3_WORD_BITS 32u
#define EXL3_STATE_BITS 16u
#define EXL3_BYTE_MASK 255u
#define EXL3_HALF_MASK 65535u
#define EXL3_MUL1_MULTIPLIER 0x83DCD12Du
#define EXL3_MUL1_BASE 0x6400u
#define EXL3_MUL1_SCALE 0x1EEEu
#define EXL3_MUL1_BIAS 0xC931u
#define EXL3_RECORD_BYTES 32u
#define EXL3_KERNEL_FILE "kernel_exl3"
#define EXL3_KERNEL_FILE_NATIVE "kernel_exl3_native" /* class 120 native */
#define EXL3_RECONSTRUCT "exl3_reconstruct"
#define EXL3_ROW_SUM "exl3_row_sum"

/* bits_x2 of a stored width; 0 for a value that is no width the format defines (whole 1-8, half 1.5-7.5) */
EXL3_WIDTH_FUNCTION uint32_t exl3_bits_x2(uint32_t stored_width)
{
	if (stored_width >= 1u && stored_width <= EXL3_WHOLE_WIDTH_MOST)
		return EXL3_HALF_BITS_PER_BIT * stored_width;
	if (stored_width % EXL3_HALF_WIDTH_SCALE == EXL3_HALF_WIDTH_RESIDUE &&
	    stored_width < EXL3_WHOLE_WIDTH_MOST * EXL3_HALF_WIDTH_SCALE)
		return stored_width / EXL3_HALF_WIDTH_RESIDUE;
	return 0;
}

EXL3_WIDTH_FUNCTION bool exl3_width_supported(uint32_t stored_width)
{
	const uint32_t bits_x2 = exl3_bits_x2(stored_width);
	return bits_x2 != 0 && ((EXL3_SUPPORTED_BITS_X2 >> bits_x2) & 1u) != 0;
}

EXL3_WIDTH_FUNCTION bool exl3_tc_width_supported(uint32_t bits_x2)
{
	return bits_x2 != 0 &&
	       bits_x2 <= EXL3_HALF_BITS_PER_BIT * EXL3_WHOLE_WIDTH_MOST &&
	       ((EXL3_TC_SUPPORTED_BITS_X2 >> bits_x2) & 1u) != 0;
}

/* S(i): the ring bit where the window of position i ends (a whole K's product kept as K's own) */
EXL3_WIDTH_FUNCTION uint32_t exl3_window_end(uint32_t position,
					     uint32_t bits_x2)
{
	return bits_x2 % EXL3_HALF_BITS_PER_BIT == 0 ?
		       (position + 1u) * (bits_x2 / EXL3_HALF_BITS_PER_BIT) :
		       (position + 1u) * bits_x2 / EXL3_HALF_BITS_PER_BIT;
}

/* Code bytes of an inputs x outputs trellis: 16 x 16 tiles of 16 bits_x2 bytes (inputs, outputs whole tiles) */
EXL3_WIDTH_FUNCTION uint64_t exl3_code_bytes(uint64_t inputs, uint64_t outputs,
					     uint32_t bits_x2)
{
	return inputs * outputs /
	       ((uint64_t)EXL3_HALF_BITS_PER_BIT * EXL3_BITS_PER_BYTE) *
	       bits_x2;
}

/* Reconstruction is a diagnostic single pass: writes FP16 [input][output], before
 * either transform or scale. Both ranges are disjoint, validated by the host. */
struct exl3_reconstruct_record {
	uint64_t matrix_offset;
	uint64_t output_offset;
	uint32_t input_size;
	uint32_t output_size;
	uint32_t bits_per_weight;
	uint32_t reserved_zero;
};

/* Per-layer expert chain. Scratch is owned by the prepared card and reused only
 * after the previous layer completes. Products are FP32, operands FP16.
 * Pair output is FP32 [pair][input_size], with its route weight applied last.
 * It overwrites the down-product buffer after each independent 128-wide transform. */
#define EXL3_EXPERT_RECORD_BYTES 80u
#define EXL3_EXPERT_LAUNCHES 5u
#define EXL3_PRODUCT_SLICES 4u
#define EXL3_RELAXED_SLICES 8u
#define EXL3_WARP 32u
#define EXL3_NORMALIZATION 0.08838834764831845f
#define EXL3_PROJECTIONS 2u
#define EXL3_RADIX 2u
struct exl3_expert_record {
	uint64_t jobs_offset;
	uint64_t input_offset;
	uint64_t operands_offset;
	uint64_t products_offset;
	uint64_t hidden_offset;
	uint64_t output_offset;
	uint32_t input_size;
	uint32_t inner_size;
	uint32_t pair_count;
	uint32_t pairs_per_row;
	uint32_t job_count;
	uint32_t bits_per_weight;
	float activation_limit;
	uint32_t reserved_zero;
};

#ifdef __cplusplus
static_assert(sizeof(struct exl3_expert_record) == EXL3_EXPERT_RECORD_BYTES,
	      "EXL3 expert ABI");
static_assert(sizeof(struct exl3_reconstruct_record) == EXL3_RECORD_BYTES,
	      "EXL3 reconstruction ABI");
#else
_Static_assert(sizeof(struct exl3_expert_record) == EXL3_EXPERT_RECORD_BYTES,
	       "EXL3 expert ABI");
_Static_assert(sizeof(struct exl3_reconstruct_record) == EXL3_RECORD_BYTES,
	       "EXL3 reconstruction ABI");
#endif
#endif
