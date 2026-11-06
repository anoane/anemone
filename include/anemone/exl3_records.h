/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* EXL3 mul1, integer K2/K3. Trellis [input/16][output/16][16*K] uint16,
 * then FP16 suh[input], svh[output]. Matrices have whole 128-wide transforms.
 * The ring is MSB-first in little-endian uint32 words. No native matrix uses this layout. */
#ifndef EXL3_RECORDS_H
#define EXL3_RECORDS_H

#include <stdint.h>

#define EXL3_TILE 16u
#define EXL3_TILE_VALUES 256u
#define EXL3_TRANSFORM 128u
#define EXL3_MINIMUM_BITS 2u
#define EXL3_MAXIMUM_BITS 3u
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
#define EXL3_RECONSTRUCT "exl3_reconstruct"
#define EXL3_ROW_SUM "exl3_row_sum"

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
