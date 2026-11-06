/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Load-time EXL3 mul1 K2/K3 slices. Uses the engine's existing storage ABI. */
#ifndef BB_EXL3_LAYOUT_H
#define BB_EXL3_LAYOUT_H

#include "anemone/exl3_records.h"
#include "anemone/status.h"

#include <stddef.h>

#define BB_EXL3_TP_RANKS 4u
#define BB_EXL3_MATRIX_SLICE_BYTES 32u
#define BB_EXL3_OUTPUT_SHARD_PLAN_BYTES 80u

/* Rows are input channels and columns are output channels. Origins and extents
 * preserve whole independent 128-channel transforms. Packed codes remain
 * [input/16][output/16][16*K] uint16, followed by suh[input], svh[output]. */
struct bb_exl3_matrix_slice {
	uint32_t input_size;
	uint32_t output_size;
	uint32_t input_begin;
	uint32_t input_count;
	uint32_t output_begin;
	uint32_t output_count;
	uint32_t bits_per_weight;
	uint32_t reserved_zero;
};

struct bb_exl3_slice_sizes {
	uint64_t source_bytes;
	uint64_t packed_bytes;
};

/* Validate before allocation. Refusal leaves sizes_out unchanged. Input shapes
 * must be positive, transform-aligned K2/K3. Both origins/counts must fit their
 * full dimensions; zero counts and nonzero reserved fields are refused. */
MUST_CHECK result_status
bb_exl3_matrix_slice_prepare(const struct bb_exl3_matrix_slice *slice,
			     struct bb_exl3_slice_sizes *sizes_out);

/* Lossless CPU load-time copy, with no decode, floating arithmetic or allocation.
 * Source/destination extents describe valid host objects and must not overlap.
 * Entire sources are required; destination may have extra untouched capacity.
 * This validates the description and bounds before writing. It does not verify
 * model provenance, finite scale domains or source hashes: the loader owns them.
 * Arena placement/padding to the engine's matrix alignment remains caller-owned. */
MUST_CHECK result_status bb_exl3_matrix_slice_copy(
	const struct bb_exl3_matrix_slice *slice, const void *source,
	size_t source_bytes, void *destination, size_t destination_bytes);

enum bb_exl3_product_order {
	BB_EXL3_REFERENCE_ORDER = 0,
	BB_EXL3_EIGHT_PART_ORDER = 1
};

/* PP uses rank_count=1. TP4 gives each matrix output shard its complete K
 * dimension, preserving the reference's four contiguous MMA accumulators.
 * Gate/up share one slice shape. Down is [full intermediate][outer output shard].
 * No weight/scalar arithmetic changes during packing. EIGHT_PART is explicitly
 * optional and must retain a separate measured accuracy/speed qualification. */
struct bb_exl3_output_shard_plan {
	struct bb_exl3_matrix_slice gate_up;
	struct bb_exl3_matrix_slice down;
	uint32_t rank_count;
	uint32_t rank;
	uint32_t product_slices;
	enum bb_exl3_product_order order;
};

_Static_assert(sizeof(struct bb_exl3_matrix_slice) ==
		       BB_EXL3_MATRIX_SLICE_BYTES,
	       "EXL3 matrix slice ABI");
_Static_assert(sizeof(struct bb_exl3_output_shard_plan) ==
		       BB_EXL3_OUTPUT_SHARD_PLAN_BYTES,
	       "EXL3 output shard plan ABI");

/* Creates immutable metadata only. Leaves plan_out unchanged on refusal.
 * Supports one rank or four ranks; each rank must receive a nonempty 128-wide
 * block. Allocation, launches, CUDA context, peer qualification and transport are
 * deliberately caller-owned. This is not admission of the old five-pass chain
 * on local dimensions: that chain assumes one common full-matrix shape.
 *
 * TP execution contract:
 *  - replicate original BF16 expert inputs and retain original route/job order;
 *  - full-K gate/up products retain their four global slices per output;
 *  - local activate and down-input 128-wide transform produce original FP16 bits;
 *  - losslessly gather those FP16 bits into full-intermediate pair-major operands;
 *  - full-K down output shards retain four global slices, then local finish;
 *  - sum routed pairs locally in original ascending slot order, then losslessly
 *    gather the summed row channel shards before original shared-expert/post.
 * Gathering finished pair channel shards before row-sum is also correct but
 * moves pairs_per_row times more FP32 values. Row-sum never mixes channels.
 * Gather has no reduction. The existing collective library supplies transport.
 * Jobs, weights, input and plan remain immutable; every flight owns disjoint
 * scratch/gather storage until all local and remote readers have completed.
 * Do not all-reduce local finished inner-shard products in reference mode.
 * End-to-end exactness still requires component and reference state/logit gates. */
MUST_CHECK result_status bb_exl3_output_shard_prepare(
	uint32_t outer_size, uint32_t inner_size, uint32_t bits_per_weight,
	uint32_t rank_count, uint32_t rank, enum bb_exl3_product_order order,
	struct bb_exl3_output_shard_plan *plan_out);

#endif
