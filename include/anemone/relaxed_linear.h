/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * relaxed_linear.h: the launches of the trunk products of class 80 in the relaxed order (relaxed_linear_records.h),
 * each prepared once at load from a description, as linear.h prepares the reference order's.
 *
 * Contract:
 *   - linear.h's rules for prepare functions hold: nothing here runs per token, touches a card, allocates or reads
 *     an arena; a description is checked against every rule of relaxed_linear_records.h that can be checked without
 *     the arena's contents, a refusal is STATUS_INVALID_ARGUMENT behind one error line with the launches
 *     untouched; every range is an array of its own (prepare_common.h).
 *   - The form follows row_count: below RELAXED_WIDE_LEAST_ROWS the decode form, in the slice count of the
 *     matrices (one count of LINEAR_EVERY_SLICE_COUNT, at most the loads of the input width); from there the
 *     wide form, in tiles of RELAXED_WIDE_ROWS_SMALL rows (three blocks a multiprocessor; the instance of
 *     RELAXED_WIDE_ROWS_LARGE rows was slower at every row count measured), in k_split_count splits. The
 *     ordered instances (is_ordered: the reference order, bit for bit the products of linear.h) take the wide form
 *     at every row count above 8 and one split.
 *     A launch count of 1, or 2 where the splits are added by the reduce launch, is written to launch_count_out.
 *   - relaxed_fold_check is the check of relaxed_linear_records.h for one matrix: its scale codes (the
 *     bytes behind its codes in the pack, (output_size + 31) / 32 * input_size / 32 of them), read where they are
 *     on the host at load. INVALID_ARGUMENT, with one error line, where the fold would not be exact: the matrix is
 *     then served in the reference order. fold_reference_out is the bfloat16 rows' reference (unused by quantized
 *     rows, which take RELAXED_FOLD_REFERENCE).
 *   - Who selects: the model's prepared programs use the ordered wide instances on class 80 by default; the
 *     startup CMP-order option permits the relaxed wide and decode forms for matrices whose fold was checked.
 *     Other arithmetic classes keep linear.h's kernels. The matrix's wide_form is not used by this module;
 *     reserved fields remain zero, as linear.h requires.
 */
#ifndef RELAXED_LINEAR_H
#define RELAXED_LINEAR_H

#include <stdint.h>

#include "anemone/backend_seam.h"
#include "anemone/linear.h"
#include "anemone/relaxed_linear_records.h"
#include "anemone/status.h"

#define RELAXED_MOST_LAUNCHES \
	2u /* a product, and the reduce launch of its splits */

enum relaxed_tile {
	RELAXED_TILE_SMALL = 0,
	RELAXED_TILE_LARGE = 1,
	RELAXED_TILES = 2
};

/* The handles of the kernels of relaxed_linear_records.h on one card of class 80. */
struct relaxed_linear_kernels {
	uint32_t wide_quantized[RELAXED_TILES];
	uint32_t ordered_wide_quantized[RELAXED_TILES];
	uint32_t wide_bfloat16[RELAXED_TILES];
	uint32_t ordered_wide_bfloat16[RELAXED_TILES];
	uint32_t decode_quantized;
	uint32_t decode_bfloat16;
	uint32_t reduce;
};

/* Prepares every kernel, as linear_kernels_prepare does, on a card of class 80 (any other class is
 * STATUS_UNSUPPORTED: the module is built for sm_80 alone). */
MUST_CHECK result_status relaxed_linear_kernels_prepare(
	const struct backend_operations *backend, struct device_card *card,
	uint32_t arithmetic_class, struct relaxed_linear_kernels *kernels_out);

enum relaxed_rows_code {
	RELAXED_QUANTIZED_ROWS = 0,
	RELAXED_BFLOAT16_ROWS = 1
};

MUST_CHECK result_status relaxed_fold_check(enum relaxed_rows_code rows,
					    const uint8_t *scale_codes,
					    uint64_t scale_count,
					    int32_t *fold_reference_out);

/* y = W x of quantized rows for one matrix or two of one input width (attn.wq_a with attn.wkv; gate and up of the
 * shared expert). Of matrices and outputs the first matrix_count are read. */
struct relaxed_trunk_quantized_description {
	struct linear_matrix matrices
		[LINEAR_MERGED_MATRICES]; /* E4M3_SCALED, one slice_count */
	struct linear_range operands; /* read: operands [row_count][input_size] */
	struct linear_range outputs
		[LINEAR_MERGED_MATRICES]; /* written: [row_count][output_size] */
	struct linear_range
		partials; /* scratch with splits: float32 [k_split_count][row_count][outputs] */
	uint32_t matrix_count;
	uint32_t row_count;
	uint32_t k_split_count; /* of the wide form; 1 in the decode form */
	uint32_t is_ordered; /* 1: the ordered instance */
};
MUST_CHECK result_status relaxed_trunk_quantized_prepare(
	const struct relaxed_trunk_quantized_description *description,
	uint64_t arena_byte_count, const struct relaxed_linear_kernels *kernels,
	struct kernel_launch *launches_out, uint32_t *launch_count_out);

/* y = W x of bfloat16 rows in segment_count runs of outputs (attn.wo_a). */
struct relaxed_trunk_bfloat16_description {
	struct linear_matrix matrix; /* E4M3_SCALED */
	struct linear_range
		values; /* read: values [row_count][segment_count * input_size] */
	struct linear_range output; /* written: values [row_count][output_size] */
	struct linear_range
		partials; /* scratch with splits: float32 [k_split_count][row_count][output_size] */
	uint32_t row_count;
	uint32_t segment_count;
	uint32_t k_split_count;
	int32_t fold_reference; /* from relaxed_fold_check */
	uint32_t is_ordered; /* 1: the ordered instance */
};
MUST_CHECK result_status relaxed_trunk_bfloat16_prepare(
	const struct relaxed_trunk_bfloat16_description *description,
	uint64_t arena_byte_count, const struct relaxed_linear_kernels *kernels,
	struct kernel_launch *launches_out, uint32_t *launch_count_out);

/* A prepared launch of the wide form (either rows, either order) takes the cone of a decoder layer
 * (relaxed_linear_records.h's cone_layer): step, the step area of its row_count rows (its block, then its rows), and
 * cone_layer, a decoder layer (MODEL_DECODER_FIRST_LAYER to MODEL_LAYERS - 1). A launch of any other
 * kernel (the decode form, a reduce, trunk_mma's) is left as it is: it computes every row. */
MUST_CHECK result_status
relaxed_trunk_cone_set(const struct relaxed_linear_kernels *kernels,
		       struct linear_range step, uint32_t cone_layer,
		       uint64_t arena_byte_count, struct kernel_launch *launch);

#endif
