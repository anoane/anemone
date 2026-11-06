/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * relaxed_linear_records.h: the records of the trunk products of class 80 in the RELAXED ORDER
 * (source/kernel_trunk_relaxed.cu), and the layout of every array they name. C11 and CUDA, of the engine
 * linear_records.h alone.
 *
 * Contract:
 *   - linear_records.h's rules hold where this header says nothing else: offsets into the arena that whoever
 *     prepares a launch has checked, its arrays (operands, values, a matrix stored as
 *     WEIGHT_FORMAT_E4M3_SCALED), its quantization, no atomic operation, nothing that depends on timing.
 *   - The relaxed order (the owner, 26 October 2026: decode_accuracy may relax the CMPs' order of additions; a
 *     startup knob forces the reference order back). A product keeps the operands of the reference: the weights'
 *     E4M3 values and UE8M0 scales, the rows' E4M3 codes and exponents (the carried operands of linear_records.h),
 *     or a bfloat16 row's 16-bit float; only the order of the additions changes:
 *       - each weight's scale is folded into its operand, an exact multiplication by a power of two: the
 *         quantized rows take bfloat16 operands, value times 2^(w - RELAXED_FOLD_REFERENCE), by the
 *         carried operands (value times 2^100); the bfloat16 rows take 16-bit float operands, value times 2^(w -
 *         fold_reference), by the row's 16-bit floats;
 *       - one float32 accumulator of the tensor cores takes every product of an output, in instructions of 16
 *         inputs (m16n8k16): in the wide form over the whole input width in ascending order (a block of
 *         k_split_count splits, each of a run of loads, is added from +0 in ascending split order by
 *         trunk_relaxed_reduce); in the decode form over the loads of a slice, and the slices added from
 *         +0 in ascending order;
 *       - the sum is multiplied by one power of two (2^(RELAXED_FOLD_REFERENCE - 227) of quantized rows,
 *         2^(fold_reference - 127) of bfloat16 rows) and rounded once to bfloat16 as linear_records.h rounds.
 *     The bits of an output depend on its matrix, its row, the form (decode below RELAXED_WIDE_LEAST_ROWS
 *     rows, wide from there on), the decode form's slice count and the wide form's k_split_count: not on the other
 *     rows of a launch, the row's place in it, the tile of rows or the grid (a column of the instruction gets the
 *     same sum wherever it stands: linear_records.h).
 *   - The fold is exact where the scale codes allow it, which relaxed_fold_check (relaxed_linear.h)
 *     checks for each matrix at load: of quantized rows every scale code w of the matrix lies in
 *     RELAXED_SCALE_SMALLEST .. RELAXED_SCALE_LARGEST (a folded weight is then a bfloat16 exactly,
 *     and no product of finite operands, nor a sum of 2^14 of them, leaves the normal float32 numbers); of bfloat16
 *     rows the codes span at most RELAXED_HALF_SCALE_SPREAD, none exceeds
 *     RELAXED_HALF_SCALE_LARGEST (above it the reference clamps its block power), and fold_reference is
 *     the largest less RELAXED_HALF_TOP (a folded weight is then a 16-bit float exactly). Matrices outside
 *     these bounds use reference order. A carried operand that is an infinity (a block above
 *     ACTIVATION_CARRIED_EXPONENT_LARGEST) makes every output it reaches infinite or a not-a-number, as in
 *     the reference order.
 *   - The ORDERED wide instances (trunk_ordered_wide_*) are the reference order itself in the wide
 *     form's tiles: class 80's carried form, a block of 32 inputs summed by two instructions on a zero accumulator,
 *     chain = fma(p, 2^e, chain) over the blocks in ascending order, rounded once from 0 + chain: every output the
 *     bits of trunk_linear_quantized_<S> or trunk_linear_bfloat16_<S> (linear_records.h), at any
 *     row count (trunk_relaxed_card_test holds them byte for byte).
 *   - Launches (grid[2] the splits; block[1] and block[2] 1; tiles of rows: ceil(row_count / rows of the instance)):
 *       kernel                                        grid[0]          grid[1]                     block[0]  shared
 *       trunk_relaxed_wide_quantized_<R>      tiles of rows    outputs / 128               256 or 128  (1)
 *       trunk_ordered_wide_quantized_<R>      tiles of rows    outputs / 128               as above    (1)
 *       trunk_relaxed_wide_bfloat16_<R>       tiles of rows    output_size / 128           as above    (1)
 *       trunk_ordered_wide_bfloat16_<R>       tiles of rows    output_size / 128           as above    (1)
 *       trunk_relaxed_decode_quantized        outputs / 16     1                           32 S        (2)
 *       trunk_relaxed_decode_bfloat16         output_size / 16 1                           32 S        (2)
 *       trunk_relaxed_reduce                  ceil(row_count outputs / 256)  1             256         0
 *     <R> is the rows of a tile of rows, RELAXED_WIDE_ROWS_SMALL or _LARGE (block of 128 or 256 threads);
 *     outputs is output_size[0] + output_size[1]. (1) RELAXED_WIDE_SHARED_BYTES(R, input_size); (2) 16
 *     bytes a thread. Above 16 tiles of rows a wide block takes its place in chunks of 16 row tiles, each chunk
 *     taking every block of outputs in turn (kernel_trunk_relaxed.cu, relaxed_block_place_of). A k_split_count of
 *     1 has grid[2] 1 and no reduce launch; above 1 the wide launch writes the partial sums and the reduce launch
 *     the outputs.
 *   - Sizes: every output_size a multiple of RELAXED_WIDE_OUTPUTS (each block of 128 outputs lies in one
 *     matrix and, of bfloat16 rows, in one segment); input_size a multiple of LINEAR_VALUES_PER_LOAD and at
 *     most RELAXED_MOST_INPUT_SIZE; row_count 1 to LINEAR_MAXIMUM_ROWS in the decode form, 1 to
 *     RELAXED_MOST_ROWS in the wide form; k_split_count 1 to the loads of the input width and at most
 *     RELAXED_MOST_K_SPLITS. The tile grid's total blocks, and the split reduction's blocks where used,
 *     fit uint32_t (the block reorder and launch record use 32-bit indices).
 *   - Records: fixed-width members, widest first, no padding, sizes asserted in both languages.
 */
#ifndef RELAXED_LINEAR_RECORDS_H
#define RELAXED_LINEAR_RECORDS_H

#include <stdint.h>

#include "anemone/linear_records.h"

#define KERNEL_FILE_TRUNK_RELAXED "kernel_trunk_relaxed"
#define KERNEL_RELAXED_WIDE_QUANTIZED_SMALL "trunk_relaxed_wide_quantized_64"
#define KERNEL_RELAXED_WIDE_QUANTIZED_LARGE "trunk_relaxed_wide_quantized_128"
#define KERNEL_ORDERED_WIDE_QUANTIZED_SMALL "trunk_ordered_wide_quantized_64"
#define KERNEL_ORDERED_WIDE_QUANTIZED_LARGE "trunk_ordered_wide_quantized_128"
#define KERNEL_RELAXED_WIDE_BFLOAT16_SMALL "trunk_relaxed_wide_bfloat16_64"
#define KERNEL_RELAXED_WIDE_BFLOAT16_LARGE "trunk_relaxed_wide_bfloat16_128"
#define KERNEL_ORDERED_WIDE_BFLOAT16_SMALL "trunk_ordered_wide_bfloat16_64"
#define KERNEL_ORDERED_WIDE_BFLOAT16_LARGE "trunk_ordered_wide_bfloat16_128"
#define KERNEL_RELAXED_DECODE_QUANTIZED "trunk_relaxed_decode_quantized"
#define KERNEL_RELAXED_DECODE_BFLOAT16 "trunk_relaxed_decode_bfloat16"
#define KERNEL_RELAXED_REDUCE "trunk_relaxed_reduce"

#define RELAXED_WIDE_OUTPUTS \
	128u /* outputs of a block of the wide form: 8 tiles of 16 */
#define RELAXED_WIDE_ROWS_SMALL \
	64u /* rows of a tile of rows: one warp of rows, 4 of outputs */
#define RELAXED_WIDE_ROWS_LARGE 128u /* two warps of rows */
#define RELAXED_WIDE_WARP_ROWS 64u /* the rows of one warp of the wide form */
#define RELAXED_WIDE_WARPS_ALONG_OUTPUTS \
	4u /* a warp takes 32 outputs of a block's 128 */
#define RELAXED_WIDE_LEAST_ROWS 9u /* fewer rows take the decode form */
#define RELAXED_WIDE_STAGES \
	3u /* loads of 64 inputs in flight in a block's shared memory */
#define RELAXED_STAGE_ROW_PITCH \
	144u /* bytes of a row's 64 inputs in shared memory: 128 and 16 */
#define RELAXED_STAGE_TILE_BYTES \
	1024u /* a tile's 64 inputs of weights: 2 blocks of 512 */
#define RELAXED_SCALE_ALIGNMENT \
	16u /* the scale codes of a block, before its stages, padded to this */
#define RELAXED_SCALE_BLOCKS 4u /* blocks of 32 outputs of a block of 128 */
#define RELAXED_MOST_INPUT_SIZE \
	16384u /* the scale codes of a block fit its shared memory */
#define RELAXED_MOST_ROWS 65536u /* rows of one wide launch */
#define RELAXED_MOST_K_SPLITS 16u
#define RELAXED_DECODE_SUM_BYTES \
	16u /* of a thread of the decode form: its four results */
#define RELAXED_REDUCE_THREADS 256u
#define RELAXED_FOLD_REFERENCE \
	177 /* quantized rows: a folded weight is its value times 2^(w - 177) */
#define RELAXED_SCALE_SMALLEST \
	53u /* below, a folded weight's last bit falls under bfloat16's 2^-133 */
/* above, 2^14 products of 448 * 2^19 by 448 * 2^(w - 127), each times 2^50, could pass 2^127 */
#define RELAXED_SCALE_LARGEST 150u
#define RELAXED_HALF_SCALE_SPREAD \
	22u /* bfloat16 rows: a 16-bit float holds 448 * 2^7 down to 2^-24 */
#define RELAXED_HALF_SCALE_LARGEST \
	246u /* the reference's largest float32 power field 254 less E4M3's placement exponent 8 */
#define RELAXED_HALF_TOP \
	7 /* the largest code's folded weights reach 448 * 2^7 */
#define TRUNK_RELAXED_QUANTIZED_ARGUMENT_BYTES 80u
#define TRUNK_RELAXED_BFLOAT16_ARGUMENT_BYTES 72u
#define TRUNK_RELAXED_REDUCE_ARGUMENT_BYTES 48u

/* The dynamic shared memory of a block of the wide form of `rows` rows for an input width: the scale codes of its 4
 * blocks of 32 outputs for every group of 32 inputs, then its stages, each the weights of its 8 tiles and the
 * rows' 64 inputs. */
#define RELAXED_WIDE_SHARED_BYTES(rows, input_size)                          \
	(((RELAXED_SCALE_BLOCKS * ((input_size) / LINEAR_VALUES_PER_BLOCK) + \
	   RELAXED_SCALE_ALIGNMENT - 1u) /                                   \
	  RELAXED_SCALE_ALIGNMENT * RELAXED_SCALE_ALIGNMENT) +               \
	 RELAXED_WIDE_STAGES *                                               \
		 (RELAXED_WIDE_OUTPUTS / LINEAR_OUTPUTS_PER_TILE *           \
			  RELAXED_STAGE_TILE_BYTES +                         \
		  (rows) * RELAXED_STAGE_ROW_PITCH))

/* trunk_relaxed_wide_quantized_<R>, trunk_ordered_wide_quantized_<R>,
 * trunk_relaxed_decode_quantized: y = W x for one or two matrices of one input width and rows of carried
 * operands. Matrix m takes the blocks of outputs from the first one's on; output_size[1] is 0 where there is one.
 *   reads   weights_offset[m], the matrix; operands_offset, 2 row_count input_size bytes: operands
 *           [row_count][input_size];
 *   writes  with k_split_count 1: output_offset[m], values [row_count][output_size[m]]; above 1:
 *           partials_offset, float32 [k_split_count][row_count][output_size[0] + output_size[1]], the sums before
 *           the power of two. The ordered instance takes a k_split_count of 1.
 *   With cone_layer (0, or a decoder layer: engine.c's prefill cone) a block of the wide form whose rows all skip at
 *   that layer (step_records.h STEP_ROW_CONE_SKIPS of the step area at step_offset: its block, then row_count
 *   rows) computes nothing: its outputs keep what they held, as no later row reads them. The decode form takes 0. */
struct trunk_relaxed_quantized_arguments {
	uint64_t weights_offset[LINEAR_MERGED_MATRICES];
	uint64_t output_offset[LINEAR_MERGED_MATRICES];
	uint64_t operands_offset;
	uint64_t partials_offset;
	uint64_t step_offset;
	uint32_t output_size[LINEAR_MERGED_MATRICES];
	uint32_t input_size;
	uint32_t row_count;
	uint32_t k_split_count;
	uint32_t cone_layer;
};

/* trunk_relaxed_wide_bfloat16_<R>, trunk_relaxed_decode_bfloat16: the same product of one matrix for
 * rows that are not quantized, in segment_count runs of outputs as linear_records.h's bfloat16 product cuts them.
 *   reads   weights_offset, the matrix; values_offset, values [row_count][segment_count * input_size];
 *   writes  with k_split_count 1: output_offset, values [row_count][output_size]; above 1: partials_offset,
 *           float32 [k_split_count][row_count][output_size].
 *   step_offset and cone_layer as the quantized rows' (trunk_mma's forms take 0). */
struct trunk_relaxed_bfloat16_arguments {
	uint64_t weights_offset;
	uint64_t values_offset;
	uint64_t output_offset;
	uint64_t partials_offset;
	uint64_t step_offset;
	uint32_t output_size;
	uint32_t input_size;
	uint32_t row_count;
	uint32_t segment_count;
	uint32_t k_split_count;
	int32_t fold_reference; /* a folded weight is its value times 2^(w - fold_reference) */
	uint32_t cone_layer;
	uint32_t reserved_zero;
};

/* trunk_mma's _64 form with its own reduce (kernel_trunk_mma.cu trunk_mma_64_fused): the wide record, then the
 * counters of its output blocks (uint32, one a block of 128 outputs, zero between launches). Each split writes its
 * partials as the _64 form does; the block's last split adds them as trunk_relaxed_reduce does (output_scale
 * 1: the same bits) and resets its counter, so no reduce launch follows. */
struct trunk_mma_fused_arguments {
	struct trunk_relaxed_bfloat16_arguments wide;
	uint64_t counters_offset;
};

/* trunk_mma's 1-8-row or grouped form over two products of one launch (kernel_trunk_mma.cu trunk_mma_*_pair,
 * --tp_launch_trims): blocks below first_blocks run first's record as its own launch's blocks would, the others
 * second's (block less first_blocks): each output the bits of its own launch. */
struct trunk_mma_pair_arguments {
	struct trunk_linear_bfloat16_arguments first, second;
	uint32_t first_blocks;
	uint32_t reserved_zero;
};

/* trunk_relaxed_reduce: for every row and output, the k_split_count partial sums added from +0 in ascending
 * split order, times output_scale (a power of two), rounded once to bfloat16.
 *   reads   partials_offset, float32 [k_split_count][row_count][output_size[0] + output_size[1]];
 *   writes  output_offset[m], values [row_count][output_size[m]]. */
struct trunk_relaxed_reduce_arguments {
	uint64_t partials_offset;
	uint64_t output_offset[LINEAR_MERGED_MATRICES];
	uint32_t output_size[LINEAR_MERGED_MATRICES];
	uint32_t row_count;
	uint32_t k_split_count;
	float output_scale;
	uint32_t reserved_zero;
};

LINEAR_RECORD_HAS(trunk_relaxed_quantized_arguments,
		  TRUNK_RELAXED_QUANTIZED_ARGUMENT_BYTES);
LINEAR_RECORD_HAS(trunk_relaxed_bfloat16_arguments,
		  TRUNK_RELAXED_BFLOAT16_ARGUMENT_BYTES);
LINEAR_RECORD_HAS(trunk_relaxed_reduce_arguments,
		  TRUNK_RELAXED_REDUCE_ARGUMENT_BYTES);

#endif
