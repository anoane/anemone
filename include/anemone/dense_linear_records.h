/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * dense_linear_records.h: the record of the dense products, y = W x for a matrix as the checkpoint stores it
 * (bfloat16 or float32, row-major) and 1 to LINEAR_MAXIMUM_ROWS rows of bfloat16 values (a wide launch: more,
 * the rows of a chunk). C11 and CUDA, including of the engine the record layouts of the step and the linear alone.
 * Contract: linear_records.h's, and:
 *   - Arrays: weights [output_size][input_size]; rows bfloat16 [row_count][input_size]; output [row_count]
 *     [output_size] of output_type; with DENSE_STEP_ROWS the step area at step_offset. output_size is a
 *     multiple of DENSE_OUTPUTS_PER_BLOCK and input_size of DENSE_CHUNK_INPUTS, neither zero.
 *   - One order of additions: thread t of DENSE_THREADS_VALUE takes chunks t, t + 256, ... of 8 inputs; its chain
 *     of an output and a row runs from +0 in ascending input order, fmaf(weight, value, chain), both widened
 *     exactly; a warp adds its lanes' chains by exchanges at distance 16 down to 1, then the warps' sums from +0 in
 *     ascending order. An output's bits are its matrix's and its row's alone (and, by rsqrtf, the card's). The
 *     reference's order is its library's choice on the card: no knob reproduces it.
 *   - DENSE_ROW_SCALE (the hyper mixes, REF:952-954): the sum times rsqrtf(q * (1 / input_size) + epsilon),
 *     q the row's squares added in the same order. Then output_scale: a float32 output is the value times it; a
 *     bfloat16 one is rounded (to nearest even, a not-a-number to 0x7FC0), multiplied and rounded again.
 *   - DENSE_STEP_ROWS (the head): only a row below the step block's row_count whose flags hold
 *     STEP_ROW_LOGITS is written; a step with no such row reads no weight (a prompt's earlier steps).
 *   - DENSE_CONE_ROWS (a wide launch: engine.c's prefill cone): a group of rows whose rows all skip at
 *     decoder layer cone_layer (step_records.h STEP_ROW_CONE_TRUNK_SKIPS of the rows at step_offset, the
 *     launch's first row's first) stores nothing: its outputs keep what they held, as no later row reads them.
 *   - A launch: function KERNEL_DENSE_LINEAR_STEM <type>_<R>, R 1 for one row and 8 for 2 to 8 (a column
 *     from row_count on reads row 0 and is stored nowhere), a block owning DENSE_OUTPUTS_PER_BLOCK outputs;
 *     where that leaves fewer than DENSE_NARROW_BELOW_BLOCKS blocks (every product but the head),
 *     <type>_<R>_narrow, a block owning one, so that more multiprocessors read the weights: the same operations for
 *     every output. Grid output_size over the outputs of a block, by 1; block DENSE_THREADS_VALUE; no dynamic shared
 *     memory.
 *   - A wide launch, of more rows and never of DENSE_STEP_ROWS: function KERNEL_DENSE_LINEAR_STEM
 *     <type>_wide_<O>, a block owning O outputs (O divides output_size) of the rows of group blockIdx.y, rows 8 g to
 *     8 g + 7 (the last group fewer), which it takes as the instance of 8 rows takes a launch of them alone: every
 *     output byte is that of the launch of 8 rows of its group, so that of a chunk's thin launches, a launch of one row
 *     being the same operations. Grid output_size / O by the groups; block DENSE_THREADS_VALUE; no dynamic shared
 *     memory. Every group reads the weights, so a wide instance reads them with the caches' default hint.
 *   - A staged wide launch (--dense_staged): function KERNEL_DENSE_LINEAR_STEM <type>_staged_<S>, a block
 *     owning S outputs of the rows of group blockIdx.y, never of DENSE_ROW_SCALE or _STEP_ROWS: the group's 8
 *     rows staged once in dynamic shared memory (8 x 16 bytes a chunk: input_size at most
 *     DENSE_STAGED_MOST_INPUTS), then each of the S outputs in turn by all the block's threads, every chain,
 *     exchange and warp sum that of the wide instance's block of that output: the same bytes, the rows read once for
 *     S outputs. Grid output_size / S by the groups; block DENSE_THREADS_VALUE; dynamic shared memory 16
 *     input_size bytes.
 */
#ifndef DENSE_LINEAR_RECORDS_H
#define DENSE_LINEAR_RECORDS_H

#include <stdint.h>

#include "anemone/linear_records.h"
#include "anemone/step_records.h"

#define DENSE_OUTPUTS_PER_BLOCK 4u
#define DENSE_NARROW_BELOW_BLOCKS \
	256u /* ours: measured 25 Oct 2026 on a CMP (the head stays wide) */
#define DENSE_CHUNK_INPUTS 8u /* 16 bytes of bfloat16 */
#define DENSE_THREADS_VALUE \
	256u /* of a block: 8 warps, each a slice of the chunks */
#define DENSE_ROW_SCALE 0x1u
#define DENSE_STEP_ROWS 0x2u
#define DENSE_CONE_ROWS 0x4u
#define DENSE_FLAGS (DENSE_ROW_SCALE | DENSE_STEP_ROWS | DENSE_CONE_ROWS)
#define DENSE_INDEX_WEIGHTS_SCALE \
	0x1p-6f /* REF:515, 555: 128^-0.5 x 32^-0.5 (CFG:34-35) in float32 */
#define DENSE_ARGUMENT_BYTES 64u
#define KERNEL_FILE_DENSE_LINEAR "kernel_dense_linear"
#define KERNEL_DENSE_LINEAR_STEM "dense_linear_"
enum dense_type_code {
	DENSE_BFLOAT16 = 0,
	DENSE_FLOAT32 = 1,
	DENSE_TYPE_COUNT = 2
};
/* DENSE_STAMP(name, type, rows, outputs, form) for every function, in the order of struct
 * dense_linear_kernels (outputs: of a block; form: empty, or _narrow): whoever expands it defines
 * DENSE_STAMP first and undefines it after. */
#define DENSE_EVERY_INSTANCE                                                \
	DENSE_STAMP(bfloat16, DENSE_BFLOAT16, 1, DENSE_OUTPUTS_PER_BLOCK, ) \
	DENSE_STAMP(bfloat16, DENSE_BFLOAT16, 8, DENSE_OUTPUTS_PER_BLOCK, ) \
	DENSE_STAMP(float32, DENSE_FLOAT32, 1, DENSE_OUTPUTS_PER_BLOCK, )   \
	DENSE_STAMP(float32, DENSE_FLOAT32, 8, DENSE_OUTPUTS_PER_BLOCK, )   \
	DENSE_STAMP(bfloat16, DENSE_BFLOAT16, 1, 1u, _narrow)               \
	DENSE_STAMP(bfloat16, DENSE_BFLOAT16, 8, 1u, _narrow)               \
	DENSE_STAMP(float32, DENSE_FLOAT32, 1, 1u, _narrow)                 \
	DENSE_STAMP(float32, DENSE_FLOAT32, 8, 1u, _narrow)
#define DENSE_FORMS 2u /* wide (4 outputs a block), narrow (1) */
#define DENSE_INSTANCES 8u /* forms by types by row counts */
#define DENSE_FORM_INSTANCES \
	4u /* of one form: the wide ones first, then the narrow */
#define DENSE_ROW_INSTANCES 2u /* of one form and type: one row, then 8 */
/* DENSE_WIDE_STAMP(name, type, outputs, blocks) for every function of a wide launch, in the order of struct
 * dense_linear_kernels' wide_products, of each type the most outputs first (outputs: of a block; blocks: the
 * fewest a multiprocessor holds, which bounds the registers of a thread): whoever expands it defines
 * DENSE_WIDE_STAMP first and undefines it after. */
#define DENSE_EVERY_WIDE_INSTANCE                        \
	DENSE_WIDE_STAMP(bfloat16, DENSE_BFLOAT16, 8, 1) \
	DENSE_WIDE_STAMP(bfloat16, DENSE_BFLOAT16, 4, 2) \
	DENSE_WIDE_STAMP(bfloat16, DENSE_BFLOAT16, 1, 2) \
	DENSE_WIDE_STAMP(float32, DENSE_FLOAT32, 8, 1)   \
	DENSE_WIDE_STAMP(float32, DENSE_FLOAT32, 4, 2)   \
	DENSE_WIDE_STAMP(float32, DENSE_FLOAT32, 1, 2)
#define DENSE_WIDE_INSTANCES 6u
#define DENSE_STAGED_LEAST_OUTPUTS 8u
#define DENSE_STAGED_MOST_OUTPUTS 64u
#define DENSE_STAGED_OUTPUTS \
	16u /* --dense_staged: the outputs of a staged block */
#define DENSE_STAGED_MOST_INPUTS \
	5120u /* its staged rows: 8 x 5,120 bfloat16 = 80 KiB of shared memory */
#define DENSE_STAGED_LEAST_ROWS \
	64u /* dense_linear.c takes it from so many rows (a BATCH step's: not) */
/* DENSE_STAGED_STAMP(name, type, outputs) for every function of a staged wide launch, by enum
 * dense_type_code: whoever expands it defines DENSE_STAGED_STAMP first and undefines it after. */
#define DENSE_EVERY_STAGED_INSTANCE                      \
	DENSE_STAGED_STAMP(bfloat16, DENSE_BFLOAT16, 8)  \
	DENSE_STAGED_STAMP(float32, DENSE_FLOAT32, 8)    \
	DENSE_STAGED_STAMP(bfloat16, DENSE_BFLOAT16, 16) \
	DENSE_STAGED_STAMP(float32, DENSE_FLOAT32, 16)   \
	DENSE_STAGED_STAMP(bfloat16, DENSE_BFLOAT16, 32) \
	DENSE_STAGED_STAMP(float32, DENSE_FLOAT32, 32)   \
	DENSE_STAGED_STAMP(bfloat16, DENSE_BFLOAT16, 64) \
	DENSE_STAGED_STAMP(float32, DENSE_FLOAT32, 64)

struct dense_linear_arguments {
	uint64_t weights_offset;
	uint64_t rows_offset;
	uint64_t output_offset;
	uint64_t step_offset; /* read with DENSE_STEP_ROWS or _CONE_ROWS only */
	uint32_t output_size;
	uint32_t input_size;
	uint32_t row_count;
	uint32_t output_type; /* enum dense_type_code */
	uint32_t flags; /* DENSE_* */
	float output_scale;
	float epsilon; /* of DENSE_ROW_SCALE */
	uint32_t cone_layer; /* of DENSE_CONE_ROWS, else 0 */
};
LINEAR_RECORD_HAS(dense_linear_arguments, DENSE_ARGUMENT_BYTES);

/* --tp_router_mma (kernel_head_mma.cu router_mma_entry, _reduce): the router of 9 to 64 rows (bfloat16 weights,
 * outputs a multiple of 16) by split-K on the tensor cores, GNmac's router class (a split-K BF16 GEMM, FP32 sums, a fixed-order
 * reduce): the dense record (no flags, float32 out), then float32 partials [split_count][row_count][output_size]; the
 * reduce adds the splits from +0 in ascending order into the output. Another order than the dense product's. */
struct router_mma_arguments {
	struct dense_linear_arguments dense;
	uint64_t partials_offset;
	uint32_t split_count;
	uint32_t reserved_zero;
};

#endif
