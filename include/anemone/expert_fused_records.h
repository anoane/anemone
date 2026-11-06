/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * expert_fused_records.h: the records of the routed experts' phase in fewer launches: the producer (gate and up of
 * every job with the hidden step, and the quantization of the hidden vectors: source/kernel_expert_linear.cu), the
 * join (a row's routed sum, the shared expert's output and the FFN site's post: source/kernel_hyper_connection.cu)
 * and the shared join (the shared expert's down with the join behind it: source/kernel_trunk_linear.cu). C11 and
 * CUDA, including of the engine linear_records.h and model_layout.h alone.
 *
 * Contract: linear_records.h's for a kernel, its record, its ranges, the job table, the arrays, the products, the
 * hidden step and the quantization; hyper_connection_records.h's for the streams, the post and the comb. A kernel here
 * does in one launch what a chain of the separate kernels does, every bit of it, on every class of card: only
 * launches merge. It writes every byte those write, but the row sums where a join is given no array for them.
 *   - expert_gate_up_quantize_<S>: expert_gate_up_<S> and behind it activation_quantize_entry of
 *     the hidden vectors of every pair, pair_count of them (1 to EXPERT_PRODUCER_MOST_PAIRS). Its blocks of
 *     grid rows 0 .. jobs - 1 are gate with up's, block for block, and store the hidden values as it does; the blocks
 *     of tiles 2 x and 2 x + 1 of a job hold one block of 32 values of the quantization, and the second of the two to
 *     arrive quantizes it for each column, from the stored values. Arrival is ONE ATOMIC ADDITION a block on its
 *     pair's counter (the exception this family makes to linear_records.h's rule): the first finds 0, the second 1
 *     and sets the counter back to 0. Each block fences its stores before it arrives and the second reads them from
 *     L2: the quantized bits are those of the stored values, whichever block arrives second; nothing else depends on
 *     timing. Block x of grid row jobs (x below output_size / 32) quantizes block x of every pair that stands in no
 *     job (a row the router left out: its hidden values are whatever the array holds, as the quantizer reads them),
 *     which no other block writes or reads.
 *       reads   as expert_gate_up_<S> (the jobs, their gate and up matrices, the rows' codes, exponents and
 *               operands), and the hidden values of every pair;
 *       writes  hidden_offset: values [pair][output_size], of the pairs that stand in a job; hidden_codes_offset:
 *               codes [pair][output_size], hidden_exponents_offset: exponents [pair][output_size / 32] and
 *               hidden_operands_offset: operands [pair][output_size], of every pair; arrivals_offset, read and
 *               written: uint32 [job][output_size / 32], zero before a launch and after it. output_size is a
 *               multiple of 32.
 *   - expert_join: expert_row_sum_entry (the float32 sum of a row's pairs_per_row pair outputs, from +0 in
 *     ascending slot order) and behind it hyper_post_entry with the combine, y_d = bfloat16(s_d + y_d) of that sum s
 *     and the shared expert's output y, then each stream as the post makes it. The width is MODEL_HIDDEN, a
 *     stream's.
 *       reads   pair_output_offset: values [row * pairs_per_row + slot][width]; shared_output_offset: values
 *               [row][width]; post_offset: float32 [row][4]; comb_offset: float32 [row][4 sources][4 targets];
 *       writes  streams_offset, read and written in place: bfloat16 [row][4][width]; row_sums_offset: float32
 *               [row][width], the sums as expert_row_sum_entry stores them, unless it is
 *               EXPERT_JOIN_NO_ROW_SUMS (the sums are then kept in registers alone).
 *   - expert_shared_join_<S>: trunk_linear_quantized_<S> of the shared expert's down (one matrix of
 *     MODEL_HIDDEN outputs) and behind it the join. A block computes and stores its tile of the output as the
 *     product's block does; after the block's barrier thread 2 r + h joins values 8 h .. 8 h + 7 of the tile for row
 *     r, reading the output where the block stored it.
 *       reads   weights_offset, the matrix (E4M3_SCALED, MODEL_HIDDEN by input_size); codes_offset,
 *               exponents_offset and operands_offset, row_count rows of input_size, as the product reads them; and
 *               what the join reads;
 *       writes  output_offset, values [row][MODEL_HIDDEN], as the product; the streams and the row sums, as
 *               the join.
 *   - Launches (struct kernel_launch of backend_seam.h; grid[2], block[1] and block[2] are 1):
 *       kernel                                grid[0]                 grid[1]   block[0]                     shared
 *       expert_gate_up_quantize_<S>   output_size / 16        jobs + 1  64 S                         64 block
 *       expert_join                   HYPER_PIECES    rows      HYPER_PIECE_THREADS  0
 *       expert_shared_join_<S>        MODEL_HIDDEN/16 1         32 S                         64 block
 *     <S> is a count of LINEAR_EVERY_SLICE_COUNT, with the limits of the exact profile on class 120 that gate
 *     with up and the trunk product have (linear_records.h).
 *   - A record has no padding and at most 96 of a launch's LAUNCH_ARGUMENT_BYTES: asserted below.
 */
#ifndef EXPERT_FUSED_RECORDS_H
#define EXPERT_FUSED_RECORDS_H

#include <stdint.h>

#include "anemone/linear_records.h"
#include "anemone/model_layout.h"

#define EXPERT_PRODUCER_TILES \
	2u /* tiles of 16 outputs: one block of 32 values of the quantization */
/* a step's pairs */
#define EXPERT_PRODUCER_MOST_PAIRS \
	((uint32_t)(LINEAR_MAXIMUM_ROWS * MODEL_EXPERTS_PER_ROW))
#define EXPERT_JOIN_VALUES \
	8u /* values of a row a thread of a join takes: one 16-byte load */
#define EXPERT_JOIN_NO_ROW_SUMS \
	UINT64_MAX /* in place of row_sums_offset: the sums are not stored */
#define KERNEL_EXPERT_GATE_UP_QUANTIZE_STEM "expert_gate_up_quantize_"
#define KERNEL_EXPERT_SHARED_JOIN_STEM "expert_shared_join_"
#define KERNEL_EXPERT_JOIN "expert_join"
#define EXPERT_GATE_UP_QUANTIZE_ARGUMENT_BYTES 96u
#define EXPERT_JOIN_ARGUMENT_BYTES 56u
#define EXPERT_SHARED_JOIN_ARGUMENT_BYTES 96u

/* The separate expert_gate_up_wire_1 entry uses this 96-byte layout with reserved_zero naming
 * a prepared wire byte pitch. That entry keeps the original code lane order and writes [pair][pitch]:
 * output_size code bytes, output_size/32 exponent bytes, then caller-initialized zero padding. Its
 * hidden_exponents_offset is unused. The original entries still require reserved_zero=0.
 * Host preparation must prove the complete wire extent, full input/output extents and zeroed arrivals. */
struct expert_gate_up_quantize_arguments {
	uint64_t jobs_offset;
	uint64_t codes_offset; /* the rows' quantized input: as expert_gate_up_arguments */
	uint64_t exponents_offset;
	uint64_t operands_offset;
	uint64_t hidden_offset; /* written, and read by the second block of a pair */
	uint64_t hidden_codes_offset; /* written */
	uint64_t hidden_exponents_offset; /* written */
	uint64_t hidden_operands_offset; /* written */
	uint64_t arrivals_offset; /* read and written */
	uint32_t output_size;
	uint32_t input_size;
	uint32_t pair_count;
	uint32_t pairs_per_row;
	float activation_limit;
	uint32_t reserved_zero;
};

struct expert_join_arguments {
	uint64_t pair_output_offset;
	uint64_t shared_output_offset;
	uint64_t streams_offset; /* read and written */
	uint64_t post_offset;
	uint64_t comb_offset;
	uint64_t row_sums_offset; /* written, or EXPERT_JOIN_NO_ROW_SUMS */
	uint32_t pairs_per_row;
	uint32_t reserved_zero;
};

struct expert_shared_join_arguments {
	uint64_t weights_offset; /* the shared expert's down */
	uint64_t codes_offset; /* its quantized input: as trunk_linear_quantized_arguments */
	uint64_t exponents_offset;
	uint64_t operands_offset;
	uint64_t output_offset; /* written: the shared expert's output */
	uint64_t pair_output_offset; /* the join's, as expert_join_arguments */
	uint64_t streams_offset;
	uint64_t post_offset;
	uint64_t comb_offset;
	uint64_t row_sums_offset;
	uint32_t input_size;
	uint32_t row_count;
	uint32_t pairs_per_row;
	uint32_t reserved_zero;
};

LINEAR_RECORD_HAS(expert_gate_up_quantize_arguments,
		  EXPERT_GATE_UP_QUANTIZE_ARGUMENT_BYTES);
LINEAR_RECORD_HAS(expert_join_arguments, EXPERT_JOIN_ARGUMENT_BYTES);
LINEAR_RECORD_HAS(expert_shared_join_arguments,
		  EXPERT_SHARED_JOIN_ARGUMENT_BYTES);
MODEL_LAYOUT_HOLDS(LINEAR_OUTPUTS_PER_TILE % EXPERT_JOIN_VALUES == 0 &&
		   LINEAR_OUTPUTS_PER_TILE / EXPERT_JOIN_VALUES *
				   LINEAR_MAXIMUM_ROWS <=
			   LINEAR_THREADS_PER_WARP);

#endif
