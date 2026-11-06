/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * attention_output_records.h: the records of the attention site's output span in two kernels
 * (source/kernel_attention_output.cu): the output groups with the quantization of what they give, and the output
 * projection with the site's post. C11 and CUDA, including of the engine linear_records.h and model_layout.h alone.
 *
 * Contract: linear_records.h's for a kernel, its record, its ranges, its arrays, its products and the quantization;
 * hyper_connection_records.h's for the post. A kernel here does in one launch what a chain of the separate kernels
 * does, every bit of it, on every class of card: only launches merge.
 *   - attention_groups_<S>: trunk_linear_bfloat16_<S> (attn.wo_a, block-diagonal over segment_count
 *     groups) and behind it activation_quantize_entry of its outputs (the width output_size). A block owns the 32
 *     outputs of one block of the quantization: its two tiles of 16, each computed by a group of S warps as a block
 *     of the separate product computes it, the first tile's group first. What the product stores is stored, and
 *     quantized.
 *       reads   weights_offset, the matrix: output_size * input_size + output_size / 32 * input_size / 32 bytes;
 *               values_offset, 2 row_count segment_count input_size bytes: values [row_count][segment_count *
 *               input_size];
 *       writes  groups_offset, 2 row_count output_size bytes: values [row_count][output_size]; codes_offset,
 *               row_count output_size bytes; exponents_offset, row_count output_size / 32 bytes; operands_offset,
 *               2 row_count output_size bytes: codes, exponents and operands [row_count][output_size].
 *   - attention_projection_<S>: trunk_linear_quantized_<S> of one matrix (attn.wo_b) and behind it
 *     hyper_post_entry of the attention site (is_combined 0). A block is one tile, computed as a block of the
 *     separate product computes it; what the product stores is stored, and the tile's 16 coordinates of each row's
 *     four streams take the post of those values. output_size is MODEL_HIDDEN, the width of a stream.
 *       reads   weights_offset, the matrix; codes_offset, exponents_offset and operands_offset: row_count rows of
 *               input_size, as the separate product reads them; post_offset, 16 row_count bytes: float32
 *               [row][4]; comb_offset, 64 row_count bytes: float32 [row][4 sources][4 targets];
 *       writes  output_offset, 2 row_count output_size bytes: values [row_count][output_size]; streams_offset,
 *               8 row_count output_size bytes, read and written in place: bfloat16 [row][4][output_size].
 *   - Launches (struct kernel_launch of backend_seam.h; grid[1], grid[2], block[1] and block[2] are 1; <S> the
 *     separate product's slice count, a count of LINEAR_EVERY_SLICE_COUNT):
 *       kernel                              grid[0]            block[0]   shared bytes
 *       attention_groups_<S>        output_size / 32   64 S       64 block[0]
 *       attention_projection_<S>    output_size / 16   32 S       64 block[0]
 *     The exact profile on class 120 emulates class 80's sums: there both take at most
 *     LINEAR_CLASS_120_EXACT_MOST_SLICES slices a tile, as its separate products do (its bits are those of
 *     any count: the reference order of linear_thread_chains); an instance of more is a stub never launched.
 *   - A record has no padding and at most 80 of a launch's LAUNCH_ARGUMENT_BYTES: asserted below.
 */
#ifndef ATTENTION_OUTPUT_RECORDS_H
#define ATTENTION_OUTPUT_RECORDS_H

#include <stdint.h>

#include "anemone/linear_records.h"
#include "anemone/model_layout.h"

#define KERNEL_FILE_ATTENTION_OUTPUT "kernel_attention_output"
#define KERNEL_FILE_ATTENTION_OUTPUT_NATIVE \
	"kernel_attention_output_native" /* class 120 native */
#define KERNEL_ATTENTION_GROUPS_STEM "attention_groups_"
#define KERNEL_ATTENTION_PROJECTION_STEM "attention_projection_"
#define ATTENTION_GROUP_TILES \
	2u /* the tiles of one block of the quantization: 32 outputs */
#define ATTENTION_GROUPS_ARGUMENT_BYTES 64u
#define ATTENTION_PROJECTION_ARGUMENT_BYTES 80u

struct attention_groups_arguments {
	uint64_t weights_offset;
	uint64_t values_offset;
	uint64_t groups_offset;
	uint64_t codes_offset;
	uint64_t exponents_offset;
	uint64_t operands_offset;
	uint32_t output_size;
	uint32_t input_size;
	uint32_t row_count;
	uint32_t segment_count;
};

struct attention_projection_arguments {
	uint64_t weights_offset;
	uint64_t codes_offset;
	uint64_t exponents_offset;
	uint64_t operands_offset;
	uint64_t output_offset;
	uint64_t streams_offset;
	uint64_t post_offset;
	uint64_t comb_offset;
	uint32_t output_size;
	uint32_t input_size;
	uint32_t row_count;
	uint32_t reserved_zero;
};

MODEL_LAYOUT_HOLDS(sizeof(struct attention_groups_arguments) ==
			   ATTENTION_GROUPS_ARGUMENT_BYTES &&
		   sizeof(struct attention_projection_arguments) ==
			   ATTENTION_PROJECTION_ARGUMENT_BYTES);
MODEL_LAYOUT_HOLDS(ATTENTION_GROUP_TILES *LINEAR_OUTPUTS_PER_TILE ==
		   LINEAR_VALUES_PER_BLOCK);

#endif
