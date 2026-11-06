/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * wide_linear_records.h: the records of the wide products (source/kernel_wide_linear.cu), for steps of up to
 * MODEL_WIDE_MOST_ROWS rows, and the arrays of a layer's routed experts the router's wide select writes for
 * them (wide_prefill.md 3.1, 3.2). C11 and CUDA, of the engine linear_records.h and step_records.h alone.
 * Class-80 chunk routed experts use expert_relaxed_records.h's ordered wide instance by default; the old job-table
 * records and kernels remain available. Class-80 chunk trunk products use relaxed_linear_records.h's ordered
 * instance. Both modules expose optional relaxed forms behind cmp_order; the sorted-pair and tile layouts below
 * are shared by both expert orders.
 *
 * Contract:
 *   - linear_records.h's rules hold: offsets into the arena that whoever prepares a launch has checked, its arrays,
 *     quantization, forms, products, sizes and scale ranges, no atomic operation, nothing that depends on timing.
 *   - The routed experts of a step of R rows, written for the rows below the step's row count:
 *       sorted pairs  struct wide_pair [6 R]: each pair of a live row (its pair_index row * 6 + slot, as a
 *                     job's column, and its route weight), by expert ascending, then row ascending; the rest unread
 *       runs          struct wide_run [384]: expert e's pairs, from first_pair of the sorted pairs on
 *       expert tiles  struct wide_expert_tile [ceil(6 R / 64) + 384]: the runs, in ascending expert order,
 *                     cut into tiles of at most WIDE_TILE_PAIRS pairs, each naming its expert's three
 *                     matrices as a job does; then empty tiles, every member 0
 *   - Class R (wide_prefill.md 8, B1): an output's bits depend on its matrix, its row's or pair's input and the slice
 *     count S alone: not on the other rows of a launch, the step's row count, the chunk, the tile or the grid.
 *   - Function <stem><S> computes linear_records.h's product with S slices in the form the one-row kernel of its
 *     family takes on the card's class; with linear.h's S for them every output is that kernel's, bit for bit
 *     (wide_prefill.md 8, B3: profile 1). Chained forms (profile 2) come with their arithmetic where Q1 or Q2 shows
 *     10% or more (wide_prefill.md 3.1).
 *   - The trunk computes the rows below the row count of the step block at step_offset (the step area or a view,
 *     model_seam.h), at most row_count: a block whose rows all lie beyond it reads and writes nothing, another may
 *     compute stale rows (step_records.h). The experts take the pairs of the first tile_count tiles; an empty tile
 *     does nothing. Their record's pair_count is the pairs the arrays hold (row capacity times pairs_per_row), which
 *     sizes every array by pair: the indices of each tile's pairs and each pair's pair_index lie below it, else the
 *     checked modules trap, as for jobs (linear_records.h). A tile's matrices are checked by router.h alone: see there.
 *   - The gate with up writes, for each pair, what activation_quantize_entry writes from the hidden values
 *     expert_gate_up_<S> computes (linear_records.h): codes, exponents, operands, not the values (the producer
 *     quantizes). The down product writes a pair's output at its pair_index: expert_row_sum_entry adds a row's six.
 *   - Launches: grid[1] covers the row count's tiles of rows (trunk) or tile_count (experts), grid[0] the outputs;
 *     the rows a block takes, its threads and its shared memory are fixed per function by wide_linear.h.
 *   - Records: fixed-width members, widest first, no padding, sizes asserted in both languages.
 */
#ifndef WIDE_LINEAR_RECORDS_H
#define WIDE_LINEAR_RECORDS_H

#include <stdint.h>

#include "anemone/linear_records.h"
#include "anemone/step_records.h"

#define KERNEL_FILE_WIDE_LINEAR "kernel_wide_linear"
#define KERNEL_WIDE_EXPERT_GATE_UP_STEM "wide_expert_gate_up_"
#define KERNEL_WIDE_EXPERT_DOWN_STEM "wide_expert_down_"
#define KERNEL_WIDE_TRUNK_QUANTIZED_STEM "wide_trunk_quantized_"
#define KERNEL_WIDE_TRUNK_BFLOAT16_STEM "wide_trunk_bfloat16_"
#define WIDE_TILE_PAIRS 64u /* ours: in columns of 8, a tail wastes 7 at most */
#define WIDE_PAIR_BYTES 8u
#define WIDE_RUN_BYTES 8u
#define WIDE_EXPERT_TILE_BYTES 32u
#define WIDE_EXPERT_GATE_UP_ARGUMENT_BYTES 88u
#define WIDE_EXPERT_DOWN_ARGUMENT_BYTES 64u
#define WIDE_TRUNK_QUANTIZED_ARGUMENT_BYTES 80u
#define WIDE_TRUNK_BFLOAT16_ARGUMENT_BYTES 48u

struct wide_pair {
	uint32_t pair_index;
	float route_weight;
};

struct wide_run {
	uint32_t first_pair;
	uint32_t pair_count; /* 0 for an expert no live row chose */
};

struct wide_expert_tile {
	uint64_t gate_offset;
	uint64_t up_offset;
	uint64_t down_offset;
	uint32_t first_pair;
	uint32_t pair_count; /* 0 to WIDE_TILE_PAIRS */
};

/* wide_expert_gate_up_<S>: gate and up of every tile's pairs, and the hidden step.
 *   reads   the tiles, their pairs and their gate and up matrices; codes_offset, exponents_offset, operands_offset:
 *           the rows' quantized input [row][input_size] (a pair's row: pair_index / pairs_per_row);
 *   writes  hidden_codes_offset [pair][output_size], hidden_exponents_offset [pair][output_size / 32] and
 *           hidden_operands_offset [pair][output_size], of the tiles' pairs only. */
struct wide_expert_gate_up_arguments {
	uint64_t tiles_offset;
	uint64_t pairs_offset;
	uint64_t codes_offset;
	uint64_t exponents_offset;
	uint64_t operands_offset;
	uint64_t hidden_codes_offset;
	uint64_t hidden_exponents_offset;
	uint64_t hidden_operands_offset;
	uint32_t output_size;
	uint32_t input_size;
	uint32_t tile_count;
	uint32_t pairs_per_row;
	float activation_limit;
	uint32_t pair_count;
};

/* wide_expert_down_<S>: the down product of every tile's pairs from their quantized hidden.
 *   reads   the tiles, their pairs and their down matrices; codes, exponents and operands [pair][input_size] of the
 *           tiles' pairs;
 *   writes  output_offset: values [pair][output_size], of the tiles' pairs only. */
struct wide_expert_down_arguments {
	uint64_t tiles_offset;
	uint64_t pairs_offset;
	uint64_t codes_offset;
	uint64_t exponents_offset;
	uint64_t operands_offset;
	uint64_t output_offset;
	uint32_t output_size;
	uint32_t input_size;
	uint32_t tile_count;
	uint32_t pair_count;
};

/* wide_trunk_quantized_<S>: trunk_linear_quantized_<S>'s product (one or two matrices of one input)
 * for the live rows of the step block at step_offset, at most row_count; reads and writes as its record says. */
struct wide_trunk_quantized_arguments {
	uint64_t weights_offset[LINEAR_MERGED_MATRICES];
	uint64_t output_offset[LINEAR_MERGED_MATRICES];
	uint64_t codes_offset;
	uint64_t exponents_offset;
	uint64_t operands_offset;
	uint64_t step_offset;
	uint32_t output_size[LINEAR_MERGED_MATRICES];
	uint32_t input_size;
	uint32_t row_count;
};

/* wide_trunk_bfloat16_<S>: trunk_linear_bfloat16_<S>'s product (segment_count runs of outputs) for
 * the live rows of the step block at step_offset, at most row_count. */
struct wide_trunk_bfloat16_arguments {
	uint64_t weights_offset;
	uint64_t values_offset;
	uint64_t output_offset;
	uint64_t step_offset;
	uint32_t output_size;
	uint32_t input_size;
	uint32_t row_count;
	uint32_t segment_count;
};

LINEAR_RECORD_HAS(wide_pair, WIDE_PAIR_BYTES);
LINEAR_RECORD_HAS(wide_run, WIDE_RUN_BYTES);
LINEAR_RECORD_HAS(wide_expert_tile, WIDE_EXPERT_TILE_BYTES);
LINEAR_RECORD_HAS(wide_expert_gate_up_arguments,
		  WIDE_EXPERT_GATE_UP_ARGUMENT_BYTES);
LINEAR_RECORD_HAS(wide_expert_down_arguments, WIDE_EXPERT_DOWN_ARGUMENT_BYTES);
LINEAR_RECORD_HAS(wide_trunk_quantized_arguments,
		  WIDE_TRUNK_QUANTIZED_ARGUMENT_BYTES);
LINEAR_RECORD_HAS(wide_trunk_bfloat16_arguments,
		  WIDE_TRUNK_BFLOAT16_ARGUMENT_BYTES);

#endif
