/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * expert_relaxed_records.h: the records of the routed experts in the RELAXED ORDER of class 80 (source/
 * kernel_expert_relaxed.cu), decode_accuracy's sums (MEMORY.md, the owner's decision of 26 October 2026: the CMPs'
 * order of additions may change, a startup knob forces the reference order back). C11 and CUDA, of the engine
 * linear_records.h and wide_linear_records.h alone.
 *
 * Contract:
 *   - linear_records.h holds but for the order of additions: its arrays, the job table, the quantization, the operands
 *     of the carried form, the weights' codes and scale codes, the hidden step with its two bfloat16 roundings and
 *     the route weight before the third, the rounding of an output to bfloat16, sizes, offsets, no atomic operation,
 *     nothing that depends on timing, no byte touched outside the ranges a record states. wide_linear_records.h's
 *     tiles and sorted pairs are what the wide form reads.
 *   - The ordered wide instance (expert_ordered_wide_gate_up, _down and their _small_ twins) shares the wide
 *     records, tile geometry, staging, group counts and six-launch chain below. Each K32 term is two carried BF16
 *     MMAs on a fresh +0 accumulator, then fmaf(term, linear_block_power, chain) in ascending input order from +0.
 *     Weight placement and the power clamp are linear_load_sums', with no folded weight or fold admission. Before
 *     the existing BF16/hidden epilogue, +0 is added as in linear_sum_of_slices. This is linear_ordered_chains' order
 *     and preserves the original expert chunk kernel's stored bytes. The relaxed width cap does not apply.
 *   - Two relaxed orders, one a form; each holds linear_records.h's operands and weights exactly:
 *       narrow (expert_relaxed_gate_up, _down; the job table, the steps of 1 to 8 rows): the carried form's
 *               block sums and powers of two (linear_records.h: two instructions on +0 a block of 32 inputs, one
 *               fused multiply-add of the block's sum by its power into the chain), in
 *               EXPERT_RELAXED_NARROW_SLICES slices of the loads (linear_slice_start), each a chain from +0
 *               in ascending input order, the slices' chains added in ascending slice order from +0, the sum rounded
 *               once: the order of class 80's own forms of linear_records.h at that slice count (linear_slice_chains;
 *               the card test of 26 October found the bits of kernel_expert_linear built with
 *               LINEAR_CLASS_FORMS), not the reference order of linear_ordered_chains;
 *       wide   (expert_relaxed_wide_gate_up_entry, _down and their _small_ twins; the tiles of a wide select, a
 *               prefill's chunks): a weight is its code's value times its scale, placed as a bfloat16 (the code
 *               placed, then one rounded multiplication by the power of two of field c + 26, 254 at most, c its scale
 *               code): exact for every scale code from EXPERT_RELAXED_SCALE_SMALLEST to
 *               EXPERT_CARRIED_SCALE_LARGEST; above, 2^(228 - c) times the true weight (the carried form's own
 *               bound). Fed by the carried operands (2^100 times the value), a product of the tensor cores is the true
 *               product, which no further power multiplies. The instructions m16n8k16 add 16 products a step to one
 *               float32 accumulator, the steps of all L loads in ascending input order from +0 (no slices, nothing
 *               added after), and the sum is rounded once. For scale codes up to
 *               EXPERT_RELAXED_SCALE_LARGEST and at most EXPERT_RELAXED_WIDE_MOST_INPUTS inputs no
 *               product or partial sum reaches 2^127 (2^14 products of 6 * 2^(c - 127) by the largest finite
 *               activation, 448 * 2^19), nor does the reference order's chain; above, true products the reference
 *               order never forms (it sums a block's placed products before its power) can pass the largest float32:
 *               an infinity, or a not-a-number from two of opposite signs, where the reference's output is finite.
 *     No fused multiply-add of their own beyond the narrow chains': the sums inside an instruction are the tensor
 *     cores', whose rounding linear_records.h defines nowhere. The bits of an output depend on its matrix, its vector
 *     and its form alone: not on the kernel, the other vectors of its job or tile, the column it takes, the grid, the
 *     row count or the card of class 80. So a row takes the same bits in a step of one row and of eight, and a pair
 *     the same bits in every chunk of a prefill; a step's bits are not a chunk's. Where an operand is infinite the
 *     outputs it reaches are infinite or not-a-number, as in the carried form.
 *   - The narrow kernels take the records of expert_gate_up_<S> and expert_down_<S> unchanged (they
 *     read the operands, not the codes or exponents) and write what those write: hidden values, pair outputs.
 *   - The wide kernels take the tiles of a wide select (wide_linear_records.h): a tile's pairs first_pair ..
 *     first_pair + pair_count - 1 of the sorted pairs, at most WIDE_TILE_PAIRS, each standing in no other tile,
 *     the tiles of one expert next to one another (as the select cuts a run); each pair_index below the record's
 *     pair_count, the pairs array holding pair_count entries. A tile of no pair does nothing. Of each product two
 *     kernels share the record, a tile's work going to one of them: the small kernel takes a tile of 1 to
 *     EXPERT_RELAXED_SMALL_TILE_PAIRS pairs that is its expert's only one (the tile before it, if any, names
 *     other matrices), the plain kernel every other tile; both run, small after plain. gate with up writes a pair's
 *     hidden values (values [pair][output_size]), which activation_quantize_entry quantizes, down a pair's outputs,
 *     which expert_row_sum_entry adds.
 *   - Launches (struct kernel_launch of backend_seam.h; grid[2], block[1] and block[2] are 1):
 *       kernel                                      grid[0]              grid[1]      block[0]   shared
 *       expert_relaxed_gate_up              output_size / 16     jobs         THREADS    NARROW
 *       expert_relaxed_down                 output_size / 32     jobs         THREADS    NARROW
 *       expert_relaxed_wide_gate_up_entry         output_size / 128    tile_count   THREADS    WIDE
 *       expert_relaxed_wide_down            output_size / 256    tile_count   THREADS    WIDE
 *       expert_relaxed_wide_small_gate_up   output_size / 128    tile_count   THREADS    WIDE_SMALL
 *       expert_relaxed_wide_small_down      output_size / 256    tile_count   THREADS    WIDE_SMALL
 *     with the constants EXPERT_RELAXED_THREADS and EXPERT_RELAXED_<NARROW|WIDE|WIDE_SMALL>_SHARED_
 *     BYTES; an output_size a multiple of its grid[0] divisor (EXPERT_RELAXED_<NARROW|WIDE>_<GATE_UP|DOWN>_
 *     OUTPUTS), an input_size a multiple of EXPERT_RELAXED_NARROW_INPUTS (narrow) or of
 *     EXPERT_RELAXED_WIDE_INPUTS and at most EXPERT_RELAXED_WIDE_MOST_INPUTS (wide).
 *   - A record has no padding and fixed-width members, widest first: asserted below, C11 and CUDA.
 */
#ifndef EXPERT_RELAXED_RECORDS_H
#define EXPERT_RELAXED_RECORDS_H

#include <stdint.h>

#include "anemone/linear_records.h"
#include "anemone/wide_linear_records.h"

#define KERNEL_FILE_EXPERT_RELAXED "kernel_expert_relaxed"
#define EXPERT_WIDE_MOST_TILES \
	65535u /* CUDA grid Y; empty router entries are allowed */
#define KERNEL_EXPERT_RELAXED_GATE_UP "expert_relaxed_gate_up"
#define KERNEL_EXPERT_RELAXED_DOWN "expert_relaxed_down"
#define KERNEL_EXPERT_RELAXED_WIDE_GATE_UP "expert_relaxed_wide_gate_up_entry"
#define KERNEL_EXPERT_RELAXED_WIDE_DOWN "expert_relaxed_wide_down"
#define KERNEL_EXPERT_RELAXED_WIDE_SMALL_GATE_UP \
	"expert_relaxed_wide_small_gate_up"
#define KERNEL_EXPERT_RELAXED_WIDE_SMALL_DOWN "expert_relaxed_wide_small_down"
#define KERNEL_EXPERT_ORDERED_WIDE_GATE_UP "expert_ordered_wide_gate_up"
#define KERNEL_EXPERT_ORDERED_WIDE_DOWN "expert_ordered_wide_down"
#define KERNEL_EXPERT_ORDERED_WIDE_SMALL_GATE_UP \
	"expert_ordered_wide_small_gate_up"
#define KERNEL_EXPERT_ORDERED_WIDE_SMALL_DOWN "expert_ordered_wide_small_down"
#define EXPERT_RELAXED_WARPS \
	8u /* of a block: eight units (wide), eight slices (narrow) */
#define EXPERT_RELAXED_THREADS 256u
#define EXPERT_RELAXED_NARROW_SLICES \
	8u /* narrow: a warp a slice of a unit's loads */
#define EXPERT_RELAXED_NARROW_STAGES \
	2u /* narrow: loads a warp holds, one in flight */
#define EXPERT_RELAXED_NARROW_STAGE_BYTES \
	2112u /* a warp's load: its weights, its columns' operands */
#define EXPERT_RELAXED_NARROW_SHARED_BYTES                     \
	(EXPERT_RELAXED_NARROW_STAGES * EXPERT_RELAXED_WARPS * \
	 EXPERT_RELAXED_NARROW_STAGE_BYTES)
#define EXPERT_RELAXED_WIDE_SHARED_BYTES \
	60928u /* three loads of weights, two chunks of 64 rows */
#define EXPERT_RELAXED_WIDE_SMALL_SHARED_BYTES \
	43520u /* three loads of weights, chunks of 32 rows */
#define EXPERT_RELAXED_SMALL_TILE_PAIRS \
	24u /* wide: the most pairs of a small kernel's tile */
#define EXPERT_RELAXED_NARROW_INPUTS \
	64u /* what an input_size is a multiple of: a load */
#define EXPERT_RELAXED_WIDE_INPUTS 128u /* a chunk of rows */
#define EXPERT_RELAXED_NARROW_GATE_UP_OUTPUTS \
	16u /* a narrow block: a unit, its warps its slices */
#define EXPERT_RELAXED_NARROW_DOWN_OUTPUTS 32u
#define EXPERT_RELAXED_WIDE_GATE_UP_OUTPUTS \
	128u /* a wide block: a tile of gate and of up a warp */
#define EXPERT_RELAXED_WIDE_DOWN_OUTPUTS 256u /* two tiles of down a warp */
#define EXPERT_RELAXED_SCALE_SMALLEST \
	95u /* a weight 0.5 * 2^(c - 127) times 2^-100: 2^-133 */
#define EXPERT_RELAXED_SCALE_LARGEST \
	209u /* 2^14 products of 6 * 2^82 by 448 * 2^19: under 2^127 */
#define EXPERT_RELAXED_WIDE_MOST_INPUTS \
	16384u /* the 2^14 inputs of the largest scale code */
#define EXPERT_RELAXED_WIDE_GATE_UP_ARGUMENT_BYTES 56u
#define EXPERT_RELAXED_WIDE_DOWN_ARGUMENT_BYTES 48u

/* expert_relaxed_wide_gate_up_entry: gate and up of every tile's pairs, and the hidden step.
 *   reads   tiles_offset, 32 tile_count bytes; pairs_offset, 8 pair_count bytes; the tiles' gate and up matrices;
 *           operands_offset: operands [row][input_size] of the rows of the tiles' pairs (pair_index / pairs_per_row);
 *   writes  hidden_offset: values [pair][output_size], of the tiles' pairs only.
 * pairs_per_row is at least 1; activation_limit is the limit of the hidden step. */
struct expert_relaxed_wide_gate_up_arguments {
	uint64_t tiles_offset;
	uint64_t pairs_offset;
	uint64_t operands_offset;
	uint64_t hidden_offset;
	uint32_t output_size;
	uint32_t input_size;
	uint32_t tile_count;
	uint32_t pair_count;
	uint32_t pairs_per_row;
	float activation_limit;
};

/* expert_relaxed_wide_down: the down product of every tile's pairs from their quantized hidden vectors.
 *   reads   the tiles, the pairs and their down matrices; operands_offset: operands [pair][input_size] of the tiles'
 *           pairs;
 *   writes  output_offset: values [pair][output_size], of the tiles' pairs only. */
struct expert_relaxed_wide_down_arguments {
	uint64_t tiles_offset;
	uint64_t pairs_offset;
	uint64_t operands_offset;
	uint64_t output_offset;
	uint32_t output_size;
	uint32_t input_size;
	uint32_t tile_count;
	uint32_t pair_count;
};

LINEAR_RECORD_HAS(expert_relaxed_wide_gate_up_arguments,
		  EXPERT_RELAXED_WIDE_GATE_UP_ARGUMENT_BYTES);
LINEAR_RECORD_HAS(expert_relaxed_wide_down_arguments,
		  EXPERT_RELAXED_WIDE_DOWN_ARGUMENT_BYTES);

#endif
