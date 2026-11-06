/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Independently prepare the existing EXL3 passes around lossless TP gathers. */
#ifndef BB_EXL3_PASS_H
#define BB_EXL3_PASS_H

#include "anemone/backend_seam.h"
#include "anemone/linear_records.h"
#include "anemone/exl3_layout.h"
#include "anemone/exl3_tp_records.h"

#define BB_EXL3_PRODUCT_KERNEL_FILE "kernel_exl3_product"
#define BB_EXL3_PRODUCT_TWO_TILES 2u
#define BB_EXL3_PRODUCT_FOUR_TILES 4u
#define BB_EXL3_PRODUCT_EIGHT_TILES 8u		/* kernel_exl3_wide's _t8_g1 (exl3_v3_claude) */
#define BB_EXL3_WIDE_KERNEL_FILE "kernel_exl3_wide"

enum bb_exl3_pass {
	BB_EXL3_INPUT_PASS = 0,
	BB_EXL3_GATE_UP_PASS = 1,
	BB_EXL3_ACTIVATE_PASS = 2,
	BB_EXL3_DOWN_PASS = 3,
	BB_EXL3_FINISH_PASS = 4,
	BB_EXL3_PASSES = 5
};

struct bb_exl3_pass_description {
	const struct bb_exl3_output_shard_plan *plan;
	struct exl3_expert_record record;
	uint64_t arena_bytes;
	uint64_t weight_offset;
	uint64_t weight_bytes;
	enum bb_exl3_pass pass;
	uint32_t product_tiles; /* zero retains original products; 1/2/4/8 select measured identical-word C products;
				   nonzero also selects kernel_exl3_wide's w_input/w_finish transforms, and a launch of
				   more than 8 rows (pair_count / pairs_per_row: the wide select's table) the grouped
				   _t4_g4p products (reference order) */
};

/* Load-time only. Refusal leaves launch_out unchanged. No execution/allocation.
 * The caller prepares the router's common job table: at most eight rows/job,
 * unique in-range pairs, zero reserved fields, finite original route weights;
 * every job names aligned complete gate/up and down matrices inside weights.
 * Their shapes are plan.gate_up and plan.down, respectively. Scale bytes must
 * pass the original loader's finite/range validation; no host tensor payload is
 * supplied here. Weight storage is disjoint from jobs and every touched buffer.
 *
 * INPUT/GATE_UP/ACTIVATE records use full outer input_size/local inner_size.
 * DOWN/FINISH use local outer input_size/full inner_size. Reference/optional
 * product order is read only from the validated plan. All jobs and scratch are
 * caller-owned until local execution and both lossless gathers have drained.
 * The DOWN operands range reserves full intermediate FP16 data for every pair;
 * the gather must fill every channel, in ascending global intermediate order.
 * FINISH retains output_offset==products_offset. It applies original scales and
 * route weights. Then bb_exl3_row_sum_prepare preserves original slot order on
 * local output channels. Gather those summed FP32 row channel shards, followed
 * by original shared-expert/post; no numerical rank reduction occurs.
 *
 * Existing input/gate/up/down/finish entries use kernel_exl3. Activation uses
 * BB_EXL3_SHARD_KERNEL_FILE and the 96-byte adapter record. The engine embeds that activation entry in
 * kernel_exl3 beside the retained products; otherwise kernel_prepare must refuse it.
 * product_tiles is an independent product implementation choice, never a change of arithmetic order.
 * Four-part admits1/2/4tiles; the explicit eight-part alternative admits2/4tiles.
 * Nonzero selection requires the separately embedded product module; otherwise preparation refuses.
 * This helper never changes production module registration or launch tables. */
MUST_CHECK result_status bb_exl3_pass_prepare(
	const struct bb_exl3_pass_description *description,
	const struct backend_operations *backend, struct device_card *card,
	struct kernel_launch *launch_out);

/* Existing exl3_row_sum, with no arithmetic change. For a finish record:
 * record.pair_output_offset=finish.output_offset;
 * record.width=plan.down.output_count;
 * record.pairs_per_row=finish.pairs_per_row;
 * rows=finish.pair_count/finish.pairs_per_row.
 * row_output_offset names disjoint FP32 [rows][local width] scratch. Both arrays
 * remain owned through the output gather. Refusal leaves launch_out unchanged. */
MUST_CHECK result_status bb_exl3_row_sum_prepare(
	const struct expert_row_sum_arguments *record, uint32_t rows,
	uint64_t arena_bytes, const struct backend_operations *backend,
	struct device_card *card, struct kernel_launch *launch_out);

#endif
