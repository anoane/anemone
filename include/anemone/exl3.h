/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* EXL3 mul1 K2/K3 admission. A refusal leaves scalar outputs unchanged.
 * This is load-time preparation; no allocation or format dispatch per token. */
#ifndef EXL3_H
#define EXL3_H

#include "anemone/backend_seam.h"
#include "anemone/exl3_records.h"
#include "anemone/weight_format_seam.h"

extern const struct weight_format_operations exl3_weight_operations;

MUST_CHECK result_status exl3_reconstruct_prepare(
	const struct exl3_reconstruct_record *record, uint64_t arena_byte_count,
	uint32_t kernel, struct kernel_launch *launch_out);

/* Expert jobs retain linear_records.h's ownership contract: at most eight rows
 * per job, unique in-range pairs and three aligned complete matrices in the
 * caller-admitted resident expert range. The model's router preparation validates
 * matrix offsets; this function validates disjoint activation/scratch ranges.
 * The finish output equals products_offset: each block loads and transforms only
 * its own 128 values before overwriting them with FP32 routed outputs. The caller
 * follows the five passes with exl3_row_sum and the ordinary hyper post.
 * Scalar outputs and the whole launch table remain unchanged on refusal. */

/* Shared scratch bytes for a maximum of pair_count routed assignments. */
uint64_t exl3_scratch_byte_count(uint32_t pair_count, uint32_t input_size);

MUST_CHECK result_status exl3_experts_prepare(
	const struct exl3_expert_record *record, uint64_t arena_byte_count,
	const struct backend_operations *backend, struct device_card *card,
	struct kernel_launch *launches_out);

/* The same five launches with kernel_exl3_wide's products and transforms (exl3_v3_claude; identical words): the
 * transforms exl3_w_{input,activate,finish} (grid x = (width / 128 + 3) / 4, block 128); the products
 * exl3_c_{gate_up,down}_k{2,3}_t8_g1 (grid x = width / 128) when a launch has at most
 * LINEAR_MAXIMUM_ROWS rows (pair_count / pairs_per_row), else _t4_g4p (grid x = width / 64: the wide
 * select's runs of 8-row jobs, an expert's adjacent jobs sharing one decode). Same record, jobs, ranges and order
 * of launches as exl3_experts_prepare, which remains the retained form; widths must be multiples of 128. */
#define EXL3_WIDE_KERNEL_FILE "kernel_exl3_wide"
MUST_CHECK result_status exl3_experts_fast_prepare(
	const struct exl3_expert_record *record, uint64_t arena_byte_count,
	const struct backend_operations *backend, struct device_card *card,
	struct kernel_launch *launches_out);
/* Explicit alternative: eight contiguous K partials change FP32 association.
 * The caller must select relaxed math. The ordinary preparation remains exact
 * relative to this EXL3 implementation's four-part arithmetic. */
MUST_CHECK result_status exl3_experts_relaxed_prepare(
	const struct exl3_expert_record *record, uint64_t arena_byte_count,
	const struct backend_operations *backend, struct device_card *card,
	struct kernel_launch *launches_out);

#endif
