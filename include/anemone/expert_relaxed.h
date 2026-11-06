/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * expert_relaxed.h: the routed experts in the relaxed order of class 80 (expert_relaxed_records.h) on the cards: their
 * launches, prepared once at load from a description, as linear.h prepares the reference order's chain.
 *
 * Contract: linear.h's for prepare functions (every check without the arena's contents, one error line for a
 * refusal, launches_out untouched then, no card touched, nothing allocated), and:
 *   - expert_relaxed_kernels_prepare asks a card of class ARITHMETIC_CLASS_SM_80 for ten kernels,
 *     once, in the order of the members; any other class is STATUS_UNSUPPORTED with one error line (the
 *     relaxed order is the CMPs': class 120 takes its native forms or the exact emulation). The first kernel_prepare
 *     that fails ends it with that status; a NULL argument or a table without kernel_prepare is
 *     STATUS_INVALID_ARGUMENT. On failure kernels_out is untouched.
 *   - expert_relaxed_prepare gives the launches of linear_routed_experts_prepare for
 *     description->experts with gate-up and down the relaxed kernels': the quantization of the hidden vectors and the
 *     row sum as that function gives them. tile_count 0 takes the narrow form over the job table: four launches in the
 *     order of enum linear_routed_experts_launch. More takes the wide form over tile_count tiles of a wide
 *     select: six launches in the order of enum expert_relaxed_wide_launch (each product's plain kernel, then
 *     its small one), its tiles and sorted pairs ranges checked as every range is (prepare_common.h), against each
 *     other and against every range of the experts' description. It refuses what that function refuses (the slice
 *     counts too, which the relaxed kernels do not read: give the model's), a reserved field that is not zero, a
 *     relaxed kernel not prepared, an output_size or input_size its form does not divide (expert_relaxed_records.h),
 *     and of the wide form an input_size above EXPERT_RELAXED_WIDE_MOST_INPUTS.
 *   - expert_relaxed_fold_check is the records header's condition on one matrix's scale codes (the bytes
 *     behind its codes, output_size * input_size / 32 of them), read where they lie on the host at load: every code
 *     from EXPERT_RELAXED_SCALE_SMALLEST to EXPERT_RELAXED_SCALE_LARGEST, where the fold is exact and
 *     no sum of the wide form can pass the largest float32 where the reference order's does not.
 *     Else STATUS_INVALID_ARGUMENT with one error line naming the least and the largest code: the experts
 *     of that layer are then served in the reference order. A missing argument or no code is refused too.
 *   - Not checked here, because it is not known at load: what the arena holds. Whoever writes the job table or the
 *     tiles and pairs holds them to linear_records.h and wide_linear_records.h: a tile of at most 64 pairs, its pairs
 *     inside the pairs array, every pair_index below row_count * pairs_per_row, every matrix of the description's
 *     shape inside expert_weights.
 */
#ifndef EXPERT_RELAXED_H
#define EXPERT_RELAXED_H

#include <stdint.h>
#include <stdbool.h>

#include "anemone/backend_seam.h"
#include "anemone/expert_relaxed_records.h"
#include "anemone/linear.h"
#include "anemone/status.h"

/* The first measured chunk size that benefits from the ordered tile chain. */
#define EXPERT_ORDERED_MINIMUM_ROWS 512u

struct expert_relaxed_kernels {
	uint32_t gate_up; /* expert_relaxed_gate_up: narrow, the job table */
	uint32_t down; /* expert_relaxed_down */
	uint32_t wide_gate_up; /* expert_relaxed_wide_gate_up_entry: wide, the tiles */
	uint32_t wide_down; /* expert_relaxed_wide_down */
	uint32_t wide_small_gate_up; /* expert_relaxed_wide_small_gate_up: the small tiles */
	uint32_t wide_small_down; /* expert_relaxed_wide_small_down */
	uint32_t ordered_wide_gate_up;
	uint32_t ordered_wide_down;
	uint32_t ordered_wide_small_gate_up;
	uint32_t ordered_wide_small_down;
};

/* The wide form's launches, in the order they run. */
enum expert_relaxed_wide_launch {
	EXPERT_RELAXED_WIDE_GATE_UP =
		0, /* the tiles of more than EXPERT_RELAXED_SMALL_TILE_PAIRS */
	EXPERT_RELAXED_WIDE_SMALL_GATE_UP = 1, /* the others */
	EXPERT_RELAXED_WIDE_QUANTIZE =
		2, /* linear_routed_experts_prepare's quantization */
	EXPERT_RELAXED_WIDE_DOWN = 3,
	EXPERT_RELAXED_WIDE_SMALL_DOWN = 4,
	EXPERT_RELAXED_WIDE_ROW_SUM =
		5, /* linear_routed_experts_prepare's row sum */
	EXPERT_RELAXED_WIDE_LAUNCH_COUNT = 6
};

MUST_CHECK result_status expert_relaxed_kernels_prepare(
	const struct backend_operations *backend, struct device_card *card,
	uint32_t arithmetic_class, struct expert_relaxed_kernels *kernels_out);

struct expert_relaxed_description {
	struct linear_routed_experts_description
		experts; /* the chain's arrays, as the reference order's */
	struct linear_range
		tiles; /* wide: read: struct wide_expert_tile [tile_count] */
	struct linear_range
		pairs; /* wide: read: struct wide_pair [row_count * pairs_per_row] */
	uint32_t tile_count; /* 0: narrow; else wide (grid[1] of its two products) */
	uint32_t reserved_zero;
};
MUST_CHECK result_status expert_relaxed_fold_check(const uint8_t *scale_codes,
						   uint64_t scale_count);

/* launches_out: LINEAR_ROUTED_EXPERTS_LAUNCH_COUNT launches (narrow), EXPERT_RELAXED_WIDE_LAUNCH_COUNT
 * (wide). */
MUST_CHECK result_status
expert_relaxed_prepare(const struct expert_relaxed_description *description,
		       uint64_t arena_byte_count,
		       const struct linear_kernels_record *linear_kernels,
		       const struct expert_relaxed_kernels *kernels,
		       struct kernel_launch *launches_out);

/* Same wide layouts, staging and six launches as the relaxed form, with the reference K32 chain. No scale-fold
 * admission or relaxed input-width cap is needed. A description without tiles is refused. Both wide forms refuse
 * more than EXPERT_WIDE_MOST_TILES tiles (the CUDA grid Y limit). Failure leaves launches_out untouched. */
MUST_CHECK result_status
expert_ordered_prepare(const struct expert_relaxed_description *description,
		       uint64_t arena_byte_count,
		       const struct linear_kernels_record *linear_kernels,
		       const struct expert_relaxed_kernels *kernels,
		       struct kernel_launch *launches_out);

/* Validate the reference description first. On class 80 select ordered wide at EXPERT_ORDERED_MINIMUM_ROWS
 * and above. Below it, reference order or a refused fold keeps the original four launches. Admitted relaxed order
 * selects wide above eight rows and narrow below, only when cmp_order requests it. Both PRO classes keep the
 * original four launches. Missing tiles for a selected wide form are refused. Only selected handles are required.
 * Both outputs are untouched on refusal; launches_out has room for
 * EXPERT_RELAXED_WIDE_LAUNCH_COUNT entries. Fold admission covers every routed matrix of the layer. */
MUST_CHECK result_status expert_select_prepare(
	const struct expert_relaxed_description *description,
	uint64_t arena_byte_count,
	const struct linear_kernels_record *linear_kernels,
	const struct expert_relaxed_kernels *kernels, uint32_t arithmetic_class,
	enum cmp_order_code cmp_order, bool folds_admitted,
	struct kernel_launch *launches_out, uint32_t *launch_count_out);

#endif
