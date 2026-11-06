/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * router.h: the launches of the router of one layer (router_records.h), prepared once at load: the select, whose job
 * table the routed experts of the layer read (linear_routed_experts_prepare, linear.h), and the wide select,
 * whose sorted pairs, runs and expert tiles the wide products read (wide_linear_records.h) and whose job table the
 * routed experts read as a select's.
 *
 * Contract: linear.h's for prepare functions (arithmetic on a description; a refusal is
 * STATUS_INVALID_ARGUMENT with one error line and the launches untouched), and:
 *   - expert_count and pick_count: MODEL_EXPERTS and MODEL_EXPERTS_PER_ROW (router_select_entry), or
 *     MODEL_DRAFT_EXPERTS and MODEL_DRAFT_EXPERTS_PER_ROW (router_select_draft, the draft's
 *     router, of the fixed order alone: dspark_v1/DESIGN.md C4); any other pair is refused. E and P below are the pair.
 *   - experts is the description of those routed experts, its reserved field zero: 1 to
 *     LINEAR_MAXIMUM_ROWS rows of P pairs, P jobs a row, E2M1-scaled or EXL3 mul1 K2/K3 matrices in
 *     shapes the format stores. EXL3 uses separate pass preparation; the native linear factory remains E2M1-only.
 *     expert_offsets is the table the model writes at offsets at load ([384][3] whatever E: the
 *     first E entries are read): each entry a multiple of LINEAR_OFFSET_ALIGNMENT_BYTES, its matrix (gate and
 *     up: gate_up_shape, down: down_shape) wholly inside the experts' range. So every job the kernel writes, whatever
 *     the scores, is one linear.h asks of whoever writes a job table. The ranges, with all of experts', are checked as
 *     linear.h's: the job table and the experts' range for what the kernel touches, the nine others for sharing no
 *     byte (their prepare sizes them); scores, bias, offsets and affinities hold E experts.
 *   - order: LINEAR_ORDER_FIXED (ascending expert order) or LINEAR_ORDER_REFERENCE; the draft's
 *     router refuses the reference order (its 3-term order is not captured).
 *   - The wide select: the same table and order, 1 to MODEL_WIDE_MOST_ROWS rows, its reserved field zero, each
 *     range its own, sized for row_count rows as its comment says. Its tiles name matrices as jobs do, checked here
 *     alone, for expert_weights (untouched), both shapes and the format: the wide products' prepare must refuse others.
 */
#ifndef ROUTER_H
#define ROUTER_H

#include <stdint.h>

#include "anemone/backend_seam.h"
#include "anemone/linear.h"
#include "anemone/router_records.h"
#include "anemone/status.h"

struct router_kernels_record {
	uint32_t select;
	uint32_t wide_select;
	uint32_t wide_count;
	uint32_t wide_place;
	uint32_t wide_jobs;
	uint32_t wide_small; /* the count, place and jobs of at most ROUTER_SORT_ROWS rows in one launch */
	uint32_t select_draft; /* the draft's: draft_kernels_prepare (draft.h) */
};

/* First user: the model's load, for each open card. Fails as linear_kernels_prepare does; prepares the first
 * five and leaves select_draft BACKEND_NO_KERNEL (a card without the draft keeps its handles: draft.h). */
MUST_CHECK result_status router_kernels_prepare(
	const struct backend_operations *backend, struct device_card *card,
	struct router_kernels_record *kernels_out);

struct router_description {
	const struct linear_routed_experts_description *experts;
	const uint64_t
		*expert_offsets; /* [384][3]: what the arena holds at offsets */
	struct linear_range step; /* read: the step block */
	struct linear_range scores; /* read: router scores [row_count][384] */
	struct linear_range bias; /* read: float32 [384] */
	struct linear_range offsets; /* read: expert offsets [384][3] */
	struct linear_range affinities; /* written: float32 [row_count][384] */
	enum linear_order_code order;
	uint32_t expert_count; /* 384, or 128 for the draft's router */
	uint32_t pick_count; /* 6, or 3 for the draft's router */
};
MUST_CHECK result_status router_select_prepare(
	const struct router_description *description, uint64_t arena_byte_count,
	const struct router_kernels_record *kernels,
	struct kernel_launch *launch_out);

enum router_wide_launch {
	ROUTER_WIDE_SELECT_LAUNCH =
		0, /* router_wide_select: scores to affinities and picks */
	ROUTER_WIDE_COUNT_LAUNCH = 1, /* router_wide_count: picks to counts */
	ROUTER_WIDE_PLACE_LAUNCH =
		2, /* router_wide_place: picks, counts to pairs, runs, tiles */
	ROUTER_WIDE_JOBS_LAUNCH =
		3, /* router_wide_jobs: pairs, runs to the job table */
	ROUTER_WIDE_LAUNCH_COUNT = 4
};
struct router_wide_description {
	const uint64_t
		*expert_offsets; /* [384][3]: what the arena holds at offsets */
	struct linear_range
		step; /* read: the step block, or a view's (model_seam.h) */
	struct linear_range scores; /* read: router scores [row_count][384] */
	struct linear_range bias; /* read: float32 [384] */
	struct linear_range offsets; /* read: expert offsets [384][3] */
	struct linear_range
		expert_weights; /* holds every matrix the table names */
	struct linear_range affinities; /* written: float32 [row_count][384] */
	struct linear_range
		picks; /* scratch: struct router_pick [row_count][6] */
	struct linear_range
		counts; /* scratch: uint32 [ceil(row_count / 384)][384] */
	struct linear_range pairs; /* written: struct wide_pair [6 row_count] */
	struct linear_range runs; /* written: struct wide_run [384] */
	struct linear_range
		tiles; /* written: struct wide_expert_tile [the tiles of row_count] */
	struct linear_range
		jobs; /* written: struct expert_job_record [the jobs of row_count] */
	struct matrix_shape_record
		gate_up_shape; /* of the gate and of the up matrix of every expert */
	struct matrix_shape_record down_shape;
	enum weight_format_code
		weight_format; /* E2M1 or EXL3; product preparation stays separately selected */
	enum linear_order_code order;
	uint32_t row_count;
	uint32_t reserved_zero;
};
/* launches_out: ROUTER_WIDE_LAUNCH_COUNT launches, run in their order. */
MUST_CHECK result_status router_wide_prepare(
	const struct router_wide_description *description,
	uint64_t arena_byte_count, const struct router_kernels_record *kernels,
	struct kernel_launch *launches_out);
/* The same of at most ROUTER_SORT_ROWS rows in ROUTER_WIDE_SMALL_LAUNCH_COUNT launches: the select,
 * then router_wide_small (the count, the place and the jobs in one block: the same picks, affinities, pairs,
 * runs, tiles and jobs; the counts scratch unwritten). More rows: STATUS_INVALID_ARGUMENT. */
#define ROUTER_WIDE_SMALL_LAUNCH_COUNT 2u
MUST_CHECK result_status router_wide_small_prepare(
	const struct router_wide_description *description,
	uint64_t arena_byte_count, const struct router_kernels_record *kernels,
	struct kernel_launch *launches_out);

#endif
