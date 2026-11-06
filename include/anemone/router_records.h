/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * router_records.h: the records of the router's kernels. C11 and CUDA, including of the engine the record layouts of
 * the step, the model and the linear, thin and wide, alone. REF is the reference's inference/model.py, a number after
 * a colon a line.
 *
 * Contract:
 *   - router_select_entry(arguments, arena), one block of ROUTER_THREADS threads, a grid of 1, no dynamic
 *     shared memory, takes the L live rows: the step block's row_count, 1 to the record's (the checked modules trap
 *     on any other count, the release modules take the smaller). Of a row, for every expert: the affinity
 *     a = sqrt(softplus(s)) of its router score s, softplus(s) being s above 20, else log1p(exp(s)), the card's
 *     float32 functions (REF:811, 817); the six experts of the largest a + bias (REF:822), compared as the
 *     reference's top-k compares float32 (a not-a-number above every number, -0 below +0), the lower expert first
 *     among equals; their weights a / (t + 1e-20) * 1.5, each operation one float32 rounding (REF:823-826). t sums
 *     the six a from +0 in ascending expert order or, with reference_order 1, as the reference's library (torch
 *     2.14) reduces six values: ((g0 + g4) + g2) + ((g1 + g5) + g3), g in the top-k's order, descending a + bias.
 *     Among picks of equal a + bias that order is open in the reference (its library sorts the six unstably); ours
 *     puts the lower expert first.
 *   - It writes the affinities of the live rows and every byte of the job table (linear_records.h): a job for each
 *     expert a live row chose, in ascending expert order, with its three offsets from the table; a column for each
 *     row that chose it, rows ascending, holding the pair row * 6 + slot (slot: the expert's place among the row's
 *     six, ascending: the order of REF:895-900) and its weight; then empty jobs; every other byte zero. A row at or
 *     behind L stands in no job: the routed experts read nothing for it.
 *   - It uses no atomic operation, and what it computes for a row depends on that row alone. router.h checks the
 *     record and the host's copy of the table; the checked modules trap on an entry of the arena's copy that is not
 *     on the alignment from weights_offset to the last offset its matrix may begin at.
 *   - router_select_draft (dspark_v1/DESIGN.md C4): the select stamped for the draft's router,
 *     MODEL_DRAFT_EXPERTS experts and MODEL_DRAFT_EXPERTS_PER_ROW picks, in the fixed order alone
 *     (reference_order 0; the checked modules trap on 1): scores, affinities and bias of 128 experts a row, the first
 *     128 entries of the table, pairs row * 3 + slot, 3 jobs a row; everything else as above.
 *   - The wide select (wide_prefill.md 3.1): four launches of one record in this order on one card, over the L live
 *     rows of the step block at step_offset (the step area or a view): its row_count, 0 to the record's (the checked
 *     modules trap above, the release modules take the smaller). router_wide_select, grid[1] ceil(row_count /
 *     8) of ROUTER_THREADS: router_select_entry's affinities of each live row, eight a block, bit for bit,
 *     and its six picks by slot (picks [row][6]). With S = ROUTER_SORT_ROWS, router_wide_count, grid[1]
 *     ceil(row_count / S) of S: counts [b][384], the pairs of each expert among live rows S b .. S b + S - 1; then
 *     router_wide_place, the same grid: the first 6 L sorted pairs (wide_linear_records.h; a pair's place the
 *     pairs of lower experts and of its expert in lower rows), the 384 runs (an empty run's first_pair too) and all
 *     ROUTER_WIDE_TILES(row_count) tiles. A block of no live row writes nothing, but the place's block 0 its
 *     runs and tiles; then router_wide_jobs, grid[1] 1 of S: the job table (linear_records.h) of the runs, as
 *     the routed experts read a select's: each run cut in jobs of at most 8 pairs in their sorted order, the last
 *     ragged, the jobs in ascending expert order, each with its expert's three offsets, its pairs' pair_index and
 *     route_weight and as row_count their count; then empty jobs, every byte zero, to ROUTER_WIDE_JOBS
 *     (row_count) in all: an expert's rows take the fewest jobs. No atomic operation; a row's affinities and picks
 *     depend on it alone. The checked modules trap on a live pick that is no expert or not above its slot's before, a
 *     count the place does not find, a place at or beyond 6 row_count, more jobs than the table holds, and an entry of
 *     the arena's table as the select does.
 */
#ifndef ROUTER_RECORDS_H
#define ROUTER_RECORDS_H

#include <stdint.h>

#include "anemone/linear_records.h"
#include "anemone/model_layout.h"
#include "anemone/step_records.h"
#include "anemone/wide_linear_records.h"

#define KERNEL_FILE_ROUTER "kernel_router"
#define KERNEL_ROUTER_SELECT "router_select_entry"
#define KERNEL_ROUTER_WIDE_SELECT "router_wide_select"
#define KERNEL_ROUTER_WIDE_COUNT "router_wide_count"
#define KERNEL_ROUTER_WIDE_PLACE "router_wide_place"
#define KERNEL_ROUTER_WIDE_JOBS "router_wide_jobs"
#define KERNEL_ROUTER_WIDE_SMALL "router_wide_small"
#define KERNEL_ROUTER_SELECT_DRAFT "router_select_draft"
#define ROUTER_SELECT_ARGUMENT_BYTES 80u
#define ROUTER_WIDE_ARGUMENT_BYTES 120u
#define ROUTER_PICK_BYTES 8u
#define ROUTER_THREADS 256u /* a warp for each row a launch takes */
#define ROUTER_SORT_ROWS \
	384u /* ours: of a block of the count and the place, a row a thread */
#define ROUTER_SOFTPLUS_THRESHOLD \
	20.0f /* the default of F.softplus (REF:817) */
#define ROUTER_WEIGHT_EPSILON 1e-20f /* REF:825, "not norm_eps" */
/* The expert tiles of a wide select of row_count rows: each run cut in 64s, the last ragged (wide_linear_records.h). */
#define ROUTER_WIDE_TILES(row_count)                                    \
	(((row_count) * MODEL_EXPERTS_PER_ROW + WIDE_TILE_PAIRS - 1u) / \
		 WIDE_TILE_PAIRS +                                      \
	 MODEL_EXPERTS)
/* The jobs of a wide select of row_count rows: an expert's run of n pairs takes ceil(n / 8), so 6 R pairs over the 384
 * experts take at most (6 R + 7 * 384) / 8 of them, and never more than one a pair. */
#define ROUTER_WIDE_JOB_BOUND(row_count)                \
	(((row_count) * MODEL_EXPERTS_PER_ROW +         \
	  (LINEAR_MAXIMUM_ROWS - 1u) * MODEL_EXPERTS) / \
	 LINEAR_MAXIMUM_ROWS)
#define ROUTER_WIDE_JOBS(row_count)                         \
	((row_count) * MODEL_EXPERTS_PER_ROW <              \
			 ROUTER_WIDE_JOB_BOUND(row_count) ? \
		 (row_count) * MODEL_EXPERTS_PER_ROW :      \
		 ROUTER_WIDE_JOB_BOUND(row_count))
enum router_matrix {
	ROUTER_GATE,
	ROUTER_UP,
	ROUTER_DOWN
}; /* an expert's 3 entries */

struct router_select_arguments {
	uint64_t step_offset; /* read: struct step_block */
	uint64_t scores_offset; /* read: router scores [row_count][384] (model_layout.h); draft 128 */
	uint64_t bias_offset; /* read: float32 [384]; draft 128 */
	uint64_t expert_offsets_offset; /* read: expert offsets [384][3] of the layer (model_layout.h) */
	uint64_t affinities_offset; /* written: float32 [row_count][384], of the live rows; draft 128 */
	uint64_t jobs_offset; /* written: struct expert_job_record [6 row_count]; draft 3 */
	uint64_t weights_offset; /* the experts' range: its first byte, */
	uint64_t gate_up_last_offset; /* the last a gate or up matrix may begin at, */
	uint64_t down_last_offset; /* the last a down matrix may begin at */
	uint32_t row_count; /* the program's: 1 to LINEAR_MAXIMUM_ROWS */
	uint32_t reference_order; /* 0 or 1 */
};

struct router_pick { /* picks [row][6]: a row's experts by slot, ascending */
	uint32_t expert;
	float route_weight;
};

struct router_wide_arguments {
	uint64_t step_offset; /* read: struct step_block, of the step area or a view */
	uint64_t scores_offset; /* read: router scores [row_count][384] */
	uint64_t bias_offset; /* read: float32 [384] */
	uint64_t expert_offsets_offset; /* read: expert offsets [384][3] of the layer */
	uint64_t affinities_offset; /* written: float32 [row_count][384], of the live rows */
	uint64_t picks_offset; /* scratch: struct router_pick [row_count][6] */
	uint64_t counts_offset; /* scratch: uint32 [ceil(row_count / 384)][384] */
	uint64_t pairs_offset; /* written: struct wide_pair [6 row_count] */
	uint64_t runs_offset; /* written: struct wide_run [384] */
	uint64_t tiles_offset; /* written: struct wide_expert_tile [the tiles above] */
	uint64_t jobs_offset; /* written: struct expert_job_record [the jobs above] */
	uint64_t weights_offset; /* as the select's */
	uint64_t gate_up_last_offset;
	uint64_t down_last_offset;
	uint32_t row_count; /* the program's: 1 to MODEL_WIDE_MOST_ROWS */
	uint32_t reference_order; /* 0 or 1 */
};

LINEAR_RECORD_HAS(router_select_arguments, ROUTER_SELECT_ARGUMENT_BYTES);
LINEAR_RECORD_HAS(router_pick, ROUTER_PICK_BYTES);
LINEAR_RECORD_HAS(router_wide_arguments, ROUTER_WIDE_ARGUMENT_BYTES);
MODEL_LAYOUT_HOLDS(ROUTER_THREADS ==
			   LINEAR_MAXIMUM_ROWS * LINEAR_THREADS_PER_WARP &&
		   ROUTER_DOWN + 1u == MODEL_EXPERT_MATRICES);
MODEL_LAYOUT_HOLDS(ROUTER_SORT_ROWS == MODEL_EXPERTS &&
		   ROUTER_SORT_ROWS % LINEAR_THREADS_PER_WARP == 0);
MODEL_LAYOUT_HOLDS(MODEL_DRAFT_EXPERTS % LINEAR_THREADS_PER_WARP == 0);

#endif
