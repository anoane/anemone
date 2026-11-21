/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * ds41f_index_scan.h: the index scan and its selection on the cards: the launches of ds41f_index_scan_records.h for one
 * index layer, prepared once at load from a description that names every arena offset and count.
 *
 * Contract: linear.h's rules for prepare functions and the checks of prepare_checks.h hold, and:
 *   - A description is one index layer's (2, 8, 14, 20, 24, 28, 32, 36): its index queries, weights and scores, the
 *     key tiles of its own group's owner (no shared slot), and the selection the layers of its group read. The key
 *     tiles of a session follow the previous session's, (context_token_limit / ratio) * DS41F_KEY_BYTES on,
 *     nothing between (ds41f_geometry.h).
 *   - A launch may take a view's rows from step_row_first on (its records' row_first); split_rows (a slab's rows over
 *     several slabs) keeps each row's selection by the kernels a launch of so many rows would take.
 *   - Refused also: a row_count of 0 or above STEP_WIDE_MOST_ROWS, with step_row_first or split_rows above
 *     it; a ratio other than
 *     DS41F_DIRECT_RATIO and DS41F_POOLED_RATIO; a context_token_limit of 0, not a multiple of
 *     DS41F_CONTEXT_GRANULE or above DS41F_CONTEXT_MOST; a session_capacity of 0; a candidate role
 *     with a ratio other than DS41F_DIRECT_RATIO.
 *   - The candidate source (layer 20) also writes its rows' candidates; a candidate reader (24, 28, 32, 36) scores
 *     and selects among them (ds41f_index_scan_records.h), with scores of DS41F_CONTEXT_WITHOUT_CANDIDATES a row.
 *   - The step rows are the step driver's (step_records.h); the checked modules trap on a row it forbids.
 *   - At most DS41F_INDEX_SPLIT_MAX_ROWS rows and at least DS41F_INDEX_SPLIT_MIN_ENTRIES score slots per row
 *     use split ordinary selection; other shapes retain the legacy selection. A candidate source splits its
 *     block selection from DS41F_INDEX_SPLIT_MIN_CANDIDATE_ENTRIES score slots. Readers retain legacy selection.
 *     From DS41F_INDEX_SPLIT_DEEP_MIN_ENTRIES slots, rows through DS41F_INDEX_SPLIT_DEEP_MAX_ROWS also split.
 *     Split shapes require a scratch annex immediately AFTER the declared scores range, of
 *     row_count * DS41F_INDEX_SPLIT_ROW_BYTES. The scores range itself names only the payload (possibly padded).
 *     The annex is checked against the arena and every other range, and is reused by ordinary and candidate
 *     selection in stream order. No other launch may use it concurrently. model_v41.c reserves it at load.
 *     A missing split handle, annex overflow, truncation or overlap is refused before publishing any output.
 */
#ifndef DS41F_INDEX_SCAN_H
#define DS41F_INDEX_SCAN_H

#include <stdint.h>

#include "engine/backend/backend_seam.h"
#include "ds41f_index_scan_records.h"
#include "engine/library/block_scaled_linear/linear.h"
#include "common/status.h"

#define DS41F_INDEX_SPLIT_MIN_ENTRIES 32768u
#define DS41F_INDEX_SPLIT_MIN_CANDIDATE_ENTRIES 131072u
#define DS41F_INDEX_SPLIT_MAX_ROWS 8u
#define DS41F_INDEX_SPLIT_DEEP_MIN_ENTRIES 524288u
#define DS41F_INDEX_SPLIT_DEEP_MAX_ROWS 32u
#define DS41F_INDEX_SCAN_KERNEL_COUNT (3u + DS41F_INDEX_SPLIT_LAUNCH_COUNT)

enum ds41f_index_split_launch {
	DS41F_INDEX_SPLIT_HISTOGRAM_HIGH = 0,
	DS41F_INDEX_SPLIT_CUT_HIGH,
	DS41F_INDEX_SPLIT_HISTOGRAM_LOW,
	DS41F_INDEX_SPLIT_CUT_LOW,
	DS41F_INDEX_SPLIT_OUTPUT
};

/* The legacy handles, then the five split stages, on one card. */
struct ds41f_index_scan_kernels {
	uint32_t scan;
	uint32_t select;
	uint32_t candidates;
	uint32_t split[DS41F_INDEX_SPLIT_LAUNCH_COUNT];
};

/* First user: the model's load, once for each open card that runs an index layer; on a card of class
 * ARITHMETIC_CLASS_SM_120_NATIVE of the file's native build. Fails as linear_kernels_prepare does. */
MUST_CHECK result_status ds41f_index_scan_kernels_prepare(
	const struct backend_operations *backend, struct device_card *card,
	uint32_t arithmetic_class,
	struct ds41f_index_scan_kernels *kernels_out);

enum ds41f_index_scan_launch {
	DS41F_INDEX_SCAN_LAUNCH_SCAN =
		0, /* index_scan_entry: queries, weights and key tiles to scores */
	DS41F_INDEX_SCAN_LAUNCH_SELECT =
		1, /* index_select_entry: scores to the selection */
	DS41F_INDEX_SCAN_LAUNCH_CANDIDATES =
		2, /* index_candidates: the candidate source's alone */
	DS41F_INDEX_SCAN_LEGACY_LAUNCH_COUNT = 3,
	DS41F_INDEX_SCAN_LAUNCH_COUNT = 1 +
		2 * DS41F_INDEX_SPLIT_LAUNCH_COUNT /* scan, selection, candidates */
};

enum ds41f_index_candidate_role {
	DS41F_INDEX_CANDIDATES_NONE = 0,
	DS41F_INDEX_CANDIDATES_SOURCE =
		1, /* writes the candidates behind its selection */
	DS41F_INDEX_CANDIDATES_READER =
		2 /* scores and selects among the candidates */
};

struct ds41f_index_scan_description {
	struct linear_range
		step; /* read: the step block and row_count rows (step_records.h) */
	struct linear_range
		keys; /* read: key tiles [session_capacity][context / ratio / 8][2,048 B] */
	struct linear_range
		queries; /* read: index queries [row_count][8,192 B] */
	struct linear_range
		weights; /* read: index weights bfloat16 [row_count][32] */
	struct linear_range
		scores; /* scratch: FP16 [row_count][context / ratio, or 16,384: a reader] */
	struct linear_range selection; /* written: int32 [row_count][512] */
	struct linear_range
		candidates; /* uint32 [row_count][2,048]: written by the source, else read */
	uint32_t context_token_limit; /* every position lies below it */
	uint32_t session_capacity;
	uint32_t ratio; /* positions an entry holds */
	uint32_t row_count;
	uint32_t candidate_role; /* enum ds41f_index_candidate_role */
	uint32_t step_row_first; /* the step row of row 0: step covers it and row_count rows more */
	uint32_t split_rows; /* the rows whose shape picks split or legacy selection; 0: row_count */
};
/* launches_out holds DS41F_INDEX_SCAN_LAUNCH_COUNT slots: run the published count in order. Legacy shapes have
 * two launches (three for a source); split shapes six (seven or eleven for a source). A refusal leaves both intact. */
MUST_CHECK result_status ds41f_index_scan_prepare(
	const struct ds41f_index_scan_description *description,
	uint64_t arena_byte_count,
	const struct ds41f_index_scan_kernels *kernels,
	struct kernel_launch *launches_out, uint32_t *launch_count_out);

#endif
