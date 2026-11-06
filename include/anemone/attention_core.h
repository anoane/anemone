/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * attention_core.h: the launch of the attention core of one layer (attention_core_records.h), prepared once at load
 * from a description that names every arena offset and count, in the order it names.
 *
 * Contract:
 *   - linear.h's rules for a prepare function hold: nothing here runs per token, touches a card, allocates or reads
 *     an arena; the launch is run as it is, as often as the model likes; any thread may call either function.
 *   - attention_core_kernels_prepare asks a card for the three kernels once: the fixed order's, then the
 *     reference order's two (of KERNEL_FILE_ATTENTION_CORE_NATIVE on a card of class
 *     ARITHMETIC_CLASS_SM_120_NATIVE). A NULL argument or a table without kernel_prepare is INVALID_ARGUMENT; a
 *     kernel_prepare that fails ends it with its status behind one error line that names the kernel. On failure
 *     kernels_out is untouched.
 *   - attention_core_prepare fills the order's launches (one for the fixed order, two for the reference
 *     order: attention_core_records.h; window_mode BLOCK: the block core's one) and their count. It refuses, with
 *     STATUS_INVALID_ARGUMENT, one error line and the launches left untouched:
 *       arguments  a NULL argument; the handle of one of the order's kernels BACKEND_NO_KERNEL;
 *       counts     (one line that states them all) an order outside enum linear_order_code; a row_count of 0
 *                  or above LINEAR_WIDE_MOST_ROWS (the block core: 64, a BATCH step's, each block of a
 *                  launch a draft's rows); a session_capacity of 0; a context_token_limit of 0, not a
 *                  multiple of MODEL_CONTEXT_GRANULE or above MODEL_CONTEXT_MOST; a
 *                  pool_ratio other than 0, MODEL_DIRECT_RATIO and MODEL_POOLED_RATIO; a first_head
 *                  or head_count that is no multiple of ATTENTION_CORE_HEADS_PER_BLOCK, a head_count of 0,
 *                  heads beyond MODEL_HEADS_VALUE; a window_mode other than CAUSAL and BLOCK, or BLOCK with a pool
 *                  or in the reference order;
 *       ranges     prepare_common.h's rules: on a multiple of LINEAR_OFFSET_ALIGNMENT_BYTES, inside the
 *                  arena, at least the bytes of its whole array (the step area of row_count rows, every array of
 *                  row_count rows, of session_capacity sessions, of the 64 heads and the 3 uses), no two sharing a
 *                  byte. Without a pool, pool and selection are neither read nor checked; in the fixed order
 *                  neither are the scores.
 *   - The rotation's use follows the pool, as the layer's theta does (REF:680-687): use 0 for the window alone, use 1
 *     with a pool. A session's ring and pool follow the previous session's, nothing between (model_layout.h).
 *   - Not checked here, as it is not known at load: the step rows (the step driver checks them) and the entries of
 *     a selection (its writer's contract): attention_core_records.h says what the kernel does with a broken one.
 */
#ifndef ATTENTION_CORE_H
#define ATTENTION_CORE_H

#include <stdint.h>

#include "anemone/attention_core_records.h"
#include "anemone/backend_seam.h"
#include "anemone/linear.h"
#include "anemone/status.h"

struct attention_core_kernels {
	uint32_t attention_core;
	uint32_t attention_core_scores; /* of LINEAR_ORDER_REFERENCE, then the next */
	uint32_t attention_core_values;
	uint32_t attention_core_block; /* the draft's: draft_kernels_prepare (draft.h) */
};

/* First user: the model's load, once for each open card that runs an attention core. It prepares the first three and
 * leaves attention_core_block BACKEND_NO_KERNEL (a card without the draft keeps its handles: draft.h). */
MUST_CHECK result_status attention_core_kernels_prepare(
	const struct backend_operations *backend, struct device_card *card,
	uint32_t arithmetic_class, struct attention_core_kernels *kernels_out);

/* The core of one layer for steps of up to row_count rows: one launch of attention_core_entry, or in the
 * reference order (P as the oracle's stand-in rounds it) the launches of its scores and of its values; with
 * window_mode ATTENTION_CORE_WINDOW_BLOCK the draft's block core (fixed order, no pool). */
#define ATTENTION_CORE_WINDOW_CAUSAL \
	0u /* the backbone's window: positions p - 127 .. p of a row */
#define ATTENTION_CORE_WINDOW_BLOCK \
	1u /* the draft's block (attention_core_records.h: the block core) */
struct attention_core_description {
	struct linear_range
		step; /* read: the step area, its block and row_count rows */
	struct linear_range queries; /* read: FP16 [row_count][64][512] */
	struct linear_range
		ring; /* read: FP16 [session_capacity][ring slots][512] */
	struct linear_range
		pool; /* read: FP16 [session_capacity][context / pool_ratio][512] */
	struct linear_range selection; /* read: int32 [row_count][512] */
	struct linear_range sinks; /* read: float32 [64] */
	struct linear_range rotation; /* read: float32 [row_count][3][32][2] */
	struct linear_range
		output; /* written: bfloat16 [row_count][64][512] (output_compact:
                                                   [row_count][head_count][512], the launch's heads alone) */
	struct linear_range
		scores; /* the reference order's scratch: float32 [row_count][64][640] */
	uint32_t row_count;
	uint32_t session_capacity;
	uint32_t context_token_limit;
	uint32_t pool_ratio; /* 0: the window alone */
	uint32_t first_head;
	uint32_t head_count;
	enum linear_order_code order;
	uint32_t arithmetic_class; /* of the card (backend_seam.h): the launch of the exact profile */
	uint32_t window_mode; /* ATTENTION_CORE_WINDOW_CAUSAL or _BLOCK */
	uint32_t output_compact; /* 0 or 1: a TP rank's output rows of its own heads */
	uint32_t cone_layer; /* 0, or the layer (a decoder layer, model_layout.h) whose rows the
                                                   step flags STEP_ROW_CONE past it skip (step_records.h) */
	uint32_t most_heads_per_block; /* 0, or the reference order's most heads a block (a multiple of 8):
                                                   more blocks, the same bits (attention_core_records.h) */
};
MUST_CHECK result_status attention_core_prepare(
	const struct attention_core_description *description,
	uint64_t arena_byte_count, const struct attention_core_kernels *kernels,
	struct kernel_launch launches_out[ATTENTION_CORE_MOST_LAUNCHES],
	uint32_t *launch_count_out);

#endif
