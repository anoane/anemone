/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * draft.h: the draft's kernels on its card (dspark_v1/DESIGN.md 5.3): the launches of kernel_draft's kernels
 * (draft_records.h), each prepared once at load from a description that names every arena offset, and the one call
 * that prepares every kernel the draft adds.
 *
 * Contract: linear.h's for the prepare functions (arithmetic on a description alone, any thread; a refusal is
 * STATUS_INVALID_ARGUMENT with one error line and the launches untouched; ranges by prepare_common.h's rules:
 * aligned, inside the arena, holding what the launches touch, a tensor of the pack exactly its bytes, no two of a
 * description sharing a byte). Also refused: a kernel not prepared; a reserved field not zero; a row_count of 0 or
 * above LINEAR_MAXIMUM_ROWS; draft_positions of 0 or above STEP_DRAFT_MOST; a position at or past
 * them; a vocabulary_size of 0, or an end token or the noise token at or past it.
 *   - draft_kernels_prepare prepares, on the one card that holds the draft (the result card) and only where
 *     the draft is admitted, after the families' own prepares: the draft's members of the hyper-connection, router,
 *     attention state and attention core families (draft_inputs and final_kept, select_draft, window,
 *     attention_core_block: the core's of KERNEL_FILE_ATTENTION_CORE_NATIVE for a class of
 *     ARITHMETIC_CLASS_SM_120_NATIVE, the draft's class on the PRO whatever the target's) and the kernels of
 *     kernel_draft into kernels_out. A card without the draft prepares none of them, so its handles and its programs
 *     stay what they were. A NULL argument or a table without kernel_prepare is INVALID_ARGUMENT; a kernel_prepare that
 *     fails ends it with its status behind one error line that names the kernel. On failure nothing is written.
 */
#ifndef DRAFT_H
#define DRAFT_H

#include <stdint.h>

#include "anemone/attention_core.h"
#include "anemone/attention_state.h"
#include "anemone/backend_seam.h"
#include "anemone/draft_records.h"
#include "anemone/hyper_connection.h"
#include "anemone/linear.h"
#include "anemone/router.h"
#include "anemone/status.h"

struct draft_kernels_record {
	uint32_t verify_matches;
	uint32_t markov_bias;
	uint32_t confidence;
};

MUST_CHECK result_status draft_kernels_prepare(
	const struct backend_operations *backend, struct device_card *card,
	uint32_t arithmetic_class, struct hyper_kernels_record *hyper,
	struct router_kernels_record *router,
	struct attention_state_kernels *state,
	struct attention_core_kernels *core,
	struct draft_kernels_record *kernels_out);

/* The verify kernel of a DRAFTS step, the draft segment's first launch. */
struct verify_matches_description {
	struct linear_range
		step; /* read: the step area, its block and row_count rows */
	struct linear_range result; /* read: the result area of row_count rows */
	struct linear_range
		round; /* written: struct step_round, its first members */
	struct linear_range
		draft_step; /* written: the draft's step area, its block and draft_positions rows */
	struct linear_range
		draft_views; /* written: draft_positions views of STEP_DRAFT_VIEW_BYTES */
	uint32_t row_count; /* the program's, a thin step's: the record's, the most rows the kernel reads */
	uint32_t draft_positions; /* R: trained block 5, optional extended block up to 7 */
	uint32_t end_token_id;
	uint32_t vocabulary_size; /* the end token and MODEL_DRAFT_NOISE_TOKEN lie below it */
	uint32_t verifier; /* 0, or with DRAFT_ROUND_TABLE the verifier'th DRAFTS row (draft_records.h) */
	uint32_t flags; /* draft_records.h's DRAFT_*: with flags, row_count to DRAFT_BATCH_MOST_ROWS */
	uint32_t round_capacity; /* ROUND_TABLE: the rounds of the table at round (1 or more), else 0 */
	uint32_t draft_slabs; /* BATCHED: draft_step's slab blocks, every verifier's rows (draft_records.h), else 0 */
};
MUST_CHECK result_status verify_matches_prepare(
	const struct verify_matches_description *description,
	uint64_t arena_byte_count, const struct draft_kernels_record *kernels,
	struct kernel_launch *launch_out);

/* The Markov bias of position j of the block, before the draft's sampler j. */
struct draft_markov_bias_description {
	struct linear_range
		draft_step; /* read: the draft's step area (its block and row 0) */
	struct linear_range
		results; /* read: draft_positions result areas of STEP_DRAFT_RESULT_BYTES */
	struct linear_range
		embedding; /* read: markov_head.embed.weight, bfloat16 [embedding_rows or V][256], exactly */
	struct linear_range
		head; /* read: markov_head.head.weight's rows of the cut, bfloat16 [V][256], exactly */
	struct linear_range
		logits; /* read and written: the draft's logits, float32 [draft_positions][V] */
	struct linear_range
		markov; /* written: the Markov rows, bfloat16 [draft_positions][256] */
	uint32_t position; /* j, below draft_positions */
	uint32_t draft_positions;
	uint32_t vocabulary_size; /* V: the logits' row and W2's rows, a cut of the vocabulary where embedding_rows */
	uint32_t embedding_rows; /* 0, or W1's rows (the whole vocabulary, at least V; tp4_v3_claude's draft shards) */
	uint32_t flags; /* 0, or DRAFT_BATCHED (draft_records.h): draft_step view 0, results the position areas,
			   logits [R][8][V] and markov [R][8][256] position-major */
};
MUST_CHECK result_status draft_markov_bias_prepare(
	const struct draft_markov_bias_description *description,
	uint64_t arena_byte_count, const struct draft_kernels_record *kernels,
	struct kernel_launch *launch_out);

/* The confidence of the block's positions and the round's proposals: the draft segment's last launch. */
struct draft_confidence_description {
	struct linear_range draft_step; /* read: the draft's step area's block */
	struct linear_range
		collapse; /* read: the collapse rows, bfloat16 [draft_positions][5,120] */
	struct linear_range
		markov; /* read: the Markov rows, bfloat16 [draft_positions][256] */
	struct linear_range
		weight; /* read: confidence_head.proj.weight, bfloat16 [5,376], exactly */
	struct linear_range results; /* read: draft_positions result areas */
	struct linear_range
		round; /* written: struct step_round, its proposals and confidences (ROUND_TABLE: the table) */
	uint32_t draft_positions;
	uint32_t flags; /* 0 or DRAFT_ROUND_TABLE (draft_records.h), with it BATCHED: draft_step view 0, markov
			   [R][8][256] and results the position areas (position-major), collapse the verifier's R rows */
	uint32_t verifier; /* BATCHED: the verifier's slot, else 0 */
};
MUST_CHECK result_status draft_confidence_prepare(
	const struct draft_confidence_description *description,
	uint64_t arena_byte_count, const struct draft_kernels_record *kernels,
	struct kernel_launch *launch_out);

#endif
