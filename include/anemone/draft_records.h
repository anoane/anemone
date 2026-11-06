/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * draft_records.h: the records of the draft's own kernels (kernel_draft.cu) and what they compute: the verify kernel,
 * which decides on the card that samples how many tokens a round emits and writes the draft's step area, and the
 * Markov bias and the confidence of the draft's head. C11 and CUDA, including of the engine model_layout.h and
 * step_records.h alone. dspark_v1/DESIGN.md C5, C6 and C8; REF is the reference's inference/model.py.
 *
 * Contract: linear_records.h's for a kernel (offsets into the arena, no atomic operation, nothing timed, nothing
 * checked but by the checked modules, which trap where the host broke a promise they can see). Arithmetic is float32,
 * each operation rounded once, fmaf where written; a bfloat16 is widened exactly.
 *   - verify_matches_entry (C5), integers only: of the step at step_offset (step_records.h), its live rows the
 *     smaller of its row_count and the record's (the checked modules trap on more than the record's), and the result
 *     area its sampler wrote at result_offset. The live rows hold exactly one row A with STEP_ROW_DRAFTS (the
 *     checked modules trap on none or more), of session s at position P; its session's rows with
 *     STEP_ROW_PROPOSED are at P + 1 .. P + k. y_i is the token the sampler drew from the session's row at
 *     P + i - 1 (i = 1 .. k + 1), d_i the token of its row at P + i (i <= k). m_i = d_i == y_i and y_i != end_token_id;
 *     a is the number of leading true m_i, e = a + 1 the tokens the round emits, y_e the anchor token and C' = P + e
 *     the anchor position. It writes the round's members sequence .. reserved_zero (the step's sequence, s, e, C', y_e,
 *     draft_positions, 0); the draft's step area: block {sequence, draft_positions, 0} and rows
 *     j = 0 .. draft_positions - 1 {C' + j, j == 0 ? y_e : noise_token_id, s, STEP_ROW_LOGITS}; view j: block
 *     {sequence, 1, 0} and a copy of row j. The draft's positions lie inside the context by the driver's rule (e)
 *     (DESIGN 5.1); the engine recomputes e and is the authority (DESIGN 7.1). Without a DRAFTS row the release modules
 *     write session STEP_NO_SESSION, e 0 and the step area and views with no rows.
 *   - draft_markov_bias (C6), launch j for position j of the block: the token t is the draft step's row 0's
 *     for j = 0, else the one the draft's sampler drew in result area j - 1 (the checked modules trap on t at or
 *     above vocabulary_size). E_k = W1[t][k], k = 0 .. 255. For each vocabulary entry v, lane l of a warp chains
 *     partial = fmaf(W2[v][8 l + i], E_(8 l + i), partial) from +0 for i = 0 .. 7, the lanes add by exchanges at
 *     distance 16 .. 1, and logits[v] = logits[v] + the sum (REF:1153-1154 adds in place). Block 0 copies W1[t] to
 *     the Markov row j (REF:1157). The reference's product is its library's float32 one (REF:1084-1088): the draft's
 *     arithmetic is its own (DESIGN 3: a draft changes proposals and acceptance, never tokens).
 *   - draft_confidence_entry (C8): R = min(the draft step's row_count, draft_positions). Block r < R: z the
 *     collapse row r (5,120) then the Markov row r (256), w the 5,376 weights of confidence_head.proj; thread t
 *     chains c = fmaf(w_i, z_i, c) from +0 over i = t, t + 256, .. ascending, the lanes of a warp add by exchanges at
 *     distance 16 .. 1, then the 8 warps' sums from +0 in ascending order: the raw score c_r (no sigmoid, REF:1100).
 *     It writes the round's proposals[r] (the token of result area r) and confidence_logits[r] = c_r; blocks r from R
 *     to STEP_DRAFT_MOST - 1 write zeros there.
 *   - Launches, block[1], block[2] and grid[2] 1, no dynamic shared memory: verify (1, 1) of
 *     VERIFY_MATCHES_THREADS; Markov bias (ceil(vocabulary_size / 8), 1) of DRAFT_MARKOV_THREADS, a
 *     warp an entry (BATCHED: ceil(vocabulary_size / 128), a warp 16 entries of every slot by m16n8k16 instructions
 *     over the 256 values, float32 sums in their order: a draft's arithmetic is its own); confidence (1, STEP_DRAFT_MOST) of DRAFT_CONFIDENCE_THREADS.
 *   - Records: fixed-width members, widest first, no padding, sizes asserted in both languages.
 */
#ifndef DRAFT_RECORDS_H
#define DRAFT_RECORDS_H

#include <stdint.h>

#include "anemone/model_layout.h"
#include "anemone/step_records.h"

#define KERNEL_FILE_DRAFT "kernel_draft"
#define KERNEL_VERIFY_MATCHES "verify_matches_entry"
#define KERNEL_DRAFT_MARKOV_BIAS "draft_markov_bias"
#define KERNEL_DRAFT_CONFIDENCE "draft_confidence_entry"
#define VERIFY_MATCHES_ARGUMENT_BYTES 72u
#define DRAFT_MARKOV_BIAS_ARGUMENT_BYTES 64u
#define DRAFT_CONFIDENCE_ARGUMENT_BYTES 64u
#define VERIFY_MATCHES_THREADS 32u /* one warp: a lane a proposal */
#define DRAFT_MARKOV_THREADS 256u /* 8 warps: an entry each */
#define DRAFT_MARKOV_ENTRIES_PER_BLOCK 8u
#define DRAFT_MARKOV_BATCHED_ENTRIES_PER_BLOCK \
	128u /* BATCHED: 8 warps of 16 entries, on the tensor cores */
#define DRAFT_MARKOV_ROW_BYTES \
	512u /* bfloat16 [256]: a row of W1 or W2, of the Markov rows */
#define DRAFT_CONFIDENCE_THREADS 256u
#define DRAFT_COLLAPSE_ROW_BYTES MODEL_HIDDEN_ROW_BYTES
/* The flags of a BATCH program's draft walk of one verifier (model_seam.h; tp4_v3_claude), 0 for a thin step's:
 *   ROUND_TABLE: the round of the verifier's session slot s at round_offset + s STEP_ROUND_BYTES (a table of
 *     round_capacity rounds); a verifier without a DRAFTS row writes no round, and its draft's step area and views
 *     hold no row, so no launch of its draft writes state and the confidence writes nothing.
 *   SLAB_RESULTS: the step's result area one per slab of 8 rows (PROGRAM_BATCH_SLAB_RESULT_BYTES apart).
 *   BATCHED (with ROUND_TABLE): every verifier's draft in one draft. Its layers: one draft step of draft_slabs slab
 *     blocks (a block and 8 rows each, DRAFT_SLAB_STEP_BYTES apart), verifier v's rows draft rows v R ..
 *     v R + R - 1 (R the draft positions), row i in slab block i / 8 at place i % 8; verifier 0 writes every slab
 *     block's header (the step's sequence and its rows of the DRAFTS rows' count D x R). Its positions, position-major
 *     (DRAFT_BATCH_SLOTS slots a position, verifier v slot v): view j (DRAFT_SLAB_STEP_BYTES apart) a
 *     block of the step's sequence and D rows (verifier 0 writes the headers), row v verifier v's row j (verifier v
 *     below D writes it); position j's logits [8][V] and Markov rows [8][256], its result area (a block and 8 rows,
 *     DRAFT_SLAB_RESULT_BYTES apart: the sampler's over view j). The Markov bias of position j, BATCHED: one
 *     launch over every slot v below view 0's row count, its token view 0's row v (j = 0) or row v of position
 *     j - 1's result area. The confidence of verifier v, BATCHED: nothing unless v is below view 0's row count; its
 *     session view 0's row v's, its collapse rows its R as given, Markov row j of slot v and token row v of position
 *     j's result area. */
#define DRAFT_ROUND_TABLE 1u
#define DRAFT_SLAB_RESULTS 2u
#define DRAFT_BATCHED 4u
/* WIDE_STEP (with BATCHED; track L's --batch_draft_wide=3): also one step block of every verifier's rows behind the
 * slab blocks (at draft_step_offset + draft_slabs DRAFT_SLAB_STEP_BYTES): verifier 0 writes its header (the
 * step's sequence, D x R rows), verifier v its rows v R .. v R + R - 1 there too (a launch over all the draft's rows
 * reads them: each verifier's block whole). */
#define DRAFT_WIDE_STEP 8u
#define DRAFT_SLAB_STEP_BYTES \
	(STEP_BLOCK_BYTES_VALUE + 8u * STEP_ROW_BYTES_VALUE)
#define DRAFT_SLAB_ROWS 8u
#define DRAFT_SLAB_RESULT_BYTES \
	(STEP_RESULT_BYTES + DRAFT_SLAB_ROWS * STEP_ROW_RESULT_BYTES)
#define DRAFT_BATCH_MOST_ROWS 64u /* a BATCH step's rows (model_seam.h) */
#define DRAFT_BATCH_SLOTS \
	DRAFT_SLAB_ROWS /* BATCHED: a position's verifier slots, the most verifiers */

/* Of the step it answers; written: the round's first members, the draft's step area and its views. */
struct verify_matches_arguments {
	uint64_t step_offset; /* read: the step area, its block and rows */
	uint64_t result_offset; /* read: the result area of the step's sampler */
	uint64_t round_offset; /* written: struct step_round, sequence .. reserved_zero */
	uint64_t draft_step_offset; /* written: the draft's step area, its block and draft_positions rows */
	uint64_t draft_views_offset; /* written: draft_positions views, STEP_DRAFT_VIEW_BYTES apart */
	uint32_t row_count; /* the most rows of a step: 1 to LINEAR_MAXIMUM_ROWS (with flags, to 64) */
	uint32_t draft_positions; /* R: 1 to STEP_DRAFT_MOST */
	uint32_t end_token_id;
	uint32_t noise_token_id;
	uint32_t verifier; /* its DRAFTS row: the step's first (0), or with ROUND_TABLE the verifier'th, in row order */
	uint32_t flags; /* DRAFT_ROUND_TABLE | DRAFT_SLAB_RESULTS, or 0 */
	uint32_t round_capacity; /* ROUND_TABLE: the table's rounds, above every session slot */
	uint32_t draft_slabs; /* BATCHED: the draft step's slab blocks (0 otherwise) */
};

/* Launch j: its token from the draft's step area (j = 0) or result area j - 1; row j of the logits and Markov rows
 * (BATCHED: every slot of position j, draft_records.h above). */
struct draft_markov_bias_arguments {
	uint64_t draft_step_offset; /* read: the draft's step area */
	uint64_t results_offset; /* read: the draft's result areas, STEP_DRAFT_RESULT_BYTES apart */
	uint64_t embedding_offset; /* read: W1, markov_head.embed.weight, bfloat16 [V][256] */
	uint64_t head_offset; /* read: W2, markov_head.head.weight, bfloat16 [V][256] */
	uint64_t logits_offset; /* read and written: float32 [V], the draft's logits row j */
	uint64_t markov_offset; /* written: bfloat16 [256], the Markov row j */
	uint32_t position; /* j */
	uint32_t vocabulary_size; /* V: of the logits row and W2's rows */
	uint32_t embedding_rows; /* W1's rows, the bound of the token: 0 for V (a vocabulary cut: the whole vocabulary) */
	uint32_t flags; /* 0, or DRAFT_BATCHED: the slots of position j (logits and Markov rows 8 a position) */
};

struct draft_confidence_arguments {
	uint64_t draft_step_offset; /* read: the draft's step area, its row count */
	uint64_t collapse_offset; /* read: the collapse rows, bfloat16 [R][5,120] */
	uint64_t markov_offset; /* read: the Markov rows, bfloat16 [R][256] */
	uint64_t weight_offset; /* read: confidence_head.proj.weight, bfloat16 [5,376] */
	uint64_t results_offset; /* read: the draft's result areas */
	uint64_t round_offset; /* written: the round's proposals and confidence_logits (ROUND_TABLE: the table's) */
	uint32_t draft_positions; /* the most rows read: R is at most it */
	uint32_t flags; /* 0, or DRAFT_ROUND_TABLE: the round of the session of the draft's row 0, | BATCHED */
	uint32_t verifier; /* BATCHED: its slot (below DRAFT_BATCH_SLOTS), else 0 */
	uint32_t reserved_zero;
};

MODEL_LAYOUT_HOLDS(sizeof(struct verify_matches_arguments) ==
			   VERIFY_MATCHES_ARGUMENT_BYTES &&
		   sizeof(struct draft_markov_bias_arguments) ==
			   DRAFT_MARKOV_BIAS_ARGUMENT_BYTES &&
		   sizeof(struct draft_confidence_arguments) ==
			   DRAFT_CONFIDENCE_ARGUMENT_BYTES);
/* a lane for each place of a thin step; the views and result areas of one row; the Markov rank a warp's 8 x 32 */
MODEL_LAYOUT_HOLDS(STEP_DRAFT_MOST + 1u == LINEAR_MAXIMUM_ROWS &&
		   LINEAR_MAXIMUM_ROWS <= VERIFY_MATCHES_THREADS &&
		   STEP_DRAFT_VIEW_BYTES ==
			   STEP_BLOCK_BYTES_VALUE + STEP_ROW_BYTES_VALUE &&
		   STEP_DRAFT_RESULT_BYTES >=
			   STEP_RESULT_BYTES + STEP_ROW_RESULT_BYTES &&
		   DRAFT_MARKOV_ROW_BYTES == MODEL_HALVES(MODEL_MARKOV_RANK) &&
		   DRAFT_MARKOV_THREADS == DRAFT_MARKOV_ENTRIES_PER_BLOCK *
						   LINEAR_THREADS_PER_WARP);
MODEL_LAYOUT_HOLDS(DRAFT_CONFIDENCE_THREADS % LINEAR_THREADS_PER_WARP == 0 &&
		   MODEL_DRAFT_CONFIDENCE_INPUTS % DRAFT_CONFIDENCE_THREADS ==
			   0 &&
		   MODEL_HIDDEN % DRAFT_CONFIDENCE_THREADS == 0);

#endif
