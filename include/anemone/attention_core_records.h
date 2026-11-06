/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * attention_core_records.h: the record of the attention core and what its kernels compute. C11 and CUDA,
 * including of the engine model_layout.h and step_records.h alone: for kernel_attention_core.cu, the host code that
 * prepares their launch (attention_core.h), the tests. REF is the reference's inference/model.py, KER its kernel.py.
 *
 * Contract:
 *   - attention_core_entry(record, arena) in kernel_attention_core.cu: the sparse attention of one layer
 *     (REF:780, KER:311-403) for each row r below the row_count of the step block at step_offset (step_records.h)
 *     and each head of first_head .. first_head + 8 grid[1] - 1. A row at position p of session s reads 640 keys
 *     in this order: the window, keys 0 .. 127 the positions p - 127 .. p, none below 0 (the decode order of
 *     REF:422-424), from slot q mod MODEL_RING_SLOTS of session s's ring; then the pool, keys 128 .. 639
 *     the entries the row's selection names, -1 for none, from session s's pool. Rows from row_count on are not
 *     touched: neither what their step rows nor their selections hold is read.
 *   - Arrays (model_layout.h), whatever heads a launch covers: queries FP16 [row][64][512]; ring FP16
 *     [session][ring slots][512]; pool FP16 [session][context_token_limit / pool_ratio][512]; selection int32
 *     [row][512]:
 *     entries ascending, -1 behind, each below (p + 1) / pool_ratio, all of them where at most 512 are
 *     (REF:561-580); sinks float32 [64] (REF:639); rotation float32 [row][3 uses][32][cos, sin], the record's offset
 *     that of row 0's use: theta 0 at p for the window alone, theta 1 at p with a pool (REF:680-687); output
 *     bfloat16 [row][64][512]; the reference order's scores float32 [row][64][640]
 *     (ATTENTION_CORE_SCORES_ROW_BYTES a row: the step's scratch, which nothing else reads or keeps).
 *     Session s's ring lies at ring_offset + s ring_session_stride, its pool at pool_offset + s pool_session_stride.
 *     pool_ratio 0 is the window alone (REF:775): pool and selection are not read.
 *   - Per row and head, in float32 with no fused multiply-add but the tensor cores' and the two of the turning back:
 *       score    q . k as 4 slices of 128 dims, each one chain from +0 of 8 tensor-core instructions of 16
 *                products; the slices added from +0 in ascending order; times ATTENTION_CORE_SCALE (REF:651);
 *                -infinity for a key of index -1.
 *       softmax  over groups of 16 keys in key order, a running maximum m from ATTENTION_CORE_SCORE_FLOOR
 *                (KER:355). g the group's largest score: where g > m + ATTENTION_CORE_RESCALE_STEP, the
 *                output and the sum are multiplied by expf(m - g) and m = g. p = expf(s - m), 0 for no key;
 *                the 16 p of a group are added (key j with key j + 8, then the eight sums 1, 2 and 4 apart) and
 *                sum = sum x factor + that. A group of no key adds nothing and is not run: the window's groups
 *                before position 0 and the pool's beyond the first min(512, (p + 1) / pool_ratio) entries.
 *       values   O += P V: p rounded to FP16 (KER:377 rounds it to bfloat16: the owner's choice), the tensor cores
 *                adding the 16 keys of a group to O.
 *       output   d = sum + expf(sink - m) (KER:383); each value O / d, rounded to bfloat16 (KER:385-387); dims 448
 *                .. 511 then turned back from those bfloat16 values as REF:781 does: x (cos - i sin), the
 *                reference's complex product with the conjugate angle in attention_state_records.h's contractions,
 *                (fmaf(x0, cos, x1 sin), fmaf(x1, cos, -(x0 sin))); rounded to bfloat16 again (REF:405).
 *     A value of the output depends on its row's inputs alone: not on the other rows of a launch, its grid or the
 *     heads a launch covers.
 *   - The block core (dspark_v1/DESIGN.md C3), attention_core_block_entry in the fixed order's launch: the draft's
 *     block window. A row's block is the run of the step's rows of its session at consecutive positions b .. e around
 *     it (a thin draft's rows are one block; a BATCHED draft's slab holds several verifiers' blocks), and pool_ratio
 *     is 0. Every row of a block reads the same keys k = 0 .. 128 + e - b, key k the position b - 128 + k (none below
 *     0), from slot q mod MODEL_RING_SLOTS of the session's ring: the 128 positions before the block and the
 *     whole block, which is not causal (REF:1023-1031).
 *     Every step of a row is the fixed order's above (groups of 16 keys in key order, the sink, the output turned
 *     back by the row's own rotation, use 0); the keys run in chronological order, not the reference's slot order.
 *   - The reference order (linear.h), with P as the oracle's stand-in rounds it, is two launches of the same record:
 *     attention_core_scores_entry forms every score of the row's groups as above into scores; then
 *     attention_core_values_entry takes each head's largest, m, from ATTENTION_CORE_SCORE_FLOOR up: the
 *     stand-in's maximum (its wide_sparse_attention, state_precision_true.py:324; KER:369's is a running one per 64
 *     keys), and runs the softmax above with that m, which no group's score then passes: p = expf(s - m) is added
 *     unrounded to the sum, and for the values rounded to bfloat16 (KER:339, KER:377), held in FP16 times
 *     ATTENTION_CORE_WEIGHT_SCALE: exact from p = 2^-31 up, below that rounded again on FP16's subnormal
 *     grid. Each value O times 1 / that scale (exact) is divided by d. Every other step is the first kernel's.
 *   - The launches, each block of ATTENTION_CORE_THREADS threads and its dynamic shared memory (above 48
 *     KiB: the backend opts every function in); heads_per_block the heads of a block, first_head the first of the
 *     launch:
 *       fixed order   grid (rows of the program, head_count / 8), ATTENTION_CORE_SHARED_BYTES_VALUE.
 *       scores        grid (rows, head_count / (8 t), 640 / ATTENTION_CORE_TILE_KEYS),
 *                     ATTENTION_CORE_SCORES_SHARED_BYTES(t), heads_per_block 8 t: t tiles of 8 heads, the
 *                     largest power of 2 not above the rows (a key a row read once at 8 rows) that divides head_count
 *                     / 8, at most 2 on a card of class 120 native (backend_seam.h).
 *       values        grid (rows, head_count / (8 t), 512 / ATTENTION_CORE_TILE_DIMS), blocks of
 *                     ATTENTION_CORE_VALUE_THREADS threads, ATTENTION_CORE_VALUES_SHARED_BYTES(t),
 *                     heads_per_block 8 t: t = 2 from 4 rows where it divides head_count / 8 (not on 120 native), else
 *                     1.
 *     On a card of class 120 (a PRO that emulates class 80: main.c --pro_emulation=1) the reference order computes
 *     class 80's sums without its tensor cores, its blocks of other shapes:
 *       scores        grid (rows, head_count / ATTENTION_CORE_EXACT_SCORE_HEADS, 640 /
 *                     ATTENTION_CORE_EXACT_TILE_KEYS), ATTENTION_CORE_EXACT_SCORES_SHARED_BYTES.
 *       values        grid (rows, head_count / h, 512 / ATTENTION_CORE_TILE_DIMS),
 *                     ATTENTION_CORE_EXACT_VALUES_SHARED_BYTES, heads_per_block h: 1, 2, 4 and 2 at 1, 2, up
 *                     to 4 and more rows (measured), the largest that divides head_count.
 *   - The kernel checks nothing. Whoever prepares a launch has checked its record and ranges (attention_core.h);
 *     the step driver the step (1 to grid[0] rows, a position below context_token_limit, a slot below
 *     session_capacity); the selection's writer its entries. The checked modules trap where the step or an entry
 *     breaks that, the release modules read no key for an entry outside the session's pool. No atomic operation; no
 *     result depends on timing.
 *   - The record has no padding and the size asserted below, C11 and CUDA.
 */
#ifndef ATTENTION_CORE_RECORDS_H
#define ATTENTION_CORE_RECORDS_H

#include <stdint.h>

#include "anemone/model_layout.h"
#include "anemone/step_records.h"

#define KERNEL_FILE_ATTENTION_CORE "kernel_attention_core"
#define KERNEL_FILE_ATTENTION_CORE_NATIVE \
	"kernel_attention_core_native" /* class 120 native */
#define KERNEL_ATTENTION_CORE "attention_core_entry"
#define KERNEL_ATTENTION_CORE_SCORES "attention_core_scores_entry"
#define KERNEL_ATTENTION_CORE_VALUES "attention_core_values_entry"
#define KERNEL_ATTENTION_CORE_BLOCK "attention_core_block_entry"
#define ATTENTION_CORE_MOST_LAUNCHES 2u /* the reference order's */
#define ATTENTION_CORE_ARGUMENT_BYTES 120u
#define ATTENTION_CORE_KEYS (MODEL_WINDOW + MODEL_SELECTED)
#define ATTENTION_CORE_SCORES_ROW_BYTES \
	163840u /* float32 [64][640]: kernel_attention_core.cu asserts it */
#define ATTENTION_CORE_HEADS_PER_BLOCK \
	8u /* the columns of the instruction: batch i11b's heads:8 */
#define ATTENTION_CORE_THREADS 128u /* 4 warps: a warp a slice of 128 dims */
/* kernel_attention_core.cu lays out each kernel's shared memory and asserts these sizes */
#define ATTENTION_CORE_SHARED_BYTES_VALUE 36256u
#define ATTENTION_CORE_TILE_KEYS \
	64u /* of a score block: a group of 16 keys a warp */
#define ATTENTION_CORE_SCORE_TILES_MOST \
	8u /* tiles of 8 heads a score block: 1, 2, 4 or 8 */
#define ATTENTION_CORE_SCORES_SHARED_BYTES(tiles) \
	(ATTENTION_CORE_SCORES_FIXED_BYTES +      \
	 (tiles) * ATTENTION_CORE_SCORES_TILE_BYTES)
#define ATTENTION_CORE_SCORES_FIXED_BYTES \
	66048u /* the warps' stages, the tile's rows */
#define ATTENTION_CORE_SCORES_TILE_BYTES 8320u /* a tile's queries */
#define ATTENTION_CORE_VALUE_THREADS 256u /* of a value block: 8 warps */
#define ATTENTION_CORE_TILE_DIMS \
	128u /* of a value block: an instruction's 16 rows a warp */
#define ATTENTION_CORE_VALUE_TILES_MOST \
	2u /* tiles of 8 heads a value block: 1 or 2 */
#define ATTENTION_CORE_VALUES_SHARED_BYTES(tiles) \
	(ATTENTION_CORE_VALUES_FIXED_BYTES +      \
	 (tiles) * ATTENTION_CORE_VALUES_TILE_BYTES)
#define ATTENTION_CORE_VALUES_FIXED_BYTES \
	37888u /* the row's keys' rows, the warps' rings */
#define ATTENTION_CORE_VALUES_TILE_BYTES \
	32064u /* a tile's scores, weights and sums */
#define ATTENTION_CORE_EXACT_TILE_KEYS \
	32u /* class 120's score block in the exact profile: 4 lanes a key */
#define ATTENTION_CORE_EXACT_SCORE_HEADS 4u
#define ATTENTION_CORE_EXACT_SCORES_SHARED_BYTES 25472u
#define ATTENTION_CORE_EXACT_VALUE_HEADS 4u /* the most of a value block */
#define ATTENTION_CORE_EXACT_VALUES_SHARED_BYTES 36528u
#define ATTENTION_CORE_SCALE \
	0.044194173824159216f /* 512^-0.5 in float32 (REF:651, KER:320) */
#define ATTENTION_CORE_RESCALE_STEP \
	8.0f /* a rise of the maximum that rescales: p stays below e^8 */
#define ATTENTION_CORE_SCORE_FLOOR (-1.0e30f) /* KER:355: no exp(-inf - -inf) */
/* 2^14: a bfloat16 p of at most 1 times it stays below FP16's largest, its 8 significant bits whole from p = 2^-31 */
#define ATTENTION_CORE_WEIGHT_SCALE 16384.0f
#define ATTENTION_CORE_WEIGHT_UNSCALE 6.103515625e-05f /* 2^-14 */

struct attention_core_arguments_record {
	uint64_t step_offset; /* the step area: its block, then its rows */
	uint64_t queries_offset;
	uint64_t ring_offset;
	uint64_t ring_session_stride; /* MODEL_RING_SLOTS rows */
	uint64_t pool_offset; /* 0 where pool_ratio is 0 */
	uint64_t pool_session_stride; /* context_token_limit / pool_ratio rows; 0 where pool_ratio is 0 */
	uint64_t selection_offset; /* 0 where pool_ratio is 0 */
	uint64_t sinks_offset;
	uint64_t rotation_offset; /* row 0's use */
	uint64_t output_offset;
	uint64_t scores_offset; /* the reference order's; 0 in the fixed order */
	uint32_t context_token_limit; /* of a session: with pool_ratio, the entries of its pool */
	uint32_t session_capacity;
	uint32_t pool_ratio; /* 0, MODEL_DIRECT_RATIO or MODEL_POOLED_RATIO */
	uint32_t first_head;
	uint32_t heads_per_block; /* of the launch (above) */
	uint32_t output_row_heads; /* 0: output rows of every head, head h at h; else rows of this
                                                   many heads, head h at h - first_head (a TP rank's own) */
	uint32_t cone_layer; /* 0, or a decoder layer: its rows that step_records.h's
                                                   STEP_ROW_CONE_SKIPS leaves alone run no block */
	uint32_t reserved_zero;
};

MODEL_LAYOUT_HOLDS(sizeof(struct attention_core_arguments_record) ==
		   ATTENTION_CORE_ARGUMENT_BYTES);

#endif
