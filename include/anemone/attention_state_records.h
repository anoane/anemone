/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * attention_state_records.h: the records of the kernels of the attention state at 16 bits (kernel_attention_state.cu).
 * C11 and CUDA, of the engine model_layout.h and step_records.h alone: for the kernel file, attention_state.c, tests.
 *
 * Contract: linear_records.h's rules of a kernel (offsets into the arena, no atomic operation, nothing timed, nothing
 * checked but by the checked modules, which trap on a launch of another block, and where state is written on a step
 * of no row or of more than the launch's, a position outside the context or a slot past the sessions);
 * model_layout.h's arrays; attention_state.h's ranges. The arithmetic, in float32 unless said:
 *   - norm (REF:285-292) of w values x: s is the sum of their squares, each rounded; thread t adds those of its pairs
 *     2 p, 2 p + 1, p = t + 128 k for ascending k, from +0; then the lanes of a warp by the butterfly xor 16 .. 1,
 *     then the warps from +0 in ascending order; f = rsqrtf(s * (1 / w) + 1e-20); a value is weight * (x * f), the
 *     weight a bfloat16. The order is the engine's, fixed, and has no reference-order form (the reference's library
 *     reduces in an order of its own): a row's bits depend on that row alone.
 *   - rotation (REF:392-406) of dims 2 i, 2 i + 1 of a vector's last 64, (a, b) by (c, s) of its use: (fmaf(a, c,
 *     -(b s)), fmaf(b, c, a s)): the reference's complex product as c10::complex writes it, a c - b s and a s + b c,
 *     with the two contractions nvcc makes of it on both architectures. The rotation step writes c = cosf(t),
 *     s = sinf(t) of t = float32(p') * word (REF:388, torch.polar on the card), p' the row's position, less 1 for
 *     ATTENTION_ROTATION_GROUP_START.
 *   - Roundings to bfloat16 and FP16: to nearest, a tie to even; a bfloat16 not-a-number is 0x7FC0.
 *   - prepare (REF:770, 705-707, 718), grid y 0: qr, the norm of wq_a's output by q_norm, to bfloat16; y 1, rows of
 *     the step: the norm of wkv's output by kv_norm, rotated at rotation_use, to FP16, ring slot p % 256 of the row's
 *     session.
 *   - window (dspark_v1/DESIGN.md C2): the prepare's window part alone, for the rows of the step, block x the row, by
 *     the same device code: a row's bits are the prepare's window part's of that row.
 *   - queries (REF:771-772, 550-552), block x below head_count: head first_head + x of wq_b's output, the last 64
 *     dims rotated and rounded to bfloat16 (REF:405 copies them back into q), to FP16, at the same head of the
 *     queries (both [row][64][512]); above it index head x - head_count of the indexer's wq_b output, rotated, to
 *     FP16, placed as model_layout.h's index queries.
 *   - compressor (REF:458-485, 751-761), rows of the step. Ratio 2: a row that does not emit ((p + 1) % 2 != 0) keeps
 *     its float32 kv and score in slot p % ATTENTION_COMPRESSOR_SLOTS of its session; an emitting row reads
 *     those of p - 1 from the step's row at p - 1 of its session, else from the slot: no row reads a slot its launch
 *     writes. Per dim, of scores s0, s1 and kv v0, v1 at p - 1 and p: m from float32's lowest, m = m < s ? s : m for
 *     s0 then s1; e = expf(s - m); w = e / ((0 + e0) + e1); (0 + v0 w0) + v1 w1, to bfloat16. Ratio 1: wkv's
 *     bfloat16 product. Its norm, the latent, goes to bfloat16 for wk and, rotated at GROUP_START (ratio 2) or
 *     POOLED (ratio 1), to FP16 as pool entry p / ratio. key (REF:537-548), emitting rows: k_norm of wk's output,
 *     rotated as the pool entry, to FP16, key p / ratio of the session's tiles.
 *   - Launches: block[0] threads, grid[0] by grid[1] blocks, no dynamic shared memory: rotation 96, rows by 1;
 *     prepare 128, rows by 2 (by 1: qr alone, of up to a chunk's rows, attention_state.h); the window, the prepare's
 *     record (its qr offsets 0), 128, rows (up to a chunk's) by 1; queries 128, head_count + index_head_count by
 *     rows (the queries alone: up to a chunk's rows); compressor, key 128, rows by 1. A state array (ring, slots, pool,
 *     keys) is written for the rows below the step's row_count only. A BATCH pool can defer compressor slot writes:
 *     SLOTS_DEFER computes pools and latents without retaining slots; a following SLOTS_KEEP launch retains only
 *     the last non-emitting row of each session and slot, without computing pools. This separates old-slot reads
 *     from writes when a prompt owner spans more rows than the compressor ring holds. INLINE keeps the original
 *     per-slab/prefill behavior. The record layout and every arithmetic operation are unchanged.
 *   - Records: fixed-width members, widest first, no padding, sizes asserted in both languages.
 *   - Risk 8: the checked modules trap on a finite value FP16 cannot hold, stale rows included; no build counts the
 *     values below FP16's normal range, which the replay counts per kind.
 */
#ifndef ATTENTION_STATE_RECORDS_H
#define ATTENTION_STATE_RECORDS_H

#include <stdint.h>

#include "anemone/model_layout.h"
#include "anemone/step_records.h"

#define KERNEL_FILE_ATTENTION_STATE "kernel_attention_state"
#define KERNEL_ATTENTION_ROTATION "attention_rotation"
#define KERNEL_ATTENTION_PREPARE "attention_prepare_entry"
#define KERNEL_ATTENTION_QUERIES "attention_queries_entry"
#define KERNEL_ATTENTION_COMPRESSOR "attention_compressor_entry"
#define KERNEL_ATTENTION_KEY "attention_key_entry"
#define KERNEL_ATTENTION_WINDOW "attention_window"
#define ATTENTION_ROTATION_THREADS 96u /* a use and a frequency each */
#define ATTENTION_STATE_THREADS 128u
#define ATTENTION_PREPARE_PARTS 2u
#define ATTENTION_COMPRESSOR_SLOTS \
	16u /* positions a session's 2-to-1 compressor keeps: ours */
#define ATTENTION_SLOT_VECTORS \
	2u /* a slot: float32 kv [512], then score [512] */
#define ATTENTION_SLOTS_SESSION_BYTES 65536u
#define ATTENTION_ROTATION_ARGUMENT_BYTES 24u
#define ATTENTION_PREPARE_ARGUMENT_BYTES 88u
#define ATTENTION_QUERIES_ARGUMENT_BYTES 64u
#define ATTENTION_COMPRESSOR_ARGUMENT_BYTES 96u
#define ATTENTION_KEY_ARGUMENT_BYTES 64u
#define ATTENTION_COMPRESSOR_SLOTS_INLINE 0u
#define ATTENTION_COMPRESSOR_SLOTS_DEFER 1u
#define ATTENTION_COMPRESSOR_SLOTS_KEEP 2u

enum attention_rotation_use { /* the three angles of a row: model_layout.h rotation */
			      ATTENTION_ROTATION_WINDOW_ONLY =
				      0, /* theta 0 at p: the layers without a pool (REF:680-687) */
			      ATTENTION_ROTATION_POOLED = 1, /* theta 1 at p */
			      ATTENTION_ROTATION_GROUP_START =
				      2 /* theta 1 at p - 1: a 2-to-1 group's first token (REF:542, 756) */
};

/* The members are the arrays of attention_state.h's descriptions, at their offsets. */
struct attention_rotation_arguments {
	uint64_t step_offset;
	uint64_t frequencies_offset;
	uint64_t rotation_offset;
};
struct attention_prepare_arguments {
	uint64_t step_offset;
	uint64_t rotation_offset;
	uint64_t query_rank_offset;
	uint64_t query_norm_offset;
	uint64_t query_normed_offset;
	uint64_t window_input_offset;
	uint64_t window_norm_offset;
	uint64_t ring_offset;
	uint64_t ring_session_stride;
	uint32_t rotation_use;
	uint32_t session_capacity;
	uint32_t context_token_limit;
	uint32_t reserved_zero;
};
struct attention_queries_arguments {
	uint64_t rotation_offset;
	uint64_t heads_offset;
	uint64_t queries_offset;
	uint64_t index_heads_offset;
	uint64_t index_queries_offset;
	uint32_t first_head;
	uint32_t head_count;
	uint32_t index_head_count; /* 0 or 32 */
	uint32_t rotation_use;
	uint32_t heads_row_heads; /* the heads a row of heads_offset holds: 64, or head_count */
	uint32_t heads_first_head; /* the launch's first head in such a row: first_head, or 0 */
};
struct attention_compressor_arguments {
	uint64_t step_offset;
	uint64_t rotation_offset;
	uint64_t kv_offset; /* ratio 2: float32; ratio 1: bfloat16 */
	uint64_t score_offset; /* ratio 2 */
	uint64_t slots_offset; /* ratio 2 */
	uint64_t slots_session_stride;
	uint64_t norm_offset;
	uint64_t latent_offset;
	uint64_t pool_offset;
	uint64_t pool_session_stride;
	uint32_t ratio; /* 1 or 2 */
	uint32_t session_capacity;
	uint32_t context_token_limit;
	uint32_t slot_mode; /* INLINE; DEFER computes pools only; KEEP writes slots only */
};
struct attention_key_arguments {
	uint64_t step_offset;
	uint64_t rotation_offset;
	uint64_t key_input_offset;
	uint64_t norm_offset;
	uint64_t keys_offset;
	uint64_t keys_session_stride;
	uint32_t ratio;
	uint32_t session_capacity;
	uint32_t context_token_limit;
	uint32_t reserved_zero;
};

MODEL_LAYOUT_HOLDS(sizeof(struct attention_rotation_arguments) ==
			   ATTENTION_ROTATION_ARGUMENT_BYTES &&
		   sizeof(struct attention_prepare_arguments) ==
			   ATTENTION_PREPARE_ARGUMENT_BYTES &&
		   sizeof(struct attention_queries_arguments) ==
			   ATTENTION_QUERIES_ARGUMENT_BYTES &&
		   sizeof(struct attention_compressor_arguments) ==
			   ATTENTION_COMPRESSOR_ARGUMENT_BYTES &&
		   sizeof(struct attention_key_arguments) ==
			   ATTENTION_KEY_ARGUMENT_BYTES);
/* a block holds a use and frequency each, or an index head's pairs; a slot's bytes; the rows of a launch and the
 * position before them never share a slot */
MODEL_LAYOUT_HOLDS(
	ATTENTION_ROTATION_THREADS == MODEL_ROTATION_USES * MODEL_FREQUENCIES &&
	MODEL_INDEX_HEAD_DIMS / MODEL_PAIR_VALUES <= ATTENTION_STATE_THREADS &&
	ATTENTION_SLOTS_SESSION_BYTES == MODEL_FLOATS(MODEL_HEAD_DIMS) *
						 ATTENTION_SLOT_VECTORS *
						 ATTENTION_COMPRESSOR_SLOTS &&
	ATTENTION_COMPRESSOR_SLOTS > LINEAR_MAXIMUM_ROWS);

#endif
