/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * attention_state.h: the launches of the attention state (attention_state_records.h), each prepared once at load
 * from a description that names every arena offset and count, as linear.h prepares the linear's.
 *
 * Contract:
 *   - linear.h's rules for prepare functions hold: arithmetic on the description alone, nothing per token, any
 *     thread; a refusal is STATUS_INVALID_ARGUMENT with one error line and the launches untouched. A range
 *     is struct linear_range, checked by prepare_common.h: aligned, inside the arena, holding what the
 *     launches touch, a norm's weight exactly its bytes, no two of a description sharing a byte.
 *   - Refused: a NULL argument, a kernel not prepared, a reserved field not zero; a row_count of 0 or above
 *     LINEAR_MAXIMUM_ROWS (qr alone and the queries alone: MODEL_WIDE_MOST_ROWS); a session_capacity of
 *     0; a context_token_limit of 0 or no multiple of
 *     MODEL_CONTEXT_GRANULE; a head_count of 0, or a head first_head + head_count - 1 past the model's; a
 *     theta or has_index other than 0 or 1; a pool_ratio other than 0, 1 or 2, or a pool on a layer without an
 *     indexer (the index keys' owner) or of theta 0; an indexer of theta 0.
 *   - A state array of sessions keeps session s at its offset plus s times one session's bytes, nothing between
 *     (model_layout.h): ring MODEL_RING_SLOTS * MODEL_STATE_ROW_BYTES, slots
 *     ATTENTION_SLOTS_SESSION_BYTES, pool (context_token_limit / pool_ratio) * MODEL_STATE_ROW_BYTES,
 *     keys (context_token_limit / pool_ratio) * MODEL_KEY_BYTES. Its range holds session_capacity of them.
 *     The step area holds the block and row_count rows. Not checked here, because not known at load: what the
 *     arena holds; the step rows are the step driver's (step_records.h), and the checked modules trap on a bad one.
 *   - Launches prepared for different row counts may name the same ranges: each uses the first rows.
 */
#ifndef ATTENTION_STATE_H
#define ATTENTION_STATE_H

#include <stdint.h>

#include "anemone/attention_state_records.h"
#include "anemone/backend_seam.h"
#include "anemone/linear.h"
#include "anemone/status.h"

/* The handles of the kernels of attention_state_records.h on one card, in the order of the members. */
struct attention_state_kernels {
	uint32_t rotation;
	uint32_t prepare;
	uint32_t queries;
	uint32_t compressor;
	uint32_t key;
	uint32_t window; /* the draft's: draft_kernels_prepare (draft.h) */
};

/* First user: the model's load, once for each open card; linear.h's linear_kernels_prepare's contract. It
 * prepares the first five and leaves window BACKEND_NO_KERNEL (a card without the draft keeps its handles:
 * draft.h). */
MUST_CHECK result_status attention_state_kernels_prepare(
	const struct backend_operations *backend, struct device_card *card,
	struct attention_state_kernels *kernels_out);

/* The rotation step of a card: once a step, before its first layer. */
struct attention_rotation_description {
	struct linear_range step; /* read: the step area */
	struct linear_range frequencies; /* read: float32 [2][32] */
	struct linear_range rotation; /* written: float32 [row_count][3][32][2] */
	uint32_t row_count;
	uint32_t reserved_zero;
};
MUST_CHECK result_status attention_rotation_prepare(
	const struct attention_rotation_description *description,
	uint64_t arena_byte_count,
	const struct attention_state_kernels *kernels,
	struct kernel_launch *launch_out);

/* The attention state of one layer. The model runs PREPARE after wq_a with wkv, QUERIES after wq_b, and on a layer
 * that pools COMPRESSOR after its products and KEY after wk; a layer without a pool gets two launches, and the
 * other two all zero (a launch of no block, which launch_run refuses). */
enum attention_state_launch {
	ATTENTION_STATE_PREPARE = 0,
	ATTENTION_STATE_QUERIES = 1,
	ATTENTION_STATE_COMPRESSOR = 2,
	ATTENTION_STATE_KEY = 3,
	ATTENTION_STATE_LAUNCH_COUNT = 4
};
struct attention_state_description {
	struct linear_range step; /* read: the step area */
	struct linear_range rotation; /* read: float32 [row_count][3][32][2] */
	struct linear_range
		query_rank; /* read: wq_a's output, bfloat16 [row][1280] */
	struct linear_range
		query_norm; /* read: attn.q_norm.weight, bfloat16 [1280] */
	struct linear_range query_normed; /* written: qr, bfloat16 [row][1280] */
	struct linear_range
		window_input; /* read: wkv's output, bfloat16 [row][512] */
	struct linear_range
		window_norm; /* read: attn.kv_norm.weight, bfloat16 [512] */
	struct linear_range ring; /* written: FP16 [session][ring slots][512] */
	struct linear_range
		heads; /* read: wq_b's output, bfloat16 [row][64][512] */
	struct linear_range
		queries; /* written: FP16 [row][64][512], the launch's heads */
	struct linear_range
		index_heads; /* has_index, read: indexer.wq_b's output, bfloat16 [row][32][128] */
	struct linear_range
		index_queries; /* has_index, written: FP16 [row][8,192 B] */
	struct linear_range
		pool_input; /* pools, read: wkv's product, ratio 2 float32, 1 bfloat16 [row][512] */
	struct linear_range
		pool_score; /* ratio 2, read: wgate's product, float32 [row][512] */
	struct linear_range
		slots; /* ratio 2, read and written: [session][16][2][512] float32 */
	struct linear_range
		pool_norm; /* pools, read: compressor.norm.weight, bfloat16 [512] */
	struct linear_range latent; /* pools, written: bfloat16 [row][512] */
	struct linear_range
		pool; /* pools, written: FP16 [session][entries][512] */
	struct linear_range
		key_input; /* pools, read: indexer.wk's output, bfloat16 [row][128] */
	struct linear_range
		key_norm; /* pools, read: indexer.k_norm.weight, bfloat16 [128] */
	struct linear_range
		keys; /* pools, written: [session][entries / 8][2,048 B] */
	uint32_t row_count;
	uint32_t session_capacity;
	uint32_t context_token_limit;
	uint32_t theta; /* 0: layers 0 and 1 (window only); 1 */
	uint32_t first_head; /* the launch's heads: first_head on, head_count of them; on */
	uint32_t head_count; /* one card 0 and 64 (a tensor-parallel rank runs some) */
	uint32_t has_index;
	uint32_t pool_ratio; /* 0: the layer writes no pool; else its compress ratio */
	uint32_t reserved_zero;
};
/* launches_out: ATTENTION_STATE_LAUNCH_COUNT launches. */
MUST_CHECK result_status
attention_state_prepare(const struct attention_state_description *description,
			uint64_t arena_byte_count,
			const struct attention_state_kernels *kernels,
			struct kernel_launch *launches_out);
/* A chunk's pool entries and index keys alone, written ahead of its slabs' scans (a row reads no entry of a later
 * row): COMPRESSOR and KEY of a layer with a pool over 1 to MODEL_WIDE_MOST_ROWS consecutive rows of one
 * session, PREPARE and QUERIES all zero; the ranges of the pool's part checked as attention_state_prepare
 * checks them. A row's partner of its 2-to-1 entry is the row before it in the launch (else the slot). */
MUST_CHECK result_status
attention_pool_prepare(const struct attention_state_description *description,
		       uint64_t arena_byte_count,
		       const struct attention_state_kernels *kernels,
		       struct kernel_launch *launches_out);

/* qr alone: the prepare's part that writes it (attention_state_records.h), for 1 to MODEL_WIDE_MOST_ROWS rows.
 * It reads no state and writes none: a wide step forms the qr of its chunk with it ahead of its slabs' prepares, so
 * that wq_b runs over the chunk; each slab's prepare forms its rows' qr again, the same bits (a row's depend on that
 * row alone). Refused as the state's launches are, the rows to MODEL_WIDE_MOST_ROWS. */
struct attention_query_norm_description {
	struct linear_range
		query_rank; /* read: wq_a's output, bfloat16 [row][1280] */
	struct linear_range
		query_norm; /* read: attn.q_norm.weight, bfloat16 [1280] */
	struct linear_range query_normed; /* written: qr, bfloat16 [row][1280] */
	uint32_t row_count;
	uint32_t reserved_zero;
};
MUST_CHECK result_status attention_query_norm_prepare(
	const struct attention_query_norm_description *description,
	uint64_t arena_byte_count,
	const struct attention_state_kernels *kernels,
	struct kernel_launch *launch_out);

/* The queries alone: the state's QUERIES launch for 1 to MODEL_WIDE_MOST_ROWS rows, of the same record. It
 * reads no state and writes none (attention_state_records.h): a wide step forms its chunk's queries with it, a row's
 * bits its own, so the bytes of the QUERIES launches of the chunk's slabs; for a slab's rows it is that launch, byte
 * for byte. Refused as the state's launches are: the heads, theta and the indexer by the layer's rules, the rows to
 * MODEL_WIDE_MOST_ROWS; the index ranges are checked only with has_index. */
struct attention_queries_description {
	struct linear_range rotation; /* read: float32 [row_count][3][32][2] */
	struct linear_range
		heads; /* read: wq_b's output, bfloat16 [row][64][512] (heads_compact:
                                                   [row][head_count][512], the launch's heads alone) */
	struct linear_range
		queries; /* written: FP16 [row][64][512], the launch's heads */
	struct linear_range
		index_heads; /* has_index, read: indexer.wq_b's output, bfloat16 [row][32][128] */
	struct linear_range
		index_queries; /* has_index, written: FP16 [row][8,192 B] */
	uint32_t row_count;
	uint32_t theta; /* 0: layers 0 and 1 (window only); 1 */
	uint32_t first_head;
	uint32_t head_count;
	uint32_t has_index;
	uint32_t heads_compact; /* 0 or 1: a TP rank's wq_b output read where it wrote it */
};
MUST_CHECK result_status attention_queries_prepare(
	const struct attention_queries_description *description,
	uint64_t arena_byte_count,
	const struct attention_state_kernels *kernels,
	struct kernel_launch *launch_out);

/* The window part alone (attention_state_records.h: the window), for 1 to MODEL_WIDE_MOST_ROWS rows: the
 * draft's context writes its rings with it over a step's or a chunk's rows (dspark_v1/DESIGN.md 4.1, C2). It reads no
 * qr and writes no other state. Refused as the state's launches are, the rows to MODEL_WIDE_MOST_ROWS, and a
 * rotation_use outside its enumeration. */
struct attention_window_description {
	struct linear_range
		step; /* read: the step area, its block and row_count rows */
	struct linear_range rotation; /* read: float32 [row_count][3][32][2] */
	struct linear_range
		window_input; /* read: wkv's output, bfloat16 [row][512] */
	struct linear_range
		window_norm; /* read: attn.kv_norm.weight, bfloat16 [512] */
	struct linear_range ring; /* written: FP16 [session][ring slots][512] */
	uint32_t row_count;
	uint32_t session_capacity;
	uint32_t context_token_limit;
	uint32_t rotation_use; /* enum attention_rotation_use: the draft's is 0 */
	uint32_t reserved_zero;
};
MUST_CHECK result_status
attention_window_prepare(const struct attention_window_description *description,
			 uint64_t arena_byte_count,
			 const struct attention_state_kernels *kernels,
			 struct kernel_launch *launch_out);

#endif
