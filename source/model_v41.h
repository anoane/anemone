/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * model_v41.h: DeepSeek-V4.1-Flash behind the model seam: the tensors its programs read, the plan of every card's
 * arena and the frequency words. Private to source/: model_v41.c makes a plan, model_v41_program.c the programs and
 * the model's table of model_seam.h, which these functions fill as model_seam.h says.
 *
 * Contract:
 *   - Numbers cite the reference as model_layout.h does; CKPT is the checkpoint's config.json. Each function answers
 *     STATUS_UNSUPPORTED for a pack that names another model.
 *   - Tensors: the table of model_v41.c, by layer, routed expert and outside the layers. Each must be in the pack
 *     with that element type, shape, weight format and bits, else STATUS_CORRUPT_PACK and one error line
 *     that names it. N-gram tables are read separately, and the draft head only when the limits ask for the draft.
 *     A vision tower is optional: if any tower tensor is present the entire official ViT/projector, all learned
 *     delimiter embeddings and every layer's image router bias must have the pinned types/shapes (model_vision.h).
 *     Tower/projector weights and the embedding lie on card 0, image routing biases on their layer's card.
 *     Its scratch/input/output/session ranges append behind ordinary state; ordinary programs retain all original
 *     records and state metadata. Image programs are prepared separately and selected only for image-bearing rows.
 *   - A layer runs on the card of its tensors, the embedding on that of embed.weight, the head and the sampler on
 *     that of norm.weight and head.weight (the result card). A routed expert lies whole on its layer's card or whole
 *     in host memory, there its three matrices together as in every other (from the first byte of one to the last
 *     of another, with less than WEIGHT_FORMAT_MATRIX_ALIGNMENT_BYTES between them). Anything else is
 *     STATUS_UNSUPPORTED with one error line.
 *   - The draft (dspark_v1/DESIGN.md 5.2), with limits.draft_positions from MODEL_DRAFT_BLOCK to
 *     STEP_DRAFT_MOST; 0 plans none, and the
 *     plan is that of a pack without a draft head, byte for byte. Its stages 0 to 2 are walk layers 40 to 42
 *     (V41_DATA_WALK_LAYERS): a layer's tensors under mtp.<stage>. but those of the compressor, the indexer and the
 *     n-gram rows, its router for MODEL_DRAFT_EXPERTS experts, which each stage has; then the draft's globals.
 *     Every draft tensor, its routed experts included, and layers 37 to 39, whose inputs the draft taps, lie on the
 *     result card, else STATUS_UNSUPPORTED with one error line that names the first apart. That card holds
 *     the draft (holder V41_DRAFT): its arrays, a ring, a prompt-checkpoint window and an expert table of each stage;
 *     card->layers keeps the 40 layers, and v41_walk_layers adds the stages.
 *   - The state of a card begins behind its weights (loader.h), at state_offset, which like every array lies on a
 *     multiple of V41_DATA_ARRAY_ALIGNMENT_BYTES (weights that end off one: STATUS_INVALID_ARGUMENT); no
 *     two arrays share a byte. In order: where a layer of the card keeps routed experts in host memory, the slots
 *     of model_seam.h's expert pauses (MODEL_EXPERTS_PER_ROW times a slab's rows, each slot_byte_count:
 *     such an expert's bytes, aligned), the expert table of each of its walk layers, the arrays of
 *     V41_DATA_ARRAYS it holds, then for each walk layer the selection of an index layer whose group has a layer
 *     on the card, the ring and prompt-checkpoint window of a walk layer on it, the compressor's slots of a 2-to-1
 *     source on it, and of a source the pool if a layer on the card reads it and the key tiles if an index layer on
 *     it scores them. A layer reads
 *     the pool of its source, the latest source at or before it (REF:745-748); the selection of its group, the
 *     latest index layer at or before it (REF:723-726); an index layer scores its source's keys (REF:548-554, no
 *     shared slot). Where a card holds none: PROGRAM_NOWHERE.
 *   - Thin fusions share one front scratch and producer arrivals per layer card. A capacity below the minimum
 *     chunk's appends their bytes after all state; larger capacities reuse the contiguous wide-only routing
 *     arrays from ROUTER_PICKS up to JOBS (exclusive), which thin programs never read. Each thin card visit
 *     clears that scratch before use, so a preceding wide program cannot leave live counters. The fronts and
 *     producers leave their counters zero. Draft stages reuse the same scratch after the target's joins.
 *     Ordinary arrays, session offsets and arena sizes of chunk-capable plans remain unchanged.
 *   - The image of a card, image_byte_count bytes behind its slots, is the expert tables, uint64 [384][gate, up, down]
 *     (arena offsets of w1, w3, w2; three times model_seam.h's tag for an expert in host memory, numbered in ascending
 *     order among its layer's; zero from a draft stage's MODEL_DRAFT_EXPERTS on), then the frequency words;
 *     every byte behind the slots but the image begins zero. Where a layer keeps experts in host memory, its experts'
 *     range runs from its first on the card (or the slots) to the end of the slots, and state_byte_count of host memory
 *     counts the bytes of all such experts: the step driver's page-locked copies. Each card is opened with at least its
 *     arena_byte_count: the backend's digest of the plan's last word refuses a shorter arena before anything is
 *     written. The card writes its zeros (fill_words_entry), the image goes up through staging memory, and a card's
 *     digest of its state behind the slots that is not the host's (pack_digest.h) is STATUS_DEVICE_ERROR with
 *     one error line. Then, with the draft, embed.weight goes from the embedding card into DRAFT_EMBEDDING of the
 *     result card through staging, V41_DATA_EMBEDDING_COPY_PIECE_BYTES at a time (no second copy in the pack), and
 *     the cards' digests of both ranges must agree, else STATUS_DEVICE_ERROR with one error line.
 *   - Limits: a context of 0, above V41_DATA_CONTEXT_LIMIT or no multiple of MODEL_CONTEXT_GRANULE, a row
 *     capacity of 0 or above MODEL_WIDE_MOST_ROWS, a session capacity of 0, a width, profile or order outside
 *     its enumeration, draft positions other than 0 or the admitted 5..7 or above a slab's rows, and state
 *     beyond 64 bits are STATUS_INVALID_ARGUMENT, as is a thin program of more rows than a slab's; what
 *     model_seam.h calls beyond this build, STATUS_UNSUPPORTED, as is a program of another kind than thin or
 *     wide, and a wide one whose layers 0 to 20 keep routed experts in host memory. A row capacity above a slab's gives
 *     the chunk arrays its rows, the span arrays up to a span's (enum v41_rows).
 */
/* Relaxed load admits routed folds per layer after checking every resident expert matrix. Host-expert layers
 * retain reference order. Reference load does no expert scale downloads; PRO selection does not read admission.
 * Failed downloads fail load and retain unfinished staging until card close. */
#ifndef MODEL_V41_H
#define MODEL_V41_H

#include <stdbool.h>
#include <stdint.h>

#include "anemone/backend_seam.h"
#include "anemone/engram.h"
#include "anemone/linear.h"
#include "anemone/loader.h"
#include "anemone/model_layout.h"
#include "anemone/model_seam.h"
#include "anemone/router_records.h"
#include "anemone/site_front_records.h"
#include "anemone/status.h"
#include "anemone/step_records.h"
#include "anemone/wide_linear_records.h"
#include "model_vision.h"

#define V41_DATA_MODEL_NAME "DeepSeek-V4.1-Flash"
#define V41_DATA_CONTEXT_LIMIT \
	1048576u /* CFG:29 original_seq_len times CFG:31 rope_factor */
#define V41_DATA_END_TOKEN 1u /* CKPT:9 eos_token_id (GEN:157) */
#define V41_DATA_EXPERT_INNER 2304u /* CFG:4 moe_inter_dim */
#define V41_DATA_ENGRAM_ROWS_PER_TOKEN \
	24u /* (CFG:46 - 1) times CFG:63: a token's rows of a table */
#define V41_DATA_SOURCE_LAYERS \
	0x104104ull /* bit l for layer l: CFG:27 kv_source_layers 2, 8, 14, 20 */
#define V41_DATA_POOLED_LAYERS \
	0x4104ull /* the sources of ratio 2 (CFG:52): 2, 8, 14 */
#define V41_DATA_INDEX_LAYERS \
	0x1111104104ull /* CFG:28 index_source_layers 2, 8, .. 36 */
#define V41_DATA_CANDIDATE_READERS \
	0x1111000000ull /* index layers after CFG:37: 24, 28, 32, 36 */
#define V41_DATA_ENGRAM_LAYERS \
	0x4002ull /* CFG:43 engram_layer_ids 1, 14: tables 0 and 1 */
#define V41_DATA_ARRAY_ALIGNMENT_BYTES \
	256u /* ours: what the cards' allocations give an array */
#define V41_DATA_RING_SESSION_BYTES \
	((uint64_t)MODEL_RING_SLOTS * MODEL_STATE_ROW_BYTES)
#define V41_DATA_CHECKPOINT_SESSION_BYTES \
	((uint64_t)(MODEL_WINDOW - 1u) * MODEL_STATE_ROW_BYTES)
#define V41_DATA_COMPRESSOR_SESSION_BYTES \
	65536u /* = ATTENTION_SLOTS_SESSION_BYTES: model_v41_program.c */
#define V41_DATA_WIDEST_ROW \
	8192u /* quantized rows of a layer: wo_a's output, CFG:24 x CFG:25 */
#define V41_DATA_SPAN_ROWS \
	64u /* ours: 8 slabs, the rows of a chunk's queries launch */
#define V41_DATA_INDEX_SPAN_BYTES \
	(32ull                    \
	 << 20u) /* ours: index_span's scores a card at most (tp2_stage.h) */
#define V41_DATA_HALF_VIEWS \
	2u /* ours: a wide plan's views of the chunk's two halves */
#define V41_DATA_PAIR_WIDTH \
	(MODEL_EXPERTS_PER_ROW * V41_DATA_EXPERT_INNER) /* a row's hiddens */
#define V41_DATA_PRODUCER_ARRIVALS_BYTES                         \
	((uint64_t)LINEAR_MAXIMUM_ROWS * MODEL_EXPERTS_PER_ROW * \
	 (V41_DATA_EXPERT_INNER / LINEAR_VALUES_PER_BLOCK) * sizeof(uint32_t))
#define V41_DATA_FUSED_SCRATCH_BYTES \
	(SITE_FRONT_SCRATCH_BYTES + V41_DATA_PRODUCER_ARRIVALS_BYTES)
#define V41_DATA_ENGRAM_WIDTH \
	(V41_DATA_ENGRAM_ROWS_PER_TOKEN * ENGRAM_VALUE_BYTES_VALUE)
#define V41_DATA_BF16 LINEAR_BFLOAT16_BYTES
#define V41_DATA_F32 LINEAR_SUM_BYTES
#define V41_DATA_WALK_LAYERS \
	43u /* the 40 layers and the draft's stages, layers 40 to 42 */
#define V41_DATA_DRAFT_LAYERS \
	0x70000000000ull /* bit l for each draft stage l (REF:1213-1214) */
#define V41_DATA_DRAFT_TAP_LAYERS \
	0xE000000000ull /* layers 37 to 39, whose inputs the draft taps (CFG:9) */
#define V41_DATA_DRAFT_EMBEDDING_BYTES \
	((uint64_t)MODEL_VOCABULARY_VALUE * MODEL_HIDDEN_ROW_BYTES)
#define V41_DATA_EMBEDDING_COPY_PIECE_BYTES \
	(16ull << 20u) /* ours: what the draft's copy stages at a time */

/* The tensors of a layer, of a routed expert (in the order of an expert table's row) and outside the layers. */
enum v41_layer_tensor {
	V41_DATA_ATTENTION_NORM,
	V41_DATA_FFN_NORM,
	V41_DATA_ATTENTION_MIX,
	V41_DATA_ATTENTION_MIX_BASE,
	V41_DATA_ATTENTION_MIX_SCALE,
	V41_DATA_FFN_MIX,
	V41_DATA_FFN_MIX_BASE,
	V41_DATA_FFN_MIX_SCALE,
	V41_DATA_SINK,
	V41_DATA_QUERY_DOWN,
	V41_DATA_QUERY_NORM,
	V41_DATA_QUERY_UP,
	V41_DATA_WINDOW,
	V41_DATA_WINDOW_NORM,
	V41_DATA_OUTPUT_GROUPS,
	V41_DATA_OUTPUT,
	V41_DATA_ROUTER,
	V41_DATA_ROUTER_BIAS,
	V41_DATA_SHARED_GATE,
	V41_DATA_SHARED_UP,
	V41_DATA_SHARED_DOWN,
	V41_DATA_COMPRESSOR,
	V41_DATA_COMPRESSOR_GATE,
	V41_DATA_COMPRESSOR_NORM,
	V41_DATA_KEY,
	V41_DATA_KEY_NORM,
	V41_DATA_INDEX_QUERY_UP,
	V41_DATA_INDEX_WEIGHTS,
	V41_DATA_ENGRAM,
	V41_DATA_ENGRAM_QUERY,
	V41_DATA_ENGRAM_KEY,
	V41_DATA_LAYER_TENSOR_COUNT
};
enum v41_expert_tensor {
	V41_DATA_GATE,
	V41_DATA_UP,
	V41_DATA_DOWN,
	V41_DATA_EXPERT_TENSOR_COUNT
};
enum v41_global_tensor {
	V41_DATA_EMBEDDING,
	V41_DATA_NORM,
	V41_DATA_HEAD,
	V41_DATA_GLOBALS
};
/* The draft's tensors outside its stages: main_proj and main_norm (stage 0's), its final norm, the Markov head's two
 * matrices and the confidence projection (stage 2's). */
enum v41_draft_global {
	V41_DATA_DRAFT_MAIN_PROJECTION,
	V41_DATA_DRAFT_MAIN_NORM,
	V41_DATA_DRAFT_NORM,
	V41_DATA_DRAFT_MARKOV_EMBEDDING,
	V41_DATA_DRAFT_MARKOV_HEAD,
	V41_DATA_DRAFT_CONFIDENCE,
	V41_DATA_DRAFT_GLOBALS
};

/* Which cards hold an array: those that run the embedding, a layer or the head; a layer; an index layer; a source;
 * a 2-to-1 source; an n-gram layer; the head (the card that samples); the candidate source or a layer that reads its
 * candidates; the draft (the result card, with the draft admitted; with draft_layers every rank of a TP stage); the
 * draft's copy of the embedding (the result card). */
enum v41_holder {
	V41_TRUNK,
	V41_LAYER,
	V41_INDEX,
	V41_SOURCE,
	V41_POOLED,
	V41_ENGRAM,
	V41_RESULT,
	V41_CANDIDATES,
	V41_DRAFT,
	V41_DRAFT_COPY
};

/* Which rows an array has: a chunk's, the plan's row capacity (what a wide step carries whole: the streams and what a
 * launch over the whole chunk reads or writes); a span's, V41_DATA_SPAN_ROWS at most (what a launch over a span of
 * a chunk's slabs writes for those slabs to read at their rows before the next span's launch); or a slab's, the rows a
 * thin launch takes (at most LINEAR_MAXIMUM_ROWS: scratch the slabs of a wide step reuse one after another). */
enum v41_rows {
	V41_CHUNK,
	V41_SPAN,
	V41_SLAB,
	V41_DRAFT_ROWS
}; /* configured draft block, never a chunk */

/* A slab's rows under a row capacity: the most rows a thin step carries. */
static inline uint32_t v41_slab_rows(uint32_t row_capacity)
{
	return row_capacity < LINEAR_MAXIMUM_ROWS ? row_capacity :
						    LINEAR_MAXIMUM_ROWS;
}

/* A span's rows under a row capacity. */
static inline uint32_t v41_span_rows(uint32_t row_capacity)
{
	return row_capacity < V41_DATA_SPAN_ROWS ? row_capacity :
						   V41_DATA_SPAN_ROWS;
}

/* X(array, bytes a row, bytes besides, holder, rows). Apart: SESSIONS has STEP_SESSION_BYTES a session slot,
 * INDEX_SCORES FP16 [row][the most entries an index layer of the card scores], plus the split selection annex
 * described in index_scan.h where its capacity is eligible; ENGRAM_ROWS uint8 [table of the card]
 * [row][24][ENGRAM_ROW_BYTES] (model_seam.h); JOBS the more of a thin step's 6 a row and the wide select's
 * table for the row capacity, ROUTER_COUNTS and ROUTER_TILES the wide select's for it (router.h); VIEWS a step view's
 * block and rows (model_seam.h) for each slab of a chunk, none where a chunk is one slab; COMPRESSOR_KV half its row
 * on a card that holds no 2-to-1 source. The output of a product is named for it. An array that a chunk's launches
 * read or write across its slabs is a chunk's: what a launch over the chunk (a piece of it: linear.h's and
 * dense_linear.h's wide launches) reads or writes, which its slabs' launches write or read at their rows (the query
 * rank, the window input, the heads and the indexer's, the attention output, the groups, the sublayer output, the
 * shared expert's gate and up, the n-gram values and projection, the indexer's weights, the compressor's products;
 * model_v41_program.c's chunk walk), and what a full program reads across its slabs: QUERY_NORMED from layer 20's
 * front to the rest of it, CANDIDATES from layer 20 to the candidate readers. The queries and the index queries are a
 * span's. A product's rows lie packed at its output's row: COMPRESSOR_KV of a 1-to-1 source, bfloat16, at half its
 * row. The draft's (dspark_v1/DESIGN.md 5.2): ROUND, the round record (step_records.h)
 * right behind RESULT, so that one download takes both; DRAFT_EMBEDDING, the copy of embed.weight; DRAFT_TAPS (model
 * layout's draft taps) and what the context makes of them, main_proj's quantized rows (a piece's packed at their
 * width), its output, main_norm's and each stage's wkv product, a chunk's; the step area the draft's launches read,
 * its one-row views, logits, one-row result areas, pre-norm collapse and Markov rows of the block's positions. */
#define V41_DATA_OFFICIAL_SCRATCH_BYTES (128ull * 1024ull)
#define V41_DATA_ARRAYS(X)                                                     \
	X(FREQUENCIES, 0, sizeof(float[MODEL_THETAS][MODEL_FREQUENCIES]),      \
	  V41_LAYER, V41_CHUNK)                                                \
	X(STEP, STEP_ROW_BYTES_VALUE, STEP_BLOCK_BYTES_VALUE, V41_TRUNK,       \
	  V41_CHUNK)                                                           \
	X(VIEWS, 0, 0, V41_TRUNK, V41_CHUNK)                                   \
	X(RESULT, STEP_ROW_RESULT_BYTES, STEP_RESULT_BYTES, V41_RESULT,        \
	  V41_SLAB)                                                            \
	X(ROUND, 0, STEP_ROUND_BYTES, V41_DRAFT, V41_CHUNK)                    \
	X(SESSIONS, 0, 0, V41_RESULT, V41_CHUNK)                               \
	X(STREAMS, MODEL_STREAMS_ROW_BYTES, 0, V41_TRUNK, V41_CHUNK)           \
	X(CARRIED_PRE, MODEL_SITE_ROW_BYTES, 0, V41_TRUNK, V41_CHUNK)          \
	X(HIDDEN, MODEL_HIDDEN_ROW_BYTES, 0, V41_TRUNK,                        \
	  V41_CHUNK) /* a site's normed input */                               \
	X(ROTATION, MODEL_ROTATION_ROW_BYTES, 0, V41_LAYER, V41_CHUNK)         \
	X(MIXES, MODEL_MIXES_ROW_BYTES, 0, V41_LAYER, V41_CHUNK)               \
	X(POST, MODEL_SITE_ROW_BYTES, 0, V41_LAYER, V41_CHUNK)                 \
	X(COMB, MODEL_COMB_ROW_BYTES, 0, V41_LAYER, V41_CHUNK)                 \
	X(ROW_CODES, V41_DATA_WIDEST_ROW, 0, V41_LAYER,                        \
	  V41_CHUNK) /* every quantized row: linear.h */                       \
	X(ROW_EXPONENTS, V41_DATA_WIDEST_ROW / LINEAR_VALUES_PER_BLOCK, 0,     \
	  V41_LAYER, V41_CHUNK)                                                \
	X(ROW_OPERANDS, V41_DATA_WIDEST_ROW *V41_DATA_BF16, 0, V41_LAYER,      \
	  V41_CHUNK)                                                           \
	X(QUERY_RANK, MODEL_QUERY_RANK *V41_DATA_BF16, 0, V41_LAYER,           \
	  V41_CHUNK) /* wq_a */                                                \
	X(WINDOW_INPUT, MODEL_HEAD_DIMS *V41_DATA_BF16, 0, V41_LAYER,          \
	  V41_CHUNK) /* wkv */                                                 \
	X(QUERY_NORMED, MODEL_QUERY_RANK *V41_DATA_BF16, 0, V41_LAYER,         \
	  V41_CHUNK) /* qr */                                                  \
	X(HEADS, MODEL_HEADS_ROW_BYTES, 0, V41_LAYER, V41_CHUNK) /* wq_b */    \
	X(QUERIES, MODEL_HEADS_ROW_BYTES, 0, V41_LAYER,                        \
	  V41_CHUNK) /* a span at a time; chunk_core's */                      \
	X(ATTENTION_OUTPUT, MODEL_HEADS_ROW_BYTES, 0, V41_LAYER, V41_CHUNK)    \
	X(ATTENTION_SCORES,                                                    \
	  MODEL_HEADS_VALUE *(MODEL_WINDOW + MODEL_SELECTED) * V41_DATA_F32,   \
	  0, V41_LAYER,                                                        \
	  V41_CHUNK) /* the core's scratch: a slab's, or chunk_core's chunk */ \
	X(GROUPS, V41_DATA_WIDEST_ROW *V41_DATA_BF16, 0, V41_LAYER,            \
	  V41_CHUNK) /* wo_a */                                                \
	X(SUBLAYER_OUTPUT, MODEL_HIDDEN_ROW_BYTES, 0, V41_LAYER,               \
	  V41_CHUNK) /* wo_b */                                                \
	X(ROUTER_SCORES, MODEL_ROUTER_ROW_BYTES, 0, V41_LAYER, V41_CHUNK)      \
	X(AFFINITIES, MODEL_ROUTER_ROW_BYTES, 0, V41_LAYER,                    \
	  V41_CHUNK) /* the router's, of each expert */                        \
	X(ROUTER_PICKS, MODEL_EXPERTS_PER_ROW * sizeof(struct router_pick), 0, \
	  V41_LAYER, V41_CHUNK)                                                \
	X(ROUTER_COUNTS, 0, 0, V41_LAYER, V41_CHUNK)                           \
	X(ROUTER_PAIRS, MODEL_EXPERTS_PER_ROW * sizeof(struct wide_pair), 0,   \
	  V41_LAYER, V41_CHUNK)                                                \
	X(ROUTER_RUNS, 0, MODEL_EXPERTS * sizeof(struct wide_run), V41_LAYER,  \
	  V41_CHUNK)                                                           \
	X(ROUTER_TILES, 0, 0, V41_LAYER, V41_CHUNK)                            \
	X(JOBS, MODEL_EXPERTS_PER_ROW *EXPERT_JOB_BYTES, 0, V41_LAYER,         \
	  V41_SLAB)                                                            \
	X(PAIR_HIDDEN, V41_DATA_PAIR_WIDTH *V41_DATA_BF16, 0, V41_LAYER,       \
	  V41_CHUNK) /* the shared expert's too */                             \
	X(PAIR_CODES, V41_DATA_PAIR_WIDTH, 0, V41_LAYER, V41_CHUNK)            \
	X(PAIR_EXPONENTS, V41_DATA_PAIR_WIDTH / LINEAR_VALUES_PER_BLOCK, 0,    \
	  V41_LAYER, V41_CHUNK)                                                \
	X(PAIR_OPERANDS, V41_DATA_PAIR_WIDTH *V41_DATA_BF16, 0, V41_LAYER,     \
	  V41_CHUNK)                                                           \
	X(PAIR_OUTPUT, MODEL_EXPERTS_PER_ROW *MODEL_HIDDEN_ROW_BYTES, 0,       \
	  V41_LAYER, V41_CHUNK)                                                \
	X(ROW_SUMS, MODEL_HIDDEN *V41_DATA_F32, 0, V41_LAYER, V41_CHUNK)       \
	X(SHARED_GATE, V41_DATA_EXPERT_INNER *V41_DATA_BF16, 0, V41_LAYER,     \
	  V41_CHUNK)                                                           \
	X(SHARED_UP, V41_DATA_EXPERT_INNER *V41_DATA_BF16, 0, V41_LAYER,       \
	  V41_CHUNK)                                                           \
	X(SHARED_OUTPUT, MODEL_HIDDEN_ROW_BYTES, 0, V41_LAYER, V41_CHUNK)      \
	X(INDEX_HEADS, MODEL_INDEX_QUERIES_ROW_BYTES, 0, V41_INDEX,            \
	  V41_CHUNK) /* the indexer's wq_b */                                  \
	X(INDEX_QUERIES, MODEL_INDEX_QUERIES_ROW_BYTES, 0, V41_INDEX,          \
	  V41_CHUNK)                                                           \
	X(INDEX_WEIGHTS, MODEL_INDEX_WEIGHTS_ROW_BYTES, 0, V41_INDEX,          \
	  V41_CHUNK)                                                           \
	X(INDEX_SCORES, 0, 0, V41_INDEX, V41_SLAB)                             \
	X(CANDIDATES, MODEL_CANDIDATES_ROW_BYTES, 0, V41_CANDIDATES,           \
	  V41_CHUNK) /* uint32 [row][2,048] */                                 \
	X(COMPRESSOR_KV, MODEL_HEAD_DIMS *V41_DATA_F32, 0, V41_SOURCE,         \
	  V41_CHUNK)                                                           \
	X(COMPRESSOR_SCORE, MODEL_HEAD_DIMS *V41_DATA_F32, 0, V41_POOLED,      \
	  V41_CHUNK)                                                           \
	X(LATENT, MODEL_HEAD_DIMS *V41_DATA_BF16, 0, V41_SOURCE,               \
	  V41_CHUNK) /* bfloat16, for wk */                                    \
	X(KEY_INPUT, MODEL_KEY_BYTES, 0, V41_SOURCE, V41_CHUNK) /* wk */       \
	X(ENGRAM_ROWS, 0, 0, V41_ENGRAM, V41_CHUNK)                            \
	X(ENGRAM_VALUES, V41_DATA_ENGRAM_WIDTH *V41_DATA_BF16, 0, V41_ENGRAM,  \
	  V41_CHUNK)                                                           \
	X(ENGRAM_PROJECTION, (MODEL_STREAMS + 1u) * MODEL_HIDDEN_ROW_BYTES, 0, \
	  V41_ENGRAM, V41_CHUNK)                                               \
	X(LOGITS, MODEL_LOGITS_ROW_BYTES, 0, V41_RESULT, V41_SLAB)             \
	X(SAMPLER_WEIGHTS, MODEL_LOGITS_ROW_BYTES, 0, V41_RESULT, V41_SLAB)    \
	X(DRAFT_EMBEDDING, 0, V41_DATA_DRAFT_EMBEDDING_BYTES, V41_DRAFT_COPY,  \
	  V41_CHUNK)                                                           \
	X(DRAFT_TAPS, MODEL_DRAFT_TAP_ROW_BYTES, 0, V41_DRAFT, V41_CHUNK)      \
	X(DRAFT_TAP_CODES, MODEL_DRAFT_FEATURES, 0, V41_DRAFT, V41_CHUNK)      \
	X(DRAFT_TAP_EXPONENTS, MODEL_DRAFT_FEATURES / LINEAR_VALUES_PER_BLOCK, \
	  0, V41_DRAFT, V41_CHUNK)                                             \
	X(DRAFT_TAP_OPERANDS, MODEL_DRAFT_FEATURES *V41_DATA_BF16, 0,          \
	  V41_DRAFT, V41_CHUNK)                                                \
	X(DRAFT_MAIN, MODEL_HIDDEN_ROW_BYTES, 0, V41_DRAFT,                    \
	  V41_CHUNK) /* main_proj */                                           \
	X(DRAFT_MAIN_X, MODEL_HIDDEN_ROW_BYTES, 0, V41_DRAFT,                  \
	  V41_CHUNK) /* main_norm */                                           \
	X(DRAFT_WINDOW_0, MODEL_HEAD_DIMS *V41_DATA_BF16, 0, V41_DRAFT,        \
	  V41_CHUNK) /* mtp.0.attn.wkv */                                      \
	X(DRAFT_WINDOW_1, MODEL_HEAD_DIMS *V41_DATA_BF16, 0, V41_DRAFT,        \
	  V41_CHUNK)                                                           \
	X(DRAFT_WINDOW_2, MODEL_HEAD_DIMS *V41_DATA_BF16, 0, V41_DRAFT,        \
	  V41_CHUNK)                                                           \
	X(DRAFT_STEP, STEP_ROW_BYTES_VALUE, STEP_BLOCK_BYTES_VALUE, V41_DRAFT, \
	  V41_DRAFT_ROWS)                                                      \
	X(DRAFT_VIEWS, STEP_DRAFT_VIEW_BYTES, 0, V41_DRAFT, V41_DRAFT_ROWS)    \
	X(DRAFT_LOGITS, MODEL_LOGITS_ROW_BYTES, 0, V41_DRAFT, V41_DRAFT_ROWS)  \
	X(DRAFT_RESULTS, STEP_DRAFT_RESULT_BYTES, 0, V41_DRAFT,                \
	  V41_DRAFT_ROWS)                                                      \
	X(DRAFT_COLLAPSE, MODEL_HIDDEN_ROW_BYTES, 0, V41_DRAFT,                \
	  V41_DRAFT_ROWS)                                                      \
	X(DRAFT_MARKOV, MODEL_MARKOV_RANK *V41_DATA_BF16, 0, V41_DRAFT,        \
	  V41_DRAFT_ROWS)
#define V41_DATA_ENUMERATOR(array, ...) V41_DATA_ARRAY_##array,
enum v41_array { V41_DATA_ARRAYS(V41_DATA_ENUMERATOR) V41_DATA_ARRAY_COUNT };
#undef V41_DATA_ENUMERATOR

/* Where a tensor lies: the arena of card, from stored.byte_offset; a vector has input_size 1. All zero for a tensor
 * of a layer that has none. */
struct v41_tensor {
	struct linear_range stored;
	struct matrix_shape_record shape;
	enum weight_format_code weight_format;
	uint32_t card;
	int32_t fold_reference; /* 0: reference order; set at relaxed load after scale checks */
};

/* A card: per walk layer the offsets of what model_v41.h's contract lists, the expert table first. */
struct v41_card {
	uint64_t array_offset[V41_DATA_ARRAY_COUNT];
	uint64_t expert_table_offset[V41_DATA_WALK_LAYERS];
	uint64_t selection_offset[V41_DATA_WALK_LAYERS]; /* int32 [row][512] */
	uint64_t ring_offset
		[V41_DATA_WALK_LAYERS]; /* V41_DATA_RING_SESSION_BYTES a session */
	uint64_t checkpoint_offset
		[V41_DATA_WALK_LAYERS]; /* V41_DATA_CHECKPOINT_SESSION_BYTES a session */
	uint64_t compressor_offset
		[V41_DATA_WALK_LAYERS]; /* V41_DATA_COMPRESSOR_SESSION_BYTES a session */
	uint64_t pool_offset
		[V41_DATA_WALK_LAYERS]; /* pool_session_bytes a session */
	uint64_t key_tiles_offset
		[V41_DATA_WALK_LAYERS]; /* key_session_bytes a session */
	uint64_t state_offset; /* the weights of the card; its slots, if any */
	uint64_t arena_byte_count; /* weights and state */
	uint64_t official_scratch_offset; /* exact recipe only; shared by serial layer passes */
	uint64_t front_scratch_offset; /* serial thin fronts; zero counters at each thin visit */
	uint64_t exl3_scratch_offset; /* cold EXL3 routed passes; NOWHERE for native weights */
	uint64_t producer_arrivals_offset; /* behind front scratch; zero after every producer */
	uint64_t slots_byte_count; /* 0 for none; the image lies behind them */
	uint64_t image_byte_count;
	unsigned char *image; /* the plan's; NULL for none */
	uint64_t layers; /* bit l: the card runs layer l */
	uint32_t holders; /* bit h: a holder h (enum v41_holder) */
	uint32_t engram_tables; /* bit t: it runs the n-gram layer of table t */
	uint32_t slot_count;
	uint32_t reserved_zero;
};

/* The walk layers whose state the card holds: its layers, and on the draft's card the draft's stages. */
static inline uint64_t v41_walk_layers(const struct v41_card *card)
{
	return card->layers |
	       ((card->holders >> V41_DRAFT & 1u) != 0 ? V41_DATA_DRAFT_LAYERS :
							 0);
}

/* Per walk layer: a draft stage's members are those of a layer that has no source, index or n-gram table. */
struct model_plan_record {
	struct model_limits limits;
	struct v41_card cards[BACKEND_MAXIMUM_CARDS];
	struct v41_tensor layer_tensors[V41_DATA_WALK_LAYERS]
				       [V41_DATA_LAYER_TENSOR_COUNT];
	struct v41_tensor globals[V41_DATA_GLOBALS];
	struct v41_tensor draft_globals
		[V41_DATA_DRAFT_GLOBALS]; /* all zero without the draft */
	struct linear_range experts
		[V41_DATA_WALK_LAYERS]; /* holds every routed expert matrix of the layer */
	uint64_t expert_tables[V41_DATA_WALK_LAYERS][MODEL_EXPERTS]
			      [MODEL_EXPERT_MATRICES]; /* images' */
	uint64_t host_expert_offsets[V41_DATA_WALK_LAYERS]
				    [MODEL_EXPERTS]; /* in the host arena, by tag */
	uint64_t expert_matrix_offsets
		[MODEL_EXPERT_MATRICES]; /* inside a host expert's bytes */
	uint64_t expert_byte_count; /* of a host expert; 0 for none */
	uint64_t slot_byte_count;
	uint32_t host_expert_count[V41_DATA_WALK_LAYERS];
	uint32_t exl3_bits
		[V41_DATA_WALK_LAYERS]; /* zero for original-format routed weights */
	bool expert_folds_admitted
		[V41_DATA_WALK_LAYERS]; /* every resident expert scale passed at relaxed load */
	uint64_t pool_session_bytes[V41_DATA_WALK_LAYERS]; /* of a source; else 0 */
	uint64_t key_session_bytes[V41_DATA_WALK_LAYERS];
	uint32_t layer_card
		[V41_DATA_WALK_LAYERS]; /* a draft stage's: the result card */
	uint32_t embedding_card;
	uint32_t result_card;
	uint32_t card_count;
	uint32_t reserved_zero;
	struct model_vision vision;
};

MUST_CHECK result_status
v41_describe(const struct pack_record *pack,
	     struct model_description_record *description_out);
MUST_CHECK result_status
v41_state_byte_count(const struct pack_record *pack,
		     const struct placement_record *placement,
		     const struct model_limits *limits, uint32_t card_index,
		     uint64_t *byte_count_out);
MUST_CHECK result_status v41_plan_create(const struct pack_record *pack,
					 const struct load_layout *layout,
					 const struct model_limits *limits,
					 struct model_plan_record **plan_out);
void v41_plan_destroy(struct model_plan_record *plan);
MUST_CHECK result_status
v41_arena_byte_count(const struct model_plan_record *plan, uint32_t card_index,
		     uint64_t *byte_count_out);
MUST_CHECK result_status v41_arena_initialise(
	struct model_plan_record *plan,
	const struct backend_operations *backend,
	struct device_card *const *cards, uint64_t wait_limit_nanoseconds);

/* Image-capable counterpart to the ordinary program, prepared only after all text programs. It retains
 * their state metadata, while image-row embedding/routing/gating uses the optional tower's appended state. */
MUST_CHECK result_status
v41_image_program_prepare(const struct model_plan_record *plan,
			  const struct backend_operations *backend,
			  struct device_card *const *cards,
			  const struct card_description_record *descriptions,
			  enum program_kind kind, uint32_t row_count,
			  struct execution_program **program_out);

/* The words of both thetas (model_layout.h), bit for bit those REF:376-387 forms on the card: the reference's own,
 * a table in model_v41.c. */
void v41_frequency_words(float words_out[MODEL_THETAS][MODEL_FREQUENCIES]);

#endif
