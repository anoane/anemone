/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * model_v41.h: DeepSeek-V4.1-Flash behind the model seam: the tensors its programs read, the plan of every card's
 * arena and the frequency words. Private to source/: model_v41.c makes a plan, model_v41_program.c the programs and
 * the model's table of model_seam.h, which these functions fill as model_seam.h says.
 *
 * Contract:
 *   - Numbers cite the reference as ds41f_geometry.h does; CKPT is the checkpoint's config.json. Each function answers
 *     STATUS_UNSUPPORTED for a pack that names another model.
 *   - Tensors: the table of model_v41.c, by layer, routed expert and outside the layers. Each must be in the pack
 *     with that element type, shape, weight format and bits, else STATUS_CORRUPT_PACK and one error line
 *     that names it. N-gram tables are read separately, and the draft head only when the limits ask for the draft.
 *     A vision tower is optional: if any tower tensor is present the entire official ViT/projector, all learned
 *     delimiter embeddings and every layer's image router bias must have the pinned types/shapes
 *     (ds41f_vision_tower.h). Tower/projector weights and the embedding lie on card 0, image routing biases on their
 *     layer's card. Its scratch/input/output/session ranges append behind ordinary state; ordinary programs retain all
 *     original records and state metadata. Image programs are prepared separately and selected only for image-bearing
 *     rows.
 *   - A layer runs on the card of its tensors, the embedding on that of embed.weight, the head and the sampler on
 *     that of norm.weight and head.weight (the result card). A routed expert lies whole on its layer's card or whole
 *     in host memory, there its three matrices together as in every other (from the first byte of one to the last
 *     of another, with less than WEIGHT_FORMAT_MATRIX_ALIGNMENT_BYTES between them). Anything else is
 *     STATUS_UNSUPPORTED with one error line.
 *   - The draft (dspark_v1/DESIGN.md 5.2), with limits.draft_positions from DS41F_DRAFT_BLOCK to
 *     STEP_DRAFT_MOST; 0 plans none, and the
 *     plan is that of a pack without a draft head, byte for byte. Its stages 0 to 2 are walk layers 40 to 42
 *     (DS41F_WALK_LAYERS): a layer's tensors under mtp.<stage>. but those of the compressor, the indexer and the
 *     n-gram rows, its router for DS41F_DRAFT_EXPERTS experts, which each stage has; then the draft's globals.
 *     Every draft tensor, its routed experts included, and layers 37 to 39, whose inputs the draft taps, lie on the
 *     result card, else STATUS_UNSUPPORTED with one error line that names the first apart. That card holds
 *     the draft (holder DS41F_HOLDER_DRAFT): its arrays, a ring, a prompt-checkpoint window and an expert table of each
 *     stage; card->layers keeps the 40 layers, and ds41f_walk_layers adds the stages.
 *   - The state of a card begins behind its weights (loader.h), at state_offset, which like every array lies on a
 *     multiple of DS41F_ARRAY_ALIGNMENT_BYTES (weights that end off one: STATUS_INVALID_ARGUMENT); no
 *     two arrays share a byte. In order: where a layer of the card keeps routed experts in host memory, the slots
 *     of model_seam.h's expert pauses (DS41F_EXPERTS_PER_ROW times a slab's rows, each slot_byte_count:
 *     such an expert's bytes, aligned), the expert table of each of its walk layers, the arrays of
 *     DS41F_ARRAYS it holds, then for each walk layer the selection of an index layer whose group has a layer
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
 *     order among its layer's; zero from a draft stage's DS41F_DRAFT_EXPERTS on), then the frequency words;
 *     every byte behind the slots but the image begins zero. Where a layer keeps experts in host memory, its experts'
 *     range runs from its first on the card (or the slots) to the end of the slots, and state_byte_count of host memory
 *     counts the bytes of all such experts: the step driver's page-locked copies. Each card is opened with at least its
 *     arena_byte_count: the backend's digest of the plan's last word refuses a shorter arena before anything is
 *     written. The card writes its zeros (fill_words_entry), the image goes up through staging memory, and a card's
 *     digest of its state behind the slots that is not the host's (pack_digest.h) is STATUS_DEVICE_ERROR with
 *     one error line. Then, with the draft, embed.weight goes from the embedding card into DRAFT_EMBEDDING of the
 *     result card through staging, DS41F_EMBEDDING_COPY_PIECE_BYTES at a time (no second copy in the pack), and
 *     the cards' digests of both ranges must agree, else STATUS_DEVICE_ERROR with one error line.
 *   - Limits: a context of 0, above DS41F_CONTEXT_MOST or no multiple of DS41F_CONTEXT_GRANULE, a row
 *     capacity of 0 or above STEP_WIDE_MOST_ROWS, a session capacity of 0, a width, profile or order outside
 *     its enumeration, draft positions other than 0 or the admitted 5..7 or above a slab's rows, and state
 *     beyond 64 bits are STATUS_INVALID_ARGUMENT, as is a thin program of more rows than a slab's; what
 *     model_seam.h calls beyond this build, STATUS_UNSUPPORTED, as is a program of another kind than thin or
 *     wide, and a wide one whose layers 0 to 20 keep routed experts in host memory. A row capacity above a slab's gives
 *     the chunk arrays its rows, the span arrays up to a span's (enum ds41f_rows).
 */
/* Relaxed load admits routed folds per layer after checking every resident expert matrix. Host-expert layers
 * retain reference order. Reference load does no expert scale downloads; PRO selection does not read admission.
 * Failed downloads fail load and retain unfinished staging until card close. */
#ifndef MODEL_V41_H
#define MODEL_V41_H

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

#include "engine/backend/backend_seam.h"
#include "engine/engram/engram.h"
#include "engine/library/block_scaled_linear/linear.h"
#include "engine/storage/load/loader.h"
#include "models/deepseek_v41_flash/ds41f_geometry.h"
#include "engine/program/model_seam.h"
#include "models/deepseek_v41_flash/moe/ds41f_router_records.h"
#include "models/deepseek_v41_flash/hyper_connection/ds41f_site_front_records.h"
#include "common/status.h"
#include "engine/program/step_records.h"
#include "engine/library/block_scaled_linear/expert_tile_records.h"
#include "models/deepseek_v41_flash/vision/ds41f_vision_tower.h"

#define DS41F_MODEL_NAME "DeepSeek-V4.1-Flash"
#define DS41F_CONFIG_MODEL_TYPE \
	"deepseek_v41" /* config.json's model_type: an EXL3 map descriptor's architecture */
/* The EXL3 packs whose quality is established without a descriptor beside them (exl3_descriptor.h), by what the pack
 * carries: TP4_EXL3's default 2.9 bpw map, its pack's header digest and size as P1's rebuilt map's descriptor b0612d2f
 * records them (EXL3_FLEXIBILITY_DESIGN.md 3a') */
#define DS41F_1017_EXL3_29_SOURCE \
	"LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw @ 0be63c33"
#define DS41F_1017_EXL3_29_HEADER_DIGEST UINT64_C(17897322250946657769)
#define DS41F_1017_EXL3_29_BYTES UINT64_C(424522379264)
#define DS41F_1017_ESTABLISHED_EXL3_PACKS                         \
	{                                                         \
		{                                                 \
			DS41F_1017_EXL3_29_SOURCE,                \
				DS41F_1017_EXL3_29_HEADER_DIGEST, \
				DS41F_1017_EXL3_29_BYTES          \
		}                                                 \
	}
#define DS41F_END_TOKEN 1u /* CKPT:9 eos_token_id (GEN:157) */
#define DS41F_EXPERT_INNER 2304u /* CFG:4 moe_inter_dim */
#define DS41F_ENGRAM_ROWS_PER_TOKEN \
	24u /* (CFG:46 - 1) times CFG:63: a token's rows of a table */
#define DS41F_SOURCE_LAYERS \
	0x104104ull /* bit l for layer l: CFG:27 kv_source_layers 2, 8, 14, 20 */
#define DS41F_POOLED_LAYERS \
	0x4104ull /* the sources of ratio 2 (CFG:52): 2, 8, 14 */
#define DS41F_INDEX_LAYERS \
	0x1111104104ull /* CFG:28 index_source_layers 2, 8, .. 36 */
#define DS41F_CANDIDATE_READERS \
	0x1111000000ull /* index layers after CFG:37: 24, 28, 32, 36 */
#define DS41F_ENGRAM_LAYERS \
	0x4002ull /* CFG:43 engram_layer_ids 1, 14: tables 0 and 1 */
#define DS41F_ARRAY_ALIGNMENT_BYTES \
	256u /* ours: what the cards' allocations give an array */
#define DS41F_CHECKPOINT_SESSION_BYTES \
	((uint64_t)(DS41F_WINDOW - 1u) * DS41F_STATE_ROW_BYTES)
#define DS41F_COMPRESSOR_SESSION_BYTES \
	65536u /* = DS41F_ATTENTION_SLOTS_SESSION_BYTES: model_v41_program.c */
#define DS41F_WIDEST_ROW \
	8192u /* quantized rows of a layer: wo_a's output, CFG:24 x CFG:25 */
#define DS41F_SPAN_ROWS \
	64u /* ours: 8 slabs, the rows of a chunk's queries launch */
#define DS41F_INDEX_SPAN_BYTES \
	(32ull                 \
	 << 20u) /* ours: index_span's scores a card at most (tp2_stage.h) */
#define DS41F_HALF_VIEWS \
	2u /* ours: a wide plan's views of the chunk's two halves */
#define DS41F_PAIR_WIDTH \
	(DS41F_EXPERTS_PER_ROW * DS41F_EXPERT_INNER) /* a row's hiddens */
#define DS41F_PRODUCER_ARRIVALS_BYTES                            \
	((uint64_t)LINEAR_MAXIMUM_ROWS * DS41F_EXPERTS_PER_ROW * \
	 (DS41F_EXPERT_INNER / LINEAR_VALUES_PER_BLOCK) * sizeof(uint32_t))
#define DS41F_FUSED_SCRATCH_BYTES \
	(DS41F_SITE_FRONT_SCRATCH_BYTES + DS41F_PRODUCER_ARRIVALS_BYTES)
#define DS41F_ENGRAM_WIDTH \
	(DS41F_ENGRAM_ROWS_PER_TOKEN * ENGRAM_VALUE_BYTES_VALUE)
#define DS41F_BF16 LINEAR_BFLOAT16_BYTES
#define DS41F_F32 LINEAR_SUM_BYTES
#define DS41F_WALK_LAYERS \
	43u /* the 40 layers and the draft's stages, layers 40 to 42 */
#define DS41F_DRAFT_LAYERS \
	0x70000000000ull /* bit l for each draft stage l (REF:1213-1214) */
#define DS41F_DRAFT_TAP_LAYERS \
	0xE000000000ull /* layers 37 to 39, whose inputs the draft taps (CFG:9) */
#define DS41F_DRAFT_EMBEDDING_BYTES \
	((uint64_t)DS41F_VOCABULARY_VALUE * DS41F_HIDDEN_ROW_BYTES)
#define DS41F_EMBEDDING_COPY_PIECE_BYTES \
	(16ull << 20u) /* ours: what the draft's copy stages at a time */

/* The tensors of a layer, of a routed expert (in the order of an expert table's row) and outside the layers. */
enum ds41f_layer_tensor {
	DS41F_TENSOR_ATTENTION_NORM,
	DS41F_TENSOR_FFN_NORM,
	DS41F_TENSOR_ATTENTION_MIX,
	DS41F_TENSOR_ATTENTION_MIX_BASE,
	DS41F_TENSOR_ATTENTION_MIX_SCALE,
	DS41F_TENSOR_FFN_MIX,
	DS41F_TENSOR_FFN_MIX_BASE,
	DS41F_TENSOR_FFN_MIX_SCALE,
	DS41F_TENSOR_SINK,
	DS41F_TENSOR_QUERY_DOWN,
	DS41F_TENSOR_QUERY_NORM,
	DS41F_TENSOR_QUERY_UP,
	DS41F_TENSOR_WINDOW,
	DS41F_TENSOR_WINDOW_NORM,
	DS41F_TENSOR_OUTPUT_GROUPS,
	DS41F_TENSOR_OUTPUT,
	DS41F_TENSOR_ROUTER,
	DS41F_TENSOR_ROUTER_BIAS,
	DS41F_TENSOR_SHARED_GATE,
	DS41F_TENSOR_SHARED_UP,
	DS41F_TENSOR_SHARED_DOWN,
	DS41F_TENSOR_COMPRESSOR,
	DS41F_TENSOR_COMPRESSOR_GATE,
	DS41F_TENSOR_COMPRESSOR_NORM,
	DS41F_TENSOR_KEY,
	DS41F_TENSOR_KEY_NORM,
	DS41F_TENSOR_INDEX_QUERY_UP,
	DS41F_TENSOR_INDEX_WEIGHTS,
	DS41F_TENSOR_ENGRAM,
	DS41F_TENSOR_ENGRAM_QUERY,
	DS41F_TENSOR_ENGRAM_KEY,
	DS41F_TENSOR_LAYER_COUNT
};
enum ds41f_expert_tensor {
	DS41F_TENSOR_GATE,
	DS41F_TENSOR_UP,
	DS41F_TENSOR_DOWN,
	DS41F_TENSOR_EXPERT_COUNT
};
enum ds41f_global_tensor {
	DS41F_TENSOR_EMBEDDING,
	DS41F_TENSOR_NORM,
	DS41F_TENSOR_HEAD,
	DS41F_TENSOR_GLOBALS
};
/* The draft's tensors outside its stages: main_proj and main_norm (stage 0's), its final norm, the Markov head's two
 * matrices and the confidence projection (stage 2's). */
enum ds41f_draft_global {
	DS41F_TENSOR_DRAFT_MAIN_PROJECTION,
	DS41F_TENSOR_DRAFT_MAIN_NORM,
	DS41F_TENSOR_DRAFT_NORM,
	DS41F_TENSOR_DRAFT_MARKOV_EMBEDDING,
	DS41F_TENSOR_DRAFT_MARKOV_HEAD,
	DS41F_TENSOR_DRAFT_CONFIDENCE,
	DS41F_TENSOR_DRAFT_GLOBALS
};

/* Which cards hold an array: those that run the embedding, a layer or the head; a layer; an index layer; a source;
 * a 2-to-1 source; an n-gram layer; the head (the card that samples); the candidate source or a layer that reads its
 * candidates; the draft (the result card, with the draft admitted; with draft_layers every rank of a TP stage); the
 * draft's copy of the embedding (the result card). */
enum ds41f_holder {
	DS41F_HOLDER_TRUNK,
	DS41F_HOLDER_LAYER,
	DS41F_HOLDER_INDEX,
	DS41F_HOLDER_SOURCE,
	DS41F_HOLDER_POOLED,
	DS41F_HOLDER_ENGRAM,
	DS41F_HOLDER_RESULT,
	DS41F_HOLDER_CANDIDATES,
	DS41F_HOLDER_DRAFT,
	DS41F_HOLDER_DRAFT_COPY
};

/* Which rows an array has: a chunk's, the plan's row capacity (what a wide step carries whole: the streams and what a
 * launch over the whole chunk reads or writes); a span's, DS41F_SPAN_ROWS at most (what a launch over a span of
 * a chunk's slabs writes for those slabs to read at their rows before the next span's launch); or a slab's, the rows a
 * thin launch takes (at most LINEAR_MAXIMUM_ROWS: scratch the slabs of a wide step reuse one after another). */
enum ds41f_rows {
	DS41F_ROWS_CHUNK,
	DS41F_ROWS_SPAN,
	DS41F_ROWS_SLAB,
	DS41F_ROWS_DRAFT
}; /* configured draft block, never a chunk */

/* A slab's rows under a row capacity: the most rows a thin step carries. */
static inline uint32_t ds41f_slab_rows(uint32_t row_capacity)
{
	return row_capacity < LINEAR_MAXIMUM_ROWS ? row_capacity :
						    LINEAR_MAXIMUM_ROWS;
}

/* A span's rows under a row capacity. */
static inline uint32_t ds41f_span_rows(uint32_t row_capacity)
{
	return row_capacity < DS41F_SPAN_ROWS ? row_capacity : DS41F_SPAN_ROWS;
}

/* X(array, bytes a row, bytes besides, holder, rows). Apart: SESSIONS has STEP_SESSION_BYTES a session slot,
 * INDEX_SCORES FP16 [row][the most entries an index layer of the card scores], plus the split selection annex
 * described in ds41f_index_scan.h where its capacity is eligible; ENGRAM_ROWS uint8 [table of the card]
 * [row][24][ENGRAM_ROW_BYTES] (model_seam.h); JOBS the more of a thin step's 6 a row and the wide select's
 * table for the row capacity, ROUTER_COUNTS and ROUTER_TILES the wide select's for it (ds41f_router.h); VIEWS a step
 * view's block and rows (model_seam.h) for each slab of a chunk, none where a chunk is one slab; COMPRESSOR_KV half its
 * row on a card that holds no 2-to-1 source. The output of a product is named for it. An array that a chunk's launches
 * read or write across its slabs is a chunk's: what a launch over the chunk (a piece of it: linear.h's and
 * dense_linear.h's wide launches) reads or writes, which its slabs' launches write or read at their rows (the query
 * rank, the window input, the heads and the indexer's, the attention output, the groups, the sublayer output, the
 * shared expert's gate and up, the n-gram values and projection, the indexer's weights, the compressor's products;
 * model_v41_program.c's chunk walk), and what a full program reads across its slabs: QUERY_NORMED from layer 20's
 * front to the rest of it, CANDIDATES from layer 20 to the candidate readers. The queries and the index queries are a
 * span's, the core's scratch a slab's, in a TP plan a chunk's (ds41f_array_rows). A product's rows lie packed at its
 * output's row: COMPRESSOR_KV of a 1-to-1 source, bfloat16, at half its
 * row. The draft's (dspark_v1/DESIGN.md 5.2): ROUND, the round record (step_records.h)
 * right behind RESULT, so that one download takes both; DRAFT_EMBEDDING, the copy of embed.weight; DRAFT_TAPS (model
 * layout's draft taps) and what the context makes of them, main_proj's quantized rows (a piece's packed at their
 * width), its output, main_norm's and each stage's wkv product, a chunk's; the step area the draft's launches read,
 * its one-row views, logits, one-row result areas, pre-norm collapse and Markov rows of the block's positions. */
#define DS41F_OFFICIAL_SCRATCH_BYTES (128ull * 1024ull)
#define DS41F_ARRAYS(X)                                                               \
	X(FREQUENCIES, 0, sizeof(float[DS41F_THETAS][DS41F_FREQUENCIES]),             \
	  DS41F_HOLDER_LAYER, DS41F_ROWS_CHUNK)                                       \
	X(STEP, STEP_ROW_BYTES_VALUE, STEP_BLOCK_BYTES_VALUE,                         \
	  DS41F_HOLDER_TRUNK, DS41F_ROWS_CHUNK)                                       \
	X(VIEWS, 0, 0, DS41F_HOLDER_TRUNK, DS41F_ROWS_CHUNK)                          \
	X(RESULT, STEP_ROW_RESULT_BYTES, STEP_RESULT_BYTES,                           \
	  DS41F_HOLDER_RESULT, DS41F_ROWS_SLAB)                                       \
	X(ROUND, 0, STEP_ROUND_BYTES, DS41F_HOLDER_DRAFT, DS41F_ROWS_CHUNK)           \
	X(SESSIONS, 0, 0, DS41F_HOLDER_RESULT, DS41F_ROWS_CHUNK)                      \
	X(STREAMS, DS41F_STREAMS_ROW_BYTES, 0, DS41F_HOLDER_TRUNK,                    \
	  DS41F_ROWS_CHUNK)                                                           \
	X(CARRIED_PRE, DS41F_SITE_ROW_BYTES, 0, DS41F_HOLDER_TRUNK,                   \
	  DS41F_ROWS_CHUNK)                                                           \
	X(HIDDEN, DS41F_HIDDEN_ROW_BYTES, 0, DS41F_HOLDER_TRUNK,                      \
	  DS41F_ROWS_CHUNK) /* a site's normed input */                               \
	X(ROTATION, DS41F_ROTATION_ROW_BYTES, 0, DS41F_HOLDER_LAYER,                  \
	  DS41F_ROWS_CHUNK)                                                           \
	X(MIXES, DS41F_MIXES_ROW_BYTES, 0, DS41F_HOLDER_LAYER,                        \
	  DS41F_ROWS_CHUNK)                                                           \
	X(POST, DS41F_SITE_ROW_BYTES, 0, DS41F_HOLDER_LAYER, DS41F_ROWS_CHUNK)        \
	X(COMB, DS41F_COMB_ROW_BYTES, 0, DS41F_HOLDER_LAYER, DS41F_ROWS_CHUNK)        \
	X(ROW_CODES, DS41F_WIDEST_ROW, 0, DS41F_HOLDER_LAYER,                         \
	  DS41F_ROWS_CHUNK) /* every quantized row: linear.h */                       \
	X(ROW_EXPONENTS, DS41F_WIDEST_ROW / LINEAR_VALUES_PER_BLOCK, 0,               \
	  DS41F_HOLDER_LAYER, DS41F_ROWS_CHUNK)                                       \
	X(ROW_OPERANDS, DS41F_WIDEST_ROW *DS41F_BF16, 0, DS41F_HOLDER_LAYER,          \
	  DS41F_ROWS_CHUNK)                                                           \
	X(QUERY_RANK, DS41F_QUERY_RANK *DS41F_BF16, 0, DS41F_HOLDER_LAYER,            \
	  DS41F_ROWS_CHUNK) /* wq_a */                                                \
	X(WINDOW_INPUT, DS41F_HEAD_DIMS *DS41F_BF16, 0, DS41F_HOLDER_LAYER,           \
	  DS41F_ROWS_CHUNK) /* wkv */                                                 \
	X(QUERY_NORMED, DS41F_QUERY_RANK *DS41F_BF16, 0, DS41F_HOLDER_LAYER,          \
	  DS41F_ROWS_CHUNK) /* qr */                                                  \
	X(HEADS, DS41F_HEADS_ROW_BYTES, 0, DS41F_HOLDER_LAYER,                        \
	  DS41F_ROWS_CHUNK) /* wq_b */                                                \
	X(QUERIES, DS41F_HEADS_ROW_BYTES, 0, DS41F_HOLDER_LAYER,                      \
	  DS41F_ROWS_CHUNK) /* a span at a time; chunk_core's */                      \
	X(ATTENTION_OUTPUT, DS41F_HEADS_ROW_BYTES, 0, DS41F_HOLDER_LAYER,             \
	  DS41F_ROWS_CHUNK)                                                           \
	X(ATTENTION_SCORES,                                                           \
	  DS41F_HEADS_VALUE *(DS41F_WINDOW + DS41F_SELECTED) * DS41F_F32, 0,          \
	  DS41F_HOLDER_LAYER,                                                         \
	  DS41F_ROWS_CHUNK) /* the core's scratch: a slab's, or chunk_core's chunk */ \
	X(GROUPS, DS41F_WIDEST_ROW *DS41F_BF16, 0, DS41F_HOLDER_LAYER,                \
	  DS41F_ROWS_CHUNK) /* wo_a */                                                \
	X(SUBLAYER_OUTPUT, DS41F_HIDDEN_ROW_BYTES, 0, DS41F_HOLDER_LAYER,             \
	  DS41F_ROWS_CHUNK) /* wo_b */                                                \
	X(ROUTER_SCORES, DS41F_ROUTER_ROW_BYTES, 0, DS41F_HOLDER_LAYER,               \
	  DS41F_ROWS_CHUNK)                                                           \
	X(AFFINITIES, DS41F_ROUTER_ROW_BYTES, 0, DS41F_HOLDER_LAYER,                  \
	  DS41F_ROWS_CHUNK) /* the router's, of each expert */                        \
	X(ROUTER_PICKS, DS41F_EXPERTS_PER_ROW * sizeof(struct router_pick), 0,        \
	  DS41F_HOLDER_LAYER, DS41F_ROWS_CHUNK)                                       \
	X(ROUTER_COUNTS, 0, 0, DS41F_HOLDER_LAYER, DS41F_ROWS_CHUNK)                  \
	X(ROUTER_PAIRS, DS41F_EXPERTS_PER_ROW * sizeof(struct wide_pair), 0,          \
	  DS41F_HOLDER_LAYER, DS41F_ROWS_CHUNK)                                       \
	X(ROUTER_RUNS, 0, DS41F_EXPERTS * sizeof(struct wide_run),                    \
	  DS41F_HOLDER_LAYER, DS41F_ROWS_CHUNK)                                       \
	X(ROUTER_TILES, 0, 0, DS41F_HOLDER_LAYER, DS41F_ROWS_CHUNK)                   \
	X(JOBS, DS41F_EXPERTS_PER_ROW *EXPERT_JOB_BYTES, 0,                           \
	  DS41F_HOLDER_LAYER, DS41F_ROWS_SLAB)                                        \
	X(PAIR_HIDDEN, DS41F_PAIR_WIDTH *DS41F_BF16, 0, DS41F_HOLDER_LAYER,           \
	  DS41F_ROWS_CHUNK) /* the shared expert's too */                             \
	X(PAIR_CODES, DS41F_PAIR_WIDTH, 0, DS41F_HOLDER_LAYER,                        \
	  DS41F_ROWS_CHUNK)                                                           \
	X(PAIR_EXPONENTS, DS41F_PAIR_WIDTH / LINEAR_VALUES_PER_BLOCK, 0,              \
	  DS41F_HOLDER_LAYER, DS41F_ROWS_CHUNK)                                       \
	X(PAIR_OPERANDS, DS41F_PAIR_WIDTH *DS41F_BF16, 0, DS41F_HOLDER_LAYER,         \
	  DS41F_ROWS_CHUNK)                                                           \
	X(PAIR_OUTPUT, DS41F_EXPERTS_PER_ROW *DS41F_HIDDEN_ROW_BYTES, 0,              \
	  DS41F_HOLDER_LAYER, DS41F_ROWS_CHUNK)                                       \
	X(ROW_SUMS, DS41F_HIDDEN *DS41F_F32, 0, DS41F_HOLDER_LAYER,                   \
	  DS41F_ROWS_CHUNK)                                                           \
	X(SHARED_GATE, DS41F_EXPERT_INNER *DS41F_BF16, 0, DS41F_HOLDER_LAYER,         \
	  DS41F_ROWS_CHUNK)                                                           \
	X(SHARED_UP, DS41F_EXPERT_INNER *DS41F_BF16, 0, DS41F_HOLDER_LAYER,           \
	  DS41F_ROWS_CHUNK)                                                           \
	X(SHARED_OUTPUT, DS41F_HIDDEN_ROW_BYTES, 0, DS41F_HOLDER_LAYER,               \
	  DS41F_ROWS_CHUNK)                                                           \
	X(INDEX_HEADS, DS41F_INDEX_QUERIES_ROW_BYTES, 0, DS41F_HOLDER_INDEX,          \
	  DS41F_ROWS_CHUNK) /* the indexer's wq_b */                                  \
	X(INDEX_QUERIES, DS41F_INDEX_QUERIES_ROW_BYTES, 0, DS41F_HOLDER_INDEX,        \
	  DS41F_ROWS_CHUNK)                                                           \
	X(INDEX_WEIGHTS, DS41F_INDEX_WEIGHTS_ROW_BYTES, 0, DS41F_HOLDER_INDEX,        \
	  DS41F_ROWS_CHUNK)                                                           \
	X(INDEX_SCORES, 0, 0, DS41F_HOLDER_INDEX, DS41F_ROWS_SLAB)                    \
	X(CANDIDATES, DS41F_CANDIDATES_ROW_BYTES, 0, DS41F_HOLDER_CANDIDATES,         \
	  DS41F_ROWS_CHUNK) /* uint32 [row][2,048] */                                 \
	X(COMPRESSOR_KV, DS41F_HEAD_DIMS *DS41F_F32, 0, DS41F_HOLDER_SOURCE,          \
	  DS41F_ROWS_CHUNK)                                                           \
	X(COMPRESSOR_SCORE, DS41F_HEAD_DIMS *DS41F_F32, 0,                            \
	  DS41F_HOLDER_POOLED, DS41F_ROWS_CHUNK)                                      \
	X(LATENT, DS41F_HEAD_DIMS *DS41F_BF16, 0, DS41F_HOLDER_SOURCE,                \
	  DS41F_ROWS_CHUNK) /* bfloat16, for wk */                                    \
	X(KEY_INPUT, DS41F_KEY_BYTES, 0, DS41F_HOLDER_SOURCE,                         \
	  DS41F_ROWS_CHUNK) /* wk */                                                  \
	X(ENGRAM_ROWS, 0, 0, DS41F_HOLDER_ENGRAM, DS41F_ROWS_CHUNK)                   \
	X(ENGRAM_VALUES, DS41F_ENGRAM_WIDTH *DS41F_BF16, 0,                           \
	  DS41F_HOLDER_ENGRAM, DS41F_ROWS_CHUNK)                                      \
	X(ENGRAM_PROJECTION, (DS41F_STREAMS + 1u) * DS41F_HIDDEN_ROW_BYTES, 0,        \
	  DS41F_HOLDER_ENGRAM, DS41F_ROWS_CHUNK)                                      \
	X(LOGITS, DS41F_LOGITS_ROW_BYTES, 0, DS41F_HOLDER_RESULT,                     \
	  DS41F_ROWS_SLAB)                                                            \
	X(SAMPLER_WEIGHTS, DS41F_LOGITS_ROW_BYTES, 0, DS41F_HOLDER_RESULT,            \
	  DS41F_ROWS_SLAB)                                                            \
	X(DRAFT_EMBEDDING, 0, DS41F_DRAFT_EMBEDDING_BYTES,                            \
	  DS41F_HOLDER_DRAFT_COPY, DS41F_ROWS_CHUNK)                                  \
	X(DRAFT_TAPS, DS41F_DRAFT_TAP_ROW_BYTES, 0, DS41F_HOLDER_DRAFT,               \
	  DS41F_ROWS_CHUNK)                                                           \
	X(DRAFT_TAP_CODES, DS41F_DRAFT_FEATURES, 0, DS41F_HOLDER_DRAFT,               \
	  DS41F_ROWS_CHUNK)                                                           \
	X(DRAFT_TAP_EXPONENTS, DS41F_DRAFT_FEATURES / LINEAR_VALUES_PER_BLOCK,        \
	  0, DS41F_HOLDER_DRAFT, DS41F_ROWS_CHUNK)                                    \
	X(DRAFT_TAP_OPERANDS, DS41F_DRAFT_FEATURES *DS41F_BF16, 0,                    \
	  DS41F_HOLDER_DRAFT, DS41F_ROWS_CHUNK)                                       \
	X(DRAFT_MAIN, DS41F_HIDDEN_ROW_BYTES, 0, DS41F_HOLDER_DRAFT,                  \
	  DS41F_ROWS_CHUNK) /* main_proj */                                           \
	X(DRAFT_MAIN_X, DS41F_HIDDEN_ROW_BYTES, 0, DS41F_HOLDER_DRAFT,                \
	  DS41F_ROWS_CHUNK) /* main_norm */                                           \
	X(DRAFT_WINDOW_0, DS41F_HEAD_DIMS *DS41F_BF16, 0, DS41F_HOLDER_DRAFT,         \
	  DS41F_ROWS_CHUNK) /* mtp.0.attn.wkv */                                      \
	X(DRAFT_WINDOW_1, DS41F_HEAD_DIMS *DS41F_BF16, 0, DS41F_HOLDER_DRAFT,         \
	  DS41F_ROWS_CHUNK)                                                           \
	X(DRAFT_WINDOW_2, DS41F_HEAD_DIMS *DS41F_BF16, 0, DS41F_HOLDER_DRAFT,         \
	  DS41F_ROWS_CHUNK)                                                           \
	X(DRAFT_STEP, STEP_ROW_BYTES_VALUE, STEP_BLOCK_BYTES_VALUE,                   \
	  DS41F_HOLDER_DRAFT, DS41F_ROWS_DRAFT)                                       \
	X(DRAFT_VIEWS, STEP_DRAFT_VIEW_BYTES, 0, DS41F_HOLDER_DRAFT,                  \
	  DS41F_ROWS_DRAFT)                                                           \
	X(DRAFT_LOGITS, DS41F_LOGITS_ROW_BYTES, 0, DS41F_HOLDER_DRAFT,                \
	  DS41F_ROWS_DRAFT)                                                           \
	X(DRAFT_RESULTS, STEP_DRAFT_RESULT_BYTES, 0, DS41F_HOLDER_DRAFT,              \
	  DS41F_ROWS_DRAFT)                                                           \
	X(DRAFT_COLLAPSE, DS41F_HIDDEN_ROW_BYTES, 0, DS41F_HOLDER_DRAFT,              \
	  DS41F_ROWS_DRAFT)                                                           \
	X(DRAFT_MARKOV, DS41F_MARKOV_RANK *DS41F_BF16, 0, DS41F_HOLDER_DRAFT,         \
	  DS41F_ROWS_DRAFT)
#define DS41F_ENUMERATOR(array, ...) DS41F_ARRAY_##array,
enum ds41f_array { DS41F_ARRAYS(DS41F_ENUMERATOR) DS41F_ARRAY_COUNT };
#undef DS41F_ENUMERATOR

/* The rows an array has in a plan (enum ds41f_rows): kind, its DS41F_ARRAYS entry's, but in a PP plan (no
 * chunk_attention_arrays: neither TP's chunk_core nor its queries_piece runs) the attention core's scratch is a slab's,
 * as each slab's core takes it in turn, and the queries and the index queries are a span's, as the chunk walk writes
 * each span's ahead of its first slab. The plan sizes an array by it and the programs lay a view's rows into the
 * array by it. */
static inline enum ds41f_rows ds41f_array_rows(bool chunk_attention_arrays,
					       enum ds41f_array array,
					       enum ds41f_rows kind)
{
	if (chunk_attention_arrays)
		return kind;
	if (array == DS41F_ARRAY_ATTENTION_SCORES)
		return DS41F_ROWS_SLAB;
	const bool queries = array == DS41F_ARRAY_QUERIES ||
			     array == DS41F_ARRAY_INDEX_QUERIES;
	return queries ? DS41F_ROWS_SPAN : kind;
}

/* Where a tensor lies: the arena of card, from stored.byte_offset; a vector has input_size 1. All zero for a tensor
 * of a layer that has none. */
struct ds41f_tensor {
	struct linear_range stored;
	struct matrix_shape_record shape;
	enum weight_format_code weight_format;
	uint32_t card;
	int32_t fold_reference; /* 0: reference order; set at relaxed load after scale checks */
};

/* A card: per walk layer the offsets of what model_v41.h's contract lists, the expert table first. */
struct ds41f_card {
	uint64_t array_offset[DS41F_ARRAY_COUNT];
	uint64_t expert_table_offset[DS41F_WALK_LAYERS];
	uint64_t selection_offset[DS41F_WALK_LAYERS]; /* int32 [row][512] */
	uint64_t ring_offset
		[DS41F_WALK_LAYERS]; /* DS41F_RING_SESSION_BYTES a session */
	uint64_t checkpoint_offset
		[DS41F_WALK_LAYERS]; /* DS41F_CHECKPOINT_SESSION_BYTES a session */
	uint64_t compressor_offset
		[DS41F_WALK_LAYERS]; /* DS41F_COMPRESSOR_SESSION_BYTES a session */
	uint64_t pool_offset[DS41F_WALK_LAYERS]; /* pool_session_bytes a session */
	uint64_t key_tiles_offset
		[DS41F_WALK_LAYERS]; /* key_session_bytes a session */
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
	uint32_t holders; /* bit h: a holder h (enum ds41f_holder) */
	uint32_t engram_tables; /* bit t: it runs the n-gram layer of table t */
	uint32_t slot_count;
	uint32_t reserved_zero;
};

#define DS41F_HEAD_RANKS 4u
#define DS41F_HEAD_CARDS (DS41F_HEAD_RANKS + 1u)
#define DS41F_HEAD_SHARD_ROWS (DS41F_VOCABULARY_VALUE / DS41F_HEAD_RANKS)
#define DS41F_HEAD_CONTROL_ROW_BYTES 16u
#define DS41F_HEAD_BANKS 8u
#define DS41F_HEAD_MODE_LEASED 1u
#define DS41F_HEAD_MODE_BANKED 2u

/* Opt-in original PP5 target head. All ranges append behind ordinary/vision state. A PRO lease spanning
 * its first target visit through its last sampler/draft visit owns these shared planes across flights. */
struct ds41f_original_head_card {
	struct linear_range weights, hidden, logits;
	struct linear_range peer_words[DS41F_HEAD_RANKS];
	struct linear_range control_source, control_input[DS41F_HEAD_RANKS];
	struct linear_range pilot_step, pilot_source, pilot_result;
	struct linear_range bank_hidden[DS41F_HEAD_BANKS - 1u],
		bank_logits[DS41F_HEAD_BANKS - 1u];
};
struct ds41f_original_head {
	struct ds41f_original_head_card cards[DS41F_HEAD_CARDS];
	bool enabled, connected, initialised;
	uint32_t bank_count; /* 0: original leased mode; 1..8: immutable per-flight banks */
};

/* The walk layers whose state the card holds: its layers, and on the draft's card the draft's stages. */
static inline uint64_t ds41f_walk_layers(const struct ds41f_card *card)
{
	return card->layers | ((card->holders >> DS41F_HOLDER_DRAFT & 1u) != 0 ?
				       DS41F_DRAFT_LAYERS :
				       0);
}

/* Per walk layer: a draft stage's members are those of a layer that has no source, index or n-gram table. */
struct ds41f_plan {
	struct program_limits limits;
	struct ds41f_card cards[BACKEND_MAXIMUM_CARDS];
	struct ds41f_tensor layer_tensors[DS41F_WALK_LAYERS]
					 [DS41F_TENSOR_LAYER_COUNT];
	struct ds41f_tensor globals[DS41F_TENSOR_GLOBALS];
	struct ds41f_tensor draft_globals
		[DS41F_TENSOR_DRAFT_GLOBALS]; /* all zero without the draft */
	struct linear_range experts
		[DS41F_WALK_LAYERS]; /* holds every routed expert matrix of the layer */
	uint64_t expert_tables[DS41F_WALK_LAYERS][DS41F_EXPERTS]
			      [DS41F_EXPERT_MATRICES]; /* images' */
	uint64_t host_expert_offsets[DS41F_WALK_LAYERS]
				    [DS41F_EXPERTS]; /* in the host arena, by tag */
	uint64_t expert_matrix_offsets
		[DS41F_EXPERT_MATRICES]; /* inside a host expert's bytes */
	uint64_t expert_byte_count; /* of a host expert; 0 for none */
	uint64_t slot_byte_count;
	uint32_t host_expert_count[DS41F_WALK_LAYERS];
	uint32_t exl3_bits
		[DS41F_WALK_LAYERS]; /* zero for original-format routed weights */
	bool expert_folds_admitted
		[DS41F_WALK_LAYERS]; /* every resident expert scale passed at relaxed load */
	uint64_t pool_session_bytes[DS41F_WALK_LAYERS]; /* of a source; else 0 */
	uint64_t key_session_bytes[DS41F_WALK_LAYERS];
	uint32_t layer_card
		[DS41F_WALK_LAYERS]; /* a draft stage's: the result card */
	uint32_t embedding_card;
	uint32_t result_card;
	uint32_t card_count;
	uint32_t reserved_zero;
	struct ds41f_model_vision vision;
	struct ds41f_original_head original_head;
	/* every TP plan's (its stages', so its canonical's and views'): the arrays ds41f_array_rows narrows hold a
	 * chunk's rows, as TP's chunk_core and queries_piece read them; false in a PP plan, where a card's queries and
	 * core scratch are one span's and one slab's, reused in program order: work that runs two parts of one PP
	 * chunk at once (as TP's split_overlap halves) must give each part its own, as rows_select checks only the
	 * bounds */
	bool chunk_attention_arrays;
};

/* Plan before card open; initialise after ordinary arena_initialise, which zeros appended ranges too.
 * No operation occurs with the default-disabled plan. Failure never admits runtime; uncertain copies retain
 * their staging. Drain returns every open group member as retained if any member's completion is unproven. */
MUST_CHECK result_status
ds41f_original_head_plan(struct ds41f_plan *plan,
			 const struct card_description_record *descriptions);
MUST_CHECK result_status ds41f_original_head_banks_plan(struct ds41f_plan *plan,
							uint32_t bank_count);
enum ds41f_head_launch_kind {
	DS41F_HEAD_BROADCAST,
	DS41F_HEAD_DENSE,
	DS41F_HEAD_GATHER,
	DS41F_HEAD_SAMPLE,
	DS41F_HEAD_LAUNCH_KINDS
};
MUST_CHECK result_status ds41f_original_head_launch(
	const struct ds41f_plan *plan, const struct backend_operations *backend,
	struct device_card *const *cards, enum ds41f_head_launch_kind kind,
	uint32_t rank, uint32_t bank, uint32_t rows,
	struct kernel_launch *launch_out);
struct ds41f_head_program_banks;
struct step_flight_programs;
MUST_CHECK result_status ds41f_original_head_banks_create(
	const struct ds41f_plan *plan, const struct backend_operations *backend,
	struct device_card *const *cards,
	const struct execution_program *const *programs,
	const struct execution_program *const *image_programs,
	uint32_t program_count, struct ds41f_head_program_banks **banks_out);
const struct step_flight_programs *
ds41f_original_head_bank_tables(const struct ds41f_head_program_banks *banks);
const struct visit_lease *
ds41f_original_head_owner_guard(const struct ds41f_head_program_banks *banks);
struct ds41f_head_bank_memory {
	size_t heap_bytes, source_launch_bytes, source_lease_count,
		program_metadata_bytes;
	uint32_t thin_programs, program_count, bank_count;
};
struct ds41f_head_bank_memory
ds41f_original_head_banks_memory(const struct ds41f_head_program_banks *banks);
void ds41f_original_head_banks_destroy(struct ds41f_head_program_banks *banks);
/* Cold address attachment only. An attempted first connection sets connected even when that call fails,
 * making whole-group retirement mandatory. No caller may retry this plan or start runtime on failure. */
MUST_CHECK result_status ds41f_original_head_connect(
	struct ds41f_plan *plan, const struct backend_operations *backend,
	struct device_card *const *cards);
MUST_CHECK result_status ds41f_original_head_initialise(
	struct ds41f_plan *plan, const struct pack_record *pack,
	const struct storage_operations_record *storage,
	const struct storage_reader *reader,
	const struct backend_operations *backend,
	struct device_card *const *cards, uint64_t wait_limit_nanoseconds);
MUST_CHECK result_status ds41f_original_head_drain(
	const struct backend_operations *backend,
	struct device_card *const *cards, uint64_t wait_limit_nanoseconds,
	uint32_t *retained_out);

MUST_CHECK result_status
ds41f_describe(const struct pack_record *pack,
	       struct model_description_record *description_out);
MUST_CHECK result_status
ds41f_state_byte_count(const struct pack_record *pack,
		       const struct placement_record *placement,
		       const struct program_limits *limits, uint32_t card_index,
		       uint64_t *byte_count_out);
MUST_CHECK result_status ds41f_plan_create(const struct pack_record *pack,
					   const struct load_layout *layout,
					   const struct program_limits *limits,
					   struct ds41f_plan **plan_out);
void ds41f_plan_destroy(struct ds41f_plan *plan);
MUST_CHECK result_status ds41f_arena_byte_count(const struct ds41f_plan *plan,
						uint32_t card_index,
						uint64_t *byte_count_out);
MUST_CHECK result_status ds41f_arena_initialise(
	struct ds41f_plan *plan, const struct backend_operations *backend,
	struct device_card *const *cards, uint64_t wait_limit_nanoseconds);

/* Image-capable counterpart to the ordinary program, prepared only after all text programs. It retains
 * their state metadata, while image-row embedding/routing/gating uses the optional tower's appended state. */
MUST_CHECK result_status ds41f_image_program_prepare(
	const struct ds41f_plan *plan, const struct backend_operations *backend,
	struct device_card *const *cards,
	const struct card_description_record *descriptions,
	enum program_kind kind, uint32_t row_count,
	struct execution_program **program_out);

/* The words of both thetas (ds41f_geometry.h), bit for bit those REF:376-387 forms on the card: the reference's own,
 * a table in model_v41.c. */
void ds41f_frequency_words(float words_out[DS41F_THETAS][DS41F_FREQUENCIES]);

#endif
