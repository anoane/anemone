/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * builder.h: the builder's interface to the TP program units and the model root (split from
 * model_v41_program.c
 * by whole items).
 */
#ifndef DS41F_BUILDER_H
#define DS41F_BUILDER_H

#include "common/checked_size.h"
#include "models/deepseek_v41_flash/plan/model_v41.h"
#include "models/deepseek_v41_flash/tensor_parallel/ds41f_tp_layer.h"
#include "models/deepseek_v41_flash/tensor_parallel/ds41f_tp_program.h"
#include "models/model_registry.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "models/deepseek_v41_flash/attention/ds41f_attention_core.h"
#include "models/deepseek_v41_flash/attention/ds41f_attention_output.h"
#include "models/deepseek_v41_flash/attention/ds41f_attention_state.h"
#include "engine/backend/fill_records.h"
#include "engine/library/bf16_linear/dense_linear.h"
#include "models/deepseek_v41_flash/official/ds41f_official_hc.h"
#include "models/deepseek_v41_flash/official/ds41f_official_reference.h"
#include "engine/library/bf16_linear/dense_stream_records.h"
#include "models/deepseek_v41_flash/draft/ds41f_dspark.h"
#include "models/deepseek_v41_flash/tensor_parallel/ds41f_draft_share_records.h"
#include "models/deepseek_v41_flash/engram/ds41f_engram_rows.h"
#include "engine/backend/copy_records.h"
#include "engine/library/block_scaled_linear/expert_relaxed.h"
#include "engine/library/exl3/exl3.h"
#include "engine/library/exl3/exl3_tc.h"
#include "engine/library/exl3/exl3_tp.h"
#include "engine/library/moe/expert_producer.h"
#include "common/hooks.h"
#include "models/deepseek_v41_flash/hyper_connection/ds41f_hyper_connection.h"
#include "engine/library/tensor_parallel/peer_row_copy.h"
#include "models/deepseek_v41_flash/indexer/ds41f_index_scan.h"
#include "models/deepseek_v41_flash/vision/ds41f_image_routing.h"
#include "models/deepseek_v41_flash/moe/ds41f_router.h"
#include "engine/library/block_scaled_linear/relaxed_linear.h"
#include "engine/library/sampling/sampler.h"
#include "engine/library/sampling/sampler_shard_records.h"
#include "models/deepseek_v41_flash/hyper_connection/ds41f_site_front.h"
#include "engine/library/tensor_parallel/tp_join.h"
#include "engine/library/moe/ep_filter.h"
#include "engine/library/rings/exact_ring_records.h"
#include "engine/library/wire_codecs/quantized_wire_records.h"
#include "engine/library/tensor_parallel/tp_exchange.h"
#include "engine/library/collectives/ordered_push_parallel.h"
#include "engine/library/prepare_checks.h"

#include "engine/library/bf16_linear/head_mma_records.h"
#include "engine/library/bf16_linear/trunk_mma_records.h"

#define DS41F_DRAFT_WIDE_PRODUCTS \
	2u /* batch_draft_wide: the batched draft's products over all its rows too */
#define DS41F_LEGACY_LAYER_MOST_LAUNCHES \
	41u /* layer 14: n-gram 4, attention 23, FFN 14 */
/* A source may replace both legacy selections with five passes; the bound also covers each wide slab. */
#define DS41F_EXL3_ATTENTION_EXTRA_LAUNCHES \
	4u /* BF16 QKV split and separate output post, per prepared slab. */
#define DS41F_EXL3_ROUTED_EXTRA_LAUNCHES \
	4u /* PP loses its native shared join; TP adds two gather-adjacent passes. */
#define DS41F_LAYER_MOST_LAUNCHES                     \
	(DS41F_LEGACY_LAYER_MOST_LAUNCHES +           \
	 2u * (DS41F_INDEX_SPLIT_LAUNCH_COUNT - 1u) + \
	 DS41F_EXL3_ROUTED_EXTRA_LAUNCHES)
#define DS41F_CARD_MOST_LAUNCHES \
	6u /* rotation, optional scratch clear, embedding, final, head, sampler */
#define DS41F_MOST_SLABS       \
	(STEP_WIDE_MOST_ROWS / \
	 LINEAR_MAXIMUM_ROWS) /* of a wide step: its views */
#define DS41F_VIEW_BYTES          \
	(STEP_BLOCK_BYTES_VALUE + \
	 LINEAR_MAXIMUM_ROWS * STEP_ROW_BYTES_VALUE) /* the plan's */
/* embedding, head, draft; head shards' heads a rank; a BATCH program's verifier walks (tp4_v3 found the overflow) */
#define DS41F_MOST_SEGMENTS                                            \
	(BACKEND_MAXIMUM_CARDS + DS41F_TP4_RANKS * DS41F_LAYERS + 4u + \
	 DS41F_TP4_RANKS + DS41F_DRAFT_SHARE_VERIFIERS)
#define DS41F_CONTEXT_PIECE_LAUNCHES \
	6u /* quantize, main projection, norm, quantize, two wkv products */
#define DS41F_DRAFT_POSITION_LAUNCHES \
	2u /* Markov product and sampler for one position */
#define DS41F_DRAFT_FIXED_LAUNCHES \
	6u /* block setup, normalization, head and confidence */
#define DS41F_DRAFT_MOST_LAUNCHES(positions)              \
	(DS41F_DRAFT_STAGES * DS41F_LAYER_MOST_LAUNCHES + \
	 DS41F_DRAFT_POSITION_LAUNCHES * (positions) +    \
	 DS41F_DRAFT_FIXED_LAUNCHES)
#define DS41F_MOST_HOST_EXPERTS (DS41F_LAYERS * DS41F_EXPERTS)
#define DS41F_COPIED_ARRAYS \
	4u /* of a layer: its selection, its pool, its key tiles, its candidates */
#define DS41F_EVERY_COPY 0xFu /* a bit an array of COPIED_ARRAYS */
#define DS41F_FRONT_COPIES \
	0x6u /* the pool's and the key tiles': behind layer 20's front */
#define DS41F_LATER_COPIES (DS41F_EVERY_COPY & ~DS41F_FRONT_COPIES)
#define DS41F_COPIES_TO_A_CARD \
	17u /* 8 index layers' selections, 4 sources' pools and keys, the candidates */
#define DS41F_RESIDUAL_HAND_OFF_ARRAYS 2u /* STREAMS and CARRIED_PRE */
#define DS41F_PP_MOST_HAND_OFFS                                 \
	(DS41F_RESIDUAL_HAND_OFF_ARRAYS * (DS41F_LAYERS + 1u) + \
	 DS41F_COPIES_TO_A_CARD * BACKEND_MAXIMUM_CARDS)
/* Embedding and each prefix layer distribute both residuals to the other three ranks. Include the native-tail
 * boundary and a conservative complete set of source/index/candidate mirrors for that tail. */
#define DS41F_TP4_OWNER_MOST_HAND_OFFS                                      \
	(DS41F_RESIDUAL_HAND_OFF_ARRAYS *                                   \
		 ((DS41F_TP2_PREFIX_LAYERS + 1u) * (DS41F_TP4_RANKS - 1u) + \
		  1u) +                                                     \
	 DS41F_COPIES_TO_A_CARD)
#define DS41F_MOST_HAND_OFFS                                        \
	(DS41F_PP_MOST_HAND_OFFS > DS41F_TP4_OWNER_MOST_HAND_OFFS ? \
		 DS41F_PP_MOST_HAND_OFFS :                          \
		 DS41F_TP4_OWNER_MOST_HAND_OFFS)
#define DS41F_SESSION_KINDS \
	4u /* a layer's session ranges: ring, compressor slots, pool, key tiles */
/* split_overlap's chunk: halves A (0) and B (1); a half of CHUNK_HALVES is the whole chunk, none selected */
#define DS41F_CHUNK_HALVES 2u
#define DS41F_MOST_SESSION_RANGES \
	(DS41F_SESSION_KINDS * DS41F_WALK_LAYERS * BACKEND_MAXIMUM_CARDS)
#define DS41F_MOST_LEASES      \
	((DS41F_LAYERS + 1u) * \
	 BACKEND_MAXIMUM_CARDS) /* a card's groups and candidates */
#define DS41F_OK STATUS_OK
#define DS41F_THEN(call)      \
	(status != DS41F_OK ? \
		 status :     \
		 (call)) /* the next step while every one before passed */

/* A program and all it points at in one allocation, the program first: its address is the allocation's; its launches
 * last, as many as its cards' parts hold (program_prepare). */
struct ds41f_made_program {
	struct execution_program program;
	struct program_card_record cards[BACKEND_MAXIMUM_CARDS];
	struct segment_record segments[DS41F_MOST_SEGMENTS];
	struct hand_off_record hand_offs[DS41F_MOST_HAND_OFFS];
	struct expert_pause expert_pauses[DS41F_LAYERS];
	uint64_t host_expert_offsets[DS41F_MOST_HOST_EXPERTS];
	struct session_range session_ranges[DS41F_MOST_SESSION_RANGES];
	struct window_checkpoint checkpoints[DS41F_WALK_LAYERS];
	struct step_view views[BACKEND_MAXIMUM_CARDS]
			      [DS41F_MOST_SLABS + DS41F_HALF_VIEWS]; /* its slabs,
                                                                     then split_overlap's halves */
	struct visit_lease leases[DS41F_MOST_LEASES];
	struct kernel_launch launches[];
};

/* A program being made; card, arena, the kernels and arrays: of the card whose launches follow, at the rows they take
 * (rows from first_row on, read through view: 0 the step area, 1 + s slab s's view). A thin program's launches take
 * its rows; a wide program's its chunk_rows or a slab of them (slab_count slabs of at most 8). */
struct ds41f_builder {
	const struct ds41f_tp2_layer_plan *tensor_layer;
	/* head_shards (ds41f_tp_stage.h): the card's stage. */
	const struct ds41f_tp2_stage_plan *head_stage;
	const struct ds41f_tp2_stage_plan
		*batch_stage; /* the result card's stage of a BATCH program */
	/* draft_layers: the program's description and tensor plan, for the draft layers' plans */
	const struct ds41f_tp2_program_description *tp_description;
	struct ds41f_tp2_layer_plan *tp_tensor;
	/* batch_draft_rows' walk (draft_slab_select): its slab blocks and collapse rows */
	uint64_t draft_slab_steps, draft_slab_collapse;
	const struct ds41f_plan *plan;
	const struct backend_operations *backend;
	struct device_card *const *cards;
	const struct card_description_record *descriptions;
	struct ds41f_made_program *made;
	struct kernel_launch *launches
		[BACKEND_MAXIMUM_CARDS]; /* each card's part of made->launches */
	struct linear_range arrays[DS41F_ARRAY_COUNT];
	uint64_t arena;
	struct ds41f_official_reference_kernels official;
	uint32_t tensor_slices;
	uint32_t tensor_join_kernel, exact_ring_kernel, unequal_ring_kernel,
		wire_quantize_kernel, wire_decode_kernel;
	uint32_t tensor_push_kernel;
	/* head_shards: the sharded sampler's and head cut's kernels. */
	uint32_t sampler_shard_kernel, head_stream_kernel, draft_share_kernel,
		head_wide_kernel;
	uint32_t head_mma_kernel; /* head_mma: kernel_head_mma's 8-row form */
	uint32_t head_mma_wide_kernels
		[HEAD_MMA_WIDE_FORMS]; /* and its 64-row forms, splits 1, 2, 4 */
	uint32_t trunk_mma_kernels
		[TRUNK_MMA_FORMS]; /* trunk_mma: kernel_trunk_mma's forms, trunk_mma_forms' order, each
                                                    prepared on the card when a product first takes it */
	uint32_t trunk_mma_wide_kernel; /* and its _64 form */
	uint32_t trunk_cone_layer; /* 0, or (a full program's decoder layer) its wide trunks' cone: relaxed_linear.h */
	uint32_t ffn_cone_layer; /* and its FFN's (the hyper site's mix, the router): the layer's own */
	uint32_t program_rows; /* the program's rows (chunk_rows: a half's while one is selected) */
	uint32_t trunk_mma_fused_kernel; /* and that form adding its own splits (trunk_mma_fused) */
	uint32_t batch_grouped_rows; /* a BATCH program's products of at most so many rows by the grouped forms */
	uint32_t batch_tc_paired; /* and from so many rows its TC MoE's short forms paired */
	uint32_t batch_row_sets; /* and above batch_grouped_rows its trunk in row sets of so many */
	uint32_t batch_core_once; /* a BATCH step's core (2: and window writes) once */
	uint32_t front_branch; /* coefficient fronts as side branches: 1 attention, 2 both */
	uint32_t core_heads; /* the reference order's core: at most so many heads a block */
	uint32_t batch_draft_wide; /* a batched draft's sites summed and posted once (1) */
	/* batch_draft_rows' walk (draft_slab_select): its slab count and rows */
	uint32_t draft_slab_count, draft_rows;
	/* draft_bf16_operands (class 80): the carry kernel the draft's quantizations take (draft_swap), else
	 * NO_KERNEL;
     * the kernels table's own quantization meanwhile */
	uint32_t draft_carry_kernel, draft_quantize_kernel;
	/* Grouped forms of 2 + g row groups, in trunk_mma_forms' order; prepared when first taken. */
	uint32_t trunk_mma_grouped_kernels[TRUNK_MMA_GROUPED_FORMS]
					  [TRUNK_MMA_FORMS];
	/* their pair forms (two products a launch, --tp_launch_trims) of TRUNK_MMA_PAIR_FORM: 8 rows, then 16 .. 48 */
	uint32_t trunk_mma_pair_kernels[TRUNK_MMA_GROUPED_FORMS + 1u];
	uint32_t router_mma_kernel,
		router_mma_reduce_kernel; /* router_mma: the split-K router and its reduce */
	uint32_t router_mma_wide_kernel; /* router_mma_wide: its form over 64-row blocks of a wide step */
	uint32_t batch_join_kernel; /* batch_draft_rows, the result card: the row copies into position-major
				     * slots */
	uint32_t settings_push_kernel; /* head_shards, the result card: the sampler settings to every rank's copy */
	uint32_t sampler_call; /* the target sampler shard's use in the step: a BATCH program's slab */
	uint32_t batch_verifier; /* the verifier of batch_draft's walk */
	uint32_t batch_slabs; /* a BATCH program's target slabs, while its draft walks run as a thin program's */
	uint32_t tensor_ep_filter_kernel;
	uint32_t tensor_parallel_kernels[ORDERED_PUSH_PARALLEL_LAUNCHES];
	uint32_t tensor_exchange_kernels
		[TP_EXCHANGE_WIDE_LAUNCHES]; /* wide_exchange's, on shared channels */
	uint32_t tensor_owner_kernels
		[TP_EXCHANGE_WIDE_LAUNCHES]; /* its owner sums' */
	uint32_t tensor_ring_kernels
		[TP_EXCHANGE_WIDE_LAUNCHES]; /* its owner ring sums' (the FFN site's) */
	uint32_t tensor_ce_kernels
		[TP_EXCHANGE_WIDE_LAUNCHES]; /* its copy-engine owner sums' */
	uint32_t tensor_ce_ring_kernels
		[TP_EXCHANGE_WIDE_LAUNCHES]; /* its copy-engine owner ring sums' (FFN) */
	uint32_t launch_capacity[BACKEND_MAXIMUM_CARDS];
	struct linear_kernels_record linear;
	struct relaxed_linear_kernels relaxed;
	struct expert_relaxed_kernels expert;
	struct ds41f_hyper_kernels_record hyper;
	struct dense_linear_kernels dense;
	struct ds41f_official_hc_kernels official_hc;
	struct router_kernels_record router;
	struct ds41f_attention_state_kernels state;
	struct ds41f_index_scan_kernels scan;
	struct ds41f_attention_core_kernels core;
	struct ds41f_engram_rows_kernels engram;
	struct sampler_kernels_record sampler;
	struct ds41f_draft_kernels_record draft;
	struct ds41f_site_front_kernels fronts;
	struct ds41f_vision_kernels vision;
	struct ds41f_image_routing_kernels image_routing;
	uint32_t fill;
	struct ds41f_attention_output_kernels output, draft_output;
	struct expert_fused_kernels fused, draft_fused;

	struct linear_kernels_record draft_linear;
	struct ds41f_attention_core_kernels draft_core;
	uint32_t card, rows, stage, arithmetic_class;
	uint32_t chunk_rows, slab_count, first_row, view;
	uint32_t chunk_first,
		chunk_view; /* split_overlap: a half of the chunk, its first row and its view (else 0, 0) */
	uint32_t half_view_first; /* split_overlap: first half view behind the slabs; 0 without half views. */
	uint32_t half_rows; /* split_overlap: the first half's rows (the program's rows over two, a multiple of 8) */
	uint32_t tcp_emit; /* split_overlap mode 7: next TC prefill call's launches (enum tcp_emit). */
	uint32_t index_row_first; /* index_span: the view's row of the next scan's first row */
	uint32_t ahead_layer
		[BACKEND_MAXIMUM_CARDS]; /* split_overlap's pipeline: 1 + the layer whose half A a card took */
	/* launches_untrimmed off: layer whose wq_b left this rank's heads only in compact rows (no copy into the
     * heads), and those rows, which its queries read (compact_rows 0: none) */
	uint32_t compact_layer, compact_first_row, compact_rows;
	/* and the layer whose core wrote the rank's heads into those rows alone (no copy for wo_a), the first row of
     * the rows wo_a then takes */
	uint32_t core_compact_layer, core_compact_first;
	bool trunk_mma; /* the card's stage asks for trunk_mma */
	/* draft_layers (ds41f_tp_stage.h): the card's stage runs its rank of the draft's layers; the draft tail's
     * first segment made */
	bool draft_layers, draft_tail_begun;
	bool trunk_mma_thin_splits; /* trunk_mma's _64 form split as the 1-8-row form (trunk_mma_thin_splits) */
	bool batch_thin_bits; /* a BATCH program's rows the thin steps' bits (batch_thin_bits) */
	bool draft_bf16_rows; /* the draft layers' attention products of BF16 rows */
	bool ffn_post_deferred; /* batch_draft_wide: ffn_launches leaves its sum and post */
	bool draft_wide_step; /* batch_draft_wide 3: all the rows through the step block behind the slab blocks */
	bool draft_wide_kernels; /* batch_draft_wide 2 over more than 8 rows: card_select prepares the wide forms
				  * too */
	bool ring_column_owners; /* the BF16 ring sums with column owners (ring_column_owners) */
	bool batch_draft; /* a BATCH program's draft walk of verifier batch_verifier (ds41f_dspark_records.h's flags) */
	bool tensor_local_stub;
	bool index_grouped; /* index_span: scan rows_select's rows from index_row_first on (index_scans). */
	bool reads_draft_step;
	bool vision_active;
	/* launches_untrimmed off, a TC-MoE FFN site of graph branches: the combine asked to add the routed sums to the
     * shared expert's rows (exl3_tc_records.h shared_sum), and whether the TC routed launches did */
	bool shared_sum_requested, shared_sum_fused;
	bool core_compact; /* whether core_compact_layer's core wrote the compact rows alone */
};

/* A gathered range: bytes of the rank's rows at own in this card's arena, copied to peers[r] in rank r's arena. */
struct ds41f_gather_range {
	uint64_t own, bytes;
	uint64_t peers[DS41F_TP4_RANKS];
};

/* The steps of a layer's launches before its FFN, in this order, of the rows the next launches take (enum
 * layer_step):
 * an n-gram layer's table rows to values, their quantization and engram.wkv, the gate into the streams (2.1 E1-E4,
 * REF:350-365); the attention site's (2.1 launches 1 to 20, REF:982-986, 770-788) mix and finish; the quantization
 * of its hidden rows and wq_a with wkv; its state: the prepare, a source's compressor products (wkv, and wgate at
 * ratio 2) and its pool entry and key (2.1 launches 10 to 13 before 5 to 9: neither reads the other's arrays); qr alone
 * (a chunk's, ahead of its slabs' prepares, which form it again with the same bits); the quantization of qr, wq_b and
 * the indexer's wq_b; the indexer's weights; the queries; the scan and the core; wo_a, the quantization of its groups
 * and wo_b; the post. A thin step takes them all but qr alone in one call (layer 20 in two: its front, then the rest);
 * a chunk takes over the chunk every launch that reads no state, the queries over each span of its slabs, and over
 * each slab the rest (chunk_walk). */
enum ds41f_layer_step {
	DS41F_ENGRAM_VALUES = 1u << 0,
	DS41F_ENGRAM_PRODUCT = 1u << 1,
	DS41F_ENGRAM_GATE = 1u << 2,
	DS41F_SITE_MIX = 1u << 3,
	DS41F_INPUT_PRODUCTS = 1u << 4,
	DS41F_STATE_WRITES = 1u << 5,
	DS41F_POOL_PRODUCTS = 1u
		<< 6, /* of the hidden rows alone, between the prepare and the compressor */
	DS41F_QUERY_NORM = 1u << 7,
	DS41F_QUERY_PRODUCTS = 1u << 8,
	DS41F_INDEX_WEIGHTS_PRODUCT = 1u << 9, /* of the hidden rows alone */
	DS41F_QUERIES = 1u << 10, /* of wq_b's outputs and the rotation alone */
	DS41F_CORE = 1u << 11,
	DS41F_OUTPUT_PRODUCTS = 1u << 12,
	DS41F_SITE_POST = 1u
		<< 13, /* the TP sum of the sublayer's output, then the post */
	DS41F_SITE_SUM = 1u
		<< 14, /* split_overlap: the TP sum alone (a graph side run) */
	DS41F_SITE_POST_ONLY = 1u
		<< 15, /* split_overlap: the post alone, behind its sum */
	DS41F_SCAN = 1u
		<< 16, /* chunk_core: CORE's index scan alone, a slab at a time */
	DS41F_CORE_ALONE = 1u
		<< 17, /* chunk_core: CORE's core alone, over the chunk */
	DS41F_STATE_POOL = 1u
		<< 18, /* chunk_core: STATE_WRITES' pool writes alone, a slab at a time */
	DS41F_WINDOW_WRITES = 1u
		<< 19, /* chunk_core: the prepare's window writes alone, over the chunk */
	DS41F_POOL_CHUNK = 1u
		<< 20 /* pool_chunk: STATE_POOL over a piece (ds41f_attention_pool_prepare) */
};
#define DS41F_ENGRAM_STEPS \
	(DS41F_ENGRAM_VALUES | DS41F_ENGRAM_PRODUCT | DS41F_ENGRAM_GATE)
#define DS41F_FRONT_STEPS                                             \
	(DS41F_SITE_MIX | DS41F_INPUT_PRODUCTS | DS41F_STATE_WRITES | \
	 DS41F_POOL_PRODUCTS) /* layer 20's front (model_seam.h) */
#define DS41F_REST_STEPS                                                      \
	(DS41F_QUERY_PRODUCTS | DS41F_INDEX_WEIGHTS_PRODUCT | DS41F_QUERIES | \
	 DS41F_CORE | DS41F_OUTPUT_PRODUCTS | DS41F_SITE_POST)

/* exl3_wide_ffn_launches' phases (split_overlap runs them for each half of a chunk, the sums as graph side runs): the
 * compute (the front: mix, router, its selection, the rows quantized; the routed experts; the shared expert), the sums
 * and the post. split_overlap's joint mode runs the fronts and shared experts a half, the routed experts once over the
 * chunk's rows (a row's experts never read another row: the same bits). */
#define DS41F_FFN_FRONT 1u
#define DS41F_FFN_SUM 2u
#define DS41F_FFN_POST 4u
#define DS41F_FFN_EXPERTS 8u
#define DS41F_FFN_SHARED 16u
#define DS41F_FFN_EXPERTS_FRONT \
	32u /* mode 7: the joint call's route and gate/up (the halves' route) */
#define DS41F_FFN_DOWN_FIRST \
	64u /* its first half's down operands, down GEMM and combine */
#define DS41F_FFN_DOWN_SECOND 128u /* its second half's */
#define DS41F_FFN_COMPUTE \
	(DS41F_FFN_FRONT | DS41F_FFN_EXPERTS | DS41F_FFN_SHARED)
extern const uint32_t ds41f_copied_bytes[DS41F_COPIED_ARRAYS];
uint32_t ds41f_session_ranges_make(const struct ds41f_plan *plan,
				   struct session_range *ranges);
void ds41f_rows_select(struct ds41f_builder *b, uint32_t view,
		       uint32_t first_row, uint32_t rows);
void ds41f_slab_select(struct ds41f_builder *b, uint32_t slab);
void ds41f_chunk_select(struct ds41f_builder *b);
void ds41f_batch_slab_select(struct ds41f_builder *b, uint32_t slab);
uint32_t ds41f_piece_count_of(const struct ds41f_builder *b);
MUST_CHECK result_status ds41f_card_select(struct ds41f_builder *b,
					   uint32_t card);
bool ds41f_thin_tail_program(const struct ds41f_builder *b);
bool ds41f_thin_forms_of(const struct ds41f_builder *b);
bool ds41f_tail_program(const struct ds41f_builder *b);
void ds41f_segment_add(struct ds41f_builder *b, uint32_t layer,
		       uint32_t first_launch);
void ds41f_hand_off(struct ds41f_builder *b, uint32_t to,
		    uint64_t source_offset, uint64_t destination_offset,
		    uint32_t byte_count, uint32_t entry_ratio,
		    uint64_t session_stride);
MUST_CHECK result_status ds41f_dense(struct ds41f_builder *b,
				     const struct ds41f_tensor *weights,
				     struct linear_range rows,
				     struct linear_range output,
				     uint32_t output_type, uint32_t flags,
				     float output_scale);
bool ds41f_gathers_apply(const struct ds41f_builder *b);
MUST_CHECK result_status ds41f_rows_gather(
	struct ds41f_builder *b, uint32_t site, uint32_t phase,
	const struct ds41f_gather_range *ranges, uint32_t range_count);
bool ds41f_half_dead(const struct ds41f_builder *b, uint32_t layer,
		     uint32_t half);
uint32_t ds41f_selected_half(const struct ds41f_builder *b);
MUST_CHECK result_status ds41f_layer_steps(struct ds41f_builder *b,
					   uint32_t layer, uint32_t source,
					   uint32_t group, uint32_t steps);
MUST_CHECK result_status ds41f_ffn_launches(struct ds41f_builder *b,
					    uint32_t layer,
					    uint32_t *first_launch);
MUST_CHECK result_status ds41f_wide_ffn_launches(struct ds41f_builder *b,
						 uint32_t layer);
MUST_CHECK result_status ds41f_exl3_wide_ffn_phases(struct ds41f_builder *b,
						    uint32_t layer,
						    uint32_t phases);
MUST_CHECK result_status ds41f_step_launches(struct ds41f_builder *b,
					     bool embeds);
MUST_CHECK result_status ds41f_settings_push_launch(struct ds41f_builder *b);
MUST_CHECK result_status ds41f_batch_wide_head_launches(
	struct ds41f_builder *b, const struct ds41f_tp2_stage_plan *stage);
MUST_CHECK result_status ds41f_head_launches(struct ds41f_builder *b);
MUST_CHECK result_status ds41f_draft_shard_rank_launches(
	struct ds41f_builder *b, uint32_t verifier, uint32_t first_call);
MUST_CHECK result_status ds41f_draft_tap_launch(struct ds41f_builder *b,
						uint32_t layer);
MUST_CHECK result_status ds41f_draft_context_launches(struct ds41f_builder *b,
						      uint32_t piece_rows);
uint32_t
ds41f_batch_verifiers(const struct ds41f_tp2_program_description *description,
		      const struct program_limits *limits, uint32_t rows);
MUST_CHECK result_status ds41f_draft_walk(struct ds41f_builder *b);
MUST_CHECK result_status ds41f_batch_draft_rows_walk(struct ds41f_builder *b,
						     uint32_t verifiers);
MUST_CHECK result_status ds41f_steps_over(struct ds41f_builder *b,
					  uint32_t layer, uint32_t source,
					  uint32_t group, uint32_t steps,
					  bool is_pieces);
MUST_CHECK result_status ds41f_program_prepare(
	const struct ds41f_plan *plan, const struct backend_operations *backend,
	struct device_card *const *cards,
	const struct card_description_record *descriptions,
	enum program_kind kind, uint32_t row_count,
	struct execution_program **program_out);
void ds41f_program_destroy(struct execution_program *program);
MUST_CHECK result_status ds41f_tp2_layer_select(struct ds41f_builder *b,
						uint32_t layer);
void ds41f_gather_share(const struct ds41f_builder *b, uint32_t units,
			uint32_t granule, uint32_t *first_out,
			uint32_t *count_out);

#endif
