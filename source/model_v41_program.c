/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * model_v41_program.c: DeepSeek-V4.1-Flash's programs (first_token_plan.md 2.1, 2.8, 2.10) and its table of the
 * model seam. Contract: model_seam.h and model_v41.h. A program walks the embedding, the 40 layers and the head. A
 * visit of a card that holds a layer begins with its step segment (its rotation step); the embedding, each layer (its
 * launches in the order of 2.1, REF:968-994) and the head (the final collapse, the head, the sampler) are segments of
 * their own, a layer that keeps routed experts in host memory two: to its router, with an expert pause behind it,
 * and the rest. A card's rows that a later visit of it reads, a group's selection and the candidates, get a lease
 * (model_seam.h). Behind the segment that writes what another card reads: the streams and the carried pre where the
 * next segment runs on another card; an index layer's selection, a source's new pool entries and keys, to each card
 * with a copy. The plan's cards are checked first; each launch is its family's prepare function's, which checks it;
 * step.h's create checks the hand-offs and the pauses.
 */
#include "model_v41.h"
#include "model_v41_tp2.h"
#include "model_v41_tp2_program.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "anemone/attention_core.h"
#include "anemone/attention_output.h"
#include "anemone/attention_state.h"
#include "anemone/backend_records.h"
#include "anemone/dense_linear.h"
#include "anemone/official_hc.h"
#include "anemone/official_reference.h"
#include "anemone/dense_stream_records.h"
#include "anemone/draft.h"
#include "anemone/draft_share_records.h"
#include "anemone/engram_rows.h"
#include "anemone/exchange_records.h"
#include "anemone/expert_relaxed.h"
#include "anemone/exl3.h"
#include "anemone/exl3_tc.h"
#include "anemone/exl3_tp.h"
#include "anemone/expert_fused.h"
#include "anemone/hooks.h"
#include "anemone/hyper_connection.h"
#include "anemone/index_scan.h"
#include "anemone/image_routing.h"
#include "anemone/router.h"
#include "anemone/relaxed_linear.h"
#include "anemone/sampler.h"
#include "anemone/sampler_shard_records.h"
#include "anemone/site_front.h"
#include "anemone/tp2_join.h"
#include "anemone/ep_filter.h"
#include "anemone/exact_ring_records.h"
#include "anemone/quantized_wire_records.h"
#include "anemone/tp_exchange.h"
#include "anemone/ordered_push_parallel.h"
#include "prepare_common.h"

_Static_assert(EP_FILTER_HIDDEN_WIDTH == V41_DATA_EXPERT_INNER,
	       "EP2 uses full routed width");

#define HEAD_MMA_KERNEL_FILE "kernel_head_mma"      /* head_mma (kernel_head_mma.cu): its forms and geometry */
#define HEAD_MMA_KERNEL_8 \
	"head_mma_8_s4" /* head02: 2-8 rows (1 row: the stream is faster) */
#define HEAD_MMA_SPLIT_8 4u
#define HEAD_MMA_LEAST_ROWS 2u
#define HEAD_MMA_WIDE_FORMS 3u                /* head02: _64_s1 to 24 rows, _s2 to 48, _s4 above */
#define HEAD_MMA_WIDE_THIN_FORM 2u            /* _64_s4: HEAD_MMA_SPLIT_8, the 8-row form's split */
_Static_assert((1u << HEAD_MMA_WIDE_THIN_FORM) == HEAD_MMA_SPLIT_8, "the thin wide head splits as the 8-row form");
#define HEAD_MMA_THREADS 256u
#define HEAD_MMA_BLOCK_OUTPUTS 128u             /* of a split of 1: the _sN forms take 128 / N */
#define HEAD_MMA_STEP_INPUTS 32u
#define HEAD_MMA_MOST_ROWS 64u
#define HEAD_MMA_TILE_OUTPUTS 16u               /* a warp's outputs */
#define ROUTER_MMA_WARPS 4u                     /* router_mma: kernel_head_mma.cu's geometry */
#define ROUTER_MMA_REDUCE_THREADS 256u
#define TRUNK_MMA_KERNEL_FILE "kernel_trunk_mma" /* trunk_mma (kernel_trunk_mma.cu): forms t<T>_s<S>, geometry */
#define TRUNK_MMA_FORMS 11u                   /* trunk_mma_forms below */
#define TRUNK_MMA_THREADS 256u
#define TRUNK_MMA_TILE_OUTPUTS 16u
#define TRUNK_MMA_GROUP_INPUTS 32u
#define TRUNK_MMA_WIDE_KERNEL "trunk_mma_64"
#define TRUNK_MMA_THIN_STEM \
	"trunk_mma_8_" /* the 1-8-row forms' names: this, then t<T>_s<S>[_w16] */
#define TRUNK_MMA_GROUPED_FORMS 5u /* batch_thin_bits: the forms of 2 to 6 row groups of 8 (16 to 48 rows) */
#define TRUNK_MMA_GROUPED_MOST_ROWS \
	(LINEAR_MAXIMUM_ROWS * (TRUNK_MMA_GROUPED_FORMS + 1u))
#define TRUNK_MMA_NAME_BYTES 64u
#define TRUNK_MMA_PAIR_FORM 8u /* trunk_mma_forms' t1_s16: the form of wq_a, wkv and the shared gate / up shards */
#define TRUNK_MMA_SET_MOST_GROUPS 3u /* batch_row_sets: row sets of 1 to 3 row groups (8, 16 or 24 rows) */
#define FRONT_BRANCH_BOTH 2u /* front_branch: the FFN site's coefficient front a side run too */
#define DRAFT_WIDE_PRODUCTS 2u /* batch_draft_wide: the batched draft's products over all its rows too */
#define DRAFT_WIDE_CORES 3u /* batch_draft_wide: and its window writes and core once, each verifier's block whole */
#define TC_GATE_UP_PRODUCT_LAUNCH 1u
#define TC_DOWN_PRODUCT_LAUNCH 3u
#define BATCH_CORE_POOLS \
	3u /* whole-step pools, followed by separate compressor slot retention */
#define TRUNK_MMA_PAIR_SUFFIX "_pair"
#define TRUNK_MMA_FUSED_KERNEL \
	"trunk_mma_64_fused" /* its splits added by the last of them */
#define TRUNK_MMA_COUNTER_BYTES (V41_DATA_TRUNK_MMA_COUNTERS * sizeof(uint32_t))
#define TRUNK_MMA_WIDE_TILES 8u                /* tiles of 16 outputs a block of the _64 form */
#define TRUNK_MMA_WIDE_MOST_ROWS 64u
#define TRUNK_MMA_WIDE_BLOCKS 140u              /* the _64 form: splits until the launch has about this many blocks */
#define TRUNK_MMA_MOST_SPLITS 16u
#define TRUNK_MMA_REDUCE_THREADS 256u
#define TRUNK_MMA_VALUES_ALIGNMENT 16u          /* the _64 form stages the rows by 16-byte copies */
#define SWIGLU_LIMIT 10.0f                  /* CFG:19 swiglu_limit: every expert's (REF:845-847, 879, 887) */
#define EXPERT_BITS 4u                      /* E2M1 */
#define LEGACY_LAYER_MOST_LAUNCHES 41u      /* layer 14: n-gram 4, attention 23, FFN 14 */
/* A source may replace both legacy selections with five passes; the bound also covers each wide slab. */
#define EXL3_ATTENTION_EXTRA_LAUNCHES 4u /* BF16 QKV split and separate output post, per prepared slab. */
#define EXL3_ROUTED_EXTRA_LAUNCHES 4u /* PP loses its native shared join; TP adds two gather-adjacent passes. */
#define LAYER_MOST_LAUNCHES                                                  \
	(LEGACY_LAYER_MOST_LAUNCHES + 2u * (INDEX_SPLIT_LAUNCH_COUNT - 1u) + \
	 EXL3_ROUTED_EXTRA_LAUNCHES)
#define CARD_MOST_LAUNCHES 6u               /* rotation, optional scratch clear, embedding, final, head, sampler */
#define CHUNK_LAUNCHES \
	(1u + ROUTER_WIDE_LAUNCH_COUNT + EXPERT_RELAXED_WIDE_LAUNCH_COUNT)
#define MOST_SLABS              \
	(MODEL_WIDE_MOST_ROWS / \
	 LINEAR_MAXIMUM_ROWS) /* of a wide step: its views */
#define VIEW_BYTES                \
	(STEP_BLOCK_BYTES_VALUE + \
	 LINEAR_MAXIMUM_ROWS * STEP_ROW_BYTES_VALUE) /* the plan's */
/* embedding, head, draft; head shards' heads a rank; a BATCH program's verifier walks (tp4_v3 found the overflow) */
#define MOST_SEGMENTS                                                        \
	(BACKEND_MAXIMUM_CARDS + TP4_RANKS * MODEL_LAYERS + 4u + TP4_RANKS + \
	 DRAFT_SHARE_VERIFIERS)
#define CONTEXT_PIECE_LAUNCHES 6u             /* quantize, main projection, norm, quantize, two wkv products */
#define DRAFT_POSITION_LAUNCHES 2u           /* Markov product and sampler for one position */
#define DRAFT_FIXED_LAUNCHES 6u              /* block setup, normalization, head and confidence */
#define DRAFT_SHARD_LAUNCHES(positions) (2u + DRAFT_POSITION_LAUNCHES * (positions)) /* wait, cut; per position */
#define DRAFT_MOST_LAUNCHES(positions)              \
	(MODEL_DRAFT_STAGES * LAYER_MOST_LAUNCHES + \
	 DRAFT_POSITION_LAUNCHES * (positions) + DRAFT_FIXED_LAUNCHES)
#define MOST_HOST_EXPERTS (MODEL_LAYERS * MODEL_EXPERTS)
#define COPIED_ARRAYS 4u                    /* of a layer: its selection, its pool, its key tiles, its candidates */
#define EVERY_COPY 0xFu                     /* a bit an array of COPIED_ARRAYS */
#define FRONT_COPIES 0x6u                   /* the pool's and the key tiles': behind layer 20's front */
#define LATER_COPIES (EVERY_COPY & ~FRONT_COPIES)
#define COPIES_TO_A_CARD 17u                /* 8 index layers' selections, 4 sources' pools and keys, the candidates */
#define RESIDUAL_HAND_OFF_ARRAYS 2u          /* STREAMS and CARRIED_PRE */
#define PP_MOST_HAND_OFFS                                 \
	(RESIDUAL_HAND_OFF_ARRAYS * (MODEL_LAYERS + 1u) + \
	 COPIES_TO_A_CARD * BACKEND_MAXIMUM_CARDS)
/* Embedding and each prefix layer distribute both residuals to the other three ranks. Include the native-tail
 * boundary and a conservative complete set of source/index/candidate mirrors for that tail. */
#define TP4_OWNER_MOST_HAND_OFFS                                               \
	(RESIDUAL_HAND_OFF_ARRAYS *                                            \
		 ((V41_DATA_TP2_PREFIX_LAYERS + 1u) * (TP4_RANKS - 1u) + 1u) + \
	 COPIES_TO_A_CARD)
#define MOST_HAND_OFFS \
    (PP_MOST_HAND_OFFS > TP4_OWNER_MOST_HAND_OFFS ? PP_MOST_HAND_OFFS : TP4_OWNER_MOST_HAND_OFFS)
#define SESSION_KINDS 4u                        /* a layer's session ranges: ring, compressor slots, pool, key tiles */
#define VISION_FFN_EXTRA_LAUNCHES 2u
#define MOST_SESSION_RANGES \
	(SESSION_KINDS * V41_DATA_WALK_LAYERS * BACKEND_MAXIMUM_CARDS)
#define MOST_LEASES            \
	((MODEL_LAYERS + 1u) * \
	 BACKEND_MAXIMUM_CARDS) /* a card's groups and candidates */
#define OK STATUS_OK
#define THEN(call) (status != OK ? status : (call))                 /* the next step while every one before passed */
#define ARRAY(name) \
	(b->arrays[V41_DATA_ARRAY_##name]) /* of the card, for the next launches' rows */
#define WEIGHTS(name) (&weights[V41_DATA_##name]) /* a tensor of the layer */
#define BF16 DENSE_BFLOAT16
#define F32 DENSE_FLOAT32
#define TRUNK LINEAR_FAMILY_TRUNK_QUANTIZED
_Static_assert(
	V41_DATA_COMPRESSOR_SESSION_BYTES == ATTENTION_SLOTS_SESSION_BYTES,
	"model_v41.h sizes the compressor's slots as attention_state_records.h lays them out");
_Static_assert(V41_DATA_ARRAY_DRAFT_WINDOW_2 + 1u -
			       V41_DATA_ARRAY_DRAFT_WINDOW_0 ==
		       MODEL_DRAFT_STAGES,
	       "the context windows follow the draft stages in order");

/* An array's bytes a row and besides, and its kind of rows (enum v41_rows). */
#define BYTES_OF(array, row_bytes, fixed_bytes, holder, rows) \
	[V41_DATA_ARRAY_##array] = { (uint64_t)(row_bytes),   \
				     (uint64_t)(fixed_bytes), rows },
static const struct array_bytes {
    uint64_t row_bytes, fixed_bytes;
    uint32_t rows;
} array_bytes[V41_DATA_ARRAY_COUNT] = { V41_DATA_ARRAYS(BYTES_OF) };
#undef BYTES_OF
_Static_assert(V41_DATA_SPAN_ROWS % LINEAR_MAXIMUM_ROWS == 0 &&
		       LINEAR_WIDE_MOST_ROWS % V41_DATA_SPAN_ROWS == 0,
	       "a span is whole slabs, a piece whole spans");
enum { GATE_UP, DOWN };
static const struct matrix_shape_record expert_shapes[] = {
	[GATE_UP] = { V41_DATA_EXPERT_INNER, MODEL_HIDDEN, EXPERT_BITS, 0 },
	[DOWN] = { MODEL_HIDDEN, V41_DATA_EXPERT_INNER, EXPERT_BITS, 0 }
};
static const uint32_t copied_bytes[COPIED_ARRAYS] = {
	MODEL_SELECTION_ROW_BYTES, MODEL_STATE_ROW_BYTES, MODEL_KEY_BYTES,
	MODEL_CANDIDATES_ROW_BYTES
};
static const struct linear_range no_range = { 0, 0 };

/* Public handle arrays use the record header's slice-stamp order. */
#define LINEAR_SLICE_STAMP(count) EXL3_ATTENTION_SLICE_INSTANCE_##count,
enum exl3_attention_slice_instance {
	LINEAR_EVERY_SLICE_COUNT EXL3_ATTENTION_SLICE_INSTANCE_COUNT
};
#undef LINEAR_SLICE_STAMP
_Static_assert(EXL3_ATTENTION_SLICE_INSTANCE_COUNT == LINEAR_SLICE_INSTANCES,
	       "EXL3 attention uses the public linear-handle ordering");
#define EXL3_ATTENTION_SLICES 4u

static bool card_has_exl3(const struct model_plan_record *plan, uint32_t card)
{
	for (uint32_t layer = 0; layer < MODEL_LAYERS; layer++) {
		if ((plan->cards[card].layers >> layer & 1u) != 0 &&
		    plan->exl3_bits[layer] != 0)
			return true;
	}
    return false;
}

/* A program and all it points at in one allocation, the program first: its address is the allocation's; its launches
 * last, as many as its cards' parts hold (program_prepare). */
struct made_program {
	struct execution_program program;
	struct program_card_record cards[BACKEND_MAXIMUM_CARDS];
	struct segment_record segments[MOST_SEGMENTS];
	struct hand_off_record hand_offs[MOST_HAND_OFFS];
	struct expert_pause expert_pauses[MODEL_LAYERS];
	uint64_t host_expert_offsets[MOST_HOST_EXPERTS];
	struct session_range session_ranges[MOST_SESSION_RANGES];
	struct window_checkpoint checkpoints[V41_DATA_WALK_LAYERS];
	struct step_view views[BACKEND_MAXIMUM_CARDS]
			      [MOST_SLABS + V41_DATA_HALF_VIEWS]; /* its slabs,
                                                                     then split_overlap's halves */
	struct visit_lease leases[MOST_LEASES];
	struct kernel_launch launches[];
};

/* A program being made; card, arena, the kernels and arrays: of the card whose launches follow, at the rows they take
 * (rows from first_row on, read through view: 0 the step area, 1 + s slab s's view). A thin program's launches take
 * its rows; a wide program's its chunk_rows or a slab of them (slab_count slabs of at most 8). */
struct builder {
	struct official_reference_kernels official;
	const struct v41_tp2_layer_plan *tensor_layer;
	uint32_t tensor_slices;
	uint32_t tensor_join_kernel, exact_ring_kernel, unequal_ring_kernel,
		wire_quantize_kernel, wire_decode_kernel;
	uint32_t tensor_push_kernel;
	/* head_shards (model_v41_tp2_stage.h): the card's stage, sharded sampler and head cut's kernel. */
	const struct v41_tp2_stage_plan *head_stage;
	const struct v41_tp2_stage_plan
		*batch_stage; /* the result card's stage of a BATCH program */
	uint32_t sampler_shard_kernel, head_stream_kernel, draft_share_kernel,
		head_wide_kernel;
	uint32_t head_mma_kernel; /* head_mma: kernel_head_mma's 8-row form */
	uint32_t head_mma_wide_kernels
		[HEAD_MMA_WIDE_FORMS]; /* and its 64-row forms, splits 1, 2, 4 */
	uint32_t trunk_mma_kernels
		[TRUNK_MMA_FORMS]; /* trunk_mma: kernel_trunk_mma's forms, trunk_mma_forms' order, each
                                                    prepared on the card when a product first takes it */
	bool trunk_mma; /* the card's stage asks for trunk_mma */
	/* draft_layers (model_v41_tp2_stage.h): the card's stage runs its rank of the draft's layers; the program's
     * description and tensor plan for their plans; the draft tail's first segment made */
	bool draft_layers, draft_tail_begun;
	const struct v41_tp2_program_description *tp_description;
	struct v41_tp2_layer_plan *tp_tensor;
	uint32_t trunk_mma_wide_kernel; /* and its _64 form */
	uint32_t trunk_cone_layer; /* 0, or (a full program's decoder layer) its wide trunks' cone: relaxed_linear.h */
	uint32_t ffn_cone_layer; /* and its FFN's (the hyper site's mix, the router): the layer's own */
	uint32_t program_rows; /* the program's rows (chunk_rows: a half's while one is selected) */
	uint32_t trunk_mma_fused_kernel; /* and that form adding its own splits (trunk_mma_fused) */
	bool trunk_mma_thin_splits; /* the _64 form split as the 1-8-row form (trunk_mma_thin_splits) */
	bool batch_thin_bits; /* a BATCH program's rows the thin steps' bits (batch_thin_bits) */
	uint32_t batch_grouped_rows; /* and its products of at most so many rows by the grouped forms */
	uint32_t batch_tc_paired; /* and from so many rows its TC MoE's short forms paired */
	uint32_t batch_row_sets; /* and above batch_grouped_rows its trunk in row sets of so many */
	uint32_t batch_core_once; /* a BATCH step's core (2: and window writes) once */
	uint32_t front_branch; /* coefficient fronts as side branches: 1 attention, 2 both */
	uint32_t core_heads; /* the reference order's core: at most so many heads a block */
	bool draft_bf16_rows; /* the draft layers' attention products of BF16 rows */
	uint32_t batch_draft_wide; /* a batched draft's sites summed and posted once (1) */
	bool ffn_post_deferred; /* batch_draft_wide: ffn_launches leaves its sum and post */
	/* batch_draft_rows' walk: its slab blocks, collapse rows and rows (draft_slab_select) */
	uint64_t draft_slab_steps, draft_slab_collapse;
	uint32_t draft_slab_count, draft_rows;
	bool draft_wide_step; /* batch_draft_wide 3: all the rows through the step block behind the slab blocks */
	bool draft_wide_kernels; /* batch_draft_wide 2 over more than 8 rows: card_select prepares the wide forms too */
	/* draft_bf16_operands (class 80): the carry kernel the draft's quantizations take (draft_swap), else NO_KERNEL;
     * the kernels table's own quantization meanwhile */
	uint32_t draft_carry_kernel, draft_quantize_kernel;
	bool ring_column_owners; /* the BF16 ring sums with column owners (ring_column_owners) */
	/* Grouped forms of 2 + g row groups, in trunk_mma_forms' order; prepared when first taken. */
	uint32_t trunk_mma_grouped_kernels[TRUNK_MMA_GROUPED_FORMS]
					  [TRUNK_MMA_FORMS];
	/* their pair forms (two products a launch, --tp_launch_trims) of TRUNK_MMA_PAIR_FORM: 8 rows, then 16 .. 48 */
	uint32_t trunk_mma_pair_kernels[TRUNK_MMA_GROUPED_FORMS + 1u];
	uint32_t router_mma_kernel,
		router_mma_reduce_kernel; /* router_mma: the split-K router and its reduce */
	uint32_t router_mma_wide_kernel; /* router_mma_wide: its form over 64-row blocks of a wide step */
	uint32_t batch_join_kernel; /* batch_draft_rows, the result card: the row copies into position-major slots */
	uint32_t settings_push_kernel; /* head_shards, the result card: the sampler settings to every rank's copy */
	uint32_t sampler_call; /* the target sampler shard's use in the step: a BATCH program's slab */
	bool batch_draft; /* a BATCH program's draft walk of verifier batch_verifier (draft_records.h's flags) */
	uint32_t batch_verifier;
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
	bool tensor_local_stub;
	const struct model_plan_record *plan;
	const struct backend_operations *backend;
	struct device_card *const *cards;
	const struct card_description_record *descriptions;
	struct made_program *made;
	struct kernel_launch *launches
		[BACKEND_MAXIMUM_CARDS]; /* each card's part of made->launches */
	uint32_t launch_capacity[BACKEND_MAXIMUM_CARDS];
	struct linear_kernels_record linear;
	struct relaxed_linear_kernels relaxed;
	struct expert_relaxed_kernels expert;
	struct hyper_kernels_record hyper;
	struct dense_linear_kernels dense;
	struct official_hc_kernels official_hc;
	struct router_kernels_record router;
	struct attention_state_kernels state;
	struct index_scan_kernels scan;
	struct attention_core_kernels core;
	struct engram_rows_kernels engram;
	struct sampler_kernels_record sampler;
	struct draft_kernels_record draft;
	struct site_front_kernels fronts;
	struct vision_kernels vision;
	struct image_routing_kernels image_routing;
	uint32_t fill;
	struct attention_output_kernels output, draft_output;
	struct expert_fused_kernels fused, draft_fused;

	struct linear_kernels_record draft_linear;
	struct attention_core_kernels draft_core;
	struct linear_range arrays[V41_DATA_ARRAY_COUNT];
	uint64_t arena;
	uint32_t card, rows, stage, arithmetic_class;
	uint32_t chunk_rows, slab_count, first_row, view;
	uint32_t chunk_first,
		chunk_view; /* split_overlap: a half of the chunk, its first row and its view (else 0, 0) */
	uint32_t half_view_first; /* split_overlap: first half view behind the slabs; 0 without half views. */
	uint32_t half_rows; /* split_overlap: the first half's rows (the program's rows over two, a multiple of 8) */
	uint32_t tcp_emit; /* split_overlap mode 7: next TC prefill call's launches (enum tcp_emit). */
	uint32_t index_row_first; /* index_span: the view's row of the next scan's first row */
	bool index_grouped; /* index_span: scan rows_select's rows from index_row_first on (index_scans). */
	uint32_t ahead_layer
		[BACKEND_MAXIMUM_CARDS]; /* its pipeline: 1 + the layer whose half A a card took */
	bool reads_draft_step;
	bool vision_active;
	/* launches_untrimmed off, a TC-MoE FFN site of graph branches: the combine asked to add the routed sums to the
     * shared expert's rows (exl3_tc_records.h shared_sum), and whether the TC routed launches did */
	bool shared_sum_requested, shared_sum_fused;
	/* launches_untrimmed off: layer whose wq_b left this rank's heads only in compact rows (no copy into the
     * heads), and those rows, which its queries read (compact_rows 0: none) */
	uint32_t compact_layer, compact_first_row, compact_rows;
	/* and the layer whose core wrote the rank's heads into those rows alone (no copy for wo_a), the first row of
     * the rows wo_a then takes */
	uint32_t core_compact_layer, core_compact_first;
	bool core_compact;
};

/* What a session keeps on the cards (model_seam.h): of each walk layer, on every card that holds them, its window ring,
 * its compressor slots, its pool's entries and its key tiles (a mirror's too), identity SESSION_KINDS * layer + kind;
 * a draft stage has its ring alone (160, 164, 168), so a plan without the draft keeps its identities. */
static uint32_t session_ranges_make(const struct model_plan_record *plan,
				    struct session_range *ranges)
{
    uint32_t count = 0;
    for (uint32_t layer = 0; layer < V41_DATA_WALK_LAYERS; layer++) {
	    const uint32_t ratio = (V41_DATA_POOLED_LAYERS >> layer & 1u) != 0 ?
					   MODEL_POOLED_RATIO :
					   MODEL_DIRECT_RATIO;
	    const uint64_t pool = plan->pool_session_bytes[layer],
			   keys = plan->key_session_bytes[layer];
	    for (uint32_t kind = 0; kind < SESSION_KINDS; kind++) {
		    for (uint32_t card = 0; card < plan->card_count; card++) {
			    const struct v41_card *const planned =
				    &plan->cards[card];
			    const struct session_range kinds[SESSION_KINDS] = {
				    { planned->ring_offset[layer],
				      V41_DATA_RING_SESSION_BYTES,
				      MODEL_RING_SLOTS, MODEL_STATE_ROW_BYTES,
				      1u, card, 0 },
				    { planned->compressor_offset[layer],
				      V41_DATA_COMPRESSOR_SESSION_BYTES, 1u,
				      V41_DATA_COMPRESSOR_SESSION_BYTES, 0,
				      card, 0 },
				    { planned->pool_offset[layer], pool,
				      pool / MODEL_STATE_ROW_BYTES,
				      MODEL_STATE_ROW_BYTES, ratio, card, 0 },
				    { planned->key_tiles_offset[layer], keys,
				      keys / MODEL_TILE_BYTES, MODEL_TILE_BYTES,
				      ratio * MODEL_KEYS_PER_TILE, card, 0 }
			    };
			    if (kinds[kind].offset != PROGRAM_NOWHERE) {
				    ranges[count] = kinds[kind];
				    ranges[count++].identity =
					    layer * SESSION_KINDS + kind;
			    }
		    }
	    }
    }
    return count;
}

static void checkpoints_make(const struct model_plan_record *plan,
			     struct window_checkpoint *ranges)
{
	const uint32_t count = plan->limits.draft_positions != 0 ?
				       V41_DATA_WALK_LAYERS :
				       MODEL_LAYERS;
	for (uint32_t layer = 0; layer < count; layer++) {
		const uint32_t card = plan->layer_card[layer];
		ranges[layer] = (struct window_checkpoint){
			plan->cards[card].ring_offset[layer],
			plan->cards[card].checkpoint_offset[layer],
			V41_DATA_RING_SESSION_BYTES,
			MODEL_RING_SLOTS,
			MODEL_STATE_ROW_BYTES,
			MODEL_WINDOW - 1u,
			card
		};
	}
}

/* The plan puts the embedding, the head and every layer on one of its cards, and gives each card the layers and the
 * n-gram tables it puts there: so every card the walk names is the plan's, and its part of the launches holds them. */
static bool plan_is_valid(const struct model_plan_record *plan)
{
	uint64_t layers[BACKEND_MAXIMUM_CARDS] = { 0 };
	uint32_t tables[BACKEND_MAXIMUM_CARDS] = { 0 };
	bool is_valid = plan->embedding_card < plan->card_count &&
			plan->result_card < plan->card_count;
	for (uint32_t layer = 0; layer < MODEL_LAYERS && is_valid; layer++) {
		const uint32_t card = plan->layer_card[layer];
		const uint64_t tables_before =
			V41_DATA_ENGRAM_LAYERS &
			((1ull << layer) - 1u); /* its table's number */
		is_valid = card < plan->card_count;
		if (is_valid) {
			layers[card] |= 1ull << layer;
			tables[card] |=
				(uint32_t)(V41_DATA_ENGRAM_LAYERS >> layer & 1u)
				<< __builtin_popcountll(tables_before);
		}
	}
    for (uint32_t card = 0; card < plan->card_count && is_valid; card++) {
        is_valid = plan->cards[card].layers == layers[card] && plan->cards[card].engram_tables == tables[card];
    }
    if (is_valid && plan->limits.draft_positions != 0) {
	    const struct v41_card *const result =
		    &plan->cards[plan->result_card];
	    is_valid = (result->layers & V41_DATA_DRAFT_TAP_LAYERS) ==
			       V41_DATA_DRAFT_TAP_LAYERS &&
		       (result->holders >> V41_DRAFT & 1u) != 0;
	    for (uint32_t layer = MODEL_DRAFT_FIRST_LAYER;
		 layer < V41_DATA_WALK_LAYERS; layer++) {
		    is_valid = is_valid &&
			       plan->layer_card[layer] == plan->result_card;
	    }
    }
    if (!is_valid) {
	    LOG_ERROR(
		    "program_prepare: the plan puts the embedding, the head or a layer on none of its %" PRIu32
		    " cards, or gives a card other layers or n-gram tables than it puts there",
		    plan->card_count);
    }
    return is_valid;
}

/* The rows the next launches take: rows from first_row on, read through view (0: the step area; 1 + s: slab s's,
 * model_seam.h). A chunk's array lies first_row rows in, a span's as many rows into the span (which begins on a
 * multiple of V41_DATA_SPAN_ROWS), a slab's from its start (model_v41.h); a view's launches read its block and
 * rows in place of the step area's, which the plan lays out a slab of 8 rows apart. */
static void rows_select(struct builder *b, uint32_t view, uint32_t first_row, uint32_t rows)
{
	const struct v41_card *const planned = &b->plan->cards[b->card];
	b->view = view;
	b->first_row = first_row;
	b->rows = rows;
	for (uint32_t array = 0; array < V41_DATA_ARRAY_COUNT; array++) {
		const uint64_t offset = planned->array_offset[array];
		const uint64_t into = array_bytes[array].rows == V41_CHUNK ?
					      first_row :
				      array_bytes[array].rows == V41_SPAN ?
					      first_row % V41_DATA_SPAN_ROWS :
					      0;
		const uint64_t array_rows =
			array_bytes[array].rows == V41_DRAFT_ROWS ?
				b->plan->limits.draft_positions :
				rows;
		b->arrays[array] = (struct linear_range){
			offset == PROGRAM_NOWHERE ?
				offset :
				offset + into * array_bytes[array].row_bytes,
			array_bytes[array].fixed_bytes +
				array_rows * array_bytes[array].row_bytes
		};
	}
    if (view != 0) {
        /* a half view lies behind the slabs' (VIEW_BYTES apart), the second behind the first's block and rows */
        const uint32_t half = b->half_view_first != 0 && view >= b->half_view_first ? view - b->half_view_first + 1u : 0;
        const uint32_t slabs_before = half != 0 ? b->half_view_first - 1u : view - 1u;
	b->arrays[V41_DATA_ARRAY_STEP] = (struct linear_range){
		planned->array_offset[V41_DATA_ARRAY_VIEWS] +
			(uint64_t)slabs_before * VIEW_BYTES +
			(half == 2u ? STEP_BLOCK_BYTES_VALUE +
					      (uint64_t)b->half_rows *
						      STEP_ROW_BYTES_VALUE :
				      0u),
		STEP_BLOCK_BYTES_VALUE + (uint64_t)rows * STEP_ROW_BYTES_VALUE
	};
    }
    if (b->reads_draft_step) {
	    b->arrays[V41_DATA_ARRAY_STEP] =
		    b->arrays[V41_DATA_ARRAY_DRAFT_STEP];
    }
}

/* The rows the next launches take of a chunk's array whose product packs them at row_bytes, not at the array's row
 * (model_v41.h). */
static struct linear_range packed_rows(const struct builder *b,
				       enum v41_array array, uint64_t row_bytes)
{
    const uint64_t offset = b->plan->cards[b->card].array_offset[array];
    return (struct linear_range){ offset == PROGRAM_NOWHERE ?
					  offset :
					  offset + b->first_row * row_bytes,
				  b->rows * row_bytes };
}

/* A wide program's slab s: its rows (8, the last fewer) through its view; the whole chunk through the step area. With
 * split_overlap's half selected, slab s of the half (its slab views are the chunk's). */
static void slab_select(struct builder *b, uint32_t slab)
{
	const uint32_t first_row = slab * LINEAR_MAXIMUM_ROWS,
		       left = b->chunk_rows - first_row;
	rows_select(b, (b->chunk_first + first_row) / LINEAR_MAXIMUM_ROWS + 1u,
		    b->chunk_first + first_row,
		    left < LINEAR_MAXIMUM_ROWS ? left : LINEAR_MAXIMUM_ROWS);
}

/* A wide program's span s: its rows (V41_DATA_SPAN_ROWS, the last fewer) through the step area, which no launch of
 * a span reads. With split_overlap's half selected, span s of the half. */
static void span_select(struct builder *b, uint32_t span)
{
	const uint32_t first_row = span * V41_DATA_SPAN_ROWS,
		       left = b->chunk_rows - first_row;
	rows_select(b, 0, b->chunk_first + first_row,
		    left < V41_DATA_SPAN_ROWS ? left : V41_DATA_SPAN_ROWS);
}

static void chunk_select(struct builder *b)
{
    rows_select(b, b->chunk_view, b->chunk_first, b->chunk_rows);
}

/* A BATCH program's slab for its heads: slab_select, or for a program of one slab (8 rows or fewer: a thin walk, no
 * views, step_validation.c) its rows through the step's own block (slab_select would read view 1, never written). */
static void batch_slab_select(struct builder *b, uint32_t slab)
{
    if (b->slab_count > 1u)
        slab_select(b, slab);
    else
        chunk_select(b);
}

/* batch_draft_rows' walk: slab s of the batched draft (its 8 rows, its slab block as the step area, its collapse
 * rows), or with s the slab count all its rows (the step area slab 0's block: no launch over them reads it). */
static void draft_slab_select(struct builder *b, uint32_t slab)
{
    const bool all = slab >= b->draft_slab_count;
    rows_select(b, 0, all ? 0u : slab * LINEAR_MAXIMUM_ROWS,
		all ? b->draft_rows : LINEAR_MAXIMUM_ROWS);
    const uint64_t first = all ? 0u : slab;
    const uint64_t slab_collapse =
	    (uint64_t)LINEAR_MAXIMUM_ROWS * MODEL_HIDDEN_ROW_BYTES;
    const bool whole = all && b->draft_wide_step; /* the block of all rows, draft_records.h's WIDE_STEP */
    const struct linear_range step = {
	    b->draft_slab_steps + (whole ? b->draft_slab_count : first) *
					  DRAFT_SLAB_STEP_BYTES,
	    whole ? STEP_BLOCK_BYTES_VALUE +
			    (uint64_t)b->draft_rows * STEP_ROW_BYTES_VALUE :
		    DRAFT_SLAB_STEP_BYTES
    };
    b->arrays[V41_DATA_ARRAY_STEP] = step;
    b->arrays[V41_DATA_ARRAY_DRAFT_STEP] = step;
    b->arrays[V41_DATA_ARRAY_DRAFT_COLLAPSE] = (struct linear_range){
	    b->draft_slab_collapse + first * slab_collapse,
	    (uint64_t)(all ? b->draft_rows : LINEAR_MAXIMUM_ROWS) *
		    MODEL_HIDDEN_ROW_BYTES
    };
}

/* A wide program's chunk in pieces of at most LINEAR_WIDE_MOST_ROWS rows, a wide trunk launch's: their
 * count, and piece p's rows through the step area (which no launch of a piece reads). */
static uint32_t piece_count_of(const struct builder *b)
{
	return (b->chunk_rows + LINEAR_WIDE_MOST_ROWS - 1u) /
	       LINEAR_WIDE_MOST_ROWS;
}
static void piece_select(struct builder *b, uint32_t piece)
{
	const uint32_t first_row = piece * LINEAR_WIDE_MOST_ROWS,
		       left = b->chunk_rows - first_row;
	rows_select(b, b->chunk_view, b->chunk_first + first_row,
		    left < LINEAR_WIDE_MOST_ROWS ? left :
						   LINEAR_WIDE_MOST_ROWS);
}

/* The card whose launches follow: its arrays for the program's rows, and its kernels (a card gives a kernel prepared
 * again its handle). */
MUST_CHECK static result_status card_select(struct builder *b, uint32_t card)
{
    if (card >= b->plan->card_count || b->cards[card] == NULL) {
	    LOG_ERROR("program_prepare: card %" PRIu32 " of %" PRIu32
		      " runs part of the model and is not given",
		      card, b->plan->card_count);
	    return STATUS_INVALID_ARGUMENT;
    }
    struct device_card *const opened = b->cards[card];
    b->card = card;
    b->arena = b->plan->cards[card].arena_byte_count;
    b->draft_carry_kernel = BACKEND_NO_KERNEL;
    chunk_select(b);
    const uint32_t arithmetic_class = b->descriptions[card].arithmetic_class;
    b->arithmetic_class = arithmetic_class;
    b->unequal_ring_kernel = BACKEND_NO_KERNEL;
    result_status status = linear_kernels_prepare(b->backend, opened,
						  arithmetic_class, &b->linear);
    status = THEN(hyper_kernels_prepare(b->backend, opened, &b->hyper));
    status = THEN(dense_linear_kernels_prepare(b->backend, opened, &b->dense));
    status = THEN(router_kernels_prepare(b->backend, opened, &b->router));
    status = THEN(
	    attention_state_kernels_prepare(b->backend, opened, &b->state));
    /* chunk_core's window writes: the window kernel (the draft's, draft.h) on every card of a chunked program */
    status = b->slab_count > 1u || b->draft_wide_kernels ?
		     THEN(b->backend->kernel_prepare(
			     opened, KERNEL_FILE_ATTENTION_STATE,
			     KERNEL_ATTENTION_WINDOW,
			     ATTENTION_PREPARE_ARGUMENT_BYTES,
			     &b->state.window)) :
		     status;
    status = THEN(index_scan_kernels_prepare(b->backend, opened,
					     arithmetic_class, &b->scan));
    status = THEN(attention_core_kernels_prepare(b->backend, opened,
						 arithmetic_class, &b->core));
    status = THEN(engram_rows_kernels_prepare(b->backend, opened, &b->engram));
    status = THEN(sampler_kernels_prepare(b->backend, opened, &b->sampler));
    if (b->vision_active) {
	    status = THEN(image_routing_kernels_prepare(b->backend, opened,
							&b->image_routing));
	    if (card == 0) {
		    status = THEN(vision_kernels_prepare(
			    b->backend, opened, arithmetic_class, &b->vision));
	    }
    }
    /* Append wide products behind the thin table; retain the ordered/relaxed trunk selector below. */
    if ((b->slab_count > 1u || b->draft_wide_kernels) && b->plan->cards[card].layers != 0) {
	    status = THEN(linear_wide_kernels_prepare(
		    b->backend, opened, b->plan->limits.linear_profile,
		    b->plan->limits.linear_order, arithmetic_class,
		    &b->linear));
	    status = THEN(dense_linear_wide_kernels_prepare(
		    b->backend, opened, arithmetic_class, &b->dense));
	    /* dense_skinny_bf16 / _f32 (model_v41_tp2_stage.h): the skinny products above 48 rows, their outputs a block */
	    const struct v41_tp2_stage_plan *const stage =
		    b->tp_description == NULL ||
				    arithmetic_class != ARITHMETIC_CLASS_SM_80 ?
			    NULL :
			    b->tp_description->stages[card];
	    status = THEN(dense_linear_skinny_kernel_prepare(
		    b->backend, opened, DENSE_BFLOAT16,
		    stage == NULL ? 0u : stage->description.dense_skinny_bf16,
		    &b->dense));
	    status = THEN(dense_linear_skinny_kernel_prepare(
		    b->backend, opened, DENSE_FLOAT32,
		    stage == NULL ? 0u : stage->description.dense_skinny_f32,
		    &b->dense));
	    /* dense_staged (model_v41_tp2_stage.h): the staged wide form of both types */
	    status = THEN(dense_linear_staged_kernels_prepare(
		    b->backend, opened,
		    stage == NULL ? 0u : stage->description.dense_staged,
		    &b->dense));
	    if (b->tensor_layer != NULL || b->draft_wide_kernels)
		    status = THEN(linear_wide_reference_kernels_prepare(
			    b->backend, opened, arithmetic_class, &b->linear));
    }
    /* EXL3's BF16 QKV shapes require the public wide-four entry in addition to the tuned WO-A policy.
     * Every replicated TP rank owns these target layers even when the canonical primary is rank0. */
    if (b->slab_count > 1u && arithmetic_class == ARITHMETIC_CLASS_SM_80 &&
	card_has_exl3(b->plan, card)) {
	    status = THEN(b->backend->kernel_prepare(
		    opened, KERNEL_FILE_TRUNK_LINEAR,
		    KERNEL_TRUNK_LINEAR_BFLOAT16_WIDE_STEM "4",
		    TRUNK_LINEAR_BFLOAT16_ARGUMENT_BYTES,
		    &b->linear.trunk_linear_bfloat16_wide
			     [EXL3_ATTENTION_SLICE_INSTANCE_4]));
    }
    /* Retain the ordered/relaxed CMP handles after the ordinary and wide tables. */
    status = arithmetic_class == ARITHMETIC_CLASS_SM_80 &&
			     b->plan->cards[card].layers != 0 &&
			     (b->slab_count > 1u || b->draft_wide_kernels ||
			      b->plan->limits.cmp_order == CMP_ORDER_RELAXED) ?
		     THEN(relaxed_linear_kernels_prepare(b->backend, opened,
							 arithmetic_class,
							 &b->relaxed)) :
		     status;
    status = arithmetic_class == ARITHMETIC_CLASS_SM_80 &&
			     b->plan->cards[card].layers != 0 &&
			     (b->chunk_rows >= EXPERT_ORDERED_MINIMUM_ROWS ||
			      b->plan->limits.cmp_order == CMP_ORDER_RELAXED) ?
		     THEN(expert_relaxed_kernels_prepare(b->backend, opened,
							 arithmetic_class,
							 &b->expert)) :
		     status;
    if (status == OK && (card == b->plan->result_card || b->draft_layers) && b->plan->limits.draft_positions != 0) {
	    const uint32_t native_class =
		    arithmetic_class == ARITHMETIC_CLASS_SM_120 ?
			    ARITHMETIC_CLASS_SM_120_NATIVE :
			    arithmetic_class;
	    b->draft_linear = b->linear;
	    b->draft_core = b->core;
	    if (native_class != arithmetic_class) {
		    status = linear_kernels_prepare(
			    b->backend, opened, native_class, &b->draft_linear);
		    status = THEN(attention_core_kernels_prepare(
			    b->backend, opened, native_class, &b->draft_core));
		    status = b->slab_count > 1u ?
				     THEN(linear_wide_kernels_prepare(
					     b->backend, opened,
					     b->plan->limits.linear_profile,
					     b->plan->limits.linear_order,
					     native_class, &b->draft_linear)) :
				     status;
	    }
	    status = THEN(draft_kernels_prepare(
		    b->backend, opened, native_class, &b->hyper, &b->router,
		    &b->state, &b->draft_core, &b->draft));
	    b->draft_carry_kernel = BACKEND_NO_KERNEL;
	    const bool carries = b->tp_description != NULL &&
				 native_class == ARITHMETIC_CLASS_SM_80 &&
				 b->tp_description->stages[card]
					 ->description.draft_bf16_operands;
	    status = carries ? THEN(b->backend->kernel_prepare(
				       opened, KERNEL_FILE_ACTIVATION_QUANTIZE,
				       KERNEL_ACTIVATION_CARRY,
				       ACTIVATION_QUANTIZE_ARGUMENT_BYTES,
				       &b->draft_carry_kernel)) :
			       status;
    }
    if (b->plan->limits.official_hc_rows != 0 &&
	b->plan->cards[card].layers != 0)
	    status = THEN(official_hc_kernels_prepare(b->backend, opened,
						      &b->official_hc));
    /* Append fusions after every existing table. Chunk programs keep their piece-wide products. */
    /* tp4_v3's stage1 epoch02 fix (its engine 16:03): EXL3 tensor slabs take the site fronts too */
    const bool tensor_slab_front = b->tensor_layer != NULL &&
        (b->tensor_layer->exact_output_shards || b->plan->exl3_bits[b->tensor_layer->layer_index] != 0);
    if ((b->slab_count == 1u || tensor_slab_front) &&
        b->plan->cards[card].layers != 0) {
	    status = THEN(
		    site_front_kernels_prepare(b->backend, opened, &b->fronts));
	    if (b->plan->limits.row_capacity >= MODEL_WIDE_LEAST_ROWS) {
		    status = THEN(b->backend->kernel_prepare(
			    opened, KERNEL_FILE_FILL, KERNEL_FILL_WORDS,
			    FILL_RECORD_BYTES, &b->fill));
	    }
	    status = THEN(attention_output_kernels_prepare(
		    b->backend, opened, arithmetic_class, &b->output));
	    status = THEN(expert_fused_kernels_prepare(
		    b->backend, opened, arithmetic_class, &b->fused));
	    b->draft_output = b->output;
	    b->draft_fused = b->fused;
	    if (card == b->plan->result_card &&
		b->plan->limits.draft_positions != 0 &&
		arithmetic_class == ARITHMETIC_CLASS_SM_120) {
		    status = THEN(attention_output_kernels_prepare(
			    b->backend, opened, ARITHMETIC_CLASS_SM_120_NATIVE,
			    &b->draft_output));
		    status = THEN(expert_fused_kernels_prepare(
			    b->backend, opened, ARITHMETIC_CLASS_SM_120_NATIVE,
			    &b->draft_fused));
	    }
    }
    if (b->plan->limits.official_reference != 0) {
	    status = THEN(official_reference_kernels_prepare(
		    b->backend, opened, arithmetic_class, &b->official));
	    if (status == OK)
		    b->hyper.final_kept = b->official.final_kept;
    }
    return status;
}

/* The context and draft always use the card's native products, even while its target emulates class 80. */
static void draft_swap(struct builder *b, bool begins)
{
	const struct linear_kernels_record linear = b->linear;
	const struct attention_core_kernels core = b->core;
	const struct attention_output_kernels output = b->output;
	const struct expert_fused_kernels fused = b->fused;
	b->output = b->draft_output;
	b->fused = b->draft_fused;
	b->draft_output = output;
	b->draft_fused = fused;

	b->linear = b->draft_linear;
	b->core = b->draft_core;
	b->draft_linear = linear;
	b->draft_core = core;
	/* draft_bf16_operands: the draft's quantizations carry its rows' BF16 values (activation_carry) */
	if (b->draft_carry_kernel != BACKEND_NO_KERNEL && begins) {
		b->draft_quantize_kernel = b->linear.activation_quantize;
		b->linear.activation_quantize = b->draft_carry_kernel;
	} else if (b->draft_carry_kernel != BACKEND_NO_KERNEL) {
		b->draft_linear.activation_quantize = b->draft_quantize_kernel;
	}
    const uint32_t target_class = b->descriptions[b->card].arithmetic_class;
    b->arithmetic_class = begins && target_class == ARITHMETIC_CLASS_SM_120 ?
				  ARITHMETIC_CLASS_SM_120_NATIVE :
				  target_class;
}

/* The card's next count launches; NULL past its part, which plan_is_valid rules out (a prepare refuses NULL). */
static struct kernel_launch *slots(struct builder *b, uint32_t count)
{
	struct program_card_record *const card = &b->made->cards[b->card];
	const bool fits = card->launch_count + count <=
			  b->launch_capacity[b->card];
	card->launch_count += fits ? count : 0;
	return fits ? b->launches[b->card] + (card->launch_count - count) :
		      NULL;
}

/* trunk_mma's 1-8-row forms: a block of `warps` warps, groups of `split` warps on `tiles` tiles each. */
static const struct trunk_mma_form {
    const char *name;
    uint32_t tiles, split, warps;
} trunk_mma_forms[TRUNK_MMA_FORMS] = { { "trunk_mma_8_t1_s1", 1, 1, 8 },
				       { "trunk_mma_8_t1_s2", 1, 2, 8 },
				       { "trunk_mma_8_t1_s4", 1, 4, 8 },
				       { "trunk_mma_8_t1_s8", 1, 8, 8 },
				       { "trunk_mma_8_t2_s1", 2, 1, 8 },
				       { "trunk_mma_8_t2_s2", 2, 2, 8 },
				       { "trunk_mma_8_t2_s4", 2, 4, 8 },
				       { "trunk_mma_8_t2_s8", 2, 8, 8 },
				       { "trunk_mma_8_t1_s16", 1, 16, 16 },
				       { "trunk_mma_8_t1_s8_w16", 1, 8, 16 },
				       { "trunk_mma_8_t1_s4_w16", 1, 4, 16 } };

/* trunk_mma: the form of a shard shape (relbench02), else the split that gives the narrow shapes the most warps. */
static uint32_t trunk_mma_form_of(const struct matrix_shape_record *shape)
{
    static const struct { uint32_t output_size, input_size, form; } measured[] = {{0, 0, 0}};
    for (uint32_t index = 0; index < sizeof(measured) / sizeof(measured[0]); index++)
        if (measured[index].output_size == shape->output_size && measured[index].input_size == shape->input_size)
            return measured[index].form;
    const uint32_t tiles = shape->output_size / TRUNK_MMA_TILE_OUTPUTS;
    return tiles <= 128u ? 8u : tiles <= 512u ? 2u : 1u;
}

/* --tp_trunk_mma: a prepared BF16-row trunk launch of 1 to 8 rows (trunk_linear_bfloat16_<S>, the same record)
 * becomes kernel_trunk_mma's form for its shape. */
static bool trunk_mma_applies(const struct builder *b,
			      const struct matrix_shape_record *shape)
{
	return b->trunk_mma && b->rows <= LINEAR_MAXIMUM_ROWS &&
	       shape->input_size % TRUNK_MMA_GROUP_INPUTS == 0 &&
	       shape->output_size % TRUNK_MMA_TILE_OUTPUTS == 0;
}

/* trunk_mma applies to the shape and its form is prepared on the card (a card prepares at most
 * BACKEND_MAXIMUM_KERNELS: only the forms the products take, when first taken). */
static bool trunk_mma_ready(struct builder *b,
			    const struct matrix_shape_record *shape)
{
    if (!trunk_mma_applies(b, shape))
        return false;
    uint32_t *const kernel = &b->trunk_mma_kernels[trunk_mma_form_of(shape)];
    if (*kernel == BACKEND_NO_KERNEL &&
	b->backend->kernel_prepare(
		b->cards[b->card], TRUNK_MMA_KERNEL_FILE,
		trunk_mma_forms[trunk_mma_form_of(shape)].name,
		TRUNK_LINEAR_BFLOAT16_ARGUMENT_BYTES, kernel) != OK)
	    *kernel = BACKEND_NO_KERNEL;
    return *kernel != BACKEND_NO_KERNEL;
}

static void trunk_mma_swap(struct builder *b, struct kernel_launch *launch,
			   const struct matrix_shape_record *shape)
{
    if (!trunk_mma_ready(b, shape))
        return; /* the prepared launch stays */
    const uint32_t index = trunk_mma_form_of(shape);
    const struct trunk_mma_form *const form = &trunk_mma_forms[index];
    const uint32_t block_tiles = form->tiles * form->warps / form->split;
    launch->kernel = b->trunk_mma_kernels[index];
    launch->grid[0] = (shape->output_size / TRUNK_MMA_TILE_OUTPUTS + block_tiles - 1u) / block_tiles;
    launch->grid[1] = launch->grid[2] = 1;
    launch->block[0] = form->warps * LINEAR_THREADS_PER_WARP;
    launch->block[1] = launch->block[2] = 1;
    launch->dynamic_shared_byte_count = 0;
}

static struct linear_range trunk_mma_partials_of(const struct builder *b);

/* --prefill_thin_tail (model_v41_tp2_stage.h thin_tail): a full program of 9 to 63 rows, each row in a thin step's
 * forms as a BATCH program's (batch_thin_bits_of, thin_forms_of) */
static bool thin_tail_program(const struct builder *b)
{
	const enum program_kind kind = b->made->program.kind;
	return b->tp_description != NULL &&
	       b->tp_description->stages[b->card]->description.thin_tail &&
	       (kind == PROGRAM_FULL || kind == PROGRAM_WIDE) &&
	       b->program_rows > LINEAR_MAXIMUM_ROWS &&
	       b->program_rows < MODEL_WIDE_LEAST_ROWS;
}

/* A BATCH program, or a thin-tail program: its rows a thin step's bits where the forms say (batch_thin_bits_of) */
static bool thin_forms_of(const struct builder *b)
{
	return b->made->program.kind == PROGRAM_BATCH || thin_tail_program(b);
}

/* batch_thin_bits: a BATCH (or thin-tail) program's target products take the forms whose rows have the thin steps'
 * bits */
static bool batch_thin_bits_of(const struct builder *b)
{
    return b->batch_thin_bits && thin_forms_of(b);
}

/* trunk_mma's _64 form: the splits of the input groups that bring a launch near TRUNK_MMA_WIDE_BLOCKS blocks, within
 * the partials (splits times outputs at most V41_DATA_TRUNK_MMA_PARTIAL_VALUES); with trunk_mma_thin_splits the
 * 1-8-row form's split of the shape where the step's rows' partials fit (its runs the same input groups, each summed
 * from +0 in ascending groups by the same instructions, the runs added in order: a row's bits the thin form's) */
static uint32_t trunk_mma_splits_of(const struct builder *b,
				    const struct matrix_shape_record *shape)
{
    const uint32_t thin = trunk_mma_forms[trunk_mma_form_of(shape)].split;
    if ((b->trunk_mma_thin_splits || batch_thin_bits_of(b)) && thin <= shape->input_size / TRUNK_MMA_GROUP_INPUTS &&
        trunk_mma_partials_of(b).byte_count >= (uint64_t)thin * b->rows * shape->output_size * sizeof(float))
        return thin;
    const uint32_t tiles = shape->output_size / TRUNK_MMA_TILE_OUTPUTS;
    const uint32_t blocks = (tiles + TRUNK_MMA_WIDE_TILES - 1u) / TRUNK_MMA_WIDE_TILES;
    uint32_t splits = 1;
    while (splits < TRUNK_MMA_MOST_SPLITS &&
	   2u * splits * blocks <= TRUNK_MMA_WIDE_BLOCKS &&
	   2u * splits * shape->output_size <=
		   V41_DATA_TRUNK_MMA_PARTIAL_VALUES &&
	   2u * splits <= shape->input_size / TRUNK_MMA_GROUP_INPUTS)
	    splits *= 2u;
    return splits;
}

/* --tp_trunk_mma, a step of 9 to 64 rows: a prepared wide BF16-row trunk launch becomes kernel_trunk_mma's _64 form
 * over the same matrix, values and output, and with more than one split the relaxed reduce behind it (a launch more);
 * where the form does not apply the prepared launch stays. */
static struct linear_range trunk_mma_partials_of(const struct builder *b)
{
    /* the fused form's counters end the area */
    const struct linear_range area =
	    b->tensor_layer != NULL ? b->tensor_layer->trunk_mma_partials :
				      (struct linear_range){ 0, 0 };
    return area.byte_count < TRUNK_MMA_COUNTER_BYTES ?
		   (struct linear_range){ 0, 0 } :
		   (struct linear_range){ area.byte_offset,
					  area.byte_count -
						  TRUNK_MMA_COUNTER_BYTES };
}

static bool
trunk_mma_wide_applies(const struct builder *b,
		       const struct linear_trunk_bfloat16_description *product)
{
	const struct matrix_shape_record *const shape = &product->matrix.shape;
	const uint32_t tiles = shape->output_size / TRUNK_MMA_TILE_OUTPUTS,
		       splits = trunk_mma_splits_of(b, shape);
	return b->trunk_mma_wide_kernel != BACKEND_NO_KERNEL &&
	       b->rows > LINEAR_MAXIMUM_ROWS &&
	       b->rows <= TRUNK_MMA_WIDE_MOST_ROWS &&
	       shape->input_size % TRUNK_MMA_GROUP_INPUTS == 0 &&
	       shape->output_size % TRUNK_MMA_TILE_OUTPUTS == 0 &&
	       product->segment_count != 0 &&
	       tiles % (product->segment_count * TRUNK_MMA_WIDE_TILES) == 0 &&
	       product->values.byte_offset % TRUNK_MMA_VALUES_ALIGNMENT == 0 &&
	       (splits == 1u ||
		(b->slab_count >= 2u &&
		 b->relaxed.reduce != BACKEND_NO_KERNEL &&
		 trunk_mma_partials_of(b).byte_count >=
			 (uint64_t)splits * b->rows * shape->output_size *
				 sizeof(float)));
}

/* batch_thin_bits, a BATCH step of 9 to batch_grouped_rows rows (at most 48): the grouped form of the shape's 1-8-row
 * form (its row groups of 8 at that form's splits: each row the thin bits, the weights decoded once), prepared on the
 * card when first taken; none past those rows or where the card refuses it. */
static uint32_t trunk_mma_grouped_ready(struct builder *b,
					const struct matrix_shape_record *shape)
{
	if (!batch_thin_bits_of(b) || b->rows <= LINEAR_MAXIMUM_ROWS ||
	    b->rows > b->batch_grouped_rows ||
	    b->rows > TRUNK_MMA_GROUPED_MOST_ROWS)
		return BACKEND_NO_KERNEL;
	const uint32_t index = trunk_mma_form_of(shape);
	const uint32_t groups =
		(b->rows + LINEAR_MAXIMUM_ROWS - 1u) / LINEAR_MAXIMUM_ROWS;
	uint32_t *const kernel =
		&b->trunk_mma_grouped_kernels[groups - 2u][index];
	if (*kernel == BACKEND_NO_KERNEL) {
		char name[TRUNK_MMA_NAME_BYTES];
		(void)snprintf(name, sizeof(name), "trunk_mma_%" PRIu32 "_%s",
			       groups * LINEAR_MAXIMUM_ROWS,
			       trunk_mma_forms[index].name +
				       strlen(TRUNK_MMA_THIN_STEM));
		if (b->backend->kernel_prepare(
			    b->cards[b->card], TRUNK_MMA_KERNEL_FILE, name,
			    TRUNK_LINEAR_BFLOAT16_ARGUMENT_BYTES, kernel) != OK)
			*kernel = BACKEND_NO_KERNEL;
	}
    return *kernel;
}

/* batch_row_sets (8, 16 or 24), a BATCH step of more than batch_grouped_rows rows (to 64): the shape's 1-8-row form
 * or its grouped form of 2 or 3 row groups over sets of so many of the step's rows (gridDim.y): each row the thin
 * bits, the weights decoded once a set; none where the card refuses it. */
static uint32_t trunk_mma_sets_ready(struct builder *b,
				     const struct matrix_shape_record *shape)
{
	const uint32_t groups = b->batch_row_sets / LINEAR_MAXIMUM_ROWS;
	if (!batch_thin_bits_of(b) || groups == 0 ||
	    groups > TRUNK_MMA_SET_MOST_GROUPS ||
	    b->rows <= LINEAR_MAXIMUM_ROWS ||
	    b->rows > TRUNK_MMA_WIDE_MOST_ROWS)
		return BACKEND_NO_KERNEL;
	const uint32_t index = trunk_mma_form_of(shape);
	uint32_t *const kernel =
		groups == 1u ?
			&b->trunk_mma_kernels[index] :
			&b->trunk_mma_grouped_kernels[groups - 2u][index];
	if (*kernel == BACKEND_NO_KERNEL) {
		char name[TRUNK_MMA_NAME_BYTES];
		(void)snprintf(name, sizeof(name), "trunk_mma_%" PRIu32 "_%s",
			       groups * LINEAR_MAXIMUM_ROWS,
			       trunk_mma_forms[index].name +
				       strlen(TRUNK_MMA_THIN_STEM));
		if (b->backend->kernel_prepare(
			    b->cards[b->card], TRUNK_MMA_KERNEL_FILE, name,
			    TRUNK_LINEAR_BFLOAT16_ARGUMENT_BYTES, kernel) != OK)
			*kernel = BACKEND_NO_KERNEL;
	}
    return *kernel;
}

/* --tp_launch_trims, a tensor layer's two independent products of one input just launched (wq_a and wkv; the shared
 * expert's gate and up): when both took TRUNK_MMA_PAIR_FORM's 1-8-row or grouped kernel over the same rows, one launch
 * of its pair form in their place (each block the bits of its own launch's: kernel_trunk_mma.cu); else as they are. */
static void trunk_mma_pair_merge(struct builder *b)
{
	struct program_card_record *const card = &b->made->cards[b->card];
	if (b->tensor_layer == NULL || b->tensor_layer->launches_untrimmed ||
	    card->launch_count < 2u)
		return;
	struct kernel_launch *const first =
		&b->launches[b->card][card->launch_count - 2u];
	const struct kernel_launch *const second = first + 1;
	uint32_t form = TRUNK_MMA_GROUPED_FORMS + 1u;
	if (first->kernel == b->trunk_mma_kernels[TRUNK_MMA_PAIR_FORM])
		form = 0;
	for (uint32_t groups = 0; groups < TRUNK_MMA_GROUPED_FORMS; groups++)
		if (first->kernel ==
		    b->trunk_mma_grouped_kernels[groups][TRUNK_MMA_PAIR_FORM])
			form = groups + 1u;
	struct trunk_linear_bfloat16_arguments records[2];
	if (form > TRUNK_MMA_GROUPED_FORMS ||
	    first->kernel == BACKEND_NO_KERNEL ||
	    second->kernel != first->kernel ||
	    second->branch != first->branch ||
	    first->argument_byte_count !=
		    TRUNK_LINEAR_BFLOAT16_ARGUMENT_BYTES ||
	    second->argument_byte_count !=
		    TRUNK_LINEAR_BFLOAT16_ARGUMENT_BYTES ||
	    first->block[0] != second->block[0] ||
	    first->grid[1] != second->grid[1])
		return; /* grid[1]: the row sets (batch_row_sets), the pair form's as its products' */
	memcpy(&records[0], first->argument, sizeof(records[0]));
	memcpy(&records[1], second->argument, sizeof(records[1]));
	if (records[0].row_count != records[1].row_count ||
	    records[0].values_offset != records[1].values_offset)
		return;
	uint32_t *const kernel = &b->trunk_mma_pair_kernels[form];
	if (*kernel == BACKEND_NO_KERNEL) {
		char name[TRUNK_MMA_NAME_BYTES];
		(void)snprintf(name, sizeof(name),
			       "trunk_mma_%" PRIu32 "_%s" TRUNK_MMA_PAIR_SUFFIX,
			       (form + 1u) * LINEAR_MAXIMUM_ROWS,
			       trunk_mma_forms[TRUNK_MMA_PAIR_FORM].name +
				       strlen(TRUNK_MMA_THIN_STEM));
		if (b->backend->kernel_prepare(
			    b->cards[b->card], TRUNK_MMA_KERNEL_FILE, name,
			    sizeof(struct trunk_mma_pair_arguments),
			    kernel) != OK) {
			*kernel = BACKEND_NO_KERNEL;
			return;
		}
	}
	const struct trunk_mma_pair_arguments pair = { records[0], records[1],
						       first->grid[0], 0 };
	const uint32_t branch = first->branch;
	prepare_launch_fill(first, *kernel, first->grid[0] + second->grid[0],
			    first->grid[1], first->block[0], 0, &pair,
			    sizeof(pair));
	first->branch = branch;
	card->launch_count--;
}

MUST_CHECK static result_status
trunk_mma_wide(struct builder *b, struct kernel_launch *launch,
	       const struct linear_trunk_bfloat16_description *product)
{
	const struct matrix_shape_record *const shape = &product->matrix.shape;
	const uint32_t tiles = shape->output_size / TRUNK_MMA_TILE_OUTPUTS,
		       splits = trunk_mma_splits_of(b, shape);
	const struct linear_range partials = trunk_mma_partials_of(b);
	if (!trunk_mma_wide_applies(b, product))
		return OK;
	uint32_t grouped = trunk_mma_grouped_ready(b, shape), sets = 1;
	if (grouped == BACKEND_NO_KERNEL) {
		grouped = trunk_mma_sets_ready(b, shape);
		sets = (b->rows + b->batch_row_sets - 1u) /
		       (b->batch_row_sets != 0 ? b->batch_row_sets : 1u);
	}
	if (grouped != BACKEND_NO_KERNEL) {
		/* the 1-8-row form's geometry and record over the step's rows (row sets: gridDim.y): one launch, no
         * partials */
		const struct trunk_mma_form *const form =
			&trunk_mma_forms[trunk_mma_form_of(shape)];
		const uint32_t block_tiles =
			form->tiles * form->warps / form->split;
		const struct trunk_linear_bfloat16_arguments thin = {
			.weights_offset = product->matrix.stored.byte_offset,
			.values_offset = product->values.byte_offset,
			.output_offset = product->output.byte_offset,
			.output_size = shape->output_size,
			.input_size = shape->input_size,
			.row_count = b->rows,
			.segment_count = product->segment_count
		};
		prepare_launch_fill(launch, grouped,
				    (tiles + block_tiles - 1u) / block_tiles,
				    sets, form->warps * LINEAR_THREADS_PER_WARP,
				    0, &thin,
				    TRUNK_LINEAR_BFLOAT16_ARGUMENT_BYTES);
		return OK;
	}
	const struct trunk_relaxed_bfloat16_arguments record = {
		.weights_offset = product->matrix.stored.byte_offset,
		.values_offset = product->values.byte_offset,
		.output_offset = product->output.byte_offset,
		.partials_offset = splits > 1u ? partials.byte_offset : 0,
		.output_size = shape->output_size,
		.input_size = shape->input_size,
		.row_count = b->rows,
		.segment_count = product->segment_count,
		.k_split_count = splits,
		.fold_reference = 0
	};
	prepare_launch_fill(launch, b->trunk_mma_wide_kernel,
			    tiles / TRUNK_MMA_WIDE_TILES * splits, 1,
			    TRUNK_MMA_THREADS, 0, &record,
			    TRUNK_RELAXED_BFLOAT16_ARGUMENT_BYTES);
	if (splits == 1u)
		return OK;
	if (b->trunk_mma_fused_kernel != BACKEND_NO_KERNEL &&
	    tiles / TRUNK_MMA_WIDE_TILES <= V41_DATA_TRUNK_MMA_COUNTERS) {
		/* trunk_mma_fused: the last split of a block adds the splits (the reduce's order): no reduce launch */
		const struct trunk_mma_fused_arguments fused = {
			record, partials.byte_offset + partials.byte_count
		};
		prepare_launch_fill(launch, b->trunk_mma_fused_kernel,
				    tiles / TRUNK_MMA_WIDE_TILES * splits, 1,
				    TRUNK_MMA_THREADS, 0, &fused,
				    sizeof(fused));
		return OK;
	}
	const struct trunk_relaxed_reduce_arguments reduce = {
		.partials_offset = partials.byte_offset,
		.output_offset = { product->output.byte_offset, 0 },
		.output_size = { shape->output_size, 0 },
		.row_count = b->rows,
		.k_split_count = splits,
		.output_scale = 1.0f,
		.reserved_zero = 0
	};
	struct kernel_launch *const reduce_launch = slots(b, 1);
	if (reduce_launch == NULL)
		return STATUS_INVALID_ARGUMENT;
	const uint32_t results = b->rows * shape->output_size;
	prepare_launch_fill(reduce_launch, b->relaxed.reduce,
			    (results + TRUNK_MMA_REDUCE_THREADS - 1u) /
				    TRUNK_MMA_REDUCE_THREADS,
			    1, TRUNK_MMA_REDUCE_THREADS, 0, &reduce,
			    TRUNK_RELAXED_REDUCE_ARGUMENT_BYTES);
	return OK;
}

/* --prefill_tail_rows (model_v41_tp2_stage.h tail_rows): a tail program, a full (or, --prefill_front_tail, a wide)
 * program of more than 64 rows and at most the stage's tail_rows, no whole multiple of the exact plan's 256 (the
 * ladder's programs above 64 rows all are: main.c), whose rows take the 64-row chunk's forms a 64-row group, one
 * decode TC MoE call over them and no split halves */
static bool tail_program(const struct builder *b)
{
    const uint32_t tail = b->tp_description != NULL ? b->tp_description->stages[b->card]->description.tail_rows : 0u;
    const enum program_kind kind = b->made->program.kind;
    return (kind == PROGRAM_FULL || kind == PROGRAM_WIDE) && tail != 0 &&
	   b->program_rows > TRUNK_MMA_WIDE_MOST_ROWS &&
	   b->program_rows <= tail &&
	   b->program_rows % MODEL_FULL_CONE_LEAST_ROWS != 0;
}

/* The rows a call of the decode TC MoE takes (exl3_tc_records.h): 64, a tail program's all of them. */
static uint32_t tc_most_rows(const struct builder *b)
{
	return tail_program(b) ? b->program_rows : EXL3_TC_MOST_ROWS;
}

/* tail_sets (model_v41_tp2_stage.h): a tail program's product, applying to its first 64-row group, as one launch of
 * the _64 form over all its groups (row sets, kernel_trunk_mma.cu: each group a one-group launch's splits and bits)
 * and with more than one split one reduce over all its rows (each sum from +0 in ascending splits: the per-group
 * reduces' bits), the partials in trunk_mma_set_partials; false where the form or the partials do not take it. */
static bool
trunk_mma_sets_put(struct builder *b, struct kernel_launch *launch,
		   const struct linear_trunk_bfloat16_description *group,
		   uint32_t rows, result_status *status_out)
{
	const struct matrix_shape_record *const shape = &group->matrix.shape;
	const uint32_t tiles = shape->output_size / TRUNK_MMA_TILE_OUTPUTS,
		       splits = trunk_mma_splits_of(b, shape);
	const struct linear_range partials =
		b->tensor_layer != NULL ?
			b->tensor_layer->trunk_mma_set_partials :
			(struct linear_range){ 0, 0 };
	if (b->tensor_layer == NULL || !b->tensor_layer->tail_sets ||
	    !trunk_mma_wide_applies(b, group) ||
	    trunk_mma_grouped_ready(b, shape) != BACKEND_NO_KERNEL ||
	    (splits > 1u && partials.byte_count < (uint64_t)splits * rows *
							  shape->output_size *
							  sizeof(float)))
		return false;
	const struct trunk_relaxed_bfloat16_arguments record = {
		.weights_offset = group->matrix.stored.byte_offset,
		.values_offset = group->values.byte_offset,
		.output_offset = group->output.byte_offset,
		.partials_offset = splits > 1u ? partials.byte_offset : 0,
		.output_size = shape->output_size,
		.input_size = shape->input_size,
		.row_count = b->rows,
		.segment_count = group->segment_count,
		.k_split_count = splits,
		.fold_reference = 0
	};
	prepare_launch_fill(launch, b->trunk_mma_wide_kernel,
			    tiles / TRUNK_MMA_WIDE_TILES * splits,
			    rows / b->rows, TRUNK_MMA_THREADS, 0, &record,
			    TRUNK_RELAXED_BFLOAT16_ARGUMENT_BYTES);
	*status_out = OK;
	if (splits == 1u)
		return true;
	const struct trunk_relaxed_reduce_arguments reduce = {
		.partials_offset = partials.byte_offset,
		.output_offset = { group->output.byte_offset, 0 },
		.output_size = { shape->output_size, 0 },
		.row_count = rows,
		.k_split_count = splits,
		.output_scale = 1.0f,
		.reserved_zero = 0
	};
	struct kernel_launch *const reduce_launch = slots(b, 1);
	const uint32_t results = rows * shape->output_size;
	if (reduce_launch == NULL)
		*status_out = STATUS_INVALID_ARGUMENT;
	else
		prepare_launch_fill(reduce_launch, b->relaxed.reduce,
				    (results + TRUNK_MMA_REDUCE_THREADS - 1u) /
					    TRUNK_MMA_REDUCE_THREADS,
				    1, TRUNK_MMA_REDUCE_THREADS, 0, &reduce,
				    TRUNK_RELAXED_REDUCE_ARGUMENT_BYTES);
	return true;
}

/* trunk_mma_wide for the selected rows (9 to 64); a tail program's in groups of 64 rows, each its own launch
 * of the group's rows (a 64-row chunk's: the same splits, the same bits), launch then the slots behind it, or with
 * tail_sets one launch over the groups (trunk_mma_sets_put); a tail program's product that no group form takes is
 * refused (its rows would take another form's bits). */
MUST_CHECK static result_status
trunk_mma_wide_rows(struct builder *b, struct kernel_launch *launch,
		    const struct linear_trunk_bfloat16_description *product)
{
    const uint32_t rows = b->rows, group_rows = TRUNK_MMA_WIDE_MOST_ROWS;
    if (!tail_program(b) || rows <= group_rows)
        return trunk_mma_wide(b, launch, product);
    if (rows % group_rows != 0)
	    return STATUS_INVALID_ARGUMENT;
    const uint64_t value_row_bytes = (uint64_t)product->segment_count *
				     product->matrix.shape.input_size *
				     LINEAR_BFLOAT16_BYTES;
    const uint64_t output_row_bytes =
	    (uint64_t)product->matrix.shape.output_size * LINEAR_BFLOAT16_BYTES;
    result_status status = OK;
    b->rows = group_rows;
    struct linear_trunk_bfloat16_description leading =
	    *product; /* the first group's */
    leading.values.byte_count = group_rows * value_row_bytes;
    leading.output.byte_count = group_rows * output_row_bytes;
    leading.row_count = group_rows;
    if (trunk_mma_sets_put(b, launch, &leading, rows, &status)) {
        b->rows = rows;
        return status;
    }
    for (uint32_t first = 0; first < rows && status == OK; first += group_rows) {
	    struct linear_trunk_bfloat16_description group = *product;
	    group.values.byte_offset += first * value_row_bytes;
	    group.values.byte_count = group_rows * value_row_bytes;
	    group.output.byte_offset += first * output_row_bytes;
	    group.output.byte_count = group_rows * output_row_bytes;
	    group.row_count = group_rows;
	    struct kernel_launch *const slot = first == 0 ? launch :
							    slots(b, 1);
	    status = slot == NULL || !trunk_mma_wide_applies(b, &group) ?
			     STATUS_INVALID_ARGUMENT :
			     trunk_mma_wide(b, slot, &group);
    }
    b->rows = rows;
    return status;
}

MUST_CHECK static result_status put(struct builder *b,
				    const struct kernel_launch *launch)
{
	struct kernel_launch *const slot = slots(b, 1);
	if (slot == NULL) {
		LOG_ERROR("program_prepare: card %" PRIu32
			  " has more launches than its part of the program",
			  b->card);
		return STATUS_INVALID_ARGUMENT;
	}
    *slot = *launch;
    return OK;
}

/* Exact passes share a fixed arena annex; all descriptions are resolved before graph preparation. */
MUST_CHECK static result_status official_norm(struct builder *b,
					      struct linear_range input,
					      struct linear_range weight,
					      struct linear_range output,
					      uint32_t width)
{
	const uint64_t bytes = (uint64_t)width * V41_DATA_BF16;
	const struct official_norm_description description = {
		{ input.byte_offset, bytes },
		weight,
		{ output.byte_offset, bytes },
		{ b->plan->cards[b->card].official_scratch_offset +
			  MODEL_HEADS_ROW_BYTES,
		  OFFICIAL_NORM_STATISTICS * sizeof(float) },
		width,
		1
	};
	return official_norm_shape_prepare(&description, 1, b->arena,
					   b->official.norm, slots(b, 1));
}

MUST_CHECK static result_status official_roundtrip(struct builder *b,
						   struct linear_range values,
						   uint32_t width)
{
	const uint64_t bytes = (uint64_t)width * V41_DATA_BF16;
	const struct official_roundtrip_description description = {
		{ values.byte_offset, bytes },
		{ values.byte_offset, bytes },
		width,
		1
	};
	return official_roundtrip_prepare(&description, b->arena,
					  b->official.roundtrip, slots(b, 1));
}

/* The card's launches from first_launch on as a segment of layer; none, no segment. */
static void segment_add(struct builder *b, uint32_t layer, uint32_t first_launch)
{
	struct execution_program *const program = &b->made->program;
	const uint32_t count =
		b->made->cards[b->card].launch_count - first_launch;
	ASSERT(program->segment_count < MOST_SEGMENTS);
	if (count != 0 && program->segment_count < MOST_SEGMENTS) {
		b->made->segments[program->segment_count++] =
			(struct segment_record){ b->card,      b->stage, layer,
						 first_launch, count,	 0 };
	}
}

/* A hand-off from the builder's card to card to, behind the last segment (model_seam.h). */
static void hand_off(struct builder *b, uint32_t to, uint64_t source_offset, uint64_t destination_offset,
                     uint32_t byte_count, uint32_t entry_ratio, uint64_t session_stride)
{
	struct execution_program *const program = &b->made->program;
	ASSERT(program->hand_off_count < MOST_HAND_OFFS &&
	       program->segment_count != 0);
	if (program->hand_off_count < MOST_HAND_OFFS) {
		b->made->hand_offs[program->hand_off_count++] =
			(struct hand_off_record){ source_offset,
						  destination_offset,
						  session_stride,
						  byte_count,
						  entry_ratio,
						  b->card,
						  to,
						  program->segment_count - 1u,
						  0 };
	}
}

/* Behind the last segment: of the copied arrays (a bit each, COPIED_ARRAYS' order) the selection of an index layer,
 * the new pool entries and keys of a source, the candidates of the candidate source, to every other card that holds
 * a copy: 16 arrays of the model at most, each to at most every other card. */
static void copies_hand_off(struct builder *b, uint32_t layer, uint32_t copied)
{
	const struct model_plan_record *const plan = b->plan;
	const struct v41_card *const from = &plan->cards[b->card];
	const uint32_t ratio = (V41_DATA_POOLED_LAYERS >> layer & 1u) != 0 ?
				       MODEL_POOLED_RATIO :
				       MODEL_DIRECT_RATIO;
	const bool is_source = (V41_DATA_SOURCE_LAYERS >> layer & 1u) != 0;
	const bool writes[COPIED_ARRAYS] = {
		(V41_DATA_INDEX_LAYERS >> layer & 1u) != 0, is_source,
		is_source, layer == MODEL_CANDIDATE_SOURCE
	};
	const uint64_t sources[COPIED_ARRAYS] = {
		from->selection_offset[layer], from->pool_offset[layer],
		from->key_tiles_offset[layer],
		from->array_offset[V41_DATA_ARRAY_CANDIDATES]
	};
	const uint64_t strides[COPIED_ARRAYS] = {
		0, plan->pool_session_bytes[layer],
		plan->key_session_bytes[layer], 0
	};
	for (uint32_t to = 0; to < plan->card_count; to++) {
		const struct v41_card *const copy = &plan->cards[to];
		const uint64_t copies[COPIED_ARRAYS] = {
			copy->selection_offset[layer], copy->pool_offset[layer],
			copy->key_tiles_offset[layer],
			copy->array_offset[V41_DATA_ARRAY_CANDIDATES]
		};
		for (uint32_t array = 0; array < COPIED_ARRAYS && to != b->card;
		     array++) {
			if (writes[array] && (copied >> array & 1u) != 0 &&
			    copies[array] != PROGRAM_NOWHERE) {
				hand_off(b, to, sources[array], copies[array],
					 copied_bytes[array],
					 strides[array] == 0 ? 0 : ratio,
					 strides[array]);
			}
		}
    }
}

/* trunk_shape_slices: a thin step's trunk product on a CMP at the slice count measured fastest for its TP4 shard shape
 * (trunk01, gnmac_port: cold weights, 1-8 rows); the exact profile's ordered chains give every output the reference
 * chain at any count, so only the time changes. Other shapes, wide steps and cards keep `fallback`. */
struct tp_shape_slice {
    uint32_t output_size, input_size, slice_count;
};
static const struct tp_shape_slice tp_shape_slices_measured[] = { /* trunk01, us at 6 rows: S=4 -> chosen */
    {1280, 5120, 8},  /* wq_a 26.6 -> 21.5 */
    {512, 5120, 16},  /* wkv 25.6 -> 15.4 */
    {8192, 1280, 4},  /* wq_b shard 19.5 (fastest at 4) */
    {2048, 4096, 8},  /* wo_a shard 16.4 -> 13.3 */
    {5120, 2048, 4},  /* wo_b shard 20.5 (fastest at 4) */
    {4096, 1280, 4},  /* indexer wq_b 14.3 (fastest at 4) */
};
static uint32_t tp_shape_slices(const struct builder *b,
				const struct matrix_shape_record *shape,
				uint32_t fallback)
{
	if (b->tensor_layer == NULL || !b->tensor_layer->trunk_shape_slices ||
	    b->rows > LINEAR_MAXIMUM_ROWS ||
	    b->arithmetic_class != ARITHMETIC_CLASS_SM_80)
		return fallback;
	for (uint32_t row = 0;
	     row < sizeof(tp_shape_slices_measured) /
			   sizeof(tp_shape_slices_measured[0]);
	     row++)
		if (tp_shape_slices_measured[row].output_size ==
			    shape->output_size &&
		    tp_shape_slices_measured[row].input_size ==
			    shape->input_size)
			return tp_shape_slices_measured[row].slice_count;
	return fallback;
}

MUST_CHECK static result_status
slices_of(const struct builder *b, enum linear_family_code family,
	  const struct matrix_shape_record *shape, uint32_t *slice_count_out)
{
	if (b->tensor_layer != NULL && !b->tensor_layer->exact_output_shards &&
	    ((shape->output_size == MODEL_HEADS_VALUE * MODEL_HEAD_DIMS /
					    b->tensor_layer->rank_count &&
	      shape->input_size == MODEL_QUERY_RANK) ||
	     (shape->output_size ==
		      V41_DATA_WIDEST_ROW / b->tensor_layer->rank_count &&
	      shape->input_size == MODEL_HEADS_VALUE * MODEL_HEAD_DIMS /
					   MODEL_OUTPUT_GROUPS) ||
	     (shape->output_size == MODEL_HIDDEN &&
	      shape->input_size ==
		      V41_DATA_WIDEST_ROW / b->tensor_layer->rank_count) ||
	     family == LINEAR_FAMILY_EXPERT_GATE_UP ||
	     family == LINEAR_FAMILY_EXPERT_DOWN)) {
		*slice_count_out =
			family == LINEAR_FAMILY_EXPERT_GATE_UP ||
					family == LINEAR_FAMILY_EXPERT_DOWN ?
				b->tensor_slices :
				tp_shape_slices(b, shape, b->tensor_slices);
		return OK;
	}
	return linear_default_slice_count(b->plan->limits.linear_profile,
					  b->plan->limits.linear_order, family,
					  shape, b->arithmetic_class,
					  slice_count_out);
}

/* exact_dense (model_v41_tp2_stage.h): wq_b, wo_a and wo_b are output-axis shards of a full matrix. */
static bool is_dense_shard(const struct builder *b,
			   const struct v41_tensor *tensor)
{
    if (b->tensor_layer == NULL || !b->tensor_layer->exact_dense)
        return false;
    const struct v41_tensor *const local =
	    b->plan->layer_tensors[b->tensor_layer->layer_index];
    return tensor == &local[V41_DATA_QUERY_UP] ||
	   tensor == &local[V41_DATA_OUTPUT_GROUPS] ||
	   tensor == &local[V41_DATA_OUTPUT];
}

/* A dense shard's slice count: the full shape's measured count times the ranks, at most the stamped maximum and the
 * loads of a row (the class-80 exact profile's ordered slices give every output the reference chain at any count,
 * linear.c); the wide form is the full shape's. */
MUST_CHECK static result_status
dense_shard_matrix(const struct builder *b, enum linear_family_code family,
		   struct linear_matrix *matrix_out)
{
	const struct model_limits *const limits = &b->plan->limits;
	struct matrix_shape_record full = matrix_out->shape;
	full.output_size *= b->tensor_layer->rank_count;
	if (b->rows > LINEAR_MAXIMUM_ROWS)
		return linear_default_wide_form(limits->linear_profile,
						limits->linear_order, family,
						&full, b->arithmetic_class,
						&matrix_out->slice_count,
						&matrix_out->wide_form);
	uint32_t slices = 0;
	const result_status status = linear_default_slice_count(
		limits->linear_profile, limits->linear_order, family, &full,
		b->arithmetic_class, &slices);
	if (status != OK)
		return status;
	uint32_t widened = slices * b->tensor_layer->rank_count;
	const uint32_t loads =
		matrix_out->shape.input_size / LINEAR_VALUES_PER_LOAD;
	while (widened > LINEAR_MAXIMUM_SLICES || widened > loads)
		widened /= 2u;
	matrix_out->slice_count = widened > slices ? widened : slices;
	matrix_out->wide_form = LINEAR_WIDE_GROUP_BLOCKS;
	return OK;
}

/* A matrix of the layer for the rows the next launches take: its default count, or above a slab's rows the count and
 * form of its wide launches (linear.h: the fastest measured, the same bits). */
MUST_CHECK static result_status matrix_of(const struct builder *b,
					  const struct v41_tensor *tensor,
					  enum linear_family_code family,
					  struct linear_matrix *matrix_out)
{
	const struct model_limits *const limits = &b->plan->limits;
	*matrix_out = (struct linear_matrix){
		tensor->stored, tensor->shape, tensor->weight_format, 0, 0, 0
	};
	if (is_dense_shard(b, tensor))
		return dense_shard_matrix(b, family, matrix_out);
	if (b->tensor_layer != NULL && !b->tensor_layer->exact_output_shards) {
		const struct v41_tensor *const local =
			b->plan->layer_tensors[b->tensor_layer->layer_index];
		if (tensor == &local[V41_DATA_SHARED_GATE] ||
		    tensor == &local[V41_DATA_SHARED_UP] ||
		    tensor == &local[V41_DATA_SHARED_DOWN]) {
			matrix_out->slice_count = b->tensor_slices;
			matrix_out->wide_form = LINEAR_WIDE_GROUP_BLOCKS;
			return OK;
		}
    }
    if (b->rows > LINEAR_MAXIMUM_ROWS && b->tensor_layer != NULL) {
	    matrix_out->wide_form = LINEAR_WIDE_GROUP_BLOCKS;
	    return slices_of(b, family, &tensor->shape,
			     &matrix_out->slice_count);
    }
    return b->rows <= LINEAR_MAXIMUM_ROWS ?
		   slices_of(b, family, &tensor->shape,
			     &matrix_out->slice_count) :
		   linear_default_wide_form(limits->linear_profile,
					    limits->linear_order, family,
					    &tensor->shape, b->arithmetic_class,
					    &matrix_out->slice_count,
					    &matrix_out->wide_form);
}

/* A product of dense_linear.h; its weights are float32 where the tensor holds 4 bytes a weight, else bfloat16. */
/* The step rows of the selected rows for a cone (relaxed_linear.h, dense_linear.h), from its first row: the step
 * area's lie first_row rows in already (a chunk array), a half's view holds the half's rows from its first; false for
 * a slab's view or the draft's step. */
static bool cone_step_of(const struct builder *b, struct linear_range *step_out)
{
    if (b->reads_draft_step || (b->view != 0 && (b->view != b->chunk_view || b->first_row < b->chunk_first)))
        return false;
    const uint64_t into = b->view != 0 ?
				  (uint64_t)(b->first_row - b->chunk_first) *
					  STEP_ROW_BYTES_VALUE :
				  0u;
    *step_out = (struct linear_range){
	    ARRAY(STEP).byte_offset + into,
	    STEP_BLOCK_BYTES_VALUE + (uint64_t)b->rows * STEP_ROW_BYTES_VALUE
    };
    return true;
}

/* A dense product; with cone_layer (a full program's decoder layer) and a wide launch, DENSE_CONE_ROWS. */
MUST_CHECK static result_status
dense_cone(struct builder *b, const struct v41_tensor *weights,
	   struct linear_range rows, struct linear_range output,
	   uint32_t output_type, uint32_t flags, float output_scale,
	   uint32_t cone_layer)
{
	const struct matrix_shape_record *const shape = &weights->shape;
	const bool is_float32 =
		weights->stored.byte_count ==
		linear_sums_byte_count(shape->output_size, shape->input_size);
	struct linear_range step =
		(flags & DENSE_STEP_ROWS) != 0 ? ARRAY(STEP) : no_range;
	const bool cone = cone_layer != 0 && b->rows > LINEAR_MAXIMUM_ROWS &&
			  (flags & DENSE_STEP_ROWS) == 0 &&
			  cone_step_of(b, &step);
	const struct dense_linear_description description = {
		weights->stored,
		rows,
		output,
		step,
		shape->output_size,
		shape->input_size,
		b->rows,
		is_float32 ? F32 : BF16,
		output_type,
		flags | (cone ? DENSE_CONE_ROWS : 0u),
		output_scale,
		(flags & DENSE_ROW_SCALE) != 0 ? MODEL_NORM_EPSILON : 0.0f,
		cone ? cone_layer : 0u
	};
	struct kernel_launch *const launch = slots(b, 1);
	const result_status status =
		dense_linear_prepare(&description, b->arena, &b->dense, launch);
	if (status == OK && b->plan->limits.official_reference != 0 &&
	    shape->input_size == MODEL_HIDDEN &&
	    shape->output_size == MODEL_HEAD_DIMS && flags == 0) {
		launch->kernel = is_float32 ? b->official.projection_float32 :
					      b->official.projection_bfloat16;
		launch->grid[0] =
			shape->output_size / OFFICIAL_PROJECTION_OUTPUTS_VALUE;
		launch->grid[1] = launch->grid[LAUNCH_DIMENSIONS - 1u] = 1;
		launch->block[0] = OFFICIAL_PROJECTION_OWNERS_VALUE;
		launch->block[1] = OFFICIAL_PROJECTION_OUTPUTS_VALUE;
		launch->block[LAUNCH_DIMENSIONS - 1u] = 1;
		launch->dynamic_shared_byte_count = 0;
	}
    return status;
}

MUST_CHECK static result_status
dense(struct builder *b, const struct v41_tensor *weights,
      struct linear_range rows, struct linear_range output,
      uint32_t output_type, uint32_t flags, float output_scale)
{
    return dense_cone(b, weights, rows, output, output_type, flags, output_scale, 0);
}

MUST_CHECK static result_status
quantize_into(struct builder *b, struct linear_range values, uint32_t width,
	      struct linear_range codes, struct linear_range exponents,
	      struct linear_range operands)
{
	const struct linear_quantize_description description = {
		values, codes, exponents, operands, width, b->rows
	};
	return linear_quantize_prepare(&description, b->arena, &b->linear,
				       slots(b, 1));
}

MUST_CHECK static result_status
quantize(struct builder *b, struct linear_range values, uint32_t width)
{
    return quantize_into(b, values, width, ARRAY(ROW_CODES), ARRAY(ROW_EXPONENTS), ARRAY(ROW_OPERANDS));
}

/* trunk_cone_layer: a launch of the wide trunk form leaves the rows the engine's prefill cone marks past that layer
 * (relaxed_linear.h; any other form computes every row), with the selection's step rows (cone_step_of); only on a
 * card of the relaxed module's class (card_select prepares b->relaxed there alone). */
MUST_CHECK static result_status trunk_cone_at(struct builder *b,
					      struct kernel_launch *launch,
					      uint32_t cone_layer)
{
	struct linear_range step = no_range;
	return cone_layer == 0 ||
			       b->arithmetic_class != ARITHMETIC_CLASS_SM_80 ||
			       !cone_step_of(b, &step) ?
		       OK :
		       relaxed_trunk_cone_set(&b->relaxed, step, cone_layer,
					      b->arena, launch);
}

MUST_CHECK static result_status trunk_cone(struct builder *b,
					   struct kernel_launch *launch)
{
    return trunk_cone_at(b, launch, b->trunk_cone_layer);
}

/* y = W x of the quantized rows for one matrix, or for two in one launch (their slice counts must agree). */
MUST_CHECK static result_status
trunk_of(struct builder *b, struct linear_range codes,
	 struct linear_range exponents, struct linear_range operands,
	 const struct v41_tensor *first, struct linear_range first_output,
	 const struct v41_tensor *second, struct linear_range second_output)
{
	struct linear_trunk_quantized_description description = {
		.codes = codes,
		.exponents = exponents,
		.operands = operands,
		.outputs = { first_output, second_output },
		.matrix_count = second == NULL ? 1u : LINEAR_MERGED_MATRICES,
		.row_count = b->rows
	};
	result_status status =
		matrix_of(b, first, TRUNK, &description.matrices[0]);
	status = second == NULL ? status :
				  THEN(matrix_of(b, second, TRUNK,
						 &description.matrices[1]));
	const int32_t folds[LINEAR_MERGED_MATRICES] = {
		first->fold_reference,
		second == NULL ? 0 : second->fold_reference
	};
	struct kernel_launch *const launch = slots(b, 1);
	if (b->plan->limits.official_reference != 0) {
		const bool native =
			b->arithmetic_class == ARITHMETIC_CLASS_SM_120_NATIVE &&
			second == NULL &&
			(first->shape.output_size ==
				 MODEL_HEADS_VALUE * MODEL_HEAD_DIMS ||
			 first->shape.input_size == V41_DATA_WIDEST_ROW);
		status = THEN(official_linear_quantized_prepare(
			&description, b->arena, &b->official.linear,
			native ? OFFICIAL_LINEAR_NATIVE :
				 OFFICIAL_LINEAR_PORTABLE,
			launch));
	} else {
		status = THEN(linear_trunk_quantized_select_prepare(
			&description, b->arena, &b->linear, &b->relaxed,
			b->arithmetic_class, b->plan->limits.cmp_order, folds,
			launch));
	}
    return THEN(trunk_cone(b, launch));
}

MUST_CHECK static result_status trunk(struct builder *b,
				      const struct v41_tensor *first,
				      struct linear_range first_output,
				      const struct v41_tensor *second,
				      struct linear_range second_output)
{
    return trunk_of(b, ARRAY(ROW_CODES), ARRAY(ROW_EXPONENTS), ARRAY(ROW_OPERANDS), first, first_output,
                    second, second_output);
}

/* Preserve the BF16 inputs of EXL3 attention, with the sibling's four ordered product slices. */
MUST_CHECK static result_status
exl3_trunk_bfloat16(struct builder *b, const struct v41_tensor *matrix,
		    struct linear_range values, struct linear_range output)
{
	const struct linear_trunk_bfloat16_description description = {
		.matrix = { matrix->stored, matrix->shape,
			    matrix->weight_format,
			    tp_shape_slices(b, &matrix->shape,
					    EXL3_ATTENTION_SLICES),
			    0, LINEAR_WIDE_GROUP_BLOCKS },
		.values = values,
		.output = output,
		.row_count = b->rows,
		.segment_count = 1
	};
	struct kernel_launch *const launch = slots(b, 1);
	/* ordered_wide: more than 8 rows by the ordered wide form (the reference's bits at any row count, the tensor
     * cores) in place of the wide-four chains */
	const bool ordered =
		b->tensor_layer != NULL && b->tensor_layer->ordered_wide &&
		b->rows > LINEAR_MAXIMUM_ROWS && b->slab_count > 1u;
	const result_status status =
		ordered ? linear_trunk_bfloat16_select_prepare(
				  &description, b->arena, &b->linear,
				  &b->relaxed, b->arithmetic_class,
				  CMP_ORDER_REFERENCE, 0, launch) :
			  linear_trunk_bfloat16_prepare(&description, b->arena,
							&b->linear, launch);
	if (status == OK)
		trunk_mma_swap(b, launch, &matrix->shape);
	const result_status wide =
		status == OK ? trunk_mma_wide_rows(b, launch, &description) :
			       status;
	return wide == OK ? trunk_cone(b, launch) : wide;
}

/* The attention output's grouped product (wo_a, block-diagonal over the groups), trunk_mma where it applies. */
MUST_CHECK static result_status
groups_product(struct builder *b,
	       const struct linear_trunk_bfloat16_description *groups,
	       int32_t fold_reference)
{
	struct kernel_launch *const launch = slots(b, 1);
	if (b->plan->limits.official_reference != 0)
		return official_linear_woa_prepare(groups, b->arena,
						   &b->official.linear, launch);
	const result_status status = linear_trunk_bfloat16_select_prepare(
		groups, b->arena, &b->linear, &b->relaxed, b->arithmetic_class,
		b->plan->limits.cmp_order, fold_reference, launch);
	if (status == OK && b->plan->limits.cmp_order == CMP_ORDER_REFERENCE)
		trunk_mma_swap(b, launch, &groups->matrix.shape);
	const result_status wide =
		status == OK && b->plan->limits.cmp_order ==
					CMP_ORDER_REFERENCE ?
			trunk_mma_wide_rows(b, launch, groups) :
			status;
	return wide == OK ? trunk_cone(b, launch) : wide;
}

/* A site's launches (hyper_connection.h) for the rows the next launches take, into launches. */
MUST_CHECK static result_status site_prepare(struct builder *b,
					     const struct v41_tensor *weights,
					     bool is_ffn,
					     struct kernel_launch *launches)
{
	const struct hyper_site_description description = {
		ARRAY(STREAMS),
		ARRAY(CARRIED_PRE),
		ARRAY(MIXES),
		is_ffn ? WEIGHTS(FFN_MIX_SCALE)->stored :
			 WEIGHTS(ATTENTION_MIX_SCALE)->stored,
		is_ffn ? WEIGHTS(FFN_MIX_BASE)->stored :
			 WEIGHTS(ATTENTION_MIX_BASE)->stored,
		is_ffn ? WEIGHTS(FFN_NORM)->stored :
			 WEIGHTS(ATTENTION_NORM)->stored,
		ARRAY(POST),
		ARRAY(COMB),
		ARRAY(HIDDEN),
		is_ffn ? ARRAY(SHARED_OUTPUT) : ARRAY(SUBLAYER_OUTPUT),
		is_ffn && b->tensor_layer == NULL ? ARRAY(ROW_SUMS) : no_range,
		b->rows,
		0
	};
	const result_status status =
		hyper_site_prepare(&description, b->arena, &b->hyper, launches);
	if (status == OK && b->plan->limits.official_reference != 0) {
		launches[HYPER_SITE_MIX_FINISH].kernel = b->official.site;
		launches[HYPER_SITE_MIX_FINISH].block[0] =
			OFFICIAL_SITE_THREADS_VALUE;
	}
    return status;
}

/* A site's mix project and mix finish (REF:982-984, 989-991); its post, for after the sublayer, in launches. */
MUST_CHECK static result_status site_begin(struct builder *b,
					   const struct v41_tensor *weights,
					   bool is_ffn,
					   struct kernel_launch *launches)
{
	const struct v41_tensor *const mix = is_ffn ? WEIGHTS(FFN_MIX) :
						      WEIGHTS(ATTENTION_MIX);
	/* a full program's decoder layer: the attention site's mix feeds the layer's window rows (the trunks' cone), the
     * FFN site's only the layer's output (the layer's own) */
	result_status status;
	if (b->plan->limits.official_hc_rows != 0) {
		const struct dense_linear_description description = {
			mix->stored,
			ARRAY(STREAMS),
			ARRAY(MIXES),
			no_range,
			MODEL_MIXES,
			OFFICIAL_HC_INPUTS,
			b->rows,
			F32,
			F32,
			DENSE_ROW_SCALE,
			1.0f,
			MODEL_NORM_EPSILON,
			0
		};
		status = official_hc_shape_prepare(
			&description, b->plan->limits.official_hc_rows,
			b->arena, &b->dense, &b->official_hc, slots(b, 1));
	} else {
		status = dense_cone(b, mix, ARRAY(STREAMS), ARRAY(MIXES), F32,
				    DENSE_ROW_SCALE, 1.0f,
				    is_ffn ? b->ffn_cone_layer :
					     b->trunk_cone_layer);
	}
    status = THEN(site_prepare(b, weights, is_ffn, launches));
    return THEN(put(b, &launches[HYPER_SITE_MIX_FINISH]));
}

/* The fused front retains the dense mix's order; the draft's smaller router remains a separate launch. kind: the
 * front's (the site front, or its input front or coefficient front: site_front_records.h); the FFN front by router. */
MUST_CHECK static result_status
site_front_of(struct builder *b, const struct v41_tensor *weights, bool is_ffn,
	      const struct router_description *router,
	      enum site_front_kind kind)
{
	const struct site_front_description front = {
		.streams = ARRAY(STREAMS),
		.carried_pre = ARRAY(CARRIED_PRE),
		.mix_weights =
			(is_ffn ? WEIGHTS(FFN_MIX) : WEIGHTS(ATTENTION_MIX))
				->stored,
		.scale = (is_ffn ? WEIGHTS(FFN_MIX_SCALE) :
				   WEIGHTS(ATTENTION_MIX_SCALE))
				 ->stored,
		.base = (is_ffn ? WEIGHTS(FFN_MIX_BASE) :
				  WEIGHTS(ATTENTION_MIX_BASE))
				->stored,
		.norm_weight =
			(is_ffn ? WEIGHTS(FFN_NORM) : WEIGHTS(ATTENTION_NORM))
				->stored,
		.post = ARRAY(POST),
		.comb = ARRAY(COMB),
		.hidden = ARRAY(HIDDEN),
		.codes = ARRAY(ROW_CODES),
		.exponents = ARRAY(ROW_EXPONENTS),
		.operands = ARRAY(ROW_OPERANDS),
		.scratch = { b->plan->cards[b->card].front_scratch_offset,
			     SITE_FRONT_SCRATCH_BYTES },
		.router = router,
		.router_weights = router != NULL ? WEIGHTS(ROUTER)->stored :
						   no_range,
		.kind = router != NULL ? SITE_FRONT_KIND_FFN : kind,
		.row_count = b->rows
	};
	return site_front_prepare(&front, b->arena,
				  router != NULL ? &b->router : NULL,
				  &b->fronts, slots(b, 1));
}
MUST_CHECK static result_status
site_front(struct builder *b, const struct v41_tensor *weights, bool is_ffn,
	   const struct router_description *router)
{
	return site_front_of(b, weights, is_ffn, router, SITE_FRONT_KIND_SITE);
}

/* front_branch: the launch at side (the coefficient front) a side run of one launch, the one behind it the resume,
 * the card's last (the site's post) the join; where any launch between holds a branch of its own, all stay a chain. */
static void front_branch_mark(struct builder *b, uint32_t side)
{
	struct kernel_launch *const launches = b->launches[b->card];
	const uint32_t last = b->made->cards[b->card].launch_count - 1u;
	bool plain = side + 1u < last;
	for (uint32_t launch = side; launch <= last && plain; launch++)
		plain = launches[launch].branch == LAUNCH_CHAIN;
	if (!plain)
		return;
	launches[side].branch = LAUNCH_SIDE;
	launches[side + 1u].branch = LAUNCH_RESUME;
	launches[last].branch = LAUNCH_JOIN;
}

/* The fusion uses the same product order. Preserve an admitted CMP relaxed product instead. */
static bool trunk_is_relaxed(const struct builder *b,
			     const struct v41_tensor *tensor)
{
	return b->arithmetic_class == ARITHMETIC_CLASS_SM_80 &&
	       b->plan->limits.cmp_order == CMP_ORDER_RELAXED &&
	       tensor->fold_reference != 0;
}

MUST_CHECK static result_status
tensor_row_copy(struct builder *b, struct linear_range source,
		struct linear_range output, uint32_t width,
		uint32_t source_stride, uint32_t output_stride)
{
	const struct tp2_join_description description = {
		.source = source,
		.output = output,
		.width = width,
		.row_count = b->rows,
		.source_row_stride = source_stride,
		.output_row_stride = output_stride,
		.operation = TP2_JOIN_BFLOAT16_COPY
	};
	return tp2_join_prepare(&description, b->arena, b->tensor_join_kernel,
				slots(b, 1));
}

MUST_CHECK static result_status exact_gather(struct builder *b, uint32_t site,
					     struct linear_range input,
					     struct linear_range output,
					     uint32_t bytes, uint32_t vectors);
MUST_CHECK static result_status
exact_gather_bank(struct builder *b, uint32_t site, uint64_t ring_offset,
		  uint64_t bank_bytes, uint64_t row_bytes,
		  struct linear_range input, struct linear_range output,
		  uint32_t bytes, uint32_t vectors);

/* exact_dense: rank r's owner-major slot of the gather input, rows of `width` bfloat16 values. */
static struct linear_range dense_slot(const struct builder *b, uint32_t width)
{
	const uint64_t bytes =
		(uint64_t)b->rows * width * LINEAR_BFLOAT16_BYTES;
	return (struct linear_range){ b->tensor_layer->dense_input.byte_offset +
					      b->tensor_layer->rank * bytes,
				      bytes };
}

/* exact_dense: every rank's slot of `width` columns gathered into the row-major output of width x ranks columns. */
MUST_CHECK static result_status dense_gather(struct builder *b, uint32_t site,
					     uint32_t width,
					     struct linear_range output)
{
	const struct v41_tp2_layer_plan *const tensor = b->tensor_layer;
	const uint32_t bytes =
		b->rows * width * tensor->rank_count * LINEAR_BFLOAT16_BYTES;
	const struct linear_range input = { tensor->dense_input.byte_offset,
					    bytes };
	return exact_gather_bank(b, site, tensor->dense_ring_offsets[site],
				 tensor->dense_ring_bank_bytes[site], 0, input,
				 output, bytes, b->rows);
}

/* exact_dense, more than 8 rows (a chunked step's piece): the rank's own outputs [rows][width] behind the slot region
 * of the thin gathers (that region the guard below the virtual bases). */
#define DENSE_SLOT_REGION_BYTES                                \
	((uint64_t)LINEAR_MAXIMUM_ROWS * V41_DATA_WIDEST_ROW * \
	 LINEAR_BFLOAT16_BYTES)
static struct linear_range dense_own(const struct builder *b, uint32_t width)
{
	return (struct linear_range){ b->tensor_layer->dense_input.byte_offset +
					      DENSE_SLOT_REGION_BYTES,
				      (uint64_t)b->rows * width *
					      LINEAR_BFLOAT16_BYTES };
}

/* exact_dense, a piece: each slab of 8 rows (the last its remaining rows) gathered from every rank's own outputs into
 * its rows of the row-major output. A rank reads only its own range of a gather's owner-major input, so its input
 * base is its slab's own rows less its owner prefix (inside the slot region's guard). */
MUST_CHECK static result_status dense_gather_slabs(struct builder *b,
						   uint32_t site,
						   uint32_t width,
						   struct linear_range output)
{
	const struct v41_tp2_layer_plan *const tensor = b->tensor_layer;
	const uint32_t rows = b->rows;
	const uint64_t own_row = (uint64_t)width * LINEAR_BFLOAT16_BYTES;
	const struct linear_range own = dense_own(b, width);
	if ((uint64_t)tensor->rank * LINEAR_MAXIMUM_ROWS * own_row >
		    DENSE_SLOT_REGION_BYTES ||
	    output.byte_count < (uint64_t)rows * own_row * tensor->rank_count)
		return STATUS_INVALID_ARGUMENT;
	result_status status = OK;
	for (uint32_t at = 0; at < rows && status == OK;
	     at += LINEAR_MAXIMUM_ROWS) {
		const uint32_t count = rows - at < LINEAR_MAXIMUM_ROWS ?
					       rows - at :
					       LINEAR_MAXIMUM_ROWS;
		const uint32_t bytes =
			(uint32_t)(count * own_row * tensor->rank_count);
		const struct linear_range input = {
			own.byte_offset + at * own_row -
				(uint64_t)tensor->rank * count * own_row,
			bytes
		};
		const struct linear_range into = {
			output.byte_offset + at * own_row * tensor->rank_count,
			bytes
		};
		b->rows = count; /* a gather's bank is its slab's */
		status = exact_gather_bank(b, site,
					   tensor->dense_ring_offsets[site],
					   tensor->dense_ring_bank_bytes[site],
					   0, input, into, bytes, count);
	}
    b->rows = rows;
    return status;
}


/* Local product rounding is part of the relaxed TP contract. The unchanged post reads a complete BF16 result. */
/* The sums' phase of the layer (tp_exchange_records.h): 8 l + 2, and 8 l + 4 in split_overlap's second half; its
 * pipeline's gathers below and between them, each on the channel no other exchange uses beside it: the index gathers
 * (half A: attention site, half B: FFN site), the router's of half A (FFN site, beside half B's attention sums) and of
 * half B (attention site, beside half A's FFN sums). */
#define SUMS_PHASE 2u
#define GATHER_INDEX_PHASE 0u
#define GATHER_ROUTER_FIRST_PHASE 1u
#define GATHER_ROUTER_SECOND_PHASE 5u
/* pool_gather's of a half, at the layer before's phase 6 (behind its channel's sums and router gather, ahead of this
 * layer's index gather): half A on the attention site's channel (beside B's FFN sums of the layer before), half B on
 * the FFN site's (beside A's attention sums) */
#define GATHER_POOL_PHASE 6u
/* the router's gather from halves of so many rows (pf29: 512-row halves +0.9%, its sync ahead of the experts costing
 * more than a quarter's product saves; 1,024-row halves -1.1%) */
#define ROUTER_GATHER_LEAST_ROWS 1024u
static uint32_t exchange_phase(const struct builder *b)
{
	return TP_EXCHANGE_LAYER_PHASES * b->tensor_layer->layer_index +
	       SUMS_PHASE +
	       (b->half_view_first != 0 &&
				b->chunk_view == b->half_view_first + 1u ?
			2u :
			0u);
}

/* A copy engine's copy of bytes from the card's arena to a place in it (word EXCHANGE_OWN_ARENA) or in the
 * arena a peer_connect address word names (backend_seam.h). */
MUST_CHECK static result_status copy_put(struct builder *b, uint64_t source,
					 uint64_t word, uint64_t offset,
					 uint64_t bytes, uint32_t branch)
{
	const struct exchange_copy_arguments copy = {
		{ EXCHANGE_OWN_ARENA, source }, { word, offset }, bytes
	};
	struct kernel_launch launch = { .kernel = BACKEND_COPY,
					.argument_byte_count = sizeof(copy),
					.branch = branch };
	memcpy(launch.argument, &copy, sizeof(copy));
	return put(b, &launch);
}

/* The branch of the step-th of an exchange's copies (rank + 1, rank + 2, ..., the rank's own last): the peers' side
 * runs at once (SIDE, then SIDE_MORE), the last on the main run (RESUME); the barrier behind them JOINs. */
static uint32_t copy_branch(uint32_t step, uint32_t steps)
{
	return step + 1u == steps ? LAUNCH_RESUME :
	       step == 0	  ? LAUNCH_SIDE :
				    LAUNCH_SIDE_MORE;
}

/* split_overlap's pipeline's gathers apply to the selected half: copy-engine sums (direct routes, four ranks, BF16) on
 * both sites' channels and whole quarters of slabs. */
static bool gathers_apply(const struct builder *b)
{
	const struct v41_tp2_layer_plan *const tensor = b->tensor_layer;
	bool apply = tensor != NULL &&
		     tensor->split_overlap >= V41_DATA_SPLIT_PIPELINE &&
		     b->half_view_first != 0 &&
		     b->chunk_view >= b->half_view_first &&
		     tensor->shared_channels && !b->tensor_local_stub &&
		     !tensor->parallel_publish &&
		     tensor->wide_exchange >=
			     V41_DATA_TP_EXCHANGE_CE_OWNER_SUMS &&
		     tensor->rank_count == TP4_RANKS &&
		     b->chunk_rows % (TP4_RANKS * LINEAR_MAXIMUM_ROWS) == 0;
	for (uint32_t site = 0; site < TP2_SITES && apply; site++)
		apply = tensor->collectives[site].arguments.incoming_host_mask ==
				0 &&
			tensor->collectives[site].arguments.payload_format ==
				ORDERED_PUSH_BF16;
	return apply;
}

/* A gathered range: bytes of the rank's rows at own in this card's arena, copied to peers[r] in rank r's arena. */
struct gather_range {
    uint64_t own, bytes;
    uint64_t peers[TP4_RANKS];
};

/* A gather of the selected half's rows: each range's copies to every peer (on the main run: no fork beside a split
 * part's), then READY and DONE on the site's channel at exchange phase `phase` (tp_exchange_records.h: 8 l + p): every
 * rank holds every rank's rows. */
MUST_CHECK static result_status rows_gather(struct builder *b, uint32_t site,
					    uint32_t phase,
					    const struct gather_range *ranges,
					    uint32_t range_count)
{
	const struct v41_tp2_layer_plan *const tensor = b->tensor_layer;
	struct ordered_push_description collective = tensor->collectives[site];
	struct ordered_push_arguments *const arguments = &collective.arguments;
	arguments->element_count = b->rows * MODEL_HIDDEN;
	arguments->step_offset = ARRAY(STEP).byte_offset;
	struct kernel_launch wide[TP_EXCHANGE_WIDE_LAUNCHES];
	result_status status = tp_exchange_wide_prepare(
		&collective, phase, TP_EXCHANGE_PHASES, b->tensor_ce_kernels,
		TP_EXCHANGE_FORM_CE_OWNER_SUMS, wide);
	const uint64_t word_of_rank =
		arguments->peer_table_offset +
		offsetof(struct ordered_collective_peer, arena_address);
	for (uint32_t step = 1; step < TP4_RANKS && status == OK; step++) {
		const uint32_t rank = (tensor->rank + step) % TP4_RANKS;
		for (uint32_t range = 0; range < range_count && status == OK;
		     range++)
			status =
				ranges[range].bytes == 0 ?
					OK :
					copy_put(
						b, ranges[range].own,
						word_of_rank +
							rank * sizeof(struct ordered_collective_peer),
						ranges[range].peers[rank],
						ranges[range].bytes,
						LAUNCH_CHAIN);
	}
    status = THEN(put(b, &wide[1])); /* READY */
    return THEN(put(b, &wide[3]));   /* DONE */
}

/* The router's scores of the selected half, a quarter of its rows a rank (each row's by the product every rank runs),
 * gathered: split_overlap's pipeline. */
MUST_CHECK static result_status router_gather(struct builder *b,
					      const struct v41_tensor *router);
static void gather_share(const struct builder *b, uint32_t units, uint32_t granule, uint32_t *first_out,
                         uint32_t *count_out);

/* prefill_dead_halves (model_v41_tp2_stage.h full_cone_behind): half h (0 or 1) of a full program of
 * MODEL_FULL_CONE_LEAST_ROWS rows or more has no row whose output at decoder layer `layer` the cone reads: its
 * last row (A's: B's rows nearer the end) lies at least the bound before its prompt's cone end's last row, so its
 * rows' cone counts are at most that row's (model_layout.h MODEL_CONE_LAYERS). */
static uint32_t cone_bound_of(const struct builder *b, uint32_t layer)
{
    const uint32_t behind = b->tp_description == NULL || b->tensor_layer == NULL ? 0u :
                            b->tp_description->stages[b->card]->description.full_cone_behind;
    /* whole granules of 256 rows: the exact plan's chunks before a tail (a tail program's lie in the tail) */
    return b->made->program.kind != PROGRAM_FULL ||
			   layer < MODEL_DECODER_FIRST_LAYER ||
			   layer >= MODEL_LAYERS ||
			   b->program_rows < MODEL_FULL_CONE_LEAST_ROWS ||
			   b->program_rows % MODEL_FULL_CONE_LEAST_ROWS != 0 ||
			   b->half_view_first == 0 ?
		   0u :
		   behind;
}
static bool half_dead(const struct builder *b, uint32_t layer, uint32_t half)
{
    const uint32_t behind = cone_bound_of(b, layer);
    const uint32_t after = half == 0 ? b->program_rows - b->half_rows : 0u;
    return behind != 0 && half <= 1u &&
	   MODEL_CONE_LAYERS(behind + after) <=
		   layer - MODEL_DECODER_FIRST_LAYER;
}

/* prefill_dead_halves 2 (model_v41_tp2_stage.h dead_rows_trimmed): of the selected rows of a split decoder layer's
 * half, the first whose output at the tensor layer the cone may read (half_dead's bound; 0: every row's). */
static uint32_t live_first_row(const struct builder *b)
{
    const uint32_t layer = b->tensor_layer == NULL ? 0u : b->tensor_layer->layer_index;
    const uint32_t behind = cone_bound_of(b, layer);
    if (behind == 0 || !b->tp_description->stages[b->card]->description.dead_rows_trimmed ||
        b->chunk_view < b->half_view_first)
        return 0;
    uint32_t row = 0;
    while (row < b->rows && MODEL_CONE_LAYERS(behind + b->program_rows - 1u -
					      (b->first_row + row)) <=
				    layer - MODEL_DECODER_FIRST_LAYER)
	    row++;
    return row;
}

/* The selected half of split_overlap's chunk: 0 (A) or 1 (B); 2 without one selected. */
static uint32_t selected_half(const struct builder *b)
{
    return b->half_view_first != 0 && b->chunk_view >= b->half_view_first ? b->chunk_view - b->half_view_first : 2u;
}

/* pool_gather (model_v41_tp2_stage.h) applies to the selected half: split_overlap's pipeline's gathers, its whole rows
 * selected (one piece). */
static bool pool_gather_applies(const struct builder *b)
{
    return b->tensor_layer != NULL && b->tp_description != NULL && b->tensor_layer->layer_index != 0 &&
           b->tp_description->stages[b->card]->description.pool_gather && gathers_apply(b) &&
           b->first_row == b->chunk_first && b->rows == b->chunk_rows;
}

/* pool_gather: rows first .. first + rows of the compressor's products (wkv packed at kv_row_bytes, wgate's scores)
 * gathered from the ranks that computed them (gather_share's parts). */
MUST_CHECK static result_status pool_rows_gather(struct builder *b,
						 uint64_t kv_row_bytes,
						 uint32_t first, uint32_t rows,
						 bool scores)
{
	const struct v41_tp2_layer_plan *const tensor = b->tensor_layer;
	static const enum v41_array arrays[] = {
		V41_DATA_ARRAY_COMPRESSOR_KV, V41_DATA_ARRAY_COMPRESSOR_SCORE
	};
	struct gather_range ranges[sizeof(arrays) / sizeof(arrays[0])];
	for (uint32_t range = 0; range < sizeof(arrays) / sizeof(arrays[0]);
	     range++) {
		const uint64_t row_bytes =
			range == 0 ? kv_row_bytes :
				     array_bytes[arrays[range]].row_bytes;
		const uint64_t into = (uint64_t)first * row_bytes;
		ranges[range] = (struct gather_range){
			b->plan->cards[b->card].array_offset[arrays[range]] +
				into,
			range == 0 || scores ? (uint64_t)rows * row_bytes : 0u,
			{ 0 }
		};
		for (uint32_t rank = 0; rank < TP4_RANKS; rank++) {
			const uint32_t card =
				tensor->card - tensor->rank + rank;
			ranges[range].peers[rank] =
				b->tp_description->views[card]
					->cards[card]
					.array_offset[arrays[range]] +
				into;
		}
    }
    const bool second = b->chunk_view == b->half_view_first + 1u;
    return rows_gather(b, second ? 1u : 0u,
		       TP_EXCHANGE_LAYER_PHASES * (tensor->layer_index - 1u) +
			       GATHER_POOL_PHASE,
		       ranges, sizeof(arrays) / sizeof(arrays[0]));
}

/* wide_exchange CE_OWNER_SUMS, the attention site (no routed sums: the BF16 values are the push's payload as they
 * are): rank d's quarter of the piece's prepared rows copied into every rank's inbox for it (slot 0, tp_exchange.h),
 * READY, the rank's quarter summed into its output array, that quarter copied into every other rank's output array,
 * DONE, and the BF16 rows copied back into values: the owner sums' bits, the bytes moved by copy engines. */
MUST_CHECK static result_status
tensor_complete_ce(struct builder *b, uint32_t site, struct linear_range values)
{
	const struct v41_tp2_layer_plan *const tensor = b->tensor_layer;
	struct ordered_push_description collective = tensor->collectives[site];
	struct ordered_push_arguments *const arguments = &collective.arguments;
	arguments->element_count = b->rows * MODEL_HIDDEN;
	arguments->step_offset = b->view != 0 ? ARRAY(STEP).byte_offset :
						arguments->step_offset;
	const uint64_t quarter = (uint64_t)b->rows / TP4_RANKS * MODEL_HIDDEN *
				 LINEAR_BFLOAT16_BYTES;
	const uint64_t word_of_rank =
		arguments->peer_table_offset +
		offsetof(struct ordered_collective_peer, arena_address);
	struct kernel_launch wide[TP_EXCHANGE_WIDE_LAUNCHES];
	result_status status = tp_exchange_wide_prepare(
		&collective, exchange_phase(b), TP_EXCHANGE_PHASES,
		b->tensor_ce_kernels, TP_EXCHANGE_FORM_CE_OWNER_SUMS, wide);
	/* dead_rows_trimmed: each quarter's rows from the half's first live one (live_first_row: a quarter of none
     * copies nothing; the side runs over the copies that remain) */
	const uint32_t live = live_first_row(b),
		       quarter_rows = b->rows / TP4_RANKS;
	uint64_t skipped[TP4_RANKS];
	uint32_t moved = 0;
	for (uint32_t rank = 0; rank < TP4_RANKS; rank++) {
		const uint32_t first = rank * quarter_rows;
		skipped[rank] = (uint64_t)(live <= first ? 0u :
					   live - first < quarter_rows ?
							   live - first :
							   quarter_rows) *
				MODEL_HIDDEN * LINEAR_BFLOAT16_BYTES;
		moved += skipped[rank] < quarter ? 1u : 0u;
	}
    /* the quarters into the inboxes */
    for (uint32_t step = 1, put_copies = 0; step <= TP4_RANKS && status == OK;
	 step++) {
	    const uint32_t rank = (tensor->rank + step) % TP4_RANKS;
	    if (skipped[rank] == quarter)
		    continue;
	    status = copy_put(
		    b, values.byte_offset + rank * quarter + skipped[rank],
		    rank == tensor->rank ?
			    EXCHANGE_OWN_ARENA :
			    word_of_rank +
				    rank * sizeof(struct ordered_collective_peer),
		    (rank == tensor->rank ? arguments->region_offset :
					    tensor->peer_regions[site][rank]) +
			    arguments->inbox_offset +
			    (uint64_t)tensor->rank *
				    arguments->inbox_rank_stride +
			    rank * quarter + skipped[rank],
		    quarter - skipped[rank],
		    moved > 1u ? copy_branch(put_copies, moved) : LAUNCH_CHAIN);
	    put_copies++;
    }
    wide[1].branch = moved > 1u ? LAUNCH_JOIN : LAUNCH_CHAIN;
    status = THEN(put(b, &wide[1])); /* READY */
    status = THEN(put(b, &wide[2])); /* the rank's quarter summed */
    const uint64_t own_skipped = skipped[tensor->rank];
    for (uint32_t step = 1;
	 step < TP4_RANKS && status == OK && own_skipped < quarter; step++) {
	    const uint32_t rank = (tensor->rank + step) % TP4_RANKS;
	    status = copy_put(
		    b,
		    arguments->region_offset + arguments->output_offset +
			    tensor->rank * quarter + own_skipped,
		    word_of_rank +
			    rank * sizeof(struct ordered_collective_peer),
		    tensor->peer_regions[site][rank] +
			    arguments->output_offset + tensor->rank * quarter +
			    own_skipped,
		    quarter - own_skipped,
		    copy_branch(step - 1u, TP4_RANKS - 1u));
    }
    wide[3].branch = own_skipped < quarter ? LAUNCH_JOIN : LAUNCH_CHAIN;
    status = THEN(put(b, &wide[3])); /* DONE */
    const struct tp2_join_description back = {
	    .source = { arguments->region_offset + arguments->output_offset,
			quarter * TP4_RANKS },
	    .output = values,
	    .width = MODEL_HIDDEN,
	    .row_count = b->rows,
	    .source_row_stride = MODEL_HIDDEN,
	    .output_row_stride = MODEL_HIDDEN,
	    .operation = TP2_JOIN_BFLOAT16_COPY
    };
    return THEN(tp2_join_prepare(&back, b->arena, b->tensor_join_kernel,
				 slots(b, 1)));
}

/* wide_exchange CE_RINGS, the FFN site of a piece of more than 8 rows: values plus the routed sums rounded to BF16
 * into the collective's input array in the ring's owner-major order (tp2_join's RING_PACK), each rank's segment d
 * copied into rank d's inbox (slot 0), READY, the rank's segment added in the ring's order into its output array,
 * copied into the other ranks' output arrays, DONE, and the rows unpacked back into values: each slab's BF16 ring
 * sum, the same bits, the bytes moved by copy engines. */
MUST_CHECK static result_status
tensor_complete_ce_rings(struct builder *b, uint32_t site,
			 struct linear_range values, struct linear_range routed)
{
	const struct v41_tp2_layer_plan *const tensor = b->tensor_layer;
	struct ordered_push_description collective = tensor->collectives[site];
	struct ordered_push_arguments *const arguments = &collective.arguments;
	arguments->element_count = b->rows * MODEL_HIDDEN;
	arguments->step_offset = b->view != 0 ? ARRAY(STEP).byte_offset :
						arguments->step_offset;
	const uint64_t segment = (uint64_t)b->rows * MODEL_HIDDEN / TP4_RANKS *
				 LINEAR_BFLOAT16_BYTES;
	const uint64_t staging =
		arguments->region_offset + arguments->input_offset;
	const uint64_t output =
		arguments->region_offset + arguments->output_offset;
	const uint64_t word_of_rank =
		arguments->peer_table_offset +
		offsetof(struct ordered_collective_peer, arena_address);
	struct kernel_launch wide[TP_EXCHANGE_WIDE_LAUNCHES];
	result_status status = tp_exchange_wide_prepare(
		&collective, exchange_phase(b), TP_EXCHANGE_PHASES,
		b->tensor_ce_ring_kernels, TP_EXCHANGE_FORM_CE_OWNER_RING_SUMS,
		wide);
	const struct tp2_join_description pack = {
		.source = values,
		.routed_sums = routed,
		.output = { staging, segment * TP4_RANKS },
		.width = MODEL_HIDDEN,
		.row_count = b->rows,
		.source_row_stride = MODEL_HIDDEN,
		.routed_row_stride = routed.byte_count == 0 ? 0 : MODEL_HIDDEN,
		.output_row_stride = MODEL_HIDDEN,
		.operation = TP2_JOIN_RING_PACK
	};
	status = THEN(tp2_join_prepare(&pack, b->arena, b->tensor_join_kernel,
				       slots(b, 1)));
	/* dead_rows_trimmed: each segment's slabs from the one of the half's first live row on (live_first_row) */
	const uint64_t skipped =
		(uint64_t)(live_first_row(b) / TP2_JOIN_RING_SLAB_ROWS) *
		TP2_JOIN_RING_SLAB_ROWS * MODEL_HIDDEN / TP4_RANKS *
		LINEAR_BFLOAT16_BYTES;
	if (skipped >= segment)
		return STATUS_INVALID_ARGUMENT; /* a half of no live row leaves its sums (half_dead) */
	for (uint32_t step = 1; step <= TP4_RANKS && status == OK;
	     step++) { /* the segments into the inboxes */
		const uint32_t rank = (tensor->rank + step) % TP4_RANKS;
		status = copy_put(
			b, staging + rank * segment + skipped,
			rank == tensor->rank ?
				EXCHANGE_OWN_ARENA :
				word_of_rank +
					rank * sizeof(struct ordered_collective_peer),
			(rank == tensor->rank ?
				 arguments->region_offset :
				 tensor->peer_regions[site][rank]) +
				arguments->inbox_offset +
				(uint64_t)tensor->rank *
					arguments->inbox_rank_stride +
				rank * segment + skipped,
			segment - skipped, copy_branch(step - 1u, TP4_RANKS));
	}
	wide[1].branch = LAUNCH_JOIN;
	status = THEN(put(b, &wide[1])); /* READY */
	status = THEN(put(b, &wide[2])); /* the rank's segment added */
	for (uint32_t step = 1; step < TP4_RANKS && status == OK; step++) {
		const uint32_t rank = (tensor->rank + step) % TP4_RANKS;
		status = copy_put(
			b, output + tensor->rank * segment + skipped,
			word_of_rank +
				rank * sizeof(struct ordered_collective_peer),
			tensor->peer_regions[site][rank] +
				arguments->output_offset +
				tensor->rank * segment + skipped,
			segment - skipped,
			copy_branch(step - 1u, TP4_RANKS - 1u));
	}
	wide[3].branch = LAUNCH_JOIN;
	status = THEN(put(b, &wide[3])); /* DONE */
	const struct tp2_join_description unpack = {
		.source = { output, segment * TP4_RANKS },
		.output = values,
		.width = MODEL_HIDDEN,
		.row_count = b->rows,
		.source_row_stride = MODEL_HIDDEN,
		.output_row_stride = MODEL_HIDDEN,
		.operation = TP2_JOIN_RING_UNPACK
	};
	return THEN(tp2_join_prepare(&unpack, b->arena, b->tensor_join_kernel,
				     slots(b, 1)));
}

MUST_CHECK static result_status tensor_complete(struct builder *b,
						uint32_t site,
						struct linear_range values,
						struct linear_range routed)
{
	struct ordered_push_description collective =
		b->tensor_layer->collectives[site];
	struct ordered_push_arguments *const arguments = &collective.arguments;
	arguments->element_count = b->rows * MODEL_HIDDEN;
	const uint64_t bytes =
		(uint64_t)arguments->element_count * sizeof(float);
	const struct linear_range step = { ARRAY(STEP).byte_offset,
					   STEP_BLOCK_BYTES_VALUE };
	const struct linear_range input = { arguments->region_offset +
						    arguments->input_offset,
					    arguments->parity_stride + bytes };
	const struct linear_range output = { arguments->region_offset +
						     arguments->output_offset,
					     arguments->parity_stride + bytes };
	/* Reuse the prepared owner exchange and its separate READY/DONE protocol. Its BF16 publisher rounds each
     * rank's partial once; the owner adds in FP32 rank order and rounds the result once. Drafts and FULL/WIDE
     * chunks retain their collectives; prompt tails using THIN/BATCH share the new sums. This changes target bits. */
	const bool round_once = !b->reads_draft_step &&
				b->tp_description != NULL &&
				b->tp_description->stages[b->card]
					->description.decode_round_once &&
				(b->made->program.kind == PROGRAM_THIN ||
				 b->made->program.kind == PROGRAM_BATCH);
	if (round_once &&
	    (!b->tensor_layer->bf16_ring ||
	     b->tensor_layer->rank_count != TP4_RANKS ||
	     !b->tensor_layer->shared_channels ||
	     b->tensor_layer->parallel_publish || b->tensor_local_stub ||
	     b->tensor_layer->wide_exchange < V41_DATA_TP_EXCHANGE_OWNER_SUMS ||
	     arguments->payload_format != ORDERED_PUSH_BF16 ||
	     arguments->incoming_host_mask != 0))
		return STATUS_UNSUPPORTED;
	if (!round_once && b->tensor_layer->bf16_ring &&
	    b->rows <= b->tensor_layer->ring_rows) {
		/* the BF16 ring: values plus the routed sums added and rounded in one pass into the packed rows (the bits of
         * the FP32 publish and the round it replaces), then the ring's BF16 sum back into values; with no routed sums
         * (the attention site, or an FFN site whose combine added them: shared_sum) the values are those rows already:
         * the ring sums them in place (launches_untrimmed) */
		if (routed.byte_count == 0 &&
		    !b->tensor_layer->launches_untrimmed)
			return exact_gather(b, site, values, values,
					    b->rows * MODEL_HIDDEN *
						    LINEAR_BFLOAT16_BYTES,
					    b->rows);
		const struct linear_range packed =
			b->tensor_layer->exact_hidden;
		const struct tp2_join_description sum = {
			.source = values,
			.routed_sums = routed,
			.output = packed,
			.width = MODEL_HIDDEN,
			.row_count = b->rows,
			.source_row_stride = MODEL_HIDDEN,
			.routed_row_stride =
				routed.byte_count == 0 ? 0 : MODEL_HIDDEN,
			.output_row_stride = MODEL_HIDDEN,
			.operation = TP2_JOIN_BFLOAT16_SUM_TO_BFLOAT16
		};
		const result_status summed = tp2_join_prepare(
			&sum, b->arena, b->tensor_join_kernel, slots(b, 1));
		return summed != OK ?
			       summed :
			       exact_gather(b, site, packed, values,
					    b->rows * MODEL_HIDDEN *
						    LINEAR_BFLOAT16_BYTES,
					    b->rows);
	}
    /* CE_OWNER_SUMS: the attention site's piece of whole quarters by copy engines (direct routes) */
    if (!round_once && b->tensor_layer->shared_channels &&
	b->tensor_layer->wide_exchange >= V41_DATA_TP_EXCHANGE_CE_OWNER_SUMS &&
	b->rows > LINEAR_MAXIMUM_ROWS && !b->tensor_local_stub &&
	!b->tensor_layer->parallel_publish && routed.byte_count == 0 &&
	arguments->incoming_host_mask == 0 &&
	b->tensor_layer->rank_count == TP4_RANKS && b->rows % TP4_RANKS == 0 &&
	arguments->payload_format == ORDERED_PUSH_BF16)
	    return tensor_complete_ce(b, site, values);
    struct tp2_join_description description = {
	    .source = values,
	    .routed_sums = routed,
	    .output = input,
	    .step = step,
	    .output_slot_stride = arguments->parity_stride,
	    .width = MODEL_HIDDEN,
	    .row_count = b->rows,
	    .source_row_stride = MODEL_HIDDEN,
	    .routed_row_stride = routed.byte_count == 0 ? 0 : MODEL_HIDDEN,
	    .output_row_stride = MODEL_HIDDEN,
	    .operation = TP2_JOIN_BFLOAT16_TO_FLOAT32
    };
    result_status status = tp2_join_prepare(&description, b->arena,
					    b->tensor_join_kernel, slots(b, 1));
    const bool wide = b->tensor_layer->shared_channels &&
		      b->tensor_layer->wide_exchange != 0 &&
		      (round_once || b->rows > LINEAR_MAXIMUM_ROWS) &&
		      !b->tensor_local_stub &&
		      !b->tensor_layer->parallel_publish;
    /* owner sums need direct routes (tp_exchange.h): a host route keeps the push, the same bits */
    const bool owner_sums = wide &&
			    b->tensor_layer->wide_exchange >=
				    V41_DATA_TP_EXCHANGE_OWNER_SUMS &&
			    arguments->incoming_host_mask == 0;
    if (b->tensor_local_stub) {
	    description = (struct tp2_join_description){
		    .source = b->tensor_layer->stub_values[site],
		    .output = output,
		    .step = step,
		    .output_slot_stride = arguments->parity_stride,
		    .width = MODEL_HIDDEN,
		    .row_count = b->rows,
		    .source_row_stride = MODEL_HIDDEN,
		    .output_row_stride = MODEL_HIDDEN,
		    .operation = TP2_JOIN_FLOAT32_COPY
	    };
	    status = THEN(tp2_join_prepare(&description, b->arena,
					   b->tensor_join_kernel, slots(b, 1)));
    } else if (b->tensor_layer->parallel_publish) {
	    status = THEN(ordered_push_parallel_prepare(
		    &collective, b->tensor_parallel_kernels,
		    slots(b, ORDERED_PUSH_PARALLEL_LAUNCHES)));
    } else if (wide) {
        /* wide_exchange: the piece's push and sums by many CTAs (the same payload, order and bits); owner sums leave
         * the BF16 rows in the output array. Through a view (split_overlap's half) its step block's rows. */
	arguments->step_offset = b->view != 0 || round_once ?
					 ARRAY(STEP).byte_offset :
					 arguments->step_offset;
	status = THEN(tp_exchange_wide_prepare(
		&collective, exchange_phase(b), TP_EXCHANGE_PHASES,
		owner_sums ? b->tensor_owner_kernels :
			     b->tensor_exchange_kernels,
		owner_sums ? TP_EXCHANGE_FORM_OWNER_SUMS :
			     TP_EXCHANGE_FORM_PUSH,
		slots(b, TP_EXCHANGE_WIDE_LAUNCHES)));
    } else {
	    status = b->tensor_layer->shared_channels ?
			     THEN(tp_exchange_prepare(
				     &collective, exchange_phase(b),
				     TP_EXCHANGE_PHASES, b->tensor_push_kernel,
				     slots(b, 1))) :
			     THEN(ordered_push_prepare(&collective,
						       b->tensor_push_kernel,
						       slots(b, 1)));
    }
    description = (struct tp2_join_description){
	    .source = output,
	    .output = values,
	    .step = step,
	    .source_slot_stride = arguments->parity_stride,
	    .width = MODEL_HIDDEN,
	    .row_count = b->rows,
	    .source_row_stride = MODEL_HIDDEN,
	    .output_row_stride = MODEL_HIDDEN,
	    .operation = owner_sums ? TP2_JOIN_BFLOAT16_COPY :
				      TP2_JOIN_FLOAT32_TO_BFLOAT16
    };
    return THEN(tp2_join_prepare(&description, b->arena, b->tensor_join_kernel,
				 slots(b, 1)));
}

/* wide_exchange OWNER_RINGS, the FFN site of a piece of more than 8 rows: values plus the routed sums rounded to BF16
 * (the ring's packed rows) into the collective's input array, the owner ring sums (tp_exchange.h: each slab's BF16
 * ring sum, the same bits) and their BF16 rows copied back into values, in place of a ring a slab. */
MUST_CHECK static result_status
tensor_complete_owner_rings(struct builder *b, uint32_t site,
			    struct linear_range values,
			    struct linear_range routed)
{
	struct ordered_push_description collective =
		b->tensor_layer->collectives[site];
	struct ordered_push_arguments *const arguments = &collective.arguments;
	arguments->element_count = b->rows * MODEL_HIDDEN;
	const uint64_t bytes =
		(uint64_t)arguments->element_count * LINEAR_BFLOAT16_BYTES;
	const struct linear_range step = { ARRAY(STEP).byte_offset,
					   STEP_BLOCK_BYTES_VALUE };
	const struct linear_range input = { arguments->region_offset +
						    arguments->input_offset,
					    arguments->parity_stride + bytes };
	const struct linear_range output = { arguments->region_offset +
						     arguments->output_offset,
					     arguments->parity_stride + bytes };
	struct tp2_join_description description = {
		.source = values,
		.routed_sums = routed,
		.output = input,
		.step = step,
		.output_slot_stride = arguments->parity_stride,
		.width = MODEL_HIDDEN,
		.row_count = b->rows,
		.source_row_stride = MODEL_HIDDEN,
		.routed_row_stride = routed.byte_count == 0 ? 0 : MODEL_HIDDEN,
		.output_row_stride = MODEL_HIDDEN,
		.operation = TP2_JOIN_BFLOAT16_SUM_TO_BFLOAT16
	};
	result_status status = tp2_join_prepare(
		&description, b->arena, b->tensor_join_kernel, slots(b, 1));
	status = THEN(tp_exchange_wide_prepare(
		&collective, exchange_phase(b), TP_EXCHANGE_PHASES,
		b->tensor_ring_kernels, TP_EXCHANGE_FORM_OWNER_RING_SUMS,
		slots(b, TP_EXCHANGE_WIDE_LAUNCHES)));
	description = (struct tp2_join_description){
		.source = output,
		.output = values,
		.step = step,
		.source_slot_stride = arguments->parity_stride,
		.width = MODEL_HIDDEN,
		.row_count = b->rows,
		.source_row_stride = MODEL_HIDDEN,
		.output_row_stride = MODEL_HIDDEN,
		.operation = TP2_JOIN_BFLOAT16_COPY
	};
	return THEN(tp2_join_prepare(&description, b->arena,
				     b->tensor_join_kernel, slots(b, 1)));
}

/* bf16_slab_rings (model_v41_tp2_stage.h), and always in a BATCH or thin-tail program (each slab the thin steps' BF16
 * ring sum, not one FP32 ordered push over the sessions' rows): a piece of more than 8 rows completes a slab at a time
 * by the BF16 ring, the last slab its remaining rows; otherwise tensor_complete over the rows. */
MUST_CHECK static result_status
tensor_complete_slabs(struct builder *b, uint32_t site,
		      struct linear_range values, struct linear_range routed)
{
    const uint32_t rows = b->rows, first = b->first_row, piece = b->tensor_layer->ring_rows;
    if (!b->tensor_layer->bf16_ring || rows <= LINEAR_MAXIMUM_ROWS ||
	(!b->tensor_layer->bf16_slab_rings && !thin_forms_of(b)) ||
	rows <= piece)
	    return tensor_complete(
		    b, site, values,
		    routed); /* at most ring_rows rows: one BF16 ring */
    if (b->view != 0)
	    return STATUS_INVALID_ARGUMENT;
    const uint64_t value_row = (uint64_t)MODEL_HIDDEN * LINEAR_BFLOAT16_BYTES;
    const uint64_t routed_row = (uint64_t)MODEL_HIDDEN * sizeof(float);
    result_status status = OK;
    for (uint32_t at = 0; at < rows && status == OK; at += piece) { /* pieces of ring_rows rows (8: slabs) */
        const uint32_t count = rows - at < piece ? rows - at : piece;
        rows_select(b, 0, first + at, count);
	status = tensor_complete(
		b, site,
		(struct linear_range){ values.byte_offset + at * value_row,
				       count * value_row },
		routed.byte_count == 0 ?
			routed :
			(struct linear_range){ routed.byte_offset +
						       at * routed_row,
					       count * routed_row });
    }
    rows_select(b, 0, first, rows);
    return status;
}

/* The steps of a layer's launches before its FFN, in this order, of the rows the next launches take (enum layer_step):
 * an n-gram layer's table rows to values, their quantization and engram.wkv, the gate into the streams (2.1 E1-E4,
 * REF:350-365); the attention site's (2.1 launches 1 to 20, REF:982-986, 770-788) mix and finish; the quantization
 * of its hidden rows and wq_a with wkv; its state: the prepare, a source's compressor products (wkv, and wgate at
 * ratio 2) and its pool entry and key (2.1 launches 10 to 13 before 5 to 9: neither reads the other's arrays); qr alone
 * (a chunk's, ahead of its slabs' prepares, which form it again with the same bits); the quantization of qr, wq_b and
 * the indexer's wq_b; the indexer's weights; the queries; the scan and the core; wo_a, the quantization of its groups
 * and wo_b; the post. A thin step takes them all but qr alone in one call (layer 20 in two: its front, then the rest);
 * a chunk takes over the chunk every launch that reads no state, the queries over each span of its slabs, and over
 * each slab the rest (chunk_walk). */
enum layer_step {
	ENGRAM_VALUES = 1u << 0,
	ENGRAM_PRODUCT = 1u << 1,
	ENGRAM_GATE = 1u << 2,
	SITE_MIX = 1u << 3,
	INPUT_PRODUCTS = 1u << 4,
	STATE_WRITES = 1u << 5,
	POOL_PRODUCTS = 1u
		<< 6, /* of the hidden rows alone, between the prepare and the compressor */
	QUERY_NORM = 1u << 7,
	QUERY_PRODUCTS = 1u << 8,
	INDEX_WEIGHTS_PRODUCT = 1u << 9, /* of the hidden rows alone */
	QUERIES = 1u << 10, /* of wq_b's outputs and the rotation alone */
	CORE = 1u << 11,
	OUTPUT_PRODUCTS = 1u << 12,
	SITE_POST = 1u
		<< 13, /* the TP sum of the sublayer's output, then the post */
	SITE_SUM = 1u
		<< 14, /* split_overlap: the TP sum alone (a graph side run) */
	SITE_POST_ONLY = 1u
		<< 15, /* split_overlap: the post alone, behind its sum */
	SCAN = 1u
		<< 16, /* chunk_core: CORE's index scan alone, a slab at a time */
	CORE_ALONE = 1u
		<< 17, /* chunk_core: CORE's core alone, over the chunk */
	STATE_POOL = 1u
		<< 18, /* chunk_core: STATE_WRITES' pool writes alone, a slab at a time */
	WINDOW_WRITES = 1u
		<< 19, /* chunk_core: the prepare's window writes alone, over the chunk */
	POOL_CHUNK = 1u
		<< 20 /* pool_chunk: STATE_POOL over a piece (attention_pool_prepare) */
};
#define ENGRAM_STEPS (ENGRAM_VALUES | ENGRAM_PRODUCT | ENGRAM_GATE)
#define FRONT_STEPS (SITE_MIX | INPUT_PRODUCTS | STATE_WRITES | POOL_PRODUCTS)   /* layer 20's front (model_seam.h) */
#define REST_STEPS (QUERY_PRODUCTS | INDEX_WEIGHTS_PRODUCT | QUERIES | CORE | OUTPUT_PRODUCTS | SITE_POST)
#define AHEAD_STEPS (ENGRAM_STEPS | SITE_MIX | INPUT_PRODUCTS | POOL_PRODUCTS) /* a chunked layer's first products */
#define QUERY_STEPS (QUERY_PRODUCTS | INDEX_WEIGHTS_PRODUCT)

/* The n-gram steps of layer, in the card's rows of its table (model_seam.h). */
MUST_CHECK static result_status engram_launches(struct builder *b,
						uint32_t layer, uint32_t steps)
{
	const struct v41_tensor *const weights = b->plan->layer_tensors[layer];
	const uint64_t token_bytes =
		(uint64_t)V41_DATA_ENGRAM_ROWS_PER_TOKEN * ENGRAM_ROW_BYTES;
	const uint32_t table = (uint32_t)__builtin_popcountll(
		V41_DATA_ENGRAM_LAYERS & ((1ull << layer) - 1u));
	const uint64_t below = (uint64_t)__builtin_popcount(
		b->plan->cards[b->card].engram_tables & ((1u << table) - 1u));
	struct kernel_launch launches[ENGRAM_ROWS_LAUNCH_COUNT];
	const struct engram_rows_description description = {
		{ ARRAY(ENGRAM_ROWS).byte_offset +
			  (below * b->plan->limits.row_capacity + b->first_row) *
				  token_bytes,
		  b->rows * token_bytes },
		ARRAY(ENGRAM_VALUES),
		ARRAY(ENGRAM_PROJECTION),
		WEIGHTS(ENGRAM_QUERY)->stored,
		WEIGHTS(ENGRAM_KEY)->stored,
		ARRAY(STREAMS),
		V41_DATA_ENGRAM_ROWS_PER_TOKEN,
		b->rows
	};
	result_status status =
		(steps & (ENGRAM_VALUES | ENGRAM_GATE)) == 0 ?
			OK :
		b->vision_active ?
			image_engram_rows_prepare(&description, ARRAY(STEP),
						  b->arena, &b->engram,
						  &b->image_routing, launches) :
			engram_rows_prepare(&description, b->arena, &b->engram,
					    launches);
	if (status == OK && b->plan->limits.official_reference != 0) {
		launches[ENGRAM_ROWS_GATE_LAUNCH].kernel = b->official.engram;
		launches[ENGRAM_ROWS_GATE_LAUNCH].block[0] =
			OFFICIAL_STATE_THREADS;
	}
    if ((steps & ENGRAM_VALUES) != 0) {
	    status = THEN(put(b, &launches[ENGRAM_ROWS_VALUES_LAUNCH]));
    }
    if ((steps & ENGRAM_PRODUCT) != 0) {
        status = THEN(quantize(b, ARRAY(ENGRAM_VALUES), WEIGHTS(ENGRAM)->shape.input_size));
        status = THEN(trunk(b, WEIGHTS(ENGRAM), ARRAY(ENGRAM_PROJECTION), NULL, no_range));
    }
    return (steps & ENGRAM_GATE) != 0 ?
		   THEN(put(b, &launches[ENGRAM_ROWS_GATE_LAUNCH])) :
		   status;
}

/* The attention steps of layer with the pool of source, the latest source at or before the layer, and the selection
 * of group, the latest index layer (0 for none: layer 0 is neither). The selections are of the rows the launches
 * take. The site's and the state's launches are prepared again for each call that puts one of them. */
MUST_CHECK static result_status
attention_launches(struct builder *b, uint32_t layer, uint32_t source,
		   uint32_t group, uint32_t steps)
{
	const uint64_t selection_into =
		(uint64_t)b->first_row * MODEL_SELECTION_ROW_BYTES;
	const struct model_plan_record *const plan = b->plan;
	const struct v41_card *const card = &plan->cards[b->card];
	const struct v41_tensor *const weights = plan->layer_tensors[layer];
	const bool is_draft = layer >= MODEL_DRAFT_FIRST_LAYER;
	const bool is_thin = b->slab_count == 1u;
	const bool is_exl3 = b->plan->exl3_bits[layer] != 0;
	/* draft_bf16_rows: a draft layer's wq_a, wkv, wq_b and wo_b of its BF16 rows, as an EXL3 layer's (drafts only) */
	const bool bf16_rows = is_exl3 || (is_draft && b->draft_bf16_rows);
	const bool fused_projection = b->plan->limits.official_reference == 0 &&
				      is_thin && !bf16_rows &&
				      b->tensor_layer == NULL &&
				      !trunk_is_relaxed(b, WEIGHTS(OUTPUT));

	const uint32_t ratio = (V41_DATA_SOURCE_LAYERS >> source & 1u) == 0 ?
				       0 :
			       (V41_DATA_POOLED_LAYERS >> source & 1u) != 0 ?
				       MODEL_POOLED_RATIO :
				       MODEL_DIRECT_RATIO;
	const bool has_index = group == layer && layer != 0,
		   writes_pool = source == layer && ratio != 0;
	const float index_weights_scale = DENSE_INDEX_WEIGHTS_SCALE;
	const uint64_t sessions = plan->limits.session_capacity, rows = b->rows;
	const uint32_t context = plan->limits.context_token_limit,
		       entries = ratio == 0 ? 0 : context / ratio;
	const struct linear_range pool = {
		card->pool_offset[source],
		sessions * plan->pool_session_bytes[source]
	};
	const struct linear_range keys = {
		card->key_tiles_offset[source],
		sessions * plan->key_session_bytes[source]
	};
	const struct linear_range ring = {
		card->ring_offset[layer], sessions * V41_DATA_RING_SESSION_BYTES
	};
	/* wkv's product, packed at its row: float32 at ratio 2, bfloat16 at 1 (model_v41.h) */
	const uint64_t kv_row_bytes =
		(uint64_t)MODEL_HEAD_DIMS *
		(ratio == MODEL_POOLED_RATIO ? V41_DATA_F32 : V41_DATA_BF16);
	const struct linear_range pool_input =
		packed_rows(b, V41_DATA_ARRAY_COMPRESSOR_KV, kv_row_bytes);
	struct kernel_launch site[HYPER_SITE_LAUNCH_COUNT],
		state[ATTENTION_STATE_LAUNCH_COUNT];
	struct kernel_launch index_launches[INDEX_SCAN_LAUNCH_COUNT];
	struct kernel_launch core_launches[ATTENTION_CORE_MOST_LAUNCHES];
	uint32_t index_launch_count = 0, core_launch_count = 0;
	const uint32_t head_count =
		b->tensor_layer == NULL ?
			MODEL_HEADS_VALUE :
			MODEL_HEADS_VALUE / b->tensor_layer->rank_count;
	const uint32_t first_head = b->tensor_layer == NULL ?
					    0 :
					    b->tensor_layer->rank * head_count;
	const struct attention_state_description state_description = {
		ARRAY(STEP),
		ARRAY(ROTATION),
		ARRAY(QUERY_RANK),
		WEIGHTS(QUERY_NORM)->stored,
		ARRAY(QUERY_NORMED),
		ARRAY(WINDOW_INPUT),
		WEIGHTS(WINDOW_NORM)->stored,
		ring,
		ARRAY(HEADS),
		ARRAY(QUERIES),
		ARRAY(INDEX_HEADS),
		ARRAY(INDEX_QUERIES),
		pool_input,
		ARRAY(COMPRESSOR_SCORE),
		{ card->compressor_offset[layer],
		  sessions * V41_DATA_COMPRESSOR_SESSION_BYTES },
		WEIGHTS(COMPRESSOR_NORM)->stored,
		ARRAY(LATENT),
		pool,
		ARRAY(KEY_INPUT),
		WEIGHTS(KEY_NORM)->stored,
		keys,
		b->rows,
		(uint32_t)sessions,
		context,
		ratio != 0 ? 1u : 0u,
		first_head,
		head_count,
		has_index ? 1u : 0u,
		writes_pool ? ratio : 0,
		0
	};
	const struct attention_query_norm_description query_norm = {
		ARRAY(QUERY_RANK), WEIGHTS(QUERY_NORM)->stored,
		ARRAY(QUERY_NORMED), b->rows, 0
	};
	const struct attention_queries_description queries = {
		ARRAY(ROTATION),
		ARRAY(HEADS),
		ARRAY(QUERIES),
		ARRAY(INDEX_HEADS),
		ARRAY(INDEX_QUERIES),
		b->rows,
		ratio != 0 ? 1u : 0u,
		first_head,
		head_count,
		has_index ? 1u : 0u,
		0
	};
	const uint32_t candidate_role =
		layer == MODEL_CANDIDATE_SOURCE ? INDEX_CANDIDATES_SOURCE :
		(V41_DATA_CANDIDATE_READERS >> layer & 1u) != 0 ?
						  INDEX_CANDIDATES_READER :
						  INDEX_CANDIDATES_NONE;
	const uint64_t row_scores = candidate_role == INDEX_CANDIDATES_READER ?
					    MODEL_CONTEXT_WITHOUT_CANDIDATES :
					    entries;
	/* index_span: a span's rows of the view from index_row_first on, their scores in the stage's (index_scans) */
	const bool grouped = b->index_grouped && b->tensor_layer != NULL;
	const struct index_scan_description scan = {
		ARRAY(STEP),
		keys,
		ARRAY(INDEX_QUERIES),
		ARRAY(INDEX_WEIGHTS),
		{ grouped ? b->tensor_layer->index_scores.byte_offset :
			    ARRAY(INDEX_SCORES).byte_offset,
		  rows * row_scores * INDEX_SCORE_BYTES },
		{ card->selection_offset[layer] + selection_into,
		  rows * MODEL_SELECTION_ROW_BYTES },
		candidate_role != INDEX_CANDIDATES_NONE ? ARRAY(CANDIDATES) :
							  no_range,
		context,
		(uint32_t)sessions,
		ratio,
		b->rows,
		candidate_role,
		grouped ? b->index_row_first : 0u,
		grouped ? LINEAR_MAXIMUM_ROWS : 0u
	};
	const struct attention_core_description core = {
		ARRAY(STEP),
		ARRAY(QUERIES),
		ring,
		pool,
		{ card->selection_offset[group] + selection_into,
		  rows * MODEL_SELECTION_ROW_BYTES },
		WEIGHTS(SINK)->stored,
		ARRAY(ROTATION),
		ARRAY(ATTENTION_OUTPUT),
		ARRAY(ATTENTION_SCORES),
		b->rows,
		(uint32_t)sessions,
		context,
		ratio,
		first_head,
		head_count,
		is_draft ? LINEAR_ORDER_FIXED : b->plan->limits.linear_order,
		b->arithmetic_class,
		is_draft ? ATTENTION_CORE_WINDOW_BLOCK :
			   ATTENTION_CORE_WINDOW_CAUSAL,
		0,
		0,
		b->core_heads
	};
	struct linear_trunk_bfloat16_description groups = {
		.values = ARRAY(ATTENTION_OUTPUT),
		.output = ARRAY(GROUPS),
		.row_count = b->rows,
		.segment_count = b->tensor_layer == NULL ?
					 MODEL_OUTPUT_GROUPS :
					 MODEL_OUTPUT_GROUPS /
						 b->tensor_layer->rank_count
	};
	/* front_branch, a thin TP step's whole attention site: its front as the input front, then the coefficient front
     * as a graph side branch (it reads the streams, which nothing writes before the site's post, and writes post,
     * comb and the next pre, which the input front read before it) beside the products, the core and the sum,
     * joined at the post: the same launches' bits (site_front_records.h), the mix project off the chain */
	const bool front_branch =
		b->plan->limits.official_hc_rows == 0 && is_thin &&
		b->front_branch != 0 && b->tensor_layer != NULL &&
		!fused_projection &&
		(steps & (SITE_MIX | SITE_POST)) == (SITE_MIX | SITE_POST);
	const uint32_t side_launch = b->made->cards[b->card].launch_count +
				     1u; /* the coefficient front's place */
	result_status status = front_branch ?
				       site_front_of(b, weights, false, NULL,
						     SITE_FRONT_KIND_INPUT) :
				       OK;
	status = front_branch ?
			 THEN(site_front_of(b, weights, false, NULL,
					    SITE_FRONT_KIND_COEFFICIENTS)) :
			 status;
	status = front_branch ?
			 status :
		 (steps & SITE_MIX) != 0 ?
			 (is_thin && b->plan->limits.official_hc_rows == 0 ?
				  site_front(b, weights, false, NULL) :
				  site_begin(b, weights, false, site))

		 :
		 (steps & (SITE_POST | SITE_POST_ONLY)) != 0 ?
			 site_prepare(b, weights, false, site) :
			 OK;
	/* draft_bf16_operands: a thin draft site front's rows carried again (the front's own operands are E4M3) */
	if (is_draft && is_thin && !front_branch && (steps & SITE_MIX) != 0 &&
	    b->draft_carry_kernel != BACKEND_NO_KERNEL)
		status = THEN(quantize(b, ARRAY(HIDDEN), MODEL_HIDDEN));
	if (is_thin && !fused_projection && (steps & SITE_POST) != 0) {
		status = THEN(site_prepare(b, weights, false, site));
	}
	status = (steps & (STATE_WRITES | STATE_POOL)) == 0 ?
			 status :
			 THEN(attention_state_prepare(&state_description,
						      b->arena, &b->state,
						      state));
	if ((steps & INPUT_PRODUCTS) != 0) {
		/* an EXL3 layer's wq_a and wkv (exl3_trunk_bfloat16) read the hidden BF16 rows: no quantization of them in a
         * chunked or BATCH step (an n-gram layer quantizes its own values; launches_untrimmed: as before) */
		const bool unread = is_exl3 && b->tensor_layer != NULL &&
				    !b->tensor_layer->launches_untrimmed;
		status = (is_thin && b->plan->limits.official_hc_rows == 0) ||
					 unread ?
				 status :
				 THEN(quantize(b, ARRAY(HIDDEN), MODEL_HIDDEN));
		if (bf16_rows) {
			status = THEN(exl3_trunk_bfloat16(
				b, WEIGHTS(QUERY_DOWN), ARRAY(HIDDEN),
				ARRAY(QUERY_RANK)));
			status = THEN(exl3_trunk_bfloat16(b, WEIGHTS(WINDOW),
							  ARRAY(HIDDEN),
							  ARRAY(WINDOW_INPUT)));
			if (status == OK)
				trunk_mma_pair_merge(
					b); /* wq_a and wkv in one launch where both took the pair form's kernel */
		} else {
			status = THEN(trunk(b, WEIGHTS(QUERY_DOWN),
					    ARRAY(QUERY_RANK), WEIGHTS(WINDOW),
					    ARRAY(WINDOW_INPUT)));
		}
    }
    if (status == OK && b->plan->limits.official_reference != 0) {
	    state[ATTENTION_STATE_PREPARE].kernel = b->official.window;
	    state[ATTENTION_STATE_PREPARE].grid[0] =
		    state[ATTENTION_STATE_PREPARE].grid[1] = 1;
	    state[ATTENTION_STATE_PREPARE].block[0] = OFFICIAL_STATE_THREADS;
	    state[ATTENTION_STATE_COMPRESSOR].kernel = b->official.compressor;
	    state[ATTENTION_STATE_COMPRESSOR].block[0] = OFFICIAL_STATE_THREADS;
	    state[ATTENTION_STATE_KEY].kernel = b->official.key;
	    state[ATTENTION_STATE_KEY].block[0] = OFFICIAL_STATE_THREADS;
    }
    if ((steps & STATE_WRITES) != 0) {
	    if (b->plan->limits.official_reference != 0)
		    status = THEN(official_norm(
			    b, ARRAY(QUERY_RANK), WEIGHTS(QUERY_NORM)->stored,
			    ARRAY(QUERY_NORMED), MODEL_QUERY_RANK));
	    status = THEN(put(b, &state[ATTENTION_STATE_PREPARE]));
    }
    if ((steps & WINDOW_WRITES) != 0) { /* the prepare's window part by its device code (its qr: QUERY_NORM's) */
	    const struct attention_window_description window = {
		    ARRAY(STEP),
		    ARRAY(ROTATION),
		    ARRAY(WINDOW_INPUT),
		    WEIGHTS(WINDOW_NORM)->stored,
		    ring,
		    b->rows,
		    (uint32_t)sessions,
		    context,
		    ratio != 0 ? ATTENTION_ROTATION_POOLED :
				 ATTENTION_ROTATION_WINDOW_ONLY,
		    0
	    };
	    status = THEN(attention_window_prepare(&window, b->arena, &b->state,
						   slots(b, 1)));
    }
    if ((steps & POOL_PRODUCTS) != 0 && writes_pool) {  /* the pool's entry's products (2.1 launch 10) */
        /* pool_gather: the rank's part of the half's rows (gather_share's), then those rows gathered */
        const bool gathered = pool_gather_applies(b);
        const uint32_t view = b->view, first = b->first_row, all_rows = b->rows;
        uint32_t own_first = 0, own_rows = all_rows;
        if (gathered) {
		gather_share(b, all_rows, LINEAR_MAXIMUM_ROWS, &own_first,
			     &own_rows);
		rows_select(b, view, first + own_first, own_rows);
        }
	const struct linear_range products =
		gathered ? packed_rows(b, V41_DATA_ARRAY_COMPRESSOR_KV,
				       kv_row_bytes) :
			   pool_input;
	status = own_rows == 0 ?
			 status :
			 THEN(dense(b, WEIGHTS(COMPRESSOR), ARRAY(HIDDEN),
				    products,
				    ratio == MODEL_POOLED_RATIO ? F32 : BF16, 0,
				    1.0f));
	status = ratio != MODEL_POOLED_RATIO || own_rows == 0 ?
			 status :
			 THEN(dense(b, WEIGHTS(COMPRESSOR_GATE), ARRAY(HIDDEN),
				    ARRAY(COMPRESSOR_SCORE), F32, 0, 1.0f));
	if (gathered) {
            rows_select(b, view, first, all_rows);
	    status = THEN(pool_rows_gather(b, kv_row_bytes, first + own_first,
					   own_rows,
					   ratio == MODEL_POOLED_RATIO));
	}
    }
    status = (steps & POOL_CHUNK) == 0 || !writes_pool ?
		     status :
		     THEN(attention_pool_prepare(&state_description, b->arena,
						 &b->state, state));
    /* the pool's entry, its key (2.1: 11 to 13) */
    if ((steps & (STATE_WRITES | STATE_POOL | POOL_CHUNK)) != 0 && writes_pool) {
	    const bool deferred_slots = (steps & POOL_CHUNK) != 0 &&
					b->made->program.kind ==
						PROGRAM_BATCH &&
					ratio == MODEL_POOLED_RATIO;
	    struct attention_compressor_arguments compressor;
	    if (deferred_slots) {
		    memcpy(&compressor,
			   state[ATTENTION_STATE_COMPRESSOR].argument,
			   sizeof(compressor));
		    compressor.slot_mode = ATTENTION_COMPRESSOR_SLOTS_DEFER;
		    memcpy(state[ATTENTION_STATE_COMPRESSOR].argument,
			   &compressor, sizeof(compressor));
	    }
	    status = THEN(put(b, &state[ATTENTION_STATE_COMPRESSOR]));
	    if (deferred_slots) {
		    compressor.slot_mode = ATTENTION_COMPRESSOR_SLOTS_KEEP;
		    memcpy(state[ATTENTION_STATE_COMPRESSOR].argument,
			   &compressor, sizeof(compressor));
		    status = THEN(put(b, &state[ATTENTION_STATE_COMPRESSOR]));
	    }
	    status = THEN(dense(b, WEIGHTS(KEY), ARRAY(LATENT),
				ARRAY(KEY_INPUT), BF16, 0, 1.0f));
	    status = THEN(put(b, &state[ATTENTION_STATE_KEY]));
    }
    if ((steps & QUERY_NORM) != 0) {
	    status = b->plan->limits.official_reference != 0 ?
			     THEN(official_norm(b, ARRAY(QUERY_RANK),
						WEIGHTS(QUERY_NORM)->stored,
						ARRAY(QUERY_NORMED),
						MODEL_QUERY_RANK)) :
			     THEN(attention_query_norm_prepare(
				     &query_norm, b->arena, &b->state,
				     slots(b, 1)));
    }
    if ((steps & QUERY_PRODUCTS) != 0) {
        /* a TP rank's wq_b writes its heads compact; its queries read them there when they take these rows: in this
         * call, or a chunk of one span and one piece (its queries over the span, the products over the piece); else
         * (and launches_untrimmed) the heads copied into the full rows */
	const bool compact_heads =
		b->tensor_layer != NULL &&
		!b->tensor_layer->launches_untrimmed &&
		((steps & QUERIES) != 0 ||
		 (b->slab_count > 1u && b->half_view_first == 0 &&
		  b->chunk_rows <= V41_DATA_SPAN_ROWS &&
		  b->first_row == b->chunk_first && b->rows == b->chunk_rows));
	b->compact_layer = layer;
        b->compact_first_row = b->first_row;
        b->compact_rows = compact_heads ? b->rows : 0u;
        if (bf16_rows) {
            status = THEN(exl3_trunk_bfloat16(b, WEIGHTS(QUERY_UP), ARRAY(QUERY_NORMED),
                b->tensor_layer == NULL ? ARRAY(HEADS) : b->tensor_layer->compact));
            if (b->tensor_layer != NULL && !compact_heads) {
		    struct linear_range heads = ARRAY(HEADS);
		    const uint64_t into = (uint64_t)first_head *
					  MODEL_HEAD_DIMS * V41_DATA_BF16;
		    heads.byte_offset += into;
		    heads.byte_count -= into;
		    status = THEN(tensor_row_copy(
			    b, b->tensor_layer->compact, heads,
			    head_count * MODEL_HEAD_DIMS,
			    head_count * MODEL_HEAD_DIMS,
			    MODEL_HEADS_VALUE * MODEL_HEAD_DIMS));
	    }
            status = !has_index ? status : THEN(exl3_trunk_bfloat16(
                b, WEIGHTS(INDEX_QUERY_UP), ARRAY(QUERY_NORMED), ARRAY(INDEX_HEADS)));
        } else {
		status = THEN(
			quantize(b, ARRAY(QUERY_NORMED), MODEL_QUERY_RANK));
		status = THEN(trunk(b, WEIGHTS(QUERY_UP),
				    b->tensor_layer == NULL ?
					    ARRAY(HEADS) :
					    b->tensor_layer->compact,
				    NULL, no_range));
		if (b->tensor_layer != NULL && !compact_heads) {
			struct linear_range heads = ARRAY(HEADS);
			const uint64_t into = (uint64_t)first_head *
					      MODEL_HEAD_DIMS * V41_DATA_BF16;
			heads.byte_offset += into;
			heads.byte_count -= into;
			status = THEN(tensor_row_copy(
				b, b->tensor_layer->compact, heads,
				head_count * MODEL_HEAD_DIMS,
				head_count * MODEL_HEAD_DIMS,
				MODEL_HEADS_VALUE * MODEL_HEAD_DIMS));
		}
            status = has_index ? THEN(trunk(b, WEIGHTS(INDEX_QUERY_UP), ARRAY(INDEX_HEADS), NULL, no_range)) : status;
        }
    }
    if ((steps & INDEX_WEIGHTS_PRODUCT) != 0 && has_index) {    /* the indexer's weights (2.1 launch 8) */
        status = THEN(dense(b, WEIGHTS(INDEX_WEIGHTS), ARRAY(HIDDEN), ARRAY(INDEX_WEIGHTS), BF16, 0,
                            index_weights_scale));
    }
    if ((steps & QUERIES) != 0) {           /* the queries (2.1 launch 9): the state's QUERIES launch alone */
	    struct attention_queries_description read = queries;
	    if (b->compact_rows != 0 &&
		b->compact_layer ==
			layer) { /* wq_b's compact rows: these rows, or refused */
		    read.heads = b->tensor_layer != NULL &&
						 b->first_row ==
							 b->compact_first_row &&
						 b->rows == b->compact_rows ?
					 b->tensor_layer->compact :
					 no_range;
		    read.heads_compact = 1;
	    }
	if (b->plan->limits.official_reference != 0) {
		const struct official_rope_description rotated = {
			ARRAY(HEADS),
			{ ARRAY(ROTATION).byte_offset +
				  (uint64_t)(ratio != 0 ? 1u : 0u) *
					  OFFICIAL_ROPE_WIDTH * sizeof(float),
			  OFFICIAL_ROPE_WIDTH * sizeof(float) },
			ARRAY(QUERIES),
			MODEL_HEAD_DIMS,
			MODEL_HEADS_VALUE,
			1,
			0
		};
		status = THEN(official_rope_prepare(
			&rotated, b->arena, b->official.rope, slots(b, 1)));
		if (has_index && status == OK) {
			struct kernel_launch launch;
			status = attention_queries_prepare(&read, b->arena,
							   &b->state, &launch);
			launch.kernel = b->official.index_queries;
			launch.grid[0] = MODEL_INDEX_HEADS;
			launch.grid[1] = launch.grid[LAUNCH_DIMENSIONS - 1u] =
				1;
			launch.block[0] = OFFICIAL_STATE_THREADS;
			status = THEN(put(b, &launch));
		}
	} else {
		status = THEN(attention_queries_prepare(
			&read, b->arena, &b->state, slots(b, 1)));
	}
    }
    if ((steps & (CORE | SCAN | CORE_ALONE)) != 0) {    /* the scan, the core (chunk_core: either alone) */
	    if (has_index && (steps & CORE_ALONE) == 0 &&
		b->plan->limits.official_reference != 0) {
		    status = THEN(official_index_prepare(&scan, b->arena,
							 b->official.index,
							 &index_launches[0]));
		    index_launch_count = 1;
	    } else {
		    status = has_index && (steps & CORE_ALONE) == 0 ?
				     THEN(index_scan_prepare(
					     &scan, b->arena, &b->scan,
					     index_launches,
					     &index_launch_count)) :
				     status;
	    }
	for (uint32_t launch = 0; status == OK && launch < index_launch_count; launch++) {
            status = put(b, &index_launches[launch]);
        }
        /* a TP rank's core writes its heads into the compact rows alone when wo_a takes them next from there: in this
         * call, or a chunk of one piece (its slabs' cores at their rows in, wo_a over the piece); else (and
         * launches_untrimmed) the full rows and the copy */
	struct attention_core_description written =
		core; /* core_heads: more blocks, the same bits */
	/* a full program's decoder layer: the rows the engine's prefill cone marks past it run no core */
	written.cone_layer =
		b->made->program.kind == PROGRAM_FULL && !is_draft &&
				layer >= MODEL_DECODER_FIRST_LAYER &&
				layer < MODEL_LAYERS ?
			layer :
			0u;
	/* core_compact_halves (model_v41_tp2_stage.h): a split half's core too (a half selected: its wo_a takes the
         * rows from the half's first; a chunk's core of a program with half views keeps the copy, its wo_a may run a
         * half at a time: a non-pipelined split layer) */
        const bool halves_compact = b->tp_description != NULL && b->tensor_layer != NULL && b->chunk_view != 0 &&
                                    b->tp_description->stages[b->card]->description.core_compact_halves;
        const bool chunk_compact =
            b->slab_count > 1u && piece_count_of(b) == 1u && (b->half_view_first == 0 || halves_compact);
        b->core_compact = b->tensor_layer != NULL && !b->tensor_layer->launches_untrimmed &&
                          ((steps & OUTPUT_PRODUCTS) != 0 || chunk_compact);
        b->core_compact_layer = layer;
        b->core_compact_first = (steps & OUTPUT_PRODUCTS) == 0 && chunk_compact ? b->chunk_first : b->first_row;
        if (b->core_compact && (steps & SCAN) == 0) {
		const uint64_t row_bytes =
			(uint64_t)head_count * MODEL_HEAD_DIMS * V41_DATA_BF16;
		const uint64_t into =
			(uint64_t)(b->first_row - b->core_compact_first) *
			row_bytes;
		written.output =
			b->tensor_layer->compact.byte_count >=
					into + b->rows * row_bytes ?
				(struct linear_range){
					b->tensor_layer->compact.byte_offset +
						into,
					b->rows * row_bytes
				} :
				no_range;
		written.output_compact = 1;
        }
	if ((steps & SCAN) == 0 && b->plan->limits.official_reference != 0) {
		status = THEN(official_attention_prepare(
			&written,
			OFFICIAL_ATTENTION_QUERY_BFLOAT16 |
				OFFICIAL_ATTENTION_RING_BFLOAT16,
			b->arena, b->official.attention, &core_launches[0]));
		core_launch_count = 1;
	} else {
		status = (steps & SCAN) != 0 ?
				 status :
				 THEN(attention_core_prepare(
					 &written, b->arena, &b->core,
					 core_launches, &core_launch_count));
	}
	for (uint32_t launch = 0; status == OK && launch < core_launch_count;
	     launch++) {
		status = put(b, &core_launches[launch]);
	}
    }
    if ((steps & OUTPUT_PRODUCTS) != 0) {
	    if (b->plan->limits.official_reference != 0)
		    status = THEN(official_roundtrip(b, ARRAY(ATTENTION_OUTPUT),
						     MODEL_HEADS_VALUE *
							     MODEL_HEAD_DIMS));
	    const bool core_compact = b->tensor_layer != NULL &&
				      b->core_compact &&
				      b->core_compact_layer == layer;
	    if (core_compact) /* the core's compact rows from this step's first row, or refused */
		    groups.values = b->first_row == b->core_compact_first ?
					    b->tensor_layer->compact :
					    no_range;
	    if (b->tensor_layer != NULL && !core_compact) {
		    struct linear_range core_values = ARRAY(ATTENTION_OUTPUT);
		    const uint64_t into = (uint64_t)first_head *
					  MODEL_HEAD_DIMS * V41_DATA_BF16;
		    core_values.byte_offset += into;
		    core_values.byte_count -= into;
		    groups.values = b->tensor_layer->compact;
		    status = THEN(
			    tensor_row_copy(b, core_values, groups.values,
					    head_count * MODEL_HEAD_DIMS,
					    MODEL_HEADS_VALUE * MODEL_HEAD_DIMS,
					    head_count * MODEL_HEAD_DIMS));
	    }
	    status = THEN(matrix_of(b, WEIGHTS(OUTPUT_GROUPS),
				    LINEAR_FAMILY_TRUNK_BFLOAT16,
				    &groups.matrix));
	    const bool exact_dense = b->tensor_layer != NULL &&
				     b->tensor_layer->exact_dense;
	    if (exact_dense) {
		    /* the rank's two groups (full-K wo_a rows) into its owner slot; all eight gathered row-major; wo_b's 1280
             * owned outputs over the full K = 8192 into the slot again; the complete sublayer output gathered */
		    const uint32_t group_width =
			    groups.matrix.shape.output_size;
		    const uint32_t owned = WEIGHTS(OUTPUT)->shape.output_size;
		    /* a piece of a chunked step (more than 8 rows): the products over the piece, the gathers a slab at a time */
		    const bool wide = b->rows > LINEAR_MAXIMUM_ROWS;
		    groups.output = wide ? dense_own(b, group_width) :
					   dense_slot(b, group_width);
		    status = THEN(groups_product(
			    b, &groups,
			    WEIGHTS(OUTPUT_GROUPS)->fold_reference));
		    status = THEN(wide ? dense_gather_slabs(b, 0, group_width,
							    ARRAY(GROUPS)) :
					 dense_gather(b, 0, group_width,
						      ARRAY(GROUPS)));
		    const struct linear_range own_outputs =
			    wide ? dense_own(b, owned) : dense_slot(b, owned);
		    if (is_exl3) {
			    status = THEN(exl3_trunk_bfloat16(
				    b, WEIGHTS(OUTPUT), ARRAY(GROUPS),
				    own_outputs));
		    } else {
			    status = THEN(quantize(
				    b, ARRAY(GROUPS),
				    WEIGHTS(OUTPUT)->shape.input_size));
			    status = THEN(trunk(b, WEIGHTS(OUTPUT), own_outputs,
						NULL, no_range));
		    }
		    status = THEN(
			    wide ? dense_gather_slabs(b, 1, owned,
						      ARRAY(SUBLAYER_OUTPUT)) :
				   dense_gather(b, 1, owned,
						ARRAY(SUBLAYER_OUTPUT)));
	    } else if (b->plan->limits.official_reference == 0 && is_thin &&
		       b->arithmetic_class == ARITHMETIC_CLASS_SM_120_NATIVE) {
		    const struct attention_groups_description fused_groups = {
			    groups.matrix,
			    groups.values,
			    groups.output,
			    ARRAY(ROW_CODES),
			    ARRAY(ROW_EXPONENTS),
			    ARRAY(ROW_OPERANDS),
			    b->rows,
			    groups.segment_count
		    };
		    status = THEN(attention_groups_prepare(
			    &fused_groups, b->arena, &b->linear, &b->output,
			    slots(b, 1)));
	    } else {
		    status = THEN(groups_product(
			    b, &groups,
			    WEIGHTS(OUTPUT_GROUPS)->fold_reference));
		    /* an EXL3 layer's wo_b (exl3_trunk_bfloat16 below) reads the groups' BF16 rows: no quantization of them
             * (launches_untrimmed: as before) */
		    const bool unread = bf16_rows && !fused_projection &&
					b->tensor_layer != NULL &&
					!b->tensor_layer->launches_untrimmed;
		    status =
			    unread ?
				    status :
				    THEN(quantize(
					    b, ARRAY(GROUPS),
					    WEIGHTS(OUTPUT)->shape.input_size));
	    }
	if (exact_dense) {
		/* wo_b and its gather are above */
	} else if (fused_projection) {
		struct attention_projection_description projection = {
			.codes = ARRAY(ROW_CODES),
			.exponents = ARRAY(ROW_EXPONENTS),
			.operands = ARRAY(ROW_OPERANDS),
			.output = ARRAY(SUBLAYER_OUTPUT),
			.streams = ARRAY(STREAMS),
			.post = ARRAY(POST),
			.comb = ARRAY(COMB),
			.row_count = b->rows
		};
		status = THEN(matrix_of(b, WEIGHTS(OUTPUT), TRUNK,
					&projection.matrix));
		status = THEN(attention_projection_prepare(
			&projection, b->arena, &b->linear, &b->output,
			slots(b, 1)));
	} else if (bf16_rows) {
		status = THEN(exl3_trunk_bfloat16(b, WEIGHTS(OUTPUT),
						  ARRAY(GROUPS),
						  ARRAY(SUBLAYER_OUTPUT)));
	} else {
		status = THEN(trunk(b, WEIGHTS(OUTPUT), ARRAY(SUBLAYER_OUTPUT),
				    NULL, no_range));
	}
    }
    if ((steps & (SITE_POST | SITE_SUM)) != 0 && b->tensor_layer != NULL && !b->tensor_layer->exact_dense)
        status = THEN(tensor_complete_slabs(b, 0, ARRAY(SUBLAYER_OUTPUT), no_range));
    status = (steps & (SITE_POST | SITE_POST_ONLY)) != 0 && !fused_projection ?
		     THEN(put(b, &site[HYPER_SITE_POST])) :
		     status;
    if (front_branch && status == OK) /* the side run (the coefficient front), its resume and its join (the post) */
        front_branch_mark(b, side_launch);
    return status;
}

/* The steps of layer's n-gram rows (an n-gram layer's) and attention site, in that order. */
MUST_CHECK static result_status layer_steps(struct builder *b, uint32_t layer,
					    uint32_t source, uint32_t group,
					    uint32_t steps)
{
	const bool has_engram = (V41_DATA_ENGRAM_LAYERS >> layer & 1u) != 0;
	result_status status = has_engram && (steps & ENGRAM_STEPS) != 0 ?
				       engram_launches(b, layer, steps) :
				       OK;
	/* a full program's decoder layer after the first: its wide trunks leave the rows whose output at the layer before
     * nothing reads (the engine's prefill cone: none of the layer's own outputs or window rows of them is read) */
	const uint32_t cone_before = b->trunk_cone_layer;
	b->trunk_cone_layer =
		b->made->program.kind == PROGRAM_FULL &&
				layer > MODEL_DECODER_FIRST_LAYER &&
				layer < MODEL_LAYERS ?
			layer - 1u :
			0u;
	status = (steps & ~(uint32_t)ENGRAM_STEPS) != 0 ?
			 THEN(attention_launches(b, layer, source, group,
						 steps)) :
			 status;
	b->trunk_cone_layer = cone_before;
	return status;
}

/* Layer 20's front ends (model_seam.h): its segment from *first_launch, then its pool's and keys' copies. */
static void front_end(struct builder *b, uint32_t layer, uint32_t *first_launch)
{
    segment_add(b, layer, *first_launch);
    copies_hand_off(b, layer, FRONT_COPIES);
    b->made->program.front_segment_count = b->made->program.segment_count;
    *first_launch = b->made->cards[b->card].launch_count;
}

/* Where the layer keeps routed experts in host memory, its segment from first_launch to its router ends with an
 * expert pause (model_seam.h) for them, in the card's slots. */
MUST_CHECK static result_status
expert_pause_add(struct builder *b, uint32_t layer, uint32_t first_launch)
{
	const struct model_plan_record *const plan = b->plan;
	struct execution_program *const program = &b->made->program;
	const uint32_t count = plan->host_expert_count[layer],
		       segments_before = program->segment_count;
	struct expert_pause pause = { ARRAY(JOBS).byte_offset,
				      plan->cards[b->card].state_offset,
				      plan->slot_byte_count,
				      plan->expert_byte_count,
				      { 0 },
				      program->host_expert_count,
				      count,
				      MODEL_EXPERTS_PER_ROW,
				      plan->cards[b->card].slot_count,
				      b->card,
				      segments_before };
	_Static_assert(sizeof(pause.matrix_offsets) ==
			       sizeof(plan->expert_matrix_offsets),
		       "gate, up, down");
	memcpy(pause.matrix_offsets, plan->expert_matrix_offsets,
	       sizeof(pause.matrix_offsets));
	segment_add(b, layer, first_launch);
	if (segments_before >= MOST_SEGMENTS ||
	    program->segment_count != segments_before + 1u ||
	    program->expert_pause_count >= MODEL_LAYERS ||
	    count > MODEL_EXPERTS) {
		LOG_ERROR(
			"program_prepare: no segment or pause for the %" PRIu32
			" host experts of layer %" PRIu32,
			count, layer);
		return STATUS_INVALID_ARGUMENT;
	}
	b->made->segments[pause.after_segment].pause =
		SEGMENT_PAUSE_EXPERT_UPLOADS;
	b->made->expert_pauses[program->expert_pause_count++] = pause;
	memcpy(&b->made->host_expert_offsets[program->host_expert_count],
	       plan->host_expert_offsets[layer], count * sizeof(uint64_t));
	program->host_expert_count += count;
	return OK;
}

/* The expert table of layer in the card's image, which is what arena_initialise put into its arena (model_v41.h):
 * whole uint64 words inside the image. NULL, behind an error line, where it is not. */
static const void *expert_table_of(const struct builder *b, uint32_t layer)
{
	const struct v41_card *const card = &b->plan->cards[b->card];
	const uint64_t image = card->state_offset + card->slots_byte_count,
		       table = card->expert_table_offset[layer];
	const uint64_t in_image = table - image;
	if (card->image == NULL || table < image ||
	    image < card->state_offset ||
	    ((uintptr_t)card->image + in_image) % sizeof(uint64_t) != 0 ||
	    card->image_byte_count < MODEL_EXPERT_OFFSETS_BYTES ||
	    in_image > card->image_byte_count - MODEL_EXPERT_OFFSETS_BYTES) {
		LOG_ERROR("program_prepare: no expert table of layer %" PRIu32
			  " in the image of card %" PRIu32,
			  layer, b->card);
		return NULL;
	}
    return card->image + in_image;
}

/* exl3_expert_rotation: the TC MoE records' rotation of this card's rank in layer (exl3_tc_records.h), 0 without. */
static uint32_t expert_rotation_of(const struct builder *b, uint32_t layer)
{
	return b->tensor_layer != NULL &&
			       b->tensor_layer->exl3_expert_rotation != 0 &&
			       layer < MODEL_LAYERS &&
			       b->plan->exl3_bits[layer] != 0 ?
		       exl3_tc_rotation_of(
			       b->tensor_layer->exl3_expert_rotation,
			       b->tensor_layer->rank, layer) :
		       0u;
}

/* The routed experts of layer for the rows the next launches take, over job_count jobs at the job table. */
MUST_CHECK static result_status
routed_of(const struct builder *b, uint32_t layer, uint32_t job_count,
	  struct linear_routed_experts_description *experts_out)
{
	*experts_out = (struct linear_routed_experts_description){
		{ ARRAY(JOBS).byte_offset, linear_jobs_byte_count(job_count) },
		b->plan->experts[layer],
		ARRAY(ROW_CODES),
		ARRAY(ROW_EXPONENTS),
		ARRAY(ROW_OPERANDS),
		ARRAY(PAIR_HIDDEN),
		ARRAY(PAIR_CODES),
		ARRAY(PAIR_EXPONENTS),
		ARRAY(PAIR_OPERANDS),
		ARRAY(PAIR_OUTPUT),
		ARRAY(ROW_SUMS),
		expert_shapes[GATE_UP],
		expert_shapes[DOWN],
		WEIGHT_FORMAT_E2M1_SCALED,
		job_count,
		b->rows,
		layer >= MODEL_DRAFT_FIRST_LAYER ? MODEL_DRAFT_EXPERTS_PER_ROW :
						   MODEL_EXPERTS_PER_ROW,
		0,
		0,
		SWIGLU_LIMIT,
		0
	};
	if (b->tensor_layer != NULL && !b->tensor_layer->exact_output_shards &&
	    !b->tensor_layer->experts_partitioned) {
		experts_out->gate_up_shape.output_size =
			b->plan->layer_tensors[layer][V41_DATA_SHARED_GATE]
				.shape.output_size;
		experts_out->down_shape.input_size =
			b->plan->layer_tensors[layer][V41_DATA_SHARED_DOWN]
				.shape.input_size;
		if (expert_rotation_of(b, layer) != 0) {
			/* every rank holds 512- and 640-column chunks: the router's matrix bounds at the narrower (each matrix
             * still lies inside the experts' range); the TC MoE records take the 640 layout */
			experts_out->gate_up_shape.output_size =
				TP4_SMALL_INNER;
			experts_out->down_shape.input_size = TP4_SMALL_INNER;
		}
    }
    if (b->plan->exl3_bits[layer] != 0) {
	    experts_out->weight_format = WEIGHT_FORMAT_EXL3;
	    experts_out->gate_up_shape.bits_per_weight =
		    b->plan->exl3_bits[layer];
	    experts_out->down_shape.bits_per_weight = b->plan->exl3_bits[layer];
	    const uint32_t product_slices =
		    b->tensor_layer != NULL &&
				    b->tensor_layer->exl3_shard != NULL ?
			    b->tensor_layer->exl3_shard->product_slices :
		    b->plan->limits.cmp_order == CMP_ORDER_RELAXED ?
			    EXL3_RELAXED_SLICES :
			    EXL3_PRODUCT_SLICES;
	    experts_out->gate_up_slice_count = experts_out->down_slice_count =
		    product_slices;
	    return OK;
    }
    result_status status = slices_of(b, LINEAR_FAMILY_EXPERT_GATE_UP,
				     &experts_out->gate_up_shape,
				     &experts_out->gate_up_slice_count);
    return THEN(slices_of(b, LINEAR_FAMILY_EXPERT_DOWN,
			  &experts_out->down_shape,
			  &experts_out->down_slice_count));
}

/* The shared expert of layer for the rows the next launches take, its hidden vectors where the routed experts' were
 * (they are summed by then); its rows' codes where a quantization of those rows left them, or with is_chunk_quantized
 * where one quantization over the whole chunk did (linear_records.h: rows packed at their width from the first). */
MUST_CHECK static result_status
shared_launches(struct builder *b, uint32_t layer, bool is_chunk_quantized)
{
	const struct v41_tensor *const weights = b->plan->layer_tensors[layer];
	const uint64_t *const arrays = b->plan->cards[b->card].array_offset;
	struct linear_shared_expert_description shared = {
		.codes = ARRAY(ROW_CODES),
		.exponents = ARRAY(ROW_EXPONENTS),
		.operands = ARRAY(ROW_OPERANDS),
		.gate_output = ARRAY(SHARED_GATE),
		.up_output = ARRAY(SHARED_UP),
		.hidden = ARRAY(PAIR_HIDDEN),
		.hidden_codes = ARRAY(PAIR_CODES),
		.hidden_exponents = ARRAY(PAIR_EXPONENTS),
		.hidden_operands = ARRAY(PAIR_OPERANDS),
		.output = ARRAY(SHARED_OUTPUT),
		.row_count = b->rows,
		.activation_limit = SWIGLU_LIMIT
	};
	if (is_chunk_quantized) { /* one quantization took the chunk: its rows lie packed at its width from its first row
                                   (split_overlap: the half's, which begins chunk_first rows into the arrays) */
		const uint32_t into = b->first_row - b->chunk_first;
		shared.codes = (struct linear_range){
			arrays[V41_DATA_ARRAY_ROW_CODES] +
				b->chunk_first *
					array_bytes[V41_DATA_ARRAY_ROW_CODES]
						.row_bytes +
				linear_codes_byte_count(into, MODEL_HIDDEN),
			linear_codes_byte_count(b->rows, MODEL_HIDDEN)
		};
		shared.exponents = (struct linear_range){
			arrays[V41_DATA_ARRAY_ROW_EXPONENTS] +
				b->chunk_first *
					array_bytes[V41_DATA_ARRAY_ROW_EXPONENTS]
						.row_bytes +
				linear_exponents_byte_count(into, MODEL_HIDDEN),
			linear_exponents_byte_count(b->rows, MODEL_HIDDEN)
		};
		shared.operands = (struct linear_range){
			arrays[V41_DATA_ARRAY_ROW_OPERANDS] +
				b->chunk_first *
					array_bytes[V41_DATA_ARRAY_ROW_OPERANDS]
						.row_bytes +
				linear_values_byte_count(into, MODEL_HIDDEN),
			linear_values_byte_count(b->rows, MODEL_HIDDEN)
		};
	}
	result_status status =
		matrix_of(b, WEIGHTS(SHARED_GATE), TRUNK, &shared.gate);
	status = THEN(matrix_of(b, WEIGHTS(SHARED_UP), TRUNK, &shared.up));
	status = THEN(matrix_of(b, WEIGHTS(SHARED_DOWN), TRUNK, &shared.down));
	const int32_t folds[EXPERT_PAUSE_MATRICES] = {
		WEIGHTS(SHARED_GATE)->fold_reference,
		WEIGHTS(SHARED_UP)->fold_reference,
		WEIGHTS(SHARED_DOWN)->fold_reference
	};
	struct kernel_launch launches[LINEAR_SHARED_EXPERT_LAUNCH_COUNT];
	status = THEN(linear_shared_expert_select_prepare(
		&shared, b->arena, &b->linear, &b->relaxed, b->arithmetic_class,
		b->plan->limits.cmp_order, folds, launches));
	if (b->plan->limits.official_reference != 0) {
		status = THEN(linear_shared_expert_prepare(
			&shared, b->arena, &b->linear, launches));
		status = THEN(trunk_of(b, shared.codes, shared.exponents,
				       shared.operands, WEIGHTS(SHARED_GATE),
				       shared.gate_output, WEIGHTS(SHARED_UP),
				       shared.up_output));
		status = THEN(put(b, &launches[LINEAR_SHARED_EXPERT_HIDDEN]));
		status = THEN(put(b, &launches[LINEAR_SHARED_EXPERT_QUANTIZE]));
		status = THEN(trunk_of(
			b, shared.hidden_codes, shared.hidden_exponents,
			shared.hidden_operands, WEIGHTS(SHARED_DOWN),
			shared.output, NULL, no_range));
		const struct expert_join_description join = {
			ARRAY(PAIR_OUTPUT),    shared.output,
			ARRAY(STREAMS),	       ARRAY(POST),
			ARRAY(COMB),	       ARRAY(ROW_SUMS),
			MODEL_EXPERTS_PER_ROW, 1
		};
		return THEN(expert_join_prepare(&join, b->arena, &b->fused,
						slots(b, 1)));
	}
    const bool is_thin = !is_chunk_quantized;
    const bool exl3 = b->plan->exl3_bits[layer] != 0;
    const bool fused_down = is_thin && !exl3 && b->tensor_layer == NULL &&
        !trunk_is_relaxed(b, WEIGHTS(SHARED_DOWN));
    const uint32_t count =
	    LINEAR_SHARED_EXPERT_LAUNCH_COUNT - (fused_down ? 1u : 0u);
    const struct linear_trunk_bfloat16_description wide_products[] = {
	    { .matrix = shared.gate,
	      .values = ARRAY(HIDDEN),
	      .output = shared.gate_output,
	      .row_count = b->rows,
	      .segment_count = 1 },
	    { .matrix = shared.up,
	      .values = ARRAY(HIDDEN),
	      .output = shared.up_output,
	      .row_count = b->rows,
	      .segment_count = 1 },
	    { .matrix = shared.down,
	      .values = shared.hidden,
	      .output = shared.output,
	      .row_count = b->rows,
	      .segment_count = 1 }
    };
    if (!is_thin && b->tensor_layer != NULL && thin_forms_of(b) &&
        trunk_mma_wide_applies(b, &wide_products[0]) && trunk_mma_wide_applies(b, &wide_products[2])) {
        /* --tp_trunk_mma, a BATCH step: the same from the chunk's BF16 rows by the _64 form */
        for (uint32_t matrix = 0; matrix < sizeof(wide_products) / sizeof(wide_products[0]) && status == OK;
             matrix++) {
            if (matrix == 2u)
		    status = put(b, &launches[LINEAR_SHARED_EXPERT_HIDDEN]);
	    struct kernel_launch *const launch = slots(b, 1);
	    status = launch == NULL ?
			     STATUS_INVALID_ARGUMENT :
			     THEN(trunk_mma_wide(b, launch,
						 &wide_products[matrix]));
	    if (matrix == 1u && status == OK)
                trunk_mma_pair_merge(b); /* gate and up (grouped forms) in one launch */
        }
        return status;
    }
    if (is_thin && b->tensor_layer != NULL && trunk_mma_ready(b, &shared.gate.shape) &&
        trunk_mma_ready(b, &shared.up.shape) && trunk_mma_ready(b, &shared.down.shape)) {
        /* --tp_trunk_mma: gate and up from the sublayer's BF16 rows, down from the hidden step's values, by
         * kernel_trunk_mma (GNmac's W8A16 class); the hidden step stays, its quantization goes */
	const struct linear_matrix *const matrices[] = { &shared.gate,
							 &shared.up,
							 &shared.down };
	const struct linear_range values[] = { ARRAY(HIDDEN), ARRAY(HIDDEN),
					       shared.hidden };
	const struct linear_range outputs[] = { shared.gate_output,
						shared.up_output,
						shared.output };
	for (uint32_t matrix = 0; matrix < sizeof(matrices) / sizeof(matrices[0]) && status == OK; matrix++) {
            if (matrices[matrix] == &shared.down)
		    status = put(b, &launches[LINEAR_SHARED_EXPERT_HIDDEN]);
	    const struct trunk_linear_bfloat16_arguments record = {
		    matrices[matrix]->stored.byte_offset,
		    values[matrix].byte_offset,
		    outputs[matrix].byte_offset,
		    matrices[matrix]->shape.output_size,
		    matrices[matrix]->shape.input_size,
		    b->rows,
		    1
	    };
	    struct kernel_launch launch;
	    prepare_launch_fill(&launch, b->trunk_mma_kernels[0], 1, 1,
				TRUNK_MMA_THREADS, 0, &record,
				TRUNK_LINEAR_BFLOAT16_ARGUMENT_BYTES);
	    trunk_mma_swap(b, &launch, &matrices[matrix]->shape);
            status = THEN(put(b, &launch));
            if (matrices[matrix] == &shared.up && status == OK)
                trunk_mma_pair_merge(b); /* gate and up in one launch */
        }
        return status;
    }
    /* a full program's decoder layer: the shared expert's wide products leave the rows the routed experts leave (the
     * cone's rows whose output at this layer nothing reads) */
    const uint32_t cone_layer =
	    b->made->program.kind == PROGRAM_FULL &&
			    layer >= MODEL_DECODER_FIRST_LAYER &&
			    layer < MODEL_LAYERS ?
		    layer :
		    0u;
    for (uint32_t index = 0; index < count && status == OK; index++) {
	    if (index == LINEAR_SHARED_EXPERT_GATE_UP ||
		index == LINEAR_SHARED_EXPERT_DOWN)
		    status = trunk_cone_at(b, &launches[index], cone_layer);
	    status = THEN(put(b, &launches[index]));
    }
    if (is_thin && exl3 && b->tensor_layer == NULL) {
	    struct kernel_launch site[HYPER_SITE_LAUNCH_COUNT];
	    status = THEN(site_prepare(b, weights, true, site));
	    return THEN(put(b, &site[HYPER_SITE_POST]));
    }
    if (is_thin && b->tensor_layer == NULL) {
	    const struct expert_join_description join = {
		    ARRAY(PAIR_OUTPUT),
		    ARRAY(SHARED_OUTPUT),
		    ARRAY(STREAMS),
		    ARRAY(POST),
		    ARRAY(COMB),
		    ARRAY(ROW_SUMS),
		    layer >= MODEL_DRAFT_FIRST_LAYER ?
			    MODEL_DRAFT_EXPERTS_PER_ROW :
			    MODEL_EXPERTS_PER_ROW,
		    b->rows
	    };
	    const struct expert_shared_join_description joined = {
		    shared.down, shared.hidden_codes, shared.hidden_exponents,
		    shared.hidden_operands, join
	    };
	    status = fused_down ?
			     THEN(expert_shared_join_prepare(
				     &joined, b->arena, &b->linear, &b->fused,
				     slots(b, 1))) :
			     THEN(expert_join_prepare(&join, b->arena,
						      &b->fused, slots(b, 1)));
    }
    return status;
}

/* All original reduction chains are prepared from the full shapes. Only complete output tiles are removed.
 * The stage loader has extracted those tiles without decoding or changing their scales. */
MUST_CHECK static result_status exact_gather(struct builder *b, uint32_t site,
					     struct linear_range input,
					     struct linear_range output,
					     uint32_t bytes, uint32_t vectors)
{
	const struct v41_tp2_layer_plan *const tensor = b->tensor_layer;
	if (tensor == NULL || site >= TP2_SITES)
		return STATUS_INVALID_ARGUMENT;
	return exact_gather_bank(b, site, tensor->ring_offsets[site],
				 tensor->ring_bank_bytes[site],
				 tensor->ring_row_bytes[site], input, output,
				 bytes, vectors);
}

/* One lossless ring gather (or the BF16 recipe's ring sum) through the bank of b->rows rows of the site's channel (its
 * peer table and region); a bank holds its mode's two epochs of slots or the ring is refused. */
MUST_CHECK static result_status
exact_gather_bank(struct builder *b, uint32_t site, uint64_t ring_offset,
		  uint64_t bank_bytes, uint64_t row_bytes,
		  struct linear_range input, struct linear_range output,
		  uint32_t bytes, uint32_t vectors)
{
	const struct v41_tp2_layer_plan *const tensor = b->tensor_layer;
	if (tensor == NULL || site >= TP2_SITES || bank_bytes == 0 ||
	    (!tensor->exact_output_shards && !tensor->bf16_ring) ||
	    b->rows == 0 ||
	    b->rows > (ring_offset == tensor->ring_offsets[site] ?
			       tensor->ring_rows :
			       LINEAR_MAXIMUM_ROWS) ||
	    bytes == 0 || bytes % (TP4_RANKS * EXACT_RING_PACK_BYTES) != 0 ||
	    input.byte_count < bytes || output.byte_count < bytes ||
	    input.byte_offset > b->arena ||
	    bytes > b->arena - input.byte_offset ||
	    output.byte_offset > b->arena ||
	    bytes > b->arena - output.byte_offset)
		return STATUS_INVALID_ARGUMENT;
	const struct ordered_push_arguments *const channel =
		&tensor->collectives[site].arguments;
	const uint64_t bank =
		ring_offset +
		v41_ring_bank_start(bank_bytes, row_bytes, b->rows);
	const struct linear_range bank_range = {
		bank, bank_bytes + row_bytes * b->rows
	};
	const uint64_t packs = bytes / (TP4_RANKS * EXACT_RING_PACK_BYTES);
	if (EXACT_RING_METADATA_BYTES +
		    packs * (tensor->bf16_ring ?
				     EXACT_RING_SUM_EPOCH_HOPS *
					     EXACT_RING_PACK_BYTES :
				     EXACT_RING_EPOCH_HOPS *
					     EXACT_RING_PACKET_BYTES) >
	    bank_range.byte_count)
		return STATUS_INVALID_ARGUMENT;
	/* the BF16 sum may run in place (input the output: kernel_cut_through.cuh reads every owner's pack before it
     * writes that pack); otherwise the input shares no byte with the output */
	const bool in_place = tensor->bf16_ring &&
			      input.byte_offset == output.byte_offset;
	const struct checked_range ranges[] = {
		{ "the ring output", &output, bytes, false },
		{ "the ring bank", &bank_range, bank_range.byte_count, false },
		{ "the ring input", &input, bytes, false }
	};
	if (bank < channel->region_offset ||
	    !prepare_ranges_are_valid("the TP ring", ranges,
				      in_place ? TP2_SITES : TP2_SITES + 1u,
				      b->arena))
		return STATUS_INVALID_ARGUMENT;
	const struct exact_ring_arguments arguments = {
		.input_offset = input.byte_offset,
		.output_offset = output.byte_offset,
		.bank_offset = bank,
		.peer_table_offset = channel->peer_table_offset,
		.peer_bank_delta = bank - channel->region_offset,
		.deadline_cycles = TP2_DEADLINE_CYCLES,
		.packs_per_owner = bytes / (TP4_RANKS * EXACT_RING_PACK_BYTES),
		.vectors = vectors,
		.rank = tensor->rank,
		/* ring_column_owners: a BF16 sum's owners the quarters of every row, not of the rows (the input row-major) */
		.mode = !tensor->bf16_ring ? EXACT_RING_GATHER :
			b->ring_column_owners && vectors != 0 &&
					(uint64_t)packs % vectors == 0 ?
					     EXACT_RING_BF16_SUM_COLUMNS :
					     EXACT_RING_BF16_SUM
	};
	uint32_t blocks = 1;
	while (blocks < EXACT_RING_BLOCKS &&
	       blocks * ORDERED_COLLECTIVE_THREADS < arguments.packs_per_owner)
		blocks *= EXACT_RING_GRID_GROWTH;
	struct kernel_launch launch;
	prepare_launch_fill(&launch, b->exact_ring_kernel, blocks, 1u,
			    ORDERED_COLLECTIVE_THREADS, 0, &arguments,
			    sizeof(arguments));
	return put(b, &launch);
}

/* EXL3 activation writes an owner-major FP16 chunk directly into PAIR_OPERANDS; this gather only transposes bits.
 * exl3_wide_entry: the chunk's pairs (total_pairs) are gathered a slab (pairs of b->rows rows) at a time, slab `slab`
 * of the chunk's owner-major chunk: each rank reads only its own range, so its input base is its slab's part less
 * the slab's owner prefix, and the output is the slab's pairs of the pair-major output. */
MUST_CHECK static result_status
exl3_hidden_gather(struct builder *b, struct linear_range input,
		   struct linear_range output, uint32_t pairs,
		   uint32_t total_pairs, uint32_t slab)
{
	const struct v41_tp2_layer_plan *const tensor = b->tensor_layer;
	if (tensor == NULL || tensor->exl3_shard == NULL ||
	    !tensor->exact_output_shards || tensor->rank_count != TP4_RANKS ||
	    tensor->original_code_wire || tensor->bf16_ring ||
	    b->unequal_ring_kernel == BACKEND_NO_KERNEL || b->rows == 0 ||
	    b->rows > LINEAR_MAXIMUM_ROWS || pairs == 0)
		return STATUS_UNSUPPORTED;
	const uint64_t bytes =
		(uint64_t)pairs * V41_DATA_EXPERT_INNER * EXL3_HALF_BYTES;
	if (total_pairs < pairs || (total_pairs != pairs && total_pairs % pairs != 0) ||
	    (uint64_t)(slab + 1u) * pairs > total_pairs)
		return STATUS_INVALID_ARGUMENT;
	/* the slab's parts: rank r's at (total - pairs) begin_r + slab pairs own_r halves past the input's base */
	const struct bb_exl3_output_shard_plan *const own = tensor->exl3_shard;
	input.byte_offset +=
		((uint64_t)(total_pairs - pairs) * own->gate_up.output_begin +
		 (uint64_t)slab * pairs * own->gate_up.output_count) *
		EXL3_HALF_BYTES;
	input.byte_count = input.byte_count >= bytes ? bytes : input.byte_count;
	output.byte_offset += (uint64_t)slab * bytes;
	output.byte_count = output.byte_count >= (uint64_t)(slab + 1u) * bytes ? bytes : 0u;
	const struct ordered_push_arguments *const channel =
		&tensor->collectives[0].arguments;
	const uint64_t bank = tensor->ring_offsets[0] +
			      v41_ring_bank_start(tensor->ring_bank_bytes[0],
						  tensor->ring_row_bytes[0],
						  b->rows);
	const struct linear_range bank_range = {
		bank,
		tensor->ring_bank_bytes[0] + tensor->ring_row_bytes[0] * b->rows
	};
	const struct checked_range ranges[] = {
		{ "EXL3 FP16 gather input", &input, bytes, false },
		{ "EXL3 FP16 gather output", &output, bytes, false },
		{ "EXL3 lossless bank", &bank_range, bank_range.byte_count,
		  false }
	};
	if (bank < channel->region_offset ||
	    !prepare_ranges_are_valid("EXL3 hidden gather", ranges,
				      TP2_SITES + 1u, b->arena))
		return STATUS_INVALID_ARGUMENT;
	struct exact_ring_unequal_arguments arguments = {
		.input_offset = input.byte_offset,
		.output_offset = output.byte_offset,
		.bank_offset = bank,
		.peer_table_offset = channel->peer_table_offset,
		.peer_bank_delta = bank - channel->region_offset,
		.deadline_cycles = TP2_DEADLINE_CYCLES,
		.vectors = pairs,
		.rank = tensor->rank
	};
	uint32_t prefix = 0;
	for (uint32_t rank = 0; rank < TP4_RANKS; rank++) {
		struct bb_exl3_output_shard_plan shard;
		const result_status status = bb_exl3_output_shard_prepare(
			MODEL_HIDDEN, V41_DATA_EXPERT_INNER,
			tensor->exl3_shard->gate_up.bits_per_weight, TP4_RANKS,
			rank, tensor->exl3_shard->order, &shard);
		if (status != OK)
			return status;
		const uint32_t count = pairs * shard.gate_up.output_count *
				       EXL3_HALF_BYTES / EXACT_RING_PACK_BYTES;
		arguments.first_pack[rank] = prefix;
		arguments.pack_count[rank] = count;
		arguments.maximum_packs = count > arguments.maximum_packs ?
						  count :
						  arguments.maximum_packs;
		prefix += count;
	}
	const uint64_t bank_bytes =
		EXACT_RING_METADATA_BYTES + (uint64_t)arguments.maximum_packs *
						    EXACT_RING_EPOCH_HOPS *
						    EXACT_RING_PACKET_BYTES;
	if (bank_bytes > bank_range.byte_count)
		return STATUS_INVALID_ARGUMENT;
	uint32_t blocks = 1;
	while (blocks < EXACT_RING_BLOCKS &&
	       blocks * ORDERED_COLLECTIVE_THREADS < arguments.maximum_packs)
		blocks *= EXACT_RING_GRID_GROWTH;
	struct kernel_launch launch;
	prepare_launch_fill(&launch, b->unequal_ring_kernel, blocks, 1,
			    ORDERED_COLLECTIVE_THREADS, 0, &arguments,
			    sizeof(arguments));
	return put(b, &launch);
}

/* Every product keeps its full K chain. The only exchanges are original FP16 operands and ordered FP32 channels. */
/* split_overlap mode 7: the launches of a joint TC prefill call with the halves' route (exl3_tcp half_rows) a phase
 * puts: all of a call without halves, its route, gate/up and retile, the first half's part (the whole call's down
 * operands, the half's down GEMM and combine) or the second's (down GEMM, combine). */
enum tcp_emit { TCP_EMIT_ALL, TCP_EMIT_FRONT, TCP_EMIT_FIRST_HALF, TCP_EMIT_SECOND_HALF };

MUST_CHECK static result_status
exl3_routed_launches(struct builder *b, uint32_t layer,
		     const struct linear_routed_experts_description *experts)
{
	if (b->arithmetic_class != ARITHMETIC_CLASS_SM_80 ||
	    layer >= MODEL_LAYERS)
		return STATUS_UNSUPPORTED;
	const uint32_t pairs = b->rows * experts->pairs_per_row;
	const uint64_t scratch = b->plan->cards[b->card].exl3_scratch_offset;
	const uint64_t scratch_bytes =
		exl3_scratch_byte_count(pairs, MODEL_HIDDEN);
	if (scratch > b->arena || scratch_bytes > b->arena - scratch)
		return STATUS_INVALID_ARGUMENT;
	const uint64_t products = scratch + (uint64_t)pairs * MODEL_HIDDEN *
						    EXL3_PROJECTIONS *
						    EXL3_HALF_BYTES;
	const struct exl3_expert_record whole = {
		.jobs_offset = experts->jobs.byte_offset,
		.input_offset = ARRAY(HIDDEN).byte_offset,
		.operands_offset = scratch,
		.products_offset = products,
		.hidden_offset = experts->hidden.byte_offset,
		.output_offset = products,
		.input_size = MODEL_HIDDEN,
		.inner_size = experts->gate_up_shape.output_size,
		.pair_count = pairs,
		.pairs_per_row = experts->pairs_per_row,
		.job_count = experts->job_count,
		.bits_per_weight = b->plan->exl3_bits[layer],
		.activation_limit = SWIGLU_LIMIT
	};
	if (b->tensor_layer == NULL || !b->tensor_layer->exact_output_shards) {
        /* Optional inner sharding changes the local K chain; the later BF16 ring sums full-output partials.
         * This deliberately differs from the default full-K/lossless recipe and needs model quality gates. */
	if (b->tensor_layer != NULL &&
	    (!b->tensor_layer->bf16_ring ||
	     b->tensor_layer->original_code_wire ||
	     b->tensor_layer->rank_count != TP4_RANKS))
		return STATUS_UNSUPPORTED;
	/* exl3_expert_rotation: the mixed forms at 640 (exl3_tc_records.h) */
        const uint32_t rotation = expert_rotation_of(b, layer);
	const uint32_t intermediate =
		rotation != 0 ? EXL3_TC_MIXED_WIDTH :
				experts->gate_up_shape.output_size;
	if (b->tensor_layer != NULL && b->tensor_layer->exl3_tc_moe && b->rows <= tc_most_rows(b)) {
            /* exl3_tc_moe: GNmac's tensor-core routed experts from the job table, each row's route-weighted sum FP32
             * into the row sums the BF16 ring adds to the shared expert's (exl3_tc.h); a tail program's workspace
             * its own (model_v41_tp2_stage.h tail_overlay) */
	    const struct linear_range workspace =
		    tail_program(b) ? b->tensor_layer->exl3_tail_workspace :
				      b->tensor_layer->exl3_tc_workspace;
	    const struct exl3_tc_record tc = {
		    .jobs_offset = experts->jobs.byte_offset,
		    .hidden_offset = ARRAY(HIDDEN).byte_offset,
		    .workspace_offset = workspace.byte_offset,
		    .output_offset = experts->row_output.byte_offset,
		    .row_count = b->rows,
		    .job_capacity = experts->job_count,
		    .intermediate = intermediate,
		    .bits_x2 = 2u * b->plan->exl3_bits[layer],
		    .activation_limit = SWIGLU_LIMIT,
		    .rotation = rotation,
		    .shared_sum = b->shared_sum_requested ? 1u : 0u,
		    .shared_offset = b->shared_sum_requested ?
					     ARRAY(SHARED_OUTPUT).byte_offset :
					     0u
	    };
	    b->shared_sum_fused = b->shared_sum_requested;
	    struct kernel_launch tc_launches[EXL3_TC_LAUNCHES];
	    result_status tc_status =
		    workspace.byte_count < exl3_tc_workspace_bytes(
						   (uint64_t)b->rows *
							   EXL3_TC_PAIRS_PER_ROW,
						   tc.intermediate) ||
				    experts->pairs_per_row !=
					    EXL3_TC_PAIRS_PER_ROW ?
			    STATUS_INVALID_ARGUMENT :
		    batch_thin_bits_of(b) && b->batch_tc_paired != 0 &&
				    b->rows >= b->batch_tc_paired ?
			    /* the short forms' partials, two slices an item */
			    exl3_tc_prepare_paired(&tc, b->arena, b->backend,
						   b->cards[b->card],
						   tc_launches) :
			    batch_thin_bits_of(
				    b) ? /* the short forms at every row count: each row the thin steps' sums */
				    exl3_tc_prepare_short(&tc, b->arena,
							  b->backend,
							  b->cards[b->card],
							  tc_launches) :
				    exl3_tc_prepare(&tc, b->arena, b->backend,
						    b->cards[b->card],
						    tc_launches);
	    /* A persistent warp owns complete product items: the grid changes their assignment, never their
             * arithmetic. Restrict the override to decode programs; the prefill TC geometry stays independent. */
	    const uint32_t product_blocks =
		    b->tp_description == NULL ?
			    0u :
			    b->tp_description->stages[b->card]
				    ->description.exl3_tc_product_blocks;
	    if (tc_status == OK && product_blocks != 0 &&
		(b->made->program.kind == PROGRAM_THIN ||
		 b->made->program.kind == PROGRAM_BATCH)) {
		    tc_launches[TC_GATE_UP_PRODUCT_LAUNCH].grid[0] =
			    product_blocks;
		    tc_launches[TC_DOWN_PRODUCT_LAUNCH].grid[0] =
			    product_blocks;
	    }
	    for (uint32_t launch = 0;
		 launch < EXL3_TC_LAUNCHES && tc_status == OK; launch++)
		    tc_status = put(b, &tc_launches[launch]);
            return tc_status;
        }
	if (b->tensor_layer != NULL && b->tensor_layer->exl3_tc_moe &&
	    b->rows <= EXL3_TCP_CHUNK_ROWS &&
	    b->tensor_layer->exl3_tcp_workspace.byte_count != 0) {
		/* exl3_tc_moe, a chunk of more than 64 rows: GNmac's prefill MoE from the wide router's picks */
		const struct exl3_tcp_record tcp = {
			.picks_offset = ARRAY(ROUTER_PICKS).byte_offset,
			.expert_offsets_offset =
				b->plan->cards[b->card]
					.expert_table_offset[layer],
			.hidden_offset = ARRAY(HIDDEN).byte_offset,
			.workspace_offset =
				b->tensor_layer->exl3_tcp_workspace.byte_offset,
			.output_offset = experts->row_output.byte_offset,
			.row_count = b->rows,
			.intermediate = intermediate,
			.bits_x2 = 2u * b->plan->exl3_bits[layer],
			.activation_limit = SWIGLU_LIMIT,
			.rotation = rotation,
			.half_rows =
				b->tcp_emit != TCP_EMIT_ALL ? b->half_rows : 0u,
			/* a full program's decoder layer: the rows the engine's prefill cone marks past it take no route */
			.step_offset = ARRAY(STEP).byte_offset,
			.cone_layer =
				b->made->program.kind == PROGRAM_FULL &&
						layer >=
							MODEL_DECODER_FIRST_LAYER &&
						layer < MODEL_LAYERS ?
					layer :
					0u,
			.two_blocks = b->tp_description != NULL &&
						      b->tp_description
							      ->stages[b->card]
							      ->description
							      .tcp_two_blocks ?
					      1u :
					      0u
		};
		struct kernel_launch tcp_launches[EXL3_TCP_MOST_LAUNCHES];
		uint32_t tcp_count = 0;
		result_status tcp_status =
			b->tensor_layer->exl3_tcp_workspace.byte_count <
					exl3_tcp_workspace_bytes(
						b->rows, tcp.intermediate) ?
				STATUS_INVALID_ARGUMENT :
				exl3_tcp_prepare(&tcp, b->arena, b->backend,
						 b->cards[b->card],
						 tcp_launches, &tcp_count);
		/* mode 7: the route, gate/up and retile, then the halves' parts (exl3_tc_records.h) */
		const uint32_t front =
			tcp.half_rows != 0 ?
				tcp_count - EXL3_TCP_PART_LAUNCHES -
					EXL3_TCP_SECOND_PART_LAUNCHES :
				tcp_count;
		const uint32_t first = b->tcp_emit == TCP_EMIT_FIRST_HALF ?
					       front :
				       b->tcp_emit == TCP_EMIT_SECOND_HALF ?
					       front + EXL3_TCP_PART_LAUNCHES :
					       0;
		const uint32_t end = b->tcp_emit == TCP_EMIT_FRONT ?
					     front :
				     b->tcp_emit == TCP_EMIT_FIRST_HALF ?
					     front + EXL3_TCP_PART_LAUNCHES :
					     tcp_count;
		if (tcp_status == OK &&
		    (tcp.half_rows == 0) != (b->tcp_emit == TCP_EMIT_ALL))
			tcp_status =
				STATUS_INVALID_ARGUMENT; /* halves need a split chunk's first half */
		for (uint32_t launch = first; launch < end && tcp_status == OK;
		     launch++)
			tcp_status = put(b, &tcp_launches[launch]);
		return tcp_status;
	}
	if (rotation != 0) /* the other forms take every expert at the rank's one width */
		return STATUS_UNSUPPORTED;
	struct kernel_launch launches[EXL3_EXPERT_LAUNCHES];
	/* exl3_fast_products: kernel_exl3_wide's transforms and products, the same words (exl3.h) */
	const bool fast = b->tensor_layer != NULL &&
			  b->tensor_layer->exl3_fast_products;
	result_status status =
		b->plan->limits.cmp_order == CMP_ORDER_RELAXED ?
			exl3_experts_relaxed_prepare(&whole, b->arena,
						     b->backend,
						     b->cards[b->card],
						     launches) :
		fast ? exl3_experts_fast_prepare(&whole, b->arena, b->backend,
						 b->cards[b->card], launches) :
		       exl3_experts_prepare(&whole, b->arena, b->backend,
					    b->cards[b->card], launches);
	for (uint32_t pass = 0; pass < EXL3_EXPERT_LAUNCHES && status == OK;
	     pass++)
		status = put(b, &launches[pass]);
	const struct expert_row_sum_arguments sum = {
		products, experts->row_output.byte_offset, MODEL_HIDDEN,
		experts->pairs_per_row
	};
	struct kernel_launch launch;
	status = THEN(bb_exl3_row_sum_prepare(&sum, b->rows, b->arena,
					      b->backend, b->cards[b->card],
					      &launch));
	return THEN(put(b, &launch));
	}
	const struct v41_tp2_layer_plan *const tensor = b->tensor_layer;
	const struct bb_exl3_output_shard_plan *const shard =
		tensor->exl3_shard;
	/* exl3_wide_entry: a chunk of more than 8 rows in one call; its gathers a slab of 8 rows at a time */
	const bool wide = b->rows > LINEAR_MAXIMUM_ROWS;
	if (shard == NULL || !tensor->exact_output_shards ||
	    tensor->bf16_ring || tensor->original_code_wire ||
	    tensor->rank_count != TP4_RANKS || shard->rank != tensor->rank ||
	    (wide &&
	     (!tensor->exl3_wide_entry || b->rows % LINEAR_MAXIMUM_ROWS != 0)))
		return STATUS_UNSUPPORTED;
	const uint64_t local_operand =
		experts->hidden_operands.byte_offset +
		(uint64_t)pairs * shard->gate_up.output_begin * EXL3_HALF_BYTES;
	struct kernel_launch launches[BB_EXL3_PASSES];
	result_status status = OK;
	for (uint32_t pass = 0; pass < BB_EXL3_PASSES && status == OK; pass++) {
		struct exl3_expert_record record = whole;
		const bool down = pass >= BB_EXL3_DOWN_PASS;
		record.input_size = down ? shard->down.output_count :
					   shard->gate_up.input_count;
		record.inner_size = down ? shard->down.input_count :
					   shard->gate_up.output_count;
		record.operands_offset =
			down ? tensor->exact_hidden.byte_offset :
			pass == BB_EXL3_ACTIVATE_PASS ? local_operand :
							scratch;
		const struct bb_exl3_pass_description description = {
			.plan = shard,
			.record = record,
			.arena_bytes = b->arena,
			.weight_offset = experts->expert_weights.byte_offset,
			.weight_bytes = experts->expert_weights.byte_count,
			.pass = (enum bb_exl3_pass)pass,
			/* exl3_fast_products: kernel_exl3_wide's _t8_g1 / _t4_g4p and transforms (exl3_tp.h) */
			.product_tiles = tensor->exl3_fast_products ? BB_EXL3_PRODUCT_EIGHT_TILES : 0u
		};
		status = bb_exl3_pass_prepare(&description, b->backend,
					      b->cards[b->card],
					      &launches[pass]);
	}
	for (uint32_t pass = 0; pass <= BB_EXL3_ACTIVATE_PASS && status == OK;
	     pass++)
		status = put(b, &launches[pass]);
	const uint32_t rows = b->rows,
		       slab_rows = wide ? LINEAR_MAXIMUM_ROWS : rows;
	const uint32_t slab_pairs = slab_rows * experts->pairs_per_row;
	b->rows = slab_rows; /* the gathers' banks are a slab's */
	for (uint32_t slab = 0; slab < rows / slab_rows && status == OK; slab++)
		status = exl3_hidden_gather(b, experts->hidden_operands, tensor->exact_hidden, slab_pairs, pairs, slab);
	b->rows = rows;
	for (uint32_t pass = BB_EXL3_DOWN_PASS;
	     pass < BB_EXL3_PASSES && status == OK; pass++)
		status = put(b, &launches[pass]);
	const struct expert_row_sum_arguments sum = {
		products,
		experts->row_output.byte_offset +
			(uint64_t)tensor->rank * b->rows *
				shard->down.output_count * sizeof(float),
		shard->down.output_count, experts->pairs_per_row
	};
	struct kernel_launch launch;
	status = THEN(bb_exl3_row_sum_prepare(&sum, b->rows, b->arena,
					      b->backend, b->cards[b->card],
					      &launch));
	status = THEN(put(b, &launch));
	/* the row sums: owner-major [rank][rows][own outputs]; slab s of each rank's part to rows 8 s .. of exact_sums */
	const uint64_t own_bytes = (uint64_t)shard->down.output_count * sizeof(float);
	const uint32_t slab_bytes =
		slab_rows * MODEL_HIDDEN * (uint32_t)sizeof(float);
	b->rows = slab_rows;
	for (uint32_t slab = 0; slab < rows / slab_rows && status == OK; slab++) {
		const struct linear_range input = {
			experts->row_output.byte_offset +
				(uint64_t)tensor->rank * (rows - slab_rows) *
					own_bytes +
				(uint64_t)slab * slab_rows * own_bytes,
			slab_bytes
		};
		const struct linear_range output = {
			tensor->exact_sums.byte_offset +
				(uint64_t)slab * slab_bytes,
			tensor->exact_sums.byte_count >=
					(uint64_t)(slab + 1u) * slab_bytes ?
				slab_bytes :
				0u
		};
		status = exact_gather(b, 1, input, output, slab_bytes, slab_rows);
	}
	b->rows = rows;
	return status;
}

/* The original products are owner-major over the chunk. Each gather still uses its immutable eight-row bank;
 * only this rank's owner range is read, so a virtual input base selects its slab without copying the products. */
MUST_CHECK static result_status
original_gather_slabs(struct builder *b, uint32_t site,
		      struct linear_range input, struct linear_range output,
		      uint32_t vectors_per_row, uint32_t width,
		      uint32_t element_bytes)
{
	const struct v41_tp2_layer_plan *const tensor = b->tensor_layer;
	const uint32_t rows = b->rows;
	const uint64_t row_bytes =
		(uint64_t)vectors_per_row * width * element_bytes;
	if (tensor == NULL || !tensor->original_wide_experts ||
	    !tensor->exact_output_shards || tensor->bf16_ring ||
	    tensor->original_code_wire || tensor->rank_count != TP4_RANKS ||
	    tensor->rank >= tensor->rank_count ||
	    width % tensor->rank_count != 0 || rows == 0 ||
	    rows % LINEAR_MAXIMUM_ROWS != 0 || vectors_per_row == 0 ||
	    row_bytes == 0 || row_bytes > UINT32_MAX / LINEAR_MAXIMUM_ROWS ||
	    input.byte_count < (uint64_t)rows * row_bytes ||
	    output.byte_count < (uint64_t)rows * row_bytes ||
	    input.byte_offset > b->arena ||
	    (uint64_t)rows * row_bytes > b->arena - input.byte_offset ||
	    output.byte_offset > b->arena ||
	    (uint64_t)rows * row_bytes > b->arena - output.byte_offset)
		return STATUS_INVALID_ARGUMENT;
	const uint64_t owner_row_bytes = row_bytes / tensor->rank_count;
	const uint32_t count = LINEAR_MAXIMUM_ROWS;
	const uint32_t bytes = (uint32_t)(count * row_bytes);
	result_status status = OK;
	for (uint32_t at = 0; at < rows && status == OK; at += count) {
		const uint64_t into =
			((uint64_t)tensor->rank * (rows - count) + at) *
			owner_row_bytes;
		if (into > input.byte_count ||
		    bytes > input.byte_count - into) {
			status = STATUS_INVALID_ARGUMENT;
			break;
		}
		const struct linear_range from = { input.byte_offset + into,
						   bytes };
		const struct linear_range to = {
			output.byte_offset + (uint64_t)at * row_bytes, bytes
		};
		b->rows = count;
		status = exact_gather(b, site, from, to, bytes,
				      count * vectors_per_row);
	}
	b->rows = rows;
	return status;
}

MUST_CHECK static result_status
exact_routed_launches(struct builder *b,
		      const struct linear_routed_experts_description *experts)
{
	const bool wide = b->rows > LINEAR_MAXIMUM_ROWS;
	if (wide && (!b->tensor_layer->original_wide_experts ||
		     b->tensor_layer->original_code_wire))
		return STATUS_UNSUPPORTED;
	struct kernel_launch launches[LINEAR_ROUTED_EXPERTS_LAUNCH_COUNT];
	result_status status = linear_routed_experts_prepare(
		experts, b->arena, &b->linear, launches);
	if (status != OK)
		return status;
	const uint32_t ranks = b->tensor_layer->rank_count,
		       rank = b->tensor_layer->rank;
	const uint32_t hidden_width =
		experts->gate_up_shape.output_size / ranks;
	const uint32_t output_width = experts->down_shape.output_size / ranks;
	const uint32_t pairs = b->rows * experts->pairs_per_row;
	struct expert_gate_up_arguments gate;
	struct activation_quantize_arguments quantize;
	struct expert_down_arguments down;
	struct expert_row_sum_arguments sum;
	memcpy(&gate, launches[LINEAR_ROUTED_EXPERTS_GATE_UP].argument,
	       sizeof(gate));
	memcpy(&quantize, launches[LINEAR_ROUTED_EXPERTS_QUANTIZE].argument,
	       sizeof(quantize));
	memcpy(&down, launches[LINEAR_ROUTED_EXPERTS_DOWN].argument,
	       sizeof(down));
	memcpy(&sum, launches[LINEAR_ROUTED_EXPERTS_ROW_SUM].argument,
	       sizeof(sum));
	gate.output_size = hidden_width;
	gate.hidden_offset +=
		(uint64_t)rank * pairs * hidden_width * sizeof(uint16_t);
	down.output_size = output_width;
	sum.width = output_width;
	sum.row_output_offset +=
		(uint64_t)rank * b->rows * output_width * sizeof(float);
	quantize.values_offset = b->tensor_layer->exact_hidden.byte_offset;
	memcpy(launches[LINEAR_ROUTED_EXPERTS_GATE_UP].argument, &gate,
	       sizeof(gate));
	memcpy(launches[LINEAR_ROUTED_EXPERTS_QUANTIZE].argument, &quantize,
	       sizeof(quantize));
	memcpy(launches[LINEAR_ROUTED_EXPERTS_DOWN].argument, &down,
	       sizeof(down));
	memcpy(launches[LINEAR_ROUTED_EXPERTS_ROW_SUM].argument, &sum,
	       sizeof(sum));
	launches[LINEAR_ROUTED_EXPERTS_GATE_UP].grid[0] /= ranks;
	launches[LINEAR_ROUTED_EXPERTS_DOWN].grid[0] /= ranks;
	launches[LINEAR_ROUTED_EXPERTS_ROW_SUM].grid[0] =
		(output_width + LINEAR_ELEMENT_THREADS - 1u) /
		LINEAR_ELEMENT_THREADS;
	if (b->tensor_layer->original_code_wire) {
		if (ranks != TP4_RANKS ||
		    hidden_width != QUANTIZED_WIRE_LOCAL_WIDTH ||
		    experts->gate_up_slice_count != 1u ||
		    pairs > EXPERT_PRODUCER_MOST_PAIRS)
			return STATUS_UNSUPPORTED;
		const uint32_t pitch = QUANTIZED_WIRE_PITCH;
		const uint32_t wire_bytes = pairs * pitch * ranks;
		const struct linear_range packed = b->tensor_layer->compact;
		if (packed.byte_count < wire_bytes ||
		    packed.byte_offset > b->arena ||
		    wire_bytes > b->arena - packed.byte_offset)
			return STATUS_INVALID_ARGUMENT;
		const struct quantize_wire_arguments producer = {
			gate.hidden_offset,
			packed.byte_offset + (uint64_t)rank * pairs * pitch,
			hidden_width,
			pairs,
			pitch,
			0
		};
		struct kernel_launch producer_launch;
		prepare_launch_fill(&producer_launch, b->wire_quantize_kernel,
				    (hidden_width + LINEAR_ELEMENT_THREADS -
				     1u) / LINEAR_ELEMENT_THREADS,
				    pairs, LINEAR_ELEMENT_THREADS, 0, &producer,
				    sizeof(producer));
		status = put(b, &launches[LINEAR_ROUTED_EXPERTS_GATE_UP]);
		status = THEN(put(b, &producer_launch));
		status = THEN(exact_gather(b, 0, packed,
					   b->tensor_layer->exact_hidden,
					   wire_bytes, pairs));
		const struct quantized_wire_arguments wire = {
			b->tensor_layer->exact_hidden.byte_offset,
			quantize.codes_offset,
			quantize.exponents_offset,
			quantize.operands_offset,
			pairs,
			hidden_width,
			pitch,
			0
		};
		struct kernel_launch decode;
		prepare_launch_fill(
			&decode, b->wire_decode_kernel,
			(V41_DATA_EXPERT_INNER + LINEAR_ELEMENT_THREADS - 1u) /
				LINEAR_ELEMENT_THREADS,
			pairs, LINEAR_ELEMENT_THREADS, 0, &wire, sizeof(wire));
		status = THEN(put(b, &decode));
	} else {
		status = put(b, &launches[LINEAR_ROUTED_EXPERTS_GATE_UP]);
		status =
			wide ? THEN(original_gather_slabs(
				       b, 0, experts->hidden,
				       b->tensor_layer->exact_hidden,
				       experts->pairs_per_row,
				       V41_DATA_EXPERT_INNER,
				       LINEAR_BFLOAT16_BYTES)) :
			       THEN(exact_gather(b, 0, experts->hidden,
						 b->tensor_layer->exact_hidden,
						 pairs * V41_DATA_EXPERT_INNER *
							 LINEAR_BFLOAT16_BYTES,
						 pairs));
		status =
			THEN(put(b, &launches[LINEAR_ROUTED_EXPERTS_QUANTIZE]));
	}
	status = THEN(put(b, &launches[LINEAR_ROUTED_EXPERTS_DOWN]));
	status = THEN(put(b, &launches[LINEAR_ROUTED_EXPERTS_ROW_SUM]));
	return wide ? THEN(original_gather_slabs(b, 1, experts->row_output,
						 b->tensor_layer->exact_sums,
						 1u, MODEL_HIDDEN,
						 LINEAR_SUM_BYTES)) :
		      THEN(exact_gather(b, 1, experts->row_output,
					b->tensor_layer->exact_sums,
					b->rows * MODEL_HIDDEN *
						LINEAR_SUM_BYTES,
					b->rows));
}

/* Routed selection changes only the product launches: quantization and the ordered row sum keep their records. */
MUST_CHECK static result_status
routed_launches(struct builder *b, uint32_t layer,
		const struct linear_routed_experts_description *experts,
		bool is_wide)
{
	if (b->plan->limits.official_reference != 0) {
		struct linear_routed_experts_description source = *experts;
		struct linear_kernels_record kernels = b->linear;
		source.gate_up_slice_count = source.down_slice_count = 1;
		kernels.expert_gate_up[0] = b->official.expert_gate_up;
		kernels.expert_down[0] = b->official.expert_down;
		struct kernel_launch
			launches[LINEAR_ROUTED_EXPERTS_LAUNCH_COUNT];
		result_status status = linear_routed_experts_prepare(
			&source, b->arena, &kernels, launches);
		for (uint32_t index = 0;
		     status == OK && index < LINEAR_ROUTED_EXPERTS_ROW_SUM;
		     index++)
			status = put(b, &launches[index]);
		return status;
	}
    if (b->plan->exl3_bits[layer] != 0)
        return exl3_routed_launches(b, layer, experts);
    struct kernel_launch launches[EXPERT_RELAXED_WIDE_LAUNCH_COUNT];
    uint32_t count = 0;
    const struct expert_relaxed_description description = {
	    .experts = *experts,
	    .tiles = is_wide ?
			     (struct linear_range){
				     ARRAY(ROUTER_TILES).byte_offset,
				     (uint64_t)ROUTER_WIDE_TILES(b->rows) *
					     WIDE_EXPERT_TILE_BYTES } :
			     no_range,
	    .pairs = is_wide ? ARRAY(ROUTER_PAIRS) : no_range,
	    .tile_count = is_wide ? ROUTER_WIDE_TILES(b->rows) : 0,
    };
    result_status status = expert_select_prepare(
	    &description, b->arena, &b->linear, &b->expert, b->arithmetic_class,
	    b->plan->limits.cmp_order, b->plan->expert_folds_admitted[layer],
	    launches, &count);
    uint32_t first = 0;
    if (!is_wide && b->arithmetic_class == ARITHMETIC_CLASS_SM_120) {
	    const struct linear_range arrivals = {
		    b->plan->cards[b->card].producer_arrivals_offset,
		    (uint64_t)experts->job_count *
			    experts->gate_up_shape.output_size /
			    LINEAR_VALUES_PER_BLOCK * sizeof(uint32_t)
	    };
	    status = THEN(expert_producer_prepare(experts, arrivals, b->arena,
						  &b->linear, &b->fused,
						  slots(b, 1)));
	    first = LINEAR_ROUTED_EXPERTS_DOWN;
    }
    /* The thin join consumes pair outputs after the shared expert, and stores the exact gate's row sums. */
    count -= !is_wide && b->tensor_layer == NULL && count != 0 ? 1u : 0u;
    for (uint32_t index = first; index < count && status == OK; index++) {
        status = put(b, &launches[index]);
    }
    return status;
}

/* The FFN site (2.1 launches 21 to 34, REF:989-993, 811-904), *first_launch where an expert pause divides it. */
MUST_CHECK static result_status ffn_launches(struct builder *b, uint32_t layer,
					     uint32_t *first_launch)
{
	const struct v41_tensor *const weights = b->plan->layer_tensors[layer];
	const void *const table = expert_table_of(b, layer);
	const bool is_draft = layer >= MODEL_DRAFT_FIRST_LAYER;
	const uint32_t picks = is_draft ? MODEL_DRAFT_EXPERTS_PER_ROW :
					  MODEL_EXPERTS_PER_ROW;
	struct linear_routed_experts_description experts;
	if (table == NULL) {
		return STATUS_INVALID_ARGUMENT;
	}
	result_status status = routed_of(b, layer, b->rows * picks, &experts);
	struct linear_routed_experts_description router_experts = experts;
	if (b->tensor_layer != NULL && b->tensor_layer->exact_output_shards) {
		if (b->plan->exl3_bits[layer] != 0) {
			const struct bb_exl3_output_shard_plan *const shard =
				b->tensor_layer->exl3_shard;
			if (shard == NULL)
				return STATUS_UNSUPPORTED;
			router_experts.gate_up_shape.output_size =
				shard->gate_up.output_count;
			router_experts.down_shape.output_size =
				shard->down.output_count;
			router_experts.down_shape.input_size =
				shard->down.input_count;
		} else {
			router_experts.gate_up_shape.output_size /=
				b->tensor_layer->rank_count;
			router_experts.down_shape.output_size /=
				b->tensor_layer->rank_count;
		}
    }
    const struct router_description router = {
	    &router_experts,
	    table,
	    ARRAY(STEP),
	    ARRAY(ROUTER_SCORES),
	    WEIGHTS(ROUTER_BIAS)->stored,
	    { b->plan->cards[b->card].expert_table_offset[layer],
	      MODEL_EXPERT_OFFSETS_BYTES },
	    ARRAY(AFFINITIES),
	    is_draft ? LINEAR_ORDER_FIXED : b->plan->limits.linear_order,
	    is_draft ? MODEL_DRAFT_EXPERTS : MODEL_EXPERTS,
	    picks
    };
    struct kernel_launch official_site[HYPER_SITE_LAUNCH_COUNT];
    const bool official_hc = b->plan->limits.official_hc_rows != 0;
    const bool separate_router = is_draft || b->vision_active || official_hc;
    const bool exact_shards = b->tensor_layer != NULL && b->tensor_layer->exact_output_shards;
    /* graph_branches, a thin TC-MoE layer: the shared expert first as a side branch of the step graph (it reads only
     * the front's rows), the routed experts resuming behind the front, the ring joining both (GNmac's shared experts
     * on an auxiliary stream; the same launches, the same bits) */
    const bool side_shared =
	    b->tensor_layer != NULL && b->tensor_layer->graph_branches != 0 &&
	    !exact_shards && b->tensor_layer->exl3_tc_moe &&
	    b->plan->exl3_bits[layer] != 0 && b->rows <= EXL3_TC_MOST_ROWS &&
	    b->slab_count == 1u;
    /* front_branch with the shared side branch: the FFN front as the input front and the separate router (the FFN
     * front's scores and select, the same bits: site_front_records.h), the coefficient front (the FFN mix project,
     * finish and Sinkhorn) a second side run of one launch behind the shared expert's, joined with it */
    const bool ffn_branch = b->front_branch >= FRONT_BRANCH_BOTH && side_shared && !separate_router &&
                            b->plan->host_expert_count[layer] == 0 && !b->tensor_layer->experts_partitioned;
    status = ffn_branch	 ? THEN(site_front_of(b, weights, true, NULL,
					      SITE_FRONT_KIND_INPUT)) :
	     official_hc ? THEN(site_begin(b, weights, true, official_site)) :
			   THEN(site_front(b, weights, true,
					   separate_router ? NULL : &router));
    status = official_hc ? THEN(quantize(b, ARRAY(HIDDEN), MODEL_HIDDEN)) :
			   status;
    /* draft_bf16_operands: the draft FFN front's rows carried again, for the routed experts */
    status = is_draft && b->draft_carry_kernel != BACKEND_NO_KERNEL ?
		     THEN(quantize(b, ARRAY(HIDDEN), MODEL_HIDDEN)) :
		     status;
    if (separate_router || ffn_branch) {
	    if (b->plan->limits.official_reference != 0) {
		    const struct official_router_projection_description
			    projection = { WEIGHTS(ROUTER)->stored,
					   ARRAY(HIDDEN), ARRAY(ROUTER_SCORES),
					   1 };
		    status = THEN(official_router_projection_prepare(
			    &projection, b->arena,
			    b->official.router_projection, slots(b, 1)));
	    } else {
		    status = THEN(dense(b, WEIGHTS(ROUTER), ARRAY(HIDDEN),
					ARRAY(ROUTER_SCORES), F32, 0, 1.0f));
	    }
	    status =
		    b->vision_active && !is_draft ?
			    THEN(image_router_select_prepare(
				    &router, b->plan->vision.router_bias[layer],
				    b->arena, &b->router, &b->image_routing,
				    slots(b, 1))) :
			    THEN(router_select_prepare(&router, b->arena,
						       &b->router,
						       slots(b, 1)));
	    if (b->plan->limits.official_reference != 0)
		    status = THEN(official_router_order_prepare(
			    &router, b->arena, b->official.router_order,
			    slots(b, 1)));
    }
    if (status == OK && b->plan->host_expert_count[layer] != 0) {
        status = expert_pause_add(b, layer, *first_launch);
        *first_launch = b->made->cards[b->card].launch_count;
    }
    if (b->tensor_layer != NULL && b->tensor_layer->experts_partitioned) {
	    const struct ep_filter_description filter = {
		    ARRAY(JOBS), ARRAY(PAIR_HIDDEN), ARRAY(PAIR_OUTPUT),
		    b->tensor_layer->unowned_gate_offset, b->rows
	    };
	    status = THEN(ep_filter_prepare(&filter, b->arena,
					    b->tensor_ep_filter_kernel,
					    slots(b, 1)));
    }
    const uint32_t first_side = b->made->cards[b->card].launch_count;
    status = side_shared ? THEN(shared_launches(b, layer, false)) : status;
    const uint32_t coefficients = b->made->cards[b->card].launch_count; /* ffn_branch: the coefficient front */
    status = ffn_branch ? THEN(site_front_of(b, weights, true, NULL,
					     SITE_FRONT_KIND_COEFFICIENTS)) :
			  status;
    const uint32_t first_routed = b->made->cards[b->card].launch_count;
    /* the shared expert a side branch before the routed experts: their combine may add the routed sums into its rows
     * (shared_sum; the combine then joins the branch and the ring sums those rows in place) */
    b->shared_sum_requested = side_shared && b->tensor_layer->bf16_ring && !b->tensor_layer->launches_untrimmed &&
        b->rows <= b->tensor_layer->ring_rows;
    b->shared_sum_fused = false;
    status = b->plan->exl3_bits[layer] != 0 ? THEN(exl3_routed_launches(b, layer, &experts)) :
        exact_shards ? THEN(exact_routed_launches(b, &experts)) : THEN(routed_launches(b, layer, &experts, false));
    const bool shared_sum = b->shared_sum_fused;
    b->shared_sum_requested = b->shared_sum_fused = false;
    status = side_shared ? status : THEN(shared_launches(b, layer, false));
    const uint32_t first_join = b->made->cards[b->card].launch_count;
    if (b->tensor_layer != NULL && !b->ffn_post_deferred) {
	    struct kernel_launch site[HYPER_SITE_LAUNCH_COUNT];
	    const struct v41_tp2_layer_plan *const tensor = b->tensor_layer;
	    const struct linear_range sums = ARRAY(ROW_SUMS);
	    if (exact_shards) {
		    b->arrays[V41_DATA_ARRAY_ROW_SUMS] = tensor->exact_sums;
		    b->tensor_layer = NULL;
	    } else {
		    status = THEN(tensor_complete(
			    b, 1, ARRAY(SHARED_OUTPUT),
			    shared_sum ? (struct linear_range){ 0, 0 } :
					 ARRAY(ROW_SUMS)));
	    }
        if (side_shared && status == OK) {
            const uint32_t end = b->made->cards[b->card].launch_count;
            /* the branch joins at the ring, or with shared_sum at the combine (the routed launches' last) */
            const uint32_t join = shared_sum ? first_join - 1u : first_join;
            if (first_side == first_routed || first_routed >= join || first_join == end)
		    return STATUS_INVALID_ARGUMENT;
	    for (uint32_t launch = first_side; launch < first_routed; launch++)
		    b->launches[b->card][launch].branch = LAUNCH_SIDE;
	    if (ffn_branch) /* a side run of its own behind the same fork */
		    b->launches[b->card][coefficients].branch =
			    LAUNCH_SIDE_MORE;
	    b->launches[b->card][first_routed].branch = LAUNCH_RESUME;
	    b->launches[b->card][join].branch = LAUNCH_JOIN;
	}
        status = THEN(site_prepare(b, weights, true, site));
        b->tensor_layer = tensor;
	b->arrays[V41_DATA_ARRAY_ROW_SUMS] = sums;
	status = THEN(put(b, &site[HYPER_SITE_POST]));
    }
    return status;
}

/* batch_draft_wide: a draft layer's FFN sum (the shared expert's rows plus the routed sums, BF16 rings of up to
 * ring_rows rows) and post over the rows the next launches take, which ffn_launches left a slab at a time
 * (ffn_post_deferred): a ring's sums are element by element, a post's row by row, so the same bits. */
MUST_CHECK static result_status ffn_post(struct builder *b, uint32_t layer)
{
	struct kernel_launch site[HYPER_SITE_LAUNCH_COUNT];
	result_status status = tensor_complete_slabs(b, 1, ARRAY(SHARED_OUTPUT),
						     ARRAY(ROW_SUMS));
	status = THEN(
		site_prepare(b, b->plan->layer_tensors[layer], true, site));
	return THEN(put(b, &site[HYPER_SITE_POST]));
}

/* batch_draft_wide 2, a draft layer's FFN site over a batched draft's rows (the builder a chunk of them, its slabs'
 * views each a slab block): over all rows the mix and finish (site_begin); a slab at a time the router's scores, the
 * rows' quantization (as the thin site front quantizes them), the draft's selection and its routed experts (the
 * narrow form of 8 rows); over all rows the shared expert (the _64 form from the BF16 rows), the TP sum and the
 * post. The scores and the codes stay a slab's: a launch over the chunk lays its rows out packed at its width (the
 * router's 128 scores, a row's 5,120 codes), not at the arrays' rows (384 scores, 8,192 codes) at which a slab's
 * launches read them. Each launch over the rows has the thin steps' bytes (the wide forms' rule, as a BATCH
 * program's target layers), so the draft's bits stay. */
MUST_CHECK static result_status draft_wide_ffn(struct builder *b,
					       uint32_t layer)
{
    const uint32_t slabs = b->draft_slab_count;
    const struct v41_tensor *const weights = b->plan->layer_tensors[layer];
    const void *const table = expert_table_of(b, layer);
    struct kernel_launch site[HYPER_SITE_LAUNCH_COUNT];
    if (table == NULL || b->tensor_layer == NULL)
	    return STATUS_INVALID_ARGUMENT;
    draft_slab_select(b, slabs);
    result_status status = site_begin(b, weights, true, site);
    for (uint32_t slab = 0; slab < slabs && status == OK; slab++) {
        draft_slab_select(b, slab);
        status = dense(b, WEIGHTS(ROUTER), ARRAY(HIDDEN), ARRAY(ROUTER_SCORES), F32, 0, 1.0f);
	status = THEN(quantize(b, ARRAY(HIDDEN),
			       MODEL_HIDDEN)); /* the thin site front's codes */
	struct linear_routed_experts_description experts;
	status = routed_of(b, layer, b->rows * MODEL_DRAFT_EXPERTS_PER_ROW,
			   &experts);
	struct linear_routed_experts_description router_experts = experts;
	const struct router_description router = {
		&router_experts,
		table,
		ARRAY(STEP),
		ARRAY(ROUTER_SCORES),
		WEIGHTS(ROUTER_BIAS)->stored,
		{ b->plan->cards[b->card].expert_table_offset[layer],
		  MODEL_EXPERT_OFFSETS_BYTES },
		ARRAY(AFFINITIES),
		LINEAR_ORDER_FIXED,
		MODEL_DRAFT_EXPERTS,
		MODEL_DRAFT_EXPERTS_PER_ROW
	};
	status = THEN(router_select_prepare(&router, b->arena, &b->router,
					    slots(b, 1)));
	status = THEN(routed_launches(b, layer, &experts, false));
    }
    draft_slab_select(b, slabs);
    status = THEN(shared_launches(b, layer, true));
    status = THEN(tensor_complete_slabs(b, 1, ARRAY(SHARED_OUTPUT), ARRAY(ROW_SUMS)));
    status = THEN(site_prepare(b, weights, true, site));
    return THEN(put(b, &site[HYPER_SITE_POST]));
}

/* batch_draft_wide 2: a draft layer over a batched draft's rows as a BATCH program runs a target layer over its slabs
 * (chunk_walk's order): over all rows the attention site's mix and finish, wq_a with wkv, qr, wq_b and the queries;
 * a slab at a time its state writes and core (through its slab block: a verifier's block as the slab walk sees it);
 * over all rows wo_a, wo_b, the TP sum and the post; then the FFN site (draft_wide_ffn). The builder takes the draft's
 * rows as its chunk meanwhile (its slabs, first row 0, view 0), then the program's again. */
MUST_CHECK static result_status draft_wide_layer(struct builder *b,
						 uint32_t layer)
{
    const uint32_t slabs = b->slab_count, first = b->chunk_first, rows = b->chunk_rows, view = b->chunk_view;
    b->slab_count = b->draft_slab_count;
    b->chunk_first = 0;
    b->chunk_rows = b->draft_rows;
    b->chunk_view = 0;
    draft_slab_select(b, b->draft_slab_count);
    result_status status = layer_steps(b, layer, 0, 0,
				       SITE_MIX | INPUT_PRODUCTS | QUERY_NORM |
					       QUERY_PRODUCTS | QUERIES);
    /* level 3: the window writes and the block core once over all rows (the prepare's window part: the same ring
     * entries; each verifier's block whole, as the reference's block: other drafts where a slab cut it) */
    status = b->draft_wide_step ? THEN(layer_steps(b, layer, 0, 0, WINDOW_WRITES | CORE)) : status;
    for (uint32_t slab = 0; slab < b->draft_slab_count && status == OK && !b->draft_wide_step; slab++) {
        draft_slab_select(b, slab);
        status = layer_steps(b, layer, 0, 0, STATE_WRITES | CORE);
    }
    draft_slab_select(b, b->draft_slab_count);
    status = THEN(layer_steps(b, layer, 0, 0, OUTPUT_PRODUCTS | SITE_POST));
    status = THEN(draft_wide_ffn(b, layer));
    b->slab_count = slabs;
    b->chunk_first = first;
    b->chunk_rows = rows;
    b->chunk_view = view;
    return status;
}

/* A wide layer's FFN site (agent H's B, wide_prefill.md 3.1): of each piece its mix, finish and router scores; of the
 * chunk the wide select (router.h, its jobs into the job table), the quantization and the routed experts; of each
 * piece the shared expert (wide trunk launches) and the post. A piece's launches have the bytes of the thin steps of
 * its slabs (dense_linear.h's wide launches, a hyper site's over its rows) and every expert's pair is computed as in a
 * thin step's job (the composed card test), so the bits are the thin steps'. */
MUST_CHECK static result_status exl3_wide_ffn_launches(struct builder *b,
						       uint32_t layer);

MUST_CHECK static result_status wide_ffn_launches(struct builder *b,
						  uint32_t layer)
{
    if (b->tensor_layer != NULL && b->tensor_layer->exl3_wide_entry && b->plan->exl3_bits[layer] != 0 &&
        (b->tensor_layer->exact_output_shards || b->tensor_layer->bf16_ring))
        return exl3_wide_ffn_launches(b, layer);
    const bool original_wide = b->tensor_layer != NULL &&
			       b->tensor_layer->original_wide_experts &&
			       b->tensor_layer->exact_output_shards &&
			       b->plan->exl3_bits[layer] == 0;
    if (b->tensor_layer != NULL && !original_wide &&
	(b->tensor_layer->exact_output_shards ||
	 b->plan->exl3_bits[layer] != 0)) {
	    result_status exact_status = OK;
	    for (uint32_t slab = 0; slab < b->slab_count && exact_status == OK;
		 slab++) {
		    slab_select(b, slab);
		    uint32_t first = b->made->cards[b->card].launch_count;
		    exact_status = ffn_launches(b, layer, &first);
	    }
	    return exact_status;
    }
    const struct v41_tensor *const weights = b->plan->layer_tensors[layer];
    const void *const table = expert_table_of(b, layer);
    struct kernel_launch site[HYPER_SITE_LAUNCH_COUNT];
    struct linear_routed_experts_description experts;
    result_status status = table == NULL ? STATUS_INVALID_ARGUMENT : OK;
    const uint32_t pieces = original_wide ? b->slab_count : piece_count_of(b);
    for (uint32_t piece = 0; piece < pieces && status == OK; piece++) {
	    if (original_wide) {
		    /* Keep the exact recipe's fused eight-row front, router arithmetic and quantizer. The wide select
             * below replaces only its temporary jobs; neither row grouping nor the shared/post products widen. */
		    slab_select(b, piece);
		    struct linear_routed_experts_description routed;
		    status = routed_of(
			    b, layer, b->rows * MODEL_EXPERTS_PER_ROW, &routed);
		    if (status != OK)
			    break;
		    routed.gate_up_shape.output_size /=
			    b->tensor_layer->rank_count;
		    routed.down_shape.output_size /=
			    b->tensor_layer->rank_count;
		    const struct router_description router = {
			    &routed,
			    table,
			    ARRAY(STEP),
			    ARRAY(ROUTER_SCORES),
			    WEIGHTS(ROUTER_BIAS)->stored,
			    { b->plan->cards[b->card].expert_table_offset[layer],
			      MODEL_EXPERT_OFFSETS_BYTES },
			    ARRAY(AFFINITIES),
			    b->plan->limits.linear_order,
			    MODEL_EXPERTS,
			    MODEL_EXPERTS_PER_ROW
		    };
		    status = THEN(site_front(b, weights, true, &router));
	    } else {
		    piece_select(b, piece);
		    status = site_begin(b, weights, true, site);
		    status = THEN(dense(b, WEIGHTS(ROUTER), ARRAY(HIDDEN),
					ARRAY(ROUTER_SCORES), F32, 0, 1.0f));
	    }
    }
    chunk_select(b);
    status = THEN(routed_of(b, layer, ROUTER_WIDE_JOBS(b->rows), &experts));
    /* Original exact shards keep the complete K chain. Only the router's bounds name local weight rectangles;
     * the product description retains the global shape so its quantizer consumes the gathered full hidden row. */
    struct matrix_shape_record gate_up_shape = experts.gate_up_shape,
			       down_shape = experts.down_shape;
    if (original_wide) {
	    gate_up_shape.output_size /= b->tensor_layer->rank_count;
	    down_shape.output_size /= b->tensor_layer->rank_count;
    }
    const struct router_wide_description select = {
	    .expert_offsets = table,
	    .step = ARRAY(STEP),
	    .scores = ARRAY(ROUTER_SCORES),
	    .bias = WEIGHTS(ROUTER_BIAS)->stored,
	    .offsets = { b->plan->cards[b->card].expert_table_offset[layer],
			 MODEL_EXPERT_OFFSETS_BYTES },
	    .expert_weights = b->plan->experts[layer],
	    .affinities = ARRAY(AFFINITIES),
	    .picks = ARRAY(ROUTER_PICKS),
	    .counts = { ARRAY(ROUTER_COUNTS).byte_offset,
			(uint64_t)(b->rows + ROUTER_SORT_ROWS - 1u) /
				ROUTER_SORT_ROWS * MODEL_EXPERTS *
				sizeof(uint32_t) },
	    .pairs = ARRAY(ROUTER_PAIRS),
	    .runs = ARRAY(ROUTER_RUNS),
	    .tiles = { ARRAY(ROUTER_TILES).byte_offset,
		       (uint64_t)ROUTER_WIDE_TILES(b->rows) *
			       sizeof(struct wide_expert_tile) },
	    .jobs = experts.jobs,
	    .gate_up_shape = gate_up_shape,
	    .down_shape = down_shape,
	    .weight_format = experts.weight_format,
	    .order = b->plan->limits.linear_order,
	    .row_count = b->rows
    };
    status = b->vision_active ?
		     THEN(image_router_wide_prepare(
			     &select, b->plan->vision.router_bias[layer],
			     b->arena, &b->router, &b->image_routing,
			     slots(b, ROUTER_WIDE_LAUNCH_COUNT))) :
		     THEN(router_wide_prepare(&select, b->arena, &b->router,
					      slots(b,
						    ROUTER_WIDE_LAUNCH_COUNT)));
    status = THEN(quantize(b, ARRAY(HIDDEN), MODEL_HIDDEN));
    status = original_wide ? THEN(exact_routed_launches(b, &experts)) :
			     THEN(routed_launches(b, layer, &experts, true));
    for (uint32_t piece = 0; piece < pieces && status == OK; piece++) {
	    if (original_wide) {
		    slab_select(b, piece);
	    } else {
		    piece_select(b, piece);
	    }
	    status = shared_launches(b, layer, true);
	    if (b->tensor_layer != NULL && !original_wide) {
		    status = THEN(tensor_complete(b, 1, ARRAY(SHARED_OUTPUT),
						  ARRAY(ROW_SUMS)));
	    }
	    if (original_wide) {
		    const struct v41_tp2_layer_plan *const tensor =
			    b->tensor_layer;
		    const struct linear_range sums = ARRAY(ROW_SUMS);
		    const uint64_t row_bytes =
			    (uint64_t)MODEL_HIDDEN * LINEAR_SUM_BYTES;
		    b->arrays[V41_DATA_ARRAY_ROW_SUMS] = (struct linear_range){
			    tensor->exact_sums.byte_offset +
				    (uint64_t)b->first_row * row_bytes,
			    (uint64_t)b->rows * row_bytes
		    };
		    b->tensor_layer = NULL;
		    status = THEN(site_prepare(b, weights, true, site));
		    b->tensor_layer = tensor;
		    b->arrays[V41_DATA_ARRAY_ROW_SUMS] = sums;
	    } else {
		    status = THEN(site_prepare(b, weights, true, site));
	    }
	    status = THEN(put(b, &site[HYPER_SITE_POST]));
    }
    return status;
}

/* exl3_wide_entry, an EXL3 tensor layer in a chunked step (model_v41_tp2_stage.h): of each piece
 * its mix, finish and router scores (wide_ffn_launches' launches, the thin fronts' bytes); of the chunk the wide
 * select (8-row jobs, an expert's runs adjacent), the quantization and ONE EXL3 expert call over its rows (the local-K
 * passes and row sums, or the exact recipe's output-shard passes with their gathers a slab at a time; every pair's
 * words those of a thin step's job); of each slab of 8 rows the shared expert, the BF16 ring sum (or the exact
 * recipe's gathered sums) and the post exactly as ffn_launches runs them for a thin step. */
/* A wide step's router scores (float32 [rows][outputs]): router_mma's split-K tensor-core product and its reduce for 9
 * to 64 rows where prepared, else the dense product (its order). */
MUST_CHECK static result_status router_scores(struct builder *b,
					      const struct v41_tensor *router)
{
	const struct matrix_shape_record *const shape = &router->shape;
	const struct linear_range partials =
		b->tensor_layer != NULL ? b->tensor_layer->router_partials :
					  (struct linear_range){ 0, 0 };
	const uint64_t partial_bytes = (uint64_t)V41_DATA_ROUTER_MMA_SPLITS *
				       b->rows * shape->output_size *
				       sizeof(float);
	/* router_mma_wide: more than 64 rows by the 64-row form over blocks of 64 rows (grid y) */
	const bool wide = b->rows > HEAD_MMA_MOST_ROWS &&
			  b->router_mma_wide_kernel != BACKEND_NO_KERNEL;
	if (b->router_mma_kernel == BACKEND_NO_KERNEL ||
	    b->rows <= LINEAR_MAXIMUM_ROWS ||
	    (b->rows > HEAD_MMA_MOST_ROWS && !wide) ||
	    shape->output_size % HEAD_MMA_TILE_OUTPUTS != 0 ||
	    shape->input_size % (HEAD_MMA_STEP_INPUTS *
				 V41_DATA_ROUTER_MMA_SPLITS) !=
		    0 ||
	    router->stored.byte_count != (uint64_t)shape->output_size *
						 shape->input_size *
						 sizeof(uint16_t) ||
	    partials.byte_count < partial_bytes)
		return dense_cone(b, router, ARRAY(HIDDEN),
				  ARRAY(ROUTER_SCORES), F32, 0, 1.0f,
				  b->ffn_cone_layer);
	const struct router_mma_arguments record = {
		{ .weights_offset = router->stored.byte_offset,
		  .rows_offset = ARRAY(HIDDEN).byte_offset,
		  .output_offset = ARRAY(ROUTER_SCORES).byte_offset,
		  .step_offset = 0,
		  .output_size = shape->output_size,
		  .input_size = shape->input_size,
		  .row_count = b->rows,
		  .output_type = F32,
		  .flags = 0,
		  .output_scale = 1.0f,
		  .epsilon = 0.0f,
		  .cone_layer = 0 },
		partials.byte_offset,
		V41_DATA_ROUTER_MMA_SPLITS,
		0
	};
	const uint32_t items = shape->output_size / HEAD_MMA_TILE_OUTPUTS *
			       V41_DATA_ROUTER_MMA_SPLITS;
	const uint32_t results = b->rows * shape->output_size;
	struct kernel_launch launches[2];
	prepare_launch_fill(&launches[0],
			    wide ? b->router_mma_wide_kernel :
				   b->router_mma_kernel,
			    (items + ROUTER_MMA_WARPS - 1u) / ROUTER_MMA_WARPS,
			    wide ? (b->rows + HEAD_MMA_MOST_ROWS - 1u) /
					    HEAD_MMA_MOST_ROWS :
				   1u,
			    32u * ROUTER_MMA_WARPS, 0, &record, sizeof(record));
	prepare_launch_fill(&launches[1], b->router_mma_reduce_kernel,
			    (results + ROUTER_MMA_REDUCE_THREADS - 1u) /
				    ROUTER_MMA_REDUCE_THREADS,
			    1, ROUTER_MMA_REDUCE_THREADS, 0, &record,
			    sizeof(record));
	result_status status = put(b, &launches[0]);
	return THEN(put(b, &launches[1]));
}

/* Whether shared_launches(b, layer, true) takes the BATCH trunk_mma_64 path: from the front's BF16 rows alone (no
 * quantized codes), so it may run as a branch beside the router and the routed experts. */
static bool shared_wide_mma(struct builder *b, uint32_t layer)
{
	const struct v41_tensor *const weights = b->plan->layer_tensors[layer];
	struct linear_matrix gate, down;
	if (b->tensor_layer == NULL || !thin_forms_of(b) ||
	    matrix_of(b, WEIGHTS(SHARED_GATE), TRUNK, &gate) != OK ||
	    matrix_of(b, WEIGHTS(SHARED_DOWN), TRUNK, &down) != OK)
		return false;
	const struct linear_trunk_bfloat16_description gate_product = {
		.matrix = gate,
		.values = ARRAY(HIDDEN),
		.output = ARRAY(SHARED_GATE),
		.row_count = b->rows,
		.segment_count = 1
	};
	const struct linear_trunk_bfloat16_description down_product = {
		.matrix = down,
		.values = ARRAY(PAIR_HIDDEN),
		.output = ARRAY(SHARED_OUTPUT),
		.row_count = b->rows,
		.segment_count = 1
	};
	return trunk_mma_wide_applies(b, &gate_product) &&
	       trunk_mma_wide_applies(b, &down_product);
}

/* exl3_wide_ffn_launches' phases (split_overlap runs them for each half of a chunk, the sums as graph side runs): the
 * compute (the front: mix, router, its selection, the rows quantized; the routed experts; the shared expert), the sums
 * and the post. split_overlap's joint mode runs the fronts and shared experts a half, the routed experts once over the
 * chunk's rows (a row's experts never read another row: the same bits). */
#define FFN_FRONT 1u
#define FFN_SUM 2u
#define FFN_POST 4u
#define FFN_EXPERTS 8u
#define FFN_SHARED 16u
#define FFN_EXPERTS_FRONT 32u /* mode 7: the joint call's route and gate/up (the halves' route) */
#define FFN_DOWN_FIRST 64u    /* its first half's down operands, down GEMM and combine */
#define FFN_DOWN_SECOND 128u  /* its second half's */
#define FFN_COMPUTE (FFN_FRONT | FFN_EXPERTS | FFN_SHARED)
#define FFN_PHASES (FFN_COMPUTE | FFN_SUM | FFN_POST)
MUST_CHECK static result_status
exl3_wide_ffn_phases(struct builder *b, uint32_t layer, uint32_t phases)
{
	const struct v41_tensor *const weights = b->plan->layer_tensors[layer];
	const void *const table = expert_table_of(b, layer);
	struct kernel_launch site[HYPER_SITE_LAUNCH_COUNT];
	struct linear_routed_experts_description experts = { 0 };
	result_status status = table == NULL || b->vision_active ?
				       STATUS_INVALID_ARGUMENT :
				       OK;
	/* a full program's decoder layer: the FFN front's mix and router leave the rows the routed experts leave */
	b->ffn_cone_layer = b->made->program.kind == PROGRAM_FULL &&
					    layer >=
						    MODEL_DECODER_FIRST_LAYER &&
					    layer < MODEL_LAYERS ?
				    layer :
				    0u;
	for (uint32_t piece = 0; piece < piece_count_of(b) && status == OK &&
				 (phases & FFN_FRONT) != 0;
	     piece++) {
		piece_select(b, piece);
		status = site_begin(b, weights, true, site);
		/* prefill_dead_halves: a dead half's router alone (its rows the cone's: no gather) */
		/* router_first_whole: the first half's alone, beside the second half's attention sums */
		const bool first_whole =
			b->tp_description != NULL &&
			b->tp_description->stages[b->card]
				->description.router_first_whole &&
			selected_half(b) == 0;
		const bool gathered = gathers_apply(b) &&
				      b->rows >= ROUTER_GATHER_LEAST_ROWS &&
				      !half_dead(b, layer, selected_half(b)) &&
				      !first_whole;
		status = THEN(gathered ? router_gather(b, WEIGHTS(ROUTER)) :
					 router_scores(b, WEIGHTS(ROUTER)));
	}
    b->ffn_cone_layer = 0;
    chunk_select(b);
    /* a BATCH step on the TC MoE: with graph_branches the shared expert first as a side branch (ffn_launches');
     * without, and launches_untrimmed off, first in the chain where its one ring takes the step (the combine then adds
     * the routed sums into its rows, as the branch's does: the same bits, no join launch) */
    const bool shared_ahead =
	    phases == FFN_PHASES && !b->tensor_layer->exact_output_shards &&
	    b->tensor_layer->exl3_tc_moe && b->rows <= EXL3_TC_MOST_ROWS &&
	    piece_count_of(b) == 1u && status == OK &&
	    shared_wide_mma(b, layer);
    const bool side_shared = shared_ahead && b->tensor_layer->graph_branches >= 2u;
    const bool chain_shared = shared_ahead && !side_shared &&
			      !b->tensor_layer->launches_untrimmed &&
			      b->tensor_layer->bf16_ring &&
			      b->tensor_layer->ring_rows >
				      LINEAR_MAXIMUM_ROWS &&
			      b->chunk_rows <= b->tensor_layer->ring_rows;
    const uint32_t first_side = b->made->cards[b->card].launch_count;
    status = side_shared || chain_shared ? THEN(shared_launches(b, layer, true)) : status;
    const uint32_t first_routed = b->made->cards[b->card].launch_count;
    status = (phases & (FFN_FRONT | FFN_EXPERTS | FFN_EXPERTS_FRONT |
			FFN_DOWN_FIRST | FFN_DOWN_SECOND)) != 0 ?
		     THEN(routed_of(b, layer, ROUTER_WIDE_JOBS(b->rows),
				    &experts)) :
		     status;
    /* the exact recipe's table names the rank's output shards (as ffn_launches' router shapes) */
    struct matrix_shape_record gate_up_shape = experts.gate_up_shape,
			       down_shape = experts.down_shape;
    if (b->tensor_layer->exact_output_shards) {
        const struct bb_exl3_output_shard_plan *const shard = b->tensor_layer->exl3_shard;
        if (shard == NULL)
		return STATUS_UNSUPPORTED;
	gate_up_shape.output_size = shard->gate_up.output_count;
        down_shape.output_size = shard->down.output_count;
        down_shape.input_size = shard->down.input_count;
    }
    const struct router_wide_description select = {
	    .expert_offsets = table,
	    .step = ARRAY(STEP),
	    .scores = ARRAY(ROUTER_SCORES),
	    .bias = WEIGHTS(ROUTER_BIAS)->stored,
	    .offsets = { b->plan->cards[b->card].expert_table_offset[layer],
			 MODEL_EXPERT_OFFSETS_BYTES },
	    .expert_weights = b->plan->experts[layer],
	    .affinities = ARRAY(AFFINITIES),
	    .picks = ARRAY(ROUTER_PICKS),
	    .counts = { ARRAY(ROUTER_COUNTS).byte_offset,
			(uint64_t)(b->rows + ROUTER_SORT_ROWS - 1u) /
				ROUTER_SORT_ROWS * MODEL_EXPERTS *
				sizeof(uint32_t) },
	    .pairs = ARRAY(ROUTER_PAIRS),
	    .runs = ARRAY(ROUTER_RUNS),
	    .tiles = { ARRAY(ROUTER_TILES).byte_offset,
		       (uint64_t)ROUTER_WIDE_TILES(b->rows) *
			       sizeof(struct wide_expert_tile) },
	    .jobs = experts.jobs,
	    .gate_up_shape = gate_up_shape,
	    .down_shape = down_shape,
	    .weight_format = experts.weight_format,
	    .order = b->plan->limits.linear_order,
	    .row_count = b->rows
    };
    if ((phases & FFN_FRONT) != 0) {
        /* the TC prefill MoE (exl3_tcp) reads the router's picks alone: the counts, sorted pairs and job table only for
         * the other forms (the experts of more than 64 rows: a half's, or with the joint mode the chunk's) */
	const bool picks_only =
		b->tensor_layer->exl3_tc_moe &&
		!b->tensor_layer->exact_output_shards &&
		b->plan->exl3_bits[layer] != 0 && b->rows > tc_most_rows(b) &&
		b->rows <= EXL3_TCP_CHUNK_ROWS &&
		b->tensor_layer->exl3_tcp_workspace.byte_count != 0;
	/* at most a sort block of rows: the count, place and jobs in one launch (the same tables; launches_untrimmed:
         * the three) */
	const bool small = !picks_only && b->rows <= ROUTER_SORT_ROWS &&
			   !b->tensor_layer->launches_untrimmed;
	struct kernel_launch router_launches[ROUTER_WIDE_LAUNCH_COUNT];
	status = THEN(small ? router_wide_small_prepare(&select, b->arena,
							&b->router,
							router_launches) :
			      router_wide_prepare(&select, b->arena, &b->router,
						  router_launches));
	const uint32_t router_count =
		picks_only ? 1u :
		small	   ? ROUTER_WIDE_SMALL_LAUNCH_COUNT :
			     (uint32_t)ROUTER_WIDE_LAUNCH_COUNT;
	for (uint32_t launch = 0; launch < router_count && status == OK; launch++)
            status = put(b, &router_launches[launch]);
        /* a BATCH step whose routed experts take the TC MoE (BF16 rows) and whose shared expert the _64 form (BF16
         * rows) reads no quantized rows: no quantization (the same bits, a launch a layer fewer) */
	const bool unquantized =
		thin_forms_of(b) && b->tensor_layer->exl3_tc_moe &&
		!b->tensor_layer->launches_untrimmed &&
		!b->tensor_layer->exact_output_shards &&
		!b->tensor_layer->experts_partitioned &&
		b->plan->exl3_bits[layer] != 0 &&
		b->rows <= EXL3_TC_MOST_ROWS && shared_wide_mma(b, layer);
	status = unquantized ? status :
			       THEN(quantize(b, ARRAY(HIDDEN), MODEL_HIDDEN));
    }
    /* a BATCH step's one ring (ring_rows above 8) with the shared expert a side branch: the combine may add the routed
     * sums into the shared rows (shared_sum, as ffn_launches') */
    b->shared_sum_requested =
	    (side_shared || chain_shared) && (phases & FFN_SUM) != 0 &&
	    b->tensor_layer->bf16_ring &&
	    !b->tensor_layer->launches_untrimmed &&
	    b->tensor_layer->ring_rows > LINEAR_MAXIMUM_ROWS &&
	    b->chunk_rows <= b->tensor_layer->ring_rows;
    b->shared_sum_fused = false;
    status = (phases & FFN_EXPERTS) != 0 ? THEN(exl3_routed_launches(b, layer, &experts)) : status;
    /* mode 7: a part of the joint call with the halves' route (exl3_routed_launches' tcp_emit) */
    b->tcp_emit = (phases & FFN_EXPERTS_FRONT) != 0 ? TCP_EMIT_FRONT :
                  (phases & FFN_DOWN_FIRST) != 0    ? TCP_EMIT_FIRST_HALF :
                  (phases & FFN_DOWN_SECOND) != 0   ? TCP_EMIT_SECOND_HALF : TCP_EMIT_ALL;
    status = b->tcp_emit != TCP_EMIT_ALL ? THEN(exl3_routed_launches(b, layer, &experts)) : status;
    b->tcp_emit = TCP_EMIT_ALL;
    const bool shared_sum = b->shared_sum_fused;
    b->shared_sum_requested = b->shared_sum_fused = false;
    const struct v41_tp2_layer_plan *const tensor = b->tensor_layer;
    /* the shared expert and the post a piece (wide launches, the thin steps' bytes); the BF16 recipe's ring sum of the
     * shared output and the routed sums a slab (its bank geometry), between them */
    for (uint32_t piece = 0; piece < piece_count_of(b) && status == OK && !side_shared && !chain_shared &&
         (phases & FFN_SHARED) != 0; piece++) {
        piece_select(b, piece);
        status = shared_launches(b, layer, true);
    }
    const uint32_t first_join = b->made->cards[b->card].launch_count;
    if ((phases & FFN_SUM) == 0) {
        /* the sums in another call */
    } else if (!tensor->exact_output_shards &&
	       tensor->ring_rows > LINEAR_MAXIMUM_ROWS) {
	    /* BATCH and piece_rings: one BF16 ring a piece of ring_rows rows (the whole of a BATCH step), not a slab */
	    for (uint32_t at = 0; at < b->chunk_rows && status == OK;
		 at += tensor->ring_rows) {
		    const uint32_t left = b->chunk_rows - at;
		    rows_select(b, b->chunk_view, b->chunk_first + at,
				left < tensor->ring_rows ? left :
							   tensor->ring_rows);
		    status = tensor_complete(
			    b, 1, ARRAY(SHARED_OUTPUT),
			    shared_sum ? (struct linear_range){ 0, 0 } :
					 ARRAY(ROW_SUMS));
	    }
    } else if (!tensor->exact_output_shards && tensor->bf16_ring &&
	       tensor->shared_channels &&
	       tensor->wide_exchange == V41_DATA_TP_EXCHANGE_OWNER_RINGS &&
	       b->chunk_rows > LINEAR_MAXIMUM_ROWS && piece_count_of(b) == 1u &&
	       !thin_forms_of(b)) {
	    /* the slabs' rings in one exchange (the same bits) */
	    chunk_select(b);
	    status = tensor_complete_owner_rings(b, 1, ARRAY(SHARED_OUTPUT),
						 ARRAY(ROW_SUMS));
    } else if (!tensor->exact_output_shards && tensor->bf16_ring &&
	       tensor->shared_channels &&
	       tensor->wide_exchange == V41_DATA_TP_EXCHANGE_CE_RINGS &&
	       b->chunk_rows > LINEAR_MAXIMUM_ROWS && piece_count_of(b) == 1u &&
	       !thin_forms_of(b) && tensor->rank_count == TP4_RANKS &&
	       tensor->collectives[1].arguments.incoming_host_mask == 0) {
	    /* the slabs' rings by copy engines (the same bits) */
	    chunk_select(b);
	    status = tensor_complete_ce_rings(b, 1, ARRAY(SHARED_OUTPUT),
					      ARRAY(ROW_SUMS));
    } else {
	    for (uint32_t slab = 0; slab < b->slab_count && status == OK &&
				    !tensor->exact_output_shards;
		 slab++) {
		    slab_select(b, slab);
		    status = tensor_complete(b, 1, ARRAY(SHARED_OUTPUT),
					     ARRAY(ROW_SUMS));
	    }
    }
    if (side_shared && status == OK) {
        const uint32_t end = b->made->cards[b->card].launch_count;
        const uint32_t join = shared_sum ? first_join - 1u : first_join; /* shared_sum: at the combine */
        if (first_side == first_routed || first_routed >= join || first_join == end)
		return STATUS_INVALID_ARGUMENT;
	for (uint32_t launch = first_side; launch < first_routed; launch++)
		b->launches[b->card][launch].branch = LAUNCH_SIDE;
	b->launches[b->card][first_routed].branch = LAUNCH_RESUME;
	b->launches[b->card][join].branch = LAUNCH_JOIN;
    }
    const uint64_t sums_row_bytes = (uint64_t)MODEL_HIDDEN * sizeof(float);
    for (uint32_t piece = 0; piece < piece_count_of(b) && status == OK && (phases & FFN_POST) != 0; piece++) {
        piece_select(b, piece);
        if (tensor->exact_output_shards) { /* the exact recipe: the piece's gathered sums, as ffn_launches' post */
		const struct linear_range sums = ARRAY(ROW_SUMS);
		b->arrays[V41_DATA_ARRAY_ROW_SUMS] = (struct linear_range){
			tensor->exact_sums.byte_offset +
				(uint64_t)b->first_row * sums_row_bytes,
			(uint64_t)b->rows * sums_row_bytes
		};
		b->tensor_layer = NULL;
		status = site_prepare(b, weights, true, site);
		b->tensor_layer = tensor;
		b->arrays[V41_DATA_ARRAY_ROW_SUMS] = sums;
	} else {
            status = site_prepare(b, weights, true, site);
        }
	status = THEN(put(b, &site[HYPER_SITE_POST]));
    }
    chunk_select(b);
    return status;
}

MUST_CHECK static result_status exl3_wide_ffn_launches(struct builder *b,
						       uint32_t layer)
{
    return exl3_wide_ffn_phases(b, layer, FFN_PHASES);
}

/* The card's step segment, its rotation step where it runs a layer (model_seam.h); or the embedding (REF:1253). */
/* The site fronts' scratch zeroed (their counters start at zero): a thin step's first launch. In a plan of wide rows
 * the scratch lies in the router picks (model_v41.c), which a chunked step's wide FFN writes, so a BATCH program's
 * draft walks, thin walks behind a wide target, zero it again before their fronts. */
MUST_CHECK static result_status front_scratch_fill(struct builder *b)
{
    const uint64_t offset = b->plan->cards[b->card].front_scratch_offset;
    if (offset > b->arena || V41_DATA_FUSED_SCRATCH_BYTES > b->arena - offset)
	    return STATUS_INVALID_ARGUMENT;
    const struct fill_record zeros = {
	    offset, V41_DATA_FUSED_SCRATCH_BYTES / FILL_WORD_BYTES, 0, 0
    };
    struct kernel_launch fill;
    const uint32_t blocks =
	    (uint32_t)((zeros.word_count + FILL_THREADS_PER_BLOCK_VALUE - 1u) /
		       FILL_THREADS_PER_BLOCK_VALUE);
    prepare_launch_fill(&fill, b->fill, blocks, 1, FILL_THREADS_PER_BLOCK_VALUE,
			0, &zeros, sizeof(zeros));
    return put(b, &fill);
}

/* A BATCH program's draft walk: its fronts' scratch zeroed first (front_scratch_fill), where the plan has one. */
MUST_CHECK static result_status batch_draft_scratch(struct builder *b)
{
	return b->made->program.kind == PROGRAM_BATCH &&
			       b->plan->cards[b->card].layers != 0 &&
			       b->plan->limits.row_capacity >=
				       MODEL_WIDE_LEAST_ROWS ?
		       front_scratch_fill(b) :
		       OK;
}

MUST_CHECK static result_status step_launches(struct builder *b, bool embeds)
{
	const struct attention_rotation_description rotation = {
		ARRAY(STEP), ARRAY(FREQUENCIES), ARRAY(ROTATION), b->rows, 0
	};
	/* the draft's rows: its copy of the embedding, or draft_layers' ranks their replicated embedding */
	const bool draft_copy =
		b->reads_draft_step &&
		b->plan->cards[b->card]
				.array_offset[V41_DATA_ARRAY_DRAFT_EMBEDDING] !=
			PROGRAM_NOWHERE;
	const struct hyper_embed_description embed = {
		ARRAY(STEP),
		draft_copy ? ARRAY(DRAFT_EMBEDDING) :
			     b->plan->globals[V41_DATA_EMBEDDING].stored,
		ARRAY(STREAMS),
		ARRAY(CARRIED_PRE),
		b->rows,
		MODEL_VOCABULARY_VALUE
	};
	result_status status = OK;
	if (!embeds && !b->reads_draft_step && b->slab_count == 1u &&
	    b->plan->cards[b->card].layers != 0 &&
	    b->plan->limits.row_capacity >= MODEL_WIDE_LEAST_ROWS) {
		status = front_scratch_fill(b);
	}
    if (embeds) {
	    status = THEN(hyper_embed_prepare(&embed, b->arena, &b->hyper,
					      slots(b, 1)));
	    if (b->vision_active && !b->reads_draft_step) {
		    status = THEN(vision_step_merge_prepare(
			    b->plan->vision.session_embeddings,
			    b->plan->vision.session_metadata, ARRAY(STEP),
			    ARRAY(STREAMS), b->rows,
			    b->plan->limits.session_capacity, b->arena,
			    &b->vision, slots(b, 1)));
	    }
        return status;
    }
    return status != OK || b->plan->cards[b->card].layers == 0 ?
		   status :
		   attention_rotation_prepare(&rotation, b->arena, &b->state,
					      slots(b, 1));
}

/* head_shards: the first row of the card's cut in its copy of the head: 0 where it holds only its rows. */
static uint32_t head_cut_first(const struct builder *b)
{
	const uint32_t shard = MODEL_VOCABULARY_VALUE / SAMPLER_SHARD_RANKS;
	return b->plan->globals[V41_DATA_HEAD].shape.output_size == shard ?
		       0u :
		       b->head_stage->description.rank * shard;
}

/* A head product: the head's rows [first, first + outputs) of the card's copy (bfloat16, no row scale) of the rows
 * (bfloat16 [b->rows][5,120]) into float32 [rows][outputs], the rows gated by the step area; by dense_stream's kernel
 * where head_stream prepared it (the same record and bits, dense_stream_records.h). */
MUST_CHECK static result_status head_product(struct builder *b, uint32_t first,
					     uint32_t outputs,
					     struct linear_range rows,
					     struct linear_range step,
					     struct linear_range output)
{
	const struct v41_tensor *const head = &b->plan->globals[V41_DATA_HEAD];
	const uint64_t skipped =
		linear_values_byte_count(first, head->shape.input_size);
	const uint64_t bytes =
		linear_values_byte_count(outputs, head->shape.input_size);
	if (first > head->shape.output_size ||
	    outputs > head->shape.output_size - first ||
	    skipped > head->stored.byte_count ||
	    bytes > head->stored.byte_count - skipped)
		return STATUS_INVALID_ARGUMENT;
	const struct dense_linear_description description = {
		{ head->stored.byte_offset + skipped, bytes },
		rows,
		output,
		step,
		outputs,
		head->shape.input_size,
		b->rows,
		BF16,
		F32,
		DENSE_STEP_ROWS,
		1.0f,
		0.0f,
		0
	};
	struct kernel_launch *const launch = slots(b, 1);
	if (b->plan->limits.official_reference != 0) {
		const struct official_head_description source = {
			description.weights, rows, output, outputs, 1
		};
		return official_head_prepare(&source, b->arena,
					     b->official.head, launch);
	}
	const result_status status =
		dense_linear_prepare(&description, b->arena, &b->dense, launch);
	if (status == OK && b->head_stream_kernel != BACKEND_NO_KERNEL) {
		const uint32_t groups = outputs / DENSE_OUTPUTS_PER_BLOCK;
		launch->kernel = b->head_stream_kernel;
		launch->grid[0] = groups < DENSE_STREAM_BLOCKS ?
					  groups :
					  DENSE_STREAM_BLOCKS;
	}
	if (status == OK && b->head_mma_kernel != BACKEND_NO_KERNEL &&
	    b->rows <= LINEAR_MAXIMUM_ROWS && b->rows >= HEAD_MMA_LEAST_ROWS &&
	    head->shape.input_size % HEAD_MMA_STEP_INPUTS == 0) {
		/* head_mma: a block a 128 outputs, 256 threads, the same record */
		const uint32_t block_outputs =
			HEAD_MMA_BLOCK_OUTPUTS / HEAD_MMA_SPLIT_8;
		launch->kernel = b->head_mma_kernel;
		launch->grid[0] =
			(outputs + block_outputs - 1u) / block_outputs;
		launch->grid[1] = launch->grid[2] = 1;
		launch->block[0] = HEAD_MMA_THREADS;
		launch->block[1] = launch->block[2] = 1;
		launch->dynamic_shared_byte_count = 0;
	}
    return status;
}

/* head_shards, the result card at the start of a step that samples: its session settings to every rank's settings
 * copy (sampler_shard_records.h), so that no sampler shard reads another card's memory. */
MUST_CHECK static result_status settings_push_launch(struct builder *b)
{
	const struct v41_tp2_stage_plan *const stage = b->head_stage;
	const uint32_t sessions = b->plan->limits.session_capacity;
	const struct sampler_settings_push_arguments arguments = {
		ARRAY(STEP).byte_offset, ARRAY(SESSIONS).byte_offset,
		stage->sampler_settings_peers.byte_offset, sessions, 0
	};
	if (stage->sampler_settings_peers.byte_count <
		    (uint64_t)SAMPLER_SHARD_RANKS * sizeof(uint64_t) ||
	    ARRAY(SESSIONS).byte_offset == PROGRAM_NOWHERE ||
	    STEP_SESSION_BYTES != SAMPLER_SETTINGS_SLOT_BYTES)
		return STATUS_INVALID_ARGUMENT;
	struct kernel_launch launch;
	prepare_launch_fill(&launch, b->settings_push_kernel, 1, 1,
			    SAMPLER_SETTINGS_PUSH_THREADS, 0, &arguments,
			    sizeof(arguments));
	return put(b, &launch);
}

/* head_shards: the card's sampler shard of a step's use `call` (sampler_shard_records.h) over the rows of step, the
 * logits [rows][V/4] at logits and the result area at result. */
MUST_CHECK static result_status
sampler_shard_launch(struct builder *b, struct linear_range step, uint32_t rows,
		     struct linear_range logits, struct linear_range result,
		     uint32_t call)
{
	const struct v41_tp2_stage_plan *const stage = b->head_stage;
	const uint32_t rank = stage->description.rank,
		       shard = MODEL_VOCABULARY_VALUE / SAMPLER_SHARD_RANKS;
	const struct sampler_shard_arguments arguments = {
		step.byte_offset,
		stage->sampler_settings.byte_offset,
		logits.byte_offset,
		result.byte_offset,
		stage->sampler_exchange.byte_offset,
		stage->sampler_peers.byte_offset,
		SAMPLER_SHARD_DEADLINE_CYCLES,
		rank * shard,
		shard,
		MODEL_VOCABULARY_VALUE,
		b->plan->limits.session_capacity,
		rank,
		call
	};
	if (rows == 0 || rows > LINEAR_MAXIMUM_ROWS ||
	    rank >= SAMPLER_SHARD_RANKS || call >= SAMPLER_SHARD_CALLS ||
	    step.byte_count < STEP_BLOCK_BYTES_VALUE +
				      (uint64_t)rows * STEP_ROW_BYTES_VALUE ||
	    logits.byte_count < (uint64_t)rows * shard * sizeof(float) ||
	    result.byte_count <
		    STEP_RESULT_BYTES +
			    (uint64_t)rows * STEP_ROW_RESULT_BYTES ||
	    stage->sampler_exchange.byte_count < SAMPLER_SHARD_EXCHANGE_BYTES)
		return STATUS_INVALID_ARGUMENT;
	struct kernel_launch launch;
	prepare_launch_fill(&launch, b->sampler_shard_kernel, 1, rows,
			    SAMPLER_THREADS, 0, &arguments, sizeof(arguments));
	return put(b, &launch);
}

/* draft_shards, on every rank: the draft head's cut over the draft's rows, then each position j's Markov bias over
 * the cut (W2's rows of the cut, the whole W1) and sampler shard (use first_call + j), into the card's own logits
 * [positions][V/4], results and Markov rows. */
MUST_CHECK static result_status
draft_cut_launches(struct builder *b, struct linear_range rows,
		   struct linear_range draft_step, struct linear_range views,
		   struct linear_range logits, struct linear_range results,
		   struct linear_range markov, uint32_t first_call)
{
    const uint32_t positions = b->plan->limits.draft_positions, rank = b->head_stage->description.rank;
    const uint32_t shard = MODEL_VOCABULARY_VALUE / SAMPLER_SHARD_RANKS;
    const struct v41_tensor *const embedding =
	    &b->plan->draft_globals[V41_DATA_DRAFT_MARKOV_EMBEDDING];
    const struct v41_tensor *const head =
	    &b->plan->draft_globals[V41_DATA_DRAFT_MARKOV_HEAD];
    const uint64_t skipped = (uint64_t)rank * shard * DRAFT_MARKOV_ROW_BYTES;
    const uint64_t cut = (uint64_t)shard * DRAFT_MARKOV_ROW_BYTES;
    if (skipped > head->stored.byte_count || cut > head->stored.byte_count - skipped)
	    return STATUS_INVALID_ARGUMENT;
    const uint32_t held = b->rows;
    b->rows = positions;
    result_status status =
	    head_product(b, head_cut_first(b), shard, rows, draft_step, logits);
    b->rows = held;
    for (uint32_t position = 0; position < positions && status == OK; position++) {
	    const struct draft_markov_bias_description bias = {
		    draft_step,
		    results,
		    embedding->stored,
		    { head->stored.byte_offset + skipped, cut },
		    logits,
		    markov,
		    position,
		    positions,
		    shard,
		    MODEL_VOCABULARY_VALUE,
		    0
	    };
	    status = draft_markov_bias_prepare(&bias, b->arena, &b->draft,
					       slots(b, 1));
	    status = THEN(sampler_shard_launch(
		    b,
		    (struct linear_range){
			    views.byte_offset +
				    (uint64_t)position * STEP_DRAFT_VIEW_BYTES,
			    STEP_DRAFT_VIEW_BYTES },
		    1u,
		    (struct linear_range){ logits.byte_offset +
						   (uint64_t)position * shard *
							   sizeof(float),
					   (uint64_t)shard * sizeof(float) },
		    (struct linear_range){
			    results.byte_offset +
				    (uint64_t)position * STEP_DRAFT_RESULT_BYTES,
			    STEP_DRAFT_RESULT_BYTES },
		    first_call + position));
    }
    return status;
}

MUST_CHECK static result_status draft_shard_rank_launches(struct builder *b,
							  uint32_t verifier,
							  uint32_t first_call);

/* batch_wide_head: the head's rows [first, first + outputs) of the card's copy over `rows` rows of hidden (bfloat16
 * [rows][5,120]) into float32 [rows][outputs] at output, one launch of dense_stream's wide form (dense_stream_records.h:
 * the bits of the thin launches); flags DENSE_STEP_ROWS gates the rows by the step area at step, 0 stores all. */
MUST_CHECK static result_status
head_wide_product(struct builder *b, uint32_t first, uint32_t outputs,
		  uint32_t rows, struct linear_range hidden, uint64_t step,
		  uint32_t flags, struct linear_range output, bool thin_split)
{
	const struct v41_tensor *const head = &b->plan->globals[V41_DATA_HEAD];
	const uint64_t row_bytes =
		(uint64_t)head->shape.input_size * sizeof(uint16_t);
	if (rows == 0 || rows > DENSE_STREAM_WIDE_ROWS ||
	    (uint64_t)first + outputs > head->shape.output_size ||
	    ((uint64_t)first + outputs) * row_bytes > head->stored.byte_count ||
	    (uint64_t)rows * MODEL_HIDDEN_ROW_BYTES > hidden.byte_count ||
	    (uint64_t)rows * outputs * sizeof(float) > output.byte_count ||
	    head->shape.input_size % DENSE_CHUNK_INPUTS != 0 ||
	    outputs % DENSE_OUTPUTS_PER_BLOCK != 0)
		return STATUS_INVALID_ARGUMENT;
	const struct dense_linear_arguments record = {
		.weights_offset = head->stored.byte_offset + first * row_bytes,
		.rows_offset = hidden.byte_offset,
		.output_offset = output.byte_offset,
		.step_offset = step,
		.output_size = outputs,
		.input_size = head->shape.input_size,
		.row_count = rows,
		.output_type = F32,
		.flags = flags,
		.output_scale = 1.0f,
		.epsilon = 0.0f,
		.cone_layer = 0
	};
	const uint32_t groups = outputs / DENSE_OUTPUTS_PER_BLOCK;
	struct kernel_launch launch;
	/* head_mma: the form head02 measured fastest for the rows (a split of N: blocks of 128 / N outputs); thin_split:
     * the 8-row form's split (each row's sums are the warps' of its split, added in warp order: the thin bits) */
	const uint32_t wide_form = thin_split  ? HEAD_MMA_WIDE_THIN_FORM :
				   rows <= 24u ? 0u :
				   rows <= 48u ? 1u :
						 2u;
	const uint32_t split = rows <= LINEAR_MAXIMUM_ROWS ? HEAD_MMA_SPLIT_8 :
							     1u << wide_form;
	if (b->head_mma_wide_kernels[0] != BACKEND_NO_KERNEL &&
	    rows <= HEAD_MMA_MOST_ROWS && rows >= HEAD_MMA_LEAST_ROWS &&
	    head->shape.input_size % HEAD_MMA_STEP_INPUTS == 0)
		prepare_launch_fill(&launch,
				    rows <= LINEAR_MAXIMUM_ROWS ?
					    b->head_mma_kernel :
					    b->head_mma_wide_kernels[wide_form],
				    (outputs + HEAD_MMA_BLOCK_OUTPUTS / split -
				     1u) / (HEAD_MMA_BLOCK_OUTPUTS / split),
				    1, HEAD_MMA_THREADS, 0, &record,
				    DENSE_ARGUMENT_BYTES);
	else
		prepare_launch_fill(&launch, b->head_wide_kernel,
				    groups < DENSE_STREAM_WIDE_BLOCKS ?
					    groups :
					    DENSE_STREAM_WIDE_BLOCKS,
				    1, DENSE_THREADS_VALUE, 0, &record,
				    DENSE_ARGUMENT_BYTES);
	return put(b, &launch);
}

/* batch_wide_head, a BATCH program's sampling card: each slab's final collapse, then the head (or the card's cut) of
 * every row in one launch of dense_stream's wide form into batch_logits [rows][outputs], then each slab's sampler (or
 * sampler shard, use = the slab) into the slab's result area. */
MUST_CHECK static result_status
batch_wide_head_launches(struct builder *b,
			 const struct v41_tp2_stage_plan *stage)
{
    const bool shards = b->head_stage != NULL;
    const uint32_t outputs =
	    shards ? MODEL_VOCABULARY_VALUE / SAMPLER_SHARD_RANKS :
		     MODEL_VOCABULARY_VALUE;
    const uint64_t sessions = b->plan->limits.session_capacity;
    result_status status = OK;
    for (uint32_t slab = 0; slab < b->slab_count && status == OK; slab++) {
        batch_slab_select(b, slab);
	const struct hyper_final_description final = {
		ARRAY(STREAMS),
		ARRAY(CARRIED_PRE),
		b->plan->globals[V41_DATA_NORM].stored,
		ARRAY(HIDDEN),
		{ 0, 0 },
		b->rows,
		0
	};
	status = hyper_final_prepare(&final, b->arena, &b->hyper, slots(b, 1));
    }
    chunk_select(b);
    const uint64_t logits_row_bytes = (uint64_t)outputs * sizeof(float);
    status = THEN(head_wide_product(
	    b, shards ? head_cut_first(b) : 0u, outputs, b->rows, ARRAY(HIDDEN),
	    ARRAY(STEP).byte_offset, DENSE_STEP_ROWS, stage->batch_logits,
	    batch_thin_bits_of(b)));
    for (uint32_t slab = 0; slab < b->slab_count && status == OK; slab++) {
        batch_slab_select(b, slab);
	const struct linear_range logits = {
		stage->batch_logits.byte_offset +
			(uint64_t)slab * LINEAR_MAXIMUM_ROWS * logits_row_bytes,
		(uint64_t)b->rows * logits_row_bytes
	};
	const struct linear_range result = {
		stage->batch_results.byte_offset +
			(uint64_t)slab * PROGRAM_BATCH_SLAB_RESULT_BYTES,
		PROGRAM_BATCH_SLAB_RESULT_BYTES
	};
	if (shards) {
            status = sampler_shard_launch(b, ARRAY(STEP), b->rows, logits, result, slab);
        } else {
		const struct sampler_description sampler = {
			ARRAY(STEP),
			{ ARRAY(SESSIONS).byte_offset,
			  sessions * STEP_SESSION_BYTES },
			logits,
			ARRAY(SAMPLER_WEIGHTS),
			result,
			MODEL_VOCABULARY_VALUE,
			(uint32_t)sessions,
			b->rows,
			0
		};
		status = sampler_prepare_entry(&sampler, b->arena, &b->sampler,
					       slots(b, 1));
	}
    }
    chunk_select(b);
    return status;
}

/* The final collapse, the head and the sampler (REF:1268-1269, 1008-1017, 1285-1292). head_shards
 * (model_v41_tp2_stage.h), on every rank: its own collapse, its rows of the head into its logits [rows][V/4], and the
 * sharded sampler into its result area, the same answers on every rank. */
MUST_CHECK static result_status head_launches(struct builder *b)
{
    const uint64_t sessions = b->plan->limits.session_capacity;
    const struct linear_range collapse =
	    b->plan->limits.official_reference != 0 ?
		    (struct linear_range){
			    b->plan->cards[b->card].official_scratch_offset,
			    MODEL_HIDDEN_ROW_BYTES
		    } :
		    no_range;
    const struct hyper_final_description final = {
	    ARRAY(STREAMS),
	    ARRAY(CARRIED_PRE),
	    b->plan->globals[V41_DATA_NORM].stored,
	    ARRAY(HIDDEN),
	    collapse,
	    b->rows,
	    0
    };
    const struct sampler_description sampler = {
	    ARRAY(STEP),
	    { ARRAY(SESSIONS).byte_offset, sessions * STEP_SESSION_BYTES },
	    ARRAY(LOGITS),
	    ARRAY(SAMPLER_WEIGHTS),
	    ARRAY(RESULT),
	    MODEL_VOCABULARY_VALUE,
	    (uint32_t)sessions,
	    b->rows,
	    0
    };
    result_status status =
	    hyper_final_prepare(&final, b->arena, &b->hyper, slots(b, 1));
    if (b->plan->limits.official_reference != 0)
	    status = THEN(official_norm(b, collapse, final.norm_weight,
					ARRAY(HIDDEN), MODEL_HIDDEN));
    if (b->head_stage == NULL) {
	    status = THEN(head_product(b, 0, MODEL_VOCABULARY_VALUE,
				       ARRAY(HIDDEN), ARRAY(STEP),
				       ARRAY(LOGITS)));
	    return THEN(sampler_prepare_entry(&sampler, b->arena, &b->sampler,
					      slots(b, 1)));
    }
    const uint32_t shard = MODEL_VOCABULARY_VALUE / SAMPLER_SHARD_RANKS;
    status = THEN(head_product(b, head_cut_first(b), shard, ARRAY(HIDDEN), ARRAY(STEP), ARRAY(LOGITS)));
    status = THEN(sampler_shard_launch(b, ARRAY(STEP), b->rows, ARRAY(LOGITS), ARRAY(RESULT), b->sampler_call));
    return b->card == b->plan->result_card ||
			   !b->head_stage->description.draft_shards ||
			   b->draft_layers ||
			   b->made->program.kind != PROGRAM_THIN ?
		   status :
		   THEN(draft_shard_rank_launches(b, 0, 1));
}

/* draft_shards, a rank other than the result card's, verifier v's draft (a thin step's: 0): in a step that holds its
 * DRAFTS row, the result card's rows of it; then its cut draft, sampler uses from first_call. */
MUST_CHECK static result_status draft_shard_rank_launches(struct builder *b,
							  uint32_t verifier,
							  uint32_t first_call)
{
	const uint32_t shard = MODEL_VOCABULARY_VALUE / SAMPLER_SHARD_RANKS;
	const uint32_t positions = b->plan->limits.draft_positions;
	const uint64_t area = b->head_stage->draft_area.byte_offset;
	const uint64_t logits = area + DRAFT_SHARE_AREA_BYTES;
	const uint64_t results =
		logits + (uint64_t)STEP_DRAFT_MOST * shard * sizeof(float);
	const uint64_t markov =
		results + (uint64_t)STEP_DRAFT_MOST * STEP_DRAFT_RESULT_BYTES;
	const struct draft_share_wait_arguments wait = {
		area, b->plan->cards[b->card].array_offset[V41_DATA_ARRAY_STEP],
		DRAFT_SHARE_DEADLINE_CYCLES, positions, verifier
	};
	if (positions == 0 || positions > STEP_DRAFT_MOST ||
	    b->head_stage->draft_area.byte_count <
		    markov - area +
			    (uint64_t)STEP_DRAFT_MOST * DRAFT_MARKOV_ROW_BYTES)
		return STATUS_INVALID_ARGUMENT;
	struct kernel_launch launch;
	prepare_launch_fill(&launch, b->draft_share_kernel, 1, 1,
			    DRAFT_SHARE_WAIT_THREADS, 0, &wait, sizeof(wait));
	const result_status status = put(b, &launch);
	return THEN(draft_cut_launches(
		b,
		(struct linear_range){ area,
				       positions * MODEL_HIDDEN_ROW_BYTES },
		(struct linear_range){ area + DRAFT_SHARE_STEP,
				       STEP_BLOCK_BYTES_VALUE +
					       positions *
						       STEP_ROW_BYTES_VALUE },
		(struct linear_range){ area + DRAFT_SHARE_VIEWS,
				       positions * STEP_DRAFT_VIEW_BYTES },
		(struct linear_range){ logits, (uint64_t)positions * shard *
						       sizeof(float) },
		(struct linear_range){ results,
				       (uint64_t)positions *
					       STEP_DRAFT_RESULT_BYTES },
		(struct linear_range){ markov, (uint64_t)positions *
						       DRAFT_MARKOV_ROW_BYTES },
		first_call));
}

/* The three stream means are taken before their layers change the streams, over the whole live chunk. */
MUST_CHECK static result_status draft_tap_launch(struct builder *b,
						 uint32_t layer)
{
	const struct hyper_draft_inputs_description description = {
		ARRAY(STREAMS),
		no_range,
		ARRAY(DRAFT_TAPS),
		b->rows,
		HYPER_STREAM_MEAN,
		layer - MODEL_DRAFT_FIRST_TAP_LAYER,
		0
	};
	return hyper_draft_inputs_prepare(&description, b->arena, &b->hyper,
					  slots(b, 1));
}

/* One piece's context products: their quantized rows have their own arrays, never the target's ROW codes. */
MUST_CHECK static result_status draft_context_piece(struct builder *b)
{
	const struct v41_tensor *const globals = b->plan->draft_globals;
	const struct v41_tensor *const first =
		b->plan->layer_tensors[MODEL_DRAFT_FIRST_LAYER];
	const struct v41_tensor *const second =
		b->plan->layer_tensors[MODEL_DRAFT_FIRST_LAYER + 1u];
	const struct v41_tensor *const last =
		b->plan->layer_tensors[MODEL_DRAFT_FIRST_LAYER +
				       MODEL_DRAFT_STAGES - 1u];
	const struct hyper_draft_inputs_description norm = {
		ARRAY(DRAFT_MAIN),
		globals[V41_DATA_DRAFT_MAIN_NORM].stored,
		ARRAY(DRAFT_MAIN_X),
		b->rows,
		HYPER_ROW_NORM,
		0,
		0
	};
	result_status status = quantize_into(b, ARRAY(DRAFT_TAPS),
					     MODEL_DRAFT_FEATURES,
					     ARRAY(DRAFT_TAP_CODES),
					     ARRAY(DRAFT_TAP_EXPONENTS),
					     ARRAY(DRAFT_TAP_OPERANDS));
	status = THEN(trunk_of(b, ARRAY(DRAFT_TAP_CODES),
			       ARRAY(DRAFT_TAP_EXPONENTS),
			       ARRAY(DRAFT_TAP_OPERANDS),
			       &globals[V41_DATA_DRAFT_MAIN_PROJECTION],
			       ARRAY(DRAFT_MAIN), NULL, no_range));
	status = THEN(hyper_draft_inputs_prepare(&norm, b->arena, &b->hyper,
						 slots(b, 1)));
	status = THEN(quantize_into(
		b, ARRAY(DRAFT_MAIN_X), MODEL_HIDDEN, ARRAY(DRAFT_TAP_CODES),
		ARRAY(DRAFT_TAP_EXPONENTS), ARRAY(DRAFT_TAP_OPERANDS)));
	status = THEN(
		trunk_of(b, ARRAY(DRAFT_TAP_CODES), ARRAY(DRAFT_TAP_EXPONENTS),
			 ARRAY(DRAFT_TAP_OPERANDS), &first[V41_DATA_WINDOW],
			 ARRAY(DRAFT_WINDOW_0), &second[V41_DATA_WINDOW],
			 ARRAY(DRAFT_WINDOW_1)));
	return THEN(trunk_of(b, ARRAY(DRAFT_TAP_CODES),
			     ARRAY(DRAFT_TAP_EXPONENTS),
			     ARRAY(DRAFT_TAP_OPERANDS), &last[V41_DATA_WINDOW],
			     ARRAY(DRAFT_WINDOW_2), NULL, no_range));
}

/* Each target row writes context at its position. A later draft overwrites every rejected or draft slot it reads.
 * The products over the chunk in pieces of piece_rows rows, or (0) over the rows selected. */
MUST_CHECK static result_status draft_context_launches(struct builder *b,
						       uint32_t piece_rows)
{
    draft_swap(b, true);
    result_status status = piece_rows == 0 ? draft_context_piece(b) : OK;
    for (uint32_t first = 0; piece_rows != 0 && first < b->chunk_rows && status == OK; first += piece_rows) {
        rows_select(b, 0, first, b->chunk_rows - first < piece_rows ? b->chunk_rows - first : piece_rows);
        status = draft_context_piece(b);
    }
    chunk_select(b);
    for (uint32_t stage = 0; stage < MODEL_DRAFT_STAGES && status == OK;
	 stage++) {
	    const uint32_t layer = MODEL_DRAFT_FIRST_LAYER + stage;
	    const struct attention_window_description description = {
		    ARRAY(STEP),
		    ARRAY(ROTATION),
		    b->arrays[V41_DATA_ARRAY_DRAFT_WINDOW_0 + stage],
		    b->plan->layer_tensors[layer][V41_DATA_WINDOW_NORM].stored,
		    { b->plan->cards[b->card].ring_offset[layer],
		      (uint64_t)b->plan->limits.session_capacity *
			      V41_DATA_RING_SESSION_BYTES },
		    b->rows,
		    b->plan->limits.session_capacity,
		    b->plan->limits.context_token_limit,
		    0,
		    0
	    };
	    status = attention_window_prepare(&description, b->arena, &b->state,
					      slots(b, 1));
    }
    draft_swap(b, false);
    return status;
}

_Static_assert(
	DRAFT_SLAB_RESULT_BYTES == PROGRAM_BATCH_SLAB_RESULT_BYTES &&
		DRAFT_BATCH_MOST_ROWS == PROGRAM_BATCH_MOST_ROWS &&
		DRAFT_SLAB_ROWS == LINEAR_MAXIMUM_ROWS &&
		PROGRAM_BATCH_MOST_ROWS / LINEAR_MAXIMUM_ROWS +
				DRAFT_SHARE_VERIFIERS * STEP_DRAFT_MOST <=
			SAMPLER_SHARD_CALLS,
	"draft_records.h's BATCH figures are model_seam.h's; a sampler use for every slab and draft position");

/* A BATCH program's verifiers (model_seam.h): a DRAFTS row and its proposals a session (or the description's
 * batch_verifiers, a step's drafting sessions apart from its rows), at most the sessions and
 * DRAFT_SHARE_VERIFIERS (8, the C8 cell). */
static uint32_t
batch_verifiers(const struct v41_tp2_program_description *description,
		const struct model_limits *limits, uint32_t rows)
{
    uint32_t verifiers = description->batch_verifiers != 0 ? description->batch_verifiers :
                                                             rows / (limits->draft_positions + 1u);
    verifiers = verifiers < limits->session_capacity ? verifiers : limits->session_capacity;
    verifiers = verifiers < DRAFT_SHARE_VERIFIERS ? verifiers :
						    DRAFT_SHARE_VERIFIERS;
    return verifiers == 0 ? 1u : verifiers;
}

/* batch_draft: verifier v's draft arrays (verifier 0 the plan's, verifier v the stage's batch draft area v - 1), the
 * slab results and the round table, behind whatever rows_select chose. */
static void batch_draft_select(struct builder *b)
{
    if (!b->batch_draft)
        return;
    const struct v41_tp2_stage_plan *const stage = b->batch_stage;
    const uint32_t positions = b->plan->limits.draft_positions;
    static const enum v41_array arrays[V41_DATA_BATCH_DRAFT_PARTS] = {
	    V41_DATA_ARRAY_DRAFT_STEP,	   V41_DATA_ARRAY_DRAFT_VIEWS,
	    V41_DATA_ARRAY_DRAFT_RESULTS,  V41_DATA_ARRAY_DRAFT_MARKOV,
	    V41_DATA_ARRAY_DRAFT_COLLAPSE, V41_DATA_ARRAY_DRAFT_LOGITS
    };
    const uint64_t area =
	    stage->batch_drafts.byte_offset +
	    (uint64_t)(b->batch_verifier - 1u) *
		    v41_batch_draft_offset(V41_DATA_BATCH_DRAFT_PARTS,
					   positions);
    for (uint32_t part = 0;
	 part < V41_DATA_BATCH_DRAFT_PARTS && b->batch_verifier != 0; part++)
	    b->arrays[arrays[part]] = (struct linear_range){
		    area + v41_batch_draft_offset(part, positions),
		    v41_batch_draft_bytes(part, positions)
	    };
    if (b->reads_draft_step)
	    b->arrays[V41_DATA_ARRAY_STEP] =
		    b->arrays[V41_DATA_ARRAY_DRAFT_STEP];
    b->arrays[V41_DATA_ARRAY_RESULT] =
	    (struct linear_range){ stage->batch_results.byte_offset,
				   (uint64_t)b->batch_slabs *
					   PROGRAM_BATCH_SLAB_RESULT_BYTES };
    b->arrays[V41_DATA_ARRAY_ROUND] = stage->batch_rounds;
}

/* The target has sampled every row. Verify selects the next anchor and writes all draft step areas before any read. */
MUST_CHECK static result_status tp2_layer_select(struct builder *b,
						 uint32_t layer);

/* diagnostics (DIAG_DRAFT_PROBE): a copy of the draft's rows of an array (rows of width elements, float32 or
 * bfloat16) into the draft logits at offset bytes, which the dump then reads (the walk leaves the logits alone) */
MUST_CHECK static result_status draft_probe(struct builder *b,
					    struct linear_range source,
					    uint32_t width, bool is_float32,
					    uint64_t offset)
{
    const uint64_t element = is_float32 ? sizeof(float) : sizeof(uint16_t);
    const uint32_t rows = b->plan->limits.draft_positions;
    const struct linear_range logits = ARRAY(DRAFT_LOGITS);
    if (offset + rows * width * element > logits.byte_count)
	    return STATUS_INVALID_ARGUMENT;
    const struct tp2_join_description copy = {
	    .source = { source.byte_offset, rows * width * element },
	    .output = { logits.byte_offset + offset, rows * width * element },
	    .width = width,
	    .row_count = rows,
	    .source_row_stride = width,
	    .output_row_stride = width,
	    .operation = is_float32 ? TP2_JOIN_FLOAT32_COPY :
				      TP2_JOIN_BFLOAT16_COPY
    };
    return tp2_join_prepare(&copy, b->arena, b->batch_join_kernel, slots(b, 1));
}

#define PROBE_QUERY_WIDTH (MODEL_HEADS_VALUE * MODEL_HEAD_DIMS)
#define PROBE_ROTATION_WIDTH (MODEL_ROTATION_ROW_BYTES / 4u)

MUST_CHECK static result_status draft_walk(struct builder *b)
{
    const uint32_t first_launch = b->made->cards[b->card].launch_count;
    const struct model_limits *const limits = &b->plan->limits;
    batch_draft_select(b);
    const uint32_t flags =
	    b->batch_draft ? DRAFT_ROUND_TABLE | DRAFT_SLAB_RESULTS : 0u;
    const struct verify_matches_description verify = {
	    ARRAY(STEP),
	    ARRAY(RESULT),
	    ARRAY(ROUND),
	    ARRAY(DRAFT_STEP),
	    ARRAY(DRAFT_VIEWS),
	    b->rows,
	    limits->draft_positions,
	    V41_DATA_END_TOKEN,
	    MODEL_VOCABULARY_VALUE,
	    b->batch_draft ? b->batch_verifier : 0u,
	    flags,
	    b->batch_draft ? limits->session_capacity : 0u,
	    0
    };
    draft_swap(b, true);
    result_status status =
	    verify_matches_prepare(&verify, b->arena, &b->draft, slots(b, 1));
    b->reads_draft_step = true;
    rows_select(b, 0, 0, limits->draft_positions);
    batch_draft_select(b);
    status = THEN(batch_draft_scratch(b));
    status = THEN(step_launches(b, false));
    status = THEN(step_launches(b, true));
    uint32_t layer_first = b->made->cards[b->card].launch_count;
    /* diagnostics (gnmac_port, the BATCH drafts): DIAG_DRAFT_STAGES walks only so many stages, and
     * DIAG_DRAFT_SKIP_FFN=1 leaves out their FFN sites, so that the collapse rows hold that point's streams */
    const char *const stages_text = getenv("DIAG_DRAFT_STAGES");
    const char *const skip_text = getenv("DIAG_DRAFT_SKIP_FFN");
    const uint32_t stages = stages_text != NULL ?
				    (uint32_t)strtoul(stages_text, NULL, 10) :
				    MODEL_DRAFT_STAGES;
    const bool skip_ffn = skip_text != NULL && strcmp(skip_text, "1") == 0;
    /* DIAG_DRAFT_PROBE: stage 0's attention in its parts, the rotations, wq_b's heads and the queries behind the
     * queries launch, then the core's output and the rotations again, into the logits (offsets 0, 790,528, 4,096,
     * 393,216, 786,432); the head and what follows it are left out */
    const bool probe = getenv("DIAG_DRAFT_PROBE") != NULL &&
		       b->batch_join_kernel != BACKEND_NO_KERNEL;
    for (uint32_t stage = 0;
	 stage < MODEL_DRAFT_STAGES && stage < stages && status == OK;
	 stage++) {
	    const uint32_t layer = MODEL_DRAFT_FIRST_LAYER + stage;
	    status = b->draft_layers ?
			     tp2_layer_select(b, layer) :
			     status; /* the card's rank of the stage */
	    if (probe && stage == 0) {
		    status = THEN(layer_steps(b, layer, 0, 0, FRONT_STEPS));
		    /* the site front's: the streams it read, its post and comb (offsets 1,257,472, 1,255,424, 1,255,936) */
		    status = THEN(draft_probe(b, ARRAY(STREAMS),
					      MODEL_STREAMS_ROW_BYTES / 2u,
					      false, 1257472u));
		    status = THEN(draft_probe(b, ARRAY(POST),
					      MODEL_SITE_ROW_BYTES / 4u, true,
					      1255424u));
		    status = THEN(draft_probe(b, ARRAY(COMB),
					      MODEL_COMB_ROW_BYTES / 4u, true,
					      1255936u));
		    status = THEN(layer_steps(
			    b, layer, 0, 0,
			    QUERY_PRODUCTS | INDEX_WEIGHTS_PRODUCT | QUERIES));
		    status = THEN(draft_probe(b, ARRAY(ROTATION),
					      PROBE_ROTATION_WIDTH, true, 0));
		    status =
			    THEN(draft_probe(b, ARRAY(HEADS), PROBE_QUERY_WIDTH,
					     false, 790528u));
		    status = THEN(draft_probe(b, ARRAY(QUERIES),
					      PROBE_QUERY_WIDTH, false, 4096u));
		    status = THEN(layer_steps(b, layer, 0, 0, CORE));
		    status = THEN(draft_probe(b, ARRAY(ATTENTION_OUTPUT),
					      PROBE_QUERY_WIDTH, false,
					      393216u));
		    status = THEN(draft_probe(b, ARRAY(ROTATION),
					      PROBE_ROTATION_WIDTH, true,
					      786432u));
		    status = THEN(layer_steps(b, layer, 0, 0,
					      OUTPUT_PRODUCTS | SITE_POST));
		    /* wo_a's groups, the sublayer output and the streams behind the post (1,122,304, 1,204,224, 1,462,272) */
		    status = THEN(draft_probe(b, ARRAY(GROUPS),
					      V41_DATA_WIDEST_ROW, false,
					      1122304u));
		    status = THEN(draft_probe(b, ARRAY(SUBLAYER_OUTPUT),
					      MODEL_HIDDEN, false, 1204224u));
		    status = THEN(draft_probe(b, ARRAY(STREAMS),
					      MODEL_STREAMS_ROW_BYTES / 2u,
					      false, 1462272u));
	    } else {
		    status = THEN(layer_steps(b, layer, 0, 0,
					      FRONT_STEPS | REST_STEPS));
	    }
	    status = skip_ffn ? status :
				THEN(ffn_launches(b, layer, &layer_first));
    }
    b->tensor_layer = NULL;
    const struct hyper_final_description final = {
	    ARRAY(STREAMS),
	    ARRAY(CARRIED_PRE),
	    b->plan->draft_globals[V41_DATA_DRAFT_NORM].stored,
	    ARRAY(HIDDEN),
	    ARRAY(DRAFT_COLLAPSE),
	    b->rows,
	    0
    };
    status =
	    THEN(hyper_final_prepare(&final, b->arena, &b->hyper, slots(b, 1)));
    const bool shares = b->head_stage != NULL && b->head_stage->description.draft_shards;
    if (probe) { /* diagnostics: no head, Markov bias, sampler or confidence (the logits hold the probes) */
    } else if (b->draft_layers) { /* every rank holds the draft's rows: its own cut of the draft */
        status = THEN(draft_cut_launches(b, ARRAY(HIDDEN), ARRAY(STEP), ARRAY(DRAFT_VIEWS), ARRAY(DRAFT_LOGITS),
                                         ARRAY(DRAFT_RESULTS), ARRAY(DRAFT_MARKOV), 1u));
    } else if (shares) { /* draft_shards: the rows to the other ranks, then the card's own cut of the draft */
	    const struct draft_share_arguments share = {
		    ARRAY(HIDDEN).byte_offset,
		    ARRAY(DRAFT_STEP).byte_offset,
		    ARRAY(DRAFT_VIEWS).byte_offset,
		    b->head_stage->draft_peers.byte_offset,
		    b->head_stage->draft_counters.byte_offset,
		    b->plan->cards[b->card].array_offset[V41_DATA_ARRAY_STEP],
		    limits->draft_positions,
		    b->head_stage->description.rank,
		    b->batch_draft ? b->batch_verifier : 0u,
		    0
	    };
	    struct kernel_launch launch;
	    prepare_launch_fill(&launch, b->draft_share_kernel,
				DRAFT_SHARE_RANKS * DRAFT_SHARE_BLOCKS_PER_RANK,
				1, DRAFT_SHARE_THREADS, 0, &share,
				sizeof(share));
	    status = THEN(put(b, &launch));
	    status = THEN(draft_cut_launches(
		    b, ARRAY(HIDDEN), ARRAY(STEP), ARRAY(DRAFT_VIEWS),
		    ARRAY(DRAFT_LOGITS), ARRAY(DRAFT_RESULTS),
		    ARRAY(DRAFT_MARKOV),
		    b->batch_draft ?
			    b->batch_slabs + b->batch_verifier *
						     limits->draft_positions :
			    1u));
    } else {
	    status = THEN(head_product(b, 0, MODEL_VOCABULARY_VALUE,
				       ARRAY(HIDDEN), ARRAY(STEP),
				       ARRAY(DRAFT_LOGITS)));
    }
    for (uint32_t position = 0; position < limits->draft_positions && status == OK && !shares && !probe; position++) {
	    const struct draft_markov_bias_description markov = {
		    ARRAY(DRAFT_STEP),
		    ARRAY(DRAFT_RESULTS),
		    b->plan->draft_globals[V41_DATA_DRAFT_MARKOV_EMBEDDING]
			    .stored,
		    b->plan->draft_globals[V41_DATA_DRAFT_MARKOV_HEAD].stored,
		    ARRAY(DRAFT_LOGITS),
		    ARRAY(DRAFT_MARKOV),
		    position,
		    limits->draft_positions,
		    MODEL_VOCABULARY_VALUE,
		    0,
		    0
	    };
	    const struct sampler_description sampler = {
		    { ARRAY(DRAFT_VIEWS).byte_offset +
			      (uint64_t)position * STEP_DRAFT_VIEW_BYTES,
		      STEP_DRAFT_VIEW_BYTES },
		    { ARRAY(SESSIONS).byte_offset,
		      (uint64_t)limits->session_capacity * STEP_SESSION_BYTES },
		    { ARRAY(DRAFT_LOGITS).byte_offset +
			      (uint64_t)position * MODEL_LOGITS_ROW_BYTES,
		      MODEL_LOGITS_ROW_BYTES },
		    { ARRAY(SAMPLER_WEIGHTS).byte_offset,
		      MODEL_LOGITS_ROW_BYTES },
		    { ARRAY(DRAFT_RESULTS).byte_offset +
			      (uint64_t)position * STEP_DRAFT_RESULT_BYTES,
		      STEP_DRAFT_RESULT_BYTES },
		    MODEL_VOCABULARY_VALUE,
		    limits->session_capacity,
		    1u,
		    0
	    };
	    status = draft_markov_bias_prepare(&markov, b->arena, &b->draft,
					       slots(b, 1));
	    status = THEN(sampler_prepare_entry(&sampler, b->arena, &b->sampler,
						slots(b, 1)));
    }
    const struct draft_confidence_description confidence = {
	    ARRAY(DRAFT_STEP),
	    ARRAY(DRAFT_COLLAPSE),
	    ARRAY(DRAFT_MARKOV),
	    b->plan->draft_globals[V41_DATA_DRAFT_CONFIDENCE].stored,
	    ARRAY(DRAFT_RESULTS),
	    ARRAY(ROUND),
	    limits->draft_positions,
	    b->batch_draft ? DRAFT_ROUND_TABLE : 0u,
	    0
    };
    if (b->card == b->plan->result_card && !probe) /* draft_layers: the other ranks write no round */
	    status = THEN(draft_confidence_prepare(&confidence, b->arena,
						   &b->draft, slots(b, 1)));
    b->reads_draft_step = false;
    chunk_select(b);
    draft_swap(b, false);
    if (!b->draft_tail_begun) /* a BATCH program's verifiers' walks, draft_layers' ranks: one draft tail */
        b->made->program.draft_segment_first = b->made->program.segment_count;
    b->draft_tail_begun = true;
    segment_add(b, SEGMENT_NO_LAYER, first_launch);
    return status;
}

/* batch_draft_rows, a BATCH program's draft on the result card (draft_records.h's BATCHED): one verify launch a
 * verifier into one draft step of slab blocks; the rotation, embedding, three layers, collapse and head of every
 * verifier's rows slab by slab (layers outermost: a slab's attention behind the earlier slabs' of its layer, as a
 * FULL program's); then each verifier's positions (Markov bias and sampler) and its confidence into the round
 * table. Run as a thin program's (slab_count 1, the card selected so). */
MUST_CHECK static result_status batch_draft_rows_walk(struct builder *b,
						      uint32_t verifiers)
{
    const uint32_t first_launch = b->made->cards[b->card].launch_count;
    const struct model_limits *const limits = &b->plan->limits;
    const struct v41_tp2_stage_plan *const stage = b->batch_stage;
    const uint32_t positions = limits->draft_positions, rows = verifiers * positions;
    const uint32_t slabs =
	    (rows + LINEAR_MAXIMUM_ROWS - 1u) / LINEAR_MAXIMUM_ROWS;
    if (rows > V41_DATA_BATCH_DRAFT_ROWS || verifiers > DRAFT_BATCH_SLOTS ||
	stage->batch_rows.byte_count <
		v41_batch_rows_offset(V41_DATA_BATCH_ROWS_PARTS))
	    return STATUS_INVALID_ARGUMENT;
    struct linear_range part[V41_DATA_BATCH_DRAFT_PARTS];
    for (uint32_t index = 0; index < V41_DATA_BATCH_DRAFT_PARTS; index++)
	    part[index] =
		    (struct linear_range){ stage->batch_rows.byte_offset +
						   v41_batch_rows_offset(index),
					   v41_batch_rows_bytes(index) };
    const uint64_t slab_step_bytes = DRAFT_SLAB_STEP_BYTES;
    const struct linear_range results = {
	    stage->batch_results.byte_offset,
	    (uint64_t)b->batch_slabs * PROGRAM_BATCH_SLAB_RESULT_BYTES
    };
    /* batch_draft_wide 3 over several slabs: the step block of all rows too (draft_wide_layer's window and core) */
    const bool wide_step = b->batch_draft_wide >= DRAFT_WIDE_CORES && slabs > 1u && b->draft_layers;
    draft_swap(b, true);
    chunk_select(b);
    result_status status = OK;
    for (uint32_t verifier = 0; verifier < verifiers && status == OK; verifier++) {
	    const struct verify_matches_description verify = {
		    ARRAY(STEP),
		    results,
		    stage->batch_rounds,
		    part[V41_DATA_BATCH_DRAFT_STEP],
		    part[V41_DATA_BATCH_DRAFT_VIEWS],
		    b->rows,
		    positions,
		    V41_DATA_END_TOKEN,
		    MODEL_VOCABULARY_VALUE,
		    verifier,
		    DRAFT_ROUND_TABLE | DRAFT_SLAB_RESULTS | DRAFT_BATCHED |
			    (wide_step ? DRAFT_WIDE_STEP : 0u),
		    limits->session_capacity,
		    slabs
	    };
	    status = verify_matches_prepare(&verify, b->arena, &b->draft,
					    slots(b, 1));
    }
    b->reads_draft_step = true;
    /* slab s: chunk rows 8 s .., its slab block as the step area, its rows of the batch area's parts */
    struct linear_range slab_step = { 0, 0 };
    const uint64_t logits_row_bytes = MODEL_LOGITS_ROW_BYTES;
#define BATCH_DRAFT_SLAB(s)                                                    \
    do {                                                                       \
	    rows_select(b, 0, (s) * LINEAR_MAXIMUM_ROWS, LINEAR_MAXIMUM_ROWS); \
	    slab_step = (struct linear_range){                                 \
		    part[V41_DATA_BATCH_DRAFT_STEP].byte_offset +              \
			    (s) * slab_step_bytes,                             \
		    slab_step_bytes                                            \
	    };                                                                 \
	    b->arrays[V41_DATA_ARRAY_STEP] = slab_step;                        \
	    b->arrays[V41_DATA_ARRAY_DRAFT_STEP] = slab_step;                  \
	    b->arrays[V41_DATA_ARRAY_DRAFT_COLLAPSE] = (struct linear_range){  \
		    part[V41_DATA_BATCH_DRAFT_COLLAPSE].byte_offset +          \
			    (uint64_t)(s) * LINEAR_MAXIMUM_ROWS *              \
				    MODEL_HIDDEN_ROW_BYTES,                    \
		    (uint64_t)LINEAR_MAXIMUM_ROWS * MODEL_HIDDEN_ROW_BYTES     \
	    };                                                                 \
    } while (0)
    status = THEN(batch_draft_scratch(b));
    for (uint32_t slab = 0; slab < slabs && status == OK; slab++) {
        BATCH_DRAFT_SLAB(slab);
        status = step_launches(b, false);
        status = THEN(step_launches(b, true));
    }
    uint32_t layer_first = b->made->cards[b->card].launch_count;
    b->draft_slab_steps = part[V41_DATA_BATCH_DRAFT_STEP].byte_offset;
    b->draft_slab_collapse = part[V41_DATA_BATCH_DRAFT_COLLAPSE].byte_offset;
    b->draft_slab_count = slabs;
    b->draft_rows = rows;
    b->draft_wide_step = wide_step;
    for (uint32_t stage_index = 0;
	 stage_index < MODEL_DRAFT_STAGES && status == OK; stage_index++) {
	    const uint32_t layer = MODEL_DRAFT_FIRST_LAYER + stage_index;
	    /* batch_draft_wide: each site's TP sum and post once over every slab's rows (one BF16 ring), behind the
         * slabs' sublayers, where the rows fit one ring; else a slab at a time. 2: the layer's products over all its
         * rows too (draft_wide_layer) */
	    bool once = b->batch_draft_wide != 0 && slabs > 1u &&
			b->draft_layers;
	    if (once) {
		    status = tp2_layer_select(b, layer);
		    once = status == OK && b->tensor_layer->bf16_ring &&
			   !b->tensor_layer->exact_dense &&
			   !b->tensor_layer->exact_output_shards &&
			   rows <= b->tensor_layer->ring_rows;
	    }
	    if (once && b->batch_draft_wide >= DRAFT_WIDE_PRODUCTS) {
		    status = draft_wide_layer(b, layer);
		    continue;
	    }
	    for (uint32_t slab = 0; slab < slabs && status == OK; slab++) {
		    BATCH_DRAFT_SLAB(slab);
		    status = b->draft_layers ?
				     tp2_layer_select(b, layer) :
				     status; /* the slab's TP sums: its ring */
		    status =
			    THEN(layer_steps(b, layer, 0, 0,
					     (FRONT_STEPS | REST_STEPS) &
						     ~(once ? SITE_POST : 0u)));
		    b->ffn_post_deferred = once;
		    status = once ? status :
				    THEN(ffn_launches(b, layer, &layer_first));
	    }
	    if (once && status == OK) {
		    BATCH_DRAFT_SLAB(0);
		    rows_select(b, 0, 0, rows);
		    status = layer_steps(b, layer, 0, 0, SITE_POST);
		    for (uint32_t slab = 0; slab < slabs && status == OK;
			 slab++) {
			    BATCH_DRAFT_SLAB(slab);
			    status = ffn_launches(b, layer, &layer_first);
		    }
		    BATCH_DRAFT_SLAB(0);
		    rows_select(b, 0, 0, rows);
		    status = THEN(ffn_post(b, layer));
	    }
	    b->ffn_post_deferred = false;
    }
    b->tensor_layer = NULL;
    /* each slab's final (the rows verifier-major, the collapse rows the confidences'); then the rows position-major,
     * slot v of position j verifier v's row j (draft_records.h's BATCHED): the head of each position's 8 slots gated
     * by its view, or with batch_wide_head one wide head over them all (the head read once); a position at a time its
     * Markov bias over every slot and one sampler over its view; then each verifier's confidence */
    for (uint32_t slab = 0; slab < slabs && status == OK; slab++) {
        BATCH_DRAFT_SLAB(slab);
	const struct hyper_final_description final = {
		ARRAY(STREAMS),
		ARRAY(CARRIED_PRE),
		b->plan->draft_globals[V41_DATA_DRAFT_NORM].stored,
		ARRAY(HIDDEN),
		ARRAY(DRAFT_COLLAPSE),
		b->rows,
		0
	};
	status = hyper_final_prepare(&final, b->arena, &b->hyper, slots(b, 1));
    }
#undef BATCH_DRAFT_SLAB
    const uint64_t hidden_row_bytes = MODEL_HIDDEN_ROW_BYTES,
		   view_bytes = DRAFT_SLAB_STEP_BYTES;
    const uint32_t slot_count = DRAFT_BATCH_SLOTS;
    const struct linear_range hidden_slots = {
	    stage->batch_rows.byte_offset +
		    v41_batch_rows_offset(V41_DATA_BATCH_ROWS_HIDDEN),
	    v41_batch_rows_bytes(V41_DATA_BATCH_ROWS_HIDDEN)
    };
    rows_select(b, 0, 0, rows);
    const struct linear_range hidden = ARRAY(HIDDEN);
    /* batch_wide_head with fewer verifiers than slots: the draft's final rows (verifier-major) are the head's rows
     * alone, then each position's logits row to its slot (position j of verifier v: row 8 j + v); the slots past the
     * verifiers hold no live row in either form (batch_draft_slots_head: the slots' head of positions x 8 rows, as
     * before). One verifier (BATCH-8 of one: C1): the head's rows are rows 0 .. positions - 1 of the logits, moved in
     * place; more: the logits rows behind the slots' hold them where they fit (the area's rows of 56 or of a cut) */
    const bool own_rows_head = verifiers < DRAFT_BATCH_SLOTS &&
			       stage->description.batch_wide_head &&
			       !stage->description.batch_draft_slots_head;
    for (uint32_t position = 0; position < positions && status == OK && !own_rows_head; position++) {
	    const struct tp2_join_description copy = {
		    .source = { hidden.byte_offset +
					position * hidden_row_bytes,
				((uint64_t)(verifiers - 1u) * positions + 1u) *
					hidden_row_bytes },
		    .output = { hidden_slots.byte_offset +
					(uint64_t)position * slot_count *
						hidden_row_bytes,
				(uint64_t)verifiers * hidden_row_bytes },
		    .width = MODEL_HIDDEN,
		    .row_count = verifiers,
		    .source_row_stride = positions * MODEL_HIDDEN,
		    .output_row_stride = MODEL_HIDDEN,
		    .operation = TP2_JOIN_BFLOAT16_COPY
	    };
	    status = tp2_join_prepare(&copy, b->arena, b->batch_join_kernel,
				      slots(b, 1));
    }
    const struct linear_range views = part[V41_DATA_BATCH_DRAFT_VIEWS];
    const struct linear_range logits = part[V41_DATA_BATCH_DRAFT_LOGITS];
    const struct linear_range view0 = { views.byte_offset, view_bytes };
    const uint64_t slots_logits_bytes = (uint64_t)slot_count * logits_row_bytes;
    rows_select(b, 0, 0, slot_count);
    /* draft_layers: every rank its cut of the vocabulary (its rows of the head, its Markov bias, the sampler shard) */
    const uint32_t shard = MODEL_VOCABULARY_VALUE / SAMPLER_SHARD_RANKS;
    const uint32_t vocabulary = b->draft_layers ? shard :
						  MODEL_VOCABULARY_VALUE;
    /* the moves in pieces of a shard's columns (a join's widest rows): one piece for a cut, four for the whole */
    const uint64_t row_bytes = (uint64_t)vocabulary * sizeof(float), piece_bytes = (uint64_t)shard * sizeof(float);
    const uint64_t behind = (uint64_t)positions * slot_count * row_bytes; /* the rows behind the slots' */
    const bool behind_head = own_rows_head && verifiers > 1u && behind + rows * row_bytes <= logits.byte_count;
    if (behind_head) {
	    status = THEN(head_wide_product(
		    b, b->draft_layers ? head_cut_first(b) : 0u, vocabulary,
		    rows, hidden, 0, 0,
		    (struct linear_range){ logits.byte_offset + behind,
					   rows * row_bytes },
		    false));
	    for (uint32_t piece = 0;
		 piece < vocabulary / shard * verifiers && status == OK;
		 piece++) {
		    const uint32_t verifier = piece / (vocabulary / shard),
				   column = piece % (vocabulary / shard);
		    const struct tp2_join_description slot_rows = {
			    .source = { logits.byte_offset + behind +
						(uint64_t)verifier * positions *
							row_bytes +
						column * piece_bytes,
					(uint64_t)(positions - 1u) * row_bytes +
						piece_bytes },
			    .output = { logits.byte_offset +
						verifier * row_bytes +
						column * piece_bytes,
					(uint64_t)(positions - 1u) *
							slot_count * row_bytes +
						piece_bytes },
			    .width = shard,
			    .row_count = positions,
			    .source_row_stride = vocabulary,
			    .output_row_stride = slot_count * vocabulary,
			    .operation = TP2_JOIN_FLOAT32_COPY
		    };
		    status = tp2_join_prepare(&slot_rows, b->arena,
					      b->batch_join_kernel,
					      slots(b, 1));
	    }
    } else if (own_rows_head && verifiers == 1u) {
        const uint32_t cut_first = b->draft_layers ? head_cut_first(b) : 0u;
        status = THEN(head_wide_product(b, cut_first, vocabulary, positions, hidden, 0, 0, logits, false));
        for (uint32_t piece = 0; piece < vocabulary / shard && positions >= 2u && status == OK; piece++) {
		const struct tp2_join_description slot_rows = {
			.source = { logits.byte_offset + row_bytes +
					    piece * piece_bytes,
				    (uint64_t)(positions - 2u) * row_bytes +
					    piece_bytes },
			.output = { logits.byte_offset +
					    slot_count * row_bytes +
					    piece * piece_bytes,
				    (uint64_t)(positions - 2u) * slot_count *
						    row_bytes +
					    piece_bytes },
			.width = shard,
			.row_count = positions - 1u,
			.source_row_stride = vocabulary,
			.output_row_stride = slot_count * vocabulary,
			.operation = TP2_JOIN_FLOAT32_COPY
		};
		status = tp2_join_prepare(&slot_rows, b->arena,
					  b->batch_join_kernel, slots(b, 1));
	}
    } else if (stage->description.batch_wide_head) {
        for (uint32_t position = 0; position < positions && status == OK && own_rows_head; position++) {
		const struct tp2_join_description copy = {
			/* the slots' head after all: the joins it reads */
			.source = { hidden.byte_offset +
					    position * hidden_row_bytes,
				    ((uint64_t)(verifiers - 1u) * positions +
				     1u) * hidden_row_bytes },
			.output = { hidden_slots.byte_offset +
					    (uint64_t)position * slot_count *
						    hidden_row_bytes,
				    (uint64_t)verifiers * hidden_row_bytes },
			.width = MODEL_HIDDEN,
			.row_count = verifiers,
			.source_row_stride = positions * MODEL_HIDDEN,
			.output_row_stride = MODEL_HIDDEN,
			.operation = TP2_JOIN_BFLOAT16_COPY
		};
		status = tp2_join_prepare(&copy, b->arena, b->batch_join_kernel,
					  slots(b, 1));
	}
        status = THEN(head_wide_product(b, b->draft_layers ? head_cut_first(b) : 0u, vocabulary, positions * slot_count,
                                        hidden_slots, 0, 0, logits, false));
    }
    if (b->draft_layers && !stage->description.batch_wide_head)
	    return STATUS_INVALID_ARGUMENT;
    for (uint32_t position = 0; position < positions && status == OK && !stage->description.batch_wide_head;
         position++)
	    status = head_product(
		    b, 0, MODEL_VOCABULARY_VALUE,
		    (struct linear_range){
			    hidden_slots.byte_offset + (uint64_t)position *
							       slot_count *
							       hidden_row_bytes,
			    (uint64_t)slot_count * hidden_row_bytes },
		    (struct linear_range){ views.byte_offset +
						   position * view_bytes,
					   view_bytes },
		    (struct linear_range){
			    logits.byte_offset + position * slots_logits_bytes,
			    slots_logits_bytes });
    const struct linear_range markov_head =
	    b->plan->draft_globals[V41_DATA_DRAFT_MARKOV_HEAD].stored;
    const uint64_t skipped = b->draft_layers ?
				     (uint64_t)b->head_stage->description.rank *
					     shard * DRAFT_MARKOV_ROW_BYTES :
				     0u;
    const uint64_t cut = (uint64_t)vocabulary * DRAFT_MARKOV_ROW_BYTES;
    if (skipped > markov_head.byte_count || cut > markov_head.byte_count - skipped)
	    return STATUS_INVALID_ARGUMENT;
    for (uint32_t position = 0; position < positions && status == OK && b->draft_layers; position++) {
        const uint64_t cut_logits_bytes = (uint64_t)slot_count * vocabulary * sizeof(float);
	const struct draft_markov_bias_description markov = {
		view0,
		part[V41_DATA_BATCH_DRAFT_RESULTS],
		b->plan->draft_globals[V41_DATA_DRAFT_MARKOV_EMBEDDING].stored,
		{ markov_head.byte_offset + skipped, cut },
		logits,
		part[V41_DATA_BATCH_DRAFT_MARKOV],
		position,
		positions,
		vocabulary,
		MODEL_VOCABULARY_VALUE,
		DRAFT_BATCHED
	};
	status = draft_markov_bias_prepare(&markov, b->arena, &b->draft,
					   slots(b, 1));
	status = THEN(sampler_shard_launch(
		b,
		(struct linear_range){
			views.byte_offset + position * view_bytes, view_bytes },
		slot_count,
		(struct linear_range){ logits.byte_offset +
					       position * cut_logits_bytes,
				       cut_logits_bytes },
		(struct linear_range){
			part[V41_DATA_BATCH_DRAFT_RESULTS].byte_offset +
				(uint64_t)position * DRAFT_SLAB_RESULT_BYTES,
			DRAFT_SLAB_RESULT_BYTES },
		b->batch_slabs + position));
    }
    for (uint32_t position = 0; position < positions && status == OK && !b->draft_layers; position++) {
	    const struct draft_markov_bias_description markov = {
		    view0,
		    part[V41_DATA_BATCH_DRAFT_RESULTS],
		    b->plan->draft_globals[V41_DATA_DRAFT_MARKOV_EMBEDDING]
			    .stored,
		    b->plan->draft_globals[V41_DATA_DRAFT_MARKOV_HEAD].stored,
		    logits,
		    part[V41_DATA_BATCH_DRAFT_MARKOV],
		    position,
		    positions,
		    MODEL_VOCABULARY_VALUE,
		    0,
		    DRAFT_BATCHED
	    };
	    const struct sampler_description sampler = {
		    { views.byte_offset + position * view_bytes, view_bytes },
		    { ARRAY(SESSIONS).byte_offset,
		      (uint64_t)limits->session_capacity * STEP_SESSION_BYTES },
		    { logits.byte_offset + position * slots_logits_bytes,
		      slots_logits_bytes },
		    { ARRAY(SAMPLER_WEIGHTS).byte_offset, slots_logits_bytes },
		    { part[V41_DATA_BATCH_DRAFT_RESULTS].byte_offset +
			      (uint64_t)position * DRAFT_SLAB_RESULT_BYTES,
		      DRAFT_SLAB_RESULT_BYTES },
		    MODEL_VOCABULARY_VALUE,
		    limits->session_capacity,
		    slot_count,
		    0
	    };
	    status = draft_markov_bias_prepare(&markov, b->arena, &b->draft,
					       slots(b, 1));
	    status = THEN(sampler_prepare_entry(&sampler, b->arena, &b->sampler,
						slots(b, 1)));
    }
    for (uint32_t verifier = 0; verifier < verifiers && status == OK && b->card == b->plan->result_card; verifier++) {
	    const struct draft_confidence_description confidence = {
		    view0,
		    { part[V41_DATA_BATCH_DRAFT_COLLAPSE].byte_offset +
			      (uint64_t)verifier * positions * hidden_row_bytes,
		      (uint64_t)positions * hidden_row_bytes },
		    part[V41_DATA_BATCH_DRAFT_MARKOV],
		    b->plan->draft_globals[V41_DATA_DRAFT_CONFIDENCE].stored,
		    part[V41_DATA_BATCH_DRAFT_RESULTS],
		    stage->batch_rounds,
		    positions,
		    DRAFT_ROUND_TABLE | DRAFT_BATCHED,
		    verifier
	    };
	    status = draft_confidence_prepare(&confidence, b->arena, &b->draft,
					      slots(b, 1));
    }
    b->reads_draft_step = false;
    chunk_select(b);
    draft_swap(b, false);
    if (!b->draft_tail_begun) /* draft_layers: one draft tail, a segment a rank */
        b->made->program.draft_segment_first = b->made->program.segment_count;
    b->draft_tail_begun = true;
    segment_add(b, SEGMENT_NO_LAYER, first_launch);
    return status;
}

/* Where the walk moves on to another card: the streams and the carried pre go with it (2.8), and the stage changes. */
static void stage_hand_off(struct builder *b, uint32_t card)
{
    const uint64_t *const from = b->plan->cards[b->card].array_offset, *const to = b->plan->cards[card].array_offset;
    hand_off(b, card, from[V41_DATA_ARRAY_STREAMS], to[V41_DATA_ARRAY_STREAMS],
	     MODEL_STREAMS_ROW_BYTES, 0, 0);
    hand_off(b, card, from[V41_DATA_ARRAY_CARRIED_PRE],
	     to[V41_DATA_ARRAY_CARRIED_PRE], MODEL_SITE_ROW_BYTES, 0, 0);
    b->stage++;
}

/* Whether the walk's unit on card begins a visit (model_seam.h) that holds a layer: the embedding where layer 0 goes on
 * on its card, or a layer on another card than the unit before. */
static bool visit_begins_layers(const struct builder *b, uint32_t unit, uint32_t card)
{
	return unit == 0 ? b->plan->layer_card[0] == card :
			   unit <= MODEL_LAYERS && card != b->card;
}

/* The walk: the embedding (unit 0), layer l (unit l + 1), the head (the last unit). Where the card changes, the stage
 * does and the card is selected; a visit that holds a layer begins with its step segment. */
MUST_CHECK static result_status program_walk(struct builder *b)
{
	const struct model_plan_record *const plan = b->plan;
	uint32_t source = 0, group = 0;
	result_status status = OK;
	for (uint32_t unit = 0; unit <= MODEL_LAYERS + 1u && status == OK;
	     unit++) {
		const uint32_t layer = unit - 1u; /* the head's: MODEL_LAYERS */
		const uint32_t card = unit == 0 ? plan->embedding_card :
				      layer < MODEL_LAYERS ?
						  plan->layer_card[layer] :
						  plan->result_card;
		const bool begins_layers = visit_begins_layers(b, unit, card);
		if (unit != 0 && card != b->card) {
			stage_hand_off(b, card);
		}
		status = unit != 0 && card == b->card ? OK :
							card_select(b, card);
		uint32_t first_launch =
			status == OK ? b->made->cards[b->card].launch_count : 0;
		if (status == OK &&
		    begins_layers) { /* the visit's step segment */
			status = step_launches(b, false);
			segment_add(b, SEGMENT_NO_LAYER, first_launch);
			first_launch = b->made->cards[b->card].launch_count;
		}
		if (status == OK && layer < MODEL_LAYERS) {
			source = (V41_DATA_SOURCE_LAYERS >> layer & 1u) != 0 ?
					 layer :
					 source;
			group = (V41_DATA_INDEX_LAYERS >> layer & 1u) != 0 ?
					layer :
					group;
			const bool is_cut = layer == MODEL_DECODER_FIRST_LAYER;
			status =
				plan->limits.draft_positions != 0 &&
						layer >=
							MODEL_DRAFT_FIRST_TAP_LAYER ?
					draft_tap_launch(b, layer) :
					status;
			status = THEN(
				layer_steps(b, layer, source, group,
					    ENGRAM_STEPS | FRONT_STEPS |
						    (is_cut ? 0 : REST_STEPS)));
			if (status == OK &&
			    is_cut) { /* the front ends: its copies follow */
				front_end(b, layer, &first_launch);
				status = layer_steps(b, layer, source, group,
						     REST_STEPS);
			}
			status = THEN(ffn_launches(b, layer, &first_launch));
			segment_add(b, layer, first_launch);
			copies_hand_off(b, layer,
					layer == MODEL_DECODER_FIRST_LAYER ?
						LATER_COPIES :
						EVERY_COPY);
			if (status == OK && plan->limits.draft_positions != 0 &&
			    layer + 1u == MODEL_LAYERS) {
				first_launch =
					b->made->cards[b->card].launch_count;
				status = draft_context_launches(b, 0);
				segment_add(b, SEGMENT_NO_LAYER, first_launch);
			}
		} else if (status ==
			   OK) { /* the embedding (unit 0), the head */
			status = unit == 0 ? step_launches(b, true) :
					     head_launches(b);
			segment_add(b, SEGMENT_NO_LAYER, first_launch);
		}
	}
    return plan->limits.draft_positions != 0 ? THEN(draft_walk(b)) : status;
}

/* The steps of layer (enum layer_step) over each slab of the chunk in turn, through its view, the queries (QUERIES)
 * over each span ahead of its first slab; with is_pieces over each piece of at most LINEAR_WIDE_MOST_ROWS rows
 * instead: the launches over the chunk. */
MUST_CHECK static result_status steps_over(struct builder *b, uint32_t layer,
					   uint32_t source, uint32_t group,
					   uint32_t steps, bool is_pieces)
{
	const uint32_t span_slabs = V41_DATA_SPAN_ROWS / LINEAR_MAXIMUM_ROWS;
	result_status status = OK;
	for (uint32_t part = 0;
	     part < (is_pieces ? piece_count_of(b) : b->slab_count) &&
	     status == OK;
	     part++) {
		if (!is_pieces && (steps & QUERIES) != 0 &&
		    part % span_slabs == 0) {
			span_select(b, part / span_slabs);
			status = layer_steps(b, layer, source, group, QUERIES);
		}
		if (is_pieces) {
			piece_select(b, part);
		} else {
			slab_select(b, part);
		}
		status = THEN(layer_steps(
			b, layer, source, group,
			is_pieces ? steps : steps & ~(uint32_t)QUERIES));
	}
    return status;
}

/* A chunk's walk (model_seam.h): a wide program's (WIDE, agent H's B) the embedding, layers 0 to 19 and layer 20's
 * front; a full program's every layer, no head (its steps carry no row of logits: the engine leaves a prompt's last
 * rows to thin steps). A card's step segment holds each slab's rotation step; the embedding is one launch over the
 * chunk; a layer is one segment (its experts resident). Of a layer every launch that reads no state runs over the
 * chunk, a piece of up to 1,024 rows at a time (the wide launches of linear.h and dense_linear.h, the n-gram rows' and
 * the hyper sites' over their rows: the bytes of the slabs' thin launches), each behind every launch that writes what
 * it reads: the n-gram rows, engram.wkv and the gate; the mix and finish; the quantization, wq_a with wkv, a source's
 * compressor products, qr alone (attention_state.h), wq_b, the indexer's and its weights; then each slab's state
 * writes, scan and core in slab order, so that a slab's attention finishes before the next slab's state writes, as
 * consecutive thin steps would, the queries of each span of 8 slabs (a span's arrays, model_v41.h) ahead of its first
 * slab; wo_a, wo_b and the post; the FFN (wide_ffn_launches). Layer 20's front (its
 * state writes) is a segment of its own behind which its copies go, as in a thin program; a wide walk ends there,
 * every segment a front step's; a full walk takes the rest of layer 20 (the front of all slabs has written every
 * window, pool entry and key the rest reads, and QUERY_NORMED is a chunk's), then its FFN and later copies. */
MUST_CHECK static result_status chunk_walk(struct builder *b, bool is_full)
{
	const struct model_plan_record *const plan = b->plan;
	const uint32_t last_unit = is_full ? MODEL_LAYERS :
					     MODEL_DECODER_FIRST_LAYER + 1u;
	uint32_t source = 0, group = 0;
	result_status status = OK;
	for (uint32_t unit = 0; unit <= last_unit && status == OK; unit++) {
		const uint32_t layer = unit - 1u;
		const uint32_t card = unit == 0 ? plan->embedding_card :
						  plan->layer_card[layer];
		const bool begins_layers = visit_begins_layers(b, unit, card);
		if (unit != 0 && card != b->card) {
			stage_hand_off(b, card);
		}
		status = unit != 0 && card == b->card ? OK :
							card_select(b, card);
		uint32_t first_launch =
			status == OK ? b->made->cards[b->card].launch_count : 0;
		if (status == OK &&
		    begins_layers) { /* the visit's step segment */
			for (uint32_t slab = 0;
			     slab < b->slab_count && status == OK; slab++) {
				slab_select(b, slab);
				status = step_launches(b, false);
			}
			segment_add(b, SEGMENT_NO_LAYER, first_launch);
			first_launch = b->made->cards[b->card].launch_count;
		}
		if (status == OK && unit == 0) { /* the embedding */
			chunk_select(b);
			status = step_launches(b, true);
			segment_add(b, SEGMENT_NO_LAYER, first_launch);
		}
		if (status != OK || unit == 0) {
			continue;
		}
		source = (V41_DATA_SOURCE_LAYERS >> layer & 1u) != 0 ? layer :
								       source;
		group = (V41_DATA_INDEX_LAYERS >> layer & 1u) != 0 ? layer :
								     group;
		const bool is_cut = layer == MODEL_DECODER_FIRST_LAYER;
		if (plan->limits.draft_positions != 0 &&
		    layer >= MODEL_DRAFT_FIRST_TAP_LAYER) {
			chunk_select(b);
			status = draft_tap_launch(b, layer);
		}
		/* over the chunk: ahead of the slabs' state writes, and the rest's products behind qr alone (layer 20's behind
         * its front, whose prepares wrote qr) */
		const uint32_t ahead = ENGRAM_STEPS | SITE_MIX |
				       INPUT_PRODUCTS | POOL_PRODUCTS;
		const uint32_t queries = QUERY_PRODUCTS | INDEX_WEIGHTS_PRODUCT;
		status = THEN(steps_over(
			b, layer, source, group,
			ahead | (is_cut ? 0 : QUERY_NORM | queries), true));
		status = THEN(steps_over(
			b, layer, source, group,
			STATE_WRITES | (is_cut ? 0 : QUERIES | CORE), false));
		if (status == OK && is_cut) {
			front_end(b, layer, &first_launch);
			status = is_full ? steps_over(b, layer, source, group,
						      queries, true) :
					   OK;
			status = is_full ?
					 THEN(steps_over(b, layer, source,
							 group, QUERIES | CORE,
							 false)) :
					 status;
		}
		if (status == OK && (!is_cut || is_full)) {
			status = steps_over(b, layer, source, group,
					    OUTPUT_PRODUCTS | SITE_POST, true);
			status = THEN(wide_ffn_launches(b, layer));
			segment_add(b, layer, first_launch);
			copies_hand_off(b, layer,
					is_cut ? LATER_COPIES : EVERY_COPY);
			if (status == OK && plan->limits.draft_positions != 0 &&
			    layer + 1u == MODEL_LAYERS) {
				first_launch =
					b->made->cards[b->card].launch_count;
				status = draft_context_launches(
					b, LINEAR_WIDE_MOST_ROWS);
				segment_add(b, SEGMENT_NO_LAYER, first_launch);
			}
		}
    }
    return status;
}

/* The program's leases (model_seam.h): on each card, of each index layer's selection and of the candidates, from the
 * visit that writes them (the index layer, layer 20) or takes them in (the card's first behind the writer, where
 * their hand-off goes up, whether or not it reads them) to the last visit of the card that reads them (the group's
 * layers, the candidate readers), where those two differ. */
static uint32_t leases_make(const struct model_plan_record *plan,
			    struct made_program *made)
{
	const struct execution_program *const program = &made->program;
	const struct segment_record *const segments = made->segments;
	uint32_t layer_segment[MODEL_LAYERS], visit_start[MOST_SEGMENTS],
		count = 0;
	for (uint32_t layer = 0; layer < MODEL_LAYERS; layer++) {
		layer_segment[layer] = UINT32_MAX;
	}
    for (uint32_t segment = 0; segment < program->segment_count; segment++) {
        visit_start[segment] = segment != 0 && segments[segment - 1u].card == segments[segment].card
                                   ? visit_start[segment - 1u] : segment;
	if (segments[segment].layer < MODEL_LAYERS) {
		layer_segment[segments[segment].layer] =
			segment; /* its last: layer 20's rest */
	}
    }
    for (uint32_t card = 0; card < plan->card_count; card++) {
	    for (uint32_t group = 0; group <= MODEL_LAYERS;
		 group++) { /* MODEL_LAYERS: candidates */
		    const bool is_candidates = group == MODEL_LAYERS;
		    const uint32_t writer =
			    is_candidates ? MODEL_CANDIDATE_SOURCE : group;
		    uint32_t last_read = UINT32_MAX,
			     taken = layer_segment[writer];
		    for (uint32_t layer = writer;
			 layer < MODEL_LAYERS &&
			 (is_candidates ||
			  (V41_DATA_INDEX_LAYERS >> group & 1u) != 0);
			 layer++) {
			    const bool reads =
				    is_candidates ?
					    (V41_DATA_CANDIDATE_READERS >>
						     layer &
					     1u) != 0 :
					    layer == group ||
						    (V41_DATA_INDEX_LAYERS >>
							     layer &
						     1u) == 0;
			    if (!is_candidates && layer != group && !reads) {
				    break; /* the next index layer: the group ends */
			    }
			    const bool is_read =
				    reads && plan->layer_card[layer] == card &&
				    layer_segment[layer] != UINT32_MAX;
			    last_read = is_read ? layer_segment[layer] :
						  last_read;
		    }
		    while (taken < program->segment_count &&
			   segments[taken].card != card) {
			    taken++; /* the writer's, or the card's first segment behind */
		    }
		    if (last_read != UINT32_MAX && taken < last_read &&
			visit_start[taken] != visit_start[last_read] &&
			count < MOST_LEASES) {
			    made->leases[count++] =
				    (struct visit_lease){ card, taken,
							  last_read };
		    }
	    }
    }
    return count;
}

MUST_CHECK static result_status
program_prepare_mode(const struct model_plan_record *plan,
		     const struct backend_operations *backend,
		     struct device_card *const *cards,
		     const struct card_description_record *descriptions,
		     enum program_kind kind, uint32_t row_count,
		     bool vision_active, struct execution_program **program_out)
{
	if (plan == NULL || backend == NULL || cards == NULL ||
	    descriptions == NULL || program_out == NULL ||
	    plan->card_count - 1u >= BACKEND_MAXIMUM_CARDS ||
	    row_count - 1u >= plan->limits.row_capacity ||
	    (uint32_t)kind >= PROGRAM_KIND_COUNT ||
	    (vision_active && !plan->vision.enabled)) {
		LOG_ERROR(
			"program_prepare: a missing argument, a plan of no cards, %" PRIu32
			" rows (1 to the plan's) "
			"or kind %" PRIu32,
			row_count, (uint32_t)kind);
		return STATUS_INVALID_ARGUMENT;
	}
	if (kind != PROGRAM_THIN && kind != PROGRAM_WIDE &&
	    kind != PROGRAM_FULL) {
		LOG_ERROR(
			"program_prepare: kind %" PRIu32
			": this build makes thin, wide and full programs alone",
			(uint32_t)kind);
		return STATUS_UNSUPPORTED;
	}
	const bool is_wide = kind != PROGRAM_THIN,
		   is_full = kind == PROGRAM_FULL; /* chunked */
	if (is_wide ? row_count < MODEL_WIDE_LEAST_ROWS :
		      row_count > v41_slab_rows(plan->limits.row_capacity)) {
		LOG_ERROR("program_prepare: a %s program of %" PRIu32
			  " rows: a thin one takes at most %" PRIu32
			  ", a chunked one at least %u",
			  is_wide ? "chunked" : "thin", row_count,
			  v41_slab_rows(plan->limits.row_capacity),
			  MODEL_WIDE_LEAST_ROWS);
		return STATUS_INVALID_ARGUMENT;
	}
    if (!plan_is_valid(plan)) {
	    return STATUS_INVALID_ARGUMENT;
    }
    if (plan->limits.official_reference != 0 &&
	(kind != PROGRAM_THIN || row_count != 1 || vision_active))
	    return STATUS_UNSUPPORTED;
    if (plan->limits.official_hc_rows != 0) {
	    if (is_wide || vision_active || plan->limits.draft_positions != 0 ||
		(row_count != 1 && row_count != OFFICIAL_HC_ODD_PROMPT_ROWS &&
		 row_count != MODEL_STREAMS &&
		 row_count != LINEAR_MAXIMUM_ROWS))
		    return STATUS_UNSUPPORTED;
	    for (uint32_t card = 0; card < plan->card_count; card++) {
		    if (card_has_exl3(plan, card))
			    return STATUS_UNSUPPORTED;
	    }
    }
    /* Ordinary PP-wide EXL3 experts are not qualified by the exact TP slab composition. */
    for (uint32_t card = 0; card < plan->card_count && is_wide; card++) {
        if (card_has_exl3(plan, card))
		return STATUS_UNSUPPORTED;
    }
    /* the layers a chunk's walk takes: a wide one layers 0 to 20 (20's front), a full one all */
    const uint64_t chunk_layers =
	    is_full ? UINT64_MAX :
		      (1ull << (MODEL_DECODER_FIRST_LAYER + 1u)) - 1u;
    for (uint32_t layer = 0; layer < MODEL_LAYERS && is_wide; layer++) {
	    if ((chunk_layers >> layer & 1u) != 0 &&
		plan->host_expert_count[layer] != 0) {
		    LOG_ERROR(
			    "program_prepare: a chunked program needs the routed experts of its layers on their "
			    "cards; layer %" PRIu32 " keeps %" PRIu32
			    " in host memory",
			    layer, plan->host_expert_count[layer]);
		    return STATUS_UNSUPPORTED;
	    }
    }
    const uint32_t slabs = is_wide ? (row_count + LINEAR_MAXIMUM_ROWS - 1u) /
					     LINEAR_MAXIMUM_ROWS :
				     1u;
    uint32_t launch_capacity[BACKEND_MAXIMUM_CARDS] = { 0 }, launch_total = 0;
    for (uint32_t card = 0; card < plan->card_count; card++) {     /* each card's part of the launches */
        const uint32_t layers = (uint32_t)__builtin_popcountll(plan->cards[card].layers & (is_wide ? chunk_layers
                                                                                                    : UINT64_MAX));
        launch_capacity[card] = is_wide ? layers * (slabs * LAYER_MOST_LAUNCHES + CHUNK_LAUNCHES) +
                                              slabs * CARD_MOST_LAUNCHES
                                        : layers * LAYER_MOST_LAUNCHES + CARD_MOST_LAUNCHES;
	for (uint32_t layer = 0; layer < MODEL_LAYERS; layer++) {
		if ((plan->cards[card].layers >> layer & 1u) != 0 &&
		    plan->exl3_bits[layer] != 0)
			launch_capacity[card] +=
				EXL3_ATTENTION_EXTRA_LAUNCHES * slabs;
	}
	if (vision_active) {
            launch_capacity[card] += layers * VISION_FFN_EXTRA_LAUNCHES + (card == 0 ? slabs : 0u);
        }
        if (plan->limits.draft_positions != 0 && card == plan->result_card && (!is_wide || is_full)) {
		const uint32_t pieces =
			is_wide ? (row_count + LINEAR_WIDE_MOST_ROWS - 1u) /
					  LINEAR_WIDE_MOST_ROWS :
				  1u;
		launch_capacity[card] +=
			MODEL_DRAFT_TAPS + pieces * CONTEXT_PIECE_LAUNCHES +
			MODEL_DRAFT_STAGES +
			(is_wide ? 0 :
				   DRAFT_MOST_LAUNCHES(
					   plan->limits.draft_positions));
	}
	if (plan->limits.official_reference != 0)
		launch_capacity[card] +=
			layers * LINEAR_MAXIMUM_ROWS + LINEAR_MAXIMUM_ROWS;
	launch_total += launch_capacity[card];
    }
    struct builder *const b = FAULT_POINT(model_v41_program_memory) ?
				      NULL :
				      calloc(1, sizeof(*b));
    struct made_program *const made =
	    b == NULL || FAULT_POINT(model_v41_program_memory) ?
		    NULL :
		    calloc(1,
			   sizeof(*made) + (size_t)launch_total *
						   sizeof(made->launches[0]));
    if (made == NULL) {
	    LOG_ERROR("program_prepare: no memory for a program of %" PRIu32
		      " rows",
		      row_count);
	    free(b);
	    return STATUS_OUT_OF_MEMORY;
    }
    *b = (struct builder){ .plan = plan,
			   .backend = backend,
			   .cards = cards,
			   .descriptions = descriptions,
			   .made = made,
			   .rows = row_count,
			   .card = plan->embedding_card,
			   .chunk_rows = row_count,
			   .program_rows = row_count,
			   .slab_count = slabs,
			   .vision_active = vision_active,
			   .head_stream_kernel = BACKEND_NO_KERNEL,
			   .head_mma_kernel = BACKEND_NO_KERNEL };
    for (uint32_t card = 0, base = 0; card < plan->card_count; card++) {
	    const struct v41_card *const planned = &plan->cards[card];
	    const uint64_t *const arrays = planned->array_offset,
				  nowhere = PROGRAM_NOWHERE;
	    const bool has_views = is_wide &&
				   (card == plan->embedding_card ||
				    (planned->layers & chunk_layers) != 0);
	    b->launches[card] = made->launches + base;
	    b->launch_capacity[card] = launch_capacity[card];
	    base += launch_capacity[card];
	    for (uint32_t slab = 0; slab < slabs && has_views; slab++) {
		    made->views[card][slab] = (struct step_view){
			    arrays[V41_DATA_ARRAY_VIEWS] +
				    (uint64_t)slab * VIEW_BYTES,
			    slab * LINEAR_MAXIMUM_ROWS, LINEAR_MAXIMUM_ROWS
		    };
	    }
	    made->cards[card] = (struct program_card_record){
		    b->launches[card],
		    card == 0 && plan->vision.enabled ?
			    plan->vision.text_arena_bytes :
			    planned->arena_byte_count,
		    arrays[V41_DATA_ARRAY_STEP],
		    arrays[V41_DATA_ARRAY_SESSIONS],
		    arrays[V41_DATA_ARRAY_RESULT],
		    arrays[V41_DATA_ARRAY_LOGITS],
		    planned->layers != 0 ? arrays[V41_DATA_ARRAY_STREAMS] :
					   nowhere,
		    planned->layers != 0 ? arrays[V41_DATA_ARRAY_CARRIED_PRE] :
					   nowhere,
		    arrays[V41_DATA_ARRAY_ENGRAM_ROWS],
		    planned->engram_tables,
		    0,
		    has_views ? made->views[card] : NULL,
		    has_views ? slabs : 0,
		    !is_wide && plan->limits.draft_positions != 0 &&
				    card == plan->result_card ?
			    arrays[V41_DATA_ARRAY_ROUND] :
			    nowhere
	    };
    }
    const result_status status = is_wide ? chunk_walk(b, is_full) :
					   program_walk(b);
    if (status == OK) {
        const uint32_t draft_first = !is_wide && plan->limits.draft_positions != 0
                                         ? made->program.draft_segment_first : made->program.segment_count;
	made->program = (struct execution_program){
		made->cards,
		made->segments,
		made->hand_offs,
		plan->limits,
		plan->card_count,
		made->program.segment_count,
		made->program.hand_off_count,
		row_count,
		plan->result_card,
		made->expert_pauses,
		made->host_expert_offsets,
		made->program.expert_pause_count,
		made->program.host_expert_count,
		kind,
		0,
		made->session_ranges,
		session_ranges_make(plan, made->session_ranges),
		made->program.front_segment_count,
		MODEL_CONE_ROWS,
		plan->limits.draft_positions != 0 ? V41_DATA_WALK_LAYERS :
						    MODEL_LAYERS,
		made->checkpoints,
		made->leases,
		draft_first
	};
	checkpoints_make(plan, made->checkpoints);
        made->program.lease_count = leases_make(plan, made);
        *program_out = &made->program;
    } else {
        free(made);
    }
    free(b);
    return status;
}

MUST_CHECK static result_status
program_prepare(const struct model_plan_record *plan,
		const struct backend_operations *backend,
		struct device_card *const *cards,
		const struct card_description_record *descriptions,
		enum program_kind kind, uint32_t row_count,
		struct execution_program **program_out)
{
    return program_prepare_mode(plan, backend, cards, descriptions, kind, row_count, false, program_out);
}

result_status
v41_image_program_prepare(const struct model_plan_record *plan,
			  const struct backend_operations *backend,
			  struct device_card *const *cards,
			  const struct card_description_record *descriptions,
			  enum program_kind kind, uint32_t row_count,
			  struct execution_program **program_out)
{
    return program_prepare_mode(plan, backend, cards, descriptions, kind, row_count, true, program_out);
}

static void program_destroy(struct execution_program *program)
{
    free(program);                          /* the address of its struct made_program */
}

result_status v41_tp2_layer_program_prepare(
	const struct v41_tp2_layer_plan *plan,
	const struct backend_operations *backend,
	struct device_card *const *cards,
	const struct card_description_record *descriptions, uint32_t rows,
	uint32_t slice_count, bool local_stub,
	struct v41_tp2_layer_program *program_out)
{
	if (plan == NULL || plan->model == NULL || backend == NULL ||
	    cards == NULL || descriptions == NULL || program_out == NULL ||
	    plan->card >= plan->model->card_count ||
	    cards[plan->card] == NULL ||
	    descriptions[plan->card].arithmetic_class !=
		    ARITHMETIC_CLASS_SM_80 ||
	    rows == 0 || rows > plan->model->limits.row_capacity ||
	    rows > LINEAR_MAXIMUM_ROWS || slice_count == 0 ||
	    slice_count > LINEAR_MAXIMUM_SLICES ||
	    (slice_count & (slice_count - 1u)) != 0 ||
	    (plan->rank_count != 1u && plan->rank_count != TP2_RANKS) ||
	    plan->rank >= plan->rank_count)
		return STATUS_INVALID_ARGUMENT;
	struct builder *const build = calloc(1, sizeof(*build));
	struct made_program *const made =
		calloc(1, sizeof(*made) + (size_t)TP2_LAYER_LAUNCHES *
						  sizeof(struct kernel_launch));
	if (build == NULL || made == NULL) {
		free(build);
		free(made);
		return STATUS_OUT_OF_MEMORY;
	}
	*build = (struct builder){
		.tensor_layer = plan->rank_count == TP2_RANKS ? plan : NULL,
		.tensor_slices = slice_count,
		.tensor_local_stub = local_stub,
		.plan = plan->model,
		.backend = backend,
		.cards = cards,
		.descriptions = descriptions,
		.made = made,
		.rows = rows,
		.chunk_rows = rows,
		.program_rows = rows,
		.slab_count = 1,
		.card = plan->card,
		.head_stream_kernel = BACKEND_NO_KERNEL,
		.head_mma_kernel = BACKEND_NO_KERNEL
	};
	build->launches[plan->card] = made->launches;
	build->launch_capacity[plan->card] = TP2_LAYER_LAUNCHES;
	made->program.kind = PROGRAM_THIN;
	result_status status = card_select(build, plan->card);
	if (status == OK && build->tensor_layer != NULL) {
		status = tp2_join_kernel_prepare(backend, cards[plan->card],
						 &build->tensor_join_kernel);
		status = plan->parallel_publish ?
				 THEN(ordered_push_parallel_kernels_prepare(
					 backend, cards[plan->card],
					 build->tensor_parallel_kernels)) :
				 THEN(ordered_push_kernel_prepare(
					 backend, cards[plan->card],
					 &build->tensor_push_kernel));
	}
    if (plan->experts_partitioned)
	    status = THEN(
		    ep_filter_kernel_prepare(backend, cards[plan->card],
					     &build->tensor_ep_filter_kernel));
    status = THEN(step_launches(build, false));
    const struct v41_tp2_layer_plan *const expert_shards = build->tensor_layer;
    if (plan->attention_replicated)
        build->tensor_layer = NULL;
    status = THEN(layer_steps(build, 0, 0, 0, FRONT_STEPS | REST_STEPS));
    build->tensor_layer = expert_shards;
    uint32_t first_launch = 0;
    status = THEN(ffn_launches(build, 0, &first_launch));
    if (status == OK) {
	    struct v41_tp2_layer_program result = { 0 };
	    result.launch_count = made->cards[plan->card].launch_count;
	    memcpy(result.launches, made->launches,
		   (size_t)result.launch_count * sizeof(result.launches[0]));
	    *program_out = result;
    }
    free(made);
    free(build);
    return status;
}

static const struct model_operations model_v41 = {
	V41_DATA_MODEL_NAME,  v41_describe,	v41_state_byte_count,
	v41_plan_create,      v41_plan_destroy, v41_arena_byte_count,
	v41_arena_initialise, program_prepare,	program_destroy
};

const struct model_operations *model_operations_of(const char *model_name)
{
	return model_name != NULL &&
			       memchr(model_name, '\0', MODEL_NAME_CAPACITY) !=
				       NULL &&
			       strcmp(model_name, model_v41.name) == 0 ?
		       &model_v41 :
		       NULL;
}

/* Full text-only TP2 composition is separate from every default PP factory above. */
#define TP2_PROGRAM_CHECKPOINTS \
	(V41_DATA_WALK_LAYERS * \
	 TP4_RANKS) /* draft_layers: every rank's draft stages too */
#define TP2_PROGRAM_ROTATION_PAIRS 2u
#define TP2_PROGRAM_FRONT_PAIRS 1u
#define TP2_PROGRAM_PAIRS \
	(MODEL_LAYERS + TP2_PROGRAM_ROTATION_PAIRS + TP2_PROGRAM_FRONT_PAIRS)

static bool tp2_program_limits_equal(const struct model_limits *left,
				     const struct model_limits *right)
{
	return left->context_token_limit == right->context_token_limit &&
	       left->row_capacity == right->row_capacity &&
	       left->session_capacity == right->session_capacity &&
	       left->state_width == right->state_width &&
	       left->linear_profile == right->linear_profile &&
	       left->linear_order == right->linear_order &&
	       left->cmp_order == right->cmp_order &&
	       left->draft_positions == right->draft_positions &&
	       left->official_hc_rows == right->official_hc_rows &&
	       left->official_reference == right->official_reference;
}

static uint32_t
tp2_program_layer_card(const struct v41_tp2_program_description *description,
		       uint32_t layer)
{
    return description->canonical->layer_card[layer];
}

static uint64_t tp2_program_layer_mask(uint32_t first, uint32_t count)
{
    return ((1ull << count) - 1u) << first;
}

static bool tp2_program_tensor_equal(const struct v41_tensor *left,
				     const struct v41_tensor *right)
{
    return left->stored.byte_offset == right->stored.byte_offset &&
           left->stored.byte_count == right->stored.byte_count && left->shape.output_size == right->shape.output_size &&
           left->shape.input_size == right->shape.input_size &&
           left->shape.bits_per_weight == right->shape.bits_per_weight &&
           left->shape.reserved_must_be_zero == right->shape.reserved_must_be_zero &&
           left->weight_format == right->weight_format &&
           left->card == right->card && left->fold_reference == right->fold_reference;
}

/* The new topology is admitted only through an explicit four-CMP, full-target description.
 * Runtime preparation retains the model's validated four-card geometry and qualified peer ownership. */
static bool
tp2_program_exl3_all(const struct v41_tp2_program_description *description)
{
    return description->stages[0] != NULL && description->stages[0]->description.exl3_all_targets;
}

static bool tp2_program_exl3_layer_is_valid(
	const struct v41_tp2_program_description *description, uint32_t card,
	uint32_t layer)
{
	const struct v41_tp2_stage_plan *const stage =
		description->stages[card];
	const struct model_plan_record *const canonical =
						      description->canonical,
					      *const view =
						      description->views[card];
	struct bb_exl3_output_shard_plan expected;
	const uint32_t bits = canonical->exl3_bits[layer];
	if (bits == 0 || bits != view->exl3_bits[layer] ||
	    bits != stage->model->exl3_bits[layer])
		return false;
	if (!stage->description.exact_output_shards) {
		const struct bb_exl3_output_shard_plan empty = { 0 };
		return stage->description.wire_format == ORDERED_PUSH_BF16 &&
		       stage->description.exl3_product_order ==
			       BB_EXL3_REFERENCE_ORDER &&
		       memcmp(&empty, &stage->exl3_shards[layer],
			      sizeof(empty)) == 0;
	}
	if (bb_exl3_output_shard_prepare(
		    MODEL_HIDDEN, V41_DATA_EXPERT_INNER, bits, TP4_RANKS, card,
		    stage->description.exl3_product_order, &expected) != OK ||
	    memcmp(&expected, &stage->exl3_shards[layer], sizeof(expected)) !=
		    0)
		return false;
	return true;
}

/* replicated_embedding (model_v41_tp2_model.h): each rank of the first stage embeds its rows itself. */
static bool
tp2_program_replicated(const struct v41_tp2_program_description *description)
{
    return description->stages[0]->description.rank_count > 1u && description->stages[1] != NULL &&
        description->stages[1]->description.include_embedding;
}

/* Original BATCH keeps target results, the round table and seven extra legacy verifier areas
 * behind the native tail's ordinary arena. Bound the complete downloads and every verifier before building. */
static bool tp2_program_original_batch_areas_are_valid(
	const struct v41_tp2_stage_plan *stage)
{
	const struct linear_range results = stage->batch_results,
				  rounds = stage->batch_rounds;
	const struct linear_range drafts = stage->batch_drafts;
	if (stage->description.card != V41_DATA_TP2_PRO_CARD)
		return results.byte_count == 0 && rounds.byte_count == 0 &&
		       drafts.byte_count == 0 &&
		       stage->batch_logits.byte_count == 0 &&
		       stage->batch_rows.byte_count == 0;
	const uint64_t arena =
		stage->model->cards[stage->description.card].arena_byte_count;
	const uint64_t result_bytes = (uint64_t)PROGRAM_BATCH_MOST_ROWS /
				      LINEAR_MAXIMUM_ROWS *
				      PROGRAM_BATCH_SLAB_RESULT_BYTES;
	const uint64_t round_bytes =
		(uint64_t)V41_DATA_ORIGINAL_BATCH_SESSIONS * STEP_ROUND_BYTES;
	const uint64_t draft_bytes =
		(uint64_t)(V41_DATA_ORIGINAL_BATCH_SESSIONS - 1u) *
		v41_batch_draft_offset(V41_DATA_BATCH_DRAFT_PARTS,
				       MODEL_DRAFT_BLOCK);
	return results.byte_count == result_bytes &&
	       rounds.byte_count == round_bytes &&
	       drafts.byte_count == draft_bytes &&
	       results.byte_offset <= arena &&
	       result_bytes <= arena - results.byte_offset &&
	       rounds.byte_offset >= results.byte_offset &&
	       rounds.byte_offset - results.byte_offset ==
		       v41_batch_aligned(result_bytes) &&
	       rounds.byte_offset <= arena &&
	       round_bytes <= arena - rounds.byte_offset &&
	       rounds.byte_offset - results.byte_offset <=
		       STEP_BATCH_RESULT_SPAN_MOST - round_bytes &&
	       drafts.byte_offset >= rounds.byte_offset &&
	       drafts.byte_offset - rounds.byte_offset ==
		       v41_batch_aligned(round_bytes) &&
	       drafts.byte_offset <= arena &&
	       draft_bytes <= arena - drafts.byte_offset &&
	       stage->batch_logits.byte_count == 0 &&
	       stage->batch_rows.byte_count == 0;
}

static bool
tp2_program_view_is_valid(const struct v41_tp2_program_description *description,
			  uint32_t card, uint32_t rows)
{
	const struct model_plan_record *const canonical =
						      description->canonical,
					      *const view =
						      description->views[card];
	const struct v41_tp2_stage_plan *const stage =
		description->stages[card];
	if (stage == NULL || stage->model == NULL ||
	    stage->description.rank > card ||
	    (stage->description.rank_count != 1u &&
	     stage->description.rank_count != TP2_RANKS &&
	     stage->description.rank_count != TP4_RANKS))
		return false;
	const uint32_t first = stage->description.first_target_layer,
		       count = stage->description.target_layer_count;
	const uint32_t ranks = stage->description.rank_count,
		       primary = card - stage->description.rank;
	const bool tensor = ranks > 1u;
	const bool exl3_all = tp2_program_exl3_all(description);
	const bool original_batch =
		description->stages[0]->description.original_batch;
	if (stage->description.original_batch != original_batch ||
	    (original_batch &&
	     (!v41_original_batch_stage_is_valid(&stage->description) ||
	      !tp2_program_original_batch_areas_are_valid(stage))))
		return false;
	const bool original_wide =
		description->stages[0]->description.original_wide_experts;
	if (stage->description.original_wide_experts !=
	    (tensor && original_wide))
		return false;
	if (tensor && original_wide) {
		const uint64_t hidden_bytes =
			(uint64_t)canonical->limits.row_capacity *
			MODEL_EXPERTS_PER_ROW * V41_DATA_EXPERT_INNER *
			LINEAR_BFLOAT16_BYTES;
		const uint64_t sum_bytes =
			(uint64_t)canonical->limits.row_capacity *
			MODEL_HIDDEN * LINEAR_SUM_BYTES;
		if (ranks != TP4_RANKS || first != 0 ||
		    count != TP4_ATTENTION_PREFIX_END ||
		    !stage->description.exact_output_shards ||
		    stage->description.original_code_wire || exl3_all ||
		    stage->description.wire_format != 0 ||
		    stage->description.batch ||
		    stage->description.split_overlap != 0 ||
		    stage->description.full_split_overlap != 0 ||
		    stage->description.exl3_wide_entry ||
		    stage->description.exl3_tc_moe ||
		    canonical->limits.linear_profile !=
			    LINEAR_PROFILE_PRECISION ||
		    canonical->limits.linear_order != LINEAR_ORDER_REFERENCE ||
		    canonical->limits.cmp_order != CMP_ORDER_REFERENCE ||
		    stage->exact_hidden.byte_count < hidden_bytes ||
		    stage->exact_sums.byte_count < sum_bytes)
			return false;
	}
	if (stage->description.exl3_all_targets != exl3_all ||
	    (exl3_all &&
	     (card != stage->description.rank || ranks != TP4_RANKS ||
	      first != 0 || count != MODEL_LAYERS ||
	      stage->description.original_code_wire ||
	      stage->description.attention_owner ||
	      stage->description.wire_format !=
		      (stage->description.exact_output_shards ?
			       0u :
			       ORDERED_PUSH_BF16) ||
	      (!stage->description.exact_output_shards &&
	       stage->description.exl3_product_order !=
		       BB_EXL3_REFERENCE_ORDER) ||
	      stage->description.exl3_product_order !=
		      description->stages[0]->description.exl3_product_order)))
		return false;
	if ((card < V41_DATA_TP2_PRO_CARD &&
	     stage->description.attention_owner !=
		     description->stages[0]->description.attention_owner) ||
	    (stage->description.attention_owner &&
	     (!stage->description.exact_output_shards || ranks != TP4_RANKS ||
	      first != 0 || count != TP4_ATTENTION_PREFIX_END)))
		return false;
	if (first >= MODEL_LAYERS || count == 0 ||
	    count > MODEL_LAYERS - first || stage->description.rank >= ranks ||
	    (tensor && primary + ranks > V41_DATA_TP2_PRO_CARD))
		return false;
	for (uint32_t layer = first; layer < first + count; layer++) {
		if (canonical->layer_card[layer] != primary)
			return false;
	}
	if (view == NULL || stage == NULL || stage->model == NULL ||
	    !tp2_program_limits_equal(&canonical->limits, &view->limits) ||
	    view->card_count != canonical->card_count || view->vision.enabled ||
	    view->embedding_card != canonical->embedding_card ||
	    view->result_card != canonical->result_card ||
	    stage->description.card != card ||
	    stage->description.first_target_layer != first ||
	    stage->description.target_layer_count != count ||
	    stage->description.include_embedding !=
		    (card == 0 || tp2_program_replicated(description)) ||
	    stage->description.include_head_and_draft !=
		    (card == canonical->result_card) ||
	    canonical->cards[card].layers !=
		    tp2_program_layer_mask(first, count) ||
	    stage->model->cards[card].arena_byte_count !=
		    canonical->cards[card].arena_byte_count ||
	    (tensor &&
	     (stage->channels == NULL ||
	      stage->channel_count != (count + (stage->description.draft_layers ?
							MODEL_DRAFT_STAGES :
							0u)) *
					      TP2_SITES ||
	      stage->channel_rows < rows || stage->compact.byte_count == 0)) ||
	    (!tensor && stage->channel_count != 0))
		return false;
	for (uint32_t other = 0; other < canonical->card_count; other++) {
		const struct v41_card *const local = &view->cards[other],
					     *const common =
						     &canonical->cards[other];
		if (local->arena_byte_count != common->arena_byte_count ||
		    local->state_offset != common->state_offset ||
		    local->front_scratch_offset !=
			    common->front_scratch_offset ||
		    (exl3_all && local->exl3_scratch_offset !=
					 common->exl3_scratch_offset) ||
		    local->producer_arrivals_offset !=
			    common->producer_arrivals_offset ||
		    local->slots_byte_count != common->slots_byte_count ||
		    local->slot_count != common->slot_count ||
		    local->image != common->image ||
		    local->image_byte_count != common->image_byte_count ||
		    local->layers != common->layers ||
		    local->engram_tables != common->engram_tables ||
		    local->holders != common->holders ||
		    memcmp(local->array_offset, common->array_offset,
			   sizeof(local->array_offset)) != 0 ||
		    memcmp(local->expert_table_offset,
			   common->expert_table_offset,
			   sizeof(local->expert_table_offset)) != 0 ||
		    memcmp(local->selection_offset, common->selection_offset,
			   sizeof(local->selection_offset)) != 0 ||
		    memcmp(local->ring_offset, common->ring_offset,
			   sizeof(local->ring_offset)) != 0 ||
		    memcmp(local->checkpoint_offset, common->checkpoint_offset,
			   sizeof(local->checkpoint_offset)) != 0 ||
		    memcmp(local->compressor_offset, common->compressor_offset,
			   sizeof(local->compressor_offset)) != 0 ||
		    memcmp(local->pool_offset, common->pool_offset,
			   sizeof(local->pool_offset)) != 0 ||
		    memcmp(local->key_tiles_offset, common->key_tiles_offset,
			   sizeof(local->key_tiles_offset)) != 0)
			return false;
	}
    if (memcmp(view->pool_session_bytes, canonical->pool_session_bytes, sizeof(view->pool_session_bytes)) != 0 ||
        memcmp(view->key_session_bytes, canonical->key_session_bytes, sizeof(view->key_session_bytes)) != 0 ||
        memcmp(view->layer_card, canonical->layer_card, sizeof(view->layer_card)) != 0)
        return false;
    for (uint32_t global = 0; global < V41_DATA_GLOBALS; global++) {
	    /* head_shards: ranks 1-3 hold their own norm and head rows; replicated_embedding: every rank its embedding */
	    const bool own = global == V41_DATA_EMBEDDING ?
				     stage->description.include_embedding :
				     stage->description.head_shards &&
					     card != canonical->result_card;
	    if (!tp2_program_tensor_equal(&view->globals[global],
					  own ? &stage->model->globals[global] :
						&canonical->globals[global]) ||
		(own && view->globals[global].card != card))
		    return false;
    }
    for (uint32_t global = 0; global < V41_DATA_DRAFT_GLOBALS; global++) {
	    /* draft_shards: ranks 1-3 hold their own Markov tables; draft_layers: main_proj, main_norm, the final norm */
	    const bool own = (((global == V41_DATA_DRAFT_MARKOV_EMBEDDING ||
				global == V41_DATA_DRAFT_MARKOV_HEAD) &&
			       stage->description.draft_shards) ||
			      ((global == V41_DATA_DRAFT_MAIN_PROJECTION ||
				global == V41_DATA_DRAFT_MAIN_NORM ||
				global == V41_DATA_DRAFT_NORM) &&
			       stage->description.draft_layers)) &&
			     card != canonical->result_card;
	    if (!tp2_program_tensor_equal(
			&view->draft_globals[global],
			own ? &stage->model->draft_globals[global] :
			      &canonical->draft_globals[global]) ||
		(own && view->draft_globals[global].card != card))
		    return false;
    }
    for (uint32_t layer = first; layer < first + count; layer++) {
	    const struct v41_tensor *const tensors = view->layer_tensors[layer];
	    if ((original_wide || original_batch) &&
		(canonical->exl3_bits[layer] != 0 ||
		 view->exl3_bits[layer] != 0 ||
		 stage->model->exl3_bits[layer] != 0))
		    return false;
	    if (view->host_expert_count[layer] != 0 ||
		tensors[V41_DATA_ATTENTION_NORM].card != card ||
		tensors[V41_DATA_FFN_NORM].card != card)
		    return false;
	    for (uint32_t tensor_index = 0;
		 tensor_index < V41_DATA_LAYER_TENSOR_COUNT; tensor_index++) {
		    if (tensors[tensor_index].stored.byte_count != 0 &&
			tensors[tensor_index].card != card)
			    return false;
	    }
	if (exl3_all && !tp2_program_exl3_layer_is_valid(description, card, layer))
            return false;
        if (tensor) {
            if (stage->description.exact_output_shards != description->stages[0]->description.exact_output_shards ||
                stage->description.exact_dense != description->stages[0]->description.exact_dense)
                return false;
            /* exact_dense: wq_b's quarter of the heads, wo_a's quarter of the groups, wo_b's quarter of the outputs,
             * each over the full inputs; the dense gathers' slots and banks */
            const bool dense = stage->description.exact_dense;
	    if (dense &&
		(stage->dense_input.byte_count < (uint64_t)LINEAR_MAXIMUM_ROWS *
							 V41_DATA_WIDEST_ROW *
							 sizeof(uint16_t) ||
		 tensors[V41_DATA_OUTPUT_GROUPS].shape.output_size !=
			 V41_DATA_WIDEST_ROW / ranks ||
		 tensors[V41_DATA_OUTPUT].shape.output_size !=
			 MODEL_HIDDEN / ranks))
		    return false;
	    if (stage->description.exact_output_shards &&
		(ranks != TP4_RANKS ||
		 tensors[V41_DATA_QUERY_UP].shape.output_size !=
			 MODEL_HEADS_VALUE * MODEL_HEAD_DIMS /
				 (dense ? ranks : 1u) ||
		 tensors[V41_DATA_OUTPUT].shape.input_size !=
			 V41_DATA_WIDEST_ROW ||
		 tensors[V41_DATA_SHARED_GATE].shape.output_size !=
			 V41_DATA_EXPERT_INNER ||
		 tensors[V41_DATA_SHARED_UP].shape.output_size !=
			 V41_DATA_EXPERT_INNER ||
		 tensors[V41_DATA_SHARED_DOWN].shape.input_size !=
			 V41_DATA_EXPERT_INNER))
		    return false;
	    const uint32_t channel = (layer - first) * TP2_SITES;
	    const uint32_t inner =
		    ranks == TP4_RANKS ?
			    (stage->description.rank % TP2_RANKS == 0 ?
				     TP4_SMALL_INNER :
				     TP4_LARGE_INNER) :
			    V41_DATA_EXPERT_INNER / ranks;
	    if (!stage->description.exact_output_shards &&
		(tensors[V41_DATA_QUERY_UP].shape.output_size !=
			 MODEL_HEADS_VALUE * MODEL_HEAD_DIMS / ranks ||
		 tensors[V41_DATA_OUTPUT].shape.input_size !=
			 V41_DATA_WIDEST_ROW / ranks ||
		 tensors[V41_DATA_SHARED_GATE].shape.output_size != inner ||
		 tensors[V41_DATA_SHARED_UP].shape.output_size != inner ||
		 tensors[V41_DATA_SHARED_DOWN].shape.input_size != inner))
		    return false;
	    for (uint32_t site = 0; site < TP2_SITES; site++) {
		    const struct v41_tp2_stage_channel *const prepared =
			    &stage->channels[channel + site];
		    const struct ordered_push_arguments *const arguments =
			    &prepared->collective.arguments;
		    if (prepared->layer != layer || prepared->site != site ||
			arguments->rank != stage->description.rank ||
			arguments->rank_count != ranks ||
			arguments->payload_format !=
				stage->description.wire_format ||
			arguments->payload_format > ORDERED_PUSH_BF16 ||
			arguments->payload_format !=
				description->stages[0]->description.wire_format ||
			arguments->step_offset !=
				canonical->cards[card]
					.array_offset[V41_DATA_ARRAY_STEP])
			    return false;
	    }
	}
    }
    return true;
}

static bool
tp2_program_is_valid(const struct v41_tp2_program_description *description,
		     const struct backend_operations *backend,
		     struct device_card *const *cards,
		     const struct card_description_record *card_descriptions,
		     enum program_kind kind, uint32_t rows)
{
	if (description == NULL || description->canonical == NULL ||
	    backend == NULL || cards == NULL || card_descriptions == NULL ||
	    description->slice_count == 0 ||
	    description->slice_count > LINEAR_MAXIMUM_SLICES ||
	    (description->slice_count & (description->slice_count - 1u)) != 0)
		return false;
	const struct model_plan_record *const canonical =
		description->canonical;
	const struct model_limits *const limits = &canonical->limits;
	const bool batch = kind == PROGRAM_BATCH;
	const bool chunked = kind == PROGRAM_WIDE || kind == PROGRAM_FULL ||
			     batch;
	const bool exl3_all = tp2_program_exl3_all(description);
	const bool original_batch =
		description->stages[0] != NULL &&
		description->stages[0]->description.original_batch;
	if ((batch && ((!exl3_all && !original_batch) ||
		       !description->stages[0]->description.batch)) ||
	    (original_batch &&
	     (limits->linear_profile != LINEAR_PROFILE_PRECISION ||
	      limits->linear_order != LINEAR_ORDER_REFERENCE ||
	      limits->cmp_order != CMP_ORDER_REFERENCE ||
	      limits->session_capacity != V41_DATA_ORIGINAL_BATCH_SESSIONS ||
	      limits->draft_positions != MODEL_DRAFT_BLOCK ||
	      (batch && (rows != V41_DATA_ORIGINAL_BATCH_ROWS ||
			 (description->batch_verifiers != 0 &&
			  description->batch_verifiers !=
				  V41_DATA_ORIGINAL_BATCH_SESSIONS))))) ||
	    canonical->card_count !=
		    (exl3_all ? TP4_RANKS : V41_DATA_TP2_PROGRAM_CARDS) ||
	    canonical->result_card != (exl3_all ? 0u : V41_DATA_TP2_PRO_CARD))
		return false;
	for (uint32_t card = canonical->card_count;
	     card < V41_DATA_TP2_PROGRAM_CARDS; card++) {
		if (description->stages[card] != NULL ||
		    description->views[card] != NULL)
			return false;
	}
    /* thin_tail (model_v41_tp2_stage.h): full and wide programs of 9 to 63 rows too */
    const uint32_t least_chunk =
	    !batch && chunked && description->stages[0]->description.thin_tail ?
		    LINEAR_MAXIMUM_ROWS + 1u :
		    MODEL_WIDE_LEAST_ROWS;
    if ((!chunked && kind != PROGRAM_THIN) ||
	(batch ? rows > PROGRAM_BATCH_MOST_ROWS : /* a BATCH program of 8 rows or fewer walks as THIN */
		 chunked ? rows < least_chunk :
			   rows > LINEAR_MAXIMUM_ROWS) ||
	canonical->embedding_card != 0 || canonical->vision.enabled ||
	rows == 0 || rows > limits->row_capacity ||
	rows > V41_DATA_TP_SPLIT_MOST_ROWS || limits->row_capacity == 0 ||
	limits->row_capacity > V41_DATA_TP_SPLIT_MOST_ROWS ||
	limits->session_capacity == 0 ||
	limits->context_token_limit < MODEL_CONTEXT_WITHOUT_CANDIDATES ||
	limits->context_token_limit > MODEL_CONTEXT_MOST ||
	limits->context_token_limit % MODEL_CONTEXT_GRANULE != 0 ||
	limits->official_hc_rows != 0 ||
	limits->state_width != STATE_WIDTH_16 ||
	limits->cmp_order != CMP_ORDER_REFERENCE ||
	(limits->linear_profile != LINEAR_PROFILE_PRECISION &&
	 limits->linear_profile != LINEAR_PROFILE_SPEED) ||
	(limits->linear_order != LINEAR_ORDER_FIXED &&
	 limits->linear_order != LINEAR_ORDER_REFERENCE) ||
	(limits->draft_positions != 0 &&
	 (limits->draft_positions < MODEL_DRAFT_BLOCK ||
	  limits->draft_positions > STEP_DRAFT_MOST ||
	  limits->draft_positions > limits->row_capacity)))
	    return false;
    const uint32_t walk_count =
	    limits->draft_positions != 0 ? V41_DATA_WALK_LAYERS : MODEL_LAYERS;
    for (uint32_t layer = 0; layer < walk_count; layer++) {
        const uint32_t owner = tp2_program_layer_card(description, layer);
	if (owner >= canonical->card_count ||
	    description->stages[owner] == NULL ||
	    description->stages[owner]->description.rank != 0 ||
	    (v41_walk_layers(&canonical->cards[owner]) >> layer & 1u) == 0 ||
	    canonical->host_expert_count[layer] != 0)
		return false;
    }
    for (uint32_t card = 0; card < canonical->card_count; card++) {
        const uint32_t arithmetic = card_descriptions[card].arithmetic_class;
	if (cards[card] == NULL ||
	    (original_batch && card == V41_DATA_TP2_PRO_CARD &&
	     arithmetic != ARITHMETIC_CLASS_SM_120_NATIVE) ||
	    (exl3_all || card < V41_DATA_TP2_PRO_CARD ?
		     arithmetic != ARITHMETIC_CLASS_SM_80 :
		     arithmetic != ARITHMETIC_CLASS_SM_120 &&
			     arithmetic != ARITHMETIC_CLASS_SM_120_NATIVE) ||
	    !tp2_program_view_is_valid(description, card, rows)) {
		LOG_ERROR("tensor program: invalid card metadata at card %u",
			  card);
		return false;
	}
    }
    return true;
}

/* The tensor plan of the card's rank of layer: a target layer, or with draft_layers a draft stage (its channels behind
 * the 40 layers'); build->plan the card's view. */
static void
tp2_program_tensor_fill(const struct builder *build,
			const struct v41_tp2_program_description *description,
			struct v41_tp2_layer_plan *tensor, uint32_t card,
			uint32_t layer)
{
	const struct v41_tp2_stage_plan *const stage =
		description->stages[card];
	const uint32_t channel =
		(layer - stage->description.first_target_layer) * TP2_SITES;
	tensor->card = card;
	tensor->rank = stage->description.rank;
	tensor->rank_count = stage->description.rank_count;
	tensor->layer_index = layer;
	tensor->shared_channels = stage->channel_rows > LINEAR_MAXIMUM_ROWS;
	tensor->compact = stage->compact;
	tensor->exact_output_shards = stage->description.exact_output_shards;
	tensor->original_code_wire = stage->description.original_code_wire;
	tensor->original_wide_experts =
		stage->description.original_wide_experts;
	tensor->exl3_shard =
		build->plan->exl3_bits[layer] != 0 &&
				stage->description.exact_output_shards ?
			&stage->exl3_shards[layer] :
			NULL;
	tensor->bf16_ring = stage->description.rank_count == TP4_RANKS &&
			    stage->description.wire_format == ORDERED_PUSH_BF16;
	tensor->exact_hidden = stage->exact_hidden;
	tensor->exact_sums = stage->exact_sums;
	tensor->exact_dense = stage->description.exact_dense;
	tensor->exl3_fast_products = stage->description.exl3_fast_products;
	tensor->exl3_wide_entry = stage->description.exl3_wide_entry;
	tensor->exl3_tc_moe = stage->description.exl3_tc_moe;
	tensor->exl3_expert_rotation = stage->description.exl3_expert_rotation;
	tensor->trunk_shape_slices = stage->description.trunk_shape_slices;
	/* a BATCH stage's banks of up to 64 rows serve its BATCH programs; its wide and full programs complete as a THIN
     * stage's do (slab rings, the copy-engine sums, split_overlap, chunk_core: the same programs and bits) */
	tensor->ring_rows = stage->description.batch &&
					    build->made->program.kind !=
						    PROGRAM_BATCH ?
				    LINEAR_MAXIMUM_ROWS :
				    stage->ring_rows;
	tensor->exl3_tc_workspace = stage->exl3_tc_workspace;
	tensor->exl3_tcp_workspace = stage->exl3_tcp_workspace;
	tensor->exl3_tail_workspace = stage->exl3_tail_workspace;
	tensor->trunk_mma_partials = stage->trunk_mma_partials;
	tensor->trunk_mma_set_partials = stage->trunk_mma_set_partials;
	tensor->tail_sets = stage->description.tail_sets;
	tensor->router_partials = stage->description.router_mma ?
					  stage->router_partials :
					  (struct linear_range){ 0, 0 };
	tensor->ordered_wide = stage->description.ordered_wide;
	tensor->graph_branches = stage->description.graph_branches;
	tensor->bf16_slab_rings = stage->description.bf16_slab_rings;
	tensor->wide_exchange = stage->description.wide_exchange;
	/* full_split_overlap: the full programs' mode (split_overlap's views are the plan's, whatever the mode) */
	const bool full_mode = build->made->program.kind == PROGRAM_FULL &&
			       stage->description.split_overlap != 0 &&
			       stage->description.full_split_overlap != 0;
	tensor->split_overlap = full_mode ?
					stage->description.full_split_overlap :
					stage->description.split_overlap;
	tensor->split_least_rows = stage->description.split_least_rows;
	tensor->launches_untrimmed = stage->description.launches_untrimmed;
	tensor->chunk_core = stage->description.chunk_core;
	tensor->dense_input = stage->dense_input;
	tensor->index_scores = stage->index_scores;
	tensor->index_rows = stage->description.index_span ? stage->index_rows :
							     0u;
	for (uint32_t site = 0; site < TP2_SITES; site++) {
		tensor->ring_offsets[site] =
			stage->channels[channel + site].ring_offset;
		tensor->ring_bank_bytes[site] =
			stage->channels[channel + site].ring_bank_bytes;
		tensor->ring_row_bytes[site] =
			stage->channels[channel + site].ring_row_bytes;
		tensor->dense_ring_offsets[site] =
			stage->channels[channel + site].dense_ring_offset;
		tensor->dense_ring_bank_bytes[site] =
			stage->channels[channel + site].dense_ring_bank_bytes;
	}
	for (uint32_t site = 0; site < TP2_SITES; site++) {
		tensor->collectives[site] =
			stage->channels[channel + site].collective;
		/* each rank's region of the site's channel (the copy-engine owner sums' destinations) */
		for (uint32_t rank = 0;
		     rank < stage->description.rank_count && rank < TP4_RANKS;
		     rank++)
			tensor->peer_regions[site][rank] =
				description
					->stages[card -
						 stage->description.rank + rank]
					->channels[channel + site]
					.collective.arguments.region_offset;
	}
}

MUST_CHECK static result_status
tp2_program_tensor_kernels(struct builder *build, uint32_t card,
			   const struct v41_tp2_layer_plan *tensor);

MUST_CHECK static result_status
tp2_program_select(struct builder *build,
		   const struct v41_tp2_program_description *description,
		   struct v41_tp2_layer_plan *tensor, uint32_t card,
		   uint32_t layer)
{
	const bool sharded = layer < MODEL_DRAFT_FIRST_LAYER &&
			     description->stages[card]->description.rank_count >
				     1u;
	if (sharded && tensor == NULL)
		return STATUS_INVALID_ARGUMENT;
	build->plan = description->views[card];
	build->tensor_layer = sharded ? tensor : NULL;
	if (sharded)
		tp2_program_tensor_fill(build, description, tensor, card,
					layer);
	build->draft_layers =
		description->stages[card]->description.draft_layers &&
		tensor != NULL;
	build->tp_description = description;
	build->tp_tensor = tensor;
	result_status status = card_select(build, card);
	const struct v41_tp2_stage_plan *const own = description->stages[card];
	build->head_stage = own->description.head_shards ? own : NULL;
	build->head_stream_kernel = BACKEND_NO_KERNEL;
	static const char *const head_mma_wide_names[HEAD_MMA_WIDE_FORMS] = {
		"head_mma_64_s1", "head_mma_64_s2", "head_mma_64_s4"
	};
	build->head_mma_kernel = BACKEND_NO_KERNEL;
	for (uint32_t form = 0; form < HEAD_MMA_WIDE_FORMS; form++)
		build->head_mma_wide_kernels[form] = BACKEND_NO_KERNEL;
	build->trunk_mma_wide_kernel = build->trunk_mma_fused_kernel =
		BACKEND_NO_KERNEL;
	build->router_mma_kernel = build->router_mma_reduce_kernel =
		BACKEND_NO_KERNEL;
	build->router_mma_wide_kernel = BACKEND_NO_KERNEL;
	if (own->description.router_mma &&
	    own->router_partials.byte_count != 0) {
		status = THEN(build->backend->kernel_prepare(
			build->cards[card], HEAD_MMA_KERNEL_FILE,
			"router_mma_entry", sizeof(struct router_mma_arguments),
			&build->router_mma_kernel));
		status = THEN(build->backend->kernel_prepare(
			build->cards[card], HEAD_MMA_KERNEL_FILE,
			"router_mma_reduce",
			sizeof(struct router_mma_arguments),
			&build->router_mma_reduce_kernel));
		if (own->description.router_mma_wide)
			status = THEN(build->backend->kernel_prepare(
				build->cards[card], HEAD_MMA_KERNEL_FILE,
				"router_mma_wide_entry",
				sizeof(struct router_mma_arguments),
				&build->router_mma_wide_kernel));
	}
    if (own->description.trunk_mma && own->trunk_mma_partials.byte_count != 0)
	    status = THEN(build->backend->kernel_prepare(
		    build->cards[card], TRUNK_MMA_KERNEL_FILE,
		    TRUNK_MMA_WIDE_KERNEL,
		    TRUNK_RELAXED_BFLOAT16_ARGUMENT_BYTES,
		    &build->trunk_mma_wide_kernel));
    if (own->description.trunk_mma && own->description.trunk_mma_fused && own->trunk_mma_partials.byte_count != 0)
	    status = THEN(build->backend->kernel_prepare(
		    build->cards[card], TRUNK_MMA_KERNEL_FILE,
		    TRUNK_MMA_FUSED_KERNEL,
		    sizeof(struct trunk_mma_fused_arguments),
		    &build->trunk_mma_fused_kernel));
    build->trunk_mma = own->description.trunk_mma;
    build->trunk_mma_thin_splits = own->description.trunk_mma_thin_splits;
    build->batch_thin_bits = own->description.batch_thin_bits;
    build->batch_grouped_rows = own->description.batch_grouped_rows;
    build->batch_tc_paired = own->description.batch_tc_paired;
    build->batch_row_sets = own->description.batch_row_sets;
    build->batch_core_once = own->description.batch_core_once;
    build->front_branch = own->description.front_branch;
    build->core_heads = own->description.core_heads;
    build->draft_bf16_rows = own->description.draft_bf16_rows;
    build->batch_draft_wide = own->description.batch_draft_wide;
    build->ring_column_owners = own->description.ring_column_owners;
    for (uint32_t groups = 0; groups < TRUNK_MMA_GROUPED_FORMS; groups++)
        for (uint32_t form = 0; form < TRUNK_MMA_FORMS; form++)
		build->trunk_mma_grouped_kernels[groups][form] =
			BACKEND_NO_KERNEL;
    for (uint32_t form = 0; form <= TRUNK_MMA_GROUPED_FORMS; form++)
	    build->trunk_mma_pair_kernels[form] = BACKEND_NO_KERNEL;
    for (uint32_t form = 0; form < TRUNK_MMA_FORMS; form++)
	    build->trunk_mma_kernels[form] =
		    BACKEND_NO_KERNEL; /* prepared when first taken (trunk_mma_swap) */
    if (own->description.head_mma && (own->description.head_shards || card == description->canonical->result_card)) {
	    status = THEN(build->backend->kernel_prepare(
		    build->cards[card], HEAD_MMA_KERNEL_FILE, HEAD_MMA_KERNEL_8,
		    DENSE_ARGUMENT_BYTES, &build->head_mma_kernel));
	    for (uint32_t form = 0; form < HEAD_MMA_WIDE_FORMS; form++)
		    status = THEN(build->backend->kernel_prepare(
			    build->cards[card], HEAD_MMA_KERNEL_FILE,
			    head_mma_wide_names[form], DENSE_ARGUMENT_BYTES,
			    &build->head_mma_wide_kernels[form]));
    }
    if (own->description.head_shards)
	    status = THEN(build->backend->kernel_prepare(
		    build->cards[card], KERNEL_FILE_SAMPLER_SHARD,
		    KERNEL_SAMPLER_SHARD, SAMPLER_SHARD_ARGUMENT_BYTES,
		    &build->sampler_shard_kernel));
    if (own->description.head_shards && card == description->canonical->result_card)
	    status = THEN(build->backend->kernel_prepare(
		    build->cards[card], KERNEL_FILE_SAMPLER_SHARD,
		    KERNEL_SAMPLER_SETTINGS_PUSH,
		    SAMPLER_SETTINGS_PUSH_ARGUMENT_BYTES,
		    &build->settings_push_kernel));
    if (own->description.draft_shards) {
        const bool sends = card == description->canonical->result_card;
	status = THEN(build->backend->kernel_prepare(
		build->cards[card], KERNEL_FILE_DRAFT_SHARE,
		sends ? KERNEL_DRAFT_SHARE : KERNEL_DRAFT_SHARE_WAIT,
		sends ? DRAFT_SHARE_ARGUMENT_BYTES :
			DRAFT_SHARE_WAIT_ARGUMENT_BYTES,
		&build->draft_share_kernel));
	if (!sends) /* card_select prepares the draft's kernels on the result card alone: the Markov bias here */
		status = THEN(build->backend->kernel_prepare(
			build->cards[card], KERNEL_FILE_DRAFT,
			KERNEL_DRAFT_MARKOV_BIAS,
			DRAFT_MARKOV_BIAS_ARGUMENT_BYTES,
			&build->draft.markov_bias));
    }
    if (own->description.head_stream)
	    status = THEN(build->backend->kernel_prepare(
		    build->cards[card], KERNEL_FILE_DENSE_STREAM,
		    KERNEL_DENSE_STREAM, DENSE_ARGUMENT_BYTES,
		    &build->head_stream_kernel));
    if ((own->description.batch_draft_rows ||
	 getenv("DIAG_DRAFT_PROBE") != NULL) &&
	(card == description->canonical->result_card ||
	 own->description.draft_layers))
	    status = THEN(tp2_join_kernel_prepare(build->backend,
						  build->cards[card],
						  &build->batch_join_kernel));
    if (own->description.batch_wide_head &&
        (own->description.head_shards || card == description->canonical->result_card))
	    status = THEN(build->backend->kernel_prepare(
		    build->cards[card], KERNEL_FILE_DENSE_STREAM,
		    KERNEL_DENSE_STREAM_WIDE, DENSE_ARGUMENT_BYTES,
		    &build->head_wide_kernel));
    return sharded ? THEN(tp2_program_tensor_kernels(build, card, tensor)) : status;
}

/* The kernels of the card's rank of a tensor-parallel layer: its joins, rings and pushes. */
MUST_CHECK static result_status
tp2_program_tensor_kernels(struct builder *build, uint32_t card,
			   const struct v41_tp2_layer_plan *tensor)
{
	result_status status = tp2_join_kernel_prepare(
		build->backend, build->cards[card], &build->tensor_join_kernel);
	if (tensor->exact_output_shards || tensor->bf16_ring)
		status = THEN(build->backend->kernel_prepare(
			build->cards[card], KERNEL_FILE_EXACT_RING,
			KERNEL_EXACT_RING, EXACT_RING_ARGUMENT_BYTES,
			&build->exact_ring_kernel));
	if (tensor->exl3_shard != NULL)
		status = THEN(build->backend->kernel_prepare(
			build->cards[card], KERNEL_FILE_EXACT_RING,
			KERNEL_EXACT_RING_UNEQUAL,
			EXACT_RING_UNEQUAL_ARGUMENT_BYTES,
			&build->unequal_ring_kernel));
	if (tensor->exact_output_shards && tensor->original_code_wire) {
		status = THEN(build->backend->kernel_prepare(
			build->cards[card], KERNEL_FILE_QUANTIZED_WIRE,
			KERNEL_QUANTIZE_WIRE,
			QUANTIZED_WIRE_PRODUCER_ARGUMENT_BYTES,
			&build->wire_quantize_kernel));
		status = THEN(build->backend->kernel_prepare(
			build->cards[card], KERNEL_FILE_QUANTIZED_WIRE,
			KERNEL_QUANTIZED_WIRE, QUANTIZED_WIRE_ARGUMENT_BYTES,
			&build->wire_decode_kernel));
	}
	status = tensor->shared_channels ?
			 THEN(tp_exchange_kernel_prepare(
				 build->backend, build->cards[card],
				 &build->tensor_push_kernel)) :
			 THEN(ordered_push_kernel_prepare(
				 build->backend, build->cards[card],
				 &build->tensor_push_kernel));
	if (tensor->shared_channels && tensor->wide_exchange != 0)
		status = THEN(tp_exchange_wide_kernels_prepare(
			build->backend, build->cards[card],
			TP_EXCHANGE_FORM_PUSH, build->tensor_exchange_kernels));
	if (tensor->shared_channels &&
	    tensor->wide_exchange >= V41_DATA_TP_EXCHANGE_OWNER_SUMS)
		status = THEN(tp_exchange_wide_kernels_prepare(
			build->backend, build->cards[card],
			TP_EXCHANGE_FORM_OWNER_SUMS,
			build->tensor_owner_kernels));
	if (tensor->shared_channels &&
	    tensor->wide_exchange >= V41_DATA_TP_EXCHANGE_CE_OWNER_SUMS)
		status = THEN(tp_exchange_wide_kernels_prepare(
			build->backend, build->cards[card],
			TP_EXCHANGE_FORM_CE_OWNER_SUMS,
			build->tensor_ce_kernels));
	if (tensor->shared_channels &&
	    tensor->wide_exchange == V41_DATA_TP_EXCHANGE_CE_RINGS)
		status = THEN(tp_exchange_wide_kernels_prepare(
			build->backend, build->cards[card],
			TP_EXCHANGE_FORM_CE_OWNER_RING_SUMS,
			build->tensor_ce_ring_kernels));
	if (tensor->shared_channels &&
	    tensor->wide_exchange == V41_DATA_TP_EXCHANGE_OWNER_RINGS)
		status = THEN(tp_exchange_wide_kernels_prepare(
			build->backend, build->cards[card],
			TP_EXCHANGE_FORM_OWNER_RING_SUMS,
			build->tensor_ring_kernels));
	return status;
}

/* The card's rank of layer, its tensor plan and kernels, for the launches that follow: draft_layers' draft stage
 * layers, split_overlap's pipeline the next target layer's first half. */
MUST_CHECK static result_status tp2_layer_select(struct builder *b,
						 uint32_t layer)
{
    if (b->tp_description == NULL || b->tp_tensor == NULL)
	    return STATUS_INVALID_ARGUMENT;
    tp2_program_tensor_fill(b, b->tp_description, b->tp_tensor, b->card, layer);
    b->tensor_layer = b->tp_tensor;
    return tp2_program_tensor_kernels(b, b->card, b->tp_tensor);
}

MUST_CHECK static result_status tp2_program_segment(struct builder *build,
						    uint32_t layer,
						    uint32_t first_launch)
{
    const uint32_t before = build->made->program.segment_count;
    segment_add(build, layer, first_launch);
    return build->made->program.segment_count == before + 1u ?
		   OK :
		   STATUS_INVALID_ARGUMENT;
}

MUST_CHECK static result_status
tp2_program_hand_off(struct builder *build, uint32_t destination,
		     uint64_t source_offset, uint64_t destination_offset,
		     uint32_t bytes, uint32_t ratio, uint64_t stride)
{
	if (build->made->program.hand_off_count >= MOST_HAND_OFFS ||
	    build->made->program.segment_count == 0 ||
	    source_offset == PROGRAM_NOWHERE ||
	    destination_offset == PROGRAM_NOWHERE)
		return STATUS_INVALID_ARGUMENT;
	hand_off(build, destination, source_offset, destination_offset, bytes,
		 ratio, stride);
	return OK;
}

/* Copies from the canonical producer precede its adjacent companion in the segment table. Peers which run that
 * source/index layer compute the same values locally and must not receive a handoff inside their companion pair. */
MUST_CHECK static result_status
tp2_program_copies(struct builder *build, uint32_t layer, uint32_t copied)
{
	const struct model_plan_record *const plan = build->plan;
	const struct v41_card *const from = &plan->cards[build->card];
	const bool source = (V41_DATA_SOURCE_LAYERS >> layer & 1u) != 0;
	const uint32_t ratio = (V41_DATA_POOLED_LAYERS >> layer & 1u) != 0 ?
				       MODEL_POOLED_RATIO :
				       MODEL_DIRECT_RATIO;
	const bool writes[COPIED_ARRAYS] = {
		(V41_DATA_INDEX_LAYERS >> layer & 1u) != 0, source, source,
		layer == MODEL_CANDIDATE_SOURCE
	};
	const uint64_t sources[COPIED_ARRAYS] = {
		from->selection_offset[layer], from->pool_offset[layer],
		from->key_tiles_offset[layer],
		from->array_offset[V41_DATA_ARRAY_CANDIDATES]
	};
	const uint64_t strides[COPIED_ARRAYS] = {
		0, plan->pool_session_bytes[layer],
		plan->key_session_bytes[layer], 0
	};
	result_status status = OK;
	for (uint32_t destination = 0;
	     destination < plan->card_count && status == OK; destination++) {
		const struct v41_card *const other = &plan->cards[destination];
		if (destination == build->card ||
		    (other->layers >> layer & 1u) != 0)
			continue;
		const uint64_t targets[COPIED_ARRAYS] = {
			other->selection_offset[layer],
			other->pool_offset[layer],
			other->key_tiles_offset[layer],
			other->array_offset[V41_DATA_ARRAY_CANDIDATES]
		};
		for (uint32_t array = 0; array < COPIED_ARRAYS && status == OK;
		     array++) {
			if (writes[array] && (copied >> array & 1u) != 0 &&
			    targets[array] != PROGRAM_NOWHERE)
				status = tp2_program_hand_off(
					build, destination, sources[array],
					targets[array], copied_bytes[array],
					strides[array] == 0 ? 0u : ratio,
					strides[array]);
		}
    }
    return status;
}

MUST_CHECK static result_status tp2_program_streams(struct builder *build,
						    uint32_t destination)
{
    const uint64_t *const from = build->plan->cards[build->card].array_offset;
    const uint64_t *const to = build->plan->cards[destination].array_offset;
    result_status status = tp2_program_hand_off(build, destination,
						from[V41_DATA_ARRAY_STREAMS],
						to[V41_DATA_ARRAY_STREAMS],
						MODEL_STREAMS_ROW_BYTES, 0, 0);
    return THEN(tp2_program_hand_off(
	    build, destination, from[V41_DATA_ARRAY_CARRIED_PRE],
	    to[V41_DATA_ARRAY_CARRIED_PRE], MODEL_SITE_ROW_BYTES, 0, 0));
}

MUST_CHECK static result_status
tp2_program_pair_add(struct builder *build, struct step_companion_pair *pairs,
		     uint32_t *pair_count, uint32_t first_segment)
{
    const uint32_t members = build->made->program.segment_count - first_segment;
    if (*pair_count >= TP2_PROGRAM_PAIRS ||
	(members != TP2_RANKS && members != TP4_RANKS))
	    return STATUS_INVALID_ARGUMENT;
    pairs[(*pair_count)++] =
	    (struct step_companion_pair){ first_segment,
					  first_segment + members - 1u };
    return OK;
}

MUST_CHECK static result_status
tp2_program_rotations(struct builder *build,
		      const struct v41_tp2_program_description *description,
		      struct v41_tp2_layer_plan *tensor,
		      struct step_companion_pair *pairs, uint32_t *pair_count,
		      uint32_t first_card, uint32_t first_layer, bool embeds)
{
    const uint32_t first_segment = build->made->program.segment_count;
    const uint32_t ranks = description->stages[first_card]->description.rank_count;
    result_status status = OK;
    for (uint32_t rank = 0; rank < ranks && status == OK; rank++) {
        status = tp2_program_select(build, description, tensor, first_card + rank, first_layer);
        const uint32_t first_launch = build->made->cards[build->card].launch_count;
        if (embeds) { /* a replicated embedding: every rank expands its own rows (the settings push first) */
		const struct v41_tp2_layer_plan *const held =
			build->tensor_layer;
		build->tensor_layer = NULL;
		const enum program_kind kind = build->made->program.kind;
		if (status == OK && build->head_stage != NULL &&
		    build->card == description->canonical->result_card &&
		    (kind == PROGRAM_THIN || kind == PROGRAM_BATCH))
			status = settings_push_launch(build);
		status = THEN(step_launches(build, true));
		build->tensor_layer = held;
        }
        for (uint32_t slab = 0; slab < build->slab_count && status == OK; slab++) {
            if (build->slab_count > 1u)
                slab_select(build, slab);
            status = step_launches(build, false);
        }
	status = THEN(
		tp2_program_segment(build, SEGMENT_NO_LAYER, first_launch));
    }
    return ranks > 1u ? THEN(tp2_program_pair_add(build, pairs, pair_count, first_segment)) : status;
}

MUST_CHECK static result_status
tp2_program_boundary(struct builder *build,
		     const struct v41_tp2_program_description *description,
		     uint32_t layer)
{
	if (layer + 1u == MODEL_LAYERS)
		return OK;
	const uint32_t owner = tp2_program_layer_card(description, layer);
	const uint32_t next = tp2_program_layer_card(description, layer + 1u);
	result_status status = OK;
	if (owner != next) {
		for (uint32_t rank = 0;
		     rank < description->stages[next]->description.rank_count &&
		     status == OK;
		     rank++)
			status = tp2_program_streams(build, next + rank);
	}
    return status;
}

/* split_overlap (model_v41_tp2_stage.h) applies to the card's layer: the BF16 recipe's EXL3 tensor layer of a wide or
 * full program with half views, its attention summed by the wide exchange and its FFN a slab ring at a time. */
static bool split_applies(const struct builder *b, uint32_t layer)
{
	const struct v41_tp2_layer_plan *const tensor = b->tensor_layer;
	return tensor != NULL && tensor->split_overlap &&
	       b->half_view_first != 0 && b->chunk_first == 0 &&
	       tensor->shared_channels && tensor->bf16_ring &&
	       !tensor->exact_output_shards && !tensor->exact_dense &&
	       tensor->exl3_wide_entry && tensor->wide_exchange != 0 &&
	       tensor->wide_exchange != V41_DATA_TP_EXCHANGE_OWNER_RINGS &&
	       !tensor->bf16_slab_rings &&
	       tensor->ring_rows <= LINEAR_MAXIMUM_ROWS &&
	       b->plan->exl3_bits[layer] != 0 && !b->vision_active &&
	       b->made->program.kind != PROGRAM_BATCH &&
	       b->half_rows > EXL3_TC_MOST_ROWS && !tail_program(b) &&
	       b->chunk_rows >= tensor->split_least_rows &&
	       b->chunk_rows - b->half_rows <=
		       LINEAR_WIDE_MOST_ROWS; /* each half a wide launch */
}

/* Half h of the chunk (rows, slabs, view) for chunk_select, piece_select and slab_select; h == 2: the whole again. */
static void half_select(struct builder *b, uint32_t half, uint32_t chunk_rows)
{
    b->chunk_first = half == 1u ? b->half_rows : 0u;
    b->chunk_rows = half == 0 ? b->half_rows : half == 1u ? chunk_rows - b->half_rows : chunk_rows;
    b->chunk_view = half < 2u ? b->half_view_first + half : 0u;
    b->slab_count =
	    (b->chunk_rows + LINEAR_MAXIMUM_ROWS - 1u) / LINEAR_MAXIMUM_ROWS;
    chunk_select(b);
}

/* split_overlap's pipeline (model_v41_tp2_stage.h) applies to the card's layer: split_layer takes its attention too
 * (spans of whole half views). */
static bool pipelined(const struct builder *b, uint32_t layer)
{
	return split_applies(b, layer) &&
	       b->tensor_layer->split_overlap >= V41_DATA_SPLIT_PIPELINE &&
	       layer != MODEL_DECODER_FIRST_LAYER &&
	       b->half_rows % V41_DATA_SPAN_ROWS == 0;
}

/* Exact original attention deliberately clears tensor_layer: its products and query/core arrays retain all
 * 64 heads. Consult the stage only for the opt-in, single-piece core schedule; no shard or arithmetic is selected.
 * Original BATCH models' FULL prefills stay excluded too, as do the separate wide-expert and owner experiments. */
static bool original_chunk_core_applies(const struct builder *b)
{
	if (b->tensor_layer != NULL || b->tp_description == NULL ||
	    b->tp_tensor == NULL || b->card >= TP4_RANKS ||
	    b->arithmetic_class != ARITHMETIC_CLASS_SM_80 ||
	    b->slab_count <= 1u || piece_count_of(b) != 1u ||
	    b->chunk_rows % LINEAR_MAXIMUM_ROWS != 0 || b->vision_active ||
	    b->made->program.kind == PROGRAM_BATCH ||
	    b->tp_description->stages[b->card] == NULL)
		return false;
	const struct v41_tp2_stage_description *const stage =
		&b->tp_description->stages[b->card]->description;
	const struct model_limits *const limits = &stage->limits;
	return stage->chunk_core && stage->exact_output_shards &&
	       stage->wire_format == ORDERED_PUSH_FP32 &&
	       !stage->original_code_wire && !stage->exl3_all_targets &&
	       !stage->exact_dense && !stage->attention_owner &&
	       !stage->original_batch && !stage->batch &&
	       !stage->original_wide_experts && stage->split_overlap == 0 &&
	       stage->full_split_overlap == 0 &&
	       stage->rank_count == TP4_RANKS &&
	       stage->first_target_layer == 0 &&
	       stage->target_layer_count == TP4_ATTENTION_PREFIX_END &&
	       limits->state_width == STATE_WIDTH_16 &&
	       limits->linear_profile == LINEAR_PROFILE_PRECISION &&
	       limits->linear_order == LINEAR_ORDER_REFERENCE &&
	       limits->cmp_order == CMP_ORDER_REFERENCE &&
	       b->tp_tensor->layer_index < TP4_ATTENTION_PREFIX_END &&
	       b->plan->exl3_bits[b->tp_tensor->layer_index] == 0;
}

/* chunk_core (model_v41_tp2_stage.h) applies to the selected chunk or half: its slabs' scans, then one core. */
static bool chunk_core_applies(const struct builder *b)
{
	return ((b->tensor_layer != NULL && b->tensor_layer->chunk_core) ||
		original_chunk_core_applies(b)) &&
	       piece_count_of(b) == 1u &&
	       b->made->program.kind != PROGRAM_BATCH;
}

/* queries_piece (model_v41_tp2_stage.h): a wide program's queries with wq_b over each piece, not a span at a time. */
static bool queries_piece_applies(const struct builder *b)
{
    return b->tensor_layer != NULL && b->tp_description != NULL &&
           b->tp_description->stages[b->card]->description.queries_piece && !thin_forms_of(b);
}

/* A gather's part of units (slabs, rows) the rank computes: its first and count. A quarter each; gather_balance: the
 * ranks of the narrower expert slice (their shared expert's inner rows, the routed experts' cut alike) in equal parts,
 * the last of them the rest, the wider ranks none (they set the pace at every exchange); _WEIGHTED: a wider rank
 * V41_DATA_GATHER_WIDE_SIXTEENTHS of the units, the narrower ones the rest alike, each part whole granules (rows: a
 * dense launch's 8-row groups), the ranks' parts in rank order. */
static void gather_share(const struct builder *b, uint32_t units, uint32_t granule, uint32_t *first_out,
                         uint32_t *count_out)
{
	const struct v41_tp2_layer_plan *const tensor = b->tensor_layer;
	*first_out = tensor->rank * (units / TP4_RANKS);
	*count_out = units / TP4_RANKS;
	const uint32_t balance =
		b->tp_description->stages[b->card]->description.gather_balance;
	if (balance == 0 || (balance == V41_DATA_GATHER_BALANCE_WIDE &&
			     b->made->program.kind != PROGRAM_WIDE))
		return;
	uint32_t widths[TP4_RANKS], narrowest = UINT32_MAX, narrow = 0,
				    before = 0;
	for (uint32_t rank = 0; rank < TP4_RANKS; rank++) {
		const uint32_t card = tensor->card - tensor->rank + rank;
		widths[rank] = b->tp_description->views[card]
				       ->layer_tensors[tensor->layer_index]
						      [V41_DATA_SHARED_GATE]
				       .shape.output_size;
		narrowest = widths[rank] < narrowest ? widths[rank] : narrowest;
	}
	for (uint32_t rank = 0; rank < TP4_RANKS; rank++) {
		before += rank < tensor->rank && widths[rank] == narrowest ?
				  1u :
				  0u;
		narrow += widths[rank] == narrowest ? 1u : 0u;
	}
	if (balance == V41_DATA_GATHER_BALANCE_WEIGHTED) {
		const uint32_t wide_part = units / granule *
					   V41_DATA_GATHER_WIDE_SIXTEENTHS /
					   V41_DATA_GATHER_SIXTEENTHS * granule;
		const uint32_t narrow_units =
			units - (TP4_RANKS - narrow) * wide_part;
		const uint32_t narrow_part =
			narrow_units / narrow / granule * granule;
		uint32_t first = 0, narrow_seen = 0;
		for (uint32_t rank = 0; rank <= tensor->rank; rank++) {
			narrow_seen += widths[rank] == narrowest ? 1u : 0u;
			const uint32_t count =
				widths[rank] != narrowest ?
					wide_part :
				narrow_seen == narrow ?
					narrow_units -
						(narrow - 1u) * narrow_part :
					narrow_part;
			*first_out = first;
			*count_out = count;
			first += count;
		}
		return;
	}
    const uint32_t part = units / narrow;
    *first_out = before * part;
    *count_out = widths[tensor->rank] != narrowest ? 0u : before + 1u == narrow ? units - before * part : part;
}

/* index_span (model_v41_tp2_stage.h) applies to the selected chunk or half of a wide program of pool_chunk (no slab's
 * state writes between its scans) with the stage's span scores. */
static bool index_span_applies(const struct builder *b)
{
	return b->tensor_layer != NULL &&
	       b->tensor_layer->index_rows > LINEAR_MAXIMUM_ROWS &&
	       b->slab_count > 1u && !thin_forms_of(b) &&
	       b->tp_description != NULL &&
	       b->tp_description->stages[b->card]->description.pool_chunk;
}

/* The scans (CORE's index scan and selection) of the selected chunk's or half's slabs first_slab to first_slab + slabs
 * - 1: each a slab at a time through its view, or (index_span) up to index_rows rows a launch through the chunk's view
 * from their first row on (index_scan.h's step_row_first), each row's selection by a slab's kernels (split_rows). */
MUST_CHECK static result_status index_scans(struct builder *b, uint32_t layer,
					    uint32_t source, uint32_t group,
					    uint32_t first_slab, uint32_t slabs)
{
    const bool spans = index_span_applies(b);
    const uint32_t step =
	    spans ? b->tensor_layer->index_rows / LINEAR_MAXIMUM_ROWS : 1u;
    result_status status = OK;
    for (uint32_t slab = first_slab; slab < first_slab + slabs && status == OK; slab += step) {
        if (!spans) {
            slab_select(b, slab);
        } else {
		const uint32_t first = slab * LINEAR_MAXIMUM_ROWS,
			       end = first_slab + slabs < slab + step ?
					     (first_slab + slabs) *
						     LINEAR_MAXIMUM_ROWS :
					     first + step * LINEAR_MAXIMUM_ROWS;
		const uint32_t last = end < b->chunk_rows ? end : b->chunk_rows;
		rows_select(b, b->chunk_view, b->chunk_first + first,
			    last - first);
		/* the view's block and its rows up to the last one (rows_select's view 0 lies first rows in) */
		b->arrays[V41_DATA_ARRAY_STEP] = (struct linear_range){
			b->chunk_view == 0 ?
				b->plan->cards[b->card]
					.array_offset[V41_DATA_ARRAY_STEP] :
				b->arrays[V41_DATA_ARRAY_STEP].byte_offset,
			STEP_BLOCK_BYTES_VALUE +
				(uint64_t)last * STEP_ROW_BYTES_VALUE
		};
		b->index_row_first = first;
		b->index_grouped = true;
        }
        status = layer_steps(b, layer, source, group, SCAN);
        b->index_grouped = false;
    }
    return status;
}

/* index_gather applies to the selected half of an index layer: the attention site's sums by copy engines (direct
 * routes, four ranks, BF16) and whole quarters of slabs. */
static bool index_gather_applies(const struct builder *b, uint32_t layer, uint32_t group)
{
    return group == layer && layer != 0 && gathers_apply(b);
}

/* split_overlap's pipeline, an index layer's half behind every slab's pool writes: the rank's quarter of its slabs
 * scanned (each row's selection by the scan every rank runs), those selection rows gathered at GATHER_INDEX_PHASE
 * (half A on the attention site's channel, beside it the FFN sums of the layer before; half B on the FFN site's,
 * beside it half A's attention sums): every rank holds every row's selection for a quarter of the scans. */
MUST_CHECK static result_status index_gather(struct builder *b, uint32_t layer,
					     uint32_t source, uint32_t group)
{
	const struct v41_tp2_layer_plan *const tensor = b->tensor_layer;
	uint32_t slabs = 0, first_slab = 0;
	gather_share(b, b->slab_count, 1u, &first_slab, &slabs);
	result_status status =
		index_scans(b, layer, source, group, first_slab, slabs);
	chunk_select(b);
	const uint64_t into = ((uint64_t)b->chunk_first +
			       (uint64_t)first_slab * LINEAR_MAXIMUM_ROWS) *
			      MODEL_SELECTION_ROW_BYTES;
	const uint64_t bytes = (uint64_t)slabs * LINEAR_MAXIMUM_ROWS *
			       MODEL_SELECTION_ROW_BYTES;
	struct gather_range range = {
		b->plan->cards[b->card].selection_offset[layer] + into,
		bytes,
		{ 0 }
	};
	for (uint32_t rank = 0; rank < TP4_RANKS; rank++) {
		const uint32_t card = tensor->card - tensor->rank + rank;
		range.peers[rank] = b->tp_description->views[card]
					    ->cards[card]
					    .selection_offset[layer] +
				    into;
	}
	return THEN(rows_gather(
		b, b->chunk_view == b->half_view_first + 1u ? 1u : 0u,
		TP_EXCHANGE_LAYER_PHASES * tensor->layer_index +
			GATHER_INDEX_PHASE,
		&range, 1u));
}

MUST_CHECK static result_status router_gather(struct builder *b,
					      const struct v41_tensor *router)
{
	const struct v41_tp2_layer_plan *const tensor = b->tensor_layer;
	const uint32_t view = b->view, first = b->first_row, rows = b->rows;
	uint32_t own_first = 0,
		 quarter =
			 0; /* the rank's rows: a quarter, or gather_balance's part */
	gather_share(b, rows, LINEAR_MAXIMUM_ROWS, &own_first, &quarter);
	rows_select(b, view, first + own_first, quarter);
	result_status status = quarter != 0 ? router_scores(b, router) : OK;
	rows_select(b, view, first, rows);
	const uint64_t into =
		(uint64_t)(first + own_first) * MODEL_ROUTER_ROW_BYTES;
	struct gather_range range = {
		b->plan->cards[b->card]
				.array_offset[V41_DATA_ARRAY_ROUTER_SCORES] +
			into,
		(uint64_t)quarter * MODEL_ROUTER_ROW_BYTES,
		{ 0 }
	};
	for (uint32_t rank = 0; rank < TP4_RANKS; rank++) {
		const uint32_t card = tensor->card - tensor->rank + rank;
		range.peers[rank] =
			b->tp_description->views[card]
				->cards[card]
				.array_offset[V41_DATA_ARRAY_ROUTER_SCORES] +
			into;
	}
    const bool second = b->chunk_view == b->half_view_first + 1u;
    return THEN(rows_gather(b, second ? 0u : 1u,
			    TP_EXCHANGE_LAYER_PHASES * tensor->layer_index +
				    (second ? GATHER_ROUTER_SECOND_PHASE :
					      GATHER_ROUTER_FIRST_PHASE),
			    &range, 1u));
}

/* The selected chunk's (split_overlap's pipeline: half's) attention to its wo_a and wo_b: its pieces' products, its
 * slabs' state writes and scans or cores in slab order; chunk_core: the window writes over its rows ahead of the slabs'
 * pool writes and scans (an index layer on copy-engine sums: index_gather's quarter and gather), then the core. */
MUST_CHECK static result_status attention_half(struct builder *b,
					       uint32_t layer, uint32_t source,
					       uint32_t group)
{
    /* prefill_dead_halves: a dead half's index scan and gather left out (its core reads no selection: the cone) */
    const bool dead = half_dead(b, layer, selected_half(b));
    const bool chunk_core = chunk_core_applies(b);
    const bool gather = chunk_core && !dead && index_gather_applies(b, layer, group);
    const bool pool_chunk = chunk_core && b->tp_description->stages[b->card]->description.pool_chunk;
    const uint32_t piece_queries = queries_piece_applies(b) ? QUERIES : 0u, span_queries = piece_queries ^ QUERIES;
    /* index_span: the slabs' scans after the slab loop, a span at a time (index_scans) */
    const bool scans = chunk_core && !gather && !dead, spans = scans && index_span_applies(b);
    result_status status = steps_over(
	    b, layer, source, group,
	    AHEAD_STEPS | QUERY_NORM | QUERY_STEPS | piece_queries, true);
    chunk_select(b);
    status = chunk_core ? THEN(layer_steps(b, layer, source, group, WINDOW_WRITES)) : status;
    status = pool_chunk ? THEN(steps_over(b, layer, source, group, POOL_CHUNK, true)) : status;
    status = THEN(steps_over(b, layer, source, group, !chunk_core ? STATE_WRITES | span_queries | CORE :
                                                      (pool_chunk ? 0u : STATE_POOL) | span_queries |
                                                          (scans && !spans ? SCAN : 0u),
                             false));
    status = spans ? THEN(index_scans(b, layer, source, group, 0, b->slab_count)) : status;
    status = gather ? THEN(index_gather(b, layer, source, group)) : status;
    chunk_select(b);
    status = chunk_core ? THEN(layer_steps(b, layer, source, group, CORE_ALONE)) : status;
    return THEN(layer_steps(b, layer, source, group, OUTPUT_PRODUCTS));
}

#define HALF_ATTENTION UINT32_MAX /* split_part's steps: attention_half's */


/* One part of split_layer: half h's launches of steps, or of the FFN's phases ffn; its first marked branch (a side
 * run's every one). */
MUST_CHECK static result_status split_part(struct builder *b, uint32_t layer,
					   uint32_t source, uint32_t group,
					   uint32_t half, uint32_t chunk_rows,
					   uint32_t steps, uint32_t ffn,
					   uint32_t branch)
{
    const uint32_t first = b->made->cards[b->card].launch_count;
    half_select(b, half, chunk_rows);
    result_status status = ffn != 0 ?
				   exl3_wide_ffn_phases(b, layer, ffn) :
			   steps == HALF_ATTENTION ?
				   attention_half(b, layer, source, group) :
				   layer_steps(b, layer, source, group, steps);
    const uint32_t end = b->made->cards[b->card].launch_count;
    if (status == OK && end == first)
	    status =
		    STATUS_INVALID_ARGUMENT; /* every part has launches: a branch never marks nothing */
    for (uint32_t launch = first;
	 launch < end && status == OK && branch != LAUNCH_CHAIN; launch++)
	    if (launch == first || branch == LAUNCH_SIDE)
		    b->launches[b->card][launch].branch = branch;
    return status;
}

/* split_overlap: the layer from wo_a on in two halves A and B, each through its half view, the sums as graph side runs
 * (backend_seam.h's branches): wo(A); sum(A) beside wo(B); post(A); sum(B) beside FFN(A); post(B); FFN sums(A)
 * beside FFN(B); FFN post(A); FFN sums(B) and post(B). Its pipeline: the attention in halves, A's (unless the layer
 * before took it) and B's in place of wo(A) and wo(B); FFN sums(B) beside the next layer's attention of A when that
 * layer is pipelined too (within the card's segment: a graph's side runs end in it), then post(B). Every row's
 * launches are the whole chunk's: the same bits. */
MUST_CHECK static result_status split_layer(struct builder *b, uint32_t layer,
					    uint32_t source, uint32_t group)
{
    /* split_overlap's modes: the attention's runs (parts 1 to 6), the FFN's (7 to 9), both or neither */
    const uint32_t mode = b->tensor_layer->split_overlap;
    struct split_step { uint32_t half, steps, ffn, branch; };
    static const struct split_step halves_parts[] = {
	    { 0, OUTPUT_PRODUCTS, 0, LAUNCH_CHAIN },
	    { 0, SITE_SUM, 0, LAUNCH_SIDE },
	    { 1, OUTPUT_PRODUCTS, 0, LAUNCH_RESUME },
	    { 0, SITE_POST_ONLY, 0, LAUNCH_JOIN },
	    { 1, SITE_SUM, 0, LAUNCH_SIDE },
	    { 0, 0, FFN_COMPUTE, LAUNCH_RESUME },
	    { 1, SITE_POST_ONLY, 0, LAUNCH_JOIN },
	    { 0, 0, FFN_SUM, LAUNCH_SIDE },
	    { 1, 0, FFN_COMPUTE, LAUNCH_RESUME },
	    { 0, 0, FFN_POST, LAUNCH_JOIN },
	    { 1, 0, FFN_SUM | FFN_POST, LAUNCH_CHAIN }
    };
    /* the joint mode: A's front and shared expert beside B's attention sums, B's front, the routed experts of the
     * whole chunk (half 2) in one call, A's FFN sums beside B's shared expert */
    static const struct split_step joint_parts[] = {
	    { 0, OUTPUT_PRODUCTS, 0, LAUNCH_CHAIN },
	    { 0, SITE_SUM, 0, LAUNCH_SIDE },
	    { 1, OUTPUT_PRODUCTS, 0, LAUNCH_RESUME },
	    { 0, SITE_POST_ONLY, 0, LAUNCH_JOIN },
	    { 1, SITE_SUM, 0, LAUNCH_SIDE },
	    { 0, 0, FFN_FRONT | FFN_SHARED, LAUNCH_RESUME },
	    { 1, SITE_POST_ONLY, 0, LAUNCH_JOIN },
	    { 1, 0, FFN_FRONT, LAUNCH_CHAIN },
	    { 2, 0, FFN_EXPERTS, LAUNCH_CHAIN },
	    { 0, 0, FFN_SUM, LAUNCH_SIDE },
	    { 1, 0, FFN_SHARED, LAUNCH_RESUME },
	    { 0, 0, FFN_POST, LAUNCH_JOIN },
	    { 1, 0, FFN_SUM | FFN_POST, LAUNCH_CHAIN }
    };
    /* mode 7: the joint call's route and gate/up over the chunk, A's down GEMM and combine, A's FFN sums beside B's
     * down GEMM, combine and shared expert */
    static const struct split_step joint_down_parts[] = {
	    { 0, OUTPUT_PRODUCTS, 0, LAUNCH_CHAIN },
	    { 0, SITE_SUM, 0, LAUNCH_SIDE },
	    { 1, OUTPUT_PRODUCTS, 0, LAUNCH_RESUME },
	    { 0, SITE_POST_ONLY, 0, LAUNCH_JOIN },
	    { 1, SITE_SUM, 0, LAUNCH_SIDE },
	    { 0, 0, FFN_FRONT | FFN_SHARED, LAUNCH_RESUME },
	    { 1, SITE_POST_ONLY, 0, LAUNCH_JOIN },
	    { 1, 0, FFN_FRONT, LAUNCH_CHAIN },
	    { 2, 0, FFN_EXPERTS_FRONT, LAUNCH_CHAIN },
	    { 2, 0, FFN_DOWN_FIRST, LAUNCH_CHAIN },
	    { 0, 0, FFN_SUM, LAUNCH_SIDE },
	    { 2, 0, FFN_DOWN_SECOND, LAUNCH_RESUME },
	    { 1, 0, FFN_SHARED, LAUNCH_CHAIN },
	    { 0, 0, FFN_POST, LAUNCH_JOIN },
	    { 1, 0, FFN_SUM | FFN_POST, LAUNCH_CHAIN }
    };
    const bool joint = mode == V41_DATA_SPLIT_JOINT,
	       joint_down = mode == V41_DATA_SPLIT_JOINT_DOWN;
    const struct split_step *const parts = joint_down ? joint_down_parts : joint ? joint_parts : halves_parts;
    const uint32_t last = (joint_down ? sizeof(joint_down_parts) / sizeof(joint_down_parts[0]) :
                           joint      ? sizeof(joint_parts) / sizeof(joint_parts[0]) :
                                        sizeof(halves_parts) / sizeof(halves_parts[0])) - 1u;
    const uint32_t chunk_rows = b->chunk_rows, next = layer + 1u;
    const bool pipeline = pipelined(b, layer); /* else, as layer 20 (its attention divided), mode 1 */
    const bool ahead = pipeline && next < MODEL_DRAFT_FIRST_TAP_LAYER &&
		       pipelined(b, next) &&
		       tp2_program_layer_card(b->tp_description, next) ==
			       tp2_program_layer_card(b->tp_description, layer);
    /* prefill_dead_halves: a dead half's sums left out (half_dead), their side runs' resume and join then the chain */
    const bool dead[2] = {half_dead(b, layer, 0u), half_dead(b, layer, 1u)};
    bool unforked = false;
    result_status status = OK;
    for (uint32_t part = pipeline && b->ahead_layer[b->card] == next ? 1u : 0u; part < last && status == OK; part++) {
	    const bool marks = mode == V41_DATA_SPLIT_OVERLAP ||
			       mode >= V41_DATA_SPLIT_PIPELINE ||
			       (mode == V41_DATA_SPLIT_ATTENTION_ONLY &&
				part >= 1u && part <= 6u) ||
			       (mode == V41_DATA_SPLIT_FFN_ONLY && part >= 7u &&
				part <= 9u);
	    const uint32_t half = parts[part].half;
	    if (half < 2u && dead[half] &&
		(parts[part].steps == SITE_SUM || parts[part].ffn == FFN_SUM)) {
		    unforked = unforked ||
			       (marks && parts[part].branch == LAUNCH_SIDE);
		    continue;
	    }
	    uint32_t branch = marks ? parts[part].branch : LAUNCH_CHAIN;
	    if (unforked &&
		(branch == LAUNCH_RESUME || branch == LAUNCH_JOIN)) {
		    unforked = branch != LAUNCH_JOIN;
		    branch = LAUNCH_CHAIN;
	    }
	status = split_part(b, layer, source, group, half, chunk_rows,
                            pipeline && parts[part].steps == OUTPUT_PRODUCTS ? HALF_ATTENTION : parts[part].steps,
                            parts[part].ffn, branch);
    }
    b->ahead_layer[b->card] = 0;
    const uint32_t last_ffn = dead[1] ? parts[last].ffn & ~(uint32_t)FFN_SUM : parts[last].ffn;
    if (!ahead) {
	    status = THEN(split_part(b, layer, source, group, parts[last].half,
				     chunk_rows, 0, last_ffn, LAUNCH_CHAIN));
    } else { /* the next layer's attention of A through its tensor plan, its pool's source and selection's group */
	    status = dead[1] ? status :
			       THEN(split_part(b, layer, source, group, 1u,
					       chunk_rows, 0, FFN_SUM,
					       LAUNCH_SIDE));
	    status = THEN(tp2_layer_select(b, next));
	    status = THEN(split_part(
		    b, next,
		    (V41_DATA_SOURCE_LAYERS >> next & 1u) != 0 ? next : source,
		    (V41_DATA_INDEX_LAYERS >> next & 1u) != 0 ? next : group, 0,
		    chunk_rows, HALF_ATTENTION, 0,
		    dead[1] ? LAUNCH_CHAIN : LAUNCH_RESUME));
	    status = THEN(tp2_layer_select(b, layer));
	    status = THEN(split_part(b, layer, source, group, 1u, chunk_rows, 0,
				     FFN_POST,
				     dead[1] ? LAUNCH_CHAIN : LAUNCH_JOIN));
	    b->ahead_layer[b->card] = status == OK ? next + 1u : 0u;
    }
    half_select(b, 2u, chunk_rows);
    return status;
}

MUST_CHECK static result_status
tp2_program_layer_steps_inner(struct builder *build, uint32_t layer,
			      uint32_t source, uint32_t group, bool front)
{
	const bool divided = layer == MODEL_DECODER_FIRST_LAYER;
	if (build->slab_count == 1u) {
		const uint32_t steps = front   ? ENGRAM_STEPS | FRONT_STEPS :
				       divided ? REST_STEPS :
						 ENGRAM_STEPS | FRONT_STEPS |
							 REST_STEPS;
		return layer_steps(build, layer, source, group, steps);
	}
    if (!front && pipelined(build, layer)) /* split_overlap's pipeline: the attention in split_layer's halves */
        return split_layer(build, layer, source, group);
    /* a tensor layer's sums are built over one wide launch's rows at most (split_overlap: a half's) */
    if (!front && build->tensor_layer != NULL && piece_count_of(build) > 1u && !split_applies(build, layer))
	    return STATUS_INVALID_ARGUMENT;
    /* chunk_core: the slabs' scans in place of their cores, then the core over the chunk. batch_core_once, a BATCH
     * program of several slabs: each slab's state writes (its window writes too), queries and scan as before, then
     * one core over the step's rows through the step's own block: a row's core reads its own queries, selection and
     * rotation, the window of positions up to its own (the 4,096-slot ring: later rows' slots are others) and pool
     * entries below its bound (later rows complete only later groups), so it sees what it saw a slab at a time: the
     * same bits */
    const bool batch_core = build->batch_core_once != 0 &&
			    build->made->program.kind == PROGRAM_BATCH &&
			    build->slab_count > 1u &&
			    piece_count_of(build) == 1u &&
			    build->tensor_layer != NULL;
    const bool chunk_core = chunk_core_applies(build) || batch_core;
    const uint32_t core = chunk_core ? SCAN : CORE;
    /* queries_piece: the queries with wq_b over each piece, not a span at a time; index_span: the slabs' scans behind
     * the slab loop, a span at a time (index_scans) */
    const uint32_t piece_queries = queries_piece_applies(build) ? QUERIES : 0u, span_queries = piece_queries ^ QUERIES;
    const bool spans = chunk_core && index_span_applies(build);
    const uint32_t slab_core = spans ? 0u : core;
    result_status status = OK;
    if (front || !divided) {
	    /* chunk_core: window writes over the chunk, pool writes per slab (layer 20's front: per slab too).
         * BATCH level 3 computes pools over the step, then retains compressor slots separately: mixed prompt rows
         * can wrap a session's slots, and must not overwrite a previous partner before a pool row reads it. */
	    const bool window =
		    chunk_core && !original_chunk_core_applies(build) &&
		    !divided && (!batch_core || build->batch_core_once >= 2u);
	    const bool pool_chunk = window && !thin_tail_program(build) &&
				    (!batch_core || build->batch_core_once >=
							    BATCH_CORE_POOLS) &&
				    build->tp_description != NULL &&
				    build->tp_description->stages[build->card]
					    ->description.pool_chunk;
	    status = steps_over(build, layer, source, group,
				AHEAD_STEPS |
					(divided ? 0u :
						   QUERY_NORM | QUERY_STEPS |
							   piece_queries),
				true);
	    chunk_select(build);
	    status = window ? THEN(layer_steps(build, layer, source, group,
					       WINDOW_WRITES)) :
			      status;
	    status = pool_chunk ? THEN(steps_over(build, layer, source, group,
						  POOL_CHUNK, true)) :
				  status;
	    status = THEN(steps_over(
		    build, layer, source, group,
		    (window ? (pool_chunk ? 0u : STATE_POOL) : STATE_WRITES) |
			    (divided ? 0u : span_queries | slab_core),
		    false));
	    status = spans && !divided ?
			     THEN(index_scans(build, layer, source, group, 0,
					      build->slab_count)) :
			     status;
    } else {
        status = steps_over(build, layer, source, group, QUERY_STEPS | piece_queries, true);
        status = THEN(steps_over(build, layer, source, group, span_queries | slab_core, false));
        status = spans ? THEN(index_scans(build, layer, source, group, 0, build->slab_count)) : status;
    }
    if (chunk_core && (!divided || !front) && status == OK) {
        chunk_select(build);
        status = layer_steps(build, layer, source, group, CORE_ALONE);
    }
    /* exact_dense: its products a piece, its gathers a slab at a time (dense_gather_slabs); split_overlap: in halves */
    return front ? status : split_applies(build, layer) ? THEN(split_layer(build, layer, source, group)) :
                            THEN(steps_over(build, layer, source, group, OUTPUT_PRODUCTS | SITE_POST, true));
}

MUST_CHECK static result_status
tp2_program_layer_steps(struct builder *build, uint32_t layer, uint32_t source,
			uint32_t group, bool front)
{
	const struct v41_tp2_layer_plan *const tensor = build->tensor_layer;
	if (tensor != NULL && tensor->exact_output_shards &&
	    !tensor->exact_dense)
		build->tensor_layer = NULL;
	const result_status status = tp2_program_layer_steps_inner(
		build, layer, source, group, front);
	build->tensor_layer = tensor;
	return status;
}

MUST_CHECK static result_status
tp4_program_owner_layer(struct builder *build,
			const struct v41_tp2_program_description *description,
			struct v41_tp2_layer_plan *tensor,
			struct step_companion_pair *pairs, uint32_t *pair_count,
			uint32_t layer, uint32_t source, uint32_t group,
			bool front)
{
	const uint32_t owner = v41_tp4_attention_owner(layer);
	result_status status =
		tp2_program_select(build, description, tensor, owner, layer);
	uint32_t first_launch = build->made->cards[owner].launch_count;
	status = THEN(
		tp2_program_layer_steps(build, layer, source, group, front));
	status = THEN(tp2_program_segment(build, layer, first_launch));
	const bool divided = layer == MODEL_DECODER_FIRST_LAYER;
	status = THEN(tp2_program_copies(build, layer,
					 front	 ? FRONT_COPIES :
					 divided ? LATER_COPIES :
						   EVERY_COPY));
	if (front) {
		if (status == OK)
			build->made->program.front_segment_count =
				build->made->program.segment_count;
		return status;
	}
    /* Attention has completed its original post-mix. Peers need both residual arrays before their FFN fronts. */
    for (uint32_t rank = 0; rank < TP4_RANKS && status == OK; rank++) {
	    if (rank != owner)
		    status = tp2_program_streams(build, rank);
    }
    const uint32_t first_segment = build->made->program.segment_count;
    for (uint32_t rank = 0; rank < TP4_RANKS && status == OK; rank++) {
	    status =
		    tp2_program_select(build, description, tensor, rank, layer);
	    first_launch = build->made->cards[rank].launch_count;
	    status = build->slab_count > 1u ?
			     THEN(wide_ffn_launches(build, layer)) :
			     THEN(ffn_launches(build, layer, &first_launch));
	    status = THEN(tp2_program_segment(build, layer, first_launch));
	    if (rank == 0)
		    status = THEN(
			    tp2_program_boundary(build, description, layer));
    }
    return THEN(tp2_program_pair_add(build, pairs, pair_count, first_segment));
}

MUST_CHECK static result_status
tp2_program_layer_pair(struct builder *build,
		       const struct v41_tp2_program_description *description,
		       struct v41_tp2_layer_plan *tensor,
		       struct step_companion_pair *pairs, uint32_t *pair_count,
		       uint32_t layer, uint32_t source, uint32_t group,
		       bool front)
{
    const uint32_t first_card = tp2_program_layer_card(description, layer);
    if (description->stages[first_card]->description.attention_owner)
        return tp4_program_owner_layer(build, description, tensor, pairs, pair_count, layer, source, group, front);
    const uint32_t first_segment = build->made->program.segment_count;
    const uint32_t ranks = description->stages[first_card]->description.rank_count;
    const bool divided = layer == MODEL_DECODER_FIRST_LAYER;
    result_status status = OK;
    for (uint32_t rank = 0; rank < ranks && status == OK; rank++) {
        status = tp2_program_select(build, description, tensor, first_card + rank, layer);
        uint32_t first_launch = build->made->cards[build->card].launch_count;
	if (!front && (rank == 0 || build->draft_layers) &&
	    tp2_program_exl3_all(description) &&
	    description->canonical->limits.draft_positions != 0 &&
	    layer >= MODEL_DRAFT_FIRST_TAP_LAYER) {
		/* Rank0 retains the original streams before the target changes them (draft_layers: every rank, its
             * own context). Wide taps span the chunk. */
		chunk_select(build);
		status = THEN(draft_tap_launch(build, layer));
	}
	/* split_overlap: the FFN in split_layer's halves, inside the layer steps */
        const bool split = !front && build->slab_count > 1u && split_applies(build, layer);
        status = THEN(tp2_program_layer_steps(build, layer, source, group, front));
        if (!front && !split)
            status = build->slab_count > 1u ? THEN(wide_ffn_launches(build, layer)) :
                THEN(ffn_launches(build, layer, &first_launch));
        status = THEN(tp2_program_segment(build, layer, first_launch));
        if (rank == 0) {
            status = THEN(tp2_program_copies(build, layer, front ? FRONT_COPIES : divided ? LATER_COPIES : EVERY_COPY));
            if (!front)
                status = THEN(tp2_program_boundary(build, description, layer));
        }
    }
    status = THEN(tp2_program_pair_add(build, pairs, pair_count, first_segment));
    if (front && status == OK)
        build->made->program.front_segment_count = build->made->program.segment_count;
    return status;
}

MUST_CHECK static result_status
tp2_program_walk(struct builder *build,
		 const struct v41_tp2_program_description *description,
		 struct v41_tp2_layer_plan *tensor,
		 struct step_companion_pair *pairs, uint32_t *pair_count)
{
    build->stage = 0;
    result_status status = tp2_program_select(build, description, tensor, 0, 0);
    uint32_t first_launch = build->made->cards[0].launch_count;
    const bool replicated = tp2_program_replicated(description);
    if (!replicated) {
        build->tensor_layer = NULL; /* embedding is a single unsharded lookup/expansion */
	const enum program_kind kind = build->made->program.kind;
	if (status == OK && build->head_stage != NULL &&
	    build->card == description->canonical->result_card &&
	    (kind == PROGRAM_THIN || kind == PROGRAM_BATCH))
		status = settings_push_launch(
			build); /* head_shards: every rank's settings copy for its samplers */
        status = THEN(step_launches(build, true));
	status = THEN(
		tp2_program_segment(build, SEGMENT_NO_LAYER, first_launch));
	for (uint32_t rank = 1; rank < description->stages[0]->description.rank_count && status == OK; rank++)
            status = tp2_program_streams(build, rank);
    }
    status = THEN(tp2_program_rotations(build, description, tensor, pairs, pair_count, 0, 0, replicated));
    uint32_t source = 0, group = 0;
    const bool chunked = build->slab_count > 1u;
    /* the draft context's pieces: batch_context_slabs, a BATCH program's slabs (the thin steps' product forms) */
    const uint32_t context_rows =
	    !chunked ? 0u :
	    build->made->program.kind == PROGRAM_BATCH &&
			    description
				    ->stages[description->canonical->result_card]
				    ->description.batch_context_slabs ?
		       LINEAR_MAXIMUM_ROWS :
		       LINEAR_WIDE_MOST_ROWS;
    const bool front_only = build->made->program.kind == PROGRAM_WIDE;
    for (uint32_t layer = 0; layer < MODEL_LAYERS && status == OK; layer++) {
	    source = (V41_DATA_SOURCE_LAYERS >> layer & 1u) != 0 ? layer :
								   source;
	    group = (V41_DATA_INDEX_LAYERS >> layer & 1u) != 0 ? layer : group;
	    const uint32_t owner = tp2_program_layer_card(description, layer);
	    if (layer != 0 &&
		owner != tp2_program_layer_card(description, layer - 1u)) {
		    build->stage++;
		    status = tp2_program_rotations(build, description, tensor,
						   pairs, pair_count, owner,
						   layer, false);
	    }
	    if (description->stages[owner]->description.rank_count > 1u) {
		    if (layer == MODEL_DECODER_FIRST_LAYER)
			    status = THEN(tp2_program_layer_pair(
				    build, description, tensor, pairs,
				    pair_count, layer, source, group, true));
		    if (layer == MODEL_DECODER_FIRST_LAYER && front_only)
			    return status;
		    status = THEN(tp2_program_layer_pair(
			    build, description, tensor, pairs, pair_count,
			    layer, source, group, false));
	    } else {
		    status = THEN(tp2_program_select(build, description, tensor,
						     owner, layer));
		    first_launch = build->made->cards[owner].launch_count;
		    status =
			    description->canonical->limits.draft_positions !=
						    0 &&
					    layer >=
						    MODEL_DRAFT_FIRST_TAP_LAYER ?
				    THEN(draft_tap_launch(build, layer)) :
				    status;
		    const bool divided = layer == MODEL_DECODER_FIRST_LAYER;
		    status = THEN(tp2_program_layer_steps(build, layer, source,
							  group, divided));
		    if (divided) {
			    status = THEN(tp2_program_segment(build, layer,
							      first_launch));
			    status = THEN(tp2_program_copies(build, layer,
							     FRONT_COPIES));
			    build->made->program.front_segment_count =
				    build->made->program.segment_count;
			    if (front_only)
				    return status;
			    first_launch =
				    build->made->cards[owner].launch_count;
			    status = THEN(tp2_program_layer_steps(
				    build, layer, source, group, false));
		    }
		    status = chunked ? THEN(wide_ffn_launches(build, layer)) :
				       THEN(ffn_launches(build, layer,
							 &first_launch));
		    status = THEN(
			    tp2_program_segment(build, layer, first_launch));
		    status = THEN(tp2_program_copies(
			    build, layer, divided ? LATER_COPIES : EVERY_COPY));
		    status = THEN(
			    tp2_program_boundary(build, description, layer));
	    }
    }
    if (status == OK && tp2_program_exl3_all(description)) {
        /* The last companion is rank3. Head and unsharded native draft always read rank0's completed rows. */
	status = tp2_program_select(build, description, tensor,
				    description->canonical->result_card,
				    MODEL_DRAFT_FIRST_LAYER);
    }
    if (status == OK && description->canonical->limits.draft_positions != 0) {
        first_launch = build->made->cards[build->card].launch_count;
        status = draft_context_launches(build, context_rows);
	status = THEN(
		tp2_program_segment(build, SEGMENT_NO_LAYER, first_launch));
    }
    for (uint32_t rank = 0; status == OK && description->canonical->limits.draft_positions != 0 &&
         rank < description->canonical->card_count; rank++) {
        /* draft_layers: every other rank's own draft context (its rank of the draft reads its own window) */
        if (rank == description->canonical->result_card || !description->stages[rank]->description.draft_layers)
            continue;
	status = tp2_program_select(build, description, tensor, rank,
				    MODEL_DRAFT_FIRST_LAYER);
	first_launch = build->made->cards[rank].launch_count;
        status = THEN(draft_context_launches(build, context_rows));
	status = THEN(
		tp2_program_segment(build, SEGMENT_NO_LAYER, first_launch));
    }
    const uint32_t result = description->canonical->result_card;
    if (build->made->program.kind == PROGRAM_BATCH) {
	    /* BATCH: each sampling card's heads slab by slab, the result card's last, each slab's own result area */
	    for (uint32_t step = 0;
		 status == OK && step < description->canonical->card_count;
		 step++) {
		    const uint32_t card =
			    step + 1u == description->canonical->card_count ?
				    result :
			    step < result ? step :
					    step + 1u;
		    if (card != result &&
			!description->stages[card]->description.head_shards)
			    continue;
		    status = tp2_program_select(build, description, tensor,
						card, MODEL_DRAFT_FIRST_LAYER);
		    first_launch = build->made->cards[card].launch_count;
		    if (description->stages[card]->description.batch_wide_head)
			    status = batch_wide_head_launches(
				    build, description->stages[card]);
		    for (uint32_t slab = 0;
			 slab < build->slab_count && status == OK &&
			 !description->stages[card]->description.batch_wide_head;
			 slab++) {
			    batch_slab_select(build, slab);
			    build->arrays
				    [V41_DATA_ARRAY_RESULT] = (struct linear_range){
				    description->stages[card]
						    ->batch_results.byte_offset +
					    (uint64_t)slab *
						    PROGRAM_BATCH_SLAB_RESULT_BYTES,
				    PROGRAM_BATCH_SLAB_RESULT_BYTES
			    };
			    build->sampler_call = slab;
			    status = head_launches(build);
		    }
		    build->sampler_call = 0;
		    chunk_select(build);
		    const struct model_limits *const own =
			    &description->canonical->limits;
		    for (uint32_t verifier = 0;
			 status == OK && card != result &&
			 own->draft_positions != 0 &&
			 description->stages[card]->description.draft_shards &&
			 !description->stages[card]->description.draft_layers &&
			 verifier < batch_verifiers(description, own,
						    build->chunk_rows);
			 verifier++)
			    status = draft_shard_rank_launches(
				    build, verifier,
				    build->slab_count +
					    verifier * own->draft_positions);
		    status = THEN(tp2_program_segment(build, SEGMENT_NO_LAYER,
						      first_launch));
	    }
	    /* then each verifier's draft walk on the result card (draft_records.h's round table, slab results), each a thin
         * program's walk: the card's kernels selected as a thin program's (the draft layers' site fronts) */
	    const struct model_limits *const limits =
		    &description->canonical->limits;
	    build->batch_stage = description->stages[result];
	    build->batch_slabs = build->slab_count;
	    build->slab_count = 1;
	    /* batch_draft_wide 2: the wide forms for a batched draft of more than 8 rows, where its layers can take them
         * (TP draft layers on the BF16 ring, no exact dense shards: batch_draft_rows_walk's once) */
	    const struct v41_tp2_stage_description *const draft_stage =
		    &build->batch_stage->description;
	    build->draft_wide_kernels =
		    draft_stage->batch_draft_rows &&
		    draft_stage->draft_layers &&
		    draft_stage->wire_format == ORDERED_PUSH_BF16 &&
		    !draft_stage->exact_dense &&
		    draft_stage->batch_draft_wide >= DRAFT_WIDE_PRODUCTS &&
		    batch_verifiers(description, limits, build->chunk_rows) *
				    limits->draft_positions >
			    LINEAR_MAXIMUM_ROWS;
	    if (status == OK && limits->draft_positions != 0)
		    status = tp2_program_select(build, description, tensor,
						result,
						MODEL_DRAFT_FIRST_LAYER);
	    if (status == OK && limits->draft_positions != 0 &&
		build->batch_stage->description.batch_draft_rows) {
		    const uint32_t verifiers = batch_verifiers(
			    description, limits, build->chunk_rows);
		    status = batch_draft_rows_walk(build, verifiers);
		    for (uint32_t rank = 0;
			 status == OK &&
			 build->batch_stage->description.draft_layers &&
			 rank < description->canonical->card_count;
			 rank++) {
			    /* draft_layers: every other rank's batched draft behind the result card's, its own areas */
			    if (rank == result)
				    continue;
			    status = tp2_program_select(
				    build, description, tensor, rank,
				    MODEL_DRAFT_FIRST_LAYER);
			    build->batch_stage = description->stages[rank];
			    status = THEN(
				    batch_draft_rows_walk(build, verifiers));
		    }
		    build->slab_count = build->batch_slabs;
		    build->draft_wide_kernels = false;
		    return status;
	    }
	    for (uint32_t verifier = 0;
		 status == OK && limits->draft_positions != 0 &&
		 verifier < batch_verifiers(description, limits,
					    build->chunk_rows);
		 verifier++) {
		    build->batch_draft = true;
		    build->batch_verifier = verifier;
		    chunk_select(build);
		    status = draft_walk(build);
	    }
	    build->batch_draft = false;
	    build->slab_count = build->batch_slabs;
	    build->draft_wide_kernels = false;
	    return status;
    }
    if (chunked)
        return status;
    for (uint32_t rank = 0; status == OK && rank < description->canonical->card_count; rank++) {
        /* head_shards: the other ranks' heads and sampler shards, before the result card's (head segments only) */
        if (rank == result || !description->stages[rank]->description.head_shards)
            continue;
	status = tp2_program_select(build, description, tensor, rank,
				    MODEL_DRAFT_FIRST_LAYER);
	first_launch = build->made->cards[rank].launch_count;
        status = THEN(head_launches(build));
	status = THEN(
		tp2_program_segment(build, SEGMENT_NO_LAYER, first_launch));
    }
    if (status == OK && build->card != result)
	    status = tp2_program_select(build, description, tensor, result,
					MODEL_DRAFT_FIRST_LAYER);
    first_launch = build->made->cards[build->card].launch_count;
    status = THEN(head_launches(build));
    status = THEN(tp2_program_segment(build, SEGMENT_NO_LAYER, first_launch));
    if (description->canonical->limits.draft_positions == 0)
        return status;
    if (!description->stages[result]->description.draft_layers)
        return THEN(draft_walk(build));
    for (uint32_t rank = 0; status == OK && rank < description->canonical->card_count; rank++) {
        /* draft_layers: the draft tail a segment a rank, the result card's first (its head's visit goes on) */
        const uint32_t card = rank == 0 ? result : rank <= result ? rank - 1u : rank;
	status = tp2_program_select(build, description, tensor, card,
				    MODEL_DRAFT_FIRST_LAYER);
	status = THEN(draft_walk(build));
    }
    return status;
}

static uint32_t
tp2_program_checkpoints(const struct v41_tp2_program_description *description,
			struct window_checkpoint *checkpoints)
{
	const struct model_plan_record *const plan = description->canonical;
	uint32_t count = 0;
	for (uint32_t layer = 0; layer < V41_DATA_WALK_LAYERS; layer++) {
		for (uint32_t card = 0; card < plan->card_count; card++) {
			if ((v41_walk_layers(&plan->cards[card]) >> layer &
			     1u) == 0)
				continue;
			if (description->stages[card]
				    ->description.attention_owner &&
			    v41_tp4_attention_owner(layer) != card) {
				if (plan->cards[card].ring_offset[layer] !=
					    PROGRAM_NOWHERE ||
				    plan->cards[card].checkpoint_offset[layer] !=
					    PROGRAM_NOWHERE)
					return UINT32_MAX;
				continue;
			}
			if (count >= TP2_PROGRAM_CHECKPOINTS ||
			    plan->cards[card].ring_offset[layer] ==
				    PROGRAM_NOWHERE ||
			    plan->cards[card].checkpoint_offset[layer] ==
				    PROGRAM_NOWHERE)
				return UINT32_MAX;
			checkpoints[count++] = (struct window_checkpoint){
				plan->cards[card].ring_offset[layer],
				plan->cards[card].checkpoint_offset[layer],
				V41_DATA_RING_SESSION_BYTES,
				MODEL_RING_SLOTS,
				MODEL_STATE_ROW_BYTES,
				MODEL_WINDOW - 1u,
				card
			};
		}
	}
    return count;
}

/* The four CMPs compute their own persistent source/index state. Their only outgoing hand-offs target the PRO,
 * whose work follows the complete prefix. Merge rank-local launch ranges up to the prefill-cone boundary,
 * then merge the remaining prefix. Embedding distribution and the native PRO/draft boundaries stay explicit. */
MUST_CHECK static result_status
tp4_graph_ranges_merge(struct made_program *made,
		       struct step_companion_pair *pairs, uint32_t *pair_count,
		       uint32_t graph_parts)
{
	struct execution_program *const program = &made->program;
	if (*pair_count == 0 || program->segment_count >= MOST_SEGMENTS)
		return STATUS_INVALID_ARGUMENT;
	struct segment_record merged[MOST_SEGMENTS];
	uint32_t mapping[MOST_SEGMENTS], count = 0, groups = 0;
	const uint32_t first = pairs[0].first_segment;
	const uint32_t end = pairs[*pair_count - 1u].second_segment + 1u;
	const uint32_t front = program->front_segment_count;
	if (first >= front || front > end)
		return STATUS_INVALID_ARGUMENT;
	for (uint32_t segment = 0; segment < first; segment++) {
		mapping[segment] = count;
		merged[count++] = made->segments[segment];
	}
    /* graph_parts (--tp_graph_parts): the cone's groups (a decode step's work; the same in every program, which the
     * driver holds equal) cut at graph_parts - 1 group starts spread evenly, so that a card's step begins after a
     * smaller graph's launch; the part behind the cone stays one */
    uint32_t boundaries[2u + TP_MOST_GRAPH_PARTS], boundary_count = 0;
    uint32_t starts[MOST_SEGMENTS], start_count = 0;
    for (uint32_t index = 0; index < *pair_count; index++)
        if (pairs[index].first_segment > first && pairs[index].first_segment < front)
            starts[start_count++] = pairs[index].first_segment;
    const uint32_t parts = graph_parts < 1u ? 1u : graph_parts > start_count + 1u ? start_count + 1u : graph_parts;
    boundaries[boundary_count++] = first;
    for (uint32_t cut = 1; cut < parts && parts <= TP_MOST_GRAPH_PARTS; cut++)
	    boundaries[boundary_count++] =
		    starts[cut * (start_count + 1u) / parts - 1u];
    boundaries[boundary_count++] = front;
    if (end > front)
        boundaries[boundary_count++] = end;
    for (uint32_t part = 0; part + 1u < boundary_count; part++) {
        if (boundaries[part] == boundaries[part + 1u])
            continue;
        const uint32_t group_first = count;
	for (uint32_t card = 0; card < TP4_RANKS; card++) {
		struct segment_record combined = { 0 };
		bool found = false;
		for (uint32_t segment = boundaries[part];
		     segment < boundaries[part + 1u]; segment++) {
			const struct segment_record *const original =
				&made->segments[segment];
			if (original->card != card)
				continue;
			if (original->pause != 0 ||
			    (found && (original->stage != combined.stage ||
				       original->first_launch !=
					       combined.first_launch +
						       combined.launch_count)))
				return STATUS_INVALID_ARGUMENT;
			if (!found) {
				combined = *original;
				combined.layer = SEGMENT_NO_LAYER;
				found = true;
			} else {
				combined.launch_count += original->launch_count;
			}
			mapping[segment] = count;
		}
		if (!found)
			return STATUS_INVALID_ARGUMENT;
		merged[count++] = combined;
	}
	pairs[groups++] =
		(struct step_companion_pair){ group_first, count - 1u };
	if (boundaries[part + 1u] == front)
            program->front_segment_count = count;
    }
    for (uint32_t segment = end; segment < program->segment_count; segment++) {
        mapping[segment] = count;
        merged[count++] = made->segments[segment];
    }
    mapping[program->segment_count] = count;
    for (uint32_t index = 0; index < program->hand_off_count; index++) {
	    struct hand_off_record *const transfer = &made->hand_offs[index];
	    if (transfer->after_segment >= program->segment_count)
		    return STATUS_INVALID_ARGUMENT;
	    /* Moving a transfer across its consumer would change the program. Group-internal transfers are refused. */
	    if (transfer->after_segment >= first &&
		transfer->after_segment < end &&
		transfer->to_card < V41_DATA_TP2_PRO_CARD)
		    return STATUS_UNSUPPORTED;
	    transfer->after_segment = mapping[transfer->after_segment];
    }
    program->draft_segment_first = mapping[program->draft_segment_first];
    memcpy(made->segments, merged, (size_t)count * sizeof(merged[0]));
    program->segment_count = count;
    *pair_count = groups;
    return OK;
}

result_status
v41_tp_program_prepare(const struct v41_tp2_program_description *description,
		       const struct backend_operations *backend,
		       struct device_card *const *cards,
		       const struct card_description_record *card_descriptions,
		       enum program_kind kind, uint32_t row_count,
		       struct execution_program **program_out,
		       struct step_companion_program *companions_out)
{
    if (program_out == NULL || companions_out == NULL ||
        !tp2_program_is_valid(description, backend, cards, card_descriptions, kind, row_count))
	    return STATUS_INVALID_ARGUMENT;
    const struct model_plan_record *const canonical = description->canonical;
    const bool batch = kind == PROGRAM_BATCH;
    /* a BATCH program of 8 rows or fewer walks as a thin one: no views (step_validation.c) */
    const bool chunked = kind != PROGRAM_THIN &&
			 !(batch && row_count <= LINEAR_MAXIMUM_ROWS);
    const bool exl3_all = tp2_program_exl3_all(description);
    const uint64_t chunk_layers =
	    kind == PROGRAM_FULL || batch ?
		    UINT64_MAX :
		    (1ull << (MODEL_DECODER_FIRST_LAYER + 1u)) - 1u;
    const uint32_t slabs = kind == PROGRAM_THIN ?
				   1u :
				   (row_count + LINEAR_MAXIMUM_ROWS - 1u) /
					   LINEAR_MAXIMUM_ROWS;
    const uint32_t context_pieces =
	    !chunked ? 1u :
	    batch && description->stages[canonical->result_card]
				    ->description.batch_context_slabs ?
		       slabs :
		       (row_count + LINEAR_WIDE_MOST_ROWS - 1u) /
			       LINEAR_WIDE_MOST_ROWS;
    uint32_t launch_capacity[V41_DATA_TP2_PROGRAM_CARDS], launch_total = 0;
    for (uint32_t card = 0; card < canonical->card_count; card++) {
        const uint32_t layers = (uint32_t)__builtin_popcountll(canonical->cards[card].layers);
	launch_capacity[card] =
		layers *
			(exl3_all || card < V41_DATA_TP2_PRO_CARD ?
				 TP2_LAYER_LAUNCHES :
				 LAYER_MOST_LAUNCHES) *
			slabs +
		CARD_MOST_LAUNCHES;
	for (uint32_t layer = 0; layer < MODEL_LAYERS; layer++) {
		if ((canonical->cards[card].layers >> layer & 1u) != 0 &&
		    canonical->exl3_bits[layer] != 0)
			launch_capacity[card] +=
				EXL3_ATTENTION_EXTRA_LAUNCHES * slabs;
	}
	if (card == canonical->result_card && canonical->limits.draft_positions != 0)
		launch_capacity[card] +=
			MODEL_DRAFT_TAPS +
			context_pieces * CONTEXT_PIECE_LAUNCHES +
			MODEL_DRAFT_STAGES +
			DRAFT_MOST_LAUNCHES(canonical->limits.draft_positions);
	if (batch) /* a BATCH program: the head of each slab on a card that samples; the verifiers' draft walks (with
                    * batch_draft_rows: a slab of draft rows as many launches as a walk at most, plus a verify a walk) */
		launch_capacity[card] +=
			slabs * CARD_MOST_LAUNCHES +
			(card == canonical->result_card &&
					 canonical->limits.draft_positions !=
						 0 ?
				 batch_verifiers(description,
						 &canonical->limits,
						 row_count) *
						 (DRAFT_MOST_LAUNCHES(
							  canonical->limits
								  .draft_positions) +
						  2u) +
					 V41_DATA_BATCH_DRAFT_SLABS *
						 DRAFT_MOST_LAUNCHES(
							 canonical->limits
								 .draft_positions) +
					 1u :
				 0u);
	if (description->stages[card]->description.head_shards && card == canonical->result_card)
            launch_capacity[card] += 1u; /* head_shards: the settings push */
        if (description->stages[card]->description.draft_layers && canonical->limits.draft_positions != 0) {
            /* draft_layers: every rank's taps, context and draft walk with its stages' tensor-parallel launches (a
             * BATCH program: the batched draft's slabs each, and its verifies) */
            const uint32_t positions = canonical->limits.draft_positions;
	    const uint32_t walks = batch ? V41_DATA_BATCH_DRAFT_SLABS : 1u;
	    launch_capacity[card] +=
		    MODEL_DRAFT_TAPS + context_pieces * CONTEXT_PIECE_LAUNCHES +
		    MODEL_DRAFT_STAGES +
		    walks * (DRAFT_MOST_LAUNCHES(positions) +
			     MODEL_DRAFT_STAGES * TP2_LAYER_LAUNCHES) +
		    (batch ? batch_verifiers(description, &canonical->limits,
					     row_count) *
					     (DRAFT_MOST_LAUNCHES(positions) +
					      1u) +
				     1u :
			     0u);
	}
        /* draft_shards: the result card's share; another rank's wait, cut and each position's Markov and sampler */
        if (description->stages[card]->description.draft_shards) /* a BATCH program: each verifier's */
            launch_capacity[card] += (batch ? batch_verifiers(description, &canonical->limits, row_count) : 1u) *
                (card == canonical->result_card ? 1u : DRAFT_SHARD_LAUNCHES(canonical->limits.draft_positions));
        launch_total += launch_capacity[card];
    }
    _Static_assert(sizeof(struct kernel_launch) %
				   _Alignof(struct window_checkpoint) ==
			   0,
		   "checkpoint tail follows aligned launch storage");
    const size_t launch_end =
	    sizeof(struct made_program) +
	    (size_t)launch_total * sizeof(struct kernel_launch);
    struct builder *const build = calloc(1, sizeof(*build));
    struct v41_tp2_layer_plan *const tensor = calloc(1, sizeof(*tensor));
    struct made_program *const made =
	    calloc(1, launch_end +
			      (size_t)TP2_PROGRAM_CHECKPOINTS *
				      sizeof(struct window_checkpoint) +
			      (size_t)TP2_PROGRAM_PAIRS *
				      sizeof(struct step_companion_pair));
    if (build == NULL || tensor == NULL || made == NULL) {
        free(made); free(tensor); free(build);
	return STATUS_OUT_OF_MEMORY;
    }
    void *const checkpoint_storage = (unsigned char *)made + launch_end;
    struct window_checkpoint *const checkpoints = checkpoint_storage;
    struct step_companion_pair *const pairs =
	    (struct step_companion_pair *)(checkpoints +
					   TP2_PROGRAM_CHECKPOINTS);
    /* split_overlap: two half views behind the slabs' where a chunked program's rows split on slabs */
    const uint32_t half_rows =
	    row_count / (2u * LINEAR_MAXIMUM_ROWS) * LINEAR_MAXIMUM_ROWS;
    const bool halves = (kind == PROGRAM_WIDE || kind == PROGRAM_FULL) &&
			description->stages[0]->description.split_overlap &&
			half_rows >= V41_DATA_SPAN_ROWS &&
			row_count - half_rows <= LINEAR_WIDE_MOST_ROWS;
    *build = (struct builder){ .plan = canonical,
			       .backend = backend,
			       .cards = cards,
			       .descriptions = card_descriptions,
			       .made = made,
			       .rows = row_count,
			       .chunk_rows = row_count,
			       .program_rows = row_count,
			       .slab_count = slabs,
			       .half_view_first = halves ? slabs + 1u : 0u,
			       .half_rows = halves ? half_rows : 0u,
			       .tensor_slices = description->slice_count,
			       .head_stream_kernel = BACKEND_NO_KERNEL };
    made->program.kind = kind;
    for (uint32_t card = 0, base = 0; card < canonical->card_count; card++) {
	    const struct v41_card *const planned = &canonical->cards[card];
	    const uint64_t *const arrays = planned->array_offset;
	    const bool has_views = chunked &&
				   (card == canonical->embedding_card ||
				    (planned->layers & chunk_layers) != 0);
	    build->launches[card] = made->launches + base;
	    build->launch_capacity[card] = launch_capacity[card];
	    base += launch_capacity[card];
	    for (uint32_t slab = 0; slab < slabs && has_views; slab++) {
		    made->views[card][slab] = (struct step_view){
			    arrays[V41_DATA_ARRAY_VIEWS] +
				    (uint64_t)slab * VIEW_BYTES,
			    slab * LINEAR_MAXIMUM_ROWS, LINEAR_MAXIMUM_ROWS
		    };
	    }
        /* split_overlap: the two halves' views behind the slabs' (their blocks the halves' rows of each step) */
	for (uint32_t half = 0;
	     half < V41_DATA_HALF_VIEWS && has_views && halves; half++)
		made->views[card][slabs + half] = (struct step_view){
			arrays[V41_DATA_ARRAY_VIEWS] +
				(uint64_t)slabs * VIEW_BYTES +
				half * (STEP_BLOCK_BYTES_VALUE +
					(uint64_t)half_rows *
						STEP_ROW_BYTES_VALUE),
			half * half_rows,
			half == 0 ? half_rows : row_count - half_rows
		};
	made->cards[card] = (struct program_card_record){
		build->launches[card], planned->arena_byte_count,
		arrays[V41_DATA_ARRAY_STEP], arrays[V41_DATA_ARRAY_SESSIONS],
		/* a BATCH model: BATCH and chunked programs' result area the slab results (one driver's cards agree; the
             * chunked ones sample nothing) */
		description->stages[0]->description.batch &&
				kind != PROGRAM_THIN &&
				card == canonical->result_card ?
			description->stages[card]->batch_results.byte_offset :
			arrays[V41_DATA_ARRAY_RESULT],
		arrays[V41_DATA_ARRAY_LOGITS], arrays[V41_DATA_ARRAY_STREAMS],
		arrays[V41_DATA_ARRAY_CARRIED_PRE],
		arrays[V41_DATA_ARRAY_ENGRAM_ROWS], planned->engram_tables, 0,
		has_views ? made->views[card] : NULL,
		has_views ? slabs + (halves ? V41_DATA_HALF_VIEWS : 0u) : 0,
		canonical->limits.draft_positions == 0 ||
				card != canonical->result_card ?
			PROGRAM_NOWHERE :
		batch	 ? description->stages[card]->batch_rounds.byte_offset :
		!chunked ? arrays[V41_DATA_ARRAY_ROUND] :
			   PROGRAM_NOWHERE
	};
    }
    uint32_t pair_count = 0;
    result_status status =
	    tp2_program_walk(build, description, tensor, pairs, &pair_count);
    if (status == OK && made->program.segment_count >= MOST_SEGMENTS)
	    status =
		    STATUS_INVALID_ARGUMENT; /* segment_add may have dropped one: no program */
    if (status != OK)
	    LOG_ERROR(
		    "tensor program: walk failed at card %u stage %u after %u segments: %u",
		    build->card, build->stage, made->program.segment_count,
		    (uint32_t)status);
    if (status == OK) {
	    const uint32_t draft_first =
		    (kind == PROGRAM_THIN || batch) &&
				    canonical->limits.draft_positions != 0 ?
			    made->program.draft_segment_first :
			    made->program.segment_count;
	    const uint32_t checkpoints_count =
		    tp2_program_checkpoints(description, checkpoints);
	    uint32_t expected_checkpoints = 0, expected_groups = 0;
	    for (uint32_t card = 0; card < canonical->card_count; card++) {
		    const struct v41_tp2_stage_description *const stage =
			    &description->stages[card]->description;
		    for (uint32_t layer = 0; layer < V41_DATA_WALK_LAYERS;
			 layer++) {
			    if ((v41_walk_layers(&canonical->cards[card]) >>
					 layer &
				 1u) != 0 &&
				(!stage->attention_owner ||
				 v41_tp4_attention_owner(layer) == card))
				    expected_checkpoints++;
		    }
		    if (stage->rank == 0 && stage->rank_count > 1u) {
			    const uint32_t last =
				    kind == PROGRAM_WIDE ?
					    MODEL_DECODER_FIRST_LAYER +
						    (stage->attention_owner ?
							     0u :
							     1u) :
					    MODEL_LAYERS;
			    if (stage->first_target_layer < last) {
				    const uint32_t left =
					    last - stage->first_target_layer;
				    expected_groups +=
					    (stage->target_layer_count < left ?
						     stage->target_layer_count :
						     left) +
					    1u;
				    expected_groups +=
					    !stage->attention_owner &&
					    kind != PROGRAM_WIDE &&
					    (canonical->cards[card].layers >>
						     MODEL_DECODER_FIRST_LAYER &
					     1u) != 0;
			    }
		    }
        }
	const uint32_t layout =
		exl3_all ? V41_DATA_TP_LAYOUT_QUAD_ALL :
		description->stages[0]->description.rank_count == TP4_RANKS ?
			   V41_DATA_TP_LAYOUT_QUAD :
		description->stages[V41_DATA_TP2_PAIR_B_CARD]
					->description.rank_count == 1u ?
			   V41_DATA_TP_LAYOUT_HYBRID :
			   V41_DATA_TP_LAYOUT_PAIRS;
	status = checkpoints_count == expected_checkpoints &&
				 pair_count == expected_groups ?
			 OK :
			 STATUS_INVALID_ARGUMENT;
	if (status != OK)
		LOG_ERROR("tensor program: %u/%u checkpoints and %u/%u groups",
			  checkpoints_count, expected_checkpoints, pair_count,
			  expected_groups);
	if (status == OK) {
		made->program = (struct execution_program){
			made->cards,
			made->segments,
			made->hand_offs,
			canonical->limits,
			canonical->card_count,
			made->program.segment_count,
			made->program.hand_off_count,
			row_count,
			canonical->result_card,
			made->expert_pauses,
			made->host_expert_offsets,
			0,
			0,
			kind,
			0,
			made->session_ranges,
			session_ranges_make(canonical, made->session_ranges),
			made->program.front_segment_count,
			MODEL_CONE_ROWS,
			checkpoints_count,
			checkpoints,
			made->leases,
			draft_first
		};
		const bool whole_graph =
			(layout == V41_DATA_TP_LAYOUT_QUAD || exl3_all) &&
			!description->stages[0]->description.attention_owner &&
			(description->stages[0]
				 ->description.exact_output_shards ||
			 description->stages[0]->description.wire_format ==
				 ORDERED_PUSH_BF16);
		if (whole_graph)
			status = tp4_graph_ranges_merge(
				made, pairs, &pair_count,
				description->stages[0]->description.graph_parts);
		if (status != OK) {
			free(made);
			free(tensor);
			free(build);
			return status;
		}
		*companions_out = (struct step_companion_program){
			&made->program, pairs, pair_count, 0,
			(description->stages[0]->description.exact_output_shards ?
				 V41_DATA_TP4_EXACT_RECIPE :
				 V41_DATA_TP2_PROGRAM_RECIPE) ^
				description->slice_count ^
				(description->stages[0]->description
								 .wire_format !=
							 0 &&
						 (layout ==
							  V41_DATA_TP_LAYOUT_QUAD ||
						  exl3_all) ?
					 V41_DATA_TP4_RING_IDENTITY :
					 0) ^
				((uint64_t)description->stages[0]
					 ->description.wire_format
				 << V41_DATA_TP2_WIRE_IDENTITY_SHIFT) ^
				((uint64_t)layout
				 << V41_DATA_TP_LAYOUT_IDENTITY_SHIFT) ^
				(canonical->limits.row_capacity >
						 LINEAR_MAXIMUM_ROWS ?
					 V41_DATA_TP_WIDE_IDENTITY :
					 0) ^
				(layout == V41_DATA_TP_LAYOUT_QUAD ?
					 V41_DATA_TP4_ALIGNED_IDENTITY :
					 0) ^
				(description->stages[0]
						 ->description.attention_owner ?
					 V41_DATA_TP4_ATTENTION_OWNER_IDENTITY :
					 0) ^
				(exl3_all ?
					 V41_DATA_TP4_EXL3_IDENTITY ^
						 ((uint64_t)description
							  ->stages[0]
							  ->description
							  .exl3_product_order
						  << V41_DATA_TP4_EXL3_ORDER_SHIFT) :
					 0)
		};
		*program_out = &made->program;
        }
    }
    if (status != OK)
        free(made);
    free(tensor); free(build);
    return status;
}

result_status
v41_tp2_program_prepare(const struct v41_tp2_program_description *description,
			const struct backend_operations *backend,
			struct device_card *const *cards,
			const struct card_description_record *card_descriptions,
			uint32_t row_count,
			struct execution_program **program_out,
			struct step_companion_program *companions_out)
{
	return v41_tp_program_prepare(description, backend, cards,
				      card_descriptions, PROGRAM_THIN,
				      row_count, program_out, companions_out);
}
