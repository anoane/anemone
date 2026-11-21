/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * model_v41.c: DeepSeek-V4.1-Flash's description, tensors, plan, arena and frequency words. Contract: model_v41.h.
 * A plan is one walk over a load layout (state_byte_count makes the loader's layout of its placement): the global and
 * layer tensors give the cards, the routed experts the expert tables (a tag where one is in host memory) and the
 * cards with slots, then each card's state is laid out behind its weights and its image takes its tables. A size is
 * a sum or product in 64 bits that stops at UINT64_MAX, refused there.
 */
#include "model_v41.h"
#include "models/deepseek_v41_flash/official/ds41f_official_hc.h"
#include "models/deepseek_v41_flash/official/ds41f_official_reference.h"
#include "models/deepseek_v41_flash/tensor_parallel/ds41f_tp_layer.h"
#include "models/deepseek_v41_flash/tensor_parallel/ds41f_tp_stage.h"
#include "engine/library/exl3/exl3_tc_records.h"
#include "engine/library/rings/exact_ring_records.h"
#include "engine/library/exl3/exl3.h"
#include "models/deepseek_v41_flash/draft/ds41f_dspark_records.h"
#include "models/deepseek_v41_flash/tensor_parallel/ds41f_draft_share_records.h"
#include "engine/library/sampling/sampler_shard_records.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "engine/backend/fill_records.h"
#include "engine/library/block_scaled_linear/expert_relaxed.h"
#include "common/hooks.h"
#include "models/deepseek_v41_flash/indexer/ds41f_index_scan.h"
#include "engine/formats/digest/pack_digest.h"
#include "engine/library/block_scaled_linear/relaxed_linear.h"
#include "engine/library/prepare_checks.h"

#define NAME_CAPACITY 64                      /* "layers.<l>.ffn.experts.<e>.w1.weight" and the longest, with NUL */
#define LAYER_PREFIX_CAPACITY 20              /* "layers.<l>." of any 32-bit l, with NUL */
#define MATRIX_DIMENSIONS 2u
#define E2M1_VALUES_PER_BYTE 2u
#define EVERY 0xFFFFFFFFFFull                 /* bit l for each of the 40 layers */
#define WALKED (EVERY | DS41F_DRAFT_LAYERS) /* and the draft's stages */
#define BIT(index) (1ull << (index))

enum {
	BF16 = PACK_ELEMENT_BFLOAT16,
	F32 = PACK_ELEMENT_FLOAT32,
	U8 = PACK_ELEMENT_UINT8
};
enum {
	PLAIN = WEIGHT_FORMAT_NONE,
	E4M3 = WEIGHT_FORMAT_E4M3_SCALED,
	E2M1 = WEIGHT_FORMAT_E2M1_SCALED
};

/* A tensor as the programs read it: element type, outputs and inputs (0 for a vector), format and its bits, the
 * layers that have it. The shapes are REF's and CFG's (ds41f_geometry.h); the formats the converter's. */
struct tensor_rule {
    const char *name;
    uint32_t element_type;
    uint32_t dimensions[MATRIX_DIMENSIONS];
    uint32_t weight_format;
    uint32_t bits_per_weight;
    uint64_t layers;
};
static const struct tensor_rule layer_rules[DS41F_TENSOR_LAYER_COUNT] = {
	[DS41F_TENSOR_ATTENTION_NORM] = { "attn_norm.weight",
				      BF16,
				      { 5120, 0 },
				      PLAIN,
				      0,
				      WALKED },
	[DS41F_TENSOR_FFN_NORM] = { "ffn_norm.weight",
				BF16,
				{ 5120, 0 },
				PLAIN,
				0,
				WALKED },
	[DS41F_TENSOR_ATTENTION_MIX] = { "hc_attn_fn",
				     F32,
				     { 24, 20480 },
				     PLAIN,
				     0,
				     WALKED },
	[DS41F_TENSOR_ATTENTION_MIX_BASE] = { "hc_attn_base",
					  F32,
					  { 24, 0 },
					  PLAIN,
					  0,
					  WALKED },
	[DS41F_TENSOR_ATTENTION_MIX_SCALE] = { "hc_attn_scale",
					   F32,
					   { 3, 0 },
					   PLAIN,
					   0,
					   WALKED },
	[DS41F_TENSOR_FFN_MIX] = { "hc_ffn_fn",
			       F32,
			       { 24, 20480 },
			       PLAIN,
			       0,
			       WALKED },
	[DS41F_TENSOR_FFN_MIX_BASE] = { "hc_ffn_base",
				    F32,
				    { 24, 0 },
				    PLAIN,
				    0,
				    WALKED },
	[DS41F_TENSOR_FFN_MIX_SCALE] = { "hc_ffn_scale",
				     F32,
				     { 3, 0 },
				     PLAIN,
				     0,
				     WALKED },
	[DS41F_TENSOR_SINK] = { "attn.attn_sink", F32, { 64, 0 }, PLAIN, 0, WALKED },
	[DS41F_TENSOR_QUERY_DOWN] = { "attn.wq_a.weight",
				  U8,
				  { 1280, 5120 },
				  E4M3,
				  8,
				  WALKED },
	[DS41F_TENSOR_QUERY_NORM] = { "attn.q_norm.weight",
				  BF16,
				  { 1280, 0 },
				  PLAIN,
				  0,
				  WALKED },
	[DS41F_TENSOR_QUERY_UP] = { "attn.wq_b.weight",
				U8,
				{ 32768, 1280 },
				E4M3,
				8,
				WALKED },
	[DS41F_TENSOR_WINDOW] = { "attn.wkv.weight",
			      U8,
			      { 512, 5120 },
			      E4M3,
			      8,
			      WALKED },
	[DS41F_TENSOR_WINDOW_NORM] = { "attn.kv_norm.weight",
				   BF16,
				   { 512, 0 },
				   PLAIN,
				   0,
				   WALKED },
	[DS41F_TENSOR_OUTPUT_GROUPS] = { "attn.wo_a.weight",
				     U8,
				     { 8192, 4096 },
				     E4M3,
				     8,
				     WALKED },
	[DS41F_TENSOR_OUTPUT] = { "attn.wo_b.weight",
			      U8,
			      { 5120, 8192 },
			      E4M3,
			      8,
			      WALKED },
	[DS41F_TENSOR_ROUTER] = { "ffn.gate.weight",
			      BF16,
			      { 384, 5120 },
			      PLAIN,
			      0,
			      EVERY },
	[DS41F_TENSOR_ROUTER_BIAS] = { "ffn.gate.bias",
				   F32,
				   { 384, 0 },
				   PLAIN,
				   0,
				   EVERY },
	[DS41F_TENSOR_SHARED_GATE] = { "ffn.shared_experts.w1.weight",
				   U8,
				   { 2304, 5120 },
				   E4M3,
				   8,
				   WALKED },
	[DS41F_TENSOR_SHARED_UP] = { "ffn.shared_experts.w3.weight",
				 U8,
				 { 2304, 5120 },
				 E4M3,
				 8,
				 WALKED },
	[DS41F_TENSOR_SHARED_DOWN] = { "ffn.shared_experts.w2.weight",
				   U8,
				   { 5120, 2304 },
				   E4M3,
				   8,
				   WALKED },
	[DS41F_TENSOR_COMPRESSOR] = { "attn.compressor.wkv.weight",
				  BF16,
				  { 512, 5120 },
				  PLAIN,
				  0,
				  DS41F_SOURCE_LAYERS },
	[DS41F_TENSOR_COMPRESSOR_GATE] = { "attn.compressor.wgate.weight",
				       BF16,
				       { 512, 5120 },
				       PLAIN,
				       0,
				       DS41F_POOLED_LAYERS },
	[DS41F_TENSOR_COMPRESSOR_NORM] = { "attn.compressor.norm.weight",
				       BF16,
				       { 512, 0 },
				       PLAIN,
				       0,
				       DS41F_SOURCE_LAYERS },
	[DS41F_TENSOR_KEY] = { "attn.indexer.wk.weight",
			   BF16,
			   { 128, 512 },
			   PLAIN,
			   0,
			   DS41F_SOURCE_LAYERS },
	[DS41F_TENSOR_KEY_NORM] = { "attn.indexer.k_norm.weight",
				BF16,
				{ 128, 0 },
				PLAIN,
				0,
				DS41F_SOURCE_LAYERS },
	[DS41F_TENSOR_INDEX_QUERY_UP] = { "attn.indexer.wq_b.weight",
				      U8,
				      { 4096, 1280 },
				      E4M3,
				      8,
				      DS41F_INDEX_LAYERS },
	[DS41F_TENSOR_INDEX_WEIGHTS] = { "attn.indexer.weights_proj.weight",
				     BF16,
				     { 32, 5120 },
				     PLAIN,
				     0,
				     DS41F_INDEX_LAYERS },
	[DS41F_TENSOR_ENGRAM] = { "engram.wkv.weight",
			      U8,
			      { 25600, 6144 },
			      E4M3,
			      8,
			      DS41F_ENGRAM_LAYERS },
	[DS41F_TENSOR_ENGRAM_QUERY] = { "engram.q_weight",
				    BF16,
				    { 4, 5120 },
				    PLAIN,
				    0,
				    DS41F_ENGRAM_LAYERS },
	[DS41F_TENSOR_ENGRAM_KEY] = { "engram.k_weight",
				  BF16,
				  { 4, 5120 },
				  PLAIN,
				  0,
				  DS41F_ENGRAM_LAYERS },
};
static const struct tensor_rule expert_rules[DS41F_TENSOR_EXPERT_COUNT] = {
	[DS41F_TENSOR_GATE] = { "w1.weight", U8, { 2304, 5120 }, E2M1, 4, EVERY },
	[DS41F_TENSOR_UP] = { "w3.weight", U8, { 2304, 5120 }, E2M1, 4, EVERY },
	[DS41F_TENSOR_DOWN] = { "w2.weight", U8, { 5120, 2304 }, E2M1, 4, EVERY },
};
static const struct tensor_rule global_rules[DS41F_TENSOR_GLOBALS] = {
	[DS41F_TENSOR_EMBEDDING] = { "embed.weight",
				 BF16,
				 { 129280, 5120 },
				 PLAIN,
				 0,
				 EVERY },
	[DS41F_TENSOR_NORM] = { "norm.weight", BF16, { 5120, 0 }, PLAIN, 0, EVERY },
	[DS41F_TENSOR_HEAD] = { "head.weight",
			    BF16,
			    { 129280, 5120 },
			    PLAIN,
			    0,
			    EVERY },
};
/* A draft stage's router in place of a layer's (CFG:11; its bias_vl is not read: drafts are text, REF:1129), and the
 * draft's globals (dspark_v1/MAP_A_draft_model.md 2.1). */
static const struct tensor_rule draft_router_rules[DS41F_TENSOR_ROUTER_BIAS -
						   DS41F_TENSOR_ROUTER + 1u] = {
	{ "ffn.gate.weight",
	  BF16,
	  { 128, 5120 },
	  PLAIN,
	  0,
	  DS41F_DRAFT_LAYERS },
	{ "ffn.gate.bias", F32, { 128, 0 }, PLAIN, 0, DS41F_DRAFT_LAYERS },
};
static const struct tensor_rule draft_rules[DS41F_TENSOR_DRAFT_GLOBALS] = {
	[DS41F_TENSOR_DRAFT_MAIN_PROJECTION] = { "mtp.0.main_proj.weight",
					     U8,
					     { 5120, 15360 },
					     E4M3,
					     8,
					     EVERY },
	[DS41F_TENSOR_DRAFT_MAIN_NORM] = { "mtp.0.main_norm.weight",
				       BF16,
				       { 5120, 0 },
				       PLAIN,
				       0,
				       EVERY },
	[DS41F_TENSOR_DRAFT_NORM] = { "mtp.2.norm.weight",
				  BF16,
				  { 5120, 0 },
				  PLAIN,
				  0,
				  EVERY },
	[DS41F_TENSOR_DRAFT_MARKOV_EMBEDDING] = { "mtp.2.markov_head.embed.weight",
					      BF16,
					      { 129280, 256 },
					      PLAIN,
					      0,
					      EVERY },
	[DS41F_TENSOR_DRAFT_MARKOV_HEAD] = { "mtp.2.markov_head.head.weight",
					 BF16,
					 { 129280, 256 },
					 PLAIN,
					 0,
					 EVERY },
	[DS41F_TENSOR_DRAFT_CONFIDENCE] = { "mtp.2.confidence_head.proj.weight",
					BF16,
					{ 1, 5376 },
					PLAIN,
					0,
					EVERY },
};
#define ARRAY(array, row_bytes, fixed_bytes, holder, rows)  \
	[DS41F_ARRAY_##array] = { (uint64_t)(row_bytes), \
				     (uint64_t)(fixed_bytes), holder, rows },
static const struct array_rule {
    uint64_t row_bytes;
    uint64_t fixed_bytes;
    uint32_t holder;
    uint32_t rows; /* enum ds41f_rows */
} array_rules[DS41F_ARRAY_COUNT] = { DS41F_ARRAYS(ARRAY) };
#undef ARRAY
_Static_assert(
	DS41F_ARRAY_FREQUENCIES == 0 && DS41F_TENSOR_GATE == 0 &&
		DS41F_TENSOR_UP == 1 &&
		DS41F_TENSOR_EXPERT_COUNT == DS41F_EXPERT_MATRICES &&
		EVERY == BIT(DS41F_LAYERS) - 1u &&
		DS41F_ARRAY_ALIGNMENT_BYTES % PACK_DIGEST_WORD_BYTES == 0 &&
		PACK_DIGEST_WORD_BYTES % FILL_WORD_BYTES == 0,
	"the image is the expert tables, then the frequency words, whole words of a digest and of a fill; a "
	"table's row is gate, up, down");
_Static_assert(DS41F_WIDEST_ROW >= DS41F_ENGRAM_WIDTH &&
		       DS41F_WIDEST_ROW >= DS41F_HIDDEN &&
		       DS41F_ENGRAM_WIDTH ==
			       DS41F_ENGRAM_ROWS_PER_TOKEN *
				       ENGRAM_VALUE_BYTES_VALUE,
	       "every quantized row of a layer fits the row arrays");
_Static_assert(
	DS41F_WALK_LAYERS == DS41F_DRAFT_FIRST_LAYER + DS41F_DRAFT_STAGES &&
		DS41F_DRAFT_LAYERS == BIT(DS41F_WALK_LAYERS) -
						 BIT(DS41F_DRAFT_FIRST_LAYER) &&
		DS41F_DRAFT_TAP_LAYERS ==
			BIT(DS41F_LAYERS) - BIT(DS41F_DRAFT_FIRST_TAP_LAYER) &&
		DS41F_EMBEDDING_COPY_PIECE_BYTES % PACK_DIGEST_WORD_BYTES ==
			0,
	"the draft's stages follow the 40 layers, its taps are their last; the copy moves whole digest words");

static uint64_t sum(uint64_t first, uint64_t second)
{
    return first > UINT64_MAX - second ? UINT64_MAX : first + second;
}

static uint64_t product(uint64_t first, uint64_t second)
{
    return second != 0 && first > UINT64_MAX / second ? UINT64_MAX : first * second;
}

/* The latest layer of mask at or before layer, or DS41F_LAYERS for none. */
static uint32_t latest_of(uint64_t mask, uint32_t layer)
{
	uint32_t latest = DS41F_LAYERS;
	for (uint32_t earlier = 0; earlier <= layer; earlier++) {
		latest = (mask & BIT(earlier)) != 0 ? earlier : latest;
	}
    return latest;
}

MUST_CHECK static result_status model_check(const struct pack_record *pack)
{
	const char *const name = pack_header_of(pack)->model_name;
	if (strncmp(name, DS41F_MODEL_NAME, sizeof(DS41F_MODEL_NAME)) !=
	    0) {
		LOG_ERROR(
			"the pack names another model than " DS41F_MODEL_NAME);
		return STATUS_UNSUPPORTED;
	}
	return STATUS_OK;
}

MUST_CHECK static result_status limits_check(const struct program_limits *limits)
{
    const uint32_t context = limits->context_token_limit;
    const uint32_t width = (uint32_t)limits->state_width;
    const uint32_t draft = limits->draft_positions;     /* the draft block's rows take a slab's arrays */
    const bool valid =
	    context != 0 && context <= DS41F_CONTEXT_MOST &&
	    context % DS41F_CONTEXT_GRANULE == 0 && limits->row_capacity != 0 &&
	    limits->row_capacity <= STEP_WIDE_MOST_ROWS &&
	    limits->session_capacity != 0 &&
	    (width == STATE_WIDTH_8 || width == STATE_WIDTH_16) &&
	    (uint32_t)limits->linear_profile <= LINEAR_PROFILE_SPEED &&
	    (uint32_t)limits->linear_order <= LINEAR_ORDER_REFERENCE &&
	    (uint32_t)limits->cmp_order <= CMP_ORDER_RELAXED &&
	    (limits->official_hc_rows == 0 || limits->official_hc_rows == 1 ||
	     limits->official_hc_rows == DS41F_OFFICIAL_HC_ODD_PROMPT_ROWS ||
	     limits->official_hc_rows == DS41F_STREAMS ||
	     limits->official_hc_rows == LINEAR_MAXIMUM_ROWS) &&
	    (draft == 0 ||
	     (draft >= DS41F_DRAFT_BLOCK && draft <= STEP_DRAFT_MOST &&
	      draft <= ds41f_slab_rows(limits->row_capacity)));
    if (!valid || width != STATE_WIDTH_16) {
	    LOG_ERROR("limits %s: %" PRIu32 " tokens, %" PRIu32
		      " rows, %" PRIu32 " sessions, %" PRIu32
		      " bits, profile %" PRIu32 ", order %" PRIu32
		      ", CMP order %" PRIu32 ", %" PRIu32 " draft positions",
		      valid ? "beyond this build" : "the model has none of",
		      context, limits->row_capacity, limits->session_capacity,
		      width, (uint32_t)limits->linear_profile,
		      (uint32_t)limits->linear_order,
		      (uint32_t)limits->cmp_order, draft);
	    return valid ? STATUS_UNSUPPORTED : STATUS_INVALID_ARGUMENT;
    }
    if (limits->official_reference > 1u ||
	(limits->official_reference != 0 &&
	 (limits->row_capacity != 1u || limits->session_capacity != 1u ||
	  limits->official_hc_rows != 1u || draft != 0 ||
	  limits->context_token_limit > DS41F_OFFICIAL_REFERENCE_CONTEXT_LIMIT)))
	    return STATUS_UNSUPPORTED;
    if (limits->official_hc_rows != 0 &&
	(draft != 0 || limits->row_capacity > LINEAR_MAXIMUM_ROWS ||
	 limits->linear_profile != LINEAR_PROFILE_PRECISION ||
	 limits->linear_order != LINEAR_ORDER_REFERENCE ||
	 limits->cmp_order != CMP_ORDER_REFERENCE))
	    return STATUS_UNSUPPORTED;
    return STATUS_OK;
}

/* The tensor of prefix and a rule's name, where the layout puts it. */
MUST_CHECK static result_status tensor_place(const struct pack_record *pack,
					     const struct load_layout *layout,
					     const char *prefix,
					     const struct tensor_rule *rule,
					     struct ds41f_tensor *place_out)
{
    char name[NAME_CAPACITY];
    const int length = snprintf(name, sizeof(name), "%s%s", prefix, rule->name);
    const uint32_t found =
	    length > 0 && length < NAME_CAPACITY ?
		    pack_find_tensor(pack, name, (uint32_t)length) :
		    PACK_NOT_FOUND;
    const struct pack_tensor *const tensor =
	    found == PACK_NOT_FOUND ? NULL : &pack_tensors_of(pack)[found];
    if (tensor == NULL || tensor->element_type != rule->element_type ||
        tensor->dimension_count != (rule->dimensions[1] == 0 ? 1u : MATRIX_DIMENSIONS) ||
        tensor->dimensions[0] != rule->dimensions[0] || tensor->dimensions[1] != rule->dimensions[1] ||
        tensor->weight_format != rule->weight_format || tensor->bits_per_weight != rule->bits_per_weight) {
	    LOG_ERROR("the pack holds no tensor %s of type %" PRIu32
		      ", %" PRIu32 " by %" PRIu32 ", format %" PRIu32,
		      name, rule->element_type, rule->dimensions[0],
		      rule->dimensions[1], rule->weight_format);
	    return STATUS_CORRUPT_PACK;
    }
    const uint32_t card = layout->card_of_extent[tensor->extent_index];
    const uint64_t offset = sum(layout->byte_offset_of_extent[tensor->extent_index], tensor->byte_offset_in_extent);
    if (card != PLACEMENT_IN_HOST_MEMORY &&
	(card >= layout->card_count || offset > layout->card_byte_count[card] ||
	 tensor->byte_count > layout->card_byte_count[card] - offset)) {
	    LOG_ERROR("the layout puts %s beyond the weights of card %" PRIu32,
		      name, card);
	    return STATUS_INVALID_ARGUMENT;
    }
    *place_out = (struct ds41f_tensor){
	    .stored = { offset, tensor->byte_count },
	    .weight_format = (enum weight_format_code)rule->weight_format,
	    .shape = { rule->dimensions[0],
		       rule->dimensions[1] == 0 ? 1u : rule->dimensions[1],
		       rule->bits_per_weight, 0 },
	    .card = card
    };
    return STATUS_OK;
}

/* Where a card's image begins: behind its slots. */
static uint64_t image_offset_of(const struct ds41f_card *card)
{
    return card->state_offset + card->slots_byte_count;
}

/* Takes the bytes of an array at the next free offset: PROGRAM_NOWHERE for none. */
static uint64_t take(uint64_t *next, uint64_t byte_count)
{
    const uint64_t offset = *next;
    const uint64_t whole = sum(byte_count, DS41F_ARRAY_ALIGNMENT_BYTES - 1u);
    const uint64_t aligned =
	    whole == UINT64_MAX ?
		    UINT64_MAX :
		    whole - whole % DS41F_ARRAY_ALIGNMENT_BYTES;
    *next = byte_count == 0 ? offset : sum(offset, aligned);
    return byte_count == 0 ? PROGRAM_NOWHERE : offset;
}

/* The bytes of an array (model_v41.h) on a card of those holders: its kind of rows, but those sized apart. A chunk of
 * one slab runs no wide select and needs no views: their arrays take none. */
static uint64_t array_bytes_of(const struct program_limits *limits,
			       bool chunk_attention_arrays, uint32_t array,
			       uint64_t index_entries, uint64_t engram_tables,
			       uint32_t holders, bool reads_candidates)
{
    const struct array_rule *const rule = &array_rules[array];
    const uint64_t rows = limits->row_capacity,
		   slab_rows = ds41f_slab_rows(limits->row_capacity);
    const uint64_t slabs = (rows + slab_rows - 1u) / slab_rows,
		   thin_jobs = slab_rows * DS41F_EXPERTS_PER_ROW;
    const uint64_t wide_jobs = (uint64_t)ROUTER_WIDE_JOBS(limits->row_capacity);
    const enum ds41f_rows kind = ds41f_array_rows(chunk_attention_arrays,
					      (enum ds41f_array)array,
					      (enum ds41f_rows)rule->rows);
    const uint64_t kind_rows =
	    kind == DS41F_ROWS_DRAFT ? limits->draft_positions :
	    kind == DS41F_ROWS_SLAB	   ? slab_rows :
	    kind == DS41F_ROWS_SPAN	   ? ds41f_span_rows(limits->row_capacity) :
				     rows;
    const uint64_t rowed = sum(rule->fixed_bytes, kind_rows * rule->row_bytes);
    const bool is_wide = slabs > 1u;
    switch (array) {
    case DS41F_ARRAY_COMPRESSOR_KV: /* a 1-to-1 source's product alone: bfloat16 */
	    return (holders & BIT(DS41F_HOLDER_POOLED)) != 0 ?
			   rowed :
			   rows * DS41F_HEAD_DIMS * DS41F_BF16;
    case DS41F_ARRAY_SESSIONS:
	    return product(limits->session_capacity, STEP_SESSION_BYTES);
    case DS41F_ARRAY_INDEX_SCORES:
	    /* Candidate readers scan a fixed score extent even when their source keys have a shorter context. */
	    if ((limits->official_reference != 0 || reads_candidates) &&
		index_entries < DS41F_CONTEXT_WITHOUT_CANDIDATES)
		    index_entries = DS41F_CONTEXT_WITHOUT_CANDIDATES;
	    /* Each prepared index description names its score payload. Split selection uses the annex immediately
         * after that payload; smaller layers may reuse the unused tail of this largest score allocation. Slabs,
         * layers and ordinary/candidate stages run serially on a card, so one annex suffices. */
	    return slab_rows * (index_entries * DS41F_BF16 +
				(index_entries >= DS41F_INDEX_SPLIT_MIN_ENTRIES ?
					 DS41F_INDEX_SPLIT_ROW_BYTES :
					 0u));
    case DS41F_ARRAY_ENGRAM_ROWS:
	    return engram_tables * rows * DS41F_ENGRAM_ROWS_PER_TOKEN *
		   ENGRAM_ROW_BYTES;
    case DS41F_ARRAY_JOBS:
	    return (wide_jobs > thin_jobs ? wide_jobs : thin_jobs) *
		   EXPERT_JOB_BYTES;
    case DS41F_ARRAY_ROUTER_COUNTS:
	    return is_wide ? (rows + ROUTER_SORT_ROWS - 1u) / ROUTER_SORT_ROWS *
				     DS41F_EXPERTS * sizeof(uint32_t) :
			     0;
    case DS41F_ARRAY_ROUTER_TILES:
	    return is_wide ? ROUTER_WIDE_TILES(rows) *
				     sizeof(struct wide_expert_tile) :
			     0;
    case DS41F_ARRAY_VIEWS: /* the slabs' views, then split_overlap's two halves' (ds41f_tp_stage.h) */
	    return is_wide ? slabs * (STEP_BLOCK_BYTES_VALUE +
				      slab_rows * STEP_ROW_BYTES_VALUE) +
				     (uint64_t)(DS41F_HALF_VIEWS *
						STEP_BLOCK_BYTES_VALUE) +
				     rows * STEP_ROW_BYTES_VALUE :
			     0;
    case DS41F_ARRAY_ROUTER_PICKS:
    case DS41F_ARRAY_ROUTER_PAIRS:
    case DS41F_ARRAY_ROUTER_RUNS:
	    return is_wide ? rowed : 0;
    default:
        return rowed;
    }
}

/* Lays out a card's state from state_offset on (model_v41.h); false when it does not fit 64 bits. A draft stage reads
 * no pool, selection or keys and has no n-gram table: it takes its table, ring and prompt-checkpoint window. */
static bool card_lay_out_attention(struct ds41f_plan *plan,
				   struct ds41f_card *card,
				   uint64_t attention_layers)
{
    const uint64_t rows = plan->limits.row_capacity;
    const uint64_t sessions = plan->limits.session_capacity;
    const uint64_t walked = ds41f_walk_layers(card);
    card->slots_byte_count = product(card->slot_count, plan->slot_byte_count);
    uint64_t next = sum(card->state_offset, card->slots_byte_count);
    uint64_t sources_read = 0, sources_scored = 0, groups_read = 0, index_entries = 0, engram_tables = 0;
    for (uint32_t layer = 0; layer < DS41F_WALK_LAYERS; layer++) {
	    const uint64_t on_card = card->layers & BIT(layer);
	    const uint64_t attention_on_card = attention_layers & on_card;
	    const uint32_t source = latest_of(DS41F_SOURCE_LAYERS, layer);
	    const uint32_t group = latest_of(DS41F_INDEX_LAYERS, layer);
	    sources_read |= attention_on_card != 0 && source != DS41F_LAYERS ?
				    BIT(source) :
				    0;
	    groups_read |= attention_on_card != 0 && group != DS41F_LAYERS ?
				   BIT(group) :
				   0;
	    if ((on_card & DS41F_INDEX_LAYERS) != 0 &&
		source < DS41F_LAYERS) {
		    sources_scored |= attention_on_card != 0 ? BIT(source) : 0;
		    index_entries =
			    plan->key_session_bytes[source] / DS41F_KEY_BYTES >
					    index_entries ?
				    plan->key_session_bytes[source] /
					    DS41F_KEY_BYTES :
				    index_entries;
	    }
	    engram_tables += (on_card & DS41F_ENGRAM_LAYERS) != 0 ? 1u : 0u;
	    card->expert_table_offset[layer] =
		    take(&next, (walked & BIT(layer)) != 0 ?
					DS41F_EXPERT_OFFSETS_BYTES :
					0);
    }
    for (uint32_t array = 0; array < DS41F_ARRAY_COUNT; array++) {
	    const uint64_t bytes = array_bytes_of(
		    &plan->limits, plan->chunk_attention_arrays, array,
		    index_entries, engram_tables, card->holders,
		    (card->layers & DS41F_CANDIDATE_READERS) != 0);
	    card->array_offset[array] =
		    take(&next,
			 (card->holders & BIT(array_rules[array].holder)) != 0 ?
				 bytes :
				 0);
	    card->image_byte_count = array == DS41F_ARRAY_FREQUENCIES &&
						     card->layers != 0 ?
					     next - image_offset_of(card) :
					     card->image_byte_count;
    }
    card->official_scratch_offset =
	    plan->limits.official_reference != 0 ?
		    take(&next, DS41F_OFFICIAL_SCRATCH_BYTES) :
		    PROGRAM_NOWHERE;
    for (uint32_t layer = 0; layer < DS41F_WALK_LAYERS; layer++) {
	    const uint64_t bit = BIT(layer);
	    const bool on_card = (attention_layers & bit) != 0;
	    card->selection_offset[layer] =
		    take(&next, (groups_read & bit) != 0 ?
					rows * DS41F_SELECTION_ROW_BYTES :
					0);
	    card->ring_offset[layer] =
		    take(&next, on_card ? product(sessions,
						  DS41F_RING_SESSION_BYTES) :
					  0);
	    card->checkpoint_offset[layer] =
		    take(&next,
			 on_card ? product(sessions,
					   DS41F_CHECKPOINT_SESSION_BYTES) :
				   0);
	    card->compressor_offset[layer] = take(
		    &next, on_card && (DS41F_POOLED_LAYERS & bit) != 0 ?
				   product(sessions,
					   DS41F_COMPRESSOR_SESSION_BYTES) :
				   0);
	    card->pool_offset[layer] = take(
		    &next,
		    (sources_read & bit) != 0 ?
			    product(sessions, plan->pool_session_bytes[layer]) :
			    0);
	    card->key_tiles_offset[layer] = take(
		    &next,
		    (sources_scored & bit) != 0 ?
			    product(sessions, plan->key_session_bytes[layer]) :
			    0);
    }
    card->front_scratch_offset = PROGRAM_NOWHERE;
    card->producer_arrivals_offset = PROGRAM_NOWHERE;
    if (card->layers != 0) {
	    if (rows >= STEP_WIDE_LEAST_ROWS) {
		    const uint64_t first =
			    card->array_offset[DS41F_ARRAY_ROUTER_PICKS];
		    const uint64_t end =
			    card->array_offset[DS41F_ARRAY_JOBS];
		    if (first == PROGRAM_NOWHERE || end < first ||
			end - first < DS41F_FUSED_SCRATCH_BYTES) {
			    return false;
		    }
		    card->front_scratch_offset = first;
	    } else {
		    card->front_scratch_offset =
			    take(&next, DS41F_FUSED_SCRATCH_BYTES);
	    }
	    card->producer_arrivals_offset =
		    sum(card->front_scratch_offset, DS41F_SITE_FRONT_SCRATCH_BYTES);
    }
    bool has_exl3 = false;
    for (uint32_t layer = 0; layer < DS41F_LAYERS; layer++)
	    has_exl3 = has_exl3 || ((card->layers & BIT(layer)) != 0 &&
				    plan->exl3_bits[layer] != 0);
    card->exl3_scratch_offset =
	    has_exl3 ?
		    take(&next,
			 exl3_scratch_byte_count(plan->limits.row_capacity *
							 DS41F_EXPERTS_PER_ROW,
						 DS41F_HIDDEN)) :
		    PROGRAM_NOWHERE;
    card->arena_byte_count = next;
    return next != UINT64_MAX;
}

static bool card_lay_out(struct ds41f_plan *plan, struct ds41f_card *card)
{
	return card_lay_out_attention(plan, card, ds41f_walk_layers(card));
}

/* The prefix of a walk layer's tensor names: "layers.<l>.", a draft stage's "mtp.<l - 40>.". */
static void layer_prefix_of(uint32_t layer, char prefix[LAYER_PREFIX_CAPACITY])
{
	const bool is_draft = layer >= DS41F_DRAFT_FIRST_LAYER;
	(void)snprintf(prefix, LAYER_PREFIX_CAPACITY, "%s%" PRIu32 ".",
		       is_draft ? "mtp." : "layers.",
		       is_draft ? layer - DS41F_DRAFT_FIRST_LAYER : layer);
}

/* The routed experts of a walk layer, before the cards are laid out: the entries of its expert table, those on its
 * card spanning plan->experts, those in host memory numbered by tag behind the slots at the end of the card's weights.
 * A draft stage's lie on its card: the draft runs no expert pause. */
MUST_CHECK static result_status experts_place(const struct pack_record *pack,
					      const struct load_layout *layout,
					      struct ds41f_plan *plan,
					      uint32_t layer)
{
    const uint32_t card = plan->layer_card[layer];
    const bool is_draft = layer >= DS41F_DRAFT_FIRST_LAYER;
    uint64_t first = UINT64_MAX, end = 0;
    char layer_prefix[LAYER_PREFIX_CAPACITY], prefix[NAME_CAPACITY];
    layer_prefix_of(layer, layer_prefix);
    for (uint32_t expert = 0;
	 expert < (is_draft ? DS41F_DRAFT_EXPERTS : DS41F_EXPERTS); expert++) {
	    struct ds41f_tensor places[DS41F_EXPERT_MATRICES];
	    uint64_t *const entries = plan->expert_tables[layer][expert];
	    uint64_t low = UINT64_MAX, high = 0, own = 0;
	    uint32_t on_card = 0, in_host = 0;
	    (void)snprintf(prefix, sizeof(prefix), "%sffn.experts.%" PRIu32 ".",
			   layer_prefix, expert);
	    for (uint32_t matrix = 0; matrix < DS41F_EXPERT_MATRICES;
		 matrix++) {
		    const result_status status = tensor_place(
			    pack, layout, prefix, &expert_rules[matrix],
			    &places[matrix]);
		    if (status != STATUS_OK) {
			    return status;
		    }
		    const struct linear_range *const stored =
			    &places[matrix].stored;
		    on_card += places[matrix].card == card ? 1u : 0u;
		    in_host += places[matrix].card == PLACEMENT_IN_HOST_MEMORY ?
				       1u :
				       0u;
		    low = stored->byte_offset < low ? stored->byte_offset : low;
		    high = sum(stored->byte_offset, stored->byte_count) > high ?
				   sum(stored->byte_offset,
				       stored->byte_count) :
				   high;
		    own = sum(own, sum(stored->byte_count,
				       WEIGHT_FORMAT_MATRIX_ALIGNMENT_BYTES));
	    }
	    const uint32_t tag = plan->host_expert_count[layer];
	    const bool is_on_card = on_card == DS41F_EXPERT_MATRICES;
	    const bool is_first_in_host = !is_on_card &&
					  plan->expert_byte_count == 0;
	    bool is_placed = is_on_card ||
			     (!is_draft && in_host == DS41F_EXPERT_MATRICES &&
			      high - low < own &&
			      high <= layout->host_byte_count &&
			      (is_first_in_host ||
			       plan->expert_byte_count == high - low));
	    for (uint32_t matrix = 0; matrix < DS41F_EXPERT_MATRICES;
		 matrix++) {
		    is_placed =
			    is_placed &&
			    (is_on_card || is_first_in_host ||
			     plan->expert_matrix_offsets[matrix] ==
				     places[matrix].stored.byte_offset - low);
	    }
	    if (!is_placed) {
		    LOG_ERROR(
			    "expert %" PRIu32 " of %s %" PRIu32 " is %s",
			    expert, is_draft ? "draft stage" : "layer",
			    is_draft ? layer - DS41F_DRAFT_FIRST_LAYER : layer,
			    is_draft ?
				    "not on the draft's card, and the draft runs no expert pause" :
				    "neither on its layer's card nor whole in host memory with its matrices "
				    "together as in the others");
		    return STATUS_UNSUPPORTED;
	    }
	    for (uint32_t matrix = 0; matrix < DS41F_EXPERT_MATRICES;
		 matrix++) {
		    entries[matrix] =
			    is_on_card ?
				    places[matrix].stored.byte_offset :
				    sum(layout->card_byte_count[card],
					(uint64_t)tag *
						LINEAR_OFFSET_ALIGNMENT_BYTES);
	    }
	    first = is_on_card && low < first ? low : first;
	    end = is_on_card && high > end ? high : end;
	    if (!is_on_card) {
		    for (uint32_t matrix = 0; matrix < DS41F_EXPERT_MATRICES;
			 matrix++) {
			    plan->expert_matrix_offsets[matrix] =
				    places[matrix].stored.byte_offset - low;
		    }
		    plan->host_expert_offsets[layer][tag] = low;
		    plan->host_expert_count[layer] = tag + 1u;
		    plan->expert_byte_count = high - low;
	    }
    }
    plan->experts[layer] = first == UINT64_MAX ?
				   (struct linear_range){ 0, 0 } :
				   (struct linear_range){ first, end - first };
    return STATUS_OK;
}

/* Where a layer keeps routed experts in host memory: its card's slots, and its experts' range to their end. */
static void slots_place(const struct load_layout *layout,
			struct ds41f_plan *plan)
{
	const uint64_t whole = sum(plan->expert_byte_count,
				   DS41F_ARRAY_ALIGNMENT_BYTES - 1u);
	plan->slot_byte_count = whole - whole % DS41F_ARRAY_ALIGNMENT_BYTES;
	for (uint32_t layer = 0; layer < DS41F_LAYERS; layer++) {
		struct ds41f_card *const card =
			&plan->cards[plan->layer_card[layer]];
		const uint64_t slots =
			layout->card_byte_count[plan->layer_card[layer]];
		if (plan->host_expert_count[layer] != 0) {
			card->slot_count =
				DS41F_EXPERTS_PER_ROW *
				ds41f_slab_rows(plan->limits.row_capacity);
			const uint64_t first =
				plan->experts[layer].byte_count == 0 ?
					slots :
					plan->experts[layer].byte_offset;
			plan->experts[layer] = (struct linear_range){
				first,
				sum(slots, product(card->slot_count,
						   plan->slot_byte_count)) -
					first
			};
		}
	}
}

/* The first tensor of the draft or of layers 37 to 39 off the head's card, which the admission's line names. */
struct apart_tensor {
    char name[NAME_CAPACITY];
    uint32_t card;
};
static void apart_note(struct apart_tensor *apart, const char *prefix,
		       const struct tensor_rule *rule,
		       const struct ds41f_tensor *place, uint32_t head_card)
{
    if (apart->name[0] == '\0' && place->stored.byte_count != 0 && place->card != head_card) {
        (void)snprintf(apart->name, sizeof(apart->name), "%s%s", prefix, rule->name);
        apart->card = place->card;
    }
}

/* The plan of a layout (model_v41.h): plan->limits is set, every other member zero. */
MUST_CHECK static result_status plan_build(const struct pack_record *pack,
					   const struct load_layout *layout,
					   struct ds41f_plan *plan)
{
	float words[DS41F_THETAS][DS41F_FREQUENCIES];
	const uint32_t card_count = layout->card_count;
	const bool has_draft = plan->limits.draft_positions != 0;
	const uint32_t walk_count = has_draft ? DS41F_WALK_LAYERS :
						DS41F_LAYERS;
	result_status status = STATUS_OK;
	uint32_t apart =
		DS41F_LAYERS; /* the first layer that is not on one card */
	struct apart_tensor draft_apart = { { 0 }, 0 };
	plan->card_count = card_count;
	char prefix[LAYER_PREFIX_CAPACITY];
	for (uint32_t tensor = 0;
	     tensor < DS41F_TENSOR_GLOBALS && status == STATUS_OK; tensor++) {
		status = tensor_place(pack, layout, "", &global_rules[tensor],
				      &plan->globals[tensor]);
	}
	const uint32_t head_card = plan->globals[DS41F_TENSOR_HEAD].card;
	for (uint32_t tensor = 0;
	     has_draft && tensor < DS41F_TENSOR_DRAFT_GLOBALS &&
	     status == STATUS_OK;
	     tensor++) {
		status = tensor_place(pack, layout, "", &draft_rules[tensor],
				      &plan->draft_globals[tensor]);
		apart_note(&draft_apart, "", &draft_rules[tensor],
			   &plan->draft_globals[tensor], head_card);
	}
	for (uint32_t layer = 0, table = 0;
	     layer < walk_count && status == STATUS_OK; layer++) {
		const bool is_draft = layer >= DS41F_DRAFT_FIRST_LAYER;
		const bool is_drafted =
			has_draft && (is_draft || (DS41F_DRAFT_TAP_LAYERS &
						   BIT(layer)) != 0);
		struct ds41f_tensor *const tensors = plan->layer_tensors[layer];
		layer_prefix_of(layer, prefix);
		for (uint32_t tensor = 0;
		     tensor < DS41F_TENSOR_LAYER_COUNT &&
		     status == STATUS_OK;
		     tensor++) {
			const struct tensor_rule *const rule =
				is_draft && tensor >= DS41F_TENSOR_ROUTER &&
						tensor <= DS41F_TENSOR_ROUTER_BIAS ?
					&draft_router_rules[tensor -
							    DS41F_TENSOR_ROUTER] :
					&layer_rules[tensor];
			status = (rule->layers & BIT(layer)) == 0 ?
					 STATUS_OK :
					 tensor_place(pack, layout, prefix,
						      rule, &tensors[tensor]);
			apart = !is_draft && layer < apart &&
						tensors[tensor].stored
								.byte_count !=
							0 &&
						tensors[tensor].card !=
							tensors[DS41F_TENSOR_ATTENTION_NORM]
								.card ?
					layer :
					apart;
			if (is_drafted) {
				apart_note(&draft_apart, prefix, rule,
					   &tensors[tensor], head_card);
			}
		}
		plan->layer_card[layer] = tensors[DS41F_TENSOR_ATTENTION_NORM].card;
		if (is_draft) {
			continue; /* its card is marked behind the checks */
		}
		apart = layer < apart && plan->layer_card[layer] >= card_count ?
				layer :
				apart;
		struct ds41f_card *const card =
			&plan->cards[apart < DS41F_LAYERS ?
					     0 :
					     plan->layer_card[layer]];
		card->layers |= BIT(layer);
		card->holders |= (uint32_t)(BIT(DS41F_HOLDER_TRUNK) | BIT(DS41F_HOLDER_LAYER));
		card->holders |=
			(uint32_t)((DS41F_INDEX_LAYERS & BIT(layer)) != 0 ?
					   BIT(DS41F_HOLDER_INDEX) :
					   0);
		card->holders |=
			(uint32_t)((DS41F_SOURCE_LAYERS & BIT(layer)) != 0 ?
					   BIT(DS41F_HOLDER_SOURCE) :
					   0);
		card->holders |=
			(uint32_t)((DS41F_POOLED_LAYERS & BIT(layer)) != 0 ?
					   BIT(DS41F_HOLDER_POOLED) :
					   0);
		card->holders |=
			(uint32_t)(layer == DS41F_CANDIDATE_SOURCE ||
						   (DS41F_CANDIDATE_READERS &
						    BIT(layer)) != 0 ?
					   BIT(DS41F_HOLDER_CANDIDATES) :
					   0);
		if ((DS41F_ENGRAM_LAYERS & BIT(layer)) != 0) {
			card->holders |= (uint32_t)BIT(DS41F_HOLDER_ENGRAM);
			card->engram_tables |= (uint32_t)BIT(table++);
		}
		if ((DS41F_SOURCE_LAYERS & BIT(layer)) != 0) {
			const uint64_t entries =
				plan->limits.context_token_limit /
				((DS41F_POOLED_LAYERS & BIT(layer)) != 0 ?
					 DS41F_POOLED_RATIO :
					 DS41F_DIRECT_RATIO);
			plan->pool_session_bytes[layer] =
				entries * DS41F_STATE_ROW_BYTES;
			plan->key_session_bytes[layer] =
				entries * DS41F_KEY_BYTES;
		}
	}
	plan->embedding_card = plan->globals[DS41F_TENSOR_EMBEDDING].card;
	plan->result_card = plan->globals[DS41F_TENSOR_HEAD].card;
	if (status == STATUS_OK && apart < DS41F_LAYERS) {
		LOG_ERROR("this build runs a layer on one card: layer %" PRIu32
			  " is not",
			  apart);
		status = STATUS_UNSUPPORTED;
	} else if (status == STATUS_OK &&
		   (plan->embedding_card >= card_count ||
		    plan->result_card >= card_count ||
		    plan->globals[DS41F_TENSOR_NORM].card != plan->result_card)) {
		LOG_ERROR(
			"this build runs the embedding on a card and the head with the norm on one card");
		status = STATUS_UNSUPPORTED;
	} else if (status == STATUS_OK && draft_apart.name[0] != '\0') {
		LOG_ERROR(
			"the draft needs layers 37 to 39, the head and the draft head on one card, but %s lies on "
			"card %" PRIu32 " and the head on card %" PRIu32
			": place them together (--layout, or "
			"--card_order for the fill)",
			draft_apart.name, draft_apart.card, plan->result_card);
		status = STATUS_UNSUPPORTED;
	}
	if (status != STATUS_OK) {
		return status;
	}
	status = ds41f_model_vision_weights(pack, layout, plan->layer_card,
				      &plan->vision);
	if (status != STATUS_OK ||
	    (plan->vision.enabled && plan->embedding_card != 0)) {
		return status != STATUS_OK ? status : STATUS_UNSUPPORTED;
	}
    plan->cards[plan->embedding_card].holders |= (uint32_t)BIT(DS41F_HOLDER_TRUNK);
    plan->cards[plan->result_card].holders |=
	    (uint32_t)(BIT(DS41F_HOLDER_TRUNK) | BIT(DS41F_HOLDER_RESULT) |
		       (has_draft ? BIT(DS41F_HOLDER_DRAFT) | BIT(DS41F_HOLDER_DRAFT_COPY) : 0));
    for (uint32_t layer = 0; layer < walk_count && status == STATUS_OK;
	 layer++) {
	    status = experts_place(pack, layout, plan, layer);
    }
    if (status == STATUS_OK) {
	    slots_place(layout, plan);
    }
    ds41f_frequency_words(words);
    for (uint32_t card_index = 0;
	 card_index < card_count && status == STATUS_OK; card_index++) {
	    struct ds41f_card *const card = &plan->cards[card_index];
	    card->state_offset = layout->card_byte_count[card_index];
	    const bool aligned =
		    card->state_offset % DS41F_ARRAY_ALIGNMENT_BYTES == 0;
	    if (!aligned || !card_lay_out(plan, card)) {
		    LOG_ERROR(
			    "card %" PRIu32 ": its weights end at byte %" PRIu64
			    ", %s %u",
			    card_index, card->state_offset,
			    aligned ?
				    "and no state fits 64 bits behind them on a multiple of" :
				    "not on a multiple of",
			    DS41F_ARRAY_ALIGNMENT_BYTES);
		    status = STATUS_INVALID_ARGUMENT;
	    }
	    if (status == STATUS_OK && card_index == 0 &&
		plan->vision.enabled) {
		    status = ds41f_model_vision_arena(&plan->vision,
						plan->limits.session_capacity,
						card->arena_byte_count,
						&card->arena_byte_count);
	    }
	    if (status == STATUS_OK && card->image_byte_count != 0) {
		    card->image = FAULT_POINT(model_v41_image_memory) ?
					  NULL :
					  calloc(1, card->image_byte_count);
		    status = card->image == NULL ? STATUS_OUT_OF_MEMORY :
						   STATUS_OK;
	    }
	    if (card->image != NULL) {
		    memcpy(card->image + (card->array_offset
						  [DS41F_ARRAY_FREQUENCIES] -
					  image_offset_of(card)),
			   words, sizeof(words));
	    }
	    for (uint32_t layer = 0;
		 layer < DS41F_WALK_LAYERS && card->image != NULL; layer++) {
		    if ((ds41f_walk_layers(card) & BIT(layer)) != 0) {
			    memcpy(card->image +
					   (card->expert_table_offset[layer] -
					    image_offset_of(card)),
				   plan->expert_tables[layer],
				   sizeof(plan->expert_tables[layer]));
		    }
	    }
    }
    return status;
}

result_status ds41f_describe(const struct pack_record *pack,
			   struct model_description_record *description_out)
{
    if (pack == NULL || description_out == NULL) {
	    LOG_ERROR("describe: a missing argument");
	    return STATUS_INVALID_ARGUMENT;
    }
    const result_status status = model_check(pack);
    if (status != STATUS_OK) {
	    return status;
    }
    *description_out = (struct model_description_record){
	    DS41F_LAYERS, DS41F_HIDDEN, DS41F_VOCABULARY_VALUE,
	    DS41F_CONTEXT_MOST, DS41F_END_TOKEN
    };
    return STATUS_OK;
}

result_status ds41f_plan_create(const struct pack_record *pack,
			      const struct load_layout *layout,
			      const struct program_limits *limits,
			      struct ds41f_plan **plan_out)
{
	struct ds41f_plan *plan = NULL;
	if (pack == NULL || layout == NULL || limits == NULL ||
	    plan_out == NULL) {
		LOG_ERROR("plan_create: a missing argument");
		return STATUS_INVALID_ARGUMENT;
	}
	result_status status = model_check(pack);
	status = status == STATUS_OK ? limits_check(limits) : status;
	if (status == STATUS_OK &&
	    (layout->extent_count != pack_header_of(pack)->extent_count ||
	     layout->card_count == 0 ||
	     layout->card_count > BACKEND_MAXIMUM_CARDS)) {
		LOG_ERROR("plan_create: a layout of %" PRIu32
			  " extents on %" PRIu32 " cards",
			  layout->extent_count, layout->card_count);
		status = STATUS_INVALID_ARGUMENT;
	}
	if (status == STATUS_OK) {
		plan = FAULT_POINT(model_v41_plan_memory) ?
			       NULL :
			       calloc(1, sizeof(*plan));
		status = plan == NULL ? STATUS_OUT_OF_MEMORY : STATUS_OK;
	}
	if (status == STATUS_OK) {
		plan->limits = *limits;
		status = plan_build(pack, layout, plan);
	}
	if (status == STATUS_OK) {
		*plan_out = plan;
	} else {
		ds41f_plan_destroy(plan);
	}
    return status;
}

void ds41f_plan_destroy(struct ds41f_plan *plan)
{
	for (uint32_t card = 0; plan != NULL && card < BACKEND_MAXIMUM_CARDS;
	     card++) {
		free(plan->cards[card].image);
	}
    free(plan);
}

/* Load-only layer inventory. Sorting carries the destination descriptor with its original source extent. */
struct tp2_pending_tensor {
	struct tp_load_job job;
	struct linear_range *stored;
	uint64_t *expert_offset;
	uint64_t byte_count;
	uint32_t extent;
};

static int tp2_pending_compare(const void *first, const void *second)
{
    const struct tp2_pending_tensor *const left = first, *const right = second;
    if (left->extent != right->extent)
        return left->extent < right->extent ? -1 : 1;
    return left->job.tensor_index < right->job.tensor_index ? -1 :
           left->job.tensor_index > right->job.tensor_index ? 1 : 0;
}

MUST_CHECK static result_status
tp2_tensor_find(const struct pack_record *pack, const char *name,
		const struct tensor_rule *rule, uint32_t card, uint32_t rank,
		uint32_t rank_count, bool split_outputs, bool split_inputs,
		bool balanced_shards, struct ds41f_tensor *tensor_out,
		struct tp2_pending_tensor *pending)
{
	const uint32_t found =
		pack_find_tensor(pack, name, (uint32_t)strlen(name));
	const struct pack_tensor *const source =
		found == PACK_NOT_FOUND ? NULL : &pack_tensors_of(pack)[found];
	if (source == NULL || source->element_type != rule->element_type ||
	    source->dimension_count !=
		    (rule->dimensions[1] == 0 ? 1u : MATRIX_DIMENSIONS) ||
	    source->dimensions[0] != rule->dimensions[0] ||
	    source->dimensions[1] != rule->dimensions[1] ||
	    source->weight_format != rule->weight_format ||
	    source->bits_per_weight != rule->bits_per_weight)
		return STATUS_CORRUPT_PACK;
	*pending = (struct tp2_pending_tensor){
		.job = { .tensor_index = found,
			 .card = card,
			 .mode = TP_LOAD_FULL },
		.stored = tensor_out == NULL ? NULL : &tensor_out->stored,
		.byte_count = source->byte_count,
		.extent = source->extent_index
	};
	struct matrix_shape_record shape = { source->dimensions[0],
					     source->dimensions[1] == 0 ?
						     1u :
						     source->dimensions[1],
					     source->bits_per_weight, 0 };
	if (rank_count > 1u && (split_outputs || split_inputs)) {
		/* a plain matrix (bfloat16): row shards, its element bits in the description (tp_weights.h) */
		struct matrix_shape_record source_shape = shape;
		source_shape.bits_per_weight =
			source->weight_format == WEIGHT_FORMAT_NONE ?
				TENSOR_SHARD_PLAIN_BITS :
				shape.bits_per_weight;
		const struct tensor_shard_description description = {
			.format =
				(enum weight_format_code)source->weight_format,
			.source_shape = source_shape,
			.first_output =
				split_outputs ?
					rank * shape.output_size / rank_count :
					0,
			.output_count = split_outputs ?
						shape.output_size / rank_count :
						shape.output_size,
			.first_input = split_inputs ? rank * shape.input_size /
							      rank_count :
						      0,
			.input_count = split_inputs ?
					       shape.input_size / rank_count :
					       shape.input_size
		};
		struct tensor_shard_description selected = description;
		if (rank_count == DS41F_TP4_RANKS &&
		    (!balanced_shards ||
		     source->weight_format == WEIGHT_FORMAT_EXL3)) {
			const uint32_t first =
				rank / DS41F_TP2_RANKS *
					(DS41F_TP4_SMALL_INNER + DS41F_TP4_LARGE_INNER) +
				rank % DS41F_TP2_RANKS * DS41F_TP4_SMALL_INNER;
			const uint32_t count = rank % DS41F_TP2_RANKS == 0 ?
						       DS41F_TP4_SMALL_INNER :
						       DS41F_TP4_LARGE_INNER;
			if (split_outputs &&
			    shape.output_size == DS41F_EXPERT_INNER) {
				selected.first_output = first;
				selected.output_count = count;
			}
			if (split_inputs &&
			    shape.input_size == DS41F_EXPERT_INNER) {
				selected.first_input = first;
				selected.input_count = count;
			}
		}
		const result_status status =
			tensor_shard_prepare(&selected, &pending->job.shard);
		if (status != STATUS_OK)
			return status;
		pending->job.mode = TP_LOAD_SHARD;
		pending->byte_count = pending->job.shard.shard_byte_count;
		shape = pending->job.shard.shard_shape;
		shape.bits_per_weight =
			source->bits_per_weight; /* a plain matrix keeps its pack's figure */
	}
    if (tensor_out != NULL)
	    *tensor_out = (struct ds41f_tensor){
		    .stored = { 0, pending->byte_count },
		    .shape = shape,
		    .weight_format =
			    (enum weight_format_code)source->weight_format,
		    .card = card
	    };
    return STATUS_OK;
}

result_status ds41f_tp2_layer_choices_create(
	const struct pack_record *pack, const struct program_limits *limits,
	uint32_t card_index, uint32_t rank, uint32_t rank_count,
	uint64_t arena_prefix_bytes, bool replicated_attention,
	bool partitioned_experts, struct ds41f_tp2_layer_plan **plan_out)
{
	if (pack == NULL || limits == NULL || plan_out == NULL ||
	    card_index >= BACKEND_MAXIMUM_CARDS ||
	    (rank_count != 1u && rank_count != DS41F_TP2_RANKS) ||
	    rank >= rank_count ||
	    (partitioned_experts && rank_count != DS41F_TP2_RANKS) ||
	    arena_prefix_bytes % DS41F_ARRAY_ALIGNMENT_BYTES != 0 ||
	    limits->row_capacity > LINEAR_MAXIMUM_ROWS ||
	    limits->draft_positions != 0 ||
	    limits->cmp_order != CMP_ORDER_REFERENCE)
		return STATUS_INVALID_ARGUMENT;
	result_status status = model_check(pack);
	status = status == STATUS_OK ? limits_check(limits) : status;
	struct ds41f_tp2_layer_plan *const result =
		status == STATUS_OK ? calloc(1, sizeof(*result)) : NULL;
	struct tp2_pending_tensor *const pending =
		result == NULL ? NULL :
				 calloc(DS41F_TP2_LAYER_JOBS, sizeof(*pending));
	if (status == STATUS_OK && (result == NULL || pending == NULL))
		status = STATUS_OUT_OF_MEMORY;
	struct ds41f_plan *const model =
		pending == NULL ? NULL : calloc(1, sizeof(*model));
	if (status == STATUS_OK && model == NULL)
		status = STATUS_OUT_OF_MEMORY;
	if (model != NULL) {
		result->model = model;
		result->card = card_index;
		result->rank = rank;
		result->rank_count = rank_count;
		result->attention_replicated = replicated_attention;
		result->experts_partitioned = partitioned_experts;
		model->limits = *limits;
		model->chunk_attention_arrays = true; /* a TP plan's */
		model->card_count = card_index + 1u;
		model->layer_card[0] = card_index;
		model->cards[card_index].layers = 1;
		model->cards[card_index].holders =
			(uint32_t)(BIT(DS41F_HOLDER_TRUNK) | BIT(DS41F_HOLDER_LAYER));
	}
	for (uint32_t tensor = 0;
	     status == STATUS_OK && tensor < DS41F_TENSOR_LAYER_COUNT;
	     tensor++) {
		if ((layer_rules[tensor].layers & 1u) == 0)
			continue;
		char name[NAME_CAPACITY];
		const int length = snprintf(name, sizeof(name), "layers.0.%s",
					    layer_rules[tensor].name);
		if (length <= 0 || length >= NAME_CAPACITY) {
			status = STATUS_INVALID_ARGUMENT;
			break;
		}
		const bool split_outputs =
			(!replicated_attention &&
			 (tensor == DS41F_TENSOR_QUERY_UP ||
			  tensor == DS41F_TENSOR_OUTPUT_GROUPS)) ||
			tensor == DS41F_TENSOR_SHARED_GATE ||
			tensor == DS41F_TENSOR_SHARED_UP;
		const bool split_inputs =
			(!replicated_attention && tensor == DS41F_TENSOR_OUTPUT) ||
			tensor == DS41F_TENSOR_SHARED_DOWN;
		status = tp2_tensor_find(pack, name, &layer_rules[tensor],
					 card_index, rank, rank_count,
					 split_outputs, split_inputs, false,
					 &model->layer_tensors[0][tensor],
					 &pending[result->job_count]);
		result->job_count += status == STATUS_OK ? 1u : 0u;
	}
    const uint32_t dummy_expert = rank ^ 1u;
    for (uint32_t expert = 0; status == STATUS_OK && expert < DS41F_EXPERTS;
	 expert++) {
	    if (partitioned_experts && expert % DS41F_TP2_RANKS != rank &&
		expert != dummy_expert)
		    continue;
	    for (uint32_t matrix = 0;
		 status == STATUS_OK && matrix < DS41F_EXPERT_MATRICES;
		 matrix++) {
		    char name[NAME_CAPACITY];
		    const int length = snprintf(name, sizeof(name),
						"layers.0.ffn.experts.%u.%s",
						expert,
						expert_rules[matrix].name);
		    if (length <= 0 || length >= NAME_CAPACITY) {
			    status = STATUS_INVALID_ARGUMENT;
			    break;
		    }
		    struct tp2_pending_tensor *const item =
			    &pending[result->job_count];
		    status = tp2_tensor_find(
			    pack, name, &expert_rules[matrix], card_index, rank,
			    rank_count,
			    !partitioned_experts && matrix != DS41F_TENSOR_DOWN,
			    !partitioned_experts && matrix == DS41F_TENSOR_DOWN,
			    false, NULL, item);
		    item->expert_offset =
			    &model->expert_tables[0][expert][matrix];
		    result->job_count += status == STATUS_OK ? 1u : 0u;
	    }
    }
    uint64_t next = arena_prefix_bytes, first_expert = UINT64_MAX, last_expert = 0;
    if (status == STATUS_OK)
	    qsort(pending, result->job_count, sizeof(*pending),
		  tp2_pending_compare);
    for (uint32_t index = 0; status == STATUS_OK && index < result->job_count;
	 index++) {
	    struct tp2_pending_tensor *const item = &pending[index];
	    item->job.destination_offset = take(&next, item->byte_count);
	    result->jobs[index] = item->job;
	    if (item->stored != NULL)
		    *item->stored =
			    (struct linear_range){ item->job.destination_offset,
						   item->byte_count };
	    if (item->expert_offset != NULL) {
		    *item->expert_offset = item->job.destination_offset;
		    first_expert = item->job.destination_offset < first_expert ?
					   item->job.destination_offset :
					   first_expert;
		    const uint64_t end =
			    sum(item->job.destination_offset, item->byte_count);
		    last_expert = end > last_expert ? end : last_expert;
	    }
	    status = next == UINT64_MAX ? STATUS_INVALID_ARGUMENT : STATUS_OK;
    }
    if (status == STATUS_OK && partitioned_experts) {
	    result->unowned_gate_offset =
		    model->expert_tables[0][dummy_expert][DS41F_TENSOR_GATE];
	    for (uint32_t expert = 0; expert < DS41F_EXPERTS; expert++)
		    if (expert % DS41F_TP2_RANKS != rank)
			    memcpy(model->expert_tables[0][expert],
				   model->expert_tables[0][dummy_expert],
				   sizeof(model->expert_tables[0][expert]));
    }
    if (status == STATUS_OK) {
	    struct ds41f_card *const card = &model->cards[card_index];
	    card->state_offset = next;
	    model->experts[0] =
		    (struct linear_range){ first_expert,
					   last_expert - first_expert };
	    status = card_lay_out(model, card) ? STATUS_OK :
						 STATUS_INVALID_ARGUMENT;
	    if (status == STATUS_OK) {
		    card->image = calloc(1, card->image_byte_count);
		    status = card->image == NULL ? STATUS_OUT_OF_MEMORY :
						   STATUS_OK;
	    }
	    if (status == STATUS_OK) {
		    float words[DS41F_THETAS][DS41F_FREQUENCIES];
		    ds41f_frequency_words(words);
		    memcpy(card->image +
				   card->array_offset[DS41F_ARRAY_FREQUENCIES] -
				   card->state_offset,
			   words, sizeof(words));
		    memcpy(card->image + card->expert_table_offset[0] -
				   card->state_offset,
			   model->expert_tables[0],
			   sizeof(model->expert_tables[0]));
		    next = card->arena_byte_count;
		    const uint64_t compact_bytes =
			    (uint64_t)limits->row_capacity *
			    DS41F_HEADS_ROW_BYTES;
		    result->compact =
			    (struct linear_range){ take(&next, compact_bytes),
						   compact_bytes };
		    for (uint32_t site = 0; site < DS41F_TP2_SITES; site++) {
			    const uint64_t values =
				    (uint64_t)limits->row_capacity *
				    DS41F_HIDDEN;
			    const uint64_t bytes = values * sizeof(float);
			    const uint64_t signal_bytes =
				    (uint64_t)ORDERED_PUSH_PHASES * rank_count *
				    ORDERED_COLLECTIVE_LINE_BYTES;
			    uint64_t stride = signal_bytes +
					      (rank_count + DS41F_TP2_RANKS) * bytes;
			    const uint64_t table =
				    take(&next,
					 (uint64_t)rank_count *
						 ORDERED_COLLECTIVE_PEER_BYTES);
			    stride = (stride + DS41F_ARRAY_ALIGNMENT_BYTES -
				      1u) /
				     DS41F_ARRAY_ALIGNMENT_BYTES *
				     DS41F_ARRAY_ALIGNMENT_BYTES;
			    const uint64_t region = take(
				    &next, stride * ORDERED_COLLECTIVE_SLOTS);
			    result->collectives[site]
				    .arguments = (struct ordered_push_arguments){
				    .peer_table_offset = table,
				    .region_offset = region,
				    .input_offset = signal_bytes,
				    .output_offset = signal_bytes + bytes,
				    .inbox_offset =
					    signal_bytes + DS41F_TP2_RANKS * bytes,
				    .step_offset =
					    card->array_offset
						    [DS41F_ARRAY_STEP],
				    .parity_stride = stride,
				    .deadline_cycles = DS41F_TP2_DEADLINE_CYCLES,
				    .inbox_rank_stride = bytes,
				    .element_count = (uint32_t)values,
				    .rank = rank,
				    .rank_count = rank_count,
				    .payload_format = ORDERED_PUSH_FP32
			    };
		    }
		    for (uint32_t site = 0; site < DS41F_TP2_SITES; site++) {
			    const uint64_t bytes =
				    (uint64_t)limits->row_capacity *
				    DS41F_HIDDEN * sizeof(float);
			    result->stub_values[site] =
				    (struct linear_range){ take(&next, bytes),
							   bytes };
		    }
		    card->arena_byte_count = next;
		    for (uint32_t site = 0; site < DS41F_TP2_SITES; site++)
			    result->collectives[site].arena_byte_count = next;
		    if (next == UINT64_MAX)
			    status = STATUS_INVALID_ARGUMENT;
	    }
    }
    free(pending);
    if (status == STATUS_OK)
	    *plan_out = result;
    else
	    ds41f_tp2_layer_plan_destroy(result);
    return status;
}

result_status ds41f_tp2_layer_variant_create(const struct pack_record *pack,
					   const struct program_limits *limits,
					   uint32_t card_index, uint32_t rank,
					   uint32_t rank_count,
					   uint64_t arena_prefix_bytes,
					   bool replicated_attention,
					   struct ds41f_tp2_layer_plan **plan_out)
{
	return ds41f_tp2_layer_choices_create(pack, limits, card_index, rank,
					    rank_count, arena_prefix_bytes,
					    replicated_attention, false,
					    plan_out);
}

result_status ds41f_tp2_layer_plan_create(const struct pack_record *pack,
					const struct program_limits *limits,
					uint32_t card_index, uint32_t rank,
					uint32_t rank_count,
					uint64_t arena_prefix_bytes,
					struct ds41f_tp2_layer_plan **plan_out)
{
	return ds41f_tp2_layer_variant_create(pack, limits, card_index, rank,
					    rank_count, arena_prefix_bytes,
					    false, plan_out);
}

void ds41f_tp2_layer_plan_destroy(struct ds41f_tp2_layer_plan *plan)
{
    if (plan != NULL)
	    ds41f_plan_destroy(plan->model);
    free(plan);
}

result_status ds41f_state_byte_count(const struct pack_record *pack,
				   const struct placement_record *placement,
				   const struct program_limits *limits,
				   uint32_t card_index,
				   uint64_t *byte_count_out)
{
	struct load_layout *layout = NULL;
	struct ds41f_plan *plan = NULL;
	if (pack == NULL || placement == NULL || limits == NULL ||
	    byte_count_out == NULL ||
	    (card_index >= placement->card_count &&
	     card_index != PLACEMENT_IN_HOST_MEMORY)) {
		LOG_ERROR(
			"state_byte_count: a missing argument, or no card %" PRIu32
			" in the placement",
			card_index);
		return STATUS_INVALID_ARGUMENT;
	}
	result_status status = model_check(pack);
	status = status == STATUS_OK ? limits_check(limits) : status;
	status = status == STATUS_OK ?
			 load_layout_create(pack, placement->card_of_unit,
					    placement->unit_count,
					    placement->card_count, &layout) :
			 status;
	status = status == STATUS_OK ?
			 ds41f_plan_create(pack, layout, limits, &plan) :
			 status;
	if (status ==
	    STATUS_OK) { /* in host memory: the page-locked copies of its experts there */
		const struct ds41f_card *const card =
			&plan->cards[card_index < placement->card_count ?
					     card_index :
					     0];
		uint64_t host_bytes = 0;
		for (uint32_t layer = 0; layer < DS41F_LAYERS; layer++) {
			host_bytes = sum(host_bytes,
					 product(plan->host_expert_count[layer],
						 plan->expert_byte_count));
		}
		*byte_count_out =
			card_index < placement->card_count ?
				card->arena_byte_count - card->state_offset :
				host_bytes;
	}
	ds41f_plan_destroy(plan);
	load_layout_destroy(layout);
	return status;
}

result_status ds41f_arena_byte_count(const struct ds41f_plan *plan,
				   uint32_t card_index,
				   uint64_t *byte_count_out)
{
    if (plan == NULL || byte_count_out == NULL || card_index >= plan->card_count) {
	    LOG_ERROR(
		    "arena_byte_count: a missing argument, or no card %" PRIu32
		    " in the plan",
		    card_index);
	    return STATUS_INVALID_ARGUMENT;
    }
    *byte_count_out = plan->cards[card_index].arena_byte_count;
    return STATUS_OK;
}

/* The state of one card behind its slots. The backend's digest of the plan's last word proves the arena holds the
 * state before the card writes the zeros behind the image (fill_words_entry); the image goes up through staging;
 * the card's digest of that state is held to the host's, worked out while the card works. staging_is_read: an upload
 * began that was not seen to finish, so staging may still be read. */
MUST_CHECK static result_status
card_initialise(const struct ds41f_card *plan_card, uint32_t card_index,
		const struct backend_operations *backend,
		struct device_card *card, unsigned char *staging,
		uint64_t wait_limit_nanoseconds, bool *staging_is_read)
{
    const uint64_t image_bytes = plan_card->image_byte_count;
    const uint64_t image_offset = image_offset_of(plan_card);
    const uint64_t state_bytes = plan_card->arena_byte_count - image_offset;
    const uint64_t word_bytes = PACK_DIGEST_WORD_BYTES;
    const struct fill_record zeros = { image_offset + image_bytes,
				       (state_bytes - image_bytes) /
					       FILL_WORD_BYTES,
				       0, 0 };
    const uint64_t blocks =
	    (zeros.word_count + FILL_THREADS_PER_BLOCK_VALUE - 1u) /
	    FILL_THREADS_PER_BLOCK_VALUE;
    const uint32_t grid = blocks == 0		    ? 1u :
			  blocks < FILL_MOST_BLOCKS ? (uint32_t)blocks :
						      FILL_MOST_BLOCKS;
    struct kernel_launch fill;
    uint64_t ticket = 0, expected = 0, digest = 0;
    uint32_t kernel = BACKEND_NO_KERNEL;
    result_status status = backend->kernel_prepare(card, KERNEL_FILE_FILL,
						   KERNEL_FILL_WORDS,
						   FILL_RECORD_BYTES, &kernel);
    status = status == STATUS_OK ?
		     backend->digest(card,
				     plan_card->arena_byte_count - word_bytes,
				     word_bytes, wait_limit_nanoseconds,
				     &digest) :
		     status;
    prepare_launch_fill(&fill, kernel, grid, 1, FILL_THREADS_PER_BLOCK_VALUE, 0,
			&zeros, sizeof(zeros));
    status = status == STATUS_OK ? backend->launch_run(card, &fill, 1) : status;
    if (status == STATUS_OK && image_bytes != 0) {
	    memcpy(staging, plan_card->image, (size_t)image_bytes);
	    status = backend->upload_begin(card, image_offset, staging,
					   image_bytes, &ticket);
	    *staging_is_read = status == STATUS_OK;
	    status = status == STATUS_OK ?
			     pack_digest_of_host_bytes(plan_card->image,
						       image_bytes, &expected) :
			     status;
    }
    for (uint64_t word = image_bytes / word_bytes; word < state_bytes / word_bytes; word++) {
	    expected += pack_digest_term(0, word);
    }
    if (status == STATUS_OK && image_bytes != 0) {
	    status = backend->upload_wait(card, ticket, wait_limit_nanoseconds);
	    *staging_is_read = status != STATUS_OK;
    }
    status = status == STATUS_OK ?
		     backend->launch_wait(card, wait_limit_nanoseconds) :
		     status;
    status = status == STATUS_OK ?
		     backend->digest(card, image_offset, state_bytes,
				     wait_limit_nanoseconds, &digest) :
		     status;
    const bool differs = status == STATUS_OK && digest != expected;
    status = differs ? STATUS_DEVICE_ERROR : status;
    if (status != STATUS_OK) {
	    LOG_ERROR(
		    "card %" PRIu32 ": its state, arena bytes %" PRIu64
		    " to %" PRIu64 ", was not set%s: %s",
		    card_index, image_offset, plan_card->arena_byte_count,
		    differs ?
			    ", the card's digest of it differs from the host's" :
			    "",
		    status_text(status));
    }
    return status;
}

/* The scales follow the weight bytes (weight_format_seam.h). A failed fold is an arithmetic fallback, while a
 * failed download still fails load. Only the CMP selection reads these folds; the PRO keeps its own arithmetic. */
MUST_CHECK static result_status
expert_folds_prepare(struct ds41f_plan *plan,
		     const struct backend_operations *backend,
		     struct device_card *const *cards, void *staging,
		     uint64_t wait_limit_nanoseconds, bool *staging_is_read)
{
	const uint64_t values = (uint64_t)DS41F_HIDDEN * DS41F_EXPERT_INNER;
	const uint64_t weight_bytes = values / E2M1_VALUES_PER_BYTE;
	const uint64_t scale_bytes = values / LINEAR_VALUES_PER_BLOCK;
	for (uint32_t layer = 0; layer < DS41F_WALK_LAYERS; layer++) {
		plan->expert_folds_admitted[layer] = false;
		if (plan->exl3_bits[layer] != 0 ||
		    plan->experts[layer].byte_count == 0 ||
		    plan->host_expert_count[layer] != 0) {
			continue;
		}
		const uint32_t card = plan->layer_card[layer];
		result_status status = STATUS_OK;
		bool admitted = true;
		const uint32_t expert_count = layer < DS41F_LAYERS ?
						      DS41F_EXPERTS :
						      DS41F_DRAFT_EXPERTS;
		for (uint32_t expert = 0; expert < expert_count && admitted;
		     expert++) {
			for (uint32_t matrix = 0;
			     matrix < DS41F_EXPERT_MATRICES && admitted;
			     matrix++) {
				const uint64_t offset =
					plan->expert_tables[layer][expert]
							   [matrix];
				uint64_t ticket = 0;
				status = backend->download_begin(
					cards[card], offset + weight_bytes,
					staging, scale_bytes, &ticket);
				*staging_is_read = status == STATUS_OK;
				status =
					status == STATUS_OK ?
						backend->upload_wait(
							cards[card], ticket,
							wait_limit_nanoseconds) :
						status;
				if (status != STATUS_OK) {
					return status;
				}
				*staging_is_read = false;
				status = expert_relaxed_fold_check(staging,
								   scale_bytes);
				if (status == STATUS_INVALID_ARGUMENT) {
					LOG_WARNING_ENTRY(
						"layers.%" PRIu32
						".ffn.experts.%" PRIu32
						".%s: scales refuse the fold; "
						"reference CMP order",
						layer, expert,
						expert_rules[matrix].name);
					admitted = false;
				} else if (status != STATUS_OK) {
					return status;
				}
			}
		}
		plan->expert_folds_admitted[layer] = admitted;
	}
	return STATUS_OK;
}

MUST_CHECK static result_status
folds_prepare(struct ds41f_plan *plan,
	      const struct backend_operations *backend,
	      struct device_card *const *cards, void *staging,
	      uint64_t wait_limit_nanoseconds, bool *staging_is_read)
{
	result_status status = STATUS_OK;
	for (uint32_t layer = 0; layer < DS41F_LAYERS && status == STATUS_OK;
	     layer++) {
		for (uint32_t index = 0;
		     index < DS41F_TENSOR_LAYER_COUNT && status == STATUS_OK;
		     index++) {
			struct ds41f_tensor *const tensor =
				&plan->layer_tensors[layer][index];
			tensor->fold_reference = 0;
			if (tensor->weight_format !=
			    WEIGHT_FORMAT_E4M3_SCALED) {
				continue;
			}
			const uint64_t weight_bytes =
				(uint64_t)tensor->shape.output_size *
				tensor->shape.input_size;
			const uint64_t scale_bytes =
				tensor->stored.byte_count - weight_bytes;
			uint64_t ticket = 0;
			status = backend->download_begin(
				cards[tensor->card],
				tensor->stored.byte_offset + weight_bytes,
				staging, scale_bytes, &ticket);
			*staging_is_read = status == STATUS_OK;
			status = status == STATUS_OK ?
					 backend->upload_wait(
						 cards[tensor->card], ticket,
						 wait_limit_nanoseconds) :
					 status;
			if (status != STATUS_OK) {
				break;
			}
			*staging_is_read = false;
			status = relaxed_fold_check(
				index == DS41F_TENSOR_OUTPUT_GROUPS ?
					RELAXED_BFLOAT16_ROWS :
					RELAXED_QUANTIZED_ROWS,
				staging, scale_bytes, &tensor->fold_reference);
			if (status == STATUS_INVALID_ARGUMENT) {
				LOG_WARNING_ENTRY(
					"layers.%" PRIu32
					".%s: scales refuse the exact fold; reference CMP order",
					layer, layer_rules[index].name);
				status = STATUS_OK;
			}
		}
	}
    return status;
}

/* The draft's copy of embed.weight (model_v41.h): from the embedding card's weights into DRAFT_EMBEDDING of the result
 * card through staging, a piece at a time, each piece seen to arrive before the next; then the cards' digests of both
 * ranges, held equal. staging_is_read as card_initialise sets it. */
MUST_CHECK static result_status
embedding_copy(const struct ds41f_plan *plan,
	       const struct backend_operations *backend,
	       struct device_card *const *cards, unsigned char *staging,
	       uint64_t wait_limit_nanoseconds, bool *staging_is_read)
{
	const struct linear_range source =
		plan->globals[DS41F_TENSOR_EMBEDDING].stored;
	const uint64_t copy =
		plan->cards[plan->result_card]
			.array_offset[DS41F_ARRAY_DRAFT_EMBEDDING];
	const uint64_t piece = DS41F_EMBEDDING_COPY_PIECE_BYTES;
	struct device_card *const from = cards[plan->embedding_card],
				  *const to = cards[plan->result_card];
	uint64_t source_digest = 0, copy_digest = 0;
	result_status status = STATUS_OK;
	for (uint64_t done = 0; done < source.byte_count && status == STATUS_OK;
	     done += piece) {
		const uint64_t bytes = source.byte_count - done < piece ?
					       source.byte_count - done :
					       piece;
		uint64_t ticket = 0;
		status = backend->download_begin(from,
						 source.byte_offset + done,
						 staging, bytes, &ticket);
		*staging_is_read = status == STATUS_OK;
		status = status == STATUS_OK ?
				 backend->upload_wait(from, ticket,
						      wait_limit_nanoseconds) :
				 status;
		*staging_is_read = *staging_is_read && status != STATUS_OK;
		if (status == STATUS_OK) {
			status = backend->upload_begin(to, copy + done, staging,
						       bytes, &ticket);
			*staging_is_read = status == STATUS_OK;
			status = status == STATUS_OK ?
					 backend->upload_wait(
						 to, ticket,
						 wait_limit_nanoseconds) :
					 status;
			*staging_is_read = *staging_is_read &&
					   status != STATUS_OK;
		}
	}
	status = status == STATUS_OK ? backend->digest(from, source.byte_offset,
						       source.byte_count,
						       wait_limit_nanoseconds,
						       &source_digest) :
				       status;
	status = status == STATUS_OK ?
			 backend->digest(to, copy, source.byte_count,
					 wait_limit_nanoseconds, &copy_digest) :
			 status;
	const bool differs = status == STATUS_OK &&
			     source_digest != copy_digest;
	status = differs ? STATUS_DEVICE_ERROR : status;
	if (status != STATUS_OK) {
		LOG_ERROR(
			"the draft's copy of embed.weight, %" PRIu64
			" bytes from card %" PRIu32 " to byte %" PRIu64
			" of card %" PRIu32 ", was not made%s: %s",
			source.byte_count, plan->embedding_card, copy,
			plan->result_card,
			differs ?
				", the cards' digests of the two ranges differ" :
				"",
			status_text(status));
	}
    return status;
}

/* A plan's copy of embed.weight lies whole in the embedding card's weights and in the result card's state behind its
 * image, in whole digest words, on cards given, with a backend that downloads. */
static bool copy_is_planned(const struct ds41f_plan *plan,
			    const struct backend_operations *backend,
			    struct device_card *const *cards)
{
    if (plan->embedding_card >= plan->card_count || plan->result_card >= plan->card_count) {
        return false;
    }
    const struct linear_range source = plan->globals[DS41F_TENSOR_EMBEDDING].stored;
    const struct ds41f_card *const from = &plan->cards[plan->embedding_card];
    const struct ds41f_card *const to = &plan->cards[plan->result_card];
    const uint64_t copy = to->array_offset[DS41F_ARRAY_DRAFT_EMBEDDING];
    return cards[plan->embedding_card] != NULL &&
	   cards[plan->result_card] != NULL && backend != NULL &&
	   backend->download_begin != NULL && source.byte_count != 0 &&
	   source.byte_count <= DS41F_DRAFT_EMBEDDING_BYTES &&
	   source.byte_offset <= from->state_offset &&
	   source.byte_count <= from->state_offset - source.byte_offset &&
	   copy >= image_offset_of(to) + to->image_byte_count &&
	   copy <= to->arena_byte_count &&
	   source.byte_count <= to->arena_byte_count - copy &&
	   (source.byte_offset | source.byte_count | copy) %
			   PACK_DIGEST_WORD_BYTES ==
		   0;
}

result_status ds41f_arena_initialise(struct ds41f_plan *plan,
				   const struct backend_operations *backend,
				   struct device_card *const *cards,
				   uint64_t wait_limit_nanoseconds)
{
    void *staging = NULL;
    bool staging_is_read = false;
    uint64_t largest_image = 0;
    bool cards_are_given = plan != NULL && cards != NULL &&
			   plan->card_count - 1u < BACKEND_MAXIMUM_CARDS;
    const bool has_draft = cards_are_given && plan->limits.draft_positions != 0;
    for (uint32_t card = 0; cards_are_given && card < plan->card_count; card++) {
	    const struct ds41f_card *const planned = &plan->cards[card];
	    cards_are_given =
		    (cards[card] != NULL ||
		     planned->arena_byte_count == planned->state_offset) &&
		    planned->state_offset <= planned->arena_byte_count &&
		    planned->slots_byte_count <=
			    planned->arena_byte_count - planned->state_offset &&
		    planned->image_byte_count <=
			    planned->arena_byte_count -
				    image_offset_of(planned) &&
		    (planned->image != NULL ||
		     planned->image_byte_count == 0) &&
		    (planned->state_offset | planned->slots_byte_count |
		     planned->arena_byte_count | planned->image_byte_count) %
				    PACK_DIGEST_WORD_BYTES ==
			    0;
	    largest_image = planned->image_byte_count > largest_image ?
				    planned->image_byte_count :
				    largest_image;
    }
    const bool check_folds = plan != NULL &&
			     plan->limits.cmp_order == CMP_ORDER_RELAXED;
    if (check_folds) {
        memset(plan->expert_folds_admitted, 0, sizeof(plan->expert_folds_admitted));
    }
    for (uint32_t layer = 0;
	 check_folds && cards_are_given && layer < DS41F_LAYERS; layer++) {
	    for (uint32_t index = 0; index < DS41F_TENSOR_LAYER_COUNT;
		 index++) {
		    const struct ds41f_tensor *const tensor =
			    &plan->layer_tensors[layer][index];
		    if (tensor->weight_format == WEIGHT_FORMAT_E4M3_SCALED) {
			    const uint64_t weight_bytes =
				    (uint64_t)tensor->shape.output_size *
				    tensor->shape.input_size;
			    cards_are_given = cards_are_given &&
					      tensor->card < plan->card_count &&
					      tensor->stored.byte_count >
						      weight_bytes;
			    const uint64_t scales =
				    tensor->stored.byte_count > weight_bytes ?
					    tensor->stored.byte_count -
						    weight_bytes :
					    0;
			    largest_image = scales > largest_image ?
						    scales :
						    largest_image;
		    }
	    }
    }
    for (uint32_t layer = 0;
	 check_folds && cards_are_given && layer < DS41F_WALK_LAYERS;
	 layer++) {
	    if (plan->experts[layer].byte_count != 0 &&
		plan->host_expert_count[layer] == 0) {
		    const uint64_t values =
			    (uint64_t)DS41F_HIDDEN * DS41F_EXPERT_INNER;
		    const uint64_t scale_bytes =
			    values / LINEAR_VALUES_PER_BLOCK;
		    const uint64_t matrix_bytes =
			    values / E2M1_VALUES_PER_BYTE + scale_bytes;
		    const struct linear_range *const range =
			    &plan->experts[layer];
		    cards_are_given = cards_are_given &&
				      plan->layer_card[layer] <
					      plan->card_count &&
				      range->byte_offset <=
					      UINT64_MAX - range->byte_count;
		    if (cards_are_given) {
			    cards_are_given =
				    range->byte_offset + range->byte_count <=
				    plan->cards[plan->layer_card[layer]]
					    .state_offset;
		    }
		    const uint32_t expert_count = layer < DS41F_LAYERS ?
							  DS41F_EXPERTS :
							  DS41F_DRAFT_EXPERTS;
		    for (uint32_t expert = 0; expert < expert_count; expert++) {
			    for (uint32_t matrix = 0;
				 matrix < DS41F_EXPERT_MATRICES; matrix++) {
				    const uint64_t offset =
					    plan->expert_tables[layer][expert]
							       [matrix];
				    cards_are_given =
					    cards_are_given &&
					    offset >= range->byte_offset &&
					    offset - range->byte_offset <=
						    range->byte_count &&
					    matrix_bytes <=
						    range->byte_count -
							    (offset -
							     range->byte_offset);
			    }
		    }
		    largest_image = scale_bytes > largest_image ? scale_bytes :
								  largest_image;
	    }
    }
    if (!cards_are_given || backend == NULL || backend->staging_allocate == NULL || backend->staging_free == NULL ||
        backend->upload_begin == NULL || backend->upload_wait == NULL || backend->digest == NULL ||
        backend->kernel_prepare == NULL || backend->launch_run == NULL || backend->launch_wait == NULL ||
        (check_folds && backend->download_begin == NULL) ||
        (has_draft && !copy_is_planned(plan, backend, cards))) {
	    LOG_ERROR(
		    "arena_initialise: a missing argument, card or function of the backend, or not a plan");
	    return STATUS_INVALID_ARGUMENT;
    }
    const uint64_t staging_bytes =
	    has_draft && largest_image < DS41F_EMBEDDING_COPY_PIECE_BYTES ?
		    DS41F_EMBEDDING_COPY_PIECE_BYTES :
		    largest_image;
    result_status status =
	    staging_bytes == 0 ?
		    STATUS_OK :
		    backend->staging_allocate(staging_bytes, &staging);
    for (uint32_t card = 0; card < plan->card_count && status == STATUS_OK;
	 card++) {
	    status = plan->cards[card].arena_byte_count ==
				     plan->cards[card].state_offset ?
			     STATUS_OK :
			     card_initialise(&plan->cards[card], card, backend,
					     cards[card], staging,
					     wait_limit_nanoseconds,
					     &staging_is_read);
    }
    status = check_folds && status == STATUS_OK ?
		     folds_prepare(plan, backend, cards, staging,
				   wait_limit_nanoseconds, &staging_is_read) :
		     status;
    status = check_folds && status == STATUS_OK ?
		     expert_folds_prepare(plan, backend, cards, staging,
					  wait_limit_nanoseconds,
					  &staging_is_read) :
		     status;
    status = status == STATUS_OK && has_draft ?
		     embedding_copy(plan, backend, cards, staging,
				    wait_limit_nanoseconds, &staging_is_read) :
		     status;
    if (staging_is_read) {
	    LOG_ERROR(
		    "arena_initialise: staging memory stays allocated: an upload may still read it");
    } else if (staging != NULL) {
        backend->staging_free(staging);
    }
    return status;
}

/* The bits of both thetas' words as the reference forms them on the card (REF:376-387), recorded by
 * stepwise_capture.py --frequencies: theta 0 CFG:30 without YaRN, theta 1 CFG:51 with YaRN (factor 16,
 * original length 65,536, beta 32 and 1). The card's power lands a place below the float32 nearest
 * 10000^(2 i / 64) at words 11 and 25 of theta 0, where the host's expression gave the nearest, and the table gate
 * takes no place (first_token_plan.md 2.10). */
static const uint32_t frequency_word_bits[DS41F_THETAS][DS41F_FREQUENCIES] = {
	{ 0x3f800000u, 0x3f3ff911u, 0x3f0ff59au, 0x3ed7e89bu, 0x3ea1e89bu,
	  0x3e72d423u, 0x3e361887u, 0x3e088d77u, 0x3dcccccdu, 0x3d99940du,
	  0x3d6655c2u, 0x3d2cba16u, 0x3d0186e3u, 0x3cc2434fu, 0x3c91ad39u,
	  0x3c5a7bf2u, 0x3c23d70au, 0x3bf5b9b0u, 0x3bb8449cu, 0x3b8a2e77u,
	  0x3b4f3e38u, 0x3b1b690du, 0x3ae91528u, 0x3aaec98eu, 0x3a83126fu,
	  0x3a44948du, 0x3a136a16u, 0x39dd1725u, 0x39a5cb60u, 0x3978a815u,
	  0x393a7753u, 0x390bd472u },
	{ 0x3f800000u, 0x3f300a3au, 0x3ef21c1fu, 0x3ea67d01u, 0x3e64f92eu,
	  0x3e1d7475u, 0x3dd88cb4u, 0x3d94e963u, 0x3d4ccccdu, 0x3d0cd4fbu,
	  0x3cc1b019u, 0x3c8530ceu, 0x3c372dbfu, 0x3bfbed88u, 0x3bad3d5eu,
	  0x3b6e4237u, 0x3b147ae1u, 0x3ab714e0u, 0x3a5ebdb6u, 0x3a0530ceu,
	  0x399bb3afu, 0x39305978u, 0x38be904du, 0x383e9b5fu, 0x37a3d708u,
	  0x36b443d0u, 0x3677eba6u, 0x362a7be8u, 0x35ea77ffu, 0x35a13bdcu,
	  0x355dbf30u, 0x35187c4du },
};
_Static_assert(sizeof(frequency_word_bits) ==
		       sizeof(float[DS41F_THETAS][DS41F_FREQUENCIES]),
	       "a word's bits are a float32");

void ds41f_frequency_words(float words_out[DS41F_THETAS][DS41F_FREQUENCIES])
{
    memcpy(words_out, frequency_word_bits, sizeof(frequency_word_bits));
}

/* Cold stage inventory: intentionally separate from the PP and executable layer-zero entry points above. */
struct tp2_stage_pending {
    struct tp2_pending_tensor tensor;
    uint32_t layer;
    bool is_expert;
};

static const struct tensor_rule *tp2_stage_rule(uint32_t layer, uint32_t tensor)
{
	return layer >= DS41F_DRAFT_FIRST_LAYER && tensor >= DS41F_TENSOR_ROUTER &&
			       tensor <= DS41F_TENSOR_ROUTER_BIAS ?
		       &draft_router_rules[tensor - DS41F_TENSOR_ROUTER] :
		       &layer_rules[tensor];
}

static int tp2_stage_pending_compare(const void *first, const void *second)
{
    const struct tp2_stage_pending *const left = first, *const right = second;
    return tp2_pending_compare(&left->tensor, &right->tensor);
}

#define TP2_STAGE_MARKOV_TABLES 2u /* draft_shards' ranks 1-3: the Markov head's embedding and head */
#define TP2_STAGE_RANK_DRAFT_GLOBALS 3u /* draft_layers' ranks 1-3: main_proj, main_norm and the draft's norm */
/* the stage's TC MoE workspaces: of the widest records (exl3_tc_records.h: rotated records keep the 640 layout) */
#define TC_WIDEST_INTERMEDIATE 640u

/* head_shards' ranks 1-3: the final norm and their rows of the head (ds41f_tp_stage.h). */
static bool
tp2_stage_head_rows(const struct ds41f_tp2_stage_description *description)
{
    return description->head_shards && !description->include_head_and_draft;
}

static uint32_t
tp2_stage_jobs(uint64_t walked,
	       const struct ds41f_tp2_stage_description *description)
{
    uint32_t count = description->include_embedding ? 1u : 0u;
    count += tp2_stage_head_rows(description) ? DS41F_TENSOR_GLOBALS - 1u : 0u;
    count += tp2_stage_head_rows(description) && description->draft_shards ? TP2_STAGE_MARKOV_TABLES : 0u;
    count += description->include_head_and_draft ?
		     DS41F_TENSOR_GLOBALS - 1u + DS41F_TENSOR_DRAFT_GLOBALS :
		     0;
    count += description->draft_layers && !description->include_head_and_draft ? TP2_STAGE_RANK_DRAFT_GLOBALS : 0;
    for (uint32_t layer = 0; layer < DS41F_WALK_LAYERS; layer++) {
	    if ((walked & BIT(layer)) == 0)
		    continue;
	    for (uint32_t tensor = 0; tensor < DS41F_TENSOR_LAYER_COUNT;
		 tensor++)
		    count += (tp2_stage_rule(layer, tensor)->layers &
			      BIT(layer)) != 0 ?
				     1u :
				     0u;
	    count += (layer >= DS41F_DRAFT_FIRST_LAYER ? DS41F_DRAFT_EXPERTS :
							 DS41F_EXPERTS) *
		     DS41F_EXPERT_MATRICES;
    }
    return count;
}

/* Populate global source geometry before the single card layout, including sources owned by earlier stages. */
static void tp2_stage_holders(struct ds41f_plan *model,
			      uint32_t card_index, uint64_t targets, bool draft,
			      bool draft_copy)
{
	struct ds41f_card *const card = &model->cards[card_index];
	card->layers = targets;
	card->holders =
		(uint32_t)(BIT(DS41F_HOLDER_TRUNK) | BIT(DS41F_HOLDER_LAYER) |
			   (draft ? BIT(DS41F_HOLDER_RESULT) | BIT(DS41F_HOLDER_DRAFT) : 0) |
			   (draft_copy ? BIT(DS41F_HOLDER_DRAFT_COPY) : 0));
	uint32_t table = 0;
	for (uint32_t layer = 0; layer < DS41F_WALK_LAYERS; layer++) {
		model->layer_card[layer] =
			(ds41f_walk_layers(card) & BIT(layer)) != 0 ?
				card_index :
				BACKEND_MAXIMUM_CARDS;
		for (uint32_t tensor = 0; tensor < DS41F_TENSOR_LAYER_COUNT;
		     tensor++)
			model->layer_tensors[layer][tensor].card =
				BACKEND_MAXIMUM_CARDS;
		if ((DS41F_SOURCE_LAYERS & BIT(layer)) != 0) {
			const uint64_t entries =
				model->limits.context_token_limit /
				((DS41F_POOLED_LAYERS & BIT(layer)) != 0 ?
					 DS41F_POOLED_RATIO :
					 DS41F_DIRECT_RATIO);
			model->pool_session_bytes[layer] =
				entries * DS41F_STATE_ROW_BYTES;
			model->key_session_bytes[layer] =
				entries * DS41F_KEY_BYTES;
		}
		if ((targets & BIT(layer)) != 0) {
			card->holders |= (uint32_t)((DS41F_INDEX_LAYERS &
						     BIT(layer)) != 0 ?
							    BIT(DS41F_HOLDER_INDEX) :
							    0);
			card->holders |= (uint32_t)((DS41F_SOURCE_LAYERS &
						     BIT(layer)) != 0 ?
							    BIT(DS41F_HOLDER_SOURCE) :
							    0);
			card->holders |= (uint32_t)((DS41F_POOLED_LAYERS &
						     BIT(layer)) != 0 ?
							    BIT(DS41F_HOLDER_POOLED) :
							    0);
			card->holders |=
				(uint32_t)(layer == DS41F_CANDIDATE_SOURCE ||
							   (DS41F_CANDIDATE_READERS &
							    BIT(layer)) != 0 ?
						   BIT(DS41F_HOLDER_CANDIDATES) :
						   0);
		}
		if ((DS41F_ENGRAM_LAYERS & BIT(layer)) != 0) {
			if ((targets & BIT(layer)) != 0) {
				card->holders |= (uint32_t)BIT(DS41F_HOLDER_ENGRAM);
				card->engram_tables |= (uint32_t)BIT(table);
			}
			table++;
		}
	}
}

/* chunk_rank: the rank whose inner chunk a routed expert's matrices take (exl3_expert_rotation's
 * ds41f_exl3_tc_chunk_of; the stage's rank otherwise). */
MUST_CHECK static result_status tp2_stage_tensor_add(
	const struct pack_record *pack, struct ds41f_tp2_stage_plan *stage,
	struct tp2_stage_pending *pending, uint32_t capacity, const char *name,
	const struct tensor_rule *rule, bool split_outputs, bool split_inputs,
	struct ds41f_tensor *tensor_out, uint32_t layer, uint64_t *expert_offset,
	uint32_t chunk_rank)
{
    if (stage->job_count >= capacity)
	    return STATUS_INVALID_ARGUMENT;
    struct tp2_stage_pending *const item = &pending[stage->job_count];
    const struct ds41f_tp2_stage_description *const description =
	    &stage->description;
    const bool whole = layer >= DS41F_DRAFT_FIRST_LAYER &&
		       !description->draft_layers;
    const result_status status = tp2_tensor_find(
	    pack, name, rule, description->card, whole ? 0u : chunk_rank,
	    whole ? 1u : description->rank_count, split_outputs, split_inputs,
	    description->exact_output_shards, tensor_out, &item->tensor);
    if (status == STATUS_OK) {
	    item->layer = layer;
	    item->is_expert = expert_offset != NULL;
	    item->tensor.expert_offset = expert_offset;
	    stage->job_count++;
    }
    return status;
}

/* The most entries an index layer of the card scores (card_lay_out's INDEX_SCORES row). */
static uint64_t card_index_entries(const struct ds41f_plan *plan,
				   const struct ds41f_card *card)
{
    uint64_t entries = 0;
    for (uint32_t layer = 0; layer < DS41F_LAYERS; layer++) {
	    const uint32_t source = latest_of(DS41F_SOURCE_LAYERS, layer);
	    const uint64_t scored =
		    source < DS41F_LAYERS ?
			    plan->key_session_bytes[source] / DS41F_KEY_BYTES :
			    0u;
	    entries = (card->layers & DS41F_INDEX_LAYERS & BIT(layer)) !=
					      0 &&
				      scored > entries ?
			      scored :
			      entries;
    }
    return entries;
}

MUST_CHECK static result_status
tp2_stage_channels(struct ds41f_tp2_stage_plan *stage)
{
	const struct ds41f_tp2_stage_description *const description =
		&stage->description;
	struct ds41f_card *const card = &stage->model->cards[description->card];
	if (description->rank_count == 1u)
		return STATUS_OK;
	stage->channel_rows = description->limits.row_capacity;
	/* draft_layers: the draft's stages' channels behind the 40 layers' */
	const uint32_t channel_layers =
		description->target_layer_count +
		(description->draft_layers ? DS41F_DRAFT_STAGES : 0u);
	stage->channel_count = channel_layers * DS41F_TP2_SITES;
	stage->channels =
		FAULT_POINT(model_v41_tp2_stage_channels_memory) ?
			NULL :
			calloc(stage->channel_count, sizeof(*stage->channels));
	if (stage->channels == NULL)
		return STATUS_OUT_OF_MEMORY;
	uint64_t next = card->arena_byte_count;
	if (description->original_wide_experts) {
		/* Fused fronts require zero counters across calls. The wide router writes the ordinary wide plan's
         * aliased scratch in ROUTER_PICKS, so this recipe keeps its scratch beyond the base state instead. */
		card->front_scratch_offset =
			take(&next, DS41F_FUSED_SCRATCH_BYTES);
		card->producer_arrivals_offset = sum(card->front_scratch_offset,
						     DS41F_SITE_FRONT_SCRATCH_BYTES);
	}
    const uint32_t ranks = description->rank_count;
    const uint64_t compact_bytes = (uint64_t)description->limits.row_capacity *
				   DS41F_HEADS_ROW_BYTES / ranks;
    stage->compact =
	    (struct linear_range){ take(&next, compact_bytes), compact_bytes };
    if (description->index_span) {
	    /* a span's scores and split annex (ds41f_index_scan.h), in whole slabs within DS41F_INDEX_SPAN_BYTES; fewer
         * than two slabs: none (the scans a slab at a time) */
	    const uint64_t row_bytes =
		    card_index_entries(stage->model, card) * DS41F_INDEX_SCORE_BYTES +
		    DS41F_INDEX_SPLIT_ROW_BYTES;
	    const uint64_t fit = DS41F_INDEX_SPAN_BYTES / row_bytes /
				 LINEAR_MAXIMUM_ROWS * LINEAR_MAXIMUM_ROWS;
	    const uint64_t rows = fit < DS41F_SPAN_ROWS ? fit :
							     DS41F_SPAN_ROWS;
	    stage->index_rows = rows > LINEAR_MAXIMUM_ROWS &&
						row_bytes >
							DS41F_INDEX_SPLIT_ROW_BYTES ?
					(uint32_t)rows :
					0u;
	    stage->index_scores = (struct linear_range){
		    take(&next, stage->index_rows * row_bytes),
		    stage->index_rows * row_bytes
	    };
    }
    if (description->exact_output_shards ||
	(description->rank_count == DS41F_TP4_RANKS &&
	 description->wire_format == ORDERED_PUSH_BF16)) {
	    /* exl3_wide_entry: a chunk's gathered pairs and sums (its gathers a slab at a time), else a slab's */
	    const uint64_t gathered_rows =
		    description->exl3_wide_entry ||
				    description->original_wide_experts ?
			    description->limits.row_capacity :
			    LINEAR_MAXIMUM_ROWS;
	    const uint64_t hidden_bytes =
		    gathered_rows * DS41F_EXPERTS_PER_ROW *
		    DS41F_EXPERT_INNER * sizeof(uint16_t);
	    const uint64_t sum_bytes =
		    gathered_rows * DS41F_HIDDEN * sizeof(float);
	    stage->exact_hidden =
		    (struct linear_range){ take(&next, hidden_bytes),
					   hidden_bytes };
	    stage->exact_sums =
		    (struct linear_range){ take(&next, sum_bytes), sum_bytes };
    }
    if (description->exact_dense) {
        /* the thin gathers' owner-major slots, then a piece's own outputs [rows][widest own] (model_v41_program.c) */
	const uint64_t dense_input_bytes =
		(uint64_t)LINEAR_MAXIMUM_ROWS * DS41F_WIDEST_ROW *
			sizeof(uint16_t) +
		(uint64_t)description->limits.row_capacity *
			(DS41F_WIDEST_ROW / ranks) * sizeof(uint16_t);
	stage->dense_input =
		(struct linear_range){ take(&next, dense_input_bytes),
				       dense_input_bytes };
    }
    const uint64_t elements = (uint64_t)stage->channel_rows * DS41F_HIDDEN;
    const uint64_t bytes = elements * sizeof(float);
    const uint64_t signals = (uint64_t)ORDERED_PUSH_PHASES * ranks *
			     ORDERED_COLLECTIVE_LINE_BYTES;
    const uint64_t stride_bytes = signals + (ranks + DS41F_TP_LOCAL_VECTORS) * bytes;
    const uint64_t stride =
	    (stride_bytes + DS41F_ARRAY_ALIGNMENT_BYTES - 1u) /
	    DS41F_ARRAY_ALIGNMENT_BYTES * DS41F_ARRAY_ALIGNMENT_BYTES;
    /* BATCH steps, and with piece_rings any step of more than 8 rows: BF16 rings of up to 64 rows (a bank a count) */
    const bool batch_rings =
	    (description->batch ||
	     (description->piece_rings &&
	      description->limits.row_capacity > LINEAR_MAXIMUM_ROWS)) &&
	    description->rank_count == DS41F_TP4_RANKS &&
	    description->wire_format == ORDERED_PUSH_BF16 &&
	    !description->exact_output_shards;
    stage->ring_rows = batch_rings ? PROGRAM_BATCH_MOST_ROWS :
				     LINEAR_MAXIMUM_ROWS;
    for (uint32_t index = 0; index < stage->channel_count; index++) {
	    struct ds41f_tp2_stage_channel *const channel =
		    &stage->channels[index];
	    channel->layer =
		    description->first_target_layer + index / DS41F_TP2_SITES;
	    channel->site = index % DS41F_TP2_SITES;
	    if (description->limits.row_capacity > LINEAR_MAXIMUM_ROWS &&
		index >= DS41F_TP2_SITES) {
		    channel->collective =
			    stage->channels[channel->site].collective;
		    channel->ring_offset =
			    stage->channels[channel->site].ring_offset;
		    channel->ring_bank_bytes =
			    stage->channels[channel->site].ring_bank_bytes;
		    channel->ring_row_bytes =
			    stage->channels[channel->site].ring_row_bytes;
		    channel->dense_ring_offset =
			    stage->channels[channel->site].dense_ring_offset;
		    channel->dense_ring_bank_bytes =
			    stage->channels[channel->site].dense_ring_bank_bytes;
		    continue;
	    }
	    const uint64_t table_offset = take(
		    &next, (uint64_t)ranks * ORDERED_COLLECTIVE_PEER_BYTES);
	    const uint64_t region =
		    take(&next, product(stride, ORDERED_COLLECTIVE_SLOTS));
	    if (description->exact_output_shards ||
		(description->rank_count == DS41F_TP4_RANKS &&
		 description->wire_format == ORDERED_PUSH_BF16)) {
		    const uint64_t payload_bytes =
			    batch_rings ?
				    (uint64_t)stage->ring_rows * DS41F_HIDDEN *
					    sizeof(uint16_t) :
			    channel->site == 0 ?
				    (uint64_t)LINEAR_MAXIMUM_ROWS *
					    DS41F_EXPERTS_PER_ROW *
					    DS41F_EXPERT_INNER *
					    sizeof(uint16_t) :
				    (uint64_t)LINEAR_MAXIMUM_ROWS *
					    DS41F_HIDDEN * sizeof(float);
		    const uint64_t maximum_packs =
			    description->exl3_all_targets &&
					    channel->site == 0 && !batch_rings ?
				    (uint64_t)LINEAR_MAXIMUM_ROWS *
					    DS41F_EXPERTS_PER_ROW *
					    DS41F_TP4_LARGE_INNER * sizeof(uint16_t) /
					    EXACT_RING_PACK_BYTES :
				    payload_bytes / ranks /
					    EXACT_RING_PACK_BYTES;
		    /* a bank a row count: one size, the gathers' slots of the most packs (the thin banks: room for the
             * BF16 sum's of 8 rows too); batch_rings: the BF16 sum's slots of its own rows (12 a pack: 64 rows in a
             * gather-sized bank overran it into the next channel's peer table, pf17) */
		    const uint64_t row_packs = (uint64_t)DS41F_HIDDEN *
					       sizeof(uint16_t) / ranks /
					       EXACT_RING_PACK_BYTES;
		    channel->ring_row_bytes =
			    batch_rings ?
				    (row_packs * EXACT_RING_SUM_EPOCH_HOPS *
					     EXACT_RING_PACK_BYTES +
				     DS41F_ARRAY_ALIGNMENT_BYTES - 1u) /
					    DS41F_ARRAY_ALIGNMENT_BYTES *
					    DS41F_ARRAY_ALIGNMENT_BYTES :
				    0;
		    channel->ring_bank_bytes =
			    ((batch_rings ?
				      0 :
				      maximum_packs * EXACT_RING_EPOCH_HOPS *
					      EXACT_RING_PACKET_BYTES) +
			     EXACT_RING_METADATA_BYTES +
			     DS41F_ARRAY_ALIGNMENT_BYTES - 1u) /
			    DS41F_ARRAY_ALIGNMENT_BYTES *
			    DS41F_ARRAY_ALIGNMENT_BYTES;
		    channel->ring_offset = take(
			    &next, ds41f_ring_bank_start(channel->ring_bank_bytes,
						       channel->ring_row_bytes,
						       stage->ring_rows + 1u));
	    }
	if (description->exact_dense) {
            /* separate banks: a bank keeps one gather geometry per prepared row count (two-epoch slots); site 0 the
             * groups (wo_a's 8192 columns), site 1 the sublayer output (5120) */
	    const uint64_t dense_payload_bytes =
		    (uint64_t)LINEAR_MAXIMUM_ROWS * sizeof(uint16_t) *
		    (channel->site == 0 ? DS41F_WIDEST_ROW : DS41F_HIDDEN);
	    const uint64_t dense_packs =
		    dense_payload_bytes / ranks / EXACT_RING_PACK_BYTES;
	    channel->dense_ring_bank_bytes =
		    (dense_packs * EXACT_RING_EPOCH_HOPS *
			     EXACT_RING_PACKET_BYTES +
		     EXACT_RING_METADATA_BYTES +
		     DS41F_ARRAY_ALIGNMENT_BYTES - 1u) /
		    DS41F_ARRAY_ALIGNMENT_BYTES *
		    DS41F_ARRAY_ALIGNMENT_BYTES;
	    channel->dense_ring_offset =
		    take(&next,
			 channel->dense_ring_bank_bytes * LINEAR_MAXIMUM_ROWS);
	}
	channel->collective.arguments = (struct ordered_push_arguments){
		.peer_table_offset = table_offset,
		.region_offset = region,
		.input_offset = signals,
		.output_offset = signals + bytes,
		.inbox_offset = signals + DS41F_TP_LOCAL_VECTORS * bytes,
		.step_offset = card->array_offset[DS41F_ARRAY_STEP],
		.parity_stride = stride,
		.deadline_cycles = DS41F_TP2_DEADLINE_CYCLES,
		.inbox_rank_stride = bytes,
		.element_count = (uint32_t)elements,
		.rank_count = ranks,
		.rank = description->rank,
		.payload_format = description->wire_format
	};
    }
    if (next == UINT64_MAX)
	    return STATUS_INVALID_ARGUMENT;
    stage->communication_byte_count = next - card->arena_byte_count;
    card->arena_byte_count = next;
    for (uint32_t index = 0; index < stage->channel_count; index++)
        stage->channels[index].collective.arena_byte_count = next;
    return STATUS_OK;
}

/* The layer's routed-expert width, from the pack: every expert matrix EXL3 of one width (one width a layer:
 * plan->exl3_bits; the pack's open refused a width this build does not decode), or none EXL3. A refusal names the
 * first matrix that breaks it, before anything loads. */
MUST_CHECK static result_status
tp2_stage_exl3_discover(const struct pack_record *pack, uint32_t layer,
			uint32_t *bits_out)
{
	uint32_t layer_width = 0;
	for (uint32_t expert = 0; expert < DS41F_EXPERTS; expert++) {
		for (uint32_t matrix = 0; matrix < DS41F_EXPERT_MATRICES;
		     matrix++) {
			char name[NAME_CAPACITY];
			const int length =
				snprintf(name, sizeof(name),
					 "layers.%u.ffn.experts.%u.%s", layer,
					 expert, expert_rules[matrix].name);
			if (length <= 0 || length >= NAME_CAPACITY)
				return STATUS_INVALID_ARGUMENT;
			const uint32_t found =
				pack_find_tensor(pack, name, (uint32_t)length);
			if (found == PACK_NOT_FOUND)
				return STATUS_CORRUPT_PACK;
			const struct pack_tensor *const tensor =
				&pack_tensors_of(pack)[found];
			const uint32_t width =
				tensor->weight_format == WEIGHT_FORMAT_EXL3 ?
					tensor->bits_per_weight :
					0u;
			if (expert == 0 && matrix == 0)
				layer_width = width;
			else if (width != layer_width) {
				LOG_ERROR(
					"%s: stored EXL3 width %" PRIu32
					" differs from layer %" PRIu32
					"'s first expert matrix (%" PRIu32
					"; 0: not EXL3): widths within a "
					"layer are not supported by this build",
					name, width, layer, layer_width);
				return STATUS_UNSUPPORTED;
			}
		}
	}
	*bits_out = layer_width;
	return STATUS_OK;
}

/* head_shards: the sampler's exchange area, peer table and settings copy (the result card: also the table of every
 * rank's copy) behind everything else of the stage; with
 * draft_shards, ranks 1-3's draft area or rank 0's share table and counters (ds41f_tp_stage.h). */
MUST_CHECK static result_status
tp2_stage_sampler_areas(struct ds41f_tp2_stage_plan *stage)
{
	const struct ds41f_tp2_stage_description *const description =
		&stage->description;
	struct ds41f_card *const card = &stage->model->cards[description->card];
	uint64_t next = card->arena_byte_count;
	const uint64_t peers = (uint64_t)TP_GROUP_MOST_RANKS * sizeof(uint64_t);
	stage->sampler_exchange = (struct linear_range){
		take(&next, (uint64_t)SAMPLER_SHARD_DEFAULT_EXCHANGE_BYTES),
		(uint64_t)SAMPLER_SHARD_DEFAULT_EXCHANGE_BYTES
	};
	stage->sampler_peers = (struct linear_range){
		take(&next, SAMPLER_SHARD_PEER_TABLE_BYTES),
		SAMPLER_SHARD_PEER_TABLE_BYTES
	};
	const uint64_t settings =
		SAMPLER_SETTINGS_SLOTS_OFFSET +
		(uint64_t)description->limits.session_capacity *
			SAMPLER_SETTINGS_SLOT_BYTES;
	stage->sampler_settings =
		(struct linear_range){ take(&next, settings), settings };
	if (description->include_head_and_draft)
		stage->sampler_settings_peers =
			(struct linear_range){ take(&next, peers), peers };
	if (description->draft_shards && description->include_head_and_draft) {
		const uint64_t counters =
			(uint64_t)DS41F_DRAFT_SHARE_RANKS * sizeof(uint32_t);
		stage->draft_peers =
			(struct linear_range){ take(&next, peers), peers };
		stage->draft_counters =
			(struct linear_range){ take(&next, counters),
					       counters };
	} else if (description->draft_shards) {
		const uint64_t draft_bytes =
			DS41F_DRAFT_SHARE_AREA_BYTES +
			(uint64_t)STEP_DRAFT_MOST *
				(DS41F_VOCABULARY_VALUE / TP_GROUP_MOST_RANKS *
					 sizeof(float) +
				 STEP_DRAFT_RESULT_BYTES +
				 DS41F_DRAFT_MARKOV_ROW_BYTES);
		stage->draft_area =
			(struct linear_range){ take(&next, draft_bytes),
					       draft_bytes };
	}
	if (next == UINT64_MAX)
		return STATUS_INVALID_ARGUMENT;
	stage->communication_byte_count = sum(stage->communication_byte_count,
					      next - card->arena_byte_count);
	card->arena_byte_count = next;
	for (uint32_t index = 0; index < stage->channel_count; index++)
		stage->channels[index].collective.arena_byte_count = next;
	return STATUS_OK;
}

/* Optional barrier storage is last, so all inherited offsets stay fixed. */
MUST_CHECK static result_status
tp2_stage_sampler_entry(struct ds41f_tp2_stage_plan *stage)
{
	struct ds41f_card *const card =
		&stage->model->cards[stage->description.card];
	uint64_t next = card->arena_byte_count;
	stage->sampler_entry = (struct linear_range){
		take(&next, (uint64_t)SAMPLER_SHARD_ENTRY_BYTES),
		(uint64_t)SAMPLER_SHARD_ENTRY_BYTES
	};
	if (next == UINT64_MAX)
		return STATUS_INVALID_ARGUMENT;
	stage->communication_byte_count = sum(stage->communication_byte_count,
					      next - card->arena_byte_count);
	card->arena_byte_count = next;
	for (uint32_t index = 0; index < stage->channel_count; index++)
		stage->channels[index].collective.arena_byte_count = next;
	return STATUS_OK;
}

/* exl3_tc_moe: the tensor-core MoE workspace (exl3_tc_records.h) of the calls it takes (at most 64 rows, a tail
 * program's tail_rows: ds41f_tp_stage.h) at the larger intermediate, then the prefill form's, behind everything
 * else; with tail_overlay a tail program's calls take theirs from the first over the second (stage.h). */
MUST_CHECK static result_status
tp2_stage_exl3_tc_workspace(struct ds41f_tp2_stage_plan *stage)
{
	struct ds41f_card *const card =
		&stage->model->cards[stage->description.card];
	uint64_t next = card->arena_byte_count;
	const uint64_t capacity = stage->description.limits.row_capacity;
	const uint64_t tail_rows = capacity < stage->description.tail_rows ?
					   capacity :
					   stage->description.tail_rows;
	const uint64_t prefill_rows =
		capacity < EXL3_TCP_CHUNK_ROWS ? capacity : EXL3_TCP_CHUNK_ROWS;
	const bool overlay = stage->description.tail_overlay &&
			     tail_rows > EXL3_TC_MOST_ROWS &&
			     prefill_rows > EXL3_TC_MOST_ROWS;
	const uint64_t calls = !overlay && tail_rows > EXL3_TC_MOST_ROWS ?
				       tail_rows :
				       EXL3_TC_MOST_ROWS;
	const uint64_t rows = capacity < calls ? capacity : calls;
	const uint64_t bytes = exl3_tc_workspace_bytes(
		rows * DS41F_EXL3_TC_PAIRS_PER_ROW, TC_WIDEST_INTERMEDIATE);
	stage->exl3_tc_workspace =
		(struct linear_range){ take(&next, bytes), bytes };
	const uint64_t tail_bytes =
		overlay ? exl3_tc_workspace_bytes(tail_rows *
							  DS41F_EXL3_TC_PAIRS_PER_ROW,
						  TC_WIDEST_INTERMEDIATE) :
			  0u;
	if (prefill_rows > EXL3_TC_MOST_ROWS) {
		const uint64_t prefill_bytes = exl3_tcp_workspace_bytes(
			prefill_rows, TC_WIDEST_INTERMEDIATE);
		const uint64_t behind =
			tail_bytes > bytes ?
				tail_bytes - bytes :
				0u; /* the tail call's past the first */
		const uint64_t span = prefill_bytes > behind ? prefill_bytes :
							       behind;
		stage->exl3_tcp_workspace =
			(struct linear_range){ take(&next, span), span };
	}
	stage->exl3_tail_workspace =
		!overlay ?
			stage->exl3_tc_workspace :
			(struct linear_range){
				stage->exl3_tc_workspace.byte_offset, tail_bytes
			};
	if (overlay && (stage->exl3_tcp_workspace.byte_offset !=
				stage->exl3_tc_workspace.byte_offset + bytes ||
			stage->exl3_tail_workspace.byte_offset + tail_bytes >
				stage->exl3_tcp_workspace.byte_offset +
					stage->exl3_tcp_workspace.byte_count))
		return STATUS_INVALID_ARGUMENT; /* the two must lie together */
	if (next == UINT64_MAX)
		return STATUS_INVALID_ARGUMENT;
	card->arena_byte_count = next;
	for (uint32_t index = 0; index < stage->channel_count;
	     index++) /* peer qualification compares them */
		stage->channels[index].collective.arena_byte_count = next;
	return STATUS_OK;
}

/* trunk_mma, a stage of more than 8 rows: the _64 form's float32 partials (ds41f_tp_stage.h). */
MUST_CHECK static result_status
tp2_stage_trunk_mma_partials(struct ds41f_tp2_stage_plan *stage)
{
	struct ds41f_card *const card =
		&stage->model->cards[stage->description.card];
	uint64_t next = card->arena_byte_count;
	const uint64_t bytes =
		(uint64_t)PROGRAM_BATCH_MOST_ROWS *
			DS41F_TRUNK_MMA_THIN_PARTIAL_VALUES * sizeof(float) +
		(uint64_t)DS41F_TRUNK_MMA_COUNTERS *
			sizeof(uint32_t); /* the fused form's counters at the end */
	stage->trunk_mma_partials =
		(struct linear_range){ take(&next, bytes), bytes };
	const uint64_t capacity = stage->description.limits.row_capacity;
	const uint64_t tail_rows = capacity < stage->description.tail_rows ?
					   capacity :
					   stage->description.tail_rows;
	if (stage->description.tail_sets && tail_rows > STEP_WIDE_LEAST_ROWS) {
		const uint64_t set_bytes = tail_rows *
					   DS41F_TRUNK_MMA_PARTIAL_VALUES *
					   sizeof(float);
		stage->trunk_mma_set_partials =
			(struct linear_range){ take(&next, set_bytes),
					       set_bytes };
	}
    if (next == UINT64_MAX)
	    return STATUS_INVALID_ARGUMENT;
    card->arena_byte_count = next;
    for (uint32_t index = 0; index < stage->channel_count; index++) /* peer qualification compares them */
        stage->channels[index].collective.arena_byte_count = next;
    return STATUS_OK;
}

/* batch: one result area a slab of a BATCH program (model_seam.h) on a card that samples, behind everything else. */
MUST_CHECK static result_status
tp2_stage_batch_results(struct ds41f_tp2_stage_plan *stage)
{
	struct ds41f_card *const card =
		&stage->model->cards[stage->description.card];
	uint64_t next = card->arena_byte_count;
	const uint64_t bytes = (uint64_t)PROGRAM_BATCH_MOST_ROWS /
			       LINEAR_MAXIMUM_ROWS *
			       PROGRAM_BATCH_SLAB_RESULT_BYTES;
	stage->batch_results =
		(struct linear_range){ take(&next, bytes), bytes };
	/* the round table right behind the slab results: the step driver downloads both as one result span */
	const uint32_t positions = stage->description.limits.draft_positions;
	/* draft_layers: every rank its own round table (its verify writes one) and batched draft area */
	const bool rounds_here = (stage->description.include_head_and_draft ||
				  stage->description.draft_layers) &&
				 positions != 0;
	const uint64_t sessions = stage->description.limits.session_capacity;
	if (rounds_here)
		stage->batch_rounds = (struct linear_range){
			take(&next, sessions * STEP_ROUND_BYTES),
			sessions * STEP_ROUND_BYTES
		};
	if (stage->description.batch_wide_head) {
		const uint64_t outputs =
			stage->description.head_shards ?
				DS41F_VOCABULARY_VALUE / TP_GROUP_MOST_RANKS :
				DS41F_VOCABULARY_VALUE;
		const uint64_t logits = (uint64_t)PROGRAM_BATCH_MOST_ROWS *
					outputs * sizeof(float);
		stage->batch_logits =
			(struct linear_range){ take(&next, logits), logits };
	}
    if (rounds_here) {
	    const uint64_t drafts =
		    stage->description.batch_draft_rows ?
			    0u :
			    (sessions - 1u) *
				    ds41f_batch_draft_offset(
					    DS41F_BATCH_DRAFT_PARTS,
					    positions);
	    stage->batch_drafts =
		    (struct linear_range){ take(&next, drafts), drafts };
	    if (stage->description.batch_draft_rows) {
		    const uint64_t rows =
			    ds41f_batch_rows_offset(DS41F_BATCH_ROWS_PARTS);
		    stage->batch_rows =
			    (struct linear_range){ take(&next, rows), rows };
	    }
    }
    if (next == UINT64_MAX)
	    return STATUS_INVALID_ARGUMENT;
    stage->communication_byte_count = sum(stage->communication_byte_count, next - card->arena_byte_count);
    card->arena_byte_count = next;
    for (uint32_t index = 0; index < stage->channel_count; index++)
        stage->channels[index].collective.arena_byte_count = next;
    return STATUS_OK;
}

result_status
ds41f_tp2_stage_plan_create(const struct pack_record *pack,
			  const struct ds41f_tp2_stage_description *description,
			  struct ds41f_tp2_stage_plan **plan_out)
{
	if (pack == NULL || description == NULL || plan_out == NULL ||
	    description->card >= BACKEND_MAXIMUM_CARDS ||
	    (description->original_batch &&
	     !ds41f_original_batch_stage_is_valid(description)) ||
	    description->wire_format > ORDERED_PUSH_BF16 ||
	    (description->rank_count != 1u &&
	     description->rank_count != DS41F_TP2_RANKS &&
	     description->rank_count != DS41F_TP4_RANKS) ||
	    description->rank >= description->rank_count ||
	    description->first_target_layer >= DS41F_LAYERS ||
	    (description->original_wide_experts &&
	     (!description->exact_output_shards ||
	      description->original_code_wire ||
	      description->wire_format != 0 ||
	      description->rank_count != DS41F_TP4_RANKS ||
	      description->first_target_layer != 0 ||
	      description->target_layer_count != DS41F_TP4_ATTENTION_PREFIX_END ||
	      description->include_head_and_draft ||
	      description->exl3_all_targets || description->exl3_wide_entry ||
	      description->exl3_tc_moe || description->batch ||
	      description->split_overlap != 0 ||
	      description->full_split_overlap != 0 ||
	      description->limits.row_capacity < STEP_WIDE_LEAST_ROWS ||
	      description->limits.linear_profile != LINEAR_PROFILE_PRECISION ||
	      description->limits.linear_order != LINEAR_ORDER_REFERENCE ||
	      description->limits.cmp_order != CMP_ORDER_REFERENCE)) ||
	    (description->attention_owner &&
	     (!description->exact_output_shards ||
	      description->rank_count != DS41F_TP4_RANKS ||
	      description->wire_format != 0 ||
	      description->first_target_layer != 0 ||
	      description->target_layer_count != DS41F_TP4_ATTENTION_PREFIX_END ||
	      description->include_head_and_draft)) ||
	    description->target_layer_count == 0 ||
	    description->target_layer_count >
		    DS41F_LAYERS - description->first_target_layer ||
	    (description->include_embedding &&
	     (description->first_target_layer != 0 ||
	      (description->rank != 0 &&
	       description->rank_count != DS41F_TP4_RANKS))) ||
	    (description->exl3_all_targets &&
	     (description->rank_count != DS41F_TP4_RANKS ||
	      description->card != description->rank ||
	      description->first_target_layer != 0 ||
	      description->target_layer_count != DS41F_LAYERS ||
	      description->original_code_wire ||
	      description->wire_format != (description->exact_output_shards ?
						   0u :
						   ORDERED_PUSH_BF16) ||
	      description->attention_owner ||
	      (!description->exact_output_shards &&
	       description->exl3_product_order != BB_EXL3_REFERENCE_ORDER) ||
	      (description->include_head_and_draft &&
	       description->rank != 0))) ||
	    (description->exl3_product_order != BB_EXL3_REFERENCE_ORDER &&
	     (!description->exl3_all_targets ||
	      description->exl3_product_order != BB_EXL3_EIGHT_PART_ORDER)) ||
	    (description->head_shards && !description->exl3_all_targets) ||
	    (description->exact_dense && (!description->exl3_all_targets ||
					  !description->exact_output_shards)) ||
	    (description->draft_shards && !description->head_shards) ||
	    (description->draft_layers &&
	     (!description->draft_shards || !description->exl3_all_targets ||
	      description->exact_output_shards ||
	      !description->include_embedding ||
	      description->limits.draft_positions == 0 ||
	      description->rank_count != DS41F_TP4_RANKS ||
	      (description->batch && (!description->batch_draft_rows ||
				      !description->batch_wide_head)))) ||
	    (description->batch && !description->exl3_all_targets &&
	     !description->original_batch) ||
	    ((description->batch_wide_head ||
	      description->batch_context_slabs) &&
	     !description->batch) ||
	    (description->batch_draft_rows &&
	     (!description->batch ||
	      (description->draft_shards && !description->draft_layers))) ||
	    (description->head_stream && !description->exl3_all_targets) ||
	    (description->include_head_and_draft &&
	     ((description->rank_count != 1u &&
	       !description->exl3_all_targets) ||
	      description->limits.draft_positions == 0 ||
	      description->first_target_layer > DS41F_DRAFT_FIRST_TAP_LAYER ||
	      description->target_layer_count !=
		      DS41F_LAYERS - description->first_target_layer)))
		return STATUS_INVALID_ARGUMENT;
	result_status status = model_check(pack);
	status = status == STATUS_OK ? limits_check(&description->limits) :
				       status;
	if (status != STATUS_OK)
		return status;
	struct ds41f_tp2_stage_plan *const stage =
		FAULT_POINT(model_v41_tp2_stage_memory) ?
			NULL :
			calloc(1, sizeof(*stage));
	if (stage == NULL)
		return STATUS_OUT_OF_MEMORY;
	stage->description = *description;
	stage->vision_excluded = true;
	stage->external_draft_embedding = description->include_head_and_draft &&
					  !description->include_embedding;
	const uint64_t targets = (BIT(description->target_layer_count) - 1u)
				 << description->first_target_layer;
	const uint64_t walked = targets |
				(description->include_head_and_draft ||
						 description->draft_layers ?
					 DS41F_DRAFT_LAYERS :
					 0);
	uint64_t attention_layers = walked;
	if (description->attention_owner) {
		attention_layers = 0;
		for (uint32_t layer = 0; layer < DS41F_TP4_ATTENTION_PREFIX_END;
		     layer++)
			attention_layers |= ds41f_tp4_attention_owner(layer) ==
							    description->rank ?
						    BIT(layer) :
						    0;
	}
    const uint32_t capacity = tp2_stage_jobs(walked, description);
    struct tp2_stage_pending *const pending =
	    FAULT_POINT(model_v41_tp2_stage_pending_memory) ?
		    NULL :
		    calloc(capacity, sizeof(*pending));
    stage->jobs = FAULT_POINT(model_v41_tp2_stage_jobs_memory) ?
			  NULL :
			  calloc(capacity, sizeof(*stage->jobs));
    stage->model = FAULT_POINT(model_v41_tp2_stage_model_memory) ?
			   NULL :
			   calloc(1, sizeof(*stage->model));
    if (pending == NULL || stage->jobs == NULL || stage->model == NULL)
	    status = STATUS_OUT_OF_MEMORY;
    struct ds41f_plan *const model = stage->model;
    if (status == STATUS_OK) {
	    model->limits = description->limits;
	    /* a TP stage's: chunk_core and queries_piece read a chunk */
	    model->chunk_attention_arrays = true;
	    model->card_count = description->card + 1u;
	    model->embedding_card = description->include_embedding ?
					    description->card :
					    BACKEND_MAXIMUM_CARDS;
	    model->result_card = description->include_head_and_draft ?
					 description->card :
					 BACKEND_MAXIMUM_CARDS;
	    for (uint32_t tensor = 0; tensor < DS41F_TENSOR_GLOBALS; tensor++)
		    model->globals[tensor].card = BACKEND_MAXIMUM_CARDS;
	    for (uint32_t tensor = 0; tensor < DS41F_TENSOR_DRAFT_GLOBALS; tensor++)
		    model->draft_globals[tensor].card = BACKEND_MAXIMUM_CARDS;
	    tp2_stage_holders(model, description->card, targets,
			      description->include_head_and_draft ||
				      description->draft_layers,
			      description->include_head_and_draft);
	    model->cards[description->card].holders |=
		    (uint32_t)(tp2_stage_head_rows(description) ?
				       BIT(DS41F_HOLDER_RESULT) :
				       0);
    }
    for (uint32_t tensor = 0; status == STATUS_OK && tensor < DS41F_TENSOR_GLOBALS;
	 tensor++) {
	    const bool head_rows = tensor != DS41F_TENSOR_EMBEDDING &&
				   tp2_stage_head_rows(description);
	    if (tensor == DS41F_TENSOR_EMBEDDING ?
			!description->include_embedding :
			!description->include_head_and_draft && !head_rows)
		    continue;
	    /* head_shards: ranks 1-3 their rows of the head; rank 0 too where the draft is cut (draft_shards) */
	    const bool head_cut = tensor == DS41F_TENSOR_HEAD &&
				  description->head_shards &&
				  (head_rows || description->draft_shards);
	    status = tp2_stage_tensor_add(pack, stage, pending, capacity,
					  global_rules[tensor].name,
					  &global_rules[tensor], head_cut,
					  false, &model->globals[tensor], 0,
					  NULL, description->rank);
    }
    for (uint32_t tensor = 0;
	 status == STATUS_OK && description->include_head_and_draft &&
	 tensor < DS41F_TENSOR_DRAFT_GLOBALS;
	 tensor++)
	    status = tp2_stage_tensor_add(pack, stage, pending, capacity,
					  draft_rules[tensor].name,
					  &draft_rules[tensor], false, false,
					  &model->draft_globals[tensor], 0,
					  NULL, description->rank);
    for (uint32_t tensor = DS41F_TENSOR_DRAFT_MARKOV_EMBEDDING;
	 status == STATUS_OK && tp2_stage_head_rows(description) &&
	 description->draft_shards &&
	 tensor < DS41F_TENSOR_DRAFT_MARKOV_EMBEDDING + TP2_STAGE_MARKOV_TABLES;
	 tensor++)
	    status = tp2_stage_tensor_add(pack, stage, pending, capacity,
					  draft_rules[tensor].name,
					  &draft_rules[tensor], false, false,
					  &model->draft_globals[tensor], 0,
					  NULL, description->rank);
    /* draft_layers' ranks 1-3: their draft context's main_proj and main_norm, the draft's final norm */
    static const uint32_t rank_draft_globals[TP2_STAGE_RANK_DRAFT_GLOBALS] = {
	    DS41F_TENSOR_DRAFT_MAIN_PROJECTION, DS41F_TENSOR_DRAFT_MAIN_NORM,
	    DS41F_TENSOR_DRAFT_NORM
    };
    for (uint32_t index = 0; status == STATUS_OK && description->draft_layers &&
			     !description->include_head_and_draft &&
			     index < TP2_STAGE_RANK_DRAFT_GLOBALS;
	 index++)
	    status = tp2_stage_tensor_add(
		    pack, stage, pending, capacity,
		    draft_rules[rank_draft_globals[index]].name,
		    &draft_rules[rank_draft_globals[index]], false, false,
		    &model->draft_globals[rank_draft_globals[index]], 0, NULL,
		    description->rank);
    for (uint32_t layer = 0;
	 status == STATUS_OK && layer < DS41F_WALK_LAYERS; layer++) {
	    if ((walked & BIT(layer)) == 0)
		    continue;
	    char prefix[LAYER_PREFIX_CAPACITY];
	    layer_prefix_of(layer, prefix);
	    if (layer < DS41F_LAYERS) {
		    status = tp2_stage_exl3_discover(pack, layer,
						     &model->exl3_bits[layer]);
		    if (status == STATUS_OK &&
			((model->exl3_bits[layer] != 0) !=
			 description->exl3_all_targets))
			    status = STATUS_UNSUPPORTED;
		    /* --exl3_tc_moe: the tensor-core families decode EXL3_TC_SUPPORTED_BITS_X2 only (no K4) */
		    if (status == STATUS_OK && description->exl3_tc_moe &&
			model->exl3_bits[layer] != 0 &&
			!exl3_tc_width_supported(
				exl3_bits_x2(model->exl3_bits[layer]))) {
			    LOG_ERROR(
				    "layers.%u.ffn.experts.0.w1.weight: EXL3 stored width %" PRIu32
				    " has no tensor-core MoE entries in this build"
				    " (--exl3_tc_moe takes K2, K2.5 and K3)",
				    layer, model->exl3_bits[layer]);
			    status = STATUS_UNSUPPORTED;
		    }
		    if (status == STATUS_OK && model->exl3_bits[layer] != 0 &&
			description->exact_output_shards)
			    status = bb_exl3_output_shard_prepare(
				    DS41F_HIDDEN, DS41F_EXPERT_INNER,
				    model->exl3_bits[layer],
				    description->rank_count, description->rank,
				    description->exl3_product_order,
				    &stage->exl3_shards[layer]);
	    }
	    for (uint32_t tensor = 0;
		 status == STATUS_OK && tensor < DS41F_TENSOR_LAYER_COUNT;
		 tensor++) {
		    const struct tensor_rule *const rule =
			    tp2_stage_rule(layer, tensor);
		    if ((rule->layers & BIT(layer)) == 0)
			    continue;
		    char name[NAME_CAPACITY];
		    const int length = snprintf(name, sizeof(name), "%s%s",
						prefix, rule->name);
		    if (length <= 0 || length >= NAME_CAPACITY) {
			    status = STATUS_INVALID_ARGUMENT;
			    break;
		    }
		    /* exact_dense: wq_b by heads, wo_a by groups, wo_b by output channels, each output's full-K
		     * chain */
		    const bool dense_tensor = tensor == DS41F_TENSOR_QUERY_UP ||
					      tensor ==
						      DS41F_TENSOR_OUTPUT_GROUPS ||
					      tensor == DS41F_TENSOR_OUTPUT;
		    const bool split_outputs =
			    description->exact_output_shards ?
				    description->exact_dense && dense_tensor :
				    (tensor == DS41F_TENSOR_QUERY_UP ||
				     tensor == DS41F_TENSOR_OUTPUT_GROUPS ||
				     tensor == DS41F_TENSOR_SHARED_GATE ||
				     tensor == DS41F_TENSOR_SHARED_UP);
		    const bool split_inputs =
			    !description->exact_output_shards &&
			    (tensor == DS41F_TENSOR_OUTPUT ||
			     tensor == DS41F_TENSOR_SHARED_DOWN);
		    status = tp2_stage_tensor_add(
			    pack, stage, pending, capacity, name, rule,
			    split_outputs, split_inputs,
			    &model->layer_tensors[layer][tensor], layer, NULL,
			    description->rank);
	    }
	    const uint32_t experts = layer >= DS41F_DRAFT_FIRST_LAYER ?
					     DS41F_DRAFT_EXPERTS :
					     DS41F_EXPERTS;
	    for (uint32_t expert = 0; status == STATUS_OK && expert < experts;
		 expert++) {
		    for (uint32_t matrix = 0;
			 status == STATUS_OK && matrix < DS41F_EXPERT_MATRICES;
			 matrix++) {
			    char name[NAME_CAPACITY];
			    const int length = snprintf(
				    name, sizeof(name), "%sffn.experts.%u.%s",
				    prefix, expert, expert_rules[matrix].name);
			    if (length <= 0 || length >= NAME_CAPACITY) {
				    status = STATUS_INVALID_ARGUMENT;
				    break;
			    }
			    struct tensor_rule rule = expert_rules[matrix];
			    if (model->exl3_bits[layer] != 0) {
				    rule.weight_format = WEIGHT_FORMAT_EXL3;
				    rule.bits_per_weight =
					    model->exl3_bits[layer];
			    }
			    /* exl3_expert_rotation: the chunk of ds41f_exl3_tc_chunk_of (exl3_tc_records.h) */
			    const uint32_t chunk_rank =
				    description->exl3_expert_rotation != 0 &&
						    layer < DS41F_LAYERS &&
						    model->exl3_bits[layer] !=
							    0 &&
						    description->rank_count ==
							    DS41F_TP4_RANKS ?
					    ds41f_exl3_tc_chunk_of(
						    description
							    ->exl3_expert_rotation,
						    description->rank, layer,
						    expert) :
					    description->rank;
			    status = tp2_stage_tensor_add(
				    pack, stage, pending, capacity, name, &rule,
				    description->exact_output_shards ||
					    matrix != DS41F_TENSOR_DOWN,
				    !description->exact_output_shards &&
					    matrix == DS41F_TENSOR_DOWN,
				    NULL, layer,
				    &model->expert_tables[layer][expert][matrix],
				    chunk_rank);
		    }
	    }
    }
    if (status == STATUS_OK && stage->job_count != capacity)
	    status = STATUS_INVALID_ARGUMENT;
    uint64_t next = 0;
    if (status == STATUS_OK)
	    qsort(pending, stage->job_count, sizeof(*pending),
		  tp2_stage_pending_compare);
    for (uint32_t layer = 0; model != NULL && layer < DS41F_WALK_LAYERS;
	 layer++)
	    model->experts[layer].byte_offset = UINT64_MAX;
    for (uint32_t index = 0; status == STATUS_OK && index < stage->job_count;
	 index++) {
	    struct tp2_stage_pending *const item = &pending[index];
	    item->tensor.job.destination_offset =
		    take(&next, item->tensor.byte_count);
	    stage->jobs[index] = item->tensor.job;
	    stage->tensor_payload_byte_count = sum(
		    stage->tensor_payload_byte_count, item->tensor.byte_count);
	    if (item->tensor.stored != NULL)
		    *item->tensor.stored = (struct linear_range){
			    item->tensor.job.destination_offset,
			    item->tensor.byte_count
		    };
	    if (item->tensor.expert_offset != NULL)
		    *item->tensor.expert_offset =
			    item->tensor.job.destination_offset;
	    if (item->is_expert) {
		    struct linear_range *const range =
			    &model->experts[item->layer];
		    range->byte_offset =
			    item->tensor.job.destination_offset <
					    range->byte_offset ?
				    item->tensor.job.destination_offset :
				    range->byte_offset;
		    const uint64_t end =
			    sum(item->tensor.job.destination_offset,
				item->tensor.byte_count);
		    range->byte_count =
			    end > range->byte_count ? end : range->byte_count;
	    }
	    if (next == UINT64_MAX ||
		stage->tensor_payload_byte_count == UINT64_MAX)
		    status = STATUS_INVALID_ARGUMENT;
    }
    free(pending);
    for (uint32_t layer = 0; model != NULL && layer < DS41F_WALK_LAYERS;
	 layer++)
	    model->experts[layer] =
		    model->experts[layer].byte_offset == UINT64_MAX ?
			    (struct linear_range){ 0, 0 } :
			    (struct linear_range){
				    model->experts[layer].byte_offset,
				    model->experts[layer].byte_count -
					    model->experts[layer].byte_offset
			    };
    if (status == STATUS_OK) {
	    struct ds41f_card *const card = &model->cards[description->card];
	    stage->aligned_weight_byte_count = card->state_offset = next;
	    status = card_lay_out_attention(model, card, attention_layers) ?
			     STATUS_OK :
			     STATUS_INVALID_ARGUMENT;
	    if (status == STATUS_OK) {
		    stage->base_state_byte_count =
			    card->arena_byte_count - card->state_offset;
		    stage->draft_embedding_byte_count =
			    description->include_head_and_draft ?
				    DS41F_DRAFT_EMBEDDING_BYTES :
				    0;
		    card->image =
			    FAULT_POINT(model_v41_tp2_stage_image_memory) ?
				    NULL :
				    calloc(1, card->image_byte_count);
		    if (card->image == NULL)
			    status = STATUS_OUT_OF_MEMORY;
	    }
	    if (status == STATUS_OK) {
		    float words[DS41F_THETAS][DS41F_FREQUENCIES];
		    ds41f_frequency_words(words);
		    memcpy(card->image +
				   card->array_offset[DS41F_ARRAY_FREQUENCIES] -
				   image_offset_of(card),
			   words, sizeof(words));
		    for (uint32_t layer = 0; layer < DS41F_WALK_LAYERS;
			 layer++) {
			    if ((walked & BIT(layer)) != 0)
				    memcpy(card->image +
						   card->expert_table_offset
							   [layer] -
						   image_offset_of(card),
					   model->expert_tables[layer],
					   sizeof(model->expert_tables[layer]));
			    if ((targets & BIT(layer)) == 0) {
				    stage->external_pool_sources |=
					    card->pool_offset[layer] !=
							    PROGRAM_NOWHERE ?
						    BIT(layer) :
						    0;
				    stage->external_key_sources |=
					    card->key_tiles_offset[layer] !=
							    PROGRAM_NOWHERE ?
						    BIT(layer) :
						    0;
				    stage->external_selection_groups |=
					    card->selection_offset[layer] !=
							    PROGRAM_NOWHERE ?
						    BIT(layer) :
						    0;
			    }
		    }
		    status = tp2_stage_channels(stage);
		    if (status == STATUS_OK && description->head_shards)
			    status = tp2_stage_sampler_areas(stage);
		    if (status == STATUS_OK && description->batch &&
			(description->head_shards ||
			 description->include_head_and_draft))
			    status = tp2_stage_batch_results(stage);
		    if (status == STATUS_OK && description->exl3_tc_moe &&
			description->exl3_all_targets)
			    status = tp2_stage_exl3_tc_workspace(stage);
		    if (status == STATUS_OK && description->trunk_mma &&
			description->limits.row_capacity > LINEAR_MAXIMUM_ROWS)
			    status = tp2_stage_trunk_mma_partials(stage);
		    if (status == STATUS_OK && description->router_mma &&
			(description->batch ||
			 (description->router_mma_wide &&
			  description->limits.row_capacity >
				  PROGRAM_BATCH_MOST_ROWS))) {
			    /* router_mma: [splits][64 rows (router_mma_wide: a wide launch's 1,024)][the larger
                 * router's 384 outputs] float32 */
			    uint64_t router_next = card->arena_byte_count;
			    const uint64_t rows =
				    description->router_mma_wide &&
						    description->limits
								    .row_capacity >
							    PROGRAM_BATCH_MOST_ROWS ?
					    LINEAR_WIDE_MOST_ROWS :
					    PROGRAM_BATCH_MOST_ROWS;
			    const uint64_t bytes =
				    (uint64_t)DS41F_ROUTER_MMA_SPLITS *
				    rows * DS41F_EXPERTS * sizeof(float);
			    stage->router_partials = (struct linear_range){
				    take(&router_next, bytes), bytes
			    };
			    card->arena_byte_count = router_next;
			    for (uint32_t index = 0;
				 index < stage->channel_count; index++)
				    stage->channels[index]
					    .collective.arena_byte_count =
					    router_next;
			    status = router_next == UINT64_MAX ?
					     STATUS_INVALID_ARGUMENT :
					     status;
		    }
	    }
	    if (status == STATUS_OK && description->head_shards)
		    status = tp2_stage_sampler_entry(stage);
	    if (status == STATUS_OK) {
		    struct tp_load_description load = {
			    .pack = pack,
			    .jobs = stage->jobs,
			    .job_count = stage->job_count,
			    .card_count = model->card_count,
			    .maximum_extent_byte_count =
				    TP_LOAD_MAXIMUM_EXTENT_BYTES
		    };
		    load.card_arena_byte_count[description->card] =
			    card->arena_byte_count;
		    status = tp_load_prepare(&load, &stage->load);
	    }
    }
    if (status == STATUS_OK)
	    *plan_out = stage;
    else
	    ds41f_tp2_stage_plan_destroy(stage);
    return status;
}

void ds41f_tp2_stage_plan_destroy(struct ds41f_tp2_stage_plan *plan)
{
    if (plan != NULL) {
	    ds41f_plan_destroy(plan->model);
	    free(plan->jobs);
	    free(plan->channels);
    }
    free(plan);
}
