/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * ds41f_tp_program_layers.c: split and half selection, chunk cores, index spans and gathers, layer steps, owner layers
 * (split from model_v41_program.c by whole items).
 */
#include "common/checked_size.h"
#include "models/deepseek_v41_flash/plan/model_v41.h"
#include "ds41f_tp_layer.h"
#include "ds41f_tp_program.h"
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
#include "ds41f_draft_share_records.h"
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

#include "models/deepseek_v41_flash/builder/ds41f_builder.h"
#include "ds41f_tp_program_internal.h"

/* batch_core_once: the window writes once over the rows too */
#define BATCH_CORE_WINDOWS 2u
#define BATCH_CORE_POOLS \
	3u /* whole-step pools, followed by separate compressor slot retention */
/* split_layer's halves_parts: the attention's runs and the FFN's (split_overlap's modes) */
#define SPLIT_ATTENTION_FIRST_PART 1u
#define SPLIT_ATTENTION_LAST_PART 6u
#define SPLIT_FFN_FIRST_PART 7u
#define SPLIT_FFN_LAST_PART 9u
#define GATHER_INDEX_PHASE 0u
#define AHEAD_STEPS                                                   \
	(DS41F_ENGRAM_STEPS | DS41F_SITE_MIX | DS41F_INPUT_PRODUCTS | \
	 DS41F_POOL_PRODUCTS) /* a chunked layer's first products */
#define QUERY_STEPS (DS41F_QUERY_PRODUCTS | DS41F_INDEX_WEIGHTS_PRODUCT)

/* split_overlap (ds41f_tp_stage.h) applies to the card's layer: the BF16 recipe's EXL3 tensor layer of a
 * wide or
 * full program with half views, its attention summed by the wide exchange and its FFN a slab ring at a time. */
static bool split_applies(const struct ds41f_builder *b, uint32_t layer)
{
	const struct ds41f_tp2_layer_plan *const tensor = b->tensor_layer;
	return tensor != NULL && tensor->split_overlap &&
	       b->half_view_first != 0 && b->chunk_first == 0 &&
	       tensor->shared_channels && tensor->bf16_ring &&
	       !tensor->exact_output_shards && !tensor->exact_dense &&
	       tensor->exl3_wide_entry && tensor->wide_exchange != 0 &&
	       tensor->wide_exchange != DS41F_TP_EXCHANGE_OWNER_RINGS &&
	       !tensor->bf16_slab_rings &&
	       tensor->ring_rows <= LINEAR_MAXIMUM_ROWS &&
	       b->plan->exl3_bits[layer] != 0 && !b->vision_active &&
	       b->made->program.kind != PROGRAM_BATCH &&
	       b->half_rows > EXL3_TC_MOST_ROWS && !ds41f_tail_program(b) &&
	       b->chunk_rows >= tensor->split_least_rows &&
	       b->chunk_rows - b->half_rows <=
		       LINEAR_WIDE_MOST_ROWS; /* each half a wide launch */
}

/* Half h of the chunk (rows, slabs, view) for chunk_select, piece_select and slab_select; h ==
 * CHUNK_HALVES: the whole
 * again. */
static void half_select(struct ds41f_builder *b, uint32_t half,
			uint32_t chunk_rows)
{
	b->chunk_first = half == 1u ? b->half_rows : 0u;
	b->chunk_rows = half == 0  ? b->half_rows :
			half == 1u ? chunk_rows - b->half_rows :
				     chunk_rows;
	b->chunk_view = half < DS41F_CHUNK_HALVES ? b->half_view_first + half :
						    0u;
	b->slab_count = (b->chunk_rows + LINEAR_MAXIMUM_ROWS - 1u) /
			LINEAR_MAXIMUM_ROWS;
	ds41f_chunk_select(b);
}

/* split_overlap's pipeline (ds41f_tp_stage.h) applies to the card's layer: split_layer takes its attention too
 * (spans of whole half views). */
static bool pipelined(const struct ds41f_builder *b, uint32_t layer)
{
	return split_applies(b, layer) &&
	       b->tensor_layer->split_overlap >= DS41F_SPLIT_PIPELINE &&
	       layer != DS41F_DECODER_FIRST_LAYER &&
	       b->half_rows % DS41F_SPAN_ROWS == 0;
}

/* Exact original attention deliberately clears tensor_layer: its products and query/core arrays retain all
 * 64 heads. Consult the stage only for the opt-in, single-piece core schedule; no shard or arithmetic is selected.
 * Original BATCH models' FULL prefills stay excluded too, as do the separate wide-expert and owner experiments. */
static bool original_chunk_core_applies(const struct ds41f_builder *b)
{
	if (b->tensor_layer != NULL || b->tp_description == NULL ||
	    b->tp_tensor == NULL || b->card >= DS41F_TP4_RANKS ||
	    b->arithmetic_class != ARITHMETIC_CLASS_SM_80 ||
	    b->slab_count <= 1u || ds41f_piece_count_of(b) != 1u ||
	    b->chunk_rows % LINEAR_MAXIMUM_ROWS != 0 || b->vision_active ||
	    b->made->program.kind == PROGRAM_BATCH ||
	    b->tp_description->stages[b->card] == NULL)
		return false;
	const struct ds41f_tp2_stage_description *const stage =
		&b->tp_description->stages[b->card]->description;
	const struct program_limits *const limits = &stage->limits;
	return stage->chunk_core && stage->exact_output_shards &&
	       stage->wire_format == ORDERED_PUSH_FP32 &&
	       !stage->original_code_wire && !stage->exl3_all_targets &&
	       !stage->exact_dense && !stage->attention_owner &&
	       !stage->original_batch && !stage->batch &&
	       !stage->original_wide_experts && stage->split_overlap == 0 &&
	       stage->full_split_overlap == 0 &&
	       stage->rank_count == DS41F_TP4_RANKS &&
	       stage->first_target_layer == 0 &&
	       stage->target_layer_count == DS41F_TP4_ATTENTION_PREFIX_END &&
	       limits->state_width == STATE_WIDTH_16 &&
	       limits->linear_profile == LINEAR_PROFILE_PRECISION &&
	       limits->linear_order == LINEAR_ORDER_REFERENCE &&
	       limits->cmp_order == CMP_ORDER_REFERENCE &&
	       b->tp_tensor->layer_index < DS41F_TP4_ATTENTION_PREFIX_END &&
	       b->plan->exl3_bits[b->tp_tensor->layer_index] == 0;
}

/* chunk_core (ds41f_tp_stage.h) applies to the selected chunk or half: its slabs' scans, then one core. */
static bool chunk_core_applies(const struct ds41f_builder *b)
{
	return ((b->tensor_layer != NULL && b->tensor_layer->chunk_core) ||
		original_chunk_core_applies(b)) &&
	       ds41f_piece_count_of(b) == 1u &&
	       b->made->program.kind != PROGRAM_BATCH;
}

/* queries_piece (ds41f_tp_stage.h): a wide program's queries with wq_b over each piece, not a span at a time. */
static bool queries_piece_applies(const struct ds41f_builder *b)
{
	return b->tensor_layer != NULL && b->tp_description != NULL &&
	       b->tp_description->stages[b->card]->description.queries_piece &&
	       !ds41f_thin_forms_of(b);
}

/* index_span (ds41f_tp_stage.h) applies to the selected chunk or half of a wide program of pool_chunk (no slab's
 * state writes between its scans) with the stage's span scores. */
static bool index_span_applies(const struct ds41f_builder *b)
{
	return b->tensor_layer != NULL &&
	       b->tensor_layer->index_rows > LINEAR_MAXIMUM_ROWS &&
	       b->slab_count > 1u && !ds41f_thin_forms_of(b) &&
	       b->tp_description != NULL &&
	       b->tp_description->stages[b->card]->description.pool_chunk;
}

/* The scans (CORE's index scan and selection) of the selected chunk's or half's slabs first_slab to first_slab +
 * slabs
 * - 1: each a slab at a time through its view, or (index_span) up to index_rows rows a launch through the chunk's view
 * from their first row on (ds41f_index_scan.h's step_row_first), each row's selection by a slab's kernels
 * (split_rows). */
MUST_CHECK static result_status index_scans(struct ds41f_builder *b,
					    uint32_t layer, uint32_t source,
					    uint32_t group, uint32_t first_slab,
					    uint32_t slabs)
{
	const bool spans = index_span_applies(b);
	const uint32_t step =
		spans ? b->tensor_layer->index_rows / LINEAR_MAXIMUM_ROWS : 1u;
	result_status status = DS41F_OK;
	for (uint32_t slab = first_slab;
	     slab < first_slab + slabs && status == DS41F_OK; slab += step) {
		if (!spans) {
			ds41f_slab_select(b, slab);
		} else {
			const uint32_t
				first = slab * LINEAR_MAXIMUM_ROWS,
				end = first_slab + slabs < slab + step ?
					      (first_slab + slabs) *
						      LINEAR_MAXIMUM_ROWS :
					      first + step * LINEAR_MAXIMUM_ROWS;
			const uint32_t last =
				end < b->chunk_rows ? end : b->chunk_rows;
			ds41f_rows_select(b, b->chunk_view,
					  b->chunk_first + first, last - first);
			/* the view's block and its rows up to the last one (rows_select's view 0 lies first rows
			 * in) */
			b->arrays[DS41F_ARRAY_STEP] = (struct linear_range){
				b->chunk_view == 0 ?
					b->plan->cards[b->card]
						.array_offset[DS41F_ARRAY_STEP] :
					b->arrays[DS41F_ARRAY_STEP].byte_offset,
				STEP_BLOCK_BYTES_VALUE +
					(uint64_t)last * STEP_ROW_BYTES_VALUE
			};
			b->index_row_first = first;
			b->index_grouped = true;
		}
		status = ds41f_layer_steps(b, layer, source, group, DS41F_SCAN);
		b->index_grouped = false;
	}
	return status;
}

/* index_gather applies to the selected half of an index layer: the attention site's sums by copy engines (direct
 * routes, four ranks, BF16) and whole quarters of slabs. */
static bool index_gather_applies(const struct ds41f_builder *b, uint32_t layer,
				 uint32_t group)
{
	return group == layer && layer != 0 && ds41f_gathers_apply(b);
}

/* split_overlap's pipeline, an index layer's half behind every slab's pool writes: the rank's quarter of its slabs
 * scanned (each row's selection by the scan every rank runs), those selection rows gathered at GATHER_INDEX_PHASE
 * (half A on the attention site's channel, beside it the FFN sums of the layer before; half B on the FFN site's,
 * beside it half A's attention sums): every rank holds every row's selection for a quarter of the scans. */
MUST_CHECK static result_status index_gather(struct ds41f_builder *b,
					     uint32_t layer, uint32_t source,
					     uint32_t group)
{
	const struct ds41f_tp2_layer_plan *const tensor = b->tensor_layer;
	uint32_t slabs = 0, first_slab = 0;
	ds41f_gather_share(b, b->slab_count, 1u, &first_slab, &slabs);
	result_status status =
		index_scans(b, layer, source, group, first_slab, slabs);
	ds41f_chunk_select(b);
	const uint64_t into = ((uint64_t)b->chunk_first +
			       (uint64_t)first_slab * LINEAR_MAXIMUM_ROWS) *
			      DS41F_SELECTION_ROW_BYTES;
	const uint64_t bytes = (uint64_t)slabs * LINEAR_MAXIMUM_ROWS *
			       DS41F_SELECTION_ROW_BYTES;
	struct ds41f_gather_range range = {
		b->plan->cards[b->card].selection_offset[layer] + into,
		bytes,
		{ 0 }
	};
	for (uint32_t rank = 0; rank < DS41F_TP4_RANKS; rank++) {
		const uint32_t card = tensor->card - tensor->rank + rank;
		range.peers[rank] = b->tp_description->views[card]
					    ->cards[card]
					    .selection_offset[layer] +
				    into;
	}
	return DS41F_THEN(ds41f_rows_gather(
		b, b->chunk_view == b->half_view_first + 1u ? 1u : 0u,
		DS41F_TP_EXCHANGE_LAYER_PHASES * tensor->layer_index +
			GATHER_INDEX_PHASE,
		&range, 1u));
}

/* The selected chunk's (split_overlap's pipeline: half's) attention to its wo_a and wo_b: its pieces' products, its
 * slabs' state writes and scans or cores in slab order; chunk_core: the window writes over its rows ahead of the slabs'
 * pool writes and scans (an index layer on copy-engine sums: index_gather's quarter and gather), then the core. */
MUST_CHECK static result_status attention_half(struct ds41f_builder *b,
					       uint32_t layer, uint32_t source,
					       uint32_t group)
{
	/* prefill_dead_halves: a dead half's index scan and gather left out (its core reads no selection: the cone) */
	const bool dead = ds41f_half_dead(b, layer, ds41f_selected_half(b));
	const bool chunk_core = chunk_core_applies(b);
	const bool gather = chunk_core && !dead &&
			    index_gather_applies(b, layer, group);
	const bool pool_chunk =
		chunk_core &&
		b->tp_description->stages[b->card]->description.pool_chunk;
	const uint32_t piece_queries =
			       queries_piece_applies(b) ? DS41F_QUERIES : 0u,
		       span_queries = piece_queries ^ DS41F_QUERIES;
	/* index_span: the slabs' scans after the slab loop, a span at a time (index_scans) */
	const bool scans = chunk_core && !gather && !dead,
		   spans = scans && index_span_applies(b);
	result_status status = ds41f_steps_over(
		b, layer, source, group,
		AHEAD_STEPS | DS41F_QUERY_NORM | QUERY_STEPS | piece_queries,
		true);
	ds41f_chunk_select(b);
	status = chunk_core ?
			 DS41F_THEN(ds41f_layer_steps(b, layer, source, group,
						      DS41F_WINDOW_WRITES)) :
			 status;
	status = pool_chunk ?
			 DS41F_THEN(ds41f_steps_over(b, layer, source, group,
						     DS41F_POOL_CHUNK, true)) :
			 status;
	status = DS41F_THEN(ds41f_steps_over(
		b, layer, source, group,
		!chunk_core ?
			DS41F_STATE_WRITES | span_queries | DS41F_CORE :
			(pool_chunk ? 0u : DS41F_STATE_POOL) | span_queries |
				(scans && !spans ? DS41F_SCAN : 0u),
		false));
	status = spans ? DS41F_THEN(index_scans(b, layer, source, group, 0,
						b->slab_count)) :
			 status;
	status = gather ? DS41F_THEN(index_gather(b, layer, source, group)) :
			  status;
	ds41f_chunk_select(b);
	status = chunk_core ?
			 DS41F_THEN(ds41f_layer_steps(b, layer, source, group,
						      DS41F_CORE_ALONE)) :
			 status;
	return DS41F_THEN(ds41f_layer_steps(b, layer, source, group,
					    DS41F_OUTPUT_PRODUCTS));
}

#define HALF_ATTENTION UINT32_MAX /* split_part's steps: attention_half's */

/* One part of split_layer: half h's launches of steps, or of the FFN's phases ffn; its first marked branch (a side
 * run's every one). */
MUST_CHECK static result_status split_part(struct ds41f_builder *b,
					   uint32_t layer, uint32_t source,
					   uint32_t group, uint32_t half,
					   uint32_t chunk_rows, uint32_t steps,
					   uint32_t ffn, uint32_t branch)
{
	const uint32_t first = b->made->cards[b->card].launch_count;
	half_select(b, half, chunk_rows);
	result_status status =
		ffn != 0 ? ds41f_exl3_wide_ffn_phases(b, layer, ffn) :
		steps == HALF_ATTENTION ?
			   attention_half(b, layer, source, group) :
			   ds41f_layer_steps(b, layer, source, group, steps);
	const uint32_t end = b->made->cards[b->card].launch_count;
	if (status == DS41F_OK && end == first)
		status =
			STATUS_INVALID_ARGUMENT; /* every part has launches: a branch never marks nothing */
	for (uint32_t launch = first;
	     launch < end && status == DS41F_OK && branch != LAUNCH_CHAIN;
	     launch++)
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
MUST_CHECK static result_status split_layer(struct ds41f_builder *b,
					    uint32_t layer, uint32_t source,
					    uint32_t group)
{
	/* split_overlap's modes: the attention's runs (parts 1 to 6), the FFN's (7 to 9), both or neither */
	const uint32_t mode = b->tensor_layer->split_overlap;
	struct split_step {
		uint32_t half, steps, ffn, branch;
	};
	static const struct split_step halves_parts[] = {
		{ 0, DS41F_OUTPUT_PRODUCTS, 0, LAUNCH_CHAIN },
		{ 0, DS41F_SITE_SUM, 0, LAUNCH_SIDE },
		{ 1, DS41F_OUTPUT_PRODUCTS, 0, LAUNCH_RESUME },
		{ 0, DS41F_SITE_POST_ONLY, 0, LAUNCH_JOIN },
		{ 1, DS41F_SITE_SUM, 0, LAUNCH_SIDE },
		{ 0, 0, DS41F_FFN_COMPUTE, LAUNCH_RESUME },
		{ 1, DS41F_SITE_POST_ONLY, 0, LAUNCH_JOIN },
		{ 0, 0, DS41F_FFN_SUM, LAUNCH_SIDE },
		{ 1, 0, DS41F_FFN_COMPUTE, LAUNCH_RESUME },
		{ 0, 0, DS41F_FFN_POST, LAUNCH_JOIN },
		{ 1, 0, DS41F_FFN_SUM | DS41F_FFN_POST, LAUNCH_CHAIN }
	};
	/* the joint mode: A's front and shared expert beside B's attention sums, B's front, the routed experts of the
     * whole chunk (half 2) in one call, A's FFN sums beside B's shared expert */
	static const struct split_step joint_parts[] = {
		{ 0, DS41F_OUTPUT_PRODUCTS, 0, LAUNCH_CHAIN },
		{ 0, DS41F_SITE_SUM, 0, LAUNCH_SIDE },
		{ 1, DS41F_OUTPUT_PRODUCTS, 0, LAUNCH_RESUME },
		{ 0, DS41F_SITE_POST_ONLY, 0, LAUNCH_JOIN },
		{ 1, DS41F_SITE_SUM, 0, LAUNCH_SIDE },
		{ 0, 0, DS41F_FFN_FRONT | DS41F_FFN_SHARED, LAUNCH_RESUME },
		{ 1, DS41F_SITE_POST_ONLY, 0, LAUNCH_JOIN },
		{ 1, 0, DS41F_FFN_FRONT, LAUNCH_CHAIN },
		{ 2, 0, DS41F_FFN_EXPERTS, LAUNCH_CHAIN },
		{ 0, 0, DS41F_FFN_SUM, LAUNCH_SIDE },
		{ 1, 0, DS41F_FFN_SHARED, LAUNCH_RESUME },
		{ 0, 0, DS41F_FFN_POST, LAUNCH_JOIN },
		{ 1, 0, DS41F_FFN_SUM | DS41F_FFN_POST, LAUNCH_CHAIN }
	};
	/* mode 7: the joint call's route and gate/up over the chunk, A's down GEMM and combine, A's FFN sums beside B's
     * down GEMM, combine and shared expert */
	static const struct split_step joint_down_parts[] = {
		{ 0, DS41F_OUTPUT_PRODUCTS, 0, LAUNCH_CHAIN },
		{ 0, DS41F_SITE_SUM, 0, LAUNCH_SIDE },
		{ 1, DS41F_OUTPUT_PRODUCTS, 0, LAUNCH_RESUME },
		{ 0, DS41F_SITE_POST_ONLY, 0, LAUNCH_JOIN },
		{ 1, DS41F_SITE_SUM, 0, LAUNCH_SIDE },
		{ 0, 0, DS41F_FFN_FRONT | DS41F_FFN_SHARED, LAUNCH_RESUME },
		{ 1, DS41F_SITE_POST_ONLY, 0, LAUNCH_JOIN },
		{ 1, 0, DS41F_FFN_FRONT, LAUNCH_CHAIN },
		{ 2, 0, DS41F_FFN_EXPERTS_FRONT, LAUNCH_CHAIN },
		{ 2, 0, DS41F_FFN_DOWN_FIRST, LAUNCH_CHAIN },
		{ 0, 0, DS41F_FFN_SUM, LAUNCH_SIDE },
		{ 2, 0, DS41F_FFN_DOWN_SECOND, LAUNCH_RESUME },
		{ 1, 0, DS41F_FFN_SHARED, LAUNCH_CHAIN },
		{ 0, 0, DS41F_FFN_POST, LAUNCH_JOIN },
		{ 1, 0, DS41F_FFN_SUM | DS41F_FFN_POST, LAUNCH_CHAIN }
	};
	const bool joint = mode == DS41F_SPLIT_JOINT,
		   joint_down = mode == DS41F_SPLIT_JOINT_DOWN;
	const struct split_step *const parts = joint_down ? joint_down_parts :
					       joint	  ? joint_parts :
							    halves_parts;
	const uint32_t last =
		(joint_down ? sizeof(joint_down_parts) /
				      sizeof(joint_down_parts[0]) :
		 joint	    ? sizeof(joint_parts) / sizeof(joint_parts[0]) :
			      sizeof(halves_parts) / sizeof(halves_parts[0])) -
		1u;
	const uint32_t chunk_rows = b->chunk_rows, next = layer + 1u;
	const bool pipeline = pipelined(
		b,
		layer); /* else, as layer 20 (its attention divided), mode 1 */
	const bool ahead =
		pipeline && next < DS41F_DRAFT_FIRST_TAP_LAYER &&
		pipelined(b, next) &&
		ds41f_tp2_program_layer_card(b->tp_description, next) ==
			ds41f_tp2_program_layer_card(b->tp_description, layer);
	/* prefill_dead_halves: a dead half's sums left out (half_dead), their side runs' resume and join then the
	 * chain */
	const bool dead[DS41F_CHUNK_HALVES] = { ds41f_half_dead(b, layer, 0u),
						ds41f_half_dead(b, layer, 1u) };
	bool unforked = false;
	result_status status = DS41F_OK;
	for (uint32_t part = pipeline && b->ahead_layer[b->card] == next ? 1u :
									   0u;
	     part < last && status == DS41F_OK; part++) {
		const bool marks = mode == DS41F_SPLIT_OVERLAP ||
				   mode >= DS41F_SPLIT_PIPELINE ||
				   (mode == DS41F_SPLIT_ATTENTION_ONLY &&
				    part >= SPLIT_ATTENTION_FIRST_PART &&
				    part <= SPLIT_ATTENTION_LAST_PART) ||
				   (mode == DS41F_SPLIT_FFN_ONLY &&
				    part >= SPLIT_FFN_FIRST_PART &&
				    part <= SPLIT_FFN_LAST_PART);
		const uint32_t half = parts[part].half;
		if (half < DS41F_CHUNK_HALVES && dead[half] &&
		    (parts[part].steps == DS41F_SITE_SUM ||
		     parts[part].ffn == DS41F_FFN_SUM)) {
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
		status = split_part(
			b, layer, source, group, half, chunk_rows,
			pipeline && parts[part].steps == DS41F_OUTPUT_PRODUCTS ?
				HALF_ATTENTION :
				parts[part].steps,
			parts[part].ffn, branch);
	}
	b->ahead_layer[b->card] = 0;
	const uint32_t last_ffn = dead[1] ? parts[last].ffn &
						    ~(uint32_t)DS41F_FFN_SUM :
					    parts[last].ffn;
	if (!ahead) {
		status = DS41F_THEN(split_part(b, layer, source, group,
					       parts[last].half, chunk_rows, 0,
					       last_ffn, LAUNCH_CHAIN));
	} else { /* the next layer's attention of A through its tensor plan, its pool's source and selection's group */
		status = dead[1] ? status :
				   DS41F_THEN(split_part(b, layer, source,
							 group, 1u, chunk_rows,
							 0, DS41F_FFN_SUM,
							 LAUNCH_SIDE));
		status = DS41F_THEN(ds41f_tp2_layer_select(b, next));
		status = DS41F_THEN(split_part(
			b, next,
			(DS41F_SOURCE_LAYERS >> next & 1u) != 0 ? next : source,
			(DS41F_INDEX_LAYERS >> next & 1u) != 0 ? next : group,
			0, chunk_rows, HALF_ATTENTION, 0,
			dead[1] ? LAUNCH_CHAIN : LAUNCH_RESUME));
		status = DS41F_THEN(ds41f_tp2_layer_select(b, layer));
		status = DS41F_THEN(split_part(
			b, layer, source, group, 1u, chunk_rows, 0,
			DS41F_FFN_POST, dead[1] ? LAUNCH_CHAIN : LAUNCH_JOIN));
		b->ahead_layer[b->card] = status == DS41F_OK ? next + 1u : 0u;
	}
	half_select(b, DS41F_CHUNK_HALVES, chunk_rows);
	return status;
}

MUST_CHECK static result_status
tp2_program_layer_steps_inner(struct ds41f_builder *build, uint32_t layer,
			      uint32_t source, uint32_t group, bool front)
{
	const bool divided = layer == DS41F_DECODER_FIRST_LAYER;
	if (build->slab_count == 1u) {
		const uint32_t steps =
			front	? DS41F_ENGRAM_STEPS | DS41F_FRONT_STEPS :
			divided ? DS41F_REST_STEPS :
				  DS41F_ENGRAM_STEPS | DS41F_FRONT_STEPS |
					  DS41F_REST_STEPS;
		return ds41f_layer_steps(build, layer, source, group, steps);
	}
	if (!front &&
	    pipelined(
		    build,
		    layer)) /* split_overlap's pipeline: the attention in split_layer's halves */
		return split_layer(build, layer, source, group);
	/* a tensor layer's sums are built over one wide launch's rows at most (split_overlap: a half's) */
	if (!front && build->tensor_layer != NULL &&
	    ds41f_piece_count_of(build) > 1u && !split_applies(build, layer))
		return STATUS_INVALID_ARGUMENT;
	/* chunk_core: the slabs' scans in place of their cores, then the core over the chunk. batch_core_once, a BATCH
     * program of several slabs: each slab's state writes (its window writes too), queries and scan as before, then
     * one core over the step's rows through the step's own block: a row's core reads its own queries, selection and
     * rotation, the window of positions up to its own (the 4,096-slot ring: later rows' slots are others) and
     * pool
     * entries below its bound (later rows complete only later groups), so it sees what it saw a slab at a time: the
     * same bits */
	const bool batch_core = build->batch_core_once != 0 &&
				build->made->program.kind == PROGRAM_BATCH &&
				build->slab_count > 1u &&
				ds41f_piece_count_of(build) == 1u &&
				build->tensor_layer != NULL;
	const bool chunk_core = chunk_core_applies(build) || batch_core;
	const uint32_t core = chunk_core ? DS41F_SCAN : DS41F_CORE;
	/* queries_piece: the queries with wq_b over each piece, not a span at a time; index_span: the slabs' scans
	 * behind
     * the slab loop, a span at a time (index_scans) */
	const uint32_t piece_queries = queries_piece_applies(build) ?
					       DS41F_QUERIES :
					       0u,
		       span_queries = piece_queries ^ DS41F_QUERIES;
	const bool spans = chunk_core && index_span_applies(build);
	const uint32_t slab_core = spans ? 0u : core;
	result_status status = DS41F_OK;
	if (front || !divided) {
		/* chunk_core: window writes over the chunk, pool writes per slab (layer 20's front: per slab too).
         * BATCH level 3 computes pools over the step, then retains compressor slots separately: mixed prompt rows
         * can wrap a session's slots, and must not overwrite a previous partner before a pool row reads it. */
		const bool window = chunk_core &&
				    !original_chunk_core_applies(build) &&
				    !divided &&
				    (!batch_core || build->batch_core_once >=
							    BATCH_CORE_WINDOWS);
		const bool pool_chunk =
			window && !ds41f_thin_tail_program(build) &&
			(!batch_core ||
			 build->batch_core_once >= BATCH_CORE_POOLS) &&
			build->tp_description != NULL &&
			build->tp_description->stages[build->card]
				->description.pool_chunk;
		status = ds41f_steps_over(
			build, layer, source, group,
			AHEAD_STEPS |
				(divided ? 0u :
					   DS41F_QUERY_NORM | QUERY_STEPS |
						   piece_queries),
			true);
		ds41f_chunk_select(build);
		status = window ? DS41F_THEN(ds41f_layer_steps(
					  build, layer, source, group,
					  DS41F_WINDOW_WRITES)) :
				  status;
		status = pool_chunk ? DS41F_THEN(ds41f_steps_over(
					      build, layer, source, group,
					      DS41F_POOL_CHUNK, true)) :
				      status;
		status = DS41F_THEN(ds41f_steps_over(
			build, layer, source, group,
			(window ? (pool_chunk ? 0u : DS41F_STATE_POOL) :
				  DS41F_STATE_WRITES) |
				(divided ? 0u : span_queries | slab_core),
			false));
		status = spans && !divided ?
				 DS41F_THEN(index_scans(build, layer, source,
							group, 0,
							build->slab_count)) :
				 status;
	} else {
		status = ds41f_steps_over(build, layer, source, group,
					  QUERY_STEPS | piece_queries, true);
		status = DS41F_THEN(
			ds41f_steps_over(build, layer, source, group,
					 span_queries | slab_core, false));
		status = spans ? DS41F_THEN(index_scans(build, layer, source,
							group, 0,
							build->slab_count)) :
				 status;
	}
	if (chunk_core && (!divided || !front) && status == DS41F_OK) {
		ds41f_chunk_select(build);
		status = ds41f_layer_steps(build, layer, source, group,
					   DS41F_CORE_ALONE);
	}
	/* exact_dense: its products a piece, its gathers a slab at a time (dense_gather_slabs); split_overlap: in
	 * halves */
	return front ? status :
	       split_applies(build, layer) ?
		       DS41F_THEN(split_layer(build, layer, source, group)) :
		       DS41F_THEN(ds41f_steps_over(
			       build, layer, source, group,
			       DS41F_OUTPUT_PRODUCTS | DS41F_SITE_POST, true));
}

MUST_CHECK result_status
ds41f_tp2_program_layer_steps(struct ds41f_builder *build, uint32_t layer,
			      uint32_t source, uint32_t group, bool front)
{
	const struct ds41f_tp2_layer_plan *const tensor = build->tensor_layer;
	if (tensor != NULL && tensor->exact_output_shards &&
	    !tensor->exact_dense)
		build->tensor_layer = NULL;
	const result_status status = tp2_program_layer_steps_inner(
		build, layer, source, group, front);
	build->tensor_layer = tensor;
	return status;
}

MUST_CHECK static result_status
tp4_program_owner_layer(struct ds41f_builder *build,
			const struct ds41f_tp2_program_description *description,
			struct ds41f_tp2_layer_plan *tensor,
			struct step_companion_pair *pairs, uint32_t *pair_count,
			uint32_t layer, uint32_t source, uint32_t group,
			bool front)
{
	const uint32_t owner = ds41f_tp4_attention_owner(layer);
	result_status status = ds41f_tp2_program_select(build, description,
							tensor, owner, layer);
	uint32_t first_launch = build->made->cards[owner].launch_count;
	status = DS41F_THEN(ds41f_tp2_program_layer_steps(build, layer, source,
							  group, front));
	status = DS41F_THEN(
		ds41f_tp2_program_segment(build, layer, first_launch));
	const bool divided = layer == DS41F_DECODER_FIRST_LAYER;
	status = DS41F_THEN(
		ds41f_tp2_program_copies(build, layer,
					 front	 ? DS41F_FRONT_COPIES :
					 divided ? DS41F_LATER_COPIES :
						   DS41F_EVERY_COPY));
	if (front) {
		if (status == DS41F_OK)
			build->made->program.front_segment_count =
				build->made->program.segment_count;
		return status;
	}
	/* Attention has completed its original post-mix. Peers need both residual arrays before their FFN fronts. */
	for (uint32_t rank = 0; rank < DS41F_TP4_RANKS && status == DS41F_OK;
	     rank++) {
		if (rank != owner)
			status = ds41f_tp2_program_streams(build, rank);
	}
	const uint32_t first_segment = build->made->program.segment_count;
	for (uint32_t rank = 0; rank < DS41F_TP4_RANKS && status == DS41F_OK;
	     rank++) {
		status = ds41f_tp2_program_select(build, description, tensor,
						  rank, layer);
		first_launch = build->made->cards[rank].launch_count;
		status = build->slab_count > 1u ?
				 DS41F_THEN(ds41f_wide_ffn_launches(build,
								    layer)) :
				 DS41F_THEN(ds41f_ffn_launches(build, layer,
							       &first_launch));
		status = DS41F_THEN(
			ds41f_tp2_program_segment(build, layer, first_launch));
		if (rank == 0)
			status = DS41F_THEN(ds41f_tp2_program_boundary(
				build, description, layer));
	}
	return DS41F_THEN(ds41f_tp2_program_pair_add(build, pairs, pair_count,
						     first_segment));
}

MUST_CHECK result_status ds41f_tp2_program_layer_pair(
	struct ds41f_builder *build,
	const struct ds41f_tp2_program_description *description,
	struct ds41f_tp2_layer_plan *tensor, struct step_companion_pair *pairs,
	uint32_t *pair_count, uint32_t layer, uint32_t source, uint32_t group,
	bool front)
{
	const uint32_t first_card =
		ds41f_tp2_program_layer_card(description, layer);
	if (description->stages[first_card]->description.attention_owner)
		return tp4_program_owner_layer(build, description, tensor,
					       pairs, pair_count, layer, source,
					       group, front);
	const uint32_t first_segment = build->made->program.segment_count;
	const uint32_t ranks =
		description->stages[first_card]->description.rank_count;
	const bool divided = layer == DS41F_DECODER_FIRST_LAYER;
	result_status status = DS41F_OK;
	for (uint32_t rank = 0; rank < ranks && status == DS41F_OK; rank++) {
		status = ds41f_tp2_program_select(build, description, tensor,
						  first_card + rank, layer);
		uint32_t first_launch =
			build->made->cards[build->card].launch_count;
		if (!front && (rank == 0 || build->draft_layers) &&
		    ds41f_tp2_program_exl3_all(description) &&
		    description->canonical->limits.draft_positions != 0 &&
		    layer >= DS41F_DRAFT_FIRST_TAP_LAYER) {
			/* Rank0 retains the original streams before the target changes them (draft_layers: every rank,
			 * its
             * own context). Wide taps span the chunk. */
			ds41f_chunk_select(build);
			status = DS41F_THEN(
				ds41f_draft_tap_launch(build, layer));
		}
		/* split_overlap: the FFN in split_layer's halves, inside the layer steps */
		const bool split = !front && build->slab_count > 1u &&
				   split_applies(build, layer);
		status = DS41F_THEN(ds41f_tp2_program_layer_steps(
			build, layer, source, group, front));
		if (!front && !split)
			status = build->slab_count > 1u ?
					 DS41F_THEN(ds41f_wide_ffn_launches(
						 build, layer)) :
					 DS41F_THEN(ds41f_ffn_launches(
						 build, layer, &first_launch));
		status = DS41F_THEN(
			ds41f_tp2_program_segment(build, layer, first_launch));
		if (rank == 0) {
			status = DS41F_THEN(ds41f_tp2_program_copies(
				build, layer,
				front	? DS41F_FRONT_COPIES :
				divided ? DS41F_LATER_COPIES :
					  DS41F_EVERY_COPY));
			if (!front)
				status = DS41F_THEN(ds41f_tp2_program_boundary(
					build, description, layer));
		}
	}
	status = DS41F_THEN(ds41f_tp2_program_pair_add(build, pairs, pair_count,
						       first_segment));
	if (front && status == DS41F_OK)
		build->made->program.front_segment_count =
			build->made->program.segment_count;
	return status;
}
