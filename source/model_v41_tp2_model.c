/* SPDX-License-Identifier: AGPL-3.0-or-later */
#include "model_v41_tp2_model.h"

#include <pthread.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#include "anemone/hooks.h"

#define TP2_MODEL_CARDS V41_DATA_TP2_PROGRAM_CARDS
#define TP2_MODEL_PRO V41_DATA_TP2_PRO_CARD
#define TP2_MODEL_SECOND_PAIR V41_DATA_TP2_PAIR_B_CARD
#define TP2_MODEL_MINIMUM_DRAFT 5u
#define TP2_MODEL_MAXIMUM_DRAFT 7u
#define TP2_FRONT_END_NAME "front_end.tables"
#define HOST_HUGE_PAGE_BYTES 2097152ull /* 2 MiB: the host arena is allocated as a whole number of them */

struct initialise_workspace {
	void *memory;
	uint64_t bytes;
	bool borrowed;
	bool bad_free;
};
static _Thread_local struct initialise_workspace *initialise_current;

struct stage_geometry {
	uint32_t first, count, rank, ranks;
};

static const struct stage_geometry
	geometries[V41_DATA_TP_LAYOUT_COLD_COUNT][TP2_MODEL_CARDS] = {
		{ { 0, 14, 0, 2 },
		  { 0, 14, 1, 2 },
		  { 14, 15, 0, 2 },
		  { 14, 15, 1, 2 },
		  { 29, 11, 0, 1 } },
		{ { 0, 29, 0, 4 },
		  { 0, 29, 1, 4 },
		  { 0, 29, 2, 4 },
		  { 0, 29, 3, 4 },
		  { 29, 11, 0, 1 } },
		{ { 0, 14, 0, 2 },
		  { 0, 14, 1, 2 },
		  { 14, 7, 0, 1 },
		  { 21, 8, 0, 1 },
		  { 29, 11, 0, 1 } },
		{ { 0, MODEL_LAYERS, 0, TP4_RANKS },
		  { 0, MODEL_LAYERS, 1, TP4_RANKS },
		  { 0, MODEL_LAYERS, TP2_RANKS, TP4_RANKS },
		  { 0, MODEL_LAYERS, TP4_RANKS - 1u, TP4_RANKS },
		  { 0, 0, 0, 0 } }
	};

static uint32_t
model_card_count(const struct v41_tp2_model_description *description)
{
	return description->tensor_layout == V41_DATA_TP_LAYOUT_QUAD_ALL ?
		       TP4_RANKS :
		       TP2_MODEL_CARDS;
}

static uint32_t
model_result_card(const struct v41_tp2_model_description *description)
{
	return description->tensor_layout == V41_DATA_TP_LAYOUT_QUAD_ALL ?
		       0u :
		       TP2_MODEL_PRO;
}

static uint32_t primary_card(const struct v41_tp2_model *model, uint32_t layer)
{
	if (layer >= MODEL_LAYERS)
		return model_result_card(&model->description);
	for (uint32_t card = 0; card < model_card_count(&model->description);
	     card++) {
		const struct stage_geometry *const geometry =
			&geometries[model->description.tensor_layout][card];
		if (geometry->rank == 0 && layer >= geometry->first &&
		    layer - geometry->first < geometry->count)
			return card;
	}
	return TP2_MODEL_CARDS;
}

static bool
description_fits(const struct v41_tp2_model_description *description)
{
	const struct model_limits *const limits = &description->limits;
	if (description->wire_format > ORDERED_PUSH_BF16 ||
	    description->tensor_layout >= V41_DATA_TP_LAYOUT_COLD_COUNT ||
	    (description->original_batch &&
	     (description->tensor_layout != V41_DATA_TP_LAYOUT_QUAD ||
	      !description->batch)) ||
	    (description->original_wide_experts &&
	     (description->tensor_layout != V41_DATA_TP_LAYOUT_QUAD ||
	      !description->exact_output_shards ||
	      description->original_code_wire ||
	      description->wire_format != 0 || description->batch ||
	      description->split_overlap != 0 ||
	      description->full_split_overlap != 0 ||
	      description->exl3_wide_entry || description->exl3_tc_moe ||
	      limits->row_capacity < MODEL_WIDE_LEAST_ROWS ||
	      limits->linear_profile != LINEAR_PROFILE_PRECISION ||
	      limits->linear_order != LINEAR_ORDER_REFERENCE)) ||
	    (description->attention_owner &&
	     !description->exact_output_shards) ||
	    (description->exact_output_shards &&
	     ((description->tensor_layout != V41_DATA_TP_LAYOUT_QUAD &&
	       description->tensor_layout != V41_DATA_TP_LAYOUT_QUAD_ALL) ||
	      description->wire_format != 0)) ||
	    (description->tensor_layout == V41_DATA_TP_LAYOUT_QUAD_ALL &&
	     (description->original_code_wire || description->attention_owner ||
	      description->wire_format != (description->exact_output_shards ?
						   0u :
						   ORDERED_PUSH_BF16) ||
	      (!description->exact_output_shards &&
	       description->exl3_product_order != BB_EXL3_REFERENCE_ORDER))) ||
	    (description->exl3_product_order != BB_EXL3_REFERENCE_ORDER &&
	     (description->tensor_layout != V41_DATA_TP_LAYOUT_QUAD_ALL ||
	      description->exl3_product_order != BB_EXL3_EIGHT_PART_ORDER)) ||
	    (description->head_shards &&
	     description->tensor_layout != V41_DATA_TP_LAYOUT_QUAD_ALL) ||
	    (description->exact_dense &&
	     (description->tensor_layout != V41_DATA_TP_LAYOUT_QUAD_ALL ||
	      !description->exact_output_shards)) ||
	    (description->draft_shards && !description->head_shards) ||
	    (description->draft_layers &&
	     (!description->draft_shards ||
	      !description->replicated_embedding ||
	      description->exact_output_shards ||
	      (description->batch && (!description->batch_draft_rows ||
				      !description->batch_wide_head)))) ||
	    (description->batch &&
	     ((!description->original_batch &&
	       description->tensor_layout != V41_DATA_TP_LAYOUT_QUAD_ALL) ||
	      description->limits.row_capacity < LINEAR_MAXIMUM_ROWS + 1u)) ||
	    ((description->batch_wide_head ||
	      description->batch_context_slabs) &&
	     !description->batch) ||
	    (description->batch_draft_rows &&
	     (!description->batch ||
	      (description->draft_shards && !description->draft_layers))) ||
	    (description->head_stream &&
	     description->tensor_layout != V41_DATA_TP_LAYOUT_QUAD_ALL) ||
	    description->wide_exchange > V41_DATA_TP_EXCHANGE_CE_RINGS ||
	    (description->replicated_embedding &&
	     description->tensor_layout != V41_DATA_TP_LAYOUT_QUAD_ALL) ||
	    limits->context_token_limit < MODEL_CONTEXT_WITHOUT_CANDIDATES ||
	    limits->state_width != STATE_WIDTH_16 ||
	    limits->cmp_order != CMP_ORDER_REFERENCE ||
	    limits->draft_positions < TP2_MODEL_MINIMUM_DRAFT ||
	    limits->draft_positions > TP2_MODEL_MAXIMUM_DRAFT ||
	    limits->row_capacity < limits->draft_positions ||
	    limits->row_capacity > (description->split_overlap != 0 ?
					    V41_DATA_TP_SPLIT_MOST_ROWS :
					    LINEAR_WIDE_MOST_ROWS) ||
	    (limits->row_capacity > LINEAR_MAXIMUM_ROWS &&
	     limits->row_capacity < MODEL_WIDE_LEAST_ROWS))
		return false;
	for (uint32_t card = 0; card < model_card_count(description); card++) {
		if (description->cards[card].arithmetic_class !=
			    (description->tensor_layout !=
						     V41_DATA_TP_LAYOUT_QUAD_ALL &&
					     card == TP2_MODEL_PRO ?
				     ARITHMETIC_CLASS_SM_120_NATIVE :
				     ARITHMETIC_CLASS_SM_80) ||
		    description->cards[card].memory_free_bytes >
			    description->cards[card].memory_total_bytes)
			return false;
	}
	return true;
}

MUST_CHECK static result_status stages_create(const struct pack_record *pack,
					      struct v41_tp2_model *model)
{
	for (uint32_t card = 0; card < model_card_count(&model->description);
	     card++) {
		const struct stage_geometry *const geometry =
			&geometries[model->description.tensor_layout][card];
		struct v41_tp2_stage_description description = {
			.limits = model->description.limits,
			.card = card,
			.rank = geometry->rank,
			.rank_count = geometry->ranks,
			.first_target_layer = geometry->first,
			.target_layer_count = geometry->count,
			.include_embedding =
				card == 0 ||
				(model->description.replicated_embedding &&
				 geometry->first == 0 &&
				 geometry->ranks == TP4_RANKS),
			.include_head_and_draft =
				card == model_result_card(&model->description),
			.exl3_all_targets = model->description.tensor_layout ==
					    V41_DATA_TP_LAYOUT_QUAD_ALL,
			.exl3_product_order =
				model->description.exl3_product_order,
			.exact_output_shards =
				model->description.exact_output_shards,
			.original_wide_experts =
				model->description.original_wide_experts &&
				card != TP2_MODEL_PRO,
			.original_batch = model->description.original_batch,
			.attention_owner = model->description.attention_owner &&
					   card != TP2_MODEL_PRO,
			.original_code_wire =
				model->description.original_code_wire,
			.head_shards = model->description.head_shards,
			.exact_dense = model->description.exact_dense,
			.draft_shards = model->description.draft_shards,
			.draft_layers = model->description.draft_layers,
			.batch = model->description.batch,
			.batch_wide_head = model->description.batch_wide_head,
			.batch_draft_rows = model->description.batch_draft_rows,
			.batch_context_slabs =
				model->description.batch_context_slabs,
			.batch_thin_bits = model->description.batch_thin_bits,
			.batch_grouped_rows =
				model->description.batch_grouped_rows,
			.batch_tc_paired = model->description.batch_tc_paired,
			.batch_draft_slots_head =
				model->description.batch_draft_slots_head,
			.batch_row_sets = model->description.batch_row_sets,
			.batch_core_once = model->description.batch_core_once,
			.front_branch = model->description.front_branch,
			.core_heads = model->description.core_heads,
			.decode_round_once =
				model->description.decode_round_once,
			.draft_bf16_rows = model->description.draft_bf16_rows,
			.batch_draft_wide = model->description.batch_draft_wide,
			.draft_bf16_operands =
				model->description.draft_bf16_operands,
			.ring_column_owners =
				model->description.ring_column_owners,
			.head_stream = model->description.head_stream,
			.exl3_fast_products =
				model->description.exl3_fast_products,
			.exl3_wide_entry = model->description.exl3_wide_entry,
			.exl3_tc_moe = model->description.exl3_tc_moe,
			.exl3_tc_product_blocks =
				model->description.exl3_tc_product_blocks,
			.exl3_expert_rotation =
				model->description.exl3_expert_rotation,
			.trunk_shape_slices =
				model->description.trunk_shape_slices,
			.head_mma = model->description.head_mma,
			.trunk_mma = model->description.trunk_mma,
			.trunk_mma_fused = model->description.trunk_mma_fused,
			.trunk_mma_thin_splits =
				model->description.trunk_mma_thin_splits,
			.router_mma = model->description.router_mma,
			.router_mma_wide = model->description.router_mma_wide,
			.gather_balance = model->description.gather_balance,
			.pool_chunk = model->description.pool_chunk,
			.pool_gather = model->description.pool_gather,
			.router_first_whole =
				model->description.router_first_whole,
			.tcp_two_blocks = model->description.tcp_two_blocks,
			.queries_piece = model->description.queries_piece,
			.index_span = model->description.index_span,
			.core_compact_halves =
				model->description.core_compact_halves,
			.dense_skinny_bf16 =
				model->description.dense_skinny_bf16,
			.dense_skinny_f32 = model->description.dense_skinny_f32,
			.dense_staged = model->description.dense_staged,
			.graph_parts = model->description.graph_parts,
			.ordered_wide = model->description.ordered_wide,
			.graph_branches = model->description.graph_branches,
			.piece_rings = model->description.piece_rings,
			.bf16_slab_rings = model->description.bf16_slab_rings,
			.wide_exchange = model->description.wide_exchange,
			.split_overlap = model->description.split_overlap,
			.split_least_rows = model->description.split_least_rows,
			.launches_untrimmed =
				model->description.launches_untrimmed,
			.full_split_overlap =
				model->description.full_split_overlap,
			.tail_rows = model->description.tail_rows,
			.tail_overlay = model->description.tail_overlay,
			.tail_sets = model->description.tail_sets,
			.thin_tail = model->description.thin_tail,
			.full_cone_behind = model->description.full_cone_behind,
			.dead_rows_trimmed =
				model->description.dead_rows_trimmed,
			.chunk_core = model->description.chunk_core,
			.wire_format = model->description.wire_format
		};
		if (card != model_result_card(&model->description) && !model->description.draft_layers)
			description.limits.draft_positions = 0;
		result_status status = v41_tp2_stage_plan_create(
			pack, &description, &model->stages[card]);
		if (status != STATUS_OK)
			return status;
		const struct card_description_record *const reported =
			&model->description.cards[card];
		const uint64_t kept = model->description.kept_free_bytes;
		const uint64_t reserved = reported->memory_reserved_bytes;
		if (kept > reported->memory_free_bytes ||
		    reserved > reported->memory_free_bytes - kept ||
		    model->stages[card]->model->cards[card].arena_byte_count >
			    reported->memory_free_bytes - kept - reserved)
			return STATUS_OUT_OF_MEMORY;
	}
	return STATUS_OK;
}

MUST_CHECK static result_status metadata_merge(struct v41_tp2_model *model)
{
	model->canonical = malloc(sizeof(*model->canonical));
	if (model->canonical == NULL)
		return STATUS_OUT_OF_MEMORY;
	struct model_plan_record *const canonical = model->canonical;
	*canonical = *model->stages[0]->model;
	canonical->limits = model->description.limits;
	canonical->card_count = model_card_count(&model->description);
	canonical->embedding_card = 0;
	canonical->result_card = model_result_card(&model->description);
	for (uint32_t card = 0; card < canonical->card_count; card++) {
		const struct model_plan_record *const stage =
			model->stages[card]->model;
		if (memcmp(canonical->pool_session_bytes,
			   stage->pool_session_bytes,
			   sizeof(canonical->pool_session_bytes)) != 0 ||
		    memcmp(canonical->key_session_bytes,
			   stage->key_session_bytes,
			   sizeof(canonical->key_session_bytes)) != 0)
			return STATUS_INVALID_ARGUMENT;
		canonical->cards[card] = stage->cards[card];
	}
	for (uint32_t layer = 0; layer < V41_DATA_WALK_LAYERS; layer++) {
		const uint32_t card = primary_card(model, layer);
		const struct model_plan_record *const stage =
			model->stages[card]->model;
		canonical->layer_card[layer] = card;
		(void)memcpy(canonical->layer_tensors[layer],
			     stage->layer_tensors[layer],
			     sizeof(canonical->layer_tensors[layer]));
		canonical->experts[layer] = stage->experts[layer];
		canonical->exl3_bits[layer] = stage->exl3_bits[layer];
		(void)memcpy(canonical->expert_tables[layer],
			     stage->expert_tables[layer],
			     sizeof(canonical->expert_tables[layer]));
		(void)memcpy(canonical->host_expert_offsets[layer],
			     stage->host_expert_offsets[layer],
			     sizeof(canonical->host_expert_offsets[layer]));
		canonical->host_expert_count[layer] =
			stage->host_expert_count[layer];
		canonical->expert_folds_admitted[layer] =
			stage->expert_folds_admitted[layer];
	}
	for (uint32_t tensor = 1; tensor < V41_DATA_GLOBALS; tensor++)
		canonical->globals[tensor] =
			model->stages[canonical->result_card]
				->model->globals[tensor];
	(void)memcpy(
		canonical->draft_globals,
		model->stages[canonical->result_card]->model->draft_globals,
		sizeof(canonical->draft_globals));
	for (uint32_t card = 0; card < canonical->card_count; card++) {
		model->views[card] = malloc(sizeof(*model->views[card]));
		if (model->views[card] == NULL)
			return STATUS_OUT_OF_MEMORY;
		struct model_plan_record *const view = model->views[card];
		*view = *model->stages[card]->model;
		view->limits = canonical->limits;
		view->card_count = canonical->card_count;
		view->embedding_card = canonical->embedding_card;
		view->result_card = canonical->result_card;
		(void)memcpy(view->cards, canonical->cards,
			     sizeof(view->cards));
		(void)memcpy(view->globals, canonical->globals,
			     sizeof(view->globals));
		if (model->stages[card]->description.include_embedding) /* replicated_embedding: the rank's own copy */
			view->globals[V41_DATA_EMBEDDING] =
				model->stages[card]
					->model->globals[V41_DATA_EMBEDDING];
		if (model->stages[card]->description.head_shards && card != canonical->result_card) {
			/* head_shards: the rank's own norm and head rows */
			view->globals[V41_DATA_NORM] =
				model->stages[card]
					->model->globals[V41_DATA_NORM];
			view->globals[V41_DATA_HEAD] =
				model->stages[card]
					->model->globals[V41_DATA_HEAD];
		}
		(void)memcpy(view->draft_globals, canonical->draft_globals,
			     sizeof(view->draft_globals));
		if (model->stages[card]->description.draft_shards && card != canonical->result_card) {
			/* draft_shards: the rank's own Markov tables */
			view->draft_globals[V41_DATA_DRAFT_MARKOV_EMBEDDING] =
				model->stages[card]->model->draft_globals
					[V41_DATA_DRAFT_MARKOV_EMBEDDING];
			view->draft_globals[V41_DATA_DRAFT_MARKOV_HEAD] =
				model->stages[card]->model->draft_globals
					[V41_DATA_DRAFT_MARKOV_HEAD];
		}
		if (model->stages[card]->description.draft_layers && card != canonical->result_card) {
			/* draft_layers: the rank's own main_proj, main_norm and final norm */
			static const uint32_t own[] = {
				V41_DATA_DRAFT_MAIN_PROJECTION,
				V41_DATA_DRAFT_MAIN_NORM, V41_DATA_DRAFT_NORM
			};
			for (uint32_t index = 0; index < sizeof(own) / sizeof(own[0]); index++)
				view->draft_globals[own[index]] = model->stages[card]->model->draft_globals[own[index]];
		}
		(void)memcpy(view->layer_card, canonical->layer_card,
			     sizeof(view->layer_card));
		(void)memcpy(view->pool_session_bytes,
			     canonical->pool_session_bytes,
			     sizeof(view->pool_session_bytes));
		(void)memcpy(view->key_session_bytes,
			     canonical->key_session_bytes,
			     sizeof(view->key_session_bytes));
	}
	return STATUS_OK;
}

static bool job_before(const struct tp_load_job *left,
		       const struct tp_load_job *right,
		       const struct pack_tensor *tensors)
{
	const uint32_t left_extent = tensors[left->tensor_index].extent_index;
	const uint32_t right_extent = tensors[right->tensor_index].extent_index;
	return left_extent != right_extent ?
		       left_extent < right_extent :
	       left->tensor_index != right->tensor_index ?
		       left->tensor_index < right->tensor_index :
		       left->card < right->card;
}

MUST_CHECK static result_status jobs_merge(const struct pack_record *pack,
					   struct v41_tp2_model *model)
{
	uint64_t count = 0;
	for (uint32_t card = 0; card < model->canonical->card_count; card++)
		count += model->stages[card]->job_count;
	if (count == 0)
		return STATUS_INVALID_ARGUMENT;
	if (count > UINT32_MAX || count > SIZE_MAX / sizeof(*model->jobs))
		return STATUS_OUT_OF_MEMORY;
	model->jobs = calloc((size_t)count, sizeof(*model->jobs));
	if (model->jobs == NULL)
		return STATUS_OUT_OF_MEMORY;
	const struct pack_tensor *const tensors = pack_tensors_of(pack);
	uint32_t cursor[TP2_MODEL_CARDS] = { 0 };
	for (uint32_t index = 0; index < count; index++) {
		uint32_t selected = TP2_MODEL_CARDS;
		for (uint32_t card = 0; card < model->canonical->card_count;
		     card++) {
			if (cursor[card] < model->stages[card]->job_count &&
			    (selected == TP2_MODEL_CARDS ||
			     job_before(&model->stages[card]->jobs[cursor[card]],
					&model->stages[selected]
						 ->jobs[cursor[selected]],
					tensors)))
				selected = card;
		}
		if (selected == TP2_MODEL_CARDS)
			return STATUS_INVALID_ARGUMENT;
		model->jobs[index] =
			model->stages[selected]->jobs[cursor[selected]++];
	}
	struct tp_load_description description = {
		.pack = pack,
		.jobs = model->jobs,
		.job_count = (uint32_t)count,
		.card_count = model->canonical->card_count,
		.maximum_extent_byte_count = TP_LOAD_MAXIMUM_EXTENT_BYTES
	};
	for (uint32_t card = 0; card < model->canonical->card_count; card++)
		description.card_arena_byte_count[card] =
			model->canonical->cards[card].arena_byte_count;
	const result_status status =
		tp_load_prepare(&description, &model->gpu_load);
	model->output_workspace_bytes =
		model->gpu_load.output_workspace_byte_count;
	for (uint32_t card = 0;
	     card < model->canonical->card_count && status == STATUS_OK;
	     card++) {
		const uint64_t bytes =
			model->canonical->cards[card].image_byte_count;
		model->output_workspace_bytes =
			bytes > model->output_workspace_bytes ?
				bytes :
				model->output_workspace_bytes;
	}
	model->output_workspace_bytes =
		model->output_workspace_bytes <
				V41_DATA_EMBEDDING_COPY_PIECE_BYTES ?
			V41_DATA_EMBEDDING_COPY_PIECE_BYTES :
			model->output_workspace_bytes;
	return status;
}

result_status
v41_tp2_model_create(const struct pack_record *pack,
		     const struct v41_tp2_model_description *description,
		     struct v41_tp2_model **model_out)
{
	if (pack == NULL || description == NULL || model_out == NULL ||
	    !description_fits(description))
		return STATUS_INVALID_ARGUMENT;
	struct v41_tp2_model *const model = calloc(1, sizeof(*model));
	if (model == NULL)
		return STATUS_OUT_OF_MEMORY;
	model->description = *description;
	model->recipe_identity =
		(description->exact_output_shards ?
			 V41_DATA_TP4_EXACT_RECIPE :
			 V41_DATA_TP2_PROGRAM_RECIPE) ^
		TP2_DEFAULT_SLICES ^
		(description->wire_format != 0 &&
				 (description->tensor_layout ==
					  V41_DATA_TP_LAYOUT_QUAD ||
				  description->tensor_layout ==
					  V41_DATA_TP_LAYOUT_QUAD_ALL) ?
			 V41_DATA_TP4_RING_IDENTITY :
			 0) ^
		((uint64_t)description->wire_format
		 << V41_DATA_TP2_WIRE_IDENTITY_SHIFT) ^
		((uint64_t)description->tensor_layout
		 << V41_DATA_TP_LAYOUT_IDENTITY_SHIFT) ^
		(description->limits.row_capacity > LINEAR_MAXIMUM_ROWS ?
			 V41_DATA_TP_WIDE_IDENTITY :
			 0) ^
		(description->tensor_layout == V41_DATA_TP_LAYOUT_QUAD ?
			 V41_DATA_TP4_ALIGNED_IDENTITY :
			 0) ^
		(description->attention_owner ?
			 V41_DATA_TP4_ATTENTION_OWNER_IDENTITY :
			 0);
	if (description->tensor_layout == V41_DATA_TP_LAYOUT_QUAD_ALL)
		model->recipe_identity ^=
			V41_DATA_TP4_EXL3_IDENTITY ^
			((uint64_t)description->exl3_product_order
			 << V41_DATA_TP4_EXL3_ORDER_SHIFT);
	result_status status = stages_create(pack, model);
	if (status == STATUS_OK)
		status = metadata_merge(model);
	if (status == STATUS_OK)
		status = jobs_merge(pack, model);
	if (status == STATUS_OK) {
		const uint32_t count = pack_header_of(pack)->extent_count;
		model->extent_offsets =
			calloc(count, sizeof(*model->extent_offsets));
		status = model->extent_offsets == NULL ?
				 STATUS_OUT_OF_MEMORY :
				 tp_host_load_prepare(pack,
						      model->extent_offsets,
						      count, &model->host_load);
	}
	if (status == STATUS_OK)
		*model_out = model;
	else
		v41_tp2_model_destroy(model);
	return status;
}

void v41_tp2_model_destroy(struct v41_tp2_model *model)
{
	if (model == NULL)
		return;
	free(model->host_arena);
	free(model->extent_offsets);
	free(model->jobs);
	free(model->canonical);
	for (uint32_t card = 0; card < TP2_MODEL_CARDS; card++) {
		free(model->views[card]);
		v41_tp2_stage_plan_destroy(model->stages[card]);
	}
	free(model);
}

result_status v41_tp2_model_program_description(
	const struct v41_tp2_model *model,
	struct v41_tp2_program_description *description_out)
{
	if (model == NULL || description_out == NULL ||
	    model->canonical == NULL)
		return STATUS_INVALID_ARGUMENT;
	struct v41_tp2_program_description description = {
		.canonical = model->canonical, .slice_count = TP2_DEFAULT_SLICES
	};
	for (uint32_t card = 0; card < model->canonical->card_count; card++) {
		description.views[card] = model->views[card];
		description.stages[card] = model->stages[card];
	}
	*description_out = description;
	return STATUS_OK;
}

MUST_CHECK static result_status initialise_allocate(uint64_t bytes,
						    void **memory_out)
{
	if (initialise_current == NULL || memory_out == NULL || bytes == 0)
		return STATUS_INVALID_ARGUMENT;
	if (initialise_current->borrowed)
		return STATUS_BUSY_VALUE;
	if (bytes > initialise_current->bytes)
		return STATUS_OUT_OF_MEMORY;
	initialise_current->borrowed = true;
	*memory_out = initialise_current->memory;
	return STATUS_OK;
}

static void initialise_free(void *memory)
{
	if (initialise_current == NULL)
		return;
	initialise_current->bad_free = initialise_current->bad_free ||
				       memory != initialise_current->memory;
	if (memory == initialise_current->memory)
		initialise_current->borrowed = false;
}

MUST_CHECK static result_status
canonical_initialise(struct v41_tp2_model *model,
		     const struct v41_tp2_model_load_request *request,
		     void *output)
{
	if (initialise_current != NULL)
		return STATUS_BUSY_VALUE;
	struct initialise_workspace workspace = {
		.memory = output, .bytes = model->output_workspace_bytes
	};
	struct backend_operations backend = *request->backend;
	backend.staging_allocate = initialise_allocate;
	backend.staging_free = initialise_free;
	initialise_current = &workspace;
	result_status status =
		v41_arena_initialise(model->canonical, &backend, request->cards,
				     request->wait_limit_nanoseconds);
	initialise_current = NULL;
	if (status == STATUS_OK && (workspace.borrowed || workspace.bad_free))
		status = STATUS_DEVICE_ERROR;
	return status;
}

static bool request_fits(const struct v41_tp2_model *model,
			 const struct v41_tp2_model_load_request *request)
{
	if (model == NULL || request == NULL || request->backend == NULL ||
	    request->cards == NULL || request->storage == NULL ||
	    request->reader == NULL ||
	    request->storage->reader_describe == NULL ||
	    request->storage->reader_read == NULL ||
	    request->wait_limit_nanoseconds == 0)
		return false;
	const struct backend_operations *const backend = request->backend;
	if (backend->staging_allocate == NULL ||
	    backend->staging_free == NULL || backend->staging_retire == NULL ||
	    backend->upload_begin == NULL || backend->upload_wait == NULL ||
	    backend->download_begin == NULL || backend->digest == NULL ||
	    backend->kernel_prepare == NULL || backend->launch_run == NULL ||
	    backend->launch_wait == NULL || backend->peer_connect == NULL)
		return false;
	for (uint32_t card = 0; card < model->canonical->card_count; card++) {
		if (request->cards[card] == NULL)
			return false;
		const struct v41_tp2_stage_description *const stage =
			&model->stages[card]->description;
		const uint32_t group_mask = ((1u << stage->rank_count) - 1u)
					    << (card - stage->rank);
		const uint32_t peers = group_mask & ~(1u << card);
		if ((request->qualified_writer_masks[card] & peers) != peers)
			return false;
	}
	return true;
}

MUST_CHECK static result_status
peer_tables_attach(const struct v41_tp2_model *model,
		   const struct v41_tp2_model_load_request *request,
		   void *workspace)
{
	struct ordered_collective_peer *const peers = workspace;
	for (uint32_t card = 0; card < model->canonical->card_count; card++) {
		const struct v41_tp2_stage_plan *const local =
			model->stages[card];
		const uint32_t ranks = local->description.rank_count;
		if (ranks == 1u)
			continue;
		const uint32_t pair = card - local->description.rank;
		for (uint32_t index = 0; index < local->channel_count;
		     index++) {
			const struct ordered_push_arguments *const arguments =
				&local->channels[index].collective.arguments;
			for (uint32_t rank = 0; rank < ranks; rank++) {
				const struct v41_tp2_stage_plan *const remote =
					model->stages[pair + rank];
				if (index >= remote->channel_count ||
				    local->channels[index].layer !=
					    remote->channels[index].layer ||
				    local->channels[index].site !=
					    remote->channels[index].site)
					return STATUS_INVALID_ARGUMENT;
				peers[rank] = (struct ordered_collective_peer){
					.region_offset =
						remote->channels[index]
							.collective.arguments
							.region_offset,
					.route = ORDERED_COLLECTIVE_DIRECT_ROUTE
				};
			}
			uint64_t ticket = 0;
			result_status status = request->backend->upload_begin(
				request->cards[card],
				arguments->peer_table_offset, workspace,
				ranks * sizeof(*peers), &ticket);
			if (status == STATUS_OK)
				status = request->backend->upload_wait(
					request->cards[card], ticket,
					request->wait_limit_nanoseconds);
			for (uint32_t rank = 0;
			     rank < ranks && status == STATUS_OK; rank++)
				status = request->backend->peer_connect(
					request->cards[card],
					request->cards[pair + rank], 0,
					arguments->peer_table_offset +
						rank * sizeof(*peers) +
						offsetof(
							struct ordered_collective_peer,
							arena_address));
			if (status == STATUS_OK)
				status = request->backend->launch_wait(
					request->cards[card],
					request->wait_limit_nanoseconds);
			if (status != STATUS_OK)
				return status;
		}
	}
	return STATUS_OK;
}

/* head_shards (sampler_shard_records.h): each rank's table of the four ranks' exchange areas; the result card's table
 * of the four ranks' settings copies (its settings push writes them). */
MUST_CHECK static result_status
sampler_peers_attach(const struct v41_tp2_model *model,
		     const struct v41_tp2_model_load_request *request)
{
	const uint32_t result = model->canonical->result_card;
	result_status status = STATUS_OK;
	for (uint32_t card = 0;
	     card < model->canonical->card_count && status == STATUS_OK;
	     card++) {
		const struct v41_tp2_stage_plan *const local =
			model->stages[card];
		for (uint32_t rank = 0; rank < TP4_RANKS && status == STATUS_OK;
		     rank++)
			status = request->backend->peer_connect(request->cards[card], request->cards[rank],
								model->stages[rank]->sampler_exchange.byte_offset,
								local->sampler_peers.byte_offset + rank * sizeof(uint64_t));
		for (uint32_t rank = 0;
		     rank < TP4_RANKS && status == STATUS_OK && card == result;
		     rank++)
			status = request->backend->peer_connect(request->cards[card], request->cards[rank],
								model->stages[rank]->sampler_settings.byte_offset,
								local->sampler_settings_peers.byte_offset + rank * sizeof(uint64_t));
		for (uint32_t rank = 0;
		     rank < TP4_RANKS && status == STATUS_OK &&
		     card == result && local->description.draft_shards;
		     rank++) {
			if (rank != result) /* draft_shards: the result card's table of the other ranks' share areas */
				status = request->backend->peer_connect(request->cards[card], request->cards[rank],
									model->stages[rank]->draft_area.byte_offset,
									local->draft_peers.byte_offset + rank * sizeof(uint64_t));
		}
		if (status == STATUS_OK)
			status = request->backend->launch_wait(request->cards[card], request->wait_limit_nanoseconds);
	}
	return status;
}

/* The host arena's load, on a thread of its own beside the cards' or inline. */
struct host_load_thread {
	struct tp_host_load_request request;
	struct tp_host_load_result result;
	result_status status;
	pthread_t thread;
};

static void *host_load_run(void *context)
{
	struct host_load_thread *const host = context;
	TIMER_BEGIN(tp_model_host_load);
	host->status = tp_host_load_run(&host->request, &host->result);
	TIMER_END(tp_model_host_load);
	return NULL;
}

result_status
v41_tp2_model_load(struct v41_tp2_model *model,
		   const struct v41_tp2_model_load_request *request)
{
	if (initialise_current != NULL)
		return STATUS_BUSY_VALUE;
	if (!request_fits(model, request) || model->load_attempted)
		return STATUS_INVALID_ARGUMENT;
	model->load_attempted = true;
	void *source = NULL, *output = NULL, *host_arena = NULL;
	result_status status = request->backend->staging_allocate(
		model->gpu_load.source_workspace_byte_count, &source);
	if (status == STATUS_OK)
		status = request->backend->staging_allocate(
			model->output_workspace_bytes, &output);
	/* the host arena (the n-gram tables a step gathers rows of at random) in 2 MiB pages, as loader.c maps its own:
	 * small pages cost a TLB walk a row */
	const size_t host_mapped =
		(size_t)((model->host_load.host_byte_count + HOST_HUGE_PAGE_BYTES - 1u) / HOST_HUGE_PAGE_BYTES *
			 HOST_HUGE_PAGE_BYTES);
	if (status == STATUS_OK && model->host_load.host_byte_count != 0 &&
	    posix_memalign(&host_arena, HOST_HUGE_PAGE_BYTES, host_mapped) != 0)
		status = STATUS_OUT_OF_MEMORY;
	if (host_arena != NULL && madvise(host_arena, host_mapped, MADV_HUGEPAGE) != 0)
		LOG_WARNING_ENTRY(
			"no 2 MiB pages for the TP host arena: the n-gram rows read through small pages");
	/* one choice of copy for both loads (tp_load.h); the serial loads read the copies in order */
	const bool choosing =
		request->load_threads != 0 && status == STATUS_OK &&
		tp_load_copies_init(&model->copies,
				    request->load_reads_per_copy) == STATUS_OK;
	struct tp_load_copies *const copies = choosing ? &model->copies : NULL;
	if (copies != NULL)
		copies->split = request->load_copy_split;
	struct host_load_thread host = {
		.request = { .plan = &model->host_load,
			     .storage = request->storage,
			     .reader = request->reader,
			     .host_arena = host_arena,
			     .host_arena_byte_count = model->host_load.host_byte_count,
			     .worker_count = request->load_host_threads != 0 ? request->load_host_threads :
										 request->load_threads,
			     .copies = copies },
		.status = status
	};
	/* the host arena beside the cards' loads (load_threads != 0), else first, inline (also when no thread starts) */
	const bool host_beside =
		status == STATUS_OK && request->load_threads != 0 &&
		pthread_create(&host.thread, NULL, host_load_run, &host) == 0;
	if (status == STATUS_OK && !host_beside)
		(void)host_load_run(&host);
	status = host_beside ? STATUS_OK : host.status;
	struct tp_load_result gpu_result = { 0 };
	const struct tp_load_request gpu_request = {
		.plan = &model->gpu_load,
		.storage = request->storage,
		.reader = request->reader,
		.backend = request->backend,
		.cards = request->cards,
		.source_workspace = source,
		.output_workspace = output,
		.source_workspace_byte_count =
			model->gpu_load.source_workspace_byte_count,
		.output_workspace_byte_count = model->output_workspace_bytes,
		.wait_limit_nanoseconds = request->wait_limit_nanoseconds,
		.worker_count = request->load_threads,
		.copies = copies
	};
	TIMER_BEGIN(tp_model_gpu_load);
	if (status == STATUS_OK)
		status = tp_load_run(&gpu_request, &gpu_result);
	TIMER_END(tp_model_gpu_load);
	if (host_beside) {
		(void)pthread_join(host.thread, NULL);
		status = host.status != STATUS_OK ?
				 host.status :
				 status; /* the serial order's first failure */
	}
	tp_load_copies_destroy(copies); /* its counters stay readable */
	TIMER_BEGIN(tp_model_initialise);
	TIMER_BEGIN(tp_model_canonical_initialise);
	if (status == STATUS_OK)
		status = canonical_initialise(model, request, output);
	TIMER_END(tp_model_canonical_initialise);
	TIMER_BEGIN(tp_model_peer_tables);
	if (status == STATUS_OK)
		status = peer_tables_attach(model, request, output);
	TIMER_END(tp_model_peer_tables);
	TIMER_BEGIN(tp_model_sampler_peers);
	if (status == STATUS_OK && model->description.head_shards)
		status = sampler_peers_attach(model, request);
	TIMER_END(tp_model_sampler_peers);
	TIMER_END(tp_model_initialise);
	/* Every attempted card is drained even when a different card failed. Failed waits are not retirement. */
	for (uint32_t card = 0; card < model->canonical->card_count; card++) {
		const result_status waited = request->backend->launch_wait(
			request->cards[card], request->wait_limit_nanoseconds);
		status = status == STATUS_OK ? waited : status;
	}
	request->backend->staging_retire(output);
	request->backend->staging_retire(source);
	if (status == STATUS_OK) {
		model->gpu_result = gpu_result;
		model->host_result = host.result;
		model->host_arena = host_arena;
		model->host_offsets = model->extent_offsets;
		model->host_bytes = model->host_load.host_byte_count;
		model->ready = true;
	} else {
		free(host_arena);
	}
	return status;
}

result_status v41_tp2_model_front_end_match(const struct v41_tp2_model *model,
					    const void *external_tables,
					    uint64_t external_bytes)
{
	if (model == NULL || !model->ready || external_tables == NULL ||
	    external_bytes == 0 || external_bytes > SIZE_MAX ||
	    model->host_offsets == NULL || model->host_bytes > SIZE_MAX)
		return STATUS_INVALID_ARGUMENT;
	const struct pack_record *const pack = model->host_load.pack;
	const uint32_t found = pack_find_tensor(
		pack, TP2_FRONT_END_NAME, sizeof(TP2_FRONT_END_NAME) - 1u);
	if (found == PACK_NOT_FOUND)
		return STATUS_UNSUPPORTED;
	const struct pack_tensor *const tensor = &pack_tensors_of(pack)[found];
	if (tensor->extent_index >= model->host_load.extent_count ||
	    tensor->element_type != PACK_ELEMENT_UINT8 ||
	    tensor->dimension_count != 1 ||
	    tensor->dimensions[0] != tensor->byte_count ||
	    tensor->weight_format != WEIGHT_FORMAT_NONE ||
	    tensor->bits_per_weight != 0)
		return STATUS_CORRUPT_PACK;
	const struct pack_extent *const extent =
		&pack_extents_of(pack)[tensor->extent_index];
	const struct pack_unit_record_2 *const unit =
		&pack_units_of(pack)[extent->unit_index];
	const uint64_t offset = model->host_offsets[tensor->extent_index];
	if (unit->unit_kind != PLACEMENT_UNIT_HOST_ONLY ||
	    unit->layer_index != PACK_NO_LAYER || unit->extent_count != 1 ||
	    model->host_arena == NULL || offset > model->host_bytes ||
	    extent->byte_count > model->host_bytes - offset ||
	    tensor->byte_offset_in_extent > extent->byte_count ||
	    tensor->byte_count >
		    extent->byte_count - tensor->byte_offset_in_extent)
		return STATUS_CORRUPT_PACK;
	if (tensor->byte_count != external_bytes)
		return STATUS_INVALID_ARGUMENT;
	const unsigned char *const packed =
		(const unsigned char *)model->host_arena + offset +
		tensor->byte_offset_in_extent;
	return memcmp(external_tables, packed, external_bytes) == 0 ?
		       STATUS_OK :
		       STATUS_INVALID_ARGUMENT;
}
