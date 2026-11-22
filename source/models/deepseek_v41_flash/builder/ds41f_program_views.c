/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * ds41f_program_views.c: plan validity, session ranges and checkpoints; row, slab, span, chunk, BATCH and draft views
 * (split from model_v41_program.c by whole items).
 */
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

#include "ds41f_builder.h"
#include "ds41f_program_internal.h"

_Static_assert(EP_FILTER_HIDDEN_WIDTH == DS41F_EXPERT_INNER,
	       "EP2 uses full routed width");
/* half views count from 1 behind the slabs': the second behind the first's block and rows */
#define SECOND_HALF_VIEW 2u
_Static_assert(
	DS41F_COMPRESSOR_SESSION_BYTES == DS41F_ATTENTION_SLOTS_SESSION_BYTES,
	"model_v41.h sizes the compressor's slots as attention_state_records.h lays them out");
_Static_assert(DS41F_ARRAY_DRAFT_WINDOW_2 + 1u - DS41F_ARRAY_DRAFT_WINDOW_0 ==
		       DS41F_DRAFT_STAGES,
	       "the context windows follow the draft stages in order");
_Static_assert(DS41F_SPAN_ROWS % LINEAR_MAXIMUM_ROWS == 0 &&
		       LINEAR_WIDE_MOST_ROWS % DS41F_SPAN_ROWS == 0,
	       "a span is whole slabs, a piece whole spans");

bool ds41f_card_has_exl3(const struct ds41f_plan *plan, uint32_t card)
{
	for (uint32_t layer = 0; layer < DS41F_LAYERS; layer++) {
		if ((plan->cards[card].layers >> layer & 1u) != 0 &&
		    plan->exl3_bits[layer] != 0)
			return true;
	}
	return false;
}

/* What a session keeps on the cards (model_seam.h): of each walk layer, on every card that holds them, its window ring,
 * its compressor slots, its pool's entries and its key tiles (a mirror's too), identity SESSION_KINDS *
 * layer + kind;
 * a draft stage has its ring alone (160, 164, 168), so a plan without the draft keeps its identities. */
uint32_t ds41f_session_ranges_make(const struct ds41f_plan *plan,
				   struct session_range *ranges)
{
	uint32_t count = 0;
	for (uint32_t layer = 0; layer < DS41F_WALK_LAYERS; layer++) {
		const uint32_t ratio = (DS41F_POOLED_LAYERS >> layer & 1u) !=
						       0 ?
					       DS41F_POOLED_RATIO :
					       DS41F_DIRECT_RATIO;
		const uint64_t pool = plan->pool_session_bytes[layer],
			       keys = plan->key_session_bytes[layer];
		for (uint32_t kind = 0; kind < DS41F_SESSION_KINDS; kind++) {
			for (uint32_t card = 0; card < plan->card_count;
			     card++) {
				const struct ds41f_card *const planned =
					&plan->cards[card];
				const struct session_range
					kinds[DS41F_SESSION_KINDS] = {
						{ planned->ring_offset[layer],
						  DS41F_RING_SESSION_BYTES,
						  DS41F_RING_SLOTS,
						  DS41F_STATE_ROW_BYTES, 1u,
						  card, 0 },
						{ planned->compressor_offset
							  [layer],
						  DS41F_COMPRESSOR_SESSION_BYTES,
						  1u,
						  DS41F_COMPRESSOR_SESSION_BYTES,
						  0, card, 0 },
						{ planned->pool_offset[layer],
						  pool,
						  pool / DS41F_STATE_ROW_BYTES,
						  DS41F_STATE_ROW_BYTES, ratio,
						  card, 0 },
						{ planned->key_tiles_offset
							  [layer],
						  keys, keys / DS41F_TILE_BYTES,
						  DS41F_TILE_BYTES,
						  ratio * DS41F_KEYS_PER_TILE,
						  card, 0 }
					};
				if (kinds[kind].offset != PROGRAM_NOWHERE) {
					ranges[count] = kinds[kind];
					ranges[count++].identity =
						layer * DS41F_SESSION_KINDS +
						kind;
				}
			}
		}
	}
	return count;
}

void ds41f_checkpoints_make(const struct ds41f_plan *plan,
			    struct window_checkpoint *ranges)
{
	const uint32_t count = plan->limits.draft_positions != 0 ?
				       DS41F_WALK_LAYERS :
				       DS41F_LAYERS;
	for (uint32_t layer = 0; layer < count; layer++) {
		const uint32_t card = plan->layer_card[layer];
		ranges[layer] = (struct window_checkpoint){
			plan->cards[card].ring_offset[layer],
			plan->cards[card].checkpoint_offset[layer],
			DS41F_RING_SESSION_BYTES,
			DS41F_RING_SLOTS,
			DS41F_STATE_ROW_BYTES,
			DS41F_WINDOW - 1u,
			card
		};
	}
}

/* The plan puts the embedding, the head and every layer on one of its cards, and gives each card the layers and the
 * n-gram tables it puts there: so every card the walk names is the plan's, and its part of the launches holds them. */
bool ds41f_plan_is_valid(const struct ds41f_plan *plan)
{
	uint64_t layers[BACKEND_MAXIMUM_CARDS] = { 0 };
	uint32_t tables[BACKEND_MAXIMUM_CARDS] = { 0 };
	bool is_valid = plan->embedding_card < plan->card_count &&
			plan->result_card < plan->card_count;
	for (uint32_t layer = 0; layer < DS41F_LAYERS && is_valid; layer++) {
		const uint32_t card = plan->layer_card[layer];
		const uint64_t tables_before =
			DS41F_ENGRAM_LAYERS &
			((1ull << layer) - 1u); /* its table's number */
		is_valid = card < plan->card_count;
		if (is_valid) {
			layers[card] |= 1ull << layer;
			tables[card] |=
				(uint32_t)(DS41F_ENGRAM_LAYERS >> layer & 1u)
				<< __builtin_popcountll(tables_before);
		}
	}
	for (uint32_t card = 0; card < plan->card_count && is_valid; card++) {
		is_valid = plan->cards[card].layers == layers[card] &&
			   plan->cards[card].engram_tables == tables[card];
	}
	if (is_valid && plan->limits.draft_positions != 0) {
		const struct ds41f_card *const result =
			&plan->cards[plan->result_card];
		is_valid = (result->layers & DS41F_DRAFT_TAP_LAYERS) ==
				   DS41F_DRAFT_TAP_LAYERS &&
			   (result->holders >> DS41F_HOLDER_DRAFT & 1u) != 0;
		for (uint32_t layer = DS41F_DRAFT_FIRST_LAYER;
		     layer < DS41F_WALK_LAYERS; layer++) {
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
 * multiple of DS41F_SPAN_ROWS), a slab's from its start (model_v41.h); a view's launches read its block and
 * rows in place of the step area's, which the plan lays out a slab of 8 rows apart. */
void ds41f_rows_select(struct ds41f_builder *b, uint32_t view,
		       uint32_t first_row, uint32_t rows)
{
	const struct ds41f_card *const planned = &b->plan->cards[b->card];
	b->view = view;
	b->first_row = first_row;
	b->rows = rows;
	const uint32_t capacity = b->plan->limits.row_capacity;
	for (uint32_t array = 0; array < DS41F_ARRAY_COUNT; array++) {
		const uint64_t offset = planned->array_offset[array];
		const enum ds41f_rows kind = ds41f_array_rows(
			b->plan->chunk_attention_arrays,
			(enum ds41f_array)array,
			(enum ds41f_rows)ds41f_array_bytes[array].rows);
		const uint64_t into =
			kind == DS41F_ROWS_CHUNK ? first_row :
			kind == DS41F_ROWS_SPAN	 ? first_row % DS41F_SPAN_ROWS :
						   0;
		const uint64_t array_rows =
			kind == DS41F_ROWS_DRAFT ?
				b->plan->limits.draft_positions :
				rows;
		/* an array ds41f_array_rows narrows holds no row past its span's or slab's: a view beyond them is no
		 * range, so that a launch which would take one is refused at its prepare */
		const bool beyond = kind != ds41f_array_bytes[array].rows &&
				    into + array_rows >
					    (kind == DS41F_ROWS_SPAN ?
						     ds41f_span_rows(capacity) :
						     ds41f_slab_rows(capacity));
		b->arrays[array] = (struct linear_range){
			offset == PROGRAM_NOWHERE || beyond ?
				PROGRAM_NOWHERE :
				offset + into * ds41f_array_bytes[array]
							 .row_bytes,
			ds41f_array_bytes[array].fixed_bytes +
				array_rows * ds41f_array_bytes[array].row_bytes
		};
	}
	if (view != 0) {
		/* a half view lies behind the slabs' (VIEW_BYTES apart), the second behind the first's block and
		 * rows */
		const uint32_t half =
			b->half_view_first != 0 && view >= b->half_view_first ?
				view - b->half_view_first + 1u :
				0;
		const uint32_t slabs_before =
			half != 0 ? b->half_view_first - 1u : view - 1u;
		b->arrays[DS41F_ARRAY_STEP] = (struct linear_range){
			planned->array_offset[DS41F_ARRAY_VIEWS] +
				(uint64_t)slabs_before * DS41F_VIEW_BYTES +
				(half == SECOND_HALF_VIEW ?
					 STEP_BLOCK_BYTES_VALUE +
						 (uint64_t)b->half_rows *
							 STEP_ROW_BYTES_VALUE :
					 0u),
			STEP_BLOCK_BYTES_VALUE +
				(uint64_t)rows * STEP_ROW_BYTES_VALUE
		};
	}
	if (b->reads_draft_step) {
		b->arrays[DS41F_ARRAY_STEP] = b->arrays[DS41F_ARRAY_DRAFT_STEP];
	}
}

/* The rows the next launches take of a chunk's array whose product packs them at row_bytes, not at the array's row
 * (model_v41.h). */
struct linear_range ds41f_packed_rows(const struct ds41f_builder *b,
				      enum ds41f_array array,
				      uint64_t row_bytes)
{
	const uint64_t offset = b->plan->cards[b->card].array_offset[array];
	return (struct linear_range){ offset == PROGRAM_NOWHERE ?
					      offset :
					      offset + b->first_row * row_bytes,
				      b->rows * row_bytes };
}

/* A wide program's slab s: its rows (8, the last fewer) through its view; the whole chunk through the step area. With
 * split_overlap's half selected, slab s of the half (its slab views are the chunk's). */
void ds41f_slab_select(struct ds41f_builder *b, uint32_t slab)
{
	const uint32_t first_row = slab * LINEAR_MAXIMUM_ROWS,
		       left = b->chunk_rows - first_row;
	ds41f_rows_select(
		b, (b->chunk_first + first_row) / LINEAR_MAXIMUM_ROWS + 1u,
		b->chunk_first + first_row,
		left < LINEAR_MAXIMUM_ROWS ? left : LINEAR_MAXIMUM_ROWS);
}

/* A wide program's span s: its rows (DS41F_SPAN_ROWS, the last fewer) through the step area, which no launch of
 * a span reads. With split_overlap's half selected, span s of the half. */
void ds41f_span_select(struct ds41f_builder *b, uint32_t span)
{
	const uint32_t first_row = span * DS41F_SPAN_ROWS,
		       left = b->chunk_rows - first_row;
	ds41f_rows_select(b, 0, b->chunk_first + first_row,
			  left < DS41F_SPAN_ROWS ? left : DS41F_SPAN_ROWS);
}

void ds41f_chunk_select(struct ds41f_builder *b)
{
	ds41f_rows_select(b, b->chunk_view, b->chunk_first, b->chunk_rows);
}

/* A BATCH program's slab for its heads: slab_select, or for a program of one slab (8 rows or fewer: a thin walk,
 * no
 * views, step_validation.c) its rows through the step's own block (slab_select would read view 1, never
 * written). */
void ds41f_batch_slab_select(struct ds41f_builder *b, uint32_t slab)
{
	if (b->slab_count > 1u)
		ds41f_slab_select(b, slab);
	else
		ds41f_chunk_select(b);
}

/* batch_draft_rows' walk: slab s of the batched draft (its 8 rows, its slab block as the step area, its collapse
 * rows), or with s the slab count all its rows (the step area slab 0's block: no launch over them reads it). */
void ds41f_draft_slab_select(struct ds41f_builder *b, uint32_t slab)
{
	const bool all = slab >= b->draft_slab_count;
	ds41f_rows_select(b, 0, all ? 0u : slab * LINEAR_MAXIMUM_ROWS,
			  all ? b->draft_rows : LINEAR_MAXIMUM_ROWS);
	const uint64_t first = all ? 0u : slab;
	const uint64_t slab_collapse =
		(uint64_t)LINEAR_MAXIMUM_ROWS * DS41F_HIDDEN_ROW_BYTES;
	const bool whole =
		all &&
		b->draft_wide_step; /* the block of all rows, ds41f_dspark_records.h's WIDE_STEP */
	const struct linear_range step = {
		b->draft_slab_steps + (whole ? b->draft_slab_count : first) *
					      DS41F_DRAFT_SLAB_STEP_BYTES,
		whole ? STEP_BLOCK_BYTES_VALUE +
				(uint64_t)b->draft_rows * STEP_ROW_BYTES_VALUE :
			DS41F_DRAFT_SLAB_STEP_BYTES
	};
	b->arrays[DS41F_ARRAY_STEP] = step;
	b->arrays[DS41F_ARRAY_DRAFT_STEP] = step;
	b->arrays[DS41F_ARRAY_DRAFT_COLLAPSE] = (struct linear_range){
		b->draft_slab_collapse + first * slab_collapse,
		(uint64_t)(all ? b->draft_rows : LINEAR_MAXIMUM_ROWS) *
			DS41F_HIDDEN_ROW_BYTES
	};
}

/* A wide program's chunk in pieces of at most LINEAR_WIDE_MOST_ROWS rows, a wide trunk launch's: their
 * count, and piece p's rows through the step area (which no launch of a piece reads). */
uint32_t ds41f_piece_count_of(const struct ds41f_builder *b)
{
	return (b->chunk_rows + LINEAR_WIDE_MOST_ROWS - 1u) /
	       LINEAR_WIDE_MOST_ROWS;
}
void ds41f_piece_select(struct ds41f_builder *b, uint32_t piece)
{
	const uint32_t first_row = piece * LINEAR_WIDE_MOST_ROWS,
		       left = b->chunk_rows - first_row;
	ds41f_rows_select(b, b->chunk_view, b->chunk_first + first_row,
			  left < LINEAR_WIDE_MOST_ROWS ? left :
							 LINEAR_WIDE_MOST_ROWS);
}

/* The card whose launches follow: its arrays for the program's rows, and its kernels (a card gives a kernel prepared
 * again its handle). */
MUST_CHECK result_status ds41f_card_select(struct ds41f_builder *b,
					   uint32_t card)
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
	ds41f_chunk_select(b);
	const uint32_t arithmetic_class =
		b->descriptions[card].arithmetic_class;
	b->arithmetic_class = arithmetic_class;
	b->unequal_ring_kernel = BACKEND_NO_KERNEL;
	result_status status = linear_kernels_prepare(
		b->backend, opened, arithmetic_class, &b->linear);
	status = DS41F_THEN(
		ds41f_hyper_kernels_prepare(b->backend, opened, &b->hyper));
	status = DS41F_THEN(
		dense_linear_kernels_prepare(b->backend, opened, &b->dense));
	status = DS41F_THEN(
		router_kernels_prepare(b->backend, opened, &b->router));
	status = DS41F_THEN(ds41f_attention_state_kernels_prepare(
		b->backend, opened, &b->state));
	/* chunk_core's window writes: the window kernel (the draft's, ds41f_dspark.h) on every card of a chunked
	 * program */
	status = b->slab_count > 1u || b->draft_wide_kernels ?
			 DS41F_THEN(b->backend->kernel_prepare(
				 opened, DS41F_MODULE_ATTENTION_STATE,
				 DS41F_KERNEL_ATTENTION_WINDOW,
				 DS41F_ATTENTION_PREPARE_ARGUMENT_BYTES,
				 &b->state.window)) :
			 status;
	status = DS41F_THEN(ds41f_index_scan_kernels_prepare(
		b->backend, opened, arithmetic_class, &b->scan));
	status = DS41F_THEN(ds41f_attention_core_kernels_prepare(
		b->backend, opened, arithmetic_class, &b->core));
	status = DS41F_THEN(ds41f_engram_rows_kernels_prepare(
		b->backend, opened, &b->engram));
	status = DS41F_THEN(
		sampler_kernels_prepare(b->backend, opened, &b->sampler));
	if (b->vision_active) {
		status = DS41F_THEN(ds41f_image_routing_kernels_prepare(
			b->backend, opened, &b->image_routing));
		if (card == 0) {
			status = DS41F_THEN(ds41f_vision_kernels_prepare(
				b->backend, opened, arithmetic_class,
				&b->vision));
		}
	}
	/* Append wide products behind the thin table; retain the ordered/relaxed trunk selector below. */
	if ((b->slab_count > 1u || b->draft_wide_kernels) &&
	    b->plan->cards[card].layers != 0) {
		status = DS41F_THEN(linear_wide_kernels_prepare(
			b->backend, opened, b->plan->limits.linear_profile,
			b->plan->limits.linear_order, arithmetic_class,
			&b->linear));
		status = DS41F_THEN(dense_linear_wide_kernels_prepare(
			b->backend, opened, arithmetic_class, &b->dense));
		/* dense_skinny_bf16 / _f32 (ds41f_tp_stage.h): the skinny products above 48 rows, their outputs a
	     * block */
		const struct ds41f_tp2_stage_plan *const stage =
			b->tp_description == NULL ||
					arithmetic_class !=
						ARITHMETIC_CLASS_SM_80 ?
				NULL :
				b->tp_description->stages[card];
		status = DS41F_THEN(dense_linear_skinny_kernel_prepare(
			b->backend, opened, DENSE_BFLOAT16,
			stage == NULL ? 0u :
					stage->description.dense_skinny_bf16,
			&b->dense));
		status = DS41F_THEN(dense_linear_skinny_kernel_prepare(
			b->backend, opened, DENSE_FLOAT32,
			stage == NULL ? 0u :
					stage->description.dense_skinny_f32,
			&b->dense));
		/* dense_staged (ds41f_tp_stage.h): the staged wide form of both types */
		status = DS41F_THEN(dense_linear_staged_kernels_prepare(
			b->backend, opened,
			stage == NULL ? 0u : stage->description.dense_staged,
			&b->dense));
		if (b->tensor_layer != NULL || b->draft_wide_kernels)
			status = DS41F_THEN(
				linear_wide_reference_kernels_prepare(
					b->backend, opened, arithmetic_class,
					&b->linear));
	}
	/* EXL3's BF16 QKV shapes require the public wide-four entry in addition to the tuned WO-A policy.
     * Every replicated TP rank owns these target layers even when the canonical primary is rank0. */
	if (b->slab_count > 1u && arithmetic_class == ARITHMETIC_CLASS_SM_80 &&
	    ds41f_card_has_exl3(b->plan, card)) {
		status = DS41F_THEN(b->backend->kernel_prepare(
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
				  b->plan->limits.cmp_order ==
					  CMP_ORDER_RELAXED) ?
			 DS41F_THEN(relaxed_linear_kernels_prepare(
				 b->backend, opened, arithmetic_class,
				 &b->relaxed)) :
			 status;
	status =
		arithmetic_class == ARITHMETIC_CLASS_SM_80 &&
				b->plan->cards[card].layers != 0 &&
				(b->chunk_rows >= EXPERT_ORDERED_MINIMUM_ROWS ||
				 b->plan->limits.cmp_order ==
					 CMP_ORDER_RELAXED) ?
			DS41F_THEN(expert_relaxed_kernels_prepare(
				b->backend, opened, arithmetic_class,
				&b->expert)) :
			status;
	if (status == DS41F_OK &&
	    (card == b->plan->result_card || b->draft_layers) &&
	    b->plan->limits.draft_positions != 0) {
		const uint32_t native_class =
			arithmetic_class == ARITHMETIC_CLASS_SM_120 ?
				ARITHMETIC_CLASS_SM_120_NATIVE :
				arithmetic_class;
		b->draft_linear = b->linear;
		b->draft_core = b->core;
		if (native_class != arithmetic_class) {
			status = linear_kernels_prepare(b->backend, opened,
							native_class,
							&b->draft_linear);
			status =
				DS41F_THEN(ds41f_attention_core_kernels_prepare(
					b->backend, opened, native_class,
					&b->draft_core));
			status = b->slab_count > 1u ?
					 DS41F_THEN(linear_wide_kernels_prepare(
						 b->backend, opened,
						 b->plan->limits.linear_profile,
						 b->plan->limits.linear_order,
						 native_class,
						 &b->draft_linear)) :
					 status;
		}
		status = DS41F_THEN(ds41f_draft_kernels_prepare(
			b->backend, opened, native_class, &b->hyper, &b->router,
			&b->state, &b->draft_core, &b->draft));
		b->draft_carry_kernel = BACKEND_NO_KERNEL;
		const bool carries = b->tp_description != NULL &&
				     native_class == ARITHMETIC_CLASS_SM_80 &&
				     b->tp_description->stages[card]
					     ->description.draft_bf16_operands;
		status = carries ? DS41F_THEN(b->backend->kernel_prepare(
					   opened,
					   KERNEL_FILE_ACTIVATION_QUANTIZE,
					   KERNEL_ACTIVATION_CARRY,
					   ACTIVATION_QUANTIZE_ARGUMENT_BYTES,
					   &b->draft_carry_kernel)) :
				   status;
	}
	if (b->plan->limits.official_hc_rows != 0 &&
	    b->plan->cards[card].layers != 0)
		status = DS41F_THEN(ds41f_official_hc_kernels_prepare(
			b->backend, opened, &b->official_hc));
	/* Append fusions after every existing table. Chunk programs keep their piece-wide products. */
	/* tp4_v3's stage1 epoch02 fix (its engine 16:03): EXL3 tensor slabs take the site fronts too */
	const bool tensor_slab_front =
		b->tensor_layer != NULL &&
		(b->tensor_layer->exact_output_shards ||
		 b->plan->exl3_bits[b->tensor_layer->layer_index] != 0);
	if ((b->slab_count == 1u || tensor_slab_front) &&
	    b->plan->cards[card].layers != 0) {
		status = DS41F_THEN(ds41f_site_front_kernels_prepare(
			b->backend, opened, &b->fronts));
		if (b->plan->limits.row_capacity >= STEP_WIDE_LEAST_ROWS) {
			status = DS41F_THEN(b->backend->kernel_prepare(
				opened, KERNEL_FILE_FILL, KERNEL_FILL_WORDS,
				FILL_RECORD_BYTES, &b->fill));
		}
		status = DS41F_THEN(ds41f_attention_output_kernels_prepare(
			b->backend, opened, arithmetic_class, &b->output));
		status = DS41F_THEN(expert_fused_kernels_prepare(
			b->backend, opened, arithmetic_class, &b->fused));
		b->draft_output = b->output;
		b->draft_fused = b->fused;
		if (card == b->plan->result_card &&
		    b->plan->limits.draft_positions != 0 &&
		    arithmetic_class == ARITHMETIC_CLASS_SM_120) {
			status = DS41F_THEN(
				ds41f_attention_output_kernels_prepare(
					b->backend, opened,
					ARITHMETIC_CLASS_SM_120_NATIVE,
					&b->draft_output));
			status = DS41F_THEN(expert_fused_kernels_prepare(
				b->backend, opened,
				ARITHMETIC_CLASS_SM_120_NATIVE,
				&b->draft_fused));
		}
	}
	if (b->plan->limits.official_reference != 0) {
		status = DS41F_THEN(ds41f_official_reference_kernels_prepare(
			b->backend, opened, arithmetic_class, &b->official));
		if (status == DS41F_OK)
			b->hyper.final_kept = b->official.final_kept;
	}
	return status;
}

/* The context and draft always use the card's native products, even while its target emulates class 80. */
void ds41f_draft_swap(struct ds41f_builder *b, bool begins)
{
	const struct linear_kernels_record linear = b->linear;
	const struct ds41f_attention_core_kernels core = b->core;
	const struct ds41f_attention_output_kernels output = b->output;
	const struct expert_fused_kernels fused = b->fused;
	b->output = b->draft_output;
	b->fused = b->draft_fused;
	b->draft_output = output;
	b->draft_fused = fused;

	b->linear = b->draft_linear;
	b->core = b->draft_core;
	b->draft_linear = linear;
	b->draft_core = core;
	/* draft_bf16_operands: the draft's quantizations carry its rows' BF16 values
	 * (activation_carry) */
	if (b->draft_carry_kernel != BACKEND_NO_KERNEL && begins) {
		b->draft_quantize_kernel = b->linear.activation_quantize;
		b->linear.activation_quantize = b->draft_carry_kernel;
	} else if (b->draft_carry_kernel != BACKEND_NO_KERNEL) {
		b->draft_linear.activation_quantize = b->draft_quantize_kernel;
	}
	const uint32_t target_class = b->descriptions[b->card].arithmetic_class;
	b->arithmetic_class = begins && target_class ==
						      ARITHMETIC_CLASS_SM_120 ?
				      ARITHMETIC_CLASS_SM_120_NATIVE :
				      target_class;
}

/* The card's next count launches; NULL past its part, which plan_is_valid rules out (a prepare refuses NULL). */
struct kernel_launch *ds41f_slots(struct ds41f_builder *b, uint32_t count)
{
	struct program_card_record *const card = &b->made->cards[b->card];
	const bool fits = card->launch_count + count <=
			  b->launch_capacity[b->card];
	card->launch_count += fits ? count : 0;
	return fits ? b->launches[b->card] + (card->launch_count - count) :
		      NULL;
}
