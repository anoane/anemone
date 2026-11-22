/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * ds41f_tp_program_tensors.c: tensor fill, kernels, select, hand-offs, copies, pairs, rotations, boundaries (split
 * from model_v41_program.c by whole items).
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

#include "engine/library/bf16_linear/head_mma_records.h"
#include "engine/library/bf16_linear/trunk_mma_records.h"
#include "models/deepseek_v41_flash/builder/ds41f_builder.h"
#include "ds41f_tp_program.h"
#include "ds41f_tp_program_internal.h"

/* The tensor plan of the card's rank of layer: a target layer, or with draft_layers a draft stage (its channels behind
 * the 40 layers'); build->plan the card's view. */
void ds41f_tp2_program_tensor_fill(
	const struct ds41f_builder *build,
	const struct ds41f_tp2_program_description *description,
	struct ds41f_tp2_layer_plan *tensor, uint32_t card, uint32_t layer)
{
	const struct ds41f_tp2_stage_plan *const stage =
		description->stages[card];
	const uint32_t channel =
		(layer - stage->description.first_target_layer) *
		DS41F_TP2_SITES;
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
	tensor->bf16_ring = stage->description.rank_count == DS41F_TP4_RANKS &&
			    stage->description.wire_format == ORDERED_PUSH_BF16;
	tensor->exact_hidden = stage->exact_hidden;
	tensor->exact_sums = stage->exact_sums;
	tensor->exact_dense = stage->description.exact_dense;
	tensor->exl3_fast_products = stage->description.exl3_fast_products;
	tensor->exl3_wide_entry = stage->description.exl3_wide_entry;
	tensor->exl3_tc_moe = stage->description.exl3_tc_moe;
	tensor->exl3_expert_rotation = stage->description.exl3_expert_rotation;
	tensor->trunk_shape_slices = stage->description.trunk_shape_slices;
	/* a BATCH stage's banks of up to 64 rows serve its BATCH programs; its wide and full programs complete as a
     * THIN stage's do (slab rings, the copy-engine sums, split_overlap, chunk_core: the same programs and bits) */
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
	for (uint32_t site = 0; site < DS41F_TP2_SITES; site++) {
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
	for (uint32_t site = 0; site < DS41F_TP2_SITES; site++) {
		tensor->collectives[site] =
			stage->channels[channel + site].collective;
		/* each rank's region of the site's channel (the copy-engine owner sums' destinations) */
		for (uint32_t rank = 0; rank < stage->description.rank_count &&
					rank < DS41F_TP4_RANKS;
		     rank++)
			tensor->peer_regions[site][rank] =
				description
					->stages[card -
						 stage->description.rank + rank]
					->channels[channel + site]
					.collective.arguments.region_offset;
	}
}

MUST_CHECK result_status ds41f_tp2_program_select(
	struct ds41f_builder *build,
	const struct ds41f_tp2_program_description *description,
	struct ds41f_tp2_layer_plan *tensor, uint32_t card, uint32_t layer)
{
	const bool sharded = layer < DS41F_DRAFT_FIRST_LAYER &&
			     description->stages[card]->description.rank_count >
				     1u;
	if (sharded && tensor == NULL)
		return STATUS_INVALID_ARGUMENT;
	build->plan = description->views[card];
	build->tensor_layer = sharded ? tensor : NULL;
	if (sharded)
		ds41f_tp2_program_tensor_fill(build, description, tensor, card,
					      layer);
	build->draft_layers =
		description->stages[card]->description.draft_layers &&
		tensor != NULL;
	build->tp_description = description;
	build->tp_tensor = tensor;
	result_status status = ds41f_card_select(build, card);
	const struct ds41f_tp2_stage_plan *const own =
		description->stages[card];
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
		status = DS41F_THEN(build->backend->kernel_prepare(
			build->cards[card], HEAD_MMA_KERNEL_FILE,
			"router_mma_entry", sizeof(struct router_mma_arguments),
			&build->router_mma_kernel));
		status = DS41F_THEN(build->backend->kernel_prepare(
			build->cards[card], HEAD_MMA_KERNEL_FILE,
			"router_mma_reduce",
			sizeof(struct router_mma_arguments),
			&build->router_mma_reduce_kernel));
		if (own->description.router_mma_wide)
			status = DS41F_THEN(build->backend->kernel_prepare(
				build->cards[card], HEAD_MMA_KERNEL_FILE,
				"router_mma_wide_entry",
				sizeof(struct router_mma_arguments),
				&build->router_mma_wide_kernel));
	}
	if (own->description.trunk_mma &&
	    own->trunk_mma_partials.byte_count != 0)
		status = DS41F_THEN(build->backend->kernel_prepare(
			build->cards[card], TRUNK_MMA_KERNEL_FILE,
			TRUNK_MMA_WIDE_KERNEL,
			TRUNK_RELAXED_BFLOAT16_ARGUMENT_BYTES,
			&build->trunk_mma_wide_kernel));
	if (own->description.trunk_mma && own->description.trunk_mma_fused &&
	    own->trunk_mma_partials.byte_count != 0)
		status = DS41F_THEN(build->backend->kernel_prepare(
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
	if (own->description.head_mma &&
	    (own->description.head_shards ||
	     card == description->canonical->result_card)) {
		status = DS41F_THEN(build->backend->kernel_prepare(
			build->cards[card], HEAD_MMA_KERNEL_FILE,
			HEAD_MMA_KERNEL_8, DENSE_ARGUMENT_BYTES,
			&build->head_mma_kernel));
		for (uint32_t form = 0; form < HEAD_MMA_WIDE_FORMS; form++)
			status = DS41F_THEN(build->backend->kernel_prepare(
				build->cards[card], HEAD_MMA_KERNEL_FILE,
				head_mma_wide_names[form], DENSE_ARGUMENT_BYTES,
				&build->head_mma_wide_kernels[form]));
	}
	if (own->description.head_shards)
		status = DS41F_THEN(build->backend->kernel_prepare(
			build->cards[card], KERNEL_FILE_SAMPLER_SHARD,
			KERNEL_SAMPLER_SHARD, SAMPLER_SHARD_ARGUMENT_BYTES,
			&build->sampler_shard_kernel));
	if (own->description.head_shards &&
	    card == description->canonical->result_card)
		status = DS41F_THEN(build->backend->kernel_prepare(
			build->cards[card], KERNEL_FILE_SAMPLER_SHARD,
			KERNEL_SAMPLER_SETTINGS_PUSH,
			SAMPLER_SETTINGS_PUSH_ARGUMENT_BYTES,
			&build->settings_push_kernel));
	if (own->description.draft_shards) {
		const bool sends = card == description->canonical->result_card;
		status = DS41F_THEN(build->backend->kernel_prepare(
			build->cards[card], DS41F_MODULE_DRAFT_SHARE,
			sends ? DS41F_KERNEL_DRAFT_SHARE :
				DS41F_KERNEL_DRAFT_SHARE_WAIT,
			sends ? DS41F_DRAFT_SHARE_ARGUMENT_BYTES :
				DS41F_DRAFT_SHARE_WAIT_ARGUMENT_BYTES,
			&build->draft_share_kernel));
		if (!sends) /* card_select prepares the draft's kernels on the result card alone: the Markov bias
			     * here */
			status = DS41F_THEN(build->backend->kernel_prepare(
				build->cards[card], DS41F_MODULE_DSPARK,
				DS41F_KERNEL_DRAFT_MARKOV_BIAS,
				DS41F_DRAFT_MARKOV_BIAS_ARGUMENT_BYTES,
				&build->draft.markov_bias));
	}
	if (own->description.head_stream)
		status = DS41F_THEN(build->backend->kernel_prepare(
			build->cards[card], KERNEL_FILE_DENSE_STREAM,
			KERNEL_DENSE_STREAM, DENSE_ARGUMENT_BYTES,
			&build->head_stream_kernel));
	if ((own->description.batch_draft_rows ||
	     getenv("DIAG_DRAFT_PROBE") != NULL) &&
	    (card == description->canonical->result_card ||
	     own->description.draft_layers))
		status = DS41F_THEN(tp2_join_kernel_prepare(
			build->backend, build->cards[card],
			&build->batch_join_kernel));
	if (own->description.batch_wide_head &&
	    (own->description.head_shards ||
	     card == description->canonical->result_card))
		status = DS41F_THEN(build->backend->kernel_prepare(
			build->cards[card], KERNEL_FILE_DENSE_STREAM,
			KERNEL_DENSE_STREAM_WIDE, DENSE_ARGUMENT_BYTES,
			&build->head_wide_kernel));
	return sharded ?
		       DS41F_THEN(ds41f_tp2_program_tensor_kernels(build, card,
								   tensor)) :
		       status;
}

/* The kernels of the card's rank of a tensor-parallel layer: its joins, rings and pushes. */
MUST_CHECK result_status
ds41f_tp2_program_tensor_kernels(struct ds41f_builder *build, uint32_t card,
				 const struct ds41f_tp2_layer_plan *tensor)
{
	result_status status = tp2_join_kernel_prepare(
		build->backend, build->cards[card], &build->tensor_join_kernel);
	if (tensor->exact_output_shards || tensor->bf16_ring)
		status = DS41F_THEN(build->backend->kernel_prepare(
			build->cards[card], KERNEL_FILE_EXACT_RING,
			KERNEL_EXACT_RING, EXACT_RING_ARGUMENT_BYTES,
			&build->exact_ring_kernel));
	if (tensor->exl3_shard != NULL)
		status = DS41F_THEN(build->backend->kernel_prepare(
			build->cards[card], KERNEL_FILE_EXACT_RING,
			KERNEL_EXACT_RING_UNEQUAL,
			EXACT_RING_UNEQUAL_ARGUMENT_BYTES,
			&build->unequal_ring_kernel));
	if (tensor->exact_output_shards && tensor->original_code_wire) {
		status = DS41F_THEN(build->backend->kernel_prepare(
			build->cards[card], KERNEL_FILE_QUANTIZED_WIRE,
			KERNEL_QUANTIZE_WIRE,
			QUANTIZED_WIRE_PRODUCER_ARGUMENT_BYTES,
			&build->wire_quantize_kernel));
		status = DS41F_THEN(build->backend->kernel_prepare(
			build->cards[card], KERNEL_FILE_QUANTIZED_WIRE,
			KERNEL_QUANTIZED_WIRE, QUANTIZED_WIRE_ARGUMENT_BYTES,
			&build->wire_decode_kernel));
	}
	status = tensor->shared_channels ?
			 DS41F_THEN(tp_exchange_kernel_prepare(
				 build->backend, build->cards[card],
				 &build->tensor_push_kernel)) :
			 DS41F_THEN(ordered_push_kernel_prepare(
				 build->backend, build->cards[card],
				 &build->tensor_push_kernel));
	if (tensor->shared_channels && tensor->wide_exchange != 0)
		status = DS41F_THEN(tp_exchange_wide_kernels_prepare(
			build->backend, build->cards[card],
			TP_EXCHANGE_FORM_PUSH, build->tensor_exchange_kernels));
	if (tensor->shared_channels &&
	    tensor->wide_exchange >= DS41F_TP_EXCHANGE_OWNER_SUMS)
		status = DS41F_THEN(tp_exchange_wide_kernels_prepare(
			build->backend, build->cards[card],
			TP_EXCHANGE_FORM_OWNER_SUMS,
			build->tensor_owner_kernels));
	if (tensor->shared_channels &&
	    tensor->wide_exchange >= DS41F_TP_EXCHANGE_CE_OWNER_SUMS)
		status = DS41F_THEN(tp_exchange_wide_kernels_prepare(
			build->backend, build->cards[card],
			TP_EXCHANGE_FORM_CE_OWNER_SUMS,
			build->tensor_ce_kernels));
	if (tensor->shared_channels &&
	    tensor->wide_exchange == DS41F_TP_EXCHANGE_CE_RINGS)
		status = DS41F_THEN(tp_exchange_wide_kernels_prepare(
			build->backend, build->cards[card],
			TP_EXCHANGE_FORM_CE_OWNER_RING_SUMS,
			build->tensor_ce_ring_kernels));
	if (tensor->shared_channels &&
	    tensor->wide_exchange == DS41F_TP_EXCHANGE_OWNER_RINGS)
		status = DS41F_THEN(tp_exchange_wide_kernels_prepare(
			build->backend, build->cards[card],
			TP_EXCHANGE_FORM_OWNER_RING_SUMS,
			build->tensor_ring_kernels));
	return status;
}

MUST_CHECK result_status ds41f_tp2_program_segment(struct ds41f_builder *build,
						   uint32_t layer,
						   uint32_t first_launch)
{
	const uint32_t before = build->made->program.segment_count;
	ds41f_segment_add(build, layer, first_launch);
	return build->made->program.segment_count == before + 1u ?
		       DS41F_OK :
		       STATUS_INVALID_ARGUMENT;
}

MUST_CHECK static result_status
tp2_program_hand_off(struct ds41f_builder *build, uint32_t destination,
		     uint64_t source_offset, uint64_t destination_offset,
		     uint32_t bytes, uint32_t ratio, uint64_t stride)
{
	if (build->made->program.hand_off_count >= DS41F_MOST_HAND_OFFS ||
	    build->made->program.segment_count == 0 ||
	    source_offset == PROGRAM_NOWHERE ||
	    destination_offset == PROGRAM_NOWHERE)
		return STATUS_INVALID_ARGUMENT;
	ds41f_hand_off(build, destination, source_offset, destination_offset,
		       bytes, ratio, stride);
	return DS41F_OK;
}

/* Copies from the canonical producer precede its adjacent companion in the segment table. Peers which run that
 * source/index layer compute the same values locally and must not receive a handoff inside their companion pair. */
MUST_CHECK result_status ds41f_tp2_program_copies(struct ds41f_builder *build,
						  uint32_t layer,
						  uint32_t copied)
{
	const struct ds41f_plan *const plan = build->plan;
	const struct ds41f_card *const from = &plan->cards[build->card];
	const bool source = (DS41F_SOURCE_LAYERS >> layer & 1u) != 0;
	const uint32_t ratio = (DS41F_POOLED_LAYERS >> layer & 1u) != 0 ?
				       DS41F_POOLED_RATIO :
				       DS41F_DIRECT_RATIO;
	const bool writes[DS41F_COPIED_ARRAYS] = {
		(DS41F_INDEX_LAYERS >> layer & 1u) != 0, source, source,
		layer == DS41F_CANDIDATE_SOURCE
	};
	const uint64_t sources[DS41F_COPIED_ARRAYS] = {
		from->selection_offset[layer], from->pool_offset[layer],
		from->key_tiles_offset[layer],
		from->array_offset[DS41F_ARRAY_CANDIDATES]
	};
	const uint64_t strides[DS41F_COPIED_ARRAYS] = {
		0, plan->pool_session_bytes[layer],
		plan->key_session_bytes[layer], 0
	};
	result_status status = DS41F_OK;
	for (uint32_t destination = 0;
	     destination < plan->card_count && status == DS41F_OK;
	     destination++) {
		const struct ds41f_card *const other =
			&plan->cards[destination];
		if (destination == build->card ||
		    (other->layers >> layer & 1u) != 0)
			continue;
		const uint64_t targets[DS41F_COPIED_ARRAYS] = {
			other->selection_offset[layer],
			other->pool_offset[layer],
			other->key_tiles_offset[layer],
			other->array_offset[DS41F_ARRAY_CANDIDATES]
		};
		for (uint32_t array = 0;
		     array < DS41F_COPIED_ARRAYS && status == DS41F_OK;
		     array++) {
			if (writes[array] && (copied >> array & 1u) != 0 &&
			    targets[array] != PROGRAM_NOWHERE)
				status = tp2_program_hand_off(
					build, destination, sources[array],
					targets[array],
					ds41f_copied_bytes[array],
					strides[array] == 0 ? 0u : ratio,
					strides[array]);
		}
	}
	return status;
}

MUST_CHECK result_status ds41f_tp2_program_streams(struct ds41f_builder *build,
						   uint32_t destination)
{
	const uint64_t *const from =
		build->plan->cards[build->card].array_offset;
	const uint64_t *const to = build->plan->cards[destination].array_offset;
	result_status status = tp2_program_hand_off(
		build, destination, from[DS41F_ARRAY_STREAMS],
		to[DS41F_ARRAY_STREAMS], DS41F_STREAMS_ROW_BYTES, 0, 0);
	return DS41F_THEN(tp2_program_hand_off(
		build, destination, from[DS41F_ARRAY_CARRIED_PRE],
		to[DS41F_ARRAY_CARRIED_PRE], DS41F_SITE_ROW_BYTES, 0, 0));
}

MUST_CHECK result_status ds41f_tp2_program_pair_add(
	struct ds41f_builder *build, struct step_companion_pair *pairs,
	uint32_t *pair_count, uint32_t first_segment)
{
	const uint32_t members =
		build->made->program.segment_count - first_segment;
	if (*pair_count >= DS41F_TP2_PROGRAM_PAIRS ||
	    (members != DS41F_TP2_RANKS && members != DS41F_TP4_RANKS))
		return STATUS_INVALID_ARGUMENT;
	pairs[(*pair_count)++] =
		(struct step_companion_pair){ first_segment,
					      first_segment + members - 1u };
	return DS41F_OK;
}

MUST_CHECK result_status ds41f_tp2_program_rotations(
	struct ds41f_builder *build,
	const struct ds41f_tp2_program_description *description,
	struct ds41f_tp2_layer_plan *tensor, struct step_companion_pair *pairs,
	uint32_t *pair_count, uint32_t first_card, uint32_t first_layer,
	bool embeds)
{
	const uint32_t first_segment = build->made->program.segment_count;
	const uint32_t ranks =
		description->stages[first_card]->description.rank_count;
	result_status status = DS41F_OK;
	for (uint32_t rank = 0; rank < ranks && status == DS41F_OK; rank++) {
		status = ds41f_tp2_program_select(build, description, tensor,
						  first_card + rank,
						  first_layer);
		const uint32_t first_launch =
			build->made->cards[build->card].launch_count;
		if (embeds) { /* a replicated embedding: every rank expands its own rows (the settings push first) */
			const struct ds41f_tp2_layer_plan *const held =
				build->tensor_layer;
			build->tensor_layer = NULL;
			const enum program_kind kind =
				build->made->program.kind;
			if (status == DS41F_OK && build->head_stage != NULL &&
			    build->card ==
				    description->canonical->result_card &&
			    (kind == PROGRAM_THIN || kind == PROGRAM_BATCH))
				status = ds41f_settings_push_launch(build);
			status = DS41F_THEN(ds41f_step_launches(build, true));
			build->tensor_layer = held;
		}
		for (uint32_t slab = 0;
		     slab < build->slab_count && status == DS41F_OK; slab++) {
			if (build->slab_count > 1u)
				ds41f_slab_select(build, slab);
			status = ds41f_step_launches(build, false);
		}
		status = DS41F_THEN(ds41f_tp2_program_segment(
			build, SEGMENT_NO_LAYER, first_launch));
	}
	return ranks > 1u ? DS41F_THEN(ds41f_tp2_program_pair_add(
				    build, pairs, pair_count, first_segment)) :
			    status;
}

MUST_CHECK result_status ds41f_tp2_program_boundary(
	struct ds41f_builder *build,
	const struct ds41f_tp2_program_description *description, uint32_t layer)
{
	if (layer + 1u == DS41F_LAYERS)
		return DS41F_OK;
	const uint32_t owner = ds41f_tp2_program_layer_card(description, layer);
	const uint32_t next =
		ds41f_tp2_program_layer_card(description, layer + 1u);
	result_status status = DS41F_OK;
	if (owner != next) {
		for (uint32_t rank = 0;
		     rank < description->stages[next]->description.rank_count &&
		     status == DS41F_OK;
		     rank++)
			status = ds41f_tp2_program_streams(build, next + rank);
	}
	return status;
}
