/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * ds41f_tp_stage_plan.c: the TP2 stage plans (split from model_v41.c by whole items).
 */
#include "models/deepseek_v41_flash/plan/model_v41.h"
#include "models/deepseek_v41_flash/official/ds41f_official_hc.h"
#include "models/deepseek_v41_flash/official/ds41f_official_reference.h"
#include "ds41f_tp_layer.h"
#include "ds41f_tp_stage.h"
#include "engine/library/exl3/exl3_tc_records.h"
#include "engine/library/rings/exact_ring_records.h"
#include "engine/library/exl3/exl3.h"
#include "models/deepseek_v41_flash/draft/ds41f_dspark_records.h"
#include "ds41f_draft_share_records.h"
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

#include "models/deepseek_v41_flash/plan/ds41f_plan.h"
#include "ds41f_tp_plan_internal.h"
#include "models/deepseek_v41_flash/tensors/ds41f_tensors.h"

/* Cold stage inventory: intentionally separate from the PP and executable layer-zero entry points above. */
struct tp2_stage_pending {
	struct ds41f_tp2_pending_tensor tensor;
	uint32_t layer;
	bool is_expert;
};

static const struct ds41f_tensor_rule *tp2_stage_rule(uint32_t layer,
						      uint32_t tensor)
{
	return layer >= DS41F_DRAFT_FIRST_LAYER &&
			       tensor >= DS41F_TENSOR_ROUTER &&
			       tensor <= DS41F_TENSOR_ROUTER_BIAS ?
		       &ds41f_draft_router_rules[tensor - DS41F_TENSOR_ROUTER] :
		       &ds41f_layer_rules[tensor];
}

static int tp2_stage_pending_compare(const void *first, const void *second)
{
	const struct tp2_stage_pending *const left = first, *const right =
								    second;
	return ds41f_tp2_pending_compare(&left->tensor, &right->tensor);
}

#define TP2_STAGE_MARKOV_TABLES \
	2u /* draft_shards' ranks 1-3: the Markov head's embedding and head */
#define TP2_STAGE_RANK_DRAFT_GLOBALS \
	3u /* draft_layers' ranks 1-3: main_proj, main_norm and the draft's norm */
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
	count += tp2_stage_head_rows(description) ? DS41F_TENSOR_GLOBALS - 1u :
						    0u;
	count += tp2_stage_head_rows(description) && description->draft_shards ?
			 TP2_STAGE_MARKOV_TABLES :
			 0u;
	count +=
		description->include_head_and_draft ?
			DS41F_TENSOR_GLOBALS - 1u + DS41F_TENSOR_DRAFT_GLOBALS :
			0;
	count += description->draft_layers &&
				 !description->include_head_and_draft ?
			 TP2_STAGE_RANK_DRAFT_GLOBALS :
			 0;
	for (uint32_t layer = 0; layer < DS41F_WALK_LAYERS; layer++) {
		if ((walked & DS41F_BIT(layer)) == 0)
			continue;
		for (uint32_t tensor = 0; tensor < DS41F_TENSOR_LAYER_COUNT;
		     tensor++)
			count += (tp2_stage_rule(layer, tensor)->layers &
				  DS41F_BIT(layer)) != 0 ?
					 1u :
					 0u;
		count += (layer >= DS41F_DRAFT_FIRST_LAYER ?
				  DS41F_DRAFT_EXPERTS :
				  DS41F_EXPERTS) *
			 DS41F_EXPERT_MATRICES;
	}
	return count;
}

/* Populate global source geometry before the single card layout, including sources owned by earlier stages. */
static void tp2_stage_holders(struct ds41f_plan *model, uint32_t card_index,
			      uint64_t targets, bool draft, bool draft_copy)
{
	struct ds41f_card *const card = &model->cards[card_index];
	card->layers = targets;
	card->holders =
		(uint32_t)(DS41F_BIT(DS41F_HOLDER_TRUNK) |
			   DS41F_BIT(DS41F_HOLDER_LAYER) |
			   (draft ? DS41F_BIT(DS41F_HOLDER_RESULT) |
					    DS41F_BIT(DS41F_HOLDER_DRAFT) :
				    0) |
			   (draft_copy ? DS41F_BIT(DS41F_HOLDER_DRAFT_COPY) :
					 0));
	uint32_t table = 0;
	for (uint32_t layer = 0; layer < DS41F_WALK_LAYERS; layer++) {
		model->layer_card[layer] =
			(ds41f_walk_layers(card) & DS41F_BIT(layer)) != 0 ?
				card_index :
				BACKEND_MAXIMUM_CARDS;
		for (uint32_t tensor = 0; tensor < DS41F_TENSOR_LAYER_COUNT;
		     tensor++)
			model->layer_tensors[layer][tensor].card =
				BACKEND_MAXIMUM_CARDS;
		if ((DS41F_SOURCE_LAYERS & DS41F_BIT(layer)) != 0) {
			const uint64_t entries =
				model->limits.context_token_limit /
				((DS41F_POOLED_LAYERS & DS41F_BIT(layer)) != 0 ?
					 DS41F_POOLED_RATIO :
					 DS41F_DIRECT_RATIO);
			model->pool_session_bytes[layer] =
				entries * DS41F_STATE_ROW_BYTES;
			model->key_session_bytes[layer] =
				entries * DS41F_KEY_BYTES;
		}
		if ((targets & DS41F_BIT(layer)) != 0) {
			card->holders |=
				(uint32_t)((DS41F_INDEX_LAYERS &
					    DS41F_BIT(layer)) != 0 ?
						   DS41F_BIT(
							   DS41F_HOLDER_INDEX) :
						   0);
			card->holders |=
				(uint32_t)((DS41F_SOURCE_LAYERS &
					    DS41F_BIT(layer)) != 0 ?
						   DS41F_BIT(
							   DS41F_HOLDER_SOURCE) :
						   0);
			card->holders |=
				(uint32_t)((DS41F_POOLED_LAYERS &
					    DS41F_BIT(layer)) != 0 ?
						   DS41F_BIT(
							   DS41F_HOLDER_POOLED) :
						   0);
			card->holders |=
				(uint32_t)(layer == DS41F_CANDIDATE_SOURCE ||
							   (DS41F_CANDIDATE_READERS &
							    DS41F_BIT(layer)) !=
								   0 ?
						   DS41F_BIT(
							   DS41F_HOLDER_CANDIDATES) :
						   0);
		}
		if ((DS41F_ENGRAM_LAYERS & DS41F_BIT(layer)) != 0) {
			if ((targets & DS41F_BIT(layer)) != 0) {
				card->holders |= (uint32_t)DS41F_BIT(
					DS41F_HOLDER_ENGRAM);
				card->engram_tables |=
					(uint32_t)DS41F_BIT(table);
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
	const struct ds41f_tensor_rule *rule, bool split_outputs,
	bool split_inputs, struct ds41f_tensor *tensor_out, uint32_t layer,
	uint64_t *expert_offset, uint32_t chunk_rank)
{
	if (stage->job_count >= capacity)
		return STATUS_INVALID_ARGUMENT;
	struct tp2_stage_pending *const item = &pending[stage->job_count];
	const struct ds41f_tp2_stage_description *const description =
		&stage->description;
	const bool whole = layer >= DS41F_DRAFT_FIRST_LAYER &&
			   !description->draft_layers;
	const result_status status = ds41f_tp2_tensor_find(
		pack, name, rule, description->card, whole ? 0u : chunk_rank,
		whole ? 1u : description->rank_count, split_outputs,
		split_inputs, description->exact_output_shards, tensor_out,
		&item->tensor);
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
		const uint32_t source =
			ds41f_latest_of(DS41F_SOURCE_LAYERS, layer);
		const uint64_t scored =
			source < DS41F_LAYERS ?
				plan->key_session_bytes[source] /
					DS41F_KEY_BYTES :
				0u;
		entries = (card->layers & DS41F_INDEX_LAYERS &
			   DS41F_BIT(layer)) != 0 &&
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
			ds41f_take(&next, DS41F_FUSED_SCRATCH_BYTES);
		card->producer_arrivals_offset =
			ds41f_sum(card->front_scratch_offset,
				  DS41F_SITE_FRONT_SCRATCH_BYTES);
	}
	const uint32_t ranks = description->rank_count;
	const uint64_t compact_bytes =
		(uint64_t)description->limits.row_capacity *
		DS41F_HEADS_ROW_BYTES / ranks;
	stage->compact =
		(struct linear_range){ ds41f_take(&next, compact_bytes),
				       compact_bytes };
	if (description->index_span) {
		/* a span's scores and split annex (ds41f_index_scan.h), in whole slabs within DS41F_INDEX_SPAN_BYTES;
		 * fewer
         * than two slabs: none (the scans a slab at a time) */
		const uint64_t row_bytes =
			card_index_entries(stage->model, card) *
				DS41F_INDEX_SCORE_BYTES +
			DS41F_INDEX_SPLIT_ROW_BYTES;
		const uint64_t fit = DS41F_INDEX_SPAN_BYTES / row_bytes /
				     LINEAR_MAXIMUM_ROWS * LINEAR_MAXIMUM_ROWS;
		const uint64_t rows = fit < DS41F_SPAN_ROWS ? fit :
							      DS41F_SPAN_ROWS;
		stage->index_rows =
			rows > LINEAR_MAXIMUM_ROWS &&
					row_bytes >
						DS41F_INDEX_SPLIT_ROW_BYTES ?
				(uint32_t)rows :
				0u;
		stage->index_scores = (struct linear_range){
			ds41f_take(&next, stage->index_rows * row_bytes),
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
			(struct linear_range){ ds41f_take(&next, hidden_bytes),
					       hidden_bytes };
		stage->exact_sums =
			(struct linear_range){ ds41f_take(&next, sum_bytes),
					       sum_bytes };
	}
	if (description->exact_dense) {
		/* the thin gathers' owner-major slots, then a piece's own outputs [rows][widest own]
		 * (model_v41_program.c) */
		const uint64_t dense_input_bytes =
			(uint64_t)LINEAR_MAXIMUM_ROWS * DS41F_WIDEST_ROW *
				sizeof(uint16_t) +
			(uint64_t)description->limits.row_capacity *
				(DS41F_WIDEST_ROW / ranks) * sizeof(uint16_t);
		stage->dense_input = (struct linear_range){
			ds41f_take(&next, dense_input_bytes), dense_input_bytes
		};
	}
	const uint64_t elements = (uint64_t)stage->channel_rows * DS41F_HIDDEN;
	const uint64_t bytes = elements * sizeof(float);
	const uint64_t signals = (uint64_t)ORDERED_PUSH_PHASES * ranks *
				 ORDERED_COLLECTIVE_LINE_BYTES;
	const uint64_t stride_bytes =
		signals + (ranks + DS41F_TP_LOCAL_VECTORS) * bytes;
	const uint64_t stride =
		(stride_bytes + DS41F_ARRAY_ALIGNMENT_BYTES - 1u) /
		DS41F_ARRAY_ALIGNMENT_BYTES * DS41F_ARRAY_ALIGNMENT_BYTES;
	/* BATCH steps, and with piece_rings any step of more than 8 rows: BF16 rings of up to 64 rows (a bank a
	 * count) */
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
		channel->layer = description->first_target_layer +
				 index / DS41F_TP2_SITES;
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
				stage->channels[channel->site]
					.dense_ring_bank_bytes;
			continue;
		}
		const uint64_t table_offset = ds41f_take(
			&next, (uint64_t)ranks * ORDERED_COLLECTIVE_PEER_BYTES);
		const uint64_t region = ds41f_take(
			&next, ds41f_product(stride, ORDERED_COLLECTIVE_SLOTS));
		if (description->exact_output_shards ||
		    (description->rank_count == DS41F_TP4_RANKS &&
		     description->wire_format == ORDERED_PUSH_BF16)) {
			const uint64_t payload_bytes =
				batch_rings ?
					(uint64_t)stage->ring_rows *
						DS41F_HIDDEN *
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
						channel->site == 0 &&
						!batch_rings ?
					(uint64_t)LINEAR_MAXIMUM_ROWS *
						DS41F_EXPERTS_PER_ROW *
						DS41F_TP4_LARGE_INNER *
						sizeof(uint16_t) /
						EXACT_RING_PACK_BYTES :
					payload_bytes / ranks /
						EXACT_RING_PACK_BYTES;
			/* a bank a row count: one size, the gathers' slots of the most packs (the thin banks: room for
			 * the
             * BF16 sum's of 8 rows too); batch_rings: the BF16 sum's slots of its own rows (12 a pack: 64
             * rows in a
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
					  maximum_packs *
						  EXACT_RING_EPOCH_HOPS *
						  EXACT_RING_PACKET_BYTES) +
				 EXACT_RING_METADATA_BYTES +
				 DS41F_ARRAY_ALIGNMENT_BYTES - 1u) /
				DS41F_ARRAY_ALIGNMENT_BYTES *
				DS41F_ARRAY_ALIGNMENT_BYTES;
			channel->ring_offset = ds41f_take(
				&next,
				ds41f_ring_bank_start(channel->ring_bank_bytes,
						      channel->ring_row_bytes,
						      stage->ring_rows + 1u));
		}
		if (description->exact_dense) {
			/* separate banks: a bank keeps one gather geometry per prepared row count (two-epoch slots);
			 * site 0 the
             * groups (wo_a's 8192 columns), site 1 the sublayer output (5120) */
			const uint64_t dense_payload_bytes =
				(uint64_t)LINEAR_MAXIMUM_ROWS *
				sizeof(uint16_t) *
				(channel->site == 0 ? DS41F_WIDEST_ROW :
						      DS41F_HIDDEN);
			const uint64_t dense_packs = dense_payload_bytes /
						     ranks /
						     EXACT_RING_PACK_BYTES;
			channel->dense_ring_bank_bytes =
				(dense_packs * EXACT_RING_EPOCH_HOPS *
					 EXACT_RING_PACKET_BYTES +
				 EXACT_RING_METADATA_BYTES +
				 DS41F_ARRAY_ALIGNMENT_BYTES - 1u) /
				DS41F_ARRAY_ALIGNMENT_BYTES *
				DS41F_ARRAY_ALIGNMENT_BYTES;
			channel->dense_ring_offset = ds41f_take(
				&next, channel->dense_ring_bank_bytes *
					       LINEAR_MAXIMUM_ROWS);
		}
		channel->collective.arguments = (struct ordered_push_arguments){
			.peer_table_offset = table_offset,
			.region_offset = region,
			.input_offset = signals,
			.output_offset = signals + bytes,
			.inbox_offset =
				signals + DS41F_TP_LOCAL_VECTORS * bytes,
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
			char name[DS41F_NAME_CAPACITY];
			const int length = snprintf(
				name, sizeof(name),
				"layers.%u.ffn.experts.%u.%s", layer, expert,
				ds41f_expert_rules[matrix].name);
			if (length <= 0 || length >= DS41F_NAME_CAPACITY)
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
		ds41f_take(&next,
			   (uint64_t)SAMPLER_SHARD_DEFAULT_EXCHANGE_BYTES),
		(uint64_t)SAMPLER_SHARD_DEFAULT_EXCHANGE_BYTES
	};
	stage->sampler_peers = (struct linear_range){
		ds41f_take(&next, SAMPLER_SHARD_PEER_TABLE_BYTES),
		SAMPLER_SHARD_PEER_TABLE_BYTES
	};
	const uint64_t settings =
		SAMPLER_SETTINGS_SLOTS_OFFSET +
		(uint64_t)description->limits.session_capacity *
			SAMPLER_SETTINGS_SLOT_BYTES;
	stage->sampler_settings =
		(struct linear_range){ ds41f_take(&next, settings), settings };
	if (description->include_head_and_draft)
		stage->sampler_settings_peers =
			(struct linear_range){ ds41f_take(&next, peers),
					       peers };
	if (description->draft_shards && description->include_head_and_draft) {
		const uint64_t counters =
			(uint64_t)DS41F_DRAFT_SHARE_RANKS * sizeof(uint32_t);
		stage->draft_peers =
			(struct linear_range){ ds41f_take(&next, peers),
					       peers };
		stage->draft_counters =
			(struct linear_range){ ds41f_take(&next, counters),
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
			(struct linear_range){ ds41f_take(&next, draft_bytes),
					       draft_bytes };
	}
	if (next == UINT64_MAX)
		return STATUS_INVALID_ARGUMENT;
	stage->communication_byte_count = ds41f_sum(
		stage->communication_byte_count, next - card->arena_byte_count);
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
		ds41f_take(&next, (uint64_t)SAMPLER_SHARD_ENTRY_BYTES),
		(uint64_t)SAMPLER_SHARD_ENTRY_BYTES
	};
	if (next == UINT64_MAX)
		return STATUS_INVALID_ARGUMENT;
	stage->communication_byte_count = ds41f_sum(
		stage->communication_byte_count, next - card->arena_byte_count);
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
		(struct linear_range){ ds41f_take(&next, bytes), bytes };
	const uint64_t tail_bytes =
		overlay ? exl3_tc_workspace_bytes(
				  tail_rows * DS41F_EXL3_TC_PAIRS_PER_ROW,
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
			(struct linear_range){ ds41f_take(&next, span), span };
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
		(struct linear_range){ ds41f_take(&next, bytes), bytes };
	const uint64_t capacity = stage->description.limits.row_capacity;
	const uint64_t tail_rows = capacity < stage->description.tail_rows ?
					   capacity :
					   stage->description.tail_rows;
	if (stage->description.tail_sets && tail_rows > STEP_WIDE_LEAST_ROWS) {
		const uint64_t set_bytes = tail_rows *
					   DS41F_TRUNK_MMA_PARTIAL_VALUES *
					   sizeof(float);
		stage->trunk_mma_set_partials =
			(struct linear_range){ ds41f_take(&next, set_bytes),
					       set_bytes };
	}
	if (next == UINT64_MAX)
		return STATUS_INVALID_ARGUMENT;
	card->arena_byte_count = next;
	for (uint32_t index = 0; index < stage->channel_count;
	     index++) /* peer qualification compares them */
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
		(struct linear_range){ ds41f_take(&next, bytes), bytes };
	/* the round table right behind the slab results: the step driver downloads both as one result span */
	const uint32_t positions = stage->description.limits.draft_positions;
	/* draft_layers: every rank its own round table (its verify writes one) and batched draft area */
	const bool rounds_here = (stage->description.include_head_and_draft ||
				  stage->description.draft_layers) &&
				 positions != 0;
	const uint64_t sessions = stage->description.limits.session_capacity;
	if (rounds_here)
		stage->batch_rounds = (struct linear_range){
			ds41f_take(&next, sessions * STEP_ROUND_BYTES),
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
			(struct linear_range){ ds41f_take(&next, logits),
					       logits };
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
			(struct linear_range){ ds41f_take(&next, drafts),
					       drafts };
		if (stage->description.batch_draft_rows) {
			const uint64_t rows =
				ds41f_batch_rows_offset(DS41F_BATCH_ROWS_PARTS);
			stage->batch_rows =
				(struct linear_range){ ds41f_take(&next, rows),
						       rows };
		}
	}
	if (next == UINT64_MAX)
		return STATUS_INVALID_ARGUMENT;
	stage->communication_byte_count = ds41f_sum(
		stage->communication_byte_count, next - card->arena_byte_count);
	card->arena_byte_count = next;
	for (uint32_t index = 0; index < stage->channel_count; index++)
		stage->channels[index].collective.arena_byte_count = next;
	return STATUS_OK;
}

result_status ds41f_tp2_stage_plan_create(
	const struct pack_record *pack,
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
	      description->target_layer_count !=
		      DS41F_TP4_ATTENTION_PREFIX_END ||
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
	      description->target_layer_count !=
		      DS41F_TP4_ATTENTION_PREFIX_END ||
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
	result_status status = ds41f_model_check(pack);
	status = status == STATUS_OK ?
			 ds41f_limits_check(&description->limits) :
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
	const uint64_t targets =
		(DS41F_BIT(description->target_layer_count) - 1u)
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
						    DS41F_BIT(layer) :
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
		for (uint32_t tensor = 0; tensor < DS41F_TENSOR_GLOBALS;
		     tensor++)
			model->globals[tensor].card = BACKEND_MAXIMUM_CARDS;
		for (uint32_t tensor = 0; tensor < DS41F_TENSOR_DRAFT_GLOBALS;
		     tensor++)
			model->draft_globals[tensor].card =
				BACKEND_MAXIMUM_CARDS;
		tp2_stage_holders(model, description->card, targets,
				  description->include_head_and_draft ||
					  description->draft_layers,
				  description->include_head_and_draft);
		model->cards[description->card].holders |=
			(uint32_t)(tp2_stage_head_rows(description) ?
					   DS41F_BIT(DS41F_HOLDER_RESULT) :
					   0);
	}
	for (uint32_t tensor = 0;
	     status == STATUS_OK && tensor < DS41F_TENSOR_GLOBALS; tensor++) {
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
					      ds41f_global_rules[tensor].name,
					      &ds41f_global_rules[tensor],
					      head_cut, false,
					      &model->globals[tensor], 0, NULL,
					      description->rank);
	}
	for (uint32_t tensor = 0;
	     status == STATUS_OK && description->include_head_and_draft &&
	     tensor < DS41F_TENSOR_DRAFT_GLOBALS;
	     tensor++)
		status = tp2_stage_tensor_add(pack, stage, pending, capacity,
					      ds41f_draft_rules[tensor].name,
					      &ds41f_draft_rules[tensor], false,
					      false,
					      &model->draft_globals[tensor], 0,
					      NULL, description->rank);
	for (uint32_t tensor = DS41F_TENSOR_DRAFT_MARKOV_EMBEDDING;
	     status == STATUS_OK && tp2_stage_head_rows(description) &&
	     description->draft_shards &&
	     tensor < DS41F_TENSOR_DRAFT_MARKOV_EMBEDDING +
			      TP2_STAGE_MARKOV_TABLES;
	     tensor++)
		status = tp2_stage_tensor_add(pack, stage, pending, capacity,
					      ds41f_draft_rules[tensor].name,
					      &ds41f_draft_rules[tensor], false,
					      false,
					      &model->draft_globals[tensor], 0,
					      NULL, description->rank);
	/* draft_layers' ranks 1-3: their draft context's main_proj and main_norm, the draft's final norm */
	static const uint32_t rank_draft_globals[TP2_STAGE_RANK_DRAFT_GLOBALS] = {
		DS41F_TENSOR_DRAFT_MAIN_PROJECTION,
		DS41F_TENSOR_DRAFT_MAIN_NORM, DS41F_TENSOR_DRAFT_NORM
	};
	for (uint32_t index = 0;
	     status == STATUS_OK && description->draft_layers &&
	     !description->include_head_and_draft &&
	     index < TP2_STAGE_RANK_DRAFT_GLOBALS;
	     index++)
		status = tp2_stage_tensor_add(
			pack, stage, pending, capacity,
			ds41f_draft_rules[rank_draft_globals[index]].name,
			&ds41f_draft_rules[rank_draft_globals[index]], false,
			false, &model->draft_globals[rank_draft_globals[index]],
			0, NULL, description->rank);
	for (uint32_t layer = 0;
	     status == STATUS_OK && layer < DS41F_WALK_LAYERS; layer++) {
		if ((walked & DS41F_BIT(layer)) == 0)
			continue;
		char prefix[DS41F_LAYER_PREFIX_CAPACITY];
		ds41f_layer_prefix_of(layer, prefix);
		if (layer < DS41F_LAYERS) {
			status = tp2_stage_exl3_discover(
				pack, layer, &model->exl3_bits[layer]);
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
			if (status == STATUS_OK &&
			    model->exl3_bits[layer] != 0 &&
			    description->exact_output_shards)
				status = bb_exl3_output_shard_prepare(
					DS41F_HIDDEN, DS41F_EXPERT_INNER,
					model->exl3_bits[layer],
					description->rank_count,
					description->rank,
					description->exl3_product_order,
					&stage->exl3_shards[layer]);
		}
		for (uint32_t tensor = 0;
		     status == STATUS_OK && tensor < DS41F_TENSOR_LAYER_COUNT;
		     tensor++) {
			const struct ds41f_tensor_rule *const rule =
				tp2_stage_rule(layer, tensor);
			if ((rule->layers & DS41F_BIT(layer)) == 0)
				continue;
			char name[DS41F_NAME_CAPACITY];
			const int length = snprintf(name, sizeof(name), "%s%s",
						    prefix, rule->name);
			if (length <= 0 || length >= DS41F_NAME_CAPACITY) {
				status = STATUS_INVALID_ARGUMENT;
				break;
			}
			/* exact_dense: wq_b by heads, wo_a by groups, wo_b by output channels, each output's full-K
		     * chain */
			const bool dense_tensor =
				tensor == DS41F_TENSOR_QUERY_UP ||
				tensor == DS41F_TENSOR_OUTPUT_GROUPS ||
				tensor == DS41F_TENSOR_OUTPUT;
			const bool split_outputs =
				description->exact_output_shards ?
					description->exact_dense &&
						dense_tensor :
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
				&model->layer_tensors[layer][tensor], layer,
				NULL, description->rank);
		}
		const uint32_t experts = layer >= DS41F_DRAFT_FIRST_LAYER ?
						 DS41F_DRAFT_EXPERTS :
						 DS41F_EXPERTS;
		for (uint32_t expert = 0;
		     status == STATUS_OK && expert < experts; expert++) {
			for (uint32_t matrix = 0;
			     status == STATUS_OK &&
			     matrix < DS41F_EXPERT_MATRICES;
			     matrix++) {
				char name[DS41F_NAME_CAPACITY];
				const int length = snprintf(
					name, sizeof(name),
					"%sffn.experts.%u.%s", prefix, expert,
					ds41f_expert_rules[matrix].name);
				if (length <= 0 ||
				    length >= DS41F_NAME_CAPACITY) {
					status = STATUS_INVALID_ARGUMENT;
					break;
				}
				struct ds41f_tensor_rule rule =
					ds41f_expert_rules[matrix];
				if (model->exl3_bits[layer] != 0) {
					rule.weight_format = WEIGHT_FORMAT_EXL3;
					rule.bits_per_weight =
						model->exl3_bits[layer];
				}
				/* exl3_expert_rotation: the chunk of ds41f_exl3_tc_chunk_of (exl3_tc_records.h) */
				const uint32_t chunk_rank =
					description->exl3_expert_rotation !=
								0 &&
							layer < DS41F_LAYERS &&
							model->exl3_bits[layer] !=
								0 &&
							description->rank_count ==
								DS41F_TP4_RANKS ?
						ds41f_exl3_tc_chunk_of(
							description
								->exl3_expert_rotation,
							description->rank,
							layer, expert) :
						description->rank;
				status = tp2_stage_tensor_add(
					pack, stage, pending, capacity, name,
					&rule,
					description->exact_output_shards ||
						matrix != DS41F_TENSOR_DOWN,
					!description->exact_output_shards &&
						matrix == DS41F_TENSOR_DOWN,
					NULL, layer,
					&model->expert_tables[layer][expert]
							     [matrix],
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
	for (uint32_t index = 0;
	     status == STATUS_OK && index < stage->job_count; index++) {
		struct tp2_stage_pending *const item = &pending[index];
		item->tensor.job.destination_offset =
			ds41f_take(&next, item->tensor.byte_count);
		stage->jobs[index] = item->tensor.job;
		stage->tensor_payload_byte_count =
			ds41f_sum(stage->tensor_payload_byte_count,
				  item->tensor.byte_count);
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
				ds41f_sum(item->tensor.job.destination_offset,
					  item->tensor.byte_count);
			range->byte_count = end > range->byte_count ?
						    end :
						    range->byte_count;
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
		struct ds41f_card *const card =
			&model->cards[description->card];
		stage->aligned_weight_byte_count = card->state_offset = next;
		status = ds41f_card_lay_out_attention(model, card,
						      attention_layers) ?
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
				       card->array_offset
					       [DS41F_ARRAY_FREQUENCIES] -
				       ds41f_image_offset_of(card),
			       words, sizeof(words));
			for (uint32_t layer = 0; layer < DS41F_WALK_LAYERS;
			     layer++) {
				if ((walked & DS41F_BIT(layer)) != 0)
					memcpy(card->image +
						       card->expert_table_offset
							       [layer] -
						       ds41f_image_offset_of(
							       card),
					       model->expert_tables[layer],
					       sizeof(model->expert_tables
							      [layer]));
				if ((targets & DS41F_BIT(layer)) == 0) {
					stage->external_pool_sources |=
						card->pool_offset[layer] !=
								PROGRAM_NOWHERE ?
							DS41F_BIT(layer) :
							0;
					stage->external_key_sources |=
						card->key_tiles_offset[layer] !=
								PROGRAM_NOWHERE ?
							DS41F_BIT(layer) :
							0;
					stage->external_selection_groups |=
						card->selection_offset[layer] !=
								PROGRAM_NOWHERE ?
							DS41F_BIT(layer) :
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
			    description->limits.row_capacity >
				    LINEAR_MAXIMUM_ROWS)
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
					ds41f_take(&router_next, bytes), bytes
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
