/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * ds41f_tp_program_checks.c: the TP2 program macros, TP program limits, layer masks, tensor and view validity (split
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

#include "models/deepseek_v41_flash/builder/ds41f_builder.h"
#include "ds41f_tp_program_internal.h"

static bool tp2_program_limits_equal(const struct program_limits *left,
				     const struct program_limits *right)
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

uint32_t ds41f_tp2_program_layer_card(
	const struct ds41f_tp2_program_description *description, uint32_t layer)
{
	return description->canonical->layer_card[layer];
}

static uint64_t tp2_program_layer_mask(uint32_t first, uint32_t count)
{
	return ((1ull << count) - 1u) << first;
}

static bool tp2_program_tensor_equal(const struct ds41f_tensor *left,
				     const struct ds41f_tensor *right)
{
	return left->stored.byte_offset == right->stored.byte_offset &&
	       left->stored.byte_count == right->stored.byte_count &&
	       left->shape.output_size == right->shape.output_size &&
	       left->shape.input_size == right->shape.input_size &&
	       left->shape.bits_per_weight == right->shape.bits_per_weight &&
	       left->shape.reserved_must_be_zero ==
		       right->shape.reserved_must_be_zero &&
	       left->weight_format == right->weight_format &&
	       left->card == right->card &&
	       left->fold_reference == right->fold_reference;
}

/* The new topology is admitted only through an explicit four-CMP, full-target description.
 * Runtime preparation retains the model's validated four-card geometry and qualified peer ownership. */
bool ds41f_tp2_program_exl3_all(
	const struct ds41f_tp2_program_description *description)
{
	return description->stages[0] != NULL &&
	       description->stages[0]->description.exl3_all_targets;
}

static bool tp2_program_exl3_layer_is_valid(
	const struct ds41f_tp2_program_description *description, uint32_t card,
	uint32_t layer)
{
	const struct ds41f_tp2_stage_plan *const stage =
		description->stages[card];
	const struct ds41f_plan *const canonical = description->canonical,
				       *const view = description->views[card];
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
	if (bb_exl3_output_shard_prepare(DS41F_HIDDEN, DS41F_EXPERT_INNER, bits,
					 DS41F_TP4_RANKS, card,
					 stage->description.exl3_product_order,
					 &expected) != DS41F_OK ||
	    memcmp(&expected, &stage->exl3_shards[layer], sizeof(expected)) !=
		    0)
		return false;
	return true;
}

/* replicated_embedding (ds41f_tp_model.h): each rank of the first stage embeds its rows itself. */
bool ds41f_tp2_program_replicated(
	const struct ds41f_tp2_program_description *description)
{
	return description->stages[0]->description.rank_count > 1u &&
	       description->stages[1] != NULL &&
	       description->stages[1]->description.include_embedding;
}

/* Original BATCH keeps target results, the round table and seven extra legacy verifier areas
 * behind the native tail's ordinary arena. Bound the complete downloads and every verifier before building. */
static bool tp2_program_original_batch_areas_are_valid(
	const struct ds41f_tp2_stage_plan *stage)
{
	const struct linear_range results = stage->batch_results,
				  rounds = stage->batch_rounds;
	const struct linear_range drafts = stage->batch_drafts;
	if (stage->description.card != DS41F_TP2_PRO_CARD)
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
		(uint64_t)DS41F_ORIGINAL_BATCH_SESSIONS * STEP_ROUND_BYTES;
	const uint64_t draft_bytes =
		(uint64_t)(DS41F_ORIGINAL_BATCH_SESSIONS - 1u) *
		ds41f_batch_draft_offset(DS41F_BATCH_DRAFT_PARTS,
					 DS41F_DRAFT_BLOCK);
	return results.byte_count == result_bytes &&
	       rounds.byte_count == round_bytes &&
	       drafts.byte_count == draft_bytes &&
	       results.byte_offset <= arena &&
	       result_bytes <= arena - results.byte_offset &&
	       rounds.byte_offset >= results.byte_offset &&
	       rounds.byte_offset - results.byte_offset ==
		       ds41f_batch_aligned(result_bytes) &&
	       rounds.byte_offset <= arena &&
	       round_bytes <= arena - rounds.byte_offset &&
	       rounds.byte_offset - results.byte_offset <=
		       STEP_BATCH_RESULT_SPAN_MOST - round_bytes &&
	       drafts.byte_offset >= rounds.byte_offset &&
	       drafts.byte_offset - rounds.byte_offset ==
		       ds41f_batch_aligned(round_bytes) &&
	       drafts.byte_offset <= arena &&
	       draft_bytes <= arena - drafts.byte_offset &&
	       stage->batch_logits.byte_count == 0 &&
	       stage->batch_rows.byte_count == 0;
}

static bool tp2_program_view_is_valid(
	const struct ds41f_tp2_program_description *description, uint32_t card,
	uint32_t rows)
{
	const struct ds41f_plan *const canonical = description->canonical,
				       *const view = description->views[card];
	const struct ds41f_tp2_stage_plan *const stage =
		description->stages[card];
	if (stage == NULL || stage->model == NULL ||
	    stage->description.rank > card ||
	    (stage->description.rank_count != 1u &&
	     stage->description.rank_count != DS41F_TP2_RANKS &&
	     stage->description.rank_count != DS41F_TP4_RANKS))
		return false;
	const uint32_t first = stage->description.first_target_layer,
		       count = stage->description.target_layer_count;
	const uint32_t ranks = stage->description.rank_count,
		       primary = card - stage->description.rank;
	const bool tensor = ranks > 1u;
	const bool exl3_all = ds41f_tp2_program_exl3_all(description);
	const bool original_batch =
		description->stages[0]->description.original_batch;
	if (stage->description.original_batch != original_batch ||
	    (original_batch &&
	     (!ds41f_original_batch_stage_is_valid(&stage->description) ||
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
			DS41F_EXPERTS_PER_ROW * DS41F_EXPERT_INNER *
			LINEAR_BFLOAT16_BYTES;
		const uint64_t sum_bytes =
			(uint64_t)canonical->limits.row_capacity *
			DS41F_HIDDEN * LINEAR_SUM_BYTES;
		if (ranks != DS41F_TP4_RANKS || first != 0 ||
		    count != DS41F_TP4_ATTENTION_PREFIX_END ||
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
	     (card != stage->description.rank || ranks != DS41F_TP4_RANKS ||
	      first != 0 || count != DS41F_LAYERS ||
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
	if ((card < DS41F_TP2_PRO_CARD &&
	     stage->description.attention_owner !=
		     description->stages[0]->description.attention_owner) ||
	    (stage->description.attention_owner &&
	     (!stage->description.exact_output_shards ||
	      ranks != DS41F_TP4_RANKS || first != 0 ||
	      count != DS41F_TP4_ATTENTION_PREFIX_END)))
		return false;
	if (first >= DS41F_LAYERS || count == 0 ||
	    count > DS41F_LAYERS - first || stage->description.rank >= ranks ||
	    (tensor && primary + ranks > DS41F_TP2_PRO_CARD))
		return false;
	for (uint32_t layer = first; layer < first + count; layer++) {
		if (canonical->layer_card[layer] != primary)
			return false;
	}
	if (view == NULL ||
	    !tp2_program_limits_equal(&canonical->limits, &view->limits) ||
	    view->card_count != canonical->card_count || view->vision.enabled ||
	    view->embedding_card != canonical->embedding_card ||
	    view->result_card != canonical->result_card ||
	    stage->description.card != card ||
	    stage->description.first_target_layer != first ||
	    stage->description.target_layer_count != count ||
	    stage->description.include_embedding !=
		    (card == 0 || ds41f_tp2_program_replicated(description)) ||
	    stage->description.include_head_and_draft !=
		    (card == canonical->result_card) ||
	    canonical->cards[card].layers !=
		    tp2_program_layer_mask(first, count) ||
	    stage->model->cards[card].arena_byte_count !=
		    canonical->cards[card].arena_byte_count ||
	    (tensor &&
	     (stage->channels == NULL ||
	      stage->channel_count != (count + (stage->description.draft_layers ?
							DS41F_DRAFT_STAGES :
							0u)) *
					      DS41F_TP2_SITES ||
	      stage->channel_rows < rows || stage->compact.byte_count == 0)) ||
	    (!tensor && stage->channel_count != 0))
		return false;
	for (uint32_t other = 0; other < canonical->card_count; other++) {
		const struct ds41f_card *const local = &view->cards[other],
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
	if (memcmp(view->pool_session_bytes, canonical->pool_session_bytes,
		   sizeof(view->pool_session_bytes)) != 0 ||
	    memcmp(view->key_session_bytes, canonical->key_session_bytes,
		   sizeof(view->key_session_bytes)) != 0 ||
	    memcmp(view->layer_card, canonical->layer_card,
		   sizeof(view->layer_card)) != 0)
		return false;
	for (uint32_t global = 0; global < DS41F_TENSOR_GLOBALS; global++) {
		/* head_shards: ranks 1-3 hold their own norm and head rows; replicated_embedding: every rank its
	     * embedding */
		const bool own = global == DS41F_TENSOR_EMBEDDING ?
					 stage->description.include_embedding :
					 stage->description.head_shards &&
						 card != canonical->result_card;
		if (!tp2_program_tensor_equal(
			    &view->globals[global],
			    own ? &stage->model->globals[global] :
				  &canonical->globals[global]) ||
		    (own && view->globals[global].card != card))
			return false;
	}
	for (uint32_t global = 0; global < DS41F_TENSOR_DRAFT_GLOBALS;
	     global++) {
		/* draft_shards: ranks 1-3 hold their own Markov tables; draft_layers: main_proj, main_norm, the final
	     * norm */
		const bool own =
			(((global == DS41F_TENSOR_DRAFT_MARKOV_EMBEDDING ||
			   global == DS41F_TENSOR_DRAFT_MARKOV_HEAD) &&
			  stage->description.draft_shards) ||
			 ((global == DS41F_TENSOR_DRAFT_MAIN_PROJECTION ||
			   global == DS41F_TENSOR_DRAFT_MAIN_NORM ||
			   global == DS41F_TENSOR_DRAFT_NORM) &&
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
		const struct ds41f_tensor *const tensors =
			view->layer_tensors[layer];
		if ((original_wide || original_batch) &&
		    (canonical->exl3_bits[layer] != 0 ||
		     view->exl3_bits[layer] != 0 ||
		     stage->model->exl3_bits[layer] != 0))
			return false;
		if (view->host_expert_count[layer] != 0 ||
		    tensors[DS41F_TENSOR_ATTENTION_NORM].card != card ||
		    tensors[DS41F_TENSOR_FFN_NORM].card != card)
			return false;
		for (uint32_t tensor_index = 0;
		     tensor_index < DS41F_TENSOR_LAYER_COUNT; tensor_index++) {
			if (tensors[tensor_index].stored.byte_count != 0 &&
			    tensors[tensor_index].card != card)
				return false;
		}
		if (exl3_all &&
		    !tp2_program_exl3_layer_is_valid(description, card, layer))
			return false;
		if (tensor) {
			if (stage->description.exact_output_shards !=
				    description->stages[0]
					    ->description.exact_output_shards ||
			    stage->description.exact_dense !=
				    description->stages[0]
					    ->description.exact_dense)
				return false;
			/* exact_dense: wq_b's quarter of the heads, wo_a's quarter of the groups, wo_b's quarter of the
			 * outputs,
             * each over the full inputs; the dense gathers' slots and banks */
			const bool ds41f_dense = stage->description.exact_dense;
			if (ds41f_dense &&
			    (stage->dense_input.byte_count <
				     (uint64_t)LINEAR_MAXIMUM_ROWS *
					     DS41F_WIDEST_ROW *
					     sizeof(uint16_t) ||
			     tensors[DS41F_TENSOR_OUTPUT_GROUPS]
					     .shape.output_size !=
				     DS41F_WIDEST_ROW / ranks ||
			     tensors[DS41F_TENSOR_OUTPUT].shape.output_size !=
				     DS41F_HIDDEN / ranks))
				return false;
			if (stage->description.exact_output_shards &&
			    (ranks != DS41F_TP4_RANKS ||
			     tensors[DS41F_TENSOR_QUERY_UP].shape.output_size !=
				     DS41F_HEADS_VALUE * DS41F_HEAD_DIMS /
					     (ds41f_dense ? ranks : 1u) ||
			     tensors[DS41F_TENSOR_OUTPUT].shape.input_size !=
				     DS41F_WIDEST_ROW ||
			     tensors[DS41F_TENSOR_SHARED_GATE]
					     .shape.output_size !=
				     DS41F_EXPERT_INNER ||
			     tensors[DS41F_TENSOR_SHARED_UP].shape.output_size !=
				     DS41F_EXPERT_INNER ||
			     tensors[DS41F_TENSOR_SHARED_DOWN].shape.input_size !=
				     DS41F_EXPERT_INNER))
				return false;
			const uint32_t channel =
				(layer - first) * DS41F_TP2_SITES;
			const uint32_t inner =
				ranks == DS41F_TP4_RANKS ?
					(stage->description.rank %
								 DS41F_TP2_RANKS ==
							 0 ?
						 DS41F_TP4_SMALL_INNER :
						 DS41F_TP4_LARGE_INNER) :
					DS41F_EXPERT_INNER / ranks;
			if (!stage->description.exact_output_shards &&
			    (tensors[DS41F_TENSOR_QUERY_UP].shape.output_size !=
				     DS41F_HEADS_VALUE * DS41F_HEAD_DIMS /
					     ranks ||
			     tensors[DS41F_TENSOR_OUTPUT].shape.input_size !=
				     DS41F_WIDEST_ROW / ranks ||
			     tensors[DS41F_TENSOR_SHARED_GATE]
					     .shape.output_size != inner ||
			     tensors[DS41F_TENSOR_SHARED_UP].shape.output_size !=
				     inner ||
			     tensors[DS41F_TENSOR_SHARED_DOWN]
					     .shape.input_size != inner))
				return false;
			for (uint32_t site = 0; site < DS41F_TP2_SITES;
			     site++) {
				const struct ds41f_tp2_stage_channel *const
					prepared =
						&stage->channels[channel + site];
				const struct ordered_push_arguments *const
					arguments =
						&prepared->collective.arguments;
				if (prepared->layer != layer ||
				    prepared->site != site ||
				    arguments->rank !=
					    stage->description.rank ||
				    arguments->rank_count != ranks ||
				    arguments->payload_format !=
					    stage->description.wire_format ||
				    arguments->payload_format >
					    ORDERED_PUSH_BF16 ||
				    arguments->payload_format !=
					    description->stages[0]
						    ->description.wire_format ||
				    arguments->step_offset !=
					    canonical->cards[card].array_offset
						    [DS41F_ARRAY_STEP])
					return false;
			}
		}
	}
	return true;
}

bool ds41f_tp2_program_is_valid(
	const struct ds41f_tp2_program_description *description,
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
	const struct ds41f_plan *const canonical = description->canonical;
	const struct program_limits *const limits = &canonical->limits;
	const bool batch = kind == PROGRAM_BATCH;
	const bool chunked = kind == PROGRAM_WIDE || kind == PROGRAM_FULL ||
			     batch;
	const bool exl3_all = ds41f_tp2_program_exl3_all(description);
	const bool original_batch =
		description->stages[0] != NULL &&
		description->stages[0]->description.original_batch;
	if ((batch && ((!exl3_all && !original_batch) ||
		       !description->stages[0]->description.batch)) ||
	    (original_batch &&
	     (limits->linear_profile != LINEAR_PROFILE_PRECISION ||
	      limits->linear_order != LINEAR_ORDER_REFERENCE ||
	      limits->cmp_order != CMP_ORDER_REFERENCE ||
	      limits->session_capacity != DS41F_ORIGINAL_BATCH_SESSIONS ||
	      limits->draft_positions != DS41F_DRAFT_BLOCK ||
	      (batch && (rows != DS41F_ORIGINAL_BATCH_ROWS ||
			 (description->batch_verifiers != 0 &&
			  description->batch_verifiers !=
				  DS41F_ORIGINAL_BATCH_SESSIONS))))) ||
	    canonical->card_count !=
		    (exl3_all ? DS41F_TP4_RANKS : DS41F_TP2_PROGRAM_CARDS) ||
	    canonical->result_card != (exl3_all ? 0u : DS41F_TP2_PRO_CARD))
		return false;
	for (uint32_t card = canonical->card_count;
	     card < DS41F_TP2_PROGRAM_CARDS; card++) {
		if (description->stages[card] != NULL ||
		    description->views[card] != NULL)
			return false;
	}
	/* thin_tail (ds41f_tp_stage.h): full and wide programs of 9 to 63 rows too */
	const uint32_t least_chunk =
		!batch && chunked &&
				description->stages[0]->description.thin_tail ?
			LINEAR_MAXIMUM_ROWS + 1u :
			STEP_WIDE_LEAST_ROWS;
	if ((!chunked && kind != PROGRAM_THIN) ||
	    (batch ? rows > PROGRAM_BATCH_MOST_ROWS : /* a BATCH program of 8 rows or fewer walks as THIN */
		     chunked ? rows < least_chunk :
			       rows > LINEAR_MAXIMUM_ROWS) ||
	    canonical->embedding_card != 0 || canonical->vision.enabled ||
	    rows == 0 || rows > limits->row_capacity ||
	    rows > DS41F_TP_SPLIT_MOST_ROWS || limits->row_capacity == 0 ||
	    limits->row_capacity > DS41F_TP_SPLIT_MOST_ROWS ||
	    limits->session_capacity == 0 ||
	    limits->context_token_limit < DS41F_CONTEXT_WITHOUT_CANDIDATES ||
	    limits->context_token_limit > DS41F_CONTEXT_MOST ||
	    limits->context_token_limit % DS41F_CONTEXT_GRANULE != 0 ||
	    limits->official_hc_rows != 0 ||
	    limits->state_width != STATE_WIDTH_16 ||
	    limits->cmp_order != CMP_ORDER_REFERENCE ||
	    (limits->linear_profile != LINEAR_PROFILE_PRECISION &&
	     limits->linear_profile != LINEAR_PROFILE_SPEED) ||
	    (limits->linear_order != LINEAR_ORDER_FIXED &&
	     limits->linear_order != LINEAR_ORDER_REFERENCE) ||
	    (limits->draft_positions != 0 &&
	     (limits->draft_positions < DS41F_DRAFT_BLOCK ||
	      limits->draft_positions > STEP_DRAFT_MOST ||
	      limits->draft_positions > limits->row_capacity)))
		return false;
	const uint32_t walk_count =
		limits->draft_positions != 0 ? DS41F_WALK_LAYERS : DS41F_LAYERS;
	for (uint32_t layer = 0; layer < walk_count; layer++) {
		const uint32_t owner =
			ds41f_tp2_program_layer_card(description, layer);
		if (owner >= canonical->card_count ||
		    description->stages[owner] == NULL ||
		    description->stages[owner]->description.rank != 0 ||
		    (ds41f_walk_layers(&canonical->cards[owner]) >> layer &
		     1u) == 0 ||
		    canonical->host_expert_count[layer] != 0)
			return false;
	}
	for (uint32_t card = 0; card < canonical->card_count; card++) {
		const uint32_t arithmetic =
			card_descriptions[card].arithmetic_class;
		if (cards[card] == NULL ||
		    (original_batch && card == DS41F_TP2_PRO_CARD &&
		     arithmetic != ARITHMETIC_CLASS_SM_120_NATIVE) ||
		    (exl3_all || card < DS41F_TP2_PRO_CARD ?
			     arithmetic != ARITHMETIC_CLASS_SM_80 :
			     arithmetic != ARITHMETIC_CLASS_SM_120 &&
				     arithmetic !=
					     ARITHMETIC_CLASS_SM_120_NATIVE) ||
		    !tp2_program_view_is_valid(description, card, rows)) {
			LOG_ERROR(
				"tensor program: invalid card metadata at card %u",
				card);
			return false;
		}
	}
	return true;
}
