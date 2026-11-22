/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * ds41f_tp_layer_plan.c: the TP2 layer plans (split from model_v41.c by whole items).
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

int ds41f_tp2_pending_compare(const void *first, const void *second)
{
	const struct ds41f_tp2_pending_tensor *const left = first,
						     *const right = second;
	if (left->extent != right->extent)
		return left->extent < right->extent ? -1 : 1;
	return left->job.tensor_index < right->job.tensor_index ? -1 :
	       left->job.tensor_index > right->job.tensor_index ? 1 :
								  0;
}

MUST_CHECK result_status ds41f_tp2_tensor_find(
	const struct pack_record *pack, const char *name,
	const struct ds41f_tensor_rule *rule, uint32_t card, uint32_t rank,
	uint32_t rank_count, bool split_outputs, bool split_inputs,
	bool balanced_shards, struct ds41f_tensor *tensor_out,
	struct ds41f_tp2_pending_tensor *pending)
{
	const uint32_t found =
		pack_find_tensor(pack, name, (uint32_t)strlen(name));
	const struct pack_tensor *const source =
		found == PACK_NOT_FOUND ? NULL : &pack_tensors_of(pack)[found];
	if (source == NULL || source->element_type != rule->element_type ||
	    source->dimension_count !=
		    (rule->dimensions[1] == 0 ? 1u : DS41F_MATRIX_DIMENSIONS) ||
	    source->dimensions[0] != rule->dimensions[0] ||
	    source->dimensions[1] != rule->dimensions[1] ||
	    source->weight_format != rule->weight_format ||
	    source->bits_per_weight != rule->bits_per_weight)
		return STATUS_CORRUPT_PACK;
	*pending = (struct ds41f_tp2_pending_tensor){
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
					(DS41F_TP4_SMALL_INNER +
					 DS41F_TP4_LARGE_INNER) +
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
	result_status status = ds41f_model_check(pack);
	status = status == STATUS_OK ? ds41f_limits_check(limits) : status;
	struct ds41f_tp2_layer_plan *const result =
		status == STATUS_OK ? calloc(1, sizeof(*result)) : NULL;
	struct ds41f_tp2_pending_tensor *const pending =
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
			(uint32_t)(DS41F_BIT(DS41F_HOLDER_TRUNK) |
				   DS41F_BIT(DS41F_HOLDER_LAYER));
	}
	for (uint32_t tensor = 0;
	     status == STATUS_OK && tensor < DS41F_TENSOR_LAYER_COUNT;
	     tensor++) {
		if ((ds41f_layer_rules[tensor].layers & 1u) == 0)
			continue;
		char name[DS41F_NAME_CAPACITY];
		const int length = snprintf(name, sizeof(name), "layers.0.%s",
					    ds41f_layer_rules[tensor].name);
		if (length <= 0 || length >= DS41F_NAME_CAPACITY) {
			status = STATUS_INVALID_ARGUMENT;
			break;
		}
		const bool split_outputs =
			(!replicated_attention &&
			 (tensor == DS41F_TENSOR_QUERY_UP ||
			  tensor == DS41F_TENSOR_OUTPUT_GROUPS)) ||
			tensor == DS41F_TENSOR_SHARED_GATE ||
			tensor == DS41F_TENSOR_SHARED_UP;
		const bool split_inputs = (!replicated_attention &&
					   tensor == DS41F_TENSOR_OUTPUT) ||
					  tensor == DS41F_TENSOR_SHARED_DOWN;
		status = ds41f_tp2_tensor_find(
			pack, name, &ds41f_layer_rules[tensor], card_index,
			rank, rank_count, split_outputs, split_inputs, false,
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
			char name[DS41F_NAME_CAPACITY];
			const int length =
				snprintf(name, sizeof(name),
					 "layers.0.ffn.experts.%u.%s", expert,
					 ds41f_expert_rules[matrix].name);
			if (length <= 0 || length >= DS41F_NAME_CAPACITY) {
				status = STATUS_INVALID_ARGUMENT;
				break;
			}
			struct ds41f_tp2_pending_tensor *const item =
				&pending[result->job_count];
			status = ds41f_tp2_tensor_find(
				pack, name, &ds41f_expert_rules[matrix],
				card_index, rank, rank_count,
				!partitioned_experts &&
					matrix != DS41F_TENSOR_DOWN,
				!partitioned_experts &&
					matrix == DS41F_TENSOR_DOWN,
				false, NULL, item);
			item->expert_offset =
				&model->expert_tables[0][expert][matrix];
			result->job_count += status == STATUS_OK ? 1u : 0u;
		}
	}
	uint64_t next = arena_prefix_bytes, first_expert = UINT64_MAX,
		 last_expert = 0;
	if (status == STATUS_OK)
		qsort(pending, result->job_count, sizeof(*pending),
		      ds41f_tp2_pending_compare);
	for (uint32_t index = 0;
	     status == STATUS_OK && index < result->job_count; index++) {
		struct ds41f_tp2_pending_tensor *const item = &pending[index];
		item->job.destination_offset =
			ds41f_take(&next, item->byte_count);
		result->jobs[index] = item->job;
		if (item->stored != NULL)
			*item->stored = (struct linear_range){
				item->job.destination_offset, item->byte_count
			};
		if (item->expert_offset != NULL) {
			*item->expert_offset = item->job.destination_offset;
			first_expert = item->job.destination_offset <
						       first_expert ?
					       item->job.destination_offset :
					       first_expert;
			const uint64_t end = ds41f_sum(
				item->job.destination_offset, item->byte_count);
			last_expert = end > last_expert ? end : last_expert;
		}
		status = next == UINT64_MAX ? STATUS_INVALID_ARGUMENT :
					      STATUS_OK;
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
		status = ds41f_card_lay_out(model, card) ?
				 STATUS_OK :
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
				       card->array_offset
					       [DS41F_ARRAY_FREQUENCIES] -
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
			result->compact = (struct linear_range){
				ds41f_take(&next, compact_bytes), compact_bytes
			};
			for (uint32_t site = 0; site < DS41F_TP2_SITES;
			     site++) {
				const uint64_t values =
					(uint64_t)limits->row_capacity *
					DS41F_HIDDEN;
				const uint64_t bytes = values * sizeof(float);
				const uint64_t signal_bytes =
					(uint64_t)ORDERED_PUSH_PHASES *
					rank_count *
					ORDERED_COLLECTIVE_LINE_BYTES;
				uint64_t stride =
					signal_bytes +
					(rank_count + DS41F_TP2_RANKS) * bytes;
				const uint64_t table = ds41f_take(
					&next,
					(uint64_t)rank_count *
						ORDERED_COLLECTIVE_PEER_BYTES);
				stride = (stride + DS41F_ARRAY_ALIGNMENT_BYTES -
					  1u) /
					 DS41F_ARRAY_ALIGNMENT_BYTES *
					 DS41F_ARRAY_ALIGNMENT_BYTES;
				const uint64_t region = ds41f_take(
					&next,
					stride * ORDERED_COLLECTIVE_SLOTS);
				result->collectives[site].arguments =
					(struct ordered_push_arguments){
						.peer_table_offset = table,
						.region_offset = region,
						.input_offset = signal_bytes,
						.output_offset =
							signal_bytes + bytes,
						.inbox_offset =
							signal_bytes +
							DS41F_TP2_RANKS * bytes,
						.step_offset =
							card->array_offset
								[DS41F_ARRAY_STEP],
						.parity_stride = stride,
						.deadline_cycles =
							DS41F_TP2_DEADLINE_CYCLES,
						.inbox_rank_stride = bytes,
						.element_count =
							(uint32_t)values,
						.rank = rank,
						.rank_count = rank_count,
						.payload_format =
							ORDERED_PUSH_FP32
					};
			}
			for (uint32_t site = 0; site < DS41F_TP2_SITES;
			     site++) {
				const uint64_t bytes =
					(uint64_t)limits->row_capacity *
					DS41F_HIDDEN * sizeof(float);
				result->stub_values[site] =
					(struct linear_range){
						ds41f_take(&next, bytes), bytes
					};
			}
			card->arena_byte_count = next;
			for (uint32_t site = 0; site < DS41F_TP2_SITES; site++)
				result->collectives[site].arena_byte_count =
					next;
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

result_status ds41f_tp2_layer_variant_create(
	const struct pack_record *pack, const struct program_limits *limits,
	uint32_t card_index, uint32_t rank, uint32_t rank_count,
	uint64_t arena_prefix_bytes, bool replicated_attention,
	struct ds41f_tp2_layer_plan **plan_out)
{
	return ds41f_tp2_layer_choices_create(pack, limits, card_index, rank,
					      rank_count, arena_prefix_bytes,
					      replicated_attention, false,
					      plan_out);
}

result_status ds41f_tp2_layer_plan_create(
	const struct pack_record *pack, const struct program_limits *limits,
	uint32_t card_index, uint32_t rank, uint32_t rank_count,
	uint64_t arena_prefix_bytes, struct ds41f_tp2_layer_plan **plan_out)
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
