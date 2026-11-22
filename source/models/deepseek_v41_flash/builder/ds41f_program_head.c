/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * ds41f_program_head.c: head cut, sampler shards, draft cut, wide head, BATCH heads, verifiers and drafts (split from
 * model_v41_program.c by whole items).
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

#include "engine/library/bf16_linear/head_mma_records.h"
#include "ds41f_builder.h"
#include "ds41f_program_internal.h"

#define DRAFT_WIDE_CORES \
	3u /* batch_draft_wide: and its window writes and core once, each verifier's block whole */

/* head_shards: the first row of the card's cut in its copy of the head: 0 where it holds only its rows. */
static uint32_t head_cut_first(const struct ds41f_builder *b)
{
	const uint32_t shard = DS41F_VOCABULARY_VALUE / TP_GROUP_MOST_RANKS;
	return b->plan->globals[DS41F_TENSOR_HEAD].shape.output_size == shard ?
		       0u :
		       b->head_stage->description.rank * shard;
}

/* A head product: the head's rows [first, first + outputs) of the card's copy (bfloat16, no row scale) of the rows
 * (bfloat16 [b->rows][5,120]) into float32 [rows][outputs], the rows gated by the step area; by dense_stream's kernel
 * where head_stream prepared it (the same record and bits, dense_stream_records.h). */
MUST_CHECK static result_status head_product(struct ds41f_builder *b,
					     uint32_t first, uint32_t outputs,
					     struct linear_range rows,
					     struct linear_range step,
					     struct linear_range output)
{
	const struct ds41f_tensor *const head =
		&b->plan->globals[DS41F_TENSOR_HEAD];
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
		DS41F_ELEMENT_BF16,
		DS41F_ELEMENT_F32,
		DENSE_STEP_ROWS,
		1.0f,
		0.0f,
		0
	};
	struct kernel_launch *const launch = ds41f_slots(b, 1);
	if (b->plan->limits.official_reference != 0) {
		const struct ds41f_official_head_description source = {
			description.weights, rows, output, outputs, 1
		};
		return ds41f_official_head_prepare(&source, b->arena,
						   b->official.head, launch);
	}
	const result_status status =
		dense_linear_prepare(&description, b->arena, &b->dense, launch);
	if (status == DS41F_OK && b->head_stream_kernel != BACKEND_NO_KERNEL) {
		const uint32_t groups = outputs / DENSE_OUTPUTS_PER_BLOCK;
		launch->kernel = b->head_stream_kernel;
		launch->grid[0] = groups < DENSE_STREAM_BLOCKS ?
					  groups :
					  DENSE_STREAM_BLOCKS;
	}
	if (status == DS41F_OK && b->head_mma_kernel != BACKEND_NO_KERNEL &&
	    b->rows <= LINEAR_MAXIMUM_ROWS && b->rows >= HEAD_MMA_LEAST_ROWS &&
	    head->shape.input_size % HEAD_MMA_STEP_INPUTS == 0) {
		/* head_mma: a block a 128 outputs, 256 threads, the same record */
		const uint32_t block_outputs =
			HEAD_MMA_BLOCK_OUTPUTS / HEAD_MMA_SPLIT_8;
		launch->kernel = b->head_mma_kernel;
		launch->grid[0] =
			(outputs + block_outputs - 1u) / block_outputs;
		launch->grid[1] = launch->grid[LAUNCH_DIMENSIONS - 1u] = 1;
		launch->block[0] = HEAD_MMA_THREADS;
		launch->block[1] = launch->block[LAUNCH_DIMENSIONS - 1u] = 1;
		launch->dynamic_shared_byte_count = 0;
	}
	return status;
}

/* head_shards, the result card at the start of a step that samples: its session settings to every rank's settings
 * copy (sampler_shard_records.h), so that no sampler shard reads another card's memory. */
MUST_CHECK result_status ds41f_settings_push_launch(struct ds41f_builder *b)
{
	const struct ds41f_tp2_stage_plan *const stage = b->head_stage;
	const uint32_t sessions = b->plan->limits.session_capacity;
	const struct sampler_settings_push_arguments arguments = {
		DS41F_ARRAY(STEP).byte_offset,
		DS41F_ARRAY(SESSIONS).byte_offset,
		stage->sampler_settings_peers.byte_offset, sessions, 0
	};
	if (stage->sampler_settings_peers.byte_count <
		    (uint64_t)TP_GROUP_MOST_RANKS * sizeof(uint64_t) ||
	    DS41F_ARRAY(SESSIONS).byte_offset == PROGRAM_NOWHERE ||
	    STEP_SESSION_BYTES != SAMPLER_SETTINGS_SLOT_BYTES)
		return STATUS_INVALID_ARGUMENT;
	struct kernel_launch launch;
	prepare_launch_fill(&launch, b->settings_push_kernel, 1, 1,
			    SAMPLER_SETTINGS_PUSH_THREADS, 0, &arguments,
			    sizeof(arguments));
	return ds41f_put(b, &launch);
}

/* head_shards: the card's sampler shard of a step's use `call` (sampler_shard_records.h) over the rows of step, the
 * logits [rows][V/4] at logits and the result area at result. */
MUST_CHECK static result_status
sampler_shard_launch(struct ds41f_builder *b, struct linear_range step,
		     uint32_t rows, struct linear_range logits,
		     struct linear_range result, uint32_t call)
{
	const struct ds41f_tp2_stage_plan *const stage = b->head_stage;
	const uint32_t rank = stage->description.rank,
		       shard = DS41F_VOCABULARY_VALUE / TP_GROUP_MOST_RANKS;
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
		DS41F_VOCABULARY_VALUE,
		b->plan->limits.session_capacity,
		rank,
		call
	};
	if (rows == 0 || rows > LINEAR_MAXIMUM_ROWS ||
	    rank >= TP_GROUP_MOST_RANKS || call >= SAMPLER_SHARD_CALLS ||
	    step.byte_count < STEP_BLOCK_BYTES_VALUE +
				      (uint64_t)rows * STEP_ROW_BYTES_VALUE ||
	    logits.byte_count < (uint64_t)rows * shard * sizeof(float) ||
	    result.byte_count <
		    STEP_RESULT_BYTES +
			    (uint64_t)rows * STEP_ROW_RESULT_BYTES ||
	    stage->sampler_exchange.byte_count <
		    (uint64_t)SAMPLER_SHARD_DEFAULT_EXCHANGE_BYTES ||
	    stage->sampler_entry.byte_count <
		    (uint64_t)SAMPLER_SHARD_ENTRY_BYTES ||
	    stage->sampler_peers.byte_count < SAMPLER_SHARD_PEER_TABLE_BYTES)
		return STATUS_INVALID_ARGUMENT;
	struct kernel_launch launch;
	prepare_launch_fill(&launch, b->sampler_shard_kernel, 1, rows,
			    SAMPLER_THREADS, 0, &arguments, sizeof(arguments));
	return ds41f_put(b, &launch);
}

/* draft_shards, on every rank: the draft head's cut over the draft's rows, then each position j's Markov bias over
 * the cut (W2's rows of the cut, the whole W1) and sampler shard (use first_call + j), into the card's own logits
 * [positions][V/4], results and Markov rows. */
MUST_CHECK static result_status
draft_cut_launches(struct ds41f_builder *b, struct linear_range rows,
		   struct linear_range draft_step, struct linear_range views,
		   struct linear_range logits, struct linear_range results,
		   struct linear_range markov, uint32_t first_call)
{
	/* the cut is the head stage's rank of the draft: draft_layers and draft_shards both hold head_shards
     * (ds41f_tp2_stage_plan_create, the tp2 model), so tp2_program_select gave the builder its head
     * stage */
	if (b->head_stage == NULL)
		return STATUS_INVALID_ARGUMENT;
	const uint32_t positions = b->plan->limits.draft_positions,
		       rank = b->head_stage->description.rank;
	const uint32_t shard = DS41F_VOCABULARY_VALUE / TP_GROUP_MOST_RANKS;
	const struct ds41f_tensor *const embedding =
		&b->plan->draft_globals[DS41F_TENSOR_DRAFT_MARKOV_EMBEDDING];
	const struct ds41f_tensor *const head =
		&b->plan->draft_globals[DS41F_TENSOR_DRAFT_MARKOV_HEAD];
	const uint64_t skipped =
		(uint64_t)rank * shard * DS41F_DRAFT_MARKOV_ROW_BYTES;
	const uint64_t cut = (uint64_t)shard * DS41F_DRAFT_MARKOV_ROW_BYTES;
	if (skipped > head->stored.byte_count ||
	    cut > head->stored.byte_count - skipped)
		return STATUS_INVALID_ARGUMENT;
	const uint32_t held = b->rows;
	b->rows = positions;
	result_status status = head_product(b, head_cut_first(b), shard, rows,
					    draft_step, logits);
	b->rows = held;
	for (uint32_t position = 0; position < positions && status == DS41F_OK;
	     position++) {
		const struct ds41f_draft_markov_bias_description bias = {
			draft_step,
			results,
			embedding->stored,
			{ head->stored.byte_offset + skipped, cut },
			logits,
			markov,
			position,
			positions,
			shard,
			DS41F_VOCABULARY_VALUE,
			0
		};
		status = ds41f_draft_markov_bias_prepare(
			&bias, b->arena, &b->draft, ds41f_slots(b, 1));
		status = DS41F_THEN(sampler_shard_launch(
			b,
			(struct linear_range){
				views.byte_offset +
					(uint64_t)position *
						STEP_DRAFT_VIEW_BYTES,
				STEP_DRAFT_VIEW_BYTES },
			1u,
			(struct linear_range){
				logits.byte_offset + (uint64_t)position *
							     shard *
							     sizeof(float),
				(uint64_t)shard * sizeof(float) },
			(struct linear_range){
				results.byte_offset +
					(uint64_t)position *
						STEP_DRAFT_RESULT_BYTES,
				STEP_DRAFT_RESULT_BYTES },
			first_call + position));
	}
	return status;
}

/* batch_wide_head: the head's rows [first, first + outputs) of the card's copy over `rows` rows of hidden (bfloat16
 * [rows][5,120]) into float32 [rows][outputs] at output, one launch of dense_stream's wide form
 * (dense_stream_records.h: the bits of the thin launches); flags DENSE_STEP_ROWS gates the rows by the step area at
 * step, 0 stores all. */
MUST_CHECK static result_status
head_wide_product(struct ds41f_builder *b, uint32_t first, uint32_t outputs,
		  uint32_t rows, struct linear_range hidden, uint64_t step,
		  uint32_t flags, struct linear_range output, bool thin_split)
{
	const struct ds41f_tensor *const head =
		&b->plan->globals[DS41F_TENSOR_HEAD];
	const uint64_t row_bytes =
		(uint64_t)head->shape.input_size * sizeof(uint16_t);
	if (rows == 0 || rows > DENSE_STREAM_WIDE_ROWS ||
	    (uint64_t)first + outputs > head->shape.output_size ||
	    ((uint64_t)first + outputs) * row_bytes > head->stored.byte_count ||
	    (uint64_t)rows * DS41F_HIDDEN_ROW_BYTES > hidden.byte_count ||
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
		.output_type = DS41F_ELEMENT_F32,
		.flags = flags,
		.output_scale = 1.0f,
		.epsilon = 0.0f,
		.cone_layer = 0
	};
	const uint32_t groups = outputs / DENSE_OUTPUTS_PER_BLOCK;
	struct kernel_launch launch;
	/* head_mma: the form head02 measured fastest for the rows (a split of N: blocks of 128 / N outputs);
     * thin_split: the 8-row form's split (each row's sums are the warps' of its split, added in warp order: the thin
     * bits) */
	const uint32_t wide_form =
		thin_split			   ? HEAD_MMA_WIDE_THIN_FORM :
		rows <= HEAD_MMA_WIDE_S1_MOST_ROWS ? 0u :
		rows <= HEAD_MMA_WIDE_S2_MOST_ROWS ? 1u :
						     HEAD_MMA_WIDE_S4_FORM;
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
	return ds41f_put(b, &launch);
}

/* batch_wide_head, a BATCH program's sampling card: each slab's final collapse, then the head (or the card's cut) of
 * every row in one launch of dense_stream's wide form into batch_logits [rows][outputs], then each slab's sampler (or
 * sampler shard, use = the slab) into the slab's result area. */
MUST_CHECK result_status ds41f_batch_wide_head_launches(
	struct ds41f_builder *b, const struct ds41f_tp2_stage_plan *stage)
{
	const bool shards = b->head_stage != NULL;
	const uint32_t outputs =
		shards ? DS41F_VOCABULARY_VALUE / TP_GROUP_MOST_RANKS :
			 DS41F_VOCABULARY_VALUE;
	const uint64_t sessions = b->plan->limits.session_capacity;
	result_status status = DS41F_OK;
	for (uint32_t slab = 0; slab < b->slab_count && status == DS41F_OK;
	     slab++) {
		ds41f_batch_slab_select(b, slab);
		const struct ds41f_hyper_final_description final = {
			DS41F_ARRAY(STREAMS),
			DS41F_ARRAY(CARRIED_PRE),
			b->plan->globals[DS41F_TENSOR_NORM].stored,
			DS41F_ARRAY(HIDDEN),
			{ 0, 0 },
			b->rows,
			0
		};
		status = ds41f_hyper_final_prepare(&final, b->arena, &b->hyper,
						   ds41f_slots(b, 1));
	}
	ds41f_chunk_select(b);
	const uint64_t logits_row_bytes = (uint64_t)outputs * sizeof(float);
	status = DS41F_THEN(head_wide_product(
		b, shards ? head_cut_first(b) : 0u, outputs, b->rows,
		DS41F_ARRAY(HIDDEN), DS41F_ARRAY(STEP).byte_offset,
		DENSE_STEP_ROWS, stage->batch_logits,
		ds41f_batch_thin_bits_of(b)));
	for (uint32_t slab = 0; slab < b->slab_count && status == DS41F_OK;
	     slab++) {
		ds41f_batch_slab_select(b, slab);
		const struct linear_range logits = {
			stage->batch_logits.byte_offset +
				(uint64_t)slab * LINEAR_MAXIMUM_ROWS *
					logits_row_bytes,
			(uint64_t)b->rows * logits_row_bytes
		};
		const struct linear_range result = {
			stage->batch_results.byte_offset +
				(uint64_t)slab *
					PROGRAM_BATCH_SLAB_RESULT_BYTES,
			PROGRAM_BATCH_SLAB_RESULT_BYTES
		};
		if (shards) {
			status = sampler_shard_launch(b, DS41F_ARRAY(STEP),
						      b->rows, logits, result,
						      slab);
		} else {
			const struct sampler_description sampler = {
				DS41F_ARRAY(STEP),
				{ DS41F_ARRAY(SESSIONS).byte_offset,
				  sessions * STEP_SESSION_BYTES },
				logits,
				DS41F_ARRAY(SAMPLER_WEIGHTS),
				result,
				DS41F_VOCABULARY_VALUE,
				(uint32_t)sessions,
				b->rows,
				0
			};
			status = sampler_prepare_entry(&sampler, b->arena,
						       &b->sampler,
						       ds41f_slots(b, 1));
		}
	}
	ds41f_chunk_select(b);
	return status;
}

/* The final collapse, the head and the sampler (REF:1268-1269, 1008-1017, 1285-1292). head_shards
 * (ds41f_tp_stage.h), on every rank: its own collapse, its rows of the head into its logits [rows][V/4], and the
 * sharded sampler into its result area, the same answers on every rank. */
MUST_CHECK result_status ds41f_head_launches(struct ds41f_builder *b)
{
	const uint64_t sessions = b->plan->limits.session_capacity;
	const struct linear_range collapse =
		b->plan->limits.official_reference != 0 ?
			(struct linear_range){
				b->plan->cards[b->card].official_scratch_offset,
				DS41F_HIDDEN_ROW_BYTES
			} :
			ds41f_no_range;
	const struct ds41f_hyper_final_description final = {
		DS41F_ARRAY(STREAMS),
		DS41F_ARRAY(CARRIED_PRE),
		b->plan->globals[DS41F_TENSOR_NORM].stored,
		DS41F_ARRAY(HIDDEN),
		collapse,
		b->rows,
		0
	};
	const struct sampler_description sampler = {
		DS41F_ARRAY(STEP),
		{ DS41F_ARRAY(SESSIONS).byte_offset,
		  sessions * STEP_SESSION_BYTES },
		DS41F_ARRAY(LOGITS),
		DS41F_ARRAY(SAMPLER_WEIGHTS),
		DS41F_ARRAY(RESULT),
		DS41F_VOCABULARY_VALUE,
		(uint32_t)sessions,
		b->rows,
		0
	};
	result_status status = ds41f_hyper_final_prepare(
		&final, b->arena, &b->hyper, ds41f_slots(b, 1));
	if (b->plan->limits.official_reference != 0)
		status = DS41F_THEN(
			ds41f_official_norm(b, collapse, final.norm_weight,
					    DS41F_ARRAY(HIDDEN), DS41F_HIDDEN));
	if (b->head_stage == NULL) {
		status = DS41F_THEN(head_product(
			b, 0, DS41F_VOCABULARY_VALUE, DS41F_ARRAY(HIDDEN),
			DS41F_ARRAY(STEP), DS41F_ARRAY(LOGITS)));
		return DS41F_THEN(sampler_prepare_entry(
			&sampler, b->arena, &b->sampler, ds41f_slots(b, 1)));
	}
	const uint32_t shard = DS41F_VOCABULARY_VALUE / TP_GROUP_MOST_RANKS;
	status = DS41F_THEN(head_product(b, head_cut_first(b), shard,
					 DS41F_ARRAY(HIDDEN), DS41F_ARRAY(STEP),
					 DS41F_ARRAY(LOGITS)));
	status = DS41F_THEN(sampler_shard_launch(
		b, DS41F_ARRAY(STEP), b->rows, DS41F_ARRAY(LOGITS),
		DS41F_ARRAY(RESULT), b->sampler_call));
	return b->card == b->plan->result_card ||
			       !b->head_stage->description.draft_shards ||
			       b->draft_layers ||
			       b->made->program.kind != PROGRAM_THIN ?
		       status :
		       DS41F_THEN(ds41f_draft_shard_rank_launches(b, 0, 1));
}

/* Original PP5: PRO alone normalizes, four independent late CMP visits produce disjoint vocabulary cuts,
 * then PRO's unchanged whole sampler and draft consume the gathered logits. Tiny ordinary hand-offs carry
 * readiness only; their recorded marks cover the masked peer stores. No future producer wait is submitted. */
MUST_CHECK result_status ds41f_original_head_walk(struct ds41f_builder *b)
{
	const struct ds41f_plan *const plan = b->plan;
	const struct ds41f_original_head *const head = &plan->original_head;
	const uint32_t pro = plan->result_card;
	uint32_t first = b->made->cards[pro].launch_count;
	const struct ds41f_hyper_final_description final = {
		DS41F_ARRAY(STREAMS),
		DS41F_ARRAY(CARRIED_PRE),
		plan->globals[DS41F_TENSOR_NORM].stored,
		DS41F_ARRAY(HIDDEN),
		ds41f_no_range,
		b->rows,
		0
	};
	result_status status = ds41f_hyper_final_prepare(
		&final, b->arena, &b->hyper, ds41f_slots(b, 1));
	for (uint32_t rank = 0; rank < DS41F_HEAD_RANKS && status == DS41F_OK;
	     rank++)
		status = ds41f_original_head_launch(plan, b->backend, b->cards,
						    DS41F_HEAD_BROADCAST, rank,
						    0, b->rows,
						    ds41f_slots(b, 1));
	ds41f_segment_add(b, SEGMENT_NO_LAYER, first);
	for (uint32_t rank = 0; rank < DS41F_HEAD_RANKS; rank++)
		ds41f_hand_off(b, rank,
			       head->cards[pro].control_source.byte_offset,
			       head->cards[rank].control_input[0].byte_offset,
			       DS41F_HEAD_CONTROL_ROW_BYTES, 0, 0);
	for (uint32_t rank = 0; rank < DS41F_HEAD_RANKS && status == DS41F_OK;
	     rank++) {
		b->stage++;
		status = ds41f_card_select(b, rank);
		if (status != DS41F_OK)
			break;
		first = b->made->cards[rank].launch_count;
		status = ds41f_original_head_launch(plan, b->backend, b->cards,
						    DS41F_HEAD_DENSE, rank, 0,
						    b->rows, ds41f_slots(b, 1));
		status = DS41F_THEN(ds41f_original_head_launch(
			plan, b->backend, b->cards, DS41F_HEAD_GATHER, rank, 0,
			b->rows, ds41f_slots(b, 1)));
		ds41f_segment_add(b, SEGMENT_NO_LAYER, first);
		ds41f_hand_off(b, pro,
			       head->cards[rank].control_source.byte_offset,
			       head->cards[pro].control_input[rank].byte_offset,
			       DS41F_HEAD_CONTROL_ROW_BYTES, 0, 0);
	}
	if (status != DS41F_OK)
		return status;
	b->stage++;
	status = ds41f_card_select(b, pro);
	if (status != DS41F_OK)
		return status;
	first = b->made->cards[pro].launch_count;
	status = ds41f_original_head_launch(plan, b->backend, b->cards,
					    DS41F_HEAD_SAMPLE, 0, 0, b->rows,
					    ds41f_slots(b, 1));
	ds41f_segment_add(b, SEGMENT_NO_LAYER, first);
	return status;
}

/* draft_shards, a rank other than the result card's, verifier v's draft (a thin step's: 0): in a step that holds its
 * DRAFTS row, the result card's rows of it; then its cut draft, sampler uses from first_call. */
MUST_CHECK result_status ds41f_draft_shard_rank_launches(
	struct ds41f_builder *b, uint32_t verifier, uint32_t first_call)
{
	const uint32_t shard = DS41F_VOCABULARY_VALUE / TP_GROUP_MOST_RANKS;
	const uint32_t positions = b->plan->limits.draft_positions;
	const uint64_t area = b->head_stage->draft_area.byte_offset;
	const uint64_t logits = area + DS41F_DRAFT_SHARE_AREA_BYTES;
	const uint64_t results =
		logits + (uint64_t)STEP_DRAFT_MOST * shard * sizeof(float);
	const uint64_t markov =
		results + (uint64_t)STEP_DRAFT_MOST * STEP_DRAFT_RESULT_BYTES;
	const struct ds41f_draft_share_wait_arguments wait = {
		area, b->plan->cards[b->card].array_offset[DS41F_ARRAY_STEP],
		DS41F_DRAFT_SHARE_DEADLINE_CYCLES, positions, verifier
	};
	if (positions == 0 || positions > STEP_DRAFT_MOST ||
	    b->head_stage->draft_area.byte_count <
		    markov - area +
			    (uint64_t)STEP_DRAFT_MOST *
				    DS41F_DRAFT_MARKOV_ROW_BYTES)
		return STATUS_INVALID_ARGUMENT;
	struct kernel_launch launch;
	prepare_launch_fill(&launch, b->draft_share_kernel, 1, 1,
			    DS41F_DRAFT_SHARE_WAIT_THREADS, 0, &wait,
			    sizeof(wait));
	const result_status status = ds41f_put(b, &launch);
	return DS41F_THEN(draft_cut_launches(
		b,
		(struct linear_range){
			area, (uint64_t)(positions * DS41F_HIDDEN_ROW_BYTES) },
		(struct linear_range){ area + (uint64_t)DS41F_DRAFT_SHARE_STEP,
				       STEP_BLOCK_BYTES_VALUE +
					       positions *
						       STEP_ROW_BYTES_VALUE },
		(struct linear_range){
			area + DS41F_DRAFT_SHARE_VIEWS,
			(uint64_t)(positions * STEP_DRAFT_VIEW_BYTES) },
		(struct linear_range){ logits, (uint64_t)positions * shard *
						       sizeof(float) },
		(struct linear_range){ results,
				       (uint64_t)positions *
					       STEP_DRAFT_RESULT_BYTES },
		(struct linear_range){ markov,
				       (uint64_t)positions *
					       DS41F_DRAFT_MARKOV_ROW_BYTES },
		first_call));
}

/* The three stream means are taken before their layers change the streams, over the whole live chunk. */
MUST_CHECK result_status ds41f_draft_tap_launch(struct ds41f_builder *b,
						uint32_t layer)
{
	const struct ds41f_hyper_draft_inputs_description description = {
		DS41F_ARRAY(STREAMS),
		ds41f_no_range,
		DS41F_ARRAY(DRAFT_TAPS),
		b->rows,
		DS41F_HYPER_STREAM_MEAN,
		layer - DS41F_DRAFT_FIRST_TAP_LAYER,
		0
	};
	return ds41f_hyper_draft_inputs_prepare(&description, b->arena,
						&b->hyper, ds41f_slots(b, 1));
}

/* One piece's context products: their quantized rows have their own arrays, never the target's ROW codes. */
MUST_CHECK static result_status draft_context_piece(struct ds41f_builder *b)
{
	const struct ds41f_tensor *const globals = b->plan->draft_globals;
	const struct ds41f_tensor *const first =
		b->plan->layer_tensors[DS41F_DRAFT_FIRST_LAYER];
	const struct ds41f_tensor *const second =
		b->plan->layer_tensors[DS41F_DRAFT_FIRST_LAYER + 1u];
	const struct ds41f_tensor *const last =
		b->plan->layer_tensors[DS41F_DRAFT_FIRST_LAYER +
				       DS41F_DRAFT_STAGES - 1u];
	const struct ds41f_hyper_draft_inputs_description norm = {
		DS41F_ARRAY(DRAFT_MAIN),
		globals[DS41F_TENSOR_DRAFT_MAIN_NORM].stored,
		DS41F_ARRAY(DRAFT_MAIN_X),
		b->rows,
		DS41F_HYPER_ROW_NORM,
		0,
		0
	};
	result_status status = ds41f_quantize_into(
		b, DS41F_ARRAY(DRAFT_TAPS), DS41F_DRAFT_FEATURES,
		DS41F_ARRAY(DRAFT_TAP_CODES), DS41F_ARRAY(DRAFT_TAP_EXPONENTS),
		DS41F_ARRAY(DRAFT_TAP_OPERANDS));
	status = DS41F_THEN(
		ds41f_trunk_of(b, DS41F_ARRAY(DRAFT_TAP_CODES),
			       DS41F_ARRAY(DRAFT_TAP_EXPONENTS),
			       DS41F_ARRAY(DRAFT_TAP_OPERANDS),
			       &globals[DS41F_TENSOR_DRAFT_MAIN_PROJECTION],
			       DS41F_ARRAY(DRAFT_MAIN), NULL, ds41f_no_range));
	status = DS41F_THEN(ds41f_hyper_draft_inputs_prepare(
		&norm, b->arena, &b->hyper, ds41f_slots(b, 1)));
	status = DS41F_THEN(ds41f_quantize_into(
		b, DS41F_ARRAY(DRAFT_MAIN_X), DS41F_HIDDEN,
		DS41F_ARRAY(DRAFT_TAP_CODES), DS41F_ARRAY(DRAFT_TAP_EXPONENTS),
		DS41F_ARRAY(DRAFT_TAP_OPERANDS)));
	status = DS41F_THEN(ds41f_trunk_of(
		b, DS41F_ARRAY(DRAFT_TAP_CODES),
		DS41F_ARRAY(DRAFT_TAP_EXPONENTS),
		DS41F_ARRAY(DRAFT_TAP_OPERANDS), &first[DS41F_TENSOR_WINDOW],
		DS41F_ARRAY(DRAFT_WINDOW_0), &second[DS41F_TENSOR_WINDOW],
		DS41F_ARRAY(DRAFT_WINDOW_1)));
	return DS41F_THEN(ds41f_trunk_of(
		b, DS41F_ARRAY(DRAFT_TAP_CODES),
		DS41F_ARRAY(DRAFT_TAP_EXPONENTS),
		DS41F_ARRAY(DRAFT_TAP_OPERANDS), &last[DS41F_TENSOR_WINDOW],
		DS41F_ARRAY(DRAFT_WINDOW_2), NULL, ds41f_no_range));
}

/* Each target row writes context at its position. A later draft overwrites every rejected or draft slot it reads.
 * The products over the chunk in pieces of piece_rows rows, or (0) over the rows selected. */
MUST_CHECK result_status ds41f_draft_context_launches(struct ds41f_builder *b,
						      uint32_t piece_rows)
{
	ds41f_draft_swap(b, true);
	result_status status = piece_rows == 0 ? draft_context_piece(b) :
						 DS41F_OK;
	for (uint32_t first = 0;
	     piece_rows != 0 && first < b->chunk_rows && status == DS41F_OK;
	     first += piece_rows) {
		ds41f_rows_select(b, 0, first,
				  b->chunk_rows - first < piece_rows ?
					  b->chunk_rows - first :
					  piece_rows);
		status = draft_context_piece(b);
	}
	ds41f_chunk_select(b);
	for (uint32_t stage = 0;
	     stage < DS41F_DRAFT_STAGES && status == DS41F_OK; stage++) {
		const uint32_t layer = DS41F_DRAFT_FIRST_LAYER + stage;
		const struct ds41f_attention_window_description description = {
			DS41F_ARRAY(STEP),
			DS41F_ARRAY(ROTATION),
			b->arrays[DS41F_ARRAY_DRAFT_WINDOW_0 + stage],
			b->plan->layer_tensors[layer][DS41F_TENSOR_WINDOW_NORM]
				.stored,
			{ b->plan->cards[b->card].ring_offset[layer],
			  (uint64_t)b->plan->limits.session_capacity *
				  DS41F_RING_SESSION_BYTES },
			b->rows,
			b->plan->limits.session_capacity,
			b->plan->limits.context_token_limit,
			0,
			0
		};
		status = ds41f_attention_window_prepare(
			&description, b->arena, &b->state, ds41f_slots(b, 1));
	}
	ds41f_draft_swap(b, false);
	return status;
}

_Static_assert(
	DS41F_DRAFT_SLAB_RESULT_BYTES == PROGRAM_BATCH_SLAB_RESULT_BYTES &&
		DS41F_DRAFT_BATCH_MOST_ROWS == PROGRAM_BATCH_MOST_ROWS &&
		DS41F_DRAFT_SLAB_ROWS == LINEAR_MAXIMUM_ROWS &&
		PROGRAM_BATCH_MOST_ROWS / LINEAR_MAXIMUM_ROWS +
				DS41F_DRAFT_SHARE_VERIFIERS * STEP_DRAFT_MOST <=
			SAMPLER_SHARD_CALLS,
	"draft_records.h's BATCH figures are model_seam.h's; a sampler use for every slab and draft position");

/* A BATCH program's verifiers (model_seam.h): a DRAFTS row and its proposals a session (or the description's
 * batch_verifiers, a step's drafting sessions apart from its rows), at most the sessions and
 * DS41F_DRAFT_SHARE_VERIFIERS (8, the C8 cell). */
uint32_t
ds41f_batch_verifiers(const struct ds41f_tp2_program_description *description,
		      const struct program_limits *limits, uint32_t rows)
{
	uint32_t verifiers = description->batch_verifiers != 0 ?
				     description->batch_verifiers :
				     rows / (limits->draft_positions + 1u);
	verifiers = verifiers < limits->session_capacity ?
			    verifiers :
			    limits->session_capacity;
	verifiers = verifiers < DS41F_DRAFT_SHARE_VERIFIERS ?
			    verifiers :
			    DS41F_DRAFT_SHARE_VERIFIERS;
	return verifiers == 0 ? 1u : verifiers;
}

/* batch_draft: verifier v's draft arrays (verifier 0 the plan's, verifier v the stage's batch draft area v - 1), the
 * slab results and the round table, behind whatever rows_select chose. */
static void batch_draft_select(struct ds41f_builder *b)
{
	if (!b->batch_draft)
		return;
	const struct ds41f_tp2_stage_plan *const stage = b->batch_stage;
	const uint32_t positions = b->plan->limits.draft_positions;
	static const enum ds41f_array arrays[DS41F_BATCH_DRAFT_PARTS] = {
		DS41F_ARRAY_DRAFT_STEP,	    DS41F_ARRAY_DRAFT_VIEWS,
		DS41F_ARRAY_DRAFT_RESULTS,  DS41F_ARRAY_DRAFT_MARKOV,
		DS41F_ARRAY_DRAFT_COLLAPSE, DS41F_ARRAY_DRAFT_LOGITS
	};
	const uint64_t area =
		stage->batch_drafts.byte_offset +
		(uint64_t)(b->batch_verifier - 1u) *
			ds41f_batch_draft_offset(DS41F_BATCH_DRAFT_PARTS,
						 positions);
	for (uint32_t part = 0;
	     part < DS41F_BATCH_DRAFT_PARTS && b->batch_verifier != 0; part++)
		b->arrays[arrays[part]] = (struct linear_range){
			area + ds41f_batch_draft_offset(part, positions),
			ds41f_batch_draft_bytes(part, positions)
		};
	if (b->reads_draft_step)
		b->arrays[DS41F_ARRAY_STEP] = b->arrays[DS41F_ARRAY_DRAFT_STEP];
	b->arrays[DS41F_ARRAY_RESULT] = (struct linear_range){
		stage->batch_results.byte_offset,
		(uint64_t)b->batch_slabs * PROGRAM_BATCH_SLAB_RESULT_BYTES
	};
	b->arrays[DS41F_ARRAY_ROUND] = stage->batch_rounds;
}

/* diagnostics (DIAG_DRAFT_PROBE): a copy of the draft's rows of an array (rows of width elements, float32 or
 * bfloat16) into the draft logits at offset bytes, which the dump then reads (the walk leaves the logits alone) */
MUST_CHECK static result_status draft_probe(struct ds41f_builder *b,
					    struct linear_range source,
					    uint32_t width, bool is_float32,
					    uint64_t offset)
{
	const uint64_t element = is_float32 ? sizeof(float) : sizeof(uint16_t);
	const uint32_t rows = b->plan->limits.draft_positions;
	const struct linear_range logits = DS41F_ARRAY(DRAFT_LOGITS);
	const uint64_t bytes = (uint64_t)(rows * width) * element;
	if (offset + bytes > logits.byte_count)
		return STATUS_INVALID_ARGUMENT;
	const struct tp2_join_description copy = {
		.source = { source.byte_offset, bytes },
		.output = { logits.byte_offset + offset, bytes },
		.width = width,
		.row_count = rows,
		.source_row_stride = width,
		.output_row_stride = width,
		.operation = is_float32 ? TP2_JOIN_FLOAT32_COPY :
					  TP2_JOIN_BFLOAT16_COPY
	};
	return tp2_join_prepare(&copy, b->arena, b->batch_join_kernel,
				ds41f_slots(b, 1));
}

#define PROBE_QUERY_WIDTH (DS41F_HEADS_VALUE * DS41F_HEAD_DIMS)
#define PROBE_ROTATION_WIDTH (DS41F_ROTATION_ROW_BYTES / 4u)
#define PROBE_STREAMS_WIDTH (DS41F_STREAMS_ROW_BYTES / 2u)
#define PROBE_POST_WIDTH (DS41F_SITE_ROW_BYTES / 4u)
#define PROBE_COMB_WIDTH (DS41F_COMB_ROW_BYTES / 4u)
/* draft_walk's DIAG_DRAFT_PROBE: where each probe lands in the logits */
#define PROBE_FRONT_STREAMS_OFFSET 1257472u
#define PROBE_POST_OFFSET 1255424u
#define PROBE_COMB_OFFSET 1255936u
#define PROBE_HEADS_OFFSET 790528u
#define PROBE_QUERIES_OFFSET 4096u
#define PROBE_CORE_OUTPUT_OFFSET 393216u
#define PROBE_ROTATION_AGAIN_OFFSET 786432u
#define PROBE_GROUPS_OFFSET 1122304u
#define PROBE_SUBLAYER_OUTPUT_OFFSET 1204224u
#define PROBE_POST_STREAMS_OFFSET 1462272u
#define DECIMAL_BASE 10 /* DIAG_DRAFT_STAGES */

MUST_CHECK result_status ds41f_draft_walk(struct ds41f_builder *b)
{
	const uint32_t first_launch = b->made->cards[b->card].launch_count;
	const struct program_limits *const limits = &b->plan->limits;
	batch_draft_select(b);
	const uint32_t flags = b->batch_draft ?
				       DS41F_DRAFT_ROUND_TABLE |
					       DS41F_DRAFT_SLAB_RESULTS :
				       0u;
	const struct ds41f_verify_matches_description verify = {
		DS41F_ARRAY(STEP),
		DS41F_ARRAY(RESULT),
		DS41F_ARRAY(ROUND),
		DS41F_ARRAY(DRAFT_STEP),
		DS41F_ARRAY(DRAFT_VIEWS),
		b->rows,
		limits->draft_positions,
		DS41F_END_TOKEN,
		DS41F_VOCABULARY_VALUE,
		b->batch_draft ? b->batch_verifier : 0u,
		flags,
		b->batch_draft ? limits->session_capacity : 0u,
		0
	};
	ds41f_draft_swap(b, true);
	result_status status = ds41f_verify_matches_prepare(
		&verify, b->arena, &b->draft, ds41f_slots(b, 1));
	b->reads_draft_step = true;
	ds41f_rows_select(b, 0, 0, limits->draft_positions);
	batch_draft_select(b);
	status = DS41F_THEN(ds41f_batch_draft_scratch(b));
	status = DS41F_THEN(ds41f_step_launches(b, false));
	status = DS41F_THEN(ds41f_step_launches(b, true));
	uint32_t layer_first = b->made->cards[b->card].launch_count;
	/* diagnostics (gnmac_port, the BATCH drafts): DIAG_DRAFT_STAGES walks only so many stages, and
     * DIAG_DRAFT_SKIP_FFN=1 leaves out their FFN sites, so that the collapse rows hold that point's streams */
	const char *const stages_text = getenv("DIAG_DRAFT_STAGES");
	const char *const skip_text = getenv("DIAG_DRAFT_SKIP_FFN");
	const uint32_t stages =
		stages_text != NULL ?
			(uint32_t)strtoul(stages_text, NULL, DECIMAL_BASE) :
			DS41F_DRAFT_STAGES;
	const bool skip_ffn = skip_text != NULL && strcmp(skip_text, "1") == 0;
	/* DIAG_DRAFT_PROBE: stage 0's attention in its parts, the rotations, wq_b's heads and the queries behind the
     * queries launch, then the core's output and the rotations again, into the logits (offsets 0, 790,528, 4,096,
     * 393,216, 786,432); the head and what follows it are left out */
	const bool probe = getenv("DIAG_DRAFT_PROBE") != NULL &&
			   b->batch_join_kernel != BACKEND_NO_KERNEL;
	for (uint32_t stage = 0;
	     stage < DS41F_DRAFT_STAGES && stage < stages && status == DS41F_OK;
	     stage++) {
		const uint32_t layer = DS41F_DRAFT_FIRST_LAYER + stage;
		status = b->draft_layers ?
				 ds41f_tp2_layer_select(b, layer) :
				 status; /* the card's rank of the stage */
		if (probe && stage == 0) {
			status = DS41F_THEN(ds41f_layer_steps(
				b, layer, 0, 0, DS41F_FRONT_STEPS));
			/* the site front's: the streams it read, its post and comb (offsets 1,257,472, 1,255,424,
		     * 1,255,936) */
			status = DS41F_THEN(draft_probe(
				b, DS41F_ARRAY(STREAMS), PROBE_STREAMS_WIDTH,
				false, PROBE_FRONT_STREAMS_OFFSET));
			status = DS41F_THEN(draft_probe(b, DS41F_ARRAY(POST),
							PROBE_POST_WIDTH, true,
							PROBE_POST_OFFSET));
			status = DS41F_THEN(draft_probe(b, DS41F_ARRAY(COMB),
							PROBE_COMB_WIDTH, true,
							PROBE_COMB_OFFSET));
			status = DS41F_THEN(ds41f_layer_steps(
				b, layer, 0, 0,
				DS41F_QUERY_PRODUCTS |
					DS41F_INDEX_WEIGHTS_PRODUCT |
					DS41F_QUERIES));
			status = DS41F_THEN(
				draft_probe(b, DS41F_ARRAY(ROTATION),
					    PROBE_ROTATION_WIDTH, true, 0));
			status = DS41F_THEN(draft_probe(
				b, DS41F_ARRAY(HEADS), PROBE_QUERY_WIDTH, false,
				PROBE_HEADS_OFFSET));
			status = DS41F_THEN(draft_probe(
				b, DS41F_ARRAY(QUERIES), PROBE_QUERY_WIDTH,
				false, PROBE_QUERIES_OFFSET));
			status = DS41F_THEN(
				ds41f_layer_steps(b, layer, 0, 0, DS41F_CORE));
			status = DS41F_THEN(
				draft_probe(b, DS41F_ARRAY(ATTENTION_OUTPUT),
					    PROBE_QUERY_WIDTH, false,
					    PROBE_CORE_OUTPUT_OFFSET));
			status = DS41F_THEN(draft_probe(
				b, DS41F_ARRAY(ROTATION), PROBE_ROTATION_WIDTH,
				true, PROBE_ROTATION_AGAIN_OFFSET));
			status = DS41F_THEN(ds41f_layer_steps(
				b, layer, 0, 0,
				DS41F_OUTPUT_PRODUCTS | DS41F_SITE_POST));
			/* wo_a's groups, the sublayer output and the streams behind the post (1,122,304, 1,204,224,
		     * 1,462,272) */
			status = DS41F_THEN(draft_probe(b, DS41F_ARRAY(GROUPS),
							DS41F_WIDEST_ROW, false,
							PROBE_GROUPS_OFFSET));
			status = DS41F_THEN(draft_probe(
				b, DS41F_ARRAY(SUBLAYER_OUTPUT), DS41F_HIDDEN,
				false, PROBE_SUBLAYER_OUTPUT_OFFSET));
			status = DS41F_THEN(draft_probe(
				b, DS41F_ARRAY(STREAMS), PROBE_STREAMS_WIDTH,
				false, PROBE_POST_STREAMS_OFFSET));
		} else {
			status = DS41F_THEN(ds41f_layer_steps(
				b, layer, 0, 0,
				DS41F_FRONT_STEPS | DS41F_REST_STEPS));
		}
		status = skip_ffn ?
				 status :
				 DS41F_THEN(ds41f_ffn_launches(b, layer,
							       &layer_first));
	}
	b->tensor_layer = NULL;
	const struct ds41f_hyper_final_description final = {
		DS41F_ARRAY(STREAMS),
		DS41F_ARRAY(CARRIED_PRE),
		b->plan->draft_globals[DS41F_TENSOR_DRAFT_NORM].stored,
		DS41F_ARRAY(HIDDEN),
		DS41F_ARRAY(DRAFT_COLLAPSE),
		b->rows,
		0
	};
	status = DS41F_THEN(ds41f_hyper_final_prepare(
		&final, b->arena, &b->hyper, ds41f_slots(b, 1)));
	const bool shares = b->head_stage != NULL &&
			    b->head_stage->description.draft_shards;
	if (probe) { /* diagnostics: no head, Markov bias, sampler or confidence (the logits hold the probes) */
	} else if (b->draft_layers) { /* every rank holds the draft's rows: its own cut of the draft */
		status = DS41F_THEN(draft_cut_launches(
			b, DS41F_ARRAY(HIDDEN), DS41F_ARRAY(STEP),
			DS41F_ARRAY(DRAFT_VIEWS), DS41F_ARRAY(DRAFT_LOGITS),
			DS41F_ARRAY(DRAFT_RESULTS), DS41F_ARRAY(DRAFT_MARKOV),
			1u));
	} else if (shares) { /* draft_shards: the rows to the other ranks, then the card's own cut of the draft */
		const struct ds41f_draft_share_arguments share = {
			DS41F_ARRAY(HIDDEN).byte_offset,
			DS41F_ARRAY(DRAFT_STEP).byte_offset,
			DS41F_ARRAY(DRAFT_VIEWS).byte_offset,
			b->head_stage->draft_peers.byte_offset,
			b->head_stage->draft_counters.byte_offset,
			b->plan->cards[b->card].array_offset[DS41F_ARRAY_STEP],
			limits->draft_positions,
			b->head_stage->description.rank,
			b->batch_draft ? b->batch_verifier : 0u,
			0
		};
		struct kernel_launch launch;
		prepare_launch_fill(&launch, b->draft_share_kernel,
				    DS41F_DRAFT_SHARE_RANKS *
					    DS41F_DRAFT_SHARE_BLOCKS_PER_RANK,
				    1, DS41F_DRAFT_SHARE_THREADS, 0, &share,
				    sizeof(share));
		status = DS41F_THEN(ds41f_put(b, &launch));
		status = DS41F_THEN(draft_cut_launches(
			b, DS41F_ARRAY(HIDDEN), DS41F_ARRAY(STEP),
			DS41F_ARRAY(DRAFT_VIEWS), DS41F_ARRAY(DRAFT_LOGITS),
			DS41F_ARRAY(DRAFT_RESULTS), DS41F_ARRAY(DRAFT_MARKOV),
			b->batch_draft ?
				b->batch_slabs +
					b->batch_verifier *
						limits->draft_positions :
				1u));
	} else {
		status = DS41F_THEN(head_product(
			b, 0, DS41F_VOCABULARY_VALUE, DS41F_ARRAY(HIDDEN),
			DS41F_ARRAY(STEP), DS41F_ARRAY(DRAFT_LOGITS)));
	}
	for (uint32_t position = 0; position < limits->draft_positions &&
				    status == DS41F_OK && !shares && !probe;
	     position++) {
		const struct ds41f_draft_markov_bias_description markov = {
			DS41F_ARRAY(DRAFT_STEP),
			DS41F_ARRAY(DRAFT_RESULTS),
			b->plan->draft_globals[DS41F_TENSOR_DRAFT_MARKOV_EMBEDDING]
				.stored,
			b->plan->draft_globals[DS41F_TENSOR_DRAFT_MARKOV_HEAD]
				.stored,
			DS41F_ARRAY(DRAFT_LOGITS),
			DS41F_ARRAY(DRAFT_MARKOV),
			position,
			limits->draft_positions,
			DS41F_VOCABULARY_VALUE,
			0,
			0
		};
		const struct sampler_description sampler = {
			{ DS41F_ARRAY(DRAFT_VIEWS).byte_offset +
				  (uint64_t)position * STEP_DRAFT_VIEW_BYTES,
			  STEP_DRAFT_VIEW_BYTES },
			{ DS41F_ARRAY(SESSIONS).byte_offset,
			  (uint64_t)limits->session_capacity *
				  STEP_SESSION_BYTES },
			{ DS41F_ARRAY(DRAFT_LOGITS).byte_offset +
				  (uint64_t)position * DS41F_LOGITS_ROW_BYTES,
			  DS41F_LOGITS_ROW_BYTES },
			{ DS41F_ARRAY(SAMPLER_WEIGHTS).byte_offset,
			  DS41F_LOGITS_ROW_BYTES },
			{ DS41F_ARRAY(DRAFT_RESULTS).byte_offset +
				  (uint64_t)position * STEP_DRAFT_RESULT_BYTES,
			  STEP_DRAFT_RESULT_BYTES },
			DS41F_VOCABULARY_VALUE,
			limits->session_capacity,
			1u,
			0
		};
		status = ds41f_draft_markov_bias_prepare(
			&markov, b->arena, &b->draft, ds41f_slots(b, 1));
		status = DS41F_THEN(sampler_prepare_entry(
			&sampler, b->arena, &b->sampler, ds41f_slots(b, 1)));
	}
	const struct ds41f_draft_confidence_description confidence = {
		DS41F_ARRAY(DRAFT_STEP),
		DS41F_ARRAY(DRAFT_COLLAPSE),
		DS41F_ARRAY(DRAFT_MARKOV),
		b->plan->draft_globals[DS41F_TENSOR_DRAFT_CONFIDENCE].stored,
		DS41F_ARRAY(DRAFT_RESULTS),
		DS41F_ARRAY(ROUND),
		limits->draft_positions,
		b->batch_draft ? DS41F_DRAFT_ROUND_TABLE : 0u,
		0
	};
	if (b->card == b->plan->result_card &&
	    !probe) /* draft_layers: the other ranks write no round */
		status = DS41F_THEN(ds41f_draft_confidence_prepare(
			&confidence, b->arena, &b->draft, ds41f_slots(b, 1)));
	b->reads_draft_step = false;
	ds41f_chunk_select(b);
	ds41f_draft_swap(b, false);
	if (!b->draft_tail_begun) /* a BATCH program's verifiers' walks, draft_layers' ranks: one draft tail */
		b->made->program.draft_segment_first =
			b->made->program.segment_count;
	b->draft_tail_begun = true;
	ds41f_segment_add(b, SEGMENT_NO_LAYER, first_launch);
	return status;
}

/* batch_draft_rows, a BATCH program's draft on the result card (ds41f_dspark_records.h's BATCHED): one verify launch a
 * verifier into one draft step of slab blocks; the rotation, embedding, three layers, collapse and head of every
 * verifier's rows slab by slab (layers outermost: a slab's attention behind the earlier slabs' of its layer, as a
 * FULL program's); then each verifier's positions (Markov bias and sampler) and its confidence into the round
 * table. Run as a thin program's (slab_count 1, the card selected so). */
MUST_CHECK result_status ds41f_batch_draft_rows_walk(struct ds41f_builder *b,
						     uint32_t verifiers)
{
	const uint32_t first_launch = b->made->cards[b->card].launch_count;
	const struct program_limits *const limits = &b->plan->limits;
	const struct ds41f_tp2_stage_plan *const stage = b->batch_stage;
	const uint32_t positions = limits->draft_positions,
		       rows = verifiers * positions;
	const uint32_t slabs =
		(rows + LINEAR_MAXIMUM_ROWS - 1u) / LINEAR_MAXIMUM_ROWS;
	if (rows > DS41F_BATCH_DRAFT_ROWS ||
	    verifiers > DS41F_DRAFT_BATCH_SLOTS ||
	    stage->batch_rows.byte_count <
		    ds41f_batch_rows_offset(DS41F_BATCH_ROWS_PARTS))
		return STATUS_INVALID_ARGUMENT;
	struct linear_range part[DS41F_BATCH_DRAFT_PARTS];
	for (uint32_t index = 0; index < DS41F_BATCH_DRAFT_PARTS; index++)
		part[index] = (struct linear_range){
			stage->batch_rows.byte_offset +
				ds41f_batch_rows_offset(index),
			ds41f_batch_rows_bytes(index)
		};
	const uint64_t slab_step_bytes = DS41F_DRAFT_SLAB_STEP_BYTES;
	const struct linear_range results = {
		stage->batch_results.byte_offset,
		(uint64_t)b->batch_slabs * PROGRAM_BATCH_SLAB_RESULT_BYTES
	};
	/* batch_draft_wide 3 over several slabs: the step block of all rows too (draft_wide_layer's window and
	 * core) */
	const bool wide_step = b->batch_draft_wide >= DRAFT_WIDE_CORES &&
			       slabs > 1u && b->draft_layers;
	ds41f_draft_swap(b, true);
	ds41f_chunk_select(b);
	result_status status = DS41F_OK;
	for (uint32_t verifier = 0; verifier < verifiers && status == DS41F_OK;
	     verifier++) {
		const struct ds41f_verify_matches_description verify = {
			DS41F_ARRAY(STEP),
			results,
			stage->batch_rounds,
			part[DS41F_BATCH_DRAFT_STEP],
			part[DS41F_BATCH_DRAFT_VIEWS],
			b->rows,
			positions,
			DS41F_END_TOKEN,
			DS41F_VOCABULARY_VALUE,
			verifier,
			DS41F_DRAFT_ROUND_TABLE | DS41F_DRAFT_SLAB_RESULTS |
				DS41F_DRAFT_BATCHED |
				(wide_step ? DS41F_DRAFT_WIDE_STEP : 0u),
			limits->session_capacity,
			slabs
		};
		status = ds41f_verify_matches_prepare(
			&verify, b->arena, &b->draft, ds41f_slots(b, 1));
	}
	b->reads_draft_step = true;
	/* slab s: chunk rows 8 s .., its slab block as the step area, its rows of the batch area's parts */
	struct linear_range slab_step = { 0, 0 };
	const uint64_t logits_row_bytes = DS41F_LOGITS_ROW_BYTES;
#define BATCH_DRAFT_SLAB(s)                                                    \
	do {                                                                   \
		ds41f_rows_select(b, 0, (s) * LINEAR_MAXIMUM_ROWS,             \
				  LINEAR_MAXIMUM_ROWS);                        \
		slab_step = (struct linear_range){                             \
			part[DS41F_BATCH_DRAFT_STEP].byte_offset +             \
				(s) * slab_step_bytes,                         \
			slab_step_bytes                                        \
		};                                                             \
		b->arrays[DS41F_ARRAY_STEP] = slab_step;                       \
		b->arrays[DS41F_ARRAY_DRAFT_STEP] = slab_step;                 \
		b->arrays[DS41F_ARRAY_DRAFT_COLLAPSE] = (struct linear_range){ \
			part[DS41F_BATCH_DRAFT_COLLAPSE].byte_offset +         \
				(uint64_t)(s) * LINEAR_MAXIMUM_ROWS *          \
					DS41F_HIDDEN_ROW_BYTES,                \
			(uint64_t)LINEAR_MAXIMUM_ROWS * DS41F_HIDDEN_ROW_BYTES \
		};                                                             \
	} while (0)
	status = DS41F_THEN(ds41f_batch_draft_scratch(b));
	for (uint32_t slab = 0; slab < slabs && status == DS41F_OK; slab++) {
		BATCH_DRAFT_SLAB(slab);
		status = ds41f_step_launches(b, false);
		status = DS41F_THEN(ds41f_step_launches(b, true));
	}
	uint32_t layer_first = b->made->cards[b->card].launch_count;
	b->draft_slab_steps = part[DS41F_BATCH_DRAFT_STEP].byte_offset;
	b->draft_slab_collapse = part[DS41F_BATCH_DRAFT_COLLAPSE].byte_offset;
	b->draft_slab_count = slabs;
	b->draft_rows = rows;
	b->draft_wide_step = wide_step;
	for (uint32_t stage_index = 0;
	     stage_index < DS41F_DRAFT_STAGES && status == DS41F_OK;
	     stage_index++) {
		const uint32_t layer = DS41F_DRAFT_FIRST_LAYER + stage_index;
		/* batch_draft_wide: each site's TP sum and post once over every slab's rows (one BF16
		 * ring), behind the
         * slabs' sublayers, where the rows fit one ring; else a slab at a time. 2: the layer's products over all its
         * rows too (draft_wide_layer) */
		bool once = b->batch_draft_wide != 0 && slabs > 1u &&
			    b->draft_layers;
		if (once) {
			status = ds41f_tp2_layer_select(b, layer);
			once = status == DS41F_OK &&
			       b->tensor_layer->bf16_ring &&
			       !b->tensor_layer->exact_dense &&
			       !b->tensor_layer->exact_output_shards &&
			       rows <= b->tensor_layer->ring_rows;
		}
		if (once && b->batch_draft_wide >= DS41F_DRAFT_WIDE_PRODUCTS) {
			status = ds41f_draft_wide_layer(b, layer);
			continue;
		}
		for (uint32_t slab = 0; slab < slabs && status == DS41F_OK;
		     slab++) {
			BATCH_DRAFT_SLAB(slab);
			status =
				b->draft_layers ?
					ds41f_tp2_layer_select(b, layer) :
					status; /* the slab's TP sums: its ring */
			status = DS41F_THEN(ds41f_layer_steps(
				b, layer, 0, 0,
				(DS41F_FRONT_STEPS | DS41F_REST_STEPS) &
					~(once ? DS41F_SITE_POST : 0u)));
			b->ffn_post_deferred = once;
			status = once ? status :
					DS41F_THEN(ds41f_ffn_launches(
						b, layer, &layer_first));
		}
		if (once && status == DS41F_OK) {
			BATCH_DRAFT_SLAB(0);
			ds41f_rows_select(b, 0, 0, rows);
			status = ds41f_layer_steps(b, layer, 0, 0,
						   DS41F_SITE_POST);
			for (uint32_t slab = 0;
			     slab < slabs && status == DS41F_OK; slab++) {
				BATCH_DRAFT_SLAB(slab);
				status = ds41f_ffn_launches(b, layer,
							    &layer_first);
			}
			BATCH_DRAFT_SLAB(0);
			ds41f_rows_select(b, 0, 0, rows);
			status = DS41F_THEN(ds41f_ffn_post(b, layer));
		}
		b->ffn_post_deferred = false;
	}
	b->tensor_layer = NULL;
	/* each slab's final (the rows verifier-major, the collapse rows the confidences'); then the rows
	 * position-major,
     * slot v of position j verifier v's row j (ds41f_dspark_records.h's BATCHED): the head of each position's 8
     * slots
     * gated by its view, or with batch_wide_head one wide head over them all (the head read once); a position at a time
     * its Markov bias over every slot and one sampler over its view; then each verifier's confidence */
	for (uint32_t slab = 0; slab < slabs && status == DS41F_OK; slab++) {
		BATCH_DRAFT_SLAB(slab);
		const struct ds41f_hyper_final_description final = {
			DS41F_ARRAY(STREAMS),
			DS41F_ARRAY(CARRIED_PRE),
			b->plan->draft_globals[DS41F_TENSOR_DRAFT_NORM].stored,
			DS41F_ARRAY(HIDDEN),
			DS41F_ARRAY(DRAFT_COLLAPSE),
			b->rows,
			0
		};
		status = ds41f_hyper_final_prepare(&final, b->arena, &b->hyper,
						   ds41f_slots(b, 1));
	}
#undef BATCH_DRAFT_SLAB
	const uint64_t hidden_row_bytes = DS41F_HIDDEN_ROW_BYTES,
		       view_bytes = DS41F_DRAFT_SLAB_STEP_BYTES;
	const uint32_t slot_count = DS41F_DRAFT_BATCH_SLOTS;
	const struct linear_range hidden_slots = {
		stage->batch_rows.byte_offset +
			ds41f_batch_rows_offset(DS41F_BATCH_ROWS_HIDDEN),
		ds41f_batch_rows_bytes(DS41F_BATCH_ROWS_HIDDEN)
	};
	ds41f_rows_select(b, 0, 0, rows);
	const struct linear_range hidden = DS41F_ARRAY(HIDDEN);
	/* batch_wide_head with fewer verifiers than slots: the draft's final rows (verifier-major) are the head's
	 * rows
     * alone, then each position's logits row to its slot (position j of verifier v: row 8 j + v); the slots past
     * the
     * verifiers hold no live row in either form (batch_draft_slots_head: the slots' head of positions x 8 rows,
     * as
     * before). One verifier (BATCH-8 of one: C1): the head's rows are rows 0 .. positions - 1 of the logits, moved in
     * place; more: the logits rows behind the slots' hold them where they fit (the area's rows of 56 or of a
     * cut) */
	const bool own_rows_head = verifiers < DS41F_DRAFT_BATCH_SLOTS &&
				   stage->description.batch_wide_head &&
				   !stage->description.batch_draft_slots_head;
	for (uint32_t position = 0;
	     position < positions && status == DS41F_OK && !own_rows_head;
	     position++) {
		const struct tp2_join_description copy = {
			.source = { hidden.byte_offset +
					    position * hidden_row_bytes,
				    ((uint64_t)(verifiers - 1u) * positions +
				     1u) * hidden_row_bytes },
			.output = { hidden_slots.byte_offset +
					    (uint64_t)position * slot_count *
						    hidden_row_bytes,
				    (uint64_t)verifiers * hidden_row_bytes },
			.width = DS41F_HIDDEN,
			.row_count = verifiers,
			.source_row_stride = positions * DS41F_HIDDEN,
			.output_row_stride = DS41F_HIDDEN,
			.operation = TP2_JOIN_BFLOAT16_COPY
		};
		status = tp2_join_prepare(&copy, b->arena, b->batch_join_kernel,
					  ds41f_slots(b, 1));
	}
	const struct linear_range views = part[DS41F_BATCH_DRAFT_VIEWS];
	const struct linear_range logits = part[DS41F_BATCH_DRAFT_LOGITS];
	const struct linear_range view0 = { views.byte_offset, view_bytes };
	const uint64_t slots_logits_bytes =
		(uint64_t)slot_count * logits_row_bytes;
	ds41f_rows_select(b, 0, 0, slot_count);
	/* draft_layers: every rank its cut of the vocabulary (its rows of the head, its Markov bias, the sampler
	 * shard) */
	const uint32_t shard = DS41F_VOCABULARY_VALUE / TP_GROUP_MOST_RANKS;
	const uint32_t vocabulary = b->draft_layers ? shard :
						      DS41F_VOCABULARY_VALUE;
	/* the moves in pieces of a shard's columns (a join's widest rows): one piece for a cut, four for the whole */
	const uint64_t row_bytes = (uint64_t)vocabulary * sizeof(float),
		       piece_bytes = (uint64_t)shard * sizeof(float);
	const uint64_t behind = (uint64_t)positions * slot_count *
				row_bytes; /* the rows behind the slots' */
	const bool behind_head = own_rows_head && verifiers > 1u &&
				 behind + rows * row_bytes <= logits.byte_count;
	if (behind_head) {
		status = DS41F_THEN(head_wide_product(
			b, b->draft_layers ? head_cut_first(b) : 0u, vocabulary,
			rows, hidden, 0, 0,
			(struct linear_range){ logits.byte_offset + behind,
					       rows * row_bytes },
			false));
		for (uint32_t piece = 0;
		     piece < vocabulary / shard * verifiers &&
		     status == DS41F_OK;
		     piece++) {
			const uint32_t verifier = piece / (vocabulary / shard),
				       column = piece % (vocabulary / shard);
			const struct tp2_join_description slot_rows = {
				.source = { logits.byte_offset + behind +
						    (uint64_t)verifier *
							    positions *
							    row_bytes +
						    column * piece_bytes,
					    (uint64_t)(positions - 1u) *
							    row_bytes +
						    piece_bytes },
				.output = { logits.byte_offset +
						    verifier * row_bytes +
						    column * piece_bytes,
					    (uint64_t)(positions - 1u) *
							    slot_count *
							    row_bytes +
						    piece_bytes },
				.width = shard,
				.row_count = positions,
				.source_row_stride = vocabulary,
				.output_row_stride = slot_count * vocabulary,
				.operation = TP2_JOIN_FLOAT32_COPY
			};
			status = tp2_join_prepare(&slot_rows, b->arena,
						  b->batch_join_kernel,
						  ds41f_slots(b, 1));
		}
	} else if (own_rows_head && verifiers == 1u) {
		const uint32_t cut_first = b->draft_layers ? head_cut_first(b) :
							     0u;
		status = DS41F_THEN(head_wide_product(b, cut_first, vocabulary,
						      positions, hidden, 0, 0,
						      logits, false));
		for (uint32_t piece = 0; piece < vocabulary / shard &&
					 positions > 1u && status == DS41F_OK;
		     piece++) {
			const uint32_t moved =
				positions - 1u; /* the rows behind the first */
			const struct tp2_join_description slot_rows = {
				.source = { logits.byte_offset + row_bytes +
						    piece * piece_bytes,
					    (uint64_t)(moved - 1u) * row_bytes +
						    piece_bytes },
				.output = { logits.byte_offset +
						    slot_count * row_bytes +
						    piece * piece_bytes,
					    (uint64_t)(moved - 1u) *
							    slot_count *
							    row_bytes +
						    piece_bytes },
				.width = shard,
				.row_count = moved,
				.source_row_stride = vocabulary,
				.output_row_stride = slot_count * vocabulary,
				.operation = TP2_JOIN_FLOAT32_COPY
			};
			status = tp2_join_prepare(&slot_rows, b->arena,
						  b->batch_join_kernel,
						  ds41f_slots(b, 1));
		}
	} else if (stage->description.batch_wide_head) {
		for (uint32_t position = 0; position < positions &&
					    status == DS41F_OK && own_rows_head;
		     position++) {
			const struct tp2_join_description copy = {
				/* the slots' head after all: the joins it reads */
				.source = { hidden.byte_offset +
						    position * hidden_row_bytes,
					    ((uint64_t)(verifiers - 1u) *
						     positions +
					     1u) * hidden_row_bytes },
				.output = { hidden_slots.byte_offset +
						    (uint64_t)position *
							    slot_count *
							    hidden_row_bytes,
					    (uint64_t)verifiers *
						    hidden_row_bytes },
				.width = DS41F_HIDDEN,
				.row_count = verifiers,
				.source_row_stride = positions * DS41F_HIDDEN,
				.output_row_stride = DS41F_HIDDEN,
				.operation = TP2_JOIN_BFLOAT16_COPY
			};
			status = tp2_join_prepare(&copy, b->arena,
						  b->batch_join_kernel,
						  ds41f_slots(b, 1));
		}
		status = DS41F_THEN(head_wide_product(
			b, b->draft_layers ? head_cut_first(b) : 0u, vocabulary,
			positions * slot_count, hidden_slots, 0, 0, logits,
			false));
	}
	if (b->draft_layers && !stage->description.batch_wide_head)
		return STATUS_INVALID_ARGUMENT;
	for (uint32_t position = 0;
	     position < positions && status == DS41F_OK &&
	     !stage->description.batch_wide_head;
	     position++)
		status = head_product(
			b, 0, DS41F_VOCABULARY_VALUE,
			(struct linear_range){
				hidden_slots.byte_offset +
					(uint64_t)position * slot_count *
						hidden_row_bytes,
				(uint64_t)slot_count * hidden_row_bytes },
			(struct linear_range){ views.byte_offset +
						       position * view_bytes,
					       view_bytes },
			(struct linear_range){
				logits.byte_offset +
					position * slots_logits_bytes,
				slots_logits_bytes });
	const struct linear_range markov_head =
		b->plan->draft_globals[DS41F_TENSOR_DRAFT_MARKOV_HEAD].stored;
	const uint64_t skipped =
		b->draft_layers ? (uint64_t)b->head_stage->description.rank *
					  shard * DS41F_DRAFT_MARKOV_ROW_BYTES :
				  0u;
	const uint64_t cut =
		(uint64_t)vocabulary * DS41F_DRAFT_MARKOV_ROW_BYTES;
	if (skipped > markov_head.byte_count ||
	    cut > markov_head.byte_count - skipped)
		return STATUS_INVALID_ARGUMENT;
	for (uint32_t position = 0;
	     position < positions && status == DS41F_OK && b->draft_layers;
	     position++) {
		const uint64_t cut_logits_bytes =
			(uint64_t)slot_count * vocabulary * sizeof(float);
		const struct ds41f_draft_markov_bias_description markov = {
			view0,
			part[DS41F_BATCH_DRAFT_RESULTS],
			b->plan->draft_globals[DS41F_TENSOR_DRAFT_MARKOV_EMBEDDING]
				.stored,
			{ markov_head.byte_offset + skipped, cut },
			logits,
			part[DS41F_BATCH_DRAFT_MARKOV],
			position,
			positions,
			vocabulary,
			DS41F_VOCABULARY_VALUE,
			DS41F_DRAFT_BATCHED
		};
		status = ds41f_draft_markov_bias_prepare(
			&markov, b->arena, &b->draft, ds41f_slots(b, 1));
		status = DS41F_THEN(sampler_shard_launch(
			b,
			(struct linear_range){ views.byte_offset +
						       position * view_bytes,
					       view_bytes },
			slot_count,
			(struct linear_range){
				logits.byte_offset + position * cut_logits_bytes,
				cut_logits_bytes },
			(struct linear_range){
				part[DS41F_BATCH_DRAFT_RESULTS].byte_offset +
					(uint64_t)position *
						DS41F_DRAFT_SLAB_RESULT_BYTES,
				DS41F_DRAFT_SLAB_RESULT_BYTES },
			b->batch_slabs + position));
	}
	for (uint32_t position = 0;
	     position < positions && status == DS41F_OK && !b->draft_layers;
	     position++) {
		const struct ds41f_draft_markov_bias_description markov = {
			view0,
			part[DS41F_BATCH_DRAFT_RESULTS],
			b->plan->draft_globals[DS41F_TENSOR_DRAFT_MARKOV_EMBEDDING]
				.stored,
			b->plan->draft_globals[DS41F_TENSOR_DRAFT_MARKOV_HEAD]
				.stored,
			logits,
			part[DS41F_BATCH_DRAFT_MARKOV],
			position,
			positions,
			DS41F_VOCABULARY_VALUE,
			0,
			DS41F_DRAFT_BATCHED
		};
		const struct sampler_description sampler = {
			{ views.byte_offset + position * view_bytes,
			  view_bytes },
			{ DS41F_ARRAY(SESSIONS).byte_offset,
			  (uint64_t)limits->session_capacity *
				  STEP_SESSION_BYTES },
			{ logits.byte_offset + position * slots_logits_bytes,
			  slots_logits_bytes },
			{ DS41F_ARRAY(SAMPLER_WEIGHTS).byte_offset,
			  slots_logits_bytes },
			{ part[DS41F_BATCH_DRAFT_RESULTS].byte_offset +
				  (uint64_t)position *
					  DS41F_DRAFT_SLAB_RESULT_BYTES,
			  DS41F_DRAFT_SLAB_RESULT_BYTES },
			DS41F_VOCABULARY_VALUE,
			limits->session_capacity,
			slot_count,
			0
		};
		status = ds41f_draft_markov_bias_prepare(
			&markov, b->arena, &b->draft, ds41f_slots(b, 1));
		status = DS41F_THEN(sampler_prepare_entry(
			&sampler, b->arena, &b->sampler, ds41f_slots(b, 1)));
	}
	for (uint32_t verifier = 0;
	     verifier < verifiers && status == DS41F_OK &&
	     b->card == b->plan->result_card;
	     verifier++) {
		const struct ds41f_draft_confidence_description confidence = {
			view0,
			{ part[DS41F_BATCH_DRAFT_COLLAPSE].byte_offset +
				  (uint64_t)verifier * positions *
					  hidden_row_bytes,
			  (uint64_t)positions * hidden_row_bytes },
			part[DS41F_BATCH_DRAFT_MARKOV],
			b->plan->draft_globals[DS41F_TENSOR_DRAFT_CONFIDENCE]
				.stored,
			part[DS41F_BATCH_DRAFT_RESULTS],
			stage->batch_rounds,
			positions,
			DS41F_DRAFT_ROUND_TABLE | DS41F_DRAFT_BATCHED,
			verifier
		};
		status = ds41f_draft_confidence_prepare(
			&confidence, b->arena, &b->draft, ds41f_slots(b, 1));
	}
	b->reads_draft_step = false;
	ds41f_chunk_select(b);
	ds41f_draft_swap(b, false);
	if (!b->draft_tail_begun) /* draft_layers: one draft tail, a segment a rank */
		b->made->program.draft_segment_first =
			b->made->program.segment_count;
	b->draft_tail_begun = true;
	ds41f_segment_add(b, SEGMENT_NO_LAYER, first_launch);
	return status;
}

/* The card's rank of layer, its tensor plan and kernels, for the launches that follow: draft_layers' draft stage
 * layers, split_overlap's pipeline the next target layer's first half. */
MUST_CHECK result_status ds41f_tp2_layer_select(struct ds41f_builder *b,
						uint32_t layer)
{
	if (b->tp_description == NULL || b->tp_tensor == NULL)
		return STATUS_INVALID_ARGUMENT;
	ds41f_tp2_program_tensor_fill(b, b->tp_description, b->tp_tensor,
				      b->card, layer);
	b->tensor_layer = b->tp_tensor;
	return ds41f_tp2_program_tensor_kernels(b, b->card, b->tp_tensor);
}
