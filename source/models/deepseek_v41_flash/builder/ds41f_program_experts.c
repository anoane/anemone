/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * ds41f_program_experts.c: layer 20's front ends, expert pauses, routed and shared launches, the EXL3 FFN, exact
 * routed (split from model_v41_program.c by whole items).
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
#include "engine/library/bf16_linear/trunk_mma_records.h"
#include "ds41f_builder.h"
#include "ds41f_program_internal.h"

#define FRONT_BRANCH_BOTH \
	2u /* front_branch: the FFN site's coefficient front a side run too */
#define TC_GATE_UP_PRODUCT_LAUNCH 1u
#define TC_DOWN_PRODUCT_LAUNCH 3u
#define SWIGLU_LIMIT \
	10.0f /* CFG:19 swiglu_limit: every expert's (REF:845-847, 879, 887) */
#define EXPERT_BITS 4u /* E2M1 */
/* shared_launches' wide_products: gate, up, down */
#define SHARED_DOWN_PRODUCT 2u
/* graph_branches: a BATCH step's shared expert a branch too */
#define GRAPH_BRANCHES_BATCH 2u
enum { GATE_UP, DOWN };
static const struct matrix_shape_record expert_shapes[] = {
	[GATE_UP] = { DS41F_EXPERT_INNER, DS41F_HIDDEN, EXPERT_BITS, 0 },
	[DOWN] = { DS41F_HIDDEN, DS41F_EXPERT_INNER, EXPERT_BITS, 0 }
};
/* the router's gather from halves of so many rows (pf29: 512-row halves +0.9%, its sync ahead of the experts costing
 * more than a quarter's product saves; 1,024-row halves -1.1%) */
#define ROUTER_GATHER_LEAST_ROWS 1024u

/* Layer 20's front ends (model_seam.h): its segment from *first_launch, then its pool's and keys' copies. */
void ds41f_front_end(struct ds41f_builder *b, uint32_t layer,
		     uint32_t *first_launch)
{
	ds41f_segment_add(b, layer, *first_launch);
	ds41f_copies_hand_off(b, layer, DS41F_FRONT_COPIES);
	b->made->program.front_segment_count = b->made->program.segment_count;
	*first_launch = b->made->cards[b->card].launch_count;
}

/* Where the layer keeps routed experts in host memory, its segment from first_launch to its router ends with an
 * expert pause (model_seam.h) for them, in the card's slots. */
MUST_CHECK static result_status
expert_pause_add(struct ds41f_builder *b, uint32_t layer, uint32_t first_launch)
{
	const struct ds41f_plan *const plan = b->plan;
	struct execution_program *const program = &b->made->program;
	const uint32_t count = plan->host_expert_count[layer],
		       segments_before = program->segment_count;
	struct expert_pause pause = { DS41F_ARRAY(JOBS).byte_offset,
				      plan->cards[b->card].state_offset,
				      plan->slot_byte_count,
				      plan->expert_byte_count,
				      { 0 },
				      program->host_expert_count,
				      count,
				      DS41F_EXPERTS_PER_ROW,
				      plan->cards[b->card].slot_count,
				      b->card,
				      segments_before };
	_Static_assert(sizeof(pause.matrix_offsets) ==
			       sizeof(plan->expert_matrix_offsets),
		       "gate, up, down");
	memcpy(pause.matrix_offsets, plan->expert_matrix_offsets,
	       sizeof(pause.matrix_offsets));
	ds41f_segment_add(b, layer, first_launch);
	if (segments_before >= DS41F_MOST_SEGMENTS ||
	    program->segment_count != segments_before + 1u ||
	    program->expert_pause_count >= DS41F_LAYERS ||
	    count > DS41F_EXPERTS) {
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
	return DS41F_OK;
}

/* The expert table of layer in the card's image, which is what arena_initialise put into its arena (model_v41.h):
 * whole uint64 words inside the image. NULL, behind an error line, where it is not. */
static const void *expert_table_of(const struct ds41f_builder *b,
				   uint32_t layer)
{
	const struct ds41f_card *const card = &b->plan->cards[b->card];
	const uint64_t image = card->state_offset + card->slots_byte_count,
		       table = card->expert_table_offset[layer];
	const uint64_t in_image = table - image;
	if (card->image == NULL || table < image ||
	    image < card->state_offset ||
	    ((uintptr_t)card->image + in_image) % sizeof(uint64_t) != 0 ||
	    card->image_byte_count < DS41F_EXPERT_OFFSETS_BYTES ||
	    in_image > card->image_byte_count - DS41F_EXPERT_OFFSETS_BYTES) {
		LOG_ERROR("program_prepare: no expert table of layer %" PRIu32
			  " in the image of card %" PRIu32,
			  layer, b->card);
		return NULL;
	}
	return card->image + in_image;
}

/* exl3_expert_rotation: the TC MoE records' rotation of this card's rank in layer (exl3_tc_records.h), 0 without. */
static uint32_t expert_rotation_of(const struct ds41f_builder *b,
				   uint32_t layer)
{
	return b->tensor_layer != NULL &&
			       b->tensor_layer->exl3_expert_rotation != 0 &&
			       layer < DS41F_LAYERS &&
			       b->plan->exl3_bits[layer] != 0 ?
		       ds41f_exl3_tc_rotation_of(
			       b->tensor_layer->exl3_expert_rotation,
			       b->tensor_layer->rank, layer) :
		       0u;
}

/* The routed experts of layer for the rows the next launches take, over job_count jobs at the job table. */
MUST_CHECK static result_status
routed_of(const struct ds41f_builder *b, uint32_t layer, uint32_t job_count,
	  struct linear_routed_experts_description *experts_out)
{
	*experts_out = (struct linear_routed_experts_description){
		{ DS41F_ARRAY(JOBS).byte_offset,
		  linear_jobs_byte_count(job_count) },
		b->plan->experts[layer],
		DS41F_ARRAY(ROW_CODES),
		DS41F_ARRAY(ROW_EXPONENTS),
		DS41F_ARRAY(ROW_OPERANDS),
		DS41F_ARRAY(PAIR_HIDDEN),
		DS41F_ARRAY(PAIR_CODES),
		DS41F_ARRAY(PAIR_EXPONENTS),
		DS41F_ARRAY(PAIR_OPERANDS),
		DS41F_ARRAY(PAIR_OUTPUT),
		DS41F_ARRAY(ROW_SUMS),
		expert_shapes[GATE_UP],
		expert_shapes[DOWN],
		WEIGHT_FORMAT_E2M1_SCALED,
		job_count,
		b->rows,
		layer >= DS41F_DRAFT_FIRST_LAYER ? DS41F_DRAFT_EXPERTS_PER_ROW :
						   DS41F_EXPERTS_PER_ROW,
		0,
		0,
		SWIGLU_LIMIT,
		0
	};
	if (b->tensor_layer != NULL && !b->tensor_layer->exact_output_shards &&
	    !b->tensor_layer->experts_partitioned) {
		experts_out->gate_up_shape.output_size =
			b->plan->layer_tensors[layer][DS41F_TENSOR_SHARED_GATE]
				.shape.output_size;
		experts_out->down_shape.input_size =
			b->plan->layer_tensors[layer][DS41F_TENSOR_SHARED_DOWN]
				.shape.input_size;
		if (expert_rotation_of(b, layer) != 0) {
			/* every rank holds 512- and 640-column chunks: the router's matrix bounds at the narrower (each
             * matrix still lies inside the experts' range); the TC MoE records take the 640 layout */
			experts_out->gate_up_shape.output_size =
				DS41F_TP4_SMALL_INNER;
			experts_out->down_shape.input_size =
				DS41F_TP4_SMALL_INNER;
		}
	}
	if (b->plan->exl3_bits[layer] != 0) {
		experts_out->weight_format = WEIGHT_FORMAT_EXL3;
		experts_out->gate_up_shape.bits_per_weight =
			b->plan->exl3_bits[layer];
		experts_out->down_shape.bits_per_weight =
			b->plan->exl3_bits[layer];
		const uint32_t product_slices =
			b->tensor_layer != NULL &&
					b->tensor_layer->exl3_shard != NULL ?
				b->tensor_layer->exl3_shard->product_slices :
			b->plan->limits.cmp_order == CMP_ORDER_RELAXED ?
				EXL3_RELAXED_SLICES :
				EXL3_PRODUCT_SLICES;
		experts_out->gate_up_slice_count =
			experts_out->down_slice_count = product_slices;
		return DS41F_OK;
	}
	result_status status = ds41f_slices_of(
		b, LINEAR_FAMILY_EXPERT_GATE_UP, &experts_out->gate_up_shape,
		&experts_out->gate_up_slice_count);
	return DS41F_THEN(ds41f_slices_of(b, LINEAR_FAMILY_EXPERT_DOWN,
					  &experts_out->down_shape,
					  &experts_out->down_slice_count));
}

/* The shared expert of layer for the rows the next launches take, its hidden vectors where the routed experts' were
 * (they are summed by then); its rows' codes where a quantization of those rows left them, or with is_chunk_quantized
 * where one quantization over the whole chunk did (linear_records.h: rows packed at their width from the first). */
MUST_CHECK static result_status shared_launches(struct ds41f_builder *b,
						uint32_t layer,
						bool is_chunk_quantized)
{
	const struct ds41f_tensor *const weights =
		b->plan->layer_tensors[layer];
	const uint64_t *const arrays = b->plan->cards[b->card].array_offset;
	struct linear_shared_expert_description shared = {
		.codes = DS41F_ARRAY(ROW_CODES),
		.exponents = DS41F_ARRAY(ROW_EXPONENTS),
		.operands = DS41F_ARRAY(ROW_OPERANDS),
		.gate_output = DS41F_ARRAY(SHARED_GATE),
		.up_output = DS41F_ARRAY(SHARED_UP),
		.hidden = DS41F_ARRAY(PAIR_HIDDEN),
		.hidden_codes = DS41F_ARRAY(PAIR_CODES),
		.hidden_exponents = DS41F_ARRAY(PAIR_EXPONENTS),
		.hidden_operands = DS41F_ARRAY(PAIR_OPERANDS),
		.output = DS41F_ARRAY(SHARED_OUTPUT),
		.row_count = b->rows,
		.activation_limit = SWIGLU_LIMIT
	};
	if (is_chunk_quantized) { /* one quantization took the chunk: its rows lie packed at its width from its first
                                   row (split_overlap: the half's, which begins chunk_first rows into the arrays) */
		const uint32_t into = b->first_row - b->chunk_first;
		shared.codes = (struct linear_range){
			arrays[DS41F_ARRAY_ROW_CODES] +
				b->chunk_first *
					ds41f_array_bytes[DS41F_ARRAY_ROW_CODES]
						.row_bytes +
				linear_codes_byte_count(into, DS41F_HIDDEN),
			linear_codes_byte_count(b->rows, DS41F_HIDDEN)
		};
		shared.exponents = (struct linear_range){
			arrays[DS41F_ARRAY_ROW_EXPONENTS] +
				b->chunk_first *
					ds41f_array_bytes
						[DS41F_ARRAY_ROW_EXPONENTS]
							.row_bytes +
				linear_exponents_byte_count(into, DS41F_HIDDEN),
			linear_exponents_byte_count(b->rows, DS41F_HIDDEN)
		};
		shared.operands = (struct linear_range){
			arrays[DS41F_ARRAY_ROW_OPERANDS] +
				b->chunk_first *
					ds41f_array_bytes[DS41F_ARRAY_ROW_OPERANDS]
						.row_bytes +
				linear_values_byte_count(into, DS41F_HIDDEN),
			linear_values_byte_count(b->rows, DS41F_HIDDEN)
		};
	}
	result_status status = ds41f_matrix_of(b, DS41F_WEIGHTS(SHARED_GATE),
					       DS41F_TRUNK, &shared.gate);
	status = DS41F_THEN(ds41f_matrix_of(b, DS41F_WEIGHTS(SHARED_UP),
					    DS41F_TRUNK, &shared.up));
	status = DS41F_THEN(ds41f_matrix_of(b, DS41F_WEIGHTS(SHARED_DOWN),
					    DS41F_TRUNK, &shared.down));
	const int32_t folds[EXPERT_PAUSE_MATRICES] = {
		DS41F_WEIGHTS(SHARED_GATE)->fold_reference,
		DS41F_WEIGHTS(SHARED_UP)->fold_reference,
		DS41F_WEIGHTS(SHARED_DOWN)->fold_reference
	};
	struct kernel_launch launches[LINEAR_SHARED_EXPERT_LAUNCH_COUNT];
	status = DS41F_THEN(linear_shared_expert_select_prepare(
		&shared, b->arena, &b->linear, &b->relaxed, b->arithmetic_class,
		b->plan->limits.cmp_order, folds, launches));
	if (b->plan->limits.official_reference != 0) {
		status = DS41F_THEN(linear_shared_expert_prepare(
			&shared, b->arena, &b->linear, launches));
		status = DS41F_THEN(ds41f_trunk_of(
			b, shared.codes, shared.exponents, shared.operands,
			DS41F_WEIGHTS(SHARED_GATE), shared.gate_output,
			DS41F_WEIGHTS(SHARED_UP), shared.up_output));
		status = DS41F_THEN(
			ds41f_put(b, &launches[LINEAR_SHARED_EXPERT_HIDDEN]));
		status = DS41F_THEN(
			ds41f_put(b, &launches[LINEAR_SHARED_EXPERT_QUANTIZE]));
		status = DS41F_THEN(ds41f_trunk_of(
			b, shared.hidden_codes, shared.hidden_exponents,
			shared.hidden_operands, DS41F_WEIGHTS(SHARED_DOWN),
			shared.output, NULL, ds41f_no_range));
		const struct ds41f_expert_join_description join = {
			DS41F_ARRAY(PAIR_OUTPUT), shared.output,
			DS41F_ARRAY(STREAMS),	  DS41F_ARRAY(POST),
			DS41F_ARRAY(COMB),	  DS41F_ARRAY(ROW_SUMS),
			DS41F_EXPERTS_PER_ROW,	  1
		};
		return DS41F_THEN(ds41f_expert_join_prepare(
			&join, b->arena, &b->fused, ds41f_slots(b, 1)));
	}
	const bool is_thin = !is_chunk_quantized;
	const bool exl3 = b->plan->exl3_bits[layer] != 0;
	const bool fused_down =
		is_thin && !exl3 && b->tensor_layer == NULL &&
		!ds41f_trunk_is_relaxed(b, DS41F_WEIGHTS(SHARED_DOWN));
	const uint32_t count =
		LINEAR_SHARED_EXPERT_LAUNCH_COUNT - (fused_down ? 1u : 0u);
	const struct linear_trunk_bfloat16_description wide_products[] = {
		{ .matrix = shared.gate,
		  .values = DS41F_ARRAY(HIDDEN),
		  .output = shared.gate_output,
		  .row_count = b->rows,
		  .segment_count = 1 },
		{ .matrix = shared.up,
		  .values = DS41F_ARRAY(HIDDEN),
		  .output = shared.up_output,
		  .row_count = b->rows,
		  .segment_count = 1 },
		{ .matrix = shared.down,
		  .values = shared.hidden,
		  .output = shared.output,
		  .row_count = b->rows,
		  .segment_count = 1 }
	};
	if (!is_thin && b->tensor_layer != NULL && ds41f_thin_forms_of(b) &&
	    ds41f_trunk_mma_wide_applies(b, &wide_products[0]) &&
	    ds41f_trunk_mma_wide_applies(b,
					 &wide_products[SHARED_DOWN_PRODUCT])) {
		/* --tp_trunk_mma, a BATCH step: the same from the chunk's BF16 rows by the _64 form */
		for (uint32_t matrix = 0;
		     matrix <
			     sizeof(wide_products) / sizeof(wide_products[0]) &&
		     status == DS41F_OK;
		     matrix++) {
			if (matrix == SHARED_DOWN_PRODUCT)
				status = ds41f_put(
					b,
					&launches[LINEAR_SHARED_EXPERT_HIDDEN]);
			struct kernel_launch *const launch = ds41f_slots(b, 1);
			status = launch == NULL ?
					 STATUS_INVALID_ARGUMENT :
					 DS41F_THEN(ds41f_trunk_mma_wide(
						 b, launch,
						 &wide_products[matrix]));
			if (matrix == 1u && status == DS41F_OK)
				ds41f_trunk_mma_pair_merge(
					b); /* gate and up (grouped forms) in one launch */
		}
		return status;
	}
	if (is_thin && b->tensor_layer != NULL &&
	    ds41f_trunk_mma_ready(b, &shared.gate.shape) &&
	    ds41f_trunk_mma_ready(b, &shared.up.shape) &&
	    ds41f_trunk_mma_ready(b, &shared.down.shape)) {
		/* --tp_trunk_mma: gate and up from the sublayer's BF16 rows, down from the hidden step's
		 * values, by
         * kernel_trunk_mma (GNmac's W8A16 class); the hidden step stays, its quantization goes */
		const struct linear_matrix *const matrices[] = { &shared.gate,
								 &shared.up,
								 &shared.down };
		const struct linear_range values[] = { DS41F_ARRAY(HIDDEN),
						       DS41F_ARRAY(HIDDEN),
						       shared.hidden };
		const struct linear_range outputs[] = { shared.gate_output,
							shared.up_output,
							shared.output };
		for (uint32_t matrix = 0;
		     matrix < sizeof(matrices) / sizeof(matrices[0]) &&
		     status == DS41F_OK;
		     matrix++) {
			if (matrices[matrix] == &shared.down)
				status = ds41f_put(
					b,
					&launches[LINEAR_SHARED_EXPERT_HIDDEN]);
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
			prepare_launch_fill(
				&launch, b->trunk_mma_kernels[0], 1, 1,
				TRUNK_MMA_THREADS, 0, &record,
				TRUNK_LINEAR_BFLOAT16_ARGUMENT_BYTES);
			ds41f_trunk_mma_swap(b, &launch,
					     &matrices[matrix]->shape);
			status = DS41F_THEN(ds41f_put(b, &launch));
			if (matrices[matrix] == &shared.up &&
			    status == DS41F_OK)
				ds41f_trunk_mma_pair_merge(
					b); /* gate and up in one launch */
		}
		return status;
	}
	/* a full program's decoder layer: the shared expert's wide products leave the rows the routed experts leave
	 * (the
     * cone's rows whose output at this layer nothing reads) */
	const uint32_t cone_layer =
		b->made->program.kind == PROGRAM_FULL &&
				layer >= DS41F_DECODER_FIRST_LAYER &&
				layer < DS41F_LAYERS ?
			layer :
			0u;
	for (uint32_t index = 0; index < count && status == DS41F_OK; index++) {
		if (index == LINEAR_SHARED_EXPERT_GATE_UP ||
		    index == LINEAR_SHARED_EXPERT_DOWN)
			status = ds41f_trunk_cone_at(b, &launches[index],
						     cone_layer);
		status = DS41F_THEN(ds41f_put(b, &launches[index]));
	}
	if (is_thin && exl3 && b->tensor_layer == NULL) {
		struct kernel_launch site[DS41F_HYPER_SITE_LAUNCH_COUNT];
		status = DS41F_THEN(ds41f_site_prepare(b, weights, true, site));
		return DS41F_THEN(ds41f_put(b, &site[DS41F_HYPER_SITE_POST]));
	}
	if (is_thin && b->tensor_layer == NULL) {
		const struct ds41f_expert_join_description join = {
			DS41F_ARRAY(PAIR_OUTPUT),
			DS41F_ARRAY(SHARED_OUTPUT),
			DS41F_ARRAY(STREAMS),
			DS41F_ARRAY(POST),
			DS41F_ARRAY(COMB),
			DS41F_ARRAY(ROW_SUMS),
			layer >= DS41F_DRAFT_FIRST_LAYER ?
				DS41F_DRAFT_EXPERTS_PER_ROW :
				DS41F_EXPERTS_PER_ROW,
			b->rows
		};
		const struct ds41f_expert_shared_join_description joined = {
			shared.down, shared.hidden_codes,
			shared.hidden_exponents, shared.hidden_operands, join
		};
		status = fused_down ?
				 DS41F_THEN(ds41f_expert_shared_join_prepare(
					 &joined, b->arena, &b->linear,
					 &b->fused, ds41f_slots(b, 1))) :
				 DS41F_THEN(ds41f_expert_join_prepare(
					 &join, b->arena, &b->fused,
					 ds41f_slots(b, 1)));
	}
	return status;
}

/* EXL3 activation writes an owner-major FP16 chunk directly into PAIR_OPERANDS; this gather only transposes bits.
 * exl3_wide_entry: the chunk's pairs (total_pairs) are gathered a slab (pairs of b->rows rows) at a time, slab `slab`
 * of the chunk's owner-major chunk: each rank reads only its own range, so its input base is its slab's part less
 * the slab's owner prefix, and the output is the slab's pairs of the pair-major output. */
MUST_CHECK static result_status
exl3_hidden_gather(struct ds41f_builder *b, struct linear_range input,
		   struct linear_range output, uint32_t pairs,
		   uint32_t total_pairs, uint32_t slab)
{
	const struct ds41f_tp2_layer_plan *const tensor = b->tensor_layer;
	if (tensor == NULL || tensor->exl3_shard == NULL ||
	    !tensor->exact_output_shards ||
	    tensor->rank_count != DS41F_TP4_RANKS ||
	    tensor->original_code_wire || tensor->bf16_ring ||
	    b->unequal_ring_kernel == BACKEND_NO_KERNEL || b->rows == 0 ||
	    b->rows > LINEAR_MAXIMUM_ROWS || pairs == 0)
		return STATUS_UNSUPPORTED;
	const uint64_t bytes =
		(uint64_t)pairs * DS41F_EXPERT_INNER * EXL3_HALF_BYTES;
	if (total_pairs < pairs ||
	    (total_pairs != pairs && total_pairs % pairs != 0) ||
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
	output.byte_count =
		output.byte_count >= (uint64_t)(slab + 1u) * bytes ? bytes : 0u;
	const struct ordered_push_arguments *const channel =
		&tensor->collectives[0].arguments;
	const uint64_t bank = tensor->ring_offsets[0] +
			      ds41f_ring_bank_start(tensor->ring_bank_bytes[0],
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
				      DS41F_TP2_SITES + 1u, b->arena))
		return STATUS_INVALID_ARGUMENT;
	struct exact_ring_unequal_arguments arguments = {
		.input_offset = input.byte_offset,
		.output_offset = output.byte_offset,
		.bank_offset = bank,
		.peer_table_offset = channel->peer_table_offset,
		.peer_bank_delta = bank - channel->region_offset,
		.deadline_cycles = DS41F_TP2_DEADLINE_CYCLES,
		.vectors = pairs,
		.rank = tensor->rank
	};
	uint32_t prefix = 0;
	for (uint32_t rank = 0; rank < DS41F_TP4_RANKS; rank++) {
		struct bb_exl3_output_shard_plan shard;
		const result_status status = bb_exl3_output_shard_prepare(
			DS41F_HIDDEN, DS41F_EXPERT_INNER,
			tensor->exl3_shard->gate_up.bits_per_weight,
			DS41F_TP4_RANKS, rank, tensor->exl3_shard->order,
			&shard);
		if (status != DS41F_OK)
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
	return ds41f_put(b, &launch);
}

/* Every product keeps its full K chain. The only exchanges are original FP16 operands and ordered FP32 channels. */
/* split_overlap mode 7: the launches of a joint TC prefill call with the halves' route (exl3_tcp half_rows) a phase
 * puts: all of a call without halves, its route, gate/up and retile, the first half's part (the whole call's down
 * operands, the half's down GEMM and combine) or the second's (down GEMM, combine). */
enum tcp_emit {
	TCP_EMIT_ALL,
	TCP_EMIT_FRONT,
	TCP_EMIT_FIRST_HALF,
	TCP_EMIT_SECOND_HALF
};

MUST_CHECK static result_status
exl3_routed_launches(struct ds41f_builder *b, uint32_t layer,
		     const struct linear_routed_experts_description *experts)
{
	/* a PRO (class 120, emulated or native) runs whole matrices only: every tensor-parallel form is the CMPs' */
	const bool is_pro = b->arithmetic_class == ARITHMETIC_CLASS_SM_120 ||
			    b->arithmetic_class ==
				    ARITHMETIC_CLASS_SM_120_NATIVE;
	if ((b->arithmetic_class != ARITHMETIC_CLASS_SM_80 &&
	     (!is_pro || b->tensor_layer != NULL)) ||
	    layer >= DS41F_LAYERS)
		return STATUS_UNSUPPORTED;
	const uint32_t pairs = b->rows * experts->pairs_per_row;
	const uint64_t scratch = b->plan->cards[b->card].exl3_scratch_offset;
	const uint64_t scratch_bytes =
		exl3_scratch_byte_count(pairs, DS41F_HIDDEN);
	if (scratch > b->arena || scratch_bytes > b->arena - scratch)
		return STATUS_INVALID_ARGUMENT;
	const uint64_t products = scratch + (uint64_t)pairs * DS41F_HIDDEN *
						    EXL3_PROJECTIONS *
						    EXL3_HALF_BYTES;
	const struct exl3_expert_record whole = {
		.jobs_offset = experts->jobs.byte_offset,
		.input_offset = DS41F_ARRAY(HIDDEN).byte_offset,
		.operands_offset = scratch,
		.products_offset = products,
		.hidden_offset = experts->hidden.byte_offset,
		.output_offset = products,
		.input_size = DS41F_HIDDEN,
		.inner_size = experts->gate_up_shape.output_size,
		.pair_count = pairs,
		.pairs_per_row = experts->pairs_per_row,
		.job_count = experts->job_count,
		.bits_per_weight = b->plan->exl3_bits[layer],
		.activation_limit = SWIGLU_LIMIT
	};
	if (b->tensor_layer == NULL || !b->tensor_layer->exact_output_shards) {
		/* Optional inner sharding changes the local K chain; the later BF16 ring sums full-output
		 * partials.
         * This deliberately differs from the default full-K/lossless recipe and needs model quality gates. */
		if (b->tensor_layer != NULL &&
		    (!b->tensor_layer->bf16_ring ||
		     b->tensor_layer->original_code_wire ||
		     b->tensor_layer->rank_count != DS41F_TP4_RANKS))
			return STATUS_UNSUPPORTED;
		/* exl3_expert_rotation: the mixed forms at 640 (exl3_tc_records.h) */
		const uint32_t rotation = expert_rotation_of(b, layer);
		const uint32_t intermediate =
			rotation != 0 ? DS41F_EXL3_TC_MIXED_WIDTH :
					experts->gate_up_shape.output_size;
		if (b->tensor_layer != NULL && b->tensor_layer->exl3_tc_moe &&
		    b->rows <= ds41f_tc_most_rows(b)) {
			/* exl3_tc_moe: GNmac's tensor-core routed experts from the job table, each row's route-weighted
			 * sum FP32
             * into the row sums the BF16 ring adds to the shared expert's (exl3_tc.h); a tail program's
             * workspace
             * its own (ds41f_tp_stage.h tail_overlay) */
			const struct linear_range workspace =
				ds41f_tail_program(b) ?
					b->tensor_layer->exl3_tail_workspace :
					b->tensor_layer->exl3_tc_workspace;
			const struct exl3_tc_record tc = {
				.jobs_offset = experts->jobs.byte_offset,
				.hidden_offset =
					DS41F_ARRAY(HIDDEN).byte_offset,
				.workspace_offset = workspace.byte_offset,
				.output_offset =
					experts->row_output.byte_offset,
				.row_count = b->rows,
				.job_capacity = experts->job_count,
				.intermediate = intermediate,
				.bits_x2 =
					exl3_bits_x2(b->plan->exl3_bits[layer]),
				.activation_limit = SWIGLU_LIMIT,
				.rotation = rotation,
				.shared_sum = b->shared_sum_requested ? 1u : 0u,
				.shared_offset =
					b->shared_sum_requested ?
						DS41F_ARRAY(SHARED_OUTPUT)
							.byte_offset :
						0u
			};
			b->shared_sum_fused = b->shared_sum_requested;
			struct kernel_launch tc_launches[EXL3_TC_LAUNCHES];
			result_status tc_status =
				workspace.byte_count < exl3_tc_workspace_bytes(
							       (uint64_t)b->rows *
								       DS41F_EXL3_TC_PAIRS_PER_ROW,
							       tc.intermediate) ||
						experts->pairs_per_row !=
							DS41F_EXL3_TC_PAIRS_PER_ROW ?
					STATUS_INVALID_ARGUMENT :
				ds41f_batch_thin_bits_of(b) &&
						b->batch_tc_paired != 0 &&
						b->rows >= b->batch_tc_paired ?
					/* the short forms' partials, two slices an item */
					exl3_tc_prepare_paired(
						&tc, b->arena, b->backend,
						b->cards[b->card],
						tc_launches) :
					ds41f_batch_thin_bits_of(
						b) ? /* the short forms at every row count: each row the thin steps'
						      * sums */
						exl3_tc_prepare_short(
							&tc, b->arena,
							b->backend,
							b->cards[b->card],
							tc_launches) :
						exl3_tc_prepare(
							&tc, b->arena,
							b->backend,
							b->cards[b->card],
							tc_launches);
			/* A persistent warp owns complete product items: the grid changes their assignment, never their
             * arithmetic. Restrict the override to decode programs; the prefill TC geometry stays independent. */
			const uint32_t product_blocks =
				b->tp_description == NULL ?
					0u :
					b->tp_description->stages[b->card]
						->description
						.exl3_tc_product_blocks;
			if (tc_status == DS41F_OK && product_blocks != 0 &&
			    (b->made->program.kind == PROGRAM_THIN ||
			     b->made->program.kind == PROGRAM_BATCH)) {
				tc_launches[TC_GATE_UP_PRODUCT_LAUNCH].grid[0] =
					product_blocks;
				tc_launches[TC_DOWN_PRODUCT_LAUNCH].grid[0] =
					product_blocks;
			}
			for (uint32_t launch = 0;
			     launch < EXL3_TC_LAUNCHES && tc_status == DS41F_OK;
			     launch++)
				tc_status = ds41f_put(b, &tc_launches[launch]);
			return tc_status;
		}
		if (b->tensor_layer != NULL && b->tensor_layer->exl3_tc_moe &&
		    b->rows <= EXL3_TCP_CHUNK_ROWS &&
		    b->tensor_layer->exl3_tcp_workspace.byte_count != 0) {
			/* exl3_tc_moe, a chunk of more than 64 rows: GNmac's prefill MoE from the wide router's
			 * picks */
			const struct exl3_tcp_record tcp = {
				.picks_offset =
					DS41F_ARRAY(ROUTER_PICKS).byte_offset,
				.expert_offsets_offset =
					b->plan->cards[b->card]
						.expert_table_offset[layer],
				.hidden_offset =
					DS41F_ARRAY(HIDDEN).byte_offset,
				.workspace_offset =
					b->tensor_layer->exl3_tcp_workspace
						.byte_offset,
				.output_offset =
					experts->row_output.byte_offset,
				.row_count = b->rows,
				.intermediate = intermediate,
				.bits_x2 =
					exl3_bits_x2(b->plan->exl3_bits[layer]),
				.activation_limit = SWIGLU_LIMIT,
				.rotation = rotation,
				.half_rows = b->tcp_emit != TCP_EMIT_ALL ?
						     b->half_rows :
						     0u,
				/* a full program's decoder layer: the rows the engine's prefill cone marks past it take
				 * no
			 * route */
				.step_offset = DS41F_ARRAY(STEP).byte_offset,
				.cone_layer =
					b->made->program.kind == PROGRAM_FULL &&
							layer >=
								DS41F_DECODER_FIRST_LAYER &&
							layer < DS41F_LAYERS ?
						layer :
						0u,
				.two_blocks =
					b->tp_description != NULL &&
							b->tp_description
								->stages[b->card]
								->description
								.tcp_two_blocks ?
						1u :
						0u
			};
			struct kernel_launch
				tcp_launches[EXL3_TCP_MOST_LAUNCHES];
			uint32_t tcp_count = 0;
			result_status tcp_status =
				b->tensor_layer->exl3_tcp_workspace.byte_count <
						exl3_tcp_workspace_bytes(
							b->rows,
							tcp.intermediate) ?
					STATUS_INVALID_ARGUMENT :
					exl3_tcp_prepare(&tcp, b->arena,
							 b->backend,
							 b->cards[b->card],
							 tcp_launches,
							 &tcp_count);
			/* mode 7: the route, gate/up and retile, then the halves' parts (exl3_tc_records.h) */
			const uint32_t front =
				tcp.half_rows != 0 ?
					tcp_count - EXL3_TCP_PART_LAUNCHES -
						EXL3_TCP_SECOND_PART_LAUNCHES :
					tcp_count;
			const uint32_t first =
				b->tcp_emit == TCP_EMIT_FIRST_HALF ?
					front :
				b->tcp_emit == TCP_EMIT_SECOND_HALF ?
					front + EXL3_TCP_PART_LAUNCHES :
					0;
			const uint32_t end =
				b->tcp_emit == TCP_EMIT_FRONT ?
					front :
				b->tcp_emit == TCP_EMIT_FIRST_HALF ?
					front + EXL3_TCP_PART_LAUNCHES :
					tcp_count;
			if (tcp_status == DS41F_OK &&
			    (tcp.half_rows == 0) !=
				    (b->tcp_emit == TCP_EMIT_ALL))
				tcp_status =
					STATUS_INVALID_ARGUMENT; /* halves need a split chunk's first half */
			for (uint32_t launch = first;
			     launch < end && tcp_status == DS41F_OK; launch++)
				tcp_status =
					ds41f_put(b, &tcp_launches[launch]);
			return tcp_status;
		}
		if (rotation !=
		    0) /* the other forms take every expert at the rank's one width */
			return STATUS_UNSUPPORTED;
		struct kernel_launch launches[EXL3_EXPERT_LAUNCHES];
		/* exl3_fast_products: kernel_exl3_wide's transforms and products, the same words (exl3.h) */
		const bool fast = b->tensor_layer != NULL &&
				  b->tensor_layer->exl3_fast_products;
		/* the relaxed order (--cmp_order) is the CMPs' */
		result_status status =
			b->plan->limits.cmp_order == CMP_ORDER_RELAXED &&
					!is_pro ?
				exl3_experts_relaxed_prepare(
					&whole, b->arena, b->backend,
					b->cards[b->card], b->arithmetic_class,
					launches) :
			fast ? exl3_experts_fast_prepare(&whole, b->arena,
							 b->backend,
							 b->cards[b->card],
							 b->arithmetic_class,
							 launches) :
			       exl3_experts_prepare(&whole, b->arena,
						    b->backend,
						    b->cards[b->card],
						    b->arithmetic_class,
						    launches);
		for (uint32_t pass = 0;
		     pass < EXL3_EXPERT_LAUNCHES && status == DS41F_OK; pass++)
			status = ds41f_put(b, &launches[pass]);
		const struct expert_row_sum_arguments sum = {
			products, experts->row_output.byte_offset, DS41F_HIDDEN,
			experts->pairs_per_row
		};
		/* the row sum is float32 adds, no product: kernel_exl3's build of it computes the same on every
		 * class */
		struct kernel_launch launch;
		status = DS41F_THEN(bb_exl3_row_sum_prepare(
			&sum, b->rows, b->arena, b->backend, b->cards[b->card],
			&launch));
		return DS41F_THEN(ds41f_put(b, &launch));
	}
	const struct ds41f_tp2_layer_plan *const tensor = b->tensor_layer;
	const struct bb_exl3_output_shard_plan *const shard =
		tensor->exl3_shard;
	/* exl3_wide_entry: a chunk of more than 8 rows in one call; its gathers a slab of 8 rows at a time */
	const bool wide = b->rows > LINEAR_MAXIMUM_ROWS;
	if (shard == NULL || !tensor->exact_output_shards ||
	    tensor->bf16_ring || tensor->original_code_wire ||
	    tensor->rank_count != DS41F_TP4_RANKS ||
	    shard->rank != tensor->rank ||
	    (wide &&
	     (!tensor->exl3_wide_entry || b->rows % LINEAR_MAXIMUM_ROWS != 0)))
		return STATUS_UNSUPPORTED;
	const uint64_t local_operand =
		experts->hidden_operands.byte_offset +
		(uint64_t)pairs * shard->gate_up.output_begin * EXL3_HALF_BYTES;
	struct kernel_launch launches[BB_EXL3_PASSES];
	result_status status = DS41F_OK;
	for (uint32_t pass = 0; pass < BB_EXL3_PASSES && status == DS41F_OK;
	     pass++) {
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
			.product_tiles = tensor->exl3_fast_products ?
						 BB_EXL3_PRODUCT_EIGHT_TILES :
						 0u
		};
		status = bb_exl3_pass_prepare(&description, b->backend,
					      b->cards[b->card],
					      &launches[pass]);
	}
	for (uint32_t pass = 0;
	     pass <= BB_EXL3_ACTIVATE_PASS && status == DS41F_OK; pass++)
		status = ds41f_put(b, &launches[pass]);
	const uint32_t rows = b->rows,
		       slab_rows = wide ? LINEAR_MAXIMUM_ROWS : rows;
	const uint32_t slab_pairs = slab_rows * experts->pairs_per_row;
	b->rows = slab_rows; /* the gathers' banks are a slab's */
	for (uint32_t slab = 0; slab < rows / slab_rows && status == DS41F_OK;
	     slab++)
		status = exl3_hidden_gather(b, experts->hidden_operands,
					    tensor->exact_hidden, slab_pairs,
					    pairs, slab);
	b->rows = rows;
	for (uint32_t pass = BB_EXL3_DOWN_PASS;
	     pass < BB_EXL3_PASSES && status == DS41F_OK; pass++)
		status = ds41f_put(b, &launches[pass]);
	const struct expert_row_sum_arguments sum = {
		products,
		experts->row_output.byte_offset +
			(uint64_t)tensor->rank * b->rows *
				shard->down.output_count * sizeof(float),
		shard->down.output_count, experts->pairs_per_row
	};
	struct kernel_launch launch;
	status = DS41F_THEN(
		bb_exl3_row_sum_prepare(&sum, b->rows, b->arena, b->backend,
					b->cards[b->card], &launch));
	status = DS41F_THEN(ds41f_put(b, &launch));
	/* the row sums: owner-major [rank][rows][own outputs]; slab s of each rank's part to rows 8 s .. of
	 * exact_sums */
	const uint64_t own_bytes =
		(uint64_t)shard->down.output_count * sizeof(float);
	const uint32_t slab_bytes =
		slab_rows * DS41F_HIDDEN * (uint32_t)sizeof(float);
	b->rows = slab_rows;
	for (uint32_t slab = 0; slab < rows / slab_rows && status == DS41F_OK;
	     slab++) {
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
		status = ds41f_exact_gather(b, 1, input, output, slab_bytes,
					    slab_rows);
	}
	b->rows = rows;
	return status;
}

/* The original products are owner-major over the chunk. Each gather still uses its immutable eight-row bank;
 * only this rank's owner range is read, so a virtual input base selects its slab without copying the products. */
MUST_CHECK static result_status
original_gather_slabs(struct ds41f_builder *b, uint32_t site,
		      struct linear_range input, struct linear_range output,
		      uint32_t vectors_per_row, uint32_t width,
		      uint32_t element_bytes)
{
	const struct ds41f_tp2_layer_plan *const tensor = b->tensor_layer;
	const uint32_t rows = b->rows;
	const uint64_t row_bytes =
		(uint64_t)vectors_per_row * width * element_bytes;
	if (tensor == NULL || !tensor->original_wide_experts ||
	    !tensor->exact_output_shards || tensor->bf16_ring ||
	    tensor->original_code_wire ||
	    tensor->rank_count != DS41F_TP4_RANKS ||
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
	result_status status = DS41F_OK;
	for (uint32_t at = 0; at < rows && status == DS41F_OK; at += count) {
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
		status = ds41f_exact_gather(b, site, from, to, bytes,
					    count * vectors_per_row);
	}
	b->rows = rows;
	return status;
}

MUST_CHECK static result_status
exact_routed_launches(struct ds41f_builder *b,
		      const struct linear_routed_experts_description *experts)
{
	const bool wide = b->rows > LINEAR_MAXIMUM_ROWS;
	if (wide && (!b->tensor_layer->original_wide_experts ||
		     b->tensor_layer->original_code_wire))
		return STATUS_UNSUPPORTED;
	struct kernel_launch launches[LINEAR_ROUTED_EXPERTS_LAUNCH_COUNT];
	result_status status = linear_routed_experts_prepare(
		experts, b->arena, &b->linear, launches);
	if (status != DS41F_OK)
		return status;
	const uint32_t ranks = b->tensor_layer->rank_count,
		       rank = b->tensor_layer->rank;
	const uint32_t hidden_width =
		experts->gate_up_shape.output_size / ranks;
	const uint32_t output_width = experts->down_shape.output_size / ranks;
	const uint32_t pairs = b->rows * experts->pairs_per_row;
	struct expert_gate_up_arguments gate;
	struct activation_quantize_arguments ds41f_quantize;
	struct expert_down_arguments down;
	struct expert_row_sum_arguments sum;
	memcpy(&gate, launches[LINEAR_ROUTED_EXPERTS_GATE_UP].argument,
	       sizeof(gate));
	memcpy(&ds41f_quantize,
	       launches[LINEAR_ROUTED_EXPERTS_QUANTIZE].argument,
	       sizeof(ds41f_quantize));
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
	ds41f_quantize.values_offset =
		b->tensor_layer->exact_hidden.byte_offset;
	memcpy(launches[LINEAR_ROUTED_EXPERTS_GATE_UP].argument, &gate,
	       sizeof(gate));
	memcpy(launches[LINEAR_ROUTED_EXPERTS_QUANTIZE].argument,
	       &ds41f_quantize, sizeof(ds41f_quantize));
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
		if (ranks != DS41F_TP4_RANKS ||
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
		status = ds41f_put(b, &launches[LINEAR_ROUTED_EXPERTS_GATE_UP]);
		status = DS41F_THEN(ds41f_put(b, &producer_launch));
		status = DS41F_THEN(ds41f_exact_gather(
			b, 0, packed, b->tensor_layer->exact_hidden, wire_bytes,
			pairs));
		const struct quantized_wire_arguments wire = {
			b->tensor_layer->exact_hidden.byte_offset,
			ds41f_quantize.codes_offset,
			ds41f_quantize.exponents_offset,
			ds41f_quantize.operands_offset,
			pairs,
			hidden_width,
			pitch,
			0
		};
		struct kernel_launch decode;
		prepare_launch_fill(
			&decode, b->wire_decode_kernel,
			(DS41F_EXPERT_INNER + LINEAR_ELEMENT_THREADS - 1u) /
				LINEAR_ELEMENT_THREADS,
			pairs, LINEAR_ELEMENT_THREADS, 0, &wire, sizeof(wire));
		status = DS41F_THEN(ds41f_put(b, &decode));
	} else {
		status = ds41f_put(b, &launches[LINEAR_ROUTED_EXPERTS_GATE_UP]);
		status = wide ? DS41F_THEN(original_gather_slabs(
					b, 0, experts->hidden,
					b->tensor_layer->exact_hidden,
					experts->pairs_per_row,
					DS41F_EXPERT_INNER,
					LINEAR_BFLOAT16_BYTES)) :
				DS41F_THEN(ds41f_exact_gather(
					b, 0, experts->hidden,
					b->tensor_layer->exact_hidden,
					pairs * DS41F_EXPERT_INNER *
						LINEAR_BFLOAT16_BYTES,
					pairs));
		status = DS41F_THEN(ds41f_put(
			b, &launches[LINEAR_ROUTED_EXPERTS_QUANTIZE]));
	}
	status =
		DS41F_THEN(ds41f_put(b, &launches[LINEAR_ROUTED_EXPERTS_DOWN]));
	status = DS41F_THEN(
		ds41f_put(b, &launches[LINEAR_ROUTED_EXPERTS_ROW_SUM]));
	return wide ? DS41F_THEN(original_gather_slabs(
			      b, 1, experts->row_output,
			      b->tensor_layer->exact_sums, 1u, DS41F_HIDDEN,
			      LINEAR_SUM_BYTES)) :
		      DS41F_THEN(ds41f_exact_gather(b, 1, experts->row_output,
						    b->tensor_layer->exact_sums,
						    b->rows * DS41F_HIDDEN *
							    LINEAR_SUM_BYTES,
						    b->rows));
}

/* Routed selection changes only the product launches: quantization and the ordered row sum keep their records. */
MUST_CHECK static result_status
routed_launches(struct ds41f_builder *b, uint32_t layer,
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
		for (uint32_t index = 0; status == DS41F_OK &&
					 index < LINEAR_ROUTED_EXPERTS_ROW_SUM;
		     index++)
			status = ds41f_put(b, &launches[index]);
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
					 DS41F_ARRAY(ROUTER_TILES).byte_offset,
					 (uint64_t)ROUTER_WIDE_TILES(b->rows) *
						 WIDE_EXPERT_TILE_BYTES } :
				 ds41f_no_range,
		.pairs = is_wide ? DS41F_ARRAY(ROUTER_PAIRS) : ds41f_no_range,
		.tile_count = is_wide ? ROUTER_WIDE_TILES(b->rows) : 0,
	};
	result_status status = expert_select_prepare(
		&description, b->arena, &b->linear, &b->expert,
		b->arithmetic_class, b->plan->limits.cmp_order,
		b->plan->expert_folds_admitted[layer], launches, &count);
	uint32_t first = 0;
	if (!is_wide && b->arithmetic_class == ARITHMETIC_CLASS_SM_120) {
		const struct linear_range arrivals = {
			b->plan->cards[b->card].producer_arrivals_offset,
			(uint64_t)experts->job_count *
				experts->gate_up_shape.output_size /
				LINEAR_VALUES_PER_BLOCK * sizeof(uint32_t)
		};
		status = DS41F_THEN(expert_producer_prepare(
			experts, arrivals, b->arena, &b->linear, &b->fused,
			ds41f_slots(b, 1)));
		first = LINEAR_ROUTED_EXPERTS_DOWN;
	}
	/* The thin join consumes pair outputs after the shared expert, and stores the exact gate's row sums. */
	count -= !is_wide && b->tensor_layer == NULL && count != 0 ? 1u : 0u;
	for (uint32_t index = first; index < count && status == DS41F_OK;
	     index++) {
		status = ds41f_put(b, &launches[index]);
	}
	return status;
}

/* The FFN site (2.1 launches 21 to 34, REF:989-993, 811-904), *first_launch where an expert pause divides it. */
MUST_CHECK result_status ds41f_ffn_launches(struct ds41f_builder *b,
					    uint32_t layer,
					    uint32_t *first_launch)
{
	const struct ds41f_tensor *const weights =
		b->plan->layer_tensors[layer];
	const void *const table = expert_table_of(b, layer);
	const bool is_draft = layer >= DS41F_DRAFT_FIRST_LAYER;
	const uint32_t picks = is_draft ? DS41F_DRAFT_EXPERTS_PER_ROW :
					  DS41F_EXPERTS_PER_ROW;
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
	const struct ds41f_router_description router = {
		&router_experts,
		table,
		DS41F_ARRAY(STEP),
		DS41F_ARRAY(ROUTER_SCORES),
		DS41F_WEIGHTS(ROUTER_BIAS)->stored,
		{ b->plan->cards[b->card].expert_table_offset[layer],
		  DS41F_EXPERT_OFFSETS_BYTES },
		DS41F_ARRAY(AFFINITIES),
		is_draft ? LINEAR_ORDER_FIXED : b->plan->limits.linear_order,
		is_draft ? DS41F_DRAFT_EXPERTS : DS41F_EXPERTS,
		picks
	};
	struct kernel_launch official_site[DS41F_HYPER_SITE_LAUNCH_COUNT];
	const bool official_hc = b->plan->limits.official_hc_rows != 0;
	const bool separate_router = is_draft || b->vision_active ||
				     official_hc;
	const bool exact_shards = b->tensor_layer != NULL &&
				  b->tensor_layer->exact_output_shards;
	/* graph_branches, a thin TC-MoE layer: the shared expert first as a side branch of the step graph (it reads
	 * only
     * the front's rows), the routed experts resuming behind the front, the ring joining both (GNmac's shared experts
     * on an auxiliary stream; the same launches, the same bits) */
	const bool side_shared =
		b->tensor_layer != NULL &&
		b->tensor_layer->graph_branches != 0 && !exact_shards &&
		b->tensor_layer->exl3_tc_moe &&
		b->plan->exl3_bits[layer] != 0 &&
		b->rows <= EXL3_TC_MOST_ROWS && b->slab_count == 1u;
	/* front_branch with the shared side branch: the FFN front as the input front and the separate router (the FFN
     * front's scores and select, the same bits: ds41f_site_front_records.h), the coefficient front (the FFN mix
     * project, finish and Sinkhorn) a second side run of one launch behind the shared expert's, joined with it */
	const bool ffn_branch = b->front_branch >= FRONT_BRANCH_BOTH &&
				side_shared && !separate_router &&
				b->plan->host_expert_count[layer] == 0 &&
				!b->tensor_layer->experts_partitioned;
	status = ffn_branch  ? DS41F_THEN(ds41f_site_front_of(
				       b, weights, true, NULL,
				       DS41F_SITE_FRONT_KIND_INPUT)) :
		 official_hc ? DS41F_THEN(ds41f_site_begin(b, weights, true,
							   official_site)) :
			       DS41F_THEN(ds41f_site_front(
				       b, weights, true,
				       separate_router ? NULL : &router));
	status = official_hc ? DS41F_THEN(ds41f_quantize(b, DS41F_ARRAY(HIDDEN),
							 DS41F_HIDDEN)) :
			       status;
	/* draft_bf16_operands: the draft FFN front's rows carried again, for the routed experts */
	status = is_draft && b->draft_carry_kernel != BACKEND_NO_KERNEL ?
			 DS41F_THEN(ds41f_quantize(b, DS41F_ARRAY(HIDDEN),
						   DS41F_HIDDEN)) :
			 status;
	if (separate_router || ffn_branch) {
		if (b->plan->limits.official_reference != 0) {
			const struct ds41f_official_router_projection_description
				projection = { DS41F_WEIGHTS(ROUTER)->stored,
					       DS41F_ARRAY(HIDDEN),
					       DS41F_ARRAY(ROUTER_SCORES), 1 };
			status = DS41F_THEN(
				ds41f_official_router_projection_prepare(
					&projection, b->arena,
					b->official.router_projection,
					ds41f_slots(b, 1)));
		} else {
			status = DS41F_THEN(ds41f_dense(
				b, DS41F_WEIGHTS(ROUTER), DS41F_ARRAY(HIDDEN),
				DS41F_ARRAY(ROUTER_SCORES), DS41F_ELEMENT_F32,
				0, 1.0f));
		}
		status =
			b->vision_active && !is_draft ?
				DS41F_THEN(ds41f_image_router_select_prepare(
					&router,
					b->plan->vision.router_bias[layer],
					b->arena, &b->router, &b->image_routing,
					ds41f_slots(b, 1))) :
				DS41F_THEN(ds41f_router_select_prepare(
					&router, b->arena, &b->router,
					ds41f_slots(b, 1)));
		if (b->plan->limits.official_reference != 0)
			status = DS41F_THEN(ds41f_official_router_order_prepare(
				&router, b->arena, b->official.router_order,
				ds41f_slots(b, 1)));
	}
	if (status == DS41F_OK && b->plan->host_expert_count[layer] != 0) {
		status = expert_pause_add(b, layer, *first_launch);
		*first_launch = b->made->cards[b->card].launch_count;
	}
	if (b->tensor_layer != NULL && b->tensor_layer->experts_partitioned) {
		const struct ep_filter_description filter = {
			DS41F_ARRAY(JOBS), DS41F_ARRAY(PAIR_HIDDEN),
			DS41F_ARRAY(PAIR_OUTPUT),
			b->tensor_layer->unowned_gate_offset, b->rows
		};
		status = DS41F_THEN(ep_filter_prepare(
			&filter, b->arena, b->tensor_ep_filter_kernel,
			ds41f_slots(b, 1)));
	}
	const uint32_t first_side = b->made->cards[b->card].launch_count;
	status = side_shared ? DS41F_THEN(shared_launches(b, layer, false)) :
			       status;
	const uint32_t coefficients =
		b->made->cards[b->card]
			.launch_count; /* ffn_branch: the coefficient front */
	status = ffn_branch ? DS41F_THEN(ds41f_site_front_of(
				      b, weights, true, NULL,
				      DS41F_SITE_FRONT_KIND_COEFFICIENTS)) :
			      status;
	const uint32_t first_routed = b->made->cards[b->card].launch_count;
	/* the shared expert a side branch before the routed experts: their combine may add the routed sums into its
	 * rows
     * (shared_sum; the combine then joins the branch and the ring sums those rows in place) */
	b->shared_sum_requested = side_shared && b->tensor_layer->bf16_ring &&
				  !b->tensor_layer->launches_untrimmed &&
				  b->rows <= b->tensor_layer->ring_rows;
	b->shared_sum_fused = false;
	status = b->plan->exl3_bits[layer] != 0 ?
			 DS41F_THEN(exl3_routed_launches(b, layer, &experts)) :
		 exact_shards ?
			 DS41F_THEN(exact_routed_launches(b, &experts)) :
			 DS41F_THEN(routed_launches(b, layer, &experts, false));
	const bool shared_sum = b->shared_sum_fused;
	b->shared_sum_requested = b->shared_sum_fused = false;
	status = side_shared ? status :
			       DS41F_THEN(shared_launches(b, layer, false));
	const uint32_t first_join = b->made->cards[b->card].launch_count;
	if (b->tensor_layer != NULL && !b->ffn_post_deferred) {
		struct kernel_launch site[DS41F_HYPER_SITE_LAUNCH_COUNT];
		const struct ds41f_tp2_layer_plan *const tensor =
			b->tensor_layer;
		const struct linear_range sums = DS41F_ARRAY(ROW_SUMS);
		if (exact_shards) {
			b->arrays[DS41F_ARRAY_ROW_SUMS] = tensor->exact_sums;
			b->tensor_layer = NULL;
		} else {
			status = DS41F_THEN(ds41f_tensor_complete(
				b, 1, DS41F_ARRAY(SHARED_OUTPUT),
				shared_sum ? (struct linear_range){ 0, 0 } :
					     DS41F_ARRAY(ROW_SUMS)));
		}
		if (side_shared && status == DS41F_OK) {
			const uint32_t end =
				b->made->cards[b->card].launch_count;
			/* the branch joins at the ring, or with shared_sum at the combine (the routed launches'
			 * last) */
			const uint32_t join = shared_sum ? first_join - 1u :
							   first_join;
			if (first_side == first_routed ||
			    first_routed >= join || first_join == end)
				return STATUS_INVALID_ARGUMENT;
			for (uint32_t launch = first_side;
			     launch < first_routed; launch++)
				b->launches[b->card][launch].branch =
					LAUNCH_SIDE;
			if (ffn_branch) /* a side run of its own behind the same fork */
				b->launches[b->card][coefficients].branch =
					LAUNCH_SIDE_MORE;
			b->launches[b->card][first_routed].branch =
				LAUNCH_RESUME;
			b->launches[b->card][join].branch = LAUNCH_JOIN;
		}
		status = DS41F_THEN(ds41f_site_prepare(b, weights, true, site));
		b->tensor_layer = tensor;
		b->arrays[DS41F_ARRAY_ROW_SUMS] = sums;
		status = DS41F_THEN(ds41f_put(b, &site[DS41F_HYPER_SITE_POST]));
	}
	return status;
}

/* batch_draft_wide: a draft layer's FFN sum (the shared expert's rows plus the routed sums, BF16 rings of
 * up to
 * ring_rows rows) and post over the rows the next launches take, which ffn_launches left a slab at a time
 * (ffn_post_deferred): a ring's sums are element by element, a post's row by row, so the same bits. */
MUST_CHECK result_status ds41f_ffn_post(struct ds41f_builder *b, uint32_t layer)
{
	struct kernel_launch site[DS41F_HYPER_SITE_LAUNCH_COUNT];
	result_status status = ds41f_tensor_complete_slabs(
		b, 1, DS41F_ARRAY(SHARED_OUTPUT), DS41F_ARRAY(ROW_SUMS));
	status = DS41F_THEN(ds41f_site_prepare(b, b->plan->layer_tensors[layer],
					       true, site));
	return DS41F_THEN(ds41f_put(b, &site[DS41F_HYPER_SITE_POST]));
}

/* batch_draft_wide 2, a draft layer's FFN site over a batched draft's rows (the builder a chunk of them, its
 * slabs'
 * views each a slab block): over all rows the mix and finish (site_begin); a slab at a time the router's scores,
 * the
 * rows' quantization (as the thin site front quantizes them), the draft's selection and its routed experts (the
 * narrow form of 8 rows); over all rows the shared expert (the _64 form from the BF16 rows), the TP sum
 * and the
 * post. The scores and the codes stay a slab's: a launch over the chunk lays its rows out packed at its width (the
 * router's 128 scores, a row's 5,120 codes), not at the arrays' rows (384 scores, 8,192 codes) at which a slab's
 * launches read them. Each launch over the rows has the thin steps' bytes (the wide forms' rule, as a BATCH
 * program's target layers), so the draft's bits stay. */
MUST_CHECK static result_status draft_wide_ffn(struct ds41f_builder *b,
					       uint32_t layer)
{
	const uint32_t slabs = b->draft_slab_count;
	const struct ds41f_tensor *const weights =
		b->plan->layer_tensors[layer];
	const void *const table = expert_table_of(b, layer);
	struct kernel_launch site[DS41F_HYPER_SITE_LAUNCH_COUNT];
	if (table == NULL || b->tensor_layer == NULL)
		return STATUS_INVALID_ARGUMENT;
	ds41f_draft_slab_select(b, slabs);
	result_status status = ds41f_site_begin(b, weights, true, site);
	for (uint32_t slab = 0; slab < slabs && status == DS41F_OK; slab++) {
		ds41f_draft_slab_select(b, slab);
		status = FAULT_POINT(model_v41_draft_slab_dense) ?
				 STATUS_INJECTED_FAULT :
				 ds41f_dense(b, DS41F_WEIGHTS(ROUTER),
					     DS41F_ARRAY(HIDDEN),
					     DS41F_ARRAY(ROUTER_SCORES),
					     DS41F_ELEMENT_F32, 0, 1.0f);
		status = DS41F_THEN(
			FAULT_POINT(model_v41_draft_slab_quantize) ?
				STATUS_INJECTED_FAULT :
				ds41f_quantize(
					b, DS41F_ARRAY(HIDDEN),
					DS41F_HIDDEN)); /* the thin site front's codes */
		struct linear_routed_experts_description experts = { 0 };
		status = DS41F_THEN(routed_of(
			b, layer, b->rows * DS41F_DRAFT_EXPERTS_PER_ROW,
			&experts));
		struct linear_routed_experts_description router_experts =
			experts;
		const struct ds41f_router_description router = {
			&router_experts,
			table,
			DS41F_ARRAY(STEP),
			DS41F_ARRAY(ROUTER_SCORES),
			DS41F_WEIGHTS(ROUTER_BIAS)->stored,
			{ b->plan->cards[b->card].expert_table_offset[layer],
			  DS41F_EXPERT_OFFSETS_BYTES },
			DS41F_ARRAY(AFFINITIES),
			LINEAR_ORDER_FIXED,
			DS41F_DRAFT_EXPERTS,
			DS41F_DRAFT_EXPERTS_PER_ROW
		};
		status = DS41F_THEN(ds41f_router_select_prepare(
			&router, b->arena, &b->router, ds41f_slots(b, 1)));
		status = DS41F_THEN(routed_launches(b, layer, &experts, false));
	}
	ds41f_draft_slab_select(b, slabs);
	status = DS41F_THEN(shared_launches(b, layer, true));
	status = DS41F_THEN(ds41f_tensor_complete_slabs(
		b, 1, DS41F_ARRAY(SHARED_OUTPUT), DS41F_ARRAY(ROW_SUMS)));
	status = DS41F_THEN(ds41f_site_prepare(b, weights, true, site));
	return DS41F_THEN(ds41f_put(b, &site[DS41F_HYPER_SITE_POST]));
}

/* batch_draft_wide 2: a draft layer over a batched draft's rows as a BATCH program runs a target layer over its slabs
 * (chunk_walk's order): over all rows the attention site's mix and finish, wq_a with wkv, qr, wq_b and the queries;
 * a slab at a time its state writes and core (through its slab block: a verifier's block as the slab walk sees it);
 * over all rows wo_a, wo_b, the TP sum and the post; then the FFN site (draft_wide_ffn). The builder takes the
 * draft's
 * rows as its chunk meanwhile (its slabs, first row 0, view 0), then the program's again. */
MUST_CHECK result_status ds41f_draft_wide_layer(struct ds41f_builder *b,
						uint32_t layer)
{
	const uint32_t slabs = b->slab_count, first = b->chunk_first,
		       rows = b->chunk_rows, view = b->chunk_view;
	b->slab_count = b->draft_slab_count;
	b->chunk_first = 0;
	b->chunk_rows = b->draft_rows;
	b->chunk_view = 0;
	ds41f_draft_slab_select(b, b->draft_slab_count);
	result_status status = ds41f_layer_steps(
		b, layer, 0, 0,
		DS41F_SITE_MIX | DS41F_INPUT_PRODUCTS | DS41F_QUERY_NORM |
			DS41F_QUERY_PRODUCTS | DS41F_QUERIES);
	/* level 3: the window writes and the block core once over all rows (the prepare's window part: the same ring
     * entries; each verifier's block whole, as the reference's block: other drafts where a slab cut it) */
	status = b->draft_wide_step ?
			 DS41F_THEN(ds41f_layer_steps(b, layer, 0, 0,
						      DS41F_WINDOW_WRITES |
							      DS41F_CORE)) :
			 status;
	for (uint32_t slab = 0; slab < b->draft_slab_count &&
				status == DS41F_OK && !b->draft_wide_step;
	     slab++) {
		ds41f_draft_slab_select(b, slab);
		status = ds41f_layer_steps(b, layer, 0, 0,
					   DS41F_STATE_WRITES | DS41F_CORE);
	}
	ds41f_draft_slab_select(b, b->draft_slab_count);
	status = DS41F_THEN(ds41f_layer_steps(
		b, layer, 0, 0, DS41F_OUTPUT_PRODUCTS | DS41F_SITE_POST));
	status = DS41F_THEN(draft_wide_ffn(b, layer));
	b->slab_count = slabs;
	b->chunk_first = first;
	b->chunk_rows = rows;
	b->chunk_view = view;
	return status;
}

/* A wide layer's FFN site (agent H's B, wide_prefill.md 3.1): of each piece its mix, finish and router scores; of the
 * chunk the wide select (ds41f_router.h, its jobs into the job table), the quantization and the routed experts; of each
 * piece the shared expert (wide trunk launches) and the post. A piece's launches have the bytes of the thin steps
 * of
 * its slabs (dense_linear.h's wide launches, a hyper site's over its rows) and every expert's pair is computed as in a
 * thin step's job (the composed card test), so the bits are the thin steps'. */
MUST_CHECK static result_status exl3_wide_ffn_launches(struct ds41f_builder *b,
						       uint32_t layer);

MUST_CHECK result_status ds41f_wide_ffn_launches(struct ds41f_builder *b,
						 uint32_t layer)
{
	if (b->tensor_layer != NULL && b->tensor_layer->exl3_wide_entry &&
	    b->plan->exl3_bits[layer] != 0 &&
	    (b->tensor_layer->exact_output_shards ||
	     b->tensor_layer->bf16_ring))
		return exl3_wide_ffn_launches(b, layer);
	const bool original_wide = b->tensor_layer != NULL &&
				   b->tensor_layer->original_wide_experts &&
				   b->tensor_layer->exact_output_shards &&
				   b->plan->exl3_bits[layer] == 0;
	if (b->tensor_layer != NULL && !original_wide &&
	    (b->tensor_layer->exact_output_shards ||
	     b->plan->exl3_bits[layer] != 0)) {
		result_status exact_status = DS41F_OK;
		for (uint32_t slab = 0;
		     slab < b->slab_count && exact_status == DS41F_OK; slab++) {
			ds41f_slab_select(b, slab);
			uint32_t first = b->made->cards[b->card].launch_count;
			exact_status = ds41f_ffn_launches(b, layer, &first);
		}
		return exact_status;
	}
	const struct ds41f_tensor *const weights =
		b->plan->layer_tensors[layer];
	const void *const table = expert_table_of(b, layer);
	struct kernel_launch site[DS41F_HYPER_SITE_LAUNCH_COUNT];
	struct linear_routed_experts_description experts;
	result_status status = table == NULL ? STATUS_INVALID_ARGUMENT :
					       DS41F_OK;
	const uint32_t pieces = original_wide ? b->slab_count :
						ds41f_piece_count_of(b);
	for (uint32_t piece = 0; piece < pieces && status == DS41F_OK;
	     piece++) {
		if (original_wide) {
			/* Keep the exact recipe's fused eight-row front, router arithmetic and quantizer. The wide
			 * select
             * below replaces only its temporary jobs; neither row grouping nor the shared/post products widen. */
			ds41f_slab_select(b, piece);
			struct linear_routed_experts_description routed;
			status = routed_of(b, layer,
					   b->rows * DS41F_EXPERTS_PER_ROW,
					   &routed);
			if (status != DS41F_OK)
				break;
			routed.gate_up_shape.output_size /=
				b->tensor_layer->rank_count;
			routed.down_shape.output_size /=
				b->tensor_layer->rank_count;
			const struct ds41f_router_description router = {
				&routed,
				table,
				DS41F_ARRAY(STEP),
				DS41F_ARRAY(ROUTER_SCORES),
				DS41F_WEIGHTS(ROUTER_BIAS)->stored,
				{ b->plan->cards[b->card]
					  .expert_table_offset[layer],
				  DS41F_EXPERT_OFFSETS_BYTES },
				DS41F_ARRAY(AFFINITIES),
				b->plan->limits.linear_order,
				DS41F_EXPERTS,
				DS41F_EXPERTS_PER_ROW
			};
			status = DS41F_THEN(
				ds41f_site_front(b, weights, true, &router));
		} else {
			ds41f_piece_select(b, piece);
			status = ds41f_site_begin(b, weights, true, site);
			status = DS41F_THEN(ds41f_dense(
				b, DS41F_WEIGHTS(ROUTER), DS41F_ARRAY(HIDDEN),
				DS41F_ARRAY(ROUTER_SCORES), DS41F_ELEMENT_F32,
				0, 1.0f));
		}
	}
	ds41f_chunk_select(b);
	status = DS41F_THEN(
		routed_of(b, layer, ROUTER_WIDE_JOBS(b->rows), &experts));
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
		.step = DS41F_ARRAY(STEP),
		.scores = DS41F_ARRAY(ROUTER_SCORES),
		.bias = DS41F_WEIGHTS(ROUTER_BIAS)->stored,
		.offsets = { b->plan->cards[b->card].expert_table_offset[layer],
			     DS41F_EXPERT_OFFSETS_BYTES },
		.expert_weights = b->plan->experts[layer],
		.affinities = DS41F_ARRAY(AFFINITIES),
		.picks = DS41F_ARRAY(ROUTER_PICKS),
		.counts = { DS41F_ARRAY(ROUTER_COUNTS).byte_offset,
			    (uint64_t)(b->rows + ROUTER_SORT_ROWS - 1u) /
				    ROUTER_SORT_ROWS * DS41F_EXPERTS *
				    sizeof(uint32_t) },
		.pairs = DS41F_ARRAY(ROUTER_PAIRS),
		.runs = DS41F_ARRAY(ROUTER_RUNS),
		.tiles = { DS41F_ARRAY(ROUTER_TILES).byte_offset,
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
			 DS41F_THEN(ds41f_image_router_wide_prepare(
				 &select, b->plan->vision.router_bias[layer],
				 b->arena, &b->router, &b->image_routing,
				 ds41f_slots(b, ROUTER_WIDE_LAUNCH_COUNT))) :
			 DS41F_THEN(router_wide_prepare(
				 &select, b->arena, &b->router,
				 ds41f_slots(b, ROUTER_WIDE_LAUNCH_COUNT)));
	status = DS41F_THEN(
		ds41f_quantize(b, DS41F_ARRAY(HIDDEN), DS41F_HIDDEN));
	status = original_wide ?
			 DS41F_THEN(exact_routed_launches(b, &experts)) :
			 DS41F_THEN(routed_launches(b, layer, &experts, true));
	for (uint32_t piece = 0; piece < pieces && status == DS41F_OK;
	     piece++) {
		if (original_wide) {
			ds41f_slab_select(b, piece);
		} else {
			ds41f_piece_select(b, piece);
		}
		status = shared_launches(b, layer, true);
		if (b->tensor_layer != NULL && !original_wide) {
			status = DS41F_THEN(ds41f_tensor_complete(
				b, 1, DS41F_ARRAY(SHARED_OUTPUT),
				DS41F_ARRAY(ROW_SUMS)));
		}
		if (original_wide) {
			const struct ds41f_tp2_layer_plan *const tensor =
				b->tensor_layer;
			const struct linear_range sums = DS41F_ARRAY(ROW_SUMS);
			const uint64_t row_bytes =
				(uint64_t)DS41F_HIDDEN * LINEAR_SUM_BYTES;
			b->arrays[DS41F_ARRAY_ROW_SUMS] = (struct linear_range){
				tensor->exact_sums.byte_offset +
					(uint64_t)b->first_row * row_bytes,
				(uint64_t)b->rows * row_bytes
			};
			b->tensor_layer = NULL;
			status = DS41F_THEN(
				ds41f_site_prepare(b, weights, true, site));
			b->tensor_layer = tensor;
			b->arrays[DS41F_ARRAY_ROW_SUMS] = sums;
		} else {
			status = DS41F_THEN(
				ds41f_site_prepare(b, weights, true, site));
		}
		status = DS41F_THEN(ds41f_put(b, &site[DS41F_HYPER_SITE_POST]));
	}
	return status;
}

/* exl3_wide_entry, an EXL3 tensor layer in a chunked step (ds41f_tp_stage.h): of each piece
 * its mix, finish and router scores (wide_ffn_launches' launches, the thin fronts' bytes); of the chunk the wide
 * select (8-row jobs, an expert's runs adjacent), the quantization and ONE EXL3 expert call over its rows (the local-K
 * passes and row sums, or the exact recipe's output-shard passes with their gathers a slab at a time; every pair's
 * words those of a thin step's job); of each slab of 8 rows the shared expert, the BF16 ring sum (or the
 * exact
 * recipe's gathered sums) and the post exactly as ffn_launches runs them for a thin step. */
/* A wide step's router scores (float32 [rows][outputs]): router_mma's split-K tensor-core product and its reduce for 9
 * to 64 rows where prepared, else the dense product (its order). */
MUST_CHECK result_status ds41f_router_scores(struct ds41f_builder *b,
					     const struct ds41f_tensor *router)
{
	const struct matrix_shape_record *const shape = &router->shape;
	const struct linear_range partials =
		b->tensor_layer != NULL ? b->tensor_layer->router_partials :
					  (struct linear_range){ 0, 0 };
	const uint64_t partial_bytes = (uint64_t)DS41F_ROUTER_MMA_SPLITS *
				       b->rows * shape->output_size *
				       sizeof(float);
	/* router_mma_wide: more than 64 rows by the 64-row form over blocks of 64 rows (grid y) */
	const bool wide = b->rows > HEAD_MMA_MOST_ROWS &&
			  b->router_mma_wide_kernel != BACKEND_NO_KERNEL;
	if (b->router_mma_kernel == BACKEND_NO_KERNEL ||
	    b->rows <= LINEAR_MAXIMUM_ROWS ||
	    (b->rows > HEAD_MMA_MOST_ROWS && !wide) ||
	    shape->output_size % HEAD_MMA_TILE_OUTPUTS != 0 ||
	    shape->input_size %
			    (HEAD_MMA_STEP_INPUTS * DS41F_ROUTER_MMA_SPLITS) !=
		    0 ||
	    router->stored.byte_count != (uint64_t)shape->output_size *
						 shape->input_size *
						 sizeof(uint16_t) ||
	    partials.byte_count < partial_bytes)
		return ds41f_dense_cone(b, router, DS41F_ARRAY(HIDDEN),
					DS41F_ARRAY(ROUTER_SCORES),
					DS41F_ELEMENT_F32, 0, 1.0f,
					b->ffn_cone_layer);
	const struct router_mma_arguments record = {
		{ .weights_offset = router->stored.byte_offset,
		  .rows_offset = DS41F_ARRAY(HIDDEN).byte_offset,
		  .output_offset = DS41F_ARRAY(ROUTER_SCORES).byte_offset,
		  .step_offset = 0,
		  .output_size = shape->output_size,
		  .input_size = shape->input_size,
		  .row_count = b->rows,
		  .output_type = DS41F_ELEMENT_F32,
		  .flags = 0,
		  .output_scale = 1.0f,
		  .epsilon = 0.0f,
		  .cone_layer = 0 },
		partials.byte_offset,
		DS41F_ROUTER_MMA_SPLITS,
		0
	};
	const uint32_t items = shape->output_size / HEAD_MMA_TILE_OUTPUTS *
			       DS41F_ROUTER_MMA_SPLITS;
	const uint32_t results = b->rows * shape->output_size;
	struct kernel_launch launches[ROUTER_MMA_LAUNCHES];
	prepare_launch_fill(&launches[0],
			    wide ? b->router_mma_wide_kernel :
				   b->router_mma_kernel,
			    (items + ROUTER_MMA_WARPS - 1u) / ROUTER_MMA_WARPS,
			    wide ? (b->rows + HEAD_MMA_MOST_ROWS - 1u) /
					    HEAD_MMA_MOST_ROWS :
				   1u,
			    LINEAR_THREADS_PER_WARP * ROUTER_MMA_WARPS, 0,
			    &record, sizeof(record));
	prepare_launch_fill(&launches[1], b->router_mma_reduce_kernel,
			    (results + ROUTER_MMA_REDUCE_THREADS - 1u) /
				    ROUTER_MMA_REDUCE_THREADS,
			    1, ROUTER_MMA_REDUCE_THREADS, 0, &record,
			    sizeof(record));
	result_status status = ds41f_put(b, &launches[0]);
	return DS41F_THEN(ds41f_put(b, &launches[1]));
}

/* Whether shared_launches(b, layer, true) takes the BATCH trunk_mma_64 path: from the front's BF16 rows
 * alone (no
 * quantized codes), so it may run as a branch beside the router and the routed experts. */
static bool shared_wide_mma(struct ds41f_builder *b, uint32_t layer)
{
	const struct ds41f_tensor *const weights =
		b->plan->layer_tensors[layer];
	struct linear_matrix gate, down;
	if (b->tensor_layer == NULL || !ds41f_thin_forms_of(b) ||
	    ds41f_matrix_of(b, DS41F_WEIGHTS(SHARED_GATE), DS41F_TRUNK,
			    &gate) != DS41F_OK ||
	    ds41f_matrix_of(b, DS41F_WEIGHTS(SHARED_DOWN), DS41F_TRUNK,
			    &down) != DS41F_OK)
		return false;
	const struct linear_trunk_bfloat16_description gate_product = {
		.matrix = gate,
		.values = DS41F_ARRAY(HIDDEN),
		.output = DS41F_ARRAY(SHARED_GATE),
		.row_count = b->rows,
		.segment_count = 1
	};
	const struct linear_trunk_bfloat16_description down_product = {
		.matrix = down,
		.values = DS41F_ARRAY(PAIR_HIDDEN),
		.output = DS41F_ARRAY(SHARED_OUTPUT),
		.row_count = b->rows,
		.segment_count = 1
	};
	return ds41f_trunk_mma_wide_applies(b, &gate_product) &&
	       ds41f_trunk_mma_wide_applies(b, &down_product);
}
#define FFN_PHASES (DS41F_FFN_COMPUTE | DS41F_FFN_SUM | DS41F_FFN_POST)
MUST_CHECK result_status ds41f_exl3_wide_ffn_phases(struct ds41f_builder *b,
						    uint32_t layer,
						    uint32_t phases)
{
	const struct ds41f_tensor *const weights =
		b->plan->layer_tensors[layer];
	const void *const table = expert_table_of(b, layer);
	struct kernel_launch site[DS41F_HYPER_SITE_LAUNCH_COUNT];
	struct linear_routed_experts_description experts = { 0 };
	result_status status = table == NULL || b->vision_active ?
				       STATUS_INVALID_ARGUMENT :
				       DS41F_OK;
	/* a full program's decoder layer: the FFN front's mix and router leave the rows the routed experts leave */
	b->ffn_cone_layer = b->made->program.kind == PROGRAM_FULL &&
					    layer >=
						    DS41F_DECODER_FIRST_LAYER &&
					    layer < DS41F_LAYERS ?
				    layer :
				    0u;
	for (uint32_t piece = 0;
	     piece < ds41f_piece_count_of(b) && status == DS41F_OK &&
	     (phases & DS41F_FFN_FRONT) != 0;
	     piece++) {
		ds41f_piece_select(b, piece);
		status = ds41f_site_begin(b, weights, true, site);
		/* prefill_dead_halves: a dead half's router alone (its rows the cone's: no gather) */
		/* router_first_whole: the first half's alone, beside the second half's attention sums */
		const bool first_whole =
			b->tp_description != NULL &&
			b->tp_description->stages[b->card]
				->description.router_first_whole &&
			ds41f_selected_half(b) == 0;
		const bool gathered =
			ds41f_gathers_apply(b) &&
			b->rows >= ROUTER_GATHER_LEAST_ROWS &&
			!ds41f_half_dead(b, layer, ds41f_selected_half(b)) &&
			!first_whole;
		status = DS41F_THEN(
			gathered ?
				ds41f_router_gather(b, DS41F_WEIGHTS(ROUTER)) :
				ds41f_router_scores(b, DS41F_WEIGHTS(ROUTER)));
	}
	b->ffn_cone_layer = 0;
	ds41f_chunk_select(b);
	/* a BATCH step on the TC MoE: with graph_branches the shared expert first as a side branch
	 * (ffn_launches');
     * without, and launches_untrimmed off, first in the chain where its one ring takes the step (the combine then adds
     * the routed sums into its rows, as the branch's does: the same bits, no join launch) */
	const bool shared_ahead =
		phases == FFN_PHASES && !b->tensor_layer->exact_output_shards &&
		b->tensor_layer->exl3_tc_moe && b->rows <= EXL3_TC_MOST_ROWS &&
		ds41f_piece_count_of(b) == 1u && status == DS41F_OK &&
		shared_wide_mma(b, layer);
	const bool side_shared = shared_ahead &&
				 b->tensor_layer->graph_branches >=
					 GRAPH_BRANCHES_BATCH;
	const bool chain_shared = shared_ahead && !side_shared &&
				  !b->tensor_layer->launches_untrimmed &&
				  b->tensor_layer->bf16_ring &&
				  b->tensor_layer->ring_rows >
					  LINEAR_MAXIMUM_ROWS &&
				  b->chunk_rows <= b->tensor_layer->ring_rows;
	const uint32_t first_side = b->made->cards[b->card].launch_count;
	status = side_shared || chain_shared ?
			 DS41F_THEN(shared_launches(b, layer, true)) :
			 status;
	const uint32_t first_routed = b->made->cards[b->card].launch_count;
	status = (phases & (DS41F_FFN_FRONT | DS41F_FFN_EXPERTS |
			    DS41F_FFN_EXPERTS_FRONT | DS41F_FFN_DOWN_FIRST |
			    DS41F_FFN_DOWN_SECOND)) != 0 ?
			 DS41F_THEN(routed_of(b, layer,
					      ROUTER_WIDE_JOBS(b->rows),
					      &experts)) :
			 status;
	/* the exact recipe's table names the rank's output shards (as ffn_launches' router shapes) */
	struct matrix_shape_record gate_up_shape = experts.gate_up_shape,
				   down_shape = experts.down_shape;
	if (b->tensor_layer->exact_output_shards) {
		const struct bb_exl3_output_shard_plan *const shard =
			b->tensor_layer->exl3_shard;
		if (shard == NULL)
			return STATUS_UNSUPPORTED;
		gate_up_shape.output_size = shard->gate_up.output_count;
		down_shape.output_size = shard->down.output_count;
		down_shape.input_size = shard->down.input_count;
	}
	const struct router_wide_description select = {
		.expert_offsets = table,
		.step = DS41F_ARRAY(STEP),
		.scores = DS41F_ARRAY(ROUTER_SCORES),
		.bias = DS41F_WEIGHTS(ROUTER_BIAS)->stored,
		.offsets = { b->plan->cards[b->card].expert_table_offset[layer],
			     DS41F_EXPERT_OFFSETS_BYTES },
		.expert_weights = b->plan->experts[layer],
		.affinities = DS41F_ARRAY(AFFINITIES),
		.picks = DS41F_ARRAY(ROUTER_PICKS),
		.counts = { DS41F_ARRAY(ROUTER_COUNTS).byte_offset,
			    (uint64_t)(b->rows + ROUTER_SORT_ROWS - 1u) /
				    ROUTER_SORT_ROWS * DS41F_EXPERTS *
				    sizeof(uint32_t) },
		.pairs = DS41F_ARRAY(ROUTER_PAIRS),
		.runs = DS41F_ARRAY(ROUTER_RUNS),
		.tiles = { DS41F_ARRAY(ROUTER_TILES).byte_offset,
			   (uint64_t)ROUTER_WIDE_TILES(b->rows) *
				   sizeof(struct wide_expert_tile) },
		.jobs = experts.jobs,
		.gate_up_shape = gate_up_shape,
		.down_shape = down_shape,
		.weight_format = experts.weight_format,
		.order = b->plan->limits.linear_order,
		.row_count = b->rows
	};
	if ((phases & DS41F_FFN_FRONT) != 0) {
		/* the TC prefill MoE (exl3_tcp) reads the router's picks alone: the counts, sorted pairs and job table
		 * only for
         * the other forms (the experts of more than 64 rows: a half's, or with the joint mode the chunk's) */
		const bool picks_only =
			b->tensor_layer->exl3_tc_moe &&
			!b->tensor_layer->exact_output_shards &&
			b->plan->exl3_bits[layer] != 0 &&
			b->rows > ds41f_tc_most_rows(b) &&
			b->rows <= EXL3_TCP_CHUNK_ROWS &&
			b->tensor_layer->exl3_tcp_workspace.byte_count != 0;
		/* at most a sort block of rows: the count, place and jobs in one launch (the same tables;
		 * launches_untrimmed:
         * the three) */
		const bool small = !picks_only && b->rows <= ROUTER_SORT_ROWS &&
				   !b->tensor_layer->launches_untrimmed;
		struct kernel_launch router_launches[ROUTER_WIDE_LAUNCH_COUNT];
		status = DS41F_THEN(
			small ? router_wide_small_prepare(&select, b->arena,
							  &b->router,
							  router_launches) :
				router_wide_prepare(&select, b->arena,
						    &b->router,
						    router_launches));
		const uint32_t router_count =
			picks_only ? 1u :
			small	   ? ROUTER_WIDE_SMALL_LAUNCH_COUNT :
				     (uint32_t)ROUTER_WIDE_LAUNCH_COUNT;
		for (uint32_t launch = 0;
		     launch < router_count && status == DS41F_OK; launch++)
			status = ds41f_put(b, &router_launches[launch]);
		/* a BATCH step whose routed experts take the TC MoE (BF16 rows) and whose shared expert
		 * the _64 form (BF16
         * rows) reads no quantized rows: no quantization (the same bits, a launch a layer fewer) */
		const bool unquantized =
			ds41f_thin_forms_of(b) &&
			b->tensor_layer->exl3_tc_moe &&
			!b->tensor_layer->launches_untrimmed &&
			!b->tensor_layer->exact_output_shards &&
			!b->tensor_layer->experts_partitioned &&
			b->plan->exl3_bits[layer] != 0 &&
			b->rows <= EXL3_TC_MOST_ROWS &&
			shared_wide_mma(b, layer);
		status = unquantized ?
				 status :
				 DS41F_THEN(ds41f_quantize(
					 b, DS41F_ARRAY(HIDDEN), DS41F_HIDDEN));
	}
	/* a BATCH step's one ring (ring_rows above 8) with the shared expert a side branch: the combine may add the
	 * routed
     * sums into the shared rows (shared_sum, as ffn_launches') */
	b->shared_sum_requested =
		(side_shared || chain_shared) &&
		(phases & DS41F_FFN_SUM) != 0 && b->tensor_layer->bf16_ring &&
		!b->tensor_layer->launches_untrimmed &&
		b->tensor_layer->ring_rows > LINEAR_MAXIMUM_ROWS &&
		b->chunk_rows <= b->tensor_layer->ring_rows;
	b->shared_sum_fused = false;
	status = (phases & DS41F_FFN_EXPERTS) != 0 ?
			 DS41F_THEN(exl3_routed_launches(b, layer, &experts)) :
			 status;
	/* mode 7: a part of the joint call with the halves' route (exl3_routed_launches' tcp_emit) */
	b->tcp_emit =
		(phases & DS41F_FFN_EXPERTS_FRONT) != 0 ? TCP_EMIT_FRONT :
		(phases & DS41F_FFN_DOWN_FIRST) != 0	? TCP_EMIT_FIRST_HALF :
		(phases & DS41F_FFN_DOWN_SECOND) != 0	? TCP_EMIT_SECOND_HALF :
							  TCP_EMIT_ALL;
	status = b->tcp_emit != TCP_EMIT_ALL ?
			 DS41F_THEN(exl3_routed_launches(b, layer, &experts)) :
			 status;
	b->tcp_emit = TCP_EMIT_ALL;
	const bool shared_sum = b->shared_sum_fused;
	b->shared_sum_requested = b->shared_sum_fused = false;
	const struct ds41f_tp2_layer_plan *const tensor = b->tensor_layer;
	/* the shared expert and the post a piece (wide launches, the thin steps' bytes); the BF16
	 * recipe's ring sum of the
     * shared output and the routed sums a slab (its bank geometry), between them */
	for (uint32_t piece = 0;
	     piece < ds41f_piece_count_of(b) && status == DS41F_OK &&
	     !side_shared && !chain_shared && (phases & DS41F_FFN_SHARED) != 0;
	     piece++) {
		ds41f_piece_select(b, piece);
		status = shared_launches(b, layer, true);
	}
	const uint32_t first_join = b->made->cards[b->card].launch_count;
	if ((phases & DS41F_FFN_SUM) == 0) {
		/* the sums in another call */
	} else if (!tensor->exact_output_shards &&
		   tensor->ring_rows > LINEAR_MAXIMUM_ROWS) {
		/* BATCH and piece_rings: one BF16 ring a piece of ring_rows rows (the whole of a BATCH
		 * step), not a slab */
		for (uint32_t at = 0; at < b->chunk_rows && status == DS41F_OK;
		     at += tensor->ring_rows) {
			const uint32_t left = b->chunk_rows - at;
			ds41f_rows_select(b, b->chunk_view, b->chunk_first + at,
					  left < tensor->ring_rows ?
						  left :
						  tensor->ring_rows);
			status = ds41f_tensor_complete(
				b, 1, DS41F_ARRAY(SHARED_OUTPUT),
				shared_sum ? (struct linear_range){ 0, 0 } :
					     DS41F_ARRAY(ROW_SUMS));
		}
	} else if (!tensor->exact_output_shards && tensor->bf16_ring &&
		   tensor->shared_channels &&
		   tensor->wide_exchange == DS41F_TP_EXCHANGE_OWNER_RINGS &&
		   b->chunk_rows > LINEAR_MAXIMUM_ROWS &&
		   ds41f_piece_count_of(b) == 1u && !ds41f_thin_forms_of(b)) {
		/* the slabs' rings in one exchange (the same bits) */
		ds41f_chunk_select(b);
		status = ds41f_tensor_complete_owner_rings(
			b, 1, DS41F_ARRAY(SHARED_OUTPUT),
			DS41F_ARRAY(ROW_SUMS));
	} else if (!tensor->exact_output_shards && tensor->bf16_ring &&
		   tensor->shared_channels &&
		   tensor->wide_exchange == DS41F_TP_EXCHANGE_CE_RINGS &&
		   b->chunk_rows > LINEAR_MAXIMUM_ROWS &&
		   ds41f_piece_count_of(b) == 1u && !ds41f_thin_forms_of(b) &&
		   tensor->rank_count == DS41F_TP4_RANKS &&
		   tensor->collectives[1].arguments.incoming_host_mask == 0) {
		/* the slabs' rings by copy engines (the same bits) */
		ds41f_chunk_select(b);
		status = ds41f_tensor_complete_ce_rings(
			b, 1, DS41F_ARRAY(SHARED_OUTPUT),
			DS41F_ARRAY(ROW_SUMS));
	} else {
		for (uint32_t slab = 0;
		     slab < b->slab_count && status == DS41F_OK &&
		     !tensor->exact_output_shards;
		     slab++) {
			ds41f_slab_select(b, slab);
			status = ds41f_tensor_complete(
				b, 1, DS41F_ARRAY(SHARED_OUTPUT),
				DS41F_ARRAY(ROW_SUMS));
		}
	}
	if (side_shared && status == DS41F_OK) {
		const uint32_t end = b->made->cards[b->card].launch_count;
		const uint32_t join =
			shared_sum ?
				first_join - 1u :
				first_join; /* shared_sum: at the combine */
		if (first_side == first_routed || first_routed >= join ||
		    first_join == end)
			return STATUS_INVALID_ARGUMENT;
		for (uint32_t launch = first_side; launch < first_routed;
		     launch++)
			b->launches[b->card][launch].branch = LAUNCH_SIDE;
		b->launches[b->card][first_routed].branch = LAUNCH_RESUME;
		b->launches[b->card][join].branch = LAUNCH_JOIN;
	}
	const uint64_t sums_row_bytes = (uint64_t)DS41F_HIDDEN * sizeof(float);
	for (uint32_t piece = 0;
	     piece < ds41f_piece_count_of(b) && status == DS41F_OK &&
	     (phases & DS41F_FFN_POST) != 0;
	     piece++) {
		ds41f_piece_select(b, piece);
		if (tensor->exact_output_shards) { /* the exact recipe: the piece's gathered sums, as
						    * ffn_launches' post */
			const struct linear_range sums = DS41F_ARRAY(ROW_SUMS);
			b->arrays[DS41F_ARRAY_ROW_SUMS] = (struct linear_range){
				tensor->exact_sums.byte_offset +
					(uint64_t)b->first_row * sums_row_bytes,
				(uint64_t)b->rows * sums_row_bytes
			};
			b->tensor_layer = NULL;
			status = ds41f_site_prepare(b, weights, true, site);
			b->tensor_layer = tensor;
			b->arrays[DS41F_ARRAY_ROW_SUMS] = sums;
		} else {
			status = ds41f_site_prepare(b, weights, true, site);
		}
		status = DS41F_THEN(ds41f_put(b, &site[DS41F_HYPER_SITE_POST]));
	}
	ds41f_chunk_select(b);
	return status;
}

MUST_CHECK static result_status exl3_wide_ffn_launches(struct ds41f_builder *b,
						       uint32_t layer)
{
	return ds41f_exl3_wide_ffn_phases(b, layer, FFN_PHASES);
}

/* The card's step segment, its rotation step where it runs a layer (model_seam.h); or the embedding (REF:1253). */
/* The site fronts' scratch zeroed (their counters start at zero): a thin step's first launch. In a plan of wide rows
 * the scratch lies in the router picks (model_v41.c), which a chunked step's wide FFN writes, so a BATCH program's
 * draft walks, thin walks behind a wide target, zero it again before their fronts. */
MUST_CHECK static result_status front_scratch_fill(struct ds41f_builder *b)
{
	const uint64_t offset = b->plan->cards[b->card].front_scratch_offset;
	if (offset > b->arena || DS41F_FUSED_SCRATCH_BYTES > b->arena - offset)
		return STATUS_INVALID_ARGUMENT;
	const struct fill_record zeros = {
		offset, DS41F_FUSED_SCRATCH_BYTES / FILL_WORD_BYTES, 0, 0
	};
	struct kernel_launch fill;
	const uint32_t blocks = (uint32_t)((zeros.word_count +
					    FILL_THREADS_PER_BLOCK_VALUE - 1u) /
					   FILL_THREADS_PER_BLOCK_VALUE);
	prepare_launch_fill(&fill, b->fill, blocks, 1,
			    FILL_THREADS_PER_BLOCK_VALUE, 0, &zeros,
			    sizeof(zeros));
	return ds41f_put(b, &fill);
}

/* A BATCH program's draft walk: its fronts' scratch zeroed first (front_scratch_fill), where the plan has one. */
MUST_CHECK result_status ds41f_batch_draft_scratch(struct ds41f_builder *b)
{
	return b->made->program.kind == PROGRAM_BATCH &&
			       b->plan->cards[b->card].layers != 0 &&
			       b->plan->limits.row_capacity >=
				       STEP_WIDE_LEAST_ROWS ?
		       front_scratch_fill(b) :
		       DS41F_OK;
}

MUST_CHECK result_status ds41f_step_launches(struct ds41f_builder *b,
					     bool embeds)
{
	const struct ds41f_attention_rotation_description rotation = {
		DS41F_ARRAY(STEP), DS41F_ARRAY(FREQUENCIES),
		DS41F_ARRAY(ROTATION), b->rows, 0
	};
	/* the draft's rows: its copy of the embedding, or draft_layers' ranks their replicated embedding */
	const bool draft_copy =
		b->reads_draft_step &&
		b->plan->cards[b->card]
				.array_offset[DS41F_ARRAY_DRAFT_EMBEDDING] !=
			PROGRAM_NOWHERE;
	const struct ds41f_hyper_embed_description embed = {
		DS41F_ARRAY(STEP),
		draft_copy ? DS41F_ARRAY(DRAFT_EMBEDDING) :
			     b->plan->globals[DS41F_TENSOR_EMBEDDING].stored,
		DS41F_ARRAY(STREAMS),
		DS41F_ARRAY(CARRIED_PRE),
		b->rows,
		DS41F_VOCABULARY_VALUE
	};
	result_status status = DS41F_OK;
	if (!embeds && !b->reads_draft_step && b->slab_count == 1u &&
	    b->plan->cards[b->card].layers != 0 &&
	    b->plan->limits.row_capacity >= STEP_WIDE_LEAST_ROWS) {
		status = front_scratch_fill(b);
	}
	if (embeds) {
		status = DS41F_THEN(ds41f_hyper_embed_prepare(
			&embed, b->arena, &b->hyper, ds41f_slots(b, 1)));
		if (b->vision_active && !b->reads_draft_step) {
			status = DS41F_THEN(ds41f_vision_step_merge_prepare(
				b->plan->vision.session_embeddings,
				b->plan->vision.session_metadata,
				DS41F_ARRAY(STEP), DS41F_ARRAY(STREAMS),
				b->rows, b->plan->limits.session_capacity,
				b->arena, &b->vision, ds41f_slots(b, 1)));
		}
		return status;
	}
	return status != DS41F_OK || b->plan->cards[b->card].layers == 0 ?
		       status :
		       ds41f_attention_rotation_prepare(&rotation, b->arena,
							&b->state,
							ds41f_slots(b, 1));
}
