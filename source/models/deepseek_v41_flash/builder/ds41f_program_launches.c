/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * ds41f_program_launches.c: segments, hand-offs, copies, slices, dense shards, quantise, trunk, groups
 * product, site
 * front (split from model_v41_program.c by whole items).
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

/* a widened slice count above the most is halved until it fits */
#define SLICE_HALVING 2u
const uint32_t ds41f_copied_bytes[DS41F_COPIED_ARRAYS] = {
	DS41F_SELECTION_ROW_BYTES, DS41F_STATE_ROW_BYTES, DS41F_KEY_BYTES,
	DS41F_CANDIDATES_ROW_BYTES
};
const struct linear_range ds41f_no_range = { 0, 0 };
#define EXL3_ATTENTION_SLICES 4u

/* The card's launches from first_launch on as a segment of layer; none, no segment. */
void ds41f_segment_add(struct ds41f_builder *b, uint32_t layer,
		       uint32_t first_launch)
{
	struct execution_program *const program = &b->made->program;
	const uint32_t count =
		b->made->cards[b->card].launch_count - first_launch;
	ASSERT(program->segment_count < DS41F_MOST_SEGMENTS);
	if (count != 0 && program->segment_count < DS41F_MOST_SEGMENTS) {
		b->made->segments[program->segment_count++] =
			(struct segment_record){ b->card,      b->stage, layer,
						 first_launch, count,	 0 };
	}
}

/* A hand-off from the builder's card to card to, behind the last segment (model_seam.h). */
void ds41f_hand_off(struct ds41f_builder *b, uint32_t to,
		    uint64_t source_offset, uint64_t destination_offset,
		    uint32_t byte_count, uint32_t entry_ratio,
		    uint64_t session_stride)
{
	struct execution_program *const program = &b->made->program;
	ASSERT(program->hand_off_count < DS41F_MOST_HAND_OFFS &&
	       program->segment_count != 0);
	if (program->hand_off_count < DS41F_MOST_HAND_OFFS) {
		b->made->hand_offs[program->hand_off_count++] =
			(struct hand_off_record){ source_offset,
						  destination_offset,
						  session_stride,
						  byte_count,
						  entry_ratio,
						  b->card,
						  to,
						  program->segment_count - 1u,
						  0 };
	}
}

/* Behind the last segment: of the copied arrays (a bit each, COPIED_ARRAYS' order) the selection of an index
 * layer,
 * the new pool entries and keys of a source, the candidates of the candidate source, to every other card that holds
 * a copy: 16 arrays of the model at most, each to at most every other card. */
void ds41f_copies_hand_off(struct ds41f_builder *b, uint32_t layer,
			   uint32_t copied)
{
	const struct ds41f_plan *const plan = b->plan;
	const struct ds41f_card *const from = &plan->cards[b->card];
	const uint32_t ratio = (DS41F_POOLED_LAYERS >> layer & 1u) != 0 ?
				       DS41F_POOLED_RATIO :
				       DS41F_DIRECT_RATIO;
	const bool is_source = (DS41F_SOURCE_LAYERS >> layer & 1u) != 0;
	const bool writes[DS41F_COPIED_ARRAYS] = {
		(DS41F_INDEX_LAYERS >> layer & 1u) != 0, is_source, is_source,
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
	for (uint32_t to = 0; to < plan->card_count; to++) {
		const struct ds41f_card *const copy = &plan->cards[to];
		const uint64_t copies[DS41F_COPIED_ARRAYS] = {
			copy->selection_offset[layer], copy->pool_offset[layer],
			copy->key_tiles_offset[layer],
			copy->array_offset[DS41F_ARRAY_CANDIDATES]
		};
		for (uint32_t array = 0;
		     array < DS41F_COPIED_ARRAYS && to != b->card; array++) {
			if (writes[array] && (copied >> array & 1u) != 0 &&
			    copies[array] != PROGRAM_NOWHERE) {
				ds41f_hand_off(b, to, sources[array],
					       copies[array],
					       ds41f_copied_bytes[array],
					       strides[array] == 0 ? 0 : ratio,
					       strides[array]);
			}
		}
	}
}

/* trunk_shape_slices: a thin step's trunk product on a CMP at the slice count measured fastest for its TP4 shard
 * shape
 * (trunk01, gnmac_port: cold weights, 1-8 rows); the exact profile's ordered chains give every output the reference
 * chain at any count, so only the time changes. Other shapes, wide steps and cards keep `fallback`. */
struct tp_shape_slice {
	uint32_t output_size, input_size, slice_count;
};
static const struct tp_shape_slice tp_shape_slices_measured[] = {
	/* trunk01, us at 6 rows: S=4 -> chosen */
	{ 1280, 5120, 8 }, /* wq_a 26.6 -> 21.5 */
	{ 512, 5120, 16 }, /* wkv 25.6 -> 15.4 */
	{ 8192, 1280, 4 }, /* wq_b shard 19.5 (fastest at 4) */
	{ 2048, 4096, 8 }, /* wo_a shard 16.4 -> 13.3 */
	{ 5120, 2048, 4 }, /* wo_b shard 20.5 (fastest at 4) */
	{ 4096, 1280, 4 }, /* indexer wq_b 14.3 (fastest at 4) */
};
static uint32_t tp_shape_slices(const struct ds41f_builder *b,
				const struct matrix_shape_record *shape,
				uint32_t fallback)
{
	if (b->tensor_layer == NULL || !b->tensor_layer->trunk_shape_slices ||
	    b->rows > LINEAR_MAXIMUM_ROWS ||
	    b->arithmetic_class != ARITHMETIC_CLASS_SM_80)
		return fallback;
	for (uint32_t row = 0;
	     row < sizeof(tp_shape_slices_measured) /
			   sizeof(tp_shape_slices_measured[0]);
	     row++)
		if (tp_shape_slices_measured[row].output_size ==
			    shape->output_size &&
		    tp_shape_slices_measured[row].input_size ==
			    shape->input_size)
			return tp_shape_slices_measured[row].slice_count;
	return fallback;
}

MUST_CHECK result_status ds41f_slices_of(
	const struct ds41f_builder *b, enum linear_family_code family,
	const struct matrix_shape_record *shape, uint32_t *slice_count_out)
{
	if (b->tensor_layer != NULL && !b->tensor_layer->exact_output_shards &&
	    ((shape->output_size == DS41F_HEADS_VALUE * DS41F_HEAD_DIMS /
					    b->tensor_layer->rank_count &&
	      shape->input_size == DS41F_QUERY_RANK) ||
	     (shape->output_size ==
		      DS41F_WIDEST_ROW / b->tensor_layer->rank_count &&
	      shape->input_size == DS41F_HEADS_VALUE * DS41F_HEAD_DIMS /
					   DS41F_OUTPUT_GROUPS) ||
	     (shape->output_size == DS41F_HIDDEN &&
	      shape->input_size ==
		      DS41F_WIDEST_ROW / b->tensor_layer->rank_count) ||
	     family == LINEAR_FAMILY_EXPERT_GATE_UP ||
	     family == LINEAR_FAMILY_EXPERT_DOWN)) {
		*slice_count_out =
			family == LINEAR_FAMILY_EXPERT_GATE_UP ||
					family == LINEAR_FAMILY_EXPERT_DOWN ?
				b->tensor_slices :
				tp_shape_slices(b, shape, b->tensor_slices);
		return DS41F_OK;
	}
	return linear_default_slice_count(b->plan->limits.linear_profile,
					  b->plan->limits.linear_order, family,
					  shape, b->arithmetic_class,
					  slice_count_out);
}

/* exact_dense (ds41f_tp_stage.h): wq_b, wo_a and wo_b are output-axis shards of a full matrix. */
static bool is_dense_shard(const struct ds41f_builder *b,
			   const struct ds41f_tensor *tensor)
{
	if (b->tensor_layer == NULL || !b->tensor_layer->exact_dense)
		return false;
	const struct ds41f_tensor *const local =
		b->plan->layer_tensors[b->tensor_layer->layer_index];
	return tensor == &local[DS41F_TENSOR_QUERY_UP] ||
	       tensor == &local[DS41F_TENSOR_OUTPUT_GROUPS] ||
	       tensor == &local[DS41F_TENSOR_OUTPUT];
}

/* A dense shard's slice count: the full shape's measured count times the ranks, at most the stamped maximum and
 * the
 * loads of a row (the class-80 exact profile's ordered slices give every output the reference chain at any count,
 * linear.c); the wide form is the full shape's. */
MUST_CHECK static result_status
dense_shard_matrix(const struct ds41f_builder *b,
		   enum linear_family_code family,
		   struct linear_matrix *matrix_out)
{
	const struct program_limits *const limits = &b->plan->limits;
	struct matrix_shape_record full = matrix_out->shape;
	full.output_size *= b->tensor_layer->rank_count;
	if (b->rows > LINEAR_MAXIMUM_ROWS)
		return linear_default_wide_form(limits->linear_profile,
						limits->linear_order, family,
						&full, b->arithmetic_class,
						&matrix_out->slice_count,
						&matrix_out->wide_form);
	uint32_t slices = 0;
	const result_status status = linear_default_slice_count(
		limits->linear_profile, limits->linear_order, family, &full,
		b->arithmetic_class, &slices);
	if (status != DS41F_OK)
		return status;
	uint32_t widened = slices * b->tensor_layer->rank_count;
	const uint32_t loads =
		matrix_out->shape.input_size / LINEAR_VALUES_PER_LOAD;
	while (widened > LINEAR_MAXIMUM_SLICES || widened > loads)
		widened /= SLICE_HALVING;
	matrix_out->slice_count = widened > slices ? widened : slices;
	matrix_out->wide_form = LINEAR_WIDE_GROUP_BLOCKS;
	return DS41F_OK;
}

/* A matrix of the layer for the rows the next launches take: its default count, or above a slab's rows the count and
 * form of its wide launches (linear.h: the fastest measured, the same bits). */
MUST_CHECK result_status ds41f_matrix_of(const struct ds41f_builder *b,
					 const struct ds41f_tensor *tensor,
					 enum linear_family_code family,
					 struct linear_matrix *matrix_out)
{
	const struct program_limits *const limits = &b->plan->limits;
	*matrix_out = (struct linear_matrix){
		tensor->stored, tensor->shape, tensor->weight_format, 0, 0, 0
	};
	if (is_dense_shard(b, tensor))
		return dense_shard_matrix(b, family, matrix_out);
	if (b->tensor_layer != NULL && !b->tensor_layer->exact_output_shards) {
		const struct ds41f_tensor *const local =
			b->plan->layer_tensors[b->tensor_layer->layer_index];
		if (tensor == &local[DS41F_TENSOR_SHARED_GATE] ||
		    tensor == &local[DS41F_TENSOR_SHARED_UP] ||
		    tensor == &local[DS41F_TENSOR_SHARED_DOWN]) {
			matrix_out->slice_count = b->tensor_slices;
			matrix_out->wide_form = LINEAR_WIDE_GROUP_BLOCKS;
			return DS41F_OK;
		}
	}
	if (b->rows > LINEAR_MAXIMUM_ROWS && b->tensor_layer != NULL) {
		matrix_out->wide_form = LINEAR_WIDE_GROUP_BLOCKS;
		return ds41f_slices_of(b, family, &tensor->shape,
				       &matrix_out->slice_count);
	}
	return b->rows <= LINEAR_MAXIMUM_ROWS ?
		       ds41f_slices_of(b, family, &tensor->shape,
				       &matrix_out->slice_count) :
		       linear_default_wide_form(limits->linear_profile,
						limits->linear_order, family,
						&tensor->shape,
						b->arithmetic_class,
						&matrix_out->slice_count,
						&matrix_out->wide_form);
}

/* A product of dense_linear.h; its weights are float32 where the tensor holds 4 bytes a weight, else bfloat16. */
/* The step rows of the selected rows for a cone (relaxed_linear.h, dense_linear.h), from its first row: the step
 * area's lie first_row rows in already (a chunk array), a half's view holds the half's rows from its first; false for
 * a slab's view or the draft's step. */
static bool cone_step_of(const struct ds41f_builder *b,
			 struct linear_range *step_out)
{
	if (b->reads_draft_step ||
	    (b->view != 0 &&
	     (b->view != b->chunk_view || b->first_row < b->chunk_first)))
		return false;
	const uint64_t into =
		b->view != 0 ? (uint64_t)(b->first_row - b->chunk_first) *
				       STEP_ROW_BYTES_VALUE :
			       0u;
	*step_out =
		(struct linear_range){ DS41F_ARRAY(STEP).byte_offset + into,
				       STEP_BLOCK_BYTES_VALUE +
					       (uint64_t)b->rows *
						       STEP_ROW_BYTES_VALUE };
	return true;
}

/* A dense product; with cone_layer (a full program's decoder layer) and a wide launch, DENSE_CONE_ROWS. */
MUST_CHECK result_status ds41f_dense_cone(struct ds41f_builder *b,
					  const struct ds41f_tensor *weights,
					  struct linear_range rows,
					  struct linear_range output,
					  uint32_t output_type, uint32_t flags,
					  float output_scale,
					  uint32_t cone_layer)
{
	const struct matrix_shape_record *const shape = &weights->shape;
	const bool is_float32 =
		weights->stored.byte_count ==
		linear_sums_byte_count(shape->output_size, shape->input_size);
	struct linear_range step = (flags & DENSE_STEP_ROWS) != 0 ?
					   DS41F_ARRAY(STEP) :
					   ds41f_no_range;
	const bool cone = cone_layer != 0 && b->rows > LINEAR_MAXIMUM_ROWS &&
			  (flags & DENSE_STEP_ROWS) == 0 &&
			  cone_step_of(b, &step);
	const struct dense_linear_description description = {
		weights->stored,
		rows,
		output,
		step,
		shape->output_size,
		shape->input_size,
		b->rows,
		is_float32 ? DS41F_ELEMENT_F32 : DS41F_ELEMENT_BF16,
		output_type,
		flags | (cone ? DENSE_CONE_ROWS : 0u),
		output_scale,
		(flags & DENSE_ROW_SCALE) != 0 ? DS41F_NORM_EPSILON : 0.0f,
		cone ? cone_layer : 0u
	};
	struct kernel_launch *const launch = ds41f_slots(b, 1);
	const result_status status =
		dense_linear_prepare(&description, b->arena, &b->dense, launch);
	if (status == DS41F_OK && b->plan->limits.official_reference != 0 &&
	    shape->input_size == DS41F_HIDDEN &&
	    shape->output_size == DS41F_HEAD_DIMS && flags == 0) {
		launch->kernel = is_float32 ? b->official.projection_float32 :
					      b->official.projection_bfloat16;
		launch->grid[0] = shape->output_size /
				  DS41F_OFFICIAL_PROJECTION_OUTPUTS_VALUE;
		launch->grid[1] = launch->grid[LAUNCH_DIMENSIONS - 1u] = 1;
		launch->block[0] = DS41F_OFFICIAL_PROJECTION_OWNERS_VALUE;
		launch->block[1] = DS41F_OFFICIAL_PROJECTION_OUTPUTS_VALUE;
		launch->block[LAUNCH_DIMENSIONS - 1u] = 1;
		launch->dynamic_shared_byte_count = 0;
	}
	return status;
}

MUST_CHECK result_status ds41f_dense(struct ds41f_builder *b,
				     const struct ds41f_tensor *weights,
				     struct linear_range rows,
				     struct linear_range output,
				     uint32_t output_type, uint32_t flags,
				     float output_scale)
{
	return ds41f_dense_cone(b, weights, rows, output, output_type, flags,
				output_scale, 0);
}

MUST_CHECK result_status ds41f_quantize_into(struct ds41f_builder *b,
					     struct linear_range values,
					     uint32_t width,
					     struct linear_range codes,
					     struct linear_range exponents,
					     struct linear_range operands)
{
	const struct linear_quantize_description description = {
		values, codes, exponents, operands, width, b->rows
	};
	return linear_quantize_prepare(&description, b->arena, &b->linear,
				       ds41f_slots(b, 1));
}

MUST_CHECK result_status ds41f_quantize(struct ds41f_builder *b,
					struct linear_range values,
					uint32_t width)
{
	return ds41f_quantize_into(b, values, width, DS41F_ARRAY(ROW_CODES),
				   DS41F_ARRAY(ROW_EXPONENTS),
				   DS41F_ARRAY(ROW_OPERANDS));
}

/* trunk_cone_layer: a launch of the wide trunk form leaves the rows the engine's prefill cone marks past that
 * layer
 * (relaxed_linear.h; any other form computes every row), with the selection's step rows (cone_step_of); only on a
 * card of the relaxed module's class (card_select prepares b->relaxed there alone). */
MUST_CHECK result_status ds41f_trunk_cone_at(struct ds41f_builder *b,
					     struct kernel_launch *launch,
					     uint32_t cone_layer)
{
	struct linear_range step = ds41f_no_range;
	return cone_layer == 0 ||
			       b->arithmetic_class != ARITHMETIC_CLASS_SM_80 ||
			       !cone_step_of(b, &step) ?
		       DS41F_OK :
		       relaxed_trunk_cone_set(&b->relaxed, step, cone_layer,
					      b->arena, launch);
}

MUST_CHECK static result_status trunk_cone(struct ds41f_builder *b,
					   struct kernel_launch *launch)
{
	return ds41f_trunk_cone_at(b, launch, b->trunk_cone_layer);
}

/* y = W x of the quantized rows for one matrix, or for two in one launch (their slice counts must agree). */
MUST_CHECK result_status ds41f_trunk_of(
	struct ds41f_builder *b, struct linear_range codes,
	struct linear_range exponents, struct linear_range operands,
	const struct ds41f_tensor *first, struct linear_range first_output,
	const struct ds41f_tensor *second, struct linear_range second_output)
{
	struct linear_trunk_quantized_description description = {
		.codes = codes,
		.exponents = exponents,
		.operands = operands,
		.outputs = { first_output, second_output },
		.matrix_count = second == NULL ? 1u : LINEAR_MERGED_MATRICES,
		.row_count = b->rows
	};
	result_status status = ds41f_matrix_of(b, first, DS41F_TRUNK,
					       &description.matrices[0]);
	status = second == NULL ?
			 status :
			 DS41F_THEN(ds41f_matrix_of(b, second, DS41F_TRUNK,
						    &description.matrices[1]));
	const int32_t folds[LINEAR_MERGED_MATRICES] = {
		first->fold_reference,
		second == NULL ? 0 : second->fold_reference
	};
	struct kernel_launch *const launch = ds41f_slots(b, 1);
	if (b->plan->limits.official_reference != 0) {
		const bool native =
			b->arithmetic_class == ARITHMETIC_CLASS_SM_120_NATIVE &&
			second == NULL &&
			(first->shape.output_size ==
				 DS41F_HEADS_VALUE * DS41F_HEAD_DIMS ||
			 first->shape.input_size == DS41F_WIDEST_ROW);
		status = DS41F_THEN(ds41f_official_linear_quantized_prepare(
			&description, b->arena, &b->official.linear,
			native ? DS41F_OFFICIAL_LINEAR_NATIVE :
				 DS41F_OFFICIAL_LINEAR_PORTABLE,
			launch));
	} else {
		status = DS41F_THEN(linear_trunk_quantized_select_prepare(
			&description, b->arena, &b->linear, &b->relaxed,
			b->arithmetic_class, b->plan->limits.cmp_order, folds,
			launch));
	}
	return DS41F_THEN(trunk_cone(b, launch));
}

MUST_CHECK result_status ds41f_trunk(struct ds41f_builder *b,
				     const struct ds41f_tensor *first,
				     struct linear_range first_output,
				     const struct ds41f_tensor *second,
				     struct linear_range second_output)
{
	return ds41f_trunk_of(b, DS41F_ARRAY(ROW_CODES),
			      DS41F_ARRAY(ROW_EXPONENTS),
			      DS41F_ARRAY(ROW_OPERANDS), first, first_output,
			      second, second_output);
}

/* Preserve the BF16 inputs of EXL3 attention, with the sibling's four ordered product slices. */
MUST_CHECK result_status ds41f_exl3_trunk_bfloat16(
	struct ds41f_builder *b, const struct ds41f_tensor *matrix,
	struct linear_range values, struct linear_range output)
{
	const struct linear_trunk_bfloat16_description description = {
		.matrix = { matrix->stored, matrix->shape,
			    matrix->weight_format,
			    tp_shape_slices(b, &matrix->shape,
					    EXL3_ATTENTION_SLICES),
			    0, LINEAR_WIDE_GROUP_BLOCKS },
		.values = values,
		.output = output,
		.row_count = b->rows,
		.segment_count = 1
	};
	struct kernel_launch *const launch = ds41f_slots(b, 1);
	/* ordered_wide: more than 8 rows by the ordered wide form (the reference's bits at any row count, the tensor
     * cores) in place of the wide-four chains */
	const bool ordered =
		b->tensor_layer != NULL && b->tensor_layer->ordered_wide &&
		b->rows > LINEAR_MAXIMUM_ROWS && b->slab_count > 1u;
	const result_status status =
		ordered ? linear_trunk_bfloat16_select_prepare(
				  &description, b->arena, &b->linear,
				  &b->relaxed, b->arithmetic_class,
				  CMP_ORDER_REFERENCE, 0, launch) :
			  linear_trunk_bfloat16_prepare(&description, b->arena,
							&b->linear, launch);
	if (status == DS41F_OK)
		ds41f_trunk_mma_swap(b, launch, &matrix->shape);
	const result_status wide =
		status == DS41F_OK ?
			ds41f_trunk_mma_wide_rows(b, launch, &description) :
			status;
	return wide == DS41F_OK ? trunk_cone(b, launch) : wide;
}

/* The attention output's grouped product (wo_a, block-diagonal over the groups), trunk_mma where it applies. */
MUST_CHECK result_status
ds41f_groups_product(struct ds41f_builder *b,
		     const struct linear_trunk_bfloat16_description *groups,
		     int32_t fold_reference)
{
	struct kernel_launch *const launch = ds41f_slots(b, 1);
	if (b->plan->limits.official_reference != 0)
		return ds41f_official_linear_woa_prepare(
			groups, b->arena, &b->official.linear, launch);
	const result_status status = linear_trunk_bfloat16_select_prepare(
		groups, b->arena, &b->linear, &b->relaxed, b->arithmetic_class,
		b->plan->limits.cmp_order, fold_reference, launch);
	if (status == DS41F_OK &&
	    b->plan->limits.cmp_order == CMP_ORDER_REFERENCE)
		ds41f_trunk_mma_swap(b, launch, &groups->matrix.shape);
	const result_status wide =
		status == DS41F_OK && b->plan->limits.cmp_order ==
					      CMP_ORDER_REFERENCE ?
			ds41f_trunk_mma_wide_rows(b, launch, groups) :
			status;
	return wide == DS41F_OK ? trunk_cone(b, launch) : wide;
}

/* A site's launches (ds41f_hyper_connection.h) for the rows the next launches take, into launches. */
MUST_CHECK result_status ds41f_site_prepare(struct ds41f_builder *b,
					    const struct ds41f_tensor *weights,
					    bool is_ffn,
					    struct kernel_launch *launches)
{
	const struct ds41f_hyper_site_description description = {
		DS41F_ARRAY(STREAMS),
		DS41F_ARRAY(CARRIED_PRE),
		DS41F_ARRAY(MIXES),
		is_ffn ? DS41F_WEIGHTS(FFN_MIX_SCALE)->stored :
			 DS41F_WEIGHTS(ATTENTION_MIX_SCALE)->stored,
		is_ffn ? DS41F_WEIGHTS(FFN_MIX_BASE)->stored :
			 DS41F_WEIGHTS(ATTENTION_MIX_BASE)->stored,
		is_ffn ? DS41F_WEIGHTS(FFN_NORM)->stored :
			 DS41F_WEIGHTS(ATTENTION_NORM)->stored,
		DS41F_ARRAY(POST),
		DS41F_ARRAY(COMB),
		DS41F_ARRAY(HIDDEN),
		is_ffn ? DS41F_ARRAY(SHARED_OUTPUT) :
			 DS41F_ARRAY(SUBLAYER_OUTPUT),
		is_ffn && b->tensor_layer == NULL ? DS41F_ARRAY(ROW_SUMS) :
						    ds41f_no_range,
		b->rows,
		0
	};
	const result_status status = ds41f_hyper_site_prepare(
		&description, b->arena, &b->hyper, launches);
	if (status == DS41F_OK && b->plan->limits.official_reference != 0) {
		launches[DS41F_HYPER_SITE_MIX_FINISH].kernel = b->official.site;
		launches[DS41F_HYPER_SITE_MIX_FINISH].block[0] =
			DS41F_OFFICIAL_SITE_THREADS_VALUE;
	}
	return status;
}

/* A site's mix project and mix finish (REF:982-984, 989-991); its post, for after the sublayer, in launches. */
MUST_CHECK result_status ds41f_site_begin(struct ds41f_builder *b,
					  const struct ds41f_tensor *weights,
					  bool is_ffn,
					  struct kernel_launch *launches)
{
	const struct ds41f_tensor *const mix =
		is_ffn ? DS41F_WEIGHTS(FFN_MIX) : DS41F_WEIGHTS(ATTENTION_MIX);
	/* a full program's decoder layer: the attention site's mix feeds the layer's window rows (the trunks' cone),
     * the FFN site's only the layer's output (the layer's own) */
	result_status status;
	if (b->plan->limits.official_hc_rows != 0) {
		const struct dense_linear_description description = {
			mix->stored,
			DS41F_ARRAY(STREAMS),
			DS41F_ARRAY(MIXES),
			ds41f_no_range,
			DS41F_MIXES,
			DS41F_OFFICIAL_HC_INPUTS,
			b->rows,
			DS41F_ELEMENT_F32,
			DS41F_ELEMENT_F32,
			DENSE_ROW_SCALE,
			1.0f,
			DS41F_NORM_EPSILON,
			0
		};
		status = ds41f_official_hc_shape_prepare(
			&description, b->plan->limits.official_hc_rows,
			b->arena, &b->dense, &b->official_hc,
			ds41f_slots(b, 1));
	} else {
		status = ds41f_dense_cone(
			b, mix, DS41F_ARRAY(STREAMS), DS41F_ARRAY(MIXES),
			DS41F_ELEMENT_F32, DENSE_ROW_SCALE, 1.0f,
			is_ffn ? b->ffn_cone_layer : b->trunk_cone_layer);
	}
	status = DS41F_THEN(ds41f_site_prepare(b, weights, is_ffn, launches));
	return DS41F_THEN(ds41f_put(b, &launches[DS41F_HYPER_SITE_MIX_FINISH]));
}

/* The fused front retains the dense mix's order; the draft's smaller router remains a separate launch. kind: the
 * front's (the site front, or its input front or coefficient front: ds41f_site_front_records.h); the FFN front by
 * router. */
MUST_CHECK result_status
ds41f_site_front_of(struct ds41f_builder *b, const struct ds41f_tensor *weights,
		    bool is_ffn, const struct ds41f_router_description *router,
		    enum ds41f_site_front_kind kind)
{
	const struct ds41f_site_front_description front = {
		.streams = DS41F_ARRAY(STREAMS),
		.carried_pre = DS41F_ARRAY(CARRIED_PRE),
		.mix_weights = (is_ffn ? DS41F_WEIGHTS(FFN_MIX) :
					 DS41F_WEIGHTS(ATTENTION_MIX))
				       ->stored,
		.scale = (is_ffn ? DS41F_WEIGHTS(FFN_MIX_SCALE) :
				   DS41F_WEIGHTS(ATTENTION_MIX_SCALE))
				 ->stored,
		.base = (is_ffn ? DS41F_WEIGHTS(FFN_MIX_BASE) :
				  DS41F_WEIGHTS(ATTENTION_MIX_BASE))
				->stored,
		.norm_weight = (is_ffn ? DS41F_WEIGHTS(FFN_NORM) :
					 DS41F_WEIGHTS(ATTENTION_NORM))
				       ->stored,
		.post = DS41F_ARRAY(POST),
		.comb = DS41F_ARRAY(COMB),
		.hidden = DS41F_ARRAY(HIDDEN),
		.codes = DS41F_ARRAY(ROW_CODES),
		.exponents = DS41F_ARRAY(ROW_EXPONENTS),
		.operands = DS41F_ARRAY(ROW_OPERANDS),
		.scratch = { b->plan->cards[b->card].front_scratch_offset,
			     DS41F_SITE_FRONT_SCRATCH_BYTES },
		.router = router,
		.router_weights = router != NULL ?
					  DS41F_WEIGHTS(ROUTER)->stored :
					  ds41f_no_range,
		.kind = router != NULL ? DS41F_SITE_FRONT_KIND_FFN : kind,
		.row_count = b->rows
	};
	return ds41f_site_front_prepare(&front, b->arena,
					router != NULL ? &b->router : NULL,
					&b->fronts, ds41f_slots(b, 1));
}
MUST_CHECK result_status
ds41f_site_front(struct ds41f_builder *b, const struct ds41f_tensor *weights,
		 bool is_ffn, const struct ds41f_router_description *router)
{
	return ds41f_site_front_of(b, weights, is_ffn, router,
				   DS41F_SITE_FRONT_KIND_SITE);
}

/* front_branch: the launch at side (the coefficient front) a side run of one launch, the one behind it the resume,
 * the card's last (the site's post) the join; where any launch between holds a branch of its own, all stay a chain. */
void ds41f_front_branch_mark(struct ds41f_builder *b, uint32_t side)
{
	struct kernel_launch *const launches = b->launches[b->card];
	const uint32_t last = b->made->cards[b->card].launch_count - 1u;
	bool plain = side + 1u < last;
	for (uint32_t launch = side; launch <= last && plain; launch++)
		plain = launches[launch].branch == LAUNCH_CHAIN;
	if (!plain)
		return;
	launches[side].branch = LAUNCH_SIDE;
	launches[side + 1u].branch = LAUNCH_RESUME;
	launches[last].branch = LAUNCH_JOIN;
}

/* The fusion uses the same product order. Preserve an admitted CMP relaxed product instead. */
bool ds41f_trunk_is_relaxed(const struct ds41f_builder *b,
			    const struct ds41f_tensor *tensor)
{
	return b->arithmetic_class == ARITHMETIC_CLASS_SM_80 &&
	       b->plan->limits.cmp_order == CMP_ORDER_RELAXED &&
	       tensor->fold_reference != 0;
}

MUST_CHECK result_status ds41f_tensor_row_copy(struct ds41f_builder *b,
					       struct linear_range source,
					       struct linear_range output,
					       uint32_t width,
					       uint32_t source_stride,
					       uint32_t output_stride)
{
	const struct tp2_join_description description = {
		.source = source,
		.output = output,
		.width = width,
		.row_count = b->rows,
		.source_row_stride = source_stride,
		.output_row_stride = output_stride,
		.operation = TP2_JOIN_BFLOAT16_COPY
	};
	return tp2_join_prepare(&description, b->arena, b->tensor_join_kernel,
				ds41f_slots(b, 1));
}

/* All original reduction chains are prepared from the full shapes. Only complete output tiles are removed.
 * The stage loader has extracted those tiles without decoding or changing their scales. */
MUST_CHECK result_status ds41f_exact_gather(struct ds41f_builder *b,
					    uint32_t site,
					    struct linear_range input,
					    struct linear_range output,
					    uint32_t bytes, uint32_t vectors)
{
	const struct ds41f_tp2_layer_plan *const tensor = b->tensor_layer;
	if (tensor == NULL || site >= DS41F_TP2_SITES)
		return STATUS_INVALID_ARGUMENT;
	return ds41f_exact_gather_bank(b, site, tensor->ring_offsets[site],
				       tensor->ring_bank_bytes[site],
				       tensor->ring_row_bytes[site], input,
				       output, bytes, vectors);
}
