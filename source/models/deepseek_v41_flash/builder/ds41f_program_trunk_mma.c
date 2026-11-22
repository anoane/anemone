/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * ds41f_program_trunk_mma.c: the BF16 tensor-core trunk forms, splits and sets (split from
 * model_v41_program.c by
 * whole items).
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

#include "engine/library/bf16_linear/trunk_mma_records.h"
#include "ds41f_builder.h"
#include "ds41f_program_internal.h"

/* the trunk MMA kernel's counters a card, in bytes (the model's count: ds41f_tp_stage.h) */
#define TRUNK_MMA_COUNTER_BYTES (DS41F_TRUNK_MMA_COUNTERS * sizeof(uint32_t))

/* trunk_mma's 1-8-row forms: a block of `warps` warps, groups of `split` warps on `tiles` tiles each. */
struct trunk_mma_form {
	const char *name;
	uint32_t tiles, split, warps;
};
static const struct trunk_mma_form trunk_mma_forms[TRUNK_MMA_FORMS] = {
	{ "trunk_mma_8_t1_s1", 1, 1, 8 },
	{ "trunk_mma_8_t1_s2", 1, 2, 8 },
	{ "trunk_mma_8_t1_s4", 1, 4, 8 },
	{ "trunk_mma_8_t1_s8", 1, 8, 8 },
	{ "trunk_mma_8_t2_s1", 2, 1, 8 },
	{ "trunk_mma_8_t2_s2", 2, 2, 8 },
	{ "trunk_mma_8_t2_s4", 2, 4, 8 },
	{ "trunk_mma_8_t2_s8", 2, 8, 8 },
	{ "trunk_mma_8_t1_s16", 1, 16, 16 },
	{ "trunk_mma_8_t1_s8_w16", 1, 8, 16 },
	{ "trunk_mma_8_t1_s4_w16", 1, 4, 16 }
};

/* trunk_mma: the form of a shard shape (relbench02), else the split that gives the narrow shapes the most warps. */
static uint32_t trunk_mma_form_of(const struct matrix_shape_record *shape)
{
	static const struct {
		uint32_t output_size, input_size, form;
	} measured[] = { { 0, 0, 0 } };
	for (uint32_t index = 0; index < sizeof(measured) / sizeof(measured[0]);
	     index++)
		if (measured[index].output_size == shape->output_size &&
		    measured[index].input_size == shape->input_size)
			return measured[index].form;
	const uint32_t tiles = shape->output_size / TRUNK_MMA_TILE_OUTPUTS;
	return tiles <= TRUNK_MMA_S16_MOST_TILES ? TRUNK_MMA_FORM_T1_S16 :
	       tiles <= TRUNK_MMA_S4_MOST_TILES	 ? TRUNK_MMA_FORM_T1_S4 :
						   TRUNK_MMA_FORM_T1_S2;
}

/* --tp_trunk_mma: a prepared BF16-row trunk launch of 1 to 8 rows (trunk_linear_bfloat16_<S>, the
 * same record)
 * becomes kernel_trunk_mma's form for its shape. */
static bool trunk_mma_applies(const struct ds41f_builder *b,
			      const struct matrix_shape_record *shape)
{
	return b->trunk_mma && b->rows <= LINEAR_MAXIMUM_ROWS &&
	       shape->input_size % TRUNK_MMA_GROUP_INPUTS == 0 &&
	       shape->output_size % TRUNK_MMA_TILE_OUTPUTS == 0;
}

/* trunk_mma applies to the shape and its form is prepared on the card (a card prepares at most
 * BACKEND_MAXIMUM_KERNELS: only the forms the products take, when first taken). */
bool ds41f_trunk_mma_ready(struct ds41f_builder *b,
			   const struct matrix_shape_record *shape)
{
	if (!trunk_mma_applies(b, shape))
		return false;
	uint32_t *const kernel =
		&b->trunk_mma_kernels[trunk_mma_form_of(shape)];
	if (*kernel == BACKEND_NO_KERNEL &&
	    b->backend->kernel_prepare(
		    b->cards[b->card], TRUNK_MMA_KERNEL_FILE,
		    trunk_mma_forms[trunk_mma_form_of(shape)].name,
		    TRUNK_LINEAR_BFLOAT16_ARGUMENT_BYTES, kernel) != DS41F_OK)
		*kernel = BACKEND_NO_KERNEL;
	return *kernel != BACKEND_NO_KERNEL;
}

void ds41f_trunk_mma_swap(struct ds41f_builder *b, struct kernel_launch *launch,
			  const struct matrix_shape_record *shape)
{
	if (!ds41f_trunk_mma_ready(b, shape))
		return; /* the prepared launch stays */
	const uint32_t index = trunk_mma_form_of(shape);
	const struct trunk_mma_form *const form = &trunk_mma_forms[index];
	const uint32_t block_tiles = form->tiles * form->warps / form->split;
	launch->kernel = b->trunk_mma_kernels[index];
	launch->grid[0] = (shape->output_size / TRUNK_MMA_TILE_OUTPUTS +
			   block_tiles - 1u) /
			  block_tiles;
	launch->grid[1] = launch->grid[LAUNCH_DIMENSIONS - 1u] = 1;
	launch->block[0] = form->warps * LINEAR_THREADS_PER_WARP;
	launch->block[1] = launch->block[LAUNCH_DIMENSIONS - 1u] = 1;
	launch->dynamic_shared_byte_count = 0;
}

static struct linear_range trunk_mma_partials_of(const struct ds41f_builder *b);

/* --prefill_thin_tail (ds41f_tp_stage.h thin_tail): a full program of 9 to 63 rows, each row in a thin step's
 * forms as a BATCH program's (batch_thin_bits_of, thin_forms_of) */
bool ds41f_thin_tail_program(const struct ds41f_builder *b)
{
	const enum program_kind kind = b->made->program.kind;
	return b->tp_description != NULL &&
	       b->tp_description->stages[b->card]->description.thin_tail &&
	       (kind == PROGRAM_FULL || kind == PROGRAM_WIDE) &&
	       b->program_rows > LINEAR_MAXIMUM_ROWS &&
	       b->program_rows < STEP_WIDE_LEAST_ROWS;
}

/* A BATCH program, or a thin-tail program: its rows a thin step's bits where the forms say
 * (batch_thin_bits_of) */
bool ds41f_thin_forms_of(const struct ds41f_builder *b)
{
	return b->made->program.kind == PROGRAM_BATCH ||
	       ds41f_thin_tail_program(b);
}

/* batch_thin_bits: a BATCH (or thin-tail) program's target products take the forms whose rows have the thin steps'
 * bits */
bool ds41f_batch_thin_bits_of(const struct ds41f_builder *b)
{
	return b->batch_thin_bits && ds41f_thin_forms_of(b);
}

/* trunk_mma's _64 form: the splits of the input groups that bring a launch near TRUNK_MMA_WIDE_BLOCKS blocks, within
 * the partials (splits times outputs at most DS41F_TRUNK_MMA_PARTIAL_VALUES); with trunk_mma_thin_splits the
 * 1-8-row form's split of the shape where the step's rows' partials fit (its runs the same input groups, each summed
 * from +0 in ascending groups by the same instructions, the runs added in order: a row's bits the thin form's) */
static uint32_t trunk_mma_splits_of(const struct ds41f_builder *b,
				    const struct matrix_shape_record *shape)
{
	const uint32_t thin = trunk_mma_forms[trunk_mma_form_of(shape)].split;
	if ((b->trunk_mma_thin_splits || ds41f_batch_thin_bits_of(b)) &&
	    thin <= shape->input_size / TRUNK_MMA_GROUP_INPUTS &&
	    trunk_mma_partials_of(b).byte_count >= (uint64_t)thin * b->rows *
							   shape->output_size *
							   sizeof(float))
		return thin;
	const uint32_t tiles = shape->output_size / TRUNK_MMA_TILE_OUTPUTS;
	const uint32_t blocks =
		(tiles + TRUNK_MMA_WIDE_TILES - 1u) / TRUNK_MMA_WIDE_TILES;
	uint32_t splits = 1;
	while (splits < TRUNK_MMA_MOST_SPLITS &&
	       TRUNK_MMA_SPLIT_GROWTH * splits * blocks <=
		       TRUNK_MMA_WIDE_BLOCKS &&
	       TRUNK_MMA_SPLIT_GROWTH * splits * shape->output_size <=
		       DS41F_TRUNK_MMA_PARTIAL_VALUES &&
	       TRUNK_MMA_SPLIT_GROWTH * splits <=
		       shape->input_size / TRUNK_MMA_GROUP_INPUTS)
		splits *= TRUNK_MMA_SPLIT_GROWTH;
	return splits;
}

/* --tp_trunk_mma, a step of 9 to 64 rows: a prepared wide BF16-row trunk launch becomes
 * kernel_trunk_mma's _64 form
 * over the same matrix, values and output, and with more than one split the relaxed reduce behind it (a launch more);
 * where the form does not apply the prepared launch stays. */
static struct linear_range trunk_mma_partials_of(const struct ds41f_builder *b)
{
	/* the fused form's counters end the area */
	const struct linear_range area =
		b->tensor_layer != NULL ? b->tensor_layer->trunk_mma_partials :
					  (struct linear_range){ 0, 0 };
	return area.byte_count < TRUNK_MMA_COUNTER_BYTES ?
		       (struct linear_range){ 0, 0 } :
		       (struct linear_range){ area.byte_offset,
					      area.byte_count -
						      TRUNK_MMA_COUNTER_BYTES };
}

bool ds41f_trunk_mma_wide_applies(
	const struct ds41f_builder *b,
	const struct linear_trunk_bfloat16_description *product)
{
	const struct matrix_shape_record *const shape = &product->matrix.shape;
	const uint32_t tiles = shape->output_size / TRUNK_MMA_TILE_OUTPUTS,
		       splits = trunk_mma_splits_of(b, shape);
	return b->trunk_mma_wide_kernel != BACKEND_NO_KERNEL &&
	       b->rows > LINEAR_MAXIMUM_ROWS &&
	       b->rows <= TRUNK_MMA_WIDE_MOST_ROWS &&
	       shape->input_size % TRUNK_MMA_GROUP_INPUTS == 0 &&
	       shape->output_size % TRUNK_MMA_TILE_OUTPUTS == 0 &&
	       product->segment_count != 0 &&
	       tiles % (product->segment_count * TRUNK_MMA_WIDE_TILES) == 0 &&
	       product->values.byte_offset % TRUNK_MMA_VALUES_ALIGNMENT == 0 &&
	       (splits == 1u ||
		(b->slab_count > 1u && b->relaxed.reduce != BACKEND_NO_KERNEL &&
		 trunk_mma_partials_of(b).byte_count >=
			 (uint64_t)splits * b->rows * shape->output_size *
				 sizeof(float)));
}

/* batch_thin_bits, a BATCH step of 9 to batch_grouped_rows rows (at most 48): the grouped form of the shape's 1-8-row
 * form (its row groups of 8 at that form's splits: each row the thin bits, the weights decoded once), prepared on the
 * card when first taken; none past those rows or where the card refuses it. */
static uint32_t trunk_mma_grouped_ready(struct ds41f_builder *b,
					const struct matrix_shape_record *shape)
{
	if (!ds41f_batch_thin_bits_of(b) || b->rows <= LINEAR_MAXIMUM_ROWS ||
	    b->rows > b->batch_grouped_rows ||
	    b->rows > TRUNK_MMA_GROUPED_MOST_ROWS)
		return BACKEND_NO_KERNEL;
	const uint32_t index = trunk_mma_form_of(shape);
	const uint32_t groups =
		(b->rows + LINEAR_MAXIMUM_ROWS - 1u) / LINEAR_MAXIMUM_ROWS;
	uint32_t *const kernel =
		&b->trunk_mma_grouped_kernels[groups -
					      TRUNK_MMA_GROUPED_LEAST_GROUPS]
					     [index];
	if (*kernel == BACKEND_NO_KERNEL) {
		char name[TRUNK_MMA_NAME_BYTES];
		(void)snprintf(name, sizeof(name), "trunk_mma_%" PRIu32 "_%s",
			       groups * LINEAR_MAXIMUM_ROWS,
			       trunk_mma_forms[index].name +
				       strlen(TRUNK_MMA_THIN_STEM));
		if (b->backend->kernel_prepare(
			    b->cards[b->card], TRUNK_MMA_KERNEL_FILE, name,
			    TRUNK_LINEAR_BFLOAT16_ARGUMENT_BYTES,
			    kernel) != DS41F_OK)
			*kernel = BACKEND_NO_KERNEL;
	}
	return *kernel;
}

/* batch_row_sets (8, 16 or 24), a BATCH step of more than batch_grouped_rows rows (to 64): the shape's 1-8-row form
 * or its grouped form of 2 or 3 row groups over sets of so many of the step's rows (gridDim.y): each row the thin
 * bits, the weights decoded once a set; none where the card refuses it. */
static uint32_t trunk_mma_sets_ready(struct ds41f_builder *b,
				     const struct matrix_shape_record *shape)
{
	const uint32_t groups = b->batch_row_sets / LINEAR_MAXIMUM_ROWS;
	if (!ds41f_batch_thin_bits_of(b) || groups == 0 ||
	    groups > TRUNK_MMA_SET_MOST_GROUPS ||
	    b->rows <= LINEAR_MAXIMUM_ROWS ||
	    b->rows > TRUNK_MMA_WIDE_MOST_ROWS)
		return BACKEND_NO_KERNEL;
	const uint32_t index = trunk_mma_form_of(shape);
	uint32_t *const kernel =
		groups == 1u ?
			&b->trunk_mma_kernels[index] :
			&b->trunk_mma_grouped_kernels
				 [groups - TRUNK_MMA_GROUPED_LEAST_GROUPS]
				 [index];
	if (*kernel == BACKEND_NO_KERNEL) {
		char name[TRUNK_MMA_NAME_BYTES];
		(void)snprintf(name, sizeof(name), "trunk_mma_%" PRIu32 "_%s",
			       groups * LINEAR_MAXIMUM_ROWS,
			       trunk_mma_forms[index].name +
				       strlen(TRUNK_MMA_THIN_STEM));
		if (b->backend->kernel_prepare(
			    b->cards[b->card], TRUNK_MMA_KERNEL_FILE, name,
			    TRUNK_LINEAR_BFLOAT16_ARGUMENT_BYTES,
			    kernel) != DS41F_OK)
			*kernel = BACKEND_NO_KERNEL;
	}
	return *kernel;
}

/* --tp_launch_trims, a tensor layer's two independent products of one input just launched (wq_a and wkv; the shared
 * expert's gate and up): when both took TRUNK_MMA_PAIR_FORM's 1-8-row or grouped kernel over the same rows, one launch
 * of its pair form in their place (each block the bits of its own launch's: kernel_trunk_mma.cu); else as they are. */
void ds41f_trunk_mma_pair_merge(struct ds41f_builder *b)
{
	struct program_card_record *const card = &b->made->cards[b->card];
	if (b->tensor_layer == NULL || b->tensor_layer->launches_untrimmed ||
	    card->launch_count < TRUNK_MMA_PAIR_LAUNCHES)
		return;
	struct kernel_launch *const first =
		&b->launches[b->card]
			    [card->launch_count - TRUNK_MMA_PAIR_LAUNCHES];
	const struct kernel_launch *const second = first + 1;
	uint32_t form = TRUNK_MMA_GROUPED_FORMS + 1u;
	if (first->kernel == b->trunk_mma_kernels[TRUNK_MMA_PAIR_FORM])
		form = 0;
	for (uint32_t groups = 0; groups < TRUNK_MMA_GROUPED_FORMS; groups++)
		if (first->kernel ==
		    b->trunk_mma_grouped_kernels[groups][TRUNK_MMA_PAIR_FORM])
			form = groups + 1u;
	struct trunk_linear_bfloat16_arguments records[TRUNK_MMA_PAIR_LAUNCHES];
	if (form > TRUNK_MMA_GROUPED_FORMS ||
	    first->kernel == BACKEND_NO_KERNEL ||
	    second->kernel != first->kernel ||
	    second->branch != first->branch ||
	    first->argument_byte_count !=
		    TRUNK_LINEAR_BFLOAT16_ARGUMENT_BYTES ||
	    second->argument_byte_count !=
		    TRUNK_LINEAR_BFLOAT16_ARGUMENT_BYTES ||
	    first->block[0] != second->block[0] ||
	    first->grid[1] != second->grid[1])
		return; /* grid[1]: the row sets (batch_row_sets), the pair form's as its products' */
	memcpy(&records[0], first->argument, sizeof(records[0]));
	memcpy(&records[1], second->argument, sizeof(records[1]));
	if (records[0].row_count != records[1].row_count ||
	    records[0].values_offset != records[1].values_offset)
		return;
	uint32_t *const kernel = &b->trunk_mma_pair_kernels[form];
	if (*kernel == BACKEND_NO_KERNEL) {
		char name[TRUNK_MMA_NAME_BYTES];
		(void)snprintf(name, sizeof(name),
			       "trunk_mma_%" PRIu32 "_%s" TRUNK_MMA_PAIR_SUFFIX,
			       (form + 1u) * LINEAR_MAXIMUM_ROWS,
			       trunk_mma_forms[TRUNK_MMA_PAIR_FORM].name +
				       strlen(TRUNK_MMA_THIN_STEM));
		if (b->backend->kernel_prepare(
			    b->cards[b->card], TRUNK_MMA_KERNEL_FILE, name,
			    sizeof(struct trunk_mma_pair_arguments),
			    kernel) != DS41F_OK) {
			*kernel = BACKEND_NO_KERNEL;
			return;
		}
	}
	const struct trunk_mma_pair_arguments pair = { records[0], records[1],
						       first->grid[0], 0 };
	const uint32_t branch = first->branch;
	prepare_launch_fill(first, *kernel, first->grid[0] + second->grid[0],
			    first->grid[1], first->block[0], 0, &pair,
			    sizeof(pair));
	first->branch = branch;
	card->launch_count--;
}

MUST_CHECK result_status
ds41f_trunk_mma_wide(struct ds41f_builder *b, struct kernel_launch *launch,
		     const struct linear_trunk_bfloat16_description *product)
{
	const struct matrix_shape_record *const shape = &product->matrix.shape;
	const uint32_t tiles = shape->output_size / TRUNK_MMA_TILE_OUTPUTS,
		       splits = trunk_mma_splits_of(b, shape);
	const struct linear_range partials = trunk_mma_partials_of(b);
	if (!ds41f_trunk_mma_wide_applies(b, product))
		return DS41F_OK;
	uint32_t grouped = trunk_mma_grouped_ready(b, shape), sets = 1;
	if (grouped == BACKEND_NO_KERNEL) {
		grouped = trunk_mma_sets_ready(b, shape);
		sets = (b->rows + b->batch_row_sets - 1u) /
		       (b->batch_row_sets != 0 ? b->batch_row_sets : 1u);
	}
	if (grouped != BACKEND_NO_KERNEL) {
		/* the 1-8-row form's geometry and record over the step's rows (row sets: gridDim.y): one launch, no
         * partials */
		const struct trunk_mma_form *const form =
			&trunk_mma_forms[trunk_mma_form_of(shape)];
		const uint32_t block_tiles =
			form->tiles * form->warps / form->split;
		const struct trunk_linear_bfloat16_arguments thin = {
			.weights_offset = product->matrix.stored.byte_offset,
			.values_offset = product->values.byte_offset,
			.output_offset = product->output.byte_offset,
			.output_size = shape->output_size,
			.input_size = shape->input_size,
			.row_count = b->rows,
			.segment_count = product->segment_count
		};
		prepare_launch_fill(launch, grouped,
				    (tiles + block_tiles - 1u) / block_tiles,
				    sets, form->warps * LINEAR_THREADS_PER_WARP,
				    0, &thin,
				    TRUNK_LINEAR_BFLOAT16_ARGUMENT_BYTES);
		return DS41F_OK;
	}
	const struct trunk_relaxed_bfloat16_arguments record = {
		.weights_offset = product->matrix.stored.byte_offset,
		.values_offset = product->values.byte_offset,
		.output_offset = product->output.byte_offset,
		.partials_offset = splits > 1u ? partials.byte_offset : 0,
		.output_size = shape->output_size,
		.input_size = shape->input_size,
		.row_count = b->rows,
		.segment_count = product->segment_count,
		.k_split_count = splits,
		.fold_reference = 0
	};
	prepare_launch_fill(launch, b->trunk_mma_wide_kernel,
			    tiles / TRUNK_MMA_WIDE_TILES * splits, 1,
			    TRUNK_MMA_THREADS, 0, &record,
			    TRUNK_RELAXED_BFLOAT16_ARGUMENT_BYTES);
	if (splits == 1u)
		return DS41F_OK;
	if (b->trunk_mma_fused_kernel != BACKEND_NO_KERNEL &&
	    tiles / TRUNK_MMA_WIDE_TILES <= DS41F_TRUNK_MMA_COUNTERS) {
		/* trunk_mma_fused: the last split of a block adds the splits (the reduce's order): no reduce launch */
		const struct trunk_mma_fused_arguments fused = {
			record, partials.byte_offset + partials.byte_count
		};
		prepare_launch_fill(launch, b->trunk_mma_fused_kernel,
				    tiles / TRUNK_MMA_WIDE_TILES * splits, 1,
				    TRUNK_MMA_THREADS, 0, &fused,
				    sizeof(fused));
		return DS41F_OK;
	}
	const struct trunk_relaxed_reduce_arguments reduce = {
		.partials_offset = partials.byte_offset,
		.output_offset = { product->output.byte_offset, 0 },
		.output_size = { shape->output_size, 0 },
		.row_count = b->rows,
		.k_split_count = splits,
		.output_scale = 1.0f,
		.reserved_zero = 0
	};
	struct kernel_launch *const reduce_launch = ds41f_slots(b, 1);
	if (reduce_launch == NULL)
		return STATUS_INVALID_ARGUMENT;
	const uint32_t results = b->rows * shape->output_size;
	prepare_launch_fill(reduce_launch, b->relaxed.reduce,
			    (results + TRUNK_MMA_REDUCE_THREADS - 1u) /
				    TRUNK_MMA_REDUCE_THREADS,
			    1, TRUNK_MMA_REDUCE_THREADS, 0, &reduce,
			    TRUNK_RELAXED_REDUCE_ARGUMENT_BYTES);
	return DS41F_OK;
}

/* --prefill_tail_rows (ds41f_tp_stage.h tail_rows): a tail program, a full (or, --prefill_front_tail, a wide)
 * program of more than 64 rows and at most the stage's tail_rows, no whole multiple of the exact plan's 256 (the
 * ladder's programs above 64 rows all are: commands.c), whose rows take the 64-row chunk's forms a 64-row group, one
 * decode TC MoE call over them and no split halves */
bool ds41f_tail_program(const struct ds41f_builder *b)
{
	const uint32_t tail = b->tp_description != NULL ?
				      b->tp_description->stages[b->card]
					      ->description.tail_rows :
				      0u;
	const enum program_kind kind = b->made->program.kind;
	return (kind == PROGRAM_FULL || kind == PROGRAM_WIDE) && tail != 0 &&
	       b->program_rows > TRUNK_MMA_WIDE_MOST_ROWS &&
	       b->program_rows <= tail &&
	       b->program_rows % STEP_FULL_GRANULE_ROWS != 0;
}

/* The rows a call of the decode TC MoE takes (exl3_tc_records.h): 64, a tail program's all of them. */
uint32_t ds41f_tc_most_rows(const struct ds41f_builder *b)
{
	return ds41f_tail_program(b) ? b->program_rows : EXL3_TC_MOST_ROWS;
}

/* tail_sets (ds41f_tp_stage.h): a tail program's product, applying to its first 64-row group, as one launch of
 * the _64 form over all its groups (row sets, kernel_trunk_mma.cu: each group a one-group launch's splits and bits)
 * and with more than one split one reduce over all its rows (each sum from +0 in ascending splits: the per-group
 * reduces' bits), the partials in trunk_mma_set_partials; false where the form or the partials do not take it. */
static bool
trunk_mma_sets_put(struct ds41f_builder *b, struct kernel_launch *launch,
		   const struct linear_trunk_bfloat16_description *group,
		   uint32_t rows, result_status *status_out)
{
	const struct matrix_shape_record *const shape = &group->matrix.shape;
	const uint32_t tiles = shape->output_size / TRUNK_MMA_TILE_OUTPUTS,
		       splits = trunk_mma_splits_of(b, shape);
	const struct linear_range partials =
		b->tensor_layer != NULL ?
			b->tensor_layer->trunk_mma_set_partials :
			(struct linear_range){ 0, 0 };
	if (b->tensor_layer == NULL || !b->tensor_layer->tail_sets ||
	    !ds41f_trunk_mma_wide_applies(b, group) ||
	    trunk_mma_grouped_ready(b, shape) != BACKEND_NO_KERNEL ||
	    (splits > 1u && partials.byte_count < (uint64_t)splits * rows *
							  shape->output_size *
							  sizeof(float)))
		return false;
	const struct trunk_relaxed_bfloat16_arguments record = {
		.weights_offset = group->matrix.stored.byte_offset,
		.values_offset = group->values.byte_offset,
		.output_offset = group->output.byte_offset,
		.partials_offset = splits > 1u ? partials.byte_offset : 0,
		.output_size = shape->output_size,
		.input_size = shape->input_size,
		.row_count = b->rows,
		.segment_count = group->segment_count,
		.k_split_count = splits,
		.fold_reference = 0
	};
	prepare_launch_fill(launch, b->trunk_mma_wide_kernel,
			    tiles / TRUNK_MMA_WIDE_TILES * splits,
			    rows / b->rows, TRUNK_MMA_THREADS, 0, &record,
			    TRUNK_RELAXED_BFLOAT16_ARGUMENT_BYTES);
	*status_out = DS41F_OK;
	if (splits == 1u)
		return true;
	const struct trunk_relaxed_reduce_arguments reduce = {
		.partials_offset = partials.byte_offset,
		.output_offset = { group->output.byte_offset, 0 },
		.output_size = { shape->output_size, 0 },
		.row_count = rows,
		.k_split_count = splits,
		.output_scale = 1.0f,
		.reserved_zero = 0
	};
	struct kernel_launch *const reduce_launch = ds41f_slots(b, 1);
	const uint32_t results = rows * shape->output_size;
	if (reduce_launch == NULL)
		*status_out = STATUS_INVALID_ARGUMENT;
	else
		prepare_launch_fill(reduce_launch, b->relaxed.reduce,
				    (results + TRUNK_MMA_REDUCE_THREADS - 1u) /
					    TRUNK_MMA_REDUCE_THREADS,
				    1, TRUNK_MMA_REDUCE_THREADS, 0, &reduce,
				    TRUNK_RELAXED_REDUCE_ARGUMENT_BYTES);
	return true;
}

/* trunk_mma_wide for the selected rows (9 to 64); a tail program's in groups of 64 rows, each its own launch
 * of the group's rows (a 64-row chunk's: the same splits, the same bits), launch then the slots behind it, or
 * with
 * tail_sets one launch over the groups (trunk_mma_sets_put); a tail program's product that no group form takes is
 * refused (its rows would take another form's bits). */
MUST_CHECK result_status ds41f_trunk_mma_wide_rows(
	struct ds41f_builder *b, struct kernel_launch *launch,
	const struct linear_trunk_bfloat16_description *product)
{
	const uint32_t rows = b->rows, group_rows = TRUNK_MMA_WIDE_MOST_ROWS;
	if (!ds41f_tail_program(b) || rows <= group_rows)
		return ds41f_trunk_mma_wide(b, launch, product);
	if (rows % group_rows != 0)
		return STATUS_INVALID_ARGUMENT;
	const uint64_t value_row_bytes = (uint64_t)product->segment_count *
					 product->matrix.shape.input_size *
					 LINEAR_BFLOAT16_BYTES;
	const uint64_t output_row_bytes =
		(uint64_t)product->matrix.shape.output_size *
		LINEAR_BFLOAT16_BYTES;
	result_status status = DS41F_OK;
	b->rows = group_rows;
	struct linear_trunk_bfloat16_description leading =
		*product; /* the first group's */
	leading.values.byte_count = group_rows * value_row_bytes;
	leading.output.byte_count = group_rows * output_row_bytes;
	leading.row_count = group_rows;
	if (trunk_mma_sets_put(b, launch, &leading, rows, &status)) {
		b->rows = rows;
		return status;
	}
	for (uint32_t first = 0; first < rows && status == DS41F_OK;
	     first += group_rows) {
		struct linear_trunk_bfloat16_description group = *product;
		group.values.byte_offset += first * value_row_bytes;
		group.values.byte_count = group_rows * value_row_bytes;
		group.output.byte_offset += first * output_row_bytes;
		group.output.byte_count = group_rows * output_row_bytes;
		group.row_count = group_rows;
		struct kernel_launch *const slot =
			first == 0 ? launch : ds41f_slots(b, 1);
		status = slot == NULL || !ds41f_trunk_mma_wide_applies(b,
								       &group) ?
				 STATUS_INVALID_ARGUMENT :
				 ds41f_trunk_mma_wide(b, slot, &group);
	}
	b->rows = rows;
	return status;
}

MUST_CHECK result_status ds41f_put(struct ds41f_builder *b,
				   const struct kernel_launch *launch)
{
	struct kernel_launch *const slot = ds41f_slots(b, 1);
	if (slot == NULL) {
		LOG_ERROR("program_prepare: card %" PRIu32
			  " has more launches than its part of the program",
			  b->card);
		return STATUS_INVALID_ARGUMENT;
	}
	*slot = *launch;
	return DS41F_OK;
}

/* Exact passes share a fixed arena annex; all descriptions are resolved before graph preparation. */
MUST_CHECK result_status ds41f_official_norm(struct ds41f_builder *b,
					     struct linear_range input,
					     struct linear_range weight,
					     struct linear_range output,
					     uint32_t width)
{
	const uint64_t bytes = (uint64_t)width * DS41F_BF16;
	const struct ds41f_official_norm_description description = {
		{ input.byte_offset, bytes },
		weight,
		{ output.byte_offset, bytes },
		{ b->plan->cards[b->card].official_scratch_offset +
			  DS41F_HEADS_ROW_BYTES,
		  DS41F_OFFICIAL_NORM_STATISTICS * sizeof(float) },
		width,
		1
	};
	return ds41f_official_norm_shape_prepare(
		&description, 1, b->arena, b->official.norm, ds41f_slots(b, 1));
}

MUST_CHECK result_status ds41f_official_roundtrip(struct ds41f_builder *b,
						  struct linear_range values,
						  uint32_t width)
{
	const uint64_t bytes = (uint64_t)width * DS41F_BF16;
	const struct ds41f_official_roundtrip_description description = {
		{ values.byte_offset, bytes },
		{ values.byte_offset, bytes },
		width,
		1
	};
	return ds41f_official_roundtrip_prepare(&description, b->arena,
						b->official.roundtrip,
						ds41f_slots(b, 1));
}
