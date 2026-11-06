/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* vision.c: the reference ViT's pass order, arena validation and fixed scratch lifetimes. */
#include "anemone/vision.h"

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>

#include "prepare_common.h"
#include "anemone/step_records.h"
#include "anemone/model_layout.h"
#include "vision_gemm_policy.h"

#define VISION_SCRATCH_ARRAYS 8u
#define VISION_GLOBAL_WEIGHTS 10u
#define VISION_LAYER_WEIGHTS 8u
#define VISION_MAXIMUM_WEIGHTS \
	(VISION_GLOBAL_WEIGHTS + VISION_MAXIMUM_LAYERS * VISION_LAYER_WEIGHTS)

static const char *const kernel_names[VISION_OPERATIONS] = {
	"vision_linear",    "vision_rms",  "vision_rope",
	"vision_attention", "vision_add",  "vision_silu",
	"vision_unfold",    "vision_gelu", "vision_merge",
	"vision_step_merge"
};

struct vision_config vision_official_config(void)
{
	return (struct vision_config){
		.layer_count = VISION_MAXIMUM_LAYERS,
		.width = VISION_MAXIMUM_WIDTH,
		.heads = VISION_HEADS,
		.intermediate_width = VISION_MAXIMUM_INTERMEDIATE,
		.language_width = VISION_MAXIMUM_LANGUAGE_WIDTH,
		.patch_size = VISION_PATCH_SIZE,
		.downsample_ratio = VISION_DOWNSAMPLE_RATIO,
		.rope_theta = VISION_ROPE_THETA,
		.epsilon = VISION_NORM_EPSILON
	};
}

static bool config_valid(const struct vision_config *config, uint32_t height,
			 uint32_t width)
{
	if (config == NULL || config->layer_count == 0 ||
	    config->layer_count > VISION_MAXIMUM_LAYERS || config->width == 0 ||
	    config->width > VISION_MAXIMUM_WIDTH || config->heads == 0 ||
	    config->heads > VISION_HEADS ||
	    config->width % config->heads != 0 ||
	    (config->width / config->heads) % VISION_ROPE_PARTS != 0 ||
	    config->width / config->heads > VISION_THREADS ||
	    config->intermediate_width == 0 ||
	    config->intermediate_width > VISION_MAXIMUM_INTERMEDIATE ||
	    config->language_width == 0 ||
	    config->language_width > VISION_MAXIMUM_LANGUAGE_WIDTH ||
	    config->patch_size == 0 || config->patch_size > VISION_PATCH_SIZE ||
	    config->downsample_ratio == 0 ||
	    config->downsample_ratio > VISION_DOWNSAMPLE_RATIO ||
	    !isfinite(config->rope_theta) || config->rope_theta <= 1.0f ||
	    !isfinite(config->epsilon) || config->epsilon <= 0.0f ||
	    height == 0 || width == 0 || height > VISION_MAXIMUM_PATCHES ||
	    width > VISION_MAXIMUM_PATCHES ||
	    height > VISION_MAXIMUM_PATCHES / width) {
		return false;
	}
	const uint32_t ratio = config->downsample_ratio;
	const uint32_t cells_height = (height + ratio - 1u) / ratio;
	const uint32_t cells_width = (width + ratio - 1u) / ratio;
	return (uint64_t)cells_height * (cells_width + 1u) +
		       VISION_HALF_DIVISOR <=
	       VISION_MAXIMUM_TOKENS;
}

static bool range_valid(struct linear_range range, uint64_t needed,
			uint64_t arena_bytes, bool exact)
{
	return needed != 0 && range.byte_offset % VISION_ALIGNMENT == 0 &&
	       range.byte_offset <= arena_bytes &&
	       range.byte_count <= arena_bytes - range.byte_offset &&
	       range.byte_count >= needed &&
	       (!exact || range.byte_count == needed);
}

static bool ranges_overlap(struct linear_range first,
			   struct linear_range second)
{
	return first.byte_offset <= second.byte_offset ?
		       second.byte_offset - first.byte_offset <
			       first.byte_count :
		       first.byte_offset - second.byte_offset <
			       second.byte_count;
}

static bool arithmetic_class_valid(uint32_t arithmetic_class)
{
	return arithmetic_class == ARITHMETIC_CLASS_SM_80 ||
	       arithmetic_class == ARITHMETIC_CLASS_SM_120 ||
	       arithmetic_class == ARITHMETIC_CLASS_SM_120_NATIVE;
}

result_status vision_kernels_prepare(const struct backend_operations *backend,
				     struct device_card *card,
				     uint32_t arithmetic_class,
				     struct vision_kernels *kernels_out)
{
	if (backend == NULL || backend->kernel_prepare == NULL ||
	    card == NULL || kernels_out == NULL ||
	    !arithmetic_class_valid(arithmetic_class)) {
		return STATUS_INVALID_ARGUMENT;
	}
	struct vision_kernels kernels = { .arithmetic_class =
						  arithmetic_class };
	for (uint32_t operation = 0; operation < VISION_OPERATIONS;
	     operation++) {
		kernels.operations[operation] = BACKEND_NO_KERNEL;
		const result_status status = backend->kernel_prepare(
			card, VISION_KERNEL_FILE, kernel_names[operation],
			VISION_ARGUMENT_BYTES, &kernels.operations[operation]);
		if (status != STATUS_OK) {
			return status;
		}
	}
	*kernels_out = kernels;
	return STATUS_OK;
}

MUST_CHECK static result_status
linear_partition_flags(uint32_t arithmetic_class,
		       enum vision_gemm_product product, uint32_t rows,
		       uint32_t *flags_out)
{
	struct vision_gemm_policy_record policy;
	const result_status status = vision_gemm_policy_prepare(
		arithmetic_class, product, rows, &policy);
	if (status != STATUS_OK) {
		return status;
	}
	if (policy.partition_width > (VISION_LINEAR_K_PARTITION_MASK >>
				      VISION_LINEAR_K_PARTITION_SHIFT)) {
		return STATUS_INVALID_ARGUMENT;
	}
	*flags_out =
		(policy.partition_width << VISION_LINEAR_K_PARTITION_SHIFT) |
		(policy.bias_last ? VISION_LINEAR_BIAS_LAST : 0u) |
		(policy.mma_k8 ? VISION_LINEAR_MMA_K8 : 0u);
	return STATUS_OK;
}

result_status vision_scratch_plan(const struct vision_config *config,
				  uint32_t grid_height, uint32_t grid_width,
				  uint64_t byte_offset,
				  struct vision_scratch *scratch_out)
{
	if (scratch_out == NULL ||
	    !config_valid(config, grid_height, grid_width) ||
	    byte_offset % VISION_ALIGNMENT != 0) {
		return STATUS_INVALID_ARGUMENT;
	}
	const uint64_t patches = (uint64_t)grid_height * grid_width;
	const uint32_t ratio = config->downsample_ratio;
	const uint64_t cells = (uint64_t)((grid_height + ratio - 1u) / ratio) *
			       ((grid_width + ratio - 1u) / ratio);
	const uint64_t sizes[VISION_SCRATCH_ARRAYS] = {
		patches * config->width,
		patches * config->width,
		patches * config->width * VISION_QKV_COMPONENTS,
		patches * config->width,
		patches * config->intermediate_width * VISION_GATE_COMPONENTS,
		patches * config->intermediate_width,
		cells * config->width * ratio * ratio,
		cells * config->language_width
	};
	struct vision_scratch scratch = { 0 };
	struct linear_range *const ranges[VISION_SCRATCH_ARRAYS] = {
		&scratch.hidden,    &scratch.normalized, &scratch.qkv,
		&scratch.attention, &scratch.gates,	 &scratch.activated,
		&scratch.unfolded,  &scratch.projected
	};
	uint64_t next = byte_offset;
	for (uint32_t array = 0; array < VISION_SCRATCH_ARRAYS; array++) {
		const uint64_t bytes = sizes[array] * VISION_BFLOAT16_BYTES;
		const uint64_t aligned = (bytes + VISION_ALIGNMENT - 1u) /
					 VISION_ALIGNMENT * VISION_ALIGNMENT;
		if (aligned > UINT64_MAX - next) {
			return STATUS_INVALID_ARGUMENT;
		}
		*ranges[array] = (struct linear_range){ .byte_offset = next,
							.byte_count = bytes };
		next += aligned;
	}
	scratch.byte_count = next - byte_offset;
	*scratch_out = scratch;
	return STATUS_OK;
}

struct weight_check {
	struct linear_range range;
	uint64_t count;
};

/* The deployed checkpoint stores BF16 norm parameters; the reference widens them into its F32 RMSNorm. */
static uint64_t norm_weight_bytes(struct linear_range range, uint64_t width)
{
	return range.byte_count == width * VISION_BFLOAT16_BYTES ?
		       width * VISION_BFLOAT16_BYTES :
		       width * VISION_FLOAT_BYTES;
}

/* Only read/read overlap is permitted: even an unused part of the supplied output/scratch range is private. */
static bool weights_valid(const struct vision_description *description,
			  uint64_t arena_bytes)
{
	const struct vision_config *const config = &description->config;
	const struct vision_weights *const weights = description->weights;
	const uint64_t hidden = config->width;
	const uint64_t language = config->language_width;
	const uint64_t patch = (uint64_t)VISION_CHANNELS * config->patch_size *
			       config->patch_size;
	const uint64_t unfolded =
		hidden * config->downsample_ratio * config->downsample_ratio;
	struct weight_check checks[VISION_MAXIMUM_WEIGHTS] = {
		{ weights->patch, hidden * patch * VISION_BFLOAT16_BYTES },
		{ weights->patch_bias, hidden * VISION_BFLOAT16_BYTES },
		{ weights->final_norm,
		  norm_weight_bytes(weights->final_norm, hidden) },
		{ weights->projector1,
		  language * unfolded * VISION_BFLOAT16_BYTES },
		{ weights->projector1_bias, language * VISION_BFLOAT16_BYTES },
		{ weights->projector2,
		  language * language * VISION_BFLOAT16_BYTES },
		{ weights->projector2_bias, language * VISION_BFLOAT16_BYTES },
		{ weights->image_start, language * VISION_BFLOAT16_BYTES },
		{ weights->image_newline, language * VISION_BFLOAT16_BYTES },
		{ weights->image_end, language * VISION_BFLOAT16_BYTES }
	};
	uint32_t count = VISION_GLOBAL_WEIGHTS;
	for (uint32_t layer = 0; layer < config->layer_count; layer++) {
		const struct vision_layer_weights *const current =
			&weights->layers[layer];
		const uint64_t intermediate = config->intermediate_width;
		const struct weight_check local[VISION_LAYER_WEIGHTS] = {
			{ current->norm1,
			  norm_weight_bytes(current->norm1, hidden) },
			{ current->qkv, VISION_QKV_COMPONENTS * hidden *
						hidden *
						VISION_BFLOAT16_BYTES },
			{ current->qkv_bias, VISION_QKV_COMPONENTS * hidden *
						     VISION_BFLOAT16_BYTES },
			{ current->attention_output,
			  hidden * hidden * VISION_BFLOAT16_BYTES },
			{ current->attention_output_bias,
			  hidden * VISION_BFLOAT16_BYTES },
			{ current->norm2,
			  norm_weight_bytes(current->norm2, hidden) },
			{ current->gate_up, VISION_GATE_COMPONENTS *
						    intermediate * hidden *
						    VISION_BFLOAT16_BYTES },
			{ current->down,
			  hidden * intermediate * VISION_BFLOAT16_BYTES }
		};
		memcpy(&checks[count], local, sizeof(local));
		count += VISION_LAYER_WEIGHTS;
	}
	for (uint32_t index = 0; index < count; index++) {
		if (!range_valid(checks[index].range, checks[index].count,
				 arena_bytes, true) ||
		    ranges_overlap(checks[index].range, description->scratch) ||
		    ranges_overlap(checks[index].range, description->output)) {
			return false;
		}
	}
	return true;
}

static void pass_add(struct kernel_launch *launches, uint32_t *count,
		     const struct vision_kernels *kernels,
		     enum vision_operation operation,
		     struct vision_arguments arguments)
{
	uint32_t blocks = (arguments.rows * arguments.output_width +
			   VISION_THREADS - 1u) /
			  VISION_THREADS;
	const uint32_t shared_bytes = 0;
	uint32_t block_rows = 1;
	if (operation == VISION_LINEAR) {
		blocks = (arguments.output_width + VISION_LINEAR_TILE - 1u) /
			 VISION_LINEAR_TILE;
		block_rows = (arguments.rows + VISION_LINEAR_TILE - 1u) /
			     VISION_LINEAR_TILE;
	}
	if (operation == VISION_RMS) {
		blocks = arguments.rows;
	} else if (operation == VISION_ATTENTION) {
		blocks = (arguments.rows + VISION_ATTENTION_QUERY_TILE - 1u) /
			 VISION_ATTENTION_QUERY_TILE * arguments.heads;
	}
	prepare_launch_fill(&launches[*count], kernels->operations[operation],
			    blocks, block_rows, VISION_THREADS, shared_bytes,
			    &arguments, VISION_ARGUMENT_BYTES);
	(*count)++;
}

static struct vision_arguments
pass_record(struct linear_range source, struct linear_range weights,
	    struct linear_range bias, struct linear_range destination,
	    uint32_t rows, uint32_t input, uint32_t output)
{
	return (struct vision_arguments){
		.source_offset = source.byte_offset,
		.weights_offset = weights.byte_offset,
		.bias_offset = bias.byte_offset,
		.destination_offset = destination.byte_offset,
		.rows = rows,
		.input_width = input,
		.output_width = output,
		.flags = bias.byte_count == 0 ? 0u : VISION_LINEAR_BIAS
	};
}

result_status vision_prepare(const struct vision_description *description,
			     uint64_t arena_byte_count,
			     const struct vision_kernels *kernels,
			     struct kernel_launch *launches_out,
			     uint32_t launch_capacity,
			     uint32_t *launch_count_out)
{
	if (description == NULL || kernels == NULL || launches_out == NULL ||
	    launch_count_out == NULL || description->weights == NULL ||
	    !arithmetic_class_valid(kernels->arithmetic_class) ||
	    !config_valid(&description->config, description->grid_height,
			  description->grid_width)) {
		return STATUS_INVALID_ARGUMENT;
	}
	const struct vision_config *const config = &description->config;
	const uint32_t rows =
		description->grid_height * description->grid_width;
	const uint32_t width = config->width;
	const uint32_t patch =
		VISION_CHANNELS * config->patch_size * config->patch_size;
	const uint32_t ratio = config->downsample_ratio;
	const uint32_t cell_height =
		(description->grid_height + ratio - 1u) / ratio;
	const uint32_t cell_width =
		(description->grid_width + ratio - 1u) / ratio;
	const uint32_t cells = cell_height * cell_width;
	const uint32_t needed = config->layer_count * VISION_PASSES_PER_LAYER +
				VISION_FIXED_PASSES;
	struct vision_scratch scratch;
	const result_status status = vision_scratch_plan(
		config, description->grid_height, description->grid_width,
		description->scratch.byte_offset, &scratch);
	if (status != STATUS_OK || launch_capacity < needed ||
	    !range_valid(description->patches,
			 (uint64_t)rows * patch * VISION_BFLOAT16_BYTES,
			 arena_byte_count, false) ||
	    !range_valid(description->output,
			 (uint64_t)cells * config->language_width *
				 VISION_BFLOAT16_BYTES,
			 arena_byte_count, false) ||
	    !range_valid(description->scratch, scratch.byte_count,
			 arena_byte_count, false) ||
	    ranges_overlap(description->patches, description->scratch) ||
	    ranges_overlap(description->patches, description->output) ||
	    ranges_overlap(description->scratch, description->output) ||
	    !weights_valid(description, arena_byte_count)) {
		return STATUS_INVALID_ARGUMENT;
	}
	for (uint32_t operation = 0; operation < VISION_OPERATIONS;
	     operation++) {
		if (kernels->operations[operation] == BACKEND_NO_KERNEL) {
			return STATUS_INVALID_ARGUMENT;
		}
	}
	uint32_t qkv_flags = 0, attention_output_flags = 0, up_flags = 0,
		 down_flags = 0, projector_first_flags = 0,
		 projector_second_flags = 0;
	if (width == VISION_MAXIMUM_WIDTH &&
	    rows >= VISION_GEMM_MINIMUM_MLP_ROWS) {
		result_status selected = linear_partition_flags(
			kernels->arithmetic_class, VISION_GEMM_QUERY_KEY_VALUE,
			rows, &qkv_flags);
		if (selected == STATUS_OK) {
			selected = linear_partition_flags(
				kernels->arithmetic_class,
				VISION_GEMM_ATTENTION_OUTPUT, rows,
				&attention_output_flags);
		}
		if (selected == STATUS_OK &&
		    config->intermediate_width == VISION_MAXIMUM_INTERMEDIATE) {
			selected = linear_partition_flags(
				kernels->arithmetic_class, VISION_GEMM_MLP_UP,
				rows, &up_flags);
			if (selected == STATUS_OK) {
				selected = linear_partition_flags(
					kernels->arithmetic_class,
					VISION_GEMM_MLP_DOWN, rows,
					&down_flags);
			}
		}
		if (selected != STATUS_OK) {
			return selected;
		}
	}
	if (config->language_width == VISION_MAXIMUM_LANGUAGE_WIDTH &&
	    cells >= VISION_GEMM_MINIMUM_PROJECTOR_ROWS) {
		result_status selected = STATUS_OK;
		if (width * ratio * ratio == VISION_MAXIMUM_WIDTH *
						     VISION_DOWNSAMPLE_RATIO *
						     VISION_DOWNSAMPLE_RATIO) {
			selected = linear_partition_flags(
				kernels->arithmetic_class,
				VISION_GEMM_PROJECTOR_FIRST, cells,
				&projector_first_flags);
		}
		if (selected == STATUS_OK) {
			selected = linear_partition_flags(
				kernels->arithmetic_class,
				VISION_GEMM_PROJECTOR_SECOND, cells,
				&projector_second_flags);
		}
		if (selected != STATUS_OK) {
			return selected;
		}
	}
	const struct linear_range empty = { 0 };
	const struct vision_weights *const weights = description->weights;
	uint32_t count = 0;
	struct vision_arguments record = pass_record(
		description->patches, weights->patch, weights->patch_bias,
		scratch.hidden, rows, patch, width);
	pass_add(launches_out, &count, kernels, VISION_LINEAR, record);
	for (uint32_t layer = 0; layer < config->layer_count; layer++) {
		const struct vision_layer_weights *const current =
			&weights->layers[layer];
		record = pass_record(scratch.hidden, current->norm1, empty,
				     scratch.normalized, rows, width, width);
		record.epsilon = config->epsilon;
		record.flags = current->norm1.byte_count ==
					       (uint64_t)width *
						       VISION_BFLOAT16_BYTES ?
				       VISION_NORM_BFLOAT16_WEIGHT :
				       0u;
		pass_add(launches_out, &count, kernels, VISION_RMS, record);
		record = pass_record(scratch.normalized, current->qkv,
				     current->qkv_bias, scratch.qkv, rows,
				     width, VISION_QKV_COMPONENTS * width);
		record.flags |= qkv_flags;
		pass_add(launches_out, &count, kernels, VISION_LINEAR, record);
		record = pass_record(scratch.qkv, empty, empty, scratch.qkv,
				     rows, width, width);
		record.grid_height = description->grid_height;
		record.grid_width = description->grid_width;
		record.heads = config->heads;
		record.theta = config->rope_theta;
		pass_add(launches_out, &count, kernels, VISION_ROPE, record);
		record.destination_offset = scratch.attention.byte_offset;
		pass_add(launches_out, &count, kernels, VISION_ATTENTION,
			 record);
		record = pass_record(scratch.attention,
				     current->attention_output,
				     current->attention_output_bias,
				     scratch.normalized, rows, width, width);
		record.flags |= attention_output_flags;
		pass_add(launches_out, &count, kernels, VISION_LINEAR, record);
		record = pass_record(scratch.hidden, empty, empty,
				     scratch.hidden, rows, width, width);
		record.auxiliary_offset = scratch.normalized.byte_offset;
		pass_add(launches_out, &count, kernels, VISION_ADD, record);
		record = pass_record(scratch.hidden, current->norm2, empty,
				     scratch.normalized, rows, width, width);
		record.epsilon = config->epsilon;
		record.flags = current->norm2.byte_count ==
					       (uint64_t)width *
						       VISION_BFLOAT16_BYTES ?
				       VISION_NORM_BFLOAT16_WEIGHT :
				       0u;
		pass_add(launches_out, &count, kernels, VISION_RMS, record);
		record = pass_record(scratch.normalized, current->gate_up,
				     empty, scratch.gates, rows, width,
				     VISION_GATE_COMPONENTS *
					     config->intermediate_width);
		record.flags |= up_flags;
		pass_add(launches_out, &count, kernels, VISION_LINEAR, record);
		record = pass_record(
			scratch.gates, empty, empty, scratch.activated, rows,
			VISION_GATE_COMPONENTS * config->intermediate_width,
			config->intermediate_width);
		pass_add(launches_out, &count, kernels, VISION_SILU, record);
		record = pass_record(scratch.activated, current->down, empty,
				     scratch.normalized, rows,
				     config->intermediate_width, width);
		record.flags |= down_flags;
		pass_add(launches_out, &count, kernels, VISION_LINEAR, record);
		record = pass_record(scratch.hidden, empty, empty,
				     scratch.hidden, rows, width, width);
		record.auxiliary_offset = scratch.normalized.byte_offset;
		pass_add(launches_out, &count, kernels, VISION_ADD, record);
	}
	record = pass_record(scratch.hidden, weights->final_norm, empty,
			     scratch.normalized, rows, width, width);
	record.epsilon = config->epsilon;
	record.flags = weights->final_norm.byte_count ==
				       (uint64_t)width * VISION_BFLOAT16_BYTES ?
			       VISION_NORM_BFLOAT16_WEIGHT :
			       0u;
	pass_add(launches_out, &count, kernels, VISION_RMS, record);
	record = pass_record(scratch.normalized, empty, empty, scratch.unfolded,
			     cells, width, width * ratio * ratio);
	record.grid_height = description->grid_height;
	record.grid_width = description->grid_width;
	record.flags = ratio;
	pass_add(launches_out, &count, kernels, VISION_UNFOLD, record);
	record = pass_record(scratch.unfolded, weights->projector1,
			     weights->projector1_bias, scratch.projected, cells,
			     width * ratio * ratio, config->language_width);
	record.flags |= projector_first_flags;
	pass_add(launches_out, &count, kernels, VISION_LINEAR, record);
	record = pass_record(scratch.projected, empty, empty, scratch.projected,
			     cells, config->language_width,
			     config->language_width);
	pass_add(launches_out, &count, kernels, VISION_GELU, record);
	record = pass_record(scratch.projected, weights->projector2,
			     weights->projector2_bias, description->output,
			     cells, config->language_width,
			     config->language_width);
	record.flags |= projector_second_flags;
	pass_add(launches_out, &count, kernels, VISION_LINEAR, record);
	*launch_count_out = count;
	return STATUS_OK;
}

result_status vision_merge_prepare(const struct vision_weights *weights,
				   const struct linear_range *encoded,
				   const struct linear_range *output,
				   uint32_t language_width,
				   uint32_t grid_height, uint32_t grid_width,
				   uint32_t copies, uint64_t arena_byte_count,
				   const struct vision_kernels *kernels,
				   struct kernel_launch *launch_out)
{
	if (weights == NULL || encoded == NULL || output == NULL ||
	    kernels == NULL || launch_out == NULL || language_width == 0 ||
	    language_width > VISION_MAXIMUM_LANGUAGE_WIDTH ||
	    grid_height == 0 || grid_width == 0 ||
	    grid_height > VISION_MAXIMUM_TOKENS ||
	    grid_width > VISION_MAXIMUM_TOKENS || copies == 0 ||
	    copies > VISION_MAXIMUM_COPIES ||
	    kernels->operations[VISION_MERGE] == BACKEND_NO_KERNEL) {
		return STATUS_INVALID_ARGUMENT;
	}
	const uint32_t tokens =
		grid_height * (grid_width + 1u) + VISION_HALF_DIVISOR;
	const uint64_t row_bytes =
		(uint64_t)language_width * VISION_BFLOAT16_BYTES;
	const struct linear_range reads[] = { *encoded, weights->image_start,
					      weights->image_newline,
					      weights->image_end };
	const uint64_t sizes[] = { (uint64_t)grid_height * grid_width *
					   row_bytes,
				   row_bytes, row_bytes, row_bytes };
	if (tokens > VISION_MAXIMUM_TOKENS ||
	    !range_valid(*output, (uint64_t)tokens * copies * row_bytes,
			 arena_byte_count, false)) {
		return STATUS_INVALID_ARGUMENT;
	}
	for (uint32_t index = 0; index < sizeof(reads) / sizeof(reads[0]);
	     index++) {
		if (!range_valid(reads[index], sizes[index], arena_byte_count,
				 index != 0) ||
		    ranges_overlap(reads[index], *output)) {
			return STATUS_INVALID_ARGUMENT;
		}
	}
	struct vision_arguments record = pass_record(
		*encoded, weights->image_start, weights->image_newline, *output,
		tokens, language_width, language_width * copies);
	record.auxiliary_offset = weights->image_end.byte_offset;
	record.grid_height = grid_height;
	record.grid_width = grid_width;
	record.copies = copies;
	uint32_t count = 0;
	pass_add(launch_out, &count, kernels, VISION_MERGE, record);
	return STATUS_OK;
}

result_status vision_step_merge_prepare(
	struct linear_range embeddings, struct linear_range metadata,
	struct linear_range step, struct linear_range output, uint32_t rows,
	uint32_t sessions, uint64_t arena_byte_count,
	const struct vision_kernels *kernels, struct kernel_launch *launch_out)
{
	const uint32_t width = VISION_MAXIMUM_LANGUAGE_WIDTH;
	const uint32_t copies = VISION_MAXIMUM_COPIES;
	const uint64_t embedding_bytes =
		(uint64_t)sessions * VISION_MAXIMUM_IMAGES *
		VISION_MAXIMUM_TOKENS * width * VISION_BFLOAT16_BYTES;
	const uint64_t output_bytes =
		(uint64_t)rows * copies * width * VISION_BFLOAT16_BYTES;
	const uint64_t step_bytes =
		STEP_BLOCK_BYTES_VALUE + (uint64_t)rows * STEP_ROW_BYTES_VALUE;
	if (kernels == NULL || launch_out == NULL || sessions == 0 ||
	    rows == 0 || rows > MODEL_WIDE_MOST_ROWS ||
	    kernels->operations[VISION_STEP_MERGE] == BACKEND_NO_KERNEL ||
	    !range_valid(embeddings, embedding_bytes, arena_byte_count,
			 false) ||
	    !range_valid(metadata,
			 (uint64_t)sessions * VISION_SESSION_METADATA_BYTES,
			 arena_byte_count, false) ||
	    !range_valid(output, output_bytes, arena_byte_count, false) ||
	    step.byte_offset % STEP_BLOCK_BYTES_VALUE != 0 ||
	    step.byte_offset > arena_byte_count ||
	    step.byte_count > arena_byte_count - step.byte_offset ||
	    step.byte_count < step_bytes ||
	    ranges_overlap(embeddings, output) ||
	    ranges_overlap(metadata, output) || ranges_overlap(step, output)) {
		return STATUS_INVALID_ARGUMENT;
	}
	struct vision_arguments record =
		pass_record(embeddings, metadata, (struct linear_range){ 0 },
			    output, rows, width, width * copies);
	record.auxiliary_offset = step.byte_offset;
	record.copies = copies;
	uint32_t count = 0;
	pass_add(launch_out, &count, kernels, VISION_STEP_MERGE, record);
	return STATUS_OK;
}
