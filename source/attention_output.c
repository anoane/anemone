/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * attention_output.c: the launches of the attention site's output span. Contract: attention_output.h; the kernels:
 * attention_output_records.h. A description is first handed, as the descriptions of its separate launches, to their
 * prepare functions (linear.c), and their launches are discarded: what they refuse is refused here, behind their one
 * error line. What the fusion adds, the post's ranges and every range against every other, is checked here.
 */
#include "anemone/attention_output.h"

#include <inttypes.h>
#include <stddef.h>

#include "anemone/hooks.h"
#include "anemone/model_layout.h"
#include "prepare_common.h"

#define OK STATUS_OK
#define THEN(call)        \
	(status != OK ?   \
		 status : \
		 (call)) /* the next call while every one before passed */
#define COUNT_OF(table) ((uint32_t)(sizeof(table) / sizeof((table)[0])))
#define INSTANCES LINEAR_SLICE_INSTANCES
#define WARP_THREADS LINEAR_THREADS_PER_WARP
#define CLASS_MEMBERS \
	2u /* of struct attention_output_kernels beside the handles */

enum kernel_kind { GROUPS_KERNEL, PROJECTION_KERNEL, KERNEL_KINDS };

/* Each count the kernels are stamped for, with the functions of the two for it. */
struct slice_instance {
	uint32_t slice_count;
	const char *function_names[KERNEL_KINDS];
};
#define LINEAR_SLICE_STAMP(slices)                \
	{ (slices),                               \
	  { KERNEL_ATTENTION_GROUPS_STEM #slices, \
	    KERNEL_ATTENTION_PROJECTION_STEM #slices } },
static const struct slice_instance slice_instances[] = {
	LINEAR_EVERY_SLICE_COUNT
};
#undef LINEAR_SLICE_STAMP
static const uint32_t argument_byte_counts[KERNEL_KINDS] = {
	ATTENTION_GROUPS_ARGUMENT_BYTES, ATTENTION_PROJECTION_ARGUMENT_BYTES
};

_Static_assert(COUNT_OF(slice_instances) == INSTANCES,
	       "an instance of each kernel for each stamped count");
_Static_assert(
	sizeof(struct attention_output_kernels) ==
		(KERNEL_KINDS * INSTANCES + CLASS_MEMBERS) * sizeof(uint32_t),
	"the handles of the two kernels, then the class and the reserved field");
_Static_assert(ATTENTION_PROJECTION_ARGUMENT_BYTES <= LAUNCH_ARGUMENT_BYTES,
	       "the larger record fits the argument of a launch");

result_status attention_output_kernels_prepare(
	const struct backend_operations *backend, struct device_card *card,
	uint32_t arithmetic_class, struct attention_output_kernels *kernels_out)
{
	const char *const file_name =
		arithmetic_class == ARITHMETIC_CLASS_SM_120_NATIVE ?
			KERNEL_FILE_ATTENTION_OUTPUT_NATIVE :
			KERNEL_FILE_ATTENTION_OUTPUT;
	struct attention_output_kernels kernels = { .arithmetic_class =
							    arithmetic_class };
	if (backend == NULL || backend->kernel_prepare == NULL ||
	    card == NULL || kernels_out == NULL) {
		LOG_ERROR(
			"the kernels of the attention's output: a missing argument");
		return STATUS_INVALID_ARGUMENT;
	}
	for (uint32_t instance = 0; instance < INSTANCES; instance++) {
		uint32_t *const handles[KERNEL_KINDS] = {
			&kernels.groups[instance], &kernels.projection[instance]
		};
		for (uint32_t kind = 0; kind < KERNEL_KINDS; kind++) {
			const char *const function_name =
				slice_instances[instance].function_names[kind];
			/* a backend that answers OK and gives no handle leaves this one, which a prepare refuses */
			*handles[kind] = BACKEND_NO_KERNEL;
			const result_status status = backend->kernel_prepare(
				card, file_name, function_name,
				argument_byte_counts[kind], handles[kind]);
			if (status != OK) {
				LOG_ERROR(
					"the kernels of the attention's output: %s of %s is not prepared: %s",
					function_name, file_name,
					status_text(status));
				return status;
			}
		}
	}
	*kernels_out = kernels;
	return OK;
}

/* The slices a launch takes of a matrix's slice_count: on class 120's emulation at most
 * LINEAR_CLASS_120_EXACT_MOST_SLICES, whose bits are those of any count (attention_output_records.h); else the
 * matrix's. */
static uint32_t launched_slices(const struct attention_output_kernels *kernels,
				uint32_t slice_count)
{
	const bool is_emulated = kernels->arithmetic_class ==
				 ARITHMETIC_CLASS_SM_120;
	return is_emulated && slice_count > LINEAR_CLASS_120_EXACT_MOST_SLICES ?
		       LINEAR_CLASS_120_EXACT_MOST_SLICES :
		       slice_count;
}

/* The place of a slice count among the stamped counts; the separate product's prepare has checked the matrix's. */
static uint32_t instance_of(uint32_t slice_count)
{
	uint32_t instance = 0;
	while (instance + 1u < INSTANCES &&
	       slice_instances[instance].slice_count != slice_count) {
		instance++;
	}
	return instance;
}

static bool arguments_are_given(const char *what, const void *description,
				const void *linear_kernels, const void *kernels,
				const void *launch_out)
{
	return prepare_arguments_are_given(what, description, kernels,
					   launch_out) &&
	       prepare_arguments_are_given(what, linear_kernels, kernels,
					   launch_out);
}

result_status
attention_groups_prepare(const struct attention_groups_description *description,
			 uint64_t arena_byte_count,
			 const struct linear_kernels_record *linear_kernels,
			 const struct attention_output_kernels *kernels,
			 struct kernel_launch *launch_out)
{
	static const char what[] = "the attention's output groups, quantized";
	struct kernel_launch separate;
	if (!arguments_are_given(what, description, linear_kernels, kernels,
				 launch_out)) {
		return STATUS_INVALID_ARGUMENT;
	}
	const struct linear_matrix *const matrix = &description->matrix;
	const uint32_t output_size = matrix->shape.output_size,
		       row_count = description->row_count;
	const struct linear_trunk_bfloat16_description product = {
		*matrix, description->values, description->groups, row_count,
		description->segment_count
	};
	const struct linear_quantize_description quantize = {
		description->groups,
		description->codes,
		description->exponents,
		description->operands,
		output_size,
		row_count
	};
	result_status status = linear_trunk_bfloat16_prepare(
		&product, arena_byte_count, linear_kernels, &separate);
	status = THEN(linear_quantize_prepare(&quantize, arena_byte_count,
					      linear_kernels, &separate));
	if (status != OK) {
		return status;
	}
	const uint32_t slice_count =
		launched_slices(kernels, matrix->slice_count);
	const uint32_t instance = instance_of(slice_count);
	const uint64_t rows = row_count, outputs = output_size;
	const struct checked_range ranges[] = {
		{ "the matrix", &matrix->stored, matrix->stored.byte_count,
		  true },
		{ "the values", &description->values,
		  linear_values_byte_count(
			  rows, (uint64_t)description->segment_count *
					matrix->shape.input_size),
		  false },
		{ "the groups", &description->groups,
		  linear_values_byte_count(rows, outputs), false },
		{ "the codes", &description->codes,
		  linear_codes_byte_count(rows, outputs), false },
		{ "the exponents", &description->exponents,
		  linear_exponents_byte_count(rows, outputs), false },
		{ "the operands", &description->operands,
		  linear_values_byte_count(rows, outputs), false },
	};
	if (!prepare_kernel_is_prepared(
		    what, kernels->groups[instance],
		    slice_instances[instance].function_names[GROUPS_KERNEL]) ||
	    !prepare_ranges_are_valid(what, ranges, COUNT_OF(ranges),
				      arena_byte_count)) {
		return STATUS_INVALID_ARGUMENT;
	}
	const struct attention_groups_arguments record = {
		.weights_offset = matrix->stored.byte_offset,
		.values_offset = description->values.byte_offset,
		.groups_offset = description->groups.byte_offset,
		.codes_offset = description->codes.byte_offset,
		.exponents_offset = description->exponents.byte_offset,
		.operands_offset = description->operands.byte_offset,
		.output_size = output_size,
		.input_size = matrix->shape.input_size,
		.row_count = row_count,
		.segment_count = description->segment_count,
	};
	const uint32_t block_threads =
		ATTENTION_GROUP_TILES * slice_count * WARP_THREADS;
	prepare_launch_fill(launch_out, kernels->groups[instance],
			    output_size / LINEAR_VALUES_PER_BLOCK, 1,
			    block_threads,
			    block_threads * LINEAR_SHARED_BYTES_PER_THREAD,
			    &record, ATTENTION_GROUPS_ARGUMENT_BYTES);
	return OK;
}

result_status attention_projection_prepare(
	const struct attention_projection_description *description,
	uint64_t arena_byte_count,
	const struct linear_kernels_record *linear_kernels,
	const struct attention_output_kernels *kernels,
	struct kernel_launch *launch_out)
{
	static const char what[] = "the attention's output projection and post";
	struct kernel_launch separate;
	if (!arguments_are_given(what, description, linear_kernels, kernels,
				 launch_out)) {
		return STATUS_INVALID_ARGUMENT;
	}
	const struct linear_matrix *const matrix = &description->matrix;
	const struct linear_trunk_quantized_description product = {
		.matrices = { *matrix },
		.codes = description->codes,
		.exponents = description->exponents,
		.operands = description->operands,
		.outputs = { description->output },
		.matrix_count = 1,
		.row_count = description->row_count,
	};
	const result_status status = linear_trunk_quantized_prepare(
		&product, arena_byte_count, linear_kernels, &separate);
	if (status != OK ||
	    !prepare_reserved_is_zero(what, description->reserved_zero)) {
		return status != OK ? status : STATUS_INVALID_ARGUMENT;
	}
	if (matrix->shape.output_size != MODEL_HIDDEN) {
		LOG_ERROR("%s: %" PRIu32
			  " outputs: the post takes the %u values of a stream",
			  what, matrix->shape.output_size, MODEL_HIDDEN);
		return STATUS_INVALID_ARGUMENT;
	}
	const uint32_t slice_count =
		launched_slices(kernels, matrix->slice_count);
	const uint32_t instance = instance_of(slice_count);
	const uint64_t rows = description->row_count,
		       inputs = matrix->shape.input_size;
	const struct checked_range ranges[] = {
		{ "the matrix", &matrix->stored, matrix->stored.byte_count,
		  true },
		{ "the codes", &description->codes,
		  linear_codes_byte_count(rows, inputs), false },
		{ "the exponents", &description->exponents,
		  linear_exponents_byte_count(rows, inputs), false },
		{ "the operands", &description->operands,
		  linear_values_byte_count(rows, inputs), false },
		{ "the output", &description->output,
		  rows * MODEL_HIDDEN_ROW_BYTES, false },
		{ "the streams", &description->streams,
		  rows * MODEL_STREAMS_ROW_BYTES, false },
		{ "the post", &description->post, rows * MODEL_SITE_ROW_BYTES,
		  false },
		{ "the comb", &description->comb, rows * MODEL_COMB_ROW_BYTES,
		  false },
	};
	if (!prepare_kernel_is_prepared(
		    what, kernels->projection[instance],
		    slice_instances[instance]
			    .function_names[PROJECTION_KERNEL]) ||
	    !prepare_ranges_are_valid(what, ranges, COUNT_OF(ranges),
				      arena_byte_count)) {
		return STATUS_INVALID_ARGUMENT;
	}
	const struct attention_projection_arguments record = {
		.weights_offset = matrix->stored.byte_offset,
		.codes_offset = description->codes.byte_offset,
		.exponents_offset = description->exponents.byte_offset,
		.operands_offset = description->operands.byte_offset,
		.output_offset = description->output.byte_offset,
		.streams_offset = description->streams.byte_offset,
		.post_offset = description->post.byte_offset,
		.comb_offset = description->comb.byte_offset,
		.output_size = matrix->shape.output_size,
		.input_size = matrix->shape.input_size,
		.row_count = description->row_count,
		.reserved_zero = 0,
	};
	const uint32_t block_threads = slice_count * WARP_THREADS;
	prepare_launch_fill(launch_out, kernels->projection[instance],
			    matrix->shape.output_size / LINEAR_OUTPUTS_PER_TILE,
			    1, block_threads,
			    block_threads * LINEAR_SHARED_BYTES_PER_THREAD,
			    &record, ATTENTION_PROJECTION_ARGUMENT_BYTES);
	return OK;
}
