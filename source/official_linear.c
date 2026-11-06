/* SPDX-License-Identifier: AGPL-3.0-or-later */
#include "anemone/official_linear.h"

#include <stddef.h>
#include <string.h>
#include "prepare_common.h"

MUST_CHECK static result_status
kernel_prepare(const struct backend_operations *backend,
	       struct device_card *card, const char *file, const char *name,
	       uint32_t argument_bytes, uint32_t *kernel_out)
{
	uint32_t kernel = BACKEND_NO_KERNEL;
	const result_status status = backend->kernel_prepare(
		card, file, name, argument_bytes, &kernel);
	if (status != STATUS_OK)
		return status;
	if (kernel == BACKEND_NO_KERNEL)
		return STATUS_INVALID_ARGUMENT;
	*kernel_out = kernel;
	return STATUS_OK;
}

result_status official_linear_kernels_prepare(
	const struct backend_operations *backend, struct device_card *card,
	uint32_t arithmetic_class, struct official_linear_kernels *kernels_out)
{
	if (backend == NULL || backend->kernel_prepare == NULL ||
	    card == NULL || kernels_out == NULL ||
	    (arithmetic_class != ARITHMETIC_CLASS_SM_80 &&
	     arithmetic_class != ARITHMETIC_CLASS_SM_120 &&
	     arithmetic_class != ARITHMETIC_CLASS_SM_120_NATIVE))
		return STATUS_INVALID_ARGUMENT;
	struct official_linear_kernels kernels = {
		.quantized = BACKEND_NO_KERNEL,
		.quantized_native = BACKEND_NO_KERNEL,
		.woa = BACKEND_NO_KERNEL
	};
	result_status status = kernel_prepare(
		backend, card, KERNEL_FILE_OFFICIAL_LINEAR,
		KERNEL_OFFICIAL_LINEAR, TRUNK_LINEAR_QUANTIZED_ARGUMENT_BYTES,
		&kernels.quantized);
	if (status == STATUS_OK &&
	    arithmetic_class == ARITHMETIC_CLASS_SM_120_NATIVE)
		status = kernel_prepare(backend, card,
					KERNEL_FILE_OFFICIAL_LINEAR,
					KERNEL_OFFICIAL_LINEAR_NATIVE,
					TRUNK_LINEAR_QUANTIZED_ARGUMENT_BYTES,
					&kernels.quantized_native);
	if (status == STATUS_OK)
		status = kernel_prepare(backend, card,
					KERNEL_FILE_OFFICIAL_LINEAR_WOA,
					KERNEL_OFFICIAL_LINEAR_WOA,
					TRUNK_LINEAR_BFLOAT16_ARGUMENT_BYTES,
					&kernels.woa);
	if (status != STATUS_OK)
		return status;
	*kernels_out = kernels;
	return STATUS_OK;
}

/* Retain the stock description checks without preparing or touching a card.
 * The numerical form does not use its slice geometry; every retained slice
 * handle names the one already prepared independent kernel. */
static struct linear_kernels_record validation_kernels(uint32_t kernel)
{
	struct linear_kernels_record kernels = { 0 };
	for (uint32_t instance = 0; instance < LINEAR_SLICE_INSTANCES;
	     instance++) {
		kernels.trunk_linear_quantized[instance] = kernel;
		kernels.trunk_linear_bfloat16[instance] = kernel;
	}
	return kernels;
}

result_status official_linear_quantized_prepare(
	const struct linear_trunk_quantized_description *description,
	uint64_t arena_byte_count,
	const struct official_linear_kernels *kernels,
	enum official_linear_form form, struct kernel_launch *launch_out)
{
	if (description == NULL || kernels == NULL || launch_out == NULL ||
	    (form != OFFICIAL_LINEAR_PORTABLE &&
	     form != OFFICIAL_LINEAR_NATIVE) ||
	    description->row_count == 0 ||
	    description->row_count > LINEAR_MAXIMUM_ROWS ||
	    description->matrix_count == 0 ||
	    description->matrix_count > LINEAR_MERGED_MATRICES)
		return STATUS_INVALID_ARGUMENT;
	const uint32_t kernel = form == OFFICIAL_LINEAR_NATIVE ?
					kernels->quantized_native :
					kernels->quantized;
	if (kernel == BACKEND_NO_KERNEL)
		return STATUS_INVALID_ARGUMENT;
	for (uint32_t matrix = 0; matrix < description->matrix_count; matrix++)
		if (form == OFFICIAL_LINEAR_NATIVE &&
		    description->matrices[matrix].shape.output_size %
				    OFFICIAL_LINEAR_NATIVE_OUTPUTS !=
			    0)
			return STATUS_INVALID_ARGUMENT;
	const struct linear_kernels_record validation =
		validation_kernels(kernel);
	struct kernel_launch launch;
	const result_status status = linear_trunk_quantized_prepare(
		description, arena_byte_count, &validation, &launch);
	if (status != STATUS_OK)
		return status;
	struct trunk_linear_quantized_arguments arguments;
	memcpy(&arguments, launch.argument, sizeof(arguments));
	const uint32_t outputs =
		arguments.output_size[0] + arguments.output_size[1];
	const bool native = form == OFFICIAL_LINEAR_NATIVE;
	prepare_launch_fill(
		&launch, kernel,
		outputs / (native ? OFFICIAL_LINEAR_NATIVE_OUTPUTS :
				    OFFICIAL_LINEAR_PORTABLE_OUTPUTS),
		native ? 1 : description->row_count, OFFICIAL_LINEAR_THREADS, 0,
		&arguments, sizeof(arguments));
	*launch_out = launch;
	return STATUS_OK;
}

result_status official_linear_woa_prepare(
	const struct linear_trunk_bfloat16_description *description,
	uint64_t arena_byte_count,
	const struct official_linear_kernels *kernels,
	struct kernel_launch *launch_out)
{
	if (description == NULL || kernels == NULL || launch_out == NULL ||
	    kernels->woa == BACKEND_NO_KERNEL || description->row_count == 0 ||
	    description->row_count > LINEAR_MAXIMUM_ROWS ||
	    description->segment_count != OFFICIAL_LINEAR_WOA_GROUPS ||
	    description->matrix.shape.input_size != OFFICIAL_LINEAR_WOA_INNER ||
	    description->matrix.shape.output_size !=
		    OFFICIAL_LINEAR_WOA_GROUPS * OFFICIAL_LINEAR_WOA_COLUMNS)
		return STATUS_INVALID_ARGUMENT;
	const struct linear_kernels_record validation =
		validation_kernels(kernels->woa);
	struct kernel_launch launch;
	const result_status status = linear_trunk_bfloat16_prepare(
		description, arena_byte_count, &validation, &launch);
	if (status != STATUS_OK)
		return status;
	struct trunk_linear_bfloat16_arguments arguments;
	memcpy(&arguments, launch.argument, sizeof(arguments));
	prepare_launch_fill(&launch, kernels->woa,
			    OFFICIAL_LINEAR_WOA_COLUMNS /
				    OFFICIAL_LINEAR_WOA_OUTPUTS,
			    description->row_count, OFFICIAL_LINEAR_THREADS, 0,
			    &arguments, sizeof(arguments));
	launch.grid[LAUNCH_DIMENSIONS - 1] = OFFICIAL_LINEAR_WOA_GROUPS;
	*launch_out = launch;
	return STATUS_OK;
}
