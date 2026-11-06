/* SPDX-License-Identifier: AGPL-3.0-or-later */
#include "anemone/official_hc.h"

#include <stddef.h>
#include <string.h>

result_status
official_hc_staged_prepare(const struct backend_operations *backend,
			   struct device_card *card,
			   struct official_hc_kernels *kernels)
{
	if (backend == NULL || backend->kernel_prepare == NULL ||
	    card == NULL || kernels == NULL)
		return STATUS_INVALID_ARGUMENT;
	uint32_t handle = BACKEND_NO_KERNEL;
	const result_status status = backend->kernel_prepare(
		card, KERNEL_FILE_OFFICIAL_HC, KERNEL_OFFICIAL_HC_STAGED,
		DENSE_ARGUMENT_BYTES, &handle);
	if (status != STATUS_OK)
		return status;
	if (handle == BACKEND_NO_KERNEL)
		return STATUS_INVALID_ARGUMENT;
	kernels->split = handle;
	return STATUS_OK;
}

result_status
official_hc_kernels_prepare(const struct backend_operations *backend,
			    struct device_card *card,
			    struct official_hc_kernels *kernels_out)
{
	if (backend == NULL || backend->kernel_prepare == NULL ||
	    card == NULL || kernels_out == NULL)
		return STATUS_INVALID_ARGUMENT;
	struct official_hc_kernels kernels = { .dot = BACKEND_NO_KERNEL,
					       .split = BACKEND_NO_KERNEL };
	result_status status = backend->kernel_prepare(
		card, KERNEL_FILE_OFFICIAL_HC, KERNEL_OFFICIAL_HC_DOT,
		DENSE_ARGUMENT_BYTES, &kernels.dot);
	if (status == STATUS_OK)
		status = backend->kernel_prepare(card, KERNEL_FILE_OFFICIAL_HC,
						 KERNEL_OFFICIAL_HC_SPLIT,
						 DENSE_ARGUMENT_BYTES,
						 &kernels.split);
	if (status == STATUS_OK && (kernels.dot == BACKEND_NO_KERNEL ||
				    kernels.split == BACKEND_NO_KERNEL))
		status = STATUS_INVALID_ARGUMENT;
	if (status == STATUS_OK)
		*kernels_out = kernels;
	return status;
}

result_status
official_hc_shape_prepare(const struct dense_linear_description *description,
			  uint32_t reference_rows, uint64_t arena_byte_count,
			  const struct dense_linear_kernels *dense,
			  const struct official_hc_kernels *official,
			  struct kernel_launch *launch_out)
{
	if (description == NULL || dense == NULL || official == NULL ||
	    launch_out == NULL)
		return STATUS_INVALID_ARGUMENT;
	const uint32_t rows = description->row_count;
	if ((rows != 1 && rows != OFFICIAL_HC_ODD_PROMPT_ROWS &&
	     rows != MODEL_STREAMS && rows != LINEAR_MAXIMUM_ROWS) ||
	    (reference_rows != 1 &&
	     reference_rows != OFFICIAL_HC_ODD_PROMPT_ROWS &&
	     reference_rows != MODEL_STREAMS &&
	     reference_rows != LINEAR_MAXIMUM_ROWS) ||
	    description->input_size != OFFICIAL_HC_INPUTS ||
	    description->output_size != MODEL_MIXES ||
	    description->weight_type != DENSE_FLOAT32 ||
	    description->output_type != DENSE_FLOAT32 ||
	    description->flags != DENSE_ROW_SCALE ||
	    description->output_scale != 1.0f ||
	    description->epsilon != MODEL_NORM_EPSILON)
		return STATUS_INVALID_ARGUMENT;
	const uint32_t kernel = reference_rows == 1 ? official->dot :
						      official->split;
	if (kernel == BACKEND_NO_KERNEL)
		return STATUS_INVALID_ARGUMENT;
	struct kernel_launch launch;
	const result_status status = dense_linear_prepare(
		description, arena_byte_count, dense, &launch);
	if (status == STATUS_OK) {
		struct dense_linear_arguments arguments;
		memcpy(&arguments, launch.argument, sizeof(arguments));
		arguments.row_count = reference_rows;
		memcpy(launch.argument, &arguments, sizeof(arguments));
		launch.kernel = kernel;
		launch.grid[0] = MODEL_MIXES;
		launch.grid[1] = rows;
		launch.block[0] = OFFICIAL_HC_THREADS;
		*launch_out = launch;
	}
	return status;
}

result_status
official_hc_prepare(const struct dense_linear_description *description,
		    uint64_t arena_byte_count,
		    const struct dense_linear_kernels *dense,
		    const struct official_hc_kernels *official,
		    struct kernel_launch *launch_out)
{
	if (description == NULL)
		return STATUS_INVALID_ARGUMENT;
	return official_hc_shape_prepare(description, description->row_count,
					 arena_byte_count, dense, official,
					 launch_out);
}
