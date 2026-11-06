/* SPDX-License-Identifier: AGPL-3.0-or-later */
#include "anemone/official_head.h"

#include <stddef.h>
#include "prepare_common.h"

#define RANGE_COUNT 3u

MUST_CHECK static result_status
prepare_kernel(const struct backend_operations *backend,
	       struct device_card *card, const char *function,
	       uint32_t *kernel_out)
{
	if (backend == NULL || backend->kernel_prepare == NULL ||
	    card == NULL || kernel_out == NULL)
		return STATUS_INVALID_ARGUMENT;
	uint32_t kernel = BACKEND_NO_KERNEL;
	const result_status status = backend->kernel_prepare(
		card, KERNEL_FILE_OFFICIAL_HEAD, function,
		OFFICIAL_HEAD_ARGUMENT_BYTES, &kernel);
	if (status != STATUS_OK)
		return status;
	if (kernel == BACKEND_NO_KERNEL)
		return STATUS_INVALID_ARGUMENT;
	*kernel_out = kernel;
	return STATUS_OK;
}

result_status
official_head_kernel_prepare(const struct backend_operations *backend,
			     struct device_card *card, uint32_t *kernel_out)
{
	return prepare_kernel(backend, card, KERNEL_OFFICIAL_HEAD, kernel_out);
}

result_status
official_head_reused_kernel_prepare(const struct backend_operations *backend,
				    struct device_card *card,
				    uint32_t *kernel_out)
{
	return prepare_kernel(backend, card, KERNEL_OFFICIAL_HEAD_REUSED,
			      kernel_out);
}

result_status
official_head_prepare(const struct official_head_description *description,
		      uint64_t arena_byte_count, uint32_t kernel,
		      struct kernel_launch *launch_out)
{
	if (description == NULL || launch_out == NULL ||
	    kernel == BACKEND_NO_KERNEL)
		return STATUS_INVALID_ARGUMENT;
	const uint32_t rows = description->row_count;
	const uint32_t outputs = description->output_size;
	if ((rows != 1 && rows != OFFICIAL_HEAD_ODD_ROWS &&
	     rows != MODEL_STREAMS && rows != LINEAR_MAXIMUM_ROWS) ||
	    outputs == 0 || outputs > MODEL_VOCABULARY_VALUE ||
	    outputs % OFFICIAL_HEAD_OUTPUTS != 0)
		return STATUS_INVALID_ARGUMENT;
	const struct checked_range ranges[] = {
		{ "weights", &description->weights,
		  (uint64_t)outputs * MODEL_HIDDEN * sizeof(uint16_t), true },
		{ "rows", &description->rows,
		  (uint64_t)rows * MODEL_HIDDEN * sizeof(uint16_t), false },
		{ "output", &description->output,
		  (uint64_t)rows * outputs * sizeof(float), false }
	};
	if (!prepare_ranges_are_valid("a canonical M1 head", ranges,
				      RANGE_COUNT, arena_byte_count))
		return STATUS_INVALID_ARGUMENT;
	const struct official_head_arguments arguments = {
		.weights_offset = description->weights.byte_offset,
		.rows_offset = description->rows.byte_offset,
		.output_offset = description->output.byte_offset,
		.output_size = outputs,
		.row_count = rows
	};
	struct kernel_launch launch;
	prepare_launch_fill(&launch, kernel, outputs / OFFICIAL_HEAD_OUTPUTS,
			    rows, OFFICIAL_HEAD_THREADS, 0, &arguments,
			    sizeof(arguments));
	launch.block[0] = OFFICIAL_HEAD_OWNERS;
	launch.block[1] = OFFICIAL_HEAD_OUTPUTS;
	*launch_out = launch;
	return STATUS_OK;
}

result_status official_head_reused_prepare(
	const struct official_head_description *description,
	uint64_t arena_byte_count, uint32_t kernel,
	struct kernel_launch *launch_out)
{
	if (launch_out == NULL)
		return STATUS_INVALID_ARGUMENT;
	struct kernel_launch launch;
	const result_status status = official_head_prepare(
		description, arena_byte_count, kernel, &launch);
	if (status != STATUS_OK)
		return status;
	launch.grid[1] = 1;
	*launch_out = launch;
	return STATUS_OK;
}
