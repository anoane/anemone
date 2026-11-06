/* SPDX-License-Identifier: AGPL-3.0-or-later */
#include "anemone/official_router_projection.h"

#include <stddef.h>
#include "prepare_common.h"

#define RANGE_COUNT 3u

result_status official_router_projection_kernel_prepare(
	const struct backend_operations *backend, struct device_card *card,
	uint32_t *kernel_out)
{
	if (backend == NULL || backend->kernel_prepare == NULL ||
	    card == NULL || kernel_out == NULL)
		return STATUS_INVALID_ARGUMENT;
	uint32_t kernel = BACKEND_NO_KERNEL;
	const result_status status = backend->kernel_prepare(
		card, KERNEL_FILE_OFFICIAL_ROUTER_PROJECTION,
		KERNEL_OFFICIAL_ROUTER_PROJECTION,
		OFFICIAL_ROUTER_PROJECTION_ARGUMENT_BYTES, &kernel);
	if (status != STATUS_OK)
		return status;
	if (kernel == BACKEND_NO_KERNEL)
		return STATUS_INVALID_ARGUMENT;
	*kernel_out = kernel;
	return STATUS_OK;
}

result_status official_router_projection_prepare(
	const struct official_router_projection_description *description,
	uint64_t arena_byte_count, uint32_t kernel,
	struct kernel_launch *launch_out)
{
	if (description == NULL || launch_out == NULL ||
	    kernel == BACKEND_NO_KERNEL)
		return STATUS_INVALID_ARGUMENT;
	const uint32_t rows = description->row_count;
	if (rows != 1 && rows != OFFICIAL_ROUTER_PROJECTION_ODD_ROWS &&
	    rows != MODEL_STREAMS && rows != LINEAR_MAXIMUM_ROWS)
		return STATUS_INVALID_ARGUMENT;
	const struct checked_range ranges[] = {
		{ "weights", &description->weights,
		  (uint64_t)MODEL_EXPERTS * MODEL_HIDDEN * sizeof(uint16_t),
		  true },
		{ "rows", &description->rows,
		  (uint64_t)rows * MODEL_HIDDEN * sizeof(uint16_t), false },
		{ "output", &description->output,
		  (uint64_t)rows * MODEL_EXPERTS * sizeof(float), false }
	};
	if (!prepare_ranges_are_valid("a cached-M1 router projection", ranges,
				      RANGE_COUNT, arena_byte_count))
		return STATUS_INVALID_ARGUMENT;
	const struct official_router_projection_arguments arguments = {
		.weights_offset = description->weights.byte_offset,
		.rows_offset = description->rows.byte_offset,
		.output_offset = description->output.byte_offset,
		.row_count = rows
	};
	struct kernel_launch launch;
	prepare_launch_fill(&launch, kernel,
			    MODEL_EXPERTS / OFFICIAL_ROUTER_PROJECTION_OUTPUTS,
			    rows, OFFICIAL_ROUTER_PROJECTION_THREADS, 0,
			    &arguments, sizeof(arguments));
	launch.block[0] = OFFICIAL_ROUTER_PROJECTION_OWNERS;
	launch.block[1] = OFFICIAL_ROUTER_PROJECTION_OUTPUTS;
	*launch_out = launch;
	return STATUS_OK;
}
