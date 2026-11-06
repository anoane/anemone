/* SPDX-License-Identifier: AGPL-3.0-or-later */
#include "anemone/official_hc_partitioned.h"

#include <stddef.h>
#include <string.h>
#include "prepare_common.h"

#define RANGE_COUNT 4u

result_status official_hc_partitioned_kernels_prepare(
	const struct backend_operations *backend, struct device_card *card,
	struct official_hc_partitioned_kernels *kernels_out)
{
	if (backend == NULL || backend->kernel_prepare == NULL ||
	    card == NULL || kernels_out == NULL)
		return STATUS_INVALID_ARGUMENT;
	struct official_hc_partitioned_kernels kernels = { BACKEND_NO_KERNEL,
							   BACKEND_NO_KERNEL };
	result_status status = backend->kernel_prepare(
		card, KERNEL_FILE_OFFICIAL_HC_PARTITIONED,
		KERNEL_OFFICIAL_HC_PARTIALS,
		OFFICIAL_HC_PARTITIONED_ARGUMENT_BYTES, &kernels.partials);
	if (status == STATUS_OK)
		status = backend->kernel_prepare(
			card, KERNEL_FILE_OFFICIAL_HC_PARTITIONED,
			KERNEL_OFFICIAL_HC_REDUCE,
			OFFICIAL_HC_PARTITIONED_ARGUMENT_BYTES,
			&kernels.reduce);
	if (status != STATUS_OK)
		return status;
	if (kernels.partials == BACKEND_NO_KERNEL ||
	    kernels.reduce == BACKEND_NO_KERNEL)
		return STATUS_INVALID_ARGUMENT;
	*kernels_out = kernels;
	return STATUS_OK;
}

result_status official_hc_partitioned_prepare(
	const struct dense_linear_description *description,
	const struct linear_range *partials, uint64_t arena_byte_count,
	const struct dense_linear_kernels *dense,
	const struct official_hc_kernels *official,
	const struct official_hc_partitioned_kernels *partitioned,
	struct kernel_launch launches_out[OFFICIAL_HC_PARTITIONED_LAUNCHES])
{
	if (description == NULL || partials == NULL || partitioned == NULL ||
	    launches_out == NULL ||
	    partitioned->partials == BACKEND_NO_KERNEL ||
	    partitioned->reduce == BACKEND_NO_KERNEL)
		return STATUS_INVALID_ARGUMENT;
	struct kernel_launch validated;
	const result_status status = official_hc_shape_prepare(
		description, 1, arena_byte_count, dense, official, &validated);
	if (status != STATUS_OK)
		return status;
	const uint32_t rows = description->row_count;
	const struct checked_range ranges[] = {
		{ "weights", &description->weights,
		  description->weights.byte_count, false },
		{ "rows", &description->rows, description->rows.byte_count,
		  false },
		{ "output", &description->output,
		  description->output.byte_count, false },
		{ "partials", partials,
		  (uint64_t)rows * MODEL_MIXES * OFFICIAL_HC_PARTITIONS *
			  sizeof(float),
		  false }
	};
	if (!prepare_ranges_are_valid("a partitioned HC projection", ranges,
				      RANGE_COUNT, arena_byte_count))
		return STATUS_INVALID_ARGUMENT;
	const struct official_hc_partitioned_arguments arguments = {
		.weights_offset = description->weights.byte_offset,
		.rows_offset = description->rows.byte_offset,
		.partials_offset = partials->byte_offset,
		.output_offset = description->output.byte_offset,
		.actual_rows = rows,
		.epsilon = description->epsilon
	};
	struct kernel_launch launches[OFFICIAL_HC_PARTITIONED_LAUNCHES];
	prepare_launch_fill(&launches[0], partitioned->partials, MODEL_MIXES,
			    OFFICIAL_HC_PARTITIONS, OFFICIAL_HC_PARTIAL_THREADS,
			    0, &arguments, sizeof(arguments));
	launches[0].grid[OFFICIAL_HC_ROW_GRID_DIMENSION] = rows;
	prepare_launch_fill(&launches[1], partitioned->reduce, 1, rows,
			    OFFICIAL_HC_THREADS, 0, &arguments,
			    sizeof(arguments));
	memcpy(launches_out, launches, sizeof(launches));
	return STATUS_OK;
}
