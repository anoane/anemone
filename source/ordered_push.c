/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Checked one-shot push launch preparation; default collective modules and arithmetic are unchanged. */
#include "anemone/ordered_push.h"
#include "ordered_collective_internal.h"
#include "ordered_push_internal.h"

#include <stddef.h>
#include <string.h>

#define PUSH_FP32_BYTES 4u
#define PUSH_BF16_BYTES 2u
#define PUSH_ADDRESS_BYTES 8u

static bool push_range_fits(uint64_t offset, uint64_t bytes, uint64_t capacity)
{
	return offset <= capacity && bytes <= capacity - offset;
}

static bool push_overlap(uint64_t first, uint64_t first_bytes, uint64_t second,
			 uint64_t second_bytes)
{
	return first < second + second_bytes && second < first + first_bytes;
}

result_status ordered_push_routes_prepare(
	const struct ordered_collective_route_description *description,
	uint32_t *host_writer_masks_out)
{
	if (description == NULL ||
	    description->operation != ORDERED_ALL_REDUCE ||
	    description->paired_scatter_gather)
		return STATUS_INVALID_ARGUMENT;
	return ordered_collective_routes_prepare_measured(
		description, NULL, 0, host_writer_masks_out);
}

result_status ordered_push_routes_prepare_capacity(
	const struct ordered_collective_route_description *description,
	uint32_t maximum_elements, uint32_t *host_writer_masks_out)
{
	if (description == NULL ||
	    description->operation != ORDERED_ALL_REDUCE ||
	    description->paired_scatter_gather)
		return STATUS_INVALID_ARGUMENT;
	struct ordered_collective_route_description baseline = *description;
	baseline.common_numa_root = false;
	return ordered_collective_routes_prepare_capacity(
		&baseline, maximum_elements, host_writer_masks_out);
}

result_status
ordered_push_kernel_prepare(const struct backend_operations *backend,
			    struct device_card *card, uint32_t *kernel_out)
{
	if (backend == NULL || backend->kernel_prepare == NULL ||
	    card == NULL || kernel_out == NULL)
		return STATUS_INVALID_ARGUMENT;
	return backend->kernel_prepare(card, KERNEL_FILE_ORDERED_PUSH,
				       KERNEL_ORDERED_PUSH,
				       ORDERED_PUSH_ARGUMENT_BYTES, kernel_out);
}

result_status ordered_push_prepare_capacity(
	const struct ordered_push_description *description, uint32_t kernel,
	uint32_t maximum_elements, struct kernel_launch *launch_out)
{
	if (description == NULL || launch_out == NULL ||
	    kernel == BACKEND_NO_KERNEL || maximum_elements == 0 ||
	    maximum_elements > UINT32_MAX - ORDERED_COLLECTIVE_THREADS)
		return STATUS_INVALID_ARGUMENT;
	const struct ordered_push_arguments *const arguments =
		&description->arguments;
	if (arguments->rank_count == 0 ||
	    arguments->rank_count > ORDERED_COLLECTIVE_MAXIMUM_RANKS ||
	    arguments->rank >= arguments->rank_count ||
	    arguments->element_count == 0 ||
	    arguments->element_count > maximum_elements ||
	    arguments->payload_format > ORDERED_PUSH_BF16 ||
	    arguments->reserved_zero != 0 || arguments->deadline_cycles == 0 ||
	    (arguments->incoming_host_mask &
	     ~((1u << arguments->rank_count) - 1u)) != 0 ||
	    (arguments->incoming_host_mask & (1u << arguments->rank)) != 0)
		return STATUS_INVALID_ARGUMENT;
	const uint64_t capacity = description->arena_byte_count;
	const uint64_t signal_bytes = (uint64_t)ORDERED_PUSH_PHASES *
				      arguments->rank_count *
				      ORDERED_COLLECTIVE_LINE_BYTES;
	const uint64_t value_bytes =
		(uint64_t)arguments->element_count * PUSH_FP32_BYTES;
	const uint64_t wire_bytes =
		(uint64_t)arguments->element_count *
		(arguments->payload_format == ORDERED_PUSH_FP32 ?
			 PUSH_FP32_BYTES :
			 PUSH_BF16_BYTES);
	const uint64_t table_bytes =
		(uint64_t)arguments->rank_count * ORDERED_COLLECTIVE_PEER_BYTES;
	if (arguments->region_offset % ORDERED_COLLECTIVE_LINE_BYTES != 0 ||
	    arguments->parity_stride % ORDERED_COLLECTIVE_LINE_BYTES != 0 ||
	    arguments->parity_stride > UINT64_MAX / ORDERED_COLLECTIVE_SLOTS ||
	    !push_range_fits(arguments->region_offset,
			     arguments->parity_stride *
				     ORDERED_COLLECTIVE_SLOTS,
			     capacity) ||
	    arguments->peer_table_offset % PUSH_ADDRESS_BYTES != 0 ||
	    arguments->step_offset % PUSH_ADDRESS_BYTES != 0 ||
	    !push_range_fits(arguments->peer_table_offset, table_bytes,
			     capacity) ||
	    !push_range_fits(arguments->step_offset, STEP_BLOCK_BYTES_VALUE,
			     capacity) ||
	    arguments->inbox_rank_stride < wire_bytes ||
	    arguments->inbox_rank_stride % PUSH_FP32_BYTES != 0 ||
	    arguments->inbox_rank_stride > UINT64_MAX / arguments->rank_count)
		return STATUS_INVALID_ARGUMENT;
	const uint64_t offsets[] = { arguments->input_offset,
				     arguments->output_offset,
				     arguments->inbox_offset };
	const uint64_t sizes[] = { value_bytes, value_bytes,
				   arguments->inbox_rank_stride *
					   arguments->rank_count };
	for (uint32_t array = 0; array < sizeof(offsets) / sizeof(offsets[0]);
	     array++) {
		if (offsets[array] < signal_bytes ||
		    offsets[array] % PUSH_FP32_BYTES != 0 ||
		    !push_range_fits(offsets[array], sizes[array],
				     arguments->parity_stride))
			return STATUS_INVALID_ARGUMENT;
		for (uint32_t previous = 0; previous < array; previous++) {
			if (push_overlap(offsets[array], sizes[array],
					 offsets[previous], sizes[previous]))
				return STATUS_INVALID_ARGUMENT;
		}
	}
	const uint64_t region_bytes =
		arguments->parity_stride * ORDERED_COLLECTIVE_SLOTS;
	if (push_overlap(arguments->peer_table_offset, table_bytes,
			 arguments->region_offset, region_bytes) ||
	    push_overlap(arguments->step_offset, STEP_BLOCK_BYTES_VALUE,
			 arguments->region_offset, region_bytes) ||
	    push_overlap(arguments->step_offset, STEP_BLOCK_BYTES_VALUE,
			 arguments->peer_table_offset, table_bytes))
		return STATUS_INVALID_ARGUMENT;
	struct kernel_launch launch = { 0 };
	launch.kernel = kernel;
	launch.grid[0] = launch.grid[1] = launch.grid[LAUNCH_DIMENSIONS - 1u] =
		1;
	launch.block[0] = ORDERED_COLLECTIVE_THREADS;
	launch.block[1] = launch.block[LAUNCH_DIMENSIONS - 1u] = 1;
	launch.argument_byte_count = ORDERED_PUSH_ARGUMENT_BYTES;
	memcpy(launch.argument, arguments, sizeof(*arguments));
	*launch_out = launch;
	return STATUS_OK;
}

result_status
ordered_push_prepare(const struct ordered_push_description *description,
		     uint32_t kernel, struct kernel_launch *launch_out)
{
	return ordered_push_prepare_capacity(
		description, kernel, ORDERED_COLLECTIVE_MAXIMUM_ELEMENTS,
		launch_out);
}
