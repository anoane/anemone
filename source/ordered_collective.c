/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* ordered_collective.c: checked, immutable launches for the small ordered peer collective. */
#include "anemone/ordered_collective.h"
#include "ordered_collective_internal.h"

#include <stddef.h>
#include <string.h>
#include <fenv.h>

#define VALUE_BYTES 4u
#define ADDRESS_BYTES 8u
#define ROUTE_CMP_COUNT 4u
#define ROUTE_MIXED_COUNT 5u
#define ROUTE_CMP_GENERATION 2u
#define ROUTE_PRO_GENERATION 5u
#define ROUTE_PCIE_LANES 16u
#define ROUTE_MIXED_MINIMUM_ELEMENTS 16u
#define ROUTE_HOST_MINIMUM_ELEMENTS 32u /* 128 B: all-reduce and two-shot. */
#define ROUTE_PAIRED_HOST_MINIMUM_ELEMENTS \
	128u /* 512 B: paired scatter/gather. */

result_status ordered_collective_routes_prepare_capacity(
	const struct ordered_collective_route_description *description,
	uint32_t maximum_elements, uint32_t *host_reader_masks_out)
{
	if (description == NULL || host_reader_masks_out == NULL ||
	    maximum_elements == 0 ||
	    maximum_elements > UINT32_MAX - ORDERED_COLLECTIVE_THREADS ||
	    description->cards == NULL ||
	    description->direct_reader_masks == NULL ||
	    description->rank_count == 0 ||
	    description->rank_count > ORDERED_COLLECTIVE_MAXIMUM_RANKS ||
	    description->element_count == 0 ||
	    description->element_count > maximum_elements ||
	    description->operation < ORDERED_ALL_REDUCE ||
	    description->operation >= ORDERED_COLLECTIVE_OPERATIONS)
		return STATUS_INVALID_ARGUMENT;
	const bool separate_phase = description->operation ==
					    ORDERED_REDUCE_SCATTER ||
				    description->operation == ORDERED_GATHER;
	if (description->paired_scatter_gather && !separate_phase)
		return STATUS_INVALID_ARGUMENT;
	const uint32_t cmp_minimum =
		separate_phase ? ROUTE_PAIRED_HOST_MINIMUM_ELEMENTS :
				 ROUTE_HOST_MINIMUM_ELEMENTS;
	const uint32_t count = description->rank_count;
	const uint32_t all = (1u << count) - 1u;
	uint32_t host[ORDERED_COLLECTIVE_MAXIMUM_RANKS] = { 0 };
	uint32_t cmp = 0, pro = 0;
	for (uint32_t source = 0; source < count; source++) {
		const struct card_description_record *const card =
			&description->cards[source];
		const uint32_t direct =
			description->direct_reader_masks[source];
		if ((direct & ~all) != 0 || (direct & (1u << source)) == 0 ||
		    card->reserved_zero != 0 ||
		    memchr(card->name, 0, sizeof(card->name)) == NULL)
			return STATUS_INVALID_ARGUMENT;
		host[source] = all & ~direct;
		if (card->arithmetic_class == ARITHMETIC_CLASS_SM_80 &&
		    card->link_generation == ROUTE_CMP_GENERATION &&
		    card->link_lane_count == ROUTE_PCIE_LANES &&
		    strstr(card->name, "CMP 170HX") != NULL)
			cmp |= 1u << source;
		if (card->arithmetic_class == ARITHMETIC_CLASS_SM_120 &&
		    card->link_generation == ROUTE_PRO_GENERATION &&
		    card->link_lane_count == ROUTE_PCIE_LANES &&
		    strstr(card->name, "RTX PRO 6000") != NULL)
			pro |= 1u << source;
	}
	bool four_clique = count == ROUTE_CMP_COUNT && cmp == all;
	bool mixed_group = count == ROUTE_MIXED_COUNT && (cmp | pro) == all;
	uint32_t cmp_count = 0, pro_count = 0;
	for (uint32_t rank = 0; rank < count; rank++) {
		cmp_count += (cmp >> rank) & 1u;
		pro_count += (pro >> rank) & 1u;
		four_clique = four_clique &&
			      description->direct_reader_masks[rank] == all;
		mixed_group = mixed_group &&
			      description->direct_reader_masks[rank] ==
				      ((cmp & (1u << rank)) != 0 ? cmp : pro);
	}
	mixed_group = mixed_group && cmp_count == ROUTE_CMP_COUNT &&
		      pro_count == 1u;
	if (description->common_numa_root &&
	    (!separate_phase || description->paired_scatter_gather) &&
	    ((mixed_group &&
	      description->element_count >= ROUTE_MIXED_MINIMUM_ELEMENTS) ||
	     (four_clique && description->element_count >= cmp_minimum))) {
		for (uint32_t source = 0; source < count; source++)
			host[source] = all & ~(1u << source);
	}
	memcpy(host_reader_masks_out, host, (size_t)count * sizeof(host[0]));
	return STATUS_OK;
}

result_status ordered_collective_routes_prepare(
	const struct ordered_collective_route_description *description,
	uint32_t *host_reader_masks_out)
{
	return ordered_collective_routes_prepare_capacity(
		description, ORDERED_COLLECTIVE_MAXIMUM_ELEMENTS,
		host_reader_masks_out);
}

static bool range_fits(uint64_t offset, uint64_t bytes, uint64_t capacity)
{
	return offset <= capacity && bytes <= capacity - offset;
}

static bool ranges_overlap(uint64_t first, uint64_t first_bytes,
			   uint64_t second, uint64_t second_bytes)
{
	return first < second + second_bytes && second < first + first_bytes;
}

result_status ordered_collective_kernels_prepare(
	const struct backend_operations *backend, struct device_card *card,
	struct ordered_collective_kernels *kernels_out)
{
	static const char *const names[ORDERED_COLLECTIVE_OPERATIONS] = {
		KERNEL_ORDERED_ALL_REDUCE, KERNEL_ORDERED_TWO_SHOT,
		KERNEL_ORDERED_REDUCE_SCATTER, KERNEL_ORDERED_GATHER
	};
	if (backend == NULL || backend->kernel_prepare == NULL ||
	    card == NULL || kernels_out == NULL) {
		return STATUS_INVALID_ARGUMENT;
	}
	struct ordered_collective_kernels prepared = { 0 };
	for (uint32_t operation = 0; operation < ORDERED_COLLECTIVE_OPERATIONS;
	     operation++) {
		const result_status status = backend->kernel_prepare(
			card, KERNEL_FILE_ORDERED_COLLECTIVE, names[operation],
			ORDERED_COLLECTIVE_ARGUMENT_BYTES,
			&prepared.handles[operation]);
		if (status != STATUS_OK) {
			return status;
		}
	}
	*kernels_out = prepared;
	return STATUS_OK;
}

result_status
ordered_collective_host(const float *const *inputs, float *const *outputs,
			uint32_t rank_count, uint32_t element_count,
			enum ordered_collective_operation operation)
{
	if (inputs == NULL || outputs == NULL || rank_count == 0 ||
	    rank_count > ORDERED_COLLECTIVE_MAXIMUM_RANKS ||
	    element_count == 0 ||
	    element_count > ORDERED_COLLECTIVE_MAXIMUM_ELEMENTS ||
	    operation < ORDERED_ALL_REDUCE ||
	    operation >= ORDERED_COLLECTIVE_OPERATIONS ||
	    fegetround() != FE_TONEAREST) {
		return STATUS_INVALID_ARGUMENT;
	}
	const uint64_t bytes = (uint64_t)element_count * VALUE_BYTES;
	for (uint32_t rank = 0; rank < rank_count; rank++) {
		const uint64_t input_address =
			(uint64_t)(uintptr_t)inputs[rank];
		const uint64_t output_address =
			(uint64_t)(uintptr_t)outputs[rank];
		if (inputs[rank] == NULL || outputs[rank] == NULL ||
		    input_address % VALUE_BYTES != 0 ||
		    output_address % VALUE_BYTES != 0 ||
		    input_address > UINT64_MAX - bytes ||
		    output_address > UINT64_MAX - bytes) {
			return STATUS_INVALID_ARGUMENT;
		}
	}
	for (uint32_t rank = 0; rank < rank_count; rank++) {
		const uint64_t output_address =
			(uint64_t)(uintptr_t)outputs[rank];
		for (uint32_t other = 0; other < rank_count; other++) {
			if (ranges_overlap(output_address, bytes,
					   (uint64_t)(uintptr_t)inputs[other],
					   bytes) ||
			    (other != rank &&
			     ranges_overlap(output_address, bytes,
					    (uint64_t)(uintptr_t)outputs[other],
					    bytes))) {
				return STATUS_INVALID_ARGUMENT;
			}
		}
	}
	const uint32_t slice = (element_count + rank_count - 1u) / rank_count;
	for (uint32_t element = 0; element < element_count; element++) {
		float sum = 0.0f;
		if (operation == ORDERED_GATHER) {
			sum = inputs[element / slice][element];
		} else {
			for (uint32_t rank = 0; rank < rank_count; rank++) {
				sum = sum + inputs[rank][element];
			}
		}
		if (operation == ORDERED_REDUCE_SCATTER) {
			outputs[element / slice][element] = sum;
		} else {
			for (uint32_t rank = 0; rank < rank_count; rank++) {
				outputs[rank][element] = sum;
			}
		}
	}
	return STATUS_OK;
}

result_status ordered_collective_prepare(
	const struct ordered_collective_description *description,
	const struct ordered_collective_kernels *kernels,
	struct kernel_launch *launch_out)
{
	if (description == NULL || kernels == NULL || launch_out == NULL ||
	    description->operation < ORDERED_ALL_REDUCE ||
	    description->operation >= ORDERED_COLLECTIVE_OPERATIONS) {
		return STATUS_INVALID_ARGUMENT;
	}
	const struct ordered_collective_arguments *const arguments =
		&description->arguments;
	const uint64_t capacity = description->arena_byte_count;
	const uint64_t bytes = (uint64_t)arguments->element_count * VALUE_BYTES;
	const uint64_t table_bytes =
		(uint64_t)arguments->rank_count * ORDERED_COLLECTIVE_PEER_BYTES;
	if (arguments->element_count == 0 ||
	    arguments->element_count > ORDERED_COLLECTIVE_MAXIMUM_ELEMENTS ||
	    arguments->rank_count == 0 ||
	    arguments->rank_count > ORDERED_COLLECTIVE_MAXIMUM_RANKS ||
	    arguments->rank >= arguments->rank_count ||
	    arguments->reserved_zero != 0 || arguments->deadline_cycles == 0 ||
	    arguments->region_offset % ORDERED_COLLECTIVE_LINE_BYTES != 0 ||
	    arguments->parity_stride % ORDERED_COLLECTIVE_LINE_BYTES != 0 ||
	    arguments->parity_stride > UINT64_MAX / ORDERED_COLLECTIVE_SLOTS ||
	    !range_fits(arguments->region_offset,
			arguments->parity_stride * ORDERED_COLLECTIVE_SLOTS,
			capacity) ||
	    arguments->peer_table_offset % ADDRESS_BYTES != 0 ||
	    arguments->step_offset % ADDRESS_BYTES != 0 ||
	    !range_fits(arguments->peer_table_offset, table_bytes, capacity) ||
	    !range_fits(arguments->step_offset, STEP_BLOCK_BYTES_VALUE,
			capacity)) {
		return STATUS_INVALID_ARGUMENT;
	}
	const uint64_t offsets[] = { arguments->input_offset,
				     arguments->output_offset,
				     arguments->scratch_offset };
	const uint32_t arrays =
		(uint32_t)(sizeof(offsets) / sizeof(offsets[0]));
	for (uint32_t array = 0; array < arrays; array++) {
		if (offsets[array] < ORDERED_COLLECTIVE_SIGNAL_BYTES ||
		    offsets[array] % VALUE_BYTES != 0 ||
		    !range_fits(offsets[array], bytes,
				arguments->parity_stride)) {
			return STATUS_INVALID_ARGUMENT;
		}
		for (uint32_t previous = 0; previous < array; previous++) {
			if (ranges_overlap(offsets[array], bytes,
					   offsets[previous], bytes)) {
				return STATUS_INVALID_ARGUMENT;
			}
		}
	}
	const uint64_t region_bytes =
		arguments->parity_stride * ORDERED_COLLECTIVE_SLOTS;
	if (ranges_overlap(arguments->peer_table_offset, table_bytes,
			   arguments->region_offset, region_bytes) ||
	    ranges_overlap(arguments->step_offset, STEP_BLOCK_BYTES_VALUE,
			   arguments->region_offset, region_bytes) ||
	    ranges_overlap(arguments->step_offset, STEP_BLOCK_BYTES_VALUE,
			   arguments->peer_table_offset, table_bytes)) {
		return STATUS_INVALID_ARGUMENT;
	}
	struct kernel_launch prepared = { 0 };
	prepared.kernel = kernels->handles[description->operation];
	prepared.grid[0] = prepared.grid[1] =
		prepared.grid[LAUNCH_DIMENSIONS - 1u] = 1;
	prepared.block[0] = ORDERED_COLLECTIVE_THREADS;
	prepared.block[1] = prepared.block[LAUNCH_DIMENSIONS - 1u] = 1;
	prepared.argument_byte_count = ORDERED_COLLECTIVE_ARGUMENT_BYTES;
	memcpy(prepared.argument, arguments, sizeof(*arguments));
	*launch_out = prepared;
	return STATUS_OK;
}
