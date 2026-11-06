/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Pure C11 preparation of a caller-qualified, immutable measured route table. */
#include "anemone/ordered_collective.h"
#include <string.h>

result_status ordered_collective_routes_prepare_measured(
	const struct ordered_collective_route_description *description,
	const struct ordered_collective_route_policy *rows, uint32_t row_count,
	uint32_t *host_reader_masks_out)
{
	if (description == NULL || host_reader_masks_out == NULL ||
	    row_count > ORDERED_ROUTE_POLICY_MAXIMUM_ROWS ||
	    (row_count != 0 && rows == NULL))
		return STATUS_INVALID_ARGUMENT;
	uint32_t host[ORDERED_COLLECTIVE_MAXIMUM_RANKS];
	struct ordered_collective_route_description baseline = *description;
	baseline.common_numa_root = false;
	result_status status =
		ordered_collective_routes_prepare(&baseline, host);
	if (status != STATUS_OK)
		return status;
	const struct ordered_collective_route_policy *chosen = NULL;
	for (uint32_t index = 0; index < row_count; index++) {
		const struct ordered_collective_route_policy *const row =
			&rows[index];
		if (row->rank_count == 0 ||
		    row->rank_count > ORDERED_COLLECTIVE_MAXIMUM_RANKS ||
		    row->element_count == 0 ||
		    row->element_count > ORDERED_COLLECTIVE_MAXIMUM_ELEMENTS ||
		    row->operation < ORDERED_ALL_REDUCE ||
		    row->operation >= ORDERED_COLLECTIVE_OPERATIONS ||
		    (row->paired_scatter_gather &&
		     row->operation != ORDERED_REDUCE_SCATTER &&
		     row->operation != ORDERED_GATHER))
			return STATUS_INVALID_ARGUMENT;
		const uint32_t all = (1u << row->rank_count) - 1u;
		for (uint32_t source = 0; source < row->rank_count; source++) {
			const uint32_t direct =
				row->expected_direct_reader_masks[source];
			const uint32_t mirrored =
				row->host_reader_masks[source];
			if ((direct & ~all) != 0 ||
			    (direct & (1u << source)) == 0 ||
			    (mirrored & ~all) != 0 ||
			    (mirrored & (1u << source)) != 0 ||
			    ((all & ~direct) & ~mirrored) != 0)
				return STATUS_INVALID_ARGUMENT;
		}
		for (uint32_t previous = 0; previous < index; previous++) {
			const struct ordered_collective_route_policy *const old =
				&rows[previous];
			if (old->rank_count == row->rank_count &&
			    old->element_count == row->element_count &&
			    old->operation == row->operation &&
			    old->paired_scatter_gather ==
				    row->paired_scatter_gather)
				return STATUS_INVALID_ARGUMENT;
			if (old->rank_count == row->rank_count &&
			    old->element_count == row->element_count &&
			    old->paired_scatter_gather &&
			    row->paired_scatter_gather) {
				for (uint32_t source = 0;
				     source < row->rank_count; source++)
					if (old->host_reader_masks[source] !=
						    row->host_reader_masks
							    [source] ||
					    old->expected_direct_reader_masks
							    [source] !=
						    row->expected_direct_reader_masks
							    [source] ||
					    old->arithmetic_classes[source] !=
						    row->arithmetic_classes
							    [source] ||
					    old->link_generations[source] !=
						    row->link_generations[source] ||
					    old->link_lane_counts[source] !=
						    row->link_lane_counts[source])
						return STATUS_INVALID_ARGUMENT;
			}
		}
		if (row->rank_count == description->rank_count &&
		    row->element_count == description->element_count &&
		    row->operation == description->operation &&
		    row->paired_scatter_gather ==
			    description->paired_scatter_gather)
			chosen = row;
	}
	if (chosen != NULL) {
		for (uint32_t source = 0; source < description->rank_count;
		     source++) {
			if (chosen->expected_direct_reader_masks[source] !=
				    description->direct_reader_masks[source] ||
			    chosen->arithmetic_classes[source] !=
				    (uint32_t)description->cards[source]
					    .arithmetic_class ||
			    chosen->link_generations[source] !=
				    description->cards[source].link_generation ||
			    chosen->link_lane_counts[source] !=
				    description->cards[source].link_lane_count)
				return STATUS_UNSUPPORTED;
			host[source] = chosen->host_reader_masks[source];
		}
	}
	memcpy(host_reader_masks_out, host,
	       (size_t)description->rank_count * sizeof(host[0]));
	return STATUS_OK;
}
