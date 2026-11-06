/* SPDX-License-Identifier: AGPL-3.0-or-later */
#include "front_end_pack.h"

#include <stddef.h>
#include <string.h>

#define FRONT_END_TENSOR_NAME "front_end.tables"

result_status front_end_pack_match(const struct pack_record *pack,
				   const struct load_layout *layout,
				   const struct load_result *loaded,
				   const void *external_tables,
				   uint64_t external_byte_count)
{
	if (pack == NULL || layout == NULL || loaded == NULL ||
	    external_tables == NULL || external_byte_count == 0 ||
	    external_byte_count > SIZE_MAX ||
	    layout->byte_offset_of_extent == NULL ||
	    layout->card_of_extent == NULL ||
	    layout->host_byte_count > SIZE_MAX ||
	    layout->extent_count != pack_header_of(pack)->extent_count) {
		return STATUS_INVALID_ARGUMENT;
	}
	const uint32_t found = pack_find_tensor(
		pack, FRONT_END_TENSOR_NAME, sizeof(FRONT_END_TENSOR_NAME) - 1);
	if (found == PACK_NOT_FOUND) {
		return STATUS_UNSUPPORTED;
	}
	const struct pack_tensor *const tensor = &pack_tensors_of(pack)[found];
	if (tensor->extent_index >= layout->extent_count ||
	    tensor->element_type != PACK_ELEMENT_UINT8 ||
	    tensor->dimension_count != 1 ||
	    tensor->dimensions[0] != tensor->byte_count ||
	    tensor->weight_format != WEIGHT_FORMAT_NONE ||
	    tensor->bits_per_weight != 0) {
		return STATUS_CORRUPT_PACK;
	}
	const struct pack_extent *const extent =
		&pack_extents_of(pack)[tensor->extent_index];
	const struct pack_unit_record_2 *const unit =
		&pack_units_of(pack)[extent->unit_index];
	const uint64_t host_offset =
		layout->byte_offset_of_extent[tensor->extent_index];
	if (unit->unit_kind != PLACEMENT_UNIT_HOST_ONLY ||
	    unit->layer_index != PACK_NO_LAYER || unit->extent_count != 1 ||
	    layout->card_of_extent[tensor->extent_index] !=
		    PLACEMENT_IN_HOST_MEMORY ||
	    loaded->host_arena == NULL ||
	    host_offset > layout->host_byte_count ||
	    extent->byte_count > layout->host_byte_count - host_offset ||
	    tensor->byte_offset_in_extent > extent->byte_count ||
	    tensor->byte_count >
		    extent->byte_count - tensor->byte_offset_in_extent) {
		return STATUS_CORRUPT_PACK;
	}
	if (tensor->byte_count != external_byte_count) {
		return STATUS_INVALID_ARGUMENT;
	}
	const unsigned char *const packed_tables =
		(const unsigned char *)loaded->host_arena +
		(size_t)host_offset + (size_t)tensor->byte_offset_in_extent;
	return memcmp(external_tables, packed_tables,
		      (size_t)external_byte_count) == 0 ?
		       STATUS_OK :
		       STATUS_INVALID_ARGUMENT;
}
