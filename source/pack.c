/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * pack.c: opens a pack and answers what it holds. Contract: pack.h.
 *
 * Block 0 and the tables come from the first copy that gives both with their digests holding and a header that
 * obeys its rules; a copy that does not (for a lack of memory too) is a warning, and the failure of the last copy
 * is the status. The extent and tensor records are served as they were read; only the units are derived. Open
 * bounds every index, offset and count a record holds, so whoever follows a record stays inside the pack, and
 * holds a matrix of a weight format against the bytes its format stores for its shape (weight_format_seam.h).
 */
#include "anemone/pack.h"

#include <inttypes.h>
#include <stdlib.h>
#include <string.h>

#include "anemone/hooks.h"
#include "anemone/pack_digest.h"

struct pack_record {
	struct pack_header *header; /* block 0 as it was read */
	struct pack_extent *
		extents; /* the tables as they were read: the extent records come first */
	const struct pack_tensor *tensors; /* inside the tables */
	const char *name_bytes; /* inside the tables */
	struct pack_unit_record_2 units[]; /* header->unit_count of them */
};

/* The tables follow one another in block-aligned memory. Each begins on a multiple of its record's alignment: an
 * extent record is a whole number of unit records, and a unit record is a multiple of a tensor record's alignment. */
_Static_assert(
	PACK_EXTENT_RECORD_BYTES % PACK_UNIT_RECORD_BYTES == 0 &&
		PACK_UNIT_RECORD_BYTES % _Alignof(struct pack_tensor) == 0,
	"a table of a pack would begin off the alignment of its records");

static const uint64_t element_byte_counts[PACK_ELEMENT_TYPE_COUNT] = {
	[PACK_ELEMENT_FLOAT16] = 2,	[PACK_ELEMENT_BFLOAT16] = 2,
	[PACK_ELEMENT_FLOAT32] = 4,	[PACK_ELEMENT_INT8] = 1,
	[PACK_ELEMENT_UINT8] = 1,	[PACK_ELEMENT_INT16] = 2,
	[PACK_ELEMENT_INT32] = 4,	[PACK_ELEMENT_INT64] = 8,
	[PACK_ELEMENT_FLOAT8_E4M3] = 1, [PACK_ELEMENT_FLOAT8_E8M0] = 1,
};
#define MATRIX_DIMENSION_COUNT 2                    /* of a matrix of a weight format: its outputs, then its inputs */

/* True when every byte is zero: the first is, and each is the same as the one after it. */
static bool bytes_are_zero(const void *bytes, uint64_t byte_count)
{
    const unsigned char *const first_byte = bytes;
    return byte_count == 0 || (first_byte[0] == 0 && memcmp(first_byte, first_byte + 1, (size_t)(byte_count - 1)) == 0);
}

/* The order of tensor names: as unsigned bytes, a shorter name before a longer one that it begins. */
static int name_order(const struct pack_record *pack,
		      const struct pack_tensor *tensor, const char *name,
		      uint32_t byte_count)
{
    const uint32_t shorter_byte_count = tensor->name_byte_count < byte_count ? tensor->name_byte_count : byte_count;
    const int order = memcmp(pack->name_bytes + tensor->name_byte_offset, name, shorter_byte_count);
    return order != 0 ? order : (tensor->name_byte_count > byte_count) - (tensor->name_byte_count < byte_count);
}

/* What is wrong with block 0 of a copy by the first rule it breaks, or NULL. Its digest: header_digest as zero. */
static const char *header_flaw(struct pack_header *header,
			       uint64_t reader_byte_count)
{
	/* four 32-bit counts, each times at most PACK_TENSOR_RECORD_BYTES: the sum is far below 2^64 */
	const uint64_t needed_byte_count =
		((uint64_t)header->extent_count * PACK_EXTENT_RECORD_BYTES +
		 (uint64_t)header->unit_count * PACK_UNIT_RECORD_BYTES +
		 (uint64_t)header->tensor_count * PACK_TENSOR_RECORD_BYTES +
		 header->name_byte_count + PACK_BLOCK_BYTES - 1) /
		PACK_BLOCK_BYTES * PACK_BLOCK_BYTES;
	const uint64_t stored_digest = header->header_digest;
	uint64_t digest = 0;
	header->header_digest = 0;
	const result_status digest_status =
		pack_digest_of_host_bytes(header, PACK_BLOCK_BYTES, &digest);
	header->header_digest = stored_digest;
	return memcmp(header->magic, PACK_MAGIC, sizeof(PACK_MAGIC)) != 0 ?
		       "the magic is not this format's" :
	       header->format_version != PACK_FORMAT_VERSION ?
		       "the format version is not this build's" :
	       header->block_byte_count != PACK_BLOCK_BYTES ?
		       "the block size is not this build's" :
	       digest_status != STATUS_OK || digest != stored_digest ?
		       "the header's digest does not hold" :
	       !bytes_are_zero(header + 1, PACK_BLOCK_BYTES - sizeof(*header)) ?
		       "the fill of block 0 is not zero" :
	       memchr(header->model_name, '\0', sizeof(header->model_name)) ==
			       NULL ?
		       "the model name holds no NUL" :
	       header->file_byte_count != reader_byte_count ?
		       "file_byte_count is not the reader's size" :
	       header->extent_count == 0 || header->unit_count == 0 ?
		       "there is no extent or no unit" :
	       header->unit_count > header->extent_count ?
		       "there are more units than extents" :
	       header->tables_byte_count > PACK_MAXIMUM_TABLES_BYTES ?
		       "tables_byte_count is above the limit" :
	       header->tables_byte_count != needed_byte_count ?
		       "tables_byte_count is not what the four counts need" :
	       needed_byte_count + PACK_BLOCK_BYTES >= reader_byte_count ?
		       "no extent follows the tables" :
		       NULL;
}

/* What is wrong with what a tensor of a known element type says it holds, by the first rule it breaks, or NULL.
 * plain_byte_count is the product of its dimensions times the size of an element, or UINT64_MAX beyond 64 bits. */
static const char *content_flaw(const struct pack_tensor *tensor,
				uint64_t plain_byte_count)
{
    /* any 32 bits may stand in the record: the table of a format is NULL for whatever this build does not carry */
    const struct weight_format_operations *const format =
	    weight_format_operations_of(
		    (enum weight_format_code)tensor->weight_format);
    const struct matrix_shape_record shape = {
	    .output_size = tensor->dimensions[0],
	    .input_size = tensor->dimensions[1],
	    .bits_per_weight = tensor->bits_per_weight
    };
    uint64_t stored_byte_count = 0;
    if (tensor->weight_format == WEIGHT_FORMAT_NONE) {
	    return plain_byte_count == UINT64_MAX ||
				   tensor->byte_count != plain_byte_count ?
			   "its byte count is its dimensions by an element" :
		   tensor->bits_per_weight != 0 ?
			   "as a plain tensor its bits_per_weight is zero" :
			   NULL;
    }
    return format == NULL ?
		   "its weight format is one this build carries" :
	   tensor->dimension_count != MATRIX_DIMENSION_COUNT ?
		   "as a matrix of a weight format it has two dimensions" :
	   tensor->element_type != PACK_ELEMENT_UINT8 ?
		   "as a matrix of a weight format its elements are UINT8" :
	   format->stored_byte_count(&shape, &stored_byte_count) != STATUS_OK ?
		   "its format stores its two dimensions and its bits_per_weight" :
	   tensor->byte_count != stored_byte_count ?
		   "its byte count is what its format stores for its shape" :
		   NULL;
}

/* Gives the pack its header and tables and derives its units; one pass over the extents, one over the tensors. */
MUST_CHECK static result_status check_tables(struct pack_record *pack,
					     struct pack_header *header,
					     struct pack_extent *extents)
{
	const struct pack_unit_record *unit_records =
		(const void *)(extents + header->extent_count);
	uint64_t byte_offset =
		PACK_BLOCK_BYTES +
		header->tables_byte_count; /* never beyond the file */
	uint32_t stage_unit_index =
		header->unit_count; /* the last stage unit; unit_count while there is none */
	uint32_t unit_count = 0; /* the units begun so far */
	pack->header = header;
	pack->extents = extents;
	pack->tensors = (const void *)(unit_records + header->unit_count);
	pack->name_bytes = (const void *)(pack->tensors + header->tensor_count);
	const char *const names_end =
		pack->name_bytes + header->name_byte_count;
	if (!bytes_are_zero(names_end, (uint64_t)((const char *)extents +
						  header->tables_byte_count -
						  names_end))) {
		LOG_ERROR(
			"corrupt pack: the fill of the tables, behind the name bytes, is not zero");
		return STATUS_CORRUPT_PACK;
	}
    for (uint32_t extent_index = 0; extent_index < header->extent_count; extent_index++) {
	    const struct pack_extent *extent = &extents[extent_index];
	    const uint64_t bytes_left = header->file_byte_count - byte_offset;
	    const bool is_last = extent_index == header->extent_count - 1;
	    if (extent->unit_index == unit_count &&
		unit_count < header->unit_count) {
		    const uint32_t unit_kind =
			    unit_records[unit_count].unit_kind;
		    stage_unit_index = unit_kind == PLACEMENT_UNIT_STAGE ?
					       unit_count :
					       stage_unit_index;
		    pack->units[unit_count] = (struct pack_unit_record_2){
			    .unit_kind = unit_kind,
			    .layer_index = unit_records[unit_count].layer_index,
			    .owner_unit_index =
				    unit_kind == PLACEMENT_UNIT_ON_DEMAND ?
					    stage_unit_index :
					    unit_count,
			    .first_extent_index = extent_index
		    };
		    unit_count++;
	    }
	    struct pack_unit_record_2 *unit =
		    &pack->units[unit_count > 0 ? unit_count - 1 : 0];
	    const char *rule =
		    unit_count == 0 || extent->unit_index != unit_count - 1 ||
				    (is_last &&
				     unit_count != header->unit_count) ?
			    "unit_index goes from 0 to unit_count - 1 by 0 or 1" :
		    unit->unit_kind >= PLACEMENT_UNIT_KIND_COUNT ?
			    "the kind of its unit is known" :
		    unit->owner_unit_index == header->unit_count ?
			    "its on-demand unit has a stage unit before it" :
		    extent->byte_count == 0 ||
				    extent->byte_count % PACK_BLOCK_BYTES != 0 ?
			    "it is one or more blocks" :
		    extent->byte_offset != byte_offset ?
			    "it begins where the one before ends, the first after the tables" :
		    extent->byte_count > bytes_left ||
				    (is_last &&
				     extent->byte_count != bytes_left) ?
			    "it ends inside the file, the last one at its end" :
		    extent->reserved_zero != 0 ? "its reserved field is zero" :
						 NULL;
	    if (rule != NULL) {
		    LOG_ERROR("corrupt pack: extent %" PRIu32
			      " breaks the rule: %s",
			      extent_index, rule);
		    return STATUS_CORRUPT_PACK;
	    }
        byte_offset += extent->byte_count;
        unit->extent_count++;
        unit->byte_count += extent->byte_count;
    }
    for (uint32_t tensor_index = 0; tensor_index < header->tensor_count; tensor_index++) {
	    const struct pack_tensor *tensor = &pack->tensors[tensor_index];
	    const uint64_t extent_byte_count =
		    tensor->extent_index < header->extent_count ?
			    extents[tensor->extent_index].byte_count :
			    0;
	    const bool type_is_known = tensor->element_type <
				       PACK_ELEMENT_TYPE_COUNT;
	    /* as plain elements; saturates at UINT64_MAX, which no tensor in an extent has: an extent is whole blocks */
	    uint64_t product =
		    type_is_known ? element_byte_counts[tensor->element_type] :
				    0;
	    uint64_t unused_dimensions = 0;
	    for (uint32_t dimension_index = 0;
		 dimension_index < PACK_MAXIMUM_DIMENSIONS; dimension_index++) {
		    const uint64_t factor =
			    dimension_index < tensor->dimension_count ?
				    tensor->dimensions[dimension_index] :
				    1;
		    unused_dimensions |=
			    dimension_index < tensor->dimension_count ?
				    0 :
				    tensor->dimensions[dimension_index];
		    product = factor != 0 && product > UINT64_MAX / factor ?
				      UINT64_MAX :
				      product * factor;
	    }
	const char *const content_rule = type_is_known ? content_flaw(tensor, product) : NULL;
	const char *rule =
		tensor->extent_index >= header->extent_count ?
			"its extent exists" :
		tensor->byte_offset_in_extent > extent_byte_count ?
			"it begins inside its extent" :
		tensor->byte_count > extent_byte_count -
					     tensor->byte_offset_in_extent ?
			"it ends inside its extent" :
		tensor->byte_offset_in_extent %
					WEIGHT_FORMAT_MATRIX_ALIGNMENT_BYTES !=
				0 ?
			"it begins aligned" :
		tensor->dimension_count > PACK_MAXIMUM_DIMENSIONS ?
			"it has at most the maximum of dimensions" :
		unused_dimensions != 0 ? "its unused dimensions are zero" :
		!type_is_known	       ? "its element type is known" :
		content_rule != NULL ?
				 content_rule :
		tensor->name_byte_count == 0 ?
				 "its name is not empty" :
		tensor->name_byte_offset > header->name_byte_count ?
				 "its name begins inside the name bytes" :
		tensor->name_byte_count > header->name_byte_count -
						  tensor->name_byte_offset ?
				 "its name ends inside them" :
		tensor_index > 0 && name_order(pack, tensor - 1,
					       pack->name_bytes +
						       tensor->name_byte_offset,
					       tensor->name_byte_count) >= 0 ?
				 "its name sorts after the name before it" :
		!bytes_are_zero(tensor->reserved_zero,
				sizeof(tensor->reserved_zero)) ?
				 "its reserved words are zero" :
				 NULL;
	if (rule != NULL) {
		LOG_ERROR("corrupt pack: tensor %" PRIu32
			  " breaks the rule: %s",
			  tensor_index, rule);
		return STATUS_CORRUPT_PACK;
	}
    }
    return STATUS_OK;
}

result_status pack_open_entry(const struct storage_operations_record *storage,
			      const struct storage_reader *reader,
			      struct pack_record **pack_out)
{
	struct storage_description description = { 0 };
	struct pack_header *header = NULL;
	struct pack_record *pack = NULL;
	void *tables = NULL;
	const char *failure = NULL;
	result_status status = STATUS_INVALID_ARGUMENT;
	if (storage == NULL || reader == NULL || pack_out == NULL ||
	    storage->reader_describe == NULL || storage->reader_read == NULL) {
		return STATUS_INVALID_ARGUMENT;
	}
	TIMER_BEGIN(pack_open);
	status = storage->reader_describe(reader, &description);
	if (status != STATUS_OK || description.copy_count == 0 ||
	    description.copy_count > STORAGE_MAXIMUM_COPIES) {
		return status != STATUS_OK ? status : STATUS_INVALID_ARGUMENT;
	}
	if (description.block_alignment_bytes == 0 ||
	    PACK_BLOCK_BYTES % description.block_alignment_bytes != 0) {
		LOG_ERROR(
			"corrupt pack: the reader's block alignment does not divide the block size");
		return STATUS_CORRUPT_PACK;
	}
	header = FAULT_POINT(pack_memory) ?
			 NULL :
			 aligned_alloc(PACK_BLOCK_BYTES, PACK_BLOCK_BYTES);
	if (header == NULL) {
		return STATUS_OUT_OF_MEMORY;
	}
    for (uint32_t copy_index = 0; tables == NULL && copy_index < description.copy_count; copy_index++) {
        uint64_t digest = 0;
	status = storage->reader_read(reader, copy_index, 0, header,
				      PACK_BLOCK_BYTES);
	failure = status == STATUS_OK ?
			  header_flaw(header, description.byte_count) :
			  "block 0 cannot be read";
	status = status == STATUS_OK && failure != NULL ? STATUS_CORRUPT_PACK :
							  status;
	if (status == STATUS_OK) {
		const uint64_t byte_count = header->tables_byte_count;
#if LOGGING_HOOKS_ENABLED
		failure =
			"the tables cannot be read, or their digest does not hold";
#endif
		tables = FAULT_POINT(pack_memory) ?
				 NULL :
				 aligned_alloc(PACK_BLOCK_BYTES, byte_count);
		status = tables == NULL ?
				 STATUS_OUT_OF_MEMORY :
				 storage->reader_read(reader, copy_index,
						      PACK_BLOCK_BYTES, tables,
						      byte_count);
		status = status == STATUS_OK ?
				 pack_digest_of_host_bytes(tables, byte_count,
							   &digest) :
				 status;
	}
	if (status != STATUS_OK || digest != header->tables_digest) {
		status = status != STATUS_OK ? status : STATUS_CORRUPT_PACK;
		LOG_WARNING_ENTRY("copy %" PRIu32 " not used (%s): %s",
				  copy_index, status_text(status), failure);
		free(tables);
		tables = NULL;
	}
    }
    if (status != STATUS_OK) {
	    LOG_ERROR("no copy of the pack is good (%s), the last: %s",
		      status_text(status), failure);
    } else {
	    pack = FAULT_POINT(pack_memory) ?
			   NULL :
			   malloc(sizeof(*pack) +
				  header->unit_count * sizeof(*pack->units));
	    status = pack != NULL ? check_tables(pack, header, tables) :
				    STATUS_OUT_OF_MEMORY;
    }
    if (status != STATUS_OK) {
	    free(pack);
	    free(tables);
	    free(header);
	    return status;
    }
    TIMER_END(pack_open);
    *pack_out = pack;
    return STATUS_OK;
}

void pack_close(struct pack_record *pack)
{
    if (pack != NULL) {
        free(pack->header);
        free(pack->extents);
        free(pack);
    }
}

const struct pack_header *pack_header_of(const struct pack_record *pack)
{
	return pack->header;
}
const struct pack_extent *pack_extents_of(const struct pack_record *pack)
{
	return pack->extents;
}
const struct pack_unit_record_2 *pack_units_of(const struct pack_record *pack)
{
	return pack->units;
}
const struct pack_tensor *pack_tensors_of(const struct pack_record *pack)
{
	return pack->tensors;
}
const char *pack_name_bytes_of(const struct pack_record *pack)
{
	return pack->name_bytes;
}

uint32_t pack_find_tensor(const struct pack_record *pack, const char *name,
			  uint32_t name_byte_count)
{
    uint32_t first_index = 0;
    uint32_t end_index = pack != NULL && name != NULL ? pack->header->tensor_count : 0;
    while (first_index < end_index) {
        const uint32_t middle_index = first_index + ((end_index - first_index) >> 1u);
        const int order = name_order(pack, &pack->tensors[middle_index], name, name_byte_count);
        if (order == 0) {
            return middle_index;
        }
        first_index = order < 0 ? middle_index + 1 : first_index;
        end_index = order < 0 ? end_index : middle_index;
    }
    return PACK_NOT_FOUND;
}

bool pack_placement_is_valid(const struct pack_record *pack,
			     const uint32_t *card_of_unit, uint32_t unit_count,
			     uint32_t card_count)
{
    if (pack == NULL || card_of_unit == NULL || unit_count != pack->header->unit_count) {
	    LOG_ERROR(
		    "placement: there must be a pack and one entry for each of its units");
	    return false;
    }
    for (uint32_t unit_index = 0; unit_index < unit_count; unit_index++) {
	    const struct pack_unit_record_2 *unit = &pack->units[unit_index];
	    const uint32_t card = card_of_unit[unit_index];
	    /* an owner is an earlier stage unit: the loop has found its entry to be a card */
	    if (unit->unit_kind == PLACEMENT_UNIT_STAGE ?
			card >= card_count :
			card != PLACEMENT_IN_HOST_MEMORY &&
				(unit->unit_kind == PLACEMENT_UNIT_HOST_ONLY ||
				 card != card_of_unit[unit->owner_unit_index])) {
		    LOG_ERROR("placement: unit %" PRIu32
			      " may not sit on %" PRIu32,
			      unit_index, card);
		    return false;
	    }
    }
    return true;
}
