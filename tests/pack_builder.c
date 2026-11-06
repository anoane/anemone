/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * pack_builder.c: the pack writer and the walk of pack_builder.h. Test support: no part of the engine.
 *
 * Contract: pack_builder.h. What it leaves open is decided here:
 *   - In a name "unit<u>.extent<e>.tensor<t>", u is the unit's index in the plan, e the extent's index within its
 *     unit and t the tensor's index within its extent. Every number has the same width, so the names sort in the
 *     order the tensors are written and the records are sorted by construction; a plan with a number that needs
 *     more digits is STATUS_INVALID_ARGUMENT.
 *   - A tensor is one dimension of plain INT8 elements, so its dimension is its byte count: a plan whose tensors would
 *     hold more bytes than a dimension can say is STATUS_INVALID_ARGUMENT. So is a plan with more than
 *     MAXIMUM_PACK_BLOCKS blocks of extents (no sum below can then overflow) or with tables beyond
 *     PACK_MAXIMUM_TABLES_BYTES.
 *   - An extent is filled with 64-bit little-endian words, each drawn from the seed, the extent's index and the
 *     word's index through the mixer of pack_digest.h. The mixer is a bijection: no two words of an extent are
 *     equal, and two extents of one seed differ in every word.
 *   - A lack of memory is STATUS_OUT_OF_MEMORY. On failure the outputs are untouched.
 *   - test_pack_reseal ends the test program when it is given no pack or a pack whose header gives a
 *     tables_byte_count that is no whole number of digest words: a test must not go on with digests that were
 *     not made.
 *   - test_pack_walk stops at the first thing that is not so, and says it. A tensor is walked only when the
 *     extents were found in order: its extent is then inside the bytes.
 */
#include "pack_builder.h"

#include <inttypes.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "anemone/pack_digest.h"
#include "weight_format_reference.h"

#define NAME_NUMBER_COUNT ((size_t)3)           /* the numbers of a name: unit, extent, tensor */
#define NAME_BYTE_COUNT (sizeof("unit.extent.tensor") - 1 + NAME_NUMBER_COUNT * TEST_PACK_NAME_DIGITS)
#define MOST_DIGITS_OF_A_NUMBER 10              /* of a 32-bit number in decimal */
#define NAME_CAPACITY (sizeof("unit.extent.tensor") + NAME_NUMBER_COUNT * MOST_DIGITS_OF_A_NUMBER)
#define MAXIMUM_PACK_BLOCKS 1048576u            /* 4 GiB of extents: far above what a test builds */
#define BITS_PER_BYTE 8
#define MATRIX_DIMENSION_COUNT 2                /* of a matrix of a weight format: its outputs, then its inputs */

static _Noreturn void give_up(const char *what)
{
    (void)fprintf(stderr, "FAILED: test_pack_reseal: %s\n", what);
    abort();
}

static uint64_t content_word(uint64_t content_seed, uint32_t extent_index, uint64_t word_index)
{
	return pack_digest_term(pack_digest_mix(content_seed + extent_index),
				word_index);
}

unsigned char test_pack_content_byte(uint64_t content_seed, uint32_t extent_index, uint64_t byte_index)
{
	const uint64_t word = content_word(content_seed, extent_index,
					   byte_index / PACK_DIGEST_WORD_BYTES);
	return (unsigned char)(word >> (BITS_PER_BYTE *
					(byte_index % PACK_DIGEST_WORD_BYTES)));
}

/* The bytes of each tensor of an extent of this unit: an equal part of the extent in whole alignment units. */
static uint64_t tensor_byte_count_of(const struct test_pack_plan *plan, const struct test_pack_unit_plan *unit)
{
	const uint64_t extent_byte_count =
		(uint64_t)unit->blocks_per_extent * PACK_BLOCK_BYTES;
	if (plan->tensors_per_extent == 0) {
		return 0;
	}
	return extent_byte_count / plan->tensors_per_extent /
	       WEIGHT_FORMAT_MATRIX_ALIGNMENT_BYTES *
	       WEIGHT_FORMAT_MATRIX_ALIGNMENT_BYTES;
}

/* Checks the plan against the rules of pack.h and the limits of this file, and gives the header of its pack with
 * every field but the two digests. */
MUST_CHECK static result_status
header_of_plan(const struct test_pack_plan *plan,
	       struct pack_header *header_out)
{
	struct pack_header header;
	uint64_t block_count =
		0; /* of the extents: at most MAXIMUM_PACK_BLOCKS */
	uint64_t extent_count = 0; /* at most block_count */
	uint64_t tensor_count = 0;
	uint64_t table_byte_count = 0;
	size_t model_name_length = 0;
	bool stage_unit_was_seen = false;
	if (plan == NULL || plan->units == NULL || plan->model_name == NULL ||
	    plan->unit_count == 0) {
		return STATUS_INVALID_ARGUMENT;
	}
	model_name_length = strnlen(plan->model_name, MODEL_NAME_CAPACITY);
	if (model_name_length >= MODEL_NAME_CAPACITY) {
		return STATUS_INVALID_ARGUMENT;
	}
    for (uint32_t unit_index = 0; unit_index < plan->unit_count; unit_index++) {
        const struct test_pack_unit_plan *unit = &plan->units[unit_index];
        const uint64_t unit_block_count = (uint64_t)unit->extent_count * unit->blocks_per_extent;
        const uint64_t tensor_byte_count = tensor_byte_count_of(plan, unit);
	if (unit->unit_kind >= PLACEMENT_UNIT_KIND_COUNT ||
	    unit->extent_count == 0 || unit->blocks_per_extent == 0 ||
	    unit_block_count > MAXIMUM_PACK_BLOCKS - block_count ||
	    (unit->unit_kind == PLACEMENT_UNIT_ON_DEMAND &&
	     !stage_unit_was_seen) ||
	    (plan->tensors_per_extent != 0 &&
	     (tensor_byte_count == 0 || tensor_byte_count > UINT32_MAX))) {
		return STATUS_INVALID_ARGUMENT;
	}
	stage_unit_was_seen = stage_unit_was_seen ||
			      unit->unit_kind == PLACEMENT_UNIT_STAGE;
	block_count += unit_block_count;
        extent_count += unit->extent_count;
    }
    tensor_count = extent_count * plan->tensors_per_extent;
    if (tensor_count > PACK_MAXIMUM_TABLES_BYTES / PACK_TENSOR_RECORD_BYTES) {
	    return STATUS_INVALID_ARGUMENT;
    }
    table_byte_count =
	    extent_count * PACK_EXTENT_RECORD_BYTES +
	    (uint64_t)plan->unit_count * PACK_UNIT_RECORD_BYTES +
	    tensor_count * (PACK_TENSOR_RECORD_BYTES + NAME_BYTE_COUNT);
    if (table_byte_count > PACK_MAXIMUM_TABLES_BYTES) {
	    return STATUS_INVALID_ARGUMENT;
    }
    memset(&header, 0, sizeof(header));
    memcpy(header.magic, PACK_MAGIC, sizeof(PACK_MAGIC));
    memcpy(header.model_name, plan->model_name, model_name_length);
    header.format_version = PACK_FORMAT_VERSION;
    header.block_byte_count = PACK_BLOCK_BYTES;
    header.tables_byte_count = (table_byte_count + PACK_BLOCK_BYTES - 1) /
			       PACK_BLOCK_BYTES * PACK_BLOCK_BYTES;
    header.file_byte_count = PACK_BLOCK_BYTES + header.tables_byte_count +
			     block_count * PACK_BLOCK_BYTES;
    header.extent_count = (uint32_t)extent_count;
    header.unit_count = plan->unit_count;
    header.tensor_count = (uint32_t)tensor_count;
    header.name_byte_count = (uint32_t)(tensor_count * NAME_BYTE_COUNT);
    *header_out = header;
    return STATUS_OK;
}

/* Writes the tables and the extents of a plan that header_of_plan accepted behind the header in pack_bytes. */
MUST_CHECK static result_status
write_tables_and_extents(const struct test_pack_plan *plan,
			 unsigned char *pack_bytes)
{
    void *header_block = pack_bytes;
    const struct pack_header *header = header_block;
    void *tables = pack_bytes + PACK_BLOCK_BYTES;
    struct pack_extent *extents = tables;
    void *unit_table = extents + header->extent_count;
    struct pack_unit_record *unit_records = unit_table;
    void *tensor_table = unit_records + header->unit_count;
    struct pack_tensor *tensors = tensor_table;
    char *name_bytes = (char *)(tensors + header->tensor_count);
    uint64_t extent_byte_offset = PACK_BLOCK_BYTES + header->tables_byte_count;
    uint32_t extent_index = 0;
    uint32_t tensor_index = 0;
    for (uint32_t unit_index = 0; unit_index < plan->unit_count; unit_index++) {
        const struct test_pack_unit_plan *unit = &plan->units[unit_index];
	const uint64_t extent_byte_count =
		(uint64_t)unit->blocks_per_extent * PACK_BLOCK_BYTES;
	const uint64_t tensor_byte_count = tensor_byte_count_of(plan, unit);
	unit_records[unit_index] =
		(struct pack_unit_record){ .unit_kind = unit->unit_kind,
					   .layer_index = unit->layer_index };
	for (uint32_t extent_in_unit = 0; extent_in_unit < unit->extent_count; extent_in_unit++) {
            void *extent_bytes = pack_bytes + extent_byte_offset;
            uint64_t *words = extent_bytes;
	    result_status status = STATUS_OK;
	    for (uint64_t word_index = 0;
		 word_index < extent_byte_count / PACK_DIGEST_WORD_BYTES;
		 word_index++) {
		    words[word_index] = content_word(plan->content_seed,
						     extent_index, word_index);
	    }
	    extents[extent_index] =
		    (struct pack_extent){ .byte_offset = extent_byte_offset,
					  .byte_count = extent_byte_count,
					  .unit_index = unit_index };
	    status = pack_digest_of_host_bytes(extent_bytes, extent_byte_count,
					       &extents[extent_index].digest);
	    if (status != STATUS_OK) {
		    return status;
	    }
	    for (uint32_t tensor_in_extent = 0; tensor_in_extent < plan->tensors_per_extent; tensor_in_extent++) {
                char name[NAME_CAPACITY];
                const int name_length =
                    snprintf(name, sizeof(name), "unit%0*" PRIu32 ".extent%0*" PRIu32 ".tensor%0*" PRIu32,
                             TEST_PACK_NAME_DIGITS, unit_index, TEST_PACK_NAME_DIGITS, extent_in_unit,
                             TEST_PACK_NAME_DIGITS, tensor_in_extent);
                if (name_length < 0 || (size_t)name_length != NAME_BYTE_COUNT) {
			return STATUS_INVALID_ARGUMENT;
		}
                memcpy(name_bytes + (uint64_t)tensor_index * NAME_BYTE_COUNT, name, NAME_BYTE_COUNT);
		tensors[tensor_index] = (struct pack_tensor){
			.byte_offset_in_extent =
				tensor_in_extent * tensor_byte_count,
			.byte_count = tensor_byte_count,
			.extent_index = extent_index,
			.name_byte_offset =
				(uint32_t)(tensor_index * NAME_BYTE_COUNT),
			.name_byte_count = (uint32_t)NAME_BYTE_COUNT,
			.element_type = PACK_ELEMENT_INT8,
			.dimension_count = 1,
			.dimensions = { (uint32_t)tensor_byte_count },
			.weight_format = WEIGHT_FORMAT_NONE,
			.bits_per_weight = 0,
		};
		tensor_index++;
            }
            extent_byte_offset += extent_byte_count;
            extent_index++;
        }
    }
    return STATUS_OK;
}

result_status test_pack_build(const struct test_pack_plan *plan,
			      void **pack_bytes_out,
			      uint64_t *pack_byte_count_out)
{
	struct pack_header header;
	unsigned char *pack_bytes = NULL;
	result_status status = STATUS_INVALID_ARGUMENT;
	if (pack_bytes_out == NULL || pack_byte_count_out == NULL) {
		return STATUS_INVALID_ARGUMENT;
	}
    status = header_of_plan(plan, &header);
    if (status != STATUS_OK) {
	    return status;
    }
    pack_bytes =
	    aligned_alloc(PACK_BLOCK_BYTES, (size_t)header.file_byte_count);
    if (pack_bytes == NULL) {
	    return STATUS_OUT_OF_MEMORY;
    }
    memset(pack_bytes, 0,
	   (size_t)(PACK_BLOCK_BYTES + header.tables_byte_count));
    memcpy(pack_bytes, &header, sizeof(header));
    status = write_tables_and_extents(plan, pack_bytes);
    if (status != STATUS_OK) {
	    free(pack_bytes);
	    return status;
    }
    test_pack_reseal(pack_bytes);
    *pack_bytes_out = pack_bytes;
    *pack_byte_count_out = header.file_byte_count;
    return STATUS_OK;
}

void test_pack_reseal(void *pack_bytes)
{
	struct pack_header *header = pack_bytes;
	uint64_t tables_digest = 0;
	uint64_t header_digest = 0;
	if (pack_bytes == NULL) {
		give_up("no pack was given");
	}
	if (pack_digest_of_host_bytes(
		    (const unsigned char *)pack_bytes + PACK_BLOCK_BYTES,
		    header->tables_byte_count, &tables_digest) != STATUS_OK) {
		give_up("the header gives a tables_byte_count that is no whole number of digest words");
	}
    header->tables_digest = tables_digest;
    header->header_digest = 0;
    if (pack_digest_of_host_bytes(pack_bytes, PACK_BLOCK_BYTES,
				  &header_digest) != STATUS_OK) {
	    give_up("the digest of block 0 could not be taken");
    }
    header->header_digest = header_digest;
}

static const uint32_t no_reserved_words[PACK_TENSOR_RESERVED_WORDS] = { 0 };

/* What pack.h says a tensor of an open pack holds: reserved words of zero, and plain elements with no bits per
 * weight, or a matrix of two dimensions of UINT8 in exactly the bytes the reference counts for its format, its
 * shape and its bits per weight. The reference knows the formats a build carries and refuses every other value. */
static bool tensor_holds_what_it_says(const struct pack_tensor *tensor)
{
	const struct matrix_shape_record shape = {
		.output_size = tensor->dimensions[0],
		.input_size = tensor->dimensions[1],
		.bits_per_weight = tensor->bits_per_weight
	};
	struct test_weight_format_sizes sizes = { 0 };
	if (memcmp(tensor->reserved_zero, no_reserved_words,
		   sizeof(no_reserved_words)) != 0) {
		return false;
	}
	if (tensor->weight_format == WEIGHT_FORMAT_NONE) {
		return tensor->bits_per_weight == 0;
	}
	return tensor->dimension_count == MATRIX_DIMENSION_COUNT &&
	       tensor->element_type == PACK_ELEMENT_UINT8 &&
	       test_weight_format_sizes_of(
		       (enum weight_format_code)tensor->weight_format, &shape,
		       &sizes) == STATUS_OK &&
	       sizes.stored_byte_count == tensor->byte_count;
}

const char *test_pack_walk(const struct pack_record *pack,
			   const unsigned char *pack_bytes,
			   uint64_t pack_byte_count, uint64_t *byte_sum)
{
    if (pack == NULL || pack_bytes == NULL || byte_sum == NULL) {
        return "a walk has a pack, the bytes it was opened from and a sum";
    }
    const struct pack_header *header = pack_header_of(pack);
    const struct pack_extent *extents = pack_extents_of(pack);
    const struct pack_unit_record_2 *units = pack_units_of(pack);
    const struct pack_tensor *tensors = pack_tensors_of(pack);
    const char *name_bytes = pack_name_bytes_of(pack);
    uint64_t byte_offset = 0;
    uint32_t extent_index = 0;
    uint32_t stage_unit_index = 0;
    bool stage_unit_was_seen = false;
    bool is_sound = false;
    if (header == NULL || extents == NULL || units == NULL || tensors == NULL || name_bytes == NULL) {
        return "no accessor of an open pack gives NULL";
    }
    is_sound = header->file_byte_count == pack_byte_count && header->extent_count > 0 && header->unit_count > 0 &&
               header->unit_count <= header->extent_count &&
               memchr(header->model_name, '\0', sizeof(header->model_name)) != NULL;
    byte_offset = PACK_BLOCK_BYTES + header->tables_byte_count;
    for (uint32_t unit_index = 0; is_sound && unit_index < header->unit_count; unit_index++) {
	    const struct pack_unit_record_2 *unit = &units[unit_index];
	    uint64_t unit_byte_count = 0;
	    if (unit->unit_kind == PLACEMENT_UNIT_STAGE) {
		    stage_unit_index = unit_index;
		    stage_unit_was_seen = true;
	    }
	    is_sound =
		    unit->unit_kind < PLACEMENT_UNIT_KIND_COUNT &&
		    unit->reserved_zero == 0 &&
		    unit->first_extent_index == extent_index &&
		    unit->extent_count > 0 &&
		    unit->extent_count <= header->extent_count - extent_index &&
		    (unit->unit_kind == PLACEMENT_UNIT_ON_DEMAND ?
			     stage_unit_was_seen && unit->owner_unit_index ==
							    stage_unit_index :
			     unit->owner_unit_index == unit_index);
	    for (uint32_t extent_in_unit = 0;
		 is_sound && extent_in_unit < unit->extent_count;
		 extent_in_unit++) {
		    const struct pack_extent *extent = &extents[extent_index];
		    is_sound = extent->unit_index == unit_index &&
			       extent->byte_offset == byte_offset &&
			       extent->byte_count > 0 &&
			       extent->byte_count % PACK_BLOCK_BYTES == 0 &&
			       extent->byte_count <=
				       pack_byte_count - byte_offset &&
			       extent->reserved_zero == 0;
		    for (uint64_t byte_index = 0;
			 is_sound && byte_index < extent->byte_count;
			 byte_index++) {
			    *byte_sum += pack_bytes[byte_offset + byte_index];
		    }
		    unit_byte_count += extent->byte_count;
		    byte_offset += extent->byte_count;
		    extent_index++;
	    }
        is_sound = is_sound && unit->byte_count == unit_byte_count;
    }
    if (!is_sound || extent_index != header->extent_count || byte_offset != pack_byte_count) {
        return "the extents of an open pack fill the file in order, and its units are what that order says";
    }
    for (uint32_t tensor_index = 0; is_sound && tensor_index < header->tensor_count; tensor_index++) {
	    const struct pack_tensor *tensor = &tensors[tensor_index];
	    const struct pack_extent *extent =
		    &extents[tensor->extent_index % header->extent_count];
	    is_sound = tensor->extent_index < header->extent_count &&
		       tensor->name_byte_count > 0 &&
		       tensor->name_byte_offset <= header->name_byte_count &&
		       tensor->name_byte_count <=
			       header->name_byte_count -
				       tensor->name_byte_offset &&
		       tensor->dimension_count <= PACK_MAXIMUM_DIMENSIONS &&
		       tensor->element_type < PACK_ELEMENT_TYPE_COUNT &&
		       tensor_holds_what_it_says(tensor) &&
		       tensor->byte_offset_in_extent <= extent->byte_count &&
		       tensor->byte_count <=
			       extent->byte_count -
				       tensor->byte_offset_in_extent &&
		       tensor->byte_offset_in_extent %
				       WEIGHT_FORMAT_MATRIX_ALIGNMENT_BYTES ==
			       0 &&
		       pack_find_tensor(pack,
					name_bytes + tensor->name_byte_offset,
					tensor->name_byte_count) ==
			       tensor_index;
	    for (uint64_t byte_index = 0;
		 is_sound && byte_index < tensor->byte_count; byte_index++) {
		    *byte_sum += pack_bytes[extent->byte_offset +
					    tensor->byte_offset_in_extent +
					    byte_index];
	    }
    }
    return is_sound ? NULL
                    : "the tensors of an open pack lie inside their extents, hold what they say and are found by "
                      "their names";
}
