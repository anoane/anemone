/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * pack_test: source/pack.c against pack.h.
 *
 * A valid pack opens and is what its plan says. Every rule of pack.h is broken once in the bytes of a valid pack,
 * which is then sealed again, so that the rule and not a digest refuses it; changes that break no rule are shown
 * to open, so that the sealing itself refuses nothing: among them a matrix of each weight format this build carries,
 * made of a plain tensor of the pack. Copies that rot or cannot be read are passed over, and the failure of the last
 * one is the status. Bytes changed at random are pack_fuzz_test's.
 */
#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "anemone/hooks.h"
#include "anemone/pack.h"
#include "anemone/pack_digest.h"
#include "anemone/status.h"
#include "memory_storage.h"
#include "pack_builder.h"

#define BLOCK \
	PACK_BLOCK_BYTES /* short names for the many places of this test that need them */
#define ALIGNMENT WEIGHT_FORMAT_MATRIX_ALIGNMENT_BYTES
#define CONTENT_SEED 0x0123456789ABCDEFull
#define COPY_PAIR 2
#define CARD_COUNT 2
#define READS_OF_AN_OPEN 2                      /* block 0 and the tables */
#define ALLOCATIONS_OF_AN_OPEN 3                /* block 0, the tables, the pack with its units */
#define NAME_CAPACITY 64                        /* more than a name of three 32-bit numbers */
#define HIGH_BYTE 0xFFu                         /* the last in the order of unsigned bytes, the first of signed ones */
#define DIMENSION_SPLIT 4u                      /* each further dimension of a tensor that is given several */
#define DIMENSION_WHOSE_FOURTH_POWER_WRAPS 65536u
#define SMALLER_BLOCK_ALIGNMENT 512u            /* divides the block size */
#define LAST_RECORD UINT32_MAX                  /* in place of a record index */
/* Three dimensions whose product is 2^64 - 2048: a sum of an offset and that many bytes wraps to a small number. */
#define WRAPPING_FIRST_DIMENSION 13027328u
#define WRAPPING_SECOND_DIMENSION 69431u
#define WRAPPING_THIRD_DIMENSION 20394401u
#define WRAPPING_DIMENSION_COUNT 3
#define COUNT_OF(table) (sizeof(table) / sizeof((table)[0]))

/* The pack whose rules are broken one by one: every kind of unit, owners of one and of two on-demand units,
 * extents of one, two and three blocks, three tensors to an extent. The figures follow from the plan and are held
 * against the pack the builder makes of it. */
enum rule_unit { UNIT_STAGE_A, UNIT_DEMAND_A, UNIT_DEMAND_B, UNIT_STAGE_B, UNIT_DEMAND_C, UNIT_HOST, UNIT_STAGE_C,
                 RULE_UNIT_COUNT };
enum rule_extent { EXTENT_STAGE_A_FIRST, EXTENT_STAGE_A_SECOND, EXTENT_DEMAND_A, EXTENT_DEMAND_B,
                   EXTENT_STAGE_B_FIRST, EXTENT_STAGE_B_SECOND, EXTENT_DEMAND_C, EXTENT_HOST, EXTENT_STAGE_C,
                   RULE_EXTENT_COUNT };
enum rule_tensor { TENSOR_FIRST, TENSOR_SECOND, TENSOR_THIRD };     /* those of the first extent */
#define RULE_TENSORS_PER_EXTENT 3
#define RULE_TENSOR_COUNT (RULE_EXTENT_COUNT * RULE_TENSORS_PER_EXTENT)
#define RULE_NAME_BYTES 33                      /* "unit00000.extent00000.tensor00000" */
#define RULE_NAME_BYTE_COUNT (RULE_TENSOR_COUNT * RULE_NAME_BYTES)
#define RULE_FIRST_EXTENT_BYTES 8192
#define RULE_FIRST_TENSOR_BYTES 2688            /* a third of the first extent in whole alignment units: 128 are left */
/* The smallest matrix of each weight format this build carries, sized by hand: [16, 64] at 4 bits is 512 bytes of
 * weights and 32 of scales, [16, 32] at 8 bits is 512 and 1. Half a tile of outputs is a shape no format stores. */
enum matrix_figure { MATRIX_DIMENSIONS = 2, MATRIX_OUTPUTS = 16, HALF_A_TILE = 8, E2M1_INPUTS = 64, E2M1_BITS = 4,
                     E2M1_BYTES = 544, E4M3_INPUTS = 32, E4M3_BITS = 8, E4M3_BYTES = 513 };
/* What the records and the names leave of the one block of the tables. */
#define RULE_FILL_BYTES                                 \
	(BLOCK - RULE_NAME_BYTE_COUNT -                 \
	 RULE_TENSOR_COUNT * PACK_TENSOR_RECORD_BYTES - \
	 RULE_EXTENT_COUNT * PACK_EXTENT_RECORD_BYTES - \
	 RULE_UNIT_COUNT * PACK_UNIT_RECORD_BYTES)

static const struct test_pack_unit_plan rule_units[RULE_UNIT_COUNT] = {
	[UNIT_STAGE_A] = { PLACEMENT_UNIT_STAGE, 0, 2, 2 },
	[UNIT_DEMAND_A] = { PLACEMENT_UNIT_ON_DEMAND, 0, 1, 1 },
	[UNIT_DEMAND_B] = { PLACEMENT_UNIT_ON_DEMAND, 0, 1, 1 },
	[UNIT_STAGE_B] = { PLACEMENT_UNIT_STAGE, 1, 2, 1 },
	[UNIT_DEMAND_C] = { PLACEMENT_UNIT_ON_DEMAND, 1, 1, 3 },
	[UNIT_HOST] = { PLACEMENT_UNIT_HOST_ONLY, PACK_NO_LAYER, 1, 2 },
	[UNIT_STAGE_C] = { PLACEMENT_UNIT_STAGE, PACK_NO_LAYER, 1, 2 },
};
static const struct test_pack_plan rule_plan = {
    .model_name = "pack-test-model", .units = rule_units, .unit_count = RULE_UNIT_COUNT,
    .tensors_per_extent = RULE_TENSORS_PER_EXTENT, .content_seed = CONTENT_SEED,
};

static const char *const copy_paths[STORAGE_MAXIMUM_COPIES] = {
	"copy 0", "copy 1", "copy 2", "copy 3",
	"copy 4", "copy 5", "copy 6", "copy 7",
};

/* The parts of a pack that the builder wrote, for a test to change. */
struct pack_view {
    unsigned char *bytes;
    struct pack_header *header;
    struct pack_extent *extents;
    struct pack_unit_record *unit_records;
    struct pack_tensor *tensors;
    char *name_bytes;
};

static int failures = 0;
static uint64_t walked_byte_sum = 0;            /* printed at the end, so that no read of a walk is optimised away */

static void expect(bool condition, const char *what)
{
    if (!condition) {
        (void)fprintf(stderr, "FAILED: %s\n", what);
        failures++;
    }
}

static void expect_status(result_status status, result_status expected_status,
			  const char *what)
{
    if (status != expected_status) {
	    (void)fprintf(stderr, "FAILED: %s: status \"%s\", not \"%s\"\n",
			  what, status_text(status),
			  status_text(expected_status));
	    failures++;
    }
}

/* How often open read each of two copies: once where it ended at block 0, READS_OF_AN_OPEN where it went on to the
 * tables, and not at all where the copy before it was good. */
static void expect_reads(uint64_t expected_first_count, uint64_t expected_second_count, const char *what)
{
    const uint64_t first_count = test_memory_storage_read_count(0);
    const uint64_t second_count = test_memory_storage_read_count(1);
    if (first_count != expected_first_count || second_count != expected_second_count) {
        (void)fprintf(stderr, "FAILED: %s: open read the copies %" PRIu64 " and %" PRIu64 " times, not %" PRIu64
                      " and %" PRIu64 "\n",
                      what, first_count, second_count, expected_first_count, expected_second_count);
        failures++;
    }
}

/* A pack of the plan, which the caller frees. The test cannot go on without it. */
static void build(const struct test_pack_plan *plan, void **pack_bytes_out, uint64_t *pack_byte_count_out)
{
	if (test_pack_build(plan, pack_bytes_out, pack_byte_count_out) !=
	    STATUS_OK) {
		expect(false, "the pack builder builds the plans of this test");
		exit(EXIT_FAILURE);
	}
}

/* The pack is block-aligned and its header holds the counts the builder wrote. */
static struct pack_view view_of(void *pack_bytes)
{
    struct pack_view view = {.bytes = pack_bytes, .header = pack_bytes};
    void *table = view.bytes + BLOCK;
    view.extents = table;
    table = view.extents + view.header->extent_count;
    view.unit_records = table;
    table = view.unit_records + view.header->unit_count;
    view.tensors = table;
    view.name_bytes = (char *)(view.tensors + view.header->tensor_count);
    return view;
}

/* Makes both digests hold again after the header or the tables were changed. test_pack_reseal needs the true
 * tables_byte_count: a changed one is put back for it and then sealed into the header alone. */
static void reseal(struct pack_header *header, uint64_t true_tables_byte_count)
{
    const uint64_t changed_tables_byte_count = header->tables_byte_count;
    uint64_t header_digest = 0;
    header->tables_byte_count = true_tables_byte_count;
    test_pack_reseal(header);
    if (changed_tables_byte_count != true_tables_byte_count) {
        header->tables_byte_count = changed_tables_byte_count;
        header->header_digest = 0;
	expect(pack_digest_of_host_bytes(header, BLOCK, &header_digest) ==
		       STATUS_OK,
	       "the digest of a header with another tables_byte_count is taken");
	header->header_digest = header_digest;
    }
}

/* The name pack_builder.h gives a tensor, and its length: the pack stores a name without its NUL. */
static uint32_t format_name(char *name, size_t name_capacity, uint32_t unit_index, uint32_t extent_in_unit,
                            uint32_t tensor_in_extent)
{
    const int length = snprintf(name, name_capacity, "unit%0*" PRIu32 ".extent%0*" PRIu32 ".tensor%0*" PRIu32,
                                TEST_PACK_NAME_DIGITS, unit_index, TEST_PACK_NAME_DIGITS, extent_in_unit,
                                TEST_PACK_NAME_DIGITS, tensor_in_extent);
    if (length <= 0 || (size_t)length >= name_capacity) {
        expect(false, "a tensor name of the test fits its buffer");
        return 0;
    }
    return (uint32_t)length;
}

static bool words_are_zero(const uint32_t *words, uint32_t word_count)
{
    for (uint32_t word_index = 0; word_index < word_count; word_index++) {
        if (words[word_index] != 0) {
            return false;
        }
    }
    return true;
}

/* Opens the pack bytes through a memory storage of copy_count copies. Faults that were set stay. The reader is
 * closed before this returns: a pack reads nothing after open. */
MUST_CHECK static result_status open_pack(const void *pack_bytes,
					  uint64_t pack_byte_count,
					  uint32_t copy_count,
					  struct pack_record **pack_out)
{
	struct storage_reader *reader = NULL;
	result_status status = STATUS_OK;
	test_memory_storage_set_pack(pack_bytes, pack_byte_count);
	status = test_memory_storage.reader_open(copy_paths, copy_count,
						 &reader);
	if (status != STATUS_OK) {
		expect(false, "the memory storage opens a reader");
		return status;
	}
	status = pack_open_entry(&test_memory_storage, reader, pack_out);
	test_memory_storage.reader_close(reader);
	return status;
}

/* Walks an open pack as pack_builder.h says: every index, offset and count it gives is followed, and whatever is not
 * as pack.h says of an open pack fails the test. */
static void walk_pack(const struct pack_record *pack,
		      const unsigned char *pack_bytes, uint64_t pack_byte_count)
{
    const char *const flaw = test_pack_walk(pack, pack_bytes, pack_byte_count, &walked_byte_sum);
    expect(flaw == NULL, flaw != NULL ? flaw : "an open pack is as pack.h says");
}

/* Opens the pack bytes and expects this status. A pack that opens is walked and closed; a refusal must leave the
 * output alone. */
static void expect_open(const void *pack_bytes, uint64_t pack_byte_count,
			uint32_t copy_count, result_status expected_status,
			const char *what)
{
    static uint64_t untouched_marker = 0;
    void *untouched_address = &untouched_marker;
    struct pack_record *const untouched_pack = untouched_address;
    struct pack_record *pack = untouched_pack;
    const result_status status =
	    open_pack(pack_bytes, pack_byte_count, copy_count, &pack);
    expect_status(status, expected_status, what);
    expect((status == STATUS_OK) != (pack == untouched_pack),
	   "only an open that succeeds writes pack_out");
    if (status == STATUS_OK && pack != untouched_pack && pack != NULL) {
	    walk_pack(pack, pack_bytes, pack_byte_count);
	    pack_close(pack);
    }
}

/* Everything the rule pack says, once open, against what its plan says: the header, each derived unit, each
 * extent with its digest, each tensor, its name and the way back from the name. */
static void expect_pack_is_the_rule_plan(const struct pack_record *pack,
					 const unsigned char *pack_bytes,
					 uint64_t pack_byte_count)
{
	const struct pack_header *header = pack_header_of(pack);
	const struct pack_extent *extents = pack_extents_of(pack);
	const struct pack_unit_record_2 *units = pack_units_of(pack);
	const struct pack_tensor *tensors = pack_tensors_of(pack);
	const char *name_bytes = pack_name_bytes_of(pack);
	char name[NAME_CAPACITY];
	uint64_t byte_offset = (uint64_t)BLOCK + BLOCK;
	uint32_t extent_index = 0;
	uint32_t tensor_index = 0;
	uint32_t stage_unit_index = 0;
	bool units_are_planned = true;
	bool extents_are_planned = true;
	bool tensors_are_planned = true;
	bool names_are_planned = true;
	expect(memcmp(header->magic, PACK_MAGIC, PACK_MAGIC_BYTES) == 0 &&
		       header->format_version == PACK_FORMAT_VERSION &&
		       header->block_byte_count == BLOCK &&
		       header->file_byte_count == pack_byte_count &&
		       strcmp(header->model_name, rule_plan.model_name) == 0,
	       "the header holds the magic, the version, the block size, the size of the file and the model's name");
	if (header->extent_count != RULE_EXTENT_COUNT ||
	    header->unit_count != RULE_UNIT_COUNT ||
	    header->tensor_count != RULE_TENSOR_COUNT ||
	    header->name_byte_count != RULE_NAME_BYTE_COUNT ||
	    header->tables_byte_count != BLOCK ||
	    extents[EXTENT_STAGE_A_FIRST].byte_count !=
		    RULE_FIRST_EXTENT_BYTES ||
	    tensors[TENSOR_FIRST].byte_count != RULE_FIRST_TENSOR_BYTES) {
		expect(false,
		       "the header counts what the plan holds, and the first extent and tensor have the planned sizes");
		return;
	}
    for (uint32_t unit_index = 0; unit_index < RULE_UNIT_COUNT; unit_index++) {
        const struct test_pack_unit_plan *unit_plan = &rule_units[unit_index];
	const struct pack_unit_record_2 *unit = &units[unit_index];
	const uint64_t extent_byte_count = (uint64_t)unit_plan->blocks_per_extent * BLOCK;
        const uint64_t tensor_byte_count = extent_byte_count / RULE_TENSORS_PER_EXTENT / ALIGNMENT * ALIGNMENT;
	stage_unit_index = unit_plan->unit_kind == PLACEMENT_UNIT_STAGE ?
				   unit_index :
				   stage_unit_index;
	units_are_planned =
		units_are_planned && unit->unit_kind == unit_plan->unit_kind &&
		unit->layer_index == unit_plan->layer_index &&
		unit->first_extent_index == extent_index &&
		unit->extent_count == unit_plan->extent_count &&
		unit->reserved_zero == 0 &&
		unit->byte_count ==
			unit_plan->extent_count * extent_byte_count &&
		unit->owner_unit_index ==
			(unit_plan->unit_kind == PLACEMENT_UNIT_ON_DEMAND ?
				 stage_unit_index :
				 unit_index);
	for (uint32_t extent_in_unit = 0; extent_in_unit < unit_plan->extent_count; extent_in_unit++) {
		const struct pack_extent *extent = &extents[extent_index];
		uint64_t digest = 0;
		extents_are_planned =
			extents_are_planned &&
			extent->byte_offset == byte_offset &&
			extent->byte_count == extent_byte_count &&
			extent->unit_index == unit_index &&
			extent->reserved_zero == 0 &&
			pack_digest_of_host_bytes(pack_bytes + byte_offset,
						  extent_byte_count,
						  &digest) == STATUS_OK &&
			extent->digest == digest;
		for (uint32_t tensor_in_extent = 0;
		     tensor_in_extent < RULE_TENSORS_PER_EXTENT;
		     tensor_in_extent++) {
			const struct pack_tensor *tensor =
				&tensors[tensor_index];
			const uint32_t length =
				format_name(name, sizeof(name), unit_index,
					    extent_in_unit, tensor_in_extent);
			tensors_are_planned =
				tensors_are_planned &&
				tensor->extent_index == extent_index &&
				tensor->byte_offset_in_extent ==
					tensor_in_extent * tensor_byte_count &&
				tensor->byte_count == tensor_byte_count &&
				tensor->element_type == PACK_ELEMENT_INT8 &&
				tensor->dimension_count == 1 &&
				tensor->dimensions[0] == tensor_byte_count &&
				words_are_zero(&tensor->dimensions[1],
					       PACK_MAXIMUM_DIMENSIONS - 1) &&
				tensor->weight_format == WEIGHT_FORMAT_NONE &&
				tensor->bits_per_weight == 0 &&
				words_are_zero(tensor->reserved_zero,
					       PACK_TENSOR_RESERVED_WORDS);
			names_are_planned =
				names_are_planned &&
				length == RULE_NAME_BYTES &&
				tensor->name_byte_count == RULE_NAME_BYTES &&
				tensor->name_byte_offset ==
					tensor_index * RULE_NAME_BYTES &&
				memcmp(name_bytes + tensor->name_byte_offset,
				       name, RULE_NAME_BYTES) == 0 &&
				pack_find_tensor(pack, name, RULE_NAME_BYTES) ==
					tensor_index;
			tensor_index++;
		}
            byte_offset += extent_byte_count;
            extent_index++;
        }
    }
    expect(units_are_planned, "every unit has the kind and layer of the plan, and the owner and extents of its place");
    expect(extents_are_planned, "every extent lies where the plan puts it and holds the digest of its bytes");
    expect(tensors_are_planned, "every tensor is an equal part of its extent, one dimension of plain INT8");
    expect(names_are_planned, "every tensor has the name of its place, and the name finds it");
}

static void expect_names_that_no_tensor_has(const struct pack_record *pack)
{
    char name[NAME_CAPACITY];
    const uint32_t name_byte_count = format_name(name, sizeof(name), 0, 0, 0);
    if (name_byte_count == 0) {
        return;
    }
    expect(pack_find_tensor(pack, name, name_byte_count) == 0,
	   "the first name finds the first tensor");
    expect(pack_find_tensor(pack, name, name_byte_count - 1) == PACK_NOT_FOUND,
	   "a name that a tensor's name begins with finds nothing");
    expect(pack_find_tensor(pack, name, 0) == PACK_NOT_FOUND,
	   "an empty name finds nothing");
    expect(pack_find_tensor(pack, NULL, name_byte_count) == PACK_NOT_FOUND &&
		   pack_find_tensor(NULL, name, name_byte_count) ==
			   PACK_NOT_FOUND,
	   "no name finds nothing, and no pack finds nothing");
    name[name_byte_count] = 'x';
    expect(pack_find_tensor(pack, name, name_byte_count + 1) == PACK_NOT_FOUND,
	   "a name that begins with a tensor's name and goes on finds nothing");
    name[name_byte_count - 1] = 'x';
    expect(pack_find_tensor(pack, name, name_byte_count) == PACK_NOT_FOUND,
	   "a name between two tensors' names finds nothing");
    name[0] = 'z';
    expect(pack_find_tensor(pack, name, name_byte_count) == PACK_NOT_FOUND,
	   "a name after every tensor's name finds nothing");
}

static void test_a_valid_pack(void)
{
    void *pack_bytes = NULL;
    uint64_t pack_byte_count = 0;
    struct pack_record *pack = NULL;
    build(&rule_plan, &pack_bytes, &pack_byte_count);
    test_memory_storage_reset();
    expect_status(open_pack(pack_bytes, pack_byte_count, 1, &pack), STATUS_OK,
		  "a valid pack opens");
    if (pack != NULL) {
	    expect(test_memory_storage_read_count(0) == READS_OF_AN_OPEN &&
			   test_memory_storage_delivered_byte_count(0) ==
				   BLOCK + pack_header_of(pack)
						   ->tables_byte_count,
		   "an open reads block 0 and the tables, each once, and no extent");
	    expect_pack_is_the_rule_plan(pack, pack_bytes, pack_byte_count);
	    expect_names_that_no_tensor_has(pack);
	    walk_pack(pack, pack_bytes, pack_byte_count);
    }
    pack_close(pack);
    pack_close(NULL);
    free(pack_bytes);
}

/* One change to the bytes of the rule pack: a number of the header, of a record or of the name bytes is set, or
 * has something added or taken away; what one number cannot say, a function does. */
enum pack_part { PART_HEADER, PART_EXTENT, PART_UNIT, PART_TENSOR, PART_NAME_BYTE };
enum change_operation { CHANGE_SET, CHANGE_ADD, CHANGE_SUBTRACT };

struct pack_change {
    const char *what;
    void (*function)(struct pack_view *view);   /* NULL for most */
    uint32_t part;                              /* enum pack_part */
    uint32_t record_index;                      /* or LAST_RECORD */
    size_t field_byte_offset;                   /* in the record */
    size_t field_byte_count;
    uint32_t operation;                         /* enum change_operation */
    uint64_t operand;
};

#define FIELD_OF(type, member) offsetof(type, member), sizeof(((type *)0)->member)
#define IN_HEADER(member) \
	NULL, PART_HEADER, 0, FIELD_OF(struct pack_header, member)
#define IN_EXTENT(record_index, member) \
	NULL, PART_EXTENT, (record_index), FIELD_OF(struct pack_extent, member)
#define IN_UNIT(record_index, member)    \
	NULL, PART_UNIT, (record_index), \
		FIELD_OF(struct pack_unit_record, member)
#define IN_TENSOR(record_index, member) \
	NULL, PART_TENSOR, (record_index), FIELD_OF(struct pack_tensor, member)
#define IN_NAME_BYTE(byte_index) NULL, PART_NAME_BYTE, (byte_index), 0, 1
#define IN_BLOCK_0(byte_index) NULL, PART_HEADER, 0, (byte_index), 1
#define IN_TABLES(byte_index) NULL, PART_EXTENT, 0, (byte_index), 1
#define BY_FUNCTION(function) (function), PART_HEADER, 0, 0, 0, CHANGE_ADD, 0
/* The first tensor is made a matrix of 4-bit weights, or that matrix in no bytes, and then one field is changed. */
#define IN_MATRIX(member)                            \
	make_e2m1_matrix, PART_TENSOR, TENSOR_FIRST, \
		FIELD_OF(struct pack_tensor, member)
#define IN_NO_BYTES(member)                           \
	make_empty_matrix, PART_TENSOR, TENSOR_FIRST, \
		FIELD_OF(struct pack_tensor, member)

static void change_pack(struct pack_view *view, const struct pack_change *change)
{
	const struct pack_header *header = view->header;
	const uint32_t part = change->part;
	unsigned char *field =
		part == PART_EXTENT    ? (unsigned char *)view->extents :
		part == PART_UNIT      ? (unsigned char *)view->unit_records :
		part == PART_TENSOR    ? (unsigned char *)view->tensors :
		part == PART_NAME_BYTE ? (unsigned char *)view->name_bytes :
					 view->bytes;
	const size_t record_byte_count =
		part == PART_EXTENT ? sizeof(struct pack_extent) :
		part == PART_UNIT   ? sizeof(struct pack_unit_record) :
		part == PART_TENSOR ? sizeof(struct pack_tensor) :
				      1;
	const uint32_t record_count =
		part == PART_EXTENT    ? header->extent_count :
		part == PART_UNIT      ? header->unit_count :
		part == PART_TENSOR    ? header->tensor_count :
		part == PART_NAME_BYTE ? header->name_byte_count :
					 1;
	uint64_t value = 0;
	if (change->function != NULL) {
		change->function(view);
	}
    /* a pack is little-endian and so is the host: a number of any width is the low bytes of a 64-bit one */
    field += (change->record_index == LAST_RECORD ? record_count - 1 : change->record_index) * record_byte_count +
             change->field_byte_offset;
    memcpy(&value, field, change->field_byte_count);
    value = change->operation == CHANGE_SET ? change->operand
            : change->operation == CHANGE_ADD ? value + change->operand
                                              : value - change->operand;
    memcpy(field, &value, change->field_byte_count);
}

static void break_model_name(struct pack_view *view)
{
    memset(view->header->model_name, 'm', sizeof(view->header->model_name));
}

/* The name bytes are counted so many that the tables, which the header then gives to match, reach the end of the
 * file: every header rule before this one still holds. */
static void break_room_for_an_extent(struct pack_view *view)
{
    const uint64_t extents_byte_count = view->header->file_byte_count - BLOCK - view->header->tables_byte_count;
    view->header->name_byte_count += (uint32_t)extents_byte_count;
    view->header->tables_byte_count += extents_byte_count;
}

/* Four dimensions whose product is 2^64: arithmetic that wraps would take it for the zero bytes the tensor says. */
static void break_dimensions_overflow(struct pack_view *view)
{
	struct pack_tensor *tensor = &view->tensors[TENSOR_FIRST];
	tensor->byte_count = 0;
	tensor->dimension_count = PACK_MAXIMUM_DIMENSIONS;
	for (uint32_t dimension_index = 0;
	     dimension_index < PACK_MAXIMUM_DIMENSIONS; dimension_index++) {
		tensor->dimensions[dimension_index] =
			DIMENSION_WHOSE_FOURTH_POWER_WRAPS;
	}
}

/* Two neighbours of one unit share their bytes on a line that is no block's: each still begins where the one
 * before ends. */
static void break_whole_blocks(struct pack_view *view)
{
	view->extents[EXTENT_STAGE_A_FIRST].byte_count +=
		PACK_DIGEST_WORD_BYTES;
	view->extents[EXTENT_STAGE_A_SECOND].byte_offset +=
		PACK_DIGEST_WORD_BYTES;
	view->extents[EXTENT_STAGE_A_SECOND].byte_count -=
		PACK_DIGEST_WORD_BYTES;
}

/* The tensors of an extent hold nothing: they lie inside it whatever becomes of its bytes. */
static void empty_the_tensors_of(struct pack_view *view, uint32_t extent_index)
{
	struct pack_tensor *tensors =
		view->tensors + (size_t)extent_index * RULE_TENSORS_PER_EXTENT;
	for (uint32_t tensor_in_extent = 0;
	     tensor_in_extent < RULE_TENSORS_PER_EXTENT; tensor_in_extent++) {
		tensors[tensor_in_extent].byte_offset_in_extent = 0;
		tensors[tensor_in_extent].byte_count = 0;
		tensors[tensor_in_extent].dimensions[0] = 0;
	}
}

/* An extent gives its bytes to the one before it and still begins where that one ends. */
static void break_an_extent_to_empty(struct pack_view *view)
{
	struct pack_extent *extent = &view->extents[EXTENT_DEMAND_A];
	view->extents[EXTENT_STAGE_A_SECOND].byte_count += extent->byte_count;
	extent->byte_offset += extent->byte_count;
	extent->byte_count = 0;
	empty_the_tensors_of(view, EXTENT_DEMAND_A);
}

/* The last extent ends a block before the file does; its tensors hold nothing, so no rule of theirs refuses it. */
static void break_end_of_the_last_extent(struct pack_view *view)
{
    view->extents[EXTENT_STAGE_C].byte_count -= BLOCK;
    empty_the_tensors_of(view, EXTENT_STAGE_C);
}

/* Every extent is a unit of its own: the last ones name units beyond those the header counts and open holds. */
static void break_units_beyond_the_count(struct pack_view *view)
{
    for (uint32_t extent_index = 0; extent_index < RULE_EXTENT_COUNT; extent_index++) {
        view->extents[extent_index].unit_index = extent_index;
    }
}

/* The second tensor of the first extent holds 2^64 - 2048 bytes, the product of its dimensions: a sum of its
 * offset and its bytes that wraps ends inside the extent. */
static void break_end_that_wraps(struct pack_view *view)
{
	struct pack_tensor *tensor = &view->tensors[TENSOR_SECOND];
	tensor->dimension_count = WRAPPING_DIMENSION_COUNT;
	tensor->dimensions[0] = WRAPPING_FIRST_DIMENSION;
	tensor->dimensions[1] = WRAPPING_SECOND_DIMENSION;
	tensor->dimensions[WRAPPING_DIMENSION_COUNT - 1] =
		WRAPPING_THIRD_DIMENSION;
	tensor->byte_count = (uint64_t)WRAPPING_FIRST_DIMENSION *
			     WRAPPING_SECOND_DIMENSION *
			     WRAPPING_THIRD_DIMENSION;
}

static void allow_a_dimension_of_zero(struct pack_view *view)
{
    view->tensors[TENSOR_FIRST].dimensions[0] = 0;
    view->tensors[TENSOR_FIRST].byte_count = 0;
}

/* A tensor without bytes lies inside any extent, also one that does not exist, and is as long as its dimensions say
 * whatever its elements are: only the rule it is named for is left to refuse it. */
static void break_extent_of_an_empty_tensor(struct pack_view *view)
{
    allow_a_dimension_of_zero(view);
    view->tensors[TENSOR_FIRST].extent_index = RULE_EXTENT_COUNT;
}

static void break_type_of_an_empty_tensor(struct pack_view *view)
{
    allow_a_dimension_of_zero(view);
    view->tensors[TENSOR_FIRST].element_type = PACK_ELEMENT_TYPE_COUNT;
}

static void allow_no_dimension(struct pack_view *view)
{
    view->tensors[TENSOR_FIRST].dimension_count = 0;
    view->tensors[TENSOR_FIRST].dimensions[0] = 0;
    view->tensors[TENSOR_FIRST].byte_count = 1;
}

static void allow_every_dimension(struct pack_view *view)
{
	struct pack_tensor *tensor = &view->tensors[TENSOR_FIRST];
	tensor->dimension_count = PACK_MAXIMUM_DIMENSIONS;
	for (uint32_t dimension_index = 1;
	     dimension_index < PACK_MAXIMUM_DIMENSIONS; dimension_index++) {
		tensor->dimensions[dimension_index] = DIMENSION_SPLIT;
		tensor->dimensions[0] /= DIMENSION_SPLIT;
	}
}

/* Every dimension is used and their product is the byte count: only the count of the dimensions is wrong. */
static void break_dimension_count(struct pack_view *view)
{
    allow_every_dimension(view);
    view->tensors[TENSOR_FIRST].dimension_count = PACK_MAXIMUM_DIMENSIONS + 1;
}

/* The first tensor becomes a matrix of 4-bit weights, the smallest its format stores, in the first bytes it had. */
static void make_e2m1_matrix(struct pack_view *view)
{
	struct pack_tensor *tensor = &view->tensors[TENSOR_FIRST];
	tensor->weight_format = WEIGHT_FORMAT_E2M1_SCALED;
	tensor->bits_per_weight = E2M1_BITS;
	tensor->element_type = PACK_ELEMENT_UINT8;
	tensor->dimension_count = MATRIX_DIMENSIONS;
	tensor->dimensions[0] = MATRIX_OUTPUTS;
	tensor->dimensions[1] = E2M1_INPUTS;
	tensor->byte_count = E2M1_BYTES;
}

/* The same of 8-bit weights: half a block of outputs, which has a scale of its own. */
static void make_e4m3_matrix(struct pack_view *view)
{
    make_e2m1_matrix(view);
    view->tensors[TENSOR_FIRST].weight_format = WEIGHT_FORMAT_E4M3_SCALED;
    view->tensors[TENSOR_FIRST].bits_per_weight = E4M3_BITS;
    view->tensors[TENSOR_FIRST].dimensions[1] = E4M3_INPUTS;
    view->tensors[TENSOR_FIRST].byte_count = E4M3_BYTES;
}

/* The 4-bit matrix in no bytes: no byte count then refuses what a missing or a refusing format is there to refuse. */
static void make_empty_matrix(struct pack_view *view)
{
    make_e2m1_matrix(view);
    view->tensors[TENSOR_FIRST].byte_count = 0;
}

/* A matrix of one dimension has no inputs left to say, and an unused dimension is zero: its format refuses the
 * shape too, so this is the one change of a matrix that breaks two rules. */
static void break_matrix_to_one_dimension(struct pack_view *view)
{
    make_e2m1_matrix(view);
    view->tensors[TENSOR_FIRST].dimension_count = 1;
    view->tensors[TENSOR_FIRST].dimensions[1] = 0;
}

/* Each breaks a rule of block 0. Open must end there, at block 0 of every copy and before it reads tables or obtains
 * memory for them and for the units: a rule of the tables that the same change breaks is no excuse for a header
 * rule that is gone. (Where two header rules refuse one change, as with tables above the limit, the test cannot
 * tell them apart: test_arguments_and_readers holds the limit alone.) */
static const struct pack_change broken_header_rules[] = {
	{ "the magic, in its first byte", IN_HEADER(magic[0]), CHANGE_ADD, 1 },
	{ "the magic, in its NUL", IN_HEADER(magic[PACK_MAGIC_BYTES - 1]),
	  CHANGE_SET, 'X' },
	{ "the format version", IN_HEADER(format_version), CHANGE_ADD, 1 },
	{ "the format version before this one, which is no longer read",
	  IN_HEADER(format_version), CHANGE_SUBTRACT, 1 },
	{ "the block size", IN_HEADER(block_byte_count), CHANGE_ADD, BLOCK },
	{ "block 0 is zero behind the header (the first byte)",
	  IN_BLOCK_0(PACK_HEADER_BYTES), CHANGE_SET, 1 },
	{ "block 0 is zero behind the header (the last byte)",
	  IN_BLOCK_0(BLOCK - 1), CHANGE_SET, HIGH_BYTE },
	{ "the model name holds a NUL", BY_FUNCTION(break_model_name) },
	{ "file_byte_count is the reader's size", IN_HEADER(file_byte_count),
	  CHANGE_SUBTRACT, BLOCK },
	{ "there is at least one extent", IN_HEADER(extent_count), CHANGE_SET,
	  0 },
	{ "there is at least one unit", IN_HEADER(unit_count), CHANGE_SET, 0 },
	{ "there are no more units than extents", IN_HEADER(unit_count),
	  CHANGE_SET, RULE_EXTENT_COUNT + 1 },
	{ "the tables fit the limit", IN_HEADER(tables_byte_count), CHANGE_SET,
	  PACK_MAXIMUM_TABLES_BYTES + BLOCK },
	{ "the tables are what the counts need (they need more)",
	  IN_HEADER(name_byte_count), CHANGE_ADD, BLOCK },
	{ "the tables are what the counts need (it gives more)",
	  IN_HEADER(tables_byte_count), CHANGE_ADD, BLOCK },
	{ "an extent follows the tables, inside the file",
	  BY_FUNCTION(break_room_for_an_extent) },
};

/* Each breaks exactly the rule of the tables it is named for: with that rule's check taken out of pack.c, its pack
 * would open (but for the matrix of one dimension, which says why). Open ends at the tables of the first copy, whose
 * digests hold. */
static const struct pack_change broken_table_rules[] = {
	{ "the tables are zero behind the name bytes (the first byte)",
	  IN_NAME_BYTE(RULE_NAME_BYTE_COUNT), CHANGE_SET, 1 },
	{ "the tables are zero behind the name bytes (the last byte)",
	  IN_TABLES(BLOCK - 1), CHANGE_SET, HIGH_BYTE },
	{ "an extent is whole blocks", BY_FUNCTION(break_whole_blocks) },
	{ "an extent is not empty", BY_FUNCTION(break_an_extent_to_empty) },
	{ "the first extent begins right after the tables",
	  IN_EXTENT(EXTENT_STAGE_A_FIRST, byte_offset), CHANGE_ADD, BLOCK },
	{ "an extent begins where the last ends (earlier)",
	  IN_EXTENT(EXTENT_DEMAND_B, byte_offset), CHANGE_SUBTRACT, BLOCK },
	{ "an extent begins where the last ends (later)",
	  IN_EXTENT(EXTENT_DEMAND_B, byte_offset), CHANGE_ADD, BLOCK },
	{ "the last extent ends with the file (past it)",
	  IN_EXTENT(LAST_RECORD, byte_count), CHANGE_ADD, BLOCK },
	{ "the last extent ends with the file (before it)",
	  BY_FUNCTION(break_end_of_the_last_extent) },
	{ "an extent's reserved field is zero",
	  IN_EXTENT(EXTENT_STAGE_A_SECOND, reserved_zero), CHANGE_SET, 1 },
	{ "unit_index is 0 in the first extent",
	  IN_EXTENT(EXTENT_STAGE_A_FIRST, unit_index), CHANGE_SET, 1 },
	{ "the first unit_index is 0 (not one less)",
	  IN_EXTENT(EXTENT_STAGE_A_FIRST, unit_index), CHANGE_SUBTRACT, 1 },
	{ "unit_index is the same or one more (two more)",
	  IN_EXTENT(EXTENT_DEMAND_A, unit_index), CHANGE_ADD, 1 },
	{ "unit_index is the same or one more (one less)",
	  IN_EXTENT(EXTENT_STAGE_B_SECOND, unit_index), CHANGE_SUBTRACT, 1 },
	{ "unit_index is unit_count - 1 in the last extent (less)",
	  IN_EXTENT(LAST_RECORD, unit_index), CHANGE_SUBTRACT, 1 },
	{ "unit_index is unit_count - 1 in the last extent (more)",
	  IN_EXTENT(LAST_RECORD, unit_index), CHANGE_ADD, 1 },
	{ "unit_index is below unit_count (every unit is begun, then one more)",
	  BY_FUNCTION(break_units_beyond_the_count) },
	{ "a unit's kind is known", IN_UNIT(UNIT_HOST, unit_kind), CHANGE_SET,
	  PLACEMENT_UNIT_KIND_COUNT },
	{ "an on-demand unit has an owner", IN_UNIT(UNIT_STAGE_A, unit_kind),
	  CHANGE_SET, PLACEMENT_UNIT_ON_DEMAND },
	{ "a tensor's extent exists",
	  BY_FUNCTION(break_extent_of_an_empty_tensor) },
	{ "a tensor begins inside its extent",
	  IN_TENSOR(TENSOR_SECOND, byte_offset_in_extent), CHANGE_SET,
	  RULE_FIRST_EXTENT_BYTES + ALIGNMENT },
	{ "a tensor ends inside it",
	  IN_TENSOR(TENSOR_SECOND, byte_offset_in_extent), CHANGE_SET,
	  RULE_FIRST_EXTENT_BYTES },
	{ "a tensor ends inside it (its end wraps)",
	  BY_FUNCTION(break_end_that_wraps) },
	{ "a tensor begins on the matrix alignment",
	  IN_TENSOR(TENSOR_FIRST, byte_offset_in_extent), CHANGE_ADD,
	  PACK_DIGEST_WORD_BYTES },
	{ "a tensor has at most the maximum of dimensions",
	  BY_FUNCTION(break_dimension_count) },
	{ "a tensor's unused dimensions are zero",
	  IN_TENSOR(TENSOR_FIRST, dimensions[1]), CHANGE_SET, 1 },
	{ "a tensor's type is known",
	  BY_FUNCTION(break_type_of_an_empty_tensor) },
	{ "a tensor's bytes are the product of its dimensions",
	  IN_TENSOR(TENSOR_FIRST, dimensions[0]), CHANGE_SUBTRACT, 1 },
	{ "a tensor's bytes are the product times an element",
	  IN_TENSOR(TENSOR_FIRST, element_type), CHANGE_SET,
	  PACK_ELEMENT_INT16 },
	{ "a tensor's bytes are the product with no overflow",
	  BY_FUNCTION(break_dimensions_overflow) },
	{ "a plain tensor has no bits per weight",
	  IN_TENSOR(TENSOR_FIRST, bits_per_weight), CHANGE_SET, E2M1_BITS },
	{ "a matrix's format is in the enumeration", IN_NO_BYTES(weight_format),
	  CHANGE_SET, WEIGHT_FORMAT_COUNT },
	{ "a matrix's format is one this build carries",
	  IN_NO_BYTES(weight_format), CHANGE_SET, WEIGHT_FORMAT_EXL3 },
	{ "a matrix has two dimensions (three)", IN_MATRIX(dimension_count),
	  CHANGE_ADD, 1 },
	{ "a matrix has two dimensions (one, and so no inputs)",
	  BY_FUNCTION(break_matrix_to_one_dimension) },
	{ "a matrix's elements are UINT8", IN_MATRIX(element_type), CHANGE_SET,
	  PACK_ELEMENT_FLOAT8_E4M3 },
	{ "a matrix's format stores its outputs (half a tile of them)",
	  IN_NO_BYTES(dimensions[0]), CHANGE_SET, HALF_A_TILE },
	{ "a matrix's format stores its bits per weight",
	  IN_NO_BYTES(bits_per_weight), CHANGE_SET, E4M3_BITS },
	{ "a matrix's bytes are its format's (an alignment unit more)",
	  IN_MATRIX(byte_count), CHANGE_ADD, ALIGNMENT },
	{ "a matrix's bytes are its format's (an alignment unit less)",
	  IN_MATRIX(byte_count), CHANGE_SUBTRACT, ALIGNMENT },
	{ "a matrix's reserved word is zero", IN_MATRIX(reserved_zero[0]),
	  CHANGE_SET, 1 },
	{ "a tensor's name is not empty",
	  IN_TENSOR(TENSOR_FIRST, name_byte_count), CHANGE_SET, 0 },
	{ "a name begins in the names",
	  IN_TENSOR(TENSOR_FIRST, name_byte_offset), CHANGE_SET,
	  RULE_NAME_BYTE_COUNT + 1 },
	{ "a name ends in the names", IN_TENSOR(LAST_RECORD, name_byte_count),
	  CHANGE_ADD, 1 },
	{ "names are sorted (the first has the third's)",
	  IN_TENSOR(TENSOR_FIRST, name_byte_offset), CHANGE_ADD,
	  RULE_NAME_BYTES + RULE_NAME_BYTES },
	{ "names are sorted (the same twice)",
	  IN_TENSOR(TENSOR_THIRD, name_byte_offset), CHANGE_SUBTRACT,
	  RULE_NAME_BYTES },
	{ "names are sorted (a shorter one after one that it begins)",
	  IN_TENSOR(TENSOR_SECOND, name_byte_count), CHANGE_SUBTRACT, 1 },
	{ "names are sorted (as unsigned bytes)",
	  IN_NAME_BYTE(RULE_NAME_BYTES - 1), CHANGE_SET, HIGH_BYTE },
	{ "a tensor's reserved words are zero (the first)",
	  IN_TENSOR(TENSOR_FIRST, reserved_zero[0]), CHANGE_SET, 1 },
	{ "a tensor's reserved words are zero (the last)",
	  IN_TENSOR(TENSOR_FIRST,
		    reserved_zero[PACK_TENSOR_RESERVED_WORDS - 1]),
	  CHANGE_SET, 1 },
};

/* Each breaks no rule: it shows that the changing and the sealing refuse nothing, and where a rule ends. */
static const struct pack_change allowed_changes[] = {
	{ "the layer of a unit is not open's to refuse",
	  IN_UNIT(UNIT_HOST, layer_index), CHANGE_SET, 1 },
	{ "the layer of an on-demand unit need not be its owner's",
	  IN_UNIT(UNIT_DEMAND_A, layer_index), CHANGE_SET, 1 },
	{ "the digest of an extent is not open's to check",
	  IN_EXTENT(EXTENT_STAGE_A_FIRST, digest), CHANGE_ADD, 1 },
	{ "a model name is any bytes with a NUL", IN_HEADER(model_name[0]),
	  CHANGE_SET, 'q' },
	{ "the last name may end in the highest byte",
	  IN_NAME_BYTE(RULE_NAME_BYTE_COUNT - 1), CHANGE_SET, HIGH_BYTE },
	{ "the names may end with their block: no fill is left",
	  IN_HEADER(name_byte_count), CHANGE_ADD, RULE_FILL_BYTES },
	{ "a shorter name comes before one that it begins",
	  IN_TENSOR(TENSOR_FIRST, name_byte_count), CHANGE_SUBTRACT, 1 },
	{ "a unit may be of another kind", IN_UNIT(UNIT_HOST, unit_kind),
	  CHANGE_SET, PLACEMENT_UNIT_ON_DEMAND },
	{ "an extent may belong to the unit before",
	  IN_EXTENT(EXTENT_STAGE_B_FIRST, unit_index), CHANGE_SUBTRACT, 1 },
	{ "a tensor may end where its extent ends",
	  IN_TENSOR(TENSOR_THIRD, byte_offset_in_extent), CHANGE_SET,
	  RULE_FIRST_EXTENT_BYTES - RULE_FIRST_TENSOR_BYTES },
	{ "a tensor with a dimension of zero holds no bytes",
	  BY_FUNCTION(allow_a_dimension_of_zero) },
	{ "a tensor with no dimension holds one element",
	  BY_FUNCTION(allow_no_dimension) },
	{ "a tensor may have every dimension",
	  BY_FUNCTION(allow_every_dimension) },
	{ "a tensor may be a matrix of 4-bit weights and their scales",
	  BY_FUNCTION(make_e2m1_matrix) },
	{ "a tensor may be a matrix of 8-bit weights and their scales",
	  BY_FUNCTION(make_e4m3_matrix) },
};

/* Builds the rule pack, makes one change, seals it, and opens it through two copies, each read this often. */
static void test_changes(const struct pack_change *changes, size_t change_count,
			 result_status expected_status,
			 uint64_t expected_first_read_count,
			 uint64_t expected_second_read_count)
{
    for (size_t change_index = 0; change_index < change_count; change_index++) {
        void *pack_bytes = NULL;
        uint64_t pack_byte_count = 0;
        uint64_t true_tables_byte_count = 0;
        struct pack_view view;
        build(&rule_plan, &pack_bytes, &pack_byte_count);
        view = view_of(pack_bytes);
        true_tables_byte_count = view.header->tables_byte_count;
        change_pack(&view, &changes[change_index]);
        reseal(view.header, true_tables_byte_count);
        test_memory_storage_reset();
        expect_open(pack_bytes, pack_byte_count, COPY_PAIR, expected_status, changes[change_index].what);
        expect_reads(expected_first_read_count, expected_second_read_count, changes[change_index].what);
        free(pack_bytes);
    }
}

/* Every element type with the size of its element: a tensor of that many bytes opens, one element more does not. */
static void test_element_types(void)
{
	static const uint32_t element_byte_counts[PACK_ELEMENT_TYPE_COUNT] = {
		[PACK_ELEMENT_FLOAT16] = 2,	[PACK_ELEMENT_BFLOAT16] = 2,
		[PACK_ELEMENT_FLOAT32] = 4,	[PACK_ELEMENT_INT8] = 1,
		[PACK_ELEMENT_UINT8] = 1,	[PACK_ELEMENT_INT16] = 2,
		[PACK_ELEMENT_INT32] = 4,	[PACK_ELEMENT_INT64] = 8,
		[PACK_ELEMENT_FLOAT8_E4M3] = 1, [PACK_ELEMENT_FLOAT8_E8M0] = 1,
	};
	for (uint32_t element_type = 0; element_type < PACK_ELEMENT_TYPE_COUNT;
	     element_type++) {
		void *pack_bytes = NULL;
		uint64_t pack_byte_count = 0;
		struct pack_view view;
		struct pack_tensor *tensor = NULL;
		build(&rule_plan, &pack_bytes, &pack_byte_count);
		view = view_of(pack_bytes);
		tensor = &view.tensors[TENSOR_FIRST];
		tensor->element_type = element_type;
		tensor->dimensions[0] =
			(uint32_t)(tensor->byte_count /
				   element_byte_counts[element_type]);
		reseal(view.header, view.header->tables_byte_count);
		test_memory_storage_reset();
		expect_open(
			pack_bytes, pack_byte_count, 1, STATUS_OK,
			"a tensor of each element type holds its elements times their size");
		tensor->dimensions[0]++;
		reseal(view.header, view.header->tables_byte_count);
		test_memory_storage_reset();
		expect_open(
			pack_bytes, pack_byte_count, 1, STATUS_CORRUPT_PACK,
			"a tensor of each element type with one element more than its bytes is refused");
		free(pack_bytes);
	}
}

/* What two copies hold and whether they can be read, what open then says, and how far it read each copy. A byte
 * that both copies deliver changed is a byte of the pack that changed without a new seal. */
struct copy_scenario {
    const char *what;
    uint64_t first_failing_read_number[COPY_PAIR];      /* 0: every read of the copy succeeds */
    uint64_t flipped_byte_offset[COPY_PAIR];            /* 0: the copy delivers the pack */
    result_status expected_status;
    uint64_t expected_read_count[COPY_PAIR];
};
#define HEADER_BYTE offsetof(struct pack_header, unit_count)
#define TABLES_BYTE (BLOCK + offsetof(struct pack_extent, byte_count))
#define LAST_BYTE_OF_BLOCK_0 (BLOCK - 1)

static const struct copy_scenario copy_scenarios[] = {
	{ "the header's digest (a changed byte)",
	  { 0, 0 },
	  { HEADER_BYTE, HEADER_BYTE },
	  STATUS_CORRUPT_PACK,
	  { 1, 1 } },
	{ "the header's digest (a changed byte of block 0 behind the header)",
	  { 0, 0 },
	  { LAST_BYTE_OF_BLOCK_0, LAST_BYTE_OF_BLOCK_0 },
	  STATUS_CORRUPT_PACK,
	  { 1, 1 } },
	{ "the tables' digest (a changed byte)",
	  { 0, 0 },
	  { TABLES_BYTE, TABLES_BYTE },
	  STATUS_CORRUPT_PACK,
	  { 2, 2 } },
	{ "a rotten header on the first copy: the next is read",
	  { 0, 0 },
	  { HEADER_BYTE, 0 },
	  STATUS_OK,
	  { 1, 2 } },
	{ "rotten tables on the first copy: the next is read",
	  { 0, 0 },
	  { TABLES_BYTE, 0 },
	  STATUS_OK,
	  { 2, 2 } },
	{ "a first copy that cannot be read is passed over",
	  { 1, 0 },
	  { 0, 0 },
	  STATUS_OK,
	  { 1, 2 } },
	{ "a first copy whose tables cannot be read is passed over",
	  { 2, 0 },
	  { 0, 0 },
	  STATUS_OK,
	  { 2, 2 } },
	{ "no copy can be read: the last copy's status",
	  { 1, 1 },
	  { 0, 0 },
	  STATUS_INPUT_OUTPUT_ERROR,
	  { 1, 1 } },
	{ "an unreadable copy, then a rotten one",
	  { 1, 0 },
	  { 0, TABLES_BYTE },
	  STATUS_CORRUPT_PACK,
	  { 1, 2 } },
	{ "a rotten copy, then an unreadable one",
	  { 0, 2 },
	  { HEADER_BYTE, 0 },
	  STATUS_INPUT_OUTPUT_ERROR,
	  { 1, 2 } },
	{ "a copy that is not needed is not read",
	  { 0, 1 },
	  { 0, HEADER_BYTE },
	  STATUS_OK,
	  { 2, 0 } },
};

/* The two digests themselves, and what open does when copies differ or fail. */
static void test_digests_and_copies(void)
{
    void *pack_bytes = NULL;
    uint64_t pack_byte_count = 0;
    build(&rule_plan, &pack_bytes, &pack_byte_count);
    for (size_t scenario_index = 0; scenario_index < COUNT_OF(copy_scenarios); scenario_index++) {
        const struct copy_scenario *scenario = &copy_scenarios[scenario_index];
        test_memory_storage_reset();
        for (uint32_t copy_index = 0; copy_index < COPY_PAIR; copy_index++) {
            test_memory_storage_fail_reads(copy_index, scenario->first_failing_read_number[copy_index]);
            if (scenario->flipped_byte_offset[copy_index] != 0) {
                test_memory_storage_flip_bit(copy_index, scenario->flipped_byte_offset[copy_index]);
            }
        }
        expect_open(pack_bytes, pack_byte_count, COPY_PAIR, scenario->expected_status, scenario->what);
        expect_reads(scenario->expected_read_count[0], scenario->expected_read_count[1], scenario->what);
    }
    test_memory_storage_reset();
    for (uint32_t copy_index = 0; copy_index < STORAGE_MAXIMUM_COPIES - 1;
	 copy_index++) {
	    test_memory_storage_fail_reads(copy_index, 1);
    }
    expect_open(pack_bytes, pack_byte_count, STORAGE_MAXIMUM_COPIES, STATUS_OK,
		"the last of the most copies is enough");
    free(pack_bytes);
}

/* What a storage says about its reader, as the test wants it said: this description, or this status. */
static struct storage_description changed_description;
static result_status changed_describe_status = STATUS_OK;

MUST_CHECK static result_status
describe_as_changed(const struct storage_reader *reader,
		    struct storage_description *description_out)
{
    (void)reader;
    if (changed_describe_status == STATUS_OK) {
	    *description_out = changed_description;
    }
    return changed_describe_status;
}

static void expect_open_gives(const struct storage_operations_record *storage,
			      const struct storage_reader *reader,
			      result_status expected_status, const char *what)
{
	struct pack_record *pack = NULL;
	const result_status status = pack_open_entry(storage, reader, &pack);
	expect_status(status, expected_status, what);
	expect((status == STATUS_OK) == (pack != NULL),
	       "only an open that succeeds writes pack_out");
	pack_close(pack);
}

/* Arguments and readers that open refuses before it reads. */
static void test_arguments_and_readers(void)
{
	struct storage_operations_record storage = test_memory_storage;
	struct storage_description true_description = { 0 };
	struct storage_reader *reader = NULL;
	struct pack_record *pack = NULL;
	struct pack_view view;
	void *pack_bytes = NULL;
	uint64_t pack_byte_count = 0;
	build(&rule_plan, &pack_bytes, &pack_byte_count);
	test_memory_storage_reset();
	test_memory_storage_set_pack(pack_bytes, pack_byte_count);
	if (test_memory_storage.reader_open(copy_paths, 1, &reader) !=
		    STATUS_OK ||
	    test_memory_storage.reader_describe(reader, &true_description) !=
		    STATUS_OK) {
		expect(false,
		       "the memory storage opens and describes a reader");
		test_memory_storage.reader_close(reader);
		free(pack_bytes);
		return;
	}
	expect(pack_open_entry(NULL, reader, &pack) ==
			       STATUS_INVALID_ARGUMENT &&
		       pack_open_entry(&storage, NULL, &pack) ==
			       STATUS_INVALID_ARGUMENT &&
		       pack_open_entry(&storage, reader, NULL) ==
			       STATUS_INVALID_ARGUMENT &&
		       pack == NULL,
	       "open refuses a missing storage, reader and output");
	storage.reader_read = NULL;
	expect_open_gives(&storage, reader, STATUS_INVALID_ARGUMENT,
			  "a storage that cannot read is refused");
	storage.reader_read = test_memory_storage.reader_read;
	storage.reader_describe = NULL;
	expect_open_gives(&storage, reader, STATUS_INVALID_ARGUMENT,
			  "a storage that cannot describe is refused");
	storage.reader_describe = describe_as_changed;
	changed_description = true_description;
	changed_description.block_alignment_bytes = SMALLER_BLOCK_ALIGNMENT;
	expect_open_gives(
		&storage, reader, STATUS_OK,
		"a block alignment that divides the block size opens");
	changed_description.block_alignment_bytes = BLOCK + BLOCK;
	expect_open_gives(
		&storage, reader, STATUS_CORRUPT_PACK,
		"a block alignment that does not divide the block size is refused");
	changed_description.block_alignment_bytes = 0;
	expect_open_gives(&storage, reader, STATUS_CORRUPT_PACK,
			  "a block alignment of zero is refused");
	changed_description = true_description;
	changed_description.byte_count += BLOCK;
	expect_open_gives(
		&storage, reader, STATUS_CORRUPT_PACK,
		"a reader of another size than file_byte_count is refused");
	changed_description = true_description;
	changed_description.copy_count = 0;
	expect_open_gives(&storage, reader, STATUS_INVALID_ARGUMENT,
			  "a reader without a copy is refused");
	changed_description.copy_count = STORAGE_MAXIMUM_COPIES + 1;
	expect_open_gives(
		&storage, reader, STATUS_INVALID_ARGUMENT,
		"a reader of more copies than a storage may have is refused");
	changed_describe_status = STATUS_DEVICE_ERROR;
	expect_open_gives(
		&storage, reader, STATUS_DEVICE_ERROR,
		"a reader that cannot describe itself keeps its own status");
	changed_describe_status = STATUS_OK;
	/* The counts need tables above the limit and the header gives them, in a file said to be large enough to hold
     * them: every other header rule holds. Without the limit open would obtain that memory and ask the reader for
     * bytes it does not have, which is another status. */
	view = view_of(pack_bytes);
	view.header->name_byte_count += (uint32_t)PACK_MAXIMUM_TABLES_BYTES;
	view.header->tables_byte_count += PACK_MAXIMUM_TABLES_BYTES;
	view.header->file_byte_count += PACK_MAXIMUM_TABLES_BYTES;
	reseal(view.header,
	       view.header->tables_byte_count - PACK_MAXIMUM_TABLES_BYTES);
	changed_description = true_description;
	changed_description.byte_count = view.header->file_byte_count;
	expect_open_gives(&storage, reader, STATUS_CORRUPT_PACK,
			  "tables above the limit are refused");
	test_memory_storage.reader_close(reader);
	free(pack_bytes);
}

struct placement_change {
    const char *what;
    uint32_t unit_index;
    uint32_t card;
};

static void test_placement(void)
{
	static const uint32_t valid_card_of_unit[RULE_UNIT_COUNT] = {
		[UNIT_STAGE_A] = 0,
		[UNIT_DEMAND_A] = 0,
		[UNIT_DEMAND_B] = PLACEMENT_IN_HOST_MEMORY,
		[UNIT_STAGE_B] = 1,
		[UNIT_DEMAND_C] = 1,
		[UNIT_HOST] = PLACEMENT_IN_HOST_MEMORY,
		[UNIT_STAGE_C] = 1,
	};
	static const struct placement_change refused_changes[] = {
		{ "a stage unit on a card that does not exist is refused",
		  UNIT_STAGE_A, CARD_COUNT },
		{ "a stage unit in host memory is refused", UNIT_STAGE_B,
		  PLACEMENT_IN_HOST_MEMORY },
		{ "an on-demand unit on a card that is not its owner's is refused",
		  UNIT_DEMAND_A, 1 },
		{ "an on-demand unit on the card of another owner is refused",
		  UNIT_DEMAND_C, 0 },
		{ "an on-demand unit on a card that does not exist is refused",
		  UNIT_DEMAND_B, CARD_COUNT },
		{ "a host-only unit on a card is refused", UNIT_HOST, 0 },
	};
	uint32_t card_of_unit[RULE_UNIT_COUNT];
	void *pack_bytes = NULL;
	uint64_t pack_byte_count = 0;
	struct pack_record *pack = NULL;
	build(&rule_plan, &pack_bytes, &pack_byte_count);
	test_memory_storage_reset();
	expect_status(open_pack(pack_bytes, pack_byte_count, 1, &pack),
		      STATUS_OK, "the pack to place opens");
	if (pack != NULL) {
		expect(pack_placement_is_valid(pack, valid_card_of_unit,
					       RULE_UNIT_COUNT, CARD_COUNT),
		       "a placement that obeys every kind is valid");
		for (size_t change_index = 0;
		     change_index < COUNT_OF(refused_changes); change_index++) {
			memcpy(card_of_unit, valid_card_of_unit,
			       sizeof(card_of_unit));
			card_of_unit[refused_changes[change_index].unit_index] =
				refused_changes[change_index].card;
			expect(!pack_placement_is_valid(pack, card_of_unit,
							RULE_UNIT_COUNT,
							CARD_COUNT),
			       refused_changes[change_index].what);
		}
		memcpy(card_of_unit, valid_card_of_unit, sizeof(card_of_unit));
		card_of_unit[UNIT_DEMAND_A] = PLACEMENT_IN_HOST_MEMORY;
		card_of_unit[UNIT_DEMAND_C] = PLACEMENT_IN_HOST_MEMORY;
		expect(pack_placement_is_valid(pack, card_of_unit,
					       RULE_UNIT_COUNT, CARD_COUNT),
		       "every on-demand unit may sit in host memory");
		expect(!pack_placement_is_valid(pack, valid_card_of_unit,
						RULE_UNIT_COUNT, 1),
		       "a placement is refused when there are fewer cards than it names");
		expect(!pack_placement_is_valid(pack, valid_card_of_unit,
						RULE_UNIT_COUNT - 1,
						CARD_COUNT) &&
			       !pack_placement_is_valid(
				       pack, valid_card_of_unit,
				       RULE_UNIT_COUNT + 1, CARD_COUNT),
		       "a placement of another number of units is refused");
		expect(!pack_placement_is_valid(pack, NULL, RULE_UNIT_COUNT,
						CARD_COUNT) &&
			       !pack_placement_is_valid(
				       NULL, valid_card_of_unit,
				       RULE_UNIT_COUNT, CARD_COUNT),
		       "no placement and no pack are refused");
	}
	pack_close(pack);
	free(pack_bytes);
}

#if CHECKING_HOOKS_ENABLED

/* Memory that is lacking at each allocation of an open. */
static void test_memory_failures(void)
{
    void *pack_bytes = NULL;
    uint64_t pack_byte_count = 0;
    build(&rule_plan, &pack_bytes, &pack_byte_count);
    for (uint64_t allocation_number = 1; allocation_number <= ALLOCATIONS_OF_AN_OPEN; allocation_number++) {
        test_memory_storage_reset();
	expect(fault_arm("pack_memory", allocation_number),
	       "a lack of memory is armed");
	expect_open(pack_bytes, pack_byte_count, 1, STATUS_OUT_OF_MEMORY,
		    "an open that lacks memory says so");
	expect(fault_disarm() == allocation_number,
	       "an open ends at the allocation that failed");
    }
    test_memory_storage_reset();
    expect(fault_arm("pack_memory", ALLOCATIONS_OF_AN_OPEN + 1),
	   "a lack of memory is armed");
    expect_open(pack_bytes, pack_byte_count, 1, STATUS_OK,
		"an open obtains memory three times and no more");
    expect(fault_disarm() == ALLOCATIONS_OF_AN_OPEN,
	   "an open that succeeds reached its three allocations");
    test_memory_storage_reset();
    expect(fault_arm("pack_memory", ALLOCATIONS_OF_AN_OPEN - 1),
	   "a lack of memory is armed");
    expect_open(
	    pack_bytes, pack_byte_count, COPY_PAIR, STATUS_OK,
	    "tables that memory was lacking for are read from the next copy");
    expect(fault_disarm() == ALLOCATIONS_OF_AN_OPEN + 1 &&
		   test_memory_storage_read_count(0) == 1 &&
		   test_memory_storage_read_count(1) == READS_OF_AN_OPEN,
	   "the copy that memory was lacking for is left after block 0");
    free(pack_bytes);
}

#endif

int main(void)
{
    test_a_valid_pack();
    test_arguments_and_readers();
    test_changes(broken_header_rules, COUNT_OF(broken_header_rules),
		 STATUS_CORRUPT_PACK, 1, 1);
    test_changes(broken_table_rules, COUNT_OF(broken_table_rules),
		 STATUS_CORRUPT_PACK, READS_OF_AN_OPEN, 0);
    test_changes(allowed_changes, COUNT_OF(allowed_changes), STATUS_OK,
		 READS_OF_AN_OPEN, 0);
    test_element_types();
    test_digests_and_copies();
    test_placement();
#if CHECKING_HOOKS_ENABLED
    test_memory_failures();
#endif
    if (failures != 0) {
        return EXIT_FAILURE;
    }
    (void)printf("pack_test passed (the bytes it walked add up to %" PRIu64 ")\n", walked_byte_sum);
    return EXIT_SUCCESS;
}
