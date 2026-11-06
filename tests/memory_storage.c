/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * memory_storage.c: the storage of memory_storage.h. Test support: no part of the engine.
 *
 * Contract: memory_storage.h and storage_seam.h. What they leave open is decided here:
 *   - A read is refused before it is counted: the counters hold the reads that reached a copy, failed ones
 *     included, and no refused one.
 *   - fail_reads counts the reads of a copy from the last reset, as test_memory_storage_read_count does; a first
 *     failing read number of 0 takes the fault away.
 *   - A setter or counter that is given a copy this storage cannot have changes nothing, answers 0 and says so in
 *     one error line.
 *   - Everything a read looks at is atomic or belongs to its reader, which keeps the bytes, the size and the count
 *     of held bytes it was opened on: reads may run in any number of threads.
 *   - A pack given to set_pack is held whole. A read beyond the held bytes is counted, as a read of a drive that
 *     cannot deliver them, and the fault of fail_reads comes first.
 */
#include "memory_storage.h"

#include <inttypes.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "anemone/hooks.h"
#include "anemone/pack.h"

#define FLIPPED_BIT 1u

_Static_assert(PACK_BLOCK_BYTES % TEST_MEMORY_STORAGE_BLOCK_BYTES == 0,
	       "the tables of a pack end on a block of this storage");

struct storage_reader {
	const unsigned char *pack_bytes;
	uint64_t pack_byte_count; /* the size it reports */
	uint64_t held_byte_count; /* the first bytes it can deliver */
	uint32_t copy_count;
};

static _Atomic(const unsigned char *) current_pack_bytes = NULL;
static _Atomic uint64_t current_pack_byte_count = 0;
static _Atomic uint64_t current_held_byte_count = 0;
static unsigned char *held_tables = NULL;       /* what hold_tables_of read and owns; NULL for a pack of set_pack */
static _Atomic uint64_t
	first_failing_read_numbers[STORAGE_MAXIMUM_COPIES]; /* 0: no read fails */
static _Atomic bool bit_is_flipped[STORAGE_MAXIMUM_COPIES];
static _Atomic uint64_t flipped_byte_offsets[STORAGE_MAXIMUM_COPIES];
static _Atomic uint64_t read_counts[STORAGE_MAXIMUM_COPIES];
static _Atomic uint64_t delivered_byte_counts[STORAGE_MAXIMUM_COPIES];

MUST_CHECK static result_status
memory_reader_open(const char *const *copy_paths, uint32_t copy_count,
		   struct storage_reader **reader_out)
{
    const unsigned char *pack_bytes = atomic_load(&current_pack_bytes);
    const uint64_t pack_byte_count = atomic_load(&current_pack_byte_count);
    const uint64_t held_byte_count = atomic_load(&current_held_byte_count);
    struct storage_reader *reader = NULL;
    if (copy_paths == NULL || reader_out == NULL || copy_count == 0 ||
	copy_count > STORAGE_MAXIMUM_COPIES) {
	    return STATUS_INVALID_ARGUMENT;
    }
    for (uint32_t copy_index = 0; copy_index < copy_count; copy_index++) {
        if (copy_paths[copy_index] == NULL) {
		return STATUS_INVALID_ARGUMENT;
	}
    }
    if (pack_bytes == NULL || pack_byte_count == 0 || pack_byte_count % TEST_MEMORY_STORAGE_BLOCK_BYTES != 0) {
	    return STATUS_CORRUPT_PACK;
    }
    reader = FAULT_POINT(test_memory_storage_reader_memory) ?
		     NULL :
		     malloc(sizeof(*reader));
    if (reader == NULL) {
	    return STATUS_OUT_OF_MEMORY;
    }
    reader->pack_bytes = pack_bytes;
    reader->pack_byte_count = pack_byte_count;
    reader->held_byte_count = held_byte_count;
    reader->copy_count = copy_count;
    *reader_out = reader;
    return STATUS_OK;
}

static void memory_reader_close(struct storage_reader *reader)
{
    free(reader);
}

MUST_CHECK static result_status
memory_reader_describe(const struct storage_reader *reader,
		       struct storage_description *description_out)
{
    if (reader == NULL || description_out == NULL) {
	    return STATUS_INVALID_ARGUMENT;
    }
    *description_out = (struct storage_description){
	    .byte_count = reader->pack_byte_count,
	    .copy_count = reader->copy_count,
	    .block_alignment_bytes = TEST_MEMORY_STORAGE_BLOCK_BYTES,
    };
    return STATUS_OK;
}

MUST_CHECK static result_status
memory_reader_read(const struct storage_reader *reader, uint32_t copy_index,
		   uint64_t byte_offset, void *destination, uint64_t byte_count)
{
    unsigned char *delivered_bytes = destination;
    uint64_t first_failing_read_number = 0;
    uint64_t flipped_byte_offset = 0;
    uint64_t read_number = 0;
    if (reader == NULL || destination == NULL || copy_index >= reader->copy_count || byte_count == 0 ||
        byte_offset % TEST_MEMORY_STORAGE_BLOCK_BYTES != 0 || byte_count % TEST_MEMORY_STORAGE_BLOCK_BYTES != 0 ||
        (uintptr_t)destination % TEST_MEMORY_STORAGE_BLOCK_BYTES != 0 || byte_offset > reader->pack_byte_count ||
        byte_count > reader->pack_byte_count - byte_offset) {
	    return STATUS_INVALID_ARGUMENT;
    }
    read_number = atomic_fetch_add(&read_counts[copy_index], 1) + 1;
    first_failing_read_number = atomic_load(&first_failing_read_numbers[copy_index]);
    if (first_failing_read_number != 0 && read_number >= first_failing_read_number) {
	    return STATUS_INPUT_OUTPUT_ERROR;
    }
    if (byte_count > reader->held_byte_count || byte_offset > reader->held_byte_count - byte_count) {
	    LOG_ERROR("test_memory_storage holds the first %" PRIu64
		      " bytes of the pack, not the %" PRIu64 " at %" PRIu64,
		      reader->held_byte_count, byte_count, byte_offset);
	    return STATUS_INPUT_OUTPUT_ERROR;
    }
    memcpy(delivered_bytes, reader->pack_bytes + byte_offset, (size_t)byte_count);
    flipped_byte_offset = atomic_load(&flipped_byte_offsets[copy_index]);
    if (atomic_load(&bit_is_flipped[copy_index]) && flipped_byte_offset >= byte_offset &&
        flipped_byte_offset - byte_offset < byte_count) {
        unsigned char *flipped_byte = &delivered_bytes[flipped_byte_offset - byte_offset];
        *flipped_byte = (unsigned char)(*flipped_byte ^ FLIPPED_BIT);
    }
    atomic_fetch_add(&delivered_byte_counts[copy_index], byte_count);
    return STATUS_OK;
}

const struct storage_operations_record test_memory_storage = {
	.name = "test memory",
	.reader_open = memory_reader_open,
	.reader_close = memory_reader_close,
	.reader_describe = memory_reader_describe,
	.reader_read = memory_reader_read,
};

/* True for a copy this storage can have; a test that names another one is told. */
static bool copy_exists(uint32_t copy_index)
{
	if (copy_index >= STORAGE_MAXIMUM_COPIES) {
		LOG_ERROR("test_memory_storage has no copy %" PRIu32,
			  copy_index);
		return false;
	}
    return true;
}

void test_memory_storage_reset(void)
{
    free(held_tables);
    held_tables = NULL;
    atomic_store(&current_pack_bytes, NULL);
    atomic_store(&current_pack_byte_count, 0);
    atomic_store(&current_held_byte_count, 0);
    for (uint32_t copy_index = 0; copy_index < STORAGE_MAXIMUM_COPIES;
	 copy_index++) {
	    atomic_store(&first_failing_read_numbers[copy_index], 0);
	    atomic_store(&bit_is_flipped[copy_index], false);
	    atomic_store(&flipped_byte_offsets[copy_index], 0);
	    atomic_store(&read_counts[copy_index], 0);
	    atomic_store(&delivered_byte_counts[copy_index], 0);
    }
}

void test_memory_storage_set_pack(const void *pack_bytes, uint64_t pack_byte_count)
{
    free(held_tables);
    held_tables = NULL;
    atomic_store(&current_pack_bytes, pack_bytes);
    atomic_store(&current_pack_byte_count, pack_byte_count);
    atomic_store(&current_held_byte_count, pack_byte_count);
}

/* What is wrong with the size of a pack and the header of its block 0, for holding its tables, or NULL. */
static const char *tables_flaw(const struct storage_description *description,
			       const struct pack_header *header)
{
    const uint64_t tables_byte_count = header->tables_byte_count;
    return memcmp(header->magic, PACK_MAGIC, sizeof(PACK_MAGIC)) != 0 ?
		   "the magic is not a pack's" :
	   header->block_byte_count != PACK_BLOCK_BYTES ?
		   "the block size is not this build's" :
	   tables_byte_count == 0 || tables_byte_count % PACK_BLOCK_BYTES != 0 ?
		   "the tables are not one or more whole blocks" :
	   tables_byte_count > PACK_MAXIMUM_TABLES_BYTES ?
		   "the tables are above the limit" :
	   tables_byte_count > description->byte_count - PACK_BLOCK_BYTES ?
		   "the tables end beyond the pack" :
		   NULL;
}

result_status test_memory_storage_hold_tables_of(
	const struct storage_operations_record *storage,
	const struct storage_reader *reader)
{
	_Alignas(PACK_BLOCK_BYTES) unsigned char first_block[PACK_BLOCK_BYTES];
	struct storage_description description = { 0 };
	struct pack_header header = { 0 };
	unsigned char *held = NULL;
	const char *flaw = NULL;
	if (storage == NULL || reader == NULL ||
	    storage->reader_describe == NULL || storage->reader_read == NULL) {
		return STATUS_INVALID_ARGUMENT;
	}
	result_status status = storage->reader_describe(reader, &description);
	if (status != STATUS_OK) {
		return status;
	}
	if (description.block_alignment_bytes == 0 ||
	    PACK_BLOCK_BYTES % description.block_alignment_bytes != 0) {
		flaw = "the reader's block alignment does not divide the block size";
	} else if (description.byte_count <= PACK_BLOCK_BYTES ||
		   description.byte_count % PACK_BLOCK_BYTES != 0) {
		flaw = "the pack is not whole blocks, block 0 and more";
	} else {
		status = storage->reader_read(reader, 0, 0, first_block,
					      sizeof(first_block));
		memcpy(&header, first_block, sizeof(header));
		flaw = status == STATUS_OK ?
			       tables_flaw(&description, &header) :
			       NULL;
	}
	if (status != STATUS_OK || flaw != NULL) {
		if (flaw != NULL) {
			LOG_ERROR("test_memory_storage_hold_tables_of: %s",
				  flaw);
		}
		return status != STATUS_OK ? status : STATUS_CORRUPT_PACK;
	}
	const uint64_t held_byte_count =
		PACK_BLOCK_BYTES + header.tables_byte_count;
	held = FAULT_POINT(test_memory_storage_tables_memory) ?
		       NULL :
		       aligned_alloc(PACK_BLOCK_BYTES, (size_t)held_byte_count);
	if (held == NULL) {
		return STATUS_OUT_OF_MEMORY;
	}
    memcpy(held, first_block, sizeof(first_block));
    status = storage->reader_read(reader, 0, PACK_BLOCK_BYTES,
				  held + PACK_BLOCK_BYTES,
				  header.tables_byte_count);
    if (status != STATUS_OK) {
	    free(held);
	    return status;
    }
    free(held_tables);
    held_tables = held;
    atomic_store(&current_pack_bytes, held);
    atomic_store(&current_pack_byte_count, description.byte_count);
    atomic_store(&current_held_byte_count, held_byte_count);
    return STATUS_OK;
}

void test_memory_storage_fail_reads(uint32_t copy_index, uint64_t first_failing_read_number)
{
    if (copy_exists(copy_index)) {
        atomic_store(&first_failing_read_numbers[copy_index], first_failing_read_number);
    }
}

void test_memory_storage_flip_bit(uint32_t copy_index, uint64_t byte_offset)
{
    if (copy_exists(copy_index)) {
        atomic_store(&flipped_byte_offsets[copy_index], byte_offset);
        atomic_store(&bit_is_flipped[copy_index], true);
    }
}

uint64_t test_memory_storage_read_count(uint32_t copy_index)
{
    return copy_exists(copy_index) ? atomic_load(&read_counts[copy_index]) : 0;
}

uint64_t test_memory_storage_delivered_byte_count(uint32_t copy_index)
{
    return copy_exists(copy_index) ? atomic_load(&delivered_byte_counts[copy_index]) : 0;
}
