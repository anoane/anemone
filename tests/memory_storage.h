/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * memory_storage.h: a storage whose copies are one byte array in memory, with the faults of a drive switched on by
 * a test; or only the first bytes of a pack, its block 0 and tables, so that the tables of a pack of any size open
 * as pack_open_entry opens them, without its extents. Test support: no part of the engine.
 *
 * Contract:
 *   - test_memory_storage obeys storage_seam.h: every refusal, every status, reads from any number of threads.
 *     reader_open ignores what the paths say and uses only their count: every copy is the bytes given to
 *     test_memory_storage_set_pack, which the storage does not copy and which must outlive every reader, or the
 *     tables held by test_memory_storage_hold_tables_of. Its block alignment is TEST_MEMORY_STORAGE_BLOCK_BYTES.
 *     Opening before a pack was set, or a pack whose size is zero or no multiple of the block, fails as
 *     storage_seam.h says (STATUS_CORRUPT_PACK).
 *   - test_memory_storage_hold_tables_of reads, through copy 0 of a reader of any storage, block 0 of a pack and
 *     the tables_byte_count bytes of tables its header names, and holds them in memory of its own: from then on a
 *     reader opened here is a pack of the size the other reader states, of which the held bytes can be read. A
 *     read that reaches beyond them is counted and fails with STATUS_INPUT_OUTPUT_ERROR and one error
 *     line: no extent is ever delivered. It checks only what it needs to know how much to hold, else
 *     STATUS_CORRUPT_PACK with one error line: the magic and the block size are pack.h's, the size of the
 *     pack is a whole number of blocks, and tables_byte_count is whole blocks, at most
 *     PACK_MAXIMUM_TABLES_BYTES and inside the pack. A missing argument is STATUS_INVALID_ARGUMENT,
 *     and a status of the other storage is returned as it is. A failure leaves the storage as it was. The held
 *     bytes live until the next hold, set_pack or reset, which are called with no reader of them open.
 *   - Faults are set before the reads they are meant for and stay until test_memory_storage_reset, which also
 *     forgets the pack and zeroes the counters:
 *       fail_reads    every read of that copy fails with STATUS_INPUT_OUTPUT_ERROR from its
 *                     first_failing_read_number-th read on, counted from 1: a drive that was pulled
 *       flip_bit      that copy delivers the byte at byte_offset with its lowest bit flipped, in every read that
 *                     covers it: a block that rotted on one drive. One offset per copy; setting again replaces it
 *   - The setters and reset are called with no read in flight; the counters may be read at any time.
 */
#ifndef TEST_MEMORY_STORAGE_H
#define TEST_MEMORY_STORAGE_H

#include <stdint.h>

#include "anemone/status.h"
#include "anemone/storage_seam.h"

#define TEST_MEMORY_STORAGE_BLOCK_BYTES 4096

extern const struct storage_operations_record test_memory_storage;

void test_memory_storage_reset(void);
void test_memory_storage_set_pack(const void *pack_bytes, uint64_t pack_byte_count);
MUST_CHECK result_status test_memory_storage_hold_tables_of(
	const struct storage_operations_record *storage,
	const struct storage_reader *reader);
void test_memory_storage_fail_reads(uint32_t copy_index, uint64_t first_failing_read_number);
void test_memory_storage_flip_bit(uint32_t copy_index, uint64_t byte_offset);

/* How many reads of that copy were asked for (failed ones included) and how many bytes it delivered. */
uint64_t test_memory_storage_read_count(uint32_t copy_index);
uint64_t test_memory_storage_delivered_byte_count(uint32_t copy_index);

#endif
