/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Immutable Engram block store. Preparation owns RAM prefixes, block digests and a bounded shared cache.
 * The storage and reader are borrowed until close; their copies must be immutable and on admitted local devices.
 * Open reads each table once with bounded workspace and checks its original extent digest. Runtime verifies each
 * disk block against that preparation, retries other full replicas after I/O/corruption and never allocates.
 * Concurrent gathers are safe; close requires every caller to have retired. Output buffers are caller-owned. */
#ifndef ENGRAM_DISK_H
#define ENGRAM_DISK_H

#include <stdbool.h>
#include <stdint.h>
#include <signal.h>
#include "engram.h"

#define ENGRAM_RAM_PERCENT 100u
#define ENGRAM_CACHE_MAXIMUM_BYTES (32ull << 30u) /* bench A2: 8-32 GiB help C4 and C8 */
#define ENGRAM_DISK_SCAN_THREADS_DEFAULT 2u
#define ENGRAM_DISK_SCAN_THREADS_MOST 4u
struct engram_disk;
struct engram_hot;
/* Readers per copy of the digest scan that every later open runs over the tables (process-wide; at most
 * ENGRAM_DISK_SCAN_THREADS_MOST): every copy at once, each reader taking the next piece, then one copy at a time if
 * that fails. 0: one reader, copy by copy (the serial scan). */
void engram_disk_scan_threads(uint32_t threads_per_copy);
/* The blocks of a table of block_count blocks that a store opened with ram_percent (below ENGRAM_RAM_PERCENT) keeps in
 * RAM, its prefix: engram_disk_open allocates exactly these, and engram_fit.h budgets them. */
uint64_t engram_disk_ram_blocks(uint64_t block_count, uint32_t ram_percent);
/* The host bytes such a store allocates for tables of these block counts and cache_bytes of cache: each table's RAM
 * prefix and block digests, and the cache's blocks and slots (UINT64_MAX when the sum wraps). */
uint64_t engram_disk_host_bytes(const uint64_t *table_blocks,
				uint32_t table_count, uint32_t ram_percent,
				uint64_t cache_bytes);
MUST_CHECK result_status
engram_disk_open(const struct engram_description *description,
		 const struct pack_record *pack,
		 const struct storage_operations_record *storage,
		 const struct storage_reader *reader, uint32_t ram_percent,
		 uint64_t cache_bytes, uint32_t interleave_copies,
		 uint32_t batch_reads, struct engram_disk **disk_out);
/* Explicit preparation may checkpoint between table scan chunks; ordinary startup uses the unchanged wrapper. */
MUST_CHECK result_status engram_disk_open_until(
	const struct engram_description *description,
	const struct pack_record *pack,
	const struct storage_operations_record *storage,
	const struct storage_reader *reader, uint32_t ram_percent,
	uint64_t cache_bytes, uint32_t interleave_copies, uint32_t batch_reads,
	uint64_t deadline_ns, const volatile sig_atomic_t *stop,
	struct engram_disk **disk_out);
/* Independent Engram sources. Each may have another model-weight layout, but every hashing constant,
 * table length and table digest must match logical_pack/engram. Returned reader is owned and closed with
 * storage_direct.reader_close after the disk store retires. Caller admits physical media before opening. */
MUST_CHECK result_status engram_disk_sources_open(
	const char *const *paths, uint32_t count,
	const struct pack_record *logical_pack,
	const struct engram_record *engram, struct storage_reader **reader_out);
void engram_disk_close(struct engram_disk *disk);
/* Set measured aggregate shares once, before attachment/gathers. All-zero or unbounded weights are refused.
 * NULL weights are never installed; the default retains the original selection exactly. */
MUST_CHECK result_status engram_disk_weights(struct engram_disk *disk,
					     const uint32_t *weights,
					     uint32_t count);
/* Explicit CPU autotune only, with every gather retired. Reuses the one validated table scan;
 * changes the production I/O mechanism/shares and clears the bounded cache between measurements. Without cached,
 * every gather reads its blocks from the copies (the raw device behaviour) until the next call. */
MUST_CHECK result_status engram_disk_tune_io(struct engram_disk *disk,
					     const uint32_t *weights,
					     uint32_t count,
					     uint32_t batch_reads, bool cached);
/* One byte per pack extent, marking only the table extents this object owns. */
const uint8_t *engram_disk_external_extents(const struct engram_disk *disk);
MUST_CHECK result_status engram_disk_gather(struct engram_disk *disk,
					    uint32_t table,
					    const uint64_t *row_ids,
					    uint32_t row_count, void *rows_out);
/* Attaches a borrowed store before any gathers. Its table description must match the Engram exactly. */
MUST_CHECK result_status engram_attach_disk(struct engram_record *engram,
					    struct engram_disk *disk);
const struct engram_description *
engram_disk_description(const struct engram_disk *disk);

/* A static hot set (engram_hot.h) served like the RAM prefix by every read path, the step plan's included; set once
 * before serving, and closed after the store. */
void engram_disk_attach_hot(struct engram_disk *disk,
			    const struct engram_hot *hot);
uint64_t engram_disk_extent_digest(const struct engram_disk *disk,
				   uint32_t table);
#endif
