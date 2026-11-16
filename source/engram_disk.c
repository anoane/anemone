/* SPDX-License-Identifier: AGPL-3.0-or-later */
#include "anemone/engram_disk.h"

#include <inttypes.h>
#include <limits.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/mman.h>

#include "anemone/pack_digest.h"
#include "anemone/hooks.h"
#include "storage_batch.h"
#include "engram_step.h"
#include "engram_hot.h"

#define VERIFY_CHUNK_BYTES (16ull << 20u)
#define CACHE_SHARDS 16u
#define READ_PASSES \
	2u /* the copies without a recent failure, then the others */
#define CACHE_EMPTY UINT64_MAX
#define PREPARATION_NS_SECOND 1000000000ull
#define MAXIMUM_DEVICE_WEIGHT 1000000u
#define BLOCK_READ_ATTEMPTS \
	2u /* per copy and block: one retry of a transient error or a torn read */

#define BLOCK_DIGEST_BYTES \
	sizeof(uint64_t) /* one block's digest in a table's block_digests */

struct disk_table {
	struct pack_extent extent;
	unsigned char *ram;
	uint64_t ram_blocks;
	uint64_t *block_digests;
};
struct cache_slot {
	uint64_t block;
	uint32_t table;
};
struct engram_disk {
	uint64_t preparation_deadline;
	const volatile sig_atomic_t *preparation_stop;
	struct storage_batch *batch;
	struct engram_description description;
	const struct storage_operations_record *storage;
	const struct storage_reader *reader;
	struct storage_description copies;
	struct disk_table tables[ENGRAM_MAXIMUM_TABLES];
	uint8_t *external;
	struct cache_slot *slots;
	unsigned char *cache;
	uint64_t slot_count;
	uint64_t slots_in_use; /* slot_count, or 0 while an explicit tune measures uncached reads */
	pthread_mutex_t mutexes[CACHE_SHARDS];
	uint32_t mutex_count;
	uint32_t interleave;
	const struct engram_hot *
		hot; /* NULL, or the static hot set held beside the RAM prefix */
	uint32_t cumulative_weights[STORAGE_MAXIMUM_COPIES];
	uint32_t total_weight;
	atomic_uint next_copy;
	atomic_uint
		failed_copies; /* copies whose last read failed: tried last, never skipped */
	_Atomic uint64_t
		verified_reads[STORAGE_MAXIMUM_COPIES]; /* per copy, any path */
};

result_status engram_disk_weights(struct engram_disk *disk,
				  const uint32_t *weights, uint32_t count)
{
	if (disk == NULL || weights == NULL || count != disk->copies.copy_count)
		return STATUS_INVALID_ARGUMENT;
	uint32_t sum = 0;
	for (uint32_t index = 0; index < count; index++) {
		if (weights[index] > MAXIMUM_DEVICE_WEIGHT)
			return STATUS_INVALID_ARGUMENT;
		sum += weights[index];
	}
	if (sum == 0)
		return STATUS_INVALID_ARGUMENT;
	disk->total_weight = sum;
	sum = 0;
	for (uint32_t index = 0; index < count; index++) {
		sum += weights[index];
		disk->cumulative_weights[index] = sum;
	}
	return STATUS_OK;
}

static uint32_t first_copy(struct engram_disk *disk, uint64_t block)
{
	if (disk->total_weight != 0) {
		const uint32_t tick =
			atomic_fetch_add_explicit(&disk->next_copy, 1,
						  memory_order_relaxed) %
			disk->total_weight;
		for (uint32_t index = 0; index < disk->copies.copy_count;
		     index++)
			if (tick < disk->cumulative_weights[index])
				return index;
	}
	return disk->interleave ?
		       (uint32_t)((block + atomic_fetch_add_explicit(
						   &disk->next_copy, 1,
						   memory_order_relaxed)) %
				  disk->copies.copy_count) :
		       0;
}

void engram_disk_close(struct engram_disk *disk)
{
	if (disk == NULL)
		return;
	storage_batch_close(disk->batch);
	for (uint32_t index = 0; index < disk->mutex_count; index++)
		(void)pthread_mutex_destroy(&disk->mutexes[index]);
	for (uint32_t table = 0; table < ENGRAM_MAXIMUM_TABLES; table++) {
		free(disk->tables[table].ram);
		free(disk->tables[table].block_digests);
	}
	free(disk->external);
	free(disk->slots);
	free(disk->cache);
	free(disk);
}

static _Atomic uint32_t scan_threads_per_copy =
	ENGRAM_DISK_SCAN_THREADS_DEFAULT;

void engram_disk_scan_threads(uint32_t threads_per_copy)
{
	atomic_store(&scan_threads_per_copy,
		     threads_per_copy < ENGRAM_DISK_SCAN_THREADS_MOST ?
			     threads_per_copy :
			     ENGRAM_DISK_SCAN_THREADS_MOST);
}

static bool preparation_over(const struct engram_disk *disk)
{
	if (disk->preparation_deadline == 0)
		return false;
	struct timespec now;
	return (disk->preparation_stop != NULL &&
		*disk->preparation_stop != 0) ||
	       clock_gettime(CLOCK_MONOTONIC, &now) != 0 ||
	       (uint64_t)now.tv_sec * PREPARATION_NS_SECOND +
			       (uint64_t)now.tv_nsec >=
		       disk->preparation_deadline;
}

/* One table's verification by readers of some copies: each takes the next piece of the extent when it is free (so a
 * faster drive reads more), adds that piece's share of the extent digest (a sum over the pieces, in any order),
 * records its blocks' digests and fills the RAM prefix. The first failure stops every reader. */
struct table_scan {
	struct engram_disk *disk;
	struct disk_table *table;
	_Atomic uint64_t cursor;
	_Atomic int status;
	_Atomic bool
		deadline_hit; /* the preparation's deadline or stop, apart from a read's own timeout */
};
struct scan_reader {
	struct table_scan *scan;
	uint32_t copy;
	bool started;
	uint64_t digest;
	unsigned char *workspace;
	pthread_t thread;
};

static void scan_fail(struct table_scan *scan, result_status status)
{
	int none = STATUS_OK;
	(void)atomic_compare_exchange_strong(&scan->status, &none, (int)status);
}

static void *scan_read(void *argument)
{
	struct scan_reader *const self = argument;
	struct table_scan *const scan = self->scan;
	struct engram_disk *const disk = scan->disk;
	struct disk_table *const table = scan->table;
	const uint64_t ram_bytes = table->ram_blocks * PACK_BLOCK_BYTES;
	while (atomic_load_explicit(&scan->status, memory_order_relaxed) ==
	       STATUS_OK) {
		if (preparation_over(disk)) {
			atomic_store(&scan->deadline_hit, true);
			scan_fail(scan, STATUS_TIMED_OUT);
			break;
		}
		const uint64_t offset =
			atomic_fetch_add(&scan->cursor, VERIFY_CHUNK_BYTES);
		if (offset >= table->extent.byte_count)
			break;
		const uint64_t left = table->extent.byte_count - offset;
		const uint64_t bytes =
			left < VERIFY_CHUNK_BYTES ? left : VERIFY_CHUNK_BYTES;
		const result_status status =
			FAULT_POINT(engram_disk_scan_timeout) ?
				STATUS_TIMED_OUT : /* a read past the reader's own limit */
				disk->storage->reader_read(
					disk->reader, self->copy,
					table->extent.byte_offset + offset,
					self->workspace, bytes);
		if (status != STATUS_OK) {
			scan_fail(scan, status);
			break;
		}
		self->digest +=
			pack_digest_of_piece(self->workspace, bytes,
					     offset / PACK_DIGEST_WORD_BYTES);
		for (uint64_t piece = 0; piece < bytes;
		     piece += PACK_BLOCK_BYTES)
			table->block_digests[(offset + piece) /
					     PACK_BLOCK_BYTES] =
				pack_digest_of_piece(self->workspace + piece,
						     PACK_BLOCK_BYTES, 0);
		if (offset < ram_bytes)
			memcpy(table->ram + offset, self->workspace,
			       (size_t)(ram_bytes - offset < bytes ?
						ram_bytes - offset :
						bytes));
	}
	return NULL;
}

/* The calling thread reads too; a reader that does not start leaves its pieces to the others. */
MUST_CHECK static result_status
table_scan(struct engram_disk *disk, uint32_t table_index, uint32_t copy_mask,
	   uint32_t threads_per_copy, bool *deadline_hit)
{
	struct table_scan scan = { .disk = disk,
				   .table = &disk->tables[table_index] };
	atomic_init(&scan.cursor, 0);
	atomic_init(&scan.status, STATUS_OK);
	atomic_init(&scan.deadline_hit, false);
	struct scan_reader
		readers[STORAGE_MAXIMUM_COPIES * ENGRAM_DISK_SCAN_THREADS_MOST];
	uint32_t count = 0;
	result_status status = STATUS_OK;
	for (uint32_t copy = 0; copy < disk->copies.copy_count; copy++)
		for (uint32_t thread = 0;
		     (copy_mask >> copy & 1u) != 0 && thread < threads_per_copy;
		     thread++) {
			readers[count] = (struct scan_reader){
				.scan = &scan,
				.copy = copy,
				.workspace = aligned_alloc(PACK_BLOCK_BYTES,
							   VERIFY_CHUNK_BYTES)
			};
			status = readers[count++].workspace == NULL ?
					 STATUS_OUT_OF_MEMORY :
					 status;
		}
	if (status == STATUS_OK && count != 0) {
		for (uint32_t index = 1; index < count; index++)
			readers[index].started =
				pthread_create(&readers[index].thread, NULL,
					       scan_read, &readers[index]) == 0;
		(void)scan_read(&readers[0]);
		for (uint32_t index = 1; index < count; index++)
			if (readers[index].started)
				(void)pthread_join(readers[index].thread, NULL);
	}
	uint64_t digest = 0;
	for (uint32_t index = 0; index < count; index++) {
		digest += readers[index].digest;
		free(readers[index].workspace);
	}
	status = status == STATUS_OK ?
			 (result_status)atomic_load(&scan.status) :
			 status;
	*deadline_hit = atomic_load(&scan.deadline_hit);
	return status != STATUS_OK		   ? status :
	       count == 0			   ? STATUS_INVALID_ARGUMENT :
	       digest == scan.table->extent.digest ? STATUS_OK :
						     STATUS_CORRUPT_PACK;
}

/* Every copy at once first; when that fails, one copy at a time (with zero threads a copy, today's serial scan), so
 * that a corrupt or failing copy spoils only its own turn: a read's own timeout is such a failure, and only the
 * preparation's deadline (or stop) or a refused allocation ends the turns. No unverified RAM is published. */
MUST_CHECK static result_status table_prepare(struct engram_disk *disk,
					      uint32_t table_index)
{
	const uint32_t copies = disk->copies.copy_count;
	const uint32_t threads = atomic_load(&scan_threads_per_copy);
	bool deadline_hit = false;
	result_status status = threads != 0 && copies > 1 ?
				       table_scan(disk, table_index,
						  (1u << copies) - 1u, threads,
						  &deadline_hit) :
				       STATUS_INPUT_OUTPUT_ERROR;
	for (uint32_t copy = 0; status != STATUS_OK && !deadline_hit &&
				status != STATUS_OUT_OF_MEMORY && copy < copies;
	     copy++)
		status = table_scan(disk, table_index, 1u << copy,
				    threads != 0 ? threads : 1u, &deadline_hit);
	return status;
}

static uint64_t cache_slot_count(uint64_t cache_bytes)
{
	return cache_bytes / PACK_BLOCK_BYTES / CACHE_SHARDS * CACHE_SHARDS;
}

uint64_t engram_disk_ram_blocks(uint64_t block_count, uint32_t ram_percent)
{
	return block_count / ENGRAM_RAM_PERCENT * ram_percent +
	       block_count % ENGRAM_RAM_PERCENT * ram_percent /
		       ENGRAM_RAM_PERCENT;
}

uint64_t engram_disk_host_bytes(const uint64_t *table_blocks,
				uint32_t table_count, uint32_t ram_percent,
				uint64_t cache_bytes)
{
	const uint64_t slots = cache_slot_count(cache_bytes);
	uint64_t bytes = slots * (PACK_BLOCK_BYTES + sizeof(struct cache_slot));
	for (uint32_t table = 0; table < table_count; table++) {
		const uint64_t blocks = table_blocks[table];
		const uint64_t table_bytes =
			engram_disk_ram_blocks(blocks, ram_percent) *
				PACK_BLOCK_BYTES +
			blocks * BLOCK_DIGEST_BYTES;
		bytes = table_bytes > UINT64_MAX - bytes ? UINT64_MAX :
							   bytes + table_bytes;
	}
	return bytes;
}

result_status
engram_disk_open_until(const struct engram_description *description,
		       const struct pack_record *pack,
		       const struct storage_operations_record *storage,
		       const struct storage_reader *reader,
		       uint32_t ram_percent, uint64_t cache_bytes,
		       uint32_t interleave_copies, uint32_t batch_reads,
		       uint64_t deadline_ns, const volatile sig_atomic_t *stop,
		       struct engram_disk **disk_out)
{
	if (description == NULL || pack == NULL || storage == NULL ||
	    storage->reader_describe == NULL || storage->reader_read == NULL ||
	    reader == NULL || disk_out == NULL ||
	    ram_percent >= ENGRAM_RAM_PERCENT ||
	    cache_bytes > ENGRAM_CACHE_MAXIMUM_BYTES || interleave_copies > 1 ||
	    batch_reads > 1 || description->table_count == 0 ||
	    description->table_count > ENGRAM_MAXIMUM_TABLES)
		return STATUS_INVALID_ARGUMENT;
	struct engram_disk *const disk = FAULT_POINT(engram_disk_memory) ?
						 NULL :
						 calloc(1, sizeof(*disk));
	if (disk == NULL)
		return STATUS_OUT_OF_MEMORY;
	disk->preparation_deadline = deadline_ns;
	disk->preparation_stop = stop;
	disk->description = *description;
	disk->storage = storage;
	disk->reader = reader;
	disk->interleave = interleave_copies;
	atomic_init(&disk->next_copy, 0);
	atomic_init(&disk->failed_copies, 0);
	result_status status = storage->reader_describe(reader, &disk->copies);
	if (status == STATUS_OK && batch_reads != 0)
		status = storage == &storage_direct ?
				 storage_batch_open(reader, &disk->batch) :
				 STATUS_UNSUPPORTED;
	if (status == STATUS_OK &&
	    (disk->copies.copy_count == 0 ||
	     disk->copies.copy_count > STORAGE_MAXIMUM_COPIES ||
	     disk->copies.copy_count > sizeof(unsigned) * CHAR_BIT ||
	     disk->copies.block_alignment_bytes == 0 ||
	     PACK_BLOCK_BYTES % disk->copies.block_alignment_bytes != 0))
		status = STATUS_INVALID_ARGUMENT;
	disk->external = calloc(pack_header_of(pack)->extent_count, 1);
	disk->slot_count = cache_slot_count(cache_bytes);
	disk->slots_in_use = disk->slot_count;
	if (disk->external == NULL)
		status = STATUS_OUT_OF_MEMORY;
	if (status == STATUS_OK && disk->slot_count != 0) {
		disk->slots =
			calloc((size_t)disk->slot_count, sizeof(*disk->slots));
		disk->cache = aligned_alloc(
			PACK_BLOCK_BYTES,
			(size_t)(disk->slot_count * PACK_BLOCK_BYTES));
		if (disk->slots == NULL || disk->cache == NULL)
			status = STATUS_OUT_OF_MEMORY;
		for (uint64_t slot = 0;
		     status == STATUS_OK && slot < disk->slot_count; slot++)
			disk->slots[slot].block = CACHE_EMPTY;
		while (status == STATUS_OK &&
		       disk->mutex_count < CACHE_SHARDS) {
			if (pthread_mutex_init(
				    &disk->mutexes[disk->mutex_count], NULL) !=
			    0)
				status = STATUS_OUT_OF_MEMORY;
			else
				disk->mutex_count++;
		}
	}
	for (uint32_t index = 0;
	     status == STATUS_OK && index < description->table_count; index++) {
		const uint32_t unit_index =
			description->table_unit_index[index];
		if (unit_index >= pack_header_of(pack)->unit_count) {
			status = STATUS_INVALID_ARGUMENT;
			break;
		}
		const struct pack_unit_record_2 *const unit =
			&pack_units_of(pack)[unit_index];
		if (unit->unit_kind != PLACEMENT_UNIT_HOST_ONLY ||
		    unit->extent_count != 1) {
			status = STATUS_INVALID_ARGUMENT;
			break;
		}
		struct disk_table *const table = &disk->tables[index];
		table->extent = pack_extents_of(pack)[unit->first_extent_index];
		const uint64_t blocks =
			table->extent.byte_count / PACK_BLOCK_BYTES;
		if (blocks == 0 || description->row_count[index] == 0 ||
		    blocks != (description->row_count[index] - 1) /
					      ENGRAM_ROWS_PER_BLOCK +
				      1 ||
		    blocks > SIZE_MAX / sizeof(uint64_t) ||
		    table->extent.byte_count > SIZE_MAX) {
			status = STATUS_INVALID_ARGUMENT;
			break;
		}
		table->ram_blocks = engram_disk_ram_blocks(blocks, ram_percent);
		table->ram = table->ram_blocks ?
				     aligned_alloc(PACK_BLOCK_BYTES,
						   (size_t)(table->ram_blocks *
							    PACK_BLOCK_BYTES)) :
				     NULL;
		if (table->ram != NULL &&
		    madvise(table->ram,
			    (size_t)(table->ram_blocks * PACK_BLOCK_BYTES),
			    MADV_HUGEPAGE) != 0)
			LOG_WARNING_ENTRY("Engram RAM prefix uses small pages");
		table->block_digests =
			calloc((size_t)blocks, BLOCK_DIGEST_BYTES);
		if ((table->ram_blocks && table->ram == NULL) ||
		    table->block_digests == NULL) {
			status = STATUS_OUT_OF_MEMORY;
			break;
		}
		disk->external[unit->first_extent_index] = 1;
		status = table_prepare(disk, index);
	}
	if (status != STATUS_OK) {
		engram_disk_close(disk);
		return status;
	}
	*disk_out = disk;
	return STATUS_OK;
}

const uint8_t *engram_disk_external_extents(const struct engram_disk *disk)
{
	return disk != NULL ? disk->external : NULL;
}
const struct engram_description *
engram_disk_description(const struct engram_disk *disk)
{
	return disk != NULL ? &disk->description : NULL;
}

MUST_CHECK static result_status block_read(struct engram_disk *disk,
					   uint32_t table_index, uint64_t block,
					   unsigned char *destination)
{
	const struct disk_table *const table = &disk->tables[table_index];
	const uint32_t first = first_copy(disk, block);
	const unsigned failed = atomic_load_explicit(&disk->failed_copies,
						     memory_order_relaxed);
	result_status status = STATUS_INPUT_OUTPUT_ERROR;
	/* Copies without a recent failure first, then the others: a read error or bad block never retires a copy. */
	for (uint32_t pass = 0; pass < READ_PASSES; pass++)
		for (uint32_t attempt = 0; attempt < disk->copies.copy_count;
		     attempt++) {
			const uint32_t copy =
				(first + attempt) % disk->copies.copy_count;
			const unsigned bit = 1u << copy;
			if (((failed & bit) != 0) != (pass != 0))
				continue;
			for (uint32_t read = 0; read < BLOCK_READ_ATTEMPTS;
			     read++) {
				status = disk->storage->reader_read(
					disk->reader, copy,
					table->extent.byte_offset +
						block * PACK_BLOCK_BYTES,
					destination, PACK_BLOCK_BYTES);
				if (status == STATUS_OK &&
				    pack_digest_of_piece(destination,
							 PACK_BLOCK_BYTES, 0) ==
					    table->block_digests[block]) {
					(void)atomic_fetch_and_explicit(
						&disk->failed_copies, ~bit,
						memory_order_relaxed);
					(void)atomic_fetch_add_explicit(
						&disk->verified_reads[copy], 1,
						memory_order_relaxed);
					return STATUS_OK;
				}
				if (status == STATUS_OK)
					status = STATUS_CORRUPT_PACK;
			}
			(void)atomic_fetch_or_explicit(&disk->failed_copies,
						       bit,
						       memory_order_relaxed);
		}
	/* A product line, in every build (a logging hook is compiled out of release); only on this failure path. */
	(void)fprintf(stderr,
		      "engram: table %u block %" PRIu64
		      " failed on every one of %u copies, %u reads each: %s\n",
		      table_index, block, disk->copies.copy_count,
		      BLOCK_READ_ATTEMPTS, status_text(status));
	return status;
}

/* Cache hits hold their shard only for a copy. Misses are batched outside cache locks. */
MUST_CHECK static result_status cached_row(struct engram_disk *disk,
					   uint32_t table, uint64_t row,
					   unsigned char *bytes, bool store,
					   bool *hit_out)
{
	*hit_out = false;
	if (disk->slots_in_use == 0)
		return STATUS_OK;
	const uint64_t block = row / ENGRAM_ROWS_PER_BLOCK;
	const uint64_t slot = (block * disk->description.table_count + table) %
			      disk->slot_count;
	pthread_mutex_t *const mutex = &disk->mutexes[slot % CACHE_SHARDS];
	struct timespec deadline;
	if (clock_gettime(CLOCK_REALTIME, &deadline) != 0)
		return STATUS_TIMED_OUT;
	deadline.tv_sec += STORAGE_READ_LIMIT_SECONDS;
	if (pthread_mutex_timedlock(mutex, &deadline) != 0)
		return STATUS_TIMED_OUT;
	struct cache_slot *const entry = &disk->slots[slot];
	unsigned char *const cached = disk->cache + slot * PACK_BLOCK_BYTES;
	if (store) {
		memcpy(cached, bytes, PACK_BLOCK_BYTES);
		entry->block = block;
		entry->table = table;
	} else if (entry->block == block && entry->table == table) {
		memcpy(bytes,
		       cached + row % ENGRAM_ROWS_PER_BLOCK * ENGRAM_ROW_BYTES,
		       ENGRAM_ROW_BYTES);
		*hit_out = true;
	}
	return pthread_mutex_unlock(mutex) == 0 ? STATUS_OK :
						  STATUS_DEVICE_ERROR;
}

MUST_CHECK static result_status batch_gather(struct engram_disk *disk,
					     uint32_t table_index,
					     const uint64_t *row_ids,
					     uint32_t row_count,
					     unsigned char *rows_out)
{
	_Alignas(PACK_BLOCK_BYTES) unsigned char
		blocks[STORAGE_BATCH_DEPTH * PACK_BLOCK_BYTES];
	const struct disk_table *const table = &disk->tables[table_index];
	for (uint32_t begin = 0; begin < row_count;
	     begin += STORAGE_BATCH_DEPTH) {
		const uint32_t left = row_count - begin;
		const uint32_t count =
			left < STORAGE_BATCH_DEPTH ? left : STORAGE_BATCH_DEPTH;
		uint32_t copies[STORAGE_BATCH_DEPTH],
			indices[STORAGE_BATCH_DEPTH], misses = 0;
		uint64_t offsets[STORAGE_BATCH_DEPTH];
		result_status results[STORAGE_BATCH_DEPTH];
		for (uint32_t index = begin; index < begin + count; index++) {
			const uint64_t row = row_ids[index],
				       block = row / ENGRAM_ROWS_PER_BLOCK;
			unsigned char *const output =
				rows_out + (size_t)index * ENGRAM_ROW_BYTES;
			const unsigned char *const held =
				engram_disk_ram_block(disk, table_index, block);
			if (held != NULL) {
				memcpy(output,
				       held + row % ENGRAM_ROWS_PER_BLOCK *
						       ENGRAM_ROW_BYTES,
				       ENGRAM_ROW_BYTES);
				continue;
			}
			bool hit = false;
			const result_status status = cached_row(
				disk, table_index, row, output, false, &hit);
			if (status != STATUS_OK)
				return status;
			if (hit)
				continue;
			uint32_t copy = first_copy(disk, block);
			for (uint32_t attempt = 0;
			     attempt < disk->copies.copy_count &&
			     (atomic_load_explicit(&disk->failed_copies,
						   memory_order_relaxed) &
			      (1u << copy)) != 0;
			     attempt++)
				copy = (copy + 1u) % disk->copies.copy_count;
			copies[misses] = copy;
			offsets[misses] = table->extent.byte_offset +
					  block * PACK_BLOCK_BYTES;
			indices[misses++] = index;
		}
		if (misses == 0)
			continue;
		const result_status batch_status = storage_batch_read(
			disk->batch, copies, offsets, misses, blocks, results);
		for (uint32_t miss = 0; miss < misses; miss++) {
			const uint32_t index = indices[miss];
			const uint64_t block =
				row_ids[index] / ENGRAM_ROWS_PER_BLOCK;
			unsigned char *const bytes =
				blocks + (size_t)miss * PACK_BLOCK_BYTES;
			if (batch_status != STATUS_OK ||
			    results[miss] != STATUS_OK ||
			    pack_digest_of_piece(bytes, PACK_BLOCK_BYTES, 0) !=
				    table->block_digests[block]) {
				if (batch_status == STATUS_OK)
					(void)atomic_fetch_or_explicit(
						&disk->failed_copies,
						1u << copies[miss],
						memory_order_relaxed);
				const result_status status = block_read(
					disk, table_index, block, bytes);
				if (status != STATUS_OK)
					return status;
			} else {
				(void)atomic_fetch_and_explicit(
					&disk->failed_copies,
					~(1u << copies[miss]),
					memory_order_relaxed);
			}
			bool hit = false;
			const result_status status =
				cached_row(disk, table_index, row_ids[index],
					   bytes, true, &hit);
			if (status != STATUS_OK)
				return status;
			memcpy(rows_out + (size_t)index * ENGRAM_ROW_BYTES,
			       bytes + row_ids[index] % ENGRAM_ROWS_PER_BLOCK *
					       ENGRAM_ROW_BYTES,
			       ENGRAM_ROW_BYTES);
		}
	}
	return STATUS_OK;
}

result_status engram_disk_gather(struct engram_disk *disk, uint32_t table_index,
				 const uint64_t *row_ids, uint32_t row_count,
				 void *rows_out)
{
	if (disk == NULL || row_ids == NULL || rows_out == NULL ||
	    table_index >= disk->description.table_count ||
	    row_count > ENGRAM_MAXIMUM_ROWS_PER_TOKEN)
		return STATUS_INVALID_ARGUMENT;
	for (uint32_t index = 0; index < row_count; index++)
		if (row_ids[index] >= disk->description.row_count[table_index])
			return STATUS_INVALID_ARGUMENT;
	if (disk->batch != NULL)
		return batch_gather(disk, table_index, row_ids, row_count,
				    rows_out);
	_Alignas(PACK_BLOCK_BYTES) unsigned char scratch[PACK_BLOCK_BYTES];
	for (uint32_t index = 0; index < row_count; index++) {
		const uint64_t block = row_ids[index] / ENGRAM_ROWS_PER_BLOCK;
		const uint64_t offset = row_ids[index] % ENGRAM_ROWS_PER_BLOCK *
					ENGRAM_ROW_BYTES;
		unsigned char *const output = (unsigned char *)rows_out +
					      (size_t)index * ENGRAM_ROW_BYTES;
		const unsigned char *const held =
			engram_disk_ram_block(disk, table_index, block);
		if (held != NULL) {
			memcpy(output, held + offset, ENGRAM_ROW_BYTES);
			continue;
		}
		if (disk->slots_in_use == 0) {
			const result_status status =
				block_read(disk, table_index, block, scratch);
			if (status != STATUS_OK)
				return status;
			memcpy(output, scratch + offset, ENGRAM_ROW_BYTES);
			continue;
		}
		const uint64_t slot =
			(block * disk->description.table_count + table_index) %
			disk->slot_count;
		pthread_mutex_t *const mutex =
			&disk->mutexes[slot % CACHE_SHARDS];
		struct timespec deadline;
		if (clock_gettime(CLOCK_REALTIME, &deadline) != 0)
			return STATUS_TIMED_OUT;
		deadline.tv_sec += STORAGE_READ_LIMIT_SECONDS;
		if (pthread_mutex_timedlock(mutex, &deadline) != 0)
			return STATUS_TIMED_OUT;
		struct cache_slot *const entry = &disk->slots[slot];
		unsigned char *const bytes =
			disk->cache + slot * PACK_BLOCK_BYTES;
		result_status status = STATUS_OK;
		if (entry->block != block || entry->table != table_index) {
			entry->block = CACHE_EMPTY;
			status = block_read(disk, table_index, block, bytes);
			if (status == STATUS_OK) {
				entry->block = block;
				entry->table = table_index;
			}
		}
		if (status == STATUS_OK)
			memcpy(output, bytes + offset, ENGRAM_ROW_BYTES);
		if (pthread_mutex_unlock(mutex) != 0)
			return STATUS_DEVICE_ERROR;
		if (status != STATUS_OK)
			return status;
	}
	return STATUS_OK;
}

result_status engram_disk_tune_io(struct engram_disk *disk,
				  const uint32_t *weights, uint32_t count,
				  uint32_t batch_reads, bool cached)
{
	if (disk == NULL || batch_reads > 1u)
		return STATUS_INVALID_ARGUMENT;
	if (batch_reads != 0 && disk->storage != &storage_direct)
		return STATUS_UNSUPPORTED;
	const result_status status = engram_disk_weights(disk, weights, count);
	if (status != STATUS_OK)
		return status;
	storage_batch_close(disk->batch);
	disk->batch = NULL;
	if (batch_reads != 0) {
		const result_status opened =
			storage_batch_open(disk->reader, &disk->batch);
		if (opened != STATUS_OK)
			return opened;
	}
	for (uint64_t slot = 0; slot < disk->slot_count; slot++)
		disk->slots[slot].block = CACHE_EMPTY;
	disk->slots_in_use = cached ? disk->slot_count : 0;
	atomic_store_explicit(&disk->failed_copies, 0, memory_order_relaxed);
	atomic_store_explicit(&disk->next_copy, 0, memory_order_relaxed);
	return STATUS_OK;
}

result_status engram_disk_open(const struct engram_description *description,
			       const struct pack_record *pack,
			       const struct storage_operations_record *storage,
			       const struct storage_reader *reader,
			       uint32_t ram_percent, uint64_t cache_bytes,
			       uint32_t interleave_copies, uint32_t batch_reads,
			       struct engram_disk **disk_out)
{
	return engram_disk_open_until(description, pack, storage, reader,
				      ram_percent, cache_bytes,
				      interleave_copies, batch_reads, 0, NULL,
				      disk_out);
}

/* ---- what step I/O (engram_step.c) reads of the store: none allocates, all are safe from any thread ---------- */

const struct storage_reader *engram_disk_reader(const struct engram_disk *disk)
{
	return disk->reader;
}

uint32_t engram_disk_copy_count(const struct engram_disk *disk)
{
	return disk->copies.copy_count;
}

uint64_t engram_disk_block_offset(const struct engram_disk *disk,
				  uint32_t table, uint64_t block)
{
	return disk->tables[table].extent.byte_offset +
	       block * PACK_BLOCK_BYTES;
}

uint64_t engram_disk_block_count(const struct engram_disk *disk, uint32_t table)
{
	return disk->tables[table].extent.byte_count / PACK_BLOCK_BYTES;
}

const unsigned char *engram_disk_ram_block(const struct engram_disk *disk,
					   uint32_t table, uint64_t block)
{
	const struct disk_table *const own = &disk->tables[table];
	return block < own->ram_blocks ?
		       own->ram + block * PACK_BLOCK_BYTES :
		       engram_hot_block(disk->hot, table, block);
}

void engram_disk_attach_hot(struct engram_disk *disk,
			    const struct engram_hot *hot)
{
	if (disk != NULL)
		disk->hot = hot;
}

uint64_t engram_disk_extent_digest(const struct engram_disk *disk,
				   uint32_t table)
{
	return disk != NULL && table < disk->description.table_count ?
		       disk->tables[table].extent.digest :
		       0;
}

bool engram_disk_block_good(const struct engram_disk *disk, uint32_t table,
			    uint64_t block, const unsigned char *bytes)
{
	return pack_digest_of_piece(bytes, PACK_BLOCK_BYTES, 0) ==
	       disk->tables[table].block_digests[block];
}

/* The cache's rows of one block, under a try-lock: a busy shard is a miss (the block is read and verified instead),
 * so no probe waits and none reads the clock. True when the block was cached and its rows are copied. */
bool engram_disk_cache_rows(struct engram_disk *disk, uint32_t table,
			    uint64_t block, const uint32_t *row_offsets,
			    unsigned char *const *outputs, uint32_t count)
{
	if (disk->slots_in_use == 0)
		return false;
	const uint64_t slot = (block * disk->description.table_count + table) %
			      disk->slot_count;
	pthread_mutex_t *const mutex = &disk->mutexes[slot % CACHE_SHARDS];
	if (pthread_mutex_trylock(mutex) != 0)
		return false;
	const bool hit = disk->slots[slot].block == block &&
			 disk->slots[slot].table == table;
	const unsigned char *const cached =
		disk->cache + slot * PACK_BLOCK_BYTES;
	for (uint32_t index = 0; hit && index < count; index++)
		memcpy(outputs[index], cached + row_offsets[index],
		       ENGRAM_ROW_BYTES);
	(void)pthread_mutex_unlock(mutex);
	return hit;
}

/* A verified block into the cache, under a try-lock: a busy shard skips the store. */
void engram_disk_cache_store(struct engram_disk *disk, uint32_t table,
			     uint64_t block, const unsigned char *bytes)
{
	if (disk->slots_in_use == 0)
		return;
	const uint64_t slot = (block * disk->description.table_count + table) %
			      disk->slot_count;
	pthread_mutex_t *const mutex = &disk->mutexes[slot % CACHE_SHARDS];
	if (pthread_mutex_trylock(mutex) != 0)
		return;
	memcpy(disk->cache + slot * PACK_BLOCK_BYTES, bytes, PACK_BLOCK_BYTES);
	disk->slots[slot].block = block;
	disk->slots[slot].table = table;
	(void)pthread_mutex_unlock(mutex);
}

/* The synchronous fallback: every copy, those without a recent failure first, two reads each. */
result_status engram_disk_block_fetch(struct engram_disk *disk, uint32_t table,
				      uint64_t block,
				      unsigned char *destination)
{
	return block_read(disk, table, block, destination);
}

/* The copies whose last read failed (tried last, never skipped), and a copy's new state. */
uint32_t engram_disk_failed_copies(const struct engram_disk *disk)
{
	return atomic_load_explicit(&disk->failed_copies, memory_order_relaxed);
}

void engram_disk_copy_state(struct engram_disk *disk, uint32_t copy,
			    bool failed)
{
	if (failed) {
		(void)atomic_fetch_or_explicit(&disk->failed_copies, 1u << copy,
					       memory_order_relaxed);
		return;
	}
	(void)atomic_fetch_and_explicit(&disk->failed_copies, ~(1u << copy),
					memory_order_relaxed);
	if (copy < STORAGE_MAXIMUM_COPIES)
		(void)atomic_fetch_add_explicit(&disk->verified_reads[copy], 1,
						memory_order_relaxed);
}

uint64_t engram_disk_copy_verified(const struct engram_disk *disk,
				   uint32_t copy)
{
	return copy < STORAGE_MAXIMUM_COPIES ?
		       atomic_load_explicit(&disk->verified_reads[copy],
					    memory_order_relaxed) :
		       0;
}
