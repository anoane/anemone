/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* engram_hot.c: a static hot set of n-gram table blocks in RAM, contract in engram_hot.h. */
#include "engram_hot.h"

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "engram_step.h"
#include "common/sha256.h"

#define MAGIC_BYTES 8u
/* "ANMHOTS1": eight bytes, no terminator */
static const unsigned char magic[MAGIC_BYTES] = { 'A', 'N', 'M', 'H',
						  'O', 'T', 'S', '1' };
#define HEADER_BYTES \
	40u /* magic, schema, sections, pack header digest, tables digest, file bytes */
#define AT_SCHEMA 8u
#define AT_SECTIONS 12u
#define AT_HEADER_DIGEST 16u
#define AT_TABLES_DIGEST 24u
#define AT_FILE_BYTES 32u
#define SECTION_BYTES \
	32u /* table, zero, extent digest, block count, id count */
#define AT_RESERVED 4u
#define AT_EXTENT_DIGEST 8u
#define AT_BLOCK_COUNT 16u
#define AT_ID_COUNT 24u
#define ID_BYTES 8u
#define LOAD_THREADS 8u
#define HEX_LETTER_VALUE 10
#define HEX_BASE 16
#define HEX_PER_BYTE 2u
#define MIB_BYTES (1ull << 20u)
#define HALF 2u

struct hot_table {
	uint64_t count,
		first; /* ids, and the index of the first one's block in the arena */
	uint64_t *ids;
};
struct engram_hot {
	struct hot_table tables[ENGRAM_MAXIMUM_TABLES];
	unsigned char *arena;
	uint64_t blocks, arena_bytes;
};

static void put32(unsigned char *at, uint32_t value)
{
	memcpy(at, &value, sizeof(value));
}
static void put64(unsigned char *at, uint64_t value)
{
	memcpy(at, &value, sizeof(value));
}
static uint32_t get32(const unsigned char *at)
{
	uint32_t value;
	memcpy(&value, at, sizeof(value));
	return value;
}
static uint64_t get64(const unsigned char *at)
{
	uint64_t value;
	memcpy(&value, at, sizeof(value));
	return value;
}

result_status engram_hot_encode(const struct pack_header *header,
				const struct engram_hot_table *tables,
				uint32_t table_count, unsigned char **bytes_out,
				size_t *byte_count_out)
{
	if (header == NULL || tables == NULL || bytes_out == NULL ||
	    byte_count_out == NULL || table_count == 0 ||
	    table_count > ENGRAM_MAXIMUM_TABLES)
		return STATUS_INVALID_ARGUMENT;
	uint64_t bytes = HEADER_BYTES;
	for (uint32_t index = 0; index < table_count; index++) {
		for (uint64_t id = 0; id < tables[index].id_count; id++)
			if (tables[index].ids[id] >=
				    tables[index].block_count ||
			    (id != 0 && tables[index].ids[id] <=
						tables[index].ids[id - 1]))
				return STATUS_INVALID_ARGUMENT;
		bytes += SECTION_BYTES + tables[index].id_count * ID_BYTES;
	}
	unsigned char *const out = calloc(1, (size_t)bytes);
	if (out == NULL)
		return STATUS_OUT_OF_MEMORY;
	for (uint32_t index = 0; index < MAGIC_BYTES; index++)
		out[index] = magic[index]; /* eight bytes, not a string */
	put32(out + AT_SCHEMA, ENGRAM_HOT_SCHEMA);
	put32(out + AT_SECTIONS, table_count);
	put64(out + AT_HEADER_DIGEST, header->header_digest);
	put64(out + AT_TABLES_DIGEST, header->tables_digest);
	put64(out + AT_FILE_BYTES, header->file_byte_count);
	unsigned char *at = out + HEADER_BYTES;
	for (uint32_t index = 0; index < table_count; index++) {
		put32(at, tables[index].table);
		put64(at + AT_EXTENT_DIGEST, tables[index].extent_digest);
		put64(at + AT_BLOCK_COUNT, tables[index].block_count);
		put64(at + AT_ID_COUNT, tables[index].id_count);
		at += SECTION_BYTES;
		for (uint64_t id = 0; id < tables[index].id_count;
		     id++, at += ID_BYTES)
			put64(at, tables[index].ids[id]);
	}
	*bytes_out = out;
	*byte_count_out = (size_t)bytes;
	return STATUS_OK;
}

static int hex_value(char digit)
{
	return digit >= '0' && digit <= '9' ? digit - '0' :
	       digit >= 'a' && digit <= 'f' ? digit - 'a' + HEX_LETTER_VALUE :
					      -1;
}

/* The file's bytes, read whole after the checks of what the path names. */
MUST_CHECK static result_status read_file(const char *path, uint64_t most_bytes,
					  unsigned char **bytes_out,
					  size_t *count_out, char *reason)
{
	/* O_NONBLOCK: a FIFO or a device at the path must not hold the open before the regular-file check (it changes
	 * nothing for the pread of a regular file) */
	const int descriptor = open(path, O_RDONLY | O_NOFOLLOW | O_CLOEXEC |
						  O_NONBLOCK | O_NOCTTY);
	struct stat file;
	if (descriptor < 0 || fstat(descriptor, &file) != 0) {
		(void)snprintf(
			reason, ENGRAM_HOT_REASON_BYTES,
			"cannot open %s (%s; a symbolic link is refused)", path,
			strerror(errno));
		if (descriptor >= 0)
			(void)close(descriptor);
		return STATUS_INVALID_ARGUMENT;
	}
	result_status status = STATUS_OK;
	if (!S_ISREG(file.st_mode) || file.st_uid != geteuid()) {
		(void)snprintf(reason, ENGRAM_HOT_REASON_BYTES,
			       "%s is not a regular file of this user", path);
		status = STATUS_INVALID_ARGUMENT;
	} else if ((uint64_t)file.st_size < HEADER_BYTES ||
		   (uint64_t)file.st_size > most_bytes) {
		(void)snprintf(
			reason, ENGRAM_HOT_REASON_BYTES,
			"%s is %lld bytes, outside %u..%llu for the budget",
			path, (long long)file.st_size, HEADER_BYTES,
			(unsigned long long)most_bytes);
		status = STATUS_INVALID_ARGUMENT;
	}
	unsigned char *const bytes =
		status == STATUS_OK ? malloc((size_t)file.st_size) : NULL;
	if (status == STATUS_OK &&
	    (bytes == NULL || pread(descriptor, bytes, (size_t)file.st_size,
				    0) != (ssize_t)file.st_size)) {
		(void)snprintf(reason, ENGRAM_HOT_REASON_BYTES,
			       "cannot read %s", path);
		status = bytes == NULL ? STATUS_OUT_OF_MEMORY :
					 STATUS_INPUT_OUTPUT_ERROR;
	}
	(void)close(descriptor);
	if (status != STATUS_OK) {
		free(bytes);
		return status;
	}
	*bytes_out = bytes;
	*count_out = (size_t)file.st_size;
	return STATUS_OK;
}

static bool digest_is(const unsigned char *bytes, size_t count, const char *hex)
{
	unsigned char digest[SHA256_DIGEST_BYTES];
	if (hex == NULL || strlen(hex) != ENGRAM_HOT_SHA256_HEX_BYTES ||
	    sha256_of(bytes, count, digest) != STATUS_OK)
		return false;
	for (uint32_t index = 0; index < SHA256_DIGEST_BYTES; index++) {
		const int high = hex_value(hex[(size_t)HEX_PER_BYTE * index]),
			  low = hex_value(
				  hex[(size_t)HEX_PER_BYTE * index + 1u]);
		if (high < 0 || low < 0 ||
		    high * HEX_BASE + low != (int)digest[index])
			return false;
	}
	return true;
}

/* Every check of the layout against the pack and the store; the sections' ids are copied into hot. */
MUST_CHECK static result_status
parse(const unsigned char *bytes, size_t count, uint64_t budget_bytes,
      const struct pack_header *header, const struct engram_hot_table *expected,
      uint32_t table_count, struct hot_table *tables, uint64_t *blocks_out,
      char *reason)
{
	const uint32_t sections = get32(bytes + AT_SECTIONS);
	const char *flaw = memcmp(bytes, magic, MAGIC_BYTES) != 0 ?
				   "not a hot-set file" :
			   get32(bytes + AT_SCHEMA) != ENGRAM_HOT_SCHEMA ?
				   "another schema" :
			   get64(bytes + AT_HEADER_DIGEST) !=
						   header->header_digest ||
					   get64(bytes + AT_TABLES_DIGEST) !=
						   header->tables_digest ||
					   get64(bytes + AT_FILE_BYTES) !=
						   header->file_byte_count ?
				   "made for another pack" :
			   sections == 0 || sections > table_count ?
				   "a section count outside the tables" :
				   NULL;
	uint64_t at = HEADER_BYTES, blocks = 0;
	for (uint32_t index = 0; flaw == NULL && index < sections; index++) {
		if (count - at < SECTION_BYTES) {
			flaw = "a section cut short";
			break;
		}
		const unsigned char *const section = bytes + at;
		const uint32_t table = get32(section);
		const uint64_t id_count = get64(section + AT_ID_COUNT);
		at += SECTION_BYTES;
		flaw = table >= table_count ? "a table index out of range" :
		       tables[table].ids != NULL ? "a table listed twice" :
		       get32(section + AT_RESERVED) != 0 ?
						   "a nonzero reserved word" :
		       get64(section + AT_EXTENT_DIGEST) !=
				       expected[table].extent_digest ?
						   "made for another table" :
		       get64(section + AT_BLOCK_COUNT) !=
				       expected[table].block_count ?
						   "a table of another size" :
		       id_count > (count - at) / ID_BYTES ? "ids cut short" :
							    NULL;
		if (flaw != NULL)
			break;
		blocks += id_count;
		if (blocks > budget_bytes / PACK_BLOCK_BYTES) {
			flaw = "more blocks than the budget";
			break;
		}
		tables[table].ids =
			malloc((size_t)(id_count != 0 ? id_count : 1u) *
			       sizeof(uint64_t));
		if (tables[table].ids == NULL)
			return STATUS_OUT_OF_MEMORY;
		tables[table].count = id_count;
		for (uint64_t id = 0; flaw == NULL && id < id_count; id++) {
			const uint64_t value =
				get64(bytes + at + id * ID_BYTES);
			flaw = value >= expected[table].block_count ?
				       "an id out of range" :
			       id != 0 && value <= tables[table].ids[id - 1] ?
				       "ids not strictly increasing" :
				       NULL;
			tables[table].ids[id] = value;
		}
		at += id_count * ID_BYTES;
	}
	if (flaw == NULL && at != count)
		flaw = "bytes after the last section";
	if (flaw != NULL) {
		(void)snprintf(reason, ENGRAM_HOT_REASON_BYTES, "%s", flaw);
		return STATUS_INVALID_ARGUMENT;
	}
	*blocks_out = blocks;
	return STATUS_OK;
}

/* The file whole, its digest, then parse. */
MUST_CHECK static result_status
read_ids(const char *path, const char *sha256_hex, uint64_t budget_bytes,
	 const struct pack_header *header,
	 const struct engram_hot_table *expected, uint32_t table_count,
	 struct hot_table *tables, uint64_t *blocks_out, char *reason)
{
	if (path == NULL || header == NULL || expected == NULL ||
	    table_count == 0 || table_count > ENGRAM_MAXIMUM_TABLES ||
	    reason == NULL ||
	    budget_bytes > (uint64_t)ENGRAM_HOT_BUDGET_MOST_MIB * MIB_BYTES)
		return STATUS_INVALID_ARGUMENT;
	reason[0] = '\0';
	const uint64_t most_bytes =
		HEADER_BYTES + (uint64_t)ENGRAM_MAXIMUM_TABLES * SECTION_BYTES +
		budget_bytes / PACK_BLOCK_BYTES * ID_BYTES;
	unsigned char *bytes = NULL;
	size_t count = 0;
	result_status status =
		read_file(path, most_bytes, &bytes, &count, reason);
	if (status == STATUS_OK && !digest_is(bytes, count, sha256_hex)) {
		(void)snprintf(reason, ENGRAM_HOT_REASON_BYTES,
			       "%s does not have the recorded SHA-256", path);
		status = STATUS_INVALID_ARGUMENT;
	}
	if (status == STATUS_OK)
		status = parse(bytes, count, budget_bytes, header, expected,
			       table_count, tables, blocks_out, reason);
	free(bytes);
	return status;
}

result_status engram_hot_read(const char *path, const char *sha256_hex,
			      uint64_t budget_bytes,
			      const struct pack_header *header,
			      const struct engram_hot_table *expected,
			      uint32_t table_count,
			      uint64_t *ids_out[ENGRAM_MAXIMUM_TABLES],
			      uint64_t counts_out[ENGRAM_MAXIMUM_TABLES],
			      char reason_out[ENGRAM_HOT_REASON_BYTES])
{
	if (ids_out == NULL || counts_out == NULL)
		return STATUS_INVALID_ARGUMENT;
	struct hot_table tables[ENGRAM_MAXIMUM_TABLES] = { 0 };
	uint64_t blocks = 0;
	const result_status status = read_ids(path, sha256_hex, budget_bytes,
					      header, expected, table_count,
					      tables, &blocks, reason_out);
	for (uint32_t table = 0; table < ENGRAM_MAXIMUM_TABLES; table++) {
		if (status != STATUS_OK)
			free(tables[table].ids);
		ids_out[table] = status == STATUS_OK ? tables[table].ids : NULL;
		counts_out[table] = status == STATUS_OK ? tables[table].count :
							  0;
	}
	return status;
}

struct load_part {
	struct engram_hot *hot;
	struct engram_disk *disk;
	uint32_t part;
	result_status status;
	pthread_t thread;
	bool started;
};

/* Every LOAD_THREADS-th hot block from part on, through the store's verified read. */
static void *load_part(void *argument)
{
	struct load_part *const self = argument;
	const struct engram_hot *const hot = self->hot;
	uint64_t index = 0;
	for (uint32_t table = 0; table < ENGRAM_MAXIMUM_TABLES; table++)
		for (uint64_t id = 0; id < hot->tables[table].count;
		     id++, index++)
			if (index % LOAD_THREADS == self->part &&
			    self->status == STATUS_OK)
				self->status = engram_disk_block_fetch(
					self->disk, table,
					hot->tables[table].ids[id],
					hot->arena + index * PACK_BLOCK_BYTES);
	return NULL;
}

result_status engram_hot_open(const char *path, const char *sha256_hex,
			      uint64_t budget_bytes,
			      const struct pack_record *pack,
			      struct engram_disk *disk,
			      struct engram_hot **hot_out,
			      char reason_out[ENGRAM_HOT_REASON_BYTES])
{
	const struct engram_description *const description =
		disk != NULL ? engram_disk_description(disk) : NULL;
	if (pack == NULL || description == NULL || hot_out == NULL ||
	    reason_out == NULL)
		return STATUS_INVALID_ARGUMENT;
	const uint32_t table_count = description->table_count;
	struct engram_hot_table expected[ENGRAM_MAXIMUM_TABLES] = { 0 };
	for (uint32_t table = 0; table < table_count; table++)
		expected[table] = (struct engram_hot_table){
			.table = table,
			.extent_digest = engram_disk_extent_digest(disk, table),
			.block_count = engram_disk_block_count(disk, table)
		};
	struct engram_hot *const hot = calloc(1, sizeof(*hot));
	if (hot == NULL)
		return STATUS_OUT_OF_MEMORY;
	result_status status = read_ids(path, sha256_hex, budget_bytes,
					pack_header_of(pack), expected,
					table_count, hot->tables, &hot->blocks,
					reason_out);
	if (status == STATUS_OK && hot->blocks != 0) {
		hot->arena_bytes = hot->blocks * PACK_BLOCK_BYTES;
		hot->arena = aligned_alloc(PACK_BLOCK_BYTES,
					   (size_t)hot->arena_bytes);
		if (hot->arena == NULL)
			status = STATUS_OUT_OF_MEMORY;
		else
			(void)madvise(hot->arena, (size_t)hot->arena_bytes,
				      MADV_HUGEPAGE);
	}
	uint64_t first = 0;
	for (uint32_t table = 0;
	     status == STATUS_OK && table < ENGRAM_MAXIMUM_TABLES; table++) {
		hot->tables[table].first = first;
		first += hot->tables[table].count;
	}
	if (status == STATUS_OK && hot->blocks != 0) {
		struct load_part parts[LOAD_THREADS];
		for (uint32_t part = 0; part < LOAD_THREADS; part++) {
			parts[part] = (struct load_part){ .hot = hot,
							  .disk = disk,
							  .part = part,
							  .status = STATUS_OK };
			parts[part].started =
				part != 0 &&
				pthread_create(&parts[part].thread, NULL,
					       load_part, &parts[part]) == 0;
		}
		for (uint32_t part = 0; part < LOAD_THREADS; part++)
			if (!parts[part].started)
				(void)load_part(
					&parts[part]); /* part 0, and any part whose thread did not start */
		for (uint32_t part = 0; part < LOAD_THREADS; part++) {
			if (parts[part].started)
				(void)pthread_join(parts[part].thread, NULL);
			status = status == STATUS_OK ? parts[part].status :
						       status;
		}
		if (status != STATUS_OK)
			(void)snprintf(reason_out, ENGRAM_HOT_REASON_BYTES,
				       "a hot block no copy delivers verified");
	}
	if (status != STATUS_OK) {
		engram_hot_close(hot);
		return status;
	}
	*hot_out = hot;
	return STATUS_OK;
}

void engram_hot_close(struct engram_hot *hot)
{
	if (hot == NULL)
		return;
	for (uint32_t table = 0; table < ENGRAM_MAXIMUM_TABLES; table++)
		free(hot->tables[table].ids);
	free(hot->arena);
	free(hot);
}

const unsigned char *engram_hot_block(const struct engram_hot *hot,
				      uint32_t table, uint64_t block)
{
	if (hot == NULL || table >= ENGRAM_MAXIMUM_TABLES)
		return NULL;
	const struct hot_table *const own = &hot->tables[table];
	uint64_t low = 0, high = own->count;
	while (low < high) {
		const uint64_t middle = low + (high - low) / HALF;
		if (own->ids[middle] < block)
			low = middle + 1u;
		else
			high = middle;
	}
	return low < own->count && own->ids[low] == block ?
		       hot->arena + (own->first + low) * PACK_BLOCK_BYTES :
		       NULL;
}

uint64_t engram_hot_block_count(const struct engram_hot *hot)
{
	return hot != NULL ? hot->blocks : 0;
}
