/* SPDX-License-Identifier: AGPL-3.0-or-later */
#include <fcntl.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "anemone/storage_exl3.h"
#include "anemone/pack.h"

#define MAP_MAGIC_BYTES 8u
#define MAP_RESERVED_BYTES 40u
#define SOURCES 2u
#define SOURCE_FILE 1u
#define VIRTUAL_BLOCKS 3ull
#define BLOCK EXL3_MAP_BLOCK_BYTES
#define INPUT_PATTERN 0x5Au
#define FILE_MODE 0600

struct test_header {
	unsigned char magic[MAP_MAGIC_BYTES];
	uint64_t bytes;
	uint32_t sources;
	uint32_t segments;
	unsigned char reserved[MAP_RESERVED_BYTES];
};
struct test_segment {
	uint64_t offset, count, source_offset;
	uint32_t source, reserved;
};

static bool written(int descriptor, const void *bytes, size_t count,
		    off_t offset)
{
	return pwrite(descriptor, bytes, count, offset) == (ssize_t)count;
}

/* Turbo load's flat copy (copy 1: the map's logical bytes in a plain file, read directly) reads as copy 0 does;
 * a flat copy of another size is refused. buffer holds VIRTUAL_BLOCKS blocks. */
static bool flat_copy_reads(const char *manifest_path, unsigned char *buffer)
{
	char flat_path[] = "/var/tmp/anemone-exl3-flat-XXXXXX";
	const int flat = mkstemp(flat_path);
	struct storage_reader *reader = NULL;
	struct storage_description description = { 0 };
	const char *const copy_paths[] = { manifest_path, flat_path };
	bool passed =
		flat >= 0 &&
		storage_exl3.reader_open(copy_paths, 1, &reader) == STATUS_OK &&
		storage_exl3.reader_read(reader, 0, 0, buffer,
					 VIRTUAL_BLOCKS * BLOCK) == STATUS_OK &&
		written(flat, buffer, VIRTUAL_BLOCKS * BLOCK, 0);
	storage_exl3.reader_close(reader);
	reader = NULL;
	passed = passed &&
		 storage_exl3.reader_open(copy_paths, 2, &reader) ==
			 STATUS_OK &&
		 storage_exl3.reader_describe(reader, &description) ==
			 STATUS_OK &&
		 description.copy_count == 2;
	for (uint32_t block = 0; passed && block < VIRTUAL_BLOCKS; block++) {
		unsigned char *const copied = buffer + VIRTUAL_BLOCKS * BLOCK;
		passed = storage_exl3.reader_read(reader, 0, block * BLOCK,
						  buffer, BLOCK) == STATUS_OK &&
			 storage_exl3.reader_read(reader, 1, block * BLOCK,
						  copied, BLOCK) == STATUS_OK &&
			 memcmp(buffer, copied, BLOCK) == 0;
	}
	storage_exl3.reader_close(reader);
	reader = NULL;
	passed = passed && ftruncate(flat, BLOCK) == 0 &&
		 storage_exl3.reader_open(copy_paths, 2, &reader) ==
			 STATUS_CORRUPT_PACK &&
		 reader == NULL;
	if (flat >= 0) {
		(void)close(flat);
		(void)unlink(flat_path);
	}
	return passed;
}

/* Every read mode (storage_exl3_read_mode: buffered, buffered then dropped from the page cache, direct where
 * aligned) delivers the same bytes. */
static bool read_modes_agree(const char *manifest_path, unsigned char *buffer)
{
	const char *const copy_paths[] = { manifest_path };
	unsigned char *const first = buffer + VIRTUAL_BLOCKS * BLOCK;
	bool passed = true;
	for (uint32_t mode = 0; mode < EXL3_READ_MODES; mode++) {
		struct storage_reader *reader = NULL;
		storage_exl3_read_mode((enum exl3_read_mode)mode);
		passed = passed &&
			 storage_exl3.reader_open(copy_paths, 1, &reader) ==
				 STATUS_OK &&
			 storage_exl3.reader_read(reader, 0, 0,
						  mode == 0 ? first : buffer,
						  VIRTUAL_BLOCKS * BLOCK) ==
				 STATUS_OK &&
			 memcmp(buffer, first, VIRTUAL_BLOCKS * BLOCK) == 0;
		storage_exl3.reader_close(reader);
	}
	return passed;
}

int main(int argument_count, char **arguments)
{
	if (storage_for_path("weights.exl3map") != &storage_exl3 ||
	    storage_for_path("weights.anemone") != &storage_direct ||
	    storage_for_path("map") != &storage_direct ||
	    storage_for_path(NULL) != &storage_direct) {
		return 1;
	}
	if (argument_count > 1) {
		struct storage_reader *reader = NULL;
		struct pack_record *pack = NULL;
		const char *paths[] = { arguments[1] };
		result_status status =
			storage_exl3.reader_open(paths, 1, &reader);
		status = status == STATUS_OK ?
				 pack_open_entry(&storage_exl3, reader, &pack) :
				 status;
		(void)printf("real virtual pack open: %s\n",
			     status_text(status));
		pack_close(pack);
		storage_exl3.reader_close(reader);
		return status == STATUS_OK ? 0 : 1;
	}
	/* /var/tmp: a disk filesystem, where the sources' direct reads open */
	char manifest_path[] = "/var/tmp/anemone-exl3-map-XXXXXX";
	char source_path[] = "/var/tmp/anemone-exl3-source-XXXXXX";
	const int manifest = mkstemp(manifest_path),
		  source = mkstemp(source_path);
	void *buffer = NULL;
	bool passed = manifest >= 0 && source >= 0 &&
		      posix_memalign(&buffer, BLOCK,
				     2u * VIRTUAL_BLOCKS * BLOCK) == 0;
	struct test_header header = { { 'A', 'N', 'E', 'X', 'L', '3', 0, 0 },
				      VIRTUAL_BLOCKS * BLOCK,
				      SOURCES,
				      1,
				      { 0 } };
	char paths[SOURCES][EXL3_MAP_PATH_BYTES] = { { 0 } };
	const size_t source_length = strlen(source_path);
	memcpy(paths[SOURCE_FILE], source_path, source_length + 1u);
	struct test_segment segment = { BLOCK, BLOCK, 0, SOURCE_FILE, 0 };
	if (passed) {
		memset(buffer, INPUT_PATTERN, BLOCK);
		passed = written(source, buffer, BLOCK, 0) &&
			 written(manifest, &header, sizeof(header), 0) &&
			 written(manifest, paths, sizeof(paths),
				 (off_t)sizeof(header)) &&
			 written(manifest, &segment, sizeof(segment),
				 (off_t)(sizeof(header) + sizeof(paths)));
	}
	const char *copy_paths[] = { manifest_path };
	struct storage_reader *reader = NULL;
	if (passed) {
		passed = storage_exl3.reader_open(copy_paths, 1, &reader) ==
			 STATUS_OK;
	}
	if (passed) {
		struct storage_description description;
		passed = storage_exl3.reader_describe(reader, &description) ==
				 STATUS_OK &&
			 description.byte_count == VIRTUAL_BLOCKS * BLOCK &&
			 storage_exl3.reader_read(reader, 0, 0, buffer,
						  VIRTUAL_BLOCKS * BLOCK) ==
				 STATUS_OK;
		const unsigned char *const bytes = buffer;
		for (uint32_t byte = 0; byte < VIRTUAL_BLOCKS * BLOCK; byte++) {
			const unsigned char expected =
				byte >= BLOCK && byte < BLOCK * SOURCES ?
					INPUT_PATTERN :
					0;
			passed = bytes[byte] == expected && passed;
		}
		passed =
			storage_exl3.reader_read(reader, 1, 0, buffer, BLOCK) ==
				STATUS_INVALID_ARGUMENT &&
			storage_exl3.reader_read(reader, 0,
						 VIRTUAL_BLOCKS * BLOCK, buffer,
						 BLOCK) ==
				STATUS_INVALID_ARGUMENT &&
			passed;
		struct storage_reader *untouched = reader;
		segment.source_offset = BLOCK;
		passed = written(manifest, &segment, sizeof(segment),
				 (off_t)(sizeof(header) + sizeof(paths))) &&
			 storage_exl3.reader_open(copy_paths, 1, &untouched) ==
				 STATUS_CORRUPT_PACK &&
			 untouched == reader && passed;
		/* Restore the valid segment so the independent reserved-header refusal has its own cause. */
		segment.source_offset = 0;
		passed = written(manifest, &segment, sizeof(segment),
				 (off_t)(sizeof(header) + sizeof(paths))) &&
			 passed;
		passed = flat_copy_reads(manifest_path, buffer) && passed;
		passed = read_modes_agree(manifest_path, buffer) && passed;
		header.reserved[0] = 1;
		passed = written(manifest, &header, sizeof(header), 0) &&
			 storage_exl3.reader_open(copy_paths, 1, &untouched) ==
				 STATUS_CORRUPT_PACK &&
			 untouched == reader && passed;
	}
	storage_exl3.reader_close(reader);
	storage_exl3.reader_close(NULL);
	if (manifest >= 0) {
		(void)close(manifest);
		(void)unlink(manifest_path);
	}
	if (source >= 0) {
		(void)close(source);
		(void)unlink(source_path);
	}
	free(buffer);
	(void)printf(
		"EXL3 virtual storage %s: mapped data, zero holes, bounds and output preservation\n",
		passed ? "PASS" : "FAIL");
	return passed ? 0 : 1;
}
