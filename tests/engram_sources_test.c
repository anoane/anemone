/* SPDX-License-Identifier: AGPL-3.0-or-later */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "anemone/engram_disk.h"
#include "storage_batch.h"
#define SOURCES 2u
#define TEST_SECONDS 60u
static unsigned failures;
static void expect(bool passed, const char *message)
{
	if (!passed) {
		failures++;
		(void)fprintf(stderr, "FAILED: %s\n", message);
	}
}
int main(int argument_count, char **arguments)
{
	(void)alarm(TEST_SECONDS);
	expect(!engram_hashing_equal(NULL, NULL),
	       "missing hashing constants do not compare equal");
	expect(engram_disk_sources_open(NULL, 0, NULL, NULL, NULL) ==
		       STATUS_INVALID_ARGUMENT,
	       "missing source-map arguments refused");
	if (argument_count != SOURCES + 1u) {
		(void)puts(
			"engram_sources_test: existing-pack mode requires two compatible table sources");
		return failures ? EXIT_FAILURE : EXIT_SUCCESS;
	}
	const char *paths[SOURCES] = { arguments[1], arguments[SOURCES] };
	struct storage_reader *original = NULL, *mapped = NULL;
	struct pack_record *pack = NULL;
	struct engram_record *engram = NULL;
	struct storage_batch *batch = NULL;
	result_status status = storage_direct.reader_open(paths, 1, &original);
	if (status == STATUS_OK)
		status = pack_open_entry(&storage_direct, original, &pack);
	if (status == STATUS_OK)
		status = engram_open(&storage_direct, original, pack, &engram);
	if (status == STATUS_OK)
		status = engram_disk_sources_open(paths, SOURCES, pack, engram,
						  &mapped);
	expect(status == STATUS_OK,
	       "different pack layouts bind only identical immutable Engram tables and hashing constants");
	if (status != STATUS_OK)
		goto finish;
	const struct engram_description *description =
		engram_description_of(engram);
	if (description == NULL) {
		expect(false, "description exists");
		goto finish;
	}
	status = storage_batch_open(mapped, &batch);
	expect(status == STATUS_OK,
	       "mapped sources prepare registered-file rings");
	if (status != STATUS_OK)
		goto finish;
	_Alignas(PACK_BLOCK_BYTES) unsigned char
		actual[STORAGE_BATCH_DEPTH * PACK_BLOCK_BYTES],
		expected[PACK_BLOCK_BYTES], scalar[PACK_BLOCK_BYTES];
	uint32_t copies[STORAGE_BATCH_DEPTH];
	uint64_t offsets[STORAGE_BATCH_DEPTH];
	result_status results[STORAGE_BATCH_DEPTH];
	for (uint32_t table = 0; table < description->table_count; table++) {
		const struct pack_extent *extent = &pack_extents_of(
			pack)[pack_units_of(
				      pack)[description->table_unit_index[table]]
				      .first_extent_index];
		for (uint32_t index = 0; index < STORAGE_BATCH_DEPTH; index++) {
			copies[index] = index % SOURCES;
			offsets[index] = extent->byte_offset +
					 (index % SOURCES == 0 ?
						  index :
						  (extent->byte_count /
						   PACK_BLOCK_BYTES) -
							  1u - index) *
						 PACK_BLOCK_BYTES;
		}
		expect(storage_batch_read(batch, copies, offsets,
					  STORAGE_BATCH_DEPTH, actual,
					  results) == STATUS_OK,
		       "interleaved batch translates each copy's own extent offset");
		for (uint32_t index = 0; index < STORAGE_BATCH_DEPTH; index++) {
			expect(results[index] == STATUS_OK &&
				       storage_direct.reader_read(
					       original, 0, offsets[index],
					       expected,
					       PACK_BLOCK_BYTES) == STATUS_OK &&
				       storage_direct.reader_read(
					       mapped, copies[index],
					       offsets[index], scalar,
					       PACK_BLOCK_BYTES) == STATUS_OK &&
				       memcmp(expected, scalar,
					      PACK_BLOCK_BYTES) == 0 &&
				       memcmp(expected,
					      actual + (size_t)index *
							       PACK_BLOCK_BYTES,
					      PACK_BLOCK_BYTES) == 0,
			       "scalar and batch source mappings return every original block byte at both ends of a "
			       "table");
		}
	}
finish:
	storage_batch_close(batch);
	storage_direct.reader_close(mapped);
	engram_close(engram);
	pack_close(pack);
	storage_direct.reader_close(original);
	if (failures)
		return EXIT_FAILURE;
	(void)puts(
		"engram_sources_test passed: hashing/table identity, cross-layout scalar/batched reads and boundaries");
	return EXIT_SUCCESS;
}
