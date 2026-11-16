/* SPDX-License-Identifier: AGPL-3.0-or-later */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "anemone/storage_seam.h"
#include "storage_batch.h"
#define TEST_ROUNDS 8u
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
	expect(storage_batch_open(NULL, NULL) == STATUS_INVALID_ARGUMENT,
	       "missing batch inputs refused");
	if (argument_count <= 1) {
		(void)puts(
			"storage_batch_test: real-file mode requires existing flat pack copies; no I/O executed");
		return failures ? EXIT_FAILURE : EXIT_SUCCESS;
	}
	struct storage_reader *reader = NULL;
	struct storage_batch *batch = NULL;
	const uint32_t copy_count = (uint32_t)argument_count - 1u;
	result_status status = storage_direct.reader_open(
		(const char *const *)(arguments + 1), copy_count, &reader);
	if (status == STATUS_OK)
		status = storage_batch_open(reader, &batch);
	expect(status == STATUS_OK,
	       "prepared registered-file rings open on existing files with verified direct alignment");
	if (status != STATUS_OK)
		goto finish;
	_Alignas(PACK_BLOCK_BYTES) unsigned char
		actual[STORAGE_BATCH_DEPTH * PACK_BLOCK_BYTES],
		reference[PACK_BLOCK_BYTES];
	uint32_t copies[STORAGE_BATCH_DEPTH];
	uint64_t offsets[STORAGE_BATCH_DEPTH];
	result_status results[STORAGE_BATCH_DEPTH];
	struct storage_description description;
	expect(storage_direct.reader_describe(reader, &description) ==
		       STATUS_OK,
	       "reader description");
	for (uint32_t round = 0; round < TEST_ROUNDS; round++) {
		for (uint32_t index = 0; index < STORAGE_BATCH_DEPTH; index++) {
			copies[index] = index % copy_count;
			offsets[index] = (uint64_t)(STORAGE_BATCH_DEPTH -
						    index + round) *
					 PACK_BLOCK_BYTES;
		}
		expect(storage_batch_read(batch, copies, offsets,
					  STORAGE_BATCH_DEPTH, actual,
					  results) == STATUS_OK,
		       "complete bounded batch");
		for (uint32_t index = 0; index < STORAGE_BATCH_DEPTH; index++) {
			expect(results[index] == STATUS_OK &&
				       storage_direct.reader_read(
					       reader, 0, offsets[index],
					       reference,
					       PACK_BLOCK_BYTES) == STATUS_OK &&
				       memcmp(actual + (size_t)index *
							       PACK_BLOCK_BYTES,
					      reference, PACK_BLOCK_BYTES) == 0,
			       "every completion matches independent source bytes in caller order");
		}
	}
	offsets[0] =
		description
			.byte_count; /* EOF is a short read, never successful data. */
	expect(storage_batch_read(batch, copies, offsets, STORAGE_BATCH_DEPTH,
				  actual, results) == STATUS_OK &&
		       results[0] == STATUS_INPUT_OUTPUT_ERROR &&
		       results[1] == STATUS_OK,
	       "short-read status belongs to its row");
	expect(storage_batch_read(batch, copies, offsets,
				  STORAGE_BATCH_DEPTH + 1u, actual,
				  results) == STATUS_INVALID_ARGUMENT,
	       "oversized batch refused");
	offsets[0] = 1;
	expect(storage_batch_read(batch, copies, offsets, 1, actual, results) ==
		       STATUS_INVALID_ARGUMENT,
	       "unaligned offset refused");
finish:
	storage_batch_close(batch);
	storage_direct.reader_close(reader);
	if (failures)
		return EXIT_FAILURE;
	(void)puts(
		"storage_batch_test passed: registered copies, bounded batches, exact bytes, EOF and alignment");
	return EXIT_SUCCESS;
}
