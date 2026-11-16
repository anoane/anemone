/* SPDX-License-Identifier: AGPL-3.0-or-later */
#include <stdbool.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <unistd.h>
#include "anemone/engram_disk.h"
#include "engram_pack.h"
#include "memory_storage.h"
#include "anemone/hooks.h"
#include "log_expectation.h"

#define TEST_THREADS 8u
#define TEST_REPEATS 64u
#define HALF_RAM 50u
#define TEST_CACHE_BYTES (16u * PACK_BLOCK_BYTES)
#define TEST_TIMEOUT_SECONDS 60u
#define CAPTURE_BYTES 4096u
#define FAIL_CORRUPT_COPY 2u
#define FAIL_ALL_DEVICES 3u
#define FAIL_ALL_CORRUPT 4u
#define WEIGHTED_COPIES 2u
#define FAST_DEVICE_WEIGHT 3u
#define TUNE_MODES 2u /* the tune's cached mode, then its uncached one */
static unsigned failures;
static void expect(bool passed, const char *message)
{
	if (!passed) {
		failures++;
		(void)fprintf(stderr, "FAILED: %s\n", message);
	}
}
struct reader_case {
	const struct engram_record *engram;
	bool passed;
};
static void *read_rows(void *context)
{
	struct reader_case *const reader = context;
	reader->passed = true;
	for (uint32_t repeat = 0; repeat < TEST_REPEATS; repeat++) {
		const struct engram_description *const description =
			engram_description_of(reader->engram);
		if (description == NULL) {
			reader->passed = false;
			return NULL;
		}
		for (uint32_t table = 0; table < description->table_count;
		     table++) {
			for (uint64_t row = 0;
			     row < description->row_count[table]; row++) {
				unsigned char bytes[ENGRAM_ROW_BYTES + 1];
				bytes[ENGRAM_ROW_BYTES] = 1;
				if (engram_gather_entry(reader->engram, table,
							&row, 1,
							bytes) != STATUS_OK ||
				    bytes[ENGRAM_ROW_BYTES] != 1)
					reader->passed = false;
				for (uint64_t byte = 0; byte < ENGRAM_ROW_BYTES;
				     byte++) {
					const uint64_t offset =
						row / ENGRAM_ROWS_PER_BLOCK *
							PACK_BLOCK_BYTES +
						row % ENGRAM_ROWS_PER_BLOCK *
							ENGRAM_ROW_BYTES +
						byte;
					if (bytes[byte] !=
					    (unsigned char)(offset *
								    TEST_ENGRAM_PACK_PATTERN_MULTIPLIER +
							    table))
						reader->passed = false;
				}
			}
		}
	}
	return NULL;
}
/* The tune's uncached mode: a repeated row reads its copies every time, with the same bytes; the cached mode serves
 * the repeat from the bounded cache. */
static void uncached_tune_reads(struct engram_record *engram,
				struct engram_disk *disk,
				const uint32_t *weights)
{
	const uint64_t row = 0;
	unsigned char first[ENGRAM_ROW_BYTES], again[ENGRAM_ROW_BYTES];
	uint64_t reads[TUNE_MODES + 1u];
	for (uint32_t mode = 0; mode < TUNE_MODES; mode++) {
		const bool cached = mode == 0;
		expect(engram_disk_tune_io(disk, weights, WEIGHTED_COPIES, 0,
					   cached) == STATUS_OK &&
			       engram_gather_entry(engram, 1, &row, 1, first) ==
				       STATUS_OK,
		       "a retired tune switches the bounded cache");
		reads[mode] = test_memory_storage_read_count(0) +
			      test_memory_storage_read_count(1);
		expect(engram_gather_entry(engram, 1, &row, 1, again) ==
				       STATUS_OK &&
			       memcmp(first, again, sizeof(first)) == 0,
		       "a repeated row keeps its bytes cached and uncached");
		reads[TUNE_MODES] = test_memory_storage_read_count(0) +
				    test_memory_storage_read_count(1);
		expect(cached ? reads[TUNE_MODES] == reads[mode] :
				reads[TUNE_MODES] > reads[mode],
		       cached ?
			       "the cached tune serves a repeated row from the cache" :
			       "the uncached tune reads a repeated row from its copies");
	}
	expect(engram_disk_tune_io(disk, weights, WEIGHTED_COPIES, 0, true) ==
		       STATUS_OK,
	       "the tune restores the cache");
}

static void scenario(uint32_t ram, uint64_t cache, uint32_t turbo,
		     uint32_t failure)
{
	unsigned char *bytes = NULL;
	struct engram_record *engram = NULL;
	struct storage_reader *reader = NULL;
	struct pack_record *pack = NULL;
	struct engram_disk *disk = NULL;
	const char *const copies[] = { "synthetic0", "synthetic1" };
	if (test_engram_pack_open(&bytes, &engram) != STATUS_OK ||
	    test_memory_storage.reader_open(copies,
					    sizeof(copies) / sizeof(copies[0]),
					    &reader) != STATUS_OK ||
	    pack_open_entry(&test_memory_storage, reader, &pack) != STATUS_OK) {
		expect(false, "synthetic pack opens");
		goto finish;
	}
	const struct engram_description *const description =
		engram_description_of(engram);
	const result_status opened =
		engram_disk_open(description, pack, &test_memory_storage,
				 reader, ram, cache, turbo, 0, &disk);
	expect(opened == STATUS_OK,
	       "disk placement verifies tables with bounded workspace");
	if (opened != STATUS_OK)
		goto finish;
	const uint32_t invalid_weights[] = { 0, 0 };
	const uint32_t measured_weights[] = { FAST_DEVICE_WEIGHT, 1 };
	const uint32_t zero_share[] = { FAST_DEVICE_WEIGHT, 0 };
	expect(engram_disk_weights(disk, invalid_weights, WEIGHTED_COPIES) ==
		       STATUS_INVALID_ARGUMENT,
	       "all-zero device shares refused before attachment");
	if (turbo != 0)
		expect(engram_disk_weights(disk, zero_share, WEIGHTED_COPIES) ==
			       STATUS_OK,
		       "one slow device may receive zero share while the remaining set stays usable");
	if (turbo != 0)
		expect(engram_disk_weights(disk, measured_weights,
					   WEIGHTED_COPIES) == STATUS_OK,
		       "measured unequal shares retain exact bytes and device failover");
	if (turbo != 0) {
		expect(engram_disk_tune_io(disk, measured_weights,
					   WEIGHTED_COPIES, 1,
					   true) == STATUS_UNSUPPORTED,
		       "CPU tuning never treats a non-direct storage reader as a native io_uring reader");
		expect(engram_disk_tune_io(disk, measured_weights,
					   WEIGHTED_COPIES, 0,
					   true) == STATUS_OK,
		       "retired tuning reuses validated digests and resets cache/shares");
	}
	struct engram_disk *cancelled = disk;
	const volatile sig_atomic_t stop = 1;
	expect(engram_disk_open_until(description, pack, &test_memory_storage,
				      reader, ram, cache, turbo, 0, 1, &stop,
				      &cancelled) == STATUS_TIMED_OUT &&
		       cancelled == disk,
	       "explicit preparation cancellation leaves output untouched and ordinary preparation remains available");
	expect(engram_attach_disk(engram, disk) == STATUS_OK,
	       "table store attaches");
	const uint8_t *const external = engram_disk_external_extents(disk);
	for (uint32_t extent = 0; extent < pack_header_of(pack)->extent_count;
	     extent++)
		expect(external[extent] ==
			       (extent != 0 && extent <=
							TEST_ENGRAM_PACK_TABLES ?
					1u :
					0u),
		       "only model table extents are external");
	const uint64_t last_row = description->row_count[1] - 1u;
	const uint64_t extent_offset =
		pack_extents_of(
			pack)[pack_units_of(
				      pack)[description->table_unit_index[1]]
				      .first_extent_index]
			.byte_offset;
	if (failure == 1)
		test_memory_storage_fail_reads(
			0, test_memory_storage_read_count(0) + 1u);
	if (failure == FAIL_CORRUPT_COPY)
		test_memory_storage_flip_bit(0,
					     extent_offset + PACK_BLOCK_BYTES);
	if (failure == FAIL_ALL_DEVICES) {
		test_memory_storage_fail_reads(
			0, test_memory_storage_read_count(0) + 1u);
		test_memory_storage_fail_reads(
			1, test_memory_storage_read_count(1) + 1u);
		unsigned char output[ENGRAM_ROW_BYTES];
		expect(engram_gather_entry(engram, 1, &last_row, 1, output) ==
			       STATUS_INPUT_OUTPUT_ERROR,
		       "all devices lost returns I/O failure");
		goto finish;
	}
	if (failure == FAIL_ALL_CORRUPT) {
		test_memory_storage_flip_bit(0,
					     extent_offset + PACK_BLOCK_BYTES);
		test_memory_storage_flip_bit(1,
					     extent_offset + PACK_BLOCK_BYTES);
		unsigned char output[ENGRAM_ROW_BYTES];
		expect(engram_gather_entry(engram, 1, &last_row, 1, output) ==
			       STATUS_CORRUPT_PACK,
		       "all replicas corrupt returns corruption");
		goto finish;
	}
	uint64_t ids[ENGRAM_MAXIMUM_ROWS_PER_TOKEN + 1];
	unsigned char
		output[(ENGRAM_MAXIMUM_ROWS_PER_TOKEN + 1) * ENGRAM_ROW_BYTES];
	for (uint32_t index = 0; index < ENGRAM_MAXIMUM_ROWS_PER_TOKEN; index++)
		ids[index] = index % description->row_count[1];
	expect(engram_gather_entry(engram, 1, ids,
				   ENGRAM_MAXIMUM_ROWS_PER_TOKEN,
				   output) == STATUS_OK,
	       "duplicate/out-of-order rows across block edges gather");
	expect(engram_gather_entry(engram, 1, ids,
				   ENGRAM_MAXIMUM_ROWS_PER_TOKEN + 1,
				   output) == STATUS_INVALID_ARGUMENT,
	       "disk batch count bounded");
	ids[0] = UINT64_MAX;
	expect(engram_gather_entry(engram, 1, ids, 1, output) ==
		       STATUS_INVALID_ARGUMENT,
	       "invalid row refused before any read");
	expect(engram_gather_entry(engram, 1, ids, 0, output) == STATUS_OK,
	       "empty gather succeeds");
	pthread_t threads[TEST_THREADS];
	struct reader_case cases[TEST_THREADS];
	uint32_t started = 0;
	for (; started < TEST_THREADS; started++) {
		cases[started] = (struct reader_case){ .engram = engram };
		if (pthread_create(&threads[started], NULL, read_rows,
				   &cases[started]) != 0)
			break;
	}
	expect(started == TEST_THREADS, "all concurrent readers start");
	for (uint32_t index = 0; index < started; index++)
		expect(pthread_join(threads[index], NULL) == 0 &&
			       cases[index].passed,
		       "concurrent rows match every reference byte and guard");
	if (ram == 0 && cache != 0 && turbo != 0)
		uncached_tune_reads(engram, disk, measured_weights);
finish:
	engram_close(engram);
	engram_disk_close(disk);
	pack_close(pack);
	test_memory_storage.reader_close(reader);
	free(bytes);
}
static void test_open_refusals(void)
{
	unsigned char *bytes = NULL;
	struct engram_record *engram = NULL;
	struct storage_reader *reader = NULL;
	struct pack_record *pack = NULL;
	const char *const copies[] = { "synthetic0", "synthetic1" };
	if (test_engram_pack_open(&bytes, &engram) != STATUS_OK ||
	    test_memory_storage.reader_open(copies,
					    sizeof(copies) / sizeof(copies[0]),
					    &reader) != STATUS_OK ||
	    pack_open_entry(&test_memory_storage, reader, &pack) != STATUS_OK) {
		expect(false, "refusal fixture opens");
		goto finish;
	}
	struct engram_disk *disk = NULL;
	const struct engram_description *description =
		engram_description_of(engram);
	expect(engram_disk_open(description, pack, &test_memory_storage, reader,
				ENGRAM_RAM_PERCENT, 0, 0, 0,
				&disk) == STATUS_INVALID_ARGUMENT &&
		       disk == NULL,
	       "all RAM uses the unchanged loader and does not create a disk store");
	expect(engram_disk_open(description, pack, &test_memory_storage, reader,
				0, ENGRAM_CACHE_MAXIMUM_BYTES + 1u, 0, 0,
				&disk) == STATUS_INVALID_ARGUMENT &&
		       disk == NULL,
	       "cache budget above the bound refused");
	expect(engram_disk_open(description, pack, &test_memory_storage, reader,
				0, 0, 0, 1, &disk) == STATUS_UNSUPPORTED &&
		       disk == NULL,
	       "io_uring refuses unsupported storage without publishing a partial store");
	test_memory_storage_fail_reads(0,
				       test_memory_storage_read_count(0) + 1u);
	test_memory_storage_fail_reads(1,
				       test_memory_storage_read_count(1) + 1u);
	expect(engram_disk_open(description, pack, &test_memory_storage, reader,
				0, 0, 0, 0,
				&disk) == STATUS_INPUT_OUTPUT_ERROR &&
		       disk == NULL,
	       "failed preparation preserves the output and releases its allocations");
#if CHECKING_HOOKS_ENABLED
	expect(fault_arm("engram_disk_memory", 1),
	       "arm preparation allocation failure");
	expect(engram_disk_open(description, pack, &test_memory_storage, reader,
				0, 0, 0, 0, &disk) == STATUS_OUT_OF_MEMORY &&
		       disk == NULL,
	       "allocation refusal preserves the output");
	expect(fault_disarm() == 1, "allocation failure reached");
	/* a copy's read past the reader's own limit fails only that copy's turn: the per-copy scans that follow verify
	 * the tables from the others (review finding 12; before, any TIMED_OUT ended the fallback) */
	test_memory_storage_fail_reads(0, 0);
	test_memory_storage_fail_reads(1, 0);
	expect(fault_arm("engram_disk_scan_timeout", 1),
	       "arm a scan read's timeout");
	expect(engram_disk_open(description, pack, &test_memory_storage, reader,
				0, 0, 0, 0, &disk) == STATUS_OK &&
		       disk != NULL,
	       "a read's timeout falls back to the next copy");
	expect(fault_disarm() != 0, "the scan timeout fault point is reached");
	engram_disk_close(disk);
	disk = NULL;
#endif
finish:
	engram_close(engram);
	pack_close(pack);
	test_memory_storage.reader_close(reader);
	free(bytes);
}

/* One copy (no replica): a transient read error is retried and never retires the copy for later lookups; a
 * lasting error fails that gather clearly. */
static void one_copy_recovers(void)
{
	unsigned char *bytes = NULL;
	struct engram_record *engram = NULL;
	struct storage_reader *reader = NULL;
	struct pack_record *pack = NULL;
	struct engram_disk *disk = NULL;
	const char *const copies[] = { "synthetic0" };
	if (test_engram_pack_open(&bytes, &engram) != STATUS_OK ||
	    test_memory_storage.reader_open(copies, 1, &reader) != STATUS_OK ||
	    pack_open_entry(&test_memory_storage, reader, &pack) != STATUS_OK) {
		expect(false, "one-copy synthetic pack opens");
		goto finish;
	}
	const struct engram_description *const description =
		engram_description_of(engram);
	if (description == NULL ||
	    engram_disk_open(description, pack, &test_memory_storage, reader, 0,
			     0, 0, 0, &disk) != STATUS_OK ||
	    engram_attach_disk(engram, disk) != STATUS_OK) {
		expect(false, "one-copy disk tables attach");
		goto finish;
	}
	const uint64_t rows[] = { 0, description->row_count[1] - 1u };
	unsigned char output[ENGRAM_ROW_BYTES];
	test_memory_storage_fail_one_read(0, test_memory_storage_read_count(0) +
						     1u);
	expect(engram_gather_entry(engram, 1, &rows[0], 1, output) == STATUS_OK,
	       "a transient read error on the only copy is retried");
	expect(engram_gather_entry(engram, 1, &rows[1], 1, output) == STATUS_OK,
	       "the copy stays in service for later lookups");
	test_memory_storage_fail_reads(0,
				       test_memory_storage_read_count(0) + 1u);
	/* stderr of the failing gather, captured in a temporary file: its one product line in every build */
	char captured_path[PATH_MAX], captured[CAPTURE_BYTES] = { 0 },
				      expected[CAPTURE_BYTES];
	const char *const temporary = getenv("TMPDIR");
	(void)snprintf(captured_path, sizeof(captured_path),
		       "%s/anemone-engram-stderr-XXXXXX",
		       temporary != NULL ? temporary : "/tmp");
	(void)fflush(stderr);
	const int capture = mkstemp(captured_path);
	const int saved = dup(STDERR_FILENO);
	const bool redirected = capture >= 0 && saved >= 0 &&
				dup2(capture, STDERR_FILENO) >= 0;
	const result_status lasting =
		engram_gather_entry(engram, 1, &rows[0], 1, output);
	(void)fflush(stderr);
	if (saved >= 0) {
		(void)dup2(saved, STDERR_FILENO);
		(void)close(saved);
	}
	const ssize_t captured_bytes =
		capture >= 0 ? test_log_capture_read(capture, captured,
						     sizeof(captured)) :
			       -1;
	if (capture >= 0) {
		(void)close(capture);
		(void)unlink(captured_path);
	}
	(void)snprintf(expected, sizeof(expected), ": %s\n",
		       status_text(STATUS_INPUT_OUTPUT_ERROR));
	const char *const line = strstr(captured, "engram: table ");
	expect(lasting == STATUS_INPUT_OUTPUT_ERROR,
	       "a lasting read error fails that gather with an I/O status");
	expect(redirected && captured_bytes > 0 && line != NULL &&
		       (line == captured || line[-1] == '\n') &&
		       strstr(line, " failed on every one of 1 copies, ") !=
			       NULL &&
		       strstr(line, expected) != NULL,
	       "the exhausted read prints its one product error line, also in release");
finish:
	engram_close(engram);
	engram_disk_close(disk);
	pack_close(pack);
	test_memory_storage.reader_close(reader);
	free(bytes);
}

/* The startup digest scan with each count of readers per copy (0: one reader, copy by copy): a copy corrupt before
 * the store opens costs only its own turn, the RAM prefix and every row read back exact; every copy corrupt refuses
 * the open. */
static void scan_threads_recover(uint32_t threads_per_copy, bool all_corrupt)
{
	unsigned char *bytes = NULL;
	struct engram_record *engram = NULL;
	struct storage_reader *reader = NULL;
	struct pack_record *pack = NULL;
	struct engram_disk *disk = NULL;
	const char *const copies[] = { "synthetic0", "synthetic1" };
	if (test_engram_pack_open(&bytes, &engram) != STATUS_OK ||
	    test_memory_storage.reader_open(copies,
					    sizeof(copies) / sizeof(copies[0]),
					    &reader) != STATUS_OK ||
	    pack_open_entry(&test_memory_storage, reader, &pack) != STATUS_OK ||
	    engram_description_of(engram) == NULL) {
		expect(false, "synthetic pack opens for the scan");
		goto finish;
	}
	const struct engram_description *const description =
		engram_description_of(engram);
	const uint64_t extent_offset =
		pack_extents_of(
			pack)[pack_units_of(
				      pack)[description->table_unit_index[1]]
				      .first_extent_index]
			.byte_offset;
	test_memory_storage_flip_bit(0, extent_offset + PACK_BLOCK_BYTES);
	if (all_corrupt)
		test_memory_storage_flip_bit(1,
					     extent_offset + PACK_BLOCK_BYTES);
	engram_disk_scan_threads(threads_per_copy);
	const result_status opened =
		engram_disk_open(description, pack, &test_memory_storage,
				 reader, HALF_RAM, 0, 1, 0, &disk);
	engram_disk_scan_threads(ENGRAM_DISK_SCAN_THREADS_DEFAULT);
	if (all_corrupt) {
		expect(opened == STATUS_CORRUPT_PACK && disk == NULL,
		       "every copy corrupt refuses the store at its scan");
		goto finish;
	}
	expect(opened == STATUS_OK,
	       "the scan recovers a table from the good copy");
	struct reader_case rows = { .engram = engram };
	if (opened == STATUS_OK &&
	    engram_attach_disk(engram, disk) == STATUS_OK)
		(void)read_rows(&rows);
	expect(rows.passed,
	       "the RAM prefix and the disk rows after the scan are exact");
finish:
	engram_close(engram);
	engram_disk_close(disk);
	pack_close(pack);
	test_memory_storage.reader_close(reader);
	free(bytes);
}

int main(void)
{
	(void)alarm(TEST_TIMEOUT_SECONDS);
	test_open_refusals();
	one_copy_recovers();
	for (uint32_t ram = 0; ram <= HALF_RAM; ram += HALF_RAM)
		for (uint32_t cache = 0; cache <= 1; cache++)
			for (uint32_t turbo = 0; turbo <= 1; turbo++)
				scenario(ram, cache ? TEST_CACHE_BYTES : 0,
					 turbo, 0);
	for (uint32_t failure = 1; failure <= FAIL_ALL_CORRUPT; failure++)
		scenario(0, 0, 0, failure);
	for (uint32_t threads = 0; threads <= ENGRAM_DISK_SCAN_THREADS_MOST;
	     threads++)
		scan_threads_recover(threads, false);
	scan_threads_recover(ENGRAM_DISK_SCAN_THREADS_DEFAULT, true);
	if (failures)
		return EXIT_FAILURE;
	(void)puts(
		"engram_disk_test passed: placements, cache, eight readers, guards, loss and corruption,"
		" parallel scans");
	return EXIT_SUCCESS;
}
