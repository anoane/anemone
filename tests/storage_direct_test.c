/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * storage_direct_test: storage_direct on real files, against every rule of storage_seam.h.
 *
 * The test makes its files in the current directory and removes them again. The file system there must take
 * direct reads: where it refuses O_DIRECT the test fails and says so, it never passes by leaving a case out. The
 * refusals that only a failing machine provokes (no memory, a copy that cannot be sized, a slow drive, a read that
 * returns an error) are provoked through fault points, in the builds that have them.
 */
#include <errno.h>
#include <inttypes.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "anemone/hooks.h"
#include "anemone/status.h"
#include "anemone/storage_seam.h"

#define TEST_TIME_LIMIT_SECONDS 300             /* the alarm ends the test here: the deadline of every wait in it */
#define FILE_BYTE_COUNT ((size_t)1 << 20u)      /* whole blocks of every alignment the test accepts */
#define MIDDLE_BYTE_OFFSET (FILE_BYTE_COUNT / 2)
#define FILE_PATH_TEMPLATE "storage_direct_test_XXXXXX"
#define CONTENT_MULTIPLIER 0x9E3779B97F4A7C15ull /* odd: no two words of the copies are the same */
#define CONTENT_SHIFT 29u
#define COPY_COUNT 3
#define RUN_BLOCK_COUNT 2
#define THREAD_COUNT 8
#define THREAD_ROUND_COUNT 64
#define THREAD_RANGE_BYTE_COUNT (FILE_BYTE_COUNT / THREAD_COUNT)
#define MOST_BLOCK_BYTE_COUNT (THREAD_RANGE_BYTE_COUNT / RUN_BLOCK_COUNT) /* so that every range is a run of blocks */
#define UNTOUCHED_VALUE 0x1234u
#define COUNT_OF(table) ((uint32_t)(sizeof(table) / sizeof((table)[0])))

enum test_file {
    TEST_FILE_FIRST_COPY,
    TEST_FILE_SECOND_COPY,
    TEST_FILE_THIRD_COPY,
    TEST_FILE_SHRINKING,                        /* a whole copy, cut to half while a reader has it open */
    TEST_FILE_SHORTER,                          /* whole blocks, but half of what a copy holds */
    TEST_FILE_OFF_BLOCK,                        /* one byte short of whole blocks */
    TEST_FILE_EMPTY,
    TEST_FILE_MISSING,                          /* made and removed at once: a name that no file has */
    TEST_FILE_COUNT
};

static const size_t file_byte_counts[TEST_FILE_COUNT] = {
    [TEST_FILE_FIRST_COPY] = FILE_BYTE_COUNT,
    [TEST_FILE_SECOND_COPY] = FILE_BYTE_COUNT,
    [TEST_FILE_THIRD_COPY] = FILE_BYTE_COUNT,
    [TEST_FILE_SHRINKING] = FILE_BYTE_COUNT,
    [TEST_FILE_SHORTER] = MIDDLE_BYTE_OFFSET,
    [TEST_FILE_OFF_BLOCK] = FILE_BYTE_COUNT - 1,
    [TEST_FILE_EMPTY] = 0,
    [TEST_FILE_MISSING] = 0,
};
static char file_paths[TEST_FILE_COUNT][sizeof(FILE_PATH_TEMPLATE)];
static bool file_exists[TEST_FILE_COUNT];
static const char *const copy_paths[COPY_COUNT] = {
    file_paths[TEST_FILE_FIRST_COPY], file_paths[TEST_FILE_SECOND_COPY], file_paths[TEST_FILE_THIRD_COPY],
};

_Static_assert(TEST_FILE_THIRD_COPY == COPY_COUNT - 1, "the copies are the first files, in the order of copy_paths");

static int failures = 0;
/* What each copy holds. The copies differ on purpose, word by word: the storage never compares them, and only so
 * does a read show which copy it came from. */
static unsigned char file_content[COPY_COUNT][FILE_BYTE_COUNT];
static size_t block_byte_count = 0;             /* the block alignment the first reader stated */
static struct storage_reader *single_copy_reader =
	NULL; /* also what a refused open must leave in place */

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
	    (void)fprintf(stderr,
			  "FAILED: %s: the status was \"%s\", not \"%s\"\n",
			  what, status_text(status),
			  status_text(expected_status));
	    failures++;
    }
}

static void remove_files(void)
{
    for (int file_index = 0; file_index < TEST_FILE_COUNT; file_index++) {
        if (file_exists[file_index] && unlink(file_paths[file_index]) != 0) {
            (void)fprintf(stderr, "FAILED: the test file %s is removed\n", file_paths[file_index]);
            failures++;
        }
        file_exists[file_index] = false;
    }
}

/* For what the test cannot go on without. */
static _Noreturn void give_up(const char *what)
{
    (void)fprintf(stderr, "FAILED: %s\n", what);
    remove_files();
    exit(EXIT_FAILURE);
}

/* Every word of every copy is another one, so a block that comes from the wrong place or the wrong copy is seen. */
static void make_content(void)
{
    for (size_t copy_index = 0; copy_index < COPY_COUNT; copy_index++) {
        for (size_t word_index = 0; word_index < FILE_BYTE_COUNT / sizeof(uint64_t); word_index++) {
            uint64_t word = ((uint64_t)(word_index * COPY_COUNT + copy_index) + 1) * CONTENT_MULTIPLIER;
            word ^= word >> CONTENT_SHIFT;
            memcpy(file_content[copy_index] + word_index * sizeof(word), &word, sizeof(word));
        }
    }
}

static void make_files(void)
{
    for (int file_index = 0; file_index < TEST_FILE_COUNT; file_index++) {
        /* a copy holds its own content; every other file holds that of the first copy */
        const unsigned char *const content = file_content[file_index < COPY_COUNT ? file_index : 0];
        size_t written_byte_count = 0;
        int descriptor = -1;
        memcpy(file_paths[file_index], FILE_PATH_TEMPLATE, sizeof(FILE_PATH_TEMPLATE));
        descriptor = mkstemp(file_paths[file_index]);
        if (descriptor < 0) {
            give_up("a test file is made in the current directory");
        }
        file_exists[file_index] = true;
        while (written_byte_count < file_byte_counts[file_index]) {
            const ssize_t result = write(descriptor, content + written_byte_count,
                                         file_byte_counts[file_index] - written_byte_count);
            if (result <= 0 && (result == 0 || errno != EINTR)) {
                give_up("the content of a test file is written");
            }
            written_byte_count += result > 0 ? (size_t)result : 0;
        }
        if (fsync(descriptor) != 0 || close(descriptor) != 0) {
            give_up("a test file reaches its drive and is closed");
        }
        if (file_index == TEST_FILE_MISSING) {
            file_exists[file_index] = unlink(file_paths[file_index]) != 0;
        }
    }
}

static bool is_power_of_two(uint32_t value)
{
    return value != 0 && (value & (value - 1)) == 0;
}

/* The lowest descriptor number that is free: the same before and after a call that leaves nothing open. */
static int lowest_free_descriptor(void)
{
    const int descriptor = dup(STDERR_FILENO);
    expect(descriptor >= 0 && close(descriptor) == 0, "the lowest free descriptor is found");
    return descriptor;
}

/* Reads a range through one copy into a fresh destination and compares it with what was written to that copy: copy
 * k of every reader in this test is a file with the content of copy k % COPY_COUNT. Called from several threads at
 * once: it touches nothing but its own destination. */
static bool range_reads_back(const struct storage_reader *reader,
			     uint32_t copy_index, uint64_t byte_offset,
			     uint64_t byte_count)
{
    unsigned char *const destination = aligned_alloc(block_byte_count, (size_t)byte_count);
    bool is_as_written = false;
    if (destination != NULL) {
        memset(destination, 0, (size_t)byte_count);      /* no word of the content is zero */
	is_as_written =
		storage_direct.reader_read(reader, copy_index, byte_offset,
					   destination,
					   byte_count) == STATUS_OK &&
		memcmp(destination,
		       file_content[copy_index % COPY_COUNT] + byte_offset,
		       (size_t)byte_count) == 0;
    }
    free(destination);
    return is_as_written;
}

/* An open that must be refused: with that status, with its output as it was, and with nothing left open. */
static void expect_open_refused(const char *const *paths, uint32_t path_count,
				result_status expected_status, const char *what)
{
	struct storage_reader *reader = single_copy_reader;
	const int free_descriptor = lowest_free_descriptor();
	const result_status status =
		storage_direct.reader_open(paths, path_count, &reader);
	expect_status(status, expected_status, what);
	if (reader != single_copy_reader) {
		(void)fprintf(stderr,
			      "FAILED: %s: the open changed its output\n",
			      what);
		failures++;
		if (status == STATUS_OK) {
			storage_direct.reader_close(reader);
		}
    }
    if (lowest_free_descriptor() != free_descriptor) {
        (void)fprintf(stderr, "FAILED: %s: the open left a descriptor open\n", what);
        failures++;
    }
}

static void expect_read_refused(const struct storage_reader *reader,
				uint32_t copy_index, uint64_t byte_offset,
				void *destination, uint64_t byte_count,
				const char *what)
{
	expect_status(storage_direct.reader_read(reader, copy_index,
						 byte_offset, destination,
						 byte_count),
		      STATUS_INVALID_ARGUMENT, what);
}

/* The paths of the three copies, over and over. */
static void fill_paths(const char **paths, uint32_t path_count)
{
    for (uint32_t path_index = 0; path_index < path_count; path_index++) {
        paths[path_index] = copy_paths[path_index % COPY_COUNT];
    }
}

static void test_description(const struct storage_reader *reader,
			     uint32_t expected_copy_count)
{
	struct storage_description description = {
		.byte_count = UNTOUCHED_VALUE,
		.copy_count = UNTOUCHED_VALUE,
		.block_alignment_bytes = UNTOUCHED_VALUE
	};
	expect_status(storage_direct.reader_describe(NULL, &description),
		      STATUS_INVALID_ARGUMENT,
		      "the description of no reader is refused");
	expect_status(storage_direct.reader_describe(reader, NULL),
		      STATUS_INVALID_ARGUMENT,
		      "a description with no output is refused");
	expect(description.byte_count == UNTOUCHED_VALUE &&
		       description.copy_count == UNTOUCHED_VALUE &&
		       description.block_alignment_bytes == UNTOUCHED_VALUE,
	       "a refused description leaves its output untouched");
	expect_status(storage_direct.reader_describe(reader, &description),
		      STATUS_OK, "a reader is described");
	expect(description.byte_count == FILE_BYTE_COUNT,
	       "the description gives the size of the pack");
	expect(description.copy_count == expected_copy_count,
	       "the description gives the number of copies");
	expect(is_power_of_two(description.block_alignment_bytes) &&
		       description.block_alignment_bytes == block_byte_count,
	       "the block alignment is a power of two, the same for every reader of these copies");
}

/* Content written is content read: single blocks, runs of blocks and the whole file, through every copy. */
static void test_reads(const struct storage_reader *reader, uint32_t copy_count)
{
    const uint64_t block = block_byte_count;
    const struct {
        uint64_t byte_offset;
        uint64_t byte_count;
        const char *what;
    } ranges[] = {
        {0, block, "the first block reads back"},
        {FILE_BYTE_COUNT - block, block, "the last block reads back"},
        {MIDDLE_BYTE_OFFSET, block, "a single block in the middle reads back"},
        {block, block * RUN_BLOCK_COUNT, "a run of blocks reads back"},
        {MIDDLE_BYTE_OFFSET - block, MIDDLE_BYTE_OFFSET + block, "a run of blocks up to the end reads back"},
        {0, FILE_BYTE_COUNT, "the whole file reads back"},
    };
    for (uint32_t copy_index = 0; copy_index < copy_count; copy_index++) {
        for (uint32_t range_index = 0; range_index < COUNT_OF(ranges); range_index++) {
            if (!range_reads_back(reader, copy_index, ranges[range_index].byte_offset,
                                  ranges[range_index].byte_count)) {
                (void)fprintf(stderr, "FAILED: %s (copy %" PRIu32 " of %" PRIu32 ")\n", ranges[range_index].what,
                              copy_index, copy_count);
                failures++;
            }
        }
    }
}

/* Every refusal of a read, through a reader of one copy; the reader reads on after them. */
static void test_refused_reads(const struct storage_reader *reader)
{
    const uint64_t block = block_byte_count;
    const uint64_t last_aligned_offset = UINT64_MAX - block + 1;
    unsigned char *const destination = aligned_alloc(block_byte_count, block_byte_count * RUN_BLOCK_COUNT);
    if (destination == NULL) {
        expect(false, "the destination of the refused reads is allocated");
        return;
    }
    expect_read_refused(NULL, 0, 0, destination, block, "a read through no reader is refused");
    expect_read_refused(reader, 0, 0, NULL, block, "a read with no destination is refused");
    expect_read_refused(reader, 1, 0, destination, block, "a read of the copy after the last is refused");
    expect_read_refused(reader, UINT32_MAX, 0, destination, block, "a read of a copy far after the last is refused");
    expect_read_refused(reader, 0, 0, destination, 0, "a read of no bytes is refused");
    expect_read_refused(reader, 0, FILE_BYTE_COUNT, destination, block, "a read from the end of the pack is refused");
    expect_read_refused(reader, 0, FILE_BYTE_COUNT - block, destination, block * RUN_BLOCK_COUNT,
                        "a read that runs past the end of the pack is refused");
    expect_read_refused(reader, 0, last_aligned_offset, destination, block * RUN_BLOCK_COUNT,
                        "a read whose end wraps around is refused");
    expect_read_refused(reader, 0, 0, destination, last_aligned_offset, "a read of more than the pack is refused");
    expect_read_refused(reader, 0, 1, destination, block, "a byte offset off the alignment is refused");
    expect_read_refused(reader, 0, 0, destination, block + 1, "a byte count off the alignment is refused");
    expect_read_refused(reader, 0, 0, destination + 1, block, "a destination off the alignment is refused");
    free(destination);
    expect(range_reads_back(reader, 0, 0, block), "a reader reads on after refused reads");
}

/* Every refusal of an open that needs no fault point. */
static void test_refused_opens(void)
{
    const char *const no_path[] = {NULL};
    const char *const no_second_path[] = {copy_paths[0], NULL};
    const char *const missing[] = {file_paths[TEST_FILE_MISSING]};
    const char *const missing_second[] = {copy_paths[0], file_paths[TEST_FILE_MISSING]};
    const char *const directory[] = {"."};
    const char *const directory_second[] = {copy_paths[0], "."};
    const char *const shorter_second[] = {copy_paths[0], file_paths[TEST_FILE_SHORTER]};
    const char *const shorter_first[] = {file_paths[TEST_FILE_SHORTER], copy_paths[0]};
    const char *const off_block[] = {file_paths[TEST_FILE_OFF_BLOCK]};
    const char *const empty[] = {file_paths[TEST_FILE_EMPTY]};
    const char *too_many[STORAGE_MAXIMUM_COPIES + 1];

    fill_paths(too_many, COUNT_OF(too_many));
    expect_open_refused(NULL, 1, STATUS_INVALID_ARGUMENT,
			"an open with no paths is refused");
    expect_open_refused(copy_paths, 0, STATUS_INVALID_ARGUMENT,
			"an open of no copies is refused");
    expect_open_refused(
	    too_many, COUNT_OF(too_many), STATUS_INVALID_ARGUMENT,
	    "an open of more copies than a reader takes is refused");
    expect_open_refused(no_path, COUNT_OF(no_path), STATUS_INVALID_ARGUMENT,
			"a path that is NULL is refused");
    expect_open_refused(no_second_path, COUNT_OF(no_second_path),
			STATUS_INVALID_ARGUMENT,
			"a second path that is NULL is refused");
    expect_status(storage_direct.reader_open(copy_paths, COPY_COUNT, NULL),
		  STATUS_INVALID_ARGUMENT, "an open with no output is refused");

    expect_open_refused(missing, COUNT_OF(missing), STATUS_INPUT_OUTPUT_ERROR,
			"a missing file is refused");
    expect_open_refused(missing_second, COUNT_OF(missing_second),
			STATUS_INPUT_OUTPUT_ERROR,
			"a missing file as the second copy is refused");
    expect_open_refused(directory, COUNT_OF(directory),
			STATUS_INPUT_OUTPUT_ERROR, "a directory is refused");
    expect_open_refused(directory_second, COUNT_OF(directory_second),
			STATUS_INPUT_OUTPUT_ERROR,
			"a directory as the second copy is refused");

    expect_open_refused(shorter_second, COUNT_OF(shorter_second),
			STATUS_CORRUPT_PACK,
			"a second copy smaller than the first is refused");
    expect_open_refused(shorter_first, COUNT_OF(shorter_first),
			STATUS_CORRUPT_PACK,
			"a second copy larger than the first is refused");
    expect_open_refused(
	    off_block, COUNT_OF(off_block), STATUS_CORRUPT_PACK,
	    "a size that is no multiple of the block alignment is refused");
    expect_open_refused(empty, COUNT_OF(empty), STATUS_CORRUPT_PACK,
			"an empty file is refused");
}

/* A reader holds nothing of the paths it was opened with: they are gone, and changed, before it reads. */
static void test_paths_are_not_kept(void)
{
	struct storage_reader *reader = NULL;
	result_status status = STATUS_OK;
	{
		char path[sizeof(FILE_PATH_TEMPLATE)];
		const char *paths[] = { path };
		memcpy(path, copy_paths[0], sizeof(path));
		status = storage_direct.reader_open(paths, COUNT_OF(paths),
						    &reader);
		memset(path, 0, sizeof(path));
		paths[0] = NULL;
	}
	expect_status(status, STATUS_OK,
		      "a copy opens from a path that is gone afterwards");
	if (status == STATUS_OK) {
		expect(range_reads_back(reader, 0, 0, block_byte_count),
		       "a reader keeps no pointer into its paths");
		storage_direct.reader_close(reader);
	}
}

/* A copy that shrinks after it was opened: a read across its new end gets a part and then nothing, and fails. */
static void test_read_of_a_copy_that_shrank(void)
{
    const char *const paths[] = {file_paths[TEST_FILE_SHRINKING]};
    const uint64_t block = block_byte_count;
    struct storage_reader *reader = NULL;
    unsigned char *const destination = aligned_alloc(block_byte_count, block_byte_count * RUN_BLOCK_COUNT);
    expect_status(storage_direct.reader_open(paths, COUNT_OF(paths), &reader),
		  STATUS_OK, "the copy that will shrink opens");
    expect(destination != NULL, "the destination of the reads of a copy that shrank is allocated");
    if (destination != NULL && reader != NULL) {
        expect(truncate(file_paths[TEST_FILE_SHRINKING], (off_t)MIDDLE_BYTE_OFFSET) == 0,
               "the open copy is cut to half");
	expect_status(storage_direct.reader_read(
			      reader, 0, MIDDLE_BYTE_OFFSET - block,
			      destination, block * RUN_BLOCK_COUNT),
		      STATUS_INPUT_OUTPUT_ERROR,
		      "a read across the end of a copy that shrank fails");
	expect_status(storage_direct.reader_read(reader, 0,
						 FILE_BYTE_COUNT - block,
						 destination, block),
		      STATUS_INPUT_OUTPUT_ERROR,
		      "a read past the end of a copy that shrank fails");
	expect(range_reads_back(reader, 0, 0, block), "a reader reads on after reads that failed");
    }
    storage_direct.reader_close(reader);
    free(destination);
}

struct range_reader {
    pthread_t thread;
    const struct storage_reader *reader;
    uint64_t first_byte_offset;                 /* of the range that only this thread reads */
    uint32_t copy_count;
    uint32_t wrong_read_count;                  /* written by the thread, read once it was joined */
};

/* Reads its whole range and then one block of it, round after round, from a copy that changes with the round. */
static void *read_own_range(void *argument)
{
    struct range_reader *const own = argument;
    const uint64_t blocks_per_range = THREAD_RANGE_BYTE_COUNT / block_byte_count;
    for (uint32_t round = 0; round < THREAD_ROUND_COUNT; round++) {
        const uint32_t copy_index = round % own->copy_count;
        const uint64_t block_byte_offset = own->first_byte_offset + (round % blocks_per_range) * block_byte_count;
        if (!range_reads_back(own->reader, copy_index, own->first_byte_offset, THREAD_RANGE_BYTE_COUNT) ||
            !range_reads_back(own->reader, copy_index, block_byte_offset, block_byte_count)) {
            own->wrong_read_count++;
        }
    }
    return NULL;
}

/* Several threads read different ranges through one reader at once, each checking its bytes. */
static void test_threads(const struct storage_reader *reader,
			 uint32_t copy_count)
{
    struct range_reader readers[THREAD_COUNT];
    for (uint32_t thread_index = 0; thread_index < THREAD_COUNT; thread_index++) {
        readers[thread_index].reader = reader;
        readers[thread_index].first_byte_offset = (uint64_t)thread_index * THREAD_RANGE_BYTE_COUNT;
        readers[thread_index].copy_count = copy_count;
        readers[thread_index].wrong_read_count = 0;
        if (pthread_create(&readers[thread_index].thread, NULL, read_own_range, &readers[thread_index]) != 0) {
            give_up("a reading thread is started");
        }
    }
    for (uint32_t thread_index = 0; thread_index < THREAD_COUNT; thread_index++) {
        if (pthread_join(readers[thread_index].thread, NULL) != 0) {
            give_up("a reading thread is joined");
        }
        expect(readers[thread_index].wrong_read_count == 0, "a thread reads its range while the others read theirs");
    }
}

#if CHECKING_HOOKS_ENABLED

struct open_fault {
    const char *fault_name;
    const char *what;
    uint64_t call_number;                       /* the copy at which the fault fires, counted from one */
    result_status expected_status;
};

static const struct open_fault open_faults[] = {
	{ "storage_allocation", "an open without memory is refused", 1,
	  STATUS_OUT_OF_MEMORY },
	{ "storage_open", "a last copy that does not open is refused",
	  COPY_COUNT, STATUS_INPUT_OUTPUT_ERROR },
	{ "storage_size", "a last copy that cannot be sized is refused",
	  COPY_COUNT, STATUS_INPUT_OUTPUT_ERROR },
	{ "storage_read_bound",
	  "a last copy with a read bound above the limit is refused",
	  COPY_COUNT, STATUS_UNSUPPORTED },
};

/* What only a failing machine provokes: each refusal of an open, and a read that fails once. */
static void test_faults(const struct storage_reader *reader)
{
    unsigned char *const destination = aligned_alloc(block_byte_count, block_byte_count);
    for (uint32_t fault_index = 0; fault_index < COUNT_OF(open_faults); fault_index++) {
        const struct open_fault *const fault = &open_faults[fault_index];
	expect(fault_arm(fault->fault_name, fault->call_number),
	       "a fault of the open is armed");
	expect_open_refused(copy_paths, COPY_COUNT, fault->expected_status, fault->what);
	expect(fault_disarm() == fault->call_number,
	       "the fault of the open fired at its copy");
    }
    if (destination == NULL) {
        expect(false, "the destination of the failed read is allocated");
        return;
    }
    expect(fault_arm("storage_read", 1), "the fault of a read is armed");
    expect_status(storage_direct.reader_read(reader, 0, 0, destination,
					     block_byte_count),
		  STATUS_INPUT_OUTPUT_ERROR,
		  "a read that fails is an input or output error");
    expect(fault_disarm() == 1, "the fault of the read fired");
    free(destination);
    expect(range_reads_back(reader, 0, 0, block_byte_count), "after a failed read the range reads from the same copy");
    expect(range_reads_back(reader, 1, 0, block_byte_count), "after a failed read the range reads from another copy");
}

#endif

int main(void)
{
	struct storage_description description = { .byte_count = 0,
						   .copy_count = 0,
						   .block_alignment_bytes = 0 };
	struct storage_reader *reader = NULL;
	const char *most_paths[STORAGE_MAXIMUM_COPIES];
	result_status first_status = STATUS_OK;
	int free_descriptor = -1;

	(void)alarm(TEST_TIME_LIMIT_SECONDS);
	make_content();
	make_files();
	free_descriptor = lowest_free_descriptor();

	first_status =
		storage_direct.reader_open(copy_paths, 1, &single_copy_reader);
	if (first_status != STATUS_OK) {
		expect_status(first_status, STATUS_OK, "one copy opens");
		give_up("no copy opens: the current directory must take direct reads (O_DIRECT), on a drive within the limit");
	}
	if (storage_direct.reader_describe(single_copy_reader, &description) !=
		    STATUS_OK ||
	    !is_power_of_two(description.block_alignment_bytes) ||
	    description.block_alignment_bytes == 1 ||
	    description.block_alignment_bytes > MOST_BLOCK_BYTE_COUNT) {
		give_up("the block alignment of a copy is a power of two from 2 bytes to half of a thread's range");
	}
    block_byte_count = description.block_alignment_bytes;
    test_description(single_copy_reader, 1);
    test_reads(single_copy_reader, 1);
    test_refused_reads(single_copy_reader);
    test_refused_opens();
    test_paths_are_not_kept();
    test_read_of_a_copy_that_shrank();

    expect_status(storage_direct.reader_open(copy_paths, COPY_COUNT, &reader),
		  STATUS_OK, "several copies open");
    if (reader != NULL) {
        test_description(reader, COPY_COUNT);
        test_reads(reader, COPY_COUNT);
        test_threads(reader, COPY_COUNT);
#if CHECKING_HOOKS_ENABLED
	test_faults(reader);
#endif
	storage_direct.reader_close(reader);
	reader = NULL;
    }

    fill_paths(most_paths, COUNT_OF(most_paths));
    expect_status(storage_direct.reader_open(most_paths, COUNT_OF(most_paths),
					     &reader),
		  STATUS_OK, "as many copies as a reader takes open");
    if (reader != NULL) {
        test_description(reader, COUNT_OF(most_paths));
        expect(range_reads_back(reader, COUNT_OF(most_paths) - 1, FILE_BYTE_COUNT - block_byte_count, block_byte_count),
               "the last block reads back through the last of the most copies");
	storage_direct.reader_close(reader);
    }

    storage_direct.reader_close(single_copy_reader);
    storage_direct.reader_close(NULL);
    expect(lowest_free_descriptor() == free_descriptor, "closing the readers closed every copy");
    remove_files();
    if (failures != 0) {
        return EXIT_FAILURE;
    }
    (void)printf(
	    "storage_direct_test passed (blocks of %zu bytes, fault points %s)\n",
	    block_byte_count, CHECKING_HOOKS_ENABLED ? "used" : "compiled out");
    return EXIT_SUCCESS;
}
