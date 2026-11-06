/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * memory_storage_test: the tables-only form of tests/memory_storage.h, which the pack's and the loader's tests
 * leave alone; and, given the path of a real pack, its tables opened through it.
 *
 *   - The tables of a builder pack held through this storage itself, and through another storage of a smaller
 *     block, open as pack_open_entry opens the whole pack: the same header, extents, units, tensors and names,
 *     every name found where it is, the pack's whole size described, and nothing read but block 0 and the tables.
 *     A read inside them delivers the pack's bytes; a read of an extent, or across the end of the tables, fails
 *     with one error line; the fault of fail_reads comes first.
 *   - Each refusal of the header, with its status and its line: missing arguments; a status of the other storage,
 *     from its description and from each of its two reads; an alignment that does not divide the block; a pack of
 *     one block or of a size off the blocks; a magic or a block size not pack.h's; tables of no block, off the
 *     blocks, above the limit or ending beyond the pack; no memory (with the checking hooks). After each one the
 *     storage holds what it held before.
 *   - With an argument, the path of a pack: its tables are read with direct reads, held, and opened as the
 *     model's test will open them; every name is found at its index and no extent is read. The figures are
 *     printed. Without one the test says that it opened no real pack.
 * A failure is one line on the standard output: the case, and what was expected of it.
 */
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "anemone/hooks.h"
#include "log_expectation.h"
#include "anemone/pack.h"
#include "anemone/status.h"
#include "anemone/storage_seam.h"
#include "memory_storage.h"
#include "pack_builder.h"

#define BLOCK PACK_BLOCK_BYTES
#define CONTENT_SEED 0x5EEDF00D12345678ull
#define TENSORS_PER_EXTENT 3u
#define CAPTURE_CAPACITY 2048u
#define SMALLER_BLOCK_ALIGNMENT 512u            /* another storage's alignment that divides the block */
#define LARGER_BLOCK_ALIGNMENT 8192u            /* one that does not */
#define OFF_BLOCK_BYTES 512u                    /* what makes a size no whole number of blocks */
#define READS_OF_AN_OPEN 2u                     /* block 0 and the tables */
#define COPY_COUNT 2u
#define BLOCK_ZERO_READ 1u                      /* the first read of the other storage */
#define TABLES_READ 2u                          /* and the second */
#define NANOSECONDS_PER_SECOND 1000000000.0
#define ARGUMENT_COUNT_WITH_A_PACK 2

static int failures = 0;
static const char *context_text = "";
static char capture_path[] = "/tmp/anemone_memory_storage_test_XXXXXX";
static int capture_descriptor = -1;             /* the temporary file that stands in for the error stream */
static int error_stream_descriptor = -1;        /* a duplicate of descriptor 2, kept to put it back */
static const char *const copy_paths[COPY_COUNT] = {"copy0", "copy1"};

static void expect(bool condition, const char *what)
{
    if (!condition) {
        (void)printf("FAILED: %s: %s\n", context_text, what);
        failures++;
    }
}

/* From here on, what is logged goes to the temporary file, from its start. */
static void capture_begin(void)
{
    expect(capture_descriptor >= 0 && fflush(stderr) == 0 && ftruncate(capture_descriptor, 0) == 0 &&
               lseek(capture_descriptor, 0, SEEK_SET) == 0 && dup2(capture_descriptor, STDERR_FILENO) == STDERR_FILENO,
           "the error stream is pointed at the temporary file");
}

/* Checks what was logged since capture_begin: nothing (words NULL) or one error line with the words. */
static void capture_end_expecting(const char *words, const char *what)
{
    char text[CAPTURE_CAPACITY];
    const bool is_open = capture_descriptor >= 0 && error_stream_descriptor >= 0 && fflush(stderr) == 0;
    const ssize_t byte_count = is_open ? pread(capture_descriptor, text, sizeof(text) - 1, 0) : -1;
    expect(is_open && dup2(error_stream_descriptor, STDERR_FILENO) == STDERR_FILENO && byte_count >= 0,
           "the temporary file is read back and the error stream restored");
    text[byte_count >= 0 ? (size_t)byte_count : 0] = '\0';
    const bool is_right = test_log_is_expected(text, words, true);
    if (!is_right) {
        (void)printf("FAILED: %s: %s: expected %s \"%s\", got \"%s\"\n", context_text, what,
                     words == NULL ? "no line" : "one error line with", words == NULL ? "" : words, text);
        failures++;
    }
}

/* ---- another storage, whose description, bytes and failures the test sets ---- */

static struct storage_description source_description;
static result_status source_describe_status = STATUS_OK;
static const unsigned char *source_bytes = NULL;
static uint32_t source_failing_read = 0;        /* counted from 1 since the test set it; 0: none fails */
static uint32_t source_read_count = 0;

MUST_CHECK static result_status
source_describe(const struct storage_reader *reader,
		struct storage_description *description_out)
{
    (void)reader;
    if (source_describe_status == STATUS_OK) {
	    *description_out = source_description;
    }
    return source_describe_status;
}

MUST_CHECK static result_status
source_read(const struct storage_reader *reader, uint32_t copy_index,
	    uint64_t byte_offset, void *destination, uint64_t byte_count)
{
    (void)reader;
    source_read_count++;
    if (source_read_count == source_failing_read) {
	    return STATUS_INPUT_OUTPUT_ERROR;
    }
    if (copy_index != 0 || byte_count == 0 || byte_offset > source_description.byte_count ||
        byte_count > source_description.byte_count - byte_offset ||
        (byte_offset | byte_count | (uintptr_t)destination) % source_description.block_alignment_bytes != 0) {
	    return STATUS_INVALID_ARGUMENT;
    }
    memcpy(destination, source_bytes + byte_offset, (size_t)byte_count);
    return STATUS_OK;
}

static const struct storage_operations_record source_storage = {
	.name = "test source",
	.reader_describe = source_describe,
	.reader_read = source_read
};
static const struct storage_operations_record storage_without_reads = {
	.name = "test source without reads",
	.reader_describe = source_describe
};

/* ---- the builder pack and what it opens as ---- */

enum plan_unit { UNIT_STAGE_A, UNIT_DEMAND, UNIT_STAGE_B, UNIT_HOST, PLAN_UNIT_COUNT };
static const struct test_pack_unit_plan units[PLAN_UNIT_COUNT] = {
	[UNIT_STAGE_A] = { PLACEMENT_UNIT_STAGE, 0, 2, 1 },
	[UNIT_DEMAND] = { PLACEMENT_UNIT_ON_DEMAND, 0, 1, 2 },
	[UNIT_STAGE_B] = { PLACEMENT_UNIT_STAGE, 1, 1, 3 },
	[UNIT_HOST] = { PLACEMENT_UNIT_HOST_ONLY, PACK_NO_LAYER, 1, 1 },
};
static const struct test_pack_plan plan = {
    .model_name = "memory-storage-test-model", .units = units, .unit_count = PLAN_UNIT_COUNT,
    .tensors_per_extent = TENSORS_PER_EXTENT, .content_seed = CONTENT_SEED,
};
static unsigned char *pack_bytes = NULL;        /* the whole builder pack */
static uint64_t pack_byte_count = 0;
static uint64_t held_byte_count = 0;            /* block 0 and its tables */

/* Opens a pack through this storage with one copy. */
MUST_CHECK static result_status pack_of_storage(struct pack_record **pack_out)
{
	struct storage_reader *reader = NULL;
	result_status status =
		test_memory_storage.reader_open(copy_paths, 1, &reader);
	if (status == STATUS_OK) {
		status =
			pack_open_entry(&test_memory_storage, reader, pack_out);
		test_memory_storage.reader_close(reader);
	}
    return status;
}

/* The pack opened from the held tables is the builder pack's: its header and tables, and every name where it is. */
static void expect_same_pack(const struct pack_record *held)
{
	const struct pack_header *const header = pack_header_of(held);
	bool names_are_found = true;
	if (pack_bytes == NULL) {
		return;
	}
    /* the extent records begin the tables, which the pack holds as they were read */
    expect(memcmp(header, pack_bytes, sizeof(*header)) == 0 &&
		   memcmp(pack_extents_of(held), pack_bytes + BLOCK,
			  header->tables_byte_count) == 0,
	   "the header and the tables are the builder pack's");
    const struct pack_tensor *const tensors = pack_tensors_of(held);
    for (uint32_t tensor_index = 0; tensor_index < header->tensor_count; tensor_index++) {
	    const char *const name = pack_name_bytes_of(held) +
				     tensors[tensor_index].name_byte_offset;
	    names_are_found =
		    names_are_found &&
		    pack_find_tensor(held, name,
				     tensors[tensor_index].name_byte_count) ==
			    tensor_index;
    }
    expect(header->tensor_count == header->extent_count * TENSORS_PER_EXTENT && names_are_found,
           "every name is found at its index");
}

/* The storage still holds the builder pack's tables, and only them. */
static void expect_still_held(const char *what)
{
	struct storage_reader *reader = NULL;
	struct storage_description description = { 0 };
	unsigned char *const block = aligned_alloc(BLOCK, BLOCK);
	const bool is_open = block != NULL && pack_bytes != NULL &&
			     test_memory_storage.reader_open(
				     copy_paths, 1, &reader) == STATUS_OK;
	capture_begin();
	expect(is_open &&
		       test_memory_storage.reader_describe(
			       reader, &description) == STATUS_OK &&
		       description.byte_count == pack_byte_count &&
		       test_memory_storage.reader_read(reader, 0, 0, block,
						       BLOCK) == STATUS_OK &&
		       memcmp(block, pack_bytes, BLOCK) == 0 &&
		       test_memory_storage.reader_read(
			       reader, 0, held_byte_count, block, BLOCK) ==
			       STATUS_INPUT_OUTPUT_ERROR,
	       what);
	capture_end_expecting("holds the first", "the read of an extent");
	if (is_open) {
		test_memory_storage.reader_close(reader);
	}
    free(block);
}

/* A hold of the other storage that must be refused with that status and line, the storage left as it was. */
static void hold_refused(const struct storage_operations_record *storage,
			 const struct storage_reader *reader,
			 result_status expected, const char *words,
			 const char *what)
{
    context_text = what;
    source_read_count = 0;
    capture_begin();
    const result_status status =
	    test_memory_storage_hold_tables_of(storage, reader);
    capture_end_expecting(words, "the refusal's line");
    expect(status == expected, "the refusal's status");
    expect_still_held("the storage holds what it held");
}

static void test_tables_only(void)
{
	struct storage_reader *reader = NULL;
	struct storage_description description = { 0 };
	struct pack_record *held = NULL;
	void *block = NULL;
	const unsigned char *const pack = pack_bytes;
	context_text = "the tables of a builder pack";
	if (pack == NULL) {
		return;
	}
    test_memory_storage_reset();
    test_memory_storage_set_pack(pack, pack_byte_count);
    expect(test_memory_storage.reader_open(copy_paths, COPY_COUNT, &reader) ==
			   STATUS_OK &&
		   test_memory_storage_hold_tables_of(&test_memory_storage,
						      reader) == STATUS_OK,
	   "held through this storage itself");
    test_memory_storage.reader_close(reader);
    const uint64_t reads_before = test_memory_storage_read_count(0);
    const uint64_t delivered_before = test_memory_storage_delivered_byte_count(0);
    expect(pack_of_storage(&held) == STATUS_OK,
	   "the pack opens from its tables");
    expect(test_memory_storage_read_count(0) - reads_before == READS_OF_AN_OPEN &&
               test_memory_storage_delivered_byte_count(0) - delivered_before == held_byte_count,
           "nothing is read but block 0 and the tables");
    if (held != NULL) {
        expect_same_pack(held);
	pack_close(held);
	held = NULL;
    }
    block = aligned_alloc(BLOCK, BLOCK + BLOCK);
    expect(block != NULL &&
		   test_memory_storage.reader_open(copy_paths, 1, &reader) ==
			   STATUS_OK &&
		   test_memory_storage.reader_describe(reader, &description) ==
			   STATUS_OK &&
		   description.byte_count == pack_byte_count &&
		   description.copy_count == 1 &&
		   description.block_alignment_bytes ==
			   TEST_MEMORY_STORAGE_BLOCK_BYTES,
	   "a reader describes the whole pack");
    if (block == NULL || reader == NULL) {
        free(block);
        return;
    }
    expect(test_memory_storage.reader_read(reader, 0, held_byte_count - BLOCK,
					   block, BLOCK) == STATUS_OK &&
		   memcmp(block, pack + held_byte_count - BLOCK, BLOCK) == 0,
	   "the last block of the tables");
    capture_begin();
    expect(test_memory_storage.reader_read(reader, 0, held_byte_count - BLOCK,
					   block, BLOCK + BLOCK) ==
		   STATUS_INPUT_OUTPUT_ERROR,
	   "a read across the end of the tables fails");
    capture_end_expecting("holds the first", "a read across the end of the tables");
    test_memory_storage_fail_reads(0, test_memory_storage_read_count(0) + 1);
    capture_begin();
    expect(test_memory_storage.reader_read(reader, 0, held_byte_count, block,
					   BLOCK) == STATUS_INPUT_OUTPUT_ERROR,
	   "a pulled drive fails a read of an extent too");
    capture_end_expecting(NULL, "fail_reads comes first: no line");
    test_memory_storage.reader_close(reader);
    test_memory_storage_reset();
    context_text = "the tables held through another storage";
    source_description =
	    (struct storage_description){ pack_byte_count, 1,
					  SMALLER_BLOCK_ALIGNMENT };
    source_describe_status = STATUS_OK;
    source_bytes = pack;
    source_failing_read = 0;
    expect(test_memory_storage_hold_tables_of(
		   &source_storage, (const void *)copy_paths) == STATUS_OK &&
		   pack_of_storage(&held) == STATUS_OK,
	   "held, and the pack opens");
    if (held != NULL) {
        expect_same_pack(held);
	pack_close(held);
    }
    free(block);
}

static void test_refusals(void)
{
    unsigned char *const broken = aligned_alloc(BLOCK, (size_t)pack_byte_count);
    const struct storage_reader *const reader =
	    (const void *)copy_paths; /* never looked at */
    struct pack_header header;
    expect(broken != NULL && pack_bytes != NULL, "a copy of the pack to break");
    if (broken == NULL || pack_bytes == NULL) {
        free(broken);
        return;
    }
    hold_refused(NULL, reader, STATUS_INVALID_ARGUMENT, NULL, "no storage");
    hold_refused(&source_storage, NULL, STATUS_INVALID_ARGUMENT, NULL,
		 "no reader");
    hold_refused(&storage_without_reads, reader, STATUS_INVALID_ARGUMENT, NULL,
		 "a storage that cannot read");
    source_describe_status = STATUS_INPUT_OUTPUT_ERROR;
    hold_refused(&source_storage, reader, STATUS_INPUT_OUTPUT_ERROR, NULL,
		 "the description fails");
    source_describe_status = STATUS_OK;
    source_failing_read = BLOCK_ZERO_READ;
    hold_refused(&source_storage, reader, STATUS_INPUT_OUTPUT_ERROR, NULL,
		 "block 0 cannot be read");
    source_failing_read = TABLES_READ;
    hold_refused(&source_storage, reader, STATUS_INPUT_OUTPUT_ERROR, NULL,
		 "the tables cannot be read");
    source_failing_read = 0;
    source_description.block_alignment_bytes = LARGER_BLOCK_ALIGNMENT;
    hold_refused(&source_storage, reader, STATUS_CORRUPT_PACK,
		 "does not divide", "a larger alignment");
    source_description.block_alignment_bytes = SMALLER_BLOCK_ALIGNMENT;
    source_description.byte_count = BLOCK;
    hold_refused(&source_storage, reader, STATUS_CORRUPT_PACK,
		 "not whole blocks", "a pack of one block");
    source_description.byte_count = pack_byte_count - OFF_BLOCK_BYTES;
    hold_refused(&source_storage, reader, STATUS_CORRUPT_PACK,
		 "not whole blocks", "a size off the blocks");
    source_description.byte_count = pack_byte_count;
    source_bytes = broken;
    const uint64_t tables_byte_count = held_byte_count - BLOCK;
    const struct {
        const char *what;
        const char *words;
        uint64_t tables_byte_count;
        bool breaks_magic;
        bool breaks_block_size;
    } cases[] = {
	    { "another magic", "the magic", tables_byte_count, true, false },
	    { "another block size", "the block size", tables_byte_count, false,
	      true },
	    { "tables of no block", "one or more whole blocks", 0, false,
	      false },
	    { "tables off the blocks", "one or more whole blocks",
	      tables_byte_count + OFF_BLOCK_BYTES, false, false },
	    { "tables above the limit", "above the limit",
	      PACK_MAXIMUM_TABLES_BYTES + BLOCK, false, false },
	    { "tables beyond the pack", "beyond the pack", pack_byte_count,
	      false, false },
    };
    for (uint32_t case_index = 0; case_index < sizeof(cases) / sizeof(cases[0]); case_index++) {
        memcpy(broken, pack_bytes, (size_t)pack_byte_count);
        memcpy(&header, broken, sizeof(header));
        if (cases[case_index].breaks_magic) {
            header.magic[0] = 'X';
        }
        header.block_byte_count += cases[case_index].breaks_block_size ? (uint32_t)BLOCK : 0u;
        header.tables_byte_count = cases[case_index].tables_byte_count;
        memcpy(broken, &header, sizeof(header));
	hold_refused(&source_storage, reader, STATUS_CORRUPT_PACK,
		     cases[case_index].words, cases[case_index].what);
    }
    source_bytes = pack_bytes;
#if CHECKING_HOOKS_ENABLED
    expect(fault_arm("test_memory_storage_tables_memory", 1),
	   "the fault of the memory is armed");
    hold_refused(&source_storage, reader, STATUS_OUT_OF_MEMORY, NULL,
		 "no memory for the tables");
    expect(fault_disarm() == 1, "the fault point of the memory was reached");
#endif
    free(broken);
}

/* The tables of a real pack, read with direct reads, held and opened; the figures printed. */
static void test_real_pack(const char *path)
{
	struct storage_reader *direct = NULL;
	struct pack_record *pack = NULL;
	struct timespec started = { 0 };
	struct timespec ended = { 0 };
	bool names_are_found = true;
	context_text = "the tables of the real pack";
	test_memory_storage_reset();
	expect(clock_gettime(CLOCK_MONOTONIC, &started) == 0 &&
		       storage_direct.reader_open(&path, 1, &direct) ==
			       STATUS_OK,
	       "the pack opens");
	if (direct == NULL) {
		return;
	}
	expect(test_memory_storage_hold_tables_of(&storage_direct, direct) ==
		       STATUS_OK,
	       "its tables are held");
	storage_direct.reader_close(direct);
	expect(pack_of_storage(&pack) == STATUS_OK &&
		       clock_gettime(CLOCK_MONOTONIC, &ended) == 0,
	       "the pack opens from its tables");
	if (pack == NULL) {
		return;
	}
	const struct pack_header *const header = pack_header_of(pack);
	const struct pack_tensor *const tensors = pack_tensors_of(pack);
	for (uint32_t tensor_index = 0; tensor_index < header->tensor_count;
	     tensor_index++) {
		const char *const name = pack_name_bytes_of(pack) +
					 tensors[tensor_index].name_byte_offset;
		names_are_found =
			names_are_found &&
			pack_find_tensor(
				pack, name,
				tensors[tensor_index].name_byte_count) ==
				tensor_index;
	}
    expect(names_are_found, "every name is found at its index");
    expect(test_memory_storage_read_count(0) == READS_OF_AN_OPEN &&
               test_memory_storage_delivered_byte_count(0) == BLOCK + header->tables_byte_count,
           "nothing is read but block 0 and the tables");
    const double seconds = (double)(ended.tv_sec - started.tv_sec) +
                           (double)(ended.tv_nsec - started.tv_nsec) / NANOSECONDS_PER_SECOND;
    (void)printf("real pack: model %s, %" PRIu64 " bytes; tables %" PRIu64 " bytes: %" PRIu32 " extents, %" PRIu32
                 " units, %" PRIu32 " tensors, every name found; opened in %.3f s\n",
                 header->model_name, header->file_byte_count, header->tables_byte_count, header->extent_count,
                 header->unit_count, header->tensor_count, seconds);
    pack_close(pack);
    test_memory_storage_reset();
}

int main(int argument_count, char **arguments)
{
    if (argument_count > ARGUMENT_COUNT_WITH_A_PACK) {
        (void)printf("usage: memory_storage_test [the path of a pack]\n");
        return EXIT_FAILURE;
    }
    void *built = NULL;
    error_stream_descriptor = dup(STDERR_FILENO);
    capture_descriptor = mkstemp(capture_path);
    if (error_stream_descriptor < 0 || capture_descriptor < 0 ||
	test_pack_build(&plan, &built, &pack_byte_count) != STATUS_OK) {
	    (void)printf(
		    "FAILED: the temporary file or the builder pack could not be made\n");
	    return EXIT_FAILURE;
    }
    struct pack_header header;
    pack_bytes = built;
    memcpy(&header, pack_bytes, sizeof(header));
    held_byte_count = BLOCK + header.tables_byte_count;
    test_tables_only();
    test_refusals();
    test_memory_storage_reset();
    if (argument_count == ARGUMENT_COUNT_WITH_A_PACK) {
        test_real_pack(arguments[1]);
    } else {
        (void)printf("memory_storage_test: no pack was given, no real pack's tables were opened\n");
    }
    free(pack_bytes);
    context_text = "the end of the test";
    expect(close(capture_descriptor) == 0 && unlink(capture_path) == 0 && close(error_stream_descriptor) == 0,
           "the temporary file is removed");
    if (failures != 0) {
        return EXIT_FAILURE;
    }
    (void)printf("memory_storage_test passed\n");
    return EXIT_SUCCESS;
}
