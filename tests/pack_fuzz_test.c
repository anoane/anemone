/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * pack_fuzz_test: source/pack.c against bytes changed at random, and the two test supports that the tests of the
 * pack, the placement and the loader stand on, each against its own header: tests/memory_storage.c (every refusal,
 * the counters, the two faults from one thread and from several at once) and tests/pack_builder.c (the refusals of
 * a plan, the packs it writes, the content function, the sealing, the walk). Not held here: a unit or extent number
 * of more digits than a name holds, which needs a pack of 100,001 blocks, and the two ways test_pack_reseal ends the
 * program.
 *
 * The fuzz loop changes bytes of the header and the tables of a valid pack at random, seals it, and follows every
 * index, offset and count of whatever still opens. Three tensors of its pack are matrices of a weight format, so
 * that the changes reach the rules of a matrix too. It provokes thousands of error lines, so while it runs
 * descriptor 2 is a temporary file whose name the test prints first. The test removes it when it has passed: the
 * file is still there, with the log and any sanitizer report, when the test failed or died in the loop.
 */
#include <inttypes.h>
#include <pthread.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "anemone/hooks.h"
#include "anemone/pack.h"
#include "anemone/pack_digest.h"
#include "anemone/status.h"
#include "memory_storage.h"
#include "pack_builder.h"

#define BLOCK \
	PACK_BLOCK_BYTES /* short names for the many places of this test that need them */
#define ALIGNMENT WEIGHT_FORMAT_MATRIX_ALIGNMENT_BYTES
#define CONTENT_SEED 0x0123456789ABCDEFull
#define COUNT_OF(table) (sizeof(table) / sizeof((table)[0]))

/* The bytes the storage serves in its own tests: no pack, only blocks that all differ. */
#define STORAGE_BLOCK TEST_MEMORY_STORAGE_BLOCK_BYTES
#define STORAGE_BLOCK_COUNT 8
#define STORAGE_BYTE_COUNT ((uint64_t)STORAGE_BLOCK_COUNT * STORAGE_BLOCK)
#define STORAGE_LAST_BLOCK_OFFSET (STORAGE_BYTE_COUNT - STORAGE_BLOCK)
#define STORAGE_SEED 0x53544F5241474521ull
#define STORAGE_COPY_COUNT 3
#define STORAGE_LAST_COPY (STORAGE_COPY_COUNT - 1)
#define OTHER_PACK_EXTENT 1                     /* the content of the second pack a test sets: another extent's */
#define NO_FLIP UINT64_MAX                      /* in place of the offset of a flipped byte */
#define POISON_BYTE 0xA5                        /* what a destination holds before a read: no byte a read left there */
#define READS_UNTIL_THE_FAULT 2                 /* one read is still delivered, the next is the first to fail */
#define UNTOUCHED_BYTE_COUNT 0x5A5A5A5A5A5A5A5Aull

#define THREAD_COUNT 4
#define THREAD_ROUND_COUNT 500                  /* each round reads one block from each of the two copies */
#define THREAD_READ_COUNT ((uint64_t)THREAD_COUNT * THREAD_ROUND_COUNT)     /* of each copy, by all threads */
#define THREAD_FLIPPED_COPY 0
#define THREAD_FAILING_COPY 1
#define THREAD_COPY_COUNT 2
#define THREAD_FLIPPED_BYTE_OFFSET (STORAGE_BLOCK + 1)
#define THREAD_FIRST_FAILING_READ_NUMBER (THREAD_READ_COUNT / 2 + 1)
#define THREAD_DELIVERED_READ_COUNT (THREAD_FIRST_FAILING_READ_NUMBER - 1)

/* Plans at the limits of the builder. A name holds TEST_PACK_NAME_DIGITS digits of each number, so the tensor of
 * this index is the first whose name has no room; the tensors of a pack whose tensor records alone pass the limit
 * of the tables; and for each the blocks of an extent in which so many tensors still get an alignment unit. The
 * last figure is the blocks of an extent whose one tensor holds a byte more than a dimension can say. */
#define FIRST_INDEX_BEYOND_THE_NAME_DIGITS 100000u
#define TENSORS_BEYOND_THE_NAME_DIGITS (FIRST_INDEX_BEYOND_THE_NAME_DIGITS + 1)
#define TENSORS_BEYOND_THE_TABLES \
	((uint32_t)(PACK_MAXIMUM_TABLES_BYTES / PACK_TENSOR_RECORD_BYTES + 1))
#define BLOCKS_THAT_HOLD(tensor_count) ((uint32_t)((uint64_t)(tensor_count) * ALIGNMENT / BLOCK + 1))
#define BLOCKS_BEYOND_A_DIMENSION ((uint32_t)((UINT32_MAX + 1ull) / BLOCK))
#define MOST_TENSORS_OF_A_BLOCK (BLOCK / ALIGNMENT)
#define BUILDER_MODEL_NAME "pack-builder-model"

#define FUZZ_SEED 0x5EED5EED5EED5EEDull
#define FUZZ_PACK_COUNT 3000
#define FUZZ_MOST_CHANGES_PER_PACK 4
#define FUZZ_HEADER_SHARE 4                     /* one change in this many goes to the header */
#define FUZZ_SEALED_BYTES ((size_t)2 * BLOCK)   /* block 0 and the one block of tables of the fuzz pack */
#define BITS_PER_BYTE 8

/* The matrices of the fuzz pack, each in the first bytes of a tensor of 2048 bytes or more, sized by hand: [16, 64]
 * at 4 bits is 512 + 32 bytes, [16, 32] at 8 bits is 512 + 1, and [48, 64] at 8 bits, a block and a half of outputs
 * by two groups of inputs, is 3072 + 2 * 2. */
struct fuzz_matrix {
    uint32_t tensor_index;
    uint32_t weight_format;
    uint32_t output_size;
    uint32_t input_size;
    uint32_t bits_per_weight;
    uint64_t byte_count;
};
static const struct fuzz_matrix fuzz_matrices[] = {
	{ 0, WEIGHT_FORMAT_E2M1_SCALED, 16, 64, 4, 544 },
	{ 3, WEIGHT_FORMAT_E4M3_SCALED, 16, 32, 8, 513 },
	{ 8, WEIGHT_FORMAT_E4M3_SCALED, 48, 64, 8, 3076 },
};
#define FUZZ_MATRIX_DIMENSIONS 2

/* A small pack for the fuzz loop: every kind of unit, and a header and tables of two blocks to change and seal. */
static const struct test_pack_unit_plan fuzz_units[] = {
	{ PLACEMENT_UNIT_STAGE, 0, 2, 1 },
	{ PLACEMENT_UNIT_ON_DEMAND, 0, 1, 1 },
	{ PLACEMENT_UNIT_HOST_ONLY, PACK_NO_LAYER, 1, 1 },
	{ PLACEMENT_UNIT_STAGE, 1, 1, 2 },
};
static const struct test_pack_plan fuzz_plan = {
    .model_name = "pack-fuzz-model", .units = fuzz_units,
    .unit_count = (uint32_t)COUNT_OF(fuzz_units), .tensors_per_extent = 2,
    .content_seed = CONTENT_SEED,
};

static const char *const copy_paths[STORAGE_MAXIMUM_COPIES] = {
	"copy 0", "copy 1", "copy 2", "copy 3",
	"copy 4", "copy 5", "copy 6", "copy 7",
};

static int failures = 0;
static FILE *failure_stream = NULL;             /* the error stream, also while descriptor 2 is the fuzz loop's file */
static char fuzz_log_path[] = "/tmp/anemone_pack_fuzz_test_XXXXXX";
static uint64_t random_state = FUZZ_SEED;
static uint64_t walked_byte_sum = 0;            /* printed at the end, so that no read of a walk is optimised away */
static unsigned char storage_bytes[STORAGE_BLOCK_COUNT * STORAGE_BLOCK];
static unsigned char other_storage_bytes[STORAGE_BLOCK];
static unsigned char *storage_destination = NULL;   /* of the reads of the main thread: the whole storage pack */
static uint64_t tallied_read_counts[STORAGE_COPY_COUNT];    /* what the storage must have counted for each copy */
static uint64_t tallied_byte_counts[STORAGE_COPY_COUNT];

static void expect(bool condition, const char *what)
{
    if (!condition) {
        (void)fprintf(failure_stream != NULL ? failure_stream : stderr, "FAILED: %s\n", what);
        failures++;
    }
}

static void expect_status(result_status status, result_status expected_status,
			  const char *what)
{
    if (status != expected_status) {
	    (void)fprintf(failure_stream != NULL ? failure_stream : stderr,
			  "FAILED: %s: status \"%s\", not \"%s\"\n", what,
			  status_text(status), status_text(expected_status));
	    failures++;
    }
}

/* A small generator: a counter through the mixer of the digest. The same numbers in every run. */
static uint64_t next_random(void)
{
	random_state += PACK_DIGEST_POSITION_MULTIPLIER;
	return pack_digest_mix(random_state);
}

/* Opens the pack bytes through a memory storage of copy_count copies. Faults that were set stay. The reader is
 * closed before this returns: a pack reads nothing after open. */
MUST_CHECK static result_status open_pack(const void *pack_bytes,
					  uint64_t pack_byte_count,
					  uint32_t copy_count,
					  struct pack_record **pack_out)
{
	struct storage_reader *reader = NULL;
	result_status status = STATUS_OK;
	test_memory_storage_set_pack(pack_bytes, pack_byte_count);
	status = test_memory_storage.reader_open(copy_paths, copy_count,
						 &reader);
	if (status != STATUS_OK) {
		expect(false, "the memory storage opens a reader");
		return status;
	}
	status = pack_open_entry(&test_memory_storage, reader, pack_out);
	test_memory_storage.reader_close(reader);
	return status;
}

/* Walks an open pack as pack_builder.h says: every index, offset and count it gives is followed, and whatever is not
 * as pack.h says of an open pack fails the test. (pack_test.c walks the packs it changes by hand the same way.) */
static void walk_pack(const struct pack_record *pack,
		      const unsigned char *pack_bytes, uint64_t pack_byte_count)
{
    const char *const flaw = test_pack_walk(pack, pack_bytes, pack_byte_count, &walked_byte_sum);
    expect(flaw == NULL, flaw != NULL ? flaw : "an open pack is as pack.h says");
}

/* An open of the storage that must be refused with this status and leave its output as it was. */
static void expect_open_refused(const char *const *paths, uint32_t copy_count,
				result_status expected_status, const char *what)
{
    static uint64_t untouched_marker = 0;
    void *untouched_address = &untouched_marker;
    struct storage_reader *const untouched_reader = untouched_address;
    struct storage_reader *reader = untouched_reader;
    expect_status(test_memory_storage.reader_open(paths, copy_count, &reader), expected_status, what);
    expect(reader == untouched_reader, "an open that the storage refuses leaves its output untouched");
    if (reader != untouched_reader) {
        test_memory_storage.reader_close(reader);
    }
}

/* Reads the range from the copy into the destination and holds what arrived against the bytes of the storage pack,
 * with the lowest bit flipped in the byte at flipped_byte_offset (NO_FLIP: in none). Any thread may call it. */
static bool range_is_delivered(const struct storage_reader *reader,
			       uint32_t copy_index, uint64_t byte_offset,
			       uint64_t byte_count, unsigned char *destination,
			       uint64_t flipped_byte_offset)
{
    memset(destination, POISON_BYTE, (size_t)byte_count);
    if (test_memory_storage.reader_read(reader, copy_index, byte_offset,
					destination, byte_count) != STATUS_OK) {
	    return false;
    }
    for (uint64_t byte_index = 0; byte_index < byte_count; byte_index++) {
        const unsigned int flipped_bit = byte_offset + byte_index == flipped_byte_offset ? 1u : 0u;
        if (destination[byte_index] != (storage_bytes[byte_offset + byte_index] ^ flipped_bit)) {
            return false;
        }
    }
    return true;
}

/* A read of the main thread that must deliver its range, tallied as the storage must count it. */
static void expect_delivered(const struct storage_reader *reader,
			     uint32_t copy_index, uint64_t byte_offset,
			     uint64_t byte_count, uint64_t flipped_byte_offset,
			     const char *what)
{
    expect(range_is_delivered(reader, copy_index, byte_offset, byte_count, storage_destination, flipped_byte_offset),
           what);
    tallied_read_counts[copy_index]++;
    tallied_byte_counts[copy_index] += byte_count;
}

/* A read of the main thread that must fail as a drive fails: it is counted and delivers nothing. */
static void expect_failed(const struct storage_reader *reader,
			  uint32_t copy_index, const char *what)
{
	expect_status(test_memory_storage.reader_read(reader, copy_index, 0,
						      storage_destination,
						      STORAGE_BLOCK),
		      STATUS_INPUT_OUTPUT_ERROR, what);
	tallied_read_counts[copy_index]++;
}

static void expect_counters(const char *what)
{
    bool counters_are_tallied = true;
    for (uint32_t copy_index = 0; copy_index < STORAGE_MAXIMUM_COPIES;
	 copy_index++) {
	    const bool is_used = copy_index < STORAGE_COPY_COUNT;
	    counters_are_tallied =
		    counters_are_tallied &&
		    test_memory_storage_read_count(copy_index) ==
			    (is_used ? tallied_read_counts[copy_index] : 0) &&
		    test_memory_storage_delivered_byte_count(copy_index) ==
			    (is_used ? tallied_byte_counts[copy_index] : 0);
    }
    expect(counters_are_tallied, what);
}

static void forget_tallies(void)
{
    memset(tallied_read_counts, 0, sizeof(tallied_read_counts));
    memset(tallied_byte_counts, 0, sizeof(tallied_byte_counts));
}

/* What a read is refused for. A refused read is no read of a copy: no counter moves. */
struct refused_read {
    const char *what;
    bool lacks_reader;
    bool lacks_destination;
    uint32_t copy_index;
    uint64_t byte_offset;
    uint64_t destination_shift;                 /* bytes the destination stands off the block */
    uint64_t byte_count;
};

static const struct refused_read refused_reads[] = {
    {"a read without a reader is refused", true, false, 0, 0, 0, STORAGE_BLOCK},
    {"a read without a destination is refused", false, true, 0, 0, 0, STORAGE_BLOCK},
    {"a read of a copy the reader does not have is refused", false, false, STORAGE_COPY_COUNT, 0, 0, STORAGE_BLOCK},
    {"a read of no bytes is refused", false, false, 0, 0, 0, 0},
    {"a read that begins off the block is refused", false, false, 0, 1, 0, STORAGE_BLOCK},
    {"a read of a block and a byte is refused", false, false, 0, 0, 0, STORAGE_BLOCK + 1},
    {"a read of less than a block is refused", false, false, 0, 0, 0, STORAGE_BLOCK - 1},
    {"a read into a destination off the block is refused", false, false, 0, 0, 1, STORAGE_BLOCK},
    {"a read that begins at the end of the pack is refused", false, false, 0, STORAGE_BYTE_COUNT, 0, STORAGE_BLOCK},
    {"a read that begins behind the pack is refused", false, false, 0, STORAGE_BYTE_COUNT + STORAGE_BLOCK, 0,
     STORAGE_BLOCK},
    {"a read that ends behind the pack is refused", false, false, 0, STORAGE_LAST_BLOCK_OFFSET, 0,
     STORAGE_BLOCK + STORAGE_BLOCK},
    {"a read of more than the pack is refused", false, false, 0, 0, 0, STORAGE_BYTE_COUNT + STORAGE_BLOCK},
    {"a read whose end wraps is refused", false, false, 0, STORAGE_BLOCK, 0, UINT64_MAX - STORAGE_BLOCK + 1},
};

/* Everything the storage refuses: an open before a pack was set or of a pack it cannot serve, the arguments of an
 * open, of a description and of a read. */
static void test_storage_refusals(void)
{
    static const char *const paths_with_a_gap[] = {"copy 0", NULL, "copy 2"};
    const struct storage_description untouched_description = {
	    .byte_count = UNTOUCHED_BYTE_COUNT
    };
    struct storage_description description = untouched_description;
    struct storage_reader *reader = NULL;
    unsigned char *const destination = aligned_alloc(STORAGE_BLOCK, (size_t)STORAGE_BYTE_COUNT);
    test_memory_storage_reset();
    expect_open_refused(copy_paths, 1, STATUS_CORRUPT_PACK,
			"an open before a pack was set is refused");
    test_memory_storage_set_pack(storage_bytes, 0);
    expect_open_refused(copy_paths, 1, STATUS_CORRUPT_PACK,
			"an open of a pack of no bytes is refused");
    test_memory_storage_set_pack(storage_bytes, STORAGE_BYTE_COUNT - 1);
    expect_open_refused(copy_paths, 1, STATUS_CORRUPT_PACK,
			"an open of a pack off the block is refused");
    test_memory_storage_set_pack(storage_bytes, STORAGE_BYTE_COUNT);
    expect_open_refused(NULL, 1, STATUS_INVALID_ARGUMENT,
			"an open without paths is refused");
    expect_open_refused(copy_paths, 0, STATUS_INVALID_ARGUMENT,
			"an open of no copy is refused");
    expect_open_refused(
	    copy_paths, STORAGE_MAXIMUM_COPIES + 1, STATUS_INVALID_ARGUMENT,
	    "an open of more copies than a storage may have is refused");
    expect_open_refused(paths_with_a_gap, (uint32_t)COUNT_OF(paths_with_a_gap),
			STATUS_INVALID_ARGUMENT,
			"an open with a path that is missing is refused");
    expect_status(test_memory_storage.reader_open(copy_paths, 1, NULL),
		  STATUS_INVALID_ARGUMENT,
		  "an open without an output is refused");
    expect_status(test_memory_storage.reader_open(copy_paths,
						  STORAGE_COPY_COUNT, &reader),
		  STATUS_OK, "the storage opens a reader of several copies");
    if (reader == NULL || destination == NULL) {
        expect(false, "the refusals of the storage have a reader and a destination");
        test_memory_storage.reader_close(reader);
        free(destination);
        return;
    }
    expect_status(test_memory_storage.reader_describe(NULL, &description),
		  STATUS_INVALID_ARGUMENT,
		  "a description without a reader is refused");
    expect(memcmp(&description, &untouched_description, sizeof(description)) == 0,
           "a description that is refused leaves its output untouched");
    expect_status(test_memory_storage.reader_describe(reader, NULL),
		  STATUS_INVALID_ARGUMENT,
		  "a description without an output is refused");
    expect_status(test_memory_storage.reader_describe(reader, &description),
		  STATUS_OK, "the storage describes its reader");
    expect(description.byte_count == STORAGE_BYTE_COUNT && description.copy_count == STORAGE_COPY_COUNT &&
               description.block_alignment_bytes == STORAGE_BLOCK,
           "a reader has the size of the pack, the copies it was opened with and the block of the storage");
    for (size_t read_index = 0; read_index < COUNT_OF(refused_reads); read_index++) {
        const struct refused_read *refused = &refused_reads[read_index];
	expect_status(test_memory_storage.reader_read(
			      refused->lacks_reader ? NULL : reader,
			      refused->copy_index, refused->byte_offset,
			      refused->lacks_destination ?
				      NULL :
				      destination + refused->destination_shift,
			      refused->byte_count),
		      STATUS_INVALID_ARGUMENT, refused->what);
    }
    forget_tallies();
    expect_counters("a read that is refused is no read of a copy: no counter moves");
    test_memory_storage.reader_close(reader);
    test_memory_storage.reader_close(NULL);
    free(destination);
}

/* The counters and the two faults, read after read from one thread. */
static void test_storage_counters_and_faults(void)
{
	struct storage_description description = { 0 };
	struct storage_reader *reader = NULL;
	struct storage_reader *other_reader = NULL;
	storage_destination =
		aligned_alloc(STORAGE_BLOCK, (size_t)STORAGE_BYTE_COUNT);
	test_memory_storage_reset();
	test_memory_storage_set_pack(storage_bytes, STORAGE_BYTE_COUNT);
	forget_tallies();
	if (storage_destination == NULL ||
	    test_memory_storage.reader_open(copy_paths, STORAGE_COPY_COUNT,
					    &reader) != STATUS_OK) {
		expect(false,
		       "the counters and faults of the storage have a reader and a destination");
		free(storage_destination);
		storage_destination = NULL;
		return;
	}
    expect_delivered(reader, 1, STORAGE_BLOCK, STORAGE_BLOCK + STORAGE_BLOCK, NO_FLIP,
                     "a read delivers the bytes of its range");
    expect_delivered(reader, 0, 0, STORAGE_BYTE_COUNT, NO_FLIP, "one read delivers the whole pack");
    expect_delivered(reader, 0, STORAGE_LAST_BLOCK_OFFSET, STORAGE_BLOCK, NO_FLIP, "a read delivers the last block");
    expect_counters("each copy counts its own reads and the bytes it delivered");

    test_memory_storage_flip_bit(STORAGE_LAST_COPY, STORAGE_BLOCK);
    expect_delivered(reader, STORAGE_LAST_COPY, STORAGE_BLOCK, STORAGE_BLOCK, STORAGE_BLOCK,
                     "a read that begins at the flipped byte delivers it flipped, and every other byte as it is");
    expect_delivered(reader, STORAGE_LAST_COPY, 0, STORAGE_BLOCK, NO_FLIP,
                     "a read that ends right before the flipped byte delivers the pack");
    expect_delivered(reader, STORAGE_LAST_COPY, 0, STORAGE_BYTE_COUNT, STORAGE_BLOCK,
                     "every read that covers the flipped byte delivers it flipped");
    expect_delivered(reader, 0, STORAGE_BLOCK, STORAGE_BLOCK, NO_FLIP,
                     "a copy delivers the byte that is flipped on another as it is");
    test_memory_storage_flip_bit(STORAGE_LAST_COPY, STORAGE_BYTE_COUNT - 1);
    expect_delivered(reader, STORAGE_LAST_COPY, STORAGE_BLOCK, STORAGE_BLOCK, NO_FLIP,
                     "a flipped byte that is set again replaces the one before");
    expect_delivered(reader, STORAGE_LAST_COPY, STORAGE_LAST_BLOCK_OFFSET, STORAGE_BLOCK, STORAGE_BYTE_COUNT - 1,
                     "a read that ends with the flipped byte delivers it flipped");
    expect_delivered(reader, STORAGE_LAST_COPY, 0, STORAGE_LAST_BLOCK_OFFSET, NO_FLIP,
                     "a read of everything before the block of the flipped byte delivers the pack");
    expect_counters("a read that delivers a flipped byte is a read like any other");

    test_memory_storage_fail_reads(0, tallied_read_counts[0] + READS_UNTIL_THE_FAULT);
    expect_delivered(reader, 0, 0, STORAGE_BLOCK, NO_FLIP, "the read before the first failing one is delivered");
    expect_failed(reader, 0, "the read of the first failing number fails, counted from the reset");
    expect_failed(reader, 0, "every read after the first failing one fails");
    expect_delivered(reader, 1, 0, STORAGE_BLOCK, NO_FLIP, "a copy reads on while another fails");
    expect_counters("a read that fails is counted and delivers nothing");
    test_memory_storage_fail_reads(0, 0);
    expect_delivered(reader, 0, 0, STORAGE_BLOCK, NO_FLIP,
                     "a first failing read number of zero takes the fault away, and the reader is still usable");

    test_memory_storage_fail_reads(STORAGE_MAXIMUM_COPIES, 1);
    test_memory_storage_flip_bit(STORAGE_MAXIMUM_COPIES, 0);
    expect(test_memory_storage_read_count(STORAGE_MAXIMUM_COPIES) == 0 &&
		   test_memory_storage_delivered_byte_count(
			   STORAGE_MAXIMUM_COPIES) == 0,
	   "a copy that no storage can have has counted nothing");
    expect_delivered(reader, 0, 0, STORAGE_BLOCK, NO_FLIP, "a fault set for a copy that cannot exist reaches no read");
    expect_counters("nor does it reach a counter");

    test_memory_storage_set_pack(other_storage_bytes, STORAGE_BLOCK);
    expect_status(test_memory_storage.reader_describe(reader, &description),
		  STATUS_OK,
		  "a reader is described after another pack was set");
    expect(description.byte_count == STORAGE_BYTE_COUNT, "a reader keeps the size of the pack it was opened on");
    expect_delivered(reader, 1, STORAGE_LAST_BLOCK_OFFSET, STORAGE_BLOCK, NO_FLIP,
                     "a reader keeps the bytes of the pack it was opened on");
    expect_status(test_memory_storage.reader_open(copy_paths, 1, &other_reader),
		  STATUS_OK, "a reader opens on the pack that was set last");
    if (other_reader != NULL) {
        memset(storage_destination, POISON_BYTE, STORAGE_BLOCK);
	expect_status(test_memory_storage.reader_read(other_reader, 0, 0,
						      storage_destination,
						      STORAGE_BLOCK),
		      STATUS_OK, "the pack that was set last is read");
	expect(memcmp(storage_destination, other_storage_bytes, STORAGE_BLOCK) == 0,
               "a reader delivers the pack that was set when it was opened");
        tallied_read_counts[0]++;
        tallied_byte_counts[0] += STORAGE_BLOCK;
        test_memory_storage.reader_close(other_reader);
    }
    expect_counters("the readers of one storage share its counters, copy by copy");

    test_memory_storage_fail_reads(1, 1);
    test_memory_storage_reset();
    forget_tallies();
    expect_counters("a reset zeroes every counter");
    expect_open_refused(copy_paths, 1, STATUS_CORRUPT_PACK,
			"a reset forgets the pack");
    expect_delivered(reader, STORAGE_LAST_COPY, STORAGE_LAST_BLOCK_OFFSET, STORAGE_BLOCK, NO_FLIP,
                     "a reset takes the flipped byte away and leaves an open reader its pack");
    expect_delivered(reader, 1, 0, STORAGE_BLOCK, NO_FLIP, "a reset takes the failing reads away");
    expect_counters("the counters count again from the reset");
    test_memory_storage.reader_close(reader);
    free(storage_destination);
    storage_destination = NULL;
}

struct storage_thread {
    pthread_t thread;
    const struct storage_reader *reader;
    unsigned char *destination;                 /* one block, this thread's alone */
    uint32_t thread_index;
    bool is_started;
    /* written by the thread, read once it was joined */
    uint64_t delivered_read_count;              /* of the failing copy */
    uint64_t failed_read_count;                 /* of the failing copy */
    uint64_t wrong_read_count;
};

/* Round after round one block from the copy with the flipped byte, which must arrive as that copy holds it, and
 * the same block from the failing copy, which must arrive as the pack holds it or fail as a drive fails. */
static void *read_both_copies(void *argument)
{
    struct storage_thread *const own = argument;
    for (uint32_t round = 0; round < THREAD_ROUND_COUNT; round++) {
        const uint64_t byte_offset = (uint64_t)((own->thread_index + round) % STORAGE_BLOCK_COUNT) * STORAGE_BLOCK;
	result_status status = STATUS_OK;
	if (!range_is_delivered(own->reader, THREAD_FLIPPED_COPY, byte_offset, STORAGE_BLOCK, own->destination,
                                THREAD_FLIPPED_BYTE_OFFSET)) {
            own->wrong_read_count++;
        }
        memset(own->destination, POISON_BYTE, STORAGE_BLOCK);
        status = test_memory_storage.reader_read(own->reader, THREAD_FAILING_COPY, byte_offset, own->destination,
                                                 STORAGE_BLOCK);
	if (status == STATUS_OK &&
	    memcmp(own->destination, storage_bytes + byte_offset,
		   STORAGE_BLOCK) == 0) {
		own->delivered_read_count++;
	} else if (status == STATUS_INPUT_OUTPUT_ERROR) {
		own->failed_read_count++;
	} else {
		own->wrong_read_count++;
	}
    }
    return NULL;
}

/* The two faults while several threads read through one reader: a read number belongs to one read, so exactly the
 * reads before the first failing one are delivered, whichever threads they came from, and no count is lost. */
static void test_storage_threads(void)
{
    struct storage_thread threads[THREAD_COUNT];
    struct storage_reader *reader = NULL;
    uint64_t delivered_read_count = 0;
    uint64_t failed_read_count = 0;
    uint64_t wrong_read_count = 0;
    test_memory_storage_reset();
    test_memory_storage_set_pack(storage_bytes, STORAGE_BYTE_COUNT);
    test_memory_storage_flip_bit(THREAD_FLIPPED_COPY, THREAD_FLIPPED_BYTE_OFFSET);
    test_memory_storage_fail_reads(THREAD_FAILING_COPY, THREAD_FIRST_FAILING_READ_NUMBER);
    if (test_memory_storage.reader_open(copy_paths, THREAD_COPY_COUNT,
					&reader) != STATUS_OK) {
	    expect(false,
		   "the storage opens the reader that the threads share");
	    return;
    }
    for (uint32_t thread_index = 0; thread_index < THREAD_COUNT; thread_index++) {
        struct storage_thread *const own = &threads[thread_index];
        *own = (struct storage_thread){.reader = reader, .thread_index = thread_index,
                                       .destination = aligned_alloc(STORAGE_BLOCK, STORAGE_BLOCK)};
        own->is_started = own->destination != NULL && pthread_create(&own->thread, NULL, read_both_copies, own) == 0;
        expect(own->is_started, "a reading thread has its block and is started");
    }
    for (uint32_t thread_index = 0; thread_index < THREAD_COUNT; thread_index++) {
        struct storage_thread *const own = &threads[thread_index];
        if (own->is_started) {
            expect(pthread_join(own->thread, NULL) == 0, "a reading thread is joined");
            delivered_read_count += own->delivered_read_count;
            failed_read_count += own->failed_read_count;
            wrong_read_count += own->wrong_read_count;
        }
        free(own->destination);
    }
    expect(wrong_read_count == 0,
           "every read of every thread delivers its range, the flipped byte flipped, or fails as a drive fails");
    expect(delivered_read_count == THREAD_DELIVERED_READ_COUNT &&
               failed_read_count == THREAD_READ_COUNT - THREAD_DELIVERED_READ_COUNT,
           "of the reads of all threads exactly those before the first failing one are delivered");
    expect(test_memory_storage_read_count(THREAD_FLIPPED_COPY) == THREAD_READ_COUNT &&
               test_memory_storage_delivered_byte_count(THREAD_FLIPPED_COPY) == THREAD_READ_COUNT * STORAGE_BLOCK &&
               test_memory_storage_read_count(THREAD_FAILING_COPY) == THREAD_READ_COUNT &&
               test_memory_storage_delivered_byte_count(THREAD_FAILING_COPY) ==
                   THREAD_DELIVERED_READ_COUNT * STORAGE_BLOCK,
           "the counters hold every read of every thread and the bytes of those that were delivered");
    test_memory_storage.reader_close(reader);
}

#if CHECKING_HOOKS_ENABLED

static void test_storage_memory_failure(void)
{
    test_memory_storage_reset();
    test_memory_storage_set_pack(storage_bytes, STORAGE_BYTE_COUNT);
    expect(fault_arm("test_memory_storage_reader_memory", 1),
	   "a lack of memory for a reader is armed");
    expect_open_refused(copy_paths, 1, STATUS_OUT_OF_MEMORY,
			"an open that lacks memory says so");
    expect(fault_disarm() == 1, "the open reached its one allocation");
}

#endif

/* The plan whose pack the builder's own tests hold against it: every kind of unit, a host-only unit first, units
 * of one and of two extents, extents of one, two and three blocks. */
static const struct test_pack_unit_plan builder_units[] = {
	{ PLACEMENT_UNIT_HOST_ONLY, PACK_NO_LAYER, 1, 1 },
	{ PLACEMENT_UNIT_STAGE, 0, 2, 1 },
	{ PLACEMENT_UNIT_ON_DEMAND, 0, 1, 3 },
	{ PLACEMENT_UNIT_STAGE, 1, 1, 2 },
};
static const struct test_pack_unit_plan one_stage_unit[] = {
	{ PLACEMENT_UNIT_STAGE, 0, 1, 1 }
};
static const struct test_pack_unit_plan unit_without_extents[] = {
	{ PLACEMENT_UNIT_STAGE, 0, 0, 1 }
};
static const struct test_pack_unit_plan unit_without_blocks[] = {
	{ PLACEMENT_UNIT_STAGE, 0, 1, 0 }
};
static const struct test_pack_unit_plan unit_of_no_kind[] = {
	{ PLACEMENT_UNIT_KIND_COUNT, 0, 1, 1 }
};
static const struct test_pack_unit_plan on_demand_unit_first[] = {
	{ PLACEMENT_UNIT_ON_DEMAND, 0, 1, 1 },
	{ PLACEMENT_UNIT_STAGE, 0, 1, 1 },
};
static const struct test_pack_unit_plan on_demand_unit_before_the_stages[] = {
	{ PLACEMENT_UNIT_HOST_ONLY, PACK_NO_LAYER, 1, 1 },
	{ PLACEMENT_UNIT_ON_DEMAND, 0, 1, 1 },
	{ PLACEMENT_UNIT_STAGE, 0, 1, 1 },
};
static const struct test_pack_unit_plan unit_of_too_many_blocks[] = {
	{ PLACEMENT_UNIT_STAGE, 0, UINT32_MAX, UINT32_MAX },
};
static const struct test_pack_unit_plan unit_for_the_name_digits[] = {
	{ PLACEMENT_UNIT_STAGE, 0, 1,
	  BLOCKS_THAT_HOLD(TENSORS_BEYOND_THE_NAME_DIGITS) },
};
static const struct test_pack_unit_plan unit_for_the_tables[] = {
	{ PLACEMENT_UNIT_STAGE, 0, 1,
	  BLOCKS_THAT_HOLD(TENSORS_BEYOND_THE_TABLES) },
};
static const struct test_pack_unit_plan unit_beyond_a_dimension[] = {
	{ PLACEMENT_UNIT_STAGE, 0, 1, BLOCKS_BEYOND_A_DIMENSION },
};

#define PLAN_OF(units, tensors_per_extent)                                                                      \
    {BUILDER_MODEL_NAME, (units), (uint32_t)COUNT_OF(units), (tensors_per_extent), CONTENT_SEED}

struct named_plan {
    const char *what;
    struct test_pack_plan plan;
};

static const struct named_plan refused_plans[] = {
    {"a plan without units is refused", {BUILDER_MODEL_NAME, NULL, 1, 0, CONTENT_SEED}},
    {"a plan of no unit is refused", {BUILDER_MODEL_NAME, one_stage_unit, 0, 0, CONTENT_SEED}},
    {"a plan without a model name is refused", {NULL, one_stage_unit, 1, 0, CONTENT_SEED}},
    {"a unit without extents is refused", PLAN_OF(unit_without_extents, 0)},
    {"a unit without blocks is refused", PLAN_OF(unit_without_blocks, 0)},
    {"a unit of no kind is refused", PLAN_OF(unit_of_no_kind, 0)},
    {"an on-demand unit first is refused", PLAN_OF(on_demand_unit_first, 0)},
    {"an on-demand unit with only a host-only unit before it is refused", PLAN_OF(on_demand_unit_before_the_stages, 0)},
    {"tensors that get no alignment unit of their extent are refused",
     PLAN_OF(one_stage_unit, MOST_TENSORS_OF_A_BLOCK + 1)},
    {"a tensor number of more digits than a name holds is refused",
     PLAN_OF(unit_for_the_name_digits, TENSORS_BEYOND_THE_NAME_DIGITS)},
    {"tensor records beyond the limit of the tables are refused",
     PLAN_OF(unit_for_the_tables, TENSORS_BEYOND_THE_TABLES)},
    {"more blocks than a test may build are refused", PLAN_OF(unit_of_too_many_blocks, 0)},
    {"a tensor of more bytes than a dimension can say is refused", PLAN_OF(unit_beyond_a_dimension, 1)},
};

static const struct named_plan built_plans[] = {
    {"a plan of every kind of unit is built", PLAN_OF(builder_units, 2)},
    {"a plan without tensors is built", PLAN_OF(builder_units, 0)},
    {"a plan of one unit of one block is built", PLAN_OF(one_stage_unit, 1)},
    {"as many tensors as a block has alignment units are built", PLAN_OF(one_stage_unit, MOST_TENSORS_OF_A_BLOCK)},
};

/* A build that must be refused as an invalid argument and leave both outputs as they were. */
static void expect_build_refused(const struct test_pack_plan *plan, const char *what)
{
    static uint64_t untouched_marker = 0;
    void *const untouched_bytes = &untouched_marker;
    void *pack_bytes = untouched_bytes;
    uint64_t pack_byte_count = UNTOUCHED_BYTE_COUNT;
    expect_status(test_pack_build(plan, &pack_bytes, &pack_byte_count),
		  STATUS_INVALID_ARGUMENT, what);
    expect(pack_bytes == untouched_bytes && pack_byte_count == UNTOUCHED_BYTE_COUNT,
           "a build that is refused leaves its outputs untouched");
    if (pack_bytes != untouched_bytes) {
        free(pack_bytes);
    }
}

/* Builds the plan and holds the pack against pack_builder.h: one block-aligned allocation that the reader opens
 * and that is sound, with the model name and the units of the plan in their order, and in every extent the bytes
 * of the content function. */
static void expect_plan_is_built(const struct test_pack_plan *plan, const char *what)
{
    void *pack_bytes = NULL;
    uint64_t pack_byte_count = 0;
    struct pack_record *pack = NULL;
    expect_status(test_pack_build(plan, &pack_bytes, &pack_byte_count),
		  STATUS_OK, what);
    if (pack_bytes == NULL) {
        return;
    }
    expect((uintptr_t)pack_bytes % BLOCK == 0, "a built pack begins on a block");
    test_memory_storage_reset();
    expect_status(open_pack(pack_bytes, pack_byte_count, 1, &pack), STATUS_OK,
		  what);
    if (pack != NULL) {
	    const struct pack_header *header = pack_header_of(pack);
	    const struct pack_extent *extents = pack_extents_of(pack);
	    const struct pack_unit_record_2 *units = pack_units_of(pack);
	    const unsigned char *bytes = pack_bytes;
	    uint64_t extent_count = 0;
	    uint64_t block_count = 0;
	    bool content_is_planned = true;
	    bool is_planned = header->unit_count == plan->unit_count &&
			      strcmp(header->model_name, plan->model_name) == 0;
	    walk_pack(pack, pack_bytes, pack_byte_count);
	    for (uint32_t unit_index = 0;
		 is_planned && unit_index < plan->unit_count; unit_index++) {
		    const struct test_pack_unit_plan *unit_plan =
			    &plan->units[unit_index];
		    const uint64_t unit_block_count =
			    (uint64_t)unit_plan->extent_count *
			    unit_plan->blocks_per_extent;
		    is_planned = units[unit_index].unit_kind ==
					 unit_plan->unit_kind &&
				 units[unit_index].layer_index ==
					 unit_plan->layer_index &&
				 units[unit_index].extent_count ==
					 unit_plan->extent_count &&
				 units[unit_index].byte_count ==
					 unit_block_count * BLOCK;
		    extent_count += unit_plan->extent_count;
		    block_count += unit_block_count;
	    }
        is_planned = is_planned && header->extent_count == extent_count &&
                     header->tensor_count == extent_count * plan->tensors_per_extent &&
                     pack_byte_count == BLOCK + header->tables_byte_count + block_count * BLOCK;
        expect(is_planned, "a built pack holds the model name and the units of its plan in their order, and no more");
        for (uint32_t extent_index = 0; is_planned && extent_index < header->extent_count; extent_index++) {
		const struct pack_extent *extent = &extents[extent_index];
		for (uint64_t byte_index = 0;
		     content_is_planned && byte_index < extent->byte_count;
		     byte_index++) {
			content_is_planned =
				bytes[extent->byte_offset + byte_index] ==
				test_pack_content_byte(plan->content_seed,
						       extent_index,
						       byte_index);
		}
        }
        expect(content_is_planned, "every byte of every extent is the one test_pack_content_byte gives");
    }
    pack_close(pack);
    free(pack_bytes);
}

/* The plans the builder refuses and what it makes of those it accepts. */
static void test_builder_plans(void)
{
	char longest_name[MODEL_NAME_CAPACITY + 1];
	struct test_pack_plan plan = built_plans[0].plan;
	for (size_t plan_index = 0; plan_index < COUNT_OF(refused_plans);
	     plan_index++) {
		expect_build_refused(&refused_plans[plan_index].plan,
				     refused_plans[plan_index].what);
	}
    expect_build_refused(NULL, "a build without a plan is refused");
    expect_status(test_pack_build(&plan, NULL, &(uint64_t){ 0 }),
		  STATUS_INVALID_ARGUMENT,
		  "a build without an output for the bytes is refused");
    expect_status(test_pack_build(&plan, &(void *){ NULL }, NULL),
		  STATUS_INVALID_ARGUMENT,
		  "a build without an output for the size is refused");
    memset(longest_name, 'm', MODEL_NAME_CAPACITY);
    longest_name[MODEL_NAME_CAPACITY] = '\0';
    plan.model_name = longest_name;
    expect_build_refused(&plan, "a model name that leaves the header no room for its NUL is refused");
    longest_name[MODEL_NAME_CAPACITY - 1] = '\0';
    expect_plan_is_built(&plan, "a model name that fills the header up to its NUL is built");
    for (size_t plan_index = 0; plan_index < COUNT_OF(built_plans); plan_index++) {
        expect_plan_is_built(&built_plans[plan_index].plan, built_plans[plan_index].what);
    }
}

/* What the content of a pack depends on, and what the sealing does. */
static void test_builder_content_and_seal(void)
{
    struct test_pack_plan plan = built_plans[0].plan;
    unsigned char *pack_bytes = NULL;
    unsigned char *same_bytes = NULL;
    unsigned char *other_bytes = NULL;
    void *built_bytes = NULL;
    uint64_t pack_byte_count = 0;
    uint64_t same_byte_count = 0;
    uint64_t other_byte_count = 0;
    struct pack_record *pack = NULL;
    result_status status =
	    test_pack_build(&plan, &built_bytes, &pack_byte_count);
    pack_bytes = built_bytes;
    status = status == STATUS_OK ?
		     test_pack_build(&plan, &built_bytes, &same_byte_count) :
		     status;
    same_bytes = status == STATUS_OK ? built_bytes : NULL;
    plan.content_seed++;
    status = status == STATUS_OK ?
		     test_pack_build(&plan, &built_bytes, &other_byte_count) :
		     status;
    other_bytes = status == STATUS_OK ? built_bytes : NULL;
    expect_status(
	    status, STATUS_OK,
	    "the builder builds one plan twice and once more with another seed");
    if (status == STATUS_OK && pack_byte_count == same_byte_count &&
	pack_byte_count == other_byte_count) {
	    const struct pack_header *header = built_bytes;
	    const uint64_t sealed_byte_count =
		    BLOCK + header->tables_byte_count;
	    bool blocks_differ = true;
	    expect(memcmp(pack_bytes, same_bytes, (size_t)pack_byte_count) == 0,
		   "one plan and one seed give one pack, byte for byte");
	    test_pack_reseal(same_bytes);
	    expect(memcmp(pack_bytes, same_bytes, (size_t)pack_byte_count) == 0,
		   "sealing a pack that was not changed changes nothing: a built pack is sealed");
	    for (uint64_t byte_offset = sealed_byte_count;
		 byte_offset < pack_byte_count; byte_offset += BLOCK) {
		    blocks_differ = blocks_differ &&
				    memcmp(pack_bytes + byte_offset,
					   other_bytes + byte_offset,
					   BLOCK) != 0;
		    for (uint64_t later_offset = byte_offset + BLOCK;
			 later_offset < pack_byte_count;
			 later_offset += BLOCK) {
			    blocks_differ = blocks_differ &&
					    memcmp(pack_bytes + byte_offset,
						   pack_bytes + later_offset,
						   BLOCK) != 0;
		    }
	    }
	    expect(blocks_differ,
		   "no two blocks of the extents hold the same bytes, and another seed changes every one");
	    /* one change to block 0 and one to the tables, neither against a rule: only the two digests refuse them */
	    same_bytes[offsetof(struct pack_header, model_name)] = 'q';
	    same_bytes[BLOCK + offsetof(struct pack_extent, digest)] =
		    (unsigned char)(same_bytes[BLOCK +
					       offsetof(struct pack_extent,
							digest)] ^
				    1u);
	    test_memory_storage_reset();
	    expect_status(open_pack(same_bytes, same_byte_count, 1, &pack),
			  STATUS_CORRUPT_PACK,
			  "a pack that was changed and not sealed is refused");
	    test_pack_reseal(same_bytes);
	    expect_status(
		    open_pack(same_bytes, same_byte_count, 1, &pack), STATUS_OK,
		    "sealing makes both digests of a changed pack hold again");
	    pack_close(pack);
    } else {
	    expect(status != STATUS_OK,
		   "the packs of one plan have one size, whatever the seed");
    }
    free(pack_bytes);
    free(same_bytes);
    free(other_bytes);
}

/* The walk of an open pack: what it says of a pack held against its own bytes, against bytes of another size, and
 * of a call that lacks an argument. (What it says of a pack that breaks a rule cannot be shown: no such pack opens.) */
static void test_builder_walk(void)
{
    void *pack_bytes = NULL;
    uint64_t pack_byte_count = 0;
    uint64_t byte_sum = 0;
    struct pack_record *pack = NULL;
    if (test_pack_build(&built_plans[0].plan, &pack_bytes, &pack_byte_count) !=
	STATUS_OK) {
	    expect(false, "the pack builder builds the pack to walk");
	    return;
    }
    test_memory_storage_reset();
    expect_status(open_pack(pack_bytes, pack_byte_count, 1, &pack), STATUS_OK,
		  "the pack to walk opens");
    if (pack != NULL) {
        expect(test_pack_walk(pack, pack_bytes, pack_byte_count, &byte_sum) == NULL && byte_sum > 0,
               "a walk of a built pack finds nothing to say and has read its bytes");
        expect(test_pack_walk(pack, pack_bytes, pack_byte_count + BLOCK, &byte_sum) != NULL,
               "a walk says so when the pack is held against bytes of another size");
        expect(test_pack_walk(NULL, pack_bytes, pack_byte_count, &byte_sum) != NULL &&
                   test_pack_walk(pack, NULL, pack_byte_count, &byte_sum) != NULL &&
                   test_pack_walk(pack, pack_bytes, pack_byte_count, NULL) != NULL,
               "a walk without a pack, without its bytes or without a sum says so");
    }
    pack_close(pack);
    free(pack_bytes);
}

/* Makes both digests hold again after the header or the tables were changed. test_pack_reseal needs the true
 * tables_byte_count: a changed one is put back for it and then sealed into the header alone. */
static void reseal(struct pack_header *header, uint64_t true_tables_byte_count)
{
    const uint64_t changed_tables_byte_count = header->tables_byte_count;
    uint64_t header_digest = 0;
    header->tables_byte_count = true_tables_byte_count;
    test_pack_reseal(header);
    if (changed_tables_byte_count != true_tables_byte_count) {
        header->tables_byte_count = changed_tables_byte_count;
        header->header_digest = 0;
	expect(pack_digest_of_host_bytes(header, BLOCK, &header_digest) ==
		       STATUS_OK,
	       "the digest of a header with another tables_byte_count is taken");
	header->header_digest = header_digest;
    }
}

enum fuzz_change { FUZZ_FLIP_A_BIT, FUZZ_ADD_ONE, FUZZ_SUBTRACT_ONE, FUZZ_ZERO, FUZZ_ANY_BYTE, FUZZ_CHANGE_COUNT };

/* Changes one byte of the header, or of the records and names of the tables: where a rule has something to refuse. */
static void change_a_byte(unsigned char *pack_bytes, uint64_t table_byte_count)
{
	const uint64_t byte_offset =
		next_random() % FUZZ_HEADER_SHARE == 0 ?
			next_random() % sizeof(struct pack_header) :
			BLOCK + next_random() % table_byte_count;
	const uint64_t change = next_random() % FUZZ_CHANGE_COUNT;
	const unsigned int flipped_bit = 1u << (next_random() % BITS_PER_BYTE);
	const unsigned int byte = pack_bytes[byte_offset];
	pack_bytes[byte_offset] =
		(unsigned char)(change == FUZZ_FLIP_A_BIT ? byte ^ flipped_bit :
				change == FUZZ_ADD_ONE	  ? byte + 1u :
				change == FUZZ_SUBTRACT_ONE ? byte - 1u :
				change == FUZZ_ZERO	    ? 0u :
							      next_random());
}

/* Makes the tensors of fuzz_matrices matrices of a weight format, in the bytes of a pack the builder wrote: the
 * record changes, the bytes of the extent stay, as any byte is a code or a scale of both formats. False when the
 * pack has no such tensor or the tensor has not the bytes of the matrix. The caller seals the pack again. */
static bool make_fuzz_matrices(void *pack_bytes)
{
	const struct pack_header *header = pack_bytes;
	void *table = (unsigned char *)pack_bytes + BLOCK;
	struct pack_extent *extents = table;
	struct pack_unit_record *unit_records = NULL;
	struct pack_tensor *tensors = NULL;
	table = extents + header->extent_count;
	unit_records = table;
	table = unit_records + header->unit_count;
	tensors = table;
	for (size_t matrix_index = 0; matrix_index < COUNT_OF(fuzz_matrices);
	     matrix_index++) {
		const struct fuzz_matrix *matrix = &fuzz_matrices[matrix_index];
		struct pack_tensor *tensor = NULL;
		if (matrix->tensor_index >= header->tensor_count ||
		    tensors[matrix->tensor_index].byte_count <
			    matrix->byte_count) {
			return false;
		}
		tensor = &tensors[matrix->tensor_index];
		tensor->weight_format = matrix->weight_format;
		tensor->bits_per_weight = matrix->bits_per_weight;
		tensor->element_type = PACK_ELEMENT_UINT8;
		tensor->dimension_count = FUZZ_MATRIX_DIMENSIONS;
		tensor->dimensions[0] = matrix->output_size;
		tensor->dimensions[1] = matrix->input_size;
		tensor->byte_count = matrix->byte_count;
	}
    return true;
}

/* A few thousand packs, each the valid pack with a handful of bytes of its header or tables changed and sealed:
 * open says yes or corrupt pack and nothing else, and what it says yes to is sound. It leaves descriptor 2 on its
 * file: main puts it back. */
static void test_fuzz(void)
{
    static unsigned char valid_bytes[FUZZ_SEALED_BYTES];
    const int failures_before = failures;
    void *pack_bytes = NULL;
    struct pack_header *header = NULL;
    struct pack_record *valid_pack = NULL;
    uint64_t pack_byte_count = 0;
    uint64_t table_byte_count = 0;
    uint32_t opened_count = 0;
    uint32_t refused_count = 0;
    int log_descriptor = 0;
    if (test_pack_build(&fuzz_plan, &pack_bytes, &pack_byte_count) !=
		STATUS_OK ||
	!make_fuzz_matrices(pack_bytes)) {
	    expect(false,
		   "the pack builder builds the pack of the fuzz loop, and three of its tensors take a matrix");
	    free(pack_bytes);
	    return;
    }
    test_pack_reseal(pack_bytes);
    test_memory_storage_reset();
    expect_status(open_pack(pack_bytes, pack_byte_count, 1, &valid_pack),
		  STATUS_OK,
		  "the pack of the fuzz loop opens with its three matrices");
    if (valid_pack != NULL) {
        walk_pack(valid_pack, pack_bytes, pack_byte_count);
	pack_close(valid_pack);
    }
    log_descriptor = mkstemp(fuzz_log_path);
    header = pack_bytes;
    table_byte_count =
	    (uint64_t)header->extent_count * PACK_EXTENT_RECORD_BYTES +
	    (uint64_t)header->unit_count * PACK_UNIT_RECORD_BYTES +
	    (uint64_t)header->tensor_count * PACK_TENSOR_RECORD_BYTES +
	    header->name_byte_count;
    (void)printf("pack_fuzz_test: from the fuzz loop on the error stream is %s\n", fuzz_log_path);
    if (BLOCK + header->tables_byte_count != sizeof(valid_bytes) || log_descriptor < 0 || fflush(stdout) != 0 ||
        fflush(stderr) != 0 || dup2(log_descriptor, STDERR_FILENO) != STDERR_FILENO || close(log_descriptor) != 0) {
        expect(false, "the fuzz pack has one block of tables, and its error stream is a temporary file");
        free(pack_bytes);
        return;
    }
    memcpy(valid_bytes, pack_bytes, sizeof(valid_bytes));
    for (uint32_t pack_index = 0; pack_index < FUZZ_PACK_COUNT && failures == failures_before; pack_index++) {
        const uint64_t change_count = 1 + next_random() % FUZZ_MOST_CHANGES_PER_PACK;
	struct pack_record *pack = NULL;
	result_status status = STATUS_OK;
	memcpy(pack_bytes, valid_bytes, sizeof(valid_bytes));
        for (uint64_t change_index = 0; change_index < change_count; change_index++) {
            change_a_byte(pack_bytes, table_byte_count);
        }
        reseal(header, BLOCK);
        test_memory_storage_reset();
        status = open_pack(pack_bytes, pack_byte_count, 1, &pack);
	if (status == STATUS_OK && pack != NULL) {
		opened_count++;
		walk_pack(pack, pack_bytes, pack_byte_count);
		pack_close(pack);
	} else {
		refused_count++;
		expect_status(status, STATUS_CORRUPT_PACK,
			      "a changed pack opens or is a corrupt pack");
		expect(pack == NULL,
		       "a refused open leaves pack_out untouched");
	}
    }
    expect(opened_count > 0 && refused_count > 0, "the fuzz loop saw packs that open and packs that are refused");
    (void)printf("pack_fuzz_test: of %" PRIu32 " changed packs %" PRIu32 " opened and %" PRIu32 " were refused\n",
                 opened_count + refused_count, opened_count, refused_count);
    free(pack_bytes);
}

int main(void)
{
    const int error_stream_copy = dup(STDERR_FILENO);
    FILE *error_stream = error_stream_copy >= 0 ? fdopen(error_stream_copy, "w") : NULL;
    if (error_stream == NULL) {
        (void)fprintf(stderr, "FAILED: the error stream could not be duplicated\n");
        return EXIT_FAILURE;
    }
    (void)setvbuf(error_stream, NULL, _IONBF, 0);
    failure_stream = error_stream;
    for (uint64_t byte_index = 0; byte_index < STORAGE_BYTE_COUNT; byte_index++) {
        storage_bytes[byte_index] = test_pack_content_byte(STORAGE_SEED, 0, byte_index);
    }
    for (uint64_t byte_index = 0; byte_index < STORAGE_BLOCK; byte_index++) {
        other_storage_bytes[byte_index] = test_pack_content_byte(STORAGE_SEED, OTHER_PACK_EXTENT, byte_index);
    }
    test_storage_refusals();
    test_storage_counters_and_faults();
    test_storage_threads();
#if CHECKING_HOOKS_ENABLED
    test_storage_memory_failure();
#endif
    test_builder_plans();
    test_builder_content_and_seal();
    test_builder_walk();
    test_fuzz();
    failure_stream = NULL;
    /* descriptor 2 is the error stream again before the test ends: a report made at exit, of a leak, must be seen */
    if (dup2(error_stream_copy, STDERR_FILENO) != STDERR_FILENO || fclose(error_stream) != 0 || failures != 0) {
        return EXIT_FAILURE;
    }
    (void)unlink(fuzz_log_path);
    (void)printf("pack_fuzz_test passed (the bytes it walked add up to %" PRIu64 ")\n", walked_byte_sum);
    return EXIT_SUCCESS;
}
