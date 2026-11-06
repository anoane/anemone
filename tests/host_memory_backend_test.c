/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * host_memory_backend_test: the kernels of tests/host_memory_backend.h, which the loader's tests leave alone.
 *
 *   - A card of class 0 has no build of any kernel, as before: kernel_prepare and launch_run answer UNSUPPORTED
 *     without a line, launch_wait OK, format_supported false.
 *   - A card of a class keeps a registry: the same kernels in the same order give the same handles on cards of
 *     both classes, the same kernel again its handle, and each refusal of the header its status and one error line:
 *     a missing argument, a function name of no bytes or of the capacity, a record of no bytes or above the most, a
 *     kernel file the program has no build of, another record size, one kernel more than a card prepares. A closed
 *     card forgets its kernels.
 *   - launch_run refuses, with one line and nothing logged, every launch the seam refuses, the first refused one
 *     named in a list; the limits themselves are accepted. Shared memory is held to each class's own most, which
 *     the attention core's block fits on both.
 *   - The work log holds uploads, downloads, digests and launches of every card in the order they began, a launch
 *     with its bytes; a stand-in runs behind the copies begun before it and a download sees what it wrote. The
 *     stand-ins' table: refusals, replacement, removal, BUSY when full.
 *   - A launch that faults, by fail_launch or a stand-in, leaves launch_run OK and fails the card: no later stand-in
 *     runs, every call that begins work and every unfinished ticket answers DEVICE_ERROR, and its staging block
 *     may then be freed. A card opened again is sound: it uploads, waits and digests, and an upload fault a test
 *     named still falls on its own ticket.
 *   - Beyond the log's capacity work is counted and not logged, without a line. Threads on four cards at once lose
 *     no entry and keep each card's order (the thread sanitizer build watches the locking). reset forgets classes,
 *     stand-ins and the log.
 * A failure is one line on the standard output: the case, and what was expected of it.
 */
#include <inttypes.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "anemone/backend_records.h"
#include "log_expectation.h"
#include "anemone/backend_seam.h"
#include "anemone/status.h"
#include "anemone/weight_format_seam.h"
#include "host_memory_backend.h"

#define CARD_COUNT 4u
#define CARD_MEMORY_BYTES 65536u
#define ARENA_BYTES 16384u
#define STAGING_BYTES 4096u
#define CAPTURE_CAPACITY 2048u
#define DIGEST_KERNEL_FILE "kernel_pack_digest"
#define DIGEST_KERNEL_FUNCTION "kernel_pack_digest"
#define DIGEST_RECORD_BYTES 16u
#define UNBUILT_KERNEL_FILE "kernel_that_no_build_holds"
#define FILLED_WORDS 64u
#define FILL_THREADS 256u
#define FILL_VALUE 0xA5A5A5A5u
#define UPLOADED_BYTE 0x3Cu
#define LAUNCH_BATCH 1024u                      /* launches of one launch_run in the case of the full log */
#define LOG_OVERFLOW 10u                        /* launches beyond the log's capacity */
#define THREAD_COUNT 4u
#define THREAD_ROUNDS 200u
#define THREAD_VALUE_STRIDE 1000000u            /* the values of one thread's launches begin at its index times it */
#define FAILING_LAUNCH 2u
#define FAILING_TICKET 2u                       /* named in the case of fail_launch: the upload after a sound one */
#define ATTENTION_CORE_SHARED_BYTES 49664u      /* what the attention core's block asks: above 48 KiB */
#define FAULT_LIST_LENGTH 3u
#define LOG_CARDS 2u                            /* the cards of the case of the log */
#define ROWS_PAST_THE_MOST 2u                   /* of a block whose rows are each of the most threads */
#define NAME_BUFFER_BYTES 32u                   /* "function_" and a number */

/* The work of the case of the log, in the order it begins. */
enum logged_work { LOGGED_UPLOAD, LOGGED_FIRST_LAUNCH, LOGGED_DOWNLOAD, LOGGED_DIGEST, LOGGED_SECOND_LAUNCH,
                   LOGGED_WORK_COUNT };
#define COUNT_OF(table) (sizeof(table) / sizeof((table)[0]))

static int failures = 0;
static const char *context_text = "";
static char capture_path[] = "/tmp/anemone_host_memory_backend_test_XXXXXX";
static int capture_descriptor = -1;             /* the temporary file that stands in for the error stream */
static int error_stream_descriptor = -1;        /* a duplicate of descriptor 2, kept to put it back */

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

/* What was logged since capture_begin, as a string in text; descriptor 2 is put back. */
static void capture_end(char *text, size_t text_capacity)
{
    const bool is_open = capture_descriptor >= 0 && error_stream_descriptor >= 0 && fflush(stderr) == 0;
    const ssize_t byte_count = is_open ? pread(capture_descriptor, text, text_capacity - 1, 0) : -1;
    expect(is_open && dup2(error_stream_descriptor, STDERR_FILENO) == STDERR_FILENO && byte_count >= 0,
           "the temporary file is read back and the error stream restored");
    text[byte_count >= 0 ? (size_t)byte_count : 0] = '\0';
}

/* The text is nothing (words NULL), or one error line that holds the words. */
static void expect_logged(const char *text, const char *words, const char *what)
{
    const bool is_right = test_log_is_expected(text, words, true);
    if (!is_right) {
        (void)printf("FAILED: %s: %s: expected %s \"%s\", got \"%s\"\n", context_text, what,
                     words == NULL ? "no line" : "one error line with", words == NULL ? "" : words, text);
        failures++;
    }
}

/* ---- kernels and launches ---- */

MUST_CHECK static result_status prepare(struct device_card *card,
					const char *file, const char *function,
					uint32_t record_bytes,
					uint32_t *kernel_out, const char *words)
{
    char logged[CAPTURE_CAPACITY];
    capture_begin();
    const result_status status = test_host_memory_backend.kernel_prepare(
	    card, file, function, record_bytes, kernel_out);
    capture_end(logged, sizeof(logged));
    expect_logged(logged, words, function != NULL ? function : "no function");
    return status;
}

/* A launch of the fill kernel, of the handle given, over FILLED_WORDS words from the offset. */
static struct kernel_launch
fill_launch(uint32_t kernel, uint64_t arena_byte_offset, uint32_t value)
{
	const struct fill_record record = { arena_byte_offset, FILLED_WORDS,
					    value, 0 };
	struct kernel_launch launch = { .kernel = kernel,
					.grid = { 1, 1, 1 },
					.block = { FILL_THREADS, 1, 1 },
					.argument_byte_count = sizeof(record) };
	memcpy(launch.argument, &record, sizeof(record));
	return launch;
}

/* launch_run, expected to refuse the list with one line that names the launch refused, logging nothing. */
static void run_refused(struct device_card *card,
			const struct kernel_launch *launches,
			uint32_t launch_count, const char *words,
			const char *what)
{
    char logged[CAPTURE_CAPACITY];
    const uint64_t work_before = test_host_memory_backend_work_count();
    context_text = what;
    capture_begin();
    const result_status status =
	    test_host_memory_backend.launch_run(card, launches, launch_count);
    capture_end(logged, sizeof(logged));
    expect(status == STATUS_INVALID_ARGUMENT &&
		   test_host_memory_backend_work_count() == work_before,
	   "refused, nothing queued or logged");
    expect_logged(logged, words, "the launch refused is named");
}

struct stand_in_calls {
    _Atomic uint32_t count;
    uint32_t last_card_index;                   /* written under the backend's mutex */
    bool fails;
};

/* The fill kernel in host memory: value into word_count words, after the checks its launch's preparer makes. */
static bool fill_stand_in(const struct kernel_launch *launch,
			  uint32_t card_index, unsigned char *arena,
			  uint64_t arena_byte_count, void *context)
{
    struct stand_in_calls *const calls = context;
    struct fill_record record;
    memcpy(&record, launch->argument, sizeof(record));
    (void)atomic_fetch_add(&calls->count, 1);
    calls->last_card_index = card_index;
    if (calls->fails || record.arena_byte_offset > arena_byte_count ||
	record.word_count > (arena_byte_count - record.arena_byte_offset) /
				    FILL_WORD_BYTES) {
	    return false;
    }
    for (uint64_t word = 0; word < record.word_count; word++) {
	    memcpy(arena + record.arena_byte_offset + word * FILL_WORD_BYTES,
		   &record.value, sizeof(record.value));
    }
    return true;
}

/* ---- cases ---- */

static void open_cards(struct device_card **cards, uint32_t card_count)
{
    for (uint32_t card_index = 0; card_index < card_count; card_index++) {
	    expect(test_host_memory_backend.card_open(card_index, ARENA_BYTES,
						      &cards[card_index]) ==
			   STATUS_OK,
		   "a card opens");
    }
}

static void close_cards(struct device_card **cards, uint32_t card_count)
{
    for (uint32_t card_index = 0; card_index < card_count; card_index++) {
        test_host_memory_backend.card_close(cards[card_index], 0);
    }
}

static void test_class_zero(void)
{
	struct device_card *card = NULL;
	struct card_description_record description;
	uint32_t kernel = 0;
	char logged[CAPTURE_CAPACITY];
	context_text = "a card of class 0";
	test_host_memory_backend_reset(CARD_COUNT, CARD_MEMORY_BYTES);
	open_cards(&card, 1);
	const struct kernel_launch launch = fill_launch(0, 0, FILL_VALUE);
	expect(test_host_memory_backend.card_describe(0, &description) ==
			       STATUS_OK &&
		       description.arithmetic_class == 0,
	       "class 0 is described");
	expect(prepare(card, KERNEL_FILE_FILL, KERNEL_FILL_WORDS,
		       FILL_RECORD_BYTES, &kernel, NULL) == STATUS_UNSUPPORTED,
	       "no build of any kernel");
	capture_begin();
	expect(test_host_memory_backend.launch_run(card, &launch, 1) ==
			       STATUS_UNSUPPORTED &&
		       test_host_memory_backend.launch_wait(card, 0) ==
			       STATUS_OK &&
		       !test_host_memory_backend.format_supported(
			       0, WEIGHT_FORMAT_E2M1_SCALED) &&
		       test_host_memory_backend_work_count() == 0,
	       "nothing runs, nothing is logged, no format is decoded");
	capture_end(logged, sizeof(logged));
	expect_logged(logged, NULL, "no line");
	close_cards(&card, 1);
}

static void test_registry(void)
{
	static const uint32_t classes[] = { ARITHMETIC_CLASS_SM_80,
					    ARITHMETIC_CLASS_SM_120 };
	char long_name[TEST_HOST_MEMORY_BACKEND_FUNCTION_NAME_CAPACITY + 1];
	struct device_card *cards[COUNT_OF(classes)] = { NULL };
	struct test_host_memory_backend_kernel recorded;
	struct card_description_record description;
	uint32_t kernel = 0;
	test_host_memory_backend_reset(CARD_COUNT, CARD_MEMORY_BYTES);
	open_cards(cards, COUNT_OF(classes));
	for (uint32_t card_index = 0; card_index < COUNT_OF(classes);
	     card_index++) {
		struct device_card *const card = cards[card_index];
		uint32_t fill = 0;
		uint32_t digest = 0;
		context_text = card_index == 0 ?
				       "the registry of a card of sm_80" :
				       "the registry of a card of sm_120";
		test_host_memory_backend_set_arithmetic_class(
			card_index, classes[card_index]);
		expect(test_host_memory_backend.card_describe(
			       card_index, &description) == STATUS_OK &&
			       description.arithmetic_class ==
				       classes[card_index],
		       "the class is described");
		expect(test_host_memory_backend.format_supported(
			       card_index, WEIGHT_FORMAT_E2M1_SCALED) &&
			       test_host_memory_backend.format_supported(
				       card_index, WEIGHT_FORMAT_E4M3_SCALED) &&
			       !test_host_memory_backend.format_supported(
				       card_index, WEIGHT_FORMAT_NONE) &&
			       !test_host_memory_backend.format_supported(
				       card_index, WEIGHT_FORMAT_COUNT) &&
			       !test_host_memory_backend.format_supported(
				       CARD_COUNT, WEIGHT_FORMAT_E2M1_SCALED),
		       "the formats this build carries, on a card that exists");
		expect(prepare(card, KERNEL_FILE_FILL, KERNEL_FILL_WORDS,
			       FILL_RECORD_BYTES, &fill, NULL) == STATUS_OK &&
			       prepare(card, DIGEST_KERNEL_FILE,
				       DIGEST_KERNEL_FUNCTION,
				       DIGEST_RECORD_BYTES, &digest,
				       NULL) == STATUS_OK &&
			       fill == 0 && digest == 1,
		       "handles in the order of preparation, the same on both classes");
		expect(prepare(card, KERNEL_FILE_FILL, KERNEL_FILL_WORDS,
			       FILL_RECORD_BYTES, &kernel, NULL) == STATUS_OK &&
			       kernel == fill,
		       "the same kernel again: its handle");
		expect(test_host_memory_backend_kernel_of(card_index, digest,
							  &recorded) &&
			       strcmp(recorded.kernel_file_name,
				      DIGEST_KERNEL_FILE) == 0 &&
			       strcmp(recorded.function_name,
				      DIGEST_KERNEL_FUNCTION) == 0 &&
			       recorded.argument_byte_count ==
				       DIGEST_RECORD_BYTES &&
			       !test_host_memory_backend_kernel_of(
				       card_index, digest + 1, &recorded),
		       "a handle names its kernel; one beyond names none");
		kernel = UINT32_MAX;
		expect(prepare(card, UNBUILT_KERNEL_FILE, KERNEL_FILL_WORDS,
			       FILL_RECORD_BYTES, &kernel,
			       "no architecture has a build of") ==
				       STATUS_INVALID_ARGUMENT &&
			       prepare(card, KERNEL_FILE_FILL,
				       KERNEL_FILL_WORDS, DIGEST_RECORD_BYTES,
				       &kernel, "takes a record of") ==
				       STATUS_INVALID_ARGUMENT &&
			       prepare(card, KERNEL_FILE_FILL, "",
				       FILL_RECORD_BYTES, &kernel,
				       "a missing argument") ==
				       STATUS_INVALID_ARGUMENT &&
			       prepare(card, KERNEL_FILE_FILL,
				       KERNEL_FILL_WORDS, 0, &kernel,
				       "a missing argument") ==
				       STATUS_INVALID_ARGUMENT &&
			       prepare(card, KERNEL_FILE_FILL,
				       KERNEL_FILL_WORDS,
				       LAUNCH_ARGUMENT_BYTES + 1, &kernel,
				       "a missing argument") ==
				       STATUS_INVALID_ARGUMENT &&
			       prepare(card, KERNEL_FILE_FILL,
				       KERNEL_FILL_WORDS, FILL_RECORD_BYTES,
				       NULL, "a missing argument") ==
				       STATUS_INVALID_ARGUMENT &&
			       kernel == UINT32_MAX,
		       "each refusal has its status and line, the handle untouched");
		memset(long_name, 'k', sizeof(long_name) - 1);
		long_name[sizeof(long_name) - 1] = '\0';
		expect(prepare(card, KERNEL_FILE_FILL, long_name,
			       FILL_RECORD_BYTES, &kernel,
			       "a missing argument") == STATUS_INVALID_ARGUMENT,
		       "a function name of the capacity");
		long_name[TEST_HOST_MEMORY_BACKEND_FUNCTION_NAME_CAPACITY - 1] =
			'\0';
		expect(prepare(card, KERNEL_FILE_FILL, long_name,
			       FILL_RECORD_BYTES, &kernel, NULL) == STATUS_OK &&
			       kernel == digest + 1,
		       "one byte shorter is taken");
		for (uint32_t index = kernel + 1;
		     index < BACKEND_MAXIMUM_KERNELS;
		     index++) { /* the next handle */
			char name[NAME_BUFFER_BYTES];
			(void)snprintf(name, sizeof(name), "function_%" PRIu32,
				       index);
			expect(prepare(card, KERNEL_FILE_FILL, name,
				       FILL_RECORD_BYTES, &kernel,
				       NULL) == STATUS_OK &&
				       kernel == index,
			       "kernels up to the most a card prepares");
		}
		expect(prepare(card, KERNEL_FILE_FILL, "one_more",
			       FILL_RECORD_BYTES, &kernel,
			       "takes no more") == STATUS_UNSUPPORTED,
		       "one kernel more");
	}
    context_text = "a closed card";
    close_cards(cards, COUNT_OF(classes));
    expect(prepare(cards[0], KERNEL_FILE_FILL, KERNEL_FILL_WORDS,
		   FILL_RECORD_BYTES, &kernel, NULL) == STATUS_INVALID_ARGUMENT,
	   "is refused without a line");
    open_cards(cards, 1);
    expect(!test_host_memory_backend_kernel_of(0, 0, &recorded) &&
		   prepare(cards[0], DIGEST_KERNEL_FILE, DIGEST_KERNEL_FUNCTION,
			   DIGEST_RECORD_BYTES, &kernel, NULL) == STATUS_OK &&
		   kernel == 0,
	   "opened again, it has forgotten its kernels");
    close_cards(cards, 1);
}

static void test_launch_refusals(void)
{
	struct device_card *card = NULL;
	uint32_t fill = 0;
	uint32_t digest = 0;
	test_host_memory_backend_reset(CARD_COUNT, CARD_MEMORY_BYTES);
	test_host_memory_backend_set_arithmetic_class(0,
						      ARITHMETIC_CLASS_SM_80);
	open_cards(&card, 1);
	context_text = "launches";
	expect(prepare(card, KERNEL_FILE_FILL, KERNEL_FILL_WORDS,
		       FILL_RECORD_BYTES, &fill, NULL) == STATUS_OK &&
		       prepare(card, DIGEST_KERNEL_FILE, DIGEST_KERNEL_FUNCTION,
			       DIGEST_RECORD_BYTES, &digest, NULL) == STATUS_OK,
	       "two kernels are prepared");
	const struct kernel_launch good = fill_launch(fill, 0, FILL_VALUE);
	struct kernel_launch list[FAULT_LIST_LENGTH] = { good, good, good };
	struct kernel_launch bad = good;
	run_refused(card, NULL, 1, "refused at launch 0 of 1", "no launches");
	run_refused(card, &good, 0, "refused at launch 0 of 0", "a count of 0");
	bad.kernel = digest + 1;
	run_refused(card, &bad, 1, "refused at launch 0 of 1",
		    "a handle not prepared");
	bad = good;
	bad.kernel = digest;
	run_refused(card, &bad, 1, "refused at launch 0 of 1",
		    "a record of another kernel's size");
	bad = good;
	bad.branch =
		LAUNCH_SIDE_MORE +
		1u; /* the last branch kind (backend_seam.h) and one more */
	run_refused(card, &bad, 1, "refused at launch 0 of 1",
		    "a reserved member that is not zero");
	bad = good;
	bad.argument[FILL_RECORD_BYTES] = 1;
	run_refused(card, &bad, 1, "refused at launch 0 of 1",
		    "a byte behind the record that is not zero");
	for (uint32_t dimension = 0; dimension < LAUNCH_DIMENSIONS;
	     dimension++) {
		bad = good;
		bad.grid[dimension] = 0;
		run_refused(card, &bad, 1, "refused at launch 0 of 1",
			    "a grid dimension of 0");
		bad = good;
		bad.block[dimension] = 0;
		run_refused(card, &bad, 1, "refused at launch 0 of 1",
			    "a block dimension of 0");
	}
    bad = good;
    bad.grid[0] = TEST_HOST_MEMORY_BACKEND_MOST_GRID_X + 1u;
    run_refused(card, &bad, 1, "refused at launch 0 of 1", "a grid above the most in x");
    bad = good;
    bad.grid[LAUNCH_DIMENSIONS - 1] =
	    TEST_HOST_MEMORY_BACKEND_MOST_GRID_Y_Z + 1u;
    run_refused(card, &bad, 1, "refused at launch 0 of 1", "a grid above the most in z");
    bad = good;
    bad.block[0] = TEST_HOST_MEMORY_BACKEND_MOST_BLOCK_X_Y + 1u;
    run_refused(card, &bad, 1, "refused at launch 0 of 1", "a block above the most in x");
    bad = good;
    bad.block[LAUNCH_DIMENSIONS - 1] =
	    TEST_HOST_MEMORY_BACKEND_MOST_BLOCK_Z + 1u;
    run_refused(card, &bad, 1, "refused at launch 0 of 1", "a block above the most in z");
    bad = good;
    bad.block[0] = TEST_HOST_MEMORY_BACKEND_MOST_BLOCK_X_Y;
    bad.block[1] = ROWS_PAST_THE_MOST;
    run_refused(card, &bad, 1, "refused at launch 0 of 1", "a block of more threads than the most");
    list[1].block[0] = 0;
    run_refused(card, list, FAULT_LIST_LENGTH, "refused at launch 1 of 3", "the second of three");
    context_text = "launches at the limits";
    bad = good;
    bad.grid[0] = TEST_HOST_MEMORY_BACKEND_MOST_GRID_X;
    bad.grid[1] = TEST_HOST_MEMORY_BACKEND_MOST_GRID_Y_Z;
    bad.block[0] = TEST_HOST_MEMORY_BACKEND_MOST_THREADS_PER_BLOCK;
    bad.dynamic_shared_byte_count = TEST_HOST_MEMORY_BACKEND_MOST_DYNAMIC_SHARED_BYTES_SM_80;
    expect(test_host_memory_backend.launch_run(card, &bad, 1) == STATUS_OK &&
		   test_host_memory_backend.launch_wait(card, 0) == STATUS_OK &&
		   test_host_memory_backend_work_count() == 1,
	   "accepted and logged");
    close_cards(&card, 1);
}

/* Each class has its own most of dynamic shared memory: the attention core's block and the most are accepted, one
 * byte more is refused. */
static void test_shared_memory(void)
{
	static const uint32_t classes[] = { ARITHMETIC_CLASS_SM_80,
					    ARITHMETIC_CLASS_SM_120 };
	static const uint32_t most_shared_bytes[] = {
		TEST_HOST_MEMORY_BACKEND_MOST_DYNAMIC_SHARED_BYTES_SM_80,
		TEST_HOST_MEMORY_BACKEND_MOST_DYNAMIC_SHARED_BYTES_SM_120
	};
	struct device_card *cards[COUNT_OF(classes)] = { NULL };
	test_host_memory_backend_reset(CARD_COUNT, CARD_MEMORY_BYTES);
	for (uint32_t card_index = 0; card_index < COUNT_OF(classes);
	     card_index++) {
		test_host_memory_backend_set_arithmetic_class(
			card_index, classes[card_index]);
	}
    open_cards(cards, COUNT_OF(classes));
    for (uint32_t card_index = 0; card_index < COUNT_OF(classes); card_index++) {
        uint32_t fill = 0;
        context_text = card_index == 0 ? "the shared memory of sm_80" : "the shared memory of sm_120";
	expect(prepare(cards[card_index], KERNEL_FILE_FILL, KERNEL_FILL_WORDS,
		       FILL_RECORD_BYTES, &fill, NULL) == STATUS_OK,
	       "the kernel");
	struct kernel_launch launch = fill_launch(fill, 0, FILL_VALUE);
	launch.dynamic_shared_byte_count = ATTENTION_CORE_SHARED_BYTES;
	expect(test_host_memory_backend.launch_run(cards[card_index], &launch,
						   1) == STATUS_OK,
	       "the attention core's block is accepted");
	launch.dynamic_shared_byte_count = most_shared_bytes[card_index];
	expect(test_host_memory_backend.launch_run(cards[card_index], &launch,
						   1) == STATUS_OK,
	       "the most of the class is accepted");
	launch.dynamic_shared_byte_count = most_shared_bytes[card_index] + 1u;
        run_refused(cards[card_index], &launch, 1, "refused at launch 0 of 1", "one byte more than the class's most");
    }
    close_cards(cards, COUNT_OF(classes));
}

static bool work_is(uint64_t work_index, uint32_t kind, uint32_t card_index, uint64_t offset, uint64_t byte_count)
{
    struct test_host_memory_backend_work work;
    return test_host_memory_backend_work_at(work_index, &work) && work.kind == kind && work.card_index == card_index &&
           work.arena_byte_offset == offset && work.byte_count == byte_count;
}

static void test_log_and_stand_ins(void)
{
	struct device_card *cards[LOG_CARDS] = { NULL };
	struct stand_in_calls calls = { 0 };
	struct test_host_memory_backend_work work;
	void *block = NULL;
	uint32_t fill = 0;
	uint64_t ticket = 0;
	uint64_t digest = 0;
	uint32_t word = 0;
	context_text = "the log and a stand-in";
	test_host_memory_backend_reset(CARD_COUNT, CARD_MEMORY_BYTES);
	for (uint32_t card_index = 0; card_index < COUNT_OF(cards);
	     card_index++) {
		test_host_memory_backend_set_arithmetic_class(
			card_index, ARITHMETIC_CLASS_SM_120);
	}
    open_cards(cards, COUNT_OF(cards));
    expect(test_host_memory_backend.staging_allocate(STAGING_BYTES, &block) ==
			   STATUS_OK &&
		   prepare(cards[0], KERNEL_FILE_FILL, KERNEL_FILL_WORDS,
			   FILL_RECORD_BYTES, &fill, NULL) == STATUS_OK &&
		   prepare(cards[1], KERNEL_FILE_FILL, KERNEL_FILL_WORDS,
			   FILL_RECORD_BYTES, &fill, NULL) == STATUS_OK &&
		   test_host_memory_backend_set_stand_in(
			   KERNEL_FILE_FILL, KERNEL_FILL_WORDS, fill_stand_in,
			   &calls) == STATUS_OK,
	   "staging, the kernels and the stand-in");
    unsigned char *const staging = block;
    if (staging == NULL) {
        return;
    }
    memset(staging, UPLOADED_BYTE, STAGING_BYTES);
    const struct kernel_launch launch =
	    fill_launch(fill, FILL_WORD_BYTES, FILL_VALUE);
    expect(test_host_memory_backend.upload_begin(
		   cards[0], 0, staging, STAGING_BYTES, &ticket) == STATUS_OK &&
		   test_host_memory_backend.launch_run(cards[0], &launch, 1) ==
			   STATUS_OK &&
		   test_host_memory_backend.download_begin(
			   cards[0], 0, staging, STAGING_BYTES, &ticket) ==
			   STATUS_OK &&
		   test_host_memory_backend.upload_wait(cards[0], ticket, 0) ==
			   STATUS_OK &&
		   test_host_memory_backend.digest(cards[1], 0, STAGING_BYTES,
						   0, &digest) == STATUS_OK &&
		   test_host_memory_backend.launch_run(cards[1], &launch, 1) ==
			   STATUS_OK,
	   "an upload, a launch, a download, a digest, a launch of another card");
    memcpy(&word, staging + FILL_WORD_BYTES, sizeof(word));
    expect(staging[0] == UPLOADED_BYTE && word == FILL_VALUE &&
		   staging[(size_t)FILL_WORD_BYTES * (FILLED_WORDS + 1u)] ==
			   UPLOADED_BYTE,
	   "the stand-in ran behind the upload and the download saw what it wrote, and only that");
    expect(atomic_load(&calls.count) == LOG_CARDS && calls.last_card_index == 1, "the stand-in ran once on each card");
    expect(test_host_memory_backend_work_count() == LOGGED_WORK_COUNT &&
               work_is(LOGGED_UPLOAD, TEST_HOST_MEMORY_BACKEND_UPLOAD, 0, 0, STAGING_BYTES) &&
               work_is(LOGGED_FIRST_LAUNCH, TEST_HOST_MEMORY_BACKEND_LAUNCH, 0, 0, 0) &&
               work_is(LOGGED_DOWNLOAD, TEST_HOST_MEMORY_BACKEND_DOWNLOAD, 0, 0, STAGING_BYTES) &&
               work_is(LOGGED_DIGEST, TEST_HOST_MEMORY_BACKEND_DIGEST, 1, 0, STAGING_BYTES) &&
               work_is(LOGGED_SECOND_LAUNCH, TEST_HOST_MEMORY_BACKEND_LAUNCH, 1, 0, 0) &&
               test_host_memory_backend_work_at(LOGGED_FIRST_LAUNCH, &work) &&
               memcmp(&work.launch, &launch, sizeof(launch)) == 0 &&
               !test_host_memory_backend_work_at(LOGGED_WORK_COUNT, &work),
           "the log: every piece of work, in order, a launch with its bytes");
    context_text = "the table of stand-ins";
    char long_name[TEST_HOST_MEMORY_BACKEND_FUNCTION_NAME_CAPACITY + 1];
    memset(long_name, 'n', sizeof(long_name) - 1);
    long_name[sizeof(long_name) - 1] = '\0';
    expect(test_host_memory_backend_set_stand_in(NULL, KERNEL_FILL_WORDS,
						 fill_stand_in, &calls) ==
			   STATUS_INVALID_ARGUMENT &&
		   test_host_memory_backend_set_stand_in(
			   KERNEL_FILE_FILL, "", fill_stand_in, &calls) ==
			   STATUS_INVALID_ARGUMENT &&
		   test_host_memory_backend_set_stand_in(
			   KERNEL_FILE_FILL, long_name, fill_stand_in,
			   &calls) == STATUS_INVALID_ARGUMENT,
	   "a missing, empty or too long name");
    for (uint32_t index = 1; index < TEST_HOST_MEMORY_BACKEND_MAXIMUM_STAND_INS; index++) {
        long_name[index] = '\0';
	expect(test_host_memory_backend_set_stand_in(KERNEL_FILE_FILL,
						     long_name, fill_stand_in,
						     &calls) == STATUS_OK,
	       "stand-ins up to the most");
	long_name[index] = 'n';
    }
    expect(test_host_memory_backend_set_stand_in(
		   DIGEST_KERNEL_FILE, DIGEST_KERNEL_FUNCTION, fill_stand_in,
		   &calls) == STATUS_BUSY_VALUE &&
		   test_host_memory_backend_set_stand_in(
			   KERNEL_FILE_FILL, KERNEL_FILL_WORDS, fill_stand_in,
			   &calls) == STATUS_OK &&
		   test_host_memory_backend_set_stand_in(
			   DIGEST_KERNEL_FILE, DIGEST_KERNEL_FUNCTION, NULL,
			   NULL) == STATUS_OK,
	   "one more is BUSY; replacing one and removing a missing one are not");
    expect(test_host_memory_backend_set_stand_in(KERNEL_FILE_FILL,
						 KERNEL_FILL_WORDS, NULL,
						 NULL) == STATUS_OK &&
		   test_host_memory_backend.launch_run(cards[0], &launch, 1) ==
			   STATUS_OK &&
		   atomic_load(&calls.count) == LOG_CARDS,
	   "a stand-in taken away runs no more");
    close_cards(cards, COUNT_OF(cards));
    test_host_memory_backend.staging_free(staging);
}

static void test_faults(void)
{
	struct device_card *card = NULL;
	struct stand_in_calls calls = { 0 };
	void *staging = NULL;
	uint32_t fill = 0;
	uint64_t ticket = 0;
	uint64_t digest = 0;
	test_host_memory_backend_reset(CARD_COUNT, CARD_MEMORY_BYTES);
	test_host_memory_backend_set_arithmetic_class(0,
						      ARITHMETIC_CLASS_SM_80);
	open_cards(&card, 1);
	expect(test_host_memory_backend.staging_allocate(
		       STAGING_BYTES, &staging) == STATUS_OK &&
		       prepare(card, KERNEL_FILE_FILL, KERNEL_FILL_WORDS,
			       FILL_RECORD_BYTES, &fill, NULL) == STATUS_OK &&
		       test_host_memory_backend_set_stand_in(
			       KERNEL_FILE_FILL, KERNEL_FILL_WORDS,
			       fill_stand_in, &calls) == STATUS_OK,
	       "staging, the kernel and the stand-in");
	if (staging == NULL) {
		return;
	}
	const struct kernel_launch launches[FAULT_LIST_LENGTH] = {
		fill_launch(fill, 0, FILL_VALUE),
		fill_launch(fill, 0, FILL_VALUE),
		fill_launch(fill, 0, FILL_VALUE)
	};
	for (uint32_t way = 0; way <= 1; way++) {
		uint32_t kernel = 0;
		context_text = way == 0 ? "fail_launch" :
					  "a stand-in that faults";
		atomic_store(&calls.count, 0);
		calls.fails = way == 1;
		test_host_memory_backend_set_polls_before_an_upload_finishes(0,
									     1);
		test_host_memory_backend_fail_launch(
			0, way == 0 ? FAILING_LAUNCH : 0);
		test_host_memory_backend_fail_upload(
			0, way == 0 ? FAILING_TICKET :
				      0); /* later than the one in flight */
		const uint64_t work_before =
			test_host_memory_backend_work_count();
		expect(test_host_memory_backend.upload_begin(
			       card, 0, staging, STAGING_BYTES, &ticket) ==
				       STATUS_OK &&
			       test_host_memory_backend.launch_run(
				       card, launches, FAULT_LIST_LENGTH) ==
				       STATUS_OK,
		       "an upload in flight; the launches are queued and launch_run answers OK");
		expect(atomic_load(&calls.count) ==
				       (way == 0 ? FAILING_LAUNCH - 1u : 1u) &&
			       test_host_memory_backend_work_count() ==
				       work_before + 1u + FAULT_LIST_LENGTH,
		       "no stand-in runs from the faulting launch on; every launch is logged");
		expect(test_host_memory_backend.launch_wait(card, 0) ==
				       STATUS_DEVICE_ERROR &&
			       test_host_memory_backend.upload_wait(
				       card, ticket, 0) ==
				       STATUS_DEVICE_ERROR &&
			       test_host_memory_backend.upload_begin(
				       card, 0, staging, STAGING_BYTES,
				       &ticket) == STATUS_DEVICE_ERROR &&
			       test_host_memory_backend.launch_run(
				       card, launches, 1) ==
				       STATUS_DEVICE_ERROR &&
			       test_host_memory_backend.digest(
				       card, 0, STAGING_BYTES, 0, &digest) ==
				       STATUS_DEVICE_ERROR &&
			       prepare(card, KERNEL_FILE_FILL,
				       KERNEL_FILL_WORDS, FILL_RECORD_BYTES,
				       &kernel, NULL) == STATUS_DEVICE_ERROR,
		       "the card has failed: the wait, the upload in flight and all new work");
		test_host_memory_backend.staging_free(staging);
		expect(test_host_memory_backend_staging_block_count() == 0,
		       "the block of the failed upload may be freed");
		expect(test_host_memory_backend.staging_allocate(
			       STAGING_BYTES, &staging) == STATUS_OK,
		       "a new staging block");
		test_host_memory_backend.card_close(card, 0);
		test_host_memory_backend_set_polls_before_an_upload_finishes(0,
									     0);
		open_cards(&card, 1);
		if (staging == NULL) {
			break;
		}
		expect(test_host_memory_backend.launch_wait(card, 0) ==
				       STATUS_OK &&
			       test_host_memory_backend.upload_begin(
				       card, 0, staging, STAGING_BYTES,
				       &ticket) == STATUS_OK &&
			       ticket == 1u &&
			       test_host_memory_backend.upload_wait(
				       card, ticket, 0) == STATUS_OK &&
			       test_host_memory_backend.digest(
				       card, 0, STAGING_BYTES, 0, &digest) ==
				       STATUS_OK,
		       "opened again, it is sound: its tickets begin at 1, an upload finishes, a digest is taken");
		const result_status named_ticket_status =
			way == 0 ? STATUS_DEVICE_ERROR : STATUS_OK;
		expect(test_host_memory_backend.upload_begin(
			       card, 0, staging, STAGING_BYTES, &ticket) ==
				       STATUS_OK &&
			       test_host_memory_backend.digest(
				       card, 0, STAGING_BYTES, 0, &digest) ==
				       named_ticket_status &&
			       test_host_memory_backend.upload_wait(
				       card, ticket, 0) == named_ticket_status,
		       "the upload a test named fails on its own ticket, for a digest and a wait, and no other does");
		test_host_memory_backend.card_close(card, 0);
		open_cards(&card, 1);
		expect(prepare(card, KERNEL_FILE_FILL, KERNEL_FILL_WORDS,
			       FILL_RECORD_BYTES, &fill, NULL) == STATUS_OK,
		       "the kernel again");
	}
    close_cards(&card, 1);
    test_host_memory_backend.staging_free(staging);
}

static void test_full_log(void)
{
	struct device_card *card = NULL;
	struct test_host_memory_backend_work work;
	char logged[CAPTURE_CAPACITY];
	uint32_t fill = 0;
	uint64_t launched = 0;
	context_text = "a full log";
	test_host_memory_backend_reset(CARD_COUNT, CARD_MEMORY_BYTES);
	test_host_memory_backend_set_arithmetic_class(0,
						      ARITHMETIC_CLASS_SM_80);
	open_cards(&card, 1);
	struct kernel_launch *const launches =
		calloc(LAUNCH_BATCH, sizeof(*launches));
	expect(launches != NULL &&
		       prepare(card, KERNEL_FILE_FILL, KERNEL_FILL_WORDS,
			       FILL_RECORD_BYTES, &fill, NULL) == STATUS_OK,
	       "the launches and their kernel");
	capture_begin();
	while (launches != NULL &&
	       launched <
		       TEST_HOST_MEMORY_BACKEND_LOG_CAPACITY + LOG_OVERFLOW) {
		const uint64_t left = TEST_HOST_MEMORY_BACKEND_LOG_CAPACITY +
				      LOG_OVERFLOW - launched;
		const uint32_t count = left < LAUNCH_BATCH ? (uint32_t)left :
							     LAUNCH_BATCH;
		for (uint32_t index = 0; index < count; index++) {
			launches[index] = fill_launch(
				fill, 0, (uint32_t)(launched + index));
		}
		expect(test_host_memory_backend.launch_run(card, launches,
							   count) == STATUS_OK,
		       "launched");
		launched += count;
	}
    capture_end(logged, sizeof(logged));
    expect_logged(logged, NULL, "no line");
    const struct kernel_launch last_logged =
	    fill_launch(fill, 0, TEST_HOST_MEMORY_BACKEND_LOG_CAPACITY - 1u);
    expect(test_host_memory_backend_work_count() == TEST_HOST_MEMORY_BACKEND_LOG_CAPACITY + LOG_OVERFLOW &&
               test_host_memory_backend_work_at(TEST_HOST_MEMORY_BACKEND_LOG_CAPACITY - 1u, &work) &&
               memcmp(&work.launch, &last_logged, sizeof(last_logged)) == 0 &&
               !test_host_memory_backend_work_at(TEST_HOST_MEMORY_BACKEND_LOG_CAPACITY, &work) &&
               !test_host_memory_backend_work_at(UINT64_MAX, &work) && !test_host_memory_backend_work_at(0, NULL),
           "work beyond the capacity is counted and not logged");
    free(launches);
    close_cards(&card, 1);
    test_host_memory_backend_reset(CARD_COUNT, CARD_MEMORY_BYTES);
    open_cards(&card, 1);
    expect(test_host_memory_backend_work_count() == 0 &&
		   !test_host_memory_backend_work_at(0, &work) &&
		   prepare(card, KERNEL_FILE_FILL, KERNEL_FILL_WORDS,
			   FILL_RECORD_BYTES, &fill,
			   NULL) == STATUS_UNSUPPORTED,
	   "reset forgets the log and the class");
    close_cards(&card, 1);
}

struct thread_case {
	struct device_card *card;
	uint32_t kernel;
	uint32_t thread_index;
	bool failed;
};

static void *launch_on_one_card(void *argument)
{
    struct thread_case *const thread_case = argument;
    for (uint32_t round = 0; round < THREAD_ROUNDS; round++) {
        const uint32_t value = thread_case->thread_index * THREAD_VALUE_STRIDE + round;
	const struct kernel_launch launch =
		fill_launch(thread_case->kernel, 0, value);
	thread_case->failed =
		thread_case->failed ||
		test_host_memory_backend.launch_run(thread_case->card, &launch,
						    1) != STATUS_OK ||
		test_host_memory_backend.launch_wait(thread_case->card, 0) !=
			STATUS_OK;
    }
    return NULL;
}

static void test_threads(void)
{
	struct device_card *cards[THREAD_COUNT] = { NULL };
	struct thread_case cases[THREAD_COUNT];
	pthread_t threads[THREAD_COUNT];
	struct stand_in_calls calls = { 0 };
	uint32_t next_round[THREAD_COUNT] = { 0 };
	context_text = "threads on four cards";
	test_host_memory_backend_reset(CARD_COUNT, CARD_MEMORY_BYTES);
	for (uint32_t index = 0; index < THREAD_COUNT; index++) {
		test_host_memory_backend_set_arithmetic_class(
			index, ARITHMETIC_CLASS_SM_80);
	}
    open_cards(cards, THREAD_COUNT);
    expect(test_host_memory_backend_set_stand_in(
		   KERNEL_FILE_FILL, KERNEL_FILL_WORDS, fill_stand_in,
		   &calls) == STATUS_OK,
	   "the stand-in");
    for (uint32_t index = 0; index < THREAD_COUNT; index++) {
        cases[index] = (struct thread_case){cards[index], 0, index, false};
	expect(prepare(cards[index], KERNEL_FILE_FILL, KERNEL_FILL_WORDS,
		       FILL_RECORD_BYTES, &cases[index].kernel,
		       NULL) == STATUS_OK,
	       "the kernel of a card");
    }
    for (uint32_t index = 0; index < THREAD_COUNT; index++) {
        expect(pthread_create(&threads[index], NULL, launch_on_one_card, &cases[index]) == 0, "a thread starts");
    }
    for (uint32_t index = 0; index < THREAD_COUNT; index++) {
        expect(pthread_join(threads[index], NULL) == 0 && !cases[index].failed, "a thread ends, every call OK");
    }
    const uint64_t launch_count = (uint64_t)THREAD_COUNT * THREAD_ROUNDS;
    expect(test_host_memory_backend_work_count() == launch_count && atomic_load(&calls.count) == launch_count,
           "every launch is logged and ran");
    for (uint64_t work_index = 0; work_index < launch_count; work_index++) {
        struct test_host_memory_backend_work work;
	struct fill_record record;
	const bool is_logged = test_host_memory_backend_work_at(work_index, &work) && work.card_index < THREAD_COUNT;
        memcpy(&record, work.launch.argument, sizeof(record));
        const uint32_t card_index = is_logged ? work.card_index : 0;
        expect(is_logged && record.value == card_index * THREAD_VALUE_STRIDE + next_round[card_index],
               "each card's launches in its own order");
        next_round[card_index]++;
    }
    close_cards(cards, THREAD_COUNT);
}

int main(void)
{
    error_stream_descriptor = dup(STDERR_FILENO);
    capture_descriptor = mkstemp(capture_path);
    if (error_stream_descriptor < 0 || capture_descriptor < 0) {
        (void)printf("FAILED: the temporary file that stands in for the error stream could not be made\n");
        return EXIT_FAILURE;
    }
    test_class_zero();
    test_registry();
    test_launch_refusals();
    test_shared_memory();
    test_log_and_stand_ins();
    test_faults();
    test_full_log();
    test_threads();
    context_text = "the end of the test";
    expect(close(capture_descriptor) == 0 && unlink(capture_path) == 0 && close(error_stream_descriptor) == 0 &&
               test_host_memory_backend_open_card_count() == 0 && test_host_memory_backend_staging_block_count() == 0,
           "the temporary file is removed, no card is open and no staging block alive");
    if (failures != 0) {
        return EXIT_FAILURE;
    }
    (void)printf("host_memory_backend_test passed\n");
    return EXIT_SUCCESS;
}
