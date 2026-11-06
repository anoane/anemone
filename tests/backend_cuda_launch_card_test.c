/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * backend_cuda_launch_card_test: what a card runs, on the real cards: kernel_prepare, launch_run, launch_wait,
 * download_begin and format_supported of backend_seam.h, through the fill kernel (source/kernel_fill.cu), and every
 * refusal of theirs provoked once with its exact status. Two are not shown, because no card of this build can have
 * them: a kernel file that only another architecture has a build of, and a function of two arguments whose second
 * is no address.
 *
 * It needs the cards free (`make card_tests`): a card without room for the arena FAILS the test, it is never skipped.
 * What a card holds is known byte for byte: the test applies every fill it launches to a mirror in host memory, then
 * downloads the arena and compares the two. The error lines of the backend between the test's own lines are the
 * refusals and faults it provokes.
 *
 * No case makes a kernel fault: every record that reaches a card names words inside the arena (fill_launch sees to
 * it), and what the backend has to refuse never reaches the driver. The driver's failures are made by fault points.
 *
 * A process initialises the driver once, so a card that does not state the limits of a launch is asked for in a
 * child process, before the test's own first call of the backend.
 */
#include <inttypes.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include "anemone/backend_records.h"
#include "anemone/backend_seam.h"
#include "anemone/hooks.h"
#include "anemone/status.h"

#define TEST_TIME_LIMIT_SECONDS 900             /* the alarm ends the test here, whatever the driver does */
#define WAIT_LIMIT_NANOSECONDS 120000000000ull  /* two minutes: far above any copy or kernel of this test */
#define SHORTEST_WAIT_LIMIT_NANOSECONDS 1ull    /* a limit that is not 0 and has passed at the first look */
#define ARENA_BYTE_COUNT (64ull << 20u)
#define WORD_BYTES ((uint64_t)FILL_WORD_BYTES)
#define ARENA_WORD_COUNT (ARENA_BYTE_COUNT / WORD_BYTES)
#define MEMORY_SLACK_BYTES (ARENA_BYTE_COUNT / 2)       /* what a driver may keep; an arena left behind is more */
#define FILL_FILE KERNEL_FILE_FILL
#define FILL_FUNCTION KERNEL_FILL_WORDS
#define DIGEST_FILE "kernel_pack_digest"        /* a kernel file every card holds: its function takes three arguments */
#define DIGEST_FUNCTION "kernel_pack_digest"
#define RECORD_BYTE_COUNT ((uint32_t)sizeof(struct fill_record))
#define LARGER_RECORD_BYTE_COUNT (RECORD_BYTE_COUNT + FILL_WORD_BYTES)
#define SMALLER_RECORD_BYTE_COUNT (RECORD_BYTE_COUNT - FILL_WORD_BYTES)
#define PATTERN_MULTIPLIER 2654435761u          /* odd: the words of the pattern differ from one another */
#define PATTERN_INCREMENT 0x3C6EF372u
#define PART_FIRST_WORD 4099ull                 /* the first fill: neither end on a block of threads */
#define PART_WORD_COUNT 1000003ull
#define PART_VALUE 0xC0FFEE11u
#define SHAPE_FIRST_WORD 2000001ull             /* one range for each shape of grid and block */
#define SHAPE_WORD_COUNT 70001ull
#define SHAPE_VALUE 0x5A5A0000u
#define ORDER_FIRST_WORD 3000001ull             /* three fills that overlap, in one call */
#define ORDER_WORD_COUNT 400000ull
#define ORDER_UPPER_FIRST_WORD (ORDER_FIRST_WORD + ORDER_WORD_COUNT / 2)
#define ORDER_VALUE 0x0D0E0000u
#define SLOW_FIRST_WORD 5000000ull              /* one thread writes them all: a kernel that is still running */
#define SLOW_WORD_COUNT (1ull << 20u)
#define SLOW_VALUE 0x51000051u
#define WHOLE_VALUE 0x77E0A11Eu
#define REFUSED_FIRST_WORD 9000000ull           /* what a refused call would have filled */
#define REFUSED_WORD_COUNT 4096ull
#define REFUSED_VALUE 0xBADF111Eu
#define RING_FIRST_WORD 12000000ull             /* the copies that fill the ring: this many words each */
#define RING_WORD_COUNT 1024ull
#define RING_SOURCE_SHIFT_WORDS 7777ull         /* an upload of the ring brings the pattern of this far ahead */
#define COPY_DIRECTIONS 2u                      /* the ring is filled with downloads and uploads in turn */
#define MIDDLE_DESTINATION_BYTES (3ull << 20u)  /* where a part of the arena is downloaded to, inside its block */
#define MARKER_BYTE 0xE7                        /* what the block holds where no download wrote */
#define THREAD_COUNT 4u
#define THREAD_ROUNDS 16u
#define THREAD_WORD_COUNT (1ull << 18u)         /* each thread's words of the arena, and of the block it downloads to */
#define THREAD_VALUE 0x7D000000u
#define SHARED_BYTES_IGNORED 4096u              /* dynamic shared memory the fill kernel is given and does not use */
#define FAULT_LAUNCH_COUNT 3u                   /* the launches of a call whose first or second the driver rejects */
#define UNSTAGED_BYTE_COUNT 64
#define LONG_NAME_LENGTH 200                    /* far above any function name */
#define UNTOUCHED_TICKET 0xABCDull
#define UNTOUCHED_KERNEL 0xABCu
#define UNTOUCHED_COUNT 0xABCu
#define FIRST_TICKET 1ull
#define SECOND_CALL 2
#define THIRD_CALL 3                            /* of cuDeviceGetAttribute for a card: its first limit of a launch */
#define NEVER_FIRING_CALL UINT64_MAX            /* a fault armed so far ahead only counts how often it is reached */
#define HELD_MODULE_COUNT 2                     /* of a card that prepared the fill kernel: the digest's and its own */
#define CONTEXT_CAPACITY 160

enum dimension { DIMENSION_X, DIMENSION_Y, DIMENSION_Z, DIMENSION_COUNT };

/* The launches of the call that shows their order: all the words, their upper half and as many behind them, all
 * but the first word, and no word at all. */
enum order_launch { ORDER_ALL, ORDER_UPPER, ORDER_INNER, ORDER_NO_WORDS, ORDER_LAUNCH_COUNT };

/* The two launches of a call that must be refused. */
enum refused_call_launch { VALID_LAUNCH, REFUSED_LAUNCH, REFUSED_CALL_LAUNCH_COUNT };

/* A grid and a block the fill kernel must give the same bytes with, and dynamic shared memory it ignores. */
struct fill_shape {
    uint32_t grid[DIMENSION_COUNT];
    uint32_t block[DIMENSION_COUNT];
    uint32_t dynamic_shared_byte_count;
};

static const struct fill_shape shapes[] = {
    {{1, 1, 1}, {1, 1, 1}, 0},                  /* one thread writes every word */
    {{1, 1, 1}, {256, 1, 1}, 0},
    {{7, 1, 1}, {33, 1, 1}, 0},                 /* a thread count that divides nothing */
    {{2, 3, 1}, {16, 2, 2}, 0},                 /* three dimensions: only the threads of the first write */
    {{1024, 1, 1}, {256, 1, 1}, 0},             /* more threads than words */
    {{3, 1, 1}, {1024, 1, 1}, 0},
    {{5, 1, 1}, {64, 1, 1}, SHARED_BYTES_IGNORED},
};
#define SHAPE_COUNT ((uint32_t)(sizeof(shapes) / sizeof(shapes[0])))

/* The widest block of the cards this engine drives, in each dimension: together far more threads than a block has. */
static const uint32_t widest_block[DIMENSION_COUNT] = {1024, 1024, 64};

#if CHECKING_HOOKS_ENABLED
/* Each driver call of kernel_prepare behind the loading of the module, at the call number it is made to fail. */
struct prepare_fault {
    const char *fault_name;
    uint64_t call_number;
};

static const struct prepare_fault prepare_faults[] = {
    {"backend_cuda_cuModuleLoadData", 1},     {"backend_cuda_cuModuleGetFunction", 1},
    {"backend_cuda_cuFuncLoad", 1},           {"backend_cuda_cuFuncGetParamCount", 1},
    {"backend_cuda_cuFuncGetParamInfo", 1},   {"backend_cuda_cuFuncGetParamInfo", 2},
    {"backend_cuda_cuFuncGetAttribute", 1},   {"backend_cuda_cuCtxGetDevice", 1},
    {"backend_cuda_cuDeviceGetAttribute", 1}, {"backend_cuda_cuFuncGetAttribute", 2},
    {"backend_cuda_cuFuncSetAttribute", 1},   {"backend_cuda_cuFuncGetAttribute", 3},
};
#define PREPARE_FAULT_COUNT ((uint32_t)(sizeof(prepare_faults) / sizeof(prepare_faults[0])))
#endif

_Static_assert(DIMENSION_COUNT == LAUNCH_DIMENSIONS,
	       "a launch has three dimensions");
_Static_assert(sizeof(struct fill_record) <= LAUNCH_ARGUMENT_BYTES,
	       "the fill record fits a launch");
_Static_assert(ARENA_BYTE_COUNT % WORD_BYTES == 0, "the arena is whole words");
_Static_assert(THREAD_WORD_COUNT *BACKEND_MAXIMUM_CARDS <= ARENA_WORD_COUNT &&
		       THREAD_COUNT <= BACKEND_MAXIMUM_CARDS,
	       "every thread and every card has its own words");

static const struct backend_operations *const backend = &backend_cuda_value;
static int failures = 0;
static char context_text[CONTEXT_CAPACITY] = "the backend";    /* what the failure lines are about */
static uint32_t *pattern = NULL;                /* page-locked and never changed once filled: what is uploaded */
static uint32_t *received = NULL;               /* page-locked: what is downloaded */
static uint32_t *mirror = NULL;                 /* host memory: what the open arena must hold, word for word */

static void expect(bool condition, const char *what)
{
    if (!condition) {
        (void)fprintf(stderr, "FAILED: %s: %s\n", context_text, what);
        failures++;
    }
}

static void expect_status(result_status actual, result_status expected,
			  const char *what)
{
    if (actual != expected) {
	    (void)fprintf(stderr,
			  "FAILED: %s: %s: the status was \"%s\", not \"%s\"\n",
			  context_text, what, status_text(actual),
			  status_text(expected));
	    failures++;
    }
}

/* A launch of the fill kernel as kernel_fill.cu says it is launched. A range that is not inside the arena is a
 * failure of the test and becomes a fill of no words: no record of this test names memory outside the arena. */
static struct kernel_launch fill_launch(uint32_t kernel, uint64_t first_word,
					uint64_t word_count, uint32_t value)
{
    const bool is_inside = first_word <= ARENA_WORD_COUNT && word_count <= ARENA_WORD_COUNT - first_word;
    const uint64_t blocks_needed =
	    (word_count + FILL_THREADS_PER_BLOCK_VALUE - 1) /
	    FILL_THREADS_PER_BLOCK_VALUE;
    const struct fill_record record = {
	    .arena_byte_offset = is_inside ? first_word * WORD_BYTES : 0,
	    .word_count = is_inside ? word_count : 0,
	    .value = value,
	    .reserved_zero = 0
    };
    struct kernel_launch launch;

    expect(is_inside, "a fill of the test lies inside the arena");
    memset(&launch, 0, sizeof(launch));
    for (uint32_t dimension = 0; dimension < DIMENSION_COUNT; dimension++) {
        launch.grid[dimension] = 1;
        launch.block[dimension] = 1;
    }
    launch.kernel = kernel;
    launch.grid[DIMENSION_X] = blocks_needed == 0 ? 1 :
			       blocks_needed < FILL_MOST_BLOCKS ?
						    (uint32_t)blocks_needed :
						    FILL_MOST_BLOCKS;
    launch.block[DIMENSION_X] = FILL_THREADS_PER_BLOCK_VALUE;
    launch.argument_byte_count = RECORD_BYTE_COUNT;
    memcpy(launch.argument, &record, sizeof(record));
    return launch;
}

/* What a fill does to the arena, done to the mirror. */
static void mirror_fill(uint64_t first_word, uint64_t word_count, uint32_t value)
{
    for (uint64_t word_index = first_word; word_index < first_word + word_count; word_index++) {
        mirror[word_index] = value;
    }
}

/* Queues launches and waits until the card has run them. */
static void run_launches(struct device_card *card,
			 const struct kernel_launch *launches,
			 uint32_t launch_count)
{
	expect_status(backend->launch_run(card, launches, launch_count),
		      STATUS_OK, "launches are queued");
	expect_status(
		backend->launch_wait(card, WAIT_LIMIT_NANOSECONDS), STATUS_OK,
		"everything queued on the card finishes within the limit");
}

/* Downloads the whole arena and compares it with the mirror. */
static void expect_arena_is_mirror(struct device_card *card, const char *what)
{
    uint64_t ticket = UNTOUCHED_TICKET;
    uint64_t word_index = 0;

    expect_status(backend->download_begin(card, 0, received, ARENA_BYTE_COUNT,
					  &ticket),
		  STATUS_OK, "the download of the arena begins");
    if (ticket == UNTOUCHED_TICKET) {
        return;
    }
    expect_status(backend->upload_wait(card, ticket, WAIT_LIMIT_NANOSECONDS),
		  STATUS_OK,
		  "the download of the arena finishes within the limit");
    while (word_index < ARENA_WORD_COUNT && received[word_index] == mirror[word_index]) {
        word_index++;
    }
    if (word_index < ARENA_WORD_COUNT) {
        (void)fprintf(stderr, "FAILED: %s: %s: word %" PRIu64 " of the arena is 0x%08" PRIX32 ", not 0x%08" PRIX32 "\n",
                      context_text, what, word_index, received[word_index], mirror[word_index]);
        failures++;
    }
}

/* Opens the card with the arena of this test; NULL, and a failure, if it does not open. */
static struct device_card *open_card(uint32_t card_index)
{
	struct device_card *card = NULL;
	expect_status(backend->card_open(card_index, ARENA_BYTE_COUNT, &card),
		      STATUS_OK, "the card opens");
	expect(card != NULL, "an open card has a handle");
	return card;
}

/* Prepares the fill kernel on an open card; BACKEND_NO_KERNEL, and a failure, if it is not prepared. */
static uint32_t prepare_fill(struct device_card *card)
{
	uint32_t kernel = BACKEND_NO_KERNEL;
	expect_status(backend->kernel_prepare(card, FILL_FILE, FILL_FUNCTION,
					      RECORD_BYTE_COUNT, &kernel),
		      STATUS_OK, "the fill kernel is prepared");
	expect(kernel != BACKEND_NO_KERNEL, "a prepared kernel has a handle");
	return kernel;
}

/* Uploads the pattern to the whole arena, waits for it, and makes the mirror the pattern. */
static void upload_pattern(struct device_card *card)
{
    uint64_t ticket = UNTOUCHED_TICKET;
    expect_status(backend->upload_begin(card, 0, pattern, ARENA_BYTE_COUNT,
					&ticket),
		  STATUS_OK, "the upload of the pattern begins");
    expect_status(backend->upload_wait(card, ticket, WAIT_LIMIT_NANOSECONDS),
		  STATUS_OK,
		  "the upload of the pattern finishes within the limit");
    memcpy(mirror, pattern, ARENA_BYTE_COUNT);
}

#if CHECKING_HOOKS_ENABLED

/* Runs a case in a child process, and fails unless the child ends with EXIT_SUCCESS. No call of the backend has been
 * made yet, so the child inherits nothing of the driver. */
static void run_in_child(void (*test_case)(void), const char *what)
{
    int wait_status = 0;
    const pid_t child = fork();

    if (child == 0) {
        (void)alarm(TEST_TIME_LIMIT_SECONDS);
        test_case();
        _exit(failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE);
    }
    expect(child > 0 && waitpid(child, &wait_status, 0) == child && WIFEXITED(wait_status) &&
               WEXITSTATUS(wait_status) == EXIT_SUCCESS,
           what);
}

/* A card that does not tell how large a grid may be: the driver is not initialised, as for any other attribute. */
static void test_limits_unread(void)
{
    uint32_t card_count = UNTOUCHED_COUNT;

    expect(fault_arm("backend_cuda_cuDeviceGetAttribute", THIRD_CALL),
	   "the fault of the limit is armed");
    expect_status(
	    backend->card_count(&card_count), STATUS_DEVICE_ERROR,
	    "a card that does not state the limits of a launch is a device error");
    expect(fault_disarm() == THIRD_CALL && card_count == UNTOUCHED_COUNT,
	   "the initialisation stopped at the limit that was not read");
}

#endif

/* The two formats of the linear kernels on every card of an arithmetic class, and nothing else. */
static void test_formats(uint32_t card_count)
{
    for (uint32_t card_index = 0; card_index < card_count; card_index++) {
	    struct card_description_record description;
	    memset(&description, 0, sizeof(description));
	    expect_status(backend->card_describe(card_index, &description),
			  STATUS_OK, "a card is described");
	    for (int format = WEIGHT_FORMAT_NONE; format <= WEIGHT_FORMAT_COUNT;
		 format++) {
		    const bool is_decoded =
			    description.arithmetic_class != 0 &&
			    (format == WEIGHT_FORMAT_E2M1_SCALED ||
			     format == WEIGHT_FORMAT_E4M3_SCALED);
		    expect(backend->format_supported(
				   card_index,
				   (enum weight_format_code)format) ==
				   is_decoded,
			   "a card decodes the two scaled formats of its kernels and no other");
	    }
    }
    expect(!backend->format_supported(card_count, WEIGHT_FORMAT_E2M1_SCALED) &&
		   !backend->format_supported(UINT32_MAX,
					      WEIGHT_FORMAT_E4M3_SCALED),
	   "an index past the cards decodes nothing");
}

/* What the new calls refuse in place of a card. */
static void test_refusals_without_a_card(void)
{
	const struct kernel_launch launch = fill_launch(0, 0, 0, 0);
	uint64_t ticket = UNTOUCHED_TICKET;
	uint32_t kernel = UNTOUCHED_KERNEL;

	expect_status(backend->kernel_prepare(NULL, FILL_FILE, FILL_FUNCTION,
					      RECORD_BYTE_COUNT, &kernel),
		      STATUS_INVALID_ARGUMENT,
		      "a kernel is prepared on no card: refused");
	expect_status(backend->launch_run(NULL, &launch, 1),
		      STATUS_INVALID_ARGUMENT,
		      "a launch on no card is refused");
	expect_status(backend->launch_wait(NULL, 0), STATUS_INVALID_ARGUMENT,
		      "a wait on no card is refused");
	expect_status(
		backend->download_begin(NULL, 0, received, WORD_BYTES, &ticket),
		STATUS_INVALID_ARGUMENT, "a download from no card is refused");
	expect(ticket == UNTOUCHED_TICKET && kernel == UNTOUCHED_KERNEL,
	       "refused calls leave their outputs untouched");
}

/* Every refusal of kernel_prepare on an open card on which nothing is prepared yet, then the kernel itself. */
static uint32_t test_prepare(struct device_card *card)
{
    static char long_name[LONG_NAME_LENGTH + 1];
    uint32_t kernel = UNTOUCHED_KERNEL;
    uint32_t same_kernel = UNTOUCHED_KERNEL;

    memset(long_name, 'a', LONG_NAME_LENGTH);
    expect_status(backend->kernel_prepare(card, "kernel_of_no_build",
					  FILL_FUNCTION, RECORD_BYTE_COUNT,
					  &kernel),
		  STATUS_INVALID_ARGUMENT,
		  "a kernel file no architecture has a build of is refused");
    expect_status(backend->kernel_prepare(card, FILL_FILE,
					  "anemone_no_such_function",
					  RECORD_BYTE_COUNT, &kernel),
		  STATUS_INVALID_ARGUMENT,
		  "a function its kernel file does not have is refused");
    expect_status(backend->kernel_prepare(card, FILL_FILE, long_name,
					  RECORD_BYTE_COUNT, &kernel),
		  STATUS_INVALID_ARGUMENT,
		  "a function name longer than any kernel's is refused");
    expect_status(backend->kernel_prepare(card, FILL_FILE, "",
					  RECORD_BYTE_COUNT, &kernel),
		  STATUS_INVALID_ARGUMENT,
		  "a function name of no bytes is refused");
    expect_status(backend->kernel_prepare(card, DIGEST_FILE, DIGEST_FUNCTION,
					  RECORD_BYTE_COUNT, &kernel),
		  STATUS_INVALID_ARGUMENT,
		  "a function that does not take two arguments is refused");
    expect_status(backend->kernel_prepare(card, FILL_FILE, FILL_FUNCTION,
					  LARGER_RECORD_BYTE_COUNT, &kernel),
		  STATUS_INVALID_ARGUMENT,
		  "a record larger than the kernel's is refused");
    expect_status(backend->kernel_prepare(card, FILL_FILE, FILL_FUNCTION,
					  SMALLER_RECORD_BYTE_COUNT, &kernel),
		  STATUS_INVALID_ARGUMENT,
		  "a record smaller than the kernel's is refused");
    expect_status(backend->kernel_prepare(card, FILL_FILE, FILL_FUNCTION, 0,
					  &kernel),
		  STATUS_INVALID_ARGUMENT, "a record of no bytes is refused");
    expect_status(backend->kernel_prepare(card, FILL_FILE, FILL_FUNCTION,
					  LAUNCH_ARGUMENT_BYTES + 1, &kernel),
		  STATUS_INVALID_ARGUMENT,
		  "a record larger than a launch holds is refused");
    expect_status(backend->kernel_prepare(card, NULL, FILL_FUNCTION,
					  RECORD_BYTE_COUNT, &kernel),
		  STATUS_INVALID_ARGUMENT, "a missing kernel file is refused");
    expect_status(backend->kernel_prepare(card, FILL_FILE, NULL,
					  RECORD_BYTE_COUNT, &kernel),
		  STATUS_INVALID_ARGUMENT,
		  "a missing function name is refused");
    expect_status(backend->kernel_prepare(card, FILL_FILE, FILL_FUNCTION,
					  RECORD_BYTE_COUNT, NULL),
		  STATUS_INVALID_ARGUMENT,
		  "a prepare without an output is refused");
#if CHECKING_HOOKS_ENABLED
    /* a module loaded for a refused prepare is unloaded again, and the module a card holds is not loaded twice */
    expect(fault_arm("backend_cuda_cuModuleLoadData", NEVER_FIRING_CALL),
	   "the loads are counted");
    expect_status(backend->kernel_prepare(card, FILL_FILE, FILL_FUNCTION,
					  LARGER_RECORD_BYTE_COUNT, &kernel),
		  STATUS_INVALID_ARGUMENT,
		  "a record larger than the kernel's is refused again");
    expect(fault_disarm() == 1, "the kernel file of a first prepare is loaded");
    expect(fault_arm("backend_cuda_cuModuleUnload", NEVER_FIRING_CALL),
	   "the unloads are counted");
    expect_status(backend->kernel_prepare(card, FILL_FILE,
					  "anemone_no_such_function",
					  RECORD_BYTE_COUNT, &kernel),
		  STATUS_INVALID_ARGUMENT,
		  "a function its kernel file does not have is refused again");
    expect(fault_disarm() == 1,
	   "the module of a refused prepare is unloaded again");
    expect(fault_arm("backend_cuda_cuModuleLoadData", NEVER_FIRING_CALL),
	   "the loads are counted");
    expect_status(backend->kernel_prepare(card, DIGEST_FILE, DIGEST_FUNCTION,
					  RECORD_BYTE_COUNT, &kernel),
		  STATUS_INVALID_ARGUMENT,
		  "the digest kernel is refused again");
    expect(fault_disarm() == 0,
	   "a kernel file the card holds since it was opened is not loaded again");
    /* the table of kernels, made full */
    expect(fault_arm("backend_cuda_kernel_table_full", 1),
	   "the fault of the full table is armed");
    expect_status(backend->kernel_prepare(card, FILL_FILE, FILL_FUNCTION,
					  RECORD_BYTE_COUNT, &kernel),
		  STATUS_UNSUPPORTED,
		  "one kernel more than a card prepares is unsupported");
    expect(fault_disarm() == 1, "the fault of the full table was reached");
#endif
    expect(kernel == UNTOUCHED_KERNEL, "refused prepares leave their output untouched");

    kernel = prepare_fill(card);
#if CHECKING_HOOKS_ENABLED
    expect(fault_arm("backend_cuda_cuModuleLoadData", NEVER_FIRING_CALL),
	   "the loads are counted");
#endif
    expect_status(backend->kernel_prepare(card, FILL_FILE, FILL_FUNCTION,
					  RECORD_BYTE_COUNT, &same_kernel),
		  STATUS_OK, "the fill kernel is prepared a second time");
#if CHECKING_HOOKS_ENABLED
    expect(fault_disarm() == 0,
	   "a kernel prepared a second time loads nothing");
#endif
    expect(same_kernel == kernel, "preparing the same kernel again gives the same handle");
    same_kernel = UNTOUCHED_KERNEL;
    expect_status(
	    backend->kernel_prepare(card, FILL_FILE, FILL_FUNCTION,
				    LARGER_RECORD_BYTE_COUNT, &same_kernel),
	    STATUS_INVALID_ARGUMENT,
	    "a prepared kernel asked for with another record size is refused");
    expect(same_kernel == UNTOUCHED_KERNEL, "and its handle is not given");
    return kernel;
}

/* A fill behind an upload and before a download, shapes of grid and block, order inside one call, a slow kernel. */
static void test_fills(struct device_card *card, uint32_t kernel)
{
	struct kernel_launch shape_launches[SHAPE_COUNT];
	struct kernel_launch order_launches[ORDER_LAUNCH_COUNT];
	struct kernel_launch launch = fill_launch(kernel, PART_FIRST_WORD,
						  PART_WORD_COUNT, PART_VALUE);
	uint64_t upload_ticket = UNTOUCHED_TICKET;
	uint64_t download_ticket = UNTOUCHED_TICKET;
	result_status status = STATUS_OK;

	/* one queue: the upload, the fill and the download are begun without a wait between them */
	expect_status(backend->upload_begin(card, 0, pattern, ARENA_BYTE_COUNT,
					    &upload_ticket),
		      STATUS_OK, "the upload of the pattern begins");
	expect_status(backend->launch_run(card, &launch, 1), STATUS_OK,
		      "a fill is queued behind the upload");
	expect_status(backend->download_begin(card, 0, received,
					      ARENA_BYTE_COUNT,
					      &download_ticket),
		      STATUS_OK, "a download is queued behind the fill");
	expect(upload_ticket == FIRST_TICKET &&
		       download_ticket == FIRST_TICKET + 1,
	       "a download takes the next ticket of the card's one sequence");
	expect_status(backend->upload_wait(card, download_ticket,
					   WAIT_LIMIT_NANOSECONDS),
		      STATUS_OK, "the download finishes within the limit");
	expect_status(
		backend->upload_wait(card, upload_ticket, 0), STATUS_OK,
		"a finished download means the upload before it has finished");
	memcpy(mirror, pattern, ARENA_BYTE_COUNT);
	mirror_fill(PART_FIRST_WORD, PART_WORD_COUNT, PART_VALUE);
	expect(memcmp(received, mirror, ARENA_BYTE_COUNT) == 0,
	       "a fill runs behind the upload and before the download queued around it: only its words changed");

	/* any grid and block give the same bytes: each shape fills a range of its own */
	for (uint32_t shape_index = 0; shape_index < SHAPE_COUNT;
	     shape_index++) {
		const uint64_t first_word =
			SHAPE_FIRST_WORD + shape_index * (SHAPE_WORD_COUNT + 1);
		struct kernel_launch *const shaped =
			&shape_launches[shape_index];
		*shaped = fill_launch(kernel, first_word, SHAPE_WORD_COUNT,
				      SHAPE_VALUE + shape_index);
		memcpy(shaped->grid, shapes[shape_index].grid,
		       sizeof(shaped->grid));
		memcpy(shaped->block, shapes[shape_index].block,
		       sizeof(shaped->block));
		shaped->dynamic_shared_byte_count =
			shapes[shape_index].dynamic_shared_byte_count;
		mirror_fill(first_word, SHAPE_WORD_COUNT,
			    SHAPE_VALUE + shape_index);
	}
    run_launches(card, shape_launches, SHAPE_COUNT);
    expect_arena_is_mirror(card, "every grid and block fills the same words");

    /* the launches of one call keep their order: where fills overlap, the last one wins; and a fill of no words */
    order_launches[ORDER_ALL] = fill_launch(kernel, ORDER_FIRST_WORD, ORDER_WORD_COUNT, ORDER_VALUE + ORDER_ALL);
    order_launches[ORDER_UPPER] =
        fill_launch(kernel, ORDER_UPPER_FIRST_WORD, ORDER_WORD_COUNT, ORDER_VALUE + ORDER_UPPER);
    order_launches[ORDER_INNER] =
        fill_launch(kernel, ORDER_FIRST_WORD + 1, ORDER_WORD_COUNT - 1, ORDER_VALUE + ORDER_INNER);
    order_launches[ORDER_NO_WORDS] = fill_launch(kernel, ARENA_WORD_COUNT, 0, ORDER_VALUE + ORDER_NO_WORDS);
    mirror_fill(ORDER_FIRST_WORD, ORDER_WORD_COUNT, ORDER_VALUE + ORDER_ALL);
    mirror_fill(ORDER_UPPER_FIRST_WORD, ORDER_WORD_COUNT, ORDER_VALUE + ORDER_UPPER);
    mirror_fill(ORDER_FIRST_WORD + 1, ORDER_WORD_COUNT - 1, ORDER_VALUE + ORDER_INNER);
    run_launches(card, order_launches, ORDER_LAUNCH_COUNT);
    expect_arena_is_mirror(card, "the launches of one call run in their order");

    /* a kernel that is still running: one thread writes a million words, a download is queued behind it, and the
     * wait is asked at once. A wait that answers OK has covered both, so the download then reports finished to a
     * wait that only asks; behind a wait that answered too early it would still be in flight */
    launch = fill_launch(kernel, SLOW_FIRST_WORD, SLOW_WORD_COUNT, SLOW_VALUE);
    launch.grid[DIMENSION_X] = 1;
    launch.block[DIMENSION_X] = 1;
    mirror_fill(SLOW_FIRST_WORD, SLOW_WORD_COUNT, SLOW_VALUE);
    expect_status(backend->launch_run(card, &launch, 1), STATUS_OK,
		  "a slow fill is queued");
    download_ticket = UNTOUCHED_TICKET;
    expect_status(backend->download_begin(
			  card, SLOW_FIRST_WORD * WORD_BYTES, received,
			  SLOW_WORD_COUNT * WORD_BYTES, &download_ticket),
		  STATUS_OK, "a download is queued behind the slow fill");
    status = backend->launch_wait(card, 0);
    expect(status == STATUS_OK || status == STATUS_TIMED_OUT,
	   "a wait with limit 0 only asks");
    /* the timeout cannot be demanded, because the kernel may have finished: the line shows whether this run saw it */
    (void)printf(
	    "backend_cuda_launch_card_test: %s: the wait with limit 0 behind the slow fill answered \"%s\"\n",
	    context_text, status_text(status));
    status = backend->launch_wait(card, SHORTEST_WAIT_LIMIT_NANOSECONDS);
    expect(status == STATUS_OK || status == STATUS_TIMED_OUT,
	   "a wait that runs out of time says so, and the card stays usable");
    expect_status(backend->launch_wait(card, WAIT_LIMIT_NANOSECONDS), STATUS_OK,
		  "the slow fill finishes within the limit");
    expect_status(
	    backend->upload_wait(card, download_ticket, 0), STATUS_OK,
	    "a wait that found the card done has covered the download queued before it");
    expect(memcmp(received, mirror + SLOW_FIRST_WORD, SLOW_WORD_COUNT * WORD_BYTES) == 0,
           "the download behind the slow fill holds every word of it");
    expect_arena_is_mirror(card, "one thread fills its words alone");

    /* the whole arena, with the grid kernel_fill.cu gives a large range */
    launch = fill_launch(kernel, 0, ARENA_WORD_COUNT, WHOLE_VALUE);
    mirror_fill(0, ARENA_WORD_COUNT, WHOLE_VALUE);
    expect(launch.grid[DIMENSION_X] == FILL_MOST_BLOCKS,
	   "a large range has the widest grid of a fill");
    run_launches(card, &launch, 1);
    expect_arena_is_mirror(card, "a fill of the whole arena leaves no other word");
}

/* One launch that launch_run must refuse, queued behind one that would change the arena: neither is queued. */
static void expect_refused(struct device_card *card, uint32_t kernel,
			   const struct kernel_launch *refused,
			   const char *what)
{
	struct kernel_launch launches[REFUSED_CALL_LAUNCH_COUNT];
	launches[VALID_LAUNCH] = fill_launch(kernel, REFUSED_FIRST_WORD,
					     REFUSED_WORD_COUNT, REFUSED_VALUE);
	launches[REFUSED_LAUNCH] = *refused;
	expect_status(backend->launch_run(card, launches,
					  REFUSED_CALL_LAUNCH_COUNT),
		      STATUS_INVALID_ARGUMENT, what);
}

/* Every refusal of launch_run on an open card with one prepared kernel: nothing is queued, the card stays usable. */
static void test_launch_refusals(struct device_card *card, uint32_t kernel)
{
	const struct kernel_launch valid = fill_launch(
		kernel, REFUSED_FIRST_WORD, REFUSED_WORD_COUNT, REFUSED_VALUE);
	struct kernel_launch launch = valid;

	launch.kernel = BACKEND_NO_KERNEL;
	expect_refused(card, kernel, &launch,
		       "what stands in place of a handle is refused");
	launch.kernel =
		kernel +
		1; /* the card has prepared one kernel: the next number names none */
	expect_refused(
		card, kernel, &launch,
		"a handle that was not prepared on this card is refused");
	launch = valid;
	launch.argument_byte_count = LARGER_RECORD_BYTE_COUNT;
	expect_refused(card, kernel, &launch,
		       "a record size that is not the kernel's is refused");
	launch.argument_byte_count = 0;
	expect_refused(card, kernel, &launch,
		       "a record of no bytes is refused");
	launch.argument_byte_count = LAUNCH_ARGUMENT_BYTES + 1;
	expect_refused(card, kernel, &launch,
		       "a record larger than a launch holds is refused");
	for (uint32_t dimension = 0; dimension < DIMENSION_COUNT; dimension++) {
		launch = valid;
		launch.grid[dimension] = 0;
		expect_refused(card, kernel, &launch,
			       "a grid dimension of zero is refused");
		launch = valid;
		launch.block[dimension] = 0;
		expect_refused(card, kernel, &launch,
			       "a block dimension of zero is refused");
		launch = valid;
		launch.grid[dimension] = UINT32_MAX;
		expect_refused(card, kernel, &launch,
			       "a grid dimension above the card's is refused");
		launch = valid;
		launch.block[dimension] = UINT32_MAX;
		expect_refused(card, kernel, &launch,
			       "a block dimension above the card's is refused");
	}
    launch = valid;
    memcpy(launch.block, widest_block, sizeof(widest_block));
    expect_refused(card, kernel, &launch, "a block of more threads than the card allows is refused");
    launch = valid;
    launch.dynamic_shared_byte_count = UINT32_MAX;
    expect_refused(card, kernel, &launch, "more shared memory than the card allows is refused");
    launch = valid;
    launch.branch = LAUNCH_SIDE_MORE + 1u;
    expect_refused(card, kernel, &launch, "a reserved member that is not zero is refused");
    launch = valid;
    launch.argument[RECORD_BYTE_COUNT] = 1;
    expect_refused(card, kernel, &launch, "a byte right behind the record that is not zero is refused");
    launch = valid;
    launch.argument[LAUNCH_ARGUMENT_BYTES - 1] = 1;
    expect_refused(card, kernel, &launch, "a last byte of the argument that is not zero is refused");
    expect_status(backend->launch_run(card, NULL, 1), STATUS_INVALID_ARGUMENT,
		  "a missing table of launches is refused");
    expect_status(backend->launch_run(card, &valid, 0), STATUS_INVALID_ARGUMENT,
		  "a count of no launches is refused");
    expect_status(backend->launch_wait(card, WAIT_LIMIT_NANOSECONDS), STATUS_OK,
		  "a card that refused launches has nothing to wait for");
    expect_arena_is_mirror(card, "a refused call queued nothing: also not the valid launch before the refused one");
}

static bool bytes_are_markers(const unsigned char *bytes, uint64_t byte_count)
{
    uint64_t byte_index = 0;
    while (byte_index < byte_count && bytes[byte_index] == MARKER_BYTE) {
        byte_index++;
    }
    return byte_index == byte_count;
}

/* Every refusal of download_begin on an open card, a download into the middle of a block, and the full ring. */
static void test_download_rules(struct device_card *card)
{
    unsigned char unstaged[UNSTAGED_BYTE_COUNT] = {0};
    unsigned char *const received_bytes = (unsigned char *)received;
    const unsigned char *const mirror_bytes = (const unsigned char *)mirror;
    const uint64_t part_byte_offset = PART_FIRST_WORD * WORD_BYTES;
    const uint64_t part_byte_count = PART_WORD_COUNT * WORD_BYTES;
    uint64_t ticket = UNTOUCHED_TICKET;
    uint64_t last_ticket = 0;
    uint32_t wrong_copy_count = 0;

    upload_pattern(card);                       /* every word differs from its neighbours again */
    expect_status(backend->download_begin(card, 0, received, 0, &ticket),
		  STATUS_INVALID_ARGUMENT, "a download of no bytes is refused");
    expect_status(backend->download_begin(card, ARENA_BYTE_COUNT - WORD_BYTES,
					  received, WORD_BYTES + WORD_BYTES,
					  &ticket),
		  STATUS_INVALID_ARGUMENT,
		  "a download across the end of the arena is refused");
    expect_status(backend->download_begin(card, ARENA_BYTE_COUNT + 1, received,
					  WORD_BYTES, &ticket),
		  STATUS_INVALID_ARGUMENT,
		  "a download from behind the arena is refused");
    expect_status(backend->download_begin(card, UINT64_MAX, received,
					  WORD_BYTES, &ticket),
		  STATUS_INVALID_ARGUMENT,
		  "a download whose range wraps is refused");
    expect_status(backend->download_begin(card, 0, unstaged, sizeof(unstaged),
					  &ticket),
		  STATUS_INVALID_ARGUMENT,
		  "a destination outside every staging block is refused");
    expect_status(backend->download_begin(card, 0, NULL, WORD_BYTES, &ticket),
		  STATUS_INVALID_ARGUMENT, "a missing destination is refused");
    expect_status(
	    backend->download_begin(
		    card, 0, received_bytes + ARENA_BYTE_COUNT - WORD_BYTES,
		    WORD_BYTES + WORD_BYTES, &ticket),
	    STATUS_INVALID_ARGUMENT,
	    "a destination across the end of its staging block is refused");
    expect_status(backend->download_begin(card, 0, received, WORD_BYTES, NULL),
		  STATUS_INVALID_ARGUMENT,
		  "a download without an output for its ticket is refused");
    expect(ticket == UNTOUCHED_TICKET, "refused downloads leave their ticket untouched");

    /* a part of the arena arrives where it is asked for, and no other byte of the block changes */
    memset(received, MARKER_BYTE, ARENA_BYTE_COUNT);
    expect_status(
	    backend->download_begin(card, part_byte_offset,
				    received_bytes + MIDDLE_DESTINATION_BYTES,
				    part_byte_count, &ticket),
	    STATUS_OK, "the download of a part begins");
    expect_status(backend->upload_wait(card, ticket, WAIT_LIMIT_NANOSECONDS),
		  STATUS_OK,
		  "the download of a part finishes within the limit");
    expect(memcmp(received_bytes + MIDDLE_DESTINATION_BYTES, mirror_bytes + part_byte_offset, part_byte_count) == 0,
           "a downloaded part holds the bytes of its range of the arena");
    expect(bytes_are_markers(received_bytes, MIDDLE_DESTINATION_BYTES) &&
               bytes_are_markers(received_bytes + MIDDLE_DESTINATION_BYTES + part_byte_count,
                                 ARENA_BYTE_COUNT - MIDDLE_DESTINATION_BYTES - part_byte_count),
           "a download writes its destination range only");

    /* as many copies as may be in flight, of both directions and none waited for; one more is refused as busy */
    for (uint32_t copy_index = 0;
	 copy_index < BACKEND_MAXIMUM_UPLOADS_IN_FLIGHT; copy_index++) {
	    const uint64_t first_word =
		    RING_FIRST_WORD + copy_index * RING_WORD_COUNT;
	    const uint32_t *const shifted_pattern =
		    pattern + first_word + RING_SOURCE_SHIFT_WORDS;
	    const result_status status =
		    copy_index % COPY_DIRECTIONS == 0 ?
			    backend->download_begin(
				    card, first_word * WORD_BYTES,
				    received + first_word,
				    RING_WORD_COUNT * WORD_BYTES,
				    &last_ticket) :
			    backend->upload_begin(card, first_word * WORD_BYTES,
						  shifted_pattern,
						  RING_WORD_COUNT * WORD_BYTES,
						  &last_ticket);
	    expect_status(status, STATUS_OK,
			  "a copy begins while the ring has room");
	    if (copy_index % COPY_DIRECTIONS != 0) {
		    memcpy(mirror + first_word, shifted_pattern,
			   RING_WORD_COUNT * WORD_BYTES);
	    }
    }
    ticket = UNTOUCHED_TICKET;
    expect_status(
	    backend->download_begin(card, 0, received, WORD_BYTES, &ticket),
	    STATUS_BUSY_VALUE,
	    "one download more than copies may be in flight is refused as busy");
    expect(ticket == UNTOUCHED_TICKET, "a download refused as busy takes no ticket");
    expect_status(
	    backend->upload_wait(card, last_ticket, WAIT_LIMIT_NANOSECONDS),
	    STATUS_OK, "the copies of a full ring finish within the limit");
    for (uint32_t copy_index = 0;
	 copy_index < BACKEND_MAXIMUM_UPLOADS_IN_FLIGHT; copy_index++) {
	    const uint64_t first_word =
		    RING_FIRST_WORD + copy_index * RING_WORD_COUNT;
	    const bool is_download = copy_index % COPY_DIRECTIONS == 0;
	    wrong_copy_count +=
		    is_download && memcmp(received + first_word,
					  mirror + first_word,
					  RING_WORD_COUNT * WORD_BYTES) != 0 ?
			    1u :
			    0u;
    }
    expect(wrong_copy_count == 0, "each download of the ring holds the words of its own range");
    expect_arena_is_mirror(card, "uploads and downloads between each other leave what the uploads brought");
}

struct fill_thread {
    pthread_t thread;
    struct device_card *card;
    uint32_t kernel;
    uint32_t slice_index;       /* its words of the arena and of the block it downloads to */
    uint32_t wrong_answer_count;
};

static uint32_t thread_value(uint32_t slice_index, uint32_t round)
{
    return THREAD_VALUE + slice_index * THREAD_ROUNDS + round;
}

/* Fills its own words of the arena, queues their download behind the fill, waits for the card and looks at every
 * word, round after round. The download is then only asked about, with a limit of 0: a wait for the card that
 * answered OK has covered it, also while other threads launch and wait on the same card. */
static void *fill_thread_run(void *argument)
{
    struct fill_thread *const self = argument;
    const uint64_t first_word = self->slice_index * THREAD_WORD_COUNT;
    const uint32_t *const slice = received + first_word;

    for (uint32_t round = 0; round < THREAD_ROUNDS; round++) {
        const uint32_t value = thread_value(self->slice_index, round);
	const struct kernel_launch launch =
		fill_launch(self->kernel, first_word, THREAD_WORD_COUNT, value);
	uint64_t ticket = 0;
        uint64_t word_index = 0;
	if (backend->launch_run(self->card, &launch, 1) != STATUS_OK ||
	    backend->download_begin(
		    self->card, first_word * WORD_BYTES, received + first_word,
		    THREAD_WORD_COUNT * WORD_BYTES, &ticket) != STATUS_OK ||
	    backend->launch_wait(self->card, WAIT_LIMIT_NANOSECONDS) !=
		    STATUS_OK ||
	    backend->upload_wait(self->card, ticket, 0) != STATUS_OK) {
		self->wrong_answer_count++;
		continue;
	}
	while (word_index < THREAD_WORD_COUNT && slice[word_index] == value) {
            word_index++;
        }
        self->wrong_answer_count += word_index < THREAD_WORD_COUNT ? 1u : 0u;
    }
    return NULL;
}

/* Starts one thread for each of the first thread_count records and joins them; the wrong answers of all of them. */
static uint32_t run_fill_threads(struct fill_thread *threads, uint32_t thread_count)
{
    uint32_t started_count = 0;
    uint32_t wrong_answer_count = 0;

    while (started_count < thread_count &&
           pthread_create(&threads[started_count].thread, NULL, fill_thread_run, &threads[started_count]) == 0) {
        started_count++;
    }
    expect(started_count == thread_count, "the fill threads start");
    for (uint32_t thread_index = 0; thread_index < started_count; thread_index++) {
        expect(pthread_join(threads[thread_index].thread, NULL) == 0, "a fill thread is joined");
        wrong_answer_count += threads[thread_index].wrong_answer_count;
    }
    return wrong_answer_count;
}

/* Any thread may call: several launch, wait and download on the same card at once, each in words of its own. */
static void test_launches_from_threads(struct device_card *card,
				       uint32_t kernel)
{
    struct fill_thread threads[THREAD_COUNT];

    memset(threads, 0, sizeof(threads));
    for (uint32_t thread_index = 0; thread_index < THREAD_COUNT; thread_index++) {
        threads[thread_index].card = card;
        threads[thread_index].kernel = kernel;
        threads[thread_index].slice_index = thread_index;
        mirror_fill(thread_index * THREAD_WORD_COUNT, THREAD_WORD_COUNT, thread_value(thread_index, THREAD_ROUNDS - 1));
    }
    expect(run_fill_threads(threads, THREAD_COUNT) == 0,
           "launches from several threads at once each fill their words, and each wait covers its own launch");
    expect_arena_is_mirror(card, "the threads left their last fills and touched no other word");
}

#if CHECKING_HOOKS_ENABLED

/* A card whose driver failed begins no new work: only the refusals of arguments and the close are left. kernel is
 * the fill kernel if the card had prepared it, else BACKEND_NO_KERNEL. */
static void expect_failed_card(struct device_card *card, uint32_t kernel)
{
	const struct kernel_launch launch = fill_launch(kernel, 0, 1, 0);
	uint64_t ticket = UNTOUCHED_TICKET;
	uint32_t prepared = UNTOUCHED_KERNEL;

	expect_status(backend->upload_begin(card, 0, pattern, WORD_BYTES,
					    &ticket),
		      STATUS_DEVICE_ERROR, "a failed card begins no upload");
	expect_status(backend->download_begin(card, 0, received, WORD_BYTES,
					      &ticket),
		      STATUS_DEVICE_ERROR, "a failed card begins no download");
	expect_status(
		backend->kernel_prepare(card, FILL_FILE, FILL_FUNCTION,
					RECORD_BYTE_COUNT, &prepared),
		STATUS_DEVICE_ERROR,
		"a failed card prepares no kernel, also not one it has prepared");
	expect_status(
		backend->launch_wait(card, WAIT_LIMIT_NANOSECONDS),
		STATUS_DEVICE_ERROR,
		"a wait on a failed card reports the device error, at once");
	expect_status(backend->launch_run(card, &launch, 1),
		      kernel == BACKEND_NO_KERNEL ? STATUS_INVALID_ARGUMENT :
						    STATUS_DEVICE_ERROR,
		      "a failed card queues no launch");
	expect_status(
		backend->launch_run(card, NULL, 1), STATUS_INVALID_ARGUMENT,
		"the refusals of arguments come first on a failed card too");
	expect(ticket == UNTOUCHED_TICKET && prepared == UNTOUCHED_KERNEL,
	       "a failed card leaves outputs untouched");
	backend->card_close(card, WAIT_LIMIT_NANOSECONDS);
}

/* Each place where the driver can fail a prepare, a launch, a wait or a download, made to fail once. */
static void test_driver_faults(uint32_t card_index)
{
	struct kernel_launch launches[FAULT_LAUNCH_COUNT];
	struct device_card *card = NULL;
	uint64_t ticket = UNTOUCHED_TICKET;
	uint32_t kernel = UNTOUCHED_KERNEL;

	/* a prepare the driver fails, at each of its calls: nothing is prepared and the card has failed */
	for (uint32_t fault_index = 0; fault_index < PREPARE_FAULT_COUNT;
	     fault_index++) {
		const struct prepare_fault *const fault =
			&prepare_faults[fault_index];
		card = open_card(card_index);
		expect(fault_arm(fault->fault_name, fault->call_number),
		       "a fault of the prepare is armed");
		expect_status(
			backend->kernel_prepare(card, FILL_FILE, FILL_FUNCTION,
						RECORD_BYTE_COUNT, &kernel),
			STATUS_DEVICE_ERROR,
			"a prepare the driver fails is a device error");
		expect(fault_disarm() == fault->call_number &&
			       kernel == UNTOUCHED_KERNEL,
		       "the prepare stopped at the call that failed and gave no handle");
		expect_failed_card(card, BACKEND_NO_KERNEL);
	}

    /* a launch the driver rejects: the first of a call, then the second of three, which leaves the third unasked */
    for (uint64_t failing_call = 1; failing_call <= SECOND_CALL; failing_call++) {
        card = open_card(card_index);
        kernel = prepare_fill(card);
        for (uint32_t launch_index = 0; launch_index < FAULT_LAUNCH_COUNT; launch_index++) {
            launches[launch_index] = fill_launch(kernel, REFUSED_FIRST_WORD, REFUSED_WORD_COUNT, REFUSED_VALUE);
        }
	expect(fault_arm("backend_cuda_cuLaunchKernel", failing_call),
	       "the fault of the launch is armed");
	expect_status(backend->launch_run(card, launches, FAULT_LAUNCH_COUNT),
		      STATUS_DEVICE_ERROR,
		      "a launch the driver rejects is a device error");
	expect(fault_disarm() == failing_call,
	       "no launch is handed to the driver behind a rejected one");
	expect_failed_card(card, kernel);
    }

    /* a card whose context cannot be made current for a launch has reported a device error: it has failed */
    card = open_card(card_index);
    kernel = prepare_fill(card);
    launches[0] = fill_launch(kernel, REFUSED_FIRST_WORD, REFUSED_WORD_COUNT, REFUSED_VALUE);
    expect(fault_arm("backend_cuda_cuCtxPushCurrent", 1),
	   "the fault of the context is armed");
    expect_status(
	    backend->launch_run(card, launches, 1), STATUS_DEVICE_ERROR,
	    "a launch for which the card's context cannot be made current is a device error");
    expect(fault_disarm() == 1, "the fault of the context was reached");
    expect_failed_card(card, kernel);

    /* a wait whose event the driver does not record, then one whose query the driver fails */
    card = open_card(card_index);
    kernel = prepare_fill(card);
    expect(fault_arm("backend_cuda_cuEventRecord", 1),
	   "the fault of the event is armed");
    expect_status(backend->launch_wait(card, WAIT_LIMIT_NANOSECONDS),
		  STATUS_DEVICE_ERROR,
		  "a wait whose event is not recorded is a device error");
    expect(fault_disarm() == 1, "the fault of the event was reached");
    expect_failed_card(card, kernel);
    card = open_card(card_index);
    kernel = prepare_fill(card);
    expect(fault_arm("backend_cuda_cuEventQuery", 1),
	   "the fault of the query is armed");
    expect_status(backend->launch_wait(card, WAIT_LIMIT_NANOSECONDS),
		  STATUS_DEVICE_ERROR,
		  "a wait whose query fails is a device error");
    expect(fault_disarm() == 1, "the fault of the query was reached");
    expect_failed_card(card, kernel);

    /* a download the driver refuses takes no ticket */
    card = open_card(card_index);
    expect(fault_arm("backend_cuda_cuMemcpyDtoHAsync", 1),
	   "the fault of the download is armed");
    expect_status(backend->download_begin(card, 0, received, WORD_BYTES,
					  &ticket),
		  STATUS_DEVICE_ERROR,
		  "a download the driver refuses is a device error");
    expect(fault_disarm() == 1 && ticket == UNTOUCHED_TICKET,
	   "the failed download took no ticket");
    expect_status(
	    backend->upload_wait(card, FIRST_TICKET, 0), STATUS_DEVICE_ERROR,
	    "the ticket the failed download would have taken reports the device error");
    expect_failed_card(card, BACKEND_NO_KERNEL);
}

#endif

/* Everything the test does with one card. */
static void test_card(uint32_t card_index)
{
    const int context_length = snprintf(context_text, sizeof(context_text), "card %" PRIu32, card_index);
    struct kernel_launch stale_launch = fill_launch(BACKEND_NO_KERNEL, 0, 0, 0);
    struct card_description_record description;
    struct card_description_record description_afterwards;
    struct device_card *card = NULL;
    uint64_t ticket = UNTOUCHED_TICKET;
    uint32_t kernel = UNTOUCHED_KERNEL;
    uint32_t kernel_afterwards = UNTOUCHED_KERNEL;

    memset(&description, 0, sizeof(description));
    memset(&description_afterwards, 0, sizeof(description_afterwards));
    if (context_length < 0 || (size_t)context_length >= sizeof(context_text) ||
	backend->card_describe(card_index, &description) != STATUS_OK) {
	    expect(false, "the card can be named and described");
	    return;
    }
    (void)printf("backend_cuda_launch_card_test: card %" PRIu32 ": %s at %s, class %" PRIu32 "\n", card_index,
                 description.name, description.bus_address, description.arithmetic_class);
    if (description.memory_free_bytes < description.memory_reserved_bytes ||
        description.memory_free_bytes - description.memory_reserved_bytes < ARENA_BYTE_COUNT) {
        expect(false, "the cards are not free: this card has no room for the arena of the test");
        return;
    }
    card = open_card(card_index);
    if (card == NULL) {
        return;
    }
    kernel = test_prepare(card);
    stale_launch.kernel = kernel;               /* a launch of no words that this open of the card would run */
    if (kernel != BACKEND_NO_KERNEL) {
	    expect_status(backend->launch_run(card, &stale_launch, 1),
			  STATUS_OK, "a fill of no words is queued");
	    test_fills(card, kernel);
	    test_launch_refusals(card, kernel);
	    test_download_rules(card);
	    test_launches_from_threads(card, kernel);
    }
#if CHECKING_HOOKS_ENABLED
    expect(fault_arm("backend_cuda_cuModuleUnload", NEVER_FIRING_CALL),
	   "the unloads are counted");
#endif
    backend->card_close(card, WAIT_LIMIT_NANOSECONDS);
#if CHECKING_HOOKS_ENABLED
    expect(fault_disarm() == HELD_MODULE_COUNT,
	   "a card that is closed unloads every module it holds");
#endif
    expect_status(backend->kernel_prepare(card, FILL_FILE, FILL_FUNCTION,
					  RECORD_BYTE_COUNT,
					  &kernel_afterwards),
		  STATUS_INVALID_ARGUMENT, "a closed card prepares no kernel");
    expect_status(backend->launch_run(card, &stale_launch, 1),
		  STATUS_INVALID_ARGUMENT, "a closed card takes no launch");
    expect_status(backend->launch_wait(card, 0), STATUS_INVALID_ARGUMENT,
		  "a closed card takes no wait");
    expect_status(backend->download_begin(card, 0, received, WORD_BYTES,
					  &ticket),
		  STATUS_INVALID_ARGUMENT, "a closed card takes no download");
    expect(ticket == UNTOUCHED_TICKET && kernel_afterwards == UNTOUCHED_KERNEL,
           "a closed card leaves outputs untouched");

    /* a card opened again has prepared nothing: the handle of the open before names no kernel, until it is prepared */
    card = open_card(card_index);
    if (card != NULL) {
	    expect_status(backend->launch_run(card, &stale_launch, 1),
			  STATUS_INVALID_ARGUMENT,
			  "a handle of an earlier open of the card is refused");
	    upload_pattern(card);
	    kernel = prepare_fill(card);
	    if (kernel != BACKEND_NO_KERNEL) {
		    const struct kernel_launch launch =
			    fill_launch(kernel, PART_FIRST_WORD,
					PART_WORD_COUNT, PART_VALUE);
		    mirror_fill(PART_FIRST_WORD, PART_WORD_COUNT, PART_VALUE);
		    run_launches(card, &launch, 1);
		    expect_arena_is_mirror(
			    card,
			    "a card opened a second time fills what it is asked to");
	    }
	backend->card_close(card, WAIT_LIMIT_NANOSECONDS);
    }
#if CHECKING_HOOKS_ENABLED
    test_driver_faults(card_index);
#endif
    expect(backend->card_describe(card_index, &description_afterwards) ==
			   STATUS_OK &&
		   description_afterwards.memory_free_bytes +
				   MEMORY_SLACK_BYTES >=
			   description.memory_free_bytes,
	   "a closed card has given its arena and its modules back");
}

/* Calls for different cards run concurrently: every card fills and downloads words of its own from its own thread. */
static void test_cards_together(uint32_t card_count)
{
	static struct fill_thread threads[BACKEND_MAXIMUM_CARDS];
	uint32_t ready_count = 0;

	(void)snprintf(context_text, sizeof(context_text),
		       "the cards together");
	for (uint32_t card_index = 0; card_index < card_count; card_index++) {
		threads[card_index].card = open_card(card_index);
		threads[card_index].slice_index = card_index;
		threads[card_index].kernel =
			threads[card_index].card == NULL ?
				BACKEND_NO_KERNEL :
				prepare_fill(threads[card_index].card);
		ready_count += threads[card_index].kernel != BACKEND_NO_KERNEL ?
				       1u :
				       0u;
	}
    if (ready_count == card_count) {
        expect(run_fill_threads(threads, card_count) == 0, "cards driven at once each fill and hand back their words");
    }
    for (uint32_t card_index = 0; card_index < card_count; card_index++) {
        backend->card_close(threads[card_index].card, WAIT_LIMIT_NANOSECONDS);
    }
}

int main(void)
{
    void *pattern_block = NULL;
    void *received_block = NULL;
    uint32_t card_count = 0;

    (void)alarm(TEST_TIME_LIMIT_SECONDS);
#if CHECKING_HOOKS_ENABLED
    run_in_child(test_limits_unread, "a card that does not state the limits of a launch fails the initialisation");
#endif
    expect_status(backend->card_count(&card_count), STATUS_OK,
		  "the cards are counted");
    expect(card_count > 0 && card_count <= BACKEND_MAXIMUM_CARDS,
	   "the machine has cards for the test");
    if (card_count == 0 || card_count > BACKEND_MAXIMUM_CARDS) {
	    return EXIT_FAILURE;
    }
    expect_status(backend->staging_allocate(ARENA_BYTE_COUNT, &pattern_block),
		  STATUS_OK, "the pattern is page-locked");
    expect_status(backend->staging_allocate(ARENA_BYTE_COUNT, &received_block),
		  STATUS_OK, "the block of the downloads is page-locked");
    pattern = pattern_block;
    received = received_block;
    mirror = malloc(ARENA_BYTE_COUNT);
    if (pattern == NULL || received == NULL || mirror == NULL) {
        (void)fprintf(stderr, "FAILED: the memory of the test could not be obtained\n");
        return EXIT_FAILURE;
    }
    for (uint64_t word_index = 0; word_index < ARENA_WORD_COUNT; word_index++) {
        pattern[word_index] = (uint32_t)word_index * PATTERN_MULTIPLIER + PATTERN_INCREMENT;
    }
    memcpy(mirror, pattern, ARENA_BYTE_COUNT);
    test_formats(card_count);
    test_refusals_without_a_card();
    for (uint32_t card_index = 0; card_index < card_count; card_index++) {
        test_card(card_index);
    }
    test_cards_together(card_count);
    free(mirror);
    backend->staging_free(received_block);
    backend->staging_free(pattern_block);
    if (failures != 0) {
        return EXIT_FAILURE;
    }
    (void)printf("backend_cuda_launch_card_test passed on %" PRIu32
		 " cards (checking hooks %s)\n",
		 card_count,
		 CHECKING_HOOKS_ENABLED ? "enabled" : "compiled out");
    return EXIT_SUCCESS;
}
