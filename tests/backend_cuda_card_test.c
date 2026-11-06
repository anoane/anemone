/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * backend_cuda_card_test: the CUDA backend on the real cards. Every card of the machine is described, opened, loaded
 * from page-locked memory and digested, and every refusal of backend_seam.h is provoked once with its exact status.
 *
 * It needs the cards free (`make card_tests`): a card without room for the arena FAILS the test, it is never skipped.
 * What a card holds is known through its digest alone: the card's digest of a range must be
 * pack_digest_of_host_bytes of the page-locked bytes uploaded there, which the test keeps as the mirror of
 * every arena. The error lines of the backend between the test's own lines are the refusals and faults it provokes.
 * At the end every card is loaded at once, each from a thread of its own, while blocks are page-locked beside them.
 *
 * A process initialises the driver once, so what the backend answers on a machine without cards, and when the
 * initialisation fails, is asked in child processes, before the test's own first call of the backend.
 */
#include <inttypes.h>
#include <limits.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include "anemone/backend_seam.h"
#include "anemone/hooks.h"
#include "anemone/pack_digest.h"
#include "anemone/status.h"

#define TEST_TIME_LIMIT_SECONDS 900             /* the alarm ends the test here, whatever the driver does */
#define WAIT_LIMIT_NANOSECONDS 120000000000ull  /* two minutes: far above any upload or digest of this test */
#define SHORTEST_WAIT_LIMIT_NANOSECONDS 1ull    /* a limit that is not 0 and has passed at the first look */
#define ARENA_BYTE_COUNT (256ull << 20u)
#define PIECE_COUNT \
	BACKEND_MAXIMUM_UPLOADS_IN_FLIGHT /* the mirror goes up in as many uploads as may fly */
#define PIECE_BYTE_COUNT (ARENA_BYTE_COUNT / PIECE_COUNT)
#define PIECE_ORDER_STRIDE 5                    /* odd, and the piece count a power of two: every piece comes once */
#define PIECE_ORDER_FIRST 3
#define WORD_BYTES PACK_DIGEST_WORD_BYTES
#define ARENA_WORD_COUNT (ARENA_BYTE_COUNT / WORD_BYTES)
#define PATTERN_SEED 0x243F6A8885A308D3ull      /* the words of the mirror only have to differ from one another */
#define PATTERN_MULTIPLIER 6364136223846793005ull
#define PATTERN_INCREMENT 1442695040888963407ull
#define PATTERN_SHIFT 29u
#define KNOWN_ANSWER_BYTE_COUNT 16              /* the bytes 00 01 .. 0F of pack_digest.h */
#define KNOWN_ANSWER_DIGEST 0x9F861E2075053703ull
#define KNOWN_ANSWER_ARENA_OFFSET (12288ull + WORD_BYTES)
#define ONE_WORD_INDEX 1234567ull
#define ODD_RANGE_FIRST_WORD 777ull
#define ODD_RANGE_WORD_COUNT 100001ull
#define FLIPPED_WORD_INDEX 20000003ull
#define FLIPPED_BIT (1ull << 37u)
#define FLIPPED_PIECE_INDEX (FLIPPED_WORD_INDEX * WORD_BYTES / PIECE_BYTE_COUNT)
#define SMALL_STAGING_BYTE_COUNT 1ull           /* rounded up by the backend to one whole alignment unit */
#define SMALL_BLOCK_ARENA_OFFSET \
	(PIECE_BYTE_COUNT + BACKEND_STAGING_ALIGNMENT_BYTES)
#define OFF_THE_WORD 4ull
#define UNTOUCHED_TICKET 0xABCDull
#define UNTOUCHED_DIGEST 0x1234ull
#define UNTOUCHED_COUNT 0xABCu
#define MAXIMUM_LINK_GENERATION 6
#define MAXIMUM_LINK_LANE_COUNT 32
#define DIGEST_THREAD_COUNT 4
#define DIGESTS_PER_THREAD 8
#define CONTEXT_CAPACITY 160
#define UNSTAGED_BYTE_COUNT 64
#define FIRST_TICKET 1ull
#define SECOND_TICKET 2ull
#define SECOND_CALL 2
#define LINK_TEXT_CAPACITY 128
#define DECIMAL 10
#define STAGING_THREAD_COUNT 8                  /* together they ask for twice the blocks the seam allows */
#define STAGING_ATTEMPTS_PER_THREAD \
	(BACKEND_MAXIMUM_STAGING_BLOCKS * 2 / STAGING_THREAD_COUNT)
#define STAGING_ROUNDS_BESIDE_THE_CARDS 4       /* blocks of the arena's size page-locked while every card uploads */
#define MEMORY_SLACK_BYTES (ARENA_BYTE_COUNT / 2)       /* what a driver may keep; an arena left behind is more */

/* The four files of a card's link, in the order source/backend_cuda.c reads them: the fault cases count by it. */
enum link_file { LINK_FILE_SPEED_NOW, LINK_FILE_CARD_MOST, LINK_FILE_PORT_MOST, LINK_FILE_WIDTH_NOW, LINK_FILE_COUNT };

/* The whole GT/s a speed file begins with ("2.5 GT/s PCIe" gives 2), at the index of its link generation. */
static const unsigned long link_speeds[] = {0, 2, 5, 8, 16, 32, 64};

struct card_link {
    uint32_t generation;
    uint32_t lane_count;
};

_Static_assert(PIECE_BYTE_COUNT * PIECE_COUNT == ARENA_BYTE_COUNT && PIECE_BYTE_COUNT % WORD_BYTES == 0,
               "the pieces are whole words and make up the arena");

static const struct backend_operations *const backend = &backend_cuda_value;
static int failures = 0;
static char context_text[CONTEXT_CAPACITY] = "the backend";    /* what the failure lines are about */
static void *mirror = NULL;                     /* page-locked: what every open arena must hold, byte for byte */
static uint64_t piece_digests[PIECE_COUNT];     /* the host's digest of each piece of the mirror */

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

static const unsigned char *mirror_bytes(uint64_t byte_offset)
{
    return (const unsigned char *)mirror + byte_offset;
}

/* The oracle: the host's digest of a range of the mirror. */
static uint64_t host_digest(uint64_t byte_offset, uint64_t byte_count)
{
    uint64_t digest = 0;
    expect_status(pack_digest_of_host_bytes(mirror_bytes(byte_offset),
					    byte_count, &digest),
		  STATUS_OK, "the host digests a range of the mirror");
    return digest;
}

/* The card's digest of a range of its arena, which must be delivered within the limit. */
static uint64_t card_digest(struct device_card *card, uint64_t byte_offset,
			    uint64_t byte_count)
{
    uint64_t digest = UNTOUCHED_DIGEST;
    expect_status(backend->digest(card, byte_offset, byte_count,
				  WAIT_LIMIT_NANOSECONDS, &digest),
		  STATUS_OK, "the card digests a range of its arena");
    return digest;
}

/* Fills the mirror from a multiply-and-add generator and lays the known answer of pack_digest.h into it. */
static void fill_mirror(void)
{
    uint64_t *const words = mirror;
    unsigned char *const known_answer_bytes = (unsigned char *)mirror + KNOWN_ANSWER_ARENA_OFFSET;
    uint64_t state = PATTERN_SEED;
    for (uint64_t word_index = 0; word_index < ARENA_WORD_COUNT; word_index++) {
        state = state * PATTERN_MULTIPLIER + PATTERN_INCREMENT;
        words[word_index] = state ^ (state >> PATTERN_SHIFT);
    }
    for (unsigned int byte_index = 0; byte_index < KNOWN_ANSWER_BYTE_COUNT; byte_index++) {
        known_answer_bytes[byte_index] = (unsigned char)byte_index;
    }
}

static bool text_is_terminated_and_zero_filled(const char *text, size_t capacity)
{
    const size_t length = strnlen(text, capacity);
    for (size_t byte_index = length; byte_index < capacity; byte_index++) {
        if (text[byte_index] != '\0') {
            return false;
        }
    }
    return length < capacity;
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

/* Begins the upload of one piece of the mirror to its place in the arena and expects the ticket it must get. */
static void upload_piece(struct device_card *card, uint32_t piece_index,
			 uint64_t expected_ticket)
{
    const uint64_t byte_offset = piece_index * PIECE_BYTE_COUNT;
    uint64_t ticket = UNTOUCHED_TICKET;
    expect_status(backend->upload_begin(card, byte_offset,
					mirror_bytes(byte_offset),
					PIECE_BYTE_COUNT, &ticket),
		  STATUS_OK, "the upload of a piece begins");
    expect(ticket == expected_ticket, "tickets start at one and rise by one");
}

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

/* A machine without cards, made by hiding them from the driver, which reads CUDA_VISIBLE_DEVICES when it is
 * initialised: the count is 0, no index is a card, and there is no card to page-lock staging memory for. */
static void test_machine_without_cards(void)
{
	struct card_description_record description;
	struct device_card *card = NULL;
	void *block = NULL;
	uint32_t card_count = UNTOUCHED_COUNT;

	memset(&description, 0, sizeof(description));
	expect(setenv("CUDA_VISIBLE_DEVICES", "", 1) == 0,
	       "the cards are hidden from the driver");
	expect_status(backend->card_count(&card_count), STATUS_OK,
		      "a machine without cards is counted");
	expect(card_count == 0, "a machine without cards has 0 cards");
	expect_status(backend->card_describe(0, &description),
		      STATUS_INVALID_ARGUMENT,
		      "without cards, index 0 is already past the last card");
	expect_status(backend->card_open(0, ARENA_BYTE_COUNT, &card),
		      STATUS_INVALID_ARGUMENT, "without cards, no card opens");
	expect_status(backend->staging_allocate(SMALL_STAGING_BYTE_COUNT,
						&block),
		      STATUS_UNSUPPORTED,
		      "without cards, there is no staging memory");
	expect(card == NULL && block == NULL,
	       "the refusals of a machine without cards leave their outputs untouched");
}

#if CHECKING_HOOKS_ENABLED

/* A driver that fails while it is initialised: every function answers a device error, and it is not tried again. */
static void test_failed_initialisation(void)
{
	struct card_description_record description;
	struct device_card *card = NULL;
	void *block = NULL;
	uint32_t card_count = UNTOUCHED_COUNT;

	memset(&description, 0, sizeof(description));
	expect(fault_arm("backend_cuda_cuDeviceGetCount", 1),
	       "the fault of the initialisation is armed");
	expect_status(backend->card_count(&card_count), STATUS_DEVICE_ERROR,
		      "a driver that cannot count its cards is a device error");
	expect_status(backend->card_count(&card_count), STATUS_DEVICE_ERROR,
		      "and stays one");
	expect_status(backend->card_describe(0, &description),
		      STATUS_DEVICE_ERROR,
		      "without a driver, no card is described");
	expect_status(backend->card_open(0, ARENA_BYTE_COUNT, &card),
		      STATUS_DEVICE_ERROR, "without a driver, no card opens");
	expect_status(backend->staging_allocate(SMALL_STAGING_BYTE_COUNT,
						&block),
		      STATUS_DEVICE_ERROR,
		      "without a driver, there is no staging memory");
	expect(fault_disarm() == 1,
	       "the driver is initialised once: a later call does not try again");
	expect(card_count == UNTOUCHED_COUNT && card == NULL && block == NULL,
	       "the calls of a backend without a driver leave their outputs untouched");
}

#endif

/* What every call refuses whatever the cards hold, and what it accepts in place of a card or a block. */
static void test_refusals_without_a_card(uint32_t card_count)
{
	struct card_description_record description;
	struct device_card *card = NULL;
	uint64_t ticket = UNTOUCHED_TICKET;
	uint64_t digest = UNTOUCHED_DIGEST;
	unsigned char unstaged[UNSTAGED_BYTE_COUNT] = { 0 };

	memset(&description, 0, sizeof(description));
	expect_status(backend->card_count(NULL), STATUS_INVALID_ARGUMENT,
		      "a count without an output is refused");
	expect_status(backend->card_describe(card_count, &description),
		      STATUS_INVALID_ARGUMENT,
		      "the description of a card past the last is refused");
	expect_status(backend->card_describe(UINT32_MAX, &description),
		      STATUS_INVALID_ARGUMENT,
		      "the description of a card far past the last is refused");
	expect_status(backend->card_describe(0, NULL), STATUS_INVALID_ARGUMENT,
		      "a description without an output is refused");
	expect_status(backend->card_open(card_count, ARENA_BYTE_COUNT, &card),
		      STATUS_INVALID_ARGUMENT,
		      "a card past the last does not open");
	expect_status(backend->card_open(0, 0, &card), STATUS_INVALID_ARGUMENT,
		      "an arena of no bytes is refused");
	expect_status(backend->card_open(0, ARENA_BYTE_COUNT, NULL),
		      STATUS_INVALID_ARGUMENT,
		      "an open without an output is refused");
	expect(card == NULL, "a refused open leaves its output untouched");
	expect_status(backend->upload_begin(NULL, 0, unstaged, sizeof(unstaged),
					    &ticket),
		      STATUS_INVALID_ARGUMENT,
		      "an upload to no card is refused");
	expect_status(backend->upload_wait(NULL, FIRST_TICKET, 0),
		      STATUS_INVALID_ARGUMENT, "a wait on no card is refused");
	expect_status(backend->digest(NULL, 0, WORD_BYTES, 0, &digest),
		      STATUS_INVALID_ARGUMENT,
		      "a digest of no card is refused");
	expect(ticket == UNTOUCHED_TICKET && digest == UNTOUCHED_DIGEST,
	       "refused calls leave their outputs untouched");
	backend->card_close(NULL, 0);
	backend->staging_free(NULL);
}

/* The limit of staging blocks, and that a block the driver refuses leaves none behind. No block exists yet. */
static void test_staging_blocks(void)
{
	static void *blocks[BACKEND_MAXIMUM_STAGING_BLOCKS];
	void *refused_block = NULL;
	uint32_t block_count = 0;
	bool every_block_is_aligned = true;

	expect_status(backend->staging_allocate(0, &refused_block),
		      STATUS_INVALID_ARGUMENT,
		      "a staging block of no bytes is refused");
	expect_status(backend->staging_allocate(UINT64_MAX, &refused_block),
		      STATUS_INVALID_ARGUMENT,
		      "a staging block too large to round up is refused");
	expect_status(backend->staging_allocate(SMALL_STAGING_BYTE_COUNT, NULL),
		      STATUS_INVALID_ARGUMENT,
		      "a staging block without an output is refused");
#if CHECKING_HOOKS_ENABLED
	expect(fault_arm("backend_cuda_cuMemHostRegister", 1),
	       "the fault of the staging allocation is armed");
	expect_status(
		backend->staging_allocate(SMALL_STAGING_BYTE_COUNT,
					  &refused_block),
		STATUS_OUT_OF_MEMORY,
		"a staging block the driver does not give is out of memory");
	expect(fault_disarm() == 1,
	       "the fault of the staging allocation was reached");
	expect(fault_arm("backend_cuda_cuCtxPushCurrent", 1),
	       "the fault of the context is armed");
	expect_status(
		backend->staging_allocate(SMALL_STAGING_BYTE_COUNT,
					  &refused_block),
		STATUS_OUT_OF_MEMORY,
		"a staging block for which no context can be made current is out of memory");
	expect(fault_disarm() == 1, "the fault of the context was reached");
#endif
	while (block_count < BACKEND_MAXIMUM_STAGING_BLOCKS &&
	       backend->staging_allocate(SMALL_STAGING_BYTE_COUNT,
					 &blocks[block_count]) == STATUS_OK) {
		every_block_is_aligned =
			every_block_is_aligned && blocks[block_count] != NULL &&
			(uintptr_t)blocks[block_count] %
					BACKEND_STAGING_ALIGNMENT_BYTES ==
				0;
		block_count++;
	}
	expect(block_count == BACKEND_MAXIMUM_STAGING_BLOCKS,
	       "as many staging blocks as the seam allows exist at a time: no refused block was left behind");
	expect(every_block_is_aligned,
	       "every staging block is aligned as the seam says");
	expect_status(backend->staging_allocate(SMALL_STAGING_BYTE_COUNT,
						&refused_block),
		      STATUS_BUSY_VALUE,
		      "one staging block more is refused as busy");
	expect(refused_block == NULL,
	       "refused staging allocations leave their output untouched");
	while (block_count > 0) {
		block_count--;
		backend->staging_free(blocks[block_count]);
	}
}

struct staging_thread {
    pthread_t thread;
    void *blocks[STAGING_ATTEMPTS_PER_THREAD];
    uint32_t block_count;
    uint32_t wrong_answer_count;
};

/* Asks for staging blocks while the other threads do: each answer is a block, or the refusal of one too many. */
static void *staging_thread_run(void *argument)
{
    struct staging_thread *const self = argument;
    for (uint32_t attempt = 0; attempt < STAGING_ATTEMPTS_PER_THREAD; attempt++) {
        void *block = NULL;
	const result_status status =
		backend->staging_allocate(SMALL_STAGING_BYTE_COUNT, &block);
	if (status == STATUS_OK && block != NULL) {
		self->blocks[self->block_count] = block;
		self->block_count++;
	} else if (status != STATUS_BUSY_VALUE || block != NULL) {
		self->wrong_answer_count++;
	}
    }
    return NULL;
}

/* The driver page-locks a block with no mutex of the backend held, so several threads obtain blocks at once: the
 * limit of the seam must still hold exactly. No block exists yet. */
static void test_staging_from_threads(void)
{
    static struct staging_thread threads[STAGING_THREAD_COUNT];
    uint32_t started_count = 0;
    uint32_t block_count = 0;
    uint32_t wrong_answer_count = 0;

    while (started_count < STAGING_THREAD_COUNT &&
           pthread_create(&threads[started_count].thread, NULL, staging_thread_run, &threads[started_count]) == 0) {
        started_count++;
    }
    expect(started_count == STAGING_THREAD_COUNT, "the staging threads start");
    for (uint32_t thread_index = 0; thread_index < started_count; thread_index++) {
        expect(pthread_join(threads[thread_index].thread, NULL) == 0, "a staging thread is joined");
        block_count += threads[thread_index].block_count;
        wrong_answer_count += threads[thread_index].wrong_answer_count;
    }
    expect(wrong_answer_count == 0, "every staging allocation is answered with a block or as busy");
    expect(block_count == BACKEND_MAXIMUM_STAGING_BLOCKS,
	   "threads that ask for twice the staging blocks the seam allows obtain exactly as many as it allows");
    for (uint32_t thread_index = 0; thread_index < started_count; thread_index++) {
        for (uint32_t block_index = 0; block_index < threads[thread_index].block_count; block_index++) {
            backend->staging_free(threads[thread_index].blocks[block_index]);
        }
    }
}

/* The description of a card, checked for sense; false, and a failure, if there is none. */
static bool describe_card(uint32_t card_index,
			  struct card_description_record *description)
{
    memset(description, UINT8_MAX, sizeof(*description));    /* every byte must be written over */
    if (backend->card_describe(card_index, description) != STATUS_OK) {
	    expect(false, "the card is described");
	    return false;
    }
    expect(text_is_terminated_and_zero_filled(description->name, sizeof(description->name)) &&
               description->name[0] != '\0',
           "the name is a text, zero-filled to its capacity");
    expect(text_is_terminated_and_zero_filled(description->bus_address, sizeof(description->bus_address)) &&
               description->bus_address[strspn(description->bus_address, "0123456789abcdef:.")] == '\0',
           "the bus address is a lower-case text, zero-filled to its capacity");
    expect(description->arithmetic_class == ARITHMETIC_CLASS_SM_80 ||
		   description->arithmetic_class == ARITHMETIC_CLASS_SM_120,
	   "the arithmetic class is one the engine has kernels for");
    expect(description->link_generation <= MAXIMUM_LINK_GENERATION &&
               description->link_lane_count <= MAXIMUM_LINK_LANE_COUNT && description->reserved_zero == 0,
           "the link is a known generation and width, or zero, and the reserved field is zero");
    expect(description->memory_total_bytes >= description->memory_free_bytes &&
               description->memory_total_bytes < UINT64_MAX && description->memory_reserved_bytes > 0 &&
               description->memory_reserved_bytes < description->memory_total_bytes,
           "the total memory is written, and holds the free and the reserved memory");
    return true;
}

/* The number a file of a device's directory begins with, as the test reads it itself; 0 when it cannot be read. */
static unsigned long link_file_number(const char *directory, const char *file_name)
{
    char path[PATH_MAX] = "";
    char text[LINK_TEXT_CAPACITY] = "";
    const int path_length = snprintf(path, sizeof(path), "%s/%s", directory, file_name);
    FILE *const file = path_length > 0 && (size_t)path_length < sizeof(path) ? fopen(path, "r") : NULL;
    const bool is_read = file != NULL && fgets(text, LINK_TEXT_CAPACITY, file) != NULL;

    if (file != NULL) {
        (void)fclose(file);
    }
    return is_read ? strtoul(text, NULL, DECIMAL) : 0;
}

/* What backend_seam.h says of the link of the card at a bus address, from the files as the test reads them itself:
 * the port above a card is the parent directory of the card's resolved path. The file `unread`, if it is one of the
 * four, counts as one that cannot be read. */
static struct card_link stated_link(const char *bus_address, uint32_t unread)
{
    char device_link[PATH_MAX] = "";
    char card_directory[PATH_MAX] = "";
    char port_directory[PATH_MAX] = "";
    unsigned long numbers[LINK_FILE_COUNT] = {0};
    uint32_t generations[LINK_FILE_COUNT] = {0};
    struct card_link stated = {0, 0};
    const int link_length = snprintf(device_link, sizeof(device_link), "/sys/bus/pci/devices/%s", bus_address);
    char *port_end = NULL;

    if (bus_address[0] != '\0' && link_length > 0 && (size_t)link_length < sizeof(device_link) &&
        realpath(device_link, card_directory) != NULL) {
        memcpy(port_directory, card_directory, sizeof(port_directory));
        port_end = strrchr(port_directory, '/');
        if (port_end != NULL) {
            *port_end = '\0';
        }
        numbers[LINK_FILE_SPEED_NOW] = link_file_number(card_directory, "current_link_speed");
        numbers[LINK_FILE_CARD_MOST] = link_file_number(card_directory, "max_link_speed");
        numbers[LINK_FILE_PORT_MOST] = link_file_number(port_directory, "max_link_speed");
        numbers[LINK_FILE_WIDTH_NOW] = link_file_number(card_directory, "current_link_width");
    }
    if (unread < LINK_FILE_COUNT) {
        numbers[unread] = 0;
    }
    for (uint32_t file_index = 0; file_index < LINK_FILE_COUNT; file_index++) {
        for (uint32_t generation = 1; generation < sizeof(link_speeds) / sizeof(link_speeds[0]); generation++) {
            if (numbers[file_index] == link_speeds[generation]) {
                generations[file_index] = generation;
            }
        }
    }
    stated.generation = generations[LINK_FILE_CARD_MOST] < generations[LINK_FILE_PORT_MOST]
                            ? generations[LINK_FILE_CARD_MOST] : generations[LINK_FILE_PORT_MOST];
    if (generations[LINK_FILE_SPEED_NOW] > stated.generation) {
        stated.generation = generations[LINK_FILE_SPEED_NOW];
    }
    stated.lane_count = numbers[LINK_FILE_WIDTH_NOW] <= UINT32_MAX ? (uint32_t)numbers[LINK_FILE_WIDTH_NOW] : 0;
    return stated;
}

/* The description of a card must carry the link its files state. They are read before and after the description is
 * taken and either reading is accepted: the speed negotiated now can change under the test. */
static void expect_stated_link(uint32_t card_index, const char *bus_address, uint32_t unread)
{
	struct card_description_record description;
	const struct card_link before = stated_link(bus_address, unread);
	const bool is_described = describe_card(card_index, &description);
	const struct card_link after = stated_link(bus_address, unread);

	expect(is_described &&
		       (description.link_generation == before.generation ||
			description.link_generation == after.generation),
	       "the link generation is the higher of the speed now and of what both the card and its port state");
	expect(is_described &&
		       (description.link_lane_count == before.lane_count ||
			description.link_lane_count == after.lane_count),
	       "the lane count is the width negotiated now");
}

/* Every refusal of upload_begin on an open card: none of them takes a ticket. */
static void test_upload_refusals(struct device_card *card,
				 const void *small_block)
{
    unsigned char unstaged[UNSTAGED_BYTE_COUNT] = {0};
    uint64_t ticket = UNTOUCHED_TICKET;

    expect_status(backend->upload_begin(card, 0, mirror, 0, &ticket),
		  STATUS_INVALID_ARGUMENT, "an upload of no bytes is refused");
    expect_status(backend->upload_begin(card, ARENA_BYTE_COUNT - WORD_BYTES,
					mirror, WORD_BYTES + WORD_BYTES,
					&ticket),
		  STATUS_INVALID_ARGUMENT,
		  "an upload across the end of the arena is refused");
    expect_status(backend->upload_begin(card, ARENA_BYTE_COUNT + 1, mirror,
					WORD_BYTES, &ticket),
		  STATUS_INVALID_ARGUMENT,
		  "an upload behind the arena is refused");
    expect_status(backend->upload_begin(card, UINT64_MAX, mirror, WORD_BYTES,
					&ticket),
		  STATUS_INVALID_ARGUMENT,
		  "an upload whose range wraps is refused");
    expect_status(backend->upload_begin(card, 0, unstaged, sizeof(unstaged),
					&ticket),
		  STATUS_INVALID_ARGUMENT,
		  "a source outside every staging block is refused");
    expect_status(backend->upload_begin(card, 0, NULL, WORD_BYTES, &ticket),
		  STATUS_INVALID_ARGUMENT, "a missing source is refused");
    expect_status(backend->upload_begin(
			  card, 0, mirror_bytes(ARENA_BYTE_COUNT - WORD_BYTES),
			  WORD_BYTES + WORD_BYTES, &ticket),
		  STATUS_INVALID_ARGUMENT,
		  "a source across the end of its staging block is refused");
    expect_status(backend->upload_begin(card, 0, small_block,
					BACKEND_STAGING_ALIGNMENT_BYTES + 1,
					&ticket),
		  STATUS_INVALID_ARGUMENT,
		  "a source longer than its staging block is refused");
    expect_status(backend->upload_begin(card, 0, mirror, WORD_BYTES, NULL),
		  STATUS_INVALID_ARGUMENT,
		  "an upload without an output for its ticket is refused");
    expect(ticket == UNTOUCHED_TICKET, "refused uploads leave their ticket untouched");
}

/* Every refusal of digest on an open card, and the digest of no words. */
static void test_digest_refusals(struct device_card *card)
{
    uint64_t digest = UNTOUCHED_DIGEST;

    expect_status(backend->digest(card, OFF_THE_WORD, WORD_BYTES,
				  WAIT_LIMIT_NANOSECONDS, &digest),
		  STATUS_INVALID_ARGUMENT,
		  "a digest from an offset off the word is refused");
    expect_status(backend->digest(card, 0, WORD_BYTES + OFF_THE_WORD,
				  WAIT_LIMIT_NANOSECONDS, &digest),
		  STATUS_INVALID_ARGUMENT,
		  "a digest of a count off the word is refused");
    expect_status(backend->digest(card, ARENA_BYTE_COUNT - WORD_BYTES,
				  WORD_BYTES + WORD_BYTES,
				  WAIT_LIMIT_NANOSECONDS, &digest),
		  STATUS_INVALID_ARGUMENT,
		  "a digest across the end of the arena is refused");
    expect_status(backend->digest(card, ARENA_BYTE_COUNT + WORD_BYTES, 0,
				  WAIT_LIMIT_NANOSECONDS, &digest),
		  STATUS_INVALID_ARGUMENT,
		  "a digest behind the arena is refused");
    expect_status(
	    backend->digest(card, 0, WORD_BYTES, WAIT_LIMIT_NANOSECONDS, NULL),
	    STATUS_INVALID_ARGUMENT, "a digest without an output is refused");
    expect(digest == UNTOUCHED_DIGEST, "refused digests leave their output untouched");
    expect_status(backend->digest(card, ARENA_BYTE_COUNT, 0, 0, &digest),
		  STATUS_OK,
		  "the digest of no words is delivered, with any limit");
    expect(digest == 0, "the digest of no words is zero");
}

struct digest_thread {
    pthread_t thread;
    struct device_card *card;
    uint32_t thread_index;
    uint32_t wrong_digest_count;
};

/* Digests pieces of the arena while the other threads digest others on the same card. */
static void *digest_thread_run(void *argument)
{
    struct digest_thread *const self = argument;
    for (uint32_t round = 0; round < DIGESTS_PER_THREAD; round++) {
        const uint32_t piece_index = (self->thread_index + round * DIGEST_THREAD_COUNT) % PIECE_COUNT;
        uint64_t digest = UNTOUCHED_DIGEST;
	const result_status status = backend->digest(
		self->card, piece_index * PIECE_BYTE_COUNT, PIECE_BYTE_COUNT,
		WAIT_LIMIT_NANOSECONDS, &digest);
	if (status != STATUS_OK || digest != piece_digests[piece_index]) {
		self->wrong_digest_count++;
	}
    }
    return NULL;
}

/* Any thread may call: several digest the same card at once, and each gets the digest of its own range. */
static void test_digests_from_threads(struct device_card *card)
{
    struct digest_thread threads[DIGEST_THREAD_COUNT];
    uint32_t started_count = 0;
    uint32_t wrong_digest_count = 0;

    memset(threads, 0, sizeof(threads));
    while (started_count < DIGEST_THREAD_COUNT) {
        threads[started_count].card = card;
        threads[started_count].thread_index = started_count;
        if (pthread_create(&threads[started_count].thread, NULL, digest_thread_run, &threads[started_count]) != 0) {
            break;
        }
        started_count++;
    }
    expect(started_count == DIGEST_THREAD_COUNT, "the digest threads start");
    for (uint32_t thread_index = 0; thread_index < started_count; thread_index++) {
        expect(pthread_join(threads[thread_index].thread, NULL) == 0, "a digest thread is joined");
        wrong_digest_count += threads[thread_index].wrong_digest_count;
    }
    expect(wrong_digest_count == 0, "digests asked for by several threads at once are each the digest of their range");
}

/* The whole life of an open card: uploads, tickets, digests, one changed bit, the refusals, the close. */
static void test_card_cycle(uint32_t card_index)
{
    uint64_t *const mirror_words = mirror;
    struct device_card *const card = open_card(card_index);
    struct device_card *second_handle = NULL;
    void *small_block = NULL;
    uint64_t ticket = UNTOUCHED_TICKET;
    uint64_t next_ticket = FIRST_TICKET;
    uint64_t whole_digest = 0;
    uint64_t changed_digest = UNTOUCHED_DIGEST;
    result_status status = STATUS_OK;

    if (card == NULL) {
        return;
    }
    expect_status(backend->card_open(card_index, ARENA_BYTE_COUNT,
				     &second_handle),
		  STATUS_INVALID_ARGUMENT, "an open card does not open again");
    expect(second_handle == NULL, "a refused open leaves its output untouched");
    expect_status(backend->staging_allocate(SMALL_STAGING_BYTE_COUNT,
					    &small_block),
		  STATUS_OK, "a small staging block is obtained");
    if (small_block == NULL) {
        backend->card_close(card, WAIT_LIMIT_NANOSECONDS);
        return;
    }
    test_upload_refusals(card, small_block);

    /* the mirror goes up in pieces that are out of order, none waited for: the ring of uploads in flight is full */
    for (uint32_t step = 0; step < PIECE_COUNT; step++) {
        upload_piece(card, (step * PIECE_ORDER_STRIDE + PIECE_ORDER_FIRST) % PIECE_COUNT, next_ticket);
        next_ticket++;
    }
    expect_status(backend->upload_begin(card, 0, mirror, WORD_BYTES, &ticket),
		  STATUS_BUSY_VALUE,
		  "one upload more than may be in flight is refused as busy");
    expect(ticket == UNTOUCHED_TICKET, "an upload refused as busy takes no ticket");
    status = backend->upload_wait(card, next_ticket - 1, 0);
    expect(status == STATUS_OK || status == STATUS_TIMED_OUT,
	   "a wait with limit 0 only asks");
    expect_status(backend->upload_wait(card, next_ticket, 0),
		  STATUS_INVALID_ARGUMENT, "a ticket never issued is refused");
    expect_status(backend->upload_wait(card, 0, 0), STATUS_INVALID_ARGUMENT,
		  "zero is never a ticket");
    expect_status(backend->upload_wait(card, next_ticket - 1,
				       WAIT_LIMIT_NANOSECONDS),
		  STATUS_OK, "the last upload finishes within the limit");
    for (uint64_t finished_ticket = FIRST_TICKET; finished_ticket < next_ticket; finished_ticket++) {
	    expect_status(
		    backend->upload_wait(card, finished_ticket, 0), STATUS_OK,
		    "a finished upload means every earlier one, and is reported finished again at once");
    }

    /* a block asked for as one byte holds a whole alignment unit: the same bytes go up again from it */
    memcpy(small_block, mirror_bytes(SMALL_BLOCK_ARENA_OFFSET),
	   BACKEND_STAGING_ALIGNMENT_BYTES);
    expect_status(
	    backend->upload_begin(card, SMALL_BLOCK_ARENA_OFFSET, small_block,
				  BACKEND_STAGING_ALIGNMENT_BYTES, &ticket),
	    STATUS_OK, "a whole small staging block is uploaded");
    expect(ticket == next_ticket, "a refused upload took no ticket");
    next_ticket++;
    expect_status(backend->upload_wait(card, ticket, WAIT_LIMIT_NANOSECONDS),
		  STATUS_OK, "the upload from the small block finishes");

    for (uint32_t piece_index = 0; piece_index < PIECE_COUNT; piece_index++) {
        expect(card_digest(card, piece_index * PIECE_BYTE_COUNT, PIECE_BYTE_COUNT) == piece_digests[piece_index],
               "the card's digest of an uploaded piece is the host's");
    }
    whole_digest = card_digest(card, 0, ARENA_BYTE_COUNT);
    expect(whole_digest == host_digest(0, ARENA_BYTE_COUNT), "the card's digest of the whole arena is the host's");
    expect(card_digest(card, ONE_WORD_INDEX * WORD_BYTES, WORD_BYTES) ==
               host_digest(ONE_WORD_INDEX * WORD_BYTES, WORD_BYTES),
           "the card's digest of one word is the host's");
    expect(card_digest(card, ODD_RANGE_FIRST_WORD * WORD_BYTES, ODD_RANGE_WORD_COUNT * WORD_BYTES) ==
               host_digest(ODD_RANGE_FIRST_WORD * WORD_BYTES, ODD_RANGE_WORD_COUNT * WORD_BYTES),
           "the card's digest of an odd number of words is the host's");
    expect(card_digest(card, KNOWN_ANSWER_ARENA_OFFSET, KNOWN_ANSWER_BYTE_COUNT) == KNOWN_ANSWER_DIGEST,
           "the card gives the known answer of pack_digest.h");
    test_digest_refusals(card);

    /* shown failing: one bit of one word changes. Its piece goes up behind every other piece, the last of a full
     * ring, and the digest is asked for at once: it must wait for all of them, with no wait for an upload here */
    mirror_words[FLIPPED_WORD_INDEX] ^= FLIPPED_BIT;
    for (uint32_t step = 1; step <= PIECE_COUNT; step++) {
        upload_piece(card, (uint32_t)((FLIPPED_PIECE_INDEX + step) % PIECE_COUNT), next_ticket);
        next_ticket++;
    }
    changed_digest = card_digest(card, 0, ARENA_BYTE_COUNT);
    expect(changed_digest != whole_digest, "one changed bit changes the card's digest");
    expect(changed_digest == host_digest(0, ARENA_BYTE_COUNT), "a digest sees every upload begun before it");
    expect_status(backend->upload_wait(card, next_ticket - 1, 0), STATUS_OK,
		  "an upload that a digest has seen has finished");
    piece_digests[FLIPPED_PIECE_INDEX] = host_digest(FLIPPED_PIECE_INDEX * PIECE_BYTE_COUNT, PIECE_BYTE_COUNT);

    /* a digest with limit 0 only asks: no value is there without a wait, and it begins no kernel to wait for */
    whole_digest = UNTOUCHED_DIGEST;
#if CHECKING_HOOKS_ENABLED
    expect(fault_arm("backend_cuda_cuLaunchKernel", 1),
	   "the fault of the launch is armed");
#endif
    expect_status(backend->digest(card, 0, ARENA_BYTE_COUNT, 0, &whole_digest),
		  STATUS_TIMED_OUT,
		  "a digest with limit 0 has timed out at once");
#if CHECKING_HOOKS_ENABLED
    expect(fault_disarm() == 0, "a digest with limit 0 launches no kernel");
#endif
    expect(whole_digest == UNTOUCHED_DIGEST, "a digest with limit 0 delivers nothing");

    /* a digest that runs out of time delivers nothing either, and the next one is not disturbed by its kernel */
    status = backend->digest(card, 0, ARENA_BYTE_COUNT, SHORTEST_WAIT_LIMIT_NANOSECONDS, &whole_digest);
    expect((status == STATUS_TIMED_OUT && whole_digest == UNTOUCHED_DIGEST) ||
		   (status == STATUS_OK && whole_digest == changed_digest),
	   "a digest delivers its value within its limit, or nothing");
    expect(card_digest(card, ONE_WORD_INDEX * WORD_BYTES, WORD_BYTES) ==
               host_digest(ONE_WORD_INDEX * WORD_BYTES, WORD_BYTES),
           "the digest after one that ran out of time is right");
    test_digests_from_threads(card);

    backend->card_close(card, WAIT_LIMIT_NANOSECONDS);
    expect_status(backend->upload_begin(card, 0, mirror, WORD_BYTES, &ticket),
		  STATUS_INVALID_ARGUMENT, "a closed card takes no upload");
    expect_status(backend->upload_wait(card, FIRST_TICKET, 0),
		  STATUS_INVALID_ARGUMENT, "a closed card takes no wait");
    expect_status(backend->digest(card, 0, WORD_BYTES, 0, &whole_digest),
		  STATUS_INVALID_ARGUMENT, "a closed card takes no digest");
    backend->staging_free(small_block);
}

#if CHECKING_HOOKS_ENABLED

/* A card whose driver failed: the failed ticket and later ones report it, and only the close is left. */
static void expect_failed_card(struct device_card *card,
			       uint64_t first_failed_ticket)
{
    uint64_t ticket = UNTOUCHED_TICKET;
    uint64_t digest = UNTOUCHED_DIGEST;

    expect_status(backend->upload_wait(card, first_failed_ticket, 0),
		  STATUS_DEVICE_ERROR,
		  "the ticket of a failed upload reports a device error");
    expect_status(
	    backend->upload_wait(card, first_failed_ticket + 1,
				 WAIT_LIMIT_NANOSECONDS),
	    STATUS_DEVICE_ERROR,
	    "every later ticket of a failed card reports a device error, at once");
    expect_status(backend->upload_begin(card, 0, mirror, PIECE_BYTE_COUNT,
					&ticket),
		  STATUS_DEVICE_ERROR, "a failed card begins no upload");
    expect_status(backend->digest(card, 0, WORD_BYTES, WAIT_LIMIT_NANOSECONDS,
				  &digest),
		  STATUS_DEVICE_ERROR, "a failed card begins no digest");
    expect(ticket == UNTOUCHED_TICKET && digest == UNTOUCHED_DIGEST, "a failed card leaves outputs untouched");
    backend->card_close(card, WAIT_LIMIT_NANOSECONDS);
}

/* Each place where the driver can fail an open card or its opening, made to fail once. */
static void test_driver_faults(uint32_t card_index)
{
	struct card_description_record description;
	struct device_card *card = NULL;
	uint64_t ticket = UNTOUCHED_TICKET;
	uint64_t digest = UNTOUCHED_DIGEST;

	/* an upload the driver refuses: the tickets reported finished stay finished */
	card = open_card(card_index);
	upload_piece(card, 0, FIRST_TICKET);
	expect_status(backend->upload_wait(card, FIRST_TICKET,
					   WAIT_LIMIT_NANOSECONDS),
		      STATUS_OK, "the first upload finishes");
	expect(fault_arm("backend_cuda_cuMemcpyHtoDAsync", 1),
	       "the fault of the upload is armed");
	expect_status(backend->upload_begin(card, 0, mirror, PIECE_BYTE_COUNT,
					    &ticket),
		      STATUS_DEVICE_ERROR,
		      "an upload the driver refuses is a device error");
	expect(fault_disarm() == 1 && ticket == UNTOUCHED_TICKET,
	       "the failed upload took no ticket");
	expect_status(
		backend->upload_wait(card, FIRST_TICKET, 0), STATUS_OK,
		"a ticket reported finished before the failure stays finished");
	expect_failed_card(card, SECOND_TICKET);

	/* the same with an upload in flight that nobody had seen finish: its ticket fails with the card */
	card = open_card(card_index);
	upload_piece(card, 0, FIRST_TICKET);
	expect(fault_arm("backend_cuda_cuMemcpyHtoDAsync", 1),
	       "the fault of the upload is armed again");
	expect_status(backend->upload_begin(card, 0, mirror, PIECE_BYTE_COUNT,
					    &ticket),
		      STATUS_DEVICE_ERROR,
		      "an upload the driver refuses is a device error");
	expect(fault_disarm() == 1, "the fault of the upload was reached");
	expect_failed_card(card, FIRST_TICKET);

	/* a query the driver fails while an upload is waited for */
	card = open_card(card_index);
	upload_piece(card, 0, FIRST_TICKET);
	expect(fault_arm("backend_cuda_cuEventQuery", 1),
	       "the fault of the query is armed");
	expect_status(backend->upload_wait(card, FIRST_TICKET,
					   WAIT_LIMIT_NANOSECONDS),
		      STATUS_DEVICE_ERROR,
		      "a wait whose query fails is a device error");
	expect(fault_disarm() == 1, "the fault of the query was reached");
	expect_failed_card(card, FIRST_TICKET);

	/* a kernel the driver does not launch */
	card = open_card(card_index);
	expect(fault_arm("backend_cuda_cuLaunchKernel", 1),
	       "the fault of the launch is armed");
	expect_status(
		backend->digest(card, 0, WORD_BYTES, WAIT_LIMIT_NANOSECONDS,
				&digest),
		STATUS_DEVICE_ERROR,
		"a digest whose kernel is not launched is a device error");
	expect(fault_disarm() == 1 && digest == UNTOUCHED_DIGEST,
	       "the failed digest delivered nothing");
	expect_failed_card(card, FIRST_TICKET);

	/* a card whose context cannot be made current for a wait has reported a device error: it has failed for good */
	card = open_card(card_index);
	upload_piece(card, 0, FIRST_TICKET);
	expect(fault_arm("backend_cuda_cuCtxPushCurrent", 1),
	       "the fault of the context is armed");
	expect_status(
		backend->upload_wait(card, FIRST_TICKET,
				     WAIT_LIMIT_NANOSECONDS),
		STATUS_DEVICE_ERROR,
		"a wait for which the card's context cannot be made current is a device error");
	expect(fault_disarm() == 1, "the fault of the context was reached");
	expect_failed_card(card, FIRST_TICKET);

	/* an upload whose event the driver does not record: no ticket, though its copy was begun; the card has failed */
	card = open_card(card_index);
	expect(fault_arm("backend_cuda_cuEventRecord", 1),
	       "the fault of the event is armed");
	expect_status(
		backend->upload_begin(card, 0, mirror, PIECE_BYTE_COUNT,
				      &ticket),
		STATUS_DEVICE_ERROR,
		"an upload whose event is not recorded is a device error");
	expect(fault_disarm() == 1 && ticket == UNTOUCHED_TICKET,
	       "the upload without an event took no ticket");
	expect_failed_card(card, FIRST_TICKET);

	/* a sum the driver does not hand over */
	card = open_card(card_index);
	expect(fault_arm("backend_cuda_cuMemcpyDtoH", 1),
	       "the fault of the sum is armed");
	expect_status(backend->digest(card, 0, WORD_BYTES,
				      WAIT_LIMIT_NANOSECONDS, &digest),
		      STATUS_DEVICE_ERROR,
		      "a digest whose sum cannot be read is a device error");
	expect(fault_disarm() == 1 && digest == UNTOUCHED_DIGEST,
	       "the digest without a sum delivered nothing");
	expect_failed_card(card, FIRST_TICKET);

	/* an opening the driver fails leaves the card closed: the module, then the arena (the first allocation) */
	card = NULL;
	expect(fault_arm("backend_cuda_cuModuleLoadData", 1),
	       "the fault of the module is armed");
	expect_status(backend->card_open(card_index, ARENA_BYTE_COUNT, &card),
		      STATUS_DEVICE_ERROR,
		      "a card whose module does not load does not open");
	expect(fault_disarm() == 1 && card == NULL,
	       "the failed open left its output untouched");
	expect(fault_arm("backend_cuda_cuMemAlloc", 1),
	       "the fault of the arena is armed");
	expect_status(
		backend->card_open(card_index, ARENA_BYTE_COUNT, &card),
		STATUS_OUT_OF_MEMORY,
		"a card whose arena the driver does not give is out of memory");
	expect(fault_disarm() == 1 && card == NULL,
	       "the failed open left its output untouched");
	expect(fault_arm("backend_cuda_cuEventCreate", SECOND_CALL),
	       "the fault of the second event is armed");
	expect_status(backend->card_open(card_index, ARENA_BYTE_COUNT, &card),
		      STATUS_DEVICE_ERROR,
		      "a card whose events cannot all be made does not open");
	expect(fault_disarm() == SECOND_CALL && card == NULL,
	       "the open stopped at the event that failed");

	/* a description the driver fails: the context, then the free memory; the card's mutex is not left taken */
	expect(fault_arm("backend_cuda_cuCtxPushCurrent", 1),
	       "the fault of the context is armed");
	expect_status(
		backend->card_describe(card_index, &description),
		STATUS_DEVICE_ERROR,
		"a card whose context cannot be made current is not described");
	expect(fault_disarm() == 1, "the fault of the context was reached");
	expect(fault_arm("backend_cuda_cuMemGetInfo", 1),
	       "the fault of the free memory is armed");
	expect_status(
		backend->card_describe(card_index, &description),
		STATUS_DEVICE_ERROR,
		"a card whose free memory the driver does not tell is not described");
	expect(fault_disarm() == 1, "the fault of the free memory was reached");

	/* a link file that cannot be read counts as unknown: each of the four in turn, and all four are asked for */
	if (!describe_card(card_index, &description)) {
		return;
	}
    for (uint32_t unread = 0; unread < LINK_FILE_COUNT; unread++) {
	    expect(fault_arm("backend_cuda_link_read", unread + 1),
		   "the fault of a link file is armed");
	    expect_stated_link(card_index, description.bus_address, unread);
	    expect(fault_disarm() == LINK_FILE_COUNT,
		   "the four files of the link were asked for");
    }
}

#endif

/* Everything a test does with one card. */
static void test_card(uint32_t card_index)
{
    const int context_length = snprintf(context_text, sizeof(context_text), "card %" PRIu32, card_index);
    struct card_description_record description;
    struct card_description_record description_afterwards;
    struct device_card *card = NULL;

    if (context_length < 0 || (size_t)context_length >= sizeof(context_text) ||
        !describe_card(card_index, &description)) {
        expect(false, "the card can be named and described");
        return;
    }
    (void)printf("backend_cuda_card_test: card %" PRIu32 ": %s at %s, class %" PRIu32 ", link %" PRIu32 " x %" PRIu32
                 "\n",
                 card_index, description.name, description.bus_address, description.arithmetic_class,
                 description.link_generation, description.link_lane_count);
    expect_stated_link(card_index, description.bus_address, LINK_FILE_COUNT);    /* with every file read */
    if (description.memory_free_bytes < description.memory_reserved_bytes ||
        description.memory_free_bytes - description.memory_reserved_bytes < ARENA_BYTE_COUNT) {
        expect(false, "the cards are not free: this card has no room for the arena of the test");
        return;
    }
    test_card_cycle(card_index);
#if CHECKING_HOOKS_ENABLED
    test_driver_faults(card_index);
#endif
    /* nothing was left behind: the card opens, loads and closes a second time */
    card = open_card(card_index);
    if (card != NULL) {
        upload_piece(card, 1, FIRST_TICKET);
        expect(card_digest(card, PIECE_BYTE_COUNT, PIECE_BYTE_COUNT) == piece_digests[1],
               "a card opened a second time holds what is uploaded to it");
        backend->card_close(card, WAIT_LIMIT_NANOSECONDS);
    }
    /* and a third time, closed with no time to wait while every upload may still be in flight: the close returns
     * whatever their state (with an error line if one was unfinished), and the card's memory is free again */
    card = open_card(card_index);
    for (uint32_t piece_index = 0; card != NULL && piece_index < PIECE_COUNT; piece_index++) {
        upload_piece(card, piece_index, FIRST_TICKET + piece_index);
    }
    backend->card_close(card, 0);
    expect(describe_card(card_index, &description_afterwards) &&
               description_afterwards.memory_free_bytes + MEMORY_SLACK_BYTES >= description.memory_free_bytes,
           "a closed card has given its arena back");
}

struct card_thread {
    pthread_t thread;
    struct device_card *card;
    uint32_t upload_count;
    uint32_t uploads_beside_a_page_lock;        /* begun and answered while the driver page-locked one block */
    uint32_t wrong_answer_count;
};

static _Atomic uint32_t page_lock_round = 0;    /* the block the driver page-locks now, counted from one; 0 for none */
static _Atomic bool page_locks_are_over = false;
static uint64_t mirror_digest = 0;              /* the host's, of the whole mirror: set before the card threads start */

/* Uploads the mirror to its card, piece after piece and again from the first, until every piece has gone up and the
 * page-locking beside it is over; then the card's digest of its arena must be the host's digest of the mirror. */
static void *card_thread_run(void *argument)
{
    struct card_thread *const self = argument;
    uint64_t digest = UNTOUCHED_DIGEST;

    while (self->wrong_answer_count == 0 &&
           (self->upload_count < PIECE_COUNT || !atomic_load(&page_locks_are_over))) {
        const uint64_t byte_offset = self->upload_count % PIECE_COUNT * PIECE_BYTE_COUNT;
        const uint32_t round_before = atomic_load(&page_lock_round);
        uint64_t ticket = 0;
	const result_status status = backend->upload_begin(
		self->card, byte_offset, mirror_bytes(byte_offset),
		PIECE_BYTE_COUNT, &ticket);
	const bool is_beside = round_before != 0 && atomic_load(&page_lock_round) == round_before;
	if (status != STATUS_OK ||
	    backend->upload_wait(self->card, ticket, WAIT_LIMIT_NANOSECONDS) !=
		    STATUS_OK) {
		self->wrong_answer_count++;
	}
	self->uploads_beside_a_page_lock +=
		status == STATUS_OK && is_beside ? 1u : 0u;
	self->upload_count++;
    }
    if (backend->digest(self->card, 0, ARENA_BYTE_COUNT, WAIT_LIMIT_NANOSECONDS,
			&digest) != STATUS_OK ||
	digest != mirror_digest) {
	    self->wrong_answer_count++;
    }
    return NULL;
}

/* Calls for different cards run concurrently, and staging memory is obtained beside them: every card is loaded from
 * a thread of its own while this one page-locks blocks of the arena's size, as a loader does with its host arena. How
 * many uploads were begun and answered during a page-lock is printed, not judged: time and the driver decide it. */
static void test_cards_together(uint32_t card_count)
{
	static struct card_thread threads[BACKEND_MAXIMUM_CARDS];
	uint32_t started_count = 0;

	(void)snprintf(context_text, sizeof(context_text),
		       "the cards together");
	mirror_digest = host_digest(0, ARENA_BYTE_COUNT);
	for (uint32_t card_index = 0; card_index < card_count; card_index++) {
		threads[card_index].card = open_card(card_index);
	}
    while (started_count < card_count &&
           pthread_create(&threads[started_count].thread, NULL, card_thread_run, &threads[started_count]) == 0) {
        started_count++;
    }
    expect(started_count == card_count, "the card threads start");
    for (uint32_t round = 1; round <= STAGING_ROUNDS_BESIDE_THE_CARDS; round++) {
        void *block = NULL;
	result_status status = STATUS_OK;

	atomic_store(&page_lock_round, round);
        status = backend->staging_allocate(ARENA_BYTE_COUNT, &block);
        atomic_store(&page_lock_round, 0);
	expect_status(
		status, STATUS_OK,
		"a block of the arena's size is page-locked while the cards upload");
	backend->staging_free(block);
    }
    atomic_store(&page_locks_are_over, true);
    for (uint32_t card_index = 0; card_index < card_count; card_index++) {
        const struct card_thread *const self = &threads[card_index];
        if (card_index < started_count) {
            expect(pthread_join(self->thread, NULL) == 0, "a card thread is joined");
            expect(self->wrong_answer_count == 0, "a card loaded beside the others holds what was uploaded to it");
            (void)printf("backend_cuda_card_test: card %" PRIu32 ": %" PRIu32 " of %" PRIu32 " uploads were begun "
                         "and answered while a staging block was page-locked\n",
                         card_index, self->uploads_beside_a_page_lock, self->upload_count);
        }
        backend->card_close(self->card, WAIT_LIMIT_NANOSECONDS);
    }
}

int main(void)
{
    uint32_t card_count = 0;

    (void)alarm(TEST_TIME_LIMIT_SECONDS);
    run_in_child(test_machine_without_cards, "a machine without cards is answered as backend_seam.h says");
#if CHECKING_HOOKS_ENABLED
    run_in_child(test_failed_initialisation, "a driver that fails while it is initialised is answered as one");
#endif
    expect_status(backend->card_count(&card_count), STATUS_OK,
		  "the cards are counted");
    expect(card_count > 0 && card_count <= BACKEND_MAXIMUM_CARDS,
	   "the machine has cards for the test");
    if (card_count == 0 || card_count > BACKEND_MAXIMUM_CARDS) {
	    return EXIT_FAILURE; /* nothing to test on: what the children found is on the lines above */
    }
    test_refusals_without_a_card(card_count);
    test_staging_blocks();
    test_staging_from_threads();
    expect_status(backend->staging_allocate(ARENA_BYTE_COUNT, &mirror),
		  STATUS_OK, "the mirror of the arenas is page-locked");
    if (mirror == NULL) {
        return EXIT_FAILURE;
    }
    fill_mirror();
    for (uint32_t piece_index = 0; piece_index < PIECE_COUNT; piece_index++) {
        piece_digests[piece_index] = host_digest(piece_index * PIECE_BYTE_COUNT, PIECE_BYTE_COUNT);
    }
    /* one error line, and the block is left alone: every upload below still comes from it */
    backend->staging_free((unsigned char *)mirror +
			  BACKEND_STAGING_ALIGNMENT_BYTES);
    for (uint32_t card_index = 0; card_index < card_count; card_index++) {
        test_card(card_index);
    }
    test_cards_together(card_count);
    backend->staging_free(mirror);
    if (failures != 0) {
        return EXIT_FAILURE;
    }
    (void)printf("backend_cuda_card_test passed on %" PRIu32
		 " cards (checking hooks %s)\n",
		 card_count,
		 CHECKING_HOOKS_ENABLED ? "enabled" : "compiled out");
    return EXIT_SUCCESS;
}
