/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * loader_fault_test: what a card, the backend, the clock or the machine makes fail under a load of loader.h, and a
 * load that stalls. Each failure ends the load with its status, leaves the result untouched and returns what the
 * loader obtained, its host arena included, but for the staging slots of uploads it could not see finish. The
 * refusals of a request, the copies that fail and the extents that arrive wrong are in loader_test.
 *
 * Two packs are built once: the small one, and the striped one, which alternates its on-demand units between a card
 * and host memory, so that every card receives more pieces than it has staging slots. A case sets both test doubles
 * back, loads one pack under one placement, and ends with no card open, no staging block alive and no host arena
 * mapped.
 *
 * The watching backend keeps the staging blocks it hands out, so that the test can free what a load left. It notes
 * what is mapped when the first upload of a case begins, and it can make the clock fail at every look the loader
 * takes after it has asked about an upload: the fault of a clock fires once, so it is armed again each time.
 *
 * A host arena is seen as what it is, a mapping: the loader advises its arena to use 2 MiB pages, which no other
 * memory of a load is, and /proc/self/smaps shows that advice in the flags of a mapping. On a machine where a trial
 * does not show it, one line says so and what a load leaves mapped goes unchecked.
 */
#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#include "anemone/hooks.h"
#include "anemone/loader.h"
#include "anemone/pack.h"
#include "anemone/status.h"

#include "host_memory_backend.h"
#include "memory_storage.h"
#include "pack_builder.h"

#define TEST_TIME_LIMIT_SECONDS 900             /* the alarm ends the test here: the deadline of every wait in it */
#define CARD_COUNT 3
#define COPY_COUNT 3
#define THREADS_PER_COPY LOADER_DEFAULT_READER_THREADS_PER_COPY
#define SLOTS_PER_CARD LOADER_STAGING_SLOTS_PER_CARD
#define HOST PLACEMENT_IN_HOST_MEMORY /* short, for the placement tables */
#define CARD_MEMORY_BYTES (1ull << 30)
#define CARD_CLOSE_LIMIT_NANOSECONDS 1000000000ull
#define RESULT_FILL_BYTE 0xA5                   /* what a result holds before a load: a failed load leaves it */
#define SHORT_STALL_LIMIT_NANOSECONDS 5000000ull
#define FAILING_TICKET_IN_FLIGHT 5              /* an upload that fails while earlier ones are still in flight */
#define POLLS_OF_THE_FAILING_CARD 3
#define POLLS_OF_A_SLOW_UPLOAD 20               /* more looks than the small pack has pieces for one thread to read */
#define HUGE_PAGE_BYTES 2097152ull              /* 2 MiB: a host arena is mapped as a whole number of them */
#define NOT_SEEN UINT64_MAX                     /* in place of a count of mapped bytes that could not be taken */
#define SMAPS_LINE_CAPACITY 1024                /* above every line of /proc/self/smaps: a longer one comes in parts */
#define HEXADECIMAL_BASE 16
#define SMALL_SEED 0x89ABCDEull
#define STRIPED_SEED 0xF0F0F0Full

/* The small pack. */
#define SMALL_UNIT_COUNT 7
static const struct test_pack_unit_plan small_units[SMALL_UNIT_COUNT] = {
	{ PLACEMENT_UNIT_STAGE, 0, 1, 6 },
	{ PLACEMENT_UNIT_ON_DEMAND, 0, 4, 1 },
	{ PLACEMENT_UNIT_ON_DEMAND, 0, 3, 2 },
	{ PLACEMENT_UNIT_STAGE, 1, 2, 3 },
	{ PLACEMENT_UNIT_ON_DEMAND, 1, 5, 1 },
	{ PLACEMENT_UNIT_STAGE, 2, 1, 4 },
	{ PLACEMENT_UNIT_ON_DEMAND, 2, 2, 2 },
};
static const uint32_t small_placement[SMALL_UNIT_COUNT] = {0, HOST, 0, 1, HOST, 2, 2};

/* The striped pack: every stage unit is followed by on-demand units of one block that go to its card and to host
 * memory in turn. */
#define STRIPED_ON_DEMAND_UNITS_PER_STAGE 24
#define STRIPED_UNITS_PER_STAGE (STRIPED_ON_DEMAND_UNITS_PER_STAGE + 1)
#define STRIPED_UNIT_COUNT (CARD_COUNT * STRIPED_UNITS_PER_STAGE)
#define STRIPED_STAGE_BLOCKS 2
#define STRIPE_PERIOD 2
static struct test_pack_unit_plan striped_units[STRIPED_UNIT_COUNT];
static uint32_t striped_placement[STRIPED_UNIT_COUNT];

struct fixture {
    void *pack_bytes;
    uint64_t pack_byte_count;
    struct pack_record *pack;
    uint32_t unit_count;
};

/* One load: what it is given, and what came back. */
struct load_case {
    const struct fixture *fixture;
    uint32_t card_count;
    bool has_loaded;                            /* the result holds what a load delivered */
    struct storage_reader *reader;
    struct load_layout *layout;
    struct device_card *cards[BACKEND_MAXIMUM_CARDS];
    struct card_description_record card_descriptions[BACKEND_MAXIMUM_CARDS];
    struct load_request request;
    struct load_result result;
};

static const char *const copy_paths[STORAGE_MAXIMUM_COPIES] = {
	"copy a", "copy b", "copy c", "copy d",
	"copy e", "copy f", "copy g", "copy h",
};

static int failures = 0;
static const char *case_name = "before the first case";

static void expect(bool condition, const char *what)
{
    if (!condition) {
        (void)fprintf(stderr, "FAILED: %s: %s\n", case_name, what);
        failures++;
    }
}

/* For what the test cannot go on without. */
static _Noreturn void give_up(const char *what)
{
    (void)fprintf(stderr, "FAILED: %s: %s\n", case_name, what);
    exit(EXIT_FAILURE);
}

static struct backend_operations
	watching_backend; /* the test backend behind the calls below */
static pthread_mutex_t watch_mutex = PTHREAD_MUTEX_INITIALIZER; /* guards the kept blocks */
static void *kept_blocks
	[BACKEND_MAXIMUM_STAGING_BLOCKS]; /* handed out here and not yet freed here */
static bool advised_mappings_are_seen = false;                  /* this machine shows what a load has mapped */
static uint64_t advised_bytes_before_the_case = NOT_SEEN;
static _Atomic bool first_upload_is_noted = true;               /* false while a case waits for its first upload */
static _Atomic uint64_t advised_bytes_at_the_first_upload = NOT_SEEN;

/* The bytes of this process that are mapped with the advice to use 2 MiB pages: every mapping of /proc/self/smaps
 * whose flags hold "hg". NOT_SEEN when the file cannot be read. */
static uint64_t advised_mapping_bytes(void)
{
    char line[SMAPS_LINE_CAPACITY];
    uint64_t mapping_byte_count = 0;
    uint64_t advised_byte_count = 0;
    FILE *const smaps = fopen("/proc/self/smaps", "r");
    if (smaps == NULL) {
        return NOT_SEEN;
    }
    while (fgets(line, sizeof(line), smaps) != NULL) {
        char *end = line;                       /* where the number ends: at the line's start when there is none */
        const unsigned long long first_address = strtoull(line, &end, HEXADECIMAL_BASE);
        if (end != line && *end == '-') {       /* a mapping begins: "<first address>-<the address behind it> ..." */
            mapping_byte_count = strtoull(end + 1, NULL, HEXADECIMAL_BASE) - first_address;
        } else if (strncmp(line, "VmFlags:", strlen("VmFlags:")) == 0 && strstr(line, " hg ") != NULL) {
            advised_byte_count += mapping_byte_count;
        }
    }
    (void)fclose(smaps);
    return advised_byte_count;
}

/* Finds out whether a host arena is seen here, apart from the other memory of a load: a staging block of the test
 * backend, which must not count, and a mapping advised as the loader advises its arena, which must, are made,
 * looked for and returned. Where that is not so, one line says what goes unchecked. */
static void look_for_advised_mappings(void)
{
    const uint64_t byte_count_before = advised_mapping_bytes();
    void *staging_block = NULL;
    const bool has_block = test_host_memory_backend.staging_allocate(
				   LOADER_CHUNK_BYTES, &staging_block) ==
			   STATUS_OK;
    const uint64_t byte_count_with_a_block = advised_mapping_bytes();
    void *const mapping = mmap(NULL, HUGE_PAGE_BYTES, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    const bool is_advised = mapping != MAP_FAILED && madvise(mapping, HUGE_PAGE_BYTES, MADV_HUGEPAGE) == 0;
    const uint64_t byte_count_with_both = advised_mapping_bytes();
    const bool is_returned = mapping != MAP_FAILED && munmap(mapping, HUGE_PAGE_BYTES) == 0;
    test_host_memory_backend.staging_free(staging_block);
    advised_mappings_are_seen = has_block && is_advised && is_returned && byte_count_before != NOT_SEEN &&
                                byte_count_with_a_block == byte_count_before &&
                                byte_count_with_both == byte_count_before + HUGE_PAGE_BYTES &&
                                advised_mapping_bytes() == byte_count_before;
    if (!advised_mappings_are_seen) {
        (void)fprintf(stderr, "loader_fault_test: this machine does not show which mappings are advised to use "
                              "2 MiB pages: what a load leaves mapped is not checked\n");
    }
}

/* True when so many bytes are mapped with the advice now, and where this machine does not show them. */
static bool advised_mapping_bytes_are(uint64_t expected_byte_count)
{
    return !advised_mappings_are_seen || advised_mapping_bytes() == expected_byte_count;
}

/* Notes, before the first upload of a case that asks for it begins, what is mapped with the advice. */
MUST_CHECK static result_status
watching_upload_begin(struct device_card *card, uint64_t arena_byte_offset,
		      const void *page_locked_source, uint64_t byte_count,
		      uint64_t *upload_ticket_out)
{
    if (!atomic_exchange(&first_upload_is_noted, true)) {
        atomic_store(&advised_bytes_at_the_first_upload, advised_mapping_bytes());
    }
    return test_host_memory_backend.upload_begin(card, arena_byte_offset, page_locked_source, byte_count,
                                                 upload_ticket_out);
}

/* Keeps the block it hands out, so that the test can free it when a load has left it allocated. */
MUST_CHECK static result_status
watching_staging_allocate(uint64_t byte_count, void **page_locked_memory_out)
{
    void *memory = NULL;
    const result_status status =
	    test_host_memory_backend.staging_allocate(byte_count, &memory);
    if (status != STATUS_OK) {
	    return status;
    }
    (void)pthread_mutex_lock(&watch_mutex);
    for (uint32_t block_index = 0; block_index < BACKEND_MAXIMUM_STAGING_BLOCKS;
	 block_index++) {
	    if (kept_blocks[block_index] == NULL) {
		    kept_blocks[block_index] = memory;
		    break;
	    }
    }
    (void)pthread_mutex_unlock(&watch_mutex);
    *page_locked_memory_out = memory;
    return STATUS_OK;
}

static void watching_staging_free(void *page_locked_memory)
{
    (void)pthread_mutex_lock(&watch_mutex);
    for (uint32_t block_index = 0; block_index < BACKEND_MAXIMUM_STAGING_BLOCKS;
	 block_index++) {
	    if (page_locked_memory != NULL &&
		kept_blocks[block_index] == page_locked_memory) {
		    kept_blocks[block_index] = NULL;
	    }
    }
    (void)pthread_mutex_unlock(&watch_mutex);
    test_host_memory_backend.staging_free(page_locked_memory);
}

#if CHECKING_HOOKS_ENABLED

static _Atomic bool clock_fails_after_a_question = false;
static _Atomic uint64_t looks_at_the_failing_clock = 0;         /* while its fault was armed, up to the last arming */

/* When a case says so, the loader's next look at the clock fails: the looks since the fault was armed last are
 * counted first, so that the test knows the fault point was reached. */
MUST_CHECK static result_status
watching_upload_wait(struct device_card *card, uint64_t upload_ticket,
		     uint64_t wait_limit_nanoseconds)
{
    if (atomic_load(&clock_fails_after_a_question)) {
	    (void)atomic_fetch_add(&looks_at_the_failing_clock, fault_disarm());
	    if (!fault_arm("loader_clock", 1)) {
		    give_up("the fault of the clock is armed again");
	    }
    }
    return test_host_memory_backend.upload_wait(card, upload_ticket, wait_limit_nanoseconds);
}

#endif

/* Frees what a load left allocated, once its cards are closed, and says how many blocks that were. */
static uint32_t free_kept_blocks(void)
{
    uint32_t freed_block_count = 0;
    (void)pthread_mutex_lock(&watch_mutex);
    for (uint32_t block_index = 0; block_index < BACKEND_MAXIMUM_STAGING_BLOCKS;
	 block_index++) {
	    if (kept_blocks[block_index] != NULL) {
		    test_host_memory_backend.staging_free(
			    kept_blocks[block_index]);
		    kept_blocks[block_index] = NULL;
		    freed_block_count++;
	    }
    }
    (void)pthread_mutex_unlock(&watch_mutex);
    return freed_block_count;
}

static void fixture_build(struct fixture *fixture, const struct test_pack_unit_plan *units, uint32_t unit_count,
                          uint64_t content_seed)
{
    const struct test_pack_plan plan = {
        .model_name = "loader_fault_test", .units = units, .unit_count = unit_count, .content_seed = content_seed};
    struct storage_reader *reader = NULL;
    memset(fixture, 0, sizeof(*fixture));
    if (test_pack_build(&plan, &fixture->pack_bytes,
			&fixture->pack_byte_count) != STATUS_OK) {
	    give_up("a pack is built");
    }
    /* a pack reads nothing after it was opened: the reader and the storage's memory of the pack may go */
    test_memory_storage_reset();
    test_memory_storage_set_pack(fixture->pack_bytes, fixture->pack_byte_count);
    if (test_memory_storage.reader_open(copy_paths, 1, &reader) != STATUS_OK ||
	pack_open_entry(&test_memory_storage, reader, &fixture->pack) !=
		STATUS_OK) {
	    give_up("a pack is opened");
    }
    test_memory_storage.reader_close(reader);
    fixture->unit_count = unit_count;
}

/* Sets both test doubles and the watching backend back, opens the copies and lays the pack out. A test then sets
 * its faults, and calls case_prepare. */
static void case_begin(struct load_case *load_case, const char *name, const struct fixture *fixture,
                       const uint32_t *card_of_unit, uint32_t copy_count)
{
    case_name = name;
    expect(test_host_memory_backend_open_card_count() == 0 && test_host_memory_backend_staging_block_count() == 0,
           "a case begins with no card open and no staging block alive");
    memset(load_case, 0, sizeof(*load_case));
    load_case->fixture = fixture;
    load_case->card_count = CARD_COUNT;
    test_host_memory_backend_reset(CARD_COUNT, CARD_MEMORY_BYTES);
    test_memory_storage_reset();
    test_memory_storage_set_pack(fixture->pack_bytes, fixture->pack_byte_count);
    atomic_store(&first_upload_is_noted, true);
    atomic_store(&advised_bytes_at_the_first_upload, NOT_SEEN);
    advised_bytes_before_the_case = advised_mapping_bytes();
    if (test_memory_storage.reader_open(copy_paths, copy_count,
					&load_case->reader) != STATUS_OK ||
	load_layout_create(fixture->pack, card_of_unit, fixture->unit_count,
			   CARD_COUNT, &load_case->layout) != STATUS_OK) {
	    give_up("the copies of a case are opened and its pack is laid out");
    }
}

/* Describes the cards, opens those that receive bytes with an arena of exactly what the layout puts on them, and
 * fills the request: a test then changes what its case is about. */
static void case_prepare(struct load_case *load_case, uint32_t reader_threads_per_copy)
{
    for (uint32_t card_index = 0; card_index < load_case->card_count; card_index++) {
        const uint64_t arena_byte_count = load_case->layout->card_byte_count[card_index];
	if (test_host_memory_backend.card_describe(
		    card_index, &load_case->card_descriptions[card_index]) !=
		    STATUS_OK ||
	    (arena_byte_count != 0 &&
	     test_host_memory_backend.card_open(
		     card_index, arena_byte_count,
		     &load_case->cards[card_index]) != STATUS_OK)) {
		give_up("the cards of a case are described and opened");
	}
    }
    load_case->request = (struct load_request){
	    .storage = &test_memory_storage,
	    .reader = load_case->reader,
	    .backend = &watching_backend,
	    .pack = load_case->fixture->pack,
	    .layout = load_case->layout,
	    .cards = load_case->cards,
	    .card_descriptions = load_case->card_descriptions,
	    .reader_threads_per_copy = reader_threads_per_copy,
	    .stall_limit_nanoseconds = LOADER_DEFAULT_STALL_LIMIT_NANOSECONDS,
    };
}

static bool result_is_untouched(const struct load_case *load_case)
{
    const unsigned char *const result_bytes = (const unsigned char *)&load_case->result;
    for (size_t byte_index = 0; byte_index < sizeof(load_case->result); byte_index++) {
        if (result_bytes[byte_index] != RESULT_FILL_BYTE) {
            return false;
        }
    }
    return true;
}

MUST_CHECK static result_status case_run(struct load_case *load_case)
{
    memset(&load_case->result, RESULT_FILL_BYTE, sizeof(load_case->result));
    const result_status status =
	    load_run_entry(&load_case->request, &load_case->result);
    load_case->has_loaded = status == STATUS_OK;
    return status;
}

/* Closes the cards before the host arena is returned, as a program does, and checks that the case leaves nothing.
 * A load that failed gave no arena: no arena is returned in its place, which is accepted and does nothing, so an
 * arena that such a load forgot to unmap is still seen here. */
static void case_end(struct load_case *load_case)
{
	for (uint32_t card_index = 0; card_index < BACKEND_MAXIMUM_CARDS;
	     card_index++) {
		test_host_memory_backend.card_close(
			load_case->cards[card_index],
			CARD_CLOSE_LIMIT_NANOSECONDS);
		load_case->cards[card_index] = NULL;
	}
	load_host_arena_release(
		load_case->has_loaded ? load_case->result.host_arena : NULL,
		load_case->layout->host_byte_count);
	expect(free_kept_blocks() == 0,
	       "nothing the loader obtained is left for the test to free");
	test_memory_storage.reader_close(load_case->reader);
	load_layout_destroy(load_case->layout);
	expect(test_host_memory_backend_open_card_count() == 0 &&
		       test_host_memory_backend_staging_block_count() == 0,
	       "a case ends with no card open and no staging block alive");
	expect(advised_mapping_bytes_are(advised_bytes_before_the_case),
	       "a case ends with no host arena mapped");
}

/* A load that must fail with that status, leave the result untouched and return everything it obtained. */
static void expect_failed_load(struct load_case *load_case,
			       result_status expected_status, const char *what)
{
    expect(case_run(load_case) == expected_status, what);
    expect(result_is_untouched(load_case), "a failed load leaves the result untouched");
    expect(test_host_memory_backend_staging_block_count() == 0,
           "a failed load returns every staging block it obtained");
}

/* A load that failed because a card failed its uploads: the result untouched, and only slots of that card left
 * allocated (a failed card may still read them), which the test frees once the cards are closed. */
static void expect_failed_card_load(struct load_case *load_case, const char *what)
{
	expect(case_run(load_case) == STATUS_DEVICE_ERROR, what);
	expect(result_is_untouched(load_case),
	       "a failed load leaves the result untouched");
	const uint32_t slots_left =
		test_host_memory_backend_staging_block_count();
	expect(slots_left >= 1 && slots_left <= SLOTS_PER_CARD,
	       "only slots of the failed card stay allocated");
	for (uint32_t card_index = 0; card_index < CARD_COUNT; card_index++) {
		test_host_memory_backend.card_close(
			load_case->cards[card_index],
			CARD_CLOSE_LIMIT_NANOSECONDS);
		load_case->cards[card_index] = NULL;
	}
    expect(free_kept_blocks() == slots_left, "once its card is closed, a slot that was left is freed");
}

/* A load that timed out with uploads it could not see finish: the result untouched, and exactly the slots of those
 * uploads left allocated, which the test frees once their cards are closed. */
static void expect_slots_left(struct load_case *load_case, uint32_t slot_count)
{
    expect(result_is_untouched(load_case), "a load that timed out leaves the result untouched");
    expect(test_host_memory_backend_staging_block_count() == slot_count,
           "the slots of uploads that were not seen to finish stay allocated, and nothing else does");
    for (uint32_t card_index = 0; card_index < CARD_COUNT; card_index++) {
        test_host_memory_backend.card_close(load_case->cards[card_index], CARD_CLOSE_LIMIT_NANOSECONDS);
        load_case->cards[card_index] = NULL;
    }
    expect(free_kept_blocks() == slot_count, "once its card is closed, a slot that was left is freed");
}

static void test_failing_cards(const struct fixture *small, const struct fixture *striped)
{
    /* the first slot, a later slot, and the last one: the host arena is no staging block */
    static const uint64_t failing_allocations[] = {1, SLOTS_PER_CARD + 2, (uint64_t)CARD_COUNT * SLOTS_PER_CARD};
    struct load_case load_case;

    case_begin(&load_case, "a failed upload", small, small_placement, COPY_COUNT);
    test_host_memory_backend_fail_upload(1, 1);
    case_prepare(&load_case, THREADS_PER_COPY);
    expect_failed_card_load(&load_case, "a failed upload fails the load");
    case_end(&load_case);

    case_begin(&load_case, "an upload that fails behind uploads in flight", striped, striped_placement, COPY_COUNT);
    test_host_memory_backend_set_polls_before_an_upload_finishes(0, POLLS_OF_THE_FAILING_CARD);
    test_host_memory_backend_fail_upload(0, FAILING_TICKET_IN_FLIGHT);
    case_prepare(&load_case, THREADS_PER_COPY);
    expect_failed_card_load(&load_case, "a failed upload fails the load");
    case_end(&load_case);

    case_begin(&load_case, "a failed digest", small, small_placement, COPY_COUNT);
    test_host_memory_backend_fail_digest(CARD_COUNT - 1, 1);
    case_prepare(&load_case, THREADS_PER_COPY);
    expect_failed_load(&load_case, STATUS_DEVICE_ERROR,
		       "a failed digest fails the load");
    case_end(&load_case);

    for (size_t failure = 0; failure < sizeof(failing_allocations) / sizeof(failing_allocations[0]); failure++) {
        case_begin(&load_case, "a failed staging allocation", small, small_placement, COPY_COUNT);
        case_prepare(&load_case, THREADS_PER_COPY);
        test_host_memory_backend_fail_staging_allocation(failing_allocations[failure]);
	expect_failed_load(&load_case, STATUS_OUT_OF_MEMORY,
			   "a lack of staging memory fails the load");
	case_end(&load_case);
    }
}

/* Uploads that never finish: the stall rule ends the load, which leaves the slots those uploads read. A thread
 * finds nothing to take, and so looks at the stall rule, only when the slots of every card are all in use: each of
 * them holds an upload by the time the threads are joined, however they were scheduled. */
static void test_stall(const struct fixture *striped)
{
    struct load_case load_case;
    case_begin(&load_case, "uploads that never finish", striped, striped_placement, COPY_COUNT);
    for (uint32_t card_index = 0; card_index < CARD_COUNT; card_index++) {
        test_host_memory_backend_set_polls_before_an_upload_finishes(card_index, UINT32_MAX);
    }
    case_prepare(&load_case, THREADS_PER_COPY);
    load_case.request.stall_limit_nanoseconds = SHORT_STALL_LIMIT_NANOSECONDS;
    expect(case_run(&load_case) == STATUS_TIMED_OUT,
	   "a load whose uploads never finish times out");
    expect_slots_left(&load_case, CARD_COUNT * SLOTS_PER_CARD);
    case_end(&load_case);
}

/* The host arena is there before the first upload begins: one mapping of whole 2 MiB pages, advised as such, which
 * the caller returns. So host memory waits for no card: one reader thread, which serves the cards first, has read
 * every piece of host memory into it before uploads that take their time are seen to finish. */
static void test_host_arena(const struct fixture *small)
{
    struct load_case load_case;
    bool host_memory_arrived_first = true;
    case_begin(&load_case, "a host arena that is there before the first upload", small, small_placement, 1);
    for (uint32_t card_index = 0; card_index < CARD_COUNT; card_index++) {
        test_host_memory_backend_set_polls_before_an_upload_finishes(card_index, POLLS_OF_A_SLOW_UPLOAD);
    }
    case_prepare(&load_case, 1);
    const uint64_t host_byte_count = load_case.layout->host_byte_count;
    const uint64_t arena_byte_count = (host_byte_count + HUGE_PAGE_BYTES - 1) / HUGE_PAGE_BYTES * HUGE_PAGE_BYTES;
    atomic_store(&first_upload_is_noted, false);
    expect(host_byte_count != 0 && case_run(&load_case) == STATUS_OK,
	   "a load with host memory succeeds");
    if (load_case.has_loaded) {
	    const struct load_result *const result = &load_case.result;
	    for (uint32_t card_index = 0; card_index < CARD_COUNT;
		 card_index++) {
		    host_memory_arrived_first =
			    host_memory_arrived_first &&
			    result->host_arrival_nanoseconds != 0 &&
			    result->host_arrival_nanoseconds <
				    result->card_arrival_nanoseconds[card_index];
	    }
        expect(atomic_load(&first_upload_is_noted) && result->host_arena != NULL,
               "the cards received uploads, and the load gave its host arena");
        expect(!advised_mappings_are_seen ||
                   atomic_load(&advised_bytes_at_the_first_upload) == advised_bytes_before_the_case + arena_byte_count,
               "the host arena was mapped, as whole 2 MiB pages and advised as such, before the first upload began");
        expect(advised_mapping_bytes_are(advised_bytes_before_the_case + arena_byte_count),
               "a load that succeeds leaves its host arena mapped, and nothing beside it");
        expect(host_memory_arrived_first,
               "with one thread, host memory had every piece before an upload of any card was seen to finish");
    }
    case_end(&load_case);
}

#if CHECKING_HOOKS_ENABLED

static void test_injected_faults(const struct fixture *small)
{
    /* the units of each stage are neighbours in the pack: with all of them on its card, a card receives one piece */
    static const uint32_t small_placement_on_cards_only[SMALL_UNIT_COUNT] = {0, 0, 0, 1, 1, 2, 2};
    struct load_case load_case;

    case_begin(&load_case, "no memory for the loader's state", small, small_placement, COPY_COUNT);
    case_prepare(&load_case, THREADS_PER_COPY);
    expect(fault_arm("loader_state_allocation", 1),
	   "the fault of the loader's memory is armed");
    expect_failed_load(&load_case, STATUS_OUT_OF_MEMORY,
		       "a lack of memory fails the load");
    expect(fault_disarm() == 1,
	   "the fault point of the loader's memory was reached");
    case_end(&load_case);

    /* the arena is mapped behind the staging slots and before any thread: the slots go back, and nothing is read */
    case_begin(&load_case, "a host arena that cannot be mapped", small, small_placement, COPY_COUNT);
    case_prepare(&load_case, THREADS_PER_COPY);
    expect(fault_arm("loader_host_arena", 1),
	   "the fault of the host arena is armed");
    expect_failed_load(&load_case, STATUS_OUT_OF_MEMORY,
		       "a host arena that cannot be mapped fails the load");
    expect(fault_disarm() == 1,
	   "the fault point of the host arena was reached");
    for (uint32_t copy_index = 0; copy_index < COPY_COUNT; copy_index++) {
        expect(test_memory_storage_read_count(copy_index) == 0, "a load without its host arena reads nothing");
    }
    case_end(&load_case);

    for (uint32_t thread_number = 1; thread_number <= COPY_COUNT * THREADS_PER_COPY; thread_number++) {
        case_begin(&load_case, "a reader thread that cannot be created", small, small_placement, COPY_COUNT);
        case_prepare(&load_case, THREADS_PER_COPY);
	expect(fault_arm("loader_thread_create", thread_number),
	       "the fault of a thread is armed");
	expect_failed_load(&load_case, STATUS_OUT_OF_MEMORY,
			   "a thread that cannot be created fails the load");
	expect(fault_disarm() == thread_number,
	       "no thread is created behind the one that failed");
	case_end(&load_case);
    }

    /* One thread, one piece for each card, uploads that never finish and a stall limit that is minutes away. The
     * clock fails at the loader's first look and at every look that follows a question about an upload: a wait
     * that cannot be timed ends at once, the one for the stall and then the one for each upload left in flight. */
    case_begin(&load_case, "a clock that cannot be read", small, small_placement_on_cards_only, 1);
    for (uint32_t card_index = 0; card_index < CARD_COUNT; card_index++) {
        test_host_memory_backend_set_polls_before_an_upload_finishes(card_index, UINT32_MAX);
    }
    case_prepare(&load_case, 1);
    atomic_store(&looks_at_the_failing_clock, 0);
    atomic_store(&clock_fails_after_a_question, true);
    expect(fault_arm("loader_clock", 1), "the fault of the clock is armed");
    expect(case_run(&load_case) == STATUS_TIMED_OUT,
	   "a load that cannot time its waits ends them at once: it times out");
    atomic_store(&clock_fails_after_a_question, false);
    expect(atomic_load(&looks_at_the_failing_clock) + fault_disarm() != 0,
	   "the fault point of the clock was reached");
    expect_slots_left(&load_case, CARD_COUNT);
    case_end(&load_case);
}

#endif

int main(void)
{
    struct fixture small;
    struct fixture striped;
    (void)alarm(TEST_TIME_LIMIT_SECONDS);
    for (uint32_t unit_index = 0; unit_index < STRIPED_UNIT_COUNT; unit_index++) {
        const uint32_t stage_index = unit_index / STRIPED_UNITS_PER_STAGE;
        const uint32_t unit_in_stage = unit_index % STRIPED_UNITS_PER_STAGE;
        const bool is_in_host_memory = unit_in_stage != 0 && unit_in_stage % STRIPE_PERIOD == 0;
	striped_units[unit_index] = (struct test_pack_unit_plan){
		.unit_kind = unit_in_stage == 0 ? PLACEMENT_UNIT_STAGE :
						  PLACEMENT_UNIT_ON_DEMAND,
		.layer_index = stage_index,
		.extent_count = 1,
		.blocks_per_extent = unit_in_stage == 0 ? STRIPED_STAGE_BLOCKS :
							  1
	};
	striped_placement[unit_index] = is_in_host_memory ? HOST : stage_index;
    }
    watching_backend = test_host_memory_backend;
    watching_backend.staging_allocate = watching_staging_allocate;
    watching_backend.staging_free = watching_staging_free;
    watching_backend.upload_begin = watching_upload_begin;
#if CHECKING_HOOKS_ENABLED
    watching_backend.upload_wait = watching_upload_wait;
#endif
    look_for_advised_mappings();
    fixture_build(&small, small_units, SMALL_UNIT_COUNT, SMALL_SEED);
    fixture_build(&striped, striped_units, STRIPED_UNIT_COUNT, STRIPED_SEED);

    test_failing_cards(&small, &striped);
    test_stall(&striped);
    test_host_arena(&small);
#if CHECKING_HOOKS_ENABLED
    test_injected_faults(&small);
#endif

    pack_close(striped.pack);
    pack_close(small.pack);
    free(striped.pack_bytes);
    free(small.pack_bytes);
    if (failures != 0) {
        return EXIT_FAILURE;
    }
    (void)printf("loader_fault_test passed (checking hooks %s)\n",
		 CHECKING_HOOKS_ENABLED ? "enabled" : "compiled out");
    return EXIT_SUCCESS;
}
