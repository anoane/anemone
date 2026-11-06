/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * loader_test: the layout and the load of loader.h against the storage and the backend in host memory, with every
 * refusal the contract names provoked once, copies that fail and extents that arrive wrong. What a card, the clock
 * or the machine makes fail, and a load that stalls, is in loader_fault_test.
 *
 * Three packs are built once. The large one holds what the cutting of pieces must get right: an extent of several
 * chunks for a card and one for host memory, runs of small extents that merge into one piece, extents of one
 * block. The small one serves the refusals. The striped one alternates its on-demand units between a card and
 * host memory, so that every card receives more pieces than it has staging slots. A case sets both test doubles
 * back, loads one pack under one placement, and ends with its host arena returned, no card open and no staging
 * block alive.
 *
 * Where the scheduler would decide what a case shows, a wrapper decides: the scripted storage holds copies back,
 * steers a read to a chosen copy or describes a reader the loader must refuse; the watching backend notes the
 * order of the uploads and counts the staging blocks a load asks for and the waits it spends on an upload.
 */
#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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
#define RESULT_FILL_BYTE 0xA5                   /* what a result holds before a load: a refused load leaves it */
#define FLIPPED_BYTE_INDEX 5                    /* inside a block: any byte of it will do */
#define POLLS_BEFORE_AN_UPLOAD_FINISHES 20      /* more than a card has slots: one thread fills the ring */
#define SLOW_CARD 1
#define SLOW_LINK_GENERATION 1
#define SLOW_LINK_LANE_COUNT 1
#define FAST_LINK_GENERATION 4
#define FAST_LINK_LANE_COUNT 16
#define UNKNOWN_LINK_GENERATION 9
#define TWO_CHUNKS (2 * LOADER_CHUNK_BYTES)
#define HUGE_PAGE_BYTES 2097152ull              /* 2 MiB: a host arena is mapped as a whole number of them */
#define NOT_SEEN UINT64_MAX                     /* in place of a count that was not taken yet */
#define AS_THE_READER_HAS_IT UINT32_MAX         /* in place of a count or an alignment the script describes */
#define READS_BEFORE_THE_COPY_FAILS 2
#define STEERED_READ_COUNT 2
#define ALIGNMENT_THAT_DIVIDES_NO_BLOCK (2 * PACK_BLOCK_BYTES)
#define LARGE_SEED 0x1234567ull
#define SMALL_SEED 0x89ABCDEull
#define STRIPED_SEED 0xF0F0F0Full

/* The large pack. Card 0 takes a run of 8,496 blocks, which is three pieces: the last one ends the first extent
 * and holds the 48 extents of one block behind it. Card 1 takes two pieces, card 2 three, host memory four. */
#define LARGE_UNIT_COUNT 10
#define LARGE_PIECE_COUNT 12
static const struct test_pack_unit_plan large_units[LARGE_UNIT_COUNT] = {
	{ PLACEMENT_UNIT_STAGE, 0, 1, 8448 },
	{ PLACEMENT_UNIT_ON_DEMAND, 0, 48, 1 },
	{ PLACEMENT_UNIT_ON_DEMAND, 0, 3, 5 },
	{ PLACEMENT_UNIT_STAGE, 1, 2, 700 },
	{ PLACEMENT_UNIT_ON_DEMAND, 1, 40, 2 },
	{ PLACEMENT_UNIT_ON_DEMAND, 1, 1, 3 },
	{ PLACEMENT_UNIT_STAGE, 2, 1, 4200 },
	{ PLACEMENT_UNIT_ON_DEMAND, 2, 1, 4300 },
	{ PLACEMENT_UNIT_HOST_ONLY, PACK_NO_LAYER, 5, 9 },
	{ PLACEMENT_UNIT_STAGE, PACK_NO_LAYER, 1, 1 },
};
static const uint32_t large_placement[LARGE_UNIT_COUNT] = {0, 0, HOST, 1, HOST, 1, 2, HOST, HOST, 2};
static const uint64_t large_upload_counts[CARD_COUNT] = {3, 2, 3};
#define LARGE_UNIT_OF_SEVERAL_CHUNKS_ON_A_CARD 0
#define LARGE_UNIT_OF_SEVERAL_CHUNKS_IN_HOST_MEMORY 7

/* The small pack: no host-only unit, so that a placement can leave host memory empty. */
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
static const uint32_t small_placement_on_cards_only[SMALL_UNIT_COUNT] = {0, 0, 0, 1, 1, 2, 2};
static const uint32_t small_placement_in_host_memory[SMALL_UNIT_COUNT] = {0, HOST, HOST, 1, HOST, 2, HOST};
#define SMALL_UNIT_ON_A_CARD 3
#define SMALL_UNIT_IN_HOST_MEMORY 1

/* The striped pack: every stage unit is followed by on-demand units of one block that go to its card and to host
 * memory in turn. No two of them are neighbours in a lane, so each is a piece of its own. */
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
    const struct pack_extent *extents;
    const struct pack_unit_record_2 *units;
    uint32_t extent_count;
    uint32_t unit_count;
    uint64_t content_seed;
    uint64_t extent_byte_total;
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

static struct storage_operations_record
	scripted_storage; /* the memory storage behind the script below */
static pthread_mutex_t script_mutex = PTHREAD_MUTEX_INITIALIZER; /* guards the script */
static pthread_cond_t script_copies_released = PTHREAD_COND_INITIALIZER;
static bool script_copies_are_held_back = false;                /* every copy but one waits for that one */
static uint32_t script_awaited_copy_index = 0;
static uint64_t script_awaited_reads_left = 0;                  /* reads of that copy until the others go on */
static bool script_steering_is_pending[STEERED_READ_COUNT];     /* the first read of a byte comes from a copy */
static uint64_t script_steered_byte_offset[STEERED_READ_COUNT];
static uint32_t script_steered_copy_index[STEERED_READ_COUNT];
static result_status script_description_status = STATUS_OK;
static uint32_t script_described_copy_count = AS_THE_READER_HAS_IT;
static uint32_t script_described_block_alignment_bytes = AS_THE_READER_HAS_IT;

static void script_reset(void)
{
    (void)pthread_mutex_lock(&script_mutex);
    script_copies_are_held_back = false;
    script_awaited_reads_left = 0;
    memset(script_steering_is_pending, 0, sizeof(script_steering_is_pending));
    script_description_status = STATUS_OK;
    script_described_copy_count = AS_THE_READER_HAS_IT;
    script_described_block_alignment_bytes = AS_THE_READER_HAS_IT;
    (void)pthread_mutex_unlock(&script_mutex);
}

/* Every other copy waits in its reads until this copy was asked for read_count reads: the copy a fault was set on
 * is then read for certain, however the threads are scheduled. The pack must have more pieces than the load has
 * threads, or the thread of this copy finds nothing left to read. */
static void script_hold_copies_back_until(uint32_t copy_index, uint64_t read_count)
{
    (void)pthread_mutex_lock(&script_mutex);
    script_copies_are_held_back = true;
    script_awaited_copy_index = copy_index;
    script_awaited_reads_left = read_count;
    (void)pthread_mutex_unlock(&script_mutex);
}

/* The first read that covers the byte is served by this copy, whichever copy the loader asked. */
static void script_steer_first_read(uint32_t steering_index, uint64_t byte_offset, uint32_t copy_index)
{
    (void)pthread_mutex_lock(&script_mutex);
    script_steering_is_pending[steering_index] = true;
    script_steered_byte_offset[steering_index] = byte_offset;
    script_steered_copy_index[steering_index] = copy_index;
    (void)pthread_mutex_unlock(&script_mutex);
}

MUST_CHECK static result_status
scripted_reader_describe(const struct storage_reader *reader,
			 struct storage_description *description_out)
{
	struct storage_description description = { 0 };
	result_status status =
		test_memory_storage.reader_describe(reader, &description);
	(void)pthread_mutex_lock(&script_mutex);
	description.copy_count = script_described_copy_count !=
						 AS_THE_READER_HAS_IT ?
					 script_described_copy_count :
					 description.copy_count;
	description.block_alignment_bytes =
		script_described_block_alignment_bytes != AS_THE_READER_HAS_IT ?
			script_described_block_alignment_bytes :
			description.block_alignment_bytes;
	status = status == STATUS_OK ? script_description_status : status;
	(void)pthread_mutex_unlock(&script_mutex);
	if (status == STATUS_OK) {
		*description_out = description;
	}
    return status;
}

MUST_CHECK static result_status
scripted_reader_read(const struct storage_reader *reader, uint32_t copy_index,
		     uint64_t byte_offset, void *destination,
		     uint64_t byte_count)
{
    uint32_t serving_copy_index = copy_index;
    (void)pthread_mutex_lock(&script_mutex);
    if (script_copies_are_held_back && copy_index == script_awaited_copy_index) {
        script_awaited_reads_left -= script_awaited_reads_left > 0 ? 1u : 0u;
        if (script_awaited_reads_left == 0) {
            script_copies_are_held_back = false;
            (void)pthread_cond_broadcast(&script_copies_released);
        }
    }
    while (script_copies_are_held_back && copy_index != script_awaited_copy_index) {
        (void)pthread_cond_wait(&script_copies_released, &script_mutex);
    }
    for (uint32_t steering_index = 0; steering_index < STEERED_READ_COUNT; steering_index++) {
        const uint64_t steered_byte_offset = script_steered_byte_offset[steering_index];
        if (script_steering_is_pending[steering_index] && steered_byte_offset >= byte_offset &&
            steered_byte_offset - byte_offset < byte_count) {
            script_steering_is_pending[steering_index] = false;
            serving_copy_index = script_steered_copy_index[steering_index];
        }
    }
    (void)pthread_mutex_unlock(&script_mutex);
    return test_memory_storage.reader_read(reader, serving_copy_index, byte_offset, destination, byte_count);
}

static struct backend_operations
	watching_backend; /* the test backend behind the three calls below */
static const struct device_card *watched_card =
	NULL; /* the slow card of a case, or none */
static _Atomic uint64_t other_uploads_at_the_first_watched_upload = NOT_SEEN;
static _Atomic uint32_t watched_staging_allocation_count = 0;   /* the staging blocks asked for since a case began */
static _Atomic uint32_t watched_wait_count = 0;                 /* the waits for an upload with a limit above zero */

/* Notes, when the watched card begins its first upload, how many uploads the other cards have begun. Of several
 * threads that come here together, the one that notes has counted before any of them began its upload. */
MUST_CHECK static result_status
watching_upload_begin(struct device_card *card, uint64_t arena_byte_offset,
		      const void *page_locked_source, uint64_t byte_count,
		      uint64_t *upload_ticket_out)
{
    if (card == watched_card) {
        uint64_t not_seen = NOT_SEEN;
        uint64_t other_upload_count = 0;
        for (uint32_t card_index = 0; card_index < CARD_COUNT; card_index++) {
            other_upload_count += card_index != SLOW_CARD ? test_host_memory_backend_upload_count(card_index) : 0;
        }
        (void)atomic_compare_exchange_strong(&other_uploads_at_the_first_watched_upload, &not_seen,
                                             other_upload_count);
    }
    return test_host_memory_backend.upload_begin(card, arena_byte_offset, page_locked_source, byte_count,
                                                 upload_ticket_out);
}

/* Counts the staging blocks a load asks for: a request that is refused asks for none. */
MUST_CHECK static result_status
watching_staging_allocate(uint64_t byte_count, void **page_locked_memory_out)
{
    (void)atomic_fetch_add(&watched_staging_allocation_count, 1);
    return test_host_memory_backend.staging_allocate(byte_count, page_locked_memory_out);
}

/* Counts the times the loader does more than ask about an upload: while a load transfers, it only asks. */
MUST_CHECK static result_status
watching_upload_wait(struct device_card *card, uint64_t upload_ticket,
		     uint64_t wait_limit_nanoseconds)
{
    (void)atomic_fetch_add(&watched_wait_count, wait_limit_nanoseconds != 0 ? 1u : 0u);
    return test_host_memory_backend.upload_wait(card, upload_ticket, wait_limit_nanoseconds);
}

static void fixture_build(struct fixture *fixture, const struct test_pack_unit_plan *units, uint32_t unit_count,
                          uint64_t content_seed)
{
    const struct test_pack_plan plan = {
        .model_name = "loader_test", .units = units, .unit_count = unit_count, .content_seed = content_seed};
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
    fixture->extents = pack_extents_of(fixture->pack);
    fixture->units = pack_units_of(fixture->pack);
    fixture->extent_count = pack_header_of(fixture->pack)->extent_count;
    fixture->unit_count = unit_count;
    fixture->content_seed = content_seed;
    for (uint32_t extent_index = 0; extent_index < fixture->extent_count; extent_index++) {
        fixture->extent_byte_total += fixture->extents[extent_index].byte_count;
    }
}

/* Sets both test doubles and both wrappers back, opens the copies and lays the pack out. A test then sets its
 * faults, and calls case_prepare. */
static void case_begin(struct load_case *load_case, const char *name, const struct fixture *fixture,
                       const uint32_t *card_of_unit, uint32_t card_count, uint32_t copy_count)
{
    case_name = name;
    expect(test_host_memory_backend_open_card_count() == 0 && test_host_memory_backend_staging_block_count() == 0,
           "a case begins with no card open and no staging block alive");
    memset(load_case, 0, sizeof(*load_case));
    load_case->fixture = fixture;
    load_case->card_count = card_count;
    test_host_memory_backend_reset(card_count, CARD_MEMORY_BYTES);
    test_memory_storage_reset();
    test_memory_storage_set_pack(fixture->pack_bytes, fixture->pack_byte_count);
    script_reset();
    watched_card = NULL;
    atomic_store(&other_uploads_at_the_first_watched_upload, NOT_SEEN);
    atomic_store(&watched_staging_allocation_count, 0);
    atomic_store(&watched_wait_count, 0);
    if (test_memory_storage.reader_open(copy_paths, copy_count,
					&load_case->reader) != STATUS_OK ||
	load_layout_create(fixture->pack, card_of_unit, fixture->unit_count,
			   card_count, &load_case->layout) != STATUS_OK) {
	    give_up("the copies of a case are opened and its pack is laid out");
    }
}

/* Describes the cards (after a test has set their links), opens those that receive bytes with an arena of exactly
 * what the layout puts on them, and fills the request. */
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
	    .backend = &test_host_memory_backend,
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

MUST_CHECK static result_status
case_run_request(struct load_case *load_case,
		 const struct load_request *request)
{
    memset(&load_case->result, RESULT_FILL_BYTE, sizeof(load_case->result));
    const result_status status = load_run_entry(request, &load_case->result);
    load_case->has_loaded = status == STATUS_OK;
    return status;
}

/* Closes the cards before the host arena is returned, as a program does, and checks that the case leaves nothing.
 * A load that failed gave no arena: no arena is returned in its place, which is accepted and does nothing. */
static void case_end(struct load_case *load_case)
{
	for (uint32_t card_index = 0; card_index < BACKEND_MAXIMUM_CARDS;
	     card_index++) {
		test_host_memory_backend.card_close(
			load_case->cards[card_index],
			CARD_CLOSE_LIMIT_NANOSECONDS);
	}
	load_host_arena_release(
		load_case->has_loaded ? load_case->result.host_arena : NULL,
		load_case->layout->host_byte_count);
	test_memory_storage.reader_close(load_case->reader);
	load_layout_destroy(load_case->layout);
	expect(test_host_memory_backend_open_card_count() == 0 &&
		       test_host_memory_backend_staging_block_count() == 0,
	       "a case ends with no card open and no staging block alive");
}

/* Where an extent lies after a load; NULL for a card that is not open or a host arena that is missing. */
static const unsigned char *delivered_extent(const struct load_case *load_case, uint32_t extent_index)
{
    const uint32_t card_index = load_case->layout->card_of_extent[extent_index];
    const unsigned char *const arena = card_index == HOST
                                           ? load_case->result.host_arena
                                           : test_host_memory_backend_arena_of(load_case->cards[card_index]);
    return arena != NULL ? arena + load_case->layout->byte_offset_of_extent[extent_index] : NULL;
}

/* Every extent against the bytes of the pack and, byte for byte, against what the builder says it wrote. */
static void expect_every_byte_delivered(const struct load_case *load_case, bool byte_for_byte)
{
    const struct fixture *const fixture = load_case->fixture;
    bool is_delivered = true;
    for (uint32_t extent_index = 0; extent_index < fixture->extent_count && is_delivered; extent_index++) {
	    const struct pack_extent *const extent =
		    &fixture->extents[extent_index];
	    const unsigned char *const delivered =
		    delivered_extent(load_case, extent_index);
	    const unsigned char *const in_the_pack =
		    (const unsigned char *)fixture->pack_bytes +
		    extent->byte_offset;
	    if (delivered == NULL) {
		    is_delivered = false;
		    break;
	    }
        is_delivered = memcmp(delivered, in_the_pack, (size_t)extent->byte_count) == 0;
        for (uint64_t byte_index = 0; byte_for_byte && is_delivered && byte_index < extent->byte_count;
             byte_index++) {
            is_delivered =
                delivered[byte_index] == test_pack_content_byte(fixture->content_seed, extent_index, byte_index);
        }
    }
    expect(is_delivered, "every extent lies where the layout puts it, with the bytes of the pack");
}

/* No sanitizer guards the end of a mapping as it guards the end of an allocation, and a host arena is mapped as
 * whole 2 MiB pages: what lies behind the bytes the layout puts in host memory is as the system gave it, zero,
 * unless a piece was written beyond its place. */
static void expect_nothing_written_behind_host_memory(const struct load_case *load_case)
{
    const unsigned char *const arena = load_case->result.host_arena;
    const uint64_t host_byte_count = load_case->layout->host_byte_count;
    const uint64_t mapped_byte_count = (host_byte_count + HUGE_PAGE_BYTES - 1) / HUGE_PAGE_BYTES * HUGE_PAGE_BYTES;
    uint64_t byte_index = host_byte_count;
    while (arena != NULL && byte_index < mapped_byte_count && arena[byte_index] == 0) {
        byte_index++;
    }
    expect(arena == NULL || byte_index == mapped_byte_count,
           "the host arena is untouched behind the last extent of host memory");
}

/* What the result of a load says was read, from all copies together. */
static uint64_t byte_count_read_of_all_copies(const struct load_case *load_case)
{
    uint64_t byte_count_read = 0;
    for (uint32_t copy_index = 0; copy_index < STORAGE_MAXIMUM_COPIES;
	 copy_index++) {
	    byte_count_read +=
		    load_case->result.byte_count_read_of_copy[copy_index];
    }
    return byte_count_read;
}

/* A load that must succeed: every byte delivered and none behind the last, the per-copy byte counts adding up to
 * what was read, an arrival time for every destination that received bytes and for no other, no staging block left:
 * the host arena is none. extents_read_again says how many extents the load had to send again. */
static void expect_complete_load(struct load_case *load_case, uint32_t extents_read_again)
{
	const struct load_result *const result = &load_case->result;
	const bool has_host_bytes = load_case->layout->host_byte_count != 0;
	bool arrivals_are_where_bytes_went = true;
	expect(case_run_request(load_case, &load_case->request) == STATUS_OK,
	       "the load succeeds");
	if (!load_case->has_loaded) {
		return;
	}
    const uint64_t byte_count_read = byte_count_read_of_all_copies(load_case);
    uint64_t last_arrival_nanoseconds = result->host_arrival_nanoseconds;
    for (uint32_t card_index = 0; card_index < BACKEND_MAXIMUM_CARDS;
	 card_index++) {
	    const uint64_t arrival_nanoseconds =
		    result->card_arrival_nanoseconds[card_index];
	    const bool receives_bytes =
		    card_index < load_case->card_count &&
		    load_case->layout->card_byte_count[card_index] != 0;
	    arrivals_are_where_bytes_went = arrivals_are_where_bytes_went &&
					    (arrival_nanoseconds != 0) ==
						    receives_bytes;
	    last_arrival_nanoseconds =
		    arrival_nanoseconds > last_arrival_nanoseconds ?
			    arrival_nanoseconds :
			    last_arrival_nanoseconds;
    }
    expect(arrivals_are_where_bytes_went && (result->host_arrival_nanoseconds != 0) == has_host_bytes,
           "a destination has an arrival time when, and only when, bytes went to it");
    expect(last_arrival_nanoseconds == result->transfer_nanoseconds,
           "the transfer ends when the last destination has its last piece: no arrival is later");
    expect_every_byte_delivered(load_case, false);
    expect((load_case->result.host_arena != NULL) == has_host_bytes,
           "there is a host arena when, and only when, host memory holds bytes");
    expect_nothing_written_behind_host_memory(load_case);
    expect(test_host_memory_backend_staging_block_count() == 0,
           "the loader has returned every staging slot, and its host arena is no staging block");
    expect(load_case->result.extents_read_again == extents_read_again, "the count of extents read again");
    expect(extents_read_again != 0 || byte_count_read == load_case->fixture->extent_byte_total,
           "with nothing read again, the per-copy byte counts add up to the extents of the pack");
    expect(byte_count_read >= load_case->fixture->extent_byte_total, "every extent was read at least once");
}

/* A load that must fail with that status, leave the result untouched and return everything it obtained. */
static void expect_failed_load(struct load_case *load_case,
			       result_status expected_status, const char *what)
{
    expect(case_run_request(load_case, &load_case->request) == expected_status, what);
    expect(result_is_untouched(load_case), "a failed load leaves the result untouched");
    expect(test_host_memory_backend_staging_block_count() == 0,
           "a failed load returns every staging block it obtained");
}

/* A request the check must refuse, before anything is obtained. The request a test broke, and the copy of the
 * layout it broke (NULL when the request names the layout of the case), are then whole again. The host arena is no
 * block the backend counts: the fault of its mapping is armed for a call that never comes, so that it only counts
 * how often the loader came to map one. */
static void expect_refused(struct load_case *load_case,
			   struct load_request *request,
			   struct load_layout *layout_copy, const char *what)
{
#if CHECKING_HOOKS_ENABLED
	expect(fault_arm("loader_host_arena", UINT64_MAX),
	       "the fault of the host arena is armed");
#endif
	expect(case_run_request(load_case, request) == STATUS_INVALID_ARGUMENT,
	       what);
#if CHECKING_HOOKS_ENABLED
	expect(fault_disarm() == 0,
	       "a refused request does not come to map a host arena, not even for a while");
#endif
    expect(result_is_untouched(load_case) && atomic_load(&watched_staging_allocation_count) == 0 &&
               test_host_memory_backend_staging_block_count() == 0,
           "a refused request leaves the result untouched and obtains nothing, not even for a while");
    if (request != NULL) {
        *request = load_case->request;
    }
    if (request != NULL && layout_copy != NULL) {
        *layout_copy = *load_case->layout;
        request->layout = layout_copy;
    }
}

/* A whole case: one load that must succeed. */
static void expect_load_delivers(const char *name, const struct fixture *fixture, const uint32_t *card_of_unit,
                                 uint32_t card_count, uint32_t copy_count, uint32_t reader_threads_per_copy)
{
    struct load_case load_case;
    case_begin(&load_case, name, fixture, card_of_unit, card_count, copy_count);
    case_prepare(&load_case, reader_threads_per_copy);
    expect_complete_load(&load_case, 0);
    case_end(&load_case);
}

static struct load_layout
	untouched_layout; /* a layout_out that points here was not written */

static bool layout_is_refused(const struct pack_record *pack,
			      const uint32_t *card_of_unit, uint32_t unit_count,
			      uint32_t card_count)
{
	struct load_layout *layout = &untouched_layout;
	return load_layout_create(pack, card_of_unit, unit_count, card_count,
				  &layout) == STATUS_INVALID_ARGUMENT &&
	       layout == &untouched_layout;
}

static void test_layout(const struct fixture *large)
{
	struct load_layout *layout = &untouched_layout;
	uint32_t placement[LARGE_UNIT_COUNT];
	uint64_t byte_count_of_card[BACKEND_MAXIMUM_CARDS] = { 0 };
	uint64_t host_byte_count = 0;
	bool is_as_expected = true;
	case_name = "layout";

	expect(layout_is_refused(NULL, large_placement, LARGE_UNIT_COUNT,
				 CARD_COUNT),
	       "no pack is refused");
	expect(layout_is_refused(large->pack, NULL, LARGE_UNIT_COUNT,
				 CARD_COUNT),
	       "no placement is refused");
	expect(load_layout_create(large->pack, large_placement,
				  LARGE_UNIT_COUNT, CARD_COUNT,
				  NULL) == STATUS_INVALID_ARGUMENT,
	       "no output is refused");
	expect(layout_is_refused(large->pack, large_placement, LARGE_UNIT_COUNT,
				 BACKEND_MAXIMUM_CARDS + 1),
	       "a card count above the maximum is refused");
	expect(layout_is_refused(large->pack, large_placement,
				 LARGE_UNIT_COUNT - 1, CARD_COUNT),
	       "a placement of another unit count is refused");
	expect(layout_is_refused(large->pack, large_placement, LARGE_UNIT_COUNT,
				 CARD_COUNT - 1),
	       "a stage unit on a card at the card count is refused");
	memcpy(placement, large_placement, sizeof(placement));
	placement[0] = HOST;
	expect(layout_is_refused(large->pack, placement, LARGE_UNIT_COUNT,
				 CARD_COUNT),
	       "a stage unit in host memory is refused");
	memcpy(placement, large_placement, sizeof(placement));
	placement[1] = 1;
	expect(layout_is_refused(large->pack, placement, LARGE_UNIT_COUNT,
				 CARD_COUNT),
	       "an on-demand unit on another card than its owner's is refused");
#if CHECKING_HOOKS_ENABLED
	expect(fault_arm("loader_layout_allocation", 1),
	       "the fault of the layout's memory is armed");
	expect(load_layout_create(large->pack, large_placement,
				  LARGE_UNIT_COUNT, CARD_COUNT,
				  &layout) == STATUS_OUT_OF_MEMORY &&
		       layout == &untouched_layout,
	       "a lack of memory is reported and leaves the output untouched");
	expect(fault_disarm() == 1,
	       "the fault point of the layout's memory was reached");
#endif
	expect(load_layout_create(large->pack, large_placement,
				  LARGE_UNIT_COUNT, CARD_COUNT,
				  &layout) == STATUS_OK,
	       "a valid placement is laid out");
	if (layout == &untouched_layout) {
		return;
	}
    is_as_expected = layout->extent_count == large->extent_count && layout->card_count == CARD_COUNT;
    for (uint32_t extent_index = 0; extent_index < large->extent_count && is_as_expected; extent_index++) {
        const uint32_t card_index = large_placement[large->extents[extent_index].unit_index];
        uint64_t *const byte_count = card_index == HOST ? &host_byte_count : &byte_count_of_card[card_index];
        is_as_expected = layout->card_of_extent[extent_index] == card_index &&
                         layout->byte_offset_of_extent[extent_index] == *byte_count;
        *byte_count += large->extents[extent_index].byte_count;
    }
    expect(is_as_expected, "every extent follows the one before it of its destination, from offset 0");
    expect(memcmp(layout->card_byte_count, byte_count_of_card, sizeof(byte_count_of_card)) == 0 &&
               layout->host_byte_count == host_byte_count, "the byte counts are the sums of the extents");
    expect(layout->card_byte_count[0] > TWO_CHUNKS &&
		   layout->host_byte_count > LOADER_CHUNK_BYTES,
	   "the large pack puts more than two chunks on a card and more than one in host memory");
    load_layout_destroy(layout);
    load_layout_destroy(
	    NULL); /* accepted: there is nothing to free, and nothing happens */
    load_host_arena_release(
	    NULL,
	    host_byte_count); /* accepted: no arena, whatever size is named */
}

/* Three cards, host memory and three copies: every byte, the pieces as the contract cuts them, the result. */
static void test_complete_load(const struct fixture *large)
{
    struct load_case load_case;
    uint64_t read_count = 0;
    bool counts_agree = true;
    case_begin(&load_case, "complete load", large, large_placement, CARD_COUNT, COPY_COUNT);
    case_prepare(&load_case, THREADS_PER_COPY);
    expect_complete_load(&load_case, 0);
    if (load_case.has_loaded) {
	    const struct load_result *const result = &load_case.result;
	    expect_every_byte_delivered(&load_case, true);
	    for (uint32_t copy_index = 0; copy_index < STORAGE_MAXIMUM_COPIES;
		 copy_index++) {
		    read_count += test_memory_storage_read_count(copy_index);
		    counts_agree =
			    counts_agree &&
			    result->byte_count_read_of_copy[copy_index] ==
				    test_memory_storage_delivered_byte_count(
					    copy_index);
	    }
	expect(counts_agree, "each copy's byte count is what the storage delivered from it");
        expect(read_count == LARGE_PIECE_COUNT, "the pack is read in the pieces the contract cuts");
        for (uint32_t card_index = 0; card_index < CARD_COUNT; card_index++) {
            expect(test_host_memory_backend_upload_count(card_index) == large_upload_counts[card_index],
                   "a card receives one upload for each of its pieces");
        }
        expect(result->copies_dropped == 0, "no copy is dropped");
        expect(result->transfer_nanoseconds > 0 && result->check_nanoseconds > 0,
               "the transfer and the check each took their time");
    }
    case_end(&load_case);
}

static void test_request_refusals(const struct fixture *small, const struct fixture *striped)
{
    struct load_case load_case;
    struct load_request request;
    struct load_layout layout;
    struct load_layout *layout_of_another_pack = NULL;
    struct device_card *cards_with_one_missing[BACKEND_MAXIMUM_CARDS] = { NULL };
    case_begin(&load_case, "request refusals", small, small_placement, CARD_COUNT, COPY_COUNT);
    case_prepare(&load_case, THREADS_PER_COPY);
    load_case.request.backend = &watching_backend;      /* it counts what a refused request must not ask for */
    cards_with_one_missing[0] = load_case.cards[0];
    request = load_case.request;

    expect_refused(&load_case, NULL, NULL, "no request is refused");
    expect(load_run_entry(&request, NULL) == STATUS_INVALID_ARGUMENT,
	   "no result is refused");
    request.storage = NULL;
    expect_refused(&load_case, &request, NULL, "no storage is refused");
    request.reader = NULL;
    expect_refused(&load_case, &request, NULL, "no reader is refused");
    request.backend = NULL;
    expect_refused(&load_case, &request, NULL, "no backend is refused");
    request.pack = NULL;
    expect_refused(&load_case, &request, NULL, "no pack is refused");
    request.layout = NULL;
    expect_refused(&load_case, &request, NULL, "no layout is refused");
    request.cards = NULL;
    expect_refused(&load_case, &request, NULL, "no cards where the layout places bytes are refused");
    request.card_descriptions = NULL;
    expect_refused(&load_case, &request, NULL, "no descriptions where the layout places bytes are refused");
    request.cards = cards_with_one_missing;
    expect_refused(&load_case, &request, NULL, "a missing card where the layout places bytes is refused");
    request.reader_threads_per_copy = 0;
    expect_refused(&load_case, &request, NULL, "no reader thread is refused");
    request.reader_threads_per_copy =
	    LOADER_MAXIMUM_READER_THREADS_PER_COPY + 1;
    expect_refused(&load_case, &request, NULL, "a thread count above the maximum is refused");
    request.reserved_zero = 1;
    expect_refused(&load_case, &request, NULL, "a reserved field that is not zero is refused");
    request.stall_limit_nanoseconds = 0;
    expect_refused(&load_case, &request, NULL, "a stall limit of zero is refused");

    /* a layout that is not this pack's: by its counts, by its tables, by byte counts that are not what its extents
     * occupy (the last card's is looked at behind the others': nothing was obtained for those either) */
    layout = *load_case.layout;
    request.layout = &layout;
    layout.extent_count -= 1;
    expect_refused(&load_case, &request, &layout, "a layout of another extent count is refused");
    layout.card_count = BACKEND_MAXIMUM_CARDS + 1;
    expect_refused(&load_case, &request, &layout, "a layout of a card count above the maximum is refused");
    layout.card_count = 1;
    expect_refused(&load_case, &request, &layout, "a layout of fewer cards than its extents name is refused");
    layout.byte_offset_of_extent = NULL;
    expect_refused(&load_case, &request, &layout, "a layout without offsets is refused");
    layout.card_of_extent = NULL;
    expect_refused(&load_case, &request, &layout, "a layout without destinations is refused");
    layout.host_byte_count -= PACK_BLOCK_BYTES;
    expect_refused(&load_case, &request, &layout, "a host byte count below what the extents need is refused");
    layout.card_byte_count[0] -= PACK_BLOCK_BYTES;
    expect_refused(&load_case, &request, &layout, "a card byte count below what the extents need is refused");
    layout.host_byte_count += PACK_BLOCK_BYTES;
    expect_refused(&load_case, &request, &layout, "a host byte count above what the extents occupy is refused");
    layout.card_byte_count[CARD_COUNT - 1] += PACK_BLOCK_BYTES;
    expect_refused(&load_case, &request, &layout, "a card byte count above what the extents occupy is refused");
    load_case.layout->byte_offset_of_extent[1] += PACK_BLOCK_BYTES;
    expect_refused(&load_case, &request, &layout, "an extent that does not follow the one before it is refused");
    load_case.layout->byte_offset_of_extent[1] -= PACK_BLOCK_BYTES;
    load_case.layout->card_of_extent[0] = HOST;
    expect_refused(&load_case, &request, &layout, "an extent whose offset is another destination's is refused");
    load_case.layout->card_of_extent[0] = small_placement[0];
    if (load_layout_create(striped->pack, striped_placement,
			   striped->unit_count, CARD_COUNT,
			   &layout_of_another_pack) != STATUS_OK) {
	    give_up("another pack is laid out");
    }
    request.layout = layout_of_another_pack;
    expect_refused(&load_case, &request, NULL, "the layout of another pack is refused");
    load_layout_destroy(layout_of_another_pack);

    /* a reader that does not fit: what it describes comes through the scripted storage */
    request.storage = &scripted_storage;
    script_described_copy_count = STORAGE_MAXIMUM_COPIES + 1;
    expect_refused(&load_case, &request, NULL, "a reader of more copies than the maximum is refused");
    request.storage = &scripted_storage;
    script_reset();
    script_described_copy_count = 0;
    expect_refused(&load_case, &request, NULL, "a reader without a copy is refused");
    request.storage = &scripted_storage;
    script_reset();
    script_described_block_alignment_bytes = ALIGNMENT_THAT_DIVIDES_NO_BLOCK;
    expect_refused(&load_case, &request, NULL, "a reader whose alignment does not divide a block is refused");
    request.storage = &scripted_storage;
    script_reset();
    script_described_block_alignment_bytes = 0;
    expect_refused(&load_case, &request, NULL, "a reader without a block alignment is refused");
    request.storage = &scripted_storage;
    script_reset();
    script_description_status = STATUS_INPUT_OUTPUT_ERROR;
    expect(case_run_request(&load_case, &request) ==
			   STATUS_INPUT_OUTPUT_ERROR &&
		   result_is_untouched(&load_case) &&
		   atomic_load(&watched_staging_allocation_count) == 0,
	   "a reader that cannot be described fails the load with its status, before anything is obtained");
    script_reset();

    expect_complete_load(&load_case, 0);        /* the request that was broken in every way above is a good one */
    case_end(&load_case);
}

/* Uploads that stay in flight, and one card on a slow link. With one reader thread the order is the contract's:
 * the slow card is served first, and its slots are all in use before a fast card receives a piece. */
static void test_uploads_in_flight(const char *name, const struct fixture *striped, uint32_t copy_count,
                                   uint32_t reader_threads_per_copy)
{
    struct load_case load_case;
    uint64_t other_upload_count = 0;
    case_begin(&load_case, name, striped, striped_placement, CARD_COUNT, copy_count);
    for (uint32_t card_index = 0; card_index < CARD_COUNT; card_index++) {
        const bool is_slow = card_index == SLOW_CARD;
        test_host_memory_backend_set_polls_before_an_upload_finishes(card_index, POLLS_BEFORE_AN_UPLOAD_FINISHES);
        test_host_memory_backend_set_link(card_index, is_slow ? SLOW_LINK_GENERATION : FAST_LINK_GENERATION,
                                          is_slow ? SLOW_LINK_LANE_COUNT : FAST_LINK_LANE_COUNT);
    }
    case_prepare(&load_case, reader_threads_per_copy);
    load_case.request.backend = &watching_backend;
    watched_card = load_case.cards[SLOW_CARD];
    expect_complete_load(&load_case, 0);
    for (uint32_t card_index = 0; card_index < CARD_COUNT; card_index++) {
        other_upload_count += card_index != SLOW_CARD ? test_host_memory_backend_upload_count(card_index) : 0;
        expect(test_host_memory_backend_most_uploads_in_flight(card_index) <= SLOTS_PER_CARD,
               "a card never has more uploads in flight than it has slots");
    }
    expect(atomic_load(&other_uploads_at_the_first_watched_upload) < other_upload_count,
           "the slow card began its first upload before the fast cards were done");
    expect(atomic_load(&watched_wait_count) == 0,
           "no thread waited for an upload: one with nothing to take looks again later");
    expect(atomic_load(&watched_staging_allocation_count) == CARD_COUNT * SLOTS_PER_CARD,
           "the backend is asked for the staging slots and for no other block: the host arena is none of its memory");
    if (copy_count * reader_threads_per_copy == 1) {
        expect(atomic_load(&other_uploads_at_the_first_watched_upload) == 0,
               "with one thread, the first piece of the load goes to the card that would finish last");
        expect(test_host_memory_backend_most_uploads_in_flight(SLOW_CARD) == SLOTS_PER_CARD,
               "with one thread, the slow card's slots are all in use before a fast card is served");
    }
    case_end(&load_case);
}

static void test_dropped_copies(const struct fixture *striped)
{
    struct load_case load_case;

    case_begin(&load_case, "a copy that fails from a later read on", striped, striped_placement, CARD_COUNT,
               COPY_COUNT);
    case_prepare(&load_case, 1);
    load_case.request.storage = &scripted_storage;
    test_memory_storage_fail_reads(1, READS_BEFORE_THE_COPY_FAILS + 1);
    script_hold_copies_back_until(1, READS_BEFORE_THE_COPY_FAILS + 1);
    expect_complete_load(&load_case, 0);
    if (load_case.has_loaded) {
        expect(load_case.result.copies_dropped == 1, "the copy that failed is dropped, and only it");
        expect(test_memory_storage_read_count(1) == READS_BEFORE_THE_COPY_FAILS + 1,
               "a dropped copy is not read again");
        expect(load_case.result.byte_count_read_of_copy[1] == test_memory_storage_delivered_byte_count(1),
               "the bytes of a dropped copy are those it delivered before it failed");
    }
    case_end(&load_case);

    case_begin(&load_case, "a copy that fails from its first read on", striped, striped_placement, CARD_COUNT,
               COPY_COUNT);
    case_prepare(&load_case, THREADS_PER_COPY);
    load_case.request.storage = &scripted_storage;
    test_memory_storage_fail_reads(0, 1);
    script_hold_copies_back_until(0, 1);
    expect_complete_load(&load_case, 0);
    if (load_case.has_loaded) {
        expect(load_case.result.copies_dropped == 1 && load_case.result.byte_count_read_of_copy[0] == 0,
               "the copy that failed is dropped with nothing read from it");
    }
    case_end(&load_case);

    case_begin(&load_case, "every copy fails", striped, striped_placement, CARD_COUNT, COPY_COUNT);
    case_prepare(&load_case, THREADS_PER_COPY);
    for (uint32_t copy_index = 0; copy_index < COPY_COUNT; copy_index++) {
        test_memory_storage_fail_reads(copy_index, 1);
    }
    expect_failed_load(&load_case, STATUS_INPUT_OUTPUT_ERROR,
		       "with no copy left the load fails as the drives did");
    case_end(&load_case);
}

/* The offset in the file of a byte inside the first extent of a unit. */
static uint64_t file_offset_in_unit(const struct fixture *fixture, uint32_t unit_index, uint64_t byte_index)
{
    return fixture->extents[fixture->units[unit_index].first_extent_index].byte_offset + byte_index;
}

static void test_flipped_bits(const struct fixture *large, const struct fixture *small)
{
    static const uint32_t flipped_large_units[STEERED_READ_COUNT] = {LARGE_UNIT_OF_SEVERAL_CHUNKS_ON_A_CARD,
                                                                     LARGE_UNIT_OF_SEVERAL_CHUNKS_IN_HOST_MEMORY};
    static const uint32_t flipped_small_units[] = {SMALL_UNIT_ON_A_CARD, SMALL_UNIT_IN_HOST_MEMORY};
    const uint32_t small_card_extent_index = small->units[SMALL_UNIT_ON_A_CARD].first_extent_index;
    struct load_case load_case;

    /* One copy has a bit of a card's extent flipped and another a bit of a host extent, each behind the first
     * chunk; the first read of either byte is steered to the copy that has it flipped, so both extents arrive
     * wrong. Each is then read again from copy 0 first: the one copy 0 holds flipped a second time, from copy 1,
     * and neither from a copy behind the one that gave the digest. Then the two copies change their parts. */
    for (uint32_t arrangement = 0; arrangement < STEERED_READ_COUNT; arrangement++) {
        uint64_t byte_count_read_again = 0;
        case_begin(&load_case, "a bit flipped on one copy", large, large_placement, CARD_COUNT, COPY_COUNT);
        case_prepare(&load_case, THREADS_PER_COPY);
        load_case.request.storage = &scripted_storage;
        for (uint32_t copy_index = 0; copy_index < STEERED_READ_COUNT; copy_index++) {
            const uint32_t unit_index = flipped_large_units[(copy_index + arrangement) % STEERED_READ_COUNT];
	    const uint64_t byte_offset = file_offset_in_unit(
		    large, unit_index, LOADER_CHUNK_BYTES + FLIPPED_BYTE_INDEX);
	    test_memory_storage_flip_bit(copy_index, byte_offset);
            script_steer_first_read(copy_index, byte_offset, copy_index);
            byte_count_read_again += (STEERED_READ_COUNT - copy_index) * large->units[unit_index].byte_count;
        }
        expect(large->units[flipped_large_units[0]].extent_count == 1 &&
                   large->units[flipped_large_units[1]].extent_count == 1, "each flipped unit is one extent");
        expect_complete_load(&load_case, STEERED_READ_COUNT);
        expect(load_case.has_loaded &&
                   byte_count_read_of_all_copies(&load_case) == large->extent_byte_total + byte_count_read_again,
               "a differing extent is read again from each copy in turn, up to the first that holds it whole");
        case_end(&load_case);
    }

    for (size_t unit = 0; unit < sizeof(flipped_small_units) / sizeof(flipped_small_units[0]); unit++) {
        case_begin(&load_case, "a bit flipped on every copy", small, small_placement, CARD_COUNT, COPY_COUNT);
        case_prepare(&load_case, THREADS_PER_COPY);
        for (uint32_t copy_index = 0; copy_index < COPY_COUNT; copy_index++) {
            test_memory_storage_flip_bit(copy_index,
                                         file_offset_in_unit(small, flipped_small_units[unit], FLIPPED_BYTE_INDEX));
        }
	expect_failed_load(&load_case, STATUS_CORRUPT_PACK,
			   "an extent that no copy holds whole fails the load");
	case_end(&load_case);
    }

    case_begin(&load_case, "a bit flipped by the link", small, small_placement, CARD_COUNT, COPY_COUNT);
    case_prepare(&load_case, THREADS_PER_COPY);
    test_host_memory_backend_flip_bit_in_next_upload(
        small_placement[SMALL_UNIT_ON_A_CARD],
        load_case.layout->byte_offset_of_extent[small_card_extent_index] + FLIPPED_BYTE_INDEX);
    expect_complete_load(&load_case, 1);
    expect(load_case.has_loaded && byte_count_read_of_all_copies(&load_case) ==
                                       small->extent_byte_total + small->extents[small_card_extent_index].byte_count,
           "the extent a link damaged is read again once, and no extent beside it");
    case_end(&load_case);
}

static void test_thread_counts_and_placements(const struct fixture *large, const struct fixture *small,
                                              const struct fixture *striped)
{
    struct load_case load_case;
    expect_load_delivers("one thread per copy", striped, striped_placement, CARD_COUNT, COPY_COUNT, 1);
    expect_load_delivers("the most threads and copies", striped,
			 striped_placement, CARD_COUNT, STORAGE_MAXIMUM_COPIES,
			 LOADER_MAXIMUM_READER_THREADS_PER_COPY);
    expect_load_delivers("one copy", large, large_placement, CARD_COUNT, 1, THREADS_PER_COPY);
    expect_load_delivers("one copy and one thread", small, small_placement, CARD_COUNT, 1, 1);
    expect_load_delivers("nothing in host memory", small, small_placement_on_cards_only, CARD_COUNT, COPY_COUNT,
                         THREADS_PER_COPY);
    expect_load_delivers("everything movable in host memory", small, small_placement_in_host_memory, CARD_COUNT,
                         COPY_COUNT, THREADS_PER_COPY);
    expect_load_delivers("a card that receives nothing", small, small_placement, CARD_COUNT + 1, COPY_COUNT,
                         THREADS_PER_COPY);

    /* a generation the table does not hold, a link without lanes, and the link of 0 by 0 a card has by default */
    case_begin(&load_case, "links that are not known", small, small_placement, CARD_COUNT, COPY_COUNT);
    test_host_memory_backend_set_link(0, UNKNOWN_LINK_GENERATION, FAST_LINK_LANE_COUNT);
    test_host_memory_backend_set_link(1, FAST_LINK_GENERATION, 0);
    case_prepare(&load_case, THREADS_PER_COPY);
    expect_complete_load(&load_case, 0);
    case_end(&load_case);
}

int main(void)
{
    struct fixture large;
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
    scripted_storage = test_memory_storage;
    scripted_storage.reader_describe = scripted_reader_describe;
    scripted_storage.reader_read = scripted_reader_read;
    watching_backend = test_host_memory_backend;
    watching_backend.upload_begin = watching_upload_begin;
    watching_backend.upload_wait = watching_upload_wait;
    watching_backend.staging_allocate = watching_staging_allocate;
    fixture_build(&large, large_units, LARGE_UNIT_COUNT, LARGE_SEED);
    fixture_build(&small, small_units, SMALL_UNIT_COUNT, SMALL_SEED);
    fixture_build(&striped, striped_units, STRIPED_UNIT_COUNT, STRIPED_SEED);

    test_layout(&large);
    test_complete_load(&large);
    test_request_refusals(&small, &striped);
    test_uploads_in_flight("uploads in flight, one reader thread", &striped, 1, 1);
    test_uploads_in_flight("uploads in flight, six reader threads", &striped, COPY_COUNT, THREADS_PER_COPY);
    test_dropped_copies(&striped);
    test_flipped_bits(&large, &small);
    test_thread_counts_and_placements(&large, &small, &striped);

    pack_close(striped.pack);
    pack_close(small.pack);
    pack_close(large.pack);
    free(striped.pack_bytes);
    free(small.pack_bytes);
    free(large.pack_bytes);
    if (failures != 0) {
        return EXIT_FAILURE;
    }
    (void)printf("loader_test passed (checking hooks %s)\n",
		 CHECKING_HOOKS_ENABLED ? "enabled" : "compiled out");
    return EXIT_SUCCESS;
}
