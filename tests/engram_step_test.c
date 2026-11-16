/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* engram_step_test: step I/O (engram_step.h, storage_flight.h) over three real copies of the test pack, one of them
 * corrupt: every row equals the pack's pattern after deduplication, the cache, a one-slot refill chain and the copy
 * fallback; two cards waiting on one step at once; the refusals; and (checking hooks) a refused ring, a thread that
 * does not start, a digest that fails, a refused buffer registration (plain reads), a refused submit (retried), a read
 * the kernel never answers (its own deadline retires the ring and quarantines the slot, also while the thread is
 * busy, the waiter reads its block from another copy, a later step skips the copy, a block bad on the last live copy
 * goes to the waiter, every copy retired leaves the blocks to the waiters), recovery (a verified read or the
 * backoff, then the probe, one a copy; damping), a late completion that gives its slot back, the ring cap and its
 * refused ask, a ring retired with reads in its inbox, and a destroy that keeps a step before the close. */
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "anemone/engram_disk.h"
#include "anemone/hooks.h"
#include "anemone/pack.h"
#include "engram_pack.h"
#include "engram_step.h"

#define COPIES 3u
#define CORRUPT_COPY 1u
#define TEST_SECONDS 120u
#define DIRECTORY_BYTES 256u
#define PATH_BYTES 512u
#define STEPS 2u
#define ROWS_MOST 8u
#define ROUNDS 200u
#define CACHE_BLOCKS 16u
#define SPIN_NANOSECONDS 20000u
#define RANDOM_MULTIPLIER 6364136223846793005ull
#define RANDOM_INCREMENT 1442695040888963407ull
#define RANDOM_SHIFT 33u
#define CORRUPT_BYTE 100u
#define FAULT_CALLS 3u
#define SOME_THREADS_PER_COPY 2u
#define SOME_SLOTS 4u
#define SHORT_READ_LIMIT_NANOSECONDS \
	50000000ull /* a read's deadline in the retirement cases: 50 ms */
#define CONCURRENT_ROUNDS 100u
#define CONCURRENT_WAITERS 2u /* two cards wait on one step's tables */
#define NANOSECONDS_PER_SECOND 1000000000ull
#define SEED_RETIREMENT 1u
#define SEED_FLIGHT 2u
#define SEED_SINGLE_COPY 3u
#define SEED_BUSY 4u
#define SEED_LIVELOCK 5u
#define SEED_BACKOFF 6u
#define SEED_CAP 7u
#define SEED_PROBE 8u
#define SEED_DAMPING 9u
#define DAMPING_BACKOFF_NANOSECONDS \
	50000000ull /* 50 ms: the stable period, 60 of them, outlasts the case */
#define DAMPED_RETIREMENTS 3u
#define HOLD_SETTLE_NANOSECONDS \
	10000000l /* an I/O thread takes a submitted read well within 10 ms */
#define FLIGHT_SLOTS (STEPS * SOME_SLOTS)
#define LONG_BACKOFF_NANOSECONDS \
	60000000000ull /* a retired copy waits longer than the test runs */
#define SHORT_BACKOFF_NANOSECONDS 20000000ull /* 20 ms */
#define KEPT_READ_LIMIT_NANOSECONDS \
	500000000ull /* the kept step's held read: still in flight when looked at */
#define CLOSE_DRAIN_LIMITS \
	2u /* the flight's close waits this many read limits */
#define CLOSE_SLACK_NANOSECONDS 300000000ull
#define RECOVERY_ROUNDS_MOST 4000u /* rounds a copy may take to come back */
#define BUSY_ROUNDS_MOST 100000u
#define BUSY_ROUNDS_LIMIT_NANOSECONDS \
	2000000000ull /* the busy step stops by then, retired or not */
#define BUSY_RETIRE_LIMITS \
	4u /* the held read's ring retires within this many deadlines */
#define LATE_CYCLES                          \
	(STORAGE_FLIGHT_RETIRED_RINGS_MOST + \
	 1u) /* more retirements than a thread keeps rings */
#define RECLAIM_LIMIT_NANOSECONDS 5000000000ull
#define ASK_COUNT_CALL \
	1000000000ull /* armed beyond reach: the point only counts the asks */
#define COUNTED_ROUNDS 30u

static unsigned failures;

static void expect(bool passed, const char *message)
{
	if (!passed) {
		failures++;
		(void)fprintf(stderr, "FAILED: %s\n", message);
	}
}

/* The pack's header at the start of its bytes (the pack writer aligns it). */
static const void *bytes_header(const unsigned char *bytes)
{
	return bytes;
}

static uint64_t random_next(uint64_t *state)
{
	*state = *state * RANDOM_MULTIPLIER + RANDOM_INCREMENT;
	return *state >> RANDOM_SHIFT;
}

/* Byte b of table t's extent is b times the pattern multiplier plus t (engram_pack.h). */
static bool row_is_right(uint32_t table, uint64_t row,
			 const unsigned char *bytes)
{
	for (uint64_t byte = 0; byte < ENGRAM_ROW_BYTES; byte++) {
		const uint64_t offset =
			row / ENGRAM_ROWS_PER_BLOCK * PACK_BLOCK_BYTES +
			row % ENGRAM_ROWS_PER_BLOCK * ENGRAM_ROW_BYTES + byte;
		if (bytes[byte] !=
		    (unsigned char)(offset *
					    TEST_ENGRAM_PACK_PATTERN_MULTIPLIER +
				    table))
			return false;
	}
	return true;
}

struct fixture {
	char directory[DIRECTORY_BYTES], paths[COPIES][PATH_BYTES];
	const char *names[COPIES];
	struct storage_reader *reader;
	struct pack_record *pack;
	struct engram_record *engram;
	struct engram_disk *disk;
};

static bool fixture_open(struct fixture *fixture)
{
	unsigned char *bytes = NULL;
	struct engram_record *memory_engram = NULL;
	if (test_engram_pack_open(&bytes, &memory_engram) != STATUS_OK)
		return false;
	const uint64_t pack_bytes =
		((const struct pack_header *)bytes_header(bytes))
			->file_byte_count;
	(void)snprintf(fixture->directory, sizeof(fixture->directory),
		       "build/engram_step_test.%ld", (long)getpid());
	bool written = mkdir(fixture->directory, S_IRWXU) == 0;
	for (uint32_t copy = 0; written && copy < COPIES; copy++) {
		(void)snprintf(fixture->paths[copy],
			       sizeof(fixture->paths[copy]),
			       "%s/copy%u.anemone", fixture->directory, copy);
		fixture->names[copy] = fixture->paths[copy];
		const int descriptor =
			open(fixture->paths[copy],
			     O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC,
			     S_IRUSR | S_IWUSR);
		written = descriptor >= 0 &&
			  write(descriptor, bytes, (size_t)pack_bytes) ==
				  (ssize_t)pack_bytes &&
			  fsync(descriptor) == 0;
		if (descriptor >= 0)
			(void)close(descriptor);
	}
	engram_close(memory_engram);
	free(bytes);
	if (!written)
		return false;
	result_status status = storage_direct.reader_open(
		fixture->names, COPIES, &fixture->reader);
	status = status == STATUS_OK ?
			 pack_open_entry(&storage_direct, fixture->reader,
					 &fixture->pack) :
			 status;
	status = status == STATUS_OK ?
			 engram_open(&storage_direct, fixture->reader,
				     fixture->pack, &fixture->engram) :
			 status;
	status = status == STATUS_OK ?
			 engram_disk_open(
				 engram_description_of(fixture->engram),
				 fixture->pack, &storage_direct,
				 fixture->reader, 0,
				 (uint64_t)CACHE_BLOCKS * PACK_BLOCK_BYTES, 1,
				 1, &fixture->disk) :
			 status;
	if (status != STATUS_OK)
		return false;
	/* corrupt one block of table 0 in one copy, after the store took its digests from copy 0 */
	const uint64_t offset =
		engram_disk_block_offset(fixture->disk, 0, 0) + CORRUPT_BYTE;
	const int descriptor =
		open(fixture->paths[CORRUPT_COPY], O_RDWR | O_CLOEXEC);
	unsigned char byte = 0;
	bool corrupted = descriptor >= 0 &&
			 pread(descriptor, &byte, 1, (off_t)offset) == 1;
	byte = (unsigned char)(byte ^ 1u);
	corrupted = corrupted &&
		    pwrite(descriptor, &byte, 1, (off_t)offset) == 1 &&
		    fsync(descriptor) == 0;
	if (descriptor >= 0)
		(void)close(descriptor);
	return corrupted;
}

static void fixture_close(struct fixture *fixture)
{
	engram_disk_close(fixture->disk);
	engram_close(fixture->engram);
	pack_close(fixture->pack);
	storage_direct.reader_close(fixture->reader);
	for (uint32_t copy = 0; copy < COPIES; copy++)
		(void)unlink(fixture->paths[copy]);
	(void)rmdir(fixture->directory);
}

static struct engram_steps_options options_of(uint32_t threads_per_copy,
					      uint32_t slots_per_step)
{
	struct engram_steps_options options = {
		.threads_per_copy = threads_per_copy,
		.steps = STEPS,
		.slots_per_step = slots_per_step,
		.rows_most = ROWS_MOST,
		.spin_nanoseconds = SPIN_NANOSECONDS,
		.read_limit_nanoseconds = 0
	};
	for (uint32_t copy = 0; copy < STORAGE_MAXIMUM_COPIES; copy++)
		for (uint32_t thread = 0;
		     thread < STORAGE_FLIGHT_THREADS_PER_COPY_MOST; thread++)
			options.cpus[copy][thread] = STORAGE_FLIGHT_CPU_FREE;
	return options;
}

/* Random rows through both steps in turn; every row compared with the pattern. */
static void rows_round_trip(struct fixture *fixture, uint32_t threads_per_copy,
			    uint32_t slots_per_step, const char *what)
{
	const struct engram_description *const description =
		engram_description_of(fixture->engram);
	if (description == NULL) {
		expect(false, what);
		return;
	}
	struct engram_steps *steps = NULL;
	struct engram_step *step[STEPS] = { NULL };
	const struct engram_steps_options options =
		options_of(threads_per_copy, slots_per_step);
	bool opened = engram_steps_open(fixture->disk, &options, &steps) ==
		      STATUS_OK;
	for (uint32_t index = 0; opened && index < STEPS; index++)
		opened = engram_step_create(steps, &step[index]) == STATUS_OK;
	expect(opened, what);
	const uint32_t tables = description->table_count,
		       per_token = description->rows_per_token;
	uint64_t row_ids[ROWS_MOST * ENGRAM_MAXIMUM_TABLES *
			 ENGRAM_MAXIMUM_ROWS_PER_TOKEN];
	static unsigned char staging[ENGRAM_MAXIMUM_TABLES]
				    [ROWS_MOST * ENGRAM_MAXIMUM_ROWS_PER_TOKEN *
				     ENGRAM_ROW_BYTES];
	unsigned char *destinations[ENGRAM_MAXIMUM_TABLES] = { NULL };
	for (uint32_t table = 0; table < tables; table++)
		destinations[table] = staging[table];
	uint64_t state = (uint64_t)threads_per_copy * slots_per_step;
	bool right = true;
	for (uint32_t round = 0; opened && round < ROUNDS; round++) {
		struct engram_step *const own = step[round % STEPS];
		const uint32_t rows =
			1u + (uint32_t)(random_next(&state) % ROWS_MOST);
		for (uint32_t row = 0; row < rows; row++)
			for (uint32_t table = 0; table < tables; table++)
				for (uint32_t index = 0; index < per_token;
				     index++)
					row_ids[((uint64_t)row * tables + table) *
							per_token +
						index] =
						random_next(&state) %
						description->row_count[table];
		memset(staging, 0, sizeof(staging));
		right = right && engram_step_issue(own, row_ids, rows,
						   destinations) == STATUS_OK;
		for (uint32_t table = 0; table < tables; table++) {
			right = right &&
				engram_step_wait(own, table) == STATUS_OK;
			for (uint32_t row = 0; row < rows; row++)
				for (uint32_t index = 0; index < per_token;
				     index++)
					right = right &&
						row_is_right(
							table,
							row_ids[((uint64_t)row *
									 tables +
								 table) *
									per_token +
								index],
							staging[table] +
								((uint64_t)row *
									 per_token +
								 index) *
									ENGRAM_ROW_BYTES);
		}
	}
	expect(right, what);
	for (uint32_t index = 0; index < STEPS; index++)
		expect(engram_step_destroy(step[index]),
		       "a waited step is destroyed");
	engram_steps_close(steps);
}

/* One step's random rows (table 0's block 0, corrupt on one copy, always among them), issued and every table waited
 * for, from waiter_count threads at once (each its own tables, as cards wait); true when every row is right. */
struct waiter {
	struct engram_step *step;
	uint32_t first, tables, stride;
	bool right;
};

static void *waiter_run(void *argument)
{
	struct waiter *const waiter = argument;
	for (uint32_t table = waiter->first; table < waiter->tables;
	     table += waiter->stride)
		waiter->right = waiter->right &&
				engram_step_wait(waiter->step, table) ==
					STATUS_OK;
	return NULL;
}

static bool round_once(struct fixture *fixture, struct engram_step *step,
		       uint64_t *state, uint32_t waiter_count)
{
	const struct engram_description *const description =
		engram_description_of(fixture->engram);
	if (description == NULL)
		return false;
	const uint32_t tables = description->table_count,
		       per_token = description->rows_per_token;
	uint64_t row_ids[ROWS_MOST * ENGRAM_MAXIMUM_TABLES *
			 ENGRAM_MAXIMUM_ROWS_PER_TOKEN];
	static unsigned char staging[ENGRAM_MAXIMUM_TABLES]
				    [ROWS_MOST * ENGRAM_MAXIMUM_ROWS_PER_TOKEN *
				     ENGRAM_ROW_BYTES];
	unsigned char *destinations[ENGRAM_MAXIMUM_TABLES] = { NULL };
	for (uint32_t table = 0; table < tables; table++)
		destinations[table] = staging[table];
	const uint32_t rows = ROWS_MOST;
	for (uint32_t row = 0; row < rows; row++)
		for (uint32_t table = 0; table < tables; table++)
			for (uint32_t index = 0; index < per_token; index++)
				row_ids[((uint64_t)row * tables + table) *
						per_token +
					index] = random_next(state) %
						 description->row_count[table];
	row_ids[0] = 0; /* table 0, block 0 */
	memset(staging, 0, sizeof(staging));
	if (engram_step_issue(step, row_ids, rows, destinations) != STATUS_OK)
		return false;
	struct waiter waiters[ENGRAM_MAXIMUM_TABLES];
	memset(waiters, 0, sizeof(waiters));
	pthread_t threads[ENGRAM_MAXIMUM_TABLES];
	bool started[ENGRAM_MAXIMUM_TABLES] = { false };
	bool right = true;
	for (uint32_t index = 0; index < waiter_count; index++) {
		waiters[index] = (struct waiter){ step, index, tables,
						  waiter_count, true };
		started[index] = index != 0 &&
				 pthread_create(&threads[index], NULL,
						waiter_run,
						&waiters[index]) == 0;
		right = right && (index == 0 || started[index]);
	}
	(void)waiter_run(&waiters[0]);
	for (uint32_t index = 0; index < waiter_count; index++) {
		if (started[index])
			(void)pthread_join(threads[index], NULL);
		right = right && waiters[index].right;
	}
	for (uint32_t table = 0; table < tables; table++)
		for (uint32_t row = 0; row < rows; row++)
			for (uint32_t index = 0; index < per_token; index++)
				right = right &&
					row_is_right(
						table,
						row_ids[((uint64_t)row * tables +
							 table) *
								per_token +
							index],
						staging[table] +
							((uint64_t)row *
								 per_token +
							 index) *
								ENGRAM_ROW_BYTES);
	return right;
}

/* Two cards wait on the same step at once, each for its own tables (review finding 10: the issued mask is cleared
 * atomically; the thread build's sanitizer sees a plain update race). */
static void concurrent_waits(struct fixture *fixture)
{
	struct engram_steps *steps = NULL;
	struct engram_step *step = NULL;
	const struct engram_steps_options options = options_of(1, SOME_SLOTS);
	bool right = engram_steps_open(fixture->disk, &options, &steps) ==
			     STATUS_OK &&
		     engram_step_create(steps, &step) == STATUS_OK;
	const struct engram_description *const description =
		engram_description_of(fixture->engram);
	const uint32_t waiters = description != NULL &&
						 description->table_count > 1u ?
					 CONCURRENT_WAITERS :
					 1u;
	uint64_t state = CONCURRENT_ROUNDS;
	for (uint32_t round = 0; right && round < CONCURRENT_ROUNDS; round++)
		right = round_once(fixture, step, &state, waiters);
	expect(right, "two cards wait on one step's tables at once");
	expect(engram_step_destroy(step), "the step is destroyed");
	engram_steps_close(steps);
}

static void refusals(struct fixture *fixture)
{
	struct engram_steps *steps = NULL;
	struct engram_step *step = NULL;
	const struct engram_steps_options options = options_of(1, 1);
	expect(engram_steps_open(NULL, &options, &steps) ==
		       STATUS_INVALID_ARGUMENT,
	       "no store is refused");
	struct engram_steps_options zero = options;
	zero.slots_per_step = 0;
	expect(engram_steps_open(fixture->disk, &zero, &steps) ==
		       STATUS_INVALID_ARGUMENT,
	       "no slot is refused");
	expect(engram_steps_open(fixture->disk, &options, &steps) == STATUS_OK,
	       "steps open");
	expect(engram_step_create(steps, &step) == STATUS_OK,
	       "a step is created");
	struct engram_step *more = NULL, *third = NULL;
	expect(engram_step_create(steps, &more) == STATUS_OK &&
		       engram_step_create(steps, &third) == STATUS_BUSY_VALUE,
	       "no more steps than the options name");
	const struct engram_description *const description =
		engram_description_of(fixture->engram);
	uint64_t row_ids[ENGRAM_MAXIMUM_TABLES *
			 ENGRAM_MAXIMUM_ROWS_PER_TOKEN] = { 0 };
	row_ids[0] = description != NULL ? description->row_count[0] :
					   UINT64_MAX;
	static unsigned char
		staging[ENGRAM_MAXIMUM_TABLES]
		       [ENGRAM_MAXIMUM_ROWS_PER_TOKEN * ENGRAM_ROW_BYTES];
	unsigned char *destinations[ENGRAM_MAXIMUM_TABLES] = { staging[0],
							       staging[1] };
	expect(engram_step_issue(step, row_ids, 1, destinations) ==
		       STATUS_INVALID_ARGUMENT,
	       "a row past its table");
	expect(engram_step_issue(step, row_ids, ROWS_MOST + 1u, destinations) ==
		       STATUS_INVALID_ARGUMENT,
	       "more rows than rows_most");
	expect(engram_step_wait(step, ENGRAM_MAXIMUM_TABLES) ==
		       STATUS_INVALID_ARGUMENT,
	       "a table past the count");
	expect(engram_step_wait(step, 0) == STATUS_OK,
	       "a table nothing was issued for is ready");
	expect(engram_step_destroy(more) && engram_step_destroy(step),
	       "unissued steps are destroyed");
	engram_steps_close(steps);
}

#if CHECKING_HOOKS_ENABLED
/* The store's copies whose rings the flight retired (the flight counts a copy it does not have as retired). */
static uint32_t copies_retired(struct engram_steps *steps, uint32_t copies)
{
	uint32_t retired = 0;
	for (uint32_t copy = 0; copy < copies; copy++)
		retired += storage_flight_copy_retired(
				   engram_steps_flight(steps), copy) ?
				   1u :
				   0u;
	return retired;
}

static uint32_t slots_quarantined(struct engram_steps *steps)
{
	uint32_t quarantined = 0;
	for (uint32_t slot = 0; slot < FLIGHT_SLOTS; slot++)
		quarantined += storage_flight_slot_quarantined(
				       engram_steps_flight(steps), slot) ?
				       1u :
				       0u;
	return quarantined;
}

static uint64_t elapsed_since(const struct timespec *start)
{
	struct timespec now;
	(void)clock_gettime(CLOCK_MONOTONIC, &now);
	return (uint64_t)(now.tv_sec - start->tv_sec) * NANOSECONDS_PER_SECOND +
	       (uint64_t)now.tv_nsec - (uint64_t)start->tv_nsec;
}

/* A store of its own over count of the fixture's files, in the order names gives, without a cache. */
struct store {
	struct storage_reader *reader;
	struct pack_record *pack;
	struct engram_disk *disk;
};

static bool store_open(const struct fixture *fixture, const char *const *names,
		       uint32_t count, struct store *store)
{
	memset(store, 0, sizeof(*store));
	result_status status =
		storage_direct.reader_open(names, count, &store->reader);
	status = status == STATUS_OK ?
			 pack_open_entry(&storage_direct, store->reader,
					 &store->pack) :
			 status;
	status = status == STATUS_OK ?
			 engram_disk_open(
				 engram_description_of(fixture->engram),
				 store->pack, &storage_direct, store->reader, 0,
				 0, 1, 1, &store->disk) :
			 status;
	return status == STATUS_OK;
}

static void store_close(struct store *store)
{
	engram_disk_close(store->disk);
	pack_close(store->pack);
	storage_direct.reader_close(store->reader);
}

/* Steps over a store: one thread a copy, a read deadline of 50 ms and the backoff given. */
static bool steps_open(struct engram_disk *disk, uint64_t backoff,
		       struct engram_steps **steps, struct engram_step **step)
{
	struct engram_steps_options options = options_of(1, SOME_SLOTS);
	options.read_limit_nanoseconds = SHORT_READ_LIMIT_NANOSECONDS;
	options.reinstate_backoff_nanoseconds = backoff;
	*steps = NULL;
	*step = NULL;
	return engram_steps_open(disk, &options, steps) == STATUS_OK &&
	       engram_step_create(*steps, step) == STATUS_OK;
}

static bool all_live(struct engram_steps *steps, uint32_t copies)
{
	return copies_retired(steps, copies) == 0 &&
	       engram_steps_probation(steps) == 0;
}

static bool any_retired(struct engram_steps *steps, uint32_t copies)
{
	return copies_retired(steps, copies) != 0;
}

/* Rounds until done holds (at most RECOVERY_ROUNDS_MOST); true when it held and every row was right. */
static bool rounds_until(struct fixture *fixture, struct engram_step *step,
			 uint64_t *state, struct engram_steps *steps,
			 uint32_t copies,
			 bool (*done)(struct engram_steps *, uint32_t))
{
	bool right = true;
	for (uint32_t round = 0;
	     right && round < RECOVERY_ROUNDS_MOST && !done(steps, copies);
	     round++)
		right = round_once(fixture, step, state, 1);
	return right && done(steps, copies);
}

/* Review findings 4, 5, 6 and the missed items on retirement, over a store without a cache (every block is read), a
 * read deadline of 50 ms, one thread a copy and a backoff longer than the test. */
static void retirement_cases(struct fixture *fixture)
{
	struct engram_steps *steps = NULL;
	struct engram_step *step = NULL;
	const bool opened = steps_open(fixture->disk, LONG_BACKOFF_NANOSECONDS,
				       &steps, &step);
	expect(opened, "a flight with a 50 ms read deadline opens");
	if (!opened) {
		engram_steps_close(steps);
		return;
	}
	uint64_t state = SEED_RETIREMENT;
	/* a read the kernel never answers, while the thread completes the step's others: its own deadline retires the
	 * ring in time (finding 4), its slot is quarantined (finding 6) and the waiter reads its block from another
	 * copy */
	struct timespec start;
	(void)clock_gettime(CLOCK_MONOTONIC, &start);
	expect(fault_arm("storage_flight_hold", 1), "the hold fault arms");
	expect(round_once(fixture, step, &state, 1),
	       "the waiter reads a held read's block from another copy: every row right");
	expect(fault_disarm() != 0, "the hold fault point is reached");
	expect(elapsed_since(&start) <
		       (uint64_t)COPIES * SHORT_READ_LIMIT_NANOSECONDS,
	       "within the read deadline and its re-read, not the waiter's limit");
	expect(copies_retired(steps, COPIES) == 1u,
	       "the held read's copy is retired");
	struct storage_flight_counts counts;
	storage_flight_counts(engram_steps_flight(steps), &counts);
	expect(counts.deadline_retirements == 1u &&
		       counts.other_retirements == 0u &&
		       counts.answered != 0u && counts.age_most_ns != 0u &&
		       counts.age_most_ns < counts.read_limit_ns,
	       "the flight counts one retirement past the deadline and the answered reads' ages");
	const uint32_t quarantined = slots_quarantined(steps);
	expect(quarantined != 0u, "the held read's slot is quarantined");
	/* later steps skip the retired copy at issue (missed item: retirement ignored at issue) and never reuse the
	 * slot */
	for (uint32_t round = 0; round < ROUNDS / STEPS; round++)
		expect(round_once(fixture, step, &state, 1),
		       "later steps skip the retired copy");
	expect(copies_retired(steps, COPIES) == 1u &&
		       slots_quarantined(steps) == quarantined,
	       "the copy stays retired and the slot stays out");
	/* retire the other two: the last held read's block has every copy tried or retired, so it goes to the waiter
	 * (finding 5: no tried copy is chosen again) */
	for (uint32_t more = 0; more < COPIES - 1u; more++) {
		expect(fault_arm("storage_flight_hold", 1),
		       "the hold fault arms again");
		expect(round_once(fixture, step, &state, 1),
		       "a held read on a later copy: every row right");
		expect(fault_disarm() != 0,
		       "the hold fault point is reached again");
	}
	/* (a copy the waiters read verified meanwhile may be reinstated at an issue: no count of the end state) */
	expect(copies_retired(steps, COPIES) != 0u,
	       "the held reads retire their copies");
	expect(engram_step_destroy(step), "the step is destroyed");
	engram_steps_close(steps);
}

/* Finding 4's busy-thread variant: one copy, one thread, and a held read in one step while the other keeps that
 * thread completing reads, round after round, as the held read ages. Its own deadline still retires the ring in time
 * (before, a thread with traffic never timed a read out), and the held step's block then comes from the waiter. */
static void busy_thread_case(struct fixture *fixture)
{
	struct store store;
	struct engram_steps *steps = NULL;
	struct engram_step *held = NULL, *busy = NULL;
	bool opened = store_open(fixture, fixture->names, 1, &store) &&
		      steps_open(store.disk, LONG_BACKOFF_NANOSECONDS, &steps,
				 &held) &&
		      engram_step_create(steps, &busy) == STATUS_OK;
	expect(opened, "a one-copy flight with two steps opens");
	struct engram_disk *const three = fixture->disk;
	fixture->disk = store.disk;
	const struct engram_description *const description =
		engram_description_of(fixture->engram);
	static unsigned char
		staging[ENGRAM_MAXIMUM_TABLES]
		       [ENGRAM_MAXIMUM_ROWS_PER_TOKEN * ENGRAM_ROW_BYTES];
	uint64_t row_ids[ENGRAM_MAXIMUM_TABLES *
			 ENGRAM_MAXIMUM_ROWS_PER_TOKEN] = { 0 };
	unsigned char *destinations[ENGRAM_MAXIMUM_TABLES] = { NULL };
	for (uint32_t table = 0;
	     description != NULL && table < description->table_count; table++)
		destinations[table] = staging[table];
	opened = opened && description != NULL;
	uint64_t state = SEED_BUSY;
	struct timespec start;
	(void)clock_gettime(CLOCK_MONOTONIC, &start);
	expect(fault_arm("storage_flight_hold", 1), "the hold fault arms");
	opened = opened &&
		 engram_step_issue(held, row_ids, 1, destinations) == STATUS_OK;
	uint64_t retired_after = UINT64_MAX;
	for (uint32_t round = 0;
	     opened && round < BUSY_ROUNDS_MOST &&
	     retired_after == UINT64_MAX &&
	     elapsed_since(&start) < BUSY_ROUNDS_LIMIT_NANOSECONDS;
	     round++) {
		expect(round_once(fixture, busy, &state, 1),
		       "the busy step's rows are right");
		if (copies_retired(steps, 1) != 0)
			retired_after = elapsed_since(&start);
	}
	expect(fault_disarm() != 0, "the hold fault point is reached");
	expect(retired_after <
		       BUSY_RETIRE_LIMITS * SHORT_READ_LIMIT_NANOSECONDS,
	       "the held read's deadline retires its ring while the thread is busy");
	bool right = opened;
	for (uint32_t table = 0; opened && table < description->table_count;
	     table++) {
		right = right && engram_step_wait(held, table) == STATUS_OK;
		for (uint32_t index = 0; index < description->rows_per_token;
		     index++)
			right = right &&
				row_is_right(table, 0,
					     staging[table] +
						     (uint64_t)index *
							     ENGRAM_ROW_BYTES);
	}
	expect(right,
	       "the held step's block comes from the waiter: rows right");
	expect(!opened ||
		       (engram_step_destroy(held) && engram_step_destroy(busy)),
	       "both steps are destroyed");
	fixture->disk = three;
	engram_steps_close(steps);
	store_close(&store);
}

/* Finding 5's old livelock: two copies, copy 0 bad for table 0's block 0 and copy 1 retired by a held read. A read of
 * that block fails on copy 0, and with every copy tried or retired it goes to the waiter, which reads copy 1 (before,
 * copy 0 was chosen again, without end, until the waiter's limit). */
static void livelock_case(struct fixture *fixture)
{
	const char *const names[] = { fixture->paths[CORRUPT_COPY],
				      fixture->paths[0] };
	struct store store;
	struct engram_steps *steps = NULL;
	struct engram_step *step = NULL;
	const bool opened =
		store_open(fixture, names, sizeof(names) / sizeof(names[0]),
			   &store) &&
		steps_open(store.disk, LONG_BACKOFF_NANOSECONDS, &steps, &step);
	expect(opened, "a two-copy store, copy 0 corrupt, opens");
	struct engram_disk *const three = fixture->disk;
	fixture->disk = store.disk;
	uint64_t state = SEED_LIVELOCK;
	if (opened) {
		/* copy 0 is tried last, so the step's reads, and the held one, go to copy 1 */
		engram_disk_copy_state(store.disk, 0, true);
		expect(fault_arm("storage_flight_hold", 1),
		       "the hold fault arms");
		expect(round_once(fixture, step, &state, 1),
		       "copy 1's held read: every row right");
		expect(fault_disarm() != 0, "the hold fault point is reached");
		const struct storage_flight *const flight =
			engram_steps_flight(steps);
		expect(storage_flight_copy_retired(flight, 1) &&
			       !storage_flight_copy_retired(flight, 0),
		       "copy 1 is retired, copy 0 live");
		struct timespec start;
		(void)clock_gettime(CLOCK_MONOTONIC, &start);
		expect(round_once(fixture, step, &state, 1),
		       "the block bad on copy 0 goes to the waiter: every row right");
		expect(elapsed_since(&start) <
			       (uint64_t)COPIES * SHORT_READ_LIMIT_NANOSECONDS,
		       "no re-read of the tried copy until the waiter's limit");
		expect(engram_step_destroy(step), "the step is destroyed");
	}
	fixture->disk = three;
	engram_steps_close(steps);
	store_close(&store);
}

/* One copy (missed items: no recovery after retirement, reinstatement; the destroy's drain): its ring retired, a step
 * destroyed unwaited with its blocks on the fallback list drains (before, the unread blocks' status read as a
 * timeout), the next step's blocks are all read by the waiters, and their verified reads bring the copy back through
 * its fresh ring and its probe. */
static void single_copy_cases(struct fixture *fixture)
{
	struct store store;
	struct engram_steps *steps = NULL;
	struct engram_step *step = NULL, *other = NULL;
	const bool opened =
		store_open(fixture, fixture->names, 1, &store) &&
		steps_open(store.disk, LONG_BACKOFF_NANOSECONDS, &steps, &step);
	expect(opened, "a one-copy store's flight opens");
	struct engram_disk *const three = fixture->disk;
	fixture->disk = store.disk;
	uint64_t state = SEED_SINGLE_COPY;
	if (opened) {
		expect(fault_arm("storage_flight_hold", 1),
		       "the hold fault arms");
		expect(round_once(fixture, step, &state, 1),
		       "the only copy's held read: the waiter reads its block");
		expect(fault_disarm() != 0, "the hold fault point is reached");
		expect(copies_retired(steps, 1) == 1u,
		       "the only copy is retired");
		const struct engram_description *const description =
			engram_description_of(fixture->engram);
		uint64_t row_ids[ENGRAM_MAXIMUM_TABLES *
				 ENGRAM_MAXIMUM_ROWS_PER_TOKEN] = { 0 };
		static unsigned char staging[ENGRAM_MAXIMUM_TABLES]
					    [ENGRAM_MAXIMUM_ROWS_PER_TOKEN *
					     ENGRAM_ROW_BYTES];
		unsigned char *destinations[ENGRAM_MAXIMUM_TABLES] = { NULL };
		for (uint32_t table = 0;
		     description != NULL && table < description->table_count;
		     table++)
			destinations[table] = staging[table];
		expect(engram_step_create(steps, &other) == STATUS_OK &&
			       engram_step_issue(other, row_ids, 1,
						 destinations) == STATUS_OK &&
			       engram_step_destroy(other),
		       "a step destroyed with its blocks left to the waiters drains: destroy keeps nothing");
		expect(engram_steps_live(steps) == 1u,
		       "only the step in use is left");
		expect(round_once(fixture, step, &state, 1),
		       "with every copy retired the waiters read every block");
		expect(!all_live(steps, 1),
		       "the copy is not back within that step");
		expect(rounds_until(fixture, step, &state, steps, 1, all_live),
		       "the copy that answered the waiters' verified reads comes back through its probe");
		for (uint32_t round = 0; round < ROUNDS / STEPS; round++)
			expect(round_once(fixture, step, &state, 1),
			       "the reinstated ring serves: every row right");
		expect(engram_step_destroy(step), "the step is destroyed");
	}
	fixture->disk = three;
	engram_steps_close(steps);
	store_close(&store);
}

/* Active recovery (missed item: reinstatement was passive): three copies, one retired by a held read. With healthy
 * peers no waiter reads it, so no verified read can bring it back; after its backoff its rings are asked for, the
 * probe answers, and it serves again. */
static void backoff_case(struct fixture *fixture)
{
	struct engram_steps *steps = NULL;
	struct engram_step *step = NULL;
	const bool opened = steps_open(fixture->disk, SHORT_BACKOFF_NANOSECONDS,
				       &steps, &step);
	expect(opened, "a flight with a 20 ms backoff opens");
	uint64_t state = SEED_BACKOFF;
	if (opened) {
		expect(fault_arm("storage_flight_hold", 1),
		       "the hold fault arms");
		expect(round_once(fixture, step, &state, 1),
		       "a held read: every row right");
		expect(fault_disarm() != 0, "the hold fault point is reached");
		expect(copies_retired(steps, COPIES) == 1u,
		       "the held read's copy is retired");
		expect(rounds_until(fixture, step, &state, steps, COPIES,
				    all_live),
		       "after its backoff the retired copy is probed and serves again");
		struct storage_flight_counts counts;
		storage_flight_counts(engram_steps_flight(steps), &counts);
		expect(engram_steps_probations(steps) == 1u &&
			       counts.reinstatements == 1u,
		       "one probation and one ring set up again are counted");
		expect(engram_step_destroy(step), "the step is destroyed");
	}
	engram_steps_close(steps);
}

static uint32_t
	lost_copy; /* probe_isolation_case's copy whose probe never answers */

static bool lost_on_probation(struct engram_steps *steps, uint32_t copies)
{
	(void)copies;
	return (engram_steps_probation(steps) >> lost_copy & 1u) != 0 &&
	       !storage_flight_copy_retired(engram_steps_flight(steps),
					    lost_copy);
}

static bool others_back(struct engram_steps *steps, uint32_t copies)
{
	return copies_retired(steps, copies) == 0 &&
	       engram_steps_probation(steps) == 1u << lost_copy;
}

/* Review NF1: one probe a copy. A copy retires and its probe is lost (its drive never answers it); then another copy
 * retires, and it comes back through its own probe while the first stays on probation (before, one probe slot served
 * every copy, and the lost probe held every other copy on probation for good). */
static void probe_isolation_case(struct fixture *fixture)
{
	struct engram_steps *steps = NULL;
	struct engram_step *step = NULL;
	const bool opened = steps_open(fixture->disk, SHORT_BACKOFF_NANOSECONDS,
				       &steps, &step);
	expect(opened, "a flight with a 20 ms backoff opens");
	uint64_t state = SEED_PROBE;
	if (opened) {
		expect(fault_arm("storage_flight_hold", 1),
		       "the hold fault arms");
		expect(round_once(fixture, step, &state, 1),
		       "a held read: every row right");
		expect(fault_disarm() != 0, "the hold fault point is reached");
		for (uint32_t copy = 0; copy < COPIES; copy++)
			lost_copy = storage_flight_copy_retired(
					    engram_steps_flight(steps), copy) ?
					    copy :
					    lost_copy;
		expect(fault_arm("engram_step_probe_lost", 1),
		       "the lost probe fault arms");
		expect(rounds_until(fixture, step, &state, steps, COPIES,
				    lost_on_probation) &&
			       round_once(fixture, step, &state, 1),
		       "the retired copy's rings come back and its probe starts");
		expect(fault_disarm() != 0, "its probe is lost");
		expect(fault_arm("storage_flight_hold", 1),
		       "the hold fault arms again");
		expect(rounds_until(fixture, step, &state, steps, COPIES,
				    any_retired),
		       "a held read retires another copy");
		expect(fault_disarm() != 0, "the hold fault point is reached");
		expect(rounds_until(fixture, step, &state, steps, COPIES,
				    others_back),
		       "the other copy comes back through its own probe, the lost one's held back alone");
		expect(engram_step_destroy(step), "the step is destroyed");
	}
	engram_steps_close(steps);
}

/* Review V1: damping. One copy retired again and again soon after it came back waits twice its last backoff each
 * time (before, the backoff started again from the first at each return). */
static void damping_case(struct fixture *fixture)
{
	struct store store;
	struct engram_steps *steps = NULL;
	struct engram_step *step = NULL;
	const bool opened = store_open(fixture, fixture->names, 1, &store) &&
			    steps_open(store.disk, DAMPING_BACKOFF_NANOSECONDS,
				       &steps, &step);
	expect(opened, "a one-copy flight opens");
	struct engram_disk *const three = fixture->disk;
	fixture->disk = store.disk;
	uint64_t state = SEED_DAMPING;
	for (uint32_t retirement = 0; opened && retirement < DAMPED_RETIREMENTS;
	     retirement++) {
		expect(rounds_until(fixture, step, &state, steps, 1, all_live),
		       "the copy is back before the next held read");
		expect(fault_arm("storage_flight_hold", 1),
		       "the hold fault arms");
		expect(rounds_until(fixture, step, &state, steps, 1,
				    any_retired),
		       "a held read retires the ring");
		expect(fault_disarm() != 0, "the hold fault point is reached");
		expect(round_once(fixture, step, &state, 1),
		       "the step after notes the retirement");
		expect(engram_steps_backoff(steps, 0) ==
			       DAMPING_BACKOFF_NANOSECONDS << retirement,
		       "each retirement soon after the copy came back doubles its backoff");
	}
	expect(!opened || engram_step_destroy(step), "the step is destroyed");
	fixture->disk = three;
	engram_steps_close(steps);
	store_close(&store);
}

/* One copy, rings retired again and again. A read whose completion lands after its ring retired gives its slot back
 * and lets the ring close, so a copy comes back more often than the rings a thread keeps (before, every retired ring
 * stayed, and the fifth retirement was the last). Rings whose reads never return fill the graveyard; the refused
 * reinstatement is then asked again only after the backoff, not at every issue (review N2). */
static void ring_cap_cases(struct fixture *fixture)
{
	struct store store;
	struct engram_steps *steps = NULL;
	struct engram_step *step = NULL;
	const bool opened =
		store_open(fixture, fixture->names, 1, &store) &&
		steps_open(store.disk, LONG_BACKOFF_NANOSECONDS, &steps, &step);
	expect(opened, "a one-copy flight opens");
	struct engram_disk *const three = fixture->disk;
	fixture->disk = store.disk;
	uint64_t state = SEED_CAP;
	for (uint32_t cycle = 0; opened && cycle < LATE_CYCLES; cycle++) {
		expect(rounds_until(fixture, step, &state, steps, 1, all_live),
		       "the copy is back before the next late read");
		expect(fault_arm("storage_flight_late", 1),
		       "the late fault arms");
		expect(rounds_until(fixture, step, &state, steps, 1,
				    any_retired),
		       "a read whose ring retired while the kernel serves it");
		expect(fault_disarm() != 0, "the late fault point is reached");
	}
	struct timespec start;
	(void)clock_gettime(CLOCK_MONOTONIC, &start);
	while (opened && slots_quarantined(steps) != 0 &&
	       elapsed_since(&start) < RECLAIM_LIMIT_NANOSECONDS)
		expect(round_once(fixture, step, &state, 1),
		       "rows right while the late completions land");
	expect(opened && slots_quarantined(steps) == 0,
	       "each late completion gave its slot back");
	expect(opened &&
		       rounds_until(fixture, step, &state, steps, 1, all_live),
	       "the copy comes back after more retirements than the rings a thread keeps");
	for (uint32_t cycle = 0;
	     opened && cycle < STORAGE_FLIGHT_RETIRED_RINGS_MOST; cycle++) {
		expect(rounds_until(fixture, step, &state, steps, 1, all_live),
		       "the copy is back before the next held read");
		expect(fault_arm("storage_flight_hold", 1),
		       "the hold fault arms");
		expect(rounds_until(fixture, step, &state, steps, 1,
				    any_retired),
		       "a held read retires the ring");
		expect(fault_disarm() != 0, "the hold fault point is reached");
	}
	/* every place taken by a ring still owed its held read: the ask is refused once, then waits for the backoff */
	expect(fault_arm("storage_flight_reinstate_ask", ASK_COUNT_CALL),
	       "the ask counter arms");
	for (uint32_t round = 0; opened && round < COUNTED_ROUNDS; round++)
		expect(round_once(fixture, step, &state, 1),
		       "the waiters serve while the copy stays retired");
	expect(fault_disarm() == 1u,
	       "one ask, refused, and none at the issues after it");
	expect(opened && copies_retired(steps, 1) == 1u,
	       "the copy with no ring place left stays retired");
	expect(!opened || engram_step_destroy(step), "the step is destroyed");
	fixture->disk = three;
	engram_steps_close(steps);
	store_close(&store);
}

/* The flight's other faults (findings 2, 7 and 8, and 9 with a kept step). */
static void flight_cases(struct fixture *fixture)
{
	struct engram_steps *steps = NULL;
	struct engram_step *step = NULL;
	struct engram_steps_options options = options_of(1, SOME_SLOTS);
	options.read_limit_nanoseconds = SHORT_READ_LIMIT_NANOSECONDS;
	options.reinstate_backoff_nanoseconds = LONG_BACKOFF_NANOSECONDS;
	uint64_t state = SEED_FLIGHT;
	/* a refused buffer registration: the ring reads with plain addresses (finding 2) */
	expect(fault_arm("storage_flight_register", 1),
	       "the registration fault arms");
	bool opened = engram_steps_open(fixture->disk, &options, &steps) ==
			      STATUS_OK &&
		      engram_step_create(steps, &step) == STATUS_OK;
	expect(fault_disarm() != 0, "the registration fault point is reached");
	expect(opened, "a ring without registered buffers opens");
	expect(opened && storage_flight_plain_rings(
				 engram_steps_flight(steps)) == 1u,
	       "the flight counts the ring that reads without registered buffers");
	for (uint32_t round = 0; opened && round < ROUNDS / STEPS; round++)
		expect(round_once(fixture, step, &state, 1),
		       "plain reads: every row right");
	/* a submit the kernel refused with EAGAIN is retried on the thread's next turn (finding 8) */
	struct timespec start;
	(void)clock_gettime(CLOCK_MONOTONIC, &start);
	expect(fault_arm("storage_flight_enter_busy", 1),
	       "the busy fault arms");
	expect(opened && round_once(fixture, step, &state, 1),
	       "a refused submit is retried: every row right");
	expect(fault_disarm() != 0, "the busy fault point is reached");
	expect(elapsed_since(&start) < SHORT_READ_LIMIT_NANOSECONDS,
	       "the refused reads go in before any deadline");
	/* a ring retired with reads still in its inbox completes them, unquarantined (finding 7) */
	expect(fault_arm("storage_flight_retire", 1), "the retire fault arms");
	expect(opened && round_once(fixture, step, &state, 1),
	       "the inbox of a retired ring is completed: every row right");
	expect(fault_disarm() != 0, "the retire fault point is reached");
	expect(opened && copies_retired(steps, COPIES) == 1u &&
		       slots_quarantined(steps) == 0u,
	       "the copy is retired and no inbox slot is quarantined");
	expect(!opened || engram_step_destroy(step), "the step is destroyed");
	engram_steps_close(steps);
	/* finding 9 and review N3: a destroy whose waiter gives up keeps a step whose one block's read is held, still
	 * in flight; the close drains it first (its deadline retires the ring), stops the threads, then frees the
	 * kept step */
	struct engram_step *kept = NULL;
	options.read_limit_nanoseconds = KEPT_READ_LIMIT_NANOSECONDS;
	steps = NULL;
	opened = engram_steps_open(fixture->disk, &options, &steps) ==
			 STATUS_OK &&
		 engram_step_create(steps, &kept) == STATUS_OK;
	uint64_t row_ids[ENGRAM_MAXIMUM_TABLES *
			 ENGRAM_MAXIMUM_ROWS_PER_TOKEN] = { 0 };
	static unsigned char
		staging[ENGRAM_MAXIMUM_ROWS_PER_TOKEN * ENGRAM_ROW_BYTES];
	unsigned char *destinations[ENGRAM_MAXIMUM_TABLES] = {
		staging
	}; /* table 0 only: its one block is the held read */
	expect(fault_arm("storage_flight_hold", 1), "the hold fault arms");
	opened = opened &&
		 engram_step_issue(kept, row_ids, 1, destinations) == STATUS_OK;
	const struct timespec moved = { .tv_sec = 0,
					.tv_nsec = HOLD_SETTLE_NANOSECONDS };
	(void)nanosleep(&moved, NULL); /* the I/O thread takes the read */
	expect(fault_disarm() != 0, "the hold fault point is reached");
	expect(fault_arm("engram_step_wait_short", 1),
	       "the short wait fault arms");
	expect(opened && !engram_step_destroy(kept),
	       "a destroy whose waiter gave up keeps the step");
	expect(fault_disarm() != 0, "the short wait fault point is reached");
	uint32_t in_flight = 0;
	for (uint32_t copy = 0; opened && copy < COPIES; copy++)
		in_flight +=
			storage_flight_reads(engram_steps_flight(steps), copy);
	expect(in_flight == 1u && engram_steps_live(steps) == 1u,
	       "the kept step's held read is still in flight");
	(void)clock_gettime(CLOCK_MONOTONIC, &start);
	engram_steps_close(steps);
	expect(elapsed_since(&start) <
		       CLOSE_DRAIN_LIMITS * KEPT_READ_LIMIT_NANOSECONDS +
			       CLOSE_SLACK_NANOSECONDS,
	       "the close drained the held read within twice its deadline");
}
#endif

static void faults(struct fixture *fixture)
{
#if CHECKING_HOOKS_ENABLED
	struct engram_steps *steps = NULL;
	const struct engram_steps_options options = options_of(1, 1);
	expect(fault_arm("storage_flight_ring", 1), "the ring fault arms");
	expect(engram_steps_open(fixture->disk, &options, &steps) ==
			       STATUS_UNSUPPORTED &&
		       steps == NULL,
	       "a refused ring fails the open and leaves nothing");
	expect(fault_disarm() != 0, "the ring fault point is reached");
	expect(fault_arm("storage_flight_thread", 1), "the thread fault arms");
	expect(engram_steps_open(fixture->disk, &options, &steps) ==
			       STATUS_OUT_OF_MEMORY &&
		       steps == NULL,
	       "a thread that does not start fails the open and leaves nothing");
	expect(fault_disarm() != 0, "the thread fault point is reached");
	/* a store without a cache, so that every round reads (the earlier rounds cached the whole test pack) */
	struct engram_disk *const cached = fixture->disk;
	struct engram_disk *bare = NULL;
	expect(engram_disk_open(engram_description_of(fixture->engram),
				fixture->pack, &storage_direct, fixture->reader,
				0, 0, 1, 1, &bare) == STATUS_OK,
	       "a store without a cache opens");
	fixture->disk = bare;
	for (uint64_t call = 1; bare != NULL && call <= FAULT_CALLS; call++) {
		expect(fault_arm("engram_step_digest", call),
		       "the digest fault arms");
		rows_round_trip(
			fixture, 1, 1,
			"a failed digest is read again from another copy");
		expect(fault_disarm() != 0,
		       "the digest fault point is reached");
	}
	if (bare != NULL) {
		retirement_cases(fixture);
		busy_thread_case(fixture);
		livelock_case(fixture);
		single_copy_cases(fixture);
		backoff_case(fixture);
		probe_isolation_case(fixture);
		damping_case(fixture);
		ring_cap_cases(fixture);
		flight_cases(fixture);
	}
	engram_disk_close(bare);
	fixture->disk = cached;
#else
	(void)fixture;
#endif
}

int main(void)
{
	(void)alarm(TEST_SECONDS);
	struct fixture fixture;
	memset(&fixture, 0, sizeof(fixture));
	const bool opened = fixture_open(&fixture);
	expect(opened,
	       "three copies of the test pack, one corrupt, open as a disk store");
	if (opened) {
		rows_round_trip(
			&fixture, 1, 1,
			"one thread a copy, one slot: every block through the refill chain");
		rows_round_trip(&fixture, SOME_THREADS_PER_COPY, SOME_SLOTS,
				"two threads a copy, four slots");
		rows_round_trip(&fixture, STORAGE_FLIGHT_THREADS_PER_COPY_MOST,
				ROWS_MOST, "the most threads a copy");
		concurrent_waits(&fixture);
		refusals(&fixture);
		faults(&fixture);
	}
	fixture_close(&fixture);
	if (failures != 0) {
		(void)fprintf(stderr, "engram_step_test: %u failures\n",
			      failures);
		return EXIT_FAILURE;
	}
	(void)puts("engram_step_test: passed");
	return EXIT_SUCCESS;
}
