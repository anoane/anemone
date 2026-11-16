/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* step_engram_io_test: a step driver whose n-gram rows come through step I/O (a disk store over three real copies of
 * the test pack, engram_step.h) destroys every step record before it frees or retires any staging, without a pipeline
 * and with one: step I/O's threads write rows into that staging. The backend's staging release counts the step records
 * still alive (review finding 1). With checking hooks: a pipeline step whose n-gram read never returns lands within
 * the driver's wait limit, its read deadline sized from that limit (review N1); and a driver torn down while a read of
 * its record never returned keeps that staging, neither freed nor retired, until the close joins the I/O threads:
 * without a pipeline, and with one (a pipeline slot's record, issued through the checking-only slot hook). */
#define TEST_STEP_FIXTURE_IMPLEMENTATION
#include "step_fixture.h"
#include "step_programs.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "anemone/engram_disk.h"
#include "anemone/hooks.h"
#include "anemone/pack.h"
#include "engram_step.h"
#include "step_driver.h"

#define COPIES 3u
#define DIRECTORY_BYTES 256u
#define PATH_BYTES 512u
#define STEP_RECORDS 8u
#define STEP_ROWS 8u
#define STEP_SLOTS 4u
#define PIPELINE_DEPTH 3u
#define LONG_BACKOFF_NANOSECONDS \
	60000000000ull /* no retired copy comes back while the test runs */
#define KEPT_READ_LIMIT_NANOSECONDS \
	1000000000ull /* the kept store's read deadline: its held read outlives the teardown */
#define HELD_TABLE_READ \
	2u /* the one-copy store's second read: table 1's (table 0's is the first) */
#define CORRUPT_BYTE 100u
#define TOKEN_STRIDE 7u
#define RECORDED_MOST 64u
#define NANOSECONDS_PER_SECOND 1000000000ull
#define HOLD_SETTLE_POLL_NANOSECONDS \
	1000000l /* between looks at the reads in flight: 1 ms */
#define HOLD_SETTLE_LIMIT_NANOSECONDS \
	200000000ull /* the held read's partner lands well inside the kept deadline */
#define CLOSE_DRAIN_LIMITS \
	2u /* the flight's close waits at most twice the read limit for a held read */
#define CLOSE_SLACK_NANOSECONDS 300000000ull

static struct engram_steps *watched_steps;
static uint32_t releases, releases_with_live_steps;

static void watch(void)
{
	releases++;
	releases_with_live_steps +=
		watched_steps != NULL && engram_steps_live(watched_steps) != 0 ?
			1u :
			0u;
}

static void watched_staging_free(void *block)
{
	if (block != NULL)
		watch();
	test_host_memory_backend.staging_free(block);
}

#if CHECKING_HOOKS_ENABLED
/* The staging blocks a driver holds now (allocated, not released): a kept one is freed after the close. */
static void *recorded[RECORDED_MOST];
static uint32_t recorded_count;

MUST_CHECK static result_status recorded_staging_allocate(uint64_t byte_count,
							  void **memory)
{
	const result_status status =
		test_host_memory_backend.staging_allocate(byte_count, memory);
	if (status == STATUS_OK && recorded_count < RECORDED_MOST)
		recorded[recorded_count++] = *memory;
	return status;
}

static void recorded_staging_free(void *block)
{
	for (uint32_t index = 0; block != NULL && index < recorded_count;
	     index++)
		if (recorded[index] == block) {
			recorded[index] = recorded[--recorded_count];
			break;
		}
	test_host_memory_backend.staging_free(block);
}

/* A retire releases the block too (the host backend has none): it is no longer held. */
static void recorded_staging_retire(void *block)
{
	recorded_staging_free(block);
}

/* At teardown (once teardown_failing is set) a card's upload wait fails, so the driver takes its staging as still
 * read by the card: the retire branch, not the free. */
static const struct backend_operations *wait_base;
static bool teardown_failing;
static uint32_t teardown_failures;
MUST_CHECK static result_status
failing_upload_wait(struct device_card *card, uint64_t ticket,
		    uint64_t wait_limit_nanoseconds)
{
	if (teardown_failing) {
		teardown_failures++;
		return STATUS_TIMED_OUT;
	}
	return wait_base->upload_wait(card, ticket, wait_limit_nanoseconds);
}
#endif

struct disk_fixture {
	char directory[DIRECTORY_BYTES], paths[COPIES][PATH_BYTES];
	const char *names[COPIES];
	uint32_t copies;
	struct storage_reader *reader;
	struct pack_record *pack;
	struct engram_record *engram;
	struct engram_disk *disk;
	struct engram_steps *steps;
};

/* The test pack's header at the start of its bytes (the pack writer aligns it). */
static const void *pack_bytes_header(void)
{
	return engram_pack;
}

/* pack_copies files of the test pack in a directory of their own, served by a disk store with step I/O whose reads have
 * the deadline given. */
static bool disk_open(struct disk_fixture *disk, uint32_t pack_copies,
		      const char *name, uint64_t read_limit)
{
	const uint64_t pack_bytes =
		((const struct pack_header *)pack_bytes_header())
			->file_byte_count;
	disk->copies = pack_copies;
	(void)snprintf(disk->directory, sizeof(disk->directory),
		       "build/step_engram_io_test.%s.%ld", name,
		       (long)getpid());
	bool written = mkdir(disk->directory, S_IRWXU) == 0;
	for (uint32_t copy = 0; written && copy < pack_copies; copy++) {
		(void)snprintf(disk->paths[copy], sizeof(disk->paths[copy]),
			       "%s/copy%u.anemone", disk->directory, copy);
		disk->names[copy] = disk->paths[copy];
		const int descriptor =
			open(disk->paths[copy],
			     O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC,
			     S_IRUSR | S_IWUSR);
		written = descriptor >= 0 &&
			  write(descriptor, engram_pack, (size_t)pack_bytes) ==
				  (ssize_t)pack_bytes &&
			  fsync(descriptor) == 0;
		if (descriptor >= 0)
			(void)close(descriptor);
	}
	if (!written)
		return false;
	result_status status = storage_direct.reader_open(
		disk->names, pack_copies, &disk->reader);
	status = status == STATUS_OK ?
			 pack_open_entry(&storage_direct, disk->reader,
					 &disk->pack) :
			 status;
	status = status == STATUS_OK ?
			 engram_open(&storage_direct, disk->reader, disk->pack,
				     &disk->engram) :
			 status;
	status = status == STATUS_OK ?
			 engram_disk_open(engram_description_of(disk->engram),
					  disk->pack, &storage_direct,
					  disk->reader, 0, 0, 1, 1,
					  &disk->disk) :
			 status;
	status = status == STATUS_OK ?
			 engram_attach_disk(disk->engram, disk->disk) :
			 status;
	struct engram_steps_options options = {
		.threads_per_copy = 1,
		.steps = STEP_RECORDS,
		.slots_per_step = STEP_SLOTS,
		.rows_most = STEP_ROWS,
		.read_limit_nanoseconds = read_limit,
		.reinstate_backoff_nanoseconds = LONG_BACKOFF_NANOSECONDS
	};
	for (uint32_t copy = 0; copy < STORAGE_MAXIMUM_COPIES; copy++)
		for (uint32_t thread = 0;
		     thread < STORAGE_FLIGHT_THREADS_PER_COPY_MOST; thread++)
			options.cpus[copy][thread] = STORAGE_FLIGHT_CPU_FREE;
	status = status == STATUS_OK ?
			 engram_steps_open(disk->disk, &options, &disk->steps) :
			 status;
	status = status == STATUS_OK ?
			 engram_attach_steps(disk->engram, disk->steps) :
			 status;
	return status == STATUS_OK;
}

static void disk_close(struct disk_fixture *disk)
{
	engram_close(disk->engram); /* commands.c's order */
	engram_steps_close(disk->steps);
	engram_disk_close(disk->disk);
	pack_close(disk->pack);
	storage_direct.reader_close(disk->reader);
	for (uint32_t copy = 0; copy < disk->copies; copy++)
		(void)unlink(disk->paths[copy]);
	(void)rmdir(disk->directory);
}

/* One driver, with depth 0 (no pipeline) or a pipeline of that depth, made and destroyed. */
static void teardown_case(struct disk_fixture *disk, uint32_t depth,
			  const char *what)
{
	context_text = what;
	struct backend_operations watched = checked_backend;
	watched.staging_free =
		watched_staging_free; /* the host backend has no retire: a kept staging is never freed */
	struct fixture fixture;
	struct step_driver *driver = NULL;
	fixture_build(&fixture, cards_open());
	expect_status(step_driver_create(&watched, cards, fixture.pointers,
					 PROGRAMS, disk->engram, WAIT_LIMIT,
					 &driver),
		      STATUS_OK, "the driver with step I/O is made");
	expect(engram_steps_live(disk->steps) != 0,
	       "the driver holds a step record");
	if (depth != 0)
		expect_status(step_pipeline_start(driver, depth), STATUS_OK,
			      "its pipeline starts");
	watched_steps = disk->steps;
	releases = releases_with_live_steps = 0;
	capture_begin();
	step_driver_destroy(driver);
	(void)capture_end(); /* the teardown's lines stay out of the test's output */
	watched_steps = NULL;
	expect(releases != 0, "the staging is released");
	expect(releases_with_live_steps == 0,
	       "no staging is released while a step record lives");
	expect(engram_steps_live(disk->steps) == 0,
	       "every step record is destroyed");
	cards_close();
}

#if CHECKING_HOOKS_ENABLED
/* ROW_CAPACITY rows of session 0 from position 0, logits for the last, and their tokens in its history. */
static void held_rows(struct step_row_record *rows)
{
	for (uint32_t row = 0; row < ROW_CAPACITY; row++) {
		rows[row] = (struct step_row_record){
			(int32_t)row,
			(int32_t)((row * TOKEN_STRIDE + 1u) % VOCABULARY), 0,
			row + 1u == ROW_CAPACITY ? STEP_ROW_LOGITS : 0
		};
		history[0][row] = (uint32_t)rows[row].token_id;
	}
}

/* Review N1: a pipeline step one of whose n-gram reads never returns. Its deadline, sized from the driver's wait limit
 * as serving sizes it (engram_steps_read_limit_within), retires the ring, and the block is read again on another
 * copy before the pipeline, which saw no progress meanwhile, fails the step (before: the storage limit, 180 s,
 * against the cards' 10 s). */
static void pipeline_hold_case(struct disk_fixture *disk)
{
	context_text = "a pipeline step whose n-gram read never returns";
	struct fixture fixture;
	struct step_driver *driver = NULL;
	struct step_row_record rows[ROW_CAPACITY];
	struct step_row_result results[ROW_CAPACITY];
	fixture_build(&fixture, cards_open());
	/* the host backend itself, as step_test's pipeline cases: the checked one keeps unlocked records */
	expect_status(step_driver_create(&test_host_memory_backend, cards,
					 fixture.pointers, PROGRAMS,
					 disk->engram, WAIT_LIMIT, &driver),
		      STATUS_OK, "the driver with step I/O is made");
	expect_status(step_pipeline_start(driver, PIPELINE_DEPTH), STATUS_OK,
		      "its pipeline starts");
	held_rows(rows);
	struct timespec start, end;
	(void)clock_gettime(CLOCK_MONOTONIC, &start);
	expect(fault_arm("storage_flight_hold", 1), "the hold fault arms");
	expect_status(step_submit(driver, rows, ROW_CAPACITY), STATUS_OK,
		      "the step goes in flight");
	expect_status(
		step_finish(driver, results), STATUS_OK,
		"the step lands: its held block came from the waiter's synchronous read");
	expect(fault_disarm() != 0, "the hold fault point is reached");
	(void)clock_gettime(CLOCK_MONOTONIC, &end);
	const uint64_t elapsed =
		(uint64_t)(end.tv_sec - start.tv_sec) * NANOSECONDS_PER_SECOND +
		(uint64_t)end.tv_nsec - (uint64_t)start.tv_nsec;
	expect(elapsed < WAIT_LIMIT, "within the driver's wait limit");
	for (uint32_t row = 0; row < ROW_CAPACITY; row++)
		expect(engram_rows_hold(&rows[row], row),
		       "the n-gram rows of the row's tables are where the program says");
	uint32_t retired = 0;
	for (uint32_t copy = 0; copy < disk->copies; copy++)
		retired += storage_flight_copy_retired(
				   engram_steps_flight(disk->steps), copy) ?
				   1u :
				   0u;
	expect(retired == 1u, "the held read's copy is retired");
	step_driver_destroy(driver);
	cards_close();
}

/* The kept branch of the teardown (review findings 1 and the CUDA retire), for a driver without a pipeline: one copy,
 * table 0's block corrupt after the store took its digests, and table 1's read held. The step fails on table
 * 0, so no card waits for table 1; the destroy's drain gives up, and the record is kept. At teardown a card's upload
 * wait fails, so the driver takes its staging as still read: the staging the record writes into is neither freed nor
 * retired, and every other block is retired or freed. The close then drains the read and frees the kept record; the
 * staging is freed after it. */
static void kept_staging_case(struct disk_fixture *disk)
{
	context_text =
		"a driver torn down while a read of its record never returned";
	const uint64_t offset =
		engram_disk_block_offset(disk->disk, 0, 0) + CORRUPT_BYTE;
	const int descriptor = open(disk->paths[0], O_RDWR | O_CLOEXEC);
	unsigned char byte = 0;
	bool corrupted = descriptor >= 0 &&
			 pread(descriptor, &byte, 1, (off_t)offset) == 1;
	byte = (unsigned char)(byte ^ 1u);
	corrupted = corrupted &&
		    pwrite(descriptor, &byte, 1, (off_t)offset) == 1 &&
		    fsync(descriptor) == 0;
	if (descriptor >= 0)
		(void)close(descriptor);
	expect(corrupted, "table 0's block is corrupt on the only copy");
	/* a pipeline's workers wait on the host backend itself: the checked one keeps unlocked records */
	wait_base = &checked_backend;
	struct backend_operations watched = *wait_base;
	watched.staging_allocate = recorded_staging_allocate;
	watched.staging_free = recorded_staging_free;
	watched.staging_retire = recorded_staging_retire;
	watched.upload_wait = failing_upload_wait;
	struct fixture fixture;
	struct step_driver *driver = NULL;
	struct step_row_record rows[ROW_CAPACITY];
	struct step_row_result results[ROW_CAPACITY];
	fixture_build(&fixture, cards_open());
	const uint32_t held_before = recorded_count;
	teardown_failing = false;
	teardown_failures = 0;
	expect_status(step_driver_create(&watched, cards, fixture.pointers,
					 PROGRAMS, disk->engram, WAIT_LIMIT,
					 &driver),
		      STATUS_OK, "the driver with step I/O is made");
	held_rows(rows);
	capture_begin();
	expect(fault_arm("storage_flight_hold", HELD_TABLE_READ),
	       "the hold fault arms");
	expect(step_run_entry(driver, rows, ROW_CAPACITY, results) != STATUS_OK,
	       "the step fails on table 0's block, bad on every copy");
	expect(fault_disarm() >= HELD_TABLE_READ,
	       "the hold fault point is reached at table 1's read");
	teardown_failing = true;
	expect(fault_arm("engram_step_wait_short", 1),
	       "the short wait fault arms");
	step_driver_destroy(driver);
	expect(fault_disarm() != 0, "the short wait fault point is reached");
	teardown_failing = false;
	(void)capture_end();
	expect(engram_steps_live(disk->steps) == 1u,
	       "the destroy keeps the record whose read did not return");
	expect(teardown_failures != 0,
	       "a card's wait failed at teardown: the staging is taken as still read");
	expect(recorded_count == held_before + 1u,
	       "the staging the kept record writes into is neither freed nor retired");
	cards_close();
}

static uint64_t elapsed_since(const struct timespec *start)
{
	struct timespec now;
	(void)clock_gettime(CLOCK_MONOTONIC, &now);
	return (uint64_t)(now.tv_sec - start->tv_sec) * NANOSECONDS_PER_SECOND +
	       (uint64_t)now.tv_nsec - (uint64_t)start->tv_nsec;
}

/* Whether the block is a staging block the driver holds (allocated, not freed or retired). */
static bool recorded_holds(const unsigned char *block)
{
	for (uint32_t index = 0; index < recorded_count; index++)
		if (recorded[index] == block)
			return true;
	return false;
}

/* The pipeline's kept branch (step_pipeline_free): slot 0's record, taken through the checking-only hook, is issued
 * on a store of one copy with its first read held (table 0's) and never waited for, which the pipeline's own cards
 * never leave (they wait on every table a step issued). With the short wait fault the slot's destroy gives up, so the
 * record is kept, and the pipeline's staging, into which that read writes, must stay allocated: not freed (a teardown
 * with no card still reading, the driver's own destroy), and not retired (staging_is_read: a card's wait failed at
 * teardown, as step_driver_destroy passes it). Every other staging block of the driver is released. The later close
 * drains the held read and frees the kept record (the caller times it); the test frees the staging after it. */
static void pipeline_kept_case(struct disk_fixture *disk, bool staging_is_read)
{
	context_text =
		staging_is_read ?
			"a pipeline torn down while a slot's read never returned, its staging taken as still read" :
			"a pipeline torn down while a slot's read never returned";
	/* a pipeline's workers wait on the host backend itself: the checked one keeps unlocked records */
	struct backend_operations watched = test_host_memory_backend;
	watched.staging_allocate = recorded_staging_allocate;
	watched.staging_free = recorded_staging_free;
	watched.staging_retire = recorded_staging_retire;
	struct fixture fixture;
	struct step_driver *driver = NULL;
	struct step_row_record rows[ROW_CAPACITY];
	uint64_t row_ids[ROW_CAPACITY * ENGRAM_MAXIMUM_TABLES *
			 ENGRAM_MAXIMUM_ROWS_PER_TOKEN] = { 0 };
	fixture_build(&fixture, cards_open());
	const uint32_t held_before = recorded_count;
	expect_status(step_driver_create(&watched, cards, fixture.pointers,
					 PROGRAMS, disk->engram, WAIT_LIMIT,
					 &driver),
		      STATUS_OK, "the driver with step I/O is made");
	expect_status(step_pipeline_start(driver, PIPELINE_DEPTH), STATUS_OK,
		      "its pipeline starts");
	struct engram_step *slot_step = NULL;
	unsigned char *slot_block = NULL;
	expect_status(step_pipeline_slot_record(driver, PIPELINE_DEPTH,
						&slot_step, &slot_block),
		      STATUS_INVALID_ARGUMENT,
		      "the hook gives no slot past the pipeline's depth");
	expect(slot_step == NULL && slot_block == NULL,
	       "a refused hook leaves its outputs untouched");
	expect_status(step_pipeline_slot_record(driver, 0, &slot_step,
						&slot_block),
		      STATUS_OK, "the hook gives slot 0's record");
	expect(slot_step != NULL && recorded_holds(slot_block),
	       "slot 0 reads through step I/O into the pipeline's staging");
	held_rows(rows);
	expect_status(step_row_ids(driver, rows, ROW_CAPACITY, row_ids),
		      STATUS_OK, "the rows' n-gram row ids");
	const struct storage_flight *const flight =
		engram_steps_flight(disk->steps);
	capture_begin();
	expect(fault_arm("storage_flight_hold", 1), "the hold fault arms");
	expect_status(step_engram_issue(driver, slot_step, slot_block, row_ids,
					ROW_CAPACITY),
		      STATUS_OK, "slot 0's reads are issued");
	/* the one I/O thread takes table 0's read first, so once any read landed the held one is flying */
	struct timespec start;
	(void)clock_gettime(CLOCK_MONOTONIC, &start);
	const struct timespec poll_interval = {
		.tv_sec = 0, .tv_nsec = HOLD_SETTLE_POLL_NANOSECONDS
	};
	while (storage_flight_reads(flight, 0) > 1u &&
	       elapsed_since(&start) < HOLD_SETTLE_LIMIT_NANOSECONDS)
		(void)nanosleep(&poll_interval, NULL);
	expect(storage_flight_reads(flight, 0) == 1u,
	       "only the held read is left in flight");
	expect(fault_disarm() != 0, "the hold fault point is reached");
	expect(fault_arm("engram_step_wait_short", 1),
	       "the short wait fault arms");
	if (staging_is_read) {
		step_pipeline_stop(driver);
		step_pipeline_free(driver, true);
	} else {
		step_driver_destroy(driver);
	}
	expect(fault_disarm() != 0, "the short wait fault point is reached");
	expect(recorded_holds(slot_block),
	       "the pipeline's staging the kept record writes into is neither freed nor retired");
	if (staging_is_read)
		step_driver_destroy(driver);
	(void)capture_end();
	expect(engram_steps_live(disk->steps) == 1u,
	       "the destroy keeps the slot's record whose read did not return");
	expect(recorded_count == held_before + 1u,
	       "every other staging block of the driver is released");
	cards_close();
}
#endif

int main(void)
{
	(void)setvbuf(stdout, NULL, _IOLBF, 0);
	/* the fixture's checked backend, as step_test wires it (the drivers here start from it) */
	checked_backend = test_host_memory_backend;
	checked_backend.download_begin = checked_download_begin;
	checked_backend.upload_begin = checked_upload_begin;
	checked_backend.upload_wait = checked_upload_wait;
	checked_backend.launch_run = checked_launch_run;
	(void)test_steps; /* step_programs.h's own cases run in step_test */
	capture_descriptor = mkstemp(capture_path);
	error_stream_descriptor = dup(STDERR_FILENO);
	if (capture_descriptor < 0 || error_stream_descriptor < 0 ||
	    unlink(capture_path) != 0)
		return EXIT_FAILURE;
	expect_status(test_engram_pack_open(&engram_pack, &engram), STATUS_OK,
		      "the n-gram tables open");
	struct disk_fixture disk;
	memset(&disk, 0, sizeof(disk));
	const bool opened =
		disk_open(&disk, COPIES, "three",
			  engram_steps_read_limit_within(WAIT_LIMIT, COPIES));
	context_text = "the disk store";
	expect(opened, "three copies of the test pack serve step I/O");
	if (opened) {
		teardown_case(&disk, 0, "a driver without a pipeline");
		teardown_case(&disk, PIPELINE_DEPTH,
			      "a driver with a pipeline");
#if CHECKING_HOOKS_ENABLED
		pipeline_hold_case(&disk);
#endif
	}
	disk_close(&disk);
#if CHECKING_HOOKS_ENABLED
	struct disk_fixture kept;
	memset(&kept, 0, sizeof(kept));
	const bool kept_opened =
		disk_open(&kept, 1, "one", KEPT_READ_LIMIT_NANOSECONDS);
	expect(kept_opened, "one copy of the test pack serves step I/O");
	if (kept_opened)
		kept_staging_case(&kept);
	disk_close(&kept);
	/* a store of one copy each: a case's held read retires its only copy */
	static const bool staging_reads[] = { false, true };
	static const char *const store_names[] = { "pipeline_free",
						   "pipeline_retire" };
	for (uint32_t index = 0;
	     index < sizeof(staging_reads) / sizeof(staging_reads[0]);
	     index++) {
		struct disk_fixture piped;
		memset(&piped, 0, sizeof(piped));
		const bool piped_opened =
			disk_open(&piped, 1, store_names[index],
				  KEPT_READ_LIMIT_NANOSECONDS);
		expect(piped_opened,
		       "one copy of the test pack serves a pipeline's step I/O");
		if (piped_opened)
			pipeline_kept_case(&piped, staging_reads[index]);
		struct timespec start;
		(void)clock_gettime(CLOCK_MONOTONIC, &start);
		disk_close(&piped);
		expect(elapsed_since(&start) <
			       CLOSE_DRAIN_LIMITS *
					       KEPT_READ_LIMIT_NANOSECONDS +
				       CLOSE_SLACK_NANOSECONDS,
		       "the close drained the held read and freed the kept record within twice its deadline");
	}
	while (recorded_count !=
	       0) /* a kept staging, after the close joined the I/O threads */
		test_host_memory_backend.staging_free(
			recorded[--recorded_count]);
#endif
	engram_close(engram);
	free(engram_pack);
	(void)close(capture_descriptor);
	(void)close(error_stream_descriptor);
	(void)printf("step_engram_io_test: %s\n",
		     failures == 0 ? "passed" : "FAILED");
	return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
