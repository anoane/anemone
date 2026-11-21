/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * engram_step_stress.c: serving's step I/O (engram_step.h) at peak load over real pack copies, to hold its per-read
 * deadline to no retirement of a healthy ring.
 *
 *   engram_step_stress --copies=A,B,C [--seconds=N] [--decode_rows=8] [--prefill_threads=3] [--prefill_rows=1024]
 *                      [--wait_limit_ms=10000] [--io_threads=2] [--spin_us=200] [--pin=1] [--scan_threads=2]
 *
 * The store opens as serving opens it (engram_disk_sources_open, the startup digest scan, every table on disk, no
 * cache), and the flight as engram_steps_start starts it: (1 + STEP_PIPELINE_MOST) steps of ENGRAM_STEP_SLOTS
 * slots, the read deadline engram_steps_read_limit_within(wait limit, copies), threads pinned to the drives'
 * completion CPUs. Then, for the seconds given, one thread issues and waits decode steps of decode_rows rows (one a
 * session) while prefill_threads threads each issue and wait chunks of prefill_rows rows at once, back to back. Row
 * ids are uniform over each table, as the n-gram hashes make them, so every read is cold. Every block is checked
 * against its pack digest by the steps themselves.
 *
 * Output: one JSON line on stdout, with the step latencies, the flight's counts (reads answered, their ages from the
 * hand-off to the ring in power-of-two buckets, the oldest), the retirements, the probations and the margin, the
 * deadline over the oldest age. Exit 0 only when no step failed, no ring retired, no copy went on probation and the
 * margin is at least --margin_least (default 10). A measurement tool: not part of the engine.
 */
#include <errno.h>
#include <inttypes.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "engine/engram/engram.h"
#include "engine/engram/engram_disk.h"
#include "engine/formats/pack/pack.h"
#include "engine/runtime/step/step.h"
#include "engine/storage/storage_seam.h"
#include "engine/engram/engram_step.h"
#include "engine/storage/io/storage_flight.h"

#define NANOSECONDS_PER_SECOND ((uint64_t)1000000000u)
#define NANOSECONDS_PER_MILLISECOND ((uint64_t)1000000u)
#define NANOSECONDS_PER_MICROSECOND ((uint64_t)1000u)
#define DECIMAL_BASE 10
#define SECONDS_DEFAULT 120u
#define DECODE_ROWS_DEFAULT 8u /* C8 decode: one row a session */
#define PREFILL_THREADS_DEFAULT 3u
#define PREFILL_THREADS_MOST \
	STEP_PIPELINE_MOST /* with the decode step: every record of the flight */
#define PREFILL_ROWS_DEFAULT 1024u /* a prefill chunk */
#define WAIT_LIMIT_MS_DEFAULT \
	10000u /* commands.h's CARD_WAIT_LIMIT_NANOSECONDS: the cards' wait limit */
#define MARGIN_LEAST_DEFAULT 10u
#define LATENCIES_MOST \
	(1u            \
	 << 20u) /* steps a thread keeps the latency of (later ones are counted, not kept) */
#define COPIES_TEXT_BYTES 4096u
#define RANDOM_MULTIPLIER 6364136223846793005ull
#define RANDOM_INCREMENT 1442695040888963407ull
#define RANDOM_SHIFT 33u
#define SEED_STRIDE 7919u
#define QUANTILE_PARTS 1000u
#define MEDIAN_PART 500u
#define TAIL_PART 990u
#define FAR_TAIL_PART 999u

struct settings {
	const char *copies_text;
	uint32_t seconds, decode_rows, prefill_threads, prefill_rows,
		wait_limit_ms, io_threads, spin_us, pin, scan_threads,
		margin_least;
};

struct worker {
	struct engram_steps *steps;
	struct engram_step *step;
	const struct engram_description *description;
	uint32_t rows;
	uint64_t seed, until_ns;
	pthread_t thread;
	bool started;
	/* results */
	uint64_t steps_done, failures, *latencies, kept;
	uint64_t *row_ids;
	unsigned char *staging[ENGRAM_MAXIMUM_TABLES];
};

static uint64_t monotonic_nanoseconds(void)
{
	struct timespec now;
	return clock_gettime(CLOCK_MONOTONIC, &now) == 0 ?
		       (uint64_t)now.tv_sec * NANOSECONDS_PER_SECOND +
			       (uint64_t)now.tv_nsec :
		       0u;
}

static uint64_t random_next(uint64_t *state)
{
	*state = *state * RANDOM_MULTIPLIER + RANDOM_INCREMENT;
	return *state >> RANDOM_SHIFT;
}

static bool number_of(const char *argument, const char *name, uint32_t *value)
{
	const size_t length = strlen(name);
	if (strncmp(argument, name, length) != 0)
		return false;
	char *end = NULL;
	errno = 0;
	const unsigned long parsed =
		strtoul(argument + length, &end, DECIMAL_BASE);
	if (errno != 0 || end == argument + length || *end != '\0' ||
	    parsed > UINT32_MAX) {
		(void)fprintf(stderr, "engram_step_stress: bad %s\n", argument);
		exit(EXIT_FAILURE);
	}
	*value = (uint32_t)parsed;
	return true;
}

static void *worker_run(void *argument)
{
	struct worker *const worker = argument;
	const struct engram_description *const description =
		worker->description;
	const uint32_t tables = description->table_count,
		       per_token = description->rows_per_token;
	unsigned char *destinations[ENGRAM_MAXIMUM_TABLES] = { NULL };
	for (uint32_t table = 0; table < tables; table++)
		destinations[table] = worker->staging[table];
	uint64_t state = worker->seed;
	while (monotonic_nanoseconds() < worker->until_ns) {
		for (uint32_t row = 0; row < worker->rows; row++)
			for (uint32_t table = 0; table < tables; table++)
				for (uint32_t index = 0; index < per_token;
				     index++)
					worker->row_ids[((uint64_t)row * tables +
							 table) *
								per_token +
							index] =
						random_next(&state) %
						description->row_count[table];
		const uint64_t started = monotonic_nanoseconds();
		bool good = engram_step_issue(worker->step, worker->row_ids,
					      worker->rows,
					      destinations) == STATUS_OK;
		for (uint32_t table = 0; table < tables; table++)
			good = engram_step_wait(worker->step, table) ==
				       STATUS_OK &&
			       good;
		const uint64_t latency = monotonic_nanoseconds() - started;
		worker->failures += good ? 0u : 1u;
		if (worker->kept < LATENCIES_MOST)
			worker->latencies[worker->kept++] = latency;
		worker->steps_done++;
	}
	return NULL;
}

static int latency_order(const void *first, const void *second)
{
	const uint64_t left = *(const uint64_t *)first,
		       right = *(const uint64_t *)second;
	return left < right ? -1 : left > right ? 1 : 0;
}

/* One worker's step latencies in microseconds: median, 99 %, 99.9 % and the most, as JSON members. */
static void latencies_print(const char *name, struct worker *worker)
{
	qsort(worker->latencies, (size_t)worker->kept, sizeof(uint64_t),
	      latency_order);
	const uint64_t kept = worker->kept;
	const uint64_t median =
		kept != 0 ?
			worker->latencies[kept * MEDIAN_PART / QUANTILE_PARTS] :
			0u;
	const uint64_t tail =
		kept != 0 ?
			worker->latencies[kept * TAIL_PART / QUANTILE_PARTS] :
			0u;
	const uint64_t far =
		kept != 0 ?
			worker->latencies[kept * FAR_TAIL_PART / QUANTILE_PARTS] :
			0u;
	const uint64_t most = kept != 0 ? worker->latencies[kept - 1u] : 0u;
	(void)printf("\"%s\":{\"rows\":%" PRIu32 ",\"steps\":%" PRIu64
		     ",\"failures\":%" PRIu64 ",\"p50_us\":%" PRIu64
		     ",\"p99_us\":%" PRIu64 ",\"p999_us\":%" PRIu64
		     ",\"most_us\":%" PRIu64 "}",
		     name, worker->rows, worker->steps_done, worker->failures,
		     median / NANOSECONDS_PER_MICROSECOND,
		     tail / NANOSECONDS_PER_MICROSECOND,
		     far / NANOSECONDS_PER_MICROSECOND,
		     most / NANOSECONDS_PER_MICROSECOND);
}

static bool worker_prepare(struct worker *worker, struct engram_steps *steps,
			   const struct engram_description *description,
			   uint32_t rows, uint64_t seed)
{
	worker->steps = steps;
	worker->description = description;
	worker->rows = rows;
	worker->seed = seed;
	worker->latencies = calloc(LATENCIES_MOST, sizeof(uint64_t));
	worker->row_ids = calloc((size_t)rows * description->table_count *
					 description->rows_per_token,
				 sizeof(uint64_t));
	bool good = worker->latencies != NULL && worker->row_ids != NULL &&
		    engram_step_create(steps, &worker->step) == STATUS_OK;
	for (uint32_t table = 0; good && table < description->table_count;
	     table++) {
		worker->staging[table] =
			calloc((size_t)rows * description->rows_per_token,
			       ENGRAM_ROW_BYTES);
		good = worker->staging[table] != NULL;
	}
	return good;
}

static void worker_free(struct worker *worker)
{
	if (worker->step != NULL && !engram_step_destroy(worker->step))
		worker->failures++; /* kept: the close frees it */
	free(worker->latencies);
	free(worker->row_ids);
	for (uint32_t table = 0; table < ENGRAM_MAXIMUM_TABLES; table++)
		free(worker->staging[table]);
}

int main(int argument_count, char **arguments)
{
	struct settings settings = {
		.seconds = SECONDS_DEFAULT,
		.decode_rows = DECODE_ROWS_DEFAULT,
		.prefill_threads = PREFILL_THREADS_DEFAULT,
		.prefill_rows = PREFILL_ROWS_DEFAULT,
		.wait_limit_ms = WAIT_LIMIT_MS_DEFAULT,
		.io_threads = ENGRAM_STEP_IO_THREADS_DEFAULT,
		.spin_us = ENGRAM_STEP_SPIN_US_DEFAULT,
		.pin = 1,
		.scan_threads = ENGRAM_DISK_SCAN_THREADS_DEFAULT,
		.margin_least = MARGIN_LEAST_DEFAULT
	};
	for (int index = 1; index < argument_count; index++) {
		const char *const argument = arguments[index];
		if (strncmp(argument, "--copies=", strlen("--copies=")) == 0)
			settings.copies_text = argument + strlen("--copies=");
		else if (!number_of(argument,
				    "--seconds=", &settings.seconds) &&
			 !number_of(argument,
				    "--decode_rows=", &settings.decode_rows) &&
			 !number_of(argument, "--prefill_threads=",
				    &settings.prefill_threads) &&
			 !number_of(argument, "--prefill_rows=",
				    &settings.prefill_rows) &&
			 !number_of(argument, "--wait_limit_ms=",
				    &settings.wait_limit_ms) &&
			 !number_of(argument,
				    "--io_threads=", &settings.io_threads) &&
			 !number_of(argument,
				    "--spin_us=", &settings.spin_us) &&
			 !number_of(argument, "--pin=", &settings.pin) &&
			 !number_of(argument, "--scan_threads=",
				    &settings.scan_threads) &&
			 !number_of(argument, "--margin_least=",
				    &settings.margin_least)) {
			(void)fprintf(stderr,
				      "engram_step_stress: unknown %s\n",
				      argument);
			return EXIT_FAILURE;
		}
	}
	if (settings.copies_text == NULL || settings.decode_rows == 0 ||
	    settings.prefill_rows == 0 ||
	    settings.prefill_threads > PREFILL_THREADS_MOST ||
	    strlen(settings.copies_text) >= COPIES_TEXT_BYTES) {
		(void)fputs(
			"engram_step_stress: --copies=A,B,C and nonzero rows, "
			"at most 8 prefill threads\n",
			stderr);
		return EXIT_FAILURE;
	}
	static char text[COPIES_TEXT_BYTES];
	(void)snprintf(text, sizeof(text), "%s", settings.copies_text);
	const char *paths[STORAGE_MAXIMUM_COPIES];
	uint32_t copies = 0;
	for (char *next = text;
	     next != NULL && copies < STORAGE_MAXIMUM_COPIES;) {
		char *const comma = strchr(next, ',');
		if (comma != NULL)
			*comma = '\0';
		paths[copies++] = next;
		next = comma != NULL ? comma + 1 : NULL;
	}
	struct storage_reader *reader = NULL, *sources = NULL;
	struct pack_record *pack = NULL;
	struct engram_record *engram = NULL;
	struct engram_disk *disk = NULL;
	struct engram_steps *steps = NULL;
	const uint64_t opening = monotonic_nanoseconds();
	result_status status =
		storage_direct.reader_open(paths, copies, &reader);
	status = status == STATUS_OK ?
			 pack_open_entry(&storage_direct, reader, &pack) :
			 status;
	status = status == STATUS_OK ?
			 engram_open(&storage_direct, reader, pack, &engram) :
			 status;
	status = status == STATUS_OK ?
			 engram_disk_sources_open(paths, copies, pack, engram,
						  &sources) :
			 status;
	engram_disk_scan_threads(settings.scan_threads);
	status = status == STATUS_OK ?
			 engram_disk_open(engram_description_of(engram), pack,
					  &storage_direct, sources, 0, 0,
					  copies > 1u ? 1u : 0u, 0, &disk) :
			 status;
	const uint64_t opened = monotonic_nanoseconds();
	struct engram_steps_options options = {
		.threads_per_copy = settings.io_threads,
		.steps = 1u + STEP_PIPELINE_MOST,
		.slots_per_step = ENGRAM_STEP_SLOTS,
		.rows_most = settings.prefill_rows > settings.decode_rows ?
				     settings.prefill_rows :
				     settings.decode_rows,
		.spin_nanoseconds = (uint64_t)settings.spin_us *
				    NANOSECONDS_PER_MICROSECOND,
		.read_limit_nanoseconds = engram_steps_read_limit_within(
			(uint64_t)settings.wait_limit_ms *
				NANOSECONDS_PER_MILLISECOND,
			copies)
	};
	for (uint32_t copy = 0; copy < STORAGE_MAXIMUM_COPIES; copy++)
		for (uint32_t thread = 0;
		     thread < STORAGE_FLIGHT_THREADS_PER_COPY_MOST; thread++)
			options.cpus[copy][thread] =
				status == STATUS_OK && settings.pin != 0 &&
						copy < copies ?
					storage_flight_queue_cpu(
						engram_disk_reader(disk), copy,
						thread) :
					STORAGE_FLIGHT_CPU_FREE;
	status = status == STATUS_OK ?
			 engram_steps_open(disk, &options, &steps) :
			 status;
	if (status != STATUS_OK) {
		(void)fprintf(stderr, "engram_step_stress: open: %s\n",
			      status_text(status));
		return EXIT_FAILURE;
	}
	const struct engram_description *const description =
		engram_disk_description(disk);
	struct worker workers[1u + PREFILL_THREADS_MOST];
	memset(workers, 0, sizeof(workers));
	const uint32_t worker_count = 1u + settings.prefill_threads;
	bool prepared = description != NULL;
	for (uint32_t index = 0; prepared && index < worker_count; index++)
		prepared = worker_prepare(&workers[index], steps, description,
					  index == 0 ? settings.decode_rows :
						       settings.prefill_rows,
					  (uint64_t)(index + 1u) * SEED_STRIDE);
	const uint64_t until =
		monotonic_nanoseconds() +
		(uint64_t)settings.seconds * NANOSECONDS_PER_SECOND;
	for (uint32_t index = 0; prepared && index < worker_count; index++) {
		workers[index].until_ns = until;
		workers[index].started = pthread_create(&workers[index].thread,
							NULL, worker_run,
							&workers[index]) == 0;
		prepared = workers[index].started;
	}
	for (uint32_t index = 0; index < worker_count; index++)
		if (workers[index].started)
			(void)pthread_join(workers[index].thread, NULL);
	struct storage_flight_counts counts;
	storage_flight_counts(engram_steps_flight(steps), &counts);
	uint32_t retired = 0;
	for (uint32_t copy = 0; copy < copies; copy++)
		retired += storage_flight_copy_retired(
				   engram_steps_flight(steps), copy) ?
				   1u :
				   0u;
	const uint32_t probations = engram_steps_probations(steps);
	uint64_t failures = 0;
	(void)printf("{\"copies\":%" PRIu32 ",\"io_threads\":%" PRIu32
		     ",\"seconds\":%" PRIu32 ",\"open_seconds\":%.3f,"
		     "\"read_limit_us\":%" PRIu64 ",",
		     copies, settings.io_threads, settings.seconds,
		     (double)(opened - opening) /
			     (double)NANOSECONDS_PER_SECOND,
		     counts.read_limit_ns / NANOSECONDS_PER_MICROSECOND);
	latencies_print("decode", &workers[0]);
	failures += workers[0].failures;
	(void)printf(",\"prefill\":[");
	for (uint32_t index = 1; index < worker_count; index++) {
		(void)printf("%s{", index != 1 ? "," : "");
		latencies_print("chunks", &workers[index]);
		(void)printf("}");
		failures += workers[index].failures;
	}
	const uint64_t margin =
		counts.age_most_ns != 0 ?
			counts.read_limit_ns / counts.age_most_ns :
			0u;
	(void)printf("],\"answered\":%" PRIu64 ",\"age_most_us\":%" PRIu64
		     ",\"age_p99_us\":%" PRIu64 ",\"age_p999_us\":%" PRIu64
		     ",\"margin\":%" PRIu64 ",\"age_buckets_first_ns\":%llu,"
		     "\"age_buckets\":[",
		     counts.answered,
		     counts.age_most_ns / NANOSECONDS_PER_MICROSECOND,
		     storage_flight_age_quantile(&counts, QUANTILE_PARTS,
						 TAIL_PART) /
			     NANOSECONDS_PER_MICROSECOND,
		     storage_flight_age_quantile(&counts, QUANTILE_PARTS,
						 FAR_TAIL_PART) /
			     NANOSECONDS_PER_MICROSECOND,
		     margin, STORAGE_FLIGHT_AGE_FIRST_NANOSECONDS);
	for (uint32_t bucket = 0; bucket < STORAGE_FLIGHT_AGE_BUCKETS; bucket++)
		(void)printf("%s%" PRIu64, bucket != 0 ? "," : "",
			     counts.age_buckets[bucket]);
	const bool passed = prepared && failures == 0 &&
			    counts.deadline_retirements == 0 &&
			    counts.other_retirements == 0 && probations == 0 &&
			    retired == 0 && counts.answered != 0 &&
			    margin >= settings.margin_least;
	(void)printf("],\"deadline_retirements\":%" PRIu32
		     ",\"other_retirements\":%" PRIu32
		     ",\"reinstatements\":%" PRIu32 ",\"probations\":%" PRIu32
		     ",\"retired_at_end\":%" PRIu32
		     ",\"step_failures\":%" PRIu64 ",\"margin_least\":%" PRIu32
		     ",\"passed\":%s}\n",
		     counts.deadline_retirements, counts.other_retirements,
		     counts.reinstatements, probations, retired, failures,
		     settings.margin_least, passed ? "true" : "false");
	for (uint32_t index = 0; index < worker_count; index++)
		worker_free(&workers[index]);
	engram_close(engram);
	engram_steps_close(steps);
	engram_disk_close(disk);
	storage_direct.reader_close(sources);
	pack_close(pack);
	storage_direct.reader_close(reader);
	return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
