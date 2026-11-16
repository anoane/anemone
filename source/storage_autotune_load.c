/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * anemone autotune PACK --stage=load: the load stage (the owner's split, 9 Nov 2026), with its own parameters,
 * mechanism, state and profile section; it shares with the other stages only the replicas and their search.
 *
 * It tunes the TP host arena's read through the loader itself: tp_host_load_run over the copies serving would read
 * (the search, replicas_find_policy, and replicas_reader_open), on serving's TP_LOAD_HOST_THREADS_DEFAULT threads,
 * with the chunk digests on the reading threads, into an arena mapped afresh for every point as serving maps its own
 * (2 MiB pages), so every point pays the first touch. Its parameters are the host arena's per-copy cap (readers a
 * copy: tp_load_copies host_reads_most, which caps the host role's reads alone, 0 none, serving's default) and the
 * read size (the piece: whole pack blocks, at most TP_HOST_LOAD_CHUNK_BYTES_MOST). The copies split by stealing as
 * serving's default, with the cards' role marked claimed: no card load runs beside, so the host arena may read every
 * copy. The section records the host threads and the split it measured with; serving applies it only with the same.
 *
 * A point reads a window of the HOST_ONLY extents (external_host_extents marks the others), whole extents from a
 * start that rotates through the arena by one window a point, so every point reads another region. The grid crosses
 * readers_options with piece_mib_options at --window_gib (default LOAD_WINDOW_GIB_DEFAULT); then the fastest arm
 * other than serving's default (no cap, TP_HOST_LOAD_CHUNK_BYTES) runs A D D A against the default at
 * LOAD_ABBA_WINDOW_FACTOR windows. The higher mean wins even inside the noise (the owner's rule, 29 Oct 2026: a
 * faster option with the same output is the default); only means equal at the reported precision go to the arm with
 * fewer readers (no cap counts as the most), then the smaller piece. The spread (the larger arm's range over its mean)
 * and the margin are recorded beside the choice; when the default wins, the section holds the default's values. Every
 * point records the drives' temperatures before and after it.
 *
 * Progress is kept in <profile>.load-state after every point; a lease (--seconds) or --points_per_run that ends
 * first exits AUTOTUNE_RETRY_EXIT and the same command resumes. At the end the profile's "load" member is written
 * (storage_load_section_write: every other byte of the profile stays), enabled 0 and qualified "candidate"; serving
 * applies it only once `--stage=load --qualify` has made it enabled, and only to the pack and copies it names.
 */
#include "storage_autotune.h"
#include "anemone/replicas.h"
#include "anemone/storage_exl3.h"
#include "anemone/tp_load.h"
#include "storage_prepare.h"
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <math.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define LOAD_STATE_SCHEMA \
	3u /* 2: with the A D D A's warm-up; 3: its passes, of the A D D A point's extents */
/* R26b (15 Nov 2026): the first two A D D A points of 195.32 GiB cost cpu 443.8 and 60.7 s, the next two 37.0 and
 * 32.7 s, so one pass of an arena that size does not reach the steady state */
#define LOAD_WARMUP_PASSES 2u
#define LOAD_WINDOW_GIB_DEFAULT 64u /* a grid point's volume */
#define LOAD_WINDOW_GIB_MOST 1024u
#define LOAD_ABBA_WINDOW_FACTOR \
	2u /* an A B B A point reads twice a grid point's window */
#define LOAD_ABBA_POINTS AUTOTUNE_LOAD_ABBA_POINTS
#define LOAD_ABBA_ARMS \
	2u /* the fastest other arm, the challenger, and the default */
#define LOAD_DRIVES_MOST 16u
#define LOAD_DRIVE_NAME_BYTES 32u
#define LOAD_TEMPERATURE_TEXT_BYTES 32u
#define LOAD_FLAG_TEXT_BYTES 128u /* the selection line's spread flag */
#define GIB_BYTES ((uint64_t)1u << 30u)
#define MIB_BYTES ((uint64_t)1u << 20u)
#define NANOSECONDS_PER_SECOND 1000000000ull
#define NANOSECONDS_PER_MICROSECOND 1000ull
#define DECIMAL_RADIX 10
#define NVME_CLASS_DIRECTORY "/sys/class/nvme"
#define NVME_NAME_PREFIX "nvme"
#define HWMON_NAME_PREFIX "hwmon"
#define TEMPERATURE_FILE \
	"temp1_input" /* the composite temperature, in millidegrees Celsius */
#define STATE_SUFFIX ".load-state"

/* readers a copy (0: no cap, serving's default) and pieces in MiB (16: serving's default, TP_HOST_LOAD_CHUNK_BYTES) */
static const uint32_t readers_options[] = { 0, 1, 2, 3, 4, 6, 8 };
static const uint32_t piece_mib_options[] = { 4, 8, 16, 32, 64 };
#define READERS_OPTIONS (sizeof(readers_options) / sizeof(readers_options[0]))
#define PIECE_OPTIONS (sizeof(piece_mib_options) / sizeof(piece_mib_options[0]))
#define LOAD_GRID_POINTS ((uint32_t)(READERS_OPTIONS * PIECE_OPTIONS))
#define LOAD_POINTS_MOST (LOAD_GRID_POINTS + LOAD_ABBA_POINTS)
/* the A D D A order (storage_autotune.h): the challenger at its first and last point, the default between */
static const uint32_t abba_is_default[LOAD_ABBA_POINTS] = { 0, 1, 1, 0 };

struct load_drives {
	uint32_t count;
	char names[LOAD_DRIVES_MOST][LOAD_DRIVE_NAME_BYTES];
	int64_t millicelsius[LOAD_DRIVES_MOST];
};
struct load_point {
	uint32_t readers_per_copy, piece_mib, first_extent, extents;
	uint64_t bytes, nanoseconds, cpu_nanoseconds, read_calls, copy_retries;
	uint64_t copy_bytes[STORAGE_MAXIMUM_COPIES];
	struct load_drives before, after;
};
/* The unmeasured arenas touched before the first A D D A point (warmup_touch): bytes a pass, times over all. */
struct load_warmup {
	uint64_t bytes, nanoseconds, cpu_nanoseconds;
	uint32_t passes;
};
/* The stage's resumable progress: the points measured, for these copies, this pack and this window. */
struct load_state {
	uint32_t schema, copy_count, window_gib, point_count;
	char pack_identity[STORAGE_PROFILE_KEY_BYTES];
	char keys[STORAGE_MAXIMUM_COPIES][STORAGE_PROFILE_KEY_BYTES];
	uint64_t file_byte_count;
	struct load_warmup warmup;
	struct load_point points[LOAD_POINTS_MOST];
};
struct load_context {
	const struct storage_operations_record *storage;
	struct storage_reader *reader;
	struct pack_record *pack;
	struct tp_host_load_plan plan;
	uint64_t *offsets;
	uint8_t *external;
	uint32_t *host_extents; /* the HOST_ONLY extents' indices, in pack order */
	uint32_t host_extent_count, copy_count;
	const char *copy_paths[STORAGE_MAXIMUM_COPIES];
	char replica_paths[REPLICAS_MOST][REPLICA_PATH_CAPACITY];
	char filesystems[STORAGE_MAXIMUM_COPIES][LOCAL_DEVICE_FILESYSTEM_BYTES];
};
struct load_selection {
	uint32_t challenger, fallback,
		winner; /* grid indices; fallback: the default */
	double abba_rates[LOAD_ABBA_POINTS];
	struct autotune_load_verdict verdict;
};

static volatile sig_atomic_t load_stop;
static void load_signal(int signal_number)
{
	(void)signal_number;
	load_stop = 1;
}

static uint64_t now_nanoseconds(void)
{
	struct timespec time;
	(void)clock_gettime(CLOCK_MONOTONIC, &time);
	return (uint64_t)time.tv_sec * NANOSECONDS_PER_SECOND +
	       (uint64_t)time.tv_nsec;
}

static uint64_t cpu_nanoseconds(void)
{
	struct rusage usage;
	if (getrusage(RUSAGE_SELF, &usage) != 0)
		return 0;
	return ((uint64_t)usage.ru_utime.tv_sec +
		(uint64_t)usage.ru_stime.tv_sec) *
		       NANOSECONDS_PER_SECOND +
	       ((uint64_t)usage.ru_utime.tv_usec +
		(uint64_t)usage.ru_stime.tv_usec) *
		       NANOSECONDS_PER_MICROSECOND;
}

/* A controller's composite temperature file: its first hwmon's. */
static bool sensor_find(const char *controller, char *file, size_t capacity)
{
	DIR *const sensors = opendir(controller);
	if (sensors == NULL)
		return false;
	bool found = false;
	const struct dirent *sensor;
	while (!found && (sensor = readdir(sensors)) != NULL)
		if (strncmp(sensor->d_name, HWMON_NAME_PREFIX,
			    strlen(HWMON_NAME_PREFIX)) == 0) {
			const int length = snprintf(file, capacity,
						    "%s/%s/" TEMPERATURE_FILE,
						    controller, sensor->d_name);
			found = length > 0 && (size_t)length < capacity;
		}
	(void)closedir(sensors);
	return found;
}

/* Every NVMe controller's composite temperature, as S2's sampler reads them; none when sysfs has none. */
static void drives_read(struct load_drives *drives)
{
	memset(drives, 0, sizeof(*drives));
	DIR *const controllers = opendir(NVME_CLASS_DIRECTORY);
	if (controllers == NULL)
		return;
	const struct dirent *entry;
	while (drives->count < LOAD_DRIVES_MOST &&
	       (entry = readdir(controllers)) != NULL) {
		char controller[PATH_MAX], file[PATH_MAX],
			text[LOAD_TEMPERATURE_TEXT_BYTES];
		const int length = snprintf(controller, sizeof(controller),
					    "%s/%s", NVME_CLASS_DIRECTORY,
					    entry->d_name);
		if (strncmp(entry->d_name, NVME_NAME_PREFIX,
			    strlen(NVME_NAME_PREFIX)) != 0 ||
		    strlen(entry->d_name) >= LOAD_DRIVE_NAME_BYTES ||
		    length <= 0 || (size_t)length >= sizeof(controller) ||
		    !sensor_find(controller, file, sizeof(file)))
			continue;
		const int descriptor = open(file, O_RDONLY | O_CLOEXEC);
		const ssize_t count =
			descriptor >= 0 ?
				read(descriptor, text, sizeof(text) - 1u) :
				-1;
		if (descriptor >= 0)
			(void)close(descriptor);
		if (count <= 0)
			continue;
		text[count] = '\0';
		char *end = NULL;
		const long long value = strtoll(text, &end, DECIMAL_RADIX);
		if (end == text)
			continue;
		(void)snprintf(drives->names[drives->count],
			       sizeof(drives->names[drives->count]), "%.*s",
			       (int)(LOAD_DRIVE_NAME_BYTES - 1u),
			       entry->d_name);
		drives->millicelsius[drives->count++] = value;
	}
	(void)closedir(controllers);
}

/* The arm of point index: the grid's piece-major order, then the A B B A arms. */
static void arm_of(uint32_t index, const struct load_selection *selection,
		   const struct load_state *state, uint32_t *readers,
		   uint32_t *piece_mib)
{
	const uint32_t grid = index < LOAD_GRID_POINTS ? index :
			      abba_is_default[index - LOAD_GRID_POINTS] != 0 ?
							 selection->fallback :
							 selection->challenger;
	*readers = index < LOAD_GRID_POINTS ?
			   readers_options[grid % READERS_OPTIONS] :
			   state->points[grid].readers_per_copy;
	*piece_mib = index < LOAD_GRID_POINTS ?
			     piece_mib_options[grid / READERS_OPTIONS] :
			     state->points[grid].piece_mib;
}

static double point_rate(const struct load_point *point)
{
	return point->nanoseconds != 0 ?
		       (double)point->bytes / (double)point->nanoseconds :
		       0.0; /* bytes a nanosecond: GB/s */
}

/* Fewer readers first (no cap counts as the most), then the smaller piece. */
static bool arm_simpler(const struct load_point *first,
			const struct load_point *second)
{
	const uint32_t first_readers = first->readers_per_copy != 0 ?
					       first->readers_per_copy :
					       UINT32_MAX;
	const uint32_t second_readers = second->readers_per_copy != 0 ?
						second->readers_per_copy :
						UINT32_MAX;
	return first_readers != second_readers ?
		       first_readers < second_readers :
		       first->piece_mib < second->piece_mib;
}

/* Whether first ranks ahead of second: the faster, and at an equal rate the simpler. */
static bool arm_ahead(const struct load_point *first,
		      const struct load_point *second)
{
	const double first_rate = point_rate(first);
	const double second_rate = point_rate(second);
	return first_rate != second_rate ? first_rate > second_rate :
					   arm_simpler(first, second);
}

/* The grid's default arm (serving's: no cap, TP_HOST_LOAD_CHUNK_BYTES) and its fastest other arm. */
static void grid_rank(const struct load_state *state,
		      struct load_selection *selection)
{
	const struct load_point *const points = state->points;
	uint32_t challenger = LOAD_GRID_POINTS, fallback = LOAD_GRID_POINTS;
	for (uint32_t index = 0; index < LOAD_GRID_POINTS; index++)
		if (points[index].readers_per_copy ==
			    TP_LOAD_READS_PER_COPY_DEFAULT &&
		    (uint64_t)points[index].piece_mib * MIB_BYTES ==
			    TP_HOST_LOAD_CHUNK_BYTES)
			fallback = index;
		else if (challenger == LOAD_GRID_POINTS ||
			 arm_ahead(&points[index], &points[challenger]))
			challenger = index;
	selection->challenger = challenger;
	selection->fallback = fallback;
}

void autotune_load_verdict(const double rates[AUTOTUNE_LOAD_ABBA_POINTS],
			   bool default_simpler,
			   struct autotune_load_verdict *verdict)
{
	double sums[LOAD_ABBA_ARMS] = { 0.0 }, lows[LOAD_ABBA_ARMS] = { 0.0 },
	       highs[LOAD_ABBA_ARMS] = { 0.0 };
	uint32_t counts[LOAD_ABBA_ARMS] = { 0 };
	for (uint32_t index = 0; index < LOAD_ABBA_POINTS; index++) {
		const uint32_t arm = abba_is_default[index];
		const double rate = rates[index];
		lows[arm] = counts[arm] == 0 || rate < lows[arm] ? rate :
								   lows[arm];
		highs[arm] = counts[arm] == 0 || rate > highs[arm] ? rate :
								     highs[arm];
		sums[arm] += rate;
		counts[arm]++;
	}
	double means[LOAD_ABBA_ARMS];
	verdict->spread = 0.0;
	for (uint32_t arm = 0; arm < LOAD_ABBA_ARMS; arm++) {
		means[arm] = sums[arm] / (double)counts[arm];
		const double range =
			means[arm] > 0.0 ?
				(highs[arm] - lows[arm]) / means[arm] :
				0.0;
		verdict->spread = range > verdict->spread ? range :
							    verdict->spread;
	}
	const double challenger = means[0], fallback = means[1];
	verdict->margin = fallback > 0.0 ? (challenger - fallback) / fallback :
					   0.0;
	verdict->exact_tie = llround(challenger * AUTOTUNE_LOAD_RATE_SCALE) ==
			     llround(fallback * AUTOTUNE_LOAD_RATE_SCALE);
	verdict->default_wins = verdict->exact_tie ? default_simpler :
						     fallback > challenger;
	verdict->spread_over_bound = verdict->spread >
				     AUTOTUNE_LOAD_SPREAD_BOUND;
}

/* The A D D A verdict over the four last points, and the winner it gives. */
static void abba_select(const struct load_state *state,
			struct load_selection *selection)
{
	for (uint32_t index = 0; index < LOAD_ABBA_POINTS; index++)
		selection->abba_rates[index] =
			point_rate(&state->points[LOAD_GRID_POINTS + index]);
	autotune_load_verdict(
		selection->abba_rates,
		arm_simpler(&state->points[selection->fallback],
			    &state->points[selection->challenger]),
		&selection->verdict);
	selection->winner = selection->verdict.default_wins ?
				    selection->fallback :
				    selection->challenger;
}

/* The window of a point: whole HOST_ONLY extents from the one holding the rotated start, in pack order and around,
 * until window_bytes are read or every extent is; every other HOST_ONLY extent is marked external. */
static uint64_t window_mark(struct load_context *context, uint32_t rotation,
			    uint64_t window_bytes, struct load_point *point)
{
	const struct pack_extent *const extents =
		pack_extents_of(context->pack);
	const uint64_t host_bytes = context->plan.host_byte_count;
	const uint64_t start = (uint64_t)rotation * window_bytes % host_bytes;
	uint32_t first = 0;
	while (first + 1u < context->host_extent_count &&
	       context->offsets[context->host_extents[first + 1u]] <= start)
		first++;
	for (uint32_t index = 0; index < context->host_extent_count; index++)
		context->external[context->host_extents[index]] = 1;
	uint64_t bytes = 0;
	uint32_t taken = 0;
	while (taken < context->host_extent_count && bytes < window_bytes) {
		const uint32_t extent =
			context->host_extents[(first + taken) %
					      context->host_extent_count];
		context->external[extent] = 0;
		bytes += extents[extent].byte_count;
		taken++;
	}
	point->first_extent = context->host_extents[first];
	point->extents = taken;
	return bytes;
}

/* Before the first A D D A point a run measures: LOAD_WARMUP_PASSES unmeasured arenas, each mapped as that point
 * maps its own (the whole host layout) and touched page by page over exactly the extents it reads (window_mark at
 * its index), so the point is not the first touch of an arena that size (R26, 15 Nov 2026: A1 read 8.865 GB/s at cpu
 * 350.7 s, D, D and A2 34.6 to 47.6 GB/s at cpu 32 to 60 s; R26b: a warm-up of 2 x --window_gib, 128 GiB, before A
 * points of 195.32 GiB left A1 at 7.07 GB/s, cpu 443.8 s). The bytes a pass equal the point's; they and the times are
 * recorded beside the points, never ranked. */
MUST_CHECK static result_status warmup_touch(struct load_context *context,
					     uint32_t index,
					     uint64_t window_bytes,
					     struct load_warmup *warmup)
{
	struct load_point point = { 0 };
	const uint64_t bytes =
		window_mark(context, index, window_bytes, &point);
	const struct pack_extent *const extents =
		pack_extents_of(context->pack);
	const long page = sysconf(_SC_PAGESIZE);
	*warmup = (struct load_warmup){ .bytes = bytes };
	result_status status = page > 0 ? STATUS_OK : STATUS_UNSUPPORTED;
	for (uint32_t pass = 0;
	     status == STATUS_OK && pass < LOAD_WARMUP_PASSES; pass++) {
		struct tp_host_arena arena;
		status = tp_host_arena_allocate(context->plan.host_byte_count,
						true, &arena);
		if (status != STATUS_OK)
			break;
		(void)madvise(arena.bytes, arena.byte_count, MADV_HUGEPAGE);
		unsigned char *const touched = arena.bytes;
		const uint64_t cpu_begin = cpu_nanoseconds();
		const uint64_t begin = now_nanoseconds();
		for (uint32_t host = 0; host < context->host_extent_count;
		     host++) {
			const uint32_t extent = context->host_extents[host];
			for (uint64_t offset = 0;
			     context->external[extent] == 0 &&
			     offset < extents[extent].byte_count;
			     offset += (uint64_t)page)
				touched[context->offsets[extent] + offset] = 1u;
		}
		warmup->nanoseconds += now_nanoseconds() - begin;
		warmup->cpu_nanoseconds += cpu_nanoseconds() - cpu_begin;
		warmup->passes++;
		/* munmap is an opaque call that takes the arena: the touches before it are kept */
		tp_host_arena_release(&arena);
	}
	if (status != STATUS_OK)
		return status;
	(void)fprintf(stderr,
		      "storage: load warm-up before the A D D A: %" PRIu32
		      " passes of %.2f GiB (the extents of point %" PRIu32
		      ") touched in %.3f s, cpu %.3f s (not measured)\n",
		      warmup->passes, (double)warmup->bytes / (double)GIB_BYTES,
		      index + 1u,
		      (double)warmup->nanoseconds /
			      (double)NANOSECONDS_PER_SECOND,
		      (double)warmup->cpu_nanoseconds /
			      (double)NANOSECONDS_PER_SECOND);
	return STATUS_OK;
}

/* One point: the window, a fresh arena, the copies' cap, and the loader's own host run, timed with its CPU time. */
MUST_CHECK static result_status point_measure(struct load_context *context,
					      uint32_t index,
					      uint64_t window_bytes,
					      struct load_point *point)
{
	const uint64_t host_bytes = context->plan.host_byte_count;
	point->bytes = window_mark(context, index, window_bytes, point);
	/* serving's TP host arena with external tables: unaccounted, as most of it is never touched here */
	struct tp_host_arena arena;
	result_status status = tp_host_arena_allocate(host_bytes, true, &arena);
	if (status != STATUS_OK)
		return status;
	/* as serving: without 2 MiB pages the arena reads through small ones */
	(void)madvise(arena.bytes, arena.byte_count, MADV_HUGEPAGE);
	struct tp_load_copies copies;
	status = tp_load_copies_init(&copies, 0);
	if (status != STATUS_OK) {
		tp_host_arena_release(&arena);
		return status;
	}
	/* the host arena's own cap, as serving applies the section's readers: alone here, both caps are the same */
	copies.host_reads_most = point->readers_per_copy;
	copies.split = TP_LOAD_SPLIT_STEAL;
	copies.role_claimed[TP_LOAD_ROLE_CARDS] = 1;
	const struct tp_host_load_request request = {
		.plan = &context->plan,
		.external_host_extents = context->external,
		.storage = context->storage,
		.reader = context->reader,
		.host_arena = arena.bytes,
		.host_arena_byte_count = host_bytes,
		.worker_count = TP_LOAD_HOST_THREADS_DEFAULT,
		.copies = &copies,
		.chunk_bytes = (uint64_t)point->piece_mib * MIB_BYTES
	};
	struct tp_host_load_result result = { 0 };
	drives_read(&point->before);
	const uint64_t cpu_begin = cpu_nanoseconds();
	const uint64_t begin = now_nanoseconds();
	status = tp_host_load_run(&request, &result);
	point->nanoseconds = now_nanoseconds() - begin;
	point->cpu_nanoseconds = cpu_nanoseconds() - cpu_begin;
	drives_read(&point->after);
	point->read_calls = result.read_calls;
	point->copy_retries = result.copy_retries;
	for (uint32_t copy = 0; copy < context->copy_count; copy++)
		point->copy_bytes[copy] = copies.delivered_bytes[copy];
	tp_load_copies_destroy(&copies);
	tp_host_arena_release(&arena);
	if (status == STATUS_OK && result.extents_verified != point->extents)
		status = STATUS_CORRUPT_PACK;
	return status;
}

static bool state_identity(const struct load_state *first,
			   const struct load_state *second)
{
	return first->schema == second->schema &&
	       first->copy_count == second->copy_count &&
	       first->window_gib == second->window_gib &&
	       first->file_byte_count == second->file_byte_count &&
	       strcmp(first->pack_identity, second->pack_identity) == 0 &&
	       memcmp(first->keys, second->keys, sizeof(first->keys)) == 0;
}

/* The prior progress when it is this user's, whole, and of the same copies, pack and window. */
static void state_resume(const char *path, struct load_state *state)
{
	struct load_state *const prior = malloc(sizeof(*prior));
	const int descriptor =
		open(path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
	struct stat file;
	if (prior != NULL && descriptor >= 0 && fstat(descriptor, &file) == 0 &&
	    S_ISREG(file.st_mode) && file.st_uid == geteuid() &&
	    (file.st_mode & (S_IWGRP | S_IWOTH)) == 0 &&
	    file.st_size == (off_t)sizeof(*prior) &&
	    read(descriptor, prior, sizeof(*prior)) ==
		    (ssize_t)sizeof(*prior) &&
	    state_identity(prior, state) &&
	    prior->point_count <= LOAD_POINTS_MOST) {
		bool same_arms = true;
		for (uint32_t index = 0;
		     same_arms && index < prior->point_count &&
		     index < LOAD_GRID_POINTS;
		     index++) {
			uint32_t readers = 0, piece_mib = 0;
			arm_of(index, NULL, prior, &readers, &piece_mib);
			same_arms = prior->points[index].readers_per_copy ==
					    readers &&
				    prior->points[index].piece_mib == piece_mib;
		}
		if (same_arms)
			*state = *prior;
	}
	free(prior);
	if (descriptor >= 0)
		(void)close(descriptor);
}

MUST_CHECK static result_status state_write(const char *path,
					    const struct load_state *state)
{
	struct prepare_file file;
	result_status status = prepare_file_open(path, &file);
	if (status != STATUS_OK)
		return status;
	if (fwrite(state, sizeof(*state), 1, file.stream) != 1u) {
		prepare_file_abort(&file);
		return STATUS_INPUT_OUTPUT_ERROR;
	}
	return prepare_file_commit(&file);
}

static void drives_json(FILE *output, const char *name,
			const struct load_drives *drives)
{
	(void)fprintf(output, ",\"%s\":{", name);
	for (uint32_t index = 0; index < drives->count; index++) {
		(void)fputs(index == 0 ? "" : ",", output);
		prepare_json_text(output, drives->names[index]);
		(void)fprintf(output, ":%" PRId64, drives->millicelsius[index]);
	}
	(void)fputc('}', output);
}

static void arm_json(FILE *output, const char *name,
		     const struct load_point *point)
{
	(void)fprintf(output,
		      "\"%s\":{\"readers_per_copy\":%" PRIu32
		      ",\"piece_bytes\":%" PRIu64 "}",
		      name, point->readers_per_copy,
		      (uint64_t)point->piece_mib * MIB_BYTES);
}

/* The profile's "load" member: a candidate, enabled 0, with its identity, every point and the selection. */
static void section_json(FILE *output, const struct load_context *context,
			 const struct load_state *state,
			 const struct load_selection *selection)
{
	const struct load_point *const winner =
		&state->points[selection->winner];
	(void)fprintf(
		output,
		"{\"schema\":%u,\"enabled\":0,\"qualification\":\"candidate\","
		"\"created_unix\":%lld,\"pack\":{\"header_digest\":\"%s\","
		"\"file_byte_count\":%" PRIu64 "},\"copies\":[",
		STORAGE_LOAD_SECTION_SCHEMA, (long long)time(NULL),
		state->pack_identity, state->file_byte_count);
	for (uint32_t copy = 0; copy < state->copy_count; copy++) {
		(void)fprintf(output, "%s{\"key\":\"%s\",\"path\":",
			      copy == 0 ? "" : ",", state->keys[copy]);
		prepare_json_text(output, context->copy_paths[copy]);
		(void)fputs(",\"filesystem\":", output);
		prepare_json_text(output, context->filesystems[copy]);
		(void)fputc('}', output);
	}
	(void)fprintf(
		output,
		"],\"pp\":{\"measured\":0,\"note\":\"the PP loader's reads are measured by real loads on "
		"the cards, not by this stage: its defaults hold\"},"
		"\"tp\":{\"readers_per_copy\":%" PRIu32
		",\"piece_bytes\":%" PRIu64 ",\"host_threads\":%u,"
		"\"split\":\"%s\",\"window_bytes\":%" PRIu64
		",\"abba_window_bytes\":%" PRIu64 "},\"measurements\":[",
		winner->readers_per_copy,
		(uint64_t)winner->piece_mib * MIB_BYTES,
		TP_LOAD_HOST_THREADS_DEFAULT,
		tp_load_split_names[TP_LOAD_SPLIT_STEAL],
		(uint64_t)state->window_gib * GIB_BYTES,
		(uint64_t)state->window_gib * GIB_BYTES *
			LOAD_ABBA_WINDOW_FACTOR);
	for (uint32_t index = 0; index < state->point_count; index++) {
		const struct load_point *const point = &state->points[index];
		(void)fprintf(
			output,
			"%s{\"phase\":\"%s\",\"readers_per_copy\":%" PRIu32
			",\"piece_bytes\":%" PRIu64 ",\"bytes\":%" PRIu64
			",\"nanoseconds\":%" PRIu64
			",\"gb_per_second\":%.3f,\"cpu_nanoseconds\":%" PRIu64
			",\"first_extent\":%" PRIu32 ",\"extents\":%" PRIu32
			",\"read_calls\":%" PRIu64 ",\"copy_retries\":%" PRIu64
			",\"copy_bytes\":[",
			index == 0 ? "" : ",",
			index < LOAD_GRID_POINTS ? "grid" : "abba",
			point->readers_per_copy,
			(uint64_t)point->piece_mib * MIB_BYTES, point->bytes,
			point->nanoseconds, point_rate(point),
			point->cpu_nanoseconds, point->first_extent,
			point->extents, point->read_calls, point->copy_retries);
		for (uint32_t copy = 0; copy < state->copy_count; copy++)
			(void)fprintf(output, "%s%" PRIu64,
				      copy == 0 ? "" : ",",
				      point->copy_bytes[copy]);
		(void)fputc(']', output);
		drives_json(output, "millicelsius_before", &point->before);
		drives_json(output, "millicelsius_after", &point->after);
		(void)fputc('}', output);
	}
	(void)fprintf(output,
		      "],\"warmup\":{\"bytes\":%" PRIu64
		      ",\"nanoseconds\":%" PRIu64
		      ",\"cpu_nanoseconds\":%" PRIu64 ",\"passes\":%" PRIu32
		      ",\"measured\":0},\"selection\":{",
		      state->warmup.bytes, state->warmup.nanoseconds,
		      state->warmup.cpu_nanoseconds, state->warmup.passes);
	arm_json(output, "winner", winner);
	(void)fputc(',', output);
	arm_json(output, "challenger", &state->points[selection->challenger]);
	(void)fputc(',', output);
	arm_json(output, "default", &state->points[selection->fallback]);
	(void)fputs(",\"adda_gb_per_second\":[", output);
	for (uint32_t index = 0; index < LOAD_ABBA_POINTS; index++)
		(void)fprintf(output, "%s%.3f", index == 0 ? "" : ",",
			      selection->abba_rates[index]);
	(void)fprintf(
		output,
		"],\"spread\":%.4f,\"spread_bound\":%.4f,\"spread_over_bound\":%u,\"margin\":%.4f,"
		"\"exact_tie\":%u,\"default_holds\":%u}}",
		selection->verdict.spread, AUTOTUNE_LOAD_SPREAD_BOUND,
		selection->verdict.spread_over_bound ? 1u : 0u,
		selection->verdict.margin,
		selection->verdict.exact_tie ? 1u : 0u,
		selection->verdict.default_wins ? 1u : 0u);
}

/* The copies serving would read: the source, then the receipted replicas that open beside it (one line names each
 * replica left out); their device keys and filesystems; the pack; the host plan and its HOST_ONLY extents. */
MUST_CHECK static result_status context_open(struct load_context *context,
					     const char *source, bool external,
					     struct load_state *state)
{
	struct pack_header header = { 0 };
	uint64_t source_bytes = 0;
	context->storage = storage_for_path(source);
	if (!storage_pack_header_read(source, &header, &source_bytes))
		return STATUS_INPUT_OUTPUT_ERROR;
	const uint32_t found = replicas_find_policy(source, &header,
						    context->replica_paths,
						    REPLICAS_MOST, external);
	/* an EXL3 map reads as serving's --map_reads auto (commands.c): direct descriptors while the copies open, then
	 * DROPPED for a lone map (a native pack's storage has no read mode) */
	storage_exl3_read_mode(EXL3_READS_DIRECT_DROPPED);
	const uint32_t kept = replicas_reader_open(
		context->storage, source,
		(const char (*)[REPLICA_PATH_CAPACITY])context->replica_paths,
		found, context->copy_paths, &context->reader, stderr);
	context->copy_count = kept + 1u;
	context->copy_paths[0] = source;
	result_status status =
		kept != 0 ? STATUS_OK :
			    context->storage->reader_open(context->copy_paths,
							  1, &context->reader);
	if (context->copy_count == 1u)
		storage_exl3_read_mode(EXL3_READS_DROPPED);
	for (uint32_t copy = 0;
	     status == STATUS_OK && copy < context->copy_count; copy++) {
		struct local_device_description device = { 0 };
		/* keyed by the drive of its bytes (an EXL3 map's data drive), as the section's matcher keys it */
		char bytes_path[EXL3_MAP_PATH_BYTES];
		const char *const keyed = storage_device_path(
			context->copy_paths[copy], bytes_path);
		status = storage_device_key_for(keyed, external,
						state->keys[copy], &device);
		(void)snprintf(context->filesystems[copy],
			       sizeof(context->filesystems[copy]), "%s",
			       device.filesystem);
		/* a key refused: by the copy's medium (serving's rule), else for the drive's missing identity */
		const char *const medium =
			status != STATUS_OK &&
					local_device_describe(keyed, &device) ==
						STATUS_OK ?
				local_device_media_refusal(&device, external) :
				NULL;
		if (status != STATUS_OK && medium != NULL)
			(void)fprintf(
				stderr,
				"autotune: refused: the copy's medium is refused (%s): %s\n",
				medium, context->copy_paths[copy]);
		else if (status != STATUS_OK)
			(void)autotune_refused(
				"the copy's drive has no hardware identity (by-uuid, mount or sysfs)",
				context->copy_paths[copy]);
	}
	if (status == STATUS_OK)
		status = pack_open_entry(context->storage, context->reader,
					 &context->pack);
	const uint32_t extent_count =
		status == STATUS_OK ?
			pack_header_of(context->pack)->extent_count :
			0;
	if (status == STATUS_OK) {
		context->offsets =
			calloc(extent_count + 1u, sizeof(*context->offsets));
		context->external =
			calloc(extent_count + 1u, sizeof(*context->external));
		context->host_extents = calloc(extent_count + 1u,
					       sizeof(*context->host_extents));
		status = context->offsets == NULL ||
					 context->external == NULL ||
					 context->host_extents == NULL ?
				 STATUS_OUT_OF_MEMORY :
				 tp_host_load_prepare(context->pack,
						      context->offsets,
						      extent_count,
						      &context->plan);
	}
	for (uint32_t index = 0; status == STATUS_OK && index < extent_count;
	     index++)
		if (context->offsets[index] != TP_HOST_LOAD_NOWHERE)
			context->host_extents[context->host_extent_count++] =
				index;
	if (status == STATUS_OK && context->plan.host_byte_count == 0) {
		status = STATUS_UNSUPPORTED;
		(void)autotune_refused(
			"the pack has no host-only extents: nothing for the TP host arena to load",
			source);
	}
	state->copy_count = context->copy_count;
	state->file_byte_count = header.file_byte_count;
	(void)snprintf(state->pack_identity, sizeof(state->pack_identity),
		       "%016" PRIx64, header.header_digest);
	return status;
}

static void context_close(struct load_context *context)
{
	pack_close(context->pack);
	if (context->reader != NULL)
		context->storage->reader_close(context->reader);
	free(context->offsets);
	free(context->external);
	free(context->host_extents);
}

/* The points still to measure, one after another; false (STATUS_TIMED_OUT) once the lease, the run's points or a
 * stop signal ends first, the progress kept. */
MUST_CHECK static result_status
points_run(struct load_context *context, struct load_state *state,
	   const char *state_path, uint64_t deadline, uint32_t points_per_run)
{
	result_status status = STATUS_OK;
	uint32_t measured = 0;
	/* once a process, before the first A D D A point it measures (a resume too) */
	bool warmed = false;
	struct load_selection selection = { 0 };
	for (uint32_t index = state->point_count;
	     status == STATUS_OK && index < LOAD_POINTS_MOST; index++) {
		if (load_stop != 0 || now_nanoseconds() >= deadline ||
		    (points_per_run != 0 && measured == points_per_run)) {
			status = STATUS_TIMED_OUT;
			break;
		}
		if (index >= LOAD_GRID_POINTS)
			grid_rank(state, &selection);
		struct load_point *const point = &state->points[index];
		memset(point, 0, sizeof(*point));
		arm_of(index, &selection, state, &point->readers_per_copy,
		       &point->piece_mib);
		const uint64_t window =
			(uint64_t)state->window_gib * GIB_BYTES *
			(index < LOAD_GRID_POINTS ? 1u :
						    LOAD_ABBA_WINDOW_FACTOR);
		if (index >= LOAD_GRID_POINTS && !warmed) {
			status = warmup_touch(context, index, window,
					      &state->warmup);
			warmed = status == STATUS_OK;
		}
		if (status == STATUS_OK)
			status = point_measure(context, index, window, point);
		if (status == STATUS_OK) {
			state->point_count = index + 1u;
			status = state_write(state_path, state);
			measured++;
		}
		(void)fprintf(
			stderr,
			"storage: load point %" PRIu32 "/%" PRIu32
			" readers a copy %" PRIu32 " piece %" PRIu32
			" MiB: %.2f GiB in %.3f s, %.3f GB/s, cpu %.3f s, retries %" PRIu64
			" status=%s\n",
			index + 1u, LOAD_POINTS_MOST, point->readers_per_copy,
			point->piece_mib,
			(double)point->bytes / (double)GIB_BYTES,
			(double)point->nanoseconds /
				(double)NANOSECONDS_PER_SECOND,
			point_rate(point),
			(double)point->cpu_nanoseconds /
				(double)NANOSECONDS_PER_SECOND,
			point->copy_retries, status_text(status));
	}
	return status;
}

/* The section from the finished points, into the profile; the progress file goes once the profile holds it. */
MUST_CHECK static result_status
section_publish(const struct load_context *context,
		const struct load_state *state, const char *profile,
		const char *real_source, const char *state_path)
{
	struct load_selection selection = { 0 };
	grid_rank(state, &selection);
	abba_select(state, &selection);
	char *text = NULL;
	size_t length = 0;
	FILE *const stream = open_memstream(&text, &length);
	if (stream == NULL)
		return STATUS_OUT_OF_MEMORY;
	section_json(stream, context, state, &selection);
	if (fclose(stream) != 0) {
		free(text);
		return STATUS_OUT_OF_MEMORY;
	}
	const char *refusal = NULL;
	result_status status = storage_load_section_write(
		profile, real_source, text, length, &refusal);
	free(text);
	if (refusal != NULL)
		(void)autotune_refused(refusal, profile);
	const struct load_point *const winner =
		&state->points[selection.winner];
	char flag[LOAD_FLAG_TEXT_BYTES] = "";
	if (selection.verdict.spread_over_bound)
		(void)snprintf(
			flag, sizeof(flag),
			"; flagged: the spread is over the %.0f %% bound, so look at the points before --qualify",
			AUTOTUNE_LOAD_SPREAD_BOUND *
				(double)LOCAL_DEVICE_PERCENT);
	if (status == STATUS_OK) {
		(void)unlink(state_path);
		(void)fprintf(
			stderr,
			"storage: load stage selects readers a copy %" PRIu32
			" piece %" PRIu32
			" MiB%s (A D D A margin %+.2f %% over the default, spread %.2f %%%s); "
			"a candidate until --qualify%s\n",
			winner->readers_per_copy, winner->piece_mib,
			selection.verdict.default_wins ? ", the default" : "",
			selection.verdict.margin * (double)LOCAL_DEVICE_PERCENT,
			selection.verdict.spread * (double)LOCAL_DEVICE_PERCENT,
			selection.verdict.exact_tie ?
				", an exact tie: the simpler arm" :
				"",
			flag);
	}
	return status;
}

/* --qualify: the candidate section, once it names this pack and these copies, becomes the enabled one. */
MUST_CHECK static result_status
section_qualify(const struct load_context *context, const char *profile,
		const char *real_source, bool external)
{
	struct storage_load_section section;
	char reason[STORAGE_PROFILE_REASON_BYTES];
	const bool usable =
		storage_load_section_read(profile, &section, reason) &&
		storage_load_section_matches(&section,
					     pack_header_of(context->pack),
					     context->copy_paths,
					     context->copy_count, external,
					     TP_LOAD_HOST_THREADS_DEFAULT,
					     TP_LOAD_SPLIT_STEAL, reason);
	if (!usable) {
		(void)autotune_refused(reason, profile);
		return STATUS_INVALID_ARGUMENT;
	}
	const char *refusal = NULL;
	const result_status status =
		storage_load_section_qualify(profile, real_source, &refusal);
	if (refusal != NULL)
		(void)autotune_refused(refusal, profile);
	return status;
}

static const char load_usage[] =
	"usage: anemone autotune PACK --stage=load [--profile=ABSOLUTE] [--seconds=1650] [--replica_external=0|1] "
	"[--window_gib=1..1024 (64)] [--points_per_run=N (0: all)] [--plan] [--qualify]\n"
	"The load stage tunes the TP host arena's read (readers a copy, piece size) through the loader over the "
	"copies serving finds, and writes the profile's \"load\" section as a candidate; --qualify enables it. "
	"Repeat the same command after exit75 to resume.\n";

/* Refused by name: the copy stage's options and the lookup tuning's. */
static const char *const foreign_options[] = {
	"--target=",   "--copy_workers=", "--copy_hash_pairs=",
	"--manifest=", "--verify=",	  "--replica_floor_percent=",
	"--tokens=",   "--prompt="
};
#define FOREIGN_OPTIONS (sizeof(foreign_options) / sizeof(foreign_options[0]))

int storage_autotune_load_cli(int count, char **arguments)
{
	const char *source = NULL, *profile = NULL;
	uint32_t seconds = AUTOTUNE_SECONDS_DEFAULT,
		 window_gib = LOAD_WINDOW_GIB_DEFAULT, points_per_run = 0,
		 external = 0;
	bool valid = true, help = false, plan = false, qualify = false;
	for (int index = AUTOTUNE_FIRST_OPTION; valid && index < count;
	     index++) {
		const char *const argument = arguments[index];
		bool foreign = false;
		for (size_t option = 0; option < FOREIGN_OPTIONS; option++)
			foreign = foreign ||
				  strncmp(argument, foreign_options[option],
					  strlen(foreign_options[option])) == 0;
		if (strcmp(argument, AUTOTUNE_STAGE_OPTION "load") == 0)
			continue;
		if (strcmp(argument, "--help") == 0) {
			help = true;
			break;
		}
		if (foreign)
			valid = autotune_refused(
				"the copy stage's or the lookup tuning's option, not the load stage's",
				argument);
		else if (strcmp(argument, "--plan") == 0)
			plan = true;
		else if (strcmp(argument, "--qualify") == 0)
			qualify = true;
		else if (strncmp(argument,
				 "--profile=", strlen("--profile=")) == 0)
			profile = argument + strlen("--profile=");
		else if (strncmp(argument,
				 "--seconds=", strlen("--seconds=")) == 0)
			valid = (autotune_number_option(
					 argument + strlen("--seconds="),
					 AUTOTUNE_SECONDS_MOST, &seconds) &&
				 seconds != 0) ||
				autotune_refused("--seconds takes 1 to 1700",
						 argument);
		else if (strncmp(argument,
				 "--window_gib=", strlen("--window_gib=")) == 0)
			valid = (autotune_number_option(
					 argument + strlen("--window_gib="),
					 LOAD_WINDOW_GIB_MOST, &window_gib) &&
				 window_gib != 0) ||
				autotune_refused("--window_gib takes 1 to 1024",
						 argument);
		else if (strncmp(argument, "--points_per_run=",
				 strlen("--points_per_run=")) == 0)
			valid = autotune_number_option(
					argument + strlen("--points_per_run="),
					LOAD_POINTS_MOST, &points_per_run) ||
				autotune_refused(
					"--points_per_run takes 0 to the stage's points",
					argument);
		else if (strncmp(argument, "--replica_external=",
				 strlen("--replica_external=")) == 0)
			valid = autotune_number_option(
					argument +
						strlen("--replica_external="),
					1, &external) ||
				autotune_refused(
					"--replica_external takes 0 or 1",
					argument);
		else if (argument[0] != '-' && source == NULL)
			source = argument;
		else
			valid = autotune_refused(
				"not an option of the load stage", argument);
	}
	char default_profile[PATH_MAX], real_source[PATH_MAX],
		state_path[PATH_MAX], folder[PATH_MAX];
	if (valid && !help && profile == NULL) {
		valid = autotune_profile_default(default_profile,
						 sizeof(default_profile));
		profile = default_profile;
	}
	if (valid && !help && source == NULL)
		valid = autotune_refused("no pack given", "PACK");
	/* --help stops the loop before any default profile: it prints the usage alone */
	valid = valid && !help &&
		((profile[0] == '/' && prepare_utf8_text(profile)) ||
		 autotune_refused("the profile path is no absolute UTF-8 path",
				  profile));
	valid = valid && source != NULL &&
		(realpath(source, real_source) != NULL ||
		 autotune_refused("the pack does not resolve", source));
	valid = valid &&
		(prepare_utf8_text(real_source) ||
		 autotune_refused(
			 "the source path is no UTF-8, so the profile cannot record it",
			 real_source));
	const int state_length = valid ? snprintf(state_path,
						  sizeof(state_path), "%s%s",
						  profile, STATE_SUFFIX) :
					 -1;
	valid = valid && state_length > 0 &&
		(size_t)state_length < sizeof(state_path);
	(void)snprintf(folder, sizeof(folder), "%s", valid ? profile : "/");
	char *const slash = strrchr(folder, '/');
	if (slash != NULL)
		*(slash == folder ? slash + 1 : slash) = '\0';
	struct stat directory;
	valid = valid &&
		((stat(folder, &directory) == 0 && S_ISDIR(directory.st_mode) &&
		  directory.st_uid == geteuid() &&
		  (directory.st_mode & (S_IWGRP | S_IWOTH)) == 0) ||
		 autotune_refused(
			 "the profile's folder is not this user's alone",
			 folder));
	if (!valid || help) {
		(void)fputs(load_usage, stderr);
		return help ? EXIT_SUCCESS : EXIT_FAILURE;
	}
	struct load_context *const context = calloc(1, sizeof(*context));
	struct load_state *const state = calloc(1, sizeof(*state));
	result_status status = context != NULL && state != NULL ?
				       STATUS_OK :
				       STATUS_OUT_OF_MEMORY;
	/* before any point: a profile the publish would refuse (the combined tuning's schema 2, another source's) */
	const char *refusal = NULL;
	if (status == STATUS_OK &&
	    storage_load_section_writable(profile, real_source, &refusal) !=
		    STATUS_OK) {
		(void)autotune_refused(refusal != NULL ?
					       refusal :
					       "the profile cannot be checked",
				       profile);
		status = STATUS_INVALID_ARGUMENT;
		if (plan)
			(void)puts("{\"valid\":false,\"stage\":\"load\"}");
	}
	if (status == STATUS_OK) {
		state->schema = LOAD_STATE_SCHEMA;
		state->window_gib = window_gib;
		status = context_open(context, real_source, external != 0,
				      state);
	}
	if (status == STATUS_OK)
		state_resume(state_path, state);
	const uint64_t deadline =
		now_nanoseconds() + (uint64_t)seconds * NANOSECONDS_PER_SECOND;
	if (status == STATUS_OK && plan)
		(void)printf(
			"{\"valid\":true,\"stage\":\"load\",\"copies\":%" PRIu32
			",\"host_bytes\":%" PRIu64 ",\"host_extents\":%" PRIu32
			",\"window_bytes\":%" PRIu64 ",\"points\":%" PRIu32
			",\"points_done\":%" PRIu32 "}\n",
			context->copy_count, context->plan.host_byte_count,
			context->host_extent_count,
			(uint64_t)window_gib * GIB_BYTES, LOAD_POINTS_MOST,
			state->point_count);
	else if (status == STATUS_OK && qualify)
		status = section_qualify(context, profile, real_source,
					 external != 0);
	else if (status == STATUS_OK) {
		load_stop = 0;
		struct sigaction action = { .sa_handler = load_signal },
				 old_term, old_int;
		(void)sigemptyset(&action.sa_mask);
		const bool term_set = sigaction(SIGTERM, &action, &old_term) ==
				      0;
		const bool int_set = sigaction(SIGINT, &action, &old_int) == 0;
		status = term_set && int_set ?
				 points_run(context, state, state_path,
					    deadline, points_per_run) :
				 STATUS_UNSUPPORTED;
		if (term_set)
			(void)sigaction(SIGTERM, &old_term, NULL);
		if (int_set)
			(void)sigaction(SIGINT, &old_int, NULL);
		if (status == STATUS_OK)
			status = section_publish(context, state, profile,
						 real_source, state_path);
	}
	if (context != NULL)
		context_close(context);
	free(context);
	free(state);
	(void)fprintf(stderr,
		      "storage: autotune stage=load status=%s resumable=%u\n",
		      status_text(status),
		      status == STATUS_TIMED_OUT ? 1u : 0u);
	return status == STATUS_TIMED_OUT ? AUTOTUNE_RETRY_EXIT :
	       status == STATUS_OK	  ? EXIT_SUCCESS :
					    EXIT_FAILURE;
}
