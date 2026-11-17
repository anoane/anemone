/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * load_arena_probe.c: the load stage's host read (tp_host_load_run over the copies serving finds: replicas_find_policy
 * and replicas_reader_open, with the copy split's stealing and the cards' role claimed), point by point, into an arena
 * mapped as the stage maps its own (tp_host_arena_allocate, 2 MiB pages advised), fresh or touched page by page on
 * TOUCH_THREADS threads before the timed read (the touch timed apart). Every point reads the whole host layout (every
 * HOST_ONLY extent), so the arms read the same bytes. It measures (17 Nov 2026) why readers 1 runs at half speed (the
 * first touch of fresh pages inside the read, against a touched arena) and what the piece size does at readers 0. Not
 * shipped: no part of the anemone binary.
 *
 *   load_arena_probe PACK --out=FILE --arm=NAME:READERS:PIECE_MIB:THREADS:fresh|touched ... --order=NAME,NAME,...
 *                    [--warmup_passes=2] [--replica_external=0|1]
 *
 * Before the first point, --warmup_passes unmeasured fresh arenas are touched over the host layout (the stage's
 * warm-up). Each point records its wall time, user and system CPU apart, the /proc/vmstat counters of its first touch,
 * the process's AnonHugePages before the arena is released, the read calls, the copy retries, the bytes of each copy,
 * the drives' temperatures before and after, and the extents verified, which must be the layout's. The run records
 * the copies, the THP settings and the free 2 MiB blocks (buddyinfo's order 9) before and after. FILE (made new) holds
 * one JSON object; exit 0 when every point read and verified every extent, 1 otherwise, 2 for the usage.
 */
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "anemone/pack.h"
#include "anemone/replicas.h"
#include "anemone/storage_exl3.h"
#include "anemone/storage_profile.h"
#include "anemone/tp_load.h"
#include "storage_autotune.h"
#include "storage_prepare.h"

#define ARMS_MOST 8u
#define ARM_NAME_BYTES 9u
#define ORDER_MOST 64u
#define WARMUP_PASSES_DEFAULT 2u
#define WARMUP_PASSES_MOST 8u
#define READERS_MOST 64u
#define TOUCH_THREADS 16u
#define LINE_BYTES 256u
#define BUDDY_HUGE_ORDER 9u /* 2 MiB blocks of 4 KiB pages */
#define BUDDY_READINGS \
	2u /* buddyinfo's free 2 MiB blocks: before the run and after it */
#define MIB_BYTES (1ull << 20u)
#define NS_SECOND 1000000000ull
#define DECIMAL 10
#define EXIT_USAGE 2

enum arena_kind { ARENA_FRESH, ARENA_TOUCHED, ARENA_KINDS };
static const char *const arena_names[ARENA_KINDS] = { "fresh", "touched" };

struct arm {
	char name[ARM_NAME_BYTES];
	uint32_t readers, piece_mib, threads, arena;
};
struct point {
	uint32_t arm, extents_verified;
	result_status status;
	uint64_t nanoseconds, touch_nanoseconds, read_calls, copy_retries,
		anon_huge_kib;
	uint64_t copy_bytes[STORAGE_MAXIMUM_COPIES];
	struct autotune_load_counters counters;
	struct autotune_drives before, after;
};
struct touch_slice {
	unsigned char *bytes;
	uint64_t first, end;
	pthread_t thread;
};

static uint64_t now_ns(void)
{
	struct timespec time;
	(void)clock_gettime(CLOCK_MONOTONIC, &time);
	return (uint64_t)time.tv_sec * NS_SECOND + (uint64_t)time.tv_nsec;
}

static bool number(const char *text, uint32_t most, uint32_t *output)
{
	char *end = NULL;
	errno = 0;
	const unsigned long value = strtoul(text, &end, DECIMAL);
	if (end == text || errno != 0 || value > most)
		return false;
	*output = (uint32_t)value;
	return *end == '\0' || *end == ':';
}

/* NAME:READERS:PIECE_MIB:THREADS:fresh|touched; the piece whole pack blocks within the loader's most. */
static bool arm_parse(const char *text, struct arm *arm)
{
	const char *field = strchr(text, ':');
	const size_t length = field != NULL ? (size_t)(field - text) : 0;
	if (length == 0 || length >= ARM_NAME_BYTES)
		return false;
	memcpy(arm->name, text, length);
	arm->name[length] = '\0';
	uint32_t *const values[] = { &arm->readers, &arm->piece_mib,
				     &arm->threads };
	const uint32_t most[] = { READERS_MOST,
				  (uint32_t)(TP_HOST_LOAD_CHUNK_BYTES_MOST /
					     MIB_BYTES),
				  TP_LOAD_MAXIMUM_WORKERS };
	for (size_t index = 0; index < sizeof(values) / sizeof(values[0]);
	     index++) {
		if (field == NULL ||
		    !number(field + 1, most[index], values[index]))
			return false;
		field = strchr(field + 1, ':');
	}
	arm->arena = ARENA_KINDS;
	for (uint32_t kind = 0; field != NULL && kind < ARENA_KINDS; kind++)
		arm->arena = strcmp(field + 1, arena_names[kind]) == 0 ?
				     kind :
				     arm->arena;
	return arm->arena < ARENA_KINDS && arm->piece_mib != 0 &&
	       arm->threads != 0 &&
	       (uint64_t)arm->piece_mib * MIB_BYTES % PACK_BLOCK_BYTES == 0;
}

/* The first line of a small text file, its newline dropped; "" when unread. */
static void first_line(const char *path, char *text, size_t capacity)
{
	FILE *const file = fopen(path, "re");
	text[0] = '\0';
	if (file != NULL && fgets(text, (int)capacity, file) != NULL)
		text[strcspn(text, "\n")] = '\0';
	if (file != NULL)
		(void)fclose(file);
}

/* The free blocks of BUDDY_HUGE_ORDER over every zone ("Node 0, zone Normal" then a count an order); UINT64_MAX when
 * unread. */
static uint64_t buddy_huge_free(void)
{
	FILE *const file = fopen("/proc/buddyinfo", "re");
	char line[LINE_BYTES];
	uint64_t total = file != NULL ? 0 : UINT64_MAX;
	while (file != NULL && fgets(line, sizeof(line), file) != NULL) {
		char *cursor = strstr(line, "zone");
		cursor = cursor != NULL ? cursor + strlen("zone") : NULL;
		cursor = cursor != NULL ? cursor + strspn(cursor, " ") : NULL;
		cursor = cursor != NULL ? cursor + strcspn(cursor, " ") :
					  NULL; /* past the zone's name */
		for (uint32_t order = 0;
		     cursor != NULL && order <= BUDDY_HUGE_ORDER; order++) {
			char *end = NULL;
			const unsigned long long blocks =
				strtoull(cursor, &end, DECIMAL);
			total += order == BUDDY_HUGE_ORDER && end != cursor ?
					 blocks :
					 0u;
			cursor = end != cursor ? end : NULL;
		}
	}
	if (file != NULL)
		(void)fclose(file);
	return total;
}

/* The process's AnonHugePages (kB) from smaps_rollup; UINT64_MAX when unread. */
static uint64_t anon_huge_kib(void)
{
	FILE *const file = fopen("/proc/self/smaps_rollup", "re");
	char line[LINE_BYTES];
	uint64_t value = UINT64_MAX;
	while (file != NULL && fgets(line, sizeof(line), file) != NULL)
		if (strncmp(line, "AnonHugePages:", strlen("AnonHugePages:")) ==
		    0)
			value = strtoull(line + strlen("AnonHugePages:"), NULL,
					 DECIMAL);
	if (file != NULL)
		(void)fclose(file);
	return value;
}

static void *touch_run(void *context)
{
	struct touch_slice *const slice = context;
	const uint64_t page = (uint64_t)sysconf(_SC_PAGESIZE);
	for (uint64_t offset = slice->first; offset < slice->end;
	     offset += page)
		slice->bytes[offset] = 1u;
	return NULL;
}

/* Every page of the arena's first byte_count bytes written once, on TOUCH_THREADS threads (one when they do not
 * start); its time. */
static uint64_t arena_touch(unsigned char *bytes, uint64_t byte_count)
{
	struct touch_slice slices[TOUCH_THREADS];
	const uint64_t begin = now_ns();
	const uint64_t share = byte_count / TOUCH_THREADS /
			       TP_HOST_ARENA_PAGE_BYTES *
			       TP_HOST_ARENA_PAGE_BYTES;
	uint32_t started = 0;
	for (uint32_t index = 0; index < TOUCH_THREADS; index++) {
		slices[index] = (struct touch_slice){
			.bytes = bytes,
			.first = share * index,
			.end = index + 1u == TOUCH_THREADS ?
				       byte_count :
				       share * (index + 1u)
		};
		if (pthread_create(&slices[index].thread, NULL, touch_run,
				   &slices[index]) != 0)
			(void)touch_run(&slices[index]);
		else
			started |= 1u << index;
	}
	for (uint32_t index = 0; index < TOUCH_THREADS; index++)
		if ((started >> index & 1u) != 0)
			(void)pthread_join(slices[index].thread, NULL);
	return now_ns() - begin;
}

struct probe {
	const struct storage_operations_record *storage;
	struct storage_reader *reader;
	struct pack_record *pack;
	struct tp_host_load_plan plan;
	uint64_t *offsets;
	uint8_t *external; /* none: every point reads the whole host layout */
	uint32_t copy_count, host_extents;
	const char *copy_paths[STORAGE_MAXIMUM_COPIES];
	char replica_paths[REPLICAS_MOST][REPLICA_PATH_CAPACITY];
};

/* The copies serving finds and the host plan, as the load stage opens them. */
static result_status probe_open(struct probe *probe, const char *source,
				bool external)
{
	struct pack_header header = { 0 };
	uint64_t source_bytes = 0;
	probe->storage = storage_for_path(source);
	if (!storage_pack_header_read(source, &header, &source_bytes))
		return STATUS_INPUT_OUTPUT_ERROR;
	const uint32_t found = replicas_find_policy(&source, 1u, &header,
						    probe->replica_paths,
						    REPLICAS_MOST, external);
	storage_exl3_read_mode(EXL3_READS_DIRECT_DROPPED);
	const uint32_t kept = replicas_reader_open(
		probe->storage, &source, 1u,
		(const char (*)[REPLICA_PATH_CAPACITY])probe->replica_paths,
		found, probe->copy_paths, &probe->reader, stderr);
	probe->copy_count = kept + 1u;
	probe->copy_paths[0] = source;
	result_status status =
		kept != 0 ? STATUS_OK :
			    probe->storage->reader_open(probe->copy_paths, 1,
							&probe->reader);
	if (probe->copy_count == 1u)
		storage_exl3_read_mode(EXL3_READS_DROPPED);
	if (status == STATUS_OK)
		status = pack_open_entry(probe->storage, probe->reader,
					 &probe->pack);
	const uint32_t extent_count =
		status == STATUS_OK ?
			pack_header_of(probe->pack)->extent_count :
			0;
	if (status == STATUS_OK) {
		probe->offsets =
			calloc(extent_count + 1u, sizeof(*probe->offsets));
		probe->external =
			calloc(extent_count + 1u, sizeof(*probe->external));
		status = probe->offsets == NULL || probe->external == NULL ?
				 STATUS_OUT_OF_MEMORY :
				 tp_host_load_prepare(probe->pack,
						      probe->offsets,
						      extent_count,
						      &probe->plan);
	}
	for (uint32_t index = 0; status == STATUS_OK && index < extent_count;
	     index++)
		probe->host_extents +=
			probe->offsets[index] != TP_HOST_LOAD_NOWHERE ? 1u : 0u;
	if (status == STATUS_OK && probe->plan.host_byte_count == 0)
		status = STATUS_UNSUPPORTED;
	return status;
}

static void probe_close(struct probe *probe)
{
	if (probe->pack != NULL)
		pack_close(probe->pack);
	if (probe->reader != NULL)
		probe->storage->reader_close(probe->reader);
	free(probe->offsets);
	free(probe->external);
}

/* One point: a fresh arena (touched first for a touched arm), the copies' cap, the loader's host run. */
static void point_measure(const struct probe *probe, const struct arm *arm,
			  struct point *point)
{
	struct tp_host_arena arena;
	point->status = tp_host_arena_allocate(probe->plan.host_byte_count,
					       true, &arena);
	if (point->status != STATUS_OK)
		return;
	(void)madvise(arena.bytes, arena.byte_count, MADV_HUGEPAGE);
	if (arm->arena == ARENA_TOUCHED)
		point->touch_nanoseconds =
			arena_touch(arena.bytes, probe->plan.host_byte_count);
	struct tp_load_copies copies;
	point->status = tp_load_copies_init(&copies, 0);
	if (point->status == STATUS_OK) {
		copies.host_reads_most = arm->readers;
		copies.split = TP_LOAD_SPLIT_STEAL;
		copies.role_claimed[TP_LOAD_ROLE_CARDS] = 1;
		const struct tp_host_load_request request = {
			.plan = &probe->plan,
			.external_host_extents = probe->external,
			.storage = probe->storage,
			.reader = probe->reader,
			.host_arena = arena.bytes,
			.host_arena_byte_count = probe->plan.host_byte_count,
			.worker_count = arm->threads,
			.copies = &copies,
			.chunk_bytes = (uint64_t)arm->piece_mib * MIB_BYTES
		};
		struct tp_host_load_result result = { 0 };
		struct autotune_load_counters counters;
		autotune_drives_read(&point->before);
		autotune_load_counters_read(&counters);
		const uint64_t begin = now_ns();
		point->status = tp_host_load_run(&request, &result);
		point->nanoseconds = now_ns() - begin;
		autotune_load_counters_since(&counters, &point->counters);
		autotune_drives_read(&point->after);
		point->anon_huge_kib = anon_huge_kib();
		point->read_calls = result.read_calls;
		point->copy_retries = result.copy_retries;
		point->extents_verified = result.extents_verified;
		for (uint32_t copy = 0; copy < probe->copy_count; copy++)
			point->copy_bytes[copy] = copies.delivered_bytes[copy];
		tp_load_copies_destroy(&copies);
	}
	tp_host_arena_release(&arena);
}

static void report(FILE *output, const struct probe *probe,
		   const struct arm *arms, uint32_t arm_count,
		   const struct point *points, uint32_t point_count,
		   const uint64_t buddy[BUDDY_READINGS], uint32_t warmup_passes,
		   uint64_t warmup_nanoseconds)
{
	char enabled[LINE_BYTES], defrag[LINE_BYTES];
	first_line("/sys/kernel/mm/transparent_hugepage/enabled", enabled,
		   sizeof(enabled));
	first_line("/sys/kernel/mm/transparent_hugepage/defrag", defrag,
		   sizeof(defrag));
	(void)fputs("{\"probe\":\"load_arena_probe\",\"copies\":[", output);
	for (uint32_t copy = 0; copy < probe->copy_count; copy++) {
		(void)fputs(copy == 0 ? "" : ",", output);
		prepare_json_text(output, probe->copy_paths[copy]);
	}
	(void)fprintf(output,
		      "],\"host_bytes\":%" PRIu64 ",\"host_extents\":%" PRIu32
		      ",\"thp\":{\"enabled\":",
		      probe->plan.host_byte_count, probe->host_extents);
	prepare_json_text(output, enabled);
	(void)fputs(",\"defrag\":", output);
	prepare_json_text(output, defrag);
	(void)fprintf(output,
		      "},\"buddy_huge_free_before\":%" PRIu64
		      ",\"buddy_huge_free_after\":%" PRIu64
		      ",\"warmup\":{\"passes\":%" PRIu32
		      ",\"nanoseconds\":%" PRIu64 "},\"arms\":[",
		      buddy[0], buddy[1], warmup_passes, warmup_nanoseconds);
	for (uint32_t index = 0; index < arm_count; index++)
		(void)fprintf(output,
			      "%s{\"name\":\"%s\",\"readers_per_copy\":%" PRIu32
			      ",\"piece_mib\":%" PRIu32 ",\"threads\":%" PRIu32
			      ",\"arena\":\"%s\"}",
			      index == 0 ? "" : ",", arms[index].name,
			      arms[index].readers, arms[index].piece_mib,
			      arms[index].threads,
			      arena_names[arms[index].arena]);
	(void)fputs("],\"points\":[", output);
	for (uint32_t index = 0; index < point_count; index++) {
		const struct point *const point = &points[index];
		(void)fprintf(
			output,
			"%s{\"arm\":\"%s\",\"status\":\"%s\",\"bytes\":%" PRIu64
			",\"nanoseconds\":%" PRIu64
			",\"gb_per_second\":%.3f,\"touch_nanoseconds\":%" PRIu64
			",\"read_calls\":%" PRIu64 ",\"copy_retries\":%" PRIu64
			",\"extents\":%" PRIu32 ",\"extents_verified\":%" PRIu32
			",\"anon_huge_kib\":%" PRIu64 ",\"copy_bytes\":[",
			index == 0 ? "" : ",", arms[point->arm].name,
			status_text(point->status), probe->plan.host_byte_count,
			point->nanoseconds,
			point->nanoseconds != 0 ?
				(double)probe->plan.host_byte_count /
					(double)point->nanoseconds :
				0.0,
			point->touch_nanoseconds, point->read_calls,
			point->copy_retries, probe->host_extents,
			point->extents_verified, point->anon_huge_kib);
		for (uint32_t copy = 0; copy < probe->copy_count; copy++)
			(void)fprintf(output, "%s%" PRIu64,
				      copy == 0 ? "" : ",",
				      point->copy_bytes[copy]);
		(void)fputc(']', output);
		autotune_load_counters_json(output, &point->counters);
		autotune_drives_json(output, "millicelsius_before",
				     &point->before);
		autotune_drives_json(output, "millicelsius_after",
				     &point->after);
		(void)fputc('}', output);
	}
	(void)fputs("]}\n", output);
}

static const char usage[] =
	"usage: load_arena_probe PACK --out=FILE --arm=NAME:READERS:PIECE_MIB:THREADS:fresh|touched ... "
	"--order=NAME,NAME,... [--warmup_passes=0..8 (2)] [--replica_external=0|1]\n";

int main(int count, char **arguments)
{
	static struct arm arms[ARMS_MOST];
	static struct point points[ORDER_MOST];
	static struct probe probe;
	const char *source = NULL, *out = NULL, *order = NULL;
	uint32_t arm_count = 0, point_count = 0,
		 warmup_passes = WARMUP_PASSES_DEFAULT, external = 0;
	bool valid = true;
	for (int index = 1; valid && index < count; index++) {
		const char *const word = arguments[index];
		if (strncmp(word, "--arm=", strlen("--arm=")) == 0) {
			valid = arm_count < ARMS_MOST &&
				arm_parse(word + strlen("--arm="),
					  &arms[arm_count]);
			arm_count += valid ? 1u : 0u;
		} else if (strncmp(word, "--order=", strlen("--order=")) == 0)
			order = word + strlen("--order=");
		else if (strncmp(word, "--out=", strlen("--out=")) == 0)
			out = word + strlen("--out=");
		else if (strncmp(word, "--warmup_passes=",
				 strlen("--warmup_passes=")) == 0)
			valid = number(word + strlen("--warmup_passes="),
				       WARMUP_PASSES_MOST, &warmup_passes);
		else if (strncmp(word, "--replica_external=",
				 strlen("--replica_external=")) == 0)
			valid = number(word + strlen("--replica_external="), 1u,
				       &external);
		else if (word[0] != '-' && source == NULL)
			source = word;
		else
			valid = false;
	}
	/* the order: each name an arm's, at most ORDER_MOST points */
	for (const char *name = order;
	     valid && name != NULL && *name != '\0';) {
		const size_t length = strcspn(name, ",");
		uint32_t found = ARMS_MOST;
		for (uint32_t index = 0; index < arm_count; index++)
			found = strlen(arms[index].name) == length &&
						strncmp(arms[index].name, name,
							length) == 0 ?
					index :
					found;
		valid = found < arm_count && point_count < ORDER_MOST;
		if (valid)
			points[point_count++].arm = found;
		name += length + (name[length] == ',' ? 1u : 0u);
	}
	const int descriptor =
		valid && source != NULL && out != NULL && point_count != 0 ?
			open(out, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC,
			     S_IRUSR | S_IWUSR) :
			-1;
	FILE *const output = descriptor >= 0 ? fdopen(descriptor, "w") : NULL;
	if (output == NULL) {
		(void)fputs(usage, stderr);
		if (descriptor >= 0)
			(void)close(descriptor);
		return EXIT_USAGE;
	}
	result_status status = probe_open(&probe, source, external != 0);
	uint64_t buddy[BUDDY_READINGS] = { buddy_huge_free(), 0 },
		 warmup_nanoseconds = 0;
	for (uint32_t pass = 0; status == STATUS_OK && pass < warmup_passes;
	     pass++) {
		struct tp_host_arena arena;
		status = tp_host_arena_allocate(probe.plan.host_byte_count,
						true, &arena);
		if (status == STATUS_OK) {
			(void)madvise(arena.bytes, arena.byte_count,
				      MADV_HUGEPAGE);
			warmup_nanoseconds += arena_touch(
				arena.bytes, probe.plan.host_byte_count);
			tp_host_arena_release(&arena);
		}
	}
	bool verified = status == STATUS_OK;
	for (uint32_t index = 0; status == STATUS_OK && index < point_count;
	     index++) {
		point_measure(&probe, &arms[points[index].arm], &points[index]);
		verified = verified && points[index].status == STATUS_OK &&
			   points[index].extents_verified == probe.host_extents;
		(void)fprintf(
			stderr,
			"load_arena_probe: point %" PRIu32 "/%" PRIu32
			" %s: %.3f GB/s, touch %.3f s, user %.3f s, "
			"system %.3f s, extents %" PRIu32 "/%" PRIu32
			" status=%s\n",
			index + 1u, point_count, arms[points[index].arm].name,
			points[index].nanoseconds != 0 ?
				(double)probe.plan.host_byte_count /
					(double)points[index].nanoseconds :
				0.0,
			(double)points[index].touch_nanoseconds /
				(double)NS_SECOND,
			(double)points[index].counters.user_nanoseconds /
				(double)NS_SECOND,
			(double)points[index].counters.system_nanoseconds /
				(double)NS_SECOND,
			points[index].extents_verified, probe.host_extents,
			status_text(points[index].status));
	}
	buddy[1] = buddy_huge_free();
	if (status == STATUS_OK)
		report(output, &probe, arms, arm_count, points, point_count,
		       buddy, warmup_passes, warmup_nanoseconds);
	else
		(void)fprintf(
			stderr,
			"load_arena_probe: %s does not open as the stage opens it: %s\n",
			source, status_text(status));
	const bool written = fclose(output) == 0;
	probe_close(&probe);
	return verified && written ? EXIT_SUCCESS : EXIT_FAILURE;
}
