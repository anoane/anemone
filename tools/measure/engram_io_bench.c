/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * engram_io_bench.c: the Engram I/O bench driver (engram_io_bench.h).
 *
 *   engram_io_bench run --copies=A,B,C --traces=LIST --arms=ARMS --out=OUT.jsonl [--seconds=N] [--repeats=R]
 *                       [--skip_done=1] [--prep_threads_per_copy=N] [--prep_piece_mib=N] [--memory_floor_gib=N]
 *   engram_io_bench digests --copies=A,B,C [--prep_threads_per_copy=N] [--prep_piece_mib=N]
 *   engram_io_bench characterise --copies=A,B,C --plan=PLAN --out=OUT.jsonl [--seconds=N] [--skip_done=1] [--raw=1]
 *   engram_io_bench plan --copies=A,B,C --traces=LIST (the pack, the cases and the steps: no table byte is read)
 *   engram_io_bench hot_build --copies=A --traces=CORPUS --hot_mib=N --out=FILE (a static hot set, engram_hot.h,
 *                   of the corpus's most referenced blocks; no table byte is read)
 *   engram_io_bench scan --copies=A,B,C --scan_threads=N (the engine's own startup digest scan, engram_disk_open,
 *                   timed with N readers a copy; 0 is the serial scan)
 *   engram_io_bench self_test --directory=DIR [--arms=ARMS]
 *
 * LIST holds one "<token file> <prompt tokens>" per line (U32 token ids). ARMS holds one "<name> key=value ..." per
 * line (see arm_parse). Each arm runs at each concurrency level of its `levels` (C sessions; a decode step holds one
 * row per session; prefill runs each session's prompt in chunks) and each repeat (repeat r > 0 shifts every row id by
 * whole blocks, so it reads blocks earlier repeats did not). Every block is checked against its pack digest, every
 * step's rows are compared with the verified oracle copy outside the timing, and one JSON row per (arm, level,
 * repeat) is appended to OUT. Exit 75 when the lease's seconds run out first (resume with --skip_done=1).
 */
#include "engram_io_bench.h"

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <linux/io_uring.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <unistd.h>

#include "anemone/engram_disk.h"
#include "anemone/pack.h"
#include "anemone/pack_digest.h"
#include "anemone/storage_seam.h"
#include "engram_hot.h"
#include "sha256.h"

#define EXIT_CHECKPOINT 75
#define DECODE_STEPS 64u
#define SHIFT_BLOCKS \
	1000003ull /* a repeat's shift of every row id, in whole blocks */
#define ORACLE_LOADERS_PER_COPY 16u
#define PREP_THREADS_MOST 32u
#define PREP_PIECE_DEFAULT_MIB 16u
#define MIB_BYTES (1ull << 20u)
#define GIB_BYTES (1ull << 30u)
#define MEMORY_FLOOR_DEFAULT_GIB 64u
#define ESTIMATE_DEFAULT_SECONDS 90u
#define ESTIMATE_FACTOR 3u
#define ESTIMATE_DIVISOR 2u
#define ARM_LINE_BYTES 1024u
#define ARMS_MOST 512u
#define LIST_LINE_BYTES (PATH_MAX + 64u)
#define TOKENS_MOST (1u << 22u)
#define SCORE_CAP 4096u
#define PERCENT 100u
#define PER_MILLE_P50 500u
#define PER_MILLE_P90 900u
#define PER_MILLE_P99 990u
#define PER_MILLE_P999 999u
#define PER_MILLE_ALL 1000u
#define DEFAULT_DEPTH 256u
#define DEFAULT_SLICE_US 200u
#define DEFAULT_POOL_THREADS 16u
#define DEFAULT_SQ_IDLE_MS 10u
#define KEY_EMPTY 0ull
#define KEY_TABLE_SHIFT 40u
#define HASH_MULTIPLIER 11400714819323198485ull
#define DEVICE_STAT_FIELDS 11u
#define DEVICE_STAT_READS 0u
#define DEVICE_STAT_TICKS 9u
#define INTERRUPT_LINE_BYTES 8192u
#define RESOLVED UINT32_MAX
/* self-test store */
#define SELF_TABLE_BLOCKS 8192u
#define SELF_ROW_SHORTFALL 7u
#define SELF_CASES 4u
#define SELF_TOKENS_BASE 300u
#define SELF_TOKENS_STEP 37u
#define SELF_PROMPT_BASE 200u
#define SELF_PROMPT_STEP 13u
#define SELF_HOT_ROWS 400u
#define SELF_HOT_PERCENT 70u
#define SELF_ROWS_PER_TOKEN 24u
#define SELF_TABLES 2u
#define SELF_CHUNK 64u
#define SELF_SEED 0x9E3779B97F4A7C15ull

static const uint32_t level_sessions[BENCH_LEVELS] = { 1, 2, 4, 8 };

/* ---- small helpers ----------------------------------------------------------------------------------------- */

static uint64_t mix64(uint64_t value)
{
	value ^= value >> 33u;
	value *= 0xff51afd7ed558ccdull;
	value ^= value >> 33u;
	value *= 0xc4ceb9fe1a85ec53ull;
	value ^= value >> 33u;
	return value;
}

static int compare_u64(const void *left, const void *right)
{
	const uint64_t a = *(const uint64_t *)left,
		       b = *(const uint64_t *)right;
	return a < b ? -1 : a > b ? 1 : 0;
}

/* Nearest rank of a sorted copy. */
static uint64_t percentile(const uint64_t *values, uint32_t count,
			   uint32_t per_mille, uint64_t *scratch)
{
	if (count == 0)
		return 0;
	memcpy(scratch, values, (size_t)count * sizeof(*values));
	qsort(scratch, count, sizeof(*scratch), compare_u64);
	uint64_t rank = ((uint64_t)count * per_mille + PER_MILLE_ALL - 1u) /
			PER_MILLE_ALL;
	rank = rank == 0 ? 1u : rank;
	return scratch[rank - 1u];
}

static uint64_t mean(const uint64_t *values, uint32_t count)
{
	uint64_t sum = 0;
	for (uint32_t index = 0; index < count; index++)
		sum += values[index];
	return count != 0 ? sum / count : 0;
}

static bool parse_u64(const char *text, uint64_t *value)
{
	if (text == NULL || !isdigit((unsigned char)text[0]))
		return false;
	char *end = NULL;
	errno = 0;
	const unsigned long long parsed = strtoull(text, &end, 10);
	if (errno != 0 || (*end != '\0' && *end != ',' && *end != ':'))
		return false;
	*value = parsed;
	return true;
}

static bool parse_u32(const char *text, uint32_t *value)
{
	uint64_t parsed = 0;
	if (!parse_u64(text, &parsed) || parsed > UINT32_MAX)
		return false;
	*value = (uint32_t)parsed;
	return true;
}

static uint64_t memory_available_bytes(void)
{
	FILE *const input = fopen("/proc/meminfo", "re");
	if (input == NULL)
		return 0;
	char line[BENCH_TEXT_BYTES];
	unsigned long long kib = 0;
	while (fgets(line, sizeof(line), input) != NULL)
		if (sscanf(line, "MemAvailable: %llu kB", &kib) == 1)
			break;
	(void)fclose(input);
	return (uint64_t)kib * 1024u;
}

/* ---- arms -------------------------------------------------------------------------------------------------- */

static int name_index(const char *value, const char *const *names,
		      uint32_t count)
{
	for (uint32_t index = 0; index < count; index++)
		if (strcmp(value, names[index]) == 0)
			return (int)index;
	return -1;
}

static const char *const engine_names[ENGINE_KINDS] = {
	"uring", "aio", "pool", "sync", "mmap", "prod"
};
static const char *const operation_names[OPERATION_KINDS] = {
	"read", "readv", "read_fixed", "readv_fixed", "uring_cmd"
};
static const char *const call_names[CALL_KINDS] = { "pread", "preadv2",
						    "ioctl" };
static const char *const batch_names[BATCH_KINDS] = { "call", "table", "step" };
static const char *const wait_names[WAIT_KINDS] = { "poll", "enter", "spin",
						    "hybrid" };
static const char *const copy_names[COPY_KINDS] = { "first", "tick", "block",
						    "jsq",   "p2c",  "rate" };
static const char *const unit_names[HOT_UNITS] = { "block", "row" };
static const char *const profile_names[PROFILES] = { "all", "other" };
static const char *const schedule_names[SCHEDULE_KINDS] = { "keep",  "other",
							    "batch", "idle",
							    "fifo",  "rr" };

struct flag_name {
	const char *name;
	uint32_t flag;
};
static const struct flag_name setup_flag_names[] = {
	{ "none", 0 },
	{ "single_issuer", IORING_SETUP_SINGLE_ISSUER },
	{ "defer_taskrun", IORING_SETUP_DEFER_TASKRUN },
	{ "coop_taskrun", IORING_SETUP_COOP_TASKRUN },
	{ "taskrun_flag", IORING_SETUP_TASKRUN_FLAG },
	{ "no_sqarray", IORING_SETUP_NO_SQARRAY },
	{ "submit_all", IORING_SETUP_SUBMIT_ALL },
	{ "sqpoll", IORING_SETUP_SQPOLL },
	{ "iopoll", IORING_SETUP_IOPOLL },
	{ "hybrid_iopoll", IORING_SETUP_HYBRID_IOPOLL },
	{ "clamp", IORING_SETUP_CLAMP },
};
static const struct flag_name rw_flag_names[] = { { "none", 0 },
						  { "nowait", RWF_NOWAIT },
						  { "hipri", RWF_HIPRI } };

static bool flags_parse(const char *value, const struct flag_name *names,
			uint32_t count, uint32_t *flags)
{
	*flags = 0;
	char copy[BENCH_TEXT_BYTES];
	(void)snprintf(copy, sizeof(copy), "%s", value);
	for (char *save = NULL, *token = strtok_r(copy, ",", &save);
	     token != NULL; token = strtok_r(NULL, ",", &save)) {
		bool found = false;
		for (uint32_t index = 0; index < count; index++)
			if (strcmp(token, names[index].name) == 0) {
				*flags |= names[index].flag;
				found = true;
			}
		if (!found)
			return false;
	}
	return true;
}

static bool list_parse(const char *value, uint32_t *items, uint32_t most,
		       uint32_t *count)
{
	*count = 0;
	for (const char *cursor = value; *cursor != '\0';) {
		if (*count == most || !parse_u32(cursor, &items[*count]))
			return false;
		(*count)++;
		cursor = strchr(cursor, ',');
		if (cursor == NULL)
			break;
		cursor++;
	}
	return *count != 0;
}

static void arm_defaults(struct arm *arm, uint32_t copies)
{
	memset(arm, 0, sizeof(*arm));
	arm->engine = ENGINE_URING;
	arm->operation = OPERATION_READ;
	arm->depth = DEFAULT_DEPTH;
	arm->sq_cpu = UINT32_MAX;
	arm->sq_idle_ms = DEFAULT_SQ_IDLE_MS;
	arm->registered_files = true;
	arm->wait = WAIT_ENTER;
	arm->slice_us = DEFAULT_SLICE_US;
	arm->pool_threads = DEFAULT_POOL_THREADS;
	arm->batch = BATCH_STEP;
	arm->dedupe = true;
	arm->copy = COPY_JSQ;
	arm->copy_mask = (1u << copies) - 1u;
	arm->poll_copy_mask = arm->copy_mask;
	arm->command_copy_mask = arm->copy_mask;
	for (uint32_t copy = 0; copy < BENCH_COPIES_MOST; copy++)
		arm->weights[copy] = 1;
	arm->step_cpu = -1;
	arm->prefill_chunk = BENCH_CHUNK_TOKENS_MOST;
	arm->levels_mask = (1u << BENCH_LEVELS) - 1u;
	arm->prefill = 1;
}

/* A list of copy indices as a mask: false for an index outside the copies. */
static bool mask_parse(const char *value, uint32_t copies, uint32_t *mask)
{
	uint32_t items[BENCH_COPIES_MOST], count = 0;
	bool valid = list_parse(value, items, BENCH_COPIES_MOST, &count);
	*mask = 0;
	for (uint32_t item = 0; valid && item < count; item++) {
		valid = items[item] < copies;
		*mask |= valid ? 1u << items[item] : 0u;
	}
	return valid;
}

/* "name key=value ...": false (with the offending key in reason) on any unknown key or value. */
static bool arm_parse(const char *line, uint32_t copies, struct arm *arm,
		      char *reason, size_t reason_bytes)
{
	arm_defaults(arm, copies);
	char text[ARM_LINE_BYTES];
	(void)snprintf(text, sizeof(text), "%s", line);
	text[strcspn(text, "\r\n")] = '\0';
	(void)snprintf(arm->text, sizeof(arm->text), "%s", text);
	for (const char *cursor = text; *cursor != '\0'; cursor++)
		if (!isalnum((unsigned char)*cursor) &&
		    strchr("_=,.: -", *cursor) == NULL) {
			(void)snprintf(reason, reason_bytes,
				       "character outside [a-z0-9_=,.: -]");
			return false;
		}
	char *save = NULL;
	char *token = strtok_r(text, " ", &save);
	if (token == NULL || strchr(token, '=') != NULL) {
		(void)snprintf(reason, reason_bytes,
			       "an arm begins with its name");
		return false;
	}
	(void)snprintf(arm->name, sizeof(arm->name), "%s", token);
	for (token = strtok_r(NULL, " ", &save); token != NULL;
	     token = strtok_r(NULL, " ", &save)) {
		char *const equals = strchr(token, '=');
		if (equals == NULL) {
			(void)snprintf(reason, reason_bytes, "%s: no value",
				       token);
			return false;
		}
		*equals = '\0';
		const char *const key = token, *const value = equals + 1;
		uint32_t number = 0, items[BENCH_OWNERS_MOST], count = 0;
		int index = -1;
		bool valid = true;
		if (strcmp(key, "engine") == 0)
			valid = (index = name_index(value, engine_names,
						    ENGINE_KINDS)) >= 0 &&
				(arm->engine = (enum engine_kind)index, true);
		else if (strcmp(key, "setup") == 0)
			valid = flags_parse(value, setup_flag_names,
					    sizeof(setup_flag_names) /
						    sizeof(setup_flag_names[0]),
					    &arm->setup_flags);
		else if (strcmp(key, "op") == 0)
			valid = (index = name_index(value, operation_names,
						    OPERATION_KINDS)) >= 0 &&
				(arm->operation = (uint32_t)index, true);
		else if (strcmp(key, "depth") == 0)
			valid = parse_u32(value, &arm->depth) &&
				arm->depth != 0;
		else if (strcmp(key, "cq") == 0)
			valid = parse_u32(value, &arm->cq_factor);
		else if (strcmp(key, "sq_cpu") == 0)
			valid = parse_u32(value, &arm->sq_cpu);
		else if (strcmp(key, "sq_idle_ms") == 0)
			valid = parse_u32(value, &arm->sq_idle_ms);
		else if (strcmp(key, "files") == 0)
			valid = parse_u32(value, &number) && number <= 1u &&
				(arm->registered_files = number != 0, true);
		else if (strcmp(key, "ring_fd") == 0)
			valid = parse_u32(value, &number) && number <= 1u &&
				(arm->registered_ring = number != 0, true);
		else if (strcmp(key, "async") == 0)
			valid = parse_u32(value, &number) && number <= 1u &&
				(arm->sqe_async = number != 0, true);
		else if (strcmp(key, "rings") == 0)
			valid = (strcmp(value, "shared") == 0 ||
				 strcmp(value, "device") == 0) &&
				(arm->per_device_rings =
					 strcmp(value, "device") == 0,
				 true);
		else if (strcmp(key, "wait") == 0)
			valid = (index = name_index(value, wait_names,
						    WAIT_KINDS)) >= 0 &&
				(arm->wait = (enum wait_kind)index, true);
		else if (strcmp(key, "spin_us") == 0)
			valid = parse_u32(value, &arm->spin_us);
		else if (strcmp(key, "slice_us") == 0)
			valid = parse_u32(value, &arm->slice_us) &&
				arm->slice_us != 0;
		else if (strcmp(key, "io_threads") == 0)
			valid = parse_u32(value, &arm->io_threads) &&
				arm->io_threads <= BENCH_OWNERS_MOST;
		else if (strcmp(key, "pool_threads") == 0)
			valid = parse_u32(value, &arm->pool_threads) &&
				arm->pool_threads != 0 &&
				arm->pool_threads <= BENCH_POOL_THREADS_MOST;
		else if (strcmp(key, "call") == 0)
			valid = (index = name_index(value, call_names,
						    CALL_KINDS)) >= 0 &&
				(arm->call = (enum pool_call)index, true);
		else if (strcmp(key, "rwf") == 0)
			valid = flags_parse(value, rw_flag_names,
					    sizeof(rw_flag_names) /
						    sizeof(rw_flag_names[0]),
					    &arm->rw_flags);
		else if (strcmp(key, "pool_spin") == 0)
			valid = parse_u32(value, &number) && number <= 1u &&
				(arm->pool_spin = number != 0, true);
		else if (strcmp(key, "buffered") == 0)
			valid = parse_u32(value, &number) && number <= 1u &&
				(arm->buffered = number != 0, true);
		else if (strcmp(key, "fadvise") == 0)
			valid = (strcmp(value, "random") == 0 ||
				 strcmp(value, "none") == 0) &&
				(arm->fadvise_random =
					 strcmp(value, "random") == 0,
				 true);
		else if (strcmp(key, "readahead") == 0)
			valid = parse_u32(value, &number) && number <= 1u &&
				(arm->readahead = number != 0, true);
		else if (strcmp(key, "willneed") == 0)
			valid = parse_u32(value, &number) && number <= 1u &&
				(arm->willneed = number != 0, true);
		else if (strcmp(key, "batch") == 0)
			valid = (index = name_index(value, batch_names,
						    BATCH_KINDS)) >= 0 &&
				(arm->batch = (enum batch_kind)index, true);
		else if (strcmp(key, "dedupe") == 0)
			valid = parse_u32(value, &number) && number <= 1u &&
				(arm->dedupe = number != 0, true);
		else if (strcmp(key, "copy") == 0)
			valid = (index = name_index(value, copy_names,
						    COPY_KINDS)) >= 0 &&
				(arm->copy = (enum copy_policy)index, true);
		else if (strcmp(key, "copies") == 0)
			valid = mask_parse(value, copies, &arm->copy_mask);
		else if (strcmp(key, "poll_copies") == 0)
			valid = mask_parse(value, copies, &arm->poll_copy_mask);
		else if (strcmp(key, "command_copies") == 0)
			valid = mask_parse(value, copies,
					   &arm->command_copy_mask);
		else if (strcmp(key, "weights") == 0) {
			valid = list_parse(value, items, BENCH_COPIES_MOST,
					   &count) &&
				count == copies;
			for (uint32_t item = 0; valid && item < count; item++)
				arm->weights[item] = items[item];
		} else if (strcmp(key, "hedge_us") == 0)
			valid = parse_u32(value, &arm->hedge_us);
		else if (strcmp(key, "hedge_percent") == 0)
			valid = parse_u32(value, &arm->hedge_percent);
		else if (strcmp(key, "hedge_step_most") == 0)
			valid = parse_u32(value, &arm->hedge_step_most);
		else if (strcmp(key, "inflight") == 0)
			valid = parse_u32(value, &arm->inflight_most);
		else if (strcmp(key, "lead_us") == 0) {
			valid = list_parse(value, items, BENCH_TABLES_MOST,
					   &count);
			for (uint32_t item = 0;
			     valid && item < BENCH_TABLES_MOST; item++)
				arm->lead_us[item] =
					items[item < count ? item : count - 1u];
		} else if (strcmp(key, "cache_mib") == 0)
			valid = parse_u32(value, &number) &&
				(arm->cache_bytes =
					 (uint64_t)number * MIB_BYTES,
				 true);
		else if (strcmp(key, "hot_mib") == 0)
			valid = parse_u32(value, &number) &&
				(arm->hot_bytes = (uint64_t)number * MIB_BYTES,
				 true);
		else if (strcmp(key, "hot_unit") == 0)
			valid = (index = name_index(value, unit_names,
						    HOT_UNITS)) >= 0 &&
				(arm->hot_unit = (enum hot_unit)index, true);
		else if (strcmp(key, "hot_profile") == 0)
			valid = (index = name_index(value, profile_names,
						    PROFILES)) >= 0 &&
				(arm->hot_profile = (enum hot_profile)index,
				 true);
		else if (strcmp(key, "hot_file") == 0)
			valid = strchr(value, '/') == NULL &&
				strlen(value) < sizeof(arm->hot_file) &&
				(memcpy(arm->hot_file, value,
					strlen(value) + 1u),
				 true);
		else if (strcmp(key, "hot_sha256") == 0)
			valid = strlen(value) == ENGRAM_HOT_SHA256_HEX_BYTES &&
				(memcpy(arm->hot_sha256, value,
					strlen(value) + 1u),
				 true);
		else if (strcmp(key, "ram_percent") == 0)
			valid = parse_u32(value, &arm->ram_percent) &&
				arm->ram_percent <= PERCENT;
		else if (strcmp(key, "all_ram") == 0)
			valid = parse_u32(value, &number) && number <= 1u &&
				(arm->all_ram = number != 0, true);
		else if (strcmp(key, "pin") == 0) {
			if (strcmp(value, "none") == 0)
				arm->pin = PIN_NONE;
			else if (strcmp(value, "irq") == 0)
				arm->pin = PIN_IRQ;
			else if (strncmp(value, "cpus:", strlen("cpus:")) ==
				 0) {
				arm->pin = PIN_LIST;
				valid = list_parse(value + strlen("cpus:"),
						   arm->pin_cpus,
						   BENCH_OWNERS_MOST,
						   &arm->pin_cpu_count);
			} else
				valid = false;
		} else if (strcmp(key, "step_cpu") == 0)
			valid = parse_u32(value, &number) &&
				(arm->step_cpu = (int32_t)number, true);
		else if (strcmp(key, "nice") == 0) {
			char *end = NULL;
			const long parsed = strtol(value, &end, 10);
			valid = *end == '\0' && parsed >= -20 && parsed <= 19;
			arm->nice = (int32_t)parsed;
		} else if (strcmp(key, "sched") == 0)
			valid = (index = name_index(value, schedule_names,
						    SCHEDULE_KINDS)) >= 0 &&
				(arm->scheduling = (enum scheduling_kind)index,
				 true);
		else if (strcmp(key, "rtprio") == 0)
			valid = parse_u32(value, &arm->rt_priority);
		else if (strcmp(key, "ioprio") == 0) {
			if (strcmp(value, "keep") == 0)
				arm->io_priority = IO_PRIORITY_KEEP;
			else if (strcmp(value, "idle") == 0)
				arm->io_priority = IO_PRIORITY_IDLE;
			else if (strncmp(value, "rt:", strlen("rt:")) == 0)
				valid = parse_u32(value + strlen("rt:"),
						  &arm->io_priority_level) &&
					(arm->io_priority = IO_PRIORITY_RT,
					 true);
			else if (strncmp(value, "be:", strlen("be:")) == 0)
				valid = parse_u32(value + strlen("be:"),
						  &arm->io_priority_level) &&
					(arm->io_priority = IO_PRIORITY_BE,
					 true);
			else
				valid = false;
		} else if (strcmp(key, "prefill_chunk") == 0)
			valid = parse_u32(value, &arm->prefill_chunk) &&
				arm->prefill_chunk != 0 &&
				arm->prefill_chunk <= BENCH_CHUNK_TOKENS_MOST;
		else if (strcmp(key, "prefill") == 0)
			valid = parse_u32(value, &arm->prefill) &&
				arm->prefill <= 1u;
		else if (strcmp(key, "levels") == 0) {
			valid = list_parse(value, items, BENCH_LEVELS, &count);
			arm->levels_mask = 0;
			for (uint32_t item = 0; valid && item < count; item++) {
				bool known = false;
				for (uint32_t level = 0; level < BENCH_LEVELS;
				     level++)
					if (level_sessions[level] ==
					    items[item]) {
						arm->levels_mask |= 1u << level;
						known = true;
					}
				valid = known;
			}
		} else
			valid = false;
		if (!valid) {
			(void)snprintf(reason, reason_bytes, "%s=%s", key,
				       value);
			return false;
		}
	}
	if (arm->copy_mask == 0) {
		(void)snprintf(reason, reason_bytes, "copies: none");
		return false;
	}
	if ((arm->hot_file[0] != '\0') != (arm->hot_sha256[0] != '\0') ||
	    (arm->hot_file[0] != '\0' &&
	     (arm->hot_bytes == 0 || arm->hot_unit != HOT_BLOCK))) {
		(void)snprintf(
			reason, reason_bytes,
			"hot_file needs hot_sha256 and hot_mib, in blocks");
		return false;
	}
	const uint32_t all = (1u << copies) - 1u;
	if ((arm->poll_copy_mask != all || arm->command_copy_mask != all) &&
	    !arm->per_device_rings) {
		(void)snprintf(
			reason, reason_bytes,
			"poll_copies and command_copies need rings=device");
		return false;
	}
	return true;
}

/* ---- the store --------------------------------------------------------------------------------------------- */

static void device_of(struct bench_store *store, uint32_t copy)
{
	struct stat file;
	(void)snprintf(store->device_names[copy],
		       sizeof(store->device_names[copy]), "unknown");
	store->irq_cpu[copy] = 0;
	if (fstat(store->direct[copy], &file) != 0)
		return;
	char link[PATH_MAX], resolved[PATH_MAX], path[PATH_MAX];
	(void)snprintf(link, sizeof(link), "/sys/dev/block/%u:%u",
		       major(file.st_dev), minor(file.st_dev));
	if (realpath(link, resolved) == NULL)
		return;
	(void)snprintf(path, sizeof(path), "%s/partition", resolved);
	if (access(path, F_OK) == 0)
		*strrchr(resolved, '/') = '\0';
	(void)snprintf(store->device_names[copy],
		       sizeof(store->device_names[copy]), "%s",
		       strrchr(resolved, '/') + 1);
	/* the completion vector of hardware queue 0 of the drive's controller: nvmeXn1 -> nvmeXq1 */
	char vector[BENCH_REASON_BYTES];
	const char *const name = store->device_names[copy];
	const char *const namespace = strchr(name + strlen("nvme"), 'n');
	if (strncmp(name, "nvme", strlen("nvme")) != 0 || namespace == NULL)
		return;
	(void)snprintf(vector, sizeof(vector), "%.*sq1",
		       (int)(namespace - name), name);
	FILE *const interrupts = fopen("/proc/interrupts", "re");
	if (interrupts == NULL)
		return;
	static char line[INTERRUPT_LINE_BYTES];
	unsigned irq = UINT32_MAX;
	while (fgets(line, sizeof(line), interrupts) != NULL) {
		char *const end = line + strcspn(line, "\n");
		*end = '\0';
		const char *const last = strrchr(line, ' ');
		if (last != NULL && strcmp(last + 1, vector) == 0) {
			irq = (unsigned)strtoul(line, NULL, 10);
			break;
		}
	}
	(void)fclose(interrupts);
	(void)snprintf(path, sizeof(path),
		       "/proc/irq/%u/effective_affinity_list", irq);
	FILE *const affinity = fopen(path, "re");
	if (affinity != NULL) {
		unsigned cpu = 0;
		if (fscanf(affinity, "%u", &cpu) == 1)
			store->irq_cpu[copy] = cpu;
		(void)fclose(affinity);
	}
}

static int store_files_open(struct bench_store *store, const char *const *paths,
			    uint32_t count)
{
	store->copy_count = count;
	for (uint32_t copy = 0; copy < count; copy++) {
		(void)snprintf(store->paths[copy], sizeof(store->paths[copy]),
			       "%s", paths[copy]);
		store->direct[copy] =
			open(paths[copy], O_RDONLY | O_DIRECT | O_CLOEXEC);
		store->buffered[copy] = open(paths[copy], O_RDONLY | O_CLOEXEC);
		store->passthrough[copy] = -1;
		struct stat file;
		if (store->direct[copy] < 0 || store->buffered[copy] < 0 ||
		    fstat(store->direct[copy], &file) != 0)
			return errno;
		if (copy == 0)
			store->file_bytes = (uint64_t)file.st_size;
		else if ((uint64_t)file.st_size != store->file_bytes)
			return EINVAL;
		device_of(store, copy);
	}
	return 0;
}

struct pack_source {
	struct storage_reader *reader;
	struct pack_record *pack;
	struct engram_record *engram;
};

static int store_pack_open(struct bench_store *store,
			   struct pack_source *source, const char *const *paths,
			   uint32_t count)
{
	result_status status =
		storage_direct.reader_open(paths, count, &source->reader);
	status = status == STATUS_OK ?
			 pack_open_entry(&storage_direct, source->reader,
					 &source->pack) :
			 status;
	status = status == STATUS_OK ?
			 engram_open(&storage_direct, source->reader,
				     source->pack, &source->engram) :
			 status;
	if (status != STATUS_OK) {
		(void)fprintf(stderr, "engram_io_bench: pack: %s\n",
			      status_text(status));
		return EINVAL;
	}
	const struct engram_description *const description =
		engram_description_of(source->engram);
	if (description == NULL || description->table_count > BENCH_TABLES_MOST)
		return EINVAL;
	store->table_count = description->table_count;
	store->rows_per_token = description->rows_per_token;
	for (uint32_t table = 0; table < store->table_count; table++) {
		const struct pack_unit_record_2 *const unit = &pack_units_of(
			source->pack)[description->table_unit_index[table]];
		const struct pack_extent *const extent = &pack_extents_of(
			source->pack)[unit->first_extent_index];
		struct bench_table *const own = &store->tables[table];
		own->byte_offset = extent->byte_offset;
		own->block_count = extent->byte_count / BENCH_BLOCK_BYTES;
		own->row_count = description->row_count[table];
		own->extent_digest = extent->digest;
		own->block_digests =
			calloc(own->block_count, sizeof(*own->block_digests));
		if (own->block_digests == NULL)
			return ENOMEM;
	}
	return store_files_open(store, paths, count);
}

/* ---- the digest scan (startup option A: parallel, striped across copies, a shared piece queue) ------------- */

struct prep_shared {
	struct bench_store *store;
	uint64_t piece_bytes, pieces_of[BENCH_TABLES_MOST], piece_total;
	_Atomic uint64_t next_piece, digest_sums[BENCH_TABLES_MOST],
		copy_bytes[BENCH_COPIES_MOST];
	_Atomic int error;
	uint32_t threads_per_copy,
		only_copy; /* only_copy: UINT32_MAX for every copy */
	uint32_t only_table;
};
struct prep_worker {
	struct prep_shared *shared;
	uint32_t copy;
	pthread_t thread;
};

static void *prep_read(void *argument)
{
	struct prep_worker *const worker = argument;
	struct prep_shared *const shared = worker->shared;
	struct bench_store *const store = shared->store;
	unsigned char *const buffer =
		aligned_alloc(BENCH_BLOCK_BYTES, shared->piece_bytes);
	if (buffer == NULL) {
		atomic_store(&shared->error, ENOMEM);
		return NULL;
	}
	for (;;) {
		const uint64_t piece = atomic_fetch_add(&shared->next_piece, 1);
		if (piece >= shared->piece_total ||
		    atomic_load(&shared->error) != 0)
			break;
		uint32_t table = 0;
		uint64_t local = piece;
		while (table < store->table_count &&
		       local >= shared->pieces_of[table])
			local -= shared->pieces_of[table++];
		if (shared->only_table != UINT32_MAX &&
		    table != shared->only_table)
			continue;
		const struct bench_table *const own = &store->tables[table];
		const uint64_t offset = local * shared->piece_bytes,
			       extent_bytes =
				       own->block_count * BENCH_BLOCK_BYTES;
		const uint64_t bytes = extent_bytes - offset <
						       shared->piece_bytes ?
					       extent_bytes - offset :
					       shared->piece_bytes;
		for (uint64_t done = 0; done < bytes;) {
			const ssize_t amount = pread(
				store->direct[worker->copy], buffer + done,
				(size_t)(bytes - done),
				(off_t)(own->byte_offset + offset + done));
			if (amount < 0 && errno == EINTR)
				continue;
			if (amount <= 0) {
				atomic_store(&shared->error,
					     amount < 0 ? errno : EIO);
				break;
			}
			done += (uint64_t)amount;
		}
		if (atomic_load(&shared->error) != 0)
			break;
		atomic_fetch_add(
			&shared->digest_sums[table],
			pack_digest_of_piece(buffer, bytes,
					     offset / PACK_DIGEST_WORD_BYTES));
		for (uint64_t block = 0; block < bytes / BENCH_BLOCK_BYTES;
		     block++)
			own->block_digests[offset / BENCH_BLOCK_BYTES + block] =
				pack_digest_of_piece(
					buffer + block * BENCH_BLOCK_BYTES,
					BENCH_BLOCK_BYTES, 0);
		atomic_fetch_add(&shared->copy_bytes[worker->copy], bytes);
	}
	free(buffer);
	return NULL;
}

static int prep_pass(struct bench_store *store, uint32_t threads_per_copy,
		     uint64_t piece_bytes, uint32_t only_copy,
		     uint32_t only_table, struct prep_shared *shared)
{
	memset(shared, 0, sizeof(*shared));
	shared->store = store;
	shared->piece_bytes = piece_bytes;
	shared->threads_per_copy = threads_per_copy;
	shared->only_table = only_table;
	for (uint32_t table = 0; table < store->table_count; table++) {
		const uint64_t extent_bytes =
			store->tables[table].block_count * BENCH_BLOCK_BYTES;
		shared->pieces_of[table] =
			(extent_bytes + piece_bytes - 1u) / piece_bytes;
		shared->piece_total += shared->pieces_of[table];
	}
	struct prep_worker workers[PREP_THREADS_MOST];
	uint32_t started = 0;
	for (uint32_t copy = 0; copy < store->copy_count; copy++)
		for (uint32_t thread = 0;
		     thread < threads_per_copy && started < PREP_THREADS_MOST;
		     thread++) {
			if (only_copy != UINT32_MAX && copy != only_copy)
				continue;
			workers[started] =
				(struct prep_worker){ .shared = shared,
						      .copy = copy };
			if (pthread_create(&workers[started].thread, NULL,
					   prep_read, &workers[started]) != 0)
				return EAGAIN;
			started++;
		}
	for (uint32_t index = 0; index < started; index++)
		(void)pthread_join(workers[index].thread, NULL);
	return atomic_load(&shared->error);
}

/* Every block digest, from all copies at once; a table whose extent digest fails is scanned again from each copy
 * alone until one gives it (as engram_disk's table_prepare does). Prints one JSON line about the scan. */
static int store_digests(struct bench_store *store, uint32_t threads_per_copy,
			 uint64_t piece_bytes, FILE *output)
{
	struct prep_shared shared;
	const uint64_t started = bench_now();
	int error = prep_pass(store, threads_per_copy, piece_bytes, UINT32_MAX,
			      UINT32_MAX, &shared);
	const uint64_t elapsed = bench_now() - started;
	uint64_t bytes = 0, copy_bytes[BENCH_COPIES_MOST] = { 0 };
	for (uint32_t copy = 0; copy < store->copy_count; copy++)
		bytes += copy_bytes[copy] =
			atomic_load(&shared.copy_bytes[copy]);
	uint32_t rescans = 0;
	for (uint32_t table = 0; error == 0 && table < store->table_count;
	     table++) {
		bool good = atomic_load(&shared.digest_sums[table]) ==
			    store->tables[table].extent_digest;
		for (uint32_t copy = 0;
		     !good && copy < store->copy_count && error == 0; copy++) {
			struct prep_shared again;
			rescans++;
			error = prep_pass(store, threads_per_copy, piece_bytes,
					  copy, table, &again);
			good = error == 0 &&
			       atomic_load(&again.digest_sums[table]) ==
				       store->tables[table].extent_digest;
		}
		if (!good && error == 0)
			error = EBADMSG;
	}
	(void)fprintf(
		output,
		"{\"schema\":1,\"kind\":\"digests\",\"status\":\"%s\",\"error\":\"%s\",\"threads_per_copy\":%u,"
		"\"piece_bytes\":%" PRIu64 ",\"bytes\":%" PRIu64
		",\"seconds\":%.3f,\"gb_per_second\":%.3f,"
		"\"rescans\":%u,\"copy_bytes\":[",
		error == 0 ? "ok" : "failed",
		error == 0 || strerrorname_np(error) == NULL ?
			"" :
			strerrorname_np(error),
		threads_per_copy, piece_bytes, bytes,
		(double)elapsed / (double)BENCH_NS_PER_SECOND,
		elapsed != 0 ? (double)bytes / (double)elapsed : 0.0, rescans);
	for (uint32_t copy = 0; copy < store->copy_count; copy++)
		(void)fprintf(output, "%s%" PRIu64, copy == 0 ? "" : ",",
			      copy_bytes[copy]);
	(void)fprintf(output, "]}\n");
	(void)fflush(output);
	return error;
}

/* ---- traces ------------------------------------------------------------------------------------------------ */

struct bench_case {
	uint32_t token_count, prompt;
	uint64_t *row_ids; /* token_count × tables × rows per token */
};
struct oracle {
	uint64_t shift_blocks, capacity, entries;
	uint64_t *keys;
	uint32_t *slot_of;
	unsigned char *arena;
	uint16_t *counts; /* entries × cases: references per case, saturating */
	uint8_t *hot;
	/* rows, for row-unit hot sets */
	uint64_t row_capacity, row_entries;
	uint64_t *row_keys;
	uint16_t *row_counts;
	uint8_t *row_hot;
	double load_seconds;
	uint64_t load_reads, load_retries;
};
struct bench {
	struct bench_store store;
	struct pack_source source;
	const char *hot_directory; /* where a hot_file arm's file is */
	struct bench_case cases[BENCH_CASES_MOST];
	uint32_t case_count;
	struct oracle oracle;
	struct engram_disk *prod;
	uint32_t prod_ram_percent;
	uint64_t prod_cache_bytes;
	uint32_t prod_interleave;
	FILE *output;
	uint64_t memory_floor_bytes;
};

static int cases_read(struct bench *bench, const char *list_path)
{
	FILE *const list = fopen(list_path, "re");
	if (list == NULL)
		return errno;
	char line[LIST_LINE_BYTES];
	const uint32_t row_width =
		bench->store.table_count * bench->store.rows_per_token;
	int error = 0;
	while (error == 0 && fgets(line, sizeof(line), list) != NULL) {
		char path[PATH_MAX];
		unsigned prompt = 0;
		if (line[0] == '#' || line[0] == '\n')
			continue;
		if (sscanf(line, "%4095s %u", path, &prompt) != 2 ||
		    bench->case_count == BENCH_CASES_MOST) {
			error = EINVAL;
			break;
		}
		const int descriptor = open(path, O_RDONLY | O_CLOEXEC);
		struct stat file;
		if (descriptor < 0 || fstat(descriptor, &file) != 0 ||
		    file.st_size <= 0 ||
		    (uint64_t)file.st_size % sizeof(uint32_t) != 0 ||
		    (uint64_t)file.st_size / sizeof(uint32_t) > TOKENS_MOST) {
			error = descriptor < 0 ? errno : EINVAL;
			if (descriptor >= 0)
				(void)close(descriptor);
			break;
		}
		struct bench_case *const own = &bench->cases[bench->case_count];
		own->token_count =
			(uint32_t)((uint64_t)file.st_size / sizeof(uint32_t));
		own->prompt = prompt != 0 && prompt <= own->token_count ?
				      prompt :
				      own->token_count;
		uint32_t *const tokens = malloc((size_t)file.st_size);
		own->row_ids = calloc((size_t)own->token_count * row_width,
				      sizeof(*own->row_ids));
		const bool read_all =
			tokens != NULL && own->row_ids != NULL &&
			pread(descriptor, tokens, (size_t)file.st_size, 0) ==
				file.st_size;
		(void)close(descriptor);
		for (uint32_t position = 0;
		     read_all && position < own->token_count; position++)
			if (engram_row_ids_entry(bench->source.engram, tokens,
						 NULL, position,
						 own->row_ids +
							 (size_t)position *
								 row_width) !=
			    STATUS_OK)
				error = EINVAL;
		free(tokens);
		if (!read_all)
			error = EIO;
		bench->case_count++;
	}
	(void)fclose(list);
	return error != 0 ? error : bench->case_count != 0 ? 0 : EINVAL;
}

static uint64_t shifted_row(const struct bench_store *store, uint32_t table,
			    uint64_t row, uint64_t shift_blocks)
{
	return shift_blocks == 0 ?
		       row :
		       (row + shift_blocks * ENGRAM_ROWS_PER_BLOCK %
				      store->tables[table].row_count) %
			       store->tables[table].row_count;
}

/* Positions a level reads of one case: the prompt (prefill) and up to DECODE_STEPS decode tokens after it. */
static uint32_t positions_used(const struct bench_case *own)
{
	const uint32_t decode_end = own->prompt - 1u + DECODE_STEPS;
	return decode_end < own->token_count ? decode_end : own->token_count;
}

/* ---- the oracle: every block the trace touches, verified, in RAM; counts per case for hot sets ------------ */

static uint64_t hash_slot(uint64_t key, uint64_t capacity)
{
	return (key * HASH_MULTIPLIER) >>
	       (64u - (uint32_t)__builtin_ctzll(capacity));
}

static uint64_t map_find(const uint64_t *keys, uint64_t capacity, uint64_t key,
			 bool *found)
{
	uint64_t slot = hash_slot(key, capacity);
	while (keys[slot] != KEY_EMPTY && keys[slot] != key)
		slot = (slot + 1u) & (capacity - 1u);
	*found = keys[slot] == key;
	return slot;
}

static uint64_t block_key(uint32_t table, uint64_t block)
{
	return ((uint64_t)table << KEY_TABLE_SHIFT | block) + 1u;
}

struct oracle_loader {
	struct bench *bench;
	uint64_t *order; /* map slots to load */
	uint64_t count;
	_Atomic uint64_t next, retries;
	_Atomic int error;
};
struct oracle_worker {
	struct oracle_loader *loader;
	uint32_t copy;
	pthread_t thread;
};

static void *oracle_read(void *argument)
{
	struct oracle_worker *const worker = argument;
	struct oracle_loader *const loader = worker->loader;
	struct bench *const bench = loader->bench;
	struct oracle *const oracle = &bench->oracle;
	for (;;) {
		const uint64_t index = atomic_fetch_add(&loader->next, 1);
		if (index >= loader->count || atomic_load(&loader->error) != 0)
			break;
		const uint64_t slot = loader->order[index],
			       key = oracle->keys[slot] - 1u;
		const uint32_t table = (uint32_t)(key >> KEY_TABLE_SHIFT);
		const uint64_t block = key & ((1ull << KEY_TABLE_SHIFT) - 1u);
		unsigned char *const bytes =
			oracle->arena +
			(uint64_t)oracle->slot_of[slot] * BENCH_BLOCK_BYTES;
		bool good = false;
		for (uint32_t attempt = 0;
		     !good && attempt < bench->store.copy_count * 2u;
		     attempt++) {
			const uint32_t copy = (worker->copy + attempt) %
					      bench->store.copy_count;
			good = pread(bench->store.direct[copy], bytes,
				     BENCH_BLOCK_BYTES,
				     (off_t)bench_copy_offset(&bench->store,
							      table, block)) ==
				       (ssize_t)BENCH_BLOCK_BYTES &&
			       bench_block_verified(&bench->store, table, block,
						    bytes);
			if (!good)
				atomic_fetch_add(&loader->retries, 1);
		}
		if (!good)
			atomic_store(&loader->error, EBADMSG);
	}
	return NULL;
}

static void oracle_free(struct oracle *oracle)
{
	free(oracle->keys);
	free(oracle->slot_of);
	if (oracle->arena != NULL)
		(void)munmap(oracle->arena,
			     oracle->entries * BENCH_BLOCK_BYTES);
	free(oracle->counts);
	free(oracle->hot);
	free(oracle->row_keys);
	free(oracle->row_counts);
	free(oracle->row_hot);
	memset(oracle, 0, sizeof(*oracle));
}

static int oracle_build(struct bench *bench, uint64_t shift_blocks, bool load)
{
	struct oracle *const oracle = &bench->oracle;
	if (oracle->arena != NULL && oracle->shift_blocks == shift_blocks)
		return 0;
	oracle_free(oracle);
	oracle->shift_blocks = shift_blocks;
	const struct bench_store *const store = &bench->store;
	const uint32_t width = store->table_count * store->rows_per_token;
	uint64_t references = 0;
	for (uint32_t index = 0; index < bench->case_count; index++)
		references +=
			(uint64_t)positions_used(&bench->cases[index]) * width;
	oracle->capacity = 1;
	while (oracle->capacity < references * 2u)
		oracle->capacity <<= 1u;
	oracle->row_capacity = oracle->capacity;
	oracle->keys = calloc(oracle->capacity, sizeof(*oracle->keys));
	oracle->slot_of = calloc(oracle->capacity, sizeof(*oracle->slot_of));
	oracle->row_keys =
		calloc(oracle->row_capacity, sizeof(*oracle->row_keys));
	if (oracle->keys == NULL || oracle->slot_of == NULL ||
	    oracle->row_keys == NULL)
		return ENOMEM;
	for (uint32_t index = 0; index < bench->case_count; index++) {
		const struct bench_case *const own = &bench->cases[index];
		for (uint64_t reference = 0;
		     reference < (uint64_t)positions_used(own) * width;
		     reference++) {
			const uint32_t table =
				(uint32_t)(reference % width /
					   store->rows_per_token);
			const uint64_t row = shifted_row(
				store, table, own->row_ids[reference],
				shift_blocks);
			bool found = false;
			const uint64_t slot = map_find(
				oracle->keys, oracle->capacity,
				block_key(table, row / ENGRAM_ROWS_PER_BLOCK),
				&found);
			if (!found) {
				oracle->keys[slot] = block_key(
					table, row / ENGRAM_ROWS_PER_BLOCK);
				oracle->slot_of[slot] =
					(uint32_t)oracle->entries++;
			}
			const uint64_t row_slot =
				map_find(oracle->row_keys, oracle->row_capacity,
					 block_key(table, row), &found);
			if (!found) {
				oracle->row_keys[row_slot] =
					block_key(table, row);
				oracle->row_entries++;
			}
		}
	}
	const uint64_t arena_bytes = oracle->entries * BENCH_BLOCK_BYTES;
	if (!load)
		return 0;
	if (memory_available_bytes() < arena_bytes + bench->memory_floor_bytes)
		return ENOMEM;
	oracle->arena = mmap(NULL, arena_bytes, PROT_READ | PROT_WRITE,
			     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
	oracle->counts = calloc(oracle->entries * bench->case_count,
				sizeof(*oracle->counts));
	oracle->hot = calloc(oracle->capacity, 1);
	oracle->row_counts = calloc(oracle->row_capacity * bench->case_count,
				    sizeof(*oracle->row_counts));
	oracle->row_hot = calloc(oracle->row_capacity, 1);
	if (oracle->arena == MAP_FAILED || oracle->counts == NULL ||
	    oracle->hot == NULL || oracle->row_counts == NULL ||
	    oracle->row_hot == NULL) {
		if (oracle->arena == MAP_FAILED)
			oracle->arena = NULL;
		return ENOMEM;
	}
	(void)madvise(oracle->arena, arena_bytes, MADV_HUGEPAGE);
	/* counts: block slots by index of entry; rows by row slot */
	for (uint32_t index = 0; index < bench->case_count; index++) {
		const struct bench_case *const own = &bench->cases[index];
		for (uint64_t reference = 0;
		     reference < (uint64_t)positions_used(own) * width;
		     reference++) {
			const uint32_t table =
				(uint32_t)(reference % width /
					   store->rows_per_token);
			const uint64_t row = shifted_row(
				store, table, own->row_ids[reference],
				shift_blocks);
			bool found = false;
			const uint64_t slot = map_find(
				oracle->keys, oracle->capacity,
				block_key(table, row / ENGRAM_ROWS_PER_BLOCK),
				&found);
			uint16_t *const count =
				&oracle->counts[(uint64_t)oracle->slot_of[slot] *
							bench->case_count +
						index];
			*count = *count != UINT16_MAX ?
					 (uint16_t)(*count + 1u) :
					 *count;
			const uint64_t row_slot =
				map_find(oracle->row_keys, oracle->row_capacity,
					 block_key(table, row), &found);
			uint16_t *const row_count =
				&oracle->row_counts[row_slot * bench->case_count +
						    index];
			*row_count = *row_count != UINT16_MAX ?
					     (uint16_t)(*row_count + 1u) :
					     *row_count;
		}
	}
	struct oracle_loader loader = { .bench = bench,
					.count = oracle->entries };
	loader.order = calloc(oracle->entries != 0 ? oracle->entries : 1u,
			      sizeof(*loader.order));
	if (loader.order == NULL)
		return ENOMEM;
	for (uint64_t slot = 0, made = 0; slot < oracle->capacity; slot++)
		if (oracle->keys[slot] != KEY_EMPTY)
			loader.order[made++] = slot;
	const uint64_t started = bench_now();
	struct oracle_worker
		workers[BENCH_COPIES_MOST * ORACLE_LOADERS_PER_COPY];
	uint32_t launched = 0;
	for (uint32_t copy = 0; copy < store->copy_count; copy++)
		for (uint32_t thread = 0; thread < ORACLE_LOADERS_PER_COPY;
		     thread++) {
			workers[launched] =
				(struct oracle_worker){ .loader = &loader,
							.copy = copy };
			if (pthread_create(&workers[launched].thread, NULL,
					   oracle_read,
					   &workers[launched]) == 0)
				launched++;
		}
	for (uint32_t index = 0; index < launched; index++)
		(void)pthread_join(workers[index].thread, NULL);
	oracle->load_seconds =
		(double)(bench_now() - started) / (double)BENCH_NS_PER_SECOND;
	oracle->load_reads = oracle->entries;
	oracle->load_retries = atomic_load(&loader.retries);
	free(loader.order);
	return atomic_load(&loader.error);
}

static const unsigned char *oracle_block(const struct oracle *oracle,
					 uint32_t table, uint64_t block,
					 uint64_t *slot)
{
	bool found = false;
	*slot = map_find(oracle->keys, oracle->capacity,
			 block_key(table, block), &found);
	return found ? oracle->arena + (uint64_t)oracle->slot_of[*slot] *
					       BENCH_BLOCK_BYTES :
		       NULL;
}

/* A hot_file arm: the blocks its file holds (made by hot_build from another corpus, checked as engram_hot_open checks
 * it) marked among the run's blocks. 0, or EINVAL with the reason. */
static int hot_from_file(struct bench *bench, const struct arm *arm,
			 char *reason, size_t reason_bytes)
{
	struct oracle *const oracle = &bench->oracle;
	/* raw files, or the self-test's synthetic copies, have no pack */
	if (bench->source.pack == NULL) {
		(void)snprintf(reason, reason_bytes,
			       "hot set: a hot_file arm needs the pack");
		return EINVAL;
	}
	char path[PATH_MAX];
	(void)snprintf(path, sizeof(path), "%s/%s",
		       bench->hot_directory != NULL ? bench->hot_directory :
						      ".",
		       arm->hot_file);
	struct engram_hot_table expected[ENGRAM_MAXIMUM_TABLES] = { 0 };
	for (uint32_t table = 0; table < bench->store.table_count; table++)
		expected[table] = (struct engram_hot_table){
			.table = table,
			.extent_digest =
				bench->store.tables[table].extent_digest,
			.block_count = bench->store.tables[table].block_count
		};
	uint64_t *ids[ENGRAM_MAXIMUM_TABLES] = { NULL },
		 counts[ENGRAM_MAXIMUM_TABLES] = { 0 };
	char refusal[ENGRAM_HOT_REASON_BYTES] = "";
	if (engram_hot_read(path, arm->hot_sha256, arm->hot_bytes,
			    pack_header_of(bench->source.pack), expected,
			    bench->store.table_count, ids, counts,
			    refusal) != STATUS_OK) {
		(void)snprintf(reason, reason_bytes, "hot set: %s", refusal);
		return EINVAL;
	}
	memset(oracle->hot, 0, oracle->capacity);
	for (uint32_t table = 0; table < ENGRAM_MAXIMUM_TABLES; table++) {
		for (uint64_t index = 0; index < counts[table]; index++) {
			/* the repeat's shift, as the run's rows have it */
			const uint64_t block =
				shifted_row(&bench->store, table,
					    ids[table][index] *
						    ENGRAM_ROWS_PER_BLOCK,
					    oracle->shift_blocks) /
				ENGRAM_ROWS_PER_BLOCK;
			bool found = false;
			const uint64_t slot =
				map_find(oracle->keys, oracle->capacity,
					 block_key(table, block), &found);
			if (found)
				oracle->hot[slot] = 1;
		}
		free(ids[table]);
	}
	return 0;
}

/* Marks the hot set for one group: the most referenced blocks (or rows) of the profile cases, up to the arm's bytes.
 * profile "other" counts only cases outside the group, so a hot set never knows the rows it is measured on. A
 * hot_file arm takes its file's blocks instead (checked at the level's start). */
static void hot_select(struct bench *bench, const struct arm *arm,
		       uint32_t group_cases_mask)
{
	char unused[BENCH_REASON_BYTES];
	if (arm->hot_file[0] != '\0') {
		(void)hot_from_file(bench, arm, unused, sizeof(unused));
		return;
	}
	struct oracle *const oracle = &bench->oracle;
	const bool rows = arm->hot_unit == HOT_ROW;
	const uint64_t capacity = rows ? oracle->row_capacity :
					 oracle->capacity;
	uint8_t *const hot = rows ? oracle->row_hot : oracle->hot;
	memset(hot, 0, capacity);
	const uint64_t unit_bytes = rows ? ENGRAM_ROW_BYTES : BENCH_BLOCK_BYTES;
	uint64_t wanted = arm->hot_bytes / unit_bytes;
	if (wanted == 0)
		return;
	static uint64_t histogram[SCORE_CAP + 1u];
	memset(histogram, 0, sizeof(histogram));
	const uint64_t *const keys = rows ? oracle->row_keys : oracle->keys;
	for (uint32_t pass = 0; pass < 2u; pass++) {
		uint64_t threshold = 0, at_threshold = 0;
		if (pass == 1u) {
			uint64_t taken = 0;
			for (uint64_t score = SCORE_CAP; score > 0; score--) {
				if (taken + histogram[score] >= wanted) {
					threshold = score;
					at_threshold = wanted - taken;
					break;
				}
				taken += histogram[score];
				threshold = score;
				at_threshold = histogram[score];
			}
		}
		for (uint64_t slot = 0; slot < capacity; slot++) {
			if (keys[slot] == KEY_EMPTY)
				continue;
			const uint16_t *const counts =
				rows ? &oracle->row_counts[slot *
							   bench->case_count] :
				       &oracle->counts[(uint64_t)oracle
							       ->slot_of[slot] *
						       bench->case_count];
			uint64_t score = 0;
			for (uint32_t index = 0; index < bench->case_count;
			     index++)
				if (arm->hot_profile == PROFILE_ALL ||
				    (group_cases_mask >> index & 1u) == 0)
					score += counts[index];
			score = score < SCORE_CAP ? score : SCORE_CAP;
			if (pass == 0)
				histogram[score]++;
			else if (score > threshold ||
				 (score == threshold && score != 0 &&
				  at_threshold-- != 0))
				hot[slot] = 1;
		}
	}
}

/* ---- the direct-mapped dynamic cache (the production slot rule) ------------------------------------------- */

struct block_cache {
	uint64_t slots, *tags;
	unsigned char *data;
};

static int cache_open(struct block_cache *cache, uint64_t bytes)
{
	memset(cache, 0, sizeof(*cache));
	cache->slots = bytes / BENCH_BLOCK_BYTES;
	if (cache->slots == 0)
		return 0;
	cache->tags = calloc(cache->slots, sizeof(*cache->tags));
	cache->data = mmap(NULL, cache->slots * BENCH_BLOCK_BYTES,
			   PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS,
			   -1, 0);
	if (cache->tags == NULL || cache->data == MAP_FAILED) {
		cache->data = NULL;
		return ENOMEM;
	}
	(void)madvise(cache->data, cache->slots * BENCH_BLOCK_BYTES,
		      MADV_HUGEPAGE);
	memset(cache->data, 0,
	       cache->slots *
		       BENCH_BLOCK_BYTES); /* first touch here, never inside a timed step */
	return 0;
}

static void cache_close(struct block_cache *cache)
{
	free(cache->tags);
	if (cache->data != NULL)
		(void)munmap(cache->data, cache->slots * BENCH_BLOCK_BYTES);
	memset(cache, 0, sizeof(*cache));
}

static uint64_t cache_slot(const struct block_cache *cache, uint32_t table,
			   uint64_t block, uint32_t tables)
{
	return (block * tables + table) % cache->slots;
}

/* ---- one level of one arm --------------------------------------------------------------------------------- */

struct level_samples {
	uint64_t *decode, *ready[BENCH_TABLES_MOST],
		*exposed[BENCH_TABLES_MOST], *prefill, *scratch;
	uint32_t decode_count, prefill_count, decode_capacity, prefill_capacity;
	uint64_t decode_rows, prefill_rows, decode_wall, prefill_wall,
		mismatched_rows;
};
struct step_scratch {
	uint64_t *rows; /* table-major references */
	uint32_t *unique_of; /* per reference: its unique block, or RESOLVED */
	uint64_t *dedupe_keys;
	uint32_t *dedupe_values, *dedupe_stamps, stamp, stamp_uniques;
	uint64_t dedupe_capacity;
	struct logical_read **unique_reads;
	const unsigned char **
		unique_sources; /* a unique block served from RAM or the cache: where it lies */
	uint32_t read_first[BENCH_TABLES_MOST], read_count[BENCH_TABLES_MOST];
	unsigned char *staging;
	uint32_t references_most;
};
struct level_context {
	struct bench *bench;
	const struct arm *arm;
	uint32_t level, repeat;
	uint64_t shift_blocks;
	struct bench_run run;
	struct block_cache cache;
	struct step_scratch scratch;
	struct level_samples samples;
	int error;
	char reason[BENCH_REASON_BYTES];
	uint64_t step_cpu_ns, voluntary, involuntary;
	struct bench_counters local;
};
struct step_token {
	uint32_t case_index, position;
};

static uint32_t dedupe_find(struct step_scratch *scratch, uint64_t key,
			    bool *found)
{
	uint64_t slot = hash_slot(key, scratch->dedupe_capacity);
	while (scratch->dedupe_stamps[slot] == scratch->stamp &&
	       scratch->dedupe_keys[slot] != key)
		slot = (slot + 1u) & (scratch->dedupe_capacity - 1u);
	*found = scratch->dedupe_stamps[slot] == scratch->stamp;
	return (uint32_t)slot;
}

static void spin_until(uint64_t when)
{
	while (bench_now() < when) {
#if defined(__x86_64__)
		__builtin_ia32_pause();
#endif
	}
}

/* Plans references [first, first + count) of one table: rows served from RAM (prefix, hot set, all RAM) or the
 * cache are copied now; the other blocks become reads (deduplicated when the arm says so). */
static void plan_references(struct level_context *context, uint32_t table,
			    uint32_t first, uint32_t count,
			    uint32_t token_count)
{
	struct bench *const bench = context->bench;
	const struct arm *const arm = context->arm;
	struct step_scratch *const scratch = &context->scratch;
	const struct bench_store *const store = &bench->store;
	const struct oracle *const oracle = &bench->oracle;
	const uint64_t prefix_blocks =
		store->tables[table].block_count * arm->ram_percent / PERCENT;
	unsigned char *const staging =
		scratch->staging + (uint64_t)table * token_count *
					   store->rows_per_token *
					   ENGRAM_ROW_BYTES;
	const uint32_t base = table * token_count * store->rows_per_token;
	for (uint32_t reference = first; reference < first + count;
	     reference++) {
		const uint64_t row = scratch->rows[base + reference],
			       block = row / ENGRAM_ROWS_PER_BLOCK;
		unsigned char *const output =
			staging + (uint64_t)reference * ENGRAM_ROW_BYTES;
		const uint64_t row_offset =
			row % ENGRAM_ROWS_PER_BLOCK * ENGRAM_ROW_BYTES;
		scratch->unique_of[base + reference] = RESOLVED;
		atomic_fetch_add_explicit(&context->local.references, 1,
					  memory_order_relaxed);
		if (arm->hot_bytes != 0 && arm->hot_unit == HOT_ROW) {
			bool found = false;
			const uint64_t slot =
				map_find(oracle->row_keys, oracle->row_capacity,
					 block_key(table, row), &found);
			if (found && oracle->row_hot[slot] != 0) {
				uint64_t block_slot = 0;
				memcpy(output,
				       oracle_block(oracle, table, block,
						    &block_slot) +
					       row_offset,
				       ENGRAM_ROW_BYTES);
				atomic_fetch_add_explicit(
					&context->local.hot_hits, 1,
					memory_order_relaxed);
				continue;
			}
		}
		uint32_t unique = UINT32_MAX;
		bool found = false;
		uint32_t dedupe_slot = 0;
		if (arm->dedupe) {
			dedupe_slot = dedupe_find(
				scratch, block_key(table, block), &found);
			if (found) {
				unique = scratch->dedupe_values[dedupe_slot];
				if (scratch->unique_reads[unique] == NULL)
					memcpy(output,
					       scratch->unique_sources[unique] +
						       row_offset,
					       ENGRAM_ROW_BYTES);
				else
					scratch->unique_of[base + reference] =
						unique;
				continue;
			}
		}
		/* a new block of this step */
		uint64_t block_slot = 0;
		const unsigned char *const known =
			oracle_block(oracle, table, block, &block_slot);
		const unsigned char *source = NULL;
		if (arm->all_ram || block < prefix_blocks) {
			source = known;
			atomic_fetch_add_explicit(&context->local.ram_hits, 1,
						  memory_order_relaxed);
		} else if (arm->hot_bytes != 0 && arm->hot_unit == HOT_BLOCK &&
			   oracle->hot[block_slot] != 0) {
			source = known;
			atomic_fetch_add_explicit(&context->local.hot_hits, 1,
						  memory_order_relaxed);
		} else if (context->cache.slots != 0) {
			const uint64_t slot = cache_slot(&context->cache, table,
							 block,
							 store->table_count);
			if (context->cache.tags[slot] ==
			    block_key(table, block)) {
				source = context->cache.data +
					 slot * BENCH_BLOCK_BYTES;
				atomic_fetch_add_explicit(
					&context->local.cache_hits, 1,
					memory_order_relaxed);
			}
		}
		struct logical_read *read = NULL;
		if (source == NULL) {
			read = bench_read_add(&context->run, table, block, 0);
			if (read == NULL)
				return;
			atomic_fetch_add_explicit(&context->local.unique_blocks,
						  1, memory_order_relaxed);
		} else {
			memcpy(output, source + row_offset, ENGRAM_ROW_BYTES);
		}
		if (arm->dedupe) {
			unique = scratch->stamp_uniques++;
			scratch->dedupe_keys[dedupe_slot] =
				block_key(table, block);
			scratch->dedupe_values[dedupe_slot] = unique;
			scratch->dedupe_stamps[dedupe_slot] = scratch->stamp;
		} else {
			unique = scratch->stamp_uniques++;
		}
		scratch->unique_reads[unique] = read;
		scratch->unique_sources[unique] = source;
		if (read != NULL)
			scratch->unique_of[base + reference] = unique;
	}
}

/* After a table's reads all landed: fill the cache and copy the rows waiting for those blocks. */
static void consume_table(struct level_context *context, uint32_t table,
			  uint32_t first, uint32_t count, uint32_t token_count)
{
	struct step_scratch *const scratch = &context->scratch;
	const struct bench_store *const store = &context->bench->store;
	const uint32_t base = table * token_count * store->rows_per_token;
	unsigned char *const staging =
		scratch->staging + (uint64_t)table * token_count *
					   store->rows_per_token *
					   ENGRAM_ROW_BYTES;
	for (uint32_t reference = first; reference < first + count;
	     reference++) {
		const uint32_t unique = scratch->unique_of[base + reference];
		if (unique == RESOLVED)
			continue;
		struct logical_read *const read = scratch->unique_reads[unique];
		const struct attempt *const winner = atomic_load(&read->winner);
		if (winner == NULL)
			continue; /* failed: the run's error says so */
		const uint64_t row = scratch->rows[base + reference];
		memcpy(staging + (uint64_t)reference * ENGRAM_ROW_BYTES,
		       winner->buffer +
			       row % ENGRAM_ROWS_PER_BLOCK * ENGRAM_ROW_BYTES,
		       ENGRAM_ROW_BYTES);
	}
	if (context->cache.slots == 0)
		return;
	for (uint32_t unique = 0; unique < scratch->stamp_uniques; unique++) {
		struct logical_read *const read = scratch->unique_reads[unique];
		if (read == NULL || read->table != table)
			continue;
		const struct attempt *const winner = atomic_load(&read->winner);
		if (winner == NULL)
			continue;
		const uint64_t slot = cache_slot(&context->cache, table,
						 read->block,
						 store->table_count);
		memcpy(context->cache.data + slot * BENCH_BLOCK_BYTES,
		       winner->buffer, BENCH_BLOCK_BYTES);
		context->cache.tags[slot] = block_key(table, read->block);
	}
}

static void check_rows(struct level_context *context, uint32_t token_count)
{
	const struct bench_store *const store = &context->bench->store;
	const struct step_scratch *const scratch = &context->scratch;
	const uint32_t references = token_count * store->rows_per_token;
	for (uint32_t table = 0; table < store->table_count; table++)
		for (uint32_t reference = 0; reference < references;
		     reference++) {
			const uint64_t row =
				scratch->rows[table * references + reference];
			uint64_t slot = 0;
			const unsigned char *const block = oracle_block(
				&context->bench->oracle, table,
				row / ENGRAM_ROWS_PER_BLOCK, &slot);
			if (block == NULL ||
			    memcmp(scratch->staging +
					   ((uint64_t)table * references +
					    reference) *
						   ENGRAM_ROW_BYTES,
				   block + row % ENGRAM_ROWS_PER_BLOCK *
						   ENGRAM_ROW_BYTES,
				   ENGRAM_ROW_BYTES) != 0)
				context->samples.mismatched_rows++;
		}
}

/* One step (a decode step of C rows, or a prefill chunk): returns its time; ready and exposed per table. */
static uint64_t step_run(struct level_context *context,
			 const struct step_token *tokens, uint32_t token_count,
			 uint64_t *ready, uint64_t *exposed)
{
	struct bench *const bench = context->bench;
	const struct arm *const arm = context->arm;
	const struct bench_store *const store = &bench->store;
	struct step_scratch *const scratch = &context->scratch;
	const uint32_t width = store->table_count * store->rows_per_token;
	const uint32_t references = token_count * store->rows_per_token;
	for (uint32_t table = 0; table < store->table_count; table++)
		for (uint32_t token = 0; token < token_count; token++) {
			const struct bench_case *const own =
				&bench->cases[tokens[token].case_index];
			for (uint32_t row = 0; row < store->rows_per_token;
			     row++)
				scratch->rows[table * references +
					      token * store->rows_per_token +
					      row] =
					shifted_row(
						store, table,
						own->row_ids
							[(uint64_t)tokens[token]
									 .position *
								 width +
							 table * store->rows_per_token +
							 row],
						context->shift_blocks);
		}
	const uint64_t started = bench_now();
	const uint64_t deadline = started + BENCH_READ_LIMIT_NS;
	if (arm->engine == ENGINE_PROD) {
		for (uint32_t table = 0; table < store->table_count; table++) {
			for (uint32_t token = 0; token < token_count; token++) {
				const result_status status = engram_disk_gather(
					bench->prod, table,
					scratch->rows + table * references +
						token * store->rows_per_token,
					store->rows_per_token,
					scratch->staging +
						((uint64_t)table * references +
						 token * store->rows_per_token) *
							ENGRAM_ROW_BYTES);
				if (status != STATUS_OK)
					bench_fail(&context->run,
						   status_text(status), EIO);
			}
			ready[table] = bench_now() - started;
			exposed[table] = ready[table];
		}
		return bench_now() - started;
	}
	if (arm->batch == BATCH_CALL) {
		for (uint32_t table = 0; table < store->table_count; table++) {
			const uint64_t table_started = bench_now();
			for (uint32_t token = 0;
			     token < token_count &&
			     atomic_load(&context->run.error) == 0;
			     token++) {
				scratch->stamp++;
				scratch->stamp_uniques = 0;
				const uint32_t first_read =
					context->run.epochs[context->run.epoch]
						.count;
				plan_references(context, table,
						token * store->rows_per_token,
						store->rows_per_token,
						token_count);
				const uint32_t made =
					context->run.epochs[context->run.epoch]
						.count -
					first_read;
				if (made != 0) {
					bench_reads_issue(&context->run,
							  first_read, made);
					(void)bench_wait_table(&context->run,
							       table, deadline);
				}
				consume_table(context, table,
					      token * store->rows_per_token,
					      store->rows_per_token,
					      token_count);
				bench_reads_release(&context->run);
			}
			ready[table] = bench_now() - started;
			exposed[table] = bench_now() - table_started;
		}
		return bench_now() - started;
	}
	scratch->stamp++;
	scratch->stamp_uniques = 0;
	for (uint32_t table = 0; table < store->table_count; table++) {
		scratch->read_first[table] =
			context->run.epochs[context->run.epoch].count;
		plan_references(context, table, 0, references, token_count);
		scratch->read_count[table] =
			context->run.epochs[context->run.epoch].count -
			scratch->read_first[table];
	}
	if (arm->batch == BATCH_STEP)
		bench_reads_issue(
			&context->run, 0,
			context->run.epochs[context->run.epoch].count);
	for (uint32_t table = 0; table < store->table_count; table++) {
		spin_until(started + (uint64_t)arm->lead_us[table] *
					     BENCH_NS_PER_MICROSECOND);
		if (arm->batch == BATCH_TABLE &&
		    scratch->read_count[table] != 0)
			bench_reads_issue(&context->run,
					  scratch->read_first[table],
					  scratch->read_count[table]);
		const uint64_t waited = bench_now();
		(void)bench_wait_table(&context->run, table, deadline);
		const uint64_t now = bench_now();
		ready[table] = now - started;
		exposed[table] = now - waited;
		consume_table(context, table, 0, references, token_count);
	}
	const uint64_t elapsed = bench_now() - started;
	bench_reads_release(&context->run);
	return elapsed;
}

static uint64_t steps_capacity(const struct bench *bench, uint32_t sessions,
			       uint32_t groups, uint32_t chunk,
			       uint64_t *prefill)
{
	*prefill = 0;
	for (uint32_t group = 0; group < groups; group++)
		for (uint32_t session = 0; session < sessions; session++) {
			const struct bench_case *const own =
				&bench->cases[(group * sessions + session) %
					      bench->case_count];
			*prefill += own->prompt > 1u ?
					    (own->prompt - 1u + chunk - 1u) /
						    chunk :
					    0u;
		}
	return (uint64_t)groups * DECODE_STEPS;
}

static void *level_main(void *argument)
{
	struct level_context *const context = argument;
	struct bench *const bench = context->bench;
	const struct arm *const arm = context->arm;
	const struct bench_store *const store = &bench->store;
	const uint32_t sessions = level_sessions[context->level];
	const uint32_t groups = (bench->case_count + sessions - 1u) / sessions;
	int32_t step_cpu = arm->step_cpu;
	if (step_cpu < 0 && arm->pin == PIN_IRQ && arm->io_threads == 0)
		step_cpu =
			(int32_t)store->irq_cpu[__builtin_ctz(arm->copy_mask)];
	const uint32_t chunk = arm->prefill_chunk;
	const uint32_t step_tokens = chunk > sessions ? chunk : sessions;
	const uint32_t references_most =
		step_tokens * store->rows_per_token * store->table_count;
	context->error =
		bench_run_open(&context->run, store, arm, references_most);
	bench_thread_attributes(&context->run, step_cpu);
	if (context->error != 0) {
		(void)snprintf(context->reason, sizeof(context->reason), "%s",
			       context->run.reason[0] != '\0' ?
				       context->run.reason :
				       "run open");
		bench_run_close(
			&context->run); /* stops any I/O thread it started */
		return NULL;
	}
	struct step_scratch *const scratch = &context->scratch;
	scratch->references_most = references_most;
	scratch->rows = calloc(references_most, sizeof(*scratch->rows));
	scratch->unique_of =
		calloc(references_most, sizeof(*scratch->unique_of));
	scratch->unique_reads =
		calloc(references_most, sizeof(*scratch->unique_reads));
	scratch->unique_sources =
		calloc(references_most, sizeof(*scratch->unique_sources));
	scratch->dedupe_capacity = 1;
	while (scratch->dedupe_capacity < (uint64_t)references_most * 2u)
		scratch->dedupe_capacity <<= 1u;
	scratch->dedupe_keys =
		calloc(scratch->dedupe_capacity, sizeof(*scratch->dedupe_keys));
	scratch->dedupe_values = calloc(scratch->dedupe_capacity,
					sizeof(*scratch->dedupe_values));
	scratch->dedupe_stamps = calloc(scratch->dedupe_capacity,
					sizeof(*scratch->dedupe_stamps));
	scratch->staging =
		aligned_alloc(BENCH_BLOCK_BYTES,
			      ((uint64_t)references_most * ENGRAM_ROW_BYTES +
			       BENCH_BLOCK_BYTES - 1u) /
				      BENCH_BLOCK_BYTES * BENCH_BLOCK_BYTES);
	uint64_t prefill_capacity = 0;
	const uint64_t decode_capacity = steps_capacity(
		bench, sessions, groups, chunk, &prefill_capacity);
	struct level_samples *const samples = &context->samples;
	samples->decode_capacity = (uint32_t)decode_capacity;
	samples->prefill_capacity = (uint32_t)prefill_capacity;
	samples->decode = calloc(decode_capacity + 1u, sizeof(uint64_t));
	samples->prefill = calloc(prefill_capacity + 1u, sizeof(uint64_t));
	samples->scratch = calloc(decode_capacity + prefill_capacity + 1u,
				  sizeof(uint64_t));
	bool allocated = scratch->rows != NULL && scratch->unique_of != NULL &&
			 scratch->unique_reads != NULL &&
			 scratch->unique_sources != NULL &&
			 scratch->dedupe_keys != NULL &&
			 scratch->dedupe_values != NULL &&
			 scratch->dedupe_stamps != NULL &&
			 scratch->staging != NULL && samples->decode != NULL &&
			 samples->prefill != NULL && samples->scratch != NULL;
	for (uint32_t table = 0; table < store->table_count; table++) {
		samples->ready[table] =
			calloc(decode_capacity + 1u, sizeof(uint64_t));
		samples->exposed[table] =
			calloc(decode_capacity + 1u, sizeof(uint64_t));
		allocated = allocated && samples->ready[table] != NULL &&
			    samples->exposed[table] != NULL;
	}
	if (!allocated || cache_open(&context->cache, arm->cache_bytes) != 0) {
		context->error = ENOMEM;
		(void)snprintf(context->reason, sizeof(context->reason),
			       "level memory");
	}
	const uint64_t cpu_started = bench_thread_cpu_ns();
	struct step_token tokens[BENCH_CHUNK_TOKENS_MOST];
	for (uint32_t group = 0; context->error == 0 && group < groups &&
				 atomic_load(&context->run.error) == 0;
	     group++) {
		uint32_t group_mask = 0;
		for (uint32_t session = 0; session < sessions; session++)
			group_mask |= 1u << ((group * sessions + session) %
					     bench->case_count);
		if (arm->hot_bytes != 0)
			hot_select(bench, arm, group_mask);
		for (uint32_t session = 0;
		     arm->prefill != 0 && session < sessions; session++) {
			const uint32_t case_index =
				(group * sessions + session) %
				bench->case_count;
			const struct bench_case *const own =
				&bench->cases[case_index];
			for (uint32_t begin = 0;
			     begin + 1u < own->prompt &&
			     atomic_load(&context->run.error) == 0;
			     begin += chunk) {
				const uint32_t count =
					own->prompt - 1u - begin < chunk ?
						own->prompt - 1u - begin :
						chunk;
				for (uint32_t token = 0; token < count; token++)
					tokens[token] = (struct step_token){
						case_index, begin + token
					};
				uint64_t ready[BENCH_TABLES_MOST],
					exposed[BENCH_TABLES_MOST];
				const uint64_t wall = bench_now();
				const uint64_t elapsed = step_run(
					context, tokens, count, ready, exposed);
				samples->prefill_wall += bench_now() - wall;
				if (samples->prefill_count <
				    samples->prefill_capacity)
					samples->prefill
						[samples->prefill_count++] =
						elapsed;
				samples->prefill_rows += (uint64_t)count *
							 store->rows_per_token *
							 store->table_count;
				check_rows(context, count);
			}
		}
		for (uint32_t step = 0; step < DECODE_STEPS &&
					atomic_load(&context->run.error) == 0;
		     step++) {
			uint32_t count = 0;
			for (uint32_t session = 0; session < sessions;
			     session++) {
				const uint32_t case_index =
					(group * sessions + session) %
					bench->case_count;
				const struct bench_case *const own =
					&bench->cases[case_index];
				const uint32_t position =
					own->prompt - 1u + step;
				if (position < own->token_count)
					tokens[count++] =
						(struct step_token){ case_index,
								     position };
			}
			if (count == 0)
				break;
			uint64_t ready[BENCH_TABLES_MOST],
				exposed[BENCH_TABLES_MOST];
			const uint64_t wall = bench_now();
			const uint64_t elapsed = step_run(
				context, tokens, count, ready, exposed);
			samples->decode_wall += bench_now() - wall;
			if (samples->decode_count < samples->decode_capacity) {
				for (uint32_t table = 0;
				     table < store->table_count; table++) {
					samples->ready[table]
						      [samples->decode_count] =
						ready[table];
					samples->exposed[table]
							[samples->decode_count] =
						exposed[table];
				}
				samples->decode[samples->decode_count++] =
					elapsed;
			}
			samples->decode_rows += (uint64_t)count *
						store->rows_per_token *
						store->table_count;
			check_rows(context, count);
		}
	}
	context->step_cpu_ns = bench_thread_cpu_ns() - cpu_started;
	struct rusage usage;
	if (getrusage(RUSAGE_THREAD, &usage) == 0) {
		context->voluntary = (uint64_t)usage.ru_nvcsw;
		context->involuntary = (uint64_t)usage.ru_nivcsw;
	}
	bench_run_close(&context->run);
	cache_close(&context->cache);
	free(scratch->rows);
	free(scratch->unique_of);
	free(scratch->unique_reads);
	free(scratch->unique_sources);
	free(scratch->dedupe_keys);
	free(scratch->dedupe_values);
	free(scratch->dedupe_stamps);
	free(scratch->staging);
	return NULL;
}

/* ---- device and interrupt counters ------------------------------------------------------------------------ */

struct device_counters {
	uint64_t reads[BENCH_COPIES_MOST], ticks_ms[BENCH_COPIES_MOST],
		interrupts[BENCH_COPIES_MOST];
};

static void counters_read(const struct bench_store *store,
			  struct device_counters *counters)
{
	memset(counters, 0, sizeof(*counters));
	for (uint32_t copy = 0; copy < store->copy_count; copy++) {
		char path[PATH_MAX];
		(void)snprintf(path, sizeof(path), "/sys/block/%s/stat",
			       store->device_names[copy]);
		FILE *const input = fopen(path, "re");
		unsigned long long fields[DEVICE_STAT_FIELDS] = { 0 };
		if (input != NULL) {
			for (uint32_t field = 0; field < DEVICE_STAT_FIELDS;
			     field++)
				if (fscanf(input, "%llu", &fields[field]) != 1)
					break;
			(void)fclose(input);
		}
		counters->reads[copy] = fields[DEVICE_STAT_READS];
		counters->ticks_ms[copy] = fields[DEVICE_STAT_TICKS];
	}
	FILE *const interrupts = fopen("/proc/interrupts", "re");
	if (interrupts == NULL)
		return;
	static char line[INTERRUPT_LINE_BYTES];
	while (fgets(line, sizeof(line), interrupts) != NULL) {
		line[strcspn(line, "\n")] = '\0';
		const char *const last = strrchr(line, ' ');
		if (last == NULL ||
		    strncmp(last + 1, "nvme", strlen("nvme")) != 0)
			continue;
		for (uint32_t copy = 0; copy < store->copy_count; copy++) {
			const char *const name = store->device_names[copy];
			const char *const namespace =
				strchr(name + strlen("nvme"), 'n');
			if (namespace == NULL ||
			    strncmp(last + 1, name,
				    (size_t)(namespace - name)) != 0 ||
			    last[1 + (namespace - name)] != 'q')
				continue;
			const char *cursor = strchr(line, ':');
			uint64_t sum = 0;
			while (cursor != NULL && *cursor != '\0') {
				cursor++;
				while (*cursor == ' ')
					cursor++;
				if (!isdigit((unsigned char)*cursor))
					break;
				char *end = NULL;
				sum += strtoull(cursor, &end, 10);
				cursor = end;
			}
			counters->interrupts[copy] += sum;
		}
	}
	(void)fclose(interrupts);
}

/* ---- output ------------------------------------------------------------------------------------------------ */

static void level_report(FILE *output, const struct level_context *context,
			 const struct device_counters *before,
			 const struct device_counters *after, uint64_t wall_ns)
{
	const struct bench *const bench = context->bench;
	const struct arm *const arm = context->arm;
	const struct level_samples *const samples = &context->samples;
	const struct bench_run *const run = &context->run;
	const struct bench_store *const store = &bench->store;
	const int run_error = atomic_load(&run->error);
	const int error = context->error != 0 ? context->error : run_error;
	const char *const reason =
		context->reason[0] != '\0' ? context->reason : run->reason;
	const bool refused = error == EOPNOTSUPP || error == EINVAL ||
			     error == EPERM || error == EACCES ||
			     error == ENODEV || error == ENOSYS;
	const char *const status = error == 0 ? "ok" :
				   refused    ? "refused" :
						"failed";
	uint64_t *const scratch = samples->scratch;
	(void)fprintf(
		output,
		"{\"schema\":1,\"kind\":\"lookup\",\"key\":\"%s|%u|%u\",\"arm\":\"%s\",\"spec\":\"%s\",\"level\":%u,"
		"\"repeat\":%u,\"shift_blocks\":%" PRIu64
		",\"status\":\"%s\",\"error\":\"%s\",\"reason\":\"%s\","
		"\"refusals\":\"%s\",\"identical\":%s,\"mismatched_rows\":%" PRIu64
		",",
		arm->name, level_sessions[context->level], context->repeat,
		arm->name, arm->text, level_sessions[context->level],
		context->repeat, context->shift_blocks, status,
		error != 0 && strerrorname_np(error) != NULL ?
			strerrorname_np(error) :
			"",
		reason, run->refusals,
		samples->mismatched_rows == 0 && error == 0 ? "true" : "false",
		samples->mismatched_rows);
	(void)fprintf(output,
		      "\"decode\":{\"steps\":%u,\"rows\":%" PRIu64
		      ",\"p50_ns\":%" PRIu64 ",\"p90_ns\":%" PRIu64
		      ",\"p99_ns\":%" PRIu64 ",\"p999_ns\":%" PRIu64
		      ",\"mean_ns\":%" PRIu64 ",\"max_ns\":%" PRIu64
		      ",\"wall_ns\":%" PRIu64 ",\"table_ready_p50_ns\":[",
		      samples->decode_count, samples->decode_rows,
		      percentile(samples->decode, samples->decode_count,
				 PER_MILLE_P50, scratch),
		      percentile(samples->decode, samples->decode_count,
				 PER_MILLE_P90, scratch),
		      percentile(samples->decode, samples->decode_count,
				 PER_MILLE_P99, scratch),
		      percentile(samples->decode, samples->decode_count,
				 PER_MILLE_P999, scratch),
		      mean(samples->decode, samples->decode_count),
		      percentile(samples->decode, samples->decode_count,
				 PER_MILLE_ALL, scratch),
		      samples->decode_wall);
	for (uint32_t table = 0; table < store->table_count; table++)
		(void)fprintf(output, "%s%" PRIu64, table == 0 ? "" : ",",
			      percentile(samples->ready[table],
					 samples->decode_count, PER_MILLE_P50,
					 scratch));
	(void)fputs("],\"table_ready_p99_ns\":[", output);
	for (uint32_t table = 0; table < store->table_count; table++)
		(void)fprintf(output, "%s%" PRIu64, table == 0 ? "" : ",",
			      percentile(samples->ready[table],
					 samples->decode_count, PER_MILLE_P99,
					 scratch));
	(void)fputs("],\"exposed_p50_ns\":[", output);
	for (uint32_t table = 0; table < store->table_count; table++)
		(void)fprintf(output, "%s%" PRIu64, table == 0 ? "" : ",",
			      percentile(samples->exposed[table],
					 samples->decode_count, PER_MILLE_P50,
					 scratch));
	(void)fputs("],\"exposed_p99_ns\":[", output);
	for (uint32_t table = 0; table < store->table_count; table++)
		(void)fprintf(output, "%s%" PRIu64, table == 0 ? "" : ",",
			      percentile(samples->exposed[table],
					 samples->decode_count, PER_MILLE_P99,
					 scratch));
	(void)fprintf(
		output,
		"]},\"prefill\":{\"chunk_tokens\":%u,\"chunks\":%u,\"rows\":%" PRIu64
		",\"p50_ns\":%" PRIu64 ",\"p99_ns\":%" PRIu64
		",\"mean_ns\":%" PRIu64 ",\"max_ns\":%" PRIu64
		",\"wall_ns\":%" PRIu64 ",\"rows_per_second\":%.1f},",
		arm->prefill_chunk, samples->prefill_count,
		samples->prefill_rows,
		percentile(samples->prefill, samples->prefill_count,
			   PER_MILLE_P50, scratch),
		percentile(samples->prefill, samples->prefill_count,
			   PER_MILLE_P99, scratch),
		mean(samples->prefill, samples->prefill_count),
		percentile(samples->prefill, samples->prefill_count,
			   PER_MILLE_ALL, scratch),
		samples->prefill_wall,
		samples->prefill_wall != 0 ?
			(double)samples->prefill_rows *
				(double)BENCH_NS_PER_SECOND /
				(double)samples->prefill_wall :
			0.0);
	const struct bench_counters *const local = &context->local;
	const uint64_t reads = atomic_load(&run->counters.reads);
	(void)fprintf(output,
		      "\"reads\":{\"issued\":%" PRIu64 ",\"retries\":%" PRIu64
		      ",\"failures\":%" PRIu64 ",\"hedges\":%" PRIu64
		      ",\"hedge_wins\":%" PRIu64 ",\"nowait_again\":%" PRIu64
		      ",\"references\":%" PRIu64 ",\"unique_blocks\":%" PRIu64
		      ",\"ram_hits\":%" PRIu64 ",\"hot_hits\":%" PRIu64
		      ",\"cache_hits\":%" PRIu64
		      ",\"per_second\":%.1f},\"copies\":[",
		      reads, atomic_load(&run->counters.retries),
		      atomic_load(&run->counters.failures),
		      atomic_load(&run->counters.hedges),
		      atomic_load(&run->counters.hedge_wins),
		      atomic_load(&run->counters.nowait_again),
		      atomic_load(&local->references),
		      atomic_load(&local->unique_blocks),
		      atomic_load(&local->ram_hits),
		      atomic_load(&local->hot_hits),
		      atomic_load(&local->cache_hits),
		      samples->decode_wall + samples->prefill_wall != 0 ?
			      (double)reads * (double)BENCH_NS_PER_SECOND /
				      (double)(samples->decode_wall +
					       samples->prefill_wall) :
			      0.0);
	for (uint32_t copy = 0; copy < store->copy_count; copy++)
		(void)fprintf(
			output,
			"%s{\"copy\":%u,\"device\":\"%s\",\"reads\":%" PRIu64
			",\"p50_ns\":%" PRIu64 ",\"p99_ns\":%" PRIu64
			",\"p999_ns\":%" PRIu64 ",\"device_reads\":%" PRIu64
			",\"util_percent\":%.2f,\"interrupts\":%" PRIu64 "}",
			copy == 0 ? "" : ",", copy, store->device_names[copy],
			atomic_load(&run->copy_reads[copy]),
			bench_histogram_percentile(&run->copy_latency[copy],
						   PER_MILLE_P50),
			bench_histogram_percentile(&run->copy_latency[copy],
						   PER_MILLE_P99),
			bench_histogram_percentile(&run->copy_latency[copy],
						   PER_MILLE_P999),
			after->reads[copy] - before->reads[copy],
			wall_ns != 0 ?
				(double)(after->ticks_ms[copy] -
					 before->ticks_ms[copy]) *
					(double)BENCH_NS_PER_MILLISECOND *
					PERCENT / (double)wall_ns :
				0.0,
			after->interrupts[copy] - before->interrupts[copy]);
	uint64_t io_ns = 0, voluntary = context->voluntary,
		 involuntary = context->involuntary;
	for (uint32_t index = 0;
	     index < run->owner_count && arm->io_threads != 0; index++) {
		io_ns += run->owners[index].cpu_ns;
		voluntary += run->owners[index].voluntary_switches;
		involuntary += run->owners[index].involuntary_switches;
	}
	(void)fprintf(output,
		      "],\"cpu\":{\"wall_ns\":%" PRIu64 ",\"step_ns\":%" PRIu64
		      ",\"io_ns\":%" PRIu64 ",\"pool_ns\":%" PRIu64
		      ",\"voluntary_switches\":%" PRIu64
		      ",\"involuntary_switches\":%" PRIu64 "}}\n",
		      wall_ns, context->step_cpu_ns, io_ns,
		      atomic_load(&run->pool_cpu_ns), voluntary, involuntary);
	(void)fflush(output);
	(void)fsync(fileno(output));
	(void)fprintf(
		stderr,
		"%-28s C%u r%u %-7s decode p50 %8.1f us p99 %8.1f us | prefill chunk p50 %9.2f ms | reads %" PRIu64
		" | %s%s\n",
		arm->name, level_sessions[context->level], context->repeat,
		status,
		(double)percentile(samples->decode, samples->decode_count,
				   PER_MILLE_P50, scratch) /
			1000.0,
		(double)percentile(samples->decode, samples->decode_count,
				   PER_MILLE_P99, scratch) /
			1000.0,
		(double)percentile(samples->prefill, samples->prefill_count,
				   PER_MILLE_P50, scratch) /
			1.0e6,
		reads, samples->mismatched_rows == 0 ? "identical" : "MISMATCH",
		error != 0 ? reason : "");
}

static void samples_free(struct level_samples *samples)
{
	free(samples->decode);
	free(samples->prefill);
	free(samples->scratch);
	for (uint32_t table = 0; table < BENCH_TABLES_MOST; table++) {
		free(samples->ready[table]);
		free(samples->exposed[table]);
	}
}

/* The production gather, opened (with its own full table scan) for each new RAM split, cache size or interleave. */
static int prod_prepare(struct bench *bench, const struct arm *arm)
{
	const uint32_t interleave =
		arm->copy == COPY_TICK ||
				__builtin_popcount(arm->copy_mask) > 1 ?
			1u :
			0u;
	if (bench->prod == NULL ||
	    bench->prod_ram_percent != arm->ram_percent ||
	    bench->prod_cache_bytes != arm->cache_bytes ||
	    bench->prod_interleave != interleave) {
		engram_disk_close(bench->prod);
		bench->prod = NULL;
		if (engram_disk_open(engram_description_of(bench->source.engram),
				     bench->source.pack, &storage_direct,
				     bench->source.reader, arm->ram_percent,
				     arm->cache_bytes, interleave, 1,
				     &bench->prod) != STATUS_OK)
			return EIO;
		bench->prod_ram_percent = arm->ram_percent;
		bench->prod_cache_bytes = arm->cache_bytes;
		bench->prod_interleave = interleave;
	}
	uint32_t weights[BENCH_COPIES_MOST] = { 0 };
	for (uint32_t copy = 0; copy < bench->store.copy_count; copy++)
		weights[copy] = (arm->copy_mask >> copy & 1u) != 0 ?
					arm->weights[copy] :
					0u;
	return engram_disk_tune_io(bench->prod, weights,
				   bench->store.copy_count, 1,
				   arm->cache_bytes != 0) == STATUS_OK ?
		       0 :
		       EIO;
}

static void page_cache_drop(const struct bench_store *store)
{
	for (uint32_t copy = 0; copy < store->copy_count; copy++)
		for (uint32_t table = 0; table < store->table_count; table++)
			(void)posix_fadvise(
				store->buffered[copy],
				(off_t)store->tables[table].byte_offset,
				(off_t)(store->tables[table].block_count *
					BENCH_BLOCK_BYTES),
				POSIX_FADV_DONTNEED);
}

static int level_measure(struct bench *bench, const struct arm *arm,
			 uint32_t level, uint32_t repeat)
{
	struct level_context *const context = calloc(1, sizeof(*context));
	if (context == NULL)
		return ENOMEM;
	context->bench = bench;
	context->arm = arm;
	context->level = level;
	context->repeat = repeat;
	context->shift_blocks = repeat * SHIFT_BLOCKS;
	int error = oracle_build(bench, context->shift_blocks, true);
	const uint64_t needed =
		arm->cache_bytes + BENCH_CHUNK_TOKENS_MOST * MIB_BYTES;
	if (error == 0 &&
	    memory_available_bytes() < needed + bench->memory_floor_bytes) {
		error = ENOMEM;
		(void)snprintf(context->reason, sizeof(context->reason),
			       "memory floor");
	}
	const bool passthrough =
		(arm->engine == ENGINE_URING &&
		 arm->operation == OPERATION_URING_CMD) ||
		(arm->engine == ENGINE_POOL && arm->call == CALL_IOCTL);
	const uint32_t mapped_mask = arm->engine == ENGINE_URING ?
					     arm->copy_mask &
						     arm->command_copy_mask :
					     arm->copy_mask;
	bool mapped = true;
	for (uint32_t copy = 0; copy < bench->store.copy_count; copy++)
		mapped = mapped && ((mapped_mask >> copy & 1u) == 0 ||
				    bench->store.maps[copy] != NULL);
	if (error == 0 && passthrough && !mapped) {
		error = bench->store.passthrough_refusal[0] != '\0' &&
					strstr(bench->store.passthrough_refusal,
					       "EACCES") ?
				EACCES :
				ENODEV;
		(void)snprintf(context->reason, sizeof(context->reason),
			       "passthrough: %s",
			       bench->store.passthrough_refusal);
	}
	if (error == 0 && arm->hot_file[0] != '\0')
		error = hot_from_file(bench, arm, context->reason,
				      sizeof(context->reason));
	if (error == 0 && arm->engine == ENGINE_PROD)
		error = prod_prepare(bench, arm);
	if (error == 0 && (arm->buffered || arm->engine == ENGINE_MMAP) &&
	    arm->fadvise_random)
		for (uint32_t copy = 0; copy < bench->store.copy_count; copy++)
			for (uint32_t table = 0;
			     table < bench->store.table_count; table++)
				(void)posix_fadvise(
					bench->store.buffered[copy],
					(off_t)bench->store.tables[table]
						.byte_offset,
					(off_t)(bench->store.tables[table]
							.block_count *
						BENCH_BLOCK_BYTES),
					POSIX_FADV_RANDOM);
	struct device_counters before, after;
	counters_read(&bench->store, &before);
	const uint64_t started = bench_now();
	if (error == 0) {
		pthread_t thread;
		if (pthread_create(&thread, NULL, level_main, context) != 0)
			error = EAGAIN;
		else
			(void)pthread_join(thread, NULL);
	}
	const uint64_t wall = bench_now() - started;
	counters_read(&bench->store, &after);
	if (error != 0 && context->error == 0) {
		context->error = error;
		if (context->reason[0] == '\0')
			(void)snprintf(context->reason, sizeof(context->reason),
				       "%s", "prepare");
	}
	if (context->samples.scratch == NULL)
		context->samples.scratch = calloc(1, sizeof(uint64_t));
	level_report(bench->output, context, &before, &after, wall);
	if (arm->buffered || arm->engine == ENGINE_MMAP)
		page_cache_drop(&bench->store);
	samples_free(&context->samples);
	const int result = context->error != 0 ?
				   context->error :
				   atomic_load(&context->run.error);
	free(context);
	return result;
}

/* ---- run: the arms over the trace ------------------------------------------------------------------------- */

static bool already_done(const char *out_path, const char *name, uint32_t level,
			 uint32_t repeat)
{
	FILE *const input = fopen(out_path, "re");
	if (input == NULL)
		return false;
	char key[BENCH_TEXT_BYTES], line[BENCH_TEXT_BYTES * 8u];
	(void)snprintf(key, sizeof(key), "\"key\":\"%s|%u|%u\"", name, level,
		       repeat);
	bool found = false;
	while (!found && fgets(line, sizeof(line), input) != NULL)
		found = strstr(line, key) != NULL;
	(void)fclose(input);
	return found;
}

struct options {
	const char *copies, *traces, *arms, *out, *plan, *directory,
		*hot_directory;
	uint64_t seconds, repeats, skip_done, prep_threads, prep_piece_mib,
		memory_floor_gib, raw, hot_mib, scan_threads;
};

static uint32_t split_paths(const char *text, char storage[][PATH_MAX],
			    const char **paths)
{
	uint32_t count = 0;
	for (const char *cursor = text;
	     cursor != NULL && *cursor != '\0' && count < BENCH_COPIES_MOST;) {
		const char *const comma = strchr(cursor, ',');
		const size_t length = comma != NULL ? (size_t)(comma - cursor) :
						      strlen(cursor);
		(void)snprintf(storage[count], PATH_MAX, "%.*s", (int)length,
			       cursor);
		paths[count] = storage[count];
		count++;
		cursor = comma != NULL ? comma + 1 : NULL;
	}
	return count;
}

/* Every arm is read first; repeats are the outer loop, so each repeat's row shift loads its oracle once. A level
 * starts only when its estimate (its arm's last level, at least ESTIMATE_DEFAULT_SECONDS for a per-call or production
 * arm, half again for safety) fits before the lease's deadline. */
static int arms_run(struct bench *bench, const struct options *options)
{
	FILE *const input = fopen(options->arms, "re");
	if (input == NULL)
		return errno;
	struct arm *const arms = calloc(ARMS_MOST, sizeof(*arms));
	uint64_t *const last_ns = calloc(ARMS_MOST, sizeof(*last_ns));
	uint32_t arm_count = 0;
	int status = arms != NULL && last_ns != NULL ? 0 : ENOMEM;
	char line[ARM_LINE_BYTES];
	while (status == 0 && fgets(line, sizeof(line), input) != NULL) {
		if (line[0] == '#' || line[0] == '\n')
			continue;
		char reason[BENCH_REASON_BYTES] = "";
		if (arm_count == ARMS_MOST ||
		    !arm_parse(line, bench->store.copy_count, &arms[arm_count],
			       reason, sizeof(reason))) {
			(void)fprintf(stderr,
				      "engram_io_bench: arm refused: %s (%s)",
				      reason, line);
			status = EINVAL;
			break;
		}
		const struct arm *const arm = &arms[arm_count];
		last_ns[arm_count++] =
			arm->engine == ENGINE_PROD || arm->batch == BATCH_CALL ?
				(uint64_t)ESTIMATE_DEFAULT_SECONDS *
					BENCH_NS_PER_SECOND :
				0;
	}
	(void)fclose(input);
	const uint64_t deadline =
		options->seconds != 0 ?
			bench_now() + options->seconds * BENCH_NS_PER_SECOND :
			UINT64_MAX;
	const uint32_t repeats =
		options->repeats != 0 ? (uint32_t)options->repeats : 1u;
	for (uint32_t repeat = 0; status == 0 && repeat < repeats; repeat++)
		for (uint32_t index = 0; status == 0 && index < arm_count;
		     index++)
			for (uint32_t level = 0;
			     status == 0 && level < BENCH_LEVELS; level++) {
				const struct arm *const arm = &arms[index];
				if ((arm->levels_mask >> level & 1u) == 0 ||
				    (options->skip_done != 0 &&
				     already_done(options->out, arm->name,
						  level_sessions[level],
						  repeat)))
					continue;
				if (bench_now() + last_ns[index] *
							  ESTIMATE_FACTOR /
							  ESTIMATE_DIVISOR >
				    deadline) {
					status = EXIT_CHECKPOINT;
					break;
				}
				const uint64_t started = bench_now();
				(void)level_measure(bench, arm, level, repeat);
				const uint64_t took = bench_now() - started;
				last_ns[index] = took > last_ns[index] ?
							 took :
							 last_ns[index];
			}
	free(arms);
	free(last_ns);
	return status;
}

static int options_parse(int count, char **arguments, struct options *options)
{
	memset(options, 0, sizeof(*options));
	options->prep_threads = 2;
	options->prep_piece_mib = PREP_PIECE_DEFAULT_MIB;
	options->memory_floor_gib = MEMORY_FLOOR_DEFAULT_GIB;
	for (int index = 2; index < count; index++) {
		const char *const argument = arguments[index];
		struct {
			const char *name;
			const char **text;
			uint64_t *number;
		} const table[] = {
			{ "--copies=", &options->copies, NULL },
			{ "--traces=", &options->traces, NULL },
			{ "--arms=", &options->arms, NULL },
			{ "--out=", &options->out, NULL },
			{ "--plan=", &options->plan, NULL },
			{ "--directory=", &options->directory, NULL },
			{ "--hot_directory=", &options->hot_directory, NULL },
			{ "--hot_mib=", NULL, &options->hot_mib },
			{ "--scan_threads=", NULL, &options->scan_threads },
			{ "--seconds=", NULL, &options->seconds },
			{ "--repeats=", NULL, &options->repeats },
			{ "--skip_done=", NULL, &options->skip_done },
			{ "--prep_threads_per_copy=", NULL,
			  &options->prep_threads },
			{ "--prep_piece_mib=", NULL, &options->prep_piece_mib },
			{ "--memory_floor_gib=", NULL,
			  &options->memory_floor_gib },
			{ "--raw=", NULL, &options->raw },
		};
		bool known = false;
		for (size_t entry = 0; entry < sizeof(table) / sizeof(table[0]);
		     entry++) {
			const size_t length = strlen(table[entry].name);
			if (strncmp(argument, table[entry].name, length) != 0)
				continue;
			known = table[entry].text != NULL ?
					(*table[entry].text = argument + length,
					 true) :
					parse_u64(argument + length,
						  table[entry].number);
		}
		if (!known) {
			(void)fprintf(stderr,
				      "engram_io_bench: unknown argument %s\n",
				      argument);
			return EINVAL;
		}
	}
	if (options->prep_threads == 0 ||
	    options->prep_threads > PREP_THREADS_MOST ||
	    options->prep_piece_mib == 0)
		return EINVAL;
	return 0;
}

/* ---- self test: a synthetic store with corrupt blocks, every arm, rows compared ---------------------------- */

static uint64_t self_random(uint64_t *state)
{
	*state = mix64(*state + SELF_SEED);
	return *state;
}

static void self_block(uint32_t table, uint64_t block, unsigned char *bytes)
{
	uint64_t state = (uint64_t)table << KEY_TABLE_SHIFT ^ block;
	for (uint32_t word = 0; word < BENCH_BLOCK_BYTES / sizeof(uint64_t);
	     word++) {
		const uint64_t value = self_random(&state);
		memcpy(bytes + word * sizeof(uint64_t), &value, sizeof(value));
	}
}

struct self_corruption {
	uint32_t copy, table;
	uint64_t block;
};
static const struct self_corruption self_corruptions[] = {
	{ 0, 0, 5 },   { 1, 1, 9 },   { 2, 0, 77 },
	{ 0, 1, 123 }, { 1, 1, 123 }, /* table 1 block 123: only copy 2 holds it */
};

static int self_store(struct bench *bench, const char *directory)
{
	struct bench_store *const store = &bench->store;
	store->table_count = SELF_TABLES;
	store->rows_per_token = SELF_ROWS_PER_TOKEN;
	const uint64_t file_bytes =
		(1u + (uint64_t)SELF_TABLES * SELF_TABLE_BLOCKS) *
		BENCH_BLOCK_BYTES;
	char storage[BENCH_COPIES_MOST][PATH_MAX];
	const char *paths[BENCH_COPIES_MOST];
	unsigned char *const block =
		aligned_alloc(BENCH_BLOCK_BYTES, BENCH_BLOCK_BYTES);
	if (block == NULL)
		return ENOMEM;
	for (uint32_t table = 0; table < SELF_TABLES; table++) {
		struct bench_table *const own = &store->tables[table];
		own->byte_offset = (1u + (uint64_t)table * SELF_TABLE_BLOCKS) *
				   BENCH_BLOCK_BYTES;
		own->block_count = SELF_TABLE_BLOCKS;
		own->row_count =
			(uint64_t)SELF_TABLE_BLOCKS * ENGRAM_ROWS_PER_BLOCK -
			SELF_ROW_SHORTFALL;
		own->block_digests =
			calloc(own->block_count, sizeof(*own->block_digests));
		if (own->block_digests == NULL)
			return ENOMEM;
		for (uint64_t index = 0; index < own->block_count; index++) {
			self_block(table, index, block);
			own->block_digests[index] = pack_digest_of_piece(
				block, BENCH_BLOCK_BYTES, 0);
			own->extent_digest += pack_digest_of_piece(
				block, BENCH_BLOCK_BYTES,
				index * BENCH_BLOCK_BYTES /
					PACK_DIGEST_WORD_BYTES);
		}
	}
	for (uint32_t copy = 0; copy < 3u; copy++) {
		(void)snprintf(storage[copy], PATH_MAX, "%s/copy%u.bin",
			       directory, copy);
		paths[copy] = storage[copy];
		const int descriptor =
			open(paths[copy],
			     O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
		if (descriptor < 0)
			return errno;
		memset(block, 0, BENCH_BLOCK_BYTES);
		bool written = pwrite(descriptor, block, BENCH_BLOCK_BYTES,
				      0) == (ssize_t)BENCH_BLOCK_BYTES;
		for (uint32_t table = 0; written && table < SELF_TABLES;
		     table++)
			for (uint64_t index = 0;
			     written && index < SELF_TABLE_BLOCKS; index++) {
				self_block(table, index, block);
				for (size_t corruption = 0;
				     corruption <
				     sizeof(self_corruptions) /
					     sizeof(self_corruptions[0]);
				     corruption++)
					if (self_corruptions[corruption].copy ==
						    copy &&
					    self_corruptions[corruption].table ==
						    table &&
					    self_corruptions[corruption].block ==
						    index)
						block[index %
						      BENCH_BLOCK_BYTES] ^= 1u;
				written =
					pwrite(descriptor, block,
					       BENCH_BLOCK_BYTES,
					       (off_t)(store->tables[table]
							       .byte_offset +
						       index * BENCH_BLOCK_BYTES)) ==
					(ssize_t)BENCH_BLOCK_BYTES;
			}
		written = written && fsync(descriptor) == 0 &&
			  (uint64_t)lseek(descriptor, 0, SEEK_END) ==
				  file_bytes;
		(void)close(descriptor);
		if (!written)
			return EIO;
	}
	free(block);
	int error = store_files_open(store, paths, 3u);
	/* synthetic cases: most rows from a small hot set, the rest anywhere, and the corrupt blocks early on */
	const uint32_t width = SELF_TABLES * SELF_ROWS_PER_TOKEN;
	for (uint32_t index = 0; error == 0 && index < SELF_CASES; index++) {
		struct bench_case *const own = &bench->cases[index];
		own->token_count = SELF_TOKENS_BASE + SELF_TOKENS_STEP * index;
		own->prompt = SELF_PROMPT_BASE + SELF_PROMPT_STEP * index;
		own->row_ids = calloc((size_t)own->token_count * width,
				      sizeof(*own->row_ids));
		if (own->row_ids == NULL)
			return ENOMEM;
		uint64_t state = index + 1u;
		for (uint64_t reference = 0;
		     reference < (uint64_t)own->token_count * width;
		     reference++) {
			const uint32_t table = (uint32_t)(reference % width /
							  SELF_ROWS_PER_TOKEN);
			const uint64_t rows = store->tables[table].row_count;
			own->row_ids[reference] =
				self_random(&state) % PERCENT <
						SELF_HOT_PERCENT ?
					self_random(&state) % SELF_HOT_ROWS :
					self_random(&state) % rows;
		}
		for (size_t corruption = 0;
		     corruption <
		     sizeof(self_corruptions) / sizeof(self_corruptions[0]);
		     corruption++)
			own->row_ids[(corruption + index) * width +
				     self_corruptions[corruption].table *
					     SELF_ROWS_PER_TOKEN] =
				self_corruptions[corruption].block *
				ENGRAM_ROWS_PER_BLOCK;
		bench->case_count++;
	}
	return error;
}

static const char *const self_arms[] = {
	"uring_read engine=uring",
	"uring_read_poll engine=uring wait=poll setup=none",
	"uring_spin engine=uring wait=spin",
	"uring_hybrid engine=uring wait=hybrid spin_us=50",
	"uring_readv engine=uring op=readv",
	"uring_fixed engine=uring op=read_fixed",
	"uring_readv_fixed engine=uring op=readv_fixed",
	"uring_unregistered engine=uring files=0",
	"uring_ring_fd engine=uring ring_fd=1",
	"uring_async engine=uring async=1",
	"uring_defer engine=uring setup=single_issuer,defer_taskrun wait=spin",
	"uring_defer_enter engine=uring setup=single_issuer,defer_taskrun wait=enter",
	"uring_coop engine=uring setup=coop_taskrun,taskrun_flag wait=spin",
	"uring_coop_noflag engine=uring setup=coop_taskrun wait=hybrid spin_us=20",
	"uring_nosqarray engine=uring setup=no_sqarray,submit_all,single_issuer",
	"uring_sqpoll engine=uring setup=sqpoll wait=spin",
	"uring_iopoll engine=uring setup=iopoll wait=spin",
	"uring_hybrid_iopoll engine=uring setup=iopoll,hybrid_iopoll wait=spin",
	"uring_devices engine=uring rings=device wait=poll",
	"uring_devices_enter engine=uring rings=device wait=enter slice_us=50",
	"uring_thread1 engine=uring io_threads=1 wait=spin",
	"uring_thread3 engine=uring io_threads=3 rings=device wait=spin pin=irq",
	"uring_thread6 engine=uring io_threads=6 wait=enter",
	"uring_thread3_defer engine=uring io_threads=3 setup=single_issuer,defer_taskrun,no_sqarray wait=hybrid "
	"spin_us=30",
	"uring_cmd engine=uring op=uring_cmd",
	"uring_cmd_mixed engine=uring op=uring_cmd rings=device command_copies=0 wait=spin",
	"uring_poll_mixed engine=uring setup=single_issuer,iopoll rings=device io_threads=3 poll_copies=0 wait=spin",
	"aio_enter engine=aio",
	"aio_spin engine=aio wait=spin rings=device",
	"aio_thread engine=aio io_threads=3 rings=device wait=spin",
	"pool_pread engine=pool pool_threads=8",
	"pool_spin engine=pool pool_threads=4 pool_spin=1 wait=spin",
	"pool_nowait engine=pool call=preadv2 rwf=nowait",
	"pool_hipri engine=pool call=preadv2 rwf=hipri",
	"pool_ioctl engine=pool call=ioctl",
	"pool_buffered engine=pool buffered=1 fadvise=random",
	"sync_direct engine=sync",
	"sync_readahead engine=sync buffered=1 readahead=1",
	"mmap_plain engine=mmap fadvise=random",
	"mmap_willneed engine=mmap willneed=1 io_threads=1 wait=spin",
	"batch_call engine=uring batch=call dedupe=0 depth=32 wait=poll copy=tick cache_mib=1",
	"batch_table engine=uring batch=table",
	"no_dedupe engine=uring dedupe=0",
	"copy_first engine=uring copy=first",
	"copy_tick engine=uring copy=tick weights=3,1,2",
	"copy_block engine=uring copy=block",
	"copy_p2c engine=uring copy=p2c",
	"copy_rate engine=uring copy=rate",
	"copy_single engine=uring copies=1",
	"hedge engine=uring hedge_us=1 io_threads=3 rings=device wait=spin",
	"hedge_inline engine=uring hedge_us=1",
	"hedge_relative engine=uring io_threads=3 rings=device wait=spin hedge_percent=50 hedge_step_most=400",
	"inflight engine=uring inflight=4",
	"depth8 engine=uring depth=8",
	"lead engine=uring lead_us=20,200",
	"cache engine=uring cache_mib=2",
	"hot_block_other engine=uring hot_mib=1 hot_profile=other",
	"hot_row_all engine=uring hot_mib=1 hot_unit=row",
	"ram_prefix engine=uring ram_percent=30",
	"all_ram engine=uring all_ram=1",
	"pin_irq engine=uring pin=irq",
	"pin_list engine=uring io_threads=2 pin=cpus:1,2 step_cpu=3 wait=spin",
	"nice engine=uring nice=5",
	"sched_batch engine=uring sched=batch",
	"sched_fifo engine=uring sched=fifo rtprio=10",
	"ioprio_be engine=uring ioprio=be:0",
	"ioprio_rt engine=uring ioprio=rt:0",
	"prefill256 engine=uring prefill_chunk=256",
};

static int self_test(const struct options *options)
{
	if (options->directory == NULL)
		return EINVAL;
	struct bench *const bench = calloc(1, sizeof(*bench));
	if (bench == NULL)
		return ENOMEM;
	bench->memory_floor_bytes = 0;
	int error = self_store(bench, options->directory);
	char out_path[PATH_MAX];
	(void)snprintf(out_path, sizeof(out_path), "%s/self_test.jsonl",
		       options->directory);
	bench->output = fopen(out_path, "we");
	if (error != 0 || bench->output == NULL) {
		(void)fprintf(stderr, "engram_io_bench self_test: store: %s\n",
			      strerrorname_np(error != 0 ? error : errno));
		return EXIT_FAILURE;
	}
	error = bench_passthrough_prepare(&bench->store);
	(void)fprintf(stderr, "self_test: passthrough %s %s\n",
		      error == 0 ? "available" : "refused",
		      bench->store.passthrough_refusal);
	/* the parallel scan over corrupt copies: its rescans must recover every table from a good copy */
	error = store_digests(&bench->store, 2u, MIB_BYTES, stderr);
	if (error != 0) {
		(void)fprintf(stderr, "self_test: digest scan failed: %s\n",
			      strerrorname_np(error));
		return EXIT_FAILURE;
	}
	uint32_t failed = 0, refused = 0, passed = 0;
	/* --arms: that file's arms (every line through the run command's parser) in place of the built-in list */
	static char lines[ARMS_MOST][ARM_LINE_BYTES];
	const char *texts[ARMS_MOST];
	size_t arm_count = 0;
	FILE *const arms_input =
		options->arms != NULL ? fopen(options->arms, "re") : NULL;
	if (options->arms != NULL && arms_input == NULL) {
		(void)fprintf(stderr, "self_test: arms %s: %s\n", options->arms,
			      strerrorname_np(errno));
		return EXIT_FAILURE;
	}
	while (arms_input != NULL && arm_count < ARMS_MOST &&
	       fgets(lines[arm_count], sizeof(lines[arm_count]), arms_input) !=
		       NULL)
		if (lines[arm_count][0] != '#' && lines[arm_count][0] != '\n') {
			lines[arm_count][strcspn(lines[arm_count], "\r\n")] =
				'\0';
			texts[arm_count] = lines[arm_count];
			arm_count++;
		}
	if (arms_input != NULL)
		(void)fclose(arms_input);
	if (options->arms == NULL)
		for (; arm_count < sizeof(self_arms) / sizeof(self_arms[0]);
		     arm_count++)
			texts[arm_count] = self_arms[arm_count];
	for (size_t index = 0; index < arm_count; index++) {
		struct arm arm;
		char reason[BENCH_REASON_BYTES] = "";
		if (!arm_parse(texts[index], bench->store.copy_count, &arm,
			       reason, sizeof(reason))) {
			(void)fprintf(stderr, "self_test: arm %s refused: %s\n",
				      texts[index], reason);
			failed++;
			continue;
		}
		arm.prefill_chunk = arm.prefill_chunk ==
						    BENCH_CHUNK_TOKENS_MOST ?
					    SELF_CHUNK :
					    arm.prefill_chunk;
		for (uint32_t level = 0; level < BENCH_LEVELS; level++) {
			const int result = level_measure(
				bench, &arm, level,
				level == BENCH_LEVELS - 1u ? 1u : 0u);
			const bool refusal =
				result == EOPNOTSUPP || result == EINVAL ||
				result == EPERM || result == EACCES ||
				result == ENODEV || result == ENOSYS;
			if (result == 0)
				passed++;
			else if (refusal)
				refused++;
			else
				failed++;
		}
	}
	(void)fclose(bench->output);
	/* every ok row must be identical: the JSON says so per row; a mismatch makes level_measure succeed but report
	 * "identical":false, so read the rows back */
	FILE *const input = fopen(out_path, "re");
	uint32_t mismatched = 0;
	char line[BENCH_TEXT_BYTES * 8u];
	while (input != NULL && fgets(line, sizeof(line), input) != NULL)
		if (strstr(line, "\"status\":\"ok\"") != NULL &&
		    strstr(line, "\"identical\":true") == NULL)
			mismatched++;
	if (input != NULL)
		(void)fclose(input);
	(void)fprintf(
		stderr,
		"self_test: %u passed, %u refused, %u failed, %u mismatched (rows in %s)\n",
		passed, refused, failed, mismatched, out_path);
	return failed == 0 && mismatched == 0 && passed != 0 ? EXIT_SUCCESS :
							       EXIT_FAILURE;
}

/* What a run would read, with no table byte read: the pack's tables, the cases, and per level the steps, chunks,
 * references and the blocks the oracle would hold. */
static int trace_plan(struct bench *bench, const struct options *options)
{
	if (options->traces == NULL || cases_read(bench, options->traces) != 0)
		return EXIT_FAILURE;
	const int error = oracle_build(bench, 0, false);
	const struct bench_store *const store = &bench->store;
	(void)printf(
		"{\"schema\":1,\"kind\":\"plan\",\"status\":\"%s\",\"copies\":%u,\"file_bytes\":%" PRIu64
		",\"tables\":[",
		error == 0 ? "ok" : "failed", store->copy_count,
		store->file_bytes);
	for (uint32_t table = 0; table < store->table_count; table++)
		(void)printf("%s{\"byte_offset\":%" PRIu64
			     ",\"blocks\":%" PRIu64 ",\"rows\":%" PRIu64 "}",
			     table == 0 ? "" : ",",
			     store->tables[table].byte_offset,
			     store->tables[table].block_count,
			     store->tables[table].row_count);
	(void)printf("],\"rows_per_token\":%u,\"oracle_blocks\":%" PRIu64
		     ",\"oracle_rows\":%" PRIu64 ",\"devices\":[",
		     store->rows_per_token, bench->oracle.entries,
		     bench->oracle.row_entries);
	for (uint32_t copy = 0; copy < store->copy_count; copy++)
		(void)printf("%s{\"device\":\"%s\",\"irq_cpu\":%u}",
			     copy == 0 ? "" : ",", store->device_names[copy],
			     store->irq_cpu[copy]);
	(void)printf("],\"cases\":[");
	for (uint32_t index = 0; index < bench->case_count; index++)
		(void)printf(
			"%s{\"tokens\":%u,\"prompt\":%u,\"first_rows\":[%" PRIu64
			",%" PRIu64 "]}",
			index == 0 ? "" : ",", bench->cases[index].token_count,
			bench->cases[index].prompt,
			bench->cases[index].row_ids[0],
			bench->cases[index].row_ids[store->rows_per_token]);
	(void)printf("],\"levels\":[");
	for (uint32_t level = 0; level < BENCH_LEVELS; level++) {
		const uint32_t sessions = level_sessions[level];
		const uint32_t groups =
			(bench->case_count + sessions - 1u) / sessions;
		uint64_t prefill_1024 = 0, prefill_256 = 0, decode_steps = 0;
		(void)steps_capacity(bench, sessions, groups,
				     BENCH_CHUNK_TOKENS_MOST, &prefill_1024);
		(void)steps_capacity(bench, sessions, groups,
				     BENCH_CHUNK_TOKENS_MOST / 4u,
				     &prefill_256);
		for (uint32_t group = 0; group < groups; group++) {
			uint32_t longest = 0;
			for (uint32_t session = 0; session < sessions;
			     session++) {
				const struct bench_case *const own =
					&bench->cases[(group * sessions +
						       session) %
						      bench->case_count];
				const uint32_t steps =
					own->token_count - (own->prompt - 1u);
				longest = steps > longest ? steps : longest;
			}
			decode_steps += longest < DECODE_STEPS ? longest :
								 DECODE_STEPS;
		}
		(void)printf(
			"%s{\"sessions\":%u,\"groups\":%u,\"decode_steps\":%" PRIu64
			",\"prefill_chunks_1024\":%" PRIu64
			",\"prefill_chunks_256\":%" PRIu64 "}",
			level == 0 ? "" : ",", sessions, groups, decode_steps,
			prefill_1024, prefill_256);
	}
	(void)printf("]}\n");
	return error == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

static int compare_keys(const void *left, const void *right)
{
	const uint64_t a = *(const uint64_t *)left,
		       b = *(const uint64_t *)right;
	return a < b ? -1 : a > b ? 1 : 0;
}

struct counted_block {
	uint64_t key, count;
};
/* the most referenced first; among equals, the lower table and block */
static int compare_counted(const void *left, const void *right)
{
	const struct counted_block *const a = left, *const b = right;
	return a->count != b->count ? (a->count > b->count ? -1 : 1) :
	       a->key < b->key	    ? -1 :
	       a->key > b->key	    ? 1 :
				      0;
}
static int compare_counted_keys(const void *left, const void *right)
{
	return compare_keys(&((const struct counted_block *)left)->key,
			    &((const struct counted_block *)right)->key);
}

/* hot_build: the most referenced blocks of a trace corpus (every position of every listed token file, all tables) up
 * to --hot_mib, written as an engram_hot file at --out (created, never replaced) with one JSON receipt on stdout. No
 * table byte is read. The corpus must not be the prompts the hot set will serve; the receipt names it. */
static int hot_build(struct bench *bench, const struct options *options)
{
	if (options->traces == NULL || options->out == NULL ||
	    bench->source.pack == NULL || options->hot_mib == 0 ||
	    options->hot_mib > ENGRAM_HOT_BUDGET_MOST_MIB)
		return EINVAL;
	int error = cases_read(bench, options->traces);
	const struct bench_store *const store = &bench->store;
	const uint32_t width = store->table_count * store->rows_per_token;
	uint64_t references = 0, tokens = 0;
	for (uint32_t index = 0; error == 0 && index < bench->case_count;
	     index++) {
		tokens += bench->cases[index].token_count;
		references += (uint64_t)bench->cases[index].token_count * width;
	}
	uint64_t *const keys =
		error == 0 ?
			malloc((size_t)(references != 0 ? references : 1u) *
			       sizeof(uint64_t)) :
			NULL;
	struct counted_block *const unique =
		error == 0 ?
			malloc((size_t)(references != 0 ? references : 1u) *
			       sizeof(*unique)) :
			NULL;
	if (error == 0 && (keys == NULL || unique == NULL))
		error = ENOMEM;
	uint64_t filled = 0;
	for (uint32_t index = 0; error == 0 && index < bench->case_count;
	     index++) {
		const struct bench_case *const own = &bench->cases[index];
		for (uint64_t at = 0; at < (uint64_t)own->token_count * width;
		     at++) {
			const uint32_t table =
				(uint32_t)(at % width / store->rows_per_token);
			keys[filled++] =
				block_key(table, own->row_ids[at] /
							 ENGRAM_ROWS_PER_BLOCK);
		}
	}
	uint64_t unique_count = 0;
	if (error == 0) {
		qsort(keys, (size_t)filled, sizeof(*keys), compare_keys);
		for (uint64_t at = 0; at < filled; at++) {
			if (unique_count == 0 ||
			    unique[unique_count - 1u].key != keys[at])
				unique[unique_count++] = (struct counted_block){
					.key = keys[at]
				};
			unique[unique_count - 1u].count++;
		}
		qsort(unique, (size_t)unique_count, sizeof(*unique),
		      compare_counted);
	}
	const uint64_t budget_blocks =
		options->hot_mib * MIB_BYTES / BENCH_BLOCK_BYTES;
	const uint64_t chosen = unique_count < budget_blocks ? unique_count :
							       budget_blocks;
	uint64_t covered = 0;
	for (uint64_t at = 0; at < chosen; at++)
		covered += unique[at].count;
	if (error == 0)
		qsort(unique, (size_t)chosen, sizeof(*unique),
		      compare_counted_keys);
	/* by table, in increasing order; block_key adds one so that zero means empty */
	uint64_t *ids = error == 0 ?
				malloc((size_t)(chosen != 0 ? chosen : 1u) *
				       sizeof(uint64_t)) :
				NULL;
	if (error == 0 && ids == NULL)
		error = ENOMEM;
	struct engram_hot_table tables[ENGRAM_MAXIMUM_TABLES] = { 0 };
	uint64_t at = 0;
	for (uint32_t table = 0; error == 0 && table < store->table_count;
	     table++) {
		tables[table] = (struct engram_hot_table){
			.table = table,
			.extent_digest = store->tables[table].extent_digest,
			.block_count = store->tables[table].block_count,
			.ids = ids + at
		};
		while (at < chosen &&
		       (unique[at].key - 1u) >> KEY_TABLE_SHIFT == table) {
			ids[at] = (unique[at].key - 1u) &
				  ((1ull << KEY_TABLE_SHIFT) - 1u);
			tables[table].id_count++;
			at++;
		}
	}
	unsigned char *bytes = NULL;
	size_t byte_count = 0;
	if (error == 0 && engram_hot_encode(pack_header_of(bench->source.pack),
					    tables, store->table_count, &bytes,
					    &byte_count) != STATUS_OK)
		error = EINVAL;
	const int descriptor =
		error == 0 ? open(options->out,
				  O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC,
				  S_IRUSR | S_IRGRP | S_IROTH) :
			     -1;
	if (error == 0 &&
	    (descriptor < 0 ||
	     write(descriptor, bytes, byte_count) != (ssize_t)byte_count ||
	     fsync(descriptor) != 0))
		error = descriptor < 0 ? errno : EIO;
	if (descriptor >= 0)
		(void)close(descriptor);
	unsigned char digest[SHA256_DIGEST_BYTES] = { 0 };
	if (error == 0 && sha256_of(bytes, byte_count, digest) != STATUS_OK)
		error = EIO;
	if (error == 0) {
		(void)printf(
			"{\"schema\":1,\"kind\":\"hot_build\",\"corpus\":\"%s\",\"cases\":%u,\"tokens\":%" PRIu64
			",\"references\":%" PRIu64 ",\"unique_blocks\":%" PRIu64
			",\"hot_blocks\":%" PRIu64 ",\"hot_bytes\":%" PRIu64
			",\"covered_references\":%" PRIu64
			",\"coverage\":%.6f,\"out\":\"%s\",\"sha256\":\"",
			options->traces, bench->case_count, tokens, references,
			unique_count, chosen, chosen * BENCH_BLOCK_BYTES,
			covered,
			references != 0 ? (double)covered / (double)references :
					  0.0,
			options->out);
		for (uint32_t index = 0; index < SHA256_DIGEST_BYTES; index++)
			(void)printf("%02x", digest[index]);
		(void)printf("\"}\n");
	} else
		(void)fprintf(stderr, "engram_io_bench hot_build: %s\n",
			      strerrorname_np(error) != NULL ?
				      strerrorname_np(error) :
				      "failed");
	free(bytes);
	free(ids);
	free(unique);
	free(keys);
	return error;
}

/* scan: the engine's startup digest scan of the tables (engram_disk_open, no RAM prefix, no cache) timed with the
 * given readers a copy, one JSON line on stdout. */
static int engine_scan(struct bench *bench, const struct options *options)
{
	if (options->scan_threads > ENGRAM_DISK_SCAN_THREADS_MOST)
		return EINVAL;
	engram_disk_scan_threads((uint32_t)options->scan_threads);
	struct engram_disk *disk = NULL;
	const uint64_t started = bench_now();
	const result_status status = engram_disk_open(
		engram_description_of(bench->source.engram), bench->source.pack,
		&storage_direct, bench->source.reader, 0, 0, 1, 0, &disk);
	const uint64_t elapsed = bench_now() - started;
	uint64_t bytes = 0;
	for (uint32_t table = 0; table < bench->store.table_count; table++)
		bytes += bench->store.tables[table].block_count *
			 BENCH_BLOCK_BYTES;
	(void)printf(
		"{\"schema\":1,\"kind\":\"scan\",\"status\":\"%s\",\"copies\":%u,\"threads_per_copy\":%" PRIu64
		",\"bytes\":%" PRIu64
		",\"seconds\":%.3f,\"gb_per_second\":%.3f}\n",
		status_text(status), bench->store.copy_count,
		options->scan_threads, bytes,
		(double)elapsed / (double)BENCH_NS_PER_SECOND,
		elapsed != 0 ? (double)bytes / (double)elapsed : 0.0);
	engram_disk_close(disk);
	return status == STATUS_OK ? 0 : EIO;
}

int main(int argument_count, char **arguments)
{
	if (argument_count < 2) {
		(void)fprintf(
			stderr,
			"usage: engram_io_bench run|digests|characterise|plan|hot_build|self_test --option=value"
			" ...\n");
		return EXIT_FAILURE;
	}
	struct options options;
	if (options_parse(argument_count, arguments, &options) != 0)
		return EXIT_FAILURE;
	if (strcmp(arguments[1], "self_test") == 0)
		return self_test(&options);
	if (options.copies == NULL)
		return EXIT_FAILURE;
	struct bench *const bench = calloc(1, sizeof(*bench));
	if (bench == NULL)
		return EXIT_FAILURE;
	bench->memory_floor_bytes = options.memory_floor_gib * GIB_BYTES;
	char storage[BENCH_COPIES_MOST][PATH_MAX];
	const char *paths[BENCH_COPIES_MOST];
	const uint32_t count = split_paths(options.copies, storage, paths);
	if (count == 0 ||
	    (options.raw != 0 ? store_files_open(&bench->store, paths, count) :
				store_pack_open(&bench->store, &bench->source,
						paths, count)) != 0)
		return EXIT_FAILURE;
	if (strcmp(arguments[1], "plan") == 0)
		return trace_plan(bench, &options);
	if (strcmp(arguments[1], "hot_build") == 0)
		return hot_build(bench, &options) == 0 ? EXIT_SUCCESS :
							 EXIT_FAILURE;
	if (strcmp(arguments[1], "scan") == 0)
		return engine_scan(bench, &options) == 0 ? EXIT_SUCCESS :
							   EXIT_FAILURE;
	bench->hot_directory = options.hot_directory;
	if (strcmp(arguments[1], "characterise") == 0) {
		if (options.plan == NULL || options.out == NULL)
			return EXIT_FAILURE;
		FILE *const output = fopen(options.out, "ae");
		if (output == NULL)
			return EXIT_FAILURE;
		const int result = bench_characterise(
			&bench->store, options.plan, output,
			options.skip_done != 0 ? options.out : NULL,
			options.seconds != 0 ?
				bench_now() +
					options.seconds * BENCH_NS_PER_SECOND :
				UINT64_MAX);
		(void)fclose(output);
		return result == 0		 ? EXIT_SUCCESS :
		       result == EXIT_CHECKPOINT ? EXIT_CHECKPOINT :
						   EXIT_FAILURE;
	}
	const bool running = strcmp(arguments[1], "run") == 0;
	if (running && (options.traces == NULL || options.arms == NULL ||
			options.out == NULL))
		return EXIT_FAILURE;
	bench->output = running ? fopen(options.out, "ae") : stdout;
	if (bench->output == NULL)
		return EXIT_FAILURE;
	if (store_digests(&bench->store, (uint32_t)options.prep_threads,
			  options.prep_piece_mib * MIB_BYTES,
			  bench->output) != 0)
		return EXIT_FAILURE;
	if (!running)
		return strcmp(arguments[1], "digests") == 0 ? EXIT_SUCCESS :
							      EXIT_FAILURE;
	if (cases_read(bench, options.traces) != 0)
		return EXIT_FAILURE;
	if (bench_passthrough_prepare(&bench->store) != 0)
		(void)fprintf(stderr,
			      "engram_io_bench: passthrough unavailable: %s\n",
			      bench->store.passthrough_refusal);
	const int error = oracle_build(bench, 0, true);
	(void)fprintf(
		bench->output,
		"{\"schema\":1,\"kind\":\"oracle\",\"status\":\"%s\",\"blocks\":%" PRIu64
		",\"rows\":%" PRIu64
		",\"seconds\":%.3f,\"reads_per_second\":%.1f,\"retries\":%" PRIu64
		",\"cases\":%u}\n",
		error == 0 ? "ok" : "failed", bench->oracle.entries,
		bench->oracle.row_entries, bench->oracle.load_seconds,
		bench->oracle.load_seconds > 0.0 ?
			(double)bench->oracle.load_reads /
				bench->oracle.load_seconds :
			0.0,
		bench->oracle.load_retries, bench->case_count);
	(void)fflush(bench->output);
	if (error != 0)
		return EXIT_FAILURE;
	const int status = arms_run(bench, &options);
	(void)fclose(bench->output);
	engram_disk_close(bench->prod);
	return status == 0		 ? EXIT_SUCCESS :
	       status == EXIT_CHECKPOINT ? EXIT_CHECKPOINT :
					   EXIT_FAILURE;
}
