/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * anemone autotune PACK --stage=engram: the Engram stage (the owner's split, 9 Nov 2026), with its own parameters,
 * mechanism, measurement, state and profile section. It shares with the load stage only the replicas and their search,
 * the schema 3 profile's container (storage_stage_section_*: the member's splice, the writable check, the qualify
 * flip), the pack and copy identity reader and matcher every stage's section holds, the drives' temperature reader and
 * the clock helpers (storage_autotune.h).
 *
 * It tunes serving's step I/O itself (engram_step.h over storage_flight.h): threads a copy, the completion spin window
 * and pinning, over the copies serving's disk store would read: the source, then the receipted replicas that open
 * beside it, each admitted by its medium and its physical device as serving admits it. The store opens as serving's
 * does, with no RAM share and no cache, so every block is read from a drive: the worst case, whose per-read path a
 * store with a RAM share runs for fewer reads. Each point opens a flight of serving's shape (engram_steps_start:
 * 1 + STEP_PIPELINE_MOST steps of ENGRAM_STEP_SLOTS, the read limit within CARD_WAIT_LIMIT_NANOSECONDS) and replays
 * real prompts (--tokens) at seven levels (level_names): decode streams of ENGRAM_TUNE_DECODE_ROWS consecutive tokens a
 * step, prefill streams of chunks of ENGRAM_TUNE_PREFILL_ROWS, and decode beside a prefill. Every stream replays the
 * trace once from its own first case; a level stops at ENGRAM_TUNE_LEVEL_SECONDS_MOST. A step records the time to its
 * first table's rows (the wait serving exposes: table 0 is read at layer 1) and to every table's. Point k reads the
 * trace moved by k * ENGRAM_TUNE_ROTATION_BLOCKS whole blocks, the real dedupe and spread kept. After its levels every
 * point gathers rows of each level again through the per-row path (engram_disk_gather) and compares them with the
 * step path's, byte for byte: a difference refuses the run.
 *
 * The grid's 32 arms run in an order seeded by the trace's identity; a disturbed point (a retired ring, a failed read,
 * a copy left failed, a drive over ENGRAM_TUNE_DRIVE_MILLICELSIUS_MOST) is measured again after the grid, and an arm
 * disturbed twice is out (the run is refused when it is the default's). Then the fastest other arm runs A D D A
 * against serving's default (2 threads a copy, 200 us, pinned) and autotune_engram_verdict decides
 * (storage_autotune.h); a disturbed A D D A point is measured again in its slot, the order kept, and a slot disturbed
 * twice holds the default, the reason named in the section and the selection line. An unmeasured warm-up of the
 * default arm runs before the first point a process measures and again before its first A D D A point (the load
 * stage's lesson, 15 Nov 2026).
 *
 * Progress is kept in <profile>.engram-state after every point; a lease (--seconds, at most
 * AUTOTUNE_ENGRAM_SECONDS_MOST), --points_per_run, a stop signal or a drive over the bound before a point exits
 * AUTOTUNE_RETRY_EXIT, and the same command resumes. At the end the profile's "engram" member is written, enabled 0
 * and qualified "candidate"; serving applies it only once --qualify has made it enabled, and only to the pack, the
 * copies and the shape it names.
 */
#include "engine/storage/autotune/storage_autotune.h"
#include "engram_disk.h"
#include "common/hooks.h"
#include "engine/storage/replicas/replicas.h"
#include "engine/runtime/step/step.h"
#include "engine/storage/io/storage_exl3.h"
#include "engram_step.h"
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <math.h>
#include <pthread.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include "engine/cli/commands.h"

#define ENGRAM_TUNE_STATE_SCHEMA 1u
#define ENGRAM_TUNE_TOKENS_MOST \
	(1u << 18u) /* every --tokens file together: 1 MiB of ids */
#define ENGRAM_TUNE_DECODE_ROWS \
	8u /* a decode step's tokens: 8-row steps with speculation */
#define ENGRAM_TUNE_PREFILL_ROWS 1024u /* a prefill chunk's tokens */
#define ENGRAM_TUNE_LEVEL_SECONDS_MOST 4u
#define ENGRAM_TUNE_ROTATION_BLOCKS \
	7919u /* a prime: a point's whole-block move */
#define ENGRAM_TUNE_CHECK_STEPS \
	4u /* a level's steps the identity check reads again */
#define ENGRAM_TUNE_DRIVE_MILLICELSIUS_MOST \
	75000 /* the drives' 75 C line: a hotter point is disturbed */
#define ENGRAM_TUNE_STREAMS_MOST 8u
#define ENGRAM_TUNE_PERMILLE 1000u
#define ENGRAM_TUNE_PERCENT 100u
#define ENGRAM_TUNE_P99_PERCENT 99u
#define ENGRAM_TUNE_P999_PERMILLE 999u
#define ENGRAM_TUNE_MEDIAN_DIVISOR 2u
#define ENGRAM_TUNE_KEY_TABLE_SHIFT \
	40u /* a (table, block) key: the table above any block number */
#define ENGRAM_TUNE_SEED_DIGITS \
	16u /* the order's seed: the trace identity's first hexadecimal digits */
#define ENGRAM_TUNE_HEX_BASE 16
#define ENGRAM_TUNE_FLAG_TEXT_BYTES 160u
#define NANOSECONDS_PER_SECOND 1000000000ull
#define NANOSECONDS_PER_MICROSECOND 1000.0
#define BYTES_PER_GIGABYTE 1000000000.0
#define MILLIDEGREES_PER_DEGREE 1000.0
#define STATE_SUFFIX ".engram-state"
/* splitmix64 (Steele, Lea and Flood, 2014): the grid order's generator */
#define SPLITMIX_INCREMENT 0x9e3779b97f4a7c15ull
#define SPLITMIX_FIRST_MULTIPLIER 0xbf58476d1ce4e5b9ull
#define SPLITMIX_SECOND_MULTIPLIER 0x94d049bb133111ebull
#define SPLITMIX_FIRST_SHIFT 30u
#define SPLITMIX_SECOND_SHIFT 27u
#define SPLITMIX_THIRD_SHIFT 31u

/* threads a copy, the spin window (us) and pinning; an arm's index: threads + 4 spin + 16 pin, by option places */
static const uint32_t threads_options[] = { 1, 2, 3, 4 };
static const uint32_t spin_us_options[] = { 0, 50, 200, 1000 };
static const uint32_t pin_options[] = { 0, 1 };
#define THREADS_OPTIONS (sizeof(threads_options) / sizeof(threads_options[0]))
#define SPIN_OPTIONS (sizeof(spin_us_options) / sizeof(spin_us_options[0]))
_Static_assert(THREADS_OPTIONS * SPIN_OPTIONS *
			       (sizeof(pin_options) / sizeof(pin_options[0])) ==
		       AUTOTUNE_ENGRAM_GRID_POINTS,
	       "the grid's arms");
/* the A D D A order (storage_autotune.h): the challenger at its first and last point, the default between */
static const uint32_t abba_is_default[AUTOTUNE_ENGRAM_ABBA_POINTS] = { 0, 1, 1,
								       0 };
#define ABBA_ARMS 2u /* the challenger, the default */
#define ABBA_POINTS_PER_ARM 2u
_Static_assert(ABBA_ARMS *ABBA_POINTS_PER_ARM == AUTOTUNE_ENGRAM_ABBA_POINTS,
	       "each arm has two A D D A points");
/* The levels: decode streams of ENGRAM_TUNE_DECODE_ROWS, prefill streams of ENGRAM_TUNE_PREFILL_ROWS; the decode
 * streams' steps are recorded when there are any, else the prefill streams'. */
static const char *const level_names[] = { "D1", "D2", "D4", "D8",
					   "P1", "P2", "M" };
static const uint32_t level_decode[AUTOTUNE_ENGRAM_LEVELS] = { 1, 2, 4, 8,
							       0, 0, 4 };
static const uint32_t level_prefill[AUTOTUNE_ENGRAM_LEVELS] = { 0, 0, 0, 0,
								1, 2, 1 };
#define PREFILL_CHECK_LEVEL \
	5u /* P2: its last stream's first chunk is the check's prefill chunk */
static const char *const phase_names[AUTOTUNE_ENGRAM_PHASES] = {
	"warmup", "grid", "remeasure", "adda"
};
static const char *const disturbance_names[AUTOTUNE_ENGRAM_DISTURBANCES] = {
	"", "retirement", "failed_read", "failed_copy", "hot_drive"
};

static volatile sig_atomic_t engram_stop;
static void engram_signal(int signal_number)
{
	(void)signal_number;
	engram_stop = 1;
}

void autotune_engram_arm_of(uint32_t arm_index, struct autotune_engram_arm *arm)
{
	arm->threads_per_copy = threads_options[arm_index % THREADS_OPTIONS];
	arm->spin_us =
		spin_us_options[arm_index / THREADS_OPTIONS % SPIN_OPTIONS];
	arm->pin = pin_options[arm_index / THREADS_OPTIONS / SPIN_OPTIONS %
			       (sizeof(pin_options) / sizeof(pin_options[0]))];
}

/* Serving's default arm (engram_step.h: ENGRAM_STEP_IO_THREADS_DEFAULT threads a copy, ENGRAM_STEP_SPIN_US_DEFAULT,
 * pinned): its grid index. */
static uint32_t default_arm(void)
{
	uint32_t found = 0;
	for (uint32_t index = 0; index < AUTOTUNE_ENGRAM_GRID_POINTS; index++) {
		struct autotune_engram_arm arm;
		autotune_engram_arm_of(index, &arm);
		if (arm.threads_per_copy == ENGRAM_STEP_IO_THREADS_DEFAULT &&
		    arm.spin_us == ENGRAM_STEP_SPIN_US_DEFAULT && arm.pin != 0)
			found = index;
	}
	return found;
}

void autotune_engram_verdict(const double first_ns[AUTOTUNE_ENGRAM_ABBA_POINTS *
						   AUTOTUNE_ENGRAM_LEVELS],
			     const double cores[AUTOTUNE_ENGRAM_ABBA_POINTS],
			     struct autotune_engram_verdict *verdict)
{
	double logs[ABBA_ARMS] = { 0.0 },
	       point_logs[AUTOTUNE_ENGRAM_ABBA_POINTS] = { 0.0 },
	       arm_cores[ABBA_ARMS] = { 0.0 };
	verdict->blocked_by = AUTOTUNE_ENGRAM_LEVELS;
	for (uint32_t level = 0; level < AUTOTUNE_ENGRAM_LEVELS; level++) {
		double sums[ABBA_ARMS] = { 0.0 }, lows[ABBA_ARMS] = { 0.0 },
		       highs[ABBA_ARMS] = { 0.0 };
		bool seen[ABBA_ARMS] = { false };
		for (uint32_t point = 0; point < AUTOTUNE_ENGRAM_ABBA_POINTS;
		     point++) {
			const uint32_t arm = abba_is_default[point];
			const double value =
				first_ns[point * AUTOTUNE_ENGRAM_LEVELS + level];
			lows[arm] = !seen[arm] || value < lows[arm] ? value :
								      lows[arm];
			highs[arm] = !seen[arm] || value > highs[arm] ?
					     value :
					     highs[arm];
			seen[arm] = true;
			sums[arm] += value;
			point_logs[point] += value > 0.0 ? log(value) :
							   HUGE_VAL;
		}
		double spread = 0.0;
		for (uint32_t arm = 0; arm < ABBA_ARMS; arm++) {
			const double mean =
				sums[arm] / (double)ABBA_POINTS_PER_ARM;
			logs[arm] += mean > 0.0 ? log(mean) : HUGE_VAL;
			const double range =
				mean > 0.0 ? (highs[arm] - lows[arm]) / mean :
					     0.0;
			spread = range > spread ? range : spread;
		}
		const double challenger = sums[0] / (double)ABBA_POINTS_PER_ARM,
			     fallback = sums[1] / (double)ABBA_POINTS_PER_ARM;
		verdict->level_spread[level] = spread;
		verdict->level_loss[level] =
			fallback > 0.0 ? challenger / fallback - 1.0 : 0.0;
		if (verdict->blocked_by == AUTOTUNE_ENGRAM_LEVELS &&
		    verdict->level_loss[level] > spread)
			verdict->blocked_by = level;
	}
	verdict->challenger_cost =
		exp(logs[0] / (double)AUTOTUNE_ENGRAM_LEVELS);
	verdict->default_cost = exp(logs[1] / (double)AUTOTUNE_ENGRAM_LEVELS);
	verdict->spread = 0.0;
	for (uint32_t arm = 0; arm < ABBA_ARMS; arm++) {
		double low = 0.0, high = 0.0, sum = 0.0;
		bool seen = false;
		for (uint32_t point = 0; point < AUTOTUNE_ENGRAM_ABBA_POINTS;
		     point++) {
			if (abba_is_default[point] != arm)
				continue;
			const double cost = exp(point_logs[point] /
						(double)AUTOTUNE_ENGRAM_LEVELS);
			low = !seen || cost < low ? cost : low;
			high = !seen || cost > high ? cost : high;
			seen = true;
			sum += cost;
			arm_cores[arm] +=
				cores[point] / (double)ABBA_POINTS_PER_ARM;
		}
		const double mean = sum / (double)ABBA_POINTS_PER_ARM;
		const double range = mean > 0.0 ? (high - low) / mean : 0.0;
		verdict->spread = range > verdict->spread ? range :
							    verdict->spread;
	}
	verdict->margin =
		verdict->challenger_cost > 0.0 ?
			verdict->default_cost / verdict->challenger_cost - 1.0 :
			0.0;
	verdict->exact_tie =
		llround(verdict->challenger_cost *
			AUTOTUNE_ENGRAM_COST_SCALE) ==
		llround(verdict->default_cost * AUTOTUNE_ENGRAM_COST_SCALE);
	const bool fewer_cores =
		llround(arm_cores[0] * AUTOTUNE_ENGRAM_CORES_SCALE) <
		llround(arm_cores[1] * AUTOTUNE_ENGRAM_CORES_SCALE);
	verdict->default_holds =
		verdict->blocked_by != AUTOTUNE_ENGRAM_LEVELS ||
		(verdict->exact_tie ?
			 !fewer_cores :
			 verdict->challenger_cost >= verdict->default_cost);
	verdict->spread_over_bound = verdict->spread >
				     AUTOTUNE_ENGRAM_SPREAD_BOUND;
}

bool autotune_engram_walk_next(const struct autotune_engram_trace *trace,
			       struct autotune_engram_walk *walk,
			       uint32_t *first_token, uint32_t *rows)
{
	while (walk->walked < trace->cases) {
		const uint32_t which =
			(walk->first_case + walk->walked) % trace->cases;
		const uint32_t length = trace->case_first[which + 1u] -
					trace->case_first[which];
		if (walk->offset < length) {
			*first_token = trace->case_first[which] + walk->offset;
			*rows = length - walk->offset < walk->rows_per_step ?
					length - walk->offset :
					walk->rows_per_step;
			walk->offset += *rows;
			return true;
		}
		walk->walked++;
		walk->offset = 0;
	}
	return false;
}

uint64_t autotune_engram_rotated(uint64_t row_id, uint64_t shift_blocks,
				 uint64_t row_count)
{
	const uint64_t blocks = (row_count + ENGRAM_ROWS_PER_BLOCK - 1u) /
				ENGRAM_ROWS_PER_BLOCK;
	if (blocks == 0)
		return row_id;
	const uint64_t moved =
		(row_id / ENGRAM_ROWS_PER_BLOCK + shift_blocks % blocks) %
			blocks * ENGRAM_ROWS_PER_BLOCK +
		row_id % ENGRAM_ROWS_PER_BLOCK;
	return moved < row_count ? moved : row_id;
}

static int compare_words(const void *left, const void *right)
{
	const uint64_t first = *(const uint64_t *)left,
		       second = *(const uint64_t *)right;
	return first < second ? -1 : first > second ? 1 : 0;
}

/* One --tokens file's ids, read whole: a regular file (no link) of whole uint32 ids, within the trace's room. */
static uint32_t *tokens_read(const char *path, uint32_t room, uint32_t *count)
{
	const int descriptor =
		open(path, O_RDONLY | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK);
	struct stat file;
	bool valid = descriptor >= 0 && fstat(descriptor, &file) == 0 &&
		     S_ISREG(file.st_mode) && file.st_size > 0 &&
		     (uint64_t)file.st_size % sizeof(uint32_t) == 0 &&
		     (uint64_t)file.st_size / sizeof(uint32_t) <= room;
	uint32_t *const tokens = valid ? malloc((size_t)file.st_size) : NULL;
	size_t done = 0;
	while (tokens != NULL && done < (size_t)file.st_size) {
		const ssize_t amount = read(descriptor,
					    (unsigned char *)tokens + done,
					    (size_t)file.st_size - done);
		if (amount < 0 && errno == EINTR)
			continue;
		if (amount <= 0)
			break;
		done += (size_t)amount;
	}
	if (descriptor >= 0)
		(void)close(descriptor);
	valid = tokens != NULL && done == (size_t)file.st_size;
	if (!valid) {
		free(tokens);
		return NULL;
	}
	*count = (uint32_t)(done / sizeof(uint32_t));
	return tokens;
}

result_status autotune_engram_trace_open(const char *const *paths,
					 uint32_t count,
					 const struct engram_record *ngrams,
					 struct autotune_engram_trace *trace,
					 uint32_t *failed)
{
	const struct engram_description *const description =
		engram_description_of(ngrams);
	if (paths == NULL || description == NULL || trace == NULL ||
	    failed == NULL || count == 0 || count > AUTOTUNE_ENGRAM_TRACES_MOST)
		return STATUS_INVALID_ARGUMENT;
	*failed = count;
	memset(trace, 0, sizeof(*trace));
	trace->references_per_token =
		description->table_count * description->rows_per_token;
	uint32_t *cases[AUTOTUNE_ENGRAM_TRACES_MOST] = { NULL };
	char hashes[AUTOTUNE_ENGRAM_TRACES_MOST][STORAGE_PROFILE_KEY_BYTES];
	memset(hashes, 0, sizeof(hashes));
	result_status status = STATUS_OK;
	for (uint32_t index = 0; status == STATUS_OK && index < count;
	     index++) {
		uint32_t tokens = 0;
		cases[index] = tokens_read(
			paths[index], ENGRAM_TUNE_TOKENS_MOST - trace->tokens,
			&tokens);
		status = cases[index] == NULL ?
				 STATUS_INVALID_ARGUMENT :
				 prepare_hash(cases[index],
					      (size_t)tokens * sizeof(uint32_t),
					      hashes[index]);
		*failed = status != STATUS_OK ? index : count;
		trace->case_first[index] = trace->tokens;
		trace->tokens += tokens;
		trace->case_first[index + 1u] = trace->tokens;
	}
	trace->cases = count;
	trace->row_ids = status == STATUS_OK ?
				 calloc((size_t)trace->tokens *
						trace->references_per_token,
					sizeof(uint64_t)) :
				 NULL;
	if (status == STATUS_OK && trace->row_ids == NULL)
		status = STATUS_OUT_OF_MEMORY;
	for (uint32_t index = 0; status == STATUS_OK && index < count;
	     index++) {
		for (uint32_t position = 0;
		     status == STATUS_OK &&
		     position < trace->case_first[index + 1u] -
					trace->case_first[index];
		     position++)
			status = engram_row_ids_entry(
				ngrams, cases[index], NULL, position,
				trace->row_ids +
					((size_t)trace->case_first[index] +
					 position) *
						trace->references_per_token);
		*failed = status != STATUS_OK ? index : count;
	}
	for (uint32_t index = 0; index < count; index++)
		free(cases[index]);
	if (status == STATUS_OK)
		status = prepare_hash(hashes, sizeof(hashes), trace->identity);
	/* the distinct (table, block) pairs the trace reads */
	const size_t references =
		(size_t)trace->tokens * trace->references_per_token;
	uint64_t *const keys = status == STATUS_OK ?
				       malloc(references * sizeof(uint64_t)) :
				       NULL;
	if (status == STATUS_OK && keys == NULL)
		status = STATUS_OUT_OF_MEMORY;
	for (size_t index = 0; keys != NULL && index < references; index++)
		keys[index] = (uint64_t)(index % trace->references_per_token /
					 description->rows_per_token)
				      << ENGRAM_TUNE_KEY_TABLE_SHIFT |
			      trace->row_ids[index] / ENGRAM_ROWS_PER_BLOCK;
	if (keys != NULL) {
		qsort(keys, references, sizeof(*keys), compare_words);
		for (size_t index = 0; index < references; index++)
			trace->unique_blocks +=
				index == 0 || keys[index] != keys[index - 1u] ?
					1u :
					0u;
	}
	free(keys);
	if (status != STATUS_OK)
		autotune_engram_trace_close(trace);
	return status;
}

void autotune_engram_trace_close(struct autotune_engram_trace *trace)
{
	free(trace->row_ids);
	trace->row_ids = NULL;
}

/* One stream of a level: its step record, its walk, its rows' rotation, and its steps' times. */
struct stream {
	struct engram_step *step;
	const struct autotune_engram_trace *trace;
	const struct engram_description *description;
	struct autotune_engram_walk walk;
	uint64_t shift_blocks, deadline_ns, count, capacity, rows;
	uint64_t *first_ns, *all_ns, *row_ids;
	unsigned char *destinations[ENGRAM_MAXIMUM_TABLES];
	result_status status;
};

/* rows tokens' row ids from first_token on, each moved by the stream's shift, into the stream's buffer. */
static void rows_rotated(const struct stream *stream, uint32_t first_token,
			 uint32_t rows)
{
	const uint32_t per_token = stream->trace->references_per_token,
		       per_table = stream->description->rows_per_token;
	const uint64_t *const source =
		stream->trace->row_ids + (size_t)first_token * per_token;
	for (size_t index = 0; index < (size_t)rows * per_token; index++)
		stream->row_ids[index] = autotune_engram_rotated(
			source[index], stream->shift_blocks,
			stream->description
				->row_count[index % per_token / per_table]);
}

/* Issues one step and waits for every table, the first first: its times, when times is true. */
MUST_CHECK static result_status step_run(struct stream *stream, uint32_t rows,
					 bool times)
{
	const uint64_t begin = autotune_now_nanoseconds();
	result_status status = engram_step_issue(stream->step, stream->row_ids,
						 rows, stream->destinations);
	uint64_t first = 0;
	for (uint32_t table = 0;
	     status == STATUS_OK && table < stream->description->table_count;
	     table++) {
		status = engram_step_wait(stream->step, table);
		first = table == 0 ? autotune_now_nanoseconds() : first;
	}
	if (status == STATUS_OK && times && stream->count < stream->capacity) {
		stream->first_ns[stream->count] = first - begin;
		stream->all_ns[stream->count++] =
			autotune_now_nanoseconds() - begin;
		stream->rows += rows;
	}
	return status;
}

static void *stream_run(void *argument)
{
	struct stream *const stream = argument;
	uint32_t first_token = 0, rows = 0;
	while (stream->status == STATUS_OK &&
	       autotune_now_nanoseconds() < stream->deadline_ns &&
	       autotune_engram_walk_next(stream->trace, &stream->walk,
					 &first_token, &rows)) {
		rows_rotated(stream, first_token, rows);
		stream->status = step_run(stream, rows, true);
	}
	return NULL;
}

/* A stream's buffers for steps of rows_per_step: its rotated ids, its destinations and, with steps, its times. */
MUST_CHECK static result_status
stream_allocate(struct stream *stream, uint32_t rows_per_step, uint64_t steps)
{
	const struct engram_description *const description =
		stream->description;
	stream->row_ids =
		malloc((size_t)rows_per_step *
		       stream->trace->references_per_token * sizeof(uint64_t));
	stream->first_ns = steps != 0 ? malloc(steps * sizeof(uint64_t)) : NULL;
	stream->all_ns = steps != 0 ? malloc(steps * sizeof(uint64_t)) : NULL;
	stream->capacity = steps;
	bool valid = stream->row_ids != NULL &&
		     (steps == 0 ||
		      (stream->first_ns != NULL && stream->all_ns != NULL));
	for (uint32_t table = 0; valid && table < description->table_count;
	     table++) {
		stream->destinations[table] =
			malloc((size_t)rows_per_step *
			       description->rows_per_token * ENGRAM_ROW_BYTES);
		valid = stream->destinations[table] != NULL;
	}
	return valid ? STATUS_OK : STATUS_OUT_OF_MEMORY;
}

/* A stream's buffers go, unless its step was kept (a read that did not return in its limit may still write into its
 * destinations: they stay allocated, as engram_step.h requires). */
static void stream_release(struct stream *stream, bool kept)
{
	free(stream->row_ids);
	free(stream->first_ns);
	free(stream->all_ns);
	for (uint32_t table = 0; !kept && table < ENGRAM_MAXIMUM_TABLES;
	     table++)
		free(stream->destinations[table]);
	memset(stream, 0, sizeof(*stream));
}

/* The steps of a stream that replays the trace once in steps of rows_per_step. */
static uint64_t trace_steps(const struct autotune_engram_trace *trace,
			    uint32_t rows_per_step)
{
	struct autotune_engram_walk walk = { .rows_per_step = rows_per_step };
	uint64_t steps = 0;
	uint32_t first_token = 0, rows = 0;
	while (autotune_engram_walk_next(trace, &walk, &first_token, &rows))
		steps++;
	return steps;
}

static uint64_t quantile(const uint64_t *sorted, uint64_t count, uint64_t part,
			 uint64_t parts)
{
	return count != 0 ? sorted[(count - 1u) * part / parts] : 0;
}

/* One level: its streams on threads of their own, each from its own first case; the recorded streams' times. */
MUST_CHECK static result_status
level_run(struct engram_steps *steps, struct engram_disk *disk,
	  const struct autotune_engram_trace *trace, uint32_t level,
	  uint64_t shift_blocks, struct autotune_engram_level *record)
{
	const uint32_t decode = level_decode[level],
		       streams_count = decode + level_prefill[level];
	struct stream streams[ENGRAM_TUNE_STREAMS_MOST];
	pthread_t threads[ENGRAM_TUNE_STREAMS_MOST];
	memset(streams, 0, sizeof(streams));
	const uint64_t deadline = autotune_now_nanoseconds() +
				  (uint64_t)ENGRAM_TUNE_LEVEL_SECONDS_MOST *
					  NANOSECONDS_PER_SECOND;
	result_status status = STATUS_OK;
	uint32_t created = 0, launched = 0;
	for (uint32_t index = 0; status == STATUS_OK && index < streams_count;
	     index++) {
		struct stream *const stream = &streams[index];
		const bool recorded = decode == 0 || index < decode;
		const uint32_t rows_per_step = index < decode ?
						       ENGRAM_TUNE_DECODE_ROWS :
						       ENGRAM_TUNE_PREFILL_ROWS;
		stream->trace = trace;
		stream->description = engram_disk_description(disk);
		stream->walk = (struct autotune_engram_walk){
			.first_case = index * trace->cases / streams_count,
			.rows_per_step = rows_per_step
		};
		stream->shift_blocks = shift_blocks;
		stream->deadline_ns = deadline;
		status = stream_allocate(
			stream, rows_per_step,
			recorded ? trace_steps(trace, rows_per_step) : 0);
		if (status == STATUS_OK)
			status = engram_step_create(steps, &stream->step);
		created += status == STATUS_OK ? 1u : 0u;
	}
	const uint64_t begin = autotune_now_nanoseconds();
	for (uint32_t index = 0; status == STATUS_OK && index < created;
	     index++) {
		if (pthread_create(&threads[index], NULL, stream_run,
				   &streams[index]) != 0)
			status = STATUS_OUT_OF_MEMORY;
		else
			launched++;
	}
	for (uint32_t index = 0; index < launched; index++)
		if (pthread_join(threads[index], NULL) != 0)
			status = STATUS_INPUT_OUTPUT_ERROR;
	record->wall_ns = autotune_now_nanoseconds() - begin;
	uint64_t total = 0;
	for (uint32_t index = 0; index < created; index++) {
		status = status == STATUS_OK ? streams[index].status : status;
		total += streams[index].count;
	}
	uint64_t *const first = malloc((total + 1u) * sizeof(uint64_t));
	uint64_t *const all = malloc((total + 1u) * sizeof(uint64_t));
	if (status == STATUS_OK && (first == NULL || all == NULL))
		status = STATUS_OUT_OF_MEMORY;
	for (uint32_t index = 0, kept = 0; index < streams_count; index++) {
		const struct stream *const stream = &streams[index];
		if (status == STATUS_OK && stream->count != 0) {
			memcpy(first + kept, stream->first_ns,
			       stream->count * sizeof(uint64_t));
			memcpy(all + kept, stream->all_ns,
			       stream->count * sizeof(uint64_t));
			kept += (uint32_t)stream->count;
			record->rows += stream->rows;
		}
		const bool destroyed = stream->step == NULL ||
				       engram_step_destroy(stream->step);
		if (!destroyed && status == STATUS_OK)
			status = STATUS_TIMED_OUT;
		stream_release(&streams[index], !destroyed);
	}
	if (status == STATUS_OK) {
		qsort(first, total, sizeof(*first), compare_words);
		qsort(all, total, sizeof(*all), compare_words);
		record->steps = total;
		record->first_p50_ns =
			quantile(first, total, 1, ENGRAM_TUNE_MEDIAN_DIVISOR);
		record->first_p99_ns = quantile(first, total,
						ENGRAM_TUNE_P99_PERCENT,
						ENGRAM_TUNE_PERCENT);
		record->first_p999_ns = quantile(first, total,
						 ENGRAM_TUNE_P999_PERMILLE,
						 ENGRAM_TUNE_PERMILLE);
		record->all_p50_ns =
			quantile(all, total, 1, ENGRAM_TUNE_MEDIAN_DIVISOR);
		record->all_p99_ns = quantile(all, total,
					      ENGRAM_TUNE_P99_PERCENT,
					      ENGRAM_TUNE_PERCENT);
	}
	free(first);
	free(all);
	return status;
}

/* The identity check: each level's last stream's first ENGRAM_TUNE_CHECK_STEPS steps, then P2's last stream's first
 * chunk, read again through the step path and gathered through the per-row path; the bytes must be the same. */
MUST_CHECK static result_status
identity_check(struct engram_steps *steps, struct engram_disk *disk,
	       const struct autotune_engram_trace *trace, uint64_t shift_blocks,
	       uint32_t rotation)
{
	const struct engram_description *const description =
		engram_disk_description(disk);
	struct stream stream = { .trace = trace,
				 .description = description,
				 .shift_blocks = shift_blocks };
	result_status status =
		stream_allocate(&stream, ENGRAM_TUNE_PREFILL_ROWS, 0);
	const size_t table_bytes = (size_t)ENGRAM_TUNE_PREFILL_ROWS *
				   description->rows_per_token *
				   ENGRAM_ROW_BYTES;
	unsigned char *const gathered = malloc(table_bytes);
	if (status == STATUS_OK && gathered == NULL)
		status = STATUS_OUT_OF_MEMORY;
	if (status == STATUS_OK)
		status = engram_step_create(steps, &stream.step);
	for (uint32_t level = 0;
	     status == STATUS_OK && level <= AUTOTUNE_ENGRAM_LEVELS; level++) {
		const bool chunk = level == AUTOTUNE_ENGRAM_LEVELS;
		const uint32_t shape = chunk ? PREFILL_CHECK_LEVEL : level;
		const uint32_t decode = level_decode[shape],
			       streams_count = decode + level_prefill[shape];
		const uint32_t last =
			(chunk || decode == 0 ? streams_count : decode) - 1u;
		stream.walk = (struct autotune_engram_walk){
			.first_case = last * trace->cases / streams_count,
			.rows_per_step = last < decode ?
						 ENGRAM_TUNE_DECODE_ROWS :
						 ENGRAM_TUNE_PREFILL_ROWS
		};
		uint32_t first_token = 0, rows = 0;
		for (uint32_t step = 0;
		     status == STATUS_OK &&
		     step < (chunk ? 1u : ENGRAM_TUNE_CHECK_STEPS) &&
		     autotune_engram_walk_next(trace, &stream.walk,
					       &first_token, &rows);
		     step++) {
			rows_rotated(&stream, first_token, rows);
			status = step_run(&stream, rows, false);
			for (uint32_t table = 0;
			     status == STATUS_OK &&
			     table < description->table_count;
			     table++) {
				const uint32_t per_table =
					description->rows_per_token;
				/* a token's rows of the table a call, as serving's per-row path asks for them (step.c):
				 * engram_disk_gather refuses more than ENGRAM_MAXIMUM_ROWS_PER_TOKEN rows a call */
				for (uint32_t row = 0;
				     status == STATUS_OK && row < rows; row++)
					status = engram_disk_gather(
						disk, table,
						stream.row_ids +
							((size_t)row *
								 description
									 ->table_count +
							 table) *
								per_table,
						per_table,
						gathered +
							(size_t)row *
								per_table *
								ENGRAM_ROW_BYTES);
				const size_t bytes = (size_t)rows * per_table *
						     ENGRAM_ROW_BYTES;
				/* a checking build's fault point: one byte of the step path's rows changed */
				if (FAULT_POINT(engram_tune_compare))
					stream.destinations[table][0] =
						(unsigned char)(stream.destinations
									[table]
									[0] ^
								1u);
				if (status == STATUS_OK &&
				    memcmp(stream.destinations[table], gathered,
					   bytes) != 0) {
					(void)fprintf(
						stderr,
						"autotune: refused: the step path's rows differ from the "
						"per-row path's: "
						"point %" PRIu32
						" level %s table %" PRIu32
						" step %" PRIu32 "\n",
						rotation + 1u,
						chunk ? "P2 chunk" :
							level_names[level],
						table, step + 1u);
					status = STATUS_CORRUPT_PACK;
				}
			}
		}
	}
	const bool destroyed = stream.step == NULL ||
			       engram_step_destroy(stream.step);
	if (!destroyed && status == STATUS_OK)
		status = STATUS_TIMED_OUT;
	stream_release(&stream, !destroyed);
	free(gathered);
	return status;
}

result_status autotune_engram_point_measure(
	struct engram_disk *disk, const struct autotune_engram_trace *trace,
	uint32_t rotation, bool check, struct autotune_engram_point *point)
{
	if (disk == NULL || trace == NULL || point == NULL ||
	    trace->tokens == 0)
		return STATUS_INVALID_ARGUMENT;
	/* at most STORAGE_MAXIMUM_COPIES by the store's construction; bounded here so that every per-copy array's index
	 * is seen in range (gcc 14 at -O3 -flto: stringop-overflow on verified[] otherwise) */
	const uint32_t store_copies = engram_disk_copy_count(disk);
	const uint32_t copies = store_copies < STORAGE_MAXIMUM_COPIES ?
					store_copies :
					STORAGE_MAXIMUM_COPIES;
	struct engram_steps_options options = {
		.threads_per_copy = point->arm.threads_per_copy,
		.steps = 1u + STEP_PIPELINE_MOST,
		.slots_per_step = ENGRAM_STEP_SLOTS,
		.rows_most = ENGRAM_TUNE_PREFILL_ROWS,
		.spin_nanoseconds = (uint64_t)point->arm.spin_us *
				    (uint64_t)NANOSECONDS_PER_MICROSECOND,
		.read_limit_nanoseconds = engram_steps_read_limit_within(
			CARD_WAIT_LIMIT_NANOSECONDS, copies)
	};
	const struct storage_reader *const copies_reader =
		engram_disk_reader(disk);
	for (uint32_t copy = 0; copy < STORAGE_MAXIMUM_COPIES; copy++)
		for (uint32_t thread = 0;
		     thread < STORAGE_FLIGHT_THREADS_PER_COPY_MOST; thread++)
			options.cpus[copy][thread] =
				point->arm.pin != 0 && copy < copies ?
					storage_flight_queue_cpu(copies_reader,
								 copy, thread) :
					STORAGE_FLIGHT_CPU_FREE;
	/* every copy starts unfailed, so one point's failure does not steer the next one's copies */
	for (uint32_t copy = 0; copy < copies; copy++)
		engram_disk_copy_state(disk, copy, false);
	uint64_t verified[STORAGE_MAXIMUM_COPIES] = { 0 };
	for (uint32_t copy = 0; copy < copies; copy++)
		verified[copy] = engram_disk_copy_verified(disk, copy);
	point->rotation = rotation;
	autotune_drives_read(&point->before);
	struct engram_steps *steps = NULL;
	result_status status = engram_steps_open(disk, &options, &steps);
	const uint64_t shift_blocks =
		(uint64_t)rotation * ENGRAM_TUNE_ROTATION_BLOCKS;
	const uint64_t cpu_begin = autotune_cpu_nanoseconds();
	const uint64_t begin = autotune_now_nanoseconds();
	for (uint32_t level = 0;
	     status == STATUS_OK && level < AUTOTUNE_ENGRAM_LEVELS; level++)
		status = level_run(steps, disk, trace, level, shift_blocks,
				   &point->levels[level]);
	point->wall_ns = autotune_now_nanoseconds() - begin;
	point->cpu_ns = autotune_cpu_nanoseconds() - cpu_begin;
	struct storage_flight_counts counts;
	storage_flight_counts(engram_steps_flight(steps), &counts);
	point->answered = counts.answered;
	point->retirements =
		counts.deadline_retirements + counts.other_retirements;
	point->plain_rings =
		storage_flight_plain_rings(engram_steps_flight(steps));
	point->failed_reads = engram_steps_failed_reads(steps);
	for (uint32_t copy = 0; copy < copies; copy++)
		point->copy_reads[copy] =
			engram_disk_copy_verified(disk, copy) - verified[copy];
	const uint32_t failed_copies = engram_disk_failed_copies(disk);
	if (status == STATUS_OK && check)
		status = identity_check(steps, disk, trace, shift_blocks,
					rotation);
	engram_steps_close(steps);
	autotune_drives_read(&point->after);
	bool hot = false;
	for (uint32_t drive = 0; drive < point->after.count; drive++)
		hot = hot || point->after.millicelsius[drive] >
				     ENGRAM_TUNE_DRIVE_MILLICELSIUS_MOST;
	point->disturbance =
		point->retirements != 0	 ? AUTOTUNE_ENGRAM_RETIREMENT :
		point->failed_reads != 0 ? AUTOTUNE_ENGRAM_FAILED_READ :
		failed_copies != 0	 ? AUTOTUNE_ENGRAM_FAILED_COPY :
		hot			 ? AUTOTUNE_ENGRAM_HOT_DRIVE :
					   AUTOTUNE_ENGRAM_QUIET;
	return status;
}

/* A point's cost: the geometric mean over the levels of its first table's p50 (infinite for a level with no step). */
static double point_cost(const struct autotune_engram_point *point)
{
	double sum = 0.0;
	for (uint32_t level = 0; level < AUTOTUNE_ENGRAM_LEVELS; level++)
		sum += point->levels[level].first_p50_ns != 0 ?
			       log((double)point->levels[level].first_p50_ns) :
			       HUGE_VAL;
	return exp(sum / (double)AUTOTUNE_ENGRAM_LEVELS);
}

/* The grid's order: the arms shuffled (Fisher and Yates) by splitmix64 from the state's seed. */
static void grid_order(uint64_t order_seed,
		       uint32_t order[AUTOTUNE_ENGRAM_GRID_POINTS])
{
	for (uint32_t index = 0; index < AUTOTUNE_ENGRAM_GRID_POINTS; index++)
		order[index] = index;
	for (uint32_t index = AUTOTUNE_ENGRAM_GRID_POINTS - 1u; index > 0;
	     index--) {
		order_seed += SPLITMIX_INCREMENT;
		uint64_t mixed = order_seed;
		mixed = (mixed ^ (mixed >> SPLITMIX_FIRST_SHIFT)) *
			SPLITMIX_FIRST_MULTIPLIER;
		mixed = (mixed ^ (mixed >> SPLITMIX_SECOND_SHIFT)) *
			SPLITMIX_SECOND_MULTIPLIER;
		mixed ^= mixed >> SPLITMIX_THIRD_SHIFT;
		const uint32_t other = (uint32_t)(mixed % (index + 1u));
		const uint32_t held = order[index];
		order[index] = order[other];
		order[other] = held;
	}
}

/* The grid's disturbed points in order: their count, and with a list their indices. */
static uint32_t disturbed_grid(const struct autotune_engram_state *state,
			       uint32_t list[AUTOTUNE_ENGRAM_GRID_POINTS])
{
	uint32_t count = 0;
	for (uint32_t index = 0;
	     index < AUTOTUNE_ENGRAM_GRID_POINTS && index < state->point_count;
	     index++)
		if (state->points[index].disturbance != AUTOTUNE_ENGRAM_QUIET)
			list[count++] = index;
	return count;
}

/* An arm's undisturbed point: its grid point, else its re-measured one; NULL when both were disturbed. */
static const struct autotune_engram_point *
arm_quiet(const struct autotune_engram_state *state, uint32_t arm_index,
	  uint32_t measured)
{
	for (uint32_t index = 0; index < measured; index++)
		if (state->points[index].arm_index == arm_index &&
		    state->points[index].disturbance == AUTOTUNE_ENGRAM_QUIET)
			return &state->points[index];
	return NULL;
}

/* After the grid and its re-measured points: the default arm's index and the challenger's; false with refusal when
 * one has no undisturbed point. */
static bool arms_ranked(const struct autotune_engram_state *state,
			uint32_t measured, uint32_t *challenger,
			const char **refusal)
{
	if (arm_quiet(state, default_arm(), measured) == NULL) {
		*refusal =
			"the default arm was disturbed twice: not quiet, or a drive is failing";
		return false;
	}
	double best = HUGE_VAL;
	*challenger = AUTOTUNE_ENGRAM_GRID_POINTS;
	for (uint32_t arm = 0; arm < AUTOTUNE_ENGRAM_GRID_POINTS; arm++) {
		const struct autotune_engram_point *const point =
			arm_quiet(state, arm, measured);
		if (arm == default_arm() || point == NULL ||
		    !(point_cost(point) < best))
			continue;
		best = point_cost(point);
		*challenger = arm;
	}
	if (*challenger == AUTOTUNE_ENGRAM_GRID_POINTS) {
		*refusal =
			"every other arm was disturbed twice: not quiet, or a drive is failing";
		return false;
	}
	return true;
}

/* The A D D A's slots from the points after measured, in order: a quiet point holds its slot, a disturbed one leaves
 * it to be measured again, and a slot's second disturbed point ends the A D D A (*twice). The slots held. */
static uint32_t adda_slots(const struct autotune_engram_state *state,
			   uint32_t measured,
			   uint32_t slots[AUTOTUNE_ENGRAM_ABBA_POINTS],
			   bool *twice)
{
	uint32_t held = 0;
	bool again = false;
	*twice = false;
	for (uint32_t index = measured;
	     index < state->point_count && held < AUTOTUNE_ENGRAM_ABBA_POINTS &&
	     !*twice;
	     index++) {
		const bool slot_quiet = state->points[index].disturbance ==
					AUTOTUNE_ENGRAM_QUIET;
		if (slot_quiet)
			slots[held++] = index;
		*twice = !slot_quiet && again;
		again = !slot_quiet;
	}
	return held;
}

bool autotune_engram_next(const struct autotune_engram_state *state,
			  uint32_t *arm_index, uint32_t *phase,
			  const char **refusal)
{
	*refusal = NULL;
	const uint32_t done = state->point_count;
	if (done < AUTOTUNE_ENGRAM_GRID_POINTS) {
		uint32_t order[AUTOTUNE_ENGRAM_GRID_POINTS];
		grid_order(state->seed, order);
		*arm_index = order[done];
		*phase = AUTOTUNE_ENGRAM_GRID;
		return true;
	}
	uint32_t list[AUTOTUNE_ENGRAM_GRID_POINTS];
	const uint32_t disturbed = disturbed_grid(state, list);
	const uint32_t measured = AUTOTUNE_ENGRAM_GRID_POINTS + disturbed;
	if (done < measured) {
		*arm_index =
			state->points[list[done - AUTOTUNE_ENGRAM_GRID_POINTS]]
				.arm_index;
		*phase = AUTOTUNE_ENGRAM_REMEASURE;
		return true;
	}
	uint32_t challenger = 0, slots[AUTOTUNE_ENGRAM_ABBA_POINTS];
	bool twice = false;
	const uint32_t held = adda_slots(state, measured, slots, &twice);
	if (held == AUTOTUNE_ENGRAM_ABBA_POINTS || twice ||
	    !arms_ranked(state, measured, &challenger, refusal))
		return false;
	*arm_index = abba_is_default[held] != 0 ? default_arm() : challenger;
	*phase = AUTOTUNE_ENGRAM_ADDA;
	return true;
}

bool autotune_engram_select(const struct autotune_engram_state *state,
			    struct autotune_engram_selection *selection,
			    const char **refusal)
{
	*refusal = NULL;
	memset(selection, 0, sizeof(*selection));
	uint32_t list[AUTOTUNE_ENGRAM_GRID_POINTS], arm = 0, phase = 0;
	selection->measured =
		AUTOTUNE_ENGRAM_GRID_POINTS + disturbed_grid(state, list);
	if (state->point_count < selection->measured ||
	    autotune_engram_next(state, &arm, &phase, refusal) ||
	    !arms_ranked(state, selection->measured, &selection->challenger,
			 refusal))
		return false;
	selection->held = adda_slots(state, selection->measured,
				     selection->slots,
				     &selection->adda_disturbed_twice);
	struct autotune_engram_verdict *const verdict = &selection->verdict;
	verdict->blocked_by = AUTOTUNE_ENGRAM_LEVELS;
	verdict->default_holds = true;
	if (selection->adda_disturbed_twice)
		return true;
	double first_ns[AUTOTUNE_ENGRAM_ABBA_POINTS * AUTOTUNE_ENGRAM_LEVELS],
		cores[AUTOTUNE_ENGRAM_ABBA_POINTS];
	for (uint32_t slot = 0; slot < AUTOTUNE_ENGRAM_ABBA_POINTS; slot++) {
		const struct autotune_engram_point *const abba =
			&state->points[selection->slots[slot]];
		for (uint32_t level = 0; level < AUTOTUNE_ENGRAM_LEVELS;
		     level++)
			first_ns[slot * AUTOTUNE_ENGRAM_LEVELS + level] =
				(double)abba->levels[level].first_p50_ns;
		cores[slot] =
			abba->wall_ns != 0 ?
				(double)abba->cpu_ns / (double)abba->wall_ns :
				0.0;
	}
	autotune_engram_verdict(first_ns, cores, verdict);
	return true;
}

static bool state_identity(const struct autotune_engram_state *first,
			   const struct autotune_engram_state *second)
{
	return first->schema == second->schema &&
	       first->copy_count == second->copy_count &&
	       first->steps == second->steps && first->slots == second->slots &&
	       first->file_byte_count == second->file_byte_count &&
	       first->seed == second->seed &&
	       strcmp(first->pack_identity, second->pack_identity) == 0 &&
	       strcmp(first->trace_identity, second->trace_identity) == 0 &&
	       memcmp(first->keys, second->keys, sizeof(first->keys)) == 0;
}

void autotune_engram_state_resume(const char *path,
				  struct autotune_engram_state *state)
{
	struct autotune_engram_state *const prior = malloc(sizeof(*prior));
	if (prior == NULL)
		return;
	const int descriptor =
		open(path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
	struct stat file;
	bool valid = descriptor >= 0 && fstat(descriptor, &file) == 0 &&
		     S_ISREG(file.st_mode) && file.st_uid == geteuid() &&
		     (file.st_mode & (S_IWGRP | S_IWOTH)) == 0 &&
		     file.st_size == (off_t)sizeof(*prior) &&
		     read(descriptor, prior, sizeof(*prior)) ==
			     (ssize_t)sizeof(*prior) &&
		     state_identity(prior, state) &&
		     prior->point_count <= AUTOTUNE_ENGRAM_POINTS_MOST;
	for (uint32_t side = 0; valid && side < AUTOTUNE_ENGRAM_WARMUP_SIDES;
	     side++)
		valid = prior->warmups[side].phase == AUTOTUNE_ENGRAM_WARMUP &&
			prior->warmups[side].disturbance <
				AUTOTUNE_ENGRAM_DISTURBANCES &&
			prior->warmups[side].before.count <=
				AUTOTUNE_DRIVES_MOST &&
			prior->warmups[side].after.count <=
				AUTOTUNE_DRIVES_MOST;
	/* every point where the schedule puts it, each record in its ranges */
	const uint32_t points = valid ? prior->point_count : 0;
	for (uint32_t index = 0; valid && index < points; index++) {
		uint32_t arm = 0, phase = 0;
		const char *refusal = NULL;
		prior->point_count = index;
		const struct autotune_engram_point *const point =
			&prior->points[index];
		valid = autotune_engram_next(prior, &arm, &phase, &refusal) &&
			point->arm_index == arm && point->phase == phase &&
			point->rotation == index &&
			point->disturbance < AUTOTUNE_ENGRAM_DISTURBANCES &&
			point->before.count <= AUTOTUNE_DRIVES_MOST &&
			point->after.count <= AUTOTUNE_DRIVES_MOST;
	}
	if (valid) {
		prior->point_count = points;
		*state = *prior;
	}
	free(prior);
	if (descriptor >= 0)
		(void)close(descriptor);
}

MUST_CHECK static result_status
state_write(const char *path, const struct autotune_engram_state *state)
{
	struct prepare_file file;
	const result_status status = prepare_file_open(path, &file);
	if (status != STATUS_OK)
		return status;
	if (fwrite(state, sizeof(*state), 1, file.stream) != 1u) {
		prepare_file_abort(&file);
		return STATUS_INPUT_OUTPUT_ERROR;
	}
	return prepare_file_commit(&file);
}

/* The copies serving's disk store would read, the pack, its n-gram object, and (for a run) the store. */
struct engram_context {
	struct storage_reader *reader, *sources;
	struct pack_record *pack;
	struct engram_record *engram;
	struct engram_disk *disk;
	const char *copy_paths[STORAGE_MAXIMUM_COPIES];
	const char *admitted[STORAGE_MAXIMUM_COPIES];
	char replica_paths[REPLICAS_MOST][REPLICA_PATH_CAPACITY];
	char filesystems[STORAGE_MAXIMUM_COPIES][LOCAL_DEVICE_FILESYSTEM_BYTES];
	uint64_t scan_bytes, scan_ns;
};

/* The source, then the receipted replicas that open beside it (one line names each one left out), each admitted as
 * serving's disk store admits its copies (engram_sources_prepare: a refused medium, a duplicate physical device); their
 * keys and filesystems; the pack and its n-gram object. */
MUST_CHECK static result_status
context_open(struct engram_context *context, const char *source, bool external,
	     struct autotune_engram_state *state)
{
	struct pack_header header = { 0 };
	uint64_t source_bytes = 0;
	if (storage_for_path(source) != &storage_direct) {
		(void)autotune_refused(
			"an EXL3 map: the disk store reads packs (give the pack its flat copies hold)",
			source);
		return STATUS_UNSUPPORTED;
	}
	if (!storage_pack_header_read(source, &header, &source_bytes)) {
		(void)autotune_refused("the pack does not open", source);
		return STATUS_INPUT_OUTPUT_ERROR;
	}
	const uint32_t found = replicas_find_policy(&source, 1u, &header,
						    context->replica_paths,
						    REPLICAS_MOST, external);
	const uint32_t kept = replicas_reader_open(
		&storage_direct, &source, 1u,
		(const char (*)[REPLICA_PATH_CAPACITY])context->replica_paths,
		found, context->copy_paths, &context->reader, stderr);
	context->copy_paths[0] = source;
	result_status status =
		kept != 0 ? STATUS_OK :
			    storage_direct.reader_open(context->copy_paths, 1,
						       &context->reader);
	struct local_device_description devices[STORAGE_MAXIMUM_COPIES];
	for (uint32_t copy = 0; status == STATUS_OK && copy <= kept; copy++) {
		struct local_device_description device = { 0 };
		const char *reason =
			local_device_describe(context->copy_paths[copy],
					      &device) != STATUS_OK ?
				"device_disappeared_or_unknown" :
				local_device_media_refusal(&device, external);
		for (uint32_t previous = 0;
		     reason == NULL && previous < state->copy_count; previous++)
			if (local_device_overlaps(&device, &devices[previous]))
				reason = "duplicate_physical_device";
		local_device_print(stderr, &device,
				   reason != NULL ? reason : "eligible", 0,
				   LOCAL_DEVICE_DEFAULT_FLOOR_PERCENT, 0, 0);
		if (reason != NULL)
			continue;
		const uint32_t place = state->copy_count++;
		devices[place] = device;
		context->admitted[place] = context->copy_paths[copy];
		status = storage_device_key_for(context->copy_paths[copy],
						external, state->keys[place],
						&device);
		(void)snprintf(context->filesystems[place],
			       sizeof(context->filesystems[place]), "%s",
			       device.filesystem);
		if (status != STATUS_OK)
			(void)autotune_refused(
				"the copy's drive has no hardware identity (by-uuid, mount or sysfs)",
				context->copy_paths[copy]);
	}
	if (status == STATUS_OK && state->copy_count == 0) {
		(void)autotune_refused(
			"no admitted copy: serving's disk store refuses every copy's medium",
			source);
		status = STATUS_UNSUPPORTED;
	}
	if (status == STATUS_OK) {
		status = pack_open_entry(&storage_direct, context->reader,
					 &context->pack);
		if (status != STATUS_OK)
			(void)fprintf(
				stderr,
				"autotune: refused: the pack does not open (%s): %s\n",
				status_text(status), source);
	}
	if (status == STATUS_OK)
		status = engram_open(&storage_direct, context->reader,
				     context->pack, &context->engram);
	if (status == STATUS_UNSUPPORTED && context->pack != NULL)
		(void)autotune_refused("the pack has no n-gram tables", source);
	state->file_byte_count = header.file_byte_count;
	(void)snprintf(state->pack_identity, sizeof(state->pack_identity),
		       "%016" PRIx64, header.header_digest);
	return status;
}

/* The store over the admitted copies, opened as serving opens it with no RAM share and no cache; its scan timed.
 * Only after context_open succeeded, so with its n-gram tables open: no description is the caller's error. */
MUST_CHECK static result_status store_open(struct engram_context *context,
					   uint32_t copies, uint64_t deadline)
{
	const struct engram_description *const description =
		engram_description_of(context->engram);
	if (description == NULL)
		return STATUS_INVALID_ARGUMENT;
	result_status status = engram_disk_sources_open(context->admitted,
							copies, context->pack,
							context->engram,
							&context->sources);
	engram_disk_scan_threads(ENGRAM_DISK_SCAN_THREADS_DEFAULT);
	const uint64_t begin = autotune_now_nanoseconds();
	if (status == STATUS_OK)
		status = engram_disk_open_until(description, context->pack,
						&storage_direct,
						context->sources, 0, 0, 0, 0,
						deadline, &engram_stop,
						&context->disk);
	context->scan_ns = autotune_now_nanoseconds() - begin;
	for (uint32_t table = 0;
	     status == STATUS_OK && table < description->table_count; table++)
		context->scan_bytes +=
			engram_disk_block_count(context->disk, table) *
			PACK_BLOCK_BYTES;
	if (status == STATUS_OK)
		(void)fprintf(
			stderr,
			"storage: engram store opened over %" PRIu32
			" copies, no RAM share, no cache: the digest scan read %.2f GB in %.3f s\n",
			copies,
			(double)context->scan_bytes / BYTES_PER_GIGABYTE,
			(double)context->scan_ns /
				(double)NANOSECONDS_PER_SECOND);
	return status;
}

static void context_close(struct engram_context *context)
{
	engram_disk_close(context->disk);
	if (context->sources != NULL)
		storage_direct.reader_close(context->sources);
	engram_close(context->engram);
	pack_close(context->pack);
	if (context->reader != NULL)
		storage_direct.reader_close(context->reader);
}

static void point_line(const struct autotune_engram_point *point,
		       uint32_t index, result_status status)
{
	(void)fprintf(stderr,
		      "storage: engram point %" PRIu32 " (%s) threads %" PRIu32
		      " spin %" PRIu32 " us pin %" PRIu32 ": first table p50",
		      index, phase_names[point->phase],
		      point->arm.threads_per_copy, point->arm.spin_us,
		      point->arm.pin);
	for (uint32_t level = 0; level < AUTOTUNE_ENGRAM_LEVELS; level++)
		(void)fprintf(stderr, " %s %.1f", level_names[level],
			      (double)point->levels[level].first_p50_ns /
				      NANOSECONDS_PER_MICROSECOND);
	(void)fprintf(stderr, " us, %.2f cores%s%s status=%s\n",
		      point->wall_ns != 0 ?
			      (double)point->cpu_ns / (double)point->wall_ns :
			      0.0,
		      point->disturbance != AUTOTUNE_ENGRAM_QUIET ?
			      ", disturbed: " :
			      "",
		      disturbance_names[point->disturbance],
		      status_text(status));
}

/* The points still to measure; STATUS_TIMED_OUT once the lease, the run's points, a stop signal or a hot drive ends
 * first, the progress kept. */
MUST_CHECK static result_status
points_run(struct engram_context *context,
	   const struct autotune_engram_trace *trace,
	   struct autotune_engram_state *state, const char *state_path,
	   uint64_t deadline, uint32_t points_per_run)
{
	bool warmed[AUTOTUNE_ENGRAM_WARMUP_SIDES] = { false };
	uint32_t measured = 0, arm_index = 0, phase = 0;
	const char *refusal = NULL;
	result_status status = STATUS_OK;
	while (status == STATUS_OK &&
	       autotune_engram_next(state, &arm_index, &phase, &refusal)) {
		/* a checking build's fault point: the lease spent before this point */
		if (engram_stop != 0 ||
		    autotune_now_nanoseconds() >= deadline ||
		    (points_per_run != 0 && measured == points_per_run) ||
		    FAULT_POINT(engram_tune_lease))
			return STATUS_TIMED_OUT;
		struct autotune_drives drives;
		autotune_drives_read(&drives);
		for (uint32_t drive = 0; drive < drives.count; drive++)
			if (drives.millicelsius[drive] >
			    ENGRAM_TUNE_DRIVE_MILLICELSIUS_MOST) {
				(void)fprintf(
					stderr,
					"storage: engram stage paused: %s at %.1f C, over %.1f C; "
					"the same command resumes\n",
					drives.names[drive],
					(double)drives.millicelsius[drive] /
						MILLIDEGREES_PER_DEGREE,
					(double)ENGRAM_TUNE_DRIVE_MILLICELSIUS_MOST /
						MILLIDEGREES_PER_DEGREE);
				return STATUS_TIMED_OUT;
			}
		const uint32_t side = phase == AUTOTUNE_ENGRAM_ADDA ? 1u : 0u;
		if (!warmed[side]) {
			struct autotune_engram_point *const warmup =
				&state->warmups[side];
			memset(warmup, 0, sizeof(*warmup));
			autotune_engram_arm_of(default_arm(), &warmup->arm);
			warmup->arm_index = default_arm();
			status = autotune_engram_point_measure(
				context->disk, trace,
				AUTOTUNE_ENGRAM_POINTS_MOST + side, false,
				warmup);
			point_line(warmup, 0, status);
			state->warmup_count++;
			warmed[side] = true;
			if (status != STATUS_OK)
				break;
		}
		struct autotune_engram_point *const point =
			&state->points[state->point_count];
		memset(point, 0, sizeof(*point));
		autotune_engram_arm_of(arm_index, &point->arm);
		point->arm_index = arm_index;
		point->phase = phase;
		status = autotune_engram_point_measure(
			context->disk, trace, state->point_count, true, point);
		point_line(point, state->point_count + 1u, status);
		if (status == STATUS_OK) {
			state->point_count++;
			status = state_write(state_path, state);
			measured++;
		}
	}
	/* the process's first flight did not open: this host's io_uring refuses step I/O */
	if (status == STATUS_UNSUPPORTED && measured == 0)
		(void)autotune_refused(
			"step I/O cannot start (io_uring refused: serving keeps per-row gathers)",
			context->admitted[0]);
	if (status == STATUS_OK && refusal != NULL) {
		(void)autotune_refused(refusal, state_path);
		status = STATUS_UNSUPPORTED;
	}
	return status;
}

static void arm_json(FILE *output, const char *name,
		     const struct autotune_engram_arm *arm)
{
	(void)fprintf(output,
		      "\"%s\":{\"threads_per_copy\":%" PRIu32
		      ",\"spin_us\":%" PRIu32 ",\"pin\":%" PRIu32 "}",
		      name, arm->threads_per_copy, arm->spin_us, arm->pin);
}

static void point_json(FILE *output, const struct autotune_engram_point *point,
		       uint32_t copies)
{
	(void)fprintf(output,
		      "{\"phase\":\"%s\",\"threads_per_copy\":%" PRIu32
		      ",\"spin_us\":%" PRIu32 ",\"pin\":%" PRIu32
		      ",\"rotation\":%" PRIu32 ",\"levels\":[",
		      phase_names[point->phase], point->arm.threads_per_copy,
		      point->arm.spin_us, point->arm.pin, point->rotation);
	for (uint32_t level = 0; level < AUTOTUNE_ENGRAM_LEVELS; level++) {
		const struct autotune_engram_level *const record =
			&point->levels[level];
		(void)fprintf(
			output,
			"%s{\"name\":\"%s\",\"steps\":%" PRIu64
			",\"first_p50_ns\":%" PRIu64
			",\"first_p99_ns\":%" PRIu64
			",\"first_p999_ns\":%" PRIu64 ",\"all_p50_ns\":%" PRIu64
			",\"all_p99_ns\":%" PRIu64 ",\"rows_per_second\":%.1f}",
			level == 0 ? "" : ",", level_names[level],
			record->steps, record->first_p50_ns,
			record->first_p99_ns, record->first_p999_ns,
			record->all_p50_ns, record->all_p99_ns,
			record->wall_ns != 0 ?
				(double)record->rows *
					(double)NANOSECONDS_PER_SECOND /
					(double)record->wall_ns :
				0.0);
	}
	(void)fprintf(output,
		      "],\"cpu_ns\":%" PRIu64 ",\"wall_ns\":%" PRIu64
		      ",\"copy_reads\":[",
		      point->cpu_ns, point->wall_ns);
	for (uint32_t copy = 0; copy < copies; copy++)
		(void)fprintf(output, "%s%" PRIu64, copy == 0 ? "" : ",",
			      point->copy_reads[copy]);
	(void)fprintf(output,
		      "],\"answered\":%" PRIu64 ",\"failed_reads\":%" PRIu64
		      ",\"retirements\":%" PRIu32 ",\"plain_rings\":%" PRIu32
		      ",\"disturbed\":\"%s\"",
		      point->answered, point->failed_reads, point->retirements,
		      point->plain_rings,
		      disturbance_names[point->disturbance]);
	autotune_drives_json(output, "millicelsius_before", &point->before);
	autotune_drives_json(output, "millicelsius_after", &point->after);
	(void)fputc('}', output);
}

/* The profile's "engram" member: a candidate, enabled 0, with its identity, its shape, the trace's counts, the store's
 * scan, every point and the selection. */
static void section_json(FILE *output, const struct engram_context *context,
			 const struct autotune_engram_trace *trace,
			 const struct autotune_engram_state *state,
			 const struct autotune_engram_selection *selection,
			 const struct autotune_engram_arm *winner,
			 const struct autotune_engram_arm *arms)
{
	const struct autotune_engram_verdict *const verdict =
		&selection->verdict;
	(void)fprintf(
		output,
		"{\"schema\":%u,\"enabled\":0,\"qualification\":\"candidate\","
		"\"created_unix\":%lld,\"pack\":{\"header_digest\":\"%s\","
		"\"file_byte_count\":%" PRIu64 "},\"copies\":[",
		STORAGE_ENGRAM_SECTION_SCHEMA, (long long)time(NULL),
		state->pack_identity, state->file_byte_count);
	for (uint32_t copy = 0; copy < state->copy_count; copy++) {
		(void)fprintf(output, "%s{\"key\":\"%s\",\"path\":",
			      copy == 0 ? "" : ",", state->keys[copy]);
		prepare_json_text(output, context->admitted[copy]);
		(void)fputs(",\"filesystem\":", output);
		prepare_json_text(output, context->filesystems[copy]);
		(void)fputc('}', output);
	}
	(void)fprintf(output,
		      "],\"shape\":{\"interface\":\"" STORAGE_ENGRAM_INTERFACE
		      "\",\"steps\":%" PRIu32 ",\"slots\":%" PRIu32
		      ",\"read_limit_ns\":%" PRIu64
		      ",\"ram_percent\":0,\"cache_bytes\":0,\"decode_rows\":%u,"
		      "\"prefill_rows\":%u,\"levels\":[",
		      state->steps, state->slots,
		      engram_steps_read_limit_within(
			      CARD_WAIT_LIMIT_NANOSECONDS, state->copy_count),
		      ENGRAM_TUNE_DECODE_ROWS, ENGRAM_TUNE_PREFILL_ROWS);
	for (uint32_t level = 0; level < AUTOTUNE_ENGRAM_LEVELS; level++)
		(void)fprintf(output, "%s\"%s\"", level == 0 ? "" : ",",
			      level_names[level]);
	(void)fprintf(output,
		      "]},\"trace\":{\"identity\":\"%s\",\"cases\":%" PRIu32
		      ",\"tokens\":%" PRIu32 ",\"rows\":%" PRIu64
		      ",\"unique_blocks\":%" PRIu64 "},\"step_io\":{",
		      trace->identity, trace->cases, trace->tokens,
		      (uint64_t)trace->tokens * trace->references_per_token,
		      trace->unique_blocks);
	(void)fprintf(output,
		      "\"threads_per_copy\":%" PRIu32 ",\"spin_us\":%" PRIu32
		      ",\"pin\":%" PRIu32 "},\"scan\":{\"threads_per_copy\":%u,"
		      "\"bytes\":%" PRIu64 ",\"nanoseconds\":%" PRIu64
		      "},\"warmups\":%" PRIu32 ",\"measurements\":[",
		      winner->threads_per_copy, winner->spin_us, winner->pin,
		      ENGRAM_DISK_SCAN_THREADS_DEFAULT, context->scan_bytes,
		      context->scan_ns, state->warmup_count);
	for (uint32_t side = 0; side < AUTOTUNE_ENGRAM_WARMUP_SIDES; side++) {
		point_json(output, &state->warmups[side], state->copy_count);
		(void)fputc(',', output);
	}
	for (uint32_t index = 0; index < state->point_count; index++) {
		(void)fputs(index == 0 ? "" : ",", output);
		point_json(output, &state->points[index], state->copy_count);
	}
	(void)fputs("],\"selection\":{", output);
	arm_json(output, "winner", winner);
	(void)fputc(',', output);
	arm_json(output, "challenger", &arms[0]);
	(void)fputc(',', output);
	arm_json(output, "default", &arms[1]);
	/* the points that hold the A D D A's slots, by their rotation (a point's index), and their costs */
	(void)fputs(",\"adda_points\":[", output);
	for (uint32_t slot = 0; slot < selection->held; slot++)
		(void)fprintf(output, "%s%" PRIu32, slot == 0 ? "" : ",",
			      selection->slots[slot]);
	(void)fputs("],\"adda_cost_ns\":[", output);
	for (uint32_t slot = 0; slot < selection->held; slot++)
		(void)fprintf(
			output, "%s%.1f", slot == 0 ? "" : ",",
			point_cost(&state->points[selection->slots[slot]]));
	(void)fputs("],\"per_level\":[", output);
	for (uint32_t level = 0; level < AUTOTUNE_ENGRAM_LEVELS; level++)
		(void)fprintf(
			output,
			"%s{\"name\":\"%s\",\"spread\":%.4f,\"loss\":%.4f}",
			level == 0 ? "" : ",", level_names[level],
			verdict->level_spread[level],
			verdict->level_loss[level]);
	(void)fprintf(
		output,
		"],\"challenger_cost_ns\":%.1f,\"default_cost_ns\":%.1f,\"margin\":%.4f,"
		"\"spread\":%.4f,\"spread_bound\":%.2f,\"spread_over_bound\":%u,\"exact_tie\":%u,"
		"\"default_holds\":%u,\"adda_disturbed_twice\":%u,\"blocked_by\":\"%s\",\"excluded\":[",
		verdict->challenger_cost, verdict->default_cost,
		verdict->margin, verdict->spread, AUTOTUNE_ENGRAM_SPREAD_BOUND,
		verdict->spread_over_bound ? 1u : 0u,
		verdict->exact_tie ? 1u : 0u, verdict->default_holds ? 1u : 0u,
		selection->adda_disturbed_twice ? 1u : 0u,
		verdict->blocked_by < AUTOTUNE_ENGRAM_LEVELS ?
			level_names[verdict->blocked_by] :
			"");
	bool first = true;
	for (uint32_t arm = 0; arm < AUTOTUNE_ENGRAM_GRID_POINTS; arm++) {
		if (arm_quiet(state, arm, selection->measured) != NULL)
			continue;
		struct autotune_engram_arm excluded;
		autotune_engram_arm_of(arm, &excluded);
		(void)fprintf(output,
			      "%s{\"threads_per_copy\":%" PRIu32
			      ",\"spin_us\":%" PRIu32 ",\"pin\":%" PRIu32 "}",
			      first ? "" : ",", excluded.threads_per_copy,
			      excluded.spin_us, excluded.pin);
		first = false;
	}
	(void)fputs("]}}", output);
}

/* The section from the finished points, into the profile; the progress file goes once the profile holds it. */
MUST_CHECK static result_status
section_publish(const struct engram_context *context,
		const struct autotune_engram_trace *trace,
		const struct autotune_engram_state *state, const char *profile,
		const char *real_source, const char *state_path)
{
	struct autotune_engram_selection selection;
	const char *refusal = NULL;
	if (!autotune_engram_select(state, &selection, &refusal))
		return STATUS_INVALID_ARGUMENT;
	const struct autotune_engram_verdict verdict = selection.verdict;
	struct autotune_engram_arm arms[ABBA_ARMS];
	autotune_engram_arm_of(selection.challenger, &arms[0]);
	autotune_engram_arm_of(default_arm(), &arms[1]);
	const struct autotune_engram_arm *const winner =
		&arms[verdict.default_holds ? 1u : 0u];
	char *text = NULL;
	size_t length = 0;
	FILE *const stream = open_memstream(&text, &length);
	if (stream == NULL)
		return STATUS_OUT_OF_MEMORY;
	section_json(stream, context, trace, state, &selection, winner, arms);
	if (fclose(stream) != 0) {
		free(text);
		return STATUS_OUT_OF_MEMORY;
	}
	const result_status status = storage_stage_section_write(
		profile, real_source, STORAGE_ENGRAM_SECTION_NAME, text, length,
		&refusal);
	free(text);
	if (refusal != NULL)
		(void)autotune_refused(refusal, profile);
	if (status != STATUS_OK)
		return status;
	(void)unlink(state_path);
	char flag[ENGRAM_TUNE_FLAG_TEXT_BYTES] = "";
	if (selection.adda_disturbed_twice)
		(void)snprintf(
			flag, sizeof(flag),
			"; the default holds: A D D A slot %" PRIu32
			" (the %s's) was disturbed twice (adda_disturbed_twice)",
			selection.held + 1u,
			abba_is_default[selection.held] != 0 ? "default" :
							       "challenger");
	else if (verdict.blocked_by < AUTOTUNE_ENGRAM_LEVELS)
		(void)snprintf(flag, sizeof(flag),
			       "; the challenger is blocked by its loss at %s",
			       level_names[verdict.blocked_by]);
	else if (verdict.spread_over_bound)
		(void)snprintf(
			flag, sizeof(flag),
			"; flagged: the spread is over the %.0f %% bound, so look at the points before --qualify",
			AUTOTUNE_ENGRAM_SPREAD_BOUND *
				(double)ENGRAM_TUNE_PERCENT);
	(void)fprintf(
		stderr,
		"storage: engram stage selects threads %" PRIu32
		" spin %" PRIu32 " us pin %" PRIu32
		"%s (A D D A cost %.1f us against the default's %.1f us, margin %+.2f %%, spread %.2f %%%s%s); a "
		"candidate until --qualify\n",
		winner->threads_per_copy, winner->spin_us, winner->pin,
		verdict.default_holds ? ", the default" : "",
		verdict.challenger_cost / NANOSECONDS_PER_MICROSECOND,
		verdict.default_cost / NANOSECONDS_PER_MICROSECOND,
		verdict.margin * (double)ENGRAM_TUNE_PERCENT,
		verdict.spread * (double)ENGRAM_TUNE_PERCENT,
		verdict.exact_tie ? ", an exact tie" : "", flag);
	return STATUS_OK;
}

/* --qualify: the candidate section, once it names this pack, these copies and serving's shape, becomes enabled. */
MUST_CHECK static result_status
section_qualify(const struct engram_context *context,
		const struct autotune_engram_state *state, const char *profile,
		const char *real_source)
{
	struct storage_engram_section section;
	char reason[STORAGE_PROFILE_REASON_BYTES];
	if (!storage_engram_section_read(profile, &section, reason) ||
	    !storage_engram_section_matches(
		    &section, pack_header_of(context->pack),
		    (const char (*)[STORAGE_PROFILE_KEY_BYTES])state->keys,
		    state->copy_count, state->steps, state->slots, reason)) {
		(void)autotune_refused(reason, profile);
		return STATUS_INVALID_ARGUMENT;
	}
	const char *refusal = NULL;
	const result_status status = storage_stage_section_qualify(
		profile, real_source, STORAGE_ENGRAM_SECTION_NAME, &refusal);
	if (refusal != NULL)
		(void)autotune_refused(refusal, profile);
	return status;
}

bool storage_autotune_engram_current(const char *source, const char *profile,
				     bool external,
				     char reason[STORAGE_PROFILE_REASON_BYTES])
{
	struct storage_engram_section section;
	if (!storage_engram_section_read(profile, &section, reason))
		return false;
	(void)snprintf(reason, STORAGE_PROFILE_REASON_BYTES, "%s",
		       AUTOTUNE_COPIES_UNOPENED);
	struct engram_context *const context = calloc(1, sizeof(*context));
	struct autotune_engram_state *const state = calloc(1, sizeof(*state));
	const bool current =
		context != NULL && state != NULL &&
		context_open(context, source, external, state) == STATUS_OK &&
		storage_engram_section_matches(
			&section, pack_header_of(context->pack),
			(const char (*)[STORAGE_PROFILE_KEY_BYTES])state->keys,
			state->copy_count, 1u + STEP_PIPELINE_MOST,
			ENGRAM_STEP_SLOTS, reason);
	if (context != NULL)
		context_close(context);
	free(context);
	free(state);
	return current;
}

static const char engram_usage[] =
	"usage: anemone autotune PACK --stage=engram --tokens=TOKEN_IDS_U32 [--tokens=... (at most 16)] "
	"[--profile=ABSOLUTE] [--seconds=1..1650 (1650)] [--points_per_run=N (0: all)] [--replica_external=0|1] "
	"[--plan] [--qualify]\n"
	"The Engram stage tunes serving's step I/O (threads a copy, spin, pin) over the copies serving's disk store "
	"reads, replaying real prompts (token ids, a case a file), and writes the profile's \"engram\" section as a "
	"candidate; --qualify enables it. Repeat the same command after exit75 to resume.\n";

/* Refused by name: the copy stage's options and the load stage's. */
static const char *const foreign_options[] = {
	"--target=",	"--copy_workers=", "--copy_hash_pairs=",
	"--manifest=",	"--verify=",	   "--replica_floor_percent=",
	"--window_gib="
};
#define FOREIGN_OPTIONS (sizeof(foreign_options) / sizeof(foreign_options[0]))

int storage_autotune_engram_cli(int count, char **arguments)
{
	const char *source = NULL, *profile = NULL;
	const char *tokens[AUTOTUNE_ENGRAM_TRACES_MOST];
	uint32_t seconds = AUTOTUNE_ENGRAM_SECONDS_MOST, points_per_run = 0,
		 external = 0, token_files = 0;
	bool valid = true, help = false, plan_only = false, qualify = false;
	for (int index = AUTOTUNE_FIRST_OPTION; valid && index < count;
	     index++) {
		const char *const argument = arguments[index];
		bool foreign = false;
		for (size_t option = 0; option < FOREIGN_OPTIONS; option++)
			foreign = foreign ||
				  strncmp(argument, foreign_options[option],
					  strlen(foreign_options[option])) == 0;
		if (strcmp(argument, AUTOTUNE_STAGE_OPTION "engram") == 0)
			continue;
		if (strcmp(argument, "--help") == 0) {
			help = true;
			break;
		}
		if (strncmp(argument, "--prompt=", strlen("--prompt=")) == 0)
			valid = autotune_refused(AUTOTUNE_PROMPT_RETIRED,
						 argument);
		else if (foreign)
			valid = autotune_refused(
				"the copy stage's or the load stage's option, not the Engram stage's",
				argument);
		else if (strcmp(argument, "--plan") == 0)
			plan_only = true;
		else if (strcmp(argument, "--qualify") == 0)
			qualify = true;
		else if (strncmp(argument,
				 "--profile=", strlen("--profile=")) == 0)
			profile = argument + strlen("--profile=");
		else if (strncmp(argument, "--tokens=", strlen("--tokens=")) ==
				 0 &&
			 token_files < AUTOTUNE_ENGRAM_TRACES_MOST) {
			tokens[token_files] = argument + strlen("--tokens=");
			token_files++;
		} else if (strncmp(argument,
				   "--tokens=", strlen("--tokens=")) == 0)
			valid = autotune_refused("at most 16 --tokens files",
						 argument);
		else if (strncmp(argument,
				 "--seconds=", strlen("--seconds=")) == 0)
			valid = (autotune_number_option(
					 argument + strlen("--seconds="),
					 AUTOTUNE_ENGRAM_SECONDS_MOST,
					 &seconds) &&
				 seconds != 0) ||
				autotune_refused("--seconds takes 1 to 1650",
						 argument);
		else if (strncmp(argument, "--points_per_run=",
				 strlen("--points_per_run=")) == 0)
			valid = autotune_number_option(
					argument + strlen("--points_per_run="),
					AUTOTUNE_ENGRAM_POINTS_MOST,
					&points_per_run) ||
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
				"not an option of the Engram stage", argument);
	}
	if (valid && !help && token_files == 0 && !qualify)
		valid = autotune_refused(
			"the Engram stage replays real prompts: give --tokens=TOKEN_IDS_U32",
			"--tokens");
	char default_profile[PATH_MAX], real_source[PATH_MAX],
		state_path[PATH_MAX], folder[PATH_MAX];
	if (valid && !help && profile == NULL) {
		valid = autotune_profile_default(default_profile,
						 sizeof(default_profile));
		profile = default_profile;
	}
	if (valid && !help && source == NULL)
		valid = autotune_refused("no pack given", "PACK");
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
	valid = valid &&
		(strcmp(profile, real_source) != 0 ||
		 autotune_refused("the profile path is the pack itself",
				  profile));
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
		(void)fputs(engram_usage, stderr);
		return help ? EXIT_SUCCESS : EXIT_FAILURE;
	}
	struct engram_context *const context = calloc(1, sizeof(*context));
	struct autotune_engram_state *const state = calloc(1, sizeof(*state));
	struct autotune_engram_trace trace = { 0 };
	result_status status = context != NULL && state != NULL ?
				       STATUS_OK :
				       STATUS_OUT_OF_MEMORY;
	/* before any copy is described: a profile the publish would refuse (the combined tuning's schema 2, another
	 * source's) */
	const char *refusal = NULL;
	if (status == STATUS_OK &&
	    storage_stage_section_writable(profile, real_source, &refusal) !=
		    STATUS_OK) {
		(void)autotune_refused(refusal != NULL ?
					       refusal :
					       "the profile cannot be checked",
				       profile);
		status = STATUS_INVALID_ARGUMENT;
		if (plan_only)
			(void)puts("{\"valid\":false,\"stage\":\"engram\"}");
	}
	if (status == STATUS_OK) {
		state->schema = ENGRAM_TUNE_STATE_SCHEMA;
		state->steps = 1u + STEP_PIPELINE_MOST;
		state->slots = ENGRAM_STEP_SLOTS;
		status = context_open(context, real_source, external != 0,
				      state);
	}
	if (status == STATUS_OK && !qualify) {
		uint32_t failed = 0;
		status = autotune_engram_trace_open(
			tokens, token_files, context->engram, &trace, &failed);
		if (status != STATUS_OK)
			(void)autotune_refused(
				failed < token_files ?
					"a --tokens file is no non-empty regular file of whole uint32 token ids of "
					"this pack's vocabulary, within 1 MiB together" :
					"the --tokens files do not fit in memory",
				failed < token_files ? tokens[failed] :
						       "--tokens");
		(void)snprintf(state->trace_identity,
			       sizeof(state->trace_identity), "%s",
			       trace.identity);
		char seed_digits[ENGRAM_TUNE_SEED_DIGITS + 1u];
		(void)snprintf(seed_digits, sizeof(seed_digits), "%.*s",
			       (int)ENGRAM_TUNE_SEED_DIGITS, trace.identity);
		state->seed = strtoull(seed_digits, NULL, ENGRAM_TUNE_HEX_BASE);
	}
	if (status == STATUS_OK && !qualify)
		autotune_engram_state_resume(state_path, state);
	const uint64_t deadline = autotune_now_nanoseconds() +
				  (uint64_t)seconds * NANOSECONDS_PER_SECOND;
	if (status == STATUS_OK && plan_only) {
		(void)printf(
			"{\"valid\":true,\"stage\":\"engram\",\"copies\":%" PRIu32
			",\"cases\":%" PRIu32 ",\"tokens\":%" PRIu32
			",\"rows\":%" PRIu64 ",\"unique_blocks\":%" PRIu64
			",\"levels\":%u,\"grid_points\":%u,\"adda_points\":%u,\"points_done\":%" PRIu32
			"}\n",
			state->copy_count, trace.cases, trace.tokens,
			(uint64_t)trace.tokens * trace.references_per_token,
			trace.unique_blocks, AUTOTUNE_ENGRAM_LEVELS,
			AUTOTUNE_ENGRAM_GRID_POINTS,
			AUTOTUNE_ENGRAM_ABBA_POINTS, state->point_count);
	} else if (status == STATUS_OK && qualify) {
		status = section_qualify(context, state, profile, real_source);
	} else if (status == STATUS_OK) {
		engram_stop = 0;
		struct sigaction action = { .sa_handler = engram_signal },
				 old_term, old_int;
		(void)sigemptyset(&action.sa_mask);
		const bool term_set = sigaction(SIGTERM, &action, &old_term) ==
				      0;
		const bool int_set = sigaction(SIGINT, &action, &old_int) == 0;
		status = term_set && int_set ?
				 store_open(context, state->copy_count,
					    deadline) :
				 STATUS_UNSUPPORTED;
		if (status == STATUS_OK)
			status = points_run(context, &trace, state, state_path,
					    deadline, points_per_run);
		if (term_set)
			(void)sigaction(SIGTERM, &old_term, NULL);
		if (int_set)
			(void)sigaction(SIGINT, &old_int, NULL);
		if (status == STATUS_OK)
			status = section_publish(context, &trace, state,
						 profile, real_source,
						 state_path);
	}
	autotune_engram_trace_close(&trace);
	if (context != NULL)
		context_close(context);
	free(context);
	free(state);
	(void)fprintf(stderr,
		      "storage: autotune stage=engram status=%s resumable=%u\n",
		      status_text(status),
		      status == STATUS_TIMED_OUT ? 1u : 0u);
	return status == STATUS_TIMED_OUT ? AUTOTUNE_RETRY_EXIT :
	       status == STATUS_OK	  ? EXIT_SUCCESS :
					    EXIT_FAILURE;
}
