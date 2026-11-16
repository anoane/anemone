/* SPDX-License-Identifier: AGPL-3.0-or-later */
#include "storage_autotune.h"
#include "anemone/engram_disk.h"
#include "anemone/loader.h"
#include "engram_step.h"
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#define TUNE_NS_SECOND 1000000000ull
#define TUNE_POINTS_MOST 128u
#define TUNE_SCHEMA \
	5u /* 4: copy sets, the uncached lookup, rows and wall time per stream level; 5: load points steal */
#define TUNE_STREAM_LEVELS 4u
#define TUNE_STREAMS_MOST 8u
#define TUNE_STEPS 64u
#define TUNE_PREFILL_CHUNK 1024u
#define TUNE_TOKEN_LIMIT 262144u
#define TUNE_TABLES 2u
#define TUNE_ROWS_PER_TABLE 24u
/* the products' values (asserted below), so that no unsigned int product is widened where they are used */
#define TUNE_TOTAL_ROWS 48u
#define TUNE_TOKEN_BYTES 12672u
#define TUNE_HASH_BYTES STORAGE_PROFILE_KEY_BYTES
#define TUNE_CACHE_BYTES (256ull << 20u)
#define TUNE_MEDIAN_DIVISOR 2u
#define TUNE_REUSE_BINS 32u
#define TUNE_TABLE_SHIFT 32u
#define TUNE_MIN_CACHE_MIB 64u
#define TUNE_MAX_CACHE_MIB 512u
#define TUNE_MIB_BYTES (1ull << 20u)
#define TUNE_CACHE_PERCENTILE 90u
#define TUNE_HASH_MULTIPLIER 11400714819323198485ull
#define TUNE_LOAD_CHUNKS \
	512u /* 8 GiB of 16 MiB chunks: every load point reads this volume (less of a smaller pack) */
#define TUNE_LOAD_THREADS_MOST 4u
#define TUNE_LOAD_WORKERS_MOST (TUNE_LOAD_THREADS_MOST * STORAGE_MAXIMUM_COPIES)
#define TUNE_QUANTILE_PERCENT 100u
#define TUNE_P99_PERCENT 99u
#define TUNE_WEIGHT_SCALE 100u
#define TUNE_WEIGHT_VARIANTS \
	3u /* equal, speed-proportional and speed-squared shares of a copy set */
#define TUNE_THREAD_OPTIONS 3u
#define TUNE_PROFILE_SCHEMA 2u
/* The Engram stage's step I/O selection: measured in serving, not by this tuner, so the profile names its evidence */
#define STEP_IO_EVIDENCE                                                                                   \
	"phase 4(b), request 7 revision 2, PP5 serving ABBA with every stream identical to all-RAM: cold " \
	"TTFT 27.38 s at 131088 tokens against 27.18 s all-RAM and 47.85 s per-row; decode C1 172.2 "      \
	"against 176.0 tok/s; 0.78 cores. Requests 9 and 9b revision 2, TP4_PRO, the same: decode C1 "     \
	"4.6-5.9 % over per-row; cold TTFT 35.18 s at 32782 tokens against 35.08 s all-RAM and 37.98 s "   \
	"per-row"
#define TUNE_SLOT_FACTOR \
	2u /* the reuse table: a power of two at least this many slots a reference */
#define TUNE_MANIFEST_MOST_BYTES (4u << 20u)
#define TUNE_IO_MECHANISMS \
	2u /* a lookup point's io: 0 pread, 1 uring_fixed_direct */
#define TUNE_SET_COPIES_MOST \
	3u /* every set of two or more copies up to this many copies; beyond, the full set only (point table) */
#define TUNE_CACHE_MODES \
	2u /* a lookup point's bounded cache: 1 cached (production), 0 uncached (raw devices) */
#define TUNE_SET_COPIES_LEAST 2u /* a copy set has two copies or more */
#define TUNE_SQUARED_VARIANT \
	2u /* the weight variant that squares the speed shares */
_Static_assert(TUNE_TOTAL_ROWS == TUNE_TABLES * TUNE_ROWS_PER_TABLE &&
		       TUNE_TOKEN_BYTES == TUNE_TOTAL_ROWS * ENGRAM_ROW_BYTES,
	       "the tuner's rows and bytes a token are its tables' products");
struct tune_trace {
	uint32_t count, prompt;
	uint64_t *rows;
	char *reference;
};
struct tune_point {
	uint32_t action, io, threads, cached, weights[STORAGE_MAXIMUM_COPIES];
	/* load: the chunks each copy's readers took */
	uint32_t copy_chunks[STORAGE_MAXIMUM_COPIES];
	result_status status;
	uint64_t ns, bytes, p50[TUNE_STREAM_LEVELS], p99[TUNE_STREAM_LEVELS],
		prefill[TUNE_STREAM_LEVELS],
		rows[TUNE_STREAM_LEVELS], /* lookup: rows gathered and the */
		wall[TUNE_STREAM_LEVELS]; /* wall time of every stream group */
	bool page_locked;
};
struct tune_state {
	uint32_t schema, count, points;
	char keys[STORAGE_MAXIMUM_COPIES][TUNE_HASH_BYTES],
		trace[TUNE_HASH_BYTES];
	uint64_t created;
	struct tune_point values[TUNE_POINTS_MOST];
};
struct tune_context {
	const struct prepare_tune_options *options;
	const struct storage_reader *reader;
	struct engram_record *engram;
	struct engram_disk *disk;
	struct tune_trace traces[PREPARE_TRACE_CASES];
	uint32_t trace_count, copies, cache_mib;
	uint64_t pack_bytes, row_references, unique_rows, cold_blocks,
		reuse[TUNE_REUSE_BINS];
	uint32_t device_indices[STORAGE_MAXIMUM_COPIES];
};
struct load_worker {
	struct tune_context *context;
	atomic_uint *cursor;
	void *buffer;
	uint32_t copy, chunks, taken;
	result_status status;
};
struct lookup_worker {
	struct tune_context *context;
	const struct tune_trace *trace;
	uint64_t steps[TUNE_STEPS], prefill, rows;
	uint32_t step_count;
	result_status status;
};
static const uint32_t stream_levels[TUNE_STREAM_LEVELS] = { 1, 2, 4, 8 };
static const uint32_t thread_options[TUNE_THREAD_OPTIONS] = { 1, 2, 4 };
static uint64_t tune_now(void)
{
	struct timespec time;
	return clock_gettime(CLOCK_MONOTONIC, &time) == 0 ?
		       (uint64_t)time.tv_sec * TUNE_NS_SECOND +
			       (uint64_t)time.tv_nsec :
		       UINT64_MAX;
}
static bool tune_stopped(const struct tune_context *context)
{
	return tune_now() >= context->options->deadline_ns ||
	       (context->options->stop != NULL && *context->options->stop != 0);
}
/* A load point's reader: bound to one copy, it takes the next 16 MiB chunk from the point's one cursor until none is
 * left (work stealing, as the loader's readers do), so a faster copy reads more. */
static void *load_reads(void *argument)
{
	struct load_worker *const worker = argument;
	worker->status = STATUS_OK;
	for (uint32_t chunk = atomic_fetch_add_explicit(worker->cursor, 1u,
							memory_order_relaxed);
	     chunk < worker->chunks && worker->status == STATUS_OK;
	     chunk = atomic_fetch_add_explicit(worker->cursor, 1u,
					       memory_order_relaxed)) {
		if (tune_stopped(worker->context)) {
			worker->status = STATUS_TIMED_OUT;
			break;
		}
		worker->status = storage_direct.reader_read(
			worker->context->reader, worker->copy,
			(uint64_t)chunk * LOADER_CHUNK_BYTES, worker->buffer,
			LOADER_CHUNK_BYTES);
		worker->taken++;
	}
	return NULL;
}
/* A stealing load point's share of one copy: the busiest member at TUNE_WEIGHT_SCALE, every member at least 1, 0 for a
 * copy outside the point. The static split --load_tuned_weights=1 would apply. */
static uint32_t load_share(const struct tune_point *point, uint32_t copy)
{
	uint32_t most = 0;
	for (uint32_t index = 0; index < STORAGE_MAXIMUM_COPIES; index++)
		if (point->copy_chunks[index] > most)
			most = point->copy_chunks[index];
	if (point->weights[copy] == 0)
		return 0;
	const uint64_t share = most != 0 ? (uint64_t)point->copy_chunks[copy] *
						   TUNE_WEIGHT_SCALE / most :
					   0;
	return share != 0 ? (uint32_t)share : 1u;
}
MUST_CHECK static result_status load_point(struct tune_context *context,
					   struct tune_point *point)
{
	struct load_worker workers[TUNE_LOAD_WORKERS_MOST];
	pthread_t threads[TUNE_LOAD_WORKERS_MOST];
	memset(workers, 0, sizeof(workers));
	atomic_uint cursor = 0;
	uint32_t launched = 0, worker_count = 0, members = 0;
	for (uint32_t copy = 0; copy < context->copies; copy++)
		members += point->weights[copy] != 0 ? 1u : 0u;
	const uint64_t whole = context->pack_bytes / LOADER_CHUNK_BYTES;
	const uint32_t chunks = whole < TUNE_LOAD_CHUNKS ? (uint32_t)whole :
							   TUNE_LOAD_CHUNKS;
	result_status status = members != 0 && chunks != 0 &&
					       members * point->threads <=
						       TUNE_LOAD_WORKERS_MOST ?
				       STATUS_OK :
				       STATUS_INVALID_ARGUMENT;
	for (uint32_t copy = 0; status == STATUS_OK && copy < context->copies;
	     copy++)
		for (uint32_t thread = 0;
		     point->weights[copy] != 0 && thread < point->threads;
		     thread++)
			workers[worker_count++].copy = copy;
	for (uint32_t index = 0; index < worker_count; index++) {
		workers[index].buffer =
			aligned_alloc(PACK_BLOCK_BYTES, LOADER_CHUNK_BYTES);
		if (workers[index].buffer == NULL ||
		    mlock(workers[index].buffer, LOADER_CHUNK_BYTES) != 0) {
			status = STATUS_UNSUPPORTED;
			break;
		}
	}
	const uint64_t started = tune_now();
	for (uint32_t index = 0; status == STATUS_OK && index < worker_count;
	     index++) {
		workers[index].context = context;
		workers[index].cursor = &cursor;
		workers[index].chunks = chunks;
		if (pthread_create(&threads[index], NULL, load_reads,
				   &workers[index]) != 0)
			status = STATUS_OUT_OF_MEMORY;
		else
			launched++;
	}
	for (uint32_t index = 0; index < launched; index++) {
		if (pthread_join(threads[index], NULL) != 0)
			status = STATUS_INPUT_OUTPUT_ERROR;
		if (workers[index].status != STATUS_OK)
			status = workers[index].status;
		point->copy_chunks[workers[index].copy] += workers[index].taken;
	}
	point->ns = tune_now() - started;
	point->bytes = (uint64_t)chunks * LOADER_CHUNK_BYTES;
	point->page_locked = launched == worker_count;
	for (uint32_t index = 0; index < worker_count; index++)
		if (workers[index].buffer != NULL) {
			(void)munlock(workers[index].buffer,
				      LOADER_CHUNK_BYTES);
			free(workers[index].buffer);
		}
	return status;
}
MUST_CHECK static result_status gather_chunk(struct tune_context *context,
					     const struct tune_trace *trace,
					     uint32_t begin, uint32_t count,
					     void *output)
{
	for (uint32_t table = 0; table < TUNE_TABLES; table++)
		for (uint32_t token = 0; token < count; token++)
			engram_prefetch_entry(
				context->engram, table,
				trace->rows +
					(size_t)(begin + token) *
						TUNE_TOTAL_ROWS +
					(size_t)(table * TUNE_ROWS_PER_TABLE),
				TUNE_ROWS_PER_TABLE);
	for (uint32_t table = 0; table < TUNE_TABLES; table++) {
		for (uint32_t token = 0; token < count; token++) {
			const result_status status = engram_gather_entry(
				context->engram, table,
				trace->rows +
					(size_t)(begin + token) *
						TUNE_TOTAL_ROWS +
					(size_t)(table * TUNE_ROWS_PER_TABLE),
				TUNE_ROWS_PER_TABLE,
				(unsigned char *)output +
					(size_t)token * TUNE_TOKEN_BYTES +
					(size_t)table * TUNE_ROWS_PER_TABLE *
						ENGRAM_ROW_BYTES);
			if (status != STATUS_OK)
				return status;
		}
	}
	return STATUS_OK;
}
MUST_CHECK static result_status gather_token(struct tune_context *context,
					     const struct tune_trace *trace,
					     uint32_t token, void *output)
{
	return gather_chunk(context, trace, token, 1u, output);
}
static void *lookup_reads(void *argument)
{
	struct lookup_worker *const worker = argument;
	struct tune_context *const context = worker->context;
	worker->status = STATUS_OK;
	unsigned char *const output =
		malloc((size_t)TUNE_PREFILL_CHUNK * TUNE_TOKEN_BYTES);
	if (output == NULL) {
		worker->status = STATUS_OUT_OF_MEMORY;
		return NULL;
	}
	uint64_t chunks = 0;
	for (uint32_t begin = 0;
	     begin + 1u < worker->trace->prompt && worker->status == STATUS_OK;
	     begin += TUNE_PREFILL_CHUNK) {
		if (tune_stopped(context)) {
			worker->status = STATUS_TIMED_OUT;
			break;
		}
		const uint32_t count =
			worker->trace->prompt - 1u - begin <
					TUNE_PREFILL_CHUNK ?
				worker->trace->prompt - 1u - begin :
				TUNE_PREFILL_CHUNK;
		const uint64_t started = tune_now();
		worker->status = gather_chunk(context, worker->trace, begin,
					      count, output);
		worker->prefill += tune_now() - started;
		worker->rows += (uint64_t)count * TUNE_TOTAL_ROWS;
		chunks++;
		for (uint32_t token = 0;
		     token < count && worker->status == STATUS_OK; token++) {
			char hash[TUNE_HASH_BYTES];
			worker->status = prepare_hash(
				output + (size_t)token * TUNE_TOKEN_BYTES,
				TUNE_TOKEN_BYTES, hash);
			if (worker->status == STATUS_OK &&
			    strcmp(hash, worker->trace->reference +
						 (size_t)(begin + token) *
							 TUNE_HASH_BYTES) != 0)
				worker->status = STATUS_CORRUPT_PACK;
		}
	}
	if (chunks != 0)
		worker->prefill /= chunks;
	for (uint32_t step = 0;
	     step < TUNE_STEPS &&
	     step < worker->trace->count - worker->trace->prompt &&
	     worker->status == STATUS_OK;
	     step++) {
		if (tune_stopped(context)) {
			worker->status = STATUS_TIMED_OUT;
			break;
		}
		const uint32_t token = worker->trace->prompt - 1u + step;
		const uint64_t started = tune_now();
		worker->status =
			gather_token(context, worker->trace, token, output);
		worker->steps[step] = tune_now() - started;
		worker->step_count++;
		worker->rows += TUNE_TOTAL_ROWS;
		char hash[TUNE_HASH_BYTES] = { 0 };
		if (worker->status == STATUS_OK)
			worker->status =
				prepare_hash(output, TUNE_TOKEN_BYTES, hash);
		if (worker->status == STATUS_OK &&
		    strcmp(hash, worker->trace->reference +
					 (size_t)token * TUNE_HASH_BYTES) != 0)
			worker->status = STATUS_CORRUPT_PACK;
	}
	free(output);
	return NULL;
}
static int compare_ns(const void *left, const void *right)
{
	const uint64_t a = *(const uint64_t *)left,
		       b = *(const uint64_t *)right;
	return a < b ? -1 : a > b ? 1 : 0;
}
MUST_CHECK static result_status lookup_point(struct tune_context *context,
					     struct tune_point *point)
{
	result_status status = engram_disk_tune_io(context->disk,
						   point->weights,
						   context->copies, point->io,
						   point->cached != 0);
	for (uint32_t level = 0;
	     status == STATUS_OK && level < TUNE_STREAM_LEVELS; level++) {
		const uint32_t streams = stream_levels[level];
		uint64_t timings[(PREPARE_TRACE_CASES + TUNE_STREAMS_MOST) *
				 TUNE_STEPS];
		uint32_t timing_count = 0;
		for (uint32_t first = 0;
		     status == STATUS_OK && first < context->trace_count;
		     first += streams) {
			pthread_t threads[TUNE_STREAMS_MOST];
			struct lookup_worker workers[TUNE_STREAMS_MOST];
			memset(workers, 0, sizeof(workers));
			uint32_t launched = 0;
			const uint64_t group_started = tune_now();
			for (uint32_t index = 0;
			     status == STATUS_OK && index < streams; index++) {
				workers[index].context = context;
				workers[index].trace =
					&context->traces[(first + index) %
							 context->trace_count];
				if (pthread_create(&threads[index], NULL,
						   lookup_reads,
						   &workers[index]) != 0)
					status = STATUS_OUT_OF_MEMORY;
				else
					launched++;
			}
			for (uint32_t index = 0; index < launched; index++) {
				if (pthread_join(threads[index], NULL) != 0)
					status = STATUS_INPUT_OUTPUT_ERROR;
				if (workers[index].status != STATUS_OK)
					status = workers[index].status;
				memcpy(timings + timing_count,
				       workers[index].steps,
				       (size_t)workers[index].step_count *
					       sizeof(*timings));
				timing_count += workers[index].step_count;
				if (workers[index].prefill >
				    point->prefill[level])
					point->prefill[level] =
						workers[index].prefill;
				point->rows[level] += workers[index].rows;
			}
			point->wall[level] += tune_now() - group_started;
		}
		if (status == STATUS_OK && timing_count == 0)
			status = STATUS_UNSUPPORTED;
		if (status == STATUS_OK && timing_count != 0) {
			qsort(timings, timing_count, sizeof(*timings),
			      compare_ns);
			point->p50[level] =
				timings[timing_count / TUNE_MEDIAN_DIVISOR];
			point->p99[level] =
				timings[(timing_count - 1u) * TUNE_P99_PERCENT /
					TUNE_QUANTILE_PERCENT];
		}
	}
	return status;
}
static bool trace_open(struct tune_context *context,
		       char digest[TUNE_HASH_BYTES])
{
	uint32_t total_tokens = 0;
	char trace_hashes[PREPARE_TRACE_CASES][TUNE_HASH_BYTES];
	memset(trace_hashes, 0, sizeof(trace_hashes));
	for (uint32_t index = 0; index < context->options->trace_count;
	     index++) {
		const struct prepare_trace_input *const input =
			&context->options->traces[index];
		const int descriptor =
			open(input->path,
			     O_RDONLY | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK);
		struct stat file;
		bool valid = descriptor >= 0 && fstat(descriptor, &file) == 0 &&
			     S_ISREG(file.st_mode) && file.st_size > 0 &&
			     (uint64_t)file.st_size % sizeof(uint32_t) == 0 &&
			     (uint64_t)file.st_size / sizeof(uint32_t) <=
				     TUNE_TOKEN_LIMIT - total_tokens;
		uint32_t *tokens = valid ? malloc((size_t)file.st_size) : NULL;
		valid = valid && tokens != NULL;
		size_t done = 0;
		while (valid && done < (size_t)file.st_size) {
			const ssize_t amount =
				read(descriptor, (unsigned char *)tokens + done,
				     (size_t)file.st_size - done);
			if (amount < 0 && errno == EINTR)
				continue;
			if (amount <= 0) {
				valid = false;
				break;
			}
			done += (size_t)amount;
		}
		if (descriptor >= 0)
			(void)close(descriptor);
		struct tune_trace *const trace = &context->traces[index];
		trace->count = valid ? (uint32_t)((uint64_t)file.st_size /
						  sizeof(uint32_t)) :
				       0;
		trace->prompt = input->prompt != 0 ? input->prompt :
				trace->count > TUNE_STEPS ?
						     trace->count - TUNE_STEPS :
						     trace->count;
		valid = valid && trace->prompt != 0 &&
			trace->prompt <= trace->count;
		trace->rows =
			valid ? malloc((size_t)trace->count * TUNE_TOTAL_ROWS *
				       sizeof(uint64_t)) :
				NULL;
		trace->reference =
			valid ? calloc(trace->count, TUNE_HASH_BYTES) : NULL;
		valid = valid && trace->rows != NULL &&
			trace->reference != NULL &&
			prepare_hash(tokens, done, trace_hashes[index]) ==
				STATUS_OK;
		for (uint32_t token = 0; valid && token < trace->count;
		     token++) {
			if (tune_stopped(context)) {
				valid = false;
				break;
			}
			valid = engram_row_ids_entry(
					context->engram, tokens, NULL, token,
					trace->rows +
						(size_t)token *
							TUNE_TOTAL_ROWS) ==
				STATUS_OK;
		}
		free(tokens);
		if (!valid)
			return false;
		total_tokens += trace->count;
		context->trace_count++;
	}
	struct {
		char hashes[PREPARE_TRACE_CASES][TUNE_HASH_BYTES];
		char pack[TUNE_HASH_BYTES];
		uint64_t pack_bytes;
		uint32_t prompts[PREPARE_TRACE_CASES];
	} identity = { 0 };
	memcpy(identity.hashes, trace_hashes, sizeof(trace_hashes));
	(void)snprintf(identity.pack, sizeof(identity.pack), "%s",
		       context->options->manifest->pack_identity);
	identity.pack_bytes = context->options->manifest->bytes;
	for (uint32_t index = 0; index < context->trace_count; index++)
		identity.prompts[index] = context->traces[index].prompt;
	return prepare_hash(&identity, sizeof(identity), digest) == STATUS_OK;
}
static uint32_t reuse_prefix(const uint32_t *tree, uint32_t position)
{
	uint32_t sum = 0;
	while (position != 0) {
		sum += tree[position];
		position &= position - 1u;
	}
	return sum;
}
static void reuse_update(uint32_t *tree, uint32_t count, uint32_t position,
			 bool add)
{
	while (position <= count) {
		if (add)
			tree[position]++;
		else
			tree[position]--;
		position += position & (~position + 1u);
	}
}
MUST_CHECK static result_status trace_statistics(struct tune_context *context)
{
	uint32_t references = 0;
	for (uint32_t index = 0; index < context->trace_count; index++)
		references += context->traces[index].count * TUNE_TOTAL_ROWS;
	if (references == 0)
		return STATUS_INVALID_ARGUMENT;
	uint64_t *const ordered = malloc((size_t)references * sizeof(*ordered));
	if (ordered == NULL)
		return STATUS_OUT_OF_MEMORY;
	uint32_t position = 0;
	for (uint32_t index = 0; index < context->trace_count; index++)
		for (uint32_t row = 0;
		     row < context->traces[index].count * TUNE_TOTAL_ROWS;
		     row++) {
			const uint64_t id = context->traces[index].rows[row];
			if (id >= UINT32_MAX) {
				free(ordered);
				return STATUS_UNSUPPORTED;
			}
			ordered[position++] =
				((uint64_t)(row % TUNE_TOTAL_ROWS /
					    TUNE_ROWS_PER_TABLE)
				 << TUNE_TABLE_SHIFT) |
				id;
		}
	qsort(ordered, references, sizeof(*ordered), compare_ns);
	context->unique_rows = 1;
	for (uint32_t index = 1; index < references; index++)
		if (ordered[index] != ordered[index - 1u])
			context->unique_rows++;
	free(ordered);
	uint32_t slots = 1;
	while (slots < references * TUNE_SLOT_FACTOR)
		slots *= TUNE_SLOT_FACTOR;
	uint64_t *const keys = calloc(slots, sizeof(*keys));
	uint32_t *const last = calloc(slots, sizeof(*last)),
			*const tree =
				calloc((size_t)references + 1u, sizeof(*tree));
	if (keys == NULL || last == NULL || tree == NULL) {
		free(keys);
		free(last);
		free(tree);
		return STATUS_OUT_OF_MEMORY;
	}
	position = 0;
	for (uint32_t index = 0; index < context->trace_count; index++)
		for (uint32_t row = 0;
		     row < context->traces[index].count * TUNE_TOTAL_ROWS;
		     row++) {
			if (tune_stopped(context)) {
				free(keys);
				free(last);
				free(tree);
				return STATUS_TIMED_OUT;
			}
			const uint64_t key =
				(((uint64_t)(row % TUNE_TOTAL_ROWS /
					     TUNE_ROWS_PER_TABLE)
				  << TUNE_TABLE_SHIFT) |
				 context->traces[index].rows[row] /
					 ENGRAM_ROWS_PER_BLOCK) +
				1u;
			uint32_t slot =
				(uint32_t)((key ^ (key >> TUNE_TABLE_SHIFT)) *
					   TUNE_HASH_MULTIPLIER) &
				(slots - 1u);
			while (keys[slot] != 0 && keys[slot] != key)
				slot = (slot + 1u) & (slots - 1u);
			position++;
			if (keys[slot] != 0) {
				uint32_t distance =
					reuse_prefix(tree, position - 1u) -
					reuse_prefix(tree, last[slot]);
				uint32_t bin = 0;
				while (distance > 1u) {
					distance >>= 1u;
					bin++;
				}
				context->reuse[bin]++;
				reuse_update(tree, references, last[slot],
					     false);
			} else {
				context->cold_blocks++;
				keys[slot] = key;
			}
			last[slot] = position;
			reuse_update(tree, references, position, true);
		}
	free(keys);
	free(last);
	free(tree);
	context->row_references = references;
	uint64_t reused = 0;
	for (uint32_t bin = 0; bin < TUNE_REUSE_BINS; bin++)
		reused += context->reuse[bin];
	uint64_t cumulative = 0, blocks = 1;
	for (uint32_t bin = 0; bin < TUNE_REUSE_BINS; bin++) {
		cumulative += context->reuse[bin];
		blocks = 1ull << (bin + 1u);
		if (cumulative * TUNE_QUANTILE_PERCENT >=
		    reused * TUNE_CACHE_PERCENTILE)
			break;
	}
	uint64_t mib = (blocks * PACK_BLOCK_BYTES + TUNE_MIB_BYTES - 1u) /
		       TUNE_MIB_BYTES;
	if (mib < TUNE_MIN_CACHE_MIB)
		mib = TUNE_MIN_CACHE_MIB;
	if (mib > TUNE_MAX_CACHE_MIB)
		mib = TUNE_MAX_CACHE_MIB;
	context->cache_mib = (uint32_t)mib;
	(void)fprintf(stderr,
		      "storage: trace rows=%" PRIu64 " unique_rows=%" PRIu64
		      " cold_blocks=%" PRIu64
		      " cache_mib=%u (conditional block-reuse p90, capped)\n",
		      context->row_references, context->unique_rows,
		      context->cold_blocks, context->cache_mib);
	return STATUS_OK;
}
MUST_CHECK static result_status references_prepare(struct tune_context *context)
{
	uint32_t weights[STORAGE_MAXIMUM_COPIES] = { 1 };
	result_status status = engram_disk_tune_io(context->disk, weights,
						   context->copies, 0, true);
	unsigned char bytes[TUNE_TOKEN_BYTES];
	for (uint32_t index = 0;
	     status == STATUS_OK && index < context->trace_count; index++)
		for (uint32_t token = 0; status == STATUS_OK &&
					 token < context->traces[index].count;
		     token++) {
			if (tune_stopped(context)) {
				status = STATUS_TIMED_OUT;
				break;
			}
			status = gather_token(context, &context->traces[index],
					      token, bytes);
			if (status == STATUS_OK)
				status = prepare_hash(
					bytes, sizeof(bytes),
					context->traces[index].reference +
						(size_t)token *
							TUNE_HASH_BYTES);
		}
	return status;
}
static bool state_valid(const struct tune_state *state)
{
	if (state->points > TUNE_POINTS_MOST || state->count == 0 ||
	    state->count > STORAGE_MAXIMUM_COPIES ||
	    state->created > (uint64_t)time(NULL))
		return false;
	for (uint32_t index = 0; index < state->points; index++) {
		const struct tune_point *const point = &state->values[index];
		if (point->action > 1u || point->io > 1u ||
		    point->cached >= TUNE_CACHE_MODES ||
		    (point->action == 0 && point->cached != 0) ||
		    point->threads == 0 ||
		    point->threads > TUNE_LOAD_THREADS_MOST ||
		    (unsigned)point->status >= STATUS_COUNT)
			return false;
		uint32_t shares = 0;
		for (uint32_t copy = 0; copy < state->count; copy++) {
			if (point->weights[copy] > TUNE_WEIGHT_SCALE)
				return false;
			shares += point->weights[copy];
		}
		if (shares == 0)
			return false;
	}
	return true;
}
static bool state_write(const char *path, const struct tune_state *state)
{
	struct prepare_file file;
	if (prepare_file_open(path, &file) != STATUS_OK)
		return false;
	if (fwrite(state, sizeof(*state), 1, file.stream) != 1) {
		prepare_file_abort(&file);
		return false;
	}
	return prepare_file_commit(&file) == STATUS_OK;
}
static bool point_same(const struct tune_point *a, const struct tune_point *b)
{
	return a->action == b->action && a->io == b->io &&
	       a->threads == b->threads && a->cached == b->cached &&
	       memcmp(a->weights, b->weights, sizeof(a->weights)) == 0;
}
MUST_CHECK static result_status point_measure(struct tune_context *context,
					      struct tune_state *state,
					      const char *path,
					      struct tune_point point)
{
	for (uint32_t index = 0; index < state->points; index++)
		if (point_same(&state->values[index], &point))
			return STATUS_OK;
	if (tune_stopped(context))
		return STATUS_TIMED_OUT;
	if (state->points == TUNE_POINTS_MOST)
		return STATUS_OUT_OF_MEMORY;
	point.status = point.action == 0 ? load_point(context, &point) :
					   lookup_point(context, &point);
	if (point.status == STATUS_TIMED_OUT)
		return point.status;
	state->values[state->points++] = point;
	(void)fprintf(
		stderr,
		"storage: tune action=%s point=%u status=%s threads=%u io=%u cached=%u page_locked=%u\n",
		point.action == 0 ? "load" : "lookup", state->points,
		status_text(point.status), point.threads, point.io,
		point.cached, point.page_locked ? 1u : 0u);
	return state_write(path, state) ? STATUS_OK : STATUS_INPUT_OUTPUT_ERROR;
}
/* A point's cost, lower is better: a load's nanoseconds per MiB read (its points read different amounts: the shares
 * set the volume), a lookup's worst p99 step over the stream levels. */
static uint64_t point_cost(const struct tune_point *point)
{
	if (point->status != STATUS_OK)
		return UINT64_MAX;
	if (point->action == 0)
		return point->bytes == 0 ||
				       point->ns > UINT64_MAX / TUNE_MIB_BYTES ?
			       UINT64_MAX :
			       point->ns * TUNE_MIB_BYTES / point->bytes;
	uint64_t worst = 0;
	for (uint32_t level = 0; level < TUNE_STREAM_LEVELS; level++)
		if (point->p99[level] > worst)
			worst = point->p99[level];
	return worst == 0 ? UINT64_MAX : worst;
}
/* The fastest point of an action: on one copy (solo) or on several; a lookup's among those of one cache mode. */
static const struct tune_point *best_point(const struct tune_state *state,
					   uint32_t action, uint32_t solo,
					   bool aggregate, uint32_t cached)
{
	const struct tune_point *best = NULL;
	for (uint32_t index = 0; index < state->points; index++) {
		const struct tune_point *const point = &state->values[index];
		if (point->action != action || point->status != STATUS_OK ||
		    (action != 0 && point->cached != cached))
			continue;
		uint32_t positive = 0, device = 0;
		for (uint32_t copy = 0; copy < state->count; copy++)
			if (point->weights[copy] != 0) {
				positive++;
				device = copy;
			}
		if ((!aggregate && (positive != 1 || device != solo)) ||
		    (aggregate && (positive == 0 ||
				   (action == 0 && positive != state->count))))
			continue;
		if (best == NULL || point_cost(point) < point_cost(best))
			best = point;
	}
	return best;
}
static void selection_json(FILE *output, const struct tune_state *state,
			   uint32_t action, const struct tune_point *winner)
{
	uint64_t runner = UINT64_MAX;
	for (uint32_t index = 0; winner != NULL && index < state->points;
	     index++) {
		const struct tune_point *const point = &state->values[index];
		if (point == winner || point->action != action ||
		    point->status != STATUS_OK ||
		    (action != 0 && point->cached != winner->cached))
			continue;
		bool eligible = true;
		if (action == 0)
			for (uint32_t copy = 0; copy < state->count; copy++)
				eligible = eligible &&
					   point->weights[copy] != 0;
		if (eligible && point_cost(point) < runner)
			runner = point_cost(point);
	}
	const uint64_t cost = winner != NULL ? point_cost(winner) : 0;
	(void)fprintf(output,
		      "{\"cost_unit\":\"%s\",\"winner_cost\":%" PRIu64
		      ",\"runner_up_cost\":%" PRIu64
		      ",\"relative_margin\":%.9f}",
		      action == 0 ? "ns_per_mib" : "worst_p99_step_ns", cost,
		      runner != UINT64_MAX ? runner : 0,
		      cost != 0 && runner != UINT64_MAX ?
			      (double)runner / (double)cost - 1.0 :
			      0.0);
}
MUST_CHECK static result_status profile_write(struct tune_context *context,
					      const struct tune_state *state)
{
	/* The candidate's lookup runs with its bounded cache, so only cached points can win it. */
	const struct tune_point *const load = best_point(state, 0, 0, true, 0),
				       *const lookup =
					       best_point(state, 1, 0, true, 1);
	const struct prepare_tune_options *const options = context->options;
	const int descriptor =
		open(options->manifest_path, O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
	struct stat file;
	bool valid = descriptor >= 0 && fstat(descriptor, &file) == 0 &&
		     file.st_size > 0 &&
		     (uint64_t)file.st_size <= TUNE_MANIFEST_MOST_BYTES;
	char *manifest = valid ? malloc((size_t)file.st_size) : NULL;
	char manifest_hash[TUNE_HASH_BYTES];
	valid = valid && manifest != NULL &&
		read(descriptor, manifest, (size_t)file.st_size) ==
			file.st_size &&
		prepare_hash(manifest, (size_t)file.st_size, manifest_hash) ==
			STATUS_OK;
	free(manifest);
	if (descriptor >= 0)
		(void)close(descriptor);
	if (!valid)
		return STATUS_INPUT_OUTPUT_ERROR;
	struct prepare_file output;
	result_status status = prepare_file_open(options->profile, &output);
	if (status != STATUS_OK)
		return status;
	(void)fprintf(
		output.stream,
		"{\"schema\":%u,\"enabled\":0,\"qualification\":\"native_gates_required\",\"created_unix\":%" PRIu64
		",\"source\":",
		TUNE_PROFILE_SCHEMA, state->created);
	prepare_json_text(output.stream, options->manifest->source);
	(void)fputs(",\"manifest\":", output.stream);
	prepare_json_text(output.stream, options->manifest_path);
	(void)fputs(",\"manifest_sha256\":", output.stream);
	prepare_json_text(output.stream, manifest_hash);
	(void)fprintf(
		output.stream,
		",\"replica_policy\":{\"external\":%u,\"floor_percent\":%u},\"turbo_load\":{\"enabled\":%u,\"threads\":"
		"%u,\"mechanism\":\"direct_pread\",\"share\":\"steal\"},\"turbo_engram\":{\"enabled\":%u,"
		"\"ram_percent\":0,\"cache_mib\":%u,\"io\":%u,\"mechanism\":\"%s\","
		"\"step_io\":{\"selected\":\"hybrid\",\"threads_per_copy\":%u,\"spin_us\":%u,\"pin\":1,"
		"\"applies\":\"serving PP5 or TP4_PRO from disk tables\",\"evidence\":\"%s\"}},\"devices\":[",
		options->external ? 1u : 0u, options->floor_percent,
		load != NULL ? 1u : 0u, load != NULL ? load->threads : 1u,
		lookup != NULL ? 1u : 0u, context->cache_mib,
		lookup != NULL ? lookup->io : 0,
		lookup != NULL && lookup->io != 0 ? "uring_fixed_direct" :
						    "direct_pread",
		ENGRAM_STEP_IO_THREADS_DEFAULT, ENGRAM_STEP_SPIN_US_DEFAULT,
		STEP_IO_EVIDENCE);
	for (uint32_t index = 0; index < context->copies; index++) {
		const struct prepare_target *const target =
			&options->devices[context->device_indices[index]];
		(void)fputs(index == 0 ? "{\"path\":" : ",{\"path\":",
			    output.stream);
		prepare_json_text(output.stream, target->path);
		(void)fputs(",\"key\":", output.stream);
		prepare_json_text(output.stream, target->key);
		(void)fprintf(output.stream,
			      ",\"load_weight\":%u,\"engram_weight\":%u}",
			      load != NULL ? load_share(load, index) : 0,
			      lookup != NULL ? lookup->weights[index] : 0);
	}
	(void)fputs("],\"decisions\":[", output.stream);
	for (uint32_t index = 0; index < options->count; index++) {
		if (index != 0)
			(void)fputc(',', output.stream);
		prepare_decision_json(output.stream, &options->devices[index]);
	}
	(void)fputs("],\"measurements\":[", output.stream);
	for (uint32_t index = 0; index < state->points; index++) {
		const struct tune_point *const point = &state->values[index];
		(void)fprintf(
			output.stream,
			"%s{\"action\":\"%s\",\"status\":%u,\"io\":%u,\"threads\":%u,\"cached\":%u,\"ns\":%" PRIu64
			",\"bytes\":%" PRIu64
			",\"page_locked\":%u,\"weights\":[",
			index == 0 ? "" : ",",
			point->action == 0 ? "load" : "lookup",
			(unsigned)point->status, point->io, point->threads,
			point->cached, point->ns, point->bytes,
			point->page_locked ? 1u : 0u);
		for (uint32_t copy = 0; copy < context->copies; copy++)
			(void)fprintf(output.stream, "%s%u",
				      copy == 0 ? "" : ",",
				      point->weights[copy]);
		(void)fputs("],\"copy_chunks\":[", output.stream);
		for (uint32_t copy = 0; copy < context->copies; copy++)
			(void)fprintf(output.stream, "%s%u",
				      copy == 0 ? "" : ",",
				      point->copy_chunks[copy]);
		(void)fputs("],\"steps\":[", output.stream);
		for (uint32_t level = 0; level < TUNE_STREAM_LEVELS; level++)
			(void)fprintf(
				output.stream,
				"%s{\"streams\":%u,\"p50_ns\":%" PRIu64
				",\"p99_ns\":%" PRIu64
				",\"prefill_chunk_mean_ns\":%" PRIu64
				",\"rows\":%" PRIu64 ",\"wall_ns\":%" PRIu64
				",\"rows_per_second\":%.3f}",
				level == 0 ? "" : ",", stream_levels[level],
				point->p50[level], point->p99[level],
				point->prefill[level], point->rows[level],
				point->wall[level],
				point->wall[level] != 0 ?
					(double)point->rows[level] *
						TUNE_NS_SECOND /
						(double)point->wall[level] :
					0.0);
		(void)fputs("]}", output.stream);
	}
	(void)fputs("],\"trace_identity\":", output.stream);
	prepare_json_text(output.stream, state->trace);
	(void)fputs(",\"selection\":{\"load\":", output.stream);
	selection_json(output.stream, state, 0, load);
	(void)fputs(",\"lookup\":", output.stream);
	selection_json(output.stream, state, 1, lookup);
	(void)fputs("}", output.stream);
	(void)fprintf(output.stream,
		      ",\"trace\":{\"row_references\":%" PRIu64
		      ",\"unique_rows\":%" PRIu64 ",\"cold_blocks\":%" PRIu64
		      ",\"reuse_distance_blocks_histogram\":[",
		      context->row_references, context->unique_rows,
		      context->cold_blocks);
	for (uint32_t bin = 0; bin < TUNE_REUSE_BINS; bin++)
		(void)fprintf(output.stream, "%s%" PRIu64, bin == 0 ? "" : ",",
			      context->reuse[bin]);
	(void)fputs(
		"]},\"measurement_scope\":\"CPU production gather and page-locked I/O only; startup-to-serving, decode "
		"slowdown, DSpark verification and native numerical/state gates unmeasured\"}\n",
		output.stream);
	status = prepare_file_commit(&output);
	if (status == STATUS_OK) {
		for (uint32_t index = 0; index < context->copies; index++) {
			const struct prepare_target *const target =
				&options->devices[context->device_indices[index]];
			local_device_print(
				stderr, &target->device,
				"verified_identity_and_bytes", 0,
				options->floor_percent,
				load != NULL ? load_share(load, index) : 0,
				lookup != NULL ? lookup->weights[index] : 0);
		}
		(void)fputs(
			"storage: tuned candidate saved; native qualification required before activation; startup "
			"never benchmarks\n",
			stderr);
	}
	return status;
}
/* One copy set (a bit per copy): load at every thread count a copy (the full set only: a load reads every copy, so
 * only its points can win; its readers steal, so a load point has no shares to vary), and lookup with every io and
 * cache mode, each with equal shares, then shares proportional to each copy's solo speed and to its square (the
 * fastest copy of the set at TUNE_WEIGHT_SCALE). */
MUST_CHECK static result_status set_measure(struct tune_context *context,
					    struct tune_state *state,
					    const char *path, uint32_t set)
{
	result_status status = STATUS_OK;
	const uint32_t full_set = (1u << context->copies) - 1u;
	for (uint32_t action = set == full_set ? 0u : 1u;
	     status == STATUS_OK && action <= 1u; action++)
		for (uint32_t mode = 0;
		     status == STATUS_OK &&
		     mode < (action == 0 ? 1u : TUNE_CACHE_MODES);
		     mode++) {
			const uint32_t cached = action != 0 && mode == 0 ? 1u :
									   0u;
			uint64_t fastest = UINT64_MAX;
			uint64_t costs[STORAGE_MAXIMUM_COPIES];
			for (uint32_t copy = 0; copy < context->copies;
			     copy++) {
				const struct tune_point *const solo =
					(set & (1u << copy)) != 0 ?
						best_point(state, action, copy,
							   false, cached) :
						NULL;
				costs[copy] = solo != NULL ? point_cost(solo) :
							     UINT64_MAX;
				if (costs[copy] < fastest)
					fastest = costs[copy];
			}
			const uint32_t mechanisms =
				action == 0 ? TUNE_THREAD_OPTIONS :
					      TUNE_IO_MECHANISMS;
			for (uint32_t variant = 0;
			     status == STATUS_OK &&
			     variant <
				     (action == 0 ? 1u : TUNE_WEIGHT_VARIANTS);
			     variant++)
				for (uint32_t mechanism = 0;
				     status == STATUS_OK &&
				     mechanism < mechanisms;
				     mechanism++) {
					struct tune_point point = {
						.action = action,
						.threads =
							action == 0 ?
								thread_options
									[mechanism] :
								1u,
						.io = action != 0 ? mechanism :
								    0u,
						.cached = cached
					};
					uint32_t total = 0;
					for (uint32_t copy = 0;
					     copy < context->copies; copy++) {
						if ((set & (1u << copy)) == 0)
							continue;
						if (variant == 0)
							point.weights[copy] = 1;
						else if (costs[copy] !=
								 UINT64_MAX &&
							 costs[copy] != 0 &&
							 fastest !=
								 UINT64_MAX) {
							uint64_t weight =
								fastest > UINT64_MAX / TUNE_WEIGHT_SCALE ?
									1u :
									fastest *
										TUNE_WEIGHT_SCALE /
										costs[copy];
							if (variant ==
							    TUNE_SQUARED_VARIANT)
								weight =
									weight *
									weight /
									TUNE_WEIGHT_SCALE;
							point.weights[copy] =
								weight != 0 ?
									(uint32_t)
										weight :
									1u;
						}
						total += point.weights[copy];
					}
					if (total != 0)
						status = point_measure(context,
								       state,
								       path,
								       point);
				}
		}
	return status;
}
result_status prepare_tune(const struct prepare_tune_options *options)
{
	if (options == NULL || options->manifest == NULL ||
	    options->profile == NULL || options->manifest_path == NULL ||
	    options->devices == NULL ||
	    options->count > LOCAL_DEVICE_CANDIDATES_MOST ||
	    options->trace_count == 0 ||
	    options->trace_count > PREPARE_TRACE_CASES)
		return STATUS_INVALID_ARGUMENT;
	struct tune_context *const context = calloc(1, sizeof(*context));
	struct tune_state *const state = calloc(1, sizeof(*state));
	if (context == NULL || state == NULL) {
		free(context);
		free(state);
		return STATUS_OUT_OF_MEMORY;
	}
	context->options = options;
	const char *paths[STORAGE_MAXIMUM_COPIES];
	for (uint32_t index = 0;
	     index < options->count && context->copies < STORAGE_MAXIMUM_COPIES;
	     index++)
		if (options->devices[index].included &&
		    options->devices[index].complete) {
			paths[context->copies] = options->devices[index].path;
			(void)snprintf(state->keys[context->copies],
				       sizeof(state->keys[context->copies]),
				       "%s", options->devices[index].key);
			context->device_indices[context->copies++] = index;
		}
	state->schema = TUNE_SCHEMA;
	state->count = context->copies;
	state->created = (uint64_t)time(NULL);
	struct storage_reader *reader = NULL;
	struct pack_record *pack = NULL;
	result_status status =
		context->copies != 0 ?
			storage_direct.reader_open(paths, context->copies,
						   &reader) :
			STATUS_UNSUPPORTED;
	context->reader = reader;
	if (status == STATUS_OK)
		status = pack_open_entry(&storage_direct, reader, &pack);
	if (status == STATUS_OK)
		context->pack_bytes = pack_header_of(pack)->file_byte_count;
	if (status == STATUS_OK)
		status = engram_open(&storage_direct, reader, pack,
				     &context->engram);
	const struct engram_description *const description =
		engram_description_of(context->engram);
	if (status == STATUS_OK &&
	    (description->table_count != TUNE_TABLES ||
	     description->rows_per_token != TUNE_ROWS_PER_TABLE))
		status = STATUS_UNSUPPORTED;
	if (status == STATUS_OK && !trace_open(context, state->trace))
		status = tune_stopped(context) ? STATUS_TIMED_OUT :
						 STATUS_INVALID_ARGUMENT;
	if (status == STATUS_OK)
		status = trace_statistics(context);
	if (status == STATUS_OK)
		status = engram_disk_open_until(description, pack,
						&storage_direct, reader, 0,
						TUNE_CACHE_BYTES, 1, 0,
						options->deadline_ns,
						options->stop, &context->disk);
	if (status == STATUS_OK)
		status = engram_attach_disk(context->engram, context->disk);
	char state_path[PATH_MAX];
	const int length = snprintf(state_path, sizeof(state_path),
				    "%s.tune-state", options->profile);
	if (status == STATUS_OK &&
	    (length < 0 || (size_t)length >= sizeof(state_path)))
		status = STATUS_INVALID_ARGUMENT;
	if (status == STATUS_OK) {
		struct tune_state *const prior = malloc(sizeof(*prior));
		const int descriptor =
			open(state_path,
			     O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
		struct stat file;
		if (prior != NULL && descriptor >= 0 &&
		    fstat(descriptor, &file) == 0 && S_ISREG(file.st_mode) &&
		    file.st_uid == geteuid() &&
		    (file.st_mode & (S_IWGRP | S_IWOTH)) == 0 &&
		    file.st_size == (off_t)sizeof(*prior) &&
		    read(descriptor, prior, sizeof(*prior)) ==
			    (ssize_t)sizeof(*prior) &&
		    prior->schema == TUNE_SCHEMA &&
		    prior->count == state->count && state_valid(prior) &&
		    memcmp(prior->keys, state->keys, sizeof(state->keys)) ==
			    0 &&
		    memcmp(prior->trace, state->trace, sizeof(state->trace)) ==
			    0)
			*state = *prior;
		free(prior);
		if (descriptor >= 0)
			(void)close(descriptor);
	}
	if (status == STATUS_OK)
		status = references_prepare(context);
	/* Solo: each copy alone, load at every thread count, lookup with every io, cached and uncached. */
	for (uint32_t copy = 0; status == STATUS_OK && copy < context->copies;
	     copy++) {
		for (uint32_t threads = 0;
		     status == STATUS_OK && threads < TUNE_THREAD_OPTIONS;
		     threads++) {
			struct tune_point point = {
				.threads = thread_options[threads]
			};
			point.weights[copy] = 1;
			status = point_measure(context, state, state_path,
					       point);
		}
		for (uint32_t mode = 0;
		     status == STATUS_OK && mode < TUNE_CACHE_MODES; mode++)
			for (uint32_t io = 0;
			     status == STATUS_OK && io < TUNE_IO_MECHANISMS;
			     io++) {
				struct tune_point point = {
					.action = 1,
					.io = io,
					.threads = 1,
					.cached = mode == 0 ? 1u : 0u
				};
				point.weights[copy] = 1;
				status = point_measure(context, state,
						       state_path, point);
			}
	}
	/* Aggregates: every set of two or more copies, the full set first, each with its weight variants. */
	const uint32_t full_set = (1u << context->copies) - 1u;
	for (uint32_t set = full_set; status == STATUS_OK && set != 0; set--)
		if ((uint32_t)__builtin_popcount(set) >=
			    TUNE_SET_COPIES_LEAST &&
		    (set == full_set ||
		     context->copies <= TUNE_SET_COPIES_MOST))
			status = set_measure(context, state, state_path, set);
	if (status == STATUS_OK) {
		for (uint32_t index = 0; index < context->copies; index++) {
			struct prepare_target *const target =
				&options->devices[context->device_indices[index]];
			char key[TUNE_HASH_BYTES];
			struct local_device_description device;
			if (storage_device_key_for(target->path,
						   options->external, key,
						   &device) != STATUS_OK ||
			    strcmp(key, target->key) != 0) {
				target->included = target->complete = false;
				target->reason =
					"hardware_identity_mismatch_retune_required";
				local_device_print(stderr, &target->device,
						   target->reason, 0,
						   options->floor_percent, 0,
						   0);
				status = STATUS_UNSUPPORTED;
			}
		}
	}
	if (status == STATUS_OK)
		status = profile_write(context, state);
	engram_disk_close(context->disk);
	engram_close(context->engram);
	pack_close(pack);
	storage_direct.reader_close(reader);
	for (uint32_t index = 0; index < PREPARE_TRACE_CASES; index++) {
		free(context->traces[index].rows);
		free(context->traces[index].reference);
	}
	free(state);
	free(context);
	return status;
}
