/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * engram_io_characterise.c: what each drive delivers, measured in-tree (engram_io_bench.h).
 *
 * A plan holds one point per line:
 *   seq  copies=0[,1,2] size_kib=N depth=N threads=N [gib=N] [seconds=N] [engine=uring|pread]
 *        [share=copy|steal|weights:W0,W1,W2] [total_gib=N] [repeat=N]
 *   rand copies=0[,1,2] size_kib=N depth=N threads=N seconds=N [region=tables|file] [engine=uring|pread]
 * Every listed copy gets `threads` threads. A uring thread keeps `depth` O_DIRECT reads of `size` in flight on its own
 * ring (single issuer, deferred task work when the kernel allows, the copy registered, page-locked registered
 * buffers on 2 MiB pages); a pread thread reads `size` at a time. seq threads read disjoint sequential regions until
 * the copy has delivered gib GiB or the seconds pass; rand threads read aligned random offsets for the seconds.
 * share splits one load of total_gib GiB (0: the whole file) over the listed copies instead: steal hands the next
 * piece of a shared cursor to whichever thread asks first, so a faster drive takes more (work stealing); weights
 * gives each copy a contiguous region in proportion to its weight (a static split; weights:1,1,1 is today's equal
 * share). The row records each copy's bytes and the time its last thread finished. repeat only tells repeated points
 * apart (an ABBA order), since a resumed plan skips a point whose exact line already measured ok.
 * One JSON row per point: rates per copy and in total, the latency of each read, the threads' CPU time.
 * Nothing is written and no digest is needed: the bytes are only timed.
 */
#include "engram_io_bench.h"

#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <linux/io_uring.h>
#include <linux/time_types.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#define EXIT_CHECKPOINT 75
#define PLAN_LINE_BYTES 512u
#define KIB_BYTES 1024ull
#define GIB_BYTES (1ull << 30u)
#define THREADS_MOST 64u
#define DEPTH_MOST 1024u
#define IN_FLIGHT_BYTES_MOST (4ull << 30u)
#define REGISTERED_SPAN (1ull << 30u)
#define SPANS_MOST 8u
#define HUGE_PAGE_BYTES (2ull << 20u)
#define SECONDS_DEFAULT 10u
#define CLOCK_CHECK_READS 64u
#define PER_MILLE_P50 500u
#define PER_MILLE_P99 990u
#define PER_MILLE_P999 999u
#define RANDOM_MULTIPLIER 0x2545F4914F6CDD1Dull
#define RANDOM_SHIFT_A 12u
#define RANDOM_SHIFT_B 25u
#define RANDOM_SHIFT_C 27u
#define NO_OFFSET UINT64_MAX

enum share_kind { SHARE_COPY, SHARE_STEAL, SHARE_WEIGHTS };
static const char *const share_names[] = { "copy", "steal", "weights" };

struct point {
	bool sequential, uring, tables_region;
	enum share_kind share;
	uint32_t copy_mask, size_bytes, depth, threads,
		weights[BENCH_COPIES_MOST];
	uint64_t bytes_per_copy, seconds, total_bytes;
};
struct point_thread {
	const struct bench_store *store;
	const struct point *point;
	uint32_t copy, ordinal,
		of; /* this thread is ordinal of `of` on its copy */
	uint64_t deadline_ns, bytes_target;
	_Atomic uint64_t *copy_bytes;
	/* share=steal: the next piece of the load and its end; share=weights: this thread's part of its region */
	_Atomic uint64_t *shared_cursor;
	uint64_t limit, region_begin, region_end, finished_ns;
	struct bench_histogram *latency;
	uint64_t bytes, reads, cpu_ns;
	int error;
	pthread_t thread;
};

static bool point_parse(const char *line, struct point *point)
{
	memset(point, 0, sizeof(*point));
	point->uring = true;
	point->tables_region = true;
	point->seconds = SECONDS_DEFAULT;
	char text[PLAN_LINE_BYTES];
	(void)snprintf(text, sizeof(text), "%s", line);
	text[strcspn(text, "\r\n")] = '\0';
	char *save = NULL;
	const char *const mode = strtok_r(text, " ", &save);
	if (mode == NULL ||
	    (strcmp(mode, "seq") != 0 && strcmp(mode, "rand") != 0))
		return false;
	point->sequential = strcmp(mode, "seq") == 0;
	for (char *token = strtok_r(NULL, " ", &save); token != NULL;
	     token = strtok_r(NULL, " ", &save)) {
		char *const equals = strchr(token, '=');
		if (equals == NULL)
			return false;
		*equals = '\0';
		const char *const key = token, *const value = equals + 1;
		char *end = NULL;
		const unsigned long long number =
			isdigit((unsigned char)value[0]) ?
				strtoull(value, &end, 10) :
				0;
		const bool numeric = end != NULL && *end == '\0';
		if (strcmp(key, "copies") == 0) {
			for (const char *cursor = value; *cursor != '\0';) {
				char *after = NULL;
				const unsigned long copy =
					strtoul(cursor, &after, 10);
				if (after == cursor ||
				    copy >= BENCH_COPIES_MOST)
					return false;
				point->copy_mask |= 1u << copy;
				cursor = *after == ',' ? after + 1 : after;
				if (*after != ',' && *after != '\0')
					return false;
			}
		} else if (strcmp(key, "size_kib") == 0 && numeric &&
			   number != 0 && number * KIB_BYTES <= UINT32_MAX)
			point->size_bytes = (uint32_t)(number * KIB_BYTES);
		else if (strcmp(key, "depth") == 0 && numeric && number != 0 &&
			 number <= DEPTH_MOST)
			point->depth = (uint32_t)number;
		else if (strcmp(key, "threads") == 0 && numeric &&
			 number != 0 && number <= THREADS_MOST)
			point->threads = (uint32_t)number;
		else if (strcmp(key, "gib") == 0 && numeric)
			point->bytes_per_copy = number * GIB_BYTES;
		else if (strcmp(key, "total_gib") == 0 && numeric)
			point->total_bytes = number * GIB_BYTES;
		else if (strcmp(key, "repeat") == 0 && numeric)
			continue;
		else if (strcmp(key, "share") == 0 &&
			 (strcmp(value, "copy") == 0 ||
			  strcmp(value, "steal") == 0))
			point->share = strcmp(value, "steal") == 0 ?
					       SHARE_STEAL :
					       SHARE_COPY;
		else if (strcmp(key, "share") == 0 &&
			 strncmp(value, "weights:", strlen("weights:")) == 0) {
			point->share = SHARE_WEIGHTS;
			uint32_t copy = 0;
			for (const char *cursor = value + strlen("weights:");
			     *cursor != '\0'; copy++) {
				char *after = NULL;
				const unsigned long weight =
					strtoul(cursor, &after, 10);
				if (after == cursor ||
				    copy >= BENCH_COPIES_MOST ||
				    weight > UINT32_MAX ||
				    (*after != ',' && *after != '\0'))
					return false;
				point->weights[copy] = (uint32_t)weight;
				cursor = *after == ',' ? after + 1 : after;
			}
		} else if (strcmp(key, "seconds") == 0 && numeric &&
			   number != 0)
			point->seconds = number;
		else if (strcmp(key, "region") == 0 &&
			 (strcmp(value, "tables") == 0 ||
			  strcmp(value, "file") == 0))
			point->tables_region = strcmp(value, "tables") == 0;
		else if (strcmp(key, "engine") == 0 &&
			 (strcmp(value, "uring") == 0 ||
			  strcmp(value, "pread") == 0))
			point->uring = strcmp(value, "uring") == 0;
		else
			return false;
	}
	bool weighted = true;
	for (uint32_t copy = 0; copy < BENCH_COPIES_MOST; copy++)
		weighted = weighted && ((point->copy_mask >> copy & 1u) == 0 ||
					point->weights[copy] != 0);
	return point->copy_mask != 0 &&
	       (point->share == SHARE_COPY ||
		(point->sequential &&
		 (point->share != SHARE_WEIGHTS || weighted))) &&
	       point->size_bytes % BENCH_BLOCK_BYTES == 0 &&
	       point->size_bytes != 0 && point->depth != 0 &&
	       point->threads != 0;
}

static uint64_t mix_seed(uint32_t copy, uint32_t ordinal)
{
	uint64_t value =
		((uint64_t)copy << 32u | ordinal) + 0x9E3779B97F4A7C15ull;
	value = (value ^ (value >> 30u)) * 0xBF58476D1CE4E5B9ull;
	value = (value ^ (value >> 27u)) * 0x94D049BB133111EBull;
	return (value ^ (value >> 31u)) | 1u;
}

static uint64_t random_step(uint64_t *state)
{
	*state ^= *state >> RANDOM_SHIFT_A;
	*state ^= *state << RANDOM_SHIFT_B;
	*state ^= *state >> RANDOM_SHIFT_C;
	return *state * RANDOM_MULTIPLIER;
}

/* The next offset of a thread: the shared cursor's next piece (steal), the next piece of its region (weights), its
 * own sequential region (wrapping inside the file), or aligned and random. NO_OFFSET once a shared load is done. */
static uint64_t next_offset(const struct point_thread *self, uint64_t *cursor,
			    uint64_t *state)
{
	const struct bench_store *const store = self->store;
	const uint64_t size = self->point->size_bytes;
	if (self->point->sequential && self->point->share == SHARE_STEAL) {
		const uint64_t offset = atomic_fetch_add_explicit(
			self->shared_cursor, size, memory_order_relaxed);
		return offset + size <= self->limit ? offset : NO_OFFSET;
	}
	if (self->point->sequential && self->point->share == SHARE_WEIGHTS) {
		const uint64_t offset = self->region_begin + *cursor;
		*cursor += size;
		return offset + size <= self->region_end ? offset : NO_OFFSET;
	}
	if (self->point->sequential) {
		const uint64_t span =
			store->file_bytes / self->of / size * size;
		const uint64_t offset = (uint64_t)self->ordinal * span +
					*cursor % (span != 0 ? span : size);
		*cursor += size;
		return offset + size <= store->file_bytes ? offset : 0;
	}
	if (self->point->tables_region && store->table_count != 0) {
		const uint32_t table =
			(uint32_t)(random_step(state) % store->table_count);
		const uint64_t bytes =
			store->tables[table].block_count * BENCH_BLOCK_BYTES;
		const uint64_t slots = bytes / size;
		return store->tables[table].byte_offset +
		       (slots != 0 ? random_step(state) % slots : 0) * size;
	}
	const uint64_t slots = store->file_bytes / size;
	return (slots != 0 ? random_step(state) % slots : 0) * size;
}

struct small_ring {
	int descriptor;
	unsigned *sq_head, *sq_tail, *sq_mask, *sq_array, *cq_head, *cq_tail,
		*cq_mask, entries;
	struct io_uring_sqe *sqes;
	struct io_uring_cqe *cqes;
	void *ring, *sqe_memory;
	size_t ring_bytes, sqe_bytes;
	unsigned local_tail;
};

static int small_ring_open(struct small_ring *ring, uint32_t entries, int file,
			   unsigned char *arena, uint64_t arena_bytes)
{
	memset(ring, 0, sizeof(*ring));
	struct io_uring_params parameters;
	memset(&parameters, 0, sizeof(parameters));
	parameters.flags = IORING_SETUP_SINGLE_ISSUER |
			   IORING_SETUP_DEFER_TASKRUN;
	ring->descriptor =
		(int)syscall(__NR_io_uring_setup, entries, &parameters);
	if (ring->descriptor < 0) {
		memset(&parameters, 0, sizeof(parameters));
		ring->descriptor =
			(int)syscall(__NR_io_uring_setup, entries, &parameters);
	}
	if (ring->descriptor < 0 ||
	    (parameters.features & IORING_FEAT_SINGLE_MMAP) == 0)
		return errno != 0 ? errno : ENOSYS;
	const size_t sq_bytes = parameters.sq_off.array +
				parameters.sq_entries * sizeof(unsigned);
	const size_t cq_bytes =
		parameters.cq_off.cqes +
		parameters.cq_entries * sizeof(struct io_uring_cqe);
	ring->ring_bytes = sq_bytes > cq_bytes ? sq_bytes : cq_bytes;
	ring->ring = mmap(NULL, ring->ring_bytes, PROT_READ | PROT_WRITE,
			  MAP_SHARED | MAP_POPULATE, ring->descriptor,
			  IORING_OFF_SQ_RING);
	ring->sqe_bytes = parameters.sq_entries * sizeof(struct io_uring_sqe);
	ring->sqe_memory = mmap(NULL, ring->sqe_bytes, PROT_READ | PROT_WRITE,
				MAP_SHARED | MAP_POPULATE, ring->descriptor,
				IORING_OFF_SQES);
	if (ring->ring == MAP_FAILED || ring->sqe_memory == MAP_FAILED)
		return ENOMEM;
	unsigned char *const base = ring->ring;
	ring->sq_head = (unsigned *)(void *)(base + parameters.sq_off.head);
	ring->sq_tail = (unsigned *)(void *)(base + parameters.sq_off.tail);
	ring->sq_mask =
		(unsigned *)(void *)(base + parameters.sq_off.ring_mask);
	ring->sq_array = (unsigned *)(void *)(base + parameters.sq_off.array);
	ring->cq_head = (unsigned *)(void *)(base + parameters.cq_off.head);
	ring->cq_tail = (unsigned *)(void *)(base + parameters.cq_off.tail);
	ring->cq_mask =
		(unsigned *)(void *)(base + parameters.cq_off.ring_mask);
	ring->cqes =
		(struct io_uring_cqe *)(void *)(base + parameters.cq_off.cqes);
	ring->sqes = ring->sqe_memory;
	ring->entries = parameters.sq_entries;
	ring->local_tail = *ring->sq_tail;
	if (syscall(__NR_io_uring_register, ring->descriptor,
		    IORING_REGISTER_FILES, &file, 1) != 0)
		return errno;
	struct iovec spans[SPANS_MOST];
	uint32_t span_count = 0;
	for (uint64_t begin = 0; begin < arena_bytes && span_count < SPANS_MOST;
	     begin += REGISTERED_SPAN) {
		const uint64_t left = arena_bytes - begin;
		spans[span_count++] = (struct iovec){
			.iov_base = arena + begin,
			.iov_len = (size_t)(left < REGISTERED_SPAN ?
						    left :
						    REGISTERED_SPAN)
		};
	}
	return syscall(__NR_io_uring_register, ring->descriptor,
		       IORING_REGISTER_BUFFERS, spans, span_count) == 0 ?
		       0 :
		       errno;
}

static void small_ring_close(struct small_ring *ring)
{
	if (ring->sqe_memory != NULL && ring->sqe_memory != MAP_FAILED)
		(void)munmap(ring->sqe_memory, ring->sqe_bytes);
	if (ring->ring != NULL && ring->ring != MAP_FAILED)
		(void)munmap(ring->ring, ring->ring_bytes);
	if (ring->descriptor > 0)
		(void)close(ring->descriptor);
}

static void *point_run(void *argument)
{
	struct point_thread *const self = argument;
	const struct point *const point = self->point;
	const uint64_t size = point->size_bytes;
	const uint64_t arena_bytes =
		((uint64_t)point->depth * size + HUGE_PAGE_BYTES - 1u) /
		HUGE_PAGE_BYTES * HUGE_PAGE_BYTES;
	unsigned char *const arena = mmap(NULL, arena_bytes,
					  PROT_READ | PROT_WRITE,
					  MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
	if (arena == MAP_FAILED) {
		self->error = ENOMEM;
		return NULL;
	}
	(void)madvise(arena, arena_bytes, MADV_HUGEPAGE);
	memset(arena, 0, arena_bytes);
	(void)mlock(arena, arena_bytes);
	uint64_t cursor = 0, state = mix_seed(self->copy, self->ordinal);
	const int file = self->store->direct[self->copy];
	uint64_t *const issued = calloc(point->depth, sizeof(*issued));
	struct small_ring ring;
	memset(&ring, 0, sizeof(ring));
	if (issued == NULL)
		self->error = ENOMEM;
	if (self->error == 0 && point->uring)
		self->error = small_ring_open(&ring, point->depth, file, arena,
					      arena_bytes);
	const uint64_t cpu_started = bench_thread_cpu_ns();
	if (self->error == 0 && point->uring) {
		uint32_t in_flight = 0;
		bool stopping = false;
		while (!stopping || in_flight != 0) {
			const uint64_t now = bench_now();
			stopping = stopping || now >= self->deadline_ns ||
				   (self->bytes_target != 0 &&
				    self->bytes + (uint64_t)in_flight * size >=
					    self->bytes_target);
			uint32_t added = 0;
			while (!stopping && in_flight < point->depth) {
				const uint64_t offset =
					next_offset(self, &cursor, &state);
				if (offset == NO_OFFSET) {
					stopping = true;
					break;
				}
				uint32_t slot = 0;
				while (issued[slot] != 0)
					slot++;
				struct io_uring_sqe *const entry =
					&ring.sqes[ring.local_tail &
						   *ring.sq_mask];
				memset(entry, 0, sizeof(*entry));
				entry->opcode = IORING_OP_READ_FIXED;
				entry->flags = IOSQE_FIXED_FILE;
				entry->fd = 0;
				entry->off = offset;
				entry->addr =
					(uint64_t)(uintptr_t)(arena +
							      (uint64_t)slot *
								      size);
				entry->len = (uint32_t)size;
				entry->buf_index =
					(uint16_t)((uint64_t)slot * size /
						   REGISTERED_SPAN);
				entry->user_data = slot;
				ring.sq_array[ring.local_tail & *ring.sq_mask] =
					ring.local_tail & *ring.sq_mask;
				ring.local_tail++;
				issued[slot] = now != 0 ? now : 1u;
				in_flight++;
				added++;
				stopping = self->bytes_target != 0 &&
					   self->bytes + (uint64_t)in_flight *
								   size >=
						   self->bytes_target;
			}
			__atomic_store_n(ring.sq_tail, ring.local_tail,
					 __ATOMIC_RELEASE);
			struct __kernel_timespec timeout = { .tv_sec = 1,
							     .tv_nsec = 0 };
			struct io_uring_getevents_arg wait = {
				.ts = (uint64_t)(uintptr_t)&timeout
			};
			const long entered = syscall(
				__NR_io_uring_enter, ring.descriptor, added,
				in_flight != 0 ? 1u : 0u,
				IORING_ENTER_GETEVENTS | IORING_ENTER_EXT_ARG,
				&wait, sizeof(wait));
			if (entered < 0 && errno != EINTR && errno != ETIME &&
			    errno != EAGAIN && errno != EBUSY) {
				self->error = errno;
				break;
			}
			unsigned head = *ring.cq_head;
			const unsigned tail =
				__atomic_load_n(ring.cq_tail, __ATOMIC_ACQUIRE);
			const uint64_t done_at = bench_now();
			while (head != tail) {
				const struct io_uring_cqe *const entry =
					&ring.cqes[head & *ring.cq_mask];
				head++;
				if (entry->user_data >= point->depth ||
				    issued[entry->user_data] == 0) {
					self->error = EPROTO;
					continue;
				}
				if (entry->res != (int)size)
					self->error = entry->res < 0 ?
							      -entry->res :
							      EIO;
				bench_histogram_add(
					self->latency,
					done_at - issued[entry->user_data]);
				issued[entry->user_data] = 0;
				in_flight--;
				self->bytes += entry->res > 0 ?
						       (uint64_t)entry->res :
						       0u;
				atomic_fetch_add_explicit(
					self->copy_bytes,
					entry->res > 0 ? (uint64_t)entry->res :
							 0u,
					memory_order_relaxed);
				self->reads++;
			}
			__atomic_store_n(ring.cq_head, head, __ATOMIC_RELEASE);
			if (self->error != 0)
				stopping = true;
		}
	} else if (self->error == 0) {
		for (uint32_t read = 0; self->error == 0; read++) {
			if (read % CLOCK_CHECK_READS == 0 &&
			    bench_now() >= self->deadline_ns)
				break;
			if (self->bytes_target != 0 &&
			    self->bytes >= self->bytes_target)
				break;
			const uint64_t offset =
				next_offset(self, &cursor, &state);
			if (offset == NO_OFFSET)
				break;
			const uint64_t started = bench_now();
			const ssize_t amount =
				pread(file, arena, (size_t)size, (off_t)offset);
			if (amount != (ssize_t)size) {
				self->error = amount < 0 ? errno : EIO;
				break;
			}
			bench_histogram_add(self->latency,
					    bench_now() - started);
			self->bytes += size;
			atomic_fetch_add_explicit(self->copy_bytes, size,
						  memory_order_relaxed);
			self->reads++;
		}
	}
	self->cpu_ns = bench_thread_cpu_ns() - cpu_started;
	self->finished_ns = bench_now();
	if (point->uring)
		small_ring_close(&ring);
	free(issued);
	(void)munmap(arena, arena_bytes);
	return NULL;
}

/* Whether an earlier output already holds this point measured ok. */
static bool point_done(const char *done_path, const char *line)
{
	FILE *const input = done_path != NULL ? fopen(done_path, "re") : NULL;
	if (input == NULL)
		return false;
	char key[PLAN_LINE_BYTES + 64u], row[PLAN_LINE_BYTES * 8u];
	(void)snprintf(key, sizeof(key), "\"point\":\"%s\",\"status\":\"ok\"",
		       line);
	bool found = false;
	while (!found && fgets(row, sizeof(row), input) != NULL)
		found = strstr(row, key) != NULL;
	(void)fclose(input);
	return found;
}

int bench_characterise(const struct bench_store *store, const char *plan_path,
		       FILE *output, const char *done_path,
		       uint64_t deadline_ns)
{
	FILE *const plan = fopen(plan_path, "re");
	if (plan == NULL)
		return errno;
	char line[PLAN_LINE_BYTES];
	int status = 0;
	uint64_t last_point_ns = 0;
	while (status == 0 && fgets(line, sizeof(line), plan) != NULL) {
		if (line[0] == '#' || line[0] == '\n')
			continue;
		line[strcspn(line, "\r\n")] = '\0';
		struct point point;
		if (point_done(done_path, line))
			continue;
		if (!point_parse(line, &point)) {
			(void)fprintf(
				output,
				"{\"schema\":1,\"kind\":\"characterise\",\"point\":\"%s\",\"status\":\"refused\","
				"\"error\":\"plan line\"}\n",
				line);
			continue;
		}
		uint32_t copies = 0;
		for (uint32_t copy = 0; copy < store->copy_count; copy++)
			copies += (point.copy_mask >> copy & 1u) != 0 ? 1u : 0u;
		const uint64_t in_flight = (uint64_t)copies * point.threads *
					   point.depth * point.size_bytes;
		if (copies == 0 || copies * point.threads > THREADS_MOST ||
		    in_flight > IN_FLIGHT_BYTES_MOST) {
			(void)fprintf(
				output,
				"{\"schema\":1,\"kind\":\"characterise\",\"point\":\"%s\",\"status\":\"refused\","
				"\"error\":\"threads or bytes in flight above the limit\"}\n",
				line);
			continue;
		}
		const uint64_t now = bench_now();
		if (now + last_point_ns + point.seconds * BENCH_NS_PER_SECOND >
		    deadline_ns) {
			status = EXIT_CHECKPOINT;
			break;
		}
		struct point_thread *const threads = calloc(
			(size_t)copies * point.threads, sizeof(*threads));
		struct bench_histogram *const latency =
			calloc(1, sizeof(*latency));
		_Atomic uint64_t copy_bytes[BENCH_COPIES_MOST];
		for (uint32_t copy = 0; copy < BENCH_COPIES_MOST; copy++)
			atomic_init(&copy_bytes[copy], 0);
		if (threads == NULL || latency == NULL) {
			free(threads);
			free(latency);
			return ENOMEM;
		}
		/* a shared load (share=steal or weights): its bytes, steal's cursor, the copies' regions of weights */
		const uint64_t size = point.size_bytes;
		const uint64_t whole = store->file_bytes / size * size;
		const uint64_t load =
			point.total_bytes != 0 && point.total_bytes < whole ?
				point.total_bytes / size * size :
				whole;
		_Atomic uint64_t shared_cursor;
		atomic_init(&shared_cursor, 0);
		uint64_t weight_sum = 0, region_at = 0;
		for (uint32_t copy = 0; copy < store->copy_count; copy++)
			weight_sum += (point.copy_mask >> copy & 1u) != 0 ?
					      point.weights[copy] :
					      0u;
		const uint64_t started = bench_now();
		uint32_t launched = 0, listed = 0;
		for (uint32_t copy = 0; copy < store->copy_count; copy++) {
			if ((point.copy_mask >> copy & 1u) == 0)
				continue;
			listed++;
			const uint64_t pieces =
				point.share != SHARE_WEIGHTS ||
						weight_sum == 0 ?
					0u :
				listed == copies ?
					(load - region_at) / size :
					load / size * point.weights[copy] /
						weight_sum;
			const uint64_t region_begin = region_at;
			region_at += pieces * size;
			for (uint32_t ordinal = 0; ordinal < point.threads;
			     ordinal++) {
				struct point_thread *const self =
					&threads[launched];
				*self = (struct point_thread){
					.store = store,
					.point = &point,
					.copy = copy,
					.ordinal = ordinal,
					.of = point.threads,
					.deadline_ns =
						started +
						point.seconds *
							BENCH_NS_PER_SECOND,
					.bytes_target =
						point.share == SHARE_COPY ?
							point.bytes_per_copy /
								point.threads :
							0u,
					.copy_bytes = &copy_bytes[copy],
					.shared_cursor = &shared_cursor,
					.limit = load,
					.region_begin = region_begin +
							pieces * ordinal /
								point.threads *
								size,
					.region_end = region_begin +
						      pieces * (ordinal + 1u) /
							      point.threads *
							      size,
					.latency = latency
				};
				if (pthread_create(&self->thread, NULL,
						   point_run, self) == 0)
					launched++;
			}
		}
		uint64_t bytes = 0, reads = 0, cpu_ns = 0;
		int error = 0;
		for (uint32_t index = 0; index < launched; index++) {
			(void)pthread_join(threads[index].thread, NULL);
			bytes += threads[index].bytes;
			reads += threads[index].reads;
			cpu_ns += threads[index].cpu_ns;
			error = error != 0 ? error : threads[index].error;
		}
		const uint64_t elapsed = bench_now() - started;
		last_point_ns = elapsed;
		(void)fprintf(
			output,
			"{\"schema\":1,\"kind\":\"characterise\",\"point\":\"%s\",\"status\":\"%s\",\"error\":\"%s\","
			"\"mode\":\"%s\",\"engine\":\"%s\",\"size_bytes\":%u,\"depth\":%u,\"threads_per_copy\":%u,"
			"\"bytes\":%" PRIu64 ",\"reads\":%" PRIu64
			",\"seconds\":%.4f,\"gb_per_second\":%.3f,"
			"\"reads_per_second\":%.1f,\"p50_us\":%.1f,\"p99_us\":%.1f,\"p999_us\":%.1f,"
			"\"cpu_seconds\":%.3f,\"share\":\"%s\",\"copies\":[",
			line,
			error == 0 && launched == copies * point.threads ?
				"ok" :
				"failed",
			error != 0 && strerrorname_np(error) != NULL ?
				strerrorname_np(error) :
				"",
			point.sequential ? "seq" : "rand",
			point.uring ? "uring" : "pread", point.size_bytes,
			point.depth, point.threads, bytes, reads,
			(double)elapsed / (double)BENCH_NS_PER_SECOND,
			elapsed != 0 ? (double)bytes / (double)elapsed : 0.0,
			elapsed != 0 ?
				(double)reads * (double)BENCH_NS_PER_SECOND /
					(double)elapsed :
				0.0,
			(double)bench_histogram_percentile(latency,
							   PER_MILLE_P50) /
				(double)BENCH_NS_PER_MICROSECOND,
			(double)bench_histogram_percentile(latency,
							   PER_MILLE_P99) /
				(double)BENCH_NS_PER_MICROSECOND,
			(double)bench_histogram_percentile(latency,
							   PER_MILLE_P999) /
				(double)BENCH_NS_PER_MICROSECOND,
			(double)cpu_ns / (double)BENCH_NS_PER_SECOND,
			share_names[point.share]);
		uint32_t written = 0;
		for (uint32_t copy = 0; copy < store->copy_count; copy++) {
			if ((point.copy_mask >> copy & 1u) == 0)
				continue;
			const uint64_t delivered =
				atomic_load(&copy_bytes[copy]);
			uint64_t finished = started;
			for (uint32_t index = 0; index < launched; index++)
				if (threads[index].copy == copy &&
				    threads[index].finished_ns > finished)
					finished = threads[index].finished_ns;
			(void)fprintf(
				output,
				"%s{\"copy\":%u,\"device\":\"%s\",\"bytes\":%" PRIu64
				",\"gb_per_second\":%.3f,\"seconds\":%.4f}",
				written++ == 0 ? "" : ",", copy,
				store->device_names[copy], delivered,
				elapsed != 0 ?
					(double)delivered / (double)elapsed :
					0.0,
				(double)(finished - started) /
					(double)BENCH_NS_PER_SECOND);
		}
		(void)fputs("]}\n", output);
		(void)fflush(output);
		(void)fprintf(
			stderr,
			"characterise %-70s %8.3f GB/s %10.0f reads/s p99 %.1f us %s\n",
			line,
			elapsed != 0 ? (double)bytes / (double)elapsed : 0.0,
			elapsed != 0 ?
				(double)reads * (double)BENCH_NS_PER_SECOND /
					(double)elapsed :
				0.0,
			(double)bench_histogram_percentile(latency,
							   PER_MILLE_P99) /
				(double)BENCH_NS_PER_MICROSECOND,
			error == 0 ? "" : strerrorname_np(error));
		free(threads);
		free(latency);
	}
	(void)fclose(plan);
	return status;
}
