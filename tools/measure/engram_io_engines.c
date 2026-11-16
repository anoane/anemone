/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * engram_io_engines.c: the bench's read machinery (engram_io_bench.h).
 *
 * A step's blocks become logical reads; each gets an attempt (a 4 KiB slot of a pinned arena) on the copy the arm's
 * policy picks, routed to the owner of that copy. An owner (the step thread itself, or a dedicated I/O thread) owns
 * engines: io_uring rings, a Linux AIO context, a thread pool of synchronous calls, inline reads, or a mapping. An
 * owner submits, reaps, checks every block against its pack digest, retries a failed or corrupt block on another
 * copy, and hedges a slow one. The first verified attempt of a read wins; the rest are freed when they complete.
 * Reads live in epochs so that a late loser never touches a read the next step reuses.
 */
#include "engram_io_bench.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <linux/fiemap.h>
#include <linux/fs.h>
#include <linux/futex.h>
#include <linux/io_uring.h>
#include <linux/ioprio.h>
#include <linux/nvme_ioctl.h>
#include <linux/time_types.h>
#include <poll.h>
#include <sched.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/sysmacros.h>
#include <time.h>
#include <unistd.h>

#include "anemone/pack_digest.h"

#define SUBMIT_BATCH_MOST 256u
#define REAP_BATCH_MOST 256u
#define ATTEMPTS_PER_COPY \
	2u /* per block and copy: one retry of a transient error or torn read */
#define SLOT_FACTOR \
	3u /* attempt slots per read of a step: the read, a hedge, a retry */
#define LATENCY_EWMA_SHIFT 3u
#define REGISTERED_BUFFER_SPAN \
	(1ull << 30u) /* the kernel's limit for one registered buffer */
#define HUGE_PAGE_BYTES (2ull << 20u)
#define IDLE_SLICE_NS (1ull * BENCH_NS_PER_MILLISECOND)
#define POLL_SLICE_MILLISECONDS_MOST 1000
#define CQE_BIG_BYTES 32u
#define CQE_PLAIN_BYTES 16u
#define SQE_BIG_BYTES 128u
#define SQE_PLAIN_BYTES 64u
#define NVME_READ_OPCODE 0x02u
#define NVME_LOGICAL_BLOCK_BYTES_LEAST 512u
#define NVME_HIGH_SHIFT 32u
#define RANDOM_MULTIPLIER 0x2545F4914F6CDD1Dull
#define RANDOM_SHIFT_A 12u
#define RANDOM_SHIFT_B 25u
#define RANDOM_SHIFT_C 27u
#define GENERATION_SHIFT 32u
#define FIEMAP_REFUSED_FLAGS                                     \
	(FIEMAP_EXTENT_UNKNOWN | FIEMAP_EXTENT_DELALLOC |        \
	 FIEMAP_EXTENT_ENCODED | FIEMAP_EXTENT_DATA_ENCRYPTED |  \
	 FIEMAP_EXTENT_NOT_ALIGNED | FIEMAP_EXTENT_DATA_INLINE | \
	 FIEMAP_EXTENT_DATA_TAIL | FIEMAP_EXTENT_UNWRITTEN)
#define SECTOR_BYTES 512u
#define HEDGE_REFRESH_NS BENCH_NS_PER_MILLISECOND
#define HEDGE_PER_MILLE 990u
#define HEDGE_PERCENT_UNIT 100u

static void cpu_relax(void)
{
#if defined(__x86_64__)
	__builtin_ia32_pause();
#endif
}

uint64_t bench_now(void)
{
	struct timespec time;
	return clock_gettime(CLOCK_MONOTONIC, &time) == 0 ?
		       (uint64_t)time.tv_sec * BENCH_NS_PER_SECOND +
			       (uint64_t)time.tv_nsec :
		       UINT64_MAX;
}

uint64_t bench_thread_cpu_ns(void)
{
	struct timespec time;
	return clock_gettime(CLOCK_THREAD_CPUTIME_ID, &time) == 0 ?
		       (uint64_t)time.tv_sec * BENCH_NS_PER_SECOND +
			       (uint64_t)time.tv_nsec :
		       0;
}

static void futex_wait(_Atomic uint32_t *word, uint32_t expected,
		       uint64_t timeout_ns)
{
	const struct timespec timeout = {
		.tv_sec = (time_t)(timeout_ns / BENCH_NS_PER_SECOND),
		.tv_nsec = (long)(timeout_ns % BENCH_NS_PER_SECOND)
	};
	(void)syscall(SYS_futex, (void *)(uintptr_t)word, FUTEX_WAIT_PRIVATE,
		      expected, &timeout, NULL, 0);
}

static void futex_wake(_Atomic uint32_t *word)
{
	(void)syscall(SYS_futex, (void *)(uintptr_t)word, FUTEX_WAKE_PRIVATE,
		      INT32_MAX, NULL, NULL, 0);
}

/* Log-linear: values below 2^SUB_BITS exactly, then 2^SUB_BITS sub-buckets per power of two. */
static uint32_t histogram_bucket(uint64_t value)
{
	if (value < (1u << BENCH_HISTOGRAM_SUB_BITS))
		return (uint32_t)value;
	const uint32_t power = 63u - (uint32_t)__builtin_clzll(value);
	const uint32_t sub =
		(uint32_t)(value >> (power - BENCH_HISTOGRAM_SUB_BITS)) &
		((1u << BENCH_HISTOGRAM_SUB_BITS) - 1u);
	const uint32_t bucket = ((power - BENCH_HISTOGRAM_SUB_BITS + 1u)
				 << BENCH_HISTOGRAM_SUB_BITS) +
				sub;
	return bucket < BENCH_HISTOGRAM_BUCKETS ? bucket :
						  BENCH_HISTOGRAM_BUCKETS - 1u;
}

static uint64_t histogram_value(uint32_t bucket)
{
	if (bucket < (1u << BENCH_HISTOGRAM_SUB_BITS))
		return bucket;
	const uint32_t power = (bucket >> BENCH_HISTOGRAM_SUB_BITS) +
			       BENCH_HISTOGRAM_SUB_BITS - 1u;
	const uint64_t sub = bucket & ((1u << BENCH_HISTOGRAM_SUB_BITS) - 1u);
	return (1ull << power) + (sub << (power - BENCH_HISTOGRAM_SUB_BITS));
}

void bench_histogram_add(struct bench_histogram *histogram, uint64_t value_ns)
{
	atomic_fetch_add_explicit(
		&histogram->buckets[histogram_bucket(value_ns)], 1,
		memory_order_relaxed);
}

/* The lower bound of the bucket holding the per-mille rank (nearest rank). */
uint64_t bench_histogram_percentile(const struct bench_histogram *histogram,
				    uint32_t per_mille)
{
	uint64_t total = 0;
	for (uint32_t bucket = 0; bucket < BENCH_HISTOGRAM_BUCKETS; bucket++)
		total += atomic_load_explicit(&histogram->buckets[bucket],
					      memory_order_relaxed);
	if (total == 0)
		return 0;
	const uint64_t rank = (total * per_mille + 999u) / 1000u;
	uint64_t seen = 0;
	for (uint32_t bucket = 0; bucket < BENCH_HISTOGRAM_BUCKETS; bucket++) {
		seen += atomic_load_explicit(&histogram->buckets[bucket],
					     memory_order_relaxed);
		if (seen >= rank && seen != 0)
			return histogram_value(bucket);
	}
	return histogram_value(BENCH_HISTOGRAM_BUCKETS - 1u);
}

/* A setting the system refused (privilege, kernel): recorded once per kind, the arm goes on without it. */
void bench_refusal(struct bench_run *run, const char *what, int error)
{
	(void)pthread_mutex_lock(&run->refusal_lock);
	if (strstr(run->refusals, what) == NULL) {
		const size_t used = strlen(run->refusals);
		(void)snprintf(run->refusals + used,
			       sizeof(run->refusals) - used, "%s%s:%s",
			       used != 0 ? "," : "", what,
			       strerrorname_np(error) != NULL ?
				       strerrorname_np(error) :
				       "unknown");
	}
	(void)pthread_mutex_unlock(&run->refusal_lock);
}

void bench_fail(struct bench_run *run, const char *reason, int error)
{
	int expected = 0;
	if (atomic_compare_exchange_strong(&run->error, &expected,
					   error != 0 ? error : EIO))
		(void)snprintf(run->reason, sizeof(run->reason), "%s", reason);
}

void bench_thread_attributes(struct bench_run *run, int32_t cpu)
{
	const struct arm *const arm = run->arm;
	if (cpu >= 0) {
		cpu_set_t set;
		CPU_ZERO(&set);
		CPU_SET((size_t)cpu, &set);
		const int error = pthread_setaffinity_np(pthread_self(),
							 sizeof(set), &set);
		if (error != 0)
			bench_refusal(run, "affinity", error);
	}
	if (arm->nice != 0 &&
	    setpriority(PRIO_PROCESS, (id_t)gettid(), arm->nice) != 0)
		bench_refusal(run, arm->nice < 0 ? "nice_negative" : "nice",
			      errno);
	if (arm->scheduling != SCHEDULE_KEEP) {
		static const int policies[SCHEDULE_KINDS] = {
			SCHED_OTHER, SCHED_OTHER, SCHED_BATCH,
			SCHED_IDLE,  SCHED_FIFO,  SCHED_RR
		};
		const bool realtime = arm->scheduling == SCHEDULE_FIFO ||
				      arm->scheduling == SCHEDULE_RR;
		const struct sched_param parameter = {
			.sched_priority = realtime ? (int)arm->rt_priority : 0
		};
		if (sched_setscheduler(0, policies[arm->scheduling],
				       &parameter) != 0)
			bench_refusal(run,
				      realtime ? "sched_realtime" : "sched",
				      errno);
	}
	if (arm->io_priority != IO_PRIORITY_KEEP) {
		static const int classes[IO_PRIORITY_KINDS] = {
			0, IOPRIO_CLASS_RT, IOPRIO_CLASS_BE, IOPRIO_CLASS_IDLE
		};
		if (syscall(SYS_ioprio_set, IOPRIO_WHO_PROCESS, 0,
			    IOPRIO_PRIO_VALUE(classes[arm->io_priority],
					      (int)arm->io_priority_level)) !=
		    0)
			bench_refusal(run,
				      arm->io_priority == IO_PRIORITY_RT ?
					      "ioprio_rt" :
					      "ioprio",
				      errno);
	}
}

uint64_t bench_copy_offset(const struct bench_store *store, uint32_t table,
			   uint64_t block)
{
	return store->tables[table].byte_offset + block * BENCH_BLOCK_BYTES;
}

bool bench_block_verified(const struct bench_store *store, uint32_t table,
			  uint64_t block, const unsigned char *bytes)
{
	return pack_digest_of_piece(bytes, BENCH_BLOCK_BYTES, 0) ==
	       store->tables[table].block_digests[block];
}

/* ---- slots, reads, routing --------------------------------------------------------------------------------- */

static struct attempt *slot_take(struct bench_run *run)
{
	(void)pthread_mutex_lock(&run->free_lock);
	struct attempt *const attempt =
		run->free_count != 0 ?
			&run->attempts[run->free_slots[--run->free_count]] :
			NULL;
	(void)pthread_mutex_unlock(&run->free_lock);
	if (attempt == NULL)
		bench_fail(run, "attempt slots exhausted", ENOBUFS);
	return attempt;
}

static void slot_give(struct bench_run *run, struct attempt *attempt)
{
	attempt->read = NULL;
	atomic_fetch_add_explicit(&attempt->generation, 1,
				  memory_order_release);
	(void)pthread_mutex_lock(&run->free_lock);
	run->free_slots[run->free_count++] = attempt->index;
	(void)pthread_mutex_unlock(&run->free_lock);
}

static uint64_t random_next(struct bench_run *run)
{
	uint64_t state =
		atomic_load_explicit(&run->random_state, memory_order_relaxed);
	state ^= state >> RANDOM_SHIFT_A;
	state ^= state << RANDOM_SHIFT_B;
	state ^= state >> RANDOM_SHIFT_C;
	atomic_store_explicit(&run->random_state, state, memory_order_relaxed);
	return state * RANDOM_MULTIPLIER;
}

static uint32_t nth_bit(uint32_t mask, uint32_t rank)
{
	for (uint32_t copy = 0; copy < BENCH_COPIES_MOST; copy++)
		if ((mask >> copy & 1u) != 0 && rank-- == 0)
			return copy;
	return 0;
}

/* The copy a new attempt reads, by the arm's policy, among the arm's copies less the excluded ones (all copies when
 * that leaves none: a fallback may always use any copy). projected: this batch's choices so far, per copy. */
static uint32_t copy_choose(struct bench_run *run,
			    const struct logical_read *read, uint32_t exclude,
			    uint32_t *projected)
{
	const struct arm *const arm = run->arm;
	const uint32_t all = (1u << run->store->copy_count) - 1u;
	uint32_t mask = arm->copy_mask & all & ~exclude;
	mask = mask != 0 ? mask : all & ~exclude;
	mask = mask != 0 ? mask : all;
	const uint32_t count = (uint32_t)__builtin_popcount(mask);
	uint32_t chosen = nth_bit(mask, 0);
	switch (arm->copy) {
	case COPY_FIRST:
	case COPY_KINDS:
		break;
	case COPY_TICK: {
		uint32_t total = 0;
		for (uint32_t copy = 0; copy < BENCH_COPIES_MOST; copy++)
			total += (mask >> copy & 1u) != 0 ? arm->weights[copy] :
							    0;
		if (total == 0)
			break;
		uint64_t tick = atomic_fetch_add_explicit(
					&run->tick, 1, memory_order_relaxed) %
				total;
		for (uint32_t copy = 0; copy < BENCH_COPIES_MOST; copy++) {
			const uint32_t weight = (mask >> copy & 1u) != 0 ?
							arm->weights[copy] :
							0;
			if (tick < weight) {
				chosen = copy;
				break;
			}
			tick -= weight;
		}
		break;
	}
	case COPY_BLOCK:
		chosen = nth_bit(
			mask, (uint32_t)((read->block + read->table) % count));
		break;
	case COPY_JSQ:
	case COPY_RATE: {
		const uint32_t start =
			(uint32_t)(atomic_fetch_add_explicit(
					   &run->tick, 1,
					   memory_order_relaxed) %
				   count);
		double best = 0.0;
		for (uint32_t rank = 0; rank < count; rank++) {
			const uint32_t copy =
				nth_bit(mask, (start + rank) % count);
			const double load = (double)atomic_load_explicit(
						    &run->in_flight[copy],
						    memory_order_relaxed) +
					    (projected != NULL ?
						     (double)projected[copy] :
						     0.0) +
					    1.0;
			const uint64_t latency = atomic_load_explicit(
				&run->latency_ewma_ns[copy],
				memory_order_relaxed);
			const double cost =
				arm->copy == COPY_RATE ?
					load * (double)(latency != 0 ? latency :
								       1u) :
					load;
			if (rank == 0 || cost < best) {
				best = cost;
				chosen = copy;
			}
		}
		break;
	}
	case COPY_P2C: {
		const uint32_t first =
			nth_bit(mask, (uint32_t)(random_next(run) % count));
		const uint32_t second =
			nth_bit(mask, (uint32_t)(random_next(run) % count));
		const uint32_t first_load =
			atomic_load_explicit(&run->in_flight[first],
					     memory_order_relaxed) +
			(projected != NULL ? projected[first] : 0u);
		const uint32_t second_load =
			atomic_load_explicit(&run->in_flight[second],
					     memory_order_relaxed) +
			(projected != NULL ? projected[second] : 0u);
		chosen = second_load < first_load ? second : first;
		break;
	}
	}
	if (projected != NULL)
		projected[chosen]++;
	return chosen;
}

static struct read_epoch *epoch_of(struct bench_run *run,
				   const struct logical_read *read)
{
	return &run->epochs[read->epoch];
}

static struct attempt *attempt_for(struct bench_run *run,
				   struct logical_read *read, uint32_t copy)
{
	struct attempt *const attempt = slot_take(run);
	if (attempt == NULL)
		return NULL;
	attempt->read = read;
	attempt->copy = copy;
	attempt->offset =
		bench_copy_offset(run->store, read->table, read->block);
	attempt->result = 0;
	atomic_fetch_add(&read->live, 1);
	atomic_fetch_add(&epoch_of(run, read)->live, 1);
	return attempt;
}

/* Hands attempts to the owners of their copies; copies with several owners alternate among them. */
static void route(struct bench_run *run, struct attempt *const *attempts,
		  uint32_t count)
{
	for (uint32_t index = 0; index < count; index++) {
		struct attempt *const attempt = attempts[index];
		const uint32_t sharers = run->owners_of_copy[attempt->copy];
		const uint32_t turn =
			sharers > 1u ?
				(uint32_t)(atomic_fetch_add_explicit(
						   &run->route_turn
							    [attempt->copy],
						   1, memory_order_relaxed) %
					   sharers) :
				0u;
		struct owner *const owner =
			&run->owners[run->owner_of_copy[attempt->copy] +
				     turn * run->store->copy_count];
		(void)pthread_mutex_lock(&owner->inbox_lock);
		owner->inbox[owner->inbox_count++] = attempt->index;
		(void)pthread_mutex_unlock(&owner->inbox_lock);
		if (atomic_exchange(&owner->inbox_signal, 1) == 0 &&
		    owner->threaded)
			futex_wake(&owner->inbox_signal);
	}
}

static void read_finish(struct bench_run *run, struct logical_read *read,
			uint32_t state)
{
	atomic_store(&read->state, state);
	if (atomic_fetch_sub(&run->pending[read->table], 1) == 1 &&
	    run->arm->wait != WAIT_SPIN)
		futex_wake(&run->pending[read->table]);
}

/* In the owner's thread: one completed attempt. */
static void attempt_complete(struct bench_run *run, struct attempt *attempt)
{
	struct logical_read *const read = attempt->read;
	const uint32_t copy = attempt->copy;
	const uint64_t latency = bench_now() - attempt->issued_ns;
	atomic_fetch_sub_explicit(&run->in_flight[copy], 1,
				  memory_order_relaxed);
	atomic_fetch_add_explicit(&run->copy_reads[copy], 1,
				  memory_order_relaxed);
	bench_histogram_add(&run->copy_latency[copy], latency);
	if (run->arm->hedge_percent != 0 &&
	    (run->arm->hedge_step_most == 0 ||
	     epoch_of(run, read)->count <= run->arm->hedge_step_most))
		bench_histogram_add(&run->small_step_latency[copy], latency);
	const uint64_t average = atomic_load_explicit(
		&run->latency_ewma_ns[copy], memory_order_relaxed);
	atomic_store_explicit(
		&run->latency_ewma_ns[copy],
		average == 0 ? latency :
			       average - (average >> LATENCY_EWMA_SHIFT) +
				       (latency >> LATENCY_EWMA_SHIFT),
		memory_order_relaxed);
	const bool good = attempt->result == (int64_t)BENCH_BLOCK_BYTES &&
			  bench_block_verified(run->store, read->table,
					       read->block, attempt->buffer);
	const uint32_t live = atomic_fetch_sub(&read->live, 1) - 1u;
	struct read_epoch *const epoch = epoch_of(run, read);
	if (good) {
		struct attempt *expected = NULL;
		if (atomic_compare_exchange_strong(&read->winner, &expected,
						   attempt)) {
			if (atomic_load(&read->hedged) != 0 && attempt->hedge)
				atomic_fetch_add_explicit(
					&run->counters.hedge_wins, 1,
					memory_order_relaxed);
			read_finish(run, read, READ_DONE);
		} else {
			slot_give(run, attempt);
		}
		atomic_fetch_sub(&epoch->live, 1);
		return;
	}
	atomic_fetch_add_explicit(&run->counters.failures, 1,
				  memory_order_relaxed);
	const uint32_t tried = atomic_fetch_or(&read->tried, 1u << copy) |
			       1u << copy;
	if (atomic_load(&read->state) != READ_WAITING || live != 0) {
		slot_give(run, attempt);
		atomic_fetch_sub(&epoch->live, 1);
		return;
	}
	const uint32_t tries = atomic_fetch_add(&read->tries, 1) + 1u;
	if (tries >= run->store->copy_count * ATTEMPTS_PER_COPY) {
		/* a read error names its errno (EOPNOTSUPP: polled I/O without poll queues); a digest mismatch is
		 * EBADMSG */
		bench_fail(
			run,
			attempt->result == -EINVAL ||
					attempt->result == -EOPNOTSUPP ?
				"a read was refused on every copy" BENCH_UNSUPPORTED :
				"a block failed its digest or read on every copy",
			attempt->result < 0 ? (int)-attempt->result : EBADMSG);
		read_finish(run, read, READ_FAILED);
		slot_give(run, attempt);
		atomic_fetch_sub(&epoch->live, 1);
		return;
	}
	const uint32_t all = (1u << run->store->copy_count) - 1u;
	const uint32_t next = (tried & all) == all ?
				      (copy + 1u) % run->store->copy_count :
				      (uint32_t)__builtin_ctz(~tried & all);
	struct attempt *const retry = attempt_for(run, read, next);
	slot_give(run, attempt);
	atomic_fetch_sub(&epoch->live, 1);
	if (retry == NULL) {
		read_finish(run, read, READ_FAILED);
		return;
	}
	retry->hedge = false;
	atomic_fetch_add_explicit(&run->counters.retries, 1,
				  memory_order_relaxed);
	route(run, &retry, 1);
}

static bool hedging(const struct arm *arm)
{
	return arm->hedge_us != 0 || arm->hedge_percent != 0;
}

/* Each copy's hedge deadline, refreshed at most every HEDGE_REFRESH_NS from the running p99 of small steps' reads.
 * Without samples yet, the floor alone (or no hedge at all when there is no floor). */
static void hedge_deadlines(struct bench_run *run, uint64_t now)
{
	const uint64_t refreshed = atomic_load_explicit(
		&run->hedge_refreshed_ns, memory_order_relaxed);
	if (now - refreshed < HEDGE_REFRESH_NS ||
	    !atomic_compare_exchange_strong(&run->hedge_refreshed_ns,
					    &(uint64_t){ refreshed }, now))
		return;
	const uint64_t floor =
		(uint64_t)run->arm->hedge_us * BENCH_NS_PER_MICROSECOND;
	for (uint32_t copy = 0; copy < run->store->copy_count; copy++) {
		const uint64_t p99 =
			run->arm->hedge_percent != 0 ?
				bench_histogram_percentile(
					&run->small_step_latency[copy],
					HEDGE_PER_MILLE) :
				0;
		const uint64_t relative =
			p99 * run->arm->hedge_percent / HEDGE_PERCENT_UNIT;
		const uint64_t deadline = relative > floor ? relative : floor;
		atomic_store_explicit(&run->hedge_deadline_ns[copy],
				      deadline != 0 ? deadline : UINT64_MAX,
				      memory_order_relaxed);
	}
}

static void hedge_check(struct owner *owner)
{
	struct bench_run *const run = owner->run;
	const uint64_t now = bench_now();
	hedge_deadlines(run, now);
	while (owner->issued_head != owner->issued_tail) {
		const uint64_t entry = owner->issued[owner->issued_head %
						     owner->issued_capacity];
		struct attempt *const attempt = &run->attempts[(uint32_t)entry];
		const uint32_t generation =
			(uint32_t)(entry >> GENERATION_SHIFT);
		struct logical_read *const read = attempt->read;
		if (atomic_load_explicit(&attempt->generation,
					 memory_order_acquire) != generation ||
		    read == NULL || atomic_load(&read->state) != READ_WAITING) {
			owner->issued_head++;
			continue;
		}
		const uint64_t deadline = atomic_load_explicit(
			&run->hedge_deadline_ns[attempt->copy],
			memory_order_relaxed);
		const bool small_step = run->arm->hedge_step_most == 0 ||
					epoch_of(run, read)->count <=
						run->arm->hedge_step_most;
		if (small_step && now - attempt->issued_ns < deadline)
			break;
		owner->issued_head++;
		if (!small_step)
			continue; /* prefill: never hedged */
		uint32_t unhedged = 0;
		if (!atomic_compare_exchange_strong(&read->hedged, &unhedged,
						    1))
			continue;
		const uint32_t copy =
			copy_choose(run, read, 1u << attempt->copy, NULL);
		if (copy == attempt->copy)
			continue;
		struct attempt *const hedge = attempt_for(run, read, copy);
		if (hedge == NULL)
			return;
		hedge->hedge = true;
		atomic_fetch_add_explicit(&run->counters.hedges, 1,
					  memory_order_relaxed);
		route(run, &hedge, 1);
	}
}

/* Submits, reaps, verifies and hedges; blocks (to the deadline) only when nothing was done and block is set.
 * True when anything happened. */
static bool owner_progress(struct owner *owner, bool block,
			   uint64_t deadline_ns)
{
	struct bench_run *const run = owner->run;
	bool worked = false;
	if (atomic_load_explicit(&owner->inbox_signal, memory_order_acquire) !=
	    0) {
		atomic_store(&owner->inbox_signal, 0);
		(void)pthread_mutex_lock(&owner->inbox_lock);
		for (uint32_t index = 0; index < owner->inbox_count; index++) {
			struct engine *const engine =
				&owner->engines
					 [owner->engine_of_copy
						  [run->attempts
							   [owner->inbox[index]]
								   .copy]];
			engine->pending[engine->pending_tail++ %
					engine->pending_capacity] =
				owner->inbox[index];
		}
		owner->inbox_count = 0;
		(void)pthread_mutex_unlock(&owner->inbox_lock);
		worked = true;
	}
	for (uint32_t index = 0; index < owner->engine_count; index++) {
		struct engine *const engine = &owner->engines[index];
		while (engine->pending_head != engine->pending_tail &&
		       engine->error == 0) {
			uint32_t room = run->arm->inflight_most != 0 ?
						run->arm->inflight_most :
						UINT32_MAX;
			room = engine->in_flight < room ?
				       room - engine->in_flight :
				       0;
			uint32_t count =
				engine->pending_tail - engine->pending_head;
			count = count < room ? count : room;
			count = count < SUBMIT_BATCH_MOST ? count :
							    SUBMIT_BATCH_MOST;
			if (count == 0)
				break;
			struct attempt *batch[SUBMIT_BATCH_MOST];
			const uint64_t now = bench_now();
			for (uint32_t slot = 0; slot < count; slot++) {
				batch[slot] =
					&run->attempts
						 [engine->pending
							  [(engine->pending_head +
							    slot) %
							   engine->pending_capacity]];
				batch[slot]->issued_ns = now;
				atomic_fetch_add_explicit(
					&run->in_flight[batch[slot]->copy], 1,
					memory_order_relaxed);
			}
			const uint32_t accepted = engine->operations->submit(
				engine, batch, count);
			for (uint32_t slot = accepted; slot < count; slot++)
				atomic_fetch_sub_explicit(
					&run->in_flight[batch[slot]->copy], 1,
					memory_order_relaxed);
			for (uint32_t slot = 0;
			     slot < accepted && hedging(run->arm); slot++) {
				if (owner->issued_tail - owner->issued_head ==
				    owner->issued_capacity)
					owner->issued_head++; /* full: the oldest goes unhedged */
				owner->issued[owner->issued_tail++ %
					      owner->issued_capacity] =
					(uint64_t)batch[slot]->index |
					(uint64_t)atomic_load(
						&batch[slot]->generation)
						<< GENERATION_SHIFT;
			}
			atomic_fetch_add_explicit(&run->counters.reads,
						  accepted,
						  memory_order_relaxed);
			engine->pending_head += accepted;
			worked = worked || accepted != 0;
			if (engine->error != 0)
				bench_fail(run, engine->reason, engine->error);
			if (accepted < count)
				break;
		}
	}
	struct attempt *completed[REAP_BATCH_MOST];
	uint32_t waiting_engines = 0, waiting_engine = 0;
	for (uint32_t index = 0; index < owner->engine_count; index++) {
		struct engine *const engine = &owner->engines[index];
		if (engine->in_flight == 0)
			continue;
		const uint32_t count = engine->operations->reap(
			engine, completed, REAP_BATCH_MOST, false, 0);
		for (uint32_t done = 0; done < count; done++)
			attempt_complete(run, completed[done]);
		worked = worked || count != 0;
		waiting_engines += engine->in_flight != 0 ? 1u : 0u;
		waiting_engine = engine->in_flight != 0 ? index :
							  waiting_engine;
	}
	if (!worked && block && waiting_engines != 0) {
		if (waiting_engines > 1u && run->arm->wait == WAIT_POLL) {
			struct pollfd polls[BENCH_COPIES_MOST];
			uint32_t poll_count = 0;
			for (uint32_t index = 0; index < owner->engine_count;
			     index++)
				if (owner->engines[index].in_flight != 0 &&
				    owner->engines[index]
						    .operations->poll_descriptor(
							    &owner->engines
								     [index]) >=
					    0)
					polls[poll_count++] = (struct pollfd){
						.fd = owner->engines[index]
							      .operations
							      ->poll_descriptor(
								      &owner->engines
									       [index]),
						.events = POLLIN
					};
			const uint64_t now = bench_now();
			const uint64_t left =
				deadline_ns > now ?
					(deadline_ns - now) /
						BENCH_NS_PER_MILLISECOND :
					0;
			(void)poll(polls, poll_count,
				   (int)(left < POLL_SLICE_MILLISECONDS_MOST ?
						 left + 1u :
						 POLL_SLICE_MILLISECONDS_MOST));
		} else {
			/* one engine (or several without a common wait): block on one, the others' slice is the
			 * deadline */
			struct engine *const engine =
				&owner->engines[waiting_engine];
			const uint64_t slice = (uint64_t)run->arm->slice_us *
					       BENCH_NS_PER_MICROSECOND;
			const uint64_t now = bench_now();
			const uint64_t until =
				waiting_engines > 1u &&
						now + slice < deadline_ns ?
					now + slice :
					deadline_ns;
			const uint32_t count = engine->operations->reap(
				engine, completed, REAP_BATCH_MOST, true,
				until);
			for (uint32_t done = 0; done < count; done++)
				attempt_complete(run, completed[done]);
			worked = count != 0;
		}
	}
	if (hedging(run->arm))
		hedge_check(owner);
	return worked;
}

static bool owner_idle(const struct owner *owner)
{
	for (uint32_t index = 0; index < owner->engine_count; index++)
		if (owner->engines[index].in_flight != 0 ||
		    owner->engines[index].pending_head !=
			    owner->engines[index].pending_tail)
			return false;
	return atomic_load(&owner->inbox_signal) == 0;
}

/* ---- engines ----------------------------------------------------------------------------------------------- */

/* Whether a uring arm reads this copy with NVMe commands (command_copies); its other copies take READ_FIXED. */
static bool copy_commanded(const struct arm *arm, uint32_t copy)
{
	return arm->engine == ENGINE_URING &&
	       arm->operation == OPERATION_URING_CMD &&
	       (arm->command_copy_mask >> copy & 1u) != 0;
}

static int descriptor_of(const struct bench_run *run, uint32_t copy)
{
	if (copy_commanded(run->arm, copy))
		return run->store->passthrough[copy];
	return run->arm->buffered ? run->store->buffered[copy] :
				    run->store->direct[copy];
}

/* Synchronous reads of one block: the pool's workers and the inline engine. */
static int64_t block_call(struct bench_run *run, struct attempt *attempt)
{
	const int descriptor = descriptor_of(run, attempt->copy);
	ssize_t result;
	switch (run->arm->call) {
	case CALL_PREADV2: {
		struct iovec vector = { .iov_base = attempt->buffer,
					.iov_len = BENCH_BLOCK_BYTES };
		do
			result = preadv2(descriptor, &vector, 1,
					 (off_t)attempt->offset,
					 (int)run->arm->rw_flags);
		while (result < 0 && errno == EINTR);
		if (result < 0 && errno == EAGAIN &&
		    (run->arm->rw_flags & RWF_NOWAIT) != 0) {
			atomic_fetch_add_explicit(&run->counters.nowait_again,
						  1, memory_order_relaxed);
			do
				result = pread(descriptor, attempt->buffer,
					       BENCH_BLOCK_BYTES,
					       (off_t)attempt->offset);
			while (result < 0 && errno == EINTR);
		}
		break;
	}
	case CALL_IOCTL: {
		const int error = bench_passthrough_read(run->store,
							 attempt->copy,
							 attempt->offset,
							 attempt->buffer);
		return error == 0 ? (int64_t)BENCH_BLOCK_BYTES :
				    -(int64_t)error;
	}
	case CALL_PREAD:
	case CALL_KINDS:
	default:
		do
			result = pread(descriptor, attempt->buffer,
				       BENCH_BLOCK_BYTES,
				       (off_t)attempt->offset);
		while (result < 0 && errno == EINTR);
		break;
	}
	return result < 0 ? -(int64_t)errno : (int64_t)result;
}

/* Finished attempts an engine holds until its next reap (inline, pool and mapped reads; refused submissions). */
struct finished_list {
	uint32_t *indices, head, count,
		capacity; /* head: the first unread entry of a FIFO use */
};

static bool finished_open(struct finished_list *list, uint32_t capacity)
{
	list->indices = calloc(capacity, sizeof(*list->indices));
	list->capacity = capacity;
	list->head = 0;
	list->count = 0;
	return list->indices != NULL;
}

static uint32_t finished_take(struct engine *engine, struct finished_list *list,
			      struct attempt **completed, uint32_t capacity)
{
	const uint32_t count = list->count < capacity ? list->count : capacity;
	for (uint32_t index = 0; index < count; index++)
		completed[index] =
			&engine->run->attempts[list->indices[list->count - 1u -
							     index]];
	list->count -= count;
	engine->in_flight -= count;
	return count;
}

/* -- io_uring -- */
struct uring_state {
	int descriptor, enter_descriptor;
	unsigned enter_flags, setup_flags, features;
	unsigned *sq_head, *sq_tail, *sq_mask, *sq_flags, *sq_array, sq_entries;
	unsigned *cq_head, *cq_tail, *cq_mask, cq_entries;
	unsigned char *sqes, *cqes;
	uint32_t sqe_bytes, cqe_bytes;
	void *ring, *completion_ring, *entries;
	size_t ring_bytes, completion_bytes, entries_bytes;
	unsigned local_tail;
	int files[BENCH_COPIES_MOST];
	bool fixed_files, fixed_buffers;
	struct finished_list refused;
};

static long uring_enter(const struct uring_state *ring, unsigned to_submit,
			unsigned min_complete, unsigned flags,
			const void *argument, size_t argument_bytes)
{
	long result;
	do
		result = syscall(__NR_io_uring_enter, ring->enter_descriptor,
				 to_submit, min_complete,
				 flags | ring->enter_flags, argument,
				 argument_bytes);
	while (result < 0 && errno == EINTR);
	return result;
}

static int uring_open(struct engine *engine)
{
	struct bench_run *const run = engine->run;
	const struct arm *const arm = run->arm;
	struct uring_state *const ring = calloc(1, sizeof(*ring));
	if (ring == NULL ||
	    !finished_open(&ring->refused, run->attempt_count)) {
		free(ring);
		return ENOMEM;
	}
	engine->state = ring;
	struct io_uring_params parameters;
	memset(&parameters, 0, sizeof(parameters));
	parameters.flags = arm->setup_flags;
	const uint32_t polled_flags = IORING_SETUP_IOPOLL |
				      IORING_SETUP_HYBRID_IOPOLL;
	if ((parameters.flags & polled_flags) != 0 &&
	    (engine->copy_mask & ~arm->poll_copy_mask) != 0) {
		parameters.flags &= ~polled_flags;
		if ((parameters.flags & IORING_SETUP_SINGLE_ISSUER) != 0 &&
		    (parameters.flags & IORING_SETUP_SQPOLL) == 0)
			parameters.flags |= IORING_SETUP_DEFER_TASKRUN;
	}
	const bool commands = arm->operation == OPERATION_URING_CMD &&
			      (engine->copy_mask & arm->command_copy_mask) != 0;
	if (commands)
		parameters.flags |= IORING_SETUP_SQE128 | IORING_SETUP_CQE32;
	if (arm->cq_factor != 0) {
		parameters.flags |= IORING_SETUP_CQSIZE;
		parameters.cq_entries = arm->depth * arm->cq_factor;
	}
	if ((parameters.flags & IORING_SETUP_SQPOLL) != 0) {
		parameters.sq_thread_idle = arm->sq_idle_ms;
		if (arm->sq_cpu != UINT32_MAX) {
			parameters.flags |= IORING_SETUP_SQ_AFF;
			parameters.sq_thread_cpu = arm->sq_cpu;
		}
	}
	ring->descriptor =
		(int)syscall(__NR_io_uring_setup, arm->depth, &parameters);
	if (ring->descriptor < 0) {
		const int error = errno;
		(void)snprintf(engine->reason, sizeof(engine->reason),
			       "io_uring_setup flags 0x%x%s", parameters.flags,
			       error == EINVAL || error == EOPNOTSUPP ?
				       BENCH_UNSUPPORTED :
				       "");
		return error;
	}
	ring->enter_descriptor = ring->descriptor;
	ring->setup_flags = parameters.flags;
	ring->features = parameters.features;
	ring->sqe_bytes = (parameters.flags & IORING_SETUP_SQE128) != 0 ?
				  SQE_BIG_BYTES :
				  SQE_PLAIN_BYTES;
	ring->cqe_bytes = (parameters.flags & IORING_SETUP_CQE32) != 0 ?
				  CQE_BIG_BYTES :
				  CQE_PLAIN_BYTES;
	ring->ring_bytes =
		(parameters.flags & IORING_SETUP_NO_SQARRAY) != 0 ?
			parameters.sq_off.flags + sizeof(unsigned) :
			parameters.sq_off.array +
				parameters.sq_entries * sizeof(unsigned);
	ring->completion_bytes =
		parameters.cq_off.cqes +
		(size_t)parameters.cq_entries * ring->cqe_bytes;
	const bool single = (parameters.features & IORING_FEAT_SINGLE_MMAP) !=
			    0;
	if (single && ring->completion_bytes > ring->ring_bytes)
		ring->ring_bytes = ring->completion_bytes;
	ring->ring = mmap(NULL, ring->ring_bytes, PROT_READ | PROT_WRITE,
			  MAP_SHARED | MAP_POPULATE, ring->descriptor,
			  IORING_OFF_SQ_RING);
	ring->completion_ring =
		single ? ring->ring :
			 mmap(NULL, ring->completion_bytes,
			      PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE,
			      ring->descriptor, IORING_OFF_CQ_RING);
	ring->entries_bytes = (size_t)parameters.sq_entries * ring->sqe_bytes;
	ring->entries = mmap(NULL, ring->entries_bytes, PROT_READ | PROT_WRITE,
			     MAP_SHARED | MAP_POPULATE, ring->descriptor,
			     IORING_OFF_SQES);
	if (ring->ring == MAP_FAILED || ring->completion_ring == MAP_FAILED ||
	    ring->entries == MAP_FAILED) {
		(void)snprintf(engine->reason, sizeof(engine->reason),
			       "io_uring mmap");
		return ENOMEM;
	}
	unsigned char *const sq = ring->ring, *const cq = ring->completion_ring;
	ring->sq_head = (unsigned *)(void *)(sq + parameters.sq_off.head);
	ring->sq_tail = (unsigned *)(void *)(sq + parameters.sq_off.tail);
	ring->sq_mask = (unsigned *)(void *)(sq + parameters.sq_off.ring_mask);
	ring->sq_flags = (unsigned *)(void *)(sq + parameters.sq_off.flags);
	ring->sq_array =
		(parameters.flags & IORING_SETUP_NO_SQARRAY) != 0 ?
			NULL :
			(unsigned *)(void *)(sq + parameters.sq_off.array);
	ring->sq_entries = parameters.sq_entries;
	ring->cq_head = (unsigned *)(void *)(cq + parameters.cq_off.head);
	ring->cq_tail = (unsigned *)(void *)(cq + parameters.cq_off.tail);
	ring->cq_mask = (unsigned *)(void *)(cq + parameters.cq_off.ring_mask);
	ring->cq_entries = parameters.cq_entries;
	ring->cqes = cq + parameters.cq_off.cqes;
	ring->sqes = ring->entries;
	ring->local_tail = *ring->sq_tail;
	int descriptors[BENCH_COPIES_MOST];
	for (uint32_t copy = 0; copy < run->store->copy_count; copy++) {
		descriptors[copy] = (engine->copy_mask >> copy & 1u) != 0 ?
					    descriptor_of(run, copy) :
					    -1;
		ring->files[copy] = descriptors[copy];
	}
	if (arm->registered_files) {
		if (syscall(__NR_io_uring_register, ring->descriptor,
			    IORING_REGISTER_FILES, descriptors,
			    run->store->copy_count) != 0) {
			const int error = errno;
			(void)snprintf(engine->reason, sizeof(engine->reason),
				       "IORING_REGISTER_FILES");
			return error;
		}
		ring->fixed_files = true;
		for (uint32_t copy = 0; copy < run->store->copy_count; copy++)
			ring->files[copy] = (int)copy;
	}
	ring->fixed_buffers = arm->operation == OPERATION_READ_FIXED ||
			      arm->operation == OPERATION_READV_FIXED ||
			      (arm->operation == OPERATION_URING_CMD &&
			       (arm->registered_files || !commands));
	if (ring->fixed_buffers) {
		struct iovec spans[BENCH_OWNERS_MOST];
		uint32_t span_count = 0;
		for (uint64_t begin = 0;
		     begin < run->arena_bytes && span_count < BENCH_OWNERS_MOST;
		     begin += REGISTERED_BUFFER_SPAN) {
			const uint64_t left = run->arena_bytes - begin;
			spans[span_count++] = (struct iovec){
				.iov_base = run->arena + begin,
				.iov_len =
					left < REGISTERED_BUFFER_SPAN ?
						(size_t)left :
						(size_t)REGISTERED_BUFFER_SPAN
			};
		}
		if (syscall(__NR_io_uring_register, ring->descriptor,
			    IORING_REGISTER_BUFFERS, spans, span_count) != 0) {
			const int error = errno;
			(void)snprintf(engine->reason, sizeof(engine->reason),
				       "IORING_REGISTER_BUFFERS");
			return error;
		}
	}
	if (arm->registered_ring) {
		struct io_uring_rsrc_update update = {
			.offset = UINT32_MAX, .data = (__u64)ring->descriptor
		};
		if (syscall(__NR_io_uring_register, ring->descriptor,
			    IORING_REGISTER_RING_FDS, &update, 1) == 1) {
			ring->enter_descriptor = (int)update.offset;
			ring->enter_flags = IORING_ENTER_REGISTERED_RING;
		} else {
			bench_refusal(run, "registered_ring", errno);
		}
	}
	return 0;
}

static void uring_close(struct engine *engine)
{
	struct uring_state *const ring = engine->state;
	if (ring == NULL)
		return;
	if (engine->in_flight != 0) {
		engine->run->arena_quarantined =
			true; /* the kernel may still write into the arena */
		return;
	}
	if (ring->entries != NULL && ring->entries != MAP_FAILED)
		(void)munmap(ring->entries, ring->entries_bytes);
	if (ring->completion_ring != NULL &&
	    ring->completion_ring != MAP_FAILED &&
	    ring->completion_ring != ring->ring)
		(void)munmap(ring->completion_ring, ring->completion_bytes);
	if (ring->ring != NULL && ring->ring != MAP_FAILED)
		(void)munmap(ring->ring, ring->ring_bytes);
	if (ring->descriptor > 0)
		(void)close(ring->descriptor);
	free(ring->refused.indices);
	free(ring);
	engine->state = NULL;
}

static void uring_flush(struct engine *engine)
{
	struct uring_state *const ring = engine->state;
	if ((ring->setup_flags & IORING_SETUP_SQPOLL) != 0) {
		if ((__atomic_load_n(ring->sq_flags, __ATOMIC_ACQUIRE) &
		     IORING_SQ_NEED_WAKEUP) != 0)
			(void)uring_enter(ring, 0, 0, IORING_ENTER_SQ_WAKEUP,
					  NULL, 0);
		return;
	}
	const unsigned unsent =
		ring->local_tail -
		__atomic_load_n(ring->sq_head, __ATOMIC_ACQUIRE);
	if (unsent == 0)
		return;
	if (uring_enter(ring, unsent, 0, 0, NULL, 0) < 0 && errno != EAGAIN &&
	    errno != EBUSY) {
		engine->error = errno;
		(void)snprintf(engine->reason, sizeof(engine->reason),
			       "io_uring_enter submit");
	}
}

static uint32_t uring_submit(struct engine *engine,
			     struct attempt *const *attempts, uint32_t count)
{
	struct uring_state *const ring = engine->state;
	struct bench_run *const run = engine->run;
	const struct arm *const arm = run->arm;
	const unsigned head = __atomic_load_n(ring->sq_head, __ATOMIC_ACQUIRE);
	const unsigned space = ring->sq_entries - (ring->local_tail - head);
	const unsigned completions =
		ring->cq_entries > engine->in_flight ?
			ring->cq_entries - engine->in_flight :
			0;
	uint32_t accepted = count < space ? count : space;
	accepted = accepted < completions ? accepted : completions;
	for (uint32_t index = 0; index < accepted; index++) {
		struct attempt *const attempt = attempts[index];
		const unsigned slot = ring->local_tail & *ring->sq_mask;
		struct io_uring_sqe *const entry =
			(void *)(ring->sqes + (size_t)slot * ring->sqe_bytes);
		memset(entry, 0, ring->sqe_bytes);
		const uint16_t buffer_index =
			(uint16_t)((uint64_t)(attempt->buffer - run->arena) /
				   REGISTERED_BUFFER_SPAN);
		attempt->vector =
			(struct iovec){ .iov_base = attempt->buffer,
					.iov_len = BENCH_BLOCK_BYTES };
		bool refused = false;
		const uint32_t operation =
			arm->operation == OPERATION_URING_CMD &&
					!copy_commanded(arm, attempt->copy) ?
				(ring->fixed_buffers ? OPERATION_READ_FIXED :
						       OPERATION_READ) :
				arm->operation;
		switch ((enum uring_operation)operation) {
		case OPERATION_READV:
		case OPERATION_READV_FIXED:
			entry->opcode = operation == OPERATION_READV ?
						IORING_OP_READV :
						BENCH_OP_READV_FIXED;
			entry->addr = (uint64_t)(uintptr_t)&attempt->vector;
			entry->len = 1;
			entry->buf_index = buffer_index;
			break;
		case OPERATION_URING_CMD: {
			uint64_t block_address = 0;
			if (bench_lba_of(run->store, attempt->copy,
					 attempt->offset,
					 &block_address) != 0) {
				attempt->result =
					-ENODATA; /* not in the extent map: the fallback reads another way */
				ring->refused.indices[ring->refused.count++] =
					attempt->index;
				refused = true;
				break;
			}
			struct nvme_uring_cmd *const command =
				(void *)entry->cmd;
			entry->opcode = IORING_OP_URING_CMD;
			entry->cmd_op = NVME_URING_CMD_IO;
			command->opcode = NVME_READ_OPCODE;
			command->nsid =
				run->store->passthrough_namespace[attempt->copy];
			command->addr = (uint64_t)(uintptr_t)attempt->buffer;
			command->data_len = BENCH_BLOCK_BYTES;
			command->cdw10 = (uint32_t)block_address;
			command->cdw11 =
				(uint32_t)(block_address >> NVME_HIGH_SHIFT);
			command->cdw12 = BENCH_BLOCK_BYTES /
						 run->store->logical_block_bytes
							 [attempt->copy] -
					 1u;
			if (ring->fixed_buffers) {
				entry->uring_cmd_flags = IORING_URING_CMD_FIXED;
				entry->buf_index = buffer_index;
			}
			break;
		}
		case OPERATION_READ_FIXED:
			entry->opcode = IORING_OP_READ_FIXED;
			entry->addr = (uint64_t)(uintptr_t)attempt->buffer;
			entry->len = BENCH_BLOCK_BYTES;
			entry->buf_index = buffer_index;
			break;
		case OPERATION_READ:
		case OPERATION_KINDS:
		default:
			entry->opcode = IORING_OP_READ;
			entry->addr = (uint64_t)(uintptr_t)attempt->buffer;
			entry->len = BENCH_BLOCK_BYTES;
			break;
		}
		if (refused)
			continue;
		entry->fd = ring->files[attempt->copy];
		entry->flags =
			(uint8_t)((ring->fixed_files ? IOSQE_FIXED_FILE : 0u) |
				  (arm->sqe_async ? IOSQE_ASYNC : 0u));
		if (entry->opcode != IORING_OP_URING_CMD)
			entry->off = attempt->offset;
		entry->user_data = attempt->index;
		if (ring->sq_array != NULL)
			ring->sq_array[slot] = slot;
		ring->local_tail++;
	}
	__atomic_store_n(ring->sq_tail, ring->local_tail, __ATOMIC_RELEASE);
	engine->in_flight += accepted;
	uring_flush(engine);
	return accepted;
}

static uint32_t uring_drain(struct engine *engine, struct attempt **completed,
			    uint32_t capacity)
{
	struct uring_state *const ring = engine->state;
	struct bench_run *const run = engine->run;
	uint32_t count =
		finished_take(engine, &ring->refused, completed, capacity);
	unsigned head = *ring->cq_head;
	const unsigned tail = __atomic_load_n(ring->cq_tail, __ATOMIC_ACQUIRE);
	while (head != tail && count < capacity) {
		const struct io_uring_cqe *const entry =
			(const void *)(ring->cqes +
				       (size_t)(head & *ring->cq_mask) *
					       ring->cqe_bytes);
		head++;
		if (entry->user_data >= run->attempt_count) {
			engine->error = EPROTO;
			(void)snprintf(engine->reason, sizeof(engine->reason),
				       "a completion names no attempt");
			continue;
		}
		struct attempt *const attempt =
			&run->attempts[entry->user_data];
		attempt->result =
			copy_commanded(run->arm, attempt->copy) ?
				(entry->res == 0 ? (int64_t)BENCH_BLOCK_BYTES :
				 entry->res > 0	 ? -(int64_t)EIO :
						   (int64_t)entry->res) :
				(int64_t)entry->res;
		completed[count++] = attempt;
		engine->in_flight--;
	}
	__atomic_store_n(ring->cq_head, head, __ATOMIC_RELEASE);
	return count;
}

static uint32_t uring_reap(struct engine *engine, struct attempt **completed,
			   uint32_t capacity, bool block, uint64_t deadline_ns)
{
	struct uring_state *const ring = engine->state;
	const struct arm *const arm = engine->run->arm;
	uring_flush(engine);
	const bool cooperative =
		(ring->setup_flags & IORING_SETUP_COOP_TASKRUN) != 0;
	const bool flagged = (ring->setup_flags & IORING_SETUP_TASKRUN_FLAG) !=
			     0;
	const bool must_enter =
		(ring->setup_flags &
		 (IORING_SETUP_DEFER_TASKRUN | IORING_SETUP_IOPOLL)) != 0 ||
		(cooperative &&
		 (!flagged ||
		  (__atomic_load_n(ring->sq_flags, __ATOMIC_ACQUIRE) &
		   IORING_SQ_TASKRUN) != 0));
	if (must_enter &&
	    uring_enter(ring, 0, 0, IORING_ENTER_GETEVENTS, NULL, 0) < 0 &&
	    errno != EAGAIN && errno != EBUSY && errno != ETIME) {
		engine->error = errno;
		(void)snprintf(engine->reason, sizeof(engine->reason),
			       "io_uring_enter getevents");
	}
	uint32_t count = uring_drain(engine, completed, capacity);
	if (count != 0 || !block || engine->in_flight == 0)
		return count;
	const uint64_t now = bench_now();
	const uint64_t left = deadline_ns > now ? deadline_ns - now : 0;
	if (arm->wait == WAIT_POLL) {
		struct pollfd wait = { .fd = ring->descriptor,
				       .events = POLLIN };
		const uint64_t milliseconds =
			left / BENCH_NS_PER_MILLISECOND + 1u;
		(void)poll(&wait, 1,
			   milliseconds < POLL_SLICE_MILLISECONDS_MOST ?
				   (int)milliseconds :
				   POLL_SLICE_MILLISECONDS_MOST);
	} else {
		struct __kernel_timespec timeout = {
			.tv_sec = (long long)(left / BENCH_NS_PER_SECOND),
			.tv_nsec = (long long)(left % BENCH_NS_PER_SECOND)
		};
		struct io_uring_getevents_arg argument = {
			.ts = (uint64_t)(uintptr_t)&timeout
		};
		if (uring_enter(ring, 0, 1,
				IORING_ENTER_GETEVENTS | IORING_ENTER_EXT_ARG,
				&argument, sizeof(argument)) < 0 &&
		    errno != ETIME && errno != EAGAIN && errno != EBUSY) {
			engine->error = errno;
			(void)snprintf(engine->reason, sizeof(engine->reason),
				       "io_uring_enter wait");
		}
	}
	if (must_enter)
		(void)uring_enter(ring, 0, 0, IORING_ENTER_GETEVENTS, NULL, 0);
	count = uring_drain(engine, completed, capacity);
	return count;
}

static int uring_poll_descriptor(const struct engine *engine)
{
	const struct uring_state *const ring = engine->state;
	return ring != NULL && (ring->setup_flags &
				IORING_SETUP_DEFER_TASKRUN) == 0 ?
		       ring->descriptor :
		       -1;
}

static const struct engine_operations uring_operations = {
	"uring",      uring_open, uring_close,
	uring_submit, uring_reap, uring_poll_descriptor
};

/* -- Linux AIO (raw syscalls) -- */
struct aio_state {
	aio_context_t context;
	uint32_t capacity;
	struct io_event *events;
	struct iocb **pointers;
};

static int aio_open(struct engine *engine)
{
	struct aio_state *const aio = calloc(1, sizeof(*aio));
	if (aio == NULL)
		return ENOMEM;
	engine->state = aio;
	aio->capacity = engine->run->arm->depth;
	aio->events = calloc(aio->capacity, sizeof(*aio->events));
	aio->pointers = calloc(aio->capacity, sizeof(*aio->pointers));
	if (aio->events == NULL || aio->pointers == NULL)
		return ENOMEM;
	if (syscall(__NR_io_setup, aio->capacity, &aio->context) != 0) {
		const int error = errno;
		(void)snprintf(engine->reason, sizeof(engine->reason),
			       "io_setup %u", aio->capacity);
		return error;
	}
	return 0;
}

static void aio_close(struct engine *engine)
{
	struct aio_state *const aio = engine->state;
	if (aio == NULL)
		return;
	if (aio->context != 0)
		(void)syscall(__NR_io_destroy,
			      aio->context); /* waits for what is in flight */
	free(aio->events);
	free(aio->pointers);
	free(aio);
	engine->state = NULL;
}

static uint32_t aio_submit(struct engine *engine,
			   struct attempt *const *attempts, uint32_t count)
{
	struct aio_state *const aio = engine->state;
	const uint32_t room = aio->capacity - engine->in_flight;
	const uint32_t wanted = count < room ? count : room;
	for (uint32_t index = 0; index < wanted; index++) {
		struct attempt *const attempt = attempts[index];
		memset(&attempt->control, 0, sizeof(attempt->control));
		attempt->control.aio_data = attempt->index;
		attempt->control.aio_lio_opcode = IOCB_CMD_PREAD;
		attempt->control.aio_fildes =
			(uint32_t)descriptor_of(engine->run, attempt->copy);
		attempt->control.aio_buf = (uint64_t)(uintptr_t)attempt->buffer;
		attempt->control.aio_nbytes = BENCH_BLOCK_BYTES;
		attempt->control.aio_offset = (int64_t)attempt->offset;
		attempt->control.aio_rw_flags = (int)engine->run->arm->rw_flags;
		aio->pointers[index] = &attempt->control;
	}
	if (wanted == 0)
		return 0;
	const long submitted = syscall(__NR_io_submit, aio->context,
				       (long)wanted, aio->pointers);
	if (submitted < 0) {
		if (errno != EAGAIN) {
			engine->error = errno;
			(void)snprintf(engine->reason, sizeof(engine->reason),
				       "io_submit");
		}
		return 0;
	}
	engine->in_flight += (uint32_t)submitted;
	return (uint32_t)submitted;
}

static uint32_t aio_reap(struct engine *engine, struct attempt **completed,
			 uint32_t capacity, bool block, uint64_t deadline_ns)
{
	struct aio_state *const aio = engine->state;
	const uint64_t now = bench_now();
	const uint64_t left = block && deadline_ns > now ? deadline_ns - now :
							   0;
	struct timespec timeout = {
		.tv_sec = (time_t)(left / BENCH_NS_PER_SECOND),
		.tv_nsec = (long)(left % BENCH_NS_PER_SECOND)
	};
	const uint32_t most = capacity < aio->capacity ? capacity :
							 aio->capacity;
	long count;
	do
		count = syscall(__NR_io_getevents, aio->context,
				block ? 1L : 0L, (long)most, aio->events,
				&timeout);
	while (count < 0 && errno == EINTR);
	if (count < 0) {
		engine->error = errno;
		(void)snprintf(engine->reason, sizeof(engine->reason),
			       "io_getevents");
		return 0;
	}
	for (long index = 0; index < count; index++) {
		struct attempt *const attempt =
			&engine->run->attempts[aio->events[index].data];
		attempt->result = aio->events[index].res;
		completed[index] = attempt;
	}
	engine->in_flight -= (uint32_t)count;
	return (uint32_t)count;
}

static int no_poll_descriptor(const struct engine *engine)
{
	(void)engine;
	return -1;
}

static const struct engine_operations aio_operations = {
	"aio", aio_open, aio_close, aio_submit, aio_reap, no_poll_descriptor
};

/* -- a pool of threads making synchronous calls (pread, preadv2 with flags, the NVMe ioctl) -- */
struct pool_state {
	struct engine *engine;
	pthread_t workers[BENCH_POOL_THREADS_MOST];
	uint32_t worker_count;
	pthread_mutex_t lock;
	pthread_cond_t work, done;
	uint32_t *queue, head, tail, capacity;
	struct finished_list finished;
	_Atomic uint32_t queued, stop;
	_Atomic uint64_t cpu_ns;
	bool spin;
};

static void *pool_worker(void *argument)
{
	struct pool_state *const pool = argument;
	struct bench_run *const run = pool->engine->run;
	bench_thread_attributes(run, -1);
	for (;;) {
		if (pool->spin)
			while (atomic_load_explicit(&pool->queued,
						    memory_order_acquire) ==
				       0 &&
			       atomic_load(&pool->stop) == 0)
				cpu_relax();
		(void)pthread_mutex_lock(&pool->lock);
		while (!pool->spin && pool->head == pool->tail &&
		       atomic_load(&pool->stop) == 0)
			(void)pthread_cond_wait(&pool->work, &pool->lock);
		if (pool->head == pool->tail) {
			const bool stopping = atomic_load(&pool->stop) != 0;
			(void)pthread_mutex_unlock(&pool->lock);
			if (stopping)
				break;
			continue;
		}
		struct attempt *const attempt =
			&run->attempts[pool->queue[pool->head++ %
						   pool->capacity]];
		atomic_fetch_sub(&pool->queued, 1);
		(void)pthread_mutex_unlock(&pool->lock);
		attempt->result = block_call(run, attempt);
		(void)pthread_mutex_lock(&pool->lock);
		pool->finished.indices[pool->finished.count++] = attempt->index;
		(void)pthread_cond_signal(&pool->done);
		(void)pthread_mutex_unlock(&pool->lock);
	}
	atomic_fetch_add(&pool->cpu_ns, bench_thread_cpu_ns());
	return NULL;
}

static int pool_open(struct engine *engine)
{
	struct bench_run *const run = engine->run;
	struct pool_state *const pool = calloc(1, sizeof(*pool));
	if (pool == NULL)
		return ENOMEM;
	engine->state = pool;
	pool->engine = engine;
	pool->spin = run->arm->pool_spin;
	pool->capacity = run->attempt_count;
	pool->queue = calloc(pool->capacity, sizeof(*pool->queue));
	if (pool->queue == NULL ||
	    !finished_open(&pool->finished, run->attempt_count) ||
	    pthread_mutex_init(&pool->lock, NULL) != 0 ||
	    pthread_cond_init(&pool->work, NULL) != 0 ||
	    pthread_cond_init(&pool->done, NULL) != 0)
		return ENOMEM;
	const uint32_t workers =
		run->arm->pool_threads / run->owner_count != 0 ?
			run->arm->pool_threads / run->owner_count :
			1u;
	for (uint32_t index = 0;
	     index < workers && index < BENCH_POOL_THREADS_MOST; index++) {
		if (pthread_create(&pool->workers[index], NULL, pool_worker,
				   pool) != 0)
			return EAGAIN;
		pool->worker_count++;
	}
	return 0;
}

static void pool_close(struct engine *engine)
{
	struct pool_state *const pool = engine->state;
	if (pool == NULL)
		return;
	(void)pthread_mutex_lock(&pool->lock);
	atomic_store(&pool->stop, 1);
	(void)pthread_cond_broadcast(&pool->work);
	(void)pthread_mutex_unlock(&pool->lock);
	for (uint32_t index = 0; index < pool->worker_count; index++)
		(void)pthread_join(pool->workers[index], NULL);
	atomic_fetch_add(&engine->run->pool_cpu_ns, atomic_load(&pool->cpu_ns));
	free(pool->queue);
	free(pool->finished.indices);
	free(pool);
	engine->state = NULL;
}

static uint32_t pool_submit(struct engine *engine,
			    struct attempt *const *attempts, uint32_t count)
{
	struct pool_state *const pool = engine->state;
	(void)pthread_mutex_lock(&pool->lock);
	for (uint32_t index = 0; index < count; index++)
		pool->queue[pool->tail++ % pool->capacity] =
			attempts[index]->index;
	atomic_fetch_add_explicit(&pool->queued, count, memory_order_release);
	if (!pool->spin)
		(void)pthread_cond_broadcast(&pool->work);
	(void)pthread_mutex_unlock(&pool->lock);
	engine->in_flight += count;
	return count;
}

static uint32_t pool_reap(struct engine *engine, struct attempt **completed,
			  uint32_t capacity, bool block, uint64_t deadline_ns)
{
	struct pool_state *const pool = engine->state;
	(void)pthread_mutex_lock(&pool->lock);
	if (pool->finished.count == 0 && block) {
		struct timespec until;
		(void)clock_gettime(CLOCK_REALTIME, &until);
		const uint64_t now = bench_now();
		const uint64_t left = deadline_ns > now ? deadline_ns - now : 0;
		const uint64_t total =
			(uint64_t)until.tv_nsec + left % BENCH_NS_PER_SECOND;
		until.tv_sec += (time_t)(left / BENCH_NS_PER_SECOND +
					 total / BENCH_NS_PER_SECOND);
		until.tv_nsec = (long)(total % BENCH_NS_PER_SECOND);
		(void)pthread_cond_timedwait(&pool->done, &pool->lock, &until);
	}
	const uint32_t count =
		finished_take(engine, &pool->finished, completed, capacity);
	(void)pthread_mutex_unlock(&pool->lock);
	return count;
}

static const struct engine_operations pool_operations = {
	"pool",	     pool_open, pool_close,
	pool_submit, pool_reap, no_poll_descriptor
};

/* -- inline synchronous reads on the owner's thread (with readahead: started at submit, read at reap) -- */
struct sync_state {
	struct finished_list queued, finished;
};

static int sync_open(struct engine *engine)
{
	struct sync_state *const state = calloc(1, sizeof(*state));
	if (state == NULL)
		return ENOMEM;
	engine->state = state;
	return finished_open(&state->queued, engine->run->attempt_count) &&
			       finished_open(&state->finished,
					     engine->run->attempt_count) ?
		       0 :
		       ENOMEM;
}

static void sync_close(struct engine *engine)
{
	struct sync_state *const state = engine->state;
	if (state == NULL)
		return;
	free(state->queued.indices);
	free(state->finished.indices);
	free(state);
	engine->state = NULL;
}

static uint32_t sync_submit(struct engine *engine,
			    struct attempt *const *attempts, uint32_t count)
{
	struct sync_state *const state = engine->state;
	struct bench_run *const run = engine->run;
	for (uint32_t index = 0; index < count; index++) {
		struct attempt *const attempt = attempts[index];
		if (run->arm->readahead) {
			if (readahead(descriptor_of(run, attempt->copy),
				      (off_t)attempt->offset,
				      BENCH_BLOCK_BYTES) != 0)
				bench_refusal(run, "readahead", errno);
			state->queued.indices[state->queued.count++] =
				attempt->index;
		} else {
			attempt->result = block_call(run, attempt);
			state->finished.indices[state->finished.count++] =
				attempt->index;
		}
	}
	engine->in_flight += count;
	return count;
}

static uint32_t sync_reap(struct engine *engine, struct attempt **completed,
			  uint32_t capacity, bool block, uint64_t deadline_ns)
{
	(void)block;
	(void)deadline_ns;
	struct sync_state *const state = engine->state;
	for (uint32_t index = 0; index < state->queued.count; index++) {
		struct attempt *const attempt =
			&engine->run->attempts[state->queued.indices[index]];
		attempt->result = block_call(engine->run, attempt);
		state->finished.indices[state->finished.count++] =
			attempt->index;
	}
	state->queued.count = 0;
	return finished_take(engine, &state->finished, completed, capacity);
}

static const struct engine_operations sync_operations = {
	"sync",	     sync_open, sync_close,
	sync_submit, sync_reap, no_poll_descriptor
};

/* -- mapped copies: page cache through mmap, MADV_RANDOM; with willneed, MADV_WILLNEED at submit -- */
struct mapped_state {
	unsigned char *maps[BENCH_COPIES_MOST];
	struct finished_list queued;
};

static int mapped_open(struct engine *engine)
{
	struct bench_run *const run = engine->run;
	struct mapped_state *const state = calloc(1, sizeof(*state));
	if (state == NULL)
		return ENOMEM;
	engine->state = state;
	if (!finished_open(&state->queued, run->attempt_count))
		return ENOMEM;
	for (uint32_t copy = 0; copy < run->store->copy_count; copy++) {
		if ((engine->copy_mask >> copy & 1u) == 0)
			continue;
		void *const map = mmap(NULL, run->store->file_bytes, PROT_READ,
				       MAP_SHARED, run->store->buffered[copy],
				       0);
		if (map == MAP_FAILED) {
			const int error = errno;
			(void)snprintf(engine->reason, sizeof(engine->reason),
				       "mmap copy %u", copy);
			return error;
		}
		state->maps[copy] = map;
		for (uint32_t table = 0; table < run->store->table_count;
		     table++)
			if (madvise(state->maps[copy] +
					    run->store->tables[table]
						    .byte_offset,
				    run->store->tables[table].block_count *
					    BENCH_BLOCK_BYTES,
				    MADV_RANDOM) != 0)
				bench_refusal(run, "madvise_random", errno);
	}
	return 0;
}

static void mapped_close(struct engine *engine)
{
	struct mapped_state *const state = engine->state;
	if (state == NULL)
		return;
	for (uint32_t copy = 0; copy < BENCH_COPIES_MOST; copy++)
		if (state->maps[copy] != NULL)
			(void)munmap(state->maps[copy],
				     engine->run->store->file_bytes);
	free(state->queued.indices);
	free(state);
	engine->state = NULL;
}

static uint32_t mapped_submit(struct engine *engine,
			      struct attempt *const *attempts, uint32_t count)
{
	struct mapped_state *const state = engine->state;
	for (uint32_t index = 0; index < count; index++) {
		if (engine->run->arm->willneed &&
		    madvise(state->maps[attempts[index]->copy] +
				    attempts[index]->offset,
			    BENCH_BLOCK_BYTES, MADV_WILLNEED) != 0)
			bench_refusal(engine->run, "madvise_willneed", errno);
		state->queued.indices[state->queued.count++] =
			attempts[index]->index;
	}
	engine->in_flight += count;
	return count;
}

static uint32_t mapped_reap(struct engine *engine, struct attempt **completed,
			    uint32_t capacity, bool block, uint64_t deadline_ns)
{
	(void)block;
	(void)deadline_ns;
	struct mapped_state *const state = engine->state;
	struct finished_list *const queued = &state->queued;
	const uint32_t left = queued->count - queued->head;
	const uint32_t count = left < capacity ? left : capacity;
	for (uint32_t index = 0; index < count; index++) {
		struct attempt *const attempt =
			&engine->run->attempts[queued->indices[queued->head +
							       index]];
		memcpy(attempt->buffer,
		       state->maps[attempt->copy] + attempt->offset,
		       BENCH_BLOCK_BYTES);
		attempt->result = BENCH_BLOCK_BYTES;
		completed[index] = attempt;
	}
	queued->head += count;
	if (queued->head == queued->count)
		queued->head = queued->count = 0;
	engine->in_flight -= count;
	return count;
}

static const struct engine_operations mapped_operations = {
	"mmap",	       mapped_open, mapped_close,
	mapped_submit, mapped_reap, no_poll_descriptor
};

static const struct engine_operations *operations_of(enum engine_kind kind)
{
	switch (kind) {
	case ENGINE_URING:
		return &uring_operations;
	case ENGINE_AIO:
		return &aio_operations;
	case ENGINE_POOL:
		return &pool_operations;
	case ENGINE_MMAP:
		return &mapped_operations;
	case ENGINE_SYNC:
	case ENGINE_PROD:
	case ENGINE_KINDS:
	default:
		return &sync_operations;
	}
}

/* ---- owners ------------------------------------------------------------------------------------------------ */

static int owner_open_engines(struct owner *owner)
{
	struct bench_run *const run = owner->run;
	for (uint32_t index = 0; index < owner->engine_count; index++) {
		struct engine *const engine = &owner->engines[index];
		engine->operations = operations_of(run->arm->engine);
		engine->run = run;
		engine->owner = owner;
		engine->pending_capacity = run->attempt_count;
		engine->pending = calloc(engine->pending_capacity,
					 sizeof(*engine->pending));
		if (engine->pending == NULL)
			return ENOMEM;
		const int error = engine->operations->open(engine);
		if (error != 0) {
			char reason[BENCH_REASON_BYTES];
			(void)snprintf(reason, sizeof(reason), "%s: %s",
				       engine->operations->name,
				       engine->reason);
			bench_fail(run, reason, error);
			return error;
		}
	}
	return 0;
}

static void owner_close_engines(struct owner *owner)
{
	for (uint32_t index = 0; index < owner->engine_count; index++) {
		struct engine *const engine = &owner->engines[index];
		if (engine->operations != NULL)
			engine->operations->close(engine);
		free(engine->pending);
		engine->pending = NULL;
	}
}

/* Reaps everything still in flight, to the deadline, so that no engine closes with I/O into the arena. */
static void owner_drain(struct owner *owner, uint64_t deadline_ns)
{
	while (!owner_idle(owner) && bench_now() < deadline_ns) {
		bool any_error = false;
		for (uint32_t index = 0; index < owner->engine_count; index++)
			any_error = any_error ||
				    owner->engines[index].error != 0;
		if (any_error)
			break;
		(void)owner_progress(owner, true, bench_now() + IDLE_SLICE_NS);
	}
}

static int32_t owner_cpu(const struct owner *owner)
{
	const struct bench_run *const run = owner->run;
	switch (run->arm->pin) {
	case PIN_IRQ:
		return owner->copy_mask != 0 ?
			       (int32_t)run->store->irq_cpu[__builtin_ctz(
				       owner->copy_mask)] :
			       -1;
	case PIN_LIST:
		return run->arm->pin_cpu_count != 0 ?
			       (int32_t)run->arm
				       ->pin_cpus[owner->index %
						  run->arm->pin_cpu_count] :
			       -1;
	case PIN_NONE:
	case PIN_KINDS:
	default:
		return -1;
	}
}

static void *owner_main(void *argument)
{
	struct owner *const owner = argument;
	struct bench_run *const run = owner->run;
	bench_thread_attributes(run, owner_cpu(owner));
	owner->open_error = owner_open_engines(owner);
	atomic_store(&owner->ready, 1);
	futex_wake(&owner->ready);
	uint64_t last_work = bench_now();
	while (owner->open_error == 0 && atomic_load(&owner->stop) == 0) {
		if (owner_progress(owner, false, 0)) {
			last_work = bench_now();
			continue;
		}
		const uint64_t now = bench_now();
		const bool spinning =
			run->arm->wait == WAIT_SPIN ||
			(run->arm->wait == WAIT_HYBRID &&
			 now - last_work < (uint64_t)run->arm->spin_us *
						   BENCH_NS_PER_MICROSECOND);
		if (spinning) {
			cpu_relax();
			continue;
		}
		if (!owner_idle(owner)) {
			if (owner_progress(
				    owner, true,
				    now + (uint64_t)run->arm->slice_us *
						    BENCH_NS_PER_MICROSECOND))
				last_work = bench_now();
		} else {
			futex_wait(&owner->inbox_signal, 0, IDLE_SLICE_NS);
		}
	}
	owner_drain(owner, bench_now() + BENCH_READ_LIMIT_NS);
	owner_close_engines(owner);
	struct rusage usage;
	if (getrusage(RUSAGE_THREAD, &usage) == 0) {
		owner->voluntary_switches = (uint64_t)usage.ru_nvcsw;
		owner->involuntary_switches = (uint64_t)usage.ru_nivcsw;
	}
	owner->cpu_ns = bench_thread_cpu_ns();
	return NULL;
}

/* ---- the run ----------------------------------------------------------------------------------------------- */

int bench_run_open(struct bench_run *run, const struct bench_store *store,
		   const struct arm *arm, uint32_t reads_most)
{
	memset(run, 0, sizeof(*run));
	run->store = store;
	run->arm = arm;
	atomic_init(&run->random_state, (uint64_t)bench_now() | 1u);
	if (pthread_mutex_init(&run->free_lock, NULL) != 0 ||
	    pthread_mutex_init(&run->refusal_lock, NULL) != 0)
		return ENOMEM;
	run->read_capacity = reads_most;
	run->attempt_count = reads_most * SLOT_FACTOR + SUBMIT_BATCH_MOST;
	run->arena_bytes = ((uint64_t)run->attempt_count * BENCH_BLOCK_BYTES +
			    HUGE_PAGE_BYTES - 1u) /
			   HUGE_PAGE_BYTES * HUGE_PAGE_BYTES;
	run->arena = mmap(NULL, run->arena_bytes, PROT_READ | PROT_WRITE,
			  MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
	if (run->arena == MAP_FAILED) {
		run->arena = NULL;
		return ENOMEM;
	}
	if (madvise(run->arena, run->arena_bytes, MADV_HUGEPAGE) != 0)
		bench_refusal(run, "arena_hugepage", errno);
	memset(run->arena, 0, run->arena_bytes);
	if (mlock(run->arena, run->arena_bytes) != 0)
		bench_refusal(run, "arena_mlock", errno);
	run->attempts = calloc(run->attempt_count, sizeof(*run->attempts));
	run->free_slots = calloc(run->attempt_count, sizeof(*run->free_slots));
	for (uint32_t epoch = 0; epoch < BENCH_EPOCHS; epoch++)
		run->epochs[epoch].reads =
			calloc(reads_most, sizeof(*run->epochs[epoch].reads));
	if (run->attempts == NULL || run->free_slots == NULL ||
	    run->epochs[BENCH_EPOCHS - 1u].reads == NULL ||
	    run->epochs[0].reads == NULL)
		return ENOMEM;
	for (uint32_t index = 0; index < run->attempt_count; index++) {
		run->attempts[index].index = index;
		run->attempts[index].buffer =
			run->arena + (uint64_t)index * BENCH_BLOCK_BYTES;
		run->free_slots[index] = run->attempt_count - 1u - index;
	}
	run->free_count = run->attempt_count;
	/* owners: inline is one owner on the step thread; threads share the copies round-robin */
	const uint32_t copies = store->copy_count;
	run->owner_count = arm->io_threads == 0 ? 1u : arm->io_threads;
	run->owner_count = run->owner_count < BENCH_OWNERS_MOST ?
				   run->owner_count :
				   BENCH_OWNERS_MOST;
	const bool shared_copies =
		run->owner_count >
		copies; /* more threads than copies: they take turns per copy */
	for (uint32_t copy = 0; copy < copies; copy++) {
		run->owner_of_copy[copy] =
			shared_copies ? copy : copy % run->owner_count;
		run->owners_of_copy[copy] =
			shared_copies ? (run->owner_count - copy + copies -
					 1u) / copies :
					1u;
	}
	for (uint32_t index = 0; index < run->owner_count; index++) {
		struct owner *const owner = &run->owners[index];
		owner->run = run;
		owner->index = index;
		owner->threaded = arm->io_threads != 0;
		owner->inbox_capacity = run->attempt_count;
		owner->inbox =
			calloc(owner->inbox_capacity, sizeof(*owner->inbox));
		owner->issued_capacity = run->attempt_count * SLOT_FACTOR;
		owner->issued =
			calloc(owner->issued_capacity, sizeof(*owner->issued));
		if (owner->inbox == NULL || owner->issued == NULL ||
		    pthread_mutex_init(&owner->inbox_lock, NULL) != 0)
			return ENOMEM;
		uint32_t owned = 0;
		for (uint32_t copy = 0; copy < copies; copy++)
			owned |= (shared_copies ?
					  index % copies == copy :
					  copy % run->owner_count == index) ?
					 1u << copy :
					 0u;
		owner->copy_mask = owned;
		if (arm->per_device_rings && (arm->engine == ENGINE_URING ||
					      arm->engine == ENGINE_AIO)) {
			for (uint32_t copy = 0; copy < copies; copy++)
				if ((owned >> copy & 1u) != 0) {
					owner->engine_of_copy[copy] =
						owner->engine_count;
					owner->engines[owner->engine_count++]
						.copy_mask = 1u << copy;
				}
		} else {
			for (uint32_t copy = 0; copy < copies; copy++)
				owner->engine_of_copy[copy] = 0;
			owner->engines[0].copy_mask = owned;
			owner->engine_count = 1;
		}
	}
	if (arm->io_threads == 0)
		return owner_open_engines(&run->owners[0]);
	for (uint32_t index = 0; index < run->owner_count; index++) {
		struct owner *const owner = &run->owners[index];
		if (pthread_create(&owner->thread, NULL, owner_main, owner) !=
		    0)
			return EAGAIN;
		while (atomic_load(&owner->ready) == 0)
			futex_wait(&owner->ready, 0, IDLE_SLICE_NS);
		if (owner->open_error != 0)
			return owner->open_error;
	}
	return 0;
}

void bench_run_close(struct bench_run *run)
{
	if (run->arm->io_threads == 0) {
		owner_drain(&run->owners[0], bench_now() + BENCH_READ_LIMIT_NS);
		owner_close_engines(&run->owners[0]);
	} else {
		for (uint32_t index = 0; index < run->owner_count; index++) {
			struct owner *const owner = &run->owners[index];
			if (owner->thread == 0)
				continue;
			atomic_store(&owner->stop, 1);
			futex_wake(&owner->inbox_signal);
			(void)pthread_join(owner->thread, NULL);
		}
	}
	for (uint32_t index = 0; index < run->owner_count; index++) {
		free(run->owners[index].inbox);
		free(run->owners[index].issued);
	}
	for (uint32_t epoch = 0; epoch < BENCH_EPOCHS; epoch++)
		free(run->epochs[epoch].reads);
	free(run->attempts);
	free(run->free_slots);
	if (run->arena != NULL && !run->arena_quarantined)
		(void)munmap(run->arena, run->arena_bytes);
}

struct logical_read *bench_read_add(struct bench_run *run, uint32_t table,
				    uint64_t block, uint32_t unique)
{
	struct read_epoch *const epoch = &run->epochs[run->epoch];
	if (epoch->count == run->read_capacity) {
		bench_fail(run, "a step needs more reads than its capacity",
			   ENOBUFS);
		return NULL;
	}
	struct logical_read *const read = &epoch->reads[epoch->count++];
	read->block = block;
	read->table = table;
	read->unique = unique;
	read->epoch = run->epoch;
	atomic_store(&read->state, READ_WAITING);
	atomic_store(&read->live, 0);
	atomic_store(&read->tried, 0);
	atomic_store(&read->tries, 0);
	atomic_store(&read->hedged, 0);
	atomic_store(&read->winner, NULL);
	atomic_fetch_add(&run->pending[table], 1);
	return read;
}

void bench_reads_issue(struct bench_run *run, uint32_t first, uint32_t count)
{
	struct read_epoch *const epoch = &run->epochs[run->epoch];
	uint32_t projected[BENCH_COPIES_MOST] = { 0 };
	struct attempt *batch[SUBMIT_BATCH_MOST];
	for (uint32_t begin = first;
	     begin < first + count && atomic_load(&run->error) == 0;) {
		uint32_t made = 0;
		for (; begin < first + count && made < SUBMIT_BATCH_MOST;
		     begin++) {
			struct logical_read *const read = &epoch->reads[begin];
			struct attempt *const attempt = attempt_for(
				run, read,
				copy_choose(run, read, 0, projected));
			if (attempt == NULL) {
				read_finish(run, read, READ_FAILED);
				continue;
			}
			attempt->hedge = false;
			batch[made++] = attempt;
		}
		route(run, batch, made);
	}
	if (run->arm->io_threads == 0)
		(void)owner_progress(
			&run->owners[0], false,
			0); /* submit now: the issue point is here */
}

int bench_wait_table(struct bench_run *run, uint32_t table,
		     uint64_t deadline_ns)
{
	const struct arm *const arm = run->arm;
	const uint64_t started = bench_now();
	const uint64_t spin_until =
		started + (uint64_t)arm->spin_us * BENCH_NS_PER_MICROSECOND;
	for (;;) {
		const uint32_t pending = atomic_load(&run->pending[table]);
		if (pending == 0 || atomic_load(&run->error) != 0)
			break;
		const uint64_t now = bench_now();
		if (now >= deadline_ns) {
			bench_fail(
				run,
				"a table's reads did not complete before the read limit",
				ETIMEDOUT);
			break;
		}
		const bool spinning =
			arm->wait == WAIT_SPIN ||
			(arm->wait == WAIT_HYBRID && now < spin_until);
		if (arm->io_threads == 0) {
			const uint64_t slice = (uint64_t)arm->slice_us *
					       BENCH_NS_PER_MICROSECOND;
			(void)owner_progress(&run->owners[0], !spinning,
					     now + slice < deadline_ns ?
						     now + slice :
						     deadline_ns);
		} else if (spinning) {
			cpu_relax();
		} else {
			futex_wait(&run->pending[table], pending,
				   (uint64_t)arm->slice_us *
					   BENCH_NS_PER_MICROSECOND);
		}
	}
	return atomic_load(&run->error);
}

/* After a step consumed its reads: winners go back; the next epoch is reused once its losers have all landed. */
void bench_reads_release(struct bench_run *run)
{
	struct read_epoch *const epoch = &run->epochs[run->epoch];
	for (uint32_t index = 0; index < epoch->count; index++) {
		struct attempt *const winner =
			atomic_load(&epoch->reads[index].winner);
		if (winner != NULL)
			slot_give(run, winner);
	}
	run->epoch = (run->epoch + 1u) % BENCH_EPOCHS;
	struct read_epoch *const next = &run->epochs[run->epoch];
	const uint64_t deadline = bench_now() + BENCH_READ_LIMIT_NS;
	while (atomic_load(&next->live) != 0 && bench_now() < deadline &&
	       atomic_load(&run->error) == 0) {
		if (run->arm->io_threads == 0)
			(void)owner_progress(&run->owners[0], true,
					     bench_now() + IDLE_SLICE_NS);
		else
			cpu_relax();
	}
	next->count = 0;
	for (uint32_t table = 0; table < BENCH_TABLES_MOST; table++)
		atomic_store(&run->pending[table], 0);
}

/* ---- NVMe passthrough: the extent map and the ioctl ------------------------------------------------------ */

struct lba_extent {
	uint64_t logical, physical, length;
};
struct lba_map {
	struct lba_extent *extents;
	uint32_t count;
	uint64_t partition_start_bytes;
};

static bool sysfs_number(const char *path, uint64_t *value)
{
	FILE *const input = fopen(path, "re");
	if (input == NULL)
		return false;
	unsigned long long parsed = 0;
	const bool read = fscanf(input, "%llu", &parsed) == 1;
	(void)fclose(input);
	*value = parsed;
	return read;
}

/* Maps a copy's file to device blocks with FIEMAP and opens its NVMe generic character device for reads. Only the
 * NVMe Read opcode is ever built. A stale map gives a digest mismatch, which falls back to another read. */
/* One copy's NVMe generic character device and extent map. */
static int passthrough_copy(struct bench_store *store, uint32_t copy)
{
	struct stat file;
	if (fstat(store->direct[copy], &file) != 0)
		return errno;
	char link[PATH_MAX], resolved[PATH_MAX], path[PATH_MAX];
	(void)snprintf(link, sizeof(link), "/sys/dev/block/%u:%u",
		       major(file.st_dev), minor(file.st_dev));
	if (realpath(link, resolved) == NULL)
		return errno;
	uint64_t start_sectors = 0;
	(void)snprintf(path, sizeof(path), "%s/start", resolved);
	if (!sysfs_number(path, &start_sectors))
		start_sectors = 0; /* a whole-disk file system */
	const char *const disk = store->device_names[copy];
	uint64_t logical_bytes = 0;
	(void)snprintf(path, sizeof(path),
		       "/sys/block/%s/queue/logical_block_size", disk);
	if (!sysfs_number(path, &logical_bytes) ||
	    logical_bytes < NVME_LOGICAL_BLOCK_BYTES_LEAST ||
	    BENCH_BLOCK_BYTES % logical_bytes != 0)
		return EINVAL;
	store->logical_block_bytes[copy] = (uint32_t)logical_bytes;
	if (strncmp(disk, "nvme", strlen("nvme")) != 0)
		return ENODEV;
	(void)snprintf(path, sizeof(path), "/dev/ng%s", disk + strlen("nvme"));
	const int descriptor = open(path, O_RDONLY | O_CLOEXEC);
	if (descriptor < 0) {
		const int error = errno;
		if (store->passthrough_refusal[0] == '\0')
			(void)snprintf(store->passthrough_refusal,
				       sizeof(store->passthrough_refusal),
				       "open %s: %s", path,
				       strerrorname_np(error) != NULL ?
					       strerrorname_np(error) :
					       "unknown");
		return error;
	}
	store->passthrough[copy] = descriptor;
	const int namespace = ioctl(descriptor, NVME_IOCTL_ID);
	if (namespace <= 0)
		return errno != 0 ? errno : EINVAL;
	store->passthrough_namespace[copy] = (uint32_t)namespace;
	struct fiemap probe = { .fm_start = 0,
				.fm_length = FIEMAP_MAX_OFFSET,
				.fm_flags = FIEMAP_FLAG_SYNC };
	if (ioctl(store->direct[copy], FS_IOC_FIEMAP, &probe) != 0)
		return errno;
	const uint32_t extents = probe.fm_mapped_extents;
	struct fiemap *const map =
		calloc(1, sizeof(*map) + (size_t)extents *
						 sizeof(struct fiemap_extent));
	struct lba_map *const lba = calloc(1, sizeof(*lba));
	if (map == NULL || lba == NULL) {
		free(map);
		free(lba);
		return ENOMEM;
	}
	*map = (struct fiemap){ .fm_start = 0,
				.fm_length = FIEMAP_MAX_OFFSET,
				.fm_flags = FIEMAP_FLAG_SYNC,
				.fm_extent_count = extents };
	if (ioctl(store->direct[copy], FS_IOC_FIEMAP, map) != 0) {
		const int error = errno;
		free(map);
		free(lba);
		return error;
	}
	lba->extents = calloc(
		map->fm_mapped_extents != 0 ? map->fm_mapped_extents : 1u,
		sizeof(*lba->extents));
	lba->partition_start_bytes = start_sectors * SECTOR_BYTES;
	for (uint32_t index = 0;
	     lba->extents != NULL && index < map->fm_mapped_extents; index++) {
		const struct fiemap_extent *const extent =
			&map->fm_extents[index];
		if ((extent->fe_flags & FIEMAP_REFUSED_FLAGS) != 0)
			continue; /* not a plain mapped extent: reads of it fall back */
		lba->extents[lba->count++] =
			(struct lba_extent){ .logical = extent->fe_logical,
					     .physical = extent->fe_physical,
					     .length = extent->fe_length };
	}
	free(map);
	if (lba->extents == NULL) {
		free(lba);
		return ENOMEM;
	}
	store->maps[copy] = lba;

	return 0;
}

/* Maps every copy it can: a copy without passthrough (no read access to its /dev/ng node) keeps no map, and the
 * first refusal is returned and named in passthrough_refusal. */
int bench_passthrough_prepare(struct bench_store *store)
{
	int first = 0;
	for (uint32_t copy = 0; copy < store->copy_count; copy++) {
		const int error = passthrough_copy(store, copy);
		first = first != 0 ? first : error;
	}
	return first;
}

int bench_lba_of(const struct bench_store *store, uint32_t copy,
		 uint64_t offset, uint64_t *block_address)
{
	const struct lba_map *const map = store->maps[copy];
	if (map == NULL)
		return ENODEV;
	uint32_t low = 0, high = map->count;
	while (low < high) {
		const uint32_t middle = low + (high - low) / 2u;
		if (map->extents[middle].logical +
			    map->extents[middle].length <=
		    offset)
			low = middle + 1u;
		else
			high = middle;
	}
	if (low == map->count || map->extents[low].logical > offset ||
	    offset + BENCH_BLOCK_BYTES >
		    map->extents[low].logical + map->extents[low].length)
		return ENODATA;
	*block_address =
		(map->partition_start_bytes + map->extents[low].physical +
		 offset - map->extents[low].logical) /
		store->logical_block_bytes[copy];
	return 0;
}

int bench_passthrough_read(const struct bench_store *store, uint32_t copy,
			   uint64_t offset, void *buffer)
{
	if (buffer == NULL)
		return 0;
	uint64_t block_address = 0;
	const int mapped = bench_lba_of(store, copy, offset, &block_address);
	if (mapped != 0)
		return mapped;
	struct nvme_passthru_cmd command = {
		.opcode = NVME_READ_OPCODE,
		.nsid = store->passthrough_namespace[copy],
		.addr = (uint64_t)(uintptr_t)buffer,
		.data_len = BENCH_BLOCK_BYTES,
		.cdw10 = (uint32_t)block_address,
		.cdw11 = (uint32_t)(block_address >> NVME_HIGH_SHIFT),
		.cdw12 = BENCH_BLOCK_BYTES / store->logical_block_bytes[copy] -
			 1u,
	};
	const int result =
		ioctl(store->passthrough[copy], NVME_IOCTL_IO_CMD, &command);
	return result == 0 ? 0 : result > 0 ? EIO : errno;
}
