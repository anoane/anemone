/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * engram_io_bench.h: the Engram I/O bench, a measurement probe that is never linked into the engine.
 *
 * The driver (engram_io_bench.c) replays real token traces in the serving step structure (decode steps of C rows,
 * prefill chunks), reads every n-gram block a step needs through one arm (interface, setup flags, completion,
 * threads, pinning, priority, copy policy, depth, batching, early issue, cache, hot set, RAM split) and checks every
 * block against its pack digest, with fallback to another copy, exactly as serving must. The read machinery is in
 * engram_io_engines.c; device characterisation (sequential rate, random IOPS) in engram_io_characterise.c.
 */
#ifndef ENGRAM_IO_BENCH_H
#define ENGRAM_IO_BENCH_H

#include <limits.h>
#include <linux/aio_abi.h>
#include <linux/io_uring.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/uio.h>

#include "anemone/engram.h"

/* Older uapi headers (Debian 13 ships Linux 6.12's) lack these; the values are the upstream uapi's. A kernel without
 * them refuses the mode at run time (EINVAL at setup or in the completion), reported as unsupported on this kernel. */
#ifndef IORING_SETUP_HYBRID_IOPOLL
#define IORING_SETUP_HYBRID_IOPOLL (1U << 17) /* Linux 6.13 */
#endif
/* IORING_OP_READV_FIXED is an enumerator (no #ifdef can test it): its upstream value, Linux 6.15 */
#define BENCH_OP_READV_FIXED 60u
#define BENCH_UNSUPPORTED ": unsupported on this kernel"
#define BENCH_COPIES_MOST 4u
#define BENCH_BLOCK_BYTES ((uint32_t)PACK_BLOCK_BYTES)
#define BENCH_TABLES_MOST ENGRAM_MAXIMUM_TABLES
#define BENCH_CASES_MOST 16u
#define BENCH_OWNERS_MOST 16u
#define BENCH_POOL_THREADS_MOST 64u
#define BENCH_CHUNK_TOKENS_MOST 1024u
#define BENCH_LEVELS 4u
#define BENCH_NS_PER_SECOND 1000000000ull
#define BENCH_NS_PER_MICROSECOND 1000ull
#define BENCH_NS_PER_MILLISECOND 1000000ull
#define BENCH_READ_LIMIT_NS \
	(180ull * BENCH_NS_PER_SECOND) /* STORAGE_READ_LIMIT_SECONDS */
#define BENCH_TEXT_BYTES 512u
#define BENCH_REASON_BYTES 160u
#define BENCH_HISTOGRAM_BUCKETS \
	512u /* log-linear: 16 sub-buckets per power of two of nanoseconds */
#define BENCH_HISTOGRAM_SUB_BITS 4u

enum engine_kind {
	ENGINE_URING,
	ENGINE_AIO,
	ENGINE_POOL,
	ENGINE_SYNC,
	ENGINE_MMAP,
	ENGINE_PROD,
	ENGINE_KINDS
};
enum uring_operation {
	OPERATION_READ,
	OPERATION_READV,
	OPERATION_READ_FIXED,
	OPERATION_READV_FIXED,
	OPERATION_URING_CMD,
	OPERATION_KINDS
};
enum pool_call { CALL_PREAD, CALL_PREADV2, CALL_IOCTL, CALL_KINDS };
enum batch_kind { BATCH_CALL, BATCH_TABLE, BATCH_STEP, BATCH_KINDS };
enum wait_kind { WAIT_POLL, WAIT_ENTER, WAIT_SPIN, WAIT_HYBRID, WAIT_KINDS };
enum copy_policy {
	COPY_FIRST,
	COPY_TICK,
	COPY_BLOCK,
	COPY_JSQ,
	COPY_P2C,
	COPY_RATE,
	COPY_KINDS
};
enum pin_kind { PIN_NONE, PIN_IRQ, PIN_LIST, PIN_KINDS };
enum hot_unit { HOT_BLOCK, HOT_ROW, HOT_UNITS };
enum hot_profile { PROFILE_ALL, PROFILE_OTHER, PROFILES };
enum scheduling_kind {
	SCHEDULE_KEEP,
	SCHEDULE_OTHER,
	SCHEDULE_BATCH,
	SCHEDULE_IDLE,
	SCHEDULE_FIFO,
	SCHEDULE_RR,
	SCHEDULE_KINDS
};
enum io_priority_kind {
	IO_PRIORITY_KEEP,
	IO_PRIORITY_RT,
	IO_PRIORITY_BE,
	IO_PRIORITY_IDLE,
	IO_PRIORITY_KINDS
};

/* One arm: everything a measurement varies. Parsed from "name key=value ..." (engram_io_bench.c). */
struct arm {
	char name[BENCH_REASON_BYTES];
	char text[BENCH_TEXT_BYTES];
	enum engine_kind engine;
	/* io_uring */
	uint32_t setup_flags, operation, depth, cq_factor, sq_cpu, sq_idle_ms;
	bool registered_files, registered_ring, sqe_async, per_device_rings;
	/* per-copy ring kinds: IOPOLL only on rings whose copies are all in poll_copy_mask (another ring keeps the
	 * other setup flags, with deferred task work under a single issuer), NVMe commands only to the copies of
	 * command_copy_mask (another copy is read with READ_FIXED, or READ without fixed buffers) */
	uint32_t poll_copy_mask, command_copy_mask;
	/* completion */
	enum wait_kind wait;
	uint32_t spin_us, slice_us;
	/* threads */
	uint32_t io_threads, pool_threads;
	enum pool_call call;
	uint32_t rw_flags;
	bool pool_spin, buffered, fadvise_random, readahead, willneed;
	/* reads */
	enum batch_kind batch;
	bool dedupe;
	enum copy_policy copy;
	uint32_t copy_mask, weights[BENCH_COPIES_MOST], hedge_us, inflight_most;
	/* hedging relative to the copy's running p99 of small steps' reads, and only for steps of at most
	 * hedge_step_most reads (decode, never prefill): the deadline is the larger of hedge_us and hedge_percent % of
	 * that p99 */
	uint32_t hedge_percent, hedge_step_most;
	uint32_t lead_us[BENCH_TABLES_MOST];
	/* RAM: dynamic cache, hot set, RAM prefix, all RAM */
	uint64_t cache_bytes, hot_bytes;
	enum hot_unit hot_unit;
	enum hot_profile hot_profile;
	uint32_t ram_percent;
	bool all_ram;
	/* a hot set from a file (engram_hot.h, made by hot_build from another corpus) in place of the profile: its
	 * name in the run's --hot_directory and its recorded SHA-256 */
	char hot_file[BENCH_REASON_BYTES], hot_sha256[BENCH_REASON_BYTES];
	/* placement and priority, for the step thread and every I/O thread */
	enum pin_kind pin;
	int32_t step_cpu;
	uint32_t pin_cpus[BENCH_OWNERS_MOST], pin_cpu_count;
	int32_t nice;
	enum scheduling_kind scheduling;
	uint32_t rt_priority;
	enum io_priority_kind io_priority;
	uint32_t io_priority_level;
	/* workload */
	uint32_t prefill_chunk, levels_mask, prefill;
};

/* The copies and the n-gram tables: every copy holds the same bytes at the same offsets. */
struct bench_table {
	uint64_t byte_offset, block_count, row_count, extent_digest;
	uint64_t *block_digests;
};
struct lba_map; /* file offset to device block, for NVMe passthrough (engram_io_engines.c) */
struct bench_store {
	uint32_t copy_count, table_count, rows_per_token;
	uint64_t file_bytes;
	char paths[BENCH_COPIES_MOST][PATH_MAX];
	char device_names[BENCH_COPIES_MOST]
			 [BENCH_REASON_BYTES]; /* the whole drive, e.g. nvme0n1 */
	int direct[BENCH_COPIES_MOST], buffered[BENCH_COPIES_MOST],
		passthrough[BENCH_COPIES_MOST];
	uint32_t passthrough_namespace[BENCH_COPIES_MOST],
		logical_block_bytes[BENCH_COPIES_MOST],
		irq_cpu[BENCH_COPIES_MOST];
	struct lba_map *maps[BENCH_COPIES_MOST];
	char passthrough_refusal[BENCH_REASON_BYTES];
	struct bench_table tables[BENCH_TABLES_MOST];
};

/* One physical read of one block from one copy. Its index names its 4 KiB slot of the run's arena. */
struct logical_read;
struct attempt {
	struct logical_read *read;
	unsigned char *buffer;
	uint64_t issued_ns, offset;
	int64_t result; /* bytes delivered, or a negative errno */
	uint32_t index, copy;
	_Atomic uint32_t
		generation; /* changes whenever the slot goes back: names one use of it */
	bool hedge;
	struct iovec vector;
	struct iocb control;
};
enum read_state { READ_WAITING, READ_DONE, READ_FAILED };
/* One block a step needs. The first verified attempt wins; the rest are freed as they complete. */
struct logical_read {
	uint64_t block;
	uint32_t table, unique, epoch;
	_Atomic uint32_t state, live, tried, tries, hedged;
	struct attempt *_Atomic winner;
};

struct bench_run;
struct owner;
struct engine;
struct engine_operations {
	const char *name;
	int (*open)(
		struct engine *
			engine); /* in the owner's thread: 0, or an errno with engine->reason */
	void (*close)(struct engine *engine);
	uint32_t (*submit)(struct engine *engine,
			   struct attempt *const *attempts, uint32_t count);
	uint32_t (*reap)(struct engine *engine, struct attempt **completed,
			 uint32_t capacity, bool block, uint64_t deadline_ns);
	int (*poll_descriptor)(
		const struct engine
			*engine); /* -1: cannot be waited on with poll() */
};
struct engine {
	const struct engine_operations *operations;
	struct bench_run *run;
	struct owner *owner;
	uint32_t copy_mask, in_flight;
	int error;
	char reason[BENCH_REASON_BYTES];
	void *state;
	/* attempts routed to this engine and not yet accepted: a ring of attempt indices */
	uint32_t *pending, pending_head, pending_tail, pending_capacity;
};
/* A thread (or the step thread itself, inline) that owns engines: submits, reaps, verifies, retries, hedges. */
struct owner {
	struct bench_run *run;
	uint32_t index, copy_mask, engine_count,
		engine_of_copy[BENCH_COPIES_MOST];
	struct engine engines[BENCH_COPIES_MOST];
	pthread_mutex_t inbox_lock;
	uint32_t *inbox, inbox_count, inbox_capacity;
	_Atomic uint32_t inbox_signal;
	uint64_t *issued; /* the hedge FIFO: attempt index and generation, in issue order */
	uint32_t issued_head, issued_tail, issued_capacity;
	pthread_t thread;
	bool threaded;
	_Atomic uint32_t stop, ready;
	int open_error;
	uint64_t cpu_ns, voluntary_switches, involuntary_switches;
};

struct bench_histogram {
	_Atomic uint64_t buckets[BENCH_HISTOGRAM_BUCKETS];
};
struct bench_counters {
	_Atomic uint64_t reads, retries, failures, hedges, hedge_wins,
		nowait_again, references, unique_blocks, ram_hits, hot_hits,
		cache_hits, coalesced;
};
/* A step's reads. A step reuses an epoch only once every attempt of its earlier use has completed. */
#define BENCH_EPOCHS 4u
struct read_epoch {
	struct logical_read *reads;
	uint32_t count;
	_Atomic uint32_t live;
};
/* One arm at one concurrency level: the attempt arena, the owners, the per-copy accounting. */
struct bench_run {
	const struct bench_store *store;
	const struct arm *arm;
	struct attempt *attempts;
	unsigned char *arena;
	uint64_t arena_bytes;
	uint32_t attempt_count;
	pthread_mutex_t free_lock;
	uint32_t *free_slots, free_count;
	struct read_epoch epochs[BENCH_EPOCHS];
	uint32_t read_capacity, epoch;
	_Atomic uint32_t pending[BENCH_TABLES_MOST];
	struct owner owners[BENCH_OWNERS_MOST];
	uint32_t owner_count, owner_of_copy[BENCH_COPIES_MOST],
		owners_of_copy[BENCH_COPIES_MOST];
	_Atomic uint32_t in_flight[BENCH_COPIES_MOST],
		route_turn[BENCH_COPIES_MOST];
	_Atomic uint64_t pool_cpu_ns;
	_Atomic uint64_t latency_ewma_ns[BENCH_COPIES_MOST], tick, random_state;
	struct bench_counters counters;
	struct bench_histogram copy_latency[BENCH_COPIES_MOST],
		small_step_latency[BENCH_COPIES_MOST];
	_Atomic uint64_t hedge_deadline_ns[BENCH_COPIES_MOST],
		hedge_refreshed_ns;
	_Atomic uint64_t copy_reads[BENCH_COPIES_MOST];
	_Atomic int error;
	char reason[BENCH_REASON_BYTES];
	bool arena_quarantined;
	char refusals[BENCH_TEXT_BYTES];
	pthread_mutex_t refusal_lock;
};

/* engram_io_engines.c */
uint64_t bench_now(void);
uint64_t bench_thread_cpu_ns(void);
void bench_histogram_add(struct bench_histogram *histogram, uint64_t value_ns);
uint64_t bench_histogram_percentile(const struct bench_histogram *histogram,
				    uint32_t per_mille);
void bench_refusal(struct bench_run *run, const char *what, int error);
void bench_fail(struct bench_run *run, const char *reason, int error);
void bench_thread_attributes(struct bench_run *run, int32_t cpu);
int bench_run_open(struct bench_run *run, const struct bench_store *store,
		   const struct arm *arm, uint32_t reads_most);
void bench_run_close(struct bench_run *run);
struct logical_read *bench_read_add(struct bench_run *run, uint32_t table,
				    uint64_t block, uint32_t unique);
void bench_reads_issue(struct bench_run *run, uint32_t first, uint32_t count);
int bench_wait_table(struct bench_run *run, uint32_t table,
		     uint64_t deadline_ns);
void bench_reads_release(struct bench_run *run);
uint64_t bench_copy_offset(const struct bench_store *store, uint32_t table,
			   uint64_t block);
bool bench_block_verified(const struct bench_store *store, uint32_t table,
			  uint64_t block, const unsigned char *bytes);
int bench_passthrough_prepare(struct bench_store *store);
int bench_lba_of(const struct bench_store *store, uint32_t copy,
		 uint64_t offset, uint64_t *block_address);
int bench_passthrough_read(const struct bench_store *store, uint32_t copy,
			   uint64_t offset, void *buffer);

/* engram_io_characterise.c */
/* done_path: an earlier output whose "ok" points are skipped (NULL: none) */
int bench_characterise(const struct bench_store *store, const char *plan_path,
		       FILE *output, const char *done_path,
		       uint64_t deadline_ns);

#endif
