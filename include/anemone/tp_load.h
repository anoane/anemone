/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * Selected original tensor loading for a prepared TP layer. The existing extent/PP loader is unchanged.
 * Jobs name original pack tensors and fixed card offsets: FULL copies a whole tensor; SHARD copies one
 * tp_weights.h rectangle without decoding or requantization. Jobs are sorted by source extent, and their
 * destination ranges increase without overlap on each card. A selected HOST_ONLY unit is unsupported.
 * Preparation bounds every job against its source tensor and destination arena, validates shard records, and
 * refuses a selected extent above maximum_extent_byte_count or the hard workspace bound. It allocates nothing.
 * Plan/pack/jobs remain immutable and alive through run. The plan reports the two caller-owned workspace sizes.
 *
 * Run reads each selected extent once from the first copy whose original positional extent digest holds. A bad
 * copy is tried in turn; no selected tensor from an unchecked extent is uploaded. Its output workspace contains
 * the copied tensor or shard, zero-padded to eight bytes for a destination digest. The caller reserves that
 * padded range. Every completed upload is digested on its card and compared with the staged bytes' digest.
 * Reads remain under storage_seam.h's bound, and upload/digest waits use the request's nonzero bound.
 *
 * Both workspaces belong to the caller, are disjoint, and are allocated through backend staging_allocate before
 * run. The source workspace is aligned to BACKEND_STAGING_ALIGNMENT_BYTES. Run allocates/frees nothing.
 * A failed upload_wait does NOT prove retirement: the caller must drain/close the affected cards before reusing
 * or freeing the output workspace. Failure leaves result_out untouched and destination arena bytes unspecified.
 * This is a load-only primitive, not TP model admission or a replacement for the unchanged Engram host loader.
 *
 * worker_count 0 is the serial run above, one job after another (the reference). worker_count N runs the same
 * checks on N threads, the calling thread one of them: each takes the next selected extent in plan order, reads
 * and verifies it in its own source slot, stages the extent's jobs (each digested) in its own output slot, begins
 * their uploads, then waits for each, digests it on its card and compares. Each worker's two slots are staging
 * blocks of at most TP_LOAD_WORKER_SLOT_BYTES that run allocates and retires before it returns; an extent
 * or job larger than a slot uses the request's two workspaces, one such extent at a time. Destination ranges never
 * overlap, so every arena range receives the serial run's bytes. The result, card receipts included, equals the
 * serial run's; a failure returns one failing extent's or job's status (not necessarily the serial run's first).
 * A card receipt sums pack_digest_term(on-card digest, destination offset / word) over the card's jobs:
 * two loads with equal receipts placed the same bytes in every loaded range (within the digest's strength).
 */
#ifndef TP_LOAD_H
#define TP_LOAD_H

#include <pthread.h>
#include <stdint.h>

#include "anemone/backend_seam.h"
#include "anemone/pack.h"
#include "anemone/storage_seam.h"
#include "anemone/tp_weights.h"

#define TP_LOAD_MAXIMUM_EXTENT_BYTES (2ull << 30u)
#define TP_LOAD_MAXIMUM_WORKERS 64u
/* serving's host arena threads beside the cards' load (--tp_load_host_threads), and the load stage's (autotune) */
#define TP_LOAD_HOST_THREADS_DEFAULT 16u
/* serving's per-copy cap on the TP load's reads (--tp_load_reads_per_copy): none */
#define TP_LOAD_READS_PER_COPY_DEFAULT 0u
/* the copy splits' names (tp_load_split), as a load section records them */
extern const char *const tp_load_split_names[];
/* the EXL3 TP4 pack: median GPU extent 13.3 MB, 11 GB in larger ones */
#define TP_LOAD_WORKER_SLOT_BYTES (64ull << 20u)

/* Turbo load's choice of copy (storage_seam.h: the caller picks the copy of each read), shared by the two runs of
 * one load: the host arena's (role HOST) and the cards' (role CARDS). A pipelined run with copies reads each extent
 * (each chunk of a host extent) first from the copy, among those split lets its role use, that would finish it
 * soonest: the fewest bytes in flight per byte a second it has delivered while busy (a copy not yet measured counts
 * as fast as the fastest), a copy with a failed read last; then from the others in order. Every extent is still
 * checked against its digest; a host extent whose striped chunks fail or differ is read again copy by copy. The
 * counters are the copies' totals once the runs are done. NULL: copies in order (the reference).
 *
 * split (--tp_load_copy_split), with two or more copies (one copy serves both roles):
 *   0 SOONEST every copy serves both roles.
 *   1 FIXED   each role reads only its preferred copy, the host arena copy 0 and the cards the last copy (turbo on
 *             an EXL3 map: the map's contiguous native extents, a flat replica's EXL3 ones), while that copy has not
 *             failed a read: a middle copy is never read, and a role that finishes first leaves its drive idle.
 *   2 STEAL   (the default) each role prefers the same copy as under FIXED, held for it while that role has reads
 *             left to claim and neither role's preferred copy has failed a read; every other copy serves both. Once
 *             a role's run has claimed its last read, its copy serves the other role too, so no drive idles while
 *             the other role still reads (request 16, FIXED: copy 1 idle 9.3 s).
 * Each run marks its role claimed once, on the copies of one load: a load initialises its own copies. */
enum tp_load_role { TP_LOAD_ROLE_HOST, TP_LOAD_ROLE_CARDS, TP_LOAD_ROLES };
enum tp_load_split {
	TP_LOAD_SPLIT_SOONEST,
	TP_LOAD_SPLIT_FIXED,
	TP_LOAD_SPLIT_STEAL,
	TP_LOAD_SPLITS
};
struct tp_load_copies {
	pthread_mutex_t mutex; /* guards everything below while runs read */
	uint64_t in_flight_bytes[STORAGE_MAXIMUM_COPIES];
	uint64_t delivered_bytes[STORAGE_MAXIMUM_COPIES];
	uint64_t role_delivered_bytes
		[TP_LOAD_ROLES][STORAGE_MAXIMUM_COPIES]; /* the same, per role */
	uint64_t busy_nanoseconds
		[STORAGE_MAXIMUM_COPIES]; /* with a read in flight */
	uint64_t busy_since[STORAGE_MAXIMUM_COPIES];
	uint32_t reads_in_flight[STORAGE_MAXIMUM_COPIES];
	uint32_t host_reads_in_flight
		[STORAGE_MAXIMUM_COPIES]; /* the host role's share of reads_in_flight */
	uint32_t failed_reads[STORAGE_MAXIMUM_COPIES];
	uint32_t role_claimed
		[TP_LOAD_ROLES]; /* 1 once the role's run has claimed its last read */
	pthread_cond_t
		read_ended; /* a read waits while the copies it may use have reads_most in flight */
	uint32_t reads_most; /* reads in flight a copy at most, both roles' (--tp_load_reads_per_copy); 0: no limit */
	uint32_t host_reads_most; /* the host role's own reads in flight a copy at most (the load stage's readers,
				   * a load section's); the cards' reads never count against it; 0: no limit */
	uint32_t split; /* enum tp_load_split */
	/* Optional measured aggregate shares; zero keeps the original online delivery-rate estimator. */
	uint32_t tuned_weights[STORAGE_MAXIMUM_COPIES];
};
/* reads_most: a drive delivers most with a few large reads at once (RND_H.md, 3 Nov: two per 9100 PRO); host_reads_most
 * starts at 0 */
MUST_CHECK result_status tp_load_copies_init(struct tp_load_copies *copies,
					     uint32_t reads_most);
void tp_load_copies_destroy(struct tp_load_copies *copies);

enum tp_load_mode { TP_LOAD_FULL, TP_LOAD_SHARD, TP_LOAD_MODE_COUNT };

struct tp_load_job {
	uint32_t tensor_index;
	uint32_t card;
	uint64_t destination_offset;
	enum tp_load_mode mode;
	uint32_t reserved_zero;
	struct tensor_shard_plan shard;
};

struct tp_load_description {
	const struct pack_record *pack;
	const struct tp_load_job *jobs;
	uint32_t job_count;
	uint32_t card_count;
	uint64_t card_arena_byte_count[BACKEND_MAXIMUM_CARDS];
	uint64_t maximum_extent_byte_count;
};

struct tp_load_plan {
	struct tp_load_description description;
	uint64_t source_workspace_byte_count;
	uint64_t output_workspace_byte_count;
};

struct tp_load_request {
	const struct tp_load_plan *plan;
	const struct storage_operations_record *storage;
	const struct storage_reader *reader;
	const struct backend_operations *backend;
	struct device_card *const *cards;
	void *source_workspace;
	void *output_workspace;
	uint64_t source_workspace_byte_count;
	uint64_t output_workspace_byte_count;
	uint64_t wait_limit_nanoseconds;
	uint32_t worker_count; /* 0: the serial run; 1..TP_LOAD_MAXIMUM_WORKERS threads */
	struct tp_load_copies
		*copies; /* pipelined runs only; NULL: copies in order */
};

struct tp_load_result {
	uint64_t source_bytes_read;
	uint64_t uploaded_bytes;
	uint32_t extents_verified;
	uint32_t tensors_uploaded;
	uint64_t card_receipts[BACKEND_MAXIMUM_CARDS];
};

MUST_CHECK result_status
tp_load_prepare(const struct tp_load_description *description,
		struct tp_load_plan *plan_out);
MUST_CHECK result_status tp_load_run(const struct tp_load_request *request,
				     struct tp_load_result *result_out);

/* Independent load-only RAM path for every HOST_ONLY extent, including Engram/frontend tables. prepare writes
 * exactly extent_count offsets: selected extents concatenated in pack order, each whole/padded pack extent;
 * others TP_HOST_LOAD_NOWHERE. The plan reports the one caller-owned host arena's required size.
 * No allocation, backend call or full-extent workspace. Pack/map/plan remain immutable and alive through run.
 * Run checks the reader's pack size, copy count and block alignment, and reads directly into that arena in at
 * most chunk_bytes per call (0: TP_HOST_LOAD_CHUNK_BYTES; else whole pack blocks, at most
 * TP_HOST_LOAD_CHUNK_BYTES_MOST). Its address must meet the reader's alignment (pack-block alignment always suffices).
 * Every chunk digest uses word indices from the extent's beginning. A read failure or digest mismatch restarts
 * the whole extent from the next copy, in order; no extent is accepted by mixing chunks from different copies.
 * All counters include retries; requested_bytes includes failed reads, whose delivered byte count is unknown.
 * Empty host selections accept NULL/zero arena. No source extent size cap; every address/sum/map is checked.
 * Outputs are disjoint from input metadata/each other; on any failure prepare/run outputs stay untouched, arena
 * bytes unspecified. Only a successful run permits Engram attachment. Caller allocates/frees RAM once at load;
 * no caller of this primitive or storage read belongs in the token path. Existing PP/GPU loaders are unchanged.
 * worker_count 0 reads the extents one chunk after another (the reference); N reads each extent's chunks on N
 * threads, the calling thread one of them, each chunk digested with word indices from its extent's start (a sum: any
 * order gives the same digest). Copies are tried per extent in order, as above; on success every counter equals the
 * serial run's, on failure requested_bytes/read_calls may include chunks other threads read meanwhile.
 */
#define TP_HOST_LOAD_CHUNK_BYTES \
	(16ull << 20u) /* one read, the chunk unless a request names another */
#define TP_HOST_LOAD_CHUNK_BYTES_MOST (64ull << 20u)
#define TP_HOST_LOAD_NOWHERE UINT64_MAX
struct tp_host_load_plan {
	const struct pack_record *pack;
	const uint64_t *extent_offsets;
	uint64_t host_byte_count;
	uint32_t extent_count;
	uint32_t host_extent_count;
};
struct tp_host_load_request {
	const struct tp_host_load_plan *plan;
	const uint8_t *
		external_host_extents; /* optional extent_count bytes; externally owned host-only extents */
	const struct storage_operations_record *storage;
	const struct storage_reader *reader;
	void *host_arena;
	uint64_t host_arena_byte_count;
	uint32_t worker_count; /* 0: the serial run; 1..TP_LOAD_MAXIMUM_WORKERS threads */
	struct tp_load_copies
		*copies; /* pipelined runs only; NULL: copies in order */
	uint64_t chunk_bytes; /* one read: 0 is TP_HOST_LOAD_CHUNK_BYTES */
};
struct tp_host_load_result {
	uint64_t requested_bytes;
	uint64_t read_calls;
	uint64_t copy_retries;
	uint32_t extents_verified;
	uint32_t reserved_zero;
};
MUST_CHECK result_status
tp_host_load_prepare(const struct pack_record *pack, uint64_t *extent_offsets,
		     uint32_t extent_count, struct tp_host_load_plan *plan_out);
MUST_CHECK result_status
tp_host_load_run(const struct tp_host_load_request *request,
		 struct tp_host_load_result *result_out);

/* The TP host arena (serving's, the load stage's points): byte_count bytes aligned to TP_HOST_ARENA_PAGE_BYTES, a
 * whole number of them. With external tables (loader.h) an anonymous MAP_NORESERVE mapping one page longer, the arena
 * aligned inside it; without, posix_memalign's, as before. The caller advises its pages. A byte_count of 0 is an empty
 * arena (no allocation); STATUS_OUT_OF_MEMORY leaves it empty; tp_host_arena_release frees either kind and empties
 * it (an empty arena too). */
#define TP_HOST_ARENA_PAGE_BYTES \
	(2ull << 20u) /* 2 MiB: small pages cost a TLB walk a row */
struct tp_host_arena {
	void *bytes; /* NULL: none */
	void *mapping; /* the NORESERVE mapping that holds bytes; NULL: posix_memalign's */
	size_t mapping_bytes, byte_count;
};
MUST_CHECK result_status tp_host_arena_allocate(uint64_t byte_count,
						bool external_tables,
						struct tp_host_arena *arena);
void tp_host_arena_release(struct tp_host_arena *arena);

#endif
