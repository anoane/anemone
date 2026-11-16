/* SPDX-License-Identifier: AGPL-3.0-or-later */
#ifndef REPLICA_PREPARE_H
#define REPLICA_PREPARE_H
#include "storage_prepare.h"
#include <signal.h>
#define PREPARE_TARGETS_MOST (STORAGE_MAXIMUM_COPIES - 1u)
/* The copy's worker threads, one pool shared by all targets: each takes one chunk at a time, reads and hashes it once
 * and writes it to every target, then flushes, reads back and compares it on each (k targets: 1 read + 1 hash + k
 * writes + k readbacks a chunk), so a copy runs at workers chunks over the sum of those stages until a drive binds
 * (S1, 14 Nov 2026: 4 workers 96 % busy, one SHA-256 stream a chunk 42 % of their time). The default is request 25b's
 * winner (15 Nov 2026, 96 GiB from nvme_8tb, A D D A against 4): one target 4.993 GB/s against 3.617 (x1.38), two
 * targets 4.545 against 2.773 (x1.64), the full re-verification about 12.3 GB/s against 5.4; the same bytes and
 * digests at any count. A copy-stage parameter, `anemone autotune --copy_workers` (4: the earlier default). */
#define PREPARE_COPY_WORKERS_DEFAULT 24u
#define PREPARE_COPY_WORKERS_MOST 32u
/* Two-chunk source hashing (prepare_copy_options.pair_source_hashes), on by default: request 27 (15 Nov 2026, 96 GiB
 * from nvme_8tb to one target at 24 workers, A D D A) copied at 4.927 and 5.024 GB/s paired against 4.843 and 4.996
 * with one hash a chunk (+1.14 %), with the same manifest digests (R23: two streams run x1.54 a thread).
 * `anemone autotune --copy_hash_pairs=0` hashes one chunk at a time (the earlier default). */
#define PREPARE_PAIR_SOURCE_HASHES_DEFAULT true
#define PREPARE_PARTIAL_SUFFIX \
	".partial" /* autotune's resumable checkpoint beside its target */
struct prepare_extent_map {
	uint64_t count, largest;
	uint64_t fragments; /* runs of physically and logically contiguous extents */
	uint64_t next_logical, next_physical; /* where the last extent ended */
	bool fragmented, needs_allocation;
};
/* One FIEMAP extent into the map; then whether a mapped file of bytes counts as fragmented. Filesystems cap one
 * extent (ext4 and btrfs at 128 MiB), so contiguous extents join one fragment and only fragments count. */
void prepare_extent_record(struct prepare_extent_map *map, uint64_t logical,
			   uint64_t physical, uint64_t length, uint32_t flags);
bool prepare_extents_fragmented(const struct prepare_extent_map *map,
				uint64_t bytes);
/* Fields in clang-analyzer-optin.performance.Padding's optimal order (4 padding bytes, not 12). */
struct prepare_target {
	const char *reason;
	uint64_t written, verified,
		trusted; /* a final's bytes its matching receipt vouches for: neither read nor spot-checked */
	double seconds;
	struct prepare_extent_map extents;
	struct local_device_description device;
	bool included, complete, resumable;
	char key[STORAGE_PROFILE_KEY_BYTES], path[PATH_MAX];
};
struct prepare_copy_options {
	const char *manifest_path;
	struct prepare_manifest *manifest;
	struct prepare_target *targets;
	uint32_t count, floor_percent, workers;
	bool external;
	uint64_t deadline_ns;
	const volatile sig_atomic_t *stop;
	uint64_t *source_bytes_read;
	/* A final whose verification receipt matches is spot-checked (spot_chunks chunks, 0: PREPARE_SPOT_CHUNKS,
	 * chosen by spot_seed, or at random when it is 0) instead of read in full; verify_full reads every final in
	 * full. */
	bool verify_full;
	uint32_t spot_chunks;
	uint64_t spot_seed;
	/* bytes written to a target after a chunk before its readback; 0: COPY_READBACK_LAG_BYTES (8 GiB). A test's
	 * small fixture uses a chunk, so that some readbacks pass their lag. */
	uint64_t readback_lag_bytes;
	/* Two-chunk source hashing: a worker keeps one chunk whose readbacks matched until it has a second of its
	 * length, then hashes both in one interleaved pass (sha256_of_two, the same digests); one more retained buffer
	 * a worker. `anemone autotune --copy_hash_pairs`. */
	bool pair_source_hashes;
};
#define PREPARE_SPOT_CHUNKS 4u
#define PREPARE_RECEIPT_SUFFIX \
	".verified" /* beside a final: what its last full verification saw */
/* The chunks a spot check reads: every chunk when there are at most wanted (0 or more than PREPARE_SPOT_CHUNKS:
 * PREPARE_SPOT_CHUNKS), else that many distinct ones from the seed. Returns their count. */
uint32_t prepare_spot_chunks(uint64_t seed, uint32_t chunks, uint32_t wanted,
			     uint32_t output[PREPARE_SPOT_CHUNKS]);
MUST_CHECK result_status prepare_direct_alignment(int descriptor,
						  uint32_t *alignment);
MUST_CHECK result_status prepare_extents(int descriptor,
					 struct prepare_extent_map *output);
/* What a run would do with the checkpoint beside a target (8 Nov 2026; r17: the same order of checks). */
enum prepare_checkpoint_kind {
	PREPARE_CHECKPOINT_NONE, /* no partial: a run starts a fresh copy, or verifies the final beside it */
	/* its engine journal matches the pack, the hardware key and the file: resumed; removed only by the lock holder
	 * for a space, read-only or policy refusal, or once a final beside it verifies every byte, and never if
	 * adopted */
	PREPARE_CHECKPOINT_ENGINE,
	/* no engine journal, but quiet and its leading chunks byte-equal to the source: adopted (a journal is written,
	 * recording the adoption), then resumed; an adopted checkpoint is never removed */
	PREPARE_CHECKPOINT_ADOPTABLE,
	/* no engine journal and unproven: refused as foreign_partial_kept, never removed or changed */
	PREPARE_CHECKPOINT_FOREIGN,
	/* a run cannot take the target's <name>.copy-lock (another copy holds it, or it is no owned file): it refuses
	 * the target as copy_in_progress and examines nothing */
	PREPARE_CHECKPOINT_HELD,
	/* the directory or the partial cannot be read (error), or the directory is refused by policy: a run refuses the
	 * target and touches nothing */
	PREPARE_CHECKPOINT_UNKNOWN,
	PREPARE_CHECKPOINT_KINDS
};
struct prepare_checkpoint {
	enum prepare_checkpoint_kind kind;
	bool beside_final,
		receipt, /* the final beside it matches its verification receipt: a run spot-checks it */
		adopted, /* adopted: an engine checkpoint adopted from a foreign partial */
		removed; /* a run removes this engine checkpoint with its journal for the refusal */
	uint64_t bytes, allocated, proven_bytes;
	/* the refusal a run raises before it adopts or resumes (a final it cannot verify, space, read-only, directory
	 * policy); NULL: none */
	const char *refusal;
	int error; /* PREPARE_CHECKPOINT_UNKNOWN: the errno, or 0 with a refusal */
	const char *
		why; /* foreign or held: why it is not adopted or examined; otherwise "" */
};
/* Classifies the checkpoint beside target_path for a run on the device whose hardware key is key, in the run's order:
 * the directory, the copy lock and its writability, the final, the copier's lock, the journal, the space
 * (floor_percent), the proof. It
 * only reads: the locks are probed shared and released at once, nothing is created, and the partial's leading chunks
 * are read with O_NOATIME. */
MUST_CHECK result_status prepare_checkpoint_classify(
	const struct prepare_manifest *manifest, const char *target_path,
	const char *key, uint32_t floor_percent,
	struct prepare_checkpoint *output);
/* A device's available bytes with the credit of an engine checkpoint beside a final (removed once the final
 * verifies), at most its total. */
uint64_t prepare_available_with_credit(uint64_t total, uint64_t available,
				       uint64_t credit);
/* Targets are owned regular files under anemone_engram_rnd. Every resumed/final chunk is rehashed.
 * At most four64MiB source buffers and four readback buffers; bounded fan-out backpressure.
 * Timeout preserves only explicitly recorded resumable partials. No source cache advice or GPU call. */
MUST_CHECK result_status
prepare_replicas(const struct prepare_copy_options *options);
/* Whether the engine journal beside target_path claims chunk (a readback after its lag proved it), and how many it
 * claims; STATUS_INPUT_OUTPUT_ERROR when no journal of this format reads there. Only reads (for tests and reports). */
MUST_CHECK result_status prepare_journal_claims(const char *target_path,
						uint32_t chunk, bool *claimed,
						uint32_t *claimed_count);
#endif
