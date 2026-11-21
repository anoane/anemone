/* SPDX-License-Identifier: AGPL-3.0-or-later */
#include "replica_prepare.h"
#include "common/hooks.h"
#include "replicas.h"
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <linux/fiemap.h>
#include <linux/fs.h>
#include <linux/io_uring.h>
#include <poll.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/random.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <time.h>
#include <unistd.h>
#define COPY_RING_DEPTH 64u
#define COPY_PIECE_BYTES (8u << 20u)
#define COPY_ALIGNMENT_MOST (1u << 20u)
#define COPY_POLL_MS 1000
#define COPY_DRAIN_SECONDS 5u
#define COPY_STOP_DRAIN_SECONDS \
	10u /* after a stop, the written chunks still waiting are read back (inside the lag) for at most this */
/* 2: records adoption; 3: bound to the hardware key and inode, not the device number a reboot may renumber */
#define COPY_STATE_SCHEMA 3u
#define COPY_BITMAP_BYTES ((PREPARE_CHUNKS_MOST + 7u) / 8u)
/* a journal's fields before its bitmap: schema and chunks; bytes, chunk_bytes, inode and adopted; key and pack */
#define COPY_STATE_WORDS 2u
#define COPY_STATE_COUNTS 4u
#define COPY_STATE_NAMES 2u
#define COPY_BITS 8u
#define ALLOCATED_BLOCK_BYTES 512u
#define EXTENT_PAGE 128u
#define EXTENTS_MOST 65536u
#define FRAGMENT_BASE 128u
#define GIB_BYTES (1ull << 30u)
#define NS_SECOND 1000000000ull
#define DECIMAL_GB 1000000000.0
#define MANIFEST_CHECKPOINT_CHUNKS 16u
/* A target's journal is saved once this many of its chunks are claimed and unsaved (1 GiB of 64 MiB chunks), and at
 * the run's end: one journal commit an epoch, not a chunk. */
#define JOURNAL_EPOCH_CHUNKS 16u
/* The readback lag: a written chunk is read back only once this many more bytes were written to its target after it,
 * and behind an fdatasync, so that the read comes from the flash and not the drive's DRAM cache (8 GB on the 9100 PRO
 * 8 TB, 4 GB on the 9100 PRO 4 TB and the 990 PRO). Its source buffer is retained until then: the run holds the lag
 * plus one chunk a worker and one more of source buffers, for every target at once. The run's last chunks, and the
 * oldest when every buffer waits (a resumed target written only in places), are read back inside the lag, after the
 * flush; the copy's report counts them. */
#define COPY_READBACK_LAG_BYTES (8ull << 30u)
/* MemAvailable must exceed the retained source buffers by this much, or the copy is refused */
#define COPY_MEMORY_RESERVE_BYTES (8ull << 30u)
#define COPY_WAIT_NANOSECONDS \
	50000000L /* a waiting worker looks at the deadline this often */
#define MEMINFO_LINE_BYTES 256u
#define KIB_BYTES 1024u
/* The leading chunks an adoption proves at most: 1 GiB of 64 MiB chunks. */
#define ADOPTION_PROOF_CHUNKS 16u
/* A foreign partial written more recently may still have a writer. */
#define ADOPTION_QUIET_SECONDS 900
#define COPY_STATE_SUFFIX ".copy-state"
#define COPY_LOCK_SUFFIX ".copy-lock"
/* The earlier Python copier's lock beside its target: ".<name>.copy.lock" (tools/storage/manager.py). */
#define COPIER_LOCK_PREFIX "."
#define COPIER_LOCK_SUFFIX ".copy.lock"
#define PROC_DESCRIPTOR_BYTES 64u
/* 2: the hardware key in place of the device number */
#define RECEIPT_SCHEMA 2u
#define RECEIPT_BYTES 1024u
#define RECEIPT_TEMPORARY ".verified.new"
#define SEARCH_RECEIPT_TEMPORARY \
	REPLICA_RECEIPT_SUFFIX   \
		".new" /* the search's receipt before its rename */
#define SEARCH_RECEIPT_BYTES \
	(PATH_MAX +          \
	 256u) /* replicas.h's receipt: the source's whole real path and two numbers */
#define HEADER_DIGEST_DIGITS \
	16u /* a pack identity that is a header digest: 16 hexadecimal digits */
#define HEXADECIMAL 16
#define SPLITMIX_INCREMENT 0x9e3779b97f4a7c15ull
#define SPLITMIX_FIRST 0xbf58476d1ce4e5b9ull
#define SPLITMIX_SECOND 0x94d049bb133111ebull
#define SPLITMIX_SHIFT_FIRST 30u
#define SPLITMIX_SHIFT_SECOND 27u
#define SPLITMIX_SHIFT_THIRD 31u
/* An exclusive lock is tried this often, this far apart, before another holder is assumed: a --plan probe holds its
 * shared lock for microseconds. */
#define LOCK_TRIES 3u
#define LOCK_RETRY_NS 1000000L
/* futimens: the access time, then the modification time */
#define FILE_TIMES 2u

struct copy_ring {
	int descriptor;
	void *submission, *completion;
	struct io_uring_sqe *entries;
	size_t submission_bytes, completion_bytes, entries_bytes;
	unsigned *head, *tail, *mask, *array, *completion_head,
		*completion_tail, *completion_mask;
	struct io_uring_cqe *completed;
	bool poisoned,
		stop_poisoned; /* stop_poisoned: by the stop drain's limit, the run's stop and no fault of it */
};
/* A journal names its partial by the device's hardware key and the partial's inode, never by the device number: NVMe,
 * dm and btrfs device numbers follow probe order, so a reboot may renumber them. */
struct copy_state {
	uint32_t schema, chunks;
	/* adopted: the proven prefix bytes of a foreign partial the engine adopted (never removed); 0: created by it */
	uint64_t bytes, chunk_bytes, inode, adopted;
	char key[STORAGE_PROFILE_KEY_BYTES], pack[STORAGE_PROFILE_KEY_BYTES];
	unsigned char bitmap[COPY_BITMAP_BYTES];
};
/* journal_valid compares the fields before the bitmap as bytes: no padding among them */
_Static_assert(offsetof(struct copy_state, bitmap) ==
		       COPY_STATE_WORDS * sizeof(uint32_t) +
			       COPY_STATE_COUNTS * sizeof(uint64_t) +
			       (size_t)(COPY_STATE_NAMES *
					STORAGE_PROFILE_KEY_BYTES),
	       "copy_state has padding before its bitmap");
struct copy_target {
	/* copier_lock: the Python copier's .<target>.copy.lock, held for every target not yet final from just after
	 * <target>.copy-lock to the end of the run (copier_busy: the copier holds it) */
	int descriptor, lock, directory, copier_lock;
	/* locked: this process holds <target>.copy-lock; only then may it remove the partial or journal (lock_busy:
	 * another process holds it). engine: the partial is this engine's checkpoint (its journal matches, it was
	 * adopted, or this run created it); adopted: it came from a foreign partial, so it is never removed; created
	 * and journaled: this run created the partial at its name, and saved its journal; unnamed: it is created
	 * without a name (O_TMPFILE) and linked at the partial's name only after its journal is saved, so no crash
	 * leaves a partial without its journal. fresh: this run created the partial, so it holds only allocated zeros
	 * and every chunk is written without a read first. journal_examined: its journal was read; a refusal before
	 * that examines nothing. */
	bool fresh, finalized, failed, locked, lock_busy, copier_busy, engine,
		adopted, created, journaled, unnamed, journal_examined;
	/* receipt_trusted: a final whose verification receipt matches it; only its spot chunks are read.
	 * verification_failed: a final's bytes or size did not match, or it changed while the run verified it; only
	 * this removes its receipt (a refusal for any other reason keeps it). */
	bool receipt_trusted, verification_failed;
	/* the target as this run opened it (a final's before any verification read); a final's receipt binds it */
	struct stat opened;
	uint32_t spot[PREPARE_SPOT_CHUNKS], spot_count;
	/* the allocated bytes of an engine checkpoint beside a final: freed once the final verifies, so credited */
	uint64_t credit;
	char partial[PATH_MAX], journal[PATH_MAX];
	const char *name, *partial_name, *foreign_why;
	struct copy_state state;
	/* unsaved: chunks claimed in state.bitmap since the journal was last saved */
	uint32_t unsaved_count;
};
/* A chunk written and waiting for its readbacks: its source bytes, and where each target it went to had got to. */
struct retained_chunk {
	uint32_t chunk;
	bool hash_after, targets[PREPARE_TARGETS_MOST];
	size_t bytes;
	void *buffer;
	uint64_t written_at[PREPARE_TARGETS_MOST];
};
/* Where a copy's time goes, summed over its workers (each a busy time, so a stage's rate is its bytes over it). */
enum copy_stage {
	STAGE_SOURCE_READ,
	STAGE_SOURCE_HASH,
	STAGE_TARGET_READ, /* a resumed or final target's chunk, before any write */
	STAGE_TARGET_HASH,
	STAGE_WRITE,
	STAGE_FLUSH,
	STAGE_READBACK_READ,
	STAGE_READBACK_HASH,
	STAGE_READBACK_COMPARE, /* a readback against the source buffer, which is hashed after it */
	STAGE_JOURNAL,
	COPY_STAGES
};
static const char *const stage_names[COPY_STAGES] = {
	"source_read",	 "source_hash",	  "target_read",
	"target_hash",	 "write",	  "flush",
	"readback_read", "readback_hash", "readback_compare",
	"journal"
};
struct copy_shared {
	const struct prepare_copy_options *options;
	struct copy_target targets[PREPARE_TARGETS_MOST];
	int source;
	uint32_t next, alignment;
	bool failure, timed_out, manifest_changed;
	bool hard_failure; /* a source, digest, manifest or ring failure that is no stop's: the run is an I/O failure */
	bool fault_stop; /* checking builds: a stop after a chunk was written (prepare_copy_stop_written) */
	uint64_t stop_drain_ns; /* when the drain after a stop ends; 0 before a stop */
	uint64_t source_bytes;
	_Atomic uint64_t stage_ns[COPY_STAGES], stage_bytes[COPY_STAGES];
	/* The lagged readback (COPY_READBACK_LAG_BYTES): every source buffer (buffers), the free ones (free_buffers,
	 * free_count of them), the written chunks waiting for their readbacks in write order, the workers writing, each
	 * target's bytes written and covered by a completed fdatasync, and the readbacks taken inside the lag. */
	void **buffers, **free_buffers;
	struct retained_chunk *retained;
	uint32_t buffer_count, free_count, retained_count, writing, inside_lag,
		readbacks;
	uint64_t target_written[PREPARE_TARGETS_MOST],
		target_flushed[PREPARE_TARGETS_MOST], lag_bytes;
	pthread_mutex_t mutex;
	pthread_cond_t changed;
};
struct copy_worker {
	struct copy_shared *shared;
	struct copy_ring ring;
	void *verify_buffer;
	/* two-chunk source hashing: a chunk whose readbacks matched, its source hash waiting for a second chunk */
	struct retained_chunk pending;
	bool pending_good[PREPARE_TARGETS_MOST], pending_claim, has_pending;
};
/* Two-chunk source hashing: the chunks one pass hashes; so the buffers a worker gives back after a readback, its
 * chunk's and the pending one's. */
#define COPY_HASH_PAIR_CHUNKS 2u
#define COPY_RELEASED_MOST COPY_HASH_PAIR_CHUNKS
/* A timed-out ring owns its buffers until process exit; never free memory referenced by the kernel. */
static struct copy_worker *quarantined[PREPARE_COPY_WORKERS_MOST];
static uint32_t quarantine_count;

static uint64_t copy_now(void)
{
	struct timespec time;
	return clock_gettime(CLOCK_MONOTONIC, &time) == 0 ?
		       (uint64_t)time.tv_sec * NS_SECOND +
			       (uint64_t)time.tv_nsec :
		       UINT64_MAX;
}
static bool copy_stopped(const struct prepare_copy_options *options)
{
	return (options->stop != NULL && *options->stop != 0) ||
	       copy_now() >= options->deadline_ns;
}
result_status prepare_direct_alignment(int descriptor, uint32_t *alignment)
{
	if (descriptor < 0 || alignment == NULL)
		return STATUS_INVALID_ARGUMENT;
	struct statx file = { 0 };
	if (statx(descriptor, "", AT_EMPTY_PATH, STATX_DIOALIGN, &file) != 0 ||
	    (file.stx_mask & STATX_DIOALIGN) == 0 ||
	    file.stx_dio_mem_align == 0 || file.stx_dio_offset_align == 0)
		return STATUS_UNSUPPORTED;
	uint32_t value = file.stx_dio_mem_align > file.stx_dio_offset_align ?
				 file.stx_dio_mem_align :
				 file.stx_dio_offset_align;
	if (value < PACK_BLOCK_BYTES)
		value = PACK_BLOCK_BYTES;
	if (value > COPY_ALIGNMENT_MOST || (value & (value - 1u)) != 0 ||
	    COPY_PIECE_BYTES % value != 0)
		return STATUS_UNSUPPORTED;
	*alignment = value;
	return STATUS_OK;
}
void prepare_extent_record(struct prepare_extent_map *map, uint64_t logical,
			   uint64_t physical, uint64_t length, uint32_t flags)
{
	if (map->count == 0 || logical != map->next_logical ||
	    physical != map->next_physical)
		map->fragments++;
	map->next_logical = logical + length;
	map->next_physical = physical + length;
	if (length > map->largest)
		map->largest = length;
	map->needs_allocation =
		map->needs_allocation ||
		(flags & (FIEMAP_EXTENT_SHARED | FIEMAP_EXTENT_UNKNOWN |
			  FIEMAP_EXTENT_ENCODED | FIEMAP_EXTENT_DELALLOC)) != 0;
	map->count++;
}

bool prepare_extents_fragmented(const struct prepare_extent_map *map,
				uint64_t bytes)
{
	return map->fragments > FRAGMENT_BASE &&
	       map->fragments > bytes / GIB_BYTES + 1u;
}

result_status prepare_extents(int descriptor, struct prepare_extent_map *output)
{
	if (descriptor < 0 || output == NULL)
		return STATUS_INVALID_ARGUMENT;
	const size_t bytes = sizeof(struct fiemap) +
			     EXTENT_PAGE * sizeof(struct fiemap_extent);
	struct fiemap *const map = calloc(1, bytes);
	if (map == NULL)
		return STATUS_OUT_OF_MEMORY;
	struct prepare_extent_map value = { 0 };
	bool valid = true, last = false;
	uint64_t offset = 0;
	while (valid && !last && value.count < EXTENTS_MOST) {
		memset(map, 0, bytes);
		map->fm_start = offset;
		map->fm_length = UINT64_MAX - offset;
		map->fm_flags = FIEMAP_FLAG_SYNC;
		map->fm_extent_count = EXTENT_PAGE;
		valid = ioctl(descriptor, FS_IOC_FIEMAP, map) == 0 &&
			map->fm_mapped_extents <= EXTENT_PAGE;
		if (!valid || map->fm_mapped_extents == 0)
			break;
		for (uint32_t index = 0; index < map->fm_mapped_extents;
		     index++) {
			const struct fiemap_extent *const extent =
				&map->fm_extents[index];
			if (extent->fe_length == 0 ||
			    extent->fe_logical >
				    UINT64_MAX - extent->fe_length ||
			    extent->fe_logical + extent->fe_length <= offset) {
				valid = false;
				break;
			}
			offset = extent->fe_logical + extent->fe_length;
			prepare_extent_record(&value, extent->fe_logical,
					      extent->fe_physical,
					      extent->fe_length,
					      extent->fe_flags);
			last = (extent->fe_flags & FIEMAP_EXTENT_LAST) != 0;
		}
	}
	valid = valid && value.count < EXTENTS_MOST;
	value.fragmented = prepare_extents_fragmented(&value, offset);
	free(map);
	if (valid)
		*output = value;
	return valid ? STATUS_OK : STATUS_UNSUPPORTED;
}
static void ring_close(struct copy_ring *ring)
{
	if (ring->poisoned)
		return;
	if (ring->entries != NULL && ring->entries != MAP_FAILED)
		(void)munmap(ring->entries, ring->entries_bytes);
	if (ring->completion != NULL && ring->completion != MAP_FAILED &&
	    ring->completion != ring->submission)
		(void)munmap(ring->completion, ring->completion_bytes);
	if (ring->submission != NULL && ring->submission != MAP_FAILED)
		(void)munmap(ring->submission, ring->submission_bytes);
	if (ring->descriptor >= 0)
		(void)close(ring->descriptor);
	memset(ring, 0, sizeof(*ring));
	ring->descriptor = -1;
}
static bool ring_open(struct copy_ring *ring)
{
	memset(ring, 0, sizeof(*ring));
	struct io_uring_params parameters = { 0 };
	ring->descriptor =
		(int)syscall(SYS_io_uring_setup, COPY_RING_DEPTH, &parameters);
	if (ring->descriptor < 0)
		return false;
	ring->submission_bytes = parameters.sq_off.array +
				 parameters.sq_entries * sizeof(unsigned);
	ring->completion_bytes =
		parameters.cq_off.cqes +
		parameters.cq_entries * sizeof(struct io_uring_cqe);
	const bool single = (parameters.features & IORING_FEAT_SINGLE_MMAP) !=
			    0;
	if (single && ring->completion_bytes > ring->submission_bytes)
		ring->submission_bytes = ring->completion_bytes;
	ring->submission = mmap(NULL, ring->submission_bytes,
				PROT_READ | PROT_WRITE, MAP_SHARED,
				ring->descriptor, IORING_OFF_SQ_RING);
	ring->completion = single ? ring->submission :
				    mmap(NULL, ring->completion_bytes,
					 PROT_READ | PROT_WRITE, MAP_SHARED,
					 ring->descriptor, IORING_OFF_CQ_RING);
	ring->entries_bytes =
		parameters.sq_entries * sizeof(struct io_uring_sqe);
	ring->entries = mmap(NULL, ring->entries_bytes, PROT_READ | PROT_WRITE,
			     MAP_SHARED, ring->descriptor, IORING_OFF_SQES);
	if (ring->submission == MAP_FAILED || ring->completion == MAP_FAILED ||
	    ring->entries == MAP_FAILED) {
		ring_close(ring);
		return false;
	}
	unsigned char *const sq = ring->submission, *const cq =
							    ring->completion;
	ring->head = (void *)(sq + parameters.sq_off.head);
	ring->tail = (void *)(sq + parameters.sq_off.tail);
	ring->mask = (void *)(sq + parameters.sq_off.ring_mask);
	ring->array = (void *)(sq + parameters.sq_off.array);
	ring->completion_head = (void *)(cq + parameters.cq_off.head);
	ring->completion_tail = (void *)(cq + parameters.cq_off.tail);
	ring->completion_mask = (void *)(cq + parameters.cq_off.ring_mask);
	ring->completed = (void *)(cq + parameters.cq_off.cqes);
	return true;
}
/* Several aligned pieces per chunk, fan-out writes to all targets in the same submission batch. */
static bool transfer(struct copy_worker *worker, const int *descriptors,
		     uint32_t count, void *buffer, size_t bytes,
		     uint64_t offset, bool write, bool *failed)
{
	const uint32_t pieces =
		(uint32_t)((bytes - 1u) / COPY_PIECE_BYTES + 1u);
	const uint32_t total = count * pieces;
	struct copy_ring *const ring = &worker->ring;
	/* a poisoned ring may hold entries the kernel never took and completions of reads still landing: none of its
	 * results is trusted again, and the worker fails (its buffers are never reused) */
	if (ring->poisoned)
		return false;
	for (uint32_t first = 0; first < total;) {
		const uint32_t batch = total - first < COPY_RING_DEPTH ?
					       total - first :
					       COPY_RING_DEPTH;
		if (ring->descriptor < 0) {
			for (uint32_t index = 0; index < batch; index++) {
				const uint32_t ordinal = first + index,
					       device = ordinal / pieces,
					       piece = ordinal % pieces;
				const size_t begin =
					(size_t)piece * COPY_PIECE_BYTES;
				const size_t length =
					bytes - begin < COPY_PIECE_BYTES ?
						bytes - begin :
						COPY_PIECE_BYTES;
				struct iovec vector = {
					.iov_base =
						(unsigned char *)buffer + begin,
					.iov_len = length
				};
				ssize_t amount;
				do {
					amount =
						write ? pwritev(descriptors
									[device],
								&vector, 1,
								(off_t)(offset +
									begin)) :
							preadv(descriptors
								       [device],
							       &vector, 1,
							       (off_t)(offset +
								       begin));
				} while (amount < 0 && errno == EINTR);
				failed[device] = failed[device] ||
						 amount != (ssize_t)length;
			}
			first += batch;
			continue;
		}
		const unsigned tail =
			__atomic_load_n(ring->tail, __ATOMIC_RELAXED);
		for (uint32_t index = 0; index < batch; index++) {
			const uint32_t ordinal = first + index,
				       device = ordinal / pieces,
				       piece = ordinal % pieces;
			const size_t begin = (size_t)piece * COPY_PIECE_BYTES;
			const uint32_t length =
				(uint32_t)(bytes - begin < COPY_PIECE_BYTES ?
						   bytes - begin :
						   COPY_PIECE_BYTES);
			const unsigned position = (tail + index) & *ring->mask;
			struct io_uring_sqe *const entry =
				&ring->entries[position];
			memset(entry, 0, sizeof(*entry));
			entry->opcode = write ? IORING_OP_WRITE :
						IORING_OP_READ;
			entry->fd = descriptors[device];
			entry->off = offset + begin;
			entry->addr =
				(uint64_t)(uintptr_t)((unsigned char *)buffer +
						      begin);
			entry->len = length;
			entry->user_data = ordinal;
			ring->array[position] = position;
		}
		__atomic_store_n(ring->tail, tail + batch, __ATOMIC_RELEASE);
		uint32_t submitted = 0;
		while (submitted < batch) {
			const long sent =
				syscall(SYS_io_uring_enter, ring->descriptor,
					batch - submitted, 0, 0, NULL, 0);
			if (sent < 0 && errno == EINTR)
				continue;
			if (sent <= 0) {
				ring->poisoned = true;
				return false;
			}
			submitted += (uint32_t)sent;
		}
		uint32_t completed = 0;
		uint64_t drain_deadline = 0;
		while (completed < batch) {
			unsigned head = __atomic_load_n(ring->completion_head,
							__ATOMIC_RELAXED);
			const unsigned end = __atomic_load_n(
				ring->completion_tail, __ATOMIC_ACQUIRE);
			while (head != end && completed < batch) {
				const struct io_uring_cqe *const result =
					&ring->completed[head &
							 *ring->completion_mask];
				if (result->user_data < first ||
				    result->user_data >= first + batch) {
					ring->poisoned = true;
					return false;
				}
				const uint32_t ordinal = (uint32_t)result
								 ->user_data,
					       device = ordinal / pieces,
					       piece = ordinal % pieces;
				const size_t begin =
					(size_t)piece * COPY_PIECE_BYTES;
				const size_t length =
					bytes - begin < COPY_PIECE_BYTES ?
						bytes - begin :
						COPY_PIECE_BYTES;
				failed[device] = failed[device] ||
						 result->res != (int)length;
				head++;
				completed++;
			}
			__atomic_store_n(ring->completion_head, head,
					 __ATOMIC_RELEASE);
			if (completed < batch) {
				if (copy_stopped(worker->shared->options) &&
				    drain_deadline == 0)
					drain_deadline =
						copy_now() +
						COPY_DRAIN_SECONDS * NS_SECOND;
				if (drain_deadline != 0 &&
				    copy_now() >= drain_deadline) {
					ring->poisoned = true;
					ring->stop_poisoned = true;
					return false;
				}
				struct pollfd poller = { .fd = ring->descriptor,
							 .events = POLLIN };
				if (poll(&poller, 1, COPY_POLL_MS) < 0 &&
				    errno != EINTR) {
					ring->poisoned = true;
					return false;
				}
			}
		}
		first += batch;
	}
	return true;
}
static bool state_save(const struct copy_target *target)
{
	struct prepare_file file;
	if (prepare_file_open(target->journal, &file) != STATUS_OK)
		return false;
	if (fwrite(&target->state, sizeof(target->state), 1, file.stream) !=
	    1) {
		prepare_file_abort(&file);
		return false;
	}
	return prepare_file_commit(&file) == STATUS_OK;
}
static bool owned_file_links(const struct stat *file, nlink_t links)
{
	return S_ISREG(file->st_mode) && file->st_uid == geteuid() &&
	       file->st_nlink == links &&
	       (file->st_mode & (S_IWGRP | S_IWOTH)) == 0;
}
static bool owned_file(const struct stat *file)
{
	return owned_file_links(file, 1);
}
static void remove_owned(const char *path)
{
	struct stat file;
	if (lstat(path, &file) == 0 && owned_file(&file))
		(void)unlink(path);
}
/* 8 Nov 2026: the refusals for which the lock holder removes its own checkpoint: space, read-only and the
 * explicit media policies. Every other refusal (a describe, identity or I/O failure, any transient) keeps it. */
static bool checkpoint_removable(const char *reason)
{
	static const char *const removable[] = {
		"insufficient_space",
		"read_only_filesystem",
		"external_device_requires_override",
		"rotational_device",
		"network_filesystem_or_transport",
		"virtual_filesystem_or_device"
	};
	for (size_t index = 0;
	     reason != NULL && index < sizeof(removable) / sizeof(removable[0]);
	     index++)
		if (strcmp(reason, removable[index]) == 0)
			return true;
	return false;
}
/* A refused target's checkpoint: removed with its journal only by the lock holder, only when it is the engine's own
 * and not adopted and the refusal is removable, and never beside a final this run has not verified. A partial this
 * run created and never journaled is removed on any refusal: no later run could resume or adopt it. Otherwise it is
 * kept and one line names it, its size, the reason and what the run examined: nothing without the lock (another
 * copy's partial) or before its journal is read (a copier holds its lock, or the target cannot be examined), the
 * journal before a refusal that precedes the adoption proof, or the proof. True when removed. */
static bool checkpoint_settle(const struct copy_target *state,
			      const char *reason)
{
	struct stat file;
	if (state->partial[0] == '\0' || lstat(state->partial, &file) != 0)
		return false;
	if (state->locked && !state->finalized &&
	    ((state->created && !state->journaled) ||
	     (state->engine && !state->adopted &&
	      checkpoint_removable(reason)))) {
		remove_owned(state->partial);
		remove_owned(state->journal);
		return true;
	}
	char shown[PATH_MAX];
	local_device_display(shown, sizeof(shown), state->partial);
	const char *const why = state->foreign_why;
	/* the line: what was examined, then why (a refusal before the journal, or a foreign partial), then advice */
	const char *examined, *detail = "", *advice = "";
	if (!state->locked)
		examined =
			state->lock_busy ?
				"; not examined: another copy holds its " COPY_LOCK_SUFFIX
				", so it is that copy's" :
				"; not examined: its " COPY_LOCK_SUFFIX
				" cannot be opened as an owned regular file";
	else if (!state->journal_examined) {
		examined = "; not examined: ";
		detail = why != NULL ? why : "the target cannot be examined";
		advice = "; the engine never removes or changes it";
	} else if (state->adopted)
		examined =
			"; adopted from a foreign partial, so the engine never removes it";
	else if (state->engine)
		examined = "";
	else if (why == NULL)
		examined =
			"; no matching engine journal, and not examined for adoption, since that refusal comes first: "
			"the engine never removes or changes it";
	else if (why[0] == '\0')
		examined =
			"; its leading chunks are byte-equal to the source, but it was not adopted: the engine never"
			" removes or changes it";
	else {
		examined = "; no matching engine journal (";
		detail = why;
		advice =
			"), so the engine never removes or changes it: move it away or remove it once no copier "
			"holds it";
	}
	(void)fprintf(stderr,
		      "storage: replica checkpoint %s kept (%" PRIu64
		      " bytes): %s%s%s%s\n",
		      shown, file.st_size < 0 ? 0u : (uint64_t)file.st_size,
		      reason != NULL ? reason : "destination_directory_policy",
		      examined, detail, advice);
	return false;
}
static void copy_state_expected(struct copy_state *state,
				const struct prepare_manifest *manifest,
				const char *key, const struct stat *file)
{
	memset(state, 0, sizeof(*state));
	state->schema = COPY_STATE_SCHEMA;
	state->chunks = manifest->chunks;
	state->bytes = manifest->bytes;
	state->chunk_bytes = manifest->chunk_bytes;
	state->inode = (uint64_t)file->st_ino;
	(void)snprintf(state->key, sizeof(state->key), "%s", key);
	(void)snprintf(state->pack, sizeof(state->pack), "%s",
		       manifest->pack_identity);
}
/* Whether journal is a valid engine journal of the partial file for this pack, geometry and hardware key; its bitmap
 * into prior. */
static bool journal_valid(const char *journal,
			  const struct prepare_manifest *manifest,
			  const char *key, const struct stat *file,
			  struct copy_state *prior)
{
	struct copy_state expected;
	copy_state_expected(&expected, manifest, key, file);
	const int descriptor =
		open(journal, O_RDONLY | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK);
	struct stat journal_file;
	bool valid = descriptor >= 0 && fstat(descriptor, &journal_file) == 0 &&
		     owned_file(&journal_file) &&
		     journal_file.st_size == (off_t)sizeof(*prior) &&
		     read(descriptor, prior, sizeof(*prior)) ==
			     (ssize_t)sizeof(*prior);
	if (valid) {
		expected.adopted =
			prior->adopted; /* any: the identities match */
		valid = memcmp(prior, &expected,
			       offsetof(struct copy_state, bitmap)) == 0;
	}
	if (descriptor >= 0)
		(void)close(descriptor);
	return valid;
}
static bool same_file_state(const struct stat *first, const struct stat *second)
{
	return first->st_dev == second->st_dev &&
	       first->st_ino == second->st_ino &&
	       first->st_size == second->st_size &&
	       first->st_blocks == second->st_blocks &&
	       first->st_mtim.tv_sec == second->st_mtim.tv_sec &&
	       first->st_mtim.tv_nsec == second->st_mtim.tv_nsec &&
	       first->st_ctim.tv_sec == second->st_ctim.tv_sec &&
	       first->st_ctim.tv_nsec == second->st_ctim.tv_nsec;
}
/* The adoption proof of a partial without an engine journal, read-only: an owned regular file of the pack's size,
 * not written for ADOPTION_QUIET_SECONDS (every write moves its modification time), whose leading chunks hash to the
 * source manifest (the first at least, then up to the first difference or ADOPTION_PROOF_CHUNKS), and whose state
 * (size, blocks, modification and change times) is the same after the reads. The proven chunk count (0: unproven,
 * with the reason in why); the proven file's state into proven. */
static uint32_t adoption_proof(int directory, const char *name,
			       const struct prepare_manifest *manifest,
			       struct stat *proven, const char **why)
{
	*why = "it is not an owned regular file of the pack's size";
	const int file = openat(directory, name,
				O_RDONLY | O_DIRECT | O_NOFOLLOW | O_NOATIME |
					O_CLOEXEC | O_NONBLOCK);
	struct stat before, after;
	if (file < 0 || fstat(file, &before) != 0 || !owned_file(&before) ||
	    before.st_size < 0 || (uint64_t)before.st_size != manifest->bytes) {
		if (file >= 0)
			(void)close(file);
		return 0;
	}
	struct timespec now;
	uint32_t alignment = 0, chunks = 0;
	void *buffer = NULL;
	if (clock_gettime(CLOCK_REALTIME, &now) != 0 ||
	    now.tv_sec - before.st_mtim.tv_sec < ADOPTION_QUIET_SECONDS)
		*why = "it was written in the last 15 minutes, so a copier may still write it";
	else if (prepare_direct_alignment(file, &alignment) != STATUS_OK ||
		 manifest->chunk_bytes % alignment != 0 ||
		 manifest->bytes % alignment != 0 ||
		 posix_memalign(&buffer, alignment,
				(size_t)manifest->chunk_bytes) != 0)
		*why = "it cannot be read with direct I/O";
	else if (manifest->hashes[0][0] == '\0')
		*why = "the source manifest does not record its first chunk yet";
	else
		*why = "its first chunk is not byte-equal to the source";
	const uint32_t most = manifest->chunks < ADOPTION_PROOF_CHUNKS ?
				      manifest->chunks :
				      ADOPTION_PROOF_CHUNKS;
	while (buffer != NULL && chunks < most &&
	       manifest->hashes[chunks][0] != '\0') {
		const uint64_t offset =
			(uint64_t)chunks * manifest->chunk_bytes;
		const size_t length =
			(size_t)(manifest->bytes - offset <
						 manifest->chunk_bytes ?
					 manifest->bytes - offset :
					 manifest->chunk_bytes);
		size_t done = 0;
		while (done < length) {
			const ssize_t got = pread(file, (char *)buffer + done,
						  length - done,
						  (off_t)(offset + done));
			if (got <= 0)
				break;
			done += (size_t)got;
		}
		char digest[STORAGE_PROFILE_KEY_BYTES];
		if (done != length ||
		    prepare_hash(buffer, length, digest) != STATUS_OK ||
		    strcmp(digest, manifest->hashes[chunks]) != 0)
			break;
		chunks++;
	}
	free(buffer);
	if (chunks != 0 &&
	    (fstat(file, &after) != 0 || !same_file_state(&before, &after))) {
		*why = "it changed while it was read";
		chunks = 0;
	}
	(void)close(file);
	if (chunks != 0) {
		*proven = before;
		*why = "";
	}
	return chunks;
}
uint64_t prepare_available_with_credit(uint64_t total, uint64_t available,
				       uint64_t credit)
{
	return available >= total || credit >= total - available ?
		       total :
		       available + credit;
}
static const char *target_room(const struct prepare_copy_options *options,
			       uint32_t index, uint64_t needed, uint64_t credit)
{
	struct prepare_target *const target = &options->targets[index];
	char parent[PATH_MAX], key[STORAGE_PROFILE_KEY_BYTES];
	(void)snprintf(parent, sizeof(parent), "%s", target->path);
	char *const slash = strrchr(parent, '/');
	if (slash == NULL)
		return "destination_directory_policy";
	*(slash == parent ? slash + 1 : slash) = '\0';
	struct local_device_description current;
	if (local_device_describe(parent, &current) != STATUS_OK)
		return "device_disappeared_or_unknown";
	target->device = current;
	const char *reason =
		local_device_media_refusal(&current, options->external);
	if (reason == NULL &&
	    storage_device_key_for(parent, options->external, key, &current) !=
		    STATUS_OK)
		reason = "hardware_identity_unavailable";
	if (reason == NULL && strcmp(target->key, key) != 0)
		reason = "hardware_identity_mismatch_retune_required";
	/* An engine checkpoint beside a final is removed once the final verifies, so its bytes count as available. */
	if (reason == NULL)
		reason = local_device_space_refusal(
			current.total_bytes,
			prepare_available_with_credit(current.total_bytes,
						      current.available_bytes,
						      credit),
			needed, options->floor_percent, 0, current.readonly);
	return reason;
}
static const char *checkpoint_room(struct copy_shared *shared, uint32_t index)
{
	struct statvfs space;
	if (fstatvfs(shared->targets[index].directory, &space) != 0 ||
	    space.f_frsize == 0 ||
	    space.f_blocks > UINT64_MAX / space.f_frsize ||
	    space.f_bavail > UINT64_MAX / space.f_frsize)
		return "copy_device_io_failure";
	struct prepare_target *const target = &shared->options->targets[index];
	target->device.total_bytes = space.f_blocks * space.f_frsize;
	target->device.available_bytes = space.f_bavail * space.f_frsize;
	return local_device_space_refusal(target->device.total_bytes,
					  target->device.available_bytes, 0,
					  shared->options->floor_percent, 0,
					  (space.f_flag & ST_RDONLY) != 0);
}
/* A file at the final name that is not a verified copy is never replaced, and a checkpoint and journal beside it are
 * kept for the run after it is removed. */
static void final_kept(const struct prepare_target *target, bool checkpoint)
{
	char shown[PATH_MAX];
	local_device_display(shown, sizeof(shown), target->path);
	(void)fprintf(
		stderr,
		"storage: replica final %s is not a verified copy: finalized_replica_unverified (left in place%s)\n",
		shown, checkpoint ? "; its checkpoint is kept" : "");
}

/* A final's verification receipt (<name>.verified): the device's hardware key and the file as fstat saw it when a full
 * verification started, and the manifest it verified against (a digest of its chunk hashes) and pack. A later run
 * trusts a final whose receipt still matches: a change of hardware key, inode, size, mtime or ctime, manifest or pack
 * forces a full read. The device number is not bound: a reboot may renumber NVMe, dm and btrfs devices, and the
 * hardware key names the device across that. The trade-off: bytes that change with no metadata change (a device's
 * silent corruption) are caught only by a spot chunk or a full read. */
static bool receipt_text(const struct stat *file,
			 const struct prepare_manifest *manifest,
			 const char *key, char *output, size_t capacity)
{
	char digest[STORAGE_PROFILE_KEY_BYTES];
	if (manifest->known != manifest->chunks ||
	    prepare_hash(manifest->hashes,
			 (size_t)manifest->chunks * sizeof(manifest->hashes[0]),
			 digest) != STATUS_OK)
		return false;
	const int length = snprintf(
		output, capacity,
		"anemone-verified %u key=%s inode=%" PRIu64 " size=%" PRIu64
		" mtime_ns=%" PRId64 " ctime_ns=%" PRId64
		" manifest=%s pack=%s chunks=%" PRIu32 "\n",
		RECEIPT_SCHEMA, key, (uint64_t)file->st_ino,
		(uint64_t)file->st_size,
		(int64_t)file->st_mtim.tv_sec * (int64_t)NS_SECOND +
			(int64_t)file->st_mtim.tv_nsec,
		(int64_t)file->st_ctim.tv_sec * (int64_t)NS_SECOND +
			(int64_t)file->st_ctim.tv_nsec,
		digest, manifest->pack_identity, manifest->chunks);
	return length > 0 && (size_t)length < capacity;
}
static bool receipt_name(char *output, size_t capacity, const char *name,
			 const char *suffix)
{
	const int length = snprintf(output, capacity, "%s%s", name, suffix);
	return length > 0 && (size_t)length < capacity;
}
/* Whether the receipt beside a final matches the final (its stat), the device's hardware key and the manifest. */
static bool receipt_matches(int directory, const char *name,
			    const struct prepare_manifest *manifest,
			    const char *key, const struct stat *file)
{
	char path[PATH_MAX], expected[RECEIPT_BYTES], found[RECEIPT_BYTES];
	if (!receipt_name(path, sizeof(path), name, PREPARE_RECEIPT_SUFFIX) ||
	    !receipt_text(file, manifest, key, expected, sizeof(expected)))
		return false;
	const int descriptor =
		openat(directory, path,
		       O_RDONLY | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK);
	struct stat receipt;
	const ssize_t count =
		descriptor >= 0 && fstat(descriptor, &receipt) == 0 &&
				owned_file(&receipt) ?
			read(descriptor, found, sizeof(found) - 1u) :
			-1;
	if (descriptor >= 0)
		(void)close(descriptor);
	if (count <= 0)
		return false;
	found[count] = '\0';
	return strcmp(found, expected) == 0;
}
/* Whether a target's state is as before: device, inode, size and mtime, and ctime when with_change (a rename changes
 * it). The device number is compared only within one run. */
static bool target_unchanged(const struct stat *before,
			     const struct stat *after, bool with_change)
{
	return before->st_dev == after->st_dev &&
	       before->st_ino == after->st_ino &&
	       before->st_size == after->st_size &&
	       before->st_mtim.tv_sec == after->st_mtim.tv_sec &&
	       before->st_mtim.tv_nsec == after->st_mtim.tv_nsec &&
	       (!with_change ||
		(before->st_ctim.tv_sec == after->st_ctim.tv_sec &&
		 before->st_ctim.tv_nsec == after->st_ctim.tv_nsec));
}
/* checking builds: another writer changes the target under the run (its mtime a second later), as `dd` would */
static void target_touch(int descriptor, const struct stat *file)
{
	const struct timespec times[FILE_TIMES] = {
		{ .tv_nsec = UTIME_OMIT },
		{ .tv_sec = file->st_mtim.tv_sec + 1,
		  .tv_nsec = file->st_mtim.tv_nsec }
	};
	(void)futimens(descriptor, times);
}
enum receipt_outcome {
	RECEIPT_NOT_WRITTEN,
	RECEIPT_WRITTEN,
	RECEIPT_WRITTEN_UNSYNCED /* renamed into place, but its directory's sync failed */
};
/* text as the final's receipt of that suffix, written under the temporary suffix, synced, renamed over the old one and
 * the directory synced, so a reader finds a whole receipt or none. A temporary that cannot be completed is removed. */
static enum receipt_outcome receipt_file_write(const struct copy_target *state,
					       const char *suffix,
					       const char *temporary_suffix,
					       const char *text)
{
	char name[PATH_MAX], temporary[PATH_MAX];
	struct stat stale, written;
	if (!receipt_name(name, sizeof(name), state->name, suffix) ||
	    !receipt_name(temporary, sizeof(temporary), state->name,
			  temporary_suffix))
		return RECEIPT_NOT_WRITTEN;
	if (fstatat(state->directory, temporary, &stale, AT_SYMLINK_NOFOLLOW) ==
		    0 &&
	    owned_file(&stale))
		(void)unlinkat(state->directory, temporary, 0);
	const int descriptor =
		openat(state->directory, temporary,
		       O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC,
		       S_IRUSR | S_IWUSR);
	if (descriptor < 0)
		return RECEIPT_NOT_WRITTEN;
	const size_t length = strlen(text);
	bool done = fstat(descriptor, &written) == 0 && owned_file(&written) &&
		    !FAULT_POINT(prepare_receipt_write) &&
		    write(descriptor, text, length) == (ssize_t)length &&
		    fdatasync(descriptor) == 0;
	done = close(descriptor) == 0 && done;
	done = done && renameat(state->directory, temporary, state->directory,
				name) == 0;
	if (!done) {
		(void)unlinkat(state->directory, temporary, 0);
		return RECEIPT_NOT_WRITTEN;
	}
	return !FAULT_POINT(prepare_receipt_directory_sync) &&
			       fsync(state->directory) == 0 ?
		       RECEIPT_WRITTEN :
		       RECEIPT_WRITTEN_UNSYNCED;
}
/* After a full verification: the receipt of file, the state the verification started from (receipt_text). */
static enum receipt_outcome
receipt_write(const struct copy_target *state,
	      const struct prepare_manifest *manifest, const char *key,
	      const struct stat *file)
{
	char text[RECEIPT_BYTES];
	return receipt_text(file, manifest, key, text, sizeof(text)) ?
		       receipt_file_write(state, PREPARE_RECEIPT_SUFFIX,
					  RECEIPT_TEMPORARY, text) :
		       RECEIPT_NOT_WRITTEN;
}
/* After the same full verification, when the pack identity is a whole pack's header digest (not one source file of
 * an EXL3 map): the replica search's receipt (replicas.h), so that a load finds the copy without a profile. */
static enum receipt_outcome
search_receipt_write(const struct copy_target *state,
		     const struct prepare_manifest *manifest)
{
	char text[SEARCH_RECEIPT_BYTES], *end = NULL;
	errno = 0;
	const unsigned long long digest =
		strtoull(manifest->pack_identity, &end, HEXADECIMAL);
	const bool whole_pack =
		strlen(manifest->pack_identity) == HEADER_DIGEST_DIGITS &&
		strspn(manifest->pack_identity, "0123456789abcdef") ==
			HEADER_DIGEST_DIGITS &&
		errno == 0 && end != NULL && *end == '\0';
	return whole_pack && replicas_receipt_text(text, sizeof(text),
						   manifest->source,
						   manifest->bytes,
						   (uint64_t)digest) ?
		       receipt_file_write(state, REPLICA_RECEIPT_SUFFIX,
					  SEARCH_RECEIPT_TEMPORARY, text) :
		       RECEIPT_NOT_WRITTEN;
}
/* A failed final's receipt, unlinked and its directory synced, so no crash after the failure leaves it in place (a
 * receipt that is no owned file never matches). False when it stays or its removal may not be durable. */
static bool receipt_remove(const struct copy_target *state)
{
	char name[PATH_MAX], search[PATH_MAX];
	struct stat receipt;
	if (!receipt_name(name, sizeof(name), state->name,
			  PREPARE_RECEIPT_SUFFIX) ||
	    !receipt_name(search, sizeof(search), state->name,
			  REPLICA_RECEIPT_SUFFIX))
		return false;
	/* the search's receipt first: no load finds a copy whose verification failed */
	if (fstatat(state->directory, search, &receipt, AT_SYMLINK_NOFOLLOW) ==
		    0 &&
	    owned_file(&receipt) && unlinkat(state->directory, search, 0) != 0)
		return false;
	if (fstatat(state->directory, name, &receipt, AT_SYMLINK_NOFOLLOW) !=
		    0 ||
	    !owned_file(&receipt))
		return true;
	return unlinkat(state->directory, name, 0) == 0 &&
	       !FAULT_POINT(prepare_receipt_remove_sync) &&
	       fsync(state->directory) == 0;
}
/* A final whose verification failed (a chunk or its size did not match, or it changed during the run): its receipt
 * goes at once (where workers share the target, under the run's mutex), so a crash before the run ends never leaves a
 * receipt for a final known to be bad. path: the final, for the line that says when that removal is not durable. */
static void verification_fail(struct copy_target *state, const char *path)
{
	if (state->verification_failed)
		return;
	state->verification_failed = true;
	if (state->directory >= 0 && !receipt_remove(state)) {
		char shown[PATH_MAX];
		local_device_display(shown, sizeof(shown), path);
		(void)fprintf(
			stderr,
			"storage: replica final %s failed its verification, and its verification receipt may outlive a "
			"crash: its removal or the directory sync failed (copy_device_io_failure)\n",
			shown);
	}
}
uint32_t prepare_spot_chunks(uint64_t seed, uint32_t chunks, uint32_t wanted,
			     uint32_t output[PREPARE_SPOT_CHUNKS])
{
	wanted = wanted == 0 || wanted > PREPARE_SPOT_CHUNKS ?
			 PREPARE_SPOT_CHUNKS :
			 wanted;
	if (chunks <= wanted) {
		for (uint32_t chunk = 0; chunk < chunks; chunk++)
			output[chunk] = chunk;
		return chunks;
	}
	uint32_t count = 0;
	uint64_t state = seed;
	while (count < wanted) {
		state += SPLITMIX_INCREMENT;
		uint64_t mixed = state;
		mixed = (mixed ^ (mixed >> SPLITMIX_SHIFT_FIRST)) *
			SPLITMIX_FIRST;
		mixed = (mixed ^ (mixed >> SPLITMIX_SHIFT_SECOND)) *
			SPLITMIX_SECOND;
		mixed ^= mixed >> SPLITMIX_SHIFT_THIRD;
		const uint32_t chunk = (uint32_t)(mixed % chunks);
		bool seen = false;
		for (uint32_t index = 0; index < count; index++)
			seen = seen || output[index] == chunk;
		if (!seen)
			output[count++] = chunk;
	}
	return count;
}
/* flock(LOCK_EX | LOCK_NB), tried LOCK_TRIES times LOCK_RETRY_NS apart; false with the last try's errno. */
static bool lock_exclusive(int descriptor)
{
	for (uint32_t attempt = 1;; attempt++) {
		if (flock(descriptor, LOCK_EX | LOCK_NB) == 0)
			return true;
		if (errno != EWOULDBLOCK || attempt == LOCK_TRIES)
			return false;
		const struct timespec pause = { .tv_nsec = LOCK_RETRY_NS };
		(void)nanosleep(&pause, NULL);
		errno = EWOULDBLOCK;
	}
}
/* The earlier copier's lock, taken for every target not yet final the way that copier takes it (O_NOFOLLOW, an owned
 * regular file, exclusive and non-blocking) and held to the end of the run, so the two never work on one partial:
 * the copier can neither resume a checkpoint nor create a partial at its name. False, with the reason in foreign_why
 * (copier_busy when the copier holds it), when it cannot be taken. */
static bool copier_lock_take(struct copy_target *state)
{
	char name[PATH_MAX];
	const int length = snprintf(name, sizeof(name),
				    COPIER_LOCK_PREFIX "%s" COPIER_LOCK_SUFFIX,
				    state->name);
	struct stat file;
	state->copier_lock =
		length > 0 && (size_t)length < sizeof(name) ?
			openat(state->directory, name,
			       O_WRONLY | O_CREAT | O_NOFOLLOW | O_CLOEXEC,
			       S_IRUSR | S_IWUSR) :
			-1;
	if (state->copier_lock < 0 || fstat(state->copier_lock, &file) != 0 ||
	    !owned_file(&file)) {
		state->foreign_why =
			"its copier's lock .<target>" COPIER_LOCK_SUFFIX
			" cannot be taken as an owned regular file";
		return false;
	}
	if (!lock_exclusive(state->copier_lock)) {
		state->copier_busy = errno == EWOULDBLOCK;
		state->foreign_why =
			"a copier holds its lock .<target>" COPIER_LOCK_SUFFIX;
		return false;
	}
	return true;
}
/* A target whose name or partial cannot be examined (fstatat failed other than ENOENT): refused, as --plan names it. */
static const char *examine_refusal(int error)
{
	return error == EACCES ? "destination_directory_policy" :
				 "copy_device_io_failure";
}
static bool target_open(struct copy_shared *shared, uint32_t index)
{
	struct prepare_target *const target = &shared->options->targets[index];
	struct copy_target *const state = &shared->targets[index];
	const struct prepare_manifest *const manifest =
		shared->options->manifest;
	char parent[PATH_MAX], lock[PATH_MAX];
	if (target->path[0] != '/' || strlen(target->path) >= sizeof(parent))
		return false;
	(void)snprintf(parent, sizeof(parent), "%s", target->path);
	char *const slash = strrchr(parent, '/');
	if (slash == NULL)
		return false;
	*slash = '\0';
	char real[PATH_MAX];
	if (realpath(parent, real) == NULL ||
	    strstr(real, "/anemone_engram_rnd/") == NULL)
		return false;
	struct stat directory;
	if (stat(real, &directory) != 0 || !S_ISDIR(directory.st_mode) ||
	    directory.st_uid != geteuid())
		return false;
	state->directory =
		open(real, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
	struct stat opened;
	if (state->directory < 0 || fstat(state->directory, &opened) != 0 ||
	    opened.st_dev != directory.st_dev ||
	    opened.st_ino != directory.st_ino ||
	    ((directory.st_mode & (S_IWGRP | S_IWOTH)) != 0 &&
	     fchmod(state->directory, S_IRWXU) != 0))
		return false;
	state->name = strrchr(target->path, '/') + 1;
	const int partial_length =
		snprintf(state->partial, sizeof(state->partial),
			 "%s" PREPARE_PARTIAL_SUFFIX, target->path);
	const int journal_length =
		snprintf(state->journal, sizeof(state->journal),
			 "%s" COPY_STATE_SUFFIX, target->path);
	const int lock_length = snprintf(lock, sizeof(lock),
					 "%s" COPY_LOCK_SUFFIX, target->path);
	if (partial_length < 0 || journal_length < 0 || lock_length < 0 ||
	    (size_t)partial_length >= sizeof(state->partial) ||
	    (size_t)journal_length >= sizeof(state->journal) ||
	    (size_t)lock_length >= sizeof(lock))
		return false;
	state->partial_name = strrchr(state->partial, '/') + 1;
	state->lock = openat(state->directory, strrchr(lock, '/') + 1,
			     O_WRONLY | O_CREAT | O_NOFOLLOW | O_CLOEXEC,
			     S_IRUSR | S_IWUSR);
	if (state->lock < 0) {
		target->reason =
			errno == EROFS	? "read_only_filesystem" :
			errno == EACCES ? "destination_directory_policy" :
					  NULL;
		return false;
	}
	struct stat lock_file;
	if (fstat(state->lock, &lock_file) != 0 || !owned_file(&lock_file))
		return false;
	if (!lock_exclusive(state->lock)) {
		state->lock_busy = errno == EWOULDBLOCK;
		if (state->lock_busy)
			target->reason = "copy_in_progress";
		return false;
	}
	state->locked = true;
	struct stat existing = { 0 }, partial = { 0 };
	state->finalized = fstatat(state->directory, state->name, &existing,
				   AT_SYMLINK_NOFOLLOW) == 0;
	if (!state->finalized && errno != ENOENT) {
		target->reason = examine_refusal(errno);
		return false;
	}
	if (!state->finalized && !copier_lock_take(state)) {
		target->reason = state->copier_busy ?
					 "copy_in_progress" :
					 "destination_directory_policy";
		return false;
	}
	const bool partial_exists = fstatat(state->directory,
					    state->partial_name, &partial,
					    AT_SYMLINK_NOFOLLOW) == 0;
	if (!partial_exists && errno != ENOENT) {
		target->reason = examine_refusal(errno);
		return false;
	}
	const bool partial_sized = partial_exists && owned_file(&partial) &&
				   partial.st_size >= 0 &&
				   (uint64_t)partial.st_size <=
					   manifest->bytes &&
				   partial.st_blocks >= 0 &&
				   (uint64_t)partial.st_blocks <=
					   UINT64_MAX / ALLOCATED_BLOCK_BYTES;
	struct copy_state prior = { 0 };
	state->engine = partial_sized &&
			journal_valid(state->journal, manifest, target->key,
				      &partial, &prior);
	state->journal_examined = true;
	state->adopted = state->engine && prior.adopted != 0;
	if (state->finalized &&
	    (!owned_file(&existing) || existing.st_size < 0 ||
	     (uint64_t)existing.st_size > manifest->bytes)) {
		target->reason = "finalized_replica_unverified";
		verification_fail(state, target->path);
		return false;
	}
	uint64_t allocated = 0;
	if (partial_sized)
		allocated = (uint64_t)partial.st_blocks * ALLOCATED_BLOCK_BYTES;
	if (state->finalized && state->engine && !state->adopted)
		state->credit = allocated;
	if (state->finalized && partial_exists && !state->engine)
		state->foreign_why = "a final file is beside it";
	if (!state->finalized && partial_exists) {
		if (!partial_sized) {
			state->foreign_why =
				"it is not an owned regular file of at most the pack's size";
			target->reason = "foreign_partial_kept";
			return false;
		}
		const int extents =
			openat(state->directory, state->partial_name,
			       O_RDONLY | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK);
		struct prepare_extent_map map;
		if (extents < 0 ||
		    prepare_extents(extents, &map) != STATUS_OK ||
		    map.needs_allocation)
			allocated = 0;
		if (extents >= 0)
			(void)close(extents);
	}
	const uint64_t remaining =
		state->finalized	    ? 0 :
		manifest->bytes > allocated ? manifest->bytes - allocated :
					      0;
	target->reason =
		target_room(shared->options, index, remaining, state->credit);
	if (target->reason != NULL)
		return false;
	/* A partial without a matching engine journal is foreign: adopted only on proof (the earlier copier's lock is
	 * held already), otherwise never touched. */
	uint32_t proven = 0;
	struct stat proof = { 0 };
	if (!state->finalized && partial_exists && !state->engine) {
		proven = adoption_proof(state->directory, state->partial_name,
					manifest, &proof, &state->foreign_why);
		if (proven == 0 || proof.st_dev != partial.st_dev ||
		    proof.st_ino != partial.st_ino) {
			target->reason = "foreign_partial_kept";
			return false;
		}
	}
	/* A fresh partial is created without a name where the filesystem allows it (O_TMPFILE), and named only once its
	 * journal is saved; elsewhere it is created at its name. */
	const bool fresh = !state->finalized && !partial_exists;
	state->descriptor =
		fresh ? openat(state->directory, ".",
			       O_TMPFILE | O_RDWR | O_DIRECT | O_CLOEXEC,
			       S_IRUSR | S_IWUSR) :
			-1;
	state->unnamed = state->descriptor >= 0;
	if (!state->unnamed &&
	    (!fresh || errno == EOPNOTSUPP || errno == EISDIR))
		state->descriptor = openat(
			state->directory,
			state->finalized ? state->name : state->partial_name,
			(state->finalized ? O_RDONLY :
			 partial_exists	  ? O_RDWR :
					    O_RDWR | O_CREAT | O_EXCL) |
				O_DIRECT | O_NOFOLLOW | O_CLOEXEC,
			S_IRUSR | S_IWUSR);
	/* created: named by this run; an unnamed partial is created only once it is linked at its name */
	state->created = state->descriptor >= 0 && fresh && !state->unnamed;
	struct stat file, source;
	uint32_t alignment = 0;
	if (state->descriptor < 0 || fstat(state->descriptor, &file) != 0 ||
	    !owned_file_links(&file, state->unnamed ? 0 : 1) ||
	    fstat(shared->source, &source) != 0 ||
	    (file.st_ino == source.st_ino && file.st_dev == source.st_dev) ||
	    (!state->finalized && partial_exists &&
	     (file.st_dev != partial.st_dev ||
	      file.st_ino != partial.st_ino)) ||
	    prepare_direct_alignment(state->descriptor, &alignment) !=
		    STATUS_OK ||
	    manifest->chunk_bytes % alignment != 0 ||
	    manifest->bytes % alignment != 0)
		return false;
	if (alignment > shared->alignment)
		shared->alignment = alignment;
	/* a partial this run creates, named or not (O_TMPFILE: linked only once journaled), holds allocated zeros */
	state->fresh = fresh;
	state->opened = file;
	copy_state_expected(&state->state, manifest, target->key, &file);
	/* A final whose receipt still matches is spot-checked instead of read in full (never with verify_full). */
	if (state->finalized && !shared->options->verify_full &&
	    receipt_matches(state->directory, state->name, manifest,
			    target->key, &file)) {
		uint64_t seed = shared->options->spot_seed;
		if (seed == 0 &&
		    getrandom(&seed, sizeof(seed), 0) != (ssize_t)sizeof(seed))
			seed = copy_now() ^ (uint64_t)file.st_ino;
		state->receipt_trusted = true;
		state->spot_count = prepare_spot_chunks(
			seed, manifest->chunks, shared->options->spot_chunks,
			state->spot);
	}
	const uint64_t proven_bytes =
		(uint64_t)proven * manifest->chunk_bytes < manifest->bytes ?
			(uint64_t)proven * manifest->chunk_bytes :
			manifest->bytes;
	state->state.adopted = state->engine ? prior.adopted : proven_bytes;
	if (state->engine)
		memcpy(state->state.bitmap, prior.bitmap, sizeof(prior.bitmap));
	for (uint32_t chunk = 0; chunk < proven; chunk++)
		state->state.bitmap[chunk / COPY_BITS] |=
			(unsigned char)(1u << (chunk % COPY_BITS));
	/* A bitmap is a durable progress hint only. Even set bits are reverified on every resume. The journal is saved
	 * before the allocation, so a partial this run creates is the engine's own from its first byte; for a foreign
	 * partial, it is the adoption. */
	if (!state->finalized) {
		/* checking builds: a process killed after the partial's creation, before its journal */
		if (FAULT_POINT(prepare_copy_killed))
			(void)raise(SIGKILL);
		if (FAULT_POINT(prepare_copy_journal) || !state_save(state))
			return false;
		state->journaled = true;
		if (state->unnamed) {
			char path[PROC_DESCRIPTOR_BYTES];
			(void)snprintf(path, sizeof(path), "/proc/self/fd/%d",
				       state->descriptor);
			/* checking builds: another writer takes the partial's name before the link */
			if (FAULT_POINT(prepare_copy_name_taken))
				(void)close(openat(
					state->directory, state->partial_name,
					O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC,
					S_IRUSR | S_IWUSR));
			if (linkat(AT_FDCWD, path, state->directory,
				   state->partial_name,
				   AT_SYMLINK_FOLLOW) != 0) {
				remove_owned(state->journal);
				state->journaled = false;
				return false;
			}
			state->created = true;
		}
		/* journaled and named: the engine's own checkpoint, even if the directory's sync below fails */
		state->engine = true;
		state->adopted = state->state.adopted != 0;
		if (state->unnamed &&
		    (FAULT_POINT(prepare_copy_directory_sync) ||
		     fsync(state->directory) != 0)) {
			target->reason = "copy_device_io_failure";
			return false;
		}
	}
	if (proven != 0) {
		char shown[PATH_MAX];
		local_device_display(shown, sizeof(shown), state->partial);
		(void)fprintf(
			stderr,
			"storage: replica checkpoint %s adopted (%" PRIu64
			" bytes): its leading %" PRIu64
			" bytes are byte-equal to the source; every chunk is verified again as the copy resumes, and"
			" the engine never removes it\n",
			shown, (uint64_t)partial.st_size, proven_bytes);
	}
	if (!state->finalized) {
		const int allocation = posix_fallocate(state->descriptor, 0,
						       (off_t)manifest->bytes);
		if (allocation != 0) {
			target->reason = allocation == ENOSPC ||
							 allocation == EDQUOT ?
						 "insufficient_space" :
						 "copy_io_error";
			return false;
		}
	}
	const bool stated = fstat(state->descriptor, &file) == 0;
	if (!stated || file.st_size < 0 ||
	    (uint64_t)file.st_size != manifest->bytes) {
		if (state->finalized) {
			target->reason = "finalized_replica_unverified";
			if (stated)
				verification_fail(state, target->path);
		}
		return false;
	}
	target->resumable = !state->finalized;
	return true;
}
enum lock_probe { PROBE_FREE, PROBE_HELD, PROBE_UNOWNED };
/* A lock beside the target that a run takes exclusively, probed shared and released at once: free (absent too, a run
 * creates it), held by another process, or no owned regular file; why names the last two. */
static enum lock_probe lock_probe(int directory, const char *name,
				  const char *held, const char *unowned,
				  const char **why)
{
	const int lock = openat(directory, name,
				O_RDONLY | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK);
	struct stat file;
	const enum lock_probe probe =
		lock < 0 ? (errno == ENOENT ? PROBE_FREE : PROBE_UNOWNED) :
		fstat(lock, &file) != 0 || !owned_file(&file) ? PROBE_UNOWNED :
		flock(lock, LOCK_SH | LOCK_NB) != 0	      ? PROBE_HELD :
								PROBE_FREE;
	if (lock >= 0)
		(void)close(lock);
	if (probe != PROBE_FREE)
		*why = probe == PROBE_HELD ? held : unowned;
	return probe;
}
/* --plan's view of one target, in target_open's order (see the header). */
static void checkpoint_examine(const struct prepare_manifest *manifest,
			       const char *parent, int directory,
			       const char *name, const char *key,
			       uint32_t floor_percent,
			       struct prepare_checkpoint *value)
{
	char partial_name[PATH_MAX], lock_name[PATH_MAX], copier_name[PATH_MAX];
	char journal[PATH_MAX];
	if (snprintf(partial_name, sizeof(partial_name),
		     "%s" PREPARE_PARTIAL_SUFFIX, name) >= (int)PATH_MAX ||
	    snprintf(lock_name, sizeof(lock_name), "%s" COPY_LOCK_SUFFIX,
		     name) >= (int)PATH_MAX ||
	    snprintf(copier_name, sizeof(copier_name),
		     COPIER_LOCK_PREFIX "%s" COPIER_LOCK_SUFFIX,
		     name) >= (int)PATH_MAX ||
	    snprintf(journal, sizeof(journal), "%s/%s" COPY_STATE_SUFFIX,
		     parent, name) >= (int)PATH_MAX) {
		value->kind = PREPARE_CHECKPOINT_UNKNOWN;
		value->error = ENAMETOOLONG;
		return;
	}
	const enum lock_probe lock = lock_probe(
		directory, lock_name,
		"another copy holds its " COPY_LOCK_SUFFIX
		", so it is that copy's",
		"its " COPY_LOCK_SUFFIX " is not an owned regular file",
		&value->why);
	if (lock != PROBE_FREE) {
		value->kind = PREPARE_CHECKPOINT_HELD;
		value->refusal = lock == PROBE_HELD ?
					 "copy_in_progress" :
					 "destination_directory_policy";
		return;
	}
	/* A run opens the lock for writing, creating it when absent: on a read-only filesystem or in a directory it may
	 * not write it refuses there, before it examines anything, and keeps what is there. */
	if (faccessat(directory, lock_name, W_OK, AT_EACCESS) != 0 &&
	    (errno != ENOENT ||
	     faccessat(directory, ".", W_OK, AT_EACCESS) != 0) &&
	    (errno == EROFS || errno == EACCES)) {
		value->kind = PREPARE_CHECKPOINT_HELD;
		value->refusal = errno == EROFS ?
					 "read_only_filesystem" :
					 "destination_directory_policy";
		value->why = "its " COPY_LOCK_SUFFIX
			     " cannot be opened for writing";
		return;
	}
	struct stat final_file, file, proof;
	value->beside_final =
		fstatat(directory, name, &final_file, AT_SYMLINK_NOFOLLOW) == 0;
	value->receipt =
		value->beside_final &&
		receipt_matches(directory, name, manifest, key, &final_file);
	if (!value->beside_final && errno != ENOENT) {
		value->kind = PREPARE_CHECKPOINT_UNKNOWN;
		value->error = errno;
		value->refusal = examine_refusal(errno);
		return;
	}
	/* Every target not yet final needs the earlier copier's lock as well. */
	const enum lock_probe copier =
		value->beside_final ?
			PROBE_FREE :
			lock_probe(
				directory, copier_name,
				"a copier holds its lock .<target>" COPIER_LOCK_SUFFIX,
				"its copier's lock .<target>" COPIER_LOCK_SUFFIX
				" cannot be taken as an owned regular file",
				&value->why);
	if (copier != PROBE_FREE) {
		value->kind = PREPARE_CHECKPOINT_HELD;
		value->refusal = copier == PROBE_HELD ?
					 "copy_in_progress" :
					 "destination_directory_policy";
		return;
	}
	const bool partial_exists = fstatat(directory, partial_name, &file,
					    AT_SYMLINK_NOFOLLOW) == 0;
	if (!partial_exists && errno != ENOENT) {
		value->kind = PREPARE_CHECKPOINT_UNKNOWN;
		value->error = errno;
		value->refusal = examine_refusal(errno);
		return;
	}
	const bool sized =
		partial_exists && owned_file(&file) && file.st_size >= 0 &&
		(uint64_t)file.st_size <= manifest->bytes &&
		file.st_blocks >= 0 &&
		(uint64_t)file.st_blocks <= UINT64_MAX / ALLOCATED_BLOCK_BYTES;
	struct copy_state prior;
	if (partial_exists) {
		value->bytes = file.st_size < 0 ? 0u : (uint64_t)file.st_size;
		value->allocated = file.st_blocks < 0 ?
					   0u :
					   (uint64_t)file.st_blocks *
						   ALLOCATED_BLOCK_BYTES;
		value->kind = sized && journal_valid(journal, manifest, key,
						     &file, &prior) ?
				      PREPARE_CHECKPOINT_ENGINE :
				      PREPARE_CHECKPOINT_FOREIGN;
		value->adopted = value->kind == PREPARE_CHECKPOINT_ENGINE &&
				 prior.adopted != 0;
	}
	const bool engine = value->kind == PREPARE_CHECKPOINT_ENGINE;
	if (value->beside_final &&
	    (!owned_file(&final_file) || final_file.st_size < 0 ||
	     (uint64_t)final_file.st_size > manifest->bytes)) {
		value->refusal = "finalized_replica_unverified";
		return;
	}
	if (partial_exists && !engine) {
		value->why =
			value->beside_final ?
				"a final file is beside it" :
			!sized ?
				"it is not an owned regular file of at most the pack's size" :
				NULL;
		if (!value->beside_final && !sized)
			return;
	}
	uint64_t allocated = sized ? value->allocated : 0u;
	if (partial_exists && sized && !value->beside_final) {
		const int extents =
			openat(directory, partial_name,
			       O_RDONLY | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK);
		struct prepare_extent_map map;
		if (extents < 0 ||
		    prepare_extents(extents, &map) != STATUS_OK ||
		    map.needs_allocation)
			allocated = 0;
		if (extents >= 0)
			(void)close(extents);
	}
	const uint64_t remaining =
		value->beside_final	    ? 0 :
		manifest->bytes > allocated ? manifest->bytes - allocated :
					      0;
	const uint64_t credit =
		value->beside_final && engine && !value->adopted ? allocated :
								   0;
	struct local_device_description device;
	value->refusal =
		local_device_describe(parent, &device) != STATUS_OK ?
			"device_disappeared_or_unknown" :
			local_device_space_refusal(
				device.total_bytes,
				prepare_available_with_credit(
					device.total_bytes,
					device.available_bytes, credit),
				remaining, floor_percent, 0, device.readonly);
	if (value->refusal != NULL) {
		/* a read-only filesystem never lets a run unlink */
		value->removed = engine && !value->adopted &&
				 !value->beside_final && !device.readonly &&
				 checkpoint_removable(value->refusal);
		return;
	}
	if (!partial_exists || engine || value->beside_final)
		return;
	const uint32_t proven = adoption_proof(directory, partial_name,
					       manifest, &proof, &value->why);
	const uint64_t bytes = (uint64_t)proven * manifest->chunk_bytes;
	value->proven_bytes = bytes < manifest->bytes ? bytes : manifest->bytes;
	if (proven != 0)
		value->kind = PREPARE_CHECKPOINT_ADOPTABLE;
}
result_status
prepare_checkpoint_classify(const struct prepare_manifest *manifest,
			    const char *target_path, const char *key,
			    uint32_t floor_percent,
			    struct prepare_checkpoint *output)
{
	char parent[PATH_MAX], real[PATH_MAX];
	if (manifest == NULL || target_path == NULL || key == NULL ||
	    output == NULL || target_path[0] != '/' ||
	    snprintf(parent, sizeof(parent), "%s", target_path) >=
		    (int)sizeof(parent))
		return STATUS_INVALID_ARGUMENT;
	*strrchr(parent, '/') = '\0';
	struct prepare_checkpoint value = { .kind = PREPARE_CHECKPOINT_NONE,
					    .why = "" };
	/* As target_open: the resolved directory, under anemone_engram_rnd and owned, opened without following. */
	struct stat directory_file, opened;
	const int directory =
		realpath(parent[0] != '\0' ? parent : "/", real) == NULL ||
				stat(real, &directory_file) != 0 ?
			-1 :
			open(real,
			     O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
	if (directory < 0 || fstat(directory, &opened) != 0) {
		value.kind = PREPARE_CHECKPOINT_UNKNOWN;
		value.error = errno;
	} else if (strstr(real, "/anemone_engram_rnd/") == NULL ||
		   !S_ISDIR(directory_file.st_mode) ||
		   directory_file.st_uid != geteuid() ||
		   opened.st_dev != directory_file.st_dev ||
		   opened.st_ino != directory_file.st_ino) {
		value.kind = PREPARE_CHECKPOINT_UNKNOWN;
		value.refusal = "destination_directory_policy";
	} else {
		checkpoint_examine(manifest, parent, directory,
				   strrchr(target_path, '/') + 1, key,
				   floor_percent, &value);
	}
	if (directory >= 0)
		(void)close(directory);
	if (value.why == NULL)
		value.why = "";
	*output = value;
	return STATUS_OK;
}
static void stage_add(struct copy_shared *shared, enum copy_stage stage,
		      uint64_t started, uint64_t bytes)
{
	const uint64_t now = copy_now();
	atomic_fetch_add(&shared->stage_ns[stage],
			 now > started ? now - started : 0);
	atomic_fetch_add(&shared->stage_bytes[stage], bytes);
}
static bool timed_transfer(struct copy_worker *worker, enum copy_stage stage,
			   const int *descriptors, uint32_t count, void *buffer,
			   size_t bytes, uint64_t offset, bool write,
			   bool *failed)
{
	const uint64_t started = copy_now();
	if (stage == STAGE_READBACK_READ && worker->ring.descriptor >= 0 &&
	    FAULT_POINT(prepare_copy_readback_poison))
		worker->ring.poisoned =
			true; /* as an io_uring_enter refused mid-batch */
	const bool done = transfer(worker, descriptors, count, buffer, bytes,
				   offset, write, failed);
	stage_add(worker->shared, stage, started, (uint64_t)bytes * count);
	return done;
}
MUST_CHECK static result_status
timed_hash(struct copy_shared *shared, enum copy_stage stage,
	   const void *buffer, size_t bytes,
	   char digest[STORAGE_PROFILE_KEY_BYTES])
{
	const uint64_t started = copy_now();
	const result_status status = prepare_hash(buffer, bytes, digest);
	stage_add(shared, stage, started, bytes);
	return status;
}
static bool timed_flush(struct copy_shared *shared, int descriptor,
			uint64_t bytes)
{
	const uint64_t started = copy_now();
	const bool done = fdatasync(descriptor) == 0;
	stage_add(shared, STAGE_FLUSH, started, bytes);
	return done;
}
static bool timed_compare(struct copy_shared *shared, const void *readback,
			  const void *source, size_t bytes)
{
	const uint64_t started = copy_now();
	const bool equal = memcmp(readback, source, bytes) == 0;
	stage_add(shared, STAGE_READBACK_COMPARE, started, bytes);
	return equal;
}
/* A readback's digest equal to the manifest's (with a final still to check against the source this chunk). */
static bool timed_digest_is(struct copy_shared *shared, const void *buffer,
			    size_t bytes, const char *manifest_digest)
{
	char digest[STORAGE_PROFILE_KEY_BYTES];
	return timed_hash(shared, STAGE_READBACK_HASH, buffer, bytes, digest) ==
		       STATUS_OK &&
	       strcmp(digest, manifest_digest) == 0;
}
/* checking builds: a source chunk that fails its digest while the run stops */
static bool source_mismatch_fault(struct copy_shared *shared)
{
	if (!FAULT_POINT(prepare_copy_stop_mismatch))
		return false;
	(void)pthread_mutex_lock(&shared->mutex);
	shared->fault_stop = true;
	(void)pthread_mutex_unlock(&shared->mutex);
	return true;
}
/* A source chunk's digest (expected) against the manifest's when it has one; recorded in the manifest when it has
 * none. False on a mismatch: a bad source read. */
static bool source_recorded(struct copy_shared *shared, uint32_t chunk,
			    size_t bytes,
			    const char expected[STORAGE_PROFILE_KEY_BYTES])
{
	struct prepare_manifest *const manifest = shared->options->manifest;
	if (manifest->hashes[chunk][0] != '\0' &&
	    strcmp(expected, manifest->hashes[chunk]) != 0)
		return false;
	(void)pthread_mutex_lock(&shared->mutex);
	shared->source_bytes += bytes;
	if (manifest->hashes[chunk][0] == '\0') {
		memcpy(manifest->hashes[chunk], expected,
		       STORAGE_PROFILE_KEY_BYTES);
		shared->manifest_changed = true;
	}
	while (manifest->known < manifest->chunks &&
	       manifest->hashes[manifest->known][0] != '\0')
		manifest->known++;
	(void)pthread_mutex_unlock(&shared->mutex);
	return true;
}
/* The source chunk's digest into expected, checked and recorded (source_recorded). False on a hash failure or a
 * mismatch: a bad source read. */
static bool source_verified(struct copy_shared *shared, uint32_t chunk,
			    size_t bytes, const void *buffer,
			    char expected[STORAGE_PROFILE_KEY_BYTES])
{
	return !source_mismatch_fault(shared) &&
	       timed_hash(shared, STAGE_SOURCE_HASH, buffer, bytes, expected) ==
		       STATUS_OK &&
	       source_recorded(shared, chunk, bytes, expected);
}
/* Two compared source chunks of one length hashed in one interleaved pass (prepare_hash_two, the same digests), each
 * then checked and recorded as source_verified does: *first_valid and *second_valid false on a hash failure or a
 * mismatch. */
static void sources_verified_two(struct copy_shared *shared,
				 const struct retained_chunk *first,
				 const struct retained_chunk *second,
				 bool *first_valid, bool *second_valid)
{
	char first_expected[STORAGE_PROFILE_KEY_BYTES],
		second_expected[STORAGE_PROFILE_KEY_BYTES];
	const bool first_fault = source_mismatch_fault(shared),
		   second_fault = source_mismatch_fault(shared);
	const uint64_t started = copy_now();
	const bool hashed = prepare_hash_two(first->buffer, second->buffer,
					     first->bytes, first_expected,
					     second_expected) == STATUS_OK;
	stage_add(shared, STAGE_SOURCE_HASH, started,
		  first->bytes * COPY_HASH_PAIR_CHUNKS);
	*first_valid = !first_fault && hashed &&
		       source_recorded(shared, first->chunk, first->bytes,
				       first_expected);
	*second_valid = !second_fault && hashed &&
			source_recorded(shared, second->chunk, second->bytes,
					second_expected);
}
static bool spot_chunk(const struct copy_target *target, uint32_t chunk)
{
	for (uint32_t index = 0; index < target->spot_count; index++)
		if (target->spot[index] == chunk)
			return true;
	return false;
}
/* With the shared mutex held: once the target has JOURNAL_EPOCH_CHUNKS claimed chunks unsaved, or at the run's end
 * (force), save its journal. A chunk is claimed only once it is durable and verified: a written one after an fdatasync
 * and its readback (chunk_verify), one that needed no write when its read from the device matched. False when the
 * save failed (the target is then failed). */
static bool journal_epoch(struct copy_shared *shared, uint32_t index,
			  bool force)
{
	struct copy_target *const target = &shared->targets[index];
	if (target->unsaved_count == 0 ||
	    (!force && target->unsaved_count < JOURNAL_EPOCH_CHUNKS))
		return true;
	target->unsaved_count = 0;
	const struct prepare_copy_options *const options = shared->options;
	const char *const refusal = checkpoint_room(shared, index);
	const uint64_t journal_started = copy_now();
	const bool saved = refusal == NULL && state_save(target);
	stage_add(shared, STAGE_JOURNAL, journal_started, 0);
	if (!saved) {
		target->failed = true;
		options->targets[index].reason =
			refusal != NULL ? refusal : "copy_io_error";
	}
	return saved;
}
/* With the shared mutex held: the chunk claimed for the target (durable and verified), saved with its epoch. */
static void chunk_claim(struct copy_shared *shared, uint32_t index,
			uint32_t chunk)
{
	struct copy_target *const target = &shared->targets[index];
	target->state.bitmap[chunk / COPY_BITS] |=
		(unsigned char)(1u << (chunk % COPY_BITS));
	target->unsaved_count++;
	(void)journal_epoch(shared, index, false);
}
/* With the shared mutex held: a claimed chunk that is to be written again loses its claim, saved at once, so the
 * journal never claims bytes that no readback after the lag proved. */
static void chunk_unclaim(struct copy_shared *shared, uint32_t index,
			  uint32_t chunk)
{
	struct copy_target *const target = &shared->targets[index];
	unsigned char *const byte = &target->state.bitmap[chunk / COPY_BITS];
	const unsigned char bit = (unsigned char)(1u << (chunk % COPY_BITS));
	if ((*byte & bit) == 0)
		return;
	*byte = (unsigned char)(*byte & ~bit);
	target->unsaved_count++;
	(void)journal_epoch(shared, index, true);
}
/* With the shared mutex held: the manifest saved every MANIFEST_CHECKPOINT_CHUNKS chunks it learns. */
static bool manifest_checkpoint(struct copy_shared *shared)
{
	const struct prepare_copy_options *const options = shared->options;
	return !shared->manifest_changed ||
	       options->manifest->known % MANIFEST_CHECKPOINT_CHUNKS != 0 ||
	       prepare_manifest_write(options->manifest_path,
				      options->manifest) == STATUS_OK;
}
/* One chunk up to its writes: targets that need no write are read and checked against the manifest (and claimed),
 * the source is read into buffer, and the targets that need it are written from it. With writes and no final to check
 * against the source this chunk (hash_after), the source is hashed only once its readbacks were compared with it
 * (chunk_verify): a readback equal to the buffer and the buffer's digest equal to the manifest's prove the target's
 * bytes, and a change of the buffer in memory at any time since its read fails one of the two. A written chunk is kept,
 * with its buffer, for its lagged readbacks (*retained). False: the run fails. */
static bool chunk_write(struct copy_worker *worker, uint32_t chunk,
			void *buffer, bool *retained)
{
	struct copy_shared *const shared = worker->shared;
	const struct prepare_copy_options *const options = shared->options;
	struct prepare_manifest *const manifest = options->manifest;
	const uint64_t offset = (uint64_t)chunk * manifest->chunk_bytes;
	const size_t bytes =
		(size_t)(manifest->bytes - offset < manifest->chunk_bytes ?
				 manifest->bytes - offset :
				 manifest->chunk_bytes);
	bool needs[PREPARE_TARGETS_MOST] = { false },
	     checked[PREPARE_TARGETS_MOST] = { false };
	bool valid = true, source_needed = manifest->hashes[chunk][0] == '\0';
	*retained = false;
	for (uint32_t index = 0; valid && index < options->count; index++) {
		(void)pthread_mutex_lock(&shared->mutex);
		const bool active = !shared->targets[index].failed;
		(void)pthread_mutex_unlock(&shared->mutex);
		if (!active)
			continue;
		struct copy_target *const target = &shared->targets[index];
		needs[index] = target->fresh ||
			       manifest->hashes[chunk][0] == '\0';
		/* a final its matching receipt vouches for: only its spot chunks are read */
		if (!needs[index] && target->receipt_trusted &&
		    !spot_chunk(target, chunk)) {
			(void)pthread_mutex_lock(&shared->mutex);
			options->targets[index].trusted += bytes;
			(void)pthread_mutex_unlock(&shared->mutex);
			continue;
		}
		if (!needs[index]) {
			bool failed = false;
			char digest[STORAGE_PROFILE_KEY_BYTES];
			valid = timed_transfer(worker, STAGE_TARGET_READ,
					       &target->descriptor, 1,
					       worker->verify_buffer, bytes,
					       offset, false, &failed);
			needs[index] =
				failed || !valid ||
				timed_hash(shared, STAGE_TARGET_HASH,
					   worker->verify_buffer, bytes,
					   digest) != STATUS_OK ||
				strcmp(digest, manifest->hashes[chunk]) != 0;
			checked[index] = valid && !needs[index];
			if (checked[index]) {
				(void)pthread_mutex_lock(&shared->mutex);
				options->targets[index].verified += bytes;
				/* a partial's chunk its journal had not claimed may be one a stopped run wrote in its
				 * last lag window, so this read may come from the drive's cache: a readback inside the
				 * lag */
				if (!target->finalized &&
				    ((unsigned)target->state.bitmap[chunk /
								    COPY_BITS] >>
					     (chunk % COPY_BITS) &
				     1u) == 0) {
					shared->readbacks++;
					shared->inside_lag++;
				}
				(void)pthread_mutex_unlock(&shared->mutex);
			}
		}
		if (needs[index] && !target->finalized) {
			(void)pthread_mutex_lock(&shared->mutex);
			chunk_unclaim(shared, index, chunk);
			(void)pthread_mutex_unlock(&shared->mutex);
		}
		if (needs[index] && target->finalized &&
		    manifest->hashes[chunk][0] != '\0') {
			(void)pthread_mutex_lock(&shared->mutex);
			target->failed = true;
			options->targets[index].reason =
				"finalized_replica_unverified";
			verification_fail(target, options->targets[index].path);
			(void)pthread_mutex_unlock(&shared->mutex);
			needs[index] = false;
		}
		source_needed = source_needed || needs[index];
	}
	bool writes = false, finals = false;
	for (uint32_t index = 0; index < options->count; index++) {
		writes = writes ||
			 (needs[index] && !shared->targets[index].finalized);
		finals = finals ||
			 (needs[index] && shared->targets[index].finalized);
	}
	const bool hash_after = writes && !finals;
	char expected[STORAGE_PROFILE_KEY_BYTES];
	if (valid && source_needed) {
		bool failed = false;
		valid = timed_transfer(worker, STAGE_SOURCE_READ,
				       &shared->source, 1, buffer, bytes,
				       offset, false, &failed) &&
			!failed;
		if (valid && !hash_after)
			valid = source_verified(shared, chunk, bytes, buffer,
						expected);
	}
	/* A pre-existing final with no prior manifest is verified after this source chunk is hashed. */
	for (uint32_t index = 0; valid && index < options->count; index++) {
		struct copy_target *const target = &shared->targets[index];
		if (!needs[index] || !target->finalized)
			continue;
		bool failed = false;
		char digest[STORAGE_PROFILE_KEY_BYTES];
		valid = timed_transfer(worker, STAGE_TARGET_READ,
				       &target->descriptor, 1,
				       worker->verify_buffer, bytes, offset,
				       false, &failed);
		const bool good = valid && !failed &&
				  timed_hash(shared, STAGE_TARGET_HASH,
					     worker->verify_buffer, bytes,
					     digest) == STATUS_OK &&
				  strcmp(digest, manifest->hashes[chunk]) == 0;
		(void)pthread_mutex_lock(&shared->mutex);
		if (good)
			options->targets[index].verified += bytes;
		else {
			target->failed = true;
			options->targets[index].reason =
				"finalized_replica_unverified";
			verification_fail(target, options->targets[index].path);
		}
		(void)pthread_mutex_unlock(&shared->mutex);
		needs[index] = false;
	}
	int descriptors[PREPARE_TARGETS_MOST];
	uint32_t indices[PREPARE_TARGETS_MOST], count = 0;
	for (uint32_t index = 0; valid && index < options->count; index++)
		if (needs[index]) {
			descriptors[count] = shared->targets[index].descriptor;
			indices[count++] = index;
		}
	bool failed[PREPARE_TARGETS_MOST] = { false };
	if (valid && count != 0)
		valid = timed_transfer(worker, STAGE_WRITE, descriptors, count,
				       buffer, bytes, offset, true, failed);
	struct retained_chunk kept = { .chunk = chunk,
				       .hash_after = hash_after,
				       .bytes = bytes,
				       .buffer = buffer };
	(void)pthread_mutex_lock(&shared->mutex);
	for (uint32_t ordinal = 0; valid && ordinal < count; ordinal++) {
		const uint32_t index = indices[ordinal];
		if (failed[ordinal]) {
			shared->targets[index].failed = true;
			options->targets[index].reason =
				"copy_device_io_failure";
			continue;
		}
		kept.targets[index] = true;
		kept.written_at[index] = shared->target_written[index];
		shared->target_written[index] += bytes;
		*retained = true;
	}
	if (*retained)
		shared->retained[shared->retained_count++] = kept;
	/* a chunk that needed no write was read from the device and matched: verified, and claimed already when its
	 * journal had it (a read at once proves no flash, so an unclaimed one stays unclaimed) */
	valid = valid && manifest_checkpoint(shared);
	(void)pthread_mutex_unlock(&shared->mutex);
	return valid;
}
/* A written chunk's readbacks, once its lag passed: for each target it went to, an fdatasync unless one that started
 * after the write completed, then its bytes read back (O_DIRECT) and compared with the retained source buffer (or,
 * without hash_after, hashed against the manifest); with hash_after the buffer is hashed last. The chunk is verified
 * for every target that matched, and claimed in its journal (claim) when its lag had passed: the journal holds only
 * chunks a read proved on the flash. False: the run fails (a bad source read). */
static bool chunk_readbacks(struct copy_worker *worker,
			    const struct retained_chunk *kept,
			    bool good[PREPARE_TARGETS_MOST])
{
	struct copy_shared *const shared = worker->shared;
	const struct prepare_copy_options *const options = shared->options;
	struct prepare_manifest *const manifest = options->manifest;
	const uint64_t offset = (uint64_t)kept->chunk * manifest->chunk_bytes;
	for (uint32_t index = 0; index < options->count; index++) {
		if (!kept->targets[index])
			continue;
		struct copy_target *const target = &shared->targets[index];
		(void)pthread_mutex_lock(&shared->mutex);
		const bool active = !target->failed,
			   unflushed = shared->target_flushed[index] <
				       kept->written_at[index] + kept->bytes;
		const uint64_t covered = shared->target_written[index],
			       since = covered - shared->target_flushed[index];
		(void)pthread_mutex_unlock(&shared->mutex);
		if (!active)
			continue;
		bool flushed = true;
		if (unflushed) {
			flushed =
				timed_flush(shared, target->descriptor, since);
			(void)pthread_mutex_lock(&shared->mutex);
			if (flushed && covered > shared->target_flushed[index])
				shared->target_flushed[index] = covered;
			(void)pthread_mutex_unlock(&shared->mutex);
		}
		bool read_failed = false;
		good[index] =
			flushed &&
			timed_transfer(worker, STAGE_READBACK_READ,
				       &target->descriptor, 1,
				       worker->verify_buffer, kept->bytes,
				       offset, false, &read_failed) &&
			!read_failed &&
			(kept->hash_after ?
				 timed_compare(shared, worker->verify_buffer,
					       kept->buffer, kept->bytes) :
				 timed_digest_is(shared, worker->verify_buffer,
						 kept->bytes,
						 manifest->hashes[kept->chunk]));
	}
	/* a readback on a poisoned ring proves nothing, for this target or a later one: no claim, the worker fails */
	return !worker->ring.poisoned;
}
/* A chunk whose readbacks ran (good) with its source settled (valid; false: a bad source read, nothing is claimed):
 * verified for every target that matched, and claimed (claim); then the manifest's checkpoint. False: the run fails. */
static bool chunk_settle(struct copy_shared *shared,
			 const struct retained_chunk *kept,
			 const bool good[PREPARE_TARGETS_MOST], bool claim,
			 bool valid)
{
	const struct prepare_copy_options *const options = shared->options;
	(void)pthread_mutex_lock(&shared->mutex);
	for (uint32_t index = 0; valid && index < options->count; index++) {
		struct copy_target *const target = &shared->targets[index];
		if (!kept->targets[index] || target->failed)
			continue;
		if (!good[index]) {
			target->failed = true;
			options->targets[index].reason =
				"copy_device_io_failure";
			continue;
		}
		options->targets[index].written += kept->bytes;
		options->targets[index].verified += kept->bytes;
		if (claim)
			chunk_claim(shared, index, kept->chunk);
	}
	const bool saved = valid && manifest_checkpoint(shared);
	(void)pthread_mutex_unlock(&shared->mutex);
	return saved;
}
/* The worker's pending chunk hashed alone (the run's end, a stop, or a next chunk of another length) and settled; its
 * buffer is given back in released. */
static bool pending_flush(struct copy_worker *worker,
			  void *released[COPY_RELEASED_MOST],
			  uint32_t *release_count)
{
	if (!worker->has_pending)
		return true;
	worker->has_pending = false;
	char expected[STORAGE_PROFILE_KEY_BYTES];
	const bool valid = source_verified(worker->shared,
					   worker->pending.chunk,
					   worker->pending.bytes,
					   worker->pending.buffer, expected);
	released[(*release_count)++] = worker->pending.buffer;
	return chunk_settle(worker->shared, &worker->pending,
			    worker->pending_good, worker->pending_claim, valid);
}
/* A written chunk's readbacks, then the source's one hash after them (hash_after; a mismatch is a bad source read) and
 * its settling. With two-chunk source hashing the first such chunk waits, its buffer kept, as the worker's pending one,
 * and the next of its length is hashed with it in one pass. The buffers done with are given back in released. */
static bool chunk_verify(struct copy_worker *worker,
			 const struct retained_chunk *kept, bool claim,
			 void *released[COPY_RELEASED_MOST],
			 uint32_t *release_count)
{
	struct copy_shared *const shared = worker->shared;
	bool good[PREPARE_TARGETS_MOST] = { false };
	*release_count = 0;
	if (!chunk_readbacks(worker, kept, good)) {
		released[(*release_count)++] = kept->buffer;
		return false;
	}
	if (!kept->hash_after || !shared->options->pair_source_hashes) {
		char expected[STORAGE_PROFILE_KEY_BYTES];
		const bool valid = !kept->hash_after ||
				   source_verified(shared, kept->chunk,
						   kept->bytes, kept->buffer,
						   expected);
		released[(*release_count)++] = kept->buffer;
		return chunk_settle(shared, kept, good, claim, valid);
	}
	if (worker->has_pending && worker->pending.bytes != kept->bytes &&
	    !pending_flush(worker, released, release_count)) {
		released[(*release_count)++] = kept->buffer;
		return false;
	}
	if (!worker->has_pending) {
		worker->pending = *kept;
		memcpy(worker->pending_good, good, sizeof(good));
		worker->pending_claim = claim;
		worker->has_pending = true;
		return true;
	}
	worker->has_pending = false;
	bool first_valid = false, second_valid = false;
	sources_verified_two(shared, &worker->pending, kept, &first_valid,
			     &second_valid);
	released[(*release_count)++] = worker->pending.buffer;
	released[(*release_count)++] = kept->buffer;
	const bool first_saved =
		chunk_settle(shared, &worker->pending, worker->pending_good,
			     worker->pending_claim, first_valid);
	return chunk_settle(shared, kept, good, claim, second_valid) &&
	       first_saved;
}
/* With the shared mutex held: the oldest written chunk whose readbacks may run, every live target it went to having had
 * the lag written after it (a failed target is written no more, and its readback is skipped), or else with relax (the
 * run's last chunks, or every buffer waiting) the oldest at all, inside the lag; -1 for none. */
static int32_t retained_ready(struct copy_shared *shared, bool relax,
			      bool *inside_lag)
{
	for (uint32_t ordinal = 0; ordinal < shared->retained_count;
	     ordinal++) {
		const struct retained_chunk *const kept =
			&shared->retained[ordinal];
		bool lagged = true;
		for (uint32_t index = 0; index < shared->options->count;
		     index++)
			lagged = lagged && (!kept->targets[index] ||
					    shared->targets[index].failed ||
					    shared->target_written[index] >=
						    kept->written_at[index] +
							    kept->bytes +
							    shared->lag_bytes);
		if (lagged) {
			*inside_lag = false;
			return (int32_t)ordinal;
		}
	}
	*inside_lag = relax && shared->retained_count != 0;
	return relax && shared->retained_count != 0 ? 0 : -1;
}
/* MemAvailable from /proc/meminfo in bytes, or 0 when it cannot be read. */
static uint64_t memory_available(void)
{
	FILE *const stream = fopen("/proc/meminfo", "re");
	if (stream == NULL)
		return 0;
	char line[MEMINFO_LINE_BYTES];
	unsigned long long kibibytes = 0;
	bool found = false;
	while (!found && fgets(line, sizeof(line), stream) != NULL)
		found = sscanf(line, "MemAvailable: %llu kB", &kibibytes) == 1;
	(void)fclose(stream);
	return found ? (uint64_t)kibibytes * KIB_BYTES : 0;
}
/* The source buffers a run keeps: one a worker and one more, and when it writes a target the lagged readback's too,
 * the lag in chunks (at most the copy's), and with two-chunk source hashing one more a worker (its pending chunk). */
static uint32_t buffers_wanted(const struct prepare_copy_options *options,
			       bool writes)
{
	const uint64_t chunk_bytes = options->manifest->chunk_bytes;
	const uint64_t lag_chunks =
		(COPY_READBACK_LAG_BYTES + chunk_bytes - 1u) / chunk_bytes;
	const uint64_t kept = !writes ? 0u :
			      lag_chunks < options->manifest->chunks ?
					lag_chunks :
					options->manifest->chunks;
	const uint32_t pending =
		writes && options->pair_source_hashes ? options->workers : 0u;
	return (uint32_t)kept + options->workers + pending + 1u;
}
/* Whether available (MemAvailable) exceeds a writing run's buffers by COPY_MEMORY_RESERVE_BYTES; a refusal is
 * printed, and named insufficient_memory for every target that has no reason yet. */
static bool memory_admits(const struct prepare_copy_options *options,
			  uint32_t buffers, uint64_t available)
{
	const uint64_t bytes =
		(uint64_t)buffers * options->manifest->chunk_bytes;
	if (available >= bytes + COPY_MEMORY_RESERVE_BYTES)
		return true;
	(void)fprintf(stderr,
		      "storage: copy refused: insufficient_memory (its %" PRIu32
		      " retained source buffers take %" PRIu64
		      " bytes; MemAvailable %" PRIu64
		      " bytes must exceed them by %llu)\n",
		      buffers, bytes, available, COPY_MEMORY_RESERVE_BYTES);
	for (uint32_t index = 0; index < options->count; index++)
		if (options->targets[index].reason == NULL)
			options->targets[index].reason = "insufficient_memory";
	return false;
}
/* The lagged readback's source buffers (buffers_wanted), refused (insufficient_memory) unless available exceeds them
 * by COPY_MEMORY_RESERVE_BYTES. A run that writes no target (every included one final: it only verifies) retains
 * nothing, and is refused for no memory. */
static bool buffers_open(struct copy_shared *shared, uint64_t available)
{
	const struct prepare_copy_options *const options = shared->options;
	const uint64_t chunk_bytes = options->manifest->chunk_bytes;
	bool writes = false;
	for (uint32_t index = 0; index < options->count; index++)
		writes = writes || (options->targets[index].included &&
				    !shared->targets[index].failed &&
				    !shared->targets[index].finalized);
	shared->lag_bytes = options->readback_lag_bytes != 0 ?
				    options->readback_lag_bytes :
				    COPY_READBACK_LAG_BYTES;
	shared->buffer_count = buffers_wanted(options, writes);
	if (writes && !memory_admits(options, shared->buffer_count, available))
		return false;
	shared->buffers = (void **)calloc(shared->buffer_count, sizeof(void *));
	shared->free_buffers =
		(void **)calloc(shared->buffer_count, sizeof(void *));
	shared->retained =
		calloc(shared->buffer_count, sizeof(*shared->retained));
	if (shared->buffers == NULL || shared->free_buffers == NULL ||
	    shared->retained == NULL)
		return false;
	for (uint32_t index = 0; index < shared->buffer_count; index++) {
		if (posix_memalign(&shared->buffers[index], shared->alignment,
				   (size_t)chunk_bytes) != 0) {
			shared->buffers[index] = NULL;
			return false;
		}
		shared->free_buffers[shared->free_count++] =
			shared->buffers[index];
	}
	return true;
}
static void buffers_close(struct copy_shared *shared)
{
	for (uint32_t index = 0;
	     shared->buffers != NULL && index < shared->buffer_count; index++)
		free(shared->buffers[index]);
	free((void *)shared->buffers);
	free((void *)shared->free_buffers);
	free(shared->retained);
}
/* With the shared mutex held: the worker's pending chunk hashed alone and settled (pending_flush) outside the mutex,
 * its buffer back among the free ones. False: the run fails. */
static bool pending_settled(struct copy_worker *worker)
{
	struct copy_shared *const shared = worker->shared;
	void *released[COPY_RELEASED_MOST];
	uint32_t release_count = 0;
	(void)pthread_mutex_unlock(&shared->mutex);
	const bool valid = pending_flush(worker, released, &release_count);
	(void)pthread_mutex_lock(&shared->mutex);
	for (uint32_t index = 0; index < release_count; index++)
		shared->free_buffers[shared->free_count++] = released[index];
	return valid;
}
static void *copy_chunks(void *context)
{
	struct copy_worker *const worker = context;
	struct copy_shared *const shared = worker->shared;
	const struct prepare_copy_options *const options = shared->options;
	const uint32_t chunks = options->manifest->chunks;
	bool valid = true;
	(void)pthread_mutex_lock(&shared->mutex);
	while (valid) {
		/* At a stop no chunk is written any more, and the written ones still waiting are read back at once
		 * (inside the lag, counted, unclaimed) for at most COPY_STOP_DRAIN_SECONDS: a bad write is found in
		 * this run, and the next one reads each unclaimed chunk again, counted inside the lag. */
		const bool stopped = copy_stopped(options) ||
				     shared->fault_stop;
		if (stopped && shared->stop_drain_ns == 0)
			shared->stop_drain_ns =
				copy_now() +
				COPY_STOP_DRAIN_SECONDS * NS_SECOND;
		const bool leaving =
			shared->failure ||
			(stopped && ((shared->retained_count == 0 &&
				      shared->writing == 0) ||
				     copy_now() >= shared->stop_drain_ns));
		if (leaving && !shared->failure && worker->has_pending) {
			valid = pending_settled(worker);
			continue;
		}
		if (leaving) {
			shared->timed_out = shared->timed_out || stopped;
			break;
		}
		const bool draining =
			(shared->next == chunks && shared->writing == 0) ||
			stopped;
		bool inside_lag = false;
		const int32_t ready = retained_ready(
			shared, draining || shared->free_count == 0,
			&inside_lag);
		if (ready >= 0) {
			const struct retained_chunk kept =
				shared->retained[ready];
			memmove(&shared->retained[ready],
				&shared->retained[ready + 1],
				(shared->retained_count - (uint32_t)ready -
				 1u) * sizeof(*shared->retained));
			shared->retained_count--;
			shared->readbacks++;
			shared->inside_lag += inside_lag ? 1u : 0u;
			(void)pthread_mutex_unlock(&shared->mutex);
			void *released[COPY_RELEASED_MOST];
			uint32_t release_count = 0;
			valid = chunk_verify(worker, &kept, !inside_lag,
					     released, &release_count);
			(void)pthread_mutex_lock(&shared->mutex);
			for (uint32_t index = 0; index < release_count; index++)
				shared->free_buffers[shared->free_count++] =
					released[index];
		} else if (!stopped && shared->next < chunks &&
			   shared->free_count != 0) {
			const uint32_t chunk = shared->next++;
			void *const buffer =
				shared->free_buffers[--shared->free_count];
			shared->writing++;
			(void)pthread_mutex_unlock(&shared->mutex);
			bool retained = false;
			valid = chunk_write(worker, chunk, buffer, &retained);
			(void)pthread_mutex_lock(&shared->mutex);
			shared->fault_stop =
				shared->fault_stop ||
				(retained &&
				 FAULT_POINT(prepare_copy_stop_written));
			shared->writing--;
			/* a timed-out ring may still read into the buffer: it is never reused (freed with the run's) */
			if (!retained && !worker->ring.poisoned)
				shared->free_buffers[shared->free_count++] =
					buffer;
		} else if (draining && shared->retained_count == 0 &&
			   worker->has_pending) {
			valid = pending_settled(worker);
		} else if (draining && shared->retained_count == 0) {
			shared->timed_out = shared->timed_out || stopped;
			break;
		} else {
			struct timespec until;
			(void)clock_gettime(CLOCK_REALTIME, &until);
			until.tv_nsec += COPY_WAIT_NANOSECONDS;
			if (until.tv_nsec >= (long)NS_SECOND) {
				until.tv_sec++;
				until.tv_nsec -= (long)NS_SECOND;
			}
			(void)pthread_cond_timedwait(&shared->changed,
						     &shared->mutex, &until);
			continue;
		}
		(void)pthread_cond_broadcast(&shared->changed);
	}
	shared->failure = shared->failure || !valid;
	/* a worker the stop drain's limit failed (its ring poisoned by the stop) ends a lease: timed out and resumable;
	 * any other failure (a source read, a digest, the manifest, a ring) is an I/O failure, stop or not */
	shared->timed_out = shared->timed_out || worker->ring.stop_poisoned;
	shared->hard_failure = shared->hard_failure ||
			       (!valid && !worker->ring.stop_poisoned);
	(void)pthread_cond_broadcast(&shared->changed);
	(void)pthread_mutex_unlock(&shared->mutex);
	return NULL;
}
result_status prepare_journal_claims(const char *target_path, uint32_t chunk,
				     bool *claimed, uint32_t *claimed_count)
{
	char journal[PATH_MAX];
	struct copy_state state;
	struct stat file;
	if (target_path == NULL || claimed == NULL || claimed_count == NULL)
		return STATUS_INVALID_ARGUMENT;
	const int length = snprintf(journal, sizeof(journal),
				    "%s" COPY_STATE_SUFFIX, target_path);
	const int descriptor =
		length > 0 && (size_t)length < sizeof(journal) ?
			open(journal,
			     O_RDONLY | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK) :
			-1;
	const bool read_whole =
		descriptor >= 0 && fstat(descriptor, &file) == 0 &&
		S_ISREG(file.st_mode) && file.st_size == (off_t)sizeof(state) &&
		read(descriptor, &state, sizeof(state)) ==
			(ssize_t)sizeof(state);
	if (descriptor >= 0)
		(void)close(descriptor);
	if (!read_whole || state.schema != COPY_STATE_SCHEMA ||
	    state.chunks > PREPARE_CHUNKS_MOST)
		return STATUS_INPUT_OUTPUT_ERROR;
	uint32_t count = 0;
	for (uint32_t index = 0; index < state.chunks; index++)
		count += ((unsigned)state.bitmap[index / COPY_BITS] >>
				  (index % COPY_BITS) &
			  1u) != 0 ?
				 1u :
				 0u;
	*claimed = chunk < state.chunks &&
		   ((unsigned)state.bitmap[chunk / COPY_BITS] >>
			    (chunk % COPY_BITS) &
		    1u) != 0;
	*claimed_count = count;
	return STATUS_OK;
}
result_status prepare_replicas(const struct prepare_copy_options *options)
{
	if (options == NULL || options->manifest == NULL ||
	    options->targets == NULL || options->manifest_path == NULL ||
	    options->count > PREPARE_TARGETS_MOST || options->workers == 0 ||
	    options->workers > PREPARE_COPY_WORKERS_MOST ||
	    options->floor_percent >= LOCAL_DEVICE_PERCENT ||
	    options->manifest->bytes == 0 ||
	    options->manifest->chunk_bytes == 0 ||
	    options->manifest->chunk_bytes > PREPARE_CHUNK_BYTES ||
	    options->manifest->chunks == 0 ||
	    options->manifest->chunks > PREPARE_CHUNKS_MOST ||
	    options->manifest->known > options->manifest->chunks ||
	    options->manifest->chunks != (options->manifest->bytes -
					  1u) / options->manifest->chunk_bytes +
						 1u ||
	    quarantine_count != 0)
		return STATUS_INVALID_ARGUMENT;
	const uint64_t started = copy_now();
	struct copy_shared *const shared = calloc(1, sizeof(*shared));
	if (shared == NULL)
		return STATUS_OUT_OF_MEMORY;
	shared->options = options;
	shared->source =
		open(options->manifest->source,
		     O_RDONLY | O_DIRECT | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
	struct stat source;
	bool valid =
		shared->source >= 0 && fstat(shared->source, &source) == 0 &&
		S_ISREG(source.st_mode) && source.st_size > 0 &&
		(uint64_t)source.st_size == options->manifest->bytes &&
		prepare_direct_alignment(shared->source, &shared->alignment) ==
			STATUS_OK &&
		options->manifest->bytes % shared->alignment == 0 &&
		options->manifest->chunk_bytes % shared->alignment == 0;
	const bool source_known = valid; /* source holds its open state */
	if (pthread_mutex_init(&shared->mutex, NULL) != 0) {
		if (shared->source >= 0)
			(void)close(shared->source);
		free(shared);
		return STATUS_OUT_OF_MEMORY;
	}
	if (pthread_cond_init(&shared->changed, NULL) != 0) {
		(void)pthread_mutex_destroy(&shared->mutex);
		if (shared->source >= 0)
			(void)close(shared->source);
		free(shared);
		return STATUS_OUT_OF_MEMORY;
	}
	for (uint32_t index = 0; index < options->count; index++) {
		shared->targets[index].descriptor = -1;
		shared->targets[index].lock = -1;
		shared->targets[index].copier_lock = -1;
		shared->targets[index].directory = -1;
	}
	/* A target with neither its final nor its partial, which passes the open's cheap checks (another device than
	 * the source's and every earlier target's, room for the copy), is created by its open at its full size: a run
	 * short of memory for a writing run's buffers is refused before it creates anything. A target the open refuses
	 * creates nothing, so a run that only verifies is not refused for memory. Otherwise the check waits for the
	 * opens. */
	bool creates = false;
	struct local_device_description source_place;
	const bool source_described =
		local_device_describe(options->manifest->source,
				      &source_place) == STATUS_OK;
	for (uint32_t index = 0; valid && index < options->count; index++) {
		struct prepare_target *const target = &options->targets[index];
		char partial[PATH_MAX];
		struct stat existing;
		const int length = snprintf(partial, sizeof(partial),
					    "%s" PREPARE_PARTIAL_SUFFIX,
					    target->path);
		bool fresh =
			length > 0 && (size_t)length < sizeof(partial) &&
			lstat(target->path, &existing) != 0 &&
			errno == ENOENT && lstat(partial, &existing) != 0 &&
			errno == ENOENT && source_described &&
			!local_device_overlaps(&target->device, &source_place);
		for (uint32_t previous = 0; fresh && previous < index;
		     previous++)
			fresh = !local_device_overlaps(
				&target->device,
				&options->targets[previous].device);
		const struct local_device_description given = target->device;
		fresh = fresh &&
			target_room(options, index, options->manifest->bytes,
				    0) == NULL;
		target->device = given; /* the open describes it again */
		creates = creates || fresh;
	}
	const bool memory_refused =
		valid && creates &&
		!memory_admits(options, buffers_wanted(options, true),
			       FAULT_POINT(prepare_copy_memory_short) ?
				       0u :
				       memory_available());
	for (uint32_t index = 0; valid && index < options->count; index++) {
		struct prepare_target *const target = &options->targets[index];
		target->written = 0;
		target->verified = 0;
		target->trusted = 0;
		target->complete = false;
		target->resumable = false;
		target->reason = NULL;
		bool duplicate = false;
		for (uint32_t previous = 0; previous < index; previous++)
			duplicate =
				duplicate ||
				(options->targets[previous].included &&
				 local_device_overlaps(
					 &target->device,
					 &options->targets[previous].device));
		struct local_device_description source_device;
		if (local_device_describe(options->manifest->source,
					  &source_device) != STATUS_OK ||
		    local_device_overlaps(&target->device, &source_device))
			duplicate = true;
		if (duplicate)
			target->reason = "duplicate_physical_device";
		else if (memory_refused)
			target->reason = "insufficient_memory";
		const bool opened = !duplicate && !memory_refused &&
				    target_open(shared, index);
		shared->targets[index].failed = !opened;
		target->included = opened;
		if (!opened)
			(void)checkpoint_settle(&shared->targets[index],
						target->reason);
		if (!opened && target->reason == NULL)
			target->reason = "destination_directory_policy";
		local_device_print(stderr, &target->device,
				   opened ? "eligible" : target->reason,
				   options->manifest->bytes,
				   options->floor_percent, 0, 0);
	}
	if (FAULT_POINT(prepare_copy_pause))
		shared->timed_out = true;
	valid = valid && !memory_refused;
	if (valid && !shared->timed_out)
		valid = buffers_open(shared, memory_available());
	pthread_t threads[PREPARE_COPY_WORKERS_MOST];
	struct copy_worker *workers[PREPARE_COPY_WORKERS_MOST] = { NULL };
	uint32_t launched = 0;
	for (uint32_t index = 0; valid && index < options->workers; index++) {
		/* a worker already started may have seen the deadline: read under the lock it writes under */
		(void)pthread_mutex_lock(&shared->mutex);
		const bool stopped = shared->timed_out;
		(void)pthread_mutex_unlock(&shared->mutex);
		if (stopped)
			break;
		struct copy_worker *const worker = calloc(1, sizeof(*worker));
		if (worker == NULL) {
			valid = false;
			break;
		}
		worker->shared = shared;
		worker->ring.descriptor = -1;
		if (posix_memalign(&worker->verify_buffer, shared->alignment,
				   (size_t)options->manifest->chunk_bytes) !=
		    0) {
			free(worker);
			valid = false;
			break;
		}
		const bool ring_ready = ring_open(&worker->ring);
		(void)fprintf(
			stderr,
			"storage: copy worker %u mechanism=%s buffer_bytes=%" PRIu64
			" queue_depth=%u\n",
			index,
			ring_ready ? "direct_io_uring" :
				     "threaded_direct_preadv_pwritev",
			options->manifest->chunk_bytes, COPY_RING_DEPTH);
		workers[launched] = worker;
		if (pthread_create(&threads[launched], NULL, copy_chunks,
				   worker) != 0) {
			ring_close(&worker->ring);
			free(worker->verify_buffer);
			free(worker);
			valid = false;
			break;
		}
		launched++;
	}
	for (uint32_t index = 0; index < launched; index++) {
		valid = pthread_join(threads[index], NULL) == 0 && valid;
		struct copy_worker *const worker = workers[index];
		if (worker->ring.poisoned) {
			quarantined[quarantine_count++] = worker;
			valid = false;
		} else {
			ring_close(&worker->ring);
			free(worker->verify_buffer);
			free(worker);
		}
	}
	(void)fprintf(
		stderr,
		"storage: copy readbacks=%" PRIu32 " inside_lag=%" PRIu32
		" lag_bytes=%" PRIu64 " retained_buffers=%" PRIu32
		" (inside the lag: the run's last chunks, the oldest when every buffer waited, the chunks a stop "
		"drained, or a resumed chunk its journal had not claimed)\n",
		shared->readbacks, shared->inside_lag, shared->lag_bytes,
		shared->buffer_count);
	/* checking builds: a process killed once its chunks are verified, before the run's end settles its targets */
	if (FAULT_POINT(prepare_copy_killed_verified))
		(void)raise(SIGKILL);
	struct stat source_after;
	const bool source_unchanged =
		source_known && fstat(shared->source, &source_after) == 0 &&
		source_after.st_size == source.st_size &&
		source_after.st_ino == source.st_ino &&
		source_after.st_dev == source.st_dev &&
		source_after.st_mtim.tv_sec == source.st_mtim.tv_sec &&
		source_after.st_mtim.tv_nsec == source.st_mtim.tv_nsec;
	/* a failure that is no stop's own ends the run as an I/O failure, even when it meets a stop */
	bool hard = shared->hard_failure || !source_unchanged;
	valid = valid && !shared->failure && source_unchanged;
	if (shared->manifest_changed &&
	    prepare_manifest_write(options->manifest_path, options->manifest) !=
		    STATUS_OK) {
		valid = false;
		hard = true;
	}
	const double seconds =
		(double)(copy_now() - started) / (double)NS_SECOND;
	for (uint32_t index = 0; index < options->count; index++) {
		struct copy_target *const state = &shared->targets[index];
		struct prepare_target *const target = &options->targets[index];
		const char *refusal =
			target->included ?
				target_room(options, index, 0, state->credit) :
				target->reason;
		char parent_path[PATH_MAX];
		(void)snprintf(parent_path, sizeof(parent_path), "%s",
			       target->path);
		char *const parent_end = strrchr(parent_path, '/');
		if (parent_end != NULL)
			*parent_end = '\0';
		struct stat bound, current, payload;
		if (target->included && refusal == NULL &&
		    (state->directory < 0 ||
		     fstat(state->directory, &bound) != 0 ||
		     stat(parent_path, &current) != 0 ||
		     bound.st_dev != current.st_dev ||
		     bound.st_ino != current.st_ino ||
		     fstatat(state->directory,
			     state->finalized ? state->name :
						state->partial_name,
			     &payload, AT_SYMLINK_NOFOLLOW) != 0 ||
		     payload.st_dev != state->opened.st_dev ||
		     payload.st_ino != state->opened.st_ino))
			refusal = "device_disappeared_or_unknown";
		/* A final is used only as this run opened it: a change of its size, mtime or ctime since (another
		 * writer) means the chunks the run verified, or the receipt it trusted, may no longer be its bytes. */
		if (target->included && refusal == NULL && state->finalized &&
		    !state->failed) {
			struct stat now;
			if (FAULT_POINT(prepare_copy_target_changed))
				target_touch(state->descriptor, &state->opened);
			if (fstat(state->descriptor, &now) != 0 ||
			    !target_unchanged(&state->opened, &now, true)) {
				state->failed = true;
				target->reason = "finalized_replica_unverified";
				verification_fail(state, target->path);
			}
		}
		bool complete = valid && !state->failed &&
				options->manifest->known ==
					options->manifest->chunks &&
				target->verified + target->trusted ==
					options->manifest->bytes &&
				!shared->timed_out;
		if (complete && refusal == NULL) {
			/* a published partial's state once its last byte is durable, before its rename */
			struct stat published = state->opened;
			complete = !FAULT_POINT(prepare_copy_publish) &&
				   fdatasync(state->descriptor) == 0;
			if (complete && !state->finalized) {
				complete = fstat(state->descriptor,
						 &published) == 0;
				if (complete &&
				    FAULT_POINT(prepare_copy_target_changed))
					target_touch(state->descriptor,
						     &published);
				complete = complete &&
					   renameat2(state->directory,
						     state->partial_name,
						     state->directory,
						     state->name,
						     RENAME_NOREPLACE) == 0;
			}
			if (complete)
				complete = fsync(state->directory) == 0;
			char shown[PATH_MAX];
			local_device_display(shown, sizeof(shown),
					     target->device.mount);
			if (complete) {
				const result_status mapped = prepare_extents(
					state->descriptor, &target->extents);
				(void)fprintf(
					stderr,
					"storage: replica %s extent_map=%s count=%" PRIu64
					" fragments=%" PRIu64
					" largest_bytes=%" PRIu64
					" fragmented=%u\n",
					shown,
					mapped == STATUS_OK ? "fiemap" :
							      "unavailable",
					target->extents.count,
					target->extents.fragments,
					target->extents.largest,
					target->extents.fragmented ? 1u : 0u);
				if (target->extents.fragmented)
					(void)fputs(
						"storage: replica warning fragmented_extents\n",
						stderr);
				/* A final (this tune's or one `anemone replicate` published) that verified every
				   manifest byte wins: the engine checkpoint beside it is never resumed, so the lock
				   holder removes it; a foreign one is kept. */
				struct stat beside;
				const bool kept =
					state->finalized &&
					(!state->engine || state->adopted) &&
					lstat(state->partial, &beside) == 0;
				if (!kept)
					remove_owned(state->journal);
				if (state->finalized && !kept)
					remove_owned(state->partial);
				if (kept)
					(void)checkpoint_settle(
						state,
						"verified_identity_and_bytes");
				/* Every byte read and verified: its receipt, of the state its verification started
				   from (a final as the run opened it; a published partial as it was once durable,
				   unchanged since but for its rename's ctime). A spot-checked final keeps the one it
				   has. */
				struct stat recorded = state->opened;
				const bool unchanged =
					state->finalized ||
					(fstat(state->descriptor, &recorded) ==
						 0 &&
					 target_unchanged(&published, &recorded,
							  false));
				const enum receipt_outcome receipt =
					state->receipt_trusted || !unchanged ?
						RECEIPT_NOT_WRITTEN :
						receipt_write(state,
							      options->manifest,
							      target->key,
							      &recorded);
				/* the search's receipt only after that full verification, and only once its receipt is
				   durable: an unsynced verification receipt leaves no search receipt claiming the copy
				   is verified; a spot-checked final keeps the one it has */
				const enum receipt_outcome search =
					receipt != RECEIPT_WRITTEN ?
						RECEIPT_NOT_WRITTEN :
						search_receipt_write(
							state,
							options->manifest);
				if (receipt != RECEIPT_NOT_WRITTEN)
					(void)fprintf(
						stderr,
						"storage: replica %s search receipt %s\n",
						shown,
						search == RECEIPT_WRITTEN ?
							"written" :
						search == RECEIPT_WRITTEN_UNSYNCED ?
							"written, but its directory sync failed" :
							"not written (not a whole pack, the verification "
							"receipt unsynced, or a write failed)");
				(void)fprintf(
					stderr,
					"storage: replica %s verification receipt %s\n",
					shown,
					state->receipt_trusted ?
						"matched; spot-checked" :
					!unchanged ?
						"not written: the file changed while it was published (the next run "
						"verifies in full)" :
					receipt == RECEIPT_WRITTEN ?
						"written" :
					receipt == RECEIPT_WRITTEN_UNSYNCED ?
						"written, but its directory sync failed: a crash may lose it (the next "
						"run then verifies in full)" :
						"not written (the next run verifies in full)");
			}
		}
		/* an unfinished target's last epoch: its claimed chunks saved in its journal */
		bool saved = true;
		if (!complete && !state->finalized && target->resumable &&
		    refusal == NULL) {
			(void)pthread_mutex_lock(&shared->mutex);
			saved = state->unsaved_count != 0 ?
					journal_epoch(shared, index, true) :
					state_save(state);
			(void)pthread_mutex_unlock(&shared->mutex);
		}
		if (!saved)
			target->resumable = false;
		if (state->descriptor >= 0)
			(void)close(state->descriptor);
		target->complete = complete && refusal == NULL;
		target->resumable = !target->complete && target->resumable &&
				    refusal == NULL;
		/* Only the lock holder removes, and before it releases the lock (checkpoint_settle): a target whose
		 * lock another process holds is that process's copy in progress. A target refused at open was settled
		 * then. */
		if (target->included && !state->finalized &&
		    !target->complete &&
		    (refusal != NULL || state->failed || !target->resumable) &&
		    checkpoint_settle(state, refusal != NULL ? refusal :
					     target->reason != NULL ?
							       target->reason :
							       "copy_io_error"))
			target->resumable = false;
		if (state->lock >= 0)
			(void)close(state->lock);
		if (state->copier_lock >= 0)
			(void)close(state->copier_lock);
		if (state->directory >= 0)
			(void)close(state->directory);
		target->seconds = seconds;
		target->included = target->complete;
		target->reason = target->complete ?
					 "verified_identity_and_bytes" :
				 refusal != NULL	? refusal :
				 target->reason != NULL ? target->reason :
							  "copy_io_error";
		struct stat checkpoint;
		if (state->finalized && !target->complete &&
		    strcmp(target->reason, "finalized_replica_unverified") == 0)
			final_kept(target, state->partial[0] != '\0' &&
						   lstat(state->partial,
							 &checkpoint) == 0);
		local_device_print(stderr, &target->device, target->reason, 0,
				   options->floor_percent, 0, 0);
		char copied[PATH_MAX];
		local_device_display(copied, sizeof(copied),
				     target->device.mount);
		(void)fprintf(stderr,
			      "storage: copy %s written_bytes=%" PRIu64
			      " verified_bytes=%" PRIu64
			      " receipt_trusted_bytes=%" PRIu64
			      " seconds=%.3f GB_per_second=%.3f resumable=%u\n",
			      copied, target->written, target->verified,
			      target->trusted, seconds,
			      seconds > 0 ? (double)target->written /
						    DECIMAL_GB / seconds :
					    0.0,
			      target->resumable ? 1u : 0u);
	}
	/* Where the copy's time went: each stage's busy seconds summed over the workers, and its bytes per busy
	 * second. */
	for (uint32_t stage = 0; stage < COPY_STAGES; stage++) {
		const uint64_t busy = atomic_load(&shared->stage_ns[stage]),
			       moved = atomic_load(&shared->stage_bytes[stage]);
		(void)fprintf(
			stderr,
			"storage: copy stage %s busy_seconds=%.3f bytes=%" PRIu64
			" GB_per_busy_second=%.3f workers=%" PRIu32 "\n",
			stage_names[stage], (double)busy / NS_SECOND, moved,
			busy != 0 ? (double)moved / DECIMAL_GB /
					    ((double)busy / NS_SECOND) :
				    0.0,
			options->workers);
	}
	if (shared->source >= 0)
		(void)close(shared->source);
	const bool timed_out = shared->timed_out && !hard;
	(void)fprintf(stderr,
		      "storage: copy source_read_bytes=%" PRIu64
		      " total_seconds=%.3f\n",
		      shared->source_bytes,
		      (double)(copy_now() - started) / (double)NS_SECOND);
	if (valid && !timed_out && options->source_bytes_read != NULL)
		*options->source_bytes_read = shared->source_bytes;
	(void)pthread_cond_destroy(&shared->changed);
	(void)pthread_mutex_destroy(&shared->mutex);
	/* a timed-out ring may still read into a source buffer: they stay with the run's state until exit */
	if (quarantine_count == 0) {
		buffers_close(shared);
		free(shared);
	}
	return timed_out ? STATUS_TIMED_OUT :
	       valid	 ? STATUS_OK :
			   STATUS_INPUT_OUTPUT_ERROR;
}
