/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * storage_flight_linux.c: storage_flight.h on Linux io_uring, raw system calls only.
 *
 * Each thread owns one ring of one copy (registered where the kernel allows: the copy's descriptor, the whole arena as
 * fixed buffers) and an inbox the submitters fill under a mutex. Its loop: move the inbox into the submission queue,
 * submit what the queue holds, reap and call the completion, check every read's own deadline, then spin for the
 * arm's window and sleep in io_uring_enter (or on the inbox futex when nothing is in flight) for a slice. A read's
 * user data is its slot; flight->slot_reads[slot] names the read while it flies. A retired ring's thread completes
 * what it held, keeps the ring open (the kernel may still write into the quarantined slots) and parks until it is
 * reinstated with a fresh ring or the flight closes. Running or parked, it reaps its retired rings: a late completion
 * of a read it was owed gives the slot back (the kernel no longer writes it), and a ring owed nothing is closed.
 */
#include "storage_flight.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <linux/futex.h>
#include <linux/io_uring.h>
#include <linux/time_types.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/sysmacros.h>
#include <sys/uio.h>
#include <time.h>
#include <unistd.h>

#include "common/hooks.h"
#include "storage_batch.h"

#define NANOSECONDS_PER_SECOND 1000000000ull
#define THREAD_NAME_BYTES \
	16u /* the kernel's TASK_COMM_LEN, its terminating byte included */
#define WAIT_SLICE_NANOSECONDS \
	1000000ull /* a sleeping thread looks at its inbox and deadlines this often */
#define READ_LIMIT_NANOSECONDS \
	((uint64_t)STORAGE_READ_LIMIT_SECONDS * NANOSECONDS_PER_SECOND)
#define HUGE_PAGE_BYTES (2ull << 20u)
#define REGISTERED_SPAN_BYTES \
	(1ull << 30u) /* the kernel's limit for one registered buffer */
#define REGISTERED_SPANS_MOST 64u
#define SUBMIT_BATCH_MOST 256u
#define RING_ENTRIES_LEAST 8u
#define SETUP_FLAG_SETS 3u /* deferred task work, then cooperative, then none */
#define FIGURE_TEXT_BYTES 32u
#define INTERRUPT_LINE_BYTES \
	8192u /* a line of /proc/interrupts on a host of a few hundred CPUs */
#define DECIMAL_BASE 10
#define NVME_PREFIX "nvme"
#define FIRST_QUEUE_SUFFIX "q1" /* nvmeXq0 is the admin queue */
#define SIBLING_FIELDS 2
#define AGE_SCAN_NANOSECONDS \
	1000000ull /* a thread looks for a read past its deadline this often, busy or not */
#define CLOSE_DRAIN_LIMITS \
	2u /* close waits this many read limits for the reads in flight */
#define WORD_BITS 64u
#define OWNER_THREAD_SHIFT \
	32u /* a quarantined slot's owner: its thread above, the ring's serial below */
#define RING_INVALID (-1)
enum reinstate_state {
	REINSTATE_IDLE,
	REINSTATE_ASKED,
	REINSTATE_DONE,
	REINSTATE_REFUSED
};

static const uint32_t setup_flag_sets[SETUP_FLAG_SETS] = {
	IORING_SETUP_SINGLE_ISSUER | IORING_SETUP_DEFER_TASKRUN |
		IORING_SETUP_NO_SQARRAY,
	IORING_SETUP_SINGLE_ISSUER | IORING_SETUP_COOP_TASKRUN,
	0u,
};

struct flight_ring {
	int descriptor;
	uint32_t flags, entries, completion_entries;
	bool fixed_file,
		fixed_buffers; /* registered with the ring, or the plain descriptor and addresses */
	unsigned *sq_head, *sq_tail, *sq_mask, *sq_array, *cq_head, *cq_tail,
		*cq_mask;
	struct io_uring_sqe *sqes;
	struct io_uring_cqe *cqes;
	void *ring_memory, *sqe_memory;
	size_t ring_bytes, sqe_bytes;
	unsigned local_tail;
};
/* A retired ring, kept open while the kernel owes it completions: owed slots are quarantined. trusted: it was retired
 * for a read past its deadline, so its completions name its own reads and give their slots back; a ring that named no
 * read of its own, or whose submit the kernel refused, is trusted no more and is kept as it is. */
struct retired_ring {
	struct flight_ring ring;
	uint32_t owed, serial;
	bool trusted;
};
struct flight_thread {
	struct storage_flight *flight;
	uint32_t copy;
	int32_t cpu;
	pthread_t thread;
	bool started;
	_Atomic bool
		retired; /* set by this thread; cleared only when a fresh ring is up */
	bool retire_trusted; /* every reason for the retirement under way was a read past its deadline */
	struct flight_ring ring;
	struct retired_ring graveyard
		[STORAGE_FLIGHT_RETIRED_RINGS_MOST]; /* a live ring has a place left here for its retirement */
	uint32_t graveyard_count, ring_serial;
	bool *unsent; /* scratch of thread_retire: slots whose submission entries the kernel never took */
	uint64_t last_grave_scan_ns;
	/* counts this thread alone writes (relaxed: storage_flight_counts reads them at any time) */
	_Atomic uint64_t age_buckets[STORAGE_FLIGHT_AGE_BUCKETS], age_most_ns,
		answered;
	_Atomic uint32_t deadline_retirements, other_retirements,
		reinstatements;
	int file; /* the copy's descriptor */
	pthread_mutex_t lock;
	uint32_t *inbox, inbox_head,
		inbox_tail; /* slots, flight->slots of room */
	_Atomic uint32_t signal, ready, reinstate;
	/* this thread's slots in flight (flying[position[slot]] == slot) and when each was given to the ring; only this
	 * thread reads or writes them */
	uint32_t *flying, *position, in_flight;
	uint64_t *given_ns, last_age_scan_ns;
	result_status open_status;
};
struct storage_flight {
	const struct storage_reader *reader;
	storage_flight_completion completion;
	void *context;
	unsigned char *arena;
	uint64_t arena_bytes;
	uint32_t slots, copy_count, threads_per_copy;
	uint64_t spin_nanoseconds;
	uint32_t ring_entries;
	struct storage_flight_read **slot_reads;
	bool *quarantined; /* slots a retired ring's kernel may still write into (atomic builtins) */
	/* per quarantined slot: the retired ring owed its completion (read and written by that thread only) */
	uint64_t *quarantine_owner;
	uint64_t read_limit_ns;
	struct flight_thread threads[STORAGE_MAXIMUM_COPIES]
				    [STORAGE_FLIGHT_THREADS_PER_COPY_MOST];
	_Atomic uint32_t in_flight[STORAGE_MAXIMUM_COPIES],
		turn[STORAGE_MAXIMUM_COPIES], stopping, plain_rings;
};

static uint64_t monotonic_nanoseconds(void)
{
	struct timespec now;
	return clock_gettime(CLOCK_MONOTONIC, &now) == 0 ?
		       (uint64_t)now.tv_sec * NANOSECONDS_PER_SECOND +
			       (uint64_t)now.tv_nsec :
		       UINT64_MAX;
}

static void futex_wait(_Atomic uint32_t *word, uint32_t expected,
		       uint64_t nanoseconds)
{
	const struct timespec timeout = {
		.tv_sec = (time_t)(nanoseconds / NANOSECONDS_PER_SECOND),
		.tv_nsec = (long)(nanoseconds % NANOSECONDS_PER_SECOND)
	};
	(void)syscall(SYS_futex, (void *)(uintptr_t)word, FUTEX_WAIT_PRIVATE,
		      expected, &timeout, NULL, 0);
}

static void futex_wake(_Atomic uint32_t *word)
{
	(void)syscall(SYS_futex, (void *)(uintptr_t)word, FUTEX_WAKE_PRIVATE, 1,
		      NULL, NULL, 0);
}

static void cpu_relax(void)
{
#if defined(__x86_64__)
	__builtin_ia32_pause();
#endif
}

static long ring_enter(const struct flight_ring *ring, unsigned to_submit,
		       unsigned min_complete, unsigned flags,
		       const void *argument, size_t argument_bytes)
{
	long result;
	do
		result = syscall(__NR_io_uring_enter, ring->descriptor,
				 to_submit, min_complete, flags, argument,
				 argument_bytes);
	while (result < 0 && errno == EINTR);
	return result;
}

static void ring_close(struct flight_ring *ring)
{
	if (ring->sqe_memory != NULL && ring->sqe_memory != MAP_FAILED)
		(void)munmap(ring->sqe_memory, ring->sqe_bytes);
	if (ring->ring_memory != NULL && ring->ring_memory != MAP_FAILED)
		(void)munmap(ring->ring_memory, ring->ring_bytes);
	if (ring->descriptor >= 0)
		(void)close(ring->descriptor);
	memset(ring, 0, sizeof(*ring));
	ring->descriptor = RING_INVALID;
}

/* In the thread that will submit: a single issuer is the creating task. */
MUST_CHECK static result_status ring_open(struct flight_thread *self,
					  uint32_t entries)
{
	struct flight_ring *const ring = &self->ring;
	struct storage_flight *const flight = self->flight;
	struct io_uring_params parameters = { 0 };
	memset(ring, 0, sizeof(*ring));
	ring->descriptor = RING_INVALID;
	if (FAULT_POINT(storage_flight_ring))
		return STATUS_UNSUPPORTED;
	for (uint32_t set = 0; set < SETUP_FLAG_SETS && ring->descriptor < 0;
	     set++) {
		memset(&parameters, 0, sizeof(parameters));
		parameters.flags = setup_flag_sets[set];
		ring->descriptor =
			(int)syscall(__NR_io_uring_setup, entries, &parameters);
	}
	if (ring->descriptor < 0 ||
	    (parameters.features & IORING_FEAT_SINGLE_MMAP) == 0)
		return STATUS_UNSUPPORTED;
	ring->flags = parameters.flags;
	const size_t submission_bytes =
		(parameters.flags & IORING_SETUP_NO_SQARRAY) != 0 ?
			parameters.sq_off.flags + sizeof(unsigned) :
			parameters.sq_off.array +
				parameters.sq_entries * sizeof(unsigned);
	const size_t completion_bytes =
		parameters.cq_off.cqes +
		parameters.cq_entries * sizeof(struct io_uring_cqe);
	ring->ring_bytes = submission_bytes > completion_bytes ?
				   submission_bytes :
				   completion_bytes;
	ring->ring_memory = mmap(NULL, ring->ring_bytes, PROT_READ | PROT_WRITE,
				 MAP_SHARED | MAP_POPULATE, ring->descriptor,
				 IORING_OFF_SQ_RING);
	ring->sqe_bytes = parameters.sq_entries * sizeof(struct io_uring_sqe);
	ring->sqe_memory = mmap(NULL, ring->sqe_bytes, PROT_READ | PROT_WRITE,
				MAP_SHARED | MAP_POPULATE, ring->descriptor,
				IORING_OFF_SQES);
	if (ring->ring_memory == MAP_FAILED || ring->sqe_memory == MAP_FAILED)
		return STATUS_OUT_OF_MEMORY;
	unsigned char *const base = ring->ring_memory;
	ring->sq_head = (void *)(base + parameters.sq_off.head);
	ring->sq_tail = (void *)(base + parameters.sq_off.tail);
	ring->sq_mask = (void *)(base + parameters.sq_off.ring_mask);
	ring->sq_array = (parameters.flags & IORING_SETUP_NO_SQARRAY) != 0 ?
				 NULL :
				 (void *)(base + parameters.sq_off.array);
	ring->cq_head = (void *)(base + parameters.cq_off.head);
	ring->cq_tail = (void *)(base + parameters.cq_off.tail);
	ring->cq_mask = (void *)(base + parameters.cq_off.ring_mask);
	ring->cqes = (void *)(base + parameters.cq_off.cqes);
	ring->sqes = ring->sqe_memory;
	ring->entries = parameters.sq_entries;
	ring->completion_entries = parameters.cq_entries;
	ring->local_tail = *ring->sq_tail;
	/* Registration is a speed-up only: the descriptor refused, or the arena's pinned pages over the locked-memory
	 * limit (each ring counts them again), leave plain reads of the descriptor into the same aligned slots. */
	ring->fixed_file = syscall(__NR_io_uring_register, ring->descriptor,
				   IORING_REGISTER_FILES, &self->file, 1) == 0;
	struct iovec spans[REGISTERED_SPANS_MOST];
	uint32_t span_count = 0;
	for (uint64_t begin = 0;
	     begin < flight->arena_bytes && span_count < REGISTERED_SPANS_MOST;
	     begin += REGISTERED_SPAN_BYTES) {
		const uint64_t left = flight->arena_bytes - begin;
		spans[span_count++] = (struct iovec){
			.iov_base = flight->arena + begin,
			.iov_len = (size_t)(left < REGISTERED_SPAN_BYTES ?
						    left :
						    REGISTERED_SPAN_BYTES)
		};
	}
	ring->fixed_buffers = !FAULT_POINT(storage_flight_register) &&
			      syscall(__NR_io_uring_register, ring->descriptor,
				      IORING_REGISTER_BUFFERS, spans,
				      span_count) == 0;
	if (!ring->fixed_buffers) {
		(void)atomic_fetch_add(&flight->plain_rings, 1);
		LOG_WARNING_ENTRY(
			"storage flight: copy %u reads without registered buffers",
			self->copy);
	}
	return STATUS_OK;
}

/* A read leaves this thread's flight (it was given to the ring) and its completion runs. */
static void read_complete(struct flight_thread *self, uint32_t slot,
			  result_status status)
{
	struct storage_flight *const flight = self->flight;
	struct storage_flight_read *const read = flight->slot_reads[slot];
	const uint32_t at = self->position[slot];
	const uint32_t last = self->flying[--self->in_flight];
	self->flying[at] = last;
	self->position[last] = at;
	flight->slot_reads[slot] = NULL;
	atomic_fetch_sub_explicit(&flight->in_flight[self->copy], 1,
				  memory_order_relaxed);
	read->status = status;
	flight->completion(flight->context, read);
}

/* A read still in the inbox (never given to the ring) completes. */
static void inbox_complete(struct flight_thread *self, uint32_t slot,
			   result_status status)
{
	struct storage_flight *const flight = self->flight;
	struct storage_flight_read *const read = flight->slot_reads[slot];
	flight->slot_reads[slot] = NULL;
	atomic_fetch_sub_explicit(&flight->in_flight[self->copy], 1,
				  memory_order_relaxed);
	read->status = status;
	flight->completion(flight->context, read);
}

/* The ring is to retire (on this thread, at its next turn); trusted when the reason is a read past its deadline. */
static void ring_retiring(struct flight_thread *self, bool trusted)
{
	self->retire_trusted =
		(!atomic_load(&self->retired) || self->retire_trusted) &&
		trusted;
	atomic_store(&self->retired, true);
}

static void read_flying(struct flight_thread *self, uint32_t slot, uint64_t now)
{
	self->position[slot] = self->in_flight;
	self->flying[self->in_flight++] = slot;
	self->given_ns[slot] = now;
}

/* Moves the inbox into the submission queue, as far as the queue and the completion queue have room, then submits
 * whatever the queue holds, whether or not anything new arrived: a submit the kernel refused (EAGAIN, EBUSY) or took
 * only in part is retried on the next turn. The lock is never held across a completion, which may hand this thread
 * more reads. */
static bool thread_submit(struct flight_thread *self, uint64_t now)
{
	struct storage_flight *const flight = self->flight;
	struct flight_ring *const ring = &self->ring;
	uint32_t added = 0, refused[SUBMIT_BATCH_MOST], refused_count = 0;
	bool late = false;
	if (atomic_load_explicit(&self->signal, memory_order_acquire) != 0 &&
	    FAULT_POINT(storage_flight_retire)) {
		ring_retiring(self, true); /* with reads still in the inbox */
		return false;
	}
	if (atomic_load_explicit(&self->signal, memory_order_acquire) != 0) {
		(void)pthread_mutex_lock(&self->lock);
		const unsigned head =
			__atomic_load_n(ring->sq_head, __ATOMIC_ACQUIRE);
		uint32_t room = ring->entries - (ring->local_tail - head);
		const uint32_t completions =
			ring->completion_entries - self->in_flight;
		room = room < completions ? room : completions;
		while (self->inbox_head != self->inbox_tail && added < room &&
		       added + refused_count < SUBMIT_BATCH_MOST) {
			const uint32_t slot =
				self->inbox[self->inbox_head++ % flight->slots];
			const struct storage_flight_read *const read =
				flight->slot_reads[slot];
			uint64_t physical = 0;
			read_flying(self, slot, now);
			if (!storage_direct_physical_offset(
				    flight->reader, read->copy, read->offset,
				    PACK_BLOCK_BYTES, &physical)) {
				refused[refused_count++] = slot;
				continue;
			}
			if (FAULT_POINT(storage_flight_hold)) {
				added++; /* flying, never submitted: only its deadline ends it */
				continue;
			}
			const unsigned index = ring->local_tail &
					       *ring->sq_mask;
			struct io_uring_sqe *const entry = &ring->sqes[index];
			memset(entry, 0, sizeof(*entry));
			entry->opcode = ring->fixed_buffers ?
						IORING_OP_READ_FIXED :
						IORING_OP_READ;
			entry->flags = ring->fixed_file ? IOSQE_FIXED_FILE : 0;
			entry->fd = ring->fixed_file ? 0 : self->file;
			entry->off = physical;
			entry->addr = (uint64_t)(uintptr_t)storage_flight_slot(
				flight, slot);
			entry->len = PACK_BLOCK_BYTES;
			entry->buf_index =
				ring->fixed_buffers ?
					(uint16_t)((uint64_t)slot *
						   PACK_BLOCK_BYTES /
						   REGISTERED_SPAN_BYTES) :
					0;
			entry->user_data = slot;
			if (ring->sq_array != NULL)
				ring->sq_array[index] = index;
			ring->local_tail++;
			added++;
			late = late || FAULT_POINT(storage_flight_late);
		}
		if (self->inbox_head == self->inbox_tail)
			atomic_store_explicit(&self->signal, 0,
					      memory_order_release);
		(void)pthread_mutex_unlock(&self->lock);
	}
	__atomic_store_n(ring->sq_tail, ring->local_tail, __ATOMIC_RELEASE);
	const unsigned unsent =
		ring->local_tail -
		__atomic_load_n(ring->sq_head, __ATOMIC_ACQUIRE);
	long sent = 0;
	if (unsent != 0) {
		if (FAULT_POINT(storage_flight_enter_busy)) {
			sent = -1;
			errno = EAGAIN;
		} else {
			sent = ring_enter(ring, unsent, 0, 0, NULL, 0);
		}
		if (sent < 0 && errno != EAGAIN && errno != EBUSY)
			ring_retiring(self, false);
	}
	/* as past its deadline, while the kernel serves the read */
	if (late)
		ring_retiring(self, true);
	for (uint32_t index = 0; index < refused_count; index++)
		read_complete(self, refused[index], STATUS_INVALID_ARGUMENT);
	return added + refused_count != 0 || sent > 0;
}

/* A counter only its thread writes: a load and a store, no locked instruction on the read path. */
static void count_add(_Atomic uint64_t *counter, uint64_t amount)
{
	atomic_store_explicit(
		counter,
		atomic_load_explicit(counter, memory_order_relaxed) + amount,
		memory_order_relaxed);
}

/* An answered read's age (from its hand-off to the ring): its power-of-two bucket and the most seen. */
static void age_count(struct flight_thread *self, uint64_t age)
{
	const uint64_t whole = age / STORAGE_FLIGHT_AGE_FIRST_NANOSECONDS;
	const uint32_t bucket =
		whole == 0 ? 0u :
			     (uint32_t)(WORD_BITS -
					(uint32_t)__builtin_clzll(whole));
	count_add(&self->age_buckets[bucket < STORAGE_FLIGHT_AGE_BUCKETS ?
					     bucket :
					     STORAGE_FLIGHT_AGE_BUCKETS - 1u],
		  1u);
	count_add(&self->answered, 1u);
	if (age >
	    atomic_load_explicit(&self->age_most_ns, memory_order_relaxed))
		atomic_store_explicit(&self->age_most_ns, age,
				      memory_order_relaxed);
}

static uint32_t thread_reap(struct flight_thread *self)
{
	struct flight_ring *const ring = &self->ring;
	struct storage_flight *const flight = self->flight;
	if ((ring->flags &
	     (IORING_SETUP_DEFER_TASKRUN | IORING_SETUP_COOP_TASKRUN)) != 0 &&
	    self->in_flight != 0)
		(void)ring_enter(ring, 0, 0, IORING_ENTER_GETEVENTS, NULL, 0);
	unsigned head = *ring->cq_head;
	const unsigned tail = __atomic_load_n(ring->cq_tail, __ATOMIC_ACQUIRE);
	const uint64_t now = head != tail ? monotonic_nanoseconds() : 0u;
	uint32_t reaped = 0;
	while (head != tail && !atomic_load(&self->retired)) {
		const struct io_uring_cqe *const entry =
			&ring->cqes[head & *ring->cq_mask];
		head++;
		const uint64_t slot = entry->user_data;
		const int result = entry->res;
		__atomic_store_n(ring->cq_head, head, __ATOMIC_RELEASE);
		if (slot >= flight->slots || flight->slot_reads[slot] == NULL ||
		    self->in_flight == 0 ||
		    self->flying[self->position[slot]] != slot) {
			/* a completion that names no read of this thread: trust nothing more of this ring */
			ring_retiring(self, false);
			continue;
		}
		age_count(self, now - self->given_ns[slot]);
		read_complete(self, (uint32_t)slot,
			      result == (int)PACK_BLOCK_BYTES ?
				      STATUS_OK :
				      STATUS_INPUT_OUTPUT_ERROR);
		reaped++;
	}
	return reaped;
}

/* A read past its own deadline (read_limit after it was given to the ring), whatever else completes meanwhile: the
 * drive gave no answer in its bound, and the ring is retired. Looked at once a scan interval, busy or not. */
static void thread_deadlines(struct flight_thread *self, uint64_t now)
{
	if (self->in_flight == 0 ||
	    now - self->last_age_scan_ns < AGE_SCAN_NANOSECONDS)
		return;
	self->last_age_scan_ns = now;
	for (uint32_t index = 0; index < self->in_flight; index++)
		if (now - self->given_ns[self->flying[index]] >
		    self->flight->read_limit_ns) {
			ring_retiring(self, true);
			return;
		}
}

/* Which retired ring of which thread owes a quarantined slot's completion. */
static uint64_t owner_tag(const struct flight_thread *self, uint32_t serial)
{
	const uint64_t index =
		(uint64_t)(self - self->flight->threads[self->copy]);
	const uint64_t thread =
		(uint64_t)self->copy * STORAGE_FLIGHT_THREADS_PER_COPY_MOST +
		index + 1u;
	return thread << OWNER_THREAD_SHIFT | serial;
}

/* The ring is retired. Every read the kernel took (or a held one, which stands for a read it never answers)
 * quarantines its slot, owed to the retired ring, and completes as timed out; a read whose submission entry the kernel
 * never took (no polling thread: it takes entries only in this thread's enter, and none follows) completes as timed
 * out unquarantined, and so does every read still in the inbox. A submit that saw the ring live under the lock is in
 * the inbox by now, and none is accepted after. The ring goes to the graveyard (a live ring always has a place there),
 * open while it is owed completions; one owed nothing (the kernel holds none of its reads) is closed at once. */
static void thread_retire(struct flight_thread *self)
{
	struct storage_flight *const flight = self->flight;
	struct flight_ring *const ring = &self->ring;
	(void)pthread_mutex_lock(&self->lock);
	atomic_store(&self->retired, true);
	(void)pthread_mutex_unlock(&self->lock);
	if (ring->descriptor >= 0)
		for (unsigned entry =
			     __atomic_load_n(ring->sq_head, __ATOMIC_ACQUIRE);
		     entry != ring->local_tail; entry++) {
			const uint64_t slot =
				ring->sqes[entry & *ring->sq_mask].user_data;
			if (slot < flight->slots)
				self->unsent[slot] = true;
		}
	struct retired_ring *const grave =
		&self->graveyard[self->graveyard_count++];
	*grave = (struct retired_ring){ .ring = *ring,
					.serial = ++self->ring_serial,
					.trusted = self->retire_trusted };
	_Atomic uint32_t *const retirements =
		self->retire_trusted ? &self->deadline_retirements :
				       &self->other_retirements;
	atomic_store_explicit(
		retirements,
		atomic_load_explicit(retirements, memory_order_relaxed) + 1u,
		memory_order_relaxed);
	memset(ring, 0, sizeof(*ring));
	ring->descriptor = RING_INVALID;
	const uint64_t tag = owner_tag(self, grave->serial);
	while (self->in_flight != 0) {
		const uint32_t slot = self->flying[self->in_flight - 1u];
		if (self->unsent[slot]) {
			self->unsent[slot] = false;
		} else {
			flight->quarantine_owner[slot] = tag;
			__atomic_store_n(&flight->quarantined[slot], true,
					 __ATOMIC_RELEASE);
			grave->owed++;
		}
		read_complete(self, slot, STATUS_TIMED_OUT);
	}
	for (;;) {
		(void)pthread_mutex_lock(&self->lock);
		const bool empty = self->inbox_head == self->inbox_tail;
		const uint32_t slot =
			empty ? 0u :
				self->inbox[self->inbox_head++ % flight->slots];
		if (self->inbox_head == self->inbox_tail)
			atomic_store(&self->signal, 0);
		(void)pthread_mutex_unlock(&self->lock);
		if (empty)
			break;
		inbox_complete(self, slot, STATUS_TIMED_OUT);
	}
	if (grave->owed == 0) {
		ring_close(&grave->ring);
		self->graveyard_count--;
	}
}

/* Reaps the trusted retired rings once a scan interval: a completion of a slot a ring is owed means the kernel no
 * longer writes it, so the slot is given back; any other completion leaves that ring untrusted (its slots stay out). A
 * ring owed nothing more is closed, and its place is free again. */
static void thread_graves(struct flight_thread *self, uint64_t now)
{
	if (self->graveyard_count == 0 ||
	    now - self->last_grave_scan_ns < AGE_SCAN_NANOSECONDS)
		return;
	self->last_grave_scan_ns = now;
	struct storage_flight *const flight = self->flight;
	for (uint32_t index = 0; index < self->graveyard_count;) {
		struct retired_ring *const grave = &self->graveyard[index];
		struct flight_ring *const ring = &grave->ring;
		const uint64_t tag = owner_tag(self, grave->serial);
		if (grave->trusted && grave->owed != 0) {
			if ((ring->flags & (IORING_SETUP_DEFER_TASKRUN |
					    IORING_SETUP_COOP_TASKRUN)) != 0)
				(void)ring_enter(ring, 0, 0,
						 IORING_ENTER_GETEVENTS, NULL,
						 0);
			unsigned head = *ring->cq_head;
			const unsigned tail = __atomic_load_n(ring->cq_tail,
							      __ATOMIC_ACQUIRE);
			while (head != tail && grave->trusted) {
				const uint64_t slot =
					ring->cqes[head & *ring->cq_mask]
						.user_data;
				head++;
				if (slot < flight->slots && grave->owed != 0 &&
				    __atomic_load_n(&flight->quarantined[slot],
						    __ATOMIC_ACQUIRE) &&
				    flight->quarantine_owner[slot] == tag) {
					flight->quarantine_owner[slot] = 0;
					__atomic_store_n(
						&flight->quarantined[slot],
						false, __ATOMIC_RELEASE);
					grave->owed--;
				} else {
					grave->trusted = false;
				}
			}
			__atomic_store_n(ring->cq_head, head, __ATOMIC_RELEASE);
		}
		if (grave->owed == 0) {
			ring_close(ring);
			*grave = self->graveyard[--self->graveyard_count];
			continue;
		}
		index++;
	}
}

/* A retired thread reaps its retired rings and waits for the ask of a fresh ring, or the close. Asked, it sets one up
 * while a graveyard place is left for its retirement, and answers. Returns true when it takes reads again. */
static bool thread_parked(struct flight_thread *self)
{
	struct storage_flight *const flight = self->flight;
	while (atomic_load(&flight->stopping) == 0) {
		thread_graves(self, monotonic_nanoseconds());
		if (atomic_load(&self->reinstate) == REINSTATE_ASKED) {
			const bool fresh =
				self->graveyard_count <
					STORAGE_FLIGHT_RETIRED_RINGS_MOST &&
				ring_open(self, flight->ring_entries) ==
					STATUS_OK;
			if (!fresh && self->ring.descriptor != RING_INVALID)
				ring_close(
					&self->ring); /* a ring that half opened took no read */
			if (fresh) {
				self->last_age_scan_ns = 0;
				self->retire_trusted = true;
				atomic_fetch_add_explicit(&self->reinstatements,
							  1u,
							  memory_order_relaxed);
				atomic_store(&self->retired, false);
			}
			atomic_store(&self->reinstate,
				     fresh ? REINSTATE_DONE :
					     REINSTATE_REFUSED);
			if (fresh)
				return true;
		}
		futex_wait(&self->signal, 0, WAIT_SLICE_NANOSECONDS);
		atomic_store(&self->signal, 0);
	}
	return false;
}

static void *flight_thread_run(void *argument)
{
	struct flight_thread *const self = argument;
	struct storage_flight *const flight = self->flight;
	if (self->cpu >= 0) {
		cpu_set_t set;
		CPU_ZERO(&set);
		CPU_SET((size_t)self->cpu, &set);
		(void)pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
	}
	self->open_status = ring_open(self, flight->ring_entries);
	atomic_store(&self->ready, 1);
	futex_wake(&self->ready);
	uint64_t last_work = monotonic_nanoseconds();
	while (self->open_status == STATUS_OK) {
		if (atomic_load(&self->retired)) {
			thread_retire(self);
			if (!thread_parked(self))
				break;
			last_work = monotonic_nanoseconds();
			continue;
		}
		const uint64_t now = monotonic_nanoseconds();
		const bool submitted = thread_submit(self, now);
		const uint32_t reaped = thread_reap(self);
		thread_deadlines(self, now);
		thread_graves(self, now);
		if (submitted || reaped != 0) {
			last_work = now;
			continue;
		}
		if (atomic_load(&flight->stopping) != 0 &&
		    self->in_flight == 0 && atomic_load(&self->signal) == 0)
			break;
		if (now - last_work < flight->spin_nanoseconds) {
			cpu_relax();
			continue;
		}
		if (self->in_flight != 0) {
			struct __kernel_timespec timeout = {
				.tv_sec = 0,
				.tv_nsec = (long long)WAIT_SLICE_NANOSECONDS
			};
			struct io_uring_getevents_arg wait = {
				.ts = (uint64_t)(uintptr_t)&timeout
			};
			(void)ring_enter(&self->ring, 0, 1,
					 IORING_ENTER_GETEVENTS |
						 IORING_ENTER_EXT_ARG,
					 &wait, sizeof(wait));
		} else {
			futex_wait(&self->signal, 0, WAIT_SLICE_NANOSECONDS);
		}
	}
	return NULL;
}

unsigned char *storage_flight_slot(struct storage_flight *flight, uint32_t slot)
{
	return flight->arena + (uint64_t)slot * PACK_BLOCK_BYTES;
}

/* A copy one of whose threads retired its ring (a drive that did not answer): no more reads go to it. */
bool storage_flight_copy_retired(const struct storage_flight *flight,
				 uint32_t copy)
{
	bool retired = copy >= flight->copy_count;
	for (uint32_t index = 0; !retired && index < flight->threads_per_copy;
	     index++)
		retired = atomic_load(&flight->threads[copy][index].retired);
	return retired;
}

uint32_t storage_flight_in_flight(const struct storage_flight *flight,
				  uint32_t copy)
{
	return !storage_flight_copy_retired(flight, copy) ?
		       atomic_load_explicit(&flight->in_flight[copy],
					    memory_order_relaxed) :
		       UINT32_MAX;
}

uint64_t storage_flight_read_limit(const struct storage_flight *flight)
{
	return flight != NULL ? flight->read_limit_ns : 0u;
}

bool storage_flight_slot_quarantined(const struct storage_flight *flight,
				     uint32_t slot)
{
	return slot >= flight->slots ||
	       __atomic_load_n(&flight->quarantined[slot], __ATOMIC_ACQUIRE);
}

uint32_t storage_flight_reads(const struct storage_flight *flight,
			      uint32_t copy)
{
	return copy < flight->copy_count ?
		       atomic_load_explicit(&flight->in_flight[copy],
					    memory_order_relaxed) :
		       0u;
}

uint32_t storage_flight_plain_rings(const struct storage_flight *flight)
{
	return flight != NULL ? atomic_load(&flight->plain_rings) : 0u;
}

void storage_flight_counts(const struct storage_flight *flight,
			   struct storage_flight_counts *counts)
{
	memset(counts, 0, sizeof(*counts));
	if (flight == NULL)
		return;
	counts->read_limit_ns = flight->read_limit_ns;
	for (uint32_t copy = 0; copy < flight->copy_count; copy++)
		for (uint32_t index = 0; index < flight->threads_per_copy;
		     index++) {
			const struct flight_thread *const self =
				&flight->threads[copy][index];
			counts->deadline_retirements += atomic_load_explicit(
				&self->deadline_retirements,
				memory_order_relaxed);
			counts->other_retirements += atomic_load_explicit(
				&self->other_retirements, memory_order_relaxed);
			counts->reinstatements += atomic_load_explicit(
				&self->reinstatements, memory_order_relaxed);
			counts->answered += atomic_load_explicit(
				&self->answered, memory_order_relaxed);
			const uint64_t most = atomic_load_explicit(
				&self->age_most_ns, memory_order_relaxed);
			counts->age_most_ns = most > counts->age_most_ns ?
						      most :
						      counts->age_most_ns;
			for (uint32_t bucket = 0;
			     bucket < STORAGE_FLIGHT_AGE_BUCKETS; bucket++)
				counts->age_buckets[bucket] +=
					atomic_load_explicit(
						&self->age_buckets[bucket],
						memory_order_relaxed);
		}
}

uint64_t storage_flight_age_quantile(const struct storage_flight_counts *counts,
				     uint64_t parts, uint64_t part)
{
	const uint64_t wanted =
		parts != 0 ? (counts->answered * part + parts - 1u) / parts :
			     0u;
	uint64_t seen = 0;
	for (uint32_t bucket = 0; bucket < STORAGE_FLIGHT_AGE_BUCKETS;
	     bucket++) {
		seen += counts->age_buckets[bucket];
		if (seen >= wanted && seen != 0)
			return STORAGE_FLIGHT_AGE_FIRST_NANOSECONDS << bucket;
	}
	return counts->age_most_ns;
}

/* Each thread's ask: IDLE, then ASKED (here, while it is retired), then its answer DONE or REFUSED (the thread), taken
 * back to IDLE here. A refusal is taken by the next call, which asks nothing of that thread then. */
bool storage_flight_reinstate(struct storage_flight *flight, uint32_t copy)
{
	if (flight == NULL || copy >= flight->copy_count ||
	    atomic_load(&flight->stopping) != 0)
		return false;
	bool refused = false;
	for (uint32_t index = 0; index < flight->threads_per_copy; index++) {
		struct flight_thread *const self =
			&flight->threads[copy][index];
		uint32_t state = atomic_load(&self->reinstate);
		if (state == REINSTATE_REFUSED) {
			refused = true;
			(void)atomic_compare_exchange_strong(
				&self->reinstate, &state, REINSTATE_IDLE);
			continue;
		}
		if (state == REINSTATE_DONE &&
		    atomic_compare_exchange_strong(&self->reinstate, &state,
						   REINSTATE_IDLE))
			state = REINSTATE_IDLE;
		uint32_t idle = REINSTATE_IDLE;
		if (state == REINSTATE_IDLE && atomic_load(&self->retired) &&
		    !FAULT_POINT(storage_flight_reinstate_ask) &&
		    atomic_compare_exchange_strong(&self->reinstate, &idle,
						   REINSTATE_ASKED)) {
			atomic_store(&self->signal, 1);
			futex_wake(&self->signal);
		}
	}
	return !refused;
}

static void thread_free(struct flight_thread *self)
{
	free(self->inbox);
	free(self->flying);
	free(self->position);
	free(self->given_ns);
	free(self->unsent);
	self->inbox = self->flying = self->position = NULL;
	self->given_ns = NULL;
	self->unsent = NULL;
}

static uint32_t reads_in_flight(const struct storage_flight *flight)
{
	uint32_t total = 0;
	for (uint32_t copy = 0; copy < flight->copy_count; copy++)
		total += atomic_load(&flight->in_flight[copy]);
	return total;
}

/* First every read lands or times out with every thread running (a completion may still resubmit, to any copy);
 * then no submit is accepted, the threads stop, and only then is any of them freed. A retired ring still owed a
 * completion stays open, and the arena stays mapped while any slot is quarantined. */
void storage_flight_close(struct storage_flight *flight)
{
	if (flight == NULL)
		return;
	const uint64_t started = monotonic_nanoseconds();
	const uint64_t bound = CLOSE_DRAIN_LIMITS * flight->read_limit_ns;
	while (reads_in_flight(flight) != 0 &&
	       monotonic_nanoseconds() - started < bound) {
		const struct timespec slice = {
			.tv_sec = 0, .tv_nsec = (long)WAIT_SLICE_NANOSECONDS
		};
		(void)nanosleep(&slice, NULL);
	}
	atomic_store(&flight->stopping, 1);
	for (uint32_t copy = 0; copy < flight->copy_count; copy++)
		for (uint32_t index = 0; index < flight->threads_per_copy;
		     index++) {
			struct flight_thread *const self =
				&flight->threads[copy][index];
			atomic_store(&self->signal, 1);
			futex_wake(&self->signal);
		}
	for (uint32_t copy = 0; copy < flight->copy_count; copy++)
		for (uint32_t index = 0; index < flight->threads_per_copy;
		     index++)
			if (flight->threads[copy][index].started)
				(void)pthread_join(
					flight->threads[copy][index].thread,
					NULL);
	for (uint32_t copy = 0; copy < flight->copy_count; copy++)
		for (uint32_t index = 0; index < flight->threads_per_copy;
		     index++) {
			struct flight_thread *const self =
				&flight->threads[copy][index];
			if (!self->started)
				continue;
			if (self->ring.descriptor >= 0)
				ring_close(&self->ring);
			for (uint32_t grave = 0; grave < self->graveyard_count;
			     grave++)
				if (self->graveyard[grave].owed == 0)
					ring_close(
						&self->graveyard[grave].ring);
			(void)pthread_mutex_destroy(&self->lock);
			thread_free(self);
		}
	bool quarantined = false;
	for (uint32_t slot = 0;
	     flight->quarantined != NULL && slot < flight->slots; slot++)
		quarantined = quarantined || flight->quarantined[slot];
	if (flight->arena != NULL && !quarantined)
		(void)munmap(flight->arena, flight->arena_bytes);
	free((void *)flight->slot_reads);
	free(flight->quarantined);
	free(flight->quarantine_owner);
	free(flight);
}

result_status storage_flight_open(const struct storage_reader *reader,
				  const struct storage_flight_options *options,
				  storage_flight_completion completion,
				  void *context,
				  struct storage_flight **flight_out)
{
	struct storage_description description;
	if (reader == NULL || options == NULL || completion == NULL ||
	    flight_out == NULL || options->slots == 0 ||
	    options->threads_per_copy == 0 ||
	    options->threads_per_copy > STORAGE_FLIGHT_THREADS_PER_COPY_MOST ||
	    storage_direct.reader_describe(reader, &description) != STATUS_OK ||
	    description.copy_count == 0 ||
	    PACK_BLOCK_BYTES % description.block_alignment_bytes != 0)
		return STATUS_INVALID_ARGUMENT;
	struct storage_flight *const flight = calloc(1, sizeof(*flight));
	if (flight == NULL)
		return STATUS_OUT_OF_MEMORY;
	flight->reader = reader;
	flight->completion = completion;
	flight->context = context;
	flight->slots = options->slots;
	flight->ring_entries = options->ring_entries > RING_ENTRIES_LEAST ?
				       options->ring_entries :
				       RING_ENTRIES_LEAST;
	flight->copy_count = description.copy_count;
	flight->threads_per_copy = options->threads_per_copy;
	flight->spin_nanoseconds = options->spin_nanoseconds;
	flight->read_limit_ns = options->read_limit_nanoseconds != 0 ?
					options->read_limit_nanoseconds :
					READ_LIMIT_NANOSECONDS;
	flight->arena_bytes = ((uint64_t)options->slots * PACK_BLOCK_BYTES +
			       HUGE_PAGE_BYTES - 1u) /
			      HUGE_PAGE_BYTES * HUGE_PAGE_BYTES;
	flight->arena = mmap(NULL, flight->arena_bytes, PROT_READ | PROT_WRITE,
			     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
	flight->slot_reads = (struct storage_flight_read **)calloc(
		options->slots, sizeof(*flight->slot_reads));
	flight->quarantined =
		calloc(options->slots, sizeof(*flight->quarantined));
	flight->quarantine_owner =
		calloc(options->slots, sizeof(*flight->quarantine_owner));
	result_status status =
		flight->arena != MAP_FAILED && flight->slot_reads != NULL &&
				flight->quarantined != NULL &&
				flight->quarantine_owner != NULL ?
			STATUS_OK :
			STATUS_OUT_OF_MEMORY;
	if (flight->arena == MAP_FAILED)
		flight->arena = NULL;
	if (status == STATUS_OK) {
		(void)madvise(flight->arena, flight->arena_bytes,
			      MADV_HUGEPAGE);
		memset(flight->arena, 0, flight->arena_bytes);
		if (mlock(flight->arena, flight->arena_bytes) != 0)
			LOG_WARNING_ENTRY(
				"storage flight: the block arena is not page-locked");
	}
	for (uint32_t copy = 0;
	     status == STATUS_OK && copy < flight->copy_count; copy++)
		for (uint32_t index = 0;
		     status == STATUS_OK && index < flight->threads_per_copy;
		     index++) {
			struct flight_thread *const self =
				&flight->threads[copy][index];
			self->flight = flight;
			self->copy = copy;
			self->cpu = options->cpus[copy][index];
			self->file = storage_direct_descriptor(reader, copy);
			self->ring.descriptor = RING_INVALID;
			self->inbox =
				calloc(flight->slots, sizeof(*self->inbox));
			self->flying =
				calloc(flight->slots, sizeof(*self->flying));
			self->position =
				calloc(flight->slots, sizeof(*self->position));
			self->given_ns =
				calloc(flight->slots, sizeof(*self->given_ns));
			self->unsent =
				calloc(flight->slots, sizeof(*self->unsent));
			self->retire_trusted = true;
			if (self->inbox == NULL || self->flying == NULL ||
			    self->position == NULL || self->given_ns == NULL ||
			    self->unsent == NULL ||
			    pthread_mutex_init(&self->lock, NULL) != 0) {
				thread_free(self);
				status = STATUS_OUT_OF_MEMORY;
				break;
			}
			if (FAULT_POINT(storage_flight_thread) ||
			    pthread_create(&self->thread, NULL,
					   flight_thread_run, self) != 0) {
				(void)pthread_mutex_destroy(&self->lock);
				thread_free(self);
				status = STATUS_OUT_OF_MEMORY;
				break;
			}
			self->started = true;
			/* "engram io C.T": per-thread CPU accounting (top -H, /proc/<pid>/task/<tid>/comm) tells the
			 * I/O threads apart; a name that does not fit or is refused changes nothing else */
			char name[THREAD_NAME_BYTES];
			const int named = snprintf(name, sizeof(name),
						   "engram io %u.%u", copy,
						   index);
			if (named > 0 && named < (int)sizeof(name))
				(void)pthread_setname_np(self->thread, name);
			while (atomic_load(&self->ready) == 0)
				futex_wait(&self->ready, 0,
					   WAIT_SLICE_NANOSECONDS);
			status = self->open_status;
		}
	if (status != STATUS_OK) {
		storage_flight_close(flight);
		return status;
	}
	*flight_out = flight;
	return STATUS_OK;
}

/* Every read is checked before any is handed over; then each goes to its copy's next thread under that thread's lock,
 * where a retired ring or a closing flight refuses it and every read after it. */
result_status storage_flight_submit(struct storage_flight *flight,
				    struct storage_flight_read *const *reads,
				    uint32_t count, uint32_t *handed)
{
	if (handed != NULL)
		*handed = 0;
	if (flight == NULL || (reads == NULL && count != 0))
		return STATUS_INVALID_ARGUMENT;
	for (uint32_t index = 0; index < count; index++)
		if (reads[index] == NULL ||
		    reads[index]->copy >= flight->copy_count ||
		    reads[index]->slot >= flight->slots ||
		    reads[index]->offset % PACK_BLOCK_BYTES != 0 ||
		    flight->slot_reads[reads[index]->slot] != NULL ||
		    __atomic_load_n(&flight->quarantined[reads[index]->slot],
				    __ATOMIC_ACQUIRE))
			return STATUS_INVALID_ARGUMENT;
	for (uint32_t index = 0; index < count; index++) {
		struct storage_flight_read *const read = reads[index];
		const uint32_t turn =
			atomic_fetch_add_explicit(&flight->turn[read->copy], 1,
						  memory_order_relaxed) %
			flight->threads_per_copy;
		struct flight_thread *const self =
			&flight->threads[read->copy][turn];
		(void)pthread_mutex_lock(&self->lock);
		if (atomic_load(&flight->stopping) != 0 ||
		    atomic_load(&self->retired) ||
		    storage_flight_copy_retired(flight, read->copy)) {
			(void)pthread_mutex_unlock(&self->lock);
			return STATUS_BUSY_VALUE;
		}
		flight->slot_reads[read->slot] = read;
		atomic_fetch_add_explicit(&flight->in_flight[read->copy], 1,
					  memory_order_relaxed);
		self->inbox[self->inbox_tail++ % flight->slots] = read->slot;
		(void)pthread_mutex_unlock(&self->lock);
		if (atomic_exchange(&self->signal, 1) == 0)
			futex_wake(&self->signal);
		if (handed != NULL)
			(*handed)++;
	}
	return STATUS_OK;
}

/* A small number from the first line of a file of /sys or /proc (the first number of a list), or -1. */
static int32_t first_number(const char *path)
{
	char text[FIGURE_TEXT_BYTES] = "";
	const int descriptor = open(path, O_RDONLY | O_CLOEXEC);
	if (descriptor < 0)
		return STORAGE_FLIGHT_CPU_FREE;
	const ssize_t length = read(descriptor, text, sizeof(text) - 1u);
	(void)close(descriptor);
	if (length <= 0 || text[0] < '0' || text[0] > '9')
		return STORAGE_FLIGHT_CPU_FREE;
	char *end = NULL;
	const unsigned long value = strtoul(text, &end, DECIMAL_BASE);
	return value <= (unsigned long)INT32_MAX ? (int32_t)value :
						   STORAGE_FLIGHT_CPU_FREE;
}

/* The interrupt number of a vector named name in /proc/interrupts, or -1. */
static int32_t vector_number(const char *name)
{
	FILE *const interrupts = fopen("/proc/interrupts", "re");
	if (interrupts == NULL)
		return STORAGE_FLIGHT_CPU_FREE;
	static char line[INTERRUPT_LINE_BYTES];
	int32_t number = STORAGE_FLIGHT_CPU_FREE;
	while (number < 0 && fgets(line, sizeof(line), interrupts) != NULL) {
		const size_t line_end = strcspn(line, "\n");
		if (line_end < sizeof(line))
			line[line_end] = '\0';
		const char *const last = strrchr(line, ' ');
		if (last != NULL && strcmp(last + 1, name) == 0) {
			char *end = NULL;
			const unsigned long value =
				strtoul(line, &end, DECIMAL_BASE);
			number = end != line && *end == ':' &&
						 value <= (unsigned long)
								  INT32_MAX ?
					 (int32_t)value :
					 STORAGE_FLIGHT_CPU_FREE;
		}
	}
	(void)fclose(interrupts);
	return number;
}

int32_t storage_flight_queue_cpu(const struct storage_reader *reader,
				 uint32_t copy, uint32_t thread)
{
	struct stat file;
	const int descriptor = storage_direct_descriptor(reader, copy);
	if (thread > 1u || descriptor < 0 || fstat(descriptor, &file) != 0)
		return STORAGE_FLIGHT_CPU_FREE;
	char path[PATH_MAX], resolved[PATH_MAX], vector[PATH_MAX];
	(void)snprintf(path, sizeof(path), "/sys/dev/block/%u:%u",
		       major(file.st_dev), minor(file.st_dev));
	if (realpath(path, resolved) == NULL)
		return STORAGE_FLIGHT_CPU_FREE;
	const bool fits = snprintf(path, sizeof(path), "%s/partition",
				   resolved) < (int)sizeof(path);
	char *const last = strrchr(resolved, '/');
	if (fits && access(path, F_OK) == 0 && last != NULL)
		*last = '\0'; /* a partition: its drive is the directory around it */
	const char *const name = strrchr(resolved, '/') != NULL ?
					 strrchr(resolved, '/') + 1 :
					 resolved;
	const char *const namespace =
		strncmp(name, NVME_PREFIX, strlen(NVME_PREFIX)) == 0 ?
			strchr(name + strlen(NVME_PREFIX), 'n') :
			NULL;
	if (namespace == NULL)
		return STORAGE_FLIGHT_CPU_FREE;
	(void)snprintf(vector, sizeof(vector), "%.*s%s",
		       (int)(namespace - name), name, FIRST_QUEUE_SUFFIX);
	const int32_t number = vector_number(vector);
	if (number < 0)
		return STORAGE_FLIGHT_CPU_FREE;
	(void)snprintf(path, sizeof(path),
		       "/proc/irq/%d/effective_affinity_list", number);
	const int32_t cpu = first_number(path);
	if (thread == 0 || cpu < 0)
		return cpu;
	(void)snprintf(
		path, sizeof(path),
		"/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list",
		cpu);
	FILE *const siblings = fopen(path, "re");
	if (siblings == NULL)
		return STORAGE_FLIGHT_CPU_FREE;
	unsigned first = 0, second = 0;
	const int found = fscanf(siblings, "%u,%u", &first, &second);
	(void)fclose(siblings);
	if (found != SIBLING_FIELDS)
		return STORAGE_FLIGHT_CPU_FREE;
	const unsigned sibling = first == (unsigned)cpu ? second : first;
	return sibling <= (unsigned)INT32_MAX ? (int32_t)sibling :
						STORAGE_FLIGHT_CPU_FREE;
}
