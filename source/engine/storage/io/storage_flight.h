/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * storage_flight.h: block reads of a direct reader's copies, kept in flight by I/O threads of their own.
 *
 * Contract:
 *   - storage_flight_open starts threads_per_copy threads for every copy of the reader, each pinned to a CPU the caller
 *     names (or left free) and each owning one io_uring of that copy only: one submitter, its task work deferred to
 *     its own waits where the kernel has that. The copy's descriptor and the flight's block arena are registered with
 *     the ring where the kernel allows it; a ring that refuses either (a locked-memory limit) reads with the plain
 *     descriptor and addresses instead. A ring the kernel refuses with the setup flags is set up again with fewer;
 *     one it refuses entirely fails the open. storage_flight_plain_rings counts the rings set up without the arena
 *     registered.
 *   - A read is one block (PACK_BLOCK_BYTES) at a logical pack offset, from a copy the caller picks, into a slot of
 *     the flight's page-locked arena that storage_flight_slot names. storage_flight_submit hands reads to the copy's
 *     threads and returns at once; the thread that completes a read calls the caller's completion with it there.
 *     The completion owns the slot until it returns; the slot is free again afterwards unless the completion
 *     resubmitted it, or the slot is quarantined.
 *   - A thread submits whatever its submission queue still holds on every turn (a refused or short submit is retried),
 *     busy-polls its ring and inbox for spin_nanoseconds after its last work, then sleeps in the ring's own wait (or on
 *     its inbox when nothing is in flight) for at most a slice. Every read has its own deadline, read_limit after it
 *     was given to the ring (0: STORAGE_READ_LIMIT_SECONDS), whatever else the thread completes meanwhile.
 *   - A ring is retired when a read passes its deadline, the kernel refuses a submit, or a completion names no read of
 *     the ring. Every read it holds then completes with STATUS_TIMED_OUT: one the kernel had taken quarantines its slot
 *     (storage_flight_slot_quarantined: the kernel may still write into it, so it is not submitted again), one whose
 *     submission entry the kernel never took or still in the inbox does not. A retired ring is kept open, never used
 *     again, while the kernel owes it completions. Its thread reaps it: when it was retired for a deadline, a late
 *     completion of a slot it is owed gives that slot back, and once it is owed nothing it is closed; a ring retired
 *     for any other reason keeps its slots out. A thread keeps at most STORAGE_FLIGHT_RETIRED_RINGS_MOST open retired
 *     rings, and takes a fresh ring only while one more place is left. The arena stays mapped at close while any slot
 *     is quarantined.
 *   - A copy with a retired ring takes no read (storage_flight_copy_retired) until storage_flight_reinstate has asked
 *     its retired threads for fresh rings and each set one up.
 *   - storage_flight_in_flight gives the reads a copy has in flight (relaxed), UINT32_MAX for a retired copy: the
 *     caller's shortest-queue choice. storage_flight_reads gives the count whether or not the copy is retired.
 *   - Nothing is allocated after open. close first waits (bounded by twice the read limit) until no read is in flight
 *     with every thread still running, so a completion may still resubmit; then it refuses every submit and stops the
 *     threads, and only then frees them.
 * Fault points: storage_flight_ring (setup refused), storage_flight_thread (a thread does not start),
 * storage_flight_register (the arena's registration refused), storage_flight_hold (a read given to the ring is never
 * submitted to the kernel, as one the kernel never answers: its deadline retires the ring, and its slot stays out),
 * storage_flight_late (a read the kernel took has its ring retired at once, as past its deadline: its completion lands
 * on the retired ring, which gives the slot back), storage_flight_enter_busy (one submit refused with EAGAIN),
 * storage_flight_retire (a ring retired with reads still in its inbox), storage_flight_reinstate_ask (a retired
 * thread not asked, as when another caller asked it first; reached once an ask).
 */
#ifndef STORAGE_FLIGHT_H
#define STORAGE_FLIGHT_H

#include <stdbool.h>
#include <stdint.h>

#include "engine/formats/pack/pack.h"
#include "engine/storage/storage_seam.h"

#define STORAGE_FLIGHT_THREADS_PER_COPY_MOST 4u
#define STORAGE_FLIGHT_CPU_FREE (-1)
#define STORAGE_FLIGHT_RETIRED_RINGS_MOST 4u
#define STORAGE_FLIGHT_AGE_BUCKETS 32u
#define STORAGE_FLIGHT_AGE_FIRST_NANOSECONDS \
	1024ull /* bucket b counts ages below this times 2^b (the last, any age above) */

struct storage_flight;
struct storage_flight_read {
	uint64_t offset; /* logical pack offset of the block */
	uint64_t tag; /* the caller's */
	uint32_t copy, slot;
	result_status status; /* set before the completion runs */
};
typedef void (*storage_flight_completion)(void *context,
					  struct storage_flight_read *read);
struct storage_flight_options {
	uint32_t threads_per_copy;
	uint32_t slots; /* blocks of the arena: reads in flight at once, over every copy */
	uint32_t ring_entries; /* per thread */
	uint64_t spin_nanoseconds;
	uint64_t read_limit_nanoseconds; /* a read's deadline after it was given to the ring; 0: the storage limit */
	int32_t cpus[STORAGE_MAXIMUM_COPIES]
		    [STORAGE_FLIGHT_THREADS_PER_COPY_MOST];
};

/* What the flight saw since its open: retired rings by reason, rings set up again, reads answered by the drive, and
 * their ages from the hand-off to the ring (power-of-two buckets and the most). */
struct storage_flight_counts {
	uint64_t read_limit_ns, answered, age_most_ns;
	uint64_t age_buckets[STORAGE_FLIGHT_AGE_BUCKETS];
	uint32_t deadline_retirements, other_retirements, reinstatements;
};

MUST_CHECK result_status
storage_flight_open(const struct storage_reader *reader,
		    const struct storage_flight_options *options,
		    storage_flight_completion completion, void *context,
		    struct storage_flight **flight_out);
void storage_flight_close(struct storage_flight *flight);
/* The slot's bytes (PACK_BLOCK_BYTES, aligned to the block), valid while the flight is open. */
unsigned char *storage_flight_slot(struct storage_flight *flight,
				   uint32_t slot);
/* Hands count reads to their copies' threads, in order. STATUS_INVALID_ARGUMENT for a copy, slot or offset out of
 * range, or a slot in flight or quarantined (then none is handed over); STATUS_BUSY_VALUE when a read's copy has a
 * retired ring or the flight is closing. *handed is how many of the reads, from the first, were handed over: the
 * caller still owns the others. */
MUST_CHECK result_status storage_flight_submit(
	struct storage_flight *flight, struct storage_flight_read *const *reads,
	uint32_t count, uint32_t *handed);
uint32_t storage_flight_in_flight(const struct storage_flight *flight,
				  uint32_t copy);
uint64_t storage_flight_read_limit(const struct storage_flight *flight);
bool storage_flight_copy_retired(const struct storage_flight *flight,
				 uint32_t copy);
bool storage_flight_slot_quarantined(const struct storage_flight *flight,
				     uint32_t slot);
/* Asks a copy's retired threads for fresh rings and returns at once: each sets its ring up on its own thread, and the
 * copy takes reads again once none is retired. False when a thread of the copy refused since the last call (its open
 * retired rings were the most it keeps, or a ring could not be set up); that answer is taken, and the next call asks
 * again. */
bool storage_flight_reinstate(struct storage_flight *flight, uint32_t copy);
uint32_t storage_flight_reads(const struct storage_flight *flight,
			      uint32_t copy);
uint32_t storage_flight_plain_rings(const struct storage_flight *flight);
/* The counts so far (each thread's own counters, read relaxed: exact once the flight is idle); all zero for NULL.
 * storage_flight_read_limit and storage_flight_plain_rings also take NULL (0). */
void storage_flight_counts(const struct storage_flight *flight,
			   struct storage_flight_counts *counts);
/* The upper bound of the bucket holding the part / parts quantile of the answered reads' ages (the most seen when no
 * read was answered). */
uint64_t storage_flight_age_quantile(const struct storage_flight_counts *counts,
				     uint64_t parts, uint64_t part);
/* Where a copy's thread is best pinned: thread 0 on the CPU that completes the drive's first I/O queue (its
 * interrupt vector's effective CPU), thread 1 on that CPU's SMT sibling, any other thread (or a copy on no NVMe
 * drive, or a figure that cannot be read) STORAGE_FLIGHT_CPU_FREE. Reads /sys and /proc only. */
int32_t storage_flight_queue_cpu(const struct storage_reader *reader,
				 uint32_t copy, uint32_t thread);

#endif
