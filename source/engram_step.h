/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * engram_step.h: a step's n-gram blocks, read at once (step I/O), for a disk-held Engram store (engram_disk.h).
 *
 * Contract:
 *   - engram_steps_open starts a storage flight (storage_flight.h) over the store's copies: threads_per_copy I/O
 *     threads per copy, each pinned where cpus says, and steps × slots_per_step block slots. engram_step_create takes
 *     one step's records (at load: one per pipeline slot, or one for a driver without a pipeline).
 *   - engram_step_issue plans every table's rows of row_count rows (row_ids: [row][table][rows per token], as
 *     step_row_ids writes them): blocks are deduplicated; rows from the RAM prefix and from the cache are copied at
 *     once, by the caller's thread; every other block becomes one read, on the copy with the fewest reads in flight
 *     among those whose rings are live (rotating on ties, a copy whose last read failed last), at most one per usable
 *     slot of the step in flight, the rest submitted as earlier ones complete. A table whose destination is NULL is not
 *     read.
 *   - A copy the flight retired comes back from the issues. Once the reads given to it before the retirement are out,
 *     its rings are asked for again after a backoff (reinstate_backoff_nanoseconds, doubled after each failed attempt
 *     up to 512 times), or at once when the store saw it answer a verified read since (engram_disk_copy_verified).
 *     With its fresh rings up it is on probation: no step reads it, and one probe read (a block of table 0, in that
 *     copy's own slot) asks the drive from an I/O thread; its answer puts the copy back. No serving thread waits on
 *     it. A copy that stalls again on probation, or within ENGRAM_STEP_STABLE_BACKOFFS first backoffs of coming
 *     back, waits twice its last backoff (damping); one back longer starts again from the first.
 *   - On the I/O thread, every block is checked against its pack digest; a failed read or digest is read again from a
 *     copy not yet tried whose ring is live. A block no such copy is left for (every copy tried or retired), or whose
 *     slot the flight quarantined, goes to the waiters: the thread in engram_step_wait (of any table of the step)
 *     reads it through the store's synchronous fallback (every copy, two reads each) into a buffer of its own, so no
 *     I/O thread blocks on a synchronous read and no byte comes from a slot a retired ring may still write. A
 *     verified block's rows go to destinations[table] + (row × rows per token + j) × ENGRAM_ROW_BYTES and the block
 *     into the cache. Rows are never taken from unverified bytes, and a quarantined slot is not used again until
 *     the flight gives it back (its read's late completion landed).
 *   - engram_step_wait returns once every row of the table is in place (STATUS_OK), when a block failed on every copy
 *     (its status), or after the flight's read limit times (copies + 1) (STATUS_TIMED_OUT): longer than a stalled
 *     read's own deadline, so its re-read on another copy lands first. It busy-polls for spin_nanoseconds, then sleeps
 *     on the step's counter. A step is issued again only after every table it read was waited for.
 *   - engram_step_destroy drains what the step still has in flight (blocks left to the waiters are finished unread)
 *     and returns true; false only when a read did not return in the waiter's limit (whatever a block's status): the
 *     step stays allocated, and so must every destination it writes into (an I/O thread may still write there).
 *   - Nothing is allocated after create. Close requires every step to be waited for and destroyed, or kept by a
 *     destroy that returned false: close then drains the reads and stops the I/O threads first, and frees the kept
 *     step after (engram_steps_live counts it until then).
 *   - engram_steps_read_limit_within gives the per-read deadline for a caller that waits at most wait_limit on a step
 *     (a pipeline's progress limit): the waiter's whole limit, that deadline times (copies + 1), takes 1 /
 *     ENGRAM_STEP_WAIT_SHARE of it, so a stalled read's re-read on another copy lands inside the caller's limit.
 * Fault points: engram_step_digest (a block's digest fails), engram_step_wait_short (one waiter's limit is a slice),
 * engram_step_probe_lost (a copy's probe its drive never answers: that copy stays on probation).
 */
#ifndef ENGRAM_STEP_H
#define ENGRAM_STEP_H

#include <stdbool.h>
#include <stdint.h>

#include "anemone/engram_disk.h"
#include "storage_flight.h"

#define ENGRAM_STEP_SLOTS \
	1024u /* block reads a step keeps in flight: 4 MiB of the page-locked arena */
#define ENGRAM_STEP_IO_THREADS_DEFAULT \
	2u /* per copy: the bench's best (two a drive) */
#define ENGRAM_STEP_SPIN_US_DEFAULT \
	200u /* busy-poll window after the last work, then sleep */
#define ENGRAM_STEP_SPIN_US_MOST 100000u
#define ENGRAM_STEP_WAIT_SHARE \
	2u /* a step's whole wait takes at most half of the caller's limit */
#define ENGRAM_STEP_REINSTATE_BACKOFF_NANOSECONDS \
	1000000000ull /* a retired copy's first wait before its rings are asked for again: 1 s */
#define ENGRAM_STEP_STABLE_BACKOFFS \
	60u /* a copy back for this many first backoffs (60 s) starts again from the first */

struct engram_steps;
struct engram_step;
struct engram_steps_options {
	uint32_t threads_per_copy, steps, slots_per_step, rows_most;
	uint64_t spin_nanoseconds;
	uint64_t read_limit_nanoseconds; /* the flight's per-read deadline; 0: STORAGE_READ_LIMIT_SECONDS */
	uint64_t reinstate_backoff_nanoseconds; /* 0: ENGRAM_STEP_REINSTATE_BACKOFF_NANOSECONDS */
	int32_t cpus[STORAGE_MAXIMUM_COPIES]
		    [STORAGE_FLIGHT_THREADS_PER_COPY_MOST];
};

MUST_CHECK result_status engram_steps_open(
	struct engram_disk *disk, const struct engram_steps_options *options,
	struct engram_steps **steps_out);
void engram_steps_close(struct engram_steps *steps);
uint64_t engram_steps_read_limit_within(uint64_t wait_limit_nanoseconds,
					uint32_t copies);
MUST_CHECK result_status engram_step_create(struct engram_steps *steps,
					    struct engram_step **step_out);
MUST_CHECK bool engram_step_destroy(struct engram_step *step);
MUST_CHECK result_status engram_step_issue(
	struct engram_step *step, const uint64_t *row_ids, uint32_t row_count,
	unsigned char *const destinations[ENGRAM_MAXIMUM_TABLES]);
MUST_CHECK result_status engram_step_wait(struct engram_step *step,
					  uint32_t table);

/* engram.c: the steps a driver creates its step records from (borrowed; detached by closing the engram). */
MUST_CHECK result_status engram_attach_steps(struct engram_record *engram,
					     struct engram_steps *steps);
struct engram_steps *engram_steps_of(const struct engram_record *engram);

/* engram_disk.c: what step I/O reads of the store; none allocates, all are safe from any thread. */
const struct storage_reader *engram_disk_reader(const struct engram_disk *disk);
uint32_t engram_disk_copy_count(const struct engram_disk *disk);
uint64_t engram_disk_block_offset(const struct engram_disk *disk,
				  uint32_t table, uint64_t block);
uint64_t engram_disk_block_count(const struct engram_disk *disk,
				 uint32_t table);
const unsigned char *engram_disk_ram_block(const struct engram_disk *disk,
					   uint32_t table, uint64_t block);
bool engram_disk_block_good(const struct engram_disk *disk, uint32_t table,
			    uint64_t block, const unsigned char *bytes);
bool engram_disk_cache_rows(struct engram_disk *disk, uint32_t table,
			    uint64_t block, const uint32_t *row_offsets,
			    unsigned char *const *outputs, uint32_t count);
void engram_disk_cache_store(struct engram_disk *disk, uint32_t table,
			     uint64_t block, const unsigned char *bytes);
MUST_CHECK result_status engram_disk_block_fetch(struct engram_disk *disk,
						 uint32_t table, uint64_t block,
						 unsigned char *destination);
uint32_t engram_disk_failed_copies(const struct engram_disk *disk);
void engram_disk_copy_state(struct engram_disk *disk, uint32_t copy,
			    bool failed);
/* How many verified reads a copy has answered (any path), which only grows. */
uint64_t engram_disk_copy_verified(const struct engram_disk *disk,
				   uint32_t copy);
/* The step I/O flight of engram_steps_open, for tests and reports (NULL for none). */
struct storage_flight *engram_steps_flight(const struct engram_steps *steps);
/* How many step records are created and not yet destroyed (a kept one until the close; for tests and reports). */
uint32_t engram_steps_live(const struct engram_steps *steps);
/* The copies on probation (bit per copy): retired, their rings asked for, and not yet answered by the probe. */
uint32_t engram_steps_probation(const struct engram_steps *steps);
/* How many times a copy was put on probation since the open (each retirement brought back counts once). */
uint32_t engram_steps_probations(const struct engram_steps *steps);
/* A copy's current backoff (for tests and reports). */
uint64_t engram_steps_backoff(struct engram_steps *steps, uint32_t copy);

#endif
