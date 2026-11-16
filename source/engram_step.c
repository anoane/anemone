/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * engram_step.c: step I/O for a disk-held Engram store, contract in engram_step.h.
 *
 * The plan is two passes. The first gives every reference of the step (table, row, row of the token) its unique block
 * and chains the references of one block. The second serves each unique block from the RAM prefix or the cache, or
 * makes it a miss. Each usable slot of the step is a lane: the first misses are submitted at once, one a lane, and the
 * completion of one (on its I/O thread) verifies the block, scatters its chain's rows, fills the cache and then gives
 * its lane to the next miss. A block the flight cannot read goes to the step's fallback list, which the waiters read
 * synchronously; a lane whose slot was quarantined goes on there, one block at a time.
 *
 * Ordering: once a table's pending count reaches zero its waiter may issue the step again, so a block's finish (or
 * its hand-over to the fallback list, after which a waiter finishes it) is the last touch of the step by any thread.
 * A lane therefore takes its next miss before its own block is finished or handed over.
 *
 * Recovery: a copy the flight retired is brought back from the issues (copies_recover), through fresh rings and one
 * probe read of its own slot, which no step uses; until the probe answers, the copy takes no step's read.
 */
#include "engram_step.h"

#include <errno.h>
#include <limits.h>
#include <linux/futex.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#include "anemone/hooks.h"

#define NONE UINT32_MAX
#define NANOSECONDS_PER_SECOND 1000000000ull
#define NO_SLOT UINT32_MAX
#define WAIT_SLICE_NANOSECONDS 1000000ull
#define SUBMIT_BATCH 64u
#define PROBE_ROWS_MOST \
	64u /* a block whose chain is longer is read rather than probed */
#define DEDUPE_FACTOR 2u
#define KEY_TABLE_SHIFT 40u
#define HASH_MULTIPLIER 11400714819323198485ull
#define WORD_BITS 64u
#define STEPS_MOST 64u
#define PROBE_BLOCK_STRIDE \
	7919u /* a prime: each probe reads another block of table 0 */
#define BACKOFF_GROWTH 2u /* each failed attempt doubles the backoff */
#define BACKOFF_DOUBLINGS_MOST \
	9u /* a backoff grows to 512 times its first (1 s to 8.5 min by default) */
enum recovery_phase {
	RECOVERY_LIVE,
	RECOVERY_RETIRED, /* noted; its rings are asked for once the backoff passed or it answered a verified read */
	RECOVERY_PROBATION /* fresh rings up; only the probe reads it */
};
enum probe_outcome { PROBE_NONE, PROBE_ANSWERED, PROBE_FAILED };

struct planned_block {
	struct storage_flight_read
		read; /* first: a completion's read is its block */
	struct engram_step *step; /* NULL: the recovery probe */
	uint64_t block;
	uint32_t table, first_reference, references, tried;
	uint32_t next_fallback; /* the fallback list's link */
	bool carries_lane; /* its lane has no slot: the waiter that serves it takes the lane's next miss */
};
struct engram_step {
	struct engram_steps *steps;
	uint32_t base_slot;
	struct planned_block *blocks;
	uint32_t block_count, *miss_order, miss_count;
	uint32_t *next_reference, *reference_row;
	unsigned char **reference_output;
	uint64_t *dedupe_keys;
	uint32_t *dedupe_values, *dedupe_stamps, stamp;
	uint64_t dedupe_capacity;
	_Atomic uint32_t pending[ENGRAM_MAXIMUM_TABLES], next_miss;
	_Atomic int
		failure; /* a result_status: the first block that failed on every copy */
	_Atomic uint32_t issued; /* tables issued and not yet waited for */
	pthread_mutex_t fallback_lock;
	_Atomic uint32_t
		fallback_head; /* blocks the waiters read synchronously; changed under fallback_lock */
	unsigned char *
		fallback_bytes; /* one block a table: its waiter's own buffer */
};
/* A copy's way back from a retirement, under recovery_lock. */
struct copy_recovery {
	enum recovery_phase phase;
	uint64_t noted_verified; /* its verified reads when it was noted retired, or at its last refusal */
	uint64_t due_ns, backoff_ns;
	uint64_t live_since_ns; /* when its probe last put it back (0: never retired) */
	bool refused; /* its threads refused fresh rings: only the backoff, not a verified read, brings the next ask */
};
struct engram_steps {
	struct engram_disk *disk;
	struct storage_flight *flight;
	struct engram_steps_options options;
	struct engram_description description;
	uint32_t copy_count, created, references_most;
	_Atomic uint32_t turn;
	bool in_use[STEPS_MOST];
	/* created and not destroyed: a kept one is freed by the close */
	struct engram_step *records[STEPS_MOST];
	pthread_mutex_t recovery_lock;
	struct copy_recovery recovery[STORAGE_MAXIMUM_COPIES];
	uint64_t backoff_first_ns, probe_count;
	_Atomic uint32_t probations, /* copies put on probation since the open */
		recovering, /* copies whose phase is not RECOVERY_LIVE */
		probation, /* copies that take no step's read */
		probes_flying, /* copies whose probe is in flight */
		probe_outcome[STORAGE_MAXIMUM_COPIES];
	/* one probe a copy, each in a slot of its own after the steps': a probe its drive never answers holds back only
	 * its own copy */
	struct planned_block probes[STORAGE_MAXIMUM_COPIES];
};

static uint64_t monotonic_nanoseconds(void)
{
	struct timespec now;
	return clock_gettime(CLOCK_MONOTONIC, &now) == 0 ?
		       (uint64_t)now.tv_sec * NANOSECONDS_PER_SECOND +
			       (uint64_t)now.tv_nsec :
		       UINT64_MAX;
}

static void cpu_relax(void)
{
#if defined(__x86_64__)
	__builtin_ia32_pause();
#endif
}

/* The copy a block's next read takes: the fewest reads in flight among the copies not yet tried whose rings are live
 * and not on probation, those not marked failed first, rotating where they tie; NONE when no such copy is left. */
static uint32_t copy_choose(struct engram_steps *steps, uint32_t tried)
{
	const uint32_t copies = steps->copy_count, all = (1u << copies) - 1u;
	const uint32_t failed = engram_disk_failed_copies(steps->disk);
	uint32_t live = 0;
	for (uint32_t copy = 0; copy < copies; copy++)
		live |= storage_flight_copy_retired(steps->flight, copy) ?
				0u :
				1u << copy;
	/* after the rings: a copy seen live again had its probation set before its rings were asked for */
	live &= ~atomic_load_explicit(&steps->probation, memory_order_relaxed);
	uint32_t mask = all & ~tried & live & ~failed;
	mask = mask != 0 ? mask : all & ~tried & live;
	if (mask == 0)
		return NONE;
	const uint32_t start = atomic_fetch_add_explicit(&steps->turn, 1,
							 memory_order_relaxed) %
			       copies;
	uint32_t chosen = NONE, least = UINT32_MAX;
	for (uint32_t rank = 0; rank < copies; rank++) {
		const uint32_t copy = (start + rank) % copies;
		if ((mask >> copy & 1u) == 0)
			continue;
		const uint32_t load =
			storage_flight_in_flight(steps->flight, copy);
		if (chosen == NONE || load < least) {
			least = load;
			chosen = copy;
		}
	}
	return chosen;
}

/* The last touch of the step by an I/O thread or a waiter: nothing of the step or the block is read after it. */
static void block_finish(struct engram_step *step,
			 const struct planned_block *block)
{
	_Atomic uint32_t *const pending = &step->pending[block->table];
	if (atomic_fetch_sub(pending, 1) == 1)
		(void)syscall(SYS_futex, (void *)(uintptr_t)pending,
			      FUTEX_WAKE_PRIVATE, INT_MAX, NULL, NULL, 0);
}

static void block_fail(struct engram_step *step, result_status status)
{
	int none = STATUS_OK;
	(void)atomic_compare_exchange_strong(&step->failure, &none,
					     (int)status);
}

/* A local chain of blocks for the fallback list, handed over in one go (the hand-over is a last touch). */
struct fallback_chain {
	uint32_t head, tail;
};

static void chain_add(struct engram_step *step, struct fallback_chain *chain,
		      uint32_t index, bool carries_lane)
{
	step->blocks[index].carries_lane = carries_lane;
	step->blocks[index].next_fallback = NONE;
	if (chain->head == NONE)
		chain->head = index;
	else
		step->blocks[chain->tail].next_fallback = index;
	chain->tail = index;
}

/* Hands a chain to the waiters and wakes those of its tables, under the list's lock: once it is released a waiter may
 * finish every block of the chain, so nothing of the step is touched after it. */
static void chain_hand_over(struct engram_step *step,
			    const struct fallback_chain *chain)
{
	if (chain->head == NONE)
		return;
	uint32_t tables = 0;
	for (uint32_t index = chain->head; index != NONE;
	     index = step->blocks[index].next_fallback)
		tables |= 1u << step->blocks[index].table;
	(void)pthread_mutex_lock(&step->fallback_lock);
	step->blocks[chain->tail].next_fallback = atomic_load_explicit(
		&step->fallback_head, memory_order_relaxed);
	atomic_store_explicit(&step->fallback_head, chain->head,
			      memory_order_release);
	for (uint32_t table = 0; table < ENGRAM_MAXIMUM_TABLES; table++)
		if ((tables >> table & 1u) != 0)
			(void)syscall(SYS_futex,
				      (void *)(uintptr_t)&step->pending[table],
				      FUTEX_WAKE_PRIVATE, INT_MAX, NULL, NULL,
				      0);
	(void)pthread_mutex_unlock(&step->fallback_lock);
}

static uint32_t index_of(const struct engram_step *step,
			 const struct planned_block *block)
{
	return (uint32_t)(block - step->blocks);
}

/* Gives a lane's slot to the step's next misses until one is in flight (or the misses run out). A miss the flight
 * cannot take joins the chain and the slot goes to the one after it; a slot the flight quarantined, or a lane with no
 * slot, puts its next miss on the chain carrying the lane. */
static void lane_next(struct engram_steps *steps, struct engram_step *step,
		      uint32_t slot, struct fallback_chain *chain)
{
	for (;;) {
		const uint32_t next = atomic_fetch_add(&step->next_miss, 1);
		if (next >= step->miss_count)
			return;
		const uint32_t index = step->miss_order[next];
		struct planned_block *const block = &step->blocks[index];
		if (slot == NO_SLOT ||
		    storage_flight_slot_quarantined(steps->flight, slot)) {
			chain_add(step, chain, index, true);
			return;
		}
		block->read.slot = slot;
		block->read.offset = engram_disk_block_offset(
			steps->disk, block->table, block->block);
		block->read.copy = copy_choose(steps, block->tried);
		struct storage_flight_read *read = &block->read;
		if (block->read.copy != NONE &&
		    storage_flight_submit(steps->flight, &read, 1, NULL) ==
			    STATUS_OK)
			return;
		chain_add(step, chain, index, false);
	}
}

/* On the I/O thread that completed the read. */
static void block_completed(void *context, struct storage_flight_read *read)
{
	struct engram_steps *const steps = context;
	struct planned_block *const block =
		(void *)read; /* the read is the block's first member */
	struct engram_step *const step = block->step;
	const uint32_t slot = read->slot;
	const bool quarantined =
		storage_flight_slot_quarantined(steps->flight, slot);
	unsigned char *const bytes = storage_flight_slot(steps->flight, slot);
	const bool faulted = FAULT_POINT(engram_step_digest);
	const bool good = !quarantined && !faulted &&
			  read->status == STATUS_OK &&
			  engram_disk_block_good(steps->disk, block->table,
						 block->block, bytes);
	engram_disk_copy_state(steps->disk, read->copy, !good);
	/* the recovery probe: the drive answered (whatever the digest), or not */
	if (step == NULL) {
		atomic_store(&steps->probe_outcome[read->copy],
			     read->status == STATUS_OK ? PROBE_ANSWERED :
							 PROBE_FAILED);
		(void)atomic_fetch_and(&steps->probes_flying,
				       ~(1u << read->copy));
		return;
	}
	struct fallback_chain chain = { NONE, NONE };
	if (good) {
		for (uint32_t reference = block->first_reference;
		     reference != NONE;
		     reference = step->next_reference[reference])
			memcpy(step->reference_output[reference],
			       bytes + step->reference_row[reference],
			       ENGRAM_ROW_BYTES);
		engram_disk_cache_store(steps->disk, block->table, block->block,
					bytes);
		lane_next(steps, step, slot, &chain);
		chain_hand_over(step, &chain);
		block_finish(step, block);
		return;
	}
	block->tried |= 1u << read->copy;
	if (quarantined) { /* the slot is lost: its lane goes on through the waiters */
		chain_add(step, &chain, index_of(step, block), true);
		chain_hand_over(step, &chain);
		return;
	}
	read->copy = copy_choose(steps, block->tried);
	if (read->copy != NONE &&
	    storage_flight_submit(steps->flight, &read, 1, NULL) == STATUS_OK)
		return;
	lane_next(steps, step, slot, &chain);
	chain_add(step, &chain, index_of(step, block), false);
	chain_hand_over(step, &chain);
}

static void step_free(struct engram_step *step);

/* The flight's close drains every read and joins its threads first, so a step a destroy kept is touched by no thread
 * when it is freed here. */
void engram_steps_close(struct engram_steps *steps)
{
	if (steps == NULL)
		return;
	storage_flight_close(steps->flight);
	for (uint32_t index = 0; index < steps->options.steps; index++)
		if (steps->records[index] != NULL) {
			(void)pthread_mutex_destroy(
				&steps->records[index]->fallback_lock);
			step_free(steps->records[index]);
		}
	(void)pthread_mutex_destroy(&steps->recovery_lock);
	free(steps);
}

uint64_t engram_steps_read_limit_within(uint64_t wait_limit_nanoseconds,
					uint32_t copies)
{
	const uint64_t limit =
		wait_limit_nanoseconds /
		(ENGRAM_STEP_WAIT_SHARE * ((uint64_t)copies + 1u));
	return limit != 0 ? limit : 1u;
}

struct storage_flight *engram_steps_flight(const struct engram_steps *steps)
{
	return steps != NULL ? steps->flight : NULL;
}

uint32_t engram_steps_probation(const struct engram_steps *steps)
{
	return steps != NULL ? atomic_load(&steps->probation) : 0u;
}

uint32_t engram_steps_probations(const struct engram_steps *steps)
{
	return steps != NULL ? atomic_load(&steps->probations) : 0u;
}

uint64_t engram_steps_backoff(struct engram_steps *steps, uint32_t copy)
{
	if (steps == NULL || copy >= steps->copy_count)
		return 0;
	(void)pthread_mutex_lock(&steps->recovery_lock);
	const uint64_t backoff = steps->recovery[copy].backoff_ns;
	(void)pthread_mutex_unlock(&steps->recovery_lock);
	return backoff;
}

uint32_t engram_steps_live(const struct engram_steps *steps)
{
	uint32_t live = 0;
	for (uint32_t index = 0; steps != NULL && index < steps->options.steps;
	     index++)
		live += steps->in_use[index] ? 1u : 0u;
	return live;
}

result_status engram_steps_open(struct engram_disk *disk,
				const struct engram_steps_options *options,
				struct engram_steps **steps_out)
{
	const struct engram_description *const description =
		disk != NULL ? engram_disk_description(disk) : NULL;
	if (disk == NULL || options == NULL || steps_out == NULL ||
	    description == NULL || options->steps == 0 ||
	    options->steps > STEPS_MOST || options->slots_per_step == 0 ||
	    options->rows_most == 0 ||
	    options->slots_per_step >
		    (UINT32_MAX - STORAGE_MAXIMUM_COPIES) / options->steps)
		return STATUS_INVALID_ARGUMENT; /* the probes' slots follow the steps' */
	const uint32_t per_row =
		description->table_count * description->rows_per_token;
	if (per_row == 0 || options->rows_most > UINT32_MAX / per_row)
		return STATUS_INVALID_ARGUMENT;
	struct engram_steps *const steps = calloc(1, sizeof(*steps));
	if (steps == NULL)
		return STATUS_OUT_OF_MEMORY;
	steps->disk = disk;
	steps->options = *options;
	steps->description = *description;
	steps->copy_count = engram_disk_copy_count(disk);
	steps->references_most = options->rows_most * description->table_count *
				 description->rows_per_token;
	steps->backoff_first_ns =
		options->reinstate_backoff_nanoseconds != 0 ?
			options->reinstate_backoff_nanoseconds :
			ENGRAM_STEP_REINSTATE_BACKOFF_NANOSECONDS;
	for (uint32_t copy = 0; copy < steps->copy_count; copy++)
		steps->probes[copy].read.slot =
			options->steps * options->slots_per_step + copy;
	if (pthread_mutex_init(&steps->recovery_lock, NULL) != 0) {
		free(steps);
		return STATUS_OUT_OF_MEMORY;
	}
	struct storage_flight_options flight = {
		.threads_per_copy = options->threads_per_copy,
		.slots = options->steps * options->slots_per_step +
			 steps->copy_count,
		.ring_entries = options->slots_per_step,
		.spin_nanoseconds = options->spin_nanoseconds,
		.read_limit_nanoseconds = options->read_limit_nanoseconds
	};
	memcpy(flight.cpus, options->cpus, sizeof(flight.cpus));
	const result_status status =
		storage_flight_open(engram_disk_reader(disk), &flight,
				    block_completed, steps, &steps->flight);
	if (status != STATUS_OK) {
		(void)pthread_mutex_destroy(&steps->recovery_lock);
		free(steps);
		return status;
	}
	*steps_out = steps;
	return STATUS_OK;
}

/* One block from the fallback list, or NONE. */
static uint32_t fallback_take(struct engram_step *step)
{
	if (atomic_load_explicit(&step->fallback_head, memory_order_acquire) ==
	    NONE)
		return NONE; /* the waiter's spin takes no lock while the list is empty */
	(void)pthread_mutex_lock(&step->fallback_lock);
	const uint32_t index = atomic_load_explicit(&step->fallback_head,
						    memory_order_relaxed);
	if (index != NONE)
		atomic_store_explicit(&step->fallback_head,
				      step->blocks[index].next_fallback,
				      memory_order_relaxed);
	(void)pthread_mutex_unlock(&step->fallback_lock);
	return index;
}

/* A waiter reads a fallback block through the store's synchronous path into its own buffer (with read false, as at
 * destroy, it finishes it unread, recording nothing), scatters its rows, takes its lane's next miss when it carries
 * one, and finishes it. */
static void fallback_serve(struct engram_step *step, uint32_t index,
			   unsigned char *buffer, bool read)
{
	struct engram_steps *const steps = step->steps;
	struct planned_block *const block = &step->blocks[index];
	const result_status status =
		read ? engram_disk_block_fetch(steps->disk, block->table,
					       block->block, buffer) :
		       STATUS_OK;
	if (status == STATUS_OK && read) {
		for (uint32_t reference = block->first_reference;
		     reference != NONE;
		     reference = step->next_reference[reference])
			memcpy(step->reference_output[reference],
			       buffer + step->reference_row[reference],
			       ENGRAM_ROW_BYTES);
		engram_disk_cache_store(steps->disk, block->table, block->block,
					buffer);
	} else if (status != STATUS_OK) {
		block_fail(step, status);
	}
	struct fallback_chain chain = { NONE, NONE };
	if (block->carries_lane)
		lane_next(steps, step, NO_SLOT, &chain);
	chain_hand_over(step, &chain);
	block_finish(step, block);
}

/* The waiter's limit: a stalled read's own deadline, its re-read on every other copy and the fallback. */
static uint64_t wait_limit(const struct engram_steps *steps)
{
	return storage_flight_read_limit(steps->flight) *
	       ((uint64_t)steps->copy_count + 1u);
}

/* Waits until a table's reads have all landed, serving the step's fallback blocks (of any table: a lane's next miss
 * may be another table's) meanwhile; the busy-poll window first, then sleeps of a slice. True once they have (the
 * step's failure says how they went), false when the limit ran out first: the limit only, never a block's status. */
MUST_CHECK static bool table_drain(struct engram_step *step, uint32_t table,
				   bool read)
{
	const uint64_t started = monotonic_nanoseconds();
	const uint64_t spin = step->steps->options.spin_nanoseconds;
	const uint64_t limit = FAULT_POINT(engram_step_wait_short) ?
				       WAIT_SLICE_NANOSECONDS :
				       wait_limit(step->steps);
	unsigned char *const buffer =
		step->fallback_bytes + (uint64_t)table * PACK_BLOCK_BYTES;
	for (;;) {
		const uint32_t pending = atomic_load(&step->pending[table]);
		if (pending == 0)
			return true;
		const uint32_t fallback = fallback_take(step);
		if (fallback != NONE) {
			fallback_serve(step, fallback, buffer, read);
			continue;
		}
		const uint64_t now = monotonic_nanoseconds();
		if (now - started >= limit)
			return false;
		if (now - started < spin) {
			cpu_relax();
			continue;
		}
		const struct timespec slice = {
			.tv_sec = 0, .tv_nsec = (long)WAIT_SLICE_NANOSECONDS
		};
		(void)syscall(SYS_futex,
			      (void *)(uintptr_t)&step->pending[table],
			      FUTEX_WAIT_PRIVATE, pending, &slice, NULL, 0);
	}
}

static void step_free(struct engram_step *step)
{
	free(step->blocks);
	free(step->miss_order);
	free(step->next_reference);
	free(step->reference_row);
	free((void *)step->reference_output);
	free(step->dedupe_keys);
	free(step->dedupe_values);
	free(step->dedupe_stamps);
	free(step->fallback_bytes);
	free(step);
}

/* Waits for whatever it still has in flight first (blocks left to the waiters fail unread); a read the drive never
 * answers keeps the step allocated, and the caller its destinations (an I/O thread may still write into them). */
bool engram_step_destroy(struct engram_step *step)
{
	if (step == NULL)
		return true;
	for (uint32_t table = 0; table < step->steps->description.table_count;
	     table++)
		if ((atomic_load(&step->issued) >> table & 1u) != 0 &&
		    !table_drain(step, table, false)) {
			LOG_ERROR(
				"engram step: a read did not return; its records stay allocated");
			return false;
		}
	const uint32_t index =
		step->base_slot / step->steps->options.slots_per_step;
	step->steps->in_use[index] = false;
	step->steps->records[index] = NULL;
	(void)pthread_mutex_destroy(&step->fallback_lock);
	step_free(step);
	return true;
}

result_status engram_step_create(struct engram_steps *steps,
				 struct engram_step **step_out)
{
	if (steps == NULL || step_out == NULL)
		return STATUS_INVALID_ARGUMENT;
	uint32_t index = 0;
	while (index < steps->options.steps && steps->in_use[index])
		index++;
	if (index == steps->options.steps)
		return STATUS_BUSY_VALUE;
	struct engram_step *const step = calloc(1, sizeof(*step));
	if (step == NULL)
		return STATUS_OUT_OF_MEMORY;
	const uint32_t references = steps->references_most;
	step->steps = steps;
	step->base_slot = index * steps->options.slots_per_step;
	step->dedupe_capacity = 1;
	while (step->dedupe_capacity < (uint64_t)references * DEDUPE_FACTOR)
		step->dedupe_capacity <<= 1u;
	atomic_store(&step->fallback_head, NONE);
	step->blocks = calloc(references, sizeof(*step->blocks));
	step->miss_order = calloc(references, sizeof(*step->miss_order));
	step->next_reference =
		calloc(references, sizeof(*step->next_reference));
	step->reference_row = calloc(references, sizeof(*step->reference_row));
	step->reference_output = (unsigned char **)calloc(
		references, sizeof(*step->reference_output));
	step->dedupe_keys =
		calloc(step->dedupe_capacity, sizeof(*step->dedupe_keys));
	step->dedupe_values =
		calloc(step->dedupe_capacity, sizeof(*step->dedupe_values));
	step->dedupe_stamps =
		calloc(step->dedupe_capacity, sizeof(*step->dedupe_stamps));
	step->fallback_bytes =
		aligned_alloc(PACK_BLOCK_BYTES,
			      (size_t)ENGRAM_MAXIMUM_TABLES * PACK_BLOCK_BYTES);
	if (step->blocks == NULL || step->miss_order == NULL ||
	    step->next_reference == NULL || step->reference_row == NULL ||
	    step->reference_output == NULL || step->dedupe_keys == NULL ||
	    step->dedupe_values == NULL || step->dedupe_stamps == NULL ||
	    step->fallback_bytes == NULL ||
	    pthread_mutex_init(&step->fallback_lock, NULL) != 0) {
		step_free(step);
		return STATUS_OUT_OF_MEMORY;
	}
	for (uint32_t block = 0; block < references; block++) {
		step->blocks[block].step = step;
		step->blocks[block].read.tag = block;
	}
	steps->in_use[index] = true;
	steps->records[index] = step;
	*step_out = step;
	return STATUS_OK;
}

result_status engram_step_wait(struct engram_step *step, uint32_t table)
{
	if (step == NULL || table >= step->steps->description.table_count)
		return STATUS_INVALID_ARGUMENT;
	if ((atomic_load_explicit(&step->issued, memory_order_acquire) >>
		     table &
	     1u) == 0)
		return STATUS_OK;
	if (!table_drain(step, table, true))
		return STATUS_TIMED_OUT;
	(void)atomic_fetch_and_explicit(&step->issued, ~(1u << table),
					memory_order_release);
	return (result_status)atomic_load(&step->failure);
}

static uint64_t hash_slot(uint64_t key, uint64_t capacity)
{
	return (key * HASH_MULTIPLIER) >>
	       (WORD_BITS - (uint32_t)__builtin_ctzll(capacity));
}

static uint64_t backoff_doubled(const struct engram_steps *steps,
				uint64_t backoff)
{
	const uint64_t most = steps->backoff_first_ns << BACKOFF_DOUBLINGS_MOST;
	return backoff >= most / BACKOFF_GROWTH ? most :
						 backoff * BACKOFF_GROWTH;
}

/* A copy's probe: one block of table 0 (another each time) read on the copy into that copy's own probe slot. */
static void probe_start(struct engram_steps *steps, uint32_t copy)
{
	struct planned_block *const probe = &steps->probes[copy];
	const uint32_t bit = 1u << copy;
	if ((atomic_load(&steps->probes_flying) & bit) != 0 ||
	    storage_flight_slot_quarantined(steps->flight, probe->read.slot))
		return;
	if (FAULT_POINT(engram_step_probe_lost)) {
		(void)atomic_fetch_or(
			&steps->probes_flying,
			bit); /* as a probe its drive never answers */
		return;
	}
	const uint64_t blocks = engram_disk_block_count(steps->disk, 0);
	probe->table = 0;
	probe->block =
		blocks != 0 ? steps->probe_count * PROBE_BLOCK_STRIDE % blocks :
			      0u;
	steps->probe_count++;
	probe->read.offset =
		engram_disk_block_offset(steps->disk, 0, probe->block);
	probe->read.copy = copy;
	(void)atomic_fetch_or(&steps->probes_flying, bit);
	struct storage_flight_read *read = &probe->read;
	if (storage_flight_submit(steps->flight, &read, 1, NULL) != STATUS_OK)
		(void)atomic_fetch_and(&steps->probes_flying, ~bit);
}

/* One copy's phase, with recovery_lock held. Retired: once the reads given to it before are out (they are no evidence),
 * it is noted, and its rings are asked for after the backoff, or at once when it answers a verified read (a waiter's)
 * meanwhile; after a refusal only the backoff brings the next ask (the thread had no ring to give, whatever the drive
 * answers). From the ask it is on probation: no step's read
 * goes to it, and with its fresh rings up the probe asks the drive (after the backoff when a probe failed). The
 * probe's answer puts it back. Each failed attempt (a refusal, a failed probe, a retirement on probation) doubles the
 * backoff. */
static void copy_recover(struct engram_steps *steps, uint32_t copy,
			 uint64_t now)
{
	struct copy_recovery *const own = &steps->recovery[copy];
	const uint32_t bit = 1u << copy;
	if (storage_flight_copy_retired(steps->flight, copy)) {
		if (own->phase != RECOVERY_RETIRED) {
			if (storage_flight_reads(steps->flight, copy) != 0)
				return;
			/* damping: stalled again on probation, or soon after it came back, it waits twice as long */
			const bool again =
				own->phase == RECOVERY_PROBATION ||
				(own->live_since_ns != 0 &&
				 now - own->live_since_ns <
					 steps->backoff_first_ns *
						 ENGRAM_STEP_STABLE_BACKOFFS);
			own->backoff_ns =
				again ? backoff_doubled(steps,
							own->backoff_ns) :
					steps->backoff_first_ns;
			own->phase = RECOVERY_RETIRED;
			own->refused = false;
			own->noted_verified =
				engram_disk_copy_verified(steps->disk, copy);
			own->due_ns = now + own->backoff_ns;
			(void)atomic_fetch_or(&steps->recovering, bit);
			return;
		}
		if ((own->refused ||
		     engram_disk_copy_verified(steps->disk, copy) ==
			     own->noted_verified) &&
		    now < own->due_ns)
			return;
		/* on probation before any ring is asked for: no step's read goes to it once it is live */
		if ((atomic_fetch_or(&steps->probation, bit) & bit) == 0)
			(void)atomic_fetch_add(&steps->probations, 1u);
		if (!storage_flight_reinstate(steps->flight, copy)) {
			own->refused = true;
			own->noted_verified =
				engram_disk_copy_verified(steps->disk, copy);
			own->backoff_ns =
				backoff_doubled(steps, own->backoff_ns);
			own->due_ns = now + own->backoff_ns;
		}
		return;
	}
	if (own->phase == RECOVERY_RETIRED) {
		own->phase = RECOVERY_PROBATION;
		own->due_ns = now;
		atomic_store(&steps->probe_outcome[copy], PROBE_NONE);
	}
	if (own->phase != RECOVERY_PROBATION ||
	    (atomic_load(&steps->probes_flying) & bit) != 0)
		return;
	const uint32_t outcome =
		atomic_exchange(&steps->probe_outcome[copy], PROBE_NONE);
	if (outcome == PROBE_ANSWERED) {
		own->phase = RECOVERY_LIVE;
		own->live_since_ns =
			now; /* the backoff is kept until it stays live (damping) */
		(void)atomic_fetch_and(&steps->probation, ~bit);
		(void)atomic_fetch_and(&steps->recovering, ~bit);
		return;
	}
	if (outcome == PROBE_FAILED) {
		own->backoff_ns = backoff_doubled(steps, own->backoff_ns);
		own->due_ns = now + own->backoff_ns;
	}
	if (now >= own->due_ns)
		probe_start(steps, copy);
}

/* At an issue, by one issuer at a time (another passes): every copy not plainly live takes its next phase. */
static void copies_recover(struct engram_steps *steps)
{
	bool attention = atomic_load(&steps->recovering) != 0;
	for (uint32_t copy = 0; !attention && copy < steps->copy_count; copy++)
		attention = storage_flight_copy_retired(steps->flight, copy);
	if (!attention || pthread_mutex_trylock(&steps->recovery_lock) != 0)
		return;
	const uint64_t now = monotonic_nanoseconds();
	for (uint32_t copy = 0; copy < steps->copy_count; copy++)
		copy_recover(steps, copy, now);
	(void)pthread_mutex_unlock(&steps->recovery_lock);
}

/* The first misses: each usable slot of the step is a lane and takes its first miss (a miss the flight refuses joins
 * the chain, and the lane takes the next). A step with no usable slot leaves every miss to the waiters. */
static void misses_start(struct engram_steps *steps, struct engram_step *step,
			 struct fallback_chain *chain)
{
	uint32_t usable = 0;
	for (uint32_t offset = 0;
	     offset < steps->options.slots_per_step &&
	     atomic_load(&step->next_miss) < step->miss_count;
	     offset++) {
		const uint32_t slot = step->base_slot + offset;
		if (storage_flight_slot_quarantined(steps->flight, slot))
			continue;
		usable++;
		lane_next(steps, step, slot, chain);
	}
	if (usable != 0)
		return;
	for (uint32_t next = atomic_fetch_add(&step->next_miss, 1);
	     next < step->miss_count;
	     next = atomic_fetch_add(&step->next_miss, 1))
		chain_add(step, chain, step->miss_order[next], false);
}

result_status
engram_step_issue(struct engram_step *step, const uint64_t *row_ids,
		  uint32_t row_count,
		  unsigned char *const destinations[ENGRAM_MAXIMUM_TABLES])
{
	if (step == NULL || row_ids == NULL || destinations == NULL)
		return STATUS_INVALID_ARGUMENT;
	struct engram_steps *const steps = step->steps;
	const struct engram_description *const description =
		&steps->description;
	const uint32_t tables = description->table_count,
		       per_token = description->rows_per_token;
	if (row_count > steps->options.rows_most)
		return STATUS_INVALID_ARGUMENT;
	for (uint32_t table = 0; table < tables; table++)
		if ((atomic_load(&step->issued) >> table & 1u) != 0) {
			/* an earlier use nobody waited for */
			if (!table_drain(step, table, true))
				return STATUS_TIMED_OUT;
			(void)atomic_fetch_and(&step->issued, ~(1u << table));
		}
	for (uint64_t reference = 0;
	     reference < (uint64_t)row_count * tables * per_token;
	     reference++) {
		const uint32_t table =
			(uint32_t)(reference / per_token % tables);
		if (destinations[table] != NULL &&
		    row_ids[reference] >= description->row_count[table])
			return STATUS_INVALID_ARGUMENT;
	}
	copies_recover(steps);
	atomic_store(&step->failure, STATUS_OK);
	step->stamp++;
	step->block_count = 0;
	step->miss_count = 0;
	atomic_store(&step->fallback_head, NONE);
	uint32_t references = 0;
	/* the first pass: every reference's unique block, the references of a block chained */
	for (uint32_t table = 0; table < tables; table++) {
		if (destinations[table] == NULL)
			continue;
		for (uint32_t row = 0; row < row_count; row++)
			for (uint32_t index = 0; index < per_token; index++) {
				const uint64_t id =
					row_ids[((uint64_t)row * tables + table) *
							per_token +
						index];
				const uint64_t block =
					id / ENGRAM_ROWS_PER_BLOCK;
				const uint64_t key =
					((uint64_t)table << KEY_TABLE_SHIFT |
					 block) +
					1u;
				uint64_t slot =
					hash_slot(key, step->dedupe_capacity);
				while (step->dedupe_stamps[slot] ==
					       step->stamp &&
				       step->dedupe_keys[slot] != key)
					slot = (slot + 1u) &
					       (step->dedupe_capacity - 1u);
				if (step->dedupe_stamps[slot] != step->stamp) {
					step->dedupe_stamps[slot] = step->stamp;
					step->dedupe_keys[slot] = key;
					step->dedupe_values[slot] =
						step->block_count;
					struct planned_block *const fresh =
						&step->blocks
							 [step->block_count++];
					fresh->table = table;
					fresh->block = block;
					fresh->first_reference = NONE;
					fresh->references = 0;
					fresh->tried = 0;
					fresh->carries_lane = false;
				}
				struct planned_block *const unique =
					&step->blocks[step->dedupe_values[slot]];
				step->reference_row[references] =
					(uint32_t)(id % ENGRAM_ROWS_PER_BLOCK *
						   ENGRAM_ROW_BYTES);
				step->reference_output[references] =
					destinations[table] +
					((uint64_t)row * per_token + index) *
						ENGRAM_ROW_BYTES;
				step->next_reference[references] =
					unique->first_reference;
				unique->first_reference = references++;
				unique->references++;
			}
	}
	/* the second pass: the RAM prefix and the cache now, every other block a miss */
	for (uint32_t index = 0; index < step->block_count; index++) {
		struct planned_block *const unique = &step->blocks[index];
		const unsigned char *const ram = engram_disk_ram_block(
			steps->disk, unique->table, unique->block);
		if (ram != NULL) {
			for (uint32_t reference = unique->first_reference;
			     reference != NONE;
			     reference = step->next_reference[reference])
				memcpy(step->reference_output[reference],
				       ram + step->reference_row[reference],
				       ENGRAM_ROW_BYTES);
			continue;
		}
		if (unique->references <= PROBE_ROWS_MOST) {
			uint32_t rows[PROBE_ROWS_MOST];
			unsigned char *outputs[PROBE_ROWS_MOST];
			uint32_t count = 0, reference = unique->first_reference;
			do { /* a unique block has its first reference */
				rows[count] = step->reference_row[reference];
				outputs[count++] =
					step->reference_output[reference];
				reference = step->next_reference[reference];
			} while (reference != NONE);
			if (engram_disk_cache_rows(steps->disk, unique->table,
						   unique->block, rows, outputs,
						   count))
				continue;
		}
		step->miss_order[step->miss_count++] = index;
		atomic_fetch_add(&step->pending[unique->table], 1);
	}
	uint32_t issued = 0;
	for (uint32_t table = 0; table < tables; table++)
		issued |= destinations[table] != NULL ? 1u << table : 0u;
	atomic_store_explicit(&step->issued, issued, memory_order_release);
	/* the first misses now, one a usable slot; each completion submits the next */
	atomic_store(&step->next_miss, 0);
	struct fallback_chain chain = { NONE, NONE };
	misses_start(steps, step, &chain);
	chain_hand_over(step, &chain);
	return STATUS_OK;
}
