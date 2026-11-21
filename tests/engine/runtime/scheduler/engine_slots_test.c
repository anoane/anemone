/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* engine_slots_test: an engine of more session slots than a step's rows and a word's bits (unbounded sessions): 32 and
 * 64 slots on the serving fixture's function model, without and with a pipeline, without and with the lookup drafter,
 * greedy and sampled.
 *   - Identity: every session's events are those of its solo run (an engine of one slot), submitted all at once, and
 *     submitted in a shuffled order with pauses while every seventh session is cancelled after its first token (a
 *     cancelled one: its solo run's tokens up to the cancel).
 *   - Fairness: without speculation, the most steps between two drawing turns of a session stay within the round
 *     robin's bound: ceil((slots - 1) / rows) + 1 on thin steps, slots + depth in a pipeline (one drawing owner a
 *     flight). The measured gaps are printed.
 *   - The engine and its statistics allocate nothing after create (counted where no sanitizer owns the allocator). */
#include <errno.h>
#include <malloc.h>
#include <poll.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include "engine/runtime/scheduler/scheduler_stats.h"
#include "engine/runtime/scheduler/front_end_seam.h"
#include "engine/runtime/scheduler/serving_cursor.h"
#include "support/serving_fixture.h"

#define CONTEXT_TOKENS 256u
#define PROMPT_LEAST 3u
#define PROMPT_SPREAD \
	13u /* a slot's prompt: PROMPT_LEAST + slot % PROMPT_SPREAD tokens */
#define PROMPT_STRIDE (PROMPT_LEAST + PROMPT_SPREAD)
#define GENERATED_LIMIT 24u
#define MOST_EVENTS (GENERATED_LIMIT + 1u)
#define PIPELINE_DEPTH 4u
#define DRAFT_POSITIONS 3u
#define SEED_STRIDE 7919u
#define SAMPLED_TEMPERATURE 0.7f
#define PROMPT_MULTIPLIER 31u
#define PROMPT_STEP 7u
#define BYTE_TOKENS 256u
#define POLL_MILLISECONDS 10000
#define CANCEL_STRIDE \
	7u /* every seventh slot cancelled in the cancelling runs */
#define CANCEL_OFFSET 3u
#define ARRIVAL_STRIDE \
	13u /* coprime with 32 and 64: the staggered submission order */
#define ARRIVAL_GROUP 8u /* staggered: a pause after every eighth submission */
#define ARRIVAL_PAUSE_NS 200000L
#define TRACE_CAPACITY 65536u /* drawing-row entries of the fairness runs */
#define MOST_SLOTS TEST_SERVING_SLOT_ROOM
#define WORD_BITS 32u /* a 32-bit mask's slots, which the test's pass */
#define SAMPLED_EVERY \
	2u /* every second session samples, the others are greedy */
#define CONTEXT_TEXT_BYTES 96u
_Static_assert(
	MOST_SLOTS <= SERVING_SLOTS_MOST && MOST_SLOTS > WORD_BITS,
	"the test's slots pass a 32-bit word and stay within SERVING_SLOTS_MOST");

#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
#define ALLOCATIONS_COUNTED false /* the sanitizer owns malloc */
#else
#define ALLOCATIONS_COUNTED true
/* The test's own allocator, replacing glibc's as its manual's "Replacing malloc" says, with the whole set (malloc,
 * free, calloc, realloc, aligned_alloc, malloc_usable_size, memalign, posix_memalign, pvalloc, valloc): every
 * allocation of every thread, glibc's and the loader's included, comes here, and each call of an allocating function
 * is counted. A block is cut from one anonymous mapping by an atomic bump and never reused (free keeps it); the size it
 * was asked for is the word before it. calloc's blocks are the mapping's zeros; every other block is filled as it is
 * handed out with a byte other than zero that changes from block to block (FRESH_FILL_VALUES, by the call's ordinal),
 * so that a read before a write gives other bytes in a solo run than in a run of many slots and the comparison of
 * their events fails. A use after free reads a block never reused here: the debug build (ASan, its own allocator)
 * catches it. */
#define ARENA_BYTES \
	(UINT64_C(4) << 30) /* reserved, not committed (MAP_NORESERVE) */
#define BLOCK_ALIGNMENT _Alignof(max_align_t)
/* a fresh block's byte: 1 + its call's ordinal % FRESH_FILL_VALUES, never 0 */
#define FRESH_FILL_VALUES 255u
static _Atomic(unsigned char *) arena;
static atomic_size_t arena_used;
static _Atomic uint64_t allocations;

/* Counts a call of an allocating function; its ordinal. */
static uint64_t allocation_counted(void)
{
	return atomic_fetch_add_explicit(&allocations, 1u,
					 memory_order_relaxed);
}

static bool is_power_of_two(size_t value)
{
	return value != 0 && (value & (value - 1u)) == 0;
}

/* A block of bytes at alignment (a power of two), or NULL with errno ENOMEM. */
static void *arena_block(size_t alignment, size_t bytes)
{
	unsigned char *mapping = atomic_load(&arena);
	/* the first block maps the arena; a thread that loses the race unmaps its own */
	if (mapping == NULL) {
		void *const made = mmap(
			NULL, ARENA_BYTES, PROT_READ | PROT_WRITE,
			MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
		if (made == MAP_FAILED)
			return NULL;
		if (atomic_compare_exchange_strong(&arena, &mapping, made))
			mapping = made;
		else
			(void)munmap(made, ARENA_BYTES);
	}
	/* step: room for the size word and the alignment; every span is a multiple of BLOCK_ALIGNMENT */
	const size_t step = alignment > BLOCK_ALIGNMENT ? alignment :
							  BLOCK_ALIGNMENT;
	if (bytes > ARENA_BYTES || step > ARENA_BYTES) {
		errno = ENOMEM;
		return NULL;
	}
	const size_t span = step + (bytes + BLOCK_ALIGNMENT - 1u) /
					   BLOCK_ALIGNMENT * BLOCK_ALIGNMENT;
	/* arena_used never passes ARENA_BYTES; an exchange lost is a block another thread took, so the loop ends within
	 * ARENA_BYTES / BLOCK_ALIGNMENT rounds */
	size_t first = atomic_load_explicit(&arena_used, memory_order_relaxed);
	do {
		if (span > ARENA_BYTES - first) {
			errno = ENOMEM;
			return NULL;
		}
	} while (!atomic_compare_exchange_strong(&arena_used, &first,
						 first + span));
	/* start: at BLOCK_ALIGNMENT (the mapping's page, spans of whole BLOCK_ALIGNMENTs); the block: at alignment,
	 * BLOCK_ALIGNMENT to step bytes past start, its size word between */
	unsigned char *const start = mapping + first;
	unsigned char *const block = start + (step - (uintptr_t)start % step);
	memcpy(block - sizeof(bytes), &bytes, sizeof(bytes));
	return block;
}

/* An arena block for the call of that ordinal, filled with its byte (a fill other than zero: no memset of it is
 * folded into a call of calloc). */
static void *fresh_block(uint64_t ordinal, size_t alignment, size_t bytes)
{
	void *const block = arena_block(alignment, bytes);
	if (block != NULL)
		memset(block, (int)(ordinal % FRESH_FILL_VALUES + 1u), bytes);
	return block;
}

void *malloc(size_t bytes)
{
	return fresh_block(allocation_counted(), BLOCK_ALIGNMENT, bytes);
}

void free(void *memory)
{
	(void)memory; /* a block is never reused */
}

void *calloc(size_t count, size_t bytes)
{
	(void)allocation_counted();
	if (bytes != 0 && count > SIZE_MAX / bytes) {
		errno = ENOMEM;
		return NULL;
	}
	return arena_block(BLOCK_ALIGNMENT, count * bytes); /* never written */
}

size_t malloc_usable_size(void *memory)
{
	size_t bytes = 0;
	if (memory != NULL)
		memcpy(&bytes, (unsigned char *)memory - sizeof(bytes),
		       sizeof(bytes));
	return bytes;
}

void *realloc(void *memory, size_t bytes)
{
	const uint64_t ordinal = allocation_counted();
	if (memory != NULL && bytes == 0)
		return NULL; /* as glibc's: the block freed */
	void *const moved = fresh_block(ordinal, BLOCK_ALIGNMENT, bytes);
	const size_t held = malloc_usable_size(memory);
	if (moved != NULL && memory != NULL)
		memcpy(moved, memory, held < bytes ? held : bytes);
	return moved;
}

void *aligned_alloc(size_t alignment, size_t bytes)
{
	const uint64_t ordinal = allocation_counted();
	if (!is_power_of_two(alignment)) {
		errno = EINVAL;
		return NULL;
	}
	return fresh_block(ordinal, alignment, bytes);
}

int posix_memalign(void **memory_out, size_t alignment, size_t bytes)
{
	const uint64_t ordinal = allocation_counted();
	if (alignment % sizeof(void *) != 0 || !is_power_of_two(alignment))
		return EINVAL;
	void *const block = fresh_block(ordinal, alignment, bytes);
	if (block == NULL)
		return ENOMEM;
	*memory_out = block;
	return 0;
}

void *memalign(size_t alignment, size_t bytes)
{
	const uint64_t ordinal = allocation_counted();
	/* as glibc's: the alignment rounded up to a power of two */
	size_t power = BLOCK_ALIGNMENT;
	while (power < alignment && power <= SIZE_MAX >> 1u)
		power <<= 1u;
	if (power < alignment) {
		errno = EINVAL;
		return NULL;
	}
	return fresh_block(ordinal, power, bytes);
}

void *valloc(size_t bytes)
{
	return fresh_block(allocation_counted(), (size_t)sysconf(_SC_PAGESIZE),
			   bytes);
}

void *pvalloc(size_t bytes)
{
	const uint64_t ordinal = allocation_counted();
	const size_t page = (size_t)sysconf(_SC_PAGESIZE);
	if (bytes > SIZE_MAX - page) {
		errno = ENOMEM;
		return NULL;
	}
	return fresh_block(ordinal, page, (bytes + page - 1u) / page * page);
}
#endif

static uint32_t failures;
static const char *context_text = "";
static uint32_t prompt[MOST_SLOTS * PROMPT_STRIDE];
static const uint32_t script[] = { TEST_SERVING_END_TOKEN };
static const uint32_t *const scripts[] = { script };

static void expect(bool condition, const char *what)
{
	if (!condition) {
		(void)printf("FAILED: %s: %s\n", context_text, what);
		failures++;
	}
}

static uint64_t allocations_now(void)
{
#if ALLOCATIONS_COUNTED
	return atomic_load(&allocations);
#else
	return 0;
#endif
}

/* slot's command, on session slot on (its own, or 0 in a solo run): greedy on even slots, sampled on odd */
static struct generation_command_record slot_command(uint32_t slot, uint32_t on)
{
	return (struct generation_command_record){
		.new_token_ids = prompt + (size_t)slot * PROMPT_STRIDE,
		.sampler_seed = (uint64_t)slot * SEED_STRIDE,
		.session_slot = on,
		.new_token_count = PROMPT_LEAST + slot % PROMPT_SPREAD,
		.generated_token_limit = GENERATED_LIMIT,
		.sampler_temperature =
			slot % SAMPLED_EVERY == 0 ? 0.0f : SAMPLED_TEMPERATURE,
		.sampler_top_p = 1.0f,
		.request_tag = slot + 1u
	};
}

/* How a run submits and cancels: solo (each command alone on slot 0), staggered (in a shuffled order, pausing
 * between), cancelling (every CANCEL_STRIDE-th slot once its first token comes). */
struct run_plan {
	bool solo, staggered, cancelling;
};

static bool is_cancelled_slot(uint32_t slot)
{
	return slot % CANCEL_STRIDE == CANCEL_OFFSET;
}

/* One event of any slot into its slot's list (by request tag), cancelling as planned; false when none came within
 * the poll's time. */
static bool event_filed(struct inference_engine *engine, int descriptor,
			const struct run_plan *plan,
			struct generation_event (*events)[MOST_EVENTS],
			uint32_t *counts, uint32_t *finals)
{
	struct pollfd poll_entry = { descriptor, POLLIN, 0 };
	if (poll(&poll_entry, 1, POLL_MILLISECONDS) != 1)
		return false;
	struct generation_event event = { 0 };
	bool is_available = false;
	if (engine_next_event(engine, &event, &is_available) != STATUS_OK)
		return false;
	const uint32_t slot = event.request_tag - 1u;
	if (is_available && slot < MOST_SLOTS && counts[slot] < MOST_EVENTS) {
		events[slot][counts[slot]++] = event;
		*finals += event.event_kind != GENERATION_EVENT_TOKEN ? 1u : 0u;
		if (plan->cancelling && is_cancelled_slot(slot) &&
		    counts[slot] == 1u &&
		    event.event_kind == GENERATION_EVENT_TOKEN)
			expect(engine_cancel(engine, slot) == STATUS_OK,
			       "a running session cancelled");
	}
	return true;
}

/* The commands of slots first..first + count - 1 on an engine as last set, as plan says, their events until every
 * final; whether the engine allocated nothing between its create and the last final. */
static bool commands_run(uint32_t first, uint32_t count,
			 const struct run_plan *plan,
			 struct generation_event (*events)[MOST_EVENTS],
			 uint32_t *counts)
{
	struct inference_engine *engine = NULL;
	int descriptor = -1;
	expect(test_serving_open(CONTEXT_TOKENS, scripts, 1, &engine) ==
			       STATUS_OK &&
		       engine_event_descriptor(engine, &descriptor) ==
			       STATUS_OK,
	       "an engine of the function model");
	if (engine == NULL)
		return false;
	struct engine_stats stats = { 0 };
	const uint32_t slots = plan->solo ? 1u : count;
	const bool has_tables = engine_stats_tables(&stats, slots) == STATUS_OK;
	expect(has_tables, "statistics tables for the engine's slots");
	const struct timespec pause = { 0, ARRIVAL_PAUSE_NS };
	const uint64_t created = allocations_now();
	uint32_t finals = 0;
	for (uint32_t index = 0; index < count; index++) {
		/* staggered: a stride coprime with the count visits every slot once */
		const uint32_t slot =
			first + (plan->staggered ?
					 index * ARRIVAL_STRIDE % count :
					 index);
		counts[slot] = 0;
		const struct generation_command_record command =
			slot_command(slot, plan->solo ? 0u : slot);
		expect(engine_submit(engine, &command) == STATUS_OK,
		       "every slot's command admitted");
		if (plan->staggered && index % ARRIVAL_GROUP == 0)
			(void)nanosleep(&pause, NULL);
	}
	bool polled = true;
	while (polled && finals < count) {
		polled = event_filed(engine, descriptor, plan, events, counts,
				     &finals);
		expect(!has_tables || engine_stats_counters(engine, &stats) ==
					      STATUS_OK,
		       "statistics copied while the sessions run");
	}
	expect(polled, "every event within the poll's time");
	const bool quiet = allocations_now() == created;
	expect(!has_tables ||
		       (engine_stats_counters(engine, &stats) == STATUS_OK &&
			stats.slots_total == slots && stats.slots_busy == 0),
	       "the statistics of every slot, all idle at the end");
	engine_stats_tables_free(&stats);
	test_serving_close();
	return quiet;
}

static bool events_equal(const struct generation_event *events, uint32_t count,
			 const struct generation_event *reference,
			 uint32_t reference_count)
{
	return count != 0 && count == reference_count &&
	       memcmp(events, reference, count * sizeof(*events)) == 0;
}

/* A cancelled session's events: its solo run's tokens, as many as it drew, then a cancelled final (or its solo run
 * whole, if it ended before the cancel took). */
static bool cancelled_events(const struct generation_event *events,
			     uint32_t count,
			     const struct generation_event *reference,
			     uint32_t reference_count)
{
	if (events_equal(events, count, reference, reference_count))
		return true;
	return count != 0 && count <= reference_count &&
	       events[count - 1u].event_kind == GENERATION_EVENT_CANCELLED &&
	       memcmp(events, reference, (count - 1u) * sizeof(*events)) == 0;
}

/* The most steps from one drawing turn of a session to its next, over the slots (the fixture's trace). */
static uint32_t widest_turn_gap(const uint32_t *trace, uint32_t count,
				uint32_t slots)
{
	static uint32_t last[MOST_SLOTS];
	uint32_t step = 0, widest = 0;
	for (uint32_t slot = 0; slot < slots; slot++)
		last[slot] = UINT32_MAX;
	for (uint32_t entry = 0; entry < count; entry++) {
		const uint32_t slot = trace[entry];
		if (slot == STEP_NO_SESSION) {
			step++;
		} else if (slot < slots) {
			if (last[slot] != UINT32_MAX &&
			    step - last[slot] > widest)
				widest = step - last[slot];
			last[slot] = step;
		}
	}
	return widest;
}

int main(void)
{
	static struct generation_event solo[MOST_SLOTS][MOST_EVENTS],
		events[MOST_SLOTS][MOST_EVENTS];
	static uint32_t solo_counts[MOST_SLOTS], counts[MOST_SLOTS],
		trace[TRACE_CAPACITY];
	static const uint32_t slot_counts[] = { 32u, MOST_SLOTS };
	static const uint32_t depths[] = { 1u, PIPELINE_DEPTH };
	static const uint32_t drafts[] = { 0u, DRAFT_POSITIONS };
	static const struct run_plan solo_plan = { .solo = true },
				     together = { 0 },
				     arriving = { .staggered = true,
						  .cancelling = true };
	for (uint32_t index = 0; index < MOST_SLOTS * PROMPT_STRIDE; index++)
		prompt[index] = (index * PROMPT_MULTIPLIER +
				 index / PROMPT_STRIDE * PROMPT_STEP) %
				BYTE_TOKENS;
	test_serving_set_function_model(true, 0);
	context_text = "solo runs";
	test_serving_set_engine(1, 1, TEST_SERVING_MOST_ROWS, 0);
	for (uint32_t slot = 0; slot < MOST_SLOTS; slot++)
		(void)commands_run(slot, 1, &solo_plan, solo, solo_counts);
	for (uint32_t size = 0;
	     size < sizeof(slot_counts) / sizeof(*slot_counts); size++) {
		const uint32_t slots = slot_counts[size];
		for (uint32_t depth = 0;
		     depth < sizeof(depths) / sizeof(*depths); depth++) {
			for (uint32_t draft = 0;
			     draft < sizeof(drafts) / sizeof(*drafts);
			     draft++) {
				static char text[CONTEXT_TEXT_BYTES];
				(void)snprintf(text, sizeof(text),
					       "%u slots, depth %u, draft %u",
					       slots, depths[depth],
					       drafts[draft]);
				context_text = text;
				test_serving_set_engine(slots, depths[depth],
							TEST_SERVING_MOST_ROWS,
							drafts[draft]);
				/* all at once, traced without speculation (one drawing row a session a step) */
				test_serving_set_trace(
					drafts[draft] == 0 ? trace : NULL,
					TRACE_CAPACITY);
				bool quiet = commands_run(0, slots, &together,
							  events, counts);
				const uint32_t traced =
					test_serving_trace_count();
				test_serving_set_trace(NULL, 0);
				for (uint32_t slot = 0; slot < slots; slot++)
					expect(events_equal(events[slot],
							    counts[slot],
							    solo[slot],
							    solo_counts[slot]),
					       "every session's events its solo run's");
				/* round robin: a thin step takes the next TEST_SERVING_MOST_ROWS ready
				 * owners, a pipeline's flight one drawing owner, so a ready session waits
				 * at most for every other. The rule (scheduler.c flight_taken_off,
				 * stepping_thread): the cursor moves only past owners that take a row; a
				 * landing takes no turn; an owner the step's make-up passed over (a
				 * drawing owner past DRAWING_ROWS) keeps it */
				const uint32_t bound =
					depths[depth] == 1u ?
						(slots - 1u +
						 TEST_SERVING_MOST_ROWS -
						 1u) / TEST_SERVING_MOST_ROWS +
							1u :
						slots + depths[depth];
				const uint32_t widest =
					widest_turn_gap(trace, traced, slots);
				expect(drafts[draft] != 0 ||
					       (traced < TRACE_CAPACITY &&
						widest != 0 && widest <= bound),
				       "every session's turns within the round-robin bound");
				if (drafts[draft] == 0)
					(void)printf(
						"engine_slots_test: %u slots, depth %u: widest turn gap"
						" %u steps (bound %u)\n",
						slots, depths[depth], widest,
						bound);
				/* staggered arrivals, every seventh session cancelled after its first token */
				quiet = commands_run(0, slots, &arriving,
						     events, counts) &&
					quiet;
				expect(!ALLOCATIONS_COUNTED || quiet,
				       "no allocation after create");
				for (uint32_t slot = 0; slot < slots; slot++)
					expect(is_cancelled_slot(slot) ?
						       cancelled_events(
							       events[slot],
							       counts[slot],
							       solo[slot],
							       solo_counts[slot]) :
						       events_equal(
							       events[slot],
							       counts[slot],
							       solo[slot],
							       solo_counts[slot]),
					       "staggered and cancelled: every other session's events its solo run's");
			}
		}
	}
	(void)printf("engine_slots_test: %u failures (allocations %s)\n",
		     failures, ALLOCATIONS_COUNTED ? "counted" : "not counted");
	return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
