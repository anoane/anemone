/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * step_pipeline.c: step.h's pipeline: a worker thread a card, each taking the next work of its card that is ready, a
 * step's visit of the card at a time.
 *
 * Visits and tasks. A visit is a card's segments between other cards' (model_seam.h), taken from the first program,
 * whose segments and hand-offs every program of the driver shares as far as it reaches (step_validation.c); the
 * hand-offs a card takes in between two of its visits go up before the second, and a closing visit of no segment
 * behind a card's last takes those behind it. A step's tasks on a card are the visits it reaches (a front step and a
 * full one stop short) and, behind them, the next visit when the step hands it something it does not defer: then only
 * that. A card runs a task whole: on the card the deferred hand-offs leave, the step's mirror syncs before the visit
 * they follow (below); the settings where it samples (the card's last task of a step with a head); the step area, the
 * n-gram rows and the views again unless its last task was of the same step (another step's visit may have run in
 * between: model_seam.h); the hand-offs in (each behind the mark of the run that sent it, or the bytes in the block
 * where the sender waited for its card past a slot's marks); its runs as step.c runs them, each behind it its
 * hand-offs down (a front step leaves the deferred ones), then its mark or its card's wait, and the hand-offs marked
 * sent; the result where it samples; and behind the card's last task of the step a wait for the card, so that the
 * step's slot may be taken again once every card's last task ended.
 *
 * Readiness. A card's next task of a step is ready once every older step in flight has run its task of the same visit
 * or has none, every older step has run the last visit of each lease from this one if it reached it, and every
 * hand-off it takes in is sent. A worker runs its card's ready task of the oldest step, or waits. A card with one visit
 * in the programs, which no deferred hand-off goes to, takes its first task of a step once the older steps ran theirs,
 * puts the step's areas up and waits for its hand-offs in turn: so its uploads go up before its inputs arrive, and a
 * contiguous layout runs as when a worker ran its card's part of each step in order.
 *
 * Progress. Order tasks by (step, place of their visit in the program, a closing visit last). A task depends only on
 * smaller ones: its hand-offs come from earlier visits of its step; the older steps' tasks of its visit and of its
 * leases' last visits are older. A task that started never waits for a task to start, but the early first task of a
 * card of one visit, which waits only for hand-offs of its step, sent by other cards' tasks whose dependencies on its
 * own card are older steps' tasks of its one visit: done before it started. So the smallest task not yet run can
 * always run, and every task does: no deadlock (tests/step_test.c holds a schedule a card's in-order parts deadlock
 * or serialise on). Device work only waits for marks set before the wait was queued: it always drains.
 *
 * Mirror syncs (step.h). Before its task of the visit the deferred hand-offs follow, the card they leave borrows the
 * cards they go to: their workers stand still at the top of their loop, which they reach within a task (they take no
 * task early); it moves the entries of the step's sync ranges (step_sessions.c) and gives them back. The tasks of the
 * step that read the mirrors follow that visit in the program, so they run behind.
 *
 * One mutex and one condition guard the slots and the tasks. Every wait is timed on the monotonic clock: a worker with
 * a task outstanding, a finish and a borrower fail the driver with STATUS_TIMED_OUT once nothing in the
 * pipeline moved for the driver's wait limit (a worker's device waits are bounded by it too). A slot of the ring holds
 * a step from its submit to its finish: its program, rows, n-gram row ids, staging block (the driver's layout), sync
 * ranges and tasks; it is taken again only after its step was finished, so every copy of its block was waited for.
 */
#include <inttypes.h>
#include <pthread.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <sys/prctl.h>
#include <time.h>

#include "anemone/hooks.h"
#include "step_driver.h"
#include "step_runs.h"

#define OK STATUS_OK
#define LEAST_DEPTH 2u /* steps a pipeline holds at least */
#define WORKER_TIMER_SLACK_NANOSECONDS 1000ul
#define NANOSECONDS_PER_SECOND 1000000000ull
#define NO_SEGMENT UINT32_MAX /* the segments of a card's closing visit */
#define NO_MARK UINT32_MAX
#define PREFIX_CARDS \
	4u /* the experimental four-CMP prefix and one native tail */
#define PREFIX_MASK ((1u << PREFIX_CARDS) - 1u)
#define OVERLAP_CARDS (PREFIX_CARDS + 1u)
#define OVERLAP_WORKERS 2u
#define HAND_OFF_TABLES \
	2u /* of the pipeline's: the hand-offs in, their marks */
#define THEN(call)        \
	(status != OK ?   \
		 status : \
		 (call)) /* the next call while every one before passed */

/* The only signals a worker takes, raised by its own faults (engine.c's rule for its thread). */
static const int fault_signals[] = { SIGSEGV, SIGBUS,  SIGFPE,
				     SIGILL,  SIGTRAP, SIGSYS };

/* A card's visit (see the top) of the first program; NO_SEGMENT for its closing visit. */
struct pipeline_visit {
	uint32_t first_segment, last_segment;
	uint32_t first_incoming,
		incoming_count; /* its hand-offs in, in the pipeline's incoming */
	uint32_t lease_end; /* the card's latest visit a lease from this one reaches; 0: none */
};

struct pipeline_slot {
	const struct execution_program *program;
	const uint32_t *run_graphs; /* NULL: eager */
	unsigned char *block; /* staging of the driver's layout */
	uint64_t *row_ids;
	uint8_t *sent; /* [hand-off]: 1 once its copies down are queued (marked) or have arrived in block */
	uint32_t *syncs; /* [sync][STEP_SYNC_FIELDS] (step_driver.h) */
	uint64_t sequence;
	uint32_t row_count, segment_count, hand_off_count, sync_count;
	uint32_t done_cards; /* a bit a card whose tasks of the step ended */
	uint32_t visit_count[BACKEND_MAXIMUM_CARDS]; /* the visits it reaches */
	uint32_t task_count[BACKEND_MAXIMUM_CARDS];
	uint32_t tasks_done[BACKEND_MAXIMUM_CARDS];
	bool sessions_changed, is_front;
	struct step_row_record *rows; /* the largest program's row count */
};

struct worker_start {
	struct step_driver *driver;
	uint32_t card;
};

struct step_pipeline {
	pthread_mutex_t
		lock; /* guards everything below but the workers and the tables of the visits */
	pthread_cond_t changed;
	pthread_t workers[BACKEND_MAXIMUM_CARDS];
	struct worker_start starts[BACKEND_MAXIMUM_CARDS];
	uint32_t depth, worker_count, every_card;
	uint32_t prefix_segments; /* nonzero only for disjoint prefix/tail workers */
	uint64_t submitted, finished; /* the last sequences */
	uint64_t moves; /* counts every change a waiter may wait for */
	result_status failure;
	bool is_ending, has_lock, has_condition;
	uint32_t slot_marks; /* marks a slot's runs may take on their card: 0 where the backend has none */
	uint32_t borrowed,
		parked; /* the cards a mirror sync borrows; those whose workers stand still */
	uint32_t early_cards; /* those that take their first task of a step before its hand-offs are sent */
	uint32_t mirror_visit; /* of the mirror source: the visit the syncs come before */
	uint32_t first_visit[BACKEND_MAXIMUM_CARDS],
		visit_count[BACKEND_MAXIMUM_CARDS]; /* a card's visits, its closing one behind them */
	uint64_t step_on_card
		[BACKEND_MAXIMUM_CARDS]; /* the step whose areas a card holds: its worker's alone */
	struct pipeline_visit *visits;
	uint32_t *incoming; /* hand-off indices, a visit's in their order */
	uint32_t *hand_off_marks; /* [hand-off]: of the run that sends it, its mark within a slot's */
	uint32_t *run_hand_offs; /* [segment]: the hand-offs behind earlier segments */
	unsigned char *staging; /* depth blocks */
	uint64_t *row_ids; /* depth slots' */
	uint8_t *sent;
	uint32_t *syncs;
	struct step_row_record *rows; /* depth slots' */
	struct pipeline_slot slots[STEP_PIPELINE_MOST];
	uint64_t live_busy_ns[BACKEND_MAXIMUM_CARDS];
	uint64_t live_steps[BACKEND_MAXIMUM_CARDS],
		live_visits[BACKEND_MAXIMUM_CARDS];
};

static uint64_t monotonic_nanoseconds(void)
{
	struct timespec now;
	return clock_gettime(CLOCK_MONOTONIC, &now) == 0 && now.tv_sec >= 0 ?
		       (uint64_t)now.tv_sec * NANOSECONDS_PER_SECOND +
			       (uint64_t)now.tv_nsec :
		       0;
}

/* With the lock held: waits for the pipeline to move; fails it with STATUS_TIMED_OUT once it has not moved for
 * the driver's wait limit while the waiter had work outstanding. */
static void progress_wait(struct step_driver *driver, uint64_t *seen,
			  uint64_t *since, bool has_work)
{
	struct step_pipeline *const pipeline = driver->pipeline;
	const uint64_t now = monotonic_nanoseconds(),
		       limit = driver->wait_limit_nanoseconds;
	*since = pipeline->moves != *seen || *since > now ? now : *since;
	*seen = pipeline->moves;
	if (has_work && now - *since >= limit) {
		pipeline->failure = pipeline->failure == OK ? STATUS_TIMED_OUT :
							      pipeline->failure;
		(void)pthread_cond_broadcast(&pipeline->changed);
		return;
	}
	const uint64_t from = has_work ? *since : now;
	const uint64_t until = from > UINT64_MAX - limit ? UINT64_MAX :
							   from + limit;
	const struct timespec deadline = {
		.tv_sec = (time_t)(until / NANOSECONDS_PER_SECOND),
		.tv_nsec = (long)(until % NANOSECONDS_PER_SECOND)
	};
	(void)pthread_cond_timedwait(&pipeline->changed, &pipeline->lock,
				     &deadline);
}

/* With the lock held: the pipeline moved; its waiters look again. */
static void moved(struct step_pipeline *pipeline)
{
	pipeline->moves++;
	(void)pthread_cond_broadcast(&pipeline->changed);
}

static const struct pipeline_visit *
visit_of(const struct step_pipeline *pipeline, uint32_t card, uint32_t task)
{
	return &pipeline->visits[pipeline->first_visit[card] + task];
}

/* Whether the step of the slot hands in hand-off index of the first program: one of its program's (a full program has
 * none of the rows only the head reads: step_validation.c), behind a segment it runs, not deferred. */
static bool is_handed(const struct step_driver *driver,
		      const struct pipeline_slot *slot, uint32_t index)
{
	return index < slot->hand_off_count &&
	       driver->programs[0]->hand_offs[index].after_segment <
		       slot->segment_count &&
	       (!slot->is_front || driver->deferred[index] == 0);
}

/* With the lock held: whether the card's next task (task) of the slot's step is ready (see the top). */
static bool task_is_ready(const struct step_driver *driver,
			  const struct pipeline_slot *slot, uint32_t card,
			  uint32_t task)
{
	const struct step_pipeline *const pipeline = driver->pipeline;
	const struct pipeline_visit *const visit =
		visit_of(pipeline, card, task);
	const uint32_t lease_end = visit->lease_end;
	bool is_ready = true;
	for (uint64_t sequence = pipeline->finished + 1u;
	     sequence < slot->sequence && is_ready; sequence++) {
		const struct pipeline_slot *const older =
			&pipeline->slots[sequence % pipeline->depth];
		is_ready = (older->tasks_done[card] > task ||
			    older->task_count[card] <= task) &&
			   (lease_end <= task ||
			    older->tasks_done[card] > lease_end ||
			    older->visit_count[card] <= lease_end);
	}
	const bool is_early = task == 0 &&
			      (pipeline->early_cards >> card & 1u) != 0;
	for (uint32_t index = 0;
	     index < visit->incoming_count && is_ready && !is_early; index++) {
		const uint32_t hand_off =
			pipeline->incoming[visit->first_incoming + index];
		is_ready = !is_handed(driver, slot, hand_off) ||
			   slot->sent[hand_off] != 0;
	}
	return is_ready;
}

/* With the lock held: the slot of the oldest step whose next task on the card is ready, or NULL; *has_work: a step in
 * flight has a task of the card left. */
static struct pipeline_slot *ready_slot(const struct step_driver *driver,
					uint32_t card, bool *has_work)
{
	struct step_pipeline *const pipeline = driver->pipeline;
	*has_work = false;
	for (uint64_t sequence = pipeline->finished + 1u;
	     sequence <= pipeline->submitted; sequence++) {
		struct pipeline_slot *const slot =
			&pipeline->slots[sequence % pipeline->depth];
		const uint32_t task = slot->tasks_done[card];
		*has_work = *has_work || task < slot->task_count[card];
		if (task < slot->task_count[card] &&
		    task_is_ready(driver, slot, card, task)) {
			return slot;
		}
	}
	return NULL;
}

/* Waits for hand-off index of the slot's step to be sent; false once the pipeline failed or ends. */
static bool sent_wait(struct step_driver *driver,
		      const struct pipeline_slot *slot, uint32_t index)
{
	struct step_pipeline *const pipeline = driver->pipeline;
	(void)pthread_mutex_lock(&pipeline->lock);
	uint64_t seen = pipeline->moves, since = monotonic_nanoseconds();
	while (slot->sent[index] == 0 && pipeline->failure == OK &&
	       !pipeline->is_ending) {
		progress_wait(driver, &seen, &since, true);
	}
	const bool is_sent = slot->sent[index] != 0;
	(void)pthread_mutex_unlock(&pipeline->lock);
	return is_sent;
}

/* The slot's mirror syncs (step.h), on the card the deferred hand-offs leave, with the cards they go to borrowed. */
MUST_CHECK static result_status mirrors_synced(struct step_driver *driver,
					       const struct pipeline_slot *slot)
{
	struct step_pipeline *const pipeline = driver->pipeline;
	const uint32_t cards = driver->mirror_cards;
	(void)pthread_mutex_lock(&pipeline->lock);
	pipeline->borrowed = cards;
	moved(pipeline);
	uint64_t seen = pipeline->moves, since = monotonic_nanoseconds();
	while ((pipeline->parked & cards) != cards && pipeline->failure == OK &&
	       !pipeline->is_ending) {
		progress_wait(driver, &seen, &since, true);
	}
	result_status status = pipeline->failure != OK ? pipeline->failure :
			       pipeline->is_ending     ? STATUS_DEVICE_ERROR :
							 OK;
	(void)pthread_mutex_unlock(&pipeline->lock);
	for (uint32_t sync = 0; sync < slot->sync_count; sync++) {
		const uint32_t *const own =
			&slot->syncs[(uint64_t)sync * STEP_SYNC_FIELDS];
		status = THEN(step_mirror_sync(driver, own[0], own[1],
					       own[STEP_SYNC_FIELDS - 1u]));
	}
	(void)pthread_mutex_lock(&pipeline->lock);
	pipeline->borrowed = 0;
	moved(pipeline);
	(void)pthread_mutex_unlock(&pipeline->lock);
	return status;
}

/* The card's task (task) of the slot's step: see the top. */
MUST_CHECK static result_status task_run(struct step_driver *driver,
					 struct pipeline_slot *slot,
					 uint32_t card, uint32_t task)
{
	struct step_pipeline *const pipeline = driver->pipeline;
	const struct execution_program *const program = slot->program;
	const struct hand_off_record *const hand_offs =
		driver->programs[0]->hand_offs;
	const struct pipeline_visit *const visit =
		visit_of(pipeline, card, task);
	const bool runs = task < slot->visit_count[card],
		   is_last = task + 1u == slot->task_count[card];
	const bool samples =
		runs && is_last && card == program->result_card &&
		!step_is_headless(program, slot->rows, slot->row_count);
	const bool is_current = pipeline->step_on_card[card] == slot->sequence;
	result_status status = runs && slot->sync_count != 0 &&
					       card == driver->mirror_source &&
					       task == pipeline->mirror_visit ?
				       mirrors_synced(driver, slot) :
				       OK;
	status = samples && slot->sessions_changed ?
			 THEN(step_copy_begin(
				 driver, card,
				 program->cards[card].session_offset,
				 slot->block + driver->session_staging,
				 (uint64_t)program->limits.session_capacity *
					 STEP_SESSION_BYTES,
				 false)) :
			 status;
	if (runs && !is_current) {
		status =
			THEN(step_card_begin(driver, slot->block, slot->row_ids,
					     slot->row_count, card, task == 0));
		status = THEN(
			step_views_begin(driver, program, slot->block, card));
		pipeline->step_on_card[card] = slot->sequence;
	}
	uint32_t followed = NO_MARK;
	for (uint32_t index = 0; index < visit->incoming_count && status == OK;
	     index++) {
		const uint32_t hand_off =
			pipeline->incoming[visit->first_incoming + index];
		const uint32_t mark = pipeline->hand_off_marks[hand_off];
		if (!is_handed(driver, slot, hand_off)) {
			continue;
		}
		status = sent_wait(driver, slot, hand_off) ?
				 OK :
				 STATUS_DEVICE_ERROR;
		status = mark >= pipeline->slot_marks || mark == followed ?
				 status :
				 THEN(driver->backend->mark_follow(
					 driver->cards[card],
					 driver->cards[hand_offs[hand_off]
							       .from_card],
					 (uint32_t)(slot - pipeline->slots) *
							 pipeline->slot_marks +
						 mark));
		followed = mark;
		status = THEN(step_hand_off_copies(driver, slot->block,
						   hand_off, slot->rows,
						   slot->row_count, false));
	}
	for (uint32_t segment = visit->first_segment, last = 0;
	     runs && segment <= visit->last_segment &&
	     segment < slot->segment_count && status == OK;
	     segment = ++last) {
		uint32_t hand_off = pipeline->run_hand_offs[segment];
		last = step_run_last(program, segment, hand_off);
		const uint32_t graph = slot->run_graphs != NULL ?
					       slot->run_graphs[segment] :
					       0;
		const uint32_t first_hand_off = hand_off;
		bool is_sent = false;
		driver->uncovered_work[card] = true;
		status = graph != 0 ?
				 driver->backend->graph_run(driver->cards[card],
							    graph - 1u) :
				 driver->backend->launch_run(
					 driver->cards[card],
					 &program->cards[card].launches
						  [program->segments[segment]
							   .first_launch],
					 step_run_launch_count(program, segment,
							       last));
		for (; hand_off < program->hand_off_count &&
		       hand_offs[hand_off].after_segment == last;
		     hand_off++) {
			const bool is_kept = is_handed(driver, slot, hand_off);
			status = is_kept ? THEN(step_hand_off_copies(
						   driver, slot->block,
						   hand_off, slot->rows,
						   slot->row_count, true)) :
					   status;
			is_sent = is_sent || is_kept;
		}
		const uint32_t mark =
			is_sent ? pipeline->hand_off_marks[first_hand_off] :
				  NO_MARK;
		status = !is_sent ?
				 status :
			 mark < pipeline->slot_marks ?
				 THEN(driver->backend->mark_record(
					 driver->cards[card],
					 (uint32_t)(slot - pipeline->slots) *
							 pipeline->slot_marks +
						 mark)) :
				 THEN(card_wait(driver, card));
		(void)pthread_mutex_lock(&pipeline->lock);
		for (uint32_t index = first_hand_off;
		     index < hand_off && status == OK; index++) {
			slot->sent[index] =
				is_handed(driver, slot, index) ? 1u : 0u;
		}
		moved(pipeline);
		(void)pthread_mutex_unlock(&pipeline->lock);
	}
	status = samples ? THEN(step_copy_begin(
				   driver, card,
				   program->cards[card].result_offset,
				   slot->block + driver->result_staging,
				   step_result_span(program, slot->rows,
						    slot->row_count),
				   true)) :
			   status;
	/* every copy of the block and every launch of the step done before its slot may be taken again */
	return is_last ? THEN(card_wait(driver, card)) : status;
}

static void *card_worker(void *argument)
{
	const struct worker_start *const start = argument;
	struct step_driver *const driver = start->driver;
	struct step_pipeline *const pipeline = driver->pipeline;
	const uint32_t card = start->card, bit = 1u << start->card;
	(void)prctl(
		PR_SET_TIMERSLACK, WORKER_TIMER_SLACK_NANOSECONDS, 0, 0,
		0); /* its waits' sleeps end when asked, not up to 50 us late */
	(void)pthread_mutex_lock(&pipeline->lock);
	uint64_t seen = pipeline->moves, since = monotonic_nanoseconds();
	while (pipeline->failure == OK && !pipeline->is_ending) {
		bool has_work = false;
		const bool is_borrowed = (pipeline->borrowed & bit) != 0;
		struct pipeline_slot *const slot =
			is_borrowed ? NULL :
				      ready_slot(driver, card, &has_work);
		if (is_borrowed != ((pipeline->parked & bit) != 0)) {
			pipeline->parked ^= bit; /* stands still, or goes on */
			moved(pipeline);
		}
		if (slot == NULL) {
			/* a borrowed worker has no deadline: its borrower's wait has */
			progress_wait(driver, &seen, &since, has_work);
			continue;
		}
		const uint32_t task = slot->tasks_done[card];
		(void)pthread_mutex_unlock(&pipeline->lock);
		const result_status status = task_run(driver, slot, card, task);
		(void)pthread_mutex_lock(&pipeline->lock);
		pipeline->live_busy_ns[card] = driver->live_busy_ns[card];
		pipeline->live_visits[card] += status == OK ? 1u : 0u;
		pipeline->live_steps[card] +=
			status == OK && task + 1u == slot->task_count[card] ?
				1u :
				0u;
		pipeline->failure = pipeline->failure == OK ? status :
							      pipeline->failure;
		slot->tasks_done[card] += status == OK ? 1u : 0u;
		slot->done_cards |=
			status == OK && slot->tasks_done[card] ==
						slot->task_count[card] ?
				bit :
				0u;
		moved(pipeline);
	}
	pipeline->parked |= bit; /* it touches its card no more */
	moved(pipeline);
	(void)pthread_mutex_unlock(&pipeline->lock);
	return NULL;
}

/* TP flights keep the public queue/finish contract but one coordinator owns every card. Whole-flight retirement
 * prevents shared STEP, scratch, mailbox or state aliases between flights. This mode promises no GPU overlap. */
MUST_CHECK static result_status group_slot_run(struct step_driver *driver,
					       struct pipeline_slot *slot)
{
	const struct execution_program *const program = slot->program;
	result_status status = OK;
	for (uint32_t sync = 0; sync < slot->sync_count && status == OK;
	     sync++) {
		const uint32_t *const own =
			&slot->syncs[(uint64_t)sync * STEP_SYNC_FIELDS];
		status = step_mirror_sync(driver, own[0], own[1],
					  own[STEP_SYNC_FIELDS - 1u]);
	}
	if (slot->sessions_changed)
		status = THEN(step_copy_begin(
			driver, program->result_card,
			program->cards[program->result_card].session_offset,
			slot->block + driver->session_staging,
			(uint64_t)program->limits.session_capacity *
				STEP_SESSION_BYTES,
			false));
	for (uint32_t card = 0; card < program->card_count && status == OK;
	     card++) {
		status = step_card_begin(driver, slot->block, slot->row_ids,
					 slot->row_count, card,
					 false); /* gathered at submit */
		status = THEN(
			step_views_begin(driver, program, slot->block, card));
	}
	status = THEN(step_companions_enqueue(driver, program, slot->run_graphs,
					      slot->rows, slot->row_count,
					      slot->block));
	/* Even a failed submit may have left peer readers queued. Attempt every card's completion proof. */
	for (uint32_t card = 0; card < program->card_count; card++) {
		const result_status waited = card_wait(driver, card);
		status = status == OK ? waited : status;
	}
	return status;
}

static void *group_worker(void *argument)
{
	const struct worker_start *const start = argument;
	struct step_driver *const driver = start->driver;
	struct step_pipeline *const pipeline = driver->pipeline;
	uint64_t next = 1u, seen = 0, since = monotonic_nanoseconds();
	(void)pthread_mutex_lock(&pipeline->lock);
	while (!pipeline->is_ending && pipeline->failure == OK) {
		if (next > pipeline->submitted) {
			progress_wait(driver, &seen, &since, false);
			continue;
		}
		struct pipeline_slot *const slot =
			&pipeline->slots[next % pipeline->depth];
		(void)pthread_mutex_unlock(&pipeline->lock);
		const result_status status = group_slot_run(driver, slot);
		(void)pthread_mutex_lock(&pipeline->lock);
		pipeline->failure = pipeline->failure == OK ? status :
							      pipeline->failure;
		if (status == OK) {
			slot->done_cards = pipeline->every_card;
			uint32_t visited = 0;
			for (uint32_t segment = 0;
			     segment < slot->segment_count; segment++) {
				const uint32_t card =
					slot->program->segments[segment].card;
				pipeline->live_visits[card] += segment == 0 ||
					slot->program->segments[segment - 1u].card != card ? 1u : 0u;
				pipeline->live_steps[card] +=
					(visited & (1u << card)) == 0 ? 1u : 0u;
				visited |= 1u << card;
			}
			next++;
		}
		moved(pipeline);
	}
	pipeline->parked = pipeline->every_card;
	moved(pipeline);
	(void)pthread_mutex_unlock(&pipeline->lock);
	return NULL;
}

/* Each side has one owner, including its backend completion bookkeeping. Downloads live in the flight's block;
 * no native tail kernel reads a CMP arena. The complete prefix wait also retires peer readers before scratch reuse. */
MUST_CHECK static result_status overlap_slot_run(struct step_driver *driver,
						 struct pipeline_slot *slot,
						 bool tail)
{
	const struct execution_program *const program = slot->program;
	const uint32_t first_card = tail ? PREFIX_CARDS : 0u;
	const uint32_t card_end = tail ? OVERLAP_CARDS : PREFIX_CARDS;
	result_status status = OK;
	if (tail && slot->sessions_changed)
		status = step_copy_begin(
			driver, program->result_card,
			program->cards[program->result_card].session_offset,
			slot->block + driver->session_staging,
			(uint64_t)program->limits.session_capacity *
				STEP_SESSION_BYTES,
			false);
	for (uint32_t card = first_card; card < card_end && status == OK;
	     card++) {
		status = step_card_begin(driver, slot->block, slot->row_ids,
					 slot->row_count, card, false);
		status = THEN(
			step_views_begin(driver, program, slot->block, card));
	}
	status = tail ? THEN(step_companions_tail_enqueue(
				driver, program, slot->run_graphs, slot->rows,
				slot->row_count, slot->block,
				driver->pipeline->prefix_segments)) :
			THEN(step_companions_prefix_enqueue(
				driver, program, slot->run_graphs, slot->rows,
				slot->row_count, slot->block,
				driver->pipeline->prefix_segments));
	/* A refused enqueue may already have submitted peer-waiting work. Drain every card owned by this worker. */
	for (uint32_t card = first_card; card < card_end; card++) {
		const result_status waited = card_wait(driver, card);
		status = status == OK ? waited : status;
	}
	return status;
}

static void *overlap_worker(void *argument)
{
	const struct worker_start *const start = argument;
	struct step_driver *const driver = start->driver;
	struct step_pipeline *const pipeline = driver->pipeline;
	const bool tail = start->card != 0;
	const uint32_t cards = tail ? 1u << PREFIX_CARDS : PREFIX_MASK;
	uint64_t next = 1u, seen = 0, since = monotonic_nanoseconds();
	(void)pthread_mutex_lock(&pipeline->lock);
	while (!pipeline->is_ending && pipeline->failure == OK) {
		if (next > pipeline->submitted) {
			progress_wait(driver, &seen, &since, false);
			continue;
		}
		struct pipeline_slot *const slot =
			&pipeline->slots[next % pipeline->depth];
		if (tail && (slot->done_cards & PREFIX_MASK) != PREFIX_MASK) {
			progress_wait(driver, &seen, &since, true);
			continue;
		}
		(void)pthread_mutex_unlock(&pipeline->lock);
		const result_status status =
			overlap_slot_run(driver, slot, tail);
		(void)pthread_mutex_lock(&pipeline->lock);
		pipeline->failure = pipeline->failure == OK ? status :
							      pipeline->failure;
		if (status == OK) {
			slot->done_cards |= cards;
			uint32_t visited = 0;
			for (uint32_t segment = 0;
			     segment < slot->segment_count; segment++) {
				const uint32_t card =
					slot->program->segments[segment].card;
				if ((cards >> card & 1u) == 0)
					continue;
				pipeline->live_visits[card] +=
					segment == 0 || slot->program->segments[segment -
										1u]
									.card !=
								card ?
						1u :
						0u;
				pipeline->live_steps[card] +=
					(visited & (1u << card)) == 0 ? 1u : 0u;
				visited |= 1u << card;
			}
			next++;
		}
		moved(pipeline);
	}
	pipeline->parked |= cards;
	moved(pipeline);
	(void)pthread_mutex_unlock(&pipeline->lock);
	return NULL;
}

/* Fail closed before allocating or starting workers. A short/front program can stop in the common prefix;
 * otherwise its first tail segment must be the common split. The driver already validates shared prefix records. */
static uint32_t overlap_split(const struct step_driver *driver)
{
	if (driver->companion_segments == NULL ||
	    driver->image_programs != NULL || driver->mirror_cards != 0)
		return 0;
	const struct execution_program *const first = driver->programs[0];
	uint32_t split = 0;
	while (split < first->segment_count &&
	       first->segments[split].card < PREFIX_CARDS)
		split++;
	if (split == 0 || split == first->segment_count)
		return 0;
	bool has_quad = false;
	for (uint32_t segment = 0; segment < split; segment++) {
		const uint32_t companion = driver->companion_segments[segment];
		if (companion == STEP_NO_COMPANION || companion < segment)
			continue;
		if (companion - segment != PREFIX_CARDS - 1u ||
		    companion >= split)
			return 0;
		uint32_t mask = 0;
		for (uint32_t member = segment; member <= companion; member++)
			mask |= 1u << first->segments[member].card;
		if (mask != PREFIX_MASK)
			return 0;
		has_quad = true;
	}
	if (!has_quad)
		return 0;
	for (uint32_t index = 0; index < driver->program_count; index++) {
		const struct execution_program *const program =
			driver->programs[index];
		if (program->card_count != OVERLAP_CARDS ||
		    program->result_card != PREFIX_CARDS ||
		    program->kind == PROGRAM_BATCH ||
		    program->expert_pause_count != 0 ||
		    program->front_segment_count > split)
			return 0;
		for (uint32_t segment = 0; segment < program->segment_count;
		     segment++) {
			const uint32_t card = program->segments[segment].card;
			if ((segment < split && card >= PREFIX_CARDS) ||
			    (segment >= split && card != PREFIX_CARDS))
				return 0;
		}
		for (uint32_t hand_off = 0; hand_off < program->hand_off_count;
		     hand_off++) {
			const struct hand_off_record *const transfer =
				&program->hand_offs[hand_off];
			if (transfer->from_card >= PREFIX_CARDS ||
			    transfer->after_segment >= split)
				return 0;
		}
	}
	return split;
}

void step_live_cards(struct step_driver *driver,
		     struct engine_card_stats *cards, uint32_t card_count)
{
	struct step_pipeline *const pipeline = driver->pipeline;
	if (pipeline != NULL) {
		(void)pthread_mutex_lock(&pipeline->lock);
		for (uint32_t card = 0; card < card_count; card++) {
			cards[card].busy_ns = pipeline->live_busy_ns[card];
			cards[card].steps = pipeline->live_steps[card];
			cards[card].visits = pipeline->live_visits[card];
		}
		(void)pthread_mutex_unlock(&pipeline->lock);
	}
	for (uint32_t card = 0; card < card_count; card++) {
		cards[card].busy_available = false;
		cards[card].busy_observed_ns = 0;
		for (uint32_t attempt = 0;
		     attempt < STEP_BUSY_SNAPSHOT_ATTEMPTS; attempt++) {
			const unsigned long long before = atomic_load_explicit(
				&driver->live_wait_sequence[card],
				memory_order_seq_cst);
			if (before % STEP_WAIT_SEQUENCE_STRIDE != 0) {
				continue;
			}
			const uint64_t completed = atomic_load_explicit(
				&driver->live_wait_completed_ns[card],
				memory_order_seq_cst);
			const uint64_t began = atomic_load_explicit(
				&driver->live_wait_started_ns[card],
				memory_order_seq_cst);
			const uint64_t observed = step_clock_ns();
			const unsigned long long after = atomic_load_explicit(
				&driver->live_wait_sequence[card],
				memory_order_seq_cst);
			if (before == after && observed != 0 &&
			    (began == 0 || observed >= began)) {
				cards[card].busy_ns =
					completed +
					(began != 0 ? observed - began : 0);
				cards[card].busy_observed_ns = observed;
				cards[card].busy_available = true;
				break;
			}
		}
	}
}

/* The visit of card that holds segment of the first program: its place among the card's visits. */
static uint32_t visit_holding(const struct step_pipeline *pipeline,
			      uint32_t card, uint32_t segment)
{
	uint32_t local = 0;
	while (local + 1u < pipeline->visit_count[card] &&
	       visit_of(pipeline, card, local)->last_segment < segment) {
		local++;
	}
	return local;
}

/* The tables of the visits (see the top) from the first program, which the driver checked: each card's visits and its
 * closing one, the hand-offs each takes in, the leases; each hand-off's mark, each segment's hand-offs before it. */
static void visits_prepare(const struct step_driver *driver,
			   struct step_pipeline *pipeline)
{
	const struct execution_program *const first = driver->programs[0];
	const struct segment_record *const segments = first->segments;
	uint32_t total = 0;
	for (uint32_t segment = 0; segment < first->segment_count; segment++) {
		pipeline->visit_count[segments[segment].card] +=
			segment == 0 || segments[segment - 1u].card !=
						segments[segment].card ?
				1u :
				0u;
	}
	for (uint32_t card = 0; card < first->card_count; card++) {
		pipeline->first_visit[card] = total;
		total += pipeline->visit_count[card] + 1u;
		pipeline->visits[total - 1u] =
			(struct pipeline_visit){ NO_SEGMENT, NO_SEGMENT, 0, 0,
						 0 };
		pipeline->early_cards |=
			pipeline->visit_count[card] == 1u &&
					(driver->mirror_cards >> card & 1u) ==
						0 ?
				1u << card :
				0u;
	}
	uint32_t seen[BACKEND_MAXIMUM_CARDS] = { 0 };
	for (uint32_t segment = 0; segment < first->segment_count; segment++) {
		const uint32_t card = segments[segment].card;
		if (segment == 0 || segments[segment - 1u].card != card) {
			pipeline->visits[pipeline->first_visit[card] +
					 seen[card]++] =
				(struct pipeline_visit){ segment, segment, 0, 0,
							 0 };
		}
		pipeline->visits[pipeline->first_visit[card] + seen[card] - 1u]
			.last_segment = segment;
	}
	/* a hand-off goes up before the first visit of its card behind its segment: count, place, fill */
	for (uint32_t pass = 0; pass <= 1u; pass++) {
		for (uint32_t index = 0; index < first->hand_off_count;
		     index++) {
			const struct hand_off_record *const hand_off =
				&first->hand_offs[index];
			uint32_t local = 0;
			while (local < pipeline->visit_count[hand_off->to_card] &&
			       visit_of(pipeline, hand_off->to_card, local)
					       ->first_segment <
				       hand_off->after_segment) {
				local++;
			}
			struct pipeline_visit *const visit =
				&pipeline->visits[pipeline->first_visit
							  [hand_off->to_card] +
						  local];
			if (pass != 0) {
				pipeline->incoming[visit->first_incoming +
						   visit->incoming_count] =
					index;
			}
			visit->incoming_count++;
		}
		for (uint32_t visit = 0, placed = 0; visit < total && pass == 0;
		     visit++) {
			pipeline->visits[visit].first_incoming = placed;
			placed += pipeline->visits[visit].incoming_count;
			pipeline->visits[visit].incoming_count = 0;
		}
	}
	for (uint32_t index = 0; index < first->lease_count; index++) {
		const struct visit_lease *const lease = &first->leases[index];
		const uint32_t end = visit_holding(pipeline, lease->card,
						   lease->last_segment);
		struct pipeline_visit *const visit =
			&pipeline->visits[pipeline->first_visit[lease->card] +
					  visit_holding(pipeline, lease->card,
							lease->first_segment)];
		visit->lease_end = end > visit->lease_end ? end :
							    visit->lease_end;
	}
	/* every worker numbers the sending runs alike: a run's mark on its card is what its receivers follow */
	for (uint32_t segment = 0, last = 0, hand_off = 0, sending = 0;
	     segment < first->segment_count; segment = ++last) {
		last = step_run_last(first, segment, hand_off);
		const uint32_t first_hand_off = hand_off;
		while (hand_off < first->hand_off_count &&
		       first->hand_offs[hand_off].after_segment == last) {
			pipeline->hand_off_marks[hand_off++] = sending;
		}
		sending += hand_off != first_hand_off ? 1u : 0u;
	}
	for (uint32_t segment = 0, hand_off = 0; segment < first->segment_count;
	     segment++) {
		while (hand_off < first->hand_off_count &&
		       first->hand_offs[hand_off].after_segment < segment) {
			hand_off++;
		}
		pipeline->run_hand_offs[segment] = hand_off;
	}
	pipeline->mirror_visit = driver->mirror_source < first->card_count ?
					 visit_holding(pipeline,
						       driver->mirror_source,
						       driver->mirror_segment) :
					 NO_SEGMENT;
}

/* A submitted step's tasks on each card (see the top); a card with none is done with it. */
static void tasks_count(const struct step_driver *driver,
			struct pipeline_slot *slot)
{
	const struct step_pipeline *const pipeline = driver->pipeline;
	slot->done_cards = 0;
	for (uint32_t card = 0; card < driver->programs[0]->card_count;
	     card++) {
		uint32_t count = 0;
		while (count < pipeline->visit_count[card] &&
		       visit_of(pipeline, card, count)->first_segment <
			       slot->segment_count) {
			count++;
		}
		const struct pipeline_visit *const next =
			visit_of(pipeline, card, count);
		bool is_handed_one = false;
		for (uint32_t index = 0;
		     index < next->incoming_count && !is_handed_one; index++) {
			is_handed_one = is_handed(
				driver, slot,
				pipeline->incoming[next->first_incoming +
						   index]);
		}
		slot->visit_count[card] = count;
		slot->task_count[card] = count + (is_handed_one ? 1u : 0u);
		slot->tasks_done[card] = 0;
		slot->done_cards |= slot->task_count[card] == 0 ? 1u << card :
								  0u;
	}
}

MUST_CHECK static result_status pipeline_start(struct step_driver *driver,
					       uint32_t depth, bool overlap)
{
	if (driver != NULL && driver->failure != OK) {
		return driver->failure;
	}
	if (step_rounds_is_batch(driver)) {
		if (overlap || depth != 1u || driver->pipeline != NULL ||
		    driver->sequence != 0 || driver->batch_eager_started)
			return STATUS_INVALID_ARGUMENT;
		/* BATCH is one eager, all-owner transaction. Queued history and rollback are not admitted. */
		driver->batch_eager_started = true;
		return OK;
	}
	if (driver == NULL || driver->pipeline != NULL ||
	    driver->sequence != 0 || depth < LEAST_DEPTH ||
	    depth > STEP_PIPELINE_MOST) {
		LOG_ERROR(
			"step pipeline: refused: no driver, a second or late start, or a depth outside %u to %u",
			LEAST_DEPTH, STEP_PIPELINE_MOST);
		return STATUS_INVALID_ARGUMENT;
	}
	const struct execution_program *const first = driver->programs[0];
	const uint32_t split = overlap ? overlap_split(driver) : 0u;
	if (overlap && split == 0)
		return STATUS_UNSUPPORTED;
	if (first->expert_pause_count != 0) {
		LOG_ERROR(
			"step pipeline: programs with expert pauses take a step at a time");
		return STATUS_UNSUPPORTED;
	}
	const uint64_t slot_bytes =
		(driver->staging_bytes + BACKEND_STAGING_ALIGNMENT_BYTES - 1u) /
		BACKEND_STAGING_ALIGNMENT_BYTES *
		BACKEND_STAGING_ALIGNMENT_BYTES;
	const uint64_t slot_row_ids = (uint64_t)driver->largest_row_count *
				      driver->table_count *
				      driver->rows_per_token;
	const uint64_t slot_syncs =
		(uint64_t)first->limits.session_capacity * STEP_SYNC_FIELDS;
	struct step_pipeline *const pipeline = calloc(1, sizeof(*pipeline));
	if (pipeline == NULL) {
		return STATUS_OUT_OF_MEMORY;
	}
	pipeline->depth = depth;
	pipeline->prefix_segments = split;
	pipeline->slot_marks = driver->backend->mark_record != NULL &&
					       driver->backend->mark_follow !=
						       NULL ?
				       BACKEND_MAXIMUM_MARKS / depth :
				       0;
	pipeline->row_ids = calloc(depth * slot_row_ids + 1u, sizeof(uint64_t));
	pipeline->sent = calloc((uint64_t)depth * first->hand_off_count + 1u,
				sizeof(uint8_t));
	pipeline->rows = calloc((uint64_t)depth * driver->largest_row_count,
				sizeof(*pipeline->rows));
	pipeline->visits =
		calloc((uint64_t)first->segment_count + first->card_count,
		       sizeof(*pipeline->visits));
	/* the syncs, then the hand-offs in, their marks, the segments' hand-offs before them */
	pipeline->syncs = calloc(depth * slot_syncs +
					 (uint64_t)HAND_OFF_TABLES *
						 first->hand_off_count +
					 first->segment_count + 1u,
				 sizeof(uint32_t));
	result_status status = pipeline->row_ids == NULL ||
					       pipeline->sent == NULL ||
					       pipeline->rows == NULL ||
					       pipeline->visits == NULL ||
					       pipeline->syncs == NULL ?
				       STATUS_OUT_OF_MEMORY :
				       driver->backend->staging_allocate(
					       depth * slot_bytes,
					       (void **)&pipeline->staging);
	pthread_condattr_t monotonic;
	pipeline->has_lock = status == OK &&
			     pthread_mutex_init(&pipeline->lock, NULL) == 0;
	const bool has_attributes = pipeline->has_lock &&
				    pthread_condattr_init(&monotonic) == 0;
	pipeline->has_condition =
		has_attributes &&
		pthread_condattr_setclock(&monotonic, CLOCK_MONOTONIC) == 0 &&
		pthread_cond_init(&pipeline->changed, &monotonic) == 0;
	if (has_attributes) {
		(void)pthread_condattr_destroy(&monotonic);
	}
	status = status == OK && !pipeline->has_condition ?
			 STATUS_OUT_OF_MEMORY :
			 status;
	if (status == OK) {
		pipeline->incoming = pipeline->syncs + depth * slot_syncs;
		pipeline->hand_off_marks =
			pipeline->incoming + first->hand_off_count;
		pipeline->run_hand_offs =
			pipeline->hand_off_marks + first->hand_off_count;
		visits_prepare(driver, pipeline);
	}
	for (uint32_t index = 0; index < depth && status == OK; index++) {
		pipeline->slots[index] = (struct pipeline_slot){
			.block = pipeline->staging + index * slot_bytes,
			.row_ids = pipeline->row_ids + index * slot_row_ids,
			.sent = pipeline->sent +
				(uint64_t)index * first->hand_off_count,
			.syncs = pipeline->syncs + index * slot_syncs,
			.rows = pipeline->rows +
				(uint64_t)index * driver->largest_row_count
		};
	}
	driver->pipeline = pipeline;
	sigset_t sent_signals, before;
	bool is_masked = status == OK && sigfillset(&sent_signals) == 0;
	for (uint32_t index = 0;
	     index < sizeof(fault_signals) / sizeof(fault_signals[0]);
	     index++) {
		is_masked = is_masked &&
			    sigdelset(&sent_signals, fault_signals[index]) == 0;
	}
	is_masked = is_masked &&
		    pthread_sigmask(SIG_SETMASK, &sent_signals, &before) == 0;
	status = status == OK && !is_masked ? STATUS_OUT_OF_MEMORY : status;
	const bool serial_groups = driver->companion_segments != NULL;
	const uint32_t workers = overlap       ? OVERLAP_WORKERS :
				 serial_groups ? 1u :
						 first->card_count;
	if (serial_groups)
		pipeline->every_card = (1u << first->card_count) - 1u;
	for (uint32_t card = 0; card < workers && status == OK; card++) {
		pipeline->starts[card] = (struct worker_start){ driver, card };
		status = pthread_create(&pipeline->workers[card], NULL,
					overlap	      ? overlap_worker :
					serial_groups ? group_worker :
							card_worker,
					&pipeline->starts[card]) == 0 ?
				 OK :
				 STATUS_OUT_OF_MEMORY;
		pipeline->worker_count += status == OK ? 1u : 0u;
		pipeline->every_card |= 1u << card;
	}
	if (is_masked) {
		(void)pthread_sigmask(SIG_SETMASK, &before, NULL);
	}
	if (status != OK) {
		step_pipeline_stop(driver);
		step_pipeline_free(driver, false);
	}
	return status;
}

result_status step_pipeline_start(struct step_driver *driver, uint32_t depth)
{
	return pipeline_start(driver, depth, false);
}

result_status step_pipeline_overlap_start(struct step_driver *driver,
					  uint32_t depth)
{
	return pipeline_start(driver, depth, true);
}

bool step_pipeline_is_idle(const struct step_driver *driver)
{
	return driver->pipeline == NULL ||
	       driver->pipeline->submitted == driver->pipeline->finished;
}

bool step_pipeline_holds(const struct step_driver *driver, uint64_t sequence)
{
	return driver->pipeline != NULL &&
	       sequence > driver->pipeline->finished;
}

uint32_t step_pipeline_depth(const struct step_driver *driver)
{
	return driver != NULL && driver->pipeline != NULL ?
		       driver->pipeline->depth :
		       1u;
}

result_status step_submit(struct step_driver *driver,
			  const struct step_row_record *rows,
			  uint32_t row_count)
{
	if (driver != NULL && driver->failure != OK) {
		return driver->failure;
	}
	struct step_pipeline *const pipeline =
		driver != NULL ? driver->pipeline : NULL;
	result_status status =
		pipeline == NULL ?
			STATUS_INVALID_ARGUMENT :
			step_rows_check(driver, rows, row_count, true);
	if (status != OK) {
		return status;
	}
	if (pipeline->submitted - pipeline->finished >= pipeline->depth) {
		return STATUS_BUSY_VALUE;
	}
	const uint64_t sequence = ++driver->sequence;
	struct pipeline_slot *const slot =
		&pipeline->slots[sequence % pipeline->depth];
	slot->program = step_program_for_rows(driver, rows, row_count,
					      &slot->run_graphs);
	slot->sequence = sequence;
	slot->row_count = row_count;
	slot->is_front = step_is_front(rows, row_count);
	slot->segment_count =
		slot->program != NULL ?
			step_segment_count(slot->program, rows, row_count) :
			0;
	slot->hand_off_count =
		slot->program != NULL ? slot->program->hand_off_count : 0;
	/* the settings go up with the next step that samples (step.h): a step without a head leaves them */
	slot->sessions_changed =
		driver->sessions_changed && slot->program != NULL &&
		!step_is_headless(slot->program, rows, row_count);
	memcpy(slot->rows, rows, row_count * sizeof(*rows));
	memset(slot->sent, 0, driver->programs[0]->hand_off_count);
	step_block_write(slot->block, sequence, rows, row_count);
	if (slot->program != NULL) {
		step_views_write(driver, slot->program, slot->block, sequence,
				 rows, row_count);
	}
	if (slot->sessions_changed) {
		memcpy(slot->block + driver->session_staging,
		       driver->staging + driver->session_staging,
		       (size_t)driver->programs[0]->limits.session_capacity *
			       STEP_SESSION_BYTES);
		driver->sessions_changed = false;
	}
	status = slot->program == NULL ?
			 STATUS_INVALID_ARGUMENT :
			 step_row_ids(driver, rows, row_count, slot->row_ids);
	/* the coordinator of TP flights (group_slot_run) puts the n-gram rows up as gathered here, behind the step in
	 * flight: a chunk's gathers (about 17 ms at 2,048 rows) leave the cards' path between steps */
	for (uint32_t card = 0; status == OK && driver->companion_segments != NULL &&
				card < slot->program->card_count;
	     card++)
		status = THEN(step_card_gather(driver, slot->block,
					       slot->row_ids, row_count, card));
	slot->sync_count = status == OK ?
				   step_mirror_ranges(driver, rows, row_count,
						      slot->syncs) :
				   0;
	tasks_count(driver, slot);
	/* Even a front step with no tail kernels owns uploads to the native card until its tail worker retires them. */
	if (pipeline->prefix_segments != 0)
		slot->done_cards = 0;
	(void)pthread_mutex_lock(&pipeline->lock);
	status = THEN(pipeline->failure);
	if (status == OK) {
		step_history_commit(driver, rows, row_count);
		pipeline->submitted = sequence;
		moved(pipeline);
	}
	(void)pthread_mutex_unlock(&pipeline->lock);
	driver->failure = status;
	return status;
}

result_status step_finish(struct step_driver *driver,
			  struct step_row_result *results_out)
{
	if (driver != NULL && driver->failure != OK) {
		return driver->failure;
	}
	struct step_pipeline *const pipeline =
		driver != NULL ? driver->pipeline : NULL;
	if (pipeline == NULL || results_out == NULL ||
	    pipeline->finished == pipeline->submitted) {
		LOG_ERROR(
			"step pipeline: finish refused: no pipeline, results or step in flight");
		return STATUS_INVALID_ARGUMENT;
	}
	const uint64_t sequence = pipeline->finished + 1u;
	const struct pipeline_slot *const slot =
		&pipeline->slots[sequence % pipeline->depth];
	(void)pthread_mutex_lock(&pipeline->lock);
	uint64_t seen = pipeline->moves, since = monotonic_nanoseconds();
	while (slot->done_cards != pipeline->every_card &&
	       pipeline->failure == OK) {
		progress_wait(driver, &seen, &since, true);
	}
	result_status status = pipeline->failure;
	(void)pthread_mutex_unlock(&pipeline->lock);
	if (slot->program == NULL ||
	    step_is_headless(slot->program, slot->rows, slot->row_count)) {
		step_front_results(slot->row_count, results_out);
	} else {
		status = THEN(step_result_take(driver, slot->block, sequence,
					       slot->rows, slot->row_count,
					       results_out));
	}
	if (status == OK) {
		if (!step_drafts(slot->rows, slot->row_count)) {
			driver->last_rounds = (struct step_round_table){ 0 };
			driver->last_round =
				(struct step_round){ .session_slot =
							     STEP_NO_SESSION };
		}
		driver->has_finished = true;
	}
	if (status != OK) {
		LOG_ERROR("step %" PRIu64
			  " failed (%s): the driver answers so from now on",
			  sequence, status_text(status));
	}
	(void)pthread_mutex_lock(&pipeline->lock);
	pipeline->failure = pipeline->failure == OK ? status :
						      pipeline->failure;
	pipeline->finished = sequence;
	moved(pipeline);
	(void)pthread_mutex_unlock(&pipeline->lock);
	driver->failure = status;
	return status;
}

void step_pipeline_stop(struct step_driver *driver)
{
	struct step_pipeline *const pipeline =
		driver != NULL ? driver->pipeline : NULL;
	if (pipeline == NULL || pipeline->worker_count == 0) {
		return;
	}
	(void)pthread_mutex_lock(&pipeline->lock);
	pipeline->is_ending = true;
	moved(pipeline);
	(void)pthread_mutex_unlock(&pipeline->lock);
	for (uint32_t card = 0; card < pipeline->worker_count; card++) {
		(void)pthread_join(pipeline->workers[card], NULL);
	}
	pipeline->worker_count = 0;
}

void step_pipeline_free(struct step_driver *driver, bool staging_is_read)
{
	struct step_pipeline *const pipeline =
		driver != NULL ? driver->pipeline : NULL;
	if (pipeline == NULL) {
		return;
	}
	if (staging_is_read && driver->backend->staging_retire != NULL) {
		driver->backend->staging_retire(pipeline->staging);
	} else if (!staging_is_read) {
		driver->backend->staging_free(pipeline->staging);
	}
	if (pipeline->has_condition) {
		(void)pthread_cond_destroy(&pipeline->changed);
	}
	if (pipeline->has_lock) {
		(void)pthread_mutex_destroy(&pipeline->lock);
	}
	free(pipeline->row_ids);
	free(pipeline->sent);
	free(pipeline->rows);
	free(pipeline->visits);
	free(pipeline->syncs);
	free(pipeline);
	driver->pipeline = NULL;
}
