/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Optional pair/quad startup schedule. Existing PP program records and eager execution remain unchanged. */
#include <pthread.h>
#include <stdlib.h>

#include "anemone/hooks.h"
#include "step_driver.h"
#include "step_runs.h"

#define COMPANION_MEMBERS 2u
#define COMPANION_QUAD_MEMBERS 4u
#define OK STATUS_OK

static bool pair_is_valid(const struct execution_program *program,
			  const struct step_companion_pair *pair,
			  uint32_t previous_end, bool has_previous,
			  uint32_t members)
{
	const uint32_t first = pair->first_segment,
		       second = pair->second_segment;
	if (first >= program->segment_count ||
	    second >= program->segment_count || second < first ||
	    second - first != members - 1u ||
	    (has_previous && first <= previous_end) ||
	    (program->front_segment_count > first &&
	     program->front_segment_count <= second) ||
	    (program->draft_segment_first > first &&
	     program->draft_segment_first <= second))
		return false;
	const struct segment_record *const left = &program->segments[first];
	uint32_t mask = 0;
	for (uint32_t member = first; member <= second; member++) {
		const struct segment_record *const segment =
			&program->segments[member];
		if (segment->card >= BACKEND_MAXIMUM_CARDS ||
		    (mask & (1u << segment->card)) != 0 ||
		    segment->stage != left->stage ||
		    segment->layer != left->layer || segment->pause != 0)
			return false;
		mask |= 1u << segment->card;
	}
	for (uint32_t index = 0; index < program->hand_off_count; index++) {
		const struct hand_off_record *const hand_off =
			&program->hand_offs[index];
		if (hand_off->after_segment >= first &&
		    hand_off->after_segment <= second &&
		    (mask & (1u << hand_off->to_card)) != 0)
			return false;
	}
	return true;
}

result_status step_groups_attach(struct step_driver *driver,
				 const struct step_companion_program *programs,
				 uint32_t program_count, uint32_t members)
{
	if (driver != NULL && driver->failure != OK)
		return driver->failure;
	if ((members != COMPANION_MEMBERS &&
	     members != COMPANION_QUAD_MEMBERS) ||
	    driver == NULL || programs == NULL || program_count == 0 ||
	    program_count != driver->program_count ||
	    driver->companion_segments != NULL || driver->sequence != 0 ||
	    driver->pipeline != NULL || driver->graphs_attempted)
		return STATUS_INVALID_ARGUMENT;
	if (driver->image_programs != NULL ||
	    driver->programs[0]->expert_pause_count != 0)
		return STATUS_UNSUPPORTED;
	for (uint32_t slot = 0;
	     slot < driver->programs[0]->limits.session_capacity; slot++) {
		if (driver->token_counts[slot] != 0)
			return STATUS_INVALID_ARGUMENT;
	}
	const struct step_companion_program *const first = &programs[0];
	if (first->pairs == NULL || first->pair_count == 0 ||
	    first->arithmetic_identity == 0)
		return STATUS_INVALID_ARGUMENT;
	for (uint32_t program_index = 0; program_index < program_count;
	     program_index++) {
		const struct step_companion_program *const description =
			&programs[program_index];
		const struct execution_program *const program =
			driver->programs[program_index];
		if (description->program != program ||
		    description->reserved_zero != 0 ||
		    description->arithmetic_identity !=
			    first->arithmetic_identity ||
		    description->pair_count >
			    program->segment_count / members ||
		    (description->pair_count != 0 &&
		     description->pairs == NULL))
			return STATUS_INVALID_ARGUMENT;
		uint32_t wanted = 0;
		for (uint32_t index = 0; index < first->pair_count; index++) {
			const struct step_companion_pair *const pair =
				&first->pairs[index];
			if (pair->first_segment >= program->segment_count)
				break;
			if (wanted >= description->pair_count ||
			    description->pairs[wanted].first_segment !=
				    pair->first_segment ||
			    description->pairs[wanted].second_segment !=
				    pair->second_segment ||
			    !pair_is_valid(program, pair,
					   index == 0 ? 0 :
							first->pairs[index - 1u]
								.second_segment,
					   index != 0, members))
				return STATUS_INVALID_ARGUMENT;
			wanted++;
		}
		if (description->pair_count != wanted)
			return STATUS_INVALID_ARGUMENT;
	}
	uint32_t *const map =
		FAULT_POINT(step_driver_memory) ?
			NULL :
			calloc(first->program->segment_count, sizeof(*map));
	if (map == NULL)
		return STATUS_OUT_OF_MEMORY;
	for (uint32_t segment = 0; segment < first->program->segment_count;
	     segment++)
		map[segment] = STEP_NO_COMPANION;
	for (uint32_t index = 0; index < first->pair_count; index++) {
		const struct step_companion_pair *const pair =
			&first->pairs[index];
		map[pair->first_segment] = pair->second_segment;
		for (uint32_t member = pair->first_segment + 1u;
		     member <= pair->second_segment; member++)
			map[member] = pair->first_segment;
	}
	driver->companion_segments = map;
	driver->companion_identity = first->arithmetic_identity;
	return OK;
}

result_status
step_companions_attach(struct step_driver *driver,
		       const struct step_companion_program *programs,
		       uint32_t program_count)
{
	return step_groups_attach(driver, programs, program_count,
				  COMPANION_MEMBERS);
}

MUST_CHECK static result_status
run_enqueue(struct step_driver *driver, const struct execution_program *program,
	    const uint32_t *graphs, uint32_t first, uint32_t last)
{
	const struct segment_record *const segment = &program->segments[first];
	const uint32_t graph = graphs == NULL ? 0 : graphs[first];
	driver->uncovered_work[segment->card] = true;
	const result_status status =
		graph != 0 ?
			driver->backend->graph_run(driver->cards[segment->card],
						   graph - 1u) :
			driver->backend->launch_run(
				driver->cards[segment->card],
				&program->cards[segment->card]
					 .launches[segment->first_launch],
				step_run_launch_count(program, first, last));
	COUNTER_ADD(step_graph_runs, graph != 0 ? 1u : 0u);
	return status;
}

/* The launchers (step_launchers_start): one thread a card; a group posts a job to each member's card and waits
 * for every one (generation counts the posts; pending, the jobs not yet submitted). */
struct launcher_job {
	const struct execution_program *program;
	const uint32_t *graphs;
	uint32_t segment;
	bool is_posted;
	result_status status;
};
struct step_launchers {
	struct step_driver *driver;
	pthread_mutex_t lock;
	pthread_cond_t posted, finished;
	uint64_t generation;
	uint32_t pending, thread_count;
	bool quit;
	struct launcher_job jobs[BACKEND_MAXIMUM_CARDS];
	pthread_t threads[BACKEND_MAXIMUM_CARDS];
	uint32_t thread_cards[BACKEND_MAXIMUM_CARDS];
};
struct launcher_start {
	struct step_launchers *launchers;
	uint32_t card;
};

static void *launcher_main(void *argument)
{
	struct step_launchers *const launchers =
		((struct launcher_start *)argument)->launchers;
	const uint32_t card = ((struct launcher_start *)argument)->card;
	free(argument);
	uint64_t seen = 0;
	(void)pthread_mutex_lock(&launchers->lock);
	for (;;) {
		while (!launchers->quit && (launchers->generation == seen ||
					    !launchers->jobs[card].is_posted))
			(void)pthread_cond_wait(&launchers->posted,
						&launchers->lock);
		if (launchers->quit)
			break;
		seen = launchers->generation;
		const struct launcher_job job = launchers->jobs[card];
		(void)pthread_mutex_unlock(&launchers->lock);
		const result_status status =
			run_enqueue(launchers->driver, job.program, job.graphs,
				    job.segment, job.segment);
		(void)pthread_mutex_lock(&launchers->lock);
		launchers->jobs[card].status = status;
		launchers->jobs[card].is_posted = false;
		if (--launchers->pending == 0)
			(void)pthread_cond_signal(&launchers->finished);
	}
	(void)pthread_mutex_unlock(&launchers->lock);
	return NULL;
}

static void launchers_stop(struct step_launchers *launchers)
{
	(void)pthread_mutex_lock(&launchers->lock);
	launchers->quit = true;
	(void)pthread_cond_broadcast(&launchers->posted);
	(void)pthread_mutex_unlock(&launchers->lock);
	for (uint32_t index = 0; index < launchers->thread_count; index++)
		(void)pthread_join(launchers->threads[index], NULL);
	(void)pthread_cond_destroy(&launchers->posted);
	(void)pthread_cond_destroy(&launchers->finished);
	(void)pthread_mutex_destroy(&launchers->lock);
	free(launchers);
}

void step_launchers_stop(struct step_driver *driver)
{
	if (driver != NULL && driver->launchers != NULL) {
		launchers_stop(driver->launchers);
		driver->launchers = NULL;
	}
}

result_status step_launchers_start(struct step_driver *driver)
{
	if (driver == NULL || driver->companion_segments == NULL ||
	    driver->launchers != NULL)
		return STATUS_INVALID_ARGUMENT;
	struct step_launchers *const launchers = calloc(1, sizeof(*launchers));
	if (launchers == NULL)
		return STATUS_OUT_OF_MEMORY;
	launchers->driver = driver;
	(void)pthread_mutex_init(&launchers->lock, NULL);
	(void)pthread_cond_init(&launchers->posted, NULL);
	(void)pthread_cond_init(&launchers->finished, NULL);
	bool started = true;
	for (uint32_t card = 0; card < BACKEND_MAXIMUM_CARDS && started;
	     card++) {
		if (driver->cards[card] == NULL)
			continue;
		struct launcher_start *const start = malloc(sizeof(*start));
		started = start != NULL;
		if (started) {
			*start = (struct launcher_start){ launchers, card };
			started = pthread_create(
					  &launchers->threads
						   [launchers->thread_count],
					  NULL, launcher_main, start) == 0;
			if (!started)
				free(start);
		}
		if (started)
			launchers->thread_cards[launchers->thread_count++] =
				card;
	}
	if (!started) {
		launchers_stop(launchers);
		return STATUS_INVALID_ARGUMENT;
	}
	driver->launchers = launchers;
	return OK;
}

/* A group's members [first, last] submitted at once by their cards' launchers; the first failure. */
MUST_CHECK static result_status
group_enqueue(struct step_driver *driver,
	      const struct execution_program *program, const uint32_t *graphs,
	      uint32_t first, uint32_t last)
{
	struct step_launchers *const launchers = driver->launchers;
	(void)pthread_mutex_lock(&launchers->lock);
	for (uint32_t member = first; member <= last; member++) {
		struct launcher_job *const job =
			&launchers->jobs[program->segments[member].card];
		*job = (struct launcher_job){ program, graphs, member, true,
					      OK };
	}
	launchers->pending = last - first + 1u;
	launchers->generation++;
	(void)pthread_cond_broadcast(&launchers->posted);
	while (launchers->pending != 0)
		(void)pthread_cond_wait(&launchers->finished, &launchers->lock);
	result_status status = OK;
	for (uint32_t member = first; member <= last; member++) {
		const result_status member_status =
			launchers->jobs[program->segments[member].card].status;
		status = status == OK ? member_status : status;
	}
	(void)pthread_mutex_unlock(&launchers->lock);
	return status;
}

MUST_CHECK static result_status
hand_offs_run(struct step_driver *driver,
	      const struct execution_program *program,
	      const struct step_row_record *rows, uint32_t row_count,
	      uint32_t boundary, uint32_t *next_hand_off, unsigned char *block,
	      bool defer_tail)
{
	const uint32_t first = *next_hand_off;
	const bool is_front = step_is_front(rows, row_count);
	uint32_t end = first;
	bool sent = false;
	result_status status = OK;
	for (; end < program->hand_off_count &&
	       program->hand_offs[end].after_segment == boundary;
	     end++) {
		const bool kept = !is_front || driver->deferred[end] == 0;
		if (kept && status == OK)
			status = step_hand_off_copies(driver, block, end, rows,
						      row_count, true);
		sent = sent || kept;
	}
	if (sent && status == OK)
		status = card_wait(driver, program->segments[boundary].card);
	for (uint32_t index = first; index < end && status == OK; index++) {
		/* Embedding copies must reach the other prefix cards before their quad runs. */
		if ((!defer_tail || program->hand_offs[index].to_card !=
					    program->result_card) &&
		    (!is_front || driver->deferred[index] == 0))
			status = step_hand_off_copies(driver, block, index,
						      rows, row_count, false);
	}
	*next_hand_off = end;
	return status;
}

MUST_CHECK static result_status segments_enqueue(
	struct step_driver *driver, const struct execution_program *program,
	const uint32_t *run_graphs, const struct step_row_record *rows,
	uint32_t row_count, unsigned char *block, uint32_t first_segment,
	uint32_t segment_count, bool defer_tail)
{
	uint32_t hand_off = 0;
	while (hand_off < program->hand_off_count &&
	       program->hand_offs[hand_off].after_segment < first_segment)
		hand_off++;
	result_status status = OK;
	for (uint32_t segment = first_segment, last = 0;
	     segment < segment_count && status == OK; segment = last + 1u) {
		const uint32_t companion = driver->companion_segments[segment];
		const bool paired = companion != STEP_NO_COMPANION;
		last = paired ? companion :
				step_run_last_companions(
					program, segment, hand_off,
					driver->companion_segments);
		if (paired && driver->launchers != NULL) {
			status = group_enqueue(driver, program, run_graphs,
					       segment, last);
		} else {
			status = run_enqueue(driver, program, run_graphs,
					     segment, paired ? segment : last);
		}
		for (uint32_t member = segment + 1u;
		     paired && driver->launchers == NULL && member <= last;
		     member++) {
			/* A failed submit may have queued a peer-waiting kernel. Attempt every member before drain. */
			const result_status peer_status = run_enqueue(
				driver, program, run_graphs, member, member);
			status = status == OK ? peer_status : status;
		}
		/* Both local graphs are submitted before either member's outgoing download or completion wait. */
		for (uint32_t boundary = paired ? segment : last;
		     boundary <= last && status == OK; boundary++)
			status = hand_offs_run(driver, program, rows, row_count,
					       boundary, &hand_off, block,
					       defer_tail);
	}
	return status;
}

MUST_CHECK static result_status
result_enqueue(struct step_driver *driver,
	       const struct execution_program *program,
	       const struct step_row_record *rows, uint32_t row_count,
	       unsigned char *block, result_status status)
{
	if (step_is_headless(program, rows, row_count))
		return status;
	const uint32_t card = program->result_card;
	if (status == OK)
		status = step_copy_begin(
			driver, card, program->cards[card].result_offset,
			block + driver->result_staging,
			step_result_span(program, rows, row_count), true);
	if (status == OK)
		status = card_wait(driver, card);
	return status;
}

result_status step_companions_enqueue(struct step_driver *driver,
				      const struct execution_program *program,
				      const uint32_t *run_graphs,
				      const struct step_row_record *rows,
				      uint32_t row_count, unsigned char *block)
{
	const result_status status = segments_enqueue(
		driver, program, run_graphs, rows, row_count, block, 0,
		step_segment_count(program, rows, row_count), false);
	return result_enqueue(driver, program, rows, row_count, block, status);
}

result_status step_companions_prefix_enqueue(
	struct step_driver *driver, const struct execution_program *program,
	const uint32_t *run_graphs, const struct step_row_record *rows,
	uint32_t row_count, unsigned char *block, uint32_t split_segment)
{
	const uint32_t count = step_segment_count(program, rows, row_count);
	return segments_enqueue(driver, program, run_graphs, rows, row_count,
				block, 0,
				count < split_segment ? count : split_segment,
				true);
}

result_status step_companions_tail_enqueue(
	struct step_driver *driver, const struct execution_program *program,
	const uint32_t *run_graphs, const struct step_row_record *rows,
	uint32_t row_count, unsigned char *block, uint32_t split_segment)
{
	const uint32_t count = step_segment_count(program, rows, row_count);
	result_status status = OK;
	/* Prefix has retired every download. The slot cannot be reused until this tail has retired its uploads. */
	for (uint32_t index = 0;
	     index < program->hand_off_count && status == OK; index++) {
		const struct hand_off_record *const hand_off =
			&program->hand_offs[index];
		if (hand_off->to_card == program->result_card &&
		    hand_off->after_segment < count &&
		    hand_off->after_segment < split_segment)
			status = step_hand_off_copies(driver, block, index,
						      rows, row_count, false);
	}
	if (status == OK && split_segment < count)
		status = segments_enqueue(driver, program, run_graphs, rows,
					  row_count, block, split_segment,
					  count, false);
	return result_enqueue(driver, program, rows, row_count, block, status);
}

result_status step_companions_run(struct step_driver *driver,
				  const struct execution_program *program,
				  const uint32_t *run_graphs,
				  const struct step_row_record *rows,
				  uint32_t row_count,
				  struct step_row_result *results_out)
{
	const result_status status = step_companions_enqueue(
		driver, program, run_graphs, rows, row_count, driver->staging);
	if (status != OK)
		return status;
	if (step_is_headless(program, rows, row_count)) {
		step_front_results(row_count, results_out);
		return OK;
	}
	return step_result_take(driver, driver->staging, driver->sequence, rows,
				row_count, results_out);
}
