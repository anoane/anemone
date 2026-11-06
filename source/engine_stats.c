/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Product counters, independent of the optional diagnostic/timing switches. */
#include <string.h>
#include <time.h>

#include "engine_internal.h"
#include "step_driver.h"
#include "step_runs.h"

#define OK STATUS_OK
#define NANOSECONDS_PER_SECOND 1000000000ull

uint64_t engine_now_ns(void)
{
	struct timespec now;
	return clock_gettime(CLOCK_MONOTONIC, &now) == 0 && now.tv_sec >= 0 ?
		       (uint64_t)now.tv_sec * NANOSECONDS_PER_SECOND +
			       (uint64_t)now.tv_nsec :
		       0;
}

void engine_live_step(struct inference_engine *engine,
		      const struct step_row_record *rows, uint32_t row_count,
		      uint64_t submitted_ns)
{
	struct engine_stats *const worker = &engine->worker_live;
	const uint64_t now = engine_now_ns();
	const uint32_t *run_graphs = NULL;
	const struct execution_program *const program = step_program_for_rows(
		engine->driver, rows, row_count, &run_graphs);
	const uint32_t kind = program != NULL && program->kind == PROGRAM_FULL ?
				      STATS_FULL_KIND :
			      program != NULL && program->kind == PROGRAM_WIDE ?
				      STATS_WIDE_KIND :
			      row_count != 0 && row_count <= STATS_THIN_KINDS ?
				      row_count - 1u :
				      STATS_WIDE_KIND;
	worker->steps_landed++;
	worker->steps[kind]++;
	worker->steps[STATS_DRAFT_KIND] += step_drafts(rows, row_count) ? 1u :
									  0u;
	worker->step_samples[worker->step_ring_next] =
		(struct stats_step_sample){
			now, submitted_ns != 0 && now >= submitted_ns ?
				     now - submitted_ns :
				     0
		};
	worker->step_ring_next =
		(worker->step_ring_next + 1u) % STATS_STEP_RING;
	worker->step_ring_count +=
		worker->step_ring_count < STATS_STEP_RING ? 1u : 0u;
	uint32_t seen = 0;
	for (uint32_t index = 0; index < row_count; index++) {
		const uint32_t owner = rows[index].session_slot;
		struct command_cursor *const cursor = &engine->cursors[owner];
		const uint32_t prompt_end = cursor->command.kept_token_count +
					    cursor->command.new_token_count;
		worker->prefill_tokens +=
			(uint32_t)rows[index].position < prompt_end ? 1u : 0u;
		if ((uint32_t)rows[index].position < prompt_end) {
			const uint32_t completed =
				(uint32_t)rows[index].position + 1u;
			cursor->live_prompt_processed =
				completed > cursor->live_prompt_processed ?
					completed :
					cursor->live_prompt_processed;
		}
		if ((uint32_t)rows[index].position < prompt_end &&
		    (seen & (1u << owner)) == 0) {
			cursor->live_chunks[kind]++;
			seen |= 1u << owner;
		}
		if ((rows[index].flags & STEP_ROW_DRAFTS) != 0) {
			cursor->live_chunks[STATS_DRAFT_KIND] +=
				(uint32_t)rows[index].position < prompt_end ?
					1u :
					0u;
			cursor->live_dspark_tails++;
		}
	}
}

void engine_live_publish(struct inference_engine *engine)
{
	struct engine_stats *const live = &engine->live;
	const struct engine_stats *const worker = &engine->worker_live;
	live->flights = engine->flight_count;
	if (live->steps_landed != worker->steps_landed) {
		live->steps_landed = worker->steps_landed;
		live->prefill_tokens = worker->prefill_tokens;
		live->step_ring_count = worker->step_ring_count;
		live->step_ring_next = worker->step_ring_next;
		memcpy(live->steps, worker->steps, sizeof(live->steps));
		memcpy(live->step_samples, worker->step_samples,
		       sizeof(live->step_samples));
		live->batch_program_count = worker->batch_program_count;
		memcpy(live->batch_programs, worker->batch_programs,
		       sizeof(live->batch_programs));
	}
	for (uint32_t owner = 0; owner < engine->description.session_slot_count;
	     owner++) {
		const struct command_cursor *const cursor =
			&engine->cursors[owner];
		struct engine_slot_stats *const slot = &live->slots[owner];
		const struct slot *const state = &engine->slots[owner];
		if (slot->request_tag != cursor->command.request_tag ||
		    cursor->session.operation != 0 || slot->submitted_ns == 0 ||
		    slot->finish_kind != 0) {
			continue;
		}
		slot->position = cursor->position;
		slot->reach = state->reach_count;
		slot->decoder_from = state->decoder_from;
		slot->checkpoint = state->checkpoint_count;
		slot->kept_lag = SERVING_KEPT_LAG;
		slot->reach_lag =
			state->reach_count > cursor->position ?
				state->reach_count - cursor->position :
				0;
		slot->prompt_processed = cursor->live_prompt_processed;
		slot->chosen_depth = cursor->live_chosen_depth;
		slot->draft_source = cursor->draft_source;
		slot->has_confidence = cursor->has_draft_confidence;
		if (slot->speculation.drawing_rounds !=
		    cursor->statistics.drawing_rounds) {
			for (uint32_t position = 0;
			     position < DRAFT_CALIBRATION_POSITIONS;
			     position++) {
				slot->confidence_bins
					[position] = draft_confidence_bin(
					cursor->draft_confidence[position]);
			}
			slot->speculation = cursor->statistics;
		}
		memcpy(slot->chunks, cursor->live_chunks, sizeof(slot->chunks));
		slot->probes = cursor->live_probes;
		slot->dspark_tails = cursor->live_dspark_tails;
	}
	if (engine->driver->pipeline == NULL) {
		for (uint32_t card = 0; card < live->card_count; card++) {
			live->cards[card].busy_ns =
				engine->driver->live_busy_ns[card];
			live->cards[card].steps =
				engine->driver->live_steps[card];
			live->cards[card].visits =
				engine->driver->live_visits[card];
		}
	}
}

result_status engine_stats_slot(struct inference_engine *engine,
				uint32_t session_slot,
				struct engine_slot_stats *stats_out)
{
	if (engine == NULL || stats_out == NULL ||
	    session_slot >= engine->description.session_slot_count) {
		return STATUS_INVALID_ARGUMENT;
	}
	(void)pthread_mutex_lock(&engine->lock);
	*stats_out = engine->live.slots[session_slot];
	(void)pthread_mutex_unlock(&engine->lock);
	return OK;
}

result_status engine_stats_counters(struct inference_engine *engine,
				    struct engine_stats *stats_out)
{
	if (engine == NULL || stats_out == NULL) {
		return STATUS_INVALID_ARGUMENT;
	}
	(void)pthread_mutex_lock(&engine->lock);
	*stats_out = engine->live;
	stats_out->now_ns = engine_now_ns();
	stats_out->slots_total = engine->description.session_slot_count;
	stats_out->queued_commands = engine->command_count;
	stats_out->event_queue_depth = engine->event_count;
	stats_out->speculation = engine->statistics;
	for (uint32_t owner = 0; owner < stats_out->slots_total; owner++) {
		stats_out->slots_busy +=
			engine->slots[owner].phase != SLOT_IDLE ? 1u : 0u;
		if (engine->slots[owner].phase == SLOT_COMMANDED &&
		    stats_out->slots[owner].finish_kind == 0) {
			draft_statistics_add(
				&stats_out->speculation,
				&stats_out->slots[owner].speculation);
			stats_out->probes += stats_out->slots[owner].probes;
			stats_out->dspark_tails +=
				stats_out->slots[owner].dspark_tails;
		}
		struct engine_slot_stats *const slot = &stats_out->slots[owner];
		struct engine_session_stats *const session =
			&stats_out->sessions[owner];
		const struct slot *const state = &engine->slots[owner];
		session->active = state->phase != SLOT_IDLE;
		session->context_capacity =
			engine->description.context_token_limit;
		session->position = state->processed_count;
		session->reach = state->reach_count;
		session->phase = session->active ? slot->phase : STATS_IDLE;
		if (session->active && session->operation == 0 &&
		    slot->request_tag == state->request_tag) {
			session->position = slot->position;
		}
		if (!session->active && session->idle_since_ns == 0) {
			session->idle_since_ns = stats_out->started_ns;
		}
		const uint64_t end = slot->finished_ns != 0 ?
					     slot->finished_ns :
					     stats_out->now_ns;
		if (slot->first_token_ns == 0 && slot->started_ns != 0) {
			slot->prefill_ns = end >= slot->started_ns ?
						   end - slot->started_ns :
						   0;
		}
		slot->decode_ns = slot->first_token_ns != 0 &&
						  end >= slot->first_token_ns ?
					  end - slot->first_token_ns :
					  0;
	}
	(void)pthread_mutex_unlock(&engine->lock);
	step_live_cards(engine->driver, stats_out->cards,
			stats_out->card_count);
	return OK;
}

result_status engine_stats_snapshot(struct inference_engine *engine,
				    struct engine_stats *stats_out)
{
	const result_status copied = engine_stats_counters(engine, stats_out);
	if (copied != OK) {
		return copied;
	}
	for (uint32_t card = 0; card < stats_out->card_count; card++) {
		struct engine_card_stats *const output =
			&stats_out->cards[card];
		struct card_description_record description;
		const result_status status =
			engine->driver->backend->card_describe(
				output->backend_index, &description);
		output->free_memory_available = status == OK;
		output->description.memory_free_bytes =
			status == OK ? description.memory_free_bytes : 0;
	}
	return OK;
}

result_status engine_stats_cards(struct inference_engine *engine,
				 const struct engine_card_stats *cards,
				 uint32_t card_count)
{
	if (engine == NULL || card_count > BACKEND_MAXIMUM_CARDS ||
	    (card_count != 0 && cards == NULL)) {
		return STATUS_INVALID_ARGUMENT;
	}
	(void)pthread_mutex_lock(&engine->lock);
	const result_status status =
		engine->command_count != 0 || engine->live.prompt_tokens != 0 ?
			STATUS_BUSY_VALUE :
			OK;
	if (status == OK) {
		engine->live.card_count = card_count;
		if (card_count != 0) {
			memcpy(engine->live.cards, cards,
			       card_count * sizeof(*cards));
		}
	}
	(void)pthread_mutex_unlock(&engine->lock);
	return status;
}
