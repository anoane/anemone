/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * scheduler_requests.c: the front end's calls on the server thread: submit, cancel, events, draft statistics (split
 * from scheduler.c by whole items).
 */
#include "front_end_seam.h"
#include "engine/library/sampling/sampler_records.h"

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/eventfd.h>
#include <unistd.h>

#include "engine/input/image/request_images.h"
#include "engine/runtime/step/step_images.h"
#include "common/hooks.h"
#include "engine/runtime/step/step.h"

#include "scheduler_internal.h"
#include "engine/runtime/speculation/draft_session.h"
#include "serving_cursor.h"
#include "engine/runtime/step/step_rounds.h"

result_status engine_draft_statistics(struct inference_engine *engine,
				      uint32_t session_slot,
				      struct draft_statistics *command_out,
				      struct draft_statistics *total_out)
{
	if (engine == NULL ||
	    session_slot >= engine->description.session_slot_count ||
	    (command_out == NULL && total_out == NULL) ||
	    (command_out != NULL && command_out == total_out)) {
		return STATUS_INVALID_ARGUMENT;
	}
	(void)pthread_mutex_lock(&engine->lock);
	const result_status status = engine->slots[session_slot].phase ==
						     SLOT_COMMANDED ?
					     STATUS_BUSY_VALUE :
					     OK;
	if (status == OK) {
		if (command_out != NULL) {
			*command_out = engine->slots[session_slot].statistics;
		}
		if (total_out != NULL) {
			*total_out = engine->statistics;
		}
	}
	(void)pthread_mutex_unlock(&engine->lock);
	return status;
}

/* engine_submit's, and engine_submit_forced's with forced (its own refusals checked). */
MUST_CHECK static result_status
command_submitted(struct inference_engine *engine,
		  const struct generation_command_record *command,
		  const struct engine_forced_continuation *forced)
{
	if (engine == NULL || command == NULL ||
	    command->session_slot >= engine->description.session_slot_count ||
	    command->new_token_ids == NULL || command->new_token_count == 0 ||
	    command->reserved_must_be_zero != 0 ||
	    (engine->description.prefill_prefix_stable != 0 &&
	     (command->kept_token_count % SERVING_STABLE_GRANULE != 0 ||
	      command->images != NULL)) ||
	    (command->images != NULL &&
	     !step_images_supported(engine->driver)) ||
	    command->control_reserved_zero != 0 ||
	    (command->reasoning_budget != 0 &&
	     command->reasoning_tokenizer == NULL) ||
	    command->sampler_top_k > SAMPLER_MOST_VOCABULARY ||
	    (command->reasoning_budget != 0 &&
	     (command->reasoning_end_token >=
		      engine->description.vocabulary_size ||
	      command->reasoning_end_token == engine->description.end_token_id ||
	      command->reasoning_budget >= command->generated_token_limit)) ||
	    !isfinite(command->sampler_temperature) ||
	    command->sampler_temperature < 0.0f ||
	    !(command->sampler_top_p > 0.0f &&
	      command->sampler_top_p <= 1.0f)) {
		return STATUS_INVALID_ARGUMENT;
	}
	const uint32_t context = engine->description.context_token_limit;
	(void)pthread_mutex_lock(&engine->lock);
	struct slot *const slot = &engine->slots[command->session_slot];
	const bool restores_checkpoint =
		command->kept_token_count != 0 &&
		slot->checkpoint_count == command->kept_token_count &&
		slot->reach_count >= command->kept_token_count &&
		slot->reach_count - command->kept_token_count >
			SERVING_KEPT_LAG;
	bool duplicate_tag = false;
	for (uint32_t other = 0; other < engine->description.session_slot_count;
	     other++) {
		duplicate_tag = duplicate_tag ||
				(engine->slots[other].phase != SLOT_IDLE &&
				 engine->slots[other].request_tag ==
					 command->request_tag);
	}
	const result_status status =
		engine->failure != OK ? engine->failure :
		engine->is_ending     ? STATUS_BUSY_VALUE :
		slot->phase != SLOT_IDLE ||
				engine->slot_read_leased[command->session_slot] ||
				engine->command_count ==
					engine->description.session_slot_count ?
				    STATUS_BUSY_VALUE :
		duplicate_tag ? STATUS_INVALID_ARGUMENT :
		command->kept_token_count > slot->token_count ||
				command->kept_token_count > context ||
				command->new_token_count >
					context - command->kept_token_count ?
				STATUS_INVALID_ARGUMENT :
		command->kept_token_count > slot->processed_count ||
				command->kept_token_count %
						SERVING_KEPT_GRANULE !=
					0 ||
				(command->kept_token_count != 0 &&
				 !restores_checkpoint &&
				 slot->reach_count - command->kept_token_count >
					 SERVING_KEPT_LAG) ?
				STATUS_INVALID_ARGUMENT :
		engine->description.session_slot_count > 1 &&
				command->generated_token_limit != 0 &&
				command_end(command) == context ?
				STATUS_INVALID_ARGUMENT :
				OK;
	if (status == OK &&
	    engine->slot_generations[command->session_slot] == UINT64_MAX) {
		(void)pthread_mutex_unlock(&engine->lock);
		return STATUS_UNSUPPORTED;
	}
	if (status == OK) {
		engine->slot_generations[command->session_slot]++;
		engine->clone_metrics[command->session_slot] =
			(struct session_clone_metrics){ 0 };
		engine->live.slots[command->session_slot] =
			(struct engine_slot_stats){
				.slot = command->session_slot,
				.request_tag = command->request_tag,
				.phase = STATS_QUEUED,
				.prompt_tokens = command_end(command),
				.reused_tokens = command->kept_token_count,
				.prompt_processed = command->kept_token_count,
				.position = command->kept_token_count,
				.context_capacity = context,
				.speculation_mode =
					engine->live.speculation_mode,
				.reused_checkpoint = restores_checkpoint,
				.submitted_ns = engine_now_ns()
			};
		engine->live.slots[command->session_slot].phase_since_ns =
			engine->live.slots[command->session_slot].submitted_ns;
		engine->live.sessions[command->session_slot].operation = 0;
		engine->live.sessions[command->session_slot].idle_since_ns = 0;
		engine->live.prompt_tokens += command_end(command);
		engine->live.cached_tokens += command->kept_token_count;
		engine->commands[(engine->command_first + engine->command_count) %
				 engine->description.session_slot_count] =
			(struct queued_command){ .generation = *command,
						 .restores_checkpoint =
							 restores_checkpoint };
		engine->command_count++;
		*slot = (struct slot){ SLOT_COMMANDED,
				       slot->token_count,
				       false,
				       command->request_tag,
				       slot->processed_count,
				       slot->decoder_from,
				       command->kept_token_count != 0 ?
					       slot->reach_count :
					       0,
				       0,
				       slot->statistics };
		if (forced != NULL)
			engine->forced = forced;
		(void)pthread_cond_signal(&engine->wake);
	}
	(void)pthread_mutex_unlock(&engine->lock);
	return status;
}

result_status engine_submit(struct inference_engine *engine,
			    const struct generation_command_record *command)
{
	return command_submitted(engine, command, NULL);
}

result_status
engine_submit_forced(struct inference_engine *engine,
		     const struct generation_command_record *command,
		     const struct engine_forced_continuation *forced)
{
	bool refused =
		engine == NULL || command == NULL || forced == NULL ||
		forced->token_ids == NULL || forced->rows_landed == NULL ||
		engine->description.session_slot_count != 1 ||
		engine->is_batch || command->kept_token_count != 0 ||
		command->sampler_temperature != 0.0f ||
		command->sampler_top_p != 1.0f || command->sampler_top_k != 0 ||
		command->reasoning_budget != 0 || command->images != NULL ||
		forced->token_count == 0 ||
		command->generated_token_limit != forced->token_count ||
		forced->rows == 0 ||
		forced->rows > engine->description.speculative_depth + 1u ||
		forced->token_count > engine->description.context_token_limit ||
		command->new_token_count >
			engine->description.context_token_limit + 1u -
				forced->token_count;
	for (uint32_t index = 0; !refused && index < forced->token_count;
	     index++)
		refused = forced->token_ids[index] >=
			  engine->description.vocabulary_size;
	return refused ? STATUS_INVALID_ARGUMENT :
			 command_submitted(engine, command, forced);
}

result_status engine_cancel(struct inference_engine *engine,
			    uint32_t session_slot)
{
	if (engine == NULL ||
	    session_slot >= engine->description.session_slot_count) {
		return STATUS_INVALID_ARGUMENT;
	}
	(void)pthread_mutex_lock(&engine->lock);
	struct slot *const slot = &engine->slots[session_slot];
	slot->is_cancelled = slot->is_cancelled ||
			     slot->phase == SLOT_COMMANDED;
	(void)pthread_mutex_unlock(&engine->lock);
	return OK;
}

/* Takes the oldest event, if there is one. */
static bool event_taken(struct inference_engine *engine,
			struct generation_event *event_out)
{
	(void)pthread_mutex_lock(&engine->lock);
	const bool is_taken = engine->event_count != 0;
	if (is_taken) {
		const struct queued_event taken =
			engine->events[engine->event_first];
		engine->event_first =
			(engine->event_first + 1) % EVENT_CAPACITY;
		engine->event_count--;
		if (taken.event.event_kind != GENERATION_EVENT_TOKEN) {
			engine->live.sessions[taken.session_slot].idle_since_ns =
				engine_now_ns();
			engine->live.sessions[taken.session_slot].operation = 0;
			struct slot *const slot =
				&engine->slots[taken.session_slot];
			*slot = (struct slot){
				SLOT_IDLE,
				taken.event.session_token_count,
				false,
				0,
				taken.processed_count,
				slot->decoder_from,
				slot->reach_count > taken.processed_count ?
					slot->reach_count :
					taken.processed_count,
				slot->checkpoint_count,
				slot->statistics
			};
		}
		if (engine->event_count == EVENT_CAPACITY - 1) {
			(void)pthread_cond_signal(
				&engine->wake); /* room again for an event */
		}
		*event_out = taken.event;
	}
	(void)pthread_mutex_unlock(&engine->lock);
	return is_taken;
}

/* A taken event's count off the descriptor (the semaphore eventfd's read takes one): the count never falls below
 * the queue's, so no burst of events between two takes can leave one queued with the descriptor clear. */
result_status engine_next_event(struct inference_engine *engine,
				struct generation_event *event_out,
				bool *event_available_out)
{
	uint64_t one = 0;
	if (engine == NULL || event_out == NULL ||
	    event_available_out == NULL) {
		return STATUS_INVALID_ARGUMENT;
	}
	*event_available_out = event_taken(engine, event_out);
	if (*event_available_out &&
	    read(engine->event_descriptor, &one, sizeof(one)) < 0) {
		LOG_ERROR("engine: the event descriptor could not be read");
		return STATUS_INPUT_OUTPUT_ERROR;
	}
	return OK;
}

result_status engine_event_descriptor(struct inference_engine *engine,
				      int *descriptor_out)
{
	if (engine == NULL || descriptor_out == NULL) {
		return STATUS_INVALID_ARGUMENT;
	}
	*descriptor_out = engine->event_descriptor;
	return OK;
}
