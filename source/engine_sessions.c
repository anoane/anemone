/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * engine_sessions.c: the engine's session commands (front_end_seam.h): a slot's snapshot (step.h) to or from the front
 * end's file, queued as a slot's command and run by the stepping thread once no step is in flight.
 */
#include <errno.h>
#include <unistd.h>

#include "engine_internal.h"

#define OK STATUS_OK

struct session_transfer {
	int descriptor;
	uint64_t bytes;
};

/* The front end's file of a session command, from or to its offset: whole pieces, an interrupted call resumed. */
static bool file_write(void *context, unsigned char *bytes, uint64_t byte_count)
{
	struct session_transfer *const transfer = context;
	const int descriptor = transfer->descriptor;
	while (byte_count != 0) {
		const ssize_t written = write(descriptor, bytes, byte_count);
		if (written <= 0 && !(written < 0 && errno == EINTR)) {
			return false;
		}
		bytes += written > 0 ? written : 0;
		byte_count -= written > 0 ? (uint64_t)written : 0;
		transfer->bytes += written > 0 ? (uint64_t)written : 0;
	}
	return true;
}

static bool file_read(void *context, unsigned char *bytes, uint64_t byte_count)
{
	struct session_transfer *const transfer = context;
	const int descriptor = transfer->descriptor;
	while (byte_count != 0) {
		const ssize_t got = read(descriptor, bytes, byte_count);
		if (got <= 0 && !(got < 0 && errno == EINTR)) {
			return false;
		}
		bytes += got > 0 ? got : 0;
		byte_count -= got > 0 ? (uint64_t)got : 0;
		transfer->bytes += got > 0 ? (uint64_t)got : 0;
	}
	return true;
}

/* Both generations are reserved by admission. The worker alone touches driver state; all flights have landed. */
static void clone_advanced(struct inference_engine *engine,
			   struct command_cursor *cursor)
{
	const struct session_clone_command *const command = &cursor->clone;
	const uint32_t source = command->source_slot,
		       destination = command->destination_slot;
	const uint64_t began = engine_now_ns();
	(void)pthread_mutex_lock(&engine->lock);
	const struct slot donor = engine->slots[source];
	bool cancelled = engine->slots[destination].is_cancelled ||
			 engine->is_ending;
	result_status status = engine->failure != OK ? engine->failure :
						       cursor->status;
	if (status == OK &&
	    (!engine->slot_read_leased[source] ||
	     engine->slot_generations[source] != command->source_generation ||
	     engine->slot_generations[destination] !=
		     command->destination_generation + 1u)) {
		status = STATUS_INVALID_ARGUMENT;
	}
	(void)pthread_mutex_unlock(&engine->lock);
	uint64_t bytes = 0;
	if (status == OK && !cancelled) {
		status = step_session_clone(engine->driver, source, destination,
					    command->token_count,
					    command->source_kind ==
						    SESSION_CLONE_CHECKPOINT,
					    &bytes);
	}
	const uint64_t ended = engine_now_ns();
	(void)pthread_mutex_lock(&engine->lock);
	cancelled = cancelled || engine->slots[destination].is_cancelled ||
		    engine->is_ending;
	if (status != OK || cancelled) {
		(void)pthread_mutex_unlock(&engine->lock);
		const result_status discarded =
			step_session_discard(engine->driver, destination);
		if (status == OK || discarded == STATUS_DEVICE_ERROR ||
		    discarded == STATUS_TIMED_OUT) {
			status = discarded;
		}
		(void)pthread_mutex_lock(&engine->lock);
	}
	const bool committed = status == OK && !cancelled;
	struct slot *const target = &engine->slots[destination];
	target->decoder_from = committed ? donor.decoder_from : 0;
	target->reach_count = committed ? donor.reach_count : 0;
	target->checkpoint_count =
		committed && command->source_kind == SESSION_CLONE_CHECKPOINT ?
			command->token_count :
			0u;
	target->statistics = (struct draft_statistics){ 0 };
	/* This is the cancellation linearization point. A later cancel observes a completed command. */
	target->phase = SLOT_FINAL_QUEUED;
	engine->clone_metrics[destination] = (struct session_clone_metrics){
		.byte_count = committed ? bytes : 0,
		.copy_ns = ended >= began ? ended - began : 0,
		.wait_ns = began >= cursor->clone_submitted_ns ?
				   began - cursor->clone_submitted_ns :
				   0
	};
	if (status == STATUS_DEVICE_ERROR || status == STATUS_TIMED_OUT) {
		engine->failure = engine->failure == OK ? status :
							  engine->failure;
	}
	engine->slot_read_leased[source] = false;
	(void)pthread_mutex_unlock(&engine->lock);
	cursor->is_ready = false;
	const uint32_t held = committed ? command->token_count : 0u;
	(void)engine_event_queued(
		engine, destination,
		(struct generation_event){
			.request_tag = command->request_tag,
			.event_kind =
				status != OK ? GENERATION_EVENT_FAILED :
				cancelled    ? GENERATION_EVENT_CANCELLED :
					       GENERATION_EVENT_FINISHED_AT_LIMIT,
			.session_token_count = held,
			.status = (uint32_t)status },
		held);
}

/* A session command (front_end_seam.h) once no step is in flight, the oldest landing first; its final event holds
 * what the session holds after it. */
void engine_session_advanced(struct inference_engine *engine,
			     struct command_cursor *cursor)
{
	if (engine->flight_count != 0) {
		engine_flight_landed(engine);
		return;
	}
	if (cursor->session.operation == SESSION_CLONE) {
		clone_advanced(engine, cursor);
		return;
	}
	const struct session_command *const command = &cursor->session;
	const bool is_save = command->operation == SESSION_SAVE;
	struct session_transfer transfer = { .descriptor =
						     command->descriptor };
	const uint64_t began = engine_now_ns();
	result_status status = cursor->status;
	status = status != OK ? status :
		 is_save ? step_session_save(engine->driver,
					     command->session_slot,
					     command->token_count, file_write,
					     &transfer) :
			   step_session_load(engine->driver,
					     command->session_slot,
					     command->token_ids,
					     command->token_count, file_read,
					     &transfer);
	const uint64_t ended = engine_now_ns();
	if (status == STATUS_DEVICE_ERROR || status == STATUS_TIMED_OUT) {
		engine_driver_failed(engine, status);
	}
	(void)pthread_mutex_lock(&engine->lock);
	struct engine_session_stats *const metrics =
		&engine->live.sessions[command->session_slot];
	const uint64_t elapsed = began != 0 && ended >= began ? ended - began :
								0;
	if (is_save) {
		metrics->snapshots_saved += status == OK ? 1u : 0u;
		metrics->snapshot_save_bytes += transfer.bytes;
		metrics->snapshot_save_ns += elapsed;
		engine->live.snapshots_saved += status == OK ? 1u : 0u;
		engine->live.snapshot_save_bytes += transfer.bytes;
		engine->live.snapshot_save_ns +=
			began != 0 && ended >= began ? ended - began : 0;
	} else {
		metrics->snapshots_loaded += status == OK ? 1u : 0u;
		metrics->snapshot_load_bytes += transfer.bytes;
		metrics->snapshot_load_ns += elapsed;
		engine->live.snapshots_loaded += status == OK ? 1u : 0u;
		engine->live.snapshot_load_bytes += transfer.bytes;
		engine->live.snapshot_load_ns +=
			began != 0 && ended >= began ? ended - began : 0;
	}
	const struct slot kept = engine->slots[command->session_slot];
	if (!is_save) {
		engine->slots[command->session_slot].decoder_from =
			status == OK ? command->decoder_from : 0;
		engine->slots[command->session_slot].reach_count =
			status == OK ? command->reach_count : 0;
		engine->slots[command->session_slot].checkpoint_count = 0;
	}
	(void)pthread_mutex_unlock(&engine->lock);
	const uint32_t held = is_save	   ? kept.token_count :
			      status == OK ? command->token_count :
					     0;
	cursor->is_ready = false;
	(void)engine_event_queued(
		engine, command->session_slot,
		(struct generation_event){
			command->request_tag,
			status == OK ? GENERATION_EVENT_FINISHED_AT_LIMIT :
				       GENERATION_EVENT_FAILED,
			0, held, (uint32_t)status, 0.0f },
		is_save ? kept.processed_count : held);
}

result_status engine_session(struct inference_engine *engine,
			     const struct session_command *command)
{
	const uint32_t slot_count =
		engine != NULL ? engine->description.session_slot_count : 0;
	const bool is_save = command != NULL &&
			     command->operation == SESSION_SAVE;
	if (engine == NULL || command == NULL ||
	    command->session_slot >= slot_count ||
	    (!is_save && command->operation != SESSION_LOAD) ||
	    command->descriptor < 0 || command->reserved_must_be_zero != 0 ||
	    (is_save ?
		     command->token_count == 0 || command->reach_count != 0 ||
			     command->decoder_from != 0 :
		     command->token_ids == NULL ||
			     command->token_count >
				     engine->description.context_token_limit ||
			     command->token_count % SERVING_KEPT_GRANULE != 0 ||
			     command->reach_count < command->token_count ||
			     command->reach_count >
				     engine->description.context_token_limit ||
			     command->decoder_from >
				     engine->description.context_token_limit)) {
		return STATUS_INVALID_ARGUMENT;
	}
	(void)pthread_mutex_lock(&engine->lock);
	struct slot *const slot = &engine->slots[command->session_slot];
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
		duplicate_tag || (is_save && command->token_count >
						     slot->processed_count) ?
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
		engine->live.sessions[command->session_slot].operation =
			command->operation;
		engine->live.sessions[command->session_slot].idle_since_ns = 0;
		struct queued_command *const queued =
			&engine->commands[(engine->command_first +
					   engine->command_count) %
					  slot_count];
		*queued = (struct queued_command){ .session = *command };
		engine->command_count++;
		*slot = (struct slot){ .phase = SLOT_COMMANDED,
				       .token_count = slot->token_count,
				       .request_tag = command->request_tag,
				       .processed_count = slot->processed_count,
				       .decoder_from = slot->decoder_from,
				       .reach_count = slot->reach_count,
				       .checkpoint_count =
					       slot->checkpoint_count,
				       .statistics = slot->statistics };
		(void)pthread_cond_signal(&engine->wake);
	}
	(void)pthread_mutex_unlock(&engine->lock);
	return status;
}

result_status engine_session_clone(struct inference_engine *engine,
				   const struct session_clone_command *command)
{
	if (engine == NULL || command == NULL ||
	    command->source_slot >= engine->description.session_slot_count ||
	    command->destination_slot >=
		    engine->description.session_slot_count ||
	    command->source_slot == command->destination_slot ||
	    command->token_count == 0 ||
	    command->token_count > engine->description.context_token_limit ||
	    command->token_count % SERVING_KEPT_GRANULE != 0 ||
	    command->source_kind > SESSION_CLONE_CHECKPOINT ||
	    command->reserved_must_be_zero != 0) {
		return STATUS_INVALID_ARGUMENT;
	}
	(void)pthread_mutex_lock(&engine->lock);
	const uint32_t source = command->source_slot,
		       destination = command->destination_slot;
	const struct slot *const donor = &engine->slots[source];
	struct slot *const target = &engine->slots[destination];
	bool duplicate = false;
	for (uint32_t index = 0; index < engine->description.session_slot_count;
	     index++) {
		duplicate = duplicate ||
			    (engine->slots[index].phase != SLOT_IDLE &&
			     engine->slots[index].request_tag ==
				     command->request_tag);
	}
	const bool ordinary = command->source_kind == SESSION_CLONE_ORDINARY;
	const result_status status =
		engine->failure != OK ? engine->failure :
		engine->is_ending || donor->phase != SLOT_IDLE ||
				target->phase != SLOT_IDLE ||
				engine->slot_read_leased[source] ||
				engine->slot_read_leased[destination] ||
				engine->command_count ==
					engine->description.session_slot_count ?
					STATUS_BUSY_VALUE :
		command->source_generation !=
					engine->slot_generations[source] ||
				command->destination_generation !=
					engine->slot_generations[destination] ||
				duplicate ||
				command->token_count > donor->processed_count ||
				command->token_count > donor->token_count ||
				donor->reach_count < command->token_count ||
				(ordinary ?
					 donor->reach_count -
							 command->token_count >
						 SERVING_KEPT_LAG :
					 donor->checkpoint_count !=
						 command->token_count) ?
					STATUS_INVALID_ARGUMENT :
		engine->slot_generations[destination] == UINT64_MAX ?
					STATUS_UNSUPPORTED :
					OK;
	if (status == OK) {
		struct queued_command *const queued =
			&engine->commands[(engine->command_first +
					   engine->command_count) %
					  engine->description.session_slot_count];
		*queued = (struct queued_command){
			.session = { .session_slot = destination,
				     .operation = SESSION_CLONE,
				     .request_tag = command->request_tag,
				     .descriptor = -1 },
			.clone = *command,
			.clone_submitted_ns = engine_now_ns()
		};
		engine->command_count++;
		engine->slot_generations[destination]++;
		engine->slot_read_leased[source] = true;
		engine->clone_metrics[destination] =
			(struct session_clone_metrics){ 0 };
		target->phase = SLOT_COMMANDED;
		target->request_tag = command->request_tag;
		target->is_cancelled = false;
		engine->live.sessions[destination].operation = SESSION_CLONE;
		engine->live.sessions[destination].idle_since_ns = 0;
		(void)pthread_cond_signal(&engine->wake);
	}
	(void)pthread_mutex_unlock(&engine->lock);
	return status;
}

result_status engine_session_state(struct inference_engine *engine,
				   uint32_t session_slot,
				   struct session_state *state_out)
{
	if (engine == NULL || state_out == NULL ||
	    session_slot >= engine->description.session_slot_count) {
		return STATUS_INVALID_ARGUMENT;
	}
	(void)pthread_mutex_lock(&engine->lock);
	const struct slot *const slot = &engine->slots[session_slot];
	const result_status status =
		slot->phase == SLOT_IDLE &&
				!engine->slot_read_leased[session_slot] ?
			OK :
			STATUS_BUSY_VALUE;
	if (status == OK) {
		*state_out = (struct session_state){
			.reach_count = slot->reach_count,
			.decoder_from = slot->decoder_from,
			.checkpoint_count = slot->checkpoint_count,
			.generation = engine->slot_generations[session_slot],
			.clone = engine->clone_metrics[session_slot]
		};
	}
	(void)pthread_mutex_unlock(&engine->lock);
	return status;
}
