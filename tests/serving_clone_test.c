/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Independent slot histories over the host function model: ordinary and real-checkpoint clones, followed by
 * different child/parent continuations. The period does not divide the ring, so stale windows change the answer. */
#include <poll.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "anemone/model_layout.h"
#include "engine_internal.h"
#include "host_memory_backend.h"
#include "serving_fixture.h"
#include "step_driver.h"

#define CONTEXT_TOKENS 16384u
#define SESSION_SLOTS 2u
#define SOURCE_SLOT 0u
#define DESTINATION_SLOT 1u
#define ORDINARY_PREFIX 512u
#define CHECKPOINT_EXTRA_ROWS 34u
#define CHECKPOINT_PROMPT (MODEL_RING_SLOTS + CHECKPOINT_EXTRA_ROWS)
#define STALE_PROMPT 256u
#define GENERATED_PAST_WINDOW 5000u
#define CONTINUATION_ROWS 9u
#define CONTINUATION_TOKENS 160u
#define CONTINUATION_EVENTS (CONTINUATION_TOKENS + 1u)
#define WINDOW_ROWS (MODEL_WINDOW - 1u)
#define KEPT_LAG (MODEL_RING_SLOTS - MODEL_WINDOW)
#define TOKEN_VALUES 256u
#define TOKEN_STRIDE 37u
#define TOKEN_TURN 101u
#define TOKEN_FIRST 11u
#define CHILD_TAIL_DELTA 17u
#define PARENT_TAIL_DELTA 83u
#define STALE_TOKEN_DELTA 151u
#define SOURCE_TAG 71u
#define STALE_TAG 72u
#define CLONE_TAG 73u
#define CHILD_TAG 74u
#define PARENT_TAG 75u
#define PIPELINE_DEPTH 6u
#define LOOKUP_DEPTH 7u
#define CONE_ROWS 2u
#define WAIT_MILLISECONDS 10000
#define WAIT_ATTEMPTS 3u
#define TEST_SECONDS 60u
#define REFUSED_COPY_NUMBER 2u

static int failures;
static struct backend_operations refusing_backend;
static uint32_t attempted_copies;
static uint32_t source_tokens[CONTEXT_TOKENS];
static uint32_t child_tokens[CONTEXT_TOKENS];
static uint32_t parent_tokens[CONTEXT_TOKENS];
static uint32_t stale_tokens[CONTEXT_TOKENS];
static const uint32_t script[] = { TEST_SERVING_END_TOKEN };
static const uint32_t *const scripts[] = { script };

static void expect(bool condition, const char *message)
{
	if (!condition) {
		(void)printf("FAILED: serving clone: %s\n", message);
		failures++;
	}
}

/* Installed only while no command is active: the first real state copy runs; the checkpoint copy is refused. */
MUST_CHECK static result_status
refused_checkpoint_copy(struct device_card *card,
			const struct kernel_launch *launches, uint32_t count)
{
	if (launches != NULL && count == 1u &&
	    launches[0].kernel == BACKEND_COPY &&
	    ++attempted_copies == REFUSED_COPY_NUMBER) {
		return STATUS_INJECTED_FAULT;
	}
	return test_host_memory_backend.launch_run(card, launches, count);
}

static struct generation_event next_event(struct inference_engine *engine)
{
	struct generation_event event = { .event_kind =
						  GENERATION_EVENT_FAILED };
	int descriptor = -1;
	expect(engine_event_descriptor(engine, &descriptor) == STATUS_OK,
	       "event descriptor opens");
	for (uint32_t attempt = 0; attempt < WAIT_ATTEMPTS; attempt++) {
		bool available = false;
		if (engine_next_event(engine, &event, &available) !=
		    STATUS_OK) {
			break;
		}
		if (available) {
			return event;
		}
		struct pollfd polling = { descriptor, POLLIN, 0 };
		if (poll(&polling, 1, WAIT_MILLISECONDS) <= 0) {
			break;
		}
	}
	expect(false, "a command produces an event before the deadline");
	return event;
}

static uint32_t generation_taken(struct inference_engine *engine,
				 uint32_t limit,
				 struct generation_event *events)
{
	for (uint32_t count = 0; count <= limit; count++) {
		const struct generation_event event = next_event(engine);
		if (events != NULL && count < CONTINUATION_EVENTS) {
			events[count] = event;
		}
		if (event.event_kind != GENERATION_EVENT_TOKEN) {
			expect(event.event_kind ==
					       GENERATION_EVENT_FINISHED_AT_LIMIT &&
				       event.status == 0,
			       "generation reaches its requested token limit");
			return count + 1u;
		}
	}
	expect(false, "generation has exactly one final event");
	return 0;
}

static struct generation_command_record
generation_command(uint32_t slot, const uint32_t *tokens, uint32_t kept,
		   uint32_t prompt_count, uint32_t limit, uint32_t tag)
{
	return (struct generation_command_record){
		.new_token_ids = tokens + kept,
		.session_slot = slot,
		.kept_token_count = kept,
		.new_token_count = prompt_count - kept,
		.generated_token_limit = limit,
		.sampler_top_p = 1.0f,
		.request_tag = tag
	};
}

static uint32_t generated(struct inference_engine *engine, uint32_t slot,
			  const uint32_t *tokens, uint32_t kept,
			  uint32_t prompt_count, uint32_t limit, uint32_t tag,
			  struct generation_event *events)
{
	const struct generation_command_record command = generation_command(
		slot, tokens, kept, prompt_count, limit, tag);
	if (engine_submit(engine, &command) != STATUS_OK) {
		expect(false, "generation admission succeeds");
		return 0;
	}
	return generation_taken(engine, limit, events);
}

static struct inference_engine *opened(uint32_t depth, uint32_t draft,
				       uint32_t cone)
{
	struct inference_engine *engine = NULL;
	test_serving_set_interleaved(0);
	test_serving_set_engine(SESSION_SLOTS, depth, TEST_SERVING_MOST_ROWS,
				draft);
	test_serving_set_ring(MODEL_RING_SLOTS, MODEL_WINDOW);
	test_serving_set_function_model(true, WINDOW_ROWS);
	test_serving_set_function_end(UINT32_MAX);
	test_serving_set_device_draft(false, 0);
	test_serving_set_prefill(0, 0, 0);
	test_serving_set_cone(cone);
	expect(test_serving_open(CONTEXT_TOKENS, scripts, 1, &engine) ==
		       STATUS_OK,
	       "two independent fixture slots open");
	return engine;
}

static struct session_state state_of(struct inference_engine *engine,
				     uint32_t slot)
{
	struct session_state state = { 0 };
	expect(engine_session_state(engine, slot, &state) == STATUS_OK,
	       "idle metadata is available");
	return state;
}

static bool metadata_equal(const struct session_state *first,
			   const struct session_state *second)
{
	return first->reach_count == second->reach_count &&
	       first->decoder_from == second->decoder_from &&
	       first->checkpoint_count == second->checkpoint_count &&
	       first->generation == second->generation &&
	       first->clone.byte_count == second->clone.byte_count &&
	       first->clone.copy_ns == second->clone.copy_ns &&
	       first->clone.wait_ns == second->clone.wait_ns;
}

static struct session_clone_command
clone_command(struct inference_engine *engine, uint32_t boundary, uint32_t kind)
{
	const struct session_state source = state_of(engine, SOURCE_SLOT);
	const struct session_state destination =
		state_of(engine, DESTINATION_SLOT);
	return (struct session_clone_command){
		.source_generation = source.generation,
		.destination_generation = destination.generation,
		.source_slot = SOURCE_SLOT,
		.destination_slot = DESTINATION_SLOT,
		.token_count = boundary,
		.source_kind = kind,
		.request_tag = CLONE_TAG
	};
}

static const unsigned char *slot_bytes(uint32_t slot)
{
	return test_serving_state() +
	       (size_t)slot * TEST_SERVING_STATE_POSITIONS;
}

static void live_window(unsigned char *bytes, uint32_t slot, uint32_t processed)
{
	for (uint32_t index = 0; index < WINDOW_ROWS; index++) {
		bytes[index] =
			slot_bytes(slot)[(processed - WINDOW_ROWS + index) %
					 MODEL_RING_SLOTS];
	}
}

static void fork_tokens(uint32_t boundary)
{
	for (uint32_t index = 0; index < CONTEXT_TOKENS; index++) {
		child_tokens[index] = index < boundary ? source_tokens[index] :
							 (source_tokens[index] +
							  CHILD_TAIL_DELTA) %
								 TOKEN_VALUES;
		parent_tokens[index] =
			index < boundary ?
				source_tokens[index] :
				(source_tokens[index] + PARENT_TAIL_DELTA) %
					TOKEN_VALUES;
	}
}

static void test_continuations(uint32_t kind, uint32_t depth, uint32_t draft,
			       uint32_t cone)
{
	const bool checkpoint = kind == SESSION_CLONE_CHECKPOINT;
	const uint32_t prompt_count = checkpoint ? CHECKPOINT_PROMPT :
						   ORDINARY_PREFIX + 1u;
	const uint32_t boundary = checkpoint ? (prompt_count - 1u) /
						       MODEL_POOLED_RATIO *
						       MODEL_POOLED_RATIO :
					       ORDINARY_PREFIX;
	struct generation_event child_events[CONTINUATION_EVENTS],
		parent_events[CONTINUATION_EVENTS],
		fresh_events[CONTINUATION_EVENTS];
	unsigned char source_before[CONTEXT_TOKENS],
		child_before[CONTEXT_TOKENS];
	unsigned char child_window[WINDOW_ROWS], parent_window[WINDOW_ROWS],
		fresh_window[WINDOW_ROWS];
	fork_tokens(boundary);
	struct inference_engine *engine = opened(depth, draft, cone);
	if (engine == NULL) {
		return;
	}
	const uint32_t initial_limit = checkpoint ? GENERATED_PAST_WINDOW : 1u;
	expect(generated(engine, SOURCE_SLOT, source_tokens, 0, prompt_count,
			 initial_limit, SOURCE_TAG, NULL) == initial_limit + 1u,
	       "donor generation completes");
	(void)generated(engine, DESTINATION_SLOT, stale_tokens, 0, STALE_PROMPT,
			1, STALE_TAG, NULL);
	const struct session_state donor_before = state_of(engine, SOURCE_SLOT);
	const struct session_clone_command command =
		clone_command(engine, boundary, kind);
	memcpy(source_before, slot_bytes(SOURCE_SLOT), sizeof(source_before));
	expect(!checkpoint || (donor_before.checkpoint_count == boundary &&
			       donor_before.reach_count - boundary > KEPT_LAG),
	       "checkpoint source has a real captured window beyond ordinary reach");
	if (checkpoint) {
		struct session_clone_command ordinary = command;
		ordinary.source_kind = SESSION_CLONE_ORDINARY;
		expect(engine_session_clone(engine, &ordinary) ==
			       STATUS_INVALID_ARGUMENT,
		       "ordinary state cannot impersonate a checkpoint beyond its retained lag");
	}
	const result_status clone_status =
		engine_session_clone(engine, &command);
	expect(clone_status == STATUS_OK, "clone is admitted");
	if (clone_status != STATUS_OK) {
		test_serving_close();
		return;
	}
	const struct generation_event copied = next_event(engine);
	expect(copied.event_kind == GENERATION_EVENT_FINISHED_AT_LIMIT &&
		       copied.status == 0 && copied.request_tag == CLONE_TAG &&
		       copied.session_token_count == boundary,
	       "clone publishes one successful destination final event");
	struct generation_event extra = { 0 };
	bool available = true;
	expect(engine_next_event(engine, &extra, &available) == STATUS_OK &&
		       !available,
	       "clone creates no donor event");
	const struct session_state donor_after = state_of(engine, SOURCE_SLOT);
	const struct session_state child_after =
		state_of(engine, DESTINATION_SLOT);
	expect(metadata_equal(&donor_before, &donor_after) &&
		       memcmp(source_before, slot_bytes(SOURCE_SLOT),
			      sizeof(source_before)) == 0,
	       "clone leaves every donor state byte and metadata field unchanged");
	expect(child_after.generation == command.destination_generation + 1u &&
		       child_after.reach_count == donor_before.reach_count &&
		       child_after.decoder_from == donor_before.decoder_from &&
		       child_after.checkpoint_count ==
			       (checkpoint ? boundary : 0u),
	       "child receives actual reach/decoder and only an independently copied checkpoint");
	expect(child_after.clone.byte_count ==
		       (uint64_t)boundary + (checkpoint ? WINDOW_ROWS : 0u),
	       "clone byte metric includes ordinary state and only the requested checkpoint");
	if (checkpoint) {
		const struct generation_command_record earlier =
			generation_command(DESTINATION_SLOT, child_tokens,
					   boundary - MODEL_POOLED_RATIO,
					   boundary + CONTINUATION_ROWS, 1,
					   CHILD_TAG);
		expect(engine_submit(engine, &earlier) ==
			       STATUS_INVALID_ARGUMENT,
		       "copied checkpoint does not authorize an earlier overwritten window");
	}
	const uint32_t prompt_end = boundary + CONTINUATION_ROWS;
	expect(generated(engine, DESTINATION_SLOT, child_tokens, boundary,
			 prompt_end, CONTINUATION_TOKENS, CHILD_TAG,
			 child_events) == CONTINUATION_EVENTS,
	       "child continuation completes");
	const struct session_state donor_after_child =
		state_of(engine, SOURCE_SLOT);
	expect(metadata_equal(&donor_before, &donor_after_child) &&
		       memcmp(source_before, slot_bytes(SOURCE_SLOT),
			      sizeof(source_before)) == 0,
	       "child generation cannot mutate donor state or checkpoint metadata");
	const struct session_state child_before_parent =
		state_of(engine, DESTINATION_SLOT);
	expect(child_before_parent.clone.byte_count == 0,
	       "later generation clears prior clone metrics");
	memcpy(child_before, slot_bytes(DESTINATION_SLOT),
	       sizeof(child_before));
	const uint32_t processed = prompt_end + CONTINUATION_TOKENS - 1u;
	live_window(child_window, DESTINATION_SLOT, processed);
	expect(generated(engine, SOURCE_SLOT, parent_tokens, boundary,
			 prompt_end, CONTINUATION_TOKENS, PARENT_TAG,
			 parent_events) == CONTINUATION_EVENTS,
	       "independent parent continuation completes");
	const struct session_state child_after_parent =
		state_of(engine, DESTINATION_SLOT);
	expect(metadata_equal(&child_before_parent, &child_after_parent) &&
		       memcmp(child_before, slot_bytes(DESTINATION_SLOT),
			      sizeof(child_before)) == 0,
	       "parent generation cannot mutate the child");
	expect(memcmp(source_before, slot_bytes(SOURCE_SLOT),
		      sizeof(source_before)) != 0,
	       "parent continuation really changes its independent state");
	live_window(parent_window, SOURCE_SLOT, processed);
	test_serving_close();
	engine = opened(depth, draft, cone);
	if (engine == NULL) {
		return;
	}
	expect(generated(engine, DESTINATION_SLOT, child_tokens, 0, prompt_end,
			 CONTINUATION_TOKENS, CHILD_TAG,
			 fresh_events) == CONTINUATION_EVENTS &&
		       memcmp(child_events, fresh_events,
			      sizeof(child_events)) == 0,
	       "every child event bit equals its independent cold run");
	live_window(fresh_window, DESTINATION_SLOT, processed);
	expect(memcmp(child_window, fresh_window, sizeof(child_window)) == 0,
	       "child live window bytes equal its cold run");
	expect(generated(engine, SOURCE_SLOT, parent_tokens, 0, prompt_end,
			 CONTINUATION_TOKENS, PARENT_TAG,
			 fresh_events) == CONTINUATION_EVENTS &&
		       memcmp(parent_events, fresh_events,
			      sizeof(parent_events)) == 0,
	       "donor checkpoint still produces every event bit of its independent cold run");
	live_window(fresh_window, SOURCE_SLOT, processed);
	expect(memcmp(parent_window, fresh_window, sizeof(parent_window)) == 0,
	       "parent live window bytes equal its cold run");
	test_serving_close();
}

static void test_admission(void)
{
	struct inference_engine *const engine = opened(1, 0, 0);
	if (engine == NULL) {
		return;
	}
	(void)generated(engine, SOURCE_SLOT, source_tokens, 0,
			ORDINARY_PREFIX + 1u, 1, SOURCE_TAG, NULL);
	struct session_clone_command command =
		clone_command(engine, ORDINARY_PREFIX, SESSION_CLONE_ORDINARY);
	expect(engine_session_clone(NULL, &command) ==
			       STATUS_INVALID_ARGUMENT &&
		       engine_session_clone(engine, NULL) ==
			       STATUS_INVALID_ARGUMENT,
	       "missing engine or clone command is refused");
	const struct session_clone_command valid = command;
	command.destination_slot = command.source_slot;
	expect(engine_session_clone(engine, &command) ==
		       STATUS_INVALID_ARGUMENT,
	       "a clone cannot alias its donor");
	command = valid;
	command.source_slot = SESSION_SLOTS;
	expect(engine_session_clone(engine, &command) ==
		       STATUS_INVALID_ARGUMENT,
	       "source slot bounds are checked");
	command = valid;
	command.destination_slot = SESSION_SLOTS;
	expect(engine_session_clone(engine, &command) ==
		       STATUS_INVALID_ARGUMENT,
	       "destination slot bounds are checked");
	command = valid;
	command.token_count = 0;
	expect(engine_session_clone(engine, &command) ==
		       STATUS_INVALID_ARGUMENT,
	       "empty copy is refused");
	command.token_count = 1;
	expect(engine_session_clone(engine, &command) ==
		       STATUS_INVALID_ARGUMENT,
	       "an unclosed granule is refused");
	command.token_count = CONTEXT_TOKENS + MODEL_POOLED_RATIO;
	expect(engine_session_clone(engine, &command) ==
		       STATUS_INVALID_ARGUMENT,
	       "context bounds are checked");
	command = valid;
	command.token_count = ORDINARY_PREFIX + MODEL_POOLED_RATIO;
	expect(engine_session_clone(engine, &command) ==
		       STATUS_INVALID_ARGUMENT,
	       "the sampled but unprocessed last token is not reusable state");
	command = valid;
	command.source_kind = SESSION_CLONE_CHECKPOINT + 1u;
	expect(engine_session_clone(engine, &command) ==
		       STATUS_INVALID_ARGUMENT,
	       "unknown clone source kind is refused");
	command = valid;
	command.reserved_must_be_zero = 1;
	expect(engine_session_clone(engine, &command) ==
		       STATUS_INVALID_ARGUMENT,
	       "reserved field stays reserved");
	command = valid;
	command.source_kind = SESSION_CLONE_CHECKPOINT;
	command.token_count -= MODEL_POOLED_RATIO;
	expect(engine_session_clone(engine, &command) ==
		       STATUS_INVALID_ARGUMENT,
	       "checkpoint existence is checked at the exact requested boundary");
	const struct generation_command_record writer =
		generation_command(SOURCE_SLOT, source_tokens, 0,
				   ORDINARY_PREFIX + 1u, 1, SOURCE_TAG);
	expect(engine_submit(engine, &writer) == STATUS_OK,
	       "donor replacement is admitted");
	expect(engine_session_clone(engine, &valid) == STATUS_BUSY_VALUE,
	       "a donor is busy until its final event is taken");
	(void)generation_taken(engine, 1, NULL);
	expect(engine_session_clone(engine, &valid) == STATUS_INVALID_ARGUMENT,
	       "same-token donor replacement invalidates a stale source generation");
	command =
		clone_command(engine, ORDINARY_PREFIX, SESSION_CLONE_ORDINARY);
	const struct generation_command_record destination_writer =
		generation_command(DESTINATION_SLOT, stale_tokens, 0,
				   STALE_PROMPT, 1, STALE_TAG);
	expect(engine_submit(engine, &destination_writer) == STATUS_OK,
	       "destination replacement is admitted");
	expect(engine_session_clone(engine, &command) == STATUS_BUSY_VALUE,
	       "a destination is busy until its final event is taken");
	(void)generation_taken(engine, 1, NULL);
	expect(engine_session_clone(engine, &command) ==
		       STATUS_INVALID_ARGUMENT,
	       "destination replacement invalidates a stale destination generation");
	test_serving_close();
}

static void test_failed_copy(void)
{
	struct inference_engine *const engine = opened(1, 0, 0);
	if (engine == NULL) {
		return;
	}
	(void)generated(engine, SOURCE_SLOT, source_tokens, 0,
			ORDINARY_PREFIX + 1u, 1, SOURCE_TAG, NULL);
	(void)generated(engine, DESTINATION_SLOT, stale_tokens, 0, STALE_PROMPT,
			1, STALE_TAG, NULL);
	const struct session_state source_before =
		state_of(engine, SOURCE_SLOT);
	const struct session_clone_command command =
		clone_command(engine, ORDINARY_PREFIX, SESSION_CLONE_ORDINARY);
	unsigned char donor_bytes[CONTEXT_TOKENS];
	memcpy(donor_bytes, slot_bytes(SOURCE_SLOT), sizeof(donor_bytes));
	test_host_memory_backend_fail_launch(0, 1);
	const result_status clone_status =
		engine_session_clone(engine, &command);
	expect(clone_status == STATUS_OK,
	       "failure occurs in a real queued copy, after admission");
	if (clone_status != STATUS_OK) {
		test_serving_close();
		return;
	}
	const struct generation_event event = next_event(engine);
	expect(event.event_kind == GENERATION_EVENT_FAILED &&
		       event.status == STATUS_DEVICE_ERROR &&
		       event.session_token_count == 0,
	       "failed copy publishes no usable child history");
	const struct session_state source_after = state_of(engine, SOURCE_SLOT);
	const struct session_state destination =
		state_of(engine, DESTINATION_SLOT);
	expect(metadata_equal(&source_before, &source_after) &&
		       memcmp(donor_bytes, slot_bytes(SOURCE_SLOT),
			      sizeof(donor_bytes)) == 0,
	       "failed copy still leaves the donor unchanged and releases its lease");
	expect(destination.reach_count == 0 && destination.decoder_from == 0 &&
		       destination.checkpoint_count == 0 &&
		       destination.clone.byte_count == 0 &&
		       destination.generation ==
			       command.destination_generation + 1u,
	       "failed destination retains no reach, decoder or checkpoint certificate");
	const struct generation_command_record cold = generation_command(
		DESTINATION_SLOT, child_tokens, 0,
		ORDINARY_PREFIX + CONTINUATION_ROWS, 1, CHILD_TAG);
	expect(engine_submit(engine, &cold) == STATUS_DEVICE_ERROR,
	       "device poison is not disguised as a cold fallback");
	test_serving_close();
}

static void test_partial_copy_cold_restart(void)
{
	struct generation_event restarted_events[CONTINUATION_EVENTS],
		fresh_events[CONTINUATION_EVENTS];
	unsigned char donor_bytes[CONTEXT_TOKENS], dirty_bytes[CONTEXT_TOKENS];
	unsigned char restarted_window[WINDOW_ROWS], fresh_window[WINDOW_ROWS];
	struct inference_engine *engine = opened(1, 0, 0);
	if (engine == NULL) {
		return;
	}
	fork_tokens(ORDINARY_PREFIX);
	(void)generated(engine, SOURCE_SLOT, source_tokens, 0,
			ORDINARY_PREFIX + 1u, 1, SOURCE_TAG, NULL);
	(void)generated(engine, DESTINATION_SLOT, stale_tokens, 0,
			CHECKPOINT_PROMPT, 1, STALE_TAG, NULL);
	const struct session_state donor_before = state_of(engine, SOURCE_SLOT);
	const struct session_clone_command command = clone_command(
		engine, ORDINARY_PREFIX, SESSION_CLONE_CHECKPOINT);
	memcpy(donor_bytes, slot_bytes(SOURCE_SLOT), sizeof(donor_bytes));
	memcpy(dirty_bytes, slot_bytes(DESTINATION_SLOT), sizeof(dirty_bytes));
	expect(donor_before.checkpoint_count == ORDINARY_PREFIX &&
		       memcmp(donor_bytes, dirty_bytes, ORDINARY_PREFIX) != 0,
	       "partial-copy fixture has a real checkpoint and a different dirty destination");
	/* Idle slots and depth one leave no card operation in flight. The next admission publishes this test table. */
	(void)pthread_mutex_lock(&engine->lock);
	refusing_backend = *engine->driver->backend;
	refusing_backend.launch_run = refused_checkpoint_copy;
	attempted_copies = 0;
	engine->driver->backend = &refusing_backend;
	(void)pthread_mutex_unlock(&engine->lock);
	const result_status clone_status =
		engine_session_clone(engine, &command);
	expect(clone_status == STATUS_OK,
	       "partial copy is admitted before its checkpoint transfer fails");
	if (clone_status != STATUS_OK) {
		test_serving_close();
		return;
	}
	const struct generation_event event = next_event(engine);
	expect(event.event_kind == GENERATION_EVENT_FAILED &&
		       event.status == STATUS_INJECTED_FAULT &&
		       event.request_tag == CLONE_TAG &&
		       event.session_token_count == 0 &&
		       attempted_copies == REFUSED_COPY_NUMBER,
	       "nonhardware failure occurs after one complete state copy and advertises no child history");
	(void)pthread_mutex_lock(&engine->lock);
	engine->driver->backend = &test_host_memory_backend;
	(void)pthread_mutex_unlock(&engine->lock);
	const struct session_state donor_after = state_of(engine, SOURCE_SLOT);
	const struct session_state destination =
		state_of(engine, DESTINATION_SLOT);
	expect(metadata_equal(&donor_before, &donor_after) &&
		       memcmp(donor_bytes, slot_bytes(SOURCE_SLOT),
			      sizeof(donor_bytes)) == 0,
	       "partial copy leaves donor bytes and metadata unchanged and releases its lease");
	expect(memcmp(donor_bytes, slot_bytes(DESTINATION_SLOT),
		      ORDINARY_PREFIX) == 0 &&
		       memcmp(dirty_bytes + ORDINARY_PREFIX,
			      slot_bytes(DESTINATION_SLOT) + ORDINARY_PREFIX,
			      sizeof(dirty_bytes) - ORDINARY_PREFIX) == 0,
	       "destination really contains the copied donor prefix and the remaining dirty state");
	expect(destination.reach_count == 0 && destination.decoder_from == 0 &&
		       destination.checkpoint_count == 0 &&
		       destination.clone.byte_count == 0 &&
		       destination.generation ==
			       command.destination_generation + 1u,
	       "partial-copy failure discards every destination reuse certificate");
	const uint32_t prompt_count = ORDINARY_PREFIX + CONTINUATION_ROWS;
	const struct generation_command_record kept = generation_command(
		DESTINATION_SLOT, child_tokens, ORDINARY_PREFIX, prompt_count,
		CONTINUATION_TOKENS, CHILD_TAG);
	expect(engine_submit(engine, &kept) == STATUS_INVALID_ARGUMENT,
	       "partially copied physical bytes cannot authorize a kept-prefix restart");
	expect(generated(engine, DESTINATION_SLOT, child_tokens, 0,
			 prompt_count, CONTINUATION_TOKENS, CHILD_TAG,
			 restarted_events) == CONTINUATION_EVENTS,
	       "discarded destination accepts an actual cold row-zero restart");
	const uint32_t processed = prompt_count + CONTINUATION_TOKENS - 1u;
	live_window(restarted_window, DESTINATION_SLOT, processed);
	const struct session_state donor_after_restart =
		state_of(engine, SOURCE_SLOT);
	expect(metadata_equal(&donor_before, &donor_after_restart) &&
		       memcmp(donor_bytes, slot_bytes(SOURCE_SLOT),
			      sizeof(donor_bytes)) == 0,
	       "cold restart also leaves the independent donor unchanged");
	test_serving_close();
	engine = opened(1, 0, 0);
	if (engine == NULL) {
		return;
	}
	expect(generated(engine, DESTINATION_SLOT, child_tokens, 0,
			 prompt_count, CONTINUATION_TOKENS, CHILD_TAG,
			 fresh_events) == CONTINUATION_EVENTS &&
		       memcmp(restarted_events, fresh_events,
			      sizeof(restarted_events)) == 0,
	       "every restarted event bit equals its fresh cold run after a real partial copy");
	live_window(fresh_window, DESTINATION_SLOT, processed);
	expect(memcmp(restarted_window, fresh_window,
		      sizeof(restarted_window)) == 0,
	       "restarted live window bytes equal the fresh cold run");
	test_serving_close();
}

int main(void)
{
	(void)alarm(TEST_SECONDS);
	for (uint32_t index = 0; index < CONTEXT_TOKENS; index++) {
		source_tokens[index] =
			(index * TOKEN_STRIDE +
			 index / MODEL_RING_SLOTS * TOKEN_TURN + TOKEN_FIRST) %
			TOKEN_VALUES;
		stale_tokens[index] =
			(source_tokens[index] + STALE_TOKEN_DELTA) %
			TOKEN_VALUES;
	}
	test_admission();
	test_continuations(SESSION_CLONE_ORDINARY, 1, 0, 0);
	test_continuations(SESSION_CLONE_CHECKPOINT, 1, 0, 0);
	test_continuations(SESSION_CLONE_CHECKPOINT, PIPELINE_DEPTH,
			   LOOKUP_DEPTH, CONE_ROWS);
	test_failed_copy();
	test_partial_copy_cold_restart();
	test_serving_set_cone(0);
	test_serving_set_engine(1, 1, TEST_SERVING_ROWS, 0);
	test_serving_set_ring(TEST_SERVING_STATE_POSITIONS,
			      TEST_SERVING_WINDOW);
	test_serving_set_function_model(false, 0);
	(void)printf("serving clone: %s\n",
		     failures == 0 ? "passed" : "FAILED");
	return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
