/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * scheduler.c: the engine of front_end_seam.h over one step driver. Contract: front_end_seam.h.
 *
 * The stepping thread takes the oldest command, sets its slot's sampler, reads its new tokens in steps of up to
 * most_rows rows (the head computes the last row's logits only, and none when nothing is to be generated), then steps
 * one row a token: each drawn token is a TOKEN event; the end token, the command's limit and the context's end each
 * end the command with its final event, and a cancel is looked for before every step. One mutex guards the commands,
 * the events and the slots, and one condition wakes the stepping thread (a command, room for an event, the end of
 * the engine): it holds the lock to take a command, to queue an event and to look for a cancel, never across a step.
 * The event descriptor is a semaphore eventfd: written once a queued event and read once a taken event, so it is
 * readable while an event is queued (and, for a moment after a take, once more: a wake that finds none). Fault point:
 * engine_resources.
 * Several sessions, or a driver with a pipeline (step.h), go by flights: steps of the rows of several owners, up to
 * the pipeline's depth of them in flight at once; an owner's prompt rows fly ahead, its next drawn row waits for the
 * token of its last.
 * Speculation (front_end_seam.h) goes by flights too, a round a flight: the owner's anchor and its drafter's proposals
 * fly together, every row drawn, at most one owner with proposals a step; when it lands, its tokens are its draws up
 * to the first that differs from the proposal behind it or the end token (keyed match), one at a time as steps of one
 * drawing row would give them, and the driver's history of the session is cut back to the rows those tokens commit.
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

void engine_destroy(struct inference_engine *engine)
{
	if (engine == NULL) {
		return;
	}
	(void)pthread_mutex_lock(&engine->lock);
	engine->is_ending = true;
	(void)pthread_cond_signal(&engine->wake);
	(void)pthread_mutex_unlock(&engine->lock);
	(void)pthread_join(engine->thread, NULL);
	(void)pthread_cond_destroy(&engine->wake);
	(void)pthread_mutex_destroy(&engine->lock);
	(void)close(engine->event_descriptor);
	free(engine);
}

result_status engine_describe(const struct inference_engine *engine,
			      struct engine_description_record *description_out)
{
	if (engine == NULL || description_out == NULL) {
		return STATUS_INVALID_ARGUMENT;
	}
	*description_out = engine->description;
	return OK;
}
