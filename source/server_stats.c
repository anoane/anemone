/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Product statistics, owned by the HTTP poller. The output worker owns only a fixed heap mailbox and its copies.
 * A slow stderr blocks that worker alone; a full mailbox drops refreshes and shutdown never joins a writer.
 * Request identity contains no headers, credentials, messages or tool definitions.
 */
#include "server_internal.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <time.h>
#include <unistd.h>

#define OK STATUS_OK
#define MILLISECONDS_PER_SECOND 1000.0
#define NANOSECONDS_PER_SECOND 1e9
#define OUTPUT_WAKE_SECONDS 1

struct live_output {
	pthread_mutex_t lock;
	pthread_cond_t wake;
	bool stopped, pending, is_json, is_table;
	int descriptor;
	struct live_snapshot snapshot, rendering;
};

static void text_copy(char *output, size_t capacity, const char *text)
{
	(void)snprintf(output, capacity, "%s", text != NULL ? text : "");
}

static uint32_t user_index(const struct live_snapshot *snapshot,
			   const char *user)
{
	for (uint32_t index = 0; index < snapshot->user_count; index++) {
		if (strcmp(snapshot->users[index].user, user) == 0) {
			return index;
		}
	}
	return snapshot->user_count;
}

bool server_stats_users_bind(struct http_server *server,
			     const struct server_keys_record *keys)
{
	const uint32_t count = keys->count != 0 ? keys->count : 1u;
	uint32_t added = 0;
	for (uint32_t index = 0; index < count; index++) {
		const char *const user = keys->count != 0 ? keys->entries[index].user : "-";
		added += user_index(&server->live, user) == server->live.user_count ? 1u : 0u;
	}
	if (added > SERVER_MOST_USERS - server->live.user_count) {
		return false;
	}
	for (uint32_t index = 0; index < count; index++) {
		const char *const user = keys->count != 0 ? keys->entries[index].user : "-";
		if (user_index(&server->live, user) == server->live.user_count) {
			text_copy(server->live.users[server->live.user_count++]
					  .user,
				  SERVER_USER_BYTES, user);
		}
	}
	return true;
}

static void user_tokens_added(struct live_user *user,
			      const struct live_request_record *request)
{
	user->prompt_tokens += request->engine.prompt_tokens;
	user->cached_tokens += request->reused_tokens;
	user->generated_tokens += request->engine.generated_tokens;
}

static void *output_worker(void *argument)
{
	struct live_output *const output = argument;
	struct live_snapshot *const snapshot = &output->rendering;
	struct chat_buffer json = { 0 }, text = { 0 };
	bool stopped = false;
	sigset_t blocked;
	if (sigemptyset(&blocked) != 0 || sigaddset(&blocked, SIGPIPE) != 0 ||
	    pthread_sigmask(SIG_BLOCK, &blocked, NULL) != 0) {
		return NULL; /* the owner still has a valid mailbox; only printing is unavailable */
	}
	while (!stopped && pthread_mutex_lock(&output->lock) == 0) {
		while (!output->pending && !output->stopped) {
			struct timespec deadline = { 0 };
			if (clock_gettime(CLOCK_REALTIME, &deadline) != 0) {
				(void)pthread_mutex_unlock(&output->lock);
				return NULL; /* retain the mailbox: the poller may still own it */
			}
			deadline.tv_sec += OUTPUT_WAKE_SECONDS;
			const int waited = pthread_cond_timedwait(
				&output->wake, &output->lock, &deadline);
			if (waited != 0 && waited != ETIMEDOUT) {
				(void)pthread_mutex_unlock(&output->lock);
				return NULL;
			}
		}
		stopped = output->stopped;
		if (!stopped) {
			*snapshot = output->snapshot;
			output->pending = false;
		}
		if (pthread_mutex_unlock(&output->lock) != 0 || stopped) {
			break;
		}
		json.count = json.sent = text.count = text.sent = 0;
		json.is_short = text.is_short = false;
		result_status status = live_stats_json_entry(snapshot, &json);
		struct chat_buffer *rendered = &json;
		if (status == OK && output->is_json) {
			chat_buffer_write(&json, "\n", 1);
		}
		if (status == OK && !output->is_json) {
			struct winsize size = { 0 };
			const bool is_tty = isatty(output->descriptor) == 1;
			const uint32_t columns =
				is_tty &&
						ioctl(output->descriptor,
						      TIOCGWINSZ, &size) == 0 &&
						size.ws_col >=
							LIVE_STATS_LEAST_COLUMNS &&
						size.ws_col <=
							LIVE_STATS_MOST_COLUMNS ?
					size.ws_col :
					LIVE_STATS_DEFAULT_COLUMNS;
			status = (output->is_table ? live_stats_table_entry :
						     live_stats_text)(
				json.bytes, json.count, columns, is_tty, &text);
			rendered = &text;
		}
		if (status == OK) {
			while (rendered->sent < rendered->count) {
				const ssize_t sent =
					write(output->descriptor,
					      rendered->bytes + rendered->sent,
					      rendered->count - rendered->sent);
				if (sent < 0 && errno == EINTR) {
					continue;
				}
				if (sent <= 0) {
					break;
				}
				rendered->sent += (size_t)sent;
			}
		}
	}
	chat_buffer_free(&json);
	chat_buffer_free(&text);
	if (!stopped) {
		return NULL; /* a lock failure cannot invalidate the poller's mailbox */
	}
	(void)close(output->descriptor);
	(void)pthread_cond_destroy(&output->wake);
	(void)pthread_mutex_destroy(&output->lock);
	free(output);
	return NULL;
}

void server_stats_open(struct http_server *server, double now)
{
	server->live.started_seconds = now;
	server->live_sample_seconds =
		now + (double)LIVE_SAMPLE_PERIOD_NS / NANOSECONDS_PER_SECOND;
	server->live_next_seconds =
		now + (double)server->live_stats_interval_ms /
			      MILLISECONDS_PER_SECOND;
	text_copy(server->live.model_alias, sizeof(server->live.model_alias),
		  server->model_alias);
	text_copy(server->live.card_order, sizeof(server->live.card_order),
		  server->stats_card_order);
	text_copy(server->live.placement, sizeof(server->live.placement),
		  server->stats_placement);
	if (live_stats_build_id(server->live.build_id) != OK) {
		text_copy(server->live.build_id, sizeof(server->live.build_id),
			  "unavailable");
	}
	if (engine_stats_counters(server->engine, &server->live.engine) != OK ||
	    live_sampler_take(&server->live_sampler, &server->live.engine) !=
		    OK) {
		return;
	}
	server->live_sample_seconds = (double)server->live_sampler.next_sample_ns / NANOSECONDS_PER_SECOND;
	if (server->live_stats_interval_ms == 0) {
		return;
	}
	struct live_output *const output = calloc(1, sizeof(*output));
	if (output == NULL) {
		return;
	}
	if (pthread_mutex_init(&output->lock, NULL) != 0) {
		free(output);
		return;
	}
	if (pthread_cond_init(&output->wake, NULL) != 0) {
		(void)pthread_mutex_destroy(&output->lock);
		free(output);
		return;
	}
	pthread_attr_t attributes;
	if (pthread_attr_init(&attributes) != 0) {
		(void)pthread_cond_destroy(&output->wake);
		(void)pthread_mutex_destroy(&output->lock);
		free(output);
		return;
	}
	output->is_json = server->live_stats_json;
	output->is_table = server->live_stats_table;
	output->descriptor = fcntl(STDERR_FILENO, F_DUPFD_CLOEXEC, 0);
	pthread_t thread;
	const int created =
		output->descriptor >= 0 &&
				pthread_attr_setdetachstate(
					&attributes, PTHREAD_CREATE_DETACHED) ==
					0 ?
			pthread_create(&thread, &attributes, output_worker,
				       output) :
			EINVAL;
	(void)pthread_attr_destroy(&attributes);
	if (created == 0) {
		server->live_output = output;
	} else {
		if (output->descriptor >= 0) {
			(void)close(output->descriptor);
		}
		(void)pthread_cond_destroy(&output->wake);
		(void)pthread_mutex_destroy(&output->lock);
		free(output);
	}
}

void server_stats_close(struct http_server *server)
{
	struct live_output *const output = server->live_output;
	server->live_output = NULL;
	if (output != NULL && pthread_mutex_lock(&output->lock) == 0) {
		output->stopped = true;
		(void)pthread_cond_signal(&output->wake);
		(void)pthread_mutex_unlock(&output->lock);
	}
}

static void phase_updated(struct live_request_record *request)
{
	static const char *const phases[] = { "idle", "queued", "prompt",
					      "decode", "finishing" };
	const struct engine_slot_stats *const engine = &request->engine;
	if (engine->request_tag != request->request_tag ||
	    request->finished_seconds != 0) {
		return;
	}
	if (engine->phase < sizeof(phases) / sizeof(phases[0])) {
		text_copy(request->phase, sizeof(request->phase),
			  phases[engine->phase]);
		request->phase_seconds =
			(double)engine->phase_since_ns / NANOSECONDS_PER_SECOND;
	}
}

MUST_CHECK static result_status snapshot_taken(struct http_server *server,
					       double now,
					       struct live_snapshot *snapshot)
{
	*snapshot = server->live;
	const result_status status =
		engine_stats_snapshot(server->engine, &snapshot->engine);
	if (status != OK) {
		return status;
	}
	const result_status measured = live_sampler_read(
		&server->live_sampler, &snapshot->engine, snapshot->windows);
	if (measured != OK) {
		return measured;
	}
	struct timespec wall = { 0 };
	if (clock_gettime(CLOCK_REALTIME, &wall) != 0) {
		return STATUS_INPUT_OUTPUT_ERROR;
	}
	snapshot->wall_seconds = (double)wall.tv_sec +
				 (double)wall.tv_nsec / NANOSECONDS_PER_SECOND;
	const double captured = snapshot->engine.now_ns != 0 ?
					(double)snapshot->engine.now_ns /
						NANOSECONDS_PER_SECOND :
					now;
	snapshot->now_seconds = captured;
	snapshot->queued_requests = server->waiting_count;
	snapshot->queue_depth = server->queue_depth;
	snapshot->user_queue_depth = server->user_queue_depth;
	for (uint32_t place = 0; place < server->waiting_count; place++) {
		const uint32_t user = user_index(snapshot, server->connections[server->waiting[place]].live.user);
		if (user < snapshot->user_count) {
			snapshot->users[user].queued_requests++;
		}
	}
	snapshot->open_connections = snapshot->request_count = 0;
	for (uint32_t index = 0; index < server->connection_limit;
	     index++) {
		const struct connection *const connection =
			&server->connections[index];
		snapshot->open_connections += connection->descriptor >= 0 ? 1u :
									    0u;
		if (connection->descriptor < 0 ||
		    !connection->live_is_request ||
		    connection->phase == PHASE_DRAINING) {
			continue;
		}
		struct live_request_record *const request =
			&snapshot->requests[snapshot->request_count++];
		*request = connection->live;
		if (request->slot < snapshot->engine.slots_total &&
		    snapshot->engine.slots[request->slot].request_tag ==
			    request->request_tag) {
			request->engine = snapshot->engine.slots[request->slot];
			phase_updated(request);
		}
	}
	for (uint32_t slot = 0; slot < server->description.session_slot_count;
	     slot++) {
		const struct active_command *const active =
			&server->active[slot];
		if (active->token_ids != NULL && active->operation == 0 &&
		    active->connection == NO_CONNECTION) {
			struct live_request_record *const request =
				&snapshot->requests[snapshot->request_count++];
			*request = active->disconnected_live;
			request->engine = snapshot->engine.slots[slot];
			phase_updated(request);
		}
	}
	for (uint32_t index = 0; index < snapshot->request_count; index++) {
		const struct live_request_record *const request =
			&snapshot->requests[index];
		const uint32_t user = user_index(snapshot, request->user);
		if (user < snapshot->user_count) {
			snapshot->users[user].concurrent_requests++;
			if (request->finished_seconds == 0 && request->slot != UINT32_MAX) {
				user_tokens_added(&snapshot->users[user], request);
			}
		}
	}
	for (uint32_t index = 0; index < server->live.finished_count; index++) {
		const uint32_t place =
			(server->finished_next + LIVE_STATS_FINISHED_CAPACITY -
			 1u - index) %
			LIVE_STATS_FINISHED_CAPACITY;
		snapshot->finished[index] = server->live.finished[place];
	}
	return OK;
}

result_status server_stats_json(struct http_server *server, double now,
				struct chat_buffer *output)
{
	struct live_snapshot *const snapshot = malloc(sizeof(*snapshot));
	if (snapshot == NULL) {
		return STATUS_OUT_OF_MEMORY;
	}
	result_status status = snapshot_taken(server, now, snapshot);
	status = status == OK ? live_stats_json_entry(snapshot, output) :
				status;
	free(snapshot);
	return status;
}

void server_stats_sample(struct http_server *server, double now)
{
	if (now < server->live_sample_seconds) {
		return;
	}
	server->live_sample_seconds =
		now + (double)LIVE_SAMPLE_PERIOD_NS / NANOSECONDS_PER_SECOND;
	struct engine_stats counters;
	if (engine_stats_counters(server->engine, &counters) == OK &&
	    live_sampler_take(&server->live_sampler, &counters) == OK) {
		server->live.engine = counters;
		server->live_sample_seconds = (double)server->live_sampler.next_sample_ns / NANOSECONDS_PER_SECOND;
	}
}

void server_stats_refresh(struct http_server *server, double now)
{
	if (server->live_stats_interval_ms == 0 ||
	    now < server->live_next_seconds) {
		return;
	}
	server->live_next_seconds =
		now + (double)server->live_stats_interval_ms /
			      MILLISECONDS_PER_SECOND;
	struct live_output *const output = server->live_output;
	if (output == NULL || pthread_mutex_trylock(&output->lock) != 0) {
		return;
	}
	if (!output->pending &&
	    snapshot_taken(server, now, &output->snapshot) == OK) {
		output->pending = true;
		(void)pthread_cond_signal(&output->wake);
	}
	(void)pthread_mutex_unlock(&output->lock);
}

void server_stats_queued(struct http_server *server,
			 struct connection *connection, double now)
{
	struct live_request_record *const request = &connection->live;
	connection->live_is_request = true;
	request->options = connection->request;
	request->tools_declared = connection->request.tools_declared;
	request->rendered_prompt_tokens = connection->token_count;
	request->engine.prompt_tokens = connection->token_count;
	request->engine.slot = UINT32_MAX;
	request->engine.speculation_mode = server->live.engine.speculation_mode;
	request->engine.context_capacity =
		server->description.context_token_limit;
	request->phase_seconds = now;
	text_copy(request->phase, sizeof(request->phase), "queued");
	text_copy(request->model, sizeof(request->model),
		  connection->request.uses_model_alias ?
			  server->model_alias :
			  server->description.model_name);
}

void server_stats_started(struct http_server *server,
			  struct connection *connection, uint32_t session_slot,
			  uint32_t kept, double now)
{
	struct live_request_record *const request = &connection->live;
	const struct slot_history *const history =
		&server->slot_histories[session_slot];
	request->slot = session_slot;
	memcpy(server->live.sessions[session_slot].user, request->user,
	       sizeof(server->live.sessions[session_slot].user));
	request->request_tag = server->active[session_slot].request_tag;
	request->queue_wait_seconds = now - request->phase_seconds;
	request->phase_seconds = now;
	request->reused_tokens = kept;
	request->reused_snapshot_tokens = history->from_snapshot ? kept : 0;
	request->reused_checkpoint_tokens =
		history->checkpoint_count != 0 &&
				history->checkpoint_count == kept &&
				history->reach_count >= kept &&
				history->reach_count - kept > SERVING_KEPT_LAG ?
			kept :
			0;
	request->reused_slot_tokens =
		request->reused_snapshot_tokens == 0 &&
				request->reused_checkpoint_tokens == 0 ?
			kept :
			0;
	text_copy(request->phase, sizeof(request->phase), "prompt");
	(void)snprintf(request->identifier, sizeof(request->identifier),
		       "chatcmpl-%" PRId64 "-%" PRIu32, server->started_seconds,
		       request->request_tag);
}

static uint32_t request_finished(struct http_server *server,
				 const struct live_request_record *request)
{
	const uint32_t index = server->finished_next;
	server->live.finished[index] = *request;
	server->finished_next = (index + 1u) % LIVE_STATS_FINISHED_CAPACITY;
	server->live.finished_count +=
		server->live.finished_count < LIVE_STATS_FINISHED_CAPACITY ?
			1u :
			0u;
	return index;
}

void server_stats_rejected(struct http_server *server,
			   struct connection *connection, uint32_t status,
			   double now)
{
	const uint32_t user = user_index(&server->live, connection->live.user);
	if (user < server->live.user_count) {
		server->live.users[user].refused++;
	}
	if (!connection->live_is_request ||
	    connection->phase != PHASE_WAITING) {
		return;
	}
	struct live_request_record *const request = &connection->live;
	request->queue_wait_seconds += now - request->phase_seconds;
	request->finished_seconds = request->phase_seconds = now;
	text_copy(request->phase, sizeof(request->phase), "finishing");
	text_copy(request->finish_reason, sizeof(request->finish_reason),
		  "error");
	text_copy(request->last_error, sizeof(request->last_error),
		  status == HTTP_SERVER_ERROR ? "engine refused the command" :
						"request refused");
	connection->finished_index = request_finished(server, request);
}

void server_stats_disconnected(struct http_server *server,
			       struct connection *connection, double now)
{
	if (connection->live_is_request && connection->phase == PHASE_WAITING) {
		connection->live.queue_wait_seconds +=
			now - connection->live.phase_seconds;
		connection->live.finished_seconds = now;
		text_copy(connection->live.phase,
			  sizeof(connection->live.phase), "finishing");
		connection->live.phase_seconds = now;
		text_copy(connection->live.finish_reason,
			  sizeof(connection->live.finish_reason), "cancelled");
		server->live.cancelled_requests++;
		(void)request_finished(server, &connection->live);
	}
}

void server_stats_final(struct http_server *server, uint32_t session_slot,
			const struct generation_event *event, double now)
{
	struct active_command *const active = &server->active[session_slot];
	struct connection *const connection =
		active->connection != NO_CONNECTION ?
			&server->connections[active->connection] :
			NULL;
	struct live_request_record *const request =
		connection != NULL ? &connection->live :
				     &active->disconnected_live;
	if (engine_stats_slot(server->engine, session_slot, &request->engine) !=
	    OK) {
		text_copy(request->last_error, sizeof(request->last_error),
			  "statistics unavailable");
	}
	request->finished_seconds = now;
	request->phase_seconds = now;
	text_copy(request->phase, sizeof(request->phase), "finishing");
	const bool failed = event->event_kind == GENERATION_EVENT_FAILED;
	const bool cancelled = event->event_kind == GENERATION_EVENT_CANCELLED;
	const bool stopped_visibly =
		(connection != NULL && connection->answer.stop_matched) ||
		strcmp(request->finish_reason, "stop") == 0 ||
		strcmp(request->finish_reason, "tool_calls") == 0;
	server->live.failed_requests += failed ? 1u : 0u;
	server->live.cancelled_requests += cancelled ? 1u : 0u;
	server->live.served_requests +=
		!failed && (!cancelled || stopped_visibly) ? 1u : 0u;
	const uint32_t user = user_index(&server->live, request->user);
	if (user < server->live.user_count) {
		server->live.users[user].served_requests += !failed && (!cancelled || stopped_visibly) ? 1u : 0u;
		user_tokens_added(&server->live.users[user], request);
	}
	const char *const reason =
		failed			      ? "error" :
		cancelled && !stopped_visibly ? "cancelled" :
		!stopped_visibly && event->event_kind ==
					    GENERATION_EVENT_FINISHED_AT_LIMIT ?
						"length" :
		(connection != NULL && connection->answer.call_count != 0) ||
				strcmp(request->finish_reason, "tool_calls") ==
					0 ?
						"tool_calls" :
						"stop";
	text_copy(request->finish_reason, sizeof(request->finish_reason),
		  reason);
	if (failed) {
		text_copy(request->last_error, sizeof(request->last_error),
			  status_text((result_status)event->status));
	}
	const uint32_t index = request_finished(server, request);
	server->live.sessions[session_slot].last_request = *request;
	server->live.sessions[session_slot].has_last_request = true;
	if (connection != NULL) {
		connection->finished_index = index;
	}
}

void server_stats_output(struct http_server *server,
			 struct connection *connection)
{
	if (connection->live_is_request && connection->live.slot < server->description.session_slot_count) {
		struct live_session *const session =
			&server->live.sessions[connection->live.slot];
		if (session->has_last_request && session->last_request.request_tag == connection->live.request_tag &&
		    session->last_request.arrival_seconds == connection->live.arrival_seconds) {
			session->last_request.bytes_sent = connection->live.bytes_sent;
			session->last_request.events_sent = connection->live.events_sent;
		}
	}
	if (connection->live_is_request &&
	    connection->finished_index < LIVE_STATS_FINISHED_CAPACITY) {
		struct live_request_record *const finished =
			&server->live.finished[connection->finished_index];
		if (finished->request_tag == connection->live.request_tag &&
		    finished->arrival_seconds ==
			    connection->live.arrival_seconds) {
			finished->bytes_sent = connection->live.bytes_sent;
			finished->events_sent = connection->live.events_sent;
		}
	}
}

void server_stats_refused(struct http_server *server, uint32_t status)
{
	server->live.refused_503 += status == HTTP_UNAVAILABLE ? 1u : 0u;
	server->live.refused_400 += status == HTTP_BAD_REQUEST ? 1u : 0u;
	server->live.refused_auth +=
		status == HTTP_UNAUTHORIZED || status == HTTP_FORBIDDEN ? 1u :
									  0u;
	server->live.failed_requests += status == HTTP_SERVER_ERROR ? 1u : 0u;
}
