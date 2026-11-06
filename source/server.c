/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * server.c: `anemone serve`, the HTTP front end. Contract: server.h; the head of a request is server_head.c's.
 *
 * One thread polls the stop descriptor, the engine's event descriptor, each connection and the listening socket, with
 * a timeout to the nearest deadline. A connection reads its head (looked at again only when a line end arrives),
 * answers what the head decides alone (a refusal, a missing key, a GET, a full queue) before any body is read, reads
 * a chat request's body, renders and encodes it and waits in its user's FIFO; round-robin users enter free session
 * slots, one active command per slot. Received data is erased when a connection closes, its credential once compared.
 * Events go by live request tag to their active command's connection. Every socket is non-blocking: output is sent as
 * far as the socket takes it, the rest when it polls writable; once a response is sent the connection shuts its side
 * and reads until the client closes (or SERVER_LINGER_SECONDS), so that the client sees the response and not
 * a reset. A new connection is taken only behind the others' events, so that a place freed in this round never
 * meets a poll result of the connection that had it.
 */
#include "anemone/server.h"

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <math.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "anemone/chat_template.h"
#include "anemone/hooks.h"
#include "anemone/chat_images.h"
#include "anemone/model_seam.h"
#include "server_internal.h"
#include "server_responses.h"
#include "serving_cursor.h"

#define OK STATUS_OK
#define LINE_END_BYTES 2u
#define KEY_CHARACTERS "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~+/="
#define READ_PIECE_BYTES 65536u                     /* the most one recv asks for */
#define DRAIN_BYTES 4096u
#define LISTEN_BACKLOG 64
#define MILLISECONDS_PER_SECOND 1000.0
#define NANOSECONDS_PER_SECOND 1e9
#define IDENTIFIER_CAPACITY 48u
#define CACHE_LEAST_SHARED_BYTES 256u              /* common template markers alone do not identify a conversation */
#define CACHE_SHARED_FRACTION 2u                  /* a shared system prompt must cover at least half the request */
#define STOP_POLL 0u                                /* the places of the poll array before the connections */
#define EVENT_POLL 1u
#define LISTEN_POLL 2u
#define POLLS_BEFORE_CONNECTIONS 3u
#define CONTINUE_RESPONSE "HTTP/1.1 100 Continue\r\n\r\n"

_Static_assert(SERVER_MOST_HEAD_BYTES <= READ_PIECE_BYTES,
	       "a head must be read into one allocation, so that "
	       "no growth of the buffer leaves a copy of its credential behind");

/* The tokens a command for a prompt may keep of a slot's history: their common prefix, at most what the cards ran
 * and one short of the whole prompt, ending where the engine keeps tokens and no further back than it keeps them behind
 * the slot's reach (serving_cursor.h); 0 for none. */
uint32_t server_prefix_kept_mode(
	const struct http_server *server, uint32_t session_slot,
	const uint32_t *token_ids, uint32_t token_count,
	const uint8_t image_identity[SERVER_IMAGE_IDENTITY_BYTES],
	uint32_t mode)
{
	if (server->disable_prefix_reuse)
		return 0;
	const struct slot_history *const history =
		&server->slot_histories[session_slot];
	if (memcmp(history->image_identity, image_identity,
		   sizeof(history->image_identity)) != 0) {
		return 0;
	}
	const uint32_t *const tokens =
		server->histories +
		(size_t)session_slot * server->description.context_token_limit;
	const uint32_t most = history->token_count < token_count ?
				      history->token_count :
				      token_count;
	uint32_t common = 0;
	while (common < most && tokens[common] == token_ids[common]) {
		common++;
	}
	uint32_t kept = common < history->processed_count ?
				common :
				history->processed_count;
	kept = kept < token_count ? kept : token_count - 1u;
	kept -= kept % SERVING_KEPT_GRANULE;
	const bool stable = server->description.prefill_prefix_stable != 0;
	/* Induction: earlier admitted histories ran cold or retained the same cold prefix. Same length pins all
	 * cold seams/forms. Mode 1 leaves the final THIN row; mode 2 observes resumable cold boundaries. Closed pairs
	 * need no slot replay. The stable form also admits nonshrinking prompts: its headless forms are fixed,
	 * and eight-row closure below the source sampling row preserves their owner phase. Images and imported
	 * snapshots have no such provenance. */
	static const uint8_t no_image[SERVER_IMAGE_IDENTITY_BYTES] = { 0 };
	if (mode != SERVER_PREFIX_LEGACY || stable) {
		if ((stable ? history->cold_prompt_count == 0 ||
				      token_count < history->cold_prompt_count :
			      history->cold_prompt_count != token_count) ||
		    history->from_snapshot ||
		    memcmp(image_identity, no_image, sizeof(no_image)) != 0)
			return 0;
		/* Generated THIN rows and the source's final sampling row are outside
		 * canonical prefill provenance. Eight-row closure preserves the
		 * slab-owner phase and leaves no compressor pair open. */
		if (stable) {
			if (kept >= history->cold_prompt_count)
				kept = history->cold_prompt_count - 1u;
			kept -= kept % SERVING_STABLE_GRANULE;
		}
		if (mode == SERVER_PREFIX_REPEAT) {
			if (kept + 1u != token_count ||
			    kept < history->decoder_from)
				return 0;
		} else if (engine_prefix_boundary(server->engine, token_count,
						  kept, history->decoder_from,
						  &kept) != OK) {
			return 0;
		}
	}
	if (history->reach_count - kept <= SERVING_KEPT_LAG) {
		return kept;
	}
	const uint32_t checkpoint = history->checkpoint_count;
	if (checkpoint == 0 || checkpoint > kept)
		return 0;
	if (mode == SERVER_PREFIX_COLD_PLAN || stable) {
		uint32_t boundary = 0;
		return engine_prefix_boundary(server->engine, token_count,
					      checkpoint, history->decoder_from,
					      &boundary) == OK &&
				       boundary == checkpoint &&
				       (mode != SERVER_PREFIX_REPEAT ||
					checkpoint == kept) ?
			       checkpoint :
			       0u;
	}
	return mode == SERVER_PREFIX_LEGACY || checkpoint == kept ? checkpoint :
								    0u;
}

uint32_t
server_prefix_kept(const struct http_server *server, uint32_t session_slot,
		   const uint32_t *token_ids, uint32_t token_count,
		   const uint8_t image_identity[SERVER_IMAGE_IDENTITY_BYTES])
{
	return server_prefix_kept_mode(server, session_slot, token_ids,
				       token_count, image_identity,
				       server->exact_prefix_reuse);
}

/* Protection is a preference over free slots, never a reservation that stalls dispatch. Namespace identity gates
 * content matching before numeric eligibility; affinity cannot make a nonmatching prefix reusable. */
uint32_t server_chosen_session_slot(const struct http_server *server,
				    const struct connection *connection,
				    double now, uint32_t *kept_out)
{
	if (connection->waiting_snapshot_slot >= WAITING_SNAPSHOT_READY &&
	    connection->waiting_snapshot_slot < NO_WAITING_SNAPSHOT) {
		const uint32_t reserved = connection->waiting_snapshot_slot -
					  WAITING_SNAPSHOT_READY;
		if (reserved < server->description.session_slot_count &&
		    server->active[reserved].token_ids == NULL &&
		    !server->slot_read_leased[reserved] &&
		    server_cache_namespace_equal(
			    &server->slot_histories[reserved].cache_namespace,
			    &connection->cache_namespace)) {
			*kept_out = server_prefix_kept_mode(
				server, reserved, connection->token_ids,
				connection->token_count,
				connection->image_identity,
				connection->fork_prepared ?
					SERVER_PREFIX_COLD_PLAN :
					server->exact_prefix_reuse);
			return reserved;
		}
	}
	uint32_t chosen = server->description.session_slot_count, best = 0;
	bool chosen_protected = true, chosen_affinity = false;
	for (uint32_t session_slot = 0;
	     session_slot < server->description.session_slot_count;
	     session_slot++) {
		if (server->active[session_slot].token_ids != NULL ||
		    server->slot_read_leased[session_slot]) {
			continue;
		}
		const struct slot_history *const history =
			&server->slot_histories[session_slot];
		const bool same = server_cache_namespace_equal(
			&history->cache_namespace,
			&connection->cache_namespace);
		const uint32_t kept =
			same ? server_prefix_kept(server, session_slot,
						  connection->token_ids,
						  connection->token_count,
						  connection->image_identity) :
			       0;
		const bool protected =
			!same && history->token_count != 0 &&
			server->session_grace_seconds != 0 &&
			now < history->last_active_seconds +
					server->session_grace_seconds;
		const bool affinity =
			same && connection->cache_namespace.has_session != 0;
		const bool is_older =
			chosen != server->description.session_slot_count &&
			server->slot_histories[session_slot].last_used <
				server->slot_histories[chosen].last_used;
		if (chosen == server->description.session_slot_count ||
		    (chosen_protected && !protected) ||
		    (chosen_protected == protected &&
		     (kept > best ||
		      (kept == best && affinity && !chosen_affinity) ||
		      (kept == best && affinity == chosen_affinity &&
		       is_older)))) {
			chosen = session_slot;
			best = kept;
			chosen_protected = protected;
			chosen_affinity = affinity;
		}
	}
	if (best == 0 && server->session_fork != 0) {
		chosen = server_fork_destination(server, connection, chosen,
						 now);
	}
	*kept_out = best;
	return chosen;
}

bool server_commands_are_active(const struct http_server *server)
{
	for (uint32_t session_slot = 0;
	     session_slot < server->description.session_slot_count;
	     session_slot++) {
		if (server->active[session_slot].token_ids != NULL) {
			return true;
		}
	}
	return false;
}

/* Wrap is permitted only past tags whose final event was already taken. */
uint32_t server_next_tag(struct http_server *server)
{
	/* At most four live tags: one of five successive candidates is free. */
	for (uint32_t attempt = 0; attempt <= SERVING_SESSION_CAPACITY;
	     attempt++) {
		server->next_tag++;
		bool is_live = false;
		for (uint32_t session_slot = 0;
		     session_slot < server->description.session_slot_count;
		     session_slot++) {
			is_live = is_live ||
				  (server->active[session_slot].token_ids !=
					   NULL &&
				   server->active[session_slot].request_tag ==
					   server->next_tag);
		}
		if (!is_live) {
			return server->next_tag;
		}
	}
	/* Unreachable with admitted slot_count<=4; callers never mutate active in
	 * this single-threaded function. Retain a defined return for C11. */
	return server->next_tag;
}

static double monotonic_seconds(void)
{
    struct timespec now = {0};
    const bool clock_was_read = clock_gettime(CLOCK_MONOTONIC, &now) == 0;
    return clock_was_read ? (double)now.tv_sec + (double)now.tv_nsec / NANOSECONDS_PER_SECOND : 0.0;
}

/* ---- connections ------------------------------------------------------------------------------------------- */

static void connection_close(struct http_server *server, uint32_t index)
{
    struct connection *const connection = &server->connections[index];
    server_stats_disconnected(server, connection, monotonic_seconds());
    uint32_t kept = 0;
    for (uint32_t session_slot = 0;
	 session_slot < server->description.session_slot_count;
	 session_slot++) {
	    struct active_command *const active = server->active + session_slot;
	    if (active->token_ids != NULL && active->connection == index) {
		    active->disconnected_live = connection->live;
		    active->connection = NO_CONNECTION;
		    if (engine_cancel(server->engine, session_slot) != OK) {
			    LOG_ERROR(
				    "server: the engine refused to cancel the command of a connection that closed");
		    }
	    }
    }
    for (uint32_t place = 0; place < server->waiting_count; place++) {
        server->waiting[kept] = server->waiting[place];
        kept += server->waiting[place] != index ? 1u : 0u;
    }
    server->waiting_count = kept;
    if (connection->descriptor >= 0 && close(connection->descriptor) != 0) {
	    LOG_DEBUG_ENTRY(
		    "server: a connection's socket did not close cleanly");
    }
    free(connection->token_ids);
    chat_images_destroy(connection->images);
    free(connection->images);
    if (connection->input.bytes != NULL) {
        explicit_bzero(connection->input.bytes, connection->input.count);     /* the request: a key, a prompt */
    }
    if (connection->responses_input.bytes != NULL) {
	    explicit_bzero(connection->responses_input.bytes,
			   connection->responses_input.count);
    }
    chat_buffer_free(&connection->responses_input);
    if (connection->responses_fields.bytes != NULL) {
	    explicit_bzero(connection->responses_fields.bytes,
			   connection->responses_fields.count);
    }
    chat_buffer_free(&connection->responses_fields);
    chat_buffer_free(&connection->input);
    chat_answer_free(&connection->answer);
    memset(connection, 0, sizeof(*connection));
    connection->descriptor = -1;
}

/* Sends what the output holds as far as the socket takes it; a response sent whole ends the sending. False when the
 * connection is to be closed. */
static bool output_sent(struct http_server *server,
			struct connection *connection, double now)
{
	struct chat_buffer *const output = &connection->answer.output;
	if (output->is_short) {
		LOG_WARNING_ENTRY(
			"server: a response was above %u bytes, or its memory was refused: its connection closes",
			CHAT_MOST_OUTPUT_BYTES);
		return false;
	}
	connection->write_deadline = connection->is_writing ?
					     connection->write_deadline :
					     now + SERVER_WRITE_SECONDS;
	while (output->sent < output->count) {
		const ssize_t sent = send(connection->descriptor,
					  output->bytes + output->sent,
					  output->count - output->sent,
					  MSG_NOSIGNAL);
		if (sent <= 0 && !(sent < 0 && errno == EINTR)) {
			connection->is_writing = true;
			return sent < 0 &&
			       (errno == EAGAIN || errno == EWOULDBLOCK);
		}
		if (sent > 0 && connection->live_is_request &&
		    connection->request.is_streamed &&
		    connection->answer.request != NULL) {
			static const char header_end[] = "\r\n\r\n";
			for (size_t index = 0; index < (size_t)sent; index++) {
				const char byte =
					output->bytes[output->sent + index];
				if (!connection->live_body_started) {
					connection->live_header_match =
						byte == header_end[connection
									   ->live_header_match] ?
							connection->live_header_match +
								1u :
						byte == header_end[0] ? 1u :
									0;
					connection->live_body_started =
						connection->live_header_match ==
						sizeof(header_end) - 1u;
				} else if (byte == '\n' &&
					   connection->live_last_newline) {
					connection->live.events_sent++;
					connection->live_last_newline = false;
				} else {
					connection->live_last_newline = byte ==
									'\n';
				}
			}
		}
		output->sent += sent > 0 ? (size_t)sent : 0;
		connection->live.bytes_sent += sent > 0 ? (uint64_t)sent : 0;
		server_stats_output(server, connection);
		connection->write_deadline = now + SERVER_WRITE_SECONDS;
	}
    output->count = output->sent = 0;
    connection->is_writing = false;
    if (connection->phase == PHASE_SENDING) {
        connection->live.events_sent = connection->live_is_request &&
            (!connection->request.is_streamed || connection->answer.request == NULL)
            ? 1u : connection->live.events_sent;
	server_stats_output(server, connection);
	connection->phase = PHASE_DRAINING;
	connection->read_deadline = now + SERVER_LINGER_SECONDS;
	return shutdown(connection->descriptor, SHUT_WR) == 0;
    }
    return true;
}

/* The response written into the output goes. False when the connection is to be closed. */
static bool response_sent(struct http_server *server,
			  struct connection *connection, double now)
{
    connection->phase = PHASE_SENDING;
    return output_sent(server, connection, now);
}

static bool refusal_sent(struct http_server *server,
			 struct connection *connection,
			 const struct chat_refusal *refusal, const char *body,
			 double now)
{
	server_stats_refused(server, refusal->status);
	server_stats_rejected(server, connection, refusal->status, now);
	chat_refusal_write(&connection->answer.output, refusal, body,
			   connection->head.route);
	return response_sent(server, connection, now);
}

/* Count from the admitted indices, so disconnect, dispatch and snapshot waits cannot leave quota debt. */
static bool waiting_is_full(const struct http_server *server, const char *user)
{
    uint32_t count = 0;
    for (uint32_t place = 0; place < server->waiting_count; place++) {
        count += strcmp(server->connections[server->waiting[place]].live.user, user) == 0 ? 1u : 0u;
    }
    return server->waiting_count >= server->queue_depth || count >= server->user_queue_depth;
}

/* Lifetime username ordinals survive key reloads. The first matching index is that user's FIFO head. */
uint32_t server_next_waiting(const struct http_server *server,
			     uint32_t *user_out)
{
    if (server == NULL || user_out == NULL) {
	    return SERVER_MOST_WAITING;
    }
    const uint32_t count = server->live.user_count;
    const uint32_t first = server->last_waiting_user < count ? (server->last_waiting_user + 1u) % count : 0;
    uint32_t chosen = server->waiting_count, chosen_user = 0;
    for (uint32_t round = 0; round < count; round++) {
        const uint32_t user = (first + round) % count;
        for (uint32_t place = 0; place < server->waiting_count; place++) {
            const struct connection *const connection = &server->connections[server->waiting[place]];
            if (strcmp(connection->live.user, server->live.users[user].user) == 0) {
                const uint32_t slot = connection->waiting_snapshot_slot;
                if (slot >= WAITING_SNAPSHOT_READY && slot < NO_WAITING_SNAPSHOT) {
			if (server->active[slot - WAITING_SNAPSHOT_READY]
					    .token_ids == NULL &&
			    !server->slot_read_leased[slot -
						      WAITING_SNAPSHOT_READY]) {
				*user_out = user;
				/* Finish the granted turn before another head consumes its prefix. */
				return place;
			}
			break;
		}
		if (slot < WAITING_SNAPSHOT_READY && slot < server->description.session_slot_count &&
                    server->active[slot].token_ids != NULL &&
                    server->active[slot].request_tag == connection->waiting_snapshot_tag) {
                    break; /* the user's later requests cannot overtake its waiting FIFO head */
                }
                if (chosen == server->waiting_count) {
                    chosen = place;
                    chosen_user = user;
                }
                break;
            }
        }
    }
    if (chosen != server->waiting_count) {
        *user_out = chosen_user;
    }
    return chosen;
}

/* Round-robin between per-user FIFO heads; only final TAKEN frees an active slot. */
static void next_started(struct http_server *server, double now)
{
	while (!server->is_stopping && server->waiting_count != 0) {
		uint32_t user = 0;
		const uint32_t place = server_next_waiting(server, &user);
		if (place == server->waiting_count) {
			break; /* each user's FIFO head is waiting on its tagged snapshot prerequisite */
		}
		const uint32_t index = server->waiting[place];
		struct connection *const connection =
			&server->connections[index];
		const bool continues_snapshot = connection->waiting_snapshot_slot >= WAITING_SNAPSHOT_READY &&
						connection->waiting_snapshot_slot < NO_WAITING_SNAPSHOT;
		uint32_t kept = 0;
		const uint32_t session_slot = server_chosen_session_slot(
			server, connection, now, &kept);
		if ((server->exact_prefix_reuse != SERVER_PREFIX_LEGACY ||
		     server->description.prefill_prefix_stable != 0) &&
		    connection->request.max_tokens == 0)
			kept = 0; /* prefill-only commands have no final THIN/logit row */
		if (!server->engine_failed &&
		    session_slot == server->description.session_slot_count) {
			break;
		}
		if (!server->engine_failed &&
		    session_slot < server->description.session_slot_count &&
		    server_session_before_namespace(server, session_slot,
						    connection, kept)) {
			if (server->active[session_slot].operation ==
			    SESSION_LOAD) {
				memcpy(server->live.sessions[session_slot].user, connection->live.user,
				       sizeof(server->live.sessions[session_slot].user));
			}
			if (!continues_snapshot) {
				server->last_waiting_user = user;
			}
			connection->waiting_snapshot_slot = session_slot;
			connection->waiting_snapshot_tag = server->active[session_slot].request_tag;
			continue; /* this head remains queued; another user's request may use a different free slot */
		}
		if (!server->engine_failed && kept == 0 &&
		    connection->request.max_tokens != 0 &&
		    server_fork_before(server, session_slot, connection)) {
			if (!continues_snapshot) {
				server->last_waiting_user = user;
			}
			connection->waiting_snapshot_slot = session_slot;
			connection->waiting_snapshot_tag =
				server->active[session_slot].request_tag;
			continue;
		}
		const struct chat_request *const request = &connection->request;
		char identifier[IDENTIFIER_CAPACITY] = { 0 };
		if (!continues_snapshot) {
			server->last_waiting_user = user;
		}
		connection->waiting_snapshot_slot = NO_WAITING_SNAPSHOT;
		memmove(server->waiting + place, server->waiting + place + 1u,
			(--server->waiting_count - place) * sizeof(server->waiting[0]));
		const uint32_t request_tag = server_next_tag(server);
		const uint32_t remaining =
			server->description.context_token_limit -
			connection->token_count;
		const uint32_t limit =
			server->description.session_slot_count > 1 &&
					request->max_tokens > remaining ?
				remaining :
				request->max_tokens;
		const struct generation_command_record command = {
			.new_token_ids = connection->token_ids + kept,
			.sampler_seed = request->seed,
			.session_slot = session_slot,
			.kept_token_count = kept,
			.new_token_count = connection->token_count - kept,
			.generated_token_limit = limit,
			.sampler_temperature = request->temperature,
			.sampler_top_p = request->top_p,
			.request_tag = request_tag,
			.images = connection->images
		};
		const result_status status =
			server->engine_failed ?
				STATUS_DEVICE_ERROR :
				engine_submit(server->engine, &command);
		const struct chat_refusal refusal = {
			HTTP_SERVER_ERROR,
			server->engine_failed ? CHAT_ENGINE_FAILED : NULL, NULL,
			0, 0
		};
		if (status != OK) {
			LOG_ERROR("server: the engine refused a command: %s",
				  status_text(status));
			free(connection->token_ids);
			connection->token_ids = NULL;
			if (!refusal_sent(server, connection, &refusal, NULL, now)) {
				connection_close(server, index);
			}
			continue;
		}
		server->active[session_slot] = (struct active_command){
			.connection = index,
			.token_ids = connection->token_ids,
			.request_tag = request_tag,
			.prompt_token_count = connection->token_count,
			.descriptor = -1,
			.images = connection->images
		};
		connection->images = NULL;
		memcpy(server->active[session_slot].image_identity,
		       connection->image_identity,
		       sizeof(connection->image_identity));
		server_stats_started(server, connection, session_slot, kept,
				     now);
		memcpy(server->histories +
			       (size_t)session_slot *
				       server->description.context_token_limit,
		       connection->token_ids,
		       (size_t)connection->token_count * sizeof(uint32_t));
		server->slot_histories[session_slot] = (struct slot_history){
			.last_used = ++server->commands_started,
			.cold_prompt_count =
				(server->exact_prefix_reuse !=
					 SERVER_PREFIX_LEGACY ||
				 server->description.prefill_prefix_stable !=
					 0 ||
				 (server->session_fork != 0 &&
				  (kept == 0 || connection->fork_prepared))) &&
						limit != 0 ?
					connection->token_count :
					0u,
			.reach_count =
				kept != 0 ? server->slot_histories[session_slot]
						    .reach_count :
					    0,
			.saved_count =
				kept < server->slot_histories[session_slot]
							.saved_count ?
					kept :
					server->slot_histories[session_slot]
						.saved_count,
			.cache_namespace = connection->cache_namespace
		};
		memcpy(server->slot_histories[session_slot].image_identity,
		       connection->image_identity,
		       sizeof(connection->image_identity));
		connection->token_ids = NULL;
		(void)snprintf(identifier, sizeof(identifier),
			       "%s-%" PRId64 "-%" PRIu32,
			       request->is_responses ? "resp" : "chatcmpl",
			       server->started_seconds, request_tag);
		connection->answer.pending_responses_fields =
			request->is_responses ? &connection->responses_fields :
						NULL;
		chat_answer_begin(&connection->answer, request,
				  server->description.model_name, identifier,
				  (int64_t)time(NULL), connection->token_count);
		connection->answer.cached_token_count = kept;
		connection->answer.defers_stop_finish = server->description.speculative_depth != 0;
		connection->phase = PHASE_GENERATING;
		connection->keep_alive_deadline =
			now + SERVER_KEEP_ALIVE_SECONDS;
		if (!output_sent(server, connection, now)) {
			connection_close(server, index);
		}
	}
}

/* One bounded text cache per session slot of capacity, owned only by the poller. Encoding precedes scheduling,
 * so these entries follow matching requests rather than GPU slot ownership. A cache hit never grants KV reuse:
 * chosen_session_slot still compares the resulting token ids and applies every reach/checkpoint restriction. */
static struct tokenizer_cache *
request_cache(struct http_server *server, size_t prompt_bytes,
	      const struct cache_namespace *identity)
{
    uint32_t chosen = server->description.session_slot_count, best = CACHE_LEAST_SHARED_BYTES - 1u;
    for (uint32_t index = 0; index < server->description.session_slot_count; index++) {
	    if (!server_cache_namespace_equal(
			&server->request_cache_namespaces[index], identity)) {
		    continue;
	    }
	    const uint32_t common =
		    tokenizer_cache_common_prefix(server->request_caches[index],
						  server->prompt, prompt_bytes);
	    if (common > best &&
		common >= prompt_bytes / CACHE_SHARED_FRACTION) {
		    best = common;
		    chosen = index;
	    }
    }
    if (chosen == server->description.session_slot_count) {
        chosen = server->next_request_cache;
        server->next_request_cache = chosen + 1u < server->description.session_slot_count ? chosen + 1u : 0;
    }
    if (chosen < SERVER_SESSION_CAPACITY &&
	!server_cache_namespace_equal(&server->request_cache_namespaces[chosen],
				      identity)) {
	    tokenizer_cache_close(server->request_caches[chosen]);
	    server->request_caches[chosen] = NULL;
	    const result_status opened = tokenizer_cache_open(
		    server->tokenizer, CHAT_MOST_OUTPUT_BYTES,
		    server->description.context_token_limit,
		    &server->request_caches[chosen]);
	    if (opened != OK) {
		    return NULL; /* ordinary full encoding remains available when the optional cache cannot allocate */
	    }
	    server->request_cache_namespaces[chosen] = *identity;
    }
    return chosen < SERVER_SESSION_CAPACITY ? server->request_caches[chosen] : NULL;
}

/* A chat request whose body is whole: refused, or encoded into the queue, and the queue's first submitted if no
 * command runs. False when the connection is to be closed. */
static bool chat_answered(struct http_server *server,
			  struct connection *connection, uint32_t index,
			  double now)
{
	const char *const original_body =
		connection->input.bytes + connection->head.head_byte_count;
	const char *body = original_body;
	const bool responses = connection->head.route == CHAT_ROUTE_RESPONSES;
	const struct chat_request *const request = &connection->request;
	const uint32_t context = server->description.context_token_limit;
	struct chat_refusal refusal = { 0 };
	size_t prompt_bytes = 0;
	uint32_t token_count = 0;
	if (server->engine_failed) {
		refusal = (struct chat_refusal){ HTTP_SERVER_ERROR,
						 CHAT_ENGINE_FAILED, NULL, 0,
						 0 };
	} else if ((responses ?
			    server_responses_read_fields(
				    original_body,
				    connection->head.body_byte_count,
				    server->description.model_name,
				    server->model_alias, &server->defaults,
				    &connection->request,
				    &connection->responses_input,
				    &connection->responses_fields, &refusal) :
			    chat_request_read(
				    body, connection->head.body_byte_count,
				    server->description.model_name,
				    server->model_alias, &server->defaults,
				    &connection->request, &refusal)) == 0) {
		body = responses ? connection->responses_input.bytes :
				   original_body;
		if (server_cache_namespace_read(server, connection, &refusal) !=
		    0) {
			return refusal_sent(server, connection, &refusal,
					    original_body, now);
		}
		const struct chat_template_options options = {
			.reasoning_effort =
				request->reasoning_effort != 0 ?
					request->reasoning_effort :
					CHAT_TEMPLATE_HIGHEST_EFFORT,
			.is_thinking = request->reasoning_effort != 0,
			.drops_thinking = true
		};
		struct chat_template_spans trusted = { 0 };
		const result_status rendered = chat_template_render_spans(
			body + request->messages_begin,
			request->messages_end - request->messages_begin,
			request->tools_end > request->tools_begin ?
				body + request->tools_begin :
				NULL,
			request->tools_end - request->tools_begin, &options,
			server->prompt,
			(size_t)context * TOKENIZER_MAXIMUM_TOKEN_BYTES,
			&prompt_bytes, &trusted);
		struct tokenizer_cache *const cache =
			rendered == OK ?
				request_cache(server, prompt_bytes,
					      &connection->cache_namespace) :
				NULL;
		result_status encoded =
			rendered != OK ? rendered :
			cache != NULL  ? tokenizer_encode_cached_spans(
						 cache, server->prompt,
						 prompt_bytes, trusted.spans,
						 trusted.count, server->encoded,
						 context, &token_count) :
					 tokenizer_encode_spans(
						 server->tokenizer,
						 server->prompt, prompt_bytes,
						 trusted.spans, trusted.count,
						 server->encoded, context,
						 &token_count);
		if (encoded == OK && trusted.image_count != 0) {
			connection->images =
				calloc(1, sizeof(*connection->images));
			if (connection->images == NULL) {
				refusal.status = HTTP_SERVER_ERROR;
			} else {
				const uint32_t image_status =
					chat_images_prepare(
						body + request->messages_begin,
						request->messages_end -
							request->messages_begin,
						&trusted, connection->images,
						&refusal);
				if (image_status != 0) {
					refusal.status = image_status;
				} else {
					encoded = chat_images_expand(
						connection->images,
						server->encoded, context,
						&token_count);
					if (encoded == OK) {
						encoded = chat_images_identity(
							connection->images,
							server->description
								.image_math_identity,
							connection
								->image_identity);
					}
				}
			}
		}
		free(trusted.spans);
		if (rendered == STATUS_INVALID_ARGUMENT) {
			refusal.status = HTTP_BAD_REQUEST;
			refusal.message =
				"the messages are no conversation the model's chat template takes";
		} else if (rendered == STATUS_UNSUPPORTED ||
			   encoded == STATUS_UNSUPPORTED ||
			   encoded == STATUS_INVALID_ARGUMENT ||
			   (encoded == OK && token_count >= context)) {
			refusal = (struct chat_refusal){
				HTTP_BAD_REQUEST,
				"context length exceeded: the prompt "
				"leaves no token of the context free",
				"context_length_exceeded", 0, 0
			};
		} else if (refusal.status ==
			   0) { /* the queue may have filled while the body came */
			refusal.status =
				encoded != OK || token_count == 0 ?
					HTTP_SERVER_ERROR :
				waiting_is_full(server, connection->live.user) ?
					HTTP_UNAVAILABLE :
					0;
		}
	}
	connection->token_ids =
		refusal.status == 0 && token_count != 0 &&
				!FAULT_POINT(server_memory) ?
			malloc((size_t)token_count * sizeof(uint32_t)) :
			NULL;
	refusal.status = refusal.status == 0 && connection->token_ids == NULL ?
				 HTTP_SERVER_ERROR :
				 refusal.status;
	if (refusal.status != 0) {
		return refusal_sent(server, connection, &refusal, original_body,
				    now);
	}
    memcpy(connection->token_ids, server->encoded, (size_t)token_count * sizeof(uint32_t));
    connection->token_count = token_count;
    /* Generation settings own their strings; Responses echoes have separate connection-owned storage. */
    explicit_bzero(connection->input.bytes, connection->input.count);
    if (connection->responses_input.bytes != NULL) {
	    explicit_bzero(connection->responses_input.bytes,
			   connection->responses_input.count);
    }
    chat_buffer_free(&connection->responses_input);
    chat_buffer_free(&connection->input);
    chat_buffer_free(&connection->answer.output);
    connection->phase = PHASE_WAITING;
    connection->waiting_snapshot_slot = NO_WAITING_SNAPSHOT;
    connection->fork_attempted = false;
    connection->fork_prepared = false;
    server_stats_queued(server, connection, now);
    server->waiting[server->waiting_count++] = index;
    next_started(server, now);                        /* closes the connection itself if it cannot answer */
    return true;
}

/* What the head decides alone: a refusal, a missing key, a GET, a full queue; a chat request goes on to read its
 * body. The credential is erased once compared, before the buffer can grow. False when the connection is to be
 * closed. */
static bool head_answered(struct http_server *server,
			  struct connection *connection, double now)
{
	const struct server_head *const head = &connection->head;
	struct chat_buffer *const output = &connection->answer.output;
	struct chat_refusal refusal = { head->refusal, NULL, NULL, 0, 0 };
	if (server->keys.count == 0) {
		memcpy(connection->live.user, "-", sizeof("-"));
	} else if (refusal.status == 0 && head->route != CHAT_ROUTE_HEALTH) {
		const uint32_t matched = server_keys_match(
			&server->keys,
			connection->input.bytes + head->key_begin,
			head->key_end - head->key_begin);
		if (matched < server->keys.count) {
			memcpy(connection->live.user,
			       server->keys.entries[matched].user,
			       sizeof(connection->live.user));
			connection->authenticated_user = true;
		} else {
			refusal.status = HTTP_UNAUTHORIZED;
		}
	}
	if (refusal.status == 0 && head->route == CHAT_ROUTE_STATS &&
	    server->stats_user_count != 0) {
		bool allowed = false;
		for (uint32_t index = 0; index < server->stats_user_count;
		     index++) {
			allowed = allowed ||
				  strcmp(connection->live.user,
					 server->stats_users[index]) == 0;
		}
		refusal.status = allowed ? 0 : HTTP_FORBIDDEN;
	}
    explicit_bzero(connection->input.bytes + head->key_begin, head->key_end - head->key_begin);
    if (refusal.status == 0 && head->route == CHAT_ROUTE_HEALTH &&
	server->engine_failed) {
	    refusal = (struct chat_refusal){ HTTP_UNAVAILABLE,
					     CHAT_ENGINE_FAILED, NULL, 0, 0 };
    } else if (refusal.status == 0 &&
	       (head->route == CHAT_ROUTE_COMPLETIONS ||
		head->route == CHAT_ROUTE_RESPONSES) &&
	       waiting_is_full(server, connection->live.user)) {
	    refusal.status =
		    HTTP_UNAVAILABLE; /* chat_answered asks again, once the body is read */
    }
    if (refusal.status != 0) {
        return refusal_sent(server, connection, &refusal, NULL, now);
    }
    if (head->route == CHAT_ROUTE_COMPLETIONS ||
	head->route == CHAT_ROUTE_RESPONSES) {
	    const bool is_early = connection->input.count <
				  head->head_byte_count + head->body_byte_count;
	    chat_buffer_write(output, CONTINUE_RESPONSE,
			      head->expects_continue && is_early ?
				      strlen(CONTINUE_RESPONSE) :
				      0);
	    return output_sent(server, connection, now) &&
		   !connection->is_writing; /* the 100 goes at once, or never */
    }
    if (head->route == CHAT_ROUTE_STATS) {
	    struct chat_buffer json = { 0 };
	    const result_status counted = server_stats_json(server, now, &json);
	    if (counted == OK) {
		    char header[CHAT_PREFIX_CAPACITY];
		    const int length = snprintf(
			    header, sizeof(header),
			    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: %zu\r\n"
			    "Connection: close\r\n\r\n",
			    json.count);
		    if (length > 0 && (size_t)length < sizeof(header)) {
			    chat_buffer_write(output, header, (size_t)length);
			    chat_buffer_write(output, json.bytes, json.count);
		    } else {
			    output->is_short = true;
		    }
	    }
	    chat_buffer_free(&json);
	    if (counted != OK) {
		    const struct chat_refusal failed = { HTTP_SERVER_ERROR,
							 NULL, NULL, 0, 0 };
		    return refusal_sent(server, connection, &failed, NULL, now);
	    }
	    return response_sent(server, connection, now);
    }
    chat_listing_write(output, head->route, server->description.model_name,
		       server->model_alias, server->started_seconds,
		       server->description.context_token_limit);
    return response_sent(server, connection, now);
}

/* Reads what the socket holds of a request, and answers it once its head or its body is whole. False when the
 * connection is to be closed. */
static bool request_bytes_read(struct http_server *server,
			       struct connection *connection, uint32_t index,
			       double now)
{
	struct chat_buffer *const input = &connection->input;
	const size_t wanted =
		connection->is_head_read ?
			connection->head.head_byte_count +
				(size_t)connection->head.body_byte_count :
			SERVER_MOST_HEAD_BYTES;
	const size_t piece = input->count >= wanted ? 0 :
			     wanted - input->count < READ_PIECE_BYTES ?
						      wanted - input->count :
						      READ_PIECE_BYTES;
	char *const room = piece != 0 ? chat_buffer_room(input, piece) : NULL;
	const ssize_t received =
		room != NULL ? recv(connection->descriptor, room, piece, 0) : 0;
	if (room == NULL || received == 0) {
		return false; /* memory refused, or the client went away */
	}
    if (received < 0) {
        return errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR;
    }
    input->count += (size_t)received;
    bool is_open = true;
    if (!connection->is_head_read &&
	(memchr(room, '\n', (size_t)received) != NULL ||
	 input->count >= SERVER_MOST_HEAD_BYTES)) {
	    connection->is_head_read = server_head_read(
		    input->bytes, input->count, &connection->head);
	    is_open = !connection->is_head_read ||
		      head_answered(server, connection, now);
    }
    if (is_open && connection->phase == PHASE_READING && connection->is_head_read &&
        input->count >= connection->head.head_byte_count + connection->head.body_byte_count) {
        is_open = chat_answered(server, connection, index, now);
    }
    return is_open;
}

/* What a poll found on a connection. */
static void connection_polled(struct http_server *server, uint32_t index,
			      const struct pollfd *poll_entry, double now)
{
    struct connection *const connection = &server->connections[index];
    char discarded[DRAIN_BYTES];
    bool is_open = true;
    if (connection->descriptor != poll_entry->fd || poll_entry->revents == 0) {
        return;                                      /* closed since it was polled */
    }
    if ((poll_entry->revents & POLLOUT) != 0) {
        is_open = output_sent(server, connection, now);
    }
    if (is_open && connection->phase == PHASE_READING && (poll_entry->revents & (POLLIN | POLLHUP | POLLERR)) != 0) {
        is_open = request_bytes_read(server, connection, index, now);
    } else if (is_open && connection->phase != PHASE_READING &&
               (poll_entry->revents & (POLLIN | POLLRDHUP | POLLHUP | POLLERR)) != 0) {
        const ssize_t received = recv(connection->descriptor, discarded, sizeof(discarded), 0);
        is_open = (poll_entry->revents & (POLLRDHUP | POLLHUP | POLLERR)) == 0 &&
                  (received > 0 || (received < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)));
    }
    if (!is_open) {
        connection_close(server, index);
    }
}

/* Route every event to its live command tag, independently of connection reuse. */
MUST_CHECK static result_status events_taken(struct http_server *server,
					     double now)
{
	struct generation_event event = { 0 };
	bool is_available = true;
	while (is_available) {
		const result_status status = engine_next_event(
			server->engine, &event, &is_available);
		if (status != OK) {
			return status;
		}
		if (!is_available) {
			continue;
		}
		uint32_t session_slot = 0;
		while (session_slot < server->description.session_slot_count &&
		       (server->active[session_slot].token_ids == NULL ||
			server->active[session_slot].request_tag !=
				event.request_tag)) {
			session_slot++;
		}
		if (session_slot == server->description.session_slot_count ||
		    event.event_kind >= GENERATION_EVENT_KIND_COUNT) {
			server->engine_failed = true;
			return STATUS_DEVICE_ERROR;
		}
		struct active_command *const active =
			server->active + session_slot;
		const bool is_final = event.event_kind !=
				      GENERATION_EVENT_TOKEN;
		if (is_final && active->operation != 0) { /* a session command's: server_sessions.c */
			bool prepared = false;
			if (active->operation == SESSION_CLONE) {
				prepared = server_fork_final(
					server, session_slot, &event, now);
			} else {
				(void)server_session_final(server, session_slot,
							   &event);
				prepared =
					event.event_kind ==
						GENERATION_EVENT_FINISHED_AT_LIMIT &&
					event.status == OK;
			}
			for (uint32_t place = 0; place < server->waiting_count; place++) {
				struct connection *const waiting = &server->connections[server->waiting[place]];
				if (waiting->waiting_snapshot_slot == session_slot &&
				    waiting->waiting_snapshot_tag == event.request_tag) {
					waiting->waiting_snapshot_slot =
						prepared ? WAITING_SNAPSHOT_READY + session_slot : NO_WAITING_SNAPSHOT;
				}
			}
			free(active->token_ids);
			*active = (struct active_command){ .connection =
								   NO_CONNECTION };
			server->idle_since = now;
			next_started(server, now);
			continue;
		}
		const uint32_t held = active->prompt_token_count +
				      active->generated_token_count;
		if (!is_final && held < server->description.context_token_limit) {
			server->histories[(size_t)session_slot *
						  server->description
							  .context_token_limit +
					  held] = event.token_id;
		}
		active->generated_token_count += is_final ? 0u : 1u;
		if (is_final) {         /* what the slot holds now, for the next command to keep */
			struct slot_history *const history =
				&server->slot_histories[session_slot];
			const uint32_t holds =
				event.event_kind == GENERATION_EVENT_FAILED ||
						event.session_token_count >
							held ?
					0u :
					event.session_token_count;
			history->token_count = holds;
			history->processed_count =
				holds != 0 && active->generated_token_count != 0 ?
					holds - 1u :
					holds;
			struct session_state state;
			const result_status described = engine_session_state(
				server->engine, session_slot, &state);
			if (described != OK) {
				return described;
			}
			history->reach_count = state.reach_count;
			history->decoder_from = state.decoder_from;
			history->checkpoint_count = state.checkpoint_count;
			history->last_active_seconds = now;
			history->saved_count =
				history->saved_count <
						history->processed_count ?
					history->saved_count :
					history->processed_count;
		}
		const bool is_end = !is_final &&
				    event.token_id ==
					    server->description.end_token_id;
		const uint32_t connection_index = active->connection;
		struct connection *const connection =
			connection_index != NO_CONNECTION ?
				&server->connections[connection_index] :
				NULL;
		const char *bytes = NULL;
		uint32_t byte_count = 0;
		if (connection != NULL && !is_final && !connection->answer.is_finished &&
		    !connection->answer.stop_matched) {
			connection->live.reasoning_tokens +=
				connection->answer.is_reasoning && !connection->answer.is_calling ? 1u : 0u;
			connection->live.tool_call_tokens += connection->answer.is_calling ? 1u : 0u;
			connection->live.content_tokens +=
				!connection->answer.is_reasoning && !connection->answer.is_calling ? 1u : 0u;
		}
		if (connection != NULL) {
			/* Publish token-phase accounting before token() can finish inline on a stop or closed call. */
			connection->answer.reasoning_token_count = connection->live.reasoning_tokens;
		}
		if (connection != NULL && is_final &&
		    server->description.speculative_depth != 0 &&
		    connection->request.draft_usage != CHAT_DRAFT_USAGE_NONE) {
			const result_status counted = engine_draft_statistics(
				server->engine, session_slot,
				&connection->answer.draft_command,
				connection->request.draft_usage ==
						CHAT_DRAFT_USAGE_FULL ?
					&connection->answer.draft_total :
					NULL);
			if (counted != OK) {
				return counted;
			}
			connection->answer.has_draft_statistics = true;
		}
		server->engine_failed = server->engine_failed ||
					event.event_kind ==
						GENERATION_EVENT_FAILED;
		if (connection != NULL && !is_final &&
		    (!is_end &&
		     tokenizer_token_bytes(server->tokenizer, event.token_id,
					   &bytes, &byte_count) != OK)) {
			LOG_ERROR(
				"server: session %u produced token %u, which has no bytes",
				session_slot, event.token_id);
			chat_answer_fail(&connection->answer);
		} else if (connection != NULL && !is_final &&
			   chat_answer_token(&connection->answer, bytes,
					     byte_count) &&
			   engine_cancel(server->engine, session_slot) != OK) {
			LOG_ERROR(
				"server: the engine refused to cancel a command that a stop string ended");
		} else if (connection != NULL &&
			   event.event_kind == GENERATION_EVENT_FAILED) {
			LOG_ERROR("server: a command failed: %s",
				  status_text((result_status)event.status));
			chat_answer_fail(&connection->answer);
		} else if (connection != NULL && is_final) {
			chat_answer_finish(
				&connection->answer,
				event.event_kind ==
						GENERATION_EVENT_FINISHED_AT_LIMIT ?
					"length" :
					"stop");
		}
		if (connection != NULL) {
			if (connection->answer.is_finished && connection->answer.stop_matched) {
				(void)snprintf(connection->live.finish_reason,
					sizeof(connection->live.finish_reason), "%s",
					connection->answer.call_count != 0 ? "tool_calls" : "stop");
			}
			connection->keep_alive_deadline =
				now + SERVER_KEEP_ALIVE_SECONDS;
			connection->phase =
				connection->phase == PHASE_GENERATING &&
						connection->answer.is_finished ?
					PHASE_SENDING :
					connection->phase;
			if (!output_sent(server, connection, now)) {
				connection_close(server, connection_index);
			}
		}
		if (is_final) {
			server_stats_final(server, session_slot, &event, now);
			chat_images_destroy(active->images);
			free(active->images);
			free(active->token_ids);
			*active = (struct active_command){
				.connection = NO_CONNECTION
			};
			server->idle_since = now;
			next_started(server, now);
		}
	}
	return OK;
}

MUST_CHECK static result_status connections_accepted(struct http_server *server,
						     double now)
{
    for (uint32_t index = 0; index < server->connection_limit; index++) {
        const int one = 1;
        if (server->connections[index].descriptor >= 0) {
            continue;
        }
        struct sockaddr_storage address = {0};
        socklen_t address_bytes = sizeof(address);
        const int descriptor = accept4(server->listener, (struct sockaddr *)&address, &address_bytes,
                                       SOCK_NONBLOCK | SOCK_CLOEXEC);
        if (descriptor < 0) {
            const bool is_passing = errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR ||
                                    errno == ECONNABORTED || errno == EPROTO || errno == EMFILE || errno == ENFILE ||
                                    errno == ENOBUFS || errno == ENOMEM || errno == EPERM;
            if (!is_passing) {
		    LOG_ERROR("server: the listening socket failed: errno %d",
			      errno);
	    }
	    return is_passing ? OK : STATUS_INPUT_OUTPUT_ERROR;
	}
        if (setsockopt(descriptor, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one)) != 0) {
            (void)close(descriptor);                 /* a piece of a stream may not wait for more */
            continue;
        }
	server->connections[index] = (struct connection){
		.descriptor = descriptor,
		.phase = PHASE_READING,
		.read_deadline = now + SERVER_READ_SECONDS,
		.finished_index = LIVE_STATS_FINISHED_CAPACITY,
		.live = { .slot = UINT32_MAX, .arrival_seconds = now }
	};
	struct live_request_record *const live =
		&server->connections[index].live;
	char numeric[INET6_ADDRSTRLEN] = {0};
        const bool is_ipv4 = address.ss_family == AF_INET;
        const struct sockaddr_in *const ipv4 = (const struct sockaddr_in *)&address;
        const struct sockaddr_in6 *const ipv6 = (const struct sockaddr_in6 *)&address;
        const void *const source = is_ipv4 ? (const void *)&ipv4->sin_addr : (const void *)&ipv6->sin6_addr;
        if (inet_ntop(address.ss_family, source, numeric, sizeof(numeric)) != NULL) {
            (void)snprintf(live->client, sizeof(live->client), is_ipv4 ? "%s:%u" : "[%s]:%u", numeric,
                           (unsigned)ntohs(is_ipv4 ? ipv4->sin_port : ipv6->sin6_port));
        }
        (void)snprintf(live->identifier, sizeof(live->identifier), "request-%" PRIu64, ++server->connections_seen);
    }
    return OK;
}

/* The deadlines that passed: a request not whole is 408, a linger or output that made no progress ends. */
static void deadlines_kept(struct http_server *server, double now)
{
	const struct chat_refusal timed_out = { HTTP_REQUEST_TIMEOUT, NULL,
						NULL, 0, 0 };
	for (uint32_t index = 0; index < server->connection_limit; index++) {
		struct connection *const connection =
			&server->connections[index];
		const bool is_late = connection->descriptor >= 0 &&
				     (connection->phase == PHASE_READING ||
				      connection->phase == PHASE_DRAINING) &&
				     now >= connection->read_deadline;
		if (is_late && connection->phase == PHASE_READING) {
			connection->is_writing = false;
			if (!refusal_sent(server, connection, &timed_out, NULL,
					  now)) {
				connection_close(server, index);
			}
		} else if (is_late || (connection->descriptor >= 0 &&
				       connection->is_writing &&
				       now >= connection->write_deadline)) {
			connection_close(server, index);
		} else if (connection->descriptor >= 0 &&
			   connection->phase == PHASE_GENERATING &&
			   connection->request.is_streamed &&
			   now >= connection->keep_alive_deadline) {
			chat_buffer_write(
				&connection->answer.output, SERVER_KEEP_ALIVE,
				strlen(SERVER_KEEP_ALIVE)); /* a long prefill: no proxy times out */
			connection->keep_alive_deadline =
				now + SERVER_KEEP_ALIVE_SECONDS;
			if (!output_sent(server, connection, now)) {
				connection_close(server, index);
			}
		}
    }
}

/* ---- the server -------------------------------------------------------------------------------------------- */

/* The key of a key file the contract takes, into key. */
result_status server_key_read(const char *path, char *key,
			      size_t *key_byte_count_out)
{
    if (path == NULL || key == NULL || key_byte_count_out == NULL) {
	    return STATUS_INVALID_ARGUMENT;
    }
    char bytes[SERVER_KEY_MOST_BYTES + LINE_END_BYTES +
	       1]; /* a byte more than a key and its line end */
    struct stat file = {0};
    size_t count = 0;
    ssize_t received = 1;
    const int descriptor = open(path, O_RDONLY | O_CLOEXEC | O_NOCTTY);
    bool is_valid = descriptor >= 0 && fstat(descriptor, &file) == 0 && S_ISREG(file.st_mode) &&
                    (file.st_mode & (S_IRWXG | S_IRWXO)) == 0;
    while (is_valid && received != 0 && count < sizeof(bytes)) {
        received = read(descriptor, bytes + count, sizeof(bytes) - count);
        is_valid = received >= 0 || errno == EINTR;
        count += received > 0 ? (size_t)received : 0;
    }
    if (descriptor >= 0) {
        (void)close(descriptor);                     /* read only: nothing is lost */
    }
    count -= count > 0 && bytes[count - 1] == '\n' ? 1u : 0u;
    count -= count > 0 && bytes[count - 1] == '\r' ? 1u : 0u;
    is_valid = is_valid && count >= SERVER_KEY_LEAST_BYTES &&
	       count <= SERVER_KEY_MOST_BYTES;
    for (size_t index = 0; is_valid && index < count; index++) {
        is_valid = bytes[index] != '\0' && strchr(KEY_CHARACTERS, bytes[index]) != NULL;
    }
    if (is_valid) {
        memcpy(key, bytes, count);
        *key_byte_count_out = count;
    } else {
	    LOG_ERROR(
		    "server: the key file %s cannot be read, is no regular file that only its owner may read, "
		    "or holds no key of 16 to 4,096 letters, digits and -._~+/=",
		    path);
    }
    explicit_bzero(bytes, sizeof(bytes));
    return is_valid ? OK : STATUS_INVALID_ARGUMENT;
}

result_status server_open(const struct server_options *options,
			  struct http_server **server_out)
{
    struct sockaddr_in address_4 = {.sin_family = AF_INET};
    struct sockaddr_in6 address_6 = {.sin6_family = AF_INET6};
    const int one = 1;
    if (options == NULL || server_out == NULL ||
	options->bind_address == NULL || options->port == 0 ||
	options->port > SERVER_MOST_PORT ||
	options->queue_depth > SERVER_MOST_WAITING ||
	options->exact_prefix_reuse > SERVER_PREFIX_COLD_PLAN ||
	((options->disable_prefix_reuse || options->exact_prefix_reuse) &&
	 options->session_directory != NULL) ||
	options->cache_scope >= SERVER_CACHE_SCOPES ||
	options->session_fork > 1u ||
	(options->session_fork != 0 &&
	 options->cache_scope != SERVER_CACHE_SESSION) ||
	options->session_grace_seconds > SERVER_MOST_GRACE_SECONDS ||
	options->session_idle_save_seconds > SERVER_MOST_IDLE_SAVE_SECONDS ||
	options->user_queue_depth > (options->queue_depth != 0 ?
					     options->queue_depth :
					     SERVER_DEFAULT_QUEUE_DEPTH) ||
	(options->live_stats_interval_ms != 0 &&
	 (options->live_stats_interval_ms < LIVE_STATS_LEAST_INTERVAL_MS ||
	  options->live_stats_interval_ms > LIVE_STATS_MOST_INTERVAL_MS)) ||
	(options->model_alias != NULL &&
	 (options->model_alias[0] == '\0' ||
	  strnlen(options->model_alias, MODEL_NAME_CAPACITY) ==
		  MODEL_NAME_CAPACITY))) {
	    LOG_ERROR(
		    "server: refused: no options, address or output, an invalid port or queue depth, or an alias "
		    "that is empty or longer than a model name, or snapshots incompatible with the prefix "
		    "policy");
	    return STATUS_INVALID_ARGUMENT;
    }
    const bool is_4 = inet_pton(AF_INET, options->bind_address, &address_4.sin_addr) == 1;
    if (!is_4 && inet_pton(AF_INET6, options->bind_address, &address_6.sin6_addr) != 1) {
	    LOG_ERROR("server: %s is no numeric IPv4 or IPv6 address",
		      options->bind_address);
	    return STATUS_INVALID_ARGUMENT;
    }
    address_4.sin_port = address_6.sin6_port = htons((uint16_t)options->port);
    struct http_server *const server =
	    FAULT_POINT(server_memory) ? NULL : calloc(1, sizeof(*server));
    if (server == NULL) {
	    return STATUS_OUT_OF_MEMORY;
    }
    server->listener = -1;
    server->queue_depth = options->queue_depth != 0 ?
				  options->queue_depth :
				  SERVER_DEFAULT_QUEUE_DEPTH;
    const uint32_t default_user_depth =
	    server->queue_depth / SERVER_USER_QUEUE_FRACTION;
    server->user_queue_depth = options->user_queue_depth != 0 ? options->user_queue_depth
                                                             : default_user_depth != 0 ? default_user_depth : 1u;
    server->last_waiting_user = UINT32_MAX;
    server->defaults = options->defaults;
    server->model_alias = options->model_alias;
    server->disable_prefix_reuse = options->disable_prefix_reuse;
    server->exact_prefix_reuse = options->exact_prefix_reuse;
    server->session_fork = options->session_fork;
    server->cache_scope = options->cache_scope;
    server->session_grace_seconds = options->session_grace_seconds;
    server->session_save_tokens = options->session_save_tokens != 0 ?
					  options->session_save_tokens :
					  SERVER_DEFAULT_SAVE_TOKENS;
    server->session_load_tokens = options->session_load_tokens != 0 ?
					  options->session_load_tokens :
					  SERVER_DEFAULT_LOAD_TOKENS;
    server->session_idle_save_seconds =
	    options->session_idle_save_seconds != 0 ?
		    options->session_idle_save_seconds :
		    SERVER_DEFAULT_IDLE_SAVE_SECONDS;
    server->session_directory = options->session_directory;
    server->session_directory_bytes = options->session_directory_bytes;
    server->live_stats_interval_ms = options->live_stats_interval_ms;
    server->live_stats_json = options->live_stats_json;
    server->live_stats_table = options->live_stats_table;
    server->stats_card_order = options->card_order;
    server->stats_placement = options->placement;
    for (uint32_t session_slot = 0; session_slot < SERVER_SESSION_CAPACITY;
	 session_slot++) {
	    server->active[session_slot].connection = NO_CONNECTION;
    }
    for (uint32_t index = 0; index < SERVER_MOST_CONNECTIONS; index++) {
	    server->connections[index].descriptor = -1;
    }
    result_status status =
	    options->key_file == NULL ?
		    OK :
		    server_keys_read(options->key_file, &server->keys);
    if (status == OK && options->key_file != NULL) {
        server->key_path = strdup(options->key_file);
	status = server->key_path != NULL ? OK : STATUS_OUT_OF_MEMORY;
    }
    if (status == OK && options->stats_users != NULL) {
        const char *cursor = options->stats_users;
        bool valid = options->key_file != NULL;
        do {
            const size_t length = strcspn(cursor, ",");
	    if (!server_user_valid(cursor, length) ||
		server->stats_user_count == SERVER_MOST_USERS) {
		    valid = false;
		    break;
	    }
	    memcpy(server->stats_users[server->stats_user_count++], cursor, length);
            cursor += length;
            if (*cursor == '\0') {
                break;
            }
            cursor++;
        } while (valid);
	status = valid ? OK : STATUS_INVALID_ARGUMENT;
    }
    if (status == OK && !server_stats_users_bind(server, &server->keys)) {
	    status = STATUS_INVALID_ARGUMENT;
    }
    if (status == OK) {
        server->listener = socket(is_4 ? AF_INET : AF_INET6, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
        const bool is_listening =
            server->listener >= 0 && setsockopt(server->listener, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one)) == 0 &&
            (is_4 || setsockopt(server->listener, IPPROTO_IPV6, IPV6_V6ONLY, &one, sizeof(one)) == 0) &&
            bind(server->listener, is_4 ? (const struct sockaddr *)&address_4 : (const struct sockaddr *)&address_6,
                 is_4 ? sizeof(address_4) : sizeof(address_6)) == 0 &&
            listen(server->listener, LISTEN_BACKLOG) == 0;
        if (!is_listening) {
		LOG_ERROR("server: cannot listen on %s port %" PRIu32
			  ": errno %d",
			  options->bind_address, options->port, errno);
		status = STATUS_INPUT_OUTPUT_ERROR;
	}
    }
    if (status != OK) {
	    server_close(server);
	    return status;
    }
    *server_out = server;
    return OK;
}

void server_close(struct http_server *server)
{
    if (server == NULL) {
        return;
    }
    if (server->listener >= 0 && close(server->listener) != 0) {
	    LOG_DEBUG_ENTRY(
		    "server: the listening socket did not close cleanly");
    }
    explicit_bzero(&server->keys, sizeof(server->keys));
    free(server->key_path);
    free(server);
}

/* Each disconnected command retains its prompt until final TAKEN. A timeout
 * quarantines host prompts; it never claims card or driver retirement. */
static void server_stopped(struct http_server *server, int event_descriptor)
{
	const double deadline = monotonic_seconds() + SERVER_WRITE_SECONDS;
	server->is_stopping = true;
	for (uint32_t index = 0; index < SERVER_MOST_CONNECTIONS; index++) {
		if (server->connections[index].descriptor >= 0) {
			connection_close(server, index);
		}
	}
	for (uint32_t session_slot = 0;
	     session_slot < server->description.session_slot_count;
	     session_slot++) {
		if (server->active[session_slot].token_ids != NULL &&
		    engine_cancel(server->engine, session_slot) != OK) {
			LOG_ERROR(
				"server: the engine refused to cancel an active command");
		}
	}
	while (server_commands_are_active(server) &&
	       monotonic_seconds() < deadline) {
		struct pollfd poll_entry = { event_descriptor, POLLIN, 0 };
		const double now = monotonic_seconds();
		const int timeout = now < deadline ?
					    (int)ceil((deadline - now) *
						      MILLISECONDS_PER_SECOND) :
					    0;
		if ((poll(&poll_entry, 1, timeout) < 0 && errno != EINTR) ||
		    events_taken(server, now) != OK) {
			break;
		}
	}
	for (uint32_t session_slot = 0;
	     session_slot < server->description.session_slot_count;
	     session_slot++) {
		if (server->active[session_slot].token_ids != NULL) {
			LOG_ERROR(
				"server: an active command did not end: its prompt stays allocated");
			server->active[session_slot].token_ids = NULL;
		}
	}
}

result_status server_run(struct http_server *server,
			 struct inference_engine *engine,
			 const struct text_tokenizer *tokenizer,
			 int stop_descriptor)
{
	struct pollfd polls[POLLS_BEFORE_CONNECTIONS + SERVER_MOST_CONNECTIONS];
	uint32_t polled[SERVER_MOST_CONNECTIONS] = {
		0
	}; /* the connection of each poll behind the three */
	int event_descriptor = -1;
	result_status status =
		server == NULL || tokenizer == NULL || stop_descriptor < 0 ?
			STATUS_INVALID_ARGUMENT :
			engine_describe(engine, &server->description);
	status = status != OK ?
			 status :
			 engine_event_descriptor(engine, &event_descriptor);
	const char *const model_name =
		status == OK ? server->description.model_name : NULL;
	if (status == OK &&
	    (model_name == NULL ||
	     server->description.context_token_limit == 0 ||
	     server->description.session_slot_count == 0 ||
	     server->description.session_slot_count > SERVER_SESSION_CAPACITY ||
	     (server->session_fork != 0 &&
	      server->description.prefix_cold_plan_qualified != 1u) ||
	     (server->description.prefill_prefix_stable != 0 &&
	      server->session_directory != NULL) ||
	     ((server->exact_prefix_reuse != SERVER_PREFIX_LEGACY ||
	       server->session_fork != 0 ||
	       server->description.prefill_prefix_stable != 0) &&
	      ((server->description.session_slot_count > 1u &&
		server->description.prefill_first == 0 &&
		server->description.prefill_order == 0) ||
	       server->description.prefill_order == PREFILL_ORDER_SHORT_ROWS ||
	       server->description.prefill_chunk_rows != 0 ||
	       server->description.prefill_decode_rows != 0)) ||
	     strnlen(model_name, MODEL_NAME_CAPACITY) == MODEL_NAME_CAPACITY ||
	     tokenizer_vocabulary_size(tokenizer) !=
		     server->description.vocabulary_size)) {
		status = STATUS_INVALID_ARGUMENT;
	}
    if (status != OK) {
	    LOG_ERROR(
		    "server: refused: no server, engine, tokenizer or stop descriptor, or an engine whose model "
		    "name, context, vocabulary or requested prefix policy does not fit");
	    return status;
    }
    const uint32_t context = server->description.context_token_limit;
    server->connection_limit =
	    SERVER_CONNECTION_FACTOR * (server->description.session_slot_count +
					server->queue_depth) +
	    SERVER_CONNECTION_HEADROOM;
    server->engine = engine;
    server->tokenizer = tokenizer;
    server->started_seconds = (int64_t)time(NULL);
    server->prompt =
	    FAULT_POINT(server_memory) ?
		    NULL :
		    malloc((size_t)context * TOKENIZER_MAXIMUM_TOKEN_BYTES);
    server->encoded = FAULT_POINT(server_memory) ?
			      NULL :
			      malloc((size_t)context * sizeof(uint32_t));
    server->histories =
	    FAULT_POINT(server_memory) ?
		    NULL :
		    calloc((size_t)server->description.session_slot_count *
				   context,
			   sizeof(uint32_t));
    status = server->prompt == NULL || server->encoded == NULL ||
			     server->histories == NULL ?
		     STATUS_OUT_OF_MEMORY :
		     OK;
    for (uint32_t index = 0; status == OK && index < server->description.session_slot_count; index++) {
	    status = tokenizer_cache_open(tokenizer, CHAT_MOST_OUTPUT_BYTES,
					  context,
					  &server->request_caches[index]);
    }
    status = status == OK ? server_sessions_open(server) : status;
    server->idle_since = monotonic_seconds();
    server_stats_open(server, server->idle_since);
    while (status == OK) {
        double now = monotonic_seconds(), nearest = INFINITY;
	server_stats_sample(server, now);
	nearest = server->live_sample_seconds;
        uint32_t poll_count = POLLS_BEFORE_CONNECTIONS;
        for (uint32_t index = 0; index < server->connection_limit; index++) {
            const struct connection *const connection = &server->connections[index];
            const bool is_reading = connection->phase == PHASE_READING || connection->phase == PHASE_DRAINING;
            if (connection->descriptor < 0) {
                continue;
            }
            nearest = is_reading && connection->read_deadline < nearest ? connection->read_deadline : nearest;
            nearest = connection->is_writing && connection->write_deadline < nearest ? connection->write_deadline
                                                                                     : nearest;
            nearest = connection->phase == PHASE_GENERATING && connection->request.is_streamed &&
                      connection->keep_alive_deadline < nearest ? connection->keep_alive_deadline : nearest;
            polled[poll_count - POLLS_BEFORE_CONNECTIONS] = index;
            polls[poll_count++] = (struct pollfd){
                connection->descriptor, (short)((connection->phase == PHASE_SENDING ? 0 : POLLIN | POLLRDHUP) |
                                                (connection->is_writing ? POLLOUT : 0)), 0};
        }
        polls[STOP_POLL] = (struct pollfd){stop_descriptor, POLLIN, 0};
        polls[EVENT_POLL] = (struct pollfd){event_descriptor, POLLIN, 0};
        polls[LISTEN_POLL] = (struct pollfd){poll_count - POLLS_BEFORE_CONNECTIONS < server->connection_limit
                                             ? server->listener : -1, POLLIN, 0};
	server_sessions_idle(server, now, &nearest);
	nearest = server->live_stats_interval_ms != 0 && server->live_next_seconds < nearest
                  ? server->live_next_seconds : nearest;
        const int timeout = isinf(nearest) ? -1 : nearest <= now ? 0
                          : (int)ceil((nearest - now) * MILLISECONDS_PER_SECOND);
        const int ready = poll(polls, poll_count, timeout);
        if (ready < 0 && errno != EINTR) {
		LOG_ERROR("server: poll failed: errno %d", errno);
		status = STATUS_INPUT_OUTPUT_ERROR;
		break;
        }
        now = monotonic_seconds();
        if (ready > 0 && (polls[STOP_POLL].revents & (POLLIN | POLLHUP | POLLERR)) != 0) {
		unsigned char signals[SERVER_MOST_HEAD_FIELDS];
		const ssize_t count =
			read(stop_descriptor, signals, sizeof(signals));
		bool stopping = count <= 0;
		bool reload = false;
		for (ssize_t index = 0; index < count; index++) {
			reload = reload || signals[index] == SIGHUP;
			stopping = stopping || signals[index] != SIGHUP;
		}
            if (stopping) {
                break;
            }
            if (reload && server->key_path != NULL) {
		    struct server_keys_record *const candidate =
			    calloc(1, sizeof(*candidate));
		    const bool loaded =
			    candidate != NULL &&
			    server_keys_read(server->key_path, candidate) ==
				    OK &&
			    server_stats_users_bind(server, candidate);
		    if (loaded) {
			    explicit_bzero(&server->keys, sizeof(server->keys));
			    server->keys = *candidate;
		    }
                if (candidate != NULL) {
                    explicit_bzero(candidate, sizeof(*candidate));
                    free(candidate);
                }
                (void)fprintf(stderr, "server: key reload %s\n", loaded ? "succeeded" : "failed; keeping old keys");
            }
        }
        status = ready > 0 && (polls[EVENT_POLL].revents & POLLIN) != 0 ? events_taken(server, now) : OK;
        for (uint32_t place = POLLS_BEFORE_CONNECTIONS; ready > 0 && place < poll_count; place++) {
            connection_polled(server, polled[place - POLLS_BEFORE_CONNECTIONS], &polls[place], now);
        }
        next_started(server, now);
        deadlines_kept(server, now);
	server_stats_refresh(server, now);
	status = status == OK && ready > 0 && (polls[LISTEN_POLL].revents & POLLIN) != 0
                 ? connections_accepted(server, now) : status;
    }
    server_stopped(server, event_descriptor);
    server_stats_close(server);
    server_sessions_kept_at_end(server, event_descriptor);
    server_sessions_close(server);
    free(server->prompt);
    free(server->encoded);
    free(server->histories);
    for (uint32_t index = 0; index < server->description.session_slot_count; index++) {
	    tokenizer_cache_close(server->request_caches[index]);
	    server->request_caches[index] = NULL;
    }
    server->next_request_cache = 0;
    server->prompt = NULL;
    server->encoded = NULL;
    server->histories = NULL;
    return status;
}
