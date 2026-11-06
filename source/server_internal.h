/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* SOURCE ONLY: the server's records (server.c), which server_sessions.c shares; the polling thread's alone. */
#ifndef SERVER_INTERNAL_H
#define SERVER_INTERNAL_H

#include <stdbool.h>
#include <stdint.h>

#include "anemone/server.h"
#include "anemone/live_stats.h"
#include "serving_cursor.h"

#define SERVER_SESSION_CAPACITY SERVING_SESSION_CAPACITY
#define NO_CONNECTION SERVER_MOST_CONNECTIONS
#define WAITING_SNAPSHOT_READY SERVER_SESSION_CAPACITY
#define NO_WAITING_SNAPSHOT (WAITING_SNAPSHOT_READY + SERVER_SESSION_CAPACITY)
#define SERVER_IMAGE_IDENTITY_BYTES 32u
#define SERVER_CACHE_NAMESPACE_BYTES 8192u
#define SERVER_CACHE_FIELD_BYTES 512u

/* Exact, tagged, length-delimited identity, never a credential or a filename hash. Fixed storage is zeroed;
 * only byte_count bytes participate in equality. Integer flags have defined validation when read from disk. */
struct cache_namespace {
	uint32_t byte_count, has_session;
	uint8_t bytes[SERVER_CACHE_NAMESPACE_BYTES];
};

enum cache_fork_hint_kind {
	CACHE_FORK_NONE,
	CACHE_FORK_SESSION,
	CACHE_FORK_THREAD
};

/* Request-local donor preference, never an authority or part of the persisted namespace. */
struct cache_fork_hint {
	uint32_t byte_count, kind;
	char bytes[SERVER_CACHE_FIELD_BYTES + 1u];
};

_Static_assert(SERVER_SESSION_CAPACITY <= STATS_SLOT_CAPACITY,
	       "connection and statistics storage cover every server slot");

enum phase {
	PHASE_FREE,
	PHASE_READING,
	PHASE_WAITING,
	PHASE_GENERATING,
	PHASE_SENDING,
	PHASE_DRAINING
};

struct connection {
	int descriptor; /* -1: the place is free */
	uint32_t phase; /* enum phase */
	double read_deadline; /* reading: of the whole request; draining: of the linger */
	double write_deadline; /* of the output's next progress, while is_writing */
	double keep_alive_deadline; /* a stream's next event comment, if no piece goes before */
	bool is_writing; /* output waits to be sent */
	bool is_head_read;
	bool authenticated_user; /* bound with live.user when the head is authorized; survives key reloads */
	struct chat_buffer input;
	struct chat_buffer
		responses_fields; /* request echoes, owned until this connection closes */
	struct chat_buffer
		responses_input; /* converted Chat spans, erased after encoding or disconnect */
	struct server_head head;
	struct chat_request request;
	struct chat_answer
		answer; /* its output carries every response of the connection */
	uint32_t *token_ids; /* the encoded prompt, while it waits */
	uint32_t token_count;
	struct chat_images
		*images; /* owned until transfer to active, then final TAKEN */
	uint8_t image_identity[SERVER_IMAGE_IDENTITY_BYTES];
	struct cache_namespace cache_namespace;
	struct cache_fork_hint fork_parent;
	bool fork_attempted; /* one bounded donor-copy attempt per request, including a failed prerequisite */
	bool fork_prepared; /* only a published independent exact clone may preserve proof under legacy policy */
	/* Pending slot, READY + slot after a successful snapshot prerequisite, or NO_WAITING_SNAPSHOT. Its
	 * user's round-robin turn was already granted; continuation has priority through SAVE/LOAD to generation. */
	uint32_t waiting_snapshot_slot, waiting_snapshot_tag;
	struct live_request_record live;
	bool live_is_request;
	uint32_t live_header_match;
	bool live_body_started, live_last_newline;
	uint32_t finished_index; /* FINISHED_CAPACITY until its engine final event */
};

struct active_command {
	uint32_t connection; /* NO_CONNECTION after disconnect; never rebind to a recycled descriptor */
	uint32_t *token_ids; /* borrowed by worker until final TAKEN; NULL means free */
	uint32_t request_tag;
	uint32_t prompt_token_count; /* the prompt's tokens, the kept ones among them */
	uint32_t generated_token_count; /* its token events so far */
	uint32_t operation; /* 0, or its session command's (front_end_seam.h) */
	int descriptor; /* a session command's file */
	uint32_t reach_count; /* a load's: its snapshot's reach */
	struct chat_images *images;
	uint8_t image_identity[SERVER_IMAGE_IDENTITY_BYTES];
	struct cache_namespace cache_namespace;
	uint32_t fork_source_slot;
	uint64_t fork_source_generation, fork_destination_generation;
	struct live_request_record
		disconnected_live; /* preserved until final TAKEN after its connection closes */
};

/* What a session slot holds between commands, for the next to keep (prefix reuse): its tokens in order, those of
 * its last final event, of which the cards ran processed_count (not the last drawn), when it was last used, and its
 * reach (front_end_seam.h), as the engine counts it. */
struct slot_history {
	uint32_t token_count;
	uint32_t processed_count;
	uint64_t last_used;
	uint32_t saved_count; /* of processed_count, those its newest snapshot holds */
	uint32_t reach_count; /* a kept prefix counts back from it (serving_cursor.h) */
	uint32_t checkpoint_count; /* a matching prompt may keep this boundary beyond the ordinary lag */
	uint32_t decoder_from; /* final engine metadata: reuse must not cause a cone rewind */
	uint32_t cold_prompt_count; /* exact policy: original prompt length; zero when provenance is unknown */
	bool from_snapshot;
	uint8_t image_identity[SERVER_IMAGE_IDENTITY_BYTES];
	struct cache_namespace cache_namespace;
	double last_active_seconds; /* monotonic time of the last generation final event, for affinity grace */
};

struct http_server {
	int listener;
	uint32_t stats_user_count;
	struct server_keys_record keys; /* owned only by the HTTP poller */
	char *key_path;
	char stats_users[SERVER_MOST_USERS][SERVER_USER_BYTES];
	struct chat_request defaults;
	const char *model_alias; /* NULL: none; the options' */
	struct inference_engine *engine; /* the run's */
	const struct text_tokenizer *tokenizer;
	struct engine_description_record description;
	char *prompt; /* the template's output: the context's tokens at their longest */
	uint32_t *encoded; /* the context's tokens */
	uint32_t *histories; /* [session slot][context]: each slot's tokens */
	struct tokenizer_cache *request_caches[SERVER_SESSION_CAPACITY];
	struct cache_namespace request_cache_namespaces[SERVER_SESSION_CAPACITY];
	uint32_t next_request_cache; /* rotating replacement on a miss, independent of GPU slot ownership */
	struct slot_history slot_histories[SERVER_SESSION_CAPACITY];
	bool slot_read_leased
		[SERVER_SESSION_CAPACITY]; /* poll-thread donor reservation until the clone final */
	uint64_t commands_started;
	/* Shared index storage interleaves bounded per-user FIFOs in arrival order; dispatch selects a user's first.
	 * User ordinals come from lifetime statistics, so removing/reordering keys cannot rebind queued identities. */
	uint32_t waiting[SERVER_MOST_WAITING], waiting_count, last_waiting_user;
	uint32_t queue_depth, user_queue_depth, connection_limit;
	struct active_command active[SERVER_SESSION_CAPACITY];
	uint32_t next_tag;
	bool is_stopping;
	int64_t started_seconds; /* of the server, for the answers' identifiers */
	bool engine_failed;
	struct connection connections[SERVER_MOST_CONNECTIONS];
	struct session_store *store; /* the snapshots: NULL without a directory */
	const char *session_directory; /* the options' */
	uint64_t session_directory_bytes;
	double idle_since; /* when the last command ended */
	struct live_snapshot live;
	uint32_t finished_next;
	uint64_t connections_seen;
	uint32_t live_stats_interval_ms;
	bool live_stats_json;
	bool live_stats_table;
	double live_next_seconds, live_sample_seconds;
	struct live_sampler_record live_sampler;
	struct live_output *live_output;
	const char *stats_card_order, *stats_placement; /* immutable while run executes */
	bool disable_prefix_reuse; /* options: no GPU prefix/session snapshot reuse */
	uint32_t exact_prefix_reuse;
	uint32_t session_fork;
	uint32_t cache_scope, session_grace_seconds, session_save_tokens,
		session_load_tokens;
	uint32_t session_idle_save_seconds;
};

/* Called only after chat_request_read validated the whole JSON. Reads identity values, never logs them.
 * Failure leaves the connection namespace unchanged and fills a value-free HTTP refusal. */
uint32_t server_cache_namespace_read(const struct http_server *server,
				     struct connection *connection,
				     struct chat_refusal *refusal);
bool server_cache_namespace_equal(const struct cache_namespace *first,
				  const struct cache_namespace *second);
/* Cross-session eligibility requires authenticated ownership and equal hard isolation fields. Unknown or
 * malformed namespace records refuse. Neither helper waives the numerical or device-state copy contract. */
bool server_cache_owner_equal(const struct cache_namespace *first,
			      const struct cache_namespace *second);
bool server_cache_fork_hint_matches(const struct cache_fork_hint *hint,
				    const struct cache_namespace *identity);
/* Poll-thread records; exposed for deterministic selection tests. No active slot is returned. */
uint32_t server_chosen_session_slot(const struct http_server *server,
				    const struct connection *connection,
				    double now, uint32_t *kept_out);

/* The poller owns request metadata and the finished ring. Only the detached output worker may write stderr;
 * it owns its heap mailbox after close, so a blocked terminal can never hold server shutdown or a refresh. */
void server_stats_open(struct http_server *server, double now);
/* Preflight the lifetime username bound before appending any names; false leaves aggregates unchanged. */
bool server_stats_users_bind(struct http_server *server,
			     const struct server_keys_record *keys);
void server_stats_close(struct http_server *server);
void server_stats_refresh(struct http_server *server, double now);
/* Always scheduled at 1 Hz, independently of observers or the optional printer. */
void server_stats_sample(struct http_server *server, double now);
void server_stats_queued(struct http_server *server,
			 struct connection *connection, double now);
void server_stats_started(struct http_server *server,
			  struct connection *connection, uint32_t session_slot,
			  uint32_t kept, double now);
void server_stats_disconnected(struct http_server *server,
			       struct connection *connection, double now);
void server_stats_final(struct http_server *server, uint32_t session_slot,
			const struct generation_event *event, double now);
void server_stats_output(struct http_server *server,
			 struct connection *connection);
void server_stats_refused(struct http_server *server, uint32_t status);
void server_stats_rejected(struct http_server *server,
			   struct connection *connection, uint32_t status,
			   double now);
MUST_CHECK result_status server_stats_json(struct http_server *server,
					   double now,
					   struct chat_buffer *output);

/* Poll-thread validated records: tokens and image digest must both match before any prefix can be kept. */
uint32_t
server_prefix_kept(const struct http_server *server, uint32_t session_slot,
		   const uint32_t *token_ids, uint32_t token_count,
		   const uint8_t image_identity[SERVER_IMAGE_IDENTITY_BYTES]);
uint32_t server_prefix_kept_mode(
	const struct http_server *server, uint32_t session_slot,
	const uint32_t *token_ids, uint32_t token_count,
	const uint8_t image_identity[SERVER_IMAGE_IDENTITY_BYTES],
	uint32_t mode);
uint32_t server_fork_destination(const struct http_server *server,
				 const struct connection *connection,
				 uint32_t proposed, double now);
bool server_fork_before(struct http_server *server, uint32_t destination,
			struct connection *connection);
bool server_fork_final(struct http_server *server, uint32_t destination,
		       const struct generation_event *event, double now);
/* server.c's: the next request tag no live command holds; whether a command is live. */
uint32_t server_next_tag(struct http_server *server);
bool server_commands_are_active(const struct http_server *server);
/* The next per-user FIFO head in round-robin order; pending snapshot heads block their user alone.
 * A completed successful snapshot prerequisite resumes its already-started turn before ordinary heads.
 * Poll-thread records must have valid bounds and bound identities. Returns waiting_count if none is eligible,
 * leaving user_out untouched; NULL arguments return MOST_WAITING. First users: dispatch and deterministic tests. */
uint32_t server_next_waiting(const struct http_server *server,
			     uint32_t *user_out);
/* server_sessions.c's (its top): the store opened at run and closed at its end; before a request takes a slot, a
 * save or a load it starts first (true: the request waits for it); a session command's final event taken (true:
 * it was one); the saves of an idle spell, and the deadline that ends it; every session saved at the end. */
MUST_CHECK result_status server_sessions_open(struct http_server *server);
void server_sessions_close(struct http_server *server);
bool server_session_before(struct http_server *server, uint32_t session_slot,
			   const uint32_t *token_ids, uint32_t token_count,
			   uint32_t kept);
bool server_session_before_images(
	struct http_server *server, uint32_t session_slot,
	const uint32_t *token_ids, uint32_t token_count, uint32_t kept,
	const uint8_t image_identity[SERVER_IMAGE_IDENTITY_BYTES]);
bool server_session_before_namespace(struct http_server *server,
				     uint32_t session_slot,
				     const struct connection *connection,
				     uint32_t kept);
bool server_session_final(struct http_server *server, uint32_t session_slot,
			  const struct generation_event *event);
void server_sessions_idle(struct http_server *server, double now,
			  double *nearest);
void server_sessions_kept_at_end(struct http_server *server,
				 int event_descriptor);

#endif
