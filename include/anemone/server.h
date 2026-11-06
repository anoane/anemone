/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * server.h: `anemone serve`, the HTTP front end over the engine (front_end_seam.h); what it answers is
 * chat_completions.h's.
 *
 * Contract:
 *   - HTTP/1.1 (1.0 taken too) on one listening socket of a numeric IPv4 or IPv6 address, one request a connection.
 *     GET /health (503 once the engine failed), GET /v1/models, GET /v1/anemone/stats,
 *     POST /v1/chat/completions; another path is 404,
 *     another method 405.
 *   - A head is CRLF lines of printable ASCII (a tab inside a value), at most SERVER_MOST_HEAD_BYTES (431) and
 *     SERVER_MOST_HEAD_FIELDS fields; HTTP/1.1 names its Host once; a POST has one Content-Length, at most
 *     CHAT_MOST_BODY_BYTES (413; 411 without one), a GET none; Transfer-Encoding is 501; Expect is only
 *     100-continue (417), answered when the head is read, and ignored from HTTP/1.0 (RFC 9110 10.1.1), which gets no
 *     100. The request must be whole SERVER_READ_SECONDS after the connection (408); output that makes no
 *     progress for SERVER_WRITE_SECONDS ends the connection, and so does output above
 *     CHAT_MOST_OUTPUT_BYTES. The connection limit is SERVER_CONNECTION_FACTOR times
 *     (engine slots + queue_depth), plus SERVER_CONNECTION_HEADROOM; further connections wait in the
 *     listening queue. SERVER_MOST_CONNECTIONS bounds the fixed storage for all supported configurations.
 *   - With a key file, every route but /health takes "Authorization: Bearer <key>", compared in constant time, else
 *     401 before any body is read. server_keys.h defines named keys and the single-line legacy fallback;
 *     a file that group or others may access is refused. Keys are never logged and are erased at close/reload;
 *     the credential a request carries is erased once compared, and all a connection received when it closes.
 *   - One or four requests generate independently from position 0, one per engine session slot;
 *     up to queue_depth more wait (1..SERVER_MOST_WAITING, default SERVER_DEFAULT_QUEUE_DEPTH),
 *     encoded, FIFO within each authenticated username and round-robin between usernames. A user may hold every
 *     active slot, but at most user_queue_depth waiting requests; 0 selects max(1, queue_depth / 2). Explicit
 *     user_queue_depth may be 1..queue_depth; choosing queue_depth waives the default reservation for another
 *     user, and queue_depth=1 cannot reserve a second place. The unauthenticated mode has one username, "-".
 *     One more is 503 with Retry-After: decided from its head, before any body
 *     is read, and again once its body is encoded, should the queue have filled meanwhile. A prompt that leaves no
 *     token of the context free is 400. A client that closes or resets its connection ends its generation (a cancel
 *     at the next step) or leaves the queue; a client must not shut its side before the answer ends.
 *     After encoding, the received JSON and interim response buffers are freed. Each waiter retains only its
 *     connection/request record, one FIFO index and token_count uint32_t ids; token_count is less than context.
 *   - Request tokenization keeps one private cache entry per engine session slot of capacity, shared across
 *     connections by matching namespace and rendered text. Each holds at most CHAT_MOST_OUTPUT_BYTES of
 *     text, one
 *     context of token ids and tokenizer.h's bounded segment records. These caches retain no credentials, are
 *     freed at run's end, and give exactly full encoding's ids. KV reuse and usage.cached_tokens still depend
 *     on namespace equality and the engine session's history/reach/checkpoint checks. Namespace rebinding discards
 *     the old tokenizer entry. Cache allocation uses tokenizer_cache_memory.
 *   - cache_scope SESSION (zero/default) combines the authenticated username with every supplied recognized
 *     identity: body user, user_id, safety_identifier, prompt_cache_key, cache_salt, session_id and conversation_id;
 *     metadata session_id/sessionId, conversation_id/conversationId, prompt_cache_key, cache_key and user_id; stable
 *     session/project headers listed in server_cache.c. Missing, null and empty values add no identity.
 *     Decoded identity strings are exact UTF-8, at most 512 bytes each and 8192 bytes together with tags/user;
 *     invalid types/limits or repeated identity headers return 400. Unrelated metadata and request IDs are ignored.
 *     USER scopes only by username; GLOBAL explicitly permits cross-user/session content reuse. Identities grant
 *     no stored-response semantics or authorization and never appear in logs/stats. Anonymous access shares "-".
 *   - session_grace_seconds protects a recently finished namespace from eviction while another unprotected free
 *     slot exists. All protected slots still allow immediate dispatch; zero disables grace. Only idle slots are
 *     eligible. Explicit-session affinity breaks equal-prefix ties; content/reach/image validity remains required.
 *   - session_save_tokens (zero/default 1024), session_load_tokens (zero/default 256), and
 *     session_idle_save_seconds (zero/default 120) control the snapshot policy. A continuing prefix preserves its
 *     already-saved count. Snapshot v4 persists the complete namespace; old unscoped v3 files stay untouched.
 *   - POST /v1/responses is stateless: full plaintext input, instructions and function tools map to the same
 *     Chat renderer and native decoder; store:false is the only storage mode. previous_response_id and
 *     conversation restoration are refused. SSE uses Responses item/text/function events and a terminal
 *     completed, incomplete or failed event. Request identity uses the original body and session headers.
 *   - disable_prefix_reuse forces every request to prefill from zero, including shared BOS/role tokens.
 *     Its zero default retains the existing policy. Combining it with a session directory is refused.
 *   - exact_prefix_reuse: 1 admits only same-length cold-provenance text histories, keeping prompt_count-1 tokens
 *     at a closed compressor group. 2 also admits resumable boundaries of that prompt's cold plan. Other prefixes
 *     run cold. THIN/BATCH rows must have invariant arithmetic across row counts (TP4 EXL3 column owners and short
 *     forms). Multiple slots require prefill_first or prefill_order so chunk rows cannot enter THIN steps.
 *     Images, prefill-only commands and session directories have no reuse under these policies.
 *   - session_fork additionally requires the engine producer's immutable prefix_cold_plan_qualified capability.
 *     Unknown, zero or unrecognized capability values refuse startup. This is not a user knob or a replacement
 *     for exact recipe qualification; it never changes the selected same-session prefix policy.
 *   - With a session directory, sessions outlive a restart and the slot another request takes: server_sessions.c
 *     saves a session before a request takes its slot, after an idle spell and at the end, and loads a snapshot into
 *     a slot first when it keeps more of a request than the slot does (front_end_seam.h's session commands).
 *   - server_run returns when the stop descriptor polls readable, or with the status of a failure of the
 *     engine's queue or listening socket. Before return, every connection closes and all active commands are
 *     cancelled, with every final event waited for up to SERVER_WRITE_SECONDS; past that an error line says
 *     so and its prompt's ids stay allocated, as the engine may still read them. Fault point: server_memory.
 */
#ifndef SERVER_H
#define SERVER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "anemone/chat_completions.h"
#include "anemone/front_end_seam.h"
#include "anemone/engine_stats.h"
#include "anemone/status.h"
#include "anemone/server_keys.h"
#include "anemone/tokenizer.h"

#define SERVER_DEFAULT_ADDRESS "127.0.0.1"
#define SERVER_DEFAULT_PORT 8321u
#define SERVER_MOST_PORT 65535u
#define SERVER_DEFAULT_QUEUE_DEPTH 16u
#define SERVER_USER_QUEUE_FRACTION 2u
#define SERVER_MOST_WAITING 64u
#define SERVER_CONNECTION_FACTOR 2u
#define SERVER_CONNECTION_HEADROOM 8u
#define SERVER_MOST_CONNECTIONS                                \
	(SERVER_CONNECTION_FACTOR *                            \
		 (STATS_SLOT_CAPACITY + SERVER_MOST_WAITING) + \
	 SERVER_CONNECTION_HEADROOM)
#define SERVER_MOST_HEAD_BYTES 16384u
#define SERVER_MOST_HEAD_FIELDS 64u
#define SERVER_READ_SECONDS \
	600.0 /* a whole request: a million-token history through a slow uplink */
#define SERVER_WRITE_SECONDS 30.0
#define SERVER_KEEP_ALIVE_SECONDS \
	15.0 /* a stream without a piece this long gets an event comment */
#define SERVER_KEEP_ALIVE ": keep-alive\n\n"
#define SERVER_LINGER_SECONDS \
	2.0 /* reading after a response, so that it is not reset */
#define SERVER_SESSION_DIRECTORY_BYTES \
	(256ull << 30u) /* the most a session directory's snapshots take */
#define SERVER_DEFAULT_SAVE_TOKENS 1024u
#define SERVER_DEFAULT_LOAD_TOKENS 256u
#define SERVER_DEFAULT_IDLE_SAVE_SECONDS 120u
#define SERVER_MOST_GRACE_SECONDS 3600u
#define SERVER_MOST_IDLE_SAVE_SECONDS 86400u

enum server_cache_scope {
	SERVER_CACHE_SESSION,
	SERVER_CACHE_USER,
	SERVER_CACHE_GLOBAL,
	SERVER_CACHE_SCOPES
};

/* A request head, as server_head_read finds it. */
struct server_head {
	uint64_t body_byte_count;
	size_t head_byte_count; /* through the empty line */
	size_t key_begin,
		key_end; /* the bearer credential among the head's bytes; equal: none */
	uint32_t route; /* enum chat_route, once nothing refuses it */
	uint32_t refusal; /* 0, or the status that answers the request */
	bool expects_continue;
};

#define SERVER_PREFIX_LEGACY 0u
#define SERVER_PREFIX_REPEAT 1u
#define SERVER_PREFIX_COLD_PLAN 2u

struct server_options {
	const char *bind_address;
	const char *key_file; /* NULL: no key is asked for */
	const char *stats_users; /* NULL: every valid user; else comma-separated valid usernames, requires key_file */
	const char *model_alias; /* NULL: none; else a second name requests may give the model */
	const char *session_directory; /* NULL: no snapshots; else where sessions are kept (server_sessions.c) */
	uint64_t session_directory_bytes; /* the most its snapshots take */
	struct chat_request
		defaults; /* max_tokens, seed and is_seeded, temperature, top_p, effort */
	uint32_t port;
	uint32_t queue_depth; /* 0 selects the default; otherwise 1..SERVER_MOST_WAITING */
	uint32_t user_queue_depth; /* 0 selects max(1, queue_depth / USER_QUEUE_FRACTION); otherwise 1..queue_depth */
	uint32_t live_stats_interval_ms; /* 0: printing off; else 100..60000; counters and endpoint always available */
	bool live_stats_json;
	bool live_stats_table;
	const char *card_order; /* NULL: unspecified; immutable startup metadata */
	const char *placement; /* NULL: unspecified; immutable startup metadata */
	bool disable_prefix_reuse; /* zero retains reuse; true forces cold GPU prompts and refuses snapshots */
	uint32_t exact_prefix_reuse; /* 0 legacy, 1 final-row repeat, 2 compatible cold boundary; no snapshots */
	uint32_t session_fork; /* 0 retains strict namespaces; 1 permits exact independent same-owner slot copies */
	uint32_t cache_scope; /* zero: authenticated user plus supplied session/cache identities */
	uint32_t session_grace_seconds; /* zero disables free-slot affinity protection; at most MOST_GRACE_SECONDS */
	uint32_t session_save_tokens; /* zero selects DEFAULT_SAVE_TOKENS; otherwise the minimum unsaved tokens */
	uint32_t session_load_tokens; /* zero selects DEFAULT_LOAD_TOKENS; otherwise minimum reusable-token gain */
	uint32_t session_idle_save_seconds; /* zero selects DEFAULT_IDLE_SAVE_SECONDS */
};

struct http_server;

/* Binds, listens and reads the key file. A NULL argument, an address that is not numeric, a port of 0 or above
 * SERVER_MOST_PORT, a queue_depth above SERVER_MOST_WAITING, user_queue_depth above the effective
 * queue_depth, a cold/exact prefix policy combined with a session directory, and a key file the rules above refuse
 * are STATUS_INVALID_ARGUMENT, a socket the
 * system refuses STATUS_INPUT_OUTPUT_ERROR (one error line each, never the key). */
MUST_CHECK result_status server_open(const struct server_options *options,
				     struct http_server **server_out);
void server_close(struct http_server *server); /* accepts NULL */
/* The engine's description gives the model's name and context. The nonblocking stop descriptor carries signal
 * bytes: SIGHUP reloads keys transactionally; any other byte/EOF stops. Main installs the signal handlers.
 * At most 64 distinct usernames are retained over this server's lifetime, including removed users' totals;
 * a reload exceeding that bound fails with the old set intact. Requests retain their authenticated username.
 * Exact prefix modes refuse multiple slots without prefill_first or prefill_order, and refuse
 * PREFILL_ORDER_SHORT_ROWS until its cold-plan reuse is qualified (INVALID_ARGUMENT). */
MUST_CHECK result_status server_run(struct http_server *server,
				    struct inference_engine *engine,
				    const struct text_tokenizer *tokenizer,
				    int stop_descriptor);

/* For the tests. head_read: false while the bytes may be the beginning of a head; true once it is whole or can never
 * be, head_out then saying what it is. key_matches: in time that depends on the key's length only. */
bool server_head_read(const char *bytes, size_t byte_count,
		      struct server_head *head_out);
bool server_key_matches(const char *key, size_t key_byte_count,
			const char *given, size_t given_byte_count);
/* Stats client: reads one selected bearer key, with the legacy private regular-file checks and grammar.
 * NULL inputs: INVALID_ARGUMENT; no output changes on failure. For server multi-user files use keys_read. */
MUST_CHECK result_status server_key_read(const char *path, char *key,
					 size_t *key_byte_count_out);

#endif
