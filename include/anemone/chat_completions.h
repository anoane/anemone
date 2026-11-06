/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * chat_completions.h: what a chat request of the OpenAI-compatible API asks for, and the bytes of every response of
 * the server (server.h): refusals, the route listings, and answers whole or streamed.
 *
 * Contract:
 *   - A request is a JSON object as json.h reads it (UTF-8, at most JSON_MAXIMUM_DEPTH deep), no object of it
 *     with a key twice. Read: model (absent, or the engine's model name: else 404), messages (1 to
 *     CHAT_MOST_MESSAGES; roles system, developer, user, assistant and tool; content a string, null,
 *     or text/image_url parts; user and tool images accept bounded PNG/JPEG base64 data URLs (chat_images.h);
 *     developer renders exactly as system;
 *     reasoning_content, tool_calls and tool_call_id kept as the template renders them; name ignored), tools (an
 *     array the template renders on the first message; empty: none), tool_choice ("auto", or "none": the tools are
 *     not rendered), stream, stream_options.include_usage, max_tokens and max_completion_tokens (the smaller),
 *     temperature 0 to 2, top_p in (0, 1], seed (0 to 2^64 - 1), stop (a string or up to CHAT_MOST_STOPS of
 *     1 to CHAT_MOST_STOP_BYTES bytes), n (1), reasoning_effort ("none" and "off" 0, "minimal" and "low" 50,
 *     "medium", "high" and "xhigh" 75, "max" 100, or 0 to 100; 0, "none" and "off" turn thinking off). Null keeps
 *     the default. The canonical 1017 encoder names low, high and max; minimal, medium and xhigh are API aliases.
 *     DeepSeek thinking requires an object
 *     holding only type "enabled" or "disabled" (null is malformed). Disabled overrides even a positive effort,
 *     independently of member order; enabled uses the request's effort or server default, refusing zero with 400.
 *     store accepts false only (no stored-response API), parallel_tool_calls true only (no single-call constraint),
 *     service_tier auto or default only. prompt_cache_retention, thinking_token_budget, thinking_budget,
 *     thinking_budget_tokens and priority are refused: no lifetime, reasoning-token cap or priority guarantee.
 *     user, user_id, metadata, prompt_cache_key, cache_salt, safety_identifier, session_id and conversation_id
 *     do not change generation settings. The HTTP server validates identity strings and uses them to scope caches.
 *     anemone_draft_usage accepts "none", "compact" or "full", absent/null keeps the server's default (zero-initialized
 *     defaults select compact). stream_options accepts only boolean include_usage. Function tools accept strict
 *     false or null; strict true and custom tools are refused. A member left out or null, except thinking, takes
 *     the server's default; a seed left out, where the server sets none, is drawn for the request from the system's
 *     random source (getrandom, never waiting: 500 when it has none to give; fault point chat_random). The API's
 *     response has no seed member. The legacy functions and function_call, a tool_choice that
 *     forces a call, other content parts, a response_format but text, and every other member that is not
 *     null, false, 0 or empty are refused with a 400 that names them: nothing is silently ignored.
 *   - A refusal is {"error": {"message", "type", "param", "code"}}: param the member blamed, as the body spells its
 *     key; 401 says WWW-Authenticate, 405 Allow, 503 Retry-After. Every response says Connection: close.
 *   - An answer: with thinking on, the text before "</think>" is reasoning_content and the text after it content;
 *     neither holds the end token. A stop string, looked for in the content only, ends the answer before it (finish
 *     "stop"); the end token is "stop" too, a limit "length". Text is given out as soon as no stop string, end of
 *     reasoning, beginning of calls or unfinished UTF-8 sequence can change it any more; what is no UTF-8 is U+FFFD.
 *     usage counts the prompt and every token generated up to the end of the answer. completion_tokens_details
 *     holds reasoning_tokens from the caller's token-phase accounting, never character counts; the server counts
 *     each accepted token in the phase before passing its bytes here, including a reasoning-closing delimiter or
 *     EOS reached in reasoning, excluding tokens already in a call block. The caller sets reasoning_token_count
 *     before token() (which may finish) and finish(). Begin resets it; completion_tokens is unchanged.
 *     With draft statistics available, usage.anemone_draft is omitted for none; compact holds version and this
 *     command's drawing_rounds and fixed sources[].{rounds,proposed[],reached[],accepted[]} only, with no confidence
 *     arrays or engine totals; full retains the original command/total structure including confidence and depth.
 *     Streamed: server-sent events, a
 *     chunk with the role, one for each piece of text, one with the finish, with include_usage one with the usage,
 *     then "data: [DONE]"; a failure after the head is an error event and no [DONE].
 *   - Calls, when the request rendered tools: the DSML block that encoding.py's parse_message_from_completion_text
 *     reads after "\n\n<｜DSML｜ calls" ends the content (or the reasoning, which the reference would refuse). Each
 *     invoke keeps the reference's attributes and parameter values, with any Python Unicode whitespace (or none)
 *     between tags instead of its required newline. Values are never trimmed; duplicate parameter names are
 *     refused. During generation a whole invoke, including its closing >, is given out once as tool_calls: an id,
 *     namespace and bare name as the reference splits them, and its decode_dsml_to_arguments arguments (a
 *     string="true" value as a JSON string, any other as written). A stream gives each call as one delta of its
 *     index. Only a properly closed calls block stops generation. Malformed blocks are held through the end token
 *     or request limit, then parsed again, including the reference's accepted EOF forms. A block of no whole call
 *     is given out as content as written; after emitted calls, only its unread suffix is returned as content.
 *     With a call given out, the end token or a closed block finishes "tool_calls"; a request limit stays "length".
 *   - A buffer grows to at most CHAT_MOST_OUTPUT_BYTES; a write beyond, or memory refused (fault point
 *     chat_memory), sets is_short, and what it holds must not be sent.
 */
#ifndef CHAT_COMPLETIONS_H
#define CHAT_COMPLETIONS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "anemone/status.h"
#include "anemone/draft_policy.h"
#include "anemone/tokenizer.h"

#define CHAT_MOST_BODY_BYTES \
	33554432u /* 32 MiB: a million-token history, escaped, with room */
#define CHAT_MOST_OUTPUT_BYTES 67108864u /* 64 MiB of one response or request */
#define CHAT_MOST_MESSAGES 4096u
#define CHAT_MOST_STOPS 4u
#define CHAT_MOST_STOP_BYTES 256u
#define CHAT_UTF8_MOST_BYTES 4u
#define CHAT_HELD_CAPACITY                                      \
	(TOKENIZER_MAXIMUM_TOKEN_BYTES + CHAT_MOST_STOP_BYTES + \
	 CHAT_UTF8_MOST_BYTES)
#define CHAT_PREFIX_CAPACITY \
	640u /* a chunk's JSON before its choices, the model escaped */
#define CHAT_IDENTIFIER_CAPACITY \
	48u /* an answer's identifier, its calls' ids made from it */
#define CHAT_RETRY_AFTER "5" /* seconds, as the header writes them */
#define CHAT_ENGINE_FAILED \
	"the engine failed: this server answers no more requests"

enum chat_route {
	CHAT_ROUTE_HEALTH,
	CHAT_ROUTE_MODELS,
	CHAT_ROUTE_COMPLETIONS,
	CHAT_ROUTE_STATS,
	CHAT_ROUTE_RESPONSES
};

enum http_status {
	HTTP_OK = 200,
	HTTP_BAD_REQUEST = 400,
	HTTP_UNAUTHORIZED = 401,
	HTTP_FORBIDDEN = 403,
	HTTP_NOT_FOUND = 404,
	HTTP_METHOD_NOT_ALLOWED = 405,
	HTTP_REQUEST_TIMEOUT = 408,
	HTTP_LENGTH_REQUIRED = 411,
	HTTP_CONTENT_TOO_LARGE = 413,
	HTTP_EXPECTATION_FAILED = 417,
	HTTP_FIELDS_TOO_LARGE = 431,
	HTTP_SERVER_ERROR = 500,
	HTTP_NOT_IMPLEMENTED = 501,
	HTTP_UNAVAILABLE = 503,
	HTTP_VERSION_NOT_SUPPORTED = 505
};

/* Zero-initialized defaults select compact. The source order in compact remains lookup, DSpark. */
enum chat_draft_usage {
	CHAT_DRAFT_USAGE_COMPACT,
	CHAT_DRAFT_USAGE_NONE,
	CHAT_DRAFT_USAGE_FULL
};

/* Bytes of one request or response: the response is sent from sent to count. */
struct chat_buffer {
	char *bytes;
	size_t count, sent, capacity;
	bool is_short;
};

/* A request as chat_request_read reads it; the server's defaults are one too. */
struct chat_request {
	size_t messages_begin,
		messages_end; /* the messages array, a span of the body */
	size_t tools_begin,
		tools_end; /* the tools array to render, a span of the body (equal: none) */
	uint64_t seed;
	uint32_t max_tokens;
	uint32_t reasoning_effort; /* 0: thinking off; else 1 to 100 */
	uint32_t draft_usage; /* enum chat_draft_usage */
	float temperature;
	float top_p;
	uint32_t image_count; /* fully validated URLs, before any image decoding */
	uint32_t stop_count;
	uint32_t stop_byte_counts[CHAT_MOST_STOPS];
	char stops[CHAT_MOST_STOPS][CHAT_MOST_STOP_BYTES];
	bool is_streamed;
	bool includes_usage;
	bool is_seeded; /* seed was given: by the request, or as the server's default */
	bool uses_model_alias; /* the body named the alias; absent/null model is canonical */
	bool is_responses; /* server route, never an HTTP setting */
	bool tools_declared; /* nonempty tools, including when tool_choice is none */
};

/* Why a request is refused: its status, a message for people (NULL: the status's own), the API's code or NULL, and the
 * member blamed as a span of the body, its key's quotes included (begin == end: none). */
struct chat_refusal {
	uint32_t status;
	const char *message;
	const char *code;
	size_t member_begin, member_end;
};

/* Responses formatting is private; the same answer decoder owns its optional per-connection state. */
struct server_responses_output;

/* An answer on its way. */
struct chat_answer {
	const struct chat_buffer *
		pending_responses_fields; /* begin consumes it; owner keeps it until finish */
	struct server_responses_output
		*responses; /* private Responses formatter, NULL for Chat */
	struct chat_buffer
		output; /* the response; every response of its connection */
	struct chat_buffer reasoning,
		content; /* a whole answer's text so far */
	struct chat_buffer
		dsml; /* the calls' block after "\n\n<｜DSML｜ calls" */
	struct chat_buffer
		calls; /* a whole answer's tool_calls, JSON objects with commas */
	const struct chat_request *request;
	char prefix[CHAT_PREFIX_CAPACITY]; /* the JSON of a chunk, or of the whole, before its choices */
	size_t prefix_count;
	char identifier[CHAT_IDENTIFIER_CAPACITY];
	char held[CHAT_HELD_CAPACITY]; /* generated bytes not given out yet */
	uint32_t held_count;
	uint32_t prompt_token_count, completion_token_count;
	uint64_t reasoning_token_count; /* caller's token-phase count, before token/finish */
	uint32_t cached_token_count; /* of the prompt's, those the session kept: usage says them */
	size_t dsml_read; /* bytes of the block whose calls were given out */
	uint32_t call_count;
	bool is_reasoning, is_calling, are_calls_closed, is_finished;
	bool are_calls_malformed; /* held for one final parse; never a stop decision */
	bool defers_stop_finish, stop_matched, has_draft_statistics;
	struct draft_statistics draft_command,
		draft_total; /* usage.anemone_draft, set before finish */
};

/* Room for more bytes behind count (NULL and is_short when there is none), a write, and the memory given back. */
char *chat_buffer_room(struct chat_buffer *buffer, size_t more);
void chat_buffer_write(struct chat_buffer *buffer, const void *bytes,
		       size_t count);
void chat_buffer_free(struct chat_buffer *buffer);

/* 0 and request_out (the defaults with what the body sets), or the status of a refusal and refusal_out; 500 for a
 * missing argument or memory refused. */
MUST_CHECK uint32_t chat_request_read(const char *body, size_t byte_count,
				      const char *model_name,
				      const char *model_alias,
				      const struct chat_request *defaults,
				      struct chat_request *request_out,
				      struct chat_refusal *refusal_out);
/* A refusal's response, in output, which it empties first; body is the request's (NULL: no member is named). */
void chat_refusal_write(struct chat_buffer *output,
			const struct chat_refusal *refusal, const char *body,
			uint32_t route);
/* The response of GET /health ({"status": "ok"}) or GET /v1/models, in output, which it empties first; a model alias
 * (NULL: none) is listed too, and a request may name it as the model. */
void chat_listing_write(struct chat_buffer *output, uint32_t route,
			const char *model_name, const char *model_alias,
			int64_t created_seconds, uint32_t context_token_limit);
/* An answer: begin (a stream's head and first chunk), the bytes of each generated token (none for the end token; true
 * when a stop string ended the answer), finish with "stop" or "length", or fail (the engine failed). */
void chat_answer_begin(struct chat_answer *answer,
		       const struct chat_request *request,
		       const char *model_name, const char *identifier,
		       int64_t created_seconds, uint32_t prompt_token_count);
bool chat_answer_token(struct chat_answer *answer, const char *bytes,
		       uint32_t byte_count);
void chat_answer_finish(struct chat_answer *answer, const char *finish_reason);
void chat_answer_fail(struct chat_answer *answer);
void chat_answer_free(
	struct chat_answer *answer); /* its buffers; the record stays */

#endif
