/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * chat_completions.c: what a chat request asks for, and the bytes of every response. Contract: chat_completions.h.
 *
 * The body is checked whole first (json.h), then read member by member; strings are unescaped into scratch memory
 * of the body's size four times over and given back. A whole response's body is written behind room for its head,
 * which goes in front of it once its length is known. An answer holds back the bytes that a stop string, the end of
 * reasoning or an unfinished UTF-8 sequence may still change: after each token it looks for the first whole
 * delimiter in what it holds, else keeps the longest tail that begins one, and gives out the rest at once, a chunk
 * of a stream or raw text of a whole answer.
 */
#include "chat_completions.h"

#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/random.h>

#include "models/deepseek_v41_flash/prompt/ds41f_chat_template.h"
#include "engine/input/image/request_images.h"
#include "common/hooks.h"
#include "common/json.h"
#include "server_responses.h"
#include "server_messages.h"
#include "engine/library/sampling/sampler_records.h"

#define BAD_REQUEST HTTP_BAD_REQUEST
#define DIGITS "0123456789"
#define HEAD_ROOM 256u                              /* before a whole body: its head is put in front of it */
#define BUFFER_FIRST_BYTES 4096u
#define GROWTH 2u                                   /* a buffer that grows doubles */
#define NUMBER_TEXT_CAPACITY 32u                    /* the longest number a member of ours can be */
#define DECIMAL_BASE 10
#define MOST_TEMPERATURE 2.0
#define ESCAPED_BYTES_PER_BYTE 6u                   /* \u00XX */
#define QUOTE_BYTES 2u
#define REPLACEMENT "\xEF\xBF\xBD"                  /* U+FFFD */
#define REPLACEMENT_BYTES 3u
#define SCRATCH_BYTES_PER_BODY_BYTE 4u              /* the keys of every open object sorted, and strings unescaped */
#define SCRATCH_SLACK_BYTES 4096u
#define REASONING_END "</think>"
#define STREAM_HEAD "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n" \
                    "Connection: close\r\n\r\n"
/* The DSML of encoding.py's parse_tool_calls, its tokens as it searches them. */
#define DSML "｜DSML｜"
#define REFERENCE_BLOCK_SPACES \
	"\n\n" /* what the reference trims before a block */
#define CALLS_BEGIN            \
	REFERENCE_BLOCK_SPACES \
	"<" DSML " calls" /* tool_calls_start_token: ends the content */
#define CALLS_END "</" DSML " calls>"
#define INVOKE_BEGIN "<" DSML " invoke"
#define INVOKE_END "</" DSML " invoke"
#define PARAMETER_BEGIN "<" DSML " parameter"
#define PARAMETER_END "/" DSML " parameter"
#define NAME_OPENING "name=\""
#define NAME_CLOSING "\">"
#define NAMESPACE_SEPARATOR "::"
#define PARAMETER_NAME_OPENING " name=\""
#define STRING_TRUE "\" string=\"true\">"
#define STRING_FALSE "\" string=\"false\">"
#define PARAMETER_VALUE_END '<'
#define PYTHON_ASCII_SPACES " \t\n\r\f\v\x1c\x1d\x1e\x1f"
/* The tolerant reading (chat_completions.h): a head's spaces, bars and words, a tag's marks. */
#define MARKER_SPACES " \t\r\n"
#define WIDE_BAR "\xEF\xBD\x9C" /* U+FF5C */
#define NARROW_BAR '|'
#define MARKER_WORD "dsml"
#define HEAD_OPENING '<'
#define HEAD_CLOSING_MARK '/'
#define TAG_END '>'
#define ATTRIBUTE_EQUALS '='
#define DOUBLE_QUOTE '"'
#define SINGLE_QUOTE '\''
#define LINE_FEED '\n'
#define LINE_FEEDS_TRIMMED 2u /* before a block: the reference's two, at most */
#define NAME_WORD "name"
#define STRING_WORD "string"
#define EXACT_NAME_OPENING " name=\""
#define EXACT_STRING_OPENING "\" string=\""
#define EXACT_TAG_CLOSING "\">"
#define WORD_TRUE "true" /* the string attribute: a JSON string */
#define DIGIT_TRUE "1"
#define WORD_FALSE "false" /* as written */
#define DIGIT_FALSE "0"
#define FOLDED_SEPARATOR \
	'_' /* '-' reads as '_' when names are compared folded */
#define UNFOLDED_SEPARATOR '-'
#define JSON_FIRST_BYTES \
	"{[-0123456789tfn" /* a number, true, false, null, an object or an array */
#define JSON_UNQUOTED_REFUSED "NI" /* NaN, Infinity */
#define TOOL_NAME_MOST_BYTES \
	1024u /* a declared name unescaped; a longer one is not kept, and unknown names are then not counted */
#define HEAD_SPACE_RUNS \
	5u /* after "<", the closing mark, each bar and the word */
/* A head at its longest: "<", the closing mark, two bars, the word, the longest keyword and its runs of spaces. */
#define MARKER_HEAD_MOST_BYTES                                             \
	(sizeof("</" WIDE_BAR MARKER_WORD WIDE_BAR "function_calls") - 1 + \
	 (size_t)(CHAT_MARKER_SPACE_MAX * HEAD_SPACE_RUNS))
_Static_assert(CHAT_MARKER_SPACE_MAX + MARKER_HEAD_MOST_BYTES <=
		       CHAT_MOST_STOP_BYTES,
	       "a held head and the spaces before it leave a token's room");
/* While a block is read, the last bytes of the content given out, at the end of held: a head the next text between
 * tags completes begins in them. */
#define JOIN_TAIL_BYTES (CHAT_MARKER_SPACE_MAX + MARKER_HEAD_MOST_BYTES)
_Static_assert(JOIN_TAIL_BYTES + CHAT_UTF8_MOST_BYTES < CHAT_HELD_CAPACITY,
	       "the given tail leaves held room to give text out");
#define TOOL_NAME_NOT_KEPT \
	UINT32_MAX /* the length of a declared name too long to keep: the names are then not whole */
#define REQUEST_MOST_BYTES \
	1128u /* v0.0.5-alpha's: every connection and live record holds a request */
#define ANSWER_MOST_BYTES \
	10304u /* v0.0.5-alpha's: every connection holds an answer */
_Static_assert(
	sizeof(struct chat_request) <= REQUEST_MOST_BYTES &&
		sizeof(struct chat_answer) <= ANSWER_MOST_BYTES,
	"a request or answer field grows every connection and live record");

static const struct { uint32_t status; const char *reason, *message;
} statuses[] = {
	{ HTTP_OK, "OK", "" },
	{ HTTP_BAD_REQUEST, "Bad Request",
	  "the request is not one this server reads" },
	{ HTTP_UNAUTHORIZED, "Unauthorized",
	  "a valid key is required: Authorization: Bearer <key>" },
	{ HTTP_FORBIDDEN, "Forbidden", "this user may not read statistics" },
	{ HTTP_NOT_FOUND, "Not Found",
	  "no such path: /health, /v1/models and /v1/chat/completions are served" },
	{ HTTP_METHOD_NOT_ALLOWED, "Method Not Allowed",
	  "the path is not served for this method" },
	{ HTTP_REQUEST_TIMEOUT, "Request Timeout",
	  "the request did not arrive in time" },
	{ HTTP_LENGTH_REQUIRED, "Length Required",
	  "a body is sent with a Content-Length" },
	{ HTTP_CONTENT_TOO_LARGE, "Content Too Large",
	  "the body is larger than this server takes" },
	{ HTTP_EXPECTATION_FAILED, "Expectation Failed",
	  "only Expect: 100-continue is taken" },
	{ HTTP_FIELDS_TOO_LARGE, "Request Header Fields Too Large",
	  "the request's head is larger than taken" },
	{ HTTP_SERVER_ERROR, "Internal Server Error",
	  "the server could not answer" },
	{ HTTP_NOT_IMPLEMENTED, "Not Implemented",
	  "Transfer-Encoding is not taken: send a Content-Length" },
	{ HTTP_UNAVAILABLE, "Service Unavailable",
	  "the queue of requests is full: retry later" },
	{ HTTP_VERSION_NOT_SUPPORTED, "HTTP Version Not Supported",
	  "only HTTP/1.1 and HTTP/1.0 are served" }
};

char *chat_buffer_room(struct chat_buffer *buffer, size_t more)
{
    if (!buffer->is_short && more > buffer->capacity - buffer->count) {
	    const bool fits = more <= CHAT_MOST_OUTPUT_BYTES - buffer->count;
	    size_t capacity = buffer->capacity < BUFFER_FIRST_BYTES ?
				      BUFFER_FIRST_BYTES :
				      buffer->capacity;
	    while (fits && capacity < buffer->count + more) {
		    capacity = capacity < CHAT_MOST_OUTPUT_BYTES / GROWTH ?
				       capacity * GROWTH :
				       CHAT_MOST_OUTPUT_BYTES;
	    }
	    char *const bytes = !fits || FAULT_POINT(chat_memory) ?
					NULL :
					realloc(buffer->bytes, capacity);
	    buffer->is_short = bytes == NULL;
	    buffer->bytes = bytes != NULL ? bytes : buffer->bytes;
	    buffer->capacity = bytes != NULL ? capacity : buffer->capacity;
    }
    return buffer->is_short || buffer->bytes == NULL ?
		   NULL :
		   buffer->bytes + buffer->count;
}

void chat_buffer_write(struct chat_buffer *buffer, const void *bytes,
		       size_t count)
{
	char *const room = chat_buffer_room(buffer, count);
	if (room != NULL && count != 0) {
		memcpy(room, bytes, count);
		buffer->count += count;
	}
}

static void buffer_text(struct chat_buffer *buffer, const char *text)
{
	chat_buffer_write(buffer, text, strlen(text));
}

/* The bytes as a JSON string. */
static void buffer_string(struct chat_buffer *buffer, const char *bytes,
			  size_t count)
{
	if (count >= CHAT_MOST_OUTPUT_BYTES) {
		buffer->is_short = true;
		return;
	}
	/* The named output cap proves this product and sum fit size_t. */
	const size_t most = count * ESCAPED_BYTES_PER_BYTE + QUOTE_BYTES;
	char *const room = chat_buffer_room(buffer, most);
	buffer->is_short = buffer->is_short || room == NULL;
	if (room != NULL) {
		struct json_writer writer = { room, most, 0 };
		json_write_string(&writer, (struct json_bytes){ bytes, count });
		buffer->count += writer.length;
	}
}

static void buffer_number(struct chat_buffer *buffer, uint64_t number)
{
    char digits[NUMBER_TEXT_CAPACITY] = {0};
    const int length = snprintf(digits, sizeof(digits), "%" PRIu64, number);
    chat_buffer_write(buffer, digits, length > 0 ? (size_t)length : 0);
}

void chat_buffer_free(struct chat_buffer *buffer)
{
    free(buffer->bytes);
    *buffer = (struct chat_buffer){ 0 };
}

/* ---- the request ------------------------------------------------------------------------------------------- */

enum member {
	MEMBER_MODEL,
	MEMBER_MESSAGES,
	MEMBER_STREAM,
	MEMBER_STREAM_OPTIONS,
	MEMBER_MAX_TOKENS,
	MEMBER_MAX_COMPLETION_TOKENS,
	MEMBER_TEMPERATURE,
	MEMBER_TOP_P,
	MEMBER_TOP_K,
	MEMBER_SEED,
	MEMBER_STOP,
	MEMBER_N,
	MEMBER_REASONING_EFFORT,
	MEMBER_THINKING,
	MEMBER_TOOLS,
	MEMBER_TOOL_CHOICE,
	MEMBER_RESPONSE_FORMAT,
	MEMBER_STORE,
	MEMBER_PARALLEL_TOOL_CALLS,
	MEMBER_SERVICE_TIER,
	MEMBER_PROMPT_CACHE_RETENTION,
	MEMBER_USER,
	MEMBER_METADATA,
	MEMBER_PROMPT_CACHE_KEY,
	MEMBER_DRAFT_USAGE,
	MEMBER_CACHE_SALT,
	MEMBER_SAFETY_IDENTIFIER,
	MEMBER_SESSION_ID,
	MEMBER_CONVERSATION_ID,
	MEMBER_USER_ID,
	MEMBER_COUNT
};
static const char *const member_names[MEMBER_COUNT] = { "model",
							"messages",
							"stream",
							"stream_options",
							"max_tokens",
							"max_completion_tokens",
							"temperature",
							"top_p",
							"top_k",
							"seed",
							"stop",
							"n",
							"reasoning_effort",
							"thinking",
							"tools",
							"tool_choice",
							"response_format",
							"store",
							"parallel_tool_calls",
							"service_tier",
							"prompt_cache_retention",
							"user",
							"metadata",
							"prompt_cache_key",
							"anemone_draft_usage",
							"cache_salt",
							"safety_identifier",
							"session_id",
							"conversation_id",
							"user_id" };
static const char *const legacy_function_names[] = {"functions", "function_call"};
/* 1017 encoding.py: low 50, high 75, max 100. The other API names are compatibility aliases, not reference names. */
static const char *const effort_names[] = {"none", "off", "minimal", "low", "medium", "high", "xhigh", "max"};
static const uint32_t efforts[] = { 0,	0,  50, 50,
				    75, 75, 75, DS41F_CHAT_TEMPLATE_HIGHEST_EFFORT };
static const char *const draft_usage_names[] = {"compact", "none", "full"};
static const char *const unsupported_control_names[] = {
    "thinking_token_budget", "thinking_budget", "thinking_budget_tokens", "priority"};
#define LEGACY_FUNCTIONS_REFUSED "the legacy functions API is not supported by this server: declare tools"

struct reading {
    struct json_scratch scratch;
    struct chat_request *request;
    struct chat_refusal *refusal;
    const char *model_alias;                     /* NULL: none */
    bool declines_tools;                         /* tool_choice "none": the tools are not rendered */
    struct json_value thinking_key;              /* mode resolved after effort, independent of JSON member order */
    bool enables_thinking, disables_thinking;
};

/* Refuses the request (the first refusal stays) for the member whose key is key. */
static void refuse(struct reading *reading, uint32_t status, const char *message, struct json_value key)
{
    if (reading->refusal->status == 0) {
	    *reading->refusal = (struct chat_refusal){ status, message, NULL,
						       key.begin, key.end };
    }
}

static uint32_t name_index(struct reading *reading, struct json_value key, const char *const *names, uint32_t count)
{
    uint32_t index = 0;
    while (index < count &&
	   !json_string_is(&reading->scratch, key, names[index])) {
	    index++;
    }
    return index;
}

/* Whether value is a number, an integer of digits alone when integer_out is not NULL; its value then. */
static bool number_of(struct json_value value, double *number_out, uint64_t *integer_out)
{
    char text[NUMBER_TEXT_CAPACITY] = {0};
    char *end = NULL;
    const size_t length = value.end - value.begin;
    if (length == 0 || length >= sizeof(text) || (value.text[value.begin] != '-' && (value.text[value.begin] < '0' ||
                                                                                     value.text[value.begin] > '9'))) {
        return false;
    }
    memcpy(text, value.text + value.begin, length);
    errno = 0;
    if (integer_out != NULL) {
        const unsigned long long integer = strspn(text, DIGITS) == length ? strtoull(text, &end, DECIMAL_BASE) : 0;
        *integer_out = integer;
        return end == text + length && errno == 0;
    }
    *number_out = strtod(text, &end);
    return end == text + length && errno == 0 && isfinite(*number_out);
}

static bool is_null(struct json_value value)
{
    return value.text[value.begin] == 'n';
}

/* Scan source objects without decoding images; all count and byte limits precede any image work. */
static void image_part_read(struct reading *reading, struct json_value key, struct json_value source)
{
    struct json_value url = source, member = {0}, value = {0};
    if (source.text != NULL && source.text[source.begin] == '{') {
        url = (struct json_value){0};
	for (size_t cursor = source.begin;
	     json_next(source, &cursor, &member, &value);) {
		if (json_string_is(&reading->scratch, member, "url")) {
			url = value;
		} else if (!json_string_is(&reading->scratch, member,
					   "detail") &&
			   !is_null(value)) {
			refuse(reading, BAD_REQUEST,
			       "image_url supports only url and detail",
			       member);
		}
	}
    }
    if (url.text == NULL || url.text[url.begin] != '"') {
        refuse(reading, BAD_REQUEST, "image_url has a string URL", key);
        if (reading->refusal->code == NULL) {
		reading->refusal->code = CHAT_IMAGE_URL_INVALID;
	}
        return;
    }
    if (reading->request->image_count == IMAGE_MAX_IMAGES) {
	    refuse(reading, BAD_REQUEST,
		   "the request exceeds ANEMONE_IMAGE_MAX_IMAGES", key);
	    if (reading->refusal->code == NULL) {
		    reading->refusal->code = CHAT_IMAGE_COUNT_LIMIT;
	    }
	    return;
    }
    const size_t mark = reading->scratch.used;
    const struct json_bytes bytes = json_unescape(&reading->scratch, url);
    size_t decoded_bytes = 0;
    struct chat_refusal refusal = { 0 };
    const uint32_t status = chat_image_url_decode(bytes.data, bytes.count, NULL,
						  0, &decoded_bytes, &refusal);
    reading->scratch.used = mark;
    if (status != 0) {
        refuse(reading, status, refusal.message, key);
        if (reading->refusal->code == NULL) {
            reading->refusal->code = refusal.code;
        }
    } else {
        reading->request->image_count++;
    }
}

static void parts_read(struct reading *reading, struct json_value key, struct json_value parts, bool images_allowed)
{
    struct json_value part = {0}, part_key = {0}, value = {0};
    for (size_t cursor = parts.begin; reading->refusal->status == 0 &&
				      json_next(parts, &cursor, NULL, &part);) {
	    struct json_value type = { 0 }, source = { 0 };
	    if (part.text[part.begin] != '{') {
		    refuse(reading, BAD_REQUEST, "a content part is an object",
			   key);
		    return;
	    }
	    for (size_t inner = part.begin;
		 json_next(part, &inner, &part_key, &value);) {
		    if (json_string_is(&reading->scratch, part_key, "type")) {
			    type = value;
		    } else if (json_string_is(&reading->scratch, part_key,
					      "image_url")) {
			    source = value;
		    } else if (json_string_is(&reading->scratch, part_key,
					      "cache_control") &&
			       !is_null(value)) {
			    refuse(reading, BAD_REQUEST,
				   "content cache_control is not supported: cache lifetime is server managed",
				   part_key);
		    }
	    }
	    if (reading->refusal->status != 0) {
		    return;
	    }
	    if (json_string_is(&reading->scratch, type, "image_url")) {
		    if (!images_allowed) {
			    refuse(reading, BAD_REQUEST,
				   "images are allowed only in user messages and tool results",
				   key);
			    if (reading->refusal->code == NULL) {
				    reading->refusal->code =
					    CHAT_IMAGE_ROLE_INVALID;
			    }
		    } else {
			    image_part_read(reading, key, source);
		    }
	    } else if (!json_string_is(&reading->scratch, type, "text")) {
		    refuse(reading, BAD_REQUEST,
			   "a content part is text or image_url", key);
	    }
    }
}

static void messages_read(struct reading *reading, struct json_value key, struct json_value messages)
{
    static const char *const known[] = {"role", "content", "reasoning_content", "name", "tool_calls", "tool_call_id"};
    static const char *const roles[] = {"system", "developer", "user", "assistant", "tool"};
    struct json_value message = {0}, member = {0}, value = {0};
    uint32_t message_count = 0;
    for (size_t cursor = messages.begin;
	 json_next(messages, &cursor, NULL, &message);) {
	    if (++message_count > CHAT_MOST_MESSAGES ||
		message.text[message.begin] != '{') {
		    refuse(reading, BAD_REQUEST,
			   "messages is an array of up to 4,096 objects", key);
		    return;
	    }
	    bool images_allowed = false;
	    for (size_t inner = message.begin;
		 json_next(message, &inner, &member, &value);) {
		    if (json_string_is(&reading->scratch, member, "role")) {
			    images_allowed = json_string_is(&reading->scratch,
							    value, "user") ||
					     json_string_is(&reading->scratch,
							    value, "tool");
		    }
	    }
	    for (size_t inner = message.begin;
		 json_next(message, &inner, &member, &value);) {
		    const uint32_t index =
			    name_index(reading, member, known,
				       sizeof(known) / sizeof(known[0]));
		    if (index == 0 &&
			name_index(reading, value, roles,
				   sizeof(roles) / sizeof(roles[0])) ==
				sizeof(roles) / sizeof(roles[0])) {
			    refuse(reading, BAD_REQUEST,
				   "a message's role is system, developer, user, assistant or tool",
				   member);
		    } else if (index == 1 && value.text[value.begin] == '[') {
			    parts_read(reading, member, value, images_allowed);
		    } else if (index == sizeof(known) / sizeof(known[0]) &&
			       json_is_truthy(&reading->scratch, value)) {
			    refuse(reading, BAD_REQUEST,
				   "this member of a message is not supported by this server",
				   member);
		    }
	    }
    }
    if (message_count == 0) {
        refuse(reading, BAD_REQUEST, "messages holds at least one message", key);
    }
    reading->request->messages_begin = messages.begin;
    reading->request->messages_end = messages.end;
}

static void stop_read(struct reading *reading, struct json_value key, struct json_value stop)
{
	struct chat_request *const request = reading->request;
	const size_t mark = reading->scratch.used;
	const struct json_bytes bytes =
		stop.text[stop.begin] == '"' ?
			json_unescape(&reading->scratch, stop) :
			(struct json_bytes){ NULL, 0 };
	if (bytes.count == 0 || bytes.count > CHAT_MOST_STOP_BYTES ||
	    request->stop_count == CHAT_MOST_STOPS) {
		refuse(reading, BAD_REQUEST,
		       "stop is a string or up to 4 strings, each of 1 to 256 bytes",
		       key);
	} else {
		memcpy(request->stops[request->stop_count], bytes.data,
		       bytes.count);
		request->stop_byte_counts[request->stop_count++] =
			(uint32_t)bytes.count;
	}
    reading->scratch.used = mark;
}

/* Whether every member is the named boolean or word; an empty object keeps the default. */
static bool object_holds_only(struct reading *reading, struct json_value object, const char *name, const char *word,
                              bool *is_true_out)
{
    struct json_value key = {0}, value = {0};
    bool holds_only = object.text[object.begin] == '{';
    for (size_t cursor = object.begin;
	 holds_only && json_next(object, &cursor, &key, &value);) {
	    const bool is_taken =
		    json_string_is(&reading->scratch, key, name) &&
		    (word != NULL ?
			     json_string_is(&reading->scratch, value, word) :
			     value.text[value.begin] == 't' ||
				     value.text[value.begin] == 'f');
	    *is_true_out = is_taken ? value.text[value.begin] == 't' :
				      *is_true_out;
	    holds_only = is_taken;
    }
    return holds_only;
}

/* DeepSeek's wire mode has a required type and no budget or other controls in the pinned harness dialect. */
static void thinking_read(struct reading *reading, struct json_value key, struct json_value thinking)
{
    struct json_value member = {0}, value = {0};
    bool has_type = false;
    bool valid = thinking.text[thinking.begin] == '{';
    for (size_t cursor = thinking.begin;
	 valid && json_next(thinking, &cursor, &member, &value);) {
	    const bool enabled =
		    json_string_is(&reading->scratch, value, "enabled");
	    const bool disabled =
		    json_string_is(&reading->scratch, value, "disabled");
	    valid = json_string_is(&reading->scratch, member, "type") &&
		    (enabled || disabled);
	    has_type = has_type || valid;
	    reading->enables_thinking = enabled;
	    reading->disables_thinking = disabled;
    }
    reading->thinking_key = key;
    if (!valid || !has_type) {
        refuse(reading, BAD_REQUEST, "thinking must be an object with only type: enabled or disabled", key);
    }
}

/* Only ordinary function tools are supported; strict schema-constrained generation is not implemented. */
static void tools_read(struct reading *reading, struct json_value key, struct json_value tools)
{
    struct json_value tool = {0}, member = {0}, value = {0}, field = {0}, setting = {0};
    for (size_t cursor = tools.begin; json_next(tools, &cursor, NULL, &tool);) {
	    for (size_t inner = tool.begin;
		 tool.text[tool.begin] == '{' &&
		 json_next(tool, &inner, &member, &value);) {
		    if (json_string_is(&reading->scratch, member, "type") &&
			!json_string_is(&reading->scratch, value, "function")) {
			    refuse(reading, BAD_REQUEST,
				   "tools supports function tools only", key);
		    } else if (json_string_is(&reading->scratch, member,
					      "cache_control") &&
			       !is_null(value)) {
			    refuse(reading, BAD_REQUEST,
				   "tools cache_control is not supported: cache lifetime is server managed",
				   key);
		    } else if (json_string_is(&reading->scratch, member,
					      "function") &&
			       value.text[value.begin] == '{') {
			    for (size_t function_cursor = value.begin;
				 json_next(value, &function_cursor, &field,
					   &setting);) {
				    if (json_string_is(&reading->scratch, field,
						       "strict") &&
					!is_null(setting) &&
					setting.text[setting.begin] != 'f') {
					    refuse(reading, BAD_REQUEST,
						   "tools.function.strict must be false or null: strict schema "
						   "generation is not supported",
						   key);
				    }
			    }
		    }
	    }
    }
}

/* One member of the request; a null one keeps the default. */
static void member_read(struct reading *reading, struct json_value key, struct json_value value, const char *model_name,
                        uint64_t *most_tokens)
{
	struct chat_request *const request = reading->request;
	struct json_scratch *const scratch = &reading->scratch;
	const char kind = value.text[value.begin];
	const uint32_t member =
		name_index(reading, key, member_names, MEMBER_COUNT);
	const uint32_t effort_count = sizeof(efforts) / sizeof(efforts[0]);
	const uint32_t effort =
		member == MEMBER_REASONING_EFFORT ?
			name_index(reading, value, effort_names, effort_count) :
			effort_count;
	double number = 0.0;
	uint64_t integer = 0;
	struct json_value element = { 0 };
	size_t tools_cursor = value.begin; /* json_next's cursor for the tools array's first element */
	if (member == MEMBER_THINKING) {
		thinking_read(reading, key, value);
		return;
	}
    if (is_null(value)) {
        return;
    }
    if (member == MEMBER_MODEL) {
	    const bool canonical = json_string_is(scratch, value, model_name);
	    request->uses_model_alias =
		    !canonical && reading->model_alias != NULL &&
		    json_string_is(scratch, value, reading->model_alias);
	    if (!canonical && !request->uses_model_alias) {
		    refuse(reading, HTTP_NOT_FOUND,
			   "no such model: /v1/models lists this server's",
			   key);
		    reading->refusal->code = "model_not_found";
	    }
    } else if (member == MEMBER_MESSAGES && kind != '[') {
        refuse(reading, BAD_REQUEST, "messages is an array of messages", key);
    } else if (member == MEMBER_MESSAGES) {
        messages_read(reading, key, value);
    } else if (member == MEMBER_STREAM && kind != 't' && kind != 'f') {
        refuse(reading, BAD_REQUEST, "stream is true or false", key);
    } else if (member == MEMBER_STREAM) {
        request->is_streamed = kind == 't';
    } else if (member == MEMBER_STREAM_OPTIONS &&
               !object_holds_only(reading, value, "include_usage", NULL, &request->includes_usage)) {
        refuse(reading, BAD_REQUEST, "stream_options holds include_usage, true or false, alone", key);
    } else if (member == MEMBER_MAX_TOKENS || member == MEMBER_MAX_COMPLETION_TOKENS) {
        if (!number_of(value, NULL, &integer) || integer == 0 || integer > UINT32_MAX) {
            refuse(reading, BAD_REQUEST, "max_tokens is an integer from 1 to 4294967295", key);
        }
        *most_tokens = integer != 0 && integer < *most_tokens ? integer : *most_tokens;
    } else if (member == MEMBER_TEMPERATURE) {
        if (!number_of(value, &number, NULL) || number < 0.0 || number > MOST_TEMPERATURE) {
            refuse(reading, BAD_REQUEST, "temperature is a number from 0 to 2", key);
        }
        request->temperature = (float)number;
    } else if (member == MEMBER_TOP_P) {
        if (!number_of(value, &number, NULL) || !((float)number > 0.0f && number <= 1.0)) {
            refuse(reading, BAD_REQUEST, "top_p is a number above 0 and at most 1", key);
        }
        request->top_p = (float)number;
    } else if (member == MEMBER_TOP_K) {
	    if (json_is_truthy(&reading->scratch, value)) {
		    if (!number_of(value, NULL, &integer) ||
			integer > SAMPLER_MOST_VOCABULARY) {
			    refuse(reading, BAD_REQUEST,
				   "top_k is an integer from 0 to the sampler vocabulary bound",
				   key);
		    } else {
			    request->top_k = (uint32_t)integer;
		    }
	    }
    } else if (member == MEMBER_SEED) {
	    if (!number_of(value, NULL, &integer)) {
		    refuse(reading, BAD_REQUEST,
			   "seed is an integer from 0 to 18446744073709551615",
			   key);
	    }
	    request->seed = integer;
	    request->is_seeded = true;
    } else if (member == MEMBER_STOP && kind == '[') {
	    for (size_t cursor = value.begin;
		 json_next(value, &cursor, NULL, &element);) {
		    stop_read(reading, key, element);
	    }
    } else if (member == MEMBER_STOP) {
	    stop_read(reading, key, value);
    } else if (member == MEMBER_N &&
	       (!number_of(value, NULL, &integer) || integer != 1)) {
	    refuse(reading, BAD_REQUEST,
		   "n above 1 is not supported by this server: it generates one choice",
		   key);
    } else if (member == MEMBER_REASONING_EFFORT) {
	    if (effort == effort_count &&
		(!number_of(value, NULL, &integer) ||
		 integer > DS41F_CHAT_TEMPLATE_HIGHEST_EFFORT)) {
		    refuse(reading, BAD_REQUEST,
			   "reasoning_effort is none, off, minimal, low, medium, high, xhigh, max or an "
			   "integer from 0 to 100",
			   key);
	    }
	request->reasoning_effort = effort < effort_count ? efforts[effort] : (uint32_t)integer;
    } else if (member == MEMBER_DRAFT_USAGE) {
	    const uint32_t mode_count =
		    sizeof(draft_usage_names) / sizeof(draft_usage_names[0]);
	    const uint32_t mode =
		    name_index(reading, value, draft_usage_names, mode_count);
	    if (mode == mode_count) {
		    refuse(reading, BAD_REQUEST,
			   "anemone_draft_usage is none, compact or full", key);
	    }
	    request->draft_usage = mode;
    } else if (member == MEMBER_TOOLS && kind != '[') {
	    refuse(reading, BAD_REQUEST, "tools is an array of tools", key);
    } else if (member == MEMBER_TOOLS &&
	       json_next(value, &tools_cursor, NULL, &element)) {
	    tools_read(reading, key, value);
	    request->tools_declared = true;
	    request->tools_begin =
		    value.begin; /* an empty array declares no tool, as in encoding.py */
	    request->tools_end = value.end;
    } else if (member == MEMBER_TOOL_CHOICE &&
	       !json_string_is(scratch, value, "none") &&
	       !json_string_is(scratch, value, "auto")) {
	    refuse(reading, BAD_REQUEST,
		   "tool_choice is auto or none: the model decides whether to call",
		   key);
    } else if (member == MEMBER_TOOL_CHOICE) {
	    reading->declines_tools = json_string_is(scratch, value, "none");
    } else if (member == MEMBER_RESPONSE_FORMAT &&
	       !object_holds_only(reading, value, "type", "text",
				  &(bool){ false })) {
	    refuse(reading, BAD_REQUEST,
		   "response formats other than text are not supported by this server yet",
		   key);
    } else if (member == MEMBER_STORE && kind != 'f') {
	    refuse(reading, BAD_REQUEST,
		   "store must be false: stored responses are not supported",
		   key);
    } else if (member == MEMBER_PARALLEL_TOOL_CALLS && kind != 't') {
	    refuse(reading, BAD_REQUEST,
		   "parallel_tool_calls must be true: limiting generation to one call is not "
		   "supported",
		   key);
    } else if (member == MEMBER_SERVICE_TIER &&
	       !json_string_is(scratch, value, "auto") &&
	       !json_string_is(scratch, value, "default")) {
	    refuse(reading, BAD_REQUEST,
		   "service_tier supports auto or default only", key);
    } else if (member == MEMBER_PROMPT_CACHE_RETENTION) {
	    refuse(reading, BAD_REQUEST,
		   "prompt_cache_retention is not supported: cache lifetime is server managed",
		   key);
    } else if (member == MEMBER_COUNT &&
	       (json_is_truthy(scratch, value) ||
		name_index(reading, key, unsupported_control_names,
			   sizeof(unsupported_control_names) /
				   sizeof(unsupported_control_names[0])) <
			sizeof(unsupported_control_names) /
				sizeof(unsupported_control_names[0]))) {
	    const uint32_t legacy_count = sizeof(legacy_function_names) /
					  sizeof(legacy_function_names[0]);
	    refuse(reading, BAD_REQUEST,
		   name_index(reading, key, legacy_function_names,
			      legacy_count) < legacy_count ?
			   LEGACY_FUNCTIONS_REFUSED :
			   "this member is not supported by this server",
		   key);
    }
}

uint32_t chat_request_read(const char *body, size_t byte_count,
			   const char *model_name, const char *model_alias,
			   const struct chat_request *defaults,
			   struct chat_request *request_out,
			   struct chat_refusal *refusal_out)
{
	struct chat_refusal refusal = { 0 };
	struct chat_request request = { 0 };
	struct json_value document = { 0 }, key = { 0 }, value = { 0 };
	struct reading reading = { .request = &request,
				   .refusal = &refusal,
				   .model_alias = model_alias };
	uint64_t most_tokens = UINT64_MAX;
	if (body == NULL || model_name == NULL || defaults == NULL ||
	    request_out == NULL || refusal_out == NULL ||
	    byte_count > CHAT_MOST_BODY_BYTES ||
	    defaults->draft_usage > CHAT_DRAFT_USAGE_FULL) {
		return HTTP_SERVER_ERROR;
	}
	request = (struct chat_request){
		.seed = defaults->seed,
		.max_tokens = defaults->max_tokens,
		.reasoning_effort = defaults->reasoning_effort,
		.draft_usage = defaults->draft_usage,
		.temperature = defaults->temperature,
		.top_p = defaults->top_p,
		.top_k = defaults->top_k,
		.is_seeded = defaults->is_seeded,
		.is_calls_reference = defaults->is_calls_reference
	};
	reading.scratch.capacity =
		SCRATCH_BYTES_PER_BODY_BYTE * byte_count + SCRATCH_SLACK_BYTES;
	reading.scratch.bytes = FAULT_POINT(chat_memory) ?
					NULL :
					malloc(reading.scratch.capacity);
	if (reading.scratch.bytes == NULL) {
		refusal.status = HTTP_SERVER_ERROR;
	} else if (json_read_document(body, byte_count, &document) !=
			   JSON_DOCUMENT_VALID ||
		   body[document.begin] != '{') {
		refuse(&reading, BAD_REQUEST,
		       "the body is a JSON object of UTF-8 text, at most 128 levels deep",
		       key);
	} else if (json_has_repeated_key(&reading.scratch, document)) {
		refuse(&reading, BAD_REQUEST,
		       "an object of the body holds a key twice", key);
	}
	for (size_t cursor = document.begin;
	     refusal.status == 0 && reading.scratch.bytes != NULL &&
	     json_next(document, &cursor, &key, &value);) {
		member_read(&reading, key, value, model_name, &most_tokens);
	}
    if (reading.disables_thinking) {
        request.reasoning_effort = 0;
    } else if (reading.enables_thinking && request.reasoning_effort == 0) {
        refuse(&reading, BAD_REQUEST, "thinking enabled requires a positive reasoning_effort in the request or "
                                    "server default (--reasoning_effort)", reading.thinking_key);
    }
    if (refusal.status == 0 && reading.scratch.is_short) {
	    refusal.status = HTTP_SERVER_ERROR;
    } else if (refusal.status == 0 && request.messages_end == 0) {
        refuse(&reading, BAD_REQUEST, "messages is required", (struct json_value){0});
    } else if (refusal.status == 0 && !request.is_seeded &&
	       (FAULT_POINT(chat_random) ||
		getrandom(&request.seed, sizeof(request.seed), GRND_NONBLOCK) !=
			(ssize_t)sizeof(request.seed))) {
	    LOG_ERROR(
		    "chat: the system's random source gave no seed for a request that names none");
	    refusal.status = HTTP_SERVER_ERROR;
    }
    free(reading.scratch.bytes);
    request.max_tokens = most_tokens != UINT64_MAX ? (uint32_t)most_tokens : request.max_tokens;
    request.tools_end = reading.declines_tools ? request.tools_begin : request.tools_end;
    if (refusal.status == 0) {
        *request_out = request;
    } else {
        *refusal_out = refusal;
    }
    return refusal.status;
}

void chat_tool_names_kept(const char *tools, size_t count,
			  struct chat_buffer *names)
{
	char unescaped[TOOL_NAME_MOST_BYTES];
	const struct json_scratch empty = { unescaped, sizeof(unescaped), 0,
					    false };
	struct json_scratch scratch = empty;
	const struct json_value document = { tools, 0, count };
	struct json_value tool = { 0 }, member = { 0 }, value = { 0 },
			  field = { 0 }, setting = { 0 };
	for (size_t cursor = 0;
	     count != 0 && json_next(document, &cursor, NULL, &tool);) {
		for (size_t inner = tool.begin;
		     tool.text[tool.begin] == '{' &&
		     json_next(tool, &inner, &member, &value);) {
			scratch = empty;
			if (!json_string_is(&scratch, member, "function") ||
			    value.text[value.begin] != '{') {
				continue;
			}
			for (size_t function_cursor = value.begin;
			     json_next(value, &function_cursor, &field,
				       &setting);) {
				scratch = empty;
				if (!json_string_is(&scratch, field,
						    NAME_WORD) ||
				    setting.text[setting.begin] !=
					    DOUBLE_QUOTE) {
					continue;
				}
				scratch = empty;
				const struct json_bytes name =
					json_unescape(&scratch, setting);
				const uint32_t length =
					scratch.is_short ? TOOL_NAME_NOT_KEPT :
							   (uint32_t)name.count;
				chat_buffer_write(names, &length,
						  sizeof(length));
				chat_buffer_write(
					names, name.data,
					scratch.is_short ? 0 : name.count);
			}
		}
	}
}

/* ---- responses --------------------------------------------------------------------------------------------- */

static uint32_t status_place(uint32_t status)
{
    uint32_t place = 0;
    while (place < sizeof(statuses) / sizeof(statuses[0]) && statuses[place].status != status) {
        place++;
    }
    if (place == sizeof(statuses) / sizeof(statuses[0])) {
	    for (place = 0; place < sizeof(statuses) / sizeof(statuses[0]);
		 place++) {
		    if (statuses[place].status == HTTP_SERVER_ERROR) {
			    break;
		    }
	    }
    }
    return place;
}

/* An empty response whose body begins behind room for its head. */

static void body_begin(struct chat_buffer *output)
{
	*output = (struct chat_buffer){ output->bytes, 0, 0, output->capacity,
					false };
	output->count =
		chat_buffer_room(output, HEAD_ROOM) != NULL ? HEAD_ROOM : 0;
}

/* The head of the response whose body follows HEAD_ROOM, in front of it. */
static void head_put(struct chat_buffer *output, uint32_t status,
		     const char *extra)
{
    char head[HEAD_ROOM] = {0};
    const uint32_t place = status_place(status);
    const int length = output->is_short ? -1
        : snprintf(head, sizeof(head), "HTTP/1.1 %" PRIu32 " %s\r\nContent-Type: application/json\r\nContent-Length: "
                   "%zu\r\nConnection: close\r\n%s\r\n", statuses[place].status, statuses[place].reason,
                   output->count - HEAD_ROOM, extra);
    output->is_short = length <= 0 || (size_t)length > HEAD_ROOM;
    if (!output->is_short) {
        memcpy(output->bytes + HEAD_ROOM - (size_t)length, head, (size_t)length);
        output->sent = HEAD_ROOM - (size_t)length;
    }
}

void chat_json_body_begin(struct chat_buffer *output)
{
	body_begin(output);
}
void chat_json_body_finish(struct chat_buffer *output, uint32_t status,
			   const char *extra)
{
	head_put(output, status, extra);
}

void chat_refusal_write(struct chat_buffer *output,
			const struct chat_refusal *refusal, const char *body,
			uint32_t route)
{
	if (route == CHAT_ROUTE_MESSAGES) {
		server_messages_refuse(output, refusal);
		return;
	}
    const uint32_t status = refusal->status;
    const char *const message = refusal->message != NULL ? refusal->message : statuses[status_place(status)].message;
    body_begin(output);
    buffer_text(output, "{\"error\":{\"message\":");
    buffer_string(output, message, strlen(message));
    buffer_text(output,
		status == HTTP_UNAUTHORIZED ?
			",\"type\":\"authentication_error\",\"param\":" :
		status < HTTP_SERVER_ERROR ?
			",\"type\":\"invalid_request_error\",\"param\":" :
			",\"type\":\"server_error\",\"param\":");
    if (body != NULL && refusal->member_end > refusal->member_begin) {
	    chat_buffer_write(output, body + refusal->member_begin,
			      refusal->member_end - refusal->member_begin);
    } else {
        buffer_text(output, "null");
    }
    buffer_text(output, ",\"code\":");
    if (refusal->code != NULL) {
        buffer_string(output, refusal->code, strlen(refusal->code));
    } else {
        buffer_text(output, "null");
    }
    buffer_text(output, "}}\n");
    head_put(output, status,
	     status == HTTP_UNAUTHORIZED ? "WWW-Authenticate: Bearer\r\n" :
	     status == HTTP_UNAVAILABLE	 ? "Retry-After: " CHAT_RETRY_AFTER
					   "\r\n" :
	     status != HTTP_METHOD_NOT_ALLOWED ? "" :
	     (route == CHAT_ROUTE_COMPLETIONS ||
	      route == CHAT_ROUTE_RESPONSES) ?
						 "Allow: POST\r\n" :
						 "Allow: GET\r\n");
}

void chat_listing_write(struct chat_buffer *output, uint32_t route,
			const char *model_name, const char *model_alias,
			int64_t created_seconds, uint32_t context_token_limit)
{
    const char *const names[] = {model_name, model_alias};
    body_begin(output);
    if (route == CHAT_ROUTE_HEALTH) {
	    buffer_text(output, "{\"status\":\"ok\"}\n");
    } else {
	    buffer_text(output, "{\"object\":\"list\",\"data\":[");
	    for (uint32_t index = 0;
		 index <
		 (model_alias != NULL ? sizeof(names) / sizeof(names[0]) : 1u);
		 index++) {
		    buffer_text(output, index == 0 ? "{\"id\":" : ",{\"id\":");
		    buffer_string(output, names[index], strlen(names[index]));
		    buffer_text(output, ",\"object\":\"model\",\"created\":");
		    buffer_number(output, created_seconds > 0 ?
						  (uint64_t)created_seconds :
						  0);
		    buffer_text(
			    output,
			    ",\"owned_by\":\"anemone\",\"context_length\":");
		    buffer_number(output, context_token_limit);
		    buffer_text(output,
				",\"input_modalities\":[\"text\",\"image\"]}");
	    }
	    buffer_text(output, "]}\n");
    }
    head_put(output, HTTP_OK, "");
}

/* ---- the answer -------------------------------------------------------------------------------------------- */

/* A chunk of a stream up to its delta, which the caller writes; chunk_end writes the finish. */
static void chunk_begin(struct chat_answer *answer)
{
    buffer_text(&answer->output, "data: ");
    chat_buffer_write(&answer->output, answer->prefix, answer->prefix_count);
    buffer_text(&answer->output, "{\"index\":0,\"delta\":");
}

static void chunk_end(struct chat_answer *answer, const char *finish_reason)
{
    buffer_text(&answer->output, finish_reason == NULL ? ",\"finish_reason\":null}]}\n\n" : ",\"finish_reason\":\"");
    if (finish_reason != NULL) {
        buffer_text(&answer->output, finish_reason);
        buffer_text(&answer->output, "\"}]}\n\n");
    }
}

static void draft_counts_write(struct chat_buffer *output,
			       const uint64_t *counts, uint32_t count)
{
    buffer_text(output, "[");
    for (uint32_t index = 0; index < count; index++) {
        buffer_text(output, index != 0 ? "," : "");
        buffer_number(output, counts[index]);
    }
    buffer_text(output, "]");
}

static void draft_statistics_write(struct chat_buffer *output,
				   const struct draft_statistics *statistics,
				   bool full)
{
    buffer_text(output, "{\"drawing_rounds\":");
    buffer_number(output, statistics->drawing_rounds);
    if (full) {
        buffer_text(output, ",\"depth_rounds\":");
	draft_counts_write(output, statistics->depth_rounds,
			   DRAFT_DEPTH_COUNTS);
    }
    buffer_text(output, ",\"sources\":[");
    for (uint32_t source = 0; source < DRAFT_SOURCES; source++) {
	    const struct draft_source_statistics *const value =
		    &statistics->source[source];
	    buffer_text(output, source != 0 ? "," : "");
	    buffer_text(output, "{\"rounds\":");
	    buffer_number(output, value->rounds);
	    buffer_text(output, ",\"proposed\":");
	    draft_counts_write(output, value->proposed, DRAFT_STAT_POSITIONS);
	    buffer_text(output, ",\"reached\":");
	    draft_counts_write(output, value->reached, DRAFT_STAT_POSITIONS);
	    buffer_text(output, ",\"accepted\":");
	    draft_counts_write(output, value->accepted, DRAFT_STAT_POSITIONS);
	    buffer_text(output, "}");
    }
    if (!full) {
        buffer_text(output, "]}");
        return;
    }
    buffer_text(output, "],\"confidence\":[");
    for (uint32_t mode = 0; mode < DRAFT_SAMPLING_MODES; mode++) {
	    buffer_text(output, mode != 0 ? ",[" : "[");
	    for (uint32_t position = 0; position < DRAFT_CALIBRATION_POSITIONS;
		 position++) {
		    buffer_text(output, position != 0 ? ",[" : "[");
		    for (uint32_t bin = 0; bin < DRAFT_CONFIDENCE_BINS; bin++) {
			    const struct draft_outcomes *const value =
				    &statistics->confidence[mode][position][bin];
			    const uint64_t counts[] = { value->produced,
							value->verified,
							value->reached,
							value->matched };
			    buffer_text(output, bin != 0 ? "," : "");
			    draft_counts_write(output, counts,
					       sizeof(counts) /
						       sizeof(counts[0]));
		    }
		    buffer_text(output, "]");
	    }
	    buffer_text(output, "]");
    }
    buffer_text(output, "]}");
}

static void usage_write(struct chat_answer *answer)
{
    buffer_text(&answer->output, "\"usage\":{\"prompt_tokens\":");
    buffer_number(&answer->output, answer->prompt_token_count);
    buffer_text(&answer->output, ",\"completion_tokens\":");
    buffer_number(&answer->output, answer->completion_token_count);
    buffer_text(&answer->output, ",\"total_tokens\":");
    buffer_number(&answer->output, (uint64_t)answer->prompt_token_count + answer->completion_token_count);
    buffer_text(&answer->output, ",\"prompt_tokens_details\":{\"cached_tokens\":");
    buffer_number(&answer->output, answer->cached_token_count);
    buffer_text(&answer->output, "},\"completion_tokens_details\":{\"reasoning_tokens\":");
    buffer_number(&answer->output, answer->reasoning_token_count);
    buffer_text(&answer->output, "}");
    if (answer->has_draft_statistics &&
	answer->request->draft_usage != CHAT_DRAFT_USAGE_NONE) {
	    const bool full = answer->request->draft_usage ==
			      CHAT_DRAFT_USAGE_FULL;
	    buffer_text(&answer->output,
			",\"anemone_draft\":{\"version\":1,\"command\":");
	    draft_statistics_write(&answer->output, &answer->draft_command,
				   full);
	    if (full) {
		    buffer_text(&answer->output, ",\"total\":");
		    draft_statistics_write(&answer->output,
					   &answer->draft_total, true);
	    }
	    buffer_text(&answer->output, "}");
    }
    buffer_text(&answer->output, "}");
}

/* Gives out the first count held bytes as text of the answer's present part, U+FFFD for what is no UTF-8, but for an
 * unfinished sequence at their end unless is_last. The bytes given out. */
static uint32_t text_given_out(struct chat_answer *answer, uint32_t count,
			       bool is_last)
{
	char text[CHAT_HELD_CAPACITY * REPLACEMENT_BYTES];
	size_t length = 0;
	uint32_t position = 0;
	while (position < count) {
		bool is_valid = false;
		const uint32_t bytes = tokenizer_utf8_sequence_bytes(
			answer->held, position, count, &is_valid);
		if (bytes == 0 ||
		    (!is_valid && position + bytes == count && !is_last)) {
			break;
		}
		memcpy(text + length,
		       is_valid ? answer->held + position : REPLACEMENT,
		       is_valid ? bytes : REPLACEMENT_BYTES);
		length += is_valid ? bytes : REPLACEMENT_BYTES;
		position += bytes;
	}
	if (length != 0 && answer->request->is_messages) {
		server_messages_text(answer, text, length);
	} else if (length != 0 && answer->request->is_responses) {
		server_responses_text(answer, text, length);
	} else if (length != 0 && answer->request->is_streamed) {
		chunk_begin(answer);
		buffer_text(&answer->output, answer->is_reasoning ?
						     "{\"reasoning_content\":" :
						     "{\"content\":");
		buffer_string(&answer->output, text, length);
		buffer_text(&answer->output, "}");
		chunk_end(answer, NULL);
	} else if (length != 0) {
		chat_buffer_write(answer->is_reasoning ? &answer->reasoning :
							 &answer->content,
				  text, length);
	}
	return position;
}

/* ---- DSML heads -------------------------------------------------------------------------------------------- */

enum marker_keyword { KEYWORD_BLOCK, KEYWORD_INVOKE, KEYWORD_PARAMETER };
/* The keywords of a head, then the reference's own of each kind (by enum marker_keyword). */
static const struct {
	const char *word;
	uint32_t keyword;
} marker_keywords[] = { { "calls", KEYWORD_BLOCK },
			{ "function_calls", KEYWORD_BLOCK },
			{ "tool_calls", KEYWORD_BLOCK },
			{ "invoke", KEYWORD_INVOKE },
			{ "parameter", KEYWORD_PARAMETER } };
static const char *const reference_keywords[] = { "calls", "invoke",
						  "parameter" };
enum head_match { HEAD_NONE, HEAD_PARTIAL, HEAD_WHOLE };
enum head_step { STEP_FIRST_BAR, STEP_WORD, STEP_SECOND_BAR, STEP_KEYWORD };

struct marker_head {
	size_t begin, end;
	uint32_t keyword; /* enum marker_keyword */
	bool is_closing, is_exact; /* exact: as the reference spells it */
};

static bool is_marker_space(char byte)
{
	return byte != '\0' && strchr(MARKER_SPACES, byte) != NULL;
}

/* The end of the run of marker spaces at at, SIZE_MAX when the run is longer than a head takes. */
static size_t marker_spaces_skipped(const char *text, size_t at, size_t end)
{
	size_t position = at;
	while (position < end && is_marker_space(text[position])) {
		position++;
	}
	return position - at <= CHAT_MARKER_SPACE_MAX ? position : SIZE_MAX;
}

/* How the bytes at at spell word (in any case when folds): whole, its end in *end_out, or cut by the end. */
static uint32_t word_read(const char *text, size_t at, size_t end,
			  const char *word, bool folds, size_t *end_out)
{
	const size_t length = strlen(word);
	const size_t available = end - at < length ? end - at : length;
	for (size_t index = 0; index < available; index++) {
		const unsigned char byte = (unsigned char)text[at + index];
		if ((folds ? tolower(byte) : byte) !=
		    (unsigned char)word[index]) {
			return HEAD_NONE;
		}
	}
	*end_out = at + length;
	return available == length ? HEAD_WHOLE : HEAD_PARTIAL;
}

/* Whether a head may begin at the byte: its "<", or the "/" of a closing head that leaves its "<" out. */
static bool is_head_start(char byte)
{
	return byte == HEAD_OPENING || byte == HEAD_CLOSING_MARK;
}

/* A head at at: "<" WS ["/" WS] bar WS "DSML" WS bar WS keyword, or a closing head without its "<" (the reference's
 * own PARAMETER_END has none): "/" WS bar WS "DSML" WS bar WS keyword, never spelled exactly. */
static uint32_t head_read(const char *text, size_t at, size_t end,
			  struct marker_head *head)
{
	const bool is_bare = text[at] == HEAD_CLOSING_MARK;
	size_t position = marker_spaces_skipped(text, at + 1, end), after = 0;
	uint32_t keyword = 0;
	const bool is_closing =
		is_bare ||
		(position < end && text[position] == HEAD_CLOSING_MARK);
	position = is_closing && !is_bare ?
			   marker_spaces_skipped(text, position + 1, end) :
			   position;
	for (uint32_t step = STEP_FIRST_BAR; step <= STEP_KEYWORD; step++) {
		if (position == SIZE_MAX) {
			return HEAD_NONE;
		}
		if (position == end) {
			return HEAD_PARTIAL;
		}
		uint32_t match = HEAD_NONE;
		if (step == STEP_WORD) {
			match = word_read(text, position, end, MARKER_WORD,
					  true, &after);
		} else if (step == STEP_KEYWORD) {
			for (keyword = 0;
			     keyword < sizeof(marker_keywords) /
					       sizeof(marker_keywords[0]);
			     keyword++) {
				match = word_read(text, position, end,
						  marker_keywords[keyword].word,
						  true, &after);
				if (match != HEAD_NONE) {
					break;
				}
			}
		} else if (text[position] == NARROW_BAR) {
			match = HEAD_WHOLE;
			after = position + 1;
		} else {
			match = word_read(text, position, end, WIDE_BAR, false,
					  &after);
		}
		if (match != HEAD_WHOLE) {
			return match;
		}
		position = step == STEP_KEYWORD ?
				   after :
				   marker_spaces_skipped(text, after, end);
	}
	const uint32_t kind = marker_keywords[keyword].keyword;
	const char *const reference = reference_keywords[kind];
	const char *const prefix = is_closing ? "</" DSML " " : "<" DSML " ";
	*head = (struct marker_head){
		.begin = at,
		.end = position,
		.keyword = kind,
		.is_closing = is_closing,
		.is_exact = position - at ==
				    strlen(prefix) + strlen(reference) &&
			    memcmp(text + at, prefix, strlen(prefix)) == 0 &&
			    memcmp(text + position - strlen(reference),
				   reference, strlen(reference)) == 0
	};
	return HEAD_WHOLE;
}

/* The first whole head of text from at to end; HEAD_PARTIAL, with head->begin, when none is and the end cuts one
 * that may still be; HEAD_NONE. */
static uint32_t next_head(const char *text, size_t at, size_t end,
			  struct marker_head *head)
{
	for (size_t position = at; position < end; position++) {
		const uint32_t match =
			is_head_start(text[position]) ?
				head_read(text, position, end, head) :
				HEAD_NONE;
		if (match == HEAD_PARTIAL) {
			head->begin = position;
		}
		if (match != HEAD_NONE) {
			return match;
		}
	}
	return HEAD_NONE;
}

static bool head_opens_block(const struct marker_head *head)
{
	return !head->is_closing && head->keyword != KEYWORD_PARAMETER;
}

/* How many marker spaces right before head_begin, none before begin, a block trims: up to its last two line feeds,
 * at most CHAT_MARKER_SPACE_MAX. */
static size_t spaces_trimmed(const char *text, size_t begin, size_t head_begin)
{
	size_t length = 0;
	uint32_t feeds = 0;
	while (head_begin - length > begin && feeds < LINE_FEEDS_TRIMMED &&
	       is_marker_space(text[head_begin - length - 1])) {
		length++;
		feeds += text[head_begin - length] == LINE_FEED ? 1u : 0u;
	}
	return length < CHAT_MARKER_SPACE_MAX ? length : CHAT_MARKER_SPACE_MAX;
}

/* The first head of the count bytes that opens a block, true, with its place and that of the spaces it trims; else
 * false and the longest tail that may still begin one: a head the end cuts and the spaces before it, or the spaces
 * at the end. The heads passed over are stray. */
static bool block_opening_found(const char *text, size_t count,
				struct marker_head *opening, size_t *begin_out,
				size_t *tail_out)
{
	for (size_t position = 0;;) {
		const uint32_t match =
			next_head(text, position, count, opening);
		if (match == HEAD_WHOLE && head_opens_block(opening)) {
			*begin_out = opening->begin -
				     spaces_trimmed(text, 0, opening->begin);
			return true;
		}
		if (match != HEAD_WHOLE) {
			const size_t from =
				match == HEAD_PARTIAL ? opening->begin : count;
			*tail_out =
				count - (from - spaces_trimmed(text, 0, from));
			return false;
		}
		position = opening->end;
	}
}

/* Where text given out up to until stops so that it cuts no whole head: that head's beginning, or until. */
static size_t head_cut_avoided(const char *text, size_t count, size_t until)
{
	struct marker_head head = { 0 };
	for (size_t position = 0;
	     next_head(text, position, count, &head) == HEAD_WHOLE;
	     position = head.end) {
		if (head.end > until) {
			return head.begin < until ? head.begin : until;
		}
	}
	return until;
}

/* The whole heads of the first count bytes: given out as text, each a stray one. */
static uint32_t heads_counted(const char *text, size_t count)
{
	struct marker_head head = { 0 };
	uint32_t heads = 0;
	for (size_t position = 0;
	     next_head(text, position, count, &head) == HEAD_WHOLE;
	     position = head.end) {
		heads++;
	}
	return heads;
}

/* Where content given out up to until stops so that it ends in no head the end cuts: that head's beginning, while
 * the bytes held back stay within a stop string's room, or until. A block's text between tags may complete such a
 * head; held, it is still in held when the block begins (join_counted). */
static size_t open_head_held(const char *text, size_t count, size_t until)
{
	struct marker_head head = { 0 };
	for (size_t at = until > MARKER_HEAD_MOST_BYTES ?
				 until - MARKER_HEAD_MOST_BYTES :
				 0;
	     at < until; at++) {
		if (is_head_start(text[at]) &&
		    head_read(text, at, until, &head) == HEAD_PARTIAL) {
			return count - at <= CHAT_MOST_STOP_BYTES ? at : until;
		}
	}
	return until;
}

/* Where the block's opening head begins: after the spaces it trimmed. */
static size_t block_head(const struct chat_buffer *block)
{
	size_t at = 0;
	while (at < block->count && is_marker_space(block->bytes[at])) {
		at++;
	}
	return at;
}

/* The block that begins at until of the held bytes takes them and every later byte: opening is its head as the
 * tolerant reading found it (NULL: the reference's CALLS_BEGIN), after the spaces it trims. A block whose head is
 * the reference's exact one is read by the reference's rule first. */
static void block_begun(struct chat_answer *answer,
			const struct marker_head *opening, size_t until)
{
	const bool is_reference =
		opening == NULL ||
		(opening->is_exact && opening->keyword == KEYWORD_BLOCK);
	answer->calls_block_read = opening == NULL ? strlen(CALLS_BEGIN) :
				   is_reference	   ? opening->end - until :
						     opening->begin - until;
	answer->are_calls_malformed = !is_reference;
	chat_buffer_write(&answer->calls_block, answer->held + until,
			  answer->held_count - until);
	/* The given tail: the content just given out before the block, none when the block begins in reasoning. */
	char *const tail = answer->held + CHAT_HELD_CAPACITY - JOIN_TAIL_BYTES;
	const size_t kept = answer->is_reasoning    ? 0 :
			    until < JOIN_TAIL_BYTES ? until :
						      JOIN_TAIL_BYTES;
	memmove(tail + JOIN_TAIL_BYTES - kept, answer->held + until - kept,
		kept);
	memset(tail, 0, JOIN_TAIL_BYTES - kept);
	answer->held_count = 0;
	answer->is_reasoning = false;
	answer->is_calling = true;
}

/* Gives out the held bytes nothing can change any more: up to the first whole delimiter (the end of reasoning, or in
 * the content a stop string; with tools in either the beginning of calls), else up to a tail that may still begin
 * one. True when a stop string ends the answer; the beginning of calls moves what follows it into the block. With
 * the tolerant reading the beginning of calls is a head that opens one, from the spaces it trims; text given out
 * cuts no head, and every head in it is counted stray. */
static bool held_released(struct chat_answer *answer, bool is_last)
{
	static const char reasoning_end[] = REASONING_END,
			  calls_begin[] = CALLS_BEGIN;
	const struct chat_request *const request = answer->request;
	const bool is_tolerant = !request->is_calls_reference;
	for (;;) {
		const bool controlled = answer->is_reasoning &&
					request->thinking_budget != 0;
		const uint32_t calls_index =
			answer->is_reasoning ?
				(controlled ? 0u : 1u) :
				request->stop_count; /* after the part's own */
		const uint32_t delimiter_count =
			calls_index +
			(request->tools_end > request->tools_begin ? 1u : 0u);
		const char *found = NULL;
		size_t found_bytes = 0, tail = 0;
		uint32_t found_index = 0;
		struct marker_head opening = { 0 };
		size_t opening_tail_most = 0;
		for (uint32_t index = 0; index < delimiter_count; index++) {
			const bool is_opening = index == calls_index &&
						is_tolerant;
			const char *const delimiter =
				index == calls_index ? calls_begin :
				answer->is_reasoning ? reasoning_end :
						       request->stops[index];
			const size_t bytes =
				index == calls_index ?
					sizeof(calls_begin) - 1 :
				answer->is_reasoning ?
					sizeof(reasoning_end) - 1 :
					request->stop_byte_counts[index];
			size_t opening_begin = 0, opening_tail = 0;
			const char *const at =
				is_opening ?
					(block_opening_found(answer->held,
							     answer->held_count,
							     &opening,
							     &opening_begin,
							     &opening_tail) ?
						 answer->held + opening_begin :
						 NULL) :
					memmem(answer->held, answer->held_count,
					       delimiter, bytes);
			found_index =
				at != NULL && (found == NULL || at < found) ?
					index :
					found_index;
			found_bytes =
				at != NULL && (found == NULL || at < found) ?
					bytes :
					found_bytes;
			found = at != NULL && (found == NULL || at < found) ?
					at :
					found;
			tail = opening_tail > tail ? opening_tail : tail;
			opening_tail_most = opening_tail > opening_tail_most ?
						    opening_tail :
						    opening_tail_most;
			for (size_t length = bytes - 1 < answer->held_count ?
						     bytes - 1 :
						     answer->held_count;
			     !is_opening && length > tail; length--) {
				tail = memcmp(answer->held +
						      answer->held_count -
						      length,
					      delimiter, length) == 0 ?
					       length :
					       tail;
			}
		}
		const bool counts_heads = is_tolerant &&
					  delimiter_count > calls_index;
		/* A head the end cuts before a whole delimiter may still begin a block there: no verdict yet. */
		found = counts_heads && found != NULL &&
					found_index != calls_index &&
					!is_last &&
					(size_t)(found - answer->held) >
						answer->held_count -
							opening_tail_most ?
				NULL :
				found;
		if (controlled && found != NULL) {
			chat_answer_fail(answer);
			return true;
		}
		const size_t held_until =
			found != NULL ?
				(size_t)(found - answer->held) :
				answer->held_count - (is_last ? 0 : tail);
		const size_t whole_until =
			counts_heads && found == NULL ?
				head_cut_avoided(answer->held,
						 answer->held_count,
						 held_until) :
				held_until;
		const uint32_t until =
			(uint32_t)(counts_heads && found == NULL && !is_last &&
						   !answer->is_reasoning ?
					   open_head_held(answer->held,
							  answer->held_count,
							  whole_until) :
					   whole_until);
		answer->heads_as_text +=
			counts_heads ? heads_counted(answer->held, until) : 0u;
		const uint32_t given =
			text_given_out(answer, until, is_last || found != NULL);
		const uint32_t dropped =
			found != NULL ? until + (uint32_t)found_bytes : given;
		if (found != NULL && !answer->is_reasoning &&
		    found_index < request->stop_count)
			answer->matched_stop = found_index;
		if (found != NULL && found_index == calls_index) {
			block_begun(answer, is_tolerant ? &opening : NULL,
				    until);
			return false;
		}
		memmove(answer->held, answer->held + dropped,
			answer->held_count - dropped);
		answer->held_count -= dropped;
		if (found == NULL || !answer->is_reasoning) {
			return found != NULL;
		}
		answer->is_reasoning = false;
	}
}

/* ---- calls ------------------------------------------------------------------------------------------------- */

enum call_reading {
	CALL_UNFINISHED,
	CALL_WHOLE,
	CALLS_CLOSED,
	CALLS_MALFORMED
};

/* The place of the first of two tokens in the block at or after from, and which is first (0 or 1); the block's count
 * when neither is there yet. */
static size_t first_found(const struct chat_buffer *block, size_t from,
			  const char *first, const char *second,
			  uint32_t *which_out)
{
	const char *const at_first = block->bytes != NULL ?
					     memmem(block->bytes + from,
						    block->count - from, first,
						    strlen(first)) :
					     NULL;
	const char *const at_second = block->bytes != NULL ?
					      memmem(block->bytes + from,
						     block->count - from,
						     second, strlen(second)) :
					      NULL;
	*which_out = at_second != NULL && (at_first == NULL ||
					   at_second < at_first) ?
			     1u :
			     0u;
	return at_first == NULL && at_second == NULL ?
		       block->count :
		       (size_t)((*which_out == 1u ? at_second : at_first) -
				block->bytes);
}

static bool is_python_space(char byte) /* \s of an ASCII text */
{
	return byte != '\0' && strchr(PYTHON_ASCII_SPACES, byte) != NULL;
}

/* The bytes of the Python Unicode \s character at at, as UTF-8 (0: none). */
static size_t space_bytes(const char *text, size_t at, size_t end)
{
	static const char *const unicode_spaces[] = {
		"\xc2\x85",	"\xc2\xa0",	"\xe1\x9a\x80", "\xe2\x80\x80",
		"\xe2\x80\x81", "\xe2\x80\x82", "\xe2\x80\x83", "\xe2\x80\x84",
		"\xe2\x80\x85", "\xe2\x80\x86", "\xe2\x80\x87", "\xe2\x80\x88",
		"\xe2\x80\x89", "\xe2\x80\x8a", "\xe2\x80\xa8", "\xe2\x80\xa9",
		"\xe2\x80\xaf", "\xe2\x81\x9f", "\xe3\x80\x80"
	};
	if (at < end && is_python_space(text[at])) {
		return 1;
	}
	for (size_t index = 0;
	     at < end &&
	     index < sizeof(unicode_spaces) / sizeof(unicode_spaces[0]);
	     index++) {
		const size_t length = strlen(unicode_spaces[index]);
		if (length <= end - at &&
		    memcmp(text + at, unicode_spaces[index], length) == 0) {
			return length;
		}
	}
	return 0;
}

/* Python's Unicode \s, as UTF-8. In particular the reference also allows it before name=. */
static size_t spaces_skipped(const char *text, size_t begin, size_t end)
{
	for (size_t bytes = space_bytes(text, begin, end); bytes != 0;
	     bytes = space_bytes(text, begin, end)) {
		begin += bytes;
	}
	return begin;
}

/* A tag's final > followed by any whitespace, including none. */
static bool is_call_separator(const char *text, size_t begin, size_t end)
{
	return begin < end && text[begin] == '>' &&
	       spaces_skipped(text, begin + 1, end) == end;
}

/* One parameter's content, between its token and "/｜DSML｜ parameter": ' name="K" string="true|false">V<' as the
 * reference's expression reads it, the name up to the first string mark. False when it is not that. */
static bool parameter_read(const char *text, size_t begin, size_t end, struct json_bytes *name_out,
                           struct json_bytes *value_out, bool *is_string_out)
{
    if (end > begin && text[end - 1] == '\n') {
        end--;                                                /* Python's $ also matches before a final LF */
    }
    const size_t name_begin = begin + strlen(PARAMETER_NAME_OPENING);
    if (end - begin < strlen(PARAMETER_NAME_OPENING) + 1 || text[end - 1] != PARAMETER_VALUE_END ||
        memcmp(text + begin, PARAMETER_NAME_OPENING, strlen(PARAMETER_NAME_OPENING)) != 0) {
        return false;
    }
    const char *const string_true = memmem(text + name_begin, end - 1 - name_begin, STRING_TRUE, strlen(STRING_TRUE));
    const char *const string_false = memmem(text + name_begin, end - 1 - name_begin, STRING_FALSE,
                                            strlen(STRING_FALSE));
    const bool is_string = string_true != NULL && (string_false == NULL || string_true < string_false);
    const char *const mark = is_string ? string_true : string_false;
    if (mark == NULL) {
        return false;
    }
    const size_t value_begin = (size_t)(mark - text) + (is_string ? strlen(STRING_TRUE) : strlen(STRING_FALSE));
    *name_out = (struct json_bytes){text + name_begin, (size_t)(mark - text) - name_begin};
    *value_out = (struct json_bytes){text + value_begin, end - 1 - value_begin};
    *is_string_out = is_string;
    return true;
}

/* Rescan already validated spans instead of imposing a parameter-count limit absent from the reference. */
static bool parameter_name_repeated(const struct chat_buffer *block,
				    size_t first, size_t current,
				    struct json_bytes name)
{
    while (first < current) {
        const char *const end = memmem(block->bytes + first, current - first, PARAMETER_END, strlen(PARAMETER_END));
        struct json_bytes previous_name = {NULL, 0}, previous_value = {NULL, 0};
        bool is_string = false;
	if (end == NULL ||
	    !parameter_read(block->bytes, first, (size_t)(end - block->bytes),
			    &previous_name, &previous_value, &is_string) ||
	    json_compare_bytes(&previous_name, &name) == 0) {
		return true;
	}
	const size_t after = (size_t)(end - block->bytes) + strlen(PARAMETER_END);
        const char *const next = memmem(block->bytes + after, current - after, PARAMETER_BEGIN,
                                        strlen(PARAMETER_BEGIN));
        if (next == NULL) {
            break;
        }
        first = (size_t)(next - block->bytes) + strlen(PARAMETER_BEGIN);
    }
    return false;
}

/* The JSON of a value as a member's text: written as JSON when a string, as the model wrote it else. */
static void arguments_member_write(struct chat_buffer *arguments,
				   struct json_bytes name,
				   struct json_bytes value, bool is_string,
				   bool is_first)
{
	chat_buffer_write(arguments, ", ", is_first ? 0 : strlen(", "));
	buffer_string(arguments, name.data, name.count);
	buffer_text(arguments, ": ");
	if (is_string) {
		buffer_string(arguments, value.data, value.count);
	} else {
		chat_buffer_write(arguments, value.data, value.count);
	}
}

/* Reference token searches and attributes, with whitespace allowed between tags. No verdict before the next whole
 * delimiter: even its bytes may still be split. Live invokes require >; final parsing also permits reference EOFs. */
static uint32_t call_read(const struct chat_buffer *block, size_t *read_at,
			  struct json_bytes *name_out,
			  struct chat_buffer *arguments, bool is_last)
{
	const char *const text = block->bytes;
	uint32_t which = 0, parameter_count = 0;
	size_t at =
		first_found(block, *read_at, INVOKE_BEGIN, CALLS_END, &which);
	if (at == block->count) {
		return CALL_UNFINISHED;
	}
	if (!is_call_separator(text, *read_at, at)) {
		return CALLS_MALFORMED;
	}
    if (which == 1u) {
	    *read_at = at + strlen(CALLS_END);
	    return CALLS_CLOSED;
    }
    size_t name_begin = at + strlen(INVOKE_BEGIN);
    at = first_found(block, name_begin, PARAMETER_BEGIN, INVOKE_END, &which);
    if (at == block->count && !is_last) {
        return CALL_UNFINISHED;
    }
    name_begin = spaces_skipped(text, name_begin, at);
    if (at - name_begin < strlen(NAME_OPENING) + strlen(NAME_CLOSING) ||
        memcmp(text + name_begin, NAME_OPENING, strlen(NAME_OPENING)) != 0) {
        return CALLS_MALFORMED;
    }
    name_begin += strlen(NAME_OPENING);
    size_t name_end = name_begin;
    for (;;) {
        const char *const closing = memmem(text + name_end, at - name_end, NAME_CLOSING, strlen(NAME_CLOSING));
        if (closing == NULL) {
            return CALLS_MALFORMED;
        }
        name_end = (size_t)(closing - text);
        if (spaces_skipped(text, name_end + strlen(NAME_CLOSING), at) == at) {
            break;
        }
        name_end++;
    }
    *name_out = (struct json_bytes){text + name_begin, name_end - name_begin};
    const char *const qualified = memmem(name_out->data, name_out->count, NAMESPACE_SEPARATOR,
                                        strlen(NAMESPACE_SEPARATOR));
    if (qualified != NULL && memmem(qualified + strlen(NAMESPACE_SEPARATOR),
                                    name_out->count - (size_t)(qualified - name_out->data) -
                                        strlen(NAMESPACE_SEPARATOR),
                                    NAMESPACE_SEPARATOR, strlen(NAMESPACE_SEPARATOR)) != NULL) {
        return CALLS_MALFORMED;
    }
    arguments->count = 0;
    buffer_text(arguments, "{");
    const size_t first_parameter = at + strlen(PARAMETER_BEGIN);
    which = at == block->count ? 1u : which;                   /* reference permits EOF after the name */
    while (which == 0u) {                                       /* a parameter's token */
        const size_t content_begin = at + strlen(PARAMETER_BEGIN);
        const char *const content_end = memmem(text + content_begin, block->count - content_begin, PARAMETER_END,
                                               strlen(PARAMETER_END));
        struct json_bytes name = {NULL, 0}, value = {NULL, 0};
        bool is_string = false;
        if (content_end == NULL) {
            return CALL_UNFINISHED;
        }
        if (!parameter_read(text, content_begin, (size_t)(content_end - text), &name, &value, &is_string) ||
            parameter_name_repeated(block, first_parameter, content_begin, name)) {
            return CALLS_MALFORMED;
        }
        arguments_member_write(arguments, name, value, is_string, parameter_count++ == 0);
        const size_t after = (size_t)(content_end - text) + strlen(PARAMETER_END);
        at = first_found(block, after, PARAMETER_BEGIN, INVOKE_END, &which);
        if (at == block->count && !is_last) {
            return CALL_UNFINISHED;
        }
        if (!is_call_separator(text, after, at)) {
            return CALLS_MALFORMED;
        }
        which = at == block->count ? 1u : which;               /* and after a parameter's separator */
    }
    const size_t after_invoke = at < block->count ? at + strlen(INVOKE_END) : at;
    if (after_invoke == block->count && !is_last) {
        return CALL_UNFINISHED;
    }
    if (after_invoke < block->count && text[after_invoke] != '>') {
        return CALLS_MALFORMED;
    }
    buffer_text(arguments, "}");
    *read_at = after_invoke; /* the next separator consumes this > */
    return CALL_WHOLE;
}

/* One call given out: a stream's tool_calls delta of its index, or a whole answer's entry. */
static void call_given_out(struct chat_answer *answer, struct json_bytes name,
			   const struct chat_buffer *arguments)
{
	if (answer->request->is_messages) {
		server_messages_call(answer, name, arguments);
		answer->call_count++;
		return;
	}
	if (answer->request->is_responses) {
		server_responses_call(answer, name, arguments);
		answer->call_count++;
		return;
	}
	struct chat_buffer *const target =
		answer->request->is_streamed ? &answer->output : &answer->calls;
	char call_identifier[CHAT_IDENTIFIER_CAPACITY + NUMBER_TEXT_CAPACITY] = {
		0
	};
	const int length = snprintf(call_identifier, sizeof(call_identifier),
				    "%s-%" PRIu32, answer->identifier,
				    answer->call_count);
	if (answer->request->is_streamed) {
		chunk_begin(answer);
		buffer_text(target, "{\"tool_calls\":[{\"index\":");
		buffer_number(target, answer->call_count);
		buffer_text(target, ",");
	} else {
		buffer_text(target, answer->call_count == 0 ? "{" : ",{");
	}
    buffer_text(target, "\"id\":");
    buffer_string(target, call_identifier, length > 0 ? (size_t)length : 0);
    buffer_text(target, ",\"type\":\"function\",\"function\":{\"name\":");
    const char *const qualified = memmem(name.data, name.count, NAMESPACE_SEPARATOR, strlen(NAMESPACE_SEPARATOR));
    const size_t namespace_bytes = qualified != NULL ? (size_t)(qualified - name.data) : 0;
    const size_t bare_begin = qualified != NULL ? namespace_bytes + strlen(NAMESPACE_SEPARATOR) : 0;
    buffer_string(target, name.data + bare_begin, name.count - bare_begin);
    buffer_text(target, ",\"arguments\":");
    buffer_string(target, arguments->bytes != NULL ? arguments->bytes : "", arguments->count);
    buffer_text(target, "}");
    if (qualified != NULL) {
        buffer_text(target, ",\"namespace\":");
        buffer_string(target, name.data, namespace_bytes);
    }
    buffer_text(target, "}");
    if (answer->request->is_streamed) {
        buffer_text(target, "]}");
        chunk_end(answer, NULL);
    }
    answer->call_count++;
}

/* ---- the tolerant reading of calls -------------------------------------------------------------------------- */

enum tag_stop { STOP_NONE, STOP_HEAD, STOP_WAIT, STOP_END };
enum string_form { FORM_MISSING, FORM_STRING, FORM_AS_WRITTEN };
enum unit_kind {
	UNIT_UNFINISHED,
	UNIT_CALL,
	UNIT_UNNAMED,
	UNIT_CLOSED,
	UNIT_END,
	UNIT_CUT /* an invoke a request limit cut: text, never a call */
};

struct marker_tag {
	size_t end; /* after its ">", or at the head or the end that ends it */
	struct json_bytes name,
		string; /* the last value of each; data NULL: none */
};

/* One parameter of the invoke being read: its name and value as places of the block. */
struct call_parameter {
	size_t name_begin, name_end, value_begin, value_end;
	uint32_t form; /* enum string_form */
};

/* One reading of the block from where it stands: by the reference's rule until it refuses an invoke, then
 * tolerantly. It gives the calls and the text between tags out to answer, or with answer NULL only counts. */
struct calls_reading {
	struct chat_answer *answer;
	/* The block, and the declared names (chat_tool_names_kept's) or NULL. */
	const struct chat_buffer *block, *names;
	/* The block's bytes so far, its opening head and where the reading stands. */
	const char *text;
	size_t end, head, at;
	uint32_t call_count;
	/* is_limited: is_last at a request limit, where an invoke the end cut is no call */
	bool is_last, is_limited, is_tolerant, is_closed;
	/* The invoke's records, and the text between tags since the last call. */
	struct chat_buffer parameters, between;
	/* The repairs of the invokes read since the last one taken, and of those taken. */
	uint32_t repairs[CHAT_CALLS_REPAIR_KINDS],
		counted[CHAT_CALLS_REPAIR_KINDS];
};

/* What ends a part of a tag at at: a whole head, or the end; STOP_WAIT where a later byte may decide (the end
 * before the last, a head it cuts). */
static uint32_t stop_at(const char *text, size_t at, size_t end, bool is_last)
{
	struct marker_head head = { 0 };
	if (at == end) {
		return is_last ? STOP_END : STOP_WAIT;
	}
	const uint32_t match = is_head_start(text[at]) ?
				       head_read(text, at, end, &head) :
				       HEAD_NONE;
	return match == HEAD_WHOLE		 ? STOP_HEAD :
	       match == HEAD_PARTIAL && !is_last ? STOP_WAIT :
						   STOP_NONE;
}

static bool is_quote(char byte)
{
	return byte == DOUBLE_QUOTE || byte == SINGLE_QUOTE;
}

/* Whether bytes are word, in any case. */
static bool is_word(struct json_bytes bytes, const char *word)
{
	bool is_same = bytes.data != NULL && bytes.count == strlen(word);
	for (size_t index = 0; is_same && index < bytes.count; index++) {
		is_same = tolower((unsigned char)bytes.data[index]) ==
			  (unsigned char)word[index];
	}
	return is_same;
}

/* The bytes without the Python whitespace at either end. */
static struct json_bytes bytes_trimmed(struct json_bytes bytes)
{
	const size_t begin = spaces_skipped(bytes.data, 0, bytes.count);
	size_t last = begin;
	for (size_t at = begin; at < bytes.count;) {
		const size_t space = space_bytes(bytes.data, at, bytes.count);
		at += space != 0 ? space : 1u;
		last = space == 0 ? at : last;
	}
	return (struct json_bytes){ bytes.data + begin, last - begin };
}

/* The tag whose attributes begin at at, its head's end: in any order, with spaces around "=", quoted with " or '
 * or bare; a quote that does not close before a head or the end ends at its first ">", which ends the tag. False
 * while bytes still to come may change it. */
static bool tag_read(const char *text, size_t at, size_t end, bool is_last,
		     struct marker_tag *tag)
{
	*tag = (struct marker_tag){ 0 };
	for (size_t position = at;;) {
		position = spaces_skipped(text, position, end);
		uint32_t stop = stop_at(text, position, end, is_last);
		if (stop == STOP_WAIT) {
			return false;
		}
		if (stop != STOP_NONE || text[position] == TAG_END) {
			tag->end = stop != STOP_NONE ? position : position + 1;
			return true;
		}
		size_t name_end = position;
		while ((stop = stop_at(text, name_end, end, is_last)) ==
			       STOP_NONE &&
		       space_bytes(text, name_end, end) == 0 &&
		       text[name_end] != ATTRIBUTE_EQUALS &&
		       text[name_end] != TAG_END && !is_quote(text[name_end])) {
			name_end++;
		}
		if (stop == STOP_WAIT) {
			return false;
		}
		if (name_end == position) {
			position++; /* a stray "=" or quote */
			continue;
		}
		const struct json_bytes name = { text + position,
						 name_end - position };
		position = spaces_skipped(text, name_end, end);
		stop = stop_at(text, position, end, is_last);
		if (stop == STOP_WAIT) {
			return false;
		}
		if (stop != STOP_NONE || text[position] != ATTRIBUTE_EQUALS) {
			continue; /* an attribute with no value */
		}
		position = spaces_skipped(text, position + 1, end);
		stop = stop_at(text, position, end, is_last);
		if (stop == STOP_WAIT) {
			return false;
		}
		const bool is_quoted = stop == STOP_NONE &&
				       is_quote(text[position]);
		const size_t value_begin = position + (is_quoted ? 1u : 0u);
		size_t scan = value_begin, greater = SIZE_MAX;
		while ((stop = stop_at(text, scan, end, is_last)) ==
			       STOP_NONE &&
		       (is_quoted ? text[scan] != text[position] :
				    space_bytes(text, scan, end) == 0 &&
					    text[scan] != TAG_END)) {
			greater = greater == SIZE_MAX && text[scan] == TAG_END ?
					  scan :
					  greater;
			scan++;
		}
		if (stop == STOP_WAIT) {
			return false;
		}
		const bool is_cut = is_quoted && stop != STOP_NONE &&
				    greater != SIZE_MAX;
		const struct json_bytes value = { text + value_begin,
						  (is_cut ? greater : scan) -
							  value_begin };
		tag->name = is_word(name, NAME_WORD) ? value : tag->name;
		tag->string = is_word(name, STRING_WORD) ? value : tag->string;
		if (is_cut) {
			tag->end = greater + 1;
			return true;
		}
		position = is_quoted && stop == STOP_NONE ? scan + 1 : scan;
	}
}

/* Whether a tag (its attributes from at) is the reference's: ">" alone, ' name="N">' after an invoke head, or
 * ' name="N" string="true">' (or "false") after a parameter head. */
static bool is_tag_exact(const char *text, size_t at,
			 const struct marker_head *head,
			 const struct marker_tag *tag)
{
	const size_t length = tag->end - at,
		     name_at = at + strlen(EXACT_NAME_OPENING);
	if (head->is_closing || head->keyword == KEYWORD_BLOCK) {
		return length == 1 && text[at] == TAG_END;
	}
	const bool is_parameter = head->keyword == KEYWORD_PARAMETER;
	const size_t string_at =
		name_at + tag->name.count + strlen(EXACT_STRING_OPENING);
	const size_t expected = strlen(EXACT_NAME_OPENING) + tag->name.count +
				strlen(EXACT_TAG_CLOSING) +
				(is_parameter ? strlen(EXACT_STRING_OPENING) +
							tag->string.count :
						0);
	return tag->name.data == text + name_at && length == expected &&
	       memcmp(text + at, EXACT_NAME_OPENING,
		      strlen(EXACT_NAME_OPENING)) == 0 &&
	       memcmp(text + tag->end - strlen(EXACT_TAG_CLOSING),
		      EXACT_TAG_CLOSING, strlen(EXACT_TAG_CLOSING)) == 0 &&
	       (!is_parameter ||
		(tag->string.data == text + string_at &&
		 memcmp(text + name_at + tag->name.count, EXACT_STRING_OPENING,
			strlen(EXACT_STRING_OPENING)) == 0 &&
		 ((tag->string.count == strlen(WORD_TRUE) &&
		   memcmp(tag->string.data, WORD_TRUE, strlen(WORD_TRUE)) ==
			   0) ||
		  (tag->string.count == strlen(WORD_FALSE) &&
		   memcmp(tag->string.data, WORD_FALSE, strlen(WORD_FALSE)) ==
			   0))));
}

static void tag_counted(struct calls_reading *reading,
			const struct marker_head *head,
			const struct marker_tag *tag)
{
	reading->repairs[CHAT_CALLS_REPAIR_MARKER_SPELLING] +=
		head->is_exact ? 0u : 1u;
	reading->repairs[CHAT_CALLS_REPAIR_ATTRIBUTE] +=
		is_tag_exact(reading->text, head->end, head, tag) ? 0u : 1u;
}

/* Text between tags, without the spaces around it, kept to go out as content. */
static void text_between(struct calls_reading *reading, size_t begin,
			 size_t end)
{
	const struct json_bytes text = bytes_trimmed(
		(struct json_bytes){ reading->text + begin, end - begin });
	if (text.count != 0) {
		chat_buffer_write(&reading->between, text.data, text.count);
		reading->repairs[CHAT_CALLS_REPAIR_TEXT_IN_BLOCK]++;
	}
}

static uint32_t string_form_of(struct json_bytes string)
{
	return is_word(string, WORD_TRUE) || is_word(string, DIGIT_TRUE) ?
		       FORM_STRING :
	       is_word(string, WORD_FALSE) || is_word(string, DIGIT_FALSE) ?
		       FORM_AS_WRITTEN :
		       FORM_MISSING;
}

/* A value a missing string attribute writes as it is: JSON as json.loads reads it, a number, literal, object or
 * array, with no NaN or Infinity. */
static bool is_json_value(struct json_bytes value)
{
	struct json_value document = { 0 };
	return value.count != 0 &&
	       json_read_document(value.data, value.count, &document) ==
		       JSON_DOCUMENT_VALID &&
	       strchr(JSON_FIRST_BYTES, document.text[document.begin]) !=
		       NULL &&
	       !json_has_unquoted(document, JSON_UNQUOTED_REFUSED);
}

/* Whether two names are one ignoring case and '-'/'_'. */
static bool names_fold_equal(struct json_bytes first, struct json_bytes second)
{
	bool is_same = first.count == second.count;
	for (size_t index = 0; is_same && index < first.count; index++) {
		const int one = tolower((unsigned char)first.data[index]),
			  other = tolower((unsigned char)second.data[index]);
		is_same =
			(one == UNFOLDED_SEPARATOR ? FOLDED_SEPARATOR : one) ==
			(other == UNFOLDED_SEPARATOR ? FOLDED_SEPARATOR :
						       other);
	}
	return is_same;
}

/* The declared tool a call's name is: itself, else the first one it is ignoring case and '-'/'_' (counted), never
 * by prefix; with none, the name as written (counted when the names are whole: none too long, no memory refused).
 * One pass over the names for each call given out. */
static struct json_bytes name_resolved(const struct chat_buffer *names,
				       struct json_bytes name,
				       uint32_t *repairs)
{
	const size_t count =
		names != NULL && names->bytes != NULL ? names->count : 0;
	struct json_bytes folded = { NULL, 0 };
	bool is_whole = count != 0 && !names->is_short;
	uint32_t length = 0;
	for (size_t at = 0; at + sizeof(length) <= count;) {
		memcpy(&length, names->bytes + at, sizeof(length));
		at += sizeof(length);
		if (length == TOOL_NAME_NOT_KEPT) {
			is_whole = false;
			continue;
		}
		if (length > count - at) {
			is_whole = false;
			break; /* a name memory cut */
		}
		const struct json_bytes tool = { names->bytes + at, length };
		if (tool.count == name.count &&
		    memcmp(tool.data, name.data, name.count) == 0) {
			return name;
		}
		folded = folded.data == NULL && names_fold_equal(tool, name) ?
				 tool :
				 folded;
		at += length;
	}
	repairs[CHAT_CALLS_REPAIR_NAME_FOLDED] += folded.data != NULL ? 1u : 0u;
	repairs[CHAT_CALLS_REPAIR_NAME_UNKNOWN] +=
		folded.data == NULL && is_whole ? 1u : 0u;
	return folded.data != NULL ? folded : name;
}

static struct call_parameter parameter_at(const struct chat_buffer *records,
					  size_t index)
{
	struct call_parameter record = { 0 };
	memcpy(&record, records->bytes + index * sizeof(record),
	       sizeof(record));
	return record;
}

static bool parameter_names_equal(const char *text, struct call_parameter first,
				  struct call_parameter second)
{
	return first.name_end - first.name_begin ==
		       second.name_end - second.name_begin &&
	       memcmp(text + first.name_begin, text + second.name_begin,
		      first.name_end - first.name_begin) == 0;
}

/* The arguments of the invoke's parameters: each name once, in the place of its first, with its last value; one
 * with no name is passed over. */
static void arguments_written(struct calls_reading *reading,
			      struct chat_buffer *arguments)
{
	const struct chat_buffer *const records = &reading->parameters;
	const size_t count = records->count / sizeof(struct call_parameter);
	bool is_first = true;
	arguments->count = 0;
	buffer_text(arguments, "{");
	for (size_t index = 0; index < count; index++) {
		const struct call_parameter parameter =
			parameter_at(records, index);
		if (parameter.name_end == parameter.name_begin) {
			continue; /* a parameter with no name is passed over; its tag is counted */
		}
		bool is_repeated = false;
		for (size_t earlier = 0; earlier < index && !is_repeated;
		     earlier++) {
			is_repeated = parameter_names_equal(
				reading->text, parameter_at(records, earlier),
				parameter);
		}
		reading->repairs[CHAT_CALLS_REPAIR_PARAMETER_REPEATED] +=
			is_repeated ? 1u : 0u;
		struct call_parameter last = parameter;
		for (size_t later = index + 1; later < count && !is_repeated;
		     later++) {
			last = parameter_names_equal(reading->text,
						     parameter_at(records,
								  later),
						     parameter) ?
				       parameter_at(records, later) :
				       last;
		}
		const struct json_bytes value = {
			reading->text + last.value_begin,
			last.value_end - last.value_begin
		};
		if (!is_repeated) {
			arguments_member_write(
				arguments,
				(struct json_bytes){
					reading->text + parameter.name_begin,
					parameter.name_end -
						parameter.name_begin },
				value,
				last.form == FORM_STRING ||
					(last.form == FORM_MISSING &&
					 !is_json_value(value)),
				is_first);
			is_first = false;
		}
	}
	buffer_text(arguments, "}");
}

static void parameter_kept(struct calls_reading *reading,
			   struct call_parameter *parameter, size_t value_end)
{
	parameter->value_end = value_end;
	chat_buffer_write(&reading->parameters, parameter, sizeof(*parameter));
}

/* The invoke from opening (an invoke head, or a parameter head, which opens one with no name) to its end: the next
 * invoke head, its closing, the block's closing or the end. */
static uint32_t invoke_read(struct calls_reading *reading,
			    const struct marker_head *opening, size_t *next_out,
			    struct json_bytes *name_out,
			    struct chat_buffer *arguments)
{
	const char *const text = reading->text;
	struct json_bytes name = { NULL, 0 };
	struct call_parameter parameter = { 0 };
	struct marker_tag tag = { 0 };
	bool is_parameter_open = false, is_cut = false;
	size_t position = opening->begin;
	/* what the group had before this invoke: a cut one gives back its repairs and text between tags */
	uint32_t repairs_before[CHAT_CALLS_REPAIR_KINDS];
	const size_t between_before = reading->between.count;
	memcpy(repairs_before, reading->repairs, sizeof(repairs_before));
	reading->parameters.count = 0;
	if (opening->keyword == KEYWORD_INVOKE) {
		if (!tag_read(text, opening->end, reading->end,
			      reading->is_last, &tag)) {
			return UNIT_UNFINISHED;
		}
		tag_counted(reading, opening, &tag);
		name = tag.name.data != NULL ? bytes_trimmed(tag.name) : name;
		position = tag.end;
	}
	for (;;) {
		struct marker_head head = { 0 };
		if (next_head(text, position, reading->end, &head) !=
		    HEAD_WHOLE) {
			if (!reading->is_last) {
				return UNIT_UNFINISHED;
			}
			if (is_parameter_open) {
				parameter_kept(reading, &parameter,
					       reading->end);
				reading->repairs
					[CHAT_CALLS_REPAIR_CLOSING_IMPLIED]++;
			} else {
				text_between(reading, position, reading->end);
			}
			reading->repairs[CHAT_CALLS_REPAIR_CLOSING_IMPLIED]++;
			*next_out = reading->end;
			is_cut = reading->is_limited;
			break;
		}
		if (is_parameter_open) {
			parameter_kept(reading, &parameter, head.begin);
			is_parameter_open = false;
			if (head.is_closing &&
			    head.keyword == KEYWORD_PARAMETER) {
				if (!tag_read(text, head.end, reading->end,
					      reading->is_last, &tag)) {
					return UNIT_UNFINISHED;
				}
				tag_counted(reading, &head, &tag);
				position = tag.end;
				continue;
			}
			reading->repairs[CHAT_CALLS_REPAIR_CLOSING_IMPLIED]++;
		} else {
			text_between(reading, position, head.begin);
		}
		if ((head.keyword == KEYWORD_INVOKE && !head.is_closing) ||
		    (head.keyword == KEYWORD_BLOCK && head.is_closing)) {
			reading->repairs[CHAT_CALLS_REPAIR_CLOSING_IMPLIED]++;
			*next_out = head.begin;
			break;
		}
		if (!tag_read(text, head.end, reading->end, reading->is_last,
			      &tag)) {
			return UNIT_UNFINISHED;
		}
		tag_counted(reading, &head, &tag);
		position = tag.end;
		if (head.keyword == KEYWORD_PARAMETER && !head.is_closing) {
			const struct json_bytes parameter_name =
				tag.name.data != NULL ?
					bytes_trimmed(tag.name) :
					(struct json_bytes){ text + tag.end,
							     0 };
			parameter = (struct call_parameter){
				.name_begin =
					(size_t)(parameter_name.data - text),
				.name_end =
					(size_t)(parameter_name.data - text) +
					parameter_name.count,
				.value_begin = tag.end,
				.form = string_form_of(tag.string)
			};
			is_parameter_open = true;
			continue;
		}
		if (head.keyword == KEYWORD_INVOKE) {
			*next_out = tag.end;
			break;
		}
		reading->repairs
			[CHAT_CALLS_REPAIR_MARKER_STRAY]++; /* an opening block or closing parameter head in it */
	}
	if (name.count == 0) {
		reading->repairs[CHAT_CALLS_REPAIR_INVOKE_UNNAMED]++;
		return UNIT_UNNAMED;
	}
	if (is_cut) { /* closed by nothing but a request limit: no call */
		memcpy(reading->repairs, repairs_before,
		       sizeof(repairs_before));
		reading->between.count = between_before;
		reading->repairs[CHAT_CALLS_REPAIR_CUT_AT_LIMIT]++;
		*next_out = opening->begin;
		return UNIT_CUT;
	}
	*name_out = name_resolved(reading->names, name, reading->repairs);
	arguments_written(reading, arguments);
	return UNIT_CALL;
}

/* The block's next step from *at: an invoke (a call, or one with no name), its closing, or its end; the text
 * between tags it passes goes to between. */
static uint32_t unit_read(struct calls_reading *reading, size_t *at,
			  struct json_bytes *name_out,
			  struct chat_buffer *arguments)
{
	for (size_t position = *at;;) {
		struct marker_head head = { 0 };
		struct marker_tag tag = { 0 };
		if (next_head(reading->text, position, reading->end, &head) !=
		    HEAD_WHOLE) {
			if (!reading->is_last) {
				return UNIT_UNFINISHED;
			}
			struct marker_head opening = { 0 };
			text_between(reading, position, reading->end);
			/* A block its head opened, closed by the end. */
			reading->repairs[CHAT_CALLS_REPAIR_CLOSING_IMPLIED] +=
				reading->head < reading->end &&
						head_read(reading->text,
							  reading->head,
							  reading->end,
							  &opening) ==
							HEAD_WHOLE &&
						opening.keyword ==
							KEYWORD_BLOCK ?
					1u :
					0u;
			*at = reading->end;
			return UNIT_END;
		}
		text_between(reading, position, head.begin);
		if (head.keyword != KEYWORD_BLOCK && !head.is_closing) {
			return invoke_read(reading, &head, at, name_out,
					   arguments);
		}
		if (!tag_read(reading->text, head.end, reading->end,
			      reading->is_last, &tag)) {
			return UNIT_UNFINISHED;
		}
		tag_counted(reading, &head, &tag);
		if (head.keyword == KEYWORD_BLOCK && head.is_closing) {
			*at = tag.end;
			return UNIT_CLOSED;
		}
		reading->repairs[CHAT_CALLS_REPAIR_MARKER_STRAY] +=
			head.begin != reading->head ? 1u : 0u;
		position = tag.end;
	}
}

/* Text of the answer that is no call: given out through the held bytes, as a whole. */
static void bytes_given_out_as_text(struct chat_answer *answer,
				    const char *bytes, size_t count)
{
	/* the tolerant reading keeps the given tail at the end of held */
	const size_t capacity =
		sizeof(answer->held) -
		(answer->request->is_calls_reference ? 0 : JOIN_TAIL_BYTES);
	size_t taken = 0;
	answer->held_count = 0;
	for (bool is_last = false; !is_last;) {
		const size_t room = capacity - answer->held_count,
			     left = count - taken;
		const size_t piece = left < room ? left : room;
		memcpy(answer->held + answer->held_count,
		       bytes != NULL ? bytes + taken : "", piece);
		answer->held_count += (uint32_t)piece;
		taken += piece;
		is_last = taken == count;
		const uint32_t given =
			text_given_out(answer, answer->held_count, is_last);
		memmove(answer->held, answer->held + given,
			answer->held_count - given);
		answer->held_count -= given;
	}
	answer->held_count = 0;
}

/* The heads that text between tags forms, alone or with the content given before it (the given tail), counted
 * stray as the client's joined text holds them (G1); then the text joins the tail. Its own pieces hold no whole
 * head, so every head counted ends in it. */
static void join_counted(struct chat_answer *answer, const char *text,
			 size_t count)
{
	char *const tail = answer->held + CHAT_HELD_CAPACITY - JOIN_TAIL_BYTES;
	char join[JOIN_TAIL_BYTES + MARKER_HEAD_MOST_BYTES];
	const size_t taken =
		count < MARKER_HEAD_MOST_BYTES ? count : MARKER_HEAD_MOST_BYTES;
	struct marker_head head = { 0 };
	size_t position = 0;
	memcpy(join, tail, JOIN_TAIL_BYTES);
	memcpy(join + JOIN_TAIL_BYTES, text, taken);
	for (size_t at = 0; at < JOIN_TAIL_BYTES; at++) {
		if (is_head_start(join[at]) &&
		    head_read(join, at, JOIN_TAIL_BYTES + taken, &head) ==
			    HEAD_WHOLE &&
		    head.end > JOIN_TAIL_BYTES) {
			answer->heads_as_text++;
			position = head.end - JOIN_TAIL_BYTES;
			break;
		}
	}
	answer->heads_as_text +=
		heads_counted(text + position, count - position);
	if (count >= JOIN_TAIL_BYTES) {
		memcpy(tail, text + count - JOIN_TAIL_BYTES, JOIN_TAIL_BYTES);
	} else {
		memmove(tail, tail + count, JOIN_TAIL_BYTES - count);
		memcpy(tail + JOIN_TAIL_BYTES - count, text, count);
	}
}

/* Whether a whole closing block head, in any spelling, begins from at to before until: in the bytes the reference's
 * reading takes, or after where it waits, it ends that reading, and the block is read tolerantly, so that the closing
 * stops generation as a tolerant block's does (a reading stuck in an invoke never sees even the exact closing). */
static bool is_closing_found(const struct calls_reading *reading, size_t at,
			     size_t until)
{
	struct marker_head head = { 0 };
	for (size_t position = at;
	     next_head(reading->text, position, reading->end, &head) ==
		     HEAD_WHOLE &&
	     head.begin < until;
	     position = head.end) {
		if (head.is_closing && head.keyword == KEYWORD_BLOCK) {
			return true;
		}
	}
	return false;
}

/* A call the reading takes: given out when it emits. */
static void call_taken(struct calls_reading *reading, struct json_bytes name,
		       const struct chat_buffer *arguments)
{
	if (reading->answer != NULL) {
		call_given_out(reading->answer, name, arguments);
	}
	reading->call_count++;
}

/* The block's calls as far as its bytes decide them: by the reference's reading while it accepts each invoke, then
 * tolerantly from the first it refuses, or at the end from where it still waits. The text between tags of the
 * invokes up to a call goes out before it, and after the last call at the block's closing or end. */
static void calls_read(struct calls_reading *reading)
{
	struct chat_buffer arguments = { 0 };
	for (bool is_ended = false;
	     !reading->is_closed && !is_ended &&
	     (reading->answer == NULL || !reading->answer->is_finished);) {
		struct json_bytes name = { NULL, 0 };
		if (!reading->is_tolerant) {
			/* at a request limit the reference's EOF forms give no call: the tolerant reading decides */
			const size_t from = reading->at;
			uint32_t verdict = call_read(
				reading->block, &reading->at, &name, &arguments,
				reading->is_last && !reading->is_limited);
			if (verdict == CALL_WHOLE &&
			    is_closing_found(reading, from, reading->at)) {
				reading->at = from;
				verdict = CALLS_MALFORMED;
			}
			if (verdict == CALL_WHOLE) {
				call_taken(reading,
					   name_resolved(reading->names,
							 bytes_trimmed(name),
							 reading->counted),
					   &arguments);
			}
			reading->is_closed = verdict == CALLS_CLOSED;
			if (verdict == CALL_UNFINISHED && !reading->is_last &&
			    !is_closing_found(reading, reading->at,
					      reading->end)) {
				break;
			}
			if (verdict == CALLS_MALFORMED ||
			    verdict == CALL_UNFINISHED) {
				const bool closes =
					reading->call_count != 0 &&
					reading->at < reading->end &&
					reading->text[reading->at] == TAG_END;
				reading->at =
					reading->call_count == 0 ?
						reading->head :
						reading->at +
							(closes ? 1u : 0u);
				reading->is_tolerant = true;
			}
			continue;
		}
		size_t next = reading->at;
		uint32_t unit = UNIT_UNNAMED;
		memset(reading->repairs, 0, sizeof(reading->repairs));
		reading->between.count = 0;
		while (unit == UNIT_UNNAMED) {
			unit = unit_read(reading, &next, &name, &arguments);
		}
		if (unit == UNIT_UNFINISHED) {
			break;
		}
		for (uint32_t kind = 0; kind < CHAT_CALLS_REPAIR_KINDS;
		     kind++) {
			reading->counted[kind] += reading->repairs[kind];
		}
		reading->at = next;
		if (reading->answer != NULL && reading->between.count != 0 &&
		    (unit == UNIT_CALL || reading->call_count != 0)) {
			join_counted(reading->answer, reading->between.bytes,
				     reading->between.count);
			bytes_given_out_as_text(reading->answer,
						reading->between.bytes,
						reading->between.count);
		}
		if (reading->answer != NULL && unit == UNIT_CUT &&
		    reading->call_count != 0) {
			bytes_given_out_as_text(reading->answer,
						reading->text + next,
						reading->end - next);
		}
		if (unit == UNIT_CALL) {
			call_taken(reading, name, &arguments);
		}
		reading->is_closed = unit == UNIT_CLOSED;
		is_ended = unit == UNIT_END || unit == UNIT_CUT;
	}
	if (reading->answer != NULL) {
		reading->answer->output.is_short =
			reading->answer->output.is_short ||
			arguments.is_short || reading->parameters.is_short ||
			reading->between.is_short;
	}
	chat_buffer_free(&arguments);
	chat_buffer_free(&reading->parameters);
	chat_buffer_free(&reading->between);
}

/* The answer's block read on from where it stands, its calls and text given out. */
static void calls_read_tolerantly(struct chat_answer *answer, bool is_last,
				  bool is_limited)
{
	struct calls_reading reading = {
		.answer = answer,
		.block = &answer->calls_block,
		.names = answer->tool_names,
		.text = answer->calls_block.bytes,
		.end = answer->calls_block.count,
		.head = block_head(&answer->calls_block),
		.at = answer->calls_block_read,
		.call_count = answer->call_count,
		.is_last = is_last,
		.is_limited = is_limited,
		.is_tolerant = answer->are_calls_malformed,
		.is_closed = answer->are_calls_closed
	};
	calls_read(&reading);
	answer->calls_block_read = reading.at;
	answer->are_calls_malformed = reading.is_tolerant;
	answer->are_calls_closed = reading.is_closed;
}

/* Malformed input is held until finish, never used to stop generation. Whole calls keep their indices and ids. */
static void calls_given_out(struct chat_answer *answer, bool is_last,
			    bool is_limited)
{
	if (!answer->request->is_calls_reference) {
		calls_read_tolerantly(answer, is_last, is_limited);
		return;
	}
	struct chat_buffer arguments = { 0 };
	struct json_bytes name = { NULL, 0 };
	uint32_t reading = CALL_WHOLE;
	while (!answer->are_calls_closed && !answer->are_calls_malformed &&
	       reading == CALL_WHOLE) {
		reading = call_read(&answer->calls_block,
				    &answer->calls_block_read, &name,
				    &arguments, is_last);
		answer->output.is_short = answer->output.is_short ||
					  arguments.is_short;
		if (reading == CALL_WHOLE) {
			call_given_out(answer, name, &arguments);
		}
		answer->are_calls_closed = reading == CALLS_CLOSED;
		answer->are_calls_malformed = reading == CALLS_MALFORMED;
	}
	chat_buffer_free(&arguments);
}

/* With no calls, return the original block; otherwise only the unread suffix, never duplicating emitted calls. */
static void block_given_out_as_text(struct chat_answer *answer)
{
	const bool skips_closing =
		answer->call_count != 0 &&
		answer->calls_block_read < answer->calls_block.count &&
		answer->calls_block.bytes[answer->calls_block_read] == TAG_END;
	const size_t taken =
		answer->call_count == 0 ?
			0 :
			answer->calls_block_read +
				(skips_closing ?
					 1u :
					 0u); /* the emitted invoke's closing > */
	bytes_given_out_as_text(answer,
				answer->calls_block.bytes != NULL ?
					answer->calls_block.bytes + taken :
					NULL,
				answer->calls_block.count - taken);
}

void chat_answer_begin(struct chat_answer *answer,
		       const struct chat_request *request,
		       const char *model_name, const char *identifier,
		       int64_t created_seconds, uint32_t prompt_token_count)
{
    struct json_writer prefix = {answer->prefix, sizeof(answer->prefix), 0};
    char created[NUMBER_TEXT_CAPACITY] = {0};
    const int created_length = snprintf(created, sizeof(created), "%" PRId64, created_seconds);
    const char *const object = request->is_streamed ? ",\"object\":\"chat.completion.chunk\",\"created\":"
                                                    : ",\"object\":\"chat.completion\",\"created\":";
    answer->output = (struct chat_buffer){ answer->output.bytes, 0, 0,
					   answer->output.capacity, false };
    answer->reasoning = (struct chat_buffer){ answer->reasoning.bytes, 0, 0,
					      answer->reasoning.capacity,
					      false };
    answer->content = (struct chat_buffer){ answer->content.bytes, 0, 0,
					    answer->content.capacity, false };
    answer->calls_block = (struct chat_buffer){ answer->calls_block.bytes, 0, 0,
					 answer->calls_block.capacity, false };
    answer->calls = (struct chat_buffer){ answer->calls.bytes, 0, 0,
					  answer->calls.capacity, false };
    answer->tool_names = NULL;
    answer->heads_as_text = 0;
    answer->request = request;
    answer->held_count = answer->completion_token_count = answer->call_count = answer->cached_token_count = 0;
    answer->reasoning_token_count = 0;
    answer->matched_stop = request->stop_count;
    answer->calls_block_read = 0;
    answer->prompt_token_count = prompt_token_count;
    answer->is_reasoning = request->reasoning_effort != 0;
    answer->is_calling = answer->are_calls_closed = answer->is_finished = false;
    answer->are_calls_malformed = false;
    answer->defers_stop_finish = answer->stop_matched = answer->has_draft_statistics = false;
    (void)snprintf(answer->identifier, sizeof(answer->identifier), "%s", identifier);
    if (request->is_messages) {
	    server_messages_begin(answer);
	    return;
    }
    if (request->is_responses) {
	    server_responses_begin(answer, model_name, created_seconds);
	    return;
    }
    answer->pending_responses_fields = NULL;
    json_write_bytes(&prefix, "{\"id\":", strlen("{\"id\":"));
    json_write_string(&prefix,
		      (struct json_bytes){ identifier, strlen(identifier) });
    json_write_bytes(&prefix, object, strlen(object));
    json_write_bytes(&prefix, created,
		     created_length > 0 ? (size_t)created_length : 0);
    json_write_bytes(&prefix, ",\"model\":", strlen(",\"model\":"));
    json_write_string(&prefix,
		      (struct json_bytes){ model_name, strlen(model_name) });
    json_write_bytes(&prefix, ",\"choices\":[", strlen(",\"choices\":["));
    answer->prefix_count = prefix.length;
    answer->output.is_short = prefix.length > prefix.capacity;
    if (request->is_streamed) {
        buffer_text(&answer->output, STREAM_HEAD);
        chunk_begin(answer);
        buffer_text(&answer->output, "{\"role\":\"assistant\",\"content\":\"\"}");
        chunk_end(answer, NULL);
    }
}

bool chat_answer_token_control(struct chat_answer *answer, const char *bytes,
			       uint32_t byte_count, bool closes_reasoning)
{
    if (answer->is_finished || answer->stop_matched) {
        return false;
    }
    answer->completion_token_count++;
    if (closes_reasoning && answer->request->thinking_budget != 0 &&
	answer->is_reasoning) {
	    (void)text_given_out(answer, answer->held_count, true);
	    answer->held_count = 0;
	    answer->is_reasoning = false;
	    return false;
    }

    if (answer->is_calling) {
	    chat_buffer_write(&answer->calls_block, bytes, byte_count);
    } else if (byte_count >
	       CHAT_HELD_CAPACITY -
		       answer->held_count) { /* a token's bytes and a tail fit */
	    answer->output.is_short = true;
	    return false;
    } else {
	    if (byte_count != 0) {
		    memcpy(answer->held + answer->held_count, bytes,
			   byte_count);
	    }
	    answer->held_count += byte_count;
    }
    bool is_stopped = !answer->is_calling && held_released(answer, false);
    if (answer->is_calling) {
	    calls_given_out(
		    answer, false,
		    false); /* a closed block ends the answer: only the end token may follow */
	    answer->output.is_short = answer->output.is_short ||
				      answer->calls_block.is_short;
	    is_stopped = answer->are_calls_closed;
    }
    if (answer->request->is_messages && answer->is_finished) {
	    return true; /* a protocol output failure also cancels the owned command */
    }
    if (is_stopped) {
        answer->held_count = 0;
        answer->stop_matched = true;
        if (!answer->defers_stop_finish) {
		chat_answer_finish(answer, "stop");
	}
    }
    return is_stopped;
}

bool chat_answer_token(struct chat_answer *answer, const char *bytes,
		       uint32_t byte_count)
{
	return chat_answer_token_control(answer, bytes, byte_count, false);
}

void chat_answer_finish(struct chat_answer *answer, const char *finish_reason)
{
	struct chat_buffer *const output = &answer->output;
	if (answer->is_finished) {
		return;
	}
    if (!answer->is_calling) {
	    /* A stop held back only by a head the end cut (the tolerant reading's wait) still ends the answer. */
	    answer->stop_matched = held_released(answer, true) ||
				   answer->stop_matched;
    }
    if (answer->is_calling && !answer->request->is_calls_reference) {
	    calls_given_out(answer, true,
			    !answer->stop_matched &&
				    strcmp(finish_reason, "length") == 0);
	    if (answer->call_count == 0) {
		    bytes_given_out_as_text(answer, answer->calls_block.bytes,
					    answer->are_calls_closed ?
						    answer->calls_block_read :
						    answer->calls_block.count);
	    }
    } else if (answer->is_calling) {
	    answer->are_calls_malformed =
		    false; /* reparse the held suffix at EOS or the token limit */
	    calls_given_out(answer, true, false);
	    if (answer->call_count == 0 ||
		(!answer->request->is_messages && !answer->are_calls_closed &&
		 answer->calls_block_read < answer->calls_block.count)) {
		    block_given_out_as_text(answer);
	    }
    }
    const char *const stopped_reason = answer->stop_matched ? "stop" : finish_reason;
    const char *const reason = answer->call_count != 0 && strcmp(stopped_reason, "stop") == 0 ? "tool_calls"
                                                                                           : stopped_reason;
    answer->is_finished = true;
    if (answer->request->is_messages) {
	    server_messages_finish(answer, reason);
	    return;
    }
    if (answer->request->is_responses) {
	    server_responses_finish(answer, reason);
	    return;
    }
    if (answer->request->is_streamed) {
        chunk_begin(answer);
        buffer_text(output, "{}");
        chunk_end(answer, reason);
        if (answer->request->includes_usage) {
            buffer_text(output, "data: ");
	    chat_buffer_write(output, answer->prefix, answer->prefix_count);
	    buffer_text(output, "],");
            usage_write(answer);
            buffer_text(output, "}\n\n");
        }
        buffer_text(output, "data: [DONE]\n\n");
        return;
    }
    const bool is_short = output->is_short || answer->reasoning.is_short || answer->content.is_short ||
                          answer->calls_block.is_short || answer->calls.is_short;
    body_begin(output);
    chat_buffer_write(output, answer->prefix, answer->prefix_count);
    buffer_text(output, "{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":");
    buffer_string(output, answer->content.bytes != NULL ? answer->content.bytes : "", answer->content.count);
    if (answer->request->reasoning_effort != 0) {
        buffer_text(output, ",\"reasoning_content\":");
        buffer_string(output, answer->reasoning.bytes != NULL ? answer->reasoning.bytes : "", answer->reasoning.count);
    }
    if (answer->call_count != 0) {
        buffer_text(output, ",\"tool_calls\":[");
	chat_buffer_write(output, answer->calls.bytes, answer->calls.count);
	buffer_text(output, "]");
    }
    buffer_text(output, "},\"finish_reason\":\"");
    buffer_text(output, reason);
    buffer_text(output, "\"}],");
    usage_write(answer);
    buffer_text(output, "}\n");
    output->is_short = output->is_short || is_short;
    head_put(output, HTTP_OK, "");
}

void chat_answer_fail(struct chat_answer *answer)
{
	if (answer->request->is_messages) {
		if (!answer->is_finished)
			server_messages_fail(answer);
		return;
	}
	if (answer->request->is_responses) {
		server_responses_fail(answer);
		return;
	}
	const struct chat_refusal refusal = { HTTP_SERVER_ERROR,
					      CHAT_ENGINE_FAILED, NULL, 0, 0 };
	if (!answer->is_finished && answer->request->is_streamed) {
		buffer_text(
			&answer->output,
			"data: {\"error\":{\"message\":\"" CHAT_ENGINE_FAILED
			"\","
			"\"type\":\"server_error\",\"param\":null,\"code\":null}}\n\n");
	} else if (!answer->is_finished) {
		chat_refusal_write(&answer->output, &refusal, NULL,
				   CHAT_ROUTE_COMPLETIONS);
	}
    answer->is_finished = true;
}

void chat_answer_repairs(const struct chat_answer *answer, bool is_limited,
			 uint32_t repairs[CHAT_CALLS_REPAIR_KINDS])
{
	const struct chat_request *const request = answer->request;
	const struct chat_buffer *const block = &answer->calls_block;
	memset(repairs, 0, CHAT_CALLS_REPAIR_KINDS * sizeof(*repairs));
	if (request == NULL || !answer->is_finished) {
		return;
	}
	repairs[CHAT_CALLS_REPAIR_MARKER_STRAY] = answer->heads_as_text;
	if (answer->is_calling &&
	    (request->is_calls_reference || answer->call_count == 0)) {
		repairs[CHAT_CALLS_REPAIR_BLOCK_AS_TEXT] =
			answer->call_count == 0 || (!request->is_messages &&
						    !answer->are_calls_closed &&
						    answer->calls_block_read <
							    block->count) ?
				1u :
				0u;
		return;
	}
	if (!answer->is_calling) {
		return;
	}
	/* The block read again, as one piece: what any split of it repaired (G2). */
	struct marker_head opening = { 0 };
	const size_t head = block_head(block);
	const bool is_reference = head < block->count &&
				  head_read(block->bytes, head, block->count,
					    &opening) == HEAD_WHOLE &&
				  opening.is_exact &&
				  opening.keyword == KEYWORD_BLOCK;
	struct calls_reading reading = {
		.block = block,
		.names = answer->tool_names,
		.text = block->bytes,
		.end = block->count,
		.head = head,
		.at = is_reference ? opening.end : head,
		.is_last = true,
		.is_limited = is_limited && !answer->stop_matched,
		.is_tolerant = !is_reference
	};
	calls_read(&reading);
	reading.counted[CHAT_CALLS_REPAIR_BLOCK_START] +=
		is_reference && head == strlen(REFERENCE_BLOCK_SPACES) &&
				memcmp(block->bytes, REFERENCE_BLOCK_SPACES,
				       head) == 0 ?
			0u :
			1u;
	for (uint32_t kind = 0; kind < CHAT_CALLS_REPAIR_KINDS; kind++) {
		repairs[kind] += reading.counted[kind];
	}
}

void chat_answer_free(struct chat_answer *answer)
{
	server_messages_free(answer);
	server_responses_free(answer);
	answer->pending_responses_fields = NULL;
	chat_buffer_free(&answer->output);
	chat_buffer_free(&answer->reasoning);
	chat_buffer_free(&answer->content);
	chat_buffer_free(&answer->calls_block);
	chat_buffer_free(&answer->calls);
}

