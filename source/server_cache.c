/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Server cache identities: exact authenticated user and tagged client signals, never prompt content or credentials.
 * JSON has already passed chat_request_read, including duplicate-key and UTF-8 validation. */
#include <string.h>
#include <strings.h>

#include "anemone/tokenizer.h"
#include "json.h"
#include "server_internal.h"

#define NAMESPACE_LENGTH_BYTES 2u
#define NAMESPACE_TAG_BYTES (1u + NAMESPACE_LENGTH_BYTES)
#define NAMESPACE_BYTE_BITS 8u
#define NAMESPACE_BYTE_MASK 0xffu
#define ESCAPED_BYTES_PER_BYTE 6u
#define KEY_SCRATCH_BYTES 256u
#define LINE_END_BYTES 2u
#define SCOPED_IDENTITY_PREFIX_BYTES \
	2u /* scope mode and authenticated/anonymous boundary */
#define BODY_TAG_BASE 1u
#define METADATA_TAG_BASE 64u
#define CLIENT_METADATA_TAG_BASE 96u
#define HEADER_TAG_BASE 128u
#define API_USER_TAG 0u
#define BODY_CACHE_KEY_INDEX 2u
#define BODY_SESSION_INDEX 4u
#define BODY_CONVERSATION_INDEX 5u
#define METADATA_USER_INDEX 6u
#define HEADER_THREAD_INDEX 9u
#define HEADER_USER_INDEX 11u
#define HEADER_TASK_INDEX 12u
#define FORK_HINT_CONTROL_END 0x20u
#define FORK_HINT_ASCII_DELETE 0x7fu

enum client_metadata_signal {
	CLIENT_METADATA_SESSION,
	CLIENT_METADATA_THREAD,
	CLIENT_METADATA_PARENT_THREAD,
	CLIENT_METADATA_SIGNALS
};
#define CLIENT_METADATA_NAMESPACE_SIGNALS CLIENT_METADATA_PARENT_THREAD

enum fork_header { FORK_HEADER_SESSION, FORK_HEADER_THREAD, FORK_HEADERS };

/* A persisted v4 format: append fields, never reorder. Separate tag ranges keep new body fields from rebinding
 * old metadata/header identities. Exact tagged bytes decide eligibility after any filename hash. */
static const char *const body_signals[] = {
	"user",	      "safety_identifier", "prompt_cache_key", "cache_salt",
	"session_id", "conversation_id",   "user_id"
};
static const char *const metadata_signals[] = {
	"session_id",	    "sessionId", "conversation_id", "conversationId",
	"prompt_cache_key", "cache_key", "user_id"
};
static const char *const client_metadata_signals[] = {
	"session_id", "thread_id", "x-codex-parent-thread-id"
};
static const char *const fork_headers[] = { "x-anemone-fork-of",
					    "x-codex-parent-thread-id" };
static const char *const header_signals[] = {
	"session_id",	      "session-id",	     "x-session-id",
	"x-session-affinity", "x-opencode-session",  "x-opencode-session-id",
	"x-conversation-id",  "agent-session-id",    "x-litellm-session-id",
	"thread-id",	      "x-openwebui-chat-id", "x-openwebui-user-id",
	"x-task-id",	      "x-opencode-project",  "openai-project",
	"openai-organization"
};
#define BODY_SIGNALS (sizeof(body_signals) / sizeof(body_signals[0]))
#define METADATA_SIGNALS \
	(sizeof(metadata_signals) / sizeof(metadata_signals[0]))
#define HEADER_SIGNALS (sizeof(header_signals) / sizeof(header_signals[0]))
#define JSON_SIGNALS (BODY_SIGNALS + METADATA_SIGNALS)
_Static_assert(BODY_TAG_BASE + BODY_SIGNALS <= METADATA_TAG_BASE,
	       "body identity tags have their own range");
_Static_assert(METADATA_TAG_BASE + METADATA_SIGNALS <= CLIENT_METADATA_TAG_BASE,
	       "metadata tags have their own range");
_Static_assert(CLIENT_METADATA_TAG_BASE + CLIENT_METADATA_NAMESPACE_SIGNALS <=
		       HEADER_TAG_BASE,
	       "stable client metadata has its own tag range");
_Static_assert(
	sizeof(client_metadata_signals) / sizeof(client_metadata_signals[0]) ==
		CLIENT_METADATA_SIGNALS,
	"client metadata includes two stable identities and one private donor hint");
_Static_assert(sizeof(fork_headers) / sizeof(fork_headers[0]) == FORK_HEADERS,
	       "each fork header has a declared identity domain");
_Static_assert(HEADER_TAG_BASE + HEADER_SIGNALS <= UINT8_MAX,
	       "namespace tags fit one byte");

bool server_cache_namespace_equal(const struct cache_namespace *first,
				  const struct cache_namespace *second)
{
	return first->byte_count <= SERVER_CACHE_NAMESPACE_BYTES &&
	       first->byte_count == second->byte_count &&
	       memcmp(first->bytes, second->bytes, first->byte_count) == 0;
}

/* These existing tags are affinity, not authority. Everything else in the known registry stays hard: API
 * user, end-user/safety identities, salt, and explicit user/project/organization headers. New tags default hard. */
static bool affinity_tag(uint32_t tag)
{
	return tag == BODY_TAG_BASE + BODY_CACHE_KEY_INDEX ||
	       tag == BODY_TAG_BASE + BODY_SESSION_INDEX ||
	       tag == BODY_TAG_BASE + BODY_CONVERSATION_INDEX ||
	       (tag >= METADATA_TAG_BASE &&
		tag < METADATA_TAG_BASE + METADATA_USER_INDEX) ||
	       (tag >= CLIENT_METADATA_TAG_BASE &&
		tag < CLIENT_METADATA_TAG_BASE +
				CLIENT_METADATA_NAMESPACE_SIGNALS) ||
	       (tag >= HEADER_TAG_BASE &&
		tag < HEADER_TAG_BASE + HEADER_USER_INDEX) ||
	       tag == HEADER_TAG_BASE + HEADER_TASK_INDEX;
}

static bool known_tag(uint32_t tag)
{
	return tag == API_USER_TAG ||
	       (tag >= BODY_TAG_BASE && tag < BODY_TAG_BASE + BODY_SIGNALS) ||
	       (tag >= METADATA_TAG_BASE &&
		tag < METADATA_TAG_BASE + METADATA_SIGNALS) ||
	       (tag >= CLIENT_METADATA_TAG_BASE &&
		tag < CLIENT_METADATA_TAG_BASE +
				CLIENT_METADATA_NAMESPACE_SIGNALS) ||
	       (tag >= HEADER_TAG_BASE &&
		tag < HEADER_TAG_BASE + HEADER_SIGNALS);
}

struct namespace_field {
	uint32_t tag;
	struct json_bytes value;
};

static bool identity_text_valid(struct json_bytes value, bool is_hint)
{
	for (size_t cursor = 0; cursor < value.count;) {
		const unsigned char byte = (unsigned char)value.data[cursor];
		bool valid = false;
		const uint32_t bytes = tokenizer_utf8_sequence_bytes(
			value.data, cursor, value.count, &valid);
		if (!valid || bytes == 0 || bytes > value.count - cursor ||
		    (is_hint && (byte < FORK_HINT_CONTROL_END ||
				 byte == FORK_HINT_ASCII_DELETE))) {
			return false;
		}
		cursor += bytes;
	}
	return true;
}

/* Counted TLV reads also serve potentially imported records: never trust a stored field length or tag. */
static bool namespace_field_read(const struct cache_namespace *identity,
				 uint32_t *cursor,
				 struct namespace_field *field)
{
	if (*cursor > identity->byte_count ||
	    identity->byte_count - *cursor < NAMESPACE_TAG_BYTES) {
		return false;
	}
	const uint32_t begin = *cursor;
	const uint32_t tag = identity->bytes[begin];
	const uint32_t count =
		identity->bytes[begin + 1u] |
		((uint32_t)identity->bytes[begin + NAMESPACE_LENGTH_BYTES]
		 << NAMESPACE_BYTE_BITS);
	if (!known_tag(tag) || count == 0 || count > SERVER_CACHE_FIELD_BYTES ||
	    count > identity->byte_count - begin - NAMESPACE_TAG_BYTES) {
		return false;
	}
	*field = (struct namespace_field){
		tag,
		{ (const char *)identity->bytes + begin + NAMESPACE_TAG_BYTES,
		  count }
	};
	*cursor = begin + NAMESPACE_TAG_BYTES + count;
	return true;
}

static bool
authenticated_namespace_valid(const struct cache_namespace *identity)
{
	if (identity == NULL ||
	    identity->byte_count > sizeof(identity->bytes) ||
	    identity->byte_count <
		    SCOPED_IDENTITY_PREFIX_BYTES + NAMESPACE_TAG_BYTES + 1u ||
	    identity->bytes[0] != SERVER_CACHE_SESSION ||
	    identity->bytes[1] != 1u) {
		return false;
	}
	uint32_t cursor = SCOPED_IDENTITY_PREFIX_BYTES;
	struct namespace_field field;
	if (!namespace_field_read(identity, &cursor, &field) ||
	    field.tag != API_USER_TAG ||
	    !identity_text_valid(field.value, false)) {
		return false;
	}
	const uint32_t has_session = cursor < identity->byte_count ? 1u : 0u;
	if (identity->has_session != has_session) {
		return false;
	}
	uint32_t previous_tag = field.tag;
	while (cursor < identity->byte_count) {
		if (!namespace_field_read(identity, &cursor, &field) ||
		    field.tag <= previous_tag ||
		    !identity_text_valid(field.value, false)) {
			return false;
		}
		previous_tag = field.tag;
	}
	return true;
}

static bool next_owner_field(const struct cache_namespace *identity,
			     uint32_t *cursor, struct namespace_field *field)
{
	while (*cursor < identity->byte_count &&
	       namespace_field_read(identity, cursor, field)) {
		if (!affinity_tag(field->tag)) {
			return true;
		}
	}
	return false;
}

bool server_cache_owner_equal(const struct cache_namespace *first,
			      const struct cache_namespace *second)
{
	if (!authenticated_namespace_valid(first) ||
	    !authenticated_namespace_valid(second)) {
		return false;
	}
	uint32_t first_cursor = SCOPED_IDENTITY_PREFIX_BYTES,
		 second_cursor = SCOPED_IDENTITY_PREFIX_BYTES;
	struct namespace_field first_field, second_field;
	for (;;) {
		const bool has_first =
			next_owner_field(first, &first_cursor, &first_field);
		const bool has_second =
			next_owner_field(second, &second_cursor, &second_field);
		if (!has_first || !has_second) {
			return has_first == has_second;
		}
		if (first_field.tag != second_field.tag ||
		    first_field.value.count != second_field.value.count ||
		    memcmp(first_field.value.data, second_field.value.data,
			   first_field.value.count) != 0) {
			return false;
		}
	}
}

bool server_cache_fork_hint_matches(const struct cache_fork_hint *hint,
				    const struct cache_namespace *identity)
{
	if (hint == NULL || hint->byte_count == 0 ||
	    hint->byte_count > SERVER_CACHE_FIELD_BYTES ||
	    (hint->kind != CACHE_FORK_SESSION &&
	     hint->kind != CACHE_FORK_THREAD) ||
	    !identity_text_valid((struct json_bytes){ hint->bytes,
						      hint->byte_count },
				 true) ||
	    !authenticated_namespace_valid(identity)) {
		return false;
	}
	uint32_t cursor = SCOPED_IDENTITY_PREFIX_BYTES;
	struct namespace_field field;
	while (namespace_field_read(identity, &cursor, &field)) {
		const bool thread =
			field.tag == HEADER_TAG_BASE + HEADER_THREAD_INDEX ||
			field.tag == CLIENT_METADATA_TAG_BASE +
					     CLIENT_METADATA_THREAD;
		const bool candidate = hint->kind == CACHE_FORK_THREAD ?
					       thread :
					       affinity_tag(field.tag) &&
						       !thread;
		if (candidate && field.value.count == hint->byte_count &&
		    memcmp(field.value.data, hint->bytes, hint->byte_count) ==
			    0) {
			return true;
		}
	}
	return false;
}

static bool namespace_appended(struct cache_namespace *identity, uint32_t tag,
			       const void *bytes, size_t count)
{
	if (count == 0) {
		return true;
	}
	if (count > SERVER_CACHE_FIELD_BYTES ||
	    count + NAMESPACE_TAG_BYTES >
		    sizeof(identity->bytes) - identity->byte_count) {
		return false;
	}
	identity->bytes[identity->byte_count++] = (uint8_t)tag;
	identity->bytes[identity->byte_count++] =
		(uint8_t)(count & NAMESPACE_BYTE_MASK);
	identity->bytes[identity->byte_count++] =
		(uint8_t)(count >> NAMESPACE_BYTE_BITS);
	memcpy(identity->bytes + identity->byte_count, bytes, count);
	identity->byte_count += (uint32_t)count;
	return true;
}

/* Keys longer than the scratch cannot name one of these short ASCII keys, even when escaped. */
static bool key_is(struct json_value key, const char *name)
{
	char bytes[KEY_SCRATCH_BYTES];
	struct json_scratch scratch = { .bytes = bytes,
					.capacity = sizeof(bytes) };
	return key.end - key.begin <= sizeof(bytes) &&
	       json_string_is(&scratch, key, name);
}

static void signals_found(struct json_value object, const char *const *names,
			  size_t count, struct json_value *values)
{
	struct json_value key, value;
	for (size_t cursor = object.begin;
	     json_next(object, &cursor, &key, &value);) {
		for (size_t index = 0; index < count; index++) {
			if (key_is(key, names[index])) {
				values[index] = value;
			}
		}
	}
}

static bool json_signal_appended(struct cache_namespace *identity, uint32_t tag,
				 struct json_value value)
{
	if (value.text == NULL || value.text[value.begin] == 'n') {
		return true;
	}
	if (value.text[value.begin] != '"' ||
	    value.end - value.begin >
		    ESCAPED_BYTES_PER_BYTE * SERVER_CACHE_FIELD_BYTES +
			    NAMESPACE_LENGTH_BYTES) {
		return false;
	}
	char bytes[ESCAPED_BYTES_PER_BYTE * SERVER_CACHE_FIELD_BYTES + 1u];
	struct json_scratch scratch = { .bytes = bytes,
					.capacity = sizeof(bytes) };
	const struct json_bytes decoded = json_unescape(&scratch, value);
	return !scratch.is_short &&
	       namespace_appended(identity, tag, decoded.data, decoded.count);
}

static bool fork_hint_recorded(struct cache_fork_hint *hint, uint32_t kind,
			       struct json_bytes value)
{
	if (value.data == NULL || value.count == 0 ||
	    value.count > SERVER_CACHE_FIELD_BYTES ||
	    !identity_text_valid(value, true)) {
		return false;
	}
	if (hint->byte_count != 0 &&
	    (hint->byte_count != value.count ||
	     memcmp(hint->bytes, value.data, value.count) != 0)) {
		return false;
	}
	memcpy(hint->bytes, value.data, value.count);
	hint->byte_count = (uint32_t)value.count;
	/* A Codex parent is a thread, even when a generic hint repeats the same bytes. Never match its root session. */
	if (hint->kind == CACHE_FORK_NONE || kind == CACHE_FORK_THREAD) {
		hint->kind = kind;
	}
	return true;
}

static bool fork_hint_json_read(struct cache_fork_hint *hint,
				struct json_value value)
{
	if (value.text == NULL || value.text[value.begin] == 'n') {
		return true;
	}
	if (value.text[value.begin] != '"' ||
	    value.end - value.begin >
		    ESCAPED_BYTES_PER_BYTE * SERVER_CACHE_FIELD_BYTES +
			    NAMESPACE_LENGTH_BYTES) {
		return false;
	}
	char bytes[ESCAPED_BYTES_PER_BYTE * SERVER_CACHE_FIELD_BYTES + 1u];
	struct json_scratch scratch = { .bytes = bytes,
					.capacity = sizeof(bytes) };
	const struct json_bytes decoded = json_unescape(&scratch, value);
	return !scratch.is_short &&
	       fork_hint_recorded(hint, CACHE_FORK_THREAD, decoded);
}

/* The validated HTTP head is scanned only when forks are enabled. Duplicate hint headers refuse even if equal;
 * a body copy of the Codex thread hint may agree with its header. No credential or unrelated metadata is read. */
static bool fork_hint_headers_read(const struct connection *connection,
				   struct cache_fork_hint *hint)
{
	struct json_bytes values[FORK_HEADERS] = { { 0 } };
	const char *const bytes = connection->input.bytes;
	const char *const end =
		bytes + connection->head.head_byte_count - LINE_END_BYTES;
	const char *line =
		memmem(bytes, (size_t)(end - bytes), "\r\n", LINE_END_BYTES);
	for (line += LINE_END_BYTES; line < end;) {
		const char *const after =
			memmem(line, (size_t)(end + LINE_END_BYTES - line),
			       "\r\n", LINE_END_BYTES);
		const char *const colon =
			memchr(line, ':', (size_t)(after - line));
		for (size_t index = 0; index < FORK_HEADERS; index++) {
			if ((size_t)(colon - line) !=
				    strlen(fork_headers[index]) ||
			    strncasecmp(line, fork_headers[index],
					(size_t)(colon - line)) != 0) {
				continue;
			}
			if (values[index].data != NULL) {
				return false;
			}
			const char *value = colon + 1, *value_end = after;
			while (value < value_end &&
			       (*value == ' ' || *value == '\t')) {
				value++;
			}
			while (value_end > value && (value_end[-1] == ' ' ||
						     value_end[-1] == '\t')) {
				value_end--;
			}
			values[index] = (struct json_bytes){
				value, (size_t)(value_end - value)
			};
		}
		line = after + LINE_END_BYTES;
	}
	for (size_t index = 0; index < FORK_HEADERS; index++) {
		const uint32_t kind = index == FORK_HEADER_THREAD ?
					      CACHE_FORK_THREAD :
					      CACHE_FORK_SESSION;
		if (values[index].data != NULL &&
		    !fork_hint_recorded(hint, kind, values[index])) {
			return false;
		}
	}
	return true;
}

static uint32_t fork_refused(struct chat_refusal *refusal)
{
	*refusal = (struct chat_refusal){
		HTTP_BAD_REQUEST,
		"fork hints require session scope, valid strings of 1 to 512 bytes, "
		"unique headers and agreeing aliases",
		"invalid_cache_fork", 0, 0
	};
	return refusal->status;
}

/* HTTP was validated and credentials erased before this call. Only allowlisted identity header values are read.
 * Canonical header order makes wire order immaterial; repeated identity headers are ambiguous and refused. */
static bool headers_appended(const struct connection *connection,
			     struct cache_namespace *identity)
{
	struct json_bytes values[HEADER_SIGNALS] = { { 0 } };
	const char *const bytes = connection->input.bytes;
	const char *const end =
		bytes + connection->head.head_byte_count - LINE_END_BYTES;
	const char *line =
		memmem(bytes, (size_t)(end - bytes), "\r\n", LINE_END_BYTES);
	for (line += LINE_END_BYTES; line < end;) {
		const char *const after =
			memmem(line, (size_t)(end + LINE_END_BYTES - line),
			       "\r\n", LINE_END_BYTES);
		const char *const colon =
			memchr(line, ':', (size_t)(after - line));
		for (size_t index = 0; index < HEADER_SIGNALS; index++) {
			if ((size_t)(colon - line) !=
				    strlen(header_signals[index]) ||
			    strncasecmp(line, header_signals[index],
					(size_t)(colon - line)) != 0) {
				continue;
			}
			if (values[index].data != NULL) {
				return false;
			}
			const char *value = colon + 1, *value_end = after;
			while (value < value_end &&
			       (*value == ' ' || *value == '\t')) {
				value++;
			}
			while (value_end > value && (value_end[-1] == ' ' ||
						     value_end[-1] == '\t')) {
				value_end--;
			}
			values[index] = (struct json_bytes){
				value, (size_t)(value_end - value)
			};
		}
		line = after + LINE_END_BYTES;
	}
	for (size_t index = 0; index < HEADER_SIGNALS; index++) {
		if (!namespace_appended(
			    identity, (uint32_t)(HEADER_TAG_BASE + index),
			    values[index].data, values[index].count)) {
			return false;
		}
	}
	return true;
}

uint32_t server_cache_namespace_read(const struct http_server *server,
				     struct connection *connection,
				     struct chat_refusal *refusal)
{
	const bool fork_enabled = server->session_fork != 0;
	if (fork_enabled && server->cache_scope != SERVER_CACHE_SESSION) {
		return fork_refused(refusal);
	}
	struct cache_fork_hint fork_parent = { 0 };
	bool fork_valid = true;
	struct cache_namespace identity = { .byte_count = 1 };
	identity.bytes[0] = (uint8_t)server->cache_scope;
	if (server->cache_scope != SERVER_CACHE_GLOBAL) {
		identity.byte_count = SCOPED_IDENTITY_PREFIX_BYTES;
		identity.bytes[1] = connection->authenticated_user ? 1u : 0u;
	}
	bool valid = server->cache_scope == SERVER_CACHE_GLOBAL ||
		     namespace_appended(&identity, 0, connection->live.user,
					strlen(connection->live.user));
	const uint32_t user_bytes = identity.byte_count;
	if (valid && server->cache_scope == SERVER_CACHE_SESSION) {
		const char *const body = connection->input.bytes +
					 connection->head.head_byte_count;
		struct json_value document = {
			.text = body,
			.end = (size_t)connection->head.body_byte_count
		};
		while (document.begin < document.end &&
		       strchr(" \t\r\n", body[document.begin]) != NULL) {
			document.begin++;
		}
		while (document.end > document.begin &&
		       strchr(" \t\r\n", body[document.end - 1u]) != NULL) {
			document.end--;
		}
		struct json_value
			values[JSON_SIGNALS] = { { 0 } },
			client_values[CLIENT_METADATA_SIGNALS] = { { 0 } }, key,
			value;
		for (size_t cursor = document.begin;
		     json_next(document, &cursor, &key, &value);) {
			for (size_t index = 0; index < BODY_SIGNALS; index++) {
				if (key_is(key, body_signals[index])) {
					values[index] = value;
				}
			}
			if (key_is(key, "metadata") &&
			    value.text[value.begin] == '{') {
				signals_found(value, metadata_signals,
					      METADATA_SIGNALS,
					      values + BODY_SIGNALS);
			}
			if (fork_enabled && key_is(key, "client_metadata") &&
			    value.text[value.begin] != 'n') {
				if (value.text[value.begin] != '{') {
					fork_valid = false;
				} else {
					signals_found(value,
						      client_metadata_signals,
						      CLIENT_METADATA_SIGNALS,
						      client_values);
				}
			}
		}
		for (size_t index = 0; valid && index < JSON_SIGNALS; index++) {
			const uint32_t tag =
				(uint32_t)(index < BODY_SIGNALS ?
						   BODY_TAG_BASE + index :
						   METADATA_TAG_BASE + index -
							   BODY_SIGNALS);
			valid = json_signal_appended(&identity, tag,
						     values[index]);
		}
		for (size_t index = 0;
		     valid && fork_enabled &&
		     index < CLIENT_METADATA_NAMESPACE_SIGNALS;
		     index++) {
			valid = json_signal_appended(
				&identity,
				(uint32_t)(CLIENT_METADATA_TAG_BASE + index),
				client_values[index]);
		}
		if (fork_enabled) {
			fork_valid =
				fork_valid &&
				fork_hint_json_read(
					&fork_parent,
					client_values
						[CLIENT_METADATA_PARENT_THREAD]) &&
				fork_hint_headers_read(connection,
						       &fork_parent);
		}
		valid = valid && headers_appended(connection, &identity);
	}
	if (!fork_valid) {
		return fork_refused(refusal);
	}
	if (!valid) {
		*refusal = (struct chat_refusal){
			HTTP_BAD_REQUEST,
			"cache identities must be strings of at most 512 bytes, fit 8192 bytes together, "
			"and have unique headers",
			"invalid_cache_identity", 0, 0
		};
		return refusal->status;
	}
	identity.has_session = identity.byte_count > user_bytes ? 1u : 0u;
	connection->cache_namespace = identity;
	connection->fork_parent = fork_parent;
	return 0;
}
