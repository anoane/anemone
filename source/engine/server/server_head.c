/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * server_head.c: the head of an HTTP request, for server.c. Contract: server.h
 * (server_head_read).
 */
#include "server.h"
#include "engine/server/api/server_messages.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#define HEAD_END "\r\n\r\n"
#define HEAD_END_BYTES 4u
#define LINE_END "\r\n"
#define LINE_END_BYTES 2u
#define DIGITS "0123456789"
#define TOKEN_PUNCTUATION "!#$%&'*+-.^_`|~"         /* of a field's name, beside letters and digits */
#define BEARER "bearer"
#define HTTP_1_0 "HTTP/1.0"
#define HTTP_1_1 "HTTP/1.1"
#define HTTP_PREFIX "HTTP/"
#define DELETE_BYTE 0x7F
#define MOST_LENGTH_DIGITS 19u                      /* of a Content-Length: below 2^63 */
#define DECIMAL_BASE 10

static bool is_token_byte(char byte)
{
    return (byte >= 'a' && byte <= 'z') || (byte >= 'A' && byte <= 'Z') || (byte >= '0' && byte <= '9') ||
           (byte != '\0' && strchr(TOKEN_PUNCTUATION, byte) != NULL);
}

static bool is_named(const char *bytes, size_t count, const char *name)
{
    return count == strlen(name) && strncasecmp(bytes, name, count) == 0;
}

/* The refusal of a whole head of printable ASCII, or 0; what it says of the request goes to head. */
static uint32_t head_refusal(const char *bytes, struct server_head *head,
			     bool metrics_enabled, bool messages_enabled)
{
	static const char *const paths[] = {
		"/health",	     "/v1/models",    "/v1/chat/completions",
		"/v1/anemone/stats", "/v1/responses", "/metrics",
		"/v1/messages"
	}; /* by route */
	static const char *const methods[] = { "GET",  "GET", "POST", "GET",
					       "POST", "GET", "POST" };
	const char *const fields_end =
		bytes + head->head_byte_count - LINE_END_BYTES;
	const char *line_end =
		memmem(bytes, head->head_byte_count, LINE_END, LINE_END_BYTES);
	const char *const method_end =
		memchr(bytes, ' ', (size_t)(line_end - bytes));
	const char *const target_end =
		method_end != NULL ?
			memchr(method_end + 1, ' ',
			       (size_t)(line_end - method_end - 1)) :
			NULL;
	uint32_t refusal = 0, field_count = 0, host_count = 0,
		 route = sizeof(paths) / sizeof(paths[0]);
	bool has_length = false, has_credential = false, has_version = false;
	if (target_end == NULL ||
	    memchr(target_end + 1, ' ', (size_t)(line_end - target_end - 1)) !=
		    NULL ||
	    memchr(bytes, '\r', (size_t)(line_end - bytes)) != NULL ||
	    memchr(bytes, '\n', (size_t)(line_end - bytes)) != NULL) {
		return HTTP_BAD_REQUEST;
	}
    const size_t version_bytes = (size_t)(line_end - target_end - 1);
    const bool is_1_1 = is_named(target_end + 1, version_bytes, HTTP_1_1);
    if (!is_1_1 && !is_named(target_end + 1, version_bytes, HTTP_1_0)) {
	    return version_bytes > strlen(HTTP_PREFIX) &&
				   strncmp(target_end + 1, HTTP_PREFIX,
					   strlen(HTTP_PREFIX)) == 0 ?
			   HTTP_VERSION_NOT_SUPPORTED :
			   HTTP_BAD_REQUEST;
    }
    for (uint32_t index = 0; index < sizeof(paths) / sizeof(paths[0]); index++) {
        const size_t target_bytes = (size_t)(target_end - method_end - 1);
	route = (index != CHAT_ROUTE_METRICS || metrics_enabled) &&
				(index != CHAT_ROUTE_MESSAGES ||
				 messages_enabled) &&
				target_bytes == strlen(paths[index]) &&
				memcmp(method_end + 1, paths[index],
				       target_bytes) == 0 ?
			index :
			route;
    }
    head->route = route;
    /* Every line's framing is checked, also after a refusal, so the protocol-error mark does not depend on
     * field order. The refusal and its effects stay those of the first failing line. */
    for (const char *line = line_end + LINE_END_BYTES; line < fields_end;
	 line = line_end + LINE_END_BYTES) {
	    line_end = memmem(line,
			      (size_t)(fields_end + LINE_END_BYTES - line),
			      LINE_END, LINE_END_BYTES);
	    const char *const colon =
		    memchr(line, ':', (size_t)(line_end - line));
	    const char *name_end = line,
		       *value = colon != NULL ? colon + 1 : line,
		       *value_end = line_end;
	    while (name_end < line_end && is_token_byte(*name_end)) {
		    name_end++;
	    }
	    while (value < line_end && (*value == ' ' || *value == '\t')) {
		    value++;
	    }
	    while (value_end > value &&
		   (value_end[-1] == ' ' || value_end[-1] == '\t')) {
		    value_end--;
	    }
	    const size_t name_bytes = (size_t)(name_end - line),
			 value_bytes = (size_t)(value_end - value);
	    const bool too_many = ++field_count > SERVER_MOST_HEAD_FIELDS;
	    /* a lone CR or LF, folding, or no name before ':' */
	    const bool malformed =
		    colon == NULL || name_end != colon || name_bytes == 0 ||
		    memchr(line, '\r', (size_t)(line_end - line)) != NULL ||
		    memchr(line, '\n', (size_t)(line_end - line)) != NULL;
	    const bool is_length = !malformed &&
				   is_named(line, name_bytes, "content-length");
	    char digits[MOST_LENGTH_DIGITS + 1] = { 0 };
	    bool length_error = is_length && (has_length || value_bytes == 0 ||
					      value_bytes > MOST_LENGTH_DIGITS);
	    if (is_length && !length_error) {
		    memcpy(digits, value, value_bytes);
		    length_error = strspn(digits, DIGITS) != value_bytes;
	    }
	    if ((malformed || length_error) && metrics_enabled)
		    head->metrics_protocol_error = true;
	    has_length = has_length || is_length;
	    if (refusal != 0) {
		    /* framing only after the first refusal */
	    } else if (too_many) {
		    refusal = HTTP_FIELDS_TOO_LARGE;
	    } else if (malformed ||
		       (route == CHAT_ROUTE_MESSAGES &&
			is_named(line, name_bytes, "anthropic-beta"))) {
		    /* or a beta feature asked of Messages: none is offered */
		    refusal = HTTP_BAD_REQUEST;
	    } else if (is_length) {
		    head->body_byte_count =
			    length_error ? 0 :
					   strtoull(digits, NULL, DECIMAL_BASE);
		    refusal = length_error ? HTTP_BAD_REQUEST :
			      head->body_byte_count > CHAT_MOST_BODY_BYTES ?
					     HTTP_CONTENT_TOO_LARGE :
					     0;
	    } else if (is_named(line, name_bytes, "transfer-encoding")) {
		    refusal = HTTP_NOT_IMPLEMENTED;
	    } else if (is_named(line, name_bytes, "expect")) {
		    const bool is_continue =
			    is_named(value, value_bytes, "100-continue");
		    head->expects_continue =
			    is_continue &&
			    is_1_1; /* HTTP/1.0 ignores it (RFC 9110 10.1.1) */
		    refusal = is_continue ? 0 : HTTP_EXPECTATION_FAILED;
	    } else if (is_named(line, name_bytes, "authorization")) {
		    const char *const scheme_end =
			    memchr(value, ' ', value_bytes);
		    const char *credential = scheme_end != NULL ? scheme_end :
								  value_end;
		    while (credential < value_end && *credential == ' ') {
			    credential++;
		    }
		    const bool is_bearer =
			    is_named(value,
				     (size_t)(scheme_end != NULL ?
						      scheme_end - value :
						      0),
				     BEARER) &&
			    credential < value_end &&
			    memchr(credential, '\t',
				   (size_t)(value_end - credential)) == NULL &&
			    memchr(credential, ' ',
				   (size_t)(value_end - credential)) == NULL;
		    refusal = has_credential ? HTTP_BAD_REQUEST : 0;
		    head->key_begin = is_bearer ? (size_t)(credential - bytes) :
						  0;
		    head->key_end = is_bearer ? (size_t)(value_end - bytes) : 0;
		    has_credential = true;
	    } else if (route == CHAT_ROUTE_MESSAGES &&
		       is_named(line, name_bytes, "x-api-key")) {
		    refusal = has_credential ? HTTP_BAD_REQUEST : 0;
		    if (!has_credential) {
			    head->key_begin = (size_t)(value - bytes);
			    head->key_end = (size_t)(value_end - bytes);
		    }
		    has_credential = true;
	    } else if (route == CHAT_ROUTE_MESSAGES &&
		       is_named(line, name_bytes, "anthropic-version")) {
		    const bool legacy =
			    value_bytes == strlen(MESSAGES_LEGACY_VERSION) &&
			    memcmp(value, MESSAGES_LEGACY_VERSION,
				   value_bytes) == 0;
		    const bool current =
			    value_bytes == strlen(MESSAGES_CURRENT_VERSION) &&
			    memcmp(value, MESSAGES_CURRENT_VERSION,
				   value_bytes) == 0;
		    refusal = has_version || (!legacy && !current) ?
				      HTTP_BAD_REQUEST :
				      0;
		    has_version = true;
		    head->messages_legacy_sampling = legacy;
	    } else if (is_named(line, name_bytes, "host")) {
		    refusal = ++host_count > 1 ? HTTP_BAD_REQUEST : 0;
	    }
    }
    head->route = route;
    return refusal != 0				     ? refusal :
	   is_1_1 && host_count == 0		     ? HTTP_BAD_REQUEST :
	   route == sizeof(paths) / sizeof(paths[0]) ? HTTP_NOT_FOUND :
	   !is_named(bytes, (size_t)(method_end - bytes), methods[route]) ?
						       HTTP_METHOD_NOT_ALLOWED :
	   (route == CHAT_ROUTE_COMPLETIONS || route == CHAT_ROUTE_RESPONSES ||
	    route == CHAT_ROUTE_MESSAGES) &&
			   !has_length ?
						       HTTP_LENGTH_REQUIRED :
	   route == CHAT_ROUTE_MESSAGES && !has_version ? HTTP_BAD_REQUEST :
	   route != CHAT_ROUTE_COMPLETIONS && route != CHAT_ROUTE_RESPONSES &&
			   route != CHAT_ROUTE_MESSAGES &&
			   head->body_byte_count != 0 ?
							  HTTP_BAD_REQUEST :
							  0;
}

bool server_head_read_options(const char *bytes, size_t byte_count,
			      bool metrics_enabled, bool messages_enabled,
			      struct server_head *head_out)
{
	const size_t searched = bytes == NULL ? 0 :
				byte_count < SERVER_MOST_HEAD_BYTES ?
						byte_count :
						SERVER_MOST_HEAD_BYTES;
	const char *const end =
		searched != 0 ?
			memmem(bytes, searched, HEAD_END, HEAD_END_BYTES) :
			NULL;
	const size_t head_bytes =
		end != NULL ? (size_t)(end - bytes) + HEAD_END_BYTES : searched;
	struct server_head head = { .head_byte_count = head_bytes,
				    .route = CHAT_ROUTE_HEALTH };
	for (size_t index = 0; index < head_bytes && head.refusal == 0;
	     index++) {
		const unsigned char byte = (unsigned char)bytes[index];
		head.refusal =
			(byte < ' ' && byte != '\t' && byte != '\r' &&
			 byte != '\n') ||
					byte >= DELETE_BYTE ?
				HTTP_BAD_REQUEST :
				0; /* printable ASCII only: nothing after makes it a head */
	}
	if (end == NULL && head.refusal == 0 &&
	    searched < SERVER_MOST_HEAD_BYTES) {
		return false;
	}
	head.refusal = head.refusal != 0 ? head.refusal :
		       end == NULL ? HTTP_FIELDS_TOO_LARGE :
				     head_refusal(bytes, &head, metrics_enabled,
						  messages_enabled);
	*head_out = head;
	return true;
}

bool server_head_read_metrics(const char *bytes, size_t byte_count,
			      bool metrics_enabled,
			      struct server_head *head_out)
{
	return server_head_read_options(bytes, byte_count, metrics_enabled,
					false, head_out);
}

bool server_head_read(const char *bytes, size_t byte_count,
		      struct server_head *head_out)
{
	return server_head_read_metrics(bytes, byte_count, false, head_out);
}
