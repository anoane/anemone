/* SPDX-License-Identifier: AGPL-3.0-or-later */
#include "anemone/stats_client.h"

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <math.h>
#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#include "anemone/server.h"
#include "json.h"

#define STATS_CLIENT_SECONDS 10.0
#define STATS_CLIENT_MILLISECONDS 1000.0
#define STATS_CLIENT_NANOSECONDS 1000000000.0
#define STATS_CLIENT_RESPONSE_BYTES 2097152u
#define STATS_CLIENT_RECEIVE_BYTES 16384u
#define STATS_CLIENT_REQUEST_BYTES (SERVER_KEY_MOST_BYTES + 512u)
#define STATS_CLIENT_OPTIONS 6u
#define STATS_CLIENT_OPTION_START 2
#define STATS_CLIENT_USAGE 2
#define STATS_CLIENT_DECIMAL_BASE 10
#define STATS_CLIENT_STATUS_OFFSET 9u
#define STATS_CLIENT_HTTP_OK "200 "
#define STATS_CLIENT_REQUEST_PREFIX "GET /v1/anemone/stats HTTP/1.1\r\nHost: "

static double client_seconds(void)
{
	struct timespec now = { 0 };
	return clock_gettime(CLOCK_MONOTONIC, &now) == 0 ?
		       (double)now.tv_sec +
			       (double)now.tv_nsec / STATS_CLIENT_NANOSECONDS :
		       -1.0;
}

static bool number_read(const char *text, uint32_t least, uint32_t most,
			uint32_t *value_out)
{
	char *end = NULL;
	if (*text < '0' || *text > '9') {
		return false;
	}
	errno = 0;
	const unsigned long value =
		strtoul(text, &end, STATS_CLIENT_DECIMAL_BASE);
	if (errno != 0 || *end != '\0' || value < least || value > most) {
		return false;
	}
	*value_out = (uint32_t)value;
	return true;
}

static bool numeric_host(const char *host)
{
	struct in6_addr address_6;
	struct in_addr address_4;
	return host != NULL && (inet_pton(AF_INET, host, &address_4) == 1 ||
				inet_pton(AF_INET6, host, &address_6) == 1);
}

bool stats_client_parse(int argument_count, char *const *arguments,
			struct stats_client_options *options_out)
{
	static const char *const names[STATS_CLIENT_OPTIONS] = {
		"--host=",	  "--port=",   "--key_file=",
		"--interval_ms=", "--format=", "--once"
	};
	enum {
		HOST_OPTION,
		PORT_OPTION,
		KEY_OPTION,
		INTERVAL_OPTION,
		FORMAT_OPTION,
		ONCE_OPTION
	};
	bool given[STATS_CLIENT_OPTIONS] = { false };
	struct stats_client_options options = {
		.host = SERVER_DEFAULT_ADDRESS,
		.interval_ms = LIVE_STATS_DEFAULT_INTERVAL_MS
	};
	if (options_out == NULL || arguments == NULL ||
	    argument_count < STATS_CLIENT_OPTION_START ||
	    arguments[1] == NULL || strcmp(arguments[1], "stats") != 0) {
		return false;
	}
	for (int index = STATS_CLIENT_OPTION_START; index < argument_count;
	     index++) {
		bool matched = false;
		if (arguments[index] == NULL) {
			return false;
		}
		for (uint32_t option = 0;
		     option < STATS_CLIENT_OPTIONS && !matched; option++) {
			const size_t length = strlen(names[option]);
			if (strncmp(arguments[index], names[option], length) !=
			    0) {
				continue;
			}
			const char *const value = arguments[index] + length;
			if (given[option]) {
				return false;
			}
			switch (option) {
			case HOST_OPTION:
				options.host = value;
				matched = numeric_host(value);
				break;
			case PORT_OPTION:
				matched = number_read(value, 1,
						      SERVER_MOST_PORT,
						      &options.port);
				break;
			case KEY_OPTION:
				options.key_file = value;
				matched = *value != '\0';
				break;
			case INTERVAL_OPTION:
				matched = number_read(
					value, LIVE_STATS_LEAST_INTERVAL_MS,
					LIVE_STATS_MOST_INTERVAL_MS,
					&options.interval_ms);
				break;
			case FORMAT_OPTION:
				matched = strcmp(value, "text") == 0 ||
					  strcmp(value, "json") == 0 ||
					  strcmp(value, "table") == 0;
				options.is_json = strcmp(value, "json") == 0;
				options.is_table = strcmp(value, "table") == 0;
				break;
			case ONCE_OPTION:
				options.once = true;
				matched = *value == '\0';
				break;
			default:
				return false;
			}
			given[option] = matched;
			if (!matched) {
				return false;
			}
		}
		if (!matched) {
			return false;
		}
	}
	if (!given[PORT_OPTION]) {
		return false;
	}
	*options_out = options;
	return true;
}

static bool socket_ready(int descriptor, short events, double deadline)
{
	for (;;) {
		const double now = client_seconds();
		const double remaining = deadline - now;
		if (now < 0.0 || remaining <= 0.0) {
			return false;
		}
		struct pollfd entry = { descriptor, events, 0 };
		const int ready =
			poll(&entry, 1,
			     (int)ceil(remaining * STATS_CLIENT_MILLISECONDS));
		if (ready >= 0) {
			return ready > 0 && (entry.revents & events) != 0;
		}
		if (errno != EINTR) {
			return false;
		}
	}
}

static int client_connected(const struct stats_client_options *options,
			    double deadline)
{
	struct sockaddr_in address_4 = { .sin_family = AF_INET,
					 .sin_port = htons(
						 (uint16_t)options->port) };
	struct sockaddr_in6 address_6 = { .sin6_family = AF_INET6,
					  .sin6_port = htons(
						  (uint16_t)options->port) };
	const bool is_4 =
		inet_pton(AF_INET, options->host, &address_4.sin_addr) == 1;
	if (!is_4 &&
	    inet_pton(AF_INET6, options->host, &address_6.sin6_addr) != 1) {
		return -1;
	}
	const int descriptor =
		socket(is_4 ? AF_INET : AF_INET6,
		       SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
	if (descriptor < 0) {
		return -1;
	}
	const int connected =
		connect(descriptor,
			is_4 ? (const struct sockaddr *)&address_4 :
			       (const struct sockaddr *)&address_6,
			is_4 ? sizeof(address_4) : sizeof(address_6));
	int error = 0;
	socklen_t length = sizeof(error);
	if ((connected < 0 && (errno != EINPROGRESS ||
			       !socket_ready(descriptor, POLLOUT, deadline))) ||
	    getsockopt(descriptor, SOL_SOCKET, SO_ERROR, &error, &length) !=
		    0 ||
	    error != 0) {
		(void)close(descriptor);
		return -1;
	}
	return descriptor;
}

static bool request_sent(int descriptor, const char *request, size_t length,
			 double deadline)
{
	size_t sent = 0;
	while (sent < length) {
		if (!socket_ready(descriptor, POLLOUT, deadline)) {
			return false;
		}
		const ssize_t written = send(descriptor, request + sent,
					     length - sent, MSG_NOSIGNAL);
		if (written > 0) {
			sent += (size_t)written;
		} else if (written == 0 ||
			   (errno != EAGAIN && errno != EWOULDBLOCK &&
			    errno != EINTR)) {
			return false;
		}
	}
	return true;
}

/* HTTP is bounded before parsing. This endpoint supplies Content-Length and never Transfer-Encoding. */
static bool response_body(struct chat_buffer *response, size_t *body_begin_out,
			  size_t *body_length_out)
{
	char *const head_end = strstr(response->bytes, "\r\n\r\n");
	if (head_end == NULL) {
		return response->count <= SERVER_MOST_HEAD_BYTES;
	}
	const size_t head_bytes =
		(size_t)(head_end - response->bytes) + strlen("\r\n\r\n");
	if (head_bytes > SERVER_MOST_HEAD_BYTES ||
	    (strncmp(response->bytes, "HTTP/1.1 ",
		     STATS_CLIENT_STATUS_OFFSET) != 0 &&
	     strncmp(response->bytes, "HTTP/1.0 ",
		     STATS_CLIENT_STATUS_OFFSET) != 0) ||
	    strncmp(response->bytes + STATS_CLIENT_STATUS_OFFSET,
		    STATS_CLIENT_HTTP_OK, strlen(STATS_CLIENT_HTTP_OK)) != 0) {
		return false;
	}
	char *cursor = strstr(response->bytes, "\r\n");
	bool found_length = false;
	size_t body_length = 0;
	while (cursor != NULL && cursor < head_end) {
		cursor += strlen("\r\n");
		char *const end = strstr(cursor, "\r\n");
		if (end == NULL || end > head_end) {
			return false;
		}
		if (strncasecmp(cursor, "Transfer-Encoding:",
				strlen("Transfer-Encoding:")) == 0) {
			return false;
		}
		if (strncasecmp(cursor, "Content-Length:",
				strlen("Content-Length:")) == 0) {
			char *number = cursor + strlen("Content-Length:");
			while (number < end &&
			       (*number == ' ' || *number == '\t')) {
				number++;
			}
			const char saved = *end;
			*end = '\0';
			uint32_t parsed = 0;
			const bool valid =
				!found_length &&
				number_read(number, 1,
					    STATS_CLIENT_RESPONSE_BYTES -
						    SERVER_MOST_HEAD_BYTES,
					    &parsed);
			*end = saved;
			if (!valid) {
				return false;
			}
			found_length = true;
			body_length = parsed;
		}
		cursor = end;
	}
	if (!found_length) {
		return false;
	}
	*body_begin_out = head_bytes;
	*body_length_out = body_length;
	return true;
}

result_status stats_client_fetch(const struct stats_client_options *options,
				 struct chat_buffer *output)
{
	if (options == NULL || output == NULL || !numeric_host(options->host) ||
	    options->port == 0 || options->port > SERVER_MOST_PORT) {
		return STATUS_INVALID_ARGUMENT;
	}
	char key[SERVER_KEY_MOST_BYTES] = { 0 };
	size_t key_length = 0;
	result_status status =
		options->key_file == NULL ?
			STATUS_OK :
			server_key_read(options->key_file, key, &key_length);
	char request[STATS_CLIENT_REQUEST_BYTES];
	const bool is_6 = strchr(options->host, ':') != NULL;
	const int length = snprintf(
		request, sizeof(request),
		STATS_CLIENT_REQUEST_PREFIX
		"%s%s%s:%" PRIu32 "\r\nConnection: close\r\n%s%.*s%s\r\n",
		is_6 ? "[" : "", options->host, is_6 ? "]" : "", options->port,
		key_length != 0 ? "Authorization: Bearer " : "",
		(int)key_length, key, key_length != 0 ? "\r\n" : "");
	explicit_bzero(key, sizeof(key));
	const double started = client_seconds();
	const double deadline = started + STATS_CLIENT_SECONDS;
	status = started < 0.0 ? STATUS_INPUT_OUTPUT_ERROR : status;
	int descriptor = -1;
	if (status == STATUS_OK) {
		descriptor = client_connected(options, deadline);
		status = descriptor >= 0 && length > 0 &&
					 (size_t)length < sizeof(request) &&
					 request_sent(descriptor, request,
						      (size_t)length,
						      deadline) ?
				 STATUS_OK :
				 STATUS_INPUT_OUTPUT_ERROR;
	}
	explicit_bzero(request, sizeof(request));
	struct chat_buffer response = { 0 };
	size_t body_begin = 0, body_length = 0;
	while (status == STATUS_OK &&
	       (body_begin == 0 || response.count < body_begin + body_length)) {
		if (response.count > STATS_CLIENT_RESPONSE_BYTES -
					     STATS_CLIENT_RECEIVE_BYTES ||
		    !socket_ready(descriptor, POLLIN, deadline)) {
			status = STATUS_INPUT_OUTPUT_ERROR;
			break;
		}
		char *const room = chat_buffer_room(
			&response, STATS_CLIENT_RECEIVE_BYTES + 1u);
		if (room == NULL) {
			status = STATUS_OUT_OF_MEMORY;
			break;
		}
		const ssize_t received =
			recv(descriptor, room, STATS_CLIENT_RECEIVE_BYTES, 0);
		if (received < 0 && (errno == EINTR || errno == EAGAIN ||
				     errno == EWOULDBLOCK)) {
			continue;
		}
		if (received <= 0 ||
		    (size_t)received > STATS_CLIENT_RECEIVE_BYTES ||
		    response.count >= response.capacity ||
		    (size_t)received >= response.capacity - response.count) {
			status = STATUS_INPUT_OUTPUT_ERROR;
			break;
		}
		response.count += (size_t)received;
		char *const terminator = chat_buffer_room(&response, 1);
		if (terminator == NULL) {
			status = STATUS_OUT_OF_MEMORY;
			break;
		}
		*terminator = '\0';
		if (body_begin == 0 &&
		    !response_body(&response, &body_begin, &body_length)) {
			status = STATUS_INPUT_OUTPUT_ERROR;
		}
	}
	if (descriptor >= 0) {
		(void)close(descriptor);
	}
	struct json_value document = { 0 };
	if (status == STATUS_OK &&
	    (json_read_document(response.bytes + body_begin, body_length,
				&document) != JSON_DOCUMENT_VALID ||
	     document.text[document.begin] != '{')) {
		status = STATUS_INPUT_OUTPUT_ERROR;
	}
	if (status == STATUS_OK) {
		memmove(response.bytes,
			response.bytes + body_begin + document.begin,
			document.end - document.begin);
		response.count = document.end - document.begin;
		response.bytes[response.count] = '\0';
		chat_buffer_free(output);
		*output = response;
	} else {
		chat_buffer_free(&response);
	}
	return status;
}

int stats_client_cli(int argument_count, char *const *arguments)
{
	struct stats_client_options options = { 0 };
	if (!stats_client_parse(argument_count, arguments, &options)) {
		(void)fprintf(
			stderr,
			"usage: anemone stats [--host=127.0.0.1] --port=N [--key_file=F]"
			" [--interval_ms=100..60000] [--format=text|json|table] [--once]\n");
		return STATS_CLIENT_USAGE;
	}
	const bool is_tty = isatty(STDOUT_FILENO) != 0;
	result_status status = STATUS_OK;
	do {
		struct chat_buffer json = { 0 }, text = { 0 };
		status = stats_client_fetch(&options, &json);
		uint32_t columns = LIVE_STATS_DEFAULT_COLUMNS;
		struct winsize terminal = { 0 };
		if (is_tty &&
		    ioctl(STDOUT_FILENO, TIOCGWINSZ, &terminal) == 0) {
			columns = terminal.ws_col < LIVE_STATS_LEAST_COLUMNS ?
					  LIVE_STATS_LEAST_COLUMNS :
				  terminal.ws_col > LIVE_STATS_MOST_COLUMNS ?
					  LIVE_STATS_MOST_COLUMNS :
					  terminal.ws_col;
		}
		if (status == STATUS_OK && !options.is_json) {
			status = (options.is_table ? live_stats_table_entry :
						     live_stats_text)(
				json.bytes, json.count, columns, is_tty, &text);
		}
		const struct chat_buffer *const view = options.is_json ? &json :
									 &text;
		if (status == STATUS_OK &&
		    (fwrite(view->bytes, 1, view->count, stdout) !=
			     view->count ||
		     (options.is_json && fputc('\n', stdout) == EOF) ||
		     fflush(stdout) != 0)) {
			status = STATUS_INPUT_OUTPUT_ERROR;
		}
		chat_buffer_free(&json);
		chat_buffer_free(&text);
		if (!options.once && status == STATUS_OK &&
		    poll(NULL, 0, (int)options.interval_ms) < 0 &&
		    errno != EINTR) {
			status = STATUS_INPUT_OUTPUT_ERROR;
		}
	} while (!options.once && status == STATUS_OK);
	if (status != STATUS_OK) {
		(void)fprintf(stderr, "stats: %s\n", status_text(status));
	}
	return status == STATUS_OK ? 0 : 1;
}
