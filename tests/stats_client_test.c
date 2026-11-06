/* SPDX-License-Identifier: AGPL-3.0-or-later */
#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <poll.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#include "anemone/stats_client.h"

#define TEST_ARGUMENT_CAPACITY 10u
#define TEST_PORT 8321u
#define TEST_INTERVAL 100u
#define TEST_SERVER_WAIT_MS 10000
#define TEST_REQUEST_BYTES 8192u
#define TEST_SEND_PIECE_BYTES 7u
#define TEST_PRIVATE_MODE 0600
#define TEST_SYNTHETIC_KEY "stats_fixture_key_12345"
#define TEST_KEY_PATH "/tmp/anemone_stats_client_key_XXXXXX"
#define TEST_RESPONSE \
	"HTTP/1.1 200 OK\r\nContent-Length: 11\r\n\r\n{\"value\":1}"
#define TEST_CLI_WORDS 6u
#define TEST_CLI_MINIMUM_WORDS 5
#define TEST_TERMINAL_ESCAPE 27

enum test_client_mode {
	TEST_FETCH,
	TEST_CLI_JSON,
	TEST_CLI_TEXT,
	TEST_CLI_TABLE
};

static int failures;

static void expect(bool condition, const char *message)
{
	if (!condition) {
		(void)printf("FAILED: %s\n", message);
		failures++;
	}
}

static bool parsed(const char *const *words, size_t count,
		   struct stats_client_options *options)
{
	char *arguments[TEST_ARGUMENT_CAPACITY] = { 0 };
	char storage[TEST_ARGUMENT_CAPACITY][TEST_REQUEST_BYTES];
	if (count > TEST_ARGUMENT_CAPACITY) {
		return false;
	}
	for (size_t index = 0; index < count; index++) {
		const int written = snprintf(storage[index],
					     sizeof(storage[index]), "%s",
					     words[index]);
		if (written < 0 || (size_t)written >= sizeof(storage[index])) {
			return false;
		}
		arguments[index] = storage[index];
	}
	return stats_client_parse((int)count, arguments, options);
}

static void test_options(void)
{
	struct stats_client_options options = { 0 };
	const char *const valid[] = { "anemone",
				      "stats",
				      "--port=8321",
				      "--host=::1",
				      "--interval_ms=100",
				      "--format=json",
				      "--once" };
	expect(parsed(valid, sizeof(valid) / sizeof(valid[0]), &options) &&
		       options.port == TEST_PORT &&
		       options.interval_ms == TEST_INTERVAL &&
		       options.is_json && options.once,
	       "stats CLI admits all documented options and IPv6");
	const char *const table[] = { "anemone", "stats", "--port=8321",
				      "--format=table", "--once" };
	expect(parsed(table, sizeof(table) / sizeof(table[0]), &options) &&
		       options.is_table && !options.is_json,
	       "table format is admitted independently of JSON");
	const char *const missing[] = { "anemone", "stats", "--once" };
	expect(!parsed(missing, sizeof(missing) / sizeof(missing[0]), &options),
	       "port is mandatory");
	const char *const invalid[] = { "--port=0",
					"--port=65536",
					"--port=-1",
					"--host=localhost",
					"--host=127.0.0.1\r\n",
					"--interval_ms=0",
					"--interval_ms=99",
					"--interval_ms=60001",
					"--format=xml",
					"--format=",
					"--once=yes",
					"--key_file=",
					"--port=1234",
					"x",
					"--" };
	for (size_t index = 0; index < sizeof(invalid) / sizeof(invalid[0]);
	     index++) {
		const char *const words[] = { "anemone", "stats", "--port=8321",
					      invalid[index] };
		expect(!parsed(words, sizeof(words) / sizeof(words[0]),
			       &options),
		       "invalid or duplicate option refused");
	}
	const char *const duplicate_once[] = { "anemone", "stats",
					       "--port=8321", "--once",
					       "--once" };
	expect(!parsed(duplicate_once,
		       sizeof(duplicate_once) / sizeof(duplicate_once[0]),
		       &options),
	       "duplicate flags refused");
}

struct fixture {
	int listener;
	const char *response;
	bool authenticated;
	bool succeeded;
};

static void *test_server(void *opaque)
{
	struct fixture *const fixture = opaque;
	struct pollfd entry = { fixture->listener, POLLIN, 0 };
	if (poll(&entry, 1, TEST_SERVER_WAIT_MS) <= 0) {
		return NULL;
	}
	const int client = accept4(fixture->listener, NULL, NULL,
				   SOCK_CLOEXEC | SOCK_NONBLOCK);
	if (client < 0) {
		return NULL;
	}
	char request[TEST_REQUEST_BYTES] = { 0 };
	size_t count = 0;
	while (count < sizeof(request) - 1u &&
	       strstr(request, "\r\n\r\n") == NULL) {
		entry = (struct pollfd){ client, POLLIN, 0 };
		if (poll(&entry, 1, TEST_SERVER_WAIT_MS) <= 0) {
			break;
		}
		const ssize_t received = recv(client, request + count,
					      sizeof(request) - count - 1u, 0);
		if (received <= 0) {
			break;
		}
		count += (size_t)received;
		request[count] = '\0';
	}
	fixture->authenticated =
		strstr(request, "Authorization: Bearer " TEST_SYNTHETIC_KEY
				"\r\n") != NULL;
	bool succeeded =
		strstr(request, "GET /v1/anemone/stats HTTP/1.1\r\n") ==
			request &&
		strstr(request, "\r\n\r\n") != NULL;
	explicit_bzero(request, sizeof(request));
	const size_t length = strlen(fixture->response);
	size_t sent = 0;
	while (succeeded && sent < length) {
		entry = (struct pollfd){ client, POLLOUT, 0 };
		if (poll(&entry, 1, TEST_SERVER_WAIT_MS) <= 0) {
			succeeded = false;
			break;
		}
		const size_t remaining = length - sent;
		const ssize_t written = send(client, fixture->response + sent,
					     remaining < TEST_SEND_PIECE_BYTES ?
						     remaining :
						     TEST_SEND_PIECE_BYTES,
					     MSG_NOSIGNAL);
		if (written > 0) {
			sent += (size_t)written;
		} else if (written == 0 || (errno != EINTR && errno != EAGAIN &&
					    errno != EWOULDBLOCK)) {
			succeeded = false;
		}
	}
	(void)close(client);
	fixture->succeeded = succeeded;
	return NULL;
}

static int cli_captured(uint32_t port, const char *key_path,
			enum test_client_mode mode, struct chat_buffer *output)
{
	char program[] = "anemone", command[] = "stats", once[] = "--once";
	char json_format[] = "--format=json", text_format[] = "--format=text";
	char table_format[] = "--format=table";
	char port_word[TEST_REQUEST_BYTES], key_word[TEST_REQUEST_BYTES];
	const int port_bytes =
		snprintf(port_word, sizeof(port_word), "--port=%" PRIu32, port);
	const int key_bytes = snprintf(key_word, sizeof(key_word),
				       "--key_file=%s",
				       key_path != NULL ? key_path : "");
	if (port_bytes <= 0 || (size_t)port_bytes >= sizeof(port_word) ||
	    key_bytes <= 0 || (size_t)key_bytes >= sizeof(key_word)) {
		expect(false, "CLI fixture arguments fit");
		return -1;
	}
	char *arguments[TEST_CLI_WORDS] = { program,
					    command,
					    port_word,
					    mode == TEST_CLI_JSON ?
						    json_format :
					    mode == TEST_CLI_TABLE ?
						    table_format :
						    text_format,
					    once,
					    key_word };
	FILE *const capture = tmpfile();
	if (capture == NULL) {
		expect(false, "CLI output capture opens");
		return -1;
	}
	const int saved_stdout = dup(STDOUT_FILENO);
	if (saved_stdout < 0 || fflush(stdout) != 0 ||
	    dup2(fileno(capture), STDOUT_FILENO) < 0) {
		if (saved_stdout >= 0) {
			(void)close(saved_stdout);
		}
		const int closed = fclose(capture);
		(void)closed;
		expect(false, "CLI stdout capture starts");
		return -1;
	}
	const int result = stats_client_cli(
		TEST_CLI_MINIMUM_WORDS + (key_path != NULL ? 1 : 0), arguments);
	const bool flushed = fflush(stdout) == 0;
	int restored;
	do {
		restored = dup2(saved_stdout, STDOUT_FILENO);
	} while (restored < 0 && errno == EINTR);
	const bool saved_closed = close(saved_stdout) == 0;
	if (restored < 0) {
		const int closed = fclose(capture);
		(void)closed;
		(void)fprintf(stderr, "FAILED: CLI stdout restore\n");
		failures++;
		return -1;
	}
	char captured[TEST_REQUEST_BYTES];
	const bool rewound = fseek(capture, 0, SEEK_SET) == 0;
	const size_t read_bytes =
		rewound ? fread(captured, 1, sizeof(captured), capture) : 0;
	const bool complete = rewound && ferror(capture) == 0 &&
			      feof(capture) != 0;
	const int closed = fclose(capture);
	expect(flushed && saved_closed && complete && closed == 0,
	       "CLI output captured and stdout restored");
	if (!flushed || !saved_closed || !complete || closed != 0) {
		return -1;
	}
	chat_buffer_write(output, captured, read_bytes);
	return output->is_short ? -1 : result;
}

static void fetched(const char *response, const char *key_path,
		    bool should_succeed, enum test_client_mode mode)
{
	const int listener = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
	struct sockaddr_in address = {
		.sin_family = AF_INET, .sin_addr.s_addr = htonl(INADDR_LOOPBACK)
	};
	socklen_t address_length = sizeof(address);
	const bool listening = listener >= 0 &&
			       bind(listener, (const struct sockaddr *)&address,
				    sizeof(address)) == 0 &&
			       getsockname(listener,
					   (struct sockaddr *)&address,
					   &address_length) == 0 &&
			       listen(listener, 1) == 0;
	expect(listening, "test listener opens");
	if (!listening) {
		if (listener >= 0) {
			(void)close(listener);
		}
		return;
	}
	struct fixture fixture = { .listener = listener, .response = response };
	pthread_t thread;
	const bool started =
		pthread_create(&thread, NULL, test_server, &fixture) == 0;
	expect(started, "test server thread starts");
	if (started) {
		const struct stats_client_options options = {
			.host = "127.0.0.1",
			.port = ntohs(address.sin_port),
			.key_file = key_path
		};
		struct chat_buffer output = { 0 };
		if (mode == TEST_FETCH) {
			const result_status status =
				stats_client_fetch(&options, &output);
			expect((status == STATUS_OK) == should_succeed,
			       "stats client fetch status");
		} else {
			const int result = cli_captured(options.port, key_path,
							mode, &output);
			expect((result == 0) == should_succeed,
			       "stats CLI once status");
		}
		if (should_succeed) {
			if (mode == TEST_CLI_TABLE) {
				expect(output.bytes != NULL &&
					       memmem(output.bytes,
						      output.count, "user",
						      strlen("user")) != NULL &&
					       memmem(output.bytes,
						      output.count, "value: 1",
						      strlen("value: 1")) ==
						       NULL,
				       "stats CLI table prints the compact header without the metric dump");
			} else if (mode == TEST_CLI_TEXT) {
				expect(output.bytes != NULL &&
					       memmem(output.bytes,
						      output.count,
						      "value: 1\n",
						      strlen("value: 1\n")) !=
						       NULL &&
					       memmem(output.bytes,
						      output.count, "user",
						      strlen("user")) != NULL &&
					       memchr(output.bytes,
						      TEST_TERMINAL_ESCAPE,
						      output.count) == NULL,
				       "stats CLI once renders timestamped text without ANSI in logs");
			} else {
				const char *const expected =
					mode == TEST_CLI_JSON ?
						"{\"value\":1}\n" :
						"{\"value\":1}";
				expect(output.count == strlen(expected) &&
					       memcmp(output.bytes, expected,
						      output.count) == 0,
				       "response body survives partial reads and CLI JSON has one newline");
			}
			expect(output.bytes != NULL &&
				       memmem(output.bytes, output.count,
					      TEST_SYNTHETIC_KEY,
					      strlen(TEST_SYNTHETIC_KEY)) ==
					       NULL,
			       "stats output never contains the bearer key");
		}
		chat_buffer_free(&output);
		expect(pthread_join(thread, NULL) == 0, "test server joins");
		expect(!should_succeed || fixture.succeeded,
		       "server receives the correct endpoint");
		expect(fixture.authenticated == (key_path != NULL),
		       "only file-sourced bearer key is sent");
	}
	(void)close(listener);
}

int main(void)
{
	test_options();
	fetched(TEST_RESPONSE, NULL, true, TEST_FETCH);
	fetched(TEST_RESPONSE, NULL, true, TEST_CLI_JSON);
	fetched(TEST_RESPONSE, NULL, true, TEST_CLI_TEXT);
	fetched(TEST_RESPONSE, NULL, true, TEST_CLI_TABLE);
	fetched("HTTP/1.1 401 Unauthorized\r\nContent-Length: 2\r\n\r\n{}",
		NULL, false, TEST_FETCH);
	fetched("HTTP/1.1 200 OK\r\nContent-Length: 3\r\n\r\n{{}", NULL, false,
		TEST_FETCH);
	fetched("HTTP/1.1 200 OK\r\nContent-Length: 2\r\nContent-Length: 2\r\n\r\n{}",
		NULL, false, TEST_FETCH);
	fetched("HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n",
		NULL, false, TEST_FETCH);
	fetched("HTTP/1.1 200 OK\r\nContent-Length: 2097153\r\n\r\n{}", NULL,
		false, TEST_FETCH);
	char key_path[] = TEST_KEY_PATH;
	const int descriptor = mkstemp(key_path);
	const bool created = descriptor >= 0 &&
			     write(descriptor, TEST_SYNTHETIC_KEY,
				   strlen(TEST_SYNTHETIC_KEY)) ==
				     (ssize_t)strlen(TEST_SYNTHETIC_KEY);
	expect(created, "private synthetic key fixture written");
	if (descriptor >= 0) {
		expect(close(descriptor) == 0, "synthetic key file closes");
	}
	if (created) {
		fetched(TEST_RESPONSE, key_path, true, TEST_FETCH);
		fetched(TEST_RESPONSE, key_path, true, TEST_CLI_JSON);
		fetched(TEST_RESPONSE, key_path, true, TEST_CLI_TEXT);
	}
	if (descriptor >= 0) {
		expect(unlink(key_path) == 0, "synthetic key removed");
	}
	(void)printf("stats client: %s\n", failures == 0 ? "PASS" : "FAIL");
	return failures == 0 ? 0 : 1;
}
