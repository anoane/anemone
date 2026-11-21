/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Real filesystem refusal cases; the driver supplies an isolated temporary directory and deadline. */
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include "engine/server/server_internal.h"

static const char *test_foreign_directory;
static bool test_nonregular_file;
static unsigned test_session_commands;

static int test_lstat(const char *path, struct stat *status)
{
	const int result = lstat(path, status);
	if (result == 0 && test_foreign_directory != NULL &&
	    strcmp(path, test_foreign_directory) == 0) {
		status->st_uid = geteuid() ^ 1u;
	}
	return result;
}

static int test_fstat(int descriptor, struct stat *status)
{
	const int result = fstat(descriptor, status);
	if (result == 0 && test_nonregular_file) {
		status->st_mode = S_IFIFO;
	}
	return result;
}

MUST_CHECK static result_status
test_session_state(struct inference_engine *engine, uint32_t slot,
		   struct session_state *output)
{
	(void)engine;
	(void)slot;
	*output = (struct session_state){ .reach_count = SERVING_KEPT_GRANULE };
	return STATUS_OK;
}

MUST_CHECK static result_status
test_session_refusal(struct inference_engine *engine,
		     const struct session_command *command)
{
	(void)engine;
	(void)command;
	test_session_commands++;
	return STATUS_INVALID_ARGUMENT;
}

#define engine_session test_session_refusal
#define lstat test_lstat
#define fstat test_fstat
#define engine_session_state test_session_state
#include "../../../source/engine/server/server_sessions.c"
#undef engine_session
#undef engine_session_state
#undef fstat
#undef lstat

#define TEST_ARGUMENT_COUNT 3
#define TEST_PERMISSIVE_GROUP 0720
#define TEST_PERMISSIVE_OTHER 0702
#define TEST_FILE_BYTES 5u

static bool test_file(const char *path, const struct session_store *store)
{
	struct snapshot_head head = { .identity = store->identity,
				      .payload_bytes = 1,
				      .version = SNAPSHOT_VERSION,
				      .token_count = 1,
				      .reach_count = 1,
				      .cache_namespace = { .byte_count = 1 } };
	memcpy(head.magic, SNAPSHOT_MAGIC, sizeof(head.magic));
	uint32_t token = 1;
	unsigned char payload = 1;
	const int descriptor =
		open(path, O_WRONLY | O_CREAT | O_EXCL, FILE_MODE);
	if (descriptor < 0) {
		return false;
	}
	const bool written =
		bytes_moved(descriptor, &head, sizeof(head), 0, true) &&
		bytes_moved(descriptor, &token, sizeof(token), sizeof(head),
			    true) &&
		bytes_moved(descriptor, &payload, sizeof(payload),
			    (off_t)(sizeof(head) + sizeof(token)), true);
	const bool closed = close(descriptor) == 0;
	return written && closed;
}

static bool test_directory(struct http_server *server, const char *name,
			   const char *base)
{
	char session[PATH_CAPACITY], version[PATH_CAPACITY],
		target[PATH_CAPACITY];
	const int session_length =
		snprintf(session, sizeof(session), "%s/sessions", base);
	const int version_length =
		snprintf(version, sizeof(version), "%s/sessions/v4", base);
	const int target_length =
		snprintf(target, sizeof(target), "%s/target", base);
	if (session_length < 0 || (size_t)session_length >= sizeof(session) ||
	    version_length < 0 || (size_t)version_length >= sizeof(version) ||
	    target_length < 0 || (size_t)target_length >= sizeof(target) ||
	    mkdir(session, DIRECTORY_MODE) != 0 ||
	    mkdir(version, DIRECTORY_MODE) != 0) {
		return false;
	}
	const char *const changed = strstr(name, "version") != NULL ? version :
								      session;
	if (strstr(name, "group") != NULL || strstr(name, "other") != NULL) {
		if (chmod(changed, strstr(name, "group") != NULL ?
					   TEST_PERMISSIVE_GROUP :
					   TEST_PERMISSIVE_OTHER) != 0) {
			return false;
		}
	} else if (strstr(name, "owner") != NULL) {
		/* Model foreign ownership without requiring a privileged chown. */
		test_foreign_directory = changed;
	} else if (strstr(name, "symlink") != NULL) {
		if (strcmp(changed, session) == 0 && rmdir(version) != 0) {
			return false;
		}
		if (rmdir(changed) != 0 || mkdir(target, DIRECTORY_MODE) != 0 ||
		    symlink(target, changed) != 0) {
			return false;
		}
	}
	server->session_directory = session;
	const result_status status = server_sessions_open(server);
	const bool safe = strcmp(name, "directory_safe") == 0;
	const bool passed = safe ? status == OK && server->store != NULL :
				   status == STATUS_INPUT_OUTPUT_ERROR &&
					    server->store == NULL;
	server_sessions_close(server);
	server->session_directory = NULL;
	return passed;
}

static bool test_partial(struct http_server *server, const char *name)
{
	struct session_store *const store = server->store;
	uint32_t tokens[SERVING_KEPT_GRANULE] = { 0 };
	server->histories = tokens;
	server->slot_histories[0].processed_count = SERVING_KEPT_GRANULE;
	/* The null engine refuses after opening and removes only our newly created file. */
	if (session_started(server, 0, SESSION_SAVE, 0)) {
		return false;
	}
	char path[PATH_CAPACITY], target[PATH_CAPACITY];
	if (!path_of(store, store->saving, PARTIAL_SUFFIX, path) ||
	    !path_of(store, "target", "", target)) {
		return false;
	}
	const char *const held = strstr(name, "symlink") != NULL ? target :
								   path;
	const int descriptor =
		open(held, O_WRONLY | O_CREAT | O_EXCL, FILE_MODE);
	if (descriptor < 0) {
		return false;
	}
	char expected[TEST_FILE_BYTES] = "held";
	const bool written =
		bytes_moved(descriptor, expected, sizeof(expected), 0, true);
	const bool closed = close(descriptor) == 0;
	if (!written || !closed ||
	    (held == target && symlink(target, path) != 0)) {
		return false;
	}
	const bool refused = !session_started(server, 0, SESSION_SAVE, 0);
	struct stat status;
	char actual[TEST_FILE_BYTES] = { 0 };
	const int read_descriptor = open(held, O_RDONLY | O_CLOEXEC);
	const bool untouched = read_descriptor >= 0 &&
			       fstat(read_descriptor, &status) == 0 &&
			       status.st_size == (off_t)sizeof(expected) &&
			       bytes_moved(read_descriptor, actual,
					   sizeof(actual), 0, false) &&
			       memcmp(actual, expected, sizeof(actual)) == 0;
	if (read_descriptor >= 0) {
		(void)close(read_descriptor);
	}
	server->histories = NULL;
	return refused && untouched;
}

static bool test_snapshot(struct http_server *server, const char *name)
{
	struct session_store *const store = server->store;
	char path[PATH_CAPACITY], target[PATH_CAPACITY];
	if (!path_of(store, "hostile", SNAPSHOT_SUFFIX, path) ||
	    !path_of(store, "target", "", target)) {
		return false;
	}
	if (strncmp(name, "partial_", strlen("partial_")) == 0) {
		return test_partial(server, name);
	}
	if (strstr(name, "fifo") != NULL) {
		if (mkfifo(path, FILE_MODE) != 0) {
			return false;
		}
	} else if (strstr(name, "symlink") != NULL) {
		if (!test_file(target, store) || symlink(target, path) != 0) {
			return false;
		}
	} else {
		if (!test_file(path, store)) {
			return false;
		}
		test_nonregular_file = true;
	}
	if (strstr(name, "read") != NULL) {
		snapshot_read(store, "hostile.snapshot");
		struct stat removed;
		return store->snapshot_count == 0 &&
		       lstat(path, &removed) != 0 && errno == ENOENT;
	}
	uint32_t held = 1, loaded = 0;
	struct snapshot *const snapshot = &store->snapshots[0];
	*snapshot =
		(struct snapshot){ .token_ids = &held,
				   .token_count = 1,
				   .reach_count = 1,
				   .file_bytes = sizeof(struct snapshot_head) +
						 sizeof(held) + 1u,
				   .cache_namespace = { .byte_count = 1 } };
	if (strstr(name, "match") != NULL) {
		const int descriptor =
			open(path, O_RDONLY | O_NONBLOCK | O_CLOEXEC);
		const bool refused = descriptor >= 0 &&
				     !snapshot_matches(descriptor, store,
						       snapshot, &loaded,
						       sizeof(loaded));
		if (descriptor >= 0) {
			(void)close(descriptor);
		}
		return refused;
	}
	/* The production load path frees the indexed tokens on refusal. */
	snapshot->token_ids = malloc(sizeof(held));
	if (snapshot->token_ids == NULL) {
		return false;
	}
	*snapshot->token_ids = held;
	memcpy(snapshot->name, "hostile", sizeof("hostile"));
	store->snapshot_count = 1;
	return !session_started(server, 0, SESSION_LOAD, 0) &&
	       store->snapshot_count == 0 && test_session_commands == 0;
}

int main(int count, char **arguments)
{
	(void)test_lstat;
	if (count != TEST_ARGUMENT_COUNT) {
		return 1;
	}
	struct http_server *const server = calloc(1, sizeof(*server));
	if (server == NULL) {
		return 1;
	}
	server->description.context_token_limit = SERVING_KEPT_GRANULE;
	server->description.session_slot_count = 1;
	/* the run's per-slot tables, as a run starts them (server_tables): session_started indexes them */
	bool passed = server_tables(server) == STATUS_OK;
	if (!passed) {
		(void)fputs(
			"FAILED: the server takes its tables for one slot\n",
			stderr);
	} else if (strncmp(arguments[1], "directory_", strlen("directory_")) ==
		   0) {
		passed = test_directory(server, arguments[1], arguments[2]);
	} else {
		server->store = calloc(1, sizeof(*server->store));
		passed = server->store != NULL;
		if (passed) {
			server->store->identity = 1;
			server->store->context_limit = SERVING_KEPT_GRANULE;
			const int length =
				snprintf(server->store->directory,
					 sizeof(server->store->directory), "%s",
					 arguments[2]);
			passed = length >= 0 &&
				 (size_t)length <
					 sizeof(server->store->directory) &&
				 test_snapshot(server, arguments[1]);
		}
		server_sessions_close(server);
	}
	server_tables_free(server);
	free(server);
	if (!passed) {
		(void)fprintf(stderr, "FAILED: snapshot path %s\n",
			      arguments[1]);
	}
	return passed ? 0 : 1;
}
