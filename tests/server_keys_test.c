/* SPDX-License-Identifier: AGPL-3.0-or-later */
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "anemone/hooks.h"
#include "anemone/server.h"

#define PATH_PATTERN "/tmp/anemone_keys_XXXXXX"
#define PRIVATE_MODE 0600
#define PUBLIC_MODE 0644
#define TEST_USERS 3u
#define USER_NAME "a.b-c_d"
#define LEGACY_KEY "legacy-fake-0123456789"
#define LINE_BYTES (SERVER_NAMED_KEY_BYTES + 1u)
#define MOST_FILE_BYTES (LINE_BYTES * (SERVER_MOST_USERS + 1u))
#define HEX_PER_BYTE 2u

static int failures;

static void expect(bool condition, const char *message)
{
	if (!condition) {
		(void)printf("FAILED: %s\n", message);
		failures++;
	}
}

static void file_write(const char *path, const char *bytes, size_t length)
{
	const int descriptor = open(path, O_WRONLY | O_TRUNC);
	expect(descriptor >= 0 &&
		       write(descriptor, bytes, length) == (ssize_t)length &&
		       close(descriptor) == 0,
	       "write generated fixture");
}

static void refused(const char *path, const char *bytes, size_t length,
		    struct server_keys_record *keys)
{
	file_write(path, bytes, length);
	const uint32_t before = keys->count;
	expect(server_keys_read(path, keys) == STATUS_INVALID_ARGUMENT &&
		       keys->count == before,
	       "refused file leaves output unchanged");
}

int main(void)
{
	char path[] = PATH_PATTERN;
	const int descriptor = mkstemp(path);
	expect(descriptor >= 0 && close(descriptor) == 0,
	       "private fixture opens");
	struct server_keys_record *const keys = calloc(1, sizeof(*keys));
	if (keys == NULL || descriptor < 0) {
		free(keys);
		return 1;
	}
	char first[SERVER_NAMED_KEY_BYTES], second[SERVER_NAMED_KEY_BYTES];
	expect(server_key_generate(USER_NAME, first) == STATUS_OK &&
		       server_key_generate(USER_NAME, second) == STATUS_OK,
	       "getrandom keys generate");
	expect(strlen(first) == strlen(USER_NAME) + 1u +
					(size_t)HEX_PER_BYTE *
						SERVER_RANDOM_KEY_BYTES &&
		       strncmp(first, USER_NAME "_", sizeof(USER_NAME)) == 0 &&
		       strspn(first + sizeof(USER_NAME), "0123456789abcdef") ==
			       (size_t)HEX_PER_BYTE * SERVER_RANDOM_KEY_BYTES &&
		       strcmp(first, second) != 0,
	       "key format and independent randomness");
	static const char *const invalid[] = {
		"",	    "space user", "bad/user",
		"bad,user", "bad\nuser",  "12345678901234567890123456789012345"
	};
	for (size_t index = 0; index < sizeof(invalid) / sizeof(invalid[0]);
	     index++) {
		memcpy(second, "unchanged", sizeof("unchanged"));
		expect(server_key_generate(invalid[index], second) ==
				       STATUS_INVALID_ARGUMENT &&
			       strcmp(second, "unchanged") == 0,
		       "invalid username leaves keygen output unchanged");
	}
	char file[MOST_FILE_BYTES] = { 0 };
	int length = snprintf(file, sizeof(file),
			      "# generated fixture\n\n%s\r\n", first);
	expect(length > 0 &&
		       server_keys_read(NULL, keys) == STATUS_INVALID_ARGUMENT,
	       "NULL path refused");
	file_write(path, file, (size_t)length);
	expect(server_keys_read(path, keys) == STATUS_OK && keys->count == 1 &&
		       strcmp(keys->entries[0].user, USER_NAME) == 0,
	       "comments, blank lines, CRLF, last underscore split");
	length = snprintf(file, sizeof(file), "%s\n%s\n", first, first);
	refused(path, file, (size_t)length, keys);
	expect(server_key_generate(USER_NAME, second) == STATUS_OK,
	       "same user new secret");
	length = snprintf(file, sizeof(file), "%s\n%s\n", first, second);
	refused(path, file, (size_t)length, keys);
	length = snprintf(file, sizeof(file), "# line 1\n%s\nmalformed\n",
			  first);
	refused(path, file, (size_t)length, keys);
	length = snprintf(file, sizeof(file), "# strict named format\n%s\n",
			  first);
	file[(size_t)length - HEX_PER_BYTE] = 'A';
	refused(path, file, (size_t)length, keys);
	file[(size_t)length - HEX_PER_BYTE] = '\0';
	refused(path, file, (size_t)length, keys);
	refused(path, "", 0, keys);
	refused(path, "\n# empty\n", sizeof("\n# empty\n") - 1u, keys);
	file_write(path, LEGACY_KEY "\r\n", sizeof(LEGACY_KEY "\r\n") - 1u);
	expect(server_keys_read(path, keys) == STATUS_OK && keys->count == 1 &&
		       strcmp(keys->entries[0].user, "default") == 0 &&
		       server_keys_match(keys, LEGACY_KEY,
					 strlen(LEGACY_KEY)) == 0,
	       "legacy exactly one physical line is default");
	refused(path, LEGACY_KEY "\n\n", sizeof(LEGACY_KEY "\n\n") - 1u, keys);
	expect(chmod(path, PUBLIC_MODE) == 0 &&
		       server_keys_read(path, keys) ==
			       STATUS_INVALID_ARGUMENT &&
		       chmod(path, PRIVATE_MODE) == 0,
	       "public file refused");
	expect(server_keys_read("/tmp", keys) == STATUS_INVALID_ARGUMENT,
	       "directory refused");
	size_t used = 0;
	char name[SERVER_USER_BYTES];
	for (uint32_t index = 0; index <= SERVER_MOST_USERS; index++) {
		(void)snprintf(name, sizeof(name), "user_%u", index);
		expect(server_key_generate(name, first) == STATUS_OK,
		       "generate user fixture");
		const int count = snprintf(file + used, sizeof(file) - used,
					   "%s\n", first);
		expect(count > 0 && (size_t)count < sizeof(file) - used,
		       "bounded fixture");
		used += (size_t)count;
		if (index + 1u == SERVER_MOST_USERS) {
			file_write(path, file, used);
			expect(server_keys_read(path, keys) == STATUS_OK &&
				       keys->count == SERVER_MOST_USERS,
			       "maximum users admitted");
		}
	}
	refused(path, file, used, keys);
	for (uint32_t index = 0; index < keys->count; index++) {
#if CHECKING_HOOKS_ENABLED
		expect(fault_arm("server_key_compare", SERVER_MOST_USERS + 1u),
		       "arm visit observer");
#endif
		expect(server_keys_match(keys, keys->entries[index].key,
					 keys->entries[index].length) == index,
		       "each configured key authenticates");
#if CHECKING_HOOKS_ENABLED
		expect(fault_disarm() == SERVER_MOST_USERS,
		       "every key visited for every match position");
#endif
	}
#if CHECKING_HOOKS_ENABLED
	expect(fault_arm("server_key_compare", SERVER_MOST_USERS + 1u),
	       "observe failed match");
#endif
	expect(server_keys_match(keys, "bad", strlen("bad")) == keys->count,
	       "unknown credential refused");
#if CHECKING_HOOKS_ENABLED
	expect(fault_disarm() == SERVER_MOST_USERS,
	       "every key visited for failed match");
	expect(fault_arm("server_key_random", 1), "arm entropy failure");
	expect(server_key_generate(USER_NAME, second) ==
		       STATUS_INPUT_OUTPUT_ERROR,
	       "entropy failure refuses generation");
	(void)fault_disarm();
#endif
	explicit_bzero(keys, sizeof(*keys));
	free(keys);
	explicit_bzero(file, sizeof(file));
	expect(unlink(path) == 0, "fixture removed");
	(void)printf("server_keys_test: %d failures\n", failures);
	return failures == 0 ? 0 : 1;
}
