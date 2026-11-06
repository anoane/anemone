/* SPDX-License-Identifier: AGPL-3.0-or-later */
#include "anemone/server_keys.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/random.h>
#include <sys/stat.h>
#include <unistd.h>

#include "anemone/hooks.h"

#define KEY_FILE_BYTES (1u << 20u)
#define KEY_CHARACTERS \
	"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~+/="
#define HEX_DIGITS "0123456789abcdef"
#define HEX_NIBBLE_BITS 4u
#define HEX_NIBBLE_MASK 15u
#define HEX_PER_BYTE 2u
#define KEYGEN_ARGUMENTS 3
#define KEYGEN_USER_ARGUMENT 2
#define USER_PREFIX "--user="

bool server_user_valid(const char *name, size_t length)
{
	if (name == NULL || length == 0 || length >= SERVER_USER_BYTES) {
		return false;
	}
	for (size_t index = 0; index < length; index++) {
		const char byte = name[index];
		if (!((byte >= 'A' && byte <= 'Z') ||
		      (byte >= 'a' && byte <= 'z') ||
		      (byte >= '0' && byte <= '9') || byte == '.' ||
		      byte == '_' || byte == '-')) {
			return false;
		}
	}
	return true;
}

/* A line is length-delimited, including embedded NULs. Only CR immediately before LF/EOF is trimmed. */
static bool line_added(struct server_keys_record *keys, const char *line,
		       size_t length, bool legacy)
{
	if (length == 0 || line[0] == '#') {
		return true;
	}
	if (keys->count == SERVER_MOST_USERS ||
	    length > SERVER_KEY_MOST_BYTES) {
		return false;
	}
	const char *separator = memrchr(line, '_', length);
	const size_t name_length =
		separator != NULL ? (size_t)(separator - line) : 0;
	bool named = separator != NULL &&
		     server_user_valid(line, name_length) &&
		     length - name_length - 1u ==
			     (size_t)HEX_PER_BYTE * SERVER_RANDOM_KEY_BYTES;
	for (size_t index = name_length + 1u; named && index < length;
	     index++) {
		named = line[index] != '\0' &&
			strchr(HEX_DIGITS, line[index]) != NULL;
	}
	if (!named) {
		if (!legacy || length < SERVER_KEY_LEAST_BYTES) {
			return false;
		}
		for (size_t index = 0; index < length; index++) {
			if (line[index] == '\0' ||
			    strchr(KEY_CHARACTERS, line[index]) == NULL) {
				return false;
			}
		}
	}
	struct server_key *const entry = &keys->entries[keys->count];
	if (named) {
		memcpy(entry->user, line, name_length);
	} else {
		memcpy(entry->user, "default", sizeof("default"));
	}
	for (uint32_t index = 0; index < keys->count; index++) {
		if (strcmp(keys->entries[index].user, entry->user) == 0 ||
		    (keys->entries[index].length == length &&
		     memcmp(keys->entries[index].key, line, length) == 0)) {
			return false;
		}
	}
	memcpy(entry->key, line, length);
	entry->length = length;
	keys->count++;
	return true;
}

result_status server_keys_read(const char *path,
			       struct server_keys_record *output)
{
	if (path == NULL || output == NULL) {
		return STATUS_INVALID_ARGUMENT;
	}
	struct server_keys_record *const candidate =
		calloc(1, sizeof(*candidate));
	char *const bytes = malloc(KEY_FILE_BYTES + 1u);
	if (candidate == NULL || bytes == NULL) {
		free(candidate);
		free(bytes);
		return STATUS_OUT_OF_MEMORY;
	}
	struct stat file = { 0 };
	const int descriptor =
		open(path, O_RDONLY | O_CLOEXEC | O_NOCTTY | O_NONBLOCK);
	bool valid = descriptor >= 0 && fstat(descriptor, &file) == 0 &&
		     S_ISREG(file.st_mode) &&
		     (file.st_mode & (S_IRWXG | S_IRWXO)) == 0;
	size_t count = 0;
	ssize_t received = 1;
	while (valid && received != 0 && count <= KEY_FILE_BYTES) {
		received = read(descriptor, bytes + count,
				KEY_FILE_BYTES + 1u - count);
		valid = received >= 0 || errno == EINTR;
		count += received > 0 ? (size_t)received : 0;
	}
	if (descriptor >= 0 && close(descriptor) != 0) {
		valid = false;
	}
	valid = valid && count != 0 && count <= KEY_FILE_BYTES;
	const char *const newline = memchr(bytes, '\n', count);
	const bool legacy = newline == NULL || newline == bytes + count - 1u;
	size_t begin = 0, line_number = 1;
	while (valid && begin < count) {
		const char *const end =
			memchr(bytes + begin, '\n', count - begin);
		const size_t next = end != NULL ? (size_t)(end - bytes) : count;
		const size_t length =
			next - begin -
			(next > begin && bytes[next - 1u] == '\r' ? 1u : 0u);
		valid = line_added(candidate, bytes + begin, length, legacy);
		begin = next + 1u;
		line_number += valid ? 1u : 0u;
	}
	valid = valid && candidate->count != 0;
	if (valid) {
		*output = *candidate;
	} else {
		/* Required product error, including release: never path, bytes or credential-derived text. */
		(void)fprintf(
			stderr,
			"server: key file refused at line %zu (format, duplicate, limit or file access)\n",
			line_number);
	}
	explicit_bzero(bytes, KEY_FILE_BYTES + 1u);
	explicit_bzero(candidate, sizeof(*candidate));
	free(bytes);
	free(candidate);
	return valid ? STATUS_OK : STATUS_INVALID_ARGUMENT;
}

uint32_t server_keys_match(const struct server_keys_record *keys,
			   const char *given, size_t length)
{
	if (keys == NULL || keys->count > SERVER_MOST_USERS || given == NULL) {
		return SERVER_MOST_USERS;
	}
	uint32_t selected = keys->count;
	for (uint32_t entry = 0; entry < keys->count; entry++) {
		/* Volatile accumulator prevents replacement with a data-dependent library comparison. */
		volatile unsigned difference =
			keys->entries[entry].length != length ? 1u : 0u;
		const bool read_failed = FAULT_POINT(server_key_compare);
		for (size_t index = 0; index < SERVER_KEY_MOST_BYTES; index++) {
			const unsigned char byte =
				index < length ? (unsigned char)given[index] :
						 0;
			difference |=
				(unsigned char)keys->entries[entry].key[index] ^
				byte;
		}
		const uint32_t mask =
			0u - (uint32_t)((difference == 0) & !read_failed);
		selected = (selected & ~mask) | (entry & mask);
	}
	return selected;
}

result_status server_key_generate(const char *user,
				  char output[SERVER_NAMED_KEY_BYTES])
{
	const size_t length = user != NULL ? strnlen(user, SERVER_USER_BYTES) :
					     0;
	if (output == NULL || !server_user_valid(user, length)) {
		return STATUS_INVALID_ARGUMENT;
	}
	unsigned char random[SERVER_RANDOM_KEY_BYTES] = { 0 };
	size_t count = 0;
	while (count < sizeof(random)) {
		errno = 0; /* fault injection has no errno; an actual interrupted getrandom is retried */
		const ssize_t received =
			FAULT_POINT(server_key_random) ?
				-1 :
				getrandom(random + count,
					  sizeof(random) - count, 0);
		if (received <= 0) {
			if (received < 0 && errno == EINTR) {
				continue;
			}
			explicit_bzero(random, sizeof(random));
			return STATUS_INPUT_OUTPUT_ERROR;
		}
		count += (size_t)received;
	}
	memcpy(output, user, length);
	output[length] = '_';
	for (size_t index = 0; index < sizeof(random); index++) {
		output[length + 1u + HEX_PER_BYTE * index] =
			HEX_DIGITS[random[index] >> HEX_NIBBLE_BITS];
		output[length + 1u + HEX_PER_BYTE * index + 1u] =
			HEX_DIGITS[random[index] & HEX_NIBBLE_MASK];
	}
	output[length + 1u + HEX_PER_BYTE * sizeof(random)] = '\0';
	explicit_bzero(random, sizeof(random));
	return STATUS_OK;
}

int server_keygen_cli(int argument_count, char **arguments)
{
	char key[SERVER_NAMED_KEY_BYTES] = { 0 };
	const bool valid = argument_count == KEYGEN_ARGUMENTS &&
			   arguments != NULL &&
			   arguments[KEYGEN_USER_ARGUMENT] != NULL &&
			   strncmp(arguments[KEYGEN_USER_ARGUMENT], USER_PREFIX,
				   sizeof(USER_PREFIX) - 1u) == 0 &&
			   server_key_generate(arguments[KEYGEN_USER_ARGUMENT] +
						       sizeof(USER_PREFIX) - 1u,
					       key) == STATUS_OK;
	const bool written = valid && printf("%s\n", key) > 0 &&
			     fflush(stdout) == 0;
	explicit_bzero(key, sizeof(key));
	if (!written) {
		(void)fprintf(
			stderr,
			"keygen: expected --user=<1..32 ASCII letters, digits, dot, underscore or hyphen>\n");
	}
	return written ? 0 : 1;
}
