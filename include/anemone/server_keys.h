/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Private regular-file credentials: at most 64 named users, one username_64-lowercase-hex key per line.
 * Usernames are 1..32 ASCII letters/digits/._-; split at the last underscore. Empty and # lines are ignored.
 * Exactly one physical line may instead hold the legacy 16..4096-byte bearer grammar, as user "default".
 * Duplicate names/keys, empty sets and malformed lines fail closed. Errors report only a line number/reason.
 * Read leaves output unchanged on failure. Match visits every entry and all 4096 bytes per entry, returning
 * count on no match. No credential belongs in a statistics record. Keygen uses Linux getrandom exclusively.
 */
#ifndef SERVER_KEYS_H
#define SERVER_KEYS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "anemone/status.h"

#define SERVER_USER_BYTES 33u
#define SERVER_MOST_USERS 64u
#define SERVER_KEY_LEAST_BYTES 16u
#define SERVER_KEY_MOST_BYTES 4096u
#define SERVER_RANDOM_KEY_BYTES 32u
#define SERVER_NAMED_KEY_BYTES \
	(SERVER_USER_BYTES + 1u + 2u * SERVER_RANDOM_KEY_BYTES)

struct server_key {
	char user[SERVER_USER_BYTES];
	char key[SERVER_KEY_MOST_BYTES]; /* zero padded; compared over the entire buffer */
	size_t length;
};

struct server_keys_record {
	uint32_t count;
	struct server_key entries[SERVER_MOST_USERS];
};

bool server_user_valid(const char *name, size_t length);
MUST_CHECK result_status server_keys_read(const char *path,
					  struct server_keys_record *output);
uint32_t server_keys_match(const struct server_keys_record *keys,
			   const char *given, size_t length);
MUST_CHECK result_status
server_key_generate(const char *user, char output[SERVER_NAMED_KEY_BYTES]);
/* CLI emits exactly one key and newline on stdout on success, only a fixed error on stderr on failure. */
int server_keygen_cli(int argument_count, char **arguments);

#endif
