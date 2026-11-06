/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Detached-server live statistics client. Numeric IPv4/IPv6 hosts only; each HTTP exchange has a fixed
 * deadline and bounded response. Credentials use server.h's key-file reader and are erased before return.
 * fetch replaces output only on success, with a validated JSON object (no trailing newline).
 * Invalid options/pointers return INVALID_ARGUMENT; HTTP refusal or malformed responses INPUT_OUTPUT_ERROR.
 * parse writes options only on success. The CLI returns 0 on success, 1 on I/O failure, 2 on invalid usage.
 */
#ifndef STATS_CLIENT_H
#define STATS_CLIENT_H

#include "anemone/live_stats.h"

struct stats_client_options {
	const char *host;
	const char *key_file;
	uint32_t port;
	uint32_t interval_ms;
	bool is_json;
	bool is_table;
	bool once;
};

bool stats_client_parse(int argument_count, char *const *arguments,
			struct stats_client_options *options_out);
MUST_CHECK result_status stats_client_fetch(
	const struct stats_client_options *options, struct chat_buffer *output);
int stats_client_cli(int argument_count, char *const *arguments);

#endif
