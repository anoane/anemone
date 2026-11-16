/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* json_test: json_write_string of a string with no bytes and no data. A reader's empty text is a buffer never grown,
 * { NULL, 0 }: the Messages reader's text of an assistant turn of only thinking and tool_use (parser_fuzz_test's
 * Messages seed with a tool call) is one. It writes "", measured and written alike, and forms no pointer from NULL. */
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "json.h"

#define OUTPUT_BYTES 8u
#define EMPTY_STRING "\"\""

int main(void)
{
	char output[OUTPUT_BYTES] = { 0 };
	struct json_writer measured = { 0 },
			   writer = { output, sizeof(output), 0 };
	json_write_string(&measured, (struct json_bytes){ NULL, 0 });
	json_write_string(&writer, (struct json_bytes){ NULL, 0 });
	const size_t length = strlen(EMPTY_STRING);
	const bool written = measured.length == length &&
			     writer.length == length &&
			     memcmp(output, EMPTY_STRING, length) == 0;
	(void)printf("json_test: %s\n",
		     written ? "passed" : "FAILED: the empty string");
	return written ? EXIT_SUCCESS : EXIT_FAILURE;
}
