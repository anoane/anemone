/* SPDX-License-Identifier: AGPL-3.0-or-later */
#ifndef TEST_LOG_EXPECTATION_H
#define TEST_LOG_EXPECTATION_H
#include "common/hooks.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define TEST_LOG_EXPECTED_LINES(count) (LOGGING_HOOKS_ENABLED ? (count) : 0u)
/* The longest log line: a message (LOG_MESSAGE_CAPACITY) after its timestamp, which grows with the uptime (up to
 * 20 + 1 + 9 digits), its level and its source position. */
#define TEST_LOG_PREFIX_BYTES_MOST 192u
#define TEST_LOG_LINE_BYTES_MOST \
	(LOG_MESSAGE_CAPACITY + TEST_LOG_PREFIX_BYTES_MOST)

/* The whole capture file, terminated, and its byte count; -1 when it does not fit the buffer. A longer capture
 * fails loudly instead of being cut (a cut capture counts fewer lines than were logged). */
static inline ssize_t test_log_capture_read(int descriptor, char *text,
					    size_t capacity)
{
	struct stat file;
	if (text == NULL || capacity == 0) {
		return -1;
	}
	text[0] = '\0';
	if (fstat(descriptor, &file) != 0 || file.st_size < 0) {
		return -1;
	}
	if ((uint64_t)file.st_size >= capacity) {
		(void)printf(
			"FAILED: a capture of %jd bytes does not fit its %zu-byte buffer\n",
			(intmax_t)file.st_size, capacity);
		return -1;
	}
	const ssize_t byte_count =
		pread(descriptor, text, (size_t)file.st_size, 0);
	if (byte_count != (ssize_t)file.st_size) {
		text[0] = '\0';
		return -1;
	}
	text[byte_count] = '\0';
	return byte_count;
}

/* Diagnostic builds require the original refusal message. Release requires
 * silence; the caller still checks status, untouched outputs and side effects.
 * A nonempty release log must fail rather than satisfy a diagnostic expectation. */
static inline bool test_log_is_expected(const char *text, const char *words, bool single_line)
{
    if (text == NULL) {
        return false;
    }
    if (!LOGGING_HOOKS_ENABLED || words == NULL) {
	    return text[0] == '\0';
    }
    const char *const line_end = strchr(text, '\n');
    return strstr(text, " anemone error ") != NULL && strstr(text, words) != NULL &&
           (!single_line || (line_end != NULL && line_end[1] == '\0'));
}
#endif
