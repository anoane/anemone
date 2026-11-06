/* SPDX-License-Identifier: AGPL-3.0-or-later */
#ifndef TEST_LOG_EXPECTATION_H
#define TEST_LOG_EXPECTATION_H
#include "anemone/hooks.h"
#include <string.h>

#define TEST_LOG_EXPECTED_LINES(count) (LOGGING_HOOKS_ENABLED ? (count) : 0u)

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
