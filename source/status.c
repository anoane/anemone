/* SPDX-License-Identifier: AGPL-3.0-or-later */
#include "anemone/status.h"

#include <stddef.h>

static const char *const status_texts[STATUS_COUNT] = {
	[STATUS_OK] = "ok",
	[STATUS_INVALID_ARGUMENT] = "invalid argument",
	[STATUS_OUT_OF_MEMORY] = "out of memory",
	[STATUS_INPUT_OUTPUT_ERROR] = "input or output error",
	[STATUS_CORRUPT_PACK] = "corrupt pack",
	[STATUS_DEVICE_ERROR] = "device error",
	[STATUS_TIMED_OUT] = "timed out",
	[STATUS_UNSUPPORTED] = "unsupported",
	[STATUS_BUSY_VALUE] = "busy",
	[STATUS_INJECTED_FAULT] = "injected fault",
};

const char *status_text(result_status status)
{
    const char *text = NULL;
    if ((int)status >= 0 && status < STATUS_COUNT) {
	    text = status_texts[status];
    }
    return text != NULL ? text : "unknown status";     /* also for a status that was added without a text */
}
