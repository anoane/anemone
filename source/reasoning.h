/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Worker-owned bounded UTF-8 state; no allocation or table beyond the existing tokenizer. */
#ifndef REASONING_H
#define REASONING_H
#include "anemone/front_end_seam.h"
#define REASONING_CONTINUATION_BYTES 3u
struct reasoning_state {
	uint32_t generated;
	uint8_t remaining, minimum, maximum;
	uint8_t queued[REASONING_CONTINUATION_BYTES];
	uint8_t queued_count, queued_at;
	bool closed, close_next;
};
MUST_CHECK result_status reasoning_token(
	const struct generation_command_record *command, uint32_t end_token,
	uint32_t drawn, struct reasoning_state *state, uint32_t *token_out);
#endif
