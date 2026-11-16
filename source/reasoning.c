/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* A hard reasoning cap closes on a complete character. A joined token may finish the current character only. */
#include "reasoning.h"
#include "anemone/tokenizer.h"
#include <string.h>
#define UTF8_ASCII_END 0x80u
#define UTF8_CONTINUATION_LAST 0xBFu
#define UTF8_TWO_FIRST 0xC2u
#define UTF8_THREE_FIRST 0xE0u
#define UTF8_FOUR_FIRST 0xF0u
#define UTF8_FOUR_LAST 0xF4u
#define UTF8_THREE_REMAINING 2u
#define UTF8_SURROGATE_FIRST 0xEDu
#define UTF8_THREE_MINIMUM 0xA0u
#define UTF8_SURROGATE_MAXIMUM 0x9Fu
#define UTF8_FOUR_MINIMUM 0x90u
#define UTF8_FOUR_MAXIMUM 0x8Fu

static bool byte_taken(struct reasoning_state *state, uint8_t byte)
{
	if (state->remaining != 0) {
		if (byte < state->minimum || byte > state->maximum)
			return false;
		state->remaining--;
		state->minimum = UTF8_ASCII_END;
		state->maximum = UTF8_CONTINUATION_LAST;
	} else if (byte < UTF8_ASCII_END) {
		return true;
	} else {
		if (byte < UTF8_TWO_FIRST || byte > UTF8_FOUR_LAST)
			return false;
		state->remaining =
			byte < UTF8_THREE_FIRST ? 1u :
			byte < UTF8_FOUR_FIRST	? UTF8_THREE_REMAINING :
						  REASONING_CONTINUATION_BYTES;
		state->minimum = byte == UTF8_THREE_FIRST ? UTF8_THREE_MINIMUM :
				 byte == UTF8_FOUR_FIRST  ? UTF8_FOUR_MINIMUM :
							    UTF8_ASCII_END;
		state->maximum =
			byte == UTF8_SURROGATE_FIRST ? UTF8_SURROGATE_MAXIMUM :
			byte == UTF8_FOUR_LAST	     ? UTF8_FOUR_MAXIMUM :
						       UTF8_CONTINUATION_LAST;
	}
	return true;
}

MUST_CHECK static result_status
queued_taken(const struct generation_command_record *command,
	     struct reasoning_state *state, uint32_t *token_out)
{
	const uint8_t byte = state->queued[state->queued_at];
	state->queued_at++;
	const result_status status = tokenizer_byte_token(
		command->reasoning_tokenizer, byte, token_out);
	return status == STATUS_OK && byte_taken(state, byte) ?
		       STATUS_OK :
		       STATUS_UNSUPPORTED;
}

result_status reasoning_token(const struct generation_command_record *command,
			      uint32_t end_token, uint32_t drawn,
			      struct reasoning_state *state,
			      uint32_t *token_out)
{
	*token_out = drawn;
	if (command->reasoning_budget == 0 || state->closed)
		return STATUS_OK;
	const uint32_t next = state->generated + 1u;
	if (state->queued_at < state->queued_count) {
		const result_status status =
			queued_taken(command, state, token_out);
		if (status != STATUS_OK)
			return status;
	} else if (state->close_next || next >= command->reasoning_budget ||
		   drawn == end_token ||
		   drawn == command->reasoning_end_token) {
		if (state->remaining != 0)
			return STATUS_UNSUPPORTED;
		*token_out = command->reasoning_end_token;
		state->closed = true;
	} else {
		const char *bytes = NULL;
		uint32_t count = 0;
		const result_status status = tokenizer_token_bytes(
			command->reasoning_tokenizer, drawn, &bytes, &count);
		if (status != STATUS_OK || count == 0)
			return STATUS_UNSUPPORTED;
		struct reasoning_state after = *state;
		for (uint32_t index = 0; index < count; index++)
			if (!byte_taken(&after, (uint8_t)bytes[index]))
				return STATUS_UNSUPPORTED;
		if (after.remaining != 0 &&
		    command->reasoning_budget - next <= after.remaining) {
			if (state->remaining == 0) {
				*token_out = command->reasoning_end_token;
				state->closed = true;
			} else {
				/* The previous token reserved one byte-token per continuation and one closing token. */
				if (count < state->remaining ||
				    state->remaining >
					    command->reasoning_budget - next)
					return STATUS_UNSUPPORTED;
				state->queued_count = state->remaining;
				state->queued_at = 0;
				memcpy(state->queued, bytes,
				       state->queued_count);
				state->close_next = true;
				const result_status queued_status =
					queued_taken(command, state, token_out);
				if (queued_status != STATUS_OK)
					return queued_status;
			}
		} else {
			*state = after;
		}
	}
	state->generated = next;
	return STATUS_OK;
}
