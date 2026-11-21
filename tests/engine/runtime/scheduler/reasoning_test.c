/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Hard-cap policy over real tokenizer byte fallbacks, including joined partial tokens and malformed output. */
#include "engine/runtime/scheduler/reasoning.h"
#include "support/serving_fixture.h"
#include "engine/input/tokenizer/tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#define BUDGET 8u
#define EMOJI_FIRST 0xF0u
#define EMOJI_SECOND 0x9Fu
#define EMOJI_THIRD 0x98u
#define EMOJI_LAST 0x80u
static int failures;
static void expect(bool holds, const char *message)
{
	if (!holds) {
		(void)printf("FAILED: %s\n", message);
		failures++;
	}
}
int main(void)
{
	uint64_t table_bytes = 0;
	void *const tables = test_serving_tables(&table_bytes);
	struct text_tokenizer *tokenizer = NULL;
	if (tables == NULL ||
	    tokenizer_open(tables, table_bytes, &tokenizer) != STATUS_OK) {
		free(tables);
		return EXIT_FAILURE;
	}
	const struct generation_command_record command = {
		.reasoning_budget = BUDGET,
		.reasoning_end_token = TEST_SERVING_REASONING_END_TOKEN,
		.reasoning_tokenizer = tokenizer
	};
	struct reasoning_state state = { 0 };
	static const uint32_t draws[BUDGET] = {
		'r',	     'r',	   'r',
		EMOJI_FIRST, EMOJI_SECOND, TEST_SERVING_JOINED_TOKEN,
		'x',	     'x'
	};
	static const uint32_t expected[BUDGET] = {
		'r',	      'r',
		'r',	      EMOJI_FIRST,
		EMOJI_SECOND, EMOJI_THIRD,
		EMOJI_LAST,   TEST_SERVING_REASONING_END_TOKEN
	};
	for (uint32_t index = 0; index < BUDGET; index++) {
		uint32_t token = 0;
		const result_status status =
			reasoning_token(&command, TEST_SERVING_END_TOKEN,
					draws[index], &state, &token);
		expect(status == STATUS_OK && token == expected[index],
		       "joined token finishes only the open character before closing");
		expect(state.generated <= BUDGET, "hard cap is never exceeded");
	}
	expect(state.closed && state.remaining == 0 &&
		       state.generated == BUDGET,
	       "engine closes at a complete UTF-8 boundary");
	state = (struct reasoning_state){ 0 };
	uint32_t token = 0;
	expect(reasoning_token(&command, TEST_SERVING_END_TOKEN, EMOJI_FIRST,
			       &state, &token) == STATUS_OK,
	       "partial character admitted with sufficient room");
	expect(reasoning_token(&command, TEST_SERVING_END_TOKEN,
			       TEST_SERVING_END_TOKEN, &state,
			       &token) == STATUS_UNSUPPORTED,
	       "malformed unfinished model character fails without forcing a split");
	const struct generation_command_record ordinary = { 0 };
	state = (struct reasoning_state){ 0 };
	expect(reasoning_token(&ordinary, TEST_SERVING_END_TOKEN,
			       TEST_SERVING_JOINED_TOKEN, &state,
			       &token) == STATUS_OK &&
		       token == TEST_SERVING_JOINED_TOKEN,
	       "unset policy leaves the original token unchanged");
	tokenizer_close(tokenizer);
	free(tables);
	(void)printf("reasoning boundary policy: %d failures\n", failures);
	return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
