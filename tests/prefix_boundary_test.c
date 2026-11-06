/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Golden cold traces, cone gaps and server provenance against the actual prepared-size planner. No CUDA. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "engine_internal.h"
#include "server_internal.h"
#include "step_driver.h"

#define CONTEXT 65536u
#define MARGIN 127u
#define TAIL 128u
#define KINDS 2u
#define LONG_PROMPT 6004u
#define MEDIUM_PROMPT 1905u
#define SHORT_PROMPT 304u
#define LONG_DECODER 5876u
#define MEDIUM_DECODER 1777u
#define SHORT_DECODER 176u
#define LONG_THIN_FROM 5952u
#define MEDIUM_THIN_FROM 1856u
#define SHORT_THIN_FROM 256u
#define BAD_BOUNDARY 0xDEADBEEFu
#define MISMATCH_TOKEN 1u
#define OFF_CHUNK_CHECKPOINT 5886u
#define UNQUALIFIED_FLIGHTS 2u

static const uint32_t sizes[] = { 64, 128, 256, 512, 1024 };
static struct execution_program
	programs[KINDS * (sizeof(sizes) / sizeof(sizes[0]))];
static const struct execution_program
	*pointers[sizeof(programs) / sizeof(programs[0])];
static struct step_driver driver;
static struct inference_engine engine;
static uint32_t tokens[CONTEXT], history_tokens[CONTEXT];
static int failures;

static void expect(bool condition, const char *what)
{
	if (!condition) {
		(void)fprintf(stderr, "FAILED: prefix boundary: %s\n", what);
		failures++;
	}
}

static uint32_t boundary(uint32_t prompt, uint32_t maximum, uint32_t decoder)
{
	uint32_t answer = BAD_BOUNDARY;
	expect(engine_prefix_boundary(&engine, prompt, maximum, decoder,
				      &answer) == STATUS_OK,
	       "a valid fixed-plan query succeeds");
	return answer;
}

static void fixture(void)
{
	const uint32_t count = sizeof(sizes) / sizeof(sizes[0]);
	for (uint32_t kind = 0; kind < KINDS; kind++)
		for (uint32_t index = 0; index < count; index++) {
			const uint32_t at = kind * count + index;
			programs[at] = (struct execution_program){
				.row_count = sizes[index],
				.kind = kind == 0 ? PROGRAM_WIDE : PROGRAM_FULL
			};
			pointers[at] = &programs[at];
		}
	driver = (struct step_driver){ .programs = pointers,
				       .program_count = KINDS * count };
	engine.driver = &driver;
	engine.description =
		(struct engine_description_record){ .context_token_limit =
							    CONTEXT,
						    .front_steps = 1,
						    .chunks_in_flight = 1,
						    .prefill_margin = MARGIN,
						    .prefill_tail_rows = TAIL };
	engine.cone_rows = MODEL_CONE_ROWS;
	engine.wide_rows = sizes[count - 1u];
	engine.full_rows = engine.wide_rows;
}

/* These traces explicitly keep the cold front seam and omit the interval that would trigger a cone replay.
 * Exhaust every possible common-prefix length, including positions inside chunks and odd pool boundaries. */
static void golden_trace(uint32_t prompt, uint32_t decoder,
			 const uint32_t *chunks, uint32_t count,
			 uint32_t thin_from)
{
	uint32_t expected = 0;
	for (uint32_t maximum = 0; maximum < prompt; maximum++) {
		for (uint32_t index = 0; index < count; index++)
			if (chunks[index] == maximum)
				expected = maximum;
		if (maximum >= thin_from && maximum % SERVING_KEPT_GRANULE == 0)
			expected = maximum;
		const uint32_t actual = boundary(prompt, maximum, decoder);
		if (actual != expected) {
			(void)fprintf(
				stderr,
				"prompt=%u maximum=%u actual=%u expected=%u\n",
				prompt, maximum, actual, expected);
			expect(false,
			       "the retained boundary follows the golden cold trace");
			return;
		}
	}
}

static void server_policy(void)
{
	static const uint8_t no_image[SERVER_IMAGE_IDENTITY_BYTES] = { 0 };
	struct http_server server = {
		.engine = &engine,
		.histories = history_tokens,
		.description = { .context_token_limit = CONTEXT },
		.exact_prefix_reuse = SERVER_PREFIX_COLD_PLAN
	};
	struct slot_history *const history = &server.slot_histories[0];
	*history = (struct slot_history){ .token_count = LONG_PROMPT,
					  .processed_count = LONG_PROMPT - 1u,
					  .reach_count = LONG_PROMPT,
					  .cold_prompt_count = LONG_PROMPT,
					  .decoder_from = LONG_DECODER };
	const uint32_t thin_boundary = LONG_PROMPT - SERVING_KEPT_GRANULE;
	expect(server_prefix_kept(&server, 0, tokens, LONG_PROMPT, no_image) ==
		       thin_boundary,
	       "mode 2 actually retains an even prompt's cold THIN tail");
	history->reach_count = thin_boundary + SERVING_KEPT_LAG + 1u;
	history->checkpoint_count = thin_boundary - SERVING_KEPT_GRANULE + 1u;
	expect(server_prefix_kept(&server, 0, tokens, LONG_PROMPT, no_image) ==
		       0,
	       "an odd checkpoint cannot bypass the observed cold boundary");
	history->checkpoint_count = OFF_CHUNK_CHECKPOINT;
	expect(server_prefix_kept(&server, 0, tokens, LONG_PROMPT, no_image) ==
		       0,
	       "an even checkpoint inside a cold tail chunk also declines reuse");
	history->checkpoint_count = thin_boundary;
	expect(server_prefix_kept(&server, 0, tokens, LONG_PROMPT, no_image) ==
		       thin_boundary,
	       "a matching cold checkpoint remains eligible beyond ring reach");
	history->reach_count = LONG_PROMPT;
	history->checkpoint_count = 0;
	history_tokens[SERVING_KEPT_GRANULE] = MISMATCH_TOKEN;
	expect(server_prefix_kept(&server, 0, tokens, LONG_PROMPT, no_image) ==
		       0,
	       "two shared template tokens do not replace a cold chunk boundary");
	history_tokens[SERVING_KEPT_GRANULE] = 0;
	history->cold_prompt_count--;
	expect(server_prefix_kept(&server, 0, tokens, LONG_PROMPT, no_image) ==
		       0,
	       "a matching token history of another prompt length has no cold-plan provenance");
}

int main(void)
{
	fixture();
	static const uint32_t long_chunks[] = {
		0, 1024, 2048, 3072, 3328, 5888
	};
	static const uint32_t medium_chunks[] = { 0, 1792 };
	static const uint32_t short_chunks[] = { 0 };
	golden_trace(LONG_PROMPT, LONG_DECODER, long_chunks,
		     sizeof(long_chunks) / sizeof(long_chunks[0]),
		     LONG_THIN_FROM);
	golden_trace(MEDIUM_PROMPT, MEDIUM_DECODER, medium_chunks,
		     sizeof(medium_chunks) / sizeof(medium_chunks[0]),
		     MEDIUM_THIN_FROM);
	golden_trace(SHORT_PROMPT, SHORT_DECODER, short_chunks,
		     sizeof(short_chunks) / sizeof(short_chunks[0]),
		     SHORT_THIN_FROM);
	server_policy();
	uint32_t answer = BAD_BOUNDARY;
	expect(engine_prefix_boundary(NULL, LONG_PROMPT, 0, 0, &answer) ==
			       STATUS_INVALID_ARGUMENT &&
		       answer == BAD_BOUNDARY,
	       "a bad engine leaves the output untouched");
	expect(engine_prefix_boundary(&engine, LONG_PROMPT, LONG_PROMPT, 0,
				      &answer) == STATUS_INVALID_ARGUMENT &&
		       answer == BAD_BOUNDARY,
	       "a complete prompt cannot be kept");
	expect(engine_prefix_boundary(&engine, LONG_PROMPT, 0, CONTEXT + 1u,
				      &answer) == STATUS_INVALID_ARGUMENT &&
		       answer == BAD_BOUNDARY,
	       "an invalid decoder boundary leaves output untouched");
	engine.description.session_slot_count = UNQUALIFIED_FLIGHTS;
	expect(boundary(LONG_PROMPT, LONG_PROMPT - 1u, 0) == 0,
	       "unreserved multi-slot prompt rows do not establish cold provenance");
	engine.description.prefill_first = 1;
	expect(boundary(LONG_PROMPT, LONG_PROMPT - 1u, 0) != 0,
	       "prefill_first reserves the prompt's chunks");
	engine.description.prefill_first = 0;
	engine.description.prefill_order = 1;
	expect(boundary(LONG_PROMPT, LONG_PROMPT - 1u, 0) != 0,
	       "an ordered head also reserves the prompt's chunks");
	engine.description.prefill_order = 0;
	engine.description.session_slot_count = 1;
	engine.description.prefill_fast_plan = 1;
	expect(boundary(LONG_PROMPT, LONG_PROMPT - 1u, 0) == 0,
	       "an unqualified fast plan declines reuse");
	engine.description.prefill_fast_plan = 0;
	engine.description.prefill_chunk_rows = MODEL_FULL_CONE_LEAST_ROWS;
	expect(boundary(LONG_PROMPT, LONG_PROMPT - 1u, 0) == 0,
	       "a static cap is outside the qualified cold-plan domain");
	engine.description.prefill_chunk_rows = 0;
	engine.description.prefill_decode_rows = MODEL_FULL_CONE_LEAST_ROWS;
	expect(boundary(LONG_PROMPT, LONG_PROMPT - 1u, 0) == 0,
	       "a decoder-dependent cap cannot establish fixed cold provenance");
	engine.description.prefill_decode_rows = 0;
	expect(boundary(LONG_PROMPT, LONG_PROMPT - 1u, 0) != 0,
	       "explicit zero caps restore fixed-plan boundary admission");
	engine.description.chunks_in_flight = UNQUALIFIED_FLIGHTS;
	expect(boundary(LONG_PROMPT, LONG_PROMPT - 1u, 0) == 0,
	       "an unqualified multi-flight plan declines reuse");
	(void)printf(
		"prefix_boundary_test: %d failures; every kept length for three golden cold traces\n",
		failures);
	return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
