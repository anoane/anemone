/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Independent byte/ownership checks for full-state copies. Host arenas only; no model or CUDA context. */
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "anemone/model_layout.h"
#include "host_memory_backend.h"
#include "step_driver.h"

#define CARDS 2u
#define SESSIONS 2u
#define SOURCE_SLOT 0u
#define DESTINATION_SLOT 1u
#define CLONE_TOKENS (MODEL_POOLED_RATIO * 2u)
#define CONTEXT_TOKENS (CLONE_TOKENS * 4u)
#define ARENA_BYTES 1024u
#define WAIT_LIMIT_NS 1000000ull
#define RING_OFFSET 64u
#define RING_BYTES 32u
#define RING_STRIDE 64u
#define POOL_OFFSET 256u
#define POOL_STRIDE 64u
#define POOL_UNIT_BYTES 3u
#define POOL_UNIT_TOKENS 3u
#define POOL_UNITS ((CONTEXT_TOKENS + POOL_UNIT_TOKENS - 1u) / POOL_UNIT_TOKENS)
#define COMPRESSOR_OFFSET 384u
#define COMPRESSOR_STRIDE 32u
#define COMPRESSOR_UNIT_BYTES 2u
#define COMPRESSOR_UNITS 5u
#define COMPRESSOR_BYTES (COMPRESSOR_UNIT_BYTES * COMPRESSOR_UNITS)
#define CHECKPOINT_OFFSET 512u
#define CHECKPOINT_ROWS (CLONE_TOKENS + MODEL_POOLED_RATIO)
#define CHECKPOINT_ROW_BYTES 1u
#define CHECKPOINT_STRIDE (CHECKPOINT_ROWS * CHECKPOINT_ROW_BYTES)
#define SATURATED_CLONE_TOKENS (CHECKPOINT_ROWS + MODEL_POOLED_RATIO)
#define SAMPLER_OFFSET 640u
#define DEVICE_ROUND_OFFSET 704u
#define SAMPLER_SOURCE_PATTERN 23u
#define SAMPLER_DESTINATION_PATTERN 83u
#define DEVICE_ROUND_SOURCE_PATTERN 143u
#define DEVICE_ROUND_DESTINATION_PATTERN 203u
#define RANGE_IDENTITIES 3u
#define RANGE_COUNT (CARDS * RANGE_IDENTITIES)
#define COMPRESSOR_RANGE_BANK 2u
#define RING_IDENTITY 1u
#define POOL_IDENTITY 2u
#define COMPRESSOR_IDENTITY 3u
#define RING_PATTERN 11u
#define POOL_PATTERN 71u
#define COMPRESSOR_PATTERN 131u
#define CHECKPOINT_PATTERN 191u
#define GUARD_PATTERN 0xa5u
#define PATTERN_PRIME 251u
#define SOURCE_TOKEN_BASE 1000u
#define DESTINATION_TOKEN_BASE 2000u
#define SOURCE_SEQUENCE 17u
#define DESTINATION_SEQUENCE 23u
#define SOURCE_ROUND_PATTERN 0x31u
#define DESTINATION_ROUND_PATTERN 0x92u
#define OUTPUT_SENTINEL 0xfeedfaceull

struct fixture {
	struct step_driver driver;
	struct execution_program program;
	const struct execution_program *programs[1];
	struct program_card_record cards[CARDS];
	struct session_range ranges[RANGE_COUNT];
	struct window_checkpoint checkpoints[CARDS];
	struct backend_operations backend;
	uint32_t histories[SESSIONS * CONTEXT_TOKENS];
	uint32_t token_counts[SESSIONS], next_positions[SESSIONS];
	uint32_t mirror_from[SESSIONS], mirror_to[SESSIONS];
	uint64_t session_sequences[SESSIONS];
	unsigned char before[CARDS][ARENA_BYTES];
	uint32_t launches[CARDS], waits[CARDS];
	uint32_t refused_launch_card, failed_wait_card;
	void *staging;
};

static int failures;
static struct fixture *active;

static void expect(bool condition, const char *message)
{
	if (!condition) {
		(void)printf("FAILED: step clone: %s\n", message);
		failures++;
	}
}

static uint32_t card_index(const struct device_card *card)
{
	for (uint32_t index = 0; index < CARDS; index++) {
		if (active->driver.cards[index] == card) {
			return index;
		}
	}
	return CARDS;
}

MUST_CHECK static result_status
tracked_launch(struct device_card *card, const struct kernel_launch *launches,
	       uint32_t count)
{
	const uint32_t index = card_index(card);
	if (index >= CARDS) {
		return STATUS_INVALID_ARGUMENT;
	}
	active->launches[index]++;
	return index == active->refused_launch_card ?
		       STATUS_INJECTED_FAULT :
		       test_host_memory_backend.launch_run(card, launches,
							   count);
}

MUST_CHECK static result_status tracked_wait(struct device_card *card,
					     uint64_t limit)
{
	const uint32_t index = card_index(card);
	if (index >= CARDS) {
		return STATUS_INVALID_ARGUMENT;
	}
	active->waits[index]++;
	const result_status status =
		test_host_memory_backend.launch_wait(card, limit);
	return status == STATUS_OK && index == active->failed_wait_card ?
		       STATUS_DEVICE_ERROR :
		       status;
}

static void pattern(unsigned char *bytes, uint32_t count, uint32_t seed)
{
	for (uint32_t index = 0; index < count; index++) {
		bytes[index] = (unsigned char)((index + seed) % PATTERN_PRIME);
	}
}

static bool opened(struct fixture *fixture)
{
	memset(fixture, 0, sizeof(*fixture));
	active = fixture;
	fixture->refused_launch_card = CARDS;
	fixture->failed_wait_card = CARDS;
	fixture->backend = test_host_memory_backend;
	fixture->backend.launch_run = tracked_launch;
	fixture->backend.launch_wait = tracked_wait;
	fixture->program = (struct execution_program){
		.cards = fixture->cards,
		.limits = { .session_capacity = SESSIONS,
			    .context_token_limit = CONTEXT_TOKENS },
		.card_count = CARDS,
		.session_ranges = fixture->ranges,
		.session_range_count = RANGE_COUNT,
		.checkpoints = fixture->checkpoints,
		.checkpoint_count = CARDS
	};
	fixture->programs[0] = &fixture->program;
	fixture->driver.backend = &fixture->backend;
	fixture->driver.programs = fixture->programs;
	fixture->driver.program_count = 1;
	fixture->driver.histories = fixture->histories;
	fixture->driver.token_counts = fixture->token_counts;
	fixture->driver.next_positions = fixture->next_positions;
	fixture->driver.mirror_from = fixture->mirror_from;
	fixture->driver.mirror_to = fixture->mirror_to;
	fixture->driver.session_sequences = fixture->session_sequences;
	fixture->driver.wait_limit_nanoseconds = WAIT_LIMIT_NS;
	for (uint32_t slot = 0; slot < SESSIONS; slot++) {
		fixture->token_counts[slot] = CONTEXT_TOKENS;
		fixture->next_positions[slot] = CONTEXT_TOKENS;
		fixture->session_sequences[slot] = slot == SOURCE_SLOT ?
							   SOURCE_SEQUENCE :
							   DESTINATION_SEQUENCE;
		for (uint32_t position = 0; position < CONTEXT_TOKENS;
		     position++) {
			fixture->histories[slot * CONTEXT_TOKENS + position] =
				position + (slot == SOURCE_SLOT ?
						    SOURCE_TOKEN_BASE :
						    DESTINATION_TOKEN_BASE);
		}
	}
	memset(&fixture->driver.last_rounds.rounds[SOURCE_SLOT],
	       SOURCE_ROUND_PATTERN,
	       sizeof(fixture->driver.last_rounds.rounds[SOURCE_SLOT]));
	fixture->driver.last_rounds.rounds[SOURCE_SLOT].session_slot =
		SOURCE_SLOT;
	memset(&fixture->driver.last_rounds.rounds[DESTINATION_SLOT],
	       DESTINATION_ROUND_PATTERN,
	       sizeof(fixture->driver.last_rounds.rounds[DESTINATION_SLOT]));
	fixture->driver.last_rounds.rounds[DESTINATION_SLOT].session_slot =
		DESTINATION_SLOT;
	fixture->driver.last_rounds.valid_mask = (1u << SESSIONS) - 1u;
	fixture->driver.last_round =
		fixture->driver.last_rounds.rounds[DESTINATION_SLOT];
	test_host_memory_backend_reset(CARDS, ARENA_BYTES);
	if (test_host_memory_backend.staging_allocate(
		    ARENA_BYTES, &fixture->staging) != STATUS_OK) {
		expect(false, "host staging opens");
		return false;
	}
	for (uint32_t card = 0; card < CARDS; card++) {
		test_host_memory_backend_set_arithmetic_class(
			card, ARITHMETIC_CLASS_SM_80);
		if (test_host_memory_backend.card_open(
			    card, ARENA_BYTES, &fixture->driver.cards[card]) !=
		    STATUS_OK) {
			expect(false, "host arena opens");
			return false;
		}
		atomic_init(&fixture->driver.live_wait_sequence[card], 0);
		atomic_init(&fixture->driver.live_wait_completed_ns[card], 0);
		atomic_init(&fixture->driver.live_wait_started_ns[card], 0);
		fixture->cards[card] = (struct program_card_record){
			.arena_byte_count = ARENA_BYTES,
			.session_offset = SAMPLER_OFFSET,
			.round_offset = DEVICE_ROUND_OFFSET
		};
		fixture->ranges[card] =
			(struct session_range){ .offset = RING_OFFSET,
						.session_stride = RING_STRIDE,
						.unit_count = RING_BYTES,
						.unit_bytes = 1,
						.card = card,
						.identity = RING_IDENTITY };
		fixture->ranges[CARDS + card] =
			(struct session_range){ .offset = POOL_OFFSET,
						.session_stride = POOL_STRIDE,
						.unit_count = POOL_UNITS,
						.unit_bytes = POOL_UNIT_BYTES,
						.unit_tokens = POOL_UNIT_TOKENS,
						.card = card,
						.identity = POOL_IDENTITY };
		fixture->ranges[CARDS * COMPRESSOR_RANGE_BANK + card] =
			(struct session_range){
				.offset = COMPRESSOR_OFFSET,
				.session_stride = COMPRESSOR_STRIDE,
				.unit_count = COMPRESSOR_UNITS,
				.unit_bytes = COMPRESSOR_UNIT_BYTES,
				.card = card,
				.identity = COMPRESSOR_IDENTITY
			};
		fixture->checkpoints[card] = (struct window_checkpoint){
			.ring_offset = RING_OFFSET,
			.checkpoint_offset = CHECKPOINT_OFFSET,
			.ring_session_stride = RING_STRIDE,
			.ring_rows = RING_BYTES,
			.row_bytes = CHECKPOINT_ROW_BYTES,
			.window_rows = CHECKPOINT_ROWS,
			.card = card
		};
		unsigned char *const bytes = fixture->staging;
		memset(bytes, (int)(GUARD_PATTERN + card), ARENA_BYTES);
		pattern(bytes + RING_OFFSET, RING_BYTES, RING_PATTERN);
		pattern(bytes + POOL_OFFSET, POOL_UNITS * POOL_UNIT_BYTES,
			POOL_PATTERN);
		pattern(bytes + COMPRESSOR_OFFSET, COMPRESSOR_BYTES,
			COMPRESSOR_PATTERN);
		pattern(bytes + CHECKPOINT_OFFSET, CHECKPOINT_STRIDE,
			CHECKPOINT_PATTERN);
		pattern(bytes + SAMPLER_OFFSET, STEP_SESSION_BYTES,
			SAMPLER_SOURCE_PATTERN);
		pattern(bytes + SAMPLER_OFFSET + STEP_SESSION_BYTES,
			STEP_SESSION_BYTES, SAMPLER_DESTINATION_PATTERN);
		pattern(bytes + DEVICE_ROUND_OFFSET, STEP_ROUND_BYTES,
			DEVICE_ROUND_SOURCE_PATTERN);
		pattern(bytes + DEVICE_ROUND_OFFSET + STEP_ROUND_BYTES,
			STEP_ROUND_BYTES, DEVICE_ROUND_DESTINATION_PATTERN);
		memcpy(fixture->before[card], bytes, ARENA_BYTES);
		uint64_t ticket = 0;
		const result_status uploaded =
			test_host_memory_backend.upload_begin(
				fixture->driver.cards[card], 0, bytes,
				ARENA_BYTES, &ticket);
		if (uploaded != STATUS_OK ||
		    test_host_memory_backend.upload_wait(
			    fixture->driver.cards[card], ticket,
			    WAIT_LIMIT_NS) != STATUS_OK) {
			expect(false, "host arena is seeded");
			return false;
		}
	}
	return true;
}

static void closed(struct fixture *fixture)
{
	for (uint32_t card = 0; card < CARDS; card++) {
		test_host_memory_backend.card_close(fixture->driver.cards[card],
						    WAIT_LIMIT_NS);
	}
	test_host_memory_backend.staging_free(fixture->staging);
	expect(test_host_memory_backend_open_card_count() == 0 &&
		       test_host_memory_backend_staging_block_count() == 0,
	       "fixture resources are released");
	active = NULL;
}

static void source_unchanged(const struct fixture *fixture)
{
	for (uint32_t card = 0; card < CARDS; card++) {
		const unsigned char *const arena =
			test_host_memory_backend_arena_of(
				fixture->driver.cards[card]);
		expect(memcmp(arena + RING_OFFSET,
			      fixture->before[card] + RING_OFFSET,
			      RING_BYTES) == 0 &&
			       memcmp(arena + POOL_OFFSET,
				      fixture->before[card] + POOL_OFFSET,
				      POOL_UNITS * POOL_UNIT_BYTES) == 0 &&
			       memcmp(arena + COMPRESSOR_OFFSET,
				      fixture->before[card] + COMPRESSOR_OFFSET,
				      COMPRESSOR_BYTES) == 0 &&
			       memcmp(arena + CHECKPOINT_OFFSET,
				      fixture->before[card] + CHECKPOINT_OFFSET,
				      CHECKPOINT_STRIDE) == 0,
		       "every donor range and checkpoint remains unchanged");
		expect(memcmp(arena + SAMPLER_OFFSET,
			      fixture->before[card] + SAMPLER_OFFSET,
			      SESSIONS * STEP_SESSION_BYTES) == 0 &&
			       memcmp(arena + DEVICE_ROUND_OFFSET,
				      fixture->before[card] +
					      DEVICE_ROUND_OFFSET,
				      SESSIONS * STEP_ROUND_BYTES) == 0,
		       "source and destination sampler settings and device draft descriptors stay distinct");
	}
	for (uint32_t position = 0; position < CONTEXT_TOKENS; position++) {
		expect(fixture->histories[position] ==
			       SOURCE_TOKEN_BASE + position,
		       "donor token history stays intact");
	}
	expect(fixture->token_counts[SOURCE_SLOT] == CONTEXT_TOKENS &&
		       fixture->next_positions[SOURCE_SLOT] == CONTEXT_TOKENS &&
		       fixture->session_sequences[SOURCE_SLOT] ==
			       SOURCE_SEQUENCE &&
		       fixture->mirror_from[SOURCE_SLOT] == 0 &&
		       fixture->mirror_to[SOURCE_SLOT] == 0,
	       "donor committed and transient counters stay intact");
	struct step_round expected;
	memset(&expected, SOURCE_ROUND_PATTERN, sizeof(expected));
	expected.session_slot = SOURCE_SLOT;
	expect(memcmp(&fixture->driver.last_rounds.rounds[SOURCE_SLOT],
		      &expected, sizeof(expected)) == 0 &&
		       (fixture->driver.last_rounds.valid_mask &
			(1u << SOURCE_SLOT)) != 0,
	       "donor draft descriptor is not consumed or replaced");
}

static void successful_clone(uint32_t tokens, bool checkpoint)
{
	struct fixture fixture;
	if (!opened(&fixture)) {
		closed(&fixture);
		return;
	}
	uint64_t bytes = OUTPUT_SENTINEL;
	expect(step_session_clone(&fixture.driver, SOURCE_SLOT,
				  DESTINATION_SLOT, tokens, checkpoint,
				  &bytes) == STATUS_OK,
	       "clone succeeds");
	const uint32_t pool_bytes =
		((tokens + POOL_UNIT_TOKENS - 1u) / POOL_UNIT_TOKENS) *
		POOL_UNIT_BYTES;
	const uint32_t checkpoint_bytes =
		(tokens < CHECKPOINT_ROWS ? tokens : CHECKPOINT_ROWS) *
		CHECKPOINT_ROW_BYTES;
	const uint64_t expected_bytes =
		(uint64_t)CARDS * (RING_BYTES + pool_bytes + COMPRESSOR_BYTES +
				   (checkpoint ? checkpoint_bytes : 0u));
	expect(bytes == expected_bytes,
	       "byte count includes each physical mirror and fixed compressor state");
	for (uint32_t card = 0; card < CARDS; card++) {
		unsigned char expected[ARENA_BYTES];
		memcpy(expected, fixture.before[card], sizeof(expected));
		memcpy(expected + RING_OFFSET + RING_STRIDE,
		       expected + RING_OFFSET, RING_BYTES);
		memcpy(expected + POOL_OFFSET + POOL_STRIDE,
		       expected + POOL_OFFSET, pool_bytes);
		memcpy(expected + COMPRESSOR_OFFSET + COMPRESSOR_STRIDE,
		       expected + COMPRESSOR_OFFSET, COMPRESSOR_BYTES);
		if (checkpoint) {
			memcpy(expected + CHECKPOINT_OFFSET + CHECKPOINT_STRIDE,
			       expected + CHECKPOINT_OFFSET, checkpoint_bytes);
		}
		expect(memcmp(test_host_memory_backend_arena_of(
				      fixture.driver.cards[card]),
			      expected, sizeof(expected)) == 0,
		       "exact destination ranges change; all guards, suffixes and settings stay intact");
		expect(fixture.waits[card] == 1 &&
			       !fixture.driver.uncovered_work[card],
		       "all copy work completes before clone returns");
	}
	for (uint32_t position = 0; position < CONTEXT_TOKENS; position++) {
		const uint32_t expected =
			position < tokens ? SOURCE_TOKEN_BASE + position :
					    DESTINATION_TOKEN_BASE + position;
		expect(fixture.histories[CONTEXT_TOKENS + position] == expected,
		       "only the committed source prefix is published to destination host history");
	}
	expect(fixture.token_counts[DESTINATION_SLOT] == tokens &&
		       fixture.next_positions[DESTINATION_SLOT] == tokens &&
		       fixture.session_sequences[DESTINATION_SLOT] == 0,
	       "destination counters describe the copied prefix");
	expect((fixture.driver.last_rounds.valid_mask &
		(1u << DESTINATION_SLOT)) == 0 &&
		       fixture.driver.last_round.session_slot ==
			       STEP_NO_SESSION,
	       "destination draft descriptors are discarded rather than inherited");
	source_unchanged(&fixture);
	closed(&fixture);
}

static void refused_clone(struct fixture *fixture, uint32_t source,
			  uint32_t destination, uint32_t tokens,
			  bool checkpoint, bool output,
			  result_status expected_status)
{
	unsigned char before_fixture[sizeof(*fixture)];
	memcpy(before_fixture, fixture, sizeof(before_fixture));
	uint64_t bytes = OUTPUT_SENTINEL;
	expect(step_session_clone(&fixture->driver, source, destination, tokens,
				  checkpoint,
				  output ? &bytes : NULL) == expected_status,
	       "invalid or ineligible clone is refused");
	expect(bytes == OUTPUT_SENTINEL && memcmp(before_fixture, fixture,
						  sizeof(before_fixture)) == 0,
	       "refusal leaves output, history and driver metadata untouched");
	for (uint32_t card = 0; card < CARDS; card++) {
		expect(fixture->launches[card] == 0 &&
			       fixture->waits[card] == 0 &&
			       memcmp(test_host_memory_backend_arena_of(
					      fixture->driver.cards[card]),
				      fixture->before[card], ARENA_BYTES) == 0,
		       "refusal queues no work and changes no arena byte");
	}
}

static void refusals(void)
{
	uint64_t bytes = OUTPUT_SENTINEL;
	expect(step_session_clone(NULL, SOURCE_SLOT, DESTINATION_SLOT,
				  CLONE_TOKENS, false,
				  &bytes) == STATUS_INVALID_ARGUMENT &&
		       bytes == OUTPUT_SENTINEL,
	       "null driver refusal preserves output");
	struct fixture fixture;
	if (!opened(&fixture)) {
		closed(&fixture);
		return;
	}
	refused_clone(&fixture, SOURCE_SLOT, SOURCE_SLOT, CLONE_TOKENS, false,
		      true, STATUS_INVALID_ARGUMENT);
	refused_clone(&fixture, SESSIONS, DESTINATION_SLOT, CLONE_TOKENS, false,
		      true, STATUS_INVALID_ARGUMENT);
	refused_clone(&fixture, SOURCE_SLOT, SESSIONS, CLONE_TOKENS, false,
		      true, STATUS_INVALID_ARGUMENT);
	refused_clone(&fixture, SOURCE_SLOT, DESTINATION_SLOT, 0, false, true,
		      STATUS_INVALID_ARGUMENT);
	refused_clone(&fixture, SOURCE_SLOT, DESTINATION_SLOT,
		      CLONE_TOKENS + 1u, false, true, STATUS_INVALID_ARGUMENT);
	refused_clone(&fixture, SOURCE_SLOT, DESTINATION_SLOT,
		      CONTEXT_TOKENS + MODEL_POOLED_RATIO, false, true,
		      STATUS_INVALID_ARGUMENT);
	refused_clone(&fixture, SOURCE_SLOT, DESTINATION_SLOT, CLONE_TOKENS,
		      false, false, STATUS_INVALID_ARGUMENT);
	fixture.token_counts[SOURCE_SLOT] = 0;
	refused_clone(&fixture, SOURCE_SLOT, DESTINATION_SLOT, CLONE_TOKENS,
		      false, true, STATUS_INVALID_ARGUMENT);
	fixture.token_counts[SOURCE_SLOT] = CONTEXT_TOKENS;
	fixture.program.checkpoint_count = 0;
	refused_clone(&fixture, SOURCE_SLOT, DESTINATION_SLOT, CLONE_TOKENS,
		      true, true, STATUS_INVALID_ARGUMENT);
	fixture.program.checkpoint_count = CARDS;
	fixture.mirror_to[SOURCE_SLOT] = CLONE_TOKENS;
	refused_clone(&fixture, SOURCE_SLOT, DESTINATION_SLOT, CLONE_TOKENS,
		      false, true, STATUS_BUSY_VALUE);
	expect(fixture.mirror_to[SOURCE_SLOT] == CLONE_TOKENS,
	       "busy donor mirrors are not synchronized or cleared");
	closed(&fixture);
}

static void partial_failure(bool device_error, bool launch_error)
{
	struct fixture fixture;
	if (!opened(&fixture)) {
		closed(&fixture);
		return;
	}
	if (device_error) {
		fixture.failed_wait_card = SOURCE_SLOT;
	}
	if (launch_error) {
		fixture.refused_launch_card = DESTINATION_SLOT;
	}
	uint64_t bytes = OUTPUT_SENTINEL;
	const result_status expected = device_error ? STATUS_DEVICE_ERROR :
						      STATUS_INJECTED_FAULT;
	expect(step_session_clone(&fixture.driver, SOURCE_SLOT,
				  DESTINATION_SLOT, CLONE_TOKENS, true,
				  &bytes) == expected,
	       "copy failure is propagated");
	expect(bytes == OUTPUT_SENTINEL &&
		       fixture.token_counts[DESTINATION_SLOT] == 0 &&
		       fixture.next_positions[DESTINATION_SLOT] == 0,
	       "partial bytes are never published as a committed destination prefix");
	expect(fixture.waits[SOURCE_SLOT] == 1 &&
		       fixture.waits[DESTINATION_SLOT] == 1,
	       "every card is waited even after an earlier launch or wait fails");
	source_unchanged(&fixture);
	if (device_error) {
		const uint32_t launches = fixture.launches[SOURCE_SLOT] +
					  fixture.launches[DESTINATION_SLOT];
		expect(fixture.driver.failure == STATUS_DEVICE_ERROR &&
			       step_session_clone(
				       &fixture.driver, SOURCE_SLOT,
				       DESTINATION_SLOT, CLONE_TOKENS, false,
				       &bytes) == STATUS_DEVICE_ERROR &&
			       fixture.launches[SOURCE_SLOT] +
					       fixture.launches[DESTINATION_SLOT] ==
				       launches,
		       "hardware failure poisons later clones without new work");
	} else {
		expect(fixture.driver.failure == STATUS_OK &&
			       !fixture.driver.uncovered_work[SOURCE_SLOT] &&
			       !fixture.driver.uncovered_work[DESTINATION_SLOT],
		       "non-device failure drains earlier work without poisoning the driver");
	}
	closed(&fixture);
}

int main(void)
{
	successful_clone(CLONE_TOKENS, false);
	successful_clone(CLONE_TOKENS, true);
	successful_clone(SATURATED_CLONE_TOKENS, true);
	refusals();
	partial_failure(false, true);
	partial_failure(true, false);
	partial_failure(true, true);
	(void)printf("step clone: %d failures\n", failures);
	return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
