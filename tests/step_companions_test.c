/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Host-only co-enqueue ordering, real handoff boundaries, immutable admission and failure retirement. */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "anemone/backend_records.h"
#include "anemone/step.h"
#include "host_memory_backend.h"
#include "prepare_common.h"

#define TEST_CARDS 3u
#define TEST_PAIR_CARDS 2u
#define TEST_PAIR_MASK ((1u << TEST_PAIR_CARDS) - 1u)
#define TEST_SEGMENTS 3u
#define TEST_HAND_OFFS 2u
#define TEST_RESULT_CARD 2u
#define TEST_ARENA_BYTES 16384u
#define TEST_RESULT_OFFSET 512u
#define TEST_SESSION_OFFSET 1024u
#define TEST_DATA_OFFSET 2048u
#define TEST_RECEIVE_OFFSET 4096u
#define TEST_FINAL_OFFSET 8192u
#define TEST_CONTEXT 64u
#define TEST_WAIT_NANOSECONDS 1000000000ull
#define TEST_ARITHMETIC_IDENTITY 0x5450325F53554D31ull
#define TEST_BASE_VALUE 17ull
#define TEST_GRAPH_SLOTS 2u
#define TEST_STAGING_SLOTS 4u

enum test_failure {
	TEST_SOUND,
	TEST_FIRST_SUBMIT_FAILS,
	TEST_SECOND_SUBMIT_FAILS,
	TEST_CANCELLED_WITH_PENDING_WORK
};

struct test_graph {
	struct kernel_launch launch;
};
struct fixture {
	struct device_card *cards[TEST_CARDS];
	struct kernel_launch launches[TEST_CARDS];
	struct program_card_record areas[TEST_CARDS];
	struct segment_record segments[TEST_SEGMENTS];
	struct hand_off_record hand_offs[TEST_HAND_OFFS];
	struct execution_program program;
	struct backend_operations backend;
	struct step_driver *driver;
};

static struct fixture *current;
static uint32_t submitted, wait_calls[TEST_CARDS], submit_calls[TEST_CARDS];
static uint32_t captures[TEST_CARDS];
static struct test_graph graphs[TEST_CARDS][TEST_GRAPH_SLOTS];
static void *staging[TEST_STAGING_SLOTS];
static bool pair_active, ordering_failure;
static enum test_failure failure_mode;
static int failures;

static void expect(bool condition, const char *message)
{
	if (!condition) {
		(void)fprintf(stderr, "step_companions_test: %s\n", message);
		failures++;
	}
}

static uint32_t card_index(const struct device_card *card)
{
	for (uint32_t index = 0; current != NULL && index < TEST_CARDS;
	     index++) {
		if (current->cards[index] == card)
			return index;
	}
	return TEST_CARDS;
}

MUST_CHECK static result_status staging_allocate(uint64_t bytes,
						 void **memory_out)
{
	if (memory_out == NULL)
		return STATUS_INVALID_ARGUMENT;
	void *memory = NULL;
	const result_status status =
		test_host_memory_backend.staging_allocate(bytes, &memory);
	if (status != STATUS_OK)
		return status;
	for (uint32_t index = 0; index < TEST_STAGING_SLOTS; index++) {
		if (staging[index] == NULL) {
			staging[index] = memory;
			*memory_out = memory;
			return STATUS_OK;
		}
	}
	test_host_memory_backend.staging_free(memory);
	return STATUS_OUT_OF_MEMORY;
}

static void staging_free(void *memory)
{
	for (uint32_t index = 0; index < TEST_STAGING_SLOTS; index++) {
		if (staging[index] == memory)
			staging[index] = NULL;
	}
	test_host_memory_backend.staging_free(memory);
}

static bool stand_in(const struct kernel_launch *launch, uint32_t card,
		     unsigned char *arena, uint64_t arena_bytes, void *context)
{
	(void)launch;
	(void)arena_bytes;
	(void)context;
	if (card < TEST_PAIR_CARDS) {
		const uint64_t value = TEST_BASE_VALUE + card;
		(void)memcpy(arena + TEST_DATA_OFFSET, &value, sizeof(value));
	} else {
		uint64_t left = 0, right = 0;
		(void)memcpy(&left, arena + TEST_RECEIVE_OFFSET, sizeof(left));
		(void)memcpy(&right, arena + TEST_RECEIVE_OFFSET + sizeof(left),
			     sizeof(right));
		const uint64_t sum = left + right;
		(void)memcpy(arena + TEST_FINAL_OFFSET, &sum, sizeof(sum));
	}
	return true;
}

MUST_CHECK static result_status launch_run(struct device_card *card,
					   const struct kernel_launch *launches,
					   uint32_t count)
{
	const uint32_t index = card_index(card);
	if (index >= TEST_CARDS)
		return STATUS_INVALID_ARGUMENT;
	submit_calls[index]++;
	if (index < TEST_PAIR_CARDS) {
		pair_active = true;
		submitted |= 1u << index;
	}
	const result_status status =
		test_host_memory_backend.launch_run(card, launches, count);
	const bool inject = failure_mode == TEST_FIRST_SUBMIT_FAILS ?
				    index == 0 :
				    failure_mode != TEST_SOUND && index == 1u;
	return status == STATUS_OK && inject ? STATUS_INJECTED_FAULT : status;
}

MUST_CHECK static result_status upload_wait(struct device_card *card,
					    uint64_t ticket, uint64_t limit)
{
	const uint32_t index = card_index(card);
	if (index >= TEST_CARDS)
		return STATUS_INVALID_ARGUMENT;
	if (pair_active && index < TEST_PAIR_CARDS) {
		wait_calls[index]++;
		ordering_failure = ordering_failure ||
				   submitted != TEST_PAIR_MASK;
		if (failure_mode == TEST_CANCELLED_WITH_PENDING_WORK &&
		    index == 0)
			return STATUS_TIMED_OUT;
	}
	return test_host_memory_backend.upload_wait(card, ticket, limit);
}

MUST_CHECK static result_status launch_wait(struct device_card *card,
					    uint64_t limit)
{
	const uint32_t index = card_index(card);
	if (index >= TEST_CARDS)
		return STATUS_INVALID_ARGUMENT;
	if (pair_active && index < TEST_PAIR_CARDS) {
		wait_calls[index]++;
		ordering_failure = ordering_failure ||
				   submitted != TEST_PAIR_MASK;
		if (failure_mode == TEST_CANCELLED_WITH_PENDING_WORK &&
		    index == 0)
			return STATUS_TIMED_OUT;
	}
	return test_host_memory_backend.launch_wait(card, limit);
}

MUST_CHECK static result_status
graph_capture(struct device_card *card, const struct kernel_launch *launches,
	      uint32_t count, uint32_t *graph_out)
{
	const uint32_t index = card_index(card);
	if (index >= TEST_CARDS || count != 1u || launches == NULL ||
	    graph_out == NULL)
		return STATUS_INVALID_ARGUMENT;
	if (captures[index] >= TEST_GRAPH_SLOTS)
		return STATUS_BUSY_VALUE;
	graphs[index][captures[index]].launch = launches[0];
	*graph_out = captures[index]++;
	return STATUS_OK;
}

MUST_CHECK static result_status graph_run(struct device_card *card,
					  uint32_t graph)
{
	const uint32_t index = card_index(card);
	return index >= TEST_CARDS || graph >= captures[index] ?
		       STATUS_INVALID_ARGUMENT :
		       launch_run(card, &graphs[index][graph].launch, 1);
}

static void fixture_close(struct fixture *fixture)
{
	step_driver_destroy(fixture->driver);
	fixture->driver = NULL;
	for (uint32_t card = 0; card < TEST_CARDS; card++)
		test_host_memory_backend.card_close(fixture->cards[card],
						    TEST_WAIT_NANOSECONDS);
	/* A failed retirement deliberately retained these caller/backend-owned blocks until all cards close. */
	for (uint32_t index = 0; index < TEST_STAGING_SLOTS; index++) {
		if (staging[index] != NULL)
			staging_free(staging[index]);
	}
	current = NULL;
}

static bool fixture_open(struct fixture *fixture, uint32_t front_segments,
			 bool peer_dependency)
{
	*fixture = (struct fixture){ 0 };
	current = fixture;
	submitted = 0;
	pair_active = false;
	ordering_failure = false;
	failure_mode = TEST_SOUND;
	(void)memset(wait_calls, 0, sizeof(wait_calls));
	(void)memset(submit_calls, 0, sizeof(submit_calls));
	(void)memset(captures, 0, sizeof(captures));
	(void)memset((void *)staging, 0, sizeof(staging));
	test_host_memory_backend_reset(TEST_CARDS, TEST_ARENA_BYTES);
	if (test_host_memory_backend_set_stand_in(KERNEL_FILE_FILL,
						  KERNEL_FILL_WORDS, stand_in,
						  NULL) != STATUS_OK)
		return false;
	fixture->backend = test_host_memory_backend;
	fixture->backend.staging_allocate = staging_allocate;
	fixture->backend.staging_free = staging_free;
	fixture->backend.staging_retire = NULL;
	fixture->backend.launch_run = launch_run;
	fixture->backend.upload_wait = upload_wait;
	fixture->backend.launch_wait = launch_wait;
	fixture->backend.graph_capture = graph_capture;
	fixture->backend.graph_run = graph_run;
	for (uint32_t card = 0; card < TEST_CARDS; card++) {
		uint32_t kernel = 0;
		test_host_memory_backend_set_arithmetic_class(
			card, ARITHMETIC_CLASS_SM_80);
		if (test_host_memory_backend.card_open(card, TEST_ARENA_BYTES,
						       &fixture->cards[card]) !=
			    STATUS_OK ||
		    test_host_memory_backend.kernel_prepare(
			    fixture->cards[card], KERNEL_FILE_FILL,
			    KERNEL_FILL_WORDS, FILL_RECORD_BYTES,
			    &kernel) != STATUS_OK)
			return false;
		const struct fill_record record = { 0 };
		prepare_launch_fill(&fixture->launches[card], kernel, 1, 1, 1,
				    0, &record, sizeof(record));
		fixture->areas[card] = (struct program_card_record){
			.launches = &fixture->launches[card],
			.launch_count = 1,
			.arena_byte_count = TEST_ARENA_BYTES,
			.step_offset = 0,
			.session_offset = card == TEST_RESULT_CARD ?
						  TEST_SESSION_OFFSET :
						  PROGRAM_NOWHERE,
			.result_offset = card == TEST_RESULT_CARD ?
						 TEST_RESULT_OFFSET :
						 PROGRAM_NOWHERE,
			.logits_offset = PROGRAM_NOWHERE,
			.round_offset = PROGRAM_NOWHERE,
			.streams_offset = PROGRAM_NOWHERE,
			.carried_pre_offset = PROGRAM_NOWHERE,
			.engram_rows_offset = PROGRAM_NOWHERE
		};
		fixture->segments[card] = (struct segment_record){
			.card = card,
			.stage = card < TEST_PAIR_CARDS ? 0u : 1u,
			.layer = card < TEST_PAIR_CARDS ? 0u : SEGMENT_NO_LAYER,
			.launch_count = 1
		};
		if (card < TEST_PAIR_CARDS)
			fixture->hand_offs[card] = (struct hand_off_record){
				.source_offset = TEST_DATA_OFFSET,
				.destination_offset = TEST_RECEIVE_OFFSET +
						      card * sizeof(uint64_t),
				.byte_count = sizeof(uint64_t),
				.from_card = card,
				.to_card = peer_dependency && card == 0 ?
						   1u :
						   TEST_RESULT_CARD,
				.after_segment = card
			};
	}
	fixture->program = (struct execution_program){
		.cards = fixture->areas,
		.segments = fixture->segments,
		.hand_offs = fixture->hand_offs,
		.limits = { .context_token_limit = TEST_CONTEXT,
			    .row_capacity = 1,
			    .session_capacity = 1,
			    .state_width = STATE_WIDTH_16 },
		.card_count = TEST_CARDS,
		.segment_count = TEST_SEGMENTS,
		.hand_off_count = TEST_HAND_OFFS,
		.row_count = 1,
		.result_card = TEST_RESULT_CARD,
		.front_segment_count = front_segments,
		.draft_segment_first = TEST_SEGMENTS
	};
	const struct execution_program *const program = &fixture->program;
	const result_status status = step_driver_create(
		&fixture->backend, fixture->cards, &program, 1, NULL,
		TEST_WAIT_NANOSECONDS, &fixture->driver);
	if (status != STATUS_OK)
		(void)fprintf(stderr, "companion fixture driver create: %s\n",
			      status_text(status));
	return status == STATUS_OK;
}

static struct step_companion_program
descriptor(const struct fixture *fixture,
	   const struct step_companion_pair *pairs)
{
	return (struct step_companion_program){
		.program = &fixture->program,
		.pairs = pairs,
		.pair_count = 1,
		.arithmetic_identity = TEST_ARITHMETIC_IDENTITY
	};
}

static void successful_step(bool use_graphs)
{
	struct fixture fixture;
	if (!fixture_open(&fixture, TEST_SEGMENTS, false)) {
		expect(false, "sound fixture opens");
		fixture_close(&fixture);
		return;
	}
	const struct step_companion_pair pair = { 0, 1 };
	const struct step_companion_program table = descriptor(&fixture, &pair);
	const uint64_t old_identity = step_session_identity(fixture.driver);
	expect(step_companions_attach(fixture.driver, &table, 1) == STATUS_OK,
	       "companion schedule attaches without queued work");
	expect(step_session_identity(fixture.driver) != old_identity,
	       "TP snapshot identity is separate");
	const uint64_t work_before = test_host_memory_backend_work_count();
	expect(step_pipeline_start(fixture.driver, 1u) ==
			       STATUS_INVALID_ARGUMENT &&
		       test_host_memory_backend_work_count() == work_before,
	       "invalid pipeline depth refuses before work");
	if (use_graphs)
		expect(step_graphs_prepare(fixture.driver) == STATUS_OK,
		       "graphs prepare separate members");
	const struct step_row_record row = { 0, 1, 0, STEP_ROW_FRONT };
	struct step_row_result result = { 0 };
	expect(step_run_entry(fixture.driver, &row, 1, &result) == STATUS_OK,
	       "paired step completes both original handoff boundaries");
	expect(submitted == TEST_PAIR_MASK && !ordering_failure &&
		       submit_calls[0] == 1 && submit_calls[1] == 1,
	       "both members enqueue exactly once before completion waits");
	uint64_t sum = 0;
	const unsigned char *const arena = test_host_memory_backend_arena_of(
		fixture.cards[TEST_RESULT_CARD]);
	expect(arena != NULL, "downstream card keeps its arena");
	if (arena != NULL)
		(void)memcpy(&sum, arena + TEST_FINAL_OFFSET, sizeof(sum));
	expect(sum == TEST_BASE_VALUE + TEST_BASE_VALUE + 1u,
	       "downstream card receives both member boundaries");
	expect(result.token_id == STEP_NO_TOKEN,
	       "front step returns no sampled token");
	fixture_close(&fixture);
}

static void failed_step(enum test_failure mode)
{
	struct fixture fixture;
	if (!fixture_open(&fixture, TEST_SEGMENTS, false)) {
		expect(false, "failure fixture opens");
		fixture_close(&fixture);
		return;
	}
	const struct step_companion_pair pair = { 0, 1 };
	const struct step_companion_program table = descriptor(&fixture, &pair);
	expect(step_companions_attach(fixture.driver, &table, 1) == STATUS_OK,
	       "failure schedule attaches");
	failure_mode = mode;
	const struct step_row_record row = { 0, 1, 0, STEP_ROW_FRONT };
	struct step_row_result result = { 0 };
	expect(step_run_entry(fixture.driver, &row, 1, &result) ==
		       STATUS_INJECTED_FAULT,
	       "partial submission failure poisons the whole step");
	expect(wait_calls[0] != 0 && wait_calls[1] != 0 && !ordering_failure,
	       "both participant retirements attempted even after one failure");
	expect(step_run_entry(fixture.driver, &row, 1, &result) ==
		       STATUS_INJECTED_FAULT,
	       "failed group cannot be reused");
	const uint32_t waits_before_destroy = wait_calls[0];
	step_driver_destroy(fixture.driver);
	fixture.driver = NULL;
	if (mode == TEST_CANCELLED_WITH_PENDING_WORK) {
		expect(wait_calls[0] > waits_before_destroy &&
			       test_host_memory_backend_staging_block_count() !=
				       0,
		       "cancelling destruction retries drain and retains unretired staging until card close");
	}
	fixture_close(&fixture);
}

static void admission_refusals(void)
{
	struct fixture fixture;
	if (!fixture_open(&fixture, TEST_SEGMENTS, false)) {
		expect(false, "admission fixture opens");
		fixture_close(&fixture);
		return;
	}
	struct step_companion_pair pair = { 0, 1 };
	struct step_companion_program table = descriptor(&fixture, &pair);
	const uint64_t work_before = test_host_memory_backend_work_count();
	expect(step_companions_attach(NULL, &table, 1) ==
		       STATUS_INVALID_ARGUMENT,
	       "null driver refused");
	expect(step_companions_attach(fixture.driver, NULL, 1) ==
		       STATUS_INVALID_ARGUMENT,
	       "null table refused");
	struct execution_program other = fixture.program;
	table.program = &other;
	expect(step_companions_attach(fixture.driver, &table, 1) ==
		       STATUS_INVALID_ARGUMENT,
	       "foreign program pointer refused");
	table.program = &fixture.program;
	pair.second_segment = TEST_RESULT_CARD;
	expect(step_companions_attach(fixture.driver, &table, 1) ==
		       STATUS_INVALID_ARGUMENT,
	       "nonadjacent or nested pair refused");
	pair = (struct step_companion_pair){ 1, TEST_RESULT_CARD };
	expect(step_companions_attach(fixture.driver, &table, 1) ==
		       STATUS_INVALID_ARGUMENT,
	       "members of different stages refused");
	pair = (struct step_companion_pair){ 0, 1 };
	table.pair_count = TEST_PAIR_CARDS;
	expect(step_companions_attach(fixture.driver, &table, 1) ==
		       STATUS_INVALID_ARGUMENT,
	       "overlapping or excessive pair table refused before reading beyond it");
	table.pair_count = 1;
	table.arithmetic_identity = 0;
	expect(step_companions_attach(fixture.driver, &table, 1) ==
		       STATUS_INVALID_ARGUMENT,
	       "missing arithmetic identity refused");
	table.arithmetic_identity = TEST_ARITHMETIC_IDENTITY;
	expect(test_host_memory_backend_work_count() == work_before,
	       "all admission refusals queue nothing");
	expect(step_companions_attach(fixture.driver, &table, 1) == STATUS_OK,
	       "sound attachment remains possible");
	expect(step_companions_attach(fixture.driver, &table, 1) ==
		       STATUS_INVALID_ARGUMENT,
	       "second attachment refused");
	fixture_close(&fixture);
	if (!fixture_open(&fixture, 1, false)) {
		expect(false, "phase fixture opens");
		fixture_close(&fixture);
		return;
	}
	table = descriptor(&fixture, &pair);
	expect(step_companions_attach(fixture.driver, &table, 1) ==
		       STATUS_INVALID_ARGUMENT,
	       "front phase cannot split participants");
	fixture_close(&fixture);
	if (!fixture_open(&fixture, TEST_SEGMENTS, true)) {
		expect(false, "peer dependency fixture opens");
		fixture_close(&fixture);
		return;
	}
	table = descriptor(&fixture, &pair);
	const uint64_t dependency_work = test_host_memory_backend_work_count();
	expect(step_companions_attach(fixture.driver, &table, 1) ==
		       STATUS_INVALID_ARGUMENT,
	       "member-to-peer handoff dependency refused");
	expect(test_host_memory_backend_work_count() == dependency_work,
	       "dependency refusal queues nothing");
	fixture_close(&fixture);
}

int main(void)
{
	successful_step(false);
	successful_step(true);
	failed_step(TEST_FIRST_SUBMIT_FAILS);
	failed_step(TEST_SECOND_SUBMIT_FAILS);
	failed_step(TEST_CANCELLED_WITH_PENDING_WORK);
	admission_refusals();
	if (failures == 0)
		(void)puts(
			"step_companions_test: eager/graph co-enqueue, member handoffs, "
			"refusals and failure retirement passed");
	return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
