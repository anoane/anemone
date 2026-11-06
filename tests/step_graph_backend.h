/* SPDX-License-Identifier: AGPL-3.0-or-later */
#ifndef TEST_STEP_GRAPH_BACKEND_H
#define TEST_STEP_GRAPH_BACKEND_H
#include <stddef.h>
#ifdef TEST_STEP_FIXTURE_IMPLEMENTATION
/* Source-only qualification candidate. Reuses the actual step_test model, delayed copies, work log and arenas. */
#define TEST_STEP_FIXTURE_IMPLEMENTATION
#include "step_fixture.h"
#include "step_programs.h"
#include "step_interleaved.h"

#define TEST_GRAPH_SAVED_NONLAUNCH_WORK 64u
#define TEST_GRAPH_SAVED_WORK \
	(TEST_GRAPH_SPLIT_SEGMENTS * CARDS + TEST_GRAPH_SAVED_NONLAUNCH_WORK)
#define TEST_GRAPH_REPLAYED_STEPS 4u
#define TEST_GRAPH_PARTIAL_RUNS 3u
#define TEST_GRAPH_FAILURE_CALL 2u
#define TEST_GRAPH_RETIRED_BLOCKS 2u
#define TEST_GRAPH_NEXT_POSITION 1u
#define TEST_GRAPH_TERMINAL_SEGMENTS 2u
#define TEST_GRAPH_SPLIT_RUNS (BACKEND_MAXIMUM_GRAPHS + 1u)
#define TEST_GRAPH_SPLIT_SEGMENTS \
	(TEST_GRAPH_SPLIT_RUNS + TEST_GRAPH_TERMINAL_SEGMENTS)

struct test_graph_record {
	struct kernel_launch launches[LAUNCHES_PER_CARD];
	uint32_t launch_count;
};

static struct test_graph_record test_graphs[CARDS][BACKEND_MAXIMUM_GRAPHS];
static uint32_t test_graph_counts[CARDS], test_graph_owned[CARDS],
	test_graph_capture_calls[CARDS];
static uint32_t test_graph_wait_calls[CARDS], test_graph_room,
	test_graph_replays, test_graph_expected_replays;
static uint32_t test_graph_failure_card, test_graph_failure_call;
static uint32_t test_graph_invalid_handle_card;
static result_status test_graph_capture_answer, test_graph_wait_answers[CARDS];
static bool test_graph_pending[CARDS], test_graph_failed[CARDS],
	test_graph_replay_failure;
static uint64_t test_graph_cover_ticket[CARDS];
static void *test_graph_retired[TEST_GRAPH_RETIRED_BLOCKS];
static uint32_t test_graph_retired_count;
static struct test_host_memory_backend_work
	test_graph_eager_work[TEST_GRAPH_SAVED_WORK];
static unsigned char test_graph_eager_arenas[CARDS][ARENA_BYTES];
static uint64_t test_graph_eager_count;
static bool test_graph_save_eager;

static uint32_t test_graph_card_index(const struct device_card *card)
{
	return card != NULL ? card_index_of(card) : CARDS;
}

/* The fake capture validates this fixture's kernel records without executing them. All launch fields are checked
 * before any resource is owned; this intentionally mirrors the local host backend's complete kernel table checks. */
static bool test_graph_launch_valid(uint32_t card,
				    const struct kernel_launch *launch)
{
	static const uint32_t most_grid[LAUNCH_DIMENSIONS] = {
		TEST_HOST_MEMORY_BACKEND_MOST_GRID_X,
		TEST_HOST_MEMORY_BACKEND_MOST_GRID_Y_Z,
		TEST_HOST_MEMORY_BACKEND_MOST_GRID_Y_Z
	};
	static const uint32_t most_block[LAUNCH_DIMENSIONS] = {
		TEST_HOST_MEMORY_BACKEND_MOST_BLOCK_X_Y,
		TEST_HOST_MEMORY_BACKEND_MOST_BLOCK_X_Y,
		TEST_HOST_MEMORY_BACKEND_MOST_BLOCK_Z
	};
	static const unsigned char zero_argument[LAUNCH_ARGUMENT_BYTES];
	struct test_host_memory_backend_kernel kernel;
	uint64_t threads = 1;
	bool valid = launch != NULL && test_host_memory_backend_kernel_of(
					       card, launch->kernel, &kernel);
	valid = valid && launch->branch <= LAUNCH_JOIN &&
		launch->argument_byte_count == kernel.argument_byte_count &&
		launch->argument_byte_count <= LAUNCH_ARGUMENT_BYTES &&
		launch->dynamic_shared_byte_count <=
			(card == LAST_CARD ?
				 TEST_HOST_MEMORY_BACKEND_MOST_DYNAMIC_SHARED_BYTES_SM_120 :
				 TEST_HOST_MEMORY_BACKEND_MOST_DYNAMIC_SHARED_BYTES_SM_80);
	if (valid) {
		valid = memcmp(launch->argument + launch->argument_byte_count,
			       zero_argument,
			       LAUNCH_ARGUMENT_BYTES -
				       launch->argument_byte_count) == 0;
	}
	for (uint32_t dimension = 0; valid && dimension < LAUNCH_DIMENSIONS;
	     dimension++) {
		valid = launch->grid[dimension] != 0 &&
			launch->grid[dimension] <= most_grid[dimension] &&
			launch->block[dimension] != 0 &&
			launch->block[dimension] <= most_block[dimension];
		threads *= valid ? launch->block[dimension] : 1u;
		valid = valid &&
			threads <=
				TEST_HOST_MEMORY_BACKEND_MOST_THREADS_PER_BLOCK;
	}
	return valid;
}

MUST_CHECK static result_status
test_graph_capture(struct device_card *card,
		   const struct kernel_launch *launches, uint32_t launch_count,
		   uint32_t *graph_out)
{
	const uint32_t index = test_graph_card_index(card);
	if (index == CARDS || launches == NULL || graph_out == NULL ||
	    launch_count == 0 || launch_count > LAUNCHES_PER_CARD) {
		return STATUS_INVALID_ARGUMENT;
	}
	for (uint32_t launch = 0; launch < launch_count; launch++) {
		if (!test_graph_launch_valid(index, &launches[launch])) {
			return STATUS_INVALID_ARGUMENT;
		}
	}
	test_graph_capture_calls[index]++;
	if (test_graph_failed[index]) {
		return STATUS_DEVICE_ERROR;
	}
	if (index == test_graph_failure_card &&
	    test_graph_capture_calls[index] == test_graph_failure_call) {
		if (test_graph_capture_answer == STATUS_DEVICE_ERROR) {
			test_graph_owned
				[index]++; /* partial definition/upload remains card-owned, with no published handle */
			test_graph_pending[index] = true;
			test_graph_failed[index] = true;
		}
		return test_graph_capture_answer;
	}
	if (test_graph_counts[index] >= test_graph_room) {
		return STATUS_BUSY_VALUE;
	}
	struct test_graph_record *const graph =
		&test_graphs[index][test_graph_counts[index]];
	memcpy(graph->launches, launches, launch_count * sizeof(*launches));
	graph->launch_count = launch_count;
	test_graph_owned[index]++;
	test_graph_pending[index] =
		true; /* executable upload, without model launches or a caller staging pointer */
	test_graph_cover_ticket[index] = 0;
	*graph_out = index == test_graph_invalid_handle_card ?
			     UINT32_MAX :
			     test_graph_counts[index];
	test_graph_counts[index]++;
	return STATUS_OK;
}

MUST_CHECK static result_status test_graph_run(struct device_card *card,
					       uint32_t graph)
{
	const uint32_t index = test_graph_card_index(card);
	if (index == CARDS || graph >= test_graph_counts[index]) {
		return STATUS_INVALID_ARGUMENT;
	}
	if (test_graph_failed[index]) {
		return STATUS_DEVICE_ERROR;
	}
	test_graph_pending[index] = true;
	test_graph_cover_ticket[index] = 0;
	launch_runs++;
	test_graph_replays++;
	if (test_graph_replay_failure && index == 1) {
		test_graph_failed[index] = true;
		return STATUS_DEVICE_ERROR;
	}
	return test_host_memory_backend.launch_run(
		card, test_graphs[index][graph].launches,
		test_graphs[index][graph].launch_count);
}

MUST_CHECK static result_status test_graph_launch_wait(struct device_card *card,
						       uint64_t limit)
{
	const uint32_t index = test_graph_card_index(card);
	if (index == CARDS || limit == 0) {
		return STATUS_INVALID_ARGUMENT;
	}
	test_graph_wait_calls[index]++;
	expect(limit == WAIT_LIMIT,
	       "graph work keeps the driver's required bounded wait deadline");
	const result_status status =
		test_graph_failed[index] ?
			STATUS_DEVICE_ERROR :
		test_graph_wait_answers[index] != STATUS_OK ?
			test_graph_wait_answers[index] :
			test_host_memory_backend.launch_wait(card, limit);
	test_graph_pending[index] =
		status == STATUS_OK ? false : test_graph_pending[index];
	return status;
}

MUST_CHECK static result_status
test_graph_download(struct device_card *card, uint64_t offset,
		    void *destination, uint64_t bytes, uint64_t *ticket_out)
{
	const uint32_t index = test_graph_card_index(card);
	const result_status status =
		index == CARDS ?
			STATUS_INVALID_ARGUMENT :
		test_graph_failed[index] ?
			STATUS_DEVICE_ERROR :
			checked_download_begin(card, offset, destination, bytes,
					       ticket_out);
	if (status == STATUS_OK && test_graph_pending[index]) {
		test_graph_cover_ticket[index] = *ticket_out;
	}
	return status;
}

MUST_CHECK static result_status
test_graph_upload(struct device_card *card, uint64_t offset, const void *source,
		  uint64_t bytes, uint64_t *ticket_out)
{
	const uint32_t index = test_graph_card_index(card);
	const result_status status =
		index == CARDS ? STATUS_INVALID_ARGUMENT :
		test_graph_failed[index] ?
				 STATUS_DEVICE_ERROR :
				 checked_upload_begin(card, offset, source,
						      bytes, ticket_out);
	if (status == STATUS_OK && test_graph_pending[index]) {
		test_graph_cover_ticket[index] = *ticket_out;
	}
	return status;
}

MUST_CHECK static result_status test_graph_upload_wait(struct device_card *card,
						       uint64_t ticket,
						       uint64_t limit)
{
	const uint32_t index = test_graph_card_index(card);
	const result_status status =
		index == CARDS ? STATUS_INVALID_ARGUMENT :
		test_graph_failed[index] ?
				 STATUS_DEVICE_ERROR :
				 checked_upload_wait(card, ticket, limit);
	if (status == STATUS_OK && test_graph_cover_ticket[index] != 0 &&
	    ticket >= test_graph_cover_ticket[index]) {
		test_graph_pending[index] = false;
	}
	return status;
}

static void test_graph_retire(void *block)
{
	if (block != NULL) {
		expect(test_graph_retired_count < TEST_GRAPH_RETIRED_BLOCKS,
		       "bounded deferred staging ownership");
		if (test_graph_retired_count < TEST_GRAPH_RETIRED_BLOCKS) {
			test_graph_retired[test_graph_retired_count++] = block;
		}
	}
}

static void test_graph_reset(uint32_t room)
{
	memset(test_graph_counts, 0, sizeof(test_graph_counts));
	memset(test_graph_owned, 0, sizeof(test_graph_owned));
	memset(test_graph_capture_calls, 0, sizeof(test_graph_capture_calls));
	memset(test_graph_wait_calls, 0, sizeof(test_graph_wait_calls));
	memset(test_graph_wait_answers, 0, sizeof(test_graph_wait_answers));
	memset(test_graph_pending, 0, sizeof(test_graph_pending));
	memset(test_graph_failed, 0, sizeof(test_graph_failed));
	memset(test_graph_cover_ticket, 0, sizeof(test_graph_cover_ticket));
	test_graph_room = room;
	test_graph_failure_card = CARDS;
	test_graph_failure_call = TEST_GRAPH_FAILURE_CALL;
	test_graph_invalid_handle_card = CARDS;
	test_graph_capture_answer = STATUS_OK;
	test_graph_replay_failure = false;
	test_graph_replays = 0;
	test_graph_retired_count = 0;
}

/* Only successful card close releases the fake's pending graph work and its quarantined staging blocks. */
static void test_graph_close(void)
{
	cards_close();
	memset(test_graph_counts, 0, sizeof(test_graph_counts));
	memset(test_graph_owned, 0, sizeof(test_graph_owned));
	memset(test_graph_pending, 0, sizeof(test_graph_pending));
	memset(test_graph_failed, 0, sizeof(test_graph_failed));
	for (uint32_t block = 0; block < test_graph_retired_count; block++) {
		test_host_memory_backend.staging_free(
			test_graph_retired[block]);
	}
	test_graph_retired_count = 0;
}

static void test_graph_observe(void)
{
	const uint64_t count = test_host_memory_backend_work_count();
	bool equal = count <= TEST_GRAPH_SAVED_WORK &&
		     (test_graph_save_eager || count == test_graph_eager_count);
	for (uint64_t index = 0; equal && index < count; index++) {
		struct test_host_memory_backend_work work = { 0 };
		equal = test_host_memory_backend_work_at(index, &work);
		if (equal && test_graph_save_eager) {
			test_graph_eager_work[index] = work;
		} else if (equal) {
			equal = memcmp(&test_graph_eager_work[index], &work,
				       sizeof(work)) == 0;
		}
	}
	for (uint32_t card = 0; card < CARDS; card++) {
		if (test_graph_save_eager) {
			memcpy(test_graph_eager_arenas[card], arena_of(card),
			       ARENA_BYTES);
		}
		equal = equal && memcmp(test_graph_eager_arenas[card],
					arena_of(card), ARENA_BYTES) == 0;
		expect(!test_graph_pending[card],
		       "successful step has a completion witness for every replay");
	}
	test_graph_eager_count = test_graph_save_eager ? count :
							 test_graph_eager_count;
	expect(equal,
	       "eager and mixed/full replay have identical complete work records and every arena byte");
	expect(test_graph_replays == test_graph_expected_replays,
	       "exact deterministic replay/eager policy");
}

#endif
#endif
