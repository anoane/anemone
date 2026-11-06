/* SPDX-License-Identifier: AGPL-3.0-or-later */
#ifndef TEST_STEP_GRAPH_FAILURE_BACKEND_H
#define TEST_STEP_GRAPH_FAILURE_BACKEND_H
#include <stddef.h>
#ifdef TEST_STEP_FIXTURE_IMPLEMENTATION
#include "step_driver.h"

#define TEST_GRAPH_API_STEPS 2u
#define TEST_GRAPH_API_REPLAY_RUNS_PER_STEP (LAUNCH_RUNS - 1u)
#define TEST_GRAPH_API_MIDDLE_CAPTURE_COUNT 2u

enum test_graph_api_run_failure {
	TEST_GRAPH_API_RUN_SOUND,
	TEST_GRAPH_API_RUN_PARTIAL_DEVICE
};

static struct step_driver *test_graph_api_driver;
static uint32_t test_graph_api_allocations, test_graph_api_frees;
static uint32_t test_graph_api_capture_entries, test_graph_api_run_entries;
static enum test_graph_api_run_failure test_graph_api_run_failure;
static struct kernel_launch test_graph_api_partial_launch;
static uint32_t test_graph_api_history[SESSIONS][CONTEXT];
static uint32_t test_graph_api_token_counts[SESSIONS];

/* These observers run at the backend boundary, before a fake queues any work. A missing pre-submit mark is
 * observable even when a later ticket would otherwise hide it. No production hook or state mutation is added. */
MUST_CHECK static result_status
test_graph_api_capture(struct device_card *card,
		       const struct kernel_launch *launches,
		       uint32_t launch_count, uint32_t *graph_out)
{
	const uint32_t index = test_graph_card_index(card);
	expect(test_graph_api_driver != NULL && index < CARDS,
	       "capture is bound to the current actual driver/card");
	if (test_graph_api_driver == NULL || index == CARDS) {
		return STATUS_INVALID_ARGUMENT;
	}
	expect(test_graph_api_driver->sequence == 0 &&
		       test_graph_api_driver->run_graphs == NULL,
	       "every capture is startup-only and the replay table stays unpublished");
	expect(test_graph_api_driver->uncovered_work[index],
	       "capture upload is uncovered before backend entry");
	test_graph_api_capture_entries++;
	return test_graph_capture(card, launches, launch_count, graph_out);
}

MUST_CHECK static result_status test_graph_api_run(struct device_card *card,
						   uint32_t graph)
{
	const uint32_t index = test_graph_card_index(card);
	expect(test_graph_api_driver != NULL && index < CARDS,
	       "replay is bound to the current actual driver/card");
	if (test_graph_api_driver == NULL || index == CARDS) {
		return STATUS_INVALID_ARGUMENT;
	}
	expect(test_graph_api_driver->uncovered_work[index],
	       "graph submission is uncovered before backend entry");
	test_graph_api_run_entries++;
	if (test_graph_api_run_failure == TEST_GRAPH_API_RUN_SOUND ||
	    index != LAST_CARD) {
		return test_graph_run(card, graph);
	}
	if (graph >= test_graph_counts[index] ||
	    test_graphs[index][graph].launch_count <= 1) {
		return STATUS_INVALID_ARGUMENT;
	}
	/* Queue the graph's actual first model launch, then lose submission completion before its sampler launch.
     * The last driver copy ticket predates this work and cannot prove it complete. */
	test_graph_pending[index] = true;
	test_graph_cover_ticket[index] = 0;
	test_graph_api_partial_launch = test_graphs[index][graph].launches[0];
	const result_status prefix_status = test_host_memory_backend.launch_run(
		card, &test_graph_api_partial_launch, 1);
	if (prefix_status != STATUS_OK) {
		return prefix_status;
	}
	launch_runs++;
	test_graph_replays++;
	test_graph_failed[index] = true;
	return STATUS_DEVICE_ERROR;
}

MUST_CHECK static result_status
test_graph_api_staging_allocate(uint64_t bytes, void **memory_out)
{
	expect(test_graph_api_driver == NULL,
	       "driver staging allocation occurs only during creation");
	test_graph_api_allocations++;
	return test_host_memory_backend.staging_allocate(bytes, memory_out);
}

static void test_graph_api_staging_free(void *memory)
{
	test_graph_api_frees += memory != NULL ? 1u : 0u;
	test_host_memory_backend.staging_free(memory);
}

static void test_graph_api_reset(void)
{
	test_graph_reset(BACKEND_MAXIMUM_GRAPHS);
	test_step_use_graphs = false;
	test_step_observe = NULL;
	test_graph_api_driver = NULL;
	test_graph_api_allocations = 0;
	test_graph_api_frees = 0;
	test_graph_api_capture_entries = 0;
	test_graph_api_run_entries = 0;
	test_graph_api_run_failure = TEST_GRAPH_API_RUN_SOUND;
}

static struct step_driver *test_graph_api_driver_of(struct fixture *fixture)
{
	fixture_build(fixture, cards_open());
	test_graph_api_driver = driver_of(fixture, PROGRAMS);
	expect(test_graph_api_driver != NULL,
	       "failure case created its actual driver");
	return test_graph_api_driver;
}

static void test_graph_api_history_save(const struct step_driver *driver)
{
	memcpy(test_graph_api_history, driver->histories,
	       sizeof(test_graph_api_history));
	memcpy(test_graph_api_token_counts, driver->token_counts,
	       sizeof(test_graph_api_token_counts));
}

static void test_graph_api_history_unchanged(const struct step_driver *driver)
{
	expect(memcmp(test_graph_api_history, driver->histories,
		      sizeof(test_graph_api_history)) == 0 &&
		       memcmp(test_graph_api_token_counts, driver->token_counts,
			      sizeof(test_graph_api_token_counts)) == 0,
	       "failed capture/submission commits no host history or token count in any session");
}

static void test_graph_api_no_work_after_failure(struct step_driver *driver,
						 result_status failure)
{
	const uint32_t captures = test_graph_api_capture_entries,
		       submissions = test_graph_api_run_entries;
	test_graph_sticky(driver, failure);
	expect(captures == test_graph_api_capture_entries &&
		       submissions == test_graph_api_run_entries,
	       "the sticky failure cannot capture, resubmit or fall back to eager work");
}

static void test_graph_api_end_capture(void)
{
#if LOGGING_HOOKS_ENABLED
	(void)capture_end();
#else
	expect(capture_end() == 0,
	       "release keeps failure statuses and emits no optional diagnostic errors");
#endif
}

#endif
#endif
