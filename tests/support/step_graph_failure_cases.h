/* SPDX-License-Identifier: AGPL-3.0-or-later */
#ifndef TEST_STEP_GRAPH_FAILURE_CASES_H
#define TEST_STEP_GRAPH_FAILURE_CASES_H
#include <stddef.h>
#ifdef TEST_STEP_FIXTURE_IMPLEMENTATION
#include "step_graph_failure_backend.h"

enum test_graph_api_startup_case {
	TEST_GRAPH_API_FIRST_CAPTURE_ERROR,
	TEST_GRAPH_API_LATER_CARD_ERROR,
	TEST_GRAPH_API_EARLIER_WAIT_ERROR,
	TEST_GRAPH_API_STARTUP_CASE_COUNT
};

static void test_graph_api_startup_failures(void)
{
	struct fixture fixture;
	for (uint32_t scenario = 0;
	     scenario < TEST_GRAPH_API_STARTUP_CASE_COUNT; scenario++) {
		test_graph_api_reset();
		struct step_driver *const driver =
			test_graph_api_driver_of(&fixture);
		if (driver == NULL) {
			test_graph_close();
			return;
		}
		test_graph_api_history_save(driver);
		test_graph_failure_card =
			scenario == TEST_GRAPH_API_FIRST_CAPTURE_ERROR ?
				0 :
				LAST_CARD;
		test_graph_failure_call = 1;
		test_graph_capture_answer =
			scenario == TEST_GRAPH_API_EARLIER_WAIT_ERROR ?
				STATUS_UNSUPPORTED :
				STATUS_DEVICE_ERROR;
		test_graph_wait_answers[0] =
			scenario == TEST_GRAPH_API_EARLIER_WAIT_ERROR ?
				STATUS_TIMED_OUT :
				STATUS_OK;
		const result_status expected =
			scenario == TEST_GRAPH_API_EARLIER_WAIT_ERROR ?
				STATUS_TIMED_OUT :
				STATUS_DEVICE_ERROR;
		context_text = "graph startup completion failures";
		capture_begin();
		expect_status(step_graphs_prepare(driver), expected,
			      "startup preserves required failure status");
		expect(driver->sequence == 0 && driver->run_graphs == NULL &&
			       driver->graphs_attempted &&
			       driver->failure == expected,
		       "failed preparation publishes no table and latches failure before any step");
		test_graph_api_history_unchanged(driver);
		expect(test_host_memory_backend_work_count() == 0 &&
			       test_graph_api_run_entries == 0,
		       "capture/upload failure executes no model launch or step copy");
		if (scenario == TEST_GRAPH_API_FIRST_CAPTURE_ERROR) {
			expect(test_graph_capture_calls[0] == 1 &&
				       test_graph_capture_calls[1] == 0 &&
				       test_graph_capture_calls[LAST_CARD] ==
					       0 &&
				       test_graph_owned[0] == 1 &&
				       test_graph_counts[0] == 0 &&
				       driver->uncovered_work[0],
			       "first failed capture keeps its partial resource and unknown "
			       "upload, with no later capture");
		} else {
			expect(test_graph_counts[0] == 1 &&
				       test_graph_counts[1] ==
					       TEST_GRAPH_API_MIDDLE_CAPTURE_COUNT &&
				       test_graph_counts[LAST_CARD] == 0 &&
				       test_graph_capture_calls[LAST_CARD] == 1,
			       "earlier cards capture successfully before the first capture on the last card fails");
			expect(test_graph_wait_calls[0] != 0 &&
				       test_graph_wait_calls[1] != 0 &&
				       test_graph_wait_calls[LAST_CARD] != 0,
			       "preparation drains every attempted card even after a later error");
			expect(!driver->uncovered_work[1] &&
				       !test_graph_pending[1],
			       "the healthy middle card completes independently of the failed card");
			expect(scenario == TEST_GRAPH_API_EARLIER_WAIT_ERROR ?
				       driver->uncovered_work[0] &&
					       !driver->uncovered_work[LAST_CARD] :
				       !driver->uncovered_work[0] &&
					       driver->uncovered_work[LAST_CARD],
			       "the lost completion proof stays with exactly its card");
		}
		test_graph_api_no_work_after_failure(driver, expected);
		uint32_t owned[CARDS];
		memcpy(owned, test_graph_owned, sizeof(owned));
		step_driver_destroy(driver);
		test_graph_api_driver = NULL;
		expect(test_graph_api_frees == 0 &&
			       test_graph_retired_count == 1 &&
			       test_host_memory_backend_staging_block_count() ==
				       1 &&
			       memcmp(owned, test_graph_owned, sizeof(owned)) ==
				       0,
		       "unknown capture completion transfers staging and retains every card-owned graph resource");
		test_graph_api_end_capture();
		test_graph_close();
		expect(test_host_memory_backend_staging_block_count() == 0,
		       "retired staging is freed only after the fake card close finishes");
	}
}

enum test_graph_api_fallback_case {
	TEST_GRAPH_API_DEFAULT_EAGER,
	TEST_GRAPH_API_FIRST_BUSY,
	TEST_GRAPH_API_FIRST_UNSUPPORTED,
	TEST_GRAPH_API_FALLBACK_CASE_COUNT
};

static void test_graph_api_fallbacks(void)
{
	struct fixture fixture;
	for (uint32_t scenario = 0;
	     scenario < TEST_GRAPH_API_FALLBACK_CASE_COUNT; scenario++) {
		test_graph_api_reset();
		struct step_driver *const driver =
			test_graph_api_driver_of(&fixture);
		if (driver == NULL) {
			test_graph_close();
			return;
		}
		context_text = "graph startup eager fallbacks";
		if (scenario != TEST_GRAPH_API_DEFAULT_EAGER) {
			test_graph_failure_card = 0;
			test_graph_failure_call = 1;
			test_graph_capture_answer =
				scenario == TEST_GRAPH_API_FIRST_BUSY ?
					STATUS_BUSY_VALUE :
					STATUS_UNSUPPORTED;
			expect_status(
				step_graphs_prepare(driver),
				scenario == TEST_GRAPH_API_FIRST_BUSY ?
					STATUS_OK :
					STATUS_UNSUPPORTED,
				"first capture refusal has its documented API status");
			expect(driver->failure == STATUS_OK &&
				       driver->graphs_attempted &&
				       !driver->uncovered_work[0],
			       "a completed non-device refusal leaves a usable eager driver");
		}
		if (scenario == TEST_GRAPH_API_FIRST_BUSY) {
			expect(test_graph_capture_calls[0] == 1 &&
				       test_graph_counts[0] == 0 &&
				       test_graph_counts[1] != 0 &&
				       test_graph_counts[LAST_CARD] != 0 &&
				       driver->run_graphs != NULL,
			       "first BUSY disables only that card and preserves prepared runs on the other cards");
			if (driver->run_graphs != NULL) {
				uint32_t graph_base = 0;
				for (uint32_t program = 0; program < PROGRAMS;
				     program++) {
					expect(driver->run_graphs[graph_base +
								  SEGMENT_EMBED] ==
						       0,
					       "every later program retains eager submission for the full card");
					graph_base += fixture.programs[program]
							      .segment_count;
				}
			}
		} else {
			expect(driver->run_graphs == NULL &&
				       test_graph_counts[0] == 0 &&
				       test_graph_counts[1] == 0 &&
				       test_graph_counts[LAST_CARD] == 0,
			       "unprepared or unsupported drivers retain the independent eager path");
		}
		const uint32_t captures = test_graph_api_capture_entries,
			       allocations = test_graph_api_allocations;
		const uint32_t *const table = driver->run_graphs;
#if CHECKING_HOOKS_ENABLED
		expect(fault_arm("step_graph_memory", 1),
		       "token-loop graph allocation canary armed");
#endif
		for (uint32_t step = 0; step < TEST_GRAPH_API_STEPS; step++) {
			test_graph_simple_step(driver, step);
			expect(test_graph_api_capture_entries == captures &&
				       test_graph_api_allocations ==
					       allocations &&
				       driver->run_graphs == table,
			       "repeated tokens neither capture graphs nor allocate driver staging/table");
		}
		expect_status(
			step_graphs_prepare(driver), STATUS_INVALID_ARGUMENT,
			"late or repeated preparation cannot allocate/capture");
#if CHECKING_HOOKS_ENABLED
		expect(fault_disarm() == 0,
		       "graph allocation fault point is absent from token execution");
#endif
		test_graph_save_eager = scenario ==
					TEST_GRAPH_API_DEFAULT_EAGER;
		test_graph_expected_replays =
			scenario == TEST_GRAPH_API_FIRST_BUSY ?
				TEST_GRAPH_API_STEPS *
					TEST_GRAPH_API_REPLAY_RUNS_PER_STEP :
				0;
		test_graph_observe();
		step_driver_destroy(driver);
		test_graph_api_driver = NULL;
		expect(test_graph_api_frees == 1 &&
			       test_graph_retired_count == 0,
		       "proved completion frees ordinary driver staging without quarantine");
		test_graph_close();
	}
}

enum test_graph_api_submission_case {
	TEST_GRAPH_API_RETIRES_PENDING,
	TEST_GRAPH_API_LEGACY_QUARANTINE,
	TEST_GRAPH_API_SUBMISSION_CASE_COUNT
};

static void test_graph_api_partial_submissions(void)
{
	struct fixture fixture;
	for (uint32_t scenario = 0;
	     scenario < TEST_GRAPH_API_SUBMISSION_CASE_COUNT; scenario++) {
		test_graph_api_reset();
		struct step_driver *const driver =
			test_graph_api_driver_of(&fixture);
		if (driver == NULL) {
			test_graph_close();
			return;
		}
		expect_status(step_graphs_prepare(driver), STATUS_OK,
			      "partial-submission fixture prepared");
		test_graph_simple_step(driver, 0);
		test_graph_api_history_save(driver);
		test_graph_api_run_failure = TEST_GRAPH_API_RUN_PARTIAL_DEVICE;
		const result_status expected = STATUS_DEVICE_ERROR;
		if (scenario == TEST_GRAPH_API_LEGACY_QUARANTINE) {
			checked_backend.staging_retire =
				NULL; /* exercise the existing legacy quarantine policy */
		}
		void *const staging = driver->staging;
		const uint32_t captures = test_graph_api_capture_entries,
			       allocations = test_graph_api_allocations;
		const struct step_row_record row = { 1, POSITION_TWO, 0,
						     STEP_ROW_LOGITS };
		struct step_row_result result = { 0 };
		context_text = "partially queued graph submission";
		capture_begin();
#if CHECKING_HOOKS_ENABLED
		expect(fault_arm("step_graph_memory", 1),
		       "failed-token graph allocation canary armed");
#endif
		expect_status(
			step_run_entry(driver, &row, 1, &result), expected,
			"submission failure preserves status after a real model-launch prefix");
		expect(driver->uncovered_work[LAST_CARD] &&
			       driver->unconfirmed[LAST_CARD] != 0 &&
			       test_graph_pending[LAST_CARD] &&
			       test_graph_cover_ticket[LAST_CARD] == 0,
		       "the earlier copy ticket cannot cover later failed graph submission");
		test_graph_api_history_unchanged(driver);
		const uint64_t count = test_host_memory_backend_work_count();
		struct test_host_memory_backend_work work = { 0 };
		expect(count != 0 &&
			       test_host_memory_backend_work_at(count - 1u,
								&work) &&
			       work.kind == TEST_HOST_MEMORY_BACKEND_LAUNCH &&
			       work.card_index == LAST_CARD &&
			       memcmp(&work.launch,
				      &test_graph_api_partial_launch,
				      sizeof(work.launch)) == 0,
		       "failure fixture really queued the captured launch prefix and no result download");
		test_graph_api_no_work_after_failure(driver, expected);
#if CHECKING_HOOKS_ENABLED
		expect(fault_disarm() == 0,
		       "failed token and refusal latch perform no graph allocation");
#endif
		expect(captures == test_graph_api_capture_entries &&
			       allocations == test_graph_api_allocations,
		       "failure never recaptures or allocates staging in the token loop");
		step_driver_destroy(driver);
		test_graph_api_driver = NULL;
		expect(test_graph_api_frees == 0 &&
			       test_graph_pending[LAST_CARD] &&
			       test_host_memory_backend_staging_block_count() ==
				       1,
		       "failed completion cannot free staging still exposed to device readers");
		expect(test_graph_retired_count ==
			       (scenario == TEST_GRAPH_API_RETIRES_PENDING ?
					1u :
					0u),
		       "supported retirement and legacy allocated quarantine are exercised independently");
		test_graph_api_end_capture();
		test_graph_close();
		if (scenario == TEST_GRAPH_API_LEGACY_QUARANTINE) {
			expect(test_host_memory_backend_staging_block_count() ==
				       1,
			       "legacy quarantine remains allocated through driver destruction "
			       "until actual card close");
			/* Actual card close returned; this test now owns a block with no remaining reader. */
			test_host_memory_backend.staging_free(staging);
			checked_backend.staging_retire = test_graph_retire;
		}
		expect(test_host_memory_backend_staging_block_count() == 0,
		       "all test-only staging is released after close");
	}
}

static void test_graph_api_failure_cases(void)
{
	const struct backend_operations saved = checked_backend;
	checked_backend.graph_capture = test_graph_api_capture;
	checked_backend.graph_run = test_graph_api_run;
	checked_backend.staging_allocate = test_graph_api_staging_allocate;
	checked_backend.staging_free = test_graph_api_staging_free;
	test_graph_api_startup_failures();
	test_graph_api_fallbacks();
	test_graph_api_partial_submissions();
	checked_backend = saved;
}

#endif
#endif
