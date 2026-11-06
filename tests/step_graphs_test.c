/* SPDX-License-Identifier: AGPL-3.0-or-later */
#define TEST_STEP_FIXTURE_IMPLEMENTATION
#include "step_graph_backend.h"
#include "step_draft_cases.h"

/* Prepared graph replay uses the same target-result and draft-tail boundary as eager runs. The target draw stays
 * intact after the tail, and an ordinary step neither replays that graph nor downloads a stale round. */
static void test_graph_device_round(void)
{
	context_text = "device round graph";
	test_graph_reset(BACKEND_MAXIMUM_GRAPHS);
	struct fixture fixture;
	struct segment_record segments[PROGRAMS][SEGMENTS + 1u];
	device_fixture_build(&fixture, segments, cards_open());
	device_draft_break = DEVICE_DRAFT_SOUND;
	expect_status(test_host_memory_backend_set_stand_in(
			      KERNEL_FILE_FILL, KERNEL_FILL_WORDS,
			      device_draft_stand_in, NULL),
		      STATUS_OK, "the graph's draft tail stand-in");
	struct step_driver *const driver = driver_of(&fixture, PROGRAMS);
	expect_status(step_graphs_prepare(driver), STATUS_OK,
		      "target and draft graphs prepared independently");
	const struct step_row_record rows[ROW_CAPACITY] = {
		{ 0, 1, 0, STEP_ROW_LOGITS | STEP_ROW_DRAFTS },
		{ 1, 1, 0, STEP_ROW_LOGITS | STEP_ROW_PROPOSED },
		{ DEVICE_DRAFT_EMITTED, 1, 0,
		  STEP_ROW_LOGITS | STEP_ROW_PROPOSED },
		{ 0, 1, 1, 0 }
	};
	struct step_row_result results[ROW_CAPACITY];
	expect_status(step_run_entry(driver, rows, ROW_CAPACITY, results),
		      STATUS_OK, "the prepared draft graph runs");
	struct step_round round = { 0 };
	expect_status(step_round_take(driver, &round), STATUS_OK,
		      "the prepared draft graph's round returned");
	expect(round.session_slot == 0 &&
		       round.emitted_count == DEVICE_DRAFT_EMITTED &&
		       round.anchor_token == results[1].token_id &&
		       round.proposal_count == MODEL_DRAFT_BLOCK,
	       "target results and the draft round survive replay together");
	const uint32_t replays = test_graph_replays;
	expect(replays == LAUNCH_RUNS + 1u,
	       "the target's runs and one distinct draft graph replay");
	expect_status(step_history_truncate(driver, 0, DEVICE_DRAFT_EMITTED),
		      STATUS_OK,
		      "the graph round's rejected suffix is truncated");
	const struct step_row_record ordinary = { DEVICE_DRAFT_EMITTED, 1, 0,
						  STEP_ROW_LOGITS };
	expect_status(step_run_entry(driver, &ordinary, 1, results), STATUS_OK,
		      "an ordinary step replays its target graphs alone");
	expect(test_graph_replays == replays + LAUNCH_RUNS,
	       "the ordinary step skips its draft graph");
	expect_status(step_round_take(driver, &round), STATUS_OK,
		      "the ordinary replay's empty round returned");
	expect(round.session_slot == STEP_NO_SESSION &&
		       round.proposal_count == 0,
	       "the ordinary graph replay exposes no stale draft record");
	step_driver_destroy(driver);
	test_graph_close();
}

static void test_graph_equivalence(void)
{
	static const uint32_t room[] = { 0, BACKEND_MAXIMUM_GRAPHS, 1, 0 };
	static const uint32_t expected[] = { 0,
					     TEST_GRAPH_REPLAYED_STEPS *
						     LAUNCH_RUNS,
					     TEST_GRAPH_PARTIAL_RUNS, 0 };
	for (uint32_t pass = 0; pass < sizeof(room) / sizeof(room[0]); pass++) {
		test_graph_reset(room[pass]);
		test_graph_save_eager = pass == 0;
		test_step_use_graphs = pass != 0;
		test_step_observe = test_graph_observe;
		test_graph_expected_replays = expected[pass];
		test_steps();
		for (uint32_t card = 0; card < CARDS; card++) {
			expect(test_graph_capture_calls[card] <=
				       room[pass] + 1u,
			       "first BUSY prevents further attempts on that card, including subsequent programs");
		}
	}
	test_step_use_graphs = false;
	test_step_observe = NULL;
}

#define INTERLEAVED_REPLAYS \
	24u /* the plan's runs: a front step's two (cards 0 and 1), another's six */

/* The interleaved model (step_interleaved.h), a card visited twice a step, eager and then with every run a graph: the
 * same complete work records and arena bytes. */
static void test_graph_interleaved(void)
{
	for (uint32_t pass = 0; pass <= 1u; pass++) {
		test_graph_reset(BACKEND_MAXIMUM_GRAPHS);
		test_graph_save_eager = pass == 0;
		test_step_use_graphs = pass != 0;
		test_step_observe = test_graph_observe;
		test_graph_expected_replays = pass == 0 ? 0 :
							  INTERLEAVED_REPLAYS;
		interleaved_eager_steps();
	}
	test_step_use_graphs = false;
	test_step_observe = NULL;
}

static struct step_driver *test_graph_driver(struct fixture *fixture)
{
	fixture_build(fixture, cards_open());
	return driver_of(fixture, PROGRAMS);
}

static void test_graph_simple_step(struct step_driver *driver,
				   uint32_t position)
{
	const struct step_row_record row = { (int32_t)position, 1, 0,
					     STEP_ROW_LOGITS };
	step_expecting(driver, &row, 1, 1);
}

static void test_graph_refusals(void)
{
	struct fixture fixture;
	expect_status(step_graphs_prepare(NULL), STATUS_INVALID_ARGUMENT,
		      "NULL preparation");
	for (uint32_t missing = 0; missing < PROGRAMS; missing++) {
		test_graph_reset(BACKEND_MAXIMUM_GRAPHS);
		struct step_driver *driver = NULL;
		fixture_build(&fixture, cards_open());
		struct backend_operations backend = checked_backend;
		backend.graph_capture = missing == 0 ? NULL :
						       backend.graph_capture;
		backend.graph_run = missing != 0 ? NULL : backend.graph_run;
		expect_status(step_driver_create(&backend, cards,
						 fixture.pointers, PROGRAMS,
						 engram, WAIT_LIMIT, &driver),
			      STATUS_OK, "eager backend still admitted");
		expect_status(step_graphs_prepare(driver), STATUS_UNSUPPORTED,
			      "missing graph entry");
		expect_status(step_graphs_prepare(driver),
			      STATUS_INVALID_ARGUMENT, "attempt consumed");
		test_graph_simple_step(driver, 0);
		step_driver_destroy(driver);
		test_graph_close();
	}
	test_graph_reset(BACKEND_MAXIMUM_GRAPHS);
	struct step_driver *driver = test_graph_driver(&fixture);
	test_graph_simple_step(driver, 0);
	expect_status(step_graphs_prepare(driver), STATUS_INVALID_ARGUMENT,
		      "startup boundary enforced");
	step_driver_destroy(driver);
	test_graph_close();
	test_graph_reset(BACKEND_MAXIMUM_GRAPHS);
	driver = test_graph_driver(&fixture);
	uint32_t untouched = UINT32_MAX;
	const uint64_t before = test_host_memory_backend_work_count();
	struct kernel_launch invalid = fixture.launches[0][0][0];
	invalid.grid[0] = 0;
	expect_status(
		test_graph_capture(cards[0], &invalid, 1, &untouched),
		STATUS_INVALID_ARGUMENT,
		"invalid capture table does not execute an earlier entry");
	expect_status(test_graph_run(cards[0], 0), STATUS_INVALID_ARGUMENT,
		      "uncaptured graph refused");
	expect(untouched == UINT32_MAX &&
		       test_host_memory_backend_work_count() == before,
	       "invalid graph arguments leave scalar output and work untouched");
	step_driver_destroy(driver);
	fixture.launches[0][0][0].grid[0] = 0;
	driver = driver_of(&fixture, PROGRAMS);
	expect_status(step_graphs_prepare(driver), STATUS_INVALID_ARGUMENT,
		      "invalid launch table is refused by capture");
	expect(test_host_memory_backend_work_count() == before,
	       "invalid preparation runs no model work");
	step_driver_destroy(driver);
	test_graph_close();
#if CHECKING_HOOKS_ENABLED
	test_graph_reset(BACKEND_MAXIMUM_GRAPHS);
	driver = test_graph_driver(&fixture);
	expect(fault_arm("step_graph_memory", 1),
	       "graph allocation fault armed");
	expect_status(step_graphs_prepare(driver), STATUS_OUT_OF_MEMORY,
		      "startup allocation fails");
	expect(fault_disarm() == 1, "allocation fault consumed");
	test_graph_simple_step(driver, 0);
	step_driver_destroy(driver);
	test_graph_close();
#endif
}

static void test_graph_partial_refusals(void)
{
	static const result_status refusals[] = { STATUS_INVALID_ARGUMENT,
						  STATUS_UNSUPPORTED,
						  STATUS_OUT_OF_MEMORY };
	struct fixture fixture;
	for (uint32_t refusal = 0;
	     refusal < sizeof(refusals) / sizeof(refusals[0]); refusal++) {
		test_graph_reset(BACKEND_MAXIMUM_GRAPHS);
		struct step_driver *const driver = test_graph_driver(&fixture);
		test_graph_failure_card = 1;
		test_graph_capture_answer = refusals[refusal];
		expect_status(step_graphs_prepare(driver), refusals[refusal],
			      "partial preparation not published");
		expect(!test_graph_pending[0] && !test_graph_pending[1],
		       "earlier graph uploads drained after refusal");
		const uint32_t owned =
			test_graph_owned[0] + test_graph_owned[1];
		test_graph_simple_step(driver, 0);
		expect(test_graph_replays == 0,
		       "refused preparation leaves every run eager");
		expect_status(step_graphs_prepare(driver),
			      STATUS_INVALID_ARGUMENT,
			      "no retry consumes slots");
		step_driver_destroy(driver);
		expect(owned != 0 && owned == test_graph_owned[0] +
						      test_graph_owned[1],
		       "abandoned graphs remain card-owned after driver destruction");
		test_graph_close();
	}
}

static void test_graph_repeated_drivers(void)
{
	struct fixture fixture;
	test_graph_reset(BACKEND_MAXIMUM_GRAPHS);
	fixture_build(&fixture, cards_open());
	for (uint32_t creation = 0; creation <= BACKEND_MAXIMUM_GRAPHS;
	     creation++) {
		struct step_driver *const driver =
			driver_of(&fixture, PROGRAMS);
		expect_status(step_graphs_prepare(driver), STATUS_OK,
			      "same-card repeated creation is bounded");
		step_driver_destroy(driver);
	}
	for (uint32_t card = 0; card < CARDS; card++) {
		expect(test_graph_counts[card] == BACKEND_MAXIMUM_GRAPHS &&
			       test_graph_owned[card] == BACKEND_MAXIMUM_GRAPHS,
		       "driver destruction never releases a graph slot or exceeds card capacity");
	}
	struct step_driver *const eager = driver_of(&fixture, PROGRAMS);
	expect_status(step_graphs_prepare(eager), STATUS_OK,
		      "exhausted cards retain eager service");
	const uint32_t replays_before = test_graph_replays;
	test_graph_simple_step(eager, 0);
	expect(test_graph_replays == replays_before,
	       "fully exhausted new driver submits no graph");
	step_driver_destroy(eager);
	test_graph_close();
}

/* Tiny, semantically unused copies still split coalesced runs. There are more runs on card 1 than graph slots;
 * the first fixed-capacity prefix is prepared, the remainder is eager, without rewriting or dropping a hand-off. */
static void test_graph_many_splits(void)
{
	struct fixture fixture;
	struct kernel_launch launches[TEST_GRAPH_SPLIT_RUNS];
	struct segment_record segments[TEST_GRAPH_SPLIT_SEGMENTS];
	struct hand_off_record hand_offs[TEST_GRAPH_SPLIT_SEGMENTS];
	for (uint32_t pass = 0; pass < PROGRAMS; pass++) {
		test_graph_reset(BACKEND_MAXIMUM_GRAPHS);
		fixture_build(&fixture, cards_open());
		segments[0] = fixture.segments[0][SEGMENT_EMBED];
		hand_offs[0] = fixture.hand_offs[0][HAND_OFF_TO_SECOND];
		for (uint32_t run = 0; run < TEST_GRAPH_SPLIT_RUNS; run++) {
			launches[run] = fixture.launches[0][1][0];
			segments[run + 1u] = (struct segment_record){
				.card = 1,
				.layer = SEGMENT_NO_LAYER,
				.first_launch = run,
				.launch_count = 1
			};
			hand_offs[run + 1u] = (struct hand_off_record){
				.source_offset = STREAMS_OFFSET,
				.destination_offset = FAR_OFFSET,
				.byte_count = sizeof(uint64_t),
				.from_card = 1,
				.to_card = 0,
				.after_segment = run + 1u
			};
		}
		segments[TEST_GRAPH_SPLIT_SEGMENTS - 1u] =
			fixture.segments[0][SEGMENT_SAMPLE];
		segments[TEST_GRAPH_SPLIT_SEGMENTS - 1u].first_launch = 0;
		segments[TEST_GRAPH_SPLIT_SEGMENTS - 1u].launch_count =
			TEST_GRAPH_TERMINAL_SEGMENTS;
		hand_offs[TEST_GRAPH_SPLIT_SEGMENTS - 1u] =
			fixture.hand_offs[0][HAND_OFF_TO_LAST];
		hand_offs[TEST_GRAPH_SPLIT_SEGMENTS - 1u].after_segment =
			TEST_GRAPH_SPLIT_RUNS;
		fixture.program_cards[0][1].launches = launches;
		fixture.program_cards[0][1].launch_count =
			TEST_GRAPH_SPLIT_RUNS;
		fixture.programs[0].segments = segments;
		fixture.programs[0].segment_count = TEST_GRAPH_SPLIT_SEGMENTS;
		fixture.programs[0].draft_segment_first =
			TEST_GRAPH_SPLIT_SEGMENTS;
		fixture.programs[0].hand_offs = hand_offs;
		fixture.programs[0].hand_off_count = TEST_GRAPH_SPLIT_SEGMENTS;
		struct step_driver *const driver = driver_of(&fixture, 1);
		if (pass != 0) {
			expect_status(step_graphs_prepare(driver), STATUS_OK,
				      "many split runs prepare bounded prefix");
		}
		const struct step_row_record row = { 0, 1, 0, STEP_ROW_LOGITS };
		struct step_row_result result = { 0 };
		expect_status(step_run_entry(driver, &row, 1, &result),
			      STATUS_OK, "many split runs complete");
		uint64_t stream = stream_after_layer(EMBED_MULTIPLIER + 1u, 0);
		for (uint32_t run = 0; run < TEST_GRAPH_SPLIT_RUNS; run++) {
			stream = stream_after_layer(stream, 1);
		}
		stream = stream_after_layer(stream, LAST_CARD);
		expect(result.token_id == (int32_t)(stream % VOCABULARY) &&
			       result.logit == (float)(stream % LOGIT_MODULUS),
		       "capacity fallback does not skip a launch or small splitting copy");
		if (pass != 0) {
			expect(test_graph_counts[1] == BACKEND_MAXIMUM_GRAPHS &&
				       test_graph_capture_calls[1] ==
					       BACKEND_MAXIMUM_GRAPHS + 1u,
			       "metadata splits exhaust precisely the fixed card capacity, then stop capture attempts");
		}
		test_graph_save_eager = pass == 0;
		test_graph_expected_replays =
			pass == 0 ? 0 :
				    BACKEND_MAXIMUM_GRAPHS +
					    TEST_GRAPH_TERMINAL_SEGMENTS;
		test_graph_observe();
		step_driver_destroy(driver);
		test_graph_close();
	}
}

static void test_graph_sticky(struct step_driver *driver, result_status failure)
{
	const struct step_row_record row = { TEST_GRAPH_NEXT_POSITION, 1, 0,
					     STEP_ROW_LOGITS };
	struct step_row_result result = { 0 };
	const struct step_session settings = { 0, 0.0f, 1.0f };
	const uint64_t before = test_host_memory_backend_work_count();
	expect_status(step_run_entry(driver, &row, 1, &result), failure,
		      "failed driver refuses further work");
	expect_status(step_session_set(driver, 0, &settings), failure,
		      "failed driver refuses settings");
	expect_status(step_graphs_prepare(driver), failure,
		      "failed driver refuses capture retries");
	expect(test_host_memory_backend_work_count() == before,
	       "failure latch prevents further submissions");
}

static void test_graph_failures(void)
{
	struct fixture fixture;
	for (uint32_t scenario = 0; scenario < CARDS; scenario++) {
		test_graph_reset(BACKEND_MAXIMUM_GRAPHS);
		struct step_driver *const driver = test_graph_driver(&fixture);
		test_graph_failure_card = scenario == 0 ? 1u : CARDS;
		test_graph_capture_answer = STATUS_DEVICE_ERROR;
		test_graph_wait_answers[0] = scenario == 1 ? STATUS_TIMED_OUT :
							     STATUS_OK;
		test_graph_invalid_handle_card = scenario == LAST_CARD ? 1u :
									 CARDS;
		const result_status expected =
			scenario == 1 ? STATUS_TIMED_OUT : STATUS_DEVICE_ERROR;
		expect_status(
			step_graphs_prepare(driver), expected,
			"capture/upload, bounded upload completion, and invalid returned handle failclosed");
		expect(test_graph_wait_calls[0] != 0,
		       "an earlier card is drained after a later capture failure");
		test_graph_sticky(driver, expected);
		step_driver_destroy(driver);
		expect((scenario == LAST_CARD) ==
			       (test_graph_retired_count == 0),
		       "unknown completion retires staging; invalid handle with proved completion can free it");
		test_graph_close();
	}
	test_graph_reset(BACKEND_MAXIMUM_GRAPHS);
	struct step_driver *const driver = test_graph_driver(&fixture);
	expect_status(step_graphs_prepare(driver), STATUS_OK,
		      "replay failure fixture prepared");
	test_graph_simple_step(driver, 0);
#if CHECKING_HOOKS_ENABLED
	uint32_t history_before[CONTEXT] = { 0 },
		 history_after[CONTEXT] = { 0 }, count_before = 0,
		 count_after = 0;
	expect_status(step_committed_history_read(driver, 0, history_before,
						  CONTEXT, &count_before),
		      STATUS_OK, "committed history witnessed after success");
#endif
	test_graph_replay_failure = true;
	const struct step_row_record row = { TEST_GRAPH_NEXT_POSITION,
					     POSITION_TWO, 0, STEP_ROW_LOGITS };
	struct step_row_result result = { 0 };
	capture_begin();
	expect_status(step_run_entry(driver, &row, 1, &result),
		      STATUS_DEVICE_ERROR, "replay submission fails");
	const uint32_t logged = capture_end();
	expect(logged == TEST_LOG_EXPECTED_LINES(1u),
	       "required failure status remains and optional release logs vanish");
#if CHECKING_HOOKS_ENABLED
	expect_status(
		step_committed_history_read(driver, 0, history_after, CONTEXT,
					    &count_after),
		STATUS_OK,
		"poisoned driver's committed state remains observable only in checking builds");
	expect(count_before == 1 && count_before == count_after &&
		       memcmp(history_before, history_after,
			      sizeof(history_before)) == 0,
	       "failed graph step commits neither input token nor token count");
#endif
	test_graph_sticky(driver, STATUS_DEVICE_ERROR);
	step_driver_destroy(driver);
	expect(test_graph_retired_count != 0 && test_graph_pending[1],
	       "failed replay cannot free staging or claim completion");
	test_graph_close();
}

#include "step_graph_failure_cases.h"

int main(void)
{
	(void)setvbuf(stdout, NULL, _IOLBF, 0);
	checked_backend = test_host_memory_backend;
	checked_backend.download_begin = test_graph_download;
	checked_backend.upload_begin = test_graph_upload;
	checked_backend.upload_wait = test_graph_upload_wait;
	checked_backend.launch_run = checked_launch_run;
	checked_backend.launch_wait = test_graph_launch_wait;
	checked_backend.graph_capture = test_graph_capture;
	checked_backend.graph_run = test_graph_run;
	checked_backend.staging_retire = test_graph_retire;
	capture_descriptor = mkstemp(capture_path);
	error_stream_descriptor = dup(STDERR_FILENO);
	if (capture_descriptor < 0 || error_stream_descriptor < 0 ||
	    unlink(capture_path) != 0) {
		return EXIT_FAILURE;
	}
	expect_status(test_engram_pack_open(&engram_pack, &engram), STATUS_OK,
		      "n-gram tables open");
	test_graph_equivalence();
	test_graph_interleaved();
	test_graph_refusals();
	test_graph_partial_refusals();
	test_graph_repeated_drivers();
	test_graph_many_splits();
	test_graph_failures();
	test_graph_api_failure_cases();
	test_graph_device_round();
	expect(test_host_memory_backend_open_card_count() == 0 &&
		       test_host_memory_backend_staging_block_count() == 0,
	       "all real fake-backend allocations retire after their owning cards close");
	engram_close(engram);
	free(engram_pack);
	(void)close(capture_descriptor);
	(void)close(error_stream_descriptor);
	(void)printf("step_graphs_test: %s\n",
		     failures == 0 ? "passed" : "FAILED");
	return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
