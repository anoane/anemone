/* SPDX-License-Identifier: AGPL-3.0-or-later */
#define TEST_STEP_FIXTURE_IMPLEMENTATION
#include "step_fixture.h"
#include "step_programs.h"
#include "step_interleaved.h"
#include "step_runs.h"
#include "step_draft_cases.h"

#include <time.h>

#include "anemone/model_layout.h"

#define PIPELINE_DEPTH 3u
#define PIPELINE_ROWS 2u
#define PIPELINE_TOKEN_STRIDE 7u
#define PIPELINE_REFUSALS 3u
#define PIPELINE_LATE_REFUSALS 2u
#define SNAPSHOT_ROWS 4u
#define SNAPSHOT_CAPACITY 4096u
#define SNAPSHOT_REFUSALS 3u
/* Pinned merge5r fixture identities: reference remains compatible; the old trunk-only relaxed state does not. */
#define BASE_REFERENCE_IDENTITY 11630104210251508399ull
#define BASE_RELAXED_TRUNK_IDENTITY 10051950646683646284ull
#define WIDE_ROWS \
	6u /* the wide program's: above the thin programs' 4, within the plan's 8 */
#define VIEW_OFFSET 40960u /* its view on cards 0 and 1: a block and 8 rows */
#define WIDE_FRONT_HAND_OFFS \
	2u /* the fixture's behind its front: to card 1, the pool */
#define WIDE_REFUSALS 2u /* steps no program runs */
#define WIDE_CREATE_REFUSALS \
	2u /* a card without its view, a view of fewer rows */
#define FULL_REFUSALS \
	2u /* a full step with a row of logits, front rows no program runs */
#define ROUND_STEPS 3u
#define ROUND_CUT \
	2u /* session 0's tokens after its first round: the anchor and one proposal accepted */
#define ROUND_SECOND_CUT 3u /* and after its second: the anchor alone */
#define ROUND_REFUSALS \
	4u /* a cut of no driver, of no slot, of more than the session holds; a row past the cut */
#define FULL_HAND_OFFS \
	2u /* of the full program: the thin one's to card 1 and the pool, not its two of rows behind its last segment */
#define INTERLEAVED_DEPTH 4u
#define GATED_STEP \
	3u /* the plan's step (its sequence) whose S_POOL card 1 holds back until card 0 embedded a later one */
#define GATE_LIMIT_NANOSECONDS 2000000000ull
#define SHORT_GATE_NANOSECONDS \
	200000000ull /* a gate that runs out: held shut by a lease, or outlasting a short wait limit */
#define SHORT_PIPELINE_WAIT \
	20000000ull /* the wait limit of a pipeline whose card 1 a short gate holds */
#define GATE_POLL_NANOSECONDS 100000l
#define INTERLEAVED_FAULTING_LAUNCH \
	3u /* of card 0: its second step's embedding */
#define INTERLEAVED_SYNC_STEP \
	2u /* the plan's first step of other rows: it syncs session 0's mirror */
/* create refuses a changed copy of the fixture with this status, leaves its output and the cards alone. */
static void expect_refused(const struct fixture *fixture,
			   result_status expected, const char *what)
{
	struct step_driver *driver =
		(void *)&failures; /* an address no create gives */
	const uint64_t work_before = test_host_memory_backend_work_count();
	const uint32_t staging_before =
		test_host_memory_backend_staging_block_count();
	capture_begin();
	expect_status(step_driver_create(&checked_backend, cards,
					 fixture->pointers, PROGRAMS, engram,
					 WAIT_LIMIT, &driver),
		      expected, what);
	expect(capture_end() == TEST_LOG_EXPECTED_LINES(1u),
	       "diagnostic error count or release silence");
	expect(driver == (void *)&failures &&
		       test_host_memory_backend_work_count() == work_before &&
		       test_host_memory_backend_staging_block_count() ==
			       staging_before,
	       "nothing is given, queued or kept");
}

/* create takes a changed copy of the fixture; destroy gives back what it took, and nothing reached a card. */
static void expect_accepted(const struct fixture *fixture, const char *what)
{
	struct step_driver *driver = NULL;
	const uint64_t work_before = test_host_memory_backend_work_count();
	const uint32_t staging_before =
		test_host_memory_backend_staging_block_count();
	expect_status(step_driver_create(&checked_backend, cards,
					 fixture->pointers, PROGRAMS, engram,
					 WAIT_LIMIT, &driver),
		      STATUS_OK, what);
	step_driver_destroy(driver);
	expect(test_host_memory_backend_work_count() == work_before &&
		       test_host_memory_backend_staging_block_count() ==
			       staging_before,
	       "nothing is queued or kept");
}

/* Where an area create measures lies in a program of the fixture, and its bytes at the largest: the plan's row
 * capacity of rows (not the larger program's ROW_CAPACITY), or every session's settings or entries. */
static uint64_t *area_of(struct fixture *fixture, uint32_t program,
			 enum area_index area, uint64_t *byte_count_out)
{
	static const uint64_t byte_counts[AREAS] = {
		STEP_BLOCK_BYTES_VALUE +
			(uint64_t)LIMIT_ROWS * STEP_ROW_BYTES_VALUE,
		(uint64_t)LIMIT_ROWS * TABLES * ENGRAM_ROW_BYTES,
		STEP_RESULT_BYTES +
			(uint64_t)LIMIT_ROWS * STEP_ROW_RESULT_BYTES,
		(uint64_t)SESSIONS * STEP_SESSION_BYTES,
		LIMIT_ROWS * sizeof(uint64_t),
		SESSIONS * POOL_STRIDE
	};
	struct program_card_record *const areas =
		fixture->program_cards[program];
	struct hand_off_record *const hand_offs = fixture->hand_offs[program];
	uint64_t *const offsets[AREAS] = {
		&areas[1].step_offset,
		&areas[0].engram_rows_offset,
		&areas[LAST_CARD].result_offset,
		&areas[LAST_CARD].session_offset,
		&hand_offs[HAND_OFF_TO_SECOND].source_offset,
		&hand_offs[HAND_OFF_POOL].destination_offset
	};
	*byte_count_out = byte_counts[area];
	return offsets[area];
}

static void test_create_refusals(void)
{
	struct fixture base;
	struct fixture changed;
	struct step_driver *driver = NULL;
	context_text = "create's refusals";
	fixture_build(&base, cards_open());
#define CHANGED(change, expected, what)                   \
	do {                                              \
		changed = base;                           \
		fixture_link(&changed);                   \
		change;                                   \
		expect_refused(&changed, expected, what); \
	} while (false)
	CHANGED(changed.programs[1].row_count = 1, STATUS_INVALID_ARGUMENT,
		"two programs of one row count");
	CHANGED(changed.programs[1].kind = PROGRAM_WIDE,
		STATUS_INVALID_ARGUMENT,
		"a wide program that is not the thin first program's front");
	CHANGED(changed.programs[0].kind = PROGRAM_CONE, STATUS_UNSUPPORTED,
		"a cone program");
	CHANGED(changed.programs[1].kind = PROGRAM_FULL,
		STATUS_INVALID_ARGUMENT,
		"a full program that is not all of the thin first program but its head");
	CHANGED(changed.programs[0].limits.row_capacity =
			changed.programs[1].limits.row_capacity =
				changed.programs[1].row_count = LIMIT_ROWS + 1u,
		STATUS_INVALID_ARGUMENT, "9 thin rows");
	CHANGED(changed.programs[1].kind = PROGRAM_KIND_COUNT,
		STATUS_INVALID_ARGUMENT, "no kind");
	CHANGED(changed.program_cards[1][1].view_count = 1,
		STATUS_INVALID_ARGUMENT, "a view in a thin program");
	CHANGED(changed.programs[1].row_count = LIMIT_ROWS + 1u,
		STATUS_INVALID_ARGUMENT, "rows over capacity");
	CHANGED(changed.programs[1].limits.session_capacity = 1,
		STATUS_INVALID_ARGUMENT, "limits differ");
	CHANGED(changed.programs[0].limits.context_token_limit = 0;
		changed.programs[1].limits.context_token_limit = 0,
		STATUS_INVALID_ARGUMENT, "no context");
	CHANGED(changed.programs[1].card_count = LAST_CARD,
		STATUS_INVALID_ARGUMENT, "card counts differ");
	CHANGED(changed.programs[0].segments = NULL, STATUS_INVALID_ARGUMENT,
		"no segments");
	CHANGED(changed.segments[0][SEGMENT_SAMPLE].card = CARDS,
		STATUS_INVALID_ARGUMENT, "a card not there");
	CHANGED(changed.segments[1][SEGMENT_SECOND_LAYER].first_launch = 0,
		STATUS_INVALID_ARGUMENT, "launches out of order");
	CHANGED(changed.segments[1][SEGMENT_SECOND_LAYER].launch_count = 0,
		STATUS_INVALID_ARGUMENT, "a segment of no launch");
	CHANGED(changed.segments[1][SEGMENT_SECOND_LAYER].stage = 0,
		STATUS_INVALID_ARGUMENT,
		"a thin program's segments that are not the first's");
	CHANGED(changed.program_cards[1][1].launch_count = POSITION_FOUR,
		STATUS_INVALID_ARGUMENT, "a launch no segment runs");
	CHANGED(changed.segments[0][SEGMENT_EMBED].pause =
			SEGMENT_PAUSE_PAIR_OUTPUTS,
		STATUS_UNSUPPORTED, "a pause for pair outputs");
	CHANGED(changed.segments[0][SEGMENT_EMBED].pause = BAD_PAUSE,
		STATUS_INVALID_ARGUMENT, "no such pause");
	CHANGED(changed.program_cards[0][0].step_offset = PROGRAM_NOWHERE;
		changed.program_cards[1][0].step_offset = PROGRAM_NOWHERE,
		STATUS_INVALID_ARGUMENT,
		"a card that runs launches without a step area");
	CHANGED(changed.program_cards[1][1].step_offset = SESSION_OFFSET,
		STATUS_INVALID_ARGUMENT, "card areas differ between programs");
	CHANGED(
		for (uint32_t program = 0; program < PROGRAMS; program++) {
			changed.program_cards[program][1].step_offset =
				ARENA_BYTES - STEP_BLOCK_BYTES_VALUE;
		},
		STATUS_INVALID_ARGUMENT,
		"a step area that ends behind the arena");
	CHANGED(
		for (uint32_t program = 0; program < PROGRAMS; program++) {
			changed.program_cards[program][1].engram_tables =
				1u << TABLES;
			changed.program_cards[program][1].engram_rows_offset =
				ENGRAM_OFFSET;
		},
		STATUS_INVALID_ARGUMENT, "a table the n-gram object lacks");
	CHANGED(
		for (uint32_t program = 0; program < PROGRAMS; program++) {
			changed.program_cards[program][0].engram_rows_offset =
				ARENA_BYTES - ENGRAM_ROW_BYTES;
		},
		STATUS_INVALID_ARGUMENT, "n-gram rows behind the arena");
	CHANGED(
		for (uint32_t program = 0; program < PROGRAMS; program++) {
			changed.program_cards[program][LAST_CARD]
				.session_offset = PROGRAM_NOWHERE;
		},
		STATUS_INVALID_ARGUMENT,
		"no session settings on the card that samples");
	CHANGED(
		for (uint32_t program = 0; program < PROGRAMS; program++) {
			changed.program_cards[program][LAST_CARD].result_offset =
				TOO_FAR;
		},
		STATUS_INVALID_ARGUMENT, "a result area outside the arena");
	CHANGED(changed.hand_offs[1][HAND_OFF_POOL].byte_count = 1,
		STATUS_INVALID_ARGUMENT, "hand-offs differ");
#define HAND_OFF_CHANGED(change, what)                                      \
	CHANGED(                                                            \
		for (uint32_t program = 0; program < PROGRAMS; program++) { \
			struct hand_off_record *const hand_off =            \
				&changed.hand_offs[program][HAND_OFF_POOL]; \
			change;                                             \
		},                                                          \
		STATUS_INVALID_ARGUMENT, what)
	HAND_OFF_CHANGED(hand_off->after_segment = SEGMENTS,
			 "a hand-off after no segment");
	CHANGED(
		for (uint32_t program = 0; program < PROGRAMS; program++) {
			changed.hand_offs[program][HAND_OFF_BACK].after_segment =
				SEGMENT_SECOND_LAYER;
		},
		STATUS_INVALID_ARGUMENT,
		"hand-offs out of the segments' order");
	HAND_OFF_CHANGED(hand_off->from_card = 0,
			 "a hand-off that leaves no card of its segment");
	HAND_OFF_CHANGED(hand_off->to_card = 1, "a hand-off to its own card");
	HAND_OFF_CHANGED(hand_off->to_card = CARDS, "a hand-off to no card");
	HAND_OFF_CHANGED(hand_off->byte_count = 0, "a hand-off of no bytes");
	HAND_OFF_CHANGED(
		hand_off->view = 1,
		"a hand-off of a view, which no card of a thin program has");
	HAND_OFF_CHANGED(hand_off->session_stride = ENTRY_BYTES,
			 "entries beyond the stride");
	HAND_OFF_CHANGED(hand_off->destination_offset =
				 ARENA_BYTES - POOL_STRIDE,
			 "entries behind the arena");
	HAND_OFF_CHANGED(hand_off->entry_ratio = 0,
			 "a stride with no entry ratio");
	CHANGED(
		for (uint32_t program = 0; program < PROGRAMS; program++) {
			changed.program_cards[program][LAST_CARD].launches =
				NULL;
		},
		STATUS_INVALID_ARGUMENT, "a card's launches missing");
	CHANGED(changed.programs[0].result_card = CARDS;
		changed.programs[1].result_card = CARDS,
		STATUS_INVALID_ARGUMENT,
		"a card that samples that is not there");
	expect(test_log_is_expected(captured_text, "out of range", false),
	       "refused for its number, not for what lies past the cards");
	/* card 0 not there and running nothing; card 1 embeds too, and hands off to it */
	struct device_card *const first_card = cards[0];
	CHANGED(
		cards[0] = NULL;
		for (uint32_t program = 0; program < PROGRAMS; program++) {
			struct program_card_record *const card_areas =
				changed.program_cards[program];
			card_areas[0].launch_count = 0;
			card_areas[0].engram_tables = 0;
			card_areas[0].step_offset = PROGRAM_NOWHERE;
			card_areas[1].launch_count = LAUNCHES_PER_CARD;
			changed.segments[program][SEGMENT_EMBED] =
				((struct segment_record){ .card = 1,
							  .launch_count = 1 });
			for (uint32_t segment = SEGMENT_FIRST_LAYER;
			     segment <= SEGMENT_THIRD_LAYER; segment++) {
				changed.segments[program][segment]
					.first_launch++;
			}
			changed.hand_offs[program][HAND_OFF_TO_SECOND]
				.from_card = 1;
			changed.hand_offs[program][HAND_OFF_TO_SECOND].to_card =
				0;
		},
		STATUS_INVALID_ARGUMENT,
		"a hand-off to a card that is not there");
	cards[0] = first_card;
#undef HAND_OFF_CHANGED
#undef CHANGED
	struct device_card *const second_card = cards[1];
	cards[1] = NULL;
	expect_refused(&base, STATUS_INVALID_ARGUMENT,
		       "a card that runs launches is missing");
	cards[1] = second_card;
	capture_begin();
	expect_status(step_driver_create(NULL, cards, base.pointers, PROGRAMS,
					 engram, WAIT_LIMIT, &driver),
		      STATUS_INVALID_ARGUMENT, "no backend");
	expect_status(step_driver_create(&checked_backend, NULL, base.pointers,
					 PROGRAMS, engram, WAIT_LIMIT, &driver),
		      STATUS_INVALID_ARGUMENT, "no cards");
	expect_status(step_driver_create(&checked_backend, cards, base.pointers,
					 0, engram, WAIT_LIMIT, &driver),
		      STATUS_INVALID_ARGUMENT, "no program");
	expect_status(step_driver_create(&checked_backend, cards, base.pointers,
					 PROGRAMS, engram, 0, &driver),
		      STATUS_INVALID_ARGUMENT, "no wait");
	expect_status(step_driver_create(&checked_backend, cards, base.pointers,
					 PROGRAMS, NULL, WAIT_LIMIT, &driver),
		      STATUS_INVALID_ARGUMENT,
		      "n-gram tables and no n-gram object");
	expect_status(step_driver_create(&checked_backend, cards, base.pointers,
					 PROGRAMS, engram, WAIT_LIMIT, NULL),
		      STATUS_INVALID_ARGUMENT, "nowhere to answer");
	struct backend_operations lacking = checked_backend;
	lacking.download_begin = NULL;
	expect_status(step_driver_create(&lacking, cards, base.pointers,
					 PROGRAMS, engram, WAIT_LIMIT, &driver),
		      STATUS_INVALID_ARGUMENT, "a backend without downloads");
	expect(capture_end() == TEST_LOG_EXPECTED_LINES(REFUSALS_OF_ARGUMENTS),
	       "diagnostic error count or release silence");
	expect(driver == NULL, "nothing is given");
	test_host_memory_backend_fail_staging_allocation(1);
	expect_status(step_driver_create(&checked_backend, cards, base.pointers,
					 PROGRAMS, engram, WAIT_LIMIT, &driver),
		      STATUS_OUT_OF_MEMORY, "no staging memory");
#if CHECKING_HOOKS_ENABLED
	for (uint64_t reach = 1; reach <= FAULT_REACHES; reach++) {
		expect(fault_arm("step_driver_memory", reach),
		       "the fault is armed");
		expect_status(step_driver_create(&checked_backend, cards,
						 base.pointers, PROGRAMS,
						 engram, WAIT_LIMIT, &driver),
			      STATUS_OUT_OF_MEMORY, "no host memory");
		expect(fault_disarm() == reach,
		       "the fault point was reached that often");
	}
#endif
	expect(driver == NULL &&
		       test_host_memory_backend_staging_block_count() == 0,
	       "nothing is given or kept");
	/* each area create measures is taken where it ends at its arena's end, and refused one byte further */
	static const char *const area_names[AREAS] = {
		"the step area",
		"the n-gram rows",
		"the result area",
		"the session settings",
		"a hand-off's source rows",
		"a hand-off's destination entries"
	};
	for (uint32_t area = 0; area < AREAS; area++) {
		context_text = area_names[area];
		for (uint64_t beyond = 0; beyond <= 1; beyond++) {
			changed = base;
			fixture_link(&changed);
			for (uint32_t program = 0; program < PROGRAMS;
			     program++) {
				uint64_t byte_count = 0;
				uint64_t *const offset = area_of(
					&changed, program,
					(enum area_index)area, &byte_count);
				*offset = ARENA_BYTES - byte_count + beyond;
			}
			if (beyond == 0) {
				expect_accepted(
					&changed,
					"an area that ends where its arena ends");
			} else {
				expect_refused(
					&changed, STATUS_INVALID_ARGUMENT,
					"an area one byte past its arena");
			}
		}
	}
	cards_close();
}

static void test_row_refusals(void)
{
	struct fixture fixture;
	struct step_row_result results[ROW_CAPACITY];
	context_text = "a step's refusals";
	fixture_build(&fixture, cards_open());
	struct step_driver *const driver = driver_of(&fixture, PROGRAMS);
	const struct step_row_record first = { 0, 1, 0, STEP_ROW_LOGITS };
	step_expecting(driver, &first, 1, 1);
	const struct step_row_record wrong[] = {
		{ -1, 1, 0, 0 },	   { CONTEXT, 1, 0, 0 },
		{ 1, -1, 0, 0 },	   { 1, VOCABULARY, 0, 0 },
		{ 1, 1, SESSIONS, 0 },	   { 1, 1, 0, BAD_FLAG },
		{ POSITION_TWO, 1, 0, 0 }, { 1, 1, 1, 0 }
	};
	const struct step_row_record wrong_pairs[][WRONG_PAIR_ROWS] = {
		{ { 1, 1, 0, 0 }, { 0, 1, 0, 0 } },
		{ { POSITION_TWO, 1, 0, 0 }, { 1, 1, 0, 0 } },
		{ { 1, 1, 0, 0 }, { 1, 1, 0, 0 } }
	};
	const uint64_t work_before = test_host_memory_backend_work_count();
	capture_begin();
	for (uint32_t index = 0; index < sizeof(wrong) / sizeof(wrong[0]);
	     index++) {
		expect_status(step_run_entry(driver, &wrong[index], 1, results),
			      STATUS_INVALID_ARGUMENT,
			      "a row that breaks a rule");
	}
	for (uint32_t index = 0;
	     index < sizeof(wrong_pairs) / sizeof(wrong_pairs[0]); index++) {
		expect_status(step_run_entry(driver, wrong_pairs[index],
					     WRONG_PAIR_ROWS, results),
			      STATUS_INVALID_ARGUMENT,
			      "two rows that break the order of a session");
	}
	expect_status(step_run_entry(driver, &first, 0, results),
		      STATUS_INVALID_ARGUMENT, "no row");
	expect_status(step_run_entry(driver, wrong, ROW_CAPACITY + 1u, results),
		      STATUS_INVALID_ARGUMENT, "more rows than any program");
	expect_status(step_run_entry(NULL, &first, 1, results),
		      STATUS_INVALID_ARGUMENT, "no driver");
	expect_status(step_run_entry(driver, NULL, 1, results),
		      STATUS_INVALID_ARGUMENT, "no rows");
	expect_status(step_run_entry(driver, &first, 1, NULL),
		      STATUS_INVALID_ARGUMENT, "no results");
	const struct step_session sessions[] = {
		{ 0, -1.0f, 1.0f }, { 0, NAN, 1.0f }, { 0, INFINITY, 1.0f },
		{ 0, 0.0f, 0.0f },  { 0, 0.0f, NAN }, { 0, 0.0f, -1.0f }
	};
	for (uint32_t index = 0; index < sizeof(sessions) / sizeof(sessions[0]);
	     index++) {
		expect_status(step_session_set(driver, 0, &sessions[index]),
			      STATUS_INVALID_ARGUMENT, "settings out of range");
	}
	expect_status(step_session_set(driver, SESSIONS, &sessions[0]),
		      STATUS_INVALID_ARGUMENT, "no such slot");
	expect_status(step_session_set(driver, 0, NULL),
		      STATUS_INVALID_ARGUMENT, "no settings");
	expect(capture_end() ==
		       TEST_LOG_EXPECTED_LINES(
			       sizeof(wrong) / sizeof(wrong[0]) +
			       sizeof(wrong_pairs) / sizeof(wrong_pairs[0]) +
			       sizeof(sessions) / sizeof(sessions[0]) +
			       STEP_REFUSALS_BESIDE_THE_TABLES),
	       "one error line for each refusal");
	expect(test_host_memory_backend_work_count() == work_before,
	       "nothing is queued");
	const struct step_row_record next = { 1, POSITION_TWO, 0,
					      STEP_ROW_LOGITS };
	step_expecting(driver, &next, 1, 1);
	/* session 0 filled to the context limit, which a row at the next position then passes */
	struct step_row_record fill[ROW_CAPACITY];
	for (uint32_t position = POSITION_TWO; position < CONTEXT;) {
		const uint32_t count = CONTEXT - position < ROW_CAPACITY ?
					       CONTEXT - position :
					       ROW_CAPACITY;
		for (uint32_t row = 0; row < count; row++, position++) {
			fill[row] = (struct step_row_record){
				(int32_t)position,
				(int32_t)(position % VOCABULARY), 0,
				row + 1u == count ? STEP_ROW_LOGITS : 0
			};
		}
		step_expecting(driver, fill, count,
			       count == 1 ? 1 : ROW_CAPACITY);
	}
	const struct step_row_record past = { CONTEXT, 1, 0, STEP_ROW_LOGITS };
	capture_begin();
	expect_status(step_run_entry(driver, &past, 1, results),
		      STATUS_INVALID_ARGUMENT, "a row past the context limit");
	expect(capture_end() == TEST_LOG_EXPECTED_LINES(1u),
	       "diagnostic error count or release silence");
	step_driver_destroy(driver);
	cards_close();
}

/* One step that fails with this status; then the driver answers it to everything, and destroy keeps the staging
 * block or frees it. */
static void failing_step(struct step_driver *driver, result_status expected,
			 bool staging_stays, uint32_t flags, const char *what)
{
	struct step_row_result results[ROW_CAPACITY];
	const struct step_row_record row = { 0, 1, 0, flags };
	const struct step_session settings = { 0, 0.0f, 1.0f };
	const uint32_t staging_before =
		test_host_memory_backend_staging_block_count();
	context_text = what;
	capture_begin();
	expect_status(step_run_entry(driver, &row, 1, results), expected,
		      "the step fails");
	expect_status(step_run_entry(driver, &row, 1, results), expected,
		      "and so does the next");
	expect_status(step_session_set(driver, 0, &settings), expected,
		      "and the settings");
	step_driver_destroy(driver);
	const uint32_t error_count = capture_end();
	expect(LOGGING_HOOKS_ENABLED ? error_count >= 1u : error_count == 0u,
	       "diagnostic error count or release silence");
	expect(test_host_memory_backend_staging_block_count() ==
		       staging_before - (staging_stays ? 0 : 1u),
	       staging_stays ? "the staging block stays" :
			       "the staging block is freed");
	cards_close();
}

static void test_failures(void)
{
	/* results that do not answer the step; without n-gram tables the driver knows no vocabulary */
	static const struct broken_answer {
		enum sample_break sample_break;
		uint32_t flags, program_count;
		bool has_engram;
		const char *what;
	} broken_answers[] = {
		{ SAMPLE_OTHER_STEP, STEP_ROW_LOGITS, PROGRAMS, true,
		  "a result of another step" },
		{ SAMPLE_OTHER_ROW_COUNT, STEP_ROW_LOGITS, PROGRAMS, true,
		  "a result of another row count" },
		{ SAMPLE_TOKEN_FLIPPED, STEP_ROW_LOGITS, 1, false,
		  "no token where logits were asked, no n-grams" },
		{ SAMPLE_TOKEN_FLIPPED, 0, PROGRAMS, true,
		  "a token where no logits were asked" },
		{ SAMPLE_TOKEN_BEYOND, STEP_ROW_LOGITS, 1, true,
		  "a token beyond the vocabulary" }
	};
	struct fixture fixture;
	for (uint32_t index = 0;
	     index < sizeof(broken_answers) / sizeof(broken_answers[0]);
	     index++) {
		const struct broken_answer *const answer =
			&broken_answers[index];
		struct step_driver *broken = NULL;
		fixture_build(&fixture, cards_open());
		sample_break = answer->sample_break;
		for (uint32_t card_area = 0;
		     card_area < PROGRAMS * CARDS && !answer->has_engram;
		     card_area++) {
			fixture.program_cards[card_area / CARDS]
					     [card_area % CARDS]
						     .engram_tables = 0;
		}
		expect_status(step_driver_create(
				      &checked_backend, cards, fixture.pointers,
				      answer->program_count,
				      answer->has_engram ? engram : NULL,
				      WAIT_LIMIT, &broken),
			      STATUS_OK, answer->what);
		failing_step(broken, STATUS_DEVICE_ERROR, false, answer->flags,
			     answer->what);
	}
	fixture_build(&fixture, cards_open());
	test_host_memory_backend_set_polls_before_an_upload_finishes(
		LAST_CARD, POLLS_TO_TIME_OUT);
	struct step_driver *driver = NULL;
	expect_status(step_driver_create(&checked_backend, cards,
					 fixture.pointers, PROGRAMS, engram,
					 SHORT_WAIT_LIMIT, &driver),
		      STATUS_OK, "a driver of short waits");
	failing_step(driver, STATUS_TIMED_OUT, false, STEP_ROW_LOGITS,
		     "a wait that runs out");
	/* many hand-offs after one segment: the driver waits before the backend's limit of copies in flight */
	fixture_build(&fixture, cards_open());
	context_text = "many hand-offs";
	for (uint32_t program = 0; program < PROGRAMS; program++) {
		for (uint32_t index = HAND_OFFS; index < MANY_HAND_OFFS;
		     index++) {
			fixture.hand_offs[program][index] =
				fixture.hand_offs[program][HAND_OFF_TO_LAST];
			fixture.hand_offs[program][index].destination_offset =
				FAR_OFFSET + ENTRY_BYTES * ROW_CAPACITY * index;
		}
		fixture.programs[program].hand_off_count = MANY_HAND_OFFS;
	}
	for (uint32_t card = 0; card < CARDS; card++) {
		test_host_memory_backend_set_polls_before_an_upload_finishes(
			card, 1);
	}
	driver = driver_of(&fixture, PROGRAMS);
	const struct step_row_record row = { 0, 1, 0, STEP_ROW_LOGITS };
	step_expecting(driver, &row, 1, 1);
	for (uint32_t index = HAND_OFFS; index < MANY_HAND_OFFS; index++) {
		expect(memcmp(arena_of(LAST_CARD) + FAR_OFFSET +
				      ENTRY_BYTES * ROW_CAPACITY * index,
			      arena_of(1) + STREAMS_OFFSET, ENTRY_BYTES) == 0,
		       "each of the many hand-offs arrived");
	}
	expect(test_host_memory_backend_most_uploads_in_flight(1) <=
		       BACKEND_MAXIMUM_UPLOADS_IN_FLIGHT,
	       "the backend's limit held");
	step_driver_destroy(driver);
	cards_close();
	/* the staging block stays where a card failed under a copy no wait saw finish: last, as the block is kept */
	fixture_build(&fixture, cards_open());
	for (uint32_t card = 0; card < CARDS; card++) {
		test_host_memory_backend_set_polls_before_an_upload_finishes(
			card, 1); /* copies a wait has not seen */
	}
	test_host_memory_backend_fail_launch(1, FAULTING_LAUNCH);
	failing_step(driver_of(&fixture, PROGRAMS), STATUS_DEVICE_ERROR, true,
		     STEP_ROW_LOGITS, "a launch that faults");
	fixture_build(&fixture, cards_open());
	test_host_memory_backend_fail_upload(LAST_CARD, FAILING_TICKET);
	failing_step(driver_of(&fixture, PROGRAMS), STATUS_DEVICE_ERROR, true,
		     STEP_ROW_LOGITS, "an upload that fails");
}

/* Whether a step's results are the model's: each row with logits the token and logit of its stream, else none. */
static bool results_are_the_models(const struct step_row_record *rows,
				   uint32_t row_count,
				   const struct step_row_result *results)
{
	bool is_held = true;
	for (uint32_t row = 0; row < row_count; row++) {
		const uint64_t stream =
			expected_stream((uint32_t)rows[row].token_id,
					(uint32_t)rows[row].position, false);
		const bool has_logits = (rows[row].flags & STEP_ROW_LOGITS) !=
					0;
		is_held = is_held &&
			  results[row].token_id ==
				  (has_logits ? (int32_t)(stream % VOCABULARY) :
						STEP_NO_TOKEN) &&
			  results[row].logit ==
				  (has_logits ?
					   (float)(stream % LOGIT_MODULUS) :
					   0.0f);
	}
	return is_held;
}

/* The rows of a pipeline case's step of session 0: PIPELINE_ROWS from step * PIPELINE_ROWS on, logits for the last. */
static void pipeline_rows(uint32_t step, struct step_row_record *rows)
{
	for (uint32_t row = 0; row < PIPELINE_ROWS; row++) {
		const uint32_t position = step * PIPELINE_ROWS + row;
		rows[row] = (struct step_row_record){
			(int32_t)position,
			(int32_t)((position * PIPELINE_TOKEN_STRIDE + 1u) %
				  VOCABULARY),
			0, row + 1u == PIPELINE_ROWS ? STEP_ROW_LOGITS : 0
		};
	}
}

/* The pipeline: its refusals; steps in flight at once, each with the results of a step at a time; a run behind
 * them; destroy gives both staging blocks back. */
static void test_pipeline(void)
{
	struct fixture fixture;
	struct step_row_record rows[PIPELINE_DEPTH][PIPELINE_ROWS];
	struct step_row_result results[ROW_CAPACITY];
	struct step_driver *driver = NULL;
	context_text = "the pipeline";
	const uint32_t staging_before =
		test_host_memory_backend_staging_block_count();
	fixture_build(&fixture, cards_open());
	expect_status(step_driver_create(&test_host_memory_backend, cards,
					 fixture.pointers, PROGRAMS, engram,
					 WAIT_LIMIT, &driver),
		      STATUS_OK, "the driver is made");
	capture_begin();
	expect_status(step_pipeline_start(driver, 1), STATUS_INVALID_ARGUMENT,
		      "a depth of 1");
	expect_status(step_pipeline_start(driver, STEP_PIPELINE_MOST + 1u),
		      STATUS_INVALID_ARGUMENT, "a depth above the most");
	expect_status(step_finish(driver, results), STATUS_INVALID_ARGUMENT,
		      "a finish without a pipeline");
	expect(capture_end() == TEST_LOG_EXPECTED_LINES(PIPELINE_REFUSALS),
	       "a line for each refusal");
	expect_status(step_pipeline_start(driver, PIPELINE_DEPTH), STATUS_OK,
		      "a pipeline starts");
	expect(step_pipeline_depth(driver) == PIPELINE_DEPTH,
	       "the pipeline has its depth");
	for (uint32_t step = 0; step < PIPELINE_DEPTH; step++) {
		pipeline_rows(step, rows[step]);
		expect_status(step_submit(driver, rows[step], PIPELINE_ROWS),
			      STATUS_OK, "a step goes in flight");
	}
	const struct step_row_record beyond = { PIPELINE_DEPTH * PIPELINE_ROWS,
						1, 0, STEP_ROW_LOGITS };
	expect_status(step_submit(driver, &beyond, 1), STATUS_BUSY_VALUE,
		      "a step beyond the depth waits");
	for (uint32_t step = 0; step < PIPELINE_DEPTH; step++) {
		expect_status(step_finish(driver, results), STATUS_OK,
			      "the oldest step lands");
		expect(results_are_the_models(rows[step], PIPELINE_ROWS,
					      results),
		       "each step in flight has the results of a step at a time");
	}
	uint64_t marks_set = 0, marks_followed = 0, unset_followed = 0;
	for (uint32_t card = 0; card < CARDS; card++) {
		uint64_t set = 0, followed = 0, unset = 0;
		test_host_memory_backend_marks(card, &set, &followed, &unset);
		marks_set += set;
		marks_followed += followed;
		unset_followed += unset;
	}
	expect(marks_set != 0 && marks_followed != 0 && unset_followed == 0,
	       "the hand-offs pass by marks, each followed one set first");
	capture_begin();
	expect_status(step_finish(driver, results), STATUS_INVALID_ARGUMENT,
		      "a finish with no step in flight");
	expect_status(step_pipeline_start(driver, PIPELINE_DEPTH),
		      STATUS_INVALID_ARGUMENT, "a second start");
	expect(capture_end() == TEST_LOG_EXPECTED_LINES(PIPELINE_LATE_REFUSALS),
	       "a line for each refusal");
	expect_status(step_run_entry(driver, &beyond, 1, results), STATUS_OK,
		      "a run is a step through the pipeline");
	expect(results_are_the_models(&beyond, 1, results),
	       "the run has the model's results");
	step_driver_destroy(driver);
	expect(test_host_memory_backend_staging_block_count() == staging_before,
	       "destroy frees the staging blocks of the driver and its pipeline");
	struct backend_operations unmarked = test_host_memory_backend;
	unmarked.mark_record = NULL;
	unmarked.mark_follow = NULL;
	driver = NULL;
	expect_status(step_driver_create(&unmarked, cards, fixture.pointers,
					 PROGRAMS, engram, WAIT_LIMIT, &driver),
		      STATUS_OK, "a driver on a backend without marks");
	expect_status(step_pipeline_start(driver, PIPELINE_DEPTH), STATUS_OK,
		      "its pipeline starts");
	for (uint32_t step = 0; step < PIPELINE_DEPTH; step++) {
		expect_status(step_submit(driver, rows[step], PIPELINE_ROWS),
			      STATUS_OK, "a step goes in flight");
	}
	for (uint32_t step = 0; step < PIPELINE_DEPTH; step++) {
		expect_status(step_finish(driver, results), STATUS_OK,
			      "the oldest step lands");
		expect(results_are_the_models(rows[step], PIPELINE_ROWS,
					      results),
		       "without marks the hand-offs wait on the host: the same results");
	}
	step_driver_destroy(driver);
	cards_close();
}

static _Atomic uint64_t gate_sequence;
static _Atomic uint64_t gate_limit = GATE_LIMIT_NANOSECONDS;
static _Atomic bool gate_missed;
static _Atomic uint64_t
	area_sequence; /* the last step whose step area went up to card 1 */

/* The backend of the contiguous pipeline's case: card 1's step area notes its step as it goes up. */
MUST_CHECK static result_status noting_upload_begin(struct device_card *card,
						    uint64_t arena_byte_offset,
						    const void *source,
						    uint64_t byte_count,
						    uint64_t *ticket_out)
{
	struct step_block block = { 0 };
	const result_status status = test_host_memory_backend.upload_begin(
		card, arena_byte_offset, source, byte_count, ticket_out);
	if (status == STATUS_OK && card == cards[1] &&
	    arena_byte_offset == STEP_OFFSET && byte_count >= sizeof(block)) {
		memcpy(&block, source, sizeof(block));
		atomic_store(&area_sequence, block.sequence);
	}
	return status;
}

/* And card 0's embedding of a step waits until card 1's step area of it went up, at most gate_limit nanoseconds (else
 * gate_missed): a card of one visit takes a step before the hand-offs it waits for are sent. */
MUST_CHECK static result_status
early_launch_run(struct device_card *card, const struct kernel_launch *launches,
		 uint32_t launch_count)
{
	struct fill_record record = { 0 };
	struct step_block block = { 0 };
	const unsigned char *const arena =
		test_host_memory_backend_arena_of(card);
	if (launches != NULL && launch_count != 0) {
		memcpy(&record, launches[0].argument, sizeof(record));
	}
	if (card == cards[0] && arena != NULL &&
	    record.value == OPERATION_EMBED) {
		memcpy(&block, arena + STEP_OFFSET, sizeof(block));
	}
	const struct timespec pause = { 0, GATE_POLL_NANOSECONDS };
	for (uint64_t waited = 0;
	     atomic_load(&area_sequence) < block.sequence &&
	     waited < atomic_load(&gate_limit);
	     waited += GATE_POLL_NANOSECONDS) {
		(void)nanosleep(&pause, NULL);
	}
	if (atomic_load(&area_sequence) < block.sequence) {
		atomic_store(&gate_missed, true);
	}
	return test_host_memory_backend.launch_run(card, launches,
						   launch_count);
}

/* A contiguous layout in a pipeline (each card visited once a step): a card takes its next step once it ran the steps
 * before, and puts the step's areas up while it waits for its hand-offs, as a worker that ran its card's part of each
 * step in order did: card 0 embeds no step before card 1 holds that step's area, though card 1 runs behind it. */
static void test_pipeline_early_areas(void)
{
	struct fixture fixture;
	struct step_row_record rows[PIPELINE_DEPTH][PIPELINE_ROWS];
	struct step_row_result results[ROW_CAPACITY];
	struct step_driver *driver = NULL;
	struct backend_operations noting = test_host_memory_backend;
	noting.upload_begin = noting_upload_begin;
	noting.launch_run = early_launch_run;
	context_text = "the step areas of a contiguous pipeline";
	fixture_build(&fixture, cards_open());
	expect_status(step_driver_create(&noting, cards, fixture.pointers,
					 PROGRAMS, engram, WAIT_LIMIT, &driver),
		      STATUS_OK, "the driver is made");
	expect_status(step_pipeline_start(driver, PIPELINE_DEPTH), STATUS_OK,
		      "a pipeline starts");
	atomic_store(&area_sequence, 0);
	atomic_store(&gate_missed, false);
	for (uint32_t step = 0; step < PIPELINE_DEPTH; step++) {
		pipeline_rows(step, rows[step]);
		expect_status(step_submit(driver, rows[step], PIPELINE_ROWS),
			      STATUS_OK, "a step goes in flight");
	}
	for (uint32_t step = 0; step < PIPELINE_DEPTH; step++) {
		expect_status(step_finish(driver, results), STATUS_OK,
			      "the oldest step lands");
		expect(results_are_the_models(rows[step], PIPELINE_ROWS,
					      results),
		       "each step has the results of a step at a time");
	}
	expect(!atomic_load(&gate_missed),
	       "card 1 puts each step's area up before card 0 embeds the step and sends its hand-off");
	step_driver_destroy(driver);
	cards_close();
}

/* The launches the log holds on card from entry first on, and its copies to or from card at offset. */
static uint32_t launches_logged(uint64_t first, uint32_t card)
{
	struct test_host_memory_backend_work work;
	uint32_t count = 0;
	for (uint64_t index = first;
	     test_host_memory_backend_work_at(index, &work); index++) {
		count += work.kind == TEST_HOST_MEMORY_BACKEND_LAUNCH &&
					 work.card_index == card ?
				 1u :
				 0u;
	}
	return count;
}

static uint32_t copies_logged(uint64_t first, uint32_t card, uint32_t kind,
			      uint64_t offset)
{
	struct test_host_memory_backend_work work;
	uint32_t count = 0;
	for (uint64_t index = first;
	     test_host_memory_backend_work_at(index, &work); index++) {
		count += work.kind == kind && work.card_index == card &&
					 work.arena_byte_offset >= offset &&
					 work.arena_byte_offset - offset <
						 POOL_STRIDE * SESSIONS ?
				 1u :
				 0u;
	}
	return count;
}

static bool results_are_front(const struct step_row_result *results,
			      uint32_t row_count)
{
	bool is_front = true;
	for (uint32_t row = 0; row < row_count; row++) {
		is_front = is_front && results[row].token_id == STEP_NO_TOKEN &&
			   results[row].logit == 0.0f;
	}
	return is_front;
}

/* Front steps (model_seam.h), the programs' front their first two segments: a run of front rows has card 0's
 * launches and card 1's first, the hand-offs behind them (the pool's entries go up to the last card), no result
 * (no token for a row); a step of whole rows behind it the model's results; the same in a pipeline. Refused: rows of
 * both kinds, a front row with logits, front rows where the programs have no front. */
static void test_front_steps(void)
{
	struct fixture fixture;
	struct step_row_result results[ROW_CAPACITY];
	struct step_row_record front[ROW_CAPACITY], whole[ROW_CAPACITY];
	context_text = "front steps";
	fixture_build(&fixture, cards_open());
	for (uint32_t row = 0; row < ROW_CAPACITY; row++) {
		front[row] = (struct step_row_record){ (int32_t)row,
						       (int32_t)(row + 1u), 0,
						       STEP_ROW_FRONT };
		whole[row] = (struct step_row_record){
			(int32_t)(ROW_CAPACITY + row), (int32_t)(row + 1u), 0,
			row + 1u == ROW_CAPACITY ? STEP_ROW_LOGITS : 0
		};
	}
	struct step_driver *driver = driver_of(&fixture, PROGRAMS);
	capture_begin();
	expect_status(step_run_entry(driver, front, ROW_CAPACITY, results),
		      STATUS_INVALID_ARGUMENT,
		      "front rows where the programs have no front");
	step_driver_destroy(driver);
	for (uint32_t program = 0; program < PROGRAMS; program++) {
		fixture.programs[program].front_segment_count = FRONT_SEGMENTS;
	}
	driver = driver_of(&fixture, PROGRAMS);
	const struct step_row_record mixed[] = { front[0], { 1, 1, 0, 0 } };
	const struct step_row_record logits = {
		0, 1, 0, STEP_ROW_FRONT | STEP_ROW_LOGITS
	};
	expect_status(step_run_entry(driver, mixed,
				     sizeof(mixed) / sizeof(mixed[0]), results),
		      STATUS_INVALID_ARGUMENT, "rows of both kinds");
	expect_status(step_run_entry(driver, &logits, 1, results),
		      STATUS_INVALID_ARGUMENT, "a front row with logits");
	expect(capture_end() == TEST_LOG_EXPECTED_LINES(FRONT_REFUSALS),
	       "a line for each refusal");
	const uint64_t work_before = test_host_memory_backend_work_count();
	expect_status(step_run_entry(driver, front, ROW_CAPACITY, results),
		      STATUS_OK, "a front step runs");
	expect(results_are_front(results, ROW_CAPACITY),
	       "a front step's rows have no token");
	expect(launches_logged(work_before, 0) ==
			       fixture.programs[1].cards[0].launch_count &&
		       launches_logged(work_before, 1) ==
			       fixture.segments[1][SEGMENT_FIRST_LAYER]
				       .launch_count &&
		       launches_logged(work_before, LAST_CARD) == 0,
	       "the front's launches alone");
	expect(copies_logged(work_before, LAST_CARD,
			     TEST_HOST_MEMORY_BACKEND_UPLOAD,
			     POOL_OFFSET) != 0 &&
		       copies_logged(work_before, 0,
				     TEST_HOST_MEMORY_BACKEND_UPLOAD,
				     BACK_OFFSET) == 0,
	       "the pool's entries go up to the last card, the later hand-offs not at all");
	expect_status(step_run_entry(driver, whole, ROW_CAPACITY, results),
		      STATUS_OK, "whole rows behind them run");
	expect(results_are_the_models(whole, ROW_CAPACITY, results),
	       "they have the model's results");
	step_driver_destroy(driver);
	expect_status(
		step_driver_create(&test_host_memory_backend, cards,
				   fixture.pointers, PROGRAMS, engram,
				   WAIT_LIMIT, &driver),
		STATUS_OK,
		"a driver on the backend itself: the pipeline's workers copy at once");
	expect_status(step_pipeline_start(driver, FRONT_DEPTH), STATUS_OK,
		      "a pipeline starts");
	const struct step_session later = { .seed = LATER_SEED,
					    .temperature = LATER_TEMPERATURE,
					    .top_p = LATER_TOP_P };
	const struct step_session first_settings = { .seed = 0,
						     .temperature = 0.0f,
						     .top_p = 1.0f };
	expect_status(step_session_set(driver, 0, &later), STATUS_OK,
		      "settings of slot 0 before a front step");
	expect_status(step_submit(driver, front, ROW_CAPACITY), STATUS_OK,
		      "a front step goes in flight");
	expect_status(step_submit(driver, whole, ROW_CAPACITY), STATUS_OK,
		      "whole rows behind it");
	expect_status(step_finish(driver, results), STATUS_OK,
		      "the front step lands");
	expect(results_are_front(results, ROW_CAPACITY),
	       "its rows have no token");
	expect_status(step_finish(driver, results), STATUS_OK,
		      "the whole rows land");
	expect(results_are_the_models(whole, ROW_CAPACITY, results),
	       "with the model's results");
	expect(settings_held(&later, &first_settings),
	       "the settings reach the card that samples with the next step that samples, not with the front step");
	step_driver_destroy(driver);
	cards_close();
}

/* The wide program of the fixture (model_seam.h): the front of the thin program of 4 rows (its first two segments and
 * the hand-offs behind them) for front steps of WIDE_ROWS rows, a view on each of the two cards that run the front. */
static void wide_program_build(const struct fixture *fixture,
			       struct execution_program *wide,
			       struct program_card_record *wide_cards,
			       const struct step_view *view)
{
	const struct execution_program *const thin = &fixture->programs[1];
	const uint32_t front_launches[CARDS] = {
		thin->cards[0].launch_count,
		fixture->segments[1][SEGMENT_FIRST_LAYER].launch_count, 0
	};
	for (uint32_t card = 0; card < CARDS; card++) {
		wide_cards[card] = thin->cards[card];
		wide_cards[card].launch_count = front_launches[card];
		wide_cards[card].views = front_launches[card] != 0 ? view :
								     NULL;
		wide_cards[card].view_count = front_launches[card] != 0 ? 1u :
									  0u;
	}
	*wide = *thin;
	wide->cards = wide_cards;
	wide->kind = PROGRAM_WIDE;
	wide->row_count = WIDE_ROWS;
	wide->segment_count = FRONT_SEGMENTS;
	wide->draft_segment_first = FRONT_SEGMENTS;
	wide->front_segment_count = FRONT_SEGMENTS;
	wide->hand_off_count = WIDE_FRONT_HAND_OFFS;
}

/* Wide steps beside the thin programs: create takes the wide program; a front step of its rows runs it, its view up on
 * the two cards of the front with the step's sequence, live rows and rows, the front's launches and hand-offs alone,
 * no result; steps no program runs refused with nothing queued (5 front rows, 6 rows that are not front rows); the
 * same step in a pipeline. Refused at create: a card of the front without its view, a view of fewer rows. */
static void test_wide_steps(void)
{
	struct fixture fixture;
	struct execution_program wide;
	struct program_card_record wide_cards[CARDS];
	const struct step_view view = { VIEW_OFFSET, 0, LIMIT_ROWS };
	const struct step_view short_view = { VIEW_OFFSET, 0, WIDE_ROWS - 1u };
	const struct execution_program *programs[PROGRAMS + 1u];
	struct step_row_result results[WIDE_ROWS];
	struct step_row_record rows[WIDE_ROWS], whole[WIDE_ROWS];
	unsigned char expected[STEP_BLOCK_BYTES_VALUE +
			       WIDE_ROWS * STEP_ROW_BYTES_VALUE];
	struct step_driver *driver = NULL;
	context_text = "wide steps";
	fixture_build(&fixture, cards_open());
	for (uint32_t program = 0; program < PROGRAMS; program++) {
		fixture.programs[program].front_segment_count = FRONT_SEGMENTS;
		programs[program] = &fixture.programs[program];
	}
	wide_program_build(&fixture, &wide, wide_cards, &view);
	programs[PROGRAMS] = &wide;
	for (uint32_t row = 0; row < WIDE_ROWS; row++) {
		rows[row] = (struct step_row_record){ (int32_t)row,
						      (int32_t)(row + 1u), 0,
						      STEP_ROW_FRONT };
		whole[row] = rows[row];
		whole[row].flags = 0;
	}
	const struct step_block header = { .sequence = 1,
					   .row_count = WIDE_ROWS };
	memcpy(expected, &header, sizeof(header));
	memcpy(expected + STEP_BLOCK_BYTES_VALUE, rows, sizeof(rows));
	expect_status(step_driver_create(&checked_backend, cards, programs,
					 PROGRAMS + 1u, engram, WAIT_LIMIT,
					 &driver),
		      STATUS_OK, "a driver takes the wide program");
	const uint64_t work_before = test_host_memory_backend_work_count();
	capture_begin();
	expect_status(step_run_entry(driver, rows, WIDE_ROWS - 1u, results),
		      STATUS_INVALID_ARGUMENT,
		      "front rows fewer than the wide program's");
	expect_status(step_run_entry(driver, whole, WIDE_ROWS, results),
		      STATUS_INVALID_ARGUMENT,
		      "the wide program's rows, not front rows");
	expect(capture_end() == TEST_LOG_EXPECTED_LINES(WIDE_REFUSALS) &&
		       test_host_memory_backend_work_count() == work_before,
	       "a line for each refusal, nothing queued");
	expect_status(step_run_entry(driver, rows, WIDE_ROWS, results),
		      STATUS_OK, "a wide step runs");
	expect(results_are_front(results, WIDE_ROWS),
	       "a wide step's rows have no token");
	expect(launches_logged(work_before, 0) == wide_cards[0].launch_count &&
		       launches_logged(work_before, 1) ==
			       wide_cards[1].launch_count &&
		       launches_logged(work_before, LAST_CARD) == 0,
	       "the front's launches alone");
	for (uint32_t card = 0; card < LAST_CARD; card++) {
		expect(memcmp(test_host_memory_backend_arena_of(cards[card]) +
				      VIEW_OFFSET,
			      expected, sizeof(expected)) == 0,
		       "the view holds the step's sequence, live rows and rows");
	}
	expect(copies_logged(work_before, LAST_CARD,
			     TEST_HOST_MEMORY_BACKEND_UPLOAD,
			     POOL_OFFSET) != 0 &&
		       copies_logged(work_before, LAST_CARD,
				     TEST_HOST_MEMORY_BACKEND_DOWNLOAD,
				     RESULT_OFFSET) == 0,
	       "the front's hand-offs go on; no result comes back");
	step_driver_destroy(driver);
	expect_status(step_driver_create(&test_host_memory_backend, cards,
					 programs, PROGRAMS + 1u, engram,
					 WAIT_LIMIT, &driver),
		      STATUS_OK, "a driver for a pipeline");
	expect_status(step_pipeline_start(driver, FRONT_DEPTH), STATUS_OK,
		      "a pipeline starts");
	expect_status(step_submit(driver, rows, WIDE_ROWS), STATUS_OK,
		      "a wide step goes in flight");
	expect_status(step_finish(driver, results), STATUS_OK, "it lands");
	expect(results_are_front(results, WIDE_ROWS), "its rows have no token");
	step_driver_destroy(driver);
	driver = (void *)&failures; /* an address no create gives */
	capture_begin();
	wide_cards[1].views = NULL;
	wide_cards[1].view_count = 0;
	expect_status(step_driver_create(&checked_backend, cards, programs,
					 PROGRAMS + 1u, engram, WAIT_LIMIT,
					 &driver),
		      STATUS_INVALID_ARGUMENT,
		      "a card of the front without its view");
	wide_program_build(&fixture, &wide, wide_cards, &short_view);
	expect_status(step_driver_create(&checked_backend, cards, programs,
					 PROGRAMS + 1u, engram, WAIT_LIMIT,
					 &driver),
		      STATUS_INVALID_ARGUMENT,
		      "a view that holds fewer rows than the wide program's");
	expect(capture_end() == TEST_LOG_EXPECTED_LINES(WIDE_CREATE_REFUSALS) &&
		       driver == (void *)&failures,
	       "a line for each refusal, no driver");
	cards_close();
}

/* A full program beside the thin ones: all of the larger thin program but its last segment (the sampling card's) and
 * the hand-offs of rows behind its own last, which only that segment would read (step.h), of WIDE_ROWS rows, a view
 * on each card that runs launches (model_seam.h's FULL, without a head in this build). */
static void full_program_build(const struct fixture *fixture,
			       struct execution_program *full,
			       struct program_card_record *full_cards,
			       const struct step_view *view)
{
	const struct execution_program *const thin = &fixture->programs[1];
	for (uint32_t card = 0; card < CARDS; card++) {
		full_cards[card] = thin->cards[card];
		full_cards[card].launch_count =
			card == LAST_CARD ? 0 : thin->cards[card].launch_count;
		full_cards[card].views = card == LAST_CARD ? NULL : view;
		full_cards[card].view_count = card == LAST_CARD ? 0 : 1u;
	}
	*full = *thin;
	full->cards = full_cards;
	full->kind = PROGRAM_FULL;
	full->row_count = WIDE_ROWS;
	full->segment_count = SEGMENTS - 1u;
	full->draft_segment_first = full->segment_count;
	full->hand_off_count = FULL_HAND_OFFS;
}

/* Full steps beside the thin programs: create takes the full program; a step of its rows, none a front row or of
 * logits, runs it: every launch but the sampling card's, no token and no result back; refused with nothing queued: the
 * same rows with the last one of logits, front rows of its count; the same step in a pipeline. */
static void test_full_steps(void)
{
	struct fixture fixture;
	struct execution_program full;
	struct program_card_record full_cards[CARDS];
	const struct step_view view = { VIEW_OFFSET, 0, LIMIT_ROWS };
	const struct execution_program *programs[PROGRAMS + 1u];
	struct step_row_result results[WIDE_ROWS];
	struct step_row_record rows[WIDE_ROWS], logits[WIDE_ROWS],
		front[WIDE_ROWS];
	struct step_driver *driver = NULL;
	context_text = "full steps";
	fixture_build(&fixture, cards_open());
	for (uint32_t program = 0; program < PROGRAMS; program++) {
		programs[program] = &fixture.programs[program];
	}
	full_program_build(&fixture, &full, full_cards, &view);
	programs[PROGRAMS] = &full;
	for (uint32_t row = 0; row < WIDE_ROWS; row++) {
		rows[row] = (struct step_row_record){ (int32_t)row,
						      (int32_t)(row + 1u), 0,
						      0 };
		logits[row] = rows[row];
		front[row] = rows[row];
		front[row].flags = STEP_ROW_FRONT;
	}
	logits[WIDE_ROWS - 1u].flags = STEP_ROW_LOGITS;
	expect_status(step_driver_create(&checked_backend, cards, programs,
					 PROGRAMS + 1u, engram, WAIT_LIMIT,
					 &driver),
		      STATUS_OK, "a driver takes the full program");
	expect(step_chunk_rows(driver, PROGRAM_FULL, UINT32_MAX) == WIDE_ROWS &&
		       step_chunk_rows(driver, PROGRAM_FULL, WIDE_ROWS - 1u) ==
			       0 &&
		       step_wide_rows(driver) == 0,
	       "the full program's rows, none under them, no wide program");
	const uint64_t work_before = test_host_memory_backend_work_count();
	capture_begin();
	expect_status(step_run_entry(driver, logits, WIDE_ROWS, results),
		      STATUS_INVALID_ARGUMENT,
		      "a full step with a row of logits");
	expect_status(step_run_entry(driver, front, WIDE_ROWS, results),
		      STATUS_INVALID_ARGUMENT,
		      "front rows of the full program's count");
	expect(capture_end() == TEST_LOG_EXPECTED_LINES(FULL_REFUSALS) &&
		       test_host_memory_backend_work_count() == work_before,
	       "a line for each refusal, nothing queued");
	expect_status(step_run_entry(driver, rows, WIDE_ROWS, results),
		      STATUS_OK, "a full step runs");
	expect(results_are_front(results, WIDE_ROWS),
	       "a full step's rows have no token");
	expect(launches_logged(work_before, 0) == full_cards[0].launch_count &&
		       launches_logged(work_before, 1) ==
			       full_cards[1].launch_count &&
		       launches_logged(work_before, LAST_CARD) == 0,
	       "every launch but the sampling card's");
	expect(copies_logged(work_before, LAST_CARD,
			     TEST_HOST_MEMORY_BACKEND_DOWNLOAD,
			     RESULT_OFFSET) == 0 &&
		       copies_logged(work_before, LAST_CARD,
				     TEST_HOST_MEMORY_BACKEND_UPLOAD,
				     STREAMS_OFFSET) == 0,
	       "no result comes back, and no rows go up to the sampling card");
	step_driver_destroy(driver);
	expect_status(step_driver_create(&test_host_memory_backend, cards,
					 programs, PROGRAMS + 1u, engram,
					 WAIT_LIMIT, &driver),
		      STATUS_OK, "a driver for a pipeline");
	expect_status(step_pipeline_start(driver, FRONT_DEPTH), STATUS_OK,
		      "a pipeline starts");
	expect_status(step_submit(driver, rows, WIDE_ROWS), STATUS_OK,
		      "a full step goes in flight");
	expect_status(step_finish(driver, results), STATUS_OK, "it lands");
	expect(results_are_front(results, WIDE_ROWS), "its rows have no token");
	step_driver_destroy(driver);
	cards_close();
}

/* Three rounds through driver: the steps' results, each row held to the model and the n-gram rows to the committed
 * tokens before it; with is_eager the refusals of a cut, without (a pipeline) a cut of a session with a step in
 * flight refused. */
static void rounds_run(struct step_driver *driver, bool is_eager,
		       bool is_interleaved,
		       struct step_row_result results[][ROW_CAPACITY])
{
	/* session 0's anchor and three proposals, all drawn; then two of its rows again behind session 1's prompt;
	 * then its next anchor */
	static const struct step_row_record steps[ROUND_STEPS][ROW_CAPACITY] = {
		{ { 0, 1, 0, STEP_ROW_LOGITS },
		  { 1, POSITION_TWO, 0, STEP_ROW_LOGITS },
		  { POSITION_TWO, POSITION_THREE, 0, STEP_ROW_LOGITS },
		  { POSITION_THREE, POSITION_FOUR, 0, STEP_ROW_LOGITS } },
		{ { 0, POSITION_THREE, 1, 0 },
		  { 1, POSITION_FOUR, 1, 0 },
		  { POSITION_TWO, POSITION_FIVE, 0, STEP_ROW_LOGITS },
		  { POSITION_THREE, 1, 0, STEP_ROW_LOGITS } },
		{ { POSITION_THREE, POSITION_TWO, 0, STEP_ROW_LOGITS } }
	};
	static const uint32_t row_counts[ROUND_STEPS] = { ROW_CAPACITY,
							  ROW_CAPACITY, 1 };
	static const uint32_t cuts[ROUND_STEPS] = { ROUND_CUT, ROUND_SECOND_CUT,
						    POSITION_FOUR };
	const struct step_row_record past = { ROUND_CUT + 1u, 1, 0,
					      STEP_ROW_LOGITS };
	for (uint32_t step = 0; step < ROUND_STEPS; step++) {
		const struct step_row_record *const rows = steps[step];
		for (uint32_t row = 0; row < row_counts[step]; row++) {
			history[rows[row].session_slot][rows[row].position] =
				(uint32_t)rows[row].token_id;
		}
		if (!is_eager) {
			expect_status(step_submit(driver, rows,
						  row_counts[step]),
				      STATUS_OK, "a step goes in flight");
			capture_begin();
			expect_status(
				step_history_truncate(driver, 0, cuts[step]),
				STATUS_INVALID_ARGUMENT,
				"a cut of a session with a step in flight");
			expect(capture_end() == TEST_LOG_EXPECTED_LINES(1u),
			       "a line for the refusal");
		}
		expect_status(is_eager ? step_run_entry(driver, rows,
							row_counts[step],
							results[step]) :
					 step_finish(driver, results[step]),
			      STATUS_OK, "the round's step");
		expect(is_interleaved ?
			       interleaved_results_hold(rows, row_counts[step],
							results[step]) :
			       results_are_the_models(rows, row_counts[step],
						      results[step]),
		       "every row drawn as a step of it alone would draw it");
		for (uint32_t row = 0; row < row_counts[step]; row++) {
			expect(engram_rows_hold(&rows[row], row),
			       "the n-gram rows of the committed tokens and the step's");
		}
		expect_status(step_history_truncate(driver, 0, cuts[step]),
			      STATUS_OK,
			      "the session cut back to its committed tokens");
#if CHECKING_HOOKS_ENABLED
		uint32_t committed[CONTEXT], count = 0;
		expect_status(step_committed_history_read(driver, 0, committed,
							  CONTEXT, &count),
			      STATUS_OK, "the committed history read");
		expect(count == cuts[step], "it holds the cut's tokens");
#endif
		if (is_eager && step == 0) {
			capture_begin();
			expect_status(step_history_truncate(NULL, 0, 0),
				      STATUS_INVALID_ARGUMENT,
				      "a cut of no driver");
			expect_status(
				step_history_truncate(driver, SESSIONS, 0),
				STATUS_INVALID_ARGUMENT, "a cut of no slot");
			expect_status(
				step_history_truncate(driver, 0,
						      ROUND_CUT + 1u),
				STATUS_INVALID_ARGUMENT,
				"a cut to more tokens than the session holds");
			struct step_row_result refused[1];
			expect_status(step_run_entry(driver, &past, 1, refused),
				      STATUS_INVALID_ARGUMENT,
				      "a row past the cut");
			expect(capture_end() ==
				       TEST_LOG_EXPECTED_LINES(ROUND_REFUSALS),
			       "a line for each refusal");
		}
	}
}

/* Speculative rounds (front_end_seam.h): a step whose session rows all carry logits, beside another session's rows of
 * none, each row drawn as a step of it alone would draw it; the session's committed tokens cut back after each round
 * (step_history_truncate) and the refusals of a cut; the same rounds through a pipeline give the same
 * results and leave the cards byte for byte as the eager driver does. */
static void test_rounds(bool is_interleaved)
{
	static unsigned char eager_arenas[CARDS][ARENA_BYTES];
	struct step_row_result eager[ROUND_STEPS][ROW_CAPACITY],
		pipelined[ROUND_STEPS][ROW_CAPACITY];
	struct fixture fixture;
	struct interleaved revisited;
	const struct execution_program *const *const programs =
		is_interleaved ? revisited.pointers : fixture.pointers;
	struct step_driver *driver = NULL;
	memset(eager, 0, sizeof(eager));
	memset(pipelined, 0, sizeof(pipelined));
	context_text = is_interleaved ? "interleaved rounds" : "rounds";
	uint32_t kernel = cards_open();
	if (is_interleaved) {
		interleaved_build(&revisited, kernel);
	} else {
		fixture_build(&fixture, kernel);
	}
	expect_status(step_driver_create(&test_host_memory_backend, cards,
					 programs, PROGRAMS, engram, WAIT_LIMIT,
					 &driver),
		      STATUS_OK, "a driver for eager rounds");
	rounds_run(driver, true, is_interleaved, eager);
	step_driver_destroy(driver);
	for (uint32_t card = 0; card < CARDS; card++) {
		memcpy(eager_arenas[card], arena_of(card), ARENA_BYTES);
	}
	cards_close();
	kernel = cards_open();
	if (is_interleaved) {
		interleaved_build(&revisited, kernel);
	} else {
		fixture_build(&fixture, kernel);
	}
	driver = NULL;
	expect_status(step_driver_create(&test_host_memory_backend, cards,
					 programs, PROGRAMS, engram, WAIT_LIMIT,
					 &driver),
		      STATUS_OK, "a driver for a pipeline");
	expect_status(step_pipeline_start(driver, PIPELINE_DEPTH), STATUS_OK,
		      "a pipeline starts");
	rounds_run(driver, false, is_interleaved, pipelined);
	step_driver_destroy(driver);
	bool is_same = true;
	for (uint32_t step = 0; step < ROUND_STEPS; step++) {
		for (uint32_t row = 0; row < ROW_CAPACITY; row++) {
			is_same = is_same &&
				  eager[step][row].token_id ==
					  pipelined[step][row].token_id &&
				  eager[step][row].logit ==
					  pipelined[step][row].logit;
		}
	}
	expect(is_same, "the pipeline's results those of the eager driver");
	for (uint32_t card = 0; card < CARDS; card++) {
		expect(memcmp(eager_arenas[card], arena_of(card),
			      ARENA_BYTES) == 0,
		       "every card's arena that of the eager driver");
	}
	cards_close();
}

/* The pipeline's backend: card 1's run of S_POOL of step gate_sequence waits until card 0 embedded a later step, at
 * most gate_limit nanoseconds (else gate_missed). A worker that ran its card's part of each step in order would wait
 * for that step's second visit of card 0 and never embed the next. Card 1's step area is its own thread's. */
MUST_CHECK static result_status
gated_launch_run(struct device_card *card, const struct kernel_launch *launches,
		 uint32_t launch_count)
{
	struct fill_record record = { 0 };
	struct step_block block = { 0 };
	const unsigned char *const arena =
		test_host_memory_backend_arena_of(card);
	if (launches != NULL && launch_count != 0) {
		memcpy(&record, launches[0].argument, sizeof(record));
	}
	if (card == cards[1] && arena != NULL && record.word_count == 1) {
		memcpy(&block, arena + STEP_OFFSET, sizeof(block));
	}
	const struct timespec pause = { 0, GATE_POLL_NANOSECONDS };
	for (uint64_t waited = 0;
	     block.sequence != 0 &&
	     block.sequence == atomic_load(&gate_sequence) &&
	     atomic_load(&embedded_sequence) <= block.sequence &&
	     waited < atomic_load(&gate_limit);
	     waited += GATE_POLL_NANOSECONDS) {
		(void)nanosleep(&pause, NULL);
	}
	if (block.sequence != 0 &&
	    block.sequence == atomic_load(&gate_sequence) &&
	    atomic_load(&embedded_sequence) <= block.sequence) {
		atomic_store(&gate_missed, true);
	}
	return test_host_memory_backend.launch_run(card, launches,
						   launch_count);
}

/* The place of a run the stand-in logged, or MOST_RUNS_SEEN. */
static uint32_t run_seen_at(uint64_t sequence, uint32_t card,
			    uint32_t operation)
{
	uint32_t index = 0;
	while (index < runs_seen_count &&
	       (runs_seen[index].sequence != sequence ||
		runs_seen[index].card != card ||
		runs_seen[index].operation != operation)) {
		index++;
	}
	return index < runs_seen_count ? index : MOST_RUNS_SEEN;
}

/* The interleaved plan (step_interleaved.h) through a started pipeline, depth steps in flight: each step lands with the
 * interleaved model's results. */
static void interleaved_pipeline_run(struct step_driver *driver, uint32_t depth)
{
	struct step_row_record rows[INTERLEAVED_STEPS][ROW_CAPACITY];
	struct step_row_result results[ROW_CAPACITY];
	uint32_t row_counts[INTERLEAVED_STEPS] = { 0 };
	for (uint32_t step = 0; step < INTERLEAVED_STEPS + depth; step++) {
		if (step >= depth) {
			const uint32_t landing = step - depth;
			expect_status(step_finish(driver, results), STATUS_OK,
				      "the oldest step lands");
			expect(interleaved_results_hold(rows[landing],
							row_counts[landing],
							results),
			       "each step in flight has the interleaved model's results");
		}
		if (step < INTERLEAVED_STEPS) {
			row_counts[step] = interleaved_rows(
				&interleaved_plan[step], rows[step]);
			expect_status(step_submit(driver, rows[step],
						  row_counts[step]),
				      STATUS_OK, "a step goes in flight");
		}
	}
}

/* The interleaved plan in pipelines of INTERLEAVED_DEPTH and of the most steps on the gated backend: each step's
 * results are the model's; card 0 ran step GATED_STEP + 1's first visit (its embedding) before step GATED_STEP's
 * second, so it ended the later front step first; every mark followed was set; the mirrors hold the owner's entries;
 * destroy gives the staging back. Then a launch of card 0 that faults: a finish fails with it and every later call.
 * Then a mirror sync that fails: card 0's upload of it never seen finished, its step stops there and fails with
 * STATUS_TIMED_OUT. */
static void test_interleaved_pipeline(void)
{
	static const uint32_t depths[] = { INTERLEAVED_DEPTH,
					   STEP_PIPELINE_MOST };
	struct interleaved made;
	struct step_row_record rows[INTERLEAVED_STEPS][ROW_CAPACITY];
	struct step_row_result results[ROW_CAPACITY];
	uint32_t row_counts[INTERLEAVED_STEPS] = { 0 };
	struct backend_operations gated = test_host_memory_backend;
	gated.launch_run = gated_launch_run;
	context_text = "interleaved steps in a pipeline";
	const uint32_t staging_before =
		test_host_memory_backend_staging_block_count();
	for (uint32_t index = 0; index < sizeof(depths) / sizeof(depths[0]);
	     index++) {
		interleaved_build(&made, cards_open());
		struct step_driver *const driver =
			interleaved_driver(&gated, &made);
		expect_status(step_pipeline_start(driver, depths[index]),
			      STATUS_OK, "a pipeline starts");
		atomic_store(&gate_sequence, GATED_STEP);
		atomic_store(&gate_missed, false);
		interleaved_pipeline_run(driver, depths[index]);
		atomic_store(&gate_sequence, 0);
		const uint32_t second_visit =
			run_seen_at(GATED_STEP, 0, OPERATION_MIRROR_LAYER);
		expect(!atomic_load(&gate_missed) &&
			       second_visit < MOST_RUNS_SEEN &&
			       run_seen_at(GATED_STEP + 1u, 0,
					   OPERATION_EMBED) < second_visit,
		       "a card visited twice runs a later step's first visit while an older step's second waits");
		uint64_t marks_set = 0, marks_followed = 0, unset_followed = 0;
		for (uint32_t card = 0; card < CARDS; card++) {
			uint64_t set = 0, followed = 0, unset = 0;
			test_host_memory_backend_marks(card, &set, &followed,
						       &unset);
			marks_set += set;
			marks_followed += followed;
			unset_followed += unset;
		}
		expect(marks_set != 0 && marks_followed != 0 &&
			       unset_followed == 0,
		       "the hand-offs pass by marks, each followed one set first");
		const uint32_t entries[SESSIONS] = {
			(INTERLEAVED_FRONT_TOKENS + ROW_CAPACITY + 1u) /
				POOL_RATIO,
			(ROW_CAPACITY + ROW_CAPACITY) / POOL_RATIO
		};
		for (uint32_t slot = 0; slot < SESSIONS; slot++) {
			expect(interleaved_mirror_holds(0, slot,
							entries[slot]) &&
				       interleaved_mirror_holds(LAST_CARD, slot,
								entries[slot]),
			       "both mirrors hold the owner's entries, the deferred one synced");
		}
		step_driver_destroy(driver);
		expect(test_host_memory_backend_staging_block_count() ==
			       staging_before,
		       "destroy frees the staging blocks of the driver and its pipeline");
		cards_close();
	}
	context_text = "a fault in an interleaved pipeline";
	interleaved_build(&made, cards_open());
	test_host_memory_backend_fail_launch(0, INTERLEAVED_FAULTING_LAUNCH);
	struct step_driver *driver =
		interleaved_driver(&test_host_memory_backend, &made);
	expect_status(step_pipeline_start(driver, INTERLEAVED_DEPTH), STATUS_OK,
		      "a pipeline starts");
	result_status failure = STATUS_OK;
	capture_begin();
	for (uint32_t step = 0; step < INTERLEAVED_DEPTH; step++) {
		row_counts[step] =
			interleaved_rows(&interleaved_plan[step], rows[step]);
		const result_status submitted =
			step_submit(driver, rows[step], row_counts[step]);
		failure = failure == STATUS_OK ? submitted : failure;
	}
	for (uint32_t step = 0;
	     step < INTERLEAVED_DEPTH && failure == STATUS_OK; step++) {
		failure = step_finish(driver, results);
	}
	expect_status(failure, STATUS_DEVICE_ERROR,
		      "a step of the pipeline fails with the fault");
	expect_status(step_finish(driver, results), STATUS_DEVICE_ERROR,
		      "and the driver answers so from then on");
	step_driver_destroy(driver);
	uint32_t error_count = capture_end();
	expect(LOGGING_HOOKS_ENABLED ? error_count >= 1u : error_count == 0u,
	       "diagnostic error count or release silence");
	cards_close();
	context_text = "a mirror sync that times out in a pipeline";
	interleaved_build(&made, cards_open());
	driver = interleaved_driver(&test_host_memory_backend, &made);
	expect_status(step_pipeline_start(driver, INTERLEAVED_DEPTH), STATUS_OK,
		      "a pipeline starts");
	for (uint32_t step = 0; step <= INTERLEAVED_SYNC_STEP; step++) {
		const bool syncs = step == INTERLEAVED_SYNC_STEP;
		row_counts[step] =
			interleaved_rows(&interleaved_plan[step], rows[step]);
		if (syncs) { /* the pipeline stands still: no call of the backend in flight */
			test_host_memory_backend_set_polls_before_an_upload_finishes(
				0, POLLS_TO_TIME_OUT);
			capture_begin();
		}
		expect_status(step_submit(driver, rows[step], row_counts[step]),
			      STATUS_OK, "a step goes in flight");
		expect_status(
			step_finish(driver, results),
			syncs ? STATUS_TIMED_OUT : STATUS_OK,
			"the front steps land, the step whose mirror sync times out fails");
	}
	step_driver_destroy(driver);
	error_count = capture_end();
	expect(run_seen_at(INTERLEAVED_SYNC_STEP + 1u, 1, OPERATION_LAYER) ==
		       MOST_RUNS_SEEN,
	       "the step stops at its sync: card 1 runs none of it");
	expect(LOGGING_HOOKS_ENABLED ? error_count >= 1u : error_count == 0u,
	       "diagnostic error count or release silence");
	cards_close();
}

/* The interleaved plan in a pipeline whose programs lease card 0's rows from its first visit to its second
 * (model_seam.h), on the gated backend with a short gate: card 0 embeds no later step before every earlier step of
 * other rows ran its second visit, though card 1 holds GATED_STEP back until it does (so the gate runs out), and each
 * step's results are the model's. A pipeline that ignored the lease would open the gate at once. */
static void test_interleaved_lease(void)
{
	static const struct visit_lease lease = { 0, S_LAYER, S_MIRROR };
	struct interleaved made;
	struct backend_operations gated = test_host_memory_backend;
	gated.launch_run = gated_launch_run;
	context_text = "an interleaved pipeline with a lease";
	interleaved_build(&made, cards_open());
	for (uint32_t program = 0; program < PROGRAMS; program++) {
		made.programs[program].lease_count = 1;
		made.programs[program].leases = &lease;
	}
	struct step_driver *const driver = interleaved_driver(&gated, &made);
	expect_status(step_pipeline_start(driver, INTERLEAVED_DEPTH), STATUS_OK,
		      "a pipeline starts");
	atomic_store(&gate_limit, SHORT_GATE_NANOSECONDS);
	atomic_store(&gate_sequence, GATED_STEP);
	atomic_store(&gate_missed, false);
	interleaved_pipeline_run(driver, INTERLEAVED_DEPTH);
	atomic_store(&gate_sequence, 0);
	atomic_store(&gate_limit, GATE_LIMIT_NANOSECONDS);
	bool holds = atomic_load(&gate_missed);
	for (uint32_t step = 0; step < INTERLEAVED_STEPS; step++) {
		const uint32_t second_visit =
			run_seen_at(step + 1u, 0, OPERATION_MIRROR_LAYER);
		for (uint32_t later = step + 1u;
		     later < INTERLEAVED_STEPS &&
		     !interleaved_plan[step].is_front;
		     later++) {
			holds = holds &&
				second_visit < run_seen_at(later + 1u, 0,
							   OPERATION_EMBED);
		}
	}
	expect(holds,
	       "a lease keeps a later step's first visit of card 0 behind an earlier step's second");
	step_driver_destroy(driver);
	cards_close();
}

/* A pipeline that stops moving: card 1 held inside the first step's S_POOL (a front step: no later step comes to open
 * the gate) past the driver's short wait limit. The finish fails with STATUS_TIMED_OUT, and so does the driver
 * from then on; destroy waits for the held worker and gives the staging back. */
static void test_pipeline_deadline(void)
{
	struct interleaved made;
	struct step_row_record rows[ROW_CAPACITY];
	struct step_row_result results[ROW_CAPACITY];
	struct step_driver *driver = NULL;
	struct backend_operations gated = test_host_memory_backend;
	gated.launch_run = gated_launch_run;
	context_text = "a pipeline that stops moving";
	const uint32_t staging_before =
		test_host_memory_backend_staging_block_count();
	interleaved_build(&made, cards_open());
	expect_status(step_driver_create(&gated, cards, made.pointers, PROGRAMS,
					 engram, SHORT_PIPELINE_WAIT, &driver),
		      STATUS_OK, "a driver of short waits");
	expect_status(step_pipeline_start(driver, INTERLEAVED_DEPTH), STATUS_OK,
		      "a pipeline starts");
	atomic_store(&gate_limit, SHORT_GATE_NANOSECONDS);
	atomic_store(&gate_sequence, 1);
	const uint32_t row_count = interleaved_rows(&interleaved_plan[0], rows);
	capture_begin();
	expect_status(step_submit(driver, rows, row_count), STATUS_OK,
		      "a step goes in flight");
	expect_status(step_finish(driver, results), STATUS_TIMED_OUT,
		      "its finish times out");
	expect_status(step_submit(driver, rows, row_count), STATUS_TIMED_OUT,
		      "and the driver answers so from then on");
	step_driver_destroy(driver);
	const uint32_t error_count = capture_end();
	atomic_store(&gate_sequence, 0);
	atomic_store(&gate_limit, GATE_LIMIT_NANOSECONDS);
	expect(LOGGING_HOOKS_ENABLED ? error_count >= 1u : error_count == 0u,
	       "diagnostic error count or release silence");
	expect(test_host_memory_backend_staging_block_count() == staging_before,
	       "destroy waits for the held worker and frees the staging blocks");
	cards_close();
}

/* The interleaved plan one step at a time; then a mirror sync that fails: card 0's upload of it never seen finished
 * within a short wait. The step fails with STATUS_TIMED_OUT, and so does the driver from then on. */
static void test_interleaved_steps(void)
{
	struct interleaved made;
	struct step_row_record rows[ROW_CAPACITY];
	struct step_row_result results[ROW_CAPACITY];
	struct step_driver *driver = NULL;
	interleaved_eager_steps();
	context_text = "a mirror sync that times out";
	interleaved_build(&made, cards_open());
	expect_status(step_driver_create(&checked_backend, cards, made.pointers,
					 PROGRAMS, engram, SHORT_WAIT_LIMIT,
					 &driver),
		      STATUS_OK, "a driver of short waits");
	for (uint32_t step = 0; step < INTERLEAVED_SYNC_STEP; step++) {
		const uint32_t row_count =
			interleaved_rows(&interleaved_plan[step], rows);
		expect_status(step_run_entry(driver, rows, row_count, results),
			      STATUS_OK, "a front step runs");
	}
	test_host_memory_backend_set_polls_before_an_upload_finishes(
		0, POLLS_TO_TIME_OUT);
	const uint32_t row_count = interleaved_rows(
		&interleaved_plan[INTERLEAVED_SYNC_STEP], rows);
	capture_begin();
	expect_status(step_run_entry(driver, rows, row_count, results),
		      STATUS_TIMED_OUT,
		      "the step whose mirror sync times out fails");
	expect_status(step_run_entry(driver, rows, row_count, results),
		      STATUS_TIMED_OUT, "and so does the next");
	step_driver_destroy(driver);
	const uint32_t error_count = capture_end();
	expect(LOGGING_HOOKS_ENABLED ? error_count >= 1u : error_count == 0u,
	       "diagnostic error count or release silence");
	cards_close();
}

/* A snapshot's bytes in host memory: what a save wrote, read back from cursor by a load. */
struct snapshot_buffer {
	unsigned char bytes[SNAPSHOT_CAPACITY];
	uint64_t byte_count, cursor;
};

static bool snapshot_write(void *context, unsigned char *bytes,
			   uint64_t byte_count)
{
	struct snapshot_buffer *const buffer = context;
	const bool fits = byte_count <= SNAPSHOT_CAPACITY - buffer->byte_count;
	if (fits) {
		memcpy(buffer->bytes + buffer->byte_count, bytes, byte_count);
		buffer->byte_count += byte_count;
	}
	return fits;
}

static bool snapshot_read(void *context, unsigned char *bytes,
			  uint64_t byte_count)
{
	struct snapshot_buffer *const buffer = context;
	const bool holds = byte_count <= buffer->byte_count - buffer->cursor;
	if (holds) {
		memcpy(bytes, buffer->bytes + buffer->cursor, byte_count);
		buffer->cursor += byte_count;
	}
	return holds;
}

static bool snapshot_refuse(void *context, unsigned char *bytes,
			    uint64_t byte_count)
{
	(void)context;
	(void)bytes;
	(void)byte_count;
	return false;
}

/* Snapshots: session 0's pool saved after its steps and loaded into slot 1, whose owner's and mirror's ranges then
 * hold it and which continues where it ends; the refusals; a function that fails. */
static void test_snapshots(void)
{
	static struct snapshot_buffer buffer;
	struct fixture fixture;
	context_text = "snapshots";
	fixture_build(&fixture, cards_open());
	struct step_driver *const driver = driver_of(&fixture, PROGRAMS);
	const uint64_t reference_identity = step_session_identity(driver);
	expect(reference_identity == BASE_REFERENCE_IDENTITY,
	       "reference snapshots retain the base identity");
	fixture.programs[0].limits.cmp_order = CMP_ORDER_RELAXED;
	expect(step_session_identity(driver) != reference_identity,
	       "relaxed CMP sums give snapshots a distinct identity");
	const uint64_t relaxed_identity = step_session_identity(driver);
	expect(relaxed_identity != BASE_RELAXED_TRUNK_IDENTITY,
	       "relaxed expert sums refuse the base trunk-only snapshot identity");
	fixture.programs[0].limits.row_capacity++;
	expect(step_session_identity(driver) != relaxed_identity,
	       "relaxed CMP snapshots also bind the capacity that permits wide products");
	fixture.programs[0].limits.row_capacity--;
	fixture.programs[0].limits.cmp_order = CMP_ORDER_REFERENCE;
	expect(step_session_identity(driver) == reference_identity,
	       "reference CMP order restores the snapshot identity");
	static const uint32_t official_rows[] = { 1, 3, 4, 8 };
	uint64_t official_identities[sizeof(official_rows) /
				     sizeof(official_rows[0])];
	for (uint32_t shape = 0;
	     shape < sizeof(official_rows) / sizeof(official_rows[0]);
	     shape++) {
		fixture.programs[0].limits.official_hc_rows =
			official_rows[shape];
		const uint64_t identity = step_session_identity(driver);
		expect(identity != reference_identity &&
			       identity != relaxed_identity,
		       "source HC snapshot identity differs from existing arithmetic");
		for (uint32_t prior = 0; prior < shape; prior++)
			expect(identity != official_identities[prior],
			       "each source HC call shape has a distinct identity");
		official_identities[shape] = identity;
		fixture.programs[0].limits.row_capacity++;
		expect(step_session_identity(driver) == identity,
		       "explicit source shape is independent of actual row capacity");
		fixture.programs[0].limits.row_capacity--;
	}
	fixture.programs[0].limits.official_hc_rows = 0;
	expect(step_session_identity(driver) == BASE_REFERENCE_IDENTITY,
	       "default0 retains the exact existing snapshot identity");
	const struct step_row_record rows[SNAPSHOT_ROWS] = {
		{ 0, 1, 0, 0 },
		{ 1, POSITION_TWO, 0, 0 },
		{ POSITION_TWO, POSITION_THREE, 0, 0 },
		{ POSITION_THREE, POSITION_FOUR, 0, STEP_ROW_LOGITS }
	};
	const uint32_t tokens[SNAPSHOT_ROWS] = { 1, POSITION_TWO,
						 POSITION_THREE,
						 POSITION_FOUR };
	const uint64_t entry_bytes = SNAPSHOT_ROWS / POOL_RATIO * ENTRY_BYTES;
	step_expecting(driver, rows, SNAPSHOT_ROWS, ROW_CAPACITY);
	expect(step_session_bytes(driver, SNAPSHOT_ROWS) == entry_bytes,
	       "a snapshot holds each identity once: the pool's entries");
	buffer.byte_count = 0;
	expect_status(step_session_save(driver, 0, SNAPSHOT_ROWS,
					snapshot_write, &buffer),
		      STATUS_OK, "session 0 is saved");
	expect(buffer.byte_count == entry_bytes &&
		       memcmp(buffer.bytes, arena_of(1) + POOL_OFFSET,
			      entry_bytes) == 0,
	       "the snapshot holds the owner's entries of session 0");
	buffer.cursor = 0;
	expect_status(step_session_load(driver, 1, tokens, SNAPSHOT_ROWS,
					snapshot_read, &buffer),
		      STATUS_OK, "the snapshot goes into slot 1");
	expect(memcmp(arena_of(1) + POOL_OFFSET + POOL_STRIDE, buffer.bytes,
		      entry_bytes) == 0 &&
		       memcmp(arena_of(LAST_CARD) + POOL_OFFSET + POOL_STRIDE,
			      buffer.bytes, entry_bytes) == 0,
	       "every range of the identity holds it in slot 1");
	memcpy(history[1], tokens, sizeof(tokens));
	const struct step_row_record next = { SNAPSHOT_ROWS, 1, 1,
					      STEP_ROW_LOGITS };
	step_expecting(driver, &next, 1, 1);
	capture_begin();
	expect_status(step_session_save(driver, 0, SNAPSHOT_ROWS + 1u,
					snapshot_write, &buffer),
		      STATUS_INVALID_ARGUMENT,
		      "a count above what the slot holds");
	expect_status(step_session_save(driver, SESSIONS, 1, snapshot_write,
					&buffer),
		      STATUS_INVALID_ARGUMENT, "a slot beyond the capacity");
	expect_status(step_session_load(driver, 1, tokens, 1, NULL, NULL),
		      STATUS_INVALID_ARGUMENT, "no function");
	expect(capture_end() == TEST_LOG_EXPECTED_LINES(SNAPSHOT_REFUSALS),
	       "a line for each refusal");
	expect_status(step_session_save(driver, 0, SNAPSHOT_ROWS,
					snapshot_refuse, NULL),
		      STATUS_INPUT_OUTPUT_ERROR,
		      "a write that fails fails the save");
	expect_status(step_session_load(driver, 1, tokens, SNAPSHOT_ROWS,
					snapshot_refuse, NULL),
		      STATUS_INPUT_OUTPUT_ERROR,
		      "a read that fails fails the load");
	struct step_row_result results[1];
	capture_begin();
	expect_status(step_run_entry(driver, &next, 1, results),
		      STATUS_INVALID_ARGUMENT,
		      "after a failed load the slot holds nothing to continue");
	expect(capture_end() == TEST_LOG_EXPECTED_LINES(1u),
	       "a line for the refusal");
	step_driver_destroy(driver);
	cards_close();
}

/* DRAFTS is a separate run, downloaded with the target result in one bounded span, while an ordinary step skips
 * it and downloads only its target results. The same protocol runs eagerly and through pipelined visits.
 * Nine independently corrupted round fields poison the driver before a caller can obtain a round. */
static void test_device_rounds(void)
{
	context_text = "device round protocol";
	for (uint32_t mode = 0; mode < DEVICE_DRAFT_MODES; mode++) {
		for (uint32_t broken = 0; broken < DEVICE_DRAFT_CORRUPTIONS;
		     broken++) {
			struct fixture fixture;
			const uint32_t kernel = cards_open();
			struct segment_record segments[PROGRAMS][SEGMENTS + 1u];
			device_fixture_build(&fixture, segments, kernel);
			device_draft_break =
				(enum device_draft_corruption)broken;
			if (mode == 0 && broken == DEVICE_DRAFT_SOUND) {
				fixture.programs[0].draft_segment_first =
					SEGMENTS + POSITION_TWO;
				expect_refused(
					&fixture, STATUS_INVALID_ARGUMENT,
					"a draft boundary past the program");
				fixture.programs[0].draft_segment_first =
					SEGMENTS;
				fixture.program_cards[0][LAST_CARD]
					.round_offset++;
				expect_refused(&fixture,
					       STATUS_INVALID_ARGUMENT,
					       "an unaligned round");
				fixture.program_cards[0][LAST_CARD]
					.round_offset = STEP_OFFSET;
				expect_refused(&fixture,
					       STATUS_INVALID_ARGUMENT,
					       "a round in the step area");
				fixture.program_cards[0][LAST_CARD]
					.round_offset =
					DEVICE_DRAFT_ROUND_OFFSET;
				segments[0][SEGMENTS].pause =
					SEGMENT_PAUSE_EXPERT_UPLOADS;
				expect_refused(&fixture,
					       STATUS_INVALID_ARGUMENT,
					       "a draft tail cannot pause");
				segments[0][SEGMENTS].pause = 0;
				fixture.program_cards[0][0].round_offset =
					DEVICE_DRAFT_ROUND_OFFSET;
				expect_refused(
					&fixture, STATUS_INVALID_ARGUMENT,
					"a round belongs to the result card");
				fixture.program_cards[0][0].round_offset =
					PROGRAM_NOWHERE;
			}
			expect_status(test_host_memory_backend_set_stand_in(
					      KERNEL_FILE_FILL,
					      KERNEL_FILL_WORDS,
					      device_draft_stand_in, NULL),
				      STATUS_OK, "the draft tail stand-in");
			struct step_driver *driver = NULL;
			expect_status(step_driver_create(
					      &test_host_memory_backend, cards,
					      fixture.pointers, PROGRAMS,
					      engram, WAIT_LIMIT, &driver),
				      STATUS_OK,
				      "a driver with the device tail");
			struct step_round round = { .sequence = UINT64_MAX };
			expect_status(step_round_take(driver, &round),
				      STATUS_INVALID_ARGUMENT,
				      "no round before the first finish");
			expect(round.sequence == UINT64_MAX,
			       "the round output stays untouched on refusal");
			expect(step_draft_positions(driver) ==
					       MODEL_DRAFT_BLOCK &&
				       step_draft_positions(NULL) == 0,
			       "the device block's admitted size");
			expect(step_run_last(fixture.pointers[0],
					     SEGMENT_SAMPLE,
					     HAND_OFFS) == SEGMENTS - 1u &&
				       step_run_last(fixture.pointers[0],
						     SEGMENTS,
						     HAND_OFFS) == SEGMENTS,
			       "graph capture and eager runs keep the draft tail separate");
			if (mode != 0) {
				expect_status(step_pipeline_start(
						      driver, PIPELINE_DEPTH),
					      STATUS_OK,
					      "draft visits pipeline starts");
			}
			struct step_row_record rows[ROW_CAPACITY] = {
				{ 0, 1, 0, STEP_ROW_LOGITS | STEP_ROW_DRAFTS },
				{ 0, 1, 1, 0 },
				{ 1, 1, 0,
				  STEP_ROW_LOGITS | STEP_ROW_PROPOSED },
				{ POSITION_TWO, 1, 0,
				  STEP_ROW_LOGITS | STEP_ROW_PROPOSED }
			};
			struct step_row_result results[ROW_CAPACITY];
			if (broken == DEVICE_DRAFT_SOUND) {
				capture_begin();
				rows[1].flags = STEP_ROW_LOGITS |
						STEP_ROW_DRAFTS;
				expect_status(step_run_entry(driver, rows,
							     ROW_CAPACITY,
							     results),
					      STATUS_INVALID_ARGUMENT,
					      "two drafting sessions refused");
				rows[1].flags = 0;
				rows[POSITION_TWO].flags = STEP_ROW_LOGITS;
				expect_status(
					step_run_entry(driver, rows,
						       ROW_CAPACITY, results),
					STATUS_INVALID_ARGUMENT,
					"DRAFTS belongs to the last unproposed row");
				rows[POSITION_TWO].flags = STEP_ROW_LOGITS |
							   STEP_ROW_PROPOSED;
				rows[0].flags &= ~(uint32_t)STEP_ROW_LOGITS;
				expect_status(step_run_entry(driver, rows,
							     ROW_CAPACITY,
							     results),
					      STATUS_INVALID_ARGUMENT,
					      "DRAFTS requires logits");
				rows[0].flags |= STEP_ROW_LOGITS;
				(void)capture_end();
			}
			const uint64_t before =
				test_host_memory_backend_work_count();
			capture_begin();
			expect_status(
				step_run_entry(driver, rows, ROW_CAPACITY,
					       results),
				broken == DEVICE_DRAFT_SOUND ?
					STATUS_OK :
					STATUS_DEVICE_ERROR,
				"only the complete checked round is accepted");
			(void)capture_end();
			if (broken == DEVICE_DRAFT_SOUND) {
				expect_status(step_round_take(driver, &round),
					      STATUS_OK, "the finished round");
				expect(round.sequence == 1 &&
					       round.session_slot == 0 &&
					       round.emitted_count ==
						       DEVICE_DRAFT_EMITTED &&
					       round.anchor_position ==
						       (int32_t)
							       DEVICE_DRAFT_EMITTED &&
					       round.proposal_count ==
						       MODEL_DRAFT_BLOCK,
				       "the card's complete record returned");
				uint32_t downloads = 0;
				for (uint64_t work = before;
				     work <
				     test_host_memory_backend_work_count();
				     work++) {
					struct test_host_memory_backend_work
						seen = { 0 };
					expect(test_host_memory_backend_work_at(
						       work, &seen),
					       "device round work logged");
					if (seen.kind ==
						    TEST_HOST_MEMORY_BACKEND_DOWNLOAD &&
					    seen.card_index == LAST_CARD &&
					    seen.arena_byte_offset ==
						    RESULT_OFFSET) {
						downloads++;
						expect(seen.byte_count ==
							       DEVICE_DRAFT_ROUND_OFFSET +
								       STEP_ROUND_BYTES -
								       RESULT_OFFSET,
						       "results and round share one download");
					}
				}
				expect(downloads == 1,
				       "exactly one result download");
				expect_status(
					step_history_truncate(
						driver, 0,
						DEVICE_DRAFT_EMITTED),
					STATUS_OK,
					"rejected rows leave committed history");
				const struct step_row_record ordinary = {
					(int32_t)DEVICE_DRAFT_EMITTED, 1, 0,
					STEP_ROW_LOGITS
				};
				expect_status(step_run_entry(driver, &ordinary,
							     1, results),
					      STATUS_OK,
					      "ordinary step skips the tail");
				expect_status(step_round_take(driver, &round),
					      STATUS_OK,
					      "ordinary round accessor");
				expect(round.session_slot == STEP_NO_SESSION &&
					       round.proposal_count == 0 &&
					       round.sequence == 0,
				       "ordinary steps erase a previous round");
			} else {
				expect_status(
					step_round_take(driver, &round),
					STATUS_DEVICE_ERROR,
					"malformed round poisons the driver");
			}
			step_driver_destroy(driver);
			cards_close();
		}
	}
}

int main(void)
{
	(void)setvbuf(stdout, NULL, _IOLBF,
		      0); /* a failure's line survives a sanitizer's stop */
	checked_backend = test_host_memory_backend;
	checked_backend.download_begin = checked_download_begin;
	checked_backend.upload_begin = checked_upload_begin;
	checked_backend.upload_wait = checked_upload_wait;
	checked_backend.launch_run = checked_launch_run;
	capture_descriptor = mkstemp(capture_path);
	error_stream_descriptor = dup(STDERR_FILENO);
	if (capture_descriptor < 0 || error_stream_descriptor < 0 ||
	    unlink(capture_path) != 0) {
		(void)printf(
			"FAILED: the temporary file for the error stream\n");
		return EXIT_FAILURE;
	}
	expect_status(test_engram_pack_open(&engram_pack, &engram), STATUS_OK,
		      "the n-gram tables open");
	test_steps();
	test_create_refusals();
	test_row_refusals();
	test_failures();
	test_pipeline();
	test_pipeline_early_areas();
	test_front_steps();
	test_wide_steps();
	test_full_steps();
	test_rounds(false);
	test_rounds(true);
	test_snapshots();
	test_interleaved_steps();
	test_interleaved_pipeline();
	test_interleaved_lease();
	test_pipeline_deadline();
	test_device_rounds();
	engram_close(engram);
	free(engram_pack);
	(void)close(capture_descriptor);
	(void)close(error_stream_descriptor);
	if (failures != 0) {
		(void)printf("step_test: %d failures\n", failures);
		return EXIT_FAILURE;
	}
	(void)printf("step_test: passed\n");
	return EXIT_SUCCESS;
}
