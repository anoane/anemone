/* SPDX-License-Identifier: AGPL-3.0-or-later */
#ifndef TEST_STEP_PROGRAMS_H
#define TEST_STEP_PROGRAMS_H
#include <stddef.h>
#ifdef TEST_STEP_FIXTURE_IMPLEMENTATION
static void fixture_link(struct fixture *fixture)
{
	for (uint32_t program = 0; program < PROGRAMS; program++) {
		for (uint32_t card = 0; card < CARDS; card++) {
			fixture->program_cards[program][card].launches =
				fixture->launches[program][card];
		}
		fixture->programs[program].cards =
			fixture->program_cards[program];
		fixture->programs[program].segments =
			fixture->segments[program];
		fixture->programs[program].hand_offs =
			fixture->hand_offs[program];
		fixture->programs[program].session_ranges =
			fixture->session_ranges;
		fixture->pointers[program] = &fixture->programs[program];
	}
}

/* The model of the header comment on open cards, with programs of 1 and ROW_CAPACITY rows. */
static void fixture_build(struct fixture *fixture, uint32_t kernel)
{
	static const uint32_t operations[CARDS][LAUNCHES_PER_CARD] = {
		{ OPERATION_EMBED, OPERATION_LAYER },
		{ OPERATION_LAYER, OPERATION_LAYER, OPERATION_LAYER },
		{ OPERATION_LAYER, OPERATION_SAMPLE }
	};
	static const uint32_t launch_counts[CARDS] = { 2, 3, 2 };
	static const uint32_t segment_cards[SEGMENTS] = { 0, 1, 1, 1, 2 };
	static const uint32_t segment_launches[SEGMENTS][SEGMENT_FIELDS] = {
		{ 0, 2 }, { 0, 1 }, { 1, 1 }, { 2, 1 }, { 0, 2 }
	};
	const struct model_limits limits = { .context_token_limit = CONTEXT,
					     .row_capacity = LIMIT_ROWS,
					     .session_capacity = SESSIONS,
					     .state_width = STATE_WIDTH_16 };
	memset(fixture, 0, sizeof(*fixture));
	for (uint32_t program = 0; program < PROGRAMS; program++) {
		const uint32_t row_count = program == 0 ? 1u : ROW_CAPACITY;
		for (uint32_t card = 0; card < CARDS; card++) {
			for (uint32_t launch = 0; launch < launch_counts[card];
			     launch++) {
				const struct fill_record record = {
					.arena_byte_offset = STREAMS_OFFSET,
					.word_count = card == 1 && launch == 0 ?
							      1u :
							      0u,
					.value = operations[card][launch]
				};
				prepare_launch_fill(
					&fixture->launches[program][card]
							  [launch],
					kernel, row_count, 1, 1, 0, &record,
					sizeof(record));
			}
			fixture->program_cards[program][card] =
				(struct program_card_record){
					.arena_byte_count = ARENA_BYTES,
					.step_offset = STEP_OFFSET,
					.session_offset =
						card == LAST_CARD ?
							SESSION_OFFSET :
							PROGRAM_NOWHERE,
					.result_offset =
						card == LAST_CARD ?
							RESULT_OFFSET :
							PROGRAM_NOWHERE,
					.logits_offset = PROGRAM_NOWHERE,
					.round_offset = PROGRAM_NOWHERE,
					.streams_offset = STREAMS_OFFSET,
					.carried_pre_offset = PROGRAM_NOWHERE,
					.engram_rows_offset =
						card == 1 ? PROGRAM_NOWHERE :
							    ENGRAM_OFFSET,
					.engram_tables =
						card == 0 ?
							(1u << TABLES) - 1u :
						card == 1 ? 0 :
							    1u << (TABLES - 1u),
					.launch_count = launch_counts[card]
				};
		}
		for (uint32_t segment = 0; segment < SEGMENTS; segment++) {
			fixture->segments[program][segment] =
				(struct segment_record){
					.card = segment_cards[segment],
					.stage = segment,
					.layer = SEGMENT_NO_LAYER,
					.first_launch =
						segment_launches[segment][0],
					.launch_count =
						segment_launches[segment][1]
				};
		}
		fixture->hand_offs[program][HAND_OFF_TO_SECOND] =
			(struct hand_off_record){
				.source_offset = STREAMS_OFFSET,
				.destination_offset = STREAMS_OFFSET,
				.byte_count = sizeof(uint64_t),
				.from_card = 0,
				.to_card = 1,
				.after_segment = SEGMENT_EMBED
			};
		fixture->hand_offs[program][HAND_OFF_TO_LAST] =
			(struct hand_off_record){
				.source_offset = STREAMS_OFFSET,
				.destination_offset = STREAMS_OFFSET,
				.byte_count = sizeof(uint64_t),
				.from_card = 1,
				.to_card = LAST_CARD,
				.after_segment = SEGMENT_THIRD_LAYER
			};
		fixture->hand_offs[program][HAND_OFF_POOL] =
			(struct hand_off_record){
				.source_offset = POOL_OFFSET,
				.destination_offset = POOL_OFFSET,
				.session_stride = POOL_STRIDE,
				.byte_count = ENTRY_BYTES,
				.entry_ratio = POOL_RATIO,
				.from_card = 1,
				.to_card = LAST_CARD,
				.after_segment = SEGMENT_FIRST_LAYER
			};
		fixture->hand_offs[program][HAND_OFF_BACK] =
			fixture->hand_offs[program][HAND_OFF_TO_LAST];
		fixture->hand_offs[program][HAND_OFF_BACK].destination_offset =
			BACK_OFFSET;
		fixture->hand_offs[program][HAND_OFF_BACK].to_card = 0;
		fixture->programs[program] =
			(struct execution_program){ .limits = limits,
						    .card_count = CARDS,
						    .segment_count = SEGMENTS,
						    .draft_segment_first =
							    SEGMENTS,
						    .hand_off_count = HAND_OFFS,
						    .row_count = row_count,
						    .result_card = LAST_CARD };
	}
	for (uint32_t range = 0; range < SESSION_RANGES; range++) {
		fixture->session_ranges[range] =
			(struct session_range){ POOL_OFFSET,
						POOL_STRIDE,
						CONTEXT / POOL_RATIO,
						ENTRY_BYTES,
						POOL_RATIO,
						range == 0 ? 1u : LAST_CARD,
						0 };
	}
	for (uint32_t program = 0; program < PROGRAMS; program++) {
		fixture->programs[program].session_range_count = SESSION_RANGES;
	}
	fixture_link(fixture);
}

/* Opens the three cards (two of class 80, one of 120) and prepares the fill kernel on each: the same handle. */
static uint32_t cards_open(void)
{
	uint32_t kernel = 0;
	test_host_memory_backend_reset(CARDS, ARENA_BYTES);
	for (uint32_t card = 0; card < CARDS; card++) {
		test_host_memory_backend_set_arithmetic_class(
			card, card == LAST_CARD ? ARITHMETIC_CLASS_SM_120 :
						  ARITHMETIC_CLASS_SM_80);
		expect_status(test_host_memory_backend.card_open(
				      card, ARENA_BYTES, &cards[card]),
			      STATUS_OK, "a card opens");
		expect_status(test_host_memory_backend.kernel_prepare(
				      cards[card], KERNEL_FILE_FILL,
				      KERNEL_FILL_WORDS, FILL_RECORD_BYTES,
				      &kernel),
			      STATUS_OK, "the fill kernel");
	}
	expect_status(test_host_memory_backend_set_stand_in(
			      KERNEL_FILE_FILL, KERNEL_FILL_WORDS,
			      model_stand_in, NULL),
		      STATUS_OK, "the stand-in");
	memset(history, 0, sizeof(history));
	memset(last_ticket, 0, sizeof(last_ticket));
	memset(waited_ticket, 0, sizeof(waited_ticket));
	copy_count = 0;
	held_byte_count = 0;
	rewritten_uploads = 0;
	runs_seen_count = 0;
	atomic_store(&embedded_sequence, 0);
	sample_break = SAMPLE_SOUND;
	return kernel;
}

static void cards_close(void)
{
	for (uint32_t card = 0; card < CARDS; card++) {
		test_host_memory_backend.card_close(cards[card], CLOSE_LIMIT);
		cards[card] = NULL;
	}
}

static struct step_driver *driver_of(const struct fixture *fixture,
				     uint32_t program_count)
{
	struct step_driver *driver = NULL;
	expect_status(step_driver_create(&checked_backend, cards,
					 fixture->pointers, program_count,
					 engram, WAIT_LIMIT, &driver),
		      STATUS_OK, "the driver is made");
	if (test_step_use_graphs) {
		expect_status(step_graphs_prepare(driver), STATUS_OK,
			      "graphs prepared at startup");
	}
	return driver;
}

/* The n-gram rows of a step's row r as card holds them, against the engram's ids and the tables' bytes. */
static bool engram_rows_hold(const struct step_row_record *row,
			     uint32_t row_index)
{
	uint64_t row_ids[TABLES] = { 0 };
	const struct pack_extent *const extents =
		(const void *)(engram_pack + PACK_BLOCK_BYTES);
	bool holds = engram_row_ids_entry(engram, history[row->session_slot],
					  NULL, (uint64_t)row->position,
					  row_ids) == STATUS_OK;
	for (uint32_t table = 0; table < TABLES && holds; table++) {
		const unsigned char *const expected =
			engram_pack + extents[table + 1u].byte_offset +
			row_ids[table] / ENGRAM_ROWS_PER_BLOCK *
				PACK_BLOCK_BYTES +
			row_ids[table] % ENGRAM_ROWS_PER_BLOCK *
				ENGRAM_ROW_BYTES;
		/* card 0 reads both tables (k = table), card 2 table 1 (k = 0) */
		holds = memcmp(arena_of(0) + ENGRAM_OFFSET +
				       ((size_t)table * LIMIT_ROWS +
					row_index) *
					       ENGRAM_ROW_BYTES,
			       expected, ENGRAM_ROW_BYTES) == 0 &&
			(table == 0 ||
			 memcmp(arena_of(LAST_CARD) + ENGRAM_OFFSET +
					(size_t)row_index * ENGRAM_ROW_BYTES,
				expected, ENGRAM_ROW_BYTES) == 0);
	}
	return holds;
}

/* Runs one step of these rows and holds everything it must leave to the model of the header comment. */
static void step_expecting(struct step_driver *driver,
			   const struct step_row_record *rows,
			   uint32_t row_count, uint32_t program_rows)
{
	struct step_row_result results[ROW_CAPACITY];
	const uint64_t work_before = test_host_memory_backend_work_count();
	launch_runs = 0;
	for (uint32_t row = 0; row < row_count; row++) {
		history[rows[row].session_slot][rows[row].position] =
			(uint32_t)rows[row].token_id;
	}
	expect_status(step_run_entry(driver, rows, row_count, results),
		      STATUS_OK, "a step runs");
	expect(launch_runs == LAUNCH_RUNS,
	       "one launch_run for each run of one card's segments no hand-off divides");
	expect(memcmp(arena_of(0) + BACK_OFFSET, arena_of(1) + STREAMS_OFFSET,
		      row_count * sizeof(uint64_t)) == 0,
	       "the hand-off back to card 0 arrived");
	for (uint32_t row = 0; row < row_count; row++) {
		const uint64_t stream =
			expected_stream((uint32_t)rows[row].token_id,
					(uint32_t)rows[row].position, false);
		const bool has_logits = (rows[row].flags & STEP_ROW_LOGITS) !=
					0;
		expect(results[row].token_id ==
				       (has_logits ?
						(int32_t)(stream % VOCABULARY) :
						STEP_NO_TOKEN) &&
			       results[row].logit ==
				       (has_logits ? (float)(stream %
							     LOGIT_MODULUS) :
						     0.0f),
		       "each row's token and logit are the model's, or no token without logits");
		expect(engram_rows_hold(&rows[row], row),
		       "the n-gram rows of the row's tables are where the program says");
		if (((uint32_t)rows[row].position + 1u) % POOL_RATIO == 0) {
			uint64_t entry = 0;
			memcpy(&entry,
			       arena_of(LAST_CARD) + POOL_OFFSET +
				       rows[row].session_slot * POOL_STRIDE +
				       (uint32_t)rows[row].position /
					       POOL_RATIO * ENTRY_BYTES,
			       sizeof(entry));
			expect(entry == expected_stream(
						(uint32_t)rows[row].token_id,
						(uint32_t)rows[row].position,
						true),
			       "the mirror holds the entry the row emitted");
		}
	}
	for (uint32_t card = 0; card < CARDS; card++) {
		expect(waited_ticket[card] >= last_ticket[card],
		       "a step ends when a wait saw every copy of every card finish");
	}
	expect(rewritten_uploads == 0,
	       "no staging byte an upload reads changes before a wait saw the upload finish");
	expect(memcmp(arena_of(1) + POOL_OFFSET,
		      arena_of(LAST_CARD) + POOL_OFFSET,
		      SESSIONS * POOL_STRIDE) == 0,
	       "the mirror is byte-equal to its owner's pool after the step");
	const uint32_t block_flags =
		row_count != 0 && (rows[0].flags & STEP_ROW_PADDED) != 0 ?
			STEP_BLOCK_PADDED :
			0u;
	for (uint32_t card = 0; card < CARDS; card++) {
		struct step_block block;
		memcpy(&block, arena_of(card) + STEP_OFFSET, sizeof(block));
		expect(block.row_count == row_count &&
			       block.flags == block_flags &&
			       memcmp(arena_of(card) + STEP_OFFSET +
					      STEP_BLOCK_BYTES_VALUE,
				      rows, row_count * sizeof(*rows)) == 0,
		       "every card holds the step area of the step");
	}
	/* the order of the work: a card's step area before its launches, a hand-off between its two cards' launches */
	int64_t last_launch[CARDS] = { -1, -1, -1 };
	int64_t first_launch[CARDS] = { -1, -1, -1 };
	int64_t step_area[CARDS] = { -1, -1, -1 };
	for (uint64_t index = work_before;
	     index < test_host_memory_backend_work_count(); index++) {
		struct test_host_memory_backend_work work = { .kind = 0 };
		expect(test_host_memory_backend_work_at(index, &work),
		       "the work is logged");
		const uint32_t card = work.card_index;
		if (work.kind == TEST_HOST_MEMORY_BACKEND_LAUNCH) {
			first_launch[card] = first_launch[card] < 0 ?
						     (int64_t)index :
						     first_launch[card];
			last_launch[card] = (int64_t)index;
			expect(work.launch.grid[0] == program_rows,
			       "the program of the fewest rows that holds the step runs");
		} else if (work.kind == TEST_HOST_MEMORY_BACKEND_UPLOAD &&
			   work.arena_byte_offset == STEP_OFFSET) {
			step_area[card] = (int64_t)index;
		} else if (work.kind == TEST_HOST_MEMORY_BACKEND_UPLOAD &&
			   work.arena_byte_offset == STREAMS_OFFSET) {
			expect(last_launch[card - 1u] >= 0 &&
				       first_launch[card] < 0,
			       "streams reach a card after the card before it ran and before it runs");
		}
	}
	for (uint32_t card = 0; card < CARDS; card++) {
		expect(step_area[card] >= 0 &&
			       step_area[card] < first_launch[card],
		       "each card has its step area before its first launch");
	}
}

/* Whether the card that samples holds these settings in slots 0 and 1, the bytes session_set was given. */
static bool settings_held(const struct step_session *slot_0,
			  const struct step_session *slot_1)
{
	unsigned char expected[SESSIONS][STEP_SESSION_BYTES];
	memcpy(expected[0], slot_0, sizeof(expected[0]));
	memcpy(expected[1], slot_1, sizeof(expected[1]));
	return memcmp(arena_of(LAST_CARD) + SESSION_OFFSET, expected,
		      sizeof(expected)) == 0;
}

static void test_steps(void)
{
	struct fixture fixture;
	context_text = "steps";
	const uint32_t staging_before =
		test_host_memory_backend_staging_block_count();
	fixture_build(&fixture, cards_open());
	for (uint32_t card = 0; card < CARDS; card++) {
		test_host_memory_backend_set_polls_before_an_upload_finishes(
			card, 1); /* unfinished until waited for */
	}
	const uint64_t work_before = test_host_memory_backend_work_count();
	struct step_driver *const driver = driver_of(&fixture, PROGRAMS);
	expect(test_host_memory_backend_work_count() == work_before,
	       "nothing reaches a card before the first step");
	const struct step_session settings = { .seed = SESSION_SEED,
					       .temperature =
						       SESSION_TEMPERATURE,
					       .top_p = SESSION_TOP_P };
	expect_status(step_session_set(driver, 1, &settings), STATUS_OK,
		      "settings of slot 1");
	/* three rows of session 0 from position 0, logits for the last */
	const struct step_row_record first_rows[FIRST_STEP_ROWS] = {
		{ 0, 1, 0, 0 },
		{ 1, POSITION_FOUR, 0, 0 },
		{ POSITION_TWO, VOCABULARY - 1u, 0, STEP_ROW_LOGITS }
	};
	step_expecting(driver, first_rows, FIRST_STEP_ROWS, ROW_CAPACITY);
	const struct step_session first_settings = { .seed = 0,
						     .temperature = 0.0f,
						     .top_p = 1.0f };
	expect(settings_held(&first_settings, &settings),
	       "the card that samples holds slot 0's first and slot 1's");
	/* one row: the program of one row; the settings are not sent again */
	const struct step_row_record one_row[1] = {
		{ POSITION_THREE, POSITION_TWO, 0, STEP_ROW_LOGITS }
	};
	const uint64_t uploads_before =
		test_host_memory_backend_upload_count(LAST_CARD);
	step_expecting(driver, one_row, 1, 1);
	expect(test_host_memory_backend_upload_count(LAST_CARD) -
			       uploads_before ==
		       UPLOADS_TO_THE_LAST_CARD,
	       "card 2 receives its step area, its n-gram rows, the streams and the pool: no settings");
	/* slot 0's settings set after a step reach the card with the next, slot 1's kept */
	const struct step_session later = { .seed = LATER_SEED,
					    .temperature = LATER_TEMPERATURE,
					    .top_p = LATER_TOP_P };
	expect_status(step_session_set(driver, 0, &later), STATUS_OK,
		      "settings of slot 0 after a step");
	/* two sessions: session 1 at 0 and 1, session 0 at 4 and 5: an entry each, one copy each */
	const struct step_row_record mixed[ROW_CAPACITY] = {
		{ 0, POSITION_THREE, 1, 0 },
		{ 1, POSITION_FOUR, 1, STEP_ROW_LOGITS },
		{ POSITION_FOUR, POSITION_THREE, 0, STEP_ROW_LOGITS },
		{ POSITION_FIVE, 1, 0, STEP_ROW_LOGITS }
	};
	const uint64_t mixed_work = test_host_memory_backend_work_count();
	step_expecting(driver, mixed, ROW_CAPACITY, ROW_CAPACITY);
	uint32_t pool_downloads = 0;
	for (uint64_t index = mixed_work;
	     index < test_host_memory_backend_work_count(); index++) {
		struct test_host_memory_backend_work work = { .kind = 0 };
		pool_downloads +=
			test_host_memory_backend_work_at(index, &work) &&
					work.kind ==
						TEST_HOST_MEMORY_BACKEND_DOWNLOAD &&
					work.arena_byte_offset >= POOL_OFFSET &&
					work.arena_byte_offset <
						POOL_OFFSET +
							SESSIONS * POOL_STRIDE ?
				1u :
				0u;
	}
	expect(pool_downloads == POOL_RUNS,
	       "the entries of a run of one session's rows go as one copy");
	expect(settings_held(&later, &settings),
	       "the card that samples holds slot 0's later settings and slot 1's first");
	/* session 0 begun again at position 0, then continued */
	const struct step_row_record again[AGAIN_ROWS] = {
		{ 0, POSITION_TWO, 0, 0 }, { 1, 1, 0, STEP_ROW_LOGITS }
	};
	step_expecting(driver, again, AGAIN_ROWS, ROW_CAPACITY);
	if (test_step_observe != NULL) {
		test_step_observe();
	}
	step_driver_destroy(driver);
	expect(test_host_memory_backend_staging_block_count() == staging_before,
	       "destroy frees the staging block: every copy was seen finished");
	cards_close();
}

#endif
#endif
