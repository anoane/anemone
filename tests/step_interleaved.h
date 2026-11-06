/* SPDX-License-Identifier: AGPL-3.0-or-later */
#ifndef TEST_STEP_INTERLEAVED_H
#define TEST_STEP_INTERLEAVED_H
#include <stddef.h>
#ifdef TEST_STEP_FIXTURE_IMPLEMENTATION
/*
 * step_interleaved.h: step_test's model with card 0 visited twice a step (model_seam.h's visits), for the step tests:
 *     S_EMBED  card 0: embed                S_LAYER card 0: a layer                (card 0's first visit)
 *     S_POOL   card 1: a layer that writes the pool entry of every emitting row (ratio 2): the front ends behind it
 *     S_REST   card 1: a layer                                                     (card 1's first visit: two runs)
 *     S_MIRROR card 0: a layer that reads card 0's mirror of the pool and the row's position     (its second visit)
 *     S_AGAIN  card 1: a layer (its second) S_LAST  card 2: a layer   S_SAMPLE card 2: the sampler   (its one visit)
 * The streams go on at every change of card; the pool's entries go from card 1 to card 0 behind S_POOL, which a front
 * step defers (card 0 runs the front before it and nothing of the front behind it: step.h), and to card 2, which it
 * does not (card 2 runs no front). The pool's session ranges: card 0's mirror first, the owner's, card 2's. Programs of
 * 1 and ROW_CAPACITY rows; card 0 reads both n-gram tables, card 2 table 1.
 *   - interleaved_eager_steps: the plan's steps one at a time: front steps leave card 0's mirror alone and fill card
 *     2's; a snapshot of a front-only session (whose first range is that mirror) equals the owner's entries and leaves
 *     the mirror synced; each step of other rows has the model's results, the mirrors the owner's bytes; the n-gram
 *     rows hold. Then destroy gives the staging back.
 */
#define INTERLEAVED_SEGMENTS 8u
#define INTERLEAVED_HAND_OFFS 6u
#define INTERLEAVED_FRONT 3u /* S_EMBED to S_POOL */
#define INTERLEAVED_STEPS 6u
#define INTERLEAVED_TOKEN_STRIDE 7u
#define INTERLEAVED_SNAPSHOT_BYTES 4096u
#define INTERLEAVED_FRONT_TOKENS \
	8u /* of session 0 before its first step of other rows */

enum interleaved_segment {
	S_EMBED,
	S_LAYER,
	S_POOL,
	S_REST,
	S_MIRROR,
	S_AGAIN,
	S_LAST,
	S_SAMPLE
};

struct interleaved {
	struct kernel_launch launches[PROGRAMS][CARDS][LAUNCHES_PER_CARD];
	struct program_card_record program_cards[PROGRAMS][CARDS];
	struct execution_program programs[PROGRAMS];
	const struct execution_program *pointers[PROGRAMS];
};

/* The segments, hand-offs and pool ranges, the same in both programs. */
static const struct segment_record interleaved_segments[INTERLEAVED_SEGMENTS] = {
	{ 0, S_EMBED, SEGMENT_NO_LAYER, 0, 1, 0 },
	{ 0, S_LAYER, SEGMENT_NO_LAYER, 1, 1, 0 },
	{ 1, S_POOL, SEGMENT_NO_LAYER, 0, 1, 0 },
	{ 1, S_REST, SEGMENT_NO_LAYER, 1, 1, 0 },
	{ 0, S_MIRROR, SEGMENT_NO_LAYER, 2, 1, 0 },
	{ 1, S_AGAIN, SEGMENT_NO_LAYER, 2, 1, 0 },
	{ 2, S_LAST, SEGMENT_NO_LAYER, 0, 1, 0 },
	{ 2, S_SAMPLE, SEGMENT_NO_LAYER, 1, 1, 0 }
};
static const struct hand_off_record
	interleaved_hand_offs[INTERLEAVED_HAND_OFFS] = {
		{ STREAMS_OFFSET, STREAMS_OFFSET, 0, sizeof(uint64_t), 0, 0, 1,
		  S_LAYER, 0 },
		{ POOL_OFFSET, POOL_OFFSET, POOL_STRIDE, ENTRY_BYTES,
		  POOL_RATIO, 1, 0, S_POOL, 0 },
		{ POOL_OFFSET, POOL_OFFSET, POOL_STRIDE, ENTRY_BYTES,
		  POOL_RATIO, 1, LAST_CARD, S_POOL, 0 },
		{ STREAMS_OFFSET, STREAMS_OFFSET, 0, sizeof(uint64_t), 0, 1, 0,
		  S_REST, 0 },
		{ STREAMS_OFFSET, STREAMS_OFFSET, 0, sizeof(uint64_t), 0, 0, 1,
		  S_MIRROR, 0 },
		{ STREAMS_OFFSET, STREAMS_OFFSET, 0, sizeof(uint64_t), 0, 1,
		  LAST_CARD, S_AGAIN, 0 }
	};
static const struct session_range interleaved_ranges[CARDS] = {
	{ POOL_OFFSET, POOL_STRIDE, CONTEXT / POOL_RATIO, ENTRY_BYTES,
	  POOL_RATIO, 0, 0 },
	{ POOL_OFFSET, POOL_STRIDE, CONTEXT / POOL_RATIO, ENTRY_BYTES,
	  POOL_RATIO, 1, 0 },
	{ POOL_OFFSET, POOL_STRIDE, CONTEXT / POOL_RATIO, ENTRY_BYTES,
	  POOL_RATIO, LAST_CARD, 0 }
};

/* The steps the tests run: a session, its first position, the rows, whether front rows (else the last asks logits). */
static const struct interleaved_step {
	uint32_t session, first_position, row_count;
	bool is_front;
} interleaved_plan[INTERLEAVED_STEPS] = {
	{ 0, 0, ROW_CAPACITY, true },
	{ 0, ROW_CAPACITY, ROW_CAPACITY, true },
	{ 0, INTERLEAVED_FRONT_TOKENS, ROW_CAPACITY, false },
	{ 1, 0, ROW_CAPACITY, true },
	{ 1, ROW_CAPACITY, ROW_CAPACITY, false },
	{ 0, INTERLEAVED_FRONT_TOKENS + ROW_CAPACITY, 1, false }
};

static void interleaved_build(struct interleaved *made, uint32_t kernel)
{
	static const uint32_t operations[CARDS][LAUNCHES_PER_CARD] = {
		{ OPERATION_EMBED, OPERATION_LAYER, OPERATION_MIRROR_LAYER },
		{ OPERATION_LAYER, OPERATION_LAYER, OPERATION_LAYER },
		{ OPERATION_LAYER, OPERATION_SAMPLE }
	};
	static const uint32_t launch_counts[CARDS] = { 3, 3, 2 };
	const struct model_limits limits = { .context_token_limit = CONTEXT,
					     .row_capacity = LIMIT_ROWS,
					     .session_capacity = SESSIONS,
					     .state_width = STATE_WIDTH_16 };
	memset(made, 0, sizeof(*made));
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
					&made->launches[program][card][launch],
					kernel, row_count, 1, 1, 0, &record,
					sizeof(record));
			}
			made->program_cards[program][card] =
				(struct program_card_record){
					.launches =
						made->launches[program][card],
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
		made->programs[program] = (struct execution_program){
			.cards = made->program_cards[program],
			.segments = interleaved_segments,
			.hand_offs = interleaved_hand_offs,
			.limits = limits,
			.card_count = CARDS,
			.segment_count = INTERLEAVED_SEGMENTS,
			.draft_segment_first = INTERLEAVED_SEGMENTS,
			.hand_off_count = INTERLEAVED_HAND_OFFS,
			.row_count = row_count,
			.result_card = LAST_CARD,
			.session_ranges = interleaved_ranges,
			.session_range_count = CARDS,
			.front_segment_count = INTERLEAVED_FRONT
		};
		made->pointers[program] = &made->programs[program];
	}
}

/* The rows of a step of the plan, their tokens written into the test's history. */
static uint32_t interleaved_rows(const struct interleaved_step *step,
				 struct step_row_record *rows)
{
	for (uint32_t row = 0; row < step->row_count; row++) {
		const uint32_t position = step->first_position + row;
		rows[row] = (struct step_row_record){
			(int32_t)position,
			(int32_t)((position * INTERLEAVED_TOKEN_STRIDE +
				   step->session + 1u) %
				  VOCABULARY),
			step->session,
			step->is_front		    ? STEP_ROW_FRONT :
			row + 1u == step->row_count ? STEP_ROW_LOGITS :
						      0
		};
		history[step->session][position] = (uint32_t)rows[row].token_id;
	}
	return step->row_count;
}

/* The pool entry the row at position of a session emitted: its stream behind S_POOL. */
static uint64_t interleaved_entry(uint32_t slot, uint32_t position)
{
	const uint64_t embedded =
		(uint64_t)history[slot][position] * EMBED_MULTIPLIER +
		position + 1u;
	return stream_after_layer(stream_after_layer(embedded, 0), 1);
}

/* Whether a step's results are the interleaved model's: a row with logits the token and logit of its stream before the
 * sampler (every pool entry it sees at S_MIRROR, card 0's mirror), any other none. */
static bool interleaved_results_hold(const struct step_row_record *rows,
				     uint32_t row_count,
				     const struct step_row_result *results)
{
	bool holds = true;
	for (uint32_t row = 0; row < row_count; row++) {
		const uint32_t slot = rows[row].session_slot,
			       position = (uint32_t)rows[row].position;
		const uint32_t entries = (position + 1u) / POOL_RATIO;
		uint64_t entry = 0;
		for (uint32_t index = 0; index < entries; index++) {
			entry ^= interleaved_entry(
				slot, (index + 1u) * POOL_RATIO - 1u);
		}
		const uint64_t mirrored =
			(stream_after_layer(
				 stream_after_layer(
					 interleaved_entry(slot, position), 1),
				 0) ^
			 entry) +
			position;
		const uint64_t stream = stream_after_layer(
			stream_after_layer(mirrored, 1), LAST_CARD);
		const bool has_logits = (rows[row].flags & STEP_ROW_LOGITS) !=
					0;
		holds = holds &&
			results[row].token_id ==
				(has_logits ? (int32_t)(stream % VOCABULARY) :
					      STEP_NO_TOKEN) &&
			results[row].logit ==
				(has_logits ? (float)(stream % LOGIT_MODULUS) :
					      0.0f);
	}
	return holds;
}

/* Whether a card's pool range holds the owner's (card 1's) entries of a session below entry_count. */
static bool interleaved_mirror_holds(uint32_t card, uint32_t slot,
				     uint32_t entry_count)
{
	return memcmp(arena_of(card) + POOL_OFFSET + slot * POOL_STRIDE,
		      arena_of(1) + POOL_OFFSET + slot * POOL_STRIDE,
		      entry_count * ENTRY_BYTES) == 0;
}

/* Whether a card's pool range of a session is all zero below entry_count: nothing went up into it. */
static bool interleaved_mirror_is_empty(uint32_t card, uint32_t slot,
					uint32_t entry_count)
{
	bool is_empty = true;
	for (uint32_t byte = 0; byte < entry_count * ENTRY_BYTES; byte++) {
		is_empty = is_empty &&
			   arena_of(card)[POOL_OFFSET + slot * POOL_STRIDE +
					  byte] == 0;
	}
	return is_empty;
}

static struct step_driver *
interleaved_driver(const struct backend_operations *backend,
		   const struct interleaved *made)
{
	struct step_driver *driver = NULL;
	expect_status(step_driver_create(backend, cards, made->pointers,
					 PROGRAMS, engram, WAIT_LIMIT, &driver),
		      STATUS_OK, "the interleaved driver is made");
	if (test_step_use_graphs) {
		expect_status(step_graphs_prepare(driver), STATUS_OK,
			      "graphs prepared at startup");
	}
	return driver;
}

/* A snapshot's bytes: what a save wrote. */
static struct interleaved_snapshot {
	unsigned char bytes[INTERLEAVED_SNAPSHOT_BYTES];
	uint64_t byte_count;
} interleaved_saved;

static bool interleaved_snapshot_write(void *context, unsigned char *bytes,
				       uint64_t byte_count)
{
	struct interleaved_snapshot *const saved = context;
	const bool fits = byte_count <=
			  INTERLEAVED_SNAPSHOT_BYTES - saved->byte_count;
	if (fits) {
		memcpy(saved->bytes + saved->byte_count, bytes, byte_count);
		saved->byte_count += byte_count;
	}
	return fits;
}

static void interleaved_eager_steps(void)
{
	struct interleaved made;
	struct step_row_record rows[ROW_CAPACITY];
	struct step_row_result results[ROW_CAPACITY];
	context_text = "interleaved steps, one at a time";
	const uint32_t staging_before =
		test_host_memory_backend_staging_block_count();
	interleaved_build(&made, cards_open());
	for (uint32_t card = 0; card < CARDS; card++) {
		test_host_memory_backend_set_polls_before_an_upload_finishes(
			card, 1); /* unfinished until waited for */
	}
	struct step_driver *const driver =
		interleaved_driver(&checked_backend, &made);
	for (uint32_t index = 0; index < INTERLEAVED_STEPS; index++) {
		const struct interleaved_step *const step =
			&interleaved_plan[index];
		const uint32_t row_count = interleaved_rows(step, rows);
		const uint32_t entries =
			(step->first_position + row_count) / POOL_RATIO;
		expect_status(step_run_entry(driver, rows, row_count, results),
			      STATUS_OK, "an interleaved step runs");
		expect(interleaved_results_hold(rows, row_count, results),
		       "each row has the interleaved model's token, or none");
		expect(interleaved_mirror_holds(LAST_CARD, step->session,
						entries),
		       "the mirror of the card that runs no front takes each step's entries");
		expect(step->is_front ?
			       interleaved_mirror_is_empty(0, step->session,
							   entries) :
			       interleaved_mirror_holds(0, step->session,
							entries),
		       "a front step leaves the deferred mirror; a step of other rows syncs it first");
		for (uint32_t row = 0; row < row_count && !step->is_front;
		     row++) {
			expect(engram_rows_hold(&rows[row], row),
			       "the n-gram rows of a card visited twice are the step's");
		}
		expect(rewritten_uploads == 0,
		       "no staging byte an upload reads changes before a wait saw the upload finish");
		if (index + 1u == INTERLEAVED_FRONT_TOKENS / ROW_CAPACITY) {
			const uint32_t entry_count =
				INTERLEAVED_FRONT_TOKENS / POOL_RATIO;
			interleaved_saved.byte_count = 0;
			expect_status(
				step_session_save(driver, 0,
						  INTERLEAVED_FRONT_TOKENS,
						  interleaved_snapshot_write,
						  &interleaved_saved),
				STATUS_OK, "a front-only session is saved");
			expect(interleaved_saved.byte_count ==
					       entry_count * ENTRY_BYTES &&
				       memcmp(interleaved_saved.bytes,
					      arena_of(1) + POOL_OFFSET,
					      entry_count * ENTRY_BYTES) == 0 &&
				       interleaved_mirror_holds(0, 0,
								entry_count),
			       "its first range, a mirror, is synced before it is read");
		}
	}
	if (test_step_observe != NULL) {
		test_step_observe();
	}
	step_driver_destroy(driver);
	expect(test_host_memory_backend_staging_block_count() == staging_before,
	       "destroy frees the staging blocks: the step's and the session block");
	cards_close();
}

#endif
#endif
