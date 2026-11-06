/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * serving_fixture.c: the engine and the tables of serving_fixture.h. Test support: no part of the engine.
 */
#include "serving_fixture.h"

#include <stdatomic.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "anemone/backend_records.h"
#include "anemone/model_layout.h"
#include "anemone/step.h"
#include "anemone/tokenizer.h"
#include "host_memory_backend.h"
#include "prepare_common.h"

#define PROGRAMS 2u
#define CARDS 2u                                /* in the interleaved programs: card 0 visited twice */
#define TARGET_SEGMENTS 3u                      /* of the interleaved programs */
#define SEGMENTS (TARGET_SEGMENTS + 1u)          /* optional device draft on card 0 */
#define SESSION_RANGES 2u
#define DRAFT_RANGE_IDENTITY 160u               /* first draft ring: 4 * layer 40 */
#define HAND_OFFS 2u
#define CHECKPOINT_OFFSET (STATE_OFFSET + TEST_SERVING_MOST_SLOTS * TEST_SERVING_STATE_POSITIONS)
#define CHECKPOINT_ROWS (MODEL_WINDOW - 1u)
#define DRAFT_STATE_OFFSET (CHECKPOINT_OFFSET + TEST_SERVING_MOST_SLOTS * CHECKPOINT_ROWS)
#define ARENA_BYTES (DRAFT_STATE_OFFSET + TEST_SERVING_MOST_SLOTS * TEST_SERVING_STATE_POSITIONS)
#define STEP_OFFSET 0u
#define SESSION_OFFSET 1024u
#define RESULT_OFFSET 2048u
#define ROUND_GAP_BYTES 256u                    /* target result's reserved block before the round */
#define ROUND_OFFSET (RESULT_OFFSET + ROUND_GAP_BYTES)
#define ECHO_OFFSET 4096u                       /* per row: its token plus its position, card 0's then card 1's */
#define BACK_OFFSET 8192u                       /* card 0's: what card 1 handed back */
#define STATE_OFFSET 32768u                     /* a byte a position: the token the model read there */
#define WAIT_LIMIT 1000000000ull
#define NANOSECONDS_PER_SECOND 1000000000ull
#define ADDED_TOKENS 3u
#define TARGET_LAUNCHES 2u                       /* with a front: its launch, then the model's */
#define LAUNCHES (TARGET_LAUNCHES + 1u)           /* optional device draft behind the target */
#define FRONT_VALUE 1u                          /* the fill record's value of a front's launch */
#define ECHO_VALUE 2u                           /* card 0's first visit: each row's token plus position to ECHO */
#define PASS_VALUE 3u                           /* card 1: holds ECHO to its own rows, adds one */
#define DRAFT_VALUE 4u                          /* the device draft's segment */
#define REASONING_END "</think>"
#define REASONING_BEGIN "<think>"
#define END_TEXT "<END>"
#define BYTE_VALUES 256u
#define HASH_MULTIPLIER 0x9E3779B97F4A7C15ull         /* the function model's mixing */
#define HASH_SHIFT 29u
#define DRAW_SHIFT 32u                                /* a draw reads the hash's high half */
#define NGRAM_MOST 4u                                 /* the drafter's n-grams (main.c's defaults) */
#define NGRAM_LEAST 2u

static struct device_card *cards[CARDS];
static struct kernel_launch launches[PROGRAMS][CARDS][LAUNCHES];
static struct segment_record segments[PROGRAMS][SEGMENTS];
static uint32_t cone_rows;                              /* test_serving_set_cone's, for the next open */
static bool interleaved;                               /* test_serving_set_interleaved's, for the next open */
static _Atomic uint32_t fault_position = UINT32_MAX;    /* the model faults on a row there */
static _Atomic uint64_t rows_seen;
static _Atomic uint64_t front_rows_seen;
static struct program_card_record program_cards[PROGRAMS][CARDS];
/* card 0's rows to card 1 behind its first visit, card 1's back to card 0 behind its one */
static const struct hand_off_record hand_offs[HAND_OFFS] = {
	{ ECHO_OFFSET, ECHO_OFFSET, 0, sizeof(uint64_t), 0, 0, 1, 0, 0 },
	{ ECHO_OFFSET, BACK_OFFSET, 0, sizeof(uint64_t), 0, 1, 0, 1, 0 }
};
static struct execution_program programs[PROGRAMS];
static const struct execution_program *pointers[PROGRAMS];
static struct session_range session_ranges
	[SESSION_RANGES]; /* target state and the optional draft ring */
static struct window_checkpoint checkpoint_range;
static struct step_driver *driver;
static struct inference_engine *engine;
static const uint32_t *const *model_scripts;            /* set before the engine's thread starts */
static uint32_t model_script_count;
static uint32_t script_cursor;                          /* the stepping thread's alone */
static int32_t last_position = -1;                      /* the last step's last row: the stepping thread's */
static _Atomic uint64_t step_delay_nanoseconds;
static _Atomic uint64_t steps;
static uint32_t slot_count = 1, pipeline_depth = 1, engine_rows = TEST_SERVING_ROWS,
                draft_positions;                        /* of the next open */
static bool is_function_model;                          /* set before an open, read by the card's thread */
static uint32_t copy_period;
static uint32_t function_end_position = UINT32_MAX;
static uint32_t ring_positions = TEST_SERVING_STATE_POSITIONS, model_window = TEST_SERVING_WINDOW;
static bool device_draft;
static uint32_t device_block = MODEL_DRAFT_BLOCK;
static uint32_t device_rejected_position = MODEL_DRAFT_BLOCK;
static uint32_t device_low_confidence = MODEL_DRAFT_BLOCK;
static bool device_wrong_round;
static uint32_t configured_prefill_first, configured_prefill_order, configured_prefill_order_wait;
static const struct draft_policy_record *configured_policy;
static uint32_t configured_lookup_min;
static uint32_t configured_learned_acceptance;
static uint32_t configured_session_acceptance;
static uint32_t configured_start_economic;
static uint32_t configured_ignore_eos;
static _Atomic uint64_t device_rounds;
static _Atomic uint32_t device_rejections;

/* The function model's draw for a row at position of a session with this state and these settings (the header). */
static uint32_t function_drawn(const unsigned char *state,
			       const struct step_session *settings,
			       uint32_t position)
{
    if (position == function_end_position) {
        return TEST_SERVING_END_TOKEN;
    }
    if (copy_period != 0) {
        return position + 1u >= copy_period ? state[(position + 1u - copy_period) % ring_positions] : 0u;
    }
    uint64_t hash = ((settings->temperature > 0.0f ? settings->seed + 1u : 0u) ^ position) * HASH_MULTIPLIER;
    for (uint32_t back = 0; back < model_window; back++) {
        hash = (hash ^ (position >= back ? state[(position - back) % ring_positions] : 0u)) * HASH_MULTIPLIER;
        hash ^= hash >> HASH_SHIFT;
    }
    const uint32_t drawn = (uint32_t)(hash >> DRAW_SHIFT);
    return drawn % TEST_SERVING_END_ODDS == 0 ? TEST_SERVING_END_TOKEN : drawn / TEST_SERVING_END_ODDS % BYTE_VALUES;
}

/* The interleaved programs' rows at offset: each row's token plus its position plus added, written where writes,
 * else held to the bytes there: false where a visit reads what another step's visit wrote. */
static bool echo_rows(const struct step_block *block, unsigned char *arena,
		      uint64_t offset, uint64_t added, bool writes)
{
    bool holds = true;
    for (uint32_t row = 0; row < block->row_count; row++) {
	    struct step_row_record step_row;
	    uint64_t value = 0;
	    memcpy(&step_row,
		   arena + STEP_OFFSET + STEP_BLOCK_BYTES_VALUE +
			   (size_t)row * STEP_ROW_BYTES_VALUE,
		   sizeof(step_row));
	    memcpy(&value, arena + offset + (size_t)row * sizeof(value),
		   sizeof(value));
	    const uint64_t expected = (uint64_t)step_row.token_id +
				      (uint64_t)step_row.position + added;
	    holds = holds && (writes || value == expected);
	    if (writes) {
		    memcpy(arena + offset + (size_t)row * sizeof(expected),
			   &expected, sizeof(expected));
	    }
    }
    return holds;
}

/* A device stand-in: keyed match from the target's results, then function-model proposals from that accepted anchor.
 * Its persistent ring is separate from the target's and is carried by snapshots, as the real draft's rings are. */
static bool device_round_written(const struct step_block *block,
				 unsigned char *arena)
{
	struct step_row_record rows[TEST_SERVING_MOST_ROWS];
	struct step_row_result results[TEST_SERVING_MOST_ROWS];
	unsigned char state[TEST_SERVING_STATE_POSITIONS];
	memcpy(rows, arena + STEP_OFFSET + STEP_BLOCK_BYTES_VALUE,
	       block->row_count * sizeof(*rows));
	memcpy(results, arena + RESULT_OFFSET + STEP_RESULT_BYTES,
	       block->row_count * sizeof(*results));
	for (uint32_t anchor = 0; anchor < block->row_count; anchor++) {
		if ((rows[anchor].flags & STEP_ROW_DRAFTS) == 0) {
			continue;
		}
		const uint32_t slot = rows[anchor].session_slot;
		uint32_t emitted = 1, proposed = 0;
		while (anchor + proposed + 1u < block->row_count &&
		       rows[anchor + proposed + 1u].session_slot == slot &&
		       (rows[anchor + proposed + 1u].flags &
			STEP_ROW_PROPOSED) != 0) {
			proposed++;
		}
		while (emitted <= proposed &&
		       results[anchor + emitted - 1u].token_id !=
			       (int32_t)TEST_SERVING_END_TOKEN &&
		       results[anchor + emitted - 1u].token_id ==
			       rows[anchor + emitted].token_id) {
			emitted++;
		}
		if (emitted <= proposed &&
		    results[anchor + emitted - 1u].token_id !=
			    (int32_t)TEST_SERVING_END_TOKEN) {
			atomic_fetch_or(&device_rejections,
					1u << (emitted - 1u));
		}
		emitted = device_wrong_round && proposed != 0 ?
				  (emitted == 1u ? TARGET_LAUNCHES : 1u) :
				  emitted;
		struct step_round round = {
			.sequence = block->sequence,
			.session_slot = slot,
			.emitted_count = emitted,
			.anchor_position =
				rows[anchor].position + (int32_t)emitted,
			.anchor_token = results[anchor + emitted - 1u].token_id,
			.proposal_count = device_block
		};
		struct step_session settings;
		memcpy(&settings,
		       arena + SESSION_OFFSET +
			       (size_t)slot * STEP_SESSION_BYTES,
		       sizeof(settings));
		memcpy(state,
		       arena + STATE_OFFSET +
			       (size_t)slot * TEST_SERVING_STATE_POSITIONS,
		       sizeof(state));
		uint32_t token = (uint32_t)round.anchor_token;
		for (uint32_t position = 0; position < device_block;
		     position++) {
			const uint32_t absolute =
				(uint32_t)round.anchor_position + position;
			state[absolute % ring_positions] = (unsigned char)token;
			arena[DRAFT_STATE_OFFSET +
			      (size_t)slot * TEST_SERVING_STATE_POSITIONS +
			      absolute % ring_positions] = (unsigned char)token;
			token = function_drawn(state, &settings, absolute);
			token = position == device_rejected_position ?
					(token + 1u) % BYTE_VALUES :
					token;
			round.proposals[position] = (int32_t)token;
			round.confidence_logits[position] =
				position == device_low_confidence ? -1.0f :
								    1.0f;
		}
		memcpy(arena + ROUND_OFFSET, &round, sizeof(round));
		atomic_fetch_add(&device_rounds, 1u);
		return true;
	}
    return false;
}

static bool model_stand_in(const struct kernel_launch *launch,
			   uint32_t card_index, unsigned char *arena,
			   uint64_t arena_byte_count, void *context)
{
	struct step_block block;
	struct step_session settings;
	struct step_row_record first;
	struct fill_record record;
	memcpy(&record, launch->argument, sizeof(record));
	const uint64_t delay = atomic_load(&step_delay_nanoseconds);
	const struct timespec pause = {
		(time_t)(delay / NANOSECONDS_PER_SECOND),
		(long)(delay % NANOSECONDS_PER_SECOND)
	};
	(void)card_index;
	(void)context;
	if (delay != 0 && nanosleep(&pause, NULL) != 0) {
		return false;
	}
    memcpy(&block, arena + STEP_OFFSET, sizeof(block));
    memcpy(&settings, arena + SESSION_OFFSET, sizeof(settings));
    memcpy(&first, arena + STEP_OFFSET + STEP_BLOCK_BYTES_VALUE, sizeof(first));
    if (arena_byte_count != ARENA_BYTES || block.row_count == 0 || block.row_count > engine_rows ||
        model_script_count == 0) {
        return false;
    }
    if (record.value == DRAFT_VALUE) {
        return device_round_written(&block, arena);
    }
    if (record.value == ECHO_VALUE || record.value == PASS_VALUE) {     /* card 0 puts, card 1 holds and adds */
        return record.value == ECHO_VALUE ? echo_rows(&block, arena, ECHO_OFFSET, 0, true)
                                          : echo_rows(&block, arena, ECHO_OFFSET, 0, false) &&
                                                echo_rows(&block, arena, ECHO_OFFSET, 1u, true);
    }
    if (interleaved && !echo_rows(&block, arena, BACK_OFFSET, 1u, false)) {
        return false;                                   /* card 0's visit again: card 1's rows of its step */
    }
    const uint32_t *const script = model_scripts[settings.seed % model_script_count];
    const struct step_result answer = { .sequence = block.sequence,
					.row_count = block.row_count };
    const bool is_front = record.value == FRONT_VALUE;
    if (is_front || cone_rows == 0) {   /* a step's first launch: a command begins where a step goes back, at 0 or
                                         * behind a kept prefix of the session */
        script_cursor = first.position == 0 || first.position <= last_position ? 0 : script_cursor;
	struct step_row_record last;
	memcpy(&last,
	       arena + STEP_OFFSET + STEP_BLOCK_BYTES_VALUE +
		       (size_t)(block.row_count - 1u) * STEP_ROW_BYTES_VALUE,
	       sizeof(last));
	last_position = last.position;
    }
    if (is_front) {
        atomic_fetch_add(&front_rows_seen, block.row_count);
        return true;
    }
    atomic_fetch_add(&rows_seen, block.row_count);
    memcpy(arena + RESULT_OFFSET, &answer, sizeof(answer));
    for (uint32_t row = 0; row < block.row_count; row++) {
	    struct step_row_record step_row;
	    struct step_session own;
	    memcpy(&step_row,
		   arena + STEP_OFFSET + STEP_BLOCK_BYTES_VALUE +
			   (size_t)row * STEP_ROW_BYTES_VALUE,
		   sizeof(step_row));
	    memcpy(&own,
		   arena + SESSION_OFFSET +
			   (size_t)step_row.session_slot * STEP_SESSION_BYTES,
		   sizeof(own));
	    const bool draws = (step_row.flags & STEP_ROW_LOGITS) != 0;
	    if ((uint32_t)step_row.position == atomic_load(&fault_position)) {
		    return false; /* the model faults: the card fails */
	    }
        unsigned char *const state = arena + STATE_OFFSET + (size_t)step_row.session_slot *
                                                                TEST_SERVING_STATE_POSITIONS;
        state[(uint32_t)step_row.position % ring_positions] = (unsigned char)step_row.token_id;
        uint32_t token = TEST_SERVING_END_TOKEN;
        for (uint32_t index = 0; !is_function_model && index <= script_cursor &&
                                 script[index] != TEST_SERVING_END_TOKEN; index++) {
            token = index == script_cursor ? script[index] : token;      /* the script ends at its end token */
        }
        token = is_function_model && draws ? function_drawn(state, &own, (uint32_t)step_row.position) : token;
	const struct step_row_result result = {
		.token_id = draws ? (int32_t)token : STEP_NO_TOKEN,
		.logit = draws ? (float)step_row.position : 0.0f
	};
	script_cursor += draws ? 1u : 0u;
	memcpy(arena + RESULT_OFFSET + STEP_RESULT_BYTES +
		       (size_t)row * STEP_ROW_RESULT_BYTES,
	       &result, sizeof(result));
    }
    atomic_fetch_add(&steps, 1u);
    return true;
}

result_status test_serving_open(uint32_t context_tokens,
				const uint32_t *const *scripts,
				uint32_t script_count,
				struct inference_engine **engine_out)
{
	const struct model_limits limits = {
		.context_token_limit = context_tokens,
		.row_capacity = engine_rows,
		.session_capacity = slot_count,
		.state_width = STATE_WIDTH_16,
		.draft_positions = device_draft ? device_block : 0
	};
	const struct engine_description_record description = {
		.model_name = TEST_SERVING_MODEL,
		.session_slot_count = slot_count,
		.context_token_limit = context_tokens,
		.vocabulary_size = TEST_SERVING_VOCABULARY,
		.end_token_id = TEST_SERVING_END_TOKEN,
		.front_steps = cone_rows != 0 ? 1u : 0u,
		.speculative_depth = draft_positions,
		.prefill_first = configured_prefill_first,
		.prefill_order = configured_prefill_order,
		.prefill_order_wait = configured_prefill_order_wait,
		.lookup_ngram_most = NGRAM_MOST,
		.lookup_ngram_least = NGRAM_LEAST,
		.draft_policy = configured_policy,
		.hybrid_lookup_min = configured_lookup_min,
		.draft_learned_acceptance = configured_learned_acceptance,
		.draft_session_acceptance = configured_session_acceptance,
		.draft_start_economic = configured_start_economic,
		.ignore_eos = configured_ignore_eos
	};
	const struct fill_record records[TARGET_LAUNCHES] = {
		{ .value = FRONT_VALUE }, { 0 }
	};
	const struct fill_record draft_record = { .value = DRAFT_VALUE };
	const struct fill_record echo_records[CARDS][LAUNCHES] = {
		{ { .value = ECHO_VALUE }, { 0 } }, { { .value = PASS_VALUE } }
	};
	/* one card: a front's launch (with a cone) and the model's; interleaved: card 0, card 1, card 0 again */
	const bool is_interleaved = interleaved;
	const uint32_t card_count = is_interleaved ? CARDS : 1u,
		       launch_count = cone_rows != 0 ? TARGET_LAUNCHES : 1u;
	static const uint32_t segment_cards[SEGMENTS] = { 0, 1, 0, 0 },
			      segment_launches[SEGMENTS] = { 0, 0, 1,
							     TARGET_LAUNCHES };
	uint32_t kernel = 0;
	model_scripts = scripts;
	model_script_count = script_count;
	script_cursor = 0;
	last_position = -1;
	atomic_store(&step_delay_nanoseconds, 0);
	atomic_store(&rows_seen, 0);
	atomic_store(&front_rows_seen, 0);
	atomic_store(&fault_position, UINT32_MAX);
	atomic_store(&device_rounds, 0);
	atomic_store(&device_rejections, 0);
	session_ranges[0] =
		(struct session_range){ STATE_OFFSET,
					TEST_SERVING_STATE_POSITIONS,
					context_tokens,
					1,
					1,
					0,
					0 };
	session_ranges[1] =
		(struct session_range){ DRAFT_STATE_OFFSET,
					TEST_SERVING_STATE_POSITIONS,
					context_tokens,
					1,
					1,
					0,
					DRAFT_RANGE_IDENTITY };
	checkpoint_range =
		(struct window_checkpoint){ STATE_OFFSET,
					    CHECKPOINT_OFFSET,
					    TEST_SERVING_STATE_POSITIONS,
					    ring_positions,
					    1,
					    model_window - 1u,
					    0 };
	test_host_memory_backend_reset(card_count, ARENA_BYTES);
	result_status status = STATUS_OK;
	for (uint32_t card = 0; card < card_count; card++) {
		test_host_memory_backend_set_arithmetic_class(
			card, ARITHMETIC_CLASS_SM_80);
		status = status != STATUS_OK ?
				 status :
				 test_host_memory_backend.card_open(
					 card, ARENA_BYTES, &cards[card]);
		status = status != STATUS_OK ?
				 status :
				 test_host_memory_backend.kernel_prepare(
					 cards[card], KERNEL_FILE_FILL,
					 KERNEL_FILL_WORDS, FILL_RECORD_BYTES,
					 &kernel);
	}
	status = status != STATUS_OK ?
			 status :
			 test_host_memory_backend_set_stand_in(
				 KERNEL_FILE_FILL, KERNEL_FILL_WORDS,
				 model_stand_in, NULL);
	for (uint32_t program = 0; program < PROGRAMS; program++) {
		const uint32_t row_count = program == 0 ? 1u : engine_rows;
		for (uint32_t card = 0; card < card_count; card++) {
			const uint32_t target_launches =
				is_interleaved ? TARGET_LAUNCHES - card :
						 launch_count;
			const uint32_t card_launches =
				target_launches +
				(card == 0 && device_draft ? 1u : 0u);
			for (uint32_t launch = 0; launch < card_launches;
			     launch++) { /* the model's last */
				const struct fill_record *const record =
					launch >= target_launches ?
						&draft_record :
					is_interleaved ?
						&echo_records[card][launch] :
						&records[TARGET_LAUNCHES -
							 launch_count + launch];
				prepare_launch_fill(
					&launches[program][card][launch],
					kernel, row_count, 1, 1, 0, record,
					sizeof(*record));
			}
			program_cards[program][card] =
				(struct program_card_record){
					.launches = launches[program][card],
					.arena_byte_count = ARENA_BYTES,
					.step_offset = STEP_OFFSET,
					.session_offset =
						card == 0 ? SESSION_OFFSET :
							    PROGRAM_NOWHERE,
					.result_offset =
						card == 0 ? RESULT_OFFSET :
							    PROGRAM_NOWHERE,
					.round_offset =
						card == 0 && device_draft ?
							ROUND_OFFSET :
							PROGRAM_NOWHERE,
					.logits_offset = PROGRAM_NOWHERE,
					.streams_offset = PROGRAM_NOWHERE,
					.carried_pre_offset = PROGRAM_NOWHERE,
					.engram_rows_offset = PROGRAM_NOWHERE,
					.launch_count = card_launches
				};
		}
		const uint32_t target_segments =
			is_interleaved ? TARGET_SEGMENTS : launch_count;
		const uint32_t segment_count =
			target_segments + (device_draft ? 1u : 0u);
		for (uint32_t segment = 0; segment < segment_count; segment++) {
			segments[program][segment] = (struct segment_record){
				.card = is_interleaved ?
						segment_cards[segment] :
						0,
				.stage = segment,
				.layer = SEGMENT_NO_LAYER,
				.first_launch =
					is_interleaved ?
						segment_launches[segment] :
						segment,
				.launch_count = 1
			};
		}
		programs[program] = (struct execution_program){
			.cards = program_cards[program],
			.segments = segments[program],
			.hand_offs = is_interleaved ? hand_offs : NULL,
			.limits = limits,
			.card_count = card_count,
			.segment_count = segment_count,
			.hand_off_count = is_interleaved ? HAND_OFFS : 0,
			.row_count = row_count,
			.session_ranges = session_ranges,
			.session_range_count =
				context_tokens > TEST_SERVING_STATE_POSITIONS ?
					0u :
				device_draft ? SESSION_RANGES :
					       1u,
			.front_segment_count =
				is_interleaved ? 0 : launch_count - 1u,
			.cone_rows = cone_rows,
			.draft_segment_first = target_segments,
			.checkpoints = &checkpoint_range,
			.checkpoint_count =
				ring_positions == MODEL_RING_SLOTS &&
						model_window == MODEL_WINDOW ?
					1u :
					0u
		};
		pointers[program] = &programs[program];
	}
	status = status != STATUS_OK ?
			 status :
			 step_driver_create(&test_host_memory_backend, cards,
					    pointers,
					    engine_rows == 1 ? 1u : PROGRAMS,
					    NULL, WAIT_LIMIT, &driver);
	status = status != STATUS_OK || pipeline_depth == 1 ?
			 status :
			 step_pipeline_start(driver, pipeline_depth);
	status = status != STATUS_OK ? status :
				       engine_create(driver, &description,
						     engine_rows, &engine);
	if (status != STATUS_OK) {
		test_serving_close();
		return status;
	}
    *engine_out = engine;
    return STATUS_OK;
}

void test_serving_close(void)
{
	engine_destroy(engine);
	step_driver_destroy(driver);
	for (uint32_t card = 0; card < CARDS; card++) {
		test_host_memory_backend.card_close(cards[card], WAIT_LIMIT);
		cards[card] = NULL;
	}
    engine = NULL;
    driver = NULL;
    atomic_store(&steps, 0);
}

void test_serving_set_interleaved(uint32_t depth)
{
    interleaved = depth != 0;
    pipeline_depth = depth != 0 ? depth : 1u;
}

void test_serving_set_fault_position(uint32_t position)
{
    atomic_store(&fault_position, position);
}

void test_serving_set_cone(uint32_t rows)
{
    cone_rows = rows;
}

void test_serving_set_engine(uint32_t slots, uint32_t depth, uint32_t rows, uint32_t positions)
{
    slot_count = slots;
    pipeline_depth = depth;
    engine_rows = rows;
    draft_positions = positions;
}

void test_serving_set_device_block(uint32_t positions)
{
    device_block = positions;
}

void test_serving_set_device_draft(bool enabled, uint32_t rejected_position)
{
    device_draft = enabled;
    device_rejected_position = rejected_position;
    device_low_confidence = MODEL_DRAFT_BLOCK;
    device_wrong_round = false;
}

void test_serving_set_prefill(uint32_t prefill_first, uint32_t prefill_order, uint32_t prefill_order_wait)
{
    configured_prefill_first = prefill_first;
    configured_prefill_order = prefill_order;
    configured_prefill_order_wait = prefill_order_wait;
}

void test_serving_set_draft_policy(const struct draft_policy_record *policy,
				   uint32_t lookup_min)
{
    configured_policy = policy;
    configured_lookup_min = lookup_min;
}

void test_serving_set_ignore_eos(uint32_t ignore_eos)
{
	configured_ignore_eos = ignore_eos;
}

void test_serving_set_learned_acceptance(bool learned)
{
    configured_learned_acceptance = learned ? 1u : 0u;
}

void test_serving_set_session_acceptance(uint32_t level)
{
    configured_session_acceptance = level;
}

void test_serving_set_start_economic(bool economic)
{
    configured_start_economic = economic ? 1u : 0u;
}

void test_serving_set_device_confidence(uint32_t low_position)
{
    device_low_confidence = low_position;
}

void test_serving_set_device_round_fault(bool wrong_emitted)
{
    device_wrong_round = wrong_emitted;
}

uint64_t test_serving_device_rounds(void)
{
    return atomic_load(&device_rounds);
}

uint32_t test_serving_device_rejections(void)
{
    return atomic_load(&device_rejections);
}

void test_serving_set_function_model(bool is_function, uint32_t period)
{
    is_function_model = is_function;
    copy_period = period;
}

void test_serving_set_function_end(uint32_t position)
{
    function_end_position = position;
}

void test_serving_set_ring(uint32_t positions, uint32_t window)
{
    ring_positions = positions;
    model_window = window;
}

uint64_t test_serving_rows_seen(void)
{
    return atomic_load(&rows_seen);
}

uint64_t test_serving_front_rows_seen(void)
{
    return atomic_load(&front_rows_seen);
}

void test_serving_set_step_delay(uint64_t nanoseconds)
{
    atomic_store(&step_delay_nanoseconds, nanoseconds);
}

uint64_t test_serving_step_count(void)
{
    return atomic_load(&steps);
}

void *test_serving_tables(uint64_t *byte_count_out)
{
    static const char *const added[ADDED_TOKENS] = {REASONING_END, END_TEXT, REASONING_BEGIN};
    const uint32_t added_ids[ADDED_TOKENS] = {TEST_SERVING_REASONING_END_TOKEN, TEST_SERVING_END_TOKEN,
                                           TEST_SERVING_REASONING_BEGIN_TOKEN};
    const uint32_t added_bytes = (uint32_t)(strlen(REASONING_END) + strlen(END_TEXT) + strlen(REASONING_BEGIN));
    struct tokenizer_tables_header header = {
	    .format_version = TOKENIZER_TABLES_FORMAT_VERSION,
	    .vocabulary_size = TEST_SERVING_VOCABULARY,
	    .added_token_count = ADDED_TOKENS,
	    .class_block_count = 1,
	    .token_byte_count = TOKENIZER_BYTE_VALUES + added_bytes
    };
    memcpy(header.magic, TOKENIZER_TABLES_MAGIC, TOKENIZER_TABLES_MAGIC_BYTES);
    const size_t byte_count =
	    sizeof(header) +
	    sizeof(uint32_t) * (TEST_SERVING_VOCABULARY + ADDED_TOKENS +
				TOKENIZER_BYTE_VALUES) +
	    sizeof(uint16_t) * TOKENIZER_CLASS_BLOCK_COUNT +
	    TOKENIZER_CLASS_BLOCK_CODE_POINTS + header.token_byte_count;
    const size_t rounded = (byte_count / TOKENIZER_TABLES_ALIGNMENT + 1u) *
			   TOKENIZER_TABLES_ALIGNMENT;
    unsigned char *const tables =
	    aligned_alloc(TOKENIZER_TABLES_ALIGNMENT, rounded);
    if (tables == NULL) {
        return NULL;
    }
    memset(tables, 0, rounded);                         /* the class blocks: no class, block 0 for every point */
    unsigned char *cursor = tables;
    memcpy(cursor, &header, sizeof(header));
    cursor += sizeof(header);
    for (uint32_t token = 0, end = 0; token < TEST_SERVING_VOCABULARY; token++, cursor += sizeof(end)) {
	    end += token < TOKENIZER_BYTE_VALUES ?
			   1u :
			   (uint32_t)strlen(
				   added[token - TOKENIZER_BYTE_VALUES]);
	    memcpy(cursor, &end, sizeof(end));
    }
    memcpy(cursor, added_ids, sizeof(added_ids));
    cursor += sizeof(added_ids);
    for (uint32_t byte = 0; byte < TOKENIZER_BYTE_VALUES;
	 byte++, cursor += sizeof(byte)) {
	    memcpy(cursor, &byte, sizeof(byte));
    }
    cursor += sizeof(uint16_t) * TOKENIZER_CLASS_BLOCK_COUNT +
	      TOKENIZER_CLASS_BLOCK_CODE_POINTS;
    for (uint32_t byte = 0; byte < TOKENIZER_BYTE_VALUES; byte++) {
	    *cursor++ = (unsigned char)byte;
    }
    for (uint32_t index = 0; index < ADDED_TOKENS; index++) {
        memcpy(cursor, added[index], strlen(added[index]));
        cursor += strlen(added[index]);
    }
    *byte_count_out = byte_count;
    return tables;
}

const unsigned char *test_serving_state(void)
{
	const unsigned char *const arena =
		test_host_memory_backend_arena_of(cards[0]);
	return arena != NULL ? arena + STATE_OFFSET : NULL;
}
