/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * engine_test: source/engine.c against front_end_seam.h, over serving_fixture.h's model in host memory: the events of
 * a command to the end token, to its limit, to the context's end and with nothing to generate; the steps of a prompt
 * longer than a step's rows; the sampler's settings reaching the model; every refusal of create and submit; BUSY
 * while a slot's final event is not taken; a cancel at a step boundary and one before the first step; a token beyond
 * the vocabulary; the event queue full (the engine waits, no event is lost); the descriptor; destroy while a command
 * runs; the fault point engine_resources at each of its places; session commands; a session longer than the window
 * ring keeps behind, then a new one on its slot; and the cone (front steps). Speculation over the fixture's function
 * model: the lookup drafter alone; the events of speculative rounds against those of steps of one drawing row, greedy
 * and with three seeds, with proposals wrong at every index, the end token proposed and in another's place at every
 * index, the first round at the prompt's last row and a step saved for each proposal accepted, max_tokens and the
 * context's end inside a round (without a pipeline and in one of six steps), four sessions with and without a pipeline
 * (each drafting) and at a short context's end, front steps, a cancel between rounds and a command that keeps what it
 * left, a snapshot taken after rounds and continued, a prompt beside a session's rounds, the lookup drafter's depth.
 * The window ring of the model's size across commands: a prefix kept behind a shorter command and its snapshot, and
 * behind rejected rows. Run under the thread sanitizer with the others.
 * A card visited twice a step in a pipeline: prompt steps, cancellation and a fault.
 */
#include <math.h>
#include <poll.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "anemone/front_end_seam.h"
#include "anemone/hooks.h"
#include "anemone/model_layout.h"
#include "draft_lookup.h"
#include "engine_internal.h"                            /* the scripted drafter goes in the lookup drafter's place */
#include "host_memory_backend.h"
#include "serving_fixture.h"
#include "step_driver.h"

#define END TEST_SERVING_END_TOKEN
#define CONTEXT 1024u                                 /* room for every token of every case */
#define SHORT_CONTEXT 16u
#define PROMPT_TOKENS 5u
#define LONG_PROMPT 9u                                  /* steps of 4, 4 and 1 rows */
#define LONG_PROMPT_STEPS 3u
#define KEPT_TOKENS 2u                                  /* a whole pool entry of ratio 2 */
#define MOST_EVENTS 1024u
#define FIRST_TOKEN 10u
#define SECOND_TOKEN 11u
#define TWO 2u
#define SESSION_SHIFT_LEVEL 2u                          /* --draft_session_acceptance=2: the log-odds shift */
#define SESSION_PLATT_LEVEL 3u                          /* --draft_session_acceptance=3: scale and shift */
#define SESSION_HINDSIGHT_LEVEL 4u                      /* --draft_session_acceptance=4: 2 with hindsight checks */
#define SESSION_POSITION_LEVEL \
	6u /* conditional counts at each draft position */
#define SESSION_SCOUT_LEVEL 7u
#define SESSION_SECOND_LEVEL 8u
#define SESSION_NATURAL_LEVEL 9u
#define SESSION_RESIDUAL_LEVEL 10u
#define SESSION_NATURAL_EDGE_ROUNDS \
	6u /* four observations from natural depth, plus initial/final edges */
#define SECOND_DRAFT_POSITION 1u
#define SESSION_DEPTH_EDGE_ROUNDS \
	2u /* initial probe and a token-limited final block */
#define SESSION_COST_LEVEL 11u
#define SESSION_ACCEPTANCE_LEVELS SESSION_COST_LEVEL
#define POSITION_POLICY_PROBABILITY \
	200u /* a shared table that underestimates this fixture */
#define SCRIPT_ENDLESS 1u                               /* the seed of a script that never reaches the end token */
#define ENDLESS_TOKENS 600u
#define IGNORE_END_GENERATED_TOKENS 5u
#define INVALID_IGNORE_END_VALUE 2u
#define BEYOND_THE_QUEUE 300u                           /* more tokens than the engine's queue of events holds */
#define QUEUE_WAIT_NANOSECONDS 100000000ull             /* time for the engine to fill its queue and wait */
#define STEP_DELAY_NANOSECONDS 2000000ull
#define POLL_MILLISECONDS 5000
#define FAULT_PLACES 3u                                 /* engine_resources: the memory, the descriptor, the thread */
#define ENGINE_EVENT_CAPACITY 256u                      /* what the engine's queue holds (source/engine.c) */
#define PROMPT_STEPS 2u                                 /* of PROMPT_TOKENS: 4 rows and 1 */
#define PROMPT_VALUES 200u
#define MOST_LOOKS 5000u                                /* of a step's delay each: ten seconds */
#define SAVED_TOKENS 6u                                 /* of PROMPT_TOKENS and a token read: the kept granule */
#define SESSION_TAG 99u
#define CONE 2u                                         /* the fixture's cone */
#define KEPT_LAG 3968u                                  /* 4,096 ring slots less the 128-row window */
#define CONE_MARGIN (KEPT_LAG - 1u)                     /* serving_cursor.h's */
#define CONE_WHOLE_ROWS (CONE + CONE_MARGIN)            /* a prompt's last rows, whole */
#define CONE_FRONT_ROWS 103u                            /* the cone's prompt's front rows: the prompt even */
#define CONE_PROMPT (CONE_WHOLE_ROWS + CONE_FRONT_ROWS)
#define CONE_CONTEXT 8192u
#define CONE_SAVED (CONE_PROMPT - KEPT_TOKENS)          /* of the session the cone's prompt leaves: a granule's */
#define LAG_PROMPT (KEPT_LAG + KEPT_TOKENS)             /* a prefix of KEPT_TOKENS kept behind it lies at the lag */
#define DRAFT_POSITIONS 7u                              /* proposals a round: a thin step of 8 rows less its anchor */
#define ROUND_INDICES (DRAFT_POSITIONS + 1u)            /* where a round's tokens may end: at 0 to 7 */
#define PROPOSAL_INDICES 0x7Fu                          /* bits 0 to 6: where a proposal may stand */
#define SPECULATIVE_ROWS (DRAFT_POSITIONS + 1u)
#define SPECULATIVE_CONTEXT 1024u
#define SPECULATIVE_PROMPT 17u                          /* steps of 8 and 8 rows, then its last row and 7 proposals */
#define SPECULATIVE_LIMIT 300u
#define SEED_CASES 4u                                   /* greedy, then three seeds */
#define SEED_STRIDE 1000003ull
#define SAMPLED_TEMPERATURE 0.7f
#define BYTE_VALUES 256u
#define LIMIT_CASES (TWO * ROUND_INDICES)              /* max_tokens 1 to 16: the limit at every index of a round */
#define SHORT_SPECULATIVE_CONTEXT 64u
#define SHORT_SPECULATIVE_PROMPT 49u                    /* its last row in a step of 7 proposals, 15 tokens left */
#define SESSIONS TEST_SERVING_MOST_SLOTS
#define PIPELINE_DEPTH 6u
#define SESSION_LIMIT 120u
#define DEVICE_CONFIDENCE_POSITION 2u
#define DEVICE_REJECTION_MASK ((1u << MODEL_DRAFT_BLOCK) - 1u)
#define DEVICE_FULL_DEPTH_ANCHOR 30u             /* copy prompt20, k0/k1/k2/k4 rounds advance to anchor30 */
#define SLOT_PROMPT_STRIDE 7u                           /* where a slot's prompt begins in the prompt */
#define COPY_PERIOD 5u
#define COPY_PROMPT (COPY_PERIOD * 4u)
#define COPY_LIMIT 64u
#define LOOKUP_SPEEDUP 4u                               /* the least ratio of steps the lookup drafter saves there */
#define CANCEL_STEPS 4u                                 /* steps run before the cancel */
#define NEW_TOKENS 3u
#define NEW_TOKEN 7u                                    /* the new tokens' first value */
#define CONTINUED_LIMIT 40u
#define SNAPSHOT_LIMIT 60u
#define LOOKUP_CAPACITY 64u
#define NGRAM_MOST 4u
#define NGRAM_LEAST 2u
#define LOOKUP_KEPT 4u
#define MILLION_TOKENS 1048576u
#define CONE_REACH (CONE_PROMPT + 1u)                   /* what the cone's command ran: its prompt and a token drawn */
#define RING_CONTEXT 16384u                             /* the ring cases' commands */
/* The ring cases' prompt: bytes at a stride from a first one, shifted at each turn of the ring, so that a position and
 * the one a turn of the ring behind it hold different tokens. */
#define RING_PROMPT_STRIDE 37u
#define RING_PROMPT_TURN 101u
#define RING_PROMPT_FIRST 11u
#define LINEAGE_PROMPT 8000u                            /* a long command's prompt: what its slot ran, its reach */
#define LINEAGE_KEPT 4040u                              /* of it, a shorter command keeps this much (3,960 behind) */
#define LINEAGE_NEW 10u
#define LINEAGE_SAVED (LINEAGE_KEPT + LINEAGE_NEW)      /* what the shorter command ran, and its snapshot holds */
#define STALE_KEPT 200u                                 /* a prefix whose window the long command's rows overwrote */
#define RING_TOKEN 77u                                  /* the new token behind a kept prefix */
#define NEVER_DRAWN TEST_SERVING_REASONING_END_TOKEN    /* no draw of the function model: a proposal always wrong */
#define BOUNDARY_PROMPT 4400u
#define BOUNDARY_LIMIT 400u
#define PREFILL_PROMPT 700u
#define DECODER_LIMIT 600u
#define DECODER_STEPS 4u                                /* the decoding session's prompt read, its rounds begun */
#define PREFILL_SLACK_STEPS 8u                          /* when the long prompt comes, give or take */
#define ORDER_PROMPT 300u                               /* test_prefill_order's prompts: 38 steps of 8 rows */
#define ORDER_LIMIT 4u
#define ORDER_SLACK_STEPS 4u                            /* the first's turn, its first token's step */
#define ORDER_SHORT_PROMPT 40u                          /* test_prefill_order_shortest's short prompt: 5 steps */
#define ORDER_TAIL_PROMPT \
	3u /* two complete tails fit one eight-row thin step */
#define ORDER_WAIT 1u                                   /* and its prefill_order_wait: every other step the oldest's */
#define DECOY_PERIOD 10u                                /* decoy_block's */
#define DECOY_PROMPT (DECOY_PERIOD * 4u)
#define DEPTH_LIMIT 120u
#define INTERLEAVED_DEPTH 3u                            /* steps in flight of the interleaved programs */
#define INTERLEAVED_PROMPT 37u                          /* ten steps of rows: nine of 4, then one */
#define INTERLEAVED_PROMPT_STEPS 10u
#define FAULTING_POSITION 2u                            /* of a prompt: its first step faults */
#define LIVE_GENERATED_TOKENS 80u                       /* wraps the fixed 64-token ring */
#define LIVE_ARENA_BYTES 1234u
#define LIVE_WAIT_SAMPLE_NANOSECONDS 20000000ull
#define LIVE_WAIT_LIMIT_NANOSECONDS 10000000000ull

enum event_place { FIRST_EVENT, SECOND_EVENT, THIRD_EVENT, FOURTH_EVENT, EVENTS_OF_THE_SCRIPT };
enum wrong_command { WRONG_SLOT, WRONG_IDS, WRONG_COUNT, WRONG_RESERVED, WRONG_TEMPERATURE, WRONG_NOT_A_NUMBER,
                     WRONG_TOP_P, WRONG_CONTEXT, WRONG_COMMANDS };
enum wrong_session { WRONG_OPERATION, WRONG_DESCRIPTOR, WRONG_SESSION_RESERVED, WRONG_GRANULE, WRONG_SAVE_BEYOND,
                     WRONG_REACH_BELOW, WRONG_REACH_BEYOND, WRONG_SAVE_REACH, WRONG_SESSIONS };
/* The scripted drafter's proposals: the n-th round of a slot has proposal n mod ROUND_INDICES wrong (or none); all
 * right; all right but the one for the end token. */
enum draft_pattern { DRAFT_WRONG_CYCLE, DRAFT_RIGHT, DRAFT_END_WRONG, DRAFT_PATTERNS };
/* How a round ends where the engine is right: at its first wrong proposal; with all right (the bonus draw behind
 * them); at an end token proposed; at an end token where another was proposed; at a bonus draw that is one. */
enum round_end { ROUND_REJECTED, ROUND_ALL_RIGHT, ROUND_END_ACCEPTED, ROUND_END_CORRECTED, ROUND_END_BONUS,
                 ROUND_ENDS };

static int failures = 0;
static const char *context_text = "";
static uint32_t endless[ENDLESS_TOKENS + 1];
static const uint32_t short_script[] = {FIRST_TOKEN, SECOND_TOKEN, END};
static const uint32_t *const scripts[] = {short_script, endless};
static uint32_t prompt[CONTEXT + 1];
static uint32_t cone_prompt[CONE_PROMPT + 1];
static uint32_t copy_prompt[COPY_PROMPT];
static uint32_t ring_prompt[RING_CONTEXT];
static uint32_t ring_tokens[RING_CONTEXT + 1u];         /* a ring case's session, or a fresh run's tokens */
static uint32_t decoy_prompt[DECOY_PROMPT];
/* A period whose 4-gram 1 2 3 4 occurs twice with other tokens behind it: the lookup drafter's proposals behind it are
 * wrong, the others right, on the copy model. */
static const uint32_t decoy_block[DECOY_PERIOD] = {1, 2, 3, 4, 5, 1, 2, 3, 4, 6};
/* The scripted drafter's script, set by the test's thread while no engine is open: a reference run's tokens of each
 * slot, the rounds it drafted, its pattern and the proposals of a command's first round at most; then, a bit an
 * index, where its rounds ended, and how, the anchor of a first round and the proposals accepted; the proposals of
 * the last round of never_drawn_draft. */
static struct {
    uint32_t tokens[SESSIONS][SPECULATIVE_CONTEXT + 1u];
    uint32_t token_counts[SESSIONS];
    uint32_t rounds[SESSIONS];
    uint32_t pattern;
    uint32_t first_count;
    uint32_t ends[ROUND_ENDS];
    uint32_t first_anchor;
    uint32_t accepted;
    uint32_t last_count;
} drafted;
/* The lookup drafter watched by test_lookup_depth: the engine's own; what the cursor said at the last round (its depth
 * and statistics); the rounds asked, those whose depth broke the rule or that proposed beyond it, the deepest, and the
 * rounds whose depth fell. */
static struct {
    engine_draft_function lookup;
    uint32_t depth, rounds, proposals, accepted;
    uint32_t calls, wrong, deepest, falls;
} depth_watch;

static void expect(bool condition, const char *what)
{
    if (!condition) {
        (void)printf("FAILED: %s: %s\n", context_text, what);
        failures++;
    }
}

static void pause_for(uint64_t nanoseconds)
{
    const struct timespec pause = {0, (long)nanoseconds};
    expect(nanosleep(&pause, NULL) == 0, "the test sleeps");
}

static struct generation_command_record
command_of(uint32_t token_count, uint32_t limit, uint64_t seed)
{
	return (struct generation_command_record){
		.new_token_ids = prompt,
		.sampler_seed = seed,
		.new_token_count = token_count,
		.generated_token_limit = limit,
		.sampler_top_p = 1.0f,
		.request_tag = (uint32_t)seed + 1u
	};
}

/* Every event until the final one, waiting on the descriptor; the count taken. */
static uint32_t events_until_final(struct inference_engine *engine,
				   struct generation_event *events)
{
    uint32_t count = 0;
    int descriptor = -1;
    expect(engine_event_descriptor(engine, &descriptor) == STATUS_OK,
	   "the descriptor");
    while (count < MOST_EVENTS) {
        struct pollfd poll_entry = {descriptor, POLLIN, 0};
        bool is_available = false;
        if (poll(&poll_entry, 1, POLL_MILLISECONDS) != 1) {
            expect(false, "an event comes within the poll's time");
            return count;
        }
	expect(engine_next_event(engine, &events[count], &is_available) ==
		       STATUS_OK,
	       "next_event");
	if (is_available &&
	    events[count++].event_kind != GENERATION_EVENT_TOKEN) {
		return count;
	}
    }
    return count;
}

static void test_ignore_end_token(void)
{
	context_text = "explicit continuation after end-token draws";
	test_serving_set_engine(1, 1, TEST_SERVING_ROWS, 0);
	for (uint32_t ignored = 0; ignored <= 1; ignored++) {
		struct inference_engine *engine = NULL;
		static struct generation_event events[MOST_EVENTS];
		test_serving_set_ignore_eos(ignored);
		expect(test_serving_open(CONTEXT, scripts,
					 sizeof(scripts) / sizeof(scripts[0]),
					 &engine) == STATUS_OK,
		       "the immutable end-token policy opens");
		const struct generation_command_record command = command_of(
			PROMPT_TOKENS, IGNORE_END_GENERATED_TOKENS, 0);
		expect(engine != NULL &&
			       engine_submit(engine, &command) == STATUS_OK,
		       "the command is submitted");
		const uint32_t count =
			engine != NULL ? events_until_final(engine, events) : 0;
		if (ignored == 0) {
			expect(count == EVENTS_OF_THE_SCRIPT &&
				       events[count - 1u].event_kind ==
					       GENERATION_EVENT_FINISHED_AT_END_TOKEN,
			       "default policy still ends at EOS");
		} else {
			expect(count == IGNORE_END_GENERATED_TOKENS + 1u &&
				       events[count - 1u].event_kind ==
					       GENERATION_EVENT_FINISHED_AT_LIMIT,
			       "the explicit policy reaches the token limit");
			for (uint32_t index = THIRD_EVENT;
			     index < IGNORE_END_GENERATED_TOKENS &&
			     index < count;
			     index++)
				expect(events[index].event_kind ==
						       GENERATION_EVENT_TOKEN &&
					       events[index].token_id == END,
				       "end-token draws remain ordinary token events");
		}
		test_serving_close();
	}
	struct inference_engine *engine = NULL;
	test_serving_set_ignore_eos(INVALID_IGNORE_END_VALUE);
	expect(test_serving_open(CONTEXT, scripts,
				 sizeof(scripts) / sizeof(scripts[0]),
				 &engine) == STATUS_INVALID_ARGUMENT,
	       "invalid end-token policy refuses");
	test_serving_close();
	test_serving_set_ignore_eos(1);
	test_serving_set_engine(1, 1, TEST_SERVING_ROWS, 1);
	expect(test_serving_open(CONTEXT, scripts,
				 sizeof(scripts) / sizeof(scripts[0]),
				 &engine) == STATUS_INVALID_ARGUMENT,
	       "speculative continuation is not admitted");
	test_serving_close();
	test_serving_set_ignore_eos(0);
	test_serving_set_engine(1, 1, TEST_SERVING_ROWS, 0);
}

static void test_commands(struct inference_engine *engine)
{
	static struct generation_event events[MOST_EVENTS];
	struct generation_command_record command =
		command_of(PROMPT_TOKENS, ENDLESS_TOKENS, 0);
	context_text = "a command to the end token";
	expect(engine_submit(engine, &command) == STATUS_OK, "submitted");
	uint32_t count = events_until_final(engine, events);
	expect(count == EVENTS_OF_THE_SCRIPT &&
		       events[FIRST_EVENT].token_id == FIRST_TOKEN &&
		       events[SECOND_EVENT].token_id == SECOND_TOKEN &&
		       events[THIRD_EVENT].token_id == END &&
		       events[FOURTH_EVENT].event_kind ==
			       GENERATION_EVENT_FINISHED_AT_END_TOKEN,
	       "the script's two tokens, the end token, then the final event");
	expect(events[FIRST_EVENT].session_token_count == PROMPT_TOKENS + 1u &&
		       events[FOURTH_EVENT].session_token_count ==
			       PROMPT_TOKENS + THIRD_EVENT + 1u &&
		       events[FIRST_EVENT].logit ==
			       (float)(PROMPT_TOKENS - 1u) &&
		       events[SECOND_EVENT].logit == (float)PROMPT_TOKENS &&
		       events[FOURTH_EVENT].request_tag == 1u &&
		       events[FOURTH_EVENT].status == STATUS_OK &&
		       events[FOURTH_EVENT].token_id == 0,
	       "the counts the session holds, the logits of each position, the tag");

	context_text = "a command to its limit";
	command = command_of(PROMPT_TOKENS, TWO, 0);
	expect(engine_submit(engine, &command) == STATUS_OK, "submitted");
	count = events_until_final(engine, events);
	expect(count == THIRD_EVENT + 1u &&
		       events[THIRD_EVENT].event_kind ==
			       GENERATION_EVENT_FINISHED_AT_LIMIT &&
		       events[THIRD_EVENT].session_token_count ==
			       PROMPT_TOKENS + TWO,
	       "two tokens, then the limit");

	context_text = "a command that generates nothing";
	command = command_of(PROMPT_TOKENS, 0, 0);
	expect(engine_submit(engine, &command) == STATUS_OK, "submitted");
	count = events_until_final(engine, events);
	expect(count == 1u &&
		       events[0].event_kind ==
			       GENERATION_EVENT_FINISHED_AT_LIMIT &&
		       events[0].session_token_count == PROMPT_TOKENS,
	       "only the final event, the prompt held");

	context_text = "a prompt longer than a step";
	const uint64_t steps_before = test_serving_step_count();
	command = command_of(LONG_PROMPT, 1, 0);
	expect(engine_submit(engine, &command) == STATUS_OK, "submitted");
	count = events_until_final(engine, events);
	expect(count == TWO && events[0].token_id == FIRST_TOKEN &&
		       events[0].logit == (float)(LONG_PROMPT - 1u) &&
		       test_serving_step_count() - steps_before ==
			       LONG_PROMPT_STEPS,
	       "steps of 4, 4 and 1 rows, one token");

	context_text = "kept tokens";
	command = command_of(PROMPT_TOKENS, 1, 0);
	command.kept_token_count = 1;
	expect(engine_submit(engine, &command) == STATUS_INVALID_ARGUMENT,
	       "part of a pool entry kept");
	command.kept_token_count =
		LONG_PROMPT + TWO; /* it holds the prompt and one token */
	expect(engine_submit(engine, &command) == STATUS_INVALID_ARGUMENT,
	       "more than the session holds");
	command.kept_token_count = KEPT_TOKENS;
	expect(engine_submit(engine, &command) == STATUS_OK,
	       "a kept prefix is taken");
	count = events_until_final(engine, events);
	expect(count == TWO && events[1].session_token_count ==
				       KEPT_TOKENS + PROMPT_TOKENS + 1u,
	       "one token behind the kept and the new ones");
}

static void test_context_end(void)
{
	static struct generation_event events[MOST_EVENTS];
	struct inference_engine *engine = NULL;
	context_text = "the context's end";
	expect(test_serving_open(SHORT_CONTEXT, scripts,
				 sizeof(scripts) / sizeof(scripts[0]),
				 &engine) == STATUS_OK,
	       "an engine of a short context");
	struct generation_command_record command =
		command_of(SHORT_CONTEXT - TWO, ENDLESS_TOKENS, SCRIPT_ENDLESS);
	expect(engine != NULL && engine_submit(engine, &command) == STATUS_OK,
	       "submitted");
	const uint32_t count =
		engine != NULL ? events_until_final(engine, events) : 0;
	expect(count == EVENTS_OF_THE_SCRIPT &&
		       events[FOURTH_EVENT].event_kind ==
			       GENERATION_EVENT_FINISHED_AT_LIMIT &&
		       events[THIRD_EVENT].logit ==
			       (float)(SHORT_CONTEXT - 1u) &&
		       events[FOURTH_EVENT].session_token_count ==
			       SHORT_CONTEXT + 1u,
	       "a token from each of the last three positions, then the limit: no position at the context is read");
	command = command_of(SHORT_CONTEXT, 0, 0);
	expect(engine != NULL && engine_submit(engine, &command) == STATUS_OK,
	       "a prompt of the context");
	command = command_of(SHORT_CONTEXT, 0, 0);
	command.new_token_count = SHORT_CONTEXT + 1u;
	(void)events_until_final(engine, events);
	expect(engine != NULL && engine_submit(engine, &command) ==
					 STATUS_INVALID_ARGUMENT,
	       "a prompt beyond the context");
	test_serving_close();
}

/* A session longer than the window ring keeps behind (serving_cursor.h): a prefix 3,961 tokens behind what the slot ran
 * is kept, one 3,962 behind refused; a command that keeps nothing starts anew on its slot (it was refused until 25
 * October 2026: every new conversation on a slot of a long one failed), and so does the slot's reach: a prefix of the
 * new session is kept behind it. */
static void test_long_session(void)
{
	static struct generation_event events[MOST_EVENTS];
	struct inference_engine *engine = NULL;
	struct generation_command_record command = { 0 };
	context_text = "a session longer than the ring keeps behind";
	expect(test_serving_open(CONE_CONTEXT, scripts,
				 sizeof(scripts) / sizeof(scripts[0]),
				 &engine) == STATUS_OK,
	       "an engine of a long context");
	for (uint32_t beyond = 0; beyond <= 1 && engine != NULL; beyond++) {
		command = command_of(LAG_PROMPT + beyond, 1, 0);
		command.new_token_ids = cone_prompt;
		expect(engine_submit(engine, &command) == STATUS_OK,
		       "the long prompt");
		(void)events_until_final(engine, events);
		command = command_of(PROMPT_TOKENS, 1, 0);
		command.kept_token_count = KEPT_TOKENS;
		expect(engine_submit(engine, &command) ==
			       (beyond == 0 ? STATUS_OK :
					      STATUS_INVALID_ARGUMENT),
		       beyond == 0 ?
			       "a prefix 3,961 behind is kept" :
			       "a prefix 3,962 behind, which the ring may no longer "
			       "hold, is refused");
		if (beyond == 0) {
			expect(events_until_final(engine, events) == TWO &&
				       events[1].session_token_count ==
					       KEPT_TOKENS + PROMPT_TOKENS + 1u,
			       "one token behind it");
		}
    }
    command.kept_token_count = 0;
    expect(engine != NULL && engine_submit(engine, &command) == STATUS_OK,
	   "a command that keeps nothing starts anew");
    uint32_t count = engine != NULL ? events_until_final(engine, events) : 0;
    expect(count == TWO && events[1].session_token_count == PROMPT_TOKENS + 1u, "the new prompt alone, one token");
    command.kept_token_count = KEPT_TOKENS;
    expect(engine != NULL && engine_submit(engine, &command) == STATUS_OK,
	   "the slot's reach starts anew with it: a prefix of the new session is kept");
    count = engine != NULL ? events_until_final(engine, events) : 0;
    expect(count == TWO && events[1].session_token_count == KEPT_TOKENS + PROMPT_TOKENS + 1u, "one token behind it");
    test_serving_close();
}

static void test_refusals(struct inference_engine *engine)
{
    static const uint32_t beyond[] = {TEST_SERVING_VOCABULARY};
    static struct generation_event events[MOST_EVENTS];
    const struct engine_description_record description = {
	    .session_slot_count = 1,
	    .context_token_limit = CONTEXT,
	    .vocabulary_size = TEST_SERVING_VOCABULARY,
	    .end_token_id = END
    };
    struct inference_engine *created = NULL;
    struct engine_description_record described = { 0 };
    context_text = "refusals";
    expect(engine_create(NULL, &description, 1, &created) ==
		   STATUS_INVALID_ARGUMENT,
	   "no driver");
    expect(engine_describe(engine, &described) == STATUS_OK &&
		   strcmp(described.model_name, TEST_SERVING_MODEL) == 0 &&
		   described.context_token_limit == CONTEXT,
	   "the description given at create");
    expect(engine_describe(NULL, &described) == STATUS_INVALID_ARGUMENT,
	   "describe without engine");
    const struct generation_command_record sound =
	    command_of(PROMPT_TOKENS, 1, 0);
    struct generation_command_record wrong[WRONG_COMMANDS];
    for (uint32_t index = 0; index < WRONG_COMMANDS; index++) {
        wrong[index] = sound;
    }
    wrong[WRONG_SLOT].session_slot = 1;
    wrong[WRONG_IDS].new_token_ids = NULL;
    wrong[WRONG_COUNT].new_token_count = 0;
    wrong[WRONG_RESERVED].reserved_must_be_zero = 1;
    wrong[WRONG_TEMPERATURE].sampler_temperature = -1.0f;
    wrong[WRONG_NOT_A_NUMBER].sampler_temperature = NAN;
    wrong[WRONG_TOP_P].sampler_top_p = 0.0f;
    wrong[WRONG_CONTEXT].new_token_count = CONTEXT + 1u;
    for (uint32_t index = 0; index < WRONG_COMMANDS; index++) {
	    expect(engine_submit(engine, &wrong[index]) ==
			   STATUS_INVALID_ARGUMENT,
		   "a wrong command");
    }
    expect(engine_submit(NULL, &sound) == STATUS_INVALID_ARGUMENT, "no engine");
    expect(engine_cancel(engine, 1) == STATUS_INVALID_ARGUMENT,
	   "no such slot to cancel");
    expect(engine_cancel(engine, 0) == STATUS_OK,
	   "a slot with no command is left alone");
    struct generation_command_record beyond_command = sound;
    beyond_command.new_token_ids = beyond;
    beyond_command.new_token_count = 1;
    expect(engine_submit(engine, &beyond_command) == STATUS_OK,
	   "a token beyond is found when read");
    const uint32_t count = events_until_final(engine, events);
    expect(count == 1u && events[0].event_kind == GENERATION_EVENT_FAILED &&
		   events[0].status == STATUS_INVALID_ARGUMENT,
	   "it fails the command");
    struct engine_slot_stats rejected = { 0 };
    expect(engine_stats_slot(engine, 0, &rejected) == STATUS_OK &&
		   rejected.prompt_processed == 0 &&
		   rejected.generated_tokens == 0,
	   "a failed attempted row cannot claim successful prompt progress");
}

static void test_busy_and_cancel(struct inference_engine *engine)
{
	static struct generation_event events[MOST_EVENTS];
	struct generation_command_record command =
		command_of(PROMPT_TOKENS, ENDLESS_TOKENS, SCRIPT_ENDLESS);
	bool is_available = false;
	const uint64_t steps_before = test_serving_step_count();
	context_text = "busy, and a cancel";
	test_serving_set_step_delay(STEP_DELAY_NANOSECONDS);
	expect(engine_submit(engine, &command) == STATUS_OK, "submitted");
	expect(engine_submit(engine, &command) == STATUS_BUSY_VALUE,
	       "the slot is busy");
	for (uint32_t look = 0;
	     look < MOST_LOOKS &&
	     test_serving_step_count() < steps_before + PROMPT_STEPS + 1u;
	     look++) {
		pause_for(
			STEP_DELAY_NANOSECONDS); /* until the prompt is read and a token drawn */
	}
	expect(engine_cancel(engine, 0) == STATUS_OK,
	       "cancelled once a token came");
	const uint32_t count = events_until_final(engine, events);
	const struct generation_event final =
		events[count != 0 ? count - 1u : 0];
	expect(count >= 1u && count < ENDLESS_TOKENS &&
		       final.event_kind == GENERATION_EVENT_CANCELLED &&
		       final.session_token_count == PROMPT_TOKENS + count - 1u,
	       "it ends at a step boundary, its tokens held");
	expect(engine_next_event(engine, &events[0], &is_available) ==
			       STATUS_OK &&
		       !is_available,
	       "nothing follows the final event");
	test_serving_set_step_delay(0);

	context_text = "the queue of events full";
	command = command_of(PROMPT_TOKENS, BEYOND_THE_QUEUE, SCRIPT_ENDLESS);
	expect(engine_submit(engine, &command) == STATUS_OK, "submitted");
	const uint64_t steps_queued = test_serving_step_count();
	pause_for(QUEUE_WAIT_NANOSECONDS);
	const uint64_t steps_waiting = test_serving_step_count() - steps_queued;
	expect(steps_waiting <= ENGINE_EVENT_CAPACITY + PROMPT_STEPS + 1u,
	       "the engine waits for room");
	struct engine_stats snapshot = { 0 };
	for (uint32_t look = 0; look < MOST_LOOKS; look++) {
		expect(engine_stats_snapshot(engine, &snapshot) == STATUS_OK,
		       "event pressure snapshot succeeds");
		if (snapshot.event_queue_depth == ENGINE_EVENT_CAPACITY &&
		    snapshot.backpressure_stalls != 0) {
			break;
		}
		pause_for(STEP_DELAY_NANOSECONDS);
	}
	expect(engine_stats_snapshot(engine, &snapshot) == STATUS_OK &&
		       snapshot.event_queue_depth == ENGINE_EVENT_CAPACITY &&
		       snapshot.backpressure_stalls != 0,
	       "release live counters observe event backpressure without consuming an event");
	const uint32_t taken = events_until_final(engine, events);
	uint32_t in_order = 0;
	for (uint32_t index = 0; index + 1u < taken; index++) {
		in_order += events[index].event_kind ==
						    GENERATION_EVENT_TOKEN &&
					    events[index].session_token_count ==
						    PROMPT_TOKENS + index + 1u ?
				    1u :
				    0u;
	}
    expect(taken == BEYOND_THE_QUEUE + 1u && in_order == BEYOND_THE_QUEUE, "every event comes, in order");
}

static void test_destroy_and_faults(void)
{
	struct inference_engine *engine = NULL;
	context_text = "destroy while a command runs";
	expect(test_serving_open(CONTEXT, scripts,
				 sizeof(scripts) / sizeof(scripts[0]),
				 &engine) == STATUS_OK,
	       "an engine");
	test_serving_set_step_delay(STEP_DELAY_NANOSECONDS);
	struct generation_command_record command =
		command_of(PROMPT_TOKENS, ENDLESS_TOKENS, SCRIPT_ENDLESS);
	expect(engine != NULL && engine_submit(engine, &command) == STATUS_OK,
	       "submitted");
	pause_for(STEP_DELAY_NANOSECONDS * LONG_PROMPT);
	test_serving_close(); /* joins the thread: it ends at the next step boundary */
#if CHECKING_HOOKS_ENABLED
	context_text = "the fault point engine_resources";
	for (uint64_t place = 1; place <= FAULT_PLACES; place++) {
		expect(fault_arm("engine_resources", place), "armed");
		engine = NULL;
		expect(test_serving_open(CONTEXT, scripts,
					 sizeof(scripts) / sizeof(scripts[0]),
					 &engine) == STATUS_OUT_OF_MEMORY &&
			       engine == NULL,
		       "create refuses, nothing left");
		expect(fault_disarm() == place, "the place was reached");
	}
#endif
}

/* A session command and its final event; the final, or a FAILED event of the status if it was refused. */
static struct generation_event
session_done(struct inference_engine *engine,
	     const struct session_command *command)
{
	static struct generation_event events[MOST_EVENTS];
	const result_status status = engine_session(engine, command);
	if (status != STATUS_OK) {
		return (struct generation_event){
			.event_kind = GENERATION_EVENT_FAILED,
			.status = (uint32_t)status
		};
	}
    const uint32_t count = events_until_final(engine, events);
    return count != 0 ? events[count - 1u] : (struct generation_event){ 0 };
}

/* Session commands: a session saved after its command; the engine of a restart, its state empty, loads it, holds its
 * tokens and its state, and goes on behind them; the refusals; a file cut short fails the load and empties the
 * session. */
static void test_sessions(void)
{
	static struct generation_event events[MOST_EVENTS];
	struct inference_engine *engine = NULL;
	char path[] = "/tmp/anemone_engine_test_XXXXXX";
	const int descriptor = mkstemp(path);
	uint32_t tokens[SAVED_TOKENS];
	unsigned char state[SAVED_TOKENS];
	context_text = "session commands";
	expect(descriptor >= 0 && unlink(path) == 0, "a file for the snapshot");
	if (descriptor < 0) {
		return;
	}
	expect(test_serving_open(CONTEXT, scripts,
				 sizeof(scripts) / sizeof(scripts[0]),
				 &engine) == STATUS_OK,
	       "the engine opens");
	const struct generation_command_record first =
		command_of(PROMPT_TOKENS, TWO, 0);
	expect(engine_submit(engine, &first) == STATUS_OK &&
		       events_until_final(engine, events) != 0,
	       "a command runs: its prompt and a token drawn and read, the last drawn not");
	memcpy(tokens, prompt, PROMPT_TOKENS * sizeof(uint32_t));
	tokens[PROMPT_TOKENS] = FIRST_TOKEN;
	memcpy(state, test_serving_state(), sizeof(state));
	const struct session_command save = {
		NULL, descriptor, 0, SAVED_TOKENS, SESSION_SAVE, SESSION_TAG,
		0,    0,	  0
	}; /* a save names no reach */
	struct generation_event final = session_done(engine, &save);
	expect(final.event_kind == GENERATION_EVENT_FINISHED_AT_LIMIT &&
		       final.session_token_count == PROMPT_TOKENS + TWO &&
		       lseek(descriptor, 0, SEEK_CUR) == SAVED_TOKENS,
	       "the save writes a byte a token of the state and keeps the session");
	struct engine_stats snapshot = { 0 };
	expect(engine_stats_snapshot(engine, &snapshot) == STATUS_OK &&
		       snapshot.snapshots_saved == 1u &&
		       snapshot.snapshot_save_bytes == SAVED_TOKENS &&
		       snapshot.snapshot_save_ns != 0 &&
		       snapshot.snapshots_loaded == 0,
	       "a successful snapshot save counts actual bytes and host elapsed time");
	expect(snapshot.sessions[0].snapshots_saved == 1 &&
		       snapshot.sessions[0].snapshot_save_bytes ==
			       SAVED_TOKENS &&
		       snapshot.sessions[0].snapshot_save_ns ==
			       snapshot.snapshot_save_ns &&
		       !snapshot.sessions[0].active &&
		       snapshot.sessions[0].idle_since_ns != 0 &&
		       snapshot.sessions[0].operation == 0,
	       "slot save counters and idle state survive the completed snapshot command");
	test_serving_close();
	expect(test_serving_open(CONTEXT, scripts,
				 sizeof(scripts) / sizeof(scripts[0]),
				 &engine) == STATUS_OK &&
		       memcmp(test_serving_state(), state, sizeof(state)) != 0,
	       "a restart: its state is not the session's");
	const struct session_command load = { tokens,
					      descriptor,
					      0,
					      SAVED_TOKENS,
					      SESSION_LOAD,
					      SESSION_TAG,
					      SAVED_TOKENS,
					      0,
					      0 }; /* the saved slot's reach */
	expect(lseek(descriptor, 0, SEEK_SET) == 0,
	       "the snapshot read from its beginning");
	final = session_done(engine, &load);
	expect(final.event_kind == GENERATION_EVENT_FINISHED_AT_LIMIT &&
		       final.session_token_count == SAVED_TOKENS &&
		       memcmp(test_serving_state(), state, sizeof(state)) == 0,
	       "the load gives the session its tokens and its state");
	expect(engine_stats_snapshot(engine, &snapshot) == STATUS_OK &&
		       snapshot.snapshots_loaded == 1u &&
		       snapshot.snapshot_load_bytes == SAVED_TOKENS &&
		       snapshot.snapshot_load_ns != 0 &&
		       snapshot.snapshots_saved == 0,
	       "a successful snapshot load counts its bytes and starts new process totals");
	expect(snapshot.sessions[0].snapshots_loaded == 1 &&
		       snapshot.sessions[0].snapshot_load_bytes ==
			       SAVED_TOKENS &&
		       snapshot.sessions[0].snapshot_load_ns ==
			       snapshot.snapshot_load_ns &&
		       snapshot.sessions[0].position == SAVED_TOKENS &&
		       snapshot.sessions[0].reach == SAVED_TOKENS &&
		       snapshot.sessions[0].context_capacity == CONTEXT &&
		       !snapshot.sessions[0].active,
	       "idle session reports held context and its own load counters");
	struct generation_command_record next = command_of(1, 1, 0);
	next.new_token_ids = &tokens[1];
	next.kept_token_count = SAVED_TOKENS;
	expect(engine_submit(engine, &next) == STATUS_OK,
	       "a command keeps the loaded tokens");
	expect(engine_stats_counters(engine, &snapshot) == STATUS_OK &&
		       snapshot.sessions[0].reach >= SAVED_TOKENS &&
		       snapshot.sessions[0].position >= SAVED_TOKENS,
	       "a newly admitted reuse keeps held reach before its first worker publication");
	expect(events_until_final(engine, events) != 0,
	       "the reuse command completes");
	expect(engine_stats_counters(engine, &snapshot) == STATUS_OK &&
		       snapshot.sessions[0].snapshots_loaded == 1 &&
		       snapshot.sessions[0].snapshot_load_bytes == SAVED_TOKENS,
	       "generation does not reset persistent per-slot transfer counters");
	struct session_command wrong[WRONG_SESSIONS];
	for (uint32_t index = 0; index < WRONG_SESSIONS; index++) {
		wrong[index] = load;
	}
    wrong[WRONG_OPERATION].operation = 0;
    wrong[WRONG_DESCRIPTOR].descriptor = -1;
    wrong[WRONG_SESSION_RESERVED].reserved_must_be_zero = 1;
    wrong[WRONG_GRANULE].token_count = SAVED_TOKENS - 1u;
    wrong[WRONG_SAVE_BEYOND] = save;
    wrong[WRONG_SAVE_BEYOND].token_count = CONTEXT;
    wrong[WRONG_REACH_BELOW].reach_count = SAVED_TOKENS - 1u;
    wrong[WRONG_REACH_BEYOND].reach_count = CONTEXT + 1u;
    wrong[WRONG_SAVE_REACH] = save;
    wrong[WRONG_SAVE_REACH].reach_count = SAVED_TOKENS;
    for (uint32_t index = 0; index < WRONG_SESSIONS; index++) {
	    expect(engine_session(engine, &wrong[index]) ==
			   STATUS_INVALID_ARGUMENT,
		   "a session command refused");
    }
    expect(ftruncate(descriptor, 1) == 0 && lseek(descriptor, 0, SEEK_SET) == 0, "the snapshot cut short");
    final = session_done(engine, &load);
    expect(final.event_kind == GENERATION_EVENT_FAILED &&
		   final.status == STATUS_INPUT_OUTPUT_ERROR &&
		   final.session_token_count == 0,
	   "a load of a file cut short fails and leaves the session empty");
    expect(engine_submit(engine, &next) == STATUS_INVALID_ARGUMENT,
	   "nothing is kept after a failed load");
    expect(engine_stats_snapshot(engine, &snapshot) == STATUS_OK &&
		   snapshot.snapshots_loaded == 1u &&
		   snapshot.snapshot_load_bytes == SAVED_TOKENS + 1u,
	   "a failed load counts transferred bytes but cannot claim another successful load");
    expect(snapshot.sessions[0].snapshots_loaded == 1 &&
           snapshot.sessions[0].snapshot_load_bytes == SAVED_TOKENS + 1u && snapshot.sessions[0].position == 0 &&
           snapshot.sessions[0].reach == 0 && !snapshot.sessions[0].active,
           "failed load is attributed to its slot and leaves an empty idle context");
    (void)close(descriptor);
    test_serving_close();
}

/* The cone (model_seam.h's front steps), the fixture's of CONE rows: a prompt's rows before its last CONE and the
 * margin run the front alone (the model's launch reads none of them, its state there untouched) and the tokens drawn
 * are the script's; a snapshot preserves its actual decoder boundary, so a continuation that keeps more than that
 * boundary reads its new token alone after loading and again after the next command. */
static void test_cone(void)
{
	static struct generation_event events[MOST_EVENTS];
	struct inference_engine *engine = NULL;
	struct session_state saved = { 0 };
	char path[] = "/tmp/anemone_engine_cone_XXXXXX";
	const int descriptor = mkstemp(path);
	context_text = "the cone";
	expect(descriptor >= 0 && unlink(path) == 0, "a file for the snapshot");
	test_serving_set_cone(CONE);
	expect(descriptor >= 0 &&
		       test_serving_open(CONE_CONTEXT, scripts,
					 sizeof(scripts) / sizeof(scripts[0]),
					 &engine) == STATUS_OK,
	       "the engine opens with a cone");
	if (engine == NULL) {
		test_serving_set_cone(0);
		return;
	}
	struct generation_command_record command =
		command_of(CONE_PROMPT, TWO, 0);
	command.new_token_ids = cone_prompt;
	expect(engine_submit(engine, &command) == STATUS_OK &&
		       events_until_final(engine, events) ==
			       EVENTS_OF_THE_SCRIPT - 1u &&
		       events[FIRST_EVENT].token_id == FIRST_TOKEN &&
		       events[SECOND_EVENT].token_id == SECOND_TOKEN,
	       "a long prompt draws the script's tokens");
	const unsigned char *const state = test_serving_state();
	bool is_front = true, is_whole = true;
	for (uint32_t position = 0; position < CONE_PROMPT; position++) {
		is_front = is_front && (position >= CONE_FRONT_ROWS ||
					state[position] == 0);
		is_whole = is_whole &&
			   (position < CONE_FRONT_ROWS ||
			    state[position] ==
				    (unsigned char)cone_prompt[position]);
	}
    expect(is_front && is_whole && test_serving_rows_seen() == CONE_WHOLE_ROWS + 1u,
           "the rows before the last CONE and the margin run the front alone");
    const struct session_command save = {
	    NULL, descriptor, 0, CONE_SAVED, SESSION_SAVE, SESSION_TAG, 0, 0, 0
    }; /* a save names no reach */
    expect(engine_session_state(engine, 0, &saved) == STATUS_OK &&
		   saved.decoder_from == CONE_FRONT_ROWS + CONE - 1u &&
		   saved.reach_count == CONE_REACH,
	   "the real decoder boundary and reach before saving");
    expect(session_done(engine, &save).event_kind ==
		   GENERATION_EVENT_FINISHED_AT_LIMIT,
	   "a save");
    test_serving_close();
    expect(test_serving_open(CONE_CONTEXT, scripts,
			     sizeof(scripts) / sizeof(scripts[0]),
			     &engine) == STATUS_OK &&
		   lseek(descriptor, 0, SEEK_SET) == 0,
	   "a restart");
    const struct session_command load = {
	    cone_prompt, descriptor,	    0, CONE_SAVED,	  SESSION_LOAD,
	    SESSION_TAG, saved.reach_count, 0, saved.decoder_from
    };
    expect(session_done(engine, &load).session_token_count == CONE_SAVED, "the load");
    command = command_of(1, 1, 0);
    command.new_token_ids = &cone_prompt[CONE_SAVED - KEPT_TOKENS];
    command.kept_token_count = CONE_SAVED - KEPT_TOKENS;
    uint64_t rows_before = test_serving_rows_seen();
    expect(engine_submit(engine, &command) == STATUS_OK &&
		   events_until_final(engine, events) != 0 &&
		   events[FIRST_EVENT].token_id == FIRST_TOKEN &&
		   test_serving_rows_seen() - rows_before == 1u,
	   "the actual saved decoder boundary permits the continuation without replay");
    rows_before = test_serving_rows_seen();
    expect(engine_submit(engine, &command) == STATUS_OK &&
		   events_until_final(engine, events) != 0 &&
		   test_serving_rows_seen() - rows_before == 1u,
	   "behind it, the same command reads its new token alone");
    (void)close(descriptor);
    test_serving_close();
    test_serving_set_cone(0);
}

/* ---- speculation (front_end_seam.h) ---------------------------------------------------------------------------- */

/* The lookup drafter (draft_lookup.h): its buckets; what followed the latest earlier occurrence of the longest suffix,
 * as many as asked; none where no suffix occurred before; a kept prefix forgets the rest, whose buckets are not taken,
 * and proposes from the tokens that come behind it, running on past them with their period. */
static void test_lookup_drafter(void)
{
    static const uint32_t sequence[] = {1, 2, 3, 4, 9, 8, 3, 4, 6, 1, 2, 3, 4};
    static const uint32_t longest[] = {9, 8, 3, 4, 6, 1, 2};
    static const uint32_t behind_kept[] = {5, 6, 5, 6, 5};
    static const uint32_t after_kept[] = {6, 5, 6, 5, 6, 5, 6};
    uint32_t tokens[LOOKUP_CAPACITY], proposals[DRAFT_POSITIONS];
    struct draft_lookup lookup;
    context_text = "the lookup drafter";
    const uint32_t bits = draft_lookup_bucket_bits(LOOKUP_CAPACITY, NGRAM_LEAST, NGRAM_MOST);
    expect((1ull << bits) >= (uint64_t)LOOKUP_CAPACITY * (NGRAM_MOST - NGRAM_LEAST + 1u) &&
           (1ull << draft_lookup_bucket_bits(MILLION_TOKENS, NGRAM_LEAST, NGRAM_MOST)) >=
           (uint64_t)MILLION_TOKENS * (NGRAM_MOST - NGRAM_LEAST + 1u), "a bucket at least for each n-gram filed");
    uint32_t *const buckets = calloc((size_t)1u << bits, sizeof(uint32_t));
    if (buckets == NULL) {
        expect(false, "memory for the buckets");
        return;
    }
    draft_lookup_begin(&lookup, tokens, LOOKUP_CAPACITY, buckets, bits, NGRAM_LEAST, NGRAM_MOST);
    expect(draft_lookup_propose(&lookup, DRAFT_POSITIONS, proposals) == 0, "nothing to propose from no token");
    for (uint32_t index = 0; index < sizeof(sequence) / sizeof(sequence[0]); index++) {
        draft_lookup_append(&lookup, sequence[index]);
    }
    expect(draft_lookup_propose(&lookup, DRAFT_POSITIONS, proposals) == DRAFT_POSITIONS &&
           memcmp(proposals, longest, sizeof(longest)) == 0,
           "what followed the latest earlier occurrence of the longest suffix (4 tokens), running into the suffix");
    expect(draft_lookup_propose(&lookup, TWO, proposals) == TWO &&
           memcmp(proposals, longest, TWO * sizeof(uint32_t)) == 0, "as many as asked");
    draft_lookup_keep(&lookup, LOOKUP_KEPT);
    expect(lookup.count == LOOKUP_KEPT && draft_lookup_propose(&lookup, DRAFT_POSITIONS, proposals) == 0,
           "a kept prefix: the buckets of the tokens behind it are not taken");
    for (uint32_t index = 0; index < sizeof(behind_kept) / sizeof(behind_kept[0]); index++) {
        draft_lookup_append(&lookup, behind_kept[index]);
    }
    expect(draft_lookup_propose(&lookup, DRAFT_POSITIONS, proposals) == sizeof(after_kept) / sizeof(after_kept[0]) &&
           memcmp(proposals, after_kept, sizeof(after_kept)) == 0,
           "then what the new tokens hold, and past them the tokens of their period");
    free(buckets);
}

/* The scripted drafter (engine_internal.h's engine_draft_function): the reference tokens behind the anchor, as many as
 * the engine allows and the first round's limit, one made wrong as the pattern says; where the round must end noted. */
static uint32_t scripted_draft(struct inference_engine *engine,
			       const struct command_cursor *cursor,
			       uint32_t anchor_position, uint32_t most,
			       uint32_t *proposals_out)
{
    const uint32_t slot = cursor->command.session_slot;
    const uint32_t *const truth = drafted.tokens[slot] + anchor_position + 1u;     /* the draws of the round's rows */
    const uint32_t known = drafted.token_counts[slot] > anchor_position + 1u ?
                           drafted.token_counts[slot] - anchor_position - 1u : 0;
    const uint32_t round = drafted.rounds[slot]++;
    uint32_t count = most < known ? most : known;
    drafted.first_anchor = round == 0 ? anchor_position : drafted.first_anchor;
    count = round == 0 && drafted.first_count < count ? drafted.first_count : count;
    uint32_t wrong = drafted.pattern == DRAFT_WRONG_CYCLE ? round % ROUND_INDICES : ROUND_INDICES;
    for (uint32_t index = 0; index < count; index++) {
        wrong = drafted.pattern == DRAFT_END_WRONG && truth[index] == END && wrong == ROUND_INDICES ? index : wrong;
        proposals_out[index] = index == wrong ? (truth[index] + 1u) % BYTE_VALUES : truth[index];
    }
    uint32_t ended = 0;
    while (ended < count && proposals_out[ended] == truth[ended] && truth[ended] != END) {
        ended++;
    }
    const enum round_end how = ended < count && truth[ended] != END ? ROUND_REJECTED
                             : ended < count && proposals_out[ended] == END ? ROUND_END_ACCEPTED
                             : ended < count ? ROUND_END_CORRECTED
                             : ended < known && truth[ended] == END ? ROUND_END_BONUS : ROUND_ALL_RIGHT;
    drafted.ends[how] |= count != 0 ? 1u << ended : 0u;
    drafted.accepted += ended;          /* the engine accepts proposals as long as the reference's draws */
    (void)engine;
    return count;
}

/* A command of the function model for slot: its prompt from the slot's place in the prompt, greedy for seed case 0,
 * else sampled with the case's seed; its tag the slot's plus one. */
static struct generation_command_record function_command(uint32_t slot,
							 uint32_t seed_case,
							 uint32_t token_count,
							 uint32_t limit)
{
	return (struct generation_command_record){
		.new_token_ids = prompt + (size_t)slot * SLOT_PROMPT_STRIDE,
		.sampler_seed = seed_case * SEED_STRIDE,
		.session_slot = slot,
		.new_token_count = token_count,
		.generated_token_limit = limit,
		.sampler_temperature = seed_case == 0 ? 0.0f :
							SAMPLED_TEMPERATURE,
		.sampler_top_p = 1.0f,
		.request_tag = slot + 1u
	};
}

/* The fixture's engine as last set, drafting with draft where it is not NULL (in the lookup drafter's place). */
static struct inference_engine *drafting_engine(uint32_t context,
						engine_draft_function draft)
{
	struct inference_engine *engine = NULL;
	expect(test_serving_open(context, scripts,
				 sizeof(scripts) / sizeof(scripts[0]),
				 &engine) == STATUS_OK,
	       "an engine of the function model");
	if (engine != NULL && draft != NULL) {
		engine->draft = draft;
	}
    return engine;
}

/* One command on a new engine as last set: its events and their count, the steps it took in *steps_out (if any). */
static uint32_t command_events(uint32_t context,
			       const struct generation_command_record *command,
			       engine_draft_function draft,
			       struct generation_event *events,
			       uint64_t *steps_out)
{
	struct inference_engine *const engine = drafting_engine(context, draft);
	uint32_t count = 0;
	if (engine != NULL) {
		expect(engine_submit(engine, command) == STATUS_OK,
		       "submitted");
		count = events_until_final(engine, events);
		if (draft != NULL) {
			struct draft_statistics statistics = { 0 };
			expect(engine_draft_statistics(
				       engine, command->session_slot,
				       &statistics, NULL) == STATUS_OK,
			       "scripted drafter statistics read");
			uint64_t proposed = 0, accepted = 0;
			for (uint32_t position = 0;
			     position < DRAFT_STAT_POSITIONS; position++) {
				proposed += statistics.source[DRAFT_LOOKUP]
						    .proposed[position];
				accepted += statistics.source[DRAFT_LOOKUP]
						    .accepted[position];
			}
			const struct command_cursor *const cursor =
				&engine->cursors[command->session_slot];
			expect(proposed == cursor->proposals &&
				       accepted == cursor->accepted &&
				       statistics.source[DRAFT_LOOKUP].rounds ==
					       cursor->rounds,
			       "scripted greedy and sampled acceptance is counted by position");
		}
    }
    if (steps_out != NULL) {
        *steps_out = test_serving_step_count();
    }
    test_serving_close();
    return count;
}

/* The tokens a slot's session holds after a reference run, for the scripted drafter: those it read, then its draws. */
static void reference_taken(uint32_t slot, const uint32_t *read_tokens,
			    uint32_t read_count,
			    const struct generation_event *events,
			    uint32_t event_count)
{
    memcpy(drafted.tokens[slot], read_tokens, read_count * sizeof(uint32_t));
    drafted.token_counts[slot] = read_count;
    for (uint32_t index = 0; index < event_count && drafted.token_counts[slot] <= SPECULATIVE_CONTEXT; index++) {
	    if (events[index].event_kind == GENERATION_EVENT_TOKEN) {
		    drafted.tokens[slot][drafted.token_counts[slot]++] =
			    events[index].token_id;
	    }
    }
    drafted.rounds[slot] = 0;
}

/* Whether the events are the reference's, every byte. */
static bool events_match(const struct generation_event *events, uint32_t count,
			 const struct generation_event *reference,
			 uint32_t reference_count)
{
    return count != 0 && count == reference_count && memcmp(events, reference, count * sizeof(*events)) == 0;
}

/* Rounds against steps of one drawing row on the function model, greedy and with three seeds: proposals of the
 * reference run with one wrong at each index in turn, all right, all right but the end token's, each with the first
 * round cut to every count, and the lookup drafter's, every event the same; the scripted drafter's first round at the
 * prompt's last row, and each proposal accepted a step fewer than the reference's; over the cases, rounds end at every
 * index rejected and all right, at an end token proposed and at an end token in another's place. A drafter of as many
 * proposals as the engine's rows is refused at create. */
static void test_speculative_rounds(void)
{
	static struct generation_event reference[MOST_EVENTS],
		events[MOST_EVENTS];
	struct inference_engine *engine = NULL;
	memset(&drafted, 0, sizeof(drafted));
	test_serving_set_function_model(true, 0);
	context_text =
		"speculation: a drafter of as many proposals as the rows";
	test_serving_set_engine(1, 1, SPECULATIVE_ROWS, SPECULATIVE_ROWS);
	expect(test_serving_open(SPECULATIVE_CONTEXT, scripts,
				 sizeof(scripts) / sizeof(scripts[0]),
				 &engine) == STATUS_INVALID_ARGUMENT &&
		       engine == NULL,
	       "refused at create");
	for (uint32_t seed_case = 0; seed_case < SEED_CASES; seed_case++) {
		const struct generation_command_record command =
			function_command(0, seed_case, SPECULATIVE_PROMPT,
					 SPECULATIVE_LIMIT);
		uint64_t reference_steps = 0, steps = 0;
		context_text = "speculation: the reference";
		test_serving_set_engine(1, 1, SPECULATIVE_ROWS, 0);
		const uint32_t reference_count =
			command_events(SPECULATIVE_CONTEXT, &command, NULL,
				       reference, &reference_steps);
		reference_taken(0, command.new_token_ids,
				command.new_token_count, reference,
				reference_count);
		test_serving_set_engine(1, 1, SPECULATIVE_ROWS,
					DRAFT_POSITIONS);
		context_text = "speculation: the lookup drafter";
		uint32_t count = command_events(SPECULATIVE_CONTEXT, &command,
						NULL, events, NULL);
		expect(events_match(events, count, reference, reference_count),
		       "the events of steps of one drawing row");
		for (uint32_t pattern = 0; pattern < DRAFT_PATTERNS;
		     pattern++) {
			for (uint32_t first = 0; first < ROUND_INDICES;
			     first++) {
				context_text =
					"speculation: a scripted drafter";
				drafted.pattern = pattern;
				drafted.first_count = first;
				drafted.rounds[0] = 0;
				drafted.accepted = 0;
				count = command_events(SPECULATIVE_CONTEXT,
						       &command, scripted_draft,
						       events, &steps);
				expect(events_match(events, count, reference,
						    reference_count),
				       "the events of steps of one drawing row");
				expect(drafted.first_anchor ==
						       SPECULATIVE_PROMPT -
							       1u &&
					       steps + drafted.accepted ==
						       reference_steps,
				       "the first round at the prompt's last row, and a step fewer for each proposal accepted");
			}
		}
    }
    context_text = "speculation: where rounds ended";
    expect((drafted.ends[ROUND_REJECTED] & PROPOSAL_INDICES) == PROPOSAL_INDICES &&
           (drafted.ends[ROUND_ALL_RIGHT] & (1u << DRAFT_POSITIONS)) != 0,
           "rejected at every proposal, and all seven right");
    expect((drafted.ends[ROUND_END_ACCEPTED] & PROPOSAL_INDICES) == PROPOSAL_INDICES &&
           (drafted.ends[ROUND_END_CORRECTED] & PROPOSAL_INDICES) == PROPOSAL_INDICES,
           "at the end token proposed, and drawn in another's place, at every index");
}

/* max_tokens at every index of a round (1 to 16 tokens) and the context's end inside a round, with all right
 * proposals (the context's end behind first rounds of every length, so that the last anchor lies at every distance from
 * it), without a pipeline and in a pipeline of six steps: the events of the serial steps of one drawing row; the limit
 * and the context's end reached. */
static void test_speculative_limits(void)
{
	static struct generation_event reference[MOST_EVENTS],
		events[MOST_EVENTS];
	static const uint32_t depths[] = { 1, PIPELINE_DEPTH };
	uint32_t limit_indices = 0, context_ends = 0;
	test_serving_set_function_model(true, 0);
	drafted.pattern = DRAFT_RIGHT;
	for (uint32_t seed_case = 0; seed_case < SEED_CASES; seed_case++) {
		for (uint32_t limit = 1; limit <= LIMIT_CASES + ROUND_INDICES;
		     limit++) {
			const bool is_context_end = limit > LIMIT_CASES;
			const uint32_t context =
				is_context_end ? SHORT_SPECULATIVE_CONTEXT :
						 SPECULATIVE_CONTEXT;
			const struct generation_command_record command =
				function_command(
					0, seed_case,
					is_context_end ?
						SHORT_SPECULATIVE_PROMPT :
						SPECULATIVE_PROMPT,
					is_context_end ? SPECULATIVE_LIMIT :
							 limit);
			context_text =
				is_context_end ?
					"speculation: the context's end" :
					"speculation: max_tokens";
			test_serving_set_engine(1, 1, SPECULATIVE_ROWS, 0);
			const uint32_t reference_count = command_events(
				context, &command, NULL, reference, NULL);
			reference_taken(0, command.new_token_ids,
					command.new_token_count, reference,
					reference_count);
			drafted.first_count = is_context_end ?
						      limit - LIMIT_CASES - 1u :
						      ROUND_INDICES;
			for (uint32_t depth = 0;
			     depth < sizeof(depths) / sizeof(depths[0]);
			     depth++) {
				test_serving_set_engine(1, depths[depth],
							SPECULATIVE_ROWS,
							DRAFT_POSITIONS);
				drafted.rounds[0] = 0;
				const uint32_t count = command_events(
					context, &command, scripted_draft,
					events, NULL);
				expect(events_match(events, count, reference,
						    reference_count),
				       "the events of the serial steps of one drawing row");
			}
			const struct generation_event final =
				reference[reference_count != 0 ?
						  reference_count - 1u :
						  0];
			const bool is_limit =
				final.event_kind ==
				GENERATION_EVENT_FINISHED_AT_LIMIT;
			limit_indices |=
				is_limit && !is_context_end ?
					1u << ((limit - 1u) % ROUND_INDICES) :
					0u;
			context_ends +=
				is_limit && is_context_end &&
						final.session_token_count ==
							context + 1u ?
					1u :
					0u;
		}
    }
    context_text = "speculation: limits";
    expect(limit_indices == (1u << ROUND_INDICES) - 1u && context_ends >= ROUND_INDICES,
           "the limit reached at every index of a round, and the context's end behind every first round");
}

/* The lookup drafter in the engine on a repetitive prompt (the copy model, whose draws go on repeating it): the
 * events of steps of one drawing row, in a fraction of their steps. */
static void test_lookup_rounds(void)
{
	static struct generation_event reference[MOST_EVENTS],
		events[MOST_EVENTS];
	uint64_t reference_steps = 0, steps = 0;
	struct generation_command_record command =
		function_command(0, 0, COPY_PROMPT, COPY_LIMIT);
	command.new_token_ids = copy_prompt;
	context_text = "speculation: the lookup drafter on a repetitive prompt";
	test_serving_set_function_model(true, COPY_PERIOD);
	test_serving_set_engine(1, 1, SPECULATIVE_ROWS, 0);
	const uint32_t reference_count =
		command_events(SPECULATIVE_CONTEXT, &command, NULL, reference,
			       &reference_steps);
	test_serving_set_engine(1, 1, SPECULATIVE_ROWS, DRAFT_POSITIONS);
	const uint32_t count = command_events(SPECULATIVE_CONTEXT, &command,
					      NULL, events, &steps);
	expect(events_match(events, count, reference, reference_count) &&
		       reference_count == COPY_LIMIT + 1u &&
		       steps * LOOKUP_SPEEDUP <= reference_steps,
	       "the events of steps of one drawing row, in a quarter of them");
}

/* The next event, waiting on the descriptor, filed under its slot (by tag) behind the counts[slot] before it; in
 * *final_slot_out its slot if it is a final event, else SESSIONS. false when none comes within the poll's time. */
static bool event_filed(struct inference_engine *engine,
			struct generation_event (*events)[MOST_EVENTS],
			uint32_t *counts, uint32_t *final_slot_out)
{
    int descriptor = -1;
    struct generation_event event = { 0 };
    bool is_available = false;
    expect(engine_event_descriptor(engine, &descriptor) == STATUS_OK,
	   "the descriptor");
    struct pollfd poll_entry = {descriptor, POLLIN, 0};
    if (poll(&poll_entry, 1, POLL_MILLISECONDS) != 1) {
        expect(false, "an event comes within the poll's time");
        return false;
    }
    expect(engine_next_event(engine, &event, &is_available) == STATUS_OK,
	   "next_event");
    const uint32_t slot = event.request_tag - 1u;
    *final_slot_out = SESSIONS;
    if (is_available && slot < SESSIONS && counts[slot] < MOST_EVENTS) {
        events[slot][counts[slot]++] = event;
	*final_slot_out = event.event_kind != GENERATION_EVENT_TOKEN ? slot :
								       SESSIONS;
    }
    return true;
}

/* The commands, one a slot, on a new engine of the context as last set, and the events of each until its final; the
 * steps they took. */
static uint64_t sessions_run(uint32_t context,
			     const struct generation_command_record *commands,
			     engine_draft_function draft,
			     struct generation_event (*events)[MOST_EVENTS],
			     uint32_t *counts)
{
	struct inference_engine *const engine = drafting_engine(context, draft);
	uint32_t finals = 0, final_slot = SESSIONS;
	memset(counts, 0, SESSIONS * sizeof(*counts));
	for (uint32_t slot = 0; slot < SESSIONS && engine != NULL; slot++) {
		expect(engine_submit(engine, &commands[slot]) == STATUS_OK,
		       "submitted");
	}
    while (engine != NULL && finals < SESSIONS && event_filed(engine, events, counts, &final_slot)) {
        finals += final_slot < SESSIONS ? 1u : 0u;
    }
    const uint64_t steps = test_serving_step_count();
    test_serving_close();
    return steps;
}

/* Four sessions, without a pipeline (rows of several owners a step, the eager driver) and in a pipeline of six steps,
 * greedy and sampled: each session's events those of the same engine without speculation, with proposals wrong at
 * each index in turn and with all right; every session drafts rounds, and with all right they take fewer steps. The
 * same events at the end of a short context, where several sessions stop one read earlier. */
static void test_speculative_pipeline(void)
{
	static struct generation_event reference[SESSIONS][MOST_EVENTS],
		events[SESSIONS][MOST_EVENTS];
	static const uint32_t depths[] = { 1, PIPELINE_DEPTH };
	uint32_t reference_counts[SESSIONS], counts[SESSIONS];
	struct generation_command_record commands[SESSIONS];
	test_serving_set_function_model(true, 0);
	for (uint32_t ending = 0; ending < TWO; ending++) {
		const uint32_t context = ending != 0 ?
						 SHORT_SPECULATIVE_CONTEXT :
						 SPECULATIVE_CONTEXT;
		for (uint32_t slot = 0; slot < SESSIONS; slot++) {
			commands[slot] = function_command(
				slot, slot % SEED_CASES,
				ending != 0 ? SHORT_SPECULATIVE_PROMPT + slot :
					      SPECULATIVE_PROMPT,
				SESSION_LIMIT);
		}
		for (uint32_t depth = 0;
		     depth < sizeof(depths) / sizeof(depths[0]); depth++) {
			context_text = "four sessions: the reference";
			test_serving_set_engine(SESSIONS, depths[depth],
						SPECULATIVE_ROWS, 0);
			const uint64_t reference_steps =
				sessions_run(context, commands, NULL, reference,
					     reference_counts);
			for (uint32_t slot = 0; slot < SESSIONS; slot++) {
				reference_taken(slot,
						commands[slot].new_token_ids,
						commands[slot].new_token_count,
						reference[slot],
						reference_counts[slot]);
			}
			test_serving_set_engine(SESSIONS, depths[depth],
						SPECULATIVE_ROWS,
						DRAFT_POSITIONS);
			for (uint32_t pattern = DRAFT_WRONG_CYCLE;
			     pattern <= DRAFT_RIGHT; pattern++) {
				context_text =
					ending != 0 ?
						"four sessions at the context's end" :
					pattern == DRAFT_RIGHT ?
						"four sessions: all right" :
						"four sessions: wrong in turn";
				drafted.pattern = pattern;
				drafted.first_count = ROUND_INDICES;
				memset(drafted.rounds, 0,
				       sizeof(drafted.rounds));
				const uint64_t steps = sessions_run(
					context, commands, scripted_draft,
					events, counts);
				uint32_t drafting = 0;
				for (uint32_t slot = 0; slot < SESSIONS;
				     slot++) {
					expect(events_match(
						       events[slot],
						       counts[slot],
						       reference[slot],
						       reference_counts[slot]),
					       "every session's events those of steps of one drawing row");
					drafting += drafted.rounds[slot] != 0 ?
							    1u :
							    0u;
				}
				expect(ending != 0 ||
					       (drafting == SESSIONS &&
						(pattern != DRAFT_RIGHT ||
						 steps < reference_steps)),
				       "every session drafts rounds; with all right, in fewer steps than without");
			}
		}
    }
}

/* Front steps (the fixture's cone) under speculation: a long prompt's rows before its cone and margin run the front
 * alone, its last row and the copy model's tokens go by rounds of the lookup drafter, wrong at first (what followed
 * the suffix earlier in the prompt), then right; the events those of the serial steps of one drawing row. */
static void test_speculative_cone(void)
{
	static struct generation_event reference[MOST_EVENTS],
		events[MOST_EVENTS];
	struct generation_command_record command =
		function_command(0, 0, CONE_PROMPT, COPY_LIMIT);
	command.new_token_ids = cone_prompt;
	context_text = "speculation: front steps";
	test_serving_set_cone(CONE);
	test_serving_set_function_model(true, COPY_PERIOD);
	test_serving_set_engine(1, 1, SPECULATIVE_ROWS, 0);
	const uint32_t reference_count =
		command_events(CONE_CONTEXT, &command, NULL, reference, NULL);
	test_serving_set_engine(1, 1, SPECULATIVE_ROWS, DRAFT_POSITIONS);
	const uint32_t count =
		command_events(CONE_CONTEXT, &command, NULL, events, NULL);
	expect(events_match(events, count, reference, reference_count) &&
		       reference_count == COPY_LIMIT + 1u,
	       "the events of steps of one drawing row");
	test_serving_set_cone(0);
}

/* A cancel between rounds (the copy model, the lookup drafter's proposals): the tokens before it are those of steps
 * of one drawing row, the final event says what the session holds; then a command that keeps what the session ran
 * and brings new tokens gives the events of a command that reads them all anew, without speculation. */
static void test_speculative_cancel(bool device)
{
	static struct generation_event reference[MOST_EVENTS],
		events[MOST_EVENTS];
	static uint32_t tokens[SPECULATIVE_CONTEXT + 1u];
	struct generation_command_record command =
		function_command(0, 0, COPY_PROMPT, SPECULATIVE_LIMIT);
	command.new_token_ids = copy_prompt;
	context_text = "speculation: a cancel between rounds";
	test_serving_set_function_model(true, COPY_PERIOD);
	test_serving_set_engine(1, 1, SPECULATIVE_ROWS, 0);
	const uint32_t reference_count = command_events(
		SPECULATIVE_CONTEXT, &command, NULL, reference, NULL);
	test_serving_set_device_draft(device, MODEL_DRAFT_BLOCK);
	test_serving_set_engine(1, device ? PIPELINE_DEPTH : 1u,
				SPECULATIVE_ROWS,
				device ? MODEL_DRAFT_BLOCK : DRAFT_POSITIONS);
	struct inference_engine *const engine =
		drafting_engine(SPECULATIVE_CONTEXT, NULL);
	if (engine == NULL) {
		return;
	}
    test_serving_set_step_delay(STEP_DELAY_NANOSECONDS);
    expect(engine_submit(engine, &command) == STATUS_OK, "submitted");
    for (uint32_t look = 0; look < MOST_LOOKS && test_serving_step_count() < CANCEL_STEPS; look++) {
        pause_for(STEP_DELAY_NANOSECONDS);
    }
    expect(engine_cancel(engine, 0) == STATUS_OK, "cancelled");
    uint32_t count = events_until_final(engine, events);
    test_serving_set_step_delay(0);
    const struct generation_event final = events[count != 0 ? count - 1u : 0];
    expect(final.event_kind == GENERATION_EVENT_CANCELLED && count > 1u &&
		   count < reference_count &&
		   memcmp(events, reference, (count - 1u) * sizeof(*events)) ==
			   0 &&
		   final.session_token_count == COPY_PROMPT + count - 1u,
	   "the tokens before it those of steps of one drawing row, the session holding them");
    memcpy(tokens, copy_prompt, sizeof(copy_prompt));
    for (uint32_t index = 0; index + 1u < count; index++) {
        tokens[COPY_PROMPT + index] = events[index].token_id;
    }
    const uint32_t kept = (final.session_token_count - 1u) / KEPT_TOKENS * KEPT_TOKENS;     /* of those it ran */
    for (uint32_t index = 0; index < NEW_TOKENS; index++) {
        tokens[kept + index] = NEW_TOKEN + index;
    }
    struct generation_command_record next =
	    function_command(0, 0, NEW_TOKENS, CONTINUED_LIMIT);
    next.new_token_ids = tokens + kept;
    next.kept_token_count = kept;
    expect(engine_submit(engine, &next) == STATUS_OK,
	   "a command keeps what the session ran");
    count = events_until_final(engine, events);
    test_serving_close();
    test_serving_set_device_draft(false, MODEL_DRAFT_BLOCK);
    next.new_token_ids = tokens;
    next.new_token_count = kept + NEW_TOKENS;
    next.kept_token_count = 0;
    test_serving_set_engine(1, 1, SPECULATIVE_ROWS, 0);
    const uint32_t anew_count = command_events(SPECULATIVE_CONTEXT, &next, NULL, reference, NULL);
    expect(events_match(events, count, reference, anew_count), "the events of a command that reads them all anew");
}

/* A snapshot after rounds (front_end_seam.h's session commands): saved, loaded into the engine of a restart and
 * continued, with proposals wrong at each index in turn; the events those of steps of one drawing row, the continued
 * command's those of a command that reads all its tokens anew. */
static void test_speculative_snapshot(bool device)
{
	static struct generation_event reference[MOST_EVENTS],
		anew[MOST_EVENTS], events[MOST_EVENTS];
	static uint32_t tokens[SPECULATIVE_CONTEXT + 1u];
	struct session_state saved = { 0 };
	char path[] = "/tmp/anemone_engine_speculative_XXXXXX";
	const int descriptor = mkstemp(path);
	context_text = "speculation: a snapshot after rounds";
	expect(descriptor >= 0 && unlink(path) == 0, "a file for the snapshot");
	if (descriptor < 0) {
		return;
	}
    test_serving_set_function_model(true, 0);
    test_serving_set_engine(1, 1, SPECULATIVE_ROWS, 0);
    const struct generation_command_record first =
	    function_command(0, 1, SPECULATIVE_PROMPT, SNAPSHOT_LIMIT);
    const uint32_t first_count = command_events(SPECULATIVE_CONTEXT, &first, NULL, reference, NULL);
    reference_taken(0, first.new_token_ids, first.new_token_count, reference, first_count);
    const uint32_t held = reference[first_count != 0 ? first_count - 1u : 0].session_token_count;
    const uint32_t kept = held > 1u ? (held - 1u) / KEPT_TOKENS * KEPT_TOKENS : 0;            /* of those it ran */
    memcpy(tokens, drafted.tokens[0], kept * sizeof(uint32_t));
    for (uint32_t index = 0; index < NEW_TOKENS; index++) {
        tokens[kept + index] = NEW_TOKEN + index;
    }
    struct generation_command_record next =
	    function_command(0, 1, kept + NEW_TOKENS, CONTINUED_LIMIT);
    next.new_token_ids = tokens;
    const uint32_t anew_count = command_events(SPECULATIVE_CONTEXT, &next, NULL, anew, NULL);
    next.new_token_ids = tokens + kept;
    next.new_token_count = NEW_TOKENS;
    next.kept_token_count = kept;
    test_serving_set_device_draft(device, device ? 0 : MODEL_DRAFT_BLOCK);
    test_serving_set_engine(1, 1, SPECULATIVE_ROWS,
			    device ? MODEL_DRAFT_BLOCK : DRAFT_POSITIONS);
    drafted.pattern = DRAFT_WRONG_CYCLE;
    drafted.first_count = ROUND_INDICES;
    struct inference_engine *engine = drafting_engine(
	    SPECULATIVE_CONTEXT, device ? NULL : scripted_draft);
    const struct session_command save = {
	    NULL, descriptor, 0, kept, SESSION_SAVE, SESSION_TAG, 0, 0, 0
    };
    expect(engine != NULL && engine_submit(engine, &first) == STATUS_OK &&
		   events_match(events, events_until_final(engine, events),
				reference, first_count),
	   "the first command's events those of steps of one drawing row");
    expect(engine != NULL &&
		   engine_session_state(engine, 0, &saved) == STATUS_OK,
	   "the saving slot's actual reach");
    expect(engine != NULL && session_done(engine, &save).event_kind ==
				     GENERATION_EVENT_FINISHED_AT_LIMIT,
	   "the session saved");
    test_serving_close();
    reference_taken(0, tokens, kept + NEW_TOKENS, anew, anew_count);
    engine = drafting_engine(SPECULATIVE_CONTEXT, device ? NULL : scripted_draft);
    const struct session_command load = { tokens,
					  descriptor,
					  0,
					  kept,
					  SESSION_LOAD,
					  SESSION_TAG,
					  saved.reach_count,
					  0,
					  saved.decoder_from };
    expect(engine != NULL && lseek(descriptor, 0, SEEK_SET) == 0 &&
           session_done(engine, &load).session_token_count == kept, "loaded by the engine of a restart");
    expect(engine != NULL && engine_submit(engine, &next) == STATUS_OK &&
		   events_match(events, events_until_final(engine, events),
				anew, anew_count),
	   "the continued command's events those of a command that reads all its tokens anew");
    test_serving_close();
    test_serving_set_device_draft(false, MODEL_DRAFT_BLOCK);
    (void)close(descriptor);
}

/* A prompt beside a session's rounds without a pipeline (two slots, both owners' rows in each step): the first
 * session decodes the copy model by rounds of the lookup drafter, and the other's prompt, which comes once those
 * rounds have begun, takes no more steps to its token than beside steps of one drawing row, give or take
 * PREFILL_SLACK_STEPS: proposals take only the rows no prompt takes (until 26 October 2026 they took the rows of a
 * prompt behind them in turn, and the prompt about twice its steps). Each session's events are those of steps of one
 * drawing row. */
static void test_prefill_beside_rounds(void)
{
	static struct generation_event events[TWO][SESSIONS][MOST_EVENTS];
	uint32_t counts[TWO][SESSIONS], prompt_steps[TWO] = { 0 };
	struct generation_command_record decoder =
		function_command(0, 0, COPY_PROMPT, DECODER_LIMIT);
	const struct generation_command_record prefill =
		function_command(1, 0, PREFILL_PROMPT, 1);
	decoder.new_token_ids = copy_prompt;
	context_text = "speculation: a prompt beside a session's rounds";
	test_serving_set_function_model(true, COPY_PERIOD);
	for (uint32_t speculates = 0; speculates < TWO; speculates++) {
		uint32_t finals = 0, final_slot = SESSIONS;
		memset(counts[speculates], 0, sizeof(counts[speculates]));
		test_serving_set_engine(TWO, 1, SPECULATIVE_ROWS,
					speculates != 0 ? DRAFT_POSITIONS : 0);
		struct inference_engine *const engine =
			drafting_engine(SPECULATIVE_CONTEXT, NULL);
		test_serving_set_step_delay(STEP_DELAY_NANOSECONDS);
		expect(engine != NULL &&
			       engine_submit(engine, &decoder) == STATUS_OK,
		       "the decoding session");
		for (uint32_t look = 0;
		     look < MOST_LOOKS &&
		     test_serving_step_count() < DECODER_STEPS;
		     look++) {
			pause_for(STEP_DELAY_NANOSECONDS);
		}
		const uint64_t begun = test_serving_step_count();
		expect(engine != NULL &&
			       engine_submit(engine, &prefill) == STATUS_OK,
		       "the prompt");
		while (engine != NULL && finals < TWO &&
		       event_filed(engine, events[speculates],
				   counts[speculates], &final_slot)) {
			finals += final_slot < SESSIONS ? 1u : 0u;
			if (final_slot == prefill.session_slot) {
				prompt_steps[speculates] =
					(uint32_t)(test_serving_step_count() -
						   begun);
				test_serving_set_step_delay(0);
			}
		}
		test_serving_set_step_delay(0);
		test_serving_close();
	}
    for (uint32_t slot = 0; slot < TWO; slot++) {
        expect(events_match(events[1][slot], counts[1][slot], events[0][slot], counts[0][slot]),
               "each session's events those of steps of one drawing row");
    }
    expect(prompt_steps[0] != 0 && prompt_steps[1] <= prompt_steps[0] + PREFILL_SLACK_STEPS,
           "the prompt's steps to its token those beside steps of one drawing row, give or take");
}

/* --prefill_order (front_end_seam.h): two sessions' prompts of ORDER_PROMPT rows on the function model, submitted
 * together, without and with the knob (and with prefill_first): with it the first command's prompt runs whole first,
 * its first token within its own prompt's steps, and the second's first token at least its own prompt's steps later;
 * every session's events the same either way. */
static void test_prefill_order(void)
{
	static struct generation_event events[TWO][SESSIONS][MOST_EVENTS];
	uint32_t counts[TWO][SESSIONS];
	uint64_t first_token_steps[TWO][TWO] = { { 0 } };
	const uint64_t prompt_steps =
		(ORDER_PROMPT + TEST_SERVING_MOST_ROWS - 1u) /
		TEST_SERVING_MOST_ROWS;
	context_text =
		"prefill_order: prompts one at a time, in their commands' order";
	test_serving_set_function_model(true, 0);
	for (uint32_t ordered = 0; ordered < TWO; ordered++) {
		uint32_t finals = 0, final_slot = SESSIONS;
		memset(counts[ordered], 0, sizeof(counts[ordered]));
		test_serving_set_engine(TWO, 1, TEST_SERVING_MOST_ROWS, 0);
		test_serving_set_prefill(ordered, ordered, 0);
		struct inference_engine *const engine =
			drafting_engine(SPECULATIVE_CONTEXT, NULL);
		test_serving_set_step_delay(STEP_DELAY_NANOSECONDS);
		const uint64_t begun = test_serving_step_count();
		for (uint32_t slot = 0; slot < TWO && engine != NULL; slot++) {
			const struct generation_command_record command =
				function_command(slot, 0, ORDER_PROMPT,
						 ORDER_LIMIT);
			expect(engine_submit(engine, &command) == STATUS_OK,
			       "submitted");
		}
		while (engine != NULL && finals < TWO &&
		       event_filed(engine, events[ordered], counts[ordered],
				   &final_slot)) {
			finals += final_slot < SESSIONS ? 1u : 0u;
			for (uint32_t slot = 0; slot < TWO; slot++) {
				first_token_steps[ordered][slot] =
					first_token_steps[ordered][slot] == 0 &&
							counts[ordered][slot] !=
								0 ?
						test_serving_step_count() -
							begun :
						first_token_steps[ordered][slot];
			}
		}
		test_serving_set_step_delay(0);
		test_serving_close();
	}
    test_serving_set_prefill(0, 0, 0);
    for (uint32_t slot = 0; slot < TWO; slot++) {
        expect(events_match(events[1][slot], counts[1][slot], events[0][slot], counts[0][slot]),
               "each session's events those of the turns' order");
    }
    expect(first_token_steps[1][0] != 0 && first_token_steps[1][0] <= prompt_steps + ORDER_SLACK_STEPS &&
               first_token_steps[1][1] >= first_token_steps[1][0] + prompt_steps - ORDER_SLACK_STEPS,
           "the first prompt's token within its own steps, the second's a prompt's steps behind it");
}

/* --prefill_order=2 (front_end_seam.h PREFILL_ORDER_SHORTEST): a long prompt of ORDER_PROMPT rows submitted
 * just before a short one of ORDER_SHORT_PROMPT rows on the function model, in the turns' order, in order 1, in order 2
 * and in order 2 with prefill_order_wait ORDER_WAIT: every session's events the same each time; in order 1 the short
 * prompt's first token at least the long prompt's steps late, in order 2 within its own steps and the long one's
 * behind both prompts, with the wait within twice the short prompt's steps. */
static void test_prefill_order_shortest(void)
{
	enum {
		TURNS,
		OLDEST,
		SHORTEST,
		WAITED,
		SHORT_ROWS,
		SHORT_ROWS_WAITED,
		RUNS
	};
	static const uint32_t orders[RUNS] = { 0,
					       1,
					       PREFILL_ORDER_SHORTEST,
					       PREFILL_ORDER_SHORTEST,
					       PREFILL_ORDER_SHORT_ROWS,
					       PREFILL_ORDER_SHORT_ROWS };
	static const uint32_t waits[RUNS] = {
		0, 0, 0, ORDER_WAIT, 0, ORDER_WAIT
	};
	static const uint32_t prompts[TWO] = { ORDER_PROMPT,
					       ORDER_SHORT_PROMPT };
	static struct generation_event events[RUNS][SESSIONS][MOST_EVENTS];
	uint32_t counts[RUNS][SESSIONS];
	uint64_t first_token_steps[RUNS][TWO] = { { 0 } };
	const uint64_t long_steps =
		(ORDER_PROMPT + TEST_SERVING_MOST_ROWS - 1u) /
		TEST_SERVING_MOST_ROWS;
	const uint64_t short_steps =
		(ORDER_SHORT_PROMPT + TEST_SERVING_MOST_ROWS - 1u) /
		TEST_SERVING_MOST_ROWS;
	context_text =
		"prefill_order 2: the prompt with the fewest rows left first";
	test_serving_set_function_model(true, 0);
	for (uint32_t run = 0; run < RUNS; run++) {
		uint32_t finals = 0, final_slot = SESSIONS;
		memset(counts[run], 0, sizeof(counts[run]));
		test_serving_set_engine(TWO, 1, TEST_SERVING_MOST_ROWS, 0);
		test_serving_set_prefill(orders[run] != 0 ? 1u : 0u,
					 orders[run], waits[run]);
		struct inference_engine *const engine =
			drafting_engine(SPECULATIVE_CONTEXT, NULL);
		test_serving_set_step_delay(STEP_DELAY_NANOSECONDS);
		const uint64_t begun = test_serving_step_count();
		for (uint32_t slot = 0; slot < TWO && engine != NULL; slot++) {
			const struct generation_command_record command =
				function_command(slot, 0, prompts[slot],
						 ORDER_LIMIT);
			expect(engine_submit(engine, &command) == STATUS_OK,
			       "submitted");
		}
		while (engine != NULL && finals < TWO &&
		       event_filed(engine, events[run], counts[run],
				   &final_slot)) {
			finals += final_slot < SESSIONS ? 1u : 0u;
			for (uint32_t slot = 0; slot < TWO; slot++) {
				first_token_steps[run][slot] =
					first_token_steps[run][slot] == 0 &&
							counts[run][slot] != 0 ?
						test_serving_step_count() -
							begun :
						first_token_steps[run][slot];
			}
		}
		test_serving_set_step_delay(0);
		test_serving_close();
	}
    test_serving_set_prefill(0, 0, 0);
    for (uint32_t run = OLDEST; run < RUNS; run++) {
        for (uint32_t slot = 0; slot < TWO; slot++) {
            expect(events_match(events[run][slot], counts[run][slot], events[TURNS][slot], counts[TURNS][slot]),
                   "each session's events those of the turns' order");
        }
    }
    expect(first_token_steps[OLDEST][1] + ORDER_SLACK_STEPS >= long_steps,
           "in order 1 the short prompt's token behind the long prompt's steps");
    for (uint32_t run = SHORTEST; run <= SHORT_ROWS; run += TWO) {
	    expect(first_token_steps[run][1] != 0 &&
			   first_token_steps[run][1] <=
				   short_steps + ORDER_SLACK_STEPS &&
			   first_token_steps[run][0] + ORDER_SLACK_STEPS >=
				   long_steps + short_steps,
		   "the shortest order keeps long prompts behind short ones");
    }
    for (uint32_t run = WAITED; run <= SHORT_ROWS_WAITED; run += TWO) {
	    expect(first_token_steps[run][1] != 0 &&
			   first_token_steps[run][1] <=
				   short_steps * (ORDER_WAIT + 1u) +
					   ORDER_SLACK_STEPS &&
			   first_token_steps[run][1] + 1u >=
				   short_steps * (ORDER_WAIT + 1u),
		   "both shortest orders retain the oldest prompt's bounded wait");
    }
}

/* Short prompt coalescing at and around the prepared thin budget, with one and two flights. The complete
 * per-session event streams must match turns, including stochastic sampling; wall-clock timing is not asserted. */
static void test_prefill_short_rows(void)
{
	static const uint32_t orders[] = { 0, PREFILL_ORDER_SHORTEST,
					   PREFILL_ORDER_SHORT_ROWS };
	static const uint32_t prompts[][TWO] = {
		{ ORDER_TAIL_PROMPT, ORDER_TAIL_PROMPT },
		{ TEST_SERVING_MOST_ROWS, ORDER_TAIL_PROMPT },
		{ TEST_SERVING_MOST_ROWS + 1u, ORDER_TAIL_PROMPT },
		{ ORDER_PROMPT, ORDER_TAIL_PROMPT }
	};
	static struct generation_event baseline[SESSIONS][MOST_EVENTS],
		events[SESSIONS][MOST_EVENTS];
	uint32_t baseline_counts[SESSIONS], counts[SESSIONS];
	context_text =
		"short prompt rows: coalesced and serial event streams agree";
	test_serving_set_function_model(true, 0);
	for (uint32_t depth = 1; depth <= TWO; depth++) {
		for (uint32_t seed = 0; seed < TWO; seed++) {
			for (uint32_t shape = 0;
			     shape < sizeof(prompts) / sizeof(prompts[0]);
			     shape++) {
				for (uint32_t run = 0;
				     run < sizeof(orders) / sizeof(orders[0]);
				     run++) {
					uint32_t finals = 0,
						 final_slot = SESSIONS;
					memset(counts, 0, sizeof(counts));
					test_serving_set_engine(
						TWO, depth,
						TEST_SERVING_MOST_ROWS, 0);
					test_serving_set_prefill(
						1, orders[run],
						ORDER_WAIT * (orders[run] != 0 ?
								      1u :
								      0u));
					struct inference_engine *const engine =
						drafting_engine(
							SPECULATIVE_CONTEXT,
							NULL);
					test_serving_set_step_delay(
						STEP_DELAY_NANOSECONDS);
					for (uint32_t slot = 0;
					     slot < TWO && engine != NULL;
					     slot++) {
						const struct generation_command_record
							command = function_command(
								slot, seed,
								prompts[shape]
								       [slot],
								ORDER_LIMIT);
						expect(engine_submit(engine,
								     &command) ==
							       STATUS_OK,
						       "submitted short rows");
					}
					while (engine != NULL && finals < TWO &&
					       event_filed(engine, events,
							   counts,
							   &final_slot)) {
						finals +=
							final_slot < SESSIONS ?
								1u :
								0u;
					}
					expect(finals == TWO,
					       "both commands terminate");
					if (run == 0) {
						memcpy(baseline, events,
						       sizeof(baseline));
						memcpy(baseline_counts, counts,
						       sizeof(baseline_counts));
					} else {
						for (uint32_t slot = 0;
						     slot < TWO; slot++) {
							expect(events_match(
								       events[slot],
								       counts[slot],
								       baseline[slot],
								       baseline_counts
									       [slot]),
							       "all sampled token and final events match the turns");
						}
					}
					test_serving_set_step_delay(0);
					test_serving_close();
				}
			}
		}
	}
	test_serving_set_prefill(0, 0, 0);
}

/* test_lookup_depth's drafter: the engine's lookup drafter, each round's depth held to the rule the cursor's
 * statistics give since the last round (front_end_seam.h's rounds: 1 at a command's first, twice the last behind a
 * round all accepted, to the draft positions at most, else the last round's accepted proposals and one; unchanged
 * where no round came between), and its proposals to the depth. */
static uint32_t depth_watched_draft(struct inference_engine *engine,
				    const struct command_cursor *cursor,
				    uint32_t anchor_position, uint32_t most,
				    uint32_t *proposals_out)
{
    const uint32_t proposed = cursor->proposals - depth_watch.proposals;
    const uint32_t accepted = cursor->accepted - depth_watch.accepted;
    const uint32_t doubled = depth_watch.depth * TWO < DRAFT_POSITIONS ? depth_watch.depth * TWO : DRAFT_POSITIONS;
    const uint32_t ruled = cursor->rounds == 0                   ? 1u
                         : cursor->rounds == depth_watch.rounds ? depth_watch.depth
                         : accepted + accepted < proposed       ? 1u
                         : accepted < proposed                  ? accepted + 1u
                                                                : doubled;
    const uint32_t count = depth_watch.lookup(engine, cursor, anchor_position, most, proposals_out);
    depth_watch.wrong += cursor->depth != ruled || count > cursor->depth ? 1u : 0u;
    depth_watch.falls += cursor->depth < depth_watch.depth ? 1u : 0u;
    depth_watch.deepest = cursor->depth > depth_watch.deepest ? cursor->depth : depth_watch.deepest;
    depth_watch.depth = cursor->depth;
    depth_watch.rounds = cursor->rounds;
    depth_watch.proposals = cursor->proposals;
    depth_watch.accepted = cursor->accepted;
    depth_watch.calls++;
    return count;
}

/* The lookup drafter's depth (engine.c) on the copy model of a prompt whose 4-gram 1 2 3 4 is followed by 5 and by 6
 * in turn, so that its proposal behind that 4-gram is wrong and the others right: every round's depth as the rule
 * gives it, its proposals within it, the depth up to the draft positions and down again; the events those of steps
 * of one drawing row. */
static void test_lookup_depth(void)
{
	static struct generation_event reference[MOST_EVENTS],
		events[MOST_EVENTS];
	struct generation_command_record command =
		function_command(0, 0, DECOY_PROMPT, DEPTH_LIMIT);
	uint32_t count = 0;
	command.new_token_ids = decoy_prompt;
	context_text = "speculation: the lookup drafter's depth";
	test_serving_set_function_model(true, DECOY_PERIOD);
	test_serving_set_engine(1, 1, SPECULATIVE_ROWS, 0);
	const uint32_t reference_count = command_events(
		SPECULATIVE_CONTEXT, &command, NULL, reference, NULL);
	test_serving_set_engine(1, 1, SPECULATIVE_ROWS, DRAFT_POSITIONS);
	struct inference_engine *const engine =
		drafting_engine(SPECULATIVE_CONTEXT, NULL);
	memset(&depth_watch, 0, sizeof(depth_watch));
	if (engine != NULL) {
		depth_watch.lookup = engine->draft;
		engine->draft = depth_watched_draft;
		expect(engine_submit(engine, &command) == STATUS_OK,
		       "submitted");
		count = events_until_final(engine, events);
	}
    test_serving_close();
    expect(events_match(events, count, reference, reference_count), "the events of steps of one drawing row");
    expect(depth_watch.calls != 0 && depth_watch.wrong == 0 && depth_watch.deepest == DRAFT_POSITIONS &&
           depth_watch.falls != 0, "each round's depth by the rule, up to the draft positions and down again");
}

/* ---- the window ring across commands (serving_cursor.h) -------------------------------------------------------- */

/* A drafter whose proposals are always wrong: as many as it may, each a token the function model never draws. */
static uint32_t never_drawn_draft(struct inference_engine *engine,
				  const struct command_cursor *cursor,
				  uint32_t anchor_position, uint32_t most,
				  uint32_t *proposals_out)
{
    most = cursor->depth < most ? cursor->depth : most;
    for (uint32_t index = 0; index < most; index++) {
        proposals_out[index] = NEVER_DRAWN;
    }
    drafted.last_count = most;
    (void)engine;
    (void)anchor_position;
    return most;
}

/* A persistently wrong drafter is probed once per cooldown plus one ordinary draw, at depth one. The copy model
 * never ends early, so both the complete event bytes and the extra target rows have a deterministic bound. */
static void test_draft_cooldown(void)
{
	static struct generation_event reference[MOST_EVENTS],
		events[MOST_EVENTS];
	context_text = "speculation: a persistently wrong drafter";
	test_serving_set_function_model(true, COPY_PERIOD);
	const struct generation_command_record command =
		function_command(0, 0, COPY_PROMPT, SPECULATIVE_LIMIT);
	for (uint32_t pipelined = 0; pipelined < TWO; pipelined++) {
		const uint32_t depth = pipelined != 0 ? PIPELINE_DEPTH : 1u;
		test_serving_set_engine(1, depth, SPECULATIVE_ROWS, 0);
		struct inference_engine *engine =
			drafting_engine(SPECULATIVE_CONTEXT, NULL);
		uint32_t reference_count = 0, count = 0;
		uint64_t reference_rows = 0, rows = 0;
		if (engine != NULL) {
			expect(engine_submit(engine, &command) == STATUS_OK,
			       "the reference submitted");
			reference_count = events_until_final(engine, reference);
			reference_rows = test_serving_rows_seen();
		}
		test_serving_close();
		test_serving_set_engine(1, depth, SPECULATIVE_ROWS,
					DRAFT_POSITIONS);
		engine =
			drafting_engine(SPECULATIVE_CONTEXT, never_drawn_draft);
		if (engine != NULL) {
			expect(engine_submit(engine, &command) == STATUS_OK,
			       "the wrong drafter submitted");
			count = events_until_final(engine, events);
			rows = test_serving_rows_seen();
		}
		test_serving_close();
		const uint32_t probe_bound =
			(SPECULATIVE_LIMIT + SERVING_DRAFT_COOLDOWN) /
			(SERVING_DRAFT_COOLDOWN + 1u);
		expect(reference_count == SPECULATIVE_LIMIT + 1u &&
			       events_match(events, count, reference,
					    reference_count),
		       "every event equals ordinary decoding");
		expect(rows > reference_rows &&
			       rows <= reference_rows + probe_bound,
		       "at most one extra row per sixteen draws, including the first probe");
	}
}

MUST_CHECK static result_status kept_drawn(struct inference_engine *engine,
					   uint32_t kept, uint32_t *drawn_out);

/* The 3,968-row bound itself, and one row past it: both reach parities, with no proposals and with one rejected
 * proposal beyond an immediate end token; serial and pipelined. Each kept prefix is a whole granule. */
static void test_ring_exact_boundary(void)
{
	static struct generation_event events[MOST_EVENTS];
	static const uint32_t end_script[] = { END };
	static const uint32_t *const end_scripts[] = { end_script };
	context_text = "the exact window-ring reuse boundary";
	test_serving_set_function_model(false, 0);
	test_serving_set_ring(MODEL_RING_SLOTS, MODEL_WINDOW);
	for (uint32_t pipelined = 0; pipelined < TWO; pipelined++) {
		for (uint32_t rejects = 0; rejects < TWO; rejects++) {
			for (uint32_t past = 0; past < TWO; past++) {
				struct inference_engine *engine = NULL;
				struct session_state saved = { 0 };
				const uint32_t prompt_count =
					BOUNDARY_PROMPT + TWO +
					(rejects != 0 ? 1u - past : past);
				test_serving_set_engine(
					1, pipelined != 0 ? PIPELINE_DEPTH : 1u,
					SPECULATIVE_ROWS,
					rejects != 0 ? DRAFT_POSITIONS : 0);
				expect(test_serving_open(RING_CONTEXT,
							 end_scripts, 1,
							 &engine) == STATUS_OK,
				       "the immediate-end engine opens");
				if (engine == NULL) {
					continue;
				}
				engine->draft = never_drawn_draft;
				struct generation_command_record command =
					function_command(0, 0, prompt_count,
							 SPECULATIVE_LIMIT);
				command.new_token_ids = ring_prompt;
				expect(engine_submit(engine, &command) ==
						       STATUS_OK &&
					       events_until_final(
						       engine, events) == TWO &&
					       engine_session_state(engine, 0,
								    &saved) ==
						       STATUS_OK,
				       "the command runs and its true reach is available");
				expect(saved.reach_count ==
					       prompt_count + rejects,
				       "rejected proposals count in the reach");
				const uint32_t kept =
					saved.reach_count - KEPT_LAG - past;
				uint32_t drawn = NEVER_DRAWN;
				expect(kept % SERVING_KEPT_GRANULE == 0 &&
					       kept_drawn(engine, kept,
							  &drawn) ==
						       (past != 0 ?
								STATUS_INVALID_ARGUMENT :
								STATUS_OK),
				       past != 0 ?
					       "a prefix one row past the lag is refused" :
					       "the exact lag is admitted");
				test_serving_close();
			}
		}
    }
    test_serving_set_ring(TEST_SERVING_STATE_POSITIONS, TEST_SERVING_WINDOW);
}

/* The first token an engine without speculation draws when it reads these tokens anew, on the ring as last set. */
static uint32_t fresh_drawn(const uint32_t *token_ids, uint32_t token_count)
{
	static struct generation_event events[MOST_EVENTS];
	struct generation_command_record command =
		function_command(0, 0, token_count, 1);
	command.new_token_ids = token_ids;
	test_serving_set_engine(1, 1, SPECULATIVE_ROWS, 0);
	return command_events(RING_CONTEXT, &command, NULL, events, NULL) != 0 ?
		       events[0].token_id :
		       NEVER_DRAWN;
}

/* A command that keeps the first kept tokens of slot 0's session and reads RING_TOKEN behind them: its submit's status,
 * and in *drawn_out the token it draws (NEVER_DRAWN if it was refused). */
MUST_CHECK static result_status kept_drawn(struct inference_engine *engine,
					   uint32_t kept, uint32_t *drawn_out)
{
	static struct generation_event events[MOST_EVENTS];
	static const uint32_t new_token[] = { RING_TOKEN };
	struct generation_command_record command = function_command(0, 0, 1, 1);
	command.new_token_ids = new_token;
	command.kept_token_count = kept;
	const result_status status = engine != NULL ?
					     engine_submit(engine, &command) :
					     STATUS_INVALID_ARGUMENT;
	*drawn_out = status == STATUS_OK &&
				     events_until_final(engine, events) != 0 ?
			     events[0].token_id :
			     NEVER_DRAWN;
	return status;
}

/* A shorter command behind a longer one, on a window ring of the model's size (4,096 slots, a draw reading 128
 * positions) and without speculation: the slot ran LINEAGE_PROMPT tokens, then a command kept LINEAGE_KEPT of them
 * (3,960 behind) and ran LINEAGE_SAVED. A prefix of STALE_KEPT, within the lag of what the slot ran last but not of its
 * reach, whose window the first command's later rows overwrote, is refused (until 26 October 2026 it was kept, and drew
 * another token than a fresh run of its tokens); the prefix the shorter command kept draws as a fresh run. The same in
 * the engine of a restart, after a load of the session with the saving slot's reach. */
static void test_ring_lineage(void)
{
	static struct generation_event events[MOST_EVENTS];
	char path[] = "/tmp/anemone_engine_ring_XXXXXX";
	const int descriptor = mkstemp(path);
	uint32_t drawn = NEVER_DRAWN;
	context_text = "the window ring behind a shorter command";
	expect(descriptor >= 0 && unlink(path) == 0, "a file for the snapshot");
	if (descriptor < 0) {
		return;
	}
    test_serving_set_function_model(true, 0);
    test_serving_set_ring(MODEL_RING_SLOTS, MODEL_WINDOW);
    memcpy(ring_tokens, ring_prompt, LINEAGE_KEPT * sizeof(uint32_t));
    ring_tokens[LINEAGE_KEPT] = RING_TOKEN;
    const uint32_t fresh = fresh_drawn(ring_tokens, LINEAGE_KEPT + 1u);
    for (uint32_t index = 0; index < LINEAGE_NEW; index++) {     /* the shorter command's new tokens: not the first's */
        ring_tokens[LINEAGE_KEPT + index] = (ring_prompt[LINEAGE_KEPT + index] + 1u) % BYTE_VALUES;
    }
    test_serving_set_engine(1, 1, SPECULATIVE_ROWS, 0);
    struct inference_engine *engine = drafting_engine(RING_CONTEXT, NULL);
    struct generation_command_record command =
	    function_command(0, 0, LINEAGE_PROMPT, 1);
    command.new_token_ids = ring_prompt;
    expect(engine != NULL && engine_submit(engine, &command) == STATUS_OK &&
		   events_until_final(engine, events) != 0,
	   "the long command");
    command = function_command(0, 0, LINEAGE_NEW, 1);
    command.new_token_ids = ring_tokens + LINEAGE_KEPT;
    command.kept_token_count = LINEAGE_KEPT;
    expect(engine != NULL && engine_submit(engine, &command) == STATUS_OK &&
		   events_until_final(engine, events) != 0,
	   "a shorter command keeps most of it");
    expect(kept_drawn(engine, STALE_KEPT, &drawn) == STATUS_INVALID_ARGUMENT,
	   "a prefix within the lag of what the slot ran last but not of its reach is refused");
    const struct session_command save = {
	    NULL, descriptor, 0, LINEAGE_SAVED, SESSION_SAVE, SESSION_TAG,
	    0,	  0,	      0
    }; /* a save names no reach */
    expect(engine != NULL && session_done(engine, &save).event_kind ==
				     GENERATION_EVENT_FINISHED_AT_LIMIT,
	   "the session saved");
    expect(kept_drawn(engine, LINEAGE_KEPT, &drawn) == STATUS_OK &&
		   drawn == fresh,
	   "the prefix the shorter command kept draws as a fresh run of its tokens");
    test_serving_close();
    engine = drafting_engine(RING_CONTEXT, NULL);
    const struct session_command load = { ring_tokens,
					  descriptor,
					  0,
					  LINEAGE_SAVED,
					  SESSION_LOAD,
					  SESSION_TAG,
					  LINEAGE_PROMPT,
					  0,
					  0 }; /* the saving slot's reach */
    expect(engine != NULL && lseek(descriptor, 0, SEEK_SET) == 0 &&
           session_done(engine, &load).session_token_count == LINEAGE_SAVED, "loaded by the engine of a restart");
    expect(kept_drawn(engine, STALE_KEPT, &drawn) == STATUS_INVALID_ARGUMENT,
	   "after the load, the same prefix is refused");
    expect(kept_drawn(engine, LINEAGE_KEPT, &drawn) == STATUS_OK &&
		   drawn == fresh,
	   "after the load, the kept prefix draws as a fresh run of its tokens");
    test_serving_close();
    (void)close(descriptor);
    test_serving_set_ring(TEST_SERVING_STATE_POSITIONS, TEST_SERVING_WINDOW);
}

/* Rows a round rejected, on a window ring of the model's size: a command whose proposals are all wrong ends at the end
 * token, and its reach includes every rejected row. A command that keeps the
 * oldest prefix the lag allows behind that draws as a fresh run of its tokens, and one a granule older is refused;
 * behind that command, which ran far less, the same again (the lag counts from the slot's reach). */
static void test_ring_rejected(void)
{
	static struct generation_event events[MOST_EVENTS];
	uint32_t count = 0, refused = NEVER_DRAWN,
		 drawn[TWO] = { NEVER_DRAWN, NEVER_DRAWN };
	struct session_state saved = { 0 };
	context_text = "the window ring behind rejected rows";
	test_serving_set_function_model(true, 0);
	test_serving_set_ring(MODEL_RING_SLOTS, MODEL_WINDOW);
	test_serving_set_engine(1, 1, SPECULATIVE_ROWS, DRAFT_POSITIONS);
	struct inference_engine *const engine =
		drafting_engine(RING_CONTEXT, never_drawn_draft);
	struct generation_command_record command =
		function_command(0, 0, BOUNDARY_PROMPT, BOUNDARY_LIMIT);
	command.new_token_ids = ring_prompt;
	if (engine != NULL && engine_submit(engine, &command) == STATUS_OK) {
		count = events_until_final(engine, events);
	}
	const struct generation_event final =
		events[count != 0 ? count - 1u : 0];
	expect(engine != NULL &&
		       engine_session_state(engine, 0, &saved) == STATUS_OK,
	       "the actual reach, including rejected rows");
	const uint32_t ran = saved.reach_count;
	const bool is_whole = final.event_kind ==
				      GENERATION_EVENT_FINISHED_AT_END_TOKEN &&
			      drafted.last_count != 0 &&
			      ran > SERVING_KEPT_LAG + SERVING_KEPT_GRANULE &&
			      ran - SERVING_KEPT_LAG < BOUNDARY_PROMPT;
	const uint32_t oldest =
		is_whole ?
			(ran - SERVING_KEPT_LAG + SERVING_KEPT_GRANULE - 1u) /
				SERVING_KEPT_GRANULE * SERVING_KEPT_GRANULE :
			SERVING_KEPT_GRANULE;
	expect(is_whole,
	       "the end token drawn after rounds with rejected proposals");
	for (uint32_t again = 0; again < TWO; again++) {
		expect(kept_drawn(engine, oldest - SERVING_KEPT_GRANULE,
				  &refused) == STATUS_INVALID_ARGUMENT,
		       again == 0 ?
			       "a prefix a granule older than the lag allows is refused" :
			       "behind a command that ran far less, the same prefix is refused");
		expect(kept_drawn(engine, oldest, &drawn[again]) == STATUS_OK,
		       "the oldest prefix the lag allows");
	}
    test_serving_close();
    memcpy(ring_tokens, ring_prompt, oldest * sizeof(uint32_t));
    ring_tokens[oldest] = RING_TOKEN;
    expect(drawn[0] == fresh_drawn(ring_tokens, oldest + 1u) && drawn[1] == drawn[0],
           "it draws as a fresh run of its tokens, and again behind the command that kept it");
    test_serving_set_ring(TEST_SERVING_STATE_POSITIONS, TEST_SERVING_WINDOW);
}

/* A card visited twice a step (serving_fixture.h's interleaved programs) in a pipeline: a long prompt's steps fly
 * several at once, each visit of card 0 reading its own step's rows (a visit that read another's faults), and the
 * script's tokens come; a cancel once a token came ends the command at a step boundary with its tokens held, and the
 * next command runs; a fault fails its command and every command behind it. */
static void test_interleaved(void)
{
	static struct generation_event events[MOST_EVENTS];
	struct inference_engine *engine = NULL;
	context_text = "a card visited twice a step, in a pipeline";
	test_serving_set_interleaved(INTERLEAVED_DEPTH);
	expect(test_serving_open(CONTEXT, scripts,
				 sizeof(scripts) / sizeof(scripts[0]),
				 &engine) == STATUS_OK,
	       "the engine opens on interleaved programs");
	if (engine == NULL) {
		test_serving_set_interleaved(0);
		return;
	}
	struct generation_command_record command =
		command_of(INTERLEAVED_PROMPT, ENDLESS_TOKENS, 0);
	const uint64_t steps_before = test_serving_step_count();
	expect(engine_submit(engine, &command) == STATUS_OK, "submitted");
	uint32_t count = events_until_final(engine, events);
	expect(count == EVENTS_OF_THE_SCRIPT &&
		       events[FIRST_EVENT].token_id == FIRST_TOKEN &&
		       events[SECOND_EVENT].token_id == SECOND_TOKEN &&
		       events[THIRD_EVENT].token_id == END &&
		       events[FIRST_EVENT].logit ==
			       (float)(INTERLEAVED_PROMPT - 1u) &&
		       events[FOURTH_EVENT].event_kind ==
			       GENERATION_EVENT_FINISHED_AT_END_TOKEN &&
		       test_serving_step_count() - steps_before ==
			       INTERLEAVED_PROMPT_STEPS + THIRD_EVENT,
	       "the prompt's steps in flight together, then the script's tokens one a step");

	context_text = "a cancel in an interleaved pipeline";
	command =
		command_of(INTERLEAVED_PROMPT, ENDLESS_TOKENS, SCRIPT_ENDLESS);
	test_serving_set_step_delay(STEP_DELAY_NANOSECONDS);
	const uint64_t cancel_before = test_serving_step_count();
	expect(engine_submit(engine, &command) == STATUS_OK, "submitted");
	const uint64_t token_drawn =
		cancel_before + INTERLEAVED_PROMPT_STEPS + 1u;
	for (uint32_t look = 0;
	     look < MOST_LOOKS && test_serving_step_count() < token_drawn;
	     look++) {
		pause_for(
			STEP_DELAY_NANOSECONDS); /* until the prompt is read and a token drawn */
	}
	expect(engine_cancel(engine, 0) == STATUS_OK,
	       "cancelled once a token came");
	count = events_until_final(engine, events);
	const struct generation_event final =
		events[count != 0 ? count - 1u : 0];
	expect(count >= 1u && count < ENDLESS_TOKENS &&
		       final.event_kind == GENERATION_EVENT_CANCELLED &&
		       final.session_token_count ==
			       INTERLEAVED_PROMPT + count - 1u,
	       "it ends at a step boundary, its tokens held");
	test_serving_set_step_delay(0);
	command = command_of(PROMPT_TOKENS, TWO, 0);
	expect(engine_submit(engine, &command) == STATUS_OK,
	       "the next command is taken");
	count = events_until_final(engine, events);
	expect(count == THIRD_EVENT + 1u &&
		       events[FIRST_EVENT].token_id == FIRST_TOKEN &&
		       events[THIRD_EVENT].event_kind ==
			       GENERATION_EVENT_FINISHED_AT_LIMIT,
	       "and runs as before");

	context_text = "a fault in an interleaved pipeline";
	test_serving_set_fault_position(FAULTING_POSITION);
	command = command_of(INTERLEAVED_PROMPT, 1, 0);
	expect(engine_submit(engine, &command) == STATUS_OK, "submitted");
	count = events_until_final(engine, events);
	expect(count == 1u && events[0].event_kind == GENERATION_EVENT_FAILED &&
		       events[0].status == STATUS_DEVICE_ERROR,
	       "the command fails with the card");
	test_serving_set_fault_position(UINT32_MAX);
	command = command_of(PROMPT_TOKENS, 1, 0);
	const result_status submitted = engine_submit(engine, &command);
	count = submitted == STATUS_OK ? events_until_final(engine, events) : 0;
	expect(submitted != STATUS_OK ||
		       (count == 1u &&
			events[0].event_kind == GENERATION_EVENT_FAILED),
	       "and so does every command behind it");
	test_serving_close();
	test_serving_set_interleaved(0);
}

/* The actual device round path (no host drafter replacement): every event agrees for greedy and seeded draws,
 * rejection at each of five positions, eager and pipelined execution, and a card visited twice. */
static void test_device_rounds(bool fixed_depth, uint32_t block)
{
	static struct generation_event reference[MOST_EVENTS],
		events[MOST_EVENTS];
	static const uint32_t depths[] = { 1, PIPELINE_DEPTH };
	uint32_t rejections = 0;
	test_serving_set_device_block(block);
	test_serving_set_function_model(true, 0);
	for (uint32_t seed_case = 0; seed_case < SEED_CASES; seed_case++) {
		const struct generation_command_record command =
			function_command(0, seed_case, SPECULATIVE_PROMPT,
					 SPECULATIVE_LIMIT);
		context_text = "device draft: reference";
		test_serving_set_device_draft(false, block);
		test_serving_set_engine(1, 1, SPECULATIVE_ROWS, 0);
		const uint32_t reference_count = command_events(
			SPECULATIVE_CONTEXT, &command, NULL, reference, NULL);
		for (uint32_t depth = 0;
		     depth < sizeof(depths) / sizeof(depths[0]); depth++) {
			for (uint32_t rejected = 0; rejected <= block;
			     rejected++) {
				context_text = "device draft: keyed rounds";
				test_serving_set_engine(1, depths[depth],
							SPECULATIVE_ROWS,
							block);
				test_serving_set_device_draft(true, rejected);
				const uint32_t count = command_events(
					SPECULATIVE_CONTEXT, &command, NULL,
					events, NULL);
				expect(events_match(events, count, reference,
						    reference_count),
				       "every event equals plain decoding");
				expect(test_serving_device_rounds() != 0,
				       "the prompt's last step returned device proposals");
				rejections |= test_serving_device_rejections();
			}
		}
		test_serving_set_interleaved(PIPELINE_DEPTH);
		test_serving_set_device_draft(true, block);
		test_serving_set_engine(1, PIPELINE_DEPTH, SPECULATIVE_ROWS,
					block);
		const uint32_t interleaved_count = command_events(
			SPECULATIVE_CONTEXT, &command, NULL, events, NULL);
		expect(events_match(events, interleaved_count, reference,
				    reference_count),
		       "device rounds preserve events when a card is visited twice");
		test_serving_set_interleaved(0);
	}
	struct draft_policy_record policy;
	draft_policy_default(&policy);
	float scores[STEP_DRAFT_MOST] = { 0 };
	for (uint32_t position = 0; position < block; position++) {
		scores[position] = 1.0f;
	}
	const uint32_t expected_depth =
		draft_depth(&policy, scores, block, false);
	expect(rejections == (fixed_depth ? ((1u << block) - 1u) :
					    (1u << expected_depth) - 1u),
	       "device proposals rejected at every verified position; fixed policy covers all five");
	test_serving_set_device_draft(false, block);
	test_serving_set_device_block(MODEL_DRAFT_BLOCK);
}

static void test_policy_event_identity(uint32_t mode)
{
	static struct generation_event reference[MOST_EVENTS],
		events[MOST_EVENTS];
	struct draft_policy_record policy;
	draft_policy_default(&policy);
	policy.depth_policy = mode;
	test_serving_set_function_model(true, 0);
	for (uint32_t seed_case = 0; seed_case < SEED_CASES; seed_case++) {
		const struct generation_command_record command =
			function_command(0, seed_case, SPECULATIVE_PROMPT,
					 SPECULATIVE_LIMIT);
		test_serving_set_device_draft(false, MODEL_DRAFT_BLOCK);
		test_serving_set_engine(1, 1, SPECULATIVE_ROWS, 0);
		const uint32_t reference_count = command_events(
			SPECULATIVE_CONTEXT, &command, NULL, reference, NULL);
		test_serving_set_draft_policy(&policy, 0);
		for (uint32_t block = MODEL_DRAFT_BLOCK;
		     block <= STEP_DRAFT_MOST; block++) {
			test_serving_set_device_block(block);
			test_serving_set_device_draft(true, block);
			test_serving_set_engine(1, PIPELINE_DEPTH,
						SPECULATIVE_ROWS, block);
			const uint32_t count =
				command_events(SPECULATIVE_CONTEXT, &command,
					       NULL, events, NULL);
			expect(events_match(events, count, reference,
					    reference_count),
			       "rate and adaptive policies retain every greedy and seeded target draw");
		}
		test_serving_set_draft_policy(NULL, 0);
	}
	test_serving_set_device_block(MODEL_DRAFT_BLOCK);
	test_serving_set_device_draft(false, MODEL_DRAFT_BLOCK);
}

/* Limits leave the final correction inside the command and context; draft writes are suppressed when five further
 * positions cannot fit. Four independent owners also retain the same complete event streams. */
static void test_device_limits_and_sessions(void)
{
	static struct generation_event reference[SESSIONS][MOST_EVENTS],
		events[SESSIONS][MOST_EVENTS];
	uint32_t reference_counts[SESSIONS], counts[SESSIONS];
	struct generation_command_record commands[SESSIONS];
	test_serving_set_function_model(true, 0);
	for (uint32_t limit = 1; limit <= LIMIT_CASES; limit++) {
		const struct generation_command_record command =
			function_command(0, limit % SEED_CASES,
					 SHORT_SPECULATIVE_PROMPT, limit);
		context_text = "device draft: command and context limits";
		test_serving_set_device_draft(false, MODEL_DRAFT_BLOCK);
		test_serving_set_engine(1, 1, SPECULATIVE_ROWS, 0);
		const uint32_t reference_count =
			command_events(SHORT_SPECULATIVE_CONTEXT, &command,
				       NULL, reference[0], NULL);
		test_serving_set_device_draft(true, MODEL_DRAFT_BLOCK);
		test_serving_set_engine(1, PIPELINE_DEPTH, SPECULATIVE_ROWS,
					MODEL_DRAFT_BLOCK);
		const uint32_t count = command_events(SHORT_SPECULATIVE_CONTEXT,
						      &command, NULL, events[0],
						      NULL);
		expect(events_match(events[0], count, reference[0],
				    reference_count),
		       "limits preserve every token and count");
	}
    for (uint32_t ending = 0; ending < TWO; ending++) {
        const uint32_t context = ending != 0 ? SHORT_SPECULATIVE_CONTEXT : SPECULATIVE_CONTEXT;
        for (uint32_t slot = 0; slot < SESSIONS; slot++) {
            commands[slot] = function_command(slot, slot % SEED_CASES,
                                              ending != 0 ? SHORT_SPECULATIVE_PROMPT + slot : SPECULATIVE_PROMPT,
                                              SESSION_LIMIT);
        }
        context_text = "device draft: four sessions";
	test_serving_set_device_draft(false, MODEL_DRAFT_BLOCK);
	test_serving_set_engine(SESSIONS, PIPELINE_DEPTH, SPECULATIVE_ROWS, 0);
        (void)sessions_run(context, commands, NULL, reference, reference_counts);
        test_serving_set_device_draft(true, DEVICE_CONFIDENCE_POSITION);
	test_serving_set_engine(SESSIONS, PIPELINE_DEPTH, SPECULATIVE_ROWS,
				MODEL_DRAFT_BLOCK);
	(void)sessions_run(context, commands, NULL, events, counts);
        for (uint32_t slot = 0; slot < SESSIONS; slot++) {
            expect(events_match(events[slot], counts[slot], reference[slot], reference_counts[slot]),
                   "each device-drafting session preserves its entire event stream");
        }
    }
    test_serving_set_device_draft(false, MODEL_DRAFT_BLOCK);
}

/* The copy model has no accidental end token. Put it at every token of a full-depth round, including its bonus draw,
 * so accepted-end and bonus-end paths must both preserve the reference's complete events. */
static void test_device_end_positions(void)
{
	static struct generation_event reference[MOST_EVENTS],
		events[MOST_EVENTS];
	struct generation_command_record command =
		function_command(0, 0, COPY_PROMPT, COPY_LIMIT);
	command.new_token_ids = copy_prompt;
	test_serving_set_function_model(true, COPY_PERIOD);
	for (uint32_t index = 0; index <= MODEL_DRAFT_BLOCK; index++) {
		context_text = "device draft: end token within a round";
		test_serving_set_function_end(DEVICE_FULL_DEPTH_ANCHOR + index);
		test_serving_set_device_draft(false, MODEL_DRAFT_BLOCK);
		test_serving_set_engine(1, 1, SPECULATIVE_ROWS, 0);
		const uint32_t reference_count = command_events(
			SPECULATIVE_CONTEXT, &command, NULL, reference, NULL);
		test_serving_set_device_draft(true, MODEL_DRAFT_BLOCK);
		test_serving_set_engine(1, PIPELINE_DEPTH, SPECULATIVE_ROWS,
					MODEL_DRAFT_BLOCK);
		const uint32_t count = command_events(
			SPECULATIVE_CONTEXT, &command, NULL, events, NULL);
		expect(events_match(events, count, reference,
				    reference_count) &&
			       count != 0 &&
			       events[count - 1u].event_kind ==
				       GENERATION_EVENT_FINISHED_AT_END_TOKEN,
		       "end token at every round position preserves the complete event stream");
	}
    test_serving_set_function_end(UINT32_MAX);
    test_serving_set_device_draft(false, MODEL_DRAFT_BLOCK);
}

/* Confidence limits only proposal work; actual draft positions contribute to reach. A structurally valid wrong
 * device acceptance count fails the flight before any of its tokens are queued. */
static void test_device_policy_and_refusals(void)
{
	static struct generation_event reference[MOST_EVENTS],
		events[MOST_EVENTS];
	struct generation_command_record command =
		function_command(0, 0, COPY_PROMPT, COPY_LIMIT);
	command.new_token_ids = copy_prompt;
	test_serving_set_function_model(true, COPY_PERIOD);
	test_serving_set_engine(1, 1, SPECULATIVE_ROWS, 0);
	const uint32_t reference_count = command_events(
		SPECULATIVE_CONTEXT, &command, NULL, reference, NULL);
	test_serving_set_device_draft(true, MODEL_DRAFT_BLOCK);
	test_serving_set_device_confidence(DEVICE_CONFIDENCE_POSITION);
	test_serving_set_engine(1, 1, SPECULATIVE_ROWS, MODEL_DRAFT_BLOCK);
	context_text = "device draft: confidence and reach";
	struct inference_engine *engine =
		drafting_engine(SPECULATIVE_CONTEXT, NULL);
	if (engine != NULL) {
		expect(engine_submit(engine, &command) == STATUS_OK,
		       "confidence command submitted");
		const uint32_t count = events_until_final(engine, events);
		const struct command_cursor *const cursor = &engine->cursors[0];
		struct session_state state;
		expect(events_match(events, count, reference, reference_count),
		       "confidence changes no emitted event");
		const float scores[MODEL_DRAFT_BLOCK] = { 1.0f, 1.0f, -1.0f,
							  1.0f, 1.0f };
		const uint32_t selected = draft_depth(&engine->draft_policy,
						      scores, MODEL_DRAFT_BLOCK,
						      false);
		expect(selected < MODEL_DRAFT_BLOCK && cursor->rounds > 1u &&
			       cursor->proposals <= cursor->rounds * selected,
		       "confidence-derived expected cost bounds verified depth");
		expect(engine_session_state(engine, 0, &state) == STATUS_OK &&
			       state.reach_count >
				       engine->slots[0].processed_count,
		       "all five persistent draft positions contribute to the slot's actual reach");
	}
    test_serving_close();
    context_text = "device draft: host acceptance authority";
    test_serving_set_device_draft(true, 0);
    test_serving_set_device_round_fault(true);
    engine = drafting_engine(SPECULATIVE_CONTEXT, NULL);
    if (engine != NULL) {
	    expect(engine_submit(engine, &command) == STATUS_OK,
		   "malformed-round command submitted");
	    const uint32_t count = events_until_final(engine, events);
	    expect(count == TWO &&
			   events[0].event_kind == GENERATION_EVENT_TOKEN &&
			   events[1].event_kind == GENERATION_EVENT_FAILED &&
			   events[1].status == STATUS_DEVICE_ERROR,
		   "wrong keyed-match count fails before any token of that flight is published");
    }
    test_serving_close();
    context_text = "device draft: admission";
    test_serving_set_device_draft(true, MODEL_DRAFT_BLOCK);
    test_serving_set_engine(1, 1, SPECULATIVE_ROWS, DRAFT_POSITIONS);
    expect(test_serving_open(SPECULATIVE_CONTEXT, scripts,
			     sizeof(scripts) / sizeof(scripts[0]),
			     &engine) == STATUS_INVALID_ARGUMENT,
	   "depth above the trained block is refused");
    test_serving_set_engine(SESSIONS, 1, SPECULATIVE_ROWS, MODEL_DRAFT_BLOCK);
    expect(test_serving_open(SPECULATIVE_CONTEXT, scripts,
			     sizeof(scripts) / sizeof(scripts[0]),
			     &engine) == STATUS_INVALID_ARGUMENT,
	   "several device-drafting sessions need a pipeline");
    test_serving_set_device_draft(false, MODEL_DRAFT_BLOCK);
}

/* Policies, hybrid switching and release statistics run through real flights and the keyed fixture drafter.
 * The unique prompt first needs DSpark; its periodic continuation then supplies lookup matches. */
static void test_policy_statistics(void)
{
	static struct generation_event reference[MOST_EVENTS],
		events[MOST_EVENTS];
	struct draft_policy_record policy;
	struct draft_statistics statistics = { 0 }, total = { 0 },
				previous = { 0 };
	test_serving_set_function_model(true, COPY_PERIOD);
	for (uint32_t seed_case = 0; seed_case < SEED_CASES; seed_case++) {
		const struct generation_command_record command =
			function_command(0, seed_case, SPECULATIVE_PROMPT,
					 COPY_LIMIT);
		test_serving_set_draft_policy(NULL, 0);
		test_serving_set_device_draft(false, MODEL_DRAFT_BLOCK);
		test_serving_set_engine(1, 1, SPECULATIVE_ROWS, 0);
		const uint32_t reference_count = command_events(
			SPECULATIVE_CONTEXT, &command, NULL, reference, NULL);
		for (uint32_t policy_kind = DRAFT_DEPTH_FIXED;
		     policy_kind <= DRAFT_DEPTH_COST + TWO; policy_kind++) {
			for (uint32_t hybrid = 0; hybrid < TWO; hybrid++) {
				context_text =
					"draft policy, hybrid switching and statistics";
				draft_policy_default(&policy);
				policy.depth_policy =
					policy_kind == DRAFT_DEPTH_FIXED ?
						policy_kind :
						DRAFT_DEPTH_COST;
				const uint32_t threshold =
					policy_kind == DRAFT_DEPTH_COST + 1u ?
						DRAFT_PROBABILITY_SCALE :
						DRAFT_DEFAULT_THRESHOLD;
				policy.threshold = threshold;
				const bool low_estimate =
					policy_kind == DRAFT_DEPTH_COST + TWO;
				if (low_estimate) {
					memset(policy.acceptance, 0,
					       sizeof(policy.acceptance));
				}
				test_serving_set_draft_policy(
					&policy,
					hybrid != 0 ? DRAFT_DEFAULT_LOOKUP_MIN :
						      0);
				test_serving_set_device_draft(
					true, MODEL_DRAFT_BLOCK);
				test_serving_set_engine(1, PIPELINE_DEPTH,
							SPECULATIVE_ROWS,
							MODEL_DRAFT_BLOCK);
				struct inference_engine *const engine =
					drafting_engine(SPECULATIVE_CONTEXT,
							NULL);
				if (engine == NULL) {
					continue;
				}
				policy.threshold = 0;
				expect(engine->draft_policy.threshold ==
					       threshold,
				       "engine pins its own calibration policy at creation");
				for (uint32_t repeat = 0; repeat < TWO;
				     repeat++) {
					expect(engine_submit(engine,
							     &command) ==
						       STATUS_OK,
					       "policy command submitted");
					const uint32_t count =
						events_until_final(engine,
								   events);
					expect(events_match(events, count,
							    reference,
							    reference_count),
					       "every policy preserves events");
					expect(engine_draft_statistics(
						       engine, 0, &statistics,
						       &total) == STATUS_OK,
					       "release statistics available after final");
					struct engine_stats live = { 0 };
					expect(engine_stats_snapshot(engine,
								     &live) ==
							       STATUS_OK &&
						       memcmp(&live.speculation,
							      &total,
							      sizeof(total)) ==
							       0 &&
						       memcmp(&live.slots[0]
								       .speculation,
							      &statistics,
							      sizeof(statistics)) ==
							       0 &&
						       live.slots[0].speculation_mode ==
							       (hybrid != 0 ?
									STATS_SPECULATION_HYBRID :
									STATS_SPECULATION_DSPARK) &&
						       live.slots[0].probes !=
							       0 &&
						       live.slots[0].dspark_tails !=
							       0 &&
						       live.slots[0]
							       .has_confidence,
					       "live speculation counters exactly equal all inherited source/position/bin counters");
					if (repeat == 0) {
						expect(memcmp(&statistics,
							      &total,
							      sizeof(total)) ==
							       0,
						       "first total equals command");
					} else {
						draft_statistics_add(
							&previous, &statistics);
						expect(memcmp(&previous, &total,
							      sizeof(total)) ==
							       0,
						       "total adds completed commands once");
					}
					previous = total;
					uint64_t proposed = 0, accepted = 0,
						 rounds = 0, produced = 0;
					for (uint32_t source = 0;
					     source < DRAFT_SOURCES; source++) {
						rounds +=
							statistics
								.source[source]
								.rounds;
						for (uint32_t position = 0;
						     position <
						     DRAFT_STAT_POSITIONS;
						     position++) {
							const struct draft_source_statistics *const value =
								&statistics.source
									 [source];
							proposed +=
								value->proposed
									[position];
							accepted +=
								value->accepted
									[position];
							expect(value->accepted[position] <=
									       value->reached
										       [position] &&
								       value->reached[position] <=
									       value->proposed
										       [position],
							       "prefix counter bounds");
						}
					}
					const struct command_cursor *const
						cursor = &engine->cursors[0];
					expect(rounds == cursor->rounds &&
						       proposed ==
							       cursor->proposals &&
						       accepted ==
							       cursor->accepted,
					       "release totals agree with flight acceptance");
					expect(statistics.drawing_rounds +
							       accepted ==
						       COPY_LIMIT,
					       "draw rounds plus accepted = tokens");
					expect(statistics.source[DRAFT_DSPARK]
							       .rounds != 0,
					       "DSpark handles the unique suffix");
					expect((statistics.source[DRAFT_LOOKUP]
							.rounds != 0) ==
						       (hybrid != 0 &&
							low_estimate),
					       "lookup wins only against low DSpark estimates; structured DSpark keeps its rounds");
					if (hybrid != 0 && low_estimate) {
						expect(statistics.source[DRAFT_DSPARK]
									       .rounds >
								       1u &&
							       statistics.source[DRAFT_DSPARK]
									       .rounds >=
								       statistics.drawing_rounds /
									       (policy.probe_rounds +
										1u),
						       "continuous lookup cannot starve periodic DSpark verification");
					}
					uint64_t depth_rounds = 0,
						 depth_proposals = 0;
					for (uint32_t depth = 0;
					     depth < DRAFT_DEPTH_COUNTS;
					     depth++) {
						depth_rounds +=
							statistics.depth_rounds
								[depth];
						depth_proposals +=
							depth *
							statistics.depth_rounds
								[depth];
					}
					expect(depth_rounds == statistics
								       .drawing_rounds &&
						       depth_proposals ==
							       proposed,
					       "chosen-depth histogram counts each drawing round and verified proposal once");
					const uint32_t sampling_mode =
						seed_case != 0 ? 1u : 0u;
					for (uint32_t position = 0;
					     position < MODEL_DRAFT_BLOCK;
					     position++) {
						uint64_t verified = 0,
							 reached = 0,
							 matched = 0;
						for (uint32_t mode = 0;
						     mode <
						     DRAFT_SAMPLING_MODES;
						     mode++) {
							for (uint32_t bin = 0;
							     bin <
							     DRAFT_CONFIDENCE_BINS;
							     bin++) {
								const struct draft_outcomes *const outcome =
									&statistics
										 .confidence
											 [mode]
											 [position]
											 [bin];
								produced +=
									outcome->produced;
								verified +=
									outcome->verified;
								reached +=
									outcome->reached;
								matched +=
									outcome->matched;
								expect((mode == sampling_mode &&
									bin == draft_confidence_bin(
										       1.0f)) ||
									       outcome->produced ==
										       0,
								       "confidence bin and sampling mode are exact");
							}
						}
						expect(verified == statistics.source[DRAFT_DSPARK]
									       .proposed[position] &&
							       reached ==
								       statistics
									       .source[DRAFT_DSPARK]
									       .reached[position] &&
							       matched ==
								       statistics
									       .source[DRAFT_DSPARK]
									       .accepted[position],
						       "confidence outcomes equal the source counters");
					}
					expect(produced != 0 &&
						       produced % MODEL_DRAFT_BLOCK ==
							       0,
					       "all five produced scores counted, including unused tails");
				}
				expect(engine_draft_statistics(
					       engine, 0, &total, &total) ==
					       STATUS_INVALID_ARGUMENT,
				       "statistics refuse aliased outputs");
				test_serving_close();
			}
		}
    }
    test_serving_set_draft_policy(NULL, 0);
    test_serving_set_device_draft(false, MODEL_DRAFT_BLOCK);
}

static engine_draft_function observed_device_draft;
static uint64_t observed_dspark_rounds, last_dspark_at, longest_dspark_gap;

/* Observe completed source choices through the existing drafter seam, leaving the device drafter in charge. */
static uint32_t observed_draft(struct inference_engine *engine,
			       const struct command_cursor *cursor,
			       uint32_t anchor, uint32_t most,
			       uint32_t *proposals)
{
	const struct draft_statistics *const statistics = &cursor->statistics;
	const uint64_t rounds = statistics->source[DRAFT_DSPARK].rounds;
	if (rounds != observed_dspark_rounds) {
		observed_dspark_rounds = rounds;
		last_dspark_at = statistics->drawing_rounds;
	}
    if (observed_dspark_rounds != 0) {
        const uint64_t gap = statistics->drawing_rounds - last_dspark_at + 1u;
        longest_dspark_gap = gap > longest_dspark_gap ? gap : longest_dspark_gap;
    }
    return observed_device_draft(engine, cursor, anchor, most, proposals);
}

/* draft_learned_acceptance: after a command whose device drafts miss at DEVICE_CONFIDENCE_POSITION (confidence 1.0 at
 * every position), each table cell of the engine with reached drafts equals (matched x 1,000 + its prior x 16) /
 * (reached + 16), rounded (engine.c's ACCEPTANCE_PRIOR_REACHED), the others their prior; without it, every cell the
 * prior. */
static void test_learned_acceptance(void)
{
	static struct generation_event events[MOST_EVENTS];
	const uint64_t prior_weight = 16u;
	struct draft_policy_record prior;
	draft_policy_default(&prior);
	test_serving_set_function_model(true, COPY_PERIOD);
	test_serving_set_draft_policy(NULL, 0);
	for (uint32_t learned = 0; learned < TWO; learned++) {
		context_text = "learned acceptance";
		test_serving_set_learned_acceptance(learned != 0);
		test_serving_set_device_draft(true, DEVICE_CONFIDENCE_POSITION);
		struct inference_engine *const engine =
			drafting_engine(SPECULATIVE_CONTEXT, NULL);
		const struct generation_command_record command =
			function_command(0, 0, SPECULATIVE_PROMPT, COPY_LIMIT);
		if (engine == NULL) {
			continue;
		}
		expect(engine_submit(engine, &command) == STATUS_OK,
		       "learning command submitted");
		(void)events_until_final(engine, events);
		struct draft_statistics statistics = { 0 };
		expect(engine_draft_statistics(engine, 0, &statistics, NULL) ==
			       STATUS_OK,
		       "learning statistics read");
		uint64_t reached = 0, matched = 0;
		for (uint32_t mode = 0; mode < DRAFT_SAMPLING_MODES; mode++) {
			for (uint32_t position = 0;
			     position < DRAFT_CALIBRATION_POSITIONS;
			     position++) {
				for (uint32_t bin = 0;
				     bin < DRAFT_CONFIDENCE_BINS; bin++) {
					const struct draft_outcomes *const outcome =
						&statistics.confidence[mode]
								      [position]
								      [bin];
					const uint64_t given =
						prior.acceptance[mode][position]
								[bin];
					const uint64_t expected =
						learned != 0 && outcome->reached !=
									0 ?
							(outcome->matched *
								 DRAFT_PROBABILITY_SCALE +
							 given * prior_weight +
							 (outcome->reached +
							  prior_weight) /
								 TWO) /
								(outcome->reached +
								 prior_weight) :
							given;
					reached += outcome->reached;
					matched += outcome->matched;
					expect(engine->draft_policy.acceptance
							       [mode][position]
							       [bin] ==
						       expected,
					       "each cell the learned estimate of its reached drafts, else its prior");
				}
			}
		}
		expect(reached > matched && matched != 0,
		       "drafts were reached, matched and missed");
		test_serving_close();
	}
    test_serving_set_learned_acceptance(false);
    test_serving_set_device_draft(false, MODEL_DRAFT_BLOCK);
}

/* draft_session_acceptance: the same command (device drafts missing at DEVICE_CONFIDENCE_POSITION) under the default
 * cost policy gives every event of plain decoding at each level; with the knob its session keeps decayed counts of
 * its reached drafts, matched (at most the command's matches, some) and expected by the table (some), and none
 * without; at levels 2 to 4 a finite fitted shift, at 3 alone a finite scale of at least 0; at every level the
 * decayed count of reached drafts at least the decayed matches. */
static void test_session_acceptance(void)
{
	static struct generation_event reference[MOST_EVENTS],
		events[MOST_EVENTS];
	test_serving_set_function_model(true, COPY_PERIOD);
	test_serving_set_draft_policy(NULL, 0);
	const struct generation_command_record command =
		function_command(0, 0, SPECULATIVE_PROMPT, COPY_LIMIT);
	test_serving_set_device_draft(false, MODEL_DRAFT_BLOCK);
	test_serving_set_engine(1, 1, SPECULATIVE_ROWS, 0);
	const uint32_t reference_count = command_events(
		SPECULATIVE_CONTEXT, &command, NULL, reference, NULL);
	test_serving_set_engine(1, PIPELINE_DEPTH, SPECULATIVE_ROWS,
				MODEL_DRAFT_BLOCK);
	for (uint32_t session = 0; session <= SESSION_ACCEPTANCE_LEVELS;
	     session++) {
		context_text = "session acceptance";
		test_serving_set_session_acceptance(session);
		test_serving_set_device_draft(true, DEVICE_CONFIDENCE_POSITION);
		struct inference_engine *const engine =
			drafting_engine(SPECULATIVE_CONTEXT, NULL);
		if (engine == NULL) {
			continue;
		}
		expect(engine_submit(engine, &command) == STATUS_OK,
		       "session command submitted");
		const uint32_t count = events_until_final(engine, events);
		expect(events_match(events, count, reference, reference_count),
		       "the session factor changes no event");
		struct draft_statistics statistics = { 0 };
		expect(engine_draft_statistics(engine, 0, &statistics, NULL) ==
			       STATUS_OK,
		       "session statistics read");
		const struct command_cursor *const cursor = &engine->cursors[0];
		const double matched =
			(double)statistics.source[DRAFT_DSPARK].accepted[0] +
			(double)statistics.source[DRAFT_DSPARK].accepted[1];
		expect(session != 0 ?
			       cursor->session_hits > 0.0 &&
				       cursor->session_expected > 0.0 &&
				       cursor->session_hits <= matched + 1.0 :
			       cursor->session_hits == 0.0 &&
				       cursor->session_expected == 0.0,
		       "a session's decayed counts with the knob, none without");
		expect(session < SESSION_SHIFT_LEVEL ||
			       session > SESSION_HINDSIGHT_LEVEL ||
			       (cursor->session_seen != 0 &&
				isfinite(cursor->session_shift)),
		       "levels 2 to 4 fit a finite shift to the reached drafts");
		expect(session == 0 ? cursor->session_reached == 0.0 :
				      cursor->session_reached > 0.0 &&
					      cursor->session_hits <=
						      cursor->session_reached,
		       "a session's decayed count of reached drafts, at least its decayed matches");
		expect(session == SESSION_PLATT_LEVEL ?
			       isfinite(cursor->session_slope_offset) &&
				       cursor->session_slope_offset >= -1.0 :
			       cursor->session_slope_offset == 0.0,
		       "level 3 alone fits a scale, finite and at least 0");
		for (uint32_t position = 0;
		     position < DRAFT_CALIBRATION_POSITIONS; position++) {
			expect(cursor->session_position_hits[position] >= 0.0 &&
				       cursor->session_position_hits[position] <=
					       cursor->session_position_reached
						       [position],
			       "position-local matches are bounded by conditional observations");
			expect(cursor->session_position_expected[position] >=
					       0.0 &&
				       cursor->session_position_expected
						       [position] <=
					       cursor->session_position_reached
						       [position],
			       "residual expectations contain only reached conditional observations");
			expect(session >= SESSION_RESIDUAL_LEVEL ||
				       cursor->session_position_expected
						       [position] == 0.0,
			       "only residual calibration records position expectations");
			expect(session >= SESSION_POSITION_LEVEL ||
				       cursor->session_position_reached
						       [position] == 0.0,
			       "only levels 6 and 7 record position-local observations");
		}
		expect(session < SESSION_POSITION_LEVEL ||
			       (cursor->session_position_reached
						[DEVICE_CONFIDENCE_POSITION] >
					0.0 &&
				cursor->session_position_hits
						[DEVICE_CONFIDENCE_POSITION] ==
					0.0),
		       "the first cut draft is labeled by the valid bonus draw, including its mismatch");
		test_serving_close();
	}
    test_serving_set_session_acceptance(0);
    test_serving_set_device_draft(false, MODEL_DRAFT_BLOCK);
}

/* A misleading shared confidence table must not keep an always-correct request shallow. A following request on
 * the same slot rejects at position 2: it must forget the earlier command and never learn from rejected history. */
static void test_session_position_policy(void)
{
	static struct generation_event reference[MOST_EVENTS],
		events[MOST_EVENTS];
	const struct generation_command_record command =
		function_command(0, 0, SPECULATIVE_PROMPT, SPECULATIVE_LIMIT);
	test_serving_set_function_model(true, COPY_PERIOD);
	test_serving_set_device_draft(false, MODEL_DRAFT_BLOCK);
	test_serving_set_engine(1, 1, SPECULATIVE_ROWS, 0);
	const uint32_t reference_count = command_events(
		SPECULATIVE_CONTEXT, &command, NULL, reference, NULL);
	struct draft_policy_record policy;
	draft_policy_default(&policy);
	policy.depth_policy = DRAFT_DEPTH_RATE;
	for (uint32_t mode = 0; mode < DRAFT_SAMPLING_MODES; mode++)
		for (uint32_t position = 0;
		     position < DRAFT_CALIBRATION_POSITIONS; position++)
			for (uint32_t bin = 0; bin < DRAFT_CONFIDENCE_BINS;
			     bin++)
				policy.acceptance[mode][position][bin] =
					POSITION_POLICY_PROBABILITY;
	test_serving_set_engine(1, 1, SPECULATIVE_ROWS, MODEL_DRAFT_BLOCK);
	test_serving_set_draft_policy(&policy, 0);
	for (uint32_t session = SESSION_POSITION_LEVEL;
	     session <= SESSION_COST_LEVEL; session++) {
		test_serving_set_session_acceptance(session);
		test_serving_set_device_draft(true, MODEL_DRAFT_BLOCK);
		struct inference_engine *const engine =
			drafting_engine(SPECULATIVE_CONTEXT, NULL);
		if (engine != NULL) {
			expect(engine_submit(engine, &command) == STATUS_OK,
			       "position policy command submitted");
			uint32_t count = events_until_final(engine, events);
			expect(events_match(events, count, reference,
					    reference_count),
			       "position learning preserves target events");
			expect(engine->cursors[0].statistics.depth_rounds
					       [MODEL_DRAFT_BLOCK] != 0,
			       "a consistently correct request learns to verify the complete draft block");
			if (session >= SESSION_SECOND_LEVEL &&
			    session != SESSION_COST_LEVEL)
				expect(engine->cursors[0].statistics.depth_rounds
							       [MODEL_DRAFT_BLOCK] +
						       (session == SESSION_NATURAL_LEVEL ?
								SESSION_NATURAL_EDGE_ROUNDS :
								SESSION_DEPTH_EDGE_ROUNDS) >=
					       engine->cursors[0]
						       .statistics
						       .source[DRAFT_DSPARK]
						       .rounds,
				       "a correct request keeps full depth after learning, apart from initial/final "
				       "edges despite an underestimating table");
			test_serving_set_device_draft(
				true, DEVICE_CONFIDENCE_POSITION);
			expect(engine_submit(engine, &command) == STATUS_OK,
			       "following position command submitted");
			count = events_until_final(engine, events);
			expect(events_match(events, count, reference,
					    reference_count),
			       "a rejecting request preserves target events");
			const struct command_cursor *const cursor =
				&engine->cursors[0];
			expect(cursor->session_position_hits
						       [DEVICE_CONFIDENCE_POSITION] ==
					       0.0 &&
				       cursor->session_position_reached
						       [DEVICE_CONFIDENCE_POSITION] >
					       0.0 &&
				       cursor->session_position_reached
						       [DEVICE_CONFIDENCE_POSITION +
							1u] == 0.0,
			       "new commands reset the conditional history and rejected suffixes supply no labels");
			if (session >= SESSION_SCOUT_LEVEL) {
				const struct draft_source_statistics *const
					source = &cursor->statistics
							  .source[DRAFT_DSPARK];
				expect(!cursor->session_full_depth &&
					       source->rounds >
						       cursor->statistics.depth_rounds
								       [MODEL_DRAFT_BLOCK] +
							       1u,
				       "a request with poor middle drafts returns to the existing adaptive policy");
			}
			if (session == SESSION_NATURAL_LEVEL) {
				test_serving_set_device_draft(
					true, SECOND_DRAFT_POSITION);
				expect(engine_submit(engine, &command) ==
					       STATUS_OK,
				       "low-second natural command submitted");
				count = events_until_final(engine, events);
				expect(events_match(events, count, reference,
						    reference_count),
				       "natural second-position selection preserves target events");
				expect(cursor->statistics.depth_rounds
							       [MODEL_DRAFT_BLOCK] ==
						       0 &&
					       cursor->session_position_hits
							       [SECOND_DRAFT_POSITION] ==
						       0.0 &&
					       cursor->session_position_reached
							       [SECOND_DRAFT_POSITION] >
						       0.0,
				       "weak second-position drafts stay at the table depth "
				       "without a forced full block");
			}
			test_serving_close();
		}
	}
	test_serving_set_session_acceptance(0);
	test_serving_set_draft_policy(NULL, 0);
	test_serving_set_device_draft(false, MODEL_DRAFT_BLOCK);
}

/* --draft_start_economic: a command's first DSpark round at the policy's depth, not a forced probe of one draft; the
 * events of plain decoding either way. */
static void test_start_economic(void)
{
	static struct generation_event reference[MOST_EVENTS],
		events[MOST_EVENTS];
	struct draft_policy_record fixed;
	draft_policy_default(&fixed);
	fixed.depth_policy = DRAFT_DEPTH_FIXED;
	test_serving_set_function_model(true, COPY_PERIOD);
	test_serving_set_draft_policy(&fixed, 0);
	const struct generation_command_record command =
		function_command(0, 0, SPECULATIVE_PROMPT, COPY_LIMIT);
	test_serving_set_device_draft(false, MODEL_DRAFT_BLOCK);
	test_serving_set_engine(1, 1, SPECULATIVE_ROWS, 0);
	const uint32_t reference_count = command_events(
		SPECULATIVE_CONTEXT, &command, NULL, reference, NULL);
	test_serving_set_engine(1, 1, SPECULATIVE_ROWS, MODEL_DRAFT_BLOCK);
	for (uint32_t economic = 0; economic < TWO; economic++) {
		context_text = "start economic";
		test_serving_set_start_economic(economic != 0);
		test_serving_set_device_draft(true, MODEL_DRAFT_BLOCK);
		struct inference_engine *const engine =
			drafting_engine(SPECULATIVE_CONTEXT, NULL);
		if (engine == NULL) {
			continue;
		}
		expect(engine_submit(engine, &command) == STATUS_OK,
		       "start command submitted");
		const uint32_t count = events_until_final(engine, events);
		expect(events_match(events, count, reference, reference_count),
		       "the first round's depth changes no event");
		const struct command_cursor *const cursor = &engine->cursors[0];
		expect(economic != 0 ? cursor->live_probes == 0 :
				       cursor->live_probes != 0,
		       "the first DSpark round a probe without the knob, at the policy's depth with it");
		test_serving_close();
	}
    test_serving_set_start_economic(false);
    test_serving_set_draft_policy(NULL, 0);
    test_serving_set_device_draft(false, MODEL_DRAFT_BLOCK);
}

static void test_dspark_probe_intervals(void)
{
	static const uint32_t intervals[] = { 1, TWO,
					      DRAFT_DEFAULT_PROBE_ROUNDS };
	static struct generation_event reference[MOST_EVENTS],
		events[MOST_EVENTS];
	const struct generation_command_record command =
		function_command(0, 1, SPECULATIVE_PROMPT, SPECULATIVE_LIMIT);
	test_serving_set_function_model(true, COPY_PERIOD);
	test_serving_set_device_draft(false, MODEL_DRAFT_BLOCK);
	test_serving_set_engine(1, 1, SPECULATIVE_ROWS, 0);
	const uint32_t reference_count = command_events(
		SPECULATIVE_CONTEXT, &command, NULL, reference, NULL);
	for (uint32_t index = 0;
	     index < sizeof(intervals) / sizeof(intervals[0]); index++) {
		for (uint32_t hybrid = 0; hybrid < TWO; hybrid++) {
			struct draft_policy_record policy;
			draft_policy_default(&policy);
			memset(policy.acceptance, 0, sizeof(policy.acceptance));
			policy.probe_rounds = intervals[index];
			test_serving_set_draft_policy(
				&policy,
				hybrid != 0 ? DRAFT_DEFAULT_LOOKUP_MIN : 0);
			test_serving_set_device_draft(true, MODEL_DRAFT_BLOCK);
			test_serving_set_engine(1, PIPELINE_DEPTH,
						SPECULATIVE_ROWS,
						MODEL_DRAFT_BLOCK);
			context_text =
				"bounded DSpark probes with and without continuous lookup";
			struct inference_engine *const engine =
				drafting_engine(SPECULATIVE_CONTEXT, NULL);
			if (engine != NULL) {
				observed_device_draft = engine->draft;
				engine->draft = observed_draft;
				observed_dspark_rounds = last_dspark_at =
					longest_dspark_gap = 0;
				expect(engine_submit(engine, &command) ==
					       STATUS_OK,
				       "probe command submitted");
				const uint32_t count =
					events_until_final(engine, events);
				struct draft_statistics statistics = { 0 };
				expect(events_match(events, count, reference,
						    reference_count),
				       "probes preserve every token event");
				expect(engine_draft_statistics(
					       engine, 0, &statistics, NULL) ==
					       STATUS_OK,
				       "probe interval statistics read");
				expect(observed_dspark_rounds > 1u &&
					       longest_dspark_gap <=
						       policy.probe_rounds,
				       "no eligible gap exceeds N, including when a skipped tail must be refreshed first");
				expect(statistics.source[DRAFT_DSPARK]
						       .proposed[1] == 0,
				       "unprofitable DSpark is refreshed and verified only at depth one");
				expect((statistics.source[DRAFT_LOOKUP].rounds !=
					0) == (hybrid != 0 &&
					       policy.probe_rounds > 1u),
				       "lookup yields every reserved DSpark round");
			}
			test_serving_close();
		}
    }
    test_serving_set_draft_policy(NULL, 0);
    test_serving_set_device_draft(false, MODEL_DRAFT_BLOCK);
}

static void test_hybrid_match_length(void)
{
	static struct generation_event events[MOST_EVENTS],
		reference[MOST_EVENTS];
	static const uint32_t tokens[] = {
		7, 2, 2, 1, 2
	}; /* Only the final two-token suffix recurs after the first draw. */
	struct generation_command_record command = function_command(
		0, 0, sizeof(tokens) / sizeof(tokens[0]), TWO + 1u);
	command.new_token_ids = tokens;
	test_serving_set_function_model(true, 1);
	test_serving_set_engine(1, 1, SPECULATIVE_ROWS, MODEL_DRAFT_BLOCK);
	for (uint32_t minimum = NGRAM_LEAST;
	     minimum <= DRAFT_DEFAULT_LOOKUP_MIN; minimum++) {
		context_text = "hybrid requires the configured suffix length";
		test_serving_set_draft_policy(NULL, minimum);
		test_serving_set_device_draft(true, MODEL_DRAFT_BLOCK);
		struct inference_engine *const engine =
			drafting_engine(SPECULATIVE_CONTEXT, NULL);
		if (engine != NULL) {
			expect(engine_submit(engine, &command) == STATUS_OK,
			       "suffix command submitted");
			(void)events_until_final(engine, events);
			struct draft_statistics statistics = { 0 };
			expect(engine_draft_statistics(engine, 0, &statistics,
						       NULL) == STATUS_OK,
			       "suffix statistics read");
			expect(statistics.source[DRAFT_DSPARK].rounds == 1 &&
				       statistics.source[DRAFT_LOOKUP].rounds ==
					       0,
			       "the initial DSpark probe wins even when a two-token suffix qualifies");
		}
		test_serving_close();
	}
    context_text = "statistics censor proposals after the first rejected token";
    test_serving_set_function_model(true, COPY_PERIOD);
    test_serving_set_draft_policy(NULL, 0);
    test_serving_set_device_draft(true, 1);
    struct inference_engine *const engine =
	    drafting_engine(SPECULATIVE_CONTEXT, NULL);
    command = function_command(0, 1, SPECULATIVE_PROMPT, COPY_LIMIT);
    if (engine != NULL) {
	    expect(engine_submit(engine, &command) == STATUS_OK,
		   "rejection command submitted");
	    (void)events_until_final(engine, events);
	    struct draft_statistics statistics = { 0 };
	    expect(engine_draft_statistics(engine, 0, &statistics, NULL) ==
			   STATUS_OK,
		   "rejection statistics read");
	    const struct draft_source_statistics *const source =
		    &statistics.source[DRAFT_DSPARK];
	    const uint32_t bin = draft_confidence_bin(1.0f);
	    expect(source->reached[1] > 0 && source->accepted[1] == 0 &&
			   source->proposed[DEVICE_CONFIDENCE_POSITION] > 0 &&
			   source->reached[DEVICE_CONFIDENCE_POSITION] == 0,
		   "first mismatch is observed, later rows are censored");
	    expect(statistics.confidence[1][1][bin].reached ==
				   source->reached[1] &&
			   statistics.confidence[1][1][bin].matched == 0 &&
			   statistics.confidence[1][DEVICE_CONFIDENCE_POSITION]
						[bin]
							.reached == 0,
		   "histograms retain the rejection and exclude conditioned-on-wrong-history draws");
    }
    test_serving_close();
    test_serving_set_device_draft(true, 0);
    context_text = "DSpark reconsiders confidence every round without an acceptance cooldown";
    struct inference_engine *const poor_engine =
	    drafting_engine(SPECULATIVE_CONTEXT, NULL);
    command = function_command(0, 1, SPECULATIVE_PROMPT, SPECULATIVE_LIMIT);
    if (poor_engine != NULL) {
	    expect(engine_submit(poor_engine, &command) == STATUS_OK,
		   "poor command submitted");
	    (void)events_until_final(poor_engine, events);
	    struct draft_statistics statistics = { 0 };
	    expect(engine_draft_statistics(poor_engine, 0, &statistics, NULL) ==
			   STATUS_OK,
		   "continuous DSpark statistics read");
	    const struct draft_source_statistics *const source =
		    &statistics.source[DRAFT_DSPARK];
	    expect(source->rounds > SPECULATIVE_LIMIT /
					    DRAFT_DEFAULT_PROBE_ROUNDS &&
			   source->proposed[1] != 0,
		   "an observed rejection never installs a blind cooldown when confidence still predicts gain");
    }
    test_serving_close();
    test_serving_set_device_draft(false, MODEL_DRAFT_BLOCK);
    context_text = "calibration still gates a one-proposal budget after a successful probe";
    test_serving_set_engine(1, 1, TWO, 0);
    command = function_command(0, 1, SPECULATIVE_PROMPT, COPY_LIMIT);
    const uint32_t reference_count = command_events(SPECULATIVE_CONTEXT, &command, NULL, reference, NULL);
    struct draft_policy_record policy;
    draft_policy_default(&policy);
    policy.threshold = DRAFT_PROBABILITY_SCALE;
    test_serving_set_draft_policy(&policy, 0);
    test_serving_set_device_draft(true, MODEL_DRAFT_BLOCK);
    test_serving_set_engine(1, 1, TWO, 1);
    struct inference_engine *const narrow_engine =
	    drafting_engine(SPECULATIVE_CONTEXT, NULL);
    if (narrow_engine != NULL) {
	    expect(engine_submit(narrow_engine, &command) == STATUS_OK,
		   "narrow command submitted");
	    const uint32_t count = events_until_final(narrow_engine, events);
	    struct draft_statistics statistics = { 0 };
	    expect(engine_draft_statistics(narrow_engine, 0, &statistics,
					   NULL) == STATUS_OK,
		   "narrow statistics read");
	    const struct draft_source_statistics *const source =
		    &statistics.source[DRAFT_DSPARK];
	    expect(events_match(events, count, reference, reference_count),
		   "narrow cost policy preserves every event");
	    expect(source->rounds > 1u &&
			   source->rounds <=
				   (COPY_LIMIT + policy.probe_rounds) /
					   policy.probe_rounds &&
			   source->proposed[1] == 0 &&
			   source->accepted[0] == source->rounds &&
			   statistics.depth_rounds[0] > source->rounds,
		   "successful one-row probes recover, then the calibrated zero-depth policy resumes");
    }
    test_serving_close();
    test_serving_set_draft_policy(NULL, 0);
    test_serving_set_device_draft(false, MODEL_DRAFT_BLOCK);
}

struct live_wait_fixture {
	struct step_driver driver;
	atomic_bool entered, is_released;
	result_status result;
};

MUST_CHECK static result_status
live_wait_blocked(struct device_card *card, uint64_t wait_limit_nanoseconds)
{
    if (card == NULL || wait_limit_nanoseconds == 0) {
	    return STATUS_INVALID_ARGUMENT;
    }
    struct live_wait_fixture *const fixture = (void *)card;
    atomic_store_explicit(&fixture->entered, true, memory_order_release);
    const struct timespec delay = {0, (long)STEP_DELAY_NANOSECONDS};
    for (uint32_t look = 0; look < MOST_LOOKS; look++) {
        if (atomic_load_explicit(&fixture->is_released, memory_order_acquire)) {
		return STATUS_OK;
	}
        if (nanosleep(&delay, NULL) != 0) {
		return STATUS_INPUT_OUTPUT_ERROR;
	}
    }
    return STATUS_TIMED_OUT;
}

static void *live_wait_worker(void *argument)
{
    struct live_wait_fixture *const fixture = argument;
    fixture->result = card_wait(&fixture->driver, 0);
    return NULL;
}

static void test_live_busy_wait(void)
{
	static const struct backend_operations operations = {
		.launch_wait = live_wait_blocked
	};
	struct live_wait_fixture fixture = {
		.driver = { .backend = &operations,
			    .wait_limit_nanoseconds =
				    LIVE_WAIT_LIMIT_NANOSECONDS }
	};
	atomic_init(&fixture.entered, false);
	atomic_init(&fixture.is_released, false);
	atomic_init(&fixture.driver.live_wait_sequence[0], 0);
	atomic_init(&fixture.driver.live_wait_completed_ns[0], 0);
	atomic_init(&fixture.driver.live_wait_started_ns[0], 0);
	fixture.driver.cards[0] = (struct device_card *)&fixture;
	fixture.driver.uncovered_work[0] = true;
	context_text =
		"release busy time includes the overlap of a still blocked existing card wait";
	pthread_t worker;
	if (pthread_create(&worker, NULL, live_wait_worker, &fixture) != 0) {
		expect(false, "the synthetic wait thread starts");
		return;
	}
    for (uint32_t look = 0; look < MOST_LOOKS &&
         !atomic_load_explicit(&fixture.entered, memory_order_acquire); look++) {
        pause_for(STEP_DELAY_NANOSECONDS);
    }
    expect(atomic_load_explicit(&fixture.entered, memory_order_acquire), "the existing wait is blocked");
    struct engine_card_stats first = { 0 }, second = { 0 }, finished = { 0 };
    step_live_cards(&fixture.driver, &first, 1);
    pause_for(LIVE_WAIT_SAMPLE_NANOSECONDS);
    step_live_cards(&fixture.driver, &second, 1);
    expect(first.busy_available && second.busy_available && first.busy_ns != 0 &&
           second.busy_ns > first.busy_ns && second.busy_observed_ns > first.busy_observed_ns &&
           second.busy_ns - first.busy_ns == second.busy_observed_ns - first.busy_observed_ns,
           "two snapshots inside one active wait advance at exactly the observed wall-time rate");
    atomic_store_explicit(&fixture.is_released, true, memory_order_release);
    expect(pthread_join(worker, NULL) == 0 && fixture.result == STATUS_OK,
	   "the existing wait finishes successfully");
    step_live_cards(&fixture.driver, &finished, 1);
    expect(finished.busy_available && finished.busy_ns == fixture.driver.live_busy_ns[0] &&
           finished.busy_ns >= second.busy_ns && finished.busy_observed_ns >= second.busy_observed_ns &&
           finished.busy_ns - second.busy_ns <= finished.busy_observed_ns - second.busy_observed_ns,
           "wait completion adds only the remaining overlap and cannot create a utilization spike");
    const uint64_t known_completed = finished.busy_ns;
    (void)atomic_fetch_add_explicit(&fixture.driver.live_wait_sequence[0], 1, memory_order_seq_cst);
    step_live_cards(&fixture.driver, &finished, 1);
    expect(!finished.busy_available && finished.busy_observed_ns == 0 && finished.busy_ns == known_completed,
           "bounded publication contention retains known time and reports the sample unavailable");
    (void)atomic_fetch_add_explicit(&fixture.driver.live_wait_sequence[0], 1, memory_order_seq_cst);
    step_live_cards(&fixture.driver, &finished, 1);
    expect(finished.busy_available && finished.busy_ns == known_completed,
           "the next coherent refresh recovers without resetting the cumulative counter");
}

static void test_live_statistics(uint32_t depth)
{
	static struct generation_event events[SESSIONS][MOST_EVENTS];
	uint32_t counts[SESSIONS] = { 0 };
	struct inference_engine *engine = NULL;
	context_text = "release statistics of independent concurrent commands";
	test_serving_set_function_model(false, 0);
	test_serving_set_engine(SESSIONS, depth, 1, 0);
	expect(test_serving_open(CONTEXT, scripts,
				 sizeof(scripts) / sizeof(scripts[0]),
				 &engine) == STATUS_OK,
	       "the concurrent fixture opens");
	if (engine == NULL) {
		test_serving_set_step_delay(0);
		return;
	}
    test_serving_set_step_delay(STEP_DELAY_NANOSECONDS);
    struct engine_stats snapshot = { 0 }, refused = { 0 };
    struct engine_slot_stats slot_record = { 0 };
    struct engine_card_stats card = { .arena_bytes = LIVE_ARENA_BYTES,
				      .layer_mask = 1 };
    expect(test_host_memory_backend.card_describe(0, &card.description) ==
			   STATUS_OK &&
		   engine_stats_cards(engine, &card, 1) == STATUS_OK,
	   "startup card metadata retained");
    expect(engine_stats_snapshot(NULL, &refused) == STATUS_INVALID_ARGUMENT &&
		   refused.started_ns == 0 &&
		   engine_stats_snapshot(engine, NULL) ==
			   STATUS_INVALID_ARGUMENT &&
		   engine_stats_slot(engine, SESSIONS, &slot_record) ==
			   STATUS_INVALID_ARGUMENT,
	   "bad snapshot inputs leave the output untouched");
    expect(engine_stats_counters(NULL, &refused) == STATUS_INVALID_ARGUMENT &&
		   refused.started_ns == 0 &&
		   engine_stats_counters(engine, NULL) ==
			   STATUS_INVALID_ARGUMENT,
	   "counter-only snapshot validates without consuming or modifying observer state");
    uint64_t expected_prompt_tokens = 0;
    for (uint32_t slot = 0; slot < SESSIONS; slot++) {
	    struct generation_command_record command =
		    command_of(PROMPT_TOKENS + slot, LIVE_GENERATED_TOKENS,
			       SCRIPT_ENDLESS);
	    command.session_slot = slot;
	    command.request_tag = slot + 1u;
	    expected_prompt_tokens += command.new_token_count;
	    expect(engine_submit(engine, &command) == STATUS_OK,
		   "independent command admitted");
    }
    expect(engine_stats_cards(engine, &card, 1) == STATUS_BUSY_VALUE,
	   "metadata cannot change under admitted commands");
    expect(engine_stats_counters(engine, &snapshot) == STATUS_OK &&
		   snapshot.slots_busy == SESSIONS,
	   "all admitted sessions remain active until their final events are taken");
    for (uint32_t slot = 0; slot < SESSIONS; slot++) {
	    expect(snapshot.sessions[slot].active &&
			   snapshot.sessions[slot].context_capacity ==
				   CONTEXT &&
			   snapshot.sessions[slot].phase != STATS_IDLE,
		   "counter-only snapshot includes each busy session without a device query");
    }
    uint32_t finals = 0, final_slot = SESSIONS;
    bool saw_prompt = false, saw_decode = false, saw_flight = false;
    while (finals < SESSIONS) {
	    expect(engine_stats_snapshot(engine, &snapshot) == STATUS_OK,
		   "snapshot runs beside live commands");
	    saw_flight = saw_flight || snapshot.flights != 0;
	    for (uint32_t slot = 0; slot < SESSIONS; slot++) {
		    const struct engine_slot_stats *const request =
			    &snapshot.slots[slot];
		    saw_prompt = saw_prompt || request->phase == STATS_PROMPT;
		    saw_decode = saw_decode || request->phase == STATS_DECODE;
		    expect(request->slot == slot &&
				   request->request_tag == slot + 1u &&
				   request->prompt_tokens ==
					   PROMPT_TOKENS + slot &&
				   request->reused_tokens == 0 &&
				   request->context_capacity == CONTEXT &&
				   request->prompt_processed <=
					   request->prompt_tokens &&
				   request->generated_tokens <=
					   LIVE_GENERATED_TOKENS,
			   "each live identity, prompt progress and generation count stays with its owner");
	    }
        if (!event_filed(engine, events, counts, &final_slot)) {
            break;
        }
        finals += final_slot < SESSIONS ? 1u : 0u;
    }
    expect(engine_stats_snapshot(engine, &snapshot) == STATUS_OK &&
		   snapshot.slots_total == SESSIONS &&
		   snapshot.slots_busy == 0 && snapshot.queued_commands == 0 &&
		   snapshot.event_queue_depth == 0 && snapshot.flights == 0 &&
		   snapshot.served == SESSIONS && snapshot.failed == 0 &&
		   snapshot.cancelled == 0 &&
		   snapshot.prompt_tokens == expected_prompt_tokens &&
		   snapshot.prefill_tokens == expected_prompt_tokens &&
		   snapshot.cached_tokens == 0 &&
		   snapshot.generated_tokens ==
			   (uint64_t)SESSIONS * LIVE_GENERATED_TOKENS &&
		   snapshot.steps_landed ==
			   expected_prompt_tokens +
				   (uint64_t)SESSIONS *
					   (LIVE_GENERATED_TOKENS - 1u) &&
		   snapshot.steps[0] == snapshot.steps_landed &&
		   snapshot.step_ring_count == STATS_STEP_RING,
	   "global counters equal all known independent work, including a wrapped step ring");
    expect(saw_prompt && saw_decode && (depth == 1 || saw_flight) && snapshot.started_ns != 0 &&
           snapshot.now_ns >= snapshot.started_ns, "live phases, flights and process timeline are visible");
    expect(snapshot.card_count == 1 &&
		   snapshot.cards[0].free_memory_available &&
		   snapshot.cards[0].arena_bytes == LIVE_ARENA_BYTES &&
		   snapshot.cards[0].layer_mask == 1 &&
		   snapshot.cards[0].description.arithmetic_class ==
			   ARITHMETIC_CLASS_SM_80 &&
		   snapshot.cards[0].steps == snapshot.steps_landed &&
		   snapshot.cards[0].visits == snapshot.steps_landed &&
		   snapshot.cards[0].busy_ns != 0,
	   "card work and existing-wait busy time are always counted");
    for (uint32_t slot = 0; slot < SESSIONS; slot++) {
	    const struct engine_slot_stats *const request =
		    &snapshot.slots[slot];
	    expect(!snapshot.sessions[slot].active &&
			   snapshot.sessions[slot].phase == STATS_IDLE &&
			   snapshot.sessions[slot].position ==
				   request->position &&
			   snapshot.sessions[slot].reach == request->reach &&
			   snapshot.sessions[slot].idle_since_ns != 0,
		   "each completed slot retains its context and independently becomes idle");
	    expect(request->phase == STATS_FINISHING &&
			   request->prompt_processed == PROMPT_TOKENS + slot &&
			   request->generated_tokens == LIVE_GENERATED_TOKENS &&
			   request->position == PROMPT_TOKENS + slot +
							LIVE_GENERATED_TOKENS -
							1u &&
			   request->reach == request->position &&
			   request->decoder_from == 0 &&
			   request->checkpoint == 0 &&
			   request->reused_checkpoint == false &&
			   request->chunks[0] == PROMPT_TOKENS + slot &&
			   request->speculation_mode == 0 &&
			   request->chosen_depth == 0 &&
			   !request->has_confidence && request->probes == 0 &&
			   request->dspark_tails == 0 &&
			   request->finish_kind ==
				   GENERATION_EVENT_FINISHED_AT_LIMIT &&
			   request->last_error == 0 &&
			   request->submitted_ns <= request->started_ns &&
			   request->started_ns <= request->first_token_ns &&
			   request->first_token_ns <= request->last_token_ns &&
			   request->last_token_ns <= request->finished_ns &&
			   request->prefill_ns != 0 &&
			   request->decode_ns != 0 &&
			   request->token_ring_count == STATS_TOKEN_RING &&
			   request->token_ring_next ==
				   LIVE_GENERATED_TOKENS % STATS_TOKEN_RING,
		   "every completed request retains exact work, state, speculation and timeline metrics");
	    uint64_t previous = 0;
	    for (uint32_t token = 0; token < STATS_TOKEN_RING; token++) {
		    const uint64_t current =
			    request->token_times[(request->token_ring_next +
						  token) %
						 STATS_TOKEN_RING];
		    expect(current != 0 && current >= previous,
			   "the fixed token ring retains ordered latest timestamps");
		    previous = current;
	    }
	    expect(engine_stats_slot(engine, slot, &slot_record) == STATUS_OK &&
			   slot_record.request_tag == slot + 1u &&
			   slot_record.generated_tokens ==
				   LIVE_GENERATED_TOKENS,
		   "final-event retention needs no device query");
    }
    test_serving_close();
    test_serving_set_step_delay(0);
    test_serving_set_engine(1, 1, TEST_SERVING_ROWS, 0);
}

int main(void)
{
	struct inference_engine *engine = NULL;
	for (uint32_t index = 0; index < ENDLESS_TOKENS; index++) {
		endless[index] = FIRST_TOKEN + index % TWO;
	}
    for (uint32_t index = 0; index < sizeof(prompt) / sizeof(prompt[0]); index++) {
        prompt[index] = index % PROMPT_VALUES + 1u;
    }
    for (uint32_t index = 0; index < sizeof(cone_prompt) / sizeof(cone_prompt[0]); index++) {
        cone_prompt[index] = index % PROMPT_VALUES + 1u;
    }
    for (uint32_t index = 0; index < COPY_PROMPT; index++) {
        copy_prompt[index] = index % COPY_PERIOD + 1u;
    }
    for (uint32_t index = 0; index < RING_CONTEXT; index++) {
	    ring_prompt[index] = (index * RING_PROMPT_STRIDE +
				  index / MODEL_RING_SLOTS * RING_PROMPT_TURN +
				  RING_PROMPT_FIRST) %
				 BYTE_VALUES;
    }
    for (uint32_t index = 0; index < DECOY_PROMPT; index++) {
        decoy_prompt[index] = decoy_block[index % DECOY_PERIOD];
    }
    endless[ENDLESS_TOKENS] = END;
    context_text = "open";
    if (test_serving_open(CONTEXT, scripts,
			  sizeof(scripts) / sizeof(scripts[0]),
			  &engine) != STATUS_OK) {
	    (void)printf("FAILED: the engine of the fixture does not open\n");
	    return 1;
    }
    test_commands(engine);
    test_refusals(engine);
    test_busy_and_cancel(engine);
    test_serving_close();
    test_ignore_end_token();
    test_context_end();
    test_long_session();
    test_destroy_and_faults();
    test_sessions();
    test_cone();
    test_lookup_drafter();
    test_speculative_rounds();
    test_speculative_limits();
    test_lookup_rounds();
    test_speculative_pipeline();
    test_speculative_cone();
    test_speculative_cancel(false);
    test_speculative_snapshot(false);
    test_prefill_beside_rounds();
    test_prefill_order();
    test_prefill_order_shortest();
    test_prefill_short_rows();
    test_lookup_depth();
    test_draft_cooldown();
    test_ring_exact_boundary();
    test_ring_lineage();
    test_ring_rejected();
    for (uint32_t block = MODEL_DRAFT_BLOCK; block <= STEP_DRAFT_MOST;
	 block++) {
	    test_device_rounds(false, block);
    }
    test_device_limits_and_sessions();
    test_device_end_positions();
    test_device_policy_and_refusals();
    test_policy_statistics();
    test_learned_acceptance();
    test_session_acceptance();
    test_session_position_policy();
    test_start_economic();
    test_dspark_probe_intervals();
    test_hybrid_match_length();
    struct draft_policy_record fixed_policy;
    draft_policy_default(&fixed_policy);
    fixed_policy.depth_policy = DRAFT_DEPTH_FIXED;
    test_serving_set_draft_policy(&fixed_policy, 0);
    test_speculative_rounds();
    for (uint32_t block = MODEL_DRAFT_BLOCK; block <= STEP_DRAFT_MOST;
	 block++) {
	    test_device_rounds(true, block);
    }
    test_serving_set_draft_policy(NULL, 0);
    test_policy_event_identity(DRAFT_DEPTH_RATE);
    test_policy_event_identity(DRAFT_DEPTH_ADAPTIVE);
    test_speculative_cancel(true);
    test_speculative_snapshot(true);
    test_serving_set_engine(1, 1, TEST_SERVING_ROWS, 0);
    test_serving_set_function_model(false, 0);
    test_interleaved();
    test_live_statistics(1);
    test_live_statistics(PIPELINE_DEPTH);
    test_live_busy_wait();
    (void)printf("engine_test: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
