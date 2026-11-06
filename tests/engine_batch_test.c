/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Host-backed BATCH scheduling and all-owner publication. Main alone executes this fixture. */
#include <poll.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "anemone/front_end_seam.h"
#include "engine_internal.h"
#include "batch_fixture.h"
#include "step_driver.h"
#include "step_validation.h"

#define OWNERS 8u
#define DRAFTS 5u
#define ROWS (OWNERS * (DRAFTS + 1u))
#define CONTEXT 256u
#define PROMPT 17u
#define LIMIT 112u
#define EVENTS 256u
#define WAIT_MS 5000
#define SNAPSHOT_TOKENS 32u
#define COST_SESSION_MODE 11u
#define COST_SINGLE_ROWS (DRAFTS + 1u)
#define COST_SHARED_OWNERS 2u
#define COST_GENERATED_TOKENS 32u
#define COST_ORDINARY_PRICE_US 1000u
#define COST_SPECULATIVE_PRICE_US 10000u
#define REQUIRE(condition)                                                 \
	do {                                                               \
		if (!(condition)) {                                        \
			fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, \
				#condition);                               \
			exit(1);                                           \
		}                                                          \
	} while (0)

struct trace {
	struct generation_event events[EVENTS];
	uint32_t count;
};
static uint32_t prompt[OWNERS][PROMPT];
static struct draft_policy_record policy;
static const uint32_t script[] = { TEST_BATCH_END_TOKEN };
static const uint32_t *const scripts[] = { script };

static struct inference_engine *open_engine(uint32_t owners, uint32_t rows,
					    uint32_t drafts, uint32_t context)
{
	struct inference_engine *engine = NULL;
	test_batch_set_batch(true);
	test_batch_set_cone(0);
	test_batch_set_interleaved(0);
	test_batch_set_ring(TEST_BATCH_STATE_POSITIONS, TEST_BATCH_WINDOW);
	test_batch_set_function_model(true, 0);
	test_batch_set_function_end(UINT32_MAX);
	test_batch_set_engine(owners, 1, rows, drafts);
	test_batch_set_device_block(DRAFTS);
	test_batch_set_device_draft(drafts != 0, DRAFTS);
	test_batch_set_draft_policy(&policy, 0);
	REQUIRE(test_batch_open(context, scripts, 1, &engine) == STATUS_OK);
	test_batch_set_step_delay(1000000u);
	return engine;
}

static struct generation_command_record command(uint32_t slot, uint32_t owner,
						float temperature)
{
	return (struct generation_command_record){
		.new_token_ids = prompt[owner],
		.sampler_seed = owner,
		.session_slot = slot,
		.new_token_count = PROMPT,
		.generated_token_limit = LIMIT,
		.sampler_temperature = temperature,
		.sampler_top_p = 1.0f,
		.request_tag = owner + 1u
	};
}

static struct generation_event next_event(struct inference_engine *engine)
{
	int descriptor = -1;
	REQUIRE(engine_event_descriptor(engine, &descriptor) == STATUS_OK);
	for (;;) {
		struct pollfd entry = { descriptor, POLLIN, 0 };
		REQUIRE(poll(&entry, 1, WAIT_MS) == 1);
		bool available = false;
		struct generation_event event = { 0 };
		REQUIRE(engine_next_event(engine, &event, &available) ==
			STATUS_OK);
		if (available)
			return event;
	}
}

static void collect(struct inference_engine *engine, uint32_t owners,
		    struct trace *traces)
{
	uint32_t finals = 0;
	while (finals != owners) {
		const struct generation_event event = next_event(engine);
		REQUIRE(event.request_tag != 0 && event.request_tag <= OWNERS);
		struct trace *const trace = &traces[event.request_tag - 1u];
		REQUIRE(trace->count < EVENTS);
		trace->events[trace->count++] = event;
		finals += event.event_kind != GENERATION_EVENT_TOKEN ? 1u : 0u;
	}
}

static void same(const struct trace *first, const struct trace *second,
		 const char *phase, uint32_t owner)
{
	if (first->count != second->count)
		fprintf(stderr,
			"BATCH identity phase=%s owner=%u reference_count=%u candidate_count=%u\n",
			phase, owner, first->count, second->count);
	const uint32_t count = first->count < second->count ? first->count :
							      second->count;
	for (uint32_t index = 0; index < count; index++) {
		if (memcmp(&first->events[index], &second->events[index],
			   sizeof(first->events[index])) != 0) {
			const struct generation_event
				*const a = &first->events[index],
				       *const b = &second->events[index];
			fprintf(stderr,
				"BATCH identity phase=%s owner=%u event=%u ref(kind=%u token=%u pos=%u status=%u) candidate(kind=%u token=%u pos=%u status=%u)\n",
				phase, owner, index, a->event_kind, a->token_id,
				a->session_token_count, a->status,
				b->event_kind, b->token_id,
				b->session_token_count, b->status);
			break;
		}
	}
	REQUIRE(first->count == second->count);
	REQUIRE(memcmp(first->events, second->events,
		       first->count * sizeof(first->events[0])) == 0);
}

static void driver_admission(void)
{
	struct inference_engine *const engine =
		open_engine(OWNERS, STEP_BATCH_MAXIMUM_ROWS, DRAFTS, CONTEXT);
	struct step_driver *const driver = test_batch_driver();
	struct step_row_record rows[STEP_BATCH_MAXIMUM_ROWS];
	for (uint32_t row = 0; row < STEP_BATCH_MAXIMUM_ROWS; row++)
		rows[row] = (struct step_row_record){ (int32_t)(row % OWNERS),
						      1, row / OWNERS, 0 };
	REQUIRE(step_rows_check(driver, rows, STEP_BATCH_MAXIMUM_ROWS, true) ==
		STATUS_OK);
	driver->round_domain_attached = false;
	REQUIRE(step_rows_check(driver, rows, 1, true) ==
		STATUS_INVALID_ARGUMENT);
	REQUIRE(step_rounds_domain_attach(driver, TEST_BATCH_VOCABULARY,
					  TEST_BATCH_END_TOKEN) == STATUS_OK);
	rows[1].position = 2;
	REQUIRE(step_rows_check(driver, rows, 3, true) ==
		STATUS_INVALID_ARGUMENT);
	rows[1].position = 1;
	for (uint32_t row = 0; row < ROWS; row++)
		rows[row] = (struct step_row_record){
			(int32_t)(row % (DRAFTS + 1u)), 1, row / (DRAFTS + 1u),
			STEP_ROW_LOGITS |
				(row % (DRAFTS + 1u) == 0 ? STEP_ROW_DRAFTS :
							    STEP_ROW_PROPOSED)
		};
	REQUIRE(step_rows_check(driver, rows, ROWS, true) == STATUS_OK);
	rows[ROWS - 1u].flags |= STEP_ROW_DRAFTS;
	REQUIRE(step_rows_check(driver, rows, ROWS, true) ==
		STATUS_INVALID_ARGUMENT);
	rows[ROWS - 1u].flags = STEP_ROW_PROPOSED;
	REQUIRE(step_rows_check(driver, rows, ROWS, true) ==
		STATUS_INVALID_ARGUMENT);
	const struct execution_program *const original = driver->programs[1];
	struct execution_program program = *original;
	struct program_card_record card = original->cards[0];
	program.cards = &card;
	bool unsupported = false;
	REQUIRE(step_program_problem(&program, &program, driver->cards, NULL,
				     &unsupported) == NULL);
	card.round_offset =
		card.result_offset +
		256u; /* legacy88B fits here; a64-row BATCH result does not */
	REQUIRE(step_program_problem(&program, &program, driver->cards, NULL,
				     &unsupported) != NULL);
	card = original->cards[0];
	card.round_offset = card.arena_byte_count - STEP_ROUND_BYTES;
	REQUIRE(step_program_problem(&program, &program, driver->cards, NULL,
				     &unsupported) != NULL);
	/* Different prepared row buckets must have enough walks for cold DRAFTS anchors too. */
	struct execution_program buckets[4] = { *original, *original, *original,
						*original };
	buckets[0].row_count = 12;
	buckets[1].row_count = 24;
	buckets[2].row_count = 64;
	buckets[3].row_count = 64;
	buckets[3].kind = PROGRAM_FULL;
	const struct execution_program *bucket_pointers[] = {
		&buckets[0], &buckets[1], &buckets[2], &buckets[3]
	};
	const struct execution_program **const retained = driver->programs;
	const uint32_t retained_count = driver->program_count;
	driver->programs = bucket_pointers;
	driver->program_count = 4;
	const uint32_t *graphs = NULL;
	for (uint32_t row = 0; row < OWNERS; row++)
		rows[row] = (struct step_row_record){
			0, 1, row, STEP_ROW_LOGITS | STEP_ROW_DRAFTS
		};
	REQUIRE(step_program_for_rows(driver, rows, OWNERS, &graphs) ==
		&buckets[2]);
	REQUIRE(step_program_for_rows(driver, rows, 3, &graphs) == &buckets[1]);
	REQUIRE(step_program_for_rows(driver, rows, 2, &graphs) == &buckets[0]);
	for (uint32_t row = 0; row < STEP_BATCH_MAXIMUM_ROWS; row++)
		rows[row] = (struct step_row_record){ (int32_t)(row % OWNERS),
						      1, row / OWNERS, 0 };
	REQUIRE(step_program_for_rows(driver, rows, STEP_BATCH_MAXIMUM_ROWS,
				      &graphs) == &buckets[3]);
	driver->programs = retained;
	driver->program_count = retained_count;
	struct inference_engine *unchanged = engine;
	REQUIRE(engine_create(driver, &engine->description,
			      STEP_BATCH_MAXIMUM_ROWS + 1u,
			      &unchanged) == STATUS_INVALID_ARGUMENT &&
		unchanged == engine);
	REQUIRE(driver->sequence == 0 && !driver->has_finished &&
		driver->last_rounds.valid_mask == 0);
	test_batch_close();
}

static void paired_runs(float temperature, uint32_t context)
{
	struct trace reference[OWNERS] = { 0 }, batched[OWNERS] = { 0 };
	for (uint32_t slot = 0; slot < OWNERS; slot++) {
		struct inference_engine *const engine =
			open_engine(OWNERS, 1, 0, context);
		const struct generation_command_record current =
			command(0, slot, temperature);
		REQUIRE(engine_submit(engine, &current) == STATUS_OK);
		collect(engine, 1, reference);
		test_batch_close();
	}
	struct inference_engine *const engine =
		open_engine(OWNERS, ROWS, DRAFTS, context);
	REQUIRE(step_pipeline_start(test_batch_driver(), 2) ==
		STATUS_INVALID_ARGUMENT);
	REQUIRE(step_pipeline_start(test_batch_driver(), 1) == STATUS_OK);
	REQUIRE(step_pipeline_start(test_batch_driver(), 1) ==
		STATUS_INVALID_ARGUMENT);
	for (uint32_t slot = 0; slot < OWNERS; slot++) {
		const struct generation_command_record current =
			command(slot, slot, temperature);
		REQUIRE(engine_submit(engine, &current) == STATUS_OK);
	}
	collect(engine, OWNERS, batched);
	for (uint32_t slot = 0; slot < OWNERS; slot++) {
		fprintf(stderr,
			"BATCH paired context=%u temperature=%g owner=%u\n",
			context, (double)temperature, slot);
		same(&reference[slot], &batched[slot], "paired", slot);
		struct draft_statistics statistics = { 0 };
		REQUIRE(engine_draft_statistics(engine, slot, &statistics,
						NULL) == STATUS_OK);
		REQUIRE(statistics.drawing_rounds != 0);
		if (context == CONTEXT)
			REQUIRE(statistics.source[DRAFT_DSPARK].rounds != 0);
	}
	if (context == CONTEXT) {
		REQUIRE(test_batch_maximum_rows() == ROWS);
		REQUIRE(batched[7].events[batched[7].count - 1u].event_kind ==
			GENERATION_EVENT_FINISHED_AT_END_TOKEN);
		REQUIRE(test_batch_device_rejections() == (1u << DRAFTS) - 1u);
	}
	test_batch_close();
}

static void late_corruption(void)
{
	struct trace traces[OWNERS] = { 0 };
	struct inference_engine *const engine =
		open_engine(OWNERS, ROWS, DRAFTS, CONTEXT);
	test_batch_set_batch_fault(OWNERS - 1u);
	for (uint32_t slot = 0; slot < OWNERS; slot++) {
		const struct generation_command_record current =
			command(slot, slot, 0.7f);
		REQUIRE(engine_submit(engine, &current) == STATUS_OK);
	}
	collect(engine, OWNERS, traces);
	REQUIRE(test_batch_maximum_rows() == ROWS);
	const struct step_driver *const driver = test_batch_driver();
	for (uint32_t slot = 0; slot < OWNERS; slot++) {
		const uint32_t anchor = test_batch_fault_anchor(slot);
		REQUIRE(anchor != UINT32_MAX);
		const struct trace *const trace = &traces[slot];
		REQUIRE(trace->count != 0);
		REQUIRE(trace->events[trace->count - 1u].event_kind ==
			GENERATION_EVENT_FAILED);
		/* no token past the anchor's position: the anchor row's own token (position anchor, the session
		 * holding anchor + 1 tokens) came from the step before, which landed, and streams; the cards ran
		 * the positions before it alone (the driver's count) */
		for (uint32_t event = 0; event < trace->count; event++)
			if (trace->events[event].event_kind ==
			    GENERATION_EVENT_TOKEN)
				REQUIRE(trace->events[event]
						.session_token_count <=
					anchor + 1u);
		REQUIRE(engine->slots[slot].reach_count ==
			test_batch_fault_reach(slot));
		REQUIRE(driver->token_counts[slot] <= anchor);
	}
	test_batch_close();
}

static struct generation_event session(struct inference_engine *engine,
				       const struct session_command *operation)
{
	REQUIRE(engine_session(engine, operation) == STATUS_OK);
	const struct generation_event event = next_event(engine);
	REQUIRE(event.request_tag == operation->request_tag);
	REQUIRE(event.event_kind == GENERATION_EVENT_FINISHED_AT_LIMIT);
	return event;
}

static void snapshots_and_cancel(void)
{
	struct trace traces[OWNERS] = { 0 };
	struct inference_engine *engine =
		open_engine(OWNERS, ROWS, DRAFTS, CONTEXT);
	for (uint32_t slot = 0; slot < OWNERS; slot++) {
		const struct generation_command_record current =
			command(slot, slot, 0.7f);
		REQUIRE(engine_submit(engine, &current) == STATUS_OK);
	}
	collect(engine, OWNERS, traces);
	uint32_t prefix[SNAPSHOT_TOKENS];
	memcpy(prefix, prompt[3], sizeof(prompt[3]));
	for (uint32_t token = PROMPT; token < SNAPSHOT_TOKENS; token++)
		prefix[token] = traces[3].events[token - PROMPT].token_id;
	char path[] = "/tmp/anemone_batch_snapshot_XXXXXX";
	const int descriptor = mkstemp(path);
	REQUIRE(descriptor >= 0 && unlink(path) == 0);
	struct session_state state = { 0 };
	REQUIRE(engine_session_state(engine, 3, &state) == STATUS_OK);
	const struct session_command save = { .descriptor = descriptor,
					      .session_slot = 3,
					      .token_count = SNAPSHOT_TOKENS,
					      .operation = SESSION_SAVE,
					      .request_tag = 99 };
	(void)session(engine, &save);
	unsigned char saved[SNAPSHOT_TOKENS];
	memcpy(saved, test_batch_slot_state(3), sizeof(saved));
	REQUIRE(lseek(descriptor, 0, SEEK_SET) == 0);
	const struct session_command load = { .token_ids = prefix,
					      .descriptor = descriptor,
					      .session_slot = 4,
					      .token_count = SNAPSHOT_TOKENS,
					      .operation = SESSION_LOAD,
					      .request_tag = 99,
					      .reach_count = state.reach_count,
					      .decoder_from =
						      state.decoder_from };
	(void)session(engine, &load);
	REQUIRE(memcmp(saved, test_batch_slot_state(4), sizeof(saved)) == 0);
	REQUIRE(close(descriptor) == 0);
	struct trace continued[OWNERS] = { 0 };
	const uint32_t new_token = 11;
	for (uint32_t index = 0; index < 2; index++) {
		const struct generation_command_record next = {
			.new_token_ids = &new_token,
			.sampler_seed = 19,
			.session_slot = index + 3u,
			.kept_token_count = SNAPSHOT_TOKENS,
			.new_token_count = 1,
			.generated_token_limit = 40,
			.sampler_temperature = 0.7f,
			.sampler_top_p = 1.0f,
			.request_tag = index + 1u
		};
		REQUIRE(engine_submit(engine, &next) == STATUS_OK);
	}
	collect(engine, 2, continued);
	for (uint32_t index = 0; index < continued[1].count; index++)
		continued[1].events[index].request_tag = 1;
	same(&continued[0], &continued[1], "snapshot", 4);
	test_batch_close();

	memset(traces, 0, sizeof(traces));
	engine = open_engine(2, 12, DRAFTS, CONTEXT);
	for (uint32_t slot = 0; slot < 2; slot++) {
		const struct generation_command_record current =
			command(slot, slot, 0.7f);
		REQUIRE(engine_submit(engine, &current) == STATUS_OK);
	}
	bool cancelled = false;
	uint32_t finals = 0;
	while (finals != 2) {
		const struct generation_event event = next_event(engine);
		REQUIRE(event.request_tag == 1 || event.request_tag == 2);
		struct trace *const trace = &traces[event.request_tag - 1u];
		REQUIRE(trace->count < EVENTS);
		trace->events[trace->count++] = event;
		if (!cancelled && event.request_tag == 1 &&
		    event.event_kind == GENERATION_EVENT_TOKEN) {
			REQUIRE(engine_cancel(engine, 0) == STATUS_OK);
			cancelled = true;
		}
		finals += event.event_kind != GENERATION_EVENT_TOKEN ? 1u : 0u;
	}
	REQUIRE(cancelled &&
		traces[0].events[traces[0].count - 1u].event_kind ==
			GENERATION_EVENT_CANCELLED);
	struct trace survivor = traces[1];
	test_batch_close();
	memset(traces, 0, sizeof(traces));
	engine = open_engine(1, 1, 0, CONTEXT);
	const struct generation_command_record current = command(0, 1, 0.7f);
	REQUIRE(engine_submit(engine, &current) == STATUS_OK);
	collect(engine, 1, traces);
	same(&survivor, &traces[1], "cancel-survivor", 1);
	test_batch_close();
}

static uint64_t personal_cost_observations(const struct command_cursor *cursor)
{
	uint64_t observations = 0;
	for (uint32_t tails = 0; tails <= 1u; tails++)
		for (uint32_t depth = 0; depth < DRAFT_DEPTH_COUNTS; depth++) {
			REQUIRE(cursor->session_costs[tails][depth]
						.observations == 0 ||
				cursor->session_costs[tails][depth]
						.estimate_us != 0);
			observations +=
				cursor->session_costs[tails][depth].observations;
		}
	return observations;
}

static uint64_t shared_cost_observations(const struct inference_engine *engine)
{
	uint64_t observations = 0;
	for (uint32_t rows = 0; rows <= SERVING_BATCH_COST_ROWS; rows++)
		for (uint32_t tails = 0; tails <= SERVING_SESSION_CAPACITY;
		     tails++) {
			REQUIRE(engine->batch_shape_costs[rows][tails]
						.observations == 0 ||
				engine->batch_shape_costs[rows][tails]
						.estimate_us != 0);
			observations += engine->batch_shape_costs[rows][tails]
						.observations;
		}
	return observations;
}

/* These are execution samples from the real worker/driver boundary, not synthetic cost observations. */
static void measured_cost_integration(void)
{
	static const uint32_t step_costs[SERVING_BATCH_COSTS] = {
		COST_ORDINARY_PRICE_US, COST_SPECULATIVE_PRICE_US
	};
	static struct draft_cost_cell retained[SERVING_BATCH_COST_ROWS + 1u]
					      [SERVING_SESSION_CAPACITY + 1u];
	const struct draft_policy_record saved_policy = policy;
	policy.depth_policy = DRAFT_DEPTH_RATE;
	test_batch_set_draft_costs(COST_SESSION_MODE, step_costs);
	struct inference_engine *engine =
		open_engine(1, COST_SINGLE_ROWS, DRAFTS, CONTEXT);
	struct generation_command_record current = command(0, 0, 0.0f);
	current.sampler_seed =
		DRAFTS; /* this fixture's all-correct device draft */
	current.generated_token_limit = 1;
	struct trace traces[OWNERS] = { 0 };
	REQUIRE(engine_submit(engine, &current) == STATUS_OK);
	collect(engine, 1, traces);
	REQUIRE(traces[0].count == current.generated_token_limit + 1u &&
		traces[0].events[traces[0].count - 1u].status == STATUS_OK &&
		traces[0].events[traces[0].count - 1u].event_kind ==
			GENERATION_EVENT_FINISHED_AT_LIMIT);
	REQUIRE(shared_cost_observations(engine) == 0 &&
		personal_cost_observations(&engine->cursors[0]) == 0);
	memset(traces, 0, sizeof(traces));
	current.generated_token_limit = COST_GENERATED_TOKENS;
	REQUIRE(engine_submit(engine, &current) == STATUS_OK);
	collect(engine, 1, traces);
	const uint64_t personal =
		personal_cost_observations(&engine->cursors[0]);
	REQUIRE(traces[0].count == current.generated_token_limit + 1u &&
		traces[0].events[traces[0].count - 1u].status == STATUS_OK &&
		traces[0].events[traces[0].count - 1u].event_kind ==
			GENERATION_EVENT_FINISHED_AT_LIMIT &&
		personal != 0 && personal == shared_cost_observations(engine) &&
		engine->cursors[0].statistics.source[DRAFT_DSPARK].rounds != 0);
	memcpy(retained, engine->batch_shape_costs, sizeof(retained));
	memset(traces, 0, sizeof(traces));
	current.generated_token_limit = 1;
	REQUIRE(engine_submit(engine, &current) == STATUS_OK);
	collect(engine, 1, traces);
	REQUIRE(traces[0].count == current.generated_token_limit + 1u &&
		traces[0].events[traces[0].count - 1u].status == STATUS_OK &&
		traces[0].events[traces[0].count - 1u].event_kind ==
			GENERATION_EVENT_FINISHED_AT_LIMIT &&
		personal_cost_observations(&engine->cursors[0]) == 0 &&
		memcmp(retained, engine->batch_shape_costs, sizeof(retained)) ==
			0);
	test_batch_close();

	/* Without proposals, row counts one/two identify singleton/shared decode independently of scheduling. */
	engine =
		open_engine(COST_SHARED_OWNERS, COST_SHARED_OWNERS, 0, CONTEXT);
	struct generation_command_record first = command(0, 0, 0.0f);
	struct generation_command_record second = command(1, 1, 0.0f);
	first.generated_token_limit = COST_GENERATED_TOKENS;
	second.generated_token_limit = COST_GENERATED_TOKENS;
	test_batch_hold_first_step(true);
	const result_status first_status = engine_submit(engine, &first);
	const bool held = first_status == STATUS_OK &&
			  test_batch_wait_first_step();
	const result_status second_status =
		held ? engine_submit(engine, &second) : STATUS_INVALID_ARGUMENT;
	test_batch_hold_first_step(false);
	if (!held || second_status != STATUS_OK) {
		test_batch_close();
		REQUIRE(held && second_status == STATUS_OK);
	}
	memset(traces, 0, sizeof(traces));
	collect(engine, COST_SHARED_OWNERS, traces);
	uint64_t personal_total = 0;
	for (uint32_t owner = 0; owner < COST_SHARED_OWNERS; owner++) {
		REQUIRE(traces[owner].count == COST_GENERATED_TOKENS + 1u &&
			traces[owner].events[traces[owner].count - 1u].status ==
				STATUS_OK &&
			traces[owner].events[traces[owner].count - 1u]
					.event_kind ==
				GENERATION_EVENT_FINISHED_AT_LIMIT);
		personal_total +=
			personal_cost_observations(&engine->cursors[owner]);
	}
	const uint64_t singletons =
		engine->batch_shape_costs[1][0].observations;
	const uint64_t shared =
		engine->batch_shape_costs[COST_SHARED_OWNERS][0].observations;
	REQUIRE(shared != 0 && personal_total == singletons &&
		shared_cost_observations(engine) == singletons + shared &&
		singletons + COST_SHARED_OWNERS * shared <=
			COST_SHARED_OWNERS * (COST_GENERATED_TOKENS - 1u));
	test_batch_close();
	test_batch_set_draft_costs(0, NULL);
	policy = saved_policy;
}

int main(void)
{
	draft_policy_default(&policy);
	policy.depth_policy = DRAFT_DEPTH_FIXED;
	for (uint32_t slot = 0; slot < OWNERS; slot++)
		for (uint32_t token = 0; token < PROMPT; token++)
			prompt[slot][token] = (slot * 31u + token * 13u) % 256u;
	driver_admission();
	paired_runs(0.0f, CONTEXT);
	paired_runs(0.7f, CONTEXT);
	paired_runs(0.7f, 48);
	late_corruption();
	snapshots_and_cancel();
	measured_cost_integration();
	puts("BATCH host scheduling: eight K5 owners, counter keys, EOS, rejection, late corruption, snapshot and cancel");
	return 0;
}
