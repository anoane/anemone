/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Stateful host model: a decoding session beside WIDE/FULL chunks, in each prompt order, with both decode drivers.
 * The actual engine/driver must keep solo token and logit bits and give decoders a round between bounded chunks. */
#include <poll.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "anemone/front_end_seam.h"
#include "anemone/backend_records.h"
#include "anemone/model_layout.h"
#include "anemone/step.h"
#include "host_memory_backend.h"
#include "prepare_common.h"

#define SESSIONS 5u
#define SHORT_OWNERS (SESSIONS - 1u)
#define LONG_SLOT SHORT_OWNERS
#define SWITCH_VALUES 2u
#define BURST_ROUNDS 4u
#define PIPELINE_DEPTH 2u
#define CONTEXT 8192u
#define CHUNK_ROWS 512u
#define SLICE_ROWS MODEL_FULL_CONE_LEAST_ROWS
#define THIN_ROWS LINEAR_MAXIMUM_ROWS
#define CHUNK_SIZES 3u
#define PROGRAMS (1u + 2u * CHUNK_SIZES)
#define SEGMENTS 3u
#define STEP_OFFSET 0u
#define SESSION_OFFSET 12288u
#define RESULT_OFFSET 12544u
#define VIEWS_OFFSET 16384u
#define STATE_OFFSET 32768u
#define ROW_VALUES_OFFSET (STATE_OFFSET + SESSIONS * CONTEXT * sizeof(uint32_t))
#define ARENA_BYTES (ROW_VALUES_OFFSET + CHUNK_ROWS * sizeof(uint32_t))
#define VIEW_BYTES (STEP_BLOCK_BYTES_VALUE + THIN_ROWS * STEP_ROW_BYTES_VALUE)
#define VIEWS (CHUNK_ROWS / THIN_ROWS)
#define VOCABULARY 257u
#define END_TOKEN (VOCABULARY - 1u)
#define LONG_PROMPT 6000u
#define SHORT_PROMPT 5u
#define SHORT_LIMIT \
	600u /* exceeds the event queue: the decoder cannot finish before the long submit */
#define LONG_LIMIT 8u
#define WARM_TOKENS 8u
#define EVENTS (SHORT_LIMIT + 1u)
#define WAIT_NS 1000000000ull
#define WAIT_MS 5000
#define HASH_MULTIPLIER 16777619u
#define LOGIT_MASK 0x007fffffu
#define LOGIT_BASE 0x3f800000u
#define DELAY_NS 100000l
#define PREFILL_MARGIN 127u
#define REQUIRE(condition)                                                 \
	do {                                                               \
		if (!(condition)) {                                        \
			fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, \
				#condition);                               \
			exit(1);                                           \
		}                                                          \
	} while (0)

enum operation { FRONT, TARGET, HEAD };
struct trace {
	struct generation_event events[EVENTS];
	uint32_t count;
};
struct fixture {
	struct device_card *card;
	struct step_driver *driver;
	struct inference_engine *engine;
	struct execution_program programs[PROGRAMS];
	struct program_card_record cards[PROGRAMS];
	struct kernel_launch launches[PROGRAMS][SEGMENTS];
	struct segment_record segments[PROGRAMS][SEGMENTS];
	struct step_view views[VIEWS];
	const struct execution_program *pointers[PROGRAMS];
	uint32_t short_draws[SESSIONS], during_prefill[SESSIONS];
	uint32_t waiting_chunks[SESSIONS], most_waiting_chunks[SESSIONS];
	uint32_t long_slot;
	uint32_t pending_chunks, most_chunks, largest_chunk;
	uint32_t rounds_since_chunk, fewest_chunk_rounds;
	bool had_chunk;
	bool long_started, long_drawn;
};
static uint32_t prompt[LONG_PROMPT];

static bool model_run(const struct kernel_launch *launch, uint32_t card,
		      unsigned char *arena, uint64_t bytes, void *context)
{
	struct fixture *const fixture = context;
	struct step_block block;
	struct fill_record record;
	memcpy(&record, launch->argument, sizeof(record));
	memcpy(&block, arena + STEP_OFFSET, sizeof(block));
	if (card != 0 || bytes != ARENA_BYTES || block.row_count == 0 ||
	    block.row_count > CHUNK_ROWS)
		return false;
	const struct timespec delay = { 0, DELAY_NS };
	if (nanosleep(&delay, NULL) != 0)
		return false;
	if (record.value == TARGET)
		return true;
	const struct step_result header = { block.sequence, block.row_count,
					    0 };
	if (record.value == HEAD)
		memcpy(arena + RESULT_OFFSET, &header, sizeof(header));
	bool draws_short = false;
	for (uint32_t row = 0; row < block.row_count; row++) {
		struct step_row_record own;
		uint32_t value = 0;
		memcpy(&own,
		       arena + STEP_OFFSET + STEP_BLOCK_BYTES_VALUE +
			       row * STEP_ROW_BYTES_VALUE,
		       sizeof(own));
		if (own.session_slot >= SESSIONS || own.position < 0 ||
		    (uint32_t)own.position >= CONTEXT)
			return false;
		if (record.value == FRONT) {
			const uint64_t state =
				STATE_OFFSET +
				own.session_slot * CONTEXT * sizeof(uint32_t);
			if (own.position != 0)
				memcpy(&value,
				       arena + state +
					       ((uint32_t)own.position - 1u) *
						       sizeof(value),
				       sizeof(value));
			value = (value ^ (uint32_t)own.token_id) *
				HASH_MULTIPLIER;
			memcpy(arena + state +
				       (uint32_t)own.position * sizeof(value),
			       &value, sizeof(value));
			memcpy(arena + ROW_VALUES_OFFSET + row * sizeof(value),
			       &value, sizeof(value));
			fixture->long_started = fixture->long_started ||
						own.session_slot ==
							fixture->long_slot;
			if (own.session_slot != fixture->long_slot &&
			    (own.flags & STEP_ROW_LOGITS) != 0) {
				/* Decode rows must use their own prepared program. */
				if (launch->grid[0] != THIN_ROWS)
					return false;
				fixture->short_draws[own.session_slot]++;
				draws_short = true;
				fixture->waiting_chunks[own.session_slot] = 0;
				if (fixture->long_started &&
				    !fixture->long_drawn)
					fixture->during_prefill
						[own.session_slot]++;
				fixture->pending_chunks = 0;
			}
			fixture->long_drawn =
				fixture->long_drawn ||
				(own.session_slot == fixture->long_slot &&
				 (own.flags & STEP_ROW_LOGITS) != 0);
		} else {
			memcpy(&value,
			       arena + ROW_VALUES_OFFSET + row * sizeof(value),
			       sizeof(value));
			const bool draws = (own.flags & STEP_ROW_LOGITS) != 0;
			const uint32_t bits = LOGIT_BASE | (value & LOGIT_MASK);
			struct step_row_result result = {
				draws ? (int32_t)(value % END_TOKEN) :
					STEP_NO_TOKEN,
				0.0f
			};
			if (draws)
				memcpy(&result.logit, &bits, sizeof(bits));
			memcpy(arena + RESULT_OFFSET + STEP_RESULT_BYTES +
				       row * sizeof(result),
			       &result, sizeof(result));
		}
	}
	bool active_decoder = false;
	for (uint32_t slot = 0; slot < SESSIONS; slot++)
		active_decoder = active_decoder ||
				 (slot != fixture->long_slot &&
				  fixture->short_draws[slot] != 0 &&
				  fixture->short_draws[slot] < SHORT_LIMIT);
	if (draws_short && fixture->long_started && !fixture->long_drawn)
		fixture->rounds_since_chunk++;
	if (record.value == FRONT && launch->grid[0] > THIN_ROWS &&
	    fixture->long_started && active_decoder) {
		for (uint32_t slot = 0; slot < SESSIONS && !fixture->long_drawn;
		     slot++) {
			if (slot == fixture->long_slot ||
			    fixture->short_draws[slot] == 0 ||
			    fixture->short_draws[slot] == SHORT_LIMIT)
				continue;
			fixture->waiting_chunks[slot]++;
			if (fixture->waiting_chunks[slot] >
			    fixture->most_waiting_chunks[slot])
				fixture->most_waiting_chunks[slot] =
					fixture->waiting_chunks[slot];
		}
		if (fixture->had_chunk && !fixture->long_drawn &&
		    fixture->rounds_since_chunk < fixture->fewest_chunk_rounds)
			fixture->fewest_chunk_rounds =
				fixture->rounds_since_chunk;
		fixture->had_chunk = true;
		fixture->rounds_since_chunk = 0;
		fixture->pending_chunks++;
		fixture->most_chunks = fixture->pending_chunks >
						       fixture->most_chunks ?
					       fixture->pending_chunks :
					       fixture->most_chunks;
		fixture->largest_chunk =
			block.row_count > fixture->largest_chunk ?
				block.row_count :
				fixture->largest_chunk;
	}
	return true;
}

static void open_fixture(struct fixture *fixture, bool batch, uint32_t order,
			 uint32_t first, uint32_t cap, uint32_t depth,
			 uint32_t rounds, uint32_t waiters, bool prompt_chunks)
{
	memset(fixture, 0, sizeof(*fixture));
	fixture->long_slot = LONG_SLOT;
	fixture->fewest_chunk_rounds = UINT32_MAX;
	test_host_memory_backend_reset(1, ARENA_BYTES);
	test_host_memory_backend_set_arithmetic_class(0,
						      ARITHMETIC_CLASS_SM_80);
	REQUIRE(test_host_memory_backend.card_open(
			0, ARENA_BYTES, &fixture->card) == STATUS_OK);
	uint32_t kernel = 0;
	REQUIRE(test_host_memory_backend.kernel_prepare(
			fixture->card, KERNEL_FILE_FILL, KERNEL_FILL_WORDS,
			FILL_RECORD_BYTES, &kernel) == STATUS_OK);
	REQUIRE(test_host_memory_backend_set_stand_in(
			KERNEL_FILE_FILL, KERNEL_FILL_WORDS, model_run,
			fixture) == STATUS_OK);
	for (uint32_t view = 0; view < VIEWS; view++)
		fixture->views[view] =
			(struct step_view){ VIEWS_OFFSET + view * VIEW_BYTES,
					    view * THIN_ROWS, THIN_ROWS };
	const struct model_limits limits = { .context_token_limit = CONTEXT,
					     .row_capacity = CHUNK_ROWS,
					     .session_capacity = SESSIONS,
					     .state_width = STATE_WIDTH_16 };
	static const uint32_t sizes[CHUNK_SIZES] = { CHUNK_ROWS, SLICE_ROWS,
						     MODEL_WIDE_LEAST_ROWS };
	for (uint32_t index = 0; index < PROGRAMS; index++) {
		const enum program_kind kind =
			index == 0 ? batch ? PROGRAM_BATCH : PROGRAM_THIN :
			index <= CHUNK_SIZES ? PROGRAM_WIDE :
					       PROGRAM_FULL;
		const uint32_t rows = index == 0 ?
					      THIN_ROWS :
					      sizes[(index - 1u) % CHUNK_SIZES];
		const uint32_t segments = index == 0	       ? SEGMENTS :
					  kind == PROGRAM_WIDE ? 1u :
								 HEAD;
		for (uint32_t segment = 0; segment < segments; segment++) {
			const struct fill_record record = { .value = segment };
			prepare_launch_fill(&fixture->launches[index][segment],
					    kernel, rows, 1, 1, 0, &record,
					    sizeof(record));
			fixture->segments[index][segment] =
				(struct segment_record){
					.first_launch = segment,
					.launch_count = 1,
					.layer = segment == HEAD ?
							 SEGMENT_NO_LAYER :
							 segment
				};
		}
		fixture->cards[index] = (struct program_card_record){
			.launches = fixture->launches[index],
			.arena_byte_count = ARENA_BYTES,
			.step_offset = STEP_OFFSET,
			.session_offset = SESSION_OFFSET,
			.result_offset = RESULT_OFFSET,
			.logits_offset = PROGRAM_NOWHERE,
			.streams_offset = PROGRAM_NOWHERE,
			.carried_pre_offset = PROGRAM_NOWHERE,
			.engram_rows_offset = PROGRAM_NOWHERE,
			.round_offset = PROGRAM_NOWHERE,
			.launch_count = segments,
			.views = index == 0 ? NULL : fixture->views,
			.view_count = index == 0 ? 0u : rows / THIN_ROWS
		};
		fixture->programs[index] = (struct execution_program){
			.cards = &fixture->cards[index],
			.segments = fixture->segments[index],
			.limits = limits,
			.card_count = 1,
			.segment_count = segments,
			.row_count = rows,
			.kind = kind,
			.front_segment_count = 1,
			.cone_rows = 1,
			.draft_segment_first = segments
		};
		fixture->pointers[index] = &fixture->programs[index];
	}
	REQUIRE(step_driver_create(&test_host_memory_backend, &fixture->card,
				   fixture->pointers, PROGRAMS, NULL, WAIT_NS,
				   &fixture->driver) == STATUS_OK);
	if (depth > 1u)
		REQUIRE(step_pipeline_start(fixture->driver, depth) ==
			STATUS_OK);
	const struct engine_description_record description = {
		.session_slot_count = SESSIONS,
		.context_token_limit = CONTEXT,
		.vocabulary_size = VOCABULARY,
		.end_token_id = END_TOKEN,
		.front_steps = 1,
		.wide_rows = CHUNK_ROWS,
		.chunks_in_flight = 1,
		.prefill_margin = PREFILL_MARGIN,
		.prefill_first = first,
		.prefill_prompt_chunks = prompt_chunks ? 1u : 0u,
		.prefill_order = order,
		.prefill_decode_rows = cap,
		.prefill_decode_rounds = rounds,
		.prefill_decode_waiters = waiters
	};
	REQUIRE(engine_create(fixture->driver, &description, THIN_ROWS,
			      &fixture->engine) == STATUS_OK);
}

static void close_fixture(struct fixture *fixture)
{
	engine_destroy(fixture->engine);
	step_driver_destroy(fixture->driver);
	test_host_memory_backend.card_close(fixture->card, WAIT_NS);
}

static void submit(struct fixture *fixture, uint32_t slot)
{
	const struct generation_command_record command = {
		.new_token_ids = prompt,
		.session_slot = slot,
		.new_token_count = slot == fixture->long_slot ? LONG_PROMPT :
								SHORT_PROMPT,
		.generated_token_limit =
			slot == fixture->long_slot ? LONG_LIMIT : SHORT_LIMIT,
		.sampler_top_p = 1.0f,
		.request_tag = slot + 1u
	};
	REQUIRE(engine_submit(fixture->engine, &command) == STATUS_OK);
}

static bool take(struct fixture *fixture, struct trace traces[SESSIONS])
{
	int descriptor = -1;
	REQUIRE(engine_event_descriptor(fixture->engine, &descriptor) ==
		STATUS_OK);
	for (;;) {
		struct pollfd poll_event = { descriptor, POLLIN, 0 };
		REQUIRE(poll(&poll_event, 1, WAIT_MS) == 1);
		struct generation_event event;
		bool available = false;
		REQUIRE(engine_next_event(fixture->engine, &event,
					  &available) == STATUS_OK);
		if (!available)
			continue;
		REQUIRE(event.request_tag >= 1u &&
			event.request_tag <= SESSIONS &&
			event.status == STATUS_OK);
		struct trace *const trace = &traces[event.request_tag - 1u];
		REQUIRE(trace->count < EVENTS);
		trace->events[trace->count++] = event;
		return event.event_kind != GENERATION_EVENT_TOKEN;
	}
}

static void mixed_run(const struct trace solo[SESSIONS], bool batch,
		      uint32_t depth, uint32_t order, uint32_t first,
		      uint32_t cap, uint32_t rounds, uint32_t short_owners,
		      uint32_t waiters)
{
	struct fixture fixture;
	struct trace traces[SESSIONS] = { 0 };
	open_fixture(&fixture, batch, order, first, cap, depth, rounds, waiters,
		     false);
	submit(&fixture, 0);
	while (traces[0].count < WARM_TOKENS)
		REQUIRE(!take(&fixture, traces));
	for (uint32_t slot = 1; slot < short_owners; slot++)
		submit(&fixture, slot);
	submit(&fixture, LONG_SLOT);
	uint32_t finals = 0;
	while (finals < short_owners + 1u)
		finals += take(&fixture, traces) ? 1u : 0u;
	close_fixture(&fixture);
	for (uint32_t slot = 0; slot < SESSIONS; slot++)
		if (slot < short_owners || slot == LONG_SLOT)
			REQUIRE(solo[slot].count == traces[slot].count &&
				memcmp(solo[slot].events, traces[slot].events,
				       solo[slot].count *
					       sizeof(solo[slot].events[0])) ==
					0);
	printf("%s depth=%u order=%u first=%u cap=%u rounds=%u shorts=%u"
	       " waiters=%u largest=%u consecutive=%u min_rounds=%u\n",
	       batch ? "BATCH" : "THIN", depth, order, first, cap, rounds,
	       short_owners, waiters, fixture.largest_chunk,
	       fixture.most_chunks, fixture.fewest_chunk_rounds);
	if (cap != 0) {
		REQUIRE((fixture.largest_chunk == SLICE_ROWS ||
			 (waiters != 0 &&
			  fixture.largest_chunk == CHUNK_ROWS)) &&
			fixture.most_chunks == 1u);
		if (short_owners == 1u)
			REQUIRE(fixture.fewest_chunk_rounds != UINT32_MAX &&
				fixture.fewest_chunk_rounds >= rounds);
		for (uint32_t slot = 0; slot < short_owners; slot++) {
			printf("short%u during prefill=%u max_chunk_gap=%u\n",
			       slot, fixture.during_prefill[slot],
			       fixture.most_waiting_chunks[slot]);
			REQUIRE(fixture.during_prefill[slot] != 0 &&
				fixture.most_waiting_chunks[slot] <=
					SHORT_OWNERS + PIPELINE_DEPTH);
		}
	} else {
		/* The old ordered THIN path can starve the prompt until decode finishes. */
		REQUIRE(fixture.largest_chunk == 0 ||
			fixture.largest_chunk == CHUNK_ROWS);
		if (first != 0)
			REQUIRE(fixture.most_chunks > 1u);
	}
}

/* Only two owners are ready. Landing a prompt flight must not consume the emitting decoder's next turn.
 * Swapping their slots and shifting the warm arrival by one token exercises both round-robin orderings/phases. */
static void guarded_decoder_run(bool batch, bool swapped, uint32_t warm_tokens)
{
	const uint32_t long_slot = swapped ? 0u : LONG_SLOT;
	const uint32_t decoder_slot = swapped ? LONG_SLOT : 0u;
	struct fixture fixture;
	struct trace solo[SESSIONS] = { 0 };
	const uint32_t owners[] = { decoder_slot, long_slot };
	for (uint32_t owner = 0; owner < sizeof(owners) / sizeof(owners[0]);
	     owner++) {
		open_fixture(&fixture, batch, 0, 0, 0, 1, 1, 0, false);
		fixture.long_slot = long_slot;
		submit(&fixture, owners[owner]);
		while (!take(&fixture, solo)) {
		}
		close_fixture(&fixture);
	}
	struct trace traces[SESSIONS] = { 0 };
	open_fixture(&fixture, batch, 0, 0, 0, 1, 1, 0, true);
	fixture.long_slot = long_slot;
	submit(&fixture, decoder_slot);
	while (traces[decoder_slot].count < warm_tokens)
		REQUIRE(!take(&fixture, traces));
	submit(&fixture, long_slot);
	uint32_t finals = 0;
	while (finals < sizeof(owners) / sizeof(owners[0]))
		finals += take(&fixture, traces) ? 1u : 0u;
	close_fixture(&fixture);
	for (uint32_t owner = 0; owner < sizeof(owners) / sizeof(owners[0]);
	     owner++) {
		const uint32_t slot = owners[owner];
		REQUIRE(solo[slot].count == traces[slot].count &&
			memcmp(solo[slot].events, traces[slot].events,
			       solo[slot].count *
				       sizeof(solo[slot].events[0])) == 0);
	}
	REQUIRE(fixture.long_started && fixture.long_drawn &&
		fixture.had_chunk);
	REQUIRE(fixture.largest_chunk == CHUNK_ROWS);
	REQUIRE(fixture.during_prefill[decoder_slot] != 0);
	REQUIRE(fixture.most_waiting_chunks[decoder_slot] == 1u);
	REQUIRE(fixture.most_chunks == 1u &&
		fixture.fewest_chunk_rounds != UINT32_MAX &&
		fixture.fewest_chunk_rounds >= 1u);
	printf("guarded %s first0/order0/cap0 depth1 decoder%u prompt%u warm%u during=%u chunk_gap=%u min_rounds=%u\n",
	       batch ? "BATCH" : "THIN", decoder_slot, long_slot, warm_tokens,
	       fixture.during_prefill[decoder_slot],
	       fixture.most_waiting_chunks[decoder_slot],
	       fixture.fewest_chunk_rounds);
}

int main(void)
{
	for (uint32_t row = 0; row < LONG_PROMPT; row++)
		prompt[row] = row % END_TOKEN;
	struct fixture fixture;
	struct trace solo[SESSIONS] = { 0 };
	for (uint32_t slot = 0; slot < SESSIONS; slot++) {
		open_fixture(&fixture, false, 0, 0, 0, 1, 1, 0, false);
		submit(&fixture, slot);
		while (!take(&fixture, solo)) {
		}
		close_fixture(&fixture);
	}
	for (uint32_t batch = 0; batch < SWITCH_VALUES; batch++)
		for (uint32_t depth = 1;
		     depth <= (batch != 0 ? 1u : PIPELINE_DEPTH); depth++)
			for (uint32_t order = 0;
			     order <= PREFILL_ORDER_SHORTEST; order++)
				for (uint32_t first = 0; first < SWITCH_VALUES;
				     first++)
					for (uint32_t capped = 0;
					     capped < SWITCH_VALUES; capped++)
						for (uint32_t burst = 0;
						     burst <
						     (capped != 0 ?
							      SWITCH_VALUES :
							      1u);
						     burst++)
							mixed_run(
								solo,
								batch != 0,
								depth, order,
								first,
								capped != 0 ?
									SLICE_ROWS :
									0u,
								burst != 0 ?
									BURST_ROUNDS :
									1u,
								SHORT_OWNERS,
								0);
	for (uint32_t order = 0; order <= PREFILL_ORDER_SHORTEST; order++)
		for (uint32_t first = 0; first < SWITCH_VALUES; first++)
			mixed_run(solo, false, PIPELINE_DEPTH, order, first,
				  SLICE_ROWS, BURST_ROUNDS, 1, 0);
	for (uint32_t batch = 0; batch < SWITCH_VALUES; batch++)
		for (uint32_t order = 0; order <= PREFILL_ORDER_SHORTEST;
		     order++)
			for (uint32_t first = 0; first < SWITCH_VALUES;
			     first++) {
				mixed_run(solo, batch != 0,
					  batch != 0 ? 1u : PIPELINE_DEPTH,
					  order, first, SLICE_ROWS,
					  BURST_ROUNDS, SHORT_OWNERS, 1);
				if (batch == 0)
					mixed_run(solo, false, PIPELINE_DEPTH,
						  order, first, SLICE_ROWS,
						  BURST_ROUNDS, 1, 1);
			}
	for (uint32_t batch = 0; batch < SWITCH_VALUES; batch++)
		for (uint32_t swapped = 0; swapped < SWITCH_VALUES; swapped++)
			for (uint32_t phase = 0; phase < SWITCH_VALUES; phase++)
				guarded_decoder_run(batch != 0, swapped != 0,
						    WARM_TOKENS + phase);
	puts("prefill_interleave_test: solo token/logit events preserved and decoders progress in all orders");
	return 0;
}
