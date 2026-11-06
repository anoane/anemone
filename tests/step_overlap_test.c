/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Prefix/tail ownership, literal state preservation and real host-worker overlap, without a GPU. */
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "anemone/backend_records.h"
#include "anemone/step.h"
#include "host_memory_backend.h"
#include "prepare_common.h"

#define OVERLAP_PREFIX_CARDS 4u
#define OVERLAP_CARDS 5u
#define OVERLAP_TAIL_CARD OVERLAP_PREFIX_CARDS
#define OVERLAP_PREFIX_MASK ((1u << OVERLAP_PREFIX_CARDS) - 1u)
#define OVERLAP_SESSIONS 3u
#define OVERLAP_WAVES 3u
#define OVERLAP_FLIGHTS (OVERLAP_SESSIONS * OVERLAP_WAVES)
#define OVERLAP_ARENA_BYTES 8192u
#define OVERLAP_RESULT_OFFSET 512u
#define OVERLAP_SETTINGS_OFFSET 1024u
#define OVERLAP_CARRY_OFFSET 2048u
#define OVERLAP_EMBEDDING_OFFSET 3072u
#define OVERLAP_EMBEDDING_ROLE 1u
#define OVERLAP_EMBEDDING_COPIES (OVERLAP_PREFIX_CARDS - 1u)
#define OVERLAP_STATE_OFFSET 4096u
#define OVERLAP_CONTEXT 16u
#define OVERLAP_FIRST_TOKEN 11
#define OVERLAP_TOKEN_STRIDE 7
#define OVERLAP_WAVE_TOKEN_STRIDE 31
#define OVERLAP_DRAW_INCREMENT 101
#define OVERLAP_SEED_BASE 900u
#define OVERLAP_WAIT_NS 5000000000ull
#define OVERLAP_GATE_SECONDS 2
#define OVERLAP_FRONT_HOLD_NS 20000000l
#define OVERLAP_NS_PER_SECOND 1000000000l
#define OVERLAP_IDENTITY 0x5450345F53554D31ull
#define OVERLAP_WITNESS_GUARD 0x505245464958544Cull

struct witness {
	uint64_t sequence;
	int32_t token;
	uint32_t session, card, position;
	uint64_t guard;
};
_Static_assert(sizeof(struct witness) == 32u,
	       "the transfer carries one complete aligned witness");

struct snapshot {
	unsigned char bytes[OVERLAP_CARDS * sizeof(struct witness)];
	uint64_t size;
};
struct outcome {
	struct step_row_result results[OVERLAP_FLIGHTS];
	struct snapshot snapshots[OVERLAP_SESSIONS];
	bool completed;
};
enum fixture_form {
	ORDINARY_FORM,
	RETURNING_PREFIX_FORM,
	BACKWARD_HAND_OFF_FORM,
	FRONT_WITHOUT_TRANSFERS_FORM,
	EMBEDDING_HAND_OFF_FORM
};
struct fixture {
	struct backend_operations backend;
	struct device_card *cards[OVERLAP_CARDS];
	struct kernel_launch launches[OVERLAP_CARDS][2];
	struct program_card_record areas[OVERLAP_CARDS];
	struct segment_record segments[OVERLAP_CARDS + 1u];
	struct hand_off_record
		hand_offs[OVERLAP_CARDS + OVERLAP_EMBEDDING_COPIES];
	struct session_range ranges[OVERLAP_CARDS];
	struct execution_program program;
	struct step_driver *driver;
	pthread_mutex_t lock;
	pthread_cond_t changed;
	uint32_t queued_prefix[OVERLAP_FLIGHTS], waits[OVERLAP_PREFIX_CARDS];
	uint32_t tail_launches;
	uint32_t embedding_uploads[OVERLAP_FLIGHTS][OVERLAP_EMBEDDING_COPIES];
	bool embedding;
	bool gate_enabled, tail_waiting, overlapped, gate_expired;
	bool has_lock, has_condition;
	bool inject_failure, waited_before_group, witness_changed;
	bool gate_uploads, upload_waiting, release_upload, upload_gate_done;
	bool finish_started, finish_returned;
};
static struct fixture *active;
static int failures;

static void expect(bool condition, const char *message)
{
	if (!condition) {
		(void)fprintf(stderr, "step_overlap_test: %s\n", message);
		failures++;
	}
}

static uint32_t card_index(const struct device_card *card)
{
	for (uint32_t index = 0; index < OVERLAP_CARDS; index++)
		if (active->cards[index] == card)
			return index;
	return OVERLAP_CARDS;
}

static struct witness witness_of(const struct step_block *block,
				 const struct step_row_record *row,
				 uint32_t card)
{
	return (struct witness){
		block->sequence,	 row->token_id,
		row->session_slot,	 card,
		(uint32_t)row->position, OVERLAP_WITNESS_GUARD
	};
}

/* This callback holds the host backend's global mutex: gates belong in the outer wrapper below. */
static bool model_stand_in(const struct kernel_launch *launch, uint32_t card,
			   unsigned char *arena, uint64_t bytes, void *context)
{
	(void)bytes;
	struct fixture *const fixture = context;
	struct step_block block;
	struct step_row_record row;
	memcpy(&block, arena, sizeof(block));
	memcpy(&row, arena + STEP_BLOCK_BYTES_VALUE, sizeof(row));
	if (block.sequence == 0 || block.sequence > OVERLAP_FLIGHTS ||
	    row.session_slot >= OVERLAP_SESSIONS || block.row_count != 1u)
		return false;
	const struct witness written = witness_of(&block, &row, card);
	struct fill_record record;
	memcpy(&record, launch->argument, sizeof(record));
	if (record.value == OVERLAP_EMBEDDING_ROLE) {
		memcpy(arena + OVERLAP_EMBEDDING_OFFSET, &written,
		       sizeof(written));
		return card == 0;
	}
	if (card < OVERLAP_PREFIX_CARDS) {
		const struct witness embedded = witness_of(&block, &row, 0);
		if (fixture->embedding &&
		    memcmp(arena + OVERLAP_EMBEDDING_OFFSET, &embedded,
			   sizeof(embedded)) != 0) {
			fixture->witness_changed = true;
			return false;
		}
		memcpy(arena + OVERLAP_CARRY_OFFSET, &written, sizeof(written));
	} else {
		for (uint32_t owner = 0; owner < OVERLAP_PREFIX_CARDS;
		     owner++) {
			const struct witness expected =
				witness_of(&block, &row, owner);
			if (memcmp(arena + OVERLAP_CARRY_OFFSET +
					   owner * sizeof(expected),
				   &expected, sizeof(expected)) != 0) {
				fixture->witness_changed = true;
				return false;
			}
		}
		struct step_session settings;
		memcpy(&settings,
		       arena + OVERLAP_SETTINGS_OFFSET +
			       row.session_slot * sizeof(settings),
		       sizeof(settings));
		if (settings.seed != OVERLAP_SEED_BASE + row.session_slot ||
		    settings.temperature != 0.0f || settings.top_p != 1.0f)
			return false;
		const struct step_result header = { block.sequence, 1u, 0u };
		const struct step_row_result result = {
			row.token_id + OVERLAP_DRAW_INCREMENT,
			(float)row.token_id
		};
		memcpy(arena + OVERLAP_RESULT_OFFSET, &header, sizeof(header));
		memcpy(arena + OVERLAP_RESULT_OFFSET + sizeof(header), &result,
		       sizeof(result));
	}
	memcpy(arena + OVERLAP_STATE_OFFSET +
		       row.session_slot * sizeof(written),
	       &written, sizeof(written));
	return true;
}

/* The host backend serializes callbacks globally. Waiting here leaves its mutex available to the other stage. */
static bool stage_gate(struct fixture *fixture, uint32_t card,
		       uint64_t sequence)
{
	if (!fixture->gate_enabled ||
	    !((card == 0 && sequence == 2u) ||
	      (card == OVERLAP_TAIL_CARD && sequence == 1u)))
		return true;
	struct timespec deadline;
	if (clock_gettime(CLOCK_REALTIME, &deadline) != 0)
		return false;
	deadline.tv_sec += OVERLAP_GATE_SECONDS;
	(void)pthread_mutex_lock(&fixture->lock);
	if (card == OVERLAP_TAIL_CARD) {
		fixture->tail_waiting = true;
		(void)pthread_cond_broadcast(&fixture->changed);
	}
	int waited = 0;
	while (waited == 0 &&
	       (card == OVERLAP_TAIL_CARD ?
			fixture->queued_prefix[1] != OVERLAP_PREFIX_MASK :
			!fixture->tail_waiting))
		waited = pthread_cond_timedwait(&fixture->changed,
						&fixture->lock, &deadline);
	const bool passed = card == OVERLAP_TAIL_CARD ?
				    fixture->queued_prefix[1] ==
					    OVERLAP_PREFIX_MASK :
				    fixture->tail_waiting;
	fixture->gate_expired = fixture->gate_expired || !passed;
	fixture->overlapped = fixture->overlapped ||
			      (passed && card == OVERLAP_TAIL_CARD);
	(void)pthread_mutex_unlock(&fixture->lock);
	return passed;
}

MUST_CHECK static result_status
noting_launch(struct device_card *card, const struct kernel_launch *launches,
	      uint32_t count)
{
	struct fixture *const fixture = active;
	const uint32_t index = card_index(card);
	if (index == OVERLAP_CARDS)
		return STATUS_INVALID_ARGUMENT;
	const unsigned char *const arena =
		test_host_memory_backend_arena_of(card);
	if (arena == NULL)
		return STATUS_INVALID_ARGUMENT;
	struct step_block block;
	memcpy(&block, arena, sizeof(block));
	struct fill_record record;
	memcpy(&record, launches[0].argument, sizeof(record));
	const bool embedding = record.value == OVERLAP_EMBEDDING_ROLE;
	if (!embedding && !stage_gate(fixture, index, block.sequence))
		return STATUS_TIMED_OUT;
	const result_status status =
		test_host_memory_backend.launch_run(card, launches, count);
	(void)pthread_mutex_lock(&fixture->lock);
	if (!embedding && index < OVERLAP_PREFIX_CARDS && block.sequence != 0 &&
	    block.sequence <= OVERLAP_FLIGHTS) {
		fixture->queued_prefix[block.sequence - 1u] |= 1u << index;
		(void)pthread_cond_broadcast(&fixture->changed);
	}
	fixture->tail_launches += index == OVERLAP_TAIL_CARD ? 1u : 0u;
	(void)pthread_mutex_unlock(&fixture->lock);
	return status == STATUS_OK && fixture->inject_failure && index == 0 &&
			       !embedding ?
		       STATUS_INJECTED_FAULT :
		       status;
}

MUST_CHECK static result_status noting_wait(struct device_card *card,
					    uint64_t limit)
{
	struct fixture *const fixture = active;
	const uint32_t index = card_index(card);
	if (index < OVERLAP_PREFIX_CARDS) {
		const unsigned char *const arena =
			test_host_memory_backend_arena_of(card);
		if (arena == NULL)
			return STATUS_INVALID_ARGUMENT;
		struct step_block block;
		memcpy(&block, arena, sizeof(block));
		(void)pthread_mutex_lock(&fixture->lock);
		if (block.sequence != 0 && block.sequence <= OVERLAP_FLIGHTS &&
		    fixture->queued_prefix[block.sequence - 1u] != 0) {
			fixture->waited_before_group =
				fixture->waited_before_group ||
				fixture->queued_prefix[block.sequence - 1u] !=
					OVERLAP_PREFIX_MASK;
			fixture->waits[index]++;
		}
		(void)pthread_mutex_unlock(&fixture->lock);
	}
	return test_host_memory_backend.launch_wait(card, limit);
}

/* A headless flight still owns its STEP upload. Hold its completion outside the host backend mutex. */
MUST_CHECK static result_status
noting_upload_wait(struct device_card *card, uint64_t ticket, uint64_t limit)
{
	struct fixture *const fixture = active;
	if (card_index(card) == OVERLAP_TAIL_CARD && fixture->gate_uploads) {
		struct timespec deadline;
		if (clock_gettime(CLOCK_REALTIME, &deadline) != 0)
			return STATUS_TIMED_OUT;
		deadline.tv_sec += OVERLAP_GATE_SECONDS;
		(void)pthread_mutex_lock(&fixture->lock);
		if (!fixture->upload_gate_done) {
			fixture->upload_waiting = true;
			(void)pthread_cond_broadcast(&fixture->changed);
			int waited = 0;
			while (!fixture->release_upload && waited == 0)
				waited = pthread_cond_timedwait(
					&fixture->changed, &fixture->lock,
					&deadline);
			fixture->gate_expired = fixture->gate_expired ||
						!fixture->release_upload;
			fixture->upload_gate_done = fixture->release_upload;
		}
		const bool passed = fixture->upload_gate_done;
		(void)pthread_mutex_unlock(&fixture->lock);
		if (!passed)
			return STATUS_TIMED_OUT;
	}
	return test_host_memory_backend.upload_wait(card, ticket, limit);
}

/* Each embedding handoff belongs to the prefix exactly once. Replaying it from the tail would race the next
 * prefix's input, even when a particular host schedule happened to hide the stale overwrite. */
MUST_CHECK static result_status
noting_upload_begin(struct device_card *card, uint64_t offset,
		    const void *source, uint64_t bytes, uint64_t *ticket)
{
	struct fixture *const fixture = active;
	const uint32_t index = card_index(card);
	if (fixture->embedding && index != 0 && index < OVERLAP_PREFIX_CARDS &&
	    offset == OVERLAP_EMBEDDING_OFFSET &&
	    bytes == sizeof(struct witness)) {
		struct witness embedded;
		memcpy(&embedded, source, sizeof(embedded));
		if (embedded.sequence == 0 ||
		    embedded.sequence > OVERLAP_FLIGHTS)
			return STATUS_INVALID_ARGUMENT;
		(void)pthread_mutex_lock(&fixture->lock);
		fixture->embedding_uploads[embedded.sequence - 1u][index - 1u]++;
		(void)pthread_mutex_unlock(&fixture->lock);
	}
	return test_host_memory_backend.upload_begin(card, offset, source,
						     bytes, ticket);
}

static result_status fixture_open(struct fixture *fixture,
				  enum fixture_form form, bool groups)
{
	memset(fixture, 0, sizeof(*fixture));
	active = fixture;
	fixture->backend = test_host_memory_backend;
	fixture->backend.launch_run = noting_launch;
	fixture->backend.launch_wait = noting_wait;
	fixture->backend.upload_wait = noting_upload_wait;
	fixture->backend.upload_begin = noting_upload_begin;
	fixture->embedding = form == EMBEDDING_HAND_OFF_FORM;
	fixture->has_lock = pthread_mutex_init(&fixture->lock, NULL) == 0;
	if (!fixture->has_lock)
		return STATUS_OUT_OF_MEMORY;
	fixture->has_condition = pthread_cond_init(&fixture->changed, NULL) ==
				 0;
	if (!fixture->has_condition)
		return STATUS_OUT_OF_MEMORY;
	test_host_memory_backend_reset(OVERLAP_CARDS, OVERLAP_ARENA_BYTES);
	result_status status = test_host_memory_backend_set_stand_in(
		KERNEL_FILE_FILL, KERNEL_FILL_WORDS, model_stand_in, fixture);
	for (uint32_t card = 0; card < OVERLAP_CARDS && status == STATUS_OK;
	     card++) {
		test_host_memory_backend_set_arithmetic_class(
			card, card == OVERLAP_TAIL_CARD ?
				      ARITHMETIC_CLASS_SM_120 :
				      ARITHMETIC_CLASS_SM_80);
		status = fixture->backend.card_open(card, OVERLAP_ARENA_BYTES,
						    &fixture->cards[card]);
		uint32_t kernel = 0;
		if (status == STATUS_OK)
			status = fixture->backend.kernel_prepare(
				fixture->cards[card], KERNEL_FILE_FILL,
				KERNEL_FILL_WORDS, FILL_RECORD_BYTES, &kernel);
		const struct fill_record record = { 0 };
		prepare_launch_fill(&fixture->launches[card][0], kernel, 1, 1,
				    1, 0, &record, sizeof(record));
		fixture->launches[card][1] = fixture->launches[card][0];
		fixture->areas[card] = (struct program_card_record){
			.launches = fixture->launches[card],
			.arena_byte_count = OVERLAP_ARENA_BYTES,
			.step_offset = 0,
			.session_offset = card == OVERLAP_TAIL_CARD ?
						  OVERLAP_SETTINGS_OFFSET :
						  PROGRAM_NOWHERE,
			.result_offset = card == OVERLAP_TAIL_CARD ?
						 OVERLAP_RESULT_OFFSET :
						 PROGRAM_NOWHERE,
			.logits_offset = PROGRAM_NOWHERE,
			.round_offset = PROGRAM_NOWHERE,
			.streams_offset = PROGRAM_NOWHERE,
			.carried_pre_offset = PROGRAM_NOWHERE,
			.engram_rows_offset = PROGRAM_NOWHERE,
			.launch_count = 1
		};
		fixture->segments[card] = (struct segment_record){
			.card = card, .stage = 0, .layer = 0, .launch_count = 1
		};
		fixture->ranges[card] = (struct session_range){
			.offset = OVERLAP_STATE_OFFSET,
			.session_stride = sizeof(struct witness),
			.unit_count = 1,
			.unit_bytes = sizeof(struct witness),
			.unit_tokens = 0,
			.card = card,
			.identity = card
		};
		if (card < OVERLAP_PREFIX_CARDS)
			fixture->hand_offs[card] = (struct hand_off_record){
				.source_offset = OVERLAP_CARRY_OFFSET,
				.destination_offset =
					OVERLAP_CARRY_OFFSET +
					card * sizeof(struct witness),
				.byte_count = sizeof(struct witness),
				.from_card = card,
				.to_card = OVERLAP_TAIL_CARD,
				.after_segment = card
			};
	}
	fixture->program = (struct execution_program){
		.cards = fixture->areas,
		.segments = fixture->segments,
		.hand_offs = fixture->hand_offs,
		.limits = { .context_token_limit = OVERLAP_CONTEXT,
			    .row_capacity = 1,
			    .session_capacity = OVERLAP_SESSIONS,
			    .state_width = STATE_WIDTH_16 },
		.card_count = OVERLAP_CARDS,
		.segment_count = OVERLAP_CARDS,
		.hand_off_count = OVERLAP_PREFIX_CARDS,
		.row_count = 1,
		.result_card = OVERLAP_TAIL_CARD,
		.kind = PROGRAM_THIN,
		.session_ranges = fixture->ranges,
		.session_range_count = OVERLAP_CARDS,
		.front_segment_count = OVERLAP_PREFIX_CARDS,
		.draft_segment_first = OVERLAP_CARDS
	};
	if (form == RETURNING_PREFIX_FORM) {
		fixture->segments[OVERLAP_CARDS] =
			(struct segment_record){ .card = 0,
						 .stage = 1,
						 .layer = 1,
						 .first_launch = 1,
						 .launch_count = 1 };
		fixture->areas[0].launch_count++;
		fixture->program.segment_count++;
		fixture->program.draft_segment_first++;
	}
	if (form == BACKWARD_HAND_OFF_FORM) {
		fixture->hand_offs[OVERLAP_PREFIX_CARDS] =
			(struct hand_off_record){
				.source_offset = OVERLAP_CARRY_OFFSET,
				.destination_offset = OVERLAP_CARRY_OFFSET,
				.byte_count = sizeof(struct witness),
				.from_card = OVERLAP_TAIL_CARD,
				.to_card = 0,
				.after_segment = OVERLAP_TAIL_CARD
			};
		fixture->program.hand_off_count++;
	}
	if (form == FRONT_WITHOUT_TRANSFERS_FORM)
		fixture->program.hand_off_count = 0;
	if (fixture->embedding) {
		/* Match original TP4_PRO: card0 embeds, three internal handoffs, then one four-member prefix. */
		memmove(&fixture->segments[1], fixture->segments,
			OVERLAP_CARDS * sizeof(fixture->segments[0]));
		fixture->segments[0] = (struct segment_record){
			.card = 0, .layer = SEGMENT_NO_LAYER, .launch_count = 1
		};
		fixture->segments[1].first_launch = 1;
		fixture->areas[0].launch_count++;
		const struct fill_record record = {
			.value = OVERLAP_EMBEDDING_ROLE
		};
		memcpy(fixture->launches[0][0].argument, &record,
		       sizeof(record));
		memmove(&fixture->hand_offs[OVERLAP_EMBEDDING_COPIES],
			fixture->hand_offs,
			OVERLAP_PREFIX_CARDS * sizeof(fixture->hand_offs[0]));
		for (uint32_t peer = 1; peer < OVERLAP_PREFIX_CARDS; peer++)
			fixture->hand_offs[peer - 1u] = (struct hand_off_record){
				.source_offset = OVERLAP_EMBEDDING_OFFSET,
				.destination_offset = OVERLAP_EMBEDDING_OFFSET,
				.byte_count = sizeof(struct witness),
				.from_card = 0,
				.to_card = peer,
				.after_segment = 0
			};
		for (uint32_t transfer = OVERLAP_EMBEDDING_COPIES;
		     transfer < OVERLAP_EMBEDDING_COPIES + OVERLAP_PREFIX_CARDS;
		     transfer++)
			fixture->hand_offs[transfer].after_segment++;
		fixture->program.segment_count++;
		fixture->program.front_segment_count++;
		fixture->program.draft_segment_first++;
		fixture->program.hand_off_count += OVERLAP_EMBEDDING_COPIES;
	}
	const struct execution_program *const program = &fixture->program;
	if (status == STATUS_OK)
		status = step_driver_create(&fixture->backend, fixture->cards,
					    &program, 1, NULL, OVERLAP_WAIT_NS,
					    &fixture->driver);
	const uint32_t first_member = fixture->embedding ? 1u : 0u;
	const struct step_companion_pair pair = {
		first_member, first_member + OVERLAP_PREFIX_CARDS - 1u
	};
	const struct step_companion_program companion = {
		.program = program,
		.pairs = &pair,
		.pair_count = 1,
		.arithmetic_identity = OVERLAP_IDENTITY
	};
	if (status == STATUS_OK && groups)
		status = step_groups_attach(fixture->driver, &companion, 1,
					    OVERLAP_PREFIX_CARDS);
	for (uint32_t slot = 0; slot < OVERLAP_SESSIONS && status == STATUS_OK;
	     slot++) {
		const struct step_session settings = { OVERLAP_SEED_BASE + slot,
						       0.0f, 1.0f };
		status = step_session_set(fixture->driver, slot, &settings);
	}
	return status;
}

static void fixture_close(struct fixture *fixture)
{
	step_driver_destroy(fixture->driver);
	for (uint32_t card = 0; card < OVERLAP_CARDS; card++)
		if (fixture->cards[card] != NULL)
			fixture->backend.card_close(fixture->cards[card],
						    OVERLAP_WAIT_NS);
	if (fixture->has_condition)
		(void)pthread_cond_destroy(&fixture->changed);
	if (fixture->has_lock)
		(void)pthread_mutex_destroy(&fixture->lock);
	active = NULL;
	expect(test_host_memory_backend_open_card_count() == 0 &&
		       test_host_memory_backend_staging_block_count() == 0,
	       "all cards and staging retire");
}

static bool save_bytes(void *context, unsigned char *bytes, uint64_t count)
{
	struct snapshot *const snapshot = context;
	if (count > sizeof(snapshot->bytes) - snapshot->size)
		return false;
	memcpy(snapshot->bytes + snapshot->size, bytes, (size_t)count);
	snapshot->size += count;
	return true;
}

static int32_t input_token(uint32_t slot, uint32_t wave)
{
	return OVERLAP_FIRST_TOKEN + (int32_t)slot * OVERLAP_TOKEN_STRIDE +
	       (int32_t)wave * OVERLAP_WAVE_TOKEN_STRIDE;
}

static struct outcome run_case(uint32_t depth, bool overlap, bool fail_prefix,
			       bool embedding)
{
	struct outcome outcome = { 0 };
	struct fixture fixture;
	result_status status = fixture_open(
		&fixture, embedding ? EMBEDDING_HAND_OFF_FORM : ORDINARY_FORM,
		true);
	expect(status == STATUS_OK, "the five-card fixture opens");
	fixture.gate_enabled = overlap && !fail_prefix;
	fixture.inject_failure = fail_prefix;
	if (status == STATUS_OK)
		status = overlap ? step_pipeline_overlap_start(fixture.driver,
							       depth) :
				   step_pipeline_start(fixture.driver, depth);
	expect(status == STATUS_OK &&
		       step_pipeline_depth(fixture.driver) == depth,
	       "the requested pipeline starts without executing work");
	const uint32_t count = fail_prefix ? 1u : depth;
	const uint32_t waves = fail_prefix ? 1u : OVERLAP_WAVES;
	for (uint32_t wave = 0; wave < waves && status == STATUS_OK; wave++) {
		for (uint32_t slot = 0; slot < count && status == STATUS_OK;
		     slot++) {
			const struct step_row_record row = {
				(int32_t)wave, input_token(slot, wave), slot,
				STEP_ROW_LOGITS
			};
			status = step_submit(fixture.driver, &row, 1);
		}
		for (uint32_t slot = 0; slot < count && status == STATUS_OK;
		     slot++)
			status = step_finish(
				fixture.driver,
				&outcome.results[wave * depth + slot]);
	}
	expect(status == (fail_prefix ? STATUS_INJECTED_FAULT : STATUS_OK),
	       "finish preserves success or the first partial-prefix failure");
	if (!fail_prefix && status == STATUS_OK) {
		expect(!fixture.witness_changed && !fixture.waited_before_group,
		       "all four prefix members enqueue before waits and each tail reads its own transfer bytes");
		expect(!overlap ||
			       (fixture.overlapped && !fixture.gate_expired),
		       "the next CMP prefix retires while the preceding PRO tail is held");
		for (uint32_t wave = 0; wave < waves; wave++) {
			for (uint32_t slot = 0; slot < count; slot++) {
				expect(outcome.results[wave * depth + slot]
						       .token_id ==
					       input_token(slot, wave) +
						       OVERLAP_DRAW_INCREMENT,
				       "each reused ring slot returns its current flight's selected token");
				expect(step_history_token(fixture.driver, slot,
							  wave) ==
					       (uint32_t)input_token(slot,
								     wave),
				       "successive waves retain every continuation token in the correct session history");
			}
		}
		for (uint32_t slot = 0; slot < count; slot++) {
			status = step_session_save(fixture.driver, slot, waves,
						   save_bytes,
						   &outcome.snapshots[slot]);
			expect(status == STATUS_OK &&
				       outcome.snapshots[slot].size ==
					       sizeof(outcome.snapshots[slot]
							      .bytes),
			       "each continued session exposes its complete literal canonical state");
			for (uint32_t card = 0; card < OVERLAP_CARDS; card++) {
				const struct witness expected = {
					(uint64_t)(waves - 1u) * depth + slot +
						1u,
					input_token(slot, waves - 1u),
					slot,
					card,
					waves - 1u,
					OVERLAP_WITNESS_GUARD
				};
				expect(memcmp(outcome.snapshots[slot].bytes +
						      card * sizeof(expected),
					      &expected, sizeof(expected)) == 0,
				       "all stage states belong to the correct session's final continuation");
			}
		}
		for (uint32_t flight = 0; embedding && flight < depth * waves;
		     flight++)
			for (uint32_t peer = 0; peer < OVERLAP_EMBEDDING_COPIES;
			     peer++)
				expect(fixture.embedding_uploads[flight][peer] ==
					       1u,
				       "each CMP peer receives one current embedding, with no tail replay");
		outcome.completed = true;
	}
	fixture_close(&fixture);
	if (fail_prefix) {
		expect(fixture.queued_prefix[0] == OVERLAP_PREFIX_MASK &&
			       !fixture.waited_before_group,
		       "a partial prefix enqueue failure still submits all peer members before draining");
		for (uint32_t card = 0; card < OVERLAP_PREFIX_CARDS; card++)
			expect(fixture.waits[card] != 0,
			       "every prefix card is drained after failure");
	}
	return outcome;
}

static void refusal_case(enum fixture_form form, bool groups)
{
	struct fixture fixture;
	const result_status status = fixture_open(&fixture, form, groups);
	expect(status == STATUS_OK,
	       "an ordinary supported fixture opens before overlap refusal");
	if (status == STATUS_OK) {
		const uint64_t work = test_host_memory_backend_work_count();
		expect(step_pipeline_overlap_start(fixture.driver, 2u) !=
				       STATUS_OK &&
			       step_pipeline_depth(fixture.driver) == 1u &&
			       test_host_memory_backend_work_count() == work,
		       "unsupported topology refuses overlap before device work or pipeline publication");
	}
	fixture_close(&fixture);
}

struct front_finish {
	struct fixture *fixture;
	struct step_row_result result;
	result_status status;
};

static void *finish_front(void *context)
{
	struct front_finish *const finished = context;
	struct fixture *const fixture = finished->fixture;
	(void)pthread_mutex_lock(&fixture->lock);
	fixture->finish_started = true;
	(void)pthread_cond_broadcast(&fixture->changed);
	(void)pthread_mutex_unlock(&fixture->lock);
	finished->status = step_finish(fixture->driver, &finished->result);
	(void)pthread_mutex_lock(&fixture->lock);
	fixture->finish_returned = true;
	(void)pthread_cond_broadcast(&fixture->changed);
	(void)pthread_mutex_unlock(&fixture->lock);
	return NULL;
}

/* No outgoing transfers means the ordinary task table initially has no PRO task. Its STEP upload still owns
 * the slot's staging until the tail worker waits it, so a headless result cannot retire on prefix completion. */
static void front_retirement(void)
{
	struct fixture fixture;
	result_status status =
		fixture_open(&fixture, FRONT_WITHOUT_TRANSFERS_FORM, true);
	expect(status == STATUS_OK,
	       "the headless fixture opens without outgoing transfers");
	fixture.gate_uploads = true;
	if (status == STATUS_OK)
		status = step_pipeline_overlap_start(fixture.driver, 2u);
	for (uint32_t position = 0; position < 2u && status == STATUS_OK;
	     position++) {
		const struct step_row_record row = { (int32_t)position,
						     input_token(0, position),
						     0, STEP_ROW_FRONT };
		status = step_submit(fixture.driver, &row, 1);
	}
	expect(status == STATUS_OK,
	       "two same-session front steps commit queued history before either tail upload completes");
	struct front_finish finished = { .fixture = &fixture,
					 .status = STATUS_INVALID_ARGUMENT };
	pthread_t thread;
	const bool has_thread =
		status == STATUS_OK &&
		pthread_create(&thread, NULL, finish_front, &finished) == 0;
	expect(has_thread, "the headless finisher starts");
	if (has_thread) {
		struct timespec deadline;
		const bool has_clock =
			clock_gettime(CLOCK_REALTIME, &deadline) == 0;
		expect(has_clock,
		       "the bounded headless completion gate has a clock");
		(void)pthread_mutex_lock(&fixture.lock);
		if (has_clock) {
			deadline.tv_sec += OVERLAP_GATE_SECONDS;
			int waited = 0;
			while (waited == 0 && (!fixture.upload_waiting ||
					       !fixture.finish_started ||
					       fixture.queued_prefix[1] !=
						       OVERLAP_PREFIX_MASK))
				waited = pthread_cond_timedwait(
					&fixture.changed, &fixture.lock,
					&deadline);
			expect(fixture.upload_waiting &&
				       fixture.finish_started &&
				       fixture.queued_prefix[1] ==
					       OVERLAP_PREFIX_MASK,
			       "the finisher and both prefixes reach the held PRO upload without a tail kernel");
			if (clock_gettime(CLOCK_REALTIME, &deadline) == 0) {
				deadline.tv_nsec += OVERLAP_FRONT_HOLD_NS;
				deadline.tv_sec += deadline.tv_nsec /
						   OVERLAP_NS_PER_SECOND;
				deadline.tv_nsec %= OVERLAP_NS_PER_SECOND;
				waited = 0;
				while (!fixture.finish_returned && waited == 0)
					waited = pthread_cond_timedwait(
						&fixture.changed, &fixture.lock,
						&deadline);
			}
			expect(!fixture.finish_returned &&
				       fixture.tail_launches == 0 &&
				       !fixture.gate_expired,
			       "a headless finish stays blocked while its native-card upload has not retired");
		}
		fixture.release_upload = true;
		(void)pthread_cond_broadcast(&fixture.changed);
		(void)pthread_mutex_unlock(&fixture.lock);
		(void)pthread_join(thread, NULL);
		expect(finished.status == STATUS_OK &&
			       finished.result.token_id == STEP_NO_TOKEN &&
			       finished.result.logit == 0.0f,
		       "retiring the first upload publishes its headless result");
		struct step_row_result second;
		status = step_finish(fixture.driver, &second);
		expect(status == STATUS_OK &&
			       second.token_id == STEP_NO_TOKEN &&
			       second.logit == 0.0f &&
			       fixture.tail_launches == 0,
		       "the next front step retires its own upload without executing the PRO tail");
		struct snapshot state = { 0 };
		status = step_session_save(fixture.driver, 0, 2u, save_bytes,
					   &state);
		expect(status == STATUS_OK && state.size == sizeof(state.bytes),
		       "headless continuations leave a complete canonical snapshot");
		for (uint32_t card = 0; card < OVERLAP_PREFIX_CARDS; card++) {
			const struct witness expected = {
				2u, input_token(0, 1u),	  0, card,
				1u, OVERLAP_WITNESS_GUARD
			};
			expect(memcmp(state.bytes + card * sizeof(expected),
				      &expected, sizeof(expected)) == 0,
			       "each prefix state contains the last queued front continuation");
		}
		const struct witness untouched = { 0 };
		expect(memcmp(state.bytes +
				      OVERLAP_TAIL_CARD * sizeof(untouched),
			      &untouched, sizeof(untouched)) == 0,
		       "headless uploads do not execute a native tail state write");
		for (uint32_t position = 0; position < 2u; position++)
			expect(step_history_token(fixture.driver, 0,
						  position) ==
				       (uint32_t)input_token(0, position),
			       "queued front histories retain both supplied tokens");
	} else if (fixture.has_lock) {
		(void)pthread_mutex_lock(&fixture.lock);
		fixture.release_upload = true;
		if (fixture.has_condition)
			(void)pthread_cond_broadcast(&fixture.changed);
		(void)pthread_mutex_unlock(&fixture.lock);
	}
	fixture_close(&fixture);
}

static void depth_refusals(void)
{
	struct fixture fixture;
	result_status status = fixture_open(&fixture, ORDINARY_FORM, true);
	expect(status == STATUS_OK,
	       "a valid fixture opens for depth admission");
	if (status == STATUS_OK) {
		const uint64_t work = test_host_memory_backend_work_count();
		expect(step_pipeline_overlap_start(fixture.driver, 1u) ==
				       STATUS_INVALID_ARGUMENT &&
			       step_pipeline_overlap_start(fixture.driver,
							   STEP_PIPELINE_MOST +
								   1u) ==
				       STATUS_INVALID_ARGUMENT &&
			       step_pipeline_depth(fixture.driver) == 1u &&
			       test_host_memory_backend_work_count() == work,
		       "depth refusals leave the cold driver and devices untouched");
		status = step_pipeline_overlap_start(fixture.driver, 2u);
		expect(status == STATUS_OK &&
			       step_pipeline_start(fixture.driver, 2u) ==
				       STATUS_INVALID_ARGUMENT &&
			       step_pipeline_overlap_start(fixture.driver,
							   2u) ==
				       STATUS_INVALID_ARGUMENT &&
			       step_pipeline_depth(fixture.driver) == 2u,
		       "a cold retry succeeds and subsequent pipeline starts preserve the installed workers");
	}
	fixture_close(&fixture);
}

int main(void)
{
	expect(step_pipeline_overlap_start(NULL, 2u) == STATUS_INVALID_ARGUMENT,
	       "a missing driver is refused");
	for (uint32_t embedding = 0; embedding <= 1u; embedding++)
		for (uint32_t depth = 2u; depth <= OVERLAP_SESSIONS; depth++) {
			const struct outcome serial =
				run_case(depth, false, false, embedding != 0);
			const struct outcome overlap =
				run_case(depth, true, false, embedding != 0);
			expect(serial.completed && overlap.completed &&
				       memcmp(serial.results, overlap.results,
					      depth * OVERLAP_WAVES *
						      sizeof(serial.results[0])) ==
					       0,
			       "serial and overlapped flights return exactly the same tokens and logit bits");
			for (uint32_t slot = 0; slot < depth; slot++)
				expect(serial.snapshots[slot].size ==
						       overlap.snapshots[slot]
							       .size &&
					       memcmp(serial.snapshots[slot]
							      .bytes,
						      overlap.snapshots[slot]
							      .bytes,
						      sizeof(serial.snapshots[slot]
								     .bytes)) ==
						       0,
				       "serial and overlapped flights preserve every canonical state byte");
		}
	(void)run_case(2u, true, true, false);
	(void)run_case(2u, true, true, true);
	refusal_case(ORDINARY_FORM, false);
	refusal_case(RETURNING_PREFIX_FORM, true);
	refusal_case(BACKWARD_HAND_OFF_FORM, true);
	depth_refusals();
	front_retirement();
	return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
