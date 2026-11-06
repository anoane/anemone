/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Four-card co-enqueue before any wait, including a submit that queued work before reporting failure. */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "anemone/backend_records.h"
#include "anemone/step.h"
#include "host_memory_backend.h"
#include "engram_pack.h"
#include "prepare_common.h"

#define GROUP_QUEUE_DEPTH 3u
#define GROUP_MASK_CARDS 2u
#define GROUP_RECENT_TOKENS 2u
#define GROUP_ENGRAM_OFFSET 2048u
#define GROUP_BOTH_TABLES ((1u << TEST_ENGRAM_PACK_TABLES) - 1u)
#define GROUP_CARDS 4u
#define GROUP_MASK ((1u << GROUP_CARDS) - 1u)
#define GROUP_ARENA_BYTES 4096u
#define GROUP_RESULT_OFFSET 512u
#define GROUP_SESSION_OFFSET 1024u
#define GROUP_CONTEXT 64u
#define GROUP_WAIT_NS 1000000000ull
#define GROUP_IDENTITY 0x5450345F53554D31ull

static struct device_card *opened[GROUP_CARDS];
static struct engram_record *engram;
static uint64_t upload_sequence[GROUP_CARDS];
static uintptr_t gathered_sources[GROUP_QUEUE_DEPTH][GROUP_CARDS]
				 [TEST_ENGRAM_PACK_TABLES];
static uint32_t submitted, waits[GROUP_CARDS];
static bool inject_failure, waited_early, sequence_mismatch;
static uint64_t group_sequence;
static uint32_t groups_seen;
static int failures;

static void expect(bool condition, const char *message)
{
	if (!condition) {
		(void)fprintf(stderr, "step_groups_test: %s\n", message);
		failures++;
	}
}

static uint32_t index_of(const struct device_card *card)
{
	for (uint32_t index = 0; index < GROUP_CARDS; index++) {
		if (opened[index] == card)
			return index;
	}
	return GROUP_CARDS;
}

static bool stand_in(const struct kernel_launch *launch, uint32_t card,
		     unsigned char *arena, uint64_t bytes, void *context)
{
	(void)launch;
	(void)bytes;
	(void)context;
	struct step_block step;
	(void)memcpy(&step, arena, sizeof(step));
	if (card == 0) {
		sequence_mismatch = sequence_mismatch ||
				    step.sequence <= group_sequence;
		group_sequence = step.sequence;
		groups_seen++;
	} else {
		sequence_mismatch = sequence_mismatch ||
				    step.sequence != group_sequence;
	}
	struct step_row_record row;
	memcpy(&row, arena + STEP_BLOCK_BYTES_VALUE, sizeof(row));
	const uint32_t position = (uint32_t)row.position;
	const uint32_t recent[GROUP_RECENT_TOKENS] = { position == 0 ? 1u :
								       position,
						       position + 1u };
	const uint8_t dead[GROUP_RECENT_TOKENS] = { 0 };
	uint64_t ids[TEST_ENGRAM_PACK_TABLES];
	if (engram_row_ids_entry(engram, recent, dead, position == 0 ? 0u : 1u,
				 ids) != STATUS_OK)
		return false;
	const uint32_t tables =
		card < GROUP_MASK_CARDS ? TEST_ENGRAM_PACK_TABLES : 1u;
	for (uint32_t table = 0; table < tables; table++) {
		unsigned char expected[ENGRAM_ROW_BYTES];
		if (engram_gather_entry(engram, table, &ids[table], 1,
					expected) != STATUS_OK ||
		    memcmp(arena + GROUP_ENGRAM_OFFSET +
				   table * ENGRAM_ROW_BYTES,
			   expected, sizeof(expected)) != 0)
			return false;
	}
	return true;
}

MUST_CHECK static result_status
noting_upload(struct device_card *card, uint64_t arena_offset,
	      const void *source, uint64_t bytes, uint64_t *ticket_out)
{
	const uint32_t index = index_of(card);
	if (index >= GROUP_CARDS)
		return STATUS_INVALID_ARGUMENT;
	if (arena_offset == 0) {
		struct step_block block;
		memcpy(&block, source, sizeof(block));
		upload_sequence[index] = block.sequence;
	} else if (arena_offset >= GROUP_ENGRAM_OFFSET &&
		   arena_offset <
			   GROUP_ENGRAM_OFFSET +
				   TEST_ENGRAM_PACK_TABLES * ENGRAM_ROW_BYTES) {
		const uint64_t sequence = upload_sequence[index];
		const uint32_t table =
			(uint32_t)(arena_offset - GROUP_ENGRAM_OFFSET) /
			ENGRAM_ROW_BYTES;
		if (sequence == 0 || sequence > GROUP_QUEUE_DEPTH ||
		    bytes != ENGRAM_ROW_BYTES)
			return STATUS_INVALID_ARGUMENT;
		gathered_sources[sequence - 1u][index][table] =
			(uintptr_t)source;
	}
	return test_host_memory_backend.upload_begin(card, arena_offset, source,
						     bytes, ticket_out);
}

MUST_CHECK static result_status enqueue(struct device_card *card,
					const struct kernel_launch *launches,
					uint32_t count)
{
	const uint32_t index = index_of(card);
	if (index >= GROUP_CARDS)
		return STATUS_INVALID_ARGUMENT;
	if (index == 0)
		submitted = 0;
	submitted |= 1u << index;
	const result_status status =
		test_host_memory_backend.launch_run(card, launches, count);
	return status == STATUS_OK && inject_failure && index == 0 ?
		       STATUS_INJECTED_FAULT :
		       status;
}

MUST_CHECK static result_status wait_card(struct device_card *card,
					  uint64_t limit)
{
	const uint32_t index = index_of(card);
	if (index >= GROUP_CARDS)
		return STATUS_INVALID_ARGUMENT;
	if (submitted != 0) {
		waited_early = waited_early || submitted != GROUP_MASK;
		waits[index]++;
	}
	return test_host_memory_backend.launch_wait(card, limit);
}

static void run_case(bool fail_first, bool pipeline, bool shares)
{
	struct backend_operations backend = test_host_memory_backend;
	backend.launch_run = enqueue;
	backend.upload_begin = noting_upload;
	backend.launch_wait = wait_card;
	struct kernel_launch launches[GROUP_CARDS] = { 0 };
	struct program_card_record areas[GROUP_CARDS] = { 0 };
	struct segment_record segments[GROUP_CARDS] = { 0 };
	struct step_driver *driver = NULL;
	(void)memset((void *)opened, 0, sizeof(opened));
	(void)memset(waits, 0, sizeof(waits));
	memset(upload_sequence, 0, sizeof(upload_sequence));
	memset(gathered_sources, 0, sizeof(gathered_sources));
	submitted = 0;
	waited_early = sequence_mismatch = false;
	group_sequence = 0;
	groups_seen = 0;
	inject_failure = fail_first;
	test_host_memory_backend_reset(GROUP_CARDS, GROUP_ARENA_BYTES);
	result_status status = test_host_memory_backend_set_stand_in(
		KERNEL_FILE_FILL, KERNEL_FILL_WORDS, stand_in, NULL);
	for (uint32_t card = 0; card < GROUP_CARDS && status == STATUS_OK;
	     card++) {
		test_host_memory_backend_set_arithmetic_class(
			card, ARITHMETIC_CLASS_SM_80);
		status = backend.card_open(card, GROUP_ARENA_BYTES,
					   &opened[card]);
		uint32_t kernel = 0;
		if (status == STATUS_OK)
			status = backend.kernel_prepare(
				opened[card], KERNEL_FILE_FILL,
				KERNEL_FILL_WORDS, FILL_RECORD_BYTES, &kernel);
		const struct fill_record record = { 0 };
		prepare_launch_fill(&launches[card], kernel, 1, 1, 1, 0,
				    &record, sizeof(record));
		areas[card] = (struct program_card_record){
			.launches = &launches[card],
			.launch_count = 1,
			.arena_byte_count = GROUP_ARENA_BYTES,
			.step_offset = 0,
			.session_offset = card + 1u == GROUP_CARDS ?
						  GROUP_SESSION_OFFSET :
						  PROGRAM_NOWHERE,
			.result_offset = card + 1u == GROUP_CARDS ?
						 GROUP_RESULT_OFFSET :
						 PROGRAM_NOWHERE,
			.logits_offset = PROGRAM_NOWHERE,
			.round_offset = PROGRAM_NOWHERE,
			.streams_offset = PROGRAM_NOWHERE,
			.carried_pre_offset = PROGRAM_NOWHERE,
			.engram_rows_offset = GROUP_ENGRAM_OFFSET,
			.engram_tables =
				card < GROUP_MASK_CARDS ? GROUP_BOTH_TABLES : 1u
		};
		segments[card] = (struct segment_record){
			.card = card, .stage = 0, .layer = 0, .launch_count = 1
		};
	}
	const struct execution_program program = {
		.cards = areas,
		.segments = segments,
		.limits = { .context_token_limit = GROUP_CONTEXT,
			    .row_capacity = 1,
			    .session_capacity = 1,
			    .state_width = STATE_WIDTH_16 },
		.card_count = GROUP_CARDS,
		.segment_count = GROUP_CARDS,
		.row_count = 1,
		.result_card = GROUP_CARDS - 1u,
		.front_segment_count = GROUP_CARDS,
		.draft_segment_first = GROUP_CARDS
	};
	const struct execution_program *const pointer = &program;
	if (status == STATUS_OK)
		status = step_driver_create(&backend, opened, &pointer, 1,
					    engram, GROUP_WAIT_NS, &driver);
	expect(status == STATUS_OK, "four-card driver opens");
	expect(step_engram_share(driver) == STATUS_INVALID_ARGUMENT,
	       "shared rows require serialized TP groups");
	const struct step_companion_pair range = { 0, GROUP_CARDS - 1u };
	const struct step_companion_program description = {
		.program = &program,
		.pairs = &range,
		.pair_count = 1,
		.arithmetic_identity = GROUP_IDENTITY
	};
	if (status == STATUS_OK) {
		expect(step_companions_attach(driver, &description, 1) ==
			       STATUS_INVALID_ARGUMENT,
		       "legacy pair API still refuses four members");
		status = step_groups_attach(driver, &description, 1,
					    GROUP_CARDS);
		expect(status == STATUS_OK,
		       "explicit four-card group attaches");
	}
	if (status == STATUS_OK && shares) {
		status = step_engram_share(driver);
		expect(status == STATUS_OK, "TP n-gram rows share at startup");
		expect(step_engram_share(driver) == STATUS_INVALID_ARGUMENT,
		       "sharing is attached once");
	}
	if (status == STATUS_OK) {
		const uint32_t depth = pipeline ? GROUP_QUEUE_DEPTH : 1u;
		if (pipeline) {
			status = step_pipeline_start(driver, depth);
			expect(status == STATUS_OK &&
				       step_pipeline_depth(driver) == depth,
			       "group coordinator starts without GPU work");
			for (uint32_t position = 0;
			     status == STATUS_OK && position < depth;
			     position++) {
				const struct step_row_record row = {
					(int32_t)position,
					(int32_t)position + 1, 0, STEP_ROW_FRONT
				};
				status = step_submit(driver, &row, 1u);
			}
			const struct step_row_record blocked = {
				(int32_t)depth, (int32_t)depth + 1, 0,
				STEP_ROW_FRONT
			};
			if (!fail_first) {
				expect(status == STATUS_OK &&
					       step_submit(driver, &blocked,
							   1u) ==
						       STATUS_BUSY_VALUE,
				       "submitted flights retain their bounded queue slots until finish");
			}
		}
		for (uint32_t position = 0;
		     status == STATUS_OK && position < depth; position++) {
			const struct step_row_record row = {
				(int32_t)position, (int32_t)position + 1, 0,
				STEP_ROW_FRONT
			};
			struct step_row_result result;
			status = pipeline ? step_finish(driver, &result) :
					    step_run_entry(driver, &row, 1u,
							   &result);
		}
		expect(status ==
			       (fail_first ? STATUS_INJECTED_FAULT : STATUS_OK),
		       "complete group preserves the first submit failure");
		expect(submitted == GROUP_MASK && !waited_early &&
			       !sequence_mismatch &&
			       groups_seen == (fail_first ? 1u : depth),
		       "each flight enqueues all members with one immutable sequence before waits or the next flight");
		for (uint32_t sequence = 0;
		     sequence < (fail_first ? 1u : depth); sequence++) {
			for (uint32_t card = 0; card < GROUP_CARDS; card++) {
				const uint32_t first = card < GROUP_MASK_CARDS ?
							       0u :
							       GROUP_MASK_CARDS;
				const uint32_t tables =
					card < GROUP_MASK_CARDS ?
						TEST_ENGRAM_PACK_TABLES :
						1u;
				for (uint32_t table = 0; table < tables;
				     table++) {
					const uintptr_t source =
						gathered_sources[sequence][card]
								[table];
					expect(source != 0 &&
						       ((source ==
							 gathered_sources
								 [sequence]
								 [first]
								 [table]) ==
							(shares ||
							 card == first)),
					       "only identical masks share gathered bytes");
					if (sequence != 0)
						expect(source !=
							       gathered_sources
								       [sequence -
									1u][card]
								       [table],
						       "queued flights retain distinct immutable staging blocks");
				}
			}
		}
		expect(step_engram_share(driver) ==
			       (fail_first ? STATUS_INJECTED_FAULT :
					     STATUS_INVALID_ARGUMENT),
		       "late sharing is refused and poison is preserved");
	}
	step_driver_destroy(driver);
	if (fail_first) {
		for (uint32_t card = 0; card < GROUP_CARDS; card++)
			expect(waits[card] != 0,
			       "every member is drained after a partial submit failure");
	}
	for (uint32_t card = 0; card < GROUP_CARDS; card++) {
		if (opened[card] != NULL)
			backend.card_close(opened[card], GROUP_WAIT_NS);
	}
}

int main(void)
{
	unsigned char *pack = NULL;
	expect(test_engram_pack_open(&pack, &engram) == STATUS_OK,
	       "n-gram fixture opens");
	expect(step_engram_share(NULL) == STATUS_INVALID_ARGUMENT,
	       "no driver is refused");
	for (uint32_t shares = 0; shares <= 1u; shares++) {
		run_case(false, false, shares != 0);
		run_case(true, false, shares != 0);
		run_case(false, true, shares != 0);
		run_case(true, true, shares != 0);
	}
	engram_close(engram);
	free(pack);
	return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
