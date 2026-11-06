/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* batch_segments_dump_test (gnmac_port diagnostics): the segments of BATCH programs of 8 and 12 rows prepared from one
 * EXL3 TP4 plan of the serving windows' BATCH recipe (head and draft shards, TP draft layers, the batched draft, the
 * wide head, TC MoE, trunk and head MMA, BF16 wire) on the host memory backend, side by side, and the step driver's
 * verdict on the pair. The driver takes BATCH programs whose segments (card, stage, layer, pause) match the first
 * one's: where the 8-row program (a thin walk) parts from the 12-row one is what a one-slab C1 program must align.
 * Always passes; its output is the evidence. */
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "anemone/step.h"
#include "host_memory_backend.h"
#include "model_v41_pack.h"
#include "model_v41_tp2_model.h"

#define DUMP_CARDS TP4_RANKS
#define DUMP_CARD_BYTES 4096u
#define DUMP_REPORTED_BYTES (1ull << 40u)
#define DUMP_SESSIONS 8u
#define DUMP_PROGRAMS 2u
#define DUMP_WAIT_NS 1000000000ull

int main(void)
{
	struct pack_record *pack = NULL;
	const struct test_model_v41_change change = { .kind = TEST_MODEL_V41_CHANGE_EXL3 };
	if (test_model_v41_draft_pack_open(change, &pack) != STATUS_OK) {
		printf("batch_segments_dump_test: the fixture did not open\n");
		return 0;
	}
	struct v41_tp2_model_description description = {
		.limits = { MODEL_CONTEXT_WITHOUT_CANDIDATES,
			    MODEL_WIDE_LEAST_ROWS, DUMP_SESSIONS,
			    STATE_WIDTH_16, LINEAR_PROFILE_PRECISION,
			    LINEAR_ORDER_FIXED, CMP_ORDER_REFERENCE,
			    MODEL_DRAFT_BLOCK, 0, 0 },
		.tensor_layout = V41_DATA_TP_LAYOUT_QUAD_ALL,
		.wire_format = ORDERED_PUSH_BF16,
		.head_shards = true,
		.draft_shards = true,
		.draft_layers = true,
		.exl3_fast_products = true,
		.exl3_wide_entry = true,
		.replicated_embedding = true,
		.batch = true,
		.batch_wide_head = true,
		.batch_draft_rows = true,
		.exl3_tc_moe = true,
		.trunk_mma = true,
		.head_mma = true,
		.wide_exchange = 1u
	};
	struct device_card *cards[DUMP_CARDS] = { 0 };
	struct card_description_record reported[DUMP_CARDS] = { 0 };
	test_host_memory_backend_reset(DUMP_CARDS, DUMP_CARD_BYTES);
	for (uint32_t card = 0; card < DUMP_CARDS; card++) {
		reported[card].arithmetic_class = ARITHMETIC_CLASS_SM_80;
		reported[card].memory_total_bytes = reported[card].memory_free_bytes = DUMP_REPORTED_BYTES;
		description.cards[card] = reported[card];
		test_host_memory_backend_set_arithmetic_class(
			card, ARITHMETIC_CLASS_SM_80);
		if (test_host_memory_backend.card_open(
			    card, DUMP_CARD_BYTES, &cards[card]) != STATUS_OK)
			printf("batch_segments_dump_test: card %" PRIu32 " did not open\n", card);
	}
	struct v41_tp2_model *model = NULL;
	struct v41_tp2_program_description build = { 0 };
	if (v41_tp2_model_create(pack, &description, &model) != STATUS_OK ||
	    v41_tp2_model_program_description(model, &build) != STATUS_OK) {
		printf("batch_segments_dump_test: the model did not prepare\n");
		return 0;
	}
	const uint32_t rows[DUMP_PROGRAMS] = { 12u, LINEAR_MAXIMUM_ROWS };
	struct execution_program *programs[DUMP_PROGRAMS] = { 0 };
	for (uint32_t index = 0; index < DUMP_PROGRAMS; index++) {
		struct step_companion_program companions = { 0 };
		const result_status prepared = v41_tp_program_prepare(
			&build, &test_host_memory_backend, cards, reported,
			PROGRAM_BATCH, rows[index], &programs[index],
			&companions);
		if (prepared != STATUS_OK || programs[index] == NULL) {
			printf("BATCH %" PRIu32 " rows: did not prepare (%s)\n",
			       rows[index], status_text(prepared));
			return 0;
		}
		const struct execution_program *const program = programs[index];
		printf("BATCH %" PRIu32 " rows: %" PRIu32 " segments, front %" PRIu32 ", draft from %" PRIu32 "\n",
		       rows[index], program->segment_count, program->front_segment_count, program->draft_segment_first);
	}
	const uint32_t most = programs[0]->segment_count > programs[1]->segment_count ? programs[0]->segment_count :
											  programs[1]->segment_count;
	for (uint32_t index = 0; index < most; index++) {
		printf("%4" PRIu32, index);
		for (uint32_t program = 0; program < DUMP_PROGRAMS; program++) {
			if (index >= programs[program]->segment_count) {
				printf(" | %-36s", "-");
				continue;
			}
			const struct segment_record *const segment =
				&programs[program]->segments[index];
			printf(" | card %" PRIu32 " stage %" PRIu32 " layer %10" PRIu32 " pause %" PRIu32 " n %4" PRIu32,
			       segment->card, segment->stage, segment->layer, segment->pause, segment->launch_count);
		}
		const bool differs = index >= programs[0]->segment_count || index >= programs[1]->segment_count ||
				     programs[0]->segments[index].card != programs[1]->segments[index].card ||
				     programs[0]->segments[index].stage != programs[1]->segments[index].stage ||
				     programs[0]->segments[index].layer != programs[1]->segments[index].layer ||
				     programs[0]->segments[index].pause != programs[1]->segments[index].pause;
		printf("%s\n", differs ? "  <<" : "");
	}
	for (uint32_t index = 0; index < DUMP_PROGRAMS; index++) /* no engram tables on the host backend */
		for (uint32_t card = 0; card < programs[index]->card_count; card++) {
			const uint32_t none = 0;
			memcpy((void *)(uintptr_t)&programs[index]->cards[card].engram_tables, &none, sizeof(none));
		}
	struct step_driver *driver = NULL;
	const result_status created = step_driver_create(
		&test_host_memory_backend, cards,
		(const struct execution_program *const *)programs,
		DUMP_PROGRAMS, NULL, DUMP_WAIT_NS, &driver);
	printf("the driver over BATCH 12 then 8: %s\n", status_text(created));
	/* the 8-row program's launches from each card's last sampler on (where pf11's device error follows) */
	for (uint32_t card = 0; card < programs[1]->card_count; card++) {
		const struct program_card_record *const own =
			&programs[1]->cards[card];
		uint32_t last = own->launch_count;
		for (uint32_t index = 0; index < own->launch_count; index++) {
			struct test_host_memory_backend_kernel kernel;
			if (test_host_memory_backend_kernel_of(card, own->launches[index].kernel, &kernel) &&
			    strstr(kernel.function_name, "sampler") != NULL)
				last = index;
		}
		printf("card %" PRIu32 ": %" PRIu32 " launches, the last sampler at %" PRIu32 "\n", card, own->launch_count,
		       last);
		for (uint32_t index = last; index < own->launch_count && index < last + 40u; index++) {
			struct test_host_memory_backend_kernel kernel;
			const bool named = test_host_memory_backend_kernel_of(card, own->launches[index].kernel, &kernel);
			printf("  %5" PRIu32 " %s grid %u,%u,%u\n", index,
			       own->launches[index].kernel == BACKEND_COPY ?
				       "(copy)" :
			       named ? kernel.function_name :
				       "?",
			       own->launches[index].grid[0],
			       own->launches[index].grid[1],
			       own->launches[index].grid[2]);
		}
	}
	step_driver_destroy(driver);
	for (uint32_t index = 0; index < DUMP_PROGRAMS; index++)
		free(programs[index]);
	v41_tp2_model_destroy(model);
	pack_close(pack);
	for (uint32_t card = 0; card < DUMP_CARDS; card++)
		if (cards[card] != NULL)
			test_host_memory_backend.card_close(cards[card], DUMP_WAIT_NS);
	printf("batch_segments_dump_test passed\n");
	return 0;
}
