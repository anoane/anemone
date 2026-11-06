/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* draft_tail_dump_test (gnmac_port diagnostics): the draft tail of a THIN program and of a BATCH program of 12 rows
 * (per-verifier draft walks, no draft shards) prepared from one EXL3 TP4 plan on the host memory backend, the BF16
 * recipe of the probes (head shards, TC MoE, trunk and head MMA, BF16 wire), printed launch by launch on the result
 * card: function, grid, block, shared bytes and the argument record in hex. The two programs share the plan's arrays,
 * so a launch of the draft's layers that differs between them beyond its verify, result and round areas is where the
 * BATCH drafts go wrong. Always passes; its output is the evidence. */
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
#define DUMP_THIN_ROWS 6u
#define DUMP_BATCH_ROWS 12u
#define DUMP_WAIT_NS 1000000000ull

static void dump(const struct execution_program *program, const char *name)
{
	const uint32_t card = program->result_card;
	uint32_t first = program->cards[card].launch_count;
	for (uint32_t segment = program->draft_segment_first; segment < program->segment_count; segment++)
		if (program->segments[segment].card == card && program->segments[segment].first_launch < first)
			first = program->segments[segment].first_launch;
	printf("== %s: result card %" PRIu32 ", draft tail from launch %" PRIu32 " of %" PRIu32 "\n", name, card, first,
	       program->cards[card].launch_count);
	for (uint32_t index = first; index < program->cards[card].launch_count; index++) {
		const struct kernel_launch *const launch =
			&program->cards[card].launches[index];
		struct test_host_memory_backend_kernel kernel;
		const bool named = test_host_memory_backend_kernel_of(card, launch->kernel, &kernel);
		printf("%4" PRIu32 " %-44s grid %u,%u,%u block %u,%u,%u shared %u args", index - first,
		       named ? kernel.function_name : "?", launch->grid[0], launch->grid[1], launch->grid[2],
		       launch->block[0], launch->block[1], launch->block[2], launch->dynamic_shared_byte_count);
		for (uint32_t byte = 0; byte < launch->argument_byte_count; byte++)
			printf("%s%02x", byte % 8u == 0 ? " " : "", launch->argument[byte]);
		printf("\n");
	}
}

int main(void)
{
	struct pack_record *pack = NULL;
	const struct test_model_v41_change change = { .kind = TEST_MODEL_V41_CHANGE_EXL3 };
	if (test_model_v41_draft_pack_open(change, &pack) != STATUS_OK) {
		printf("draft_tail_dump_test: the fixture did not open\n");
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
		.exl3_fast_products = true,
		.exl3_wide_entry = true,
		.replicated_embedding = true,
		.batch = true,
		.exl3_tc_moe = true,
		.trunk_mma = true,
		.head_mma = true,
		.graph_branches = 1u
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
			printf("draft_tail_dump_test: card %" PRIu32 " did not open\n", card);
	}
	struct v41_tp2_model *model = NULL;
	struct v41_tp2_program_description build = { 0 };
	if (v41_tp2_model_create(pack, &description, &model) != STATUS_OK ||
	    v41_tp2_model_program_description(model, &build) != STATUS_OK) {
		printf("draft_tail_dump_test: the model did not prepare\n");
		v41_tp2_model_destroy(model);
		pack_close(pack);
		return 0;
	}
	const enum program_kind kinds[] = { PROGRAM_THIN, PROGRAM_BATCH };
	const uint32_t rows[] = { DUMP_THIN_ROWS, DUMP_BATCH_ROWS };
	const char *const names[] = { "THIN", "BATCH" };
	for (uint32_t kind = 0; kind < 2u; kind++) {
		struct execution_program *program = NULL;
		struct step_companion_program companions = { 0 };
		const result_status prepared = v41_tp_program_prepare(
			&build, &test_host_memory_backend, cards, reported,
			kinds[kind], rows[kind], &program, &companions);
		if (prepared != STATUS_OK || program == NULL) {
			printf("== %s: did not prepare (%s)\n", names[kind],
			       status_text(prepared));
			continue;
		}
		dump(program, names[kind]);
		free(program);
	}
	/* everything back, so that the sanitizers' leak check passes */
	v41_tp2_model_destroy(model);
	pack_close(pack);
	for (uint32_t card = 0; card < DUMP_CARDS; card++)
		if (cards[card] != NULL)
			test_host_memory_backend.card_close(cards[card], DUMP_WAIT_NS);
	printf("draft_tail_dump_test passed\n");
	return 0;
}
