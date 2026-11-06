/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* batch_plan_driver_test (gnmac_port, track A): the BATCH programs of the step plan's serve window (gsa04:
 * --batch_rows=8,12,16,24,32,48 --batch_verifiers=1,2,4,4,8,8, TP draft layers, the batched draft, the wide head,
 * 1,024-row wide and full programs) prepared on the host memory backend, taken by one step driver with its TP groups
 * and verifier counts attached; then the program each drawing step of S drafting sessions and R rows takes
 * (step_image_programs.c): the fewest rows holding R whose verifiers hold S, so a C8 step of 4 rows a session takes
 * BATCH-32 and of 6 rows BATCH-48. Then the same with track D's layouts: --batch_verifiers=2,4,4,4,8,8 (a C2 step of
 * 4 rows a session takes BATCH-8, a thin walk drafting for two sessions, and a C4 step of 3 rows BATCH-12) and
 * --batch_rows=12,8,8,12,16,24,32,48 --batch_verifiers=4,2,1,2,4,4,8,8 (equal rows told apart by their verifiers: C1
 * takes BATCH-8 of one draft walk, C2 of two; step_batch_layout lists them in the order steps look). */
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
#include "step_driver.h"

#define PLAN_CARDS TP4_RANKS
#define PLAN_CARD_BYTES 4096u
#define PLAN_REPORTED_BYTES (1ull << 40u)
#define PLAN_SESSIONS 8u
#define PLAN_WIDE_ROWS 1024u
#define PLAN_MOST_BATCH 8u /* main.c's MOST_BATCH_PROGRAMS */
#define PLAN_CHUNK_PROGRAMS \
	6u /* wide and full programs of 1,024, 256 and 64 rows */
#define PLAN_MOST_PROGRAMS (PLAN_MOST_BATCH + PLAN_CHUNK_PROGRAMS)
#define PLAN_WAIT_NS 1000000000ull
#define PLAN_DRAFT_ROWS (MODEL_DRAFT_BLOCK + 1u)
#define PLAN_LAYOUTS 3u
#define PLAN_CASES 8u

/* A drawing step of sessions drafting sessions, rows_each rows a session, and the program (rows, verifiers) it
 * takes. */
struct plan_case {
	uint32_t sessions, rows_each, expected_rows, expected_verifiers;
};
/* --batch_rows / --batch_verifiers: batch_count BATCH programs, and the steps to check */
struct plan_layout {
	uint32_t batch_count, rows[PLAN_MOST_BATCH], verifiers[PLAN_MOST_BATCH];
	struct plan_case cases[PLAN_CASES];
};

static uint32_t failures;

static void expect(bool condition, const char *what)
{
	if (!condition) {
		printf("batch_plan_driver_test: %s\n", what);
		failures++;
	}
}

/* The program a drawing step of sessions drafting sessions, rows_each rows a session, takes. */
static const struct execution_program *
chosen_program(const struct step_driver *driver, uint32_t sessions,
	       uint32_t rows_each)
{
	struct step_row_record rows[STEP_BATCH_MAXIMUM_ROWS];
	uint32_t count = 0;
	for (uint32_t session = 0; session < sessions; session++)
		for (uint32_t row = 0;
		     row < rows_each && count < STEP_BATCH_MAXIMUM_ROWS; row++)
			rows[count++] = (struct step_row_record){
				(int32_t)(MODEL_CONE_ROWS + row), 1, session,
				STEP_ROW_LOGITS | (row == 0 ? STEP_ROW_DRAFTS :
							      STEP_ROW_PROPOSED)
			};
	const uint32_t *graphs = NULL;
	return step_program_for_rows(driver, rows, count, &graphs);
}

int main(void)
{
	struct pack_record *pack = NULL;
	const struct test_model_v41_change change = {
		.kind = TEST_MODEL_V41_CHANGE_EXL3
	};
	if (test_model_v41_draft_pack_open(change, &pack) != STATUS_OK) {
		printf("batch_plan_driver_test: the fixture did not open\n");
		return 1;
	}
	struct v41_tp2_model_description description = {
		.limits = { MODEL_CONTEXT_WITHOUT_CANDIDATES, PLAN_WIDE_ROWS,
			    PLAN_SESSIONS, STATE_WIDTH_16,
			    LINEAR_PROFILE_PRECISION, LINEAR_ORDER_FIXED,
			    CMP_ORDER_REFERENCE, MODEL_DRAFT_BLOCK, 0, 0 },
		.tensor_layout = V41_DATA_TP_LAYOUT_QUAD_ALL,
		.wire_format = ORDERED_PUSH_BF16,
		.head_shards = true,
		.head_stream = true,
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
		.ordered_wide = true,
		.graph_branches = 1u,
		.wide_exchange = V41_DATA_TP_EXCHANGE_CE_RINGS,
		.split_overlap = V41_DATA_SPLIT_PIPELINE,
		.chunk_core = true
	};
	struct device_card *cards[PLAN_CARDS] = { 0 };
	struct card_description_record reported[PLAN_CARDS] = { 0 };
	test_host_memory_backend_reset(PLAN_CARDS, PLAN_CARD_BYTES);
	for (uint32_t card = 0; card < PLAN_CARDS; card++) {
		reported[card].arithmetic_class = ARITHMETIC_CLASS_SM_80;
		reported[card].memory_total_bytes =
			reported[card].memory_free_bytes = PLAN_REPORTED_BYTES;
		description.cards[card] = reported[card];
		test_host_memory_backend_set_arithmetic_class(
			card, ARITHMETIC_CLASS_SM_80);
		expect(test_host_memory_backend.card_open(card, PLAN_CARD_BYTES,
							  &cards[card]) ==
			       STATUS_OK,
		       "a card opens");
	}
	struct v41_tp2_model *model = NULL;
	struct v41_tp2_program_description build = { 0 };
	if (v41_tp2_model_create(pack, &description, &model) != STATUS_OK ||
	    v41_tp2_model_program_description(model, &build) != STATUS_OK) {
		printf("batch_plan_driver_test: the model did not prepare\n");
		return 1;
	}
	/* each layout's BATCH programs (rows, verifiers) and its steps: drafting sessions, rows a session, and the
	 * program (rows, verifiers) the step takes */
	static const struct plan_layout layouts[PLAN_LAYOUTS] = {
		{ 6u,
		  { 8u, 12u, 16u, 24u, 32u, 48u },
		  { 1u, 2u, 4u, 4u, 8u, 8u },
		  { { 1u, PLAN_DRAFT_ROWS, 8u, 1u },
		    { 2u, PLAN_DRAFT_ROWS, 12u, 2u },
		    { 4u, 4u, 16u, 4u },
		    { 4u, PLAN_DRAFT_ROWS, 24u, 4u },
		    { 8u, 3u, 32u, 8u },
		    { 8u, 4u, 32u, 8u },
		    { 8u, PLAN_DRAFT_ROWS, 48u, 8u },
		    { 3u, 2u, 16u, 4u } } },
		{ 6u,
		  { 8u, 12u, 16u, 24u, 32u, 48u },
		  { 2u, 4u, 4u, 4u, 8u, 8u },
		  { { 1u, PLAN_DRAFT_ROWS, 8u, 2u },
		    { 2u, 4u, 8u, 2u },
		    { 2u, PLAN_DRAFT_ROWS, 12u, 4u },
		    { 4u, 3u, 12u, 4u },
		    { 4u, 4u, 16u, 4u },
		    { 4u, PLAN_DRAFT_ROWS, 24u, 4u },
		    { 8u, 3u, 32u, 8u },
		    { 8u, PLAN_DRAFT_ROWS, 48u, 8u } } },
		/* equal rows told apart by verifiers, listed out of order: C1 takes BATCH-8 of one, C2 of two */
		{ 8u,
		  { 12u, 8u, 8u, 12u, 16u, 24u, 32u, 48u },
		  { 4u, 2u, 1u, 2u, 4u, 4u, 8u, 8u },
		  { { 1u, PLAN_DRAFT_ROWS, 8u, 1u },
		    { 2u, 4u, 8u, 2u },
		    { 2u, PLAN_DRAFT_ROWS, 12u, 2u },
		    { 4u, 3u, 12u, 4u },
		    { 3u, 4u, 12u, 4u },
		    { 4u, PLAN_DRAFT_ROWS, 24u, 4u },
		    { 8u, 3u, 32u, 8u },
		    { 8u, PLAN_DRAFT_ROWS, 48u, 8u } } }
	};
	for (uint32_t layout = 0; layout < PLAN_LAYOUTS; layout++) {
		const struct plan_layout *const own = &layouts[layout];
		const uint32_t program_count =
			own->batch_count + PLAN_CHUNK_PROGRAMS;
		const uint32_t chunk_rows[PLAN_CHUNK_PROGRAMS / 2u] = {
			PLAN_WIDE_ROWS, PLAN_WIDE_ROWS / 4u,
			MODEL_WIDE_LEAST_ROWS
		};
		struct execution_program *programs[PLAN_MOST_PROGRAMS] = { 0 };
		struct step_companion_program companions[PLAN_MOST_PROGRAMS] = {
			0
		};
		uint32_t verifiers[PLAN_MOST_PROGRAMS] = { 0 };
		for (uint32_t index = 0; index < program_count; index++) {
			const bool batch = index < own->batch_count;
			const uint32_t chunk =
				batch ? 0 :
					(index - own->batch_count) %
						(PLAN_CHUNK_PROGRAMS / 2u);
			const enum program_kind kind =
				batch ? PROGRAM_BATCH :
				index - own->batch_count <
						PLAN_CHUNK_PROGRAMS / 2u ?
					PROGRAM_WIDE :
					PROGRAM_FULL;
			const uint32_t rows = batch ? own->rows[index] :
						      chunk_rows[chunk];
			build.batch_verifiers = batch ? own->verifiers[index] :
							0;
			verifiers[index] = build.batch_verifiers;
			const result_status prepared = v41_tp_program_prepare(
				&build, &test_host_memory_backend, cards,
				reported, kind, rows, &programs[index],
				&companions[index]);
			printf("layout %" PRIu32 " program %" PRIu32
			       " (kind %u, %" PRIu32 " rows, %" PRIu32
			       " verifiers): %s\n",
			       layout, index, (unsigned)kind, rows,
			       verifiers[index], status_text(prepared));
			expect(prepared == STATUS_OK && programs[index] != NULL,
			       "every program prepares");
			if (prepared != STATUS_OK || programs[index] == NULL)
				return 1;
			for (uint32_t card = 0;
			     card < programs[index]->card_count; card++) {
				const uint32_t none =
					0; /* no engram tables on the host backend */
				memcpy((void *)(uintptr_t)&programs[index]
					       ->cards[card]
					       .engram_tables,
				       &none, sizeof(none));
			}
		}
		struct step_driver *driver = NULL;
		const result_status created = step_driver_create(
			&test_host_memory_backend, cards,
			(const struct execution_program *const *)programs,
			program_count, NULL, PLAN_WAIT_NS, &driver);
		expect(created == STATUS_OK,
		       "the driver takes the BATCH, wide and full programs");
		expect(created != STATUS_OK ||
			       step_groups_attach(driver, companions,
						  program_count,
						  PLAN_CARDS) == STATUS_OK,
		       "the TP groups attach");
		if (created == STATUS_OK && layout + 1u == PLAN_LAYOUTS) {
			/* two BATCH programs of one row count must verify for different counts of sessions */
			uint32_t equal[PLAN_MOST_PROGRAMS];
			memcpy(equal, verifiers, sizeof(equal));
			equal[1] = equal[2];
			expect(step_batch_verifiers_attach(driver, equal,
							   program_count) ==
				       STATUS_INVALID_ARGUMENT,
			       "equal rows of equal verifiers refused");
		}
		expect(created != STATUS_OK ||
			       step_batch_verifiers_attach(driver, verifiers,
							   program_count) ==
				       STATUS_OK,
		       "the verifier counts attach");
		if (created == STATUS_OK) {
			for (uint32_t index = 0; index < PLAN_CASES; index++) {
				const struct plan_case *const step =
					&own->cases[index];
				const struct execution_program *const program =
					chosen_program(driver, step->sessions,
						       step->rows_each);
				uint32_t taken_verifiers = 0;
				for (uint32_t other = 0; other < program_count;
				     other++)
					taken_verifiers =
						programs[other] == program ?
							verifiers[other] :
							taken_verifiers;
				printf("%" PRIu32
				       " drafting sessions of %" PRIu32
				       " rows: BATCH-%" PRIu32 " of %" PRIu32
				       " verifiers\n",
				       step->sessions, step->rows_each,
				       program != NULL ? program->row_count : 0,
				       taken_verifiers);
				expect(program != NULL &&
					       program->row_count ==
						       step->expected_rows &&
					       taken_verifiers ==
						       step->expected_verifiers,
				       "a step takes the fewest rows, then verifiers, holding its rows and sessions");
			}
			uint32_t layout_rows[PLAN_MOST_BATCH],
				layout_verifiers[PLAN_MOST_BATCH];
			const uint32_t count = step_batch_layout(
				driver, layout_rows, layout_verifiers,
				PLAN_MOST_BATCH);
			expect(count == own->batch_count,
			       "the layout lists every BATCH program");
			for (uint32_t index = 0; index < count; index++) {
				expect(index == 0 ||
					       layout_rows[index - 1u] <
						       layout_rows[index] ||
					       (layout_rows[index - 1u] ==
							layout_rows[index] &&
						layout_verifiers[index - 1u] <
							layout_verifiers[index]),
				       "the layout runs fewest rows, then verifiers, first");
				uint32_t fewest = UINT32_MAX;
				for (uint32_t other = 0;
				     other < own->batch_count; other++)
					fewest =
						own->rows[other] == layout_rows[index] &&
								own->verifiers[other] <
									fewest ?
							own->verifiers[other] :
							fewest;
				expect(step_batch_verifiers(
					       driver, layout_rows[index]) ==
					       fewest,
				       "the driver answers the fewest verifiers of a row count");
			}
			expect(step_batch_layout(driver, layout_rows,
						 layout_verifiers,
						 own->batch_count - 1u) == 0,
			       "a layout longer than asked for answers none");
		}
		step_driver_destroy(driver);
		for (uint32_t index = 0; index < program_count; index++)
			free(programs[index]);
	}
	v41_tp2_model_destroy(model);
	pack_close(pack);
	for (uint32_t card = 0; card < PLAN_CARDS; card++)
		if (cards[card] != NULL)
			test_host_memory_backend.card_close(cards[card],
							    PLAN_WAIT_NS);
	printf("batch_plan_driver_test %s\n",
	       failures == 0 ? "passed" : "failed");
	return failures == 0 ? 0 : 1;
}
