/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Startup descriptor, immutable ownership and refusal checks. No device is opened or run. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "models/deepseek_v41_flash/plan/model_v41.h"
#include "engine/runtime/step/step.h"

#define TEST_PROGRAMS 3u
#define TEST_BANKS 5u
#define TEST_KERNELS 64u
#define TEST_ARENA_BYTES (32ull << 20u)
#define TEST_RANGE_STRIDE (6ull << 20u)
#define TEST_SEGMENTS (DS41F_HEAD_RANKS + 2u)
#define TEST_RECORDS (DS41F_HEAD_SAMPLE * DS41F_HEAD_RANKS + 1u)
#define TEST_RANK_LAUNCHES 2u /* a rank card's launches and its segment's */

struct device_card {
	uint32_t index;
};
static struct {
	const char *file, *name;
	uint32_t bytes;
} kernels[TEST_KERNELS];
static uint32_t kernel_count, failures;

static void expect(bool accepted, const char *message)
{
	if (!accepted) {
		(void)fprintf(stderr, "FAILED: %s\n", message);
		failures++;
	}
}

MUST_CHECK static result_status prepare_kernel(struct device_card *card,
					       const char *file,
					       const char *name, uint32_t bytes,
					       uint32_t *kernel_out)
{
	if (card == NULL || file == NULL || name == NULL || kernel_out == NULL)
		return STATUS_INVALID_ARGUMENT;
	for (uint32_t index = 0; index < kernel_count; index++) {
		if (strcmp(kernels[index].file, file) == 0 &&
		    strcmp(kernels[index].name, name) == 0) {
			if (kernels[index].bytes != bytes)
				return STATUS_INVALID_ARGUMENT;
			*kernel_out = index;
			return STATUS_OK;
		}
	}
	if (kernel_count == TEST_KERNELS)
		return STATUS_OUT_OF_MEMORY;
	kernels[kernel_count].file = file;
	kernels[kernel_count].name = name;
	kernels[kernel_count].bytes = bytes;
	*kernel_out = kernel_count;
	kernel_count++;
	return STATUS_OK;
}

int main(void)
{
	struct ds41f_plan *const plan = calloc(1, sizeof(*plan));
	if (plan == NULL)
		return EXIT_FAILURE;
	plan->card_count = DS41F_HEAD_CARDS;
	plan->result_card = DS41F_HEAD_RANKS;
	plan->limits.row_capacity = LINEAR_MAXIMUM_ROWS;
	plan->limits.session_capacity = TEST_BANKS;
	static const uint32_t last[] = { 8, 15, 22, 29, 40 };
	for (uint32_t layer = 0, card = 0; layer < DS41F_LAYERS; layer++) {
		card += layer == last[card] ? 1u : 0u;
		plan->layer_card[layer] = card;
	}
	struct device_card devices[DS41F_HEAD_CARDS];
	struct device_card *cards[DS41F_HEAD_CARDS];
	struct card_description_record descriptions[DS41F_HEAD_CARDS] = { 0 };
	for (uint32_t card = 0; card < DS41F_HEAD_CARDS; card++) {
		devices[card].index = card;
		cards[card] = &devices[card];
		const bool pro = card == DS41F_HEAD_RANKS;
		const char *const name =
			pro ? "NVIDIA RTX PRO 6000 Blackwell Workstation Edition" :
			      "NVIDIA CMP 170HX";
		memcpy(descriptions[card].name, name, strlen(name) + 1u);
		descriptions[card].arithmetic_class =
			pro ? ARITHMETIC_CLASS_SM_120_NATIVE :
			      ARITHMETIC_CLASS_SM_80;
		plan->cards[card].arena_byte_count = TEST_ARENA_BYTES;
		static const uint32_t arrays[] = { DS41F_ARRAY_HIDDEN,
						   DS41F_ARRAY_LOGITS,
						   DS41F_ARRAY_SAMPLER_WEIGHTS,
						   DS41F_ARRAY_SESSIONS,
						   DS41F_ARRAY_RESULT };
		for (size_t index = 0;
		     index < sizeof(arrays) / sizeof(arrays[0]); index++)
			plan->cards[card].array_offset[arrays[index]] =
				(index + 1u) * TEST_RANGE_STRIDE;
	}
	expect(ds41f_original_head_plan(plan, descriptions) == STATUS_OK,
	       "canonical head storage plans");
	expect(ds41f_original_head_banks_plan(plan, TEST_BANKS) == STATUS_OK,
	       "five bank planes plan");
	const struct backend_operations backend = { .kernel_prepare =
							    prepare_kernel };
	struct execution_program programs[TEST_PROGRAMS] = { 0 };
	struct program_card_record program_cards[TEST_PROGRAMS]
						[DS41F_HEAD_CARDS] = { 0 };
	struct kernel_launch launches[TEST_PROGRAMS][DS41F_HEAD_CARDS]
				     [DS41F_HEAD_RANKS + 1u] = { 0 };
	struct segment_record segments[TEST_PROGRAMS][TEST_SEGMENTS] = { 0 };
	const struct execution_program *pointers[TEST_PROGRAMS];
	for (uint32_t program = 0; program < TEST_PROGRAMS; program++) {
		pointers[program] = &programs[program];
		programs[program] = (struct execution_program){
			.cards = program_cards[program],
			.segments = segments[program],
			.limits = plan->limits,
			.card_count = DS41F_HEAD_CARDS,
			.segment_count = TEST_SEGMENTS,
			.row_count = program == 0 ? 1u : LINEAR_MAXIMUM_ROWS,
			.result_card = DS41F_HEAD_RANKS,
			.kind = program < TEST_PROGRAMS - 1u ? PROGRAM_THIN :
							       PROGRAM_FULL,
			.draft_segment_first = TEST_SEGMENTS
		};
		for (uint32_t card = 0; card < DS41F_HEAD_CARDS; card++) {
			program_cards[program][card].arena_byte_count =
				plan->cards[card].arena_byte_count;
			program_cards[program][card].launches =
				launches[program][card];
			program_cards[program][card].launch_count =
				card == DS41F_HEAD_RANKS ?
					DS41F_HEAD_RANKS + 1u :
					TEST_RANK_LAUNCHES;
		}
		segments[program][0] =
			(struct segment_record){ .card = DS41F_HEAD_RANKS,
						 .layer = SEGMENT_NO_LAYER,
						 .launch_count =
							 DS41F_HEAD_RANKS };
		for (uint32_t rank = 0; rank < DS41F_HEAD_RANKS; rank++) {
			segments[program][rank + 1u] = (struct segment_record){
				.card = rank,
				.layer = SEGMENT_NO_LAYER,
				.launch_count = TEST_RANK_LAUNCHES
			};
			for (enum ds41f_head_launch_kind kind =
				     DS41F_HEAD_BROADCAST;
			     kind < DS41F_HEAD_SAMPLE; kind++) {
				const uint32_t card =
					kind == DS41F_HEAD_BROADCAST ?
						DS41F_HEAD_RANKS :
						rank;
				const uint32_t index =
					kind == DS41F_HEAD_BROADCAST ?
						rank :
						(kind == DS41F_HEAD_DENSE ? 0u :
									    1u);
				expect(ds41f_original_head_launch(
					       plan, &backend, cards, kind,
					       rank, 0,
					       programs[program].row_count,
					       &launches[program][card][index]) ==
					       STATUS_OK,
				       "base head descriptor prepares");
			}
		}
		segments[program][TEST_SEGMENTS - 1u] =
			(struct segment_record){ .card = DS41F_HEAD_RANKS,
						 .layer = SEGMENT_NO_LAYER,
						 .first_launch =
							 DS41F_HEAD_RANKS,
						 .launch_count = 1 };
		expect(ds41f_original_head_launch(
			       plan, &backend, cards, DS41F_HEAD_SAMPLE, 0, 0,
			       programs[program].row_count,
			       &launches[program][DS41F_HEAD_RANKS]
					[DS41F_HEAD_RANKS]) == STATUS_OK,
		       "base sampler descriptor prepares");
	}
	struct ds41f_head_program_banks *banks = NULL;
	expect(ds41f_original_head_banks_create(plan, &backend, cards, pointers,
						NULL, TEST_PROGRAMS,
						&banks) == STATUS_OK,
	       "immutable banks create over both thin widths and a shared full program");
	if (banks != NULL) {
		const struct step_flight_programs *const tables =
			ds41f_original_head_bank_tables(banks);
		for (uint32_t bank = 0; bank < TEST_BANKS; bank++) {
			uint32_t changed = 0;
			for (uint32_t program = 0; program < TEST_PROGRAMS;
			     program++) {
				const struct execution_program *const made =
					tables[bank].programs[program];
				expect((made == pointers[program]) ==
					       (bank == 0 ||
						program == TEST_PROGRAMS - 1u),
				       "bank0 and full programs borrow originals; other thin programs are distinct");
				expect(made->leases ==
					       pointers[program]->leases,
				       "immutable leases are shared");
				if (program == TEST_PROGRAMS - 1u)
					continue;
				for (uint32_t card = 0; card < DS41F_HEAD_CARDS;
				     card++)
					for (uint32_t index = 0;
					     index <
					     program_cards[program][card]
						     .launch_count;
					     index++)
						changed +=
							memcmp(&made->cards[card]
									.launches[index],
							       &launches[program]
									[card]
									[index],
							       sizeof(struct kernel_launch)) !=
									0 ?
								1u :
								0u;
			}
			expect(changed == (bank == 0 ? 0u :
						       (TEST_PROGRAMS - 1u) *
							       TEST_RECORDS),
			       "every one of the thirteen head records per thin program is relocated, and bank0 is "
			       "untouched");
		}
		(void)printf(
			"head banks retained heap: %zu bytes; two thin programs, five banks, thirteen records each\n",
			ds41f_original_head_banks_memory(banks).heap_bytes);
	}
	(void)printf(
		"host memory type sizes: pointer=%zu lease=%zu kernel_launch=%zu\n",
		sizeof(void *), sizeof(struct visit_lease),
		sizeof(struct kernel_launch));
	ds41f_original_head_banks_destroy(banks);
	banks = NULL;
	launches[0][0][0].grid[0]++;
	expect(ds41f_original_head_banks_create(plan, &backend, cards, pointers,
						NULL, TEST_PROGRAMS, &banks) ==
			       STATUS_INVALID_ARGUMENT &&
		       banks == NULL,
	       "a changed head descriptor refuses without publishing partial banks");
	free(plan);
	(void)printf("model_v41_head_banks_test: %s\n",
		     failures == 0 ? "passed" : "FAILED");
	return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
