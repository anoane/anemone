/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * subset_placement_test: the placement of a layer subset (subset_placement.h) on the tables of DeepSeek-V4.1-Flash's
 * pack (model_v41_pack.h), for the four cards the teacher-forced card test has until the fifth: three of class SM_80
 * with budgets of 61.7 GiB and one of class SM_120 with 93.2 GiB, 1 GiB kept free on each.
 *
 *   - Sources: a layer brings the source and the index layer it reads (16: 14; 37: 20 and 36; 3: 2), and no other.
 *   - Layers 0 to 35: each run layer's stage unit with all its experts on one card, each other layer's with at least
 *     its first, the stage units' cards in the order of the pack, no card above its budget, the embedding and the head
 *     on a card; the plan's copy of the layout puts every expert of a layer that stayed in host memory where the
 *     layer's first lies, and the loader's layout leaves it in host memory; plan_create takes the copy (every layer on
 *     its stage unit's card, every card's arena within its budget and what is kept free); program_prepare makes the
 *     program on cards in host memory, and the program view reads every run layer on its card.
 *   - The same for layers 0 to 36; all 40 are refused, out of memory, on these budgets.
 *   - Refused: a missing argument, no card, a layer beyond the model's; a plan layout of a placement never made.
 */
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "anemone/model_layout.h"
#include "host_memory_backend.h"
#include "memory_storage.h"
#include "model_v41.h"
#include "model_v41_pack.h"
#include "program_view.h"
#include "subset_placement.h"

#define CARDS 4u
#define RESULT_CARD 3u                          /* the RTX PRO 6000, last in the card test's order */
#define SMALL_BUDGET_MIB 63181ull               /* 61.7 GiB: a CMP 170HX's free memory less the reserve and 1 GiB */
#define LARGE_BUDGET_MIB 95437ull               /* 93.2 GiB: the RTX PRO 6000's */
#define KEPT_FREE_MIB 1024ull
#define BYTES_PER_MIB 1048576ull
#define CARD_BYTES 4096u                        /* no launch runs: a program only names offsets */
#define CARD_MEMORY_BYTES 4194304ull
#define WAIT_LIMIT_NANOSECONDS 1000000000ull
#define FEWER_LAYERS 36u                        /* 0 to 35 */
#define MORE_LAYERS 37u                         /* 0 to 36 */
#define BIT(index) (1ull << (index))
#define FIRST_LAYERS(count) (BIT(count) - 1u)
#define EVERY_LAYER FIRST_LAYERS(MODEL_LAYERS)
#define BEYOND_THE_MODEL BIT(MODEL_LAYERS)
#define BREAK_LAYER 16u                         /* reads the pool and the index of 14 */
#define BREAK_SOURCE 14u
#define LATE_LAYER 37u                          /* the pool of 20, the index of 36 */
#define LATE_SOURCE 20u
#define LATE_INDEX 36u
#define EARLY_LAYER 3u                          /* both of 2 */
#define EARLY_SOURCE 2u

static const struct backend_operations *const backend =
	&test_host_memory_backend;
static int failures = 0;

static void expect(bool condition, const char *what)
{
    if (!condition) {
        (void)fprintf(stderr, "FAILED: %s\n", what);
        failures++;
    }
}

static void test_sources(void)
{
    expect(test_subset_layers_with_sources(BIT(BREAK_LAYER)) == (BIT(BREAK_LAYER) | BIT(BREAK_SOURCE)) &&
               test_subset_layers_with_sources(BIT(LATE_LAYER)) ==
                   (BIT(LATE_LAYER) | BIT(LATE_INDEX) | BIT(LATE_SOURCE)) &&
               test_subset_layers_with_sources(BIT(EARLY_LAYER)) == (BIT(EARLY_LAYER) | BIT(EARLY_SOURCE)) &&
               test_subset_layers_with_sources(BIT(0) | BIT(1)) == (BIT(0) | BIT(1)) &&
               test_subset_layers_with_sources(FIRST_LAYERS(FEWER_LAYERS)) == FIRST_LAYERS(FEWER_LAYERS) &&
               test_subset_layers_with_sources(EVERY_LAYER) == EVERY_LAYER,
           "a layer brings the source and the index layer it reads, and no other");
}

/* The placement's rules, unit by unit; the plan's copy against the loader's layout. */
static bool placement_holds(const struct pack_record *pack,
			    const struct test_subset_placement *placement,
			    const struct load_layout *layout,
			    const uint64_t *budgets, uint64_t layers)
{
	const struct pack_unit_record_2 *const units = pack_units_of(pack);
	const uint32_t *const card_of_unit = placement->card_of_unit;
	bool holds = card_of_unit[0] < CARDS &&
		     card_of_unit[TEST_MODEL_V41_HEAD_UNIT] < CARDS;
	for (uint32_t card = 0; card < CARDS; card++) {
		holds = holds && placement->used[card] <= budgets[card];
	}
	for (uint32_t layer = 0; layer < MODEL_LAYERS; layer++) {
		const uint32_t stage = TEST_MODEL_V41_LAYER_UNIT(layer),
			       first = TEST_MODEL_V41_EXPERT_UNIT(layer, 0);
		const uint32_t card = card_of_unit[stage];
		holds = holds && placement->layer_of_unit[stage] == layer &&
			card < CARDS && card_of_unit[first] == card &&
			(layer == 0 ||
			 card >= card_of_unit[TEST_MODEL_V41_LAYER_UNIT(layer -
									1u)]);
		for (uint32_t expert = 0; holds && expert < MODEL_EXPERTS;
		     expert++) {
			const uint32_t unit =
				TEST_MODEL_V41_EXPERT_UNIT(layer, expert);
			const uint32_t extent = units[unit].first_extent_index,
				       stand_in =
					       units[first].first_extent_index;
			const bool on_card = card_of_unit[unit] == card;
			const bool in_host = (layers & BIT(layer)) == 0 &&
					     card_of_unit[unit] ==
						     PLACEMENT_IN_HOST_MEMORY;
			holds = (on_card || in_host) &&
				layout->card_of_extent[extent] ==
					card_of_unit[unit] &&
				placement->plan_layout.card_of_extent[extent] ==
					card &&
				placement->plan_layout
						.byte_offset_of_extent[extent] ==
					layout->byte_offset_of_extent
						[on_card ? extent : stand_in];
		}
	}
    return holds;
}

/* The plan of the placement and its program on cards in host memory, read by the view. */
static bool program_holds(const struct pack_record *pack,
			  const struct test_subset_placement *placement,
			  const uint64_t *budgets, uint64_t layers)
{
	const struct model_limits limits = { MODEL_CONTEXT_WITHOUT_CANDIDATES,
					     1,
					     1,
					     STATE_WIDTH_16,
					     LINEAR_PROFILE_PRECISION,
					     LINEAR_ORDER_FIXED,
					     CMP_ORDER_REFERENCE,
					     0,
					     0,
					     0 };
	const struct model_operations *const model =
		model_operations_of(V41_DATA_MODEL_NAME);
	struct device_card *cards[CARDS] = { NULL };
	struct card_description_record descriptions[CARDS];
	struct model_plan_record *plan = NULL;
	struct execution_program *program = NULL;
	struct test_program_view *view = NULL;
	test_host_memory_backend_reset(CARDS, CARD_MEMORY_BYTES);
	bool holds = model != NULL &&
		     model->plan_create(pack, &placement->plan_layout, &limits,
					&plan) == STATUS_OK;
	for (uint32_t card = 0; holds && card < CARDS; card++) {
		test_host_memory_backend_set_arithmetic_class(
			card, card == RESULT_CARD ? ARITHMETIC_CLASS_SM_120 :
						    ARITHMETIC_CLASS_SM_80);
		holds = backend->card_describe(card, &descriptions[card]) ==
				STATUS_OK &&
			backend->card_open(card, CARD_BYTES, &cards[card]) ==
				STATUS_OK &&
			plan->cards[card].arena_byte_count <=
				budgets[card] + KEPT_FREE_MIB * BYTES_PER_MIB;
	}
	for (uint32_t layer = 0; holds && layer < MODEL_LAYERS; layer++) {
		holds = plan->layer_card[layer] ==
			placement->card_of_unit[TEST_MODEL_V41_LAYER_UNIT(
				layer)];
	}
	holds = holds &&
		model->program_prepare(plan, backend, cards, descriptions,
				       PROGRAM_THIN, 1,
				       &program) == STATUS_OK &&
		test_program_view_create(backend, cards, descriptions, program,
					 MODEL_LAYERS, layers,
					 &view) == STATUS_OK;
	for (uint32_t layer = 0; holds && layer < MODEL_LAYERS; layer++) {
		holds = (layers & BIT(layer)) == 0 ||
			program->segments[view->layers[layer].segment].card ==
				plan->layer_card[layer];
	}
    test_program_view_destroy(view);
    if (program != NULL) {
        model->program_destroy(program);
    }
    if (plan != NULL) {
        model->plan_destroy(plan);
    }
    for (uint32_t card = 0; card < CARDS; card++) {
        backend->card_close(cards[card], WAIT_LIMIT_NANOSECONDS);
    }
    return holds;
}

/* One subset on the budgets: placed, laid out, the plan's copy made, its rules held, and its plan and program. */
static bool subset_holds(const struct pack_record *pack,
			 const uint64_t *budgets, uint64_t layers)
{
    struct test_subset_placement placement = {.card_of_unit = NULL}, never_made = {.card_of_unit = NULL};
    struct load_layout *layout = NULL;
    bool holds = test_subset_place(pack, budgets, CARDS, layers, &placement) ==
			 STATUS_OK &&
		 load_layout_create(pack, placement.card_of_unit,
				    pack_header_of(pack)->unit_count, CARDS,
				    &layout) == STATUS_OK &&
		 test_subset_plan_layout(pack, layout, &never_made) ==
			 STATUS_INVALID_ARGUMENT &&
		 test_subset_plan_layout(pack, layout, &placement) ==
			 STATUS_OK &&
		 placement_holds(pack, &placement, layout, budgets, layers) &&
		 program_holds(pack, &placement, budgets, layers);
    load_layout_destroy(layout);
    test_subset_placement_release(&placement);
    return holds;
}

static void test_placement(const struct pack_record *pack)
{
    const uint64_t budgets[CARDS] = {SMALL_BUDGET_MIB * BYTES_PER_MIB, SMALL_BUDGET_MIB * BYTES_PER_MIB,
                                     SMALL_BUDGET_MIB * BYTES_PER_MIB, LARGE_BUDGET_MIB * BYTES_PER_MIB};
    struct test_subset_placement placement = {.card_of_unit = NULL};
    expect(subset_holds(pack, budgets, test_subset_layers_with_sources(FIRST_LAYERS(FEWER_LAYERS))) &&
               subset_holds(pack, budgets, test_subset_layers_with_sources(FIRST_LAYERS(MORE_LAYERS))),
           "layers 0 to 35, and 0 to 36, are placed (each run layer whole on its card, every other layer's experts in "
           "host memory where its first stands in the plan's copy), the plan takes the copy within the budgets and "
           "the view reads the program's run layers");
    expect(test_subset_place(pack, budgets, CARDS, EVERY_LAYER, &placement) ==
		   STATUS_OUT_OF_MEMORY,
	   "all 40 layers do not fit the four cards");
    test_subset_placement_release(&placement);
    expect(test_subset_place(NULL, budgets, CARDS, EVERY_LAYER, &placement) ==
			   STATUS_INVALID_ARGUMENT &&
		   test_subset_place(pack, budgets, 0, EVERY_LAYER,
				     &placement) == STATUS_INVALID_ARGUMENT &&
		   test_subset_place(pack, budgets, CARDS, BEYOND_THE_MODEL,
				     &placement) == STATUS_INVALID_ARGUMENT,
	   "a missing argument, no card and a layer beyond the model's are refused");
    test_subset_placement_release(&placement);
}

int main(void)
{
	struct pack_record *pack = NULL;
	test_sources();
	const bool opened =
		test_model_v41_pack_open(
			(struct test_model_v41_change){
				TEST_MODEL_V41_CHANGE_NONE, NULL, 0 },
			&pack) == STATUS_OK;
	expect(opened, "the pack's tables open");
	if (opened) {
		test_placement(pack);
	}
	pack_close(pack);
	test_memory_storage_reset();
	if (failures != 0) {
		(void)fprintf(stderr,
			      "subset_placement_test FAILED: %d failures\n",
			      failures);
		return EXIT_FAILURE;
	}
    (void)printf("subset_placement_test passed\n");
    return EXIT_SUCCESS;
}
