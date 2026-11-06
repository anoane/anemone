/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * subset_placement.h: the placement of a run of some of DeepSeek-V4.1-Flash's layers on fewer cards than all of them
 * take, and the copy of the load layout its plan reads: the teacher-forced card test's, four cards until the fifth
 * (first_token_plan.md 4.3). Test support: no part of the engine.
 *
 * Contract:
 *   - test_subset_layers_with_sources: the layers asked, with the source and the index layer each reads (model_v41.h:
 *     the latest of V41_DATA_SOURCE_LAYERS and of V41_DATA_INDEX_LAYERS at or before it), until nothing more is
 *     added; bits at or above MODEL_LAYERS are left as they are.
 *   - test_subset_place, on a placement all zero or released: a layer is the stage unit that holds
 *     layers.<l>.attn_norm.weight. Stage units go to the cards in the pack's order as placement_fill_in_order
 *     puts them (a card takes units until the next does not fit in what is left of its budget; none returns to an
 *     earlier card), each with what it keeps: a layer of layers all its on-demand units, any other layer its first
 *     (the first it owns in the pack), a stage unit of no layer none. Then every on-demand unit left goes to its stage
 *     unit's card while it fits there, in the pack's order; what does not fit and every host-only unit stay in host
 *     memory. Refused with one error line: a NULL argument, a card count of 0 or above BACKEND_MAXIMUM_CARDS,
 *     or layers beyond the model's (STATUS_INVALID_ARGUMENT); a pack without a stage unit of its own for each
 *     layer (STATUS_CORRUPT_PACK); a stage unit whose bytes with what it keeps fit no card left, or no memory
 *     (STATUS_OUT_OF_MEMORY).
 *   - test_subset_plan_layout: a copy of layout (made from the placement) in which each extent of an on-demand unit of
 *     a layer that stayed in host memory stands where the same extent of the layer's first on-demand unit lies, so
 *     that a plan sees every expert of a layer on the layer's card; the loader still puts each where layout says. A
 *     layer whose first on-demand unit is not on its card or not of the shape of the others is refused
 *     (STATUS_INVALID_ARGUMENT, one error line); no memory is STATUS_OUT_OF_MEMORY.
 *   - test_subset_placement_release frees what both made; a placement all zero (never made) is accepted.
 */
#ifndef TEST_SUBSET_PLACEMENT_H
#define TEST_SUBSET_PLACEMENT_H

#include <stdint.h>

#include "anemone/backend_seam.h"
#include "anemone/loader.h"
#include "anemone/pack.h"
#include "anemone/status.h"

#define TEST_SUBSET_NO_LAYER UINT32_MAX

struct test_subset_placement {
    uint32_t *card_of_unit;                         /* the pack's unit_count: a card or in host memory */
    uint32_t *layer_of_unit;                        /* a layer's stage unit: its layer; else TEST_SUBSET_NO_LAYER */
    struct load_layout plan_layout; /* test_subset_plan_layout's copy */
    uint64_t used[BACKEND_MAXIMUM_CARDS]; /* bytes placed on each card */
};

uint64_t test_subset_layers_with_sources(uint64_t layers);
MUST_CHECK result_status
test_subset_place(const struct pack_record *pack, const uint64_t *budgets,
		  uint32_t card_count, uint64_t layers,
		  struct test_subset_placement *placement);
MUST_CHECK result_status test_subset_plan_layout(
	const struct pack_record *pack, const struct load_layout *layout,
	struct test_subset_placement *placement);
void test_subset_placement_release(struct test_subset_placement *placement);

#endif
