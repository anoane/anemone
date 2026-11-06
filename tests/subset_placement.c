/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * subset_placement.c: the placement of subset_placement.h. Test support: no part of the engine.
 *
 * Contract: subset_placement.h. How this file keeps it: one walk over the units places the stage units with what each
 * keeps, as placement.c's walk does with a count per stage unit instead of one for all; a second walk gives the
 * on-demand units left the room their stage unit's card still has. The plan's copy changes only the extents of
 * on-demand units in host memory.
 */
#include "subset_placement.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "anemone/hooks.h"
#include "anemone/model_layout.h"
#include "anemone/placement.h"
#include "model_v41.h"

#define LAYER_NAME_CAPACITY 64u
#define BYTES_PER_GIB 1073741824.0
#define NONE TEST_SUBSET_NO_LAYER
#define BIT(index) (1ull << (index))

/* The latest layer of mask at or before layer, or MODEL_LAYERS for none. */
static uint32_t latest_of(uint64_t mask, uint32_t layer)
{
	uint32_t latest = MODEL_LAYERS;
	for (uint32_t earlier = 0; earlier <= layer; earlier++) {
		latest = (mask & BIT(earlier)) != 0 ? earlier : latest;
	}
    return latest;
}

uint64_t test_subset_layers_with_sources(uint64_t layers)
{
    for (uint64_t before = ~layers; before != layers;) {
        before = layers;
	for (uint32_t layer = 0; layer < MODEL_LAYERS; layer++) {
		const uint32_t source =
			latest_of(V41_DATA_SOURCE_LAYERS, layer);
		const uint32_t group = latest_of(V41_DATA_INDEX_LAYERS, layer);
		layers |= (layers & BIT(layer)) != 0 && source < MODEL_LAYERS ?
				  BIT(source) :
				  0;
		layers |= (layers & BIT(layer)) != 0 && group < MODEL_LAYERS ?
				  BIT(group) :
				  0;
	}
    }
    return layers;
}

/* Which stage unit is which layer's. */
MUST_CHECK static result_status layers_find(const struct pack_record *pack,
					    uint32_t *layer_of_unit)
{
	const struct pack_unit_record_2 *const units = pack_units_of(pack);
	const struct pack_tensor *const tensors = pack_tensors_of(pack);
	const struct pack_extent *const extents = pack_extents_of(pack);
	char name[LAYER_NAME_CAPACITY];
	for (uint32_t layer = 0; layer < MODEL_LAYERS; layer++) {
		const int length =
			snprintf(name, sizeof(name),
				 "layers.%" PRIu32 ".attn_norm.weight", layer);
		const uint32_t tensor =
			length <= 0 || (size_t)length >= sizeof(name) ?
				PACK_NOT_FOUND :
				pack_find_tensor(pack, name, (uint32_t)length);
		const uint32_t unit =
			tensor == PACK_NOT_FOUND ?
				NONE :
				extents[tensors[tensor].extent_index].unit_index;
		if (unit == NONE ||
		    units[unit].unit_kind != PLACEMENT_UNIT_STAGE ||
		    layer_of_unit[unit] != NONE) {
			LOG_ERROR(
				"subset placement: the pack has no stage unit of its own that holds %s",
				name);
			return STATUS_CORRUPT_PACK;
		}
		layer_of_unit[unit] = layer;
	}
	return STATUS_OK;
}

/* The first on-demand unit a stage unit owns, between it and the next stage unit; NONE for none. */
static uint32_t first_owned(const struct pack_unit_record_2 *units,
			    uint32_t unit_count, uint32_t stage)
{
    uint32_t owned = stage + 1u;
    while (owned < unit_count &&
	   units[owned].unit_kind != PLACEMENT_UNIT_STAGE &&
	   (units[owned].unit_kind != PLACEMENT_UNIT_ON_DEMAND ||
	    units[owned].owner_unit_index != stage)) {
	    owned++;
    }
    return owned < unit_count &&
			   units[owned].unit_kind == PLACEMENT_UNIT_ON_DEMAND ?
		   owned :
		   NONE;
}

/* What a stage unit keeps on its card: all its on-demand units (a layer run), its first (another layer), none. */
static bool is_kept(const struct pack_unit_record_2 *units, uint32_t unit_count,
		    uint32_t stage, uint32_t owned, uint32_t layer,
		    uint64_t layers)
{
	return units[owned].unit_kind == PLACEMENT_UNIT_ON_DEMAND &&
	       units[owned].owner_unit_index == stage && layer != NONE &&
	       ((layers & BIT(layer)) != 0 ||
		owned == first_owned(units, unit_count, stage));
}

result_status test_subset_place(const struct pack_record *pack,
				const uint64_t *budgets, uint32_t card_count,
				uint64_t layers,
				struct test_subset_placement *placement)
{
	if (pack == NULL || budgets == NULL || placement == NULL ||
	    card_count - 1u >= BACKEND_MAXIMUM_CARDS ||
	    layers >> MODEL_LAYERS != 0) {
		LOG_ERROR("subset placement: a missing argument, %" PRIu32
			  " cards, or layers beyond the model's",
			  card_count);
		return STATUS_INVALID_ARGUMENT;
	}
	const struct pack_unit_record_2 *const units = pack_units_of(pack);
	const uint32_t unit_count = pack_header_of(pack)->unit_count;
	uint64_t *const used = placement->used;
	*placement = (struct test_subset_placement){
		.card_of_unit = calloc(unit_count, sizeof(uint32_t)),
		.layer_of_unit = calloc(unit_count, sizeof(uint32_t))
	};
	if (placement->card_of_unit == NULL ||
	    placement->layer_of_unit == NULL) {
		LOG_ERROR("subset placement: no memory for %" PRIu32 " units",
			  unit_count);
		return STATUS_OUT_OF_MEMORY;
	}
    uint32_t *const card_of_unit = placement->card_of_unit, *const layer_of_unit = placement->layer_of_unit;
    for (uint32_t unit = 0; unit < unit_count; unit++) {
	    card_of_unit[unit] = PLACEMENT_IN_HOST_MEMORY;
	    layer_of_unit[unit] = NONE;
    }
    result_status status = layers_find(pack, layer_of_unit);
    for (uint32_t unit = 0, card = 0; status == STATUS_OK && unit < unit_count;
	 unit++) {
	    if (units[unit].unit_kind != PLACEMENT_UNIT_STAGE) {
		    continue;
	    }
	    uint64_t need = units[unit].byte_count;
	    for (uint32_t owned = unit + 1u;
		 owned < unit_count &&
		 units[owned].unit_kind != PLACEMENT_UNIT_STAGE;
		 owned++) {
		    need += is_kept(units, unit_count, unit, owned,
				    layer_of_unit[unit], layers) ?
				    units[owned].byte_count :
				    0;
	    }
	    while (card < card_count && need > budgets[card] - used[card]) {
		    card++;
	    }
	    if (card == card_count) {
		    double gib = 0.0;
#if LOGGING_HOOKS_ENABLED
		    gib = (double)need / BYTES_PER_GIB;
#endif
		    LOG_ERROR("subset placement: stage unit %" PRIu32
			      "%s and the experts it keeps, %.2f GiB, fit no "
			      "card left: run fewer layers",
			      unit,
			      layer_of_unit[unit] == NONE ? "" : " of a layer",
			      gib);
		    return STATUS_OUT_OF_MEMORY;
	}
        card_of_unit[unit] = card;
        used[card] += need;
	for (uint32_t owned = unit + 1u;
	     owned < unit_count &&
	     units[owned].unit_kind != PLACEMENT_UNIT_STAGE;
	     owned++) {
		card_of_unit[owned] = is_kept(units, unit_count, unit, owned,
					      layer_of_unit[unit], layers) ?
					      card :
					      card_of_unit[owned];
	}
    }
    for (uint32_t unit = 0; status == STATUS_OK && unit < unit_count; unit++) {
	    const uint32_t owner = units[unit].owner_unit_index;
	    const uint32_t card = owner < unit_count ? card_of_unit[owner] :
						       PLACEMENT_IN_HOST_MEMORY;
	    if (units[unit].unit_kind == PLACEMENT_UNIT_ON_DEMAND &&
		card_of_unit[unit] == PLACEMENT_IN_HOST_MEMORY &&
		card < card_count &&
		units[unit].byte_count <= budgets[card] - used[card]) {
		    card_of_unit[unit] = card;
		    used[card] += units[unit].byte_count;
	    }
    }
    return status;
}

result_status test_subset_plan_layout(const struct pack_record *pack,
				      const struct load_layout *layout,
				      struct test_subset_placement *placement)
{
	if (pack == NULL || layout == NULL || placement == NULL ||
	    placement->card_of_unit == NULL ||
	    placement->layer_of_unit == NULL ||
	    layout->extent_count != pack_header_of(pack)->extent_count) {
		LOG_ERROR(
			"subset placement: a missing argument, no placement, or a layout of another pack");
		return STATUS_INVALID_ARGUMENT;
	}
	const struct pack_unit_record_2 *const units = pack_units_of(pack);
	const struct pack_extent *const extents = pack_extents_of(pack);
	const uint32_t unit_count = pack_header_of(pack)->unit_count;
	const uint32_t *const card_of_unit = placement->card_of_unit;
	struct load_layout *const copy = &placement->plan_layout;
	*copy = *layout;
	copy->byte_offset_of_extent =
		calloc(layout->extent_count, sizeof(uint64_t));
	copy->card_of_extent = calloc(layout->extent_count, sizeof(uint32_t));
	if (copy->byte_offset_of_extent == NULL ||
	    copy->card_of_extent == NULL) {
		LOG_ERROR("subset placement: no memory for a layout of %" PRIu32
			  " extents",
			  layout->extent_count);
		return STATUS_OUT_OF_MEMORY;
	}
    memcpy(copy->byte_offset_of_extent, layout->byte_offset_of_extent, layout->extent_count * sizeof(uint64_t));
    memcpy(copy->card_of_extent, layout->card_of_extent, layout->extent_count * sizeof(uint32_t));
    for (uint32_t unit = 0; unit < unit_count; unit++) {
        const uint32_t owner = units[unit].owner_unit_index;
	if (units[unit].unit_kind != PLACEMENT_UNIT_ON_DEMAND ||
	    owner >= unit_count || placement->layer_of_unit[owner] == NONE ||
	    card_of_unit[unit] != PLACEMENT_IN_HOST_MEMORY) {
		continue;
	}
	const uint32_t first = first_owned(units, unit_count, owner);
	bool alike = first != NONE &&
		     card_of_unit[first] == card_of_unit[owner] &&
		     card_of_unit[owner] != PLACEMENT_IN_HOST_MEMORY &&
		     units[first].extent_count == units[unit].extent_count;
	for (uint32_t extent = 0; alike && extent < units[unit].extent_count; extent++) {
            const uint32_t stand_in = units[first].first_extent_index + extent;
            const uint32_t own = units[unit].first_extent_index + extent;
            alike = extents[stand_in].byte_count == extents[own].byte_count;
            copy->card_of_extent[own] = layout->card_of_extent[stand_in];
            copy->byte_offset_of_extent[own] = layout->byte_offset_of_extent[stand_in];
        }
        if (!alike) {
		LOG_ERROR("subset placement: on-demand unit %" PRIu32
			  " of layer %" PRIu32 " has no unit of its "
			  "shape on its layer's card to stand in its place",
			  unit, placement->layer_of_unit[owner]);
		return STATUS_INVALID_ARGUMENT;
	}
    }
    return STATUS_OK;
}

void test_subset_placement_release(struct test_subset_placement *placement)
{
    if (placement != NULL) {
        free(placement->card_of_unit);
        free(placement->layer_of_unit);
        free(placement->plan_layout.byte_offset_of_extent);
        free(placement->plan_layout.card_of_extent);
        *placement = (struct test_subset_placement){.card_of_unit = NULL};
    }
}
