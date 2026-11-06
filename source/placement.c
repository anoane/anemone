/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * placement.c: the placement a load uses when it is given none, and a layout text's. Contract: placement.h.
 *
 *   - One walk over the units serves a candidate count N. A stage unit asks for its own bytes and those of the
 *     first N on-demand units it owns, and takes the present card if that fits in what is left of its budget,
 *     otherwise the next card that holds it. No card before the present one is looked at again.
 *   - A larger N never fits where a smaller one fails (every unit then asks for no less), so the largest N that
 *     fits is found by bisection between 0 and the most on-demand units one stage unit owns.
 *   - Nothing is allocated. The walk writes only when given somewhere to write: once, with a count seen to fit.
 *   - The bytes missing of the error line are what the last card lacks for the stage units to fit.
 *   - A layout text is read one statement at a time, each placed by one walk over the units into the output, where a
 *     unit no statement named yet is UNPLACED and an expert an "experts kept" statement keeps is KEPT_WITH_OWNER; a
 *     last walk gives the experts their owner's card (it lies before them) and adds up the cards. Nothing is allocated.
 */
#include "anemone/placement.h"

#include <errno.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "anemone/hooks.h"

#define BISECTION_PARTS 2
#define DECIMAL_BASE 10
#define UNPLACED                    \
	(PLACEMENT_IN_HOST_MEMORY - \
	 1u) /* in the output: a unit no statement named yet */
#define KEPT_WITH_OWNER             \
	(PLACEMENT_IN_HOST_MEMORY - \
	 2u) /* in the output: an expert a statement keeps */

/* The names of the units outside the layers, in the order they lie in a pack (placement.h). */
enum outside_name { OUTSIDE_VISION, OUTSIDE_EMBEDDING, OUTSIDE_HEAD, OUTSIDE_DRAFT_HEAD, OUTSIDE_NAMES };
static const char *const outside_names[OUTSIDE_NAMES] = {"vision", "embedding", "head", "draft head"};

/* Where those units lie: the first stage unit of a layer, the embedding and the head (unit_count for none). */
struct outside_places {
    uint32_t first_layer, embedding, head;
};

/* A statement as read: ranges of layers, or a name when range_count is 0; its card, or with is_kept (card 0) the
 * on-demand units each of its units keeps. */
struct statement {
	uint32_t first_layer[PLACEMENT_MOST_RANGES];
	uint32_t last_layer[PLACEMENT_MOST_RANGES];
	uint32_t range_count, name, card, kept;
	bool is_kept;
};

/* The sum, or UINT64_MAX when it is beyond 64 bits. */
static uint64_t sum_or_most(uint64_t sum, uint64_t amount)
{
    return amount > UINT64_MAX - sum ? UINT64_MAX : sum + amount;
}

/* What a stage unit asks of its card: its own bytes and those of the first on-demand units it owns, which lie
 * between it and the next stage unit. UINT64_MAX when that is beyond 64 bits. */
static uint64_t bytes_asked_by_stage(const struct pack_unit_record_2 *units,
				     uint32_t unit_count,
				     uint32_t stage_unit_index,
				     uint32_t on_demand_units_per_owner)
{
    uint64_t bytes_asked = units[stage_unit_index].byte_count;
    uint32_t units_counted = 0;
    for (uint32_t unit_index = stage_unit_index + 1;
	 unit_index < unit_count && units_counted < on_demand_units_per_owner &&
	 units[unit_index].unit_kind != PLACEMENT_UNIT_STAGE;
	 unit_index++) {
	    if (units[unit_index].unit_kind == PLACEMENT_UNIT_ON_DEMAND) {
		    bytes_asked = sum_or_most(bytes_asked,
					      units[unit_index].byte_count);
		    units_counted++;
	    }
    }
    return bytes_asked;
}

/* One walk with a candidate count. Returns the bytes the last card lacks: 0 when every stage unit found a card.
 * Writes the placement when card_of_unit_out is given; the caller gives it only with a count that fits. */
static uint64_t
walk_units(const struct pack_unit_record_2 *units, uint32_t unit_count,
	   const uint64_t *weight_budget_bytes_of_card, uint32_t card_count,
	   uint32_t on_demand_units_per_owner, uint32_t *card_of_unit_out)
{
    uint32_t card_index = 0;
    uint32_t units_kept = on_demand_units_per_owner;    /* of the present owner; before the first owner: no room */
    uint64_t bytes_left = weight_budget_bytes_of_card[0];
    uint64_t bytes_lacking = 0;
    for (uint32_t unit_index = 0; unit_index < unit_count; unit_index++) {
	    uint32_t card_of_unit = PLACEMENT_IN_HOST_MEMORY;
	    if (units[unit_index].unit_kind == PLACEMENT_UNIT_STAGE) {
		    const uint64_t bytes_asked =
			    bytes_asked_by_stage(units, unit_count, unit_index,
						 on_demand_units_per_owner);
		    if (bytes_asked == UINT64_MAX) {
			    return UINT64_MAX; /* beyond 64 bits: no budget holds it */
		    }
		    while (bytes_asked > bytes_left &&
			   card_index + 1 < card_count) {
			    card_index++;
			    bytes_left =
				    weight_budget_bytes_of_card[card_index];
		    }
		    /* past the last card the walk goes on only to add up what is lacking */
		    bytes_lacking = sum_or_most(
			    bytes_lacking, bytes_asked > bytes_left ?
						   bytes_asked - bytes_left :
						   0);
		    bytes_left = bytes_asked > bytes_left ?
					 0 :
					 bytes_left - bytes_asked;
		    units_kept = 0;
		    card_of_unit = card_index;
	    } else if (units[unit_index].unit_kind ==
			       PLACEMENT_UNIT_ON_DEMAND &&
		       units_kept < on_demand_units_per_owner) {
		    units_kept++;
		    card_of_unit = card_index;
	    }
	if (card_of_unit_out != NULL) {
            card_of_unit_out[unit_index] = card_of_unit;
        }
    }
    return bytes_lacking;
}

result_status
placement_fill_in_order(const struct pack_record *pack,
			const uint64_t *weight_budget_bytes_of_card,
			uint32_t card_count, uint32_t unit_count,
			uint32_t *card_of_unit_out,
			uint32_t *on_demand_units_per_owner_out)
{
	const struct pack_unit_record_2 *units = NULL;
	uint32_t fitting_count = 0; /* the largest count seen to fit */
	uint32_t highest_count = 0; /* no count above it is tried */
	uint32_t owned_count = 0;
	uint64_t bytes_lacking = 0;
	if (pack == NULL || weight_budget_bytes_of_card == NULL ||
	    card_of_unit_out == NULL || on_demand_units_per_owner_out == NULL ||
	    card_count == 0 || card_count > BACKEND_MAXIMUM_CARDS ||
	    unit_count != pack_header_of(pack)->unit_count) {
		return STATUS_INVALID_ARGUMENT;
	}
	units = pack_units_of(pack);
	for (uint32_t unit_index = 0; unit_index < unit_count; unit_index++) {
		if (units[unit_index].unit_kind == PLACEMENT_UNIT_STAGE) {
			owned_count = 0;
		} else if (units[unit_index].unit_kind ==
			   PLACEMENT_UNIT_ON_DEMAND) {
			owned_count++;
			highest_count = owned_count > highest_count ?
						owned_count :
						highest_count;
		}
    }
    bytes_lacking = walk_units(units, unit_count, weight_budget_bytes_of_card, card_count, 0, NULL);
    if (bytes_lacking != 0) {
	    LOG_ERROR("the stage units do not fit on %" PRIu32
		      " cards: the last card lacks %" PRIu64 " bytes",
		      card_count, bytes_lacking);
	    return STATUS_OUT_OF_MEMORY;
    }
    while (fitting_count < highest_count) {
        const uint32_t middle_count = highest_count - (highest_count - fitting_count) / BISECTION_PARTS;
        if (walk_units(units, unit_count, weight_budget_bytes_of_card, card_count, middle_count, NULL) == 0) {
            fitting_count = middle_count;
        } else {
            highest_count = middle_count - 1;
        }
    }
    (void)walk_units(units, unit_count, weight_budget_bytes_of_card, card_count, fitting_count, card_of_unit_out);
    ASSERT_TEXT(
	    pack_placement_is_valid(pack, card_of_unit_out, unit_count,
				    card_count),
	    "anemone_pack_placement_is_valid(pack, card_of_unit_out, unit_count, card_count)");
    *on_demand_units_per_owner_out = fitting_count;
    LOG_DEBUG_ENTRY("every stage unit keeps its first %" PRIu32
		    " on-demand units on its card",
		    fitting_count);
    return STATUS_OK;
}

/* The literal at the cursor, which then moves past it. */
static bool literal_take(const char **cursor, const char *literal)
{
    const size_t length = strlen(literal);
    const bool is_there = strncmp(*cursor, literal, length) == 0;
    *cursor += is_there ? length : 0;
    return is_there;
}

/* Decimal digits below UINT32_MAX at the cursor, which then moves past them (strtoull alone takes a blank or sign). */
static bool number_take(const char **cursor, uint32_t *value_out)
{
    char *end = NULL;
    errno = 0;
    const unsigned long long value = **cursor >= '0' && **cursor <= '9' ? strtoull(*cursor, &end, DECIMAL_BASE)
                                                                       : UINT32_MAX;
    if (errno != 0 || value >= UINT32_MAX) {
        return false;
    }
    *cursor = end;
    *value_out = (uint32_t)value;
    return true;
}

/* One statement of the form of placement.h, and the "; " behind it unless the text ends there. */
static bool statement_read(const char **cursor, struct statement *stated)
{
    bool is_read = true;
    const bool is_kept = literal_take(cursor, "experts kept: ");
    *stated = (struct statement){.name = OUTSIDE_NAMES, .is_kept = is_kept};
    if (literal_take(cursor, "layers ")) {
        do {
            const uint32_t range = stated->range_count++;
	    is_read = range < PLACEMENT_MOST_RANGES &&
		      number_take(cursor, &stated->first_layer[range]);
	    if (is_read) {
                stated->last_layer[range] = stated->first_layer[range];
                is_read = (!literal_take(cursor, "..") || number_take(cursor, &stated->last_layer[range])) &&
                          stated->first_layer[range] <= stated->last_layer[range];
            }
        } while (is_read && literal_take(cursor, ", "));
    } else {
        for (uint32_t name = 0; name < OUTSIDE_NAMES && stated->name == OUTSIDE_NAMES; name++) {
            stated->name = literal_take(cursor, outside_names[name]) ? name : stated->name;
        }
        is_read = stated->name != OUTSIDE_NAMES;
    }
    is_read = is_read && (is_kept ? literal_take(cursor, ": ") && number_take(cursor, &stated->kept)
                                  : literal_take(cursor, ": card ") && number_take(cursor, &stated->card));
    return is_read && (**cursor == '\0' || (literal_take(cursor, "; ") && **cursor != '\0'));
}

MUST_CHECK static result_status statement_refuse(uint32_t statement_number,
						 const char *what)
{
	LOG_ERROR("layout text: statement %" PRIu32 " %s", statement_number,
		  what);
	return STATUS_INVALID_ARGUMENT;
}

/* One walk: the stage units a statement names go on its card, or their first on-demand units are kept (placement.h).
 * A range's stage units hold its layers one by one in order: the next holds the first layer plus those met before. */
MUST_CHECK static result_status
statement_place(const struct pack_unit_record_2 *units, uint32_t unit_count,
		const struct outside_places *places,
		const struct statement *stated, uint32_t statement_number,
		uint32_t *card_of_unit)
{
	uint64_t found[PLACEMENT_MOST_RANGES] = {
		0
	}; /* stage units in each range */
	uint32_t named = 0, most_owned = 0, owned = 0,
		 hits = 0; /* hits: of the last stage unit met */
	bool is_named_twice = false, is_each_layer_once = true;
	for (uint32_t unit_index = 0; unit_index < unit_count; unit_index++) {
		const struct pack_unit_record_2 *const unit =
			&units[unit_index];
		if (unit->unit_kind == PLACEMENT_UNIT_STAGE) {
			const uint32_t name =
				unit_index < places->first_layer ?
					(unit_index == places->embedding ?
						 OUTSIDE_EMBEDDING :
						 OUTSIDE_VISION) :
				unit_index == places->head ? OUTSIDE_HEAD :
							     OUTSIDE_DRAFT_HEAD;
			hits = stated->range_count == 0 &&
					       unit->layer_index ==
						       PACK_NO_LAYER &&
					       name == stated->name ?
				       1u :
				       0u;
			for (uint32_t range = 0; range < stated->range_count;
			     range++) { /* NO_LAYER lies above every range */
				const bool is_in =
					unit->layer_index >=
						stated->first_layer[range] &&
					unit->layer_index <=
						stated->last_layer[range];
				is_each_layer_once =
					is_each_layer_once &&
					(!is_in ||
					 unit->layer_index ==
						 stated->first_layer[range] +
							 found[range]);
				found[range] += is_in ? 1u : 0u;
				hits += is_in ? 1u : 0u;
			}
			is_named_twice = is_named_twice || hits > 1u ||
					 (!stated->is_kept && hits != 0 &&
					  card_of_unit[unit_index] != UNPLACED);
			card_of_unit[unit_index] =
				hits != 0 && !stated->is_kept ?
					stated->card :
					card_of_unit[unit_index];
			named += hits;
			owned = 0;
		} else if (unit->unit_kind == PLACEMENT_UNIT_ON_DEMAND &&
			   hits != 0 && stated->is_kept) {
			is_named_twice = is_named_twice ||
					 card_of_unit[unit_index] != UNPLACED;
			card_of_unit[unit_index] =
				owned < stated->kept ? KEPT_WITH_OWNER :
						       PLACEMENT_IN_HOST_MEMORY;
			owned++;
			most_owned = owned > most_owned ? owned : most_owned;
		}
    }
    for (uint32_t range = 0; range < stated->range_count; range++) {
        is_each_layer_once = is_each_layer_once &&
                             found[range] == (uint64_t)stated->last_layer[range] - stated->first_layer[range] + 1u;
    }
    return is_named_twice ?
		   statement_refuse(
			   statement_number,
			   "names a unit that a range or statement names already") :
	   !is_each_layer_once ?
		   statement_refuse(
			   statement_number,
			   "names a layer that the stage units do not hold once, in order") :
	   named == 0 ?
		   statement_refuse(statement_number,
				    "names no unit of the pack") :
	   stated->is_kept && stated->kept > most_owned ?
		   statement_refuse(
			   statement_number,
			   "keeps more experts than a unit it names owns") :
		   STATUS_OK;
}

result_status
placement_from_layout_text(const struct pack_record *pack, const char *text,
			   const uint64_t *weight_budget_bytes_of_card,
			   uint32_t card_count, uint32_t unit_count,
			   uint32_t *card_of_unit_out)
{
	if (pack == NULL || text == NULL ||
	    weight_budget_bytes_of_card == NULL || card_of_unit_out == NULL ||
	    card_count == 0 || card_count > BACKEND_MAXIMUM_CARDS ||
	    unit_count != pack_header_of(pack)->unit_count) {
		return STATUS_INVALID_ARGUMENT;
	}
	if (strnlen(text, PLACEMENT_MOST_TEXT_BYTES + 1u) >
	    PLACEMENT_MOST_TEXT_BYTES) {
		LOG_ERROR("layout text: longer than %u bytes",
			  PLACEMENT_MOST_TEXT_BYTES);
		return STATUS_INVALID_ARGUMENT;
	}
	const struct pack_unit_record_2 *const units = pack_units_of(pack);
	struct outside_places places = { unit_count, unit_count, unit_count };
	for (uint32_t unit_index = 0; unit_index < unit_count; unit_index++) {
		const bool is_stage = units[unit_index].unit_kind ==
				      PLACEMENT_UNIT_STAGE;
		const bool is_outside = is_stage &&
					units[unit_index].layer_index ==
						PACK_NO_LAYER;
		places.first_layer = is_stage && !is_outside &&
						     places.first_layer ==
							     unit_count ?
					     unit_index :
					     places.first_layer;
		places.embedding = is_outside && places.first_layer ==
							   unit_count ?
					   unit_index :
					   places.embedding;
		places.head = is_outside && places.first_layer < unit_index &&
					      places.head == unit_count ?
				      unit_index :
				      places.head;
		card_of_unit_out[unit_index] =
			units[unit_index].unit_kind ==
					PLACEMENT_UNIT_HOST_ONLY ?
				PLACEMENT_IN_HOST_MEMORY :
				UNPLACED;
	}
	result_status status = STATUS_OK;
	const char *cursor = text;
	uint32_t number = 0;
	do {
		struct statement stated;
		number++;
		status =
			!statement_read(&cursor, &stated) ?
				statement_refuse(
					number,
					"is not \"<units>: card <c>\" or \"experts kept: <units>: <n>\"") :
			stated.card >= card_count ?
				statement_refuse(
					number,
					"names a card the load does not have") :
				statement_place(units, unit_count, &places,
						&stated, number,
						card_of_unit_out);
	} while (status == STATUS_OK && *cursor != '\0');
	uint64_t bytes_of_card[BACKEND_MAXIMUM_CARDS] = { 0 };
	for (uint32_t unit_index = 0;
	     unit_index < unit_count && status == STATUS_OK; unit_index++) {
		const struct pack_unit_record_2 *const unit =
			&units[unit_index];
		uint32_t *const card =
			&card_of_unit_out
				[unit_index]; /* an expert's owner lies before it: placed already */
		*card = unit->unit_kind == PLACEMENT_UNIT_ON_DEMAND &&
					*card != PLACEMENT_IN_HOST_MEMORY ?
				card_of_unit_out[unit->owner_unit_index] :
				*card;
		if (*card == UNPLACED) {
			LOG_ERROR(
				"layout text: no statement gives %s %" PRIu32
				" a card",
				unit->layer_index == PACK_NO_LAYER ?
					"the unit outside the layers numbered" :
					"layer",
				unit->layer_index == PACK_NO_LAYER ?
					unit_index :
					unit->layer_index);
			status = STATUS_INVALID_ARGUMENT;
		} else if (*card < card_count) {
			bytes_of_card[*card] = sum_or_most(bytes_of_card[*card],
							   unit->byte_count);
		}
	}
	for (uint32_t card = 0; card < card_count && status == STATUS_OK;
	     card++) {
		if (bytes_of_card[card] > weight_budget_bytes_of_card[card]) {
			LOG_ERROR("layout text: card %" PRIu32 " holds %" PRIu64
				  " bytes of weights, %" PRIu64 " above its "
				  "budget",
				  card, bytes_of_card[card],
				  bytes_of_card[card] -
					  weight_budget_bytes_of_card[card]);
			status = STATUS_OUT_OF_MEMORY;
		}
	}
	ASSERT_TEXT(
		status != STATUS_OK ||
			pack_placement_is_valid(pack, card_of_unit_out,
						unit_count, card_count),
		"status != ANEMONE_STATUS_OK || anemone_pack_placement_is_valid(pack, card_of_unit_out, unit_count, card_count)");
	return status;
}
