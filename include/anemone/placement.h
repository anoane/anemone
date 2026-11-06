/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * placement.h: the placement a load uses when it is given none, and the one a layout text states.
 *
 * Contract of the fill (placement_fill_in_order):
 *   - Stage units go to the cards in the order they run, card after card in the order of the budgets: a card
 *     takes units until the next does not fit, and no later unit returns to an earlier card. A unit that does
 *     not fit the next card either passes over it (that card stays empty) to the first one that holds it. So
 *     every card holds one run of consecutive stages.
 *   - Every stage unit keeps the first N of its on-demand units on its own card (all of them when it owns fewer;
 *     a host-only unit lying between them is not counted) and the rest go to host memory. N is the largest count
 *     for which everything fits, and at most the most on-demand units one stage unit owns; with N the same for
 *     all, a converter that orders a unit's on-demand units by use decides which ones the cards hold.
 *   - Host-only units go to host memory.
 *   - A budget is the bytes of weights a card may take: what its arena can hold beside the state of the model.
 *   - It is a pure function of the pack and the budgets: the same inputs give the same placement on every run.
 * Contract of the layout text (placement_from_layout_text; wide_prefill.md 5.1, 6.3; KN PL3), a choice made
 * explicitly, which leaves the fill as it is:
 *   - Statements joined by "; ", each "<units>: card <c>" or "experts kept: <units>: <n>" (auto_placement.md 14), in
 *     any order, numbers in decimal digits below UINT32_MAX, one blank where one is shown. <units> is "layers " and
 *     ranges joined by ", " (a range "<a>" or "<a>..<b>", a <= b: the stage units of those layer indices), or a name
 *     of the stage units outside the layers by their place in the pack (units lie in the order they run, pack.h):
 *     "embedding" the last one before the first layer's unit, "vision" those before the embedding, "head" the first
 *     one after that layer's unit, "draft head" the rest.
 *   - Each stage unit goes on card c of the statement that names it, with its first n on-demand units when an
 *     "experts kept" statement names it, else all of them (a host-only unit between them not counted); the rest go to
 *     host memory, as host-only units do. A card's units must fit its budget, as in the fill.
 *   - Refused with STATUS_INVALID_ARGUMENT and one error line: a text not of this form, of more than
 *     PLACEMENT_MOST_TEXT_BYTES or with more than PLACEMENT_MOST_RANGES ranges a statement; a range
 *     whose layers the stage units in it do not hold once each, in order; a name that names no unit, a card at or
 *     above card_count, n above the most on-demand units a unit of its statement owns; a unit named twice (a stage
 *     unit by two card statements or two ranges of one statement, an on-demand unit by two "experts kept"
 *     statements); a stage unit no card statement names. With STATUS_OUT_OF_MEMORY and one error line: a
 *     card whose units pass its budget.
 */
#ifndef PLACEMENT_H
#define PLACEMENT_H

#include <stdint.h>

#include "anemone/backend_seam.h"
#include "anemone/pack.h"
#include "anemone/status.h"

/* generate's placement (main.c) is this many fills, each in the budgets less the state the fill before gives each
 * card: the third for a second that is the first to leave experts in host memory, and so to need slots. */
#define PLACEMENT_STATE_PASSES 3u
#define PLACEMENT_MOST_TEXT_BYTES \
	4096u /* of a layout text, its NUL not counted */
#define PLACEMENT_MOST_RANGES 16u /* of layers, in one statement */

/* First user: the program's load. card_of_unit_out has one entry per unit of the pack (unit_count must be the
 * pack's); the result obeys pack_placement_is_valid. on_demand_units_per_owner_out receives N. Fails with
 * STATUS_INVALID_ARGUMENT for a NULL argument, a wrong unit_count and a card count of 0 or above
 * BACKEND_MAXIMUM_CARDS, and with STATUS_OUT_OF_MEMORY when the stage units alone do not fit (one
 * error line with what the last card lacks for them to fit). On failure nothing is written. */
MUST_CHECK result_status placement_fill_in_order(
	const struct pack_record *pack,
	const uint64_t *weight_budget_bytes_of_card, uint32_t card_count,
	uint32_t unit_count, uint32_t *card_of_unit_out,
	uint32_t *on_demand_units_per_owner_out);

/* First user: a load given a layout text. Arguments as the fill's, and also refused with
 * STATUS_INVALID_ARGUMENT: a NULL text. On failure card_of_unit_out is unspecified. */
MUST_CHECK result_status placement_from_layout_text(
	const struct pack_record *pack, const char *text,
	const uint64_t *weight_budget_bytes_of_card, uint32_t card_count,
	uint32_t unit_count, uint32_t *card_of_unit_out);

#endif
