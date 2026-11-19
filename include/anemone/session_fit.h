/* SPDX-License-Identifier: AGPL-3.0-or-later */
#ifndef SESSION_FIT_H
#define SESSION_FIT_H
/*
 * session_fit.h: serve's --session_slots=max (owner, 8 Nov 2026), the most sessions of a context the cards hold,
 * found by the placement a serve makes (commands.c) before anything is loaded. A count fits when:
 *   - the placement passes place every unit: the layout text, or the fill in card order, each pass in the budgets
 *     less the state the pass before gives each card;
 *   - no state needs page-locked host memory;
 *   - the layout and the plan are made, grown as the open grows it (plan_grown), and every card's arena is within
 *     its budget.
 * Host only: the plan is made and destroyed, and no card is opened. tests/model_v41_placement_test.c calls these.
 */
#include <stdint.h>

#include "anemone/model_seam.h"
#include "anemone/pack.h"
#include "anemone/status.h"

struct session_fit_cards {
	const struct model_operations *model;
	const struct pack_record *pack;
	const char *layout_text; /* NULL: the fill in card order */
	/* each card's budget: its free memory less the backend's reserve and what is kept free */
	const uint64_t *budget_bytes_of_card;
	uint32_t card_count, unit_count;
	uint32_t *card_of_unit; /* scratch of unit_count entries; holds the last count's placement */
	/* NULL, or what the open adds to a plan's arenas after plan_create (commands.c: the original head offload's
	 * shards and flight banks), so the fit sees the arenas that are opened; its refusal is the plan's own */
	MUST_CHECK result_status (*plan_grown)(struct model_plan_record *plan);
};

/* STATUS_OK when limits' session_capacity sessions fit (the header comment), STATUS_OUT_OF_MEMORY when they do not: a
 * placement pass misses a budget, a state needs host memory, or an arena passes its card's budget. Any other status
 * is a refusal of the placement, the limits or the plan (a layout text, a DSpark card rule), which no count
 * changes. */
MUST_CHECK result_status session_fit(const struct session_fit_cards *cards,
				     const struct model_limits *limits);

/* The most sessions, 1 to most, that fit with limits' other values, by bisection (the fit re-cuts per count, so a count
 * past a refused one is not searched; a trial count's refusal is taken as not fitting); the count returned fits. 0
 * when not even one does: then *refusal_out is one session's refusal other than STATUS_OUT_OF_MEMORY, else
 * STATUS_OK. */
uint32_t session_fit_most(const struct session_fit_cards *cards,
			  const struct model_limits *limits, uint32_t most,
			  result_status *refusal_out);
#endif
