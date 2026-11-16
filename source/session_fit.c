/* SPDX-License-Identifier: AGPL-3.0-or-later */
#include "anemone/session_fit.h"

#include <string.h>

#include "anemone/backend_seam.h"
#include "anemone/loader.h"
#include "anemone/placement.h"

#define FIT_BISECTION_PARTS 2u /* each trial halves the counts left */

result_status session_fit(const struct session_fit_cards *cards,
			  const struct model_limits *limits)
{
	if (cards == NULL || limits == NULL || cards->model == NULL ||
	    cards->pack == NULL || cards->card_of_unit == NULL ||
	    cards->budget_bytes_of_card == NULL || cards->card_count == 0 ||
	    cards->card_count > BACKEND_MAXIMUM_CARDS)
		return STATUS_INVALID_ARGUMENT;
	const struct model_operations *const model = cards->model;
	uint64_t weight_budget[BACKEND_MAXIMUM_CARDS] = { 0 };
	memcpy(weight_budget, cards->budget_bytes_of_card,
	       cards->card_count * sizeof(weight_budget[0]));
	const struct placement_record placement = { cards->card_of_unit,
						    cards->unit_count,
						    cards->card_count };
	uint32_t on_demand = 0;
	result_status status = STATUS_OK;
	for (uint32_t pass = 0;
	     pass < PLACEMENT_STATE_PASSES && status == STATUS_OK; pass++) {
		status = cards->layout_text != NULL ?
				 placement_from_layout_text(
					 cards->pack, cards->layout_text,
					 weight_budget, cards->card_count,
					 cards->unit_count,
					 cards->card_of_unit) :
				 placement_fill_in_order(
					 cards->pack, weight_budget,
					 cards->card_count, cards->unit_count,
					 cards->card_of_unit, &on_demand);
		for (uint32_t card = 0;
		     card < cards->card_count && status == STATUS_OK; card++) {
			uint64_t state_bytes = 0;
			status = model->state_byte_count(cards->pack,
							 &placement, limits,
							 card, &state_bytes);
			weight_budget[card] =
				cards->budget_bytes_of_card[card] >
						state_bytes ?
					cards->budget_bytes_of_card[card] -
						state_bytes :
					0;
		}
	}
	uint64_t pinned_bytes = 0;
	if (status == STATUS_OK)
		status = model->state_byte_count(cards->pack, &placement,
						 limits,
						 PLACEMENT_IN_HOST_MEMORY,
						 &pinned_bytes);
	if (status == STATUS_OK && pinned_bytes != 0)
		status =
			STATUS_OUT_OF_MEMORY; /* a routed expert in host memory */
	struct load_layout *probe_layout = NULL;
	struct model_plan_record *probe_plan = NULL;
	if (status == STATUS_OK)
		status = load_layout_create(cards->pack, cards->card_of_unit,
					    cards->unit_count,
					    cards->card_count, &probe_layout);
	if (status == STATUS_OK)
		status = model->plan_create(cards->pack, probe_layout, limits,
					    &probe_plan);
	if (status == STATUS_OK && cards->plan_grown != NULL)
		status = cards->plan_grown(probe_plan);
	for (uint32_t card = 0; status == STATUS_OK && card < cards->card_count;
	     card++) {
		uint64_t arena_bytes = 0;
		status =
			model->arena_byte_count(probe_plan, card, &arena_bytes);
		if (status == STATUS_OK &&
		    arena_bytes > cards->budget_bytes_of_card[card])
			status = STATUS_OUT_OF_MEMORY;
	}
	model->plan_destroy(probe_plan);
	load_layout_destroy(probe_layout);
	return status;
}

uint32_t session_fit_most(const struct session_fit_cards *cards,
			  const struct model_limits *limits, uint32_t most,
			  result_status *refusal_out)
{
	if (limits == NULL || most == UINT32_MAX || refusal_out == NULL)
		return 0;
	struct model_limits trial = *limits;
	uint32_t fits = 0,
		 refused =
			 most +
			 1u; /* fits: a count that fits (0: none yet); refused: one that does not */
	/* one session first: a refusal other than memory there is the placement's own, which no count changes */
	trial.session_capacity = 1u;
	result_status status = most != 0u ? session_fit(cards, &trial) :
					    STATUS_OUT_OF_MEMORY;
	if (status == STATUS_OK)
		fits = 1u;
	else
		refused = 1u;
	while (refused - fits > 1u) {
		trial.session_capacity =
			fits + (refused - fits) / FIT_BISECTION_PARTS;
		if (session_fit(cards, &trial) == STATUS_OK)
			fits = trial.session_capacity;
		else
			refused = trial.session_capacity;
	}
	*refusal_out = fits == 0u && status != STATUS_OUT_OF_MEMORY ? status :
								      STATUS_OK;
	return fits;
}
