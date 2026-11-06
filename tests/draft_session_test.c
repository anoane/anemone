/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Residual calibration preserves confidence differences and learns both positive and negative errors. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "draft_session.h"
#include "engine_internal.h"

#define LOW_ESTIMATE 200u
#define HIGH_ESTIMATE 800u
#define OBSERVATIONS 16.0
#define EXPECTED_HITS 10.0
#define TOO_FEW_HITS 8.0
#define MANY_HITS 12.0
#define ERROR_ADJUSTMENT 100u
#define PLANNING_OWNERS 2u
#define PLANNING_PROGRAMS 2u
#define PLANNING_ANCHOR_ROWS 1u
#define PLANNING_CONFIDENT_POSITIONS 2u
#define PLANNING_DRAFTS 5u
#define PLANNING_SMALL_ROWS 8u
#define PLANNING_MOST_ROWS 12u
#define PLANNING_SMALL_COST 20000u
#define PLANNING_LARGE_COST 26000u
#define PLANNING_CONTEXT 128u
#define PLANNING_ANCHOR 4u
#define PLANNING_TOKEN 1u
#define PLANNING_LIMIT 64u
#define PLANNING_ROUND 100u
#define PLANNING_PRIOR 900u
#define PLANNING_REACHED 10.0
#define PLANNING_DEEP_EXPECTED 8.0
#define PLANNING_TABLE_LEVEL 8u
#define PLANNING_RESIDUAL_LEVEL 10u
#define PLANNING_EXPECTED_DEPTH 3u
#define PLANNING_COST_LEVEL 11u
#define PLANNING_COST_ROW_US 100u
#define PLANNING_CHANGED_MENU_US 1000u
#define PLANNING_OBSERVED_FULL_US 14000u
#define PLANNING_COST_OBSERVATIONS 64u
#define PLANNING_SHORT_LIMIT 2u
#define PLANNING_CONTEXT_ROOM 3u

/* Change only observed execution costs: the real planner must retain the request's probability state and use
 * immutable configured seeds for unsupported shapes, then obey legal limits independently of the cheap plan. */
static bool batch_shape_costs(struct inference_engine *engine,
			      struct flight *flight, const uint32_t *taken)
{
	static struct command_cursor before[PLANNING_OWNERS];
	uint32_t caps[PLANNING_OWNERS];
	engine->description.draft_session_acceptance = PLANNING_COST_LEVEL;
	engine->draft_policy.row_us = PLANNING_COST_ROW_US;
	engine->draft_policy.tail_us = 0;
	for (uint32_t owner = 0; owner < PLANNING_OWNERS; owner++) {
		struct command_cursor *const cursor = &engine->cursors[owner];
		cursor->command.generated_token_limit = PLANNING_LIMIT;
		cursor->session_had_full_round = true;
		cursor->session_last_full_round = PLANNING_ROUND;
	}
	memcpy(before, engine->cursors, sizeof(before));
	engine_draft_depths(engine, flight, taken, PLANNING_OWNERS, caps);
	if (caps[0] != PLANNING_EXPECTED_DEPTH ||
	    caps[1] != PLANNING_EXPECTED_DEPTH)
		return false;
	engine->batch_cost_us[1] = PLANNING_CHANGED_MENU_US;
	engine_draft_depths(engine, flight, taken, PLANNING_OWNERS, caps);
	if (caps[0] != PLANNING_EXPECTED_DEPTH ||
	    caps[1] != PLANNING_EXPECTED_DEPTH ||
	    engine->batch_cost_prior_us[1] != PLANNING_LARGE_COST)
		return false;
	/* The same rows with one tail do not label a two-tail program. */
	for (uint32_t observation = 0; observation < PLANNING_COST_OBSERVATIONS;
	     observation++)
		if (!draft_cost_observe(
			    &engine->batch_shape_costs[PLANNING_MOST_ROWS][1],
			    PLANNING_LARGE_COST, PLANNING_OBSERVED_FULL_US))
			return false;
	engine_draft_depths(engine, flight, taken, PLANNING_OWNERS, caps);
	if (caps[0] != PLANNING_EXPECTED_DEPTH ||
	    caps[1] != PLANNING_EXPECTED_DEPTH ||
	    engine->batch_shape_costs[PLANNING_MOST_ROWS][PLANNING_OWNERS]
			    .observations != 0)
		return false;
	for (uint32_t observation = 0; observation < PLANNING_COST_OBSERVATIONS;
	     observation++)
		if (!draft_cost_observe(
			    &engine->batch_shape_costs[PLANNING_MOST_ROWS]
						      [PLANNING_OWNERS],
			    PLANNING_LARGE_COST, PLANNING_OBSERVED_FULL_US))
			return false;
	engine_draft_depths(engine, flight, taken, PLANNING_OWNERS, caps);
	if (caps[0] != PLANNING_DRAFTS || caps[1] != PLANNING_DRAFTS ||
	    memcmp(before, engine->cursors, sizeof(before)) != 0)
		return false;
	engine->most_rows = PLANNING_SMALL_ROWS;
	engine_draft_depths(engine, flight, taken, PLANNING_OWNERS, caps);
	if (caps[0] != PLANNING_EXPECTED_DEPTH ||
	    caps[1] != PLANNING_EXPECTED_DEPTH ||
	    caps[0] + caps[1] + PLANNING_OWNERS > engine->most_rows)
		return false;
	engine->most_rows = PLANNING_MOST_ROWS;
	engine->cursors[0].command.generated_token_limit = PLANNING_SHORT_LIMIT;
	engine_draft_depths(engine, flight, taken, PLANNING_OWNERS, caps);
	if (caps[0] != PLANNING_SHORT_LIMIT - 1u || caps[1] > PLANNING_DRAFTS ||
	    caps[0] + caps[1] + PLANNING_OWNERS > engine->most_rows)
		return false;
	for (uint32_t owner = 0; owner < PLANNING_OWNERS; owner++)
		engine->cursors[owner].command.generated_token_limit =
			PLANNING_LIMIT;
	engine->description.context_token_limit =
		PLANNING_ANCHOR + PLANNING_CONTEXT_ROOM;
	engine_draft_depths(engine, flight, taken, PLANNING_OWNERS, caps);
	if (caps[0] != PLANNING_ANCHOR_ROWS || caps[1] != PLANNING_ANCHOR_ROWS)
		return false;
	engine->description.context_token_limit = PLANNING_CONTEXT;
	for (uint32_t owner = 0; owner < PLANNING_OWNERS; owner++)
		engine->cursors[owner].command.generated_token_limit =
			PLANNING_ANCHOR_ROWS;
	engine_draft_depths(engine, flight, taken, PLANNING_OWNERS, caps);
	if (caps[0] != UINT32_MAX || caps[1] != UINT32_MAX)
		return false;
	/* A legal C1 round delegates depth to its request-local actual-price decision. */
	engine->cursors[0].command.generated_token_limit = PLANNING_LIMIT;
	flight->owner_count = 1;
	caps[0] = 0;
	engine_draft_depths(engine, flight, taken, PLANNING_ANCHOR_ROWS, caps);
	return caps[0] == UINT32_MAX;
}

/* Exercise the real worker planner: certainty changes the profitable program, while a full-depth state alone
 * must not replace residual probabilities. Real probes remain separate and cannot exceed the row budget. */
static bool batch_probabilities(void)
{
	static struct inference_engine engine;
	static struct command_cursor cursors[PLANNING_OWNERS];
	memset(&engine, 0, sizeof(engine));
	memset(cursors, 0, sizeof(cursors));
	engine.cursors = cursors;
	engine.most_rows = PLANNING_MOST_ROWS;
	engine.draft_positions = PLANNING_DRAFTS;
	engine.description.context_token_limit = PLANNING_CONTEXT;
	engine.description.session_slot_count = PLANNING_OWNERS;
	engine.description.speculative_depth = PLANNING_DRAFTS;
	engine.batch_cost_count = PLANNING_PROGRAMS;
	engine.batch_cost_rows[0] = PLANNING_SMALL_ROWS;
	engine.batch_cost_rows[1] = PLANNING_MOST_ROWS;
	engine.batch_cost_verifiers[0] = PLANNING_OWNERS;
	engine.batch_cost_verifiers[1] = PLANNING_OWNERS;
	engine.batch_cost_us[0] = PLANNING_SMALL_COST;
	engine.batch_cost_us[1] = PLANNING_LARGE_COST;
	engine.batch_cost_prior_us[0] = PLANNING_SMALL_COST;
	engine.batch_cost_prior_us[1] = PLANNING_LARGE_COST;
	draft_policy_default(&engine.draft_policy);
	for (uint32_t mode = 0; mode < DRAFT_SAMPLING_MODES; mode++)
		for (uint32_t position = 0;
		     position < DRAFT_CALIBRATION_POSITIONS; position++)
			for (uint32_t bin = 0; bin < DRAFT_CONFIDENCE_BINS;
			     bin++)
				engine.draft_policy
					.acceptance[mode][position][bin] =
					PLANNING_PRIOR;
	struct flight flight = { .owner_count = PLANNING_OWNERS };
	uint32_t taken[PLANNING_OWNERS], caps[PLANNING_OWNERS];
	for (uint32_t owner = 0; owner < PLANNING_OWNERS; owner++) {
		flight.owners[owner] = owner;
		taken[owner] = PLANNING_ANCHOR_ROWS;
		struct command_cursor *const cursor = &cursors[owner];
		cursor->position = PLANNING_ANCHOR;
		cursor->token_id = PLANNING_TOKEN;
		cursor->command.session_slot = owner;
		cursor->command.generated_token_limit = PLANNING_LIMIT;
		cursor->has_draft_confidence = true;
		cursor->session_full_depth = true;
		cursor->session_had_full_round = true;
		cursor->session_last_full_round = PLANNING_ROUND;
		cursor->statistics.source[DRAFT_DSPARK].rounds = PLANNING_ROUND;
		cursor->draft_round =
			(struct step_round){ .session_slot = owner,
					     .anchor_position = PLANNING_ANCHOR,
					     .anchor_token = PLANNING_TOKEN,
					     .proposal_count =
						     PLANNING_DRAFTS };
		for (uint32_t position = 0; position < PLANNING_DRAFTS;
		     position++) {
			cursor->session_position_reached[position] =
				PLANNING_REACHED;
			cursor->session_position_hits[position] =
				position < PLANNING_CONFIDENT_POSITIONS ?
					PLANNING_REACHED :
					0.0;
			cursor->session_position_expected[position] =
				position < PLANNING_CONFIDENT_POSITIONS ?
					PLANNING_REACHED :
					PLANNING_DEEP_EXPECTED;
		}
	}
	engine.description.draft_session_acceptance = PLANNING_TABLE_LEVEL;
	engine_draft_depths(&engine, &flight, taken, PLANNING_OWNERS, caps);
	if (caps[0] != PLANNING_DRAFTS || caps[1] != PLANNING_DRAFTS)
		return false;
	engine.description.draft_session_acceptance = PLANNING_RESIDUAL_LEVEL;
	engine_draft_depths(&engine, &flight, taken, PLANNING_OWNERS, caps);
	if (caps[0] != PLANNING_EXPECTED_DEPTH ||
	    caps[1] != PLANNING_EXPECTED_DEPTH)
		return false;
	cursors[0].session_had_full_round = false;
	cursors[1].session_had_full_round = false;
	engine_draft_depths(&engine, &flight, taken, PLANNING_OWNERS, caps);
	if (caps[0] != PLANNING_DRAFTS || caps[1] != PLANNING_EXPECTED_DEPTH ||
	    caps[0] + caps[1] + PLANNING_OWNERS > engine.most_rows)
		return false;
	cursors[0].session_had_full_round = true;
	engine_draft_depths(&engine, &flight, taken, PLANNING_OWNERS, caps);
	if (caps[0] != PLANNING_EXPECTED_DEPTH || caps[1] != PLANNING_DRAFTS)
		return false;
	for (uint32_t owner = 0; owner < PLANNING_OWNERS; owner++)
		cursors[owner].command.generated_token_limit =
			PLANNING_ANCHOR_ROWS;
	engine_draft_depths(&engine, &flight, taken, PLANNING_OWNERS, caps);
	return caps[0] == UINT32_MAX && caps[1] == UINT32_MAX &&
	       batch_shape_costs(&engine, &flight, taken);
}

int main(void)
{
	const uint32_t unchanged =
		draft_session_probability(LOW_ESTIMATE, 0.0, 0.0, 0.0);
	const uint32_t lowered = draft_session_probability(
		HIGH_ESTIMATE, TOO_FEW_HITS, EXPECTED_HITS, OBSERVATIONS);
	const uint32_t raised = draft_session_probability(
		LOW_ESTIMATE, MANY_HITS, EXPECTED_HITS, OBSERVATIONS);
	const uint32_t lower_bin = draft_session_probability(
		LOW_ESTIMATE, TOO_FEW_HITS, EXPECTED_HITS, OBSERVATIONS);
	const uint32_t floor = draft_session_probability(
		LOW_ESTIMATE, 0.0, OBSERVATIONS, OBSERVATIONS);
	const uint32_t ceiling = draft_session_probability(
		HIGH_ESTIMATE, OBSERVATIONS, 0.0, OBSERVATIONS);
	const bool okay = unchanged == LOW_ESTIMATE &&
			  lowered == HIGH_ESTIMATE - ERROR_ADJUSTMENT &&
			  raised == LOW_ESTIMATE + ERROR_ADJUSTMENT &&
			  lowered - lower_bin == HIGH_ESTIMATE - LOW_ESTIMATE &&
			  floor == DRAFT_SESSION_LEAST_PROBABILITY &&
			  ceiling == DRAFT_SESSION_MOST_PROBABILITY &&
			  batch_probabilities();
	(void)printf("draft session residual calibration: %s\n",
		     okay ? "PASS" : "FAIL");
	return okay ? EXIT_SUCCESS : EXIT_FAILURE;
}
