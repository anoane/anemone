/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Worker-private costs of observed draft plans. The caller owns each cell's work shape and supplies successful
 * round durations only. Unobserved shapes retain their supplied priors; observing one shape labels no other. */
#ifndef DRAFT_COST_H
#define DRAFT_COST_H

#include <stddef.h>
#include <stdint.h>

#include "anemone/draft_policy.h"

#define DRAFT_COST_WEIGHT 8u
#define DRAFT_COST_ROUNDING_BIAS (DRAFT_COST_WEIGHT / 2u)

struct draft_cost_cell {
	uint32_t estimate_us, observations;
};

/* A zero-initialized cell has no observations. The prepared seed is positive and at most MOST_COST_US. */
static inline uint32_t draft_cost_estimate(const struct draft_cost_cell *cell,
					   uint32_t seed_us)
{
	return cell == NULL || cell->observations == 0 ? seed_us :
							 cell->estimate_us;
}

/* Successful elapsed microseconds only: reject a missing cell, invalid seed or zero/oversized duration without
 * mutation. The first sample moves one eighth from the seed; later samples start at the rounded estimate.
 * Count saturation preserves subsequent learning. All products fit uint64_t at the admitted cost limit. */
static inline bool draft_cost_observe(struct draft_cost_cell *cell,
				      uint32_t seed_us, uint64_t elapsed_us)
{
	if (cell == NULL || seed_us == 0 || seed_us > DRAFT_MOST_COST_US ||
	    elapsed_us == 0 || elapsed_us > DRAFT_MOST_COST_US)
		return false;
	const uint64_t previous_us = draft_cost_estimate(cell, seed_us);
	cell->estimate_us = (uint32_t)((previous_us * (DRAFT_COST_WEIGHT - 1u) +
					elapsed_us + DRAFT_COST_ROUNDING_BIAS) /
				       DRAFT_COST_WEIGHT);
	if (cell->observations != UINT32_MAX)
		cell->observations++;
	return true;
}

/* Prepared valid RATE policy, confidence[most], and total round prices for depths 0..most, each positive and at
 * most MOST_COST_US. most is 1..CALIBRATION_POSITIONS. Maximize expected target tokens / price, including the
 * ordinary draw at depth zero. Conditional probabilities and the floor retain draft_policy.h's meanings;
 * prices already include any active tail, so no affine cost is added here. Strict improvement keeps less-work
 * ties. Optional output receives the selected total expected tokens, including its ordinary draw.
 * Invalid pointers/counts, another policy kind or invalid prices return zero and an expected-token output of
 * one. Positions and prices beyond most are not read; censored or unobserved costs are the caller's priors. */
static inline uint32_t
draft_rate_depth(const struct draft_policy_record *policy,
		 const float *confidence, uint32_t most, bool sampled,
		 const uint32_t prices_us[DRAFT_DEPTH_COUNTS],
		 double *expected_tokens_out)
{
	if (expected_tokens_out != NULL)
		*expected_tokens_out = 1.0;
	if (policy == NULL || confidence == NULL || prices_us == NULL ||
	    most == 0 || most > DRAFT_CALIBRATION_POSITIONS ||
	    policy->depth_policy != DRAFT_DEPTH_RATE)
		return 0;
	for (uint32_t depth = 0; depth <= most; depth++)
		if (prices_us[depth] == 0 ||
		    prices_us[depth] > DRAFT_MOST_COST_US)
			return 0;
	double reach = 1.0, tokens = 1.0, best_tokens = 1.0;
	uint32_t best = 0;
	for (uint32_t position = 0; position < most; position++) {
		const uint32_t probability =
			policy->acceptance[sampled ? 1u : 0u][position]
					  [draft_confidence_bin(
						  confidence[position])];
		if (probability < policy->threshold)
			break;
		reach *= (double)probability / DRAFT_PROBABILITY_SCALE;
		tokens += reach;
		const uint32_t depth = position + 1u;
		if (tokens * (double)prices_us[best] >
		    best_tokens * (double)prices_us[depth]) {
			best = depth;
			best_tokens = tokens;
		}
	}
	if (expected_tokens_out != NULL)
		*expected_tokens_out = best_tokens;
	return best;
}

#endif
