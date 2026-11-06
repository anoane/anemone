/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Worker-private residual calibration. Observations are conditional reached outcomes, including valid bonus
 * comparisons. The current table remains an empirical predictor; these counts are not an independent prior. */
#ifndef DRAFT_SESSION_H
#define DRAFT_SESSION_H

#include <math.h>
#include <stdint.h>

#include "anemone/draft_policy.h"

#define DRAFT_SESSION_RESIDUAL_WEIGHT 4.0
#define DRAFT_SESSION_LEAST_PROBABILITY 1u
#define DRAFT_SESSION_MOST_PROBABILITY (DRAFT_PROBABILITY_SCALE - 1u)

/* Finite counts: 0 <= hits, expected <= reached. Preserve differences between confidence bins while correcting
 * their common position residual in either direction. An untouched position keeps its table estimate within
 * the open probability bounds. */
static inline uint32_t draft_session_probability(uint32_t table, double hits,
						 double expected,
						 double reached)
{
	const double probability =
		(double)table +
		DRAFT_PROBABILITY_SCALE * (hits - expected) /
			(reached + DRAFT_SESSION_RESIDUAL_WEIGHT);
	return probability < DRAFT_SESSION_LEAST_PROBABILITY ?
		       DRAFT_SESSION_LEAST_PROBABILITY :
	       probability > DRAFT_SESSION_MOST_PROBABILITY ?
		       DRAFT_SESSION_MOST_PROBABILITY :
		       (uint32_t)lround(probability);
}

#endif
