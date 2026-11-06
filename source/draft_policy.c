/* SPDX-License-Identifier: AGPL-3.0-or-later */
#include "anemone/draft_policy.h"

#include <math.h>
#include <stddef.h>

#define LOOKUP_PRIOR_WEIGHT 2.0

void draft_policy_default(struct draft_policy_record *policy)
{
	static const uint32_t priors[DRAFT_CONFIDENCE_BINS] = { 10,  50,  180,
								140, 520, 650,
								920, 993 };
	if (policy == NULL) {
		return;
	}
	policy->depth_policy = DRAFT_DEPTH_COST;
	policy->threshold = DRAFT_DEFAULT_THRESHOLD;
	policy->base_us = DRAFT_DEFAULT_BASE_US;
	policy->tail_us = DRAFT_DEFAULT_TAIL_US;
	policy->row_us = DRAFT_DEFAULT_ROW_US;
	policy->probe_rounds = DRAFT_DEFAULT_PROBE_ROUNDS;
	policy->lookup_prior = DRAFT_DEFAULT_LOOKUP_PRIOR;
	for (uint32_t mode = 0; mode < DRAFT_SAMPLING_MODES; mode++) {
		for (uint32_t position = 0;
		     position < DRAFT_CALIBRATION_POSITIONS; position++) {
			for (uint32_t bin = 0; bin < DRAFT_CONFIDENCE_BINS;
			     bin++) {
				policy->acceptance[mode][position][bin] =
					priors[bin];
			}
		}
	}
}

bool draft_policy_valid(const struct draft_policy_record *policy)
{
	if (policy == NULL || policy->depth_policy > DRAFT_DEPTH_ADAPTIVE ||
	    policy->threshold > DRAFT_PROBABILITY_SCALE ||
	    policy->base_us == 0 || policy->base_us > DRAFT_MOST_COST_US ||
	    policy->tail_us > DRAFT_MOST_COST_US || policy->row_us == 0 ||
	    policy->row_us > DRAFT_MOST_COST_US || policy->probe_rounds == 0 ||
	    policy->probe_rounds > DRAFT_MOST_PROBE_ROUNDS ||
	    policy->lookup_prior > DRAFT_PROBABILITY_SCALE) {
		return false;
	}
	for (uint32_t mode = 0; mode < DRAFT_SAMPLING_MODES; mode++) {
		for (uint32_t position = 0;
		     position < DRAFT_CALIBRATION_POSITIONS; position++) {
			for (uint32_t bin = 0; bin < DRAFT_CONFIDENCE_BINS;
			     bin++) {
				if (policy->acceptance[mode][position][bin] >
				    DRAFT_PROBABILITY_SCALE) {
					return false;
				}
			}
		}
	}
	return true;
}

uint32_t draft_confidence_bin(float score)
{
	static const float edges[] = { -4.0f, -2.0f, -1.0f, 0.0f,
				       1.0f,  2.0f,  4.0f };
	uint32_t bin = 0;
	while (isfinite(score) && bin + 1u < DRAFT_CONFIDENCE_BINS &&
	       score >= edges[bin]) {
		bin++;
	}
	return bin;
}

/* All arithmetic is double after conversion; even maximum configured costs cannot overflow an integer product. */
static double gain_at(const struct draft_policy_record *policy,
		      double extra_tokens, uint32_t depth)
{
	if (policy->depth_policy == DRAFT_DEPTH_RATE) {
		const double extra_us = depth == 0 ? 0.0 : (double)policy->tail_us +
			(double)depth * (double)policy->row_us;
		return (1.0 + extra_tokens) / ((double)policy->base_us + extra_us);
	}
	return depth == 0 ? 0.0 :
			    extra_tokens * (double)policy->base_us -
				    (double)policy->tail_us -
				    (double)depth * (double)policy->row_us;
}

static double dspark_tokens(const struct draft_policy_record *policy,
			    const float *confidence, uint32_t depth,
			    bool sampled)
{
	double survival = 1.0, extra_tokens = 0.0;
	for (uint32_t index = 0; index < depth; index++) {
		survival *=
			(double)policy->acceptance[sampled ? 1u : 0u][index]
						  [draft_confidence_bin(
							  confidence[index])] /
			DRAFT_PROBABILITY_SCALE;
		extra_tokens += survival;
	}
	return extra_tokens;
}

uint32_t draft_depth(const struct draft_policy_record *policy,
		     const float *confidence, uint32_t most, bool sampled)
{
	if (policy == NULL || confidence == NULL || most == 0 ||
	    most > DRAFT_CALIBRATION_POSITIONS) {
		return 0;
	}
	if (policy->depth_policy == DRAFT_DEPTH_FIXED) {
		return most;
	}
	if (policy->depth_policy == DRAFT_DEPTH_ADAPTIVE) {
		uint32_t depth = 1;
		while (depth < most && confidence[0] >= 0.0f && confidence[depth] >= 0.0f) {
			depth++;
		}
		return depth;
	}
	double survival = 1.0, extra_tokens = 0.0;
	double best_gain = gain_at(policy, 0.0, 0);
	uint32_t best = 0;
	for (uint32_t index = 0; index < most; index++) {
		const uint32_t probability =
			policy->acceptance[sampled ? 1u : 0u][index]
					  [draft_confidence_bin(
						  confidence[index])];
		if (probability < policy->threshold) {
			break;
		}
		survival *= (double)probability / DRAFT_PROBABILITY_SCALE;
		extra_tokens += survival;
		const double gain = gain_at(policy, extra_tokens, index + 1u);
		if (gain > best_gain) {
			best_gain = gain;
			best = index + 1u;
		}
	}
	return best;
}

bool draft_probe_due(const struct draft_policy_record *policy,
		     uint32_t rounds_since_dspark)
{
	return policy != NULL && policy->probe_rounds != 0 &&
	       rounds_since_dspark >= policy->probe_rounds - 1u;
}

uint32_t draft_hybrid_depth(const struct draft_policy_record *policy,
			    const float *confidence, uint32_t most,
			    bool sampled, uint32_t dspark_depth, bool probe_due,
			    const struct draft_source_statistics *lookup)
{
	if (policy == NULL || confidence == NULL || lookup == NULL ||
	    probe_due || most > DRAFT_CALIBRATION_POSITIONS ||
	    dspark_depth > most) {
		return 0;
	}
	double best_gain = gain_at(policy,
				   dspark_tokens(policy, confidence,
						 dspark_depth, sampled),
				   dspark_depth);
	/* A fixed DSpark prefix can lose money; lookup must still beat ordinary decode. */
	const double ordinary_gain = gain_at(policy, 0.0, 0);
	best_gain = best_gain > ordinary_gain ? best_gain : ordinary_gain;
	double survival = 1.0, extra_tokens = 0.0;
	uint32_t best = 0;
	for (uint32_t index = 0; index < most; index++) {
		const double probability =
			((double)lookup->accepted[index] +
			 LOOKUP_PRIOR_WEIGHT * (double)policy->lookup_prior /
				 DRAFT_PROBABILITY_SCALE) /
			((double)lookup->reached[index] + LOOKUP_PRIOR_WEIGHT);
		if (probability * DRAFT_PROBABILITY_SCALE <
		    (double)policy->threshold) {
			break;
		}
		survival *= probability;
		extra_tokens += survival;
		const double gain = gain_at(policy, extra_tokens, index + 1u);
		if (gain > best_gain) {
			best_gain = gain;
			best = index + 1u;
		}
	}
	return best;
}

void draft_statistics_add(struct draft_statistics *total,
			  const struct draft_statistics *command)
{
	if (total == NULL || command == NULL) {
		return;
	}
	total->drawing_rounds += command->drawing_rounds;
	for (uint32_t depth = 0; depth < DRAFT_DEPTH_COUNTS; depth++) {
		total->depth_rounds[depth] += command->depth_rounds[depth];
	}
	for (uint32_t source = 0; source < DRAFT_SOURCES; source++) {
		total->source[source].rounds += command->source[source].rounds;
		for (uint32_t position = 0; position < DRAFT_STAT_POSITIONS;
		     position++) {
			total->source[source].proposed[position] +=
				command->source[source].proposed[position];
			total->source[source].reached[position] +=
				command->source[source].reached[position];
			total->source[source].accepted[position] +=
				command->source[source].accepted[position];
		}
	}
	for (uint32_t mode = 0; mode < DRAFT_SAMPLING_MODES; mode++) {
		for (uint32_t position = 0;
		     position < DRAFT_CALIBRATION_POSITIONS; position++) {
			for (uint32_t bin = 0; bin < DRAFT_CONFIDENCE_BINS;
			     bin++) {
				struct draft_outcomes *const sum =
					&total->confidence[mode][position][bin];
				const struct draft_outcomes *const value =
					&command->confidence[mode][position]
							    [bin];
				sum->produced += value->produced;
				sum->verified += value->verified;
				sum->reached += value->reached;
				sum->matched += value->matched;
			}
		}
	}
}
