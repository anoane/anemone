/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Actual-price decisions must preserve useful non-certain drafts and leave unobserved work unlabelled. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "draft_cost.h"

#define COST_SEED_US 1000u
#define COST_SAMPLE_US 1800u
#define COST_FIRST_ESTIMATE_US 1100u
#define COST_REPEATS 256u
#define COST_CONVERGENCE_TOLERANCE_US 3u
#define OBSERVED_DEPTH 3u
#define CODE_DRAFTS 5u
#define CODE_SHORT_DEPTH 4u
#define CODE_BASE_COST_US 14300u
#define CODE_ROW_COST_US 690u
#define CODE_FULL_COST_US 18250u
#define CODE_UNECONOMIC_FULL_COST_US 21000u
#define CODE_FULL_TOKENS_LEAST 4.94
#define CODE_FULL_TOKENS_MOST 4.95
#define CODE_SHORT_TOKENS_LEAST 4.36
#define CODE_SHORT_TOKENS_MOST 4.37
#define HALF_PROBABILITY 500u
#define HIGH_PROBABILITY 900u
#define LOW_PROBABILITY 100u
#define HALF_ACCEPTANCE_TIE_US 1500u
#define HALF_ACCEPTANCE_WIN_US 1400u
#define LOSING_FIRST_COST_US 2000u
#define AMORTIZED_COST_US 2200u

static uint32_t failures;

static void expect(bool condition, const char *message)
{
	if (!condition) {
		(void)fprintf(stderr, "draft cost: %s\n", message);
		failures++;
	}
}

static void fill_probabilities(struct draft_policy_record *policy,
			       uint32_t greedy, uint32_t sampled)
{
	draft_policy_default(policy);
	policy->depth_policy = DRAFT_DEPTH_RATE;
	for (uint32_t mode = 0; mode < DRAFT_SAMPLING_MODES; mode++)
		for (uint32_t position = 0;
		     position < DRAFT_CALIBRATION_POSITIONS; position++)
			for (uint32_t bin = 0; bin < DRAFT_CONFIDENCE_BINS;
			     bin++)
				policy->acceptance[mode][position][bin] =
					mode == 0 ? greedy : sampled;
}

static void observed_costs(void)
{
	struct draft_cost_cell cells[DRAFT_DEPTH_COUNTS] = { 0 };
	struct draft_cost_cell *const observed = &cells[OBSERVED_DEPTH];
	expect(draft_cost_estimate(observed, COST_SEED_US) == COST_SEED_US,
	       "an unobserved shape uses its supplied prior");
	expect(draft_cost_observe(observed, COST_SEED_US, COST_SAMPLE_US) &&
		       observed->estimate_us == COST_FIRST_ESTIMATE_US &&
		       observed->observations == 1u,
	       "the first real sample smooths from the prior");
	for (uint32_t depth = 0; depth < DRAFT_DEPTH_COUNTS; depth++)
		if (depth != OBSERVED_DEPTH)
			expect(cells[depth].observations == 0 &&
				       cells[depth].estimate_us == 0 &&
				       draft_cost_estimate(&cells[depth],
							   COST_SAMPLE_US) ==
					       COST_SAMPLE_US,
			       "observing one plan supplies no labels for other or censored plans");
	const struct draft_cost_cell before = *observed;
	expect(!draft_cost_observe(observed, COST_SEED_US, 0) &&
		       !draft_cost_observe(observed, COST_SEED_US,
					   (uint64_t)DRAFT_MOST_COST_US + 1u) &&
		       !draft_cost_observe(observed, COST_SEED_US,
					   UINT64_MAX) &&
		       observed->estimate_us == before.estimate_us &&
		       observed->observations == before.observations,
	       "invalid durations neither train nor age a cost cell");
	for (uint32_t sample = 0; sample < COST_REPEATS; sample++)
		expect(draft_cost_observe(observed, COST_SEED_US,
					  COST_SAMPLE_US),
		       "a valid repeated observation is admitted");
	expect(observed->estimate_us > COST_FIRST_ESTIMATE_US &&
		       observed->estimate_us <= COST_SAMPLE_US &&
		       COST_SAMPLE_US - observed->estimate_us <=
			       COST_CONVERGENCE_TOLERANCE_US &&
		       observed->observations == COST_REPEATS + 1u,
	       "repeated observed costs converge within integer rounding");
	observed->observations = UINT32_MAX - 1u;
	const uint32_t before_lowering = observed->estimate_us;
	expect(draft_cost_observe(observed, COST_SEED_US, COST_SEED_US) &&
		       observed->observations == UINT32_MAX &&
		       observed->estimate_us < before_lowering,
	       "the observation count saturates while a cheaper sample still learns");
	const uint32_t before_raising = observed->estimate_us;
	expect(draft_cost_observe(observed, COST_SEED_US, COST_SAMPLE_US) &&
		       observed->observations == UINT32_MAX &&
		       observed->estimate_us > before_raising,
	       "a saturated counter does not freeze its estimate");
	struct draft_cost_cell maximum = { 0 };
	expect(draft_cost_observe(&maximum, DRAFT_MOST_COST_US,
				  DRAFT_MOST_COST_US) &&
		       maximum.estimate_us == DRAFT_MOST_COST_US &&
		       maximum.observations == 1u,
	       "the admitted cost boundary does not overflow the weighted sum");
}

static void code_prices(void)
{
	static const uint32_t conditional[CODE_DRAFTS] = { 895u, 999u, 882u,
							   999u, 733u };
	struct draft_policy_record policy;
	fill_probabilities(&policy, LOW_PROBABILITY, LOW_PROBABILITY);
	for (uint32_t position = 0; position < CODE_DRAFTS; position++)
		for (uint32_t bin = 0; bin < DRAFT_CONFIDENCE_BINS; bin++)
			policy.acceptance[0][position][bin] =
				conditional[position];
	const float confidence[DRAFT_CALIBRATION_POSITIONS] = { 0 };
	uint32_t prices_us[DRAFT_DEPTH_COUNTS];
	for (uint32_t depth = 0; depth < DRAFT_DEPTH_COUNTS; depth++)
		prices_us[depth] = CODE_BASE_COST_US + depth * CODE_ROW_COST_US;
	prices_us[CODE_DRAFTS] = CODE_FULL_COST_US;
	double expected_tokens = 0.0;
	expect(draft_rate_depth(&policy, confidence, CODE_DRAFTS, false,
				prices_us, &expected_tokens) == CODE_DRAFTS &&
		       expected_tokens > CODE_FULL_TOKENS_LEAST &&
		       expected_tokens < CODE_FULL_TOKENS_MOST,
	       "non-certain code drafts buy full depth at a seven-percent final cost increase");
	prices_us[CODE_DRAFTS] = CODE_UNECONOMIC_FULL_COST_US;
	expect(draft_rate_depth(&policy, confidence, CODE_DRAFTS, false,
				prices_us,
				&expected_tokens) == CODE_SHORT_DEPTH &&
		       expected_tokens > CODE_SHORT_TOKENS_LEAST &&
		       expected_tokens < CODE_SHORT_TOKENS_MOST,
	       "an actual expensive fifth row loses to the profitable four-row prefix");
	prices_us[CODE_DRAFTS] = 0;
	expect(draft_rate_depth(&policy, confidence, CODE_SHORT_DEPTH, false,
				prices_us, NULL) == CODE_SHORT_DEPTH,
	       "unsupported deeper prices are outside the available prefix");
	policy.threshold = DRAFT_PROBABILITY_SCALE;
	prices_us[CODE_DRAFTS] = CODE_FULL_COST_US;
	expect(draft_rate_depth(&policy, confidence, CODE_DRAFTS, false,
				prices_us, &expected_tokens) == 0 &&
		       expected_tokens == 1.0,
	       "the conditional floor can retain the ordinary target draw");
}

static void ordinary_draw_and_bounds(void)
{
	struct draft_policy_record policy;
	fill_probabilities(&policy, HALF_PROBABILITY, LOW_PROBABILITY);
	float confidence[DRAFT_CALIBRATION_POSITIONS] = { 0 };
	uint32_t prices_us[DRAFT_DEPTH_COUNTS] = { COST_SEED_US,
						   HALF_ACCEPTANCE_TIE_US };
	double expected_tokens = 0.0;
	expect(draft_rate_depth(&policy, confidence, 1, false, prices_us,
				&expected_tokens) == 0 &&
		       expected_tokens == 1.0,
	       "equal target-token rates keep ordinary decoding and less work");
	prices_us[1] = HALF_ACCEPTANCE_WIN_US;
	policy.threshold = HALF_PROBABILITY;
	expect(draft_rate_depth(&policy, confidence, 1, false, prices_us,
				NULL) == 1,
	       "a conditional probability equal to the floor remains eligible");
	policy.threshold = HALF_PROBABILITY + 1u;
	expect(draft_rate_depth(&policy, confidence, 1, false, prices_us,
				NULL) == 0,
	       "a probability below the floor cannot buy a row");
	policy.threshold = DRAFT_DEFAULT_THRESHOLD;
	prices_us[1] = LOSING_FIRST_COST_US;
	expect(draft_rate_depth(&policy, confidence, 1, false, prices_us,
				NULL) == 0,
	       "depth zero competes using its positive total round cost");
	fill_probabilities(&policy, HIGH_PROBABILITY, LOW_PROBABILITY);
	prices_us[1] = AMORTIZED_COST_US;
	prices_us[2] = AMORTIZED_COST_US;
	expect(draft_rate_depth(&policy, confidence, 1, false, prices_us,
				NULL) == 0 &&
		       draft_rate_depth(&policy, confidence, 2, false,
					prices_us, NULL) == 2,
	       "a losing first prefix does not hide profitable tail amortization");
	prices_us[1] = HALF_ACCEPTANCE_TIE_US;
	expect(draft_rate_depth(&policy, confidence, 1, false, prices_us,
				NULL) == 1 &&
		       draft_rate_depth(&policy, confidence, 1, true, prices_us,
					NULL) == 0,
	       "greedy and sampled conditional tables select their own work");
	policy.acceptance[0][0][0] = 0;
	confidence[0] = NAN;
	expect(draft_rate_depth(&policy, confidence, 1, false, prices_us,
				NULL) == 0,
	       "a nonfinite confidence uses the conservative lowest bin");
	confidence[0] = 0.0f;
	prices_us[0] = 0;
	expect(draft_rate_depth(&policy, confidence, 1, false, prices_us,
				&expected_tokens) == 0 &&
		       expected_tokens == 1.0,
	       "an invalid ordinary price cannot enter a rate comparison");
	prices_us[0] = COST_SEED_US;
	prices_us[1] = 0;
	expect(draft_rate_depth(&policy, confidence, 1, false, prices_us,
				NULL) == 0 &&
		       draft_rate_depth(NULL, confidence, 1, false, prices_us,
					NULL) == 0 &&
		       draft_rate_depth(&policy, NULL, 1, false, prices_us,
					NULL) == 0 &&
		       draft_rate_depth(&policy, confidence, 1, false, NULL,
					NULL) == 0 &&
		       draft_rate_depth(&policy, confidence, 0, false,
					prices_us, NULL) == 0 &&
		       draft_rate_depth(&policy, confidence,
					DRAFT_CALIBRATION_POSITIONS + 1u, false,
					prices_us, NULL) == 0,
	       "invalid prices, pointers and prefix bounds return no draft work");
}

int main(void)
{
	observed_costs();
	code_prices();
	ordinary_draw_and_bounds();
	(void)printf("draft cost calibration: %s\n",
		     failures == 0 ? "PASS" : "FAIL");
	return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
