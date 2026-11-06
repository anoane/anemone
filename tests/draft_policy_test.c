/* SPDX-License-Identifier: AGPL-3.0-or-later */
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "anemone/draft_policy.h"

#define FIRST_NEGATIVE_EDGE (-4.0f)
#define BELOW_LOWEST_EDGE (-5.0f)
#define HIGHEST_EDGE 4.0f
#define NEGATIVE_ZERO_BIN 4u
#define LAST_POSITION (DRAFT_CALIBRATION_POSITIONS - 1u)
#define CHAT_ACCEPTANCE 680u
#define SAMPLED_ACCEPTANCE 480u
#define STRUCTURED_ACCEPTANCE 950u
#define CHAT_BEST_DEPTH 4u
#define CHAT_RATE_DEPTH 3u
#define SAMPLED_BEST_DEPTH 2u
#define MANY_OUTCOMES 100u
#define HIGH_COST_US 1000000u
#define AMORTIZED_TAIL_US 15000u
#define TIE_BASE_US 10000u
#define TIE_TAIL_US 4000u
#define TIE_ROW_US 1000u
#define HALF_PROBABILITY 500u

static int failures;

static void expect(bool condition, const char *message)
{
	if (!condition) {
		(void)printf("FAILED: %s\n", message);
		failures++;
	}
}

static void fill_table(struct draft_policy_record *policy, uint32_t greedy,
		       uint32_t sampled)
{
	for (uint32_t mode = 0; mode < DRAFT_SAMPLING_MODES; mode++) {
		for (uint32_t position = 0;
		     position < DRAFT_CALIBRATION_POSITIONS; position++) {
			for (uint32_t bin = 0; bin < DRAFT_CONFIDENCE_BINS;
			     bin++) {
				policy->acceptance[mode][position][bin] =
					mode == 0 ? greedy : sampled;
			}
		}
	}
}

static void test_depth(void)
{
	struct draft_policy_record policy;
	const float confidence[DRAFT_CALIBRATION_POSITIONS] = {
		HIGHEST_EDGE, HIGHEST_EDGE, HIGHEST_EDGE, HIGHEST_EDGE,
		HIGHEST_EDGE, HIGHEST_EDGE, HIGHEST_EDGE
	};
	draft_policy_default(&policy);
	expect(draft_confidence_bin(BELOW_LOWEST_EDGE) == 0 &&
		       draft_confidence_bin(FIRST_NEGATIVE_EDGE) == 1 &&
		       draft_confidence_bin(-0.0f) == NEGATIVE_ZERO_BIN &&
		       draft_confidence_bin(HIGHEST_EDGE) ==
			       DRAFT_CONFIDENCE_BINS - 1u &&
		       draft_confidence_bin(NAN) == 0,
	       "bin boundaries include equality on the right");
	expect(draft_depth(&policy, confidence, DRAFT_CALIBRATION_POSITIONS,
			   false) == DRAFT_CALIBRATION_POSITIONS,
	       "high confidence pays for five rows");
	fill_table(&policy, CHAT_ACCEPTANCE, SAMPLED_ACCEPTANCE);
	expect(draft_depth(&policy, confidence, 1, false) == 1,
	       "68 percent chat acceptance buys depth one: gain 3736 microseconds");
	expect(draft_depth(&policy, confidence, DRAFT_CALIBRATION_POSITIONS,
			   false) == CHAT_BEST_DEPTH &&
		       draft_depth(&policy, confidence,
				   DRAFT_CALIBRATION_POSITIONS,
				   true) == SAMPLED_BEST_DEPTH,
	       "conditional products select four greedy rows and two sampled rows at measured costs");
	policy.acceptance[0][1][DRAFT_CONFIDENCE_BINS - 1u] = 0;
	expect(draft_depth(&policy, confidence, DRAFT_CALIBRATION_POSITIONS,
			   false) == 1,
	       "position-specific conditional zero removes all later benefit");
	fill_table(&policy, STRUCTURED_ACCEPTANCE, STRUCTURED_ACCEPTANCE);
	expect(draft_depth(&policy, confidence, DRAFT_CALIBRATION_POSITIONS,
			   false) == DRAFT_CALIBRATION_POSITIONS,
	       "structured acceptance pays for five rows");
	policy.tail_us = AMORTIZED_TAIL_US;
	expect(draft_depth(&policy, confidence, 1, false) == 0 &&
		       draft_depth(&policy, confidence,
				   DRAFT_CALIBRATION_POSITIONS,
				   false) == DRAFT_CALIBRATION_POSITIONS,
	       "a losing first prefix can still amortize the tail at larger depth");
	policy.tail_us = HIGH_COST_US;
	expect(draft_depth(&policy, confidence, DRAFT_CALIBRATION_POSITIONS,
			   false) == 0,
	       "tail cost can make every depth lose money");
	draft_policy_default(&policy);
	policy.row_us = HIGH_COST_US;
	expect(draft_depth(&policy, confidence, DRAFT_CALIBRATION_POSITIONS,
			   false) == 0,
	       "row cost can make every depth lose money");
	policy.row_us = TIE_ROW_US;
	policy.tail_us = TIE_TAIL_US;
	policy.base_us = TIE_BASE_US;
	fill_table(&policy, HALF_PROBABILITY, HALF_PROBABILITY);
	expect(draft_depth(&policy, confidence, 1, false) == 0,
	       "zero-gain tie chooses ordinary decode without charging its tail");
	policy.threshold = DRAFT_PROBABILITY_SCALE;
	expect(draft_depth(&policy, confidence, DRAFT_CALIBRATION_POSITIONS,
			   false) == 0,
	       "conditional floor excludes candidates");
	policy.depth_policy = DRAFT_DEPTH_FIXED;
	expect(draft_depth(&policy, confidence, LAST_POSITION, false) ==
		       LAST_POSITION,
	       "fixed policy respects available rows and overrides floor and costs");
	expect(draft_depth(NULL, confidence, 1, false) == 0 &&
		       draft_depth(&policy, NULL, 1, false) == 0 &&
		       draft_depth(&policy, confidence, 0, false) == 0 &&
		       draft_depth(&policy, confidence,
				   DRAFT_CALIBRATION_POSITIONS + 1u,
				   false) == 0,
	       "invalid pointers and row bounds do not read confidence");
}

static void test_rate_and_adaptive(void)
{
	struct draft_policy_record policy;
	float confidence[DRAFT_CALIBRATION_POSITIONS] = { 0 };
	draft_policy_default(&policy);
	fill_table(&policy, CHAT_ACCEPTANCE, SAMPLED_ACCEPTANCE);
	policy.depth_policy = DRAFT_DEPTH_RATE;
	expect(draft_depth(&policy, confidence, DRAFT_CALIBRATION_POSITIONS,
			   false) == CHAT_RATE_DEPTH,
	       "rate chooses three where saved-time gain chooses four");
	policy.depth_policy = DRAFT_DEPTH_ADAPTIVE;
	expect(draft_depth(&policy, confidence, DRAFT_CALIBRATION_POSITIONS,
			   false) == DRAFT_CALIBRATION_POSITIONS,
	       "zero confidence is included by the sign gate");
	confidence[0] = -1.0f;
	expect(draft_depth(&policy, confidence, DRAFT_CALIBRATION_POSITIONS,
			   false) == 1,
	       "adaptive retains the first recovery proposal after negative confidence");
	confidence[0] = 0.0f;
	confidence[1] = -1.0f;
	expect(draft_depth(&policy, confidence, DRAFT_CALIBRATION_POSITIONS,
			   false) == 1,
	       "the second proposal requires its own nonnegative confidence, not only the first score");
}

static void test_probe_and_hybrid(void)
{
	struct draft_policy_record policy;
	struct draft_source_statistics lookup = { 0 };
	const float confidence[DRAFT_CALIBRATION_POSITIONS] = { 0 };
	draft_policy_default(&policy);
	expect(!draft_probe_due(&policy, 0) &&
		       !draft_probe_due(&policy,
					policy.probe_rounds - 1u - 1u) &&
		       draft_probe_due(&policy, policy.probe_rounds - 1u) &&
		       draft_probe_due(&policy, UINT32_MAX),
	       "probe age has a finite inclusive bound");
	policy.probe_rounds = 1;
	expect(draft_probe_due(&policy, 0) && !draft_probe_due(NULL, 0),
	       "interval one reserves every opportunity for DSpark");
	fill_table(&policy, STRUCTURED_ACCEPTANCE, STRUCTURED_ACCEPTANCE);
	expect(draft_hybrid_depth(
		       &policy, confidence, DRAFT_CALIBRATION_POSITIONS, false,
		       DRAFT_CALIBRATION_POSITIONS, false, &lookup) == 0,
	       "a suffix hit cannot displace high-confidence DSpark");
	fill_table(&policy, 0, 0);
	expect(draft_hybrid_depth(&policy, confidence,
				  DRAFT_CALIBRATION_POSITIONS, false, 0, false,
				  &lookup) != 0,
	       "lookup may beat unprofitable DSpark");
	expect(draft_hybrid_depth(&policy, confidence,
				  DRAFT_CALIBRATION_POSITIONS, false, 1, true,
				  &lookup) == 0,
	       "lookup can never suppress a due DSpark probe");
	lookup.proposed[0] = MANY_OUTCOMES;
	expect(draft_hybrid_depth(&policy, confidence, 1, false, 0, false,
				  &lookup) == 1,
	       "censored lookup rows do not lower the conditional estimate");
	lookup.reached[0] = MANY_OUTCOMES;
	expect(draft_hybrid_depth(&policy, confidence,
				  DRAFT_CALIBRATION_POSITIONS, false, 0, false,
				  &lookup) == 0,
	       "observed lookup failures make lookup yield");
	lookup.accepted[0] = MANY_OUTCOMES;
	fill_table(&policy, HALF_PROBABILITY, HALF_PROBABILITY);
	expect(draft_hybrid_depth(&policy, confidence, 1, false, 1, false,
				  &lookup) == 1,
	       "strong recent lookup evidence can beat a profitable DSpark prefix");
	memset(&lookup, 0, sizeof(lookup));
	expect(draft_hybrid_depth(&policy, confidence, 1, false, 1, false,
				  &lookup) == 0,
	       "equal source gains defer to DSpark");
}

static void test_validation(void)
{
	struct draft_policy_record policy;
	draft_policy_default(&policy);
	expect(draft_policy_valid(&policy) && !draft_policy_valid(NULL),
	       "default and missing policies");
	uint32_t *const fields[] = {
		&policy.base_us,      &policy.tail_us,
		&policy.row_us,	      &policy.probe_rounds,
		&policy.lookup_prior, &policy.threshold,
		&policy.depth_policy, &policy.acceptance[0][LAST_POSITION][0]
	};
	static const uint32_t maximums[] = {
		DRAFT_MOST_COST_US,	 DRAFT_MOST_COST_US,
		DRAFT_MOST_COST_US,	 DRAFT_MOST_PROBE_ROUNDS,
		DRAFT_PROBABILITY_SCALE, DRAFT_PROBABILITY_SCALE,
		DRAFT_DEPTH_ADAPTIVE,	 DRAFT_PROBABILITY_SCALE
	};
	for (uint32_t index = 0; index < sizeof(fields) / sizeof(fields[0]);
	     index++) {
		const uint32_t saved = *fields[index];
		*fields[index] = maximums[index];
		expect(draft_policy_valid(&policy), "upper bound admitted");
		*fields[index] = maximums[index] + 1u;
		expect(!draft_policy_valid(&policy),
		       "above upper bound refused");
		*fields[index] = saved;
	}
	policy.base_us = 0;
	expect(!draft_policy_valid(&policy), "zero base cost refused");
	draft_policy_default(&policy);
	policy.row_us = 0;
	expect(!draft_policy_valid(&policy), "zero row cost refused");
	draft_policy_default(&policy);
	policy.probe_rounds = 0;
	expect(!draft_policy_valid(&policy), "unbounded probing refused");
	draft_policy_default(&policy);
	policy.tail_us = 0;
	expect(draft_policy_valid(&policy), "zero tail cost admitted");
}

int main(void)
{
	test_depth();
	test_rate_and_adaptive();
	test_probe_and_hybrid();
	test_validation();
	(void)printf("draft_policy_test: %d failures\n", failures);
	return failures == 0 ? 0 : 1;
}
