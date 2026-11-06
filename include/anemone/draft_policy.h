/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Draft work policy and release usage records. No target sampler or acceptance rule depends on this policy.
 * Tables are conditional keyed-match probabilities, split by greedy/sampled, position, then confidence bin.
 * Bins have upper edges -4,-2,-1,0,1,2,4,+infinity; equality enters the bin on the right. Defaults are priors,
 * not fitted model measurements. Refit with matched/reached: unreached tails are censored.
 * Fixed policy verifies the available prefix; cost policy maximizes expected saved microseconds per round,
 * subject to the conditional probability floor, including depth zero. Costs are for one ordinary target draw,
 * the fixed draft tail and each extra verified row. The engine retains the last returned scores across skipped
 * tails and probes initially and periodically, including during lookup runs. Rate policy maximizes expected
 * tokens per microsecond, including the ordinary draw. Adaptive policy restores doubling, the confidence-sign
 * prefix gate, reset to one below half acceptance and a 15-opportunity cooldown. Other DSpark policies have
 * no cooldown.
 * All records are fixed-size. Statistics add in constant space; uint64_t counters wrap modulo 2^64.
 */
#ifndef DRAFT_POLICY_H
#define DRAFT_POLICY_H

#include <stdbool.h>
#include <stdint.h>

#define DRAFT_CONFIDENCE_BINS 8u
#define DRAFT_CALIBRATION_POSITIONS 7u
#define DRAFT_STAT_POSITIONS 7u
#define DRAFT_SAMPLING_MODES 2u
#define DRAFT_PROBABILITY_SCALE 1000u
#define DRAFT_DEFAULT_THRESHOLD 100u
#define DRAFT_DEFAULT_LOOKUP_MIN 3u
#define DRAFT_DEFAULT_BASE_US 12700u
#define DRAFT_DEFAULT_TAIL_US 2200u
#define DRAFT_DEFAULT_ROW_US 2700u
#define DRAFT_MOST_COST_US 1000000000u
#define DRAFT_DEFAULT_PROBE_ROUNDS 8u
#define DRAFT_MOST_PROBE_ROUNDS 1048576u
#define DRAFT_DEFAULT_LOOKUP_PRIOR 500u
#define DRAFT_DEPTH_COUNTS (DRAFT_STAT_POSITIONS + 1u)

enum draft_source_code { DRAFT_LOOKUP, DRAFT_DSPARK, DRAFT_SOURCES };
enum draft_depth_policy {
	DRAFT_DEPTH_FIXED,
	DRAFT_DEPTH_COST,
	DRAFT_DEPTH_RATE,
	DRAFT_DEPTH_ADAPTIVE
};

struct draft_policy_record {
	uint32_t depth_policy; /* enum draft_depth_policy */
	uint32_t threshold; /* conditional acceptance floor, per thousand */
	uint32_t base_us, tail_us,
		row_us; /* base and row positive; tail may be zero; at most MOST_COST_US */
	uint32_t probe_rounds; /* 1..MOST_PROBE_ROUNDS: DSpark opportunities between forced probes */
	uint32_t lookup_prior; /* conditional lookup acceptance prior, per thousand; weight two reached outcomes */
	uint32_t acceptance[DRAFT_SAMPLING_MODES][DRAFT_CALIBRATION_POSITIONS]
			   [DRAFT_CONFIDENCE_BINS]; /* per thousand */
};

struct draft_outcomes {
	uint64_t produced, verified, reached, matched;
};

struct draft_source_statistics {
	uint64_t rounds;
	uint64_t proposed[DRAFT_STAT_POSITIONS];
	uint64_t reached[DRAFT_STAT_POSITIONS];
	uint64_t accepted[DRAFT_STAT_POSITIONS]; /* matching non-end-token prefix */
};

struct draft_statistics {
	uint64_t drawing_rounds; /* successful owner steps with logits, including ordinary decode */
	uint64_t depth_rounds
		[DRAFT_DEPTH_COUNTS]; /* histogram of actual verified depth, including zero */
	struct draft_source_statistics source[DRAFT_SOURCES];
	struct draft_outcomes confidence[DRAFT_SAMPLING_MODES]
					[DRAFT_CALIBRATION_POSITIONS]
					[DRAFT_CONFIDENCE_BINS];
};

void draft_policy_default(struct draft_policy_record *policy);
bool draft_policy_valid(const struct draft_policy_record *policy);
/* Finite score required; a nonfinite score maps to the lowest bin. */
uint32_t draft_confidence_bin(float score);
/* Prepared valid policy, confidence[most], most <= 7; invalid pointers/counts return zero. Ties keep less work.
 * Expected extra tokens = sum of conditional prefix products; gain = extra_tokens * base_us - tail_us - d*row_us.
 * Depth zero has zero gain and no tail cost. Rate uses (1 + extra_tokens)/(base_us + tail_us + d*row_us).
 * Fixed policy ignores scores, floor and costs. Adaptive returns the nonnegative-score prefix, at least one;
 * the engine additionally applies its retained depth and cooldown. */
uint32_t draft_depth(const struct draft_policy_record *policy,
		     const float *confidence, uint32_t most, bool sampled);
/* Saturating age in eligible drawing opportunities since the last verified DSpark round. The engine refreshes
 * a skipped tail one opportunity before this deadline; after the cold tail, verification is at least every N
 * opportunities when rows and draft context capacity permit. Initial probe is separate. */
bool draft_probe_due(const struct draft_policy_record *policy,
		     uint32_t rounds_since_dspark);
/* Prepared policy and DSpark choice (0..most), most <= 7. Lookup already passed the suffix-token check.
 * Its conditional estimate uses only reached/accepted outcomes plus two prior observations; censored rows do not
 * count as failures. Both sources pay the fused tail. Returns a lookup depth only for strictly greater gain.
 * A due DSpark probe always wins, even over profitable lookup, bounding DSpark starvation. */
uint32_t draft_hybrid_depth(const struct draft_policy_record *policy,
			    const float *confidence, uint32_t most,
			    bool sampled, uint32_t dspark_depth, bool probe_due,
			    const struct draft_source_statistics *lookup);
void draft_statistics_add(struct draft_statistics *total,
			  const struct draft_statistics *command);

#endif
