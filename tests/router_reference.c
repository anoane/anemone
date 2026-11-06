/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * router_reference.c: the host reference of the router. Contract: router_reference.h.
 */
#include "router_reference.h"

#include <float.h>
#include <stddef.h>
#include <string.h>

#include "anemone/router_records.h"

#define EXPERTS MODEL_EXPERTS
#define PICKS MODEL_EXPERTS_PER_ROW
#define EXPONENTIAL_REDUCTION 1024.0            /* 2^10, undone by ten squarings */
#define EXPONENTIAL_SQUARINGS 10u
#define EXPONENTIAL_TERMS 30u                   /* of the series of e^r, |r| < 0.8 */
#define EXPONENTIAL_SMALLEST (-800.0)           /* e^x is 0 in double below it */
#define LOGARITHM_TERMS 40u                     /* of the series of atanh(z), z <= 1/3 */
#define ROOT_STEPS 8u                           /* of Newton's, from 1 for a number in [1/4, 1) */
#define QUARTER 0.25
#define HALF 0.5
#define TWO 2.0
#define FOUR 4.0
#define SIGN_BIT 0x80000000u
#define FLOAT32_EXPONENT_MASK 0x7F800000u

enum top_k_place { FIRST, SECOND, THIRD, FOURTH, FIFTH, SIXTH };

static uint32_t bits_of(float value)
{
    uint32_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static float float_of(uint32_t bits)
{
    float value = 0.0f;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

/* e^x for x <= 0. */
static double exponential_of_negative(double x)
{
    const double reduced = x / EXPONENTIAL_REDUCTION;
    double term = 1.0, sum = 1.0;
    if (x < EXPONENTIAL_SMALLEST) {
        return 0.0;
    }
    for (uint32_t power = 1; power < EXPONENTIAL_TERMS; power++) {
        term *= reduced / power;
        sum += term;
    }
    for (uint32_t squaring = 0; squaring < EXPONENTIAL_SQUARINGS; squaring++) {
        sum *= sum;
    }
    return sum;
}

/* log(1 + y) for 0 <= y <= 1: 2 atanh(y / (2 + y)). */
static double logarithm_of_one_plus(double y)
{
    const double ratio = y / (TWO + y), ratio_squared = ratio * ratio;
    double power = ratio, sum = 0.0;
    for (uint32_t term = 0; term < LOGARITHM_TERMS; term++) {
        sum += power / (TWO * term + 1.0);
        power *= ratio_squared;
    }
    return TWO * sum;
}

/* The root of a finite number above zero. */
static double root(double value)
{
    double scale = 1.0, estimate = 1.0;
    while (value >= 1.0) {
        value *= QUARTER;
        scale *= TWO;
    }
    while (value < QUARTER) {
        value *= FOUR;
        scale *= HALF;
    }
    for (uint32_t step = 0; step < ROOT_STEPS; step++) {
        estimate = HALF * (estimate + value / estimate);
    }
    return estimate * scale;
}

static double softplus(float score)
{
    const double x = (double)score;
    if (x > (double)ROUTER_SOFTPLUS_THRESHOLD) {
	    return x;
    }
    return (x > 0.0 ? x : 0.0) + logarithm_of_one_plus(exponential_of_negative(x > 0.0 ? -x : x));
}

double test_router_affinity(float score)
{
    if (score != score) {
        return (double)score;
    }
    const double value = softplus(score);
    return value == 0.0 || value > DBL_MAX ? value : root(value);
}

bool test_router_affinity_holds(float score, float affinity)
{
    const double expected = test_router_affinity(score);
    if (expected != expected || expected > DBL_MAX) {
        return expected != expected ? affinity != affinity : (double)affinity == expected;
    }
    if (!(affinity >= 0.0f) || affinity > FLT_MAX) {
        return false;
    }
    if (score < 0.0f && exponential_of_negative((double)score) < (double)FLT_MIN) {
        const double square = (double)affinity * (double)affinity, softplus_value = softplus(score);
        return square - softplus_value <= TEST_ROUTER_SUBNORMAL_SLACK &&
               softplus_value - square <= TEST_ROUTER_SUBNORMAL_SLACK;
    }
    const uint32_t exponent_bits = bits_of(affinity) & FLOAT32_EXPONENT_MASK;
    const double place =
        exponent_bits == 0 ? (double)FLT_TRUE_MIN : (double)float_of(exponent_bits) * (double)FLT_EPSILON;
    const double difference = (double)affinity - expected;
    return difference <= TEST_ROUTER_AFFINITY_PLACES * place && -difference <= TEST_ROUTER_AFFINITY_PLACES * place;
}

/* Whether first comes before second in the order of the reference's top-k. */
static bool precedes(float first, uint32_t first_expert, float second, uint32_t second_expert)
{
    const bool first_is_nan = first != first, second_is_nan = second != second;
    if (first_is_nan || second_is_nan) {
        return first_is_nan && (!second_is_nan || first_expert < second_expert);
    }
    const bool first_is_negative = (bits_of(first) & SIGN_BIT) != 0;
    const bool second_is_negative = (bits_of(second) & SIGN_BIT) != 0;
    if (first == 0.0f && second == 0.0f && first_is_negative != second_is_negative) {
        return second_is_negative;
    }
    return first > second || (first == second && first_expert < second_expert);
}

/* The choice of a select of expert_count experts and pick_count picks (the reference order: six picks alone). */
static void select_of(const float *affinities, const float *bias, uint32_t expert_count, uint32_t pick_count,
                      bool reference_order, uint32_t *experts_out, float *weights_out)
{
    bool is_taken[EXPERTS] = {false};
    float biased[EXPERTS];
    uint32_t chosen[PICKS];
    uint32_t slot = 0;
    float sum = 0.0f;
    for (uint32_t expert = 0; expert < expert_count; expert++) {
        biased[expert] = affinities[expert] + bias[expert];
    }
    for (uint32_t pick = 0; pick < pick_count; pick++) {
        uint32_t best = 0;
        bool is_found = false;
        for (uint32_t expert = 0; expert < expert_count; expert++) {
            if (!is_taken[expert] && (!is_found || precedes(biased[expert], expert, biased[best], best))) {
                best = expert;
                is_found = true;
            }
        }
        is_taken[best] = is_found;
        chosen[pick] = best;
    }
    for (uint32_t expert = 0; expert < expert_count; expert++) {
        if (is_taken[expert]) {
            experts_out[slot++] = expert;
            sum += affinities[expert];
        }
    }
    if (reference_order && pick_count == PICKS) {
        sum = ((affinities[chosen[FIRST]] + affinities[chosen[FIFTH]]) + affinities[chosen[THIRD]]) +
              ((affinities[chosen[SECOND]] + affinities[chosen[SIXTH]]) + affinities[chosen[FOURTH]]);
    }
    const float denominator = sum + ROUTER_WEIGHT_EPSILON;
    for (slot = 0; slot < pick_count; slot++) {
	    weights_out[slot] = affinities[experts_out[slot]] / denominator *
				MODEL_ROUTE_SCALE;
    }
}

void test_router_select(const float *affinities, const float *bias, bool reference_order, uint32_t *experts_out,
                        float *weights_out)
{
    select_of(affinities, bias, EXPERTS, PICKS, reference_order, experts_out, weights_out);
}

void test_router_select_draft(const float *affinities, const float *bias, uint32_t *experts_out, float *weights_out)
{
	select_of(affinities, bias, MODEL_DRAFT_EXPERTS,
		  MODEL_DRAFT_EXPERTS_PER_ROW, false, experts_out, weights_out);
}

/* The job table of choices of pick_count a row, experts [row][pick_count] and weights alike. */
static void jobs_of(const uint32_t *experts, const float *weights,
		    uint32_t pick_count, uint32_t live_rows, uint32_t row_count,
		    const uint64_t *expert_offsets,
		    struct expert_job_record *jobs_out)
{
    const uint32_t job_count = row_count * pick_count;
    uint32_t job = 0;
    memset(jobs_out, 0, sizeof(*jobs_out) * job_count);
    for (uint32_t expert = 0; expert < EXPERTS && job < job_count; expert++) {
	    struct expert_job_record *const target = &jobs_out[job];
	    for (uint32_t row = 0; row < live_rows; row++) {
		    for (uint32_t slot = 0;
			 slot < pick_count &&
			 target->row_count < LINEAR_MAXIMUM_ROWS;
			 slot++) {
			    if (experts[row * pick_count + slot] == expert) {
				    target->pair_index[target->row_count] =
					    row * pick_count + slot;
				    target->route_weight[target->row_count] =
					    weights[row * pick_count + slot];
				    target->row_count++;
			    }
		    }
	}
        if (target->row_count != 0) {
		const uint64_t *const offsets =
			&expert_offsets[(size_t)expert * MODEL_EXPERT_MATRICES];
		target->gate_offset = offsets[ROUTER_GATE];
		target->up_offset = offsets[ROUTER_UP];
		target->down_offset = offsets[ROUTER_DOWN];
		target->expert = expert;
		job++;
        }
    }
}

void test_router_jobs(const uint32_t (*experts)[MODEL_EXPERTS_PER_ROW],
		      const float (*weights)[MODEL_EXPERTS_PER_ROW],
		      uint32_t live_rows, uint32_t row_count,
		      const uint64_t *expert_offsets,
		      struct expert_job_record *jobs_out)
{
    jobs_of((const uint32_t *)experts, (const float *)weights, PICKS, live_rows, row_count, expert_offsets,
            jobs_out);
}

void test_router_draft_jobs(
	const uint32_t (*experts)[MODEL_DRAFT_EXPERTS_PER_ROW],
	const float (*weights)[MODEL_DRAFT_EXPERTS_PER_ROW], uint32_t live_rows,
	uint32_t row_count, const uint64_t *expert_offsets,
	struct expert_job_record *jobs_out)
{
	jobs_of((const uint32_t *)experts, (const float *)weights,
		MODEL_DRAFT_EXPERTS_PER_ROW, live_rows, row_count,
		expert_offsets, jobs_out);
}

uint32_t test_router_jobs_differ(const struct expert_job_record *first,
				 const struct expert_job_record *second,
				 uint32_t job_count)
{
    uint32_t differing = 0;
    for (uint32_t job = 0; job < job_count; job++) {
	    const struct expert_job_record *const one = &first[job],
						  *const other = &second[job];
	    bool is_same = one->gate_offset == other->gate_offset &&
			   one->up_offset == other->up_offset &&
			   one->down_offset == other->down_offset &&
			   one->row_count == other->row_count &&
			   one->expert == other->expert &&
			   memcmp(one->pair_index, other->pair_index,
				  sizeof(one->pair_index)) == 0;
	    for (uint32_t column = 0; column < LINEAR_MAXIMUM_ROWS; column++) {
		    const float one_weight = one->route_weight[column],
				other_weight = other->route_weight[column];
		    is_same = is_same &&
			      (bits_of(one_weight) == bits_of(other_weight) ||
			       (one_weight != one_weight &&
				other_weight != other_weight));
	    }
	differing += is_same ? 0u : 1u;
    }
    return differing;
}

void test_router_wide_sort(const uint32_t (*experts)[MODEL_EXPERTS_PER_ROW],
			   const float (*weights)[MODEL_EXPERTS_PER_ROW],
			   uint32_t live_rows, uint32_t row_count,
			   const uint64_t *expert_offsets,
			   struct wide_pair *pairs_out,
			   struct wide_run *runs_out,
			   struct wide_expert_tile *tiles_out)
{
    uint32_t pair = 0, tile = 0;
    memset(tiles_out, 0, sizeof(*tiles_out) * ROUTER_WIDE_TILES(row_count));
    for (uint32_t expert = 0; expert < EXPERTS; expert++) {
	    const uint64_t *const offsets =
		    &expert_offsets[(size_t)expert * MODEL_EXPERT_MATRICES];
	    runs_out[expert].first_pair = pair;
	    for (uint32_t row = 0; row < live_rows; row++) {
		    for (uint32_t slot = 0; slot < PICKS; slot++) {
			    if (experts[row][slot] == expert) {
				    pairs_out[pair].pair_index =
					    row * PICKS + slot;
				    pairs_out[pair].route_weight =
					    weights[row][slot];
				    pair++;
			    }
		    }
        }
        runs_out[expert].pair_count = pair - runs_out[expert].first_pair;
	for (uint32_t taken = 0; taken < runs_out[expert].pair_count;
	     taken += WIDE_TILE_PAIRS) {
		const uint32_t left = runs_out[expert].pair_count - taken;
		tiles_out[tile].gate_offset = offsets[ROUTER_GATE];
		tiles_out[tile].up_offset = offsets[ROUTER_UP];
		tiles_out[tile].down_offset = offsets[ROUTER_DOWN];
		tiles_out[tile].first_pair =
			runs_out[expert].first_pair + taken;
		tiles_out[tile].pair_count =
			left < WIDE_TILE_PAIRS ? left : WIDE_TILE_PAIRS;
		tile++;
	}
    }
}

void test_router_wide_jobs(const uint32_t (*experts)[MODEL_EXPERTS_PER_ROW],
			   const float (*weights)[MODEL_EXPERTS_PER_ROW],
			   uint32_t live_rows, uint32_t row_count,
			   const uint64_t *expert_offsets,
			   struct expert_job_record *jobs_out)
{
    uint32_t first_job = 0;
    memset(jobs_out, 0, sizeof(*jobs_out) * ROUTER_WIDE_JOBS(row_count));
    for (uint32_t expert = 0; expert < EXPERTS; expert++) {
	    const uint64_t *const offsets =
		    &expert_offsets[(size_t)expert * MODEL_EXPERT_MATRICES];
	    uint32_t taken = 0; /* the expert's pairs so far */
	    for (uint32_t row = 0; row < live_rows; row++) {
		    for (uint32_t slot = 0; slot < PICKS; slot++) {
			    if (experts[row][slot] != expert) {
				    continue;
			    }
			    struct expert_job_record *const job =
				    &jobs_out[first_job +
					      taken / LINEAR_MAXIMUM_ROWS];
			    job->gate_offset = offsets[ROUTER_GATE];
			    job->up_offset = offsets[ROUTER_UP];
			    job->down_offset = offsets[ROUTER_DOWN];
			    job->expert = expert;
			    job->pair_index[job->row_count] =
				    row * PICKS + slot;
			    job->route_weight[job->row_count] =
				    weights[row][slot];
			    job->row_count++;
			    taken++;
		    }
        }
	first_job += (taken + LINEAR_MAXIMUM_ROWS - 1u) / LINEAR_MAXIMUM_ROWS;
    }
}

uint32_t test_router_wide_pairs_differ(const struct wide_pair *first,
				       const struct wide_pair *second,
				       uint32_t pair_count)
{
    uint32_t differing = 0;
    for (uint32_t pair = 0; pair < pair_count; pair++) {
        const float one = first[pair].route_weight, other = second[pair].route_weight;
        const bool is_same = first[pair].pair_index == second[pair].pair_index &&
                             (bits_of(one) == bits_of(other) || (one != one && other != other));
        differing += is_same ? 0u : 1u;
    }
    return differing;
}
