/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * sampler_reference.h: the sampler of sampler_records.h on the host, written from its contract and sharing nothing
 * with kernel_sampler.cu, in double where the contract leaves the card's own functions (expf, logf): what the tests
 * hold the cards to. Test support: no part of the engine.
 *
 * Contract:
 *   - test_sampler_greedy: the token of the first largest logit (0 of a row of no number).
 *   - test_sampler_weight: e_t of the contract with the exponential in double, rounded to float32; a card's own is
 *     held to it within TEST_SAMPLER_WEIGHT_PLACES places of float32. test_sampler_within_places: two finite float32
 *     of one sign no farther apart than places units in the last place of the larger one's binade (2^-149 among
 *     the subnormals), so that a result just below a power of two is not held to the finer places there.
 *   - test_sampler_threshold: tau of a row of weights as a card wrote them, by a sort of their bits (0 for top_p 1,
 *     and for weights of no mass); false when the memory of the sort cannot be had.
 *   - test_sampler_kept: the tokens whose weight is at least tau, ascending; their count.
 *   - test_sampler_uniform: u_t of the contract, exact in double.
 *   - test_sampler_exponential (of x <= 0) and test_sampler_logarithm (of a positive normal double): series in double
 *     within a few places, for the tests have no mathematics library.
 *   - test_sampler_race: over kept tokens, e_t / -log u_t in double: the winner and the best other token
 *     (TEST_SAMPLER_NO_TOKEN of one token), lowest token first among equals. A card's float32 scores may order two
 *     scores within TEST_SAMPLER_NEAR of each other (relative) either way: test_sampler_race_allows takes the winner,
 *     or the other one there.
 */
#ifndef TEST_SAMPLER_REFERENCE_H
#define TEST_SAMPLER_REFERENCE_H

#include <stdbool.h>
#include <stdint.h>

#define TEST_SAMPLER_WEIGHT_PLACES 2u           /* the card's expf: within 2 places of the true value */
#define TEST_SAMPLER_NEAR 1e-6                  /* far above the float32 error of a quotient by a logarithm */
#define TEST_SAMPLER_NO_TOKEN UINT32_MAX

struct test_sampler_race {
    double winner_score;
    double other_score;
    uint32_t winner;
    uint32_t other;
};

uint32_t test_sampler_greedy(const float *logits, uint32_t vocabulary_size);
float test_sampler_weight(float logit, float largest, float temperature);
bool test_sampler_within_places(float first, float second, uint32_t places);
bool test_sampler_threshold(const float *weights, uint32_t vocabulary_size, float top_p, uint32_t *threshold_out);
uint32_t test_sampler_kept(const float *weights, uint32_t vocabulary_size, uint32_t threshold, uint32_t *kept_out);
double test_sampler_uniform(uint64_t seed, int32_t position, uint32_t token);
void test_sampler_race(const float *weights, const uint32_t *kept, uint32_t kept_count, uint64_t seed,
                       int32_t position, struct test_sampler_race *race_out);
bool test_sampler_race_allows(const struct test_sampler_race *race, uint32_t token);
double test_sampler_exponential(double x);
double test_sampler_logarithm(double value);

#endif
