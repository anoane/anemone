/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * sampler_reference.c: the sampler on the host. Contract: sampler_reference.h; what it computes: sampler_records.h.
 * The tests have no mathematics library: the exponential and the logarithm are series in double here, each within a
 * few places of double, far inside what the card is held to.
 */
#include "sampler_reference.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "anemone/pack_digest.h"
#include "anemone/sampler_records.h"

#define FLOAT32_MAGNITUDE_MASK 0x7FFFFFFFu
#define FLOAT32_EXPONENT_MASK 0x7F800000u
#define FLOAT32_PLACE 1.1920928955078125e-7      /* 2^-23: a place of 1 */
#define DOUBLE_EXPONENT_SHIFT 52u
#define DOUBLE_EXPONENT_MASK 0x7FFull
#define DOUBLE_EXPONENT_BIAS 1023
#define DOUBLE_FRACTION_MASK 0x000FFFFFFFFFFFFFull
#define DOUBLE_LEAST_NORMAL_EXPONENT (-1022)
#define LOWEST_EXPONENT (-746.0)                /* exp of anything below is under half the least double */
#define LOGARITHM_OF_TWO 0.6931471805599453
#define LOGARITHM_OF_TWO_HIGH 0.6931471803691238    /* its first 32 bits: k times it is exact */
#define LOGARITHM_OF_TWO_LOW 1.9082149292705877e-10
#define SQUARE_ROOT_OF_TWO 1.4142135623730951
#define SERIES_TERMS 16u                        /* beyond a last place for |r| <= ln 2 / 2 and |z| <= 0.172 */
#define HALF 0.5
#define TWO 2.0

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

/* 2^exponent for an exponent of a normal double. */
static double power_of_two(int64_t exponent)
{
    const uint64_t bits = (uint64_t)(exponent + DOUBLE_EXPONENT_BIAS) << DOUBLE_EXPONENT_SHIFT;
    double value = 0.0;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

/* e^x for x <= 0 (0 for -infinity): x = k ln 2 + r, the series of e^r, times 2^k in two steps where 2^k is below the
 * normal doubles. */
static double exponential(double x)
{
    if (!(x > LOWEST_EXPONENT)) {
        return 0.0;
    }
    const int64_t k = -(int64_t)(-x / LOGARITHM_OF_TWO + HALF);
    const double r = (x - (double)k * LOGARITHM_OF_TWO_HIGH) - (double)k * LOGARITHM_OF_TWO_LOW;
    double term = 1.0;
    double sum = 1.0;
    for (uint32_t order = 1; order <= SERIES_TERMS; order++) {
        term *= r / (double)order;
        sum += term;
    }
    if (k < DOUBLE_LEAST_NORMAL_EXPONENT) {
        return sum * power_of_two(k - DOUBLE_LEAST_NORMAL_EXPONENT) * power_of_two(DOUBLE_LEAST_NORMAL_EXPONENT);
    }
    return sum * power_of_two(k);
}

/* The natural logarithm of a positive normal double: m 2^e with m around 1, and 2 atanh((m - 1) / (m + 1)). */
static double logarithm(double value)
{
    uint64_t bits = 0;
    double mantissa = 0.0;
    memcpy(&bits, &value, sizeof(bits));
    int64_t exponent = (int64_t)((bits >> DOUBLE_EXPONENT_SHIFT) & DOUBLE_EXPONENT_MASK) - DOUBLE_EXPONENT_BIAS;
    const uint64_t mantissa_bits = (bits & DOUBLE_FRACTION_MASK) |
                                   ((uint64_t)DOUBLE_EXPONENT_BIAS << DOUBLE_EXPONENT_SHIFT);
    memcpy(&mantissa, &mantissa_bits, sizeof(mantissa));
    if (mantissa > SQUARE_ROOT_OF_TWO) {
        mantissa *= HALF;
        exponent++;
    }
    const double z = (mantissa - 1.0) / (mantissa + 1.0);
    double power = z;
    double sum = 0.0;
    for (uint32_t order = 1; order <= SERIES_TERMS; order++) {
        sum += power / (double)(TWO * order - 1.0);
        power *= z * z;
    }
    return (double)exponent * LOGARITHM_OF_TWO + TWO * sum;
}

uint32_t test_sampler_greedy(const float *logits, uint32_t vocabulary_size)
{
    uint32_t best = TEST_SAMPLER_NO_TOKEN;
    for (uint32_t token = 0; token < vocabulary_size; token++) {
        if (!isnan(logits[token]) && (best == TEST_SAMPLER_NO_TOKEN || logits[token] > logits[best])) {
            best = token;
        }
    }
    return best == TEST_SAMPLER_NO_TOKEN ? 0 : best;
}

float test_sampler_weight(float logit, float largest, float temperature)
{
	const float floored = temperature > SAMPLER_TEMPERATURE_FLOOR ?
				      temperature :
				      SAMPLER_TEMPERATURE_FLOOR;
	const float inverse = 1.0f / floored;
	const float exponent = logit * inverse - largest * inverse;
	return isnan(exponent) ? 0.0f : (float)exponential((double)exponent);
}

bool test_sampler_within_places(float first, float second, uint32_t places)
{
    const uint32_t first_bits = bits_of(first) & FLOAT32_MAGNITUDE_MASK;
    const uint32_t second_bits = bits_of(second) & FLOAT32_MAGNITUDE_MASK;
    const uint32_t larger_exponent = (first_bits > second_bits ? first_bits : second_bits) & FLOAT32_EXPONENT_MASK;
    const double place = larger_exponent == 0 ? (double)float_of(1u)
                                              : (double)float_of(larger_exponent) * FLOAT32_PLACE;
    const double difference = (double)first > (double)second ? (double)first - (double)second
                                                             : (double)second - (double)first;
    const bool same_sign = signbit(first) == signbit(second) || (first_bits == 0 && second_bits == 0);
    return same_sign && isfinite(first) && isfinite(second) && difference <= (double)places * place;
}

/* e 2^32 rounded to the nearest whole number, a tie to even: exact in double. */
static uint64_t mass_of(uint32_t weight_bits)
{
	const double scaled =
		(double)float_of(weight_bits) * (double)SAMPLER_MASS_SCALE;
	const uint64_t whole = (uint64_t)scaled;
	const double fraction = scaled - (double)whole;
	return whole +
	       (fraction > HALF || (fraction == HALF && (whole & 1u) != 0) ?
			1u :
			0u);
}

static int descending(const void *first, const void *second)
{
    const uint32_t first_bits = *(const uint32_t *)first;
    const uint32_t second_bits = *(const uint32_t *)second;
    return first_bits < second_bits ? 1 : (first_bits > second_bits ? -1 : 0);
}

bool test_sampler_threshold(const float *weights, uint32_t vocabulary_size, float top_p, uint32_t *threshold_out)
{
    uint32_t *const keys = malloc(sizeof(uint32_t) * (vocabulary_size != 0 ? vocabulary_size : 1u));
    uint64_t total = 0;
    uint64_t reached = 0;
    if (keys == NULL) {
        return false;
    }
    for (uint32_t token = 0; token < vocabulary_size; token++) {
        keys[token] = bits_of(weights[token]);
        total += mass_of(keys[token]);
    }
    qsort(keys, vocabulary_size, sizeof(uint32_t), descending);
    const double product = (double)top_p * (double)total;
    const uint64_t target = (uint64_t)product + ((double)(uint64_t)product < product ? 1u : 0u);    /* rounded up */
    *threshold_out = 0;
    for (uint32_t index = 0; top_p < 1.0f && index < vocabulary_size; index++) {
        reached += mass_of(keys[index]);
        if (reached >= target && reached != 0 && (index + 1 == vocabulary_size || keys[index + 1] != keys[index])) {
            *threshold_out = keys[index];
            break;
        }
    }
    free(keys);
    return true;
}

uint32_t test_sampler_kept(const float *weights, uint32_t vocabulary_size, uint32_t threshold, uint32_t *kept_out)
{
    uint32_t kept_count = 0;
    for (uint32_t token = 0; token < vocabulary_size; token++) {
        if (bits_of(weights[token]) >= threshold) {
            kept_out[kept_count] = token;
            kept_count++;
        }
    }
    return kept_count;
}

double test_sampler_uniform(uint64_t seed, int32_t position, uint32_t token)
{
	const uint64_t key =
		((uint64_t)(uint32_t)position << SAMPLER_POSITION_SHIFT) |
		token;
	const uint64_t hash = pack_digest_term(pack_digest_mix(seed), key);
	return ((double)(hash >> SAMPLER_UNIFORM_SHIFT) + HALF) *
	       (double)SAMPLER_UNIFORM_SCALE;
}

void test_sampler_race(const float *weights, const uint32_t *kept, uint32_t kept_count, uint64_t seed,
                       int32_t position, struct test_sampler_race *race_out)
{
    struct test_sampler_race race = {-1.0, -1.0, TEST_SAMPLER_NO_TOKEN, TEST_SAMPLER_NO_TOKEN};
    for (uint32_t index = 0; index < kept_count; index++) {
        const uint32_t token = kept[index];
        const double score = (double)weights[token] / -logarithm(test_sampler_uniform(seed, position, token));
        if (score > race.winner_score) {
            race.other_score = race.winner_score;
            race.other = race.winner;
            race.winner_score = score;
            race.winner = token;
        } else if (score > race.other_score) {
            race.other_score = score;
            race.other = token;
        }
    }
    *race_out = race;
}

bool test_sampler_race_allows(const struct test_sampler_race *race, uint32_t token)
{
    return token == race->winner ||
           (token == race->other && race->other_score >= race->winner_score * (1.0 - TEST_SAMPLER_NEAR));
}

double test_sampler_exponential(double x)
{
    return exponential(x);
}

double test_sampler_logarithm(double value)
{
    return logarithm(value);
}
