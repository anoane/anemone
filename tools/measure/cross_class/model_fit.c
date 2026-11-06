/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * model_fit.c: reads cases (as random_cases prints them) and the mma_cases results for them, and counts, for each
 * variant of a model of one m16n8k16 instruction, the cases whose result it reproduces on each card.
 *   model_fit <cases file> <results file>
 * Model: products exact; terms = the 16 products and C; E = the exponent of the largest term (variant: of the
 * largest product's ea + eb, its significand product unnormalized); each term truncated toward zero (variant: toward
 * minus infinity) to a multiple of 2^(E - W); summed exactly; the sum truncated toward zero to float32 (variant:
 * rounded to nearest).
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TERMS 16
#define MOST_CASES 8192

struct case_record {
    double c, a[TERMS], b[TERMS];
    float result_80, result_120;
};

static struct case_record cases[MOST_CASES];
static int minimum_exponent = -100000;     /* of a normal input: -126 for bfloat16, -14 for f16; else none */

/* The float32 of an exact double sum, toward zero or to nearest. */
static float to_float(long double sum, int nearest)
{
    if (nearest) {
        return (float)sum;
    }
    float value = (float)sum;                    /* nearest, then step toward zero if it went past */
    if (fabsl((long double)value) > fabsl(sum)) {
        value = nextafterf(value, 0.0f);
    }
    return value;
}

/* One group: the accumulator and `count` products from `first`, aligned, truncated, summed, truncated to float32. */
static float group_model(const struct case_record *record, float accumulator, int first, int count, int window,
                         int anchor_unnormalized)
{
    long double terms[TERMS + 1];
    int anchor = -100000;
    for (int term = 0; term < count; term++) {
        terms[term] = (long double)record->a[first + term] * (long double)record->b[first + term];
        if (terms[term] != 0.0L) {
            int exponent = 0, ea = 0, eb = 0;
            if (anchor_unnormalized) {
                (void)frexp(record->a[first + term], &ea);
                (void)frexp(record->b[first + term], &eb);
                ea = ea - 1 < minimum_exponent ? minimum_exponent + 1 : ea;   /* a subnormal: the smallest field's */
                eb = eb - 1 < minimum_exponent ? minimum_exponent + 1 : eb;
                exponent = ea + eb - 2;
            } else {
                (void)frexpl(terms[term], &exponent);
                exponent -= 1;
            }
            anchor = exponent > anchor ? exponent : anchor;
        }
    }
    terms[count] = accumulator;
    if (accumulator != 0.0f) {
        int exponent = 0;
        (void)frexp(accumulator, &exponent);
        anchor = exponent - 1 > anchor ? exponent - 1 : anchor;
    }
    if (anchor == -100000) {
        return 0.0f;
    }
    const long double unit = ldexpl(1.0L, anchor - window);
    long double sum = 0.0L;
    for (int term = 0; term <= count; term++) {
        sum += truncl(terms[term] / unit) * unit;
    }
    return to_float(sum, 0);
}

static float grouped_model(const struct case_record *record, int group, int window, int anchor_unnormalized)
{
    float accumulator = (float)record->c;
    for (int first = 0; first < TERMS; first += group) {
        accumulator = group_model(record, accumulator, first, group, window, anchor_unnormalized);
    }
    return accumulator;
}

static float model(const struct case_record *record, int window, int anchor_unnormalized, int floor_negative,
                   int nearest)
{
    long double terms[TERMS + 1];
    int anchor = -100000;
    for (int term = 0; term < TERMS; term++) {
        terms[term] = (long double)record->a[term] * (long double)record->b[term];
        if (terms[term] != 0.0L) {
            int exponent = 0;
            if (anchor_unnormalized) {
                int ea = 0, eb = 0;
                (void)frexp(record->a[term], &ea);
                (void)frexp(record->b[term], &eb);
                exponent = ea + eb - 2;          /* each significand in [1, 2): the unnormalized product's */
            } else {
                (void)frexpl(terms[term], &exponent);
                exponent -= 1;
            }
            anchor = exponent > anchor ? exponent : anchor;
        }
    }
    terms[TERMS] = record->c;
    if (record->c != 0.0) {
        int exponent = 0;
        (void)frexp(record->c, &exponent);
        anchor = exponent - 1 > anchor ? exponent - 1 : anchor;
    }
    if (anchor == -100000) {
        return 0.0f;
    }
    const long double unit = ldexpl(1.0L, anchor - window);
    long double sum = 0.0L;
    for (int term = 0; term <= TERMS; term++) {
        const long double scaled = terms[term] / unit;
        const long double kept = floor_negative ? floorl(scaled) : truncl(scaled);
        sum += kept * unit;
    }
    return to_float(sum, nearest);
}

int main(int argument_count, char **arguments)
{
    FILE *case_file = argument_count == 3 ? fopen(arguments[1], "r") : NULL;
    FILE *result_file = argument_count == 3 ? fopen(arguments[2], "r") : NULL;
    char line[8192];
    int count = 0;
    if (case_file == NULL || result_file == NULL) {
        return 1;
    }
    while (count < MOST_CASES && fgets(line, sizeof(line), case_file) != NULL) {
        char *cursor = line;
        (void)strtol(cursor, &cursor, 10);
        cases[count].c = strtod(cursor, &cursor);
        for (int term = 0; term < TERMS; term++) {
            cases[count].a[term] = strtod(cursor, &cursor);
            cases[count].b[term] = strtod(cursor, &cursor);
        }
        count++;
    }
    for (int index = 0; index < count && fgets(line, sizeof(line), result_file) != NULL; index++) {
        char *cursor = strchr(line, ':');
        cases[index].result_80 = strtof(cursor + 1, &cursor);
        cases[index].result_120 = strtof(cursor, &cursor);
    }
    minimum_exponent = getenv("MINIMUM_EXPONENT") != NULL ? atoi(getenv("MINIMUM_EXPONENT")) : -100000;
    for (int group = 2; group <= 8; group *= 2) {
        for (int window = 22; window <= 27; window++) {
            for (int anchor_unnormalized = 0; anchor_unnormalized < 2; anchor_unnormalized++) {
                int matches_80 = 0, matches_120 = 0;
                for (int index = 0; index < count; index++) {
                    const float predicted = grouped_model(&cases[index], group, window, anchor_unnormalized);
                    matches_80 += memcmp(&predicted, &cases[index].result_80, sizeof(float)) == 0;
                    matches_120 += memcmp(&predicted, &cases[index].result_120, sizeof(float)) == 0;
                }
                if (matches_80 * 10 > count * 9 || matches_120 * 10 > count * 9) {
                    printf("groups of %d W=%d anchor=%s: class80 %d/%d class120 %d/%d\n", group, window,
                           anchor_unnormalized ? "ea+eb" : "lead", matches_80, count, matches_120, count);
                }
            }
        }
    }
    for (int window = 22; window <= 27; window++) {
        for (int variant = 0; variant < 8; variant++) {
            const int anchor_unnormalized = variant & 1, floor_negative = (variant >> 1) & 1, nearest = (variant >> 2) & 1;
            int matches_80 = 0, matches_120 = 0;
            for (int index = 0; index < count; index++) {
                const float predicted = model(&cases[index], window, anchor_unnormalized, floor_negative, nearest);
                matches_80 += memcmp(&predicted, &cases[index].result_80, sizeof(float)) == 0;
                matches_120 += memcmp(&predicted, &cases[index].result_120, sizeof(float)) == 0;
            }
            if (matches_80 * 10 > count * 9 || matches_120 * 10 > count * 9) {
                printf("W=%d anchor=%s negative=%s final=%s: class80 %d/%d class120 %d/%d\n", window,
                       anchor_unnormalized ? "ea+eb" : "lead", floor_negative ? "floor" : "trunc",
                       nearest ? "nearest" : "zero", matches_80, count, matches_120, count);
            }
        }
    }
    return 0;
}
