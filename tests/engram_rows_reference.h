/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * engram_rows_reference.h: the host's references of the two kernels of an n-gram layer (engram_rows_records.h), in
 * double precision, and the synthetic cases their tests share. Test support: no part of the engine.
 *
 * Contract:
 *   - test_engram_rows_value: the bfloat16 bits engram_values_entry gives for an E4M3 byte and its scale code, the
 *     exact product rounded once to float32 and then to bfloat16.
 *   - test_engram_rows_gate_of: the gate of one stream of hidden values in double precision, and the interval the
 *     card's gate lies in under the float32 arithmetic engram_rows_records.h states: every rounding's relative error
 *     at most 2^-24 (plus 2^-150 where a result is subnormal), the sums' longest chain of additions, rsqrtf and expf
 *     within two last places, a dot of exactly +0 where every term is 0. A not-a-number in a sum makes the gate one.
 *     It stands for plan 3.2's "gate within 1e-6 relative", which no float32 sum of 5,120 terms holds on every input
 *     and no bfloat16 output shows. Its width, as a share of the gate: 8e-7 where the clamp or a gate near 1 decides,
 *     5e-6 to 7e-6 where epsilon does, 1.3e-5 to 9e-5 on the other streams of the cases, up to 2e-3 on 200 random
 *     streams, 3.4% to 3.7% where the dot cancels exactly.
 *   - test_engram_rows_gate_outside: how many of a stream's outputs lie outside the two bfloat16 the kernel's last two
 *     roundings make of the interval's ends (or are no not-a-number where the arithmetic gives one), the first such
 *     place into first_outside; how many are the bits of the central gate rounded to float32 into equal_count.
 *   - The cases: test_engram_rows_rows_fill writes table rows each of which holds every E4M3 byte once under eight
 *     scale codes, and any 256 consecutive ones every pair of a byte and a code once (8 rows of
 *     TEST_ENGRAM_ROWS_PAIR_TABLE_ROWS: a launch of 65,536 values, every pair); test_engram_rows_gate_fill
 *     writes TEST_ENGRAM_ROWS_GATE_CASE_COUNT rows of streams [row][stream][hidden], projections [row][keys, value]
 *     and the two weights [stream][hidden], row r of case r (enum test_engram_rows_gate_case); every stream is zero
 *     from TEST_ENGRAM_ROWS_HALF on, where the value spans every bfloat16 fraction. Both are fixed numbers: the same
 *     on every machine.
 */
#ifndef TEST_ENGRAM_ROWS_REFERENCE_H
#define TEST_ENGRAM_ROWS_REFERENCE_H

#include <stdbool.h>
#include <stdint.h>

#include "anemone/engram_rows_records.h"

#define TEST_ENGRAM_ROWS_TABLE_ROWS 24u            /* a token's rows of one table, as the model has them */
#define TEST_ENGRAM_ROWS_PAIR_TABLE_ROWS 32u       /* 8 rows of 32: 256 table rows, each pair of a byte and a code */
#define TEST_ENGRAM_ROWS_HALF (MODEL_HIDDEN / 2u)
#define TEST_ENGRAM_ROWS_NOT_A_NUMBER_PLACE 7u     /* of stream 0 of the case's row */
#define TEST_ENGRAM_ROWS_INFINITY_PLACE (TEST_ENGRAM_ROWS_HALF + 5u)   /* of that row's value */

/* plain; keys that make every term h w key of the dot positive (a gate near 1); every one negative (near 0); zero
 * keys (the clamp of 1e-6 decides); a zero stream; keys whose mean square is far below epsilon (which decides);
 * large numbers; a not-a-number in a stream, an infinity in the value */
enum test_engram_rows_gate_case {
    TEST_ENGRAM_ROWS_PLAIN,
    TEST_ENGRAM_ROWS_ALONG,
    TEST_ENGRAM_ROWS_AGAINST,
    TEST_ENGRAM_ROWS_ZERO_KEY,
    TEST_ENGRAM_ROWS_ZERO_STREAM,
    TEST_ENGRAM_ROWS_TINY_KEY,
    TEST_ENGRAM_ROWS_LARGE,
    TEST_ENGRAM_ROWS_NOT_A_NUMBER,
    TEST_ENGRAM_ROWS_GATE_CASE_COUNT
};

struct test_engram_rows_gate {
    double central;
    double lowest;
    double highest;
    bool is_not_a_number;
};

uint16_t test_engram_rows_value(uint32_t code, uint32_t scale_code);
/* Each of the four: MODEL_HIDDEN bfloat16. */
struct test_engram_rows_gate test_engram_rows_gate_of(const uint16_t *stream, const uint16_t *key,
                                                      const uint16_t *query_weight, const uint16_t *key_weight);
uint64_t test_engram_rows_gate_outside(const struct test_engram_rows_gate *gate, const uint16_t *stream,
                                       const uint16_t *value, const uint16_t *output, uint64_t *equal_count,
                                       uint32_t *first_outside);
double test_engram_rows_square_root(double value);                  /* of 0, a number above it, or +infinity */
double test_engram_rows_sigmoid(double value);                      /* 1 / (1 + e^-value), of a number not 0 */

/* rows_out: table_row_count * ENGRAM_ROWS_ROW_BYTES bytes. */
void test_engram_rows_rows_fill(uint8_t *rows_out, uint32_t table_row_count);
/* streams_out [cases][streams][hidden], projection_out [cases][ENGRAM_ROWS_PROJECTION], the weights
 * [streams][hidden]. */
void test_engram_rows_gate_fill(uint16_t *streams_out, uint16_t *projection_out, uint16_t *query_weight_out,
                                uint16_t *key_weight_out);

#endif
