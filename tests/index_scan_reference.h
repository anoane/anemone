/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * index_scan_reference.h: the index scan and its selection (index_scan_records.h) on the host, in double precision
 * from keys and queries in their natural order, with the lane orders model_layout.h gives them on a card. It shares
 * nothing with the kernel file. Test support: no part of the engine.
 *
 * Contract:
 *   - FP16 is IEEE binary16. test_index_half_value is a value exactly (an infinity, or NAN, where the bits are one);
 *     test_index_half_nearest rounds a double once: to nearest, a tie to the even fraction, to an infinity from
 *     65,520 in magnitude on, a not-a-number to 0x7E00, the sign of a zero kept.
 *   - test_index_place_key lays key `key` (128 FP16 values, dims in order) into its 256 bytes of a tile of 8:
 *     run t of it at 64 t holds dims 16 j + 2 t + {0, 1, 8, 9} for j = 0 .. 7. test_index_place_queries lays a row's
 *     32 heads of 128 into 8,192 bytes: [head tile h][step j][lane 4 g + t][register r] holds the values of head
 *     16 h + g + 8 (r % 2) at dims 16 j + 2 t + 8 (r / 2) + {0, 1}.
 *   - test_index_scores gives each key's score before its rounding to FP16, its dots in double from the FP16 values
 *     (every dot of 128 is exact there for the values a test uses: a double holds 53 bits). TEST_INDEX_SUM_DOUBLE:
 *     over heads, the bfloat16 weight times the dot where it is above 0, in double. TEST_INDEX_SUM_TREE: in float32,
 *     as index_scan_records.h states it: each dot to float32 (exact where the test's values make it so), +0 where it
 *     is not above 0, times the weight, the heads in the records' tree. TEST_INDEX_SUM_HEADS_IN_ORDER: the same with
 *     the heads one after another from head 0 (a break).
 *   - test_index_select writes the top min(512, count) of count FP16 scores by score descending, then position
 *     ascending (highest_on_tie: position descending, a break), as positions ascending, then -1 up to 512.
 */
#ifndef TEST_INDEX_SCAN_REFERENCE_H
#define TEST_INDEX_SCAN_REFERENCE_H

#include <stdbool.h>
#include <stdint.h>

double test_index_half_value(uint16_t bits);
uint16_t test_index_half_nearest(double value);
uint32_t test_index_order_key(uint16_t score);         /* the order of index_scan_records.h, as a number */

/* natural [128]; tile: the 2,048 bytes of the tile that holds the key; key: its place there, 0 .. 7. */
void test_index_place_key(const uint16_t *natural, unsigned char *tile, uint32_t key);
/* natural [32][128]; placed [8,192 bytes]. */
void test_index_place_queries(const uint16_t *natural, unsigned char *placed);

enum test_index_sum { TEST_INDEX_SUM_DOUBLE, TEST_INDEX_SUM_TREE, TEST_INDEX_SUM_HEADS_IN_ORDER };

/* query [32][128] and keys [key_count][128] FP16, weights [32] bfloat16; scores_out [key_count]. */
void test_index_scores(const uint16_t *query, const uint16_t *weights, const uint16_t *keys, uint32_t key_count,
                       enum test_index_sum sum, double *scores_out);

/* scores [count]; selected_out [512]. */
void test_index_select(const uint16_t *scores, uint32_t count, bool highest_on_tie, int32_t *selected_out);

#endif
