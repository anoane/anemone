/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * hyper_connection_reference.h: the host reference of the hyper-connection kernels (hyper_connection_records.h):
 * what they must give, from what the model's reference code computes (model.py:288-293, 903-904, 957-966, 1253-1269;
 * kernel.py:406-460 as the source it generates computes it). The oracle of their card tests. Test support: no part
 * of the engine.
 *
 * Contract:
 *   - Written from the definitions and the orders hyper_connection_records.h states, with no call into the engine and
 *     no formula of a kernel. The tests link no mathematics library: the exponential is its series, a square root
 *     Newton's; its one function used here, fmaf (the reference's m s + b is fused), is one instruction under the
 *     build's -march.
 *   - test_hyper_sinkhorn: the float32 operations of the reference's Sinkhorn, each rounded, in its order (the sums
 *     of a row or a column (x_0 + x_2) + (x_1 + x_3)), with e^x the float32 nearest it. A card's expf is within 2
 *     places of that: its pre and post lie within TEST_HYPER_SINKHORN_PLACES float32 places of these, its comb
 *     within TEST_HYPER_COMB_TOLERANCE (comb's twenty steps carry 2 places of each exponential to up to 29 places of
 *     an entry, but never 2^-21.7 of one: measured on the host over 12.8 million entries, scales 1 to 20).
 *   - test_hyper_collapse: the float32 operations of model.py:959 in the order of the reference's library, rounded
 *     once to bfloat16: a card's bits. test_hyper_norm: model.py:288-293 in double of those bfloat16 values, with
 *     the weight: a card's bfloat16 lies within one place of it.
 *   - The norm as the records order it, each operation rounded: test_hyper_norm_operand gives q float32(1 / 5,120)
 *     + eps, q the sum of the squares in the records' fixed tree (a card's operand of rsqrtf, bit for bit);
 *     test_hyper_reciprocal_root the float32 nearest 1 / sqrt(operand), a tie decided within a double's places;
 *     test_hyper_norm_with bfloat16(w_d (h_d f)) of a factor f (a card's bits, given the factor its rsqrtf gave).
 *   - test_hyper_post: the float32 operations of model.py:965 (with routed_sums, first those of :903-904) in the
 *     reference's order, rounded once: a card's bits.
 *   - Arrays: streams [4][5,120] bfloat16 bits, one stream after another; pre and post [4]; comb [4][4], comb[j][k]
 *     the weight of stream j in output k; mixes [24]; scale [3]; base [24]; every other array [5,120].
 *   - test_hyper_float_places: how many float32 places apart two finite float32 values are (0 for both zeros).
 */
#ifndef TEST_HYPER_CONNECTION_REFERENCE_H
#define TEST_HYPER_CONNECTION_REFERENCE_H

#include <stdint.h>

#define TEST_HYPER_SINKHORN_PLACES 4u
#define TEST_HYPER_COMB_TOLERANCE 0x1p-20

void test_hyper_sinkhorn(const float *mixes, const float *scale, const float *base, float *pre_out, float *post_out,
                         float *comb_out);
void test_hyper_collapse(const uint16_t *streams, const float *pre, uint16_t *collapsed_out);
void test_hyper_norm(const uint16_t *collapsed, const uint16_t *weight, double *normed_out);
float test_hyper_norm_operand(const uint16_t *collapsed);
float test_hyper_reciprocal_root(float operand);   /* of a positive normal float32 */
void test_hyper_norm_with(const uint16_t *collapsed, const uint16_t *weight, float factor, uint16_t *normed_out);
/* routed_sums: NULL where the post adds none. */
void test_hyper_post(const uint16_t *streams, const float *post, const float *comb, const uint16_t *output,
                     const float *routed_sums, uint16_t *streams_out);
uint32_t test_hyper_float_places(float first, float second);

#endif
