/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * attention_state_reference.h: the host reference of the attention state (attention_state_records.h): what the
 * model's reference code computes (model.py RMSNorm, apply_rotary_emb, precompute_freqs_cis, Compressor.forward),
 * in double precision, and where model_layout.h places a value. The oracle of the attention state's tests. Test
 * support: no part of the engine.
 *
 * Contract:
 *   - Written from the definitions, with no call into the engine and no formula of a kernel. No function of the
 *     mathematics library is called (the tests are linked without it): the square root is Newton's, the exponential
 *     and the cosine and sine are their series after an exact reduction of the argument, each within a few places
 *     of a double, far below the last place of a float32.
 *   - test_attention_norm is weight * x / sqrt(mean of the squares + 1e-20), the weights bfloat16 bits. A rotation
 *     is the complex product of the pair and (cosine, sine). test_attention_pooled is the softmax of two scores
 *     applied to two values.
 *   - FP16 is IEEE binary16. test_attention_half_nearest rounds a double once: to nearest, a tie to even, to
 *     infinity from 65520, a not-a-number to 0x7E00, the sign of a zero kept. An ordinal is a whole number one apart
 *     for neighbouring values and 0 for both zeros, of an FP16 or a float32: a difference in last places.
 *   - Placements are found by walking the stored order of model_layout.h, not by inverting it: the index queries'
 *     [tile][step][lane][register][half] and a key's [run][step][value]; test_attention_layout_index_queries puts
 *     queries [32][128] in that order, test_attention_layout_key a key [128].
 *   - The window part of a row as the records order its float32 operations (the prepare's y 1, the window;
 *     dspark_v1/DESIGN.md C2): test_attention_window_operand is the norm's operand of wkv's bfloat16 output x [512],
 *     s (1 / 512) + 1e-20 of the sum of the squares in the records' order (a card's bits); given the factor f a card's
 *     rsqrtf made of it, test_attention_window_row is the ring row: w (x f) of the weights w, the pairs from dim 448
 *     on turned by angles [32][cos, sin] (fmaf(a, c, -(b s)), fmaf(b, c, a s)), to FP16. Of a card's rsqrtf only the
 *     factor is open: a card's bits for one factor near the reciprocal root of the operand.
 */
#ifndef TEST_ATTENTION_STATE_REFERENCE_H
#define TEST_ATTENTION_STATE_REFERENCE_H

#include <stdbool.h>
#include <stdint.h>

#define TEST_ATTENTION_HALF_NOT_A_NUMBER 0x7E00u

double test_attention_square_root(double value);         /* of a value above zero */
double test_attention_exponential(double value);         /* 0 below -700 */
void test_attention_cosine_sine(double angle, double *cosine_out, double *sine_out);

double test_attention_half_value(uint16_t bits);
uint16_t test_attention_half_nearest(double value);
int32_t test_attention_half_ordinal(uint16_t bits);
int32_t test_attention_float_ordinal(float value);
/* The distance from a value to the next of a format of fraction_bits bits whose normal numbers begin at
 * 2^smallest_exponent: bfloat16 (7, -126), FP16 (10, -14), float32 (23, -126). */
double test_attention_last_place(double value, int32_t fraction_bits, int32_t smallest_exponent);

/* values [width] as doubles, weights [width] bfloat16 bits; normed_out [width]. */
void test_attention_norm(const double *values, const uint16_t *weights, uint32_t width, double *normed_out);
void test_attention_rotate(double *real, double *imaginary, double cosine, double sine);
double test_attention_pooled(double first_kv, double first_score, double kv, double score);

/* halves [32 * 128] of one row, head-major, into placed_out [32 * 128] (8,192 bytes). */
void test_attention_layout_index_queries(const uint16_t *halves, uint16_t *placed_out);
/* halves [128] of one key into placed_out [128] (256 bytes). */
void test_attention_layout_key(const uint16_t *halves, uint16_t *placed_out);

float test_attention_window_operand(const uint16_t *window_input);
void test_attention_window_row(const uint16_t *window_input, const uint16_t *weights, float factor,
                               const float *angles, uint16_t *ring_row_out);

#endif
