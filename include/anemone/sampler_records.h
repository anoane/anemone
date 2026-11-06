/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * sampler_records.h: the record of the sampler, which answers each row of a step on the card that samples, and what
 * it computes. C11 and CUDA, including of the engine the record layouts of the step and the linear alone: for
 * kernel_sampler.cu, sampler.c, tests.
 *
 * Contract (step_records.h: the step, session and result areas; linear_records.h: a kernel checks nothing):
 *   - Launch: grid (1, R, 1) of blocks of (SAMPLER_THREADS, 1, 1), no dynamic shared memory. Block 0 writes
 *     the result's sequence and row_count, the step's (the checked modules trap on more than R, the release modules
 *     take R); block r below that row_count answers row r from logits [r][vocabulary_size] and, if the row draws,
 *     writes weights [r][vocabulary_size]. Nothing else is written.
 *   - No STEP_ROW_LOGITS: STEP_NO_TOKEN and +0. Else the token below and l at it, its bits as read.
 *     l_max is the first largest logit, never a not-a-number (of a row of no number: token 0); x_t is l_t (1 /
 *     max(temperature, SAMPLER_TEMPERATURE_FLOOR)) in float32, as REF:1290 rounds it. Greedy, at
 *     temperature 0 or an x_max that is not finite: the token of l_max (REF:1288-1289).
 *   - Keyed: e_t = expf(x_t - x_max) in float32, +0 for a not-a-number: the exponentials of REF:1291's softmax, rounded
 *     alike. m_t = e_t 2^32 rounded to a whole number, a tie to even, W their sum (at most 2^53: exact in double).
 *     Kept, for top_p below 1: the t of e_t >= tau, the largest float32 with the m_t of e_t >= tau adding up to the
 *     double product top_p W rounded up, or more; for top_p 1: every t. The token is the kept t of the largest
 *     e_t / -logf(u_t), the lowest t among equals: the reference's race of probabilities over exponential draws
 *     (REF:1292), its draws keyed. u_t = ((h >> SAMPLER_UNIFORM_SHIFT) + 1/2) 2^-23 of h =
 *     pack_digest_term(pack_digest_mix(seed), 2^32 position + t) (pack_digest.h) of the session's
 *     seed and the row's position.
 *   - Nothing of a row's answer depends on the other rows, on R, or on an order of additions. The masses are summed
 *     by integer atomic additions in shared memory, whose sum no order changes: no float and no global atomics.
 */
#ifndef SAMPLER_RECORDS_H
#define SAMPLER_RECORDS_H

#include <stdint.h>

#include "anemone/linear_records.h"
#include "anemone/step_records.h"

#define SAMPLER_THREADS 1024u
#define SAMPLER_TOKENS_PER_LOAD_VALUE \
	4u /* a thread loads 16 bytes of a row at once */
#define SAMPLER_MOST_VOCABULARY \
	2097152u /* 2^21 tokens: W, at most 2^53, is exact in double */
#define SAMPLER_TEMPERATURE_FLOOR 1e-5f /* REF:1290 */
#define SAMPLER_MASS_SCALE 4294967296.0f /* 2^32 */
#define SAMPLER_POSITION_SHIFT 32u
#define SAMPLER_UNIFORM_SHIFT 41u /* the top 23 bits of h */
#define SAMPLER_UNIFORM_SCALE 1.1920928955078125e-7f /* 2^-23 */
#define KERNEL_FILE_SAMPLER "kernel_sampler"
#define KERNEL_SAMPLER "token_sampler"
#define SAMPLER_ARGUMENT_BYTES 48u

/* kernel_sampler, token_sampler. With R = grid[1] rows:
 *   reads   step_offset: the step area, its block and R rows; session_offset: the settings [session_capacity];
 *           logits_offset: float32 [R][vocabulary_size];
 *   writes  weights_offset: float32 [R][vocabulary_size]; result_offset: the result area, its block and R rows. */
struct sampler_arguments {
	uint64_t step_offset;
	uint64_t session_offset;
	uint64_t logits_offset;
	uint64_t weights_offset;
	uint64_t result_offset;
	uint32_t vocabulary_size; /* a multiple of SAMPLER_TOKENS_PER_LOAD_VALUE, at most the most */
	uint32_t session_capacity;
};
LINEAR_RECORD_HAS(sampler_arguments, SAMPLER_ARGUMENT_BYTES);

#endif
