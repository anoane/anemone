/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * router_reference.h: the host reference of the router (router_records.h), from the reference's definitions
 * (model.py Gate.forward, REF:811-826; the order in which MoE.forward adds a row's experts, REF:893-900). The oracle
 * of the router's tests. Test support: no part of the engine.
 *
 * Contract:
 *   - test_router_affinity is sqrt(softplus(score)) in double, softplus(score) being score above 20 and else
 *     log(1 + e^score), with no function of the mathematics library (the tests are linked without it): the
 *     exponential by its series on a 1,024th of its argument, squared ten times; the logarithm as 2 atanh(y / (2 +
 *     y)) by its series, of y = e^-|score| <= 1; the root by Newton's steps on a number brought into [1/4, 1) by
 *     powers of four. A not-a-number gives one, +infinity itself. Each result is within 10^-12 of its value.
 *   - test_router_affinity_holds says whether a card's float32 is the affinity of a score: a not-a-number or an
 *     infinity where the reference has one, else within TEST_ROUTER_AFFINITY_PLACES last places of it (the card's
 *     expf and log1pf are within 2 and 1 places and its sqrtf rounds once: 2.4e-7 of the value, 4 places at the top
 *     of a binade); where e^score is below the normal float32 range, where the card's exponential keeps fewer bits,
 *     its square within TEST_ROUTER_SUBNORMAL_SLACK of softplus.
 *   - test_router_select chooses from affinities as a card computed them, in float32 arithmetic: of the 384 experts
 *     the six largest affinity + bias, one at a time, in the order of the reference's top-k (a not-a-number above
 *     every number, -0 below +0, the lower expert first among equals); their weights as router_records.h states for
 *     the order given. experts_out holds them by slot (ascending), weights_out the weight of each.
 *   - test_router_jobs is the job table of live_rows rows of choices, for a launch of row_count rows (row_count * 6
 *     entries): the experts walked in ascending order, each with the rows that chose it in ascending order, every
 *     byte not named zero.
 *   - test_router_select_draft and test_router_draft_jobs are the two above for the draft's router (router_records.h:
 *     router_select_draft): 128 experts, 3 picks, the fixed order; row_count * 3 entries, pairs row * 3 +
 *     slot.
 *   - test_router_jobs_differ counts the entries of two tables that differ: bytes, but for weights that are both a
 *     not-a-number (the card's and the host's differ in their payload).
 *   - test_router_wide_sort is the wide select's arrays (router_records.h, wide_linear_records.h) of live_rows rows of
 *     choices, for a launch of row_count rows: a stable sort of the pairs by expert, by walking the experts in
 *     ascending order and, for each, the rows in ascending order (no count, no prefix); its 6 live_rows sorted pairs
 *     (nothing behind them is written), its 384 runs, and every one of its ROUTER_WIDE_TILES(row_count)
 *     tiles: each run cut into tiles of 64 pairs, the last ragged, with the expert's three offsets, then zeros.
 *     test_router_wide_pairs_differ counts the pairs of two arrays that differ, as test_router_jobs_differ does.
 *   - test_router_wide_jobs is the wide select's job table of the same choices, every one of its
 *     ROUTER_WIDE_JOBS(row_count) entries: the experts walked in ascending order, the pairs of each (rows
 *     ascending, then slots) cut in jobs of eight, the last ragged, each with the expert's three offsets; every byte
 *     not named zero. It reads the choices, not the sort's pairs.
 *   Every function may be called from any thread.
 */
#ifndef TEST_ROUTER_REFERENCE_H
#define TEST_ROUTER_REFERENCE_H

#include <stdbool.h>
#include <stdint.h>

#include "anemone/linear_records.h"
#include "anemone/model_layout.h"
#include "anemone/wide_linear_records.h"

#define TEST_ROUTER_AFFINITY_PLACES 4u
#define TEST_ROUTER_SUBNORMAL_SLACK 0x1p-146

double test_router_affinity(float score);
bool test_router_affinity_holds(float score, float affinity);
void test_router_select(const float *affinities, const float *bias, bool reference_order, uint32_t *experts_out,
                        float *weights_out);
void test_router_jobs(const uint32_t (*experts)[MODEL_EXPERTS_PER_ROW],
		      const float (*weights)[MODEL_EXPERTS_PER_ROW],
		      uint32_t live_rows, uint32_t row_count,
		      const uint64_t *expert_offsets,
		      struct expert_job_record *jobs_out);
uint32_t test_router_jobs_differ(const struct expert_job_record *first,
				 const struct expert_job_record *second,
				 uint32_t job_count);
void test_router_select_draft(const float *affinities, const float *bias, uint32_t *experts_out, float *weights_out);
void test_router_draft_jobs(
	const uint32_t (*experts)[MODEL_DRAFT_EXPERTS_PER_ROW],
	const float (*weights)[MODEL_DRAFT_EXPERTS_PER_ROW], uint32_t live_rows,
	uint32_t row_count, const uint64_t *expert_offsets,
	struct expert_job_record *jobs_out);
void test_router_wide_sort(const uint32_t (*experts)[MODEL_EXPERTS_PER_ROW],
			   const float (*weights)[MODEL_EXPERTS_PER_ROW],
			   uint32_t live_rows, uint32_t row_count,
			   const uint64_t *expert_offsets,
			   struct wide_pair *pairs_out,
			   struct wide_run *runs_out,
			   struct wide_expert_tile *tiles_out);
uint32_t test_router_wide_pairs_differ(const struct wide_pair *first,
				       const struct wide_pair *second,
				       uint32_t pair_count);
void test_router_wide_jobs(const uint32_t (*experts)[MODEL_EXPERTS_PER_ROW],
			   const float (*weights)[MODEL_EXPERTS_PER_ROW],
			   uint32_t live_rows, uint32_t row_count,
			   const uint64_t *expert_offsets,
			   struct expert_job_record *jobs_out);

#endif
