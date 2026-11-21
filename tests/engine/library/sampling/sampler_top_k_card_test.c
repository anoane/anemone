/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Prepared GPU gate: independent rank masks, nucleus/race reference and unset equivalence. */
#include "engine/library/sampling/sampler.h"
#include "engine/library/block_scaled_linear/linear_card_fixture.h"
#include "sampler_reference.h"
#include "support/byte_identity.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#define VOCABULARY 32u
#define WORK_BYTES 65536u
#define DEADLINE_SECONDS 120u
#define LOGIT_GROUP 3u
#define LOGIT_BASE 9.0f
#define NUCLEUS 0.6f
#define SEED 0x123456789ABCDEF0ull
#define COOL 0.5f
#define TOP_TWO 2u
#define TOP_FIVE 5u
static bool rank_kept(const float *logits, uint32_t token, uint32_t top_k)
{
	uint32_t rank = 0;
	for (uint32_t other = 0; other < VOCABULARY; other++)
		rank += logits[other] > logits[token] ||
					(other < token &&
					 logits[other] == logits[token]) ?
				1u :
				0u;
	return top_k == 0 || rank < top_k;
}
static void cases(struct test_linear_card *card,
		  const struct sampler_kernels_record *kernels)
{
	static const uint32_t counts[] = {
		0, 1, TOP_TWO, TOP_FIVE, VOCABULARY, VOCABULARY + 1u
	};
	static const float temperatures[] = { 0.0f, COOL, 1.0f };
	static const float nuclei[] = { NUCLEUS, 1.0f };
	float logits[VOCABULARY];
	for (uint32_t token = 0; token < VOCABULARY; token++) {
		const uint32_t logit_group = token / LOGIT_GROUP;
		logits[token] = LOGIT_BASE - (float)logit_group;
	}
	for (size_t temperature = 0;
	     temperature < sizeof(temperatures) / sizeof(temperatures[0]);
	     temperature++) {
		for (size_t nucleus = 0;
		     nucleus < sizeof(nuclei) / sizeof(nuclei[0]); nucleus++) {
			struct step_row_result baseline = { 0 };
			float baseline_weights[VOCABULARY] = { 0 };
			for (size_t count = 0;
			     count < sizeof(counts) / sizeof(counts[0]);
			     count++) {
				test_linear_card_clear(card);
				const struct linear_range step =
					test_linear_card_work(
						card,
						STEP_BLOCK_BYTES_VALUE +
							STEP_ROW_BYTES_VALUE);
				const struct linear_range sessions =
					test_linear_card_work(
						card,
						sizeof(struct step_session));
				const struct linear_range values =
					test_linear_card_work(card,
							      sizeof(logits));
				const struct linear_range weights =
					test_linear_card_work(card,
							      sizeof(logits));
				const struct linear_range result =
					test_linear_card_work(
						card,
						STEP_RESULT_BYTES +
							STEP_ROW_RESULT_BYTES);
				const struct step_block block = {
					.sequence = 1, .row_count = 1
				};
				const struct step_row_record row = {
					.flags = STEP_ROW_LOGITS
				};
				const struct step_session settings = {
					.seed = { (uint32_t)SEED,
						  (uint32_t)(SEED >>
							     STEP_SESSION_SEED_BITS) },
					.temperature =
						temperatures[temperature],
					.top_p = nuclei[nucleus],
					.top_k = counts[count]
				};
				const struct sampler_description description = {
					.step = step,
					.sessions = sessions,
					.logits = values,
					.weights = weights,
					.result = result,
					.vocabulary_size = VOCABULARY,
					.session_capacity = 1,
					.row_count = 1
				};
				struct kernel_launch launch = { 0 };
				struct step_row_result drawn = { 0 };
				float actual[VOCABULARY] = { 0 };
				bool ran =
					test_linear_card_put(card, step, 0,
							     &block,
							     sizeof(block)) &&
					test_linear_card_put(card, step,
							     sizeof(block),
							     &row,
							     sizeof(row)) &&
					test_linear_card_put(
						card, sessions, 0, &settings,
						sizeof(settings)) &&
					test_linear_card_put(card, values, 0,
							     logits,
							     sizeof(logits)) &&
					test_linear_expect_status(
						sampler_prepare_entry(
							&description,
							card->arena_byte_count,
							kernels, &launch),
						STATUS_OK,
						"prepare top-k sampler") &&
					test_linear_card_load(card) &&
					test_linear_card_run(card, &launch,
							     1) &&
					test_linear_card_fetch(card) &&
					test_linear_card_get(card, result,
							     STEP_RESULT_BYTES,
							     &drawn,
							     sizeof(drawn));
				if (!test_linear_expect(
					    ran, "execute top-k sampler"))
					continue;
				if (settings.temperature == 0.0f)
					test_linear_expect(
						drawn.token_id == 0,
						"temperature zero ignores truncation and chooses first maximum");
				else {
					test_linear_card_get(card, weights, 0,
							     actual,
							     sizeof(actual));
					for (uint32_t token = 0;
					     token < VOCABULARY; token++) {
						const float expected =
							rank_kept(
								logits, token,
								settings.top_k) ?
								test_sampler_weight(
									logits[token],
									logits[0],
									settings.temperature) :
								0.0f;
						test_linear_expect(
							test_sampler_within_places(
								actual[token],
								expected,
								TEST_SAMPLER_WEIGHT_PLACES),
							"top-k mask follows logit rank with lowest-ID ties before "
							"temperature/nucleus");
					}
					uint32_t threshold = 0,
						 kept[VOCABULARY];
					if (test_linear_expect(
						    test_sampler_threshold(
							    actual, VOCABULARY,
							    settings.top_p,
							    &threshold),
						    "reference nucleus")) {
						const uint32_t kept_count =
							test_sampler_kept(
								actual,
								VOCABULARY,
								threshold,
								kept);
						struct test_sampler_race
							reference = { 0 };
						test_sampler_race(
							actual, kept,
							kept_count,
							step_session_seed(
								&settings),
							0, &reference);
						test_linear_expect(
							test_sampler_race_allows(
								&reference,
								(uint32_t)drawn
									.token_id),
							"top-k and top-p keyed race matches independent reference");
					}
				}
				if (count == 0) {
					baseline = drawn;
					memcpy(baseline_weights, actual,
					       sizeof(actual));
				}
				if (settings.top_k >= VOCABULARY)
					test_linear_expect(
						bytes_identical(
							&baseline, &drawn,
							sizeof(drawn)) &&
							bytes_identical(
								baseline_weights,
								actual,
								sizeof(actual)),
						"untruncated top-k is byte-identical to unset");
				test_linear_card_expect_untouched(
					card, step, "step unchanged");
			}
		}
	}
}
int main(void)
{
	(void)alarm(DEADLINE_SECONDS);
	uint32_t cards = 0;
	if (!test_linear_expect_status(backend_cuda_value.card_count(&cards),
				       STATUS_OK, "GPU count"))
		return EXIT_FAILURE;
	for (uint32_t index = 0; index < cards; index++) {
		struct test_linear_card card = { 0 };
		struct sampler_kernels_record kernels = { 0 };
		if (test_linear_card_open(&card, index, WORK_BYTES, 0) &&
		    test_linear_expect_status(
			    sampler_kernels_prepare(&backend_cuda_value,
						    card.card, &kernels),
			    STATUS_OK, "sampler kernels"))
			cases(&card, &kernels);
		test_linear_card_close(&card);
	}
	(void)printf("top-k CUDA gate: %d failures\n", test_linear_failures());
	return test_linear_failures() == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
