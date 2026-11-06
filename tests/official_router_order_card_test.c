/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Source-order normalization witness: the qualified width32 topk order of six
 * equal selected keys is [1,0,4,5,2,3]. Unequal power-of-two affinities make its
 * sum exactly the next F32 above one, whereas ascending-expert accumulation
 * gives one. Expected weights use that independently specified denominator.
 * All bytes except six live weights are protected, including a malformed job.
 */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "anemone/official_router_order.h"
#include "linear_card_fixture.h"

#define WORK_BYTES (1ull << 20u)
#define WEIGHT_BYTES (1ull << 16u)
#define TIME_LIMIT_SECONDS 120u
#define BIASED_CUT 64.0f
#define INITIAL_ROUTE_WEIGHT (-123.0f)
#define SOURCE_DENOMINATOR_BITS 0x3f800001u
#define REPEATS 2u

static const uint32_t affinity_bits[MODEL_EXPERTS_PER_ROW] = {
	0x3f800000u, 0x33800000u, 0x33000000u,
	0x32800000u, 0x32000000u, 0x31800000u
};

static float float_of(uint32_t bits)
{
	float value;
	memcpy(&value, &bits, sizeof(value));
	return value;
}

static bool check_case(struct test_linear_card *card, uint32_t kernel,
		       bool malformed)
{
	test_linear_card_clear(card);
	test_linear_context("source router equal-key witness malformed %u",
			    malformed ? 1u : 0u);
	const struct linear_range step_range =
		test_linear_card_work(card, STEP_BLOCK_BYTES_VALUE);
	const struct linear_range affinities =
		test_linear_card_work(card, MODEL_ROUTER_ROW_BYTES);
	const struct linear_range bias =
		test_linear_card_work(card, MODEL_ROUTER_ROW_BYTES);
	const struct linear_range jobs_range = test_linear_card_work(
		card, (uint64_t)MODEL_EXPERTS_PER_ROW * EXPERT_JOB_BYTES);
	if (!test_linear_card_has_room(card))
		return false;
	const struct step_block step = { .sequence = 1, .row_count = 1 };
	struct expert_job_record jobs[MODEL_EXPERTS_PER_ROW] = { 0 };
	struct expert_job_record expected[MODEL_EXPERTS_PER_ROW];
	bool passed =
		test_linear_card_put(card, step_range, 0, &step, sizeof(step));
	for (uint32_t slot = 0; passed && slot < MODEL_EXPERTS_PER_ROW;
	     slot++) {
		const float affinity = float_of(affinity_bits[slot]);
		const float offset = BIASED_CUT - affinity;
		passed = test_linear_card_put(card, affinities,
					      (uint64_t)slot * sizeof(affinity),
					      &affinity, sizeof(affinity)) &&
			 test_linear_card_put(card, bias,
					      (uint64_t)slot * sizeof(offset),
					      &offset, sizeof(offset));
		jobs[slot].expert = slot;
		jobs[slot].row_count = 1;
		jobs[slot].pair_index[0] = slot;
		jobs[slot].route_weight[0] = INITIAL_ROUTE_WEIGHT;
	}
	if (malformed)
		jobs[0].row_count = REPEATS;
	memcpy(expected, jobs, sizeof(expected));
	if (!malformed) {
		const float denominator = float_of(SOURCE_DENOMINATOR_BITS);
		for (uint32_t slot = 0; slot < MODEL_EXPERTS_PER_ROW; slot++)
			expected[slot].route_weight[0] =
				float_of(affinity_bits[slot]) /
				(denominator + ROUTER_WEIGHT_EPSILON) *
				MODEL_ROUTE_SCALE;
	}
	passed = passed &&
		 test_linear_card_put(card, jobs_range, 0, jobs, sizeof(jobs));
	const struct router_select_arguments arguments = {
		.step_offset = step_range.byte_offset,
		.bias_offset = bias.byte_offset,
		.affinities_offset = affinities.byte_offset,
		.jobs_offset = jobs_range.byte_offset,
		.row_count = 1,
		.reference_order = 1
	};
	struct kernel_launch launch = { .kernel = kernel,
					.grid = { 1, 1, 1 },
					.block = { 1, 1, 1 },
					.argument_byte_count =
						ROUTER_SELECT_ARGUMENT_BYTES };
	memcpy(launch.argument, &arguments, sizeof(arguments));
	for (uint32_t repeat = 0; passed && repeat < REPEATS; repeat++)
		passed =
			test_linear_card_load(card) &&
			test_linear_card_run(card, &launch, 1) &&
			test_linear_card_fetch(card) &&
			test_linear_card_expect_untouched(card, step_range,
							  "step preserved") &&
			test_linear_card_expect_untouched(
				card, affinities, "affinities preserved") &&
			test_linear_card_expect_untouched(
				card, bias, "selection biases preserved") &&
			test_linear_card_expect_bytes(
				card, jobs_range, 0, expected, sizeof(expected),
				malformed ?
					"malformed job refusal is atomic" :
					"source sum and only live weights change");
	return passed;
}

int main(void)
{
	(void)alarm(TIME_LIMIT_SECONDS);
	struct test_linear_card card = { 0 };
	bool passed = test_linear_card_open(&card, 0, WORK_BYTES, WEIGHT_BYTES);
	uint32_t kernel = BACKEND_NO_KERNEL;
	passed = passed &&
		 test_linear_expect_status(
			 official_router_order_kernel_prepare(
				 &backend_cuda_value, card.card, &kernel),
			 STATUS_OK, "prepare independent source router pass") &&
		 check_case(&card, kernel, false) &&
		 check_case(&card, kernel, true);
	test_linear_card_close(&card);
	(void)printf("official_router_order_card_test: %s; %d failures\n",
		     passed ? "passed" : "failed", test_linear_failures());
	return passed && test_linear_failures() == 0 ? EXIT_SUCCESS :
						       EXIT_FAILURE;
}
