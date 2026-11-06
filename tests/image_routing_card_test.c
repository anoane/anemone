/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Deferred card test: mixed text/image rows select independent VL bias and suppress image Engram gates.
 * Select exactly one approved physical card using CUDA_VISIBLE_DEVICES before running this executable.
 */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "anemone/image_routing.h"
#include "linear_card_fixture.h"
#include "router_reference.h"

#define TEST_ROWS 2u
#define IMAGE_ROW 1u
#define WORK_BYTES (2ull << 20u)
#define WEIGHT_BYTES 65536u
#define MATRIX_BYTES 2176u
#define TEST_SECONDS 120u
#define HIGH_BIAS 100.0f
#define BF16_ONE 0x3f80u
#define BF16_HALF 0x3f00u
#define TEST_JOBS (TEST_ROWS * MODEL_EXPERTS_PER_ROW)
#define TABLE_ENTRIES (MODEL_EXPERTS * MODEL_EXPERT_MATRICES)
#define STREAM_VALUES (MODEL_STREAMS * MODEL_HIDDEN)
#define ROW_VALUES ENGRAM_ROWS_PROJECTION
#define LANES LINEAR_THREADS_PER_WARP

static void launch_fill(struct kernel_launch *launch, uint32_t kernel,
			uint32_t columns, uint32_t rows, uint32_t threads,
			const void *record, uint32_t bytes)
{
	*launch = (struct kernel_launch){ .kernel = kernel,
					  .grid = { columns, rows, 1 },
					  .block = { threads, 1, 1 },
					  .argument_byte_count = bytes };
	memcpy(launch->argument, record, bytes);
}

static bool router_run(struct test_linear_card *card,
		       const struct image_routing_kernels *kernels)
{
	struct step_block block = { .sequence = 1, .row_count = TEST_ROWS };
	struct step_row_record rows[TEST_ROWS] = { { 0 } };
	rows[IMAGE_ROW].flags = STEP_ROW_IMAGE;
	const struct linear_range step =
		test_linear_card_work(card, sizeof(block) + sizeof(rows));
	const struct linear_range scores = test_linear_card_work(
		card, (uint64_t)TEST_ROWS * MODEL_ROUTER_ROW_BYTES);
	const struct linear_range bias =
		test_linear_card_work(card, MODEL_ROUTER_ROW_BYTES);
	const struct linear_range image_bias =
		test_linear_card_work(card, MODEL_ROUTER_ROW_BYTES);
	const struct linear_range offsets_range =
		test_linear_card_work(card, MODEL_EXPERT_OFFSETS_BYTES);
	const struct linear_range affinities = test_linear_card_work(
		card, (uint64_t)TEST_ROWS * MODEL_ROUTER_ROW_BYTES);
	const struct linear_range jobs = test_linear_card_work(
		card, (uint64_t)TEST_JOBS * EXPERT_JOB_BYTES);
	const struct linear_range picks = test_linear_card_work(
		card, (uint64_t)TEST_JOBS * ROUTER_PICK_BYTES);
	const struct linear_range weights =
		test_linear_card_weights(card, MATRIX_BYTES);
	float text_bias[MODEL_EXPERTS] = { 0 },
	      vision_bias[MODEL_EXPERTS] = { 0 };
	for (uint32_t index = 0; index < MODEL_EXPERTS_PER_ROW; index++) {
		text_bias[index] = HIGH_BIAS;
		vision_bias[MODEL_EXPERTS - MODEL_EXPERTS_PER_ROW + index] =
			HIGH_BIAS;
	}
	uint64_t offsets[TABLE_ENTRIES] = { 0 };
	for (uint32_t index = 0; index < TABLE_ENTRIES; index++) {
		offsets[index] = weights.byte_offset;
	}
	struct image_router_arguments record = {
		.base = { .step_offset = step.byte_offset,
			  .scores_offset = scores.byte_offset,
			  .bias_offset = bias.byte_offset,
			  .expert_offsets_offset = offsets_range.byte_offset,
			  .affinities_offset = affinities.byte_offset,
			  .jobs_offset = jobs.byte_offset,
			  .weights_offset = weights.byte_offset,
			  .gate_up_last_offset = weights.byte_offset,
			  .down_last_offset = weights.byte_offset,
			  .row_count = TEST_ROWS },
		.vision_bias_offset = image_bias.byte_offset
	};
	struct kernel_launch launch;
	launch_fill(&launch, kernels->select, 1, 1, ROUTER_THREADS, &record,
		    sizeof(record));
	if (!test_linear_card_has_room(card) ||
	    !test_linear_card_put(card, step, 0, &block, sizeof(block)) ||
	    !test_linear_card_put(card, step, sizeof(block), rows,
				  sizeof(rows)) ||
	    !test_linear_card_put(card, bias, 0, text_bias,
				  sizeof(text_bias)) ||
	    !test_linear_card_put(card, image_bias, 0, vision_bias,
				  sizeof(vision_bias)) ||
	    !test_linear_card_put(card, offsets_range, 0, offsets,
				  sizeof(offsets)) ||
	    !test_linear_card_load(card) ||
	    !test_linear_card_run(card, &launch, 1) ||
	    !test_linear_card_fetch(card)) {
		return false;
	}
	float actual_affinities[TEST_ROWS][MODEL_EXPERTS] = { { 0 } };
	struct expert_job_record actual[TEST_JOBS], expected[TEST_JOBS];
	uint32_t expected_experts[TEST_ROWS][MODEL_EXPERTS_PER_ROW] = { { 0 } };
	float expected_weights[TEST_ROWS][MODEL_EXPERTS_PER_ROW] = { { 0 } };
	if (!test_linear_card_get(card, affinities, 0, actual_affinities,
				  sizeof(actual_affinities)) ||
	    !test_linear_card_get(card, jobs, 0, actual, sizeof(actual))) {
		return false;
	}
	for (uint32_t row = 0; row < TEST_ROWS; row++) {
		test_router_select(actual_affinities[row],
				   row == IMAGE_ROW ? vision_bias : text_bias,
				   false, expected_experts[row],
				   expected_weights[row]);
	}
	test_router_jobs(
		(const uint32_t (*)[MODEL_EXPERTS_PER_ROW])expected_experts,
		(const float (*)[MODEL_EXPERTS_PER_ROW])expected_weights,
		TEST_ROWS, TEST_ROWS, offsets, expected);
	if (!test_linear_expect(
		    test_router_jobs_differ(actual, expected, TEST_JOBS) == 0,
		    "mixed image rows use the reference VL bias independently") ||
	    !test_linear_card_expect_untouched(card, step,
					       "step rows are read only") ||
	    !test_linear_card_expect_untouched(card, bias,
					       "text bias is read only") ||
	    !test_linear_card_expect_untouched(card, image_bias,
					       "image bias is read only")) {
		return false;
	}
	struct image_router_wide_arguments wide = {
		.base = { .step_offset = step.byte_offset,
			  .scores_offset = scores.byte_offset,
			  .bias_offset = bias.byte_offset,
			  .affinities_offset = affinities.byte_offset,
			  .picks_offset = picks.byte_offset,
			  .row_count = TEST_ROWS },
		.vision_bias_offset = image_bias.byte_offset
	};
	launch_fill(&launch, kernels->wide_select, 1, 1, ROUTER_THREADS, &wide,
		    sizeof(wide));
	struct router_pick actual_picks[TEST_ROWS][MODEL_EXPERTS_PER_ROW];
	if (!test_linear_card_run(card, &launch, 1) ||
	    !test_linear_card_fetch(card) ||
	    !test_linear_card_get(card, picks, 0, actual_picks,
				  sizeof(actual_picks))) {
		return false;
	}
	bool valid = true;
	for (uint32_t row = 0; row < TEST_ROWS; row++) {
		for (uint32_t slot = 0; slot < MODEL_EXPERTS_PER_ROW; slot++) {
			valid = valid &&
				actual_picks[row][slot].expert ==
					expected_experts[row][slot] &&
				actual_picks[row][slot].route_weight ==
					expected_weights[row][slot];
		}
	}
	return test_linear_expect(
		valid,
		"wide image select keeps the thin select's experts and weights");
}

static bool engram_run(struct test_linear_card *card,
		       const struct image_routing_kernels *kernels)
{
	test_linear_card_clear(card);
	struct step_block block = { .sequence = 1, .row_count = TEST_ROWS };
	struct step_row_record rows[TEST_ROWS] = { { 0 } };
	rows[IMAGE_ROW].flags = STEP_ROW_IMAGE;
	const struct linear_range step =
		test_linear_card_work(card, sizeof(block) + sizeof(rows));
	const struct linear_range projection = test_linear_card_work(
		card, (uint64_t)TEST_ROWS * ROW_VALUES * sizeof(uint16_t));
	const struct linear_range query = test_linear_card_work(
		card, (uint64_t)STREAM_VALUES * sizeof(uint16_t));
	const struct linear_range key = test_linear_card_work(
		card, (uint64_t)STREAM_VALUES * sizeof(uint16_t));
	const struct linear_range streams = test_linear_card_work(
		card, (uint64_t)TEST_ROWS * (uint64_t)STREAM_VALUES *
			      sizeof(uint16_t));
	static uint16_t projection_values[TEST_ROWS][ROW_VALUES],
		weight_values[STREAM_VALUES];
	static uint16_t output[TEST_ROWS][STREAM_VALUES];
	memset(projection_values, 0, sizeof(projection_values));
	for (uint32_t index = 0; index < STREAM_VALUES; index++) {
		weight_values[index] = BF16_ONE;
	}
	for (uint32_t row = 0; row < TEST_ROWS; row++) {
		for (uint32_t index = STREAM_VALUES; index < ROW_VALUES;
		     index++) {
			projection_values[row][index] = BF16_ONE;
		}
	}
	const struct image_engram_arguments record = {
		.base = { .projection_offset = projection.byte_offset,
			  .query_weight_offset = query.byte_offset,
			  .key_weight_offset = key.byte_offset,
			  .streams_offset = streams.byte_offset },
		.step_offset = step.byte_offset
	};
	struct kernel_launch launch;
	launch_fill(&launch, kernels->engram_gate, MODEL_STREAMS, TEST_ROWS,
		    ENGRAM_ROWS_GATE_THREADS, &record, sizeof(record));
	if (!test_linear_card_has_room(card) ||
	    !test_linear_card_put(card, step, 0, &block, sizeof(block)) ||
	    !test_linear_card_put(card, step, sizeof(block), rows,
				  sizeof(rows)) ||
	    !test_linear_card_put(card, projection, 0, projection_values,
				  sizeof(projection_values)) ||
	    !test_linear_card_put(card, query, 0, weight_values,
				  sizeof(weight_values)) ||
	    !test_linear_card_put(card, key, 0, weight_values,
				  sizeof(weight_values)) ||
	    !test_linear_card_load(card) ||
	    !test_linear_card_run(card, &launch, 1) ||
	    !test_linear_card_fetch(card) ||
	    !test_linear_card_get(card, streams, 0, output, sizeof(output))) {
		return false;
	}
	bool valid = true;
	for (uint32_t index = 0; index < STREAM_VALUES; index++) {
		valid = valid && output[0][index] == BF16_HALF &&
			output[IMAGE_ROW][index] == 0;
	}
	return test_linear_expect(
		valid,
		"image Engram rows remain unchanged while text rows receive their gate");
}

int main(void)
{
	(void)alarm(TEST_SECONDS);
	uint32_t count = 0;
	struct card_description_record description = { 0 };
	if (!test_linear_expect_status(backend_cuda_value.card_count(&count),
				       STATUS_OK, "count cards") ||
	    !test_linear_expect(count == 1u,
				"exactly one approved card must be visible") ||
	    !test_linear_expect_status(
		    backend_cuda_value.card_describe(0, &description),
		    STATUS_OK, "describe selected card") ||
	    !test_linear_expect(strstr(description.name, "170HX") != NULL ||
					strstr(description.name,
					       "RTX PRO 6000") != NULL,
				"selected card has an approved device name")) {
		return EXIT_FAILURE;
	}
	static struct test_linear_card card;
	struct image_routing_kernels kernels = { 0 };
	bool valid =
		test_linear_card_open(&card, 0, WORK_BYTES, WEIGHT_BYTES) &&
		test_linear_expect_status(
			image_routing_kernels_prepare(&backend_cuda_value,
						      card.card, &kernels),
			STATUS_OK, "prepare image routing kernels");
	valid = valid && router_run(&card, &kernels) &&
		engram_run(&card, &kernels);
	test_linear_card_close(&card);
	return valid && test_linear_failures() == 0 ? EXIT_SUCCESS :
						      EXIT_FAILURE;
}
