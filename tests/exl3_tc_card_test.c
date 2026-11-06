/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * exl3_tc_card_test.c: the tensor-core EXL3 MoE (exl3_tc.h, --exl3_tc_moe) against the engine's EXL3 routed experts
 * (exl3_experts_fast_prepare and the row sum) on one card, from the same job table (router_records.h: the
 * routed experts ascending, a run of an expert's pairs cut in jobs of at most 8, then empty jobs), the same synthetic
 * EXL3 matrices (random mul1 trellis words, suh and svh of both signs) and the same bfloat16 rows. The two orders of
 * arithmetic differ (GNmac's FP16 hi/lo split against the engine's), so the FP32 row sums are compared with a
 * tolerance: the largest difference relative to the largest magnitude of a case at most MOST_RELATIVE, and the
 * root-mean-square relative difference at most MOST_RMS. Cases: rows 8 with 6 live (a thin K5 verify), 48 and 64,
 * intermediate 512 and 640, K3. It needs the cards free (gpu_run).
 */
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "anemone/backend_seam.h"
#include "anemone/exl3.h"
#include "anemone/exl3_tc.h"
#include "anemone/exl3_tc_records.h"
#include "anemone/exl3_tp.h"
#include "anemone/linear_records.h"
#include "linear_card_fixture.h"

#define HIDDEN 5120u
#define EXPERTS 48u
#define PAIRS EXL3_TC_PAIRS_PER_ROW
#define JOB_ROWS LINEAR_MAXIMUM_ROWS
#define BITS 3u
#define ACTIVATION_LIMIT 10.0f
#define WORK_CAPACITY (512ull << 20)
#define WEIGHTS_CAPACITY (768ull << 20)
#define MOST_RELATIVE 2e-2
#define MOST_RMS 2e-3
#define HALF_ONE 0x3C00u
#define HALF_SIGN 0x8000u
#define HALF_MANTISSA_MASK 0x03FFu

struct test_case {
	uint32_t rows, live, inner;
};
static const struct test_case cases[] = { { 8, 6, 512 },
					  { 48, 48, 512 },
					  { 64, 64, 640 },
					  { 8, 6, 640 } };

static uint64_t matrix_bytes(uint64_t input, uint64_t output)
{
	return input * output * BITS / 8u + (input + output) * 2u;
}

/* random trellis words; suh and svh FP16 of magnitude 1 to 2 and either sign */
static void matrix_fill(unsigned char *stored, uint64_t input, uint64_t output,
			uint64_t stream)
{
	const uint64_t trellis = input * output * BITS / 8u;
	for (uint64_t word = 0; word < trellis / 2u; word++) {
		const uint16_t bits = (uint16_t)test_linear_hash(stream, word);
		memcpy(stored + 2u * word, &bits, 2);
	}
	for (uint64_t value = 0; value < input + output; value++) {
		const uint64_t hash = test_linear_hash(stream ^ 0x5u, value);
		const uint16_t half =
			(uint16_t)(HALF_ONE | (hash & HALF_MANTISSA_MASK) |
				   ((hash >> 16) & 1u ? HALF_SIGN : 0u));
		memcpy(stored + trellis + 2u * value, &half, 2);
	}
}

/* the router's table: each live row picks 6 distinct experts (slot = the expert's place among its six, ascending) */
static uint32_t jobs_fill(struct expert_job_record *jobs, uint32_t capacity,
			  uint32_t live, const uint64_t *offsets,
			  uint32_t inner, uint64_t stream)
{
	uint32_t picks[64][PAIRS];
	float weights[64][PAIRS];
	for (uint32_t row = 0; row < live; row++) {
		uint32_t count = 0;
		for (uint64_t draw = 0; count < PAIRS; draw++) {
			const uint32_t expert =
				(uint32_t)(test_linear_hash(
						   stream, row * 1000u + draw) %
					   EXPERTS);
			bool taken = false;
			for (uint32_t slot = 0; slot < count; slot++)
				taken = taken || picks[row][slot] == expert;
			if (!taken)
				picks[row][count++] = expert;
		}
		for (uint32_t a = 0; a < PAIRS; a++) /* ascending */
			for (uint32_t b = a + 1u; b < PAIRS; b++)
				if (picks[row][b] < picks[row][a]) {
					const uint32_t swap = picks[row][a];
					picks[row][a] = picks[row][b];
					picks[row][b] = swap;
				}
		for (uint32_t slot = 0; slot < PAIRS; slot++)
			weights[row][slot] =
				0.05f +
				(float)(test_linear_hash(stream ^ 0x9u,
							 row * PAIRS + slot) %
					1000u) /
					2500.0f;
	}
	memset(jobs, 0, sizeof(*jobs) * capacity);
	uint32_t job = 0;
	const uint64_t gate_up_bytes = matrix_bytes(HIDDEN, inner);
	for (uint32_t expert = 0; expert < EXPERTS; expert++) {
		for (uint32_t row = 0; row < live; row++) {
			for (uint32_t slot = 0; slot < PAIRS; slot++) {
				if (picks[row][slot] != expert)
					continue;
				if (job < capacity &&
				    jobs[job].row_count == JOB_ROWS)
					job++;
				if (job >= capacity)
					return job;
				struct expert_job_record *const at = &jobs[job];
				at->gate_offset = offsets[expert];
				at->up_offset = offsets[expert] + gate_up_bytes;
				at->down_offset =
					offsets[expert] + 2u * gate_up_bytes;
				at->pair_index[at->row_count] =
					row * PAIRS + slot;
				at->route_weight[at->row_count] =
					weights[row][slot];
				at->row_count++;
			}
		}
		if (job < capacity && jobs[job].row_count != 0)
			job++;
	}
	return job;
}

static bool case_run(struct test_linear_card *card,
		     const struct test_case *test, uint64_t stream)
{
	test_linear_context("rows %u (live %u), intermediate %u", test->rows,
			    test->live, test->inner);
	test_linear_card_clear(card);
	const uint32_t pairs = test->rows * PAIRS,
		       capacity = test->rows * PAIRS;
	const uint64_t expert_bytes = 2u * matrix_bytes(HIDDEN, test->inner) +
				      matrix_bytes(test->inner, HIDDEN);
	uint64_t offsets[EXPERTS];
	for (uint32_t expert = 0; expert < EXPERTS; expert++) {
		const struct linear_range range =
			test_linear_card_weights(card, expert_bytes);
		unsigned char *const stored =
			test_linear_card_writable(card, range);
		if (stored == NULL)
			return test_linear_expect(false,
						  "room for the experts");
		offsets[expert] = range.byte_offset;
		matrix_fill(stored, HIDDEN, test->inner,
			    stream * 131u + 3u * expert);
		matrix_fill(stored + matrix_bytes(HIDDEN, test->inner), HIDDEN,
			    test->inner, stream * 131u + 3u * expert + 1u);
		matrix_fill(stored + 2u * matrix_bytes(HIDDEN, test->inner),
			    test->inner, HIDDEN,
			    stream * 131u + 3u * expert + 2u);
	}
	const struct linear_range hidden =
		test_linear_card_work(card, (uint64_t)test->rows * HIDDEN * 2u);
	const struct linear_range jobs = test_linear_card_work(
		card, sizeof(struct expert_job_record) * capacity);
	const uint64_t scratch_bytes = exl3_scratch_byte_count(pairs, HIDDEN);
	const struct linear_range scratch =
		test_linear_card_work(card, scratch_bytes);
	const struct linear_range pair_hidden =
		test_linear_card_work(card, (uint64_t)pairs * test->inner * 4u);
	const struct linear_range reference =
		test_linear_card_work(card, (uint64_t)test->rows * HIDDEN * 4u);
	const uint64_t workspace_bytes =
		exl3_tc_workspace_bytes(pairs, test->inner);
	const struct linear_range workspace =
		test_linear_card_work(card, workspace_bytes);
	const struct linear_range candidate =
		test_linear_card_work(card, (uint64_t)test->rows * HIDDEN * 4u);
	if (!test_linear_card_has_room(card))
		return false;
	uint16_t *const rows = malloc((size_t)test->rows * HIDDEN * 2u);
	struct expert_job_record *const table =
		malloc(sizeof(*table) * capacity);
	for (uint32_t row = 0; row < test->rows; row++)
		test_linear_values(rows + (size_t)row * HIDDEN, HIDDEN,
				   stream * 977u + row,
				   TEST_LINEAR_VALUES_PLAIN);
	const uint32_t job_count = jobs_fill(table, capacity, test->live,
					     offsets, test->inner, stream);
	bool passed =
		test_linear_expect(job_count < capacity,
				   "the table holds the jobs") &&
		test_linear_card_put(card, hidden, 0, rows,
				     (uint64_t)test->rows * HIDDEN * 2u) &&
		test_linear_card_put(card, jobs, 0, table,
				     sizeof(*table) * capacity) &&
		test_linear_card_load(card);
	/* the engine's EXL3 routed experts: five passes, then the row sums */
	const uint64_t products =
		scratch.byte_offset +
		(uint64_t)pairs * HIDDEN * EXL3_PROJECTIONS * EXL3_HALF_BYTES;
	const struct exl3_expert_record whole = {
		.jobs_offset = jobs.byte_offset,
		.input_offset = hidden.byte_offset,
		.operands_offset = scratch.byte_offset,
		.products_offset = products,
		.hidden_offset = pair_hidden.byte_offset,
		.output_offset = products,
		.input_size = HIDDEN,
		.inner_size = test->inner,
		.pair_count = pairs,
		.pairs_per_row = PAIRS,
		.job_count = capacity,
		.bits_per_weight = BITS,
		.activation_limit = ACTIVATION_LIMIT
	};
	struct kernel_launch launches[EXL3_EXPERT_LAUNCHES + 1u];
	passed = passed &&
		 test_linear_expect_status(
			 exl3_experts_fast_prepare(
				 &whole, card->arena_byte_count,
				 &backend_cuda_value, card->card, launches),
			 STATUS_OK, "the EXL3 experts prepare");
	const struct expert_row_sum_arguments sum = { products,
						      reference.byte_offset,
						      HIDDEN, PAIRS };
	passed = passed &&
		 test_linear_expect_status(
			 bb_exl3_row_sum_prepare(
				 &sum, test->rows, card->arena_byte_count,
				 &backend_cuda_value, card->card,
				 &launches[EXL3_EXPERT_LAUNCHES]),
			 STATUS_OK, "the row sum prepare");
	passed = passed && test_linear_card_run(card, launches,
						EXL3_EXPERT_LAUNCHES + 1u);
	/* the tensor-core MoE */
	const struct exl3_tc_record tc = {
		.jobs_offset = jobs.byte_offset,
		.hidden_offset = hidden.byte_offset,
		.workspace_offset = workspace.byte_offset,
		.output_offset = candidate.byte_offset,
		.row_count = test->rows,
		.job_capacity = capacity,
		.intermediate = test->inner,
		.bits_x2 = 2u * BITS,
		.activation_limit = ACTIVATION_LIMIT
	};
	struct kernel_launch tc_launches[EXL3_TC_LAUNCHES];
	passed = passed && test_linear_expect_status(
				   exl3_tc_prepare(&tc, card->arena_byte_count,
						   &backend_cuda_value,
						   card->card, tc_launches),
				   STATUS_OK, "the tensor-core MoE prepare");
	passed = passed &&
		 test_linear_card_run(card, tc_launches, EXL3_TC_LAUNCHES) &&
		 test_linear_card_fetch(card);
	float *const expected = malloc((size_t)test->rows * HIDDEN * 4u),
		     *const actual = malloc((size_t)test->rows * HIDDEN * 4u);
	passed = passed &&
		 test_linear_card_get(card, reference, 0, expected,
				      (uint64_t)test->live * HIDDEN * 4u) &&
		 test_linear_card_get(card, candidate, 0, actual,
				      (uint64_t)test->live * HIDDEN * 4u);
	if (passed) {
		double largest = 0.0, difference = 0.0, squares = 0.0,
		       magnitudes = 0.0;
		uint64_t worst = 0;
		for (uint64_t value = 0; value < (uint64_t)test->live * HIDDEN;
		     value++) {
			const double gap = fabs((double)actual[value] -
						(double)expected[value]);
			largest = fmax(largest, fabs((double)expected[value]));
			if (gap > difference || isnan(gap)) {
				difference = gap;
				worst = value;
			}
			squares += gap * gap;
			magnitudes += (double)expected[value] *
				      (double)expected[value];
		}
		const double relative = largest > 0.0 ? difference / largest :
							difference;
		const double rms = magnitudes > 0.0 ?
					   sqrt(squares / magnitudes) :
					   sqrt(squares);
		(void)printf(
			"rows %u live %u intermediate %u jobs %u: largest relative difference %.3e (row %" PRIu64
			" output %" PRIu64
			": %.6g vs %.6g), rms relative %.3e\n",
			test->rows, test->live, test->inner, job_count,
			relative, worst / HIDDEN, worst % HIDDEN,
			(double)actual[worst], (double)expected[worst], rms);
		passed = test_linear_expect(relative <= MOST_RELATIVE &&
						    rms <= MOST_RMS,
					    "the row sums agree");
	}
	free(rows);
	free(table);
	free(expected);
	free(actual);
	return passed;
}

int main(void)
{
	struct test_linear_card card;
	if (!test_linear_card_open(&card, 0, WORK_CAPACITY, WEIGHTS_CAPACITY))
		return EXIT_FAILURE;
	for (uint32_t index = 0; index < sizeof(cases) / sizeof(cases[0]);
	     index++)
		(void)case_run(&card, &cases[index], 11u + index);
	test_linear_card_close(&card);
	(void)printf("exl3_tc_card_test: %d failures\n",
		     test_linear_failures());
	return test_linear_failures() == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
