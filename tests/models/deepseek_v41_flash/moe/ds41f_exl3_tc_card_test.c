/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * ds41f_exl3_tc_card_test.c: the tensor-core EXL3 MoE (exl3_tc.h, --exl3_tc_moe) against the engine's EXL3 routed
 * experts (exl3_experts_fast_prepare and the row sum) on one card, from the same job table (ds41f_router_records.h: the
 * routed experts ascending, a run of an expert's pairs cut in jobs of at most 8, then empty jobs), the same synthetic
 * EXL3 matrices (random mul1 trellis words, suh and svh of both signs) and the same bfloat16 rows. The two orders of
 * arithmetic differ (GNmac's FP16 hi/lo split against the engine's), so the FP32 row sums are compared with a
 * tolerance: the largest difference relative to the largest magnitude of a case at most MOST_RELATIVE, and the
 * root-mean-square relative difference at most MOST_RMS. Cases: rows 8 with 6 live (a thin K5 verify), 48 and 64,
 * intermediate 512 and 640, at each width this build decodes (K2, K2.5, K3, K4; the tensor-core MoE refuses K4,
 * unsupported, and a K4 case checks the engine's products and the reconstruction). Per case, the reconstruction
 * kernel's FP16 weights of expert 0's gate matrix (its first RECONSTRUCT_ROWS inputs, every output) equal the host
 * oracle's (exl3_reference.h) bit for bit. It needs the cards free (gpu_run).
 */
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "engine/backend/backend_seam.h"
#include "engine/library/exl3/exl3.h"
#include "engine/library/exl3/exl3_tc.h"
#include "engine/library/exl3/exl3_tc_records.h"
#include "engine/library/exl3/exl3_tp.h"
#include "engine/library/block_scaled_linear/linear_records.h"
#include "engine/library/exl3/exl3_reference.h"
#include "engine/library/block_scaled_linear/linear_card_fixture.h"

#define HIDDEN 5120u
#define EXPERTS 48u
#define PAIRS DS41F_EXL3_TC_PAIRS_PER_ROW
#define JOB_ROWS LINEAR_MAXIMUM_ROWS
#define RECONSTRUCT_ROWS 32u
#define ACTIVATION_LIMIT 10.0f
#define WORK_CAPACITY (512ull << 20)
#define WEIGHTS_CAPACITY (768ull << 20)
#define MOST_RELATIVE 2e-2
#define MOST_RMS 2e-3
#define HALF_ONE 0x3C00u
#define HALF_SIGN 0x8000u
#define HALF_MANTISSA_MASK 0x03FFu
#define HALF_BYTES 2u /* an FP16 suh or svh value */
#define WORD_BYTES 2u /* a trellis word */
#define BF16_BYTES 2u /* an input row's value */
#define FLOAT_BYTES 4u
#define SIGN_SHIFT 16 /* the hash bit that signs a scale */
/* an expert's matrices: gate, up, then down, each with its own stream */
#define EXPERT_MATRICES 3u
#define DOWN_MATRIX 2u
#define MOST_LIVE_ROWS 64
/* the hash streams: salts and strides that keep the scales, weights, picks, matrices and rows apart */
#define SCALES_SALT 0x5u
#define WEIGHTS_SALT 0x9u
#define DRAWS_A_ROW 1000u
#define MATRICES_STRIDE 131u
#define ROWS_STRIDE 977u
#define FIRST_STREAM 11u
/* a pair's routing weight: WEIGHT_LEAST plus up to WEIGHT_STEPS steps of 1 / WEIGHT_STEP_DIVISOR */
#define WEIGHT_LEAST 0.05f
#define WEIGHT_STEPS 1000u
#define WEIGHT_STEP_DIVISOR 2500.0f

struct test_case {
	uint32_t rows, live, inner,
		width; /* width: the stored width (exl3_records.h) */
};
static const struct test_case cases[] = {
	{ 8, 6, 512, EXL3_K3 },	    { 48, 48, 512, EXL3_K3 },
	{ 64, 64, 640, EXL3_K3 },   { 8, 6, 640, EXL3_K3 },
	{ 8, 6, 640, EXL3_K2 },	    { 64, 64, 512, EXL3_K2 },
	{ 8, 6, 512, EXL3_K2_5 },   { 48, 48, 512, EXL3_K2_5 },
	{ 64, 64, 640, EXL3_K2_5 }, { 8, 6, 640, EXL3_K4 },
	{ 64, 64, 512, EXL3_K4 }
};

static uint64_t matrix_bytes(uint64_t input, uint64_t output, uint32_t width)
{
	return exl3_code_bytes(input, output, exl3_bits_x2(width)) +
	       (input + output) * HALF_BYTES;
}

/* random trellis words; suh and svh FP16 of magnitude 1 to 2 and either sign */
static void matrix_fill(unsigned char *stored, uint64_t input, uint64_t output,
			uint32_t width, uint64_t stream)
{
	const uint64_t trellis =
		exl3_code_bytes(input, output, exl3_bits_x2(width));
	for (uint64_t word = 0; word < trellis / WORD_BYTES; word++) {
		const uint16_t bits = (uint16_t)test_linear_hash(stream, word);
		memcpy(stored + WORD_BYTES * word, &bits, WORD_BYTES);
	}
	for (uint64_t value = 0; value < input + output; value++) {
		const uint64_t hash =
			test_linear_hash(stream ^ SCALES_SALT, value);
		const uint16_t half =
			(uint16_t)(HALF_ONE | (hash & HALF_MANTISSA_MASK) |
				   ((hash >> SIGN_SHIFT) & 1u ? HALF_SIGN :
								0u));
		memcpy(stored + trellis + HALF_BYTES * value, &half,
		       HALF_BYTES);
	}
}

/* the router's table: each live row picks 6 distinct experts (slot = the expert's place among its six, ascending) */
static uint32_t jobs_fill(struct expert_job_record *jobs, uint32_t capacity,
			  uint32_t live, const uint64_t *offsets,
			  uint32_t inner, uint32_t width, uint64_t stream)
{
	uint32_t picks[MOST_LIVE_ROWS][PAIRS];
	float weights[MOST_LIVE_ROWS][PAIRS];
	for (uint32_t row = 0; row < live; row++) {
		uint32_t count = 0;
		for (uint64_t draw = 0; count < PAIRS; draw++) {
			const uint32_t expert =
				(uint32_t)(test_linear_hash(
						   stream,
						   (uint64_t)(row *
							      DRAWS_A_ROW) +
							   draw) %
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
				WEIGHT_LEAST +
				(float)(test_linear_hash(stream ^ WEIGHTS_SALT,
							 row * PAIRS + slot) %
					WEIGHT_STEPS) /
					WEIGHT_STEP_DIVISOR;
	}
	memset(jobs, 0, sizeof(*jobs) * capacity);
	uint32_t job = 0;
	const uint64_t gate_up_bytes = matrix_bytes(HIDDEN, inner, width);
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
				at->down_offset = offsets[expert] +
						  DOWN_MATRIX * gate_up_bytes;
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
	test_linear_context(
		"rows %u (live %u), intermediate %u, stored width %u",
		test->rows, test->live, test->inner, test->width);
	test_linear_card_clear(card);
	const uint32_t pairs = test->rows * PAIRS,
		       capacity = test->rows * PAIRS;
	const uint64_t gate_bytes =
		matrix_bytes(HIDDEN, test->inner, test->width);
	const uint64_t expert_bytes =
		DOWN_MATRIX * gate_bytes +
		matrix_bytes(test->inner, HIDDEN, test->width);
	uint64_t offsets[EXPERTS];
	struct linear_range first_expert = { 0, 0 };
	for (uint32_t expert = 0; expert < EXPERTS; expert++) {
		const struct linear_range range =
			test_linear_card_weights(card, expert_bytes);
		unsigned char *const stored =
			test_linear_card_writable(card, range);
		if (stored == NULL)
			return test_linear_expect(false,
						  "room for the experts");
		offsets[expert] = range.byte_offset;
		first_expert = expert == 0 ? range : first_expert;
		matrix_fill(stored, HIDDEN, test->inner, test->width,
			    stream * MATRICES_STRIDE +
				    (uint64_t)(EXPERT_MATRICES * expert));
		matrix_fill(stored + gate_bytes, HIDDEN, test->inner,
			    test->width,
			    stream * MATRICES_STRIDE +
				    (uint64_t)(EXPERT_MATRICES * expert) + 1u);
		matrix_fill(stored + DOWN_MATRIX * gate_bytes, test->inner,
			    HIDDEN, test->width,
			    stream * MATRICES_STRIDE +
				    (uint64_t)(EXPERT_MATRICES * expert) +
				    DOWN_MATRIX);
	}
	const struct linear_range hidden = test_linear_card_work(
		card, (uint64_t)test->rows * HIDDEN * BF16_BYTES);
	const struct linear_range jobs = test_linear_card_work(
		card, sizeof(struct expert_job_record) * capacity);
	const uint64_t scratch_bytes = exl3_scratch_byte_count(pairs, HIDDEN);
	const struct linear_range scratch =
		test_linear_card_work(card, scratch_bytes);
	const struct linear_range pair_hidden = test_linear_card_work(
		card, (uint64_t)pairs * test->inner * FLOAT_BYTES);
	const struct linear_range reference = test_linear_card_work(
		card, (uint64_t)test->rows * HIDDEN * FLOAT_BYTES);
	const uint64_t workspace_bytes =
		exl3_tc_workspace_bytes(pairs, test->inner);
	const struct linear_range workspace =
		test_linear_card_work(card, workspace_bytes);
	const struct linear_range candidate = test_linear_card_work(
		card, (uint64_t)test->rows * HIDDEN * FLOAT_BYTES);
	const struct linear_range decoded = test_linear_card_work(
		card, (uint64_t)HIDDEN * test->inner * EXL3_HALF_BYTES);
	if (!test_linear_card_has_room(card))
		return false;
	uint16_t *const rows = malloc((size_t)test->rows * HIDDEN * BF16_BYTES);
	struct expert_job_record *const table =
		malloc(sizeof(*table) * capacity);
	if (rows == NULL || table == NULL) {
		free(table);
		free(rows);
		return false;
	}
	for (uint32_t row = 0; row < test->rows; row++)
		test_linear_values(rows + (size_t)row * HIDDEN, HIDDEN,
				   stream * ROWS_STRIDE + row,
				   TEST_LINEAR_VALUES_PLAIN);
	const uint32_t job_count = jobs_fill(table, capacity, test->live,
					     offsets, test->inner, test->width,
					     stream);
	bool passed = test_linear_expect(job_count < capacity,
					 "the table holds the jobs") &&
		      test_linear_card_put(card, hidden, 0, rows,
					   (uint64_t)test->rows * HIDDEN *
						   BF16_BYTES) &&
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
		.bits_per_weight = test->width,
		.activation_limit = ACTIVATION_LIMIT
	};
	struct kernel_launch launches[EXL3_EXPERT_LAUNCHES + 1u];
	passed = passed && test_linear_expect_status(
				   exl3_experts_fast_prepare(
					   &whole, card->arena_byte_count,
					   &backend_cuda_value, card->card,
					   card->arithmetic_class, launches),
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
		.bits_x2 = exl3_bits_x2(test->width),
		.activation_limit = ACTIVATION_LIMIT
	};
	/* the tensor-core MoE takes no K4 (EXL3_TC_SUPPORTED_BITS_X2): it refuses, unsupported, and the case checks the
	 * engine's products and the reconstruction alone */
	const bool tensor_cores = exl3_tc_width_supported(tc.bits_x2);
	struct kernel_launch tc_launches[EXL3_TC_LAUNCHES];
	passed = passed &&
		 test_linear_expect_status(
			 exl3_tc_prepare(&tc, card->arena_byte_count,
					 &backend_cuda_value, card->card,
					 tc_launches),
			 tensor_cores ? STATUS_OK : STATUS_UNSUPPORTED,
			 "the tensor-core MoE prepare");
	/* the reconstruction of expert 0's gate matrix, FP16 [input][output] */
	const struct exl3_reconstruct_record reconstruct = {
		.matrix_offset = offsets[0],
		.output_offset = decoded.byte_offset,
		.input_size = HIDDEN,
		.output_size = test->inner,
		.bits_per_weight = test->width
	};
	uint32_t reconstruct_kernel = BACKEND_NO_KERNEL;
	struct kernel_launch reconstruct_launch;
	passed = passed &&
		 test_linear_expect_status(
			 backend_cuda_value.kernel_prepare(
				 card->card, EXL3_KERNEL_FILE, EXL3_RECONSTRUCT,
				 sizeof(reconstruct), &reconstruct_kernel),
			 STATUS_OK, "the reconstruction kernel") &&
		 test_linear_expect_status(
			 exl3_reconstruct_prepare(
				 &reconstruct, card->arena_byte_count,
				 reconstruct_kernel, &reconstruct_launch),
			 STATUS_OK, "the reconstruction prepares");
	passed = passed &&
		 (!tensor_cores ||
		  test_linear_card_run(card, tc_launches, EXL3_TC_LAUNCHES)) &&
		 test_linear_card_run(card, &reconstruct_launch, 1u) &&
		 test_linear_card_fetch(card);
	uint16_t *const decoded_rows = malloc((size_t)RECONSTRUCT_ROWS *
					      test->inner * EXL3_HALF_BYTES);
	passed = passed && decoded_rows != NULL &&
		 test_linear_card_get(card, decoded, 0, decoded_rows,
				      (uint64_t)RECONSTRUCT_ROWS * test->inner *
					      EXL3_HALF_BYTES);
	if (passed) {
		uint64_t mismatches = 0;
		for (uint32_t input = 0; input < RECONSTRUCT_ROWS; input++)
			for (uint32_t output = 0; output < test->inner;
			     output++)
				mismatches +=
					decoded_rows[(size_t)input * test->inner +
						     output] !=
					test_exl3_reconstruct(
						test_linear_card_stored(
							card, first_expert),
						HIDDEN, test->inner,
						test->width, input, output);
		passed = test_linear_expect(
			mismatches == 0,
			"the reconstruction is the oracle's, bit for bit");
	}
	free(decoded_rows);
	float *const expected =
			     malloc((size_t)test->rows * HIDDEN * FLOAT_BYTES),
		     *const actual =
			     malloc((size_t)test->rows * HIDDEN * FLOAT_BYTES);
	passed = passed &&
		 test_linear_card_get(card, reference, 0, expected,
				      (uint64_t)test->live * HIDDEN *
					      FLOAT_BYTES) &&
		 test_linear_card_get(card, candidate, 0, actual,
				      (uint64_t)test->live * HIDDEN *
					      FLOAT_BYTES);
	if (passed && tensor_cores) {
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
		(void)case_run(&card, &cases[index], FIRST_STREAM + index);
	test_linear_card_close(&card);
	(void)printf("exl3_tc_card_test: %d failures\n",
		     test_linear_failures());
	return test_linear_failures() == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
