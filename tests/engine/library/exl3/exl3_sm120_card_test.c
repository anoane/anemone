/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * exl3_sm120_card_test.c: the EXL3 kernels on every card the backend finds, in every build the card's class takes
 * (kernel_exl3_mma.cuh): a card of class 80 the sm_80 modules (and the relaxed _s8 products); a card of class 120 the
 * plain sm_120a modules (class 80's sums emulated, ARITHMETIC_CLASS_SM_120) and the *_native ones
 * (ARITHMETIC_CLASS_SM_120_NATIVE). On each:
 *   - decode: exl3_reconstruct of every place of a matrix of each width this build decodes (K2, K2.5, K3, K4) against
 *     the bit-walking oracle (exl3_reference.h);
 *   - products, at each width: the gate/up and down products of the retained (kernel_exl3) and wide
 *     (kernel_exl3_wide; class 120's emulation of class 80 gets the retained ones for it, exl3.h) forms, from
 *     operands written in place, against the host's chain (exl3_chain_reference.h) of the build's class, bit for
 *     bit: one operand of one job row is an infinity, whose outputs must all be not finite (the host's model says
 *     not-a-number; the bits a card gives are printed) while every other row's, the same job's other columns
 *     included, keeps its bits; jobs of 8, 6 and 1 rows;
 *   - block, at K3: the five launches and the row sum from bfloat16 rows; class 80's sums emulated must give the
 *     bits of a card of class 80 that ran before or after it, class 120's own within MOST_RELATIVE / MOST_RMS of
 *     them.
 * Prints one line a case and the digest of every build's row sums. It needs the cards free (gpu_run).
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
#include "engine/library/exl3/exl3_records.h"
#include "engine/library/exl3/exl3_tp.h"
#include "engine/library/block_scaled_linear/linear_records.h"
#include "exl3_chain_reference.h"
#include "engine/library/block_scaled_linear/linear_card_fixture.h"

#define INPUT 256u /* the experts' input (hidden) width */
#define INNER 128u /* their intermediate width */
#define ROWS 4u
#define PAIRS_PER_ROW 6u
#define PAIRS \
	24u /* ROWS * PAIRS_PER_ROW, one literal so that no use widens a 32-bit product */
#define EXPERTS 3u
#define JOBS 5u
#define DECODE_SIDE 128u
#define RETAINED_SLICES 4u /* EXL3_PRODUCT_SLICES */
#define RELAXED_SLICES 8u /* EXL3_RELAXED_SLICES */
#define ACTIVATION_LIMIT 10.0f
#define WORK_CAPACITY (64ull << 20)
#define WEIGHTS_CAPACITY (16ull << 20)
#define HALF_BYTES 2u
#define FLOAT_BYTES 4u
#define WORD_BYTES 4u
#define HALF_ONE 0x3C00u
#define HALF_SIGN 0x8000u
#define HALF_MANTISSA_MASK 0x03FFu
#define HALF_INFINITY 0x7C00u
#define HALF_FIELD_SHIFT 10u
#define HALF_BIAS 15u
#define EXPONENT_SPREAD 6u
#define EXPONENT_CHOICES 13u /* 2 EXPONENT_SPREAD + 1 */
#define SIGN_SHIFT 16u
#define FIELD_SHIFT 20u
#define POISONED_PAIR 3u /* a column of the 8-row job */
#define POISONED_INPUT 5u
#define ROUTE_WEIGHT_BASE 0.05f
#define ROUTE_WEIGHT_STEPS 1000u
#define ROUTE_WEIGHT_SCALE 2500.0f
#define MOST_RELATIVE 1e-2
#define MOST_RMS 1e-3
#define MOST_SHOWN 4u
#define STREAM_PRIME 131u
#define DECODE_STREAM 7u
#define OPERAND_STREAM 11u
#define ROWS_STREAM 13u
#define WEIGHT_STREAM 17u
#define MOST_BUILDS 2u
#define GATE_UP_MATRICES 2u /* gate, then up: INPUT x INNER each, then down */
#define WHAT_BYTES 96u
/* exl3_experts_prepare's five launches: input, gate/up products, activation, down products, finish */
#define GATE_UP_LAUNCH 1u
#define DOWN_LAUNCH 3u

enum job_field { JOB_EXPERT, JOB_FIRST_PAIR, JOB_ROWS, JOB_FIELDS };
/* The jobs: expert, first pair, rows (8, 6 and 1 rows; columns 0 to 7, 0 to 5, 0) */
static const uint32_t job_shape[JOBS][JOB_FIELDS] = { { 0u, 0u, 8u },
						      { 1u, 8u, 6u },
						      { 2u, 14u, 1u },
						      { 0u, 15u, 8u },
						      { 1u, 23u, 1u } };
/* every stored width this build decodes (EXL3_SUPPORTED_BITS_X2) */
static const uint32_t widths[] = { EXL3_K2, EXL3_K2_5, EXL3_K3, EXL3_K4 };
_Static_assert(PAIRS == ROWS * PAIRS_PER_ROW, "PAIRS is ROWS * PAIRS_PER_ROW");

/* The row sums of the block per build (class 80's sums: emulated or not; class 120's own), the first card's of each. */
struct block_result {
	bool is_set;
	float sums[ROWS * INPUT];
};
static struct block_result class_80_block, class_120_block;
static bool emulated_seen;
static float emulated_sums[ROWS * INPUT];

static uint64_t matrix_bytes(uint32_t inputs, uint32_t outputs, uint32_t bits)
{
	return exl3_code_bytes(inputs, outputs, exl3_bits_x2(bits)) +
	       ((uint64_t)inputs + outputs) * HALF_BYTES;
}

/* a stored width printed as K<whole>[.5]: K2, K2.5, K3, K4 */
static uint32_t width_whole(uint32_t bits)
{
	return exl3_bits_x2(bits) / EXL3_HALF_BITS_PER_BIT;
}

static const char *width_half(uint32_t bits)
{
	return exl3_bits_x2(bits) % EXL3_HALF_BITS_PER_BIT != 0 ? ".5" : "";
}

/* random trellis words; suh and svh of magnitude 1 to 2 and either sign */
static void matrix_fill(unsigned char *stored, uint32_t inputs,
			uint32_t outputs, uint32_t bits, uint64_t stream)
{
	const uint64_t trellis =
		exl3_code_bytes(inputs, outputs, exl3_bits_x2(bits));
	for (uint64_t word = 0; word < trellis / WORD_BYTES; word++) {
		const uint32_t value = (uint32_t)test_linear_hash(stream, word);
		memcpy(stored + WORD_BYTES * word, &value, WORD_BYTES);
	}
	for (uint64_t at = 0; at < (uint64_t)inputs + outputs; at++) {
		const uint64_t hash =
			test_linear_hash(stream ^ WEIGHT_STREAM, at);
		const uint16_t half =
			(uint16_t)(HALF_ONE | (hash & HALF_MANTISSA_MASK) |
				   (((hash >> SIGN_SHIFT) & 1u) != 0 ?
					    HALF_SIGN :
					    0u));
		memcpy(stored + trellis + HALF_BYTES * at, &half, HALF_BYTES);
	}
}

/* a finite 16-bit operand within 2^-EXPONENT_SPREAD to 2^EXPONENT_SPREAD, either sign */
static uint16_t operand_value(uint64_t stream, uint64_t position)
{
	const uint64_t hash = test_linear_hash(stream, position);
	const uint32_t field =
		HALF_BIAS - EXPONENT_SPREAD +
		(uint32_t)((hash >> FIELD_SHIFT) % EXPONENT_CHOICES);
	return (uint16_t)((((hash >> SIGN_SHIFT) & 1u) != 0 ? HALF_SIGN : 0u) |
			  (field << HALF_FIELD_SHIFT) |
			  (hash & HALF_MANTISSA_MASK));
}

/* outputs [count] of the card against expected: bits equal; not finite where expected is a not-a-number (an input
 * that is not finite entered it), the first such output's bits printed */
static bool expect_floats(const char *what, const float *outputs,
			  const float *expected, uint64_t count)
{
	uint64_t wrong = 0, poisoned = 0;
	uint32_t first_poisoned_bits = 0;
	for (uint64_t at = 0; at < count; at++) {
		uint32_t output_bits = 0, expected_bits = 0;
		memcpy(&output_bits, &outputs[at], sizeof(output_bits));
		memcpy(&expected_bits, &expected[at], sizeof(expected_bits));
		const bool is_poisoned = isnan(expected[at]);
		first_poisoned_bits = is_poisoned && poisoned++ == 0 ?
					      output_bits :
					      first_poisoned_bits;
		const bool good = is_poisoned ? !isfinite(outputs[at]) :
						output_bits == expected_bits;
		if (!good && wrong++ < MOST_SHOWN)
			test_linear_fail("%s: value %" PRIu64
					 " is %08x, expected %08x",
					 what, at, output_bits, expected_bits);
	}
	(void)printf("%s: %" PRIu64 " values, %" PRIu64 " differ; %" PRIu64
		     " poisoned, the first %08x\n",
		     what, count, wrong, poisoned, first_poisoned_bits);
	return wrong == 0;
}

/* what, the name of a case: false when it does not fit */
static bool case_name(char what[WHAT_BYTES], const char *form,
		      const char *projections, uint32_t bits,
		      uint32_t module_class)
{
	const int length = snprintf(what, WHAT_BYTES, "K%u%s %s %s, class %u",
				    width_whole(bits), width_half(bits), form,
				    projections, module_class);
	return test_linear_expect(length > 0 && (uint32_t)length < WHAT_BYTES,
				  "the name of a case");
}

enum projection { GATE, UP, DOWN, PROJECTIONS };
struct experts {
	uint64_t offsets[EXPERTS]; /* gate, then up, then down */
	/* [inputs][outputs] of gate, up, down */
	uint16_t *decoded[EXPERTS][PROJECTIONS];
};

static bool experts_lay_out(struct test_linear_card *card,
			    struct experts *experts, uint32_t bits,
			    uint64_t stream)
{
	const uint64_t gate_up = matrix_bytes(INPUT, INNER, bits),
		       down = matrix_bytes(INNER, INPUT, bits);
	for (uint32_t expert = 0; expert < EXPERTS; expert++) {
		const struct linear_range range = test_linear_card_weights(
			card, GATE_UP_MATRICES * gate_up + down);
		unsigned char *const stored =
			test_linear_card_writable(card, range);
		if (stored == NULL)
			return test_linear_expect(false,
						  "room for the experts");
		experts->offsets[expert] = range.byte_offset;
		for (uint32_t projection = 0; projection < PROJECTIONS;
		     projection++) {
			/* gate, up: INPUT x INNER each; down behind them */
			const uint64_t at = projection * gate_up;
			const uint32_t inputs = projection == DOWN ? INNER :
								     INPUT;
			const uint32_t outputs = projection == DOWN ? INPUT :
								      INNER;
			matrix_fill(stored + at, inputs, outputs, bits,
				    stream * STREAM_PRIME +
					    (uint64_t)PROJECTIONS * expert +
					    projection);
			experts->decoded[expert][projection] =
				malloc((size_t)inputs * outputs * HALF_BYTES);
			if (experts->decoded[expert][projection] == NULL)
				return test_linear_expect(
					false,
					"host memory for the decoded weights");
			test_exl3_decode_matrix(
				stored + at, inputs, outputs, bits,
				experts->decoded[expert][projection]);
		}
	}
	return true;
}

static void experts_free(struct experts *experts)
{
	for (uint32_t expert = 0; expert < EXPERTS; expert++)
		for (uint32_t projection = 0; projection < PROJECTIONS;
		     projection++)
			free(experts->decoded[expert][projection]);
}

static void jobs_fill(struct expert_job_record *jobs,
		      const struct experts *experts, uint32_t bits)
{
	const uint64_t gate_up = matrix_bytes(INPUT, INNER, bits);
	memset(jobs, 0, sizeof(*jobs) * JOBS);
	for (uint32_t job = 0; job < JOBS; job++) {
		const uint64_t base =
			experts->offsets[job_shape[job][JOB_EXPERT]];
		jobs[job].gate_offset = base;
		jobs[job].up_offset = base + gate_up;
		jobs[job].down_offset = base + GATE_UP_MATRICES * gate_up;
		jobs[job].row_count = job_shape[job][JOB_ROWS];
		jobs[job].expert = job_shape[job][JOB_EXPERT];
		for (uint32_t row = 0; row < job_shape[job][JOB_ROWS]; row++) {
			jobs[job].pair_index[row] =
				job_shape[job][JOB_FIRST_PAIR] + row;
			jobs[job].route_weight[row] =
				ROUTE_WEIGHT_BASE +
				(float)(test_linear_hash(job, row) %
					ROUTE_WEIGHT_STEPS) /
					ROUTE_WEIGHT_SCALE;
		}
	}
}

static uint32_t expert_of_pair(uint32_t pair)
{
	for (uint32_t job = 0; job < JOBS; job++)
		if (pair >= job_shape[job][JOB_FIRST_PAIR] &&
		    pair < job_shape[job][JOB_FIRST_PAIR] +
				    job_shape[job][JOB_ROWS])
			return job_shape[job][JOB_EXPERT];
	return EXPERTS; /* not reached: every pair is in a job */
}

/* The host's products of one projection: [pair][outputs] (gate/up: [projection][pair][outputs]) */
static void expected_products(uint32_t sum_class, const struct experts *experts,
			      enum projection projection, uint32_t slices,
			      const uint16_t *operands, float *expected)
{
	const uint32_t inputs = projection == DOWN ? INNER : INPUT,
		       outputs = projection == DOWN ? INPUT : INNER;
	for (uint32_t pair = 0; pair < PAIRS; pair++)
		test_exl3_chain(
			sum_class,
			experts->decoded[expert_of_pair(pair)][projection],
			inputs, outputs, slices,
			operands + (uint64_t)pair * inputs, 1u,
			expected + (uint64_t)pair * outputs);
}

struct case_ranges {
	struct linear_range jobs, rows, scratch, hidden, row_sums;
};

static struct exl3_expert_record record_of(const struct case_ranges *ranges,
					   uint32_t bits)
{
	const uint64_t products =
		ranges->scratch.byte_offset +
		(uint64_t)PAIRS * INPUT * EXL3_PROJECTIONS * EXL3_HALF_BYTES;
	const struct exl3_expert_record record = {
		.jobs_offset = ranges->jobs.byte_offset,
		.input_offset = ranges->rows.byte_offset,
		.operands_offset = ranges->scratch.byte_offset,
		.products_offset = products,
		.hidden_offset = ranges->hidden.byte_offset,
		.output_offset = products,
		.input_size = INPUT,
		.inner_size = INNER,
		.pair_count = PAIRS,
		.pairs_per_row = PAIRS_PER_ROW,
		.job_count = JOBS,
		.bits_per_weight = bits,
		.activation_limit = ACTIVATION_LIMIT
	};
	return record;
}

typedef MUST_CHECK result_status (*experts_prepare_function)(
	const struct exl3_expert_record *, uint64_t,
	const struct backend_operations *, struct device_card *, uint32_t,
	struct kernel_launch *);

/* The products of one form: gate/up from operands in place, then down; each against the host's chain. */
static bool products_case(struct test_linear_card *card,
			  const struct case_ranges *ranges,
			  const struct experts *experts, uint32_t bits,
			  uint32_t module_class,
			  experts_prepare_function prepare, uint32_t slices,
			  const char *form)
{
	const struct exl3_expert_record record = record_of(ranges, bits);
	struct kernel_launch launches[EXL3_EXPERT_LAUNCHES];
	if (!test_linear_expect_status(prepare(&record, card->arena_byte_count,
					       &backend_cuda_value, card->card,
					       module_class, launches),
				       STATUS_OK, "the products prepare"))
		return false;
	const uint64_t gate_up_count =
		(uint64_t)EXL3_PROJECTIONS * PAIRS * INNER;
	uint16_t *const operands =
		malloc((size_t)EXL3_PROJECTIONS * PAIRS * INPUT * HALF_BYTES);
	float *const expected =
		malloc((size_t)PAIRS * INPUT * FLOAT_BYTES * EXL3_PROJECTIONS);
	float *const outputs =
		malloc((size_t)PAIRS * INPUT * FLOAT_BYTES * EXL3_PROJECTIONS);
	bool passed = operands != NULL && expected != NULL && outputs != NULL;
	const uint64_t products_at =
		record.products_offset - ranges->scratch.byte_offset;
	char what[WHAT_BYTES];
	if (passed) {
		for (uint64_t at = 0;
		     at < (uint64_t)EXL3_PROJECTIONS * PAIRS * INPUT; at++)
			operands[at] = operand_value(OPERAND_STREAM + bits, at);
		/* gate's operand of one job row; down then reads it as pair 6's (POISONED_PAIR * INPUT / INNER) */
		operands[(uint64_t)POISONED_PAIR * INPUT + POISONED_INPUT] =
			HALF_INFINITY;
		for (uint32_t projection = GATE; projection <= UP; projection++)
			expected_products(module_class, experts,
					  (enum projection)projection, slices,
					  operands + (uint64_t)projection *
							     PAIRS * INPUT,
					  expected + (uint64_t)projection *
							     PAIRS * INNER);
		passed = case_name(what, form, "gate/up", bits, module_class) &&
			 test_linear_card_put(
				 card, ranges->scratch, 0, operands,
				 (uint64_t)EXL3_PROJECTIONS * PAIRS * INPUT *
					 HALF_BYTES) &&
			 test_linear_card_upload(card, ranges->scratch, 0,
						 (uint64_t)EXL3_PROJECTIONS *
							 PAIRS * INPUT *
							 HALF_BYTES) &&
			 test_linear_card_run(card, &launches[GATE_UP_LAUNCH],
					      1u) &&
			 test_linear_card_fetch(card) &&
			 test_linear_card_get(card, ranges->scratch,
					      products_at, outputs,
					      gate_up_count * FLOAT_BYTES) &&
			 expect_floats(what, outputs, expected,
				       gate_up_count); /* gate, then up */
	}
	if (passed) { /* down: its own operands [pair][INNER] where gate/up's were */
		expected_products(module_class, experts, DOWN, slices, operands,
				  expected);
		passed = case_name(what, form, "down", bits, module_class) &&
			 test_linear_card_put(
				 card, ranges->scratch, 0, operands,
				 (uint64_t)PAIRS * INNER * HALF_BYTES) &&
			 test_linear_card_upload(card, ranges->scratch, 0,
						 (uint64_t)PAIRS * INNER *
							 HALF_BYTES) &&
			 test_linear_card_run(card, &launches[DOWN_LAUNCH],
					      1u) &&
			 test_linear_card_fetch(card) &&
			 test_linear_card_get(
				 card, ranges->scratch, products_at, outputs,
				 (uint64_t)PAIRS * INPUT * FLOAT_BYTES) &&
			 expect_floats(what, outputs, expected,
				       (uint64_t)PAIRS * INPUT);
	}
	free(operands);
	free(expected);
	free(outputs);
	return passed;
}

/* exl3_reconstruct of the build of module_class against the oracle, every place of a matrix of each width. */
static bool decode_case(struct test_linear_card *card, uint32_t module_class)
{
	bool passed = true;
	for (uint32_t index = 0;
	     index < sizeof(widths) / sizeof(widths[0]) && passed; index++) {
		test_linear_card_clear(card);
		const uint32_t bits = widths[index];
		const struct linear_range matrix = test_linear_card_weights(
			card, matrix_bytes(DECODE_SIDE, DECODE_SIDE, bits));
		const struct linear_range output = test_linear_card_work(
			card, (uint64_t)DECODE_SIDE * DECODE_SIDE * HALF_BYTES);
		unsigned char *const stored =
			test_linear_card_writable(card, matrix);
		uint16_t *const expected =
			malloc((size_t)DECODE_SIDE * DECODE_SIDE * HALF_BYTES);
		uint16_t *const decoded =
			malloc((size_t)DECODE_SIDE * DECODE_SIDE * HALF_BYTES);
		uint32_t kernel = BACKEND_NO_KERNEL;
		struct kernel_launch launch;
		passed = stored != NULL && expected != NULL &&
			 decoded != NULL && test_linear_card_has_room(card);
		if (passed) {
			matrix_fill(stored, DECODE_SIDE, DECODE_SIDE, bits,
				    DECODE_STREAM + bits);
			test_exl3_decode_matrix(stored, DECODE_SIDE,
						DECODE_SIDE, bits, expected);
			const struct exl3_reconstruct_record record = {
				matrix.byte_offset,
				output.byte_offset,
				DECODE_SIDE,
				DECODE_SIDE,
				bits,
				0
			};
			passed =
				test_linear_expect_status(
					backend_cuda_value.kernel_prepare(
						card->card,
						module_class == ARITHMETIC_CLASS_SM_120_NATIVE ?
							EXL3_KERNEL_FILE_NATIVE :
							EXL3_KERNEL_FILE,
						EXL3_RECONSTRUCT,
						sizeof(record), &kernel),
					STATUS_OK,
					"the reconstruction kernel") &&
				test_linear_expect_status(
					exl3_reconstruct_prepare(
						&record, card->arena_byte_count,
						kernel, &launch),
					STATUS_OK,
					"the reconstruction prepare") &&
				test_linear_card_load(card) &&
				test_linear_card_run(card, &launch, 1u) &&
				test_linear_card_fetch(card) &&
				test_linear_card_get(card, output, 0, decoded,
						     (uint64_t)DECODE_SIDE *
							     DECODE_SIDE *
							     HALF_BYTES);
			const bool same =
				passed &&
				memcmp(decoded, expected,
				       (size_t)DECODE_SIDE * DECODE_SIDE *
					       HALF_BYTES) == 0;
			(void)printf("decode K%u%s, class %u: %s\n",
				     width_whole(bits), width_half(bits),
				     module_class,
				     same ? "every place" : "DIFFERS");
			passed = passed && test_linear_expect(
						   same, "the decoded weights");
		}
		free(expected);
		free(decoded);
	}
	return passed;
}

/* The five launches and the row sum from bfloat16 rows; the row sums kept for the comparison across cards. */
static bool block_case(struct test_linear_card *card,
		       const struct case_ranges *ranges, uint32_t bits,
		       uint32_t module_class, float *sums)
{
	const struct exl3_expert_record record = record_of(ranges, bits);
	struct kernel_launch launches[EXL3_EXPERT_LAUNCHES + 1u];
	const struct expert_row_sum_arguments sum = {
		record.products_offset, ranges->row_sums.byte_offset, INPUT,
		PAIRS_PER_ROW
	};
	uint16_t rows[ROWS * INPUT];
	for (uint32_t row = 0; row < ROWS; row++)
		test_linear_values(rows + (size_t)row * INPUT, INPUT,
				   ROWS_STREAM + row, TEST_LINEAR_VALUES_PLAIN);
	return test_linear_expect_status(
		       exl3_experts_prepare(&record, card->arena_byte_count,
					    &backend_cuda_value, card->card,
					    module_class, launches),
		       STATUS_OK, "the block prepare") &&
	       test_linear_expect_status(
		       bb_exl3_row_sum_prepare(&sum, ROWS,
					       card->arena_byte_count,
					       &backend_cuda_value, card->card,
					       &launches[EXL3_EXPERT_LAUNCHES]),
		       STATUS_OK, "the row sum prepare") &&
	       test_linear_card_put(card, ranges->rows, 0, rows,
				    sizeof(rows)) &&
	       test_linear_card_upload(card, ranges->rows, 0, sizeof(rows)) &&
	       test_linear_card_run(card, launches,
				    EXL3_EXPERT_LAUNCHES + 1u) &&
	       test_linear_card_fetch(card) &&
	       test_linear_card_get(card, ranges->row_sums, 0, sums,
				    (uint64_t)ROWS * INPUT * FLOAT_BYTES);
}

static uint64_t sums_digest(const float *sums)
{
	uint64_t digest = 0;
	for (uint32_t at = 0; at < ROWS * INPUT; at++) {
		uint32_t bits = 0;
		memcpy(&bits, &sums[at], sizeof(bits));
		digest = test_linear_hash(digest ^ bits, at);
	}
	return digest;
}

/* Every case of one build of a card. */
static bool build_cases(struct test_linear_card *card, uint32_t module_class)
{
	bool passed = decode_case(card, module_class);
	for (uint32_t index = 0; index < sizeof(widths) / sizeof(widths[0]);
	     index++) {
		const uint32_t bits = widths[index];
		test_linear_context("K%u%s, class %u", width_whole(bits),
				    width_half(bits), module_class);
		test_linear_card_clear(card);
		struct experts experts;
		memset(&experts, 0, sizeof(experts));
		struct case_ranges ranges;
		ranges.jobs = test_linear_card_work(
			card, sizeof(struct expert_job_record) * JOBS);
		ranges.rows = test_linear_card_work(
			card, (uint64_t)ROWS * INPUT * HALF_BYTES);
		ranges.scratch = test_linear_card_work(
			card, exl3_scratch_byte_count(PAIRS, INPUT));
		ranges.hidden = test_linear_card_work(
			card, (uint64_t)PAIRS * INNER * HALF_BYTES);
		ranges.row_sums = test_linear_card_work(
			card, (uint64_t)ROWS * INPUT * FLOAT_BYTES);
		struct expert_job_record jobs[JOBS];
		bool laid_out = test_linear_card_has_room(card) &&
				experts_lay_out(card, &experts, bits, bits);
		if (laid_out) {
			jobs_fill(jobs, &experts, bits);
			laid_out = test_linear_card_put(card, ranges.jobs, 0,
							jobs, sizeof(jobs)) &&
				   test_linear_card_load(card);
		}
		passed = passed && laid_out;
		if (laid_out) {
			passed = products_case(card, &ranges, &experts, bits,
					       module_class,
					       exl3_experts_prepare,
					       RETAINED_SLICES, "retained") &&
				 passed;
			passed = products_case(card, &ranges, &experts, bits,
					       module_class,
					       exl3_experts_fast_prepare,
					       RETAINED_SLICES, "wide") &&
				 passed;
			if (module_class == ARITHMETIC_CLASS_SM_80)
				passed = products_case(
						 card, &ranges, &experts, bits,
						 module_class,
						 exl3_experts_relaxed_prepare,
						 RELAXED_SLICES, "relaxed") &&
					 passed;
		}
		if (laid_out && bits == EXL3_K3) {
			float sums[ROWS * INPUT];
			const bool ran = block_case(card, &ranges, bits,
						    module_class, sums);
			passed = ran && passed;
			if (ran) {
				(void)printf(
					"block K%u, class %u: row sums digest %016" PRIx64
					"\n",
					bits, module_class, sums_digest(sums));
				struct block_result *const kept =
					module_class == ARITHMETIC_CLASS_SM_120_NATIVE ?
						&class_120_block :
						&class_80_block;
				if (module_class == ARITHMETIC_CLASS_SM_120) {
					memcpy(emulated_sums, sums,
					       sizeof(sums));
					emulated_seen = true;
				} else if (!kept->is_set) {
					memcpy(kept->sums, sums, sizeof(sums));
					kept->is_set = true;
				}
			}
		}
		experts_free(&experts);
	}
	return passed;
}

/* class 80's sums emulated against a card of class 80; class 120's own against class 80's within the bounds */
static bool block_comparisons(void)
{
	bool passed = true;
	if (class_80_block.is_set && emulated_seen)
		passed = expect_floats(
			"block: class 80's sums emulated against a card of class 80",
			emulated_sums, class_80_block.sums,
			(uint64_t)ROWS * INPUT);
	if (class_80_block.is_set && class_120_block.is_set) {
		double largest = 0.0, worst = 0.0, squares = 0.0;
		for (uint32_t at = 0; at < ROWS * INPUT; at++) {
			const double reference =
				(double)class_80_block.sums[at];
			const double difference =
				(double)class_120_block.sums[at] - reference;
			largest = fmax(largest, fabs(reference));
			worst = fmax(worst, fabs(difference));
			squares += difference * difference;
		}
		const double relative = largest > 0.0 ? worst / largest : worst;
		const double rms =
			largest > 0.0 ? sqrt(squares / (double)(ROWS * INPUT)) /
						largest :
					0.0;
		(void)printf(
			"block: class 120's own sums against class 80's: largest relative %.3e, rms %.3e\n",
			relative, rms);
		passed = test_linear_expect(
				 relative <= MOST_RELATIVE && rms <= MOST_RMS,
				 "class 120's own block within the bounds") &&
			 passed;
	}
	if (!(class_80_block.is_set &&
	      (emulated_seen || class_120_block.is_set)))
		(void)printf(
			"block: no card of the other class: nothing compared across cards\n");
	return passed;
}

int main(void)
{
	uint32_t card_count = 0;
	if (!test_linear_expect_status(
		    backend_cuda_value.card_count(&card_count), STATUS_OK,
		    "the cards") ||
	    !test_linear_expect(card_count != 0, "a card"))
		return 1;
	for (uint32_t index = 0; index < card_count; index++) {
		struct test_linear_card card;
		if (!test_linear_card_open(&card, index, WORK_CAPACITY,
					   WEIGHTS_CAPACITY))
			continue;
		const uint32_t builds[MOST_BUILDS] = {
			card.arithmetic_class, ARITHMETIC_CLASS_SM_120_NATIVE
		};
		const uint32_t build_count =
			card.arithmetic_class == ARITHMETIC_CLASS_SM_120 ?
				MOST_BUILDS :
				1u;
		for (uint32_t build = 0; build < build_count; build++) {
			(void)printf("card %u, build of class %u\n", index,
				     builds[build]);
			(void)build_cases(&card, builds[build]);
		}
		test_linear_card_close(&card);
	}
	(void)block_comparisons();
	const int failures = test_linear_failures();
	(void)printf("EXL3 SM_120 card test %s: %d failures\n",
		     failures == 0 ? "PASS" : "FAIL", failures);
	return failures == 0 ? 0 : 1;
}
