/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * trunk_relaxed_card_test: the trunk products of class 80 in the relaxed order and the ordered wide instance
 * (relaxed_linear_records.h) on ONE card of class 80, against the reference order's thin launches
 * (trunk_linear_quantized_<S>, trunk_linear_bfloat16_<S>; linear_records.h) and the double
 * reference (linear_reference.h).
 *
 *     trunk_relaxed_card_test --bus=<the card's PCI address> [--time]
 *
 * The card is the one card the process sees (CUDA_VISIBLE_DEVICES), and it must sit at the bus address given: any
 * other card, or more than one, FAILS the test before a byte is placed on it. It shares the card: its arena is
 * WORK_CAPACITY plus WEIGHTS_CAPACITY bytes. For every trunk shape of the model:
 *   - The matrices (scale codes SCALE_CODE_BASE to that plus SCALE_CODE_SPREAD) and MOST_ROWS rows (plain values,
 *     every fourth row of the edges, or small values of bfloat16 rows) are laid out; quantized rows are quantized on
 *     the card. They run through the thin launches of 8 rows (the reference order), the relaxed launches of 1, 8, 64,
 *     100, 256, 1,024 and 4,096 rows (decode form below 9, wide form above, as relaxed_linear.c prepares them) and,
 *     the ordered instance at 100 and 4,096 rows. Every output starts as rubbish of its own.
 *   - Compared: the ordered instance's outputs must be the thin launches' byte for byte. The wide form's rows must
 *     have the same bytes at every row count (a row's bits depend on its matrix and itself); the decode form at 1
 *     and 8 rows likewise among themselves. The relaxed outputs of 4,096 rows against the thin launches': how many
 *     bfloat16 values differ, the largest distance in last places, the rel-RMS difference and the largest relative
 *     difference. Both against the double reference over REFERENCE_ROWS rows of each form: rel-RMS error,
 *     largest error relative to |y| and to the magnitude sum m, the share bit-equal to y rounded once, and the
 *     outputs outside the tolerance of linear_reference.h (its S: the instructions of 16 inputs of the relaxed
 *     order, 1 of the reference order). These are measurements; the test fails on a byte that must be equal, a
 *     guard written, or an output outside the tolerance.
 *   - The cancelling case: the same shapes with a matrix whose even tiles' upper half of each input run is the
 *     lower half of the other sign (one scale code) and rows whose upper half is the lower half, one value in
 *     CANCEL_ONE_IN of the other sign: the products cancel to a small part of their magnitudes. 64 rows (and 8 of
 *     the decode form), measured against the double reference as above.
 *   - With --time each launch is timed: the median of TIMED_RUNS runs after one that warms the card, each after a
 *     read of the work region (WORK_CAPACITY, more than four of a 170HX's 32 MiB L2), so that the weights come from
 *     memory: thin launches (back to back), relaxed (as prepared, and with the tiles of 128 rows, splits 2 to 4
 *     where their partial sums fit PARTIALS_BYTES, the decode form in every stamped slice count), ordered.
 *     Reported: microseconds, TFLOP/s (2 N K rows) and GB/s (weights, operands or values, outputs), against the
 *     card's roofs. A marker launch before each timed group lets a trace of the card (nsys) give each
 *     run's own time (receipts: trace_times.py): the host clock here waits in steps of about 75 us.
 * Run it with the card's clocks locked (clocks_lock.sh on the build workstation).
 */
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "anemone/backend_seam.h"
#include "anemone/linear.h"
#include "anemone/linear_records.h"
#include "anemone/relaxed_linear.h"
#include "anemone/status.h"
#include "anemone/weight_format_seam.h"
#include "linear_card_fixture.h"
#include "linear_reference.h"

#define TEST_TIME_LIMIT_SECONDS 7200
#define WORK_CAPACITY \
	(1280ull      \
	 << 20u) /* attn.wq_b: 13,841 rows of outputs of 64 KiB, its rows and splits */
#define WEIGHTS_CAPACITY (160ull << 20u) /* engram.wkv: 150.1 MiB */
#define COUNT_OF(table) ((uint32_t)(sizeof(table) / sizeof((table)[0])))
#define MATRICES LINEAR_MERGED_MATRICES
#define GROUP_ROWS LINEAR_MAXIMUM_ROWS
#define MOST_ROWS 4096u
#define CANCEL_ROWS 64u
#define DECODE_CANCEL_ROWS 8u
#define REFERENCE_ROWS \
	6u /* of each case: rows of each kind and of both forms */
#define E4M3 WEIGHT_FORMAT_E4M3_SCALED
#define E4M3_BITS_PER_WEIGHT 8u
#define E4M3_SIGN 0x80u
#define BFLOAT16_SIGN 0x8000u
#define SCALE_CODE_BASE \
	110u /* the folds are exact from 53 to 150; 16-bit floats span 22 */
#define SCALE_CODE_SPREAD 12u
#define CANCEL_SCALE_CODE 116u
#define CANCEL_ONE_IN 256u
#define EDGE_ROW_ONE_IN \
	4u /* every fourth row: the edges (quantized) or small values (bfloat16) */
#define STREAM 0x14000000ull
#define MATRIX_STREAM 0x100ull
#define ROW_STREAM 0x10000ull
#define CANCEL_STREAM 0x800000ull
#define RUBBISH_BYTE 0x5Bu
#define TIMED_RUNS 5u
#define HALVES 2u
#define MEDIAN_RUN 2u /* the middle of five sorted runs */
#define MOST_ARGUMENTS 3
#define BUS_OPTION "--bus="
#define TIME_OPTION "--time"
#define NANOSECONDS_PER_SECOND 1000000000ull
#define NANOSECONDS_PER_MICROSECOND 1000.0
#define FLOPS_PER_WEIGHT_ROW 2.0 /* a multiply and an add */
#define TERA 1.0e12
#define GIGA 1.0e9
#define MICROSECONDS_PER_SECOND 1.0e6
#define ROOF_BFLOAT16_TFLOPS 172.0 /* SWEEP_RESULTS/08: a 170HX at 150 W */
#define ROOF_HALF_TFLOPS 159.0
#define ROOF_GIGABYTES_PER_SECOND 1440.0
#define SPLIT_TWO 2u
#define SPLIT_FOUR 4u
#define MOST_LAUNCHES 1024u
#define PERCENT 100.0
#define MARKER_BASE 60000u /* grids of the trace markers of the timed runs */
#define TEXT_CAPACITY 96u
#define MEBIBYTE_SHIFT 20u
#define PARTIALS_BYTES \
	(128ull        \
	 << 20u) /* the split variants' partial sums: timed where they fit */
#define STEP_INPUTS \
	16u /* inputs of one instruction: the relaxed order rounds once a step */

enum test_status { TEST_PASSED = 0, TEST_FAILED = 1, TEST_WRONG_USAGE = 2 };

/* A trunk shape of the model; decode_slices: the decode form's slice count of it (from the sweep of this test). */
struct trunk_shape {
	const char *name;
	uint32_t output_sizes[MATRICES];
	uint32_t input_size;
	bool is_bfloat16;
	uint32_t segment_count;
	uint32_t decode_slices;
};
static const struct trunk_shape trunk_shapes[] = {
	{ "attn.wq_a with attn.wkv", { 1280, 512 }, 5120, false, 1, 16 },
	{ "attn.wq_b", { 32768, 0 }, 1280, false, 1, 1 },
	{ "attn.indexer.wq_b", { 4096, 0 }, 1280, false, 1, 4 },
	{ "attn.wo_b", { 5120, 0 }, 8192, false, 1, 4 },
	{ "engram.wkv", { 25600, 0 }, 6144, false, 1, 4 },
	{ "the shared expert's gate with up", { 2304, 2304 }, 5120, false, 1, 8 },
	{ "the shared expert's down", { 5120, 0 }, 2304, false, 1, 4 },
	{ "attn.wo_a", { 8192, 0 }, 4096, true, 8, 4 },
	/* the draft context's (model_v41_program.c draft_context_piece): a BATCH step's 9-64 rows take the wide forms */
	{ "mtp main_proj", { 5120, 0 }, 15360, false, 1, 4 },
	{ "mtp.0.attn.wkv with mtp.1.attn.wkv", { 512, 512 }, 5120, false, 1, 16 },
};

/* The relaxed row counts; the ordered instance's are the wide ones of these at ORDERED_FIRST_CASE and the last. */
static const uint32_t case_rows[] = { 1, 8, 64, 100, 256, 1024, MOST_ROWS };
#define CASES COUNT_OF(case_rows)
#define MOST_CASES 7u
#define FIRST_WIDE_CASE 2u
#define UNEVEN_CASE 3u
#define LAST_CASE (MOST_CASES - 1u)
static const uint32_t stamped_slices[] = { 1, 2, 4, 8, 16 };
/* rows held against the double reference: plain and edge rows of the decode form's 8, then the middle and the last */
static const uint32_t reference_early_rows[] = { 0, 3, 5, 7 };

/* What one shape lays out, and where its launches lie. */
struct shape_case {
	const struct trunk_shape *shape;
	bool is_cancelling;
	uint32_t matrix_count;
	uint32_t row_count; /* laid out */
	uint32_t row_width; /* segment_count * input_size */
	int32_t fold_reference;
	struct linear_matrix
		matrices[MATRICES]; /* the reference order's slice counts */
	struct linear_range values, codes, exponents, operands, partials;
	struct linear_range thin_outputs[MATRICES];
	struct linear_range relaxed_outputs[MOST_CASES][MATRICES];
	struct linear_range ordered_outputs[MOST_CASES][MATRICES];
	uint32_t thin_first, thin_count;
	uint32_t relaxed_first[MOST_CASES], relaxed_count[MOST_CASES];
	uint32_t ordered_first[MOST_CASES];
	uint32_t launch_count;
};

static struct kernel_launch launches[MOST_LAUNCHES];
static struct relaxed_linear_kernels relaxed_kernels;
static uint16_t row_values[RELAXED_MOST_INPUT_SIZE * GROUP_ROWS];
static uint32_t
	cancelling_half; /* input_size / 2 of the cancelling matrix being laid out */

_Static_assert(
	COUNT_OF(case_rows) == MOST_CASES,
	"the cases: two of the decode form's, then the wide form's, the last of the most rows");

static uint64_t clock_nanoseconds(void)
{
	struct timespec now = { 0 };
	return clock_gettime(CLOCK_MONOTONIC, &now) == 0 ?
		       (uint64_t)now.tv_sec * NANOSECONDS_PER_SECOND +
			       (uint64_t)now.tv_nsec :
		       0;
}

static struct linear_range rows_of(struct linear_range range,
				   uint64_t row_bytes, uint32_t first_row,
				   uint32_t row_count)
{
	return (struct linear_range){ range.byte_offset + first_row * row_bytes,
				      row_count * row_bytes };
}

/* The codes of the cancelling matrix: even tiles' upper half the lower half of the other sign. */
static uint8_t cancelling_code(uint64_t stream, uint32_t output_index,
			       uint32_t input_index)
{
	const bool is_upper = input_index >= cancelling_half;
	const uint8_t code = test_linear_e4m3_code(
		stream, output_index,
		is_upper ? input_index - cancelling_half : input_index);
	const bool is_flipped =
		is_upper &&
		output_index / LINEAR_OUTPUTS_PER_TILE % HALVES == 0;
	return (uint8_t)(is_flipped ? code ^ E4M3_SIGN : code);
}

/* Row `row` of a case's values: plain, edges or small; cancelling: each segment's upper half its lower half. */
static void case_row_values(const struct shape_case *shape_case,
			    uint32_t shape_index, uint32_t row,
			    uint16_t *values)
{
	const struct trunk_shape *const shape = shape_case->shape;
	const bool is_edge = row % EDGE_ROW_ONE_IN == EDGE_ROW_ONE_IN - 1u &&
			     !shape_case->is_cancelling;
	const enum test_linear_values_kind kind =
		!is_edge	   ? TEST_LINEAR_VALUES_PLAIN :
		shape->is_bfloat16 ? TEST_LINEAR_VALUES_SMALL :
				     TEST_LINEAR_VALUES_EDGES;
	const uint64_t stream =
		STREAM + ROW_STREAM * (shape_index + 1u) + row +
		(shape_case->is_cancelling ? CANCEL_STREAM : 0u);
	test_linear_values(values, shape_case->row_width, stream, kind);
	const uint32_t half = shape->input_size / HALVES;
	for (uint32_t segment = 0;
	     shape_case->is_cancelling && segment < shape->segment_count;
	     segment++) {
		uint16_t *const segment_values =
			values + (size_t)segment * shape->input_size;
		for (uint32_t index = 0; index < half; index++) {
			const bool is_flipped =
				test_linear_hash(stream, index) %
					CANCEL_ONE_IN ==
				0;
			segment_values[half + index] =
				(uint16_t)(segment_values[index] ^
					   (is_flipped ? BFLOAT16_SIGN : 0u));
		}
	}
}

/* The reference order's thin launch of rows first_row .. + 8 (or fewer) into outputs. */
MUST_CHECK static result_status
thin_prepare(const struct test_linear_card *card,
	     const struct shape_case *shape_case, uint32_t first_row,
	     uint32_t row_count, struct kernel_launch *launch_out)
{
	const struct trunk_shape *const shape = shape_case->shape;
	if (shape->is_bfloat16) {
		const struct linear_trunk_bfloat16_description description = {
			.matrix = shape_case->matrices[0],
			.values = rows_of(shape_case->values,
					  (uint64_t)shape_case->row_width *
						  LINEAR_BFLOAT16_BYTES,
					  first_row, row_count),
			.output = rows_of(shape_case->thin_outputs[0],
					  (uint64_t)shape->output_sizes[0] *
						  LINEAR_BFLOAT16_BYTES,
					  first_row, row_count),
			.row_count = row_count,
			.segment_count = shape->segment_count
		};
		return linear_trunk_bfloat16_prepare(&description,
						     card->arena_byte_count,
						     &card->kernels,
						     launch_out);
	}
	struct linear_trunk_quantized_description description = {
		.matrices = { shape_case->matrices[0],
			      shape_case->matrices[1] },
		.codes = rows_of(shape_case->codes, shape->input_size,
				 first_row, row_count),
		.exponents = rows_of(shape_case->exponents,
				     shape->input_size /
					     TEST_LINEAR_VALUES_PER_BLOCK,
				     first_row, row_count),
		.operands = rows_of(shape_case->operands,
				    (uint64_t)shape->input_size *
					    LINEAR_BFLOAT16_BYTES,
				    first_row, row_count),
		.matrix_count = shape_case->matrix_count,
		.row_count = row_count
	};
	for (uint32_t matrix = 0; matrix < shape_case->matrix_count; matrix++) {
		description.outputs[matrix] =
			rows_of(shape_case->thin_outputs[matrix],
				(uint64_t)shape->output_sizes[matrix] *
					LINEAR_BFLOAT16_BYTES,
				first_row, row_count);
	}
	return linear_trunk_quantized_prepare(&description,
					      card->arena_byte_count,
					      &card->kernels, launch_out);
}

/* A relaxed (or ordered) launch of rows 0 .. row_count - 1 into outputs, in slice_count slices (decode form) and
 * k_split_count splits; launches_out holds 1 or 2. */
MUST_CHECK static result_status
relaxed_prepare(const struct test_linear_card *card,
		const struct shape_case *shape_case, uint32_t row_count,
		const struct linear_range *outputs, uint32_t slice_count,
		uint32_t k_split_count, bool is_ordered,
		struct kernel_launch *launches_out, uint32_t *launch_count_out)
{
	const struct trunk_shape *const shape = shape_case->shape;
	struct linear_matrix matrices[MATRICES] = { shape_case->matrices[0],
						    shape_case->matrices[1] };
	const struct linear_range partials = {
		shape_case->partials.byte_offset,
		k_split_count > 1u ?
			linear_sums_byte_count(
				(uint64_t)k_split_count * row_count,
				(uint64_t)shape->output_sizes[0] +
					shape->output_sizes[1]) :
			0u
	};
	matrices[0].slice_count = slice_count;
	matrices[1].slice_count = slice_count;
	if (shape->is_bfloat16) {
		const struct relaxed_trunk_bfloat16_description description = {
			.matrix = matrices[0],
			.values = rows_of(shape_case->values,
					  (uint64_t)shape_case->row_width *
						  LINEAR_BFLOAT16_BYTES,
					  0, row_count),
			.output = rows_of(outputs[0],
					  (uint64_t)shape->output_sizes[0] *
						  LINEAR_BFLOAT16_BYTES,
					  0, row_count),
			.partials = partials,
			.row_count = row_count,
			.segment_count = shape->segment_count,
			.k_split_count = k_split_count,
			.fold_reference = shape_case->fold_reference,
			.is_ordered = is_ordered ? 1u : 0u
		};
		return relaxed_trunk_bfloat16_prepare(
			&description, card->arena_byte_count, &relaxed_kernels,
			launches_out, launch_count_out);
	}
	struct relaxed_trunk_quantized_description description = {
		.matrices = { matrices[0], matrices[1] },
		.operands = rows_of(shape_case->operands,
				    (uint64_t)shape->input_size *
					    LINEAR_BFLOAT16_BYTES,
				    0, row_count),
		.partials = partials,
		.matrix_count = shape_case->matrix_count,
		.row_count = row_count,
		.k_split_count = k_split_count,
		.is_ordered = is_ordered ? 1u : 0u
	};
	for (uint32_t matrix = 0; matrix < shape_case->matrix_count; matrix++) {
		description.outputs[matrix] =
			rows_of(outputs[matrix],
				(uint64_t)shape->output_sizes[matrix] *
					LINEAR_BFLOAT16_BYTES,
				0, row_count);
	}
	return relaxed_trunk_quantized_prepare(&description,
					       card->arena_byte_count,
					       &relaxed_kernels, launches_out,
					       launch_count_out);
}

/* A range of outputs of `rows` rows for each matrix, filled with rubbish. */
static bool outputs_set_aside(struct test_linear_card *card,
			      const struct shape_case *shape_case,
			      uint32_t rows, struct linear_range *outputs)
{
	static unsigned char rubbish[RELAXED_MOST_INPUT_SIZE];
	bool is_set = true;
	memset(rubbish, RUBBISH_BYTE, sizeof(rubbish));
	for (uint32_t matrix = 0; matrix < shape_case->matrix_count; matrix++) {
		outputs[matrix] = test_linear_card_work(
			card, (uint64_t)rows *
				      shape_case->shape->output_sizes[matrix] *
				      LINEAR_BFLOAT16_BYTES);
		for (uint64_t done = 0;
		     done < outputs[matrix].byte_count && is_set;
		     done += sizeof(rubbish)) {
			const uint64_t left = outputs[matrix].byte_count - done;
			is_set = test_linear_card_put(card, outputs[matrix],
						      done, rubbish,
						      left < sizeof(rubbish) ?
							      left :
							      sizeof(rubbish));
		}
	}
	return is_set;
}

/* Lays a case out: matrices, rows (and their quantization), outputs; then every launch. */
static bool case_lay_out(struct test_linear_card *card,
			 struct shape_case *shape_case, uint32_t shape_index)
{
	const struct trunk_shape *const shape = shape_case->shape;
	const uint32_t input_size = shape->input_size,
		       row_count = shape_case->row_count;
	shape_case->matrix_count = shape->output_sizes[1] != 0 ? MATRICES : 1u;
	shape_case->row_width = shape->segment_count * input_size;
	cancelling_half = input_size / HALVES;
	test_linear_card_clear(card);
	bool is_laid_out = true;
	for (uint32_t matrix = 0;
	     matrix < shape_case->matrix_count && is_laid_out; matrix++) {
		const struct test_linear_matrix_recipe recipe = {
			.format = E4M3,
			.shape = { shape->output_sizes[matrix], input_size,
				   E4M3_BITS_PER_WEIGHT, 0 },
			.stream = STREAM +
				  MATRIX_STREAM * (shape_index * MATRICES +
						   matrix + 1u),
			.code = shape_case->is_cancelling ?
					cancelling_code :
					test_linear_e4m3_code,
			.scale_code_base = shape_case->is_cancelling ?
						   CANCEL_SCALE_CODE :
						   SCALE_CODE_BASE,
			.scale_code_spread = shape_case->is_cancelling ?
						     0u :
						     SCALE_CODE_SPREAD
		};
		uint32_t slices = 0;
		is_laid_out =
			test_linear_expect_status(
				linear_default_slice_count(
					LINEAR_PROFILE_PRECISION,
					LINEAR_ORDER_FIXED,
					shape->is_bfloat16 ?
						LINEAR_FAMILY_TRUNK_BFLOAT16 :
						LINEAR_FAMILY_TRUNK_QUANTIZED,
					&recipe.shape, card->arithmetic_class,
					&slices),
				STATUS_OK,
				"the shape has a default slice count") &&
			test_linear_card_matrix(
				card, &recipe,
				&shape_case->matrices[matrix].stored);
		shape_case->matrices[matrix].shape = recipe.shape;
		shape_case->matrices[matrix].weight_format = E4M3;
		shape_case->matrices[matrix].slice_count = slices;
		/* the fold check of the matrix, on its scale codes in the image */
		const uint8_t *const stored = test_linear_card_stored(
			card, shape_case->matrices[matrix].stored);
		const uint64_t weight_count =
			(uint64_t)recipe.shape.output_size * input_size;
		const uint64_t scale_count =
			shape_case->matrices[matrix].stored.byte_count -
			weight_count;
		is_laid_out =
			is_laid_out && stored != NULL &&
			test_linear_expect_status(
				relaxed_fold_check(
					shape->is_bfloat16 ?
						RELAXED_BFLOAT16_ROWS :
						RELAXED_QUANTIZED_ROWS,
					stored + weight_count, scale_count,
					&shape_case->fold_reference),
				STATUS_OK, "the fold of the matrix is exact");
	}
	shape_case->values = test_linear_card_work(
		card, (uint64_t)row_count * shape_case->row_width *
			      LINEAR_BFLOAT16_BYTES);
	if (!shape->is_bfloat16) {
		shape_case->codes = test_linear_card_work(
			card, (uint64_t)row_count * input_size);
		shape_case->exponents = test_linear_card_work(
			card,
			(uint64_t)row_count *
				(input_size / TEST_LINEAR_VALUES_PER_BLOCK));
		shape_case->operands = test_linear_card_work(
			card, (uint64_t)row_count * input_size *
				      LINEAR_BFLOAT16_BYTES);
	}
	is_laid_out = is_laid_out &&
		      outputs_set_aside(card, shape_case, row_count,
					shape_case->thin_outputs);
	for (uint32_t case_index = 0; case_index < CASES && is_laid_out;
	     case_index++) {
		const uint32_t rows = case_rows[case_index];
		const bool is_ordered_case = case_index == UNEVEN_CASE ||
					     case_index == LAST_CASE;
		if (rows > row_count) {
			continue;
		}
		is_laid_out =
			outputs_set_aside(
				card, shape_case, rows,
				shape_case->relaxed_outputs[case_index]) &&
			(!is_ordered_case ||
			 outputs_set_aside(
				 card, shape_case, rows,
				 shape_case->ordered_outputs[case_index]));
	}
	shape_case->partials = test_linear_card_work(card, PARTIALS_BYTES);
	is_laid_out = is_laid_out && test_linear_card_has_room(card);
	for (uint32_t row = 0; row < row_count && is_laid_out; row++) {
		case_row_values(shape_case, shape_index, row, row_values);
		const uint64_t row_bytes =
			(uint64_t)shape_case->row_width * LINEAR_BFLOAT16_BYTES;
		is_laid_out = test_linear_card_put(card, shape_case->values,
						   row * row_bytes, row_values,
						   row_bytes);
	}
	/* the launches: the quantization, the thin launches, the relaxed and ordered ones of each case */
	uint32_t count = 0;
	if (is_laid_out && !shape->is_bfloat16) {
		const struct linear_quantize_description quantize = {
			shape_case->values,
			shape_case->codes,
			shape_case->exponents,
			shape_case->operands,
			input_size,
			row_count
		};
		is_laid_out = test_linear_expect_status(
			linear_quantize_prepare(
				&quantize, card->arena_byte_count,
				&card->kernels, &launches[count++]),
			STATUS_OK, "the quantization is prepared");
	}
	shape_case->thin_first = count;
	for (uint32_t first_row = 0; first_row < row_count && is_laid_out;
	     first_row += GROUP_ROWS) {
		const uint32_t rows = row_count - first_row < GROUP_ROWS ?
					      row_count - first_row :
					      GROUP_ROWS;
		is_laid_out = test_linear_expect_status(
			thin_prepare(card, shape_case, first_row, rows,
				     &launches[count++]),
			STATUS_OK, "a thin launch is prepared");
	}
	shape_case->thin_count = count - shape_case->thin_first;
	for (uint32_t case_index = 0; case_index < CASES && is_laid_out;
	     case_index++) {
		const uint32_t rows = case_rows[case_index];
		const bool is_ordered_case = case_index == UNEVEN_CASE ||
					     case_index == LAST_CASE;
		uint32_t added = 0;
		if (rows > row_count) {
			continue;
		}
		shape_case->relaxed_first[case_index] = count;
		is_laid_out = test_linear_expect_status(
			relaxed_prepare(card, shape_case, rows,
					shape_case->relaxed_outputs[case_index],
					shape->decode_slices, 1, false,
					&launches[count], &added),
			STATUS_OK, "a relaxed launch is prepared");
		shape_case->relaxed_count[case_index] = added;
		count += added;
		shape_case->ordered_first[case_index] = count;
		if (is_laid_out && is_ordered_case) {
			is_laid_out = test_linear_expect_status(
				relaxed_prepare(
					card, shape_case, rows,
					shape_case->ordered_outputs[case_index],
					shape->decode_slices, 1, true,
					&launches[count], &added),
				STATUS_OK, "an ordered launch is prepared");
			count += added;
		}
	}
	shape_case->launch_count = count;
	return is_laid_out;
}

/* Differences between two output arrays of `count` bfloat16 values. */
struct output_difference {
	uint64_t count;
	uint64_t differing;
	uint64_t beyond_one_place; /* finite pairs more than one last place apart */
	uint32_t largest_places;
	double squared_difference, squared_value, largest_relative;
};

/* rel-RMS over every finite pair; the relative difference of the pairs that differ. */
static void difference_add(struct output_difference *difference,
			   const uint16_t *outputs, const uint16_t *against,
			   uint64_t count)
{
	for (uint64_t index = 0; index < count; index++) {
		const double value = test_linear_bfloat16_value(outputs[index]);
		const double expected =
			test_linear_bfloat16_value(against[index]);
		difference->count++;
		if (!test_linear_bfloat16_is_finite(outputs[index]) ||
		    !test_linear_bfloat16_is_finite(against[index])) {
			difference->differing +=
				outputs[index] != against[index] ? 1u : 0u;
			continue;
		}
		difference->squared_value += expected * expected;
		if (outputs[index] == against[index]) {
			continue;
		}
		const int32_t places =
			test_linear_bfloat16_ordinal(outputs[index]) -
			test_linear_bfloat16_ordinal(against[index]);
		const uint32_t distance =
			(uint32_t)(places < 0 ? -places : places);
		difference->differing++;
		difference->beyond_one_place += distance > 1u ? 1u : 0u;
		difference->largest_places =
			distance > difference->largest_places ?
				distance :
				difference->largest_places;
		difference->squared_difference +=
			(value - expected) * (value - expected);
		if (expected != 0.0) {
			const double relative =
				fabs(value - expected) / fabs(expected);
			difference->largest_relative =
				relative > difference->largest_relative ?
					relative :
					difference->largest_relative;
		}
	}
}

/* Errors of outputs against the double reference. */
struct reference_error {
	struct test_linear_comparison comparison;
	double squared_error, squared_sum, largest_relative,
		largest_of_magnitude;
};

/* Every tile_step-th tile of 16 outputs (2: the cancelling case's even tiles alone). */
static void reference_error_add(struct reference_error *error,
				const uint16_t *outputs,
				const struct test_linear_product *products,
				uint64_t count, uint32_t input_size,
				uint32_t slice_count, uint32_t tile_step)
{
	for (uint64_t first = 0; first < count;
	     first += (uint64_t)tile_step * LINEAR_OUTPUTS_PER_TILE) {
		(void)test_linear_compare(outputs + first, products + first,
					  LINEAR_OUTPUTS_PER_TILE, input_size,
					  slice_count, &error->comparison);
		for (uint64_t index = first;
		     index < first + LINEAR_OUTPUTS_PER_TILE; index++) {
			if (!test_linear_bfloat16_is_finite(outputs[index]) ||
			    !isfinite(products[index].sum)) {
				continue;
			}
			const double difference =
				test_linear_bfloat16_value(outputs[index]) -
				products[index].sum;
			const double sum = products[index].sum;
			const double magnitude = products[index].magnitude_sum;
			error->squared_error += difference * difference;
			error->squared_sum += sum * sum;
			if (sum != 0.0 && fabs(difference) / fabs(sum) >
						  error->largest_relative) {
				error->largest_relative =
					fabs(difference) / fabs(sum);
			}
			if (magnitude != 0.0 &&
			    fabs(difference) / magnitude >
				    error->largest_of_magnitude) {
				error->largest_of_magnitude =
					fabs(difference) / magnitude;
			}
		}
	}
}

static double rms_ratio(double squared_numerator, double squared_denominator)
{
	return squared_denominator > 0.0 ?
		       sqrt(squared_numerator / squared_denominator) :
		       0.0;
}

static void reference_error_print(const char *what,
				  const struct reference_error *error)
{
	(void)printf(
		"    %s against the double reference: rel-RMS %.3e, largest |c - y| / |y| %.3e, largest |c - y| / m "
		"%.3e, %.2f%% bit-equal to y rounded once, %" PRIu64
		" of %" PRIu64 " outside the tolerance, "
		"largest %" PRIu32 " last places\n",
		what, rms_ratio(error->squared_error, error->squared_sum),
		error->largest_relative, error->largest_of_magnitude,
		test_linear_percent(error->comparison.equal_count,
				    error->comparison.output_count),
		error->comparison.outside_count, error->comparison.output_count,
		error->comparison.largest_distance);
	(void)test_linear_expect(error->comparison.outside_count == 0,
				 "every output lies within the tolerance");
}

/* The outputs of row `row` (all matrices, concatenated) from ranges of `rows` rows into outputs_out. */
static bool row_outputs(const struct test_linear_card *card,
			const struct shape_case *shape_case,
			const struct linear_range *ranges, uint32_t row,
			uint16_t *outputs_out)
{
	bool is_got = true;
	uint32_t offset = 0;
	for (uint32_t matrix = 0; matrix < shape_case->matrix_count && is_got;
	     matrix++) {
		const uint64_t row_bytes =
			(uint64_t)shape_case->shape->output_sizes[matrix] *
			LINEAR_BFLOAT16_BYTES;
		is_got = test_linear_card_get(card, ranges[matrix],
					      row * row_bytes,
					      outputs_out + offset, row_bytes);
		offset += shape_case->shape->output_sizes[matrix];
	}
	return is_got;
}

/* The double reference of row `row`: products_out [outputs of all matrices]. */
static bool row_products(const struct test_linear_card *card,
			 const struct shape_case *shape_case,
			 uint32_t shape_index, uint32_t row, double *fed_values,
			 uint8_t *codes, uint8_t *exponents,
			 struct test_linear_product *products_out)
{
	const struct trunk_shape *const shape = shape_case->shape;
	bool is_made = true;
	uint32_t offset = 0;
	case_row_values(shape_case, shape_index, row, row_values);
	if (shape->is_bfloat16) {
		is_made = test_linear_expect_status(
			test_linear_fed_values_of_bfloat16(
				row_values, shape_case->row_width, fed_values),
			STATUS_OK, "the reference feeds the row");
	} else {
		is_made = test_linear_expect_status(
				  test_linear_quantize(row_values,
						       shape->input_size, codes,
						       exponents),
				  STATUS_OK,
				  "the reference quantizes the row") &&
			  test_linear_expect_status(
				  test_linear_fed_values_of_codes(
					  codes, shape->input_size, fed_values),
				  STATUS_OK, "the reference feeds the row");
	}
	for (uint32_t matrix = 0; matrix < shape_case->matrix_count && is_made;
	     matrix++) {
		const struct linear_matrix *const of =
			&shape_case->matrices[matrix];
		is_made = test_linear_expect_status(
			test_linear_products(
				E4M3, &of->shape,
				test_linear_card_stored(card, of->stored),
				of->stored.byte_count, 0, of->shape.output_size,
				fed_values,
				shape->is_bfloat16 ? NULL : exponents, 1,
				shape->segment_count, products_out + offset),
			STATUS_OK, "the reference computes the row's products");
		offset += of->shape.output_size;
	}
	return is_made;
}

static void comparison_memory_free(uint16_t *outputs, uint16_t *against,
				   struct test_linear_product *products,
				   double *fed_values, uint8_t *codes,
				   uint8_t *exponents)
{
	free(outputs);
	free(against);
	free(products);
	free(fed_values);
	free(codes);
	free(exponents);
}

/* Every comparison of a case (see the top). */
static void case_check(struct test_linear_card *card,
		       const struct shape_case *shape_case,
		       uint32_t shape_index)
{
	const struct trunk_shape *const shape = shape_case->shape;
	const uint32_t output_total =
		shape->output_sizes[0] + shape->output_sizes[1];
	const uint32_t row_count = shape_case->row_count;
	uint16_t *const outputs = calloc(output_total, LINEAR_BFLOAT16_BYTES);
	uint16_t *const against = calloc(output_total, LINEAR_BFLOAT16_BYTES);
	struct test_linear_product *const products =
		calloc(output_total, sizeof(*products));
	double *const fed_values =
		malloc((size_t)shape_case->row_width * sizeof(double));
	uint8_t *const codes = malloc(shape->input_size);
	uint8_t *const exponents =
		malloc(shape->input_size / TEST_LINEAR_VALUES_PER_BLOCK);
	if (outputs == NULL || against == NULL || products == NULL ||
	    fed_values == NULL || codes == NULL || exponents == NULL) {
		(void)test_linear_expect(false, "the comparison has memory");
		comparison_memory_free(outputs, against, products, fed_values,
				       codes, exponents);
		return;
	}
	const uint32_t last_case = shape_case->is_cancelling ? FIRST_WIDE_CASE :
							       LAST_CASE;
	struct output_difference relaxed_against_thin = { 0 },
				 row_count_changes = { 0 },
				 ordered_against_thin = { 0 };
	struct reference_error thin_error = { 0 }, decode_error = { 0 },
			       wide_error = { 0 };
	const uint32_t tile_step = shape_case->is_cancelling ?
					   HALVES :
					   1u; /* the cancelling tiles alone */
	const uint32_t relaxed_slices =
		shape->input_size /
		STEP_INPUTS; /* each instruction rounds as a slice would */
	bool is_got = true;
	for (uint32_t row = 0; row < row_count && is_got; row++) {
		is_got = row_outputs(card, shape_case, shape_case->thin_outputs,
				     row, against) &&
			 row_outputs(card, shape_case,
				     shape_case->relaxed_outputs[last_case],
				     row, outputs);
		if (!is_got) {
			break;
		}
		difference_add(&relaxed_against_thin, outputs, against,
			       output_total);
		/* the same row in every other case of its form */
		for (uint32_t case_index = 0; case_index < CASES && is_got;
		     case_index++) {
			const bool is_same_form =
				(case_index < FIRST_WIDE_CASE) ==
				(last_case < FIRST_WIDE_CASE);
			if (case_index == last_case ||
			    case_rows[case_index] <= row || !is_same_form ||
			    case_rows[case_index] > row_count) {
				continue;
			}
			is_got = row_outputs(
				card, shape_case,
				shape_case->relaxed_outputs[case_index], row,
				against);
			if (!is_got) {
				break;
			}
			difference_add(&row_count_changes, against, outputs,
				       output_total);
		}
		if (row == 0) { /* the decode form's row 0 alone and among 8 */
			is_got = is_got &&
				 row_outputs(card, shape_case,
					     shape_case->relaxed_outputs[0],
					     row, outputs) &&
				 row_outputs(card, shape_case,
					     shape_case->relaxed_outputs
						     [FIRST_WIDE_CASE - 1u],
					     row, against);
			if (!is_got) {
				break;
			}
			difference_add(&row_count_changes, outputs, against,
				       output_total);
		}
		for (uint32_t case_index = UNEVEN_CASE;
		     !shape_case->is_cancelling && case_index < CASES && is_got;
		     case_index += LAST_CASE - UNEVEN_CASE) {
			if (case_rows[case_index] <= row) {
				continue;
			}
			is_got =
				row_outputs(card, shape_case,
					    shape_case->thin_outputs, row,
					    against) &&
				row_outputs(
					card, shape_case,
					shape_case->ordered_outputs[case_index],
					row, outputs);
			if (!is_got) {
				break;
			}
			difference_add(&ordered_against_thin, outputs, against,
				       output_total);
		}
	}
	/* against the double reference: rows of each kind, in the decode form's case of 8 rows and the wide one */
	for (uint32_t index = 0; index < REFERENCE_ROWS && is_got; index++) {
		const uint32_t row =
			index < COUNT_OF(reference_early_rows) ?
				reference_early_rows[index] :
			index == COUNT_OF(reference_early_rows) ?
				row_count / HALVES + EDGE_ROW_ONE_IN - 1u :
				row_count - 1u;
		is_got = row_products(card, shape_case, shape_index, row,
				      fed_values, codes, exponents, products) &&
			 row_outputs(card, shape_case, shape_case->thin_outputs,
				     row, outputs);
		if (!is_got) {
			break;
		}
		reference_error_add(&thin_error, outputs, products,
				    output_total, shape->input_size, 1u,
				    tile_step);
		is_got = is_got &&
			 row_outputs(card, shape_case,
				     shape_case->relaxed_outputs[last_case],
				     row, outputs);
		if (!is_got) {
			break;
		}
		reference_error_add(&wide_error, outputs, products,
				    output_total, shape->input_size,
				    relaxed_slices, tile_step);
		if (is_got && row < case_rows[FIRST_WIDE_CASE - 1u]) {
			is_got = row_outputs(
				card, shape_case,
				shape_case
					->relaxed_outputs[FIRST_WIDE_CASE - 1u],
				row, outputs);
			if (!is_got) {
				break;
			}
			reference_error_add(&decode_error, outputs, products,
					    output_total, shape->input_size,
					    relaxed_slices, tile_step);
		}
	}
	(void)printf("  %s%s, %" PRIu32
		     " rows: relaxed against the reference order: %" PRIu64
		     " of %" PRIu64 " outputs differ (%.2f%%), %" PRIu64
		     " by more than one last place, largest %" PRIu32
		     " last places, rel-RMS %.3e, largest relative %.3e\n",
		     shape->name,
		     shape_case->is_cancelling ? " (cancelling)" : "",
		     row_count, relaxed_against_thin.differing,
		     relaxed_against_thin.count,
		     test_linear_percent(relaxed_against_thin.differing,
					 relaxed_against_thin.count),
		     relaxed_against_thin.beyond_one_place,
		     relaxed_against_thin.largest_places,
		     rms_ratio(relaxed_against_thin.squared_difference,
			       relaxed_against_thin.squared_value),
		     relaxed_against_thin.largest_relative);
	(void)printf(
		"    the same rows at other row counts of the form: %" PRIu64
		" of %" PRIu64 " differ; ordered "
		"instance against the reference order: %" PRIu64 " of %" PRIu64
		" differ\n",
		row_count_changes.differing, row_count_changes.count,
		ordered_against_thin.differing, ordered_against_thin.count);
	(void)test_linear_expect(
		row_count_changes.differing == 0,
		"a row has its bits at every row count of its form");
	(void)test_linear_expect(ordered_against_thin.differing == 0,
				 "the ordered instance is the reference order");
	reference_error_print("reference order", &thin_error);
	reference_error_print("relaxed, decode form", &decode_error);
	reference_error_print(shape_case->is_cancelling ?
				      "relaxed, wide form (64 rows)" :
				      "relaxed, wide form",
			      &wide_error);
	comparison_memory_free(outputs, against, products, fed_values, codes,
			       exponents);
}

static bool l2_flush(struct test_linear_card *card)
{
	uint64_t unused_sum = 0;
	return test_linear_expect_status(
		backend_cuda_value.digest(card->card, 0, card->work_capacity,
					  TEST_LINEAR_WAIT_LIMIT_NANOSECONDS,
					  &unused_sum),
		STATUS_OK, "the work region is read to empty the L2");
}

static double median_microseconds(struct test_linear_card *card,
				  const struct kernel_launch *timed,
				  uint32_t launch_count)
{
	uint64_t durations[TIMED_RUNS];
	bool is_timed = test_linear_card_run(card, timed, launch_count);
	for (uint32_t run = 0; run < TIMED_RUNS; run++) {
		is_timed = is_timed && l2_flush(card);
		const uint64_t started = clock_nanoseconds();
		is_timed = is_timed &&
			   test_linear_card_run(card, timed, launch_count);
		durations[run] = clock_nanoseconds() - started;
	}
	for (uint32_t index = 1; index < TIMED_RUNS; index++) {
		const uint64_t duration = durations[index];
		uint32_t place = index;
		while (place > 0 && durations[place - 1u] > duration) {
			durations[place] = durations[place - 1u];
			place--;
		}
		durations[place] = duration;
	}
	return is_timed ? (double)durations[MEDIAN_RUN] /
				  NANOSECONDS_PER_MICROSECOND :
			  0.0;
}

/* One timing line: microseconds, TFLOP/s and GB/s of rows rows. */
static void time_print(struct test_linear_card *card,
		       const struct shape_case *shape_case, const char *what,
		       uint32_t rows, const struct kernel_launch *timed,
		       uint32_t launch_count)
{
	static uint32_t time_index = 0;
	const struct trunk_shape *const shape = shape_case->shape;
	/* a marker for a trace of the card (nsys): the reduce kernel with no row, its grid MARKER_BASE + the index */
	struct kernel_launch marker = {
		.kernel = relaxed_kernels.reduce,
		.grid = { MARKER_BASE + time_index, 1, 1 },
		.block = { LINEAR_THREADS_PER_WARP, 1, 1 },
		.argument_byte_count = TRUNK_RELAXED_REDUCE_ARGUMENT_BYTES
	};
	(void)test_linear_card_run(card, &marker, 1);
	const double microseconds =
		median_microseconds(card, timed, launch_count);
	const double outputs =
		(double)shape->output_sizes[0] + shape->output_sizes[1];
	const double weights = outputs * shape->input_size;
	const double row_bytes =
		(double)shape_case->row_width * LINEAR_BFLOAT16_BYTES +
		outputs * LINEAR_BFLOAT16_BYTES;
	const double seconds = microseconds / MICROSECONDS_PER_SECOND;
	const double tflops = seconds > 0.0 ? FLOPS_PER_WEIGHT_ROW * weights *
						      rows / seconds / TERA :
					      0.0;
	const double gigabytes =
		seconds > 0.0 ? (weights + row_bytes * rows) / seconds / GIGA :
				0.0;
	const double roof = shape->is_bfloat16 ? ROOF_HALF_TFLOPS :
						 ROOF_BFLOAT16_TFLOPS;
	(void)printf(
		"    TIME %" PRIu32 " %s, %" PRIu32
		" rows: %.1f us, %.1f TFLOP/s (%.0f%% of %.0f), %.0f GB/s (%.0f%% of %.0f); launches %" PRIu32
		", flops %.6e, bytes %.6e\n",
		time_index++, what, rows, microseconds, tflops,
		PERCENT * tflops / roof, roof, gigabytes,
		PERCENT * gigabytes / ROOF_GIGABYTES_PER_SECOND,
		ROOF_GIGABYTES_PER_SECOND, launch_count,
		FLOPS_PER_WEIGHT_ROW * weights * rows,
		weights + row_bytes * rows);
}

/* The timings of a case (see the top). */
static void case_time(struct test_linear_card *card,
		      const struct shape_case *shape_case)
{
	static struct kernel_launch variant[RELAXED_MOST_LAUNCHES];
	const struct trunk_shape *const shape = shape_case->shape;
	for (uint32_t case_index = 0; case_index < CASES; case_index++) {
		const uint32_t rows = case_rows[case_index];
		uint32_t added = 0;
		if (rows > shape_case->row_count || case_index == UNEVEN_CASE) {
			continue;
		}
		time_print(card, shape_case, "reference order, thin launches",
			   rows, &launches[shape_case->thin_first],
			   (rows + GROUP_ROWS - 1u) / GROUP_ROWS);
		time_print(card, shape_case, "relaxed, as prepared", rows,
			   &launches[shape_case->relaxed_first[case_index]],
			   shape_case->relaxed_count[case_index]);
		if (case_index < FIRST_WIDE_CASE) {
			for (uint32_t index = 0;
			     index < COUNT_OF(stamped_slices); index++) {
				char what[TEXT_CAPACITY];
				if (stamped_slices[index] >
					    shape->input_size /
						    LINEAR_VALUES_PER_LOAD ||
				    !test_linear_expect_status(
					    relaxed_prepare(
						    card, shape_case, rows,
						    shape_case->relaxed_outputs
							    [case_index],
						    stamped_slices[index], 1,
						    false, variant, &added),
					    STATUS_OK,
					    "a decode variant is prepared")) {
					continue;
				}
				(void)snprintf(what, sizeof(what),
					       "relaxed decode form, %" PRIu32
					       " slices",
					       stamped_slices[index]);
				time_print(card, shape_case, what, rows,
					   variant, added);
			}
			continue;
		}
		/* the other tile of rows */
		if (test_linear_expect_status(
			    relaxed_prepare(
				    card, shape_case, rows,
				    shape_case->relaxed_outputs[case_index],
				    shape->decode_slices, 1, false, variant,
				    &added),
			    STATUS_OK, "a wide variant is prepared")) {
			const uint32_t tile_rows = RELAXED_WIDE_ROWS_LARGE;
			variant[0].kernel =
				(shape->is_bfloat16 ?
					 relaxed_kernels.wide_bfloat16 :
					 relaxed_kernels.wide_quantized)
					[RELAXED_TILE_LARGE];
			variant[0].grid[0] =
				(rows + tile_rows - 1u) / tile_rows;
			variant[0].block[0] = tile_rows /
					      RELAXED_WIDE_WARP_ROWS *
					      RELAXED_WIDE_WARPS_ALONG_OUTPUTS *
					      LINEAR_THREADS_PER_WARP;
			variant[0].dynamic_shared_byte_count =
				RELAXED_WIDE_SHARED_BYTES(tile_rows,
							  shape->input_size);
			time_print(card, shape_case,
				   "relaxed, tiles of 128 rows", rows, variant,
				   added);
		}
		for (uint32_t splits = SPLIT_TWO; splits <= SPLIT_FOUR;
		     splits++) {
			char what[TEXT_CAPACITY];
			if (linear_sums_byte_count(
				    (uint64_t)splits * rows,
				    (uint64_t)shape->output_sizes[0] +
					    shape->output_sizes[1]) >
				    PARTIALS_BYTES ||
			    !test_linear_expect_status(
				    relaxed_prepare(
					    card, shape_case, rows,
					    shape_case
						    ->relaxed_outputs[case_index],
					    shape->decode_slices, splits, false,
					    variant, &added),
				    STATUS_OK, "a split variant is prepared")) {
				continue;
			}
			(void)snprintf(what, sizeof(what),
				       "relaxed, %" PRIu32
				       " splits and their reduce",
				       splits);
			time_print(card, shape_case, what, rows, variant,
				   added);
		}
		if (test_linear_expect_status(
			    relaxed_prepare(
				    card, shape_case, rows,
				    shape_case->relaxed_outputs[case_index],
				    shape->decode_slices, 1, true, variant,
				    &added),
			    STATUS_OK, "an ordered variant is prepared")) {
			time_print(card, shape_case,
				   "ordered instance (reference order)", rows,
				   variant, added);
		}
	}
}

/* Every shape, then its cancelling case, on the card. */
static void card_run(struct test_linear_card *card, bool is_timed)
{
	static struct shape_case shape_case;
	for (uint32_t pass = 0; pass < HALVES; pass++) {
		for (uint32_t shape_index = 0;
		     shape_index < COUNT_OF(trunk_shapes); shape_index++) {
			memset(&shape_case, 0, sizeof(shape_case));
			shape_case.shape = &trunk_shapes[shape_index];
			shape_case.is_cancelling = pass == 1u;
			shape_case.row_count = shape_case.is_cancelling ?
						       CANCEL_ROWS :
						       MOST_ROWS;
			test_linear_context(
				"%s%s", shape_case.shape->name,
				shape_case.is_cancelling ? ", cancelling" : "");
			if (!test_linear_expect(case_lay_out(card, &shape_case,
							     shape_index),
						"the case is laid out") ||
			    !test_linear_card_load(card) ||
			    !test_linear_card_run(card, launches,
						  shape_case.launch_count) ||
			    !test_linear_card_fetch(card)) {
				continue;
			}
			(void)test_linear_card_expect_untouched(
				card, shape_case.values,
				"the rows' values are only read");
			(void)test_linear_card_expect_weights_untouched(card);
			case_check(card, &shape_case, shape_index);
			if (is_timed && !shape_case.is_cancelling) {
				case_time(card, &shape_case);
			}
			(void)fflush(stdout);
		}
	}
}

int main(int argument_count, char **arguments)
{
	static struct test_linear_card card;
	struct card_description_record description;
	const char *bus_address = NULL;
	bool is_timed = false;
	uint32_t card_count = 0;
	for (int index = 1; index < argument_count; index++) {
		if (strncmp(arguments[index], BUS_OPTION, strlen(BUS_OPTION)) ==
		    0) {
			bus_address = arguments[index] + strlen(BUS_OPTION);
		} else if (strcmp(arguments[index], TIME_OPTION) == 0) {
			is_timed = true;
		} else {
			bus_address = NULL;
			break;
		}
	}
	if (bus_address == NULL || argument_count > MOST_ARGUMENTS) {
		(void)fprintf(
			stderr,
			"usage: trunk_relaxed_card_test --bus=<PCI address of the one visible card> [--time]\n");
		return TEST_WRONG_USAGE;
	}
	(void)alarm(TEST_TIME_LIMIT_SECONDS);
	memset(&description, 0, sizeof(description));
	if (test_linear_expect_status(backend_cuda_value.card_count(&card_count),
				      STATUS_OK, "the cards are counted") &&
	    test_linear_expect(
		    card_count == 1u,
		    "the process sees exactly one card (CUDA_VISIBLE_DEVICES)") &&
	    test_linear_expect_status(
		    backend_cuda_value.card_describe(0, &description),
		    STATUS_OK, "the card is described") &&
	    test_linear_expect(
		    strcmp(description.bus_address, bus_address) == 0 &&
			    description.arithmetic_class ==
				    ARITHMETIC_CLASS_SM_80,
		    "the card is the class 80 card at the bus address given") &&
	    test_linear_card_open(&card, 0, WORK_CAPACITY, WEIGHTS_CAPACITY) &&
	    test_linear_expect_status(
		    relaxed_linear_kernels_prepare(
			    &backend_cuda_value, card.card,
			    card.arithmetic_class, &relaxed_kernels),
		    STATUS_OK, "the relaxed kernels are prepared")) {
		(void)printf("card %s (%s): arena %" PRIu64 " MiB\n",
			     description.name, description.bus_address,
			     card.arena_byte_count >> MEBIBYTE_SHIFT);
		card_run(&card, is_timed);
		(void)printf("digest of everything it computed 0x%016" PRIX64
			     "\n",
			     card.digest);
	}
	test_linear_card_close(&card);
	if (test_linear_failures() != 0) {
		(void)fprintf(stderr,
			      "trunk_relaxed_card_test FAILED: %d failures\n",
			      test_linear_failures());
		return TEST_FAILED;
	}
	(void)printf("trunk_relaxed_card_test passed\n");
	return TEST_PASSED;
}
