/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * attention_output_card_test: on the real cards, the attention site's output span in two launches
 * (kernel_attention_output.cu) against today's four (the output groups of kernel_trunk_linear.cu, the quantizer of
 * kernel_activation_quantize.cu, the output projection of kernel_trunk_linear.cu, the post of
 * kernel_hyper_connection.cu), at the model's shapes (attn.wo_a 8,192 by 4,096 in 8 groups, attn.wo_b 5,120 by 8,192)
 * on synthetic numbers (linear_card_fixture.h).
 *
 * It needs room on the card for its arena (a card without it FAILS the test, it is never skipped). On every card, in
 * each arithmetic class the card runs (class 80; on class 120 its native arithmetic, the PRO's default, and its
 * emulation of class 80), with the slice counts linear_default_slice_count gives that class, the card opened
 * afresh for each class:
 *   - For 1 to 8 rows: the same inputs in two sets of arrays, today's chain run on one and the two launches on the
 *     other, the arrays each writes starting as a filler byte of its set's own (a byte one form leaves unwritten
 *     shows); every byte of the case's rows of the groups, their codes, exponents and operands and the projection's
 *     output must be equal, each set's rows beyond them untouched, every byte of the four streams equal, and the post
 *     must have changed the streams. The rows of the
 *     attention's output are of every kind the fixture makes (plain, small, edges, extremes: an infinity reaches some
 *     outputs, and the not-a-numbers it makes run through the quantizer, the projection and the post).
 *   - Negative controls, each of which must make the comparison fail: the sign of one value of the attention's
 *     output flipped in the second set (the groups and the streams must differ), and one comb coefficient negated
 *     there (the streams must differ, the groups must not).
 *   - On the classes whose bits do not depend on the slice count (class 80 and class 120's emulation): the fused
 *     launches at every count their kernels run there, each held to today's bytes at 1 to 8 rows.
 *   - The guards behind every range, the ranges a launch only reads, and the card's digest of the matrices.
 * With --time (for a release build, the cards' clocks locked), at 1 and 8 rows: each form of a part captured as one
 * graph for each of RING_COPIES copies of the matrices, so that a run reads weights that the RING_COPIES - 1 runs
 * before it did not (cold, as in a decode step, the activations hot); PAIR lines sample today's launches and the fused
 * ones in pairs on the card's clock (card_timing.h), TIMED_SAMPLES samples of one run or of BACK_TO_BACK_RUNS runs
 * queued back to back after WARM_RUNS runs, for the span (four launches against two), the groups (wo_a and the
 * quantizer against one) and the projection (wo_b and the post against one): both forms' 10th, 50th and 90th
 * percentiles and those of the saving, today's sample less the fused one beside it.
 */
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "anemone/attention_output.h"
#include "anemone/backend_seam.h"
#include "anemone/hyper_connection.h"
#include "anemone/linear.h"
#include "anemone/model_layout.h"
#include "anemone/status.h"
#include "card_timing.h"
#include "linear_card_fixture.h"

#define TEST_TIME_LIMIT_SECONDS \
	1800 /* the alarm ends the test here, whatever the driver does */
#define TIME_SWITCH "--time"
#define TIMED_ARGUMENTS 2 /* the program's name and the switch */
#define WORK_CAPACITY \
	(8ull << 20u) /* two sets of arrays of 8 rows, their guards and room */
/* attn.wo_a, 33,587,200 bytes, and attn.wo_b, 41,984,000, once; timed, RING_COPIES times: 302 MB, beyond the L2 of
 * either card (40 MB on a CMP 170HX, 128 MB on the PRO), so that a timed run reads its weights cold, as a decode step
 * does, and its activations hot */
#define WEIGHTS_CAPACITY (80ull << 20u)
#define RING_COPIES 4u
#define COUNT_OF(table) ((uint32_t)(sizeof(table) / sizeof((table)[0])))
#define ROWS LINEAR_MAXIMUM_ROWS
#define STREAMS MODEL_STREAMS
#define HIDDEN MODEL_HIDDEN
#define GROUPS_OUTPUTS 8192u /* attn.wo_a: 8 groups of 1,024 outputs */
#define GROUP_INPUTS 4096u /* of a group: 8 heads of 512 values */
#define GROUP_COUNT MODEL_OUTPUT_GROUPS
/* a row of the attention's output */
#define HEADS_WIDTH ((uint32_t)(GROUP_COUNT * GROUP_INPUTS))
#define E4M3 WEIGHT_FORMAT_E4M3_SCALED
#define E4M3_BITS_PER_WEIGHT 8u
#define GROUPS_SCALE_CODE_BASE \
	115u /* the scales of attn.wo_a: 2^-12 to 2^-10 */
#define PROJECTION_SCALE_CODE_BASE 112u /* of attn.wo_b: 2^-15 to 2^-13 */
#define SCALE_CODE_SPREAD 2u
#define GROUPS_STREAM \
	0x1A000000ull /* the streams of the hashes: one for each use, far apart */
#define PROJECTION_STREAM 0x1B000000ull
#define VALUES_STREAM 0x1C000000ull
#define RESIDUAL_STREAM 0x1D000000ull
#define POST_STREAM 0x1E000000ull
#define COMB_STREAM 0x1F000000ull
#define POST_STEPS 1024u /* post: (1 + hash % 1,024) / 512, from 2^-9 to 2 */
#define POST_DIVISOR 512.0f
#define COMB_STEPS 4096u /* comb: (1 + hash % 4,096) / 4,096, from 2^-12 to 1 */
#define COMB_DIVISOR 4096.0f
#define BFLOAT16_SIGN_BYTE \
	1u /* the byte of a bfloat16 that holds its sign, and the bit */
#define BFLOAT16_SIGN_BYTE_BIT 0x80u
#define FLOAT32_SIGN_BYTE \
	3u /* the byte of a float32 that holds its sign, and the bit */
#define FLOAT32_SIGN_BYTE_BIT 0x80u
#define WARM_RUNS 20u
#define TIMED_SAMPLES 201u
#define BACK_TO_BACK_RUNS 20u
#define TODAY_FILLER 0x3Cu
#define FUSED_FILLER 0x5Du

/* The arrays of one set: today's or the fused launches'. The site's own (carried pre, mixes, scale, base, norm weight,
 * its input) are only for the post's description: the post reads none of them. */
enum set_array {
	VALUES, /* read: the attention's output */
	GROUP_VALUES, /* written: the groups' outputs */
	CODES,
	EXPONENTS,
	OPERANDS,
	OUTPUT, /* written: the projection's output, the sublayer's */
	RESIDUALS, /* read and written: the streams */
	POST,
	COMB,
	CARRIED_PRE,
	MIXES,
	SITE_INPUT,
	SCALE,
	BASE,
	NORM_WEIGHT,
	SET_ARRAYS
};
enum set_kind { TODAY, FUSED, SETS };
enum today_launch {
	TODAY_GROUPS,
	TODAY_QUANTIZE,
	TODAY_PROJECTION,
	TODAY_POST,
	TODAY_LAUNCHES
};
enum fused_launch { FUSED_GROUPS, FUSED_PROJECTION, FUSED_LAUNCHES };
enum matrix_kind { GROUPS_MATRIX, PROJECTION_MATRIX, MATRICES };

static const struct backend_operations *const backend = &backend_cuda_value;
static const char *const set_names[SETS] = { "today's chain",
					     "the fused launches" };
/* What each set's written arrays hold before a run, a byte of its own: a byte one form leaves unwritten shows. */
static const unsigned char set_fillers[SETS] = { TODAY_FILLER, FUSED_FILLER };
/* The arrays the launches write row by row (the streams are read too, from inputs both sets share). */
static const enum set_array written_arrays[] = { GROUP_VALUES, CODES, EXPONENTS,
						 OPERANDS, OUTPUT };
static const uint32_t timed_rows[] = { 1, ROWS };
static const uint32_t swept_slices[] = {
	1, 2, 4, 8, 16
}; /* the stamped counts, ascending */
/* The kind of each row's attention output (linear_card_fixture.h): rows that are not quantized. */
static const enum test_linear_values_kind row_kinds[ROWS] = {
	TEST_LINEAR_VALUES_PLAIN, TEST_LINEAR_VALUES_EDGES,
	TEST_LINEAR_VALUES_SMALL, TEST_LINEAR_VALUES_PLAIN,
	TEST_LINEAR_VALUES_SMALL, TEST_LINEAR_VALUES_EXTREMES,
	TEST_LINEAR_VALUES_PLAIN, TEST_LINEAR_VALUES_EDGES
};

/* What every case of a class shares: the card, its kernels, the matrices and both sets' ranges. */
struct span_case {
	struct test_linear_card card;
	uint32_t arithmetic_class;
	struct linear_kernels_record linear;
	struct hyper_kernels_record hyper;
	struct attention_output_kernels fused;
	struct linear_matrix matrices[MATRICES];
	/* the matrices as the fused launches name them: their slices */
	struct linear_matrix fused_matrices[MATRICES];
	/* the copies of the matrices laid out (copy 0: the matrices' own), their count */
	struct linear_range ring[RING_COPIES][MATRICES];
	uint32_t ring_count;
	struct linear_range arrays[SETS][SET_ARRAYS];
	struct linear_range
		work; /* the work region set aside: every array of both sets */
};

/* The bytes of each array for ROWS rows. */
static uint64_t array_byte_count(enum set_array array)
{
	const uint64_t rows = ROWS;
	switch (array) {
	case VALUES:
		return linear_values_byte_count(rows, HEADS_WIDTH);
	case GROUP_VALUES:
	case OPERANDS:
		return linear_values_byte_count(rows, GROUPS_OUTPUTS);
	case CODES:
		return linear_codes_byte_count(rows, GROUPS_OUTPUTS);
	case EXPONENTS:
		return linear_exponents_byte_count(rows, GROUPS_OUTPUTS);
	case OUTPUT:
	case SITE_INPUT:
		return rows * MODEL_HIDDEN_ROW_BYTES;
	case RESIDUALS:
		return rows * MODEL_STREAMS_ROW_BYTES;
	case POST:
	case CARRIED_PRE:
		return rows * MODEL_SITE_ROW_BYTES;
	case COMB:
		return rows * MODEL_COMB_ROW_BYTES;
	case MIXES:
		return rows * MODEL_MIXES_ROW_BYTES;
	case SCALE:
		return (uint64_t)HYPER_SCALES * sizeof(float);
	case BASE:
		return MODEL_MIXES_ROW_BYTES;
	case NORM_WEIGHT:
		return MODEL_HIDDEN_ROW_BYTES;
	case SET_ARRAYS:
		break;
	}
	return 0;
}

/* The two matrices in the weights region and both sets' arrays, the inputs of the second set the first's: the
 * attention's output of row_kinds, plain streams, post and comb from the hash. False behind a failure line. */
static bool span_lay_out(struct span_case *span)
{
	static const struct matrix_shape_record shapes[MATRICES] = {
		[GROUPS_MATRIX] = { GROUPS_OUTPUTS, GROUP_INPUTS,
				    E4M3_BITS_PER_WEIGHT, 0 },
		[PROJECTION_MATRIX] = { HIDDEN, GROUPS_OUTPUTS,
					E4M3_BITS_PER_WEIGHT, 0 }
	};
	static const uint32_t scale_bases[MATRICES] = {
		GROUPS_SCALE_CODE_BASE, PROJECTION_SCALE_CODE_BASE
	};
	static const uint64_t matrix_streams[MATRICES] = { GROUPS_STREAM,
							   PROJECTION_STREAM };
	static const enum linear_family_code families[MATRICES] = {
		LINEAR_FAMILY_TRUNK_BFLOAT16, LINEAR_FAMILY_TRUNK_QUANTIZED
	};
	struct test_linear_card *const card = &span->card;
	bool is_laid_out = true;
	test_linear_card_clear(card);
	for (uint32_t matrix = 0; matrix < MATRICES && is_laid_out; matrix++) {
		const struct test_linear_matrix_recipe recipe = {
			.format = E4M3,
			.shape = shapes[matrix],
			.stream = matrix_streams[matrix],
			.code = test_linear_e4m3_code,
			.scale_code_base = scale_bases[matrix],
			.scale_code_spread = SCALE_CODE_SPREAD
		};
		span->matrices[matrix] = (struct linear_matrix){
			{ 0, 0 }, shapes[matrix],	    E4M3,
			0,	  LINEAR_WIDE_GROUP_BLOCKS, 0
		};
		is_laid_out =
			test_linear_card_matrix(
				card, &recipe,
				&span->matrices[matrix].stored) &&
			test_linear_expect_status(
				linear_default_slice_count(
					LINEAR_PROFILE_SPEED,
					LINEAR_ORDER_FIXED, families[matrix],
					&shapes[matrix], span->arithmetic_class,
					&span->matrices[matrix].slice_count),
				STATUS_OK,
				"the class has a slice count for the matrix");
	}
	memcpy(span->fused_matrices, span->matrices, sizeof(span->matrices));
	for (uint32_t copy = 0; copy < span->ring_count && is_laid_out;
	     copy++) {
		for (uint32_t matrix = 0; matrix < MATRICES && is_laid_out;
		     matrix++) {
			const struct linear_range stored =
				span->matrices[matrix].stored;
			span->ring[copy][matrix] =
				copy == 0 ? stored :
					    test_linear_card_weights(
						    card, stored.byte_count);
			unsigned char *const bytes = test_linear_card_writable(
				card, span->ring[copy][matrix]);
			is_laid_out = test_linear_expect(
				bytes != NULL, "a copy of a matrix has room");
			if (bytes != NULL && copy != 0) {
				memcpy(bytes,
				       test_linear_card_stored(card, stored),
				       stored.byte_count);
			}
		}
	}
	for (uint32_t set = 0; set < SETS && is_laid_out; set++) {
		for (uint32_t array = 0; array < SET_ARRAYS; array++) {
			span->arrays[set][array] = test_linear_card_work(
				card, array_byte_count((enum set_array)array));
		}
	}
	if (!is_laid_out || !test_linear_card_has_room(card)) {
		return false;
	}
	for (uint32_t set = 0; set < SETS; set++) {
		for (uint32_t index = 0; index < COUNT_OF(written_arrays);
		     index++) {
			const struct linear_range written =
				span->arrays[set][written_arrays[index]];
			memset(card->image + written.byte_offset,
			       set_fillers[set], written.byte_count);
		}
	}
	span->work = (struct linear_range){ 0, card->work_byte_count };
	_Static_assert(HEADS_WIDTH >= STREAMS * HIDDEN,
		       "a row of the attention's output is the wider row");
	uint16_t *const row_values =
		malloc(linear_values_byte_count(1, HEADS_WIDTH));
	if (!test_linear_expect(row_values != NULL,
				"the host has memory for a row of values")) {
		return false;
	}
	for (uint32_t row = 0; row < ROWS && is_laid_out; row++) {
		float post[STREAMS], comb[STREAMS * STREAMS];
		for (uint32_t stream = 0; stream < STREAMS; stream++) {
			post[stream] =
				(float)(1u + test_linear_hash(POST_STREAM + row,
							      stream) %
						     POST_STEPS) /
				POST_DIVISOR;
		}
		for (uint32_t coefficient = 0; coefficient < STREAMS * STREAMS;
		     coefficient++) {
			comb[coefficient] =
				(float)(1u + test_linear_hash(COMB_STREAM + row,
							      coefficient) %
						     COMB_STEPS) /
				COMB_DIVISOR;
		}
		for (uint32_t set = 0; set < SETS && is_laid_out; set++) {
			const struct linear_range *const arrays =
				span->arrays[set];
			test_linear_values(row_values, HEADS_WIDTH,
					   VALUES_STREAM + row, row_kinds[row]);
			is_laid_out = test_linear_card_put(
				card, arrays[VALUES],
				linear_values_byte_count(row, HEADS_WIDTH),
				row_values,
				linear_values_byte_count(1, HEADS_WIDTH));
			test_linear_values(row_values, STREAMS * HIDDEN,
					   RESIDUAL_STREAM + row,
					   TEST_LINEAR_VALUES_PLAIN);
			is_laid_out =
				is_laid_out &&
				test_linear_card_put(
					card, arrays[RESIDUALS],
					(uint64_t)row * MODEL_STREAMS_ROW_BYTES,
					row_values, MODEL_STREAMS_ROW_BYTES) &&
				test_linear_card_put(
					card, arrays[POST],
					(uint64_t)row * MODEL_SITE_ROW_BYTES,
					post, sizeof(post)) &&
				test_linear_card_put(
					card, arrays[COMB],
					(uint64_t)row * MODEL_COMB_ROW_BYTES,
					comb, sizeof(comb));
		}
	}
	free(row_values);
	return is_laid_out && test_linear_card_load(card);
}

/* Today's four launches and the two fused ones for row_count rows. False behind a failure line. */
static bool span_prepare(struct span_case *span, uint32_t row_count,
			 uint32_t copy,
			 struct kernel_launch today[TODAY_LAUNCHES],
			 struct kernel_launch fused[FUSED_LAUNCHES])
{
	struct linear_matrix today_matrices[MATRICES], fused_matrices[MATRICES];
	for (uint32_t matrix = 0; matrix < MATRICES; matrix++) {
		today_matrices[matrix] = span->matrices[matrix];
		fused_matrices[matrix] = span->fused_matrices[matrix];
		today_matrices[matrix].stored = span->ring[copy][matrix];
		fused_matrices[matrix].stored = span->ring[copy][matrix];
	}
	const uint64_t arena = span->card.arena_byte_count;
	const struct linear_range *const today_arrays = span->arrays[TODAY];
	const struct linear_range *const fused_arrays = span->arrays[FUSED];
	const struct linear_range none = { 0, 0 };
	struct kernel_launch site[HYPER_SITE_LAUNCH_COUNT];
	const struct linear_trunk_bfloat16_description groups = {
		today_matrices[GROUPS_MATRIX], today_arrays[VALUES],
		today_arrays[GROUP_VALUES], row_count, GROUP_COUNT
	};
	const struct linear_quantize_description quantize = {
		today_arrays[GROUP_VALUES],
		today_arrays[CODES],
		today_arrays[EXPONENTS],
		today_arrays[OPERANDS],
		GROUPS_OUTPUTS,
		row_count
	};
	const struct linear_trunk_quantized_description projection = {
		.matrices = { today_matrices[PROJECTION_MATRIX] },
		.codes = today_arrays[CODES],
		.exponents = today_arrays[EXPONENTS],
		.operands = today_arrays[OPERANDS],
		.outputs = { today_arrays[OUTPUT] },
		.matrix_count = 1,
		.row_count = row_count
	};
	const struct hyper_site_description post = { today_arrays[RESIDUALS],
						     today_arrays[CARRIED_PRE],
						     today_arrays[MIXES],
						     today_arrays[SCALE],
						     today_arrays[BASE],
						     today_arrays[NORM_WEIGHT],
						     today_arrays[POST],
						     today_arrays[COMB],
						     today_arrays[SITE_INPUT],
						     today_arrays[OUTPUT],
						     none,
						     row_count,
						     0 };
	const struct attention_groups_description fused_groups = {
		fused_matrices[GROUPS_MATRIX],
		fused_arrays[VALUES],
		fused_arrays[GROUP_VALUES],
		fused_arrays[CODES],
		fused_arrays[EXPONENTS],
		fused_arrays[OPERANDS],
		row_count,
		GROUP_COUNT
	};
	const struct attention_projection_description fused_projection = {
		fused_matrices[PROJECTION_MATRIX],
		fused_arrays[CODES],
		fused_arrays[EXPONENTS],
		fused_arrays[OPERANDS],
		fused_arrays[OUTPUT],
		fused_arrays[RESIDUALS],
		fused_arrays[POST],
		fused_arrays[COMB],
		row_count,
		0
	};
	const bool is_prepared =
		test_linear_expect_status(linear_trunk_bfloat16_prepare(
						  &groups, arena, &span->linear,
						  &today[TODAY_GROUPS]),
					  STATUS_OK,
					  "today's groups are prepared") &&
		test_linear_expect_status(
			linear_quantize_prepare(&quantize, arena, &span->linear,
						&today[TODAY_QUANTIZE]),
			STATUS_OK, "today's quantization is prepared") &&
		test_linear_expect_status(
			linear_trunk_quantized_prepare(
				&projection, arena, &span->linear,
				&today[TODAY_PROJECTION]),
			STATUS_OK, "today's projection is prepared") &&
		test_linear_expect_status(
			hyper_site_prepare(&post, arena, &span->hyper, site),
			STATUS_OK, "today's site is prepared") &&
		test_linear_expect_status(
			attention_groups_prepare(&fused_groups, arena,
						 &span->linear, &span->fused,
						 &fused[FUSED_GROUPS]),
			STATUS_OK, "the fused groups are prepared") &&
		test_linear_expect_status(
			attention_projection_prepare(
				&fused_projection, arena, &span->linear,
				&span->fused, &fused[FUSED_PROJECTION]),
			STATUS_OK, "the fused projection is prepared");
	today[TODAY_POST] = site[HYPER_SITE_POST];
	return is_prepared;
}

/* The bytes of a range of the work region that came back other than the image holds them. */
static uint64_t changed_byte_count(const struct span_case *span,
				   struct linear_range range)
{
	uint64_t changed_count = 0;
	for (uint64_t byte_index = 0; byte_index < range.byte_count;
	     byte_index++) {
		changed_count +=
			span->card.image[range.byte_offset + byte_index] !=
					span->card.received[range.byte_offset +
							    byte_index] ?
				1u :
				0u;
	}
	return changed_count;
}

/* Both forms on fresh arrays (the work region uploaded again), and back. False behind a failure line. */
static bool span_run(struct span_case *span, uint32_t row_count)
{
	struct kernel_launch today[TODAY_LAUNCHES], fused[FUSED_LAUNCHES];
	return span_prepare(span, row_count, 0, today, fused) &&
	       test_linear_card_upload(&span->card, span->work, 0,
				       span->work.byte_count) &&
	       test_linear_card_run(&span->card, today, TODAY_LAUNCHES) &&
	       test_linear_card_run(&span->card, fused, FUSED_LAUNCHES) &&
	       test_linear_card_fetch(&span->card);
}

/* The bytes in which the two sets' copies of an array differ, every row of it. */
static uint64_t sets_differing(const struct span_case *span,
			       enum set_array array)
{
	uint64_t first_differing = TEST_LINEAR_NOTHING_DIFFERS;
	return test_linear_card_differing(&span->card,
					  span->arrays[TODAY][array], 0,
					  span->arrays[FUSED][array], 0,
					  span->arrays[TODAY][array].byte_count,
					  &first_differing);
}

/* Every written array equal in both sets over the rows of the case (each set's filler a byte of its own: a byte one
 * form leaves unwritten shows), each set's rows beyond the case untouched, the streams equal, and what the launches
 * only read untouched. */
static void span_expect_equal(const struct span_case *span, uint32_t row_count)
{
	static const char *const written_names[] = {
		"the groups", "their codes", "their exponents",
		"their operands", "the projection's output"
	};
	static const enum set_array read_only[] = { VALUES, POST, COMB };
	_Static_assert(COUNT_OF(written_arrays) == COUNT_OF(written_names),
		       "a name for each array compared");
	for (uint32_t index = 0; index < COUNT_OF(written_arrays); index++) {
		const enum set_array array = written_arrays[index];
		const uint64_t row_bytes = array_byte_count(array) / ROWS,
			       case_bytes = row_bytes * row_count;
		(void)test_linear_card_expect_same(
			&span->card, span->arrays[TODAY][array], 0,
			span->arrays[FUSED][array], 0, case_bytes,
			written_names[index]);
		for (uint32_t set = 0; set < SETS && row_count < ROWS; set++) {
			const struct linear_range whole =
				span->arrays[set][array];
			(void)test_linear_card_expect_untouched(
				&span->card,
				(struct linear_range){
					whole.byte_offset + case_bytes,
					whole.byte_count - case_bytes },
				"the rows beyond the case");
		}
	}
	(void)test_linear_card_expect_same(
		&span->card, span->arrays[TODAY][RESIDUALS], 0,
		span->arrays[FUSED][RESIDUALS], 0,
		span->arrays[TODAY][RESIDUALS].byte_count, "the streams");
	for (uint32_t set = 0; set < SETS; set++) {
		for (uint32_t index = 0; index < COUNT_OF(read_only); index++) {
			(void)test_linear_card_expect_untouched(
				&span->card,
				span->arrays[set][read_only[index]],
				set_names[set]);
		}
	}
	const struct linear_range streams = span->arrays[TODAY][RESIDUALS];
	(void)test_linear_expect(
		changed_byte_count(span,
				   (struct linear_range){
					   streams.byte_offset,
					   (uint64_t)row_count *
						   MODEL_STREAMS_ROW_BYTES }) !=
			0,
		"the post changed the streams");
}

/* One byte of the second set's image xored with flip, both forms run on ROWS rows, the byte put back: the
 * comparison must find the arrays of must_differ different and those of must_agree equal. */
static void negative_control(struct span_case *span, const char *what,
			     enum set_array array, uint64_t byte_offset,
			     unsigned char flip, enum set_array must_differ,
			     enum set_array must_agree)
{
	unsigned char *const changed = span->card.image +
				       span->arrays[FUSED][array].byte_offset +
				       byte_offset;
	test_linear_context("card %" PRIu32 ", class %" PRIu32
			    ", negative control: %s",
			    span->card.card_index, span->arithmetic_class,
			    what);
	*changed ^= flip;
	const bool is_run = span_run(span, ROWS);
	*changed ^= flip;
	if (is_run) {
		const uint64_t differing = sets_differing(span, must_differ),
			       agreeing = sets_differing(span, must_agree);
		(void)test_linear_expect(differing != 0,
					 "the comparison fails, as it must");
		(void)test_linear_expect(
			agreeing == 0,
			"what the change does not reach stays equal");
		(void)printf("card %" PRIu32 " class %" PRIu32
			     ": negative control, %s: %" PRIu64
			     " bytes differ where they must, %" PRIu64
			     " where they must not\n",
			     span->card.card_index, span->arithmetic_class,
			     what, differing, agreeing);
	}
}

/* ---- timing: pairs of samples on the card's clock (card_timing.h) ---- */

/* A form of a part as one graph per copy of the matrices: run r of a sample takes copy (first + r) % ring_count, so
 * no run reads weights that one of the ring_count - 1 runs before it read. */
struct ring_graphs {
	uint32_t graphs[RING_COPIES];
};

/* One PAIR line: today's launches of a part and the fused ones sampled in turn after WARM_RUNS runs of each,
 * TIMED_SAMPLES pairs of runs runs each: both forms' percentiles and those of the saving. */
static void pair_time(struct span_case *span,
		      const struct ring_graphs forms[SETS], uint32_t row_count,
		      const char *part, uint32_t runs)
{
	static double samples[SETS * TIMED_SAMPLES], savings[TIMED_SAMPLES];
	const struct test_card_timing_form timed[SETS] = {
		{ forms[TODAY].graphs, span->ring_count },
		{ forms[FUSED].graphs, span->ring_count }
	};
	if (test_card_timing_pair(span->card.card, timed, WARM_RUNS, runs,
				  TIMED_SAMPLES, samples, savings)) {
		(void)printf(
			"PAIR card=%" PRIu32 " class=%" PRIu32 " rows=%" PRIu32
			" part=%s runs=%" PRIu32 " fused_slices=%" PRIu32
			",%" PRIu32 " copies=%" PRIu32,
			span->card.card_index, span->arithmetic_class,
			row_count, part, runs,
			span->fused_matrices[GROUPS_MATRIX].slice_count,
			span->fused_matrices[PROJECTION_MATRIX].slice_count,
			span->ring_count);
		test_card_timing_print(samples, TIMED_SAMPLES, 1.0, "today");
		test_card_timing_print(samples + TIMED_SAMPLES, TIMED_SAMPLES,
				       1.0, "fused");
		test_card_timing_print(savings, TIMED_SAMPLES, 1.0, "saving");
		(void)printf("\n");
	}
}

/* The parts PAIR lines compare: today's launches from first_today, today_count of them, against the fused ones from
 * first_fused, fused_count of them. */
static const struct timed_part {
	const char *name;
	uint32_t first_today, today_count, first_fused, fused_count;
} timed_parts[] = {
	{ "span", TODAY_GROUPS, TODAY_LAUNCHES, FUSED_GROUPS, FUSED_LAUNCHES },
	{ "groups", TODAY_GROUPS, TODAY_PROJECTION - TODAY_GROUPS, FUSED_GROUPS,
	  1 },
	{ "projection", TODAY_PROJECTION, TODAY_LAUNCHES - TODAY_PROJECTION,
	  FUSED_PROJECTION, 1 },
};

/* PAIR lines of every part at 1 and 8 rows, one run and BACK_TO_BACK_RUNS runs a sample. */
static void span_time(struct span_case *span)
{
	for (uint32_t index = 0; index < COUNT_OF(timed_rows); index++) {
		struct kernel_launch launches[RING_COPIES][SETS][TODAY_LAUNCHES];
		const uint32_t row_count = timed_rows[index];
		bool is_prepared = true;
		test_linear_context("card %" PRIu32 ", class %" PRIu32
				    ", timing %" PRIu32 " rows",
				    span->card.card_index,
				    span->arithmetic_class, row_count);
		for (uint32_t copy = 0; copy < span->ring_count && is_prepared;
		     copy++) {
			is_prepared = span_prepare(span, row_count, copy,
						   launches[copy][TODAY],
						   launches[copy][FUSED]);
		}
		if (!is_prepared ||
		    !test_linear_card_upload(&span->card, span->work, 0,
					     span->work.byte_count)) {
			return;
		}
		for (uint32_t part = 0; part < COUNT_OF(timed_parts); part++) {
			const struct timed_part *const of = &timed_parts[part];
			const uint32_t firsts[SETS] = { of->first_today,
							of->first_fused };
			const uint32_t counts[SETS] = { of->today_count,
							of->fused_count };
			struct ring_graphs forms[SETS];
			bool is_captured = true;
			for (uint32_t set = 0; set < SETS && is_captured;
			     set++) {
				for (uint32_t copy = 0;
				     copy < span->ring_count && is_captured;
				     copy++) {
					is_captured = test_linear_expect_status(
						backend->graph_capture(
							span->card.card,
							&launches[copy][set]
								 [firsts[set]],
							counts[set],
							&forms[set]
								 .graphs[copy]),
						STATUS_OK,
						"the launches are captured as a graph");
				}
			}
			if (is_captured) {
				pair_time(span, forms, row_count, of->name, 1);
				pair_time(span, forms, row_count, of->name,
					  BACK_TO_BACK_RUNS);
			}
		}
	}
}

/* Rows 1 to 8, each byte of both forms held equal; false where one differs. */
static bool rows_test(struct span_case *span)
{
	const int failures_before = test_linear_failures();
	for (uint32_t row_count = 1; row_count <= ROWS; row_count++) {
		test_linear_context(
			"card %" PRIu32 ", class %" PRIu32
			", fused slices %" PRIu32 " and %" PRIu32 ", %" PRIu32
			" rows",
			span->card.card_index, span->arithmetic_class,
			span->fused_matrices[GROUPS_MATRIX].slice_count,
			span->fused_matrices[PROJECTION_MATRIX].slice_count,
			row_count);
		if (span_run(span, row_count)) {
			span_expect_equal(span, row_count);
		}
	}
	return test_linear_failures() == failures_before;
}

/* On the classes whose bits do not depend on the slice count (class 80 and class 120's emulation, the exact profile):
 * the fused launches at each count their kernels run there, held to today's bits at 1 to 8 rows. The matrices' own
 * counts afterwards. */
static void slices_sweep(struct span_case *span)
{
	for (uint32_t index = 0; index < COUNT_OF(swept_slices); index++) {
		const uint32_t slices = swept_slices[index];
		if (span->arithmetic_class == ARITHMETIC_CLASS_SM_120 &&
		    slices > LINEAR_CLASS_120_EXACT_MOST_SLICES) {
			continue;
		}
		span->fused_matrices[GROUPS_MATRIX].slice_count = slices;
		span->fused_matrices[PROJECTION_MATRIX].slice_count = slices;
		(void)printf(
			"card %" PRIu32 " class %" PRIu32
			": fused launches in %" PRIu32
			" slices, every byte equal to today's at 1 to %u rows: %s\n",
			span->card.card_index, span->arithmetic_class, slices,
			ROWS, rows_test(span) ? "yes" : "NO");
	}
	memcpy(span->fused_matrices, span->matrices, sizeof(span->matrices));
}

/* One class on a card opened for it alone (a card prepares at most BACKEND_MAXIMUM_KERNELS kernels). */
static void class_test(uint32_t card_index, uint32_t arithmetic_class,
		       bool is_timed)
{
	static struct span_case span;
	memset(&span, 0, sizeof(span));
	span.arithmetic_class = arithmetic_class;
	span.ring_count = is_timed ? RING_COPIES : 1u;
	if (!test_linear_card_open(&span.card, card_index, WORK_CAPACITY,
				   span.ring_count * WEIGHTS_CAPACITY)) {
		return;
	}
	test_linear_context("card %" PRIu32 ", class %" PRIu32, card_index,
			    arithmetic_class);
	if (test_linear_expect_status(
		    linear_kernels_prepare(backend, span.card.card,
					   arithmetic_class, &span.linear),
		    STATUS_OK,
		    "the linear's kernels of the class are prepared") &&
	    test_linear_expect_status(
		    hyper_kernels_prepare(backend, span.card.card, &span.hyper),
		    STATUS_OK, "the hyper-connection kernels are prepared") &&
	    test_linear_expect_status(
		    attention_output_kernels_prepare(backend, span.card.card,
						     arithmetic_class,
						     &span.fused),
		    STATUS_OK, "the fused kernels of the class are prepared") &&
	    span_lay_out(&span)) {
		(void)printf("card %" PRIu32 " class %" PRIu32
			     ": wo_a in %" PRIu32 " slices, wo_b in %" PRIu32
			     " slices\n",
			     card_index, arithmetic_class,
			     span.matrices[GROUPS_MATRIX].slice_count,
			     span.matrices[PROJECTION_MATRIX].slice_count);
		const int failures_before = test_linear_failures();
		(void)rows_test(&span);
		negative_control(&span,
				 "the sign of the first value of row 0 flipped",
				 VALUES, BFLOAT16_SIGN_BYTE,
				 BFLOAT16_SIGN_BYTE_BIT, GROUP_VALUES, COMB);
		negative_control(&span,
				 "comb's first coefficient of row 0 negated",
				 COMB, FLOAT32_SIGN_BYTE, FLOAT32_SIGN_BYTE_BIT,
				 RESIDUALS, GROUP_VALUES);
		test_linear_context("card %" PRIu32 ", class %" PRIu32,
				    card_index, arithmetic_class);
		(void)test_linear_card_expect_weights_untouched(&span.card);
		(void)printf(
			"card %" PRIu32 " class %" PRIu32
			": 1 to %u rows, every byte of both forms equal and "
			"both controls caught: %s; digest 0x%016" PRIx64 "\n",
			card_index, arithmetic_class, ROWS,
			test_linear_failures() == failures_before ? "yes" :
								    "NO",
			span.card.digest);
		if (arithmetic_class != ARITHMETIC_CLASS_SM_120_NATIVE) {
			slices_sweep(&span);
		}
		if (is_timed) {
			span_time(&span);
		}
	}
	test_linear_card_close(&span.card);
}

int main(int argument_count, char **arguments)
{
	const bool is_timed = argument_count == TIMED_ARGUMENTS &&
			      strcmp(arguments[1], TIME_SWITCH) == 0;
	uint32_t card_count = 0;
	(void)alarm(TEST_TIME_LIMIT_SECONDS);
	if (argument_count > TIMED_ARGUMENTS ||
	    (argument_count == TIMED_ARGUMENTS && !is_timed)) {
		(void)fprintf(stderr, "usage: %s [%s]\n", arguments[0],
			      TIME_SWITCH);
		return EXIT_FAILURE;
	}
	if (!test_linear_expect_status(backend->card_count(&card_count),
				       STATUS_OK, "the cards are counted") ||
	    !test_linear_expect(card_count != 0, "a card is there")) {
		return EXIT_FAILURE;
	}
	for (uint32_t card_index = 0; card_index < card_count; card_index++) {
		struct card_description_record description;
		if (!test_linear_expect_status(
			    backend->card_describe(card_index, &description),
			    STATUS_OK, "the card is described")) {
			continue;
		}
		if (description.arithmetic_class == ARITHMETIC_CLASS_SM_120) {
			class_test(card_index, ARITHMETIC_CLASS_SM_120_NATIVE,
				   is_timed);
		}
		class_test(card_index, description.arithmetic_class, is_timed);
	}
	(void)printf("attention_output_card_test: %d failures\n",
		     test_linear_failures());
	return test_linear_failures() == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
