/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * dense_linear_card_test: on the real cards, kernel_dense_linear.cu against a host reference in double, on synthetic
 * numbers and, given a capture of the reference and its pack, replayed on the reference's own inputs, weights and
 * outputs (first_token_plan.md 3.2 and 3.5).
 *
 *     dense_linear_card_test [--break word] [<capture folder> <pack copy> [<pack copy> ...]]
 *
 * The reference of an output and the bound it is held to are dense_linear_reference.h's: y, the sum of its products in
 * double, and 2^-20 of the sum of their magnitudes (plan 3.2).
 *
 * Synthetic, on every card it opens (all of them: a card without room FAILS the test), for the products of the model
 * at their shapes (the head's outputs cut to 8,192, and its whole grid with 64 inputs), with rows of
 * plain, small and edge values (linear_card_fixture.h) and weights of plain bfloat16 or float32 of 24 bits:
 *   - every output of a launch of 8 rows holds against the reference;
 *   - ROW EXACTNESS: each row launched alone (the function of one row) and 5 rows in another order (the function of 8
 *     at 5 rows) have the bits the launch of 8 gave them;
 *   - DENSE_STEP_ROWS: of 8 rows the step block asks 6 and flags 4: those have the bits of the launch without
 *     the mode, every other row keeps the bytes laid there; and the head's launch of M1b, one row in a step of one
 *     row: written with the bits of the row alone where its flag asks the logits, untouched where it does not;
 *   - SHOWN FAILING: one weight is changed on the card to 64 and the launch of 8 rows runs again: only that weight's
 *     output changes, it fails the reference before the change and holds against the one after;
 *   - no launch writes what is not its own: the guards behind every range, the ranges launches only read, the card's
 *     digest of the weights.
 * A card's last line is the digest of everything it computed: the kernels use no tensor-core instruction, so cards of
 * both classes may agree but for rsqrtf (printed, and said whether they do: a result, not a gate).
 *
 * Replay, with a capture folder of tools/oracle/stepwise_capture.py (record names: compare_engine_capture.py; the
 * head's logits.all_rows needs --all-logits) and the pack: for each product (replay_products), every capture point
 * that holds its input and its output, up to 8 points of one layer a launch, the weights read from the pack by the
 * reference's names. A product of which the capture holds no point FAILS. Per product and card: every output holds
 * against the host reference of the same weights and inputs; a float32 output lies within the bound of the
 * reference's own, a bfloat16 output is the reference's on 99.9% of them and one place off at most elsewhere. It
 * also counts the reference's own outputs outside the host reference (printed, not a gate): a capture whose library
 * rounded a bfloat16 product more than once (partial sums of a split reduction) shows there, not as the card's.
 * --break word changes one weight of each matrix (the one the first point's largest input meets) before its launch:
 * every product must then FAIL against the reference's outputs, and it says "reported"; the test passes only when
 * each did and nothing else failed. Without a capture it says that it compared nothing.
 */
#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "anemone/backend_seam.h"
#include "anemone/dense_linear.h"
#include "anemone/dense_linear_records.h"
#include "anemone/linear.h"
#include "anemone/model_layout.h"
#include "anemone/pack.h"
#include "anemone/status.h"
#include "anemone/step_records.h"
#include "anemone/storage_seam.h"
#include "capture_reader.h"
#include "dense_linear_reference.h"
#include "linear_card_fixture.h"
#include "linear_reference.h"

#define COUNT_OF(table) (sizeof(table) / sizeof((table)[0]))
#define BFLOAT16 DENSE_BFLOAT16
#define FLOAT32 DENSE_FLOAT32
#define ROWS LINEAR_MAXIMUM_ROWS
#define SOME_ROWS 5u                            /* the launch of the function of 8 at fewer rows */
#define ASKED_ROWS 6u /* the step's row_count in the case of DENSE_STEP_ROWS */
#define TEST_TIME_LIMIT_SECONDS 7200            /* the alarm ends the test here, whatever a drive or a driver does */
#define WORK_CAPACITY (32ull << 20u)            /* the arrays of a case: 8 rows of the head's outputs, 4.25 times */
#define WEIGHTS_CAPACITY (1408ull << 20u)       /* the head's 1.233 GiB, read in whole blocks */
#define CHANGED_WEIGHT 64.0                     /* what the shown failing and the break set a weight to */
#define CHANGED_OUTPUT 5u                       /* the output whose weight the shown failing changes */
#define CHANGED_INPUT 3u
#define STREAM_OF_WEIGHTS 1000003u              /* hash streams of a case: weights, rows, and the case's own */
#define STREAM_OF_ROWS 2000003u
#define STREAM_OF_CASES 7919u
#define FLOAT32_SIGN_SHIFT 31u
#define FLOAT32_EXPONENT_SHIFT 23u
#define FLOAT32_FRACTION_MASK 0x7FFFFFu
#define FLOAT32_SMALLEST_EXPONENT_FIELD 119u    /* weights of float32 from 2^-8 to below 2: 9 binades */
#define FLOAT32_EXPONENT_SPREAD 9u
#define HASH_SIGN_SHIFT 63u
#define HASH_EXPONENT_SHIFT 40u
#define SENTINEL_BYTE 0x5A                      /* laid in an output a step does not ask */
#define LARGEST_BFLOAT16_PLACES 1
#define MOST_POINTS 512u                        /* capture points of one product and layer */
#define NAME_CAPACITY 128u
#define DECIMAL 10
#define MOST_DIGITS 9u                          /* of a layer or a position in a record's name */
#define MATRIX_DIMENSIONS 2u
#define LOGIT_DIMENSIONS 3u                     /* logits.all_rows [1, positions, vocabulary] */
#define BREAK_ARGUMENTS 2                       /* --break and its name */
#define ONE_ROW_STEPS 2u                        /* steps of one row: one that asks its logits, one that does not */
#define HEAD_CUT_OUTPUTS 8192u                  /* of the head at its inputs */
#define HEAD_GRID_INPUTS 64u                    /* of the head at its outputs */
#define STREAM_VALUES \
	(MODEL_STREAMS * MODEL_HIDDEN) /* a hyper mix's inputs (REF:952) */
#define HIDDEN MODEL_HIDDEN
#define INDEX_SCALE \
	DENSE_INDEX_WEIGHTS_SCALE /* also the scale of the mixes' bfloat16 case */

enum test_status { TEST_PASSED = 0, TEST_FAILED = 1, TEST_WRONG_USAGE = 2 };

/* A synthetic case: a product of the model at its shape. */
struct dense_case {
    const char *name;
    uint32_t output_size;
    uint32_t input_size;
    uint32_t weight_type;
    uint32_t output_type;
    uint32_t flags;
    float output_scale;
    float epsilon;
};
static const struct dense_case cases[] = {
	{ "the head, cut to 8,192 outputs", HEAD_CUT_OUTPUTS, HIDDEN, BFLOAT16,
	  FLOAT32, DENSE_STEP_ROWS, 1.0f, 0.0f },
	{ "the head's grid, of 64 inputs", MODEL_VOCABULARY_VALUE,
	  HEAD_GRID_INPUTS, BFLOAT16, FLOAT32, DENSE_STEP_ROWS, 1.0f, 0.0f },
	{ "the router's scores", MODEL_EXPERTS, HIDDEN, BFLOAT16, FLOAT32, 0,
	  1.0f, 0.0f },
	{ "wkv and wgate at ratio 2", MODEL_HEAD_DIMS, HIDDEN, BFLOAT16,
	  FLOAT32, 0, 1.0f, 0.0f },
	{ "wkv at ratio 1", MODEL_HEAD_DIMS, HIDDEN, BFLOAT16, BFLOAT16, 0,
	  1.0f, 0.0f },
	{ "indexer.wk", MODEL_INDEX_HEAD_DIMS, MODEL_HEAD_DIMS, BFLOAT16,
	  BFLOAT16, 0, 1.0f, 0.0f },
	{ "the index weights", MODEL_INDEX_HEADS, HIDDEN, BFLOAT16, BFLOAT16, 0,
	  INDEX_SCALE, 0.0f },
	{ "a hyper mix", MODEL_MIXES, STREAM_VALUES, FLOAT32, FLOAT32,
	  DENSE_ROW_SCALE, 1.0f, MODEL_NORM_EPSILON },
	{ "a hyper mix to bfloat16, scaled", MODEL_MIXES, STREAM_VALUES,
	  FLOAT32, BFLOAT16, DENSE_ROW_SCALE, INDEX_SCALE, MODEL_NORM_EPSILON },
};
/* The kind of values of each row, and the rows of the launch of 5 in their order there. */
static const enum test_linear_values_kind row_kinds[ROWS] = {
    TEST_LINEAR_VALUES_PLAIN, TEST_LINEAR_VALUES_PLAIN, TEST_LINEAR_VALUES_SMALL, TEST_LINEAR_VALUES_PLAIN,
    TEST_LINEAR_VALUES_EDGES, TEST_LINEAR_VALUES_PLAIN, TEST_LINEAR_VALUES_PLAIN, TEST_LINEAR_VALUES_SMALL};
static const uint32_t reordered_rows[SOME_ROWS] = {6, 3, 7, 0, 5};
static const uint32_t flagged_rows[ROWS] = {1, 1, 0, 1, 0, 1, 1, 1};   /* of the step; rows 6 and 7 not asked */

/* One product of the replay: the records and the weights it reads, and how it is described. */
struct replay_product {
    const char *name;
    const char *input_record;
    const char *output_record;
    const char *weight_suffix;                  /* of layers.<L>.; the whole name for the head */
    uint32_t weight_type;
    uint32_t flags;
    float output_scale;
    float epsilon;
    uint32_t output_size;
    uint32_t input_size;
    bool is_head;                               /* points are position_P.final.normed, outputs logits.all_rows */
};
static const struct replay_product replay_products[] = {
	{ "the attention site's hyper mixes", "attn_site.streams",
	  "attn_site.mixes", ".hc_attn_fn", FLOAT32, DENSE_ROW_SCALE, 1.0f,
	  MODEL_NORM_EPSILON, MODEL_MIXES, STREAM_VALUES, false },
	{ "the FFN site's hyper mixes", "ffn_site.streams", "ffn_site.mixes",
	  ".hc_ffn_fn", FLOAT32, DENSE_ROW_SCALE, 1.0f, MODEL_NORM_EPSILON,
	  MODEL_MIXES, STREAM_VALUES, false },
	{ "the router's scores", "ffn_site.normed", "router.logits",
	  ".ffn.gate.weight", BFLOAT16, 0, 1.0f, 0.0f, MODEL_EXPERTS, HIDDEN,
	  false },
	{ "a compressor's wkv", "attn_site.normed", "compressor.wkv",
	  ".attn.compressor.wkv.weight", BFLOAT16, 0, 1.0f, 0.0f,
	  MODEL_HEAD_DIMS, HIDDEN, false },
	{ "a compressor's wgate", "attn_site.normed", "compressor.wgate",
	  ".attn.compressor.wgate.weight", BFLOAT16, 0, 1.0f, 0.0f,
	  MODEL_HEAD_DIMS, HIDDEN, false },
	{ "the weights projection", "attn_site.normed",
	  "indexer.weights_projection", ".attn.indexer.weights_proj.weight",
	  BFLOAT16, 0, 1.0f, 0.0f, MODEL_INDEX_HEADS, HIDDEN, false },
	{ "the index weights", "attn_site.normed", "indexer.weights",
	  ".attn.indexer.weights_proj.weight", BFLOAT16, 0, INDEX_SCALE, 0.0f,
	  MODEL_INDEX_HEADS, HIDDEN, false },
	{ "indexer.wk", "indexer.latent_bfloat16", "indexer.wk",
	  ".attn.indexer.wk.weight", BFLOAT16, 0, 1.0f, 0.0f,
	  MODEL_INDEX_HEAD_DIMS, MODEL_HEAD_DIMS, false },
	{ "the head", "final.normed", "logits.all_rows", "head.weight",
	  BFLOAT16, 0, 1.0f, 0.0f, MODEL_VOCABULARY_VALUE, HIDDEN, true },
};

static struct dense_linear_kernels dense_kernels;
static bool is_word_broken = false;             /* --break word */

/* ---- synthetic cases ---- */

/* Where a case lies on a card, and its launches: 8 rows, each row alone, 5 rows reordered, 8 rows of a step, row 0
 * alone in each step of one row. */
struct case_plan {
	struct linear_range weights;
	struct linear_range rows;
	struct linear_range reordered;
	struct linear_range step;
	struct linear_range one_row_steps[ONE_ROW_STEPS];
	struct linear_range all_output;
	struct linear_range alone_outputs[ROWS];
	struct linear_range some_output;
	struct linear_range step_output;
	struct linear_range one_row_outputs[ONE_ROW_STEPS];
	struct kernel_launch launches[1u + ROWS + 1u + 1u + ONE_ROW_STEPS];
	uint32_t launch_count;
};

/* Lays a case out in the image: weights, rows and step areas, the sentinel in the steps' outputs. */
static bool case_lay_out(struct test_linear_card *card, const struct dense_case *dense, uint64_t stream,
                         struct case_plan *plan, uint16_t *rows)
{
    const uint64_t weight_bytes = dense->weight_type == FLOAT32 ? sizeof(float) : sizeof(uint16_t);
    const uint64_t output_bytes = dense->output_type == FLOAT32 ? sizeof(float) : sizeof(uint16_t);
    const uint64_t row_bytes = (uint64_t)dense->input_size * sizeof(uint16_t);
    const uint64_t output_row_bytes = (uint64_t)dense->output_size * output_bytes;
    unsigned char
	    step_area[STEP_BLOCK_BYTES_VALUE + ROWS * STEP_ROW_BYTES_VALUE];
    const struct step_block block = { stream, ASKED_ROWS, 0 };
    test_linear_card_clear(card);
    plan->weights = test_linear_card_weights(card, (uint64_t)dense->output_size * dense->input_size * weight_bytes);
    plan->rows = test_linear_card_work(card, ROWS * row_bytes);
    plan->reordered = test_linear_card_work(card, SOME_ROWS * row_bytes);
    plan->step = test_linear_card_work(card, sizeof(step_area));
    plan->all_output = test_linear_card_work(card, ROWS * output_row_bytes);
    for (uint32_t row = 0; row < ROWS; row++) {
        plan->alone_outputs[row] = test_linear_card_work(card, output_row_bytes);
    }
    plan->some_output = test_linear_card_work(card, SOME_ROWS * output_row_bytes);
    plan->step_output = test_linear_card_work(card, ROWS * output_row_bytes);
    for (uint32_t step = 0; step < ONE_ROW_STEPS; step++) {
	    plan->one_row_steps[step] = test_linear_card_work(
		    card, STEP_BLOCK_BYTES_VALUE + STEP_ROW_BYTES_VALUE);
	    plan->one_row_outputs[step] =
		    test_linear_card_work(card, output_row_bytes);
    }
    unsigned char *const weights = test_linear_card_writable(card, plan->weights);
    unsigned char *const sentinel = malloc(ROWS * output_row_bytes);
    bool is_laid = weights != NULL && sentinel != NULL && test_linear_card_has_room(card);
    for (uint32_t output = 0; is_laid && output < dense->output_size; output++) {
        unsigned char *const output_weights = weights + (uint64_t)output * dense->input_size * weight_bytes;
        if (dense->weight_type == BFLOAT16) {           /* rows is free until the rows are made below */
            test_linear_values(rows, dense->input_size, stream + STREAM_OF_WEIGHTS + output, TEST_LINEAR_VALUES_PLAIN);
            memcpy(output_weights, rows, (size_t)dense->input_size * sizeof(uint16_t));
        }
        for (uint32_t input = 0; dense->weight_type == FLOAT32 && input < dense->input_size; input++) {
            const uint64_t hash =
                test_linear_hash(stream + STREAM_OF_WEIGHTS, (uint64_t)output * dense->input_size + input);
            const uint32_t float32_bits =
                (uint32_t)(hash >> HASH_SIGN_SHIFT) << FLOAT32_SIGN_SHIFT |
                (FLOAT32_SMALLEST_EXPONENT_FIELD + (uint32_t)(hash >> HASH_EXPONENT_SHIFT) % FLOAT32_EXPONENT_SPREAD)
                    << FLOAT32_EXPONENT_SHIFT |
                ((uint32_t)hash & FLOAT32_FRACTION_MASK);
            memcpy(output_weights + input * sizeof(float), &float32_bits, sizeof(float32_bits));
        }
    }
    for (uint32_t row = 0; is_laid && row < ROWS; row++) {
        test_linear_values(rows + (size_t)row * dense->input_size, dense->input_size, stream + STREAM_OF_ROWS + row,
                           row_kinds[row]);
    }
    memcpy(step_area, &block, sizeof(block));
    for (uint32_t row = 0; row < ROWS; row++) {
	    const struct step_row_record step_row = {
		    (int32_t)row, 0, 0, flagged_rows[row] * STEP_ROW_LOGITS
	    };
	    memcpy(step_area + STEP_BLOCK_BYTES_VALUE +
			   (size_t)row * STEP_ROW_BYTES_VALUE,
		   &step_row, sizeof(step_row));
    }
    if (sentinel != NULL) {
        memset(sentinel, SENTINEL_BYTE, ROWS * output_row_bytes);
    }
    is_laid = is_laid && test_linear_card_put(card, plan->rows, 0, rows, ROWS * row_bytes) &&
              test_linear_card_put(card, plan->step, 0, step_area, sizeof(step_area)) &&
              test_linear_card_put(card, plan->step_output, 0, sentinel, ROWS * output_row_bytes);
    for (uint32_t step = 0; is_laid && step < ONE_ROW_STEPS; step++) {     /* row 0 asked in the first, not after */
	    const struct step_block one_row_block = { stream, 1, 0 };
	    const struct step_row_record step_row = {
		    0, 0, 0, step == 0 ? STEP_ROW_LOGITS : 0
	    };
	    is_laid = test_linear_card_put(card, plan->one_row_steps[step], 0,
					   &one_row_block,
					   sizeof(one_row_block)) &&
		      test_linear_card_put(card, plan->one_row_steps[step],
					   STEP_BLOCK_BYTES_VALUE, &step_row,
					   sizeof(step_row)) &&
		      test_linear_card_put(card, plan->one_row_outputs[step], 0,
					   sentinel, output_row_bytes);
    }
    for (uint32_t index = 0; is_laid && index < SOME_ROWS; index++) {
        is_laid = test_linear_card_put(card, plan->reordered, index * row_bytes,
                                       rows + (size_t)reordered_rows[index] * dense->input_size, row_bytes);
    }
    free(sentinel);
    return test_linear_expect(is_laid, "the case is laid out in the image");
}

/* One launch of a case: rows of a range from a row on, row_count of them, to an output; with a step area, with the
 * case's flags, DENSE_STEP_ROWS among them, and without one, with the others. */
static bool case_launch(struct test_linear_card *card,
			const struct dense_case *dense, struct case_plan *plan,
			struct linear_range rows, uint32_t first_row,
			uint32_t row_count, struct linear_range output,
			const struct linear_range *step)
{
    const uint64_t row_bytes = (uint64_t)dense->input_size * sizeof(uint16_t);
    const struct dense_linear_description description = {
	    .weights = plan->weights,
	    .rows = { rows.byte_offset + first_row * row_bytes,
		      row_count * row_bytes },
	    .output = output,
	    .step = step != NULL ? *step : (struct linear_range){ 0, 0 },
	    .output_size = dense->output_size,
	    .input_size = dense->input_size,
	    .row_count = row_count,
	    .weight_type = dense->weight_type,
	    .output_type = dense->output_type,
	    .flags = step != NULL ? dense->flags :
				    dense->flags & ~DENSE_STEP_ROWS,
	    .output_scale = dense->output_scale,
	    .epsilon = dense->epsilon
    };
    struct kernel_launch *const launch = &plan->launches[plan->launch_count];
    plan->launch_count++;
    return test_linear_expect_status(
	    dense_linear_prepare(&description, card->arena_byte_count,
				 &dense_kernels, launch),
	    STATUS_OK, "the launch is prepared");
}

/* Holds every output of rows [ROWS] of an output range against the reference; the count outside. */
static uint64_t outside_count_of(const struct test_linear_card *card,
				 const struct dense_case *dense,
				 struct linear_range output, const double *sums,
				 const double *magnitudes,
				 unsigned char *received)
{
    uint64_t outside = 0;
    if (!test_linear_card_get(card, output, 0, received, output.byte_count)) {
        return UINT64_MAX;
    }
    for (uint64_t index = 0; index < (uint64_t)ROWS * dense->output_size; index++) {
        outside += test_dense_output_holds(dense->output_type, dense->output_scale, received, index, sums[index],
                                           magnitudes[index]) ? 0u : 1u;
    }
    return outside;
}

/* The shown failing of a case: one weight to CHANGED_WEIGHT on the card, the launch of 8 rows again. */
static void case_shown_failing(struct test_linear_card *card, const struct dense_case *dense, struct case_plan *plan,
                               const uint16_t *rows, double *sums, double *magnitudes, unsigned char *before,
                               unsigned char *after)
{
    const uint64_t weight_bytes = dense->weight_type == FLOAT32 ? sizeof(float) : sizeof(uint16_t);
    const uint64_t output_bytes = dense->output_type == FLOAT32 ? sizeof(float) : sizeof(uint16_t);
    const uint64_t place = ((uint64_t)CHANGED_OUTPUT * dense->input_size + CHANGED_INPUT) * weight_bytes;
    const float changed_float32 = (float)CHANGED_WEIGHT;
    const uint16_t changed_bfloat16 = test_linear_bfloat16_nearest(CHANGED_WEIGHT);
    unsigned char *const weights = test_linear_card_writable(card, plan->weights);
    double changed_sums[ROWS];
    double changed_magnitudes[ROWS];
    bool is_only_its_output = true;
    bool fails_before = true;
    bool holds_after = true;
    if (weights == NULL || !test_linear_card_get(card, plan->all_output, 0, before, plan->all_output.byte_count)) {
        test_linear_fail("the shown failing finds no weights or outputs");
        return;
    }
    memcpy(weights + place, dense->weight_type == FLOAT32 ? (const void *)&changed_float32
                                                          : (const void *)&changed_bfloat16, weight_bytes);
    test_dense_reference(dense->weight_type, weights, dense->input_size, dense->flags, dense->epsilon, rows, ROWS,
                         CHANGED_OUTPUT, 1, changed_sums, changed_magnitudes);
    if (!test_linear_card_upload(card, plan->weights, place, weight_bytes) ||
        !test_linear_card_run(card, plan->launches, 1) || !test_linear_card_fetch(card) ||
        !test_linear_card_get(card, plan->all_output, 0, after, plan->all_output.byte_count)) {
        return;
    }
    for (uint32_t row = 0; row < ROWS; row++) {
        for (uint32_t output = 0; output < dense->output_size; output++) {
            const uint64_t index = (uint64_t)row * dense->output_size + output;
            const bool is_same = memcmp(before + index * output_bytes, after + index * output_bytes, output_bytes) == 0;
            is_only_its_output = is_only_its_output && (output == CHANGED_OUTPUT || is_same);
        }
        const uint64_t index = (uint64_t)row * dense->output_size + CHANGED_OUTPUT;
        /* a row whose value there is zero, or so small that the change hides in the bound, cannot show it */
        const bool can_show = row_kinds[row] == TEST_LINEAR_VALUES_PLAIN;
        fails_before = fails_before && (!can_show || !test_dense_output_holds(dense->output_type, dense->output_scale,
                                                                               after, index, sums[index],
                                                                               magnitudes[index]));
        holds_after = holds_after && test_dense_output_holds(dense->output_type, dense->output_scale, after, index,
                                                             changed_sums[row], changed_magnitudes[row]);
    }
    test_linear_expect(is_only_its_output, "a changed weight changes no output but its own");
    test_linear_expect(fails_before, "shown failing: against the reference before the change, its output fails");
    test_linear_expect(holds_after, "against the reference after the change, its output holds");
}

/* One case on one card. */
static void case_run(struct test_linear_card *card, const struct dense_case *dense, uint64_t stream)
{
    static struct case_plan plan;
    const uint64_t output_count = (uint64_t)ROWS * dense->output_size;
    const uint64_t output_bytes = dense->output_type == FLOAT32 ? sizeof(float) : sizeof(uint16_t);
    const uint64_t row_output_bytes = dense->output_size * output_bytes;
    const bool is_step = (dense->flags & DENSE_STEP_ROWS) != 0;
    uint16_t *const rows = malloc((size_t)ROWS * dense->input_size * sizeof(uint16_t));
    double *const sums = malloc(output_count * sizeof(double));
    double *const magnitudes = malloc(output_count * sizeof(double));
    unsigned char *const received = malloc(output_count * output_bytes);
    unsigned char *const other = malloc(output_count * output_bytes);
    test_linear_context("card %" PRIu32 ", %s", card->card_index, dense->name);
    memset(&plan, 0, sizeof(plan));
    bool is_ready = test_linear_expect(rows != NULL && sums != NULL && magnitudes != NULL && received != NULL &&
                                           other != NULL,
                                       "the host memory of the case is obtained") &&
                    case_lay_out(card, dense, stream, &plan, rows) &&
                    case_launch(card, dense, &plan, plan.rows, 0, ROWS, plan.all_output, NULL);
    for (uint32_t row = 0; is_ready && row < ROWS; row++) {
        is_ready = case_launch(card, dense, &plan, plan.rows, row, 1, plan.alone_outputs[row], NULL);
    }
    is_ready = is_ready && case_launch(card, dense, &plan, plan.reordered, 0, SOME_ROWS, plan.some_output, NULL) &&
               (!is_step || case_launch(card, dense, &plan, plan.rows, 0, ROWS, plan.step_output, &plan.step));
    for (uint32_t step = 0; is_ready && is_step && step < ONE_ROW_STEPS; step++) {
        is_ready =
            case_launch(card, dense, &plan, plan.rows, 0, 1, plan.one_row_outputs[step], &plan.one_row_steps[step]);
    }
    is_ready = is_ready && test_linear_card_load(card) && test_linear_card_expect_weights_untouched(card) &&
               test_linear_card_run(card, plan.launches, plan.launch_count) && test_linear_card_fetch(card);
    if (is_ready) {
        test_dense_reference(dense->weight_type, test_linear_card_stored(card, plan.weights), dense->input_size,
                             dense->flags, dense->epsilon, rows, ROWS, 0, dense->output_size, sums, magnitudes);
        const uint64_t outside = outside_count_of(card, dense, plan.all_output, sums, magnitudes, received);
        if (outside != 0) {
            test_linear_fail("%" PRIu64 " of %" PRIu64 " outputs of 8 rows are outside the bound of the reference",
                             outside, output_count);
        }
        for (uint32_t row = 0; row < ROWS; row++) {
            (void)test_linear_card_expect_same(card, plan.alone_outputs[row], 0, plan.all_output,
                                               row * row_output_bytes, row_output_bytes,
                                               "a row launched alone has the bits of the launch of 8 rows");
        }
        for (uint32_t index = 0; index < SOME_ROWS; index++) {
            (void)test_linear_card_expect_same(card, plan.some_output, index * row_output_bytes, plan.all_output,
                                               reordered_rows[index] * row_output_bytes, row_output_bytes,
                                               "a row of 5 in another order has the bits of the launch of 8 rows");
        }
        for (uint32_t row = 0; is_step && row < ROWS; row++) {
            const bool is_asked = row < ASKED_ROWS && flagged_rows[row] != 0;
            if (is_asked) {
                (void)test_linear_card_expect_same(card, plan.step_output, row * row_output_bytes, plan.all_output,
                                                   row * row_output_bytes, row_output_bytes,
                                                   "a row the step asks has the bits of the launch without the mode");
            } else {
                (void)test_linear_card_expect_bytes(card, plan.step_output, row * row_output_bytes,
                                                    test_linear_card_stored(card, plan.step_output) +
                                                        row * row_output_bytes,
                                                    row_output_bytes, "a row the step does not ask is not written");
            }
        }
        if (is_step) {
            (void)test_linear_card_expect_same(card, plan.one_row_outputs[0], 0, plan.alone_outputs[0], 0,
                                               row_output_bytes, "a row alone that its step asks has its bits alone");
            (void)test_linear_card_expect_bytes(card, plan.one_row_outputs[1], 0,
                                                test_linear_card_stored(card, plan.one_row_outputs[1]),
                                                row_output_bytes, "a row alone its step does not ask is not written");
        }
        (void)test_linear_card_expect_untouched(card, plan.rows, "the rows");
        (void)test_linear_card_expect_untouched(card, plan.reordered, "the rows reordered");
        (void)test_linear_card_expect_untouched(card, plan.step, "the step area");
        for (uint32_t step = 0; step < ONE_ROW_STEPS; step++) {
            (void)test_linear_card_expect_untouched(card, plan.one_row_steps[step], "a step area of one row");
        }
        (void)test_linear_card_expect_weights_untouched(card);
        (void)printf("card %" PRIu32 ", %s: %" PRIu64 " outputs of 8 rows within the bound\n", card->card_index,
                     dense->name, output_count - outside);
        case_shown_failing(card, dense, &plan, rows, sums, magnitudes, received, other);
    }
    free(other);
    free(received);
    free(magnitudes);
    free(sums);
    free(rows);
}

/* ---- replay ---- */

/* What the replay reads: the capture, and the pack through its storage reader. */
struct replay_sources {
    struct test_capture *capture;
    struct storage_reader *reader;
    struct pack_record *pack;
    uint32_t block_alignment_bytes;
};

/* What one product added up to on one card; by output type where a product has both (wkv: float32 at ratio 2,
 * bfloat16 at ratio 1), since plan 3.2's share of bit-equal outputs is the bfloat16 outputs' alone. */
struct replay_total {
    uint64_t point_count;
    uint64_t output_counts[DENSE_TYPE_COUNT];
    uint64_t equal_counts[DENSE_TYPE_COUNT]; /* bit-equal to the reference's */
    uint64_t largest_places
	    [DENSE_TYPE_COUNT]; /* from the reference's, in places of the type */
    uint64_t outside_host_count;                /* outside the bound of the host reference */
    uint64_t outside_reference_count;           /* float32: outside the bound of the reference's; bfloat16: beyond
                                                   one place of it */
    uint64_t reference_outside_host_count;      /* the reference's own outputs outside the host reference */
};

/* One layer of one product on one card: its points, its ranges, and the rows, outputs and references of a batch. */
struct replay_plan {
    const struct replay_product *product;
    uint32_t layer;
    uint32_t output_type;
    uint32_t point_count;
    uint32_t positions[MOST_POINTS];
    struct linear_range weights;
    struct linear_range rows;
    struct linear_range output;
    uint16_t *row_values;                       /* [ROWS][input_size] */
    unsigned char *outputs;                     /* [ROWS][output_size] of output_type: the card's */
    unsigned char *expected;                    /* the same: the reference's */
    double *sums;                               /* [ROWS][output_size] */
    double *magnitudes;
};

/* A whole number of at most nine digits at *cursor, which it passes. */
static bool number_at(const char **cursor, uint32_t *number_out)
{
    uint32_t number = 0;
    uint32_t digit_count = 0;
    while (digit_count < MOST_DIGITS && **cursor >= '0' && **cursor <= '9') {
        number = number * DECIMAL + (uint32_t)(**cursor - '0');
        (*cursor)++;
        digit_count++;
    }
    *number_out = number;
    return digit_count != 0;
}

/* Whether a record's name is the input of a capture point of the product at the layer, and its position. */
static bool position_of(const char *name, const struct replay_product *product, uint32_t layer, uint32_t *position_out)
{
    static const char layer_prefix[] = "layer_";
    static const char position_prefix[] = "position_";
    const char *cursor = name;
    uint32_t number = 0;
    if (!product->is_head) {
        if (strncmp(cursor, layer_prefix, sizeof(layer_prefix) - 1) != 0) {
            return false;
        }
        cursor += sizeof(layer_prefix) - 1;
        if (!number_at(&cursor, &number) || number != layer || *cursor != '.') {
            return false;
        }
        cursor++;
    }
    if (strncmp(cursor, position_prefix, sizeof(position_prefix) - 1) != 0) {
        return false;
    }
    cursor += sizeof(position_prefix) - 1;
    return number_at(&cursor, position_out) && *cursor == '.' && strcmp(cursor + 1, product->input_record) == 0;
}

/* The name of the record of a point: layer_<L>.position_<P>.<record>, or the head's. */
static bool point_name(const struct replay_plan *plan, uint32_t position, bool is_output, char *name_out)
{
    const struct replay_product *const product = plan->product;
    const char *const record = is_output ? product->output_record : product->input_record;
    const int length =
        product->is_head && is_output ? snprintf(name_out, NAME_CAPACITY, "%s", record)
        : product->is_head ? snprintf(name_out, NAME_CAPACITY, "position_%" PRIu32 ".%s", position, record)
                           : snprintf(name_out, NAME_CAPACITY, "layer_%" PRIu32 ".position_%" PRIu32 ".%s",
                                      plan->layer, position, record);
    return test_linear_expect(length > 0 && length < (int)NAME_CAPACITY, "the name of a record fits");
}

/* The positions of the layer whose points hold the product's input and output, each as the product reads it; the
 * output's type. False when the capture holds a record of either name in another form. */
static bool points_find(const struct replay_sources *sources, struct replay_plan *plan)
{
    const struct replay_product *const product = plan->product;
    const uint32_t record_count = test_capture_tensor_count(sources->capture);
    plan->point_count = 0;
    for (uint32_t record_index = 0; record_index < record_count; record_index++) {
        const struct test_capture_tensor *const input = test_capture_tensor_at(sources->capture, record_index);
        char name[NAME_CAPACITY];
        uint32_t position = 0;
        if (input == NULL || !position_of(input->name, product, plan->layer, &position) ||
            !point_name(plan, position, true, name)) {
            continue;
        }
        const struct test_capture_tensor *const output = test_capture_find(sources->capture, name);
        if (output == NULL || (product->is_head && (output->dimension_count != LOGIT_DIMENSIONS ||
                                                    position >= output->dimensions[1]))) {
            continue;                           /* a point where the layer computed no such output, or no logits */
        }
        const uint64_t output_elements =
            product->is_head ? output->dimensions[LOGIT_DIMENSIONS - 1] : output->element_count;
        const uint32_t output_type = output->type == TEST_CAPTURE_FLOAT32 ? FLOAT32 : BFLOAT16;
        if (input->type != TEST_CAPTURE_BFLOAT16 || input->element_count != product->input_size ||
            (output->type != TEST_CAPTURE_FLOAT32 && output->type != TEST_CAPTURE_BFLOAT16) ||
            output_elements != product->output_size || (plan->point_count != 0 && output_type != plan->output_type) ||
            plan->point_count == MOST_POINTS) {
            test_linear_fail("%s of %s: the capture holds it as type %s, %" PRIu64 " values, and its output %s as %s, "
                             "%" PRIu64 " values: expected bfloat16 of %" PRIu32 ", and float32 or bfloat16 of %" PRIu32
                             " (one type in a layer), at most %u points", input->name, product->name,
                             test_capture_type_name((enum test_capture_type)input->type), input->element_count, name,
                             test_capture_type_name((enum test_capture_type)output->type), output_elements,
                             product->input_size, product->output_size, MOST_POINTS);
            return false;
        }
        plan->output_type = output_type;
        plan->positions[plan->point_count++] = position;
    }
    return true;
}

/* The product's matrix of the layer, from the pack, into the image's weights region: its name, element type, shape
 * and bytes as the product reads it. */
static bool weights_read(struct test_linear_card *card, const struct replay_sources *sources, struct replay_plan *plan)
{
    const struct replay_product *const product = plan->product;
    const uint32_t element_type = product->weight_type == FLOAT32 ?
					  PACK_ELEMENT_FLOAT32 :
					  PACK_ELEMENT_BFLOAT16;
    const uint64_t weight_bytes = product->weight_type == FLOAT32 ? sizeof(float) : sizeof(uint16_t);
    const uint64_t alignment = sources->block_alignment_bytes;
    char name[NAME_CAPACITY];
    const int length = product->is_head ? snprintf(name, sizeof(name), "%s", product->weight_suffix)
                                        : snprintf(name, sizeof(name), "layers.%" PRIu32 "%s", plan->layer,
                                                   product->weight_suffix);
    const uint32_t tensor_index =
	    length > 0 && length < (int)sizeof(name) ?
		    pack_find_tensor(sources->pack, name, (uint32_t)length) :
		    PACK_NOT_FOUND;
    if (tensor_index == PACK_NOT_FOUND || alignment == 0) {
	    test_linear_fail(
		    "the pack holds no tensor %s, or its reader states no block",
		    name);
	    return false;
    }
    const struct pack_tensor *const tensor =
	    &pack_tensors_of(sources->pack)[tensor_index];
    const struct pack_extent *const extent =
	    &pack_extents_of(sources->pack)[tensor->extent_index];
    if (tensor->weight_format != WEIGHT_FORMAT_NONE ||
	tensor->element_type != element_type ||
	tensor->dimension_count != MATRIX_DIMENSIONS ||
	tensor->dimensions[0] != product->output_size ||
	tensor->dimensions[1] != product->input_size ||
	tensor->byte_count != (uint64_t)product->output_size *
				      product->input_size * weight_bytes) {
	    test_linear_fail("%s is, in the pack, of element type %" PRIu32
			     " and %" PRIu32 " by %" PRIu32 ": %s reads "
			     "type %" PRIu32 " of %" PRIu32 " by %" PRIu32,
			     name, tensor->element_type, tensor->dimensions[0],
			     tensor->dimensions[1], product->name, element_type,
			     product->output_size, product->input_size);
	    return false;
    }
    /* a read is whole blocks of the reader: from the block the matrix begins in to the end of the one it ends in */
    const uint64_t file_byte_offset = extent->byte_offset + tensor->byte_offset_in_extent;
    const uint64_t lead_byte_count = file_byte_offset % alignment;
    const uint64_t read_byte_count = (lead_byte_count + tensor->byte_count + alignment - 1) / alignment * alignment;
    const struct linear_range read_range =
	    test_linear_card_weights(card, read_byte_count);
    unsigned char *const destination = test_linear_card_writable(card, read_range);
    plan->weights =
	    (struct linear_range){ read_range.byte_offset + lead_byte_count,
				   tensor->byte_count };
    return test_linear_expect(destination != NULL,
			      "the weights region has room for the matrix") &&
	   test_linear_expect_status(storage_direct.reader_read(
					     sources->reader, 0,
					     file_byte_offset - lead_byte_count,
					     destination, read_byte_count),
				     STATUS_OK,
				     "the matrix is read from the pack");
}

/* The ordinal of a float32: one apart for neighbours, 0 for both zeros. */
static int64_t float32_ordinal(float value)
{
    uint32_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    const int64_t magnitude = (int64_t)(bits & ~((uint32_t)1u << FLOAT32_SIGN_SHIFT));
    return (bits >> FLOAT32_SIGN_SHIFT) != 0 ? -magnitude : magnitude;
}

/* Changes, for --break word, the weight of output 0 that meets the largest value of the first row, on the card. */
static bool weight_broken(struct test_linear_card *card, const struct replay_plan *plan)
{
    const struct replay_product *const product = plan->product;
    const uint64_t weight_bytes = product->weight_type == FLOAT32 ? sizeof(float) : sizeof(uint16_t);
    const float changed_float32 = (float)CHANGED_WEIGHT;
    const uint16_t changed_bfloat16 = test_linear_bfloat16_nearest(CHANGED_WEIGHT);
    unsigned char *const weights = test_linear_card_writable(card, plan->weights);
    uint32_t largest_input = 0;
    for (uint32_t input = 1; input < product->input_size; input++) {
        largest_input = test_dense_magnitude(test_linear_bfloat16_value(plan->row_values[input])) >
                                test_dense_magnitude(test_linear_bfloat16_value(plan->row_values[largest_input]))
                            ? input
                            : largest_input;
    }
    if (weights == NULL) {
        return test_linear_expect(false, "the broken weight lies in the image");
    }
    memcpy(weights + largest_input * weight_bytes, product->weight_type == FLOAT32 ? (const void *)&changed_float32
                                                                                   : (const void *)&changed_bfloat16,
           weight_bytes);
    return test_linear_card_upload(card, plan->weights, largest_input * weight_bytes, weight_bytes);
}

/* One batch of point_count points from first_point on: their rows, one launch, and the comparisons. */
static bool batch_run(struct test_linear_card *card, const struct replay_sources *sources, struct replay_plan *plan,
                      uint32_t first_point, uint32_t point_count, struct replay_total *total)
{
    const struct replay_product *const product = plan->product;
    const uint32_t output_size = product->output_size;
    const uint64_t output_bytes = plan->output_type == FLOAT32 ? sizeof(float) : sizeof(uint16_t);
    const enum test_capture_type output_capture_type =
        plan->output_type == FLOAT32 ? TEST_CAPTURE_FLOAT32 : TEST_CAPTURE_BFLOAT16;
    struct kernel_launch launch;
    for (uint32_t row = 0; row < point_count; row++) {
        const uint32_t position = plan->positions[first_point + row];
        char input_name[NAME_CAPACITY];
        char output_name[NAME_CAPACITY];
        if (!point_name(plan, position, false, input_name) || !point_name(plan, position, true, output_name)) {
            return false;
        }
        /* points_find saw both records in these forms */
        const struct test_capture_tensor *const input = test_capture_find(sources->capture, input_name);
        const struct test_capture_tensor *const output = test_capture_find(sources->capture, output_name);
        if (input == NULL || output == NULL) {
            test_linear_fail("the records of a point are not in the capture");
            return false;
        }
	if (!test_linear_expect_status(
		    test_capture_read(
			    sources->capture, input_name, TEST_CAPTURE_BFLOAT16,
			    input->dimensions, input->dimension_count, 0,
			    product->input_size,
			    plan->row_values + (size_t)row * product->input_size,
			    (uint64_t)product->input_size * sizeof(uint16_t)),
		    STATUS_OK, "a point's input is read") ||
	    !test_linear_expect_status(
		    test_capture_read(
			    sources->capture, output_name, output_capture_type,
			    output->dimensions, output->dimension_count,
			    product->is_head ?
				    (uint64_t)position * output_size :
				    0,
			    output_size,
			    plan->expected +
				    (uint64_t)row * output_size * output_bytes,
			    output_size * output_bytes),
		    STATUS_OK, "a point's output is read")) {
		return false;
	}
    }
    const struct dense_linear_description description = {
	    .weights = plan->weights,
	    .rows = plan->rows,
	    .output = plan->output,
	    .step = { 0, 0 },
	    .output_size = output_size,
	    .input_size = product->input_size,
	    .row_count = point_count,
	    .weight_type = product->weight_type,
	    .output_type = plan->output_type,
	    .flags = product->flags,
	    .output_scale = product->output_scale,
	    .epsilon = product->epsilon
    };
    if (!test_linear_card_put(card, plan->rows, 0, plan->row_values,
			      (uint64_t)point_count * product->input_size *
				      sizeof(uint16_t)) ||
	!test_linear_card_upload(card, plan->rows, 0,
				 (uint64_t)point_count * product->input_size *
					 sizeof(uint16_t)) ||
	(is_word_broken && first_point == 0 && !weight_broken(card, plan)) ||
	!test_linear_expect_status(
		dense_linear_prepare(&description, card->arena_byte_count,
				     &dense_kernels, &launch),
		STATUS_OK, "the launch of a batch is prepared") ||
	!test_linear_card_run(card, &launch, 1) ||
	!test_linear_card_fetch(card) ||
	!test_linear_card_get(card, plan->output, 0, plan->outputs,
			      (uint64_t)point_count * output_size *
				      output_bytes)) {
	    return false;
    }
    test_dense_reference(product->weight_type, test_linear_card_stored(card, plan->weights), product->input_size,
                         product->flags, product->epsilon, plan->row_values, point_count, 0, output_size, plan->sums,
                         plan->magnitudes);
    for (uint64_t index = 0; index < (uint64_t)point_count * output_size; index++) {
        const double bound = (TEST_DENSE_TOLERANCE * plan->magnitudes[index] + TEST_DENSE_SUBNORMAL_ALLOWANCE) *
                             (double)product->output_scale;
        int64_t places = 0;
        bool is_near = false;
        if (plan->output_type == FLOAT32) {
            float card_value = 0.0f, reference_value = 0.0f;
            memcpy(&card_value, plan->outputs + index * sizeof(float), sizeof(card_value));
            memcpy(&reference_value, plan->expected + index * sizeof(float), sizeof(reference_value));
            places = float32_ordinal(card_value) - float32_ordinal(reference_value);
            is_near = test_dense_magnitude((double)card_value - (double)reference_value) <= bound;
        } else {
            uint16_t card_bits = 0, reference_bits = 0;
            memcpy(&card_bits, plan->outputs + index * sizeof(uint16_t), sizeof(card_bits));
            memcpy(&reference_bits, plan->expected + index * sizeof(uint16_t), sizeof(reference_bits));
            places = (int64_t)test_linear_bfloat16_ordinal(card_bits) - test_linear_bfloat16_ordinal(reference_bits);
            is_near = places <= LARGEST_BFLOAT16_PLACES && places >= -LARGEST_BFLOAT16_PLACES;
        }
        const uint64_t magnitude = (uint64_t)(places < 0 ? -places : places);
        total->equal_counts[plan->output_type] +=
            memcmp(plan->outputs + index * output_bytes, plan->expected + index * output_bytes, output_bytes) == 0
                ? 1u : 0u;
        total->outside_reference_count += is_near ? 0u : 1u;
        total->outside_host_count += test_dense_output_holds(plan->output_type, product->output_scale, plan->outputs,
                                                             index, plan->sums[index], plan->magnitudes[index])
                                         ? 0u : 1u;
        total->reference_outside_host_count +=
            test_dense_output_holds(plan->output_type, product->output_scale, plan->expected, index, plan->sums[index],
                                    plan->magnitudes[index]) ? 0u : 1u;
        total->largest_places[plan->output_type] = magnitude > total->largest_places[plan->output_type]
                                                       ? magnitude : total->largest_places[plan->output_type];
    }
    total->point_count += point_count;
    total->output_counts[plan->output_type] += (uint64_t)point_count * output_size;
    return true;
}

/* One layer of one product on one card: its points in batches of up to 8. */
static bool layer_run(struct test_linear_card *card, const struct replay_sources *sources, struct replay_plan *plan,
                      struct replay_total *total)
{
    const struct replay_product *const product = plan->product;
    if (!points_find(sources, plan)) {
        return false;
    }
    if (plan->point_count == 0) {
        return true;
    }
    const uint64_t output_bytes = plan->output_type == FLOAT32 ? sizeof(float) : sizeof(uint16_t);
    test_linear_card_clear(card);
    if (!weights_read(card, sources, plan)) {
        return false;
    }
    plan->rows = test_linear_card_work(card, (uint64_t)ROWS * product->input_size * sizeof(uint16_t));
    plan->output = test_linear_card_work(card, (uint64_t)ROWS * product->output_size * output_bytes);
    if (!test_linear_card_load(card)) {
        return false;
    }
    for (uint32_t first_point = 0; first_point < plan->point_count; first_point += ROWS) {
        const uint32_t left = plan->point_count - first_point;
        if (!batch_run(card, sources, plan, first_point, left < ROWS ? left : ROWS, total)) {
            return false;
        }
    }
    return test_linear_card_expect_weights_untouched(card);
}

/* Every product over every layer on one card, each held to its gate (or, broken, to failing it). */
static void replay_run(struct test_linear_card *card, const struct replay_sources *sources)
{
    static struct replay_plan plan;
    uint64_t most_values = 0;
    uint64_t most_inputs = 0;
    for (size_t product_index = 0; product_index < COUNT_OF(replay_products); product_index++) {
        const struct replay_product *const product = &replay_products[product_index];
        most_values = product->output_size > most_values ? product->output_size : most_values;
        most_inputs = product->input_size > most_inputs ? product->input_size : most_inputs;
    }
    most_values *= ROWS;
    memset(&plan, 0, sizeof(plan));
    const size_t widest_output_bytes = sizeof(float);
    plan.row_values = malloc((size_t)ROWS * most_inputs * sizeof(uint16_t));
    plan.outputs = malloc(most_values * widest_output_bytes);
    plan.expected = malloc(most_values * widest_output_bytes);
    plan.sums = malloc(most_values * sizeof(double));
    plan.magnitudes = malloc(most_values * sizeof(double));
    for (size_t product_index = 0; product_index < COUNT_OF(replay_products); product_index++) {
        const struct replay_product *const product = &replay_products[product_index];
        struct replay_total total;
        bool is_run = test_linear_expect(plan.row_values != NULL && plan.outputs != NULL && plan.expected != NULL &&
                                             plan.sums != NULL && plan.magnitudes != NULL,
                                         "the host memory of the replay is obtained");
        memset(&total, 0, sizeof(total));
        plan.product = product;
        test_linear_context("card %" PRIu32 ", replay of %s", card->card_index, product->name);
	for (uint32_t layer = 0;
	     is_run && layer < (product->is_head ? 1u : MODEL_LAYERS);
	     layer++) {
		plan.layer = layer;
		is_run = layer_run(card, sources, &plan, &total);
	}
	const bool holds = total.outside_host_count == 0 && total.outside_reference_count == 0 &&
                           (total.output_counts[BFLOAT16] == 0 ||
                            test_linear_share_holds(total.equal_counts[BFLOAT16], total.output_counts[BFLOAT16]));
        (void)printf("card %" PRIu32 ", replay of %s: %" PRIu64 " points; float32 and bfloat16 outputs %" PRIu64
                     " and %" PRIu64 ", %.4f%% and %.4f%% bit-equal to the reference's, the largest differences %"
                     PRIu64 " and %" PRIu64 " places; %" PRIu64 " beyond the gate, %" PRIu64 " outside the host "
                     "reference, and %" PRIu64 " of the reference's own\n", card->card_index, product->name,
                     total.point_count, total.output_counts[FLOAT32], total.output_counts[BFLOAT16],
                     test_linear_percent(total.equal_counts[FLOAT32], total.output_counts[FLOAT32]),
                     test_linear_percent(total.equal_counts[BFLOAT16], total.output_counts[BFLOAT16]),
                     total.largest_places[FLOAT32], total.largest_places[BFLOAT16], total.outside_reference_count,
                     total.outside_host_count, total.reference_outside_host_count);
        if (!is_run) {
            test_linear_fail("the replay of %s did not run to its end", product->name);
        } else if (total.point_count == 0) {
            test_linear_fail("%s: the capture holds no point with both %s and %s (logits.all_rows: --all-logits)",
                             product->name, product->input_record, product->output_record);
        } else if (is_word_broken) {
            if (holds || total.outside_host_count != 0) {
                test_linear_fail("--break word: %s is not reported, or the card differs from the host's reference",
                                 product->name);
            } else {
                (void)printf("card %" PRIu32 ", replay of %s: --break word reported\n", card->card_index,
                             product->name);
            }
        } else if (!holds) {
            test_linear_fail("%s: %" PRIu64 " outputs beyond the gate against the reference's, %" PRIu64 " outside "
                             "the host reference and %" PRIu64 " of the reference's own (where only these: the "
                             "capture's rounding, not the card's)", product->name, total.outside_reference_count,
                             total.outside_host_count, total.reference_outside_host_count);
        }
    }
    free(plan.magnitudes);
    free(plan.sums);
    free(plan.expected);
    free(plan.outputs);
    free(plan.row_values);
}

/* ---- the cards ---- */

/* Every case, and the replay with a capture, on one card; the digest of everything it computed. */
static uint64_t card_run(uint32_t card_index, const struct replay_sources *sources)
{
    static struct test_linear_card card;
    uint64_t digest = 0;
    if (test_linear_card_open(&card, card_index, WORK_CAPACITY,
			      WEIGHTS_CAPACITY) &&
	test_linear_expect_status(
		dense_linear_kernels_prepare(&backend_cuda_value, card.card,
					     &dense_kernels),
		STATUS_OK, "the kernels of the dense products are prepared")) {
	    for (size_t case_index = 0; case_index < COUNT_OF(cases);
		 case_index++) {
		    case_run(&card, &cases[case_index],
			     (uint64_t)case_index * STREAM_OF_CASES);
	    }
	    digest = card.digest;
	    (void)printf("card %" PRIu32 " (class %" PRIu32
			 "): digest of everything the synthetic cases computed "
			 "0x%016" PRIX64 "\n",
			 card_index, card.arithmetic_class, digest);
	    if (sources->capture != NULL) {
		    replay_run(&card, sources);
	    }
    }
    test_linear_card_close(&card);
    return digest;
}

int main(int argument_count, char **arguments)
{
    static struct replay_sources sources;
    static const char break_option[] = "--break";
    static const char word_break[] = "word";
    struct storage_description description = { 0, 0, 0 };
    uint64_t digests[BACKEND_MAXIMUM_CARDS];
    uint32_t card_count = 0;
    int first_argument = 1;
    if (argument_count > first_argument && strcmp(arguments[first_argument], break_option) == 0) {
        is_word_broken = argument_count > first_argument + 1 && strcmp(arguments[first_argument + 1], word_break) == 0;
        first_argument += BREAK_ARGUMENTS;
    }
    const int path_count = argument_count - first_argument;
    if ((first_argument != 1 && !is_word_broken) ||
	(is_word_broken && path_count == 0) || path_count == 1 ||
	path_count > 1 + STORAGE_MAXIMUM_COPIES) {
	    (void)fprintf(
		    stderr,
		    "usage: dense_linear_card_test [--break word] [<capture folder> <pack copy> ...]\n");
	    return TEST_WRONG_USAGE;
    }
    (void)alarm(TEST_TIME_LIMIT_SECONDS);
    if (path_count != 0) {
        const char *const *const copy_paths = (const char *const *)&arguments[first_argument + 1];
	result_status status =
		test_capture_open(arguments[first_argument], &sources.capture);
	const uint32_t copy_count = (uint32_t)(path_count - 1);
	status = status == STATUS_OK ?
			 storage_direct.reader_open(copy_paths, copy_count,
						    &sources.reader) :
			 status;
	status = status == STATUS_OK ?
			 storage_direct.reader_describe(sources.reader,
							&description) :
			 status;
	status = status == STATUS_OK ?
			 pack_open_entry(&storage_direct, sources.reader,
					 &sources.pack) :
			 status;
	sources.block_alignment_bytes = description.block_alignment_bytes;
	if (!test_linear_expect_status(status, STATUS_OK,
				       "the capture and the pack open") ||
	    !test_linear_expect(
		    description.block_alignment_bytes != 0 &&
			    BACKEND_STAGING_ALIGNMENT_BYTES %
					    description.block_alignment_bytes ==
				    0,
		    "the reader's blocks divide a staging page")) {
		test_capture_close(sources.capture);
		sources.capture = NULL;
	}
    }
    if (test_linear_expect_status(backend_cuda_value.card_count(&card_count),
				  STATUS_OK, "the cards are counted") &&
	test_linear_expect(card_count > 0 &&
				   card_count <= BACKEND_MAXIMUM_CARDS,
			   "the machine has cards for the test")) {
	    bool are_same = true;
	    for (uint32_t card_index = 0; card_index < card_count;
		 card_index++) {
		    digests[card_index] = card_run(card_index, &sources);
		    are_same = are_same && digests[card_index] == digests[0];
	    }
	    (void)printf("the synthetic cases gave %s bits on every card\n",
			 are_same ? "the same" : "NOT the same");
    }
    test_capture_close(sources.capture);
    pack_close(sources.pack);
    storage_direct.reader_close(sources.reader);
    if (test_linear_failures() != 0) {
        (void)fprintf(stderr, "dense_linear_card_test FAILED: %d failure lines\n", test_linear_failures());
        return TEST_FAILED;
    }
    (void)printf("dense_linear_card_test passed on %" PRIu32 " cards%s\n", card_count,
                 path_count == 0 ? " (replay: nothing compared, it needs a capture folder and a pack)"
                 : is_word_broken ? " (--break word: every product of the replay reported)" : "");
    return TEST_PASSED;
}
