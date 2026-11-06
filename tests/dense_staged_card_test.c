/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * dense_staged_card_test.c: the staged wide dense form (--dense_staged, dense_linear_records.h) against the wide form
 * the card's class takes without it, on one card: every output byte the same, and times (TIME lines: microseconds a
 * launch of each form as a graph, sampled in turn on the card's clock, card_timing.h). The model's wide dense products
 * at their shapes: the compressor's wkv and wgate (512 outputs, 5,120 inputs, float32 out), the indexer's weights (32
 * outputs), the keys (128 outputs of 512 inputs, bfloat16 out) and the routers (384 outputs), at a tail's 448 rows, a
 * split half's 1,024 and a chunk's 2,048. Random bfloat16 rows and weights (linear_card_fixture.h's plain values);
 * dense_linear.c takes the staged form for bfloat16 weights alone (rk06_card: float32 weights' routers 2-4% slower
 * staged). It needs the cards free (gpu_run).
 */
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "anemone/backend_seam.h"
#include "anemone/dense_linear.h"
#include "anemone/dense_linear_records.h"
#include "anemone/linear_records.h"
#include "card_timing.h"
#include "linear_card_fixture.h"

#define WORK_CAPACITY (192ull << 20)
#define WEIGHTS_CAPACITY (32ull << 20)
#define MOST_ROWS 2048u
#define MOST_INPUTS 5120u
#define MOST_OUTPUTS 512u
#define BFLOAT16_BYTES 2u
#define FLOAT_BYTES 4u
#define WEIGHT_STREAM 0x51u
#define ROW_STREAM 0x77u
#define FLOAT_MANTISSA_BITS 23u
#define FLOAT_WEIGHT_EXPONENT 0x3C000000u /* a float32 weight of magnitude 2^-7 to 2^-6, either sign */
#define FLOAT_SIGN 0x80000000u
#define WARM_RUNS 3u
#define TIME_SAMPLES 9u
#define FIRST_STREAM 17u /* the cases' streams from here on */
enum { WIDE_FORM, STAGED_FORM, FORMS };

struct dense_case {
    const char *name;
    uint32_t outputs, inputs, rows, weight_type, output_type;
};
static const struct dense_case cases[] = {
	{ "compressor", 512, 5120, 448, DENSE_BFLOAT16, DENSE_FLOAT32 },
	{ "compressor", 512, 5120, 1024, DENSE_BFLOAT16, DENSE_FLOAT32 },
	{ "compressor", 512, 5120, 2048, DENSE_BFLOAT16, DENSE_FLOAT32 },
	{ "index_weights", 32, 5120, 1024, DENSE_BFLOAT16, DENSE_BFLOAT16 },
	{ "keys", 128, 512, 1024, DENSE_BFLOAT16, DENSE_BFLOAT16 },
	{ "router", 384, 5120, 448, DENSE_BFLOAT16, DENSE_FLOAT32 },
	{ "router", 384, 5120, 1024, DENSE_BFLOAT16, DENSE_FLOAT32 },
	{ "router", 384, 5120, 2048, DENSE_BFLOAT16, DENSE_FLOAT32 },
};

static bool case_run(struct test_linear_card *card,
		     const struct dense_linear_kernels kernels[FORMS],
		     const struct dense_case *dense, uint64_t stream,
		     uint32_t staged_outputs)
{
    test_linear_context("%s, %u outputs, %u inputs, %u rows", dense->name, dense->outputs, dense->inputs, dense->rows);
    test_linear_card_clear(card);
    const uint64_t weight_bytes = (uint64_t)dense->outputs * dense->inputs *
				  (dense->weight_type == DENSE_FLOAT32 ?
					   FLOAT_BYTES :
					   BFLOAT16_BYTES);
    const uint64_t row_bytes = (uint64_t)dense->rows * dense->inputs * BFLOAT16_BYTES;
    const uint64_t output_bytes = (uint64_t)dense->rows * dense->outputs *
				  (dense->output_type == DENSE_FLOAT32 ?
					   FLOAT_BYTES :
					   BFLOAT16_BYTES);
    const struct linear_range weights =
	    test_linear_card_weights(card, weight_bytes);
    unsigned char *const stored = test_linear_card_writable(card, weights);
    if (stored == NULL)
        return test_linear_expect(false, "room for the weights");
    if (dense->weight_type == DENSE_FLOAT32) {
	    for (uint64_t index = 0; index < weight_bytes / FLOAT_BYTES;
		 index++) {
		    const uint64_t hash =
			    test_linear_hash(stream ^ WEIGHT_STREAM, index);
		    const uint32_t bits =
			    FLOAT_WEIGHT_EXPONENT |
			    (uint32_t)(hash &
				       ((1u << FLOAT_MANTISSA_BITS) - 1u)) |
			    ((hash >> 32) & 1u ? FLOAT_SIGN : 0u);
		    memcpy(stored + index * FLOAT_BYTES, &bits, FLOAT_BYTES);
	    }
    } else {
	    for (uint32_t output = 0; output < dense->outputs; output++)
		    test_linear_values(
			    (uint16_t *)(void *)(stored +
						 (uint64_t)output *
							 dense->inputs *
							 BFLOAT16_BYTES),
			    dense->inputs,
			    (stream ^ WEIGHT_STREAM) * MOST_OUTPUTS + output,
			    TEST_LINEAR_VALUES_PLAIN);
    }
    static uint16_t rows[MOST_ROWS * MOST_INPUTS];
    for (uint32_t row = 0; row < dense->rows; row++)
        test_linear_values(rows + (size_t)row * dense->inputs, dense->inputs, (stream ^ ROW_STREAM) * MOST_ROWS + row,
                           TEST_LINEAR_VALUES_PLAIN);
    const struct linear_range values = test_linear_card_work(card, row_bytes);
    struct linear_range outputs[FORMS];
    for (uint32_t form = 0; form < FORMS; form++)
        outputs[form] = test_linear_card_work(card, output_bytes);
    bool passed = test_linear_card_has_room(card) && test_linear_card_put(card, values, 0, rows, row_bytes) &&
                  test_linear_card_load(card);
    struct kernel_launch launches[FORMS];
    for (uint32_t form = 0; form < FORMS && passed; form++) {
	    const struct dense_linear_description description = {
		    .weights = weights,
		    .rows = values,
		    .output = outputs[form],
		    .step = { 0, 0 },
		    .output_size = dense->outputs,
		    .input_size = dense->inputs,
		    .row_count = dense->rows,
		    .weight_type = dense->weight_type,
		    .output_type = dense->output_type,
		    .flags = 0,
		    .output_scale = 1.0f,
		    .epsilon = 0.0f,
		    .cone_layer = 0
	    };
	    passed = test_linear_expect_status(
			     dense_linear_prepare(
				     &description, card->arena_byte_count,
				     &kernels[form], &launches[form]),
			     STATUS_OK, "the product prepares") &&
		     test_linear_card_run(card, &launches[form], 1);
    }
    const bool staged =
        passed && launches[STAGED_FORM].kernel == kernels[STAGED_FORM].staged_products[dense->weight_type];
    passed =
	    passed &&
	    test_linear_expect(
		    staged == (dense->outputs % staged_outputs == 0),
		    "whole staged blocks take the product; other outputs retain the wide form") &&
	    test_linear_card_fetch(card) &&
	    test_linear_card_expect_same(
		    card, outputs[WIDE_FORM], 0, outputs[STAGED_FORM], 0,
		    output_bytes,
		    "every output byte of the staged form is the wide form's");
    uint32_t graphs[FORMS];
    for (uint32_t form = 0; form < FORMS && passed; form++)
	    passed = test_linear_expect_status(
		    backend_cuda_value.graph_capture(
			    card->card, &launches[form], 1, &graphs[form]),
		    STATUS_OK, "a launch is captured as a graph");
    static double samples[TEST_CARD_TIMING_FORMS * TIME_SAMPLES], savings[TIME_SAMPLES];
    const struct test_card_timing_form timed[TEST_CARD_TIMING_FORMS] = {{&graphs[WIDE_FORM], 1},
                                                                       {&graphs[STAGED_FORM], 1}};
    if (passed && test_card_timing_pair(card->card, timed, WARM_RUNS, 1, TIME_SAMPLES, samples, savings)) {
	    (void)printf(
		    "TIME %s block_outputs=%u outputs=%u inputs=%u rows=%u",
		    dense->name, staged_outputs, dense->outputs, dense->inputs,
		    dense->rows);
	    test_card_timing_print(samples, TIME_SAMPLES, 1.0, "wide");
	    test_card_timing_print(samples + TIME_SAMPLES, TIME_SAMPLES, 1.0,
				   "staged");
	    test_card_timing_print(savings, TIME_SAMPLES, 1.0, "staged_saving");
	    (void)printf("\n");
    }
    return passed;
}

int main(void)
{
    struct test_linear_card card;
    if (!test_linear_card_open(&card, 0, WORK_CAPACITY, WEIGHTS_CAPACITY))
        return EXIT_FAILURE;
    struct dense_linear_kernels kernels[FORMS];
    bool ready = true;
    for (uint32_t form = 0; form < FORMS && ready; form++)
	    ready = test_linear_expect_status(
			    dense_linear_kernels_prepare(&backend_cuda_value,
							 card.card,
							 &kernels[form]),
			    STATUS_OK, "the dense kernels prepare") &&
		    test_linear_expect_status(
			    dense_linear_wide_kernels_prepare(
				    &backend_cuda_value, card.card,
				    ARITHMETIC_CLASS_SM_80, &kernels[form]),
			    STATUS_OK, "the wide dense kernels prepare");
    static const uint32_t block_outputs[] = { 8, 16, 32, 64 };
    for (uint32_t form = 0;
	 ready && form < sizeof(block_outputs) / sizeof(block_outputs[0]);
	 form++) {
	    const uint32_t outputs = block_outputs[form];
	    ready = test_linear_expect_status(
		    dense_linear_staged_kernels_prepare(&backend_cuda_value,
							card.card, outputs,
							&kernels[STAGED_FORM]),
		    STATUS_OK, "the staged forms prepare");
	    for (uint32_t index = 0;
		 ready && index < sizeof(cases) / sizeof(cases[0]); index++)
		    (void)case_run(&card, kernels, &cases[index],
				   FIRST_STREAM + index, outputs);
    }
    test_linear_card_close(&card);
    (void)printf("dense_staged_card_test: %d failures\n", test_linear_failures());
    return test_linear_failures() == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
