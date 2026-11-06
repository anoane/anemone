/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * exl3_tc_forms_card_test.c: the tensor-core EXL3 MoE (exl3_tc.h) of a BATCH step's rows against the thin steps' calls
 * on one card, bit for bit, and times (track G, --batch_thin_bits). Synthetic EXL3 experts (EXPERT_SETS distinct sets
 * of random mul1 trellis words and scales of both signs, expert e taking set e % EXPERT_SETS, so a call streams most
 * of its weights from memory), each row six distinct experts of the 384 with weights. A case of R rows runs:
 *   - the thin reference: each 8-row slab of the rows as its own 8-row call (its job table of its rows: a thin step's);
 *   - exl3_tc_prepare_short over all R rows (the short forms at every row count): every output row must equal
 *     the reference's;
 *   - exl3_tc_prepare_paired over all R rows (the short forms' products two slices an item): every output row
 *     must equal the reference's;
 *   - exl3_tc_prepare over all R rows (the long forms from EXL3_TC_LONG_PAIRS pairs): its rows unlike
 *     the reference's are counted (expected from 27 rows on);
 * then the whole-row calls as graphs, sampled in turn on the card's clock (card_timing.h): microseconds a call of the
 * short forms against the paired and against the default forms (TIME lines), and each of the five launches alone
 * against its paired and default counterpart (LAUNCH lines: the same workspace, each launch's inputs in place).
 * Rows 8 (a C1 step's call), 16, 24, 32 and 48; intermediate 640 and 512; K3. It needs the cards free (gpu_run).
 */
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "anemone/backend_seam.h"
#include "anemone/exl3_tc.h"
#include "anemone/exl3_tc_records.h"
#include "anemone/linear_records.h"
#include "anemone/model_layout.h"
#include "card_timing.h"
#include "linear_card_fixture.h"

#define HIDDEN EXL3_TC_HIDDEN
#define EXPERTS EXL3_TCP_EXPERTS
#define PAIRS EXL3_TC_PAIRS_PER_ROW
#define SLAB_ROWS EXL3_TC_JOB_ROWS /* a thin step's rows */
#define MOST_ROWS 48u
#define EXPERT_SETS 96u
#define ACTIVATION_LIMIT 10.0f
#define WORK_CAPACITY (256ull << 20)
#define WEIGHTS_CAPACITY (384ull << 20)
#define HALF_ONE 0x3C00u
#define HALF_SIGN 0x8000u
#define HALF_MANTISSA_MASK 0x03FFu
#define BITS 3u
#define BITS_PER_BYTE 8u
#define HALF_BYTES 2u
#define FLOAT_BYTES 4u
#define WEIGHT_BASE 0.05f
#define WEIGHT_STEPS 1000u
#define WEIGHT_SCALE 2500.0f
#define SCALE_STREAM 0x5u
#define WEIGHT_STREAM 0x9u
#define STREAM_STRIDE 131u
#define ROW_STREAM_STRIDE 977u
#define DRAW_STRIDE 1000u
#define FORMS 3u /* over all rows: the short forms, the paired forms, the default forms */
enum { SHORT_FORM, PAIRED_FORM, DEFAULT_FORM };
#define WARM_RUNS 3u
#define TIME_SAMPLES 9u
#define TIME_NAME_BYTES 32u

struct test_case {
    uint32_t rows, inner;
};
/* 8 rows: a C1 step's call (the thin reference is the call itself; the paired forms against it) */
static const struct test_case cases[] = {{8, 640}, {8, 512}, {16, 640}, {24, 640}, {32, 640}, {48, 640}, {48, 512}};

static uint64_t matrix_bytes(uint64_t input, uint64_t output)
{
    return input * output * BITS / BITS_PER_BYTE + (input + output) * HALF_BYTES;
}

/* random trellis words; suh and svh FP16 of magnitude 1 to 2 and either sign */
static void matrix_fill(unsigned char *stored, uint64_t input, uint64_t output, uint64_t stream)
{
    const uint64_t trellis = input * output * BITS / BITS_PER_BYTE;
    for (uint64_t word = 0; word < trellis / HALF_BYTES; word++) {
        const uint16_t value = (uint16_t)test_linear_hash(stream, word);
        memcpy(stored + HALF_BYTES * word, &value, HALF_BYTES);
    }
    for (uint64_t value = 0; value < input + output; value++) {
        const uint64_t hash = test_linear_hash(stream ^ SCALE_STREAM, value);
        const uint16_t half =
            (uint16_t)(HALF_ONE | (hash & HALF_MANTISSA_MASK) | ((hash >> 16) & 1u ? HALF_SIGN : 0u));
        memcpy(stored + trellis + HALF_BYTES * value, &half, HALF_BYTES);
    }
}

/* each row's six distinct experts (ascending) and their weights */
static void picks_fill(uint32_t (*picks)[PAIRS], float (*weights)[PAIRS], uint32_t rows, uint64_t stream)
{
    for (uint32_t row = 0; row < rows; row++) {
        uint32_t count = 0;
        for (uint64_t draw = 0; count < PAIRS; draw++) {
            const uint32_t expert = (uint32_t)(test_linear_hash(stream, row * DRAW_STRIDE + draw) % EXPERTS);
            bool taken = false;
            for (uint32_t slot = 0; slot < count; slot++)
                taken = taken || picks[row][slot] == expert;
            if (!taken)
                picks[row][count++] = expert;
        }
        for (uint32_t first = 0; first < PAIRS; first++)
            for (uint32_t second = first + 1u; second < PAIRS; second++)
                if (picks[row][second] < picks[row][first]) {
                    const uint32_t swap = picks[row][first];
                    picks[row][first] = picks[row][second];
                    picks[row][second] = swap;
                }
        for (uint32_t slot = 0; slot < PAIRS; slot++)
            weights[row][slot] =
                WEIGHT_BASE +
                (float)(test_linear_hash(stream ^ WEIGHT_STREAM, row * PAIRS + slot) % WEIGHT_STEPS) / WEIGHT_SCALE;
    }
}

/* the router's job table of rows [first, first + rows) (router_records.h): the routed experts ascending, a run of an
 * expert's pairs cut in jobs of at most 8, then empty jobs; pair indices counted from the first row */
static void jobs_fill(struct expert_job_record *jobs, uint32_t capacity,
		      uint32_t first, uint32_t rows,
		      const uint32_t (*picks)[PAIRS],
		      const float (*weights)[PAIRS], const uint64_t *offsets,
		      uint64_t gate_up_bytes)
{
    memset(jobs, 0, sizeof(*jobs) * capacity);
    uint32_t job = 0;
    for (uint32_t expert = 0; expert < EXPERTS; expert++) {
        for (uint32_t row = 0; row < rows; row++)
            for (uint32_t slot = 0; slot < PAIRS; slot++) {
                if (picks[first + row][slot] != expert)
                    continue;
                if (jobs[job].row_count == SLAB_ROWS)
                    job++;
		struct expert_job_record *const at = &jobs[job];
		at->gate_offset = offsets[expert % EXPERT_SETS];
                at->up_offset = offsets[expert % EXPERT_SETS] + gate_up_bytes;
                at->down_offset = offsets[expert % EXPERT_SETS] + 2u * gate_up_bytes;
                at->pair_index[at->row_count] = row * PAIRS + slot;
                at->route_weight[at->row_count] = weights[first + row][slot];
                at->row_count++;
            }
        if (jobs[job].row_count != 0)
            job++;
    }
}

static uint32_t rows_unlike(const float *candidate, const float *reference, uint32_t rows)
{
    uint32_t unlike = 0;
    for (uint32_t row = 0; row < rows; row++)
        unlike += memcmp(candidate + (size_t)row * HIDDEN, reference + (size_t)row * HIDDEN, HIDDEN * FLOAT_BYTES) != 0
                      ? 1u
                      : 0u;
    return unlike;
}

static bool case_run(struct test_linear_card *card, const struct test_case *test, uint64_t stream)
{
    test_linear_context("rows %u, intermediate %u", test->rows, test->inner);
    test_linear_card_clear(card);
    const uint64_t gate_up = matrix_bytes(HIDDEN, test->inner);
    const uint64_t expert = 2u * gate_up + matrix_bytes(test->inner, HIDDEN);
    uint64_t sets[EXPERT_SETS];
    for (uint32_t set = 0; set < EXPERT_SETS; set++) {
	    const struct linear_range range =
		    test_linear_card_weights(card, expert);
	    unsigned char *const stored =
		    test_linear_card_writable(card, range);
	    if (stored == NULL)
		    return test_linear_expect(false, "room for the experts");
	    sets[set] = range.byte_offset;
	    const uint64_t base =
		    stream * STREAM_STRIDE + set * MODEL_EXPERT_MATRICES;
	    matrix_fill(stored, HIDDEN, test->inner, base);
	    matrix_fill(stored + gate_up, HIDDEN, test->inner, base + 1u);
	    matrix_fill(stored + 2u * gate_up, test->inner, HIDDEN, base + 2u);
    }
    static uint32_t picks[MOST_ROWS][PAIRS];
    static float weights[MOST_ROWS][PAIRS];
    static struct expert_job_record whole_jobs[MOST_ROWS * PAIRS],
	    slab_jobs[SLAB_ROWS * PAIRS];
    static uint16_t rows[MOST_ROWS * HIDDEN];
    static float outputs[FORMS + 1u][MOST_ROWS * HIDDEN]; /* the reference, then each form's */
    picks_fill(picks, weights, test->rows, stream);
    for (uint32_t row = 0; row < test->rows; row++)
        test_linear_values(rows + (size_t)row * HIDDEN, HIDDEN, stream * ROW_STREAM_STRIDE + row,
                           TEST_LINEAR_VALUES_PLAIN);
    const uint32_t capacity = test->rows * PAIRS, slab_capacity = SLAB_ROWS * PAIRS, slabs = test->rows / SLAB_ROWS;
    jobs_fill(whole_jobs, capacity, 0, test->rows, (const uint32_t(*)[PAIRS])picks,
              (const float(*)[PAIRS])weights, sets, gate_up);
    const struct linear_range hidden = test_linear_card_work(
	    card, (uint64_t)test->rows * HIDDEN * HALF_BYTES);
    const struct linear_range jobs =
	    test_linear_card_work(card, sizeof(*whole_jobs) * capacity);
    struct linear_range slab_job_ranges[MOST_ROWS / SLAB_ROWS];
    for (uint32_t slab = 0; slab < slabs; slab++)
        slab_job_ranges[slab] = test_linear_card_work(card, sizeof(*slab_jobs) * slab_capacity);
    const struct linear_range workspace = test_linear_card_work(
	    card, exl3_tc_workspace_bytes(capacity, test->inner));
    struct linear_range output_ranges[FORMS + 1u];
    for (uint32_t form = 0; form <= FORMS; form++)
        output_ranges[form] = test_linear_card_work(card, (uint64_t)test->rows * HIDDEN * FLOAT_BYTES);
    bool passed = test_linear_card_has_room(card) &&
                  test_linear_card_put(card, hidden, 0, rows, (uint64_t)test->rows * HIDDEN * HALF_BYTES) &&
                  test_linear_card_put(card, jobs, 0, whole_jobs, sizeof(*whole_jobs) * capacity);
    for (uint32_t slab = 0; slab < slabs && passed; slab++) {
        jobs_fill(slab_jobs, slab_capacity, slab * SLAB_ROWS, SLAB_ROWS, (const uint32_t(*)[PAIRS])picks,
                  (const float(*)[PAIRS])weights, sets, gate_up);
        passed = test_linear_card_put(card, slab_job_ranges[slab], 0, slab_jobs, sizeof(*slab_jobs) * slab_capacity);
    }
    passed = passed && test_linear_card_load(card);
    /* the thin reference: each slab's own 8-row call (the short forms by its pairs), its rows into the reference */
    for (uint32_t slab = 0; slab < slabs && passed; slab++) {
	    const struct exl3_tc_record thin = {
		    .jobs_offset = slab_job_ranges[slab].byte_offset,
		    .hidden_offset =
			    hidden.byte_offset +
			    (uint64_t)slab * SLAB_ROWS * HIDDEN * HALF_BYTES,
		    .workspace_offset = workspace.byte_offset,
		    .output_offset =
			    output_ranges[0].byte_offset +
			    (uint64_t)slab * SLAB_ROWS * HIDDEN * FLOAT_BYTES,
		    .row_count = SLAB_ROWS,
		    .job_capacity = slab_capacity,
		    .intermediate = test->inner,
		    .bits_x2 = 2u * BITS,
		    .activation_limit = ACTIVATION_LIMIT
	    };
	    struct kernel_launch launches[EXL3_TC_LAUNCHES];
	    passed = test_linear_expect_status(
			     exl3_tc_prepare(&thin, card->arena_byte_count,
					     &backend_cuda_value, card->card,
					     launches),
			     STATUS_OK, "a slab's thin call prepares") &&
		     test_linear_card_run(card, launches, EXL3_TC_LAUNCHES);
    }
    struct kernel_launch whole[FORMS][EXL3_TC_LAUNCHES];
    for (uint32_t form = 0; form < FORMS && passed; form++) {
	    const struct exl3_tc_record record = {
		    .jobs_offset = jobs.byte_offset,
		    .hidden_offset = hidden.byte_offset,
		    .workspace_offset = workspace.byte_offset,
		    .output_offset = output_ranges[form + 1u].byte_offset,
		    .row_count = test->rows,
		    .job_capacity = capacity,
		    .intermediate = test->inner,
		    .bits_x2 = 2u * BITS,
		    .activation_limit = ACTIVATION_LIMIT
	    };
	    const uint64_t bytes = card->arena_byte_count;
	    passed = test_linear_expect_status(
			     form == SHORT_FORM ?
				     exl3_tc_prepare_short(&record, bytes,
							   &backend_cuda_value,
							   card->card,
							   whole[form]) :
			     form == PAIRED_FORM ?
				     exl3_tc_prepare_paired(&record, bytes,
							    &backend_cuda_value,
							    card->card,
							    whole[form]) :
				     exl3_tc_prepare(&record, bytes,
						     &backend_cuda_value,
						     card->card, whole[form]),
			     STATUS_OK, "a whole-row call prepares") &&
		     test_linear_card_run(card, whole[form], EXL3_TC_LAUNCHES);
    }
    passed = passed && test_linear_card_fetch(card);
    for (uint32_t form = 0; form <= FORMS && passed; form++)
        passed = test_linear_card_get(card, output_ranges[form], 0, outputs[form],
                                      (uint64_t)test->rows * HIDDEN * FLOAT_BYTES);
    if (passed) {
        const uint32_t short_unlike = rows_unlike(outputs[1u + SHORT_FORM], outputs[0], test->rows);
        const uint32_t paired_unlike = rows_unlike(outputs[1u + PAIRED_FORM], outputs[0], test->rows);
        const uint32_t default_unlike = rows_unlike(outputs[1u + DEFAULT_FORM], outputs[0], test->rows);
	(void)printf(
		"rows %u intermediate %u: %u rows of the short forms, %u of the paired forms (%s) and %u of the "
		"default forms (%s) unlike the thin calls'\n",
		test->rows, test->inner, short_unlike, paired_unlike, "paired",
		default_unlike,
		test->rows * PAIRS >= EXL3_TC_LONG_PAIRS ? "long" : "short");
	passed = test_linear_expect(short_unlike == 0, "every row of the short forms is the thin calls', bit for bit");
        passed = test_linear_expect(paired_unlike == 0,
                                    "every row of the paired forms is the thin calls', bit for bit") && passed;
    }
    /* graph 0 of a form: its whole call; graph 1 + k: its launch k alone */
    static uint32_t graphs[FORMS][1u + EXL3_TC_LAUNCHES];
    for (uint32_t form = 0; form < FORMS && passed; form++) {
	    passed = test_linear_expect_status(
		    backend_cuda_value.graph_capture(card->card, whole[form],
						     EXL3_TC_LAUNCHES,
						     &graphs[form][0]),
		    STATUS_OK, "a call is captured as a graph");
	    for (uint32_t launch = 0; launch < EXL3_TC_LAUNCHES && passed;
		 launch++)
		    passed = test_linear_expect_status(
			    backend_cuda_value.graph_capture(
				    card->card, &whole[form][launch], 1,
				    &graphs[form][1u + launch]),
			    STATUS_OK, "a launch is captured as a graph");
    }
    static double samples[TEST_CARD_TIMING_FORMS * TIME_SAMPLES], savings[TIME_SAMPLES];
    static const char *const names[FORMS] = {"short", "paired", "default"};
    for (uint32_t part = 0; part <= EXL3_TC_LAUNCHES && passed; part++)
	    for (uint32_t form = PAIRED_FORM; form <= DEFAULT_FORM && passed;
		 form++) {
		    const struct test_card_timing_form
			    timed[TEST_CARD_TIMING_FORMS] = {
				    { &graphs[SHORT_FORM][part], 1 },
				    { &graphs[form][part], 1 }
			    };
		    if (!test_card_timing_pair(card->card, timed, WARM_RUNS, 1,
					       TIME_SAMPLES, samples, savings))
			    break;
		    char saving[TIME_NAME_BYTES];
		    (void)snprintf(saving, sizeof(saving), "%s_saving",
				   names[form]);
		    if (part == 0)
			    (void)printf("TIME rows=%u intermediate=%u",
					 test->rows, test->inner);
		    else
			    (void)printf(
				    "LAUNCH rows=%u intermediate=%u launch=%u",
				    test->rows, test->inner, part - 1u);
		    test_card_timing_print(samples, TIME_SAMPLES, 1.0,
					   names[SHORT_FORM]);
		    test_card_timing_print(samples + TIME_SAMPLES, TIME_SAMPLES,
					   1.0, names[form]);
		    test_card_timing_print(savings, TIME_SAMPLES, 1.0, saving);
		    (void)printf("\n");
	    }
    return passed;
}

int main(void)
{
    struct test_linear_card card;
    if (!test_linear_card_open(&card, 0, WORK_CAPACITY, WEIGHTS_CAPACITY))
        return EXIT_FAILURE;
    for (uint32_t index = 0; index < sizeof(cases) / sizeof(cases[0]); index++)
        (void)case_run(&card, &cases[index], 61u + index);
    test_linear_card_close(&card);
    (void)printf("exl3_tc_forms_card_test: %d failures\n", test_linear_failures());
    return test_linear_failures() == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
