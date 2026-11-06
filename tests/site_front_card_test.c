/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * site_front_card_test: the fused fronts of a block's sites (kernel_site_front.cu: the site front and the FFN front)
 * on the real cards, against the launches they stand for, run on the same image: every output byte equal.
 *
 *     site_front_card_test [--time]
 *
 * On every card:
 *   - 1 to 8 rows; the attention site (today: the mix project, the mix finish and the quantization of its input; fused:
 *     the site front) and the FFN site (today: those and the router's dense product and select, between the finish and
 *     the quantization, as model_v41_program.c launches them; fused: the FFN front), with live rows all and one fewer
 *     and in both orders: today's launches, then the fused one from the same image. Every byte of post, comb, the
 *     carried pre, hidden, codes, exponents, operands, scores, affinities and the job table equal (all of each range,
 *     rubbish where neither writes), and today's mixes array the fused launch's scratch mixes; the scratch's counters
 *     zero again; the guards and the ranges only read untouched.
 *   - The order of the chains, on the host: every warp sum of the mix project the fused launch published in its
 *     scratch, and every score it wrote, are the host's fmaf chains in the dense product's order (logical thread t:
 *     chunks t, t + 256, ...; a warp by exchanges 16 down to 1; the warps from +0 in order), bit for bit; the same
 *     warp sums added in descending order differ in some mixes and scores, so that a launch that changed the order
 *     would fail.
 *   - Negative controls, which must fail: the top mantissa bit of one weight of the mix matrix, and of one of the
 *     router's, flipped on the card for the fused launch alone: its outputs must then differ from today's.
 *   - --time: the median (and the least) of ROUNDS measurements (after a warm one; L2 is not flushed) of
 *     GRAPH_REPLAYS replays of a graph of a site's launches, today's and fused, at 1 and 8 rows: COPIES copies that
 *     read the same weights (hot), or one copy for each of the site's own copies of its weights, which a replay reads
 *     in turn (cold: about COLD_CYCLE_BYTES a pass, three times the PRO's L2 and nearly ten times a CMP's); and of
 *     each launch alone (hot). Microseconds a copy, host clock around the queue and the wait.
 * It FAILS on any other difference, and when a card has no room.
 */
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "anemone/backend_seam.h"
#include "anemone/dense_linear.h"
#include "anemone/hyper_connection.h"
#include "anemone/linear.h"
#include "anemone/model_layout.h"
#include "anemone/router.h"
#include "anemone/site_front.h"
#include "anemone/status.h"
#include "anemone/step_records.h"
#include "linear_card_fixture.h"
#include "linear_reference.h"

#define COUNT_OF(table) (sizeof(table) / sizeof((table)[0]))
#define TEST_TIME_LIMIT_SECONDS 3600
#define ROWS LINEAR_MAXIMUM_ROWS
#define HIDDEN MODEL_HIDDEN
#define STREAMS MODEL_STREAMS
#define MIXES MODEL_MIXES
#define EXPERTS MODEL_EXPERTS
#define PICKS MODEL_EXPERTS_PER_ROW
#define SCALES HYPER_SCALES
#define MIX_INPUTS ((uint32_t)SITE_FRONT_MIX_INPUTS)
#define STREAM_VALUES (STREAMS * HIDDEN)
#define TABLE_ENTRIES (EXPERTS * MODEL_EXPERT_MATRICES)
#define CHUNK_VALUES DENSE_CHUNK_INPUTS
#define DENSE_THREADS DENSE_THREADS_VALUE
#define WARP_LANES LINEAR_THREADS_PER_WARP
#define DENSE_WARPS (DENSE_THREADS / WARP_LANES)
#define WORK_CAPACITY (4ull << 20u)             /* far above a case's ranges and guards */
/* Cold weights: each site's own copies in turn, enough that one pass over them reads three times the largest L2 (the
 * PRO's 128 MB; a CMP's 40 MB): a run then reads weights the runs before it evicted, as a decode step does. */
#define COLD_CYCLE_BYTES (384ull << 20u)
#define SITE_WEIGHT_BYTES /* a site's mix weights, scale, base and norm */ \
	((uint64_t)MODEL_MIXES * MIX_INPUTS * sizeof(float) +              \
	 HYPER_SCALES * sizeof(float) + MODEL_MIXES * sizeof(float) +      \
	 MODEL_HIDDEN * sizeof(uint16_t))
#define ROUTER_WEIGHT_BYTES \
	((uint64_t)MODEL_EXPERTS * MODEL_HIDDEN * sizeof(uint16_t))
#define COLD_COPIES(bytes) ((uint32_t)((COLD_CYCLE_BYTES + (bytes) - 1u) / (bytes)))
#define ATTENTION_COLD_COPIES COLD_COPIES(SITE_WEIGHT_BYTES)
#define FFN_COLD_COPIES COLD_COPIES(SITE_WEIGHT_BYTES + ROUTER_WEIGHT_BYTES)   /* the FFN site's own and the router's */
#define MOST_WEIGHT_COPIES ATTENTION_COLD_COPIES  /* of a site: the attention's, of the fewer bytes */
#define WEIGHTS_SLACK_BYTES (32ull << 20u)      /* the ranges' alignment, the bias and the experts' table */
#define WEIGHTS_CAPACITY (COLD_CYCLE_BYTES + WEIGHTS_SLACK_BYTES)  /* with --time: one site's cold copies at a time */
#define CHECK_WEIGHTS_CAPACITY (12ull << 20u)   /* without: one copy of the weights */
/* The experts of the router's job table: matrices of 64 by 64 at 4 bits, every entry of the table its own place, 16
 * bytes after the one before, in a range that holds the last one whole (router_card_test's). */
#define SMALL_SIZE 64u
#define E2M1_BITS_PER_WEIGHT 4u
#define SMALL_MATRIX_BYTES 2176u
#define EXPERT_WEIGHTS_BYTES \
	(SMALL_MATRIX_BYTES + TABLE_ENTRIES * LINEAR_OFFSET_ALIGNMENT_BYTES)
#define RUBBISH_BYTE 0x5A
#define SEED 0x53465254u
#define HASH_UNIT 0x1p-64
#define HALF 0.5
#define MIX_WEIGHT_SPAN 0.125                   /* a mix weight from -1/16 to 1/16 */
#define ROUTER_WEIGHT_SPAN 0.0625               /* a router weight from -1/32 to 1/32: scores about +-3 */
#define MIX_SPREAD 2.0                          /* bases from -1 to 1 */
#define SCALE_SPREAD 3.0                        /* scales from 0 to 3 */
#define PRE_LEAST 0.25                          /* carried pre from 0.25 to 1.75 */
#define PRE_SPREAD 1.5
#define BIAS_SPAN 1.0                           /* a bias from -0.5 to 0.5 */
#define TOLD_APART_ROW 2u                       /* one large value among small ones (hyper_connection_card_test's) */
#define TOLD_APART_LARGE 1024.0
#define TOLD_APART_LEAST 0.18
#define TOLD_APART_SPREAD 0.06
#define FLOAT32_TOP_MANTISSA_BIT 0x00400000u    /* the negative controls' bit */
#define BFLOAT16_TOP_MANTISSA_BIT 0x0040u
#define BROKEN_MIX 7u                           /* the mix and the expert whose weight a negative control changes */
#define BROKEN_EXPERT 101u
#define STEP_SEQUENCE 1u
#define COPIES 64u                              /* copies of a site's launches in one graph */
#define GRAPH_REPLAYS 8u                        /* replays queued back to back for one measurement */
#define ROUNDS 31u                              /* measurements of one graph: their median and least */
#define MEDIAN_ROUND 15u                        /* of ROUNDS sorted */
#define MOST_SITE_LAUNCHES 5u                   /* today's FFN site: mix, finish, router, select, quantization */
#define NANOSECONDS_PER_SECOND 1000000000ull
#define NANOSECONDS_PER_MICROSECOND 1000.0
#define FAILURES_SHOWN 8u
#define BITS_PER_HALF 16u
#define LIVE_KINDS 2u                           /* live rows: all, and one fewer */
#define ARGUMENTS_WITH_TIME 2                   /* the program's name and --time */

enum test_status { TEST_PASSED = 0, TEST_FAILED = 1, TEST_WRONG_USAGE = 2 };
enum site { ATTENTION_SITE, FFN_SITE, SITES };
enum weights_kind { HOT_WEIGHTS, COLD_WEIGHTS, WEIGHTS_KINDS };
enum output_array { MIXES_OUT, POST_OUT, COMB_OUT, PRE_OUT, HIDDEN_OUT, CODES_OUT, EXPONENTS_OUT, OPERANDS_OUT,
                    SCORES_OUT, AFFINITIES_OUT, JOBS_OUT, OUTPUT_ARRAYS };
static const char *const site_names[SITES] = {"attention", "FFN"};
static const char *const output_names[OUTPUT_ARRAYS] = {"mixes", "post", "comb", "carried pre", "hidden", "codes",
                                                        "exponents", "operands", "scores", "affinities", "jobs"};
static const uint32_t timed_rows[] = {1u, 8u};

/* One copy of the weights a site front reads, and the router's. */
struct site_weights {
	struct linear_range mix_weights, scale, base, norm;
};
struct weights_copy {
    struct site_weights sites[SITES];
    struct linear_range router_weights;
};
/* Where a card's case lies: the work ranges for ROWS rows and copy_counts[site] copies of each site's weights (the
 * FFN site's with the router's): copy c holds a site's where c is below its count. */
struct layout {
	struct linear_range streams, carried, mixes, post, comb, hidden, output,
		codes, exponents, operands;
	struct linear_range scores, affinities, jobs, step, offsets, scratch,
		bias, experts;
	struct weights_copy copies[MOST_WEIGHT_COPIES];
	uint32_t copy_counts[SITES];
};
/* The launches of one case: today's, with their count, the fused one, and the site front cut in two (the input front,
 * then the coefficient front; of the attention site). */
enum split_launch { INPUT_LAUNCH, COEFFICIENT_LAUNCH, SPLIT_LAUNCHES };
struct case_launches {
	struct kernel_launch today[MOST_SITE_LAUNCHES], fused,
		split[SPLIT_LAUNCHES];
	uint32_t today_count;
};
/* The kernels of one card. */
struct card_kernels {
	struct hyper_kernels_record hyper;
	struct dense_linear_kernels dense;
	struct router_kernels_record router;
	struct site_front_kernels front;
};

static uint16_t streams_in[ROWS][STREAM_VALUES];
static float carried_in[ROWS][STREAMS];
static float mix_weights_in[SITES][MIXES][MIX_INPUTS];
static float scale_in[SITES][SCALES];
static float base_in[SITES][MIXES];
static uint16_t norm_in[SITES][HIDDEN];
static uint16_t router_weights_in[EXPERTS][HIDDEN];
static float bias_in[EXPERTS];
static uint64_t expert_offsets[TABLE_ENTRIES];
static unsigned char expected_work[WORK_CAPACITY];      /* the work region after today's launches */
static struct kernel_launch
	timed_launches[MOST_WEIGHT_COPIES * MOST_SITE_LAUNCHES];
static const struct linear_range no_range = { 0, 0 };
static bool is_timed = false;
static uint32_t control_failures = 0;               /* differences the negative controls found, as they must */

static double fraction_of(uint64_t stream, uint64_t position)
{
    return (double)test_linear_hash(stream, position) * HASH_UNIT;
}

static uint32_t bits_of(float value)
{
    uint32_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static float float_of(uint32_t bits)
{
    float value = 0.0f;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static float bfloat16_float(uint16_t bits)
{
    return float_of((uint32_t)bits << BITS_PER_HALF);
}

static uint64_t clock_nanoseconds(void)
{
    struct timespec now = {0};
    return clock_gettime(CLOCK_MONOTONIC, &now) == 0
               ? (uint64_t)now.tv_sec * NANOSECONDS_PER_SECOND + (uint64_t)now.tv_nsec : 0;
}

/* ---- the inputs and the layout ---- */

static void inputs_make(void)
{
    for (uint32_t row = 0; row < ROWS; row++) {
        const uint64_t stream = SEED + row + 1u;
        test_linear_values(streams_in[row], STREAM_VALUES, stream,
                           (row & 1u) == 0 ? TEST_LINEAR_VALUES_PLAIN : TEST_LINEAR_VALUES_SMALL);
        for (uint32_t index = 0; index < STREAMS; index++) {
            carried_in[row][index] = (float)(PRE_LEAST + fraction_of(stream, index) * PRE_SPREAD);
        }
    }
    for (uint32_t index = 0; index < HIDDEN; index++) {
        const double small = TOLD_APART_LEAST + fraction_of(SEED + TOLD_APART_ROW, index) * TOLD_APART_SPREAD;
        streams_in[TOLD_APART_ROW][index] = test_linear_bfloat16_nearest(index == 0 ? TOLD_APART_LARGE : small);
    }
    for (uint32_t index = 0; index < STREAMS; index++) {
        carried_in[TOLD_APART_ROW][index] = index == 0 ? 1.0f : 0.0f;
    }
    for (uint32_t site = 0; site < SITES; site++) {
        const uint64_t stream = (uint64_t)SEED * (site + 1u);
        for (uint32_t mix = 0; mix < MIXES; mix++) {
            for (uint32_t input = 0; input < MIX_INPUTS; input++) {
                mix_weights_in[site][mix][input] =
                    (float)((fraction_of(stream, (uint64_t)mix * MIX_INPUTS + input) - HALF) * MIX_WEIGHT_SPAN);
            }
            base_in[site][mix] = (float)((fraction_of(stream + 1u, mix) - HALF) * MIX_SPREAD);
        }
        for (uint32_t index = 0; index < SCALES; index++) {
            scale_in[site][index] = (float)(fraction_of(stream + 1u, MIXES + index) * SCALE_SPREAD);
        }
        test_linear_values(norm_in[site], HIDDEN, stream + SITES, TEST_LINEAR_VALUES_PLAIN);
    }
    for (uint32_t expert = 0; expert < EXPERTS; expert++) {
        for (uint32_t input = 0; input < HIDDEN; input++) {
            router_weights_in[expert][input] = test_linear_bfloat16_nearest(
                (fraction_of(SEED + EXPERTS, (uint64_t)expert * HIDDEN + input) - HALF) * ROUTER_WEIGHT_SPAN);
        }
        bias_in[expert] = (float)((fraction_of(SEED + EXPERTS + 1u, expert) - HALF) * BIAS_SPAN);
    }
}

/* Sets the ranges aside, the copies of each site's weights among them (the attention site's attention_copies, the FFN
 * site's ffn_copies), and lays the inputs out in the image. */
static bool layout_make(struct test_linear_card *card, struct layout *layout, uint32_t attention_copies,
                        uint32_t ffn_copies)
{
    const uint64_t rows = ROWS;
    const uint32_t copy_count = attention_copies > ffn_copies ? attention_copies : ffn_copies;
    test_linear_card_clear(card);
    layout->copy_counts[ATTENTION_SITE] = attention_copies;
    layout->copy_counts[FFN_SITE] = ffn_copies;
    layout->streams =
	    test_linear_card_work(card, rows * MODEL_STREAMS_ROW_BYTES);
    layout->carried = test_linear_card_work(card, rows * MODEL_SITE_ROW_BYTES);
    layout->mixes = test_linear_card_work(card, rows * MODEL_MIXES_ROW_BYTES);
    layout->post = test_linear_card_work(card, rows * MODEL_SITE_ROW_BYTES);
    layout->comb = test_linear_card_work(card, rows * MODEL_COMB_ROW_BYTES);
    layout->hidden = test_linear_card_work(card, rows * MODEL_HIDDEN_ROW_BYTES);
    layout->output = test_linear_card_work(card, rows * MODEL_HIDDEN_ROW_BYTES);
    layout->codes =
	    test_linear_card_work(card, linear_codes_byte_count(rows, HIDDEN));
    layout->exponents = test_linear_card_work(
	    card, linear_exponents_byte_count(rows, HIDDEN));
    layout->operands =
	    test_linear_card_work(card, linear_values_byte_count(rows, HIDDEN));
    layout->scores = test_linear_card_work(card, rows * MODEL_ROUTER_ROW_BYTES);
    layout->affinities =
	    test_linear_card_work(card, rows * MODEL_ROUTER_ROW_BYTES);
    layout->jobs =
	    test_linear_card_work(card, linear_jobs_byte_count(rows * PICKS));
    layout->step = test_linear_card_work(
	    card, STEP_BLOCK_BYTES_VALUE + rows * STEP_ROW_BYTES_VALUE);
    layout->offsets = test_linear_card_work(card, MODEL_EXPERT_OFFSETS_BYTES);
    layout->scratch = test_linear_card_work(card, SITE_FRONT_SCRATCH_BYTES);
    layout->bias = test_linear_card_weights(card, MODEL_ROUTER_ROW_BYTES);
    layout->experts = test_linear_card_weights(card, EXPERT_WEIGHTS_BYTES);
    for (uint32_t copy = 0; copy < copy_count; copy++) {
        struct weights_copy *const weights = &layout->copies[copy];
        for (uint32_t site = 0; site < SITES; site++) {
            const bool is_laid = copy < layout->copy_counts[site];    /* a range of no bytes past a site's copies */
            weights->sites[site].mix_weights = is_laid ? test_linear_card_weights(card, sizeof(mix_weights_in[site]))
                                                       : no_range;
            weights->sites[site].scale = is_laid ? test_linear_card_weights(card, sizeof(scale_in[site])) : no_range;
            weights->sites[site].base = is_laid ? test_linear_card_weights(card, sizeof(base_in[site])) : no_range;
            weights->sites[site].norm = is_laid ? test_linear_card_weights(card, sizeof(norm_in[site])) : no_range;
        }
        weights->router_weights = copy < ffn_copies ? test_linear_card_weights(card, sizeof(router_weights_in))
                                                    : no_range;
    }
    for (uint32_t entry = 0; entry < TABLE_ENTRIES; entry++) {
	    expert_offsets[entry] =
		    layout->experts.byte_offset +
		    (uint64_t)entry * LINEAR_OFFSET_ALIGNMENT_BYTES;
    }
    bool is_laid_out = test_linear_card_has_room(card) &&
                       test_linear_card_put(card, layout->bias, 0, bias_in, sizeof(bias_in)) &&
                       test_linear_card_put(card, layout->offsets, 0, expert_offsets, sizeof(expert_offsets));
    for (uint32_t copy = 0; is_laid_out && copy < copy_count; copy++) {
        const struct weights_copy *const weights = &layout->copies[copy];
        for (uint32_t site = 0; is_laid_out && site < SITES; site++) {
            const struct site_weights *const own = &weights->sites[site];
            is_laid_out = copy >= layout->copy_counts[site] ||
                          (test_linear_card_put(card, own->mix_weights, 0, mix_weights_in[site],
                                                sizeof(mix_weights_in[site])) &&
                           test_linear_card_put(card, own->scale, 0, scale_in[site], sizeof(scale_in[site])) &&
                           test_linear_card_put(card, own->base, 0, base_in[site], sizeof(base_in[site])) &&
                           test_linear_card_put(card, own->norm, 0, norm_in[site], sizeof(norm_in[site])));
        }
        is_laid_out = is_laid_out && (copy >= ffn_copies || test_linear_card_put(card, weights->router_weights, 0,
                                                                                 router_weights_in,
                                                                                 sizeof(router_weights_in)));
    }
    return is_laid_out;
}

/* The work inputs of a case in the image: streams, carried pre, the step block of live_rows, rubbish in every output
 * range (what neither launch writes stays rubbish in both), zero counters. */
static bool work_put(struct test_linear_card *card, const struct layout *layout, uint32_t live_rows)
{
	static unsigned char rubbish[ROWS * MODEL_STREAMS_ROW_BYTES];
	static unsigned char
		step[STEP_BLOCK_BYTES_VALUE + ROWS * STEP_ROW_BYTES_VALUE];
	const struct step_block block = { STEP_SEQUENCE, live_rows, 0 };
	const struct linear_range outputs[] = {
		layout->mixes,	  layout->post,	  layout->comb,
		layout->hidden,	  layout->codes,  layout->exponents,
		layout->operands, layout->scores, layout->affinities,
		layout->jobs,	  layout->output
	};
	memset(rubbish, RUBBISH_BYTE, sizeof(rubbish));
	memset(step, 0, sizeof(step));
	memcpy(step, &block, sizeof(block));
	for (uint32_t row = 0; row < ROWS; row++) {
		const struct step_row_record step_row = {
			.position = (int32_t)row,
			.token_id = 0,
			.session_slot = 0,
			.flags = STEP_ROW_LOGITS
		};
		memcpy(&step[STEP_BLOCK_BYTES_VALUE +
			     row * STEP_ROW_BYTES_VALUE],
		       &step_row, sizeof(step_row));
	}
    bool is_put = test_linear_card_put(card, layout->streams, 0, streams_in, sizeof(streams_in)) &&
                  test_linear_card_put(card, layout->carried, 0, carried_in, sizeof(carried_in)) &&
                  test_linear_card_put(card, layout->step, 0, step, sizeof(step));
    for (size_t index = 0; is_put && index < COUNT_OF(outputs); index++) {
        is_put = test_linear_card_put(card, outputs[index], 0, rubbish, outputs[index].byte_count);
    }
    return is_put;
}

/* ---- the launches ---- */

/* The router's description (router.h) for row_count rows, with the experts' description it points at. */
static void router_describe(const struct layout *layout, uint32_t row_count,
			    bool is_reference_order,
			    struct linear_routed_experts_description *experts,
			    struct router_description *router)
{
	*experts = (struct linear_routed_experts_description){
		.jobs = { layout->jobs.byte_offset,
			  linear_jobs_byte_count((uint64_t)row_count * PICKS) },
		.expert_weights = layout->experts,
		.gate_up_shape = { SMALL_SIZE, SMALL_SIZE, E2M1_BITS_PER_WEIGHT,
				   0 },
		.down_shape = { SMALL_SIZE, SMALL_SIZE, E2M1_BITS_PER_WEIGHT,
				0 },
		.weight_format = WEIGHT_FORMAT_E2M1_SCALED,
		.job_count = row_count * PICKS,
		.row_count = row_count,
		.pairs_per_row = PICKS
	};
	*router = (struct router_description){
		.experts = experts,
		.expert_offsets = expert_offsets,
		.step = layout->step,
		.scores = layout->scores,
		.bias = layout->bias,
		.offsets = layout->offsets,
		.affinities = layout->affinities,
		.order = is_reference_order ? LINEAR_ORDER_REFERENCE :
					      LINEAR_ORDER_FIXED,
		.expert_count = MODEL_EXPERTS,
		.pick_count = MODEL_EXPERTS_PER_ROW
	};
}

/* Today's launches of a site, in the program's order, and the fused one, of the weights of copy `copy`. */
static bool launches_prepare(const struct test_linear_card *card, const struct card_kernels *kernels,
                             const struct layout *layout, enum site site, uint32_t row_count, bool is_reference_order,
                             uint32_t copy, struct case_launches *launches)
{
    const struct site_weights *const own = &layout->copies[copy % layout->copy_counts[site]].sites[site];
    const struct linear_range router_weights =
	    layout->copies[copy % layout->copy_counts[FFN_SITE]].router_weights;
    const uint64_t arena = card->arena_byte_count;
    const bool is_ffn = site == FFN_SITE;
    struct linear_routed_experts_description experts;
    struct router_description router;
    router_describe(layout, row_count, is_reference_order, &experts, &router);
    const struct dense_linear_description mix = { own->mix_weights,
						  layout->streams,
						  layout->mixes,
						  no_range,
						  MIXES,
						  MIX_INPUTS,
						  row_count,
						  DENSE_FLOAT32,
						  DENSE_FLOAT32,
						  DENSE_ROW_SCALE,
						  1.0f,
						  MODEL_NORM_EPSILON,
						  0 };
    const struct hyper_site_description finish = { layout->streams,
						   layout->carried,
						   layout->mixes,
						   own->scale,
						   own->base,
						   own->norm,
						   layout->post,
						   layout->comb,
						   layout->hidden,
						   layout->output,
						   no_range,
						   row_count,
						   0 };
    const struct dense_linear_description scores = { router_weights,
						     layout->hidden,
						     layout->scores,
						     no_range,
						     EXPERTS,
						     HIDDEN,
						     row_count,
						     DENSE_BFLOAT16,
						     DENSE_FLOAT32,
						     0,
						     1.0f,
						     0.0f,
						     0 };
    const struct linear_quantize_description quantize = {
	    layout->hidden,   layout->codes, layout->exponents,
	    layout->operands, HIDDEN,	     row_count
    };
    struct site_front_description front = { layout->streams,
					    layout->carried,
					    own->mix_weights,
					    own->scale,
					    own->base,
					    own->norm,
					    layout->post,
					    layout->comb,
					    layout->hidden,
					    layout->codes,
					    layout->exponents,
					    layout->operands,
					    layout->scratch,
					    is_ffn ? &router : NULL,
					    is_ffn ? router_weights : no_range,
					    is_ffn ? SITE_FRONT_KIND_FFN :
						     SITE_FRONT_KIND_SITE,
					    row_count,
					    0 };
    struct kernel_launch site_launches[HYPER_SITE_LAUNCH_COUNT];
    uint32_t count = 0;
    result_status status = dense_linear_prepare(&mix, arena, &kernels->dense,
						&launches->today[count++]);
    if (status == STATUS_OK) {
	    status = hyper_site_prepare(&finish, arena, &kernels->hyper,
					site_launches);
	    launches->today[count++] = site_launches[HYPER_SITE_MIX_FINISH];
    }
    if (status == STATUS_OK && is_ffn) {
	    status = dense_linear_prepare(&scores, arena, &kernels->dense,
					  &launches->today[count++]);
	    status = status != STATUS_OK ?
			     status :
			     router_select_prepare(&router, arena,
						   &kernels->router,
						   &launches->today[count++]);
    }
    if (status == STATUS_OK) {
	    status = linear_quantize_prepare(&quantize, arena, &card->kernels,
					     &launches->today[count++]);
    }
    launches->today_count = count;
    status = status != STATUS_OK ?
		     status :
		     site_front_prepare(&front, arena,
					is_ffn ? &kernels->router : NULL,
					&kernels->front, &launches->fused);
    for (uint32_t launch = 0;
	 status == STATUS_OK && !is_ffn && launch < SPLIT_LAUNCHES; launch++) {
	    front.kind = launch == INPUT_LAUNCH ? SITE_FRONT_KIND_INPUT :
						  SITE_FRONT_KIND_COEFFICIENTS;
	    status = site_front_prepare(&front, arena, NULL, &kernels->front,
					&launches->split[launch]);
    }
    return test_linear_expect_status(status, STATUS_OK,
				     "the launches of the case are prepared");
}

/* ---- what came back ---- */

/* The output ranges of a site's launches. */
static uint32_t outputs_of(const struct layout *layout, enum site site,
			   struct linear_range *ranges)
{
	const struct linear_range every[OUTPUT_ARRAYS] = {
		layout->mixes,	    layout->post,     layout->comb,
		layout->carried,    layout->hidden,   layout->codes,
		layout->exponents,  layout->operands, layout->scores,
		layout->affinities, layout->jobs
	};
	memcpy(ranges, every, sizeof(every));
	return site == FFN_SITE ? OUTPUT_ARRAYS : SCORES_OUT;
}

/* Bytes of each output range that differ between what came back and expected_work, each array's first one shown:
 * of the mixes, today's array of row_count rows against the fused launch's scratch mixes. */
static uint64_t outputs_differ(const struct test_linear_card *card, const struct layout *layout, enum site site,
                               uint32_t row_count, bool is_shown, uint32_t *array_mask_out)
{
	struct linear_range ranges[OUTPUT_ARRAYS];
	const uint32_t count = outputs_of(layout, site, ranges);
	const uint64_t scratch_mixes =
		layout->scratch.byte_offset +
		offsetof(struct site_front_scratch, mixes);
	uint64_t differing = 0;
	uint32_t array_mask = 0;
	ranges[MIXES_OUT].byte_count =
		(uint64_t)row_count * MODEL_MIXES_ROW_BYTES;
	for (uint32_t array = 0; array < count; array++) {
		uint64_t first = UINT64_MAX, in_array = 0;
		for (uint64_t byte = 0; byte < ranges[array].byte_count;
		     byte++) {
			const uint64_t at = ranges[array].byte_offset + byte;
			const uint64_t fused_at =
				array == MIXES_OUT ? scratch_mixes + byte : at;
			if (card->received[fused_at] != expected_work[at]) {
				first = first == UINT64_MAX ? byte : first;
				in_array++;
			}
		}
		if (in_array != 0 && is_shown) {
			test_linear_fail(
				"%s: %" PRIu64
				" bytes differ from today's launches, the first byte %" PRIu64,
				output_names[array], in_array, first);
		}
		array_mask |= in_array != 0 ? 1u << array : 0u;
		differing += in_array;
	}
    *array_mask_out = array_mask;
    return differing;
}

/* The scratch's counters are zero, and the ranges only read are untouched. */
static bool after_fused_check(struct test_linear_card *card, const struct layout *layout)
{
	struct site_front_scratch scratch;
	return test_linear_card_get(card, layout->scratch, 0, &scratch,
				    sizeof(scratch)) &&
	       test_linear_expect(
		       scratch.tickets == 0 && scratch.arrived == 0 &&
			       scratch.rows_ready == 0 && scratch.scored == 0,
		       "the fused launch leaves its counters zero") &&
	       test_linear_card_expect_untouched(card, layout->streams,
						 "the streams are only read") &&
	       test_linear_card_expect_untouched(
		       card, layout->step, "the step block is only read") &&
	       test_linear_card_expect_untouched(
		       card, layout->offsets,
		       "the table of offsets is only read");
}

/* ---- the order of the chains, on the host ---- */

/* The warp sum of one logical warp of the dense product (dense_linear_records.h) for one output and one row: weights
 * one output's inputs as float32, values the row's bfloat16, chunks t, t + 256, ... while below chunk_count. */
static float warp_sum_of(const float *weights, const uint16_t *values, uint32_t chunk_count, uint32_t warp)
{
    float lanes[WARP_LANES];
    for (uint32_t lane = 0; lane < WARP_LANES; lane++) {
        float chain = 0.0f;
        for (uint32_t chunk = warp * WARP_LANES + lane; chunk < chunk_count; chunk += DENSE_THREADS) {
            for (uint32_t index = 0; index < CHUNK_VALUES; index++) {
                const uint32_t input = chunk * CHUNK_VALUES + index;
                chain = fmaf(weights[input], bfloat16_float(values[input]), chain);
            }
        }
        lanes[lane] = chain;
    }
    for (uint32_t distance = WARP_LANES >> 1u; distance != 0; distance >>= 1u) {
        float next[WARP_LANES];
        for (uint32_t lane = 0; lane < WARP_LANES; lane++) {
            next[lane] = lanes[lane] + lanes[lane ^ distance];
        }
        memcpy(lanes, next, sizeof(lanes));
    }
    return lanes[0];
}

/* The sum of a dense output's warp sums from +0, ascending (the product's order) or descending (another). */
static float warps_added(const float *warp_sums, bool is_descending)
{
    float sum = 0.0f;
    for (uint32_t warp = 0; warp < DENSE_WARPS; warp++) {
        sum = sum + warp_sums[is_descending ? DENSE_WARPS - 1u - warp : warp];
    }
    return sum;
}

/* The fused launch's published warp sums of the mix project, and its scores of an FFN case, against the host's chains
 * bit for bit; and how many of them the descending order would change (counted into order_sensitive). */
static bool chains_check(const struct test_linear_card *card, const struct layout *layout, enum site site,
                         uint32_t row_count, uint64_t *order_sensitive)
{
	static struct site_front_scratch scratch;
	static float mix_weights_row[MIX_INPUTS], router_row[HIDDEN];
	static uint16_t hidden_back[ROWS][HIDDEN];
	static float scores_back[ROWS][EXPERTS];
	uint32_t wrong = 0;
	if (!test_linear_card_get(card, layout->scratch, 0, &scratch,
				  sizeof(scratch)) ||
	    !test_linear_card_get(card, layout->hidden, 0, hidden_back,
				  sizeof(hidden_back)) ||
	    !test_linear_card_get(card, layout->scores, 0, scores_back,
				  sizeof(scores_back))) {
		return false;
	}
    for (uint32_t mix = 0; mix < MIXES; mix++) {
        memcpy(mix_weights_row, mix_weights_in[site][mix], sizeof(mix_weights_row));
        for (uint32_t row = 0; row < row_count; row++) {
            float warp_sums[DENSE_WARPS];
            for (uint32_t warp = 0; warp < DENSE_WARPS; warp++) {
                warp_sums[warp] = warp_sum_of(mix_weights_row, streams_in[row], MIX_INPUTS / CHUNK_VALUES, warp);
                wrong += bits_of(warp_sums[warp]) != bits_of(scratch.warp_sums[mix][warp][row]) ? 1u : 0u;
            }
            const bool is_told_apart = bits_of(warps_added(warp_sums, false)) != bits_of(warps_added(warp_sums, true));
            *order_sensitive += is_told_apart ? 1u : 0u;
        }
    }
    for (uint32_t expert = 0; expert < EXPERTS && site == FFN_SITE; expert++) {
        for (uint32_t input = 0; input < HIDDEN; input++) {
            router_row[input] = bfloat16_float(router_weights_in[expert][input]);
        }
        for (uint32_t row = 0; row < row_count; row++) {
            float warp_sums[DENSE_WARPS];
            for (uint32_t warp = 0; warp < DENSE_WARPS; warp++) {
                warp_sums[warp] = warp_sum_of(router_row, hidden_back[row], HIDDEN / CHUNK_VALUES, warp);
            }
            const float score = warps_added(warp_sums, false);
            wrong += bits_of(score) != bits_of(scores_back[row][expert]) ? 1u : 0u;
            *order_sensitive += bits_of(score) != bits_of(warps_added(warp_sums, true)) ? 1u : 0u;
        }
    }
    if (wrong != 0) {
        test_linear_fail("%" PRIu32 " warp sums or scores are not the host's chains in the dense product's order",
                         wrong);
    }
    return wrong == 0;
}

/* ---- the cases ---- */

/* Loads the case's image, runs launches and fetches the work region. */
static bool run_and_fetch(struct test_linear_card *card,
			  const struct kernel_launch *launches, uint32_t count)
{
    return test_linear_card_load(card) && test_linear_card_run(card, launches, count) && test_linear_card_fetch(card);
}

/* One case: today's launches, then the fused one, from the same image; and the host's chains. */
static bool case_run(struct test_linear_card *card, const struct card_kernels *kernels, const struct layout *layout,
                     enum site site, uint32_t row_count, uint32_t live_rows, bool is_reference_order,
                     uint64_t *order_sensitive)
{
    struct case_launches launches;
    uint32_t array_mask = 0;
    test_linear_context("card %" PRIu32 ", the %s site, %" PRIu32 " rows (%" PRIu32 " live), %s order",
                        card->card_index, site_names[site], row_count, live_rows,
                        is_reference_order ? "the reference's" : "the fixed");
    bool is_passed = work_put(card, layout, live_rows) &&
                     launches_prepare(card, kernels, layout, site, row_count, is_reference_order, 0, &launches) &&
                     run_and_fetch(card, launches.today, launches.today_count);
    if (is_passed) {
        memcpy(expected_work, card->received, card->work_byte_count);
    }
    is_passed = is_passed && run_and_fetch(card, &launches.fused, 1) &&
                outputs_differ(card, layout, site, row_count, true, &array_mask) == 0 &&
                after_fused_check(card, layout) && chains_check(card, layout, site, row_count, order_sensitive);
    if (is_passed && site == ATTENTION_SITE) {      /* the site front cut in two, from the same image */
        test_linear_context("card %" PRIu32 ", the attention site cut in two, %" PRIu32 " rows", card->card_index,
                            row_count);
        is_passed = run_and_fetch(card, launches.split, SPLIT_LAUNCHES) &&
                    outputs_differ(card, layout, site, row_count, true, &array_mask) == 0 &&
                    after_fused_check(card, layout) && chains_check(card, layout, site, row_count, order_sensitive);
    }
    return is_passed;
}

/* A negative control: one weight flipped in the image for the fused launches alone (its top mantissa bit); their
 * outputs must differ from today's in the array expected, then the image is put back. */
static bool control_run(struct test_linear_card *card, const struct card_kernels *kernels, const struct layout *layout,
                        bool is_router_weight)
{
    struct case_launches launches;
    const struct weights_copy *const weights = &layout->copies[0];
    const uint32_t input = 0;                   /* the told-apart row's large value is input 0 of stream 0 */
    const enum output_array expected_array = is_router_weight ? SCORES_OUT : MIXES_OUT;
    uint32_t array_mask = 0;
    float mix_weight = mix_weights_in[FFN_SITE][BROKEN_MIX][input];
    uint16_t router_weight = router_weights_in[BROKEN_EXPERT][input];
    test_linear_context("card %" PRIu32 ", the negative control of a %s weight", card->card_index,
                        is_router_weight ? "router" : "mix");
    if (!work_put(card, layout, ROWS) ||
        !launches_prepare(card, kernels, layout, FFN_SITE, ROWS, false, 0, &launches) ||
        !run_and_fetch(card, launches.today, launches.today_count)) {
        return false;
    }
    memcpy(expected_work, card->received, card->work_byte_count);
    mix_weight = is_router_weight ? mix_weight : float_of(bits_of(mix_weight) ^ FLOAT32_TOP_MANTISSA_BIT);
    router_weight = (uint16_t)(is_router_weight ? router_weight ^ BFLOAT16_TOP_MANTISSA_BIT : router_weight);
    const uint64_t mix_byte = ((uint64_t)BROKEN_MIX * MIX_INPUTS + input) * sizeof(float);
    const uint64_t router_byte = ((uint64_t)BROKEN_EXPERT * HIDDEN + input) * sizeof(uint16_t);
    bool is_run = test_linear_card_put(card, weights->sites[FFN_SITE].mix_weights, mix_byte, &mix_weight,
                                       sizeof(mix_weight)) &&
                  test_linear_card_put(card, weights->router_weights, router_byte, &router_weight,
                                       sizeof(router_weight)) &&
                  run_and_fetch(card, &launches.fused, 1);
    const uint64_t differing = is_run ? outputs_differ(card, layout, FFN_SITE, ROWS, false, &array_mask) : 0;
    (void)printf("card %" PRIu32 ": negative control, a %s weight changed for the fused launches alone: %" PRIu64
                 " output bytes differ from today's (arrays 0x%" PRIx32 ")\n", card->card_index,
                 is_router_weight ? "router" : "mix", differing, array_mask);
    is_run = is_run && test_linear_expect((array_mask >> expected_array & 1u) != 0,
                                          "the negative control fails the comparison where it must");
    control_failures += differing != 0 ? 1u : 0u;
    return is_run &&
           test_linear_card_put(card, weights->sites[FFN_SITE].mix_weights, mix_byte,
                                &mix_weights_in[FFN_SITE][BROKEN_MIX][input], sizeof(float)) &&
           test_linear_card_put(card, weights->router_weights, router_byte, &router_weights_in[BROKEN_EXPERT][input],
                                sizeof(uint16_t)) &&
           test_linear_card_upload(card, weights->sites[FFN_SITE].mix_weights, mix_byte, sizeof(float)) &&
           test_linear_card_upload(card, weights->router_weights, router_byte, sizeof(uint16_t));
}

/* ---- the times ---- */

static int microseconds_compare(const void *first, const void *second)
{
    const double left = *(const double *)first, right = *(const double *)second;
    return (left > right) - (left < right);
}

/* The time of a copy of a graph of `copies` copies of `count` launches each: the median and the least of ROUNDS
 * measurements (other work on the card only adds time to a round). */
struct timing {
    double median, least;
};
static bool graph_time(struct test_linear_card *card,
		       const struct kernel_launch *table, uint32_t count,
		       uint32_t copies, struct timing *timing_out)
{
    double rounds[ROUNDS];
    uint32_t graph = 0;
    bool is_timed_run = test_linear_expect_status(
	    backend_cuda_value.graph_capture(card->card, table, count, &graph),
	    STATUS_OK, "the graph is captured");
    for (uint32_t round = 0; is_timed_run && round <= ROUNDS; round++) {  /* round 0 warms */
        const uint64_t started = clock_nanoseconds();
        for (uint32_t replay = 0; is_timed_run && replay < GRAPH_REPLAYS; replay++) {
		is_timed_run = test_linear_expect_status(
			backend_cuda_value.graph_run(card->card, graph),
			STATUS_OK, "the graph is queued");
	}
	is_timed_run =
		is_timed_run &&
		test_linear_expect_status(
			backend_cuda_value.launch_wait(
				card->card, TEST_LINEAR_WAIT_LIMIT_NANOSECONDS),
			STATUS_OK, "the replays finish within the limit");
	if (round != 0) {
            rounds[round - 1u] = (double)(clock_nanoseconds() - started) / NANOSECONDS_PER_MICROSECOND /
                                 (double)(GRAPH_REPLAYS * copies);
        }
    }
    if (is_timed_run) {
        qsort(rounds, ROUNDS, sizeof(rounds[0]), microseconds_compare);
        *timing_out = (struct timing){rounds[MEDIAN_ROUND], rounds[0]};
    }
    return is_timed_run;
}

/* Copies of a site's launches (today's or the fused one) in one graph: of hot weights COPIES copies of copy 0's, of
 * cold ones a copy of each copy of the site's weights in turn, so that a replay reads them all once. */
static bool site_time(struct test_linear_card *card, const struct card_kernels *kernels, const struct layout *layout,
                      enum site site, uint32_t row_count, bool is_fused, bool is_cold, struct timing *timing_out)
{
    const uint32_t copies = is_cold ? layout->copy_counts[site] : COPIES;
    uint32_t count = 0;
    bool is_prepared = true;
    for (uint32_t copy = 0; is_prepared && copy < copies; copy++) {
        struct case_launches launches;
        is_prepared = launches_prepare(card, kernels, layout, site, row_count, false, is_cold ? copy : 0u,
                                       &launches);
        const uint32_t launch_count = is_fused ? 1u : launches.today_count;
        memcpy(&timed_launches[count], is_fused ? &launches.fused : launches.today,
               launch_count * sizeof(timed_launches[0]));
        count += is_prepared ? launch_count : 0u;
    }
    return is_prepared && graph_time(card, timed_launches, count, copies, timing_out);
}

/* COPIES copies of one launch alone, of hot weights. */
static bool launch_time(struct test_linear_card *card,
			const struct kernel_launch *launch,
			struct timing *timing_out)
{
    for (uint32_t copy = 0; copy < COPIES; copy++) {
        timed_launches[copy] = *launch;
    }
    return graph_time(card, timed_launches, COPIES, COPIES, timing_out);
}

/* Every time of one card, printed as TIME lines: the median (and the least) in microseconds a copy. */
static bool time_runs(struct test_linear_card *card, const struct card_kernels *kernels, struct layout *layout)
{
    static const char *const today_names[] = {"mix project", "mix finish", "router scores", "router select",
                                              "quantization"};
    static const char *const weights_names[WEIGHTS_KINDS] = {"hot", "cold"};
    static const char *const split_names[SPLIT_LAUNCHES] = {"input", "coefficient"};
    bool is_passed = true;
    for (uint32_t site = 0; is_passed && site < SITES; site++) {     /* a site's cold copies laid out for it alone */
        is_passed = layout_make(card, layout, site == ATTENTION_SITE ? ATTENTION_COLD_COPIES : 1u,
                                site == FFN_SITE ? FFN_COLD_COPIES : 1u);
        for (size_t rows = 0; is_passed && rows < COUNT_OF(timed_rows); rows++) {
            const uint32_t row_count = timed_rows[rows];
            is_passed = work_put(card, layout, row_count) && test_linear_card_load(card);  /* a step of row_count */
            for (uint32_t weights = 0; is_passed && weights < WEIGHTS_KINDS; weights++) {
                const bool is_cold = weights == COLD_WEIGHTS;
                struct timing today, fused;
                is_passed = site_time(card, kernels, layout, (enum site)site, row_count, false, is_cold, &today) &&
                            site_time(card, kernels, layout, (enum site)site, row_count, true, is_cold, &fused);
                if (is_passed) {
                    (void)printf("TIME card %" PRIu32 " (class %" PRIu32 "): %s site, %" PRIu32 " rows, %s weights: "
                                 "today %.2f us (least %.2f), fused %.2f us (least %.2f), %.2fx\n", card->card_index,
                                 card->arithmetic_class, site_names[site], row_count, weights_names[weights],
                                 today.median, today.least, fused.median, fused.least, today.median / fused.median);
                }
            }
        }
    }
    for (size_t rows = 0; is_passed && rows < COUNT_OF(timed_rows); rows++) {   /* hot, on copy 0 of each site */
        const uint32_t row_count = timed_rows[rows];
        struct case_launches launches[SITES];
        is_passed = work_put(card, layout, row_count) && test_linear_card_load(card);
        for (uint32_t site = 0; is_passed && site < SITES; site++) {
            is_passed = launches_prepare(card, kernels, layout, (enum site)site, row_count, false, 0,
                                         &launches[site]);
        }
        for (uint32_t launch = 0; is_passed && launch < launches[FFN_SITE].today_count; launch++) {
            struct timing alone;
            is_passed = launch_time(card, &launches[FFN_SITE].today[launch], &alone);
            (void)printf("TIME card %" PRIu32 " (class %" PRIu32 "): %" PRIu32 " rows, today's %s alone: %.2f us "
                         "(least %.2f)\n", card->card_index, card->arithmetic_class, row_count, today_names[launch],
                         alone.median, alone.least);
        }
        for (uint32_t site = 0; is_passed && site < SITES; site++) {
            struct timing alone;
            is_passed = launch_time(card, &launches[site].fused, &alone);
            (void)printf("TIME card %" PRIu32 " (class %" PRIu32 "): %" PRIu32 " rows, the %s front alone: %.2f us "
                         "(least %.2f)\n", card->card_index, card->arithmetic_class, row_count,
                         site == FFN_SITE ? "FFN" : "site", alone.median, alone.least);
        }
        struct timing split[SPLIT_LAUNCHES], both;
        for (uint32_t launch = 0; is_passed && launch < SPLIT_LAUNCHES; launch++) {
            is_passed = launch_time(card, &launches[ATTENTION_SITE].split[launch], &split[launch]);
        }
        for (uint32_t copy = 0; is_passed && copy < COPIES; copy++) {
            memcpy(&timed_launches[(size_t)copy * SPLIT_LAUNCHES], launches[ATTENTION_SITE].split,
                   sizeof(launches[0].split));
        }
        is_passed = is_passed && graph_time(card, timed_launches, COPIES * SPLIT_LAUNCHES, COPIES, &both);
        if (is_passed) {
            (void)printf("TIME card %" PRIu32 " (class %" PRIu32 "): %" PRIu32 " rows, the attention site cut in two: "
                         "the %s front alone %.2f us (least %.2f), the %s front alone %.2f us (least %.2f), both in "
                         "turn %.2f us (least %.2f)\n", card->card_index, card->arithmetic_class, row_count,
                         split_names[INPUT_LAUNCH], split[INPUT_LAUNCH].median, split[INPUT_LAUNCH].least,
                         split_names[COEFFICIENT_LAUNCH], split[COEFFICIENT_LAUNCH].median,
                         split[COEFFICIENT_LAUNCH].least, both.median, both.least);
        }
    }
    return is_passed;
}

/* ---- the program ---- */

/* Every case of one card, the controls and, with --time, the times. */
static void card_run(uint32_t card_index)
{
    static struct test_linear_card card;
    static struct layout layout;
    struct card_kernels kernels;
    uint64_t order_sensitive = 0;
    uint32_t case_count = 0;
    bool is_passed =
	    test_linear_card_open(&card, card_index, WORK_CAPACITY,
				  is_timed ? WEIGHTS_CAPACITY :
					     CHECK_WEIGHTS_CAPACITY) &&
	    test_linear_expect_status(
		    hyper_kernels_prepare(&backend_cuda_value, card.card,
					  &kernels.hyper),
		    STATUS_OK, "the hyper kernels are prepared") &&
	    test_linear_expect_status(
		    dense_linear_kernels_prepare(&backend_cuda_value, card.card,
						 &kernels.dense),
		    STATUS_OK, "the dense kernels are prepared") &&
	    test_linear_expect_status(
		    router_kernels_prepare(&backend_cuda_value, card.card,
					   &kernels.router),
		    STATUS_OK, "the router kernels are prepared") &&
	    test_linear_expect_status(
		    site_front_kernels_prepare(&backend_cuda_value, card.card,
					       &kernels.front),
		    STATUS_OK, "the site front kernels are prepared") &&
	    layout_make(&card, &layout, 1u, 1u);
    for (uint32_t row_count = 1; is_passed && row_count <= ROWS; row_count++) {
        for (uint32_t site = 0; is_passed && site < SITES; site++) {
            const uint32_t live_kinds = site == FFN_SITE && row_count > 1u ? LIVE_KINDS : 1u;
            for (uint32_t live = 0; is_passed && live < live_kinds; live++) {
                for (uint32_t order = 0; is_passed && order <= (site == FFN_SITE ? 1u : 0u); order++) {
                    is_passed = case_run(&card, &kernels, &layout, (enum site)site, row_count, row_count - live,
                                         order != 0, &order_sensitive);
                    case_count++;
                }
            }
        }
    }
    is_passed = is_passed && control_run(&card, &kernels, &layout, false) &&
                control_run(&card, &kernels, &layout, true) && test_linear_card_expect_weights_untouched(&card);
    if (is_passed) {
        (void)printf("card %" PRIu32 " (class %" PRIu32 "): %" PRIu32 " cases, every output byte of the fused launch "
                     "today's; the host's chains bit for bit, %" PRIu64 " sums that another order of the warps "
                     "changes\n", card.card_index, card.arithmetic_class, case_count, order_sensitive);
        is_passed = test_linear_expect(order_sensitive != 0, "the inputs tell another order of the warps apart");
    }
    if (is_passed && is_timed) {
        (void)time_runs(&card, &kernels, &layout);
    }
    test_linear_card_close(&card);
}

int main(int argument_count, char **arguments)
{
    uint32_t card_count = 0;
    if (argument_count > ARGUMENTS_WITH_TIME ||
        (argument_count == ARGUMENTS_WITH_TIME && strcmp(arguments[1], "--time") != 0)) {
        (void)fprintf(stderr, "usage: site_front_card_test [--time]\n");
        return TEST_WRONG_USAGE;
    }
    is_timed = argument_count == ARGUMENTS_WITH_TIME;
    (void)alarm(TEST_TIME_LIMIT_SECONDS);
    inputs_make();
    if (test_linear_expect_status(backend_cuda_value.card_count(&card_count),
				  STATUS_OK, "the cards are counted") &&
	test_linear_expect(card_count > 0 &&
				   card_count <= BACKEND_MAXIMUM_CARDS,
			   "the machine has cards for the test")) {
	    for (uint32_t card_index = 0; card_index < card_count;
		 card_index++) {
		    card_run(card_index);
	    }
    }
    if (test_linear_failures() != 0 || control_failures != SITES * card_count) {
        (void)fprintf(stderr, "site_front_card_test FAILED: %d failure lines, %" PRIu32 " of %" PRIu32 " negative "
                              "controls failed as they must\n", test_linear_failures(), control_failures,
                      SITES * card_count);
        return TEST_FAILED;
    }
    (void)printf("site_front_card_test passed on %" PRIu32 " cards (both negative controls failed on each, as they "
                 "must)\n", card_count);
    return TEST_PASSED;
}
