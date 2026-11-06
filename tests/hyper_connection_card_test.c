/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * hyper_connection_card_test: the kernels of the hyper-connection sites on every card, against the host reference
 * (hyper_connection_reference.h) and, given captures, replayed on the model's reference's own inputs.
 *
 *     hyper_connection_card_test [--break scale|weight|output|embedding|tap] [<capture folder> <pack copy> ...]
 *
 * On every card, with LINEAR_MAXIMUM_ROWS rows of synthetic inputs (plain values and, on odd rows, zeros of
 * both signs and values down to 2^-28; row TOLD_APART_ROW collapses with pre (1, 0, 0, 0) to one large value among
 * small ones, which a sum of squares from +0 in order loses and the records' tree keeps: its roots lie 1,019 places
 * apart, and at least 113 of the 5,120 normed values differ between factors within RSQRT_PLACES of each, on the host):
 *   - an attention site and an FFN site (the mix finish, then the post; the FFN's with the combine), then the final
 *     collapse with the embedding: pre and post within TEST_HYPER_SINKHORN_PLACES float32 places of the reference's,
 *     comb within TEST_HYPER_COMB_TOLERANCE, the site's pre in place of the carried one; the sublayer's input and the
 *     head's within one bfloat16 place of the norm of the collapse, and bit for bit the norm in the records' order
 *     for one factor within RSQRT_PLACES float32 places of the reciprocal root of its operand (rsqrtf is
 *     approximate), on row TOLD_APART_ROW for none that near the root of the sum in order; the streams after a post
 *     bit for bit those of the reference fed the post and comb the card computed; the embedding's streams the rows of
 *     the tokens and the carried pre (1, 0, 0, 0) bit for bit; no guard behind a range written, no parameter written;
 *   - R: each row run again alone, as the one row of a launch of one row: every output the bits of its row in the
 *     launch of all rows.
 *   - The draft's inputs (dspark_v1/DESIGN.md K1), DRAFT_ROWS rows in one launch: each of the three taps bit for bit
 *     the reference's collapse with pre 1/4 each, and the final's kept collapse of the same streams with a carried
 *     pre of 1/4 each (its first 8 rows); main_norm of that kept collapse bit for bit the final's normed output; and
 *     main_norm of DRAFT_ROWS plain rows (row TOLD_APART_ROW told apart as above) within one bfloat16 place of the
 *     double norm and bit for bit the records' norm for a factor within RSQRT_PLACES of the reciprocal root.
 * The replay (first_token_plan.md 3.2), given a capture folder of tools/oracle/stepwise_capture.py and the pack: at
 * every capture point layer_L.position_P, each site's mix finish run as one row from the captured mixes, streams and
 * carried pre (the FFN site's: the attention site's pre) with the pack's scale, base and norm: pre, post and comb
 * within 1e-6 of the captured ones, relative, and on a card of class 120, the oracle's (batch_m1a_oracle_captures.sh
 * runs it on its sm_120a card), bit for bit: the card's expf, divisions and fmaf in the generated source's order; the
 * sublayer's input within one bfloat16 place of the captured normed input on 99.9% of its values; then the post
 * alone, from the captured streams, post, comb and sublayer output (attn.wo_b; experts.row_sum with
 * shared_expert.output): the streams after the site (the FFN site's captured streams; the block's output) bit for bit
 * on 99.9% and the rest within one place; the host reference's collapse the captured collapse bit for bit on 99.9%.
 * At every capture point of layer 39 (one at least), the final collapse of its output with its carried weights
 * against final.normed, as the sublayer's input; the embedding of every token against input.embeddings, bit for bit.
 * It prints the shares bit-equal.
 * A break changes the card's copy of an input behind the reference's back, in the synthetic runs and in the replay,
 * so that one comparison must fail in both: scale (the pre's scale plus one: Sinkhorn), weight (the signs of the
 * last BROKEN_WEIGHT_WORDS words of the norm's weight, more than 99.9% of a row leaves: the norms), output (row 0's
 * last sublayer output BROKEN_OUTPUT: the posts), embedding (the low bit of a word of row 0's token: the embedding),
 * tap (the low bit of one word of a stream of the draft's row 0: the taps; no replay holds the draft).
 * With a break the test passes only when that comparison failed, in the replay too where there is one, and nothing
 * else did. Without a capture the replay says that it compared nothing.
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
#include "anemone/draft.h"
#include "anemone/hyper_connection.h"
#include "anemone/model_layout.h"
#include "anemone/pack.h"
#include "anemone/status.h"
#include "anemone/step_records.h"
#include "anemone/storage_seam.h"
#include "capture_reader.h"
#include "hyper_connection_reference.h"
#include "linear_card_fixture.h"
#include "linear_reference.h"

#define TEST_TIME_LIMIT_SECONDS 3600                /* the alarm ends the test here, whatever a drive or a card does */
#define ROWS LINEAR_MAXIMUM_ROWS
#define HIDDEN MODEL_HIDDEN
#define STREAMS MODEL_STREAMS
#define MIXES MODEL_MIXES
#define SCALES HYPER_SCALES
#define COMB_VALUES (STREAMS * STREAMS)
#define STREAM_VALUES ((uint64_t)STREAMS * HIDDEN)
#define STREAMS_BYTES MODEL_STREAMS_ROW_BYTES
#define VOCABULARY 64u                              /* rows of the test's embedding */
#define WORK_CAPACITY (1ull << 23u)                 /* eight rows of every array, the draft's 64, their guards */
#define WEIGHTS_CAPACITY (1ull << 20u)              /* the parameters and the test's embedding */
#define SEED 0x4859u
#define STREAMS_PER_ROW 4u                          /* hash streams of a row: of its streams, output and numbers */
#define TOKEN_STEP 11u                              /* row r reads token (TOKEN_STEP r + TOKEN_FIRST) % VOCABULARY */
#define TOKEN_FIRST 5u
#define HASH_UNIT 0x1p-64
#define HALF 0.5
#define MIX_SPREAD 2.0                              /* mixes and bases from -1 to 1 */
#define SCALE_SPREAD 3.0                            /* scales from 0 to 3 */
#define PRE_LEAST 0.25                              /* carried pre from 0.25 to 1.75 */
#define PRE_SPREAD 1.5
#define SUM_SPREAD 8.0                              /* routed sums from -4 to 4 */
#define REPLAY_RELATIVE 1e-6
#define SHARE_PER_THOUSAND 999u                     /* 99.9% */
#define PER_THOUSAND 1000u
#define PERCENT 100.0
#define FAILURES_SHOWN 16u                          /* failure lines of one category on one card */
#define BREAK_ARGUMENTS 2                           /* --break and its part */
#define READ_BUFFER_BYTES 65536u                    /* the largest tensor read here, 10 KiB, with a block each side */
#define NAME_CAPACITY 160u
#define DECIMAL 10
#define FINAL_LAYER (MODEL_LAYERS - 1u)
#define BFLOAT16_BYTES sizeof(uint16_t)
#define BFLOAT16_SIGN_BIT 0x8000u
#define BROKEN_WEIGHT_WORDS 16u                     /* 0.3% of a row's normed values: past the replay's 99.9% */
#define BROKEN_OUTPUT 0x1p64                        /* post y then far above any stream: more than one place off */
#define RSQRT_PLACES 3u                             /* rsqrtf: within 2 places of 1 / sqrt, 3 across a power of two */
#define NO_FACTOR (RSQRT_PLACES + 1u)
#define MEAN_FACTOR (1.0f / (float)HIDDEN)
#define TOLD_APART_ROW 2u
#define TOLD_APART_LARGE 1024.0                     /* its square 2^20, of which half a float32 place, 2^-4, ... */
#define TOLD_APART_LEAST 0.18                       /* ... is above the square of every small value, 0.18 to 0.24 */
#define TOLD_APART_SPREAD 0.06
#define INPUT_ARRAYS 6u
#define OUTPUT_ARRAYS 5u
#define LAUNCHES \
	HYPER_SITE_LAUNCH_COUNT /* of every run: a site's two, or the final and the embedding */
#define ONE_LAUNCH 1u
#define DRAFT_ROWS 64u                              /* the draft's inputs over more rows than a thin step's */
#define DRAFT_SEED 0x44524146u
#define DRAFT_FEATURES MODEL_DRAFT_FEATURES
#define DRAFT_TAPS MODEL_DRAFT_TAPS
#define BROKEN_TAP_WORD HIDDEN                      /* stream 1's first value of row 0 */

enum test_status { TEST_PASSED = 0, TEST_FAILED = 1, TEST_WRONG_USAGE = 2 };
enum category { SINKHORN, NORM, POST, EMBED, DRAFT, ALONE, COLLAPSE, CATEGORY_COUNT };
enum row_input { STREAMS_INPUT, OUTPUT_INPUT, NUMBERS_INPUT };
enum run_kind { ATTENTION_RUN, FFN_RUN, FINAL_RUN, RUN_KINDS };
enum row_kind { ONE_ROW, ALL_ROWS, ROW_KINDS };
enum comparison { PRE_POST_COMB, NORMED, AFTER_POST, EMBEDDED, HOST_COLLAPSE, FINAL_NORMED, COMPARISON_COUNT };
enum final_launch { FINAL_LAUNCH, EMBED_LAUNCH };   /* the launches of the run FINAL_RUN */
/* the draft's run: the final keeping its collapse, the three taps, main_norm of the rows, then of the collapse */
enum draft_launch { DRAFT_FINAL, DRAFT_FIRST_TAP, DRAFT_SECOND_TAP, DRAFT_THIRD_TAP, DRAFT_ROWS_NORM,
                    DRAFT_COLLAPSE_NORM, DRAFT_LAUNCHES };
_Static_assert(DRAFT_ROWS_NORM == DRAFT_FIRST_TAP + DRAFT_TAPS, "a launch for each tap");

static const char *const category_names[CATEGORY_COUNT] = {"Sinkhorn", "a norm", "a post", "the embedding",
                                                           "the draft's inputs", "a row alone",
                                                           "the host's collapse"};
static const char *const break_names[] = {"scale", "weight", "output", "embedding", "tap"};     /* SINKHORN .. DRAFT */
static const char *const comparison_names[COMPARISON_COUNT] = {"pre, post and comb", "normed inputs",
                                                               "streams after a post", "embedded streams",
                                                               "the host's collapse", "the head's inputs"};
static uint32_t category_failures[CATEGORY_COUNT];
static int shown_failures = 0;
static int break_category = -1;

/* Where everything of a case lies: the work ranges of ROWS rows, the parameters, and the launches of each run, for
 * one row and for ROWS. */
struct layout {
	struct linear_range streams, carried, mixes, post, comb, hidden, output,
		sums, step;
	struct linear_range scale, base, weight, embedding;
	struct kernel_launch launches[RUN_KINDS][ROW_KINDS][LAUNCHES];
};

/* An array of rows in the work region: its range, where its first row lies in it, the bytes of a row. */
struct row_array {
	struct linear_range range;
	uint64_t first_byte;
	uint64_t row_bytes;
};

/* What the synthetic runs on one card measured. */
struct measures {
    uint64_t normed_equal;                      /* normed values equal to the double norm rounded once */
    uint64_t normed_compared;
    uint32_t largest_places;                    /* of pre and post from the reference's */
    uint32_t largest_factor_places;             /* of the factor a row's norm needed from the reciprocal root */
    double largest_comb;                        /* difference of comb from the reference's */
};

/* Counts of the replay on one card: compared, bit-equal. */
struct shares {
    uint64_t compared[COMPARISON_COUNT];
    uint64_t equal[COMPARISON_COUNT];
};

/* What the pack and the capture are read through. */
struct replay_sources {
    struct test_capture *capture;
    struct storage_reader *reader;
    struct pack_record *pack;
    uint32_t block_alignment_bytes;
};

/* The synthetic inputs (the same on every card) and what a run left; one site's case of the replay. */
static uint16_t streams_in[ROWS][STREAM_VALUES];
static float carried_in[ROWS][STREAMS];
static float mixes_in[ROWS][MIXES];
static uint16_t output_in[ROWS][HIDDEN];
static float sums_in[ROWS][HIDDEN];
static float scale_in[SCALES];
static float base_in[MIXES];
static uint16_t weight_in[HIDDEN];
static uint16_t embedding_in[VOCABULARY][HIDDEN];
static unsigned char
	step_in[STEP_BLOCK_BYTES_VALUE + ROWS * STEP_ROW_BYTES_VALUE];
static uint16_t streams_out[ROWS][STREAM_VALUES];
static float carried_out[ROWS][STREAMS];
static float post_out[ROWS][STREAMS];
static float comb_out[ROWS][COMB_VALUES];
static uint16_t hidden_out[ROWS][HIDDEN];
static uint16_t expected_streams[STREAM_VALUES];
static uint16_t collapsed[HIDDEN];
static double normed[HIDDEN];
static uint16_t ordered_normed[HIDDEN];
static float captured_pre[STREAMS];
static float captured_post[STREAMS];
static float captured_comb[COMB_VALUES];
static uint16_t captured_collapsed[HIDDEN];
static uint16_t captured_normed[HIDDEN];
static unsigned char read_buffer[READ_BUFFER_BYTES]
	__attribute__((aligned(BACKEND_STAGING_ALIGNMENT_BYTES)));
static uint16_t draft_streams[DRAFT_ROWS][STREAM_VALUES];
static uint16_t draft_rows_in[DRAFT_ROWS][HIDDEN];
static uint16_t draft_taps_out[DRAFT_ROWS][DRAFT_FEATURES];
static uint16_t draft_normed_out[DRAFT_ROWS][HIDDEN];
static uint16_t draft_collapse_out[ROWS][HIDDEN];
static uint16_t draft_final_out[ROWS][HIDDEN];
static uint16_t draft_collapse_normed_out[ROWS][HIDDEN];
static float draft_carried[ROWS][STREAMS];

static void category_fail(enum category category, const char *what, uint32_t row, uint32_t index)
{
    if (category_failures[category] < FAILURES_SHOWN) {
        test_linear_fail("%s: %s (row %" PRIu32 ", value %" PRIu32 ")", category_names[category], what, row, index);
        shown_failures++;
    }
    category_failures[category]++;
}

static double fraction_of(uint64_t stream, uint64_t position)
{
    return (double)test_linear_hash(stream, position) * HASH_UNIT;
}

static uint32_t bfloat16_places(uint16_t first, uint16_t second)
{
    const int32_t distance = test_linear_bfloat16_ordinal(first) - test_linear_bfloat16_ordinal(second);
    return (uint32_t)(distance < 0 ? -distance : distance);
}

static double magnitude_of(double value)
{
    return value < 0.0 ? -value : value;
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

/* ---- the layout of a card and its launches ---- */

static bool layout_prepare(struct test_linear_card *card,
			   const struct hyper_kernels_record *kernels,
			   struct layout *layout)
{
    test_linear_card_clear(card);
    layout->streams = test_linear_card_work(card, (uint64_t)ROWS * STREAMS_BYTES);
    layout->carried =
	    test_linear_card_work(card, (uint64_t)ROWS * MODEL_SITE_ROW_BYTES);
    layout->mixes =
	    test_linear_card_work(card, (uint64_t)ROWS * MODEL_MIXES_ROW_BYTES);
    layout->post =
	    test_linear_card_work(card, (uint64_t)ROWS * MODEL_SITE_ROW_BYTES);
    layout->comb =
	    test_linear_card_work(card, (uint64_t)ROWS * MODEL_COMB_ROW_BYTES);
    layout->hidden = test_linear_card_work(
	    card, (uint64_t)ROWS * MODEL_HIDDEN_ROW_BYTES);
    layout->output = test_linear_card_work(
	    card, (uint64_t)ROWS * MODEL_HIDDEN_ROW_BYTES);
    layout->sums = test_linear_card_work(card, (uint64_t)ROWS * HIDDEN * sizeof(float));
    layout->step = test_linear_card_work(card, sizeof(step_in));
    layout->scale = test_linear_card_weights(card, SCALES * sizeof(float));
    layout->base = test_linear_card_weights(card, MODEL_MIXES_ROW_BYTES);
    layout->weight = test_linear_card_weights(card, MODEL_HIDDEN_ROW_BYTES);
    layout->embedding = test_linear_card_weights(
	    card, (uint64_t)VOCABULARY * MODEL_HIDDEN_ROW_BYTES);
    bool is_prepared = test_linear_card_has_room(card);
    for (uint32_t row_kind = 0; is_prepared && row_kind < ROW_KINDS; row_kind++) {
        const uint32_t row_count = row_kind == ONE_ROW ? 1u : ROWS;
	struct hyper_site_description site = { .streams = layout->streams,
					       .carried_pre = layout->carried,
					       .mixes = layout->mixes,
					       .scale = layout->scale,
					       .base = layout->base,
					       .norm_weight = layout->weight,
					       .post = layout->post,
					       .comb = layout->comb,
					       .hidden = layout->hidden,
					       .output = layout->output,
					       .row_count = row_count };
	const struct hyper_final_description final = {
		.streams = layout->streams,
		.carried_pre = layout->carried,
		.norm_weight = layout->weight,
		.hidden = layout->hidden,
		.row_count = row_count
	};
	const struct hyper_embed_description embed = {
		.step = layout->step,
		.embedding = layout->embedding,
		.streams = layout->streams,
		.carried_pre = layout->carried,
		.row_count = row_count,
		.vocabulary_size = VOCABULARY
	};
	struct kernel_launch *const final_launches =
		layout->launches[FINAL_RUN][row_kind];
	is_prepared = test_linear_expect_status(
		hyper_site_prepare(&site, card->arena_byte_count, kernels,
				   layout->launches[ATTENTION_RUN][row_kind]),
		STATUS_OK, "the attention site is prepared");
	site.routed_sums = layout->sums;
	is_prepared =
		is_prepared &&
		test_linear_expect_status(
			hyper_site_prepare(&site, card->arena_byte_count,
					   kernels,
					   layout->launches[FFN_RUN][row_kind]),
			STATUS_OK, "the FFN site is prepared") &&
		test_linear_expect_status(
			hyper_final_prepare(&final, card->arena_byte_count,
					    kernels,
					    &final_launches[FINAL_LAUNCH]),
			STATUS_OK, "the final collapse is prepared") &&
		test_linear_expect_status(
			hyper_embed_prepare(&embed, card->arena_byte_count,
					    kernels,
					    &final_launches[EMBED_LAUNCH]),
			STATUS_OK, "the embedding is prepared");
    }
    return is_prepared;
}

/* Loads the image, runs launch_count launches of a run from first_launch, for one row or for all, and fetches the
 * work region. */
static bool run_and_fetch(struct test_linear_card *card, const struct layout *layout, enum run_kind run,
                          enum row_kind rows, uint32_t first_launch, uint32_t launch_count)
{
    return test_linear_card_load(card) &&
           test_linear_card_run(card, &layout->launches[run][rows][first_launch], launch_count) &&
           test_linear_card_fetch(card);
}

/* ---- the synthetic case ---- */

static void synthetic_inputs(void)
{
	const struct step_block block = { .sequence = 1,
					  .row_count = ROWS,
					  .flags = 0 };
	memcpy(step_in, &block, sizeof(block));
	for (uint32_t row = 0; row < ROWS; row++) {
		const uint64_t stream =
			SEED + (uint64_t)STREAMS_PER_ROW * (row + 1u);
		const struct step_row_record step_row = {
			.position = (int32_t)row,
			.token_id = (int32_t)((TOKEN_STEP * row + TOKEN_FIRST) %
					      VOCABULARY),
			.session_slot = 0,
			.flags = STEP_ROW_LOGITS
		};
		memcpy(&step_in[STEP_BLOCK_BYTES_VALUE +
				row * STEP_ROW_BYTES_VALUE],
		       &step_row, sizeof(step_row));
		test_linear_values(streams_in[row], STREAM_VALUES,
				   stream + STREAMS_INPUT,
				   (row & 1u) == 0 ? TEST_LINEAR_VALUES_PLAIN :
						     TEST_LINEAR_VALUES_SMALL);
		test_linear_values(output_in[row], HIDDEN,
				   stream + OUTPUT_INPUT,
				   TEST_LINEAR_VALUES_PLAIN);
		for (uint32_t index = 0; index < HIDDEN; index++) {
			sums_in[row][index] =
				(float)((fraction_of(stream + NUMBERS_INPUT,
						     index) -
					 HALF) *
					SUM_SPREAD);
		}
		for (uint32_t index = 0; index < MIXES; index++) {
			mixes_in[row][index] =
				(float)((fraction_of(stream + NUMBERS_INPUT,
						     HIDDEN + index) -
					 HALF) *
					MIX_SPREAD);
		}
		for (uint32_t index = 0; index < STREAMS; index++) {
			carried_in[row][index] =
				(float)(PRE_LEAST +
					fraction_of(stream + NUMBERS_INPUT,
						    HIDDEN + MIXES + index) *
						PRE_SPREAD);
		}
    }
    for (uint32_t index = 0; index < MIXES; index++) {
        base_in[index] = (float)((fraction_of(SEED, index) - HALF) * MIX_SPREAD);
    }
    for (uint32_t index = 0; index < SCALES; index++) {
        scale_in[index] = (float)(fraction_of(SEED, MIXES + index) * SCALE_SPREAD);
    }
    test_linear_values(weight_in, HIDDEN, SEED + OUTPUT_INPUT, TEST_LINEAR_VALUES_PLAIN);
    test_linear_values(&embedding_in[0][0], VOCABULARY * HIDDEN, SEED + NUMBERS_INPUT, TEST_LINEAR_VALUES_PLAIN);
    for (uint32_t index = 0; index < HIDDEN; index++) {
        const double small =
            TOLD_APART_LEAST + fraction_of(SEED + STREAMS_PER_ROW * (TOLD_APART_ROW + 1u), index) * TOLD_APART_SPREAD;
        streams_in[TOLD_APART_ROW][index] = test_linear_bfloat16_nearest(index == 0 ? TOLD_APART_LARGE : small);
    }
    for (uint32_t stream = 0; stream < STREAMS; stream++) {
        carried_in[TOLD_APART_ROW][stream] = stream == 0 ? 1.0f : 0.0f;
    }
}

/* The break's words put over a case laid out in the image, behind the reference's back; token: row 0's row of the
 * embedding. */
static bool break_put(struct test_linear_card *card, const struct layout *layout, uint32_t token)
{
    const float scale = scale_in[0] + 1.0f;
    const uint16_t output = test_linear_bfloat16_nearest(BROKEN_OUTPUT);
    const uint16_t embedding = (uint16_t)(embedding_in[token][0] ^ 1u);
    uint16_t weight[BROKEN_WEIGHT_WORDS];
    for (uint32_t index = 0; index < BROKEN_WEIGHT_WORDS; index++) {
        weight[index] = (uint16_t)(weight_in[HIDDEN - BROKEN_WEIGHT_WORDS + index] ^ BFLOAT16_SIGN_BIT);
    }
    return (break_category != SINKHORN ||
	    test_linear_card_put(card, layout->scale, 0, &scale,
				 sizeof(scale))) &&
	   (break_category != NORM ||
	    test_linear_card_put(card, layout->weight,
				 (HIDDEN - BROKEN_WEIGHT_WORDS) *
					 BFLOAT16_BYTES,
				 weight, sizeof(weight))) &&
	   (break_category != POST ||
	    test_linear_card_put(card, layout->output,
				 (HIDDEN - 1u) * BFLOAT16_BYTES, &output,
				 sizeof(output))) &&
	   (break_category != EMBED ||
	    test_linear_card_put(card, layout->embedding,
				 (uint64_t)token * MODEL_HIDDEN_ROW_BYTES,
				 &embedding, sizeof(embedding)));
}

/* Lays the synthetic case out in the image, with the break. */
static bool synthetic_put(struct test_linear_card *card, const struct layout *layout)
{
    return test_linear_card_put(card, layout->streams, 0, streams_in, sizeof(streams_in)) &&
           test_linear_card_put(card, layout->carried, 0, carried_in, sizeof(carried_in)) &&
           test_linear_card_put(card, layout->mixes, 0, mixes_in, sizeof(mixes_in)) &&
           test_linear_card_put(card, layout->output, 0, output_in, sizeof(output_in)) &&
           test_linear_card_put(card, layout->sums, 0, sums_in, sizeof(sums_in)) &&
           test_linear_card_put(card, layout->step, 0, step_in, sizeof(step_in)) &&
           test_linear_card_put(card, layout->scale, 0, scale_in, sizeof(scale_in)) &&
           test_linear_card_put(card, layout->base, 0, base_in, sizeof(base_in)) &&
           test_linear_card_put(card, layout->weight, 0, weight_in, sizeof(weight_in)) &&
           test_linear_card_put(card, layout->embedding, 0, embedding_in, sizeof(embedding_in)) &&
           break_put(card, layout, TOKEN_FIRST % VOCABULARY);
}

/* What a run left in the rows of the outputs. */
static bool outputs_get(const struct test_linear_card *card, const struct layout *layout, uint32_t row_count)
{
    const uint64_t rows = row_count;
    return test_linear_card_get(card, layout->streams, 0, streams_out,
				rows * STREAMS_BYTES) &&
	   test_linear_card_get(card, layout->carried, 0, carried_out,
				rows * MODEL_SITE_ROW_BYTES) &&
	   test_linear_card_get(card, layout->post, 0, post_out,
				rows * MODEL_SITE_ROW_BYTES) &&
	   test_linear_card_get(card, layout->comb, 0, comb_out,
				rows * MODEL_COMB_ROW_BYTES) &&
	   test_linear_card_get(card, layout->hidden, 0, hidden_out,
				rows * MODEL_HIDDEN_ROW_BYTES);
}

/* Of the collapse in collapsed: the places from the factor root_bits of the nearest factor (tried nearest first) for
 * which the norm in the records' order gives hidden, or NO_FACTOR. */
static uint32_t factor_places(const uint16_t *hidden, const uint16_t *weight, uint32_t root_bits)
{
    for (uint32_t step = 0; step <= RSQRT_PLACES + RSQRT_PLACES; step++) {
        const uint32_t places = (step + 1u) >> 1u;     /* 0, then 1 above, 1 below, 2 above, ... */
        test_hyper_norm_with(collapsed, weight, float_of((step & 1u) != 0 ? root_bits + places : root_bits - places),
                             ordered_normed);
        if (memcmp(ordered_normed, hidden, sizeof(ordered_normed)) == 0) {
            return places;
        }
    }
    return NO_FACTOR;
}

/* hidden against the norm of the collapse of streams with pre: within one bfloat16 place of the double norm; and bit
 * for bit the norm in the records' order for a factor near the reciprocal root of its operand, on TOLD_APART_ROW for
 * none near that of the squares added from +0 in order. */
static void expect_normed(const uint16_t *hidden, const uint16_t *streams, const float *pre, const uint16_t *weight,
                          uint32_t row, struct measures *measures)
{
    float square_sum = 0.0f;
    test_hyper_collapse(streams, pre, collapsed);
    test_hyper_norm(collapsed, weight, normed);
    for (uint32_t index = 0; index < HIDDEN; index++) {
        const uint16_t nearest = test_linear_bfloat16_nearest(normed[index]);
        const float value = (float)test_linear_bfloat16_value(collapsed[index]);
        square_sum = square_sum + value * value;
        measures->normed_equal += hidden[index] == nearest ? 1u : 0u;
        if (bfloat16_places(hidden[index], nearest) > 1) {
            category_fail(NORM, "the norm is more than one bfloat16 place from the reference's", row, index);
        }
    }
    measures->normed_compared += HIDDEN;
    const uint32_t places =
        factor_places(hidden, weight, bits_of(test_hyper_reciprocal_root(test_hyper_norm_operand(collapsed))));
    const uint32_t in_order_root_bits = bits_of(test_hyper_reciprocal_root(
	    square_sum * MEAN_FACTOR + MODEL_NORM_EPSILON));
    if (places == NO_FACTOR) {
        category_fail(NORM, "no factor near the reciprocal root gives the norm in the records' order", row, 0);
    } else if (places > measures->largest_factor_places) {
        measures->largest_factor_places = places;
    }
    if (row == TOLD_APART_ROW && factor_places(hidden, weight, in_order_root_bits) != NO_FACTOR) {
        category_fail(NORM, "a factor near the root of the squares added in order gives the norm too", row, 0);
    }
}

/* The outputs of the final collapse and the embedding of one row. */
static void final_check(uint32_t row, struct measures *measures)
{
    const uint32_t token = (TOKEN_STEP * row + TOKEN_FIRST) % VOCABULARY;
    expect_normed(hidden_out[row], streams_in[row], carried_in[row], weight_in, row, measures);
    for (uint32_t index = 0; index < STREAM_VALUES; index++) {
        if (streams_out[row][index] != embedding_in[token][index % HIDDEN]) {
            category_fail(EMBED, "a stream is not the token's row", row, index);
        }
    }
    for (uint32_t stream = 0; stream < STREAMS; stream++) {
        const float expected = stream == 0 ? 1.0f : 0.0f;
        if (bits_of(carried_out[row][stream]) != bits_of(expected)) {
            category_fail(EMBED, "the carried pre is not (1, 0, 0, 0)", row, stream);
        }
    }
}

/* The outputs of one site of one row: Sinkhorn, the norm, the post. */
static void site_check(enum run_kind run, uint32_t row, struct measures *measures)
{
    float pre[STREAMS];
    float post[STREAMS];
    float comb[COMB_VALUES];
    test_hyper_sinkhorn(mixes_in[row], scale_in, base_in, pre, post, comb);
    for (uint32_t index = 0; index < STREAMS; index++) {
        const uint32_t pre_places = test_hyper_float_places(carried_out[row][index], pre[index]);
        const uint32_t post_places = test_hyper_float_places(post_out[row][index], post[index]);
        const uint32_t places = pre_places > post_places ? pre_places : post_places;
        measures->largest_places = places > measures->largest_places ? places : measures->largest_places;
        if (places > TEST_HYPER_SINKHORN_PLACES) {
            category_fail(SINKHORN, "pre or post is too far from the reference's", row, index);
        }
    }
    for (uint32_t index = 0; index < COMB_VALUES; index++) {
        const double difference = magnitude_of((double)comb_out[row][index] - (double)comb[index]);
        measures->largest_comb = difference > measures->largest_comb ? difference : measures->largest_comb;
        if (!(difference <= TEST_HYPER_COMB_TOLERANCE)) {
            category_fail(SINKHORN, "comb is too far from the reference's", row, index);
        }
    }
    expect_normed(hidden_out[row], streams_in[row], carried_in[row], weight_in, row, measures);
    test_hyper_post(streams_in[row], post_out[row], comb_out[row], output_in[row],
                    run == FFN_RUN ? sums_in[row] : NULL, expected_streams);
    for (uint32_t index = 0; index < STREAM_VALUES; index++) {
        if (streams_out[row][index] != expected_streams[index]) {
            category_fail(POST, "a stream after the post is not the reference's", row, index);
        }
    }
}

/* R: each row of the image moved to row 0 (its step row too) and run alone in a launch of one row; every output
 * byte of that row must be the byte the launch of all rows left. The image's row 0 is put back after each. */
static bool rows_alone(struct test_linear_card *card, const struct layout *layout, enum run_kind run)
{
    static unsigned char all_rows[OUTPUT_ARRAYS][ROWS * STREAMS_BYTES];
    static unsigned char row_zero[INPUT_ARRAYS][STREAMS_BYTES];
    static unsigned char moved[STREAMS_BYTES];
    const struct row_array inputs[INPUT_ARRAYS] = {
	    { layout->streams, 0, STREAMS_BYTES },
	    { layout->carried, 0, MODEL_SITE_ROW_BYTES },
	    { layout->mixes, 0, MODEL_MIXES_ROW_BYTES },
	    { layout->output, 0, MODEL_HIDDEN_ROW_BYTES },
	    { layout->sums, 0, HIDDEN * sizeof(float) },
	    { layout->step, STEP_BLOCK_BYTES_VALUE, STEP_ROW_BYTES_VALUE }
    };
    const struct row_array outputs[OUTPUT_ARRAYS] = {
	    { layout->streams, 0, STREAMS_BYTES },
	    { layout->carried, 0, MODEL_SITE_ROW_BYTES },
	    { layout->post, 0, MODEL_SITE_ROW_BYTES },
	    { layout->comb, 0, MODEL_COMB_ROW_BYTES },
	    { layout->hidden, 0, MODEL_HIDDEN_ROW_BYTES }
    };
    bool is_run = true;
    for (uint32_t index = 0; is_run && index < OUTPUT_ARRAYS; index++) {
        is_run = test_linear_card_get(card, outputs[index].range, 0, all_rows[index], ROWS * outputs[index].row_bytes);
    }
    for (uint32_t index = 0; is_run && index < INPUT_ARRAYS; index++) {
        const uint8_t *const image = test_linear_card_stored(card, inputs[index].range);
        is_run = image != NULL;
        if (is_run) {
            memcpy(row_zero[index], image + inputs[index].first_byte, inputs[index].row_bytes);
        }
    }
    for (uint32_t row = 0; is_run && row < ROWS; row++) {
        for (uint32_t index = 0; is_run && index < INPUT_ARRAYS; index++) {
            const struct row_array *const array = &inputs[index];
            const uint8_t *const image = test_linear_card_stored(card, array->range);
            is_run = image != NULL;
            if (is_run) {
                memcpy(moved, image + array->first_byte + row * array->row_bytes, array->row_bytes);
                is_run = test_linear_card_put(card, array->range, array->first_byte, moved, array->row_bytes);
            }
        }
        is_run = is_run && run_and_fetch(card, layout, run, ONE_ROW, 0, LAUNCHES);
        for (uint32_t index = 0; is_run && index < OUTPUT_ARRAYS; index++) {
            is_run = test_linear_card_get(card, outputs[index].range, 0, moved, outputs[index].row_bytes);
            const uint64_t row_bytes = outputs[index].row_bytes;
            if (is_run && memcmp(moved, all_rows[index] + row * row_bytes, row_bytes) != 0) {
                category_fail(ALONE, "a row run alone does not give the bits of the launch of all rows", row, index);
            }
        }
        for (uint32_t index = 0; is_run && index < INPUT_ARRAYS; index++) {
            is_run = test_linear_card_put(card, inputs[index].range, inputs[index].first_byte, row_zero[index],
                                          inputs[index].row_bytes);
        }
    }
    return is_run;
}

/* The three synthetic runs on one card, each with its rows alone; what they measured, printed. */
static bool synthetic_runs(struct test_linear_card *card, const struct layout *layout)
{
    static const char *const run_names[RUN_KINDS] = {"an attention site", "an FFN site",
                                                     "the final collapse and the embedding"};
    struct measures measures;
    memset(&measures, 0, sizeof(measures));
    for (uint32_t run = 0; run < RUN_KINDS; run++) {
        test_linear_context("card %" PRIu32 ", %s", card->card_index, run_names[run]);
        if (!synthetic_put(card, layout) || !run_and_fetch(card, layout, (enum run_kind)run, ALL_ROWS, 0, LAUNCHES) ||
            !outputs_get(card, layout, ROWS)) {
            return false;
        }
        for (uint32_t row = 0; row < ROWS; row++) {
            if (run == FINAL_RUN) {
                final_check(row, &measures);
            } else {
                site_check((enum run_kind)run, row, &measures);
            }
        }
        if (!rows_alone(card, layout, (enum run_kind)run) || !test_linear_card_expect_weights_untouched(card)) {
            return false;
        }
    }
    (void)printf("card %" PRIu32 " (class %" PRIu32 "): synthetic rows: pre and post within %" PRIu32 " float32 "
                 "places, comb within %.3g, normed inputs %.4f%% the double norm rounded once and the records' norm "
                 "with a factor %" PRIu32 " places from the reciprocal root at most\n", card->card_index,
                 card->arithmetic_class, measures.largest_places, measures.largest_comb,
                 test_linear_percent(measures.normed_equal, measures.normed_compared), measures.largest_factor_places);
    return true;
}

/* ---- the draft's inputs ---- */

/* main_norm's output of one plain row against the norm of that row: within one bfloat16 place of the double norm,
 * and bit for bit the records' norm for a factor near the reciprocal root of its operand; on TOLD_APART_ROW for none
 * near that of the squares added from +0 in order. */
static void expect_row_normed(const uint16_t *normed_row, const uint16_t *row_values, uint32_t row)
{
    float square_sum = 0.0f;
    memcpy(collapsed, row_values, sizeof(collapsed));
    test_hyper_norm(collapsed, weight_in, normed);
    for (uint32_t index = 0; index < HIDDEN; index++) {
        const float value = (float)test_linear_bfloat16_value(collapsed[index]);
        square_sum = square_sum + value * value;
        if (bfloat16_places(normed_row[index], test_linear_bfloat16_nearest(normed[index])) > 1) {
            category_fail(NORM, "main_norm is more than one bfloat16 place from the reference's", row, index);
        }
    }
    if (factor_places(normed_row, weight_in,
                      bits_of(test_hyper_reciprocal_root(test_hyper_norm_operand(collapsed)))) == NO_FACTOR) {
        category_fail(NORM, "no factor near the reciprocal root gives main_norm in the records' order", row, 0);
    }
    if (row == TOLD_APART_ROW &&
	factor_places(normed_row, weight_in,
		      bits_of(test_hyper_reciprocal_root(
			      square_sum * MEAN_FACTOR +
			      MODEL_NORM_EPSILON))) != NO_FACTOR) {
	    category_fail(
		    NORM,
		    "a factor near the root of the squares added in order gives main_norm too",
		    row, 0);
    }
}

/* The draft's inputs on one card (the header's K1): one run of the final keeping its collapse, the three taps and the
 * two main_norms, every output held to the reference and to the final. */
static bool draft_runs(struct test_linear_card *card,
		       const struct hyper_kernels_record *kernels)
{
	static const float quarters[STREAMS] = { HYPER_STREAM_MEAN_PRE,
						 HYPER_STREAM_MEAN_PRE,
						 HYPER_STREAM_MEAN_PRE,
						 HYPER_STREAM_MEAN_PRE };
	static uint16_t tap_expected[HIDDEN];
	struct kernel_launch launches[DRAFT_LAUNCHES];
	const uint32_t failures_before =
		category_failures[DRAFT] + category_failures[NORM];
	test_linear_context("card %" PRIu32 ", the draft's inputs",
			    card->card_index);
	for (uint32_t row = 0; row < DRAFT_ROWS; row++) {
		test_linear_values(draft_streams[row], STREAM_VALUES,
				   DRAFT_SEED + row,
				   (row & 1u) == 0 ? TEST_LINEAR_VALUES_PLAIN :
						     TEST_LINEAR_VALUES_SMALL);
		test_linear_values(draft_rows_in[row], HIDDEN,
				   DRAFT_SEED + DRAFT_ROWS + row,
				   TEST_LINEAR_VALUES_PLAIN);
	}
    memcpy(draft_rows_in[TOLD_APART_ROW], streams_in[TOLD_APART_ROW], sizeof(draft_rows_in[TOLD_APART_ROW]));
    for (uint32_t row = 0; row < ROWS; row++) {
        memcpy(draft_carried[row], quarters, sizeof(quarters));
    }
    test_linear_card_clear(card);
    const struct linear_range streams =
	    test_linear_card_work(card, (uint64_t)DRAFT_ROWS * STREAMS_BYTES);
    const struct linear_range carried =
	    test_linear_card_work(card, sizeof(draft_carried));
    const struct linear_range hidden =
	    test_linear_card_work(card, sizeof(draft_final_out));
    const struct linear_range collapse =
	    test_linear_card_work(card, sizeof(draft_collapse_out));
    const struct linear_range taps =
	    test_linear_card_work(card, sizeof(draft_taps_out));
    const struct linear_range rows_in =
	    test_linear_card_work(card, sizeof(draft_rows_in));
    const struct linear_range normed_rows =
	    test_linear_card_work(card, sizeof(draft_normed_out));
    const struct linear_range collapse_normed =
	    test_linear_card_work(card, sizeof(draft_collapse_normed_out));
    const struct linear_range weight =
	    test_linear_card_weights(card, MODEL_HIDDEN_ROW_BYTES);
    const struct hyper_final_description final = { .streams = streams,
						   .carried_pre = carried,
						   .norm_weight = weight,
						   .hidden = hidden,
						   .collapse = collapse,
						   .row_count = ROWS };
    const struct hyper_draft_inputs_description rows_norm = {
	    .input = rows_in,
	    .weight = weight,
	    .output = normed_rows,
	    .row_count = DRAFT_ROWS,
	    .mode = HYPER_ROW_NORM
    };
    const struct hyper_draft_inputs_description collapse_norm = {
	    .input = collapse,
	    .weight = weight,
	    .output = collapse_normed,
	    .row_count = ROWS,
	    .mode = HYPER_ROW_NORM
    };
    const uint16_t broken_word = (uint16_t)(draft_streams[0][BROKEN_TAP_WORD] ^ 1u);
    bool is_run =
	    test_linear_card_has_room(card) &&
	    test_linear_expect_status(
		    hyper_final_prepare(&final, card->arena_byte_count, kernels,
					&launches[DRAFT_FINAL]),
		    STATUS_OK, "the final keeping its collapse is prepared") &&
	    test_linear_expect_status(
		    hyper_draft_inputs_prepare(&rows_norm,
					       card->arena_byte_count, kernels,
					       &launches[DRAFT_ROWS_NORM]),
		    STATUS_OK, "main_norm of the rows is prepared") &&
	    test_linear_expect_status(
		    hyper_draft_inputs_prepare(&collapse_norm,
					       card->arena_byte_count, kernels,
					       &launches[DRAFT_COLLAPSE_NORM]),
		    STATUS_OK, "main_norm of the collapse is prepared");
    for (uint32_t tap = 0; is_run && tap < DRAFT_TAPS; tap++) {
	    const struct hyper_draft_inputs_description tap_description = {
		    .input = streams,
		    .output = taps,
		    .row_count = DRAFT_ROWS,
		    .mode = HYPER_STREAM_MEAN,
		    .tap = tap
	    };
	    is_run = test_linear_expect_status(
		    hyper_draft_inputs_prepare(
			    &tap_description, card->arena_byte_count, kernels,
			    &launches[DRAFT_FIRST_TAP + tap]),
		    STATUS_OK, "a tap is prepared");
    }
    is_run = is_run && test_linear_card_put(card, streams, 0, draft_streams, sizeof(draft_streams)) &&
             test_linear_card_put(card, carried, 0, draft_carried, sizeof(draft_carried)) &&
             test_linear_card_put(card, rows_in, 0, draft_rows_in, sizeof(draft_rows_in)) &&
             test_linear_card_put(card, weight, 0, weight_in, sizeof(weight_in)) &&
             (break_category != DRAFT || test_linear_card_put(card, streams, BROKEN_TAP_WORD * BFLOAT16_BYTES,
                                                              &broken_word, sizeof(broken_word))) &&
             test_linear_card_load(card) && test_linear_card_run(card, launches, DRAFT_LAUNCHES) &&
             test_linear_card_fetch(card) &&
             test_linear_card_get(card, taps, 0, draft_taps_out, sizeof(draft_taps_out)) &&
             test_linear_card_get(card, collapse, 0, draft_collapse_out, sizeof(draft_collapse_out)) &&
             test_linear_card_get(card, hidden, 0, draft_final_out, sizeof(draft_final_out)) &&
             test_linear_card_get(card, normed_rows, 0, draft_normed_out, sizeof(draft_normed_out)) &&
             test_linear_card_get(card, collapse_normed, 0, draft_collapse_normed_out,
                                  sizeof(draft_collapse_normed_out)) &&
             test_linear_card_expect_untouched(card, streams, "the streams") &&
             test_linear_card_expect_untouched(card, rows_in, "main_norm's input") &&
             test_linear_card_expect_weights_untouched(card);
    for (uint32_t row = 0; is_run && row < DRAFT_ROWS; row++) {
        test_hyper_collapse(draft_streams[row], quarters, tap_expected);
        for (uint32_t tap = 0; tap < DRAFT_TAPS; tap++) {
            for (uint32_t index = 0; index < HIDDEN; index++) {
                const uint16_t value = draft_taps_out[row][tap * HIDDEN + index];
                if (value != tap_expected[index]) {
                    category_fail(DRAFT, "a tap is not the reference's mean of the streams", row, index);
                }
                if (row < ROWS && value != draft_collapse_out[row][index]) {
                    category_fail(DRAFT, "a tap is not the final's kept collapse with pre 1/4", row, index);
                }
            }
        }
        if (row < ROWS && memcmp(draft_collapse_normed_out[row], draft_final_out[row], HIDDEN * BFLOAT16_BYTES) != 0) {
            category_fail(DRAFT, "main_norm of the kept collapse is not the final's norm of it", row, 0);
        }
        expect_row_normed(draft_normed_out[row], draft_rows_in[row], row);
    }
    if (is_run) {
        (void)printf("card %" PRIu32 " (class %" PRIu32 "): the draft's inputs: %u taps of %u rows each the "
                     "reference's mean and the final's kept collapse, main_norm the final's norm and the records' "
                     "norm: %s\n", card->card_index, card->arithmetic_class, DRAFT_TAPS, DRAFT_ROWS,
                     category_failures[DRAFT] + category_failures[NORM] == failures_before ? "bit for bit"
                                                                                           : "FAILED");
    }
    return is_run;
}

/* ---- the replay ---- */

/* Bytes of a plain tensor of the pack, from first_byte on: false, behind a failure line, where the pack has no such
 * tensor, it is of another element type or count, or the bytes lie outside it. */
static bool pack_read(const struct replay_sources *sources, const char *name, uint32_t element_type,
                      uint64_t element_count, uint64_t first_byte, void *destination, uint64_t byte_count)
{
	const uint32_t tensor_index =
		pack_find_tensor(sources->pack, name, (uint32_t)strlen(name));
	if (tensor_index == PACK_NOT_FOUND) {
		test_linear_fail("the pack holds no tensor %s", name);
		return false;
	}
	const struct pack_tensor *const tensor =
		&pack_tensors_of(sources->pack)[tensor_index];
	const struct pack_extent *const extent =
		&pack_extents_of(sources->pack)[tensor->extent_index];
	uint64_t elements = 1;
	for (uint32_t dimension = 0; dimension < tensor->dimension_count;
	     dimension++) {
		elements *= tensor->dimensions[dimension];
	}
    const uint64_t alignment = sources->block_alignment_bytes;
    const uint64_t file_byte = extent->byte_offset + tensor->byte_offset_in_extent + first_byte;
    const uint64_t lead_byte_count = file_byte % alignment;
    const uint64_t read_byte_count = (lead_byte_count + byte_count + alignment - 1) / alignment * alignment;
    if (tensor->weight_format != WEIGHT_FORMAT_NONE ||
	tensor->element_type != element_type || elements != element_count ||
	first_byte > tensor->byte_count ||
	byte_count > tensor->byte_count - first_byte ||
	read_byte_count > sizeof(read_buffer)) {
	    test_linear_fail("%s is not a plain tensor of %" PRIu64
			     " elements of type %" PRIu32
			     " that holds bytes %" PRIu64 " to %" PRIu64,
			     name, element_count, element_type, first_byte,
			     first_byte + byte_count);
	    return false;
    }
    if (!test_linear_expect_status(
		storage_direct.reader_read(sources->reader, 0,
					   file_byte - lead_byte_count,
					   read_buffer, read_byte_count),
		STATUS_OK, "a tensor is read from the pack")) {
	    return false;
    }
    memcpy(destination, read_buffer + lead_byte_count, byte_count);
    return true;
}

/* element_count elements of a capture record from first_element on, of the record's own shape. */
static bool capture_read(const struct replay_sources *sources, const char *name, enum test_capture_type type,
                         uint64_t first_element, uint64_t element_count, void *destination)
{
    const struct test_capture_tensor *const tensor = test_capture_find(sources->capture, name);
    if (tensor == NULL) {
        test_linear_fail("the capture holds no record %s", name);
        return false;
    }
    return test_linear_expect_status(
	    test_capture_read(sources->capture, name, type, tensor->dimensions,
			      tensor->dimension_count, first_element,
			      element_count, destination,
			      element_count * test_capture_element_bytes(type)),
	    STATUS_OK, "a record of the capture is read");
}

/* layer and position of a name layer_<layer>.position_<position>.attn_site.mixes, or false. */
static bool point_of(const char *name, uint32_t *layer_out, uint32_t *position_out)
{
    static const char layer_word[] = "layer_";
    static const char position_word[] = ".position_";
    static const char suffix[] = ".attn_site.mixes";
    char *end = NULL;
    if (strncmp(name, layer_word, sizeof(layer_word) - 1) != 0 || name[sizeof(layer_word) - 1] < '0' ||
        name[sizeof(layer_word) - 1] > '9') {
        return false;
    }
    const unsigned long layer = strtoul(name + sizeof(layer_word) - 1, &end, DECIMAL);
    if (strncmp(end, position_word, sizeof(position_word) - 1) != 0 || end[sizeof(position_word) - 1] < '0' ||
        end[sizeof(position_word) - 1] > '9') {
        return false;
    }
    const unsigned long position = strtoul(end + sizeof(position_word) - 1, &end, DECIMAL);
    if (strcmp(end, suffix) != 0 || layer >= MODEL_LAYERS ||
	position >= MODEL_CONTEXT_WITHOUT_CANDIDATES) {
	    return false;
    }
    *layer_out = (uint32_t)layer;
    *position_out = (uint32_t)position;
    return true;
}

/* Of one record, the bfloat16 values of the card against the captured, the bit-equal ones counted: it fails with a
 * share within one place (is_bit_equal: bit-equal, and every value within one place) below per_thousand. */
static void expect_share(const uint16_t *card_values, const uint16_t *captured, uint64_t count, bool is_bit_equal,
                         uint32_t per_thousand, enum category category, enum comparison comparison,
                         struct shares *shares, uint32_t layer)
{
    uint64_t equal = 0;
    uint64_t near = 0;
    for (uint64_t index = 0; index < count; index++) {
        equal += card_values[index] == captured[index] ? 1u : 0u;
        near += bfloat16_places(card_values[index], captured[index]) <= 1 ? 1u : 0u;
    }
    shares->compared[comparison] += count;
    shares->equal[comparison] += equal;
    if ((is_bit_equal ? equal : near) * PER_THOUSAND < count * per_thousand || (is_bit_equal && near < count)) {
        category_fail(category, comparison_names[comparison], layer, (uint32_t)(is_bit_equal ? equal : near));
    }
}

/* One site at one capture point, as one row: its mix finish from the inputs of the capture and the pack, then its
 * post alone from the captured post and comb, so that the card's Sinkhorn is charged nothing there (plan 3.2). */
static bool site_replay(struct test_linear_card *card, const struct layout *layout,
                        const struct replay_sources *sources, uint32_t layer, uint32_t position, bool is_ffn,
                        struct shares *shares)
{
    char point[NAME_CAPACITY];
    char name[NAME_CAPACITY];
    const char *const site = is_ffn ? "ffn_site" : "attn_site";
    const enum run_kind run = is_ffn ? FFN_RUN : ATTENTION_RUN;
    const char *const parameters = is_ffn ? "ffn" : "attn";
    (void)snprintf(point, sizeof(point), "layer_%" PRIu32 ".position_%" PRIu32 ".", layer, position);
#define RECORD(format, ...) (snprintf(name, sizeof(name), format, __VA_ARGS__) > 0 ? name : "")
    bool is_read =
	    capture_read(sources, RECORD("%s%s.streams", point, site),
			 TEST_CAPTURE_BFLOAT16, 0, STREAM_VALUES,
			 streams_in[0]) &&
	    capture_read(sources, RECORD("%s%s.mixes", point, site),
			 TEST_CAPTURE_FLOAT32, 0, MIXES, mixes_in[0]) &&
	    capture_read(sources,
			 RECORD("%sattn_site.%s", point,
				is_ffn ? "pre" : "carried_pre"),
			 TEST_CAPTURE_FLOAT32, 0, STREAMS, carried_in[0]) &&
	    capture_read(sources, RECORD("%s%s.pre", point, site),
			 TEST_CAPTURE_FLOAT32, 0, STREAMS, captured_pre) &&
	    capture_read(sources, RECORD("%s%s.post", point, site),
			 TEST_CAPTURE_FLOAT32, 0, STREAMS, captured_post) &&
	    capture_read(sources, RECORD("%s%s.comb", point, site),
			 TEST_CAPTURE_FLOAT32, 0, (uint64_t)COMB_VALUES,
			 captured_comb) &&
	    capture_read(sources, RECORD("%s%s.collapsed", point, site),
			 TEST_CAPTURE_BFLOAT16, 0, HIDDEN,
			 captured_collapsed) &&
	    capture_read(sources, RECORD("%s%s.normed", point, site),
			 TEST_CAPTURE_BFLOAT16, 0, HIDDEN, captured_normed) &&
	    capture_read(sources,
			 RECORD("%s%s", point,
				is_ffn ? "shared_expert.output" : "attn.wo_b"),
			 TEST_CAPTURE_BFLOAT16, 0, HIDDEN, output_in[0]) &&
	    (!is_ffn ||
	     capture_read(sources, RECORD("%sexperts.row_sum", point),
			  TEST_CAPTURE_FLOAT32, 0, HIDDEN, sums_in[0])) &&
	    (is_ffn ? capture_read(sources,
				   RECORD("layer_%" PRIu32 ".block.output",
					  layer),
				   TEST_CAPTURE_BFLOAT16,
				   (uint64_t)position * STREAM_VALUES,
				   STREAM_VALUES, expected_streams) :
		      capture_read(sources, RECORD("%sffn_site.streams", point),
				   TEST_CAPTURE_BFLOAT16, 0, STREAM_VALUES,
				   expected_streams)) &&
	    pack_read(sources,
		      RECORD("layers.%" PRIu32 ".hc_%s_scale", layer,
			     parameters),
		      PACK_ELEMENT_FLOAT32, SCALES, 0, scale_in,
		      sizeof(scale_in)) &&
	    pack_read(
		    sources,
		    RECORD("layers.%" PRIu32 ".hc_%s_base", layer, parameters),
		    PACK_ELEMENT_FLOAT32, MIXES, 0, base_in, sizeof(base_in)) &&
	    pack_read(sources,
		      RECORD("layers.%" PRIu32 ".%s_norm.weight", layer,
			     parameters),
		      PACK_ELEMENT_BFLOAT16, HIDDEN, 0, weight_in,
		      sizeof(weight_in));
#undef RECORD
    if (!is_read ||
	!test_linear_card_put(card, layout->streams, 0, streams_in[0],
			      STREAMS_BYTES) ||
	!test_linear_card_put(card, layout->carried, 0, carried_in[0],
			      MODEL_SITE_ROW_BYTES) ||
	!test_linear_card_put(card, layout->mixes, 0, mixes_in[0],
			      MODEL_MIXES_ROW_BYTES) ||
	!test_linear_card_put(card, layout->output, 0, output_in[0],
			      MODEL_HIDDEN_ROW_BYTES) ||
	!test_linear_card_put(card, layout->sums, 0, sums_in[0],
			      HIDDEN * sizeof(float)) ||
	!test_linear_card_put(card, layout->scale, 0, scale_in,
			      sizeof(scale_in)) ||
	!test_linear_card_put(card, layout->base, 0, base_in,
			      sizeof(base_in)) ||
	!test_linear_card_put(card, layout->weight, 0, weight_in,
			      sizeof(weight_in)) ||
	!break_put(card, layout, 0) ||
	!run_and_fetch(card, layout, run, ONE_ROW, HYPER_SITE_MIX_FINISH,
		       ONE_LAUNCH) ||
	!outputs_get(card, layout, 1)) {
	    return false;
    }
    const float *const card_values[] = {carried_out[0], post_out[0], comb_out[0]};
    const float *const captured_values[] = {captured_pre, captured_post, captured_comb};
    const uint32_t counts[] = {STREAMS, STREAMS, COMB_VALUES};
    for (uint32_t part = 0; part < sizeof(counts) / sizeof(counts[0]); part++) {
        for (uint32_t index = 0; index < counts[part]; index++) {
            const double captured_value = (double)captured_values[part][index];
            const double difference = magnitude_of((double)card_values[part][index] - captured_value);
            const bool is_equal = bits_of(card_values[part][index]) == bits_of(captured_values[part][index]);
            shares->compared[PRE_POST_COMB]++;
            shares->equal[PRE_POST_COMB] += is_equal ? 1u : 0u;
	    if (!(difference <=
		  REPLAY_RELATIVE * magnitude_of(captured_value)) ||
		(!is_equal &&
		 card->arithmetic_class == ARITHMETIC_CLASS_SM_120)) {
		    category_fail(
			    SINKHORN,
			    "pre, post or comb is too far from the captured (class 120: not its bits)",
			    layer, index);
	    }
	}
    }
    test_hyper_collapse(streams_in[0], carried_in[0], collapsed);
    expect_share(collapsed, captured_collapsed, HIDDEN, true, SHARE_PER_THOUSAND, COLLAPSE, HOST_COLLAPSE, shares,
                 layer);
    expect_share(hidden_out[0], captured_normed, HIDDEN, false, SHARE_PER_THOUSAND, NORM, NORMED, shares, layer);
    if (!test_linear_card_put(card, layout->post, 0, captured_post,
			      sizeof(captured_post)) ||
	!test_linear_card_put(card, layout->comb, 0, captured_comb,
			      sizeof(captured_comb)) ||
	!run_and_fetch(card, layout, run, ONE_ROW, HYPER_SITE_POST,
		       ONE_LAUNCH) ||
	!outputs_get(card, layout, 1)) {
	    return false;
    }
    expect_share(streams_out[0], expected_streams, STREAM_VALUES, true, SHARE_PER_THOUSAND, POST, AFTER_POST, shares,
                 layer);
    return true;
}

/* The final collapse at a capture point of layer 39, as one row: its output and carried weights, norm.weight. */
static bool final_replay(struct test_linear_card *card, const struct layout *layout,
                         const struct replay_sources *sources, uint32_t position, struct shares *shares)
{
    char name[NAME_CAPACITY];
    char output_name[NAME_CAPACITY];
    char weights_name[NAME_CAPACITY];
    (void)snprintf(name, sizeof(name), "position_%" PRIu32 ".final.normed", position);
    (void)snprintf(output_name, sizeof(output_name), "layer_%" PRIu32 ".block.output", FINAL_LAYER);
    (void)snprintf(weights_name, sizeof(weights_name), "layer_%" PRIu32 ".block.output_collapse_weights", FINAL_LAYER);
    if (!capture_read(sources, name, TEST_CAPTURE_BFLOAT16, 0, HIDDEN,
		      captured_normed) ||
	!capture_read(sources, output_name, TEST_CAPTURE_BFLOAT16,
		      (uint64_t)position * STREAM_VALUES, STREAM_VALUES,
		      streams_in[0]) ||
	!capture_read(sources, weights_name, TEST_CAPTURE_FLOAT32,
		      (uint64_t)position * STREAMS, STREAMS, carried_in[0]) ||
	!pack_read(sources, "norm.weight", PACK_ELEMENT_BFLOAT16, HIDDEN, 0,
		   weight_in, sizeof(weight_in)) ||
	!test_linear_card_put(card, layout->streams, 0, streams_in[0],
			      STREAMS_BYTES) ||
	!test_linear_card_put(card, layout->carried, 0, carried_in[0],
			      MODEL_SITE_ROW_BYTES) ||
	!test_linear_card_put(card, layout->weight, 0, weight_in,
			      sizeof(weight_in)) ||
	!break_put(card, layout, 0) ||
	!run_and_fetch(card, layout, FINAL_RUN, ONE_ROW, FINAL_LAUNCH,
		       ONE_LAUNCH) ||
	!outputs_get(card, layout, 1)) {
	    return false;
    }
    expect_share(hidden_out[0], captured_normed, HIDDEN, false, SHARE_PER_THOUSAND, NORM, FINAL_NORMED, shares,
                 FINAL_LAYER);
    return true;
}

/* The embedding of every captured position, ROWS at a time: the pack's rows of their tokens as the test's table. */
static bool embed_replay(struct test_linear_card *card, const struct layout *layout,
                         const struct replay_sources *sources, struct shares *shares)
{
    static const char tokens_name[] = "input.token_ids";
    static const char embeddings_name[] = "input.embeddings";
    const struct test_capture_tensor *const tokens = test_capture_find(sources->capture, tokens_name);
    if (tokens == NULL || tokens->element_count == 0) {
        test_linear_fail("the capture holds no token ids");
        return false;
    }
    for (uint64_t first = 0; first < tokens->element_count; first += ROWS) {
        const uint64_t left = tokens->element_count - first;
        const uint32_t row_count = left < ROWS ? (uint32_t)left : ROWS;
        int64_t token_ids[ROWS];
        bool is_run = capture_read(sources, tokens_name, TEST_CAPTURE_INT64, first, row_count, token_ids) &&
                      capture_read(sources, embeddings_name, TEST_CAPTURE_BFLOAT16, first * HIDDEN,
                                   (uint64_t)row_count * HIDDEN, output_in);
        for (uint32_t row = 0; is_run && row < row_count; row++) {
		const struct step_row_record step_row = {
			.position = (int32_t)(first + row),
			.token_id = (int32_t)row,
			.session_slot = 0,
			.flags = STEP_ROW_LOGITS
		};
		is_run =
			token_ids[row] >= 0 &&
			token_ids[row] < (int64_t)MODEL_VOCABULARY_VALUE &&
			pack_read(sources, "embed.weight",
				  PACK_ELEMENT_BFLOAT16,
				  (uint64_t)MODEL_VOCABULARY_VALUE * HIDDEN,
				  (uint64_t)token_ids[row] *
					  MODEL_HIDDEN_ROW_BYTES,
				  embedding_in[row], MODEL_HIDDEN_ROW_BYTES) &&
			test_linear_card_put(
				card, layout->embedding,
				(uint64_t)row * MODEL_HIDDEN_ROW_BYTES,
				embedding_in[row], MODEL_HIDDEN_ROW_BYTES) &&
			test_linear_card_put(card, layout->step,
					     STEP_BLOCK_BYTES_VALUE +
						     row * STEP_ROW_BYTES_VALUE,
					     &step_row, sizeof(step_row));
	}
        if (!is_run || !break_put(card, layout, 0) ||
            !run_and_fetch(card, layout, FINAL_RUN, ALL_ROWS, EMBED_LAUNCH, ONE_LAUNCH) ||
            !outputs_get(card, layout, ROWS)) {
            return false;
        }
        for (uint32_t row = 0; row < row_count; row++) {
            for (uint32_t stream = 0; stream < STREAMS; stream++) {
                expect_share(&streams_out[row][(size_t)stream * HIDDEN], output_in[row], HIDDEN, true, PER_THOUSAND,
                             EMBED, EMBEDDED, shares, (uint32_t)(first + row));
            }
        }
    }
    return true;
}

/* Every capture point of the capture, both sites, the final collapse and the embedding, on one card; a break must
 * fail here too. */
static bool replay_runs(struct test_linear_card *card, const struct layout *layout,
                        const struct replay_sources *sources)
{
    struct shares shares;
    uint32_t point_count = 0;
    uint32_t final_count = 0;
    const uint32_t broken_before = break_category >= 0 ? category_failures[break_category] : 0;
    memset(&shares, 0, sizeof(shares));
    test_linear_context("card %" PRIu32 ", the replay", card->card_index);
    for (uint32_t index = 0; index < test_capture_tensor_count(sources->capture); index++) {
        uint32_t layer = 0;
        uint32_t position = 0;
        if (!point_of(test_capture_tensor_at(sources->capture, index)->name, &layer, &position)) {
            continue;
        }
        test_linear_context("card %" PRIu32 ", the replay at layer %" PRIu32 ", position %" PRIu32,
                            card->card_index, layer, position);
        if (!site_replay(card, layout, sources, layer, position, false, &shares) ||
            !site_replay(card, layout, sources, layer, position, true, &shares) ||
            (layer == FINAL_LAYER && !final_replay(card, layout, sources, position, &shares))) {
            return false;
        }
        point_count++;
        final_count += layer == FINAL_LAYER ? 1u : 0u;
    }
    test_linear_context("card %" PRIu32 ", the replay of the embedding", card->card_index);
    if (final_count == 0) {
        test_linear_fail("the capture holds no capture point of layer %u: nothing replays the final collapse",
                         FINAL_LAYER);
        return false;
    }
    if (!embed_replay(card, layout, sources, &shares) || !test_linear_card_expect_weights_untouched(card)) {
        return false;
    }
    for (uint32_t comparison = 0; comparison < COMPARISON_COUNT; comparison++) {
        (void)printf("card %" PRIu32 " (class %" PRIu32 "): replay of %" PRIu32 " points (%" PRIu32 " of the last "
                     "layer): %s %.4f%% bit-equal to the reference's (%" PRIu64 " values)\n", card->card_index,
                     card->arithmetic_class, point_count, final_count, comparison_names[comparison],
                     test_linear_percent(shares.equal[comparison], shares.compared[comparison]),
                     shares.compared[comparison]);
    }
    if (break_category >= 0 && break_category != DRAFT && category_failures[break_category] == broken_before) {
        test_linear_fail("the break of %s did not fail the replay", break_names[break_category]);
    }
    return true;
}

/* ---- the program ---- */

static bool replay_open(char **paths, uint32_t path_count, struct replay_sources *sources)
{
	struct storage_description description = { 0, 0, 0 };
	result_status status = test_capture_open(paths[0], &sources->capture);
	status = status == STATUS_OK ?
			 storage_direct.reader_open(
				 (const char *const *)&paths[1], path_count - 1,
				 &sources->reader) :
			 status;
	status = status == STATUS_OK ?
			 storage_direct.reader_describe(sources->reader,
							&description) :
			 status;
	status = status == STATUS_OK ?
			 pack_open_entry(&storage_direct, sources->reader,
					 &sources->pack) :
			 status;
	sources->block_alignment_bytes = description.block_alignment_bytes;
	return test_linear_expect_status(status, STATUS_OK,
					 "the capture and the pack open") &&
	       test_linear_expect(
		       description.block_alignment_bytes != 0 &&
			       BACKEND_STAGING_ALIGNMENT_BYTES %
					       description
						       .block_alignment_bytes ==
				       0,
		       "the reader's blocks divide a staging page");
}

/* One card: its kernels, the synthetic runs and, given sources, the replay. */
static void card_run(uint32_t card_index, const struct replay_sources *sources)
{
    static struct test_linear_card card;
    static struct layout layout;
    struct hyper_kernels_record kernels;
    struct router_kernels_record
	    router_kernels; /* the families draft.h fills beside the hyper connection's */
    struct attention_state_kernels state_kernels;
    struct attention_core_kernels core_kernels;
    struct draft_kernels_record draft_kernels;
    if (test_linear_card_open(&card, card_index, WORK_CAPACITY,
			      WEIGHTS_CAPACITY) &&
	test_linear_expect_status(
		hyper_kernels_prepare(&backend_cuda_value, card.card, &kernels),
		STATUS_OK,
		"the kernels of the hyper-connection sites are prepared") &&
	layout_prepare(&card, &kernels, &layout) &&
	synthetic_runs(&card, &layout) &&
	test_linear_expect_status(
		draft_kernels_prepare(&backend_cuda_value, card.card,
				      card.arithmetic_class, &kernels,
				      &router_kernels, &state_kernels,
				      &core_kernels, &draft_kernels),
		STATUS_OK, "the draft's kernels are prepared") &&
	draft_runs(&card, &kernels) && sources->pack != NULL) {
	    (void)replay_runs(&card, &layout, sources);
    }
    test_linear_card_close(&card);
}

int main(int argument_count, char **arguments)
{
    static struct replay_sources sources;
    int first_path = 1;
    uint32_t card_count = 0;
    if (argument_count > BREAK_ARGUMENTS && strcmp(arguments[1], "--break") == 0) {
        for (uint32_t part = 0; part < sizeof(break_names) / sizeof(break_names[0]); part++) {
            break_category = strcmp(arguments[BREAK_ARGUMENTS], break_names[part]) == 0 ? (int)part : break_category;
        }
        first_path += BREAK_ARGUMENTS;
    }
    const int path_count = argument_count - first_path;
    if ((first_path > 1 && break_category < 0) || path_count == 1 ||
	path_count > STORAGE_MAXIMUM_COPIES + 1) {
	    (void)fprintf(
		    stderr,
		    "usage: hyper_connection_card_test [--break scale|weight|output|embedding] "
		    "[<capture folder> <pack copy> [<pack copy> ...]]\n");
	    return TEST_WRONG_USAGE;
    }
    (void)alarm(TEST_TIME_LIMIT_SECONDS);
    synthetic_inputs();
    if ((path_count == 0 ||
	 replay_open(&arguments[first_path], (uint32_t)path_count, &sources)) &&
	test_linear_expect_status(backend_cuda_value.card_count(&card_count),
				  STATUS_OK, "the cards are counted") &&
	test_linear_expect(card_count > 0 &&
				   card_count <= BACKEND_MAXIMUM_CARDS,
			   "the machine has cards for the test")) {
	    for (uint32_t card_index = 0; card_index < card_count;
		 card_index++) {
		    card_run(card_index, &sources);
	    }
    }
    pack_close(sources.pack);
    storage_direct.reader_close(sources.reader);
    test_capture_close(sources.capture);
    if (path_count == 0) {
        (void)printf("hyper_connection_card_test: the replay compared nothing (it needs a capture folder and a "
                     "pack)\n");
    }
    for (uint32_t category = 0; category < CATEGORY_COUNT; category++) {
        if (category_failures[category] != 0) {
            (void)printf("%s: %" PRIu32 " failures\n", category_names[category], category_failures[category]);
        }
    }
    const bool only_the_break = break_category >= 0 && category_failures[break_category] != 0 &&
                                test_linear_failures() == shown_failures;
    for (uint32_t category = 0; only_the_break && category < CATEGORY_COUNT; category++) {
        if ((int)category != break_category && category_failures[category] != 0) {
            (void)printf("hyper_connection_card_test FAILED: the break of %s also broke %s\n",
                         break_names[break_category], category_names[category]);
            return TEST_FAILED;
        }
    }
    if (break_category >= 0) {
        (void)printf("hyper_connection_card_test %s: the break of %s\n", only_the_break ? "passed" : "FAILED",
                     break_names[break_category]);
        return only_the_break ? TEST_PASSED : TEST_FAILED;
    }
    if (test_linear_failures() != 0) {
        (void)fprintf(stderr, "hyper_connection_card_test FAILED: %d failure lines\n", test_linear_failures());
        return TEST_FAILED;
    }
    (void)printf("hyper_connection_card_test passed on %" PRIu32 " cards\n", card_count);
    return TEST_PASSED;
}
