/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * wide_small_card_test: the wide forms of the small launches of a chunk's layer that read no attention state, on the
 * real cards, against the thin launches of 8 rows that a chunk program runs for its slabs today.
 *
 *     wide_small_card_test [--time]
 *
 * It needs the cards free (`make card_tests`): a card without room for the arena FAILS the test, it is never skipped.
 * On every card in the class of its build (class 80 on a 170HX; class 120 native on the PRO, the default: these kernel
 * files have one sm_120a build, which class 120's emulation takes too), each launch below is a case of its own (the
 * card opened for it, so that its graphs fit the card's):
 *   - the dense products of dense_linear.h: a hyper mix (24 by 20,480, float32 weights, the row's scale), the router's
 *     scores (384 by 5,120), the index weights (32 by 5,120, scaled bfloat16 outputs), wkv and wgate at ratio 2 (512
 *     by 5,120), wkv at ratio 1 (layer 20's: bfloat16 outputs); their wide launch as dense_linear_prepare
 *     makes it with the class's wide forms (dense_linear_wide_kernels_prepare), then given each stamped wide
 *     form of the weights' type in turn (8, 4 and 1 outputs a block: its instance, asked of the card by name, and its
 *     grid);
 *   - the hyper sites' mix finish and post, the post without and with the combine, and the embedding of the step
 *     rows' tokens, of a vocabulary of EMBED_VOCABULARY (hyper_connection.h): a slab's launch reads its rows of the
 *     step area as its view holds them, behind a block the kernel does not read;
 *   - the n-gram rows' values and gate (engram_rows.h);
 *   - the queries alone (attention_state.h) of a layer with the indexer (theta 1) and of one of the window alone;
 *   - the quantization of hidden rows (linear.h), which the chunk programs already run over the chunk;
 * at 64, 65, 100, 256 and 1,024 rows. The inputs are laid out once for 1,024 rows; each row count has, for its thin
 * launches and for each wide form, its own outputs, which begin as rubbish of their own byte, and its own copy of
 * what a launch writes in place (the streams, the carried pre), laid out alike. The thin launches are those of the
 * row count's slabs of 8 rows, the last fewer (65 rows end with a slab of one row: the dense products' instance of
 * one row), as a chunk program prepares them; the wide launch takes all the rows. Every byte a wide launch writes
 * must be its thin launches': the outputs, and what it writes in place. The guards behind every range, the inputs and
 * the weights come back untouched.
 *   - Shown failing once: the top bit of every element of one input row, CONTROL_ROW (slab 64, row 5), is flipped on
 *     the card (of the step rows, the lowest bit of the row's token: another token of the vocabulary), what the 1,024
 *     rows' wide launch writes in place laid out again, and that launch run again: that row's bytes must then differ
 *     from the thin launches', and no other byte.
 *   - With --time each run of launches is timed as production runs them, in a graph of the card: a graph of REPEATS
 *     copies of [an L2 flush, the run] and one of REPEATS flushes alone run one after the other (the host's clock
 *     around each graph's run and wait), TIMED_RUNS such pairs after one that warms the card; the median of the pairs'
 *     differences, over REPEATS. The flush writes FLUSH_BYTES_* of the work region (fill_words_entry), so that a run
 *     reads its weights and rows from memory and finds the L2 full of lines to write back, as behind another launch;
 *     the thin launches are back to back, as in a chunk program's graph, where the rest of a slab's launches run
 *     between them in production. The wait sleeps between its looks at the card, so a graph's time comes in steps
 *     (about 75 us on the build workstation): REPEATS is WIDE_REPEATS for a wide launch, and for the thin launches THIN_TIMED_LAUNCHES
 *     launches' worth, at least THIN_REPEATS copies, so that a step is a small part of a run's time.
 * One line per launch and row count says what was compared, and with --time what was measured. It FAILS on any
 * difference.
 */
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "anemone/attention_state.h"
#include "anemone/backend_records.h"
#include "anemone/backend_seam.h"
#include "anemone/dense_linear.h"
#include "anemone/engram_rows.h"
#include "anemone/hyper_connection.h"
#include "anemone/linear.h"
#include "anemone/model_layout.h"
#include "anemone/status.h"
#include "anemone/step_records.h"
#include "linear_card_fixture.h"

#define TEST_TIME_LIMIT_SECONDS 3600            /* the alarm ends the test here, whatever the driver does */
#define WORK_CAPACITY_CLASS_80 (512ull << 20u)  /* the queries' case, 285 MiB, and the flush */
#define WORK_CAPACITY_CLASS_120 (456ull << 20u) /* the same: the PRO beside production has about 0.73 GiB free */
#define WEIGHTS_CAPACITY (8ull << 20u)          /* wkv's 5 MiB, the embedding's 5 MiB */
#define EMBED_VOCABULARY 512u                   /* tokens of the embedding: even, so a token's lowest bit flipped
                                                   is another of them */
#define FLUSH_BYTES_CLASS_80 (64ull << 20u)     /* twice a 170HX's L2 of 32 MiB */
#define FLUSH_BYTES_CLASS_120 (160ull << 20u)   /* 1.25 times the PRO's L2 of 128 MiB: what its room leaves */
#define FLUSH_WORD 0x5A5A5A5Au
#define COUNT_OF(table) ((uint32_t)(sizeof(table) / sizeof((table)[0])))
#define GROUP_ROWS \
	LINEAR_MAXIMUM_ROWS /* a slab's: the rows of a thin launch and of a dense group */
#define CHUNK_ROWS MODEL_WIDE_LEAST_ROWS
#define ONE_ROW_MORE (CHUNK_ROWS + 1u)          /* a last slab of one row */
#define UNEVEN_ROWS 100u                        /* twelve slabs of 8 and one of 4 */
#define QUARTER_ROWS 256u
#define MOST_ROWS LINEAR_WIDE_MOST_ROWS /* a production chunk */
#define MOST_SLABS (MOST_ROWS / GROUP_ROWS)
#define CONTROL_ROW 517u                        /* slab 64, row 5 */
#define PLAIN_ROWS 4u                           /* the first rows of values hold plain values, the others small ones */
#define MOST_VARIANTS 3u                        /* the wide forms of a weight type */
#define MOST_CASES 5u
#define MOST_CASE (MOST_CASES - 1u)             /* of the most rows */
#define MOST_COPIES (MOST_CASES * (1u + MOST_VARIANTS))
#define MOST_ARRAYS 11u                         /* a hyper site's */
#define MOST_LAUNCHES 256u                      /* of a case: 190 thin, 15 wide */
#define THIN_REPEATS 8u                         /* the fewest copies of a case's thin launches */
#define THIN_TIMED_LAUNCHES 512u                /* of thin launches in a timed graph, where 8 copies hold fewer */
#define WIDE_REPEATS 128u
#define MOST_TIMED_LAUNCHES (THIN_REPEATS * (1u + MOST_SLABS))
#define TIMED_RUNS 11u                          /* odd: the median is the middle one */
#define HALVES 2u
#define MEDIAN_RUN (TIMED_RUNS / HALVES)
#define PIECE_BYTES 65536u                      /* of the row buffer: the widest row (the heads), a piece of weights */
#define FUNCTION_NAME_CAPACITY 64u              /* bytes of a kernel's function name, its terminating zero among them */
#define RUBBISH_FIRST_BYTE 0x3Bu                /* copy c's outputs begin as this plus c */
#define BYTE_MASK 0xFFu
#define BITS_PER_BYTE 8u
#define TOP_BIT_OF_A_BYTE 0x80u
#define MOST_ARGUMENTS 2                        /* the program's name and --time */
#define NANOSECONDS_PER_SECOND 1000000000ull
#define NANOSECONDS_PER_MICROSECOND 1000.0
#define STREAM_OF_THE_TEST 0x57000000ull        /* the streams of the hashes, far from those of the other tests */
#define STREAM_OF_AN_ITEM 0x100000ull
#define STREAM_OF_AN_ARRAY 0x10000ull
#define FRACTION_SHIFT 8u                       /* a float of a hash: bits 8 to 14 its fraction, bit 15 its sign */
#define FRACTION_MASK 0x7Fu
#define FRACTION_SCALE 128.0f
#define SIGN_SHIFT 15u
#define EXPONENT_SHIFT 16u
#define ANGLE_SHIFT 40u                         /* an angle of a hash: its top 24 bits, in units of 2^-20 radians */
#define ANGLE_UNIT 0x1p-20
#define ENGRAM_ROWS_PER_TOKEN 24u               /* model_v41.h: (4 - 1) n-gram sizes of 8 heads */
#define ENGRAM_VALUE_BYTES (ENGRAM_ROWS_VALUES * sizeof(uint16_t))
#define BF16 DENSE_BFLOAT16
#define F32 DENSE_FLOAT32
#define FLOAT_BYTES ((uint32_t)sizeof(float))
#define VALUE_BYTES ((uint32_t)sizeof(uint16_t))
#define HIDDEN MODEL_HIDDEN

enum test_status { TEST_PASSED = 0, TEST_FAILED = 1, TEST_WRONG_USAGE = 2 };
static const uint32_t case_rows[MOST_CASES] = {CHUNK_ROWS, ONE_ROW_MORE, UNEVEN_ROWS, QUARTER_ROWS, MOST_ROWS};

/* An array of a launch: its weights; read by every copy; a copy's, read and written in place (each copy laid out
 * alike); a copy's, written. */
enum array_role { WEIGHTS, INPUT, STATE, OUTPUT };
/* What an array holds at first: bfloat16 values (PLAIN_ROWS plain rows, then small ones, test_linear_values), the
 * quantizer's edges, float32 numbers from 2^least_exponent up exponent_spread binades, cosines and sines, bytes, step
 * rows (step_records.h: a row's position and a token of the vocabulary; ahead of row 0 the step area's block); rubbish
 * (an output). */
enum array_fill { VALUES, EDGES, FLOATS, ANGLES, BYTES, TOKENS, RUBBISH };
struct array_kind {
    uint64_t row_bytes;                         /* of a row; of weights, all their bytes */
    enum array_role role;
    enum array_fill fill;
    uint32_t element_bytes;                     /* the control flips the top bit of each element */
    int32_t least_exponent;
    uint32_t exponent_spread;
    bool is_signed;
};

enum item_kind { DENSE, MIX_FINISH, POST, EMBED, ENGRAM_VALUES, ENGRAM_GATE, QUERIES, QUANTIZE };
/* option: POST the combine; QUERIES the indexer (theta 1), else the window alone (theta 0). */
struct item {
    const char *name;
    enum item_kind kind;
    uint32_t option;
    uint32_t output_size, input_size, weight_type, output_type, flags;     /* of DENSE */
    float output_scale, epsilon;
};
static const struct item items[] = {
	{ "the hyper mix", DENSE, 0, MODEL_MIXES, MODEL_STREAMS * HIDDEN, F32,
	  F32, DENSE_ROW_SCALE, 1.0f, MODEL_NORM_EPSILON },
	{ "the router's scores", DENSE, 0, MODEL_EXPERTS, HIDDEN, BF16, F32, 0,
	  1.0f, 0.0f },
	{ "the index weights", DENSE, 0, MODEL_INDEX_HEADS, HIDDEN, BF16, BF16,
	  0, DENSE_INDEX_WEIGHTS_SCALE, 0.0f },
	{ "wkv and wgate at ratio 2", DENSE, 0, MODEL_HEAD_DIMS, HIDDEN, BF16,
	  F32, 0, 1.0f, 0.0f },
	{ "wkv at ratio 1", DENSE, 0, MODEL_HEAD_DIMS, HIDDEN, BF16, BF16, 0,
	  1.0f, 0.0f },
	{ "the mix finish", MIX_FINISH, 0, 0, 0, 0, 0, 0, 0.0f, 0.0f },
	{ "the attention site's post", POST, 0, 0, 0, 0, 0, 0, 0.0f, 0.0f },
	{ "the FFN site's post with the combine", POST, 1, 0, 0, 0, 0, 0, 0.0f,
	  0.0f },
	{ "the embedding", EMBED, 0, 0, 0, 0, 0, 0, 0.0f, 0.0f },
	{ "the n-gram values", ENGRAM_VALUES, 0, 0, 0, 0, 0, 0, 0.0f, 0.0f },
	{ "the n-gram gate", ENGRAM_GATE, 0, 0, 0, 0, 0, 0, 0.0f, 0.0f },
	{ "the queries with the indexer", QUERIES, 1, 0, 0, 0, 0, 0, 0.0f,
	  0.0f },
	{ "the queries of the window alone", QUERIES, 0, 0, 0, 0, 0, 0, 0.0f,
	  0.0f },
	{ "the quantization of hidden rows", QUANTIZE, 0, 0, 0, 0, 0, 0, 0.0f,
	  0.0f },
};

/* The arrays of each kind of launch, in the order its description takes them. */
enum { DENSE_WEIGHTS, DENSE_ROWS, DENSE_OUTPUT, DENSE_ARRAYS };
enum { SITE_STREAMS, SITE_PRE, SITE_MIXES, SITE_SCALE, SITE_BASE, SITE_NORM, SITE_POST, SITE_COMB, SITE_HIDDEN,
       SITE_OUTPUT, SITE_SUMS, SITE_ARRAYS };
enum { EMBED_STEP, EMBED_TABLE, EMBED_STREAMS, EMBED_PRE, EMBED_ARRAYS };
enum { ENGRAM_TABLE_ROWS, ENGRAM_VALUES_OUT, ENGRAM_PROJECTION, ENGRAM_QUERY_WEIGHT, ENGRAM_KEY_WEIGHT, ENGRAM_STREAMS,
       ENGRAM_ARRAYS };
enum { QUERY_ROTATION, QUERY_HEADS, QUERY_QUERIES, QUERY_INDEX_HEADS, QUERY_INDEX_QUERIES, QUERY_ARRAYS };
enum { QUANTIZE_VALUES, QUANTIZE_CODES, QUANTIZE_EXPONENTS, QUANTIZE_OPERANDS, QUANTIZE_ARRAYS };
_Static_assert(SITE_ARRAYS == MOST_ARRAYS, "a site has the most arrays");

/* The ranges of numbers of the float32 arrays, by hand: weights of a mix, mixes, scales, bases, a carried pre, a post,
 * a comb, the routed sums (exponents from least, spread binades, of either sign where signed). */
#define MIX_WEIGHTS_LEAST (-8)
#define MIX_WEIGHTS_SPREAD 4u
#define MIXES_LEAST (-3)
#define MIXES_SPREAD 4u
#define SCALES_LEAST (-3)
#define SCALES_SPREAD 3u
#define PRE_LEAST (-2)
#define PRE_SPREAD 2u
#define POST_LEAST (-1)
#define POST_SPREAD 2u
#define COMB_LEAST (-3)
#define COMB_SPREAD 1u
#define SUMS_LEAST (-4)
#define SUMS_SPREAD 6u

/* The wide forms of kernel_dense_linear, in the order of DENSE_EVERY_WIDE_INSTANCE. */
#define DENSE_WIDE_STAMP(name, type, outputs, blocks) \
	{ type, outputs, #name "_wide_" #outputs },
static const struct wide_form {
    uint32_t weight_type, block_outputs;
    const char *name;
} wide_forms[] = { DENSE_EVERY_WIDE_INSTANCE };
#undef DENSE_WIDE_STAMP
_Static_assert(COUNT_OF(wide_forms) == DENSE_WIDE_INSTANCES &&
		       DENSE_WIDE_INSTANCES == MOST_VARIANTS * DENSE_TYPE_COUNT,
	       "each weight type has MOST_VARIANTS wide forms");

/* The kernels of a card that the launches take, every wide dense form by name, and the fill of the flush. */
struct kernels {
	struct dense_linear_kernels dense;
	uint32_t wide_forms[DENSE_WIDE_INSTANCES];
	struct hyper_kernels_record hyper;
	struct engram_rows_kernels engram;
	struct attention_state_kernels state;
	uint32_t fill;
};

/* What one launch lays out on a card: its arrays' ranges (weights and inputs at copy 0, a copy's at its index: case
 * c's thin launches' copy c (1 + variants), its variant v's that plus 1 + v), its launches (case c's thin ones from
 * thin_first[c], its variants' from wide_first[c]) and the flush. */
struct layout {
    const struct item *item;
    struct array_kind arrays[MOST_ARRAYS];
    uint32_t array_count, control_array, variant_count;
    uint32_t variants[MOST_VARIANTS];           /* of a dense product: the wide forms of its weight type */
    uint32_t prepared_variant
	    [MOST_CASES]; /* the variant dense_linear_prepare chose */
    struct linear_range ranges[MOST_ARRAYS][MOST_COPIES];
    struct linear_range flush;
    struct kernel_launch flush_launch;
    uint32_t thin_first[MOST_CASES], wide_first[MOST_CASES], launch_count;
};

static struct kernel_launch launches[MOST_LAUNCHES];
static struct kernel_launch timed[MOST_TIMED_LAUNCHES];
static unsigned char row_buffer[PIECE_BYTES];
static uint16_t piece_values[PIECE_BYTES / sizeof(uint16_t)];

static uint64_t clock_nanoseconds(void)
{
    struct timespec now = {0};
    return clock_gettime(CLOCK_MONOTONIC, &now) == 0
               ? (uint64_t)now.tv_sec * NANOSECONDS_PER_SECOND + (uint64_t)now.tv_nsec : 0;
}

static uint32_t slabs_of(uint32_t row_count)
{
    return (row_count + GROUP_ROWS - 1u) / GROUP_ROWS;
}

static uint32_t copy_of(const struct layout *layout, uint32_t case_index, uint32_t slot)    /* slot 0 thin, 1 + v */
{
    return case_index * (1u + layout->variant_count) + slot;
}

static uint32_t copy_count_of(const struct layout *layout)
{
    return MOST_CASES * (1u + layout->variant_count);
}

static bool is_written(const struct array_kind *kind)
{
    return kind->role == STATE || kind->role == OUTPUT;
}

static struct array_kind rows_of_values(uint64_t row_bytes, enum array_role role)
{
    return (struct array_kind){row_bytes, role, VALUES, VALUE_BYTES, 0, 0, false};
}
static struct array_kind rows_of_floats(uint64_t row_bytes, enum array_role role, int32_t least_exponent,
                                        uint32_t exponent_spread, bool is_signed)
{
    return (struct array_kind){row_bytes, role, FLOATS, FLOAT_BYTES, least_exponent, exponent_spread, is_signed};
}
static struct array_kind outputs_of(uint64_t row_bytes, uint32_t element_bytes)
{
    return (struct array_kind){row_bytes, OUTPUT, RUBBISH, element_bytes, 0, 0, false};
}

/* The arrays of a launch, in its description's order, and the input whose row the control flips. */
static uint32_t arrays_of(const struct item *item, struct array_kind *arrays, uint32_t *control_out)
{
    const bool is_finish = item->kind == MIX_FINISH, is_values = item->kind == ENGRAM_VALUES;
    switch (item->kind) {
    case DENSE: {
        const uint64_t weight_count = (uint64_t)item->output_size * item->input_size;
        const uint32_t output_bytes = item->output_type == F32 ? FLOAT_BYTES : VALUE_BYTES;
        arrays[DENSE_WEIGHTS] = item->weight_type == F32 ? rows_of_floats(weight_count * FLOAT_BYTES, WEIGHTS,
                                                                          MIX_WEIGHTS_LEAST, MIX_WEIGHTS_SPREAD, true)
                                                         : rows_of_values(weight_count * VALUE_BYTES, WEIGHTS);
        arrays[DENSE_ROWS] = rows_of_values((uint64_t)item->input_size * VALUE_BYTES, INPUT);
        arrays[DENSE_OUTPUT] = outputs_of((uint64_t)item->output_size * output_bytes, output_bytes);
        *control_out = DENSE_ROWS;
        return DENSE_ARRAYS;
    }
    case MIX_FINISH:
    case POST:
	    arrays[SITE_STREAMS] = rows_of_values(MODEL_STREAMS_ROW_BYTES,
						  is_finish ? INPUT : STATE);
	    arrays[SITE_PRE] = rows_of_floats(MODEL_SITE_ROW_BYTES,
					      is_finish ? STATE : INPUT,
					      PRE_LEAST, PRE_SPREAD, false);
	    arrays[SITE_MIXES] = rows_of_floats(MODEL_MIXES_ROW_BYTES, INPUT,
						MIXES_LEAST, MIXES_SPREAD,
						true);
	    arrays[SITE_SCALE] =
		    rows_of_floats((uint64_t)HYPER_SCALES * FLOAT_BYTES,
				   WEIGHTS, SCALES_LEAST, SCALES_SPREAD, false);
	    arrays[SITE_BASE] = rows_of_floats(MODEL_MIXES_ROW_BYTES, WEIGHTS,
					       SCALES_LEAST, SCALES_SPREAD,
					       true);
	    arrays[SITE_NORM] = rows_of_values(MODEL_HIDDEN_ROW_BYTES, WEIGHTS);
	    arrays[SITE_POST] =
		    is_finish ? outputs_of(MODEL_SITE_ROW_BYTES, FLOAT_BYTES) :
				rows_of_floats(MODEL_SITE_ROW_BYTES, INPUT,
					       POST_LEAST, POST_SPREAD, false);
	    arrays[SITE_COMB] =
		    is_finish ? outputs_of(MODEL_COMB_ROW_BYTES, FLOAT_BYTES) :
				rows_of_floats(MODEL_COMB_ROW_BYTES, INPUT,
					       COMB_LEAST, COMB_SPREAD, false);
	    arrays[SITE_HIDDEN] =
		    is_finish ?
			    outputs_of(MODEL_HIDDEN_ROW_BYTES, VALUE_BYTES) :
			    rows_of_values(MODEL_HIDDEN_ROW_BYTES, INPUT);
	    arrays[SITE_OUTPUT] = rows_of_values(MODEL_HIDDEN_ROW_BYTES, INPUT);
	    arrays[SITE_SUMS] = rows_of_floats((uint64_t)HIDDEN * FLOAT_BYTES,
					       INPUT, SUMS_LEAST, SUMS_SPREAD,
					       true);
	    *control_out = is_finish ? SITE_MIXES : SITE_OUTPUT;
	    return item->option != 0 ? SITE_ARRAYS : SITE_ARRAYS - 1u;
    case EMBED:
	    arrays[EMBED_STEP] = (struct array_kind){ STEP_ROW_BYTES_VALUE,
						      INPUT,
						      TOKENS,
						      STEP_ROW_BYTES_VALUE,
						      0,
						      0,
						      false };
	    arrays[EMBED_TABLE] = rows_of_values((uint64_t)EMBED_VOCABULARY *
							 MODEL_HIDDEN_ROW_BYTES,
						 WEIGHTS);
	    arrays[EMBED_STREAMS] =
		    outputs_of(MODEL_STREAMS_ROW_BYTES, VALUE_BYTES);
	    arrays[EMBED_PRE] = outputs_of(MODEL_SITE_ROW_BYTES, FLOAT_BYTES);
	    *control_out = EMBED_STEP;
	    return EMBED_ARRAYS;
    case ENGRAM_VALUES:
    case ENGRAM_GATE:
	    arrays[ENGRAM_TABLE_ROWS] =
		    (struct array_kind){ (uint64_t)ENGRAM_ROWS_PER_TOKEN *
						 ENGRAM_ROWS_ROW_BYTES,
					 INPUT,
					 BYTES,
					 1u,
					 0,
					 0,
					 false };
	    arrays[ENGRAM_VALUES_OUT] =
		    is_values ? outputs_of((uint64_t)ENGRAM_ROWS_PER_TOKEN *
						   ENGRAM_VALUE_BYTES,
					   VALUE_BYTES) :
				rows_of_values((uint64_t)ENGRAM_ROWS_PER_TOKEN *
						       ENGRAM_VALUE_BYTES,
					       INPUT);
	    arrays[ENGRAM_PROJECTION] = rows_of_values(
		    (uint64_t)ENGRAM_ROWS_PROJECTION * VALUE_BYTES, INPUT);
	    arrays[ENGRAM_QUERY_WEIGHT] =
		    rows_of_values(MODEL_STREAMS_ROW_BYTES, WEIGHTS);
	    arrays[ENGRAM_KEY_WEIGHT] =
		    rows_of_values(MODEL_STREAMS_ROW_BYTES, WEIGHTS);
	    arrays[ENGRAM_STREAMS] = rows_of_values(MODEL_STREAMS_ROW_BYTES,
						    is_values ? INPUT : STATE);
	    *control_out = is_values ? ENGRAM_TABLE_ROWS : ENGRAM_PROJECTION;
	    return ENGRAM_ARRAYS;
    case QUERIES:
	    arrays[QUERY_ROTATION] =
		    (struct array_kind){ MODEL_ROTATION_ROW_BYTES,
					 INPUT,
					 ANGLES,
					 FLOAT_BYTES,
					 0,
					 0,
					 false };
	    arrays[QUERY_HEADS] = rows_of_values(MODEL_HEADS_ROW_BYTES, INPUT);
	    arrays[QUERY_QUERIES] =
		    outputs_of(MODEL_HEADS_ROW_BYTES, VALUE_BYTES);
	    arrays[QUERY_INDEX_HEADS] =
		    rows_of_values(MODEL_INDEX_QUERIES_ROW_BYTES, INPUT);
	    arrays[QUERY_INDEX_QUERIES] =
		    outputs_of(MODEL_INDEX_QUERIES_ROW_BYTES, VALUE_BYTES);
	    *control_out = QUERY_HEADS;
	    return item->option != 0 ? QUERY_ARRAYS : QUERY_INDEX_HEADS;
    case QUANTIZE:
        arrays[QUANTIZE_VALUES] = (struct array_kind){(uint64_t)HIDDEN * VALUE_BYTES, INPUT, EDGES, VALUE_BYTES, 0, 0,
                                                      false};
	arrays[QUANTIZE_CODES] =
		outputs_of(linear_codes_byte_count(1, HIDDEN), 1u);
	arrays[QUANTIZE_EXPONENTS] =
		outputs_of(linear_exponents_byte_count(1, HIDDEN), 1u);
	arrays[QUANTIZE_OPERANDS] =
		outputs_of(linear_values_byte_count(1, HIDDEN), VALUE_BYTES);
	*control_out = QUANTIZE_VALUES;
        return QUANTIZE_ARRAYS;
    }
    return 0;
}

/* A float32 of a hash for an array of FLOATS: 2^(least + a binade of its spread) times 1 + 7 bits, signed where it
 * is. */
static float float_of_hash(uint64_t hash, const struct array_kind *kind)
{
    const float fraction = (float)((uint32_t)(hash >> FRACTION_SHIFT) & FRACTION_MASK) / FRACTION_SCALE;
    const int exponent = kind->least_exponent + (int)((uint32_t)(hash >> EXPONENT_SHIFT) % kind->exponent_spread);
    const float magnitude = ldexpf(1.0f + fraction, exponent);
    return kind->is_signed && (hash >> SIGN_SHIFT & 1u) != 0 ? -magnitude : magnitude;
}

/* The first byte_count bytes of piece `piece` of an array (a row of it, or a piece of weights) into row_buffer, a
 * function of the stream and the piece alone. */
static void piece_fill(const struct array_kind *kind, uint64_t stream, uint64_t piece, uint64_t byte_count)
{
    const uint64_t piece_stream = stream + piece;
    const uint32_t count = (uint32_t)(byte_count / kind->element_bytes);
    for (uint32_t index = 0; index < count && kind->fill != VALUES && kind->fill != EDGES; index++) {
        const uint64_t hash = test_linear_hash(piece_stream, index);
        float number = 0.0f;
        switch (kind->fill) {
        case FLOATS:
            number = float_of_hash(hash, kind);
            memcpy(row_buffer + (size_t)index * FLOAT_BYTES, &number, sizeof(number));
            break;
        case ANGLES: {                          /* a pair of floats: the cosine and the sine of one angle */
            const double angle = (double)(test_linear_hash(piece_stream, index / HALVES) >> ANGLE_SHIFT) * ANGLE_UNIT;
            number = (float)(index % HALVES == 0 ? cos(angle) : sin(angle));
            memcpy(row_buffer + (size_t)index * FLOAT_BYTES, &number, sizeof(number));
            break;
        }
        case BYTES:
            row_buffer[index] = (unsigned char)(hash & BYTE_MASK);
            break;
        case TOKENS: {                          /* step row `piece`: its position and a token */
		const struct step_row_record row = {
			(int32_t)piece, (int32_t)(hash % EMBED_VOCABULARY), 0, 0
		};
		memcpy(row_buffer, &row, sizeof(row));
		break;
        }
        case VALUES:
        case EDGES:
        case RUBBISH:
            break;
        }
    }
    if (kind->fill == VALUES || kind->fill == EDGES) {
        test_linear_values(piece_values, count, piece_stream,
                           kind->fill == EDGES     ? TEST_LINEAR_VALUES_EDGES
                           : kind->role == WEIGHTS ? TEST_LINEAR_VALUES_PLAIN
                           : piece < PLAIN_ROWS    ? TEST_LINEAR_VALUES_PLAIN
                                                   : TEST_LINEAR_VALUES_SMALL);
        memcpy(row_buffer, piece_values, (size_t)count * sizeof(uint16_t));
    }
}

/* Fills a range of the image with one byte, a piece at a time. */
static bool range_fill_byte(struct test_linear_card *card,
			    struct linear_range range, unsigned char byte)
{
    bool is_filled = true;
    memset(row_buffer, byte, sizeof(row_buffer));
    for (uint64_t done = 0; done < range.byte_count && is_filled; done += sizeof(row_buffer)) {
        const uint64_t count = range.byte_count - done < sizeof(row_buffer) ? range.byte_count - done
                                                                            : sizeof(row_buffer);
        is_filled = test_linear_card_put(card, range, done, row_buffer, count);
    }
    return is_filled;
}

/* The ranges of a launch's arrays for rows first_row .. first_row + row_count - 1 of a copy: a copy's own of what is
 * written, the shared rows of an input, the weights as they are; {0, 0} behind the launch's arrays. */
static void rows_ranges(const struct layout *layout, uint32_t copy,
			uint32_t first_row, uint32_t row_count,
			struct linear_range ranges_out[MOST_ARRAYS])
{
    for (uint32_t array = 0; array < MOST_ARRAYS; array++) {
        const struct array_kind *const kind = &layout->arrays[array];
	const struct linear_range base =
		layout->ranges[array]
			      [kind->role == WEIGHTS || kind->role == INPUT ?
				       0 :
				       copy];
	ranges_out[array] =
		array >= layout->array_count ?
			(struct linear_range){ 0, 0 } :
		kind->role == WEIGHTS ?
			base :
			(struct linear_range){
				base.byte_offset + first_row * kind->row_bytes,
				row_count * kind->row_bytes
			};
    }
}

/* The launch of an item for rows first_row .. of a copy, as its family's prepare function makes it. */
MUST_CHECK static result_status
launch_prepare(const struct test_linear_card *card,
	       const struct kernels *kernels, const struct layout *layout,
	       uint32_t copy, uint32_t first_row, uint32_t row_count,
	       struct kernel_launch *launch_out)
{
    const struct item *const item = layout->item;
    const uint64_t arena = card->arena_byte_count;
    struct linear_range
	    r[MOST_ARRAYS]; /* short: the descriptions below take them in their order */
    struct kernel_launch pair[HYPER_SITE_LAUNCH_COUNT];
    result_status status = STATUS_INVALID_ARGUMENT;
    _Static_assert((uint32_t)HYPER_SITE_LAUNCH_COUNT ==
			   (uint32_t)ENGRAM_ROWS_LAUNCH_COUNT,
		   "a pair of launches");
    rows_ranges(layout, copy, first_row, row_count, r);
    switch (item->kind) {
    case DENSE: {
	    const struct dense_linear_description dense = { r[DENSE_WEIGHTS],
							    r[DENSE_ROWS],
							    r[DENSE_OUTPUT],
							    { 0, 0 },
							    item->output_size,
							    item->input_size,
							    row_count,
							    item->weight_type,
							    item->output_type,
							    item->flags,
							    item->output_scale,
							    item->epsilon,
							    0 };
	    return dense_linear_prepare(&dense, arena, &kernels->dense,
					launch_out);
    }
    case MIX_FINISH:
    case POST: {
	    const struct hyper_site_description site = { r[SITE_STREAMS],
							 r[SITE_PRE],
							 r[SITE_MIXES],
							 r[SITE_SCALE],
							 r[SITE_BASE],
							 r[SITE_NORM],
							 r[SITE_POST],
							 r[SITE_COMB],
							 r[SITE_HIDDEN],
							 r[SITE_OUTPUT],
							 r[SITE_SUMS],
							 row_count,
							 0 };
	    status = hyper_site_prepare(&site, arena, &kernels->hyper, pair);
	    *launch_out =
		    pair[item->kind == MIX_FINISH ? HYPER_SITE_MIX_FINISH :
						    HYPER_SITE_POST];
	    return status;
    }
    case EMBED: {                               /* the step area: a block ahead of the rows */
	    const struct hyper_embed_description embed = {
		    { r[EMBED_STEP].byte_offset - STEP_BLOCK_BYTES_VALUE,
		      STEP_BLOCK_BYTES_VALUE + r[EMBED_STEP].byte_count },
		    r[EMBED_TABLE],
		    r[EMBED_STREAMS],
		    r[EMBED_PRE],
		    row_count,
		    EMBED_VOCABULARY
	    };
	    return hyper_embed_prepare(&embed, arena, &kernels->hyper,
				       launch_out);
    }
    case ENGRAM_VALUES:
    case ENGRAM_GATE: {
	    const struct engram_rows_description engram = {
		    r[ENGRAM_TABLE_ROWS],  r[ENGRAM_VALUES_OUT],
		    r[ENGRAM_PROJECTION],  r[ENGRAM_QUERY_WEIGHT],
		    r[ENGRAM_KEY_WEIGHT],  r[ENGRAM_STREAMS],
		    ENGRAM_ROWS_PER_TOKEN, row_count
	    };
	    status =
		    engram_rows_prepare(&engram, arena, &kernels->engram, pair);
	    *launch_out = pair[item->kind == ENGRAM_VALUES ?
				       ENGRAM_ROWS_VALUES_LAUNCH :
				       ENGRAM_ROWS_GATE_LAUNCH];
	    return status;
    }
    case QUERIES: {
	    const struct attention_queries_description queries = {
		    r[QUERY_ROTATION],
		    r[QUERY_HEADS],
		    r[QUERY_QUERIES],
		    r[QUERY_INDEX_HEADS],
		    r[QUERY_INDEX_QUERIES],
		    row_count,
		    item->option,
		    0,
		    MODEL_HEADS_VALUE,
		    item->option,
		    0
	    };
	    return attention_queries_prepare(&queries, arena, &kernels->state,
					     launch_out);
    }
    case QUANTIZE: {
	    const struct linear_quantize_description quantize = {
		    r[QUANTIZE_VALUES],
		    r[QUANTIZE_CODES],
		    r[QUANTIZE_EXPONENTS],
		    r[QUANTIZE_OPERANDS],
		    HIDDEN,
		    row_count
	    };
	    return linear_quantize_prepare(&quantize, arena, &card->kernels,
					   launch_out);
    }
    }
    return status;
}

/* Every launch of an item: of each case the thin launches of its slabs, then its wide launch in each variant (a
 * dense product's prepared, then given the variant's form: its instance and grid). */
static bool launches_prepare(const struct test_linear_card *card, const struct kernels *kernels,
                             struct layout *layout)
{
    const struct item *const item = layout->item;
    bool is_prepared = true;
    uint32_t count = 0;
    for (uint32_t case_index = 0; case_index < MOST_CASES && is_prepared; case_index++) {
        const uint32_t row_count = case_rows[case_index];
        layout->thin_first[case_index] = count;
        for (uint32_t slab = 0; slab < slabs_of(row_count) && is_prepared; slab++) {
            const uint32_t first_row = slab * GROUP_ROWS, left = row_count - first_row;
	    is_prepared = test_linear_expect_status(
		    launch_prepare(card, kernels, layout,
				   copy_of(layout, case_index, 0), first_row,
				   left < GROUP_ROWS ? left : GROUP_ROWS,
				   &launches[count++]),
		    STATUS_OK, "a thin launch is prepared");
	}
        layout->wide_first[case_index] = count;
        for (uint32_t variant = 0; variant < layout->variant_count && is_prepared; variant++) {
		struct kernel_launch *const launch = &launches[count++];
		is_prepared =
			test_linear_expect_status(
				launch_prepare(card, kernels, layout,
					       copy_of(layout, case_index,
						       1u + variant),
					       0, row_count, launch),
				STATUS_OK, "a wide launch is prepared") &&
			test_linear_expect(
				launch->grid[1] ==
					(item->kind == DENSE ?
						 slabs_of(row_count) :
						 row_count),
				"a wide launch takes its rows: a dense product's groups of 8, else rows");
		uint32_t prepared =
			layout->variant_count; /* of the first variant: which form the prepare took */
		for (uint32_t other = 0; other < layout->variant_count &&
					 variant == 0 && item->kind == DENSE;
		     other++) {
			prepared =
				launch->kernel ==
						kernels->wide_forms
							[layout->variants[other]] ?
					other :
					prepared;
		}
            is_prepared = is_prepared && (variant != 0 || item->kind != DENSE ||
                                          test_linear_expect(prepared < layout->variant_count,
                                                             "a wide dense launch takes a wide form of its type"));
            layout->prepared_variant[case_index] = variant == 0 ? prepared : layout->prepared_variant[case_index];
            if (is_prepared && item->kind == DENSE) {
                launch->kernel = kernels->wide_forms[layout->variants[variant]];
                launch->grid[0] = item->output_size / wide_forms[layout->variants[variant]].block_outputs;
            }
        }
    }
    layout->launch_count = count;
    return is_prepared;
}

/* Lays an item out: its flush, weights, inputs and copies (a state array's rows the same in every copy, an output's
 * rubbish of its copy), then its launches. */
static bool layout_make(struct test_linear_card *card, const struct kernels *kernels, uint32_t item_index,
                        struct layout *layout)
{
    const struct item *const item = &items[item_index];
    memset(layout, 0, sizeof(*layout));
    layout->item = item;
    layout->array_count = arrays_of(item, layout->arrays, &layout->control_array);
    layout->variant_count = item->kind == DENSE ? MOST_VARIANTS : 1u;
    for (uint32_t form = 0, variant = 0;
	 form < DENSE_WIDE_INSTANCES && item->kind == DENSE; form++) {
	    if (wide_forms[form].weight_type == item->weight_type &&
		variant < MOST_VARIANTS) {
		    layout->variants[variant++] = form;
	    }
    }
    test_linear_card_clear(card);
    layout->flush = test_linear_card_work(
	    card, card->arithmetic_class == ARITHMETIC_CLASS_SM_80 ?
			  FLUSH_BYTES_CLASS_80 :
			  FLUSH_BYTES_CLASS_120);
    const struct fill_record flush = { layout->flush.byte_offset,
				       layout->flush.byte_count /
					       sizeof(uint32_t),
				       FLUSH_WORD, 0 };
    layout->flush_launch = (struct kernel_launch){
	    .kernel = kernels->fill,
	    .grid = { FILL_MOST_BLOCKS, 1, 1 },
	    .block = { FILL_THREADS_PER_BLOCK_VALUE, 1, 1 },
	    .argument_byte_count = FILL_RECORD_BYTES
    };
    memcpy(layout->flush_launch.argument, &flush, sizeof(flush));
    for (uint32_t array = 0; array < layout->array_count; array++) {
        const struct array_kind *const kind = &layout->arrays[array];
	const uint64_t ahead = kind->fill == TOKENS ?
				       STEP_BLOCK_BYTES_VALUE :
				       0; /* the step area's block */
	for (uint32_t copy = 0; copy < copy_count_of(layout); copy++) {
            const uint32_t case_index = copy / (1u + layout->variant_count);
            const bool is_shared = !is_written(kind);
            const uint64_t row_count = is_shared ? MOST_ROWS : case_rows[case_index];
            if (copy == 0 || !is_shared) {
		    const struct linear_range range =
			    kind->role == WEIGHTS ?
				    test_linear_card_weights(card,
							     kind->row_bytes) :
				    test_linear_card_work(
					    card,
					    ahead + row_count *
							    kind->row_bytes);
		    layout->ranges[array][copy] = (struct linear_range){
			    range.byte_offset + ahead,
			    range.byte_count == 0 ? 0 : range.byte_count - ahead
		    };
	    }
        }
    }
    bool is_laid_out = test_linear_card_has_room(card);
    for (uint32_t array = 0; array < layout->array_count && is_laid_out; array++) {
        const struct array_kind *const kind = &layout->arrays[array];
        const uint64_t stream = STREAM_OF_THE_TEST + STREAM_OF_AN_ITEM * item_index + STREAM_OF_AN_ARRAY * array;
        const uint64_t piece_bytes = kind->role == WEIGHTS ? PIECE_BYTES : kind->row_bytes;
        const uint64_t pieces = kind->role == WEIGHTS ? (kind->row_bytes + PIECE_BYTES - 1u) / PIECE_BYTES : MOST_ROWS;
        for (uint32_t copy = 0; kind->role == OUTPUT && copy < copy_count_of(layout); copy++) {
            const unsigned char rubbish = (unsigned char)(RUBBISH_FIRST_BYTE + copy);
            is_laid_out = is_laid_out && range_fill_byte(card, layout->ranges[array][copy], rubbish);
        }
        for (uint64_t piece = 0; kind->role != OUTPUT && piece < pieces && is_laid_out; piece++) {
            const uint64_t byte_offset = piece * piece_bytes;
            const uint64_t byte_count = kind->role == WEIGHTS && kind->row_bytes - byte_offset < piece_bytes
                                            ? kind->row_bytes - byte_offset : piece_bytes;
            piece_fill(kind, stream, piece, byte_count);
            for (uint32_t copy = 0; copy < copy_count_of(layout) && is_laid_out; copy++) {
                const uint32_t case_index = copy / (1u + layout->variant_count);
                if (kind->role == STATE ? piece < case_rows[case_index] : copy == 0) {
                    is_laid_out = test_linear_card_put(card, layout->ranges[array][copy], byte_offset, row_buffer,
                                                       byte_count);
                }
            }
        }
    }
    return is_laid_out && launches_prepare(card, kernels, layout);
}

/* Each case: every variant's written arrays against the thin launches', byte for byte; one line each. */
static void cases_check(const struct test_linear_card *card, const struct layout *layout)
{
    const struct item *const item = layout->item;
    for (uint32_t case_index = 0; case_index < MOST_CASES; case_index++) {
        const uint32_t row_count = case_rows[case_index];
        uint64_t compared_bytes = 0;
        bool is_same = true;
        for (uint32_t array = 0; array < layout->array_count; array++) {
            const struct array_kind *const kind = &layout->arrays[array];
            for (uint32_t variant = 0; variant < layout->variant_count && is_written(kind); variant++) {
                const uint64_t byte_count = row_count * kind->row_bytes;
                is_same = test_linear_card_expect_same(card,
                                                       layout->ranges[array][copy_of(layout, case_index, 1u + variant)],
                                                       0, layout->ranges[array][copy_of(layout, case_index, 0)], 0,
                                                       byte_count, "the wide launch writes the thin launches' bytes") &&
                          is_same;
                compared_bytes += byte_count;
            }
        }
        (void)printf("card %" PRIu32 " (class %" PRIu32 "): %s, %" PRIu32 " rows (%" PRIu32 " thin launches): %s, %"
                     PRIu64 " bytes of %" PRIu32 " wide forms against the thin launches", card->card_index,
                     card->arithmetic_class, item->name, row_count, slabs_of(row_count),
                     is_same ? "every byte the same" : "DIFFERENT", compared_bytes, layout->variant_count);
        if (item->kind == DENSE) {
            (void)printf(" (prepared: %s)", wide_forms[layout->variants[layout->prepared_variant[case_index]]].name);
        }
        (void)printf("\n");
    }
}

/* Shown failing once: the top bit of every element of row CONTROL_ROW of the control input flipped, what the wide
 * launch of the most rows (the prepared variant) writes in place laid out again, that launch run again: that row's
 * bytes differ from the thin launches', and no other byte. */
static void control_check(struct test_linear_card *card, const struct layout *layout)
{
    const struct array_kind *const control = &layout->arrays[layout->control_array];
    const struct linear_range control_range =
	    layout->ranges[layout->control_array][0];
    const uint32_t variant = layout->item->kind == DENSE ? layout->prepared_variant[MOST_CASE] : 0;
    const uint32_t wide_copy = copy_of(layout, MOST_CASE, 1u + variant), thin_copy = copy_of(layout, MOST_CASE, 0);
    const uint8_t *const image = test_linear_card_stored(card, control_range);
    uint64_t differing_total = 0, differing_in_row = 0;
    bool is_run = image != NULL && control->row_bytes <= sizeof(row_buffer);
    if (is_run) {
        memcpy(row_buffer, image + CONTROL_ROW * control->row_bytes, (size_t)control->row_bytes);
        for (uint64_t byte = control->element_bytes - 1u; byte < control->row_bytes && control->fill != TOKENS;
             byte += control->element_bytes) {
            row_buffer[byte] ^= TOP_BIT_OF_A_BYTE;          /* little-endian: an element's top bit is in its last */
        }
        if (control->fill == TOKENS) {                      /* a step row: another token of the vocabulary */
		row_buffer[offsetof(struct step_row_record, token_id)] ^= 1u;
	}
        is_run = test_linear_card_put(card, control_range, CONTROL_ROW * control->row_bytes, row_buffer,
                                      control->row_bytes) &&
                 test_linear_card_upload(card, control_range, CONTROL_ROW * control->row_bytes, control->row_bytes);
    }
    for (uint32_t array = 0; array < layout->array_count && is_run; array++) {
        if (layout->arrays[array].role == STATE) {      /* back to the rows the first run began with */
            is_run = test_linear_card_upload(card, layout->ranges[array][wide_copy], 0,
                                             layout->ranges[array][wide_copy].byte_count);
        }
    }
    if (!test_linear_expect(is_run, "the control row is changed on the card") ||
        !test_linear_card_run(card, &launches[layout->wide_first[MOST_CASE] + variant], 1) ||
        !test_linear_card_fetch(card)) {
        return;
    }
    for (uint32_t array = 0; array < layout->array_count; array++) {
        const struct array_kind *const kind = &layout->arrays[array];
        uint64_t first_differing = 0;
        if (is_written(kind)) {
            const uint64_t differing = test_linear_card_differing(card, layout->ranges[array][wide_copy], 0,
                                                                  layout->ranges[array][thin_copy], 0,
                                                                  MOST_ROWS * kind->row_bytes, &first_differing);
            const uint64_t there = test_linear_card_differing(
                card, layout->ranges[array][wide_copy], CONTROL_ROW * kind->row_bytes, layout->ranges[array][thin_copy],
                CONTROL_ROW * kind->row_bytes, kind->row_bytes, &first_differing);
            (void)test_linear_expect(differing == there, "only the control row differs");
            differing_total += differing;
            differing_in_row += there;
        }
    }
    (void)test_linear_expect(differing_in_row != 0, "the control row's bytes differ");
    (void)printf("  control: row %u with %s: %" PRIu64 " bytes differ, %" PRIu64 " of them that row's\n", CONTROL_ROW,
                 control->fill == TOKENS ? "another token" : "the top bit of each element flipped", differing_total,
                 differing_in_row);
}

/* Captures a graph of the launches; false behind a failure line. */
static bool timed_graph_capture(struct test_linear_card *card,
				const struct kernel_launch *graph_launches,
				uint32_t count, uint32_t *graph_out)
{
	return test_linear_expect_status(
		backend_cuda_value.graph_capture(card->card, graph_launches,
						 count, graph_out),
		STATUS_OK, "a graph is captured");
}

/* Microseconds of a run of launches: a graph of repeats copies of [the flush, the run] and one of repeats flushes run
 * one after the other, TIMED_RUNS pairs behind one that warms the card (the host's clock around each graph's run and
 * wait); the median of the pairs' differences, over repeats. A pair's difference leaves out what drifts between pairs,
 * such as production's work on the card. 0 behind a failure line, or where the flushes alone took longer. */
static double run_microseconds(struct test_linear_card *card,
			       const struct layout *layout,
			       const struct kernel_launch *run,
			       uint32_t run_count, uint32_t repeats)
{
    int64_t differences[TIMED_RUNS] = {0};
    uint32_t graphs[HALVES] = {0, 0};           /* with the runs, the flushes alone */
    uint32_t count = 0;
    bool is_timed = test_linear_expect((uint64_t)repeats * ((uint64_t)run_count + 1u) <= (uint64_t)MOST_TIMED_LAUNCHES,
                                       "a timed graph fits the table of its launches");
    for (uint32_t repeat = 0; repeat < repeats && is_timed; repeat++) {
        timed[count++] = layout->flush_launch;
        memcpy(&timed[count], run, run_count * sizeof(*run));
        count += run_count;
    }
    is_timed = is_timed && timed_graph_capture(card, timed, count, &graphs[0]);
    for (count = 0; count < repeats; count++) {
        timed[count] = layout->flush_launch;
    }
    is_timed = is_timed && timed_graph_capture(card, timed, count, &graphs[1]);
    for (uint32_t pair = 0; pair <= TIMED_RUNS && is_timed; pair++) {   /* pair 0 warms the card */
        uint64_t durations[HALVES] = {0, 0};
        for (uint32_t half = 0; half < HALVES && is_timed; half++) {
            const uint64_t started = clock_nanoseconds();
	    is_timed =
		    test_linear_expect_status(backend_cuda_value.graph_run(
						      card->card, graphs[half]),
					      STATUS_OK, "a graph runs") &&
		    test_linear_expect_status(
			    backend_cuda_value.launch_wait(
				    card->card,
				    TEST_LINEAR_WAIT_LIMIT_NANOSECONDS),
			    STATUS_OK, "a graph finishes within the limit");
	    durations[half] = clock_nanoseconds() - started;
        }
        differences[pair == 0 ? 0 : pair - 1u] = (int64_t)durations[0] - (int64_t)durations[1];
    }
    for (uint32_t index = 1; index < TIMED_RUNS; index++) {   /* sorted: the median is the middle one */
        const int64_t difference = differences[index];
        uint32_t place = index;
        while (place > 0 && differences[place - 1u] > difference) {
            differences[place] = differences[place - 1u];
            place--;
        }
        differences[place] = difference;
    }
    const int64_t median = differences[MEDIAN_RUN];
    return is_timed && median > 0 ? (double)median / repeats / NANOSECONDS_PER_MICROSECOND : 0.0;
}

/* The thin launches of each case and its wide launch in each variant, timed; one line each. */
static void cases_time(struct test_linear_card *card, const struct layout *layout)
{
    for (uint32_t case_index = 0; case_index < MOST_CASES; case_index++) {
        const uint32_t row_count = case_rows[case_index], slabs = slabs_of(row_count);
        (void)printf("card %" PRIu32 " (class %" PRIu32 "): %s, %" PRIu32 " rows: TIME in a graph after a flush of "
                     "the L2: thin (%" PRIu32 " launches) %.1f us", card->card_index, card->arithmetic_class,
                     layout->item->name, row_count, slabs,
                     run_microseconds(card, layout, &launches[layout->thin_first[case_index]], slabs,
                                      THIN_TIMED_LAUNCHES / slabs > THIN_REPEATS ? THIN_TIMED_LAUNCHES / slabs
                                                                                 : THIN_REPEATS));
        for (uint32_t variant = 0; variant < layout->variant_count; variant++) {
            (void)printf(", wide%s%s %.1f us", layout->item->kind == DENSE ? " " : "",
                         layout->item->kind == DENSE ? wide_forms[layout->variants[variant]].name : "",
                         run_microseconds(card, layout, &launches[layout->wide_first[case_index] + variant], 1,
                                          WIDE_REPEATS));
        }
        (void)printf("\n");
    }
}

/* The kernels the launches take: class 120 native takes the linear's native builds (the fixture prepared the card's
 * own class); the dense products' wide forms of the class, then every one by name (the same function asked again
 * gives the same handle); the fill of the flush. */
static bool kernels_prepare(struct test_linear_card *card, uint32_t arithmetic_class, struct kernels *kernels)
{
	const struct backend_operations *const backend = &backend_cuda_value;
	bool is_prepared = true;
	card->arithmetic_class = arithmetic_class;
	for (uint32_t form = 0; form < DENSE_WIDE_INSTANCES && is_prepared;
	     form++) {
		char name[FUNCTION_NAME_CAPACITY];
		(void)snprintf(name, sizeof(name), "%s%s",
			       KERNEL_DENSE_LINEAR_STEM, wide_forms[form].name);
		is_prepared = test_linear_expect_status(
			backend->kernel_prepare(card->card,
						KERNEL_FILE_DENSE_LINEAR, name,
						DENSE_ARGUMENT_BYTES,
						&kernels->wide_forms[form]),
			STATUS_OK,
			"a wide form of the dense products is prepared");
	}
	return is_prepared &&
	       (arithmetic_class != ARITHMETIC_CLASS_SM_120_NATIVE ||
		test_linear_expect_status(
			linear_kernels_prepare(backend, card->card,
					       arithmetic_class,
					       &card->kernels),
			STATUS_OK,
			"the kernels of class 120 native are prepared")) &&
	       test_linear_expect_status(
		       dense_linear_kernels_prepare(backend, card->card,
						    &kernels->dense),
		       STATUS_OK, "the dense products' kernels are prepared") &&
	       test_linear_expect_status(
		       dense_linear_wide_kernels_prepare(backend, card->card,
							 arithmetic_class,
							 &kernels->dense),
		       STATUS_OK, "the wide forms of the class are prepared") &&
	       test_linear_expect_status(
		       hyper_kernels_prepare(backend, card->card,
					     &kernels->hyper),
		       STATUS_OK, "the hyper sites' kernels are prepared") &&
	       test_linear_expect_status(
		       engram_rows_kernels_prepare(backend, card->card,
						   &kernels->engram),
		       STATUS_OK, "the n-gram rows' kernels are prepared") &&
	       test_linear_expect_status(
		       attention_state_kernels_prepare(backend, card->card,
						       &kernels->state),
		       STATUS_OK,
		       "the attention state's kernels are prepared") &&
	       test_linear_expect_status(
		       backend->kernel_prepare(
			       card->card, KERNEL_FILE_FILL, KERNEL_FILL_WORDS,
			       FILL_RECORD_BYTES, &kernels->fill),
		       STATUS_OK, "the fill is prepared");
}

/* One item on one card: opened, laid out, loaded, run, fetched and checked, shown failing, timed. */
static void item_run(uint32_t card_index, uint32_t arithmetic_class, uint32_t item_index, bool is_timed,
                     uint64_t *digest)
{
    static struct test_linear_card card;
    static struct layout layout;
    static struct kernels kernels;
    const uint64_t work_capacity = arithmetic_class == ARITHMETIC_CLASS_SM_80 ?
					   WORK_CAPACITY_CLASS_80 :
					   WORK_CAPACITY_CLASS_120;
    if (test_linear_card_open(&card, card_index, work_capacity, WEIGHTS_CAPACITY)) {
        test_linear_context("card %" PRIu32 ", class %" PRIu32 ", %s", card_index, arithmetic_class,
                            items[item_index].name);
        if (kernels_prepare(&card, arithmetic_class, &kernels) &&
            test_linear_expect(layout_make(&card, &kernels, item_index, &layout), "the launch is laid out") &&
            test_linear_card_load(&card) && test_linear_card_run(&card, launches, layout.launch_count) &&
            test_linear_card_fetch(&card)) {
            for (uint32_t array = 0; array < layout.array_count; array++) {
                if (layout.arrays[array].role == INPUT) {
                    (void)test_linear_card_expect_untouched(&card, layout.ranges[array][0], "an input is only read");
                }
            }
            (void)test_linear_card_expect_weights_untouched(&card);
            cases_check(&card, &layout);
            control_check(&card, &layout);
            if (is_timed) {
                cases_time(&card, &layout);
            }
            *digest += card.digest;
        }
    }
    test_linear_card_close(&card);
}

int main(int argument_count, char **arguments)
{
    uint32_t card_count = 0;
    const bool is_timed = argument_count == MOST_ARGUMENTS && strcmp(arguments[1], "--time") == 0;
    if (argument_count > MOST_ARGUMENTS || (argument_count == MOST_ARGUMENTS && !is_timed)) {
        (void)fprintf(stderr, "usage: wide_small_card_test [--time]\n");
        return TEST_WRONG_USAGE;
    }
    (void)alarm(TEST_TIME_LIMIT_SECONDS);
    if (test_linear_expect_status(backend_cuda_value.card_count(&card_count),
				  STATUS_OK, "the cards are counted") &&
	test_linear_expect(card_count > 0 &&
				   card_count <= BACKEND_MAXIMUM_CARDS,
			   "the machine has cards for the test")) {
	    for (uint32_t card_index = 0; card_index < card_count;
		 card_index++) {
		    struct card_description_record description;
		    uint64_t digest = 0;
		    memset(&description, 0, sizeof(description));
		    test_linear_context("card %" PRIu32, card_index);
		    if (!test_linear_expect_status(
				backend_cuda_value.card_describe(card_index,
								 &description),
				STATUS_OK, "the card is described")) {
			    continue;
		    }
		    /* the PRO in its own arithmetic, the default: these kernel files have no other build for it */
		    const uint32_t arithmetic_class =
			    description.arithmetic_class ==
					    ARITHMETIC_CLASS_SM_120 ?
				    ARITHMETIC_CLASS_SM_120_NATIVE :
				    description.arithmetic_class;
		    for (uint32_t item_index = 0; item_index < COUNT_OF(items);
			 item_index++) {
			    item_run(card_index, arithmetic_class, item_index,
				     is_timed, &digest);
		    }
		    (void)printf(
			    "card %" PRIu32 " (class %" PRIu32
			    "): digest of everything it computed 0x%016" PRIX64
			    "\n",
			    card_index, arithmetic_class, digest);
	    }
    }
    if (test_linear_failures() != 0) {
        (void)fprintf(stderr, "wide_small_card_test FAILED: %d failures\n", test_linear_failures());
        return TEST_FAILED;
    }
    (void)printf("wide_small_card_test passed on %" PRIu32 " cards\n", card_count);
    return TEST_PASSED;
}
