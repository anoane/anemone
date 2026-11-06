/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * attention_core_test: source/attention_core.c against attention_core.h and attention_core_records.h, and the host
 * reference of the core (attention_core_reference.h) on rows whose results are known exactly. No card: a prepare
 * function is arithmetic on a description, the arena a number, the kernel handle a made-up number, and the kernel is
 * prepared on the registry of the host memory backend.
 *
 *   - The reference: its exponential at known points; its FP16 rounding at every boundary, and back to the bits of
 *     each of the 63,488 finite FP16 numbers; rows of exact results in the three forms: one key (the output is the
 *     key, rounded to bfloat16 in the kernels' forms), two keys of one score (their mean), a sink of the key's score
 *     (half the key), the rotation by a quarter turn (the pair swapped, one negated); one stored word of a key
 *     changed changes the output; a row whose maximum rises by more than the step: every form within the tolerance;
 *     two keys whose p FP16 and bfloat16 round apart: each kernel's form its own weight, by hand; the turning back
 *     of pairs whose bfloat16 bits tell its contraction from the other two forms: the record header's, written out
 *     here.
 *   - The launch of the model's layout (8 rows, 3 sessions, 16,384 tokens, pool ratio 2, so theta 1): its kernel in
 *     each order, grid, block, shared memory and every member of its record against figures computed by hand, every
 *     byte behind the record zero; heads 24 .. 39 alone; the window alone (theta 0), whose pool and selection are
 *     neither read nor checked.
 *   - The ranges laid out tight are accepted; one byte less in any of them, or in the arena, is refused.
 *   - Every refusal of attention_core.h, each alone in a description otherwise accepted: the status, one error line
 *     with the words of its rule, the launch untouched.
 *   - The kernels' preparation: its refusals, a card of no build (its status back, one error line), a card whose
 *     table takes the first kernel and not the second (the second named, both handles untouched), and on a card of
 *     each class both functions asked by their file, their names and the size of their record; the draft's block
 *     core left unprepared.
 *   - The block core (dspark_v1/DESIGN.md C3, H3): window_mode BLOCK at 1 to 8 rows, the fixed order's launch of its
 *     own kernel, no pool, rotation use 0; refused: with a pool of either ratio, in the reference order, a window
 *     mode beyond BLOCK, its kernel unprepared.
 * A failure is one line on the standard output: the case, and what was expected of it.
 */
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "anemone/attention_core.h"
#include "log_expectation.h"
#include "anemone/backend_seam.h"
#include "attention_core_reference.h"
#include "host_memory_backend.h"
#include "linear_card_fixture.h"
#include "linear_reference.h"

#define COUNT_OF(table) (sizeof(table) / sizeof((table)[0]))
#define INVALID STATUS_INVALID_ARGUMENT
#define CLASS_80 ARITHMETIC_CLASS_SM_80 /* the class of the host table's card */
#define MADE_UP_KERNEL 5u
#define MADE_UP_SCORES_KERNEL 6u
#define MADE_UP_VALUES_KERNEL 7u
#define MADE_UP_BLOCK_KERNEL 8u
#define UNTOUCHED_BYTE 0xA5
#define CAPTURE_CAPACITY 4096u
#define CARD_MEMORY_BYTES 65536u
#define HEADS MODEL_HEADS_VALUE
#define DIMS MODEL_HEAD_DIMS
/* The model's layout of a launch, laid out by hand in the order of the description's members. */
#define ROWS 8u
#define SESSIONS 3u
#define CONTEXT_LIMIT 16384u
#define STEP_OFFSET 0u
#define QUERIES_OFFSET 144u                     /* 16 + 8 * 16 */
#define RING_OFFSET 524432u                     /* + 8 * 65,536 */
#define POOL_OFFSET 13107344u                   /* + 3 * 4,194,304 */
#define SELECTION_OFFSET 38273168u              /* + 3 * 8,192 * 1,024 */
#define SINKS_OFFSET 38289552u                  /* + 8 * 2,048 */
#define ROTATION_OFFSET 38289808u               /* + 256 */
#define OUTPUT_OFFSET 38295952u                 /* + 8 * 768 */
#define SCORES_OFFSET 38820240u                 /* + 8 * 65,536 */
#define ARENA_BYTES 40130960u                   /* + 8 * 163,840 */
#define THETA_1_OFFSET 256u                     /* of use 1 in a row of the rotation */
#define RING_STRIDE 4194304u                    /* 4,096 slots of 1,024 */
#define POOL_STRIDE 8388608u                    /* 16,384 / 2 entries of 1,024 */
#define HEAD_BLOCKS 8u                          /* 64 heads in blocks of 8 */
#define KEY_TILES 10u                           /* 640 keys in tiles of 64 */
#define DIM_TILES 4u                            /* 512 dims in tiles of 128 */
#define EXACT_KEY_TILES 20u                     /* class 120 in the exact profile: tiles of 32 keys */
#define EXACT_DIM_TILES 4u                      /* and of 128 dims */
#define EXACT_VALUE_HEADS_AT_8_ROWS 2u          /* its value blocks' heads at 8 rows (attention_core_records.h) */
#define SOME_FIRST_HEAD 24u
#define SOME_HEAD_COUNT 16u
/* The reference's rows. */
#define TINY_SINK (-200.0f)                     /* exp(-200) is far under a last place of 1 */
#define VALUE_STEPS 2048u                       /* keys' values: (k - 1024) / 512, every one an FP16 number */
#define VALUE_CENTRE 1024
#define VALUE_SCALE 512.0
#define KEY_STREAM 7u
#define RISING_SCORE_CODE 0x5E40u               /* FP16 400: a key 400 along the query scores 17.7 */
#define EXP_ONE 2.718281828459045               /* e, to the last place of a double */
#define EXP_TOLERANCE 4.0e-16
#define EXP_SUBNORMAL_KEPT (-745.0)             /* exp of it: the least subnormal double */
#define EXP_SUBNORMAL_LOST (-746.0)             /* exp of it: zero */
#define EXP_BELOW_OVERFLOW 709.0
#define EXP_OVERFLOWING 710.0
#define UNTOUCHED_WORD 0xA5A5A5A5u              /* four untouched bytes */
#define RISING_KEYS 32u                         /* keys 0 .. 15 score 0, keys 16 .. 31 score 17.7 */
#define RISING_FIRST_KEY 16u
#define FP16_FINITE_COUNT 63488u                /* 2 * (31 * 1,024) */
#define FP16_EXPONENT_ALL_ONES 0x7C00u
#define FP16_SIGN 0x8000u
#define TURN_STREAM 0x21u
#define TURN_KEY        \
	(MODEL_WINDOW - \
	 1u) /* the one key of a row of the turning back: the newest window row */
#define APART_SCORE_CODE 0xCC00u                /* FP16 -16: a key -16 along the query scores -0.7071, p 0.49307 */
#define FILLER_NAME_CAPACITY 32u
#define ORDER_BEYOND 2u
#define ARITHMETIC_CLASS_PAIR 2u

enum launch_dimension { DIMENSION_X, DIMENSION_Y, DIMENSION_Z };

static int failures = 0;
static const char *context_text = "";
static char capture_path[] = "/tmp/anemone_attention_core_test_XXXXXX";
static int capture_descriptor = -1;             /* the temporary file that stands in for the error stream */
static int error_stream_descriptor = -1;        /* a duplicate of descriptor 2, kept to put it back */

static void expect(bool condition, const char *what)
{
    if (!condition) {
        (void)printf("FAILED: %s: %s\n", context_text, what);
        failures++;
    }
}

/* From here on, what is logged goes to the temporary file, from its start. */
static void capture_begin(void)
{
    expect(capture_descriptor >= 0 && fflush(stderr) == 0 && ftruncate(capture_descriptor, 0) == 0 &&
               lseek(capture_descriptor, 0, SEEK_SET) == 0 && dup2(capture_descriptor, STDERR_FILENO) == STDERR_FILENO,
           "the error stream is pointed at the temporary file");
}

/* What was logged since capture_begin, as a string; descriptor 2 is put back. */
static void capture_end(char *text, size_t text_capacity)
{
    const bool is_open = capture_descriptor >= 0 && error_stream_descriptor >= 0 && fflush(stderr) == 0;
    const ssize_t byte_count = is_open ? pread(capture_descriptor, text, text_capacity - 1, 0) : -1;
    expect(is_open && dup2(error_stream_descriptor, STDERR_FILENO) == STDERR_FILENO && byte_count >= 0,
           "the temporary file is read back and the error stream restored");
    text[byte_count >= 0 ? (size_t)byte_count : 0] = '\0';
}

static void expect_one_error_line_with(const char *text, const char *words, const char *what)
{
    if (!test_log_is_expected(text, words, true)) {
        (void)printf("FAILED: %s: %s: no single error line with \"%s\": \"%s\"\n", context_text, what, words, text);
        failures++;
    }
}

/* ---- the reference ---- */

static uint16_t key_value(uint32_t key, uint32_t dim)
{
    const uint64_t hash = test_linear_hash(KEY_STREAM, (uint64_t)key * DIMS + dim);
    return test_attention_half_bits(((double)(hash % VALUE_STEPS) - VALUE_CENTRE) / VALUE_SCALE);
}

struct exact_row {
    uint16_t queries[TEST_ATTENTION_ROW_VALUES];
    uint16_t keys[TEST_ATTENTION_KEYS][DIMS];
    float sinks[HEADS];
    float rotation[MODEL_FREQUENCIES * MODEL_PAIR_VALUES];
    double kernel[TEST_ATTENTION_ROW_VALUES];
    double reference_order[TEST_ATTENTION_ROW_VALUES];
    double exact[TEST_ATTENTION_ROW_VALUES];
    struct test_attention_row row;
};

/* A row of zero queries, of no key, sinks far under every score, every pair turned by cos, sin. */
static void exact_row_begin(struct exact_row *self, float cosine, float sine)
{
    memset(self, 0, sizeof(*self));
    for (uint32_t head = 0; head < HEADS; head++) {
        self->sinks[head] = TINY_SINK;
    }
    for (uint32_t pair = 0; pair < MODEL_FREQUENCIES; pair++) {
	    self->rotation[(size_t)pair * MODEL_PAIR_VALUES] = cosine;
	    self->rotation[(size_t)pair * MODEL_PAIR_VALUES + 1u] = sine;
    }
    self->row.queries = self->queries;
    self->row.sinks = self->sinks;
    self->row.rotation = self->rotation;
    self->row.head_count = HEADS;
}

static void exact_row_key(struct exact_row *self, uint32_t key)
{
    for (uint32_t dim = 0; dim < DIMS; dim++) {
        self->keys[key][dim] = key_value(key, dim);
    }
    self->row.keys[key] = self->keys[key];
}

static void exact_row_compute(struct exact_row *self)
{
    self->row.form = TEST_ATTENTION_KERNEL;
    expect(test_attention_core_row(&self->row, self->kernel) == STATUS_OK,
	   "the kernel's form is computed");
    self->row.form = TEST_ATTENTION_REFERENCE_ORDER;
    expect(test_attention_core_row(&self->row, self->reference_order) ==
		   STATUS_OK,
	   "the reference order's form is computed");
    self->row.form = TEST_ATTENTION_EXACT;
    expect(test_attention_core_row(&self->row, self->exact) == STATUS_OK,
	   "the exact form is computed");
}

/* Every head's dim of the three forms against what the definition gives exactly: expected (of every head,
 * unrounded). */
static void expect_row(const struct exact_row *self, const double *expected, const char *what)
{
    uint64_t differing = 0;
    for (uint32_t index = 0; index < TEST_ATTENTION_ROW_VALUES; index++) {
        const double rounded = test_linear_bfloat16_value(test_linear_bfloat16_nearest(expected[index]));
        differing += self->exact[index] != expected[index] || self->kernel[index] != rounded ||
                             self->reference_order[index] != rounded
                         ? 1u
                         : 0u;
    }
    expect(differing == 0, what);
}

struct boundary {
    double value;
    uint16_t bits;
};

static void test_exponential_and_rounding(void)
{
    context_text = "the reference's exponential and FP16";
    expect(test_attention_exp(0.0) == 1.0, "exp(0) is 1");
    const double one = test_attention_exp(1.0);
    const double minus_one = test_attention_exp(-1.0);
    expect(one > EXP_ONE * (1.0 - EXP_TOLERANCE) && one < EXP_ONE * (1.0 + EXP_TOLERANCE), "exp(1) is e");
    expect(minus_one * EXP_ONE > 1.0 - EXP_TOLERANCE && minus_one * EXP_ONE < 1.0 + EXP_TOLERANCE, "exp(-1) is 1/e");
    expect(test_attention_exp(EXP_SUBNORMAL_KEPT) > 0.0 && test_attention_exp(EXP_SUBNORMAL_LOST) == 0.0,
           "the subnormal end");
    const double above = test_attention_exp(EXP_OVERFLOWING);
    const double below = test_attention_exp(EXP_BELOW_OVERFLOW);
    expect(above > 0.0 && above + above == above && below + below != below, "an infinity above 709.78 alone");
    static const struct boundary boundaries[] = {
        {1.0, 0x3C00u}, {1.0 + 0x1p-11, 0x3C00u}, {1.0 + 0x3p-11, 0x3C02u}, {65504.0, 0x7BFFu},
        {65519.0, 0x7BFFu}, {65520.0, 0x7C00u}, {-65520.0, 0xFC00u}, {0x1p-24, 0x0001u}, {0x1p-25, 0x0000u},
        {0x3p-26, 0x0001u}, {-0x1p-26, 0x8000u}, {0x1p-14 - 0x1p-25, 0x0400u}, {0x3p-25, 0x0002u}, {-0.0, 0x8000u},
    };
    for (size_t index = 0; index < COUNT_OF(boundaries); index++) {
        expect(test_attention_half_bits(boundaries[index].value) == boundaries[index].bits,
               "a boundary of FP16 rounds as IEEE 754 says");
    }
    uint32_t finite_count = 0;
    uint32_t back_count = 0;
    for (uint32_t bits = 0; bits <= UINT16_MAX; bits++) {
        if ((bits & FP16_EXPONENT_ALL_ONES) != FP16_EXPONENT_ALL_ONES) {
            finite_count++;
            back_count += test_attention_half_bits(test_attention_core_half_value((uint16_t)bits)) == bits ? 1u : 0u;
        }
    }
    expect(finite_count == FP16_FINITE_COUNT && back_count == FP16_FINITE_COUNT,
           "every finite FP16 number rounds back to its bits");
    expect(test_attention_half_bits(test_attention_core_half_value(FP16_EXPONENT_ALL_ONES | 1u)) ==
               TEST_ATTENTION_HALF_NOT_A_NUMBER,
           "a not-a-number stays one");
}

static void test_exact_rows(void)
{
    static struct exact_row self;
    static double expected[TEST_ATTENTION_ROW_VALUES];
    context_text = "the reference's exact rows";
    /* one key: p = 1, the sink adds nothing: the output is the key in every head */
    exact_row_begin(&self, 1.0f, 0.0f);
    exact_row_key(&self, MODEL_WINDOW - 1u);
    exact_row_compute(&self);
    for (uint32_t index = 0; index < TEST_ATTENTION_ROW_VALUES; index++) {
	    expected[index] = test_attention_core_half_value(
		    self.keys[MODEL_WINDOW - 1u][index % DIMS]);
    }
    expect_row(&self, expected, "one key: the output is the key");
    self.keys[MODEL_WINDOW - 1u][0] ^= 1u;
    exact_row_compute(&self);
    expect(self.exact[0] != expected[0] && self.exact[DIMS] != expected[DIMS],
           "one stored word of the key changed: the output changes in every head");
    /* two keys of one score, one in the window and one in the pool: their mean */
    exact_row_begin(&self, 1.0f, 0.0f);
    exact_row_key(&self, 0);
    exact_row_key(&self, TEST_ATTENTION_KEYS - 1u);
    exact_row_compute(&self);
    for (uint32_t index = 0; index < TEST_ATTENTION_ROW_VALUES; index++) {
	    expected[index] =
		    (test_attention_core_half_value(
			     self.keys[0][index % DIMS]) +
		     test_attention_core_half_value(
			     self.keys[TEST_ATTENTION_KEYS - 1u][index % DIMS])) /
		    (double)MODEL_PAIR_VALUES;
    }
    expect_row(&self, expected, "two keys of one score: their mean");
    /* a sink of the key's score: half the key; a quarter turn: (x0, x1) to (x1, -x0) */
    exact_row_begin(&self, 0.0f, 1.0f);
    exact_row_key(&self, MODEL_WINDOW);
    for (uint32_t head = 0; head < HEADS; head++) {
        self.sinks[head] = 0.0f;
    }
    exact_row_compute(&self);
    for (uint32_t dim = 0; dim < DIMS; dim++) {
	    const uint32_t source = dim < MODEL_FIRST_ROTATED_DIM ? dim :
								    dim ^ 1u;
	    const double half = test_attention_core_half_value(
					self.keys[MODEL_WINDOW][source]) /
				(double)MODEL_PAIR_VALUES;
	    for (uint32_t head = 0; head < HEADS; head++) {
		    expected[head * DIMS + dim] =
			    dim >= MODEL_FIRST_ROTATED_DIM &&
					    dim % MODEL_PAIR_VALUES != 0 ?
				    -half :
				    half;
	    }
    }
    expect_row(&self, expected, "a sink of the key's score, turned a quarter: half the key, its pairs turned");
    /* the maximum rises by more than the step from group 0 to group 1: the kernel's form rescales */
    exact_row_begin(&self, 1.0f, 0.0f);
    for (uint32_t head = 0; head < HEADS; head++) {
        self.queries[(size_t)head * DIMS] = test_attention_half_bits(1.0);
    }
    for (uint32_t key = 0; key < RISING_KEYS; key++) {
        exact_row_key(&self, key);
        self.keys[key][0] = key >= RISING_FIRST_KEY ? RISING_SCORE_CODE : 0u;
    }
    exact_row_compute(&self);
    struct test_attention_distance distance = {0};
    uint16_t kernel_bits[TEST_ATTENTION_ROW_VALUES];
    for (uint32_t index = 0; index < TEST_ATTENTION_ROW_VALUES; index++) {
        kernel_bits[index] = test_linear_bfloat16_nearest(self.kernel[index]);
    }
    test_attention_distance_add(&distance, kernel_bits, self.exact, TEST_ATTENTION_ROW_VALUES);
    expect(test_attention_core_within(&distance, TEST_ATTENTION_TOLERANCE) &&
               !test_attention_core_within(&distance, 0.0) && distance.largest_places <= 1,
           "a rescaled row: its two forms differ by the rounding to bfloat16, a place at most");
    struct test_attention_distance reference_order = {0};
    for (uint32_t index = 0; index < TEST_ATTENTION_ROW_VALUES; index++) {
        kernel_bits[index] = test_linear_bfloat16_nearest(self.reference_order[index]);
    }
    test_attention_distance_add(&reference_order, kernel_bits, self.exact, TEST_ATTENTION_ROW_VALUES);
    expect(test_attention_core_within(&reference_order, TEST_ATTENTION_TOLERANCE) &&
               reference_order.largest_places <= 1,
           "a rescaled row in the reference order's form: within a place of the definition");
    struct test_attention_distance same = {0};
    test_attention_distance_add(&same, kernel_bits, self.kernel, TEST_ATTENTION_ROW_VALUES);
    expect(test_attention_core_within(&same, 0.0) && same.equal_count == TEST_ATTENTION_ROW_VALUES &&
               same.largest_places == 0,
           "a row against itself: no distance");
    self.row.first_head = HEADS;
    self.row.head_count = 1u;
    expect(test_attention_core_row(&self.row, self.kernel) == INVALID &&
               test_attention_core_row(NULL, self.kernel) == INVALID,
           "heads beyond the 64 and a missing row are refused");
}

/* Two keys of scores 0 and -0.7071: p = 1 and 0.49307, which FP16 and bfloat16 round apart. Each kernel's form is
 * (key 0 + w key 1) / (1 + p + exp(sink)) rounded to bfloat16 with its own w, the exact form with p itself; the
 * angle of 0 turns nothing back. */
static void test_weights_apart(void)
{
    static struct exact_row self;
    context_text = "the reference's two weights";
    exact_row_begin(&self, 1.0f, 0.0f);
    for (uint32_t head = 0; head < HEADS; head++) {
        self.queries[(size_t)head * DIMS] = test_attention_half_bits(1.0);
    }
    exact_row_key(&self, 0);
    exact_row_key(&self, 1u);
    self.keys[0][0] = 0;
    self.keys[1u][0] = APART_SCORE_CODE;
    exact_row_compute(&self);
    const double probability = test_attention_exp(
	    test_attention_core_half_value(APART_SCORE_CODE) *
	    (double)ATTENTION_CORE_SCALE);
    const double scale = (double)ATTENTION_CORE_WEIGHT_SCALE;
    const double weights[] = {
        test_attention_core_half_value(test_attention_half_bits(probability)),
        test_attention_core_half_value(test_attention_half_bits(
            test_linear_bfloat16_value(test_linear_bfloat16_nearest(probability)) * scale)) / scale};
    const double denominator = 1.0 + probability + test_attention_exp((double)TINY_SINK);
    uint64_t differing = 0, apart = 0;
    for (uint32_t index = 0; index < TEST_ATTENTION_ROW_VALUES; index++) {
        const double first = test_attention_core_half_value(self.keys[0][index % DIMS]);
        const double second = test_attention_core_half_value(self.keys[1u][index % DIMS]);
        const double kernel = (first + weights[0] * second) / denominator;
        const double reference_order = (first + weights[1u] * second) / denominator;
        differing += self.kernel[index] != test_linear_bfloat16_value(test_linear_bfloat16_nearest(kernel)) ||
                             self.reference_order[index] !=
                                 test_linear_bfloat16_value(test_linear_bfloat16_nearest(reference_order)) ||
                             self.exact[index] != (first + probability * second) / denominator
                         ? 1u
                         : 0u;
        apart += self.kernel[index] != self.reference_order[index] ? 1u : 0u;
    }
    expect(weights[0] != weights[1u] && differing == 0 && apart != 0,
           "each form weighs the second key with its own rounding of p, and the two forms differ");
}

/* The turning back of a row of one key (the output is the key) by pairs found where the record header's form,
 * written out here, and form give other bfloat16 bits in part: the reference holds the record header's. */
static void test_turn_contraction(void)
{
    static struct exact_row self;
    context_text = "the turning back's contraction";
    for (uint32_t form = TEST_ATTENTION_TURN_OTHER; form < TEST_ATTENTION_TURN_FORMS; form++) {
	    for (uint32_t part = 0; part < MODEL_PAIR_VALUES; part++) {
		    float pair[MODEL_PAIR_VALUES] = { 0.0f, 0.0f };
		    float angle[MODEL_PAIR_VALUES] = { 0.0f, 0.0f };
		    expect(test_attention_turn_case(
				   TURN_STREAM + form * MODEL_PAIR_VALUES +
					   part,
				   part, (enum test_attention_turn_form)form,
				   pair, angle),
			   "a pair whose bits tell the forms apart is found");
		    const float documented[MODEL_PAIR_VALUES] = {
			    fmaf(pair[0], angle[0], pair[1] * angle[1]),
			    fmaf(pair[1], angle[0], -(pair[0] * angle[1]))
		    };
		    const float other = test_attention_turn_back(
			    pair[0], pair[1], angle[0], angle[1], part,
			    (enum test_attention_turn_form)form);
		    exact_row_begin(&self, angle[0], angle[1]);
		    exact_row_key(&self, TURN_KEY);
		    for (uint32_t value = 0; value < MODEL_PAIR_VALUES;
			 value++) {
			    self.keys[TURN_KEY][MODEL_FIRST_ROTATED_DIM + value] =
				    test_attention_half_bits(pair[value]);
		    }
		    exact_row_compute(&self);
		    uint64_t differing = 0;
		    for (uint32_t head = 0; head < HEADS; head++) {
			    const double *const turned =
				    self.kernel + (size_t)head * DIMS +
				    MODEL_FIRST_ROTATED_DIM;
			    for (uint32_t value = 0; value < MODEL_PAIR_VALUES;
				 value++) {
				    const uint16_t expected =
					    test_linear_bfloat16_nearest(
						    (double)documented[value]);
				    differing +=
					    turned[value] !=
							    test_linear_bfloat16_value(
								    expected) ?
						    1u :
						    0u;
			    }
		    }
		    expect(differing == 0 &&
				   test_linear_bfloat16_nearest(
					   (double)other) !=
					   test_linear_bfloat16_nearest(
						   (double)documented[part]),
			   "the reference turns back as the record header says, which the other form does not give");
	    }
    }
}

/* ---- the launch ---- */

static struct attention_core_kernels made_up_kernels = { MADE_UP_KERNEL,
							 MADE_UP_SCORES_KERNEL,
							 MADE_UP_VALUES_KERNEL,
							 MADE_UP_BLOCK_KERNEL };

static struct attention_core_description model_description(void)
{
	return (struct attention_core_description){
		.step = { STEP_OFFSET, QUERIES_OFFSET - STEP_OFFSET },
		.queries = { QUERIES_OFFSET, RING_OFFSET - QUERIES_OFFSET },
		.ring = { RING_OFFSET, POOL_OFFSET - RING_OFFSET },
		.pool = { POOL_OFFSET, SELECTION_OFFSET - POOL_OFFSET },
		.selection = { SELECTION_OFFSET,
			       SINKS_OFFSET - SELECTION_OFFSET },
		.sinks = { SINKS_OFFSET, ROTATION_OFFSET - SINKS_OFFSET },
		.rotation = { ROTATION_OFFSET,
			      OUTPUT_OFFSET - ROTATION_OFFSET },
		.output = { OUTPUT_OFFSET, SCORES_OFFSET - OUTPUT_OFFSET },
		.scores = { SCORES_OFFSET, ARENA_BYTES - SCORES_OFFSET },
		.row_count = ROWS,
		.session_capacity = SESSIONS,
		.context_token_limit = CONTEXT_LIMIT,
		.pool_ratio = MODEL_POOLED_RATIO,
		.first_head = 0,
		.head_count = HEADS,
	};
}

/* A launch of `kernel` on grid (rows, grid_rows, grid_tiles), its blocks of `threads` threads and shared memory,
 * and the record. */
static void expect_launch(const struct kernel_launch *launch, uint32_t kernel,
			  uint32_t rows, uint32_t grid_rows,
			  uint32_t grid_tiles, uint32_t threads,
			  uint32_t shared_bytes,
			  const struct attention_core_arguments_record *record,
			  const char *what)
{
	static const unsigned char zeros[LAUNCH_ARGUMENT_BYTES] = { 0 };
	expect(launch->kernel == kernel && launch->grid[DIMENSION_X] == rows &&
		       launch->grid[DIMENSION_Y] == grid_rows &&
		       launch->grid[DIMENSION_Z] == grid_tiles &&
		       launch->block[DIMENSION_X] == threads &&
		       launch->block[DIMENSION_Y] == 1u &&
		       launch->block[DIMENSION_Z] == 1u &&
		       launch->dynamic_shared_byte_count == shared_bytes &&
		       launch->argument_byte_count ==
			       ATTENTION_CORE_ARGUMENT_BYTES &&
		       launch->branch == LAUNCH_CHAIN &&
		       memcmp(launch->argument, record, sizeof(*record)) == 0 &&
		       memcmp(launch->argument + sizeof(*record), zeros,
			      sizeof(zeros) - sizeof(*record)) == 0,
	       what);
}

/* The order's launches of description, each record record with its heads_per_block (heads_per_block[launch]). */
static uint32_t
launches_of(const struct attention_core_description *description,
	    struct kernel_launch launches[ATTENTION_CORE_MOST_LAUNCHES])
{
    uint32_t count = 0;
    memset(launches, UNTOUCHED_BYTE,
	   ATTENTION_CORE_MOST_LAUNCHES * sizeof(launches[0]));
    return attention_core_prepare(description, ARENA_BYTES, &made_up_kernels,
				  launches, &count) == STATUS_OK ?
		   count :
		   0u;
}

static void expect_refused(const struct attention_core_description *description,
			   uint64_t arena_byte_count,
			   const struct attention_core_kernels *kernels,
			   const char *words, const char *what);

/* most_heads_per_block (--tp_core_heads): the reference order's blocks of at most so many heads, 8 rows of a TP rank's
 * 16 heads: scores and values each two blocks of 8 heads where the rule gives one of 16; 0 the rule; 4 refused. */
static void test_most_heads(void)
{
	struct kernel_launch launches[ATTENTION_CORE_MOST_LAUNCHES];
	struct attention_core_description description = model_description();
	context_text = "the most heads a block";
	description.order = LINEAR_ORDER_REFERENCE;
	description.head_count = 2u * ATTENTION_CORE_HEADS_PER_BLOCK;
	expect(launches_of(&description, launches) ==
			       ATTENTION_CORE_MOST_LAUNCHES &&
		       launches[0].grid[DIMENSION_Y] == 1u &&
		       launches[1].grid[DIMENSION_Y] == 1u,
	       "the rule: 8 rows of 16 heads, a block of 16 heads in either launch");
	description.most_heads_per_block = ATTENTION_CORE_HEADS_PER_BLOCK;
	struct attention_core_arguments_record scores_record, values_record;
	expect(launches_of(&description, launches) ==
			       ATTENTION_CORE_MOST_LAUNCHES &&
		       launches[0].grid[DIMENSION_Y] == 2u &&
		       launches[1].grid[DIMENSION_Y] == 2u &&
		       launches[0].dynamic_shared_byte_count ==
			       ATTENTION_CORE_SCORES_SHARED_BYTES(1u) &&
		       launches[1].dynamic_shared_byte_count ==
			       ATTENTION_CORE_VALUES_SHARED_BYTES(1u),
	       "at most 8: two blocks of 8 heads in either launch, the shared memory of one tile");
	memcpy(&scores_record, launches[0].argument, sizeof(scores_record));
	memcpy(&values_record, launches[1].argument, sizeof(values_record));
	expect(scores_record.heads_per_block ==
			       ATTENTION_CORE_HEADS_PER_BLOCK &&
		       values_record.heads_per_block ==
			       ATTENTION_CORE_HEADS_PER_BLOCK,
	       "at most 8: each record names 8 heads a block");
	description.most_heads_per_block = ATTENTION_CORE_HEADS_PER_BLOCK / 2u;
	struct kernel_launch untouched;
	memset(&untouched, UNTOUCHED_BYTE, sizeof(untouched));
	expect(launches_of(&description, launches) == 0u &&
		       memcmp(&launches[0], &untouched, sizeof(untouched)) == 0,
	       "at most 4: refused, nothing written");
}

static void test_model_launches(void)
{
	struct kernel_launch launches[ATTENTION_CORE_MOST_LAUNCHES];
	struct attention_core_description description = model_description();
	struct attention_core_arguments_record record = {
		STEP_OFFSET,
		QUERIES_OFFSET,
		RING_OFFSET,
		RING_STRIDE,
		POOL_OFFSET,
		POOL_STRIDE,
		SELECTION_OFFSET,
		SINKS_OFFSET,
		ROTATION_OFFSET + THETA_1_OFFSET,
		OUTPUT_OFFSET,
		0,
		CONTEXT_LIMIT,
		SESSIONS,
		MODEL_POOLED_RATIO,
		0,
		ATTENTION_CORE_HEADS_PER_BLOCK,
		0,
		0,
		0
	};
	const uint32_t threads = ATTENTION_CORE_THREADS,
		       value_threads = ATTENTION_CORE_VALUE_THREADS;
	context_text = "the model's launches";
	expect(sizeof(record) == ATTENTION_CORE_ARGUMENT_BYTES,
	       "the record has its size");
	expect(launches_of(&description, launches) == 1u,
	       "the model's layout is accepted: one launch");
	expect_launch(
		&launches[0], MADE_UP_KERNEL, ROWS, HEAD_BLOCKS, 1u, threads,
		ATTENTION_CORE_SHARED_BYTES_VALUE, &record,
		"every head: 8 rows by 8 blocks of heads, the record by hand");
	description.order = LINEAR_ORDER_REFERENCE;
	record.scores_offset = SCORES_OFFSET;
	record.heads_per_block = HEADS;
	expect(launches_of(&description, launches) ==
		       ATTENTION_CORE_MOST_LAUNCHES,
	       "the reference order: two launches");
	expect_launch(
		&launches[0], MADE_UP_SCORES_KERNEL, ROWS, 1u, KEY_TILES,
		threads,
		ATTENTION_CORE_SCORES_SHARED_BYTES(
			ATTENTION_CORE_SCORE_TILES_MOST),
		&record,
		"its scores at 8 rows: every head a block (a key read once a row), the scores' offset");
	record.heads_per_block = ATTENTION_CORE_VALUE_TILES_MOST *
				 ATTENTION_CORE_HEADS_PER_BLOCK;
	expect_launch(
		&launches[1], MADE_UP_VALUES_KERNEL, ROWS,
		HEADS / record.heads_per_block, DIM_TILES, value_threads,
		ATTENTION_CORE_VALUES_SHARED_BYTES(
			ATTENTION_CORE_VALUE_TILES_MOST),
		&record,
		"its values at 8 rows: blocks of 16 heads and tiles of dims, the same record");
	description.row_count = 1;
	description.step.byte_count =
		STEP_BLOCK_BYTES_VALUE + STEP_ROW_BYTES_VALUE;
	record.heads_per_block = ATTENTION_CORE_HEADS_PER_BLOCK;
	expect(launches_of(&description, launches) ==
		       ATTENTION_CORE_MOST_LAUNCHES,
	       "the reference order at one row");
	expect_launch(&launches[0], MADE_UP_SCORES_KERNEL, 1u, HEAD_BLOCKS,
		      KEY_TILES, threads,
		      ATTENTION_CORE_SCORES_SHARED_BYTES(1u), &record,
		      "its scores at a row: 8 heads a block");
	expect_launch(&launches[1], MADE_UP_VALUES_KERNEL, 1u, HEAD_BLOCKS,
		      DIM_TILES, value_threads,
		      ATTENTION_CORE_VALUES_SHARED_BYTES(1u), &record,
		      "its values at a row: 8 heads a block");
#if !defined(LINEAR_CLASS_FORMS)
	description = model_description();
	description.order = LINEAR_ORDER_REFERENCE;
	description.arithmetic_class = ARITHMETIC_CLASS_SM_120;
	record.heads_per_block = ATTENTION_CORE_EXACT_SCORE_HEADS;
	expect(launches_of(&description, launches) ==
		       ATTENTION_CORE_MOST_LAUNCHES,
	       "the exact profile on class 120: two launches");
	expect_launch(
		&launches[0], MADE_UP_SCORES_KERNEL, ROWS,
		HEADS / ATTENTION_CORE_EXACT_SCORE_HEADS, EXACT_KEY_TILES,
		threads, ATTENTION_CORE_EXACT_SCORES_SHARED_BYTES, &record,
		"class 120's scores: its blocks of heads and tiles of keys");
	record.heads_per_block = EXACT_VALUE_HEADS_AT_8_ROWS;
	expect_launch(&launches[1], MADE_UP_VALUES_KERNEL, ROWS,
		      HEADS / EXACT_VALUE_HEADS_AT_8_ROWS, EXACT_DIM_TILES,
		      threads, ATTENTION_CORE_EXACT_VALUES_SHARED_BYTES,
		      &record, "class 120's values at 8 rows");
	description.row_count = 1;
	description.step.byte_count =
		STEP_BLOCK_BYTES_VALUE + STEP_ROW_BYTES_VALUE;
	record.heads_per_block = 1;
	expect(launches_of(&description, launches) ==
		       ATTENTION_CORE_MOST_LAUNCHES,
	       "class 120 at one row");
	expect_launch(&launches[1], MADE_UP_VALUES_KERNEL, 1u, HEADS,
		      EXACT_DIM_TILES, threads,
		      ATTENTION_CORE_EXACT_VALUES_SHARED_BYTES, &record,
		      "class 120's values at a row: a head a block");
#endif
    description = model_description();
    description.first_head = SOME_FIRST_HEAD;
    description.head_count = SOME_HEAD_COUNT;
    record.first_head = SOME_FIRST_HEAD;
    record.scores_offset = 0;
    record.heads_per_block = ATTENTION_CORE_HEADS_PER_BLOCK;
    expect(launches_of(&description, launches) == 1u, "heads 24 .. 39 are accepted");
    expect_launch(
	    &launches[0], MADE_UP_KERNEL, ROWS,
	    SOME_HEAD_COUNT / ATTENTION_CORE_HEADS_PER_BLOCK, 1u, threads,
	    ATTENTION_CORE_SHARED_BYTES_VALUE, &record,
	    "heads 24 .. 39: two blocks of heads, the first head in the record");
    description.order = LINEAR_ORDER_REFERENCE;
    record.scores_offset = SCORES_OFFSET;
    record.heads_per_block = SOME_HEAD_COUNT;
    expect(launches_of(&description, launches) == ATTENTION_CORE_MOST_LAUNCHES,
	   "heads 24 .. 39 in the reference order");
    expect_launch(&launches[0], MADE_UP_SCORES_KERNEL, ROWS, 1u, KEY_TILES,
		  threads,
		  ATTENTION_CORE_SCORES_SHARED_BYTES(
			  SOME_HEAD_COUNT / ATTENTION_CORE_HEADS_PER_BLOCK),
		  &record,
		  "heads 24 .. 39: the most heads that divide them, a block");
    /* output_compact: heads 24 .. 39 into rows of those heads alone (output_row_heads 16), 16 heads' bytes a row */
    description.order = model_description().order;
    description.output_compact = 1;
    description.output.byte_count =
	    (uint64_t)ROWS * SOME_HEAD_COUNT * MODEL_STATE_ROW_BYTES;
    record.scores_offset = 0;
    record.heads_per_block = ATTENTION_CORE_HEADS_PER_BLOCK;
    record.output_row_heads = SOME_HEAD_COUNT;
    expect(launches_of(&description, launches) == 1u, "heads 24 .. 39 into compact rows are accepted");
    expect_launch(&launches[0], MADE_UP_KERNEL, ROWS,
		  SOME_HEAD_COUNT / ATTENTION_CORE_HEADS_PER_BLOCK, 1u, threads,
		  ATTENTION_CORE_SHARED_BYTES_VALUE, &record,
		  "compact rows: the head count a row");
    description.output.byte_count -= 1u;
    expect_refused(&description, ARENA_BYTES, &made_up_kernels, "the output", "compact rows one byte short refused");
    description.output.byte_count += 1u;
    description.output_compact = 2;
    expect_refused(&description, ARENA_BYTES, &made_up_kernels, "output_compact 2: 0 or 1", "output_compact 2 refused");
    description = model_description();
    description.pool_ratio = 0;
    description.pool = (struct linear_range){ UINT64_MAX, UINT64_MAX };
    description.selection = description.ring;
    description.scores = description.ring;
    record = (struct attention_core_arguments_record){
	    STEP_OFFSET,
	    QUERIES_OFFSET,
	    RING_OFFSET,
	    RING_STRIDE,
	    0,
	    0,
	    0,
	    SINKS_OFFSET,
	    ROTATION_OFFSET,
	    OUTPUT_OFFSET,
	    0,
	    CONTEXT_LIMIT,
	    SESSIONS,
	    0,
	    0,
	    ATTENTION_CORE_HEADS_PER_BLOCK,
	    0,
	    0,
	    0
    };
    expect(launches_of(&description, launches) == 1u,
           "the window alone is accepted whatever its pool, selection and (in the fixed order) scores");
    expect_launch(&launches[0], MADE_UP_KERNEL, ROWS, HEAD_BLOCKS, 1u, threads,
		  ATTENTION_CORE_SHARED_BYTES_VALUE, &record,
		  "the window alone: no pool, theta 0");
}

/* ---- the refusals ---- */

static void expect_refused(const struct attention_core_description *description,
			   uint64_t arena_byte_count,
			   const struct attention_core_kernels *kernels,
			   const char *words, const char *what)
{
	struct kernel_launch launches[ATTENTION_CORE_MOST_LAUNCHES];
	struct kernel_launch untouched[ATTENTION_CORE_MOST_LAUNCHES];
	uint32_t count = UNTOUCHED_WORD;
	char logged[CAPTURE_CAPACITY];
	memset(launches, UNTOUCHED_BYTE, sizeof(launches));
	memset(untouched, UNTOUCHED_BYTE, sizeof(untouched));
	capture_begin();
	expect(attention_core_prepare(description, arena_byte_count, kernels,
				      launches, &count) == INVALID,
	       what);
	capture_end(logged, sizeof(logged));
	expect_one_error_line_with(logged, words, what);
	expect(memcmp(launches, untouched, sizeof(launches)) == 0 &&
		       count == UNTOUCHED_WORD,
	       "a refusal leaves the launches and their count untouched");
}

enum member { ROWS_MEMBER, SESSIONS_MEMBER, LIMIT_MEMBER, RATIO_MEMBER, FIRST_HEAD_MEMBER, HEAD_COUNT_MEMBER };

struct member_refusal {
    enum member member;
    uint32_t value;
    const char *words;
};
/* The words: the refused count as the error line states it (each line states every rule). */
static const struct member_refusal member_refusals[] = {
	{ ROWS_MEMBER, 0, ", 0 rows," },
	{ ROWS_MEMBER, LINEAR_WIDE_MOST_ROWS + 1u,
	  ", 1025 rows," }, /* chunk_core */
	{ SESSIONS_MEMBER, 0, " 0 sessions," },
	{ LIMIT_MEMBER, 0, " 0 tokens," },
	{ LIMIT_MEMBER,
	  CONTEXT_LIMIT - MODEL_CONTEXT_GRANULE / MODEL_POOLED_RATIO,
	  " 16376 tokens," },
	{ LIMIT_MEMBER, MODEL_CONTEXT_MOST + MODEL_CONTEXT_GRANULE,
	  " 1048592 tokens," },
	{ RATIO_MEMBER, MODEL_POOLED_RATIO + 1u, "ratio 3," },
	{ FIRST_HEAD_MEMBER,
	  ATTENTION_CORE_HEADS_PER_BLOCK / MODEL_POOLED_RATIO,
	  "heads from 4:" },
	{ FIRST_HEAD_MEMBER, HEADS + ATTENTION_CORE_HEADS_PER_BLOCK,
	  "heads from 72:" },
	{ HEAD_COUNT_MEMBER, 0, " 0 heads from" },
	{ HEAD_COUNT_MEMBER, SOME_HEAD_COUNT - 1u, " 15 heads from" },
	{ HEAD_COUNT_MEMBER, HEADS + ATTENTION_CORE_HEADS_PER_BLOCK,
	  " 72 heads from" },
};

static void test_member_refusals(void)
{
    context_text = "a count refused";
    for (size_t index = 0; index < COUNT_OF(member_refusals); index++) {
	    struct attention_core_description description = model_description();
	    uint32_t *const members[] = { &description.row_count,
					  &description.session_capacity,
					  &description.context_token_limit,
					  &description.pool_ratio,
					  &description.first_head,
					  &description.head_count };
	    *members[member_refusals[index].member] =
		    member_refusals[index].value;
	    expect_refused(&description, ARENA_BYTES, &made_up_kernels,
			   member_refusals[index].words,
			   member_refusals[index].words);
    }
    struct attention_core_description description = model_description();
    description.first_head = HEADS - ATTENTION_CORE_HEADS_PER_BLOCK;
    description.head_count = SOME_HEAD_COUNT;
    expect_refused(&description, ARENA_BYTES, &made_up_kernels, " 16 heads from 56:", "heads past the 64th");
    description = model_description();
    description.order = (enum linear_order_code)ORDER_BEYOND;
    expect_refused(&description, ARENA_BYTES, &made_up_kernels, ": order 2,", "an order beyond the reference's");
}

static void test_range_refusals(void)
{
	struct kernel_launch launches[ATTENTION_CORE_MOST_LAUNCHES];
	context_text = "a range refused";
	struct attention_core_description tight = model_description();
	tight.order = LINEAR_ORDER_REFERENCE;
	struct attention_core_description description = tight;
	struct linear_range *const ranges[] = {
		&description.step,	&description.queries,
		&description.ring,	&description.pool,
		&description.selection, &description.sinks,
		&description.rotation,	&description.output,
		&description.scores
	};
	expect(launches_of(&tight, launches) == ATTENTION_CORE_MOST_LAUNCHES,
	       "the ranges laid out tight are accepted");
	for (size_t index = 0; index < COUNT_OF(ranges); index++) {
		description = tight;
		ranges[index]->byte_count--;
		expect_refused(&description, ARENA_BYTES, &made_up_kernels,
			       "the launches touch",
			       "one byte less in a range");
	}
    expect_refused(&tight, ARENA_BYTES - 1u, &made_up_kernels, "leaves the arena", "one byte less in the arena");
    description = tight;
    description.ring.byte_offset +=
	    LINEAR_OFFSET_ALIGNMENT_BYTES / MODEL_POOLED_RATIO;
    expect_refused(&description, ARENA_BYTES, &made_up_kernels, "an offset is a multiple of", "a range off its line");
    description = tight;
    description.output = description.queries;
    expect_refused(&description, ARENA_BYTES, &made_up_kernels, "share a byte", "the output on the queries");
    description = tight;
    description.scores.byte_offset = description.output.byte_offset;    /* its size, from the output's first byte */
    expect_refused(&description, ARENA_BYTES, &made_up_kernels, "share a byte", "the scores on the output");
}

static void test_argument_refusals(void)
{
	struct attention_core_description description = model_description();
	const struct attention_core_kernels no_fixed = { BACKEND_NO_KERNEL,
							 MADE_UP_SCORES_KERNEL,
							 MADE_UP_VALUES_KERNEL,
							 MADE_UP_BLOCK_KERNEL };
	const struct attention_core_kernels no_reference = {
		MADE_UP_KERNEL, MADE_UP_SCORES_KERNEL, BACKEND_NO_KERNEL,
		MADE_UP_BLOCK_KERNEL
	};
	struct kernel_launch launches[ATTENTION_CORE_MOST_LAUNCHES];
	uint32_t count = 0;
	char logged[CAPTURE_CAPACITY];
	context_text = "an argument refused";
	expect_refused(NULL, ARENA_BYTES, &made_up_kernels,
		       "a missing argument", "no description");
	expect_refused(&description, ARENA_BYTES, NULL, "a missing argument",
		       "no kernels");
	expect_refused(&description, ARENA_BYTES, &no_fixed,
		       KERNEL_ATTENTION_CORE " is not prepared",
		       "the fixed order's kernel not prepared");
	expect(attention_core_prepare(&description, ARENA_BYTES, &no_reference,
				      launches, &count) == STATUS_OK &&
		       count == 1u && launches[0].kernel == MADE_UP_KERNEL,
	       "the fixed order needs no reference kernel");
	description.order = LINEAR_ORDER_REFERENCE;
	expect_refused(&description, ARENA_BYTES, &no_reference,
		       KERNEL_ATTENTION_CORE_VALUES " is not prepared",
		       "a reference order's kernel not prepared");
	expect(attention_core_prepare(&description, ARENA_BYTES, &no_fixed,
				      launches, &count) == STATUS_OK &&
		       count == ATTENTION_CORE_MOST_LAUNCHES &&
		       launches[0].kernel == MADE_UP_SCORES_KERNEL &&
		       launches[1].kernel == MADE_UP_VALUES_KERNEL,
	       "the reference order needs no fixed kernel");
	description.order = LINEAR_ORDER_FIXED;
	capture_begin();
	expect(attention_core_prepare(&description, ARENA_BYTES,
				      &made_up_kernels, NULL,
				      &count) == INVALID &&
		       attention_core_prepare(&description, ARENA_BYTES,
					      &made_up_kernels, launches,
					      NULL) == INVALID,
	       "nowhere to put the launches or their count");
	capture_end(logged, sizeof(logged));
	expect(test_log_is_expected(logged, "a missing argument", false),
	       "nowhere to put the launches or their count");
}

/* ---- the block core ---- */

static void test_block(void)
{
	struct kernel_launch launches[ATTENTION_CORE_MOST_LAUNCHES];
	context_text = "the block core";
	for (uint32_t rows = 1; rows <= ROWS; rows++) {
		struct attention_core_description description =
			model_description();
		description.pool_ratio = 0;
		description.window_mode = ATTENTION_CORE_WINDOW_BLOCK;
		description.row_count = rows;
		description.step.byte_count =
			STEP_BLOCK_BYTES_VALUE +
			(uint64_t)rows * STEP_ROW_BYTES_VALUE;
		const struct attention_core_arguments_record record = {
			STEP_OFFSET,
			QUERIES_OFFSET,
			RING_OFFSET,
			RING_STRIDE,
			0,
			0,
			0,
			SINKS_OFFSET,
			ROTATION_OFFSET,
			OUTPUT_OFFSET,
			0,
			CONTEXT_LIMIT,
			SESSIONS,
			0,
			0,
			ATTENTION_CORE_HEADS_PER_BLOCK,
			0,
			0,
			0
		};
		expect(launches_of(&description, launches) == 1u,
		       "accepted: one launch");
		expect_launch(
			&launches[0], MADE_UP_BLOCK_KERNEL, rows, HEAD_BLOCKS,
			1u, ATTENTION_CORE_THREADS,
			ATTENTION_CORE_SHARED_BYTES_VALUE, &record,
			"the fixed order's launch of the block kernel: no pool, the rotation's use 0");
	}
	struct attention_core_description description = model_description();
	description.window_mode = ATTENTION_CORE_WINDOW_BLOCK;
	expect_refused(&description, ARENA_BYTES, &made_up_kernels, "window 1,",
		       "the block with a pool of ratio 2");
	description.pool_ratio = MODEL_DIRECT_RATIO;
	expect_refused(&description, ARENA_BYTES, &made_up_kernels, "window 1,",
		       "the block with a pool of ratio 1");
	description.pool_ratio = 0;
	description.order = LINEAR_ORDER_REFERENCE;
	expect_refused(&description, ARENA_BYTES, &made_up_kernels,
		       "order 1, window 1,",
		       "the block in the reference order");
	description.order = LINEAR_ORDER_FIXED;
	description.window_mode = ATTENTION_CORE_WINDOW_BLOCK + 1u;
	expect_refused(&description, ARENA_BYTES, &made_up_kernels, "window 2,",
		       "a window mode beyond the block");
	description.window_mode = ATTENTION_CORE_WINDOW_BLOCK;
	struct attention_core_kernels no_block = made_up_kernels;
	no_block.attention_core_block = BACKEND_NO_KERNEL;
	expect_refused(&description, ARENA_BYTES, &no_block,
		       KERNEL_ATTENTION_CORE_BLOCK " is not prepared",
		       "the block kernel not prepared");
	description.window_mode = ATTENTION_CORE_WINDOW_CAUSAL;
	uint32_t count = 0;
	expect(attention_core_prepare(&description, ARENA_BYTES, &no_block,
				      launches, &count) == STATUS_OK &&
		       count == 1u && launches[0].kernel == MADE_UP_KERNEL,
	       "the causal window needs no block kernel");
}

/* ---- the kernel's preparation ---- */

/* Every kernel of the core asked of a card of the class by their file, their names and the size of their record. */
static bool every_kernel_asked(const struct attention_core_kernels *kernels,
			       const char *file_name)
{
	static const char *const names[] = { KERNEL_ATTENTION_CORE,
					     KERNEL_ATTENTION_CORE_SCORES,
					     KERNEL_ATTENTION_CORE_VALUES };
	const uint32_t handles[] = { kernels->attention_core,
				     kernels->attention_core_scores,
				     kernels->attention_core_values };
	bool asked = true;
	for (size_t first = 0; first < COUNT_OF(handles); first++) {
		for (size_t second = first + 1u; second < COUNT_OF(handles);
		     second++) {
			asked = asked && handles[first] != handles[second];
		}
    }
    for (size_t index = 0; index < COUNT_OF(names); index++) {
        struct test_host_memory_backend_kernel prepared;
	asked = asked &&
		test_host_memory_backend_kernel_of(0, handles[index],
						   &prepared) &&
		strcmp(prepared.kernel_file_name, file_name) == 0 &&
		strcmp(prepared.function_name, names[index]) == 0 &&
		prepared.argument_byte_count == ATTENTION_CORE_ARGUMENT_BYTES;
    }
    return asked;
}

/* A card whose table has room for one kernel more: the first is prepared, the second refused (the backend's line,
 * then the one that names the kernel). The third is not asked. */
static void test_second_kernel_refused(struct attention_core_kernels *kernels)
{
	struct device_card *card = NULL;
	char logged[CAPTURE_CAPACITY];
	char name[FILLER_NAME_CAPACITY];
	uint32_t handle = BACKEND_NO_KERNEL;
	bool filled = true;
	test_host_memory_backend_reset(1, CARD_MEMORY_BYTES);
	test_host_memory_backend_set_arithmetic_class(0,
						      ARITHMETIC_CLASS_SM_80);
	expect(test_host_memory_backend.card_open(0, CARD_MEMORY_BYTES,
						  &card) == STATUS_OK,
	       "a card opens");
	for (uint32_t kernel = 0;
	     kernel + 1u < BACKEND_MAXIMUM_KERNELS && filled; kernel++) {
		const int length =
			snprintf(name, sizeof(name), "filler_%" PRIu32, kernel);
		filled = length > 0 && (size_t)length < sizeof(name) &&
			 test_host_memory_backend.kernel_prepare(
				 card, KERNEL_FILE_ATTENTION_CORE, name,
				 ATTENTION_CORE_ARGUMENT_BYTES,
				 &handle) == STATUS_OK;
	}
    expect(filled, "the card's table is filled to one kernel short");
    capture_begin();
    expect(attention_core_kernels_prepare(&test_host_memory_backend, card,
					  CLASS_80,
					  kernels) == STATUS_UNSUPPORTED,
	   "a table with room for the first kernel alone: its status");
    capture_end(logged, sizeof(logged));
    expect(test_log_is_expected(logged, "takes no more", false) &&
		   test_log_is_expected(
			   logged, KERNEL_ATTENTION_CORE_SCORES " of ", false),
	   "the second kernel is named");
    expect(kernels->attention_core == UNTOUCHED_WORD && kernels->attention_core_scores == UNTOUCHED_WORD &&
               kernels->attention_core_values == UNTOUCHED_WORD,
           "the second kernel refused leaves every handle untouched");
    test_host_memory_backend.card_close(card, 0);
}

static void test_kernels_prepare(void)
{
	static const struct backend_operations backend_without_kernels = {
		.name = "no kernels"
	};
	/* the card's class as the backend describes it, and the class the plan prepares it for (120 native on 120) */
	static const uint32_t classes[][ARITHMETIC_CLASS_PAIR] = {
		{ ARITHMETIC_CLASS_SM_80, ARITHMETIC_CLASS_SM_80 },
		{ ARITHMETIC_CLASS_SM_120, ARITHMETIC_CLASS_SM_120 },
		{ ARITHMETIC_CLASS_SM_120, ARITHMETIC_CLASS_SM_120_NATIVE }
	};
	struct attention_core_kernels kernels;
	struct device_card *card = NULL;
	char logged[CAPTURE_CAPACITY];
	context_text = "the kernels of a card";
	memset(&kernels, UNTOUCHED_BYTE, sizeof(kernels));
	test_host_memory_backend_reset(1, CARD_MEMORY_BYTES);
	expect(test_host_memory_backend.card_open(0, CARD_MEMORY_BYTES,
						  &card) == STATUS_OK,
	       "a card opens");
	capture_begin();
	expect(attention_core_kernels_prepare(NULL, card, CLASS_80, &kernels) ==
			       INVALID &&
		       attention_core_kernels_prepare(&backend_without_kernels,
						      card, CLASS_80,
						      &kernels) == INVALID &&
		       attention_core_kernels_prepare(&test_host_memory_backend,
						      NULL, CLASS_80,
						      &kernels) == INVALID &&
		       attention_core_kernels_prepare(&test_host_memory_backend,
						      card, CLASS_80,
						      NULL) == INVALID,
	       "a missing argument, or a table without kernel_prepare");
	capture_end(logged, sizeof(logged));
	capture_begin();
	expect(attention_core_kernels_prepare(&test_host_memory_backend, card,
					      CLASS_80,
					      &kernels) == STATUS_UNSUPPORTED,
	       "a card without a build of the kernels: its status");
	capture_end(logged, sizeof(logged));
	expect_one_error_line_with(logged, KERNEL_ATTENTION_CORE " of",
				   "the kernel that fails is named");
	expect(kernels.attention_core == UNTOUCHED_WORD &&
		       kernels.attention_core_scores == UNTOUCHED_WORD &&
		       kernels.attention_core_values == UNTOUCHED_WORD &&
		       kernels.attention_core_block == UNTOUCHED_WORD,
	       "a failure leaves the handles untouched");
	test_host_memory_backend.card_close(card, 0);
	test_second_kernel_refused(&kernels);
	for (size_t index = 0; index < COUNT_OF(classes); index++) {
		test_host_memory_backend_reset(1, CARD_MEMORY_BYTES);
		test_host_memory_backend_set_arithmetic_class(
			0, classes[index][0]);
		expect(test_host_memory_backend.card_open(0, CARD_MEMORY_BYTES,
							  &card) == STATUS_OK,
		       "a card opens");
		const char *const file_name =
			classes[index][1] == ARITHMETIC_CLASS_SM_120_NATIVE ?
				KERNEL_FILE_ATTENTION_CORE_NATIVE :
				KERNEL_FILE_ATTENTION_CORE;
		expect(attention_core_kernels_prepare(&test_host_memory_backend,
						      card, classes[index][1],
						      &kernels) == STATUS_OK &&
			       every_kernel_asked(&kernels, file_name) &&
			       kernels.attention_core_block ==
				       BACKEND_NO_KERNEL,
		       "every kernel is asked of a card of each class by its file (120 native: the native build), name and the "
		       "size of its record");
		test_host_memory_backend.card_close(card, 0);
	}
}

int main(void)
{
    error_stream_descriptor = dup(STDERR_FILENO);
    capture_descriptor = mkstemp(capture_path);
    if (error_stream_descriptor < 0 || capture_descriptor < 0) {
        (void)printf("FAILED: the temporary file that stands in for the error stream could not be made\n");
        return EXIT_FAILURE;
    }
    test_exponential_and_rounding();
    test_exact_rows();
    test_weights_apart();
    test_turn_contraction();
    test_model_launches();
    test_most_heads();
    test_member_refusals();
    test_range_refusals();
    test_argument_refusals();
    test_block();
    test_kernels_prepare();
    context_text = "the end of the test";
    expect(close(capture_descriptor) == 0 && unlink(capture_path) == 0 && close(error_stream_descriptor) == 0 &&
               test_host_memory_backend_open_card_count() == 0,
           "the temporary file is removed and no card is left open");
    if (failures != 0) {
        return EXIT_FAILURE;
    }
    (void)printf("attention_core_test passed (%zu counts refused)\n", COUNT_OF(member_refusals));
    return EXIT_SUCCESS;
}
