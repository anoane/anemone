/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * sampler_test: source/sampler.c against sampler.h and sampler_records.h, and the host reference of the sampler
 * (sampler_reference.h) on rows made by hand. No card: the kernel is prepared on test_host_memory_backend, whose cards
 * keep a registry of kernels, and the arena is a number.
 *
 *   - sampler_kernels_prepare: the function of kernel_sampler with the size of its record and the handle the
 *     card gave, the same handle again; refused, each with one error line and the handle untouched: a missing
 *     argument, a table without kernel_prepare, a card with no build of the kernel (its status kept, the function
 *     named), a kernel_prepare that fails (its status kept).
 *   - sampler_prepare_entry, for 1 to 8 rows, vocabularies of 4, 1,000, the model's and the most, 1 and 64
 *     sessions, every offset beyond 32 bits: the launch against one built by hand (kernel, grid, block, record, zeros
 *     behind it), which launch_run of the card takes; ranges of exactly the bytes the launch touches are accepted,
 *     one byte less in any of them, or an arena one byte short, is refused. Refused besides, each with one error
 *     line that holds the words of its rule and the launch untouched: a NULL argument, the kernel not prepared, a
 *     reserved field, rows 0 and 9, vocabularies 0, 1,002 (no multiple of 4) and the most plus 4, no session, an
 *     offset off the alignment, two ranges that share a byte.
 *   - The reference: greedy rows of ties, not-a-numbers, infinities and zeros of both signs; weights at the largest
 *     logit, of a not-a-number and of an infinity, and one whose exponent is l inv - l_max inv as REF:1290-1291
 *     rounds it, not (l - l_max) inv; tau and the kept tokens of weights whose masses are exact; the uniform inside
 *     (0, 1); the race's allowance; the constants of the records against their definitions.
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

#include "anemone/backend_seam.h"
#include "log_expectation.h"
#include "anemone/model_layout.h"
#include "anemone/sampler.h"
#include "anemone/sampler_records.h"
#include "anemone/status.h"
#include "anemone/step_records.h"
#include "host_memory_backend.h"
#include "sampler_reference.h"

#define COUNT_OF(table) (sizeof(table) / sizeof((table)[0]))
#define CARD_MEMORY_BYTES 4096u                 /* of the card in host memory: its arena is never read here */
#define UNTOUCHED_BYTE 0xA5
#define CAPTURE_CAPACITY 4096u
#define FIRST_OFFSET (1ull << 33u)              /* every offset of a description lies beyond 32 bits */
#define LAYOUT_ALIGNMENT 64u
#define MISALIGNMENT 8u                         /* half the alignment of an offset */
#define RANGE_COUNT 5u
#define RECORD_BYTES 48u                        /* by hand: five offsets and two counts */
#define BLOCK_THREADS 1024u
#define TOO_MANY_ROWS 9u
#define SOME_TOKENS 1000u
#define NOT_A_MULTIPLE \
	1002u /* tokens: no multiple of SAMPLER_TOKENS_PER_LOAD_VALUE */
#define ABOVE_THE_MOST             \
	(SAMPLER_MOST_VOCABULARY + \
	 SAMPLER_TOKENS_PER_LOAD_VALUE) /* a multiple */
#define MANY_SESSIONS 64u
#define FLOAT_BYTES 4u
#define STEP_BLOCK_BYTES 16u                    /* by hand, step_records.h: a block and its rows, sessions, results */
#define STEP_ROW_BYTES 16u
#define SESSION_BYTES 16u
#define RESULT_BLOCK_BYTES 16u
#define RESULT_ROW_BYTES 8u
#define TWO_TO_THE_32 4294967296.0
#define TWO_TO_THE_53 9007199254740992.0
#define UNIFORM_BITS 23
#define WORD_BITS 64u
#define TEMPERATURE_FLOOR 1e-5                  /* model.py:1290 */
#define TOP_P_OF_TWO_GROUPS 0.6f                /* of the weights 1, 0.5, 0.5, 0.25: keeps the first three */
#define TOP_P_OF_ONE 0.4f                       /* keeps the first alone */
#define HALF 0.5f
#define TWO 2.0f
#define THREE 3.0f
#define UNIFORM_SAMPLES 65536u
#define SEED 0x5EEDull
#define OTHER_SEED 0x5EEEull
#define POSITION 1234
#define NEAR_SCORE 0.9999995                    /* within TEST_SAMPLER_NEAR of 1 */
#define FAR_SCORE 0.999
#define WINNER 3u
#define OTHER 4u
#define NEITHER 5u
#define GREEDY_ROW_TOKENS 4u
#define BITS_OF_HALF 0x3F000000u
#define BITS_OF_ONE 0x3F800000u
#define TWO_PLACES 2u
#define KNOWN_PAIR 2u                           /* an argument and its known value */
#define THREE_PLACES 3u
#define MEAN_BOUND 0.004                        /* 3.5 standard deviations of a mean of 65,536 uniforms */
#define TWO_TO_THE_MINUS_23 1.1920928955078125e-7
#define SERIES_TOLERANCE 1e-14                  /* relative, of the reference's exponential and logarithm */
#define HALF_WORD_BITS 32u
#define BELOW_THE_FLOOR 1e-7f
#define ABOVE_THE_FLOOR 2e-5f
#define LOGIT_BELOW (-1e-5f)                    /* minus the floor: a weight of about exp(-1) at temperature 0 */
#define COOL_TEMPERATURE 0.6f
#define FORMS_LOGIT 2.5f                        /* at COOL_TEMPERATURE, of FORMS_LARGEST: */
#define FORMS_LARGEST 9.75f
#define BITS_OF_FORMS_WEIGHT 0x36BDAE6Du        /* exp of l inv - l_max inv in float32; of (l - l_max) inv: ..79 */

/* Words of the one error line of each rule. */
#define RULE_MISSING "a missing argument"
#define RULE_NOT_PREPARED "is not prepared"
#define RULE_RESERVED "a reserved field"
#define RULE_COUNTS "1 to 8 rows"
#define RULE_ALIGNMENT "an offset is a multiple of"
#define RULE_ARENA "leaves the arena"
#define RULE_TOUCHED "the launches touch"
#define RULE_SHARED "share a byte"

enum dimension { DIMENSION_X, DIMENSION_Y, DIMENSION_Z };
enum count_field { FIELD_ROWS, FIELD_TOKENS, FIELD_SESSIONS, COUNT_FIELDS };

static int failures = 0;
static const char *context_text = "";
static char capture_path[] = "/tmp/anemone_sampler_test_XXXXXX";
static int capture_descriptor = -1;             /* the temporary file that stands in for the error stream */
static int error_stream_descriptor = -1;        /* a duplicate of descriptor 2, kept to put it back */

static float float_of(uint32_t bits)
{
    float value = 0.0f;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static bool near(double value, double known)
{
    const double difference = value > known ? value - known : known - value;
    return difference <= SERIES_TOLERANCE * (known > 0.0 ? known : -known);
}

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

/* What was logged since capture_begin, as a string in text, and descriptor 2 put back. */
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

/* ---- the kernel of a card ---- */

MUST_CHECK static result_status
failing_kernel_prepare(struct device_card *card, const char *kernel_file_name,
		       const char *function_name, uint32_t argument_byte_count,
		       uint32_t *kernel_out)
{
    (void)card;
    (void)kernel_file_name;
    (void)function_name;
    (void)argument_byte_count;
    (void)kernel_out;
    return STATUS_DEVICE_ERROR;
}

/* Opens card 0 of class 80 and prepares the kernel on it, into kernels_out. */
static struct device_card *
test_kernels_prepare(struct sampler_kernels_record *kernels_out)
{
	static const struct backend_operations without_kernel_prepare = {
		.name = "no kernels"
	};
	struct backend_operations failing_table = test_host_memory_backend;
	struct sampler_kernels_record kernels;
	struct sampler_kernels_record untouched;
	struct sampler_kernels_record again;
	struct test_host_memory_backend_kernel recorded;
	struct device_card *card = NULL;
	char logged[CAPTURE_CAPACITY];
	context_text = "the kernel of a card";
	failing_table.kernel_prepare = failing_kernel_prepare;
	memset(&untouched, UNTOUCHED_BYTE, sizeof(untouched));
	kernels = untouched;
	test_host_memory_backend_reset(1, CARD_MEMORY_BYTES);
	expect(test_host_memory_backend.card_open(0, CARD_MEMORY_BYTES,
						  &card) == STATUS_OK,
	       "a card opens");
	const struct {
		const struct backend_operations *backend;
		struct device_card *card;
		struct sampler_kernels_record *kernels_out;
	} missing[] = { { NULL, card, &kernels },
			{ &without_kernel_prepare, card, &kernels },
			{ &test_host_memory_backend, NULL, &kernels },
			{ &test_host_memory_backend, card, NULL } };
    for (size_t case_index = 0; case_index < COUNT_OF(missing); case_index++) {
        capture_begin();
	expect(sampler_kernels_prepare(missing[case_index].backend,
				       missing[case_index].card,
				       missing[case_index].kernels_out) ==
		       STATUS_INVALID_ARGUMENT,
	       "a missing argument is refused");
	capture_end(logged, sizeof(logged));
        expect_one_error_line_with(logged, RULE_MISSING, "a missing argument");
    }
    /* a card of no class has no build of any kernel: the status comes back as it is */
    capture_begin();
    expect(sampler_kernels_prepare(&test_host_memory_backend, card, &kernels) ==
		   STATUS_UNSUPPORTED,
	   "a card without a build of the kernel");
    capture_end(logged, sizeof(logged));
    expect_one_error_line_with(logged, "token_sampler " RULE_NOT_PREPARED,
			       "the function named");
    test_host_memory_backend_set_arithmetic_class(0, ARITHMETIC_CLASS_SM_80);
    capture_begin();
    expect(sampler_kernels_prepare(&failing_table, card, &kernels) ==
		   STATUS_DEVICE_ERROR,
	   "the status of the kernel_prepare that fails");
    capture_end(logged, sizeof(logged));
    expect_one_error_line_with(logged, "token_sampler " RULE_NOT_PREPARED,
			       "the function that fails is named");
    expect(memcmp(&kernels, &untouched, sizeof(kernels)) == 0, "a refusal or a failure leaves the handle untouched");
    expect(sampler_kernels_prepare(&test_host_memory_backend, card, &kernels) ==
			   STATUS_OK &&
		   sampler_kernels_prepare(&test_host_memory_backend, card,
					   &again) == STATUS_OK &&
		   kernels.sampler == again.sampler,
	   "the kernel is prepared, twice, with one handle");
    expect(test_host_memory_backend_kernel_of(0, kernels.sampler, &recorded) &&
		   strcmp(recorded.kernel_file_name, "kernel_sampler") == 0 &&
		   strcmp(recorded.function_name, "token_sampler") == 0 &&
		   recorded.argument_byte_count == RECORD_BYTES,
	   "the function by its file, its name and the size of its record");
    *kernels_out = kernels;
    return card;
}

/* ---- the launch ---- */

static struct linear_range *range_at(struct sampler_description *description,
				     uint32_t index)
{
	struct linear_range *const ranges[RANGE_COUNT] = {
		&description->step, &description->sessions,
		&description->logits, &description->weights,
		&description->result
	};
	return ranges[index];
}

/* Every range of exactly the bytes the launch touches, one behind the other from FIRST_OFFSET; the end of the last
 * one. */
static uint64_t lay_out(struct sampler_description *description,
			uint32_t row_count, uint32_t vocabulary_size,
			uint32_t session_capacity)
{
    const uint64_t row_bytes = (uint64_t)row_count * vocabulary_size * FLOAT_BYTES;
    const uint64_t counts[RANGE_COUNT] = {STEP_BLOCK_BYTES + (uint64_t)row_count * STEP_ROW_BYTES,
                                          (uint64_t)session_capacity * SESSION_BYTES, row_bytes, row_bytes,
                                          RESULT_BLOCK_BYTES + (uint64_t)row_count * RESULT_ROW_BYTES};
    uint64_t byte_offset = FIRST_OFFSET;
    uint64_t end = FIRST_OFFSET;
    memset(description, 0, sizeof(*description));
    description->row_count = row_count;
    description->vocabulary_size = vocabulary_size;
    description->session_capacity = session_capacity;
    for (uint32_t index = 0; index < RANGE_COUNT; index++) {
	    *range_at(description, index) =
		    (struct linear_range){ byte_offset, counts[index] };
	    end = byte_offset + counts[index];
	    byte_offset = (end + LAYOUT_ALIGNMENT - 1) / LAYOUT_ALIGNMENT *
			  LAYOUT_ALIGNMENT;
    }
    return end;
}

static struct kernel_launch
launch_by_hand(uint32_t kernel, const struct sampler_description *description)
{
    const uint64_t offsets[RANGE_COUNT] = {description->step.byte_offset, description->sessions.byte_offset,
                                           description->logits.byte_offset, description->weights.byte_offset,
                                           description->result.byte_offset};
    const uint32_t counts[] = {description->vocabulary_size, description->session_capacity};
    struct kernel_launch launch;
    memset(&launch, 0, sizeof(launch));
    launch.kernel = kernel;
    launch.grid[DIMENSION_X] = 1;
    launch.grid[DIMENSION_Y] = description->row_count;
    launch.grid[DIMENSION_Z] = 1;
    launch.block[DIMENSION_X] = BLOCK_THREADS;
    launch.block[DIMENSION_Y] = 1;
    launch.block[DIMENSION_Z] = 1;
    launch.argument_byte_count = RECORD_BYTES;
    memcpy(launch.argument, offsets, sizeof(offsets));
    memcpy(launch.argument + sizeof(offsets), counts, sizeof(counts));
    return launch;
}

/* Prepares with a description, expecting a refusal by the rule of these words and the launch untouched. */
static void expect_refused(const struct sampler_description *description,
			   uint64_t arena_byte_count,
			   const struct sampler_kernels_record *kernels,
			   const char *words, const char *what)
{
	struct kernel_launch launch;
	struct kernel_launch untouched;
	char logged[CAPTURE_CAPACITY];
	memset(&launch, UNTOUCHED_BYTE, sizeof(launch));
	memset(&untouched, UNTOUCHED_BYTE, sizeof(untouched));
	capture_begin();
	const result_status status = sampler_prepare_entry(
		description, arena_byte_count, kernels, &launch);
	capture_end(logged, sizeof(logged));
	expect(status == STATUS_INVALID_ARGUMENT, what);
	expect(memcmp(&launch, &untouched, sizeof(launch)) == 0,
	       "a refusal leaves the launch untouched");
	expect_one_error_line_with(logged, words, what);
}

static void test_launches(struct device_card *card,
			  const struct sampler_kernels_record *kernels)
{
	static const uint32_t vocabularies[] = { SAMPLER_TOKENS_PER_LOAD_VALUE,
						 SOME_TOKENS,
						 MODEL_VOCABULARY_VALUE,
						 SAMPLER_MOST_VOCABULARY };
	static const uint32_t session_capacities[] = { 1, MANY_SESSIONS };
	struct sampler_description description;
	struct kernel_launch launch;
	for (uint32_t row_count = 1; row_count <= LINEAR_MAXIMUM_ROWS;
	     row_count++) {
		for (size_t vocabulary_index = 0;
		     vocabulary_index < COUNT_OF(vocabularies);
		     vocabulary_index++) {
			for (size_t sessions_index = 0;
			     sessions_index < COUNT_OF(session_capacities);
			     sessions_index++) {
				context_text = "a description of exact ranges";
				const uint64_t arena_byte_count = lay_out(
					&description, row_count,
					vocabularies[vocabulary_index],
					session_capacities[sessions_index]);
				const struct kernel_launch expected =
					launch_by_hand(kernels->sampler,
						       &description);
				memset(&launch, UNTOUCHED_BYTE, sizeof(launch));
				expect(sampler_prepare_entry(
					       &description, arena_byte_count,
					       kernels, &launch) == STATUS_OK &&
					       memcmp(&launch, &expected,
						      sizeof(launch)) == 0,
				       "the launch is the one the records header gives, every byte");
				expect(test_host_memory_backend.launch_run(
					       card, &launch, 1) == STATUS_OK,
				       "the card that prepared the kernel takes the launch");
				expect_refused(&description,
					       arena_byte_count - 1, kernels,
					       RULE_ARENA,
					       "an arena one byte short");
				for (uint32_t index = 0; index < RANGE_COUNT;
				     index++) {
					context_text = "a range one byte short";
					range_at(&description, index)
						->byte_count--;
					expect_refused(
						&description, arena_byte_count,
						kernels, RULE_TOUCHED,
						"a range one byte short");
					range_at(&description, index)
						->byte_count++;
					context_text =
						"an offset off the alignment";
					range_at(&description, index)
						->byte_offset += MISALIGNMENT;
					expect_refused(
						&description,
						arena_byte_count + MISALIGNMENT,
						kernels, RULE_ALIGNMENT,
						"an offset off the alignment");
					range_at(&description, index)
						->byte_offset -= MISALIGNMENT;
				}
			}
		}
	}
}

static void test_refusals(const struct sampler_kernels_record *kernels)
{
    static const uint32_t refused_counts[][COUNT_FIELDS] = {{0, SOME_TOKENS, 1},
                                                 {TOO_MANY_ROWS, SOME_TOKENS, 1},
                                                 {1, 0, 1},
                                                 {1, NOT_A_MULTIPLE, 1},
                                                 {1, ABOVE_THE_MOST, 1},
                                                 {1, SOME_TOKENS, 0}};
    struct sampler_description description;
    const struct sampler_kernels_record not_prepared = { BACKEND_NO_KERNEL };
    char logged[CAPTURE_CAPACITY];
    uint64_t arena_byte_count = lay_out(&description, LINEAR_MAXIMUM_ROWS,
					SOME_TOKENS, MANY_SESSIONS);
    context_text = "a refusal";
    expect_refused(NULL, arena_byte_count, kernels, RULE_MISSING, "no description");
    expect_refused(&description, arena_byte_count, NULL, RULE_MISSING, "no kernels");
    capture_begin();
    expect(sampler_prepare_entry(&description, arena_byte_count, kernels,
				 NULL) == STATUS_INVALID_ARGUMENT,
	   "nowhere to put the launch");
    capture_end(logged, sizeof(logged));
    expect_one_error_line_with(logged, RULE_MISSING, "nowhere to put the launch");
    expect_refused(&description, arena_byte_count, &not_prepared,
		   "token_sampler " RULE_NOT_PREPARED,
		   "the kernel not prepared");
    description.reserved_zero = 1;
    expect_refused(&description, arena_byte_count, kernels, RULE_RESERVED, "a reserved field");
    for (size_t case_index = 0; case_index < COUNT_OF(refused_counts); case_index++) {
        arena_byte_count = lay_out(&description, 1, SOME_TOKENS, 1);
        description.row_count = refused_counts[case_index][FIELD_ROWS];
        description.vocabulary_size = refused_counts[case_index][FIELD_TOKENS];
        description.session_capacity = refused_counts[case_index][FIELD_SESSIONS];
        expect_refused(&description, arena_byte_count, kernels, RULE_COUNTS, "rows, tokens or sessions out of range");
    }
    for (uint32_t first = 0; first < RANGE_COUNT; first++) {
        for (uint32_t second = 0; second < RANGE_COUNT; second++) {
            /* room behind the last range for any range moved onto it */
	    arena_byte_count = lay_out(&description, LINEAR_MAXIMUM_ROWS,
				       SOME_TOKENS, MANY_SESSIONS) +
			       description.logits.byte_count;
	    if (first != second) {
                range_at(&description, second)->byte_offset = range_at(&description, first)->byte_offset;
                expect_refused(&description, arena_byte_count, kernels, RULE_SHARED, "two ranges share a byte");
            }
        }
    }
}

/* ---- the reference ---- */

static void test_reference(void)
{
    static const float greedy_rows[][GREEDY_ROW_TOKENS] = {{1.0f, 3.0f, NAN, 3.0f},
                                                           {NAN, NAN, NAN, NAN},
                                                           {-INFINITY, -INFINITY, -INFINITY, -INFINITY},
                                                           {-0.0f, 0.0f, -1.0f, -1.0f},
                                                           {NAN, 1.0f, INFINITY, INFINITY}};
    static const uint32_t greedy_tokens[] = {1, 0, 0, 0, 2};
    static const float weights[] = {1.0f, 0.5f, 0.5f, 0.25f};
    static const uint32_t three_largest[] = {0, 1, 2};
    uint32_t kept[COUNT_OF(weights)];
    uint32_t threshold = 0;
    context_text = "the reference";
    for (size_t row = 0; row < COUNT_OF(greedy_rows); row++) {
        expect(test_sampler_greedy(greedy_rows[row], COUNT_OF(greedy_rows[row])) == greedy_tokens[row],
               "greedy: the first largest logit, never a not-a-number, token 0 of a row of no number");
    }
    expect(test_sampler_weight(THREE, THREE, 1.0f) == 1.0f && test_sampler_weight(-INFINITY, THREE, 1.0f) == 0.0f &&
               test_sampler_weight(NAN, THREE, 1.0f) == 0.0f,
           "a weight: 1 at the largest logit, 0 of an infinity below it and of a not-a-number");
    expect(test_sampler_weight(LOGIT_BELOW, 0.0f, 0.0f) ==
			   test_sampler_weight(LOGIT_BELOW, 0.0f,
					       BELOW_THE_FLOOR) &&
		   test_sampler_weight(LOGIT_BELOW, 0.0f, 0.0f) ==
			   test_sampler_weight(LOGIT_BELOW, 0.0f,
					       SAMPLER_TEMPERATURE_FLOOR) &&
		   test_sampler_weight(LOGIT_BELOW, 0.0f, 0.0f) !=
			   test_sampler_weight(LOGIT_BELOW, 0.0f,
					       ABOVE_THE_FLOOR),
	   "a temperature below the floor is the floor");
    expect(test_sampler_weight(FORMS_LOGIT, FORMS_LARGEST, COOL_TEMPERATURE) == float_of(BITS_OF_FORMS_WEIGHT),
           "a weight's exponent is x_t - x_max, each x rounded to float32 as REF:1290 rounds it");
    expect(test_sampler_within_places(1.0f, float_of(BITS_OF_ONE + TWO_PLACES), TEST_SAMPLER_WEIGHT_PLACES) &&
               !test_sampler_within_places(1.0f, float_of(BITS_OF_ONE + THREE_PLACES), TEST_SAMPLER_WEIGHT_PLACES) &&
               test_sampler_within_places(0.0f, -0.0f, 0) && !test_sampler_within_places(HALF, -HALF, 1),
           "places of float32 are counted within one sign");
    expect(test_sampler_threshold(weights, COUNT_OF(weights), TOP_P_OF_TWO_GROUPS, &threshold) &&
               threshold == BITS_OF_HALF &&
               test_sampler_kept(weights, COUNT_OF(weights), threshold, kept) == COUNT_OF(three_largest) &&
               memcmp(kept, three_largest, sizeof(three_largest)) == 0,
           "top_p 0.6 of 1, 0.5, 0.5, 0.25 keeps the three largest: equal weights together");
    expect(test_sampler_threshold(weights, COUNT_OF(weights), TOP_P_OF_ONE, &threshold) && threshold == BITS_OF_ONE &&
               test_sampler_threshold(weights, COUNT_OF(weights), 1.0f, &threshold) && threshold == 0,
           "top_p 0.4 keeps the largest alone, top_p 1 everything");
    static const double exponentials[][KNOWN_PAIR] = {{0.0, 1.0},
                                             {-0.5, 0.6065306597126334},
                                             {-1.0, 0.36787944117144233},
                                             {-30.0, 9.357622968840175e-14},
                                             {-700.0, 9.85967654375977e-305}};
    static const double logarithms[][KNOWN_PAIR] = {{0.5, -0.6931471805599453},
                                           {0.75, -0.2876820724517809},
                                           {1e-7, -16.11809565095832},
                                           {0.9999999403953552, -5.960464655174753e-08}};
    for (size_t index = 0; index < COUNT_OF(exponentials); index++) {
        expect(near(test_sampler_exponential(exponentials[index][0]), exponentials[index][1]),
               "the reference's exponential at a known value");
    }
    for (size_t index = 0; index < COUNT_OF(logarithms); index++) {
        expect(near(test_sampler_logarithm(logarithms[index][0]), logarithms[index][1]),
               "the reference's logarithm at a known value");
    }
    expect(test_sampler_exponential(-INFINITY) == 0.0, "the exponential of -infinity is 0");
    bool inside = true;
    double sum = 0.0;
    for (uint32_t token = 0; token < UNIFORM_SAMPLES; token++) {
        const double uniform = test_sampler_uniform(SEED, POSITION, token);
        inside = inside && uniform > 0.0 && uniform < 1.0 && uniform == (double)(float)uniform;
        sum += uniform;
    }
    expect(inside && sum / UNIFORM_SAMPLES > (double)HALF - MEAN_BOUND &&
               sum / UNIFORM_SAMPLES < (double)HALF + MEAN_BOUND &&
               test_sampler_uniform(SEED, POSITION, 0) != test_sampler_uniform(OTHER_SEED, POSITION, 0) &&
               test_sampler_uniform(SEED, POSITION, 0) != test_sampler_uniform(SEED, POSITION + 1, 0),
           "the uniform: inside (0, 1), a float32, of mean 1/2, another for another seed or position");
    const struct test_sampler_race close_race = {1.0, NEAR_SCORE, WINNER, OTHER};
    const struct test_sampler_race clear_race = {1.0, FAR_SCORE, WINNER, OTHER};
    expect(test_sampler_race_allows(&close_race, WINNER) && test_sampler_race_allows(&close_race, OTHER) &&
               test_sampler_race_allows(&clear_race, WINNER) && !test_sampler_race_allows(&clear_race, OTHER) &&
               !test_sampler_race_allows(&close_race, NEITHER),
           "the race allows the winner, and the other within TEST_SAMPLER_NEAR");
    context_text = "the constants of the records";
    expect(sizeof(struct sampler_arguments) == RECORD_BYTES &&
		   SAMPLER_THREADS == BLOCK_THREADS &&
		   STEP_BLOCK_BYTES_VALUE == STEP_BLOCK_BYTES &&
		   STEP_ROW_BYTES_VALUE == STEP_ROW_BYTES &&
		   STEP_SESSION_BYTES == SESSION_BYTES &&
		   STEP_RESULT_BYTES == RESULT_BLOCK_BYTES &&
		   STEP_ROW_RESULT_BYTES == RESULT_ROW_BYTES,
	   "the record, the block and the areas of step_records.h");
    expect((double)SAMPLER_MASS_SCALE == TWO_TO_THE_32 &&
		   (double)SAMPLER_MOST_VOCABULARY * TWO_TO_THE_32 <=
			   TWO_TO_THE_53 &&
		   MODEL_VOCABULARY_VALUE <= SAMPLER_MOST_VOCABULARY &&
		   (double)SAMPLER_UNIFORM_SCALE == TWO_TO_THE_MINUS_23 &&
		   SAMPLER_UNIFORM_SHIFT == WORD_BITS - UNIFORM_BITS &&
		   SAMPLER_POSITION_SHIFT == HALF_WORD_BITS &&
		   SAMPLER_TEMPERATURE_FLOOR == (float)TEMPERATURE_FLOOR,
	   "the mass of a row is exact in double, the uniform has 23 bits, the floor is the reference's");
}

int main(void)
{
	struct sampler_kernels_record kernels;
	error_stream_descriptor = dup(STDERR_FILENO);
	capture_descriptor = mkstemp(capture_path);
	if (error_stream_descriptor < 0 || capture_descriptor < 0) {
		(void)printf(
			"FAILED: the temporary file that stands in for the error stream could not be made\n");
		return EXIT_FAILURE;
	}
	struct device_card *const card = test_kernels_prepare(&kernels);
	test_launches(card, &kernels);
	test_refusals(&kernels);
	test_reference();
	test_host_memory_backend.card_close(card, 0);
	context_text = "the end of the test";
	expect(close(capture_descriptor) == 0 && unlink(capture_path) == 0 &&
		       close(error_stream_descriptor) == 0 &&
		       test_host_memory_backend_open_card_count() == 0,
	       "the temporary file is removed and no card is left open");
	if (failures != 0) {
		return EXIT_FAILURE;
	}
    (void)printf("sampler_test passed\n");
    return EXIT_SUCCESS;
}
