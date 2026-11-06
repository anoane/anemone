/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * engram_rows_test: source/engram_rows.c against engram_rows.h and engram_rows_records.h. No card: the kernels are
 * prepared on test_host_memory_backend, whose cards keep a registry of them, and the arena is a number.
 *
 *   - engram_rows_kernels_prepare: both functions of kernel_engram_rows with the sizes of their records and
 *     the handles the card gave, the same handles again on a second call; refused, each with one error line and the
 *     handles untouched: a missing argument, a table without kernel_prepare, a card with no build of the kernels
 *     (its status kept, the first function named), the second function failing (named).
 *   - engram_rows_prepare, for 1 to 8 rows and a chunk's (MODEL_WIDE_MOST_ROWS, a wide step's) and 1,
 *     24 and 64 table rows a token, every offset beyond 32 bits: the two launches against launches built by hand from
 *     the records' layout (kernel, grid, block, shared memory, record, zeros behind it), which launch_run of the card
 *     that prepared the kernels takes; ranges of exactly the bytes the launches touch are accepted, one byte less in
 *     any of them, or an arena one byte short, is refused. Refused besides, each alone, with one error line that holds
 *     the words of its rule and the launches left untouched: a NULL argument, a kernel not prepared, rows_per_token 0
 *     and 65, a row_count 0 and 4,097, an offset off the alignment, two ranges that share a byte.
 *   - The constants of the records against the model's numbers, engram.h's rows and the reference's float32
 *     epsilon.
 *   - The table rows of the card test's values case: each holds every byte once, and 8 rows of
 *     TEST_ENGRAM_ROWS_PAIR_TABLE_ROWS hold each pair of a byte and a scale code exactly once.
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
#include "anemone/engram.h"
#include "anemone/engram_rows.h"
#include "anemone/engram_rows_records.h"
#include "anemone/model_layout.h"
#include "anemone/status.h"
#include "engram_rows_reference.h"
#include "host_memory_backend.h"
#include "linear_reference.h"
#include "weight_format_reference.h"

#define COUNT_OF(table) (sizeof(table) / sizeof((table)[0]))
#define CARD_MEMORY_BYTES 4096u                 /* of the card in host memory: its arena is never read here */
#define UNTOUCHED_BYTE 0xA5
#define CAPTURE_CAPACITY 4096u
#define FIRST_OFFSET (1ull << 33u)              /* every offset of a description lies beyond 32 bits */
#define LAYOUT_ALIGNMENT 64u
#define MISALIGNMENT 8u                         /* half the alignment of an offset */
#define MODEL_TABLE_ROWS 24u                    /* by hand: (4 - 1) n-gram sizes of 8 heads */
#define MOST_TABLE_ROWS 64u                     /* engram.h's limit */
#define TABLE_ROW_BYTES 264u                    /* by hand: 256 codes and 8 scale codes */
#define VALUES_BYTES_PER_TABLE_ROW 512u         /* 256 bfloat16 */
#define PROJECTION_ROW_BYTES 51200u             /* 25,600 bfloat16: four keys and a value of 5,120 */
#define WEIGHT_BYTES 40960u                     /* 4 by 5,120 bfloat16 */
#define STREAMS_ROW_BYTES 40960u
#define VALUES_BLOCK 256u
#define GATE_BLOCK 512u
#define STREAM_COUNT 4u
#define VALUES_RECORD_BYTES 16u
#define GATE_RECORD_BYTES 32u
#define TABLE_ROW_COUNTS 3u
#define TOO_MANY_ROWS (MODEL_WIDE_MOST_ROWS + 1u)
#define RANGE_COUNT 6u
#define REFERENCE_EPSILON 1e-20                 /* model.py:346 norm_eps, a double the reference turns to float32 */
#define FAILING_CALL_COUNT 2u
#define ROOM_FOR_A_MOVED_RANGE (1ull << 20u)    /* more than the largest range: a range moved onto another fits */
#define HIDDEN MODEL_HIDDEN
#define STREAM_VALUES 20480u                    /* of a row: 4 streams of 5,120 */
#define CASES TEST_ENGRAM_ROWS_GATE_CASE_COUNT
#define WARPS (GATE_BLOCK / LANES)
#define LANES 32u
#define CODE_COUNT 256u
#define CHANGED_PLACE 3u
#define LEAST_CENTRAL_SHARE 0.9                 /* of an emulation within a last place of the card's functions */
#define PERCENT 100.0
#define CLAMP 1.0e-6                            /* model.py:341 */
#define BFLOAT16_NOT_A_NUMBER 0x7FC0u
#define BFLOAT16_INFINITY 0x7F80u
#define BFLOAT16_ONE 0x3F80u
#define BFLOAT16_448 0x43E0u
#define BFLOAT16_MINUS_ZERO 0x8000u
#define E4M3_ONE 0x38u
#define E4M3_448 0x7Eu
#define E4M3_SMALLEST 0x01u                     /* 2^-9: under a scale of 2^-127 below every bfloat16 */
#define E4M3_MINUS_ZERO 0x80u
#define E4M3_NOT_A_NUMBER 0x7Fu
#define E8M0_ONE 127u
#define E8M0_LARGEST 254u
#define E8M0_NOT_A_NUMBER 255u

/* Words of the one error line of each rule. */
#define RULE_MISSING "a missing argument"
#define RULE_NOT_PREPARED "is not prepared"
#define RULE_COUNTS "a launch takes 1 to"
#define RULE_ALIGNMENT "an offset is a multiple of"
#define RULE_ARENA "leaves the arena"
#define RULE_TOUCHED "the launches touch"
#define RULE_SHARED "share a byte"

enum dimension { DIMENSION_X, DIMENSION_Y, DIMENSION_Z };
enum gate_sum { STREAM_SQUARES, KEY_SQUARES, DOT, SUM_COUNT };
/* What the emulation of the gate kernel changes, to show the bound reporting it. */
enum emulation_fault { FAULT_NONE, FAULT_NO_EPSILON, FAULT_NO_CLAMP, FAULT_FIRST_THREAD_TWICE };

static int failures = 0;
static const char *context_text = "";
static char capture_path[] = "/tmp/anemone_engram_rows_test_XXXXXX";
static int capture_descriptor = -1;             /* the temporary file that stands in for the error stream */
static int error_stream_descriptor = -1;        /* a duplicate of descriptor 2, kept to put it back */
static const uint32_t table_row_counts[TABLE_ROW_COUNTS] = {1, MODEL_TABLE_ROWS, MOST_TABLE_ROWS};
static uint16_t case_streams[CASES * STREAM_VALUES];
static uint16_t case_projection[CASES * ENGRAM_ROWS_PROJECTION];
static uint16_t case_query_weight[STREAM_VALUES];
static uint16_t case_key_weight[STREAM_VALUES];
static uint16_t case_outputs[CASES * STREAM_VALUES];
static uint8_t pair_rows[LINEAR_MAXIMUM_ROWS *
			 TEST_ENGRAM_ROWS_PAIR_TABLE_ROWS * TABLE_ROW_BYTES];
static uint32_t pair_counts[CODE_COUNT][CODE_COUNT];    /* [scale code][byte] */

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

/* ---- the kernels of a card ---- */

static uint32_t calls_before_failure = 0;       /* of the failing table: 0, no call fails */
static uint32_t call_count = 0;

/* kernel_prepare of a table whose call number calls_before_failure + 1 fails; the others pass to the card. */
MUST_CHECK static result_status
failing_kernel_prepare(struct device_card *card, const char *kernel_file_name,
		       const char *function_name, uint32_t argument_byte_count,
		       uint32_t *kernel_out)
{
    call_count++;
    if (call_count == calls_before_failure + 1) {
	    return STATUS_DEVICE_ERROR;
    }
    return test_host_memory_backend.kernel_prepare(card, kernel_file_name, function_name, argument_byte_count,
                                                   kernel_out);
}

/* The kernel of a handle on card 0 is the function of that name with a record of that size. */
static bool kernel_is(uint32_t handle, const char *function_name, uint32_t argument_byte_count)
{
    struct test_host_memory_backend_kernel kernel;
    return test_host_memory_backend_kernel_of(0, handle, &kernel) &&
           strcmp(kernel.kernel_file_name, "kernel_engram_rows") == 0 &&
           strcmp(kernel.function_name, function_name) == 0 && kernel.argument_byte_count == argument_byte_count;
}

/* Opens card 0 of class 80 and prepares the kernels on it, into kernels_out. */
static struct device_card *
test_kernels_prepare(struct engram_rows_kernels *kernels_out)
{
	static const struct backend_operations without_kernel_prepare = {
		.name = "no kernels"
	};
	struct backend_operations failing_table = test_host_memory_backend;
	struct engram_rows_kernels kernels;
	struct engram_rows_kernels untouched;
	struct engram_rows_kernels again;
	struct device_card *card = NULL;
	char logged[CAPTURE_CAPACITY];
	context_text = "the kernels of a card";
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
		struct engram_rows_kernels *kernels_out;
	} missing[] = { { NULL, card, &kernels },
			{ &without_kernel_prepare, card, &kernels },
			{ &test_host_memory_backend, NULL, &kernels },
			{ &test_host_memory_backend, card, NULL } };
    for (size_t case_index = 0; case_index < COUNT_OF(missing); case_index++) {
        capture_begin();
	expect(engram_rows_kernels_prepare(missing[case_index].backend,
					   missing[case_index].card,
					   missing[case_index].kernels_out) ==
		       STATUS_INVALID_ARGUMENT,
	       "a missing argument is refused");
	capture_end(logged, sizeof(logged));
        expect_one_error_line_with(logged, RULE_MISSING, "a missing argument");
    }
    /* a card of no class has no build of any kernel: the status comes back as it is */
    capture_begin();
    expect(engram_rows_kernels_prepare(&test_host_memory_backend, card,
				       &kernels) == STATUS_UNSUPPORTED,
	   "a card without a build of the kernels");
    capture_end(logged, sizeof(logged));
    expect_one_error_line_with(logged, KERNEL_ENGRAM_VALUES " is not prepared",
			       "the first function named");
    test_host_memory_backend_set_arithmetic_class(0, ARITHMETIC_CLASS_SM_80);
    for (uint32_t failing = 0; failing < FAILING_CALL_COUNT; failing++) {
        calls_before_failure = failing;
        call_count = 0;
        capture_begin();
	expect(engram_rows_kernels_prepare(&failing_table, card, &kernels) ==
		       STATUS_DEVICE_ERROR,
	       "the status of the kernel_prepare that fails");
	capture_end(logged, sizeof(logged));
	expect_one_error_line_with(logged,
				   failing == 0 ? KERNEL_ENGRAM_VALUES :
						  KERNEL_ENGRAM_GATE,
				   "the function that fails is named");
	expect(call_count == failing + 1, "the first failure ends the preparation");
    }
    expect(memcmp(&kernels, &untouched, sizeof(kernels)) == 0, "a refusal or a failure leaves the handles untouched");
    expect(engram_rows_kernels_prepare(&test_host_memory_backend, card,
				       &kernels) == STATUS_OK &&
		   engram_rows_kernels_prepare(&test_host_memory_backend, card,
					       &again) == STATUS_OK,
	   "both kernels are prepared, twice");
    expect(kernel_is(kernels.values, "engram_values_entry",
		     VALUES_RECORD_BYTES) &&
		   kernel_is(kernels.gate, "engram_gate_entry",
			     GATE_RECORD_BYTES) &&
		   kernels.values != kernels.gate &&
		   memcmp(&kernels, &again, sizeof(kernels)) == 0,
	   "each function by its file, its name and the size of its record, and the same handles again");
    *kernels_out = kernels;
    return card;
}

/* ---- the launches ---- */

/* The ranges of a description in one order, with the bytes the launches touch in each. */
static struct linear_range *
range_at(struct engram_rows_description *description, uint32_t index)
{
	struct linear_range *const ranges[RANGE_COUNT] = {
		&description->rows,	  &description->values,
		&description->projection, &description->query_weight,
		&description->key_weight, &description->streams
	};
	return ranges[index];
}
static uint64_t touched_byte_count(uint32_t index, uint32_t table_rows, uint32_t row_count)
{
    const uint64_t counts[RANGE_COUNT] = {
        (uint64_t)row_count * table_rows * TABLE_ROW_BYTES,
        (uint64_t)row_count * table_rows * VALUES_BYTES_PER_TABLE_ROW,
        (uint64_t)row_count * PROJECTION_ROW_BYTES,
        WEIGHT_BYTES,
        WEIGHT_BYTES,
        (uint64_t)row_count * STREAMS_ROW_BYTES};
    return counts[index];
}

/* Every range of exactly the bytes touched, one behind the other from FIRST_OFFSET; the end of the last one. */
static uint64_t lay_out(struct engram_rows_description *description,
			uint32_t table_rows, uint32_t row_count)
{
    uint64_t byte_offset = FIRST_OFFSET;
    memset(description, 0, sizeof(*description));
    description->rows_per_token = table_rows;
    description->row_count = row_count;
    for (uint32_t index = 0; index < RANGE_COUNT; index++) {
        const uint64_t byte_count = touched_byte_count(index, table_rows, row_count);
	*range_at(description, index) =
		(struct linear_range){ byte_offset, byte_count };
	byte_offset = (byte_offset + byte_count + LAYOUT_ALIGNMENT - 1) / LAYOUT_ALIGNMENT * LAYOUT_ALIGNMENT;
    }
    return byte_offset;
}

/* A launch as the records header says, built by hand. */
static struct kernel_launch launch_by_hand(uint32_t kernel, uint32_t grid_x,
					   uint32_t row_count, uint32_t block_x,
					   const uint64_t *record_words,
					   uint32_t record_byte_count)
{
	struct kernel_launch launch;
	memset(&launch, 0, sizeof(launch));
	launch.kernel = kernel;
	launch.grid[DIMENSION_X] = grid_x;
	launch.grid[DIMENSION_Y] = row_count;
	launch.grid[DIMENSION_Z] = 1;
	launch.block[DIMENSION_X] = block_x;
	launch.block[DIMENSION_Y] = 1;
	launch.block[DIMENSION_Z] = 1;
	launch.argument_byte_count = record_byte_count;
	memcpy(launch.argument, record_words, record_byte_count);
	return launch;
}

/* Prepares with a description, expecting a refusal by the rule of these words and the launches untouched. */
static void expect_refused(const struct engram_rows_description *description,
			   uint64_t arena_byte_count,
			   const struct engram_rows_kernels *kernels,
			   const char *words, const char *what)
{
	struct kernel_launch launches[ENGRAM_ROWS_LAUNCH_COUNT];
	struct kernel_launch untouched[ENGRAM_ROWS_LAUNCH_COUNT];
	char logged[CAPTURE_CAPACITY];
	memset(launches, UNTOUCHED_BYTE, sizeof(launches));
	memset(untouched, UNTOUCHED_BYTE, sizeof(untouched));
	capture_begin();
	const result_status status = engram_rows_prepare(
		description, arena_byte_count, kernels, launches);
	capture_end(logged, sizeof(logged));
	expect(status == STATUS_INVALID_ARGUMENT, what);
	expect(memcmp(launches, untouched, sizeof(launches)) == 0,
	       "a refusal leaves the launches untouched");
	expect_one_error_line_with(logged, words, what);
}

static void test_launches(struct device_card *card,
			  const struct engram_rows_kernels *kernels)
{
	struct engram_rows_description description;
	struct kernel_launch launches[ENGRAM_ROWS_LAUNCH_COUNT];
	for (uint32_t count_index = 0; count_index < TABLE_ROW_COUNTS;
	     count_index++) {
		const uint32_t table_rows = table_row_counts[count_index];
		for (uint32_t step = 1; step <= LINEAR_MAXIMUM_ROWS + 1u;
		     step++) {
			/* 1 to 8 rows, then a chunk's */
			const uint32_t row_count = step <= LINEAR_MAXIMUM_ROWS ?
							   step :
							   MODEL_WIDE_MOST_ROWS;
			context_text = "a description of exact ranges";
			const uint64_t arena_byte_count =
				lay_out(&description, table_rows, row_count);
			const uint64_t values_record[] = {
				description.rows.byte_offset,
				description.values.byte_offset
			};
			const uint64_t gate_record[] = {
				description.projection.byte_offset,
				description.query_weight.byte_offset,
				description.key_weight.byte_offset,
				description.streams.byte_offset
			};
			const struct kernel_launch
				expected[ENGRAM_ROWS_LAUNCH_COUNT] = {
					launch_by_hand(kernels->values,
						       table_rows, row_count,
						       VALUES_BLOCK,
						       values_record,
						       VALUES_RECORD_BYTES),
					launch_by_hand(kernels->gate,
						       STREAM_COUNT, row_count,
						       GATE_BLOCK, gate_record,
						       GATE_RECORD_BYTES)
				};
			memset(launches, UNTOUCHED_BYTE, sizeof(launches));
			expect(engram_rows_prepare(&description,
						   arena_byte_count, kernels,
						   launches) == STATUS_OK &&
				       memcmp(launches, expected,
					      sizeof(launches)) == 0,
			       "the two launches are those the records header gives, every byte");
			expect(test_host_memory_backend.launch_run(
				       card, launches,
				       ENGRAM_ROWS_LAUNCH_COUNT) == STATUS_OK,
			       "the card that prepared the kernels takes the launches");
			expect_refused(&description, arena_byte_count - 1,
				       kernels, RULE_ARENA,
				       "an arena one byte short");
			for (uint32_t index = 0; index < RANGE_COUNT; index++) {
				context_text = "a range one byte short";
				range_at(&description, index)->byte_count--;
				expect_refused(&description, arena_byte_count,
					       kernels, RULE_TOUCHED,
					       "a range one byte short");
				range_at(&description, index)->byte_count++;
				context_text = "an offset off the alignment";
				range_at(&description, index)->byte_offset +=
					MISALIGNMENT;
				expect_refused(&description,
					       arena_byte_count + MISALIGNMENT,
					       kernels, RULE_ALIGNMENT,
					       "an offset off the alignment");
				range_at(&description, index)->byte_offset -=
					MISALIGNMENT;
			}
		}
    }
}

static void test_refusals(const struct engram_rows_kernels *kernels)
{
    static const struct {
        uint32_t table_rows;
        uint32_t row_count;
    } refused_counts[] = {{0, 1}, {MOST_TABLE_ROWS + 1, 1}, {1, 0}, {1, TOO_MANY_ROWS}};
    struct engram_rows_description description;
    const struct engram_rows_kernels no_values = { BACKEND_NO_KERNEL,
						   kernels->gate };
    const struct engram_rows_kernels no_gate = { kernels->values,
						 BACKEND_NO_KERNEL };
    char logged[CAPTURE_CAPACITY];
    const uint64_t arena_byte_count =
	    lay_out(&description, MODEL_TABLE_ROWS, LINEAR_MAXIMUM_ROWS);
    context_text = "a refusal";
    expect_refused(NULL, arena_byte_count, kernels, RULE_MISSING, "no description");
    expect_refused(&description, arena_byte_count, NULL, RULE_MISSING, "no kernels");
    capture_begin();
    expect(engram_rows_prepare(&description, arena_byte_count, kernels, NULL) ==
		   STATUS_INVALID_ARGUMENT,
	   "nowhere to put the launches");
    capture_end(logged, sizeof(logged));
    expect_one_error_line_with(logged, RULE_MISSING, "nowhere to put the launches");
    expect_refused(&description, arena_byte_count, &no_values,
		   KERNEL_ENGRAM_VALUES " " RULE_NOT_PREPARED,
		   "the values kernel not prepared");
    expect_refused(&description, arena_byte_count, &no_gate,
		   KERNEL_ENGRAM_GATE " " RULE_NOT_PREPARED,
		   "the gate kernel not prepared");
    for (size_t case_index = 0; case_index < COUNT_OF(refused_counts); case_index++) {
        description.rows_per_token = refused_counts[case_index].table_rows;
        description.row_count = refused_counts[case_index].row_count;
        expect_refused(&description, arena_byte_count, kernels, RULE_COUNTS, "table rows or rows out of their range");
    }
    for (uint32_t first = 0; first < RANGE_COUNT; first++) {
        for (uint32_t second = 0; second < RANGE_COUNT; second++) {
		(void)lay_out(&description, MODEL_TABLE_ROWS,
			      LINEAR_MAXIMUM_ROWS);
		if (first != second) {
			range_at(&description, second)->byte_offset =
				range_at(&description, first)->byte_offset;
			expect_refused(&description,
				       arena_byte_count +
					       ROOM_FOR_A_MOVED_RANGE,
				       kernels, RULE_SHARED,
				       "two ranges share a byte");
		}
        }
    }
}

static void test_constants(void)
{
    context_text = "the constants of the records";
    expect(ENGRAM_ROWS_VALUES == ENGRAM_VALUE_BYTES_VALUE &&
		   ENGRAM_ROWS_ROW_BYTES == ENGRAM_ROW_BYTES &&
		   ENGRAM_ROW_BYTES == TABLE_ROW_BYTES &&
		   ENGRAM_MAXIMUM_ROWS_PER_TOKEN == MOST_TABLE_ROWS,
	   "the rows the kernel reads are engram.h's");
    expect(ENGRAM_ROWS_PROJECTION * sizeof(uint16_t) == PROJECTION_ROW_BYTES &&
		   MODEL_STREAMS_ROW_BYTES == STREAMS_ROW_BYTES &&
		   MODEL_STREAMS == STREAM_COUNT &&
		   ENGRAM_ROWS_GATE_THREADS == GATE_BLOCK,
	   "the projection, the streams and the gate's block are the model's");
    expect((float)REFERENCE_EPSILON == MODEL_NORM_EPSILON,
	   "the epsilon of the gate is the float32 the reference makes of its double");
}

static void test_rows_fill(void)
{
	const uint32_t table_row_count =
		LINEAR_MAXIMUM_ROWS * TEST_ENGRAM_ROWS_PAIR_TABLE_ROWS;
	bool each_row_holds_every_byte = true;
	bool each_pair_once = true;
	context_text = "the table rows of the values case";
	test_engram_rows_rows_fill(pair_rows, table_row_count);
	for (uint32_t table_row = 0; table_row < table_row_count; table_row++) {
		const uint8_t *const row =
			&pair_rows[(size_t)table_row * TABLE_ROW_BYTES];
		bool is_seen[CODE_COUNT] = { false };
		for (uint32_t place = 0; place < ENGRAM_ROWS_VALUES; place++) {
			each_row_holds_every_byte = each_row_holds_every_byte &&
						    !is_seen[row[place]];
			is_seen[row[place]] = true;
			pair_counts[row[ENGRAM_ROWS_VALUES +
					place / ENGRAM_ROWS_VALUES_PER_SCALE]]
				   [row[place]]++;
		}
    }
    for (uint32_t pair = 0; pair < CODE_COUNT * CODE_COUNT; pair++) {
        each_pair_once = each_pair_once && pair_counts[pair / CODE_COUNT][pair % CODE_COUNT] == 1;
    }
    expect(each_row_holds_every_byte, "each table row holds every byte once");
    expect(each_pair_once, "each pair of a byte and a scale code appears once");
}

/* ---- the host's references, against an emulation of the kernels' float32 arithmetic ---- */

static float float_of(uint16_t bits)
{
    return (float)test_linear_bfloat16_value(bits);
}

/* The gate kernel on one stream in its order (engram_rows_records.h), its rsqrtf, sqrtf and expf replaced by the
 * float32 nearest the double: within the last places the bound allows. A fault breaks what the kernel must do. */
static void gate_emulate(const uint16_t *stream, const uint16_t *key, const uint16_t *query_weight,
                         const uint16_t *key_weight, const uint16_t *value, enum emulation_fault fault,
                         uint16_t *output)
{
    static float lane_sums[GATE_BLOCK][SUM_COUNT];
    float sums[SUM_COUNT] = {0.0f, 0.0f, 0.0f};
    for (uint32_t thread = 0; thread < GATE_BLOCK; thread++) {
        memset(lane_sums[thread], 0, sizeof(lane_sums[thread]));
        for (uint32_t place = thread; place < HIDDEN; place += GATE_BLOCK) {
            const float h = float_of(stream[place]);
            const float key_value = float_of(key[place]);
            const float weight = float_of(query_weight[place]) * float_of(key_weight[place]);
            const float term = h * weight;
            lane_sums[thread][STREAM_SQUARES] += h * h;
            lane_sums[thread][KEY_SQUARES] += key_value * key_value;
            lane_sums[thread][DOT] += term * key_value;
        }
    }
    for (uint32_t distance = LANES >> 1u; distance != 0; distance >>= 1u) {
        static float partner[GATE_BLOCK][SUM_COUNT];
        memcpy(partner, lane_sums, sizeof(partner));
        for (uint32_t thread = 0; thread < GATE_BLOCK; thread++) {
            for (uint32_t sum = 0; sum < SUM_COUNT; sum++) {
                lane_sums[thread][sum] += partner[thread ^ distance][sum];
            }
        }
    }
    for (uint32_t warp = 0; warp < WARPS; warp++) {
        for (uint32_t sum = 0; sum < SUM_COUNT; sum++) {
            sums[sum] += lane_sums[(size_t)warp * LANES][sum];
        }
    }
    sums[DOT] += fault == FAULT_FIRST_THREAD_TWICE ? lane_sums[0][DOT] : 0.0f;
    const float epsilon = fault == FAULT_NO_EPSILON ? 0.0f : MODEL_NORM_EPSILON;
    const float inverse_hidden = 1.0f / (float)HIDDEN;
    const float rstd = (float)(1.0 / test_engram_rows_square_root((double)(sums[STREAM_SQUARES] * inverse_hidden +
                                                                           epsilon))) *
                       (float)(1.0 / test_engram_rows_square_root((double)(sums[KEY_SQUARES] * inverse_hidden +
                                                                           epsilon)));
    const float dot = sums[DOT] * rstd * (float)(1.0 / test_engram_rows_square_root((double)HIDDEN));
    const float magnitude = dot < 0.0f ? -dot : dot;
    const float clamp = fault == FAULT_NO_CLAMP ? 0.0f : (float)CLAMP;
    const float bounded = magnitude < clamp ? clamp : magnitude;
    const float root = bounded == 0.0f || isnan(bounded) ? bounded : (float)test_engram_rows_square_root(bounded);
    const float signed_root = signbit(dot) ? -root : root;
    /* e^-s = 1 / sigmoid(s) - 1; the sigmoid of 0 is 1/2 */
    const float exponential = signed_root == 0.0f ? 1.0f : (float)(1.0 / test_engram_rows_sigmoid(signed_root) - 1.0);
    const float gate = 1.0f / (1.0f + exponential);
    for (uint32_t place = 0; place < HIDDEN; place++) {
        const float product = gate * float_of(value[place]);
        output[place] = test_linear_bfloat16_nearest((double)(float_of(stream[place]) + product));
    }
}

/* The outputs of the emulation (run again unless is_kept) outside the reference's bound, of every case or one;
 * the share of central ones. */
static uint64_t emulation_outside(enum emulation_fault fault, uint32_t only_case, bool is_kept,
                                  double *central_share_out)
{
    uint64_t outside = 0;
    uint64_t equal_count = 0;
    uint32_t first = 0;
    for (uint32_t row = 0; row < CASES; row++) {
        for (uint32_t stream = 0; stream < STREAM_COUNT && (only_case == CASES || only_case == row); stream++) {
            const size_t stream_place = ((size_t)row * STREAM_COUNT + stream) * HIDDEN;
	    const uint16_t *const projection =
		    &case_projection[(size_t)row * ENGRAM_ROWS_PROJECTION];
	    const uint16_t *const key = &projection[(size_t)stream * HIDDEN];
            const uint16_t *const value = &projection[STREAM_VALUES];
            const uint16_t *const query_weight = &case_query_weight[(size_t)stream * HIDDEN];
            const uint16_t *const key_weight = &case_key_weight[(size_t)stream * HIDDEN];
            const struct test_engram_rows_gate gate =
                test_engram_rows_gate_of(&case_streams[stream_place], key, query_weight, key_weight);
            if (!is_kept) {
                gate_emulate(&case_streams[stream_place], key, query_weight, key_weight, value, fault,
                             &case_outputs[stream_place]);
            }
            expect(gate.lowest <= gate.central && gate.central <= gate.highest &&
                       (gate.is_not_a_number || (gate.lowest >= 0.0 && gate.highest <= 1.0)),
                   "the bound holds the central gate, inside 0 to 1");
            outside += test_engram_rows_gate_outside(&gate, &case_streams[stream_place], value,
                                                     &case_outputs[stream_place], &equal_count, &first);
        }
    }
    *central_share_out = (double)equal_count / (double)(only_case == CASES ? CASES * STREAM_VALUES : STREAM_VALUES);
    return outside;
}

static void test_references(void)
{
    static const struct {
        uint32_t code;
        uint32_t scale_code;
        uint16_t bits;
    } value_cases[] = {{E4M3_ONE, E8M0_ONE, BFLOAT16_ONE},
                       {E4M3_448, E8M0_ONE, BFLOAT16_448},
                       {E4M3_SMALLEST, 0, 0},
                       {E4M3_MINUS_ZERO, E8M0_ONE, BFLOAT16_MINUS_ZERO},
                       {E4M3_NOT_A_NUMBER, E8M0_ONE, BFLOAT16_NOT_A_NUMBER},
                       {E4M3_ONE, E8M0_NOT_A_NUMBER, BFLOAT16_NOT_A_NUMBER},
                       {E4M3_448, E8M0_LARGEST, BFLOAT16_INFINITY}};
    double central_share = 0.0;
    uint64_t differing = 0;
    context_text = "the host's references";
    for (size_t case_index = 0; case_index < COUNT_OF(value_cases); case_index++) {
        expect(test_engram_rows_value(value_cases[case_index].code, value_cases[case_index].scale_code) ==
                   value_cases[case_index].bits,
               "a value of a byte and a scale code is the one worked out by hand");
    }
    for (uint32_t code = 0; code < CODE_COUNT; code++) {
        for (uint32_t scale_code = 0; scale_code < CODE_COUNT; scale_code++) {
            const float product =
                (float)test_weight_format_e4m3_value(code) * (float)test_weight_format_e8m0_value(scale_code);
            differing += test_linear_bfloat16_nearest((double)product) != test_engram_rows_value(code, scale_code);
        }
    }
    expect(differing == 0, "the float32 product of the byte's number and the scale is the exact one, rounded once");
    test_engram_rows_gate_fill(case_streams, case_projection, case_query_weight, case_key_weight);
    expect(emulation_outside(FAULT_NO_EPSILON, TEST_ENGRAM_ROWS_TINY_KEY, false, &central_share) != 0,
           "a gate without epsilon is reported where epsilon decides");
    expect(emulation_outside(FAULT_NO_CLAMP, TEST_ENGRAM_ROWS_ZERO_KEY, false, &central_share) != 0,
           "a gate without the clamp is reported where the clamp decides");
    expect(emulation_outside(FAULT_FIRST_THREAD_TWICE, TEST_ENGRAM_ROWS_PLAIN, false, &central_share) != 0,
           "a dot that adds one thread's sum twice is reported");
    expect(emulation_outside(FAULT_NONE, CASES, false, &central_share) == 0,
           "every output of the kernel's arithmetic lies within the bound");
    (void)printf("the emulated gate: %.4f%% of the outputs are those of the central gate\n", central_share * PERCENT);
    expect(central_share >= LEAST_CENTRAL_SHARE, "most outputs are those of the central gate");
    for (uint32_t place = 0; place < HIDDEN; place++) {
        expect(case_outputs[(size_t)TEST_ENGRAM_ROWS_NOT_A_NUMBER * STREAM_VALUES + place] == BFLOAT16_NOT_A_NUMBER,
               "a not-a-number in a stream makes each of its outputs one");
    }
    case_streams[CHANGED_PLACE] ^= BFLOAT16_MINUS_ZERO;     /* one stored word changed in the reference's input */
    expect(emulation_outside(FAULT_NONE, TEST_ENGRAM_ROWS_PLAIN, true, &central_share) != 0,
           "a changed stream value is reported");
    case_streams[CHANGED_PLACE] ^= BFLOAT16_MINUS_ZERO;
}

int main(void)
{
	struct engram_rows_kernels kernels;
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
	test_constants();
	test_rows_fill();
	test_references();
	test_host_memory_backend.card_close(card, 0);
	context_text = "the end of the test";
	expect(close(capture_descriptor) == 0 && unlink(capture_path) == 0 &&
		       close(error_stream_descriptor) == 0 &&
		       test_host_memory_backend_open_card_count() == 0,
	       "the temporary file is removed and no card is left open");
	if (failures != 0) {
		return EXIT_FAILURE;
	}
    (void)printf("engram_rows_test passed\n");
    return EXIT_SUCCESS;
}
