/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * linear_launch_test: what source/linear.c does with the instances of the product kernels, beside linear_test. No
 * card: a prepare function is arithmetic on a description, the arena is a number, the kernel handles are made up.
 *
 *   - linear_kernels_prepare against a table that records what it is asked: every function once, in the
 *     order of the members, with its file and the size of its record; the first failure ends it with that status,
 *     names the function and leaves the handles untouched.
 *   - Slice counts. For every count from 0 to 17 and every product of every kind of description: a count of
 *     LINEAR_EVERY_SLICE_COUNT is accepted and launches the instance stamped for it, with its block; any
 *     other is refused with one error line.
 *   - Two matrices in one trunk launch: an accepted pair, its launch by hand; refused, each with one error line: a
 *     matrix_count of 0 or 3, two matrices that differ in input_size, slice_count or weight format, or that have
 *     more outputs than 32 bits count, and outputs that share a byte.
 *   - The default slice counts, for every matrix of the model in every family, class and profile: a count a kernel
 *     is stamped for and the matrix has the loads of, one count for the two matrices of one trunk launch
 *     (attn.wq_a with attn.wkv); the reference's order gives 1; refused, with one error line:
 *     a shape or a class with no row, an unknown profile, order or family, and a missing argument. The draft's
 *     main_proj (dspark_v1/DESIGN.md K8): its counts and wide forms on class 80 and class 120 native, by hand, and no
 *     row in class 120's emulation, which the draft never runs.
 *   - Wide trunk launches. linear_kernels_prepare leaves the wide instances unprepared and
 *     linear_wide_kernels_prepare asks for those of its class's wide forms alone, by hand for each class, in
 *     order, the rest of the table as it was (its failures as the other's). The wide forms of every matrix of the model
 *     in every class and profile, by hand: class 120 native keeps the default count. For 9, 64, 1,001 and 1,024 rows,
 *     a trunk launch of one matrix, of two, of bfloat16 rows in two segments, and the shared expert's two trunk
 *     launches are of the wide instance of their slice count, with the thin one's block and shared memory, a block for
 *     each group of 8 rows of each tile and a record of those rows; each range that grows with the rows, one byte
 *     short, is refused, and so is an arena one byte short of the last range; 0 and 1,025 rows are refused, and a wide
 *     instance that is not prepared, which 8 rows (the thin instance, its grid and record as before) never need. A
 *     matrix of so many tiles that a block for each group would pass 32 bits gets the most blocks a tile that 32 bits
 *     count. The other forms: a block a tile; the interleaved instance of one slice, a block a chunk of 32 rows or a
 *     block a tile; refused: an interleaved form of more slices or of bfloat16 rows, a form not known, a reserved field
 *     of a matrix, two matrices of other forms, the interleaved instance not prepared.
 * A failure is one line on the standard output: the case, and what was expected of it.
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
#include "log_expectation.h"
#include "anemone/linear.h"
#include "anemone/linear_records.h"
#include "anemone/model_layout.h"
#include "anemone/status.h"
#include "anemone/weight_format_seam.h"
#include "host_memory_backend.h"
#include "weight_format_reference.h"

#define COUNT_OF(table) (sizeof(table) / sizeof((table)[0]))
#define E2M1 WEIGHT_FORMAT_E2M1_SCALED
#define CLASS_80 ARITHMETIC_CLASS_SM_80 /* the class of the tables' card */
#define E4M3 WEIGHT_FORMAT_E4M3_SCALED
#define E2M1_BITS_PER_WEIGHT 4u
#define E4M3_BITS_PER_WEIGHT 8u
#if defined(LINEAR_CLASS_FORMS)
#define REFERENCE_SLICES(fixed) 1u              /* each class its own forms: the reference order is one slice */
#else
#define REFERENCE_SLICES(fixed) (fixed)         /* the exact profile: ordered slices keep the reference order */
#endif
#define THIN_KERNEL_COUNT \
	23u /* linear_kernels_prepare's: three, and four of five counts */
#define WIDE_KERNEL_COUNT 11u                   /* the wide instances: two of five, one more */
#define MOST_CLASS_WIDE_KERNELS 5u              /* those of one class's wide forms */
#define KERNEL_COUNT         \
	(THIN_KERNEL_COUNT + \
	 WIDE_KERNEL_COUNT) /* the members of struct linear_kernels_record */
#define FIRST_RECORDED_HANDLE 40u               /* what the recording table gives the first kernel it is asked */
#define FIRST_MADE_UP_HANDLE 11u
#define CARD_MEMORY_BYTES 4096u                 /* of the stand-in card that the kernels are asked of */
#define UNTOUCHED_BYTE 0xA5
#define CAPTURE_CAPACITY 2048u
#define MOST_LAUNCHES 4u
#define MOST_SLICES_TRIED 17u /* 0 to 17: one beyond LINEAR_MAXIMUM_SLICES */
#define WIDE_INPUTS 1024u                       /* 16 loads: room for every count */
#define FEW_OUTPUTS 64u
#define OTHER_INPUTS 2048u
#define ROWS 2u
#define PAIRS_PER_ROW 3u
#define JOBS 4u
#define ACTIVATION_LIMIT 10.0f
#define HALF_OF_32_BITS_OF_OUTPUTS 2147483648u  /* 2^31: two such matrices have more outputs than 32 bits count */
#define ARENA_BYTES (1ull << 50u)               /* more than any case lays out */
#define WIDE_CHUNK_ROWS 64u                     /* the wide row counts tried, beside 9 and the most */
#define UNEVEN_WIDE_ROWS 1001u                  /* 125 groups of 8 and one of 1 */
#define WIDE_SEGMENTS 2u                        /* of the bfloat16 case: 2 tiles each */
#define WIDE_SLICES 4u                          /* of every wide case but the shared expert's down */
#define WIDE_DOWN_SLICES 2u
#define MOST_SCALED_RANGES 10u                  /* the shared expert's ranges that grow with the rows */
#define HUGE_OUTPUTS 536870928u                 /* 2^29 + 16: 2^25 + 1 tiles, a block for each of 128 groups would
                                                   pass 32 bits */

/* Words of the one error line of each rule. */
#define RULE_STAMPED "a product kernel is stamped for"
#define RULE_MATRICES "a launch reads 1 or"
#define RULE_PAIR "the two matrices of one launch differ"
#define RULE_SHARED "share a byte"
#define RULE_NO_ROW "no row for family"
#define RULE_NOT_KNOWN "a missing argument, or a profile, order or family not known"
#define RULE_MISSING "a missing argument"
#define RULE_ROWS "rows: a trunk launch takes"
#define RULE_TOUCHED "the launches touch"
#define RULE_ARENA "leaves the arena"
#define RULE_NOT_PREPARED "is not prepared"
#define RULE_FORM "an interleaved one takes quantized rows in one slice"
#define RULE_NOT_TRUNK "a family other than the trunk's"
#define RULE_RESERVED "a reserved field of the description"
#define PROFILE_COUNT 2u
#define PRECISION LINEAR_PROFILE_PRECISION
#define SPEED LINEAR_PROFILE_SPEED
#define FIXED LINEAR_ORDER_FIXED
#define UNKNOWN_ORDER (LINEAR_ORDER_REFERENCE + 1)
#define GROUPS LINEAR_WIDE_GROUP_BLOCKS
#define TILES LINEAR_WIDE_TILE_BLOCKS
#define INTERLEAVED_CHUNKS LINEAR_INTERLEAVED_CHUNK_BLOCKS
#define INTERLEAVED_TILES LINEAR_INTERLEAVED_TILE_BLOCKS

enum described { TRUNK_QUANTIZED, TRUNK_BFLOAT16, ROUTED_GATE_UP, ROUTED_DOWN, SHARED_GATE_UP, SHARED_DOWN,
                 DESCRIBED_COUNT };

static int failures = 0;
static const char *context_text = "";
static char capture_path[] = "/tmp/anemone_linear_launch_test_XXXXXX";
static int capture_descriptor = -1;             /* the temporary file that stands in for the error stream */
static int error_stream_descriptor = -1;        /* a duplicate of descriptor 2, kept to put it back */
static struct linear_kernels_record made_up_kernels;
static const uint32_t stamped_counts[LINEAR_SLICE_INSTANCES] = {
	1, 2, 4, 8, 16
}; /* by hand */

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

/* Copies what was logged since capture_begin into text, as a string, and puts descriptor 2 back. */
static void capture_end(char *text, size_t text_capacity)
{
    const bool is_open = capture_descriptor >= 0 && error_stream_descriptor >= 0 && fflush(stderr) == 0;
    const ssize_t byte_count = is_open ? pread(capture_descriptor, text, text_capacity - 1, 0) : -1;
    expect(is_open && dup2(error_stream_descriptor, STDERR_FILENO) == STDERR_FILENO && byte_count >= 0,
           "the temporary file is read back and the error stream restored");
    text[byte_count >= 0 ? (size_t)byte_count : 0] = '\0';
}

/* The text is one line, of an error, that holds the words. */
static void expect_one_error_line_with(const char *text, const char *words, const char *what)
{
    if (!test_log_is_expected(text, words, true)) {
        (void)printf("FAILED: %s: %s: no single error line with \"%s\": \"%s\"\n", context_text, what, words, text);
        failures++;
    }
}

/* ---- the kernels of a card ---- */

struct kernel_asked {
    const char *kernel_file_name;
    const char *function_name;
    size_t argument_byte_count;
};
#define QUANTIZE_RECORD sizeof(struct activation_quantize_arguments)
#define GATE_UP_RECORD sizeof(struct expert_gate_up_arguments)
#define DOWN_RECORD sizeof(struct expert_down_arguments)
#define ROW_SUM_RECORD sizeof(struct expert_row_sum_arguments)
#define TRUNK_RECORD sizeof(struct trunk_linear_quantized_arguments)
#define BFLOAT16_RECORD sizeof(struct trunk_linear_bfloat16_arguments)
#define HIDDEN_RECORD sizeof(struct trunk_hidden_arguments)
/* What linear_records.h says of each function, in the order of the members of struct linear_kernels_record. */
static const struct kernel_asked kernels_expected[KERNEL_COUNT] = {
	{ "kernel_activation_quantize", "activation_quantize_entry",
	  QUANTIZE_RECORD },
	{ "kernel_expert_linear", "expert_gate_up_1", GATE_UP_RECORD },
	{ "kernel_expert_linear", "expert_gate_up_2", GATE_UP_RECORD },
	{ "kernel_expert_linear", "expert_gate_up_4", GATE_UP_RECORD },
	{ "kernel_expert_linear", "expert_gate_up_8", GATE_UP_RECORD },
	{ "kernel_expert_linear", "expert_gate_up_16", GATE_UP_RECORD },
	{ "kernel_expert_linear", "expert_down_1", DOWN_RECORD },
	{ "kernel_expert_linear", "expert_down_2", DOWN_RECORD },
	{ "kernel_expert_linear", "expert_down_4", DOWN_RECORD },
	{ "kernel_expert_linear", "expert_down_8", DOWN_RECORD },
	{ "kernel_expert_linear", "expert_down_16", DOWN_RECORD },
	{ "kernel_expert_linear", "expert_row_sum_entry", ROW_SUM_RECORD },
	{ "kernel_trunk_linear", "trunk_linear_quantized_1", TRUNK_RECORD },
	{ "kernel_trunk_linear", "trunk_linear_quantized_2", TRUNK_RECORD },
	{ "kernel_trunk_linear", "trunk_linear_quantized_4", TRUNK_RECORD },
	{ "kernel_trunk_linear", "trunk_linear_quantized_8", TRUNK_RECORD },
	{ "kernel_trunk_linear", "trunk_linear_quantized_16", TRUNK_RECORD },
	{ "kernel_trunk_linear", "trunk_linear_bfloat16_1", BFLOAT16_RECORD },
	{ "kernel_trunk_linear", "trunk_linear_bfloat16_2", BFLOAT16_RECORD },
	{ "kernel_trunk_linear", "trunk_linear_bfloat16_4", BFLOAT16_RECORD },
	{ "kernel_trunk_linear", "trunk_linear_bfloat16_8", BFLOAT16_RECORD },
	{ "kernel_trunk_linear", "trunk_linear_bfloat16_16", BFLOAT16_RECORD },
	{ "kernel_trunk_linear", "trunk_hidden_entry", HIDDEN_RECORD },
	{ "kernel_trunk_linear", "trunk_linear_quantized_wide_1",
	  TRUNK_RECORD },
	{ "kernel_trunk_linear", "trunk_linear_quantized_wide_2",
	  TRUNK_RECORD },
	{ "kernel_trunk_linear", "trunk_linear_quantized_wide_4",
	  TRUNK_RECORD },
	{ "kernel_trunk_linear", "trunk_linear_quantized_wide_8",
	  TRUNK_RECORD },
	{ "kernel_trunk_linear", "trunk_linear_quantized_wide_16",
	  TRUNK_RECORD },
	{ "kernel_trunk_linear", "trunk_linear_bfloat16_wide_1",
	  BFLOAT16_RECORD },
	{ "kernel_trunk_linear", "trunk_linear_bfloat16_wide_2",
	  BFLOAT16_RECORD },
	{ "kernel_trunk_linear", "trunk_linear_bfloat16_wide_4",
	  BFLOAT16_RECORD },
	{ "kernel_trunk_linear", "trunk_linear_bfloat16_wide_8",
	  BFLOAT16_RECORD },
	{ "kernel_trunk_linear", "trunk_linear_bfloat16_wide_16",
	  BFLOAT16_RECORD },
	{ "kernel_trunk_linear", "trunk_linear_quantized_interleaved_1",
	  TRUNK_RECORD },
};
/* The wide instances of each class's wide forms (the table of linear.c, by hand), as their places behind the thin
 * members, in order: class 80 quantized rows of 4 and 8 slices, bfloat16 rows of 1, the interleaved instance; class
 * 120 native quantized rows of 2, 4, 8 and 16 slices, bfloat16 rows of 8; class 120 quantized rows of 4 and 8,
 * bfloat16 rows of 1. */
enum { WIDE_QUANTIZED_2 = 1, WIDE_QUANTIZED_4 = 2, WIDE_QUANTIZED_8 = 3, WIDE_QUANTIZED_16 = 4, WIDE_BFLOAT16_1 = 5,
       WIDE_BFLOAT16_8 = 8, WIDE_INTERLEAVED = 10 };
struct class_wide {
    uint32_t arithmetic_class;
    uint32_t count;
    uint32_t places[MOST_CLASS_WIDE_KERNELS];
};
static const struct class_wide class_wide_kernels[] = {
	{ ARITHMETIC_CLASS_SM_80,
	  4,
	  { WIDE_QUANTIZED_4, WIDE_QUANTIZED_8, WIDE_BFLOAT16_1,
	    WIDE_INTERLEAVED } },
	{ ARITHMETIC_CLASS_SM_120_NATIVE,
	  5,
	  { WIDE_QUANTIZED_2, WIDE_QUANTIZED_4, WIDE_QUANTIZED_8,
	    WIDE_QUANTIZED_16, WIDE_BFLOAT16_8 } },
	{ ARITHMETIC_CLASS_SM_120,
	  3,
	  { WIDE_QUANTIZED_4, WIDE_QUANTIZED_8, WIDE_BFLOAT16_1 } },
};
static struct kernel_asked kernels_asked[KERNEL_COUNT];
static uint32_t asked_count = 0;
static uint32_t failing_call_number = 0;        /* counted from 1; 0: no call fails */

_Static_assert(
	sizeof(struct linear_kernels_record) == KERNEL_COUNT * sizeof(uint32_t),
	"the test reads the handles as 34 numbers: 23 thin, then 11 wide");

/* kernel_prepare of a table that records what it is asked and gives the handles FIRST_RECORDED_HANDLE and on. */
MUST_CHECK static result_status
recording_kernel_prepare(struct device_card *card, const char *kernel_file_name,
			 const char *function_name,
			 uint32_t argument_byte_count, uint32_t *kernel_out)
{
    if (card != NULL && asked_count < KERNEL_COUNT) {
        kernels_asked[asked_count] = (struct kernel_asked){kernel_file_name, function_name, argument_byte_count};
    }
    asked_count++;
    if (asked_count == failing_call_number) {
	    return STATUS_DEVICE_ERROR;
    }
    *kernel_out = FIRST_RECORDED_HANDLE + asked_count - 1;
    return STATUS_OK;
}

static void test_kernels_prepare(void)
{
	static const struct backend_operations recording_backend = {
		.name = "recording", .kernel_prepare = recording_kernel_prepare
	};
	static const struct backend_operations backend_without_kernels = {
		.name = "no kernels"
	};
	struct linear_kernels_record kernels;
	uint32_t handles[KERNEL_COUNT];
	uint32_t untouched_handles[KERNEL_COUNT];
	uint32_t handles_before[KERNEL_COUNT];
	struct device_card *card = NULL;
	char logged[CAPTURE_CAPACITY];
	context_text = "the kernels of a card";
	memset(&kernels, UNTOUCHED_BYTE, sizeof(kernels));
	memset(untouched_handles, UNTOUCHED_BYTE, sizeof(untouched_handles));
	test_host_memory_backend_reset(1, CARD_MEMORY_BYTES);
	expect(test_host_memory_backend.card_open(0, CARD_MEMORY_BYTES,
						  &card) == STATUS_OK,
	       "a card opens");
	capture_begin();
	expect(linear_kernels_prepare(NULL, card, CLASS_80, &kernels) ==
		       STATUS_INVALID_ARGUMENT,
	       "no table");
	expect(linear_kernels_prepare(&backend_without_kernels, card, CLASS_80,
				      &kernels) == STATUS_INVALID_ARGUMENT,
	       "a table without kernel_prepare");
	expect(linear_kernels_prepare(&recording_backend, NULL, CLASS_80,
				      &kernels) == STATUS_INVALID_ARGUMENT,
	       "no card");
	expect(linear_kernels_prepare(&recording_backend, card, CLASS_80,
				      NULL) == STATUS_INVALID_ARGUMENT,
	       "nowhere to put the handles");
	/* the stand-in has no build of any kernel: its status comes back as it is */
	expect(linear_kernels_prepare(&test_host_memory_backend, card, CLASS_80,
				      &kernels) == STATUS_UNSUPPORTED,
	       "a card without a build of the kernels");
	capture_end(logged, sizeof(logged));
	for (uint32_t call_number = 1; call_number <= THIN_KERNEL_COUNT;
	     call_number++) {
		expect(asked_count == 0,
		       "a table is asked nothing more once a call was refused or has failed");
		failing_call_number = call_number;
		capture_begin();
		expect(linear_kernels_prepare(&recording_backend, card,
					      CLASS_80,
					      &kernels) == STATUS_DEVICE_ERROR,
		       "the status of the kernel_prepare that fails");
		capture_end(logged, sizeof(logged));
		expect_one_error_line_with(
			logged, kernels_expected[call_number - 1].function_name,
			"the kernel that fails is named");
		expect(asked_count == call_number,
		       "the first failure ends the preparation");
		asked_count = 0;
	}
    failing_call_number = 0;
    memcpy(handles, &kernels, sizeof(handles));
    expect(memcmp(handles, untouched_handles, sizeof(handles)) == 0, "a failure leaves the handles untouched");
    expect(linear_kernels_prepare(&recording_backend, card, CLASS_80,
				  &kernels) == STATUS_OK,
	   "every kernel is prepared");
    memcpy(handles, &kernels, sizeof(handles));
    for (uint32_t kernel_index = 0; kernel_index < THIN_KERNEL_COUNT; kernel_index++) {
        const struct kernel_asked *const asked = &kernels_asked[kernel_index];
        const struct kernel_asked *const expected = &kernels_expected[kernel_index];
        expect(asked_count == THIN_KERNEL_COUNT && asked->kernel_file_name != NULL && asked->function_name != NULL &&
                   strcmp(asked->kernel_file_name, expected->kernel_file_name) == 0 &&
                   strcmp(asked->function_name, expected->function_name) == 0 &&
                   asked->argument_byte_count == expected->argument_byte_count &&
                   handles[kernel_index] == FIRST_RECORDED_HANDLE + kernel_index,
               "each function is asked of the card once, in the order of the members, by its file, its name and "
               "the size of its record, and its member holds its handle");
    }
    for (uint32_t kernel_index = THIN_KERNEL_COUNT; kernel_index < KERNEL_COUNT; kernel_index++) {
	    expect(handles[kernel_index] == BACKEND_NO_KERNEL,
		   "the wide instances are left unprepared");
    }
    asked_count = 0;
    expect(linear_kernels_prepare(&recording_backend, card,
				  ARITHMETIC_CLASS_SM_120_NATIVE,
				  &kernels) == STATUS_OK &&
		   asked_count == THIN_KERNEL_COUNT,
	   "every kernel is prepared on a card of class 120 native");
    for (uint32_t kernel_index = 0; kernel_index < THIN_KERNEL_COUNT; kernel_index++) {
        const char *const expected_file = kernels_expected[kernel_index].kernel_file_name;
	const char *const native_file =
		strcmp(expected_file, KERNEL_FILE_EXPERT_LINEAR) == 0 ?
			KERNEL_FILE_EXPERT_LINEAR_NATIVE :
		strcmp(expected_file, KERNEL_FILE_TRUNK_LINEAR) == 0 ?
			KERNEL_FILE_TRUNK_LINEAR_NATIVE :
			expected_file;
	expect(strcmp(kernels_asked[kernel_index].kernel_file_name, native_file) == 0 &&
                   strcmp(kernels_asked[kernel_index].function_name, kernels_expected[kernel_index].function_name) == 0,
               "class 120 native asks the product kernels of their files' native builds, the quantizer of its own "
               "file");
    }
    /* the wide instances of each class's forms alone, in the order of their members, into the table above */
    context_text = "the wide kernels of a card";
    asked_count = 0;
    memcpy(handles_before, &kernels, sizeof(handles_before));
    capture_begin();
    expect(linear_wide_kernels_prepare(NULL, card, PRECISION, FIXED, CLASS_80,
				       &kernels) == STATUS_INVALID_ARGUMENT &&
		   linear_wide_kernels_prepare(
			   &backend_without_kernels, card, PRECISION, FIXED,
			   CLASS_80, &kernels) == STATUS_INVALID_ARGUMENT &&
		   linear_wide_kernels_prepare(
			   &recording_backend, NULL, PRECISION, FIXED, CLASS_80,
			   &kernels) == STATUS_INVALID_ARGUMENT &&
		   linear_wide_kernels_prepare(
			   &recording_backend, card, PRECISION, FIXED, CLASS_80,
			   NULL) == STATUS_INVALID_ARGUMENT &&
		   linear_wide_kernels_prepare(
			   &recording_backend, card, PROFILE_COUNT, FIXED,
			   CLASS_80, &kernels) == STATUS_INVALID_ARGUMENT &&
		   linear_wide_kernels_prepare(
			   &recording_backend, card, PRECISION, UNKNOWN_ORDER,
			   CLASS_80, &kernels) == STATUS_INVALID_ARGUMENT &&
		   asked_count == 0,
	   "no table, a table without kernel_prepare, no card, no handles, a profile or an order not known are "
	   "refused, and nothing is asked");
    capture_end(logged, sizeof(logged));
    for (uint32_t class_index = 0; class_index < COUNT_OF(class_wide_kernels); class_index++) {
        const struct class_wide *const wide = &class_wide_kernels[class_index];
	const bool is_native = wide->arithmetic_class ==
			       ARITHMETIC_CLASS_SM_120_NATIVE;
	for (uint32_t call_number = 1; call_number <= wide->count; call_number++) {
            failing_call_number = call_number;
            capture_begin();
	    expect(linear_wide_kernels_prepare(&recording_backend, card, SPEED,
					       FIXED, wide->arithmetic_class,
					       &kernels) == STATUS_DEVICE_ERROR,
		   "the status of the kernel_prepare that fails");
	    capture_end(logged, sizeof(logged));
            expect_one_error_line_with(
                logged, kernels_expected[THIN_KERNEL_COUNT + wide->places[call_number - 1u]].function_name,
                "the wide kernel that fails is named");
            memcpy(handles, &kernels, sizeof(handles));
            expect(asked_count == call_number && memcmp(handles, handles_before, sizeof(handles)) == 0,
                   "the first failure ends the preparation and leaves the handles untouched");
            asked_count = 0;
        }
        failing_call_number = 0;
	expect(linear_wide_kernels_prepare(&recording_backend, card, PRECISION,
					   FIXED, wide->arithmetic_class,
					   &kernels) == STATUS_OK &&
		       asked_count == wide->count,
	       "the wide kernels of a class's forms are prepared, those alone");
	memcpy(handles, &kernels, sizeof(handles));
        uint32_t next = 0;                      /* of wide->places */
        for (uint32_t kernel_index = 0; kernel_index < KERNEL_COUNT; kernel_index++) {
            const bool is_asked = next < wide->count && kernel_index == THIN_KERNEL_COUNT + wide->places[next];
            const struct kernel_asked *const asked = &kernels_asked[is_asked ? next : 0];
            const struct kernel_asked *const expected = &kernels_expected[kernel_index];
	    const char *const file_name =
		    is_native ? KERNEL_FILE_TRUNK_LINEAR_NATIVE :
				expected->kernel_file_name;
	    expect(!is_asked ? handles[kernel_index] == handles_before[kernel_index]
                             : strcmp(asked->kernel_file_name, file_name) == 0 &&
                                   strcmp(asked->function_name, expected->function_name) == 0 &&
                                   asked->argument_byte_count == expected->argument_byte_count &&
                                   handles[kernel_index] == FIRST_RECORDED_HANDLE + next,
                   "each wide function of the class is asked once, in the order of the members, by its file (class "
                   "120 native's build of it), name and record size, its member holds its handle, and every other "
                   "member is as it was");
            next += is_asked ? 1u : 0u;
        }
        asked_count = 0;
        memcpy(&kernels, handles_before, sizeof(kernels));
    }
    asked_count = 0;
    test_host_memory_backend.card_close(card, 0);
}

/* ---- descriptions laid out one range behind another in an arena of ARENA_BYTES ---- */

/* The next range of byte_count bytes from *cursor on, on a multiple of the alignment. */
static struct linear_range next_range(uint64_t *cursor, uint64_t byte_count)
{
	const struct linear_range range = { *cursor, byte_count };
	*cursor += (byte_count + LINEAR_OFFSET_ALIGNMENT_BYTES - 1) /
		   LINEAR_OFFSET_ALIGNMENT_BYTES *
		   LINEAR_OFFSET_ALIGNMENT_BYTES;
	return range;
}

/* A matrix laid out at the cursor with the bytes its format stores. */
static struct linear_matrix
next_matrix(uint64_t *cursor, enum weight_format_code format,
	    uint32_t output_size, uint32_t input_size, uint32_t slice_count)
{
    struct test_weight_format_sizes sizes = {0};
    struct linear_matrix matrix = { .shape = { output_size, input_size,
					       format == E2M1 ?
						       E2M1_BITS_PER_WEIGHT :
						       E4M3_BITS_PER_WEIGHT,
					       0 },
				    .weight_format = format,
				    .slice_count = slice_count };
    expect(test_weight_format_sizes_of(format, &matrix.shape, &sizes) ==
		   STATUS_OK,
	   "the reference sizes a matrix of the test");
    matrix.stored = next_range(cursor, sizes.stored_byte_count);
    return matrix;
}

/* The arrays of rows of a width: values (and operands, of the same bytes), codes, exponents. */
static struct linear_range next_codes(uint64_t *cursor, uint64_t vectors,
				      uint32_t width)
{
	return next_range(cursor, linear_codes_byte_count(vectors, width));
}
static struct linear_range next_exponents(uint64_t *cursor, uint64_t vectors,
					  uint32_t width)
{
	return next_range(cursor, linear_exponents_byte_count(vectors, width));
}
static struct linear_range next_values(uint64_t *cursor, uint64_t vectors,
				       uint32_t width)
{
	return next_range(cursor, linear_values_byte_count(vectors, width));
}

/* A trunk launch of row_count quantized rows: one matrix of FEW_OUTPUTS by input_size, or two of them. */
static struct linear_trunk_quantized_description
trunk_quantized_of(uint32_t matrix_count, uint32_t input_size,
		   uint32_t slice_count, uint32_t row_count)
{
	uint64_t cursor = LINEAR_OFFSET_ALIGNMENT_BYTES;
	struct linear_trunk_quantized_description description = {
		.matrix_count = matrix_count, .row_count = row_count
	};
	description.codes = next_codes(&cursor, row_count, input_size);
	description.exponents = next_exponents(&cursor, row_count, input_size);
	description.operands = next_values(&cursor, row_count, input_size);
	for (uint32_t matrix = 0; matrix < matrix_count; matrix++) {
		description.matrices[matrix] = next_matrix(
			&cursor, E4M3, FEW_OUTPUTS, input_size, slice_count);
		description.outputs[matrix] =
			next_values(&cursor, row_count, FEW_OUTPUTS);
	}
    return description;
}

/* A trunk launch of row_count bfloat16 rows: a matrix of FEW_OUTPUTS by WIDE_INPUTS in segment_count segments. */
static struct linear_trunk_bfloat16_description
trunk_bfloat16_of(uint32_t slice_count, uint32_t row_count,
		  uint32_t segment_count)
{
	uint64_t cursor = LINEAR_OFFSET_ALIGNMENT_BYTES;
	struct linear_trunk_bfloat16_description description = {
		.row_count = row_count, .segment_count = segment_count
	};
	description.matrix = next_matrix(&cursor, E4M3, FEW_OUTPUTS,
					 WIDE_INPUTS, slice_count);
	description.values =
		next_values(&cursor, row_count, segment_count * WIDE_INPUTS);
	description.output = next_values(&cursor, row_count, FEW_OUTPUTS);
	return description;
}

/* The routed experts: gate, up and down of WIDE_INPUTS by WIDE_INPUTS. */
static struct linear_routed_experts_description
routed_of(uint32_t gate_up_slices, uint32_t down_slices)
{
	uint64_t cursor = LINEAR_OFFSET_ALIGNMENT_BYTES;
	const uint64_t pairs = (uint64_t)ROWS * PAIRS_PER_ROW;
	struct linear_routed_experts_description description = {
		.gate_up_shape = { WIDE_INPUTS, WIDE_INPUTS,
				   E2M1_BITS_PER_WEIGHT, 0 },
		.down_shape = { WIDE_INPUTS, WIDE_INPUTS, E2M1_BITS_PER_WEIGHT,
				0 },
		.weight_format = E2M1,
		.job_count = JOBS,
		.row_count = ROWS,
		.pairs_per_row = PAIRS_PER_ROW,
		.gate_up_slice_count = gate_up_slices,
		.down_slice_count = down_slices,
		.activation_limit = ACTIVATION_LIMIT
	};
	description.expert_weights =
		next_matrix(&cursor, E2M1, WIDE_INPUTS, WIDE_INPUTS, 1).stored;
	description.jobs = next_range(&cursor, linear_jobs_byte_count(JOBS));
	description.codes = next_codes(&cursor, ROWS, WIDE_INPUTS);
	description.exponents = next_exponents(&cursor, ROWS, WIDE_INPUTS);
	description.operands = next_values(&cursor, ROWS, WIDE_INPUTS);
	description.hidden = next_values(&cursor, pairs, WIDE_INPUTS);
	description.hidden_codes = next_codes(&cursor, pairs, WIDE_INPUTS);
	description.hidden_exponents =
		next_exponents(&cursor, pairs, WIDE_INPUTS);
	description.hidden_operands = next_values(&cursor, pairs, WIDE_INPUTS);
	description.pair_output = next_values(&cursor, pairs, WIDE_INPUTS);
	description.row_output =
		next_range(&cursor, linear_sums_byte_count(ROWS, WIDE_INPUTS));
	return description;
}

/* The shared expert of row_count rows: gate, up and down of WIDE_INPUTS by WIDE_INPUTS. */
static struct linear_shared_expert_description
shared_of(uint32_t gate_up_slices, uint32_t down_slices, uint32_t row_count)
{
	uint64_t cursor = LINEAR_OFFSET_ALIGNMENT_BYTES;
	struct linear_shared_expert_description description = {
		.row_count = row_count, .activation_limit = ACTIVATION_LIMIT
	};
	description.gate = next_matrix(&cursor, E4M3, WIDE_INPUTS, WIDE_INPUTS,
				       gate_up_slices);
	description.up = next_matrix(&cursor, E4M3, WIDE_INPUTS, WIDE_INPUTS,
				     gate_up_slices);
	description.down = next_matrix(&cursor, E4M3, WIDE_INPUTS, WIDE_INPUTS,
				       down_slices);
	description.codes = next_codes(&cursor, row_count, WIDE_INPUTS);
	description.exponents = next_exponents(&cursor, row_count, WIDE_INPUTS);
	description.operands = next_values(&cursor, row_count, WIDE_INPUTS);
	description.gate_output = next_values(&cursor, row_count, WIDE_INPUTS);
	description.up_output = next_values(&cursor, row_count, WIDE_INPUTS);
	description.hidden = next_values(&cursor, row_count, WIDE_INPUTS);
	description.hidden_codes = next_codes(&cursor, row_count, WIDE_INPUTS);
	description.hidden_exponents =
		next_exponents(&cursor, row_count, WIDE_INPUTS);
	description.hidden_operands =
		next_values(&cursor, row_count, WIDE_INPUTS);
	description.output = next_values(&cursor, row_count, WIDE_INPUTS);
	return description;
}

/* ---- slice counts ---- */

/* Prepares the description of a kind whose one product (the kind says which) has slice_count slices, the other
 * products of its chain one; launch_out: the launch of that product. */
MUST_CHECK static result_status
prepare_with_slices(enum described kind, uint32_t slice_count,
		    struct kernel_launch *launch_out)
{
	struct kernel_launch launches[MOST_LAUNCHES];
	result_status status = STATUS_INVALID_ARGUMENT;
	memset(launches, UNTOUCHED_BYTE, sizeof(launches));
	if (kind == TRUNK_QUANTIZED) {
		const struct linear_trunk_quantized_description description =
			trunk_quantized_of(1, WIDE_INPUTS, slice_count, ROWS);
		status = linear_trunk_quantized_prepare(
			&description, ARENA_BYTES, &made_up_kernels, launches);
	} else if (kind == TRUNK_BFLOAT16) {
		const struct linear_trunk_bfloat16_description description =
			trunk_bfloat16_of(slice_count, ROWS, 1);
		status = linear_trunk_bfloat16_prepare(
			&description, ARENA_BYTES, &made_up_kernels, launches);
	} else if (kind == ROUTED_GATE_UP || kind == ROUTED_DOWN) {
		const struct linear_routed_experts_description description =
			routed_of(kind == ROUTED_GATE_UP ? slice_count : 1,
				  kind == ROUTED_DOWN ? slice_count : 1);
		status = linear_routed_experts_prepare(
			&description, ARENA_BYTES, &made_up_kernels, launches);
		launches[0] = launches[kind == ROUTED_GATE_UP ?
					       LINEAR_ROUTED_EXPERTS_GATE_UP :
					       LINEAR_ROUTED_EXPERTS_DOWN];
	} else {
		const struct linear_shared_expert_description description =
			shared_of(kind == SHARED_GATE_UP ? slice_count : 1,
				  kind == SHARED_DOWN ? slice_count : 1, ROWS);
		status = linear_shared_expert_prepare(
			&description, ARENA_BYTES, &made_up_kernels, launches);
		launches[0] = launches[kind == SHARED_GATE_UP ?
					       LINEAR_SHARED_EXPERT_GATE_UP :
					       LINEAR_SHARED_EXPERT_DOWN];
	}
    *launch_out = launches[0];
    return status;
}

/* The handles of the instances a kind's product launches. */
static const uint32_t *instances_of(enum described kind)
{
    return kind == TRUNK_BFLOAT16                               ? made_up_kernels.trunk_linear_bfloat16
           : kind == ROUTED_GATE_UP                             ? made_up_kernels.expert_gate_up
           : kind == ROUTED_DOWN                                ? made_up_kernels.expert_down
                                                                : made_up_kernels.trunk_linear_quantized;
}

static void test_slice_counts(void)
{
    static const char *const kind_names[DESCRIBED_COUNT] = {
        "a trunk linear of quantized rows", "a trunk linear of bfloat16 rows", "gate and up of the routed experts",
        "down of the routed experts", "gate and up of the shared expert", "down of the shared expert"};
    for (uint32_t kind_index = 0; kind_index < DESCRIBED_COUNT; kind_index++) {
        const enum described kind = (enum described)kind_index;
	const uint32_t threads_per_slice =
		LINEAR_THREADS_PER_WARP *
		(kind == ROUTED_GATE_UP ? EXPERT_GATE_UP_MATRICES : 1u);
	context_text = kind_names[kind_index];
        for (uint32_t slice_count = 0; slice_count <= MOST_SLICES_TRIED; slice_count++) {
		struct kernel_launch launch;
		char logged[CAPTURE_CAPACITY];
		uint32_t instance = 0;
		while (instance < LINEAR_SLICE_INSTANCES &&
		       stamped_counts[instance] != slice_count) {
			instance++;
		}
	    capture_begin();
	    const result_status status =
		    prepare_with_slices(kind, slice_count, &launch);
	    capture_end(logged, sizeof(logged));
	    if (instance < LINEAR_SLICE_INSTANCES) {
		    expect(status == STATUS_OK && logged[0] == '\0' &&
				   launch.kernel ==
					   instances_of(kind)[instance] &&
				   launch.block[0] ==
					   slice_count * threads_per_slice &&
				   launch.dynamic_shared_byte_count ==
					   slice_count * threads_per_slice *
						   LINEAR_SHARED_BYTES_PER_THREAD,
			   "a stamped count is accepted and launches its instance with its block");
	    } else {
		    expect(status == STATUS_INVALID_ARGUMENT,
			   "a count with no instance is refused");
		    expect_one_error_line_with(logged, RULE_STAMPED,
					       "a count with no instance");
	    }
	}
    }
}

/* ---- two matrices in one trunk launch ---- */

/* A trunk description that must be refused: the status, one error line with the words, the launch untouched. */
static void
trunk_refused(const struct linear_trunk_quantized_description *description,
	      const char *rule, const char *what)
{
	struct kernel_launch launch;
	unsigned char untouched[sizeof(launch)];
	char logged[CAPTURE_CAPACITY];
	memset(&launch, UNTOUCHED_BYTE, sizeof(launch));
	memset(untouched, UNTOUCHED_BYTE, sizeof(untouched));
	context_text = what;
	capture_begin();
	const result_status status = linear_trunk_quantized_prepare(
		description, ARENA_BYTES, &made_up_kernels, &launch);
	capture_end(logged, sizeof(logged));
	expect(status == STATUS_INVALID_ARGUMENT, "refused");
	expect_one_error_line_with(logged, rule, what);
	expect(memcmp(&launch, untouched, sizeof(untouched)) == 0,
	       "a refusal leaves the launch untouched");
}

static void test_merged_matrices(void)
{
	struct kernel_launch launch;
	const uint32_t slice_count =
		stamped_counts[LINEAR_SLICE_INSTANCES - 1u];
	struct linear_trunk_quantized_description pair = trunk_quantized_of(
		LINEAR_MERGED_MATRICES, WIDE_INPUTS, slice_count, ROWS);
	const struct trunk_linear_quantized_arguments record = {
		{ pair.matrices[0].stored.byte_offset,
		  pair.matrices[1].stored.byte_offset },
		{ pair.outputs[0].byte_offset, pair.outputs[1].byte_offset },
		pair.codes.byte_offset,
		pair.exponents.byte_offset,
		pair.operands.byte_offset,
		{ FEW_OUTPUTS, FEW_OUTPUTS },
		WIDE_INPUTS,
		ROWS
	};
	context_text = "two matrices in one launch";
	expect(linear_trunk_quantized_prepare(&pair, ARENA_BYTES,
					      &made_up_kernels,
					      &launch) == STATUS_OK &&
		       launch.kernel == made_up_kernels.trunk_linear_quantized
						[LINEAR_SLICE_INSTANCES - 1u] &&
		       launch.grid[0] == (FEW_OUTPUTS + FEW_OUTPUTS) /
						 LINEAR_OUTPUTS_PER_TILE &&
		       launch.grid[1] == 1 &&
		       launch.argument_byte_count == sizeof(record) &&
		       memcmp(launch.argument, &record, sizeof(record)) == 0,
	       "a pair is accepted: one launch of the tiles of both, its record as linear_records.h states it");
	for (uint32_t matrix_count = 0;
	     matrix_count <= LINEAR_MERGED_MATRICES + 1u;
	     matrix_count += LINEAR_MERGED_MATRICES + 1u) {
		struct linear_trunk_quantized_description other = pair;
		other.matrix_count = matrix_count;
		trunk_refused(&other, RULE_MATRICES, "no matrix, and three");
	}
	struct linear_trunk_quantized_description other = pair;
	other.matrices[1] = trunk_quantized_of(LINEAR_MERGED_MATRICES,
					       OTHER_INPUTS, slice_count, ROWS)
				    .matrices[1];
	trunk_refused(&other, RULE_PAIR, "matrices of other input sizes");
	other = pair;
	other.matrices[1].slice_count = stamped_counts[0];
	trunk_refused(&other, RULE_PAIR, "matrices of other slice counts");
	other = pair;
	other.matrices[1].weight_format = E2M1;
	other.matrices[1].shape.bits_per_weight = E2M1_BITS_PER_WEIGHT;
	trunk_refused(&other, "its kernel reads format",
		      "a second matrix of the 4-bit format");
	other = pair;
	other.matrices[0].shape.output_size = HALF_OF_32_BITS_OF_OUTPUTS;
	other.matrices[1].shape.output_size = HALF_OF_32_BITS_OF_OUTPUTS;
	trunk_refused(&other, RULE_PAIR, "two matrices of 2^31 outputs each");
	other = pair;
	other.outputs[1].byte_offset =
		other.outputs[0].byte_offset + LINEAR_OFFSET_ALIGNMENT_BYTES;
	trunk_refused(&other, RULE_SHARED, "outputs that share a byte");
	other = pair;
	other.matrices[1].stored = other.matrices[0].stored;
	trunk_refused(&other, RULE_SHARED, "one matrix twice");
}

/* The matrices of the model by family: {family, outputs, inputs}. The first two of the trunk share a launch. */
enum { MATRIX_FAMILY, MATRIX_OUTPUTS, MATRIX_INPUTS, MATRIX_FIELDS };
enum { MERGED_FIRST = 2, MERGED_SECOND = 3 };   /* attn.wq_a and attn.wkv */
static const uint32_t model_matrices[][MATRIX_FIELDS] = {
	{ LINEAR_FAMILY_EXPERT_GATE_UP, 2304, 5120 },
	{ LINEAR_FAMILY_EXPERT_DOWN, 5120, 2304 },
	{ LINEAR_FAMILY_TRUNK_QUANTIZED, 1280, 5120 },
	{ LINEAR_FAMILY_TRUNK_QUANTIZED, 512, 5120 },
	{ LINEAR_FAMILY_TRUNK_QUANTIZED, 32768, 1280 },
	{ LINEAR_FAMILY_TRUNK_QUANTIZED, 4096, 1280 },
	{ LINEAR_FAMILY_TRUNK_QUANTIZED, 5120, 8192 },
	{ LINEAR_FAMILY_TRUNK_QUANTIZED, 25600, 6144 },
	{ LINEAR_FAMILY_TRUNK_QUANTIZED, 2304, 5120 },
	{ LINEAR_FAMILY_TRUNK_QUANTIZED, 5120, 2304 },
	{ LINEAR_FAMILY_TRUNK_BFLOAT16, 8192, 4096 }
};

/* A call that must be refused with one error line holding the words, the count untouched. */
static void default_refused(enum linear_profile_code profile,
			    enum linear_order_code order,
			    enum linear_family_code family,
			    const struct matrix_shape_record *shape,
			    uint32_t arithmetic_class, bool gives_count,
			    const char *words, const char *what)
{
    uint32_t slice_count = UNTOUCHED_BYTE;
    char logged[CAPTURE_CAPACITY];
    context_text = what;
    capture_begin();
    const result_status status = linear_default_slice_count(
	    profile, order, family, shape, arithmetic_class,
	    gives_count ? &slice_count : NULL);
    capture_end(logged, sizeof(logged));
    expect(status == STATUS_INVALID_ARGUMENT && slice_count == UNTOUCHED_BYTE,
	   "refused, the count untouched");
    expect_one_error_line_with(logged, words, what);
}

static void test_slice_defaults(void)
{
	static const uint32_t classes[] = { ARITHMETIC_CLASS_SM_80,
					    ARITHMETIC_CLASS_SM_120 };
	const struct matrix_shape_record unknown = { FEW_OUTPUTS,
						     LINEAR_VALUES_PER_LOAD,
						     E4M3_BITS_PER_WEIGHT, 0 };
	context_text = "the default slice counts";
	for (uint32_t index = 0;
	     index <
	     COUNT_OF(model_matrices) * COUNT_OF(classes) * PROFILE_COUNT;
	     index++) {
		const uint32_t *const matrix =
			model_matrices[index /
				       (COUNT_OF(classes) * PROFILE_COUNT)];
		const uint32_t arithmetic_class =
			classes[index / PROFILE_COUNT % COUNT_OF(classes)];
		const enum linear_profile_code profile =
			(enum linear_profile_code)(index % PROFILE_COUNT);
		const enum linear_family_code family =
			(enum linear_family_code)matrix[MATRIX_FAMILY];
		const struct matrix_shape_record shape = {
			matrix[MATRIX_OUTPUTS], matrix[MATRIX_INPUTS], 0, 0
		};
		uint32_t fixed = 0;
		uint32_t reference = 0;
		uint32_t instance = 0;
		expect(linear_default_slice_count(
			       profile, LINEAR_ORDER_FIXED, family, &shape,
			       arithmetic_class, &fixed) == STATUS_OK &&
			       linear_default_slice_count(
				       profile, LINEAR_ORDER_REFERENCE, family,
				       &shape, arithmetic_class,
				       &reference) == STATUS_OK &&
			       reference == REFERENCE_SLICES(fixed) &&
			       fixed <= shape.input_size /
						LINEAR_VALUES_PER_LOAD,
		       "a matrix of the model has a count of its loads in its fixed order, and the reference's count");
		while (instance < LINEAR_SLICE_INSTANCES &&
		       stamped_counts[instance] != fixed) {
			instance++;
		}
		expect(instance < LINEAR_SLICE_INSTANCES,
		       "the count of a row is one a kernel is stamped for");
		if (index / (COUNT_OF(classes) * PROFILE_COUNT) ==
		    MERGED_FIRST) {
			const uint32_t *const second =
				model_matrices[MERGED_SECOND];
			const struct matrix_shape_record second_shape = {
				second[MATRIX_OUTPUTS], second[MATRIX_INPUTS],
				0, 0
			};
			uint32_t second_count = 0;
			expect(linear_default_slice_count(
				       profile, LINEAR_ORDER_FIXED,
				       (enum linear_family_code)
					       second[MATRIX_FAMILY],
				       &second_shape, arithmetic_class,
				       &second_count) == STATUS_OK &&
				       second_count == fixed,
			       "attn.wq_a and attn.wkv, one launch, have one count in every class and profile");
		}
    }
    const struct matrix_shape_record known = {
	    model_matrices[0][MATRIX_OUTPUTS], model_matrices[0][MATRIX_INPUTS],
	    E2M1_BITS_PER_WEIGHT, 0
    };
    const enum linear_family_code known_family =
	    (enum linear_family_code)model_matrices[0][MATRIX_FAMILY];
    default_refused(LINEAR_PROFILE_SPEED, LINEAR_ORDER_FIXED, known_family,
		    &unknown, ARITHMETIC_CLASS_SM_80, true, RULE_NO_ROW,
		    "a shape with no row");
    default_refused(LINEAR_PROFILE_PRECISION, LINEAR_ORDER_REFERENCE,
		    known_family, &unknown, ARITHMETIC_CLASS_SM_120, true,
		    RULE_NO_ROW, "a shape with no row, the reference's order");
    default_refused(LINEAR_PROFILE_SPEED, LINEAR_ORDER_FIXED,
		    LINEAR_FAMILY_EXPERT_DOWN, &known, ARITHMETIC_CLASS_SM_80,
		    true, RULE_NO_ROW, "a shape of another family");
    default_refused(LINEAR_PROFILE_SPEED, LINEAR_ORDER_FIXED, known_family,
		    &known, 0, true, RULE_NO_ROW, "a card no build runs on");
    default_refused(PROFILE_COUNT, LINEAR_ORDER_FIXED, known_family, &known,
		    ARITHMETIC_CLASS_SM_80, true, RULE_NOT_KNOWN,
		    "a profile not known");
    default_refused(LINEAR_PROFILE_SPEED, LINEAR_ORDER_REFERENCE + 1,
		    known_family, &known, ARITHMETIC_CLASS_SM_80, true,
		    RULE_NOT_KNOWN, "an order not known");
    default_refused(LINEAR_PROFILE_SPEED, LINEAR_ORDER_FIXED,
		    LINEAR_FAMILY_TRUNK_BFLOAT16 + 1, &known,
		    ARITHMETIC_CLASS_SM_80, true, RULE_NOT_KNOWN,
		    "a family not known");
    default_refused(LINEAR_PROFILE_SPEED, LINEAR_ORDER_FIXED, known_family,
		    NULL, ARITHMETIC_CLASS_SM_80, true, RULE_NOT_KNOWN,
		    "no shape");
    default_refused(LINEAR_PROFILE_SPEED, LINEAR_ORDER_FIXED, known_family,
		    &known, ARITHMETIC_CLASS_SM_80, false, RULE_NOT_KNOWN,
		    "nowhere to put the count");
}

/* The wide launches of the trunk matrices of model_matrices (from MERGED_FIRST on) in the classes of wide_classes, by
 * hand from the table of linear.c: {slice count, form}; class 120 native keeps the default count. */
static const uint32_t wide_classes[] = { ARITHMETIC_CLASS_SM_80,
					 ARITHMETIC_CLASS_SM_120,
					 ARITHMETIC_CLASS_SM_120_NATIVE };
enum { WIDE_COUNT_FIELD, WIDE_FORM_FIELD, WIDE_FIELDS };
#define WIDE_CLASS_COUNT 3u
static const uint32_t wide_choices[][WIDE_CLASS_COUNT][WIDE_FIELDS] = {
    {{1, INTERLEAVED_CHUNKS}, {4, GROUPS}, {16, GROUPS}},       /* attn.wq_a */
    {{1, INTERLEAVED_CHUNKS}, {4, GROUPS}, {16, GROUPS}},       /* attn.wkv */
    {{1, INTERLEAVED_TILES}, {4, GROUPS}, {2, GROUPS}},         /* attn.wq_b */
    {{8, GROUPS}, {4, GROUPS}, {8, GROUPS}},                    /* attn.indexer.wq_b */
    {{4, GROUPS}, {8, GROUPS}, {8, GROUPS}},                    /* attn.wo_b */
    {{1, INTERLEAVED_CHUNKS}, {4, GROUPS}, {16, GROUPS}},       /* engram.wkv */
    {{8, GROUPS}, {8, GROUPS}, {16, GROUPS}},                   /* the shared expert's gate and up */
    {{4, GROUPS}, {4, GROUPS}, {4, GROUPS}},                    /* its down */
    {{1, GROUPS}, {1, GROUPS}, {8, TILES}},                     /* attn.wo_a */
};
_Static_assert(COUNT_OF(wide_classes) == WIDE_CLASS_COUNT && COUNT_OF(wide_choices) + MERGED_FIRST ==
                   COUNT_OF(model_matrices),
               "a wide choice for each class and trunk matrix");

/* The draft's main_proj in a class it runs in, by hand from the table of linear.c. */
struct draft_row {
    uint32_t arithmetic_class, slice_count, wide_count;
    enum linear_wide_form wide_form;
};

/* The draft's rows, then the refusal of class 120's emulation (dspark_v1/DESIGN.md decision 7: the draft runs in its
 * card's own class). */
static void test_draft_main_projection(void)
{
	static const struct draft_row draft_rows[] = {
		{ ARITHMETIC_CLASS_SM_80, 4, 1, INTERLEAVED_CHUNKS },
		{ ARITHMETIC_CLASS_SM_120_NATIVE, 16, 16, GROUPS }
	};
	const struct matrix_shape_record shape = { MODEL_HIDDEN,
						   MODEL_DRAFT_FEATURES, 0, 0 };
	context_text = "the draft's main_proj";
	for (uint32_t index = 0; index < COUNT_OF(draft_rows) * PROFILE_COUNT;
	     index++) {
		const uint32_t row = index / PROFILE_COUNT;
		const enum linear_profile_code profile =
			(enum linear_profile_code)(index % PROFILE_COUNT);
		uint32_t slice_count = 0, wide_count = 0;
		enum linear_wide_form form = LINEAR_WIDE_FORM_COUNT;
		expect(linear_default_slice_count(
			       profile, FIXED, LINEAR_FAMILY_TRUNK_QUANTIZED,
			       &shape, draft_rows[row].arithmetic_class,
			       &slice_count) == STATUS_OK &&
			       linear_default_wide_form(
				       profile, FIXED,
				       LINEAR_FAMILY_TRUNK_QUANTIZED, &shape,
				       draft_rows[row].arithmetic_class,
				       &wide_count, &form) == STATUS_OK &&
			       slice_count == draft_rows[row].slice_count &&
			       wide_count == draft_rows[row].wide_count &&
			       form == draft_rows[row].wide_form,
		       "its measured count and wide form in each class it runs in");
	}
	default_refused(LINEAR_PROFILE_SPEED, FIXED,
			LINEAR_FAMILY_TRUNK_QUANTIZED, &shape,
			ARITHMETIC_CLASS_SM_120, true, RULE_NO_ROW,
			"no row in class 120's emulation");
}

/* The wide forms: every trunk matrix of the model in every class and profile, and the refusals. */
static void test_wide_forms_of_matrices(void)
{
    context_text = "the wide forms of the model's matrices";
    for (uint32_t index = 0; index < COUNT_OF(wide_choices) * WIDE_CLASS_COUNT * PROFILE_COUNT; index++) {
        const uint32_t matrix_index = index / (WIDE_CLASS_COUNT * PROFILE_COUNT), class_index =
            index / PROFILE_COUNT % WIDE_CLASS_COUNT;
        const uint32_t *const matrix = model_matrices[MERGED_FIRST + matrix_index];
        const uint32_t *const choice = wide_choices[matrix_index][class_index];
	const enum linear_profile_code profile =
		(enum linear_profile_code)(index % PROFILE_COUNT);
	const enum linear_family_code family =
		(enum linear_family_code)matrix[MATRIX_FAMILY];
	const struct matrix_shape_record shape = { matrix[MATRIX_OUTPUTS],
						   matrix[MATRIX_INPUTS], 0,
						   0 };
	const uint32_t arithmetic_class = wide_classes[class_index];
        uint32_t slice_count = 0, default_count = 0;
	enum linear_wide_form form = LINEAR_WIDE_FORM_COUNT;
	expect(linear_default_wide_form(profile, FIXED, family, &shape,
					arithmetic_class, &slice_count,
					&form) == STATUS_OK &&
		       linear_default_slice_count(
			       profile, FIXED, family, &shape, arithmetic_class,
			       &default_count) == STATUS_OK &&
		       slice_count == choice[WIDE_COUNT_FIELD] &&
		       (uint32_t)form == choice[WIDE_FORM_FIELD] &&
		       (arithmetic_class != ARITHMETIC_CLASS_SM_120_NATIVE ||
			slice_count == default_count),
	       "a trunk matrix's wide count and form are the table's, class 120 native's count its default");
    }
    const struct matrix_shape_record trunk = {
	    model_matrices[MERGED_FIRST][MATRIX_OUTPUTS],
	    model_matrices[MERGED_FIRST][MATRIX_INPUTS], 0, 0
    };
    const struct matrix_shape_record expert = {
	    model_matrices[0][MATRIX_OUTPUTS], model_matrices[0][MATRIX_INPUTS],
	    0, 0
    };
    const struct matrix_shape_record unknown = { FEW_OUTPUTS,
						 LINEAR_VALUES_PER_LOAD, 0, 0 };
    const struct {
	    const struct matrix_shape_record *shape;
	    const char *rule, *what;
	    enum linear_family_code family;
	    bool gives_count, gives_form;
    } refused[] = {
	    { &expert, RULE_NOT_TRUNK, "an expert's matrix",
	      LINEAR_FAMILY_EXPERT_GATE_UP, true, true },
	    { &trunk, RULE_NOT_TRUNK, "nowhere to put the count",
	      LINEAR_FAMILY_TRUNK_QUANTIZED, false, true },
	    { &trunk, RULE_NOT_TRUNK, "nowhere to put the form",
	      LINEAR_FAMILY_TRUNK_QUANTIZED, true, false },
	    { &unknown, RULE_NO_ROW, "a shape with no row",
	      LINEAR_FAMILY_TRUNK_QUANTIZED, true, true },
    };
    for (uint32_t index = 0; index < COUNT_OF(refused); index++) {
        uint32_t slice_count = UNTOUCHED_BYTE;
	enum linear_wide_form form = LINEAR_WIDE_FORM_COUNT;
	char logged[CAPTURE_CAPACITY];
        context_text = refused[index].what;
        capture_begin();
	const result_status status = linear_default_wide_form(
		SPEED, FIXED, refused[index].family, refused[index].shape,
		CLASS_80, refused[index].gives_count ? &slice_count : NULL,
		refused[index].gives_form ? &form : NULL);
	capture_end(logged, sizeof(logged));
	expect(status == STATUS_INVALID_ARGUMENT &&
		       slice_count == UNTOUCHED_BYTE &&
		       form == LINEAR_WIDE_FORM_COUNT,
	       "refused, the count and the form untouched");
	expect_one_error_line_with(logged, refused[index].rule, refused[index].what);
    }
}

/* ---- wide trunk launches ---- */

/* The kinds of a wide case: a trunk launch of one matrix, of two, of bfloat16 rows, and the shared expert's chain. */
enum wide_kind { WIDE_ONE_MATRIX, WIDE_TWO_MATRICES, WIDE_BFLOAT16, WIDE_SHARED, WIDE_KINDS };

/* One description of a kind for some rows, and the ranges of it whose bytes grow with the rows. */
struct wide_case {
    enum wide_kind kind;
    struct linear_trunk_quantized_description quantized;
    struct linear_trunk_bfloat16_description bfloat16;
    struct linear_shared_expert_description shared;
    struct linear_range *scaled[MOST_SCALED_RANGES];
    uint32_t scaled_count;
};

/* The ranges of a case that grow with the rows: count of them. */
static void wide_case_scaled(struct wide_case *wide,
			     struct linear_range *const *scaled, uint32_t count)
{
    for (uint32_t range_index = 0; range_index < count; range_index++) {
        wide->scaled[range_index] = scaled[range_index];
    }
    wide->scaled_count = count;
}

/* Lays a case out, every range of exactly the bytes its launches touch, the ranges that grow with the rows last. */
static void wide_case_make(struct wide_case *wide, enum wide_kind kind, uint32_t row_count)
{
    memset(wide, 0, sizeof(*wide));
    wide->kind = kind;
    if (kind == WIDE_BFLOAT16) {
        wide->bfloat16 = trunk_bfloat16_of(WIDE_SLICES, row_count, WIDE_SEGMENTS);
	struct linear_range *const scaled[] = { &wide->bfloat16.values,
						&wide->bfloat16.output };
	wide_case_scaled(wide, scaled, COUNT_OF(scaled));
    } else if (kind == WIDE_SHARED) {
	    struct linear_shared_expert_description *const shared =
		    &wide->shared;
	    *shared = shared_of(WIDE_SLICES, WIDE_DOWN_SLICES, row_count);
	    struct linear_range *const scaled[] = {
		    &shared->codes,	      &shared->exponents,
		    &shared->operands,	      &shared->gate_output,
		    &shared->up_output,	      &shared->hidden,
		    &shared->hidden_codes,    &shared->hidden_exponents,
		    &shared->hidden_operands, &shared->output
	    };
	    wide_case_scaled(wide, scaled, COUNT_OF(scaled));
    } else {
	    const uint32_t matrix_count =
		    kind == WIDE_TWO_MATRICES ? LINEAR_MERGED_MATRICES : 1u;
	    wide->quantized = trunk_quantized_of(matrix_count, WIDE_INPUTS,
						 WIDE_SLICES, row_count);
	    struct linear_range *const scaled[] = {
		    &wide->quantized.codes, &wide->quantized.exponents,
		    &wide->quantized.operands, &wide->quantized.outputs[0],
		    &wide->quantized.outputs[1]
	    };
	    wide_case_scaled(wide, scaled,
			     (uint32_t)COUNT_OF(scaled) -
				     (LINEAR_MERGED_MATRICES - matrix_count));
    }
}

/* The byte behind the last range of a case: the least arena that holds it. */
static uint64_t wide_case_end(const struct wide_case *wide)
{
    uint64_t end = 0;
    for (uint32_t range_index = 0; range_index < wide->scaled_count; range_index++) {
        const uint64_t range_end = wide->scaled[range_index]->byte_offset + wide->scaled[range_index]->byte_count;
        end = range_end > end ? range_end : end;
    }
    return end;
}

MUST_CHECK static result_status
wide_case_prepare(const struct wide_case *wide, uint64_t arena_byte_count,
		  const struct linear_kernels_record *kernels,
		  struct kernel_launch *launches_out)
{
	return wide->kind == WIDE_BFLOAT16 ?
		       linear_trunk_bfloat16_prepare(&wide->bfloat16,
						     arena_byte_count, kernels,
						     launches_out) :
	       wide->kind == WIDE_SHARED ?
		       linear_shared_expert_prepare(&wide->shared,
						    arena_byte_count, kernels,
						    launches_out) :
		       linear_trunk_quantized_prepare(&wide->quantized,
						      arena_byte_count, kernels,
						      launches_out);
}

/* A case that must be refused: the status, one error line with the words, the launches untouched. */
static void wide_case_refused(const struct wide_case *wide,
			      uint64_t arena_byte_count,
			      const struct linear_kernels_record *kernels,
			      const char *rule, const char *what)
{
	struct kernel_launch launches[MOST_LAUNCHES];
	unsigned char untouched[sizeof(launches)];
	char logged[CAPTURE_CAPACITY];
	memset(launches, UNTOUCHED_BYTE, sizeof(launches));
	memset(untouched, UNTOUCHED_BYTE, sizeof(untouched));
	capture_begin();
	const result_status status =
		wide_case_prepare(wide, arena_byte_count, kernels, launches);
	capture_end(logged, sizeof(logged));
	expect(status == STATUS_INVALID_ARGUMENT, what);
	expect_one_error_line_with(logged, rule, what);
	expect(memcmp(launches, untouched, sizeof(untouched)) == 0,
	       "a refusal leaves the launches untouched");
}

/* The place of a stamped count in slice_instances. */
static uint32_t instance_of(uint32_t slice_count)
{
    uint32_t instance = 0;
    while (instance < LINEAR_SLICE_INSTANCES &&
	   stamped_counts[instance] != slice_count) {
	    instance++;
    }
    return instance;
}

/* The launch of a trunk product is of `kernel`, has the block and shared memory of its thin instance for slice_count
 * slices, a block for each tile of output_size outputs (above 8 rows in its form one for each group of 8 rows, of the
 * interleaved instance each chunk of 32, of each tile, or one a tile), and holds `record`. */
static void expect_form_launch(const struct kernel_launch *launch,
			       uint32_t kernel, uint32_t output_size,
			       uint32_t slice_count, uint32_t row_count,
			       enum linear_wide_form form, const void *record,
			       size_t record_byte_count, const char *what)
{
	const uint32_t block_threads = slice_count * LINEAR_THREADS_PER_WARP;
	const uint32_t unit_rows = form >= INTERLEAVED_CHUNKS ?
					   LINEAR_INTERLEAVED_ROWS :
					   LINEAR_MAXIMUM_ROWS;
	const uint32_t tile_blocks =
		row_count <= LINEAR_MAXIMUM_ROWS || form == TILES ||
				form == INTERLEAVED_TILES ?
			1u :
			(row_count + unit_rows - 1u) / unit_rows;
	expect(launch->kernel == kernel &&
		       launch->grid[0] == output_size /
						  LINEAR_OUTPUTS_PER_TILE *
						  tile_blocks &&
		       launch->grid[1] == 1 &&
		       launch->block[0] == block_threads &&
		       launch->dynamic_shared_byte_count ==
			       block_threads * LINEAR_SHARED_BYTES_PER_THREAD &&
		       launch->argument_byte_count == record_byte_count &&
		       memcmp(launch->argument, record, record_byte_count) == 0,
	       what);
}
static void expect_trunk_launch(const struct kernel_launch *launch,
				uint32_t kernel, uint32_t output_size,
				uint32_t slice_count, uint32_t row_count,
				const void *record, size_t record_byte_count,
				const char *what)
{
    expect_form_launch(launch, kernel, output_size, slice_count, row_count, GROUPS, record, record_byte_count, what);
}

/* The launches of an accepted case: its trunk products of the instances for its rows (wide above 8), with the
 * thin block and shared memory, a block a group of 8 rows of each tile and a record of those rows; the shared
 * expert's other two of the rows. */
static void expect_wide_launches(const struct wide_case *wide,
				 const struct kernel_launch *launches)
{
	const bool is_wide =
		wide->kind == WIDE_SHARED ?
			wide->shared.row_count > LINEAR_MAXIMUM_ROWS :
		wide->kind == WIDE_BFLOAT16 ?
			wide->bfloat16.row_count > LINEAR_MAXIMUM_ROWS :
			wide->quantized.row_count > LINEAR_MAXIMUM_ROWS;
	const uint32_t *const quantized =
		is_wide ? made_up_kernels.trunk_linear_quantized_wide :
			  made_up_kernels.trunk_linear_quantized;
	const uint32_t *const bfloat16 =
		is_wide ? made_up_kernels.trunk_linear_bfloat16_wide :
			  made_up_kernels.trunk_linear_bfloat16;
	if (wide->kind == WIDE_BFLOAT16) {
		const struct linear_trunk_bfloat16_description *const of =
			&wide->bfloat16;
		const struct trunk_linear_bfloat16_arguments record = {
			of->matrix.stored.byte_offset,
			of->values.byte_offset,
			of->output.byte_offset,
			FEW_OUTPUTS,
			WIDE_INPUTS,
			of->row_count,
			of->segment_count
		};
		expect_trunk_launch(
			&launches[0], bfloat16[instance_of(WIDE_SLICES)],
			FEW_OUTPUTS, WIDE_SLICES, of->row_count, &record,
			sizeof(record),
			"bfloat16 rows: the instance, grid, block and record");
	} else if (wide->kind == WIDE_SHARED) {
		const struct linear_shared_expert_description *const of =
			&wide->shared;
		const struct trunk_linear_quantized_arguments gate_up_record = {
			{ of->gate.stored.byte_offset,
			  of->up.stored.byte_offset },
			{ of->gate_output.byte_offset,
			  of->up_output.byte_offset },
			of->codes.byte_offset,
			of->exponents.byte_offset,
			of->operands.byte_offset,
			{ WIDE_INPUTS, WIDE_INPUTS },
			WIDE_INPUTS,
			of->row_count
		};
		const struct trunk_linear_quantized_arguments down_record = {
			{ of->down.stored.byte_offset, 0 },
			{ of->output.byte_offset, 0 },
			of->hidden_codes.byte_offset,
			of->hidden_exponents.byte_offset,
			of->hidden_operands.byte_offset,
			{ WIDE_INPUTS, 0 },
			WIDE_INPUTS,
			of->row_count
		};
		expect_trunk_launch(
			&launches[LINEAR_SHARED_EXPERT_GATE_UP],
			quantized[instance_of(WIDE_SLICES)],
			LINEAR_MERGED_MATRICES * WIDE_INPUTS, WIDE_SLICES,
			of->row_count, &gate_up_record, sizeof(gate_up_record),
			"the shared expert's gate and up: instance, grid, block, record");
		expect_trunk_launch(
			&launches[LINEAR_SHARED_EXPERT_DOWN],
			quantized[instance_of(WIDE_DOWN_SLICES)], WIDE_INPUTS,
			WIDE_DOWN_SLICES, of->row_count, &down_record,
			sizeof(down_record),
			"the shared expert's down: its own count's instance, grid, block and record");
		expect(launches[LINEAR_SHARED_EXPERT_HIDDEN].grid[1] ==
				       of->row_count &&
			       launches[LINEAR_SHARED_EXPERT_QUANTIZE].grid[1] ==
				       of->row_count,
		       "the hidden step and the quantization take every row");
	} else {
		const struct linear_trunk_quantized_description *const of =
			&wide->quantized;
		const bool is_merged = wide->kind == WIDE_TWO_MATRICES;
		const struct trunk_linear_quantized_arguments record = {
			{ of->matrices[0].stored.byte_offset,
			  is_merged ? of->matrices[1].stored.byte_offset : 0 },
			{ of->outputs[0].byte_offset,
			  is_merged ? of->outputs[1].byte_offset : 0 },
			of->codes.byte_offset,
			of->exponents.byte_offset,
			of->operands.byte_offset,
			{ FEW_OUTPUTS, is_merged ? FEW_OUTPUTS : 0 },
			WIDE_INPUTS,
			of->row_count
		};
		expect_trunk_launch(
			&launches[0], quantized[instance_of(WIDE_SLICES)],
			is_merged ? LINEAR_MERGED_MATRICES * FEW_OUTPUTS :
				    FEW_OUTPUTS,
			WIDE_SLICES, of->row_count, &record, sizeof(record),
			"quantized rows: the instance, grid, block and record");
	}
}

static void test_wide_rows(void)
{
    static const char *const kind_names[WIDE_KINDS] = {
        "a wide trunk launch of one matrix", "a wide trunk launch of two matrices",
        "a wide trunk launch of bfloat16 rows in two segments", "the wide shared expert"};
    static const uint32_t row_counts[] = { LINEAR_MAXIMUM_ROWS + 1u,
					   WIDE_CHUNK_ROWS, UNEVEN_WIDE_ROWS,
					   LINEAR_WIDE_MOST_ROWS };
    static struct wide_case wide;               /* its ranges point into it */
    struct kernel_launch launches[MOST_LAUNCHES];
    struct linear_kernels_record thin_only = made_up_kernels;
    uint32_t handles[KERNEL_COUNT];
    char logged[CAPTURE_CAPACITY];
    memcpy(handles, &thin_only, sizeof(handles));
    for (uint32_t kernel_index = THIN_KERNEL_COUNT; kernel_index < KERNEL_COUNT; kernel_index++) {
	    handles[kernel_index] = BACKEND_NO_KERNEL;
    }
    memcpy(&thin_only, handles, sizeof(thin_only));
    for (uint32_t case_index = 0; case_index < WIDE_KINDS * COUNT_OF(row_counts); case_index++) {
        const enum wide_kind kind = (enum wide_kind)(case_index / COUNT_OF(row_counts));
        const uint32_t row_count = row_counts[case_index % COUNT_OF(row_counts)];
        context_text = kind_names[kind];
        wide_case_make(&wide, kind, row_count);
        const uint64_t arena_byte_count = wide_case_end(&wide);
        capture_begin();
	const result_status status = wide_case_prepare(
		&wide, arena_byte_count, &made_up_kernels, launches);
	capture_end(logged, sizeof(logged));
	expect(status == STATUS_OK && logged[0] == '\0',
	       "accepted, without a word, in the least arena");
	if (status == STATUS_OK) {
		expect_wide_launches(&wide, launches);
	}
	for (uint32_t range_index = 0; range_index < wide.scaled_count; range_index++) {
            wide.scaled[range_index]->byte_count -= 1u;
            wide_case_refused(&wide, arena_byte_count, &made_up_kernels, RULE_TOUCHED,
                              "a range that grows with the rows, one byte short of them");
            wide.scaled[range_index]->byte_count += 1u;
        }
        wide_case_refused(&wide, arena_byte_count - 1u, &made_up_kernels, RULE_ARENA,
                          "an arena one byte short of the last range");
        wide_case_refused(&wide, arena_byte_count, &thin_only, RULE_NOT_PREPARED,
                          "wide rows where the wide instances are not prepared");
    }
    for (uint32_t kind_index = 0; kind_index < WIDE_KINDS; kind_index++) {
        const enum wide_kind kind = (enum wide_kind)kind_index;
        uint32_t *const row_count = kind == WIDE_SHARED      ? &wide.shared.row_count
                                    : kind == WIDE_BFLOAT16 ? &wide.bfloat16.row_count
                                                            : &wide.quantized.row_count;
        context_text = kind_names[kind];
	wide_case_make(&wide, kind, LINEAR_WIDE_MOST_ROWS);
	*row_count = LINEAR_WIDE_MOST_ROWS + 1u;
	wide_case_refused(&wide, ARENA_BYTES, &made_up_kernels, RULE_ROWS, "one row more than a wide launch takes");
        *row_count = 0;
        wide_case_refused(&wide, ARENA_BYTES, &made_up_kernels, RULE_ROWS, "no row");
        /* 8 rows: the thin instance and its record as they were, the wide instances never asked for */
	wide_case_make(&wide, kind, LINEAR_MAXIMUM_ROWS);
	capture_begin();
	const result_status status = wide_case_prepare(
		&wide, wide_case_end(&wide), &thin_only, launches);
	capture_end(logged, sizeof(logged));
	expect(status == STATUS_OK && logged[0] == '\0',
	       "8 rows are accepted where the wide instances are not prepared");
	if (status == STATUS_OK) {
		expect_wide_launches(&wide, launches);
	}
    }
    /* a grid of a block for each group of each tile would pass 32 bits: the most blocks a tile that 32 bits count */
    context_text = "a wide trunk launch of a matrix of 2^25 + 1 tiles";
    uint64_t cursor = LINEAR_OFFSET_ALIGNMENT_BYTES;
    const uint32_t huge_tiles = HUGE_OUTPUTS / LINEAR_OUTPUTS_PER_TILE;
    struct linear_trunk_quantized_description huge = {
	    .matrix_count = 1, .row_count = LINEAR_WIDE_MOST_ROWS
    };
    huge.matrices[0] = next_matrix(&cursor, E4M3, HUGE_OUTPUTS, WIDE_INPUTS, WIDE_SLICES);
    huge.codes = next_codes(&cursor, LINEAR_WIDE_MOST_ROWS, WIDE_INPUTS);
    huge.exponents =
	    next_exponents(&cursor, LINEAR_WIDE_MOST_ROWS, WIDE_INPUTS);
    huge.operands = next_values(&cursor, LINEAR_WIDE_MOST_ROWS, WIDE_INPUTS);
    huge.outputs[0] = next_values(&cursor, LINEAR_WIDE_MOST_ROWS, HUGE_OUTPUTS);
    expect(linear_trunk_quantized_prepare(&huge, ARENA_BYTES, &made_up_kernels,
					  launches) == STATUS_OK &&
		   launches[0].grid[0] ==
			   huge_tiles * (UINT32_MAX / huge_tiles) &&
		   (uint64_t)huge_tiles * (LINEAR_WIDE_MOST_ROWS /
					   LINEAR_MAXIMUM_ROWS) >
			   UINT32_MAX,
	   "accepted, with as many blocks a tile as 32 bits count where one for each group would pass them");
}

/* The other wide forms (a block a tile; the interleaved instance of one slice, a block a chunk or a tile) at every
 * wide row count, for one matrix, two, bfloat16 rows and the shared expert; 8 rows the thin launch whatever the form;
 * the refusals of a form. */
static void test_wide_forms(void)
{
	static const uint32_t row_counts[] = { LINEAR_MAXIMUM_ROWS + 1u,
					       WIDE_CHUNK_ROWS,
					       UNEVEN_WIDE_ROWS,
					       LINEAR_WIDE_MOST_ROWS };
	const uint32_t *const wide =
		made_up_kernels.trunk_linear_quantized_wide;
	const uint32_t interleaved =
		made_up_kernels.trunk_linear_quantized_interleaved;
	struct kernel_launch launches[MOST_LAUNCHES];
	uint32_t handles[KERNEL_COUNT];
	for (uint32_t index = 0; index < COUNT_OF(row_counts); index++) {
		const uint32_t rows = row_counts[index],
			       one_slice = stamped_counts[0];
		struct linear_trunk_quantized_description tiled =
			trunk_quantized_of(1, WIDE_INPUTS, WIDE_SLICES, rows);
		struct linear_trunk_quantized_description chunked =
			trunk_quantized_of(1, WIDE_INPUTS, one_slice, rows);
		struct linear_trunk_quantized_description pair =
			trunk_quantized_of(LINEAR_MERGED_MATRICES, WIDE_INPUTS,
					   one_slice, rows);
		struct linear_trunk_bfloat16_description values =
			trunk_bfloat16_of(WIDE_SLICES, rows, WIDE_SEGMENTS);
		struct linear_shared_expert_description shared =
			shared_of(one_slice, WIDE_DOWN_SLICES, rows);
		tiled.matrices[0].wide_form = TILES;
		chunked.matrices[0].wide_form = INTERLEAVED_CHUNKS;
		pair.matrices[0].wide_form = INTERLEAVED_TILES;
		pair.matrices[1].wide_form = INTERLEAVED_TILES;
		values.matrix.wide_form = TILES;
		shared.gate.wide_form = INTERLEAVED_CHUNKS;
		shared.up.wide_form = INTERLEAVED_CHUNKS;
		shared.down.wide_form = TILES;
		context_text = "the other wide forms";
		const struct trunk_linear_quantized_arguments tiled_record = {
			{ tiled.matrices[0].stored.byte_offset, 0 },
			{ tiled.outputs[0].byte_offset, 0 },
			tiled.codes.byte_offset,
			tiled.exponents.byte_offset,
			tiled.operands.byte_offset,
			{ FEW_OUTPUTS, 0 },
			WIDE_INPUTS,
			rows
		};
		expect(linear_trunk_quantized_prepare(&tiled, ARENA_BYTES,
						      &made_up_kernels,
						      launches) == STATUS_OK,
		       "a block a tile is accepted");
		expect_form_launch(
			&launches[0], wide[instance_of(WIDE_SLICES)],
			FEW_OUTPUTS, WIDE_SLICES, rows, TILES, &tiled_record,
			sizeof(tiled_record),
			"a block a tile: the wide instance, a block a tile");
		const struct trunk_linear_quantized_arguments chunked_record = {
			{ chunked.matrices[0].stored.byte_offset, 0 },
			{ chunked.outputs[0].byte_offset, 0 },
			chunked.codes.byte_offset,
			chunked.exponents.byte_offset,
			chunked.operands.byte_offset,
			{ FEW_OUTPUTS, 0 },
			WIDE_INPUTS,
			rows
		};
		expect(linear_trunk_quantized_prepare(&chunked, ARENA_BYTES,
						      &made_up_kernels,
						      launches) == STATUS_OK,
		       "the interleaved instance, a block a chunk, is accepted");
		expect_form_launch(
			&launches[0], interleaved, FEW_OUTPUTS, one_slice, rows,
			INTERLEAVED_CHUNKS, &chunked_record,
			sizeof(chunked_record),
			"the interleaved instance: the thin block of one slice, a block a chunk of 32 rows");
		const struct trunk_linear_quantized_arguments pair_record = {
			{ pair.matrices[0].stored.byte_offset,
			  pair.matrices[1].stored.byte_offset },
			{ pair.outputs[0].byte_offset,
			  pair.outputs[1].byte_offset },
			pair.codes.byte_offset,
			pair.exponents.byte_offset,
			pair.operands.byte_offset,
			{ FEW_OUTPUTS, FEW_OUTPUTS },
			WIDE_INPUTS,
			rows
		};
		expect(linear_trunk_quantized_prepare(&pair, ARENA_BYTES,
						      &made_up_kernels,
						      launches) == STATUS_OK,
		       "two matrices of the interleaved instance, a block a tile, are accepted");
		expect_form_launch(
			&launches[0], interleaved,
			LINEAR_MERGED_MATRICES * FEW_OUTPUTS, one_slice, rows,
			INTERLEAVED_TILES, &pair_record, sizeof(pair_record),
			"two matrices of the interleaved instance: the tiles of both, a block a tile");
		const struct trunk_linear_bfloat16_arguments values_record = {
			values.matrix.stored.byte_offset,
			values.values.byte_offset,
			values.output.byte_offset,
			FEW_OUTPUTS,
			WIDE_INPUTS,
			rows,
			WIDE_SEGMENTS
		};
		expect(linear_trunk_bfloat16_prepare(&values, ARENA_BYTES,
						     &made_up_kernels,
						     launches) == STATUS_OK,
		       "bfloat16 rows a block a tile are accepted");
		expect_form_launch(
			&launches[0],
			made_up_kernels.trunk_linear_bfloat16_wide[instance_of(
				WIDE_SLICES)],
			FEW_OUTPUTS, WIDE_SLICES, rows, TILES, &values_record,
			sizeof(values_record),
			"bfloat16 rows a block a tile: the wide instance, a block a tile");
		const struct trunk_linear_quantized_arguments gate_up_record = {
			{ shared.gate.stored.byte_offset,
			  shared.up.stored.byte_offset },
			{ shared.gate_output.byte_offset,
			  shared.up_output.byte_offset },
			shared.codes.byte_offset,
			shared.exponents.byte_offset,
			shared.operands.byte_offset,
			{ WIDE_INPUTS, WIDE_INPUTS },
			WIDE_INPUTS,
			rows
		};
		const struct trunk_linear_quantized_arguments down_record = {
			{ shared.down.stored.byte_offset, 0 },
			{ shared.output.byte_offset, 0 },
			shared.hidden_codes.byte_offset,
			shared.hidden_exponents.byte_offset,
			shared.hidden_operands.byte_offset,
			{ WIDE_INPUTS, 0 },
			WIDE_INPUTS,
			rows
		};
		expect(linear_shared_expert_prepare(&shared, ARENA_BYTES,
						    &made_up_kernels,
						    launches) == STATUS_OK,
		       "the shared expert of two forms is accepted");
		expect_form_launch(&launches[LINEAR_SHARED_EXPERT_GATE_UP],
				   interleaved,
				   LINEAR_MERGED_MATRICES * WIDE_INPUTS,
				   one_slice, rows, INTERLEAVED_CHUNKS,
				   &gate_up_record, sizeof(gate_up_record),
				   "the shared expert's gate and up: its form");
		expect_form_launch(&launches[LINEAR_SHARED_EXPERT_DOWN],
				   wide[instance_of(WIDE_DOWN_SLICES)],
				   WIDE_INPUTS, WIDE_DOWN_SLICES, rows, TILES,
				   &down_record, sizeof(down_record),
				   "the shared expert's down: its own form");
		/* refusals, each alone */
		struct linear_trunk_quantized_description other = tiled;
		other.matrices[0].wide_form = INTERLEAVED_CHUNKS;
		trunk_refused(
			&other, RULE_FORM,
			"the interleaved instance of more than one slice");
		other = tiled;
		other.matrices[0].wide_form = LINEAR_WIDE_FORM_COUNT;
		trunk_refused(&other, RULE_FORM, "a wide form not known");
		other = tiled;
		other.matrices[0].reserved_zero = 1u;
		trunk_refused(&other, RULE_RESERVED,
			      "a reserved field of a matrix");
		other = pair;
		other.matrices[1].wide_form = INTERLEAVED_CHUNKS;
		trunk_refused(&other, RULE_PAIR,
			      "two matrices of other wide forms");
		memcpy(handles, &made_up_kernels, sizeof(handles));
		handles[KERNEL_COUNT - 1u] =
			BACKEND_NO_KERNEL; /* the interleaved instance, the last member */
		struct linear_kernels_record without_interleaved;
		memcpy(&without_interleaved, handles,
		       sizeof(without_interleaved));
		struct kernel_launch launch;
		char logged[CAPTURE_CAPACITY];
		context_text = "the interleaved instance not prepared";
		capture_begin();
		const result_status status = linear_trunk_quantized_prepare(
			&chunked, ARENA_BYTES, &without_interleaved, &launch);
		capture_end(logged, sizeof(logged));
		expect(status == STATUS_INVALID_ARGUMENT, "refused");
		expect_one_error_line_with(
			logged, RULE_NOT_PREPARED,
			"the interleaved instance not prepared");
		struct linear_trunk_bfloat16_description interleaved_values =
			values;
		interleaved_values.matrix.slice_count = one_slice;
		interleaved_values.matrix.wide_form = INTERLEAVED_TILES;
		context_text = "bfloat16 rows in the interleaved instance";
		capture_begin();
		expect(linear_trunk_bfloat16_prepare(
			       &interleaved_values, ARENA_BYTES,
			       &made_up_kernels,
			       &launch) == STATUS_INVALID_ARGUMENT,
		       "refused");
		capture_end(logged, sizeof(logged));
		expect_one_error_line_with(
			logged, RULE_FORM,
			"bfloat16 rows in the interleaved instance");
	}
    /* 8 rows: the thin launch whatever the form, the interleaved one of more slices too */
    context_text = "8 rows in a wide form";
    struct linear_trunk_quantized_description thin = trunk_quantized_of(
	    1, WIDE_INPUTS, WIDE_SLICES, LINEAR_MAXIMUM_ROWS);
    thin.matrices[0].wide_form = INTERLEAVED_CHUNKS;
    const struct trunk_linear_quantized_arguments thin_record = {
	    { thin.matrices[0].stored.byte_offset, 0 },
	    { thin.outputs[0].byte_offset, 0 },
	    thin.codes.byte_offset,
	    thin.exponents.byte_offset,
	    thin.operands.byte_offset,
	    { FEW_OUTPUTS, 0 },
	    WIDE_INPUTS,
	    LINEAR_MAXIMUM_ROWS
    };
    expect(linear_trunk_quantized_prepare(&thin, ARENA_BYTES, &made_up_kernels,
					  launches) == STATUS_OK,
	   "8 rows are accepted in any form");
    expect_trunk_launch(
	    &launches[0],
	    made_up_kernels.trunk_linear_quantized[instance_of(WIDE_SLICES)],
	    FEW_OUTPUTS, WIDE_SLICES, LINEAR_MAXIMUM_ROWS, &thin_record,
	    sizeof(thin_record),
	    "8 rows: the thin instance, its grid and record");
}

int main(void)
{
    uint32_t handles[KERNEL_COUNT];
    error_stream_descriptor = dup(STDERR_FILENO);
    capture_descriptor = mkstemp(capture_path);
    if (error_stream_descriptor < 0 || capture_descriptor < 0) {
        (void)printf("FAILED: the temporary file that stands in for the error stream could not be made\n");
        return EXIT_FAILURE;
    }
    for (uint32_t kernel_index = 0; kernel_index < KERNEL_COUNT; kernel_index++) {
        handles[kernel_index] = FIRST_MADE_UP_HANDLE + kernel_index;
    }
    memcpy(&made_up_kernels, handles, sizeof(made_up_kernels));
    test_kernels_prepare();
    test_slice_counts();
    test_merged_matrices();
    test_slice_defaults();
    test_wide_forms_of_matrices();
    test_draft_main_projection();
    test_wide_rows();
    test_wide_forms();
    context_text = "the end of the test";
    expect(close(capture_descriptor) == 0 && unlink(capture_path) == 0 && close(error_stream_descriptor) == 0 &&
               test_host_memory_backend_open_card_count() == 0,
           "the temporary file is removed and no card is left open");
    if (failures != 0) {
        return EXIT_FAILURE;
    }
    (void)printf("linear_launch_test passed\n");
    return EXIT_SUCCESS;
}
