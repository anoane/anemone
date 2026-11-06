/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * dense_linear_test: what source/dense_linear.c does, without a card. A prepare function is arithmetic on a
 * description, the arena is a number, the handles are made up or given by the registry of the host memory backend.
 *
 *   - dense_linear_kernels_prepare: refused for a missing argument; a card without a build of the kernels
 *     gives its status as it is; on a card of each class of test_host_memory_backend every function is asked once,
 *     in the order of the members, by its file, its name and the size of its record, the wide forms left
 *     BACKEND_NO_KERNEL of 0 outputs; a kernel_prepare that fails at any call ends it with that status and one
 *     error line naming the function, the handles untouched. dense_linear_wide_kernels_prepare the same of
 *     each class's two wide forms (by hand: class 80 bfloat16_wide_1 and float32_wide_8, class 120 and 120 native
 *     bfloat16_wide_8 and float32_wide_4), the products' handles kept, their outputs a block set; another class
 *     refused.
 *   - The products of the model (dense_linear.h), each at 1 and at 8 rows, and shapes from 4 to 2^20 outputs by 8 to
 *     2^16 inputs: the function of its form (narrow below DENSE_NARROW_BELOW_BLOCKS blocks of 4 outputs),
 *     weight type and row count, the grid of its outputs, a block of
 *     DENSE_THREADS_VALUE, no shared memory, the record member by member and zero behind it; on a card of each
 *     class of the host memory backend, launch_run takes the launches of the products.
 *   - Wide launches of every product but the head at 9, 64, 65, 100, 1,024 and 4,096 rows, on a card of class 80 and
 *     of class 120 of the host memory backend: the wide form of the class prepared for the weights' type, grid
 *     output_size over its outputs by the groups of 8 rows, the block of the family, the record as the thin
 *     launches'; launch_run takes them.
 *   - Every refusal of dense_linear.h once: STATUS_INVALID_ARGUMENT, one error line with the words of its
 *     rule (the numbers of a description share one), the launch untouched; a wide launch whose wide form is not
 *     prepared or does not divide its outputs, the head above 8 rows, any product above a chunk's rows. Without
 *     DENSE_STEP_ROWS the step area is no range: not checked.
 *   - The host reference (dense_linear_reference.h) on numbers worked by hand: the reciprocal square root, the sums and
 *     magnitudes of two outputs of either weight type, the row's scale, the bound of either output type with a scale;
 *     and one stored weight changed: the output of the weights before fails against the reference after. The index
 *     weights' scale is the reference's, from the model's numbers.
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
#include "anemone/dense_linear.h"
#include "anemone/dense_linear_records.h"
#include "anemone/linear.h"
#include "anemone/model_layout.h"
#include "anemone/status.h"
#include "anemone/step_records.h"
#include "dense_linear_reference.h"
#include "host_memory_backend.h"

#define COUNT_OF(table) (sizeof(table) / sizeof((table)[0]))
#define BFLOAT16 DENSE_BFLOAT16
#define FLOAT32 DENSE_FLOAT32
#define FIRST_RECORDED_HANDLE 40u               /* what the recording table gives the first kernel it is asked */
#define FIRST_MADE_UP_HANDLE 11u
#define CARD_MEMORY_BYTES 4096u                 /* of the stand-in card that the kernels are asked of */
#define UNTOUCHED_BYTE 0xA5
#define CAPTURE_CAPACITY 2048u
#define ARENA_BYTES (1ull << 40u)               /* more than any case lays out */
#define HIDDEN MODEL_HIDDEN
#define ROW_COUNTS 2u                           /* 1, and the most */
#define SMALLEST_OUTPUTS DENSE_OUTPUTS_PER_BLOCK
#define MOST_OUTPUTS (1u << 20u)
#define SMALLEST_INPUTS DENSE_CHUNK_INPUTS
#define MOST_INPUTS (1u << 16u)
#define CHUNK_BYTES 16u                         /* of bfloat16 rows */
#define MISALIGNMENT 8u                         /* bytes: half the alignment of an offset */
#define FLAG_NOT_KNOWN 0x4u
#define KNOWN_OUTPUTS 2u                        /* of the matrix worked by hand */
#define KNOWN_INPUTS 8u
#define ROOT_TOLERANCE 3.552713678800501e-15    /* 2^-48 */
#define CHANGED_INPUT 3u                        /* the weight a stored word change reaches: 1 to 64 */
#define CHANGED_WEIGHT 64.0f
#define BFLOAT16_SHIFT 16u                      /* a bfloat16 is the upper half of a float32 */
#define INDEX_SCALE DENSE_INDEX_WEIGHTS_SCALE
#define UNEVEN_ROWS 100u                        /* twelve groups of 8 and one of 4 */
#define CLASS_CASES 3u                          /* class 80, class 120, class 120 native */
#define CLASS_NONE 90u                          /* a class no card has */
#define NOT_EIGHT_OUTPUTS 12u                   /* outputs a form of 8 a block does not divide */

/* Words of the one error line of each rule. */
#define RULE_MISSING "a missing argument"
#define RULE_NOT_PREPARED "is not prepared"
#define RULE_RESERVED "a reserved field"
#define CONE_ROWS_TESTED 64u /* a wide launch's */
#define CONE_LAYER_TESTED 30u /* a decoder layer */
#define RULE_NUMBERS "known types and flags, outputs in blocks"
#define RULE_FORMAT "its format stores"
#define RULE_TOUCHED "the launches touch"
#define RULE_ALIGNED "an offset is a multiple of"
#define RULE_ARENA "leaves the arena"
#define RULE_SHARED "share a byte"
#define RULE_DIVIDE "outputs a block, which must divide them"
#define RULE_CLASS "classes 80, 120 and 120 native take them"

static int failures = 0;
static const char *context_text = "";
static char capture_path[] = "/tmp/anemone_dense_linear_test_XXXXXX";
static int capture_descriptor = -1;             /* the temporary file that stands in for the error stream */
static int error_stream_descriptor = -1;        /* a duplicate of descriptor 2, kept to put it back */
static struct dense_linear_kernels made_up_kernels;

/* The wide forms each class prepares, by hand, in the order asked (the stamps' order, each form's skinny regimes
 * before its wide one): bfloat16 weights, then float32; their outputs a block. Class 80 also takes float32's skinny
 * forms (dense_linear.c: 4 outputs a block above DENSE_SKINNY_MANY_ROWS rows, 1 to them). */
enum { CLASS_CASE_80, CLASS_CASE_120, CLASS_CASE_120_NATIVE };
#define WIDE_ASKS_MOST 4u
static const uint32_t classes_of_cases[CLASS_CASES] = {
	ARITHMETIC_CLASS_SM_80, ARITHMETIC_CLASS_SM_120,
	ARITHMETIC_CLASS_SM_120_NATIVE
};
static const char *const wide_functions_expected[CLASS_CASES][WIDE_ASKS_MOST] = {
	{ "dense_linear_bfloat16_wide_1", "dense_linear_float32_wide_8",
	  "dense_linear_float32_wide_4", "dense_linear_float32_wide_1" },
	{ "dense_linear_bfloat16_wide_8", "dense_linear_float32_wide_4" },
	{ "dense_linear_bfloat16_wide_8", "dense_linear_float32_wide_4" }
};
static const uint32_t wide_asks_expected[CLASS_CASES] = { WIDE_ASKS_MOST,
							  DENSE_TYPE_COUNT,
							  DENSE_TYPE_COUNT };
static const uint32_t wide_outputs_expected[CLASS_CASES][DENSE_TYPE_COUNT] = {
	{ 1, 8 },
	{ 8, 4 },
	{ 8, 4 }
};
/* class 80's skinny forms of float32 weights (regime 0 to DENSE_SKINNY_MANY_ROWS rows, 1 above): their
 * outputs a block and the handles of their asks (the third and fourth) */
static const uint32_t skinny_outputs_expected[DENSE_SKINNY_REGIMES] = { 1, 4 };
static const uint32_t skinny_handles_expected[DENSE_SKINNY_REGIMES] = {
	FIRST_RECORDED_HANDLE + 3u, FIRST_RECORDED_HANDLE + 2u
};

/* The functions in the order of the members, by hand. */
static const char *const functions_expected[DENSE_INSTANCES] = {
	"dense_linear_bfloat16_1",	  "dense_linear_bfloat16_8",
	"dense_linear_float32_1",	  "dense_linear_float32_8",
	"dense_linear_bfloat16_1_narrow", "dense_linear_bfloat16_8_narrow",
	"dense_linear_float32_1_narrow",  "dense_linear_float32_8_narrow"
};

/* The outputs a block of a launch of output_size owns: 4, or 1 where 4 would leave too few blocks (the narrow form). */
static uint32_t block_outputs_of(uint32_t output_size)
{
	return output_size / DENSE_OUTPUTS_PER_BLOCK <
			       DENSE_NARROW_BELOW_BLOCKS ?
		       1u :
		       DENSE_OUTPUTS_PER_BLOCK;
}

/* A product of the model. */
struct product_case {
    const char *name;
    uint32_t output_size;
    uint32_t input_size;
    uint32_t weight_type;
    uint32_t output_type;
    uint32_t flags;
    float output_scale;
    float epsilon;
};
static const struct product_case products[] = {
	{ "the head", MODEL_VOCABULARY_VALUE, HIDDEN, BFLOAT16, FLOAT32,
	  DENSE_STEP_ROWS, 1.0f, 0.0f },
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
	{ "a hyper mix", MODEL_MIXES, MODEL_STREAMS * HIDDEN, FLOAT32, FLOAT32,
	  DENSE_ROW_SCALE, 1.0f, MODEL_NORM_EPSILON },
};
enum { HEAD_CASE = 0, MIX_CASE = 6 };
enum { DIMENSION_X, DIMENSION_Y, DIMENSION_Z };
static const uint32_t row_counts[ROW_COUNTS] = { 1, LINEAR_MAXIMUM_ROWS };

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

static uint32_t asked_count = 0;
static uint32_t failing_call_number = 0;        /* counted from 1; 0: no call fails */
static bool asked_as_expected = true;
static const char *const *names_expected = functions_expected;  /* the functions to be asked, in order */
static uint32_t names_expected_count = DENSE_INSTANCES;

/* kernel_prepare of a table that checks what it is asked and gives the handles FIRST_RECORDED_HANDLE and on. */
MUST_CHECK static result_status
recording_kernel_prepare(struct device_card *card, const char *kernel_file_name,
			 const char *function_name,
			 uint32_t argument_byte_count, uint32_t *kernel_out)
{
	asked_as_expected =
		asked_as_expected && card != NULL &&
		asked_count < names_expected_count &&
		strcmp(kernel_file_name, KERNEL_FILE_DENSE_LINEAR) == 0 &&
		strcmp(function_name, names_expected[asked_count]) == 0 &&
		argument_byte_count == DENSE_ARGUMENT_BYTES;
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
	struct dense_linear_kernels kernels;
	struct dense_linear_kernels untouched;
	struct device_card *card = NULL;
	char logged[CAPTURE_CAPACITY];
	context_text = "the kernels of a card";
	memset(&kernels, UNTOUCHED_BYTE, sizeof(kernels));
	memset(&untouched, UNTOUCHED_BYTE, sizeof(untouched));
	test_host_memory_backend_reset(1, CARD_MEMORY_BYTES);
	expect(test_host_memory_backend.card_open(0, CARD_MEMORY_BYTES,
						  &card) == STATUS_OK,
	       "a card opens");
	const struct {
		const struct backend_operations *backend;
		struct device_card *card;
		struct dense_linear_kernels *kernels_out;
	} missing[] = { { NULL, card, &kernels },
			{ &backend_without_kernels, card, &kernels },
			{ &recording_backend, NULL, &kernels },
			{ &recording_backend, card, NULL } };
    for (size_t index = 0; index < COUNT_OF(missing); index++) {
        capture_begin();
	const result_status status = dense_linear_kernels_prepare(
		missing[index].backend, missing[index].card,
		missing[index].kernels_out);
	capture_end(logged, sizeof(logged));
	expect(status == STATUS_INVALID_ARGUMENT && asked_count == 0,
	       "a missing argument is refused");
	expect_one_error_line_with(logged, RULE_MISSING, "a missing argument");
    }
    capture_begin();
    expect(dense_linear_kernels_prepare(&test_host_memory_backend, card,
					&kernels) == STATUS_UNSUPPORTED,
	   "a card without a build of the kernels gives its status");
    capture_end(logged, sizeof(logged));
    expect(memcmp(&kernels, &untouched, sizeof(kernels)) == 0, "a failure leaves the handles untouched");
    for (uint32_t call_number = 1; call_number <= DENSE_INSTANCES;
	 call_number++) {
	    failing_call_number = call_number;
	    asked_count = 0;
	    capture_begin();
	    expect(dense_linear_kernels_prepare(&recording_backend, card,
						&kernels) ==
			   STATUS_DEVICE_ERROR,
		   "the status of the kernel_prepare that fails");
	    capture_end(logged, sizeof(logged));
	    expect_one_error_line_with(logged,
				       functions_expected[call_number - 1],
				       "the function that fails is named");
	    expect(asked_count == call_number &&
			   memcmp(&kernels, &untouched, sizeof(kernels)) == 0,
		   "the first failure ends the preparation and leaves the handles untouched");
    }
    failing_call_number = 0;
    asked_count = 0;
    expect(dense_linear_kernels_prepare(&recording_backend, card, &kernels) ==
			   STATUS_OK &&
		   asked_as_expected && asked_count == DENSE_INSTANCES,
	   "each function is asked once, in the order of the members, by its file, its name and its record's size");
    for (uint32_t instance = 0; instance < DENSE_INSTANCES; instance++) {
	    expect(kernels.products[instance] ==
			   FIRST_RECORDED_HANDLE + instance,
		   "a member holds its handle");
    }
    for (uint32_t type = 0; type < DENSE_TYPE_COUNT; type++) {
	    expect(kernels.wide_products[type] == BACKEND_NO_KERNEL &&
			   kernels.wide_block_outputs[type] == 0,
		   "a wide form is left unprepared, of no outputs");
    }
    /* the wide forms of each class, behind the products */
    const struct dense_linear_kernels products_alone = kernels;
    for (size_t index = 0; index < COUNT_OF(missing); index++) {
        capture_begin();
	const result_status status = dense_linear_wide_kernels_prepare(
		missing[index].backend, missing[index].card,
		ARITHMETIC_CLASS_SM_80, missing[index].kernels_out);
	capture_end(logged, sizeof(logged));
	expect(status == STATUS_INVALID_ARGUMENT,
	       "a missing argument is refused");
	expect_one_error_line_with(logged, RULE_MISSING, "a missing argument");
    }
    capture_begin();
    expect(dense_linear_wide_kernels_prepare(&recording_backend, card,
					     CLASS_NONE, &kernels) ==
		   STATUS_INVALID_ARGUMENT,
	   "a class without wide forms is refused");
    capture_end(logged, sizeof(logged));
    expect_one_error_line_with(logged, RULE_CLASS, "a class without wide forms");
    for (uint32_t class_case = 0; class_case < CLASS_CASES; class_case++) {
        names_expected = wide_functions_expected[class_case];
        names_expected_count = wide_asks_expected[class_case];
        for (uint32_t call_number = 1; call_number <= wide_asks_expected[class_case]; call_number++) {
            failing_call_number = call_number;
            asked_count = 0;
            kernels = products_alone;
            capture_begin();
	    expect(dense_linear_wide_kernels_prepare(
			   &recording_backend, card,
			   classes_of_cases[class_case],
			   &kernels) == STATUS_DEVICE_ERROR,
		   "the status of the wide kernel_prepare that fails");
	    capture_end(logged, sizeof(logged));
            expect_one_error_line_with(logged, names_expected[call_number - 1],
                                       "the wide function that fails is named");
            expect(memcmp(&kernels, &products_alone, sizeof(kernels)) == 0, "a failure leaves the handles untouched");
        }
        failing_call_number = 0;
        asked_count = 0;
        asked_as_expected = true;
	expect(dense_linear_wide_kernels_prepare(&recording_backend, card,
						 classes_of_cases[class_case],
						 &kernels) == STATUS_OK &&
		       asked_as_expected &&
		       asked_count == wide_asks_expected[class_case] &&
		       memcmp(kernels.products, products_alone.products,
			      sizeof(kernels.products)) == 0,
	       "the class's wide form of each type is asked once, in order, the products' handles kept");
	for (uint32_t type = 0; type < DENSE_TYPE_COUNT; type++) {
		expect(kernels.wide_products[type] ==
				       FIRST_RECORDED_HANDLE + type &&
			       kernels.wide_block_outputs[type] ==
				       wide_outputs_expected[class_case][type],
		       "a wide form holds its handle and its outputs a block");
	}
	for (uint32_t regime = 0; regime < DENSE_SKINNY_REGIMES; regime++) {
		const bool is_skinny = class_case == CLASS_CASE_80;
		expect(kernels.skinny_products[DENSE_FLOAT32][regime] ==
				       (is_skinny ?
						skinny_handles_expected[regime] :
						BACKEND_NO_KERNEL) &&
			       kernels.skinny_block_outputs[DENSE_FLOAT32]
							   [regime] ==
				       (is_skinny ?
						skinny_outputs_expected[regime] :
						0u) &&
			       kernels.skinny_products[DENSE_BFLOAT16][regime] ==
				       BACKEND_NO_KERNEL &&
			       kernels.skinny_block_outputs[DENSE_BFLOAT16]
							   [regime] == 0,
		       "class 80's skinny forms of float32 weights hold their handles and outputs a block, no other");
	}
    }
    names_expected = functions_expected;
    names_expected_count = DENSE_INSTANCES;
    test_host_memory_backend.card_close(card, 0);
}

/* ---- descriptions laid out one range behind another ---- */

/* The next range of byte_count bytes from *cursor on, on a multiple of the alignment. */
static struct linear_range next_range(uint64_t *cursor, uint64_t byte_count)
{
	const struct linear_range range = { *cursor, byte_count };
	*cursor += (byte_count + LINEAR_OFFSET_ALIGNMENT_BYTES - 1) /
		   LINEAR_OFFSET_ALIGNMENT_BYTES *
		   LINEAR_OFFSET_ALIGNMENT_BYTES;
	return range;
}

static uint64_t type_bytes(uint32_t type)
{
    return type == FLOAT32 ? sizeof(float) : sizeof(uint16_t);
}

/* The description of a product at a row count, its ranges laid out from the first aligned byte on. */
static struct dense_linear_description
description_of(const struct product_case *product, uint32_t row_count)
{
	uint64_t cursor = LINEAR_OFFSET_ALIGNMENT_BYTES;
	struct dense_linear_description description = {
		.output_size = product->output_size,
		.input_size = product->input_size,
		.row_count = row_count,
		.weight_type = product->weight_type,
		.output_type = product->output_type,
		.flags = product->flags,
		.output_scale = product->output_scale,
		.epsilon = product->epsilon
	};
	description.weights = next_range(
		&cursor, (uint64_t)product->output_size * product->input_size *
				 type_bytes(product->weight_type));
	description.rows =
		next_range(&cursor, (uint64_t)row_count * product->input_size *
					    sizeof(uint16_t));
	description.output =
		next_range(&cursor, (uint64_t)row_count * product->output_size *
					    type_bytes(product->output_type));
	description.step = next_range(
		&cursor, STEP_BLOCK_BYTES_VALUE +
				 (uint64_t)row_count * STEP_ROW_BYTES_VALUE);
	return description;
}

/* New block widths retain old handles on partial preparation failure; each selected grid must cover every output.
 * A width that does not divide the matrix leaves it in its existing wide form. Device bytes are the card test's. */
static void test_staged_widths(void)
{
	static const uint32_t widths[] = { 8, 16, 32, 64 };
	static const char *const names[][DENSE_TYPE_COUNT] = {
		{ "dense_linear_bfloat16_staged_8",
		  "dense_linear_float32_staged_8" },
		{ "dense_linear_bfloat16_staged_16",
		  "dense_linear_float32_staged_16" },
		{ "dense_linear_bfloat16_staged_32",
		  "dense_linear_float32_staged_32" },
		{ "dense_linear_bfloat16_staged_64",
		  "dense_linear_float32_staged_64" }
	};
	static const struct backend_operations backend = {
		.kernel_prepare = recording_kernel_prepare
	};
	struct device_card *card = NULL;
	struct dense_linear_kernels kernels = made_up_kernels;
	context_text = "staged block widths";
	test_host_memory_backend_reset(1, CARD_MEMORY_BYTES);
	expect(test_host_memory_backend.card_open(0, CARD_MEMORY_BYTES,
						  &card) == STATUS_OK,
	       "a card opens");
	for (uint32_t index = 0; index < COUNT_OF(widths); index++) {
		names_expected = names[index];
		names_expected_count = DENSE_TYPE_COUNT;
		for (uint32_t fail = 1; fail <= DENSE_TYPE_COUNT; fail++) {
			const struct dense_linear_kernels before = kernels;
			asked_count = 0;
			failing_call_number = fail;
			expect(dense_linear_staged_kernels_prepare(
				       &backend, card, widths[index],
				       &kernels) == STATUS_DEVICE_ERROR &&
				       asked_count == fail &&
				       memcmp(&before, &kernels,
					      sizeof(kernels)) == 0,
			       "partial preparation leaves the previous width and handles intact");
		}
		asked_count = failing_call_number = 0;
		asked_as_expected = true;
		expect(dense_linear_staged_kernels_prepare(
			       &backend, card, widths[index], &kernels) ==
				       STATUS_OK &&
			       asked_as_expected &&
			       asked_count == DENSE_TYPE_COUNT &&
			       kernels.staged_block_outputs == widths[index],
		       "the named width prepares atomically");
		static const uint32_t outputs[] = { 32, 64 };
		for (uint32_t output = 0; output < COUNT_OF(outputs);
		     output++) {
			const struct product_case product = {
				"staged shape", outputs[output],
				HIDDEN,		BFLOAT16,
				FLOAT32,	0,
				1.0f,		0.0f
			};
			const struct dense_linear_description description =
				description_of(&product,
					       DENSE_STAGED_LEAST_ROWS);
			const bool staged = outputs[output] % widths[index] ==
					    0;
			const uint32_t block =
				staged ? widths[index] :
					 kernels.wide_block_outputs[BFLOAT16];
			struct kernel_launch launch;
			expect(dense_linear_prepare(&description, ARENA_BYTES,
						    &kernels,
						    &launch) == STATUS_OK &&
				       launch.grid[DIMENSION_X] ==
					       outputs[output] / block &&
				       launch.grid[DIMENSION_Y] ==
					       DENSE_STAGED_LEAST_ROWS /
						       LINEAR_MAXIMUM_ROWS &&
				       launch.kernel ==
					       (staged ? kernels.staged_products
								 [BFLOAT16] :
							 kernels.wide_products
								 [BFLOAT16]) &&
				       launch.dynamic_shared_byte_count ==
					       (staged ?
							HIDDEN *
								LINEAR_MAXIMUM_ROWS *
								sizeof(uint16_t) :
							0u),
			       "whole output coverage and shared rows match the selected width or wide fallback");
		}
	}
	test_host_memory_backend.card_close(card, 1);
}

/* The bytes of the record a launch of the description must carry, by hand, and zero behind them. */
static void record_bytes_of(const struct dense_linear_description *description,
			    unsigned char bytes_out[LAUNCH_ARGUMENT_BYTES])
{
	const struct dense_linear_arguments record = {
		description->weights.byte_offset,
		description->rows.byte_offset,
		description->output.byte_offset,
		(description->flags & DENSE_STEP_ROWS) != 0 ?
			description->step.byte_offset :
			0,
		description->output_size,
		description->input_size,
		description->row_count,
		description->output_type,
		description->flags,
		description->output_scale,
		description->epsilon,
		0
	};
	memset(bytes_out, 0, LAUNCH_ARGUMENT_BYTES);
	memcpy(bytes_out, &record, sizeof(record));
}

static void test_products(void)
{
	static const uint32_t classes[] = { ARITHMETIC_CLASS_SM_80,
					    ARITHMETIC_CLASS_SM_120 };
	for (size_t case_index = 0;
	     case_index < COUNT_OF(products) * ROW_COUNTS; case_index++) {
		const struct product_case *const product =
			&products[case_index / ROW_COUNTS];
		const uint32_t row_count = row_counts[case_index % ROW_COUNTS];
		const struct dense_linear_description description =
			description_of(product, row_count);
		const uint32_t block_outputs =
			block_outputs_of(product->output_size);
		const uint32_t instance =
			(block_outputs == 1u ? DENSE_FORM_INSTANCES : 0u) +
			product->weight_type * DENSE_ROW_INSTANCES +
			(row_count == 1 ? 0u : 1u);
		unsigned char record_bytes[LAUNCH_ARGUMENT_BYTES];
		struct kernel_launch launch;
		context_text = product->name;
		record_bytes_of(&description, record_bytes);
		memset(&launch, UNTOUCHED_BYTE, sizeof(launch));
		expect(dense_linear_prepare(&description, ARENA_BYTES,
					    &made_up_kernels,
					    &launch) == STATUS_OK &&
			       launch.kernel ==
				       made_up_kernels.products[instance] &&
			       launch.grid[DIMENSION_X] ==
				       product->output_size / block_outputs &&
			       launch.grid[DIMENSION_Y] == 1 &&
			       launch.grid[DIMENSION_Z] == 1 &&
			       launch.block[DIMENSION_X] ==
				       DENSE_THREADS_VALUE &&
			       launch.block[DIMENSION_Y] == 1 &&
			       launch.block[DIMENSION_Z] == 1 &&
			       launch.dynamic_shared_byte_count == 0 &&
			       launch.argument_byte_count ==
				       DENSE_ARGUMENT_BYTES &&
			       launch.branch == LAUNCH_CHAIN,
		       "the function of its form, type and row count, the grid of its outputs, the block of the family");
		expect(memcmp(launch.argument, record_bytes,
			      sizeof(record_bytes)) == 0,
		       "the record member by member, zero behind it");
		/* on a card of each class of the host memory backend, whose registry gives the handles */
		for (size_t class_index = 0; class_index < COUNT_OF(classes);
		     class_index++) {
			struct dense_linear_kernels kernels;
			struct device_card *card = NULL;
			test_host_memory_backend_reset(1, CARD_MEMORY_BYTES);
			test_host_memory_backend_set_arithmetic_class(
				0, classes[class_index]);
			expect(test_host_memory_backend.card_open(
				       0, CARD_MEMORY_BYTES, &card) ==
					       STATUS_OK &&
				       dense_linear_kernels_prepare(
					       &test_host_memory_backend, card,
					       &kernels) == STATUS_OK &&
				       dense_linear_prepare(
					       &description, ARENA_BYTES,
					       &kernels,
					       &launch) == STATUS_OK &&
				       test_host_memory_backend.launch_run(
					       card, &launch, 1) == STATUS_OK,
			       "the launch is one a card of either class takes");
			test_host_memory_backend.card_close(card, 0);
		}
    }
}

static void test_wide_products(void)
{
	static const uint32_t wide_row_counts[] = {
		LINEAR_MAXIMUM_ROWS + 1u,   MODEL_WIDE_LEAST_ROWS,
		MODEL_WIDE_LEAST_ROWS + 1u, UNEVEN_ROWS,
		LINEAR_WIDE_MOST_ROWS,	    MODEL_WIDE_MOST_ROWS
	};
	for (uint32_t class_case = 0; class_case < CLASS_CASES; class_case++) {
		struct dense_linear_kernels kernels;
		struct device_card *card = NULL;
		memset(&kernels, 0, sizeof(kernels));
		context_text = "the wide launches of a class";
		test_host_memory_backend_reset(1, CARD_MEMORY_BYTES);
		test_host_memory_backend_set_arithmetic_class(
			0,
			classes_of_cases[class_case] == ARITHMETIC_CLASS_SM_80 ?
				ARITHMETIC_CLASS_SM_80 :
				ARITHMETIC_CLASS_SM_120);
		expect(test_host_memory_backend.card_open(0, CARD_MEMORY_BYTES,
							  &card) == STATUS_OK &&
			       dense_linear_kernels_prepare(
				       &test_host_memory_backend, card,
				       &kernels) == STATUS_OK &&
			       dense_linear_wide_kernels_prepare(
				       &test_host_memory_backend, card,
				       classes_of_cases[class_case],
				       &kernels) == STATUS_OK,
		       "a card of the class prepares its wide forms");
		for (size_t case_index = 0;
		     case_index <
		     COUNT_OF(products) * COUNT_OF(wide_row_counts);
		     case_index++) {
			const struct product_case *const product =
				&products[case_index /
					  COUNT_OF(wide_row_counts)];
			const uint32_t row_count =
				wide_row_counts[case_index %
						COUNT_OF(wide_row_counts)];
			const uint32_t groups =
				(row_count + LINEAR_MAXIMUM_ROWS - 1u) /
				LINEAR_MAXIMUM_ROWS;
			/* class 80's skinny form for a matrix of at most DENSE_SKINNY_MOST_OUTPUTS_VALUE float32 outputs */
			const uint32_t regime =
				row_count > DENSE_SKINNY_MANY_ROWS ? 1u : 0u;
			const bool is_skinny =
				class_case == CLASS_CASE_80 &&
				product->weight_type == DENSE_FLOAT32 &&
				product->output_size <=
					DENSE_SKINNY_MOST_OUTPUTS_VALUE &&
				product->output_size %
						skinny_outputs_expected[regime] ==
					0;
			const uint32_t block_outputs =
				is_skinny ? skinny_outputs_expected[regime] :
					    wide_outputs_expected
						    [class_case]
						    [product->weight_type];
			const uint32_t kernel =
				is_skinny ?
					kernels.skinny_products
						[product->weight_type][regime] :
					kernels.wide_products
						[product->weight_type];
			if ((product->flags & DENSE_STEP_ROWS) != 0) {
				continue; /* the head is never wide */
			}
			const struct dense_linear_description description =
				description_of(product, row_count);
			unsigned char record_bytes[LAUNCH_ARGUMENT_BYTES];
			struct kernel_launch launch;
			context_text = product->name;
			record_bytes_of(&description, record_bytes);
			memset(&launch, UNTOUCHED_BYTE, sizeof(launch));
			expect(dense_linear_prepare(&description, ARENA_BYTES,
						    &kernels,
						    &launch) == STATUS_OK &&
				       launch.kernel == kernel &&
				       launch.kernel != BACKEND_NO_KERNEL &&
				       launch.grid[DIMENSION_X] ==
					       product->output_size /
						       block_outputs &&
				       launch.grid[DIMENSION_Y] == groups &&
				       launch.grid[DIMENSION_Z] == 1 &&
				       launch.block[DIMENSION_X] ==
					       DENSE_THREADS_VALUE &&
				       launch.block[DIMENSION_Y] == 1 &&
				       launch.block[DIMENSION_Z] == 1 &&
				       launch.dynamic_shared_byte_count == 0 &&
				       launch.argument_byte_count ==
					       DENSE_ARGUMENT_BYTES &&
				       launch.branch == LAUNCH_CHAIN &&
				       memcmp(launch.argument, record_bytes,
					      sizeof(record_bytes)) == 0 &&
				       test_host_memory_backend.launch_run(
					       card, &launch, 1) == STATUS_OK,
			       "a wide launch: the class's form of the weights' type (class 80's skinny one for a skinny float32 "
			       "matrix), its grid over the groups, the thin record, taken by the card");
		}
		test_host_memory_backend.card_close(card, 0);
	}
}

/* DENSE_CONE_ROWS: a wide launch of a decoder layer takes the step rows and the layer; a launch of 8 rows or
 * fewer, a layer of the front or the head's step rows with it are refused. */
static void test_cone(void)
{
	struct dense_linear_description description =
		description_of(&products[MIX_CASE], CONE_ROWS_TESTED);
	struct kernel_launch launch;
	struct dense_linear_arguments record;
	char logged[CAPTURE_CAPACITY];
	context_text = "the prefill cone's rows";
	description.flags |= DENSE_CONE_ROWS;
	description.cone_layer = CONE_LAYER_TESTED;
	expect(dense_linear_prepare(&description, ARENA_BYTES, &made_up_kernels,
				    &launch) == STATUS_OK,
	       "a wide launch takes the cone");
	memcpy(&record, launch.argument, sizeof(record));
	expect(record.step_offset == description.step.byte_offset &&
		       record.cone_layer == CONE_LAYER_TESTED &&
		       (record.flags & DENSE_CONE_ROWS) != 0,
	       "its record holds the step rows and the layer");
	const uint32_t refused_layers[] = { MODEL_DECODER_FIRST_LAYER - 1u,
					    MODEL_LAYERS };
	for (uint32_t index = 0; index < COUNT_OF(refused_layers); index++) {
		struct dense_linear_description changed = description;
		changed.cone_layer = refused_layers[index];
		capture_begin();
		const result_status status = dense_linear_prepare(
			&changed, ARENA_BYTES, &made_up_kernels, &launch);
		capture_end(logged, sizeof(logged));
		expect(status == STATUS_INVALID_ARGUMENT,
		       "a layer outside the decoder's is refused");
	}
	struct dense_linear_description thin =
		description_of(&products[MIX_CASE], LINEAR_MAXIMUM_ROWS);
	thin.flags |= DENSE_CONE_ROWS;
	thin.cone_layer = CONE_LAYER_TESTED;
	capture_begin();
	const result_status thin_status = dense_linear_prepare(
		&thin, ARENA_BYTES, &made_up_kernels, &launch);
	capture_end(logged, sizeof(logged));
	expect(thin_status == STATUS_INVALID_ARGUMENT,
	       "a launch of 8 rows takes no cone");
}

static void test_shapes(void)
{
	struct dense_linear_description description =
		description_of(&products[HEAD_CASE], 1);
	description.flags = 0;
	context_text = "shapes of whole blocks and chunks";
	for (uint32_t output_size = SMALLEST_OUTPUTS;
	     output_size <= MOST_OUTPUTS; output_size += output_size) {
		for (uint32_t input_size = SMALLEST_INPUTS;
		     input_size <= MOST_INPUTS; input_size += input_size) {
			struct kernel_launch launch;
			uint64_t cursor = LINEAR_OFFSET_ALIGNMENT_BYTES;
			description.output_size = output_size;
			description.input_size = input_size;
			description.weights = next_range(
				&cursor, (uint64_t)output_size * input_size *
						 sizeof(uint16_t));
			description.rows =
				next_range(&cursor, (uint64_t)input_size *
							    sizeof(uint16_t));
			description.output = next_range(
				&cursor, (uint64_t)output_size * sizeof(float));
			expect(dense_linear_prepare(&description, ARENA_BYTES,
						    &made_up_kernels,
						    &launch) == STATUS_OK &&
				       launch.grid[DIMENSION_X] ==
					       output_size /
						       block_outputs_of(
							       output_size) &&
				       launch.block[DIMENSION_X] ==
					       DENSE_THREADS_VALUE,
			       "every shape of whole blocks and chunks is taken: its grid, the family's block");
		}
    }
}

/* A description that must be refused: the status, one error line with the words, the launch untouched. */
static void refused(const struct dense_linear_description *description,
		    const struct dense_linear_kernels *kernels,
		    bool gives_launch, const char *rule, const char *what)
{
	struct kernel_launch launch;
	unsigned char untouched[sizeof(launch)];
	char logged[CAPTURE_CAPACITY];
	memset(&launch, UNTOUCHED_BYTE, sizeof(launch));
	memset(untouched, UNTOUCHED_BYTE, sizeof(untouched));
	context_text = what;
	capture_begin();
	const result_status status =
		dense_linear_prepare(description, ARENA_BYTES, kernels,
				     gives_launch ? &launch : NULL);
	capture_end(logged, sizeof(logged));
	expect(status == STATUS_INVALID_ARGUMENT, "refused");
	expect_one_error_line_with(logged, rule, what);
	expect(memcmp(&launch, untouched, sizeof(untouched)) == 0,
	       "a refusal leaves the launch untouched");
}

/* A description changed by one member: refused for the rule. */
#define REFUSED_WITH(member, value, rule, what)                            \
	do {                                                               \
		struct dense_linear_description changed = base;            \
		changed.member = (value);                                  \
		refused(&changed, &made_up_kernels, true, (rule), (what)); \
	} while (false)

static void test_refusals(void)
{
    const float not_a_number = NAN;
    const float infinity = HUGE_VALF;
    const struct dense_linear_description base =
	    description_of(&products[HEAD_CASE], 1);
    struct dense_linear_kernels unprepared = made_up_kernels;
    struct dense_linear_description without_step =
	    description_of(&products[MIX_CASE], 1);
    struct kernel_launch launch;
    unsigned char record_bytes[LAUNCH_ARGUMENT_BYTES];
    refused(NULL, &made_up_kernels, true, RULE_MISSING, "no description");
    refused(&base, NULL, true, RULE_MISSING, "no kernels");
    refused(&base, &made_up_kernels, false, RULE_MISSING, "no launch");
    unprepared.products[0] = BACKEND_NO_KERNEL;
    refused(&base, &unprepared, true, RULE_NOT_PREPARED, "a function not prepared");
    REFUSED_WITH(cone_layer, 1, RULE_NUMBERS, "a cone layer without its flag");
    REFUSED_WITH(weight_type, DENSE_TYPE_COUNT, RULE_NUMBERS,
		 "a weight type not known");
    REFUSED_WITH(output_type, DENSE_TYPE_COUNT, RULE_NUMBERS,
		 "an output type not known");
    REFUSED_WITH(flags, FLAG_NOT_KNOWN, RULE_NUMBERS, "a flag not known");
    REFUSED_WITH(output_size, 0, RULE_NUMBERS, "no outputs");
    REFUSED_WITH(output_size, base.output_size + 1, RULE_NUMBERS, "outputs that are no whole blocks");
    REFUSED_WITH(input_size, 0, RULE_NUMBERS, "no inputs");
    REFUSED_WITH(input_size, base.input_size + 1, RULE_NUMBERS, "no whole chunks");
    REFUSED_WITH(row_count, 0, RULE_NUMBERS, "no rows");
    REFUSED_WITH(row_count, LINEAR_MAXIMUM_ROWS + 1, RULE_NUMBERS,
		 "the head of a row more than a slab");
    REFUSED_WITH(output_scale, 0.0f, RULE_NUMBERS, "a scale of zero");
    REFUSED_WITH(output_scale, -1.0f, RULE_NUMBERS, "a negative scale");
    REFUSED_WITH(output_scale, infinity, RULE_NUMBERS, "an infinite scale");
    REFUSED_WITH(output_scale, not_a_number, RULE_NUMBERS, "a scale that is no number");
    REFUSED_WITH(epsilon, -1.0f, RULE_NUMBERS, "a negative epsilon");
    REFUSED_WITH(epsilon, infinity, RULE_NUMBERS, "an infinite epsilon");
    REFUSED_WITH(epsilon, not_a_number, RULE_NUMBERS, "an epsilon that is no number");
    REFUSED_WITH(weights.byte_count, base.weights.byte_count + 1, RULE_FORMAT, "weights of a byte more");
    REFUSED_WITH(weights.byte_count, base.weights.byte_count - 1, RULE_FORMAT, "weights of a byte less");
    REFUSED_WITH(rows.byte_count, base.rows.byte_count - 1, RULE_TOUCHED, "rows of a byte less");
    REFUSED_WITH(output.byte_count, base.output.byte_count - 1, RULE_TOUCHED, "an output of a byte less");
    REFUSED_WITH(step.byte_count, base.step.byte_count - 1, RULE_TOUCHED, "a step area of a byte less");
    REFUSED_WITH(rows.byte_offset, base.rows.byte_offset + MISALIGNMENT, RULE_ALIGNED, "rows off the alignment");
    REFUSED_WITH(step.byte_offset, ARENA_BYTES, RULE_ARENA, "a step area beyond the arena");
    REFUSED_WITH(output.byte_offset, base.rows.byte_offset, RULE_SHARED, "an output on the rows");
    REFUSED_WITH(step.byte_offset, base.output.byte_offset, RULE_SHARED, "a step area on the output");
    struct dense_linear_description wide =
	    description_of(&products[MIX_CASE], MODEL_WIDE_MOST_ROWS);
    struct dense_linear_kernels no_wide = made_up_kernels;
    for (uint32_t type = 0; type < DENSE_TYPE_COUNT; type++) {
	    no_wide.wide_products[type] = BACKEND_NO_KERNEL;
	    no_wide.wide_block_outputs[type] = 0;
    }
    refused(&wide, &no_wide, true, RULE_NOT_PREPARED, "a wide launch whose wide form is not prepared");
    wide.row_count = MODEL_WIDE_MOST_ROWS + 1u;
    refused(&wide, &made_up_kernels, true, RULE_NUMBERS, "a row more than a chunk");
    struct product_case narrow = products[HEAD_CASE];   /* bfloat16 weights, of 8 outputs a block in made_up_kernels */
    narrow.flags = 0;
    narrow.output_size = NOT_EIGHT_OUTPUTS;
    wide = description_of(&narrow, MODEL_WIDE_LEAST_ROWS);
    refused(&wide, &made_up_kernels, true, RULE_DIVIDE, "outputs the wide form's outputs a block do not divide");
    /* the step area is no range of a product that reads none */
    context_text = "a product without the step rows";
    without_step.step = without_step.rows;
    record_bytes_of(&without_step, record_bytes);
    expect(dense_linear_prepare(&without_step, ARENA_BYTES, &made_up_kernels,
				&launch) == STATUS_OK &&
		   memcmp(launch.argument, record_bytes,
			  sizeof(record_bytes)) == 0 &&
		   without_step.step.byte_offset != 0,
	   "its step area is not checked and the record names none (its offset 0)");
}

/* ---- the host reference ---- */

/* The matrix worked by hand: output 0 all 1, output 1 alternating 2 and -1; the row 1 to 8; the row of 2s. */
static const float known_weights[KNOWN_OUTPUTS][KNOWN_INPUTS] = {{1, 1, 1, 1, 1, 1, 1, 1},
                                                                  {2, -1, 2, -1, 2, -1, 2, -1}};
static const uint16_t known_row[KNOWN_INPUTS] = {0x3F80, 0x4000, 0x4040, 0x4080, 0x40A0, 0x40C0, 0x40E0, 0x4100};
static const uint16_t twos_row[KNOWN_INPUTS] = {0x4000, 0x4000, 0x4000, 0x4000, 0x4000, 0x4000, 0x4000, 0x4000};
static const double known_sums[KNOWN_OUTPUTS] = {36.0, 12.0};
static const double known_magnitudes[KNOWN_OUTPUTS] = {36.0, 52.0};
static const double twos_scaled_sums[KNOWN_OUTPUTS] = {8.0, 4.0};     /* 16 and 8, times (32 / 8)^-1/2 */
static const double roots[] = {1.0, 4.0, 0.25, 2.0, 3.0, 0.75, 1e-20, 3e10, 7.5e-300};
/* 36 in bfloat16, 36 times 2^-6, and the neighbour of 36 */
enum { EXACT_BITS, SCALED_BITS, NEIGHBOUR_BITS, OUTPUT_BITS_COUNT };
static const uint16_t output_bits[OUTPUT_BITS_COUNT] = {0x4210, 0x3F10, 0x4211};

static void test_reference(void)
{
	unsigned char weights[DENSE_TYPE_COUNT]
			     [sizeof(float) * KNOWN_OUTPUTS * KNOWN_INPUTS];
	double sums[KNOWN_OUTPUTS];
	double magnitudes[KNOWN_OUTPUTS];
	context_text = "the host reference";
	for (size_t index = 0; index < COUNT_OF(roots); index++) {
		const double root =
			test_dense_reciprocal_square_root(roots[index]);
		expect(test_dense_magnitude(root * root * roots[index] - 1.0) <=
			       ROOT_TOLERANCE,
		       "1 / sqrt within 2^-48");
	}
    for (uint32_t index = 0; index < KNOWN_OUTPUTS * KNOWN_INPUTS; index++) {
        const float value = known_weights[index / KNOWN_INPUTS][index % KNOWN_INPUTS];
        uint32_t bits = 0;
        memcpy(&bits, &value, sizeof(bits));
        const uint16_t high_half = (uint16_t)(bits >> BFLOAT16_SHIFT);
        memcpy(weights[FLOAT32] + index * sizeof(float), &value, sizeof(value));
        memcpy(weights[BFLOAT16] + index * sizeof(uint16_t), &high_half, sizeof(high_half));
    }
    for (uint32_t type = 0; type < DENSE_TYPE_COUNT; type++) {
	    test_dense_reference(type, weights[type], KNOWN_INPUTS, 0, 0.0f,
				 known_row, 1, 0, KNOWN_OUTPUTS, sums,
				 magnitudes);
	    expect(sums[0] == known_sums[0] && sums[1] == known_sums[1] &&
			   magnitudes[0] == known_magnitudes[0] &&
			   magnitudes[1] == known_magnitudes[1],
		   "the sums and magnitudes of either weight type, worked by hand");
	    test_dense_reference(type, weights[type], KNOWN_INPUTS,
				 DENSE_ROW_SCALE, 0.0f, twos_row, 1, 0,
				 KNOWN_OUTPUTS, sums, magnitudes);
	    expect(test_dense_magnitude(sums[0] - twos_scaled_sums[0]) <=
				   ROOT_TOLERANCE &&
			   test_dense_magnitude(sums[1] -
						twos_scaled_sums[1]) <=
				   ROOT_TOLERANCE,
		   "the row's scale: the mean of its squares to the power -1/2");
    }
    const float exact = (float)known_sums[0];
    const float beyond = (float)(known_sums[0] + known_magnitudes[0]);
    const unsigned char *const bits = (const unsigned char *)output_bits;
    const double sum = known_sums[0];
    const double magnitude_sum = known_magnitudes[0];
    /* REF:515, 555: index_head_dim^-0.5 times index_n_heads^-0.5 in double, a float32 where it scales bfloat16 */
    expect((float)(test_dense_reciprocal_square_root(
			   (double)MODEL_INDEX_HEAD_DIMS) *
		   test_dense_reciprocal_square_root(
			   (double)MODEL_INDEX_HEADS)) == INDEX_SCALE,
	   "the index weights' scale is the reference's");
    expect(test_dense_output_holds(FLOAT32, 1.0f, (const unsigned char *)&exact, 0, sum, magnitude_sum) &&
               !test_dense_output_holds(FLOAT32, 1.0f, (const unsigned char *)&beyond, 0, sum, magnitude_sum) &&
               test_dense_output_holds(BFLOAT16, 1.0f, bits, EXACT_BITS, sum, magnitude_sum) &&
               !test_dense_output_holds(BFLOAT16, 1.0f, bits, NEIGHBOUR_BITS, sum, magnitude_sum) &&
               test_dense_output_holds(BFLOAT16, INDEX_SCALE, bits, SCALED_BITS, sum, magnitude_sum) &&
               !test_dense_output_holds(BFLOAT16, INDEX_SCALE, bits, EXACT_BITS, sum, magnitude_sum),
           "the bound: the exact output holds, one beyond the bound or one place off does not, the scale applies");
    /* one stored word changed: what the weights before give fails against the reference after */
    memcpy(weights[FLOAT32] + CHANGED_INPUT * sizeof(float), &(const float){CHANGED_WEIGHT}, sizeof(float));
    test_dense_reference(FLOAT32, weights[FLOAT32], KNOWN_INPUTS, 0, 0.0f, known_row, 1, 0, 1, sums, magnitudes);
    expect(!test_dense_output_holds(FLOAT32, 1.0f, (const unsigned char *)&exact, 0, sums[0], magnitudes[0]),
           "an output of the weights before a stored word changed fails against the reference after");
}

int main(void)
{
    error_stream_descriptor = dup(STDERR_FILENO);
    capture_descriptor = mkstemp(capture_path);
    if (error_stream_descriptor < 0 || capture_descriptor < 0) {
        (void)printf("FAILED: the temporary file that stands in for the error stream could not be made\n");
        return EXIT_FAILURE;
    }
    for (uint32_t instance = 0; instance < DENSE_INSTANCES; instance++) {
	    made_up_kernels.products[instance] =
		    FIRST_MADE_UP_HANDLE + instance;
    }
    for (uint32_t type = 0; type < DENSE_TYPE_COUNT;
	 type++) { /* class 120's forms */
	    made_up_kernels.wide_products[type] =
		    FIRST_MADE_UP_HANDLE + DENSE_INSTANCES + type;
	    made_up_kernels.wide_block_outputs[type] =
		    wide_outputs_expected[CLASS_CASE_120][type];
    }
    test_kernels_prepare();
    test_products();
    test_wide_products();
    test_staged_widths();
    test_shapes();
    test_cone();
    test_refusals();
    test_reference();
    context_text = "the end of the test";
    expect(close(capture_descriptor) == 0 && unlink(capture_path) == 0 && close(error_stream_descriptor) == 0 &&
               test_host_memory_backend_open_card_count() == 0,
           "the temporary file is removed and no card is left open");
    if (failures != 0) {
        return EXIT_FAILURE;
    }
    (void)printf("dense_linear_test passed\n");
    return EXIT_SUCCESS;
}
