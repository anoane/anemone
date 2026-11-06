/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * weight_format_test: source/weight_format.c against weight_format_seam.h, and tests/weight_format_reference.c
 * against weight_format_reference.h.
 *
 * The engine's part is small: which formats this build carries, and the bytes a matrix of a shape occupies. Every
 * refusal of stored_byte_count is shown for each format, and every matrix shape of the real checkpoint is held
 * against a byte count computed by hand. The reference counts the bytes a second time, lays matrices out and reads
 * them back. A matrix filled from a hash of its positions must come back weight for weight; one stored bit flipped
 * must be seen in exactly the weights that own it; and for a tiny matrix of each format the places of a few
 * weights, worked out by hand from the formulas of the seam, are held against the lay-out and against the read,
 * each without the other.
 *
 * Not run here: a matrix of the real expert shape, 11.8 million weights, which this test would read back several
 * times in every build. The largest shapes of the round trip have its structure (many tiles by several loads, one
 * wide and one tall) at about a sixteenth of each size: the inputs of a 4-bit matrix are whole loads.
 */
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "anemone/pack_digest.h"
#include "anemone/status.h"
#include "anemone/weight_format_seam.h"
#include "weight_format_reference.h"

#define COUNT_OF(table) (sizeof(table) / sizeof((table)[0]))
#define E2M1 \
	WEIGHT_FORMAT_E2M1_SCALED /* short names for the tables of this test */
#define E4M3 WEIGHT_FORMAT_E4M3_SCALED
#define NO_ERROR STATUS_OK
#define INVALID STATUS_INVALID_ARGUMENT
#define UNTOUCHED_COUNT 0x5A5A5A5A5A5A5A5Aull   /* what an output holds before a call that must leave it alone */
#define BITS_PER_BYTE 8u
#define BYTE_MASK 0xFFu
#define HIGH_BYTE 0xFF                          /* what a buffer holds before a lay-out that must write all of it */
#define E2M1_CODE_MASK 15u
#define E2M1_BITS_PER_CODE 4u
#define INPUTS_PER_SCALE 32u                    /* consecutive inputs that share a scale, in both formats */
#define E4M3_OUTPUTS_PER_SCALE 32u              /* the outputs of a block that share its scales */
#define MOST_TEST_BYTES 1048576u                /* far above any matrix this test holds in memory */
#define MOST_INDEX_DIGITS 8u                    /* an index below MOST_TEST_BYTES has 5 digits of 4 bits */
#define FILL_SEED 0x57454947485453ull
#define FLIP_SEED 0x464C495042495453ull

/* The values of the codes a second time, for the comparison with the reference. */
#define E4M3_SIGN_BIT 0x80u
#define E4M3_EXPONENT_SHIFT 3u
#define E4M3_EXPONENT_MASK 15u
#define E4M3_MANTISSA_MASK 7u
#define E4M3_HIDDEN_ONE 8u                      /* 1.0 in eighths */
#define E4M3_WHOLE_NUMBER_DIVISOR 1024.0        /* 2^(bias + mantissa bits) */
#define E4M3_LARGEST_CODE 0x7Eu                 /* 448: the last byte before the one that is not-a-number */
#define E8M0_ONE 127u                           /* the E8M0 byte of 2^0 */
#define E8M0_LARGEST 254u                       /* the last E8M0 byte that is a number: 2^127 */
#define DOUBLING 2.0

/* The matrices of the real checkpoint and the bytes each occupies, computed by hand from the formulas of the seam.
 * 4 bits per weight, outputs * inputs / 2 + outputs * inputs / 32: both expert matrices hold 2304 * 5120 =
 * 11,796,480 weights, which is 5,898,240 + 368,640 bytes. */
#define BYTES_OF_AN_EXPERT_MATRIX 6266880ull
/* 8 bits per weight, outputs * inputs + ceil(outputs / 32) * (inputs / 32). */
#define BYTES_OF_512_BY_5120 2624000ull         /* 2,621,440 + 16 * 160 */
#define BYTES_OF_1280_BY_5120 6560000ull        /* 6,553,600 + 40 * 160 */
#define BYTES_OF_32768_BY_1280 41984000ull      /* 41,943,040 + 1024 * 40 */
#define BYTES_OF_8192_BY_4096 33587200ull       /* 33,554,432 + 256 * 128 */
#define BYTES_OF_5120_BY_8192 41984000ull       /* 41,943,040 + 160 * 256 */
#define BYTES_OF_4096_BY_1280 5248000ull        /* 5,242,880 + 128 * 40 */
#define BYTES_OF_25600_BY_6144 157440000ull     /* 157,286,400 + 800 * 192 */
#define BYTES_OF_2304_BY_5120 11808000ull       /* 11,796,480 + 72 * 160 */
#define BYTES_OF_5120_BY_2304 11808000ull       /* 11,796,480 + 160 * 72 */
#define BYTES_OF_5120_BY_15360 78720000ull      /* 78,643,200 + 160 * 480 */
/* The smallest matrices, and one of a block and a half. */
#define BYTES_OF_E2M1_16_BY_64 544ull           /* 512 + 32 */
#define BYTES_OF_E4M3_16_BY_32 513ull           /* 512 + 1: half a block has a scale of its own */
#define BYTES_OF_E4M3_32_BY_32 1025ull          /* 1,024 + 1 */
#define BYTES_OF_E4M3_48_BY_64 3076ull          /* 3,072 + 2 * 2 */
/* The sizes at the end of 32 bits. A 4-bit matrix of the largest sizes holds 17 / 32 of (2^32 - 16) * (2^32 - 64)
 * bytes, which is 17 * (2^59 - 80 * 2^27 + 32): no shape of that format passes 64 bits. */
#define LARGEST_OUTPUT_SIZE 4294967280u         /* 2^32 - 16 */
#define LARGEST_E2M1_INPUT_SIZE 4294967232u     /* 2^32 - 64 */
#define LARGEST_E4M3_INPUT_SIZE 4294967264u     /* 2^32 - 32 */
#define BYTES_OF_THE_LARGEST_E2M1 9799832606622089760ull
/* 2^32 - 16 outputs are 2^27 blocks, so with I inputs an 8-bit matrix holds I * (2^32 - 16) + (I / 32) * 2^27 =
 * I * 4,299,161,584 bytes. 4,290,777,088 inputs hold 18,446,744,022,236,987,392 bytes, 51,472,564,224 below 2^64;
 * the 32 inputs of one more group hold 137,573,170,688 more, which is beyond it. */
#define MOST_E4M3_INPUTS_OF_THE_LARGEST_OUTPUT 4290777088u
#define MOST_E4M3_BYTES_OF_THE_LARGEST_OUTPUT 18446744022236987392ull
#define E4M3_INPUTS_BEYOND_64_BITS 4290777120u

_Static_assert(
	sizeof(enum weight_format_code) == sizeof(uint32_t) &&
		sizeof(double) == sizeof(uint64_t),
	"the test builds a format from the 32 bits a pack holds, and compares doubles by their 64 bits");

static int failures = 0;
static uint64_t weights_read = 0;               /* printed at the end */

static void expect(bool condition, const char *what)
{
    if (!condition) {
        (void)fprintf(stderr, "FAILED: %s\n", what);
        failures++;
    }
}

static void expect_status(result_status status, result_status expected_status,
			  const char *what)
{
    if (status != expected_status) {
	    (void)fprintf(stderr, "FAILED: %s: status \"%s\", not \"%s\"\n",
			  what, status_text(status),
			  status_text(expected_status));
	    failures++;
    }
}

static void expect_count(uint64_t count, uint64_t expected_count, const char *what)
{
    if (count != expected_count) {
        (void)fprintf(stderr, "FAILED: %s: %" PRIu64 ", not %" PRIu64 "\n", what, count, expected_count);
        failures++;
    }
}

/* A format from the 32 bits a pack holds: built from its bytes, because a cast of a number that is no enumerator
 * is what the analyser refuses. */
static enum weight_format_code format_of_bits(uint32_t bits)
{
	enum weight_format_code format = WEIGHT_FORMAT_NONE;
	memcpy(&format, &bits, sizeof(format));
	return format;
}

/* The same number: both not-a-number, or the same 64 bits, so that minus zero is not zero. */
static bool is_same_double(double first, double second)
{
    uint64_t first_bits = 0;
    uint64_t second_bits = 0;
    memcpy(&first_bits, &first, sizeof(first_bits));
    memcpy(&second_bits, &second, sizeof(second_bits));
    return isnan(first) || isnan(second) ? isnan(first) && isnan(second) : first_bits == second_bits;
}

/* ---- the formats this build carries ---- */

static void test_operations_of(void)
{
	static const bool is_carried[WEIGHT_FORMAT_COUNT] = {
		[WEIGHT_FORMAT_EXL3] = true,
		[WEIGHT_FORMAT_E2M1_SCALED] = true,
		[WEIGHT_FORMAT_E4M3_SCALED] = true,
	};
	static const uint32_t bits_outside_the_enumeration[] = {
		WEIGHT_FORMAT_COUNT,
		WEIGHT_FORMAT_COUNT + 1,
		0x100u + WEIGHT_FORMAT_E2M1_SCALED,
		INT32_MAX,
		0x80000000u,
		0x80000000u + WEIGHT_FORMAT_E4M3_SCALED,
		UINT32_MAX,
	};
	const struct weight_format_operations *const e2m1 =
		weight_format_operations_of(E2M1);
	const struct weight_format_operations *const e4m3 =
		weight_format_operations_of(E4M3);
	for (uint32_t bits = 0; bits < WEIGHT_FORMAT_COUNT; bits++) {
		const struct weight_format_operations *operations =
			weight_format_operations_of(format_of_bits(bits));
		expect((operations != NULL) == is_carried[bits],
		       "a format has a table exactly when this build carries it");
		expect(operations == NULL ||
			       (operations->name != NULL &&
				operations->name[0] != '\0' &&
				operations->stored_byte_count != NULL),
		       "the table of a format holds its name and its function");
	}
	expect(weight_format_operations_of(WEIGHT_FORMAT_NONE) == NULL,
	       "NONE is no format: no table");
	expect(weight_format_operations_of(WEIGHT_FORMAT_FLOAT16) == NULL &&
		       weight_format_operations_of(WEIGHT_FORMAT_INT4_SCALED) ==
			       NULL,
	       "the two formats this build does not carry have no table");
	expect(e2m1 != NULL && e4m3 != NULL && e2m1 != e4m3 &&
		       e2m1->name != NULL && e4m3->name != NULL &&
		       strcmp(e2m1->name, "e2m1_scaled") == 0 &&
		       strcmp(e4m3->name, "e4m3_scaled") == 0 &&
		       e2m1->stored_byte_count != e4m3->stored_byte_count,
	       "the two native formats have a table each, with the name and the function of the format");
	expect(weight_format_operations_of(WEIGHT_FORMAT_COUNT) == NULL,
	       "the number of entries is no format: no table");
	for (size_t value_index = 0;
	     value_index < COUNT_OF(bits_outside_the_enumeration);
	     value_index++) {
		expect(weight_format_operations_of(format_of_bits(
			       bits_outside_the_enumeration[value_index])) ==
			       NULL,
		       "a value outside the enumeration has no table");
	}
}

/* ---- the bytes of a matrix ---- */

struct counted_shape {
    const char *what;
    enum weight_format_code format;
    struct matrix_shape_record shape;
    uint64_t byte_count;                        /* computed by hand: see the constants */
};

static const struct counted_shape accepted_shapes[] = {
    {"4 bits: the smallest matrix", E2M1, {16, 64, 4, 0}, BYTES_OF_E2M1_16_BY_64},
    {"4 bits: an expert's matrix into its inner size", E2M1, {2304, 5120, 4, 0}, BYTES_OF_AN_EXPERT_MATRIX},
    {"4 bits: an expert's matrix out of its inner size", E2M1, {5120, 2304, 4, 0}, BYTES_OF_AN_EXPERT_MATRIX},
    {"4 bits: the largest sizes", E2M1, {LARGEST_OUTPUT_SIZE, LARGEST_E2M1_INPUT_SIZE, 4, 0},
     BYTES_OF_THE_LARGEST_E2M1},
    {"8 bits: the smallest matrix, half a block", E4M3, {16, 32, 8, 0}, BYTES_OF_E4M3_16_BY_32},
    {"8 bits: one whole block", E4M3, {32, 32, 8, 0}, BYTES_OF_E4M3_32_BY_32},
    {"8 bits: a block and a half by two groups", E4M3, {48, 64, 8, 0}, BYTES_OF_E4M3_48_BY_64},
    {"8 bits: [512, 5120]", E4M3, {512, 5120, 8, 0}, BYTES_OF_512_BY_5120},
    {"8 bits: [1280, 5120]", E4M3, {1280, 5120, 8, 0}, BYTES_OF_1280_BY_5120},
    {"8 bits: [32768, 1280]", E4M3, {32768, 1280, 8, 0}, BYTES_OF_32768_BY_1280},
    {"8 bits: [8192, 4096]", E4M3, {8192, 4096, 8, 0}, BYTES_OF_8192_BY_4096},
    {"8 bits: [5120, 8192]", E4M3, {5120, 8192, 8, 0}, BYTES_OF_5120_BY_8192},
    {"8 bits: [4096, 1280]", E4M3, {4096, 1280, 8, 0}, BYTES_OF_4096_BY_1280},
    {"8 bits: [25600, 6144]", E4M3, {25600, 6144, 8, 0}, BYTES_OF_25600_BY_6144},
    {"8 bits: [2304, 5120]", E4M3, {2304, 5120, 8, 0}, BYTES_OF_2304_BY_5120},
    {"8 bits: [5120, 2304]", E4M3, {5120, 2304, 8, 0}, BYTES_OF_5120_BY_2304},
    {"8 bits: [5120, 15360]", E4M3, {5120, 15360, 8, 0}, BYTES_OF_5120_BY_15360},
    {"8 bits: the most inputs of the most outputs that 64 bits count", E4M3,
     {LARGEST_OUTPUT_SIZE, MOST_E4M3_INPUTS_OF_THE_LARGEST_OUTPUT, 8, 0}, MOST_E4M3_BYTES_OF_THE_LARGEST_OUTPUT},
};

struct refused_shape {
    const char *what;
    enum weight_format_code format;
    struct matrix_shape_record shape;
};

static const struct refused_shape refused_shapes[] = {
    {"4 bits: no outputs", E2M1, {0, 64, 4, 0}},
    {"4 bits: no inputs", E2M1, {16, 0, 4, 0}},
    {"4 bits: half a tile of outputs", E2M1, {8, 64, 4, 0}},
    {"4 bits: a tile and a half of outputs", E2M1, {24, 64, 4, 0}},
    {"4 bits: half a load of inputs, which is one scale group", E2M1, {16, 32, 4, 0}},
    {"4 bits: a load and a half of inputs", E2M1, {16, 96, 4, 0}},
    {"4 bits: the bits per weight of the other format", E2M1, {16, 64, 8, 0}},
    {"4 bits: no bits per weight", E2M1, {16, 64, 0, 0}},
    {"4 bits: a reserved field that is not zero", E2M1, {16, 64, 4, 1}},
    {"8 bits: no outputs", E4M3, {0, 32, 8, 0}},
    {"8 bits: no inputs", E4M3, {16, 0, 8, 0}},
    {"8 bits: half a tile of outputs", E4M3, {8, 32, 8, 0}},
    {"8 bits: a tile and a half of outputs", E4M3, {24, 32, 8, 0}},
    {"8 bits: half a group of inputs", E4M3, {16, 16, 8, 0}},
    {"8 bits: a group and a half of inputs", E4M3, {16, 48, 8, 0}},
    {"8 bits: the bits per weight of the other format", E4M3, {16, 32, 4, 0}},
    {"8 bits: no bits per weight", E4M3, {16, 32, 0, 0}},
    {"8 bits: a reserved field that is not zero", E4M3, {16, 32, 8, 1}},
    {"8 bits: a byte count beyond 64 bits by one group of inputs", E4M3,
     {LARGEST_OUTPUT_SIZE, E4M3_INPUTS_BEYOND_64_BITS, 8, 0}},
    {"8 bits: a byte count beyond 64 bits, the largest sizes", E4M3,
     {LARGEST_OUTPUT_SIZE, LARGEST_E4M3_INPUT_SIZE, 8, 0}},
};

static void test_stored_byte_count(void)
{
    for (size_t shape_index = 0; shape_index < COUNT_OF(accepted_shapes); shape_index++) {
        const struct counted_shape *accepted = &accepted_shapes[shape_index];
	const struct weight_format_operations *operations =
		weight_format_operations_of(accepted->format);
	struct test_weight_format_sizes sizes = {0};
        uint64_t byte_count = 0;
        if (operations == NULL) {
            expect(false, accepted->what);
            continue;
        }
        expect_status(operations->stored_byte_count(&accepted->shape, &byte_count), NO_ERROR, accepted->what);
        expect_count(byte_count, accepted->byte_count, accepted->what);
        expect_status(test_weight_format_sizes_of(accepted->format, &accepted->shape, &sizes), NO_ERROR,
                      accepted->what);
        expect_count(sizes.stored_byte_count, accepted->byte_count, "the reference counts the bytes of a shape too");
        expect_count(sizes.weight_count, (uint64_t)accepted->shape.output_size * accepted->shape.input_size,
                     "a matrix has a weight for every output and input");
        byte_count = UNTOUCHED_COUNT;
        expect_status(operations->stored_byte_count(NULL, &byte_count), INVALID, "no shape is refused");
        expect_count(byte_count, UNTOUCHED_COUNT, "a call without a shape leaves the byte count untouched");
        expect_status(operations->stored_byte_count(&accepted->shape, NULL), INVALID, "no output is refused");
    }
    for (size_t shape_index = 0; shape_index < COUNT_OF(refused_shapes); shape_index++) {
        const struct refused_shape *refused = &refused_shapes[shape_index];
	const struct weight_format_operations *operations =
		weight_format_operations_of(refused->format);
	struct test_weight_format_sizes sizes = {.stored_byte_count = UNTOUCHED_COUNT};
        uint64_t byte_count = UNTOUCHED_COUNT;
        if (operations == NULL) {
            expect(false, refused->what);
            continue;
        }
        expect_status(operations->stored_byte_count(&refused->shape, &byte_count), INVALID, refused->what);
        expect_count(byte_count, UNTOUCHED_COUNT, "a shape that is refused leaves the byte count untouched");
        expect_status(test_weight_format_sizes_of(refused->format, &refused->shape, &sizes), INVALID, refused->what);
        expect(sizes.stored_byte_count == UNTOUCHED_COUNT && sizes.weight_count == 0 && sizes.scale_count == 0,
               "a shape that the reference refuses leaves its sizes untouched");
    }
}

/* ---- the values of the codes ---- */

struct known_value {
    uint32_t code;
    double value;
};

static const double e2m1_values_by_hand[TEST_WEIGHT_FORMAT_E2M1_CODE_COUNT] = {
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
};

static const struct known_value e4m3_values_by_hand[] = {
    {0x00, 0.0},
    {0x01, 0.001953125},                        /* the smallest: 1/8 * 2^-6 */
    {0x07, 0.013671875},                        /* the largest without a hidden one: 7/8 * 2^-6 */
    {0x08, 0.015625},                           /* the smallest with one: 2^-6 */
    {0x35, 0.8125},                             /* exponent 6, mantissa 5: (1 + 5/8) * 2^-1 */
    {0x38, 1.0},                                /* exponent 7 is the bias */
    {0x3C, 1.5},
    {0x40, 2.0},
    {0x77, 240.0},                              /* exponent 14, mantissa 7: (1 + 7/8) * 2^7 */
    {0x7E, 448.0},                              /* the largest: (1 + 6/8) * 2^8 */
    {0x80, -0.0},
    {0xB5, -0.8125},
    {0xB8, -1.0},
    {0xFE, -448.0},
};

static const struct known_value e8m0_values_by_hand[] = {
    {117, 0.0009765625},                        /* 2^-10 */
    {126, 0.5},
    {127, 1.0},
    {128, 2.0},
    {130, 8.0},
    {137, 1024.0},
};

/* The value of an E4M3 byte a second way: a whole number over 2^10, which is (8 + m) * 2^e, or 2 m where e is 0. */
static double e4m3_value_from_a_whole_number(uint32_t code)
{
    const uint32_t exponent = (code >> E4M3_EXPONENT_SHIFT) & E4M3_EXPONENT_MASK;
    const uint32_t mantissa = code & E4M3_MANTISSA_MASK;
    const uint32_t whole_number = exponent == 0 ? mantissa << 1u : (E4M3_HIDDEN_ONE + mantissa) << exponent;
    return ((code & E4M3_SIGN_BIT) != 0 ? -1.0 : 1.0) * (double)whole_number / E4M3_WHOLE_NUMBER_DIVISOR;
}

static void test_values(void)
{
    bool e4m3_holds = true;
    bool e8m0_holds = true;
    double power = 1.0;
    for (uint32_t code = 0; code < TEST_WEIGHT_FORMAT_E2M1_CODE_COUNT; code++) {
        expect(is_same_double(test_weight_format_e2m1_value(code), e2m1_values_by_hand[code]),
               "a 4-bit code has the value of the seam's list, and code 8 is minus zero");
    }
    expect(isnan(test_weight_format_e2m1_value(TEST_WEIGHT_FORMAT_E2M1_CODE_COUNT)) &&
               isnan(test_weight_format_e2m1_value(UINT32_MAX)),
           "a number above 15 is no 4-bit code: not-a-number");
    for (size_t value_index = 0; value_index < COUNT_OF(e4m3_values_by_hand); value_index++) {
        expect(is_same_double(test_weight_format_e4m3_value(e4m3_values_by_hand[value_index].code),
                              e4m3_values_by_hand[value_index].value),
               "an E4M3 byte has the value computed by hand");
    }
    for (uint32_t code = 0; code <= E4M3_LARGEST_CODE; code++) {
        const double value = test_weight_format_e4m3_value(code);
        e4m3_holds = e4m3_holds && is_same_double(value, e4m3_value_from_a_whole_number(code)) &&
                     is_same_double(test_weight_format_e4m3_value(code | E4M3_SIGN_BIT), -value) &&
                     (code == 0 || value > test_weight_format_e4m3_value(code - 1));
    }
    expect(e4m3_holds, "an E4M3 byte is its whole number over 1024, larger than the byte before it, and with the "
                       "sign bit the same number below zero");
    expect(isnan(test_weight_format_e4m3_value(TEST_WEIGHT_FORMAT_E4M3_NOT_A_NUMBER_MAGNITUDE)) &&
               isnan(test_weight_format_e4m3_value(TEST_WEIGHT_FORMAT_E4M3_NOT_A_NUMBER_MAGNITUDE | E4M3_SIGN_BIT)) &&
               isnan(test_weight_format_e4m3_value(TEST_WEIGHT_FORMAT_BYTE_CODE_COUNT)) &&
               isnan(test_weight_format_e4m3_value(UINT32_MAX)),
           "the E4M3 bytes 0x7F and 0xFF are not-a-number, and so is a number above 255");
    for (size_t value_index = 0; value_index < COUNT_OF(e8m0_values_by_hand); value_index++) {
        expect(is_same_double(test_weight_format_e8m0_value(e8m0_values_by_hand[value_index].code),
                              e8m0_values_by_hand[value_index].value),
               "an E8M0 byte has the value computed by hand");
    }
    /* from 2^0 upwards each byte doubles the one before, and downwards halves it: doubled and halved here */
    for (uint32_t code = E8M0_ONE; code <= E8M0_LARGEST; code++) {
        e8m0_holds = e8m0_holds && is_same_double(test_weight_format_e8m0_value(code), power) &&
                     is_same_double(test_weight_format_e8m0_value(E8M0_ONE + E8M0_ONE - code), 1.0 / power);
        power *= DOUBLING;
    }
    expect(e8m0_holds, "an E8M0 byte c is 2^(c - 127), from 2^-127 to 2^127");
    expect(isnan(test_weight_format_e8m0_value(TEST_WEIGHT_FORMAT_E8M0_NOT_A_NUMBER)) &&
               isnan(test_weight_format_e8m0_value(TEST_WEIGHT_FORMAT_BYTE_CODE_COUNT)) &&
               isnan(test_weight_format_e8m0_value(UINT32_MAX)),
           "the E8M0 byte 0xFF is not-a-number, and so is a number above 255");
}

/* ---- matrices: laid out, read back, changed ---- */

/* A matrix in the checkpoint's form and in two stored forms: the one under test, and a spare that is laid out over
 * other bytes or built by hand. */
struct matrix {
	enum weight_format_code format;
	struct matrix_shape_record shape;
	struct test_weight_format_sizes sizes;
	uint8_t *codes;
	uint8_t *scale_codes;
	uint8_t *stored;
	uint8_t *spare;
};

/* A matrix of zeros, which the caller ends with free_matrix. The test cannot go on without it. */
static struct matrix new_matrix(enum weight_format_code format,
				const struct matrix_shape_record *shape)
{
    struct matrix matrix = {.format = format, .shape = *shape};
    if (test_weight_format_sizes_of(format, shape, &matrix.sizes) != NO_ERROR ||
        matrix.sizes.stored_byte_count > MOST_TEST_BYTES || matrix.sizes.weight_count > MOST_TEST_BYTES) {
        expect(false, "the reference gives the sizes of the matrices of this test, and they are small");
        exit(EXIT_FAILURE);
    }
    matrix.codes = calloc((size_t)matrix.sizes.weight_count, 1);
    matrix.scale_codes = calloc((size_t)matrix.sizes.scale_count, 1);
    matrix.stored = calloc((size_t)matrix.sizes.stored_byte_count, 1);
    matrix.spare = calloc((size_t)matrix.sizes.stored_byte_count, 1);
    if (matrix.codes == NULL || matrix.scale_codes == NULL || matrix.stored == NULL || matrix.spare == NULL) {
        expect(false, "the test has memory for its matrices");
        exit(EXIT_FAILURE);
    }
    return matrix;
}

static void free_matrix(struct matrix *matrix)
{
    free(matrix->codes);
    free(matrix->scale_codes);
    free(matrix->stored);
    free(matrix->spare);
}

MUST_CHECK static result_status lay_out(const struct matrix *matrix,
					uint8_t *stored)
{
    return test_weight_format_lay_out(matrix->format, &matrix->shape, matrix->codes, matrix->sizes.weight_count,
                                      matrix->scale_codes, matrix->sizes.scale_count, stored,
                                      matrix->sizes.stored_byte_count);
}

/* Where the checkpoint keeps the scale of a weight: [output][input / 32] for 4 bits, [output / 32][input / 32]
 * for 8. */
static uint64_t checkpoint_scale_index(const struct matrix *matrix, uint64_t output_index, uint64_t input_index)
{
    const uint64_t scales_per_row = matrix->shape.input_size / INPUTS_PER_SCALE;
    const uint64_t row = matrix->format == E2M1 ? output_index : output_index / E4M3_OUTPUTS_PER_SCALE;
    return row * scales_per_row + input_index / INPUTS_PER_SCALE;
}

/* Reads every weight of a stored form and counts those that are not the weight of the checkpoint's form: its code,
 * the code of its scale, and the product of their two values. Also gives how many read a code and a scale code
 * that is not zero. */
static uint64_t wrong_weight_count(const struct matrix *matrix, const uint8_t *stored, uint64_t *code_count_out,
                                   uint64_t *scale_code_count_out)
{
    uint64_t wrong_count = 0;
    uint64_t code_count = 0;
    uint64_t scale_code_count = 0;
    /* over the codes as they are held, so that no index is formed that the count of the codes does not bound */
    for (uint64_t weight_index = 0; weight_index < matrix->sizes.weight_count; weight_index++) {
        const uint32_t output_index = (uint32_t)(weight_index / matrix->shape.input_size);
        const uint32_t input_index = (uint32_t)(weight_index % matrix->shape.input_size);
        struct test_weight_format_weight weight = {0};
        const uint32_t code = matrix->codes[weight_index];
        const uint32_t scale_code = matrix->scale_codes[checkpoint_scale_index(matrix, output_index, input_index)];
        const double value = (matrix->format == E2M1 ? test_weight_format_e2m1_value(code)
                                                     : test_weight_format_e4m3_value(code)) *
                             test_weight_format_e8m0_value(scale_code);
	const result_status status = test_weight_format_read_weight(
		matrix->format, &matrix->shape, stored,
		matrix->sizes.stored_byte_count, output_index, input_index,
		&weight);
	wrong_count += status != NO_ERROR || weight.code != code || weight.scale_code != scale_code ||
                               !is_same_double(weight.value, value)
                           ? 1u : 0u;
        code_count += weight.code != 0 ? 1u : 0u;
        scale_code_count += weight.scale_code != 0 ? 1u : 0u;
    }
    weights_read += matrix->sizes.weight_count;
    *code_count_out = code_count;
    *scale_code_count_out = scale_code_count;
    return wrong_count;
}

/* The weights that a stored bit belongs to: one for a bit of a code; for a bit of a scale, the 32 inputs of an
 * output (4 bits) or the 32 inputs of each output of its block, which holds 32 outputs or the 16 that are left. */
static uint64_t owner_count_of_bit(const struct matrix *matrix, uint64_t bit_index)
{
    const uint64_t weight_byte_count = matrix->sizes.stored_byte_count - matrix->sizes.scale_count;
    const uint64_t byte_index = bit_index / BITS_PER_BYTE;
    if (byte_index < weight_byte_count) {
        return 1;
    }
    if (matrix->format == E2M1) {
        return INPUTS_PER_SCALE;
    }
    const uint64_t block_index = (byte_index - weight_byte_count) / (matrix->shape.input_size / INPUTS_PER_SCALE);
    const uint64_t outputs_left = matrix->shape.output_size - block_index * E4M3_OUTPUTS_PER_SCALE;
    return INPUTS_PER_SCALE * (outputs_left < E4M3_OUTPUTS_PER_SCALE ? outputs_left : E4M3_OUTPUTS_PER_SCALE);
}

/* Flips one bit of the stored form: the read must differ from the checkpoint's form in exactly the weights that own
 * the bit. The bit is put back. */
static void expect_flip_is_seen(struct matrix *matrix, uint64_t bit_index, const char *what)
{
    const uint64_t byte_index = bit_index / BITS_PER_BYTE;
    const uint8_t flipped_bit = (uint8_t)(1u << (bit_index % BITS_PER_BYTE));
    uint64_t code_count = 0;
    uint64_t scale_code_count = 0;
    if (byte_index >= matrix->sizes.stored_byte_count) {
        expect(false, "a bit to flip lies inside the stored bytes");
        return;
    }
    matrix->stored[byte_index] = (uint8_t)(matrix->stored[byte_index] ^ flipped_bit);
    expect_count(wrong_weight_count(matrix, matrix->stored, &code_count, &scale_code_count),
                 owner_count_of_bit(matrix, bit_index), what);
    matrix->stored[byte_index] = (uint8_t)(matrix->stored[byte_index] ^ flipped_bit);
}

/* Fills the checkpoint's form from a hash of each position: every code of the format and every scale byte, the
 * not-a-number ones among them. */
static void fill_from_hash(struct matrix *matrix, uint64_t matrix_number)
{
    const uint64_t code_mask = matrix->format == E2M1 ? E2M1_CODE_MASK : BYTE_MASK;
    const uint64_t code_seed = pack_digest_mix(FILL_SEED + matrix_number);
    const uint64_t scale_seed = pack_digest_mix(~FILL_SEED + matrix_number);
    for (uint64_t weight_index = 0; weight_index < matrix->sizes.weight_count; weight_index++) {
	    matrix->codes[weight_index] =
		    (uint8_t)(pack_digest_term(code_seed, weight_index) &
			      code_mask);
    }
    for (uint64_t scale_index = 0; scale_index < matrix->sizes.scale_count; scale_index++) {
	    matrix->scale_codes[scale_index] =
		    (uint8_t)(pack_digest_term(scale_seed, scale_index) &
			      BYTE_MASK);
    }
}

/* Fills every code with one digit of the index of its weight, and every scale code with one digit of the index of
 * its scale. Two indices differ in a digit: a weight that reads its own digit back in every round was read from
 * its own place and scaled by its own scale, and from no other. False, with nothing filled, when no index of the
 * matrix has a digit of this number. */
static bool fill_from_index_digit(struct matrix *matrix, uint32_t digit_number)
{
    const uint32_t code_shift = (matrix->format == E2M1 ? E2M1_BITS_PER_CODE : BITS_PER_BYTE) * digit_number;
    const uint32_t scale_shift = BITS_PER_BYTE * digit_number;
    const uint64_t code_mask = matrix->format == E2M1 ? E2M1_CODE_MASK : BYTE_MASK;
    /* the shifts stay below 64 bits: the digits are counted first */
    if (digit_number >= MOST_INDEX_DIGITS ||
        (digit_number != 0 && ((matrix->sizes.weight_count - 1) >> code_shift) == 0 &&
         ((matrix->sizes.scale_count - 1) >> scale_shift) == 0)) {
        return false;
    }
    for (uint64_t weight_index = 0; weight_index < matrix->sizes.weight_count; weight_index++) {
        matrix->codes[weight_index] = (uint8_t)((weight_index >> code_shift) & code_mask);
    }
    for (uint64_t scale_index = 0; scale_index < matrix->sizes.scale_count; scale_index++) {
        matrix->scale_codes[scale_index] = (uint8_t)((scale_index >> scale_shift) & BYTE_MASK);
    }
    return true;
}

struct round_trip_shape {
    const char *what;
    enum weight_format_code format;
    struct matrix_shape_record shape;
    uint32_t drawn_flip_count;                  /* bits drawn from a hash, beside the four at the ends */
};

static const struct round_trip_shape round_trip_shapes[] = {
    {"4 bits: the smallest matrix, one tile by one load", E2M1, {16, 64, 4, 0}, 12},
    {"4 bits: one tile by two loads", E2M1, {16, 128, 4, 0}, 12},
    {"4 bits: two tiles by one load", E2M1, {32, 64, 4, 0}, 12},
    {"4 bits: three tiles by three loads", E2M1, {48, 192, 4, 0}, 12},
    {"4 bits: a sixteenth of an expert's wide matrix", E2M1, {144, 320, 4, 0}, 0},
    {"4 bits: the outputs of a sixteenth of an expert's tall matrix by whole loads", E2M1, {320, 192, 4, 0}, 0},
    {"8 bits: the smallest matrix, half a block by one group", E4M3, {16, 32, 8, 0}, 12},
    {"8 bits: one whole block", E4M3, {32, 32, 8, 0}, 12},
    {"8 bits: a block and a half by two groups", E4M3, {48, 64, 8, 0}, 12},
    {"8 bits: two blocks and a half by three groups", E4M3, {80, 96, 8, 0}, 12},
    {"8 bits: a sixteenth of an expert's wide matrix, with half a block last", E4M3, {144, 320, 8, 0}, 0},
    {"8 bits: a tall matrix of whole blocks", E4M3, {320, 192, 8, 0}, 0},
};

static void test_round_trips(void)
{
    for (size_t shape_index = 0; shape_index < COUNT_OF(round_trip_shapes); shape_index++) {
        const struct round_trip_shape *round_trip = &round_trip_shapes[shape_index];
	const struct weight_format_operations *operations =
		weight_format_operations_of(round_trip->format);
	struct matrix matrix = new_matrix(round_trip->format, &round_trip->shape);
        const uint64_t stored_bit_count = matrix.sizes.stored_byte_count * BITS_PER_BYTE;
        const uint64_t weight_bit_count = stored_bit_count - matrix.sizes.scale_count * BITS_PER_BYTE;
        uint64_t engine_byte_count = 0;
        uint64_t code_count = 0;
        uint64_t scale_code_count = 0;
        expect(operations != NULL && operations->stored_byte_count(&matrix.shape, &engine_byte_count) == NO_ERROR &&
                   engine_byte_count == matrix.sizes.stored_byte_count,
               "the engine and the reference count the same bytes for a shape");
        for (uint32_t digit_number = 0; fill_from_index_digit(&matrix, digit_number); digit_number++) {
            expect_status(lay_out(&matrix, matrix.stored), NO_ERROR, round_trip->what);
            expect_count(wrong_weight_count(&matrix, matrix.stored, &code_count, &scale_code_count), 0,
                         "every weight reads a digit of its own index back, and of the index of its own scale");
        }
        fill_from_hash(&matrix, shape_index);
        memset(matrix.spare, HIGH_BYTE, (size_t)matrix.sizes.stored_byte_count);
        expect_status(lay_out(&matrix, matrix.stored), NO_ERROR, round_trip->what);
        expect_status(lay_out(&matrix, matrix.spare), NO_ERROR, round_trip->what);
        expect(memcmp(matrix.stored, matrix.spare, (size_t)matrix.sizes.stored_byte_count) == 0,
               "a lay-out writes every stored byte: what the bytes held before does not show");
        expect_count(wrong_weight_count(&matrix, matrix.stored, &code_count, &scale_code_count), 0, round_trip->what);
        expect_flip_is_seen(&matrix, 0, "the first bit of the stored weights belongs to one weight");
        expect_flip_is_seen(&matrix, weight_bit_count - 1, "the last bit of the stored weights belongs to one weight");
        expect_flip_is_seen(&matrix, weight_bit_count, "the first bit of the stored scales belongs to its sharers");
        expect_flip_is_seen(&matrix, stored_bit_count - 1, "the last bit of the stored scales belongs to its sharers");
        for (uint32_t flip_index = 0; flip_index < round_trip->drawn_flip_count; flip_index++) {
		const uint64_t bit_index =
			pack_digest_mix(FLIP_SEED + shape_index + flip_index) %
			stored_bit_count;
		expect_flip_is_seen(
			&matrix, bit_index,
			"a stored bit belongs to one weight, or to the weights that share its scale");
        }
        expect_count(wrong_weight_count(&matrix, matrix.stored, &code_count, &scale_code_count), 0,
                     "a matrix whose flipped bits were put back reads as it was laid out");
        free_matrix(&matrix);
    }
}

/* ---- known answers: the place of a weight, worked out by hand from the formulas of the seam ---- */

struct known_place {
    uint32_t output_index;
    uint32_t input_index;
    uint64_t unit_index;                        /* the stored word (4 bits) or byte (8 bits) that holds the code */
    uint32_t bit_in_unit;                       /* the lowest bit of the code in it */
    uint64_t scale_byte_index;                  /* in the stored bytes */
    uint64_t sharer_count;                      /* the weights that share the scale */
};

/* 4 bits, [32, 128]: 2 tiles by 2 loads, 512 words of weights and then, from byte 2048 on, 32 words of scales.
 * A word is [(tile * 2 + load) * 32 + lane] * 4 + step with lane = 4 g + t; a scale word is (tile * 2 + load) * 8 + g.
 *   W[0][0]      tile 0, g 0, lower half; load 0, step 0, input 0 of the step: t 0, h 0, r 0: word 0, bit 0.
 *                Scale: word 0, byte 2 * 0 + 0.
 *   W[31][127]   tile 1, g 7, upper half; load 1, step 3, input 15 of the step: upper half step, t 3, h 1, so
 *                r = 1 + 2 = 3: lane 31, word (3 * 32 + 31) * 4 + 3 = 511, bit 4 * 3 + 16 = 28. Scale: word
 *                3 * 8 + 7 = 31, byte 2 * 1 + 1 = 3: 2048 + 124 + 3.
 *   W[9][70]     tile 0, g 1, upper half; load 1, step 0, input 6 of the step: lower half step, t 3, h 0, so
 *                r = 1: lane 7, word (1 * 32 + 7) * 4 + 0 = 156, bit 4. Scale: word 1 * 8 + 1 = 9, byte
 *                2 * 0 + 1 = 1: 2048 + 36 + 1.
 *   W[20][45]    tile 1, g 4, lower half; load 0, step 2, input 13 of the step: upper half step, t 2, h 1, so
 *                r = 0 + 2 = 2: lane 18, word (2 * 32 + 18) * 4 + 2 = 330, bit 4 * 2 + 16 = 24. Scale: word
 *                2 * 8 + 4 = 20, byte 2 * 1 + 0 = 2: 2048 + 80 + 2. */
static const struct known_place e2m1_known_places[] = {
    {0, 0, 0, 0, 2048, 32}, {31, 127, 511, 28, 2175, 32}, {9, 70, 156, 4, 2085, 32}, {20, 45, 330, 24, 2130, 32},
};

/* 8 bits, [48, 64]: 3 tiles by 2 groups, 3072 bytes of weights and then 2 blocks by 2 groups of scales; the second
 * block holds the 16 outputs that are left. A byte is [(tile * 2 + group) * 32 + lane] * 16 + b with lane = 4 g + t.
 *   W[0][0]      tile 0, g 0, lower half; group 0, input 0: quarter 0, t 0, even: b 0: byte 0. Scale: block 0,
 *                group 0.
 *   W[47][63]    tile 2, g 7, upper half; group 1, input 31 of it: quarter 3, t 3, odd: b = 12 + 2 + 1 = 15: lane 31,
 *                byte (5 * 32 + 31) * 16 + 15 = 3071. Scale: block 1, group 1: 3072 + 3.
 *   W[9][38]     tile 0, g 1, upper half; group 1, input 6 of it: quarter 0, t 3, even: b = 0 + 0 + 1 = 1: lane 7,
 *                byte (1 * 32 + 7) * 16 + 1 = 625. Scale: block 0, group 1: 3072 + 1.
 *   W[36][21]    tile 2, g 4, lower half; group 0, input 21: quarter 2, t 2, odd: b = 8 + 2 + 0 = 10: lane 18,
 *                byte (4 * 32 + 18) * 16 + 10 = 2346. Scale: block 1, group 0: 3072 + 2. */
static const struct known_place e4m3_known_places[] = {
    {0, 0, 0, 0, 3072, 1024}, {47, 63, 3071, 0, 3075, 512}, {9, 38, 625, 0, 3073, 1024}, {36, 21, 2346, 0, 3074, 512},
};

struct known_matrix {
    const char *what;
    enum weight_format_code format;
    struct matrix_shape_record shape;
    uint64_t stored_byte_count;
    uint32_t unit_byte_count;                   /* of a stored word of 4-bit codes, or of a stored byte */
    uint32_t code;
    uint32_t scale_code;
    double value;                               /* of the code times the scale, by hand */
    const struct known_place *places;
    size_t place_count;
};

/* 4-bit code 0xB is the sign and 1.5; E4M3 byte 0xB5 is the sign, exponent 6 and mantissa 5, -(1 + 5/8) / 2; the
 * scale byte 130 is 2^3. */
static const struct known_matrix known_matrices[] = {
    {"4 bits, [32, 128]", E2M1, {32, 128, 4, 0}, 2176, 4, 0xB, 130, -12.0, e2m1_known_places,
     COUNT_OF(e2m1_known_places)},
    {"8 bits, [48, 64]", E4M3, {48, 64, 8, 0}, 3076, 1, 0xB5, 130, -6.5, e4m3_known_places,
     COUNT_OF(e4m3_known_places)},
};

static void test_known_places(void)
{
    for (size_t matrix_index = 0; matrix_index < COUNT_OF(known_matrices); matrix_index++) {
        const struct known_matrix *known = &known_matrices[matrix_index];
        struct matrix matrix = new_matrix(known->format, &known->shape);
        expect_count(matrix.sizes.stored_byte_count, known->stored_byte_count, known->what);
        for (size_t place_index = 0; place_index < known->place_count; place_index++) {
            const struct known_place *place = &known->places[place_index];
            const uint64_t code_index = (uint64_t)place->output_index * known->shape.input_size + place->input_index;
            const uint64_t scale_index = checkpoint_scale_index(&matrix, place->output_index, place->input_index);
            const uint64_t code_byte_index =
                place->unit_index * known->unit_byte_count + place->bit_in_unit / BITS_PER_BYTE;
            struct test_weight_format_weight weight = {0};
            uint64_t code_count = 0;
            uint64_t scale_code_count = 0;
            if (place->output_index >= known->shape.output_size || place->input_index >= known->shape.input_size ||
                code_byte_index >= known->stored_byte_count || place->scale_byte_index >= known->stored_byte_count ||
                known->stored_byte_count != matrix.sizes.stored_byte_count) {
                expect(false, "a place computed by hand lies inside its matrix and inside the stored bytes");
                continue;
            }
            /* one weight and one scale that are not zero, in the checkpoint's form and, by hand, in the stored one */
            matrix.codes[code_index] = (uint8_t)known->code;
            matrix.scale_codes[scale_index] = (uint8_t)known->scale_code;
            matrix.spare[code_byte_index] = (uint8_t)(known->code << (place->bit_in_unit % BITS_PER_BYTE));
            matrix.spare[place->scale_byte_index] = (uint8_t)known->scale_code;
            memset(matrix.stored, HIGH_BYTE, (size_t)matrix.sizes.stored_byte_count);
            expect_status(lay_out(&matrix, matrix.stored), NO_ERROR, known->what);
            expect(memcmp(matrix.stored, matrix.spare, (size_t)matrix.sizes.stored_byte_count) == 0,
                   "the lay-out puts a weight and its scale where the formulas of the seam do, and nothing elsewhere");
            /* the read, from the bytes built by hand: no lay-out is behind it */
            expect_status(test_weight_format_read_weight(known->format, &known->shape, matrix.spare,
                                                         matrix.sizes.stored_byte_count, place->output_index,
                                                         place->input_index, &weight),
                          NO_ERROR, known->what);
            expect(weight.code == known->code && weight.scale_code == known->scale_code &&
                       is_same_double(weight.value, known->value),
                   "the read finds a weight and its scale where the formulas of the seam put them");
            expect_count(wrong_weight_count(&matrix, matrix.spare, &code_count, &scale_code_count), 0, known->what);
            expect_count(code_count, 1, "one code that is not zero is read as one weight");
            expect_count(scale_code_count, place->sharer_count, "one scale is read by every weight that shares it");
            matrix.codes[code_index] = 0;
            matrix.scale_codes[scale_index] = 0;
            matrix.spare[code_byte_index] = 0;
            matrix.spare[place->scale_byte_index] = 0;
        }
        free_matrix(&matrix);
    }
}

/* ---- what the reference refuses ---- */

static void test_reference_refusals(void)
{
	static const struct matrix_shape_record shape = { 16, 64, 4, 0 };
	static const uint32_t unknown_format_bits[] = {
		WEIGHT_FORMAT_NONE,  WEIGHT_FORMAT_FLOAT16,
		WEIGHT_FORMAT_EXL3,  WEIGHT_FORMAT_INT4_SCALED,
		WEIGHT_FORMAT_COUNT, UINT32_MAX,
	};
	const struct test_weight_format_weight untouched_weight = {
		.code = UINT32_MAX, .scale_code = UINT32_MAX
	};
	struct test_weight_format_weight weight = untouched_weight;
	struct test_weight_format_sizes sizes = { .stored_byte_count =
							  UNTOUCHED_COUNT };
	struct matrix matrix = new_matrix(E2M1, &shape);
	const uint64_t weight_count = matrix.sizes.weight_count;
	const uint64_t scale_count = matrix.sizes.scale_count;
	const uint64_t byte_count = matrix.sizes.stored_byte_count;
	for (size_t format_index = 0;
	     format_index < COUNT_OF(unknown_format_bits); format_index++) {
		const enum weight_format_code format =
			format_of_bits(unknown_format_bits[format_index]);
		expect_status(test_weight_format_sizes_of(format, &shape,
							  &sizes),
			      INVALID, "sizes of an unknown format");
		expect_status(test_weight_format_lay_out(
				      format, &shape, matrix.codes,
				      weight_count, matrix.scale_codes,
				      scale_count, matrix.stored, byte_count),
			      INVALID, "a lay-out of an unknown format");
		expect_status(test_weight_format_read_weight(
				      format, &shape, matrix.stored, byte_count,
				      0, 0, &weight),
			      INVALID, "a read of an unknown format");
	}
    expect_status(test_weight_format_sizes_of(E2M1, NULL, &sizes), INVALID, "sizes of no shape");
    expect_status(test_weight_format_sizes_of(E2M1, &shape, NULL), INVALID, "sizes without an output");
    expect(sizes.stored_byte_count == UNTOUCHED_COUNT && sizes.weight_count == 0 && sizes.scale_count == 0,
           "sizes that are refused are untouched");
    expect_status(test_weight_format_lay_out(E2M1, NULL, matrix.codes, weight_count, matrix.scale_codes, scale_count,
                                             matrix.stored, byte_count),
                  INVALID, "a lay-out of no shape");
    expect_status(test_weight_format_lay_out(E2M1, &shape, NULL, weight_count, matrix.scale_codes, scale_count,
                                             matrix.stored, byte_count),
                  INVALID, "a lay-out without codes");
    expect_status(test_weight_format_lay_out(E2M1, &shape, matrix.codes, weight_count, NULL, scale_count,
                                             matrix.stored, byte_count),
                  INVALID, "a lay-out without scale codes");
    expect_status(test_weight_format_lay_out(E2M1, &shape, matrix.codes, weight_count, matrix.scale_codes,
                                             scale_count, NULL, byte_count),
                  INVALID, "a lay-out without stored bytes");
    expect_status(test_weight_format_lay_out(E2M1, &shape, matrix.codes, weight_count - 1, matrix.scale_codes,
                                             scale_count, matrix.stored, byte_count),
                  INVALID, "a lay-out of one code too few");
    expect_status(test_weight_format_lay_out(E2M1, &shape, matrix.codes, weight_count, matrix.scale_codes,
                                             scale_count + 1, matrix.stored, byte_count),
                  INVALID, "a lay-out of one scale code too many");
    expect_status(test_weight_format_lay_out(E2M1, &shape, matrix.codes, weight_count, matrix.scale_codes,
                                             scale_count, matrix.stored, byte_count - 1),
                  INVALID, "a lay-out into one stored byte too few");
    matrix.codes[weight_count - 1] = (uint8_t)TEST_WEIGHT_FORMAT_E2M1_CODE_COUNT;
    expect_status(lay_out(&matrix, matrix.stored), INVALID, "a lay-out of a 4-bit code above 15");
    matrix.codes[weight_count - 1] = (uint8_t)(TEST_WEIGHT_FORMAT_E2M1_CODE_COUNT - 1);
    expect_status(lay_out(&matrix, matrix.stored), NO_ERROR, "a lay-out of the 4-bit code 15");
    expect_status(test_weight_format_read_weight(E2M1, NULL, matrix.stored, byte_count, 0, 0, &weight), INVALID,
                  "a read of no shape");
    expect_status(test_weight_format_read_weight(E2M1, &shape, NULL, byte_count, 0, 0, &weight), INVALID,
                  "a read without stored bytes");
    expect_status(test_weight_format_read_weight(E2M1, &shape, matrix.stored, byte_count + 1, 0, 0, &weight),
                  INVALID, "a read of one stored byte too many");
    expect_status(test_weight_format_read_weight(E2M1, &shape, matrix.stored, byte_count, shape.output_size, 0,
                                                 &weight),
                  INVALID, "a read of the output behind the last");
    expect_status(test_weight_format_read_weight(E2M1, &shape, matrix.stored, byte_count, 0, shape.input_size,
                                                 &weight),
                  INVALID, "a read of the input behind the last");
    expect_status(test_weight_format_read_weight(E2M1, &shape, matrix.stored, byte_count, 0, 0, NULL), INVALID,
                  "a read without an output");
    expect(weight.code == untouched_weight.code && weight.scale_code == untouched_weight.scale_code &&
               is_same_double(weight.value, untouched_weight.value),
           "a read that is refused leaves its output untouched");
    expect_status(test_weight_format_read_weight(E2M1, &shape, matrix.stored, byte_count, shape.output_size - 1,
                                                 shape.input_size - 1, &weight),
                  NO_ERROR, "a read of the last weight");
    expect(weight.code == TEST_WEIGHT_FORMAT_E2M1_CODE_COUNT - 1 && weight.scale_code == 0,
           "the last weight is the one that was laid out");
    free_matrix(&matrix);
}

int main(void)
{
    test_operations_of();
    test_stored_byte_count();
    test_values();
    test_reference_refusals();
    test_known_places();
    test_round_trips();
    if (failures != 0) {
        return EXIT_FAILURE;
    }
    (void)printf("weight_format_test passed (%" PRIu64 " weights read back)\n", weights_read);
    return EXIT_SUCCESS;
}
