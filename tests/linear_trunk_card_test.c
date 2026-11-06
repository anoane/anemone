/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * linear_trunk_card_test: on the real cards, the kernels of kernel_activation_quantize.cu and kernel_trunk_linear.cu
 * against the host reference of the linear (linear_reference.h), on synthetic numbers (linear_card_fixture.h): the
 * activation quantization, the product of a matrix of 8-bit weights with quantized rows and with rows that are not
 * quantized, and the shared expert of a layer.
 *
 * It needs the cards free (`make card_tests`): a card without room for the arena FAILS the test, it is never skipped.
 * On every card:
 *   - Quantization, for the widths 64, 192, 2304 and 5120 and for 1 to 8 rows of every kind of value the fixture
 *     makes (zeros of both signs, negative values that round to zero, blocks below the floor, 448 at its scale,
 *     ties, E4M3 subnormal codes, the largest finite bfloat16): every code and every exponent is the reference's,
 *     byte for byte, which finds each code by a search over all of them, and every operand the reference's carry
 *     of them (infinities where an exponent is above the carried form's, as the largest bfloat16 makes it).
 *   - The product, with quantized rows and with rows that are not quantized, for a small shape and for shapes of
 *     the model (trunk_cases): every output lies within the tolerance linear_reference.h states and justifies, and
 *     of all outputs of a kernel on a card at least 99.9% are bit-equal to the double result rounded once. The last
 *     row that is not quantized holds 65536 in two places, one in each block of 32 inputs of one load: an output
 *     those values reach must be no finite number, on the card as in the reference. It is an infinity where its two
 *     infinite terms agree in sign and a not-a-number where they do not, which a kernel must not store as a zero.
 *   - The shared expert (one launch of two products of the same quantized rows, the hidden step, the quantization
 *     of the hidden vectors, the product of down: four launches), small and of the model's shape. The rows' codes
 *     are the reference's. Gate, up and the output are held as products, the output as one of the card's own
 *     hidden codes. A hidden value is
 *     held against the reference's hidden step of the CARD'S OWN gate and up: within one last place (float32
 *     arithmetic and the card's exponential against double, with no subtraction: at most one rounding boundary of
 *     bfloat16 lies between them), and 99.9% of them bit-equal. The hidden codes are the reference's quantization of
 *     the card's own hidden values, byte for byte, and so their operands.
 *   - ROW EXACTNESS, for every launch above: a launch of R rows, R from 1 to 8, gives each row the bits it has in
 *     the launch of 8 rows, and every row launched alone has those bits too, also beside the row that holds 65536.
 *   - TWO MATRICES IN ONE LAUNCH: the matrix of a case of quantized rows with a second one of SECOND_OUTPUT_SIZE
 *     outputs, for R from 1 to 8 rows: the outputs of each are bit for bit those of its launch alone.
 *   - Each check shown failing once: between two launches one stored byte is changed ON THE CARD, and exactly what
 *     reads it must change. Quantization: the sign of one value, and of all codes and exponents exactly the one
 *     byte of its code differs, by its sign bit. A product: one weight (set to 448 of the other sign where row 0
 *     feeds its largest input): no output but that weight's changes; an output there must change wherever the bits
 *     it had are outside the tolerance of the reference's product with the changed weight, must stay where the row
 *     feeds a zero, and is held against the changed product; against the product before, the comparison fails. The
 *     shared expert: one weight of gate: of gate's outputs and of the hidden values only that weight's column
 *     changes, up not at all, and the whole chain is held again against the reference with the changed weight.
 *   - No launch writes what is not its own: the guards behind every range, the ranges a launch only reads, and the
 *     card's digest of the matrices are looked at in every case.
 * --draft_native runs only the draft's main_proj in native arithmetic, with the linear API's default slice count:
 * class 80 uses four slices, class 120 native sixteen, including when the physical PRO reports class 120.
 * One line per case says what was measured; a card's last line holds a digest of everything it computed, which is
 * the same for cards that computed the same bits (printed, not judged: whether two architectures agree is a result).
 */
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "anemone/backend_seam.h"
#include "anemone/linear.h"
#include "anemone/linear_records.h"
#include "anemone/model_layout.h"
#include "anemone/status.h"
#include "anemone/weight_format_seam.h"
#include "linear_card_fixture.h"
#include "linear_reference.h"
#include "weight_format_reference.h"

#define TEST_TIME_LIMIT_SECONDS 1800            /* the alarm ends the test here, whatever the driver does */
#define WORK_CAPACITY (16ull << 20u)
#define WEIGHTS_CAPACITY (80ull << 20u)         /* the draft's main_proj and a second matrix: 75.8 MiB */
#define COUNT_OF(table) ((uint32_t)(sizeof(table) / sizeof((table)[0])))
#define ROWS LINEAR_MAXIMUM_ROWS
#define PLAIN_ROWS 4u                           /* the first rows hold plain values, the others the edges */
#define VALUES_PER_BLOCK TEST_LINEAR_VALUES_PER_BLOCK
#define VALUE_BYTES ((uint64_t)LINEAR_BFLOAT16_BYTES)
#define MAXIMUM_FED_WIDTH 32768u                /* 8 segments of 4096 values */
#define MAXIMUM_OUTPUT_SIZE 8192u
#define E4M3 WEIGHT_FORMAT_E4M3_SCALED
#define E4M3_BITS_PER_WEIGHT 8u
#define E4M3_SIGN_BIT 0x80u
#define E4M3_LARGEST_MAGNITUDE_CODE 0x7Eu       /* 448 */
#define BFLOAT16_SIGN_BYTE_BIT 0x80u            /* the sign of a bfloat16, in its second byte */
#define BFLOAT16_SIGN_BIT 0x8000u
#define BFLOAT16_OF_65536 0x4780u               /* the least value a 16-bit float makes an infinity of */
#define BITS_PER_BYTE 8u
#define ACTIVATION_LIMIT 10.0f                  /* the model's */
#define HIDDEN_STEP_LAST_PLACES 1u              /* what a hidden value may differ from the reference's by */
#define PRODUCT_SCALE_CODE_BASE 118u            /* the scale codes of a product's matrix: 2^-9 to 2^-6 */
#define DOWN_SCALE_CODE_BASE 112u
#define SCALE_CODE_SPREAD 3u
#define QUANTIZER_STREAM 0x0A000000ull          /* the streams of the hashes: one for each use, far apart */
#define TRUNK_STREAM 0x0B000000ull
#define SHARED_STREAM 0x0C000000ull
#define CHANGE_STREAM 0x0D000000ull
#define STREAMS_PER_CASE 4096ull
#define CHANGED_ROW 3u                          /* the row of the value whose sign the quantization check changes */
#define SECOND_OUTPUT_SIZE 48u                  /* of the matrix a case's own shares its launches with */
#define SECOND_STREAM 0x0E000000ull
#define DRAFT_NATIVE_ARGUMENTS 2                 /* executable and --draft_native */

/* A product: output_size by input_size in slice_count slices, of quantized rows (segment_count 0) or of rows that
 * are not quantized, whose outputs are cut into segment_count runs. */
struct trunk_case {
    const char *name;
    uint32_t output_size;
    uint32_t input_size;
    uint32_t slice_count;
    uint32_t segment_count;
};

static const struct trunk_case trunk_cases[] = {
    {"80 by 320 in 4 slices", 80, 320, 4, 0},
    {"1280 by 5120 in 16 slices", 1280, 5120, 16, 0},
    {"512 by 5120 in 1 slice", 512, 5120, 1, 0},
    {"5120 by 2304 in 8 slices", 5120, 2304, 8, 0},
    {"5120 by 15360 in 4 slices (the draft's main_proj)", 5120, 15360, 4, 0},
    {"64 by 192 in 2 slices and 2 segments, rows not quantized", 64, 192, 2, 2},
    {"8192 by 4096 in 8 slices and 8 segments, rows not quantized", 8192, 4096, 8, 8},
};

/* A shared expert: gate and up of inner_size by hidden_size, down the other way, each in slice_count slices. */
struct shared_case {
    const char *name;
    uint32_t hidden_size;
    uint32_t inner_size;
    uint32_t slice_count;
    uint32_t gate_scale_code_base;              /* so that gate and up reach beyond the activation limit */
};

static const struct shared_case shared_cases[] = {
    {"256 to 128 and back in 2 slices", 256, 128, 2, 118},
    {"5120 to 2304 and back in 8 slices", 5120, 2304, 8, 115},
};

static const uint32_t quantizer_widths[] = {64, 192, 2304, 5120};

/* The arrays of a shared expert's chain, and its launches: the quantization of the rows, then the five of linear.h. */
enum chain_range {
    CHAIN_CODES,
    CHAIN_EXPONENTS,
    CHAIN_OPERANDS,
    CHAIN_GATE,
    CHAIN_UP,
    CHAIN_HIDDEN,
    CHAIN_HIDDEN_CODES,
    CHAIN_HIDDEN_EXPONENTS,
    CHAIN_HIDDEN_OPERANDS,
    CHAIN_OUTPUT,
    CHAIN_RANGE_COUNT
};
enum shared_matrix { MATRIX_GATE, MATRIX_UP, MATRIX_DOWN, MATRIX_COUNT };
#define CHAIN_LAUNCH_COUNT (1u + LINEAR_SHARED_EXPERT_LAUNCH_COUNT)

struct shared_chain {
	struct linear_range ranges[CHAIN_RANGE_COUNT];
	struct kernel_launch launches[CHAIN_LAUNCH_COUNT];
};

/* What a card's cases add up to: the products of each kernel, and the hidden steps. */
struct card_totals {
    struct test_linear_comparison quantized;
    struct test_linear_comparison unquantized;
    uint64_t hidden_count;
    uint64_t hidden_equal_count;
};

/* The rows of the case at hand, and what the reference makes of them. */
static uint16_t row_values[ROWS][MAXIMUM_FED_WIDTH];
static uint8_t row_codes[ROWS][MAXIMUM_FED_WIDTH];             /* in the order of the values */
static uint8_t row_exponents[ROWS][MAXIMUM_FED_WIDTH / VALUES_PER_BLOCK];
static double fed_values[ROWS * MAXIMUM_FED_WIDTH];            /* [row][width of the case] */
static uint8_t fed_exponents[ROWS * (MAXIMUM_FED_WIDTH / VALUES_PER_BLOCK)];
static struct test_linear_product products[ROWS * MAXIMUM_OUTPUT_SIZE];
static uint16_t outputs[ROWS * MAXIMUM_OUTPUT_SIZE];           /* what a card gave: [row][width of the case] */
static uint16_t second_outputs[ROWS * MAXIMUM_OUTPUT_SIZE];
static uint16_t third_outputs[ROWS * MAXIMUM_OUTPUT_SIZE];

/* A sub-range: bytes byte_offset .. byte_offset + byte_count - 1 of a range. */
static struct linear_range part_of(struct linear_range range,
				   uint64_t byte_offset, uint64_t byte_count)
{
	return (struct linear_range){ range.byte_offset + byte_offset,
				      byte_count };
}

/* Puts rows first_row .. of the reference's codes, exponents and operands into the image, the codes in lane order
 * and the operands in operand order. */
static bool put_quantized_rows(struct test_linear_card *card,
			       struct linear_range codes,
			       struct linear_range exponents,
			       struct linear_range operands, uint32_t width,
			       uint32_t first_row, uint32_t row_count)
{
    const uint32_t exponents_per_row = width / VALUES_PER_BLOCK;
    bool is_put = true;
    for (uint32_t row = 0; row < row_count; row++) {
        is_put = is_put && test_linear_card_put_codes(card, codes, (uint64_t)row * width, row_codes[first_row + row],
                                                      width) &&
                 test_linear_card_put(card, exponents, (uint64_t)row * exponents_per_row,
                                      row_exponents[first_row + row], exponents_per_row) &&
                 test_linear_card_put_operands(card, operands, (uint64_t)row * width * VALUE_BYTES,
                                               row_codes[first_row + row], row_exponents[first_row + row], width);
    }
    return is_put;
}

/* Whether received holds, at codes, exponents and operands, the reference's codes (in lane order), exponents and
 * operands of rows first_row .. first_row + row_count - 1. */
static bool expect_quantized_rows(const struct test_linear_card *card,
				  struct linear_range codes,
				  struct linear_range exponents,
				  struct linear_range operands, uint32_t width,
				  uint32_t first_row, uint32_t row_count,
				  const char *what)
{
    const uint32_t exponents_per_row = width / VALUES_PER_BLOCK;
    bool is_same = true;
    for (uint32_t row = 0; row < row_count && is_same; row++) {
        is_same = test_linear_card_expect_codes(card, codes, (uint64_t)row * width, row_codes[first_row + row],
                                                width, what) &&
                  test_linear_card_expect_bytes(card, exponents, (uint64_t)row * exponents_per_row,
                                                row_exponents[first_row + row], exponents_per_row, what) &&
                  test_linear_card_expect_operands(card, operands, (uint64_t)row * width * VALUE_BYTES,
                                                   row_codes[first_row + row], row_exponents[first_row + row], width,
                                                   what);
    }
    return is_same;
}

/* Quantizes rows first_row .. of row_values (row_codes, row_exponents) and decodes them for a product (fed_values,
 * fed_exponents, both [row_count][width]). */
static bool quantize_rows(uint32_t width, uint32_t first_row, uint32_t row_count)
{
    const uint32_t exponents_per_row = width / VALUES_PER_BLOCK;
    bool is_quantized = true;
    for (uint32_t row = 0; row < row_count; row++) {
        const uint32_t source_row = first_row + row;
	is_quantized = is_quantized &&
		       test_linear_quantize(row_values[source_row], width,
					    row_codes[source_row],
					    row_exponents[source_row]) ==
			       STATUS_OK &&
		       test_linear_fed_values_of_codes(
			       row_codes[source_row], width,
			       fed_values + (size_t)row * width) == STATUS_OK;
	memcpy(fed_exponents + (size_t)row * exponents_per_row, row_exponents[source_row], exponents_per_row);
    }
    return test_linear_expect(is_quantized, "the reference quantizes the rows of the case");
}

/* The eight rows of a case: plain values, then the edges; of rows that are not quantized, plain, then small
 * magnitudes, and a last row of plain values with 65536 in two places. */
static bool make_rows(uint32_t width, uint64_t stream, bool is_quantized)
{
    bool is_made = true;
    for (uint32_t row = 0; row < ROWS; row++) {
        const bool is_plain = row < PLAIN_ROWS || (!is_quantized && row == ROWS - 1);
        test_linear_values(row_values[row], width, stream + row,
                           is_plain ? TEST_LINEAR_VALUES_PLAIN
                                    : is_quantized ? TEST_LINEAR_VALUES_EDGES : TEST_LINEAR_VALUES_SMALL);
        if (!is_quantized) {
            if (row == ROWS - 1) {
                /* in both blocks of one load: where the two infinite terms differ in sign the sum is no number */
                const uint32_t infinite_place = (uint32_t)(test_linear_hash(stream, row) % width);
                row_values[row][infinite_place] = BFLOAT16_OF_65536;
                row_values[row][infinite_place ^ VALUES_PER_BLOCK] = BFLOAT16_OF_65536;
            }
	    is_made = is_made &&
		      test_linear_fed_values_of_bfloat16(
			      row_values[row], width,
			      fed_values + (size_t)row * width) == STATUS_OK;
	}
    }
    return is_quantized ? quantize_rows(width, 0, ROWS)
                        : test_linear_expect(is_made, "the reference decodes the rows of the case");
}

/* ---- the activation quantization ---- */

/* The two directions of the reference's lane order, held against each other: no card is asked. */
static void test_lane_order(void)
{
    uint8_t *const codes = row_codes[0];        /* in the order of the values */
    uint8_t *const stored = row_codes[1];       /* in lane order */
    bool is_same = true;
    for (uint32_t value_index = 0; value_index < MAXIMUM_FED_WIDTH; value_index++) {
        codes[value_index] = (uint8_t)test_linear_hash(QUANTIZER_STREAM, value_index);
    }
    is_same = test_linear_codes_lay_out(codes, MAXIMUM_FED_WIDTH, stored) ==
	      STATUS_OK;
    for (uint32_t value_index = 0; is_same && value_index < MAXIMUM_FED_WIDTH; value_index++) {
        const uint32_t place = test_linear_code_place(value_index);
        is_same = place < MAXIMUM_FED_WIDTH && stored[place] == codes[value_index];
    }
    (void)test_linear_expect(is_same, "the place of a code is where the lay-out put it");
}

static void test_quantizer(struct test_linear_card *card, uint32_t width)
{
	struct kernel_launch launches
		[ROWS +
		 ROWS]; /* of 1 to 8 rows, of rows 1 to 7 alone, and after the change */
	struct linear_range company_codes[ROWS];
	struct linear_range company_exponents[ROWS];
	struct linear_range company_operands[ROWS];
	struct linear_range alone_codes[ROWS];
	struct linear_range alone_exponents[ROWS];
	struct linear_range alone_operands[ROWS];
	const uint64_t row_value_bytes = (uint64_t)width * VALUE_BYTES;
	const uint32_t exponents_per_row = width / VALUES_PER_BLOCK;
	const uint32_t changed_value =
		(uint32_t)(test_linear_hash(CHANGE_STREAM, width) % width);
	const uint64_t sign_byte_offset =
		CHANGED_ROW * row_value_bytes +
		(uint64_t)changed_value * VALUE_BYTES + 1;
	uint64_t first_differing = TEST_LINEAR_NOTHING_DIFFERS;
	uint8_t changed_sign_byte = 0;
	uint32_t launch_count = 0;
	bool is_ready = true;

	test_linear_context("card %" PRIu32 ", the quantization of %" PRIu32
			    " values",
			    card->card_index, width);
	test_linear_card_clear(card);
	const struct linear_range values =
		test_linear_card_work(card, ROWS * row_value_bytes);
	const struct linear_range changed_codes =
		test_linear_card_work(card, (uint64_t)ROWS * width);
	const struct linear_range changed_exponents =
		test_linear_card_work(card, (uint64_t)ROWS * exponents_per_row);
	const struct linear_range changed_operands =
		test_linear_card_work(card, ROWS * row_value_bytes);
	for (uint32_t row = 0; row < ROWS; row++) {
		test_linear_values(row_values[row], width,
				   QUANTIZER_STREAM + width + row,
				   row == 0 ? TEST_LINEAR_VALUES_PLAIN :
					      TEST_LINEAR_VALUES_EXTREMES);
		is_ready = is_ready &&
			   test_linear_card_put(card, values,
						row * row_value_bytes,
						row_values[row],
						row_value_bytes);
	}
    is_ready = is_ready && quantize_rows(width, 0, ROWS);
    changed_sign_byte =
        (uint8_t)(((uint32_t)row_values[CHANGED_ROW][changed_value] >> BITS_PER_BYTE) ^ BFLOAT16_SIGN_BYTE_BIT);
    for (uint32_t row_count = 1; row_count <= ROWS; row_count++) {     /* the first row_count rows, in company */
	    struct linear_quantize_description description = {
		    .values = values, .width = width, .row_count = row_count
	    };
	    description.codes =
		    test_linear_card_work(card, (uint64_t)row_count * width);
	    description.exponents = test_linear_card_work(
		    card, (uint64_t)row_count * exponents_per_row);
	    description.operands =
		    test_linear_card_work(card, row_count * row_value_bytes);
	    company_codes[row_count - 1] = description.codes;
	    company_exponents[row_count - 1] = description.exponents;
	    company_operands[row_count - 1] = description.operands;
	    is_ready =
		    is_ready &&
		    test_linear_expect_status(
			    linear_quantize_prepare(
				    &description, card->arena_byte_count,
				    &card->kernels, &launches[launch_count]),
			    STATUS_OK,
			    "the quantization of rows in company is prepared");
	    launch_count++;
    }
    for (uint32_t row = 1; row < ROWS; row++) {                         /* each further row alone */
	    struct linear_quantize_description description = {
		    .values = part_of(values, row * row_value_bytes,
				      row_value_bytes),
		    .width = width,
		    .row_count = 1
	    };
	    description.codes = test_linear_card_work(card, width);
	    description.exponents =
		    test_linear_card_work(card, exponents_per_row);
	    description.operands = test_linear_card_work(card, row_value_bytes);
	    alone_codes[row] = description.codes;
	    alone_exponents[row] = description.exponents;
	    alone_operands[row] = description.operands;
	    is_ready = is_ready &&
		       test_linear_expect_status(
			       linear_quantize_prepare(
				       &description, card->arena_byte_count,
				       &card->kernels, &launches[launch_count]),
			       STATUS_OK,
			       "the quantization of a row alone is prepared");
	    launch_count++;
    }
    const struct linear_quantize_description changed_description = {
	    .values = values,
	    .codes = changed_codes,
	    .exponents = changed_exponents,
	    .operands = changed_operands,
	    .width = width,
	    .row_count = ROWS
    };
    is_ready = is_ready &&
	       test_linear_expect_status(
		       linear_quantize_prepare(
			       &changed_description, card->arena_byte_count,
			       &card->kernels, &launches[launch_count]),
		       STATUS_OK,
		       "the quantization after the change is prepared");
    /* every launch on the values as they are; then one sign changes on the card, and one launch more */
    if (!is_ready || !test_linear_card_has_room(card) || !test_linear_card_load(card) ||
        !test_linear_card_run(card, launches, launch_count) ||
        !test_linear_card_put(card, values, sign_byte_offset, &changed_sign_byte, 1) ||
        !test_linear_card_upload(card, values, sign_byte_offset, 1) ||
        !test_linear_card_run(card, &launches[launch_count], 1) || !test_linear_card_fetch(card)) {
        return;
    }
    for (uint32_t row_count = 1; row_count <= ROWS; row_count++) {
        (void)expect_quantized_rows(card, company_codes[row_count - 1], company_exponents[row_count - 1],
                                    company_operands[row_count - 1], width, 0, row_count, "rows quantized in company");
    }
    for (uint32_t row = 1; row < ROWS; row++) {
        (void)expect_quantized_rows(card, alone_codes[row], alone_exponents[row], alone_operands[row], width, row, 1,
                                    "a row quantized alone");
    }
    /* shown failing: exactly the code of the changed value differs from the launch before, by its sign bit */
    const uint64_t changed_place = (uint64_t)CHANGED_ROW * width + test_linear_code_place(changed_value);
    const uint64_t differing_count = test_linear_card_differing(
        card, changed_codes, 0, company_codes[ROWS - 1], 0, (uint64_t)ROWS * width, &first_differing);
    (void)test_linear_expect(differing_count == 1 && first_differing == changed_place,
                             "a changed sign changes exactly the byte of its code");
    (void)test_linear_card_expect_same(card, changed_exponents, 0, company_exponents[ROWS - 1], 0,
                                       (uint64_t)ROWS * exponents_per_row, "a changed sign changes no exponent");
    row_values[CHANGED_ROW][changed_value] = (uint16_t)(row_values[CHANGED_ROW][changed_value] ^ BFLOAT16_SIGN_BIT);
    if (quantize_rows(width, CHANGED_ROW, 1)) {
        (void)expect_quantized_rows(card, part_of(changed_codes, (uint64_t)CHANGED_ROW * width, width),
                                    part_of(changed_exponents, (uint64_t)CHANGED_ROW * exponents_per_row,
                                            exponents_per_row),
                                    part_of(changed_operands, CHANGED_ROW * row_value_bytes, row_value_bytes), width,
                                    CHANGED_ROW, 1, "the row of the changed sign");
    }
    (void)test_linear_card_expect_untouched(card, values, "the values");
    (void)printf("  quantization of %" PRIu32 " values: 1 to %u rows in company and alone are the reference's "
                 "bytes; a changed sign changed %" PRIu64 " byte\n", width, ROWS, differing_count);
}

/* ---- the products ---- */

/* The launch of a case's product for row_count rows: fed holds their codes (with their exponents and operands), or
 * their values. A product of quantized rows reads matrix_count matrices from matrix on, each into its output from
 * output on. */
static bool
prepare_product(const struct test_linear_card *card,
		const struct trunk_case *trunk_case,
		const struct linear_matrix *matrix, uint32_t matrix_count,
		struct linear_range fed, struct linear_range exponents,
		struct linear_range operands, const struct linear_range *output,
		uint32_t row_count, struct kernel_launch *launch_out)
{
    if (trunk_case->segment_count == 0) {
	    struct linear_trunk_quantized_description description = {
		    .codes = fed,
		    .exponents = exponents,
		    .operands = operands,
		    .matrix_count = matrix_count,
		    .row_count = row_count
	    };
	    for (uint32_t matrix_index = 0; matrix_index < matrix_count;
		 matrix_index++) {
		    description.matrices[matrix_index] = matrix[matrix_index];
		    description.outputs[matrix_index] = output[matrix_index];
	    }
	    return test_linear_expect_status(
		    linear_trunk_quantized_prepare(&description,
						   card->arena_byte_count,
						   &card->kernels, launch_out),
		    STATUS_OK, "a product of quantized rows is prepared");
    }
    const struct linear_trunk_bfloat16_description description = {
	    .matrix = *matrix,
	    .values = fed,
	    .output = *output,
	    .row_count = row_count,
	    .segment_count = trunk_case->segment_count
    };
    return test_linear_expect_status(
	    linear_trunk_bfloat16_prepare(&description, card->arena_byte_count,
					  &card->kernels, launch_out),
	    STATUS_OK, "a product of rows that are not quantized is prepared");
}

/* The input of a run of input_size inputs that row 0 feeds with its largest magnitude: values and scales as a
 * product takes them (scales NULL for rows that are not quantized). */
static uint32_t largest_fed_input(const double *values, const uint8_t *exponents, uint32_t input_size)
{
    uint32_t largest_input = 0;
    double largest = -1.0;
    for (uint32_t input_index = 0; input_index < input_size; input_index++) {
        const double scale =
            exponents != NULL ? test_weight_format_e8m0_value(exponents[input_index / VALUES_PER_BLOCK]) : 1.0;
        const double magnitude = (values[input_index] < 0.0 ? -values[input_index] : values[input_index]) * scale;
        if (magnitude > largest) {
            largest = magnitude;
            largest_input = input_index;
        }
    }
    return largest_input;
}

/* Sets the weight of an output and an input to 448 of the other sign, in the image and on the card. */
static bool change_a_weight(struct test_linear_card *card,
			    const struct linear_matrix *matrix,
			    uint32_t output_index, uint32_t input_index)
{
    struct test_weight_format_weight weight = {0};
    uint64_t byte_offset = 0;
    const uint8_t *const stored = test_linear_card_stored(card, matrix->stored);
    return test_linear_expect(
		   stored != NULL &&
			   test_weight_format_read_weight(
				   E4M3, &matrix->shape, stored,
				   matrix->stored.byte_count, output_index,
				   input_index, &weight) == STATUS_OK,
		   "the reference reads the weight that is to change") &&
	   test_linear_card_set_weight(
		   card, E4M3, &matrix->shape, matrix->stored, output_index,
		   input_index,
		   ((weight.code ^ E4M3_SIGN_BIT) & E4M3_SIGN_BIT) |
			   E4M3_LARGEST_MAGNITUDE_CODE,
		   &byte_offset) &&
	   test_linear_card_upload(card, matrix->stored, byte_offset, 1);
}

/* A product shown failing: see the top. outputs and products hold the launch of 8 rows before the change. */
static void show_a_changed_weight(struct test_linear_card *card,
				  const struct trunk_case *trunk_case,
				  const struct linear_matrix *matrix,
				  const struct kernel_launch *changed_launch,
				  struct linear_range changed_output,
				  uint32_t segment_count)
{
    static struct test_linear_product changed_products[ROWS];
    const uint32_t output_size = trunk_case->output_size;
    const uint32_t input_size = trunk_case->input_size;
    const bool is_quantized = trunk_case->segment_count == 0;
    const uint32_t output_index = (uint32_t)(test_linear_hash(CHANGE_STREAM, output_size) % output_size);
    const size_t segment = output_index / (output_size / segment_count);
    const uint32_t input_index =
        largest_fed_input(fed_values + segment * input_size,
                          is_quantized ? fed_exponents + segment * (input_size / VALUES_PER_BLOCK) : NULL, input_size);
    uint64_t stray_count = 0;
    uint32_t must_change_count = 0;
    uint32_t shown_failing_count = 0;
    uint32_t wrong_count = 0;

    if (!change_a_weight(card, matrix, output_index, input_index) ||
	!test_linear_card_run(card, changed_launch, 1) ||
	!test_linear_card_fetch(card) ||
	!test_linear_card_get(card, changed_output, 0, second_outputs,
			      (uint64_t)ROWS * output_size * VALUE_BYTES) ||
	!test_linear_expect_status(
		test_linear_products(
			E4M3, &matrix->shape,
			test_linear_card_stored(card, matrix->stored),
			matrix->stored.byte_count, output_index, 1, fed_values,
			is_quantized ? fed_exponents : NULL, ROWS,
			segment_count, changed_products),
		STATUS_OK,
		"the reference computes the outputs of the changed weight")) {
	    return;
    }
    for (size_t row = 0; row < ROWS; row++) {
        const size_t at_weight = row * output_size + output_index;
        const struct test_linear_product *const before = &products[at_weight];
        const struct test_linear_product *const after = &changed_products[row];
        const bool is_changed = second_outputs[at_weight] != outputs[at_weight];
        const bool is_fed_a_zero = before->sum == after->sum && before->magnitude_sum == after->magnitude_sum;
        for (size_t other = 0; other < output_size; other++) {
            stray_count += other != output_index && second_outputs[row * output_size + other] !=
                                                        outputs[row * output_size + other] ? 1u : 0u;
        }
        /* the bits the output had are no answer to the changed product: it has to change */
        const bool must_change = !test_linear_output_is_within_tolerance(outputs[at_weight], after, input_size,
                                                                         trunk_case->slice_count);
        must_change_count += must_change ? 1u : 0u;
        wrong_count += (must_change && !is_changed) || (is_fed_a_zero && is_changed) ||
                       !test_linear_output_is_within_tolerance(second_outputs[at_weight], after, input_size,
                                                               trunk_case->slice_count) ? 1u : 0u;
        shown_failing_count += !test_linear_output_is_within_tolerance(second_outputs[at_weight], before, input_size,
                                                                       trunk_case->slice_count) ? 1u : 0u;
    }
    (void)test_linear_expect(stray_count == 0, "a changed weight changes no output but its own");
    (void)test_linear_expect(wrong_count == 0, "the output of a changed weight changes where it must, stays where "
                                               "its row feeds a zero, and is the changed product");
    (void)test_linear_expect(must_change_count != 0 && shown_failing_count != 0,
                             "the change is large enough to fail the comparison with the product before");
    (void)printf("  one weight changed on the card (output %" PRIu32 ", input %zu): %" PRIu32 " of %u rows had to "
                 "change and did, %" PRIu32 " fail the comparison with the product before, %" PRIu64 " other "
                 "outputs changed\n", output_index, segment * input_size + input_index, must_change_count, ROWS,
                 shown_failing_count, stray_count);
}

/* Two matrices in one launch: the case's matrix and a second one, each against its launch alone of 8 rows, whose
 * first rows are those of a launch of fewer (row exactness). The arena is laid out again: the image holds what the
 * rows feed, and every range set aside here is new. */
static void show_two_matrices(struct test_linear_card *card,
			      const struct trunk_case *trunk_case,
			      const struct linear_matrix *matrix,
			      struct linear_range fed,
			      struct linear_range exponents,
			      struct linear_range operands, uint64_t stream)
{
	struct kernel_launch launches[LINEAR_MERGED_MATRICES +
				      ROWS]; /* each alone, then both */
	struct linear_range alone_outputs[LINEAR_MERGED_MATRICES];
	struct linear_range merged_outputs[ROWS][LINEAR_MERGED_MATRICES];
	struct linear_matrix matrices[LINEAR_MERGED_MATRICES] = { *matrix,
								  *matrix };
	const uint64_t row_bytes[LINEAR_MERGED_MATRICES] = {
		trunk_case->output_size * VALUE_BYTES,
		SECOND_OUTPUT_SIZE * VALUE_BYTES
	};
	const struct test_linear_matrix_recipe recipe = {
		.format = E4M3,
		.shape = { SECOND_OUTPUT_SIZE, trunk_case->input_size,
			   E4M3_BITS_PER_WEIGHT, 0 },
		.stream = stream + SECOND_STREAM,
		.code = test_linear_e4m3_code,
		.scale_code_base = PRODUCT_SCALE_CODE_BASE,
		.scale_code_spread = SCALE_CODE_SPREAD
	};
	matrices[1].shape = recipe.shape;
	bool is_ready =
		test_linear_card_matrix(card, &recipe, &matrices[1].stored);
	for (uint32_t matrix_index = 0;
	     is_ready && matrix_index < LINEAR_MERGED_MATRICES;
	     matrix_index++) {
		alone_outputs[matrix_index] = test_linear_card_work(
			card, ROWS * row_bytes[matrix_index]);
		is_ready = prepare_product(card, trunk_case,
					   &matrices[matrix_index], 1, fed,
					   exponents, operands,
					   &alone_outputs[matrix_index], ROWS,
					   &launches[matrix_index]);
	}
    for (uint32_t row_count = 1; is_ready && row_count <= ROWS; row_count++) {
	    for (uint32_t matrix_index = 0;
		 matrix_index < LINEAR_MERGED_MATRICES; matrix_index++) {
		    merged_outputs[row_count - 1][matrix_index] =
			    test_linear_card_work(
				    card, row_count * row_bytes[matrix_index]);
	    }
	    is_ready = prepare_product(
		    card, trunk_case, matrices, LINEAR_MERGED_MATRICES, fed,
		    exponents, operands, merged_outputs[row_count - 1],
		    row_count,
		    &launches[LINEAR_MERGED_MATRICES + row_count - 1]);
    }
    if (!is_ready || !test_linear_card_has_room(card) ||
	!test_linear_card_load(card) ||
	!test_linear_card_run(card, launches, LINEAR_MERGED_MATRICES + ROWS) ||
	!test_linear_card_fetch(card)) {
	    (void)test_linear_expect(
		    is_ready, "the launches of two matrices are laid out");
	    return;
    }
    for (uint32_t index = 0; index < ROWS * LINEAR_MERGED_MATRICES; index++) {
	    const uint32_t row_count = index / LINEAR_MERGED_MATRICES + 1;
	    const uint32_t matrix_index = index % LINEAR_MERGED_MATRICES;
	    (void)test_linear_card_expect_same(
		    card, merged_outputs[row_count - 1][matrix_index], 0,
		    alone_outputs[matrix_index], 0,
		    row_count * row_bytes[matrix_index],
		    "each of two matrices in one launch has the bits it has alone");
    }
    (void)printf("  with a second matrix of %u outputs in one launch, 1 to %u rows: the bits of each alone\n",
                 SECOND_OUTPUT_SIZE, ROWS);
}

static void test_trunk_case(struct test_linear_card *card, const struct trunk_case *trunk_case, uint32_t case_index,
                            struct card_totals *totals)
{
	struct kernel_launch launches
		[ROWS +
		 ROWS]; /* of 1 to 8 rows, of rows 1 to 7 alone, and after the change */
	struct linear_range company_output[ROWS];
	struct linear_range alone_output[ROWS];
	struct linear_range alone_exponents[ROWS];
	const bool is_quantized = trunk_case->segment_count == 0;
	const uint32_t segment_count = is_quantized ? 1 :
						      trunk_case->segment_count;
	const uint32_t output_size = trunk_case->output_size;
	const uint32_t fed_width = segment_count * trunk_case->input_size;
	const uint32_t exponents_per_row = fed_width / VALUES_PER_BLOCK;
	const uint64_t fed_row_bytes = is_quantized ? fed_width :
						      fed_width * VALUE_BYTES;
	const uint64_t output_row_bytes = output_size * VALUE_BYTES;
	const uint64_t stream = TRUNK_STREAM + case_index * STREAMS_PER_CASE;
	const struct test_linear_matrix_recipe recipe = {
		.format = E4M3,
		.shape = { output_size, trunk_case->input_size,
			   E4M3_BITS_PER_WEIGHT, 0 },
		.stream = stream,
		.code = test_linear_e4m3_code,
		.scale_code_base = PRODUCT_SCALE_CODE_BASE,
		.scale_code_spread = SCALE_CODE_SPREAD
	};
	struct linear_matrix matrix = { .shape = recipe.shape,
					.weight_format = E4M3,
					.slice_count =
						trunk_case->slice_count };
	uint32_t launch_count = 0;

	test_linear_context("card %" PRIu32 ", the product %s",
			    card->card_index, trunk_case->name);
	(void)printf("%s\n", trunk_case->name);
	test_linear_card_clear(card);
	const struct linear_range fed =
		test_linear_card_work(card, ROWS * fed_row_bytes);
	const struct linear_range exponents =
		is_quantized ?
			test_linear_card_work(card, (uint64_t)ROWS *
							    exponents_per_row) :
			fed;
	const struct linear_range operands =
		is_quantized ?
			test_linear_card_work(card, (uint64_t)ROWS * fed_width *
							    VALUE_BYTES) :
			fed;
	const struct linear_range changed_output =
		test_linear_card_work(card, ROWS * output_row_bytes);
	bool is_ready =
		fed_width <= MAXIMUM_FED_WIDTH &&
		output_size <= MAXIMUM_OUTPUT_SIZE &&
		test_linear_card_matrix(card, &recipe, &matrix.stored) &&
		make_rows(fed_width, stream, is_quantized);
	if (is_ready && is_quantized) {
		is_ready = put_quantized_rows(card, fed, exponents, operands,
					      fed_width, 0, ROWS);
	}
    for (uint32_t row = 0; is_ready && !is_quantized && row < ROWS; row++) {
        is_ready = test_linear_card_put(card, fed, row * fed_row_bytes, row_values[row], fed_row_bytes);
    }
    for (uint32_t row_count = 1; is_ready && row_count <= ROWS; row_count++) {     /* the first rows, in company */
        company_output[row_count - 1] = test_linear_card_work(card, row_count * output_row_bytes);
        is_ready = prepare_product(card, trunk_case, &matrix, 1, fed, exponents, operands,
                                   &company_output[row_count - 1], row_count, &launches[launch_count]);
        launch_count++;
    }
    for (uint32_t row = 1; is_ready && row < ROWS; row++) {                         /* each further row alone */
        alone_output[row] = test_linear_card_work(card, output_row_bytes);
        alone_exponents[row] = is_quantized ? test_linear_card_work(card, exponents_per_row) : fed;
        is_ready = (!is_quantized || test_linear_card_put(card, alone_exponents[row], 0, row_exponents[row],
                                                          exponents_per_row)) &&
                   prepare_product(card, trunk_case, &matrix, 1, part_of(fed, row * fed_row_bytes, fed_row_bytes),
                                   alone_exponents[row],
                                   is_quantized ? part_of(operands, (uint64_t)row * fed_width * VALUE_BYTES,
                                                          fed_width * VALUE_BYTES) : fed,
                                   &alone_output[row], 1, &launches[launch_count]);
        launch_count++;
    }
    if (!is_ready || !test_linear_card_has_room(card) ||
	!prepare_product(card, trunk_case, &matrix, 1, fed, exponents, operands,
			 &changed_output, ROWS, &launches[launch_count]) ||
	!test_linear_expect_status(
		test_linear_products(
			E4M3, &matrix.shape,
			test_linear_card_stored(card, matrix.stored),
			matrix.stored.byte_count, 0, output_size, fed_values,
			is_quantized ? fed_exponents : NULL, ROWS,
			segment_count, products),
		STATUS_OK, "the reference computes the products of the case") ||
	!test_linear_card_load(card) ||
	!test_linear_card_run(card, launches, launch_count) ||
	!test_linear_card_fetch(card) ||
	!test_linear_card_get(card, company_output[ROWS - 1], 0, outputs,
			      ROWS * output_row_bytes)) {
	    (void)test_linear_expect(is_ready, "the case is laid out");
	    return;
    }
    struct test_linear_comparison *const total = is_quantized ? &totals->quantized : &totals->unquantized;
    const uint64_t finite_before = total->output_count;
    test_linear_expect_products("8 rows", outputs, products, (uint64_t)ROWS * output_size, trunk_case->input_size,
                    trunk_case->slice_count, total);
    /* the tolerance holds an output to be no finite number where the reference's is none: some must be none, and
     * some of them a not-a-number */
    uint64_t not_a_number_count = 0;
    for (size_t index = (size_t)(ROWS - 1) * output_size; index < (size_t)ROWS * output_size; index++) {
        not_a_number_count +=
            test_linear_bfloat16_nearest(products[index].sum) == TEST_LINEAR_BFLOAT16_NOT_A_NUMBER ? 1u : 0u;
    }
    (void)test_linear_expect(is_quantized || (total->output_count - finite_before < (uint64_t)ROWS * output_size &&
                                              not_a_number_count != 0),
                             "the two 65536 of the last row reach outputs of the reference, some as a not-a-number");
    for (uint32_t row_count = 1; row_count < ROWS; row_count++) {
        (void)test_linear_card_expect_same(card, company_output[row_count - 1], 0, company_output[ROWS - 1], 0,
                                           row_count * output_row_bytes,
                                           "rows in a smaller company keep the bits they have among 8");
    }
    for (uint32_t row = 1; row < ROWS; row++) {
        (void)test_linear_card_expect_same(card, alone_output[row], 0, company_output[ROWS - 1],
                                           row * output_row_bytes, output_row_bytes,
                                           "a row alone keeps the bits it has among 8");
        if (is_quantized) {
            (void)test_linear_card_expect_untouched(card, alone_exponents[row], "the exponents of a row alone");
        }
    }
    (void)test_linear_card_expect_untouched(card, fed, "what the rows feed");
    (void)test_linear_card_expect_untouched(card, exponents, "the exponents of the rows");
    (void)test_linear_card_expect_untouched(card, operands, "the operands of the rows");
    (void)test_linear_card_expect_weights_untouched(card);
    if (is_quantized) {
        show_two_matrices(card, trunk_case, &matrix, fed, exponents, operands, stream);
        (void)test_linear_card_expect_weights_untouched(card);
    }
    show_a_changed_weight(card, trunk_case, &matrix, &launches[launch_count], changed_output, segment_count);
    (void)test_linear_card_expect_weights_untouched(card);
}

/* ---- the shared expert ---- */

/* The bytes one row has in an array of a chain. */
static uint64_t chain_row_bytes(const struct shared_case *shared_case, enum chain_range range)
{
    switch (range) {
    case CHAIN_CODES:
        return shared_case->hidden_size;
    case CHAIN_EXPONENTS:
        return shared_case->hidden_size / VALUES_PER_BLOCK;
    case CHAIN_OPERANDS:
        return shared_case->hidden_size * VALUE_BYTES;
    case CHAIN_HIDDEN_CODES:
        return shared_case->inner_size;
    case CHAIN_HIDDEN_EXPONENTS:
        return shared_case->inner_size / VALUES_PER_BLOCK;
    case CHAIN_OUTPUT:
        return shared_case->hidden_size * VALUE_BYTES;
    case CHAIN_GATE:
    case CHAIN_UP:
    case CHAIN_HIDDEN:
    case CHAIN_HIDDEN_OPERANDS:
    case CHAIN_RANGE_COUNT:
    default:
        return shared_case->inner_size * VALUE_BYTES;
    }
}

/* Sets the arrays of a chain of row_count rows aside and prepares its launches: values holds the rows. */
static bool chain_prepare(struct test_linear_card *card,
			  const struct shared_case *shared_case,
			  const struct linear_matrix *matrices,
			  struct linear_range values, uint32_t row_count,
			  struct shared_chain *chain)
{
    for (uint32_t range = 0; range < CHAIN_RANGE_COUNT; range++) {
        chain->ranges[range] =
            test_linear_card_work(card, row_count * chain_row_bytes(shared_case, (enum chain_range)range));
    }
    const struct linear_quantize_description quantize = {
	    .values = values,
	    .codes = chain->ranges[CHAIN_CODES],
	    .exponents = chain->ranges[CHAIN_EXPONENTS],
	    .operands = chain->ranges[CHAIN_OPERANDS],
	    .width = shared_case->hidden_size,
	    .row_count = row_count
    };
    const struct linear_shared_expert_description expert = {
	    .gate = matrices[MATRIX_GATE],
	    .up = matrices[MATRIX_UP],
	    .down = matrices[MATRIX_DOWN],
	    .codes = chain->ranges[CHAIN_CODES],
	    .exponents = chain->ranges[CHAIN_EXPONENTS],
	    .operands = chain->ranges[CHAIN_OPERANDS],
	    .gate_output = chain->ranges[CHAIN_GATE],
	    .up_output = chain->ranges[CHAIN_UP],
	    .hidden = chain->ranges[CHAIN_HIDDEN],
	    .hidden_codes = chain->ranges[CHAIN_HIDDEN_CODES],
	    .hidden_exponents = chain->ranges[CHAIN_HIDDEN_EXPONENTS],
	    .hidden_operands = chain->ranges[CHAIN_HIDDEN_OPERANDS],
	    .output = chain->ranges[CHAIN_OUTPUT],
	    .row_count = row_count,
	    .activation_limit = ACTIVATION_LIMIT
    };
    return test_linear_card_has_room(card) &&
	   test_linear_expect_status(
		   linear_quantize_prepare(&quantize, card->arena_byte_count,
					   &card->kernels, &chain->launches[0]),
		   STATUS_OK,
		   "the quantization of a chain's rows is prepared") &&
	   test_linear_expect_status(
		   linear_shared_expert_prepare(&expert, card->arena_byte_count,
						&card->kernels,
						&chain->launches[1]),
		   STATUS_OK, "the shared expert is prepared");
}

/* One product of a chain of 8 rows against the reference, from fed_values and fed_exponents: into card_outputs. */
static bool chain_product(const struct test_linear_card *card,
			  const struct shared_case *shared_case,
			  const struct linear_matrix *matrix,
			  struct linear_range output, const char *what,
			  uint16_t *card_outputs, struct card_totals *totals)
{
    const uint64_t count = (uint64_t)ROWS * matrix->shape.output_size;
    if (!test_linear_expect_status(
		test_linear_products(
			E4M3, &matrix->shape,
			test_linear_card_stored(card, matrix->stored),
			matrix->stored.byte_count, 0, matrix->shape.output_size,
			fed_values, fed_exponents, ROWS, 1, products),
		STATUS_OK, "the reference computes a product of the chain") ||
	!test_linear_card_get(card, output, 0, card_outputs,
			      count * VALUE_BYTES)) {
	    return false;
    }
    test_linear_expect_products(what, card_outputs, products, count, matrix->shape.input_size, shared_case->slice_count,
                    &totals->quantized);
    return true;
}

/* A chain of 8 rows against the reference: see the top. */
static void chain_check(struct test_linear_card *card,
			const struct shared_case *shared_case,
			const struct linear_matrix *matrices,
			const struct shared_chain *chain, uint64_t stream,
			struct card_totals *totals)
{
    const uint32_t hidden_size = shared_case->hidden_size;
    const uint32_t inner_size = shared_case->inner_size;
    const uint64_t hidden_count = (uint64_t)ROWS * inner_size;
    uint64_t equal_count = 0;
    uint64_t far_count = 0;
    if (!make_rows(hidden_size, stream, true) ||
        !expect_quantized_rows(card, chain->ranges[CHAIN_CODES], chain->ranges[CHAIN_EXPONENTS],
                               chain->ranges[CHAIN_OPERANDS], hidden_size, 0, ROWS,
                               "the rows of a chain, quantized by the card") ||
        !chain_product(card, shared_case, &matrices[MATRIX_GATE], chain->ranges[CHAIN_GATE], "gate", outputs,
                       totals) ||
        !chain_product(card, shared_case, &matrices[MATRIX_UP], chain->ranges[CHAIN_UP], "up", second_outputs,
                       totals) ||
        !test_linear_card_get(card, chain->ranges[CHAIN_HIDDEN], 0, third_outputs, hidden_count * VALUE_BYTES)) {
        return;
    }
    for (uint64_t index = 0; index < hidden_count; index++) {
        const uint16_t expected = test_linear_bfloat16_nearest(
            test_linear_hidden_value(test_linear_bfloat16_value(outputs[index]),
                                     test_linear_bfloat16_value(second_outputs[index]), (double)ACTIVATION_LIMIT,
                                     1.0));
        const int32_t distance =
            test_linear_bfloat16_ordinal(third_outputs[index]) - test_linear_bfloat16_ordinal(expected);
        equal_count += third_outputs[index] == expected ? 1u : 0u;
        if (distance > (int32_t)HIDDEN_STEP_LAST_PLACES || -distance > (int32_t)HIDDEN_STEP_LAST_PLACES) {
            if (far_count == 0) {
                test_linear_fail("hidden value %" PRIu64 " is 0x%04X, the reference's 0x%04X, of gate 0x%04X and up "
                                 "0x%04X", index, (unsigned int)third_outputs[index], (unsigned int)expected,
                                 (unsigned int)outputs[index], (unsigned int)second_outputs[index]);
            }
            far_count++;
        }
    }
    totals->hidden_count += hidden_count;
    totals->hidden_equal_count += equal_count;
    (void)printf("  the hidden step: %" PRIu64 " values, %.4f%% bit-equal to the reference's of the card's gate and "
                 "up, %" PRIu64 " further than %u last place\n", hidden_count,
                 test_linear_percent(equal_count, hidden_count), far_count, HIDDEN_STEP_LAST_PLACES);
    /* the card's hidden values become the rows of the reference: their codes must be the card's hidden codes */
    for (uint32_t row = 0; row < ROWS; row++) {
        memcpy(row_values[row], third_outputs + (size_t)row * inner_size, inner_size * sizeof(uint16_t));
    }
    if (quantize_rows(inner_size, 0, ROWS) &&
        expect_quantized_rows(card, chain->ranges[CHAIN_HIDDEN_CODES], chain->ranges[CHAIN_HIDDEN_EXPONENTS],
                              chain->ranges[CHAIN_HIDDEN_OPERANDS],
                              inner_size, 0, ROWS, "the card's quantization of its own hidden values")) {
        (void)chain_product(card, shared_case, &matrices[MATRIX_DOWN], chain->ranges[CHAIN_OUTPUT],
                            "down, of the card's hidden codes", outputs, totals);
    }
}

/* Of two arrays [ROWS][width] of bfloat16 in received: how many values differ outside a column; *inside_out gets
 * how many differ at it. */
static uint64_t differing_outside_column(const struct test_linear_card *card,
					 struct linear_range first,
					 struct linear_range second,
					 uint32_t width, uint32_t column,
					 uint64_t *inside_out)
{
    uint64_t outside_count = 0;
    uint64_t inside_count = 0;
    if (test_linear_card_get(card, first, 0, second_outputs, (uint64_t)ROWS * width * VALUE_BYTES) &&
        test_linear_card_get(card, second, 0, third_outputs, (uint64_t)ROWS * width * VALUE_BYTES)) {
        for (size_t index = 0; index < (size_t)ROWS * width; index++) {
            const bool differs = second_outputs[index] != third_outputs[index];
            inside_count += differs && index % width == column ? 1u : 0u;
            outside_count += differs && index % width != column ? 1u : 0u;
        }
    }
    *inside_out = inside_count;
    return outside_count;
}

static void test_shared_expert(struct test_linear_card *card, const struct shared_case *shared_case,
                               uint32_t case_index, struct card_totals *totals)
{
    static struct shared_chain chains[ROWS + 1];    /* of 8 rows, of rows 1 to 7 alone, and after the change */
    static struct kernel_launch launches[(ROWS + 1) * CHAIN_LAUNCH_COUNT];
    struct linear_matrix matrices[MATRIX_COUNT];
    const uint32_t hidden_size = shared_case->hidden_size;
    const uint32_t inner_size = shared_case->inner_size;
    const uint64_t row_value_bytes = hidden_size * VALUE_BYTES;
    const uint64_t stream = SHARED_STREAM + case_index * STREAMS_PER_CASE;
    const uint32_t changed_column = (uint32_t)(test_linear_hash(CHANGE_STREAM, inner_size) % inner_size);
    uint64_t inside_count = 0;
    bool is_ready = hidden_size <= MAXIMUM_OUTPUT_SIZE && inner_size <= MAXIMUM_OUTPUT_SIZE;

    test_linear_context("card %" PRIu32 ", the shared expert %s", card->card_index, shared_case->name);
    (void)printf("shared expert, %s\n", shared_case->name);
    test_linear_card_clear(card);
    const struct linear_range values =
	    test_linear_card_work(card, ROWS * row_value_bytes);
    for (uint32_t matrix_index = 0; is_ready && matrix_index < MATRIX_COUNT; matrix_index++) {
        const bool is_down = matrix_index == MATRIX_DOWN;
        const struct test_linear_matrix_recipe recipe = {
            .format = E4M3,
            .shape = {is_down ? hidden_size : inner_size, is_down ? inner_size : hidden_size, E4M3_BITS_PER_WEIGHT, 0},
            .stream = stream + ROWS + matrix_index, .code = test_linear_e4m3_code,
            .scale_code_base = is_down ? DOWN_SCALE_CODE_BASE : shared_case->gate_scale_code_base,
            .scale_code_spread = SCALE_CODE_SPREAD};
	matrices[matrix_index] =
		(struct linear_matrix){ .shape = recipe.shape,
					.weight_format = E4M3,
					.slice_count =
						shared_case->slice_count };
	is_ready = test_linear_card_matrix(card, &recipe, &matrices[matrix_index].stored);
    }
    is_ready = is_ready && make_rows(hidden_size, stream, true);
    for (uint32_t row = 0; is_ready && row < ROWS; row++) {
        is_ready = test_linear_card_put(card, values, row * row_value_bytes, row_values[row], row_value_bytes);
    }
    is_ready = is_ready && chain_prepare(card, shared_case, matrices, values, ROWS, &chains[0]) &&
               chain_prepare(card, shared_case, matrices, values, ROWS, &chains[ROWS]);
    for (uint32_t row = 1; is_ready && row < ROWS; row++) {
        is_ready = chain_prepare(card, shared_case, matrices, part_of(values, row * row_value_bytes, row_value_bytes),
                                 1, &chains[row]);
    }
    if (!is_ready) {
        (void)test_linear_expect(is_ready, "the case is laid out");
        return;
    }
    for (uint32_t chain = 0; chain <= ROWS; chain++) {
        memcpy(&launches[(size_t)chain * CHAIN_LAUNCH_COUNT], chains[chain].launches, sizeof(chains[chain].launches));
    }
    if (!test_linear_card_load(card) || !test_linear_card_run(card, launches, ROWS * CHAIN_LAUNCH_COUNT) ||
        !test_linear_card_fetch(card)) {
        return;
    }
    chain_check(card, shared_case, matrices, &chains[0], stream, totals);
    for (uint32_t row = 1; row < ROWS; row++) {
        for (uint32_t range = 0; range < CHAIN_RANGE_COUNT; range++) {
            const uint64_t row_bytes = chain_row_bytes(shared_case, (enum chain_range)range);
            (void)test_linear_card_expect_same(card, chains[row].ranges[range], 0, chains[0].ranges[range],
                                               row * row_bytes, row_bytes,
                                               "a row alone keeps, in every array of the chain, the bits it has "
                                               "among 8");
        }
    }
    (void)test_linear_card_expect_untouched(card, values, "the rows' values");
    (void)test_linear_card_expect_weights_untouched(card);
    /* shown failing: one weight of gate changes on the card, where row 0 feeds its largest input */
    if (!make_rows(hidden_size, stream, true) ||
        !change_a_weight(card, &matrices[MATRIX_GATE], changed_column,
                         largest_fed_input(fed_values, fed_exponents, hidden_size)) ||
        !test_linear_card_run(card, &launches[(size_t)ROWS * CHAIN_LAUNCH_COUNT], CHAIN_LAUNCH_COUNT) ||
        !test_linear_card_fetch(card)) {
        return;
    }
    (void)printf("shared expert, %s, one weight of gate changed on the card (output %" PRIu32 ")\n",
                 shared_case->name, changed_column);
    chain_check(card, shared_case, matrices, &chains[ROWS], stream, totals);
    (void)test_linear_expect(differing_outside_column(card, chains[0].ranges[CHAIN_GATE],
                                                      chains[ROWS].ranges[CHAIN_GATE], inner_size, changed_column,
                                                      &inside_count) == 0 && inside_count != 0,
                             "a changed weight of gate changes gate's outputs of its own column, and no other");
    (void)printf("  gate: %" PRIu64 " of %u rows changed in the weight's column, nothing elsewhere\n", inside_count,
                 ROWS);
    (void)test_linear_expect(differing_outside_column(card, chains[0].ranges[CHAIN_HIDDEN],
                                                      chains[ROWS].ranges[CHAIN_HIDDEN], inner_size, changed_column,
                                                      &inside_count) == 0,
                             "a changed weight of gate changes hidden values of its own column only");
    (void)test_linear_card_expect_same(card, chains[0].ranges[CHAIN_UP], 0, chains[ROWS].ranges[CHAIN_UP], 0,
                                       ROWS * chain_row_bytes(shared_case, CHAIN_UP),
                                       "a changed weight of gate changes nothing of up");
    (void)test_linear_card_expect_weights_untouched(card);
}

/* ---- every card ---- */

/* Everything the test does with one card; its digest of what it computed, or 0 if it did not open. */
static uint64_t test_card(uint32_t card_index, bool draft_native)
{
    static struct test_linear_card card;
    struct card_totals totals;
    memset(&totals, 0, sizeof(totals));
    if (!test_linear_card_open(&card, card_index, WORK_CAPACITY, WEIGHTS_CAPACITY)) {
        test_linear_card_close(&card);
        return 0;
    }
    if (draft_native) {
	    card.arithmetic_class = card.arithmetic_class ==
						    ARITHMETIC_CLASS_SM_80 ?
					    ARITHMETIC_CLASS_SM_80 :
					    ARITHMETIC_CLASS_SM_120_NATIVE;
	    struct trunk_case trunk_case = { "the draft's native main_proj",
					     MODEL_HIDDEN, MODEL_DRAFT_FEATURES,
					     0, 0 };
	    const struct matrix_shape_record shape = { trunk_case.output_size,
						       trunk_case.input_size,
						       E4M3_BITS_PER_WEIGHT,
						       0 };
	    const bool prepared =
		    test_linear_expect_status(
			    linear_kernels_prepare(
				    &backend_cuda_value, card.card,
				    card.arithmetic_class, &card.kernels),
			    STATUS_OK,
			    "the native draft's linear kernels are prepared") &&
		    test_linear_expect_status(
			    linear_default_slice_count(
				    LINEAR_PROFILE_PRECISION,
				    LINEAR_ORDER_FIXED,
				    LINEAR_FAMILY_TRUNK_QUANTIZED, &shape,
				    card.arithmetic_class,
				    &trunk_case.slice_count),
			    STATUS_OK,
			    "the main projection's native slice count is selected");
	    if (prepared) {
		    (void)printf("native draft main_proj: class %" PRIu32
				 ", %" PRIu32 " slices\n",
				 card.arithmetic_class, trunk_case.slice_count);
		    test_trunk_case(&card, &trunk_case, 0, &totals);
		    (void)test_linear_expect(
			    totals.quantized.output_count != 0,
			    "the native projection checked outputs");
	    }
    } else {
        for (uint32_t width_index = 0; width_index < COUNT_OF(quantizer_widths); width_index++) {
            test_quantizer(&card, quantizer_widths[width_index]);
        }
        for (uint32_t case_index = 0; case_index < COUNT_OF(trunk_cases); case_index++) {
            struct trunk_case trunk_case = trunk_cases[case_index];     /* in the slices the card runs */
            trunk_case.slice_count = test_linear_card_slices(&card, trunk_case.slice_count, false);
            test_trunk_case(&card, &trunk_case, case_index, &totals);
        }
        for (uint32_t case_index = 0; case_index < COUNT_OF(shared_cases); case_index++) {
            test_shared_expert(&card, &shared_cases[case_index], case_index, &totals);
        }
    }
    test_linear_context("card %" PRIu32, card_index);
    (void)printf("card %" PRIu32 ": %" PRIu64 " outputs of products of quantized rows, %.4f%% bit-equal; %" PRIu64
                 " of rows that are not quantized, %.4f%%; %" PRIu64 " hidden values, %.4f%%\n", card_index,
                 totals.quantized.output_count,
                 test_linear_percent(totals.quantized.equal_count, totals.quantized.output_count),
                 totals.unquantized.output_count,
                 test_linear_percent(totals.unquantized.equal_count, totals.unquantized.output_count),
                 totals.hidden_count, test_linear_percent(totals.hidden_equal_count, totals.hidden_count));
    (void)test_linear_expect(
        test_linear_share_holds(totals.quantized.equal_count, totals.quantized.output_count) &&
            test_linear_share_holds(totals.unquantized.equal_count, totals.unquantized.output_count),
        "99.9% of the outputs of each product kernel are bit-equal to the double result rounded once");
    (void)test_linear_expect(test_linear_share_holds(totals.hidden_equal_count, totals.hidden_count),
                             "99.9% of the hidden values are bit-equal to the reference's hidden step");
    (void)printf("card %" PRIu32 " (class %" PRIu32 "): digest of everything it computed 0x%016" PRIX64 "\n",
                 card_index, card.arithmetic_class, card.digest);
    const uint64_t digest = card.digest;
    test_linear_card_close(&card);
    return digest;
}

int main(int argument_count, char **arguments)
{
    const bool draft_native = argument_count == DRAFT_NATIVE_ARGUMENTS && strcmp(arguments[1], "--draft_native") == 0;
    if (argument_count != 1 && !draft_native) {
        (void)fprintf(stderr, "usage: linear_trunk_card_test [--draft_native]\n");
        return EXIT_FAILURE;
    }
    uint64_t digests[BACKEND_MAXIMUM_CARDS] = { 0 };
    uint32_t card_count = 0;
    uint32_t agreeing_count = 0;

    (void)alarm(TEST_TIME_LIMIT_SECONDS);
    test_lane_order();
    (void)test_linear_expect_status(backend_cuda_value.card_count(&card_count),
				    STATUS_OK, "the cards are counted");
    if (!test_linear_expect(card_count > 0 &&
				    card_count <= BACKEND_MAXIMUM_CARDS,
			    "the machine has cards for the test")) {
	    return EXIT_FAILURE;
    }
    for (uint32_t card_index = 0; card_index < card_count; card_index++) {
        digests[card_index] = test_card(card_index, draft_native);
        agreeing_count += digests[card_index] == digests[0] ? 1u : 0u;
    }
    (void)printf("%" PRIu32 " of %" PRIu32 " cards computed the bits of card 0 in every case\n", agreeing_count,
                 card_count);
    if (test_linear_failures() != 0) {
        (void)fprintf(stderr, "linear_trunk_card_test FAILED: %d failures\n", test_linear_failures());
        return EXIT_FAILURE;
    }
    (void)printf("linear_trunk_card_test passed on %" PRIu32 " cards\n", card_count);
    return EXIT_SUCCESS;
}
