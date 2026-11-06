/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * linear_exact_card_test: on the real cards, the product kernels of the linear (trunk_linear_quantized_value,
 * trunk_linear_bfloat16_value, expert_down_value as the product of one matrix of 4-bit weights, and the gate and
 * the up of expert_gate_up_value, each shown by the hidden values) on numbers made so that the host has the card's
 * bits, and at and beyond the ends of the scale codes a kernel supports.
 *
 * It needs the cards free (`make card_tests`): a card without room for the arena FAILS the test, it is never skipped.
 * On every card, for every case of exact_cases and for EVERY number of slices a kernel is stamped for (1, 2, 4, 8 and
 * 16) up to the most the matrix allows:
 *   - The cases held exactly. Every weight code and every fed value is a small whole multiple of one power of two,
 *     and the magnitudes of all products of an output add up to less than 2^24 of that unit, which the test proves
 *     on the host for every output before it asks a card. Every sum a card can form, in any order and inside the
 *     tensor-core instructions too, is then a float32 exactly, so EVERY output must be bit-equal to the double result
 *     rounded once, whatever the slices: layouts, scale exponents, slice cuts and the addressing of every byte are
 *     held with no tolerance. Cases feed E4M3 subnormal codes only, and rows that are not quantized of whole
 *     multiples of 2^-24 only: both reach the instruction as subnormal 16-bit floats, as the code of 0.5 among the
 *     4-bit weights does: a card that flushed any of them to zero would give other bits. One case of each of the
 *     three kernels that store a product has the model's width of 5120.
 *   - The cases held in order. Every block of 32 inputs is exact as above, but the blocks are of sizes far apart and
 *     two of every three cancel each other exactly (the same codes fed to weights of the other sign), so that a
 *     float32 chain loses bits of the third, and which bits depends on the order and on where the slices are cut.
 *     The host adds the blocks in float32 exactly as linear_records.h states it (a chain per slice from +0 in
 *     ascending blocks, the slices added from +0 in ascending order, one rounding to bfloat16): EVERY output must be
 *     those bits, for every number of slices. Some of them must differ from the double result rounded once, or the
 *     case would hold nothing. Rows that are not quantized have no exponents: there the values of the two blocks
 *     that cancel are themselves 2^20 above those of the third, every one a 16-bit float exactly.
 *   - The ends of the scale codes (linear_records.h): the smallest and the largest sum of a weight scale code and an
 *     activation exponent for which the power of two is the true one, for both weight formats, and the scale codes
 *     0 and TRUNK_BFLOAT16_SCALE_LARGEST for rows that are not quantized. These outputs are held to the
 *     tolerance of linear_reference.h; none may be zero or infinite where the reference's is neither.
 *   - Beyond the ends (linear_records.h says what a kernel does there; the scale codes of such a case have no
 *     spread). Two below the smallest sum the exponent field of the power of two would be -1: in the codes form
 *     every block adds nothing, and EVERY output must be +0; the carried form has no lower end, and its outputs are
 *     held to the tolerance. One above the largest sum, and one above the largest scale code of rows that are not
 *     quantized, the power would be 2^128: a card multiplies by 2^127, so its outputs are held to the tolerance of
 *     HALF the reference's products, and none may be infinite.
 *   - The ends of the carried form, in which a card of class 80 makes its quantized products: blocks at the largest
 *     exponent of its operands, whose outputs are exact, and above it, where EVERY output must be no finite number;
 *     the largest weight scale code and one above it (a card multiplies by 2^127: half the true power). Each case
 *     holds a card of the other class to what its own form gives there. On a card of class 120 the experts take the
 *     block-scaled form, which has no end of its own: at and beyond the ends of the codes form its outputs are held
 *     bit for bit to the float32 sum, in the stated order, of the terms the host makes (the exact sum of a block's
 *     products times both true scales, cut toward zero to a float32, subnormal ones among them), and none may be
 *     zero or infinite where the reference's is neither. A case whose terms are beyond the largest float32, all of
 *     one sign, must leave every output infinite on both classes (on class 80 its vectors are above the largest
 *     exponent of the carried form).
 * In the exact build (no LINEAR_CLASS_FORMS, the default) every class takes class 80's forms and the slices
 * keep the reference order: a case held in order is one chain from +0 over every block, whatever the slices, and a
 * card of class 120 runs at most the slices of its emulation (test_linear_card_slices).
 * The product of 4-bit weights is the third launch of linear_routed_experts_prepare, run alone: one job
 * whose down matrix is the matrix and whose eight columns carry the eight vectors in a shuffled order, behind an
 * empty job filled with rubbish; the hidden codes it reads are laid into the arena by the test.
 * The gate and the up are the first launch of that chain, run alone, with no activation limit and route weights of
 * 1. The kernel stores hidden values only, so one of its two matrices is the matrix of the case and the other a
 * constant: it holds one weight of 1 for each output, at input 0, where every vector holds 2^-5, and its scale
 * makes of them 1 (as the up) or 32 (as the gate). A gate of 32 or more is its own hidden step in float32
 * (exp(-32) is below half a last place of 1, so 1 + exp(-gate) is 1), and a product by 1 or by 32 is exact: the
 * hidden values are then the bits of the gate, or of 32 times the up, and are held as any product is. The gate of
 * such a case is positive where its blocks do not cancel, and the host proves for every output and number of
 * slices that the gate a card must find is at least 32 before it asks a card.
 * One line per case says what was measured.
 */
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "anemone/backend_seam.h"
#include "anemone/linear.h"
#include "anemone/linear_records.h"
#include "anemone/status.h"
#include "anemone/weight_format_seam.h"
#include "linear_card_fixture.h"
#include "linear_reference.h"
#include "weight_format_reference.h"

#define TEST_TIME_LIMIT_SECONDS 900             /* the alarm ends the test here, whatever the driver does */
#define WORK_CAPACITY (8ull << 20u)
#define WEIGHTS_CAPACITY (4ull << 20u)
#define COUNT_OF(table) ((uint32_t)(sizeof(table) / sizeof((table)[0])))
#define VECTORS \
	LINEAR_MAXIMUM_ROWS /* rows of a trunk launch, pairs of the one job */
#define VALUES_PER_BLOCK TEST_LINEAR_VALUES_PER_BLOCK
#define VALUE_BYTES ((uint64_t)LINEAR_BFLOAT16_BYTES)
#define MAXIMUM_FED_WIDTH 5120u
#define MAXIMUM_OUTPUT_SIZE 64u
#define E2M1 WEIGHT_FORMAT_E2M1_SCALED
#define E4M3 WEIGHT_FORMAT_E4M3_SCALED
#define E2M1_BITS_PER_WEIGHT 4u
#define E4M3_BITS_PER_WEIGHT 8u
#define E8M0_BIAS 127u                          /* an E8M0 byte c stands for 2^(c - 127) */
#define E8M0_LARGEST_CODE 254u
#define EXACT_FLOAT32_WHOLE_NUMBERS 16777216.0  /* 2^24: a float32 holds every whole number below it */
#define EXACT_STREAM 0x0E000000ull              /* the streams of the hashes, far from those of the other tests */
#define STREAMS_PER_CASE 4096ull
#define EXPONENT_STREAM 0xE0ull
#define OUTPUT_INDEX_SHIFT 32u
#define HASH_SIGN_SHIFT 7u
#define HASH_CHOICE_SHIFT 8u
#define HASH_MANTISSA_SHIFT 16u
#define E4M3_SIGN_SHIFT 7u
#define E4M3_SIGN_BIT 0x80u
#define E2M1_SIGN_BIT 8u
#define E4M3_BYTE_MASK 0xFFu
#define E4M3_NOT_A_NUMBER_MAGNITUDE 0x7Fu
#define E4M3_MANTISSA_MASK 7u
#define E4M3_LARGEST_CODE 0x7Eu                 /* 448 */
#define E2M1_LARGEST_CODE 7u                    /* 6 */
#define E2M1_MAGNITUDE_MASK 7u
#define E4M3_MAGNITUDE_MASK 0x7Fu
#define E2M1_CODE_OF_ONE 2u                     /* the one weight of each output of a constant matrix */
#define CONSTANT_FED_CODE 0x10u                 /* the E4M3 code of 2^-5, a low code: what that weight multiplies */
#define CONSTANT_FED_EXPONENT (-5)
#define GATE_CONSTANT_EXPONENT 5                /* a constant gate is 32: its own hidden step in float32 */
#define GATE_LEAST 32.0                         /* from here on a gate is its own hidden step: see the top */
#define NO_ACTIVATION_LIMIT INFINITY            /* linear.h: what a model without a limit gives */
#define SCALE_SUM_STEPS_BELOW 2u                /* below the smallest sum: the exponent field of the power is then -1 */
#define CARRIED_CASE_EXPONENT 120u              /* of the vectors of the cases at the carried form's scale codes */
#define ONE_BINADE_FIRST_CODE 0x40u             /* the E4M3 codes 0x40 to 0x47: 8 to 15 quarters */
#define ONE_BINADE_UNIT_EXPONENT (-2)
#define ONE_BINADE_ZERO_ONE_IN 8u               /* one weight in eight is a zero, of either sign */
#define SMALL_CODE_COUNT 40u                    /* the E4M3 codes 0x00 to 0x27: magnitudes below 1/4 */
#define E2M1_UNIT_EXPONENT (-1)                 /* every E2M1 number is a whole number of halves */
#define LOW_CODE_COUNT 32u                      /* the E4M3 codes 0x00 to 0x1F: 0 to 60 units of 2^-9 */
#define SUBNORMAL_CODE_COUNT 7u                 /* the E4M3 codes 0x01 to 0x07 */
#define HIGH_FIRST_CODE 0x70u                   /* the E4M3 codes 0x70 to 0x7E: 128 to 448 */
#define HIGH_CODE_COUNT 15u
#define CODE_UNIT_EXPONENT (-9)                 /* the low and the subnormal codes are whole numbers of 2^-9 */
#define EIGHTHS_SPREAD 121u                     /* -60 to 60 eighths */
#define EIGHTHS_LARGEST 60
#define EIGHTHS_UNIT_EXPONENT (-3)
#define SUBNORMAL_HALF_UNIT_EXPONENT (-24)       /* the last place of a subnormal 16-bit float */
#define SMALL_VALUES_UNIT_EXPONENT (-6)         /* -60 to 60 sixty-fourths: magnitudes below 1 */
#define CANCELLING_BLOCKS 3u                    /* of every three blocks two cancel each other */
#define CANCELLING_EXPONENT_STEP 20u            /* and are 2^20 above the third */
#define CANCELLING_VALUES_UNIT_EXPONENT (-16)   /* rows not quantized: -60 to 60 times 2^-16, or times 2^4 */
#define MAXIMUM_ORDERED_BLOCKS 32u              /* the blocks of 32 inputs of a case held in order */
#define BLOCKS_PER_LOAD (LINEAR_VALUES_PER_LOAD / VALUES_PER_BLOCK)
#define ACTIVATION_LIMIT 10.0f                  /* of the chain the product of 4-bit weights is a launch of */
#define DUMMY_GATE_INPUT_SIZE \
	LINEAR_VALUES_PER_LOAD /* of the gate and up a case of down does not run */
#define DUMMY_DOWN_OUTPUT_SIZE \
	LINEAR_VALUES_PER_LOAD /* of the down a case of gate and up does not run */
#define JOB_COUNT 2u                            /* an empty job, then the job of the matrix */
#define RUBBISH_BYTE 0xA7
#define COLUMN_STRIDE 3u                        /* column c carries vector (3 c + 5) mod 8: every vector once */
#define COLUMN_FIRST 5u
#if defined(LINEAR_CLASS_FORMS)
#define IS_EXACT_PROFILE false                  /* each class its own forms, and a chain per slice */
#else
#define IS_EXACT_PROFILE true                   /* every class class 80's forms, and one chain at any slices */
#endif

/* KERNEL_EXPERT_GATE and KERNEL_EXPERT_UP are expert_gate_up_value with the matrix of the case as its gate, or as
 * its up, beside a constant: see the top. */
enum exact_kernel {
    KERNEL_TRUNK_QUANTIZED,
    KERNEL_TRUNK_BFLOAT16,
    KERNEL_EXPERT_DOWN,
    KERNEL_EXPERT_GATE,
    KERNEL_EXPERT_UP
};

/* The codes of a matrix: of one binade (8-bit weights: 8 to 15 quarters, and zeros), any, small (8-bit weights
 * below 1/4), the largest positive number of the format, or cancelling: of every three blocks of 32 inputs the
 * second holds the codes of the first with the other sign. */
enum weights_kind { WEIGHTS_ONE_BINADE, WEIGHTS_ANY, WEIGHTS_SMALL, WEIGHTS_LARGEST, WEIGHTS_CANCELLING };

/* What the vectors hold: codes (low: 0 to 60 units of 2^-9; subnormal only; any; high: 128 to 448, positive;
 * cancelling: low codes, of every three blocks the second a copy of the first and both 2^20 above the third), or
 * for rows that are not quantized values (whole eighths; whole multiples of 2^-24, which are subnormal 16-bit
 * floats; plain; whole sixty-fourths; cancelling: of every three blocks the second a copy of the first and the
 * values of both 2^20 above those of the third). */
enum fed_kind {
    FED_LOW_CODES,
    FED_SUBNORMAL_CODES,
    FED_ANY_CODES,
    FED_HIGH_CODES,
    FED_CANCELLING_CODES,
    FED_EIGHTHS,
    FED_SUBNORMAL_HALVES,
    FED_PLAIN_VALUES,
    FED_SMALL_VALUES,
    FED_CANCELLING_VALUES
};

/* What a card's outputs are held to: the double result rounded once, bit for bit; the float32 sum in the order of
 * linear_records.h, bit for bit; the tolerance of linear_reference.h; +0, bit for bit, below the scale codes; no
 * finite number. */
enum held_as { HELD_EXACTLY, HELD_IN_ORDER, HELD_TO_TOLERANCE, HELD_AS_NOTHING, HELD_NOT_FINITE };
enum product_form { FORM_CODES, FORM_CARRIED, FORM_BLOCK_SCALED, FORM_BFLOAT16_ROWS };

/* One case: a kernel, a matrix of output_size by input_size whose scale codes are scale_code_base plus up to
 * scale_code_spread, and eight vectors whose blocks' exponents are exponent_base plus up to exponent_spread. */
struct exact_case {
    const char *name;
    enum exact_kernel kernel;
    uint32_t output_size;
    uint32_t input_size;
    uint32_t segment_count;
    enum weights_kind weights;
    uint32_t scale_code_base;
    uint32_t scale_code_spread;
    enum fed_kind fed;
    uint32_t exponent_base;
    uint32_t exponent_spread;
    enum held_as held;
};

static const struct exact_case exact_cases[] = {
	{ "8-bit weights by low codes", KERNEL_TRUNK_QUANTIZED, 48, 256, 1,
	  WEIGHTS_ONE_BINADE, 125, 3, FED_LOW_CODES, 120, 3, HELD_EXACTLY },
	{ "8-bit weights, blocks that cancel", KERNEL_TRUNK_QUANTIZED, 32, 768,
	  1, WEIGHTS_CANCELLING, 125, 0, FED_CANCELLING_CODES, 105, 0,
	  HELD_IN_ORDER },
	{ "8-bit weights by subnormal codes only", KERNEL_TRUNK_QUANTIZED, 32,
	  128, 1, WEIGHTS_ONE_BINADE, 125, 0, FED_SUBNORMAL_CODES, 120, 0,
	  HELD_EXACTLY },
	{ "8-bit weights by low codes, 5120 wide", KERNEL_TRUNK_QUANTIZED, 64,
	  5120, 1, WEIGHTS_ONE_BINADE, 125, 1, FED_LOW_CODES, 120, 0,
	  HELD_EXACTLY },
	{ "8-bit weights, the smallest sum of scale codes",
	  KERNEL_TRUNK_QUANTIZED, 32, 256, 1, WEIGHTS_ANY,
	  TRUNK_SCALE_SUM_SMALLEST - ACTIVATION_EXPONENT_SMALLEST_VALUE, 0,
	  FED_ANY_CODES, ACTIVATION_EXPONENT_SMALLEST_VALUE, 0,
	  HELD_TO_TOLERANCE },
	{ "8-bit weights, the largest sum of scale codes",
	  KERNEL_TRUNK_QUANTIZED, 32, 256, 1, WEIGHTS_SMALL, E8M0_LARGEST_CODE,
	  0, FED_ANY_CODES, TRUNK_SCALE_SUM_LARGEST - E8M0_LARGEST_CODE, 0,
	  HELD_TO_TOLERANCE },
	{ "8-bit weights by eighths, rows not quantized", KERNEL_TRUNK_BFLOAT16,
	  32, 256, 1, WEIGHTS_ONE_BINADE, 125, 3, FED_EIGHTHS, 0, 0,
	  HELD_EXACTLY },
	{ "8-bit weights by eighths in 2 segments, rows not quantized",
	  KERNEL_TRUNK_BFLOAT16, 64, 128, 2, WEIGHTS_ONE_BINADE, 125, 0,
	  FED_EIGHTHS, 0, 0, HELD_EXACTLY },
	{ "8-bit weights by subnormal 16-bit floats, rows not quantized",
	  KERNEL_TRUNK_BFLOAT16, 32, 256, 1, WEIGHTS_ONE_BINADE, 125, 3,
	  FED_SUBNORMAL_HALVES, 0, 0, HELD_EXACTLY },
	{ "8-bit weights by eighths, 5120 wide, rows not quantized",
	  KERNEL_TRUNK_BFLOAT16, 64, 5120, 1, WEIGHTS_ONE_BINADE, 125, 1,
	  FED_EIGHTHS, 0, 0, HELD_EXACTLY },
	{ "8-bit weights, the smallest scale code, rows not quantized",
	  KERNEL_TRUNK_BFLOAT16, 32, 256, 1, WEIGHTS_ANY, 0, 0,
	  FED_PLAIN_VALUES, 0, 0, HELD_TO_TOLERANCE },
	{ "8-bit weights, the largest scale code, rows not quantized",
	  KERNEL_TRUNK_BFLOAT16, 32, 256, 1, WEIGHTS_SMALL,
	  TRUNK_BFLOAT16_SCALE_LARGEST, 0, FED_SMALL_VALUES, 0, 0,
	  HELD_TO_TOLERANCE },
	{ "4-bit weights by low codes", KERNEL_EXPERT_DOWN, 64, 256, 1,
	  WEIGHTS_ANY, 125, 3, FED_LOW_CODES, 120, 3, HELD_EXACTLY },
	{ "4-bit weights, blocks that cancel", KERNEL_EXPERT_DOWN, 64, 768, 1,
	  WEIGHTS_CANCELLING, 125, 0, FED_CANCELLING_CODES, 105, 0,
	  HELD_IN_ORDER },
	{ "4-bit weights by subnormal codes only", KERNEL_EXPERT_DOWN, 64, 128,
	  1, WEIGHTS_ANY, 125, 0, FED_SUBNORMAL_CODES, 120, 0, HELD_EXACTLY },
	{ "4-bit weights by low codes, 5120 wide", KERNEL_EXPERT_DOWN, 64, 5120,
	  1, WEIGHTS_ANY, 125, 1, FED_LOW_CODES, 120, 0, HELD_EXACTLY },
	{ "4-bit weights, the smallest sum of scale codes", KERNEL_EXPERT_DOWN,
	  64, 1024, 1, WEIGHTS_LARGEST,
	  EXPERT_SCALE_SUM_SMALLEST - ACTIVATION_EXPONENT_SMALLEST_VALUE, 0,
	  FED_HIGH_CODES, ACTIVATION_EXPONENT_SMALLEST_VALUE, 0,
	  HELD_TO_TOLERANCE },
	{ "4-bit weights, the largest sum of scale codes", KERNEL_EXPERT_DOWN,
	  64, 128, 1, WEIGHTS_ANY,
	  EXPERT_SCALE_SUM_LARGEST - ACTIVATION_EXPONENT_SMALLEST_VALUE, 0,
	  FED_ANY_CODES, ACTIVATION_EXPONENT_SMALLEST_VALUE, 0,
	  HELD_TO_TOLERANCE },
	{ "8-bit weights, blocks that cancel, rows not quantized",
	  KERNEL_TRUNK_BFLOAT16, 32, 768, 1, WEIGHTS_CANCELLING, 125, 0,
	  FED_CANCELLING_VALUES, 0, 0, HELD_IN_ORDER },
	{ "8-bit weights, below the smallest sum of scale codes",
	  KERNEL_TRUNK_QUANTIZED, 32, 256, 1, WEIGHTS_ANY,
	  TRUNK_SCALE_SUM_SMALLEST - ACTIVATION_EXPONENT_SMALLEST_VALUE -
		  SCALE_SUM_STEPS_BELOW,
	  0, FED_ANY_CODES, ACTIVATION_EXPONENT_SMALLEST_VALUE, 0,
	  HELD_AS_NOTHING },
	{ "8-bit weights, above the largest sum of scale codes",
	  KERNEL_TRUNK_QUANTIZED, 32, 256, 1, WEIGHTS_SMALL, E8M0_LARGEST_CODE,
	  0, FED_ANY_CODES, TRUNK_SCALE_SUM_LARGEST - E8M0_LARGEST_CODE + 1, 0,
	  HELD_TO_TOLERANCE },
	{ "8-bit weights, above the largest scale code, rows not quantized",
	  KERNEL_TRUNK_BFLOAT16, 32, 256, 1, WEIGHTS_SMALL,
	  TRUNK_BFLOAT16_SCALE_LARGEST + 1, 0, FED_SMALL_VALUES, 0, 0,
	  HELD_TO_TOLERANCE },
	{ "4-bit weights, below the smallest sum of scale codes",
	  KERNEL_EXPERT_DOWN, 64, 1024, 1, WEIGHTS_LARGEST, 0, 0,
	  FED_HIGH_CODES, EXPERT_SCALE_SUM_SMALLEST - SCALE_SUM_STEPS_BELOW, 0,
	  HELD_AS_NOTHING },
	{ "4-bit weights, above the largest sum of scale codes",
	  KERNEL_EXPERT_DOWN, 64, 128, 1, WEIGHTS_ANY, E8M0_LARGEST_CODE, 0,
	  FED_ANY_CODES, EXPERT_SCALE_SUM_LARGEST - E8M0_LARGEST_CODE + 1, 0,
	  HELD_TO_TOLERANCE },
	{ "8-bit weights by low codes in blocks of the largest carried exponent",
	  KERNEL_TRUNK_QUANTIZED, 32, 256, 1, WEIGHTS_ONE_BINADE, 125, 0,
	  FED_LOW_CODES, TEST_LINEAR_CARRIED_EXPONENT_LARGEST, 0,
	  HELD_EXACTLY },
	{ "8-bit weights by low codes in blocks above the largest carried exponent",
	  KERNEL_TRUNK_QUANTIZED, 32, 256, 1, WEIGHTS_ONE_BINADE, 125, 0,
	  FED_LOW_CODES, TEST_LINEAR_CARRIED_EXPONENT_LARGEST + 1, 0,
	  HELD_EXACTLY },
	{ "8-bit weights, the largest carried scale code",
	  KERNEL_TRUNK_QUANTIZED, 32, 256, 1, WEIGHTS_SMALL,
	  TRUNK_CARRIED_SCALE_LARGEST, 0, FED_ANY_CODES, CARRIED_CASE_EXPONENT,
	  0, HELD_TO_TOLERANCE },
	{ "8-bit weights, above the largest carried scale code",
	  KERNEL_TRUNK_QUANTIZED, 32, 256, 1, WEIGHTS_SMALL,
	  TRUNK_CARRIED_SCALE_LARGEST + 1, 0, FED_ANY_CODES,
	  CARRIED_CASE_EXPONENT, 0, HELD_TO_TOLERANCE },
	{ "4-bit weights by low codes in blocks of the largest carried exponent",
	  KERNEL_EXPERT_DOWN, 64, 256, 1, WEIGHTS_ANY, 125, 0, FED_LOW_CODES,
	  TEST_LINEAR_CARRIED_EXPONENT_LARGEST, 0, HELD_EXACTLY },
	{ "4-bit weights by low codes in blocks above the largest carried exponent",
	  KERNEL_EXPERT_DOWN, 64, 256, 1, WEIGHTS_ANY, 125, 0, FED_LOW_CODES,
	  TEST_LINEAR_CARRIED_EXPONENT_LARGEST + 1, 0, HELD_EXACTLY },
	{ "4-bit weights, the largest carried scale code", KERNEL_EXPERT_DOWN,
	  64, 128, 1, WEIGHTS_ANY, EXPERT_CARRIED_SCALE_LARGEST, 0,
	  FED_ANY_CODES, CARRIED_CASE_EXPONENT, 0, HELD_TO_TOLERANCE },
	{ "4-bit weights, above the largest carried scale code",
	  KERNEL_EXPERT_DOWN, 64, 128, 1, WEIGHTS_ANY,
	  EXPERT_CARRIED_SCALE_LARGEST + 1, 0, FED_ANY_CODES,
	  CARRIED_CASE_EXPONENT, 0, HELD_TO_TOLERANCE },
	{ "4-bit weights, terms beyond the largest float32", KERNEL_EXPERT_DOWN,
	  64, 128, 1, WEIGHTS_LARGEST, E8M0_LARGEST_CODE, 0, FED_HIGH_CODES,
	  ACTIVATION_EXPONENT_LARGEST, 0, HELD_NOT_FINITE },
	{ "4-bit weights by low codes, the up beside a gate of 32",
	  KERNEL_EXPERT_UP, 64, 256, 1, WEIGHTS_ANY, 125, 3, FED_LOW_CODES, 120,
	  0, HELD_EXACTLY },
	{ "4-bit weights, blocks that cancel, the gate beside an up of 1",
	  KERNEL_EXPERT_GATE, 64, 768, 1, WEIGHTS_CANCELLING, 151, 0,
	  FED_CANCELLING_CODES, 105, 0, HELD_IN_ORDER },
	{ "4-bit weights, blocks that cancel, the up beside a gate of 32",
	  KERNEL_EXPERT_UP, 64, 768, 1, WEIGHTS_CANCELLING, 125, 0,
	  FED_CANCELLING_CODES, 105, 0, HELD_IN_ORDER },
};

/* The vectors of the case at hand, and what the reference makes of them. */
static uint16_t vector_values[MAXIMUM_FED_WIDTH];
static uint8_t vector_codes[MAXIMUM_FED_WIDTH];                 /* in the order of the values */
static double fed_values[VECTORS * MAXIMUM_FED_WIDTH];
static uint8_t fed_exponents[VECTORS * (MAXIMUM_FED_WIDTH / VALUES_PER_BLOCK)];
static struct test_linear_product products[VECTORS * MAXIMUM_OUTPUT_SIZE];
static uint16_t outputs[VECTORS * MAXIMUM_OUTPUT_SIZE];
static double block_sums[VECTORS * MAXIMUM_ORDERED_BLOCKS];    /* of one output: [vector][block] */
static float block_terms[VECTORS * MAXIMUM_OUTPUT_SIZE * MAXIMUM_ORDERED_BLOCKS];   /* [vector][output][block] */

static const char *const held_names[] = {
    [HELD_EXACTLY] = "the double result rounded once", [HELD_IN_ORDER] = "the float32 sum in the stated order",
    [HELD_TO_TOLERANCE] = "the tolerance", [HELD_AS_NOTHING] = "+0, as blocks that add nothing leave it",
    [HELD_NOT_FINITE] = "no finite number"};

/* Whether the matrix of a kernel is one of 4-bit weights, read by a launch of the routed experts' chain. */
static bool is_of_experts(enum exact_kernel kernel)
{
    return kernel == KERNEL_EXPERT_DOWN || kernel == KERNEL_EXPERT_GATE || kernel == KERNEL_EXPERT_UP;
}

static bool is_gate_or_up(enum exact_kernel kernel)
{
    return kernel == KERNEL_EXPERT_GATE || kernel == KERNEL_EXPERT_UP;
}

static uint8_t one_binade_code(uint64_t stream, uint32_t output_index, uint32_t input_index)
{
    const uint64_t hash = test_linear_hash(stream, ((uint64_t)output_index << OUTPUT_INDEX_SHIFT) | input_index);
    const uint32_t sign = ((uint32_t)(hash >> HASH_SIGN_SHIFT) & 1u) << E4M3_SIGN_SHIFT;
    const uint32_t mantissa = (uint32_t)(hash >> HASH_MANTISSA_SHIFT) & E4M3_MANTISSA_MASK;
    const bool is_zero = (hash >> HASH_CHOICE_SHIFT) % ONE_BINADE_ZERO_ONE_IN == 0;
    return (uint8_t)(is_zero ? sign : sign | ONE_BINADE_FIRST_CODE | mantissa);
}

static uint8_t small_code(uint64_t stream, uint32_t output_index, uint32_t input_index)
{
    const uint64_t hash = test_linear_hash(stream, ((uint64_t)output_index << OUTPUT_INDEX_SHIFT) | input_index);
    const uint32_t sign = ((uint32_t)(hash >> HASH_SIGN_SHIFT) & 1u) << E4M3_SIGN_SHIFT;
    return (uint8_t)(sign | (uint32_t)((hash >> HASH_CHOICE_SHIFT) % SMALL_CODE_COUNT));
}

static uint8_t largest_e4m3_code(uint64_t stream, uint32_t output_index, uint32_t input_index)
{
    (void)stream;
    (void)output_index;
    (void)input_index;
    return E4M3_LARGEST_CODE;
}

static uint8_t largest_e2m1_code(uint64_t stream, uint32_t output_index, uint32_t input_index)
{
    (void)stream;
    (void)output_index;
    (void)input_index;
    return E2M1_LARGEST_CODE;
}

/* Of every three blocks of 32 inputs the second holds the codes of the first with the other sign. */
static uint8_t cancelling_e4m3_code(uint64_t stream, uint32_t output_index, uint32_t input_index)
{
    const bool is_second = input_index / VALUES_PER_BLOCK % CANCELLING_BLOCKS == 1;
    const uint32_t code =
        one_binade_code(stream, output_index, is_second ? input_index - VALUES_PER_BLOCK : input_index);
    return (uint8_t)(is_second ? code ^ E4M3_SIGN_BIT : code);
}

static uint8_t cancelling_e2m1_code(uint64_t stream, uint32_t output_index, uint32_t input_index)
{
    const bool is_second = input_index / VALUES_PER_BLOCK % CANCELLING_BLOCKS == 1;
    const uint32_t code =
        test_linear_e2m1_code(stream, output_index, is_second ? input_index - VALUES_PER_BLOCK : input_index);
    return (uint8_t)(is_second ? code ^ E2M1_SIGN_BIT : code);
}

/* The cancelling codes with no sign in the third block of every three: for a gate, which those blocks then raise. */
static uint8_t raising_e2m1_code(uint64_t stream, uint32_t output_index, uint32_t input_index)
{
    const bool is_third = input_index / VALUES_PER_BLOCK % CANCELLING_BLOCKS == CANCELLING_BLOCKS - 1;
    const uint32_t code = cancelling_e2m1_code(stream, output_index, input_index);
    return (uint8_t)(is_third ? code & E2M1_MAGNITUDE_MASK : code);
}

/* The constant matrix beside a gate or an up: one weight of 1 for each output, at input 0. */
static uint8_t constant_e2m1_code(uint64_t stream, uint32_t output_index, uint32_t input_index)
{
    (void)stream;
    (void)output_index;
    return (uint8_t)(input_index == 0 ? E2M1_CODE_OF_ONE : 0);
}

/* The code function of a case's matrix. */
static uint8_t (*code_function_of(const struct exact_case *exact_case))(uint64_t, uint32_t, uint32_t)
{
    const bool is_e2m1 = is_of_experts(exact_case->kernel);
    switch (exact_case->weights) {
    case WEIGHTS_ONE_BINADE:
        return one_binade_code;
    case WEIGHTS_SMALL:
        return small_code;
    case WEIGHTS_LARGEST:
        return is_e2m1 ? largest_e2m1_code : largest_e4m3_code;
    case WEIGHTS_CANCELLING:
        if (exact_case->kernel == KERNEL_EXPERT_GATE) {
            return raising_e2m1_code;
        }
        return is_e2m1 ? cancelling_e2m1_code : cancelling_e4m3_code;
    case WEIGHTS_ANY:
    default:
        return is_e2m1 ? test_linear_e2m1_code : test_linear_e4m3_code;
    }
}

/* One fed code of a kind, from a hash. */
static uint8_t fed_code_of(enum fed_kind kind, uint64_t hash)
{
    const uint32_t sign = ((uint32_t)(hash >> HASH_SIGN_SHIFT) & 1u) << E4M3_SIGN_SHIFT;
    const uint32_t choice = (uint32_t)(hash >> HASH_CHOICE_SHIFT);
    switch (kind) {
    case FED_LOW_CODES:
    case FED_CANCELLING_CODES:
        return (uint8_t)(sign | choice % LOW_CODE_COUNT);
    case FED_SUBNORMAL_CODES:
        return (uint8_t)(sign | (1u + choice % SUBNORMAL_CODE_COUNT));
    case FED_HIGH_CODES:
        return (uint8_t)(HIGH_FIRST_CODE + choice % HIGH_CODE_COUNT);
    case FED_ANY_CODES:
    case FED_EIGHTHS:
    case FED_SUBNORMAL_HALVES:
    case FED_PLAIN_VALUES:
    case FED_SMALL_VALUES:
    case FED_CANCELLING_VALUES:
    default:
        return (uint8_t)((choice & E4M3_NOT_A_NUMBER_MAGNITUDE) == E4M3_NOT_A_NUMBER_MAGNITUDE
                             ? (choice & E4M3_BYTE_MASK) ^ 1u : choice & E4M3_BYTE_MASK);
    }
}

/* The exponent, to base 2, of the unit the values of a row that is not quantized are whole multiples of. */
static int32_t value_unit_exponent_of(enum fed_kind kind)
{
    return kind == FED_EIGHTHS ? EIGHTHS_UNIT_EXPONENT
           : kind == FED_SUBNORMAL_HALVES ? SUBNORMAL_HALF_UNIT_EXPONENT
           : kind == FED_CANCELLING_VALUES ? CANCELLING_VALUES_UNIT_EXPONENT : SMALL_VALUES_UNIT_EXPONENT;
}

/* The exponent, to base 2, of the unit every product of an exact case is a whole multiple of. */
static int32_t unit_exponent_of(const struct exact_case *exact_case)
{
    const int32_t weight_unit = is_of_experts(exact_case->kernel) ? E2M1_UNIT_EXPONENT : ONE_BINADE_UNIT_EXPONENT;
    const bool is_quantized = exact_case->kernel != KERNEL_TRUNK_BFLOAT16;
    return weight_unit + (is_quantized ? CODE_UNIT_EXPONENT + (int32_t)exact_case->exponent_base - (int32_t)E8M0_BIAS
                                       : value_unit_exponent_of(exact_case->fed)) +
           (int32_t)exact_case->scale_code_base - (int32_t)E8M0_BIAS;
}

/* The form of linear_records.h a card's class takes for the product of a kernel: this test's copy of that table. */
static enum product_form form_on(const struct test_linear_card *card, enum exact_kernel kernel)
{
    if (kernel == KERNEL_TRUNK_BFLOAT16) {
        return FORM_BFLOAT16_ROWS;
    }
    if (IS_EXACT_PROFILE || card->arithmetic_class == ARITHMETIC_CLASS_SM_80) {
	    return FORM_CARRIED;
    }
    return is_of_experts(kernel) ? FORM_BLOCK_SCALED : FORM_CODES;
}

/* What a case's outputs are held to in a form: a block below the sums of the codes form adds nothing there alone;
 * in the carried form a vector above its largest exponent leaves no output a finite number; the block-scaled form's
 * terms are known at the ends too. */
static enum held_as held_on(const struct exact_case *exact_case, enum product_form form)
{
    const bool is_at_an_end = exact_case->held == HELD_TO_TOLERANCE || exact_case->held == HELD_AS_NOTHING;
    if (form == FORM_CARRIED &&
        exact_case->exponent_base + exact_case->exponent_spread > TEST_LINEAR_CARRIED_EXPONENT_LARGEST) {
        return HELD_NOT_FINITE;
    }
    if (form == FORM_BLOCK_SCALED && is_at_an_end) {
        return HELD_IN_ORDER;
    }
    return exact_case->held == HELD_AS_NOTHING && form != FORM_CODES ? HELD_TO_TOLERANCE : exact_case->held;
}

/* The exponent, to base 2, of what a card's output is of the reference's product of the case's matrix: the 32 of a
 * constant gate beside an up; and above the scale codes of its form a card multiplies by 2^127 where the true power
 * is one binade more for every step above the largest (linear_records.h; such a case has no spread): of the sum of
 * the two codes in the codes form, of the weights' code in the carried form and for rows that are not quantized.
 * The block-scaled form has no end: its instruction applies the true scales. */
static int32_t card_exponent_of(const struct exact_case *exact_case, enum product_form form)
{
    const bool is_e2m1 = is_of_experts(exact_case->kernel);
    const uint32_t scale_sum = exact_case->scale_code_base + (form == FORM_CODES ? exact_case->exponent_base : 0);
    const uint32_t largest =
	    form == FORM_BFLOAT16_ROWS ? TRUNK_BFLOAT16_SCALE_LARGEST :
	    form == FORM_CARRIED ? (is_e2m1 ? EXPERT_CARRIED_SCALE_LARGEST :
					      TRUNK_CARRIED_SCALE_LARGEST) :
	    is_e2m1		 ? EXPERT_SCALE_SUM_LARGEST :
				   TRUNK_SCALE_SUM_LARGEST;
    const int32_t steps_above = form != FORM_BLOCK_SCALED && scale_sum > largest ? (int32_t)(scale_sum - largest) : 0;
    return (exact_case->kernel == KERNEL_EXPERT_UP ? GATE_CONSTANT_EXPONENT : 0) - steps_above;
}

/* The exponent of the first block of 32 values of every vector of a case of gate and up, which has no spread. */
static uint32_t first_block_exponent_of(const struct exact_case *exact_case)
{
    return exact_case->exponent_base + (exact_case->fed == FED_CANCELLING_CODES ? CANCELLING_EXPONENT_STEP : 0);
}

/* Makes vector `vector` of a case, lays it into the image at fed (and, quantized, its exponents at exponents and
 * its operands at operands), and keeps what a product is fed with in fed_values and fed_exponents. */
static bool make_vector(struct test_linear_card *card,
			const struct exact_case *exact_case, uint64_t stream,
			uint32_t vector, struct linear_range fed,
			struct linear_range exponents,
			struct linear_range operands)
{
    const uint32_t width = exact_case->segment_count * exact_case->input_size;
    const uint32_t exponents_per_vector = width / VALUES_PER_BLOCK;
    double *const vector_fed_values = fed_values + (size_t)vector * width;
    uint8_t *const vector_exponents = fed_exponents + (size_t)vector * exponents_per_vector;
    if (exact_case->kernel != KERNEL_TRUNK_BFLOAT16) {
        const bool is_cancelling = exact_case->fed == FED_CANCELLING_CODES;
        for (uint32_t value_index = 0; value_index < width; value_index++) {
            /* cancelling: the second block of every three is fed the codes of the first */
            const uint32_t block_of_three = value_index / VALUES_PER_BLOCK % CANCELLING_BLOCKS;
            const bool is_copy = is_cancelling && block_of_three == 1;
            const uint32_t source_index = is_copy ? value_index - VALUES_PER_BLOCK : value_index;
            /* a gate: the third block, which does not cancel, is fed codes without a sign */
            const uint32_t sign_mask = exact_case->kernel == KERNEL_EXPERT_GATE && is_cancelling &&
                                               block_of_three == CANCELLING_BLOCKS - 1 ? E4M3_MAGNITUDE_MASK
                                                                                       : E4M3_BYTE_MASK;
            const uint32_t code = fed_code_of(exact_case->fed, test_linear_hash(stream + vector, source_index));
            /* gate and up: value 0 is what the one weight of the constant matrix multiplies */
            vector_codes[value_index] = (uint8_t)(is_gate_or_up(exact_case->kernel) && source_index == 0
                                                      ? CONSTANT_FED_CODE : code & sign_mask);
        }
        for (uint32_t block_index = 0; block_index < exponents_per_vector; block_index++) {
            /* cancelling: the two blocks that cancel are 2^20 above the third */
            const bool is_large = is_cancelling && block_index % CANCELLING_BLOCKS != CANCELLING_BLOCKS - 1;
            const uint32_t step = is_large ? CANCELLING_EXPONENT_STEP : 0;
            vector_exponents[block_index] =
                (uint8_t)(exact_case->exponent_base + step +
                          test_linear_hash(stream + EXPONENT_STREAM + vector, block_index) %
                              ((uint64_t)exact_case->exponent_spread + 1));
        }
	return test_linear_fed_values_of_codes(
		       vector_codes, width, vector_fed_values) == STATUS_OK &&
	       test_linear_card_put_codes(card, fed, (uint64_t)vector * width,
					  vector_codes, width) &&
	       test_linear_card_put(card, exponents,
				    (uint64_t)vector * exponents_per_vector,
				    vector_exponents, exponents_per_vector) &&
	       test_linear_card_put_operands(
		       card, operands, (uint64_t)vector * width * VALUE_BYTES,
		       vector_codes, vector_exponents, width);
    }
    test_linear_values(vector_values, width, stream + vector, TEST_LINEAR_VALUES_PLAIN);
    for (uint32_t value_index = 0; exact_case->fed != FED_PLAIN_VALUES && value_index < width; value_index++) {
        /* cancelling: the second block of every three holds the values of the first, both 2^20 above the third */
        const bool is_cancelling = exact_case->fed == FED_CANCELLING_VALUES;
        const uint32_t block_of_three = value_index / VALUES_PER_BLOCK % CANCELLING_BLOCKS;
        const uint32_t source_index =
            is_cancelling && block_of_three == 1 ? value_index - VALUES_PER_BLOCK : value_index;
        const bool is_large = is_cancelling && block_of_three != CANCELLING_BLOCKS - 1;
        /* a whole number from -60 to 60 of the kind's unit: a bfloat16 and a 16-bit float exactly */
        const int32_t whole =
            (int32_t)(test_linear_hash(stream + vector, source_index) % EIGHTHS_SPREAD) - EIGHTHS_LARGEST;
        const int32_t unit =
            value_unit_exponent_of(exact_case->fed) + (is_large ? (int32_t)CANCELLING_EXPONENT_STEP : 0);
        vector_values[value_index] = test_linear_bfloat16_nearest(
            (double)whole * test_weight_format_e8m0_value((uint32_t)((int32_t)E8M0_BIAS + unit)));
    }
    return test_linear_fed_values_of_bfloat16(vector_values, width,
					      vector_fed_values) == STATUS_OK &&
	   test_linear_card_put(card, fed,
				(uint64_t)vector * width * VALUE_BYTES,
				vector_values, width * VALUE_BYTES);
}

/* What the one launch of 4-bit weights needs beside its matrices, the codes it is fed, their exponents and what it
 * writes: a table of two jobs and the arrays the chain's other launches would use, which no launch of this test
 * touches. Of a product of down those are the rows' codes and exponents, the hidden values and the row sums; of
 * gate and up the hidden codes and exponents, the pair outputs and the row sums. The others stay ranges of no
 * bytes. */
struct chain_ranges {
	struct linear_range jobs;
	struct linear_range row_codes;
	struct linear_range row_exponents;
	struct linear_range row_operands;
	struct linear_range hidden;
	struct linear_range hidden_codes;
	struct linear_range hidden_exponents;
	struct linear_range hidden_operands;
	struct linear_range pair_output;
	struct linear_range row_output;
};

/* matrix is the matrix of the case and constant the constant matrix beside a gate or an up; a product of down has
 * none, and its job names matrix three times. */
static bool chain_ranges_make(struct test_linear_card *card,
			      const struct exact_case *exact_case,
			      struct linear_range matrix,
			      struct linear_range constant,
			      struct chain_ranges *ranges)
{
    const bool is_down = exact_case->kernel == KERNEL_EXPERT_DOWN;
    const uint32_t row_sum_width = is_down ? exact_case->output_size : DUMMY_DOWN_OUTPUT_SIZE;
    struct expert_job_record jobs[JOB_COUNT];
    unsigned char *const table_bytes = (unsigned char *)jobs;
    memset(table_bytes, RUBBISH_BYTE, sizeof(jobs));
    jobs[0].row_count = 0;                      /* an empty job: nothing else of it is read */
    jobs[1] = (struct expert_job_record){
	    .gate_offset = exact_case->kernel == KERNEL_EXPERT_UP ?
				   constant.byte_offset :
				   matrix.byte_offset,
	    .up_offset = exact_case->kernel == KERNEL_EXPERT_GATE ?
				 constant.byte_offset :
				 matrix.byte_offset,
	    .down_offset = matrix.byte_offset,
	    .row_count = VECTORS
    };
    for (uint32_t column = 0; column < VECTORS; column++) {
        jobs[1].pair_index[column] = (COLUMN_STRIDE * column + COLUMN_FIRST) % VECTORS;
        jobs[1].route_weight[column] = 1.0f;
    }
    ranges->jobs = test_linear_card_work(card, sizeof(jobs));
    if (is_down) {
        ranges->row_codes = test_linear_card_work(card, (uint64_t)VECTORS * DUMMY_GATE_INPUT_SIZE);
        ranges->row_operands = test_linear_card_work(card, (uint64_t)VECTORS * DUMMY_GATE_INPUT_SIZE * VALUE_BYTES);
        ranges->row_exponents =
            test_linear_card_work(card, (uint64_t)VECTORS * DUMMY_GATE_INPUT_SIZE / VALUES_PER_BLOCK);
        ranges->hidden = test_linear_card_work(card, (uint64_t)VECTORS * exact_case->input_size * VALUE_BYTES);
    } else {
        ranges->hidden_codes = test_linear_card_work(card, (uint64_t)VECTORS * exact_case->output_size);
        ranges->hidden_operands =
            test_linear_card_work(card, (uint64_t)VECTORS * exact_case->output_size * VALUE_BYTES);
        ranges->hidden_exponents =
            test_linear_card_work(card, (uint64_t)VECTORS * exact_case->output_size / VALUES_PER_BLOCK);
        ranges->pair_output = test_linear_card_work(card, (uint64_t)VECTORS * DUMMY_DOWN_OUTPUT_SIZE * VALUE_BYTES);
    }
    ranges->row_output = test_linear_card_work(
	    card, (uint64_t)VECTORS * row_sum_width * LINEAR_SUM_BYTES);
    return test_linear_card_put(card, ranges->jobs, 0, jobs, sizeof(jobs));
}

/* The launch of a case's product in slice_count slices: fed and exponents hold the vectors, output takes the
 * outputs. */
static bool
prepare_launch(const struct test_linear_card *card,
	       const struct exact_case *exact_case, struct linear_range matrix,
	       const struct chain_ranges *chain_ranges, struct linear_range fed,
	       struct linear_range exponents, struct linear_range operands,
	       struct linear_range output, uint32_t slice_count,
	       struct kernel_launch *launch_out)
{
    const bool is_e2m1 = is_of_experts(exact_case->kernel);
    const struct matrix_shape_record shape = {
	    exact_case->output_size, exact_case->input_size,
	    is_e2m1 ? E2M1_BITS_PER_WEIGHT : E4M3_BITS_PER_WEIGHT, 0
    };
    const struct linear_matrix trunk_matrix = { .stored = matrix,
						.shape = shape,
						.weight_format = E4M3,
						.slice_count = slice_count };
    if (exact_case->kernel == KERNEL_TRUNK_QUANTIZED) {
	    const struct linear_trunk_quantized_description description = {
		    .matrices = { trunk_matrix },
		    .codes = fed,
		    .exponents = exponents,
		    .operands = operands,
		    .outputs = { output },
		    .matrix_count = 1,
		    .row_count = VECTORS
	    };
	    return test_linear_expect_status(
		    linear_trunk_quantized_prepare(&description,
						   card->arena_byte_count,
						   &card->kernels, launch_out),
		    STATUS_OK, "a product of quantized rows is prepared");
    }
    if (exact_case->kernel == KERNEL_TRUNK_BFLOAT16) {
	    const struct linear_trunk_bfloat16_description description = {
		    .matrix = trunk_matrix,
		    .values = fed,
		    .output = output,
		    .row_count = VECTORS,
		    .segment_count = exact_case->segment_count
	    };
	    return test_linear_expect_status(
		    linear_trunk_bfloat16_prepare(&description,
						  card->arena_byte_count,
						  &card->kernels, launch_out),
		    STATUS_OK,
		    "a product of rows that are not quantized is prepared");
    }
    struct kernel_launch chain[LINEAR_ROUTED_EXPERTS_LAUNCH_COUNT];
    if (is_gate_or_up(exact_case->kernel)) {
        /* fed holds the rows' codes; output takes the hidden values of the eight pairs, one to a row */
	const struct linear_routed_experts_description description = {
		.jobs = chain_ranges->jobs,
		.expert_weights = test_linear_card_all_weights(card),
		.codes = fed,
		.exponents = exponents,
		.operands = operands,
		.hidden = output,
		.hidden_codes = chain_ranges->hidden_codes,
		.hidden_exponents = chain_ranges->hidden_exponents,
		.hidden_operands = chain_ranges->hidden_operands,
		.pair_output = chain_ranges->pair_output,
		.row_output = chain_ranges->row_output,
		.gate_up_shape = shape,
		.down_shape = { DUMMY_DOWN_OUTPUT_SIZE, exact_case->output_size,
				E2M1_BITS_PER_WEIGHT, 0 },
		.weight_format = E2M1,
		.job_count = JOB_COUNT,
		.row_count = VECTORS,
		.pairs_per_row = 1,
		.gate_up_slice_count = slice_count,
		.down_slice_count = 1,
		.activation_limit = NO_ACTIVATION_LIMIT
	};
	if (!test_linear_expect_status(
		    linear_routed_experts_prepare(&description,
						  card->arena_byte_count,
						  &card->kernels, chain),
		    STATUS_OK,
		    "the chain of the routed experts is prepared for its gate and up")) {
		return false;
	}
	*launch_out = chain[LINEAR_ROUTED_EXPERTS_GATE_UP];
	return true;
    }
    const struct linear_routed_experts_description description = {
	    .jobs = chain_ranges->jobs,
	    .expert_weights = matrix,
	    .codes = chain_ranges->row_codes,
	    .exponents = chain_ranges->row_exponents,
	    .operands = chain_ranges->row_operands,
	    .hidden = chain_ranges->hidden,
	    .hidden_codes = fed,
	    .hidden_exponents = exponents,
	    .hidden_operands = operands,
	    .pair_output = output,
	    .row_output = chain_ranges->row_output,
	    .gate_up_shape = { exact_case->input_size, DUMMY_GATE_INPUT_SIZE,
			       E2M1_BITS_PER_WEIGHT, 0 },
	    .down_shape = shape,
	    .weight_format = E2M1,
	    .job_count = JOB_COUNT,
	    .row_count = VECTORS,
	    .pairs_per_row = 1,
	    .gate_up_slice_count = 1,
	    .down_slice_count = slice_count,
	    .activation_limit = ACTIVATION_LIMIT
    };
    if (!test_linear_expect_status(
		linear_routed_experts_prepare(&description,
					      card->arena_byte_count,
					      &card->kernels, chain),
		STATUS_OK,
		"the chain of the routed experts is prepared for its product of down")) {
	    return false;
    }
    *launch_out = chain[LINEAR_ROUTED_EXPERTS_DOWN];
    return true;
}

/* A number cut toward zero to a float32: the float32 nearest it on that side, a subnormal one or the largest. */
static float float32_toward_zero(double value)
{
    float cut = (float)value;
    uint32_t bits = 0;
    if (value >= 0.0 ? (double)cut > value : (double)cut < value) {
        memcpy(&bits, &cut, sizeof(bits));
        bits -= 1u;                             /* one step toward zero: the magnitude is not zero */
        memcpy(&cut, &bits, sizeof(cut));
    }
    return cut;
}

/* The terms of the blocks of 32 inputs of every output of a case held in order, in block_terms: the sum of the
 * block's products times both scales (rows that are not quantized have one, the weights') and times card_factor,
 * for the vectors in fed_values and fed_exponents. With is_cut each is cut toward zero to a float32, as the
 * block-scaled instruction gives it; else false, behind a failure line, if a term is no float32 exactly: the case is
 * then not what it says. */
static bool block_terms_make(const struct test_linear_card *card,
			     const struct test_linear_matrix_recipe *recipe,
			     struct linear_range matrix, bool is_quantized,
			     double card_factor, bool is_cut)
{
    const uint8_t *const stored = test_linear_card_stored(card, matrix);
    const size_t input_size = recipe->shape.input_size;
    const size_t block_count = input_size / VALUES_PER_BLOCK;
    if (stored == NULL || block_count > MAXIMUM_ORDERED_BLOCKS || recipe->shape.output_size > MAXIMUM_OUTPUT_SIZE) {
        test_linear_fail("a case held in order has more blocks or outputs than the test keeps terms for");
        return false;
    }
    for (uint32_t output_index = 0; output_index < recipe->shape.output_size; output_index++) {
        for (uint32_t input_index = 0; input_index < recipe->shape.input_size; input_index++) {
            struct test_weight_format_weight weight = {0};
            const size_t block_index = input_index / VALUES_PER_BLOCK;
	    if (test_weight_format_read_weight(recipe->format, &recipe->shape,
					       stored, matrix.byte_count,
					       output_index, input_index,
					       &weight) != STATUS_OK) {
		    test_linear_fail(
			    "the reference does not read a weight of a case held in order");
		    return false;
	    }
	    for (size_t vector = 0; vector < VECTORS; vector++) {
                double *const sum = &block_sums[vector * block_count + block_index];
                *sum = (input_index % VALUES_PER_BLOCK == 0 ? 0.0 : *sum) +
                       weight.value * fed_values[vector * input_size + input_index];
            }
        }
        for (size_t index = 0; index < VECTORS * block_count; index++) {
            const double fed_scale = is_quantized ? test_weight_format_e8m0_value(fed_exponents[index]) : 1.0;
            const double term = block_sums[index] * fed_scale * card_factor;
            const float float_term = is_cut ? float32_toward_zero(term) : (float)term;
            if (!is_cut && (double)float_term != term) {
                test_linear_fail("a block of a case held in order is no float32 exactly: the case is not what it says");
                return false;
            }
            block_terms[(index / block_count * recipe->shape.output_size + output_index) * block_count +
                        index % block_count] = float_term;
        }
    }
    return true;
}

/* The bfloat16 bits of one output as linear_records.h orders its additions, from the float32 terms of its blocks:
 * slice i holds loads floor(i L / S) to floor((i + 1) L / S) - 1 and is one chain from +0 in ascending blocks (a
 * term is exact, so adding it is the kernel's fused multiply-add); the chains are added from +0 in ascending
 * order; one rounding to bfloat16. */
static uint16_t ordered_bits(const float *terms, uint32_t load_count, uint32_t slice_count)
{
    float sum = 0.0f;
    for (uint32_t slice = 0; slice < slice_count; slice++) {
        float chain = 0.0f;
        for (uint32_t block_index = slice * load_count / slice_count * BLOCKS_PER_LOAD;
             block_index < (slice + 1) * load_count / slice_count * BLOCKS_PER_LOAD; block_index++) {
            chain = terms[block_index] + chain;
        }
        sum += chain;
    }
    return test_linear_bfloat16_nearest((double)sum);
}

/* The bits output `index` of a case is held to bit for bit in slice_count slices, from products and block_terms;
 * of a case held to the tolerance, the double result rounded once. */
static uint16_t held_bits(const struct exact_case *exact_case, enum held_as held, uint64_t index,
                          uint32_t slice_count)
{
    const uint32_t block_count = exact_case->input_size / VALUES_PER_BLOCK;
    if (held == HELD_IN_ORDER) {
	    return ordered_bits(block_terms + index * block_count,
				exact_case->input_size / LINEAR_VALUES_PER_LOAD,
				IS_EXACT_PROFILE ? 1u : slice_count);
    }
    return held == HELD_AS_NOTHING ? 0 : test_linear_bfloat16_nearest(products[index].sum);
}

static void test_exact_case(struct test_linear_card *card, const struct exact_case *exact_case, uint32_t case_index)
{
	struct kernel_launch launches[LINEAR_SLICE_INSTANCES];
	struct linear_range slice_outputs[LINEAR_SLICE_INSTANCES];
	struct chain_ranges chain_ranges;
	const enum product_form form = form_on(card, exact_case->kernel);
	const enum held_as held = held_on(exact_case, form);
	const bool is_e2m1 = is_of_experts(exact_case->kernel);
	const bool is_quantized = exact_case->kernel != KERNEL_TRUNK_BFLOAT16;
	const uint32_t width =
		exact_case->segment_count * exact_case->input_size;
	const uint32_t output_size = exact_case->output_size;
	const uint64_t output_count = (uint64_t)VECTORS * output_size;
	const uint32_t load_count =
		exact_case->input_size / LINEAR_VALUES_PER_LOAD;
	const uint32_t card_slices = test_linear_card_slices(
		card, LINEAR_MAXIMUM_SLICES, is_gate_or_up(exact_case->kernel));
	const uint32_t most_slices = load_count < card_slices ? load_count :
								card_slices;
	uint32_t variant_count =
		0; /* stamped counts up to most_slices: 1 on, each twice the last */
	const uint64_t stream = EXACT_STREAM + case_index * STREAMS_PER_CASE;
	const struct test_linear_matrix_recipe recipe = {
		.format = is_e2m1 ? E2M1 : E4M3,
		.shape = { output_size, exact_case->input_size,
			   is_e2m1 ? E2M1_BITS_PER_WEIGHT :
				     E4M3_BITS_PER_WEIGHT,
			   0 },
		.stream = stream + VECTORS,
		.code = code_function_of(exact_case),
		.scale_code_base = exact_case->scale_code_base,
		.scale_code_spread = exact_case->scale_code_spread
	};
	/* gate and up: the constant matrix makes of its one weight of 1 and of the 2^-5 every vector holds at value 0,
     * in a block of the one exponent every vector has there, 1 as the up or 32 as the gate */
	const struct test_linear_matrix_recipe constant_recipe = {
		.format = E2M1,
		.shape = recipe.shape,
		.stream = stream,
		.code = constant_e2m1_code,
		.scale_code_base =
			(uint32_t)((int32_t)(E8M0_BIAS + E8M0_BIAS) -
				   CONSTANT_FED_EXPONENT -
				   (int32_t)first_block_exponent_of(exact_case) +
				   (exact_case->kernel == KERNEL_EXPERT_UP ?
					    GATE_CONSTANT_EXPONENT :
					    0)),
		.scale_code_spread = 0
	};
	/* what a card's outputs are of the reference's products: see card_exponent_of */
	const double card_factor = test_weight_format_e8m0_value(
		(uint32_t)((int32_t)E8M0_BIAS +
			   card_exponent_of(exact_case, form)));
	/* 2^24 units: the unit of an exact case is a power of two an E8M0 byte stands for */
	const double exact_limit =
		EXACT_FLOAT32_WHOLE_NUMBERS * card_factor *
		test_weight_format_e8m0_value(
			(uint32_t)((int32_t)E8M0_BIAS +
				   unit_exponent_of(exact_case)));
	struct linear_range matrix = { 0, 0 };
	struct linear_range constant = { 0, 0 };
	uint64_t unequal_count = 0;
	uint64_t wrong_total = 0;
	uint64_t lost_count = 0;
	bool is_ready = width <= MAXIMUM_FED_WIDTH &&
			output_size <= MAXIMUM_OUTPUT_SIZE;

	test_linear_context("card %" PRIu32 ", %s", card->card_index,
			    exact_case->name);
	(void)printf("%s (%" PRIu32 " by %" PRIu32 ")\n", exact_case->name,
		     output_size, exact_case->input_size);
	test_linear_card_clear(card);
	const struct linear_range fed = test_linear_card_work(
		card,
		(uint64_t)VECTORS * width * (is_quantized ? 1 : VALUE_BYTES));
	const struct linear_range exponents =
		is_quantized ?
			test_linear_card_work(card, (uint64_t)VECTORS * width /
							    VALUES_PER_BLOCK) :
			fed;
	const struct linear_range operands =
		is_quantized ?
			test_linear_card_work(card, (uint64_t)VECTORS * width *
							    VALUE_BYTES) :
			fed;
	memset(&chain_ranges, 0, sizeof(chain_ranges));
	is_ready = is_ready && test_linear_card_matrix(card, &recipe, &matrix);
	if (is_ready && is_gate_or_up(exact_case->kernel)) {
		is_ready =
			test_linear_expect(
				exact_case->exponent_spread == 0 &&
					exact_case->segment_count == 1,
				"a case of gate and up has one exponent in the first block of every vector") &&
			test_linear_card_matrix(card, &constant_recipe,
						&constant);
	}
    is_ready = is_ready && (!is_e2m1 || chain_ranges_make(card, exact_case, matrix, constant, &chain_ranges));
    for (uint32_t vector = 0; is_ready && vector < VECTORS; vector++) {
        is_ready = make_vector(card, exact_case, stream, vector, fed, exponents, operands);
    }
    for (uint32_t slice_count = 1; is_ready && slice_count <= most_slices; slice_count += slice_count) {
        slice_outputs[variant_count] = test_linear_card_work(card, output_count * VALUE_BYTES);
        is_ready = prepare_launch(card, exact_case, matrix, &chain_ranges, fed, exponents, operands,
                                  slice_outputs[variant_count], slice_count, &launches[variant_count]);
        variant_count++;
    }
    if (!is_ready || !test_linear_card_has_room(card) ||
	!test_linear_expect_status(
		test_linear_products(
			recipe.format, &recipe.shape,
			test_linear_card_stored(card, matrix),
			matrix.byte_count, 0, output_size, fed_values,
			is_quantized ? fed_exponents : NULL, VECTORS,
			exact_case->segment_count, products),
		STATUS_OK, "the reference computes the products of the case")) {
	    (void)test_linear_expect(is_ready, "the case is laid out");
	    return;
    }
    /* from here on the products are what a card must give: below the scale codes every block adds nothing */
    for (uint64_t index = 0; index < output_count; index++) {
        const bool is_nothing = held == HELD_AS_NOTHING;
        products[index].sum = is_nothing ? 0.0 : products[index].sum * card_factor;
        products[index].magnitude_sum = is_nothing ? 0.0 : products[index].magnitude_sum * card_factor;
    }
    /* the proof of an exact case: every magnitude sum is below 2^24 of the unit of its products */
    for (uint64_t index = 0; held == HELD_EXACTLY && index < output_count; index++) {
        if (!(products[index].magnitude_sum < exact_limit)) {
            test_linear_fail("output %" PRIu64 " is not exact in float32: the case is not what it says", index);
            return;
        }
    }
    if (held == HELD_IN_ORDER && !block_terms_make(card, &recipe, matrix, is_quantized, card_factor,
                                                   form == FORM_BLOCK_SCALED && exact_case->held != HELD_IN_ORDER)) {
        return;
    }
    /* the proof of a case that shows a gate: every gate a card must find is its own hidden step */
    for (uint32_t slice_count = 1; exact_case->kernel == KERNEL_EXPERT_GATE && slice_count <= most_slices;
         slice_count += slice_count) {
        for (uint64_t index = 0; index < output_count; index++) {
            if (!(test_linear_bfloat16_value(held_bits(exact_case, held, index, slice_count)) >= GATE_LEAST)) {
                test_linear_fail("in %" PRIu32 " slices the gate of output %" PRIu64 " is below 32: the case is not "
                                 "what it says", slice_count, index);
                return;
            }
        }
    }
    if (!test_linear_card_load(card) || !test_linear_card_run(card, launches, variant_count) ||
        !test_linear_card_fetch(card)) {
        return;
    }
    for (uint32_t variant = 0, slice_count = 1; variant < variant_count; variant++, slice_count += slice_count) {
        struct test_linear_comparison comparison = {0};
        uint64_t wrong_count = 0;
        if (!test_linear_card_get(card, slice_outputs[variant], 0, outputs, output_count * VALUE_BYTES)) {
            return;
        }
        const bool is_finite = held != HELD_NOT_FINITE;
        const uint64_t first_outside =
            is_finite ? test_linear_compare(outputs, products, output_count, exact_case->input_size, slice_count,
                                            &comparison)
                      : TEST_LINEAR_NO_OUTPUT;
        for (uint64_t index = 0; index < output_count; index++) {
            const uint16_t rounded_once = test_linear_bfloat16_nearest(products[index].sum);
            /* at an end of the scale codes a power of two that became 0 or infinite shows as such an output */
            const bool is_lost = !test_linear_bfloat16_is_finite(outputs[index]) ||
                                 test_linear_bfloat16_ordinal(outputs[index]) == 0;
            lost_count += is_finite && is_lost && test_linear_bfloat16_is_finite(rounded_once) &&
                          test_linear_bfloat16_ordinal(rounded_once) != 0 ? 1u : 0u;
            if (held == HELD_NOT_FINITE) {
                wrong_count += test_linear_bfloat16_is_finite(outputs[index]) ? 1u : 0u;
            } else if (held != HELD_TO_TOLERANCE) {
                wrong_count += outputs[index] != held_bits(exact_case, held, index, slice_count) ? 1u : 0u;
            }
        }
        if (first_outside != TEST_LINEAR_NO_OUTPUT || wrong_count != 0) {
            test_linear_fail("in %" PRIu32 " slices %" PRIu64 " of %" PRIu64 " outputs are not %s, and %" PRIu64
                             " are outside the tolerance", slice_count, wrong_count, output_count, held_names[held],
                             comparison.outside_count);
        }
        wrong_total += wrong_count + (held == HELD_TO_TOLERANCE ? comparison.outside_count : 0);
        unequal_count += is_finite ? output_count - comparison.equal_count : output_count;
    }
    /* (a chain that cancels may rightly end at zero where the double result does not) */
    (void)test_linear_expect(exact_case->held == HELD_IN_ORDER || lost_count == 0,
                             "no output is zero or infinite where the reference's is neither");
    (void)test_linear_expect(exact_case->held != HELD_IN_ORDER || unequal_count != 0,
                             "the order of the additions shows in outputs of a case that is held in order");
    (void)test_linear_card_expect_untouched(card, fed, "what the vectors feed");
    (void)test_linear_card_expect_untouched(card, exponents, "the exponents of the vectors");
    (void)test_linear_card_expect_untouched(card, operands, "the operands of the vectors");
    (void)test_linear_card_expect_weights_untouched(card);
    (void)printf("  1, 2, 4, .. up to %" PRIu32 " slices, %" PRIu64 " outputs each, held to %s: %" PRIu64 " are not; %"
                 PRIu64 " differ from the double result rounded once\n", 1u << (variant_count - 1u), output_count,
                 held_names[held], wrong_total, unequal_count);
}

int main(void)
{
    static struct test_linear_card card;
    uint32_t card_count = 0;

    (void)alarm(TEST_TIME_LIMIT_SECONDS);
    (void)test_linear_expect_status(backend_cuda_value.card_count(&card_count),
				    STATUS_OK, "the cards are counted");
    if (!test_linear_expect(card_count > 0 &&
				    card_count <= BACKEND_MAXIMUM_CARDS,
			    "the machine has cards for the test")) {
	    return EXIT_FAILURE;
    }
    for (uint32_t card_index = 0; card_index < card_count; card_index++) {
        if (test_linear_card_open(&card, card_index, WORK_CAPACITY, WEIGHTS_CAPACITY)) {
            for (uint32_t case_index = 0; case_index < COUNT_OF(exact_cases); case_index++) {
                test_exact_case(&card, &exact_cases[case_index], case_index);
            }
            (void)printf("card %" PRIu32 " (class %" PRIu32 "): digest of everything it computed 0x%016" PRIX64 "\n",
                         card_index, card.arithmetic_class, card.digest);
        }
        test_linear_card_close(&card);
    }
    if (test_linear_failures() != 0) {
        (void)fprintf(stderr, "linear_exact_card_test FAILED: %d failures\n", test_linear_failures());
        return EXIT_FAILURE;
    }
    (void)printf("linear_exact_card_test passed on %" PRIu32 " cards\n", card_count);
    return EXIT_SUCCESS;
}
