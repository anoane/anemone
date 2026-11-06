/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * trunk_wide_card_test: the wide trunk instances (trunk_linear_quantized_wide_<S>,
 * trunk_linear_bfloat16_wide_<S>; linear_records.h) on the real cards, against today's thin launches of 8 rows.
 *
 *     trunk_wide_card_test [--time]
 *
 * It needs the cards free (`make card_tests`): a card without room for the arena FAILS the test, it is never skipped.
 * On every card, in each class its kernels are built for (class 80 on a 170HX; on the PRO class 120 native, the
 * default, then class 120's emulation of class 80), for every trunk shape of the model at the default slice count of
 * that class (linear.c): attn.wq_a with attn.wkv (one launch of two matrices), attn.wq_b, attn.indexer.wq_b, attn.wo_b,
 * engram.wkv, the shared expert's gate with up (two matrices) and down, all of quantized rows, and attn.wo_a of
 * bfloat16 rows in its 8 segments; attn.wq_a with attn.wkv and attn.wo_a also at every other count a kernel is
 * stamped for that the class runs (the emulation at most LINEAR_CLASS_120_EXACT_MOST_SLICES), so that every
 * wide instance runs, and the interleaved one, with two matrices, in every class. The draft's mtp.0.main_proj
 * (dspark_v1/DESIGN.md K8) runs at every count too, in a card's own class alone: the draft never runs in class 120's
 * emulation (its decision 7), which has no row for it:
 *   - 1,024 rows are laid out (quantized once on the card: the first PLAIN_ROWS of plain values, the others of the
 *     edges; bfloat16 rows of small values) and run through the thin launches of 8 rows (128 of them) and, over the
 *     same first rows, through the wide launch of 64, 100, 256 and 1,024 rows as linear.c prepares it (a block for each
 *     group of 8 rows of each tile) and with a block a tile (every group in turn, which linear_records.h allows); a
 *     shape of quantized rows in one slice also through the interleaved instance in its place, a block a chunk of 32
 *     rows and a block a tile. Every output byte of a wide launch must be the thin launches': 100 rows are twelve
 *     groups of 8 and one of 4, held against the thin launch of rows 96 to 103, and a last chunk of one group. Every
 *     output starts as rubbish, a byte of its own for the thin launches and for each variant, so that a byte a launch
 *     does not write shows.
 *   - Counts against the default, on class 80 and in class 120's emulation, whose slices add in the reference order
 *     whatever the count (linear.h): the thin launches of attn.wq_a with attn.wkv and of attn.wo_a at every other
 *     count must give the default count's bytes over all 1,024 rows, and so, by the check above, every wide variant
 *     too. This holds the counts the wide forms take there (linear.c: attn.wq_a with attn.wkv in one slice on class
 *     80, whose default is 16, and in 4 in the emulation, whose default is 8; attn.wo_a in one slice on both, against
 *     4 and 8) to the bits of the model's thin launches, 16 slices among them, which the cases held in order of
 *     linear_exact_card_test (12 loads) never reach. Class 120 native, whose bits depend on the count, is held at each
 *     count alone.
 *   - The guards behind every range, the rows' values (only read) and the matrices come back untouched.
 *   - Shown failing once: the sign of every value of row CONTROL_ROW (group 64, column 5) is flipped on the card, the
 *     rows quantized again and the wide launch of 1,024 rows run again: that row's outputs must then differ from the
 *     thin launches', and no other byte.
 *   - With --time each case is timed too: the thin launches of its rows and every variant of the wide launch, each the
 *     median of TIMED_RUNS runs (host clock around the queue and the wait) after one that warms the card, each run
 *     after l2_flush, so that it reads its weights from memory (research/speed/CONVENTIONS.md 3.5). The thin launches
 *     are queued back to back, as a step without graphs queues them: from the second on they find what of the matrix
 *     the L2 kept (on the PRO every shape but engram.wkv), which production, running the rest of a layer between two
 *     slabs, reads from memory again; so the thin time is a lower bound, and the gain shown one too.
 * One line per case says what was compared, and with --time what was measured. It FAILS on any difference.
 */
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "anemone/backend_seam.h"
#include "anemone/linear.h"
#include "anemone/linear_records.h"
#include "anemone/status.h"
#include "anemone/weight_format_seam.h"
#include "linear_card_fixture.h"

#define TEST_TIME_LIMIT_SECONDS 3600            /* the alarm ends the test here, whatever the driver does */
#define WORK_CAPACITY (512ull << 20u)           /* attn.wq_b, 432 MiB: 1,024 rows, outputs of 1,024 rows and 4 times
                                                   1,444; and what l2_flush reads */
#define WEIGHTS_CAPACITY (160ull << 20u)        /* engram.wkv: 150.1 MiB */
#define LARGEST_L2_BYTES (128ull << 20u)        /* the PRO's; a 170HX has 32 MiB */
#define FLUSHED_L2_SIZES 4u                     /* research/speed/CONVENTIONS.md 3.5 */
#define COUNT_OF(table) ((uint32_t)(sizeof(table) / sizeof((table)[0])))
#define MATRICES LINEAR_MERGED_MATRICES
#define GROUP_ROWS \
	LINEAR_MAXIMUM_ROWS /* the rows of a thin launch, and of a group of a wide one */
#define CHUNK_GROUPS 4u                         /* the groups of a chunk of the interleaved instance */
#define MOST_ROWS LINEAR_WIDE_MOST_ROWS
#define MOST_GROUPS (MOST_ROWS / GROUP_ROWS)
#define CHUNK_ROWS 64u                          /* the wide row counts of the cases: these, 256 and the most */
#define UNEVEN_ROWS 100u                        /* twelve groups of 8 and one of 4: no whole number of chunks */
#define QUARTER_ROWS 256u
#define CONTROL_ROW 517u                        /* group 64, column 5 */
#define PLAIN_ROWS 4u                           /* the first rows hold plain values, the others the edges */
#define MOST_ROW_WIDTH 32768u                   /* the values of a row: attn.wo_a's 8 segments of 4,096 */
#define MOST_SWEPT_OUTPUTS 8192u                /* attn.wo_a's: the most outputs of a swept shape (matrices together) */
#define E4M3 WEIGHT_FORMAT_E4M3_SCALED
#define E4M3_BITS_PER_WEIGHT 8u
#define SCALE_CODE_BASE 118u                    /* every form's codes in range for the rows' exponents (linear.h) */
#define SCALE_CODE_SPREAD 3u
#define VALUES_PER_BLOCK TEST_LINEAR_VALUES_PER_BLOCK
#define VALUE_BYTES ((uint64_t)LINEAR_BFLOAT16_BYTES)
#define BFLOAT16_SIGN_BIT 0x8000u
#define WIDE_STREAM 0x13000000ull               /* the streams of the hashes, far from those of the other tests */
#define MATRIX_STREAM 0x100ull
#define ROW_STREAM 0x10000ull
#define THIN_RUBBISH_BYTE 0x5C
#define WIDE_RUBBISH_BYTE 0xA7
#define FUNCTION_NAME_CAPACITY 64u              /* bytes of a kernel's function name, its terminating zero among them */
#define WIDE_TILE_RUBBISH_BYTE 0x3E
#define INTERLEAVED_RUBBISH_BYTE 0x96
#define INTERLEAVED_TILE_RUBBISH_BYTE 0x69
#define FILL_PIECE_BYTES 65536u
#define TIMED_RUNS 5u                           /* odd: the median is the middle one */
#define MEDIAN_RUN (TIMED_RUNS / HALVES)        /* its place among the sorted runs */
#define HALVES 2u
#define MOST_ARGUMENTS 2                        /* the program's name and --time */
#define MOST_PASSES 2u                          /* the classes of one card: the PRO's native and emulated */
#define SIZES_CAPACITY 64u                      /* the text of a shape's sizes */
#define NANOSECONDS_PER_SECOND 1000000000ull
#define NANOSECONDS_PER_MICROSECOND 1000.0

enum test_status { TEST_PASSED = 0, TEST_FAILED = 1, TEST_WRONG_USAGE = 2 };

/* The row counts of the wide launches of a shape, and their variants: the wide instance as linear.c prepares it (a
 * block for each group of 8 rows of each tile) and with a block a tile; the interleaved instance in its place
 * (quantized rows in one slice), a block a chunk and a block a tile. */
static const uint32_t case_rows[] = {CHUNK_ROWS, UNEVEN_ROWS, QUARTER_ROWS, MOST_ROWS};
#define CASES COUNT_OF(case_rows)
#define MOST_CASES 4u
#define MOST_CASE (MOST_CASES - 1u)             /* the case of 1,024 rows */
enum wide_variant {
    VARIANT_WIDE,
    VARIANT_WIDE_A_BLOCK_A_TILE,
    VARIANT_INTERLEAVED,
    VARIANT_INTERLEAVED_A_BLOCK_A_TILE,
    VARIANT_COUNT
};
#define VARIANTS_OF_EVERY_SHAPE 2u              /* the wide instance's; the interleaved ones follow */
static const char *const variant_names[VARIANT_COUNT] = {
    "wide, a block a group", "wide, a block a tile", "interleaved, a block a chunk", "interleaved, a block a tile"};
static const unsigned char rubbish_of[VARIANT_COUNT] = {WIDE_RUBBISH_BYTE, WIDE_TILE_RUBBISH_BYTE,
                                                        INTERLEAVED_RUBBISH_BYTE, INTERLEAVED_TILE_RUBBISH_BYTE};

/* A trunk shape of the model: its one or two matrices (output_sizes[1] 0 for one) and, of bfloat16 rows, the segments
 * of its block-diagonal matrix, whose slice count is that of the whole matrix's shape (linear.h); is_swept: run at
 * every slice count the class runs, not only its default; is_draft: the draft's, never in class 120's emulation. */
struct trunk_shape {
    const char *name;
    uint32_t output_sizes[MATRICES];
    uint32_t input_size;
    bool is_bfloat16;
    uint32_t segment_count;
    bool is_swept;
    bool is_draft;
};
static const struct trunk_shape trunk_shapes[] = {
    {"attn.wq_a with attn.wkv", {1280, 512}, 5120, false, 1, true, false},
    {"attn.wq_b", {32768, 0}, 1280, false, 1, false, false},
    {"attn.indexer.wq_b", {4096, 0}, 1280, false, 1, false, false},
    {"attn.wo_b", {5120, 0}, 8192, false, 1, false, false},
    {"engram.wkv", {25600, 0}, 6144, false, 1, false, false},
    {"the shared expert's gate with up", {2304, 2304}, 5120, false, 1, false, false},
    {"the shared expert's down", {5120, 0}, 2304, false, 1, false, false},
    {"attn.wo_a", {8192, 0}, 4096, true, 8, true, false},
    {"mtp.0.main_proj", {5120, 0}, 15360, false, 1, true, true},
};

/* What one shape lays out on a card, and where its launches lie in launches. */
struct shape_case {
    const struct trunk_shape *shape;
    uint32_t matrix_count;
    uint32_t slice_count;
    uint32_t variant_count;                     /* VARIANT_COUNT for quantized rows in one slice, else 2 */
    uint32_t row_width;                         /* the values of a row: segment_count * input_size */
    struct linear_matrix matrices[MATRICES];
    struct linear_range values, codes, exponents, operands;
    struct linear_range thin_outputs[MATRICES];
    struct linear_range wide_outputs[VARIANT_COUNT][MOST_CASES][MATRICES];
    uint32_t thin_first, wide_first[VARIANT_COUNT], launch_count;   /* the quantization, if any, is launch 0 */
};

#define MOST_LAUNCHES (1u + MOST_GROUPS + VARIANT_COUNT * MOST_CASES)
static struct kernel_launch launches[MOST_LAUNCHES];
static uint16_t row_values[MOST_ROW_WIDTH];

_Static_assert(COUNT_OF(case_rows) == MOST_CASES && CONTROL_ROW < MOST_ROWS && UNEVEN_ROWS % GROUP_ROWS != 0 &&
                   (UNEVEN_ROWS + GROUP_ROWS - 1u) / GROUP_ROWS % CHUNK_GROUPS != 0,
               "MOST_CASES cases, the last of the most rows; the control row lies in it; one ends with a short group, "
               "in a chunk of the interleaved instance that its last row does not fill");
_Static_assert(WORK_CAPACITY >= FLUSHED_L2_SIZES * LARGEST_L2_BYTES, "l2_flush reads four L2 of the largest");

static uint64_t clock_nanoseconds(void)
{
    struct timespec now = {0};
    return clock_gettime(CLOCK_MONOTONIC, &now) == 0
               ? (uint64_t)now.tv_sec * NANOSECONDS_PER_SECOND + (uint64_t)now.tv_nsec : 0;
}

static uint32_t groups_of(uint32_t row_count)
{
    return (row_count + GROUP_ROWS - 1u) / GROUP_ROWS;
}

/* The blocks a tile of a variant takes for row_count rows (linear_records.h). */
static uint32_t blocks_per_tile_of(enum wide_variant variant, uint32_t row_count)
{
    return variant == VARIANT_WIDE          ? groups_of(row_count)
           : variant == VARIANT_INTERLEAVED ? (groups_of(row_count) + CHUNK_GROUPS - 1u) / CHUNK_GROUPS
                                            : 1u;
}

/* Rows first_row .. first_row + row_count - 1 of an array of rows of row_bytes. */
static struct linear_range rows_of(struct linear_range range,
				   uint64_t row_bytes, uint32_t first_row,
				   uint32_t row_count)
{
	return (struct linear_range){ range.byte_offset + first_row * row_bytes,
				      row_count * row_bytes };
}

/* Fills a range of the image with one byte, a piece at a time. */
static bool fill(struct test_linear_card *card, struct linear_range range,
		 unsigned char byte)
{
    static unsigned char piece[FILL_PIECE_BYTES];
    bool is_filled = true;
    memset(piece, byte, sizeof(piece));
    for (uint64_t done = 0; done < range.byte_count && is_filled; done += sizeof(piece)) {
        const uint64_t count = range.byte_count - done < sizeof(piece) ? range.byte_count - done : sizeof(piece);
        is_filled = test_linear_card_put(card, range, done, piece, count);
    }
    return is_filled;
}

/* A trunk launch of a shape's rows first_row .. first_row + row_count - 1 into outputs (one range a matrix). */
MUST_CHECK static result_status
trunk_prepare(const struct test_linear_card *card,
	      const struct shape_case *shape_case, uint32_t first_row,
	      uint32_t row_count, const struct linear_range *outputs,
	      struct kernel_launch *launch_out)
{
    const uint32_t input_size = shape_case->shape->input_size;
    if (shape_case->shape->is_bfloat16) {
	    const struct linear_trunk_bfloat16_description description = {
		    .matrix = shape_case->matrices[0],
		    .values = rows_of(shape_case->values,
				      shape_case->row_width * VALUE_BYTES,
				      first_row, row_count),
		    .output = outputs[0],
		    .row_count = row_count,
		    .segment_count = shape_case->shape->segment_count
	    };
	    return linear_trunk_bfloat16_prepare(&description,
						 card->arena_byte_count,
						 &card->kernels, launch_out);
    }
    const struct linear_trunk_quantized_description description = {
	    .matrices = { shape_case->matrices[0], shape_case->matrices[1] },
	    .codes = rows_of(shape_case->codes, input_size, first_row,
			     row_count),
	    .exponents = rows_of(shape_case->exponents,
				 input_size / VALUES_PER_BLOCK, first_row,
				 row_count),
	    .operands = rows_of(shape_case->operands, input_size * VALUE_BYTES,
				first_row, row_count),
	    .outputs = { outputs[0], outputs[1] },
	    .matrix_count = shape_case->matrix_count,
	    .row_count = row_count
    };
    return linear_trunk_quantized_prepare(&description, card->arena_byte_count,
					  &card->kernels, launch_out);
}

/* A shape's launches from launches[0] on: the rows' quantization (quantized rows), the thin launches of each group of
 * 8 rows, then the wide launch of each case in each variant: prepared, then given the variant's instance and grid. */
static bool launches_prepare(const struct test_linear_card *card, struct shape_case *shape_case)
{
    const struct trunk_shape *const shape = shape_case->shape;
    const uint32_t tile_count =
	    (shape->output_sizes[0] + shape->output_sizes[1]) /
	    LINEAR_OUTPUTS_PER_TILE;
    uint32_t count = 0;
    bool is_prepared = true;
    if (!shape->is_bfloat16) {
	    const struct linear_quantize_description quantize = {
		    shape_case->values,	   shape_case->codes,
		    shape_case->exponents, shape_case->operands,
		    shape->input_size,	   MOST_ROWS
	    };
	    is_prepared = test_linear_expect_status(
		    linear_quantize_prepare(&quantize, card->arena_byte_count,
					    &card->kernels, &launches[count++]),
		    STATUS_OK, "the quantization of the rows is prepared");
    }
    shape_case->thin_first = count;
    for (uint32_t group = 0; group < MOST_GROUPS && is_prepared; group++) {
	    struct linear_range outputs[MATRICES] = { { 0, 0 }, { 0, 0 } };
	    for (uint32_t matrix = 0; matrix < shape_case->matrix_count;
		 matrix++) {
		    outputs[matrix] =
			    rows_of(shape_case->thin_outputs[matrix],
				    shape->output_sizes[matrix] * VALUE_BYTES,
				    group * GROUP_ROWS, GROUP_ROWS);
	    }
	    is_prepared = test_linear_expect_status(
		    trunk_prepare(card, shape_case, group * GROUP_ROWS,
				  GROUP_ROWS, outputs, &launches[count++]),
		    STATUS_OK, "a thin launch is prepared");
    }
    for (uint32_t variant = 0; variant < shape_case->variant_count && is_prepared; variant++) {
        shape_case->wide_first[variant] = count;
        for (uint32_t case_index = 0; case_index < CASES && is_prepared; case_index++) {
		struct kernel_launch *const launch = &launches[count++];
		const uint32_t row_count = case_rows[case_index];
		is_prepared =
			test_linear_expect_status(
				trunk_prepare(
					card, shape_case, 0, row_count,
					shape_case->wide_outputs[variant]
								[case_index],
					launch),
				STATUS_OK, "a wide launch is prepared") &&
			test_linear_expect(
				launch->kernel !=
						launches[shape_case->thin_first]
							.kernel &&
					launch->grid[0] ==
						tile_count *
							groups_of(row_count),
				"a launch of more than 8 rows is of the wide instance, a block a group of a tile");
		launch->kernel =
			variant >= VARIANT_INTERLEAVED ?
				card->kernels
					.trunk_linear_quantized_interleaved :
				launch->kernel;
		launch->grid[0] = tile_count *
				  blocks_per_tile_of((enum wide_variant)variant,
						     row_count);
        }
    }
    shape_case->launch_count = count;
    return is_prepared;
}

/* Sets a shape's ranges aside, lays its matrices, rows and rubbish into the image and prepares its launches, in the
 * shape case's slice_count. */
static bool shape_lay_out(struct test_linear_card *card, struct shape_case *shape_case, uint32_t shape_index)
{
    const struct trunk_shape *const shape = shape_case->shape;
    const uint32_t input_size = shape->input_size;
    const uint32_t matrix_count = shape->output_sizes[1] != 0 ? MATRICES : 1u;
    shape_case->matrix_count = matrix_count;
    shape_case->row_width = shape->segment_count * input_size;
    bool is_laid_out = test_linear_expect(shape_case->row_width <= MOST_ROW_WIDTH, "a row fits the test's buffer");
    shape_case->variant_count = !shape->is_bfloat16 && shape_case->slice_count == 1u ? VARIANT_COUNT
                                                                                     : VARIANTS_OF_EVERY_SHAPE;
    test_linear_card_clear(card);
    for (uint32_t matrix = 0; matrix < matrix_count && is_laid_out; matrix++) {
        const struct test_linear_matrix_recipe recipe = {
            .format = E4M3, .shape = {shape->output_sizes[matrix], input_size, E4M3_BITS_PER_WEIGHT, 0},
            .stream = WIDE_STREAM + MATRIX_STREAM * (shape_index * MATRICES + matrix + 1u),
            .code = test_linear_e4m3_code, .scale_code_base = SCALE_CODE_BASE,
            .scale_code_spread = SCALE_CODE_SPREAD};
	shape_case->matrices[matrix] =
		(struct linear_matrix){ .shape = recipe.shape,
					.weight_format = E4M3,
					.slice_count =
						shape_case->slice_count };
	is_laid_out = test_linear_card_matrix(card, &recipe, &shape_case->matrices[matrix].stored);
    }
    shape_case->values = test_linear_card_work(card, (uint64_t)MOST_ROWS * shape_case->row_width * VALUE_BYTES);
    if (!shape->is_bfloat16) {
        shape_case->codes = test_linear_card_work(card, (uint64_t)MOST_ROWS * input_size);
        shape_case->exponents = test_linear_card_work(card, (uint64_t)MOST_ROWS * (input_size / VALUES_PER_BLOCK));
        shape_case->operands = test_linear_card_work(card, (uint64_t)MOST_ROWS * input_size * VALUE_BYTES);
    }
    for (uint32_t matrix = 0; matrix < matrix_count; matrix++) {
        const uint64_t output_row_bytes = shape->output_sizes[matrix] * VALUE_BYTES;
        shape_case->thin_outputs[matrix] = test_linear_card_work(card, MOST_ROWS * output_row_bytes);
        for (uint32_t variant = 0; variant < shape_case->variant_count; variant++) {
            for (uint32_t case_index = 0; case_index < CASES; case_index++) {
                shape_case->wide_outputs[variant][case_index][matrix] =
                    test_linear_card_work(card, case_rows[case_index] * output_row_bytes);
            }
        }
    }
    is_laid_out = is_laid_out && test_linear_card_has_room(card);
    for (uint32_t matrix = 0; matrix < matrix_count && is_laid_out; matrix++) {
        is_laid_out = fill(card, shape_case->thin_outputs[matrix], THIN_RUBBISH_BYTE);
        for (uint32_t index = 0; index < shape_case->variant_count * CASES && is_laid_out; index++) {
            is_laid_out = fill(card, shape_case->wide_outputs[index / CASES][index % CASES][matrix],
                               rubbish_of[index / CASES]);
        }
    }
    for (uint32_t row = 0; row < MOST_ROWS && is_laid_out; row++) {
        const enum test_linear_values_kind kind = row < PLAIN_ROWS        ? TEST_LINEAR_VALUES_PLAIN
                                                  : shape->is_bfloat16 ? TEST_LINEAR_VALUES_SMALL
                                                                       : TEST_LINEAR_VALUES_EDGES;
        test_linear_values(row_values, shape_case->row_width, WIDE_STREAM + ROW_STREAM * (shape_index + 1u) + row,
                           kind);
        const uint64_t row_bytes = shape_case->row_width * VALUE_BYTES;
        is_laid_out = test_linear_card_put(card, shape_case->values, row * row_bytes, row_values, row_bytes);
    }
    return is_laid_out && launches_prepare(card, shape_case);
}

/* Reads the whole work region on the card (the digest of backend_seam.h, its sum unused): at least FLUSHED_L2_SIZES
 * of the largest L2 of bytes that are no weights, so that the weights a run reads next come from memory. */
static bool l2_flush(struct test_linear_card *card)
{
    uint64_t unused_sum = 0;
    return test_linear_expect_status(
	    backend_cuda_value.digest(card->card, 0, card->work_capacity,
				      TEST_LINEAR_WAIT_LIMIT_NANOSECONDS,
				      &unused_sum),
	    STATUS_OK,
	    "the work region is read to empty the L2 of the weights");
}

/* The median of TIMED_RUNS runs of a run of launches, each after l2_flush, queued and waited for, after one that warms
 * the card, in microseconds; 0 behind a failure line. */
static double median_microseconds(struct test_linear_card *card,
				  const struct kernel_launch *timed,
				  uint32_t launch_count)
{
    uint64_t durations[TIMED_RUNS];
    bool is_timed = test_linear_card_run(card, timed, launch_count);
    for (uint32_t run = 0; run < TIMED_RUNS; run++) {
        is_timed = is_timed && l2_flush(card);
        const uint64_t started = clock_nanoseconds();
        is_timed = is_timed && test_linear_card_run(card, timed, launch_count);
        durations[run] = clock_nanoseconds() - started;
    }
    for (uint32_t index = 1; index < TIMED_RUNS; index++) {   /* sorted: the median is the middle one */
        const uint64_t duration = durations[index];
        uint32_t place = index;
        while (place > 0 && durations[place - 1u] > duration) {
            durations[place] = durations[place - 1u];
            place--;
        }
        durations[place] = duration;
    }
    const uint64_t median = durations[MEDIAN_RUN];
    return is_timed ? (double)median / NANOSECONDS_PER_MICROSECOND : 0.0;
}

/* Each case of a shape: every variant's outputs against the thin launches', byte for byte; with is_timed the times. */
static void shape_check(struct test_linear_card *card, const struct shape_case *shape_case, uint32_t arithmetic_class,
                        bool is_timed)
{
    const struct trunk_shape *const shape = shape_case->shape;
    char sizes[SIZES_CAPACITY];
    if (shape_case->matrix_count == MATRICES) {
        (void)snprintf(sizes, sizeof(sizes), "%" PRIu32 " and %" PRIu32 " by %" PRIu32, shape->output_sizes[0],
                       shape->output_sizes[1], shape->input_size);
    } else {
        (void)snprintf(sizes, sizeof(sizes), "%" PRIu32 " by %" PRIu32, shape->output_sizes[0], shape->input_size);
    }
    for (uint32_t case_index = 0; case_index < CASES; case_index++) {
        const uint32_t row_count = case_rows[case_index];
        uint64_t compared_bytes = 0;
        bool is_same = true;
        for (uint32_t index = 0; index < shape_case->variant_count * shape_case->matrix_count; index++) {
            const uint32_t variant = index / shape_case->matrix_count, matrix = index % shape_case->matrix_count;
            const uint64_t byte_count = (uint64_t)row_count * shape->output_sizes[matrix] * VALUE_BYTES;
            is_same = test_linear_card_expect_same(card, shape_case->wide_outputs[variant][case_index][matrix], 0,
                                                   shape_case->thin_outputs[matrix], 0, byte_count,
                                                   variant_names[variant]) &&
                      is_same;
            compared_bytes += byte_count;
        }
        (void)printf("card %" PRIu32 " (class %" PRIu32 "): %s (%s), %" PRIu32 " slices, %" PRIu32 " rows: %s, %"
                     PRIu64 " bytes of %" PRIu32 " variants against the thin launches", card->card_index,
                     arithmetic_class, shape->name, sizes, shape_case->slice_count, row_count,
                     is_same ? "every byte the same" : "DIFFERENT", compared_bytes, shape_case->variant_count);
        if (is_timed) {
            (void)printf("; TIME after a flush of the L2: thin (%" PRIu32 " launches) %.1f us", groups_of(row_count),
                         median_microseconds(card, &launches[shape_case->thin_first], groups_of(row_count)));
            for (uint32_t variant = 0; variant < shape_case->variant_count; variant++) {
                (void)printf(", %s %.1f us", variant_names[variant],
                             median_microseconds(card, &launches[shape_case->wide_first[variant] + case_index], 1));
            }
        }
        (void)printf("\n");
    }
}

/* A swept shape on a class whose slices add in the reference order at any count: the run of the default count (the
 * first) keeps its thin launches' outputs on the host, and every other count's thin launches must give those bytes;
 * each wide variant, held to its own count's thin launches by shape_check, then has them too. is_kept says whether
 * the default's outputs are on the host; without them there is nothing to hold a count to. */
static void count_check(struct test_linear_card *card, const struct shape_case *shape_case, uint32_t arithmetic_class,
                        uint32_t default_slice_count, bool *is_kept)
{
    static uint8_t default_outputs[(uint64_t)MOST_ROWS * MOST_SWEPT_OUTPUTS * VALUE_BYTES];
    const struct trunk_shape *const shape = shape_case->shape;
    const bool is_default = shape_case->slice_count == default_slice_count;
    uint64_t kept_bytes = 0;
    bool is_same = test_linear_expect(shape->output_sizes[0] + shape->output_sizes[1] <= MOST_SWEPT_OUTPUTS,
                                      "a swept shape's outputs fit the host's copy of the default count's");
    if (!is_same || (!is_default && !*is_kept)) {
        return;
    }
    for (uint32_t matrix = 0; matrix < shape_case->matrix_count; matrix++) {
        const uint64_t byte_count = (uint64_t)MOST_ROWS * shape->output_sizes[matrix] * VALUE_BYTES;
        is_same = (is_default ? test_linear_card_get(card, shape_case->thin_outputs[matrix], 0,
                                                     default_outputs + kept_bytes, byte_count)
                              : test_linear_card_expect_bytes(card, shape_case->thin_outputs[matrix], 0,
                                                              default_outputs + kept_bytes, byte_count,
                                                              "the thin launches give the default count's bytes")) &&
                  is_same;
        kept_bytes += byte_count;
    }
    if (is_default) {
        *is_kept = is_same;
        return;
    }
    (void)printf("card %" PRIu32 " (class %" PRIu32 "): %s, %" PRIu32 " slices: %s, %" PRIu64 " bytes of the thin "
                 "launches against those of the default %" PRIu32 " slices\n", card->card_index, arithmetic_class,
                 shape->name, shape_case->slice_count, is_same ? "every byte the same" : "DIFFERENT", kept_bytes,
                 default_slice_count);
}

/* Shown failing once: row CONTROL_ROW's values of the other sign, the rows quantized again and the wide launch of
 * 1,024 rows (as linear.c prepares it) run again; that row's outputs differ from the thin launches', and no other
 * byte. */
static void control_check(struct test_linear_card *card, const struct shape_case *shape_case)
{
    const struct trunk_shape *const shape = shape_case->shape;
    const uint64_t row_bytes = shape_case->row_width * VALUE_BYTES;
    const uint8_t *const image_values = test_linear_card_stored(card, shape_case->values);
    uint64_t differing_total = 0, differing_in_row = 0;
    bool is_only_the_row = true;
    if (image_values == NULL) {
        test_linear_fail("the rows' values do not lie in the image");
        return;
    }
    memcpy(row_values, image_values + CONTROL_ROW * row_bytes, row_bytes);
    for (uint32_t value_index = 0; value_index < shape_case->row_width; value_index++) {
        row_values[value_index] ^= BFLOAT16_SIGN_BIT;
    }
    if (!test_linear_card_put(card, shape_case->values, CONTROL_ROW * row_bytes, row_values, row_bytes) ||
        !test_linear_card_upload(card, shape_case->values, CONTROL_ROW * row_bytes, row_bytes) ||
        (!shape->is_bfloat16 && !test_linear_card_run(card, &launches[0], 1)) ||
        !test_linear_card_run(card, &launches[shape_case->wide_first[VARIANT_WIDE] + MOST_CASE], 1) ||
        !test_linear_card_fetch(card)) {
        return;
    }
    for (uint32_t matrix = 0; matrix < shape_case->matrix_count; matrix++) {
        const uint64_t output_row_bytes = shape->output_sizes[matrix] * VALUE_BYTES;
        uint64_t first_differing = 0;
        const uint64_t differing = test_linear_card_differing(
            card, shape_case->wide_outputs[VARIANT_WIDE][MOST_CASE][matrix], 0, shape_case->thin_outputs[matrix], 0,
            MOST_ROWS * output_row_bytes, &first_differing);
        const uint64_t differing_there = test_linear_card_differing(
            card, shape_case->wide_outputs[VARIANT_WIDE][MOST_CASE][matrix], CONTROL_ROW * output_row_bytes,
            shape_case->thin_outputs[matrix], CONTROL_ROW * output_row_bytes, output_row_bytes, &first_differing);
        is_only_the_row = is_only_the_row && differing == differing_there && differing_there != 0;
        differing_total += differing;
        differing_in_row += differing_there;
    }
    (void)test_linear_expect(is_only_the_row, "a row of values of the other sign changes that row's outputs, and "
                                              "nothing else");
    (void)printf("  control: row %u of the other sign: %" PRIu64 " bytes differ, %" PRIu64 " of them that row's\n",
                 CONTROL_ROW, differing_total, differing_in_row);
}

/* The default slice count of a shape on a class (linear.c): one for the two matrices of one launch, which
 * linear_launch_test shows the table gives. */
static bool default_slice_count_of(const struct trunk_shape *shape, uint32_t arithmetic_class,
                                   uint32_t *slice_count_out)
{
	const struct matrix_shape_record first_shape = { shape->output_sizes[0],
							 shape->input_size,
							 E4M3_BITS_PER_WEIGHT,
							 0 };
	return test_linear_expect_status(
		linear_default_slice_count(
			LINEAR_PROFILE_PRECISION, LINEAR_ORDER_FIXED,
			shape->is_bfloat16 ? LINEAR_FAMILY_TRUNK_BFLOAT16 :
					     LINEAR_FAMILY_TRUNK_QUANTIZED,
			&first_shape, arithmetic_class, slice_count_out),
		STATUS_OK, "the shape has a default slice count on the class");
}

/* Every shape on one card in one class, at its default slice count and, a swept shape, at every other count the
 * class runs (the stamped counts 1, 2, 4, ..): laid out, loaded, run and fetched, then held wide against thin and,
 * where the slices are ordered (class 80 and class 120's emulation, not class 120 native), each count against the
 * default. */
static void card_run(struct test_linear_card *card, uint32_t arithmetic_class, bool is_timed)
{
    static struct shape_case shape_case;
    const uint32_t most_slices = arithmetic_class == ARITHMETIC_CLASS_SM_120 ?
					 LINEAR_CLASS_120_EXACT_MOST_SLICES :
					 LINEAR_MAXIMUM_SLICES;
    const bool is_ordered = arithmetic_class != ARITHMETIC_CLASS_SM_120_NATIVE;
    for (uint32_t shape_index = 0; shape_index < COUNT_OF(trunk_shapes); shape_index++) {
        const struct trunk_shape *const shape = &trunk_shapes[shape_index];
	uint32_t slice_counts[LINEAR_SLICE_INSTANCES] = { 0 };
	uint32_t run_count = 1;
        bool is_default_kept = false;
        test_linear_context("card %" PRIu32 ", class %" PRIu32 ", %s", card->card_index, arithmetic_class,
                            shape->name);
	if ((shape->is_draft && arithmetic_class == ARITHMETIC_CLASS_SM_120) ||
	    !default_slice_count_of(shape, arithmetic_class,
				    &slice_counts[0])) {
		continue;
	}
	for (uint32_t slice_count = 1;
	     shape->is_swept && slice_count <= most_slices &&
	     run_count < LINEAR_SLICE_INSTANCES;
	     slice_count += slice_count) {
		if (slice_count != slice_counts[0]) {
			slice_counts[run_count++] = slice_count;
		}
	}
	for (uint32_t run = 0; run < run_count; run++) {
            memset(&shape_case, 0, sizeof(shape_case));
            shape_case.shape = shape;
            shape_case.slice_count = slice_counts[run];
            test_linear_context("card %" PRIu32 ", class %" PRIu32 ", %s, %" PRIu32 " slices", card->card_index,
                                arithmetic_class, shape->name, shape_case.slice_count);
            if (!test_linear_expect(shape_lay_out(card, &shape_case, shape_index), "the shape is laid out") ||
                !test_linear_card_load(card) || !test_linear_card_run(card, launches, shape_case.launch_count) ||
                !test_linear_card_fetch(card)) {
                continue;
            }
            (void)test_linear_card_expect_untouched(card, shape_case.values, "the rows' values are only read");
            (void)test_linear_card_expect_weights_untouched(card);
            shape_check(card, &shape_case, arithmetic_class, is_timed);
            if (is_ordered && shape->is_swept) {
                count_check(card, &shape_case, arithmetic_class, slice_counts[0], &is_default_kept);
            }
            control_check(card, &shape_case);
        }
    }
}

/* Every wide trunk instance and the interleaved one, asked of the card by name into the table: a model's card prepares
 * those its class's wide forms take alone (linear_wide_kernels_prepare), the test runs each. */
static bool every_wide_kernel_prepare(struct test_linear_card *card, uint32_t arithmetic_class)
{
	static const uint32_t stamped_counts[LINEAR_SLICE_INSTANCES] = { 1, 2,
									 4, 8,
									 16 };
	const char *const file_name =
		arithmetic_class == ARITHMETIC_CLASS_SM_120_NATIVE ?
			KERNEL_FILE_TRUNK_LINEAR_NATIVE :
			KERNEL_FILE_TRUNK_LINEAR;
	bool is_prepared = test_linear_expect_status(
		backend_cuda_value.kernel_prepare(
			card->card, file_name,
			KERNEL_TRUNK_LINEAR_QUANTIZED_INTERLEAVED,
			TRUNK_LINEAR_QUANTIZED_ARGUMENT_BYTES,
			&card->kernels.trunk_linear_quantized_interleaved),
		STATUS_OK, "the interleaved instance is prepared");
	for (uint32_t instance = 0;
	     instance < LINEAR_SLICE_INSTANCES && is_prepared; instance++) {
		char quantized[FUNCTION_NAME_CAPACITY],
			bfloat16[FUNCTION_NAME_CAPACITY];
		(void)snprintf(quantized, sizeof(quantized), "%s%" PRIu32,
			       KERNEL_TRUNK_LINEAR_QUANTIZED_WIDE_STEM,
			       stamped_counts[instance]);
		(void)snprintf(bfloat16, sizeof(bfloat16), "%s%" PRIu32,
			       KERNEL_TRUNK_LINEAR_BFLOAT16_WIDE_STEM,
			       stamped_counts[instance]);
		is_prepared =
			test_linear_expect_status(
				backend_cuda_value.kernel_prepare(
					card->card, file_name, quantized,
					TRUNK_LINEAR_QUANTIZED_ARGUMENT_BYTES,
					&card->kernels.trunk_linear_quantized_wide
						 [instance]),
				STATUS_OK,
				"a wide instance of quantized rows is prepared") &&
			test_linear_expect_status(
				backend_cuda_value.kernel_prepare(
					card->card, file_name, bfloat16,
					TRUNK_LINEAR_BFLOAT16_ARGUMENT_BYTES,
					&card->kernels.trunk_linear_bfloat16_wide
						 [instance]),
				STATUS_OK,
				"a wide instance of bfloat16 rows is prepared");
	}
    return is_prepared;
}

/* The kernels a pass of a class runs: the fixture prepared the card's own class; class 120 native takes the native
 * builds in their place. Then every wide trunk instance. */
static bool pass_kernels_prepare(struct test_linear_card *card, uint32_t arithmetic_class)
{
    test_linear_context("card %" PRIu32 ", class %" PRIu32, card->card_index, arithmetic_class);
    return (arithmetic_class != ARITHMETIC_CLASS_SM_120_NATIVE ||
	    test_linear_expect_status(
		    linear_kernels_prepare(&backend_cuda_value, card->card,
					   arithmetic_class, &card->kernels),
		    STATUS_OK,
		    "the kernels of class 120 native are prepared")) &&
	   every_wide_kernel_prepare(card, arithmetic_class);
}

int main(int argument_count, char **arguments)
{
    static struct test_linear_card card;
    uint32_t card_count = 0;
    const bool is_timed = argument_count == MOST_ARGUMENTS && strcmp(arguments[1], "--time") == 0;
    if (argument_count > MOST_ARGUMENTS || (argument_count == MOST_ARGUMENTS && !is_timed)) {
        (void)fprintf(stderr, "usage: trunk_wide_card_test [--time]\n");
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
		    memset(&description, 0, sizeof(description));
		    test_linear_context("card %" PRIu32, card_index);
		    if (!test_linear_expect_status(
				backend_cuda_value.card_describe(card_index,
								 &description),
				STATUS_OK, "the card is described")) {
			    continue;
		    }
		    /* the PRO in its own forms first (the default), then in class 120's emulation (--pro_emulation=1) */
		    const bool is_class_120 = description.arithmetic_class ==
					      ARITHMETIC_CLASS_SM_120;
		    const uint32_t classes[MOST_PASSES] = {
			    is_class_120 ? ARITHMETIC_CLASS_SM_120_NATIVE :
					   description.arithmetic_class,
			    ARITHMETIC_CLASS_SM_120
		    };
		    for (uint32_t pass = 0;
			 pass < (is_class_120 ? MOST_PASSES : 1u); pass++) {
			    if (test_linear_card_open(&card, card_index,
						      WORK_CAPACITY,
						      WEIGHTS_CAPACITY) &&
				pass_kernels_prepare(&card, classes[pass])) {
				    card_run(&card, classes[pass], is_timed);
				    (void)printf(
					    "card %" PRIu32 " (class %" PRIu32
					    "): digest of everything it computed 0x%016" PRIX64
					    "\n",
					    card_index, classes[pass],
					    card.digest);
			    }
			    test_linear_card_close(&card);
		    }
	    }
    }
    if (test_linear_failures() != 0) {
        (void)fprintf(stderr, "trunk_wide_card_test FAILED: %d failures\n", test_linear_failures());
        return TEST_FAILED;
    }
    (void)printf("trunk_wide_card_test passed on %" PRIu32 " cards\n", card_count);
    return TEST_PASSED;
}
