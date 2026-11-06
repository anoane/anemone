/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * linear_expert_card_test: on the real cards, the routed experts of a layer (the four launches of
 * linear_routed_experts_prepare: gate and up with the hidden step, the quantization of the hidden vectors,
 * down, the sum of a row's experts) against the host reference of the linear (linear_reference.h), as model.py
 * states an expert and the sum of a row's experts, on synthetic numbers (linear_card_fixture.h).
 *
 * It needs the cards free (`make card_tests`): a card without room for the arena FAILS the test, it is never skipped.
 * On every card, for 1, 2, 4, 6 and 8 rows whose experts are shared by no rows, by half (each row's first experts
 * are every row's) and by all (expert_cases); a small shape with 16 slices, one with uneven slices, and the model's;
 * and for 3 and 7 rows that share experts, whose jobs leave one column of a pair of columns without a row:
 *   - The rows are quantized on the card: their codes and operands are the reference's, byte for byte. Rows of
 *     plain values and of the edges, so that gate and up reach beyond the activation limit and hidden vectors hold
 *     zeros of both signs, subnormal codes and values at the limit.
 *   - A job is an expert with every row routed to it, the rows in ascending order or, for every other expert, in
 *     descending order: a row takes any column. Empty jobs filled with rubbish stand first and last in the table,
 *     and the entries of a job's columns that are not in use hold rubbish too: none of it may be read.
 *   - Hidden values. The fused kernel shows neither gate nor up, so a hidden value is held to
 *     test_linear_hidden_range: what the hidden step gives for every gate and up within the tolerance of their
 *     products, one last place wider. Of all hidden values of a card 99.7% must be bit-equal to the hidden step of
 *     the two products rounded once: gate, up and the step may each differ in one of a thousand.
 *   - The hidden codes, exponents and operands are the reference's quantization of the CARD'S OWN hidden values,
 *     byte for byte, for every pair.
 *   - The outputs of every pair are held as products of down with the card's own hidden codes: each within the
 *     tolerance of linear_reference.h, 99.9% of a card's bit-equal to the double result rounded once.
 *   - The sum of a row is the float32 sum of the card's own pair outputs in slot order: bit for bit.
 *   - ROW EXACTNESS: every row is launched again alone, as a batch of one row with a job table of its own experts
 *     (one row to a job, in column 0) and arrays of its own: its codes, hidden values, hidden codes and exponents,
 *     pair outputs and row sum must be the bits it has in company.
 *   - A pair computed elsewhere: in one case an expert has no job; the outputs of its pair are in the arena before
 *     the launches, must come back untouched and are summed with the others. They are values of the edges, far
 *     larger and far smaller than an expert's outputs, so that the float32 sum of that row rounds and its bits
 *     show the order of the slots.
 *   - Each check shown failing once, in one case: between launches one weight of gate of the first expert of row 0
 *     is changed ON THE CARD (to 6 of the other sign, where row 0 feeds its largest input). Of the hidden values
 *     only that expert's pairs at that weight's output may change, at least one does and lies outside what the
 *     reference allowed before; the codes, outputs and sums of every other pair and row keep their bits; and the
 *     whole chain is held again against the reference with the changed weight. Then one weight of down: no hidden
 *     value changes; of the pair outputs and row sums only that expert's at that weight's output; an output must
 *     change where its old bits are outside the tolerance of the changed product, and against the product before
 *     the comparison fails.
 *   - No launch writes what is not its own: the guards behind every range, the ranges a launch only reads, and the
 *     card's digest of the matrices are looked at in every case.
 * One line per chain says what was measured; a card's last line holds a digest of everything it computed.
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
#include "anemone/status.h"
#include "anemone/weight_format_seam.h"
#include "linear_card_fixture.h"
#include "linear_reference.h"
#include "weight_format_reference.h"

#define TEST_TIME_LIMIT_SECONDS 1800            /* the alarm ends the test here, whatever the driver does */
#define WORK_CAPACITY (16ull << 20u)
#define WEIGHTS_CAPACITY (128ull << 20u)        /* six experts of the model's shape, and room */
#define COUNT_OF(table) ((uint32_t)(sizeof(table) / sizeof((table)[0])))
#define ROWS LINEAR_MAXIMUM_ROWS
#define PLAIN_ROWS 4u                           /* the first rows hold plain values, the others the edges */
#define VALUES_PER_BLOCK TEST_LINEAR_VALUES_PER_BLOCK
#define VALUE_BYTES ((uint64_t)LINEAR_BFLOAT16_BYTES)
#define SUM_BYTES ((uint64_t)LINEAR_SUM_BYTES)
#define MAXIMUM_HIDDEN_SIZE 5120u
#define MAXIMUM_INNER_SIZE 2304u
#define MAXIMUM_PAIRS_PER_ROW 6u
#define MAXIMUM_PAIRS (ROWS * MAXIMUM_PAIRS_PER_ROW)
#define MAXIMUM_EXPERTS MAXIMUM_PAIRS           /* where no expert is shared */
#define MAXIMUM_JOBS (MAXIMUM_EXPERTS + 2u)     /* an empty job before the experts' and one behind */
#define NO_EXPERT UINT32_MAX
#define E2M1 WEIGHT_FORMAT_E2M1_SCALED
#define E2M1_BITS_PER_WEIGHT 4u
#define E2M1_SIGN_BIT 8u
#define E2M1_LARGEST_MAGNITUDE_CODE 7u          /* 6 */
#define ACTIVATION_LIMIT 10.0f                  /* the model's */
#define HIDDEN_SHARE_REQUIRED_PER_THOUSAND 997u /* 99.7% of the hidden values: see the top */
#define THOUSAND 1000u
#define DOWN_SCALE_CODE_BASE 118u
#define SCALE_CODE_SPREAD 3u
#define ROUTE_WEIGHT_STEPS 12u                  /* a route weight is 1 to 12 eighths */
#define ROUTE_WEIGHT_STEP 0.125f
#define RUBBISH_BYTE 0xA7
#define EXPERT_STREAM 0x0F000000ull             /* the streams of the hashes, far from those of the other tests */
#define STREAMS_PER_CASE 65536ull
#define MATRIX_STREAM 0x100ull
#define ROUTE_STREAM 0x8000ull
#define ELSEWHERE_STREAM 0x9000ull
#define CHANGE_STREAM 0xA000ull
#define ELSEWHERE_ROW 2u                        /* the pair computed elsewhere: this row at this slot */
#define ELSEWHERE_SLOT 1u
#define SET_LAUNCH_COUNT (1u + LINEAR_ROUTED_EXPERTS_LAUNCH_COUNT)
#define HALVES 2u                               /* half of a row's experts are shared; every other expert */
#define GATE_CHANGED_SET (ROWS + 1u)            /* the sets of a case: of all rows, of each row alone, */
#define DOWN_CHANGED_SET (ROWS + 2u)            /* and after each changed weight */
#define SET_COUNT (ROWS + 3u)

struct expert_shape {
    const char *name;
    uint32_t hidden_size;
    uint32_t inner_size;
    uint32_t gate_up_slice_count;
    uint32_t down_slice_count;
    uint32_t pairs_per_row;
    uint32_t gate_scale_code_base;              /* so that gate and up reach beyond the activation limit */
};

enum shape_index { SHAPE_SMALL, SHAPE_UNEVEN, SHAPE_MODEL };

static const struct expert_shape stamped_shapes[] = {
    [SHAPE_SMALL] = {"1024 to 128 and back, 6 experts to a row, 16 and 2 slices", 1024, 128, 16, 2, 6, 120},
    [SHAPE_UNEVEN] = {"256 to 192 and back, 3 experts to a row, 4 and 2 slices", 256, 192, 4, 2, 3, 121},
    [SHAPE_MODEL] = {"5120 to 2304 and back, 6 experts to a row, 8 and 8 slices", 5120, 2304, 8, 8, 6, 119},
};
static struct expert_shape shapes[COUNT_OF(stamped_shapes)];     /* those of the card under test: main */

enum sharing { SHARE_NONE, SHARE_HALF, SHARE_ALL };

static const char *const sharing_names[] = {
    [SHARE_NONE] = "no expert shared", [SHARE_HALF] = "half the experts shared", [SHARE_ALL] = "all experts shared"};

struct expert_case {
    enum shape_index shape;
    uint32_t row_count;
    enum sharing sharing;
    bool is_shown_failing;
    bool has_expert_elsewhere;
};

static const struct expert_case expert_cases[] = {
    {SHAPE_SMALL, 1, SHARE_NONE, false, false},  /* one row shares with nobody: the three routings are one */
    {SHAPE_SMALL, 2, SHARE_NONE, false, false},  {SHAPE_SMALL, 2, SHARE_HALF, false, false},
    {SHAPE_SMALL, 2, SHARE_ALL, false, false},   {SHAPE_SMALL, 4, SHARE_NONE, false, false},
    {SHAPE_SMALL, 4, SHARE_HALF, false, false},  {SHAPE_SMALL, 4, SHARE_ALL, false, false},
    {SHAPE_SMALL, 6, SHARE_NONE, false, false},  {SHAPE_SMALL, 6, SHARE_HALF, false, false},
    {SHAPE_SMALL, 6, SHARE_ALL, false, false},   {SHAPE_SMALL, 8, SHARE_NONE, false, false},
    {SHAPE_SMALL, 8, SHARE_HALF, false, false},  {SHAPE_SMALL, 8, SHARE_ALL, false, false},
    {SHAPE_UNEVEN, 8, SHARE_HALF, true, false},  {SHAPE_UNEVEN, 6, SHARE_NONE, false, true},
    {SHAPE_UNEVEN, 3, SHARE_ALL, false, false},  {SHAPE_UNEVEN, 7, SHARE_HALF, false, false},
    {SHAPE_MODEL, 4, SHARE_ALL, false, false},
};

enum expert_matrix { MATRIX_GATE, MATRIX_UP, MATRIX_DOWN, MATRIX_COUNT };

/* The arrays of one chain: of its rows (their codes and exponents, their sums) and of its pairs. */
enum set_range {
    SET_CODES,
    SET_EXPONENTS,
    SET_OPERANDS,
    SET_HIDDEN,
    SET_HIDDEN_CODES,
    SET_HIDDEN_EXPONENTS,
    SET_HIDDEN_OPERANDS,
    SET_PAIR_OUTPUT,
    SET_ROW_OUTPUT,
    SET_RANGE_COUNT
};

/* One chain for rows first_row .. first_row + row_count - 1: its job table, its arrays, and its launches: the
 * quantization of the rows, then the four of linear.h. */
struct expert_set {
	struct linear_range jobs;
	struct linear_range ranges[SET_RANGE_COUNT];
	struct kernel_launch launches[SET_LAUNCH_COUNT];
	uint32_t first_row;
	uint32_t row_count;
};

/* The pairs of one expert among some rows, in ascending row order. */
struct pair_list {
    uint32_t count;
    uint32_t rows[ROWS];
    uint32_t slots[ROWS];
};

struct card_totals {
    struct test_linear_comparison down;
    uint64_t hidden_count;
    uint64_t hidden_equal_count;
};

/* The case at hand: its rows as the reference quantizes them, its matrices, and what a card gave. */
static uint16_t row_values[ROWS][MAXIMUM_HIDDEN_SIZE];
static uint8_t row_codes[ROWS][MAXIMUM_HIDDEN_SIZE];           /* in the order of the values */
static uint8_t row_exponents[ROWS][MAXIMUM_HIDDEN_SIZE / VALUES_PER_BLOCK];
static struct linear_range matrices[MAXIMUM_EXPERTS][MATRIX_COUNT];
static uint16_t card_hidden[MAXIMUM_PAIRS * MAXIMUM_INNER_SIZE];
static uint16_t other_hidden[MAXIMUM_PAIRS * MAXIMUM_INNER_SIZE];
static uint16_t card_pair_output[MAXIMUM_PAIRS * MAXIMUM_HIDDEN_SIZE];
static uint16_t other_pair_output[MAXIMUM_PAIRS * MAXIMUM_HIDDEN_SIZE];
static uint32_t card_row_sums[ROWS * MAXIMUM_HIDDEN_SIZE];     /* the bits of float32 sums */
static uint32_t other_row_sums[ROWS * MAXIMUM_HIDDEN_SIZE];
static uint8_t pair_codes[MAXIMUM_PAIRS][MAXIMUM_INNER_SIZE];  /* the reference's, of the card's hidden values */
static uint8_t pair_exponents[MAXIMUM_PAIRS][MAXIMUM_INNER_SIZE / VALUES_PER_BLOCK];
static double gathered_values[ROWS * MAXIMUM_HIDDEN_SIZE];     /* what one expert's products are fed with */
static uint8_t gathered_exponents[ROWS * (MAXIMUM_HIDDEN_SIZE / VALUES_PER_BLOCK)];
static uint16_t gathered_outputs[ROWS * MAXIMUM_HIDDEN_SIZE];
static struct test_linear_product gate_products[ROWS * MAXIMUM_INNER_SIZE];
static struct test_linear_product up_products[ROWS * MAXIMUM_INNER_SIZE];
static struct test_linear_product down_products[ROWS * MAXIMUM_HIDDEN_SIZE];

static struct linear_range part_of(struct linear_range range,
				   uint64_t byte_offset, uint64_t byte_count)
{
	return (struct linear_range){ range.byte_offset + byte_offset,
				      byte_count };
}

/* ---- the routing ---- */

/* The slots of a row whose experts every row has. */
static uint32_t shared_slots(const struct expert_case *expert_case)
{
    const uint32_t pairs_per_row = shapes[expert_case->shape].pairs_per_row;
    return expert_case->sharing == SHARE_ALL ? pairs_per_row
           : expert_case->sharing == SHARE_HALF ? pairs_per_row / HALVES : 0;
}

/* The expert of a row at a slot: ascending in the slot, as the row sum adds them. */
static uint32_t expert_of(const struct expert_case *expert_case, uint32_t row, uint32_t slot)
{
    const uint32_t shared = shared_slots(expert_case);
    return slot < shared ? slot : shared + row * (shapes[expert_case->shape].pairs_per_row - shared) + slot - shared;
}

static uint32_t expert_count_of(const struct expert_case *expert_case)
{
    const uint32_t shared = shared_slots(expert_case);
    return shared + expert_case->row_count * (shapes[expert_case->shape].pairs_per_row - shared);
}

static float route_weight_of(uint64_t stream, uint32_t row, uint32_t slot)
{
    const uint64_t hash = test_linear_hash(stream + ROUTE_STREAM, (uint64_t)row * MAXIMUM_PAIRS_PER_ROW + slot);
    return (float)(1u + (uint32_t)(hash % ROUTE_WEIGHT_STEPS)) * ROUTE_WEIGHT_STEP;
}

/* The pairs of an expert among rows first_row .. first_row + row_count - 1. */
static struct pair_list pairs_of_expert(const struct expert_case *expert_case, uint32_t expert, uint32_t first_row,
                                        uint32_t row_count)
{
    struct pair_list list = {0, {0}, {0}};
    for (uint32_t row = first_row; row < first_row + row_count; row++) {
        for (uint32_t slot = 0; slot < shapes[expert_case->shape].pairs_per_row; slot++) {
            if (expert_of(expert_case, row, slot) == expert && list.count < ROWS) {
                list.rows[list.count] = row;
                list.slots[list.count] = slot;
                list.count++;
            }
        }
    }
    return list;
}

/* The job table of a set: an empty job, a job for every expert its rows are routed to (none for the expert that
 * is computed elsewhere, whose place holds an empty job), an empty job. The count of its entries. */
static uint32_t make_jobs(const struct expert_case *expert_case,
			  const struct expert_set *set, uint64_t stream,
			  uint32_t expert_elsewhere,
			  struct expert_job_record *jobs)
{
    const uint32_t pairs_per_row = shapes[expert_case->shape].pairs_per_row;
    unsigned char *const table_bytes = (unsigned char *)jobs;
    uint32_t job_count = 1;
    memset(table_bytes, RUBBISH_BYTE, MAXIMUM_JOBS * sizeof(*jobs));    /* what an empty job holds is never read */
    jobs[0].row_count = 0;
    for (uint32_t expert = 0; expert < expert_count_of(expert_case); expert++) {
        const struct pair_list list = pairs_of_expert(expert_case, expert, set->first_row, set->row_count);
	struct expert_job_record *const job = &jobs[job_count];
	if (list.count == 0) {
            continue;
        }
        job_count++;
        job->row_count = 0;
        if (expert == expert_elsewhere) {
            continue;
        }
        /* the entries of the columns not in use keep the rubbish: linear_records.h says they are not read */
        job->gate_offset = matrices[expert][MATRIX_GATE].byte_offset;
        job->up_offset = matrices[expert][MATRIX_UP].byte_offset;
        job->down_offset = matrices[expert][MATRIX_DOWN].byte_offset;
        job->row_count = list.count;
        job->expert = 0;
        for (uint32_t column = 0; column < list.count; column++) {
            /* every other expert takes its rows in descending order: a row takes any column */
            const uint32_t place = expert % HALVES == 0 ? column : list.count - 1 - column;
            job->pair_index[column] = (list.rows[place] - set->first_row) * pairs_per_row + list.slots[place];
            job->route_weight[column] = route_weight_of(stream, list.rows[place], list.slots[place]);
        }
    }
    jobs[job_count].row_count = 0;
    return job_count + 1;
}

/* ---- a chain ---- */

/* The bytes of one unit of an array of a set, and its units in one row: a row's, or those of its pairs. */
static uint64_t unit_bytes(const struct expert_shape *shape, enum set_range range)
{
    switch (range) {
    case SET_CODES:
        return shape->hidden_size;
    case SET_EXPONENTS:
        return shape->hidden_size / VALUES_PER_BLOCK;
    case SET_OPERANDS:
        return shape->hidden_size * VALUE_BYTES;
    case SET_HIDDEN:
    case SET_HIDDEN_OPERANDS:
        return shape->inner_size * VALUE_BYTES;
    case SET_HIDDEN_CODES:
        return shape->inner_size;
    case SET_HIDDEN_EXPONENTS:
        return shape->inner_size / VALUES_PER_BLOCK;
    case SET_PAIR_OUTPUT:
        return shape->hidden_size * VALUE_BYTES;
    case SET_ROW_OUTPUT:
    case SET_RANGE_COUNT:
    default:
        return shape->hidden_size * SUM_BYTES;
    }
}

static uint32_t units_per_row(const struct expert_shape *shape, enum set_range range)
{
    return range == SET_CODES || range == SET_EXPONENTS || range == SET_OPERANDS || range == SET_ROW_OUTPUT
               ? 1 : shape->pairs_per_row;
}

/* Sets the arrays and the job table of a chain aside and prepares its launches; the outputs of a pair computed
 * elsewhere are laid into the image. values holds the set's rows. */
static bool set_prepare(struct test_linear_card *card,
			const struct expert_case *expert_case, uint64_t stream,
			uint32_t expert_elsewhere, struct linear_range values,
			uint32_t first_row, uint32_t row_count,
			struct expert_set *set)
{
	static struct expert_job_record jobs[MAXIMUM_JOBS];
	static uint16_t elsewhere_values[MAXIMUM_HIDDEN_SIZE];
	const struct expert_shape *const shape = &shapes[expert_case->shape];
	set->first_row = first_row;
	set->row_count = row_count;
	const uint32_t job_count =
		make_jobs(expert_case, set, stream, expert_elsewhere, jobs);
	const struct pair_list elsewhere = pairs_of_expert(
		expert_case, expert_elsewhere, first_row, row_count);
	bool is_ready = true;
	set->jobs = test_linear_card_work(card, (uint64_t)job_count *
							sizeof(jobs[0]));
	for (uint32_t range = 0; range < SET_RANGE_COUNT; range++) {
		set->ranges[range] = test_linear_card_work(
			card,
			(uint64_t)row_count *
				units_per_row(shape, (enum set_range)range) *
				unit_bytes(shape, (enum set_range)range));
	}
    is_ready = test_linear_card_has_room(card) &&
               test_linear_card_put(card, set->jobs, 0, jobs, (uint64_t)job_count * sizeof(jobs[0]));
    for (uint32_t index = 0; is_ready && index < elsewhere.count; index++) {
        const uint64_t pair = (uint64_t)(elsewhere.rows[index] - first_row) * shape->pairs_per_row +
                              elsewhere.slots[index];
        test_linear_values(elsewhere_values, shape->hidden_size,
                           stream + ELSEWHERE_STREAM + (uint64_t)elsewhere.rows[index] * MAXIMUM_PAIRS_PER_ROW +
                               elsewhere.slots[index],
                           TEST_LINEAR_VALUES_EDGES);
        is_ready = test_linear_card_put(card, set->ranges[SET_PAIR_OUTPUT],
                                        pair * unit_bytes(shape, SET_PAIR_OUTPUT), elsewhere_values,
                                        unit_bytes(shape, SET_PAIR_OUTPUT));
    }
    const struct linear_quantize_description quantize = {
	    .values = values,
	    .codes = set->ranges[SET_CODES],
	    .exponents = set->ranges[SET_EXPONENTS],
	    .operands = set->ranges[SET_OPERANDS],
	    .width = shape->hidden_size,
	    .row_count = row_count
    };
    const struct linear_routed_experts_description experts = {
	    .jobs = set->jobs,
	    .expert_weights = test_linear_card_all_weights(card),
	    .codes = set->ranges[SET_CODES],
	    .exponents = set->ranges[SET_EXPONENTS],
	    .operands = set->ranges[SET_OPERANDS],
	    .hidden = set->ranges[SET_HIDDEN],
	    .hidden_codes = set->ranges[SET_HIDDEN_CODES],
	    .hidden_exponents = set->ranges[SET_HIDDEN_EXPONENTS],
	    .hidden_operands = set->ranges[SET_HIDDEN_OPERANDS],
	    .pair_output = set->ranges[SET_PAIR_OUTPUT],
	    .row_output = set->ranges[SET_ROW_OUTPUT],
	    .gate_up_shape = { shape->inner_size, shape->hidden_size,
			       E2M1_BITS_PER_WEIGHT, 0 },
	    .down_shape = { shape->hidden_size, shape->inner_size,
			    E2M1_BITS_PER_WEIGHT, 0 },
	    .weight_format = E2M1,
	    .job_count = job_count,
	    .row_count = row_count,
	    .pairs_per_row = shape->pairs_per_row,
	    .gate_up_slice_count = shape->gate_up_slice_count,
	    .down_slice_count = shape->down_slice_count,
	    .activation_limit = ACTIVATION_LIMIT
    };
    return is_ready &&
	   test_linear_expect_status(
		   linear_quantize_prepare(&quantize, card->arena_byte_count,
					   &card->kernels, &set->launches[0]),
		   STATUS_OK,
		   "the quantization of a chain's rows is prepared") &&
	   test_linear_expect_status(linear_routed_experts_prepare(
					     &experts, card->arena_byte_count,
					     &card->kernels, &set->launches[1]),
				     STATUS_OK,
				     "the routed experts are prepared");
}

/* What the products of an expert are fed with, into gathered_values and gathered_exponents: for gate and up the
 * rows of its pairs (width hidden_size), for down the reference's codes of the card's hidden vectors of its pairs. */
static bool gather(const struct expert_case *expert_case, const struct pair_list *list, bool is_for_down)
{
    const struct expert_shape *const shape = &shapes[expert_case->shape];
    const uint32_t width = is_for_down ? shape->inner_size : shape->hidden_size;
    const uint32_t exponents_per_vector = width / VALUES_PER_BLOCK;
    bool is_gathered = true;
    for (uint32_t index = 0; index < list->count; index++) {
        const uint32_t pair = list->rows[index] * shape->pairs_per_row + list->slots[index];
        const uint8_t *const codes = is_for_down ? pair_codes[pair] : row_codes[list->rows[index]];
        const uint8_t *const exponents = is_for_down ? pair_exponents[pair] : row_exponents[list->rows[index]];
	is_gathered = is_gathered &&
		      test_linear_fed_values_of_codes(
			      codes, width,
			      gathered_values + (size_t)index * width) ==
			      STATUS_OK;
	memcpy(gathered_exponents + (size_t)index * exponents_per_vector, exponents, exponents_per_vector);
    }
    return test_linear_expect(is_gathered, "the reference decodes what an expert is fed with");
}

/* Outputs first_output .. of one matrix of an expert for what gather left: products_out [pairs][output_count]. */
static bool expert_products(const struct test_linear_card *card, const struct expert_case *expert_case,
                            uint32_t expert, enum expert_matrix matrix, const struct pair_list *list,
                            uint32_t first_output, uint32_t output_count, struct test_linear_product *products_out)
{
    const struct expert_shape *const shape = &shapes[expert_case->shape];
    const bool is_down = matrix == MATRIX_DOWN;
    const struct matrix_shape_record matrix_shape = {
	    is_down ? shape->hidden_size : shape->inner_size,
	    is_down ? shape->inner_size : shape->hidden_size,
	    E2M1_BITS_PER_WEIGHT, 0
    };
    return test_linear_expect_status(
	    test_linear_products(
		    E2M1, &matrix_shape,
		    test_linear_card_stored(card, matrices[expert][matrix]),
		    matrices[expert][matrix].byte_count, first_output,
		    output_count, gathered_values, gathered_exponents,
		    list->count, 1, products_out),
	    STATUS_OK, "the reference computes the products of an expert");
}

/* A chain of all the rows of a case against the reference: see the top. Leaves the card's hidden values, pair
 * outputs and row sums of the set in card_hidden, card_pair_output and card_row_sums, and the reference's codes of
 * the hidden values in pair_codes and pair_exponents. */
static void check_set(struct test_linear_card *card, const struct expert_case *expert_case,
                      const struct expert_set *set, uint32_t expert_elsewhere, const char *what,
                      struct card_totals *totals)
{
    const struct expert_shape *const shape = &shapes[expert_case->shape];
    const uint32_t hidden_size = shape->hidden_size;
    const uint32_t inner_size = shape->inner_size;
    const uint32_t pair_count = expert_case->row_count * shape->pairs_per_row;
    struct test_linear_comparison down = {0};
    uint64_t hidden_count = 0;
    uint64_t hidden_equal_count = 0;
    uint64_t hidden_outside_count = 0;
    uint64_t wrong_sum_count = 0;
    bool is_same = true;

    for (uint32_t row = 0; is_same && row < expert_case->row_count; row++) {
        is_same = test_linear_card_expect_codes(card, set->ranges[SET_CODES], (uint64_t)row * hidden_size,
                                                row_codes[row], hidden_size, "a row quantized by the card") &&
                  test_linear_card_expect_bytes(card, set->ranges[SET_EXPONENTS],
                                                (uint64_t)row * (hidden_size / VALUES_PER_BLOCK), row_exponents[row],
                                                hidden_size / VALUES_PER_BLOCK, "the exponents of a row") &&
                  test_linear_card_expect_operands(card, set->ranges[SET_OPERANDS],
                                                   (uint64_t)row * unit_bytes(shape, SET_OPERANDS), row_codes[row],
                                                   row_exponents[row], hidden_size, "the operands of a row");
    }
    if (!test_linear_card_get(card, set->ranges[SET_HIDDEN], 0, card_hidden,
                              pair_count * unit_bytes(shape, SET_HIDDEN)) ||
        !test_linear_card_get(card, set->ranges[SET_PAIR_OUTPUT], 0, card_pair_output,
                              pair_count * unit_bytes(shape, SET_PAIR_OUTPUT)) ||
        !test_linear_card_get(card, set->ranges[SET_ROW_OUTPUT], 0, card_row_sums,
                              expert_case->row_count * unit_bytes(shape, SET_ROW_OUTPUT))) {
        return;
    }
    /* the hidden codes of every pair: the reference's quantization of the card's own hidden values */
    is_same = true;
    for (uint32_t pair = 0; is_same && pair < pair_count; pair++) {
	    is_same =
		    test_linear_expect_status(
			    test_linear_quantize(
				    card_hidden + (size_t)pair * inner_size,
				    inner_size, pair_codes[pair],
				    pair_exponents[pair]),
			    STATUS_OK,
			    "the reference quantizes a hidden vector of the card") &&
		    test_linear_card_expect_codes(
			    card, set->ranges[SET_HIDDEN_CODES],
			    (uint64_t)pair * inner_size, pair_codes[pair],
			    inner_size, "the codes of a hidden vector") &&
		    test_linear_card_expect_bytes(
			    card, set->ranges[SET_HIDDEN_EXPONENTS],
			    (uint64_t)pair * (inner_size / VALUES_PER_BLOCK),
			    pair_exponents[pair], inner_size / VALUES_PER_BLOCK,
			    "the exponents of a hidden vector") &&
		    test_linear_card_expect_operands(
			    card, set->ranges[SET_HIDDEN_OPERANDS],
			    (uint64_t)pair *
				    unit_bytes(shape, SET_HIDDEN_OPERANDS),
			    pair_codes[pair], pair_exponents[pair], inner_size,
			    "the operands of a hidden vector");
    }
    for (uint32_t expert = 0; is_same && expert < expert_count_of(expert_case); expert++) {
        const struct pair_list list = pairs_of_expert(expert_case, expert, 0, expert_case->row_count);
        if (expert == expert_elsewhere) {
            for (uint32_t index = 0; index < list.count; index++) {
                const uint64_t pair = (uint64_t)list.rows[index] * shape->pairs_per_row + list.slots[index];
                (void)test_linear_card_expect_untouched(
                    card, part_of(set->ranges[SET_HIDDEN], pair * unit_bytes(shape, SET_HIDDEN),
                                  unit_bytes(shape, SET_HIDDEN)), "the hidden values of a pair that has no job");
                (void)test_linear_card_expect_untouched(
                    card, part_of(set->ranges[SET_PAIR_OUTPUT], pair * unit_bytes(shape, SET_PAIR_OUTPUT),
                                  unit_bytes(shape, SET_PAIR_OUTPUT)), "the outputs of a pair computed elsewhere");
            }
            continue;
        }
        if (!gather(expert_case, &list, false) ||
            !expert_products(card, expert_case, expert, MATRIX_GATE, &list, 0, inner_size, gate_products) ||
            !expert_products(card, expert_case, expert, MATRIX_UP, &list, 0, inner_size, up_products)) {
            return;
        }
        for (size_t index = 0; index < list.count; index++) {
            const size_t pair = (size_t)list.rows[index] * shape->pairs_per_row + list.slots[index];
            const double route_weight =
                (double)route_weight_of(EXPERT_STREAM, list.rows[index], list.slots[index]);
            for (size_t output_index = 0; output_index < inner_size; output_index++) {
                struct test_linear_hidden_range range = {0, 0, 0, 0};
                const uint16_t hidden = card_hidden[pair * inner_size + output_index];
                test_linear_hidden_range(&gate_products[index * inner_size + output_index],
                                         &up_products[index * inner_size + output_index], hidden_size,
                                         shape->gate_up_slice_count, (double)ACTIVATION_LIMIT, route_weight, &range);
                const int32_t ordinal = test_linear_bfloat16_ordinal(hidden);
                const bool is_outside = !test_linear_bfloat16_is_finite(hidden) || ordinal < range.lowest_ordinal ||
                                        ordinal > range.highest_ordinal;
                if (is_outside && hidden_outside_count == 0) {
                    test_linear_fail("%s: hidden value %zu of expert %" PRIu32 " for row %" PRIu32 " is 0x%04X: the "
                                     "reference has 0x%04X and allows the ordinals %" PRId32 " to %" PRId32, what,
                                     output_index, expert, list.rows[index], (unsigned int)hidden,
                                     (unsigned int)range.central_bits, range.lowest_ordinal, range.highest_ordinal);
                }
                hidden_count++;
                hidden_equal_count += hidden == range.central_bits ? 1u : 0u;
                hidden_outside_count += is_outside ? 1u : 0u;
            }
            memcpy(gathered_outputs + index * hidden_size, card_pair_output + pair * hidden_size,
                   hidden_size * sizeof(gathered_outputs[0]));
        }
        if (!gather(expert_case, &list, true) ||
            !expert_products(card, expert_case, expert, MATRIX_DOWN, &list, 0, hidden_size, down_products)) {
            return;
        }
        const uint64_t outside_before = down.outside_count;
        const uint64_t first_outside = test_linear_compare(gathered_outputs, down_products,
                                                           (uint64_t)list.count * hidden_size, inner_size,
                                                           shape->down_slice_count, &down);
        if (first_outside != TEST_LINEAR_NO_OUTPUT && outside_before == 0) {
            test_linear_fail("%s: output %" PRIu64 " of the pairs of expert %" PRIu32 " is 0x%04X, outside the "
                             "tolerance of the reference's %.9g", what, first_outside, expert,
                             (unsigned int)gathered_outputs[first_outside], down_products[first_outside].sum);
        }
    }
    for (size_t row = 0; row < expert_case->row_count; row++) {
        for (uint32_t column = 0; column < hidden_size; column++) {
            const float sum = test_linear_row_sum(card_pair_output + row * shape->pairs_per_row * hidden_size,
                                                  shape->pairs_per_row, hidden_size, column);
            uint32_t sum_bits = 0;
            memcpy(&sum_bits, &sum, sizeof(sum_bits));
            wrong_sum_count += sum_bits != card_row_sums[row * hidden_size + column] ? 1u : 0u;
        }
    }
    (void)test_linear_expect(hidden_outside_count == 0 && down.outside_count == 0,
                             "every hidden value and every output of a pair lies where the reference allows it");
    (void)test_linear_expect(wrong_sum_count == 0, "a row's sum is the float32 sum of its pair outputs in slot order");
    (void)printf("  %s: %" PRIu64 " hidden values, %.4f%% bit-equal, %" PRIu64 " outside their range; %" PRIu64
                 " outputs of pairs, %.4f%% bit-equal, the largest difference %" PRIu32 " last places, %" PRIu64
                 " outside the tolerance; %" PRIu64 " row sums differ\n", what, hidden_count,
                 test_linear_percent(hidden_equal_count, hidden_count), hidden_outside_count, down.output_count,
                 test_linear_percent(down.equal_count, down.output_count), down.largest_distance, down.outside_count,
                 wrong_sum_count);
    if (totals != NULL) {
        totals->hidden_count += hidden_count;
        totals->hidden_equal_count += hidden_equal_count;
        totals->down.output_count += down.output_count;
        totals->down.equal_count += down.equal_count;
    }
}

/* ---- shown failing ---- */

/* Whether a pair is one of an expert's. */
static bool pair_is_of(const struct expert_case *expert_case, uint32_t pair, uint32_t expert)
{
    const uint32_t pairs_per_row = shapes[expert_case->shape].pairs_per_row;
    return expert_of(expert_case, pair / pairs_per_row, pair % pairs_per_row) == expert;
}

/* The value of the vector `codes` with exponents `exponents` that is largest in magnitude: its index. */
static uint32_t largest_fed_input(const uint8_t *codes, const uint8_t *exponents, uint32_t width)
{
    uint32_t largest_input = 0;
    double largest = -1.0;
    for (uint32_t input_index = 0; input_index < width; input_index++) {
        const double value = test_weight_format_e4m3_value(codes[input_index]) *
                             test_weight_format_e8m0_value(exponents[input_index / VALUES_PER_BLOCK]);
        const double magnitude = value < 0.0 ? -value : value;
        if (magnitude > largest) {
            largest = magnitude;
            largest_input = input_index;
        }
    }
    return largest_input;
}

/* Sets one weight of a matrix of an expert to 6 of the other sign, in the image and on the card. */
static bool change_a_weight(struct test_linear_card *card, const struct expert_case *expert_case, uint32_t expert,
                            enum expert_matrix matrix, uint32_t output_index, uint32_t input_index)
{
    const struct expert_shape *const shape = &shapes[expert_case->shape];
    const bool is_down = matrix == MATRIX_DOWN;
    const struct matrix_shape_record matrix_shape = {
	    is_down ? shape->hidden_size : shape->inner_size,
	    is_down ? shape->inner_size : shape->hidden_size,
	    E2M1_BITS_PER_WEIGHT, 0
    };
    const struct linear_range stored = matrices[expert][matrix];
    struct test_weight_format_weight weight = {0};
    uint64_t byte_offset = 0;
    return test_linear_expect_status(
		   test_weight_format_read_weight(
			   E2M1, &matrix_shape,
			   test_linear_card_stored(card, stored),
			   stored.byte_count, output_index, input_index,
			   &weight),
		   STATUS_OK,
		   "the reference reads the weight that is to change") &&
	   test_linear_card_set_weight(
		   card, E2M1, &matrix_shape, stored, output_index, input_index,
		   ((weight.code ^ E2M1_SIGN_BIT) & E2M1_SIGN_BIT) |
			   E2M1_LARGEST_MAGNITUDE_CODE,
		   &byte_offset) &&
	   test_linear_card_upload(card, stored, byte_offset, 1);
}

/* Holds every unit of an array that is not one of an expert's (a pair of its, or a row routed to it) to be the
 * same in two sets. */
static void expect_others_same(const struct test_linear_card *card, const struct expert_case *expert_case,
                               const struct expert_set *first, const struct expert_set *second,
                               enum set_range range, uint32_t expert, const char *what)
{
    const struct expert_shape *const shape = &shapes[expert_case->shape];
    const uint32_t per_row = units_per_row(shape, range);
    const uint64_t bytes = unit_bytes(shape, range);
    for (uint32_t unit = 0; unit < expert_case->row_count * per_row; unit++) {
        const bool is_of_expert =
            per_row == 1 ? pairs_of_expert(expert_case, expert, unit, 1).count != 0
                         : pair_is_of(expert_case, unit, expert);
        if (!is_of_expert && !test_linear_card_expect_same(card, first->ranges[range], unit * bytes,
                                                           second->ranges[range], unit * bytes, bytes, what)) {
            return;
        }
    }
}

/* One weight of gate, then one of down, of the first expert of row 0, changed on the card: see the top. before is
 * the chain that ran on the weights as they were; it was checked last, so the arrays check_set leaves are its. */
static void show_changed_weights(struct test_linear_card *card, const struct expert_case *expert_case,
                                 const struct expert_set *before, const struct expert_set *gate_changed,
                                 const struct expert_set *down_changed)
{
    static struct test_linear_hidden_range ranges_before[ROWS];
    static struct test_linear_product products_before[ROWS];
    static struct test_linear_product products_after[ROWS];
    const struct expert_shape *const shape = &shapes[expert_case->shape];
    const uint32_t hidden_size = shape->hidden_size;
    const uint32_t inner_size = shape->inner_size;
    const uint32_t pair_count = expert_case->row_count * shape->pairs_per_row;
    const uint32_t expert = expert_of(expert_case, 0, 0);
    const struct pair_list list = pairs_of_expert(expert_case, expert, 0, expert_case->row_count);
    const uint32_t gate_output = (uint32_t)(test_linear_hash(CHANGE_STREAM, inner_size) % inner_size);
    const uint32_t down_output = (uint32_t)(test_linear_hash(CHANGE_STREAM, hidden_size) % hidden_size);
    uint64_t inside_count = 0;
    uint64_t outside_count = 0;
    uint32_t shown_failing_count = 0;
    uint32_t must_change_count = 0;
    uint32_t wrong_count = 0;

    /* a weight of gate: what the reference allowed the hidden values at its output before */
    if (!gather(expert_case, &list, false) ||
        !expert_products(card, expert_case, expert, MATRIX_GATE, &list, gate_output, 1, gate_products) ||
        !expert_products(card, expert_case, expert, MATRIX_UP, &list, gate_output, 1, up_products)) {
        return;
    }
    for (uint32_t index = 0; index < list.count; index++) {
        test_linear_hidden_range(&gate_products[index], &up_products[index], hidden_size,
                                 shape->gate_up_slice_count, (double)ACTIVATION_LIMIT,
                                 (double)route_weight_of(EXPERT_STREAM, list.rows[index], list.slots[index]),
                                 &ranges_before[index]);
    }
    memcpy(other_hidden, card_hidden, (size_t)pair_count * inner_size * sizeof(card_hidden[0]));
    if (!change_a_weight(card, expert_case, expert, MATRIX_GATE, gate_output,
                         largest_fed_input(row_codes[0], row_exponents[0], hidden_size)) ||
        !test_linear_card_run(card, gate_changed->launches, SET_LAUNCH_COUNT) || !test_linear_card_fetch(card)) {
        return;
    }
    check_set(card, expert_case, gate_changed, NO_EXPERT, "after a weight of gate changed on the card", NULL);
    for (uint32_t pair = 0; pair < pair_count; pair++) {
        for (uint32_t column = 0; column < inner_size; column++) {
            const bool differs = card_hidden[(size_t)pair * inner_size + column] !=
                                 other_hidden[(size_t)pair * inner_size + column];
            const bool may_differ = column == gate_output && pair_is_of(expert_case, pair, expert);
            inside_count += differs && may_differ ? 1u : 0u;
            outside_count += differs && !may_differ ? 1u : 0u;
        }
    }
    for (uint32_t index = 0; index < list.count; index++) {
        const size_t pair = (size_t)list.rows[index] * shape->pairs_per_row + list.slots[index];
        const int32_t ordinal = test_linear_bfloat16_ordinal(card_hidden[pair * inner_size + gate_output]);
        shown_failing_count += ordinal < ranges_before[index].lowest_ordinal ||
                               ordinal > ranges_before[index].highest_ordinal ? 1u : 0u;
    }
    (void)test_linear_expect(outside_count == 0 && inside_count != 0 && shown_failing_count != 0,
                             "a changed weight of gate changes hidden values of its expert at its output, no other, "
                             "and by more than the reference allowed before");
    expect_others_same(card, expert_case, before, gate_changed, SET_HIDDEN_CODES, expert, "hidden codes of others");
    expect_others_same(card, expert_case, before, gate_changed, SET_HIDDEN_EXPONENTS, expert, "exponents of others");
    expect_others_same(card, expert_case, before, gate_changed, SET_HIDDEN_OPERANDS, expert, "operands of others");
    expect_others_same(card, expert_case, before, gate_changed, SET_PAIR_OUTPUT, expert, "outputs of other pairs");
    expect_others_same(card, expert_case, before, gate_changed, SET_ROW_OUTPUT, expert, "sums of other rows");
    (void)printf("  %" PRIu64 " hidden values of the expert's %" PRIu32 " pairs changed, %" PRIu32 " lie outside "
                 "what the reference allowed before; %" PRIu64 " other hidden values changed\n", inside_count,
                 list.count, shown_failing_count, outside_count);

    /* a weight of down, where the card's hidden vector of row 0 at this expert is largest */
    const uint32_t first_pair = list.rows[0] * shape->pairs_per_row + list.slots[0];
    memcpy(other_pair_output, card_pair_output, (size_t)pair_count * hidden_size * sizeof(card_pair_output[0]));
    memcpy(other_row_sums, card_row_sums, (size_t)expert_case->row_count * hidden_size * sizeof(card_row_sums[0]));
    if (!gather(expert_case, &list, true) ||
        !expert_products(card, expert_case, expert, MATRIX_DOWN, &list, down_output, 1, products_before) ||
        !change_a_weight(card, expert_case, expert, MATRIX_DOWN, down_output,
                         largest_fed_input(pair_codes[first_pair], pair_exponents[first_pair], inner_size)) ||
        !test_linear_card_run(card, down_changed->launches, SET_LAUNCH_COUNT) || !test_linear_card_fetch(card)) {
        return;
    }
    check_set(card, expert_case, down_changed, NO_EXPERT, "after a weight of down changed on the card", NULL);
    if (!gather(expert_case, &list, true) ||
        !expert_products(card, expert_case, expert, MATRIX_DOWN, &list, down_output, 1, products_after)) {
        return;
    }
    for (uint32_t range = SET_HIDDEN; range <= SET_HIDDEN_OPERANDS; range++) {
        (void)test_linear_card_expect_same(card, gate_changed->ranges[range], 0, down_changed->ranges[range], 0,
                                           pair_count * unit_bytes(shape, (enum set_range)range),
                                           "a changed weight of down changes nothing of the hidden vectors");
    }
    inside_count = 0;
    outside_count = 0;
    shown_failing_count = 0;
    for (size_t index = 0; index < (size_t)pair_count * hidden_size; index++) {
        const bool differs = card_pair_output[index] != other_pair_output[index];
        const bool may_differ = index % hidden_size == down_output &&
                                pair_is_of(expert_case, (uint32_t)(index / hidden_size), expert);
        inside_count += differs && may_differ ? 1u : 0u;
        outside_count += differs && !may_differ ? 1u : 0u;
    }
    for (size_t index = 0; index < (size_t)expert_case->row_count * hidden_size; index++) {
        const bool may_differ = index % hidden_size == down_output &&
                                pairs_of_expert(expert_case, expert, (uint32_t)(index / hidden_size), 1).count != 0;
        outside_count += card_row_sums[index] != other_row_sums[index] && !may_differ ? 1u : 0u;
    }
    for (uint32_t index = 0; index < list.count; index++) {
        const size_t at_weight = ((size_t)list.rows[index] * shape->pairs_per_row + list.slots[index]) * hidden_size +
                                 down_output;
        /* the bits the output had are no answer to the changed product: it has to change */
        const bool must_change = !test_linear_output_is_within_tolerance(
            other_pair_output[at_weight], &products_after[index], inner_size, shape->down_slice_count);
        must_change_count += must_change ? 1u : 0u;
        wrong_count += must_change && card_pair_output[at_weight] == other_pair_output[at_weight] ? 1u : 0u;
        shown_failing_count += !test_linear_output_is_within_tolerance(
            card_pair_output[at_weight], &products_before[index], inner_size, shape->down_slice_count) ? 1u : 0u;
    }
    (void)test_linear_expect(outside_count == 0 && wrong_count == 0 && must_change_count != 0 &&
                                 shown_failing_count != 0,
                             "a changed weight of down changes the outputs and sums of its expert at its output, no "
                             "other, where it must, and fails the comparison with the product before");
    (void)printf("  %" PRIu64 " outputs of the expert's %" PRIu32 " pairs changed, %" PRIu32 " had to, %" PRIu32
                 " fail the comparison with the product before; %" PRIu64 " other outputs or sums changed\n",
                 inside_count, list.count, must_change_count, shown_failing_count, outside_count);
}

/* ---- a case ---- */

static void test_expert_case(struct test_linear_card *card, const struct expert_case *expert_case,
                             struct card_totals *totals)
{
    static struct expert_set sets[SET_COUNT];
    static struct kernel_launch launches[(ROWS + 1) * SET_LAUNCH_COUNT];
    const struct expert_shape *const shape = &shapes[expert_case->shape];
    const uint32_t row_count = expert_case->row_count;
    const uint32_t expert_count = expert_count_of(expert_case);
    const uint64_t row_value_bytes = shape->hidden_size * VALUE_BYTES;
    const uint32_t expert_elsewhere =
        expert_case->has_expert_elsewhere ? expert_of(expert_case, ELSEWHERE_ROW, ELSEWHERE_SLOT) : NO_EXPERT;
    const uint32_t alone_count = row_count > 1 ? row_count : 0;
    struct expert_set *const gate_changed = &sets[GATE_CHANGED_SET];
    struct expert_set *const down_changed = &sets[DOWN_CHANGED_SET];
    uint32_t launch_count = 0;
    bool is_ready = shape->hidden_size <= MAXIMUM_HIDDEN_SIZE && shape->inner_size <= MAXIMUM_INNER_SIZE &&
                    shape->pairs_per_row <= MAXIMUM_PAIRS_PER_ROW && row_count <= ROWS;

    test_linear_context("card %" PRIu32 ", %" PRIu32 " rows, %s, %s", card->card_index, row_count,
                        sharing_names[expert_case->sharing], shape->name);
    (void)printf("%" PRIu32 " rows, %s (%" PRIu32 " experts), %s\n", row_count, sharing_names[expert_case->sharing],
                 expert_count, shape->name);
    test_linear_card_clear(card);
    const struct linear_range values =
	    test_linear_card_work(card, row_count * row_value_bytes);
    for (uint32_t expert = 0; is_ready && expert < expert_count; expert++) {
        for (uint32_t matrix = 0; is_ready && matrix < MATRIX_COUNT; matrix++) {
            const bool is_down = matrix == MATRIX_DOWN;
            const struct test_linear_matrix_recipe recipe = {
                .format = E2M1,
                .shape = {is_down ? shape->hidden_size : shape->inner_size,
                          is_down ? shape->inner_size : shape->hidden_size, E2M1_BITS_PER_WEIGHT, 0},
                .stream = EXPERT_STREAM + MATRIX_STREAM + (uint64_t)expert * MATRIX_COUNT + matrix,
                .code = test_linear_e2m1_code,
                .scale_code_base = is_down ? DOWN_SCALE_CODE_BASE : shape->gate_scale_code_base,
                .scale_code_spread = SCALE_CODE_SPREAD};
            is_ready = test_linear_card_matrix(card, &recipe, &matrices[expert][matrix]);
        }
    }
    for (uint32_t row = 0; is_ready && row < row_count; row++) {
        test_linear_values(row_values[row], shape->hidden_size, EXPERT_STREAM + row,
                           row < PLAIN_ROWS ? TEST_LINEAR_VALUES_PLAIN : TEST_LINEAR_VALUES_EDGES);
	is_ready = test_linear_quantize(row_values[row], shape->hidden_size,
					row_codes[row],
					row_exponents[row]) == STATUS_OK &&
		   test_linear_card_put(card, values, row * row_value_bytes,
					row_values[row], row_value_bytes);
    }
    is_ready = is_ready &&
               set_prepare(card, expert_case, EXPERT_STREAM, expert_elsewhere, values, 0, row_count, &sets[0]);
    for (uint32_t row = 0; is_ready && row < alone_count; row++) {
        is_ready = set_prepare(card, expert_case, EXPERT_STREAM, expert_elsewhere,
                               part_of(values, row * row_value_bytes, row_value_bytes), row, 1, &sets[1 + row]);
    }
    if (is_ready && expert_case->is_shown_failing) {
        is_ready = set_prepare(card, expert_case, EXPERT_STREAM, NO_EXPERT, values, 0, row_count, gate_changed) &&
                   set_prepare(card, expert_case, EXPERT_STREAM, NO_EXPERT, values, 0, row_count, down_changed);
    }
    if (!is_ready) {
        (void)test_linear_expect(is_ready, "the case is laid out");
        return;
    }
    for (uint32_t set = 0; set <= alone_count; set++) {
        memcpy(&launches[launch_count], sets[set].launches, sizeof(sets[set].launches));
        launch_count += SET_LAUNCH_COUNT;
    }
    if (!test_linear_card_load(card) || !test_linear_card_run(card, launches, launch_count) ||
        !test_linear_card_fetch(card)) {
        return;
    }
    for (uint32_t row = 0; row < alone_count; row++) {
        for (uint32_t range = 0; range < SET_RANGE_COUNT; range++) {
            const uint64_t row_bytes = units_per_row(shape, (enum set_range)range) *
                                       unit_bytes(shape, (enum set_range)range);
            if (!test_linear_card_expect_same(card, sets[1 + row].ranges[range], 0, sets[0].ranges[range],
                                              row * row_bytes, row_bytes,
                                              "a row alone keeps, in every array of the chain, the bits it has in "
                                              "company")) {
                break;
            }
        }
        (void)test_linear_card_expect_untouched(card, sets[1 + row].jobs, "the job table of a row alone");
    }
    (void)test_linear_card_expect_untouched(card, values, "the rows' values");
    (void)test_linear_card_expect_untouched(card, sets[0].jobs, "the job table");
    (void)test_linear_card_expect_weights_untouched(card);
    check_set(card, expert_case, &sets[0], expert_elsewhere, "the rows in company", totals);
    if (expert_case->is_shown_failing) {
        show_changed_weights(card, expert_case, &sets[0], gate_changed, down_changed);
        (void)test_linear_card_expect_weights_untouched(card);
    }
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
        struct card_totals totals;
        memset(&totals, 0, sizeof(totals));
        if (test_linear_card_open(&card, card_index, WORK_CAPACITY, WEIGHTS_CAPACITY)) {
            for (uint32_t index = 0; index < COUNT_OF(shapes); index++) {   /* in the slices the card runs */
                struct expert_shape *const shape = &shapes[index];
                *shape = stamped_shapes[index];
                shape->gate_up_slice_count = test_linear_card_slices(&card, shape->gate_up_slice_count, true);
                shape->down_slice_count = test_linear_card_slices(&card, shape->down_slice_count, false);
            }
            for (uint32_t case_index = 0; case_index < COUNT_OF(expert_cases); case_index++) {
                test_expert_case(&card, &expert_cases[case_index], &totals);
            }
            test_linear_context("card %" PRIu32, card_index);
            (void)test_linear_expect(totals.hidden_equal_count * THOUSAND >=
                                         totals.hidden_count * HIDDEN_SHARE_REQUIRED_PER_THOUSAND,
                                     "99.7% of the hidden values are bit-equal to the hidden step of the two "
                                     "products rounded once");
            (void)test_linear_expect(test_linear_share_holds(totals.down.equal_count, totals.down.output_count),
                                     "99.9% of the outputs of pairs are bit-equal to the double result rounded once");
            (void)printf("card %" PRIu32 " (class %" PRIu32 "): %" PRIu64 " hidden values, %.4f%% bit-equal; %" PRIu64
                         " outputs of pairs, %.4f%% bit-equal; digest of everything it computed 0x%016" PRIX64 "\n",
                         card_index, card.arithmetic_class, totals.hidden_count,
                         test_linear_percent(totals.hidden_equal_count, totals.hidden_count),
                         totals.down.output_count,
                         test_linear_percent(totals.down.equal_count, totals.down.output_count), card.digest);
        }
        test_linear_card_close(&card);
    }
    if (test_linear_failures() != 0) {
        (void)fprintf(stderr, "linear_expert_card_test FAILED: %d failures\n", test_linear_failures());
        return EXIT_FAILURE;
    }
    (void)printf("linear_expert_card_test passed on %" PRIu32 " cards\n", card_count);
    return EXIT_SUCCESS;
}
