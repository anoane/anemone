/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * linear_gate_card_test: the engine's linear against the model's reference code, on the real weights of a pack and
 * on the real cards.
 *
 *     linear_gate_card_test <gate file> <pack copy> [<pack copy> ...]
 *
 * The gate file is written by tools/golden/make_linear_gate.py from a capture of the reference (its text states the
 * format): for every quantized linear the reference ran, the input before quantization, the input as the reference
 * quantized it, its scales, and the output. The pack is one that holds the model's matrices. Without arguments, as
 * `make card_tests` runs every card test, there is nothing to read: it says that nothing was compared and passes.
 *
 * On every card, for every linear of the gate file:
 *   - its matrix is found in the pack by the name the reference gave its weight (pack_find_tensor), must be
 *     of the capture's format and sizes, is read through the storage reader into page-locked memory and uploaded;
 *     the card's digest of it must then be the host's;
 *   - the engine's quantization (activation_quantize_entry) runs on the captured input: every code and every
 *     exponent must be the reference's, byte for byte, and every operand the carry of them
 *     (test_linear_operands_lay_out);
 *   - the engine's product runs on those codes, once for each number of slices of slice_counts, and its bfloat16
 *     outputs are compared with the reference's bit for bit;
 *   - a linear of 8-bit weights runs also as both matrices of one launch, with a copy of itself, for each number
 *     of slices: each half of the outputs must be the bits of the launch of the matrix alone.
 * The launches. A linear of 8-bit weights: linear_quantize_prepare and
 * linear_trunk_quantized_prepare. A linear of 4-bit weights (a call of one matrix of a routed expert with
 * its rows): the second and the third launch of linear_routed_experts_prepare, the quantization of the
 * hidden vectors and the product of down, with the captured input as the hidden values of pairs_per_row 1 and one
 * job whose down matrix is the matrix, be it the expert's gate, up or down. Rows go eight at a time, the most a
 * launch takes; the rows of a last, shorter batch run beside stale rows, which no row's bits depend on.
 *
 * It prints, per kind of linear and number of slices, the share of outputs bit-equal to the reference's and the
 * largest difference in last places of bfloat16, and per card three digests: of everything it computed; of the
 * arrays the first version of this test laid out (the ranges of 1, 4, 8 and 16 slices, which come first in the
 * same order), which batch g01 printed; and of the outputs of 2 slices of the rows that are not stale. (The operands
 * need none: they are held to the reference's carry of the captured codes, byte for byte; nor do the outputs of two
 * matrices in one launch, held to those of each alone.) It FAILS when a linear has, at any number of slices, fewer than
 * 99.9% of its outputs bit-equal; when a code or an exponent is not the reference's; when a linear is not in the
 * pack as the capture has it, or is of a shape no launch takes; when the gate file is not sound. Every count the
 * file states is bounded before it is used, and nothing is sized by what the file says: the rows of a batch and the
 * matrix of a linear have their limits here.
 */
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>

#include "anemone/backend_seam.h"
#include "anemone/linear.h"
#include "anemone/linear_records.h"
#include "anemone/pack.h"
#include "anemone/pack_digest.h"
#include "anemone/status.h"
#include "anemone/storage_seam.h"
#include "anemone/weight_format_seam.h"
#include "linear_card_fixture.h"
#include "linear_reference.h"

#define TEST_TIME_LIMIT_SECONDS 3600            /* the alarm ends the test here, whatever a drive or a driver does */
#define ARGUMENTS_BEFORE_THE_COPIES 2           /* the program's name and the gate file */
#define GATE_MAGIC "LINEARGT"
#define GATE_MAGIC_BYTES ((size_t)8)
#define GATE_FORMAT_VERSION 1u
#define GATE_HEADER_BYTES ((size_t)64)
#define GATE_RECORD_BYTES ((size_t)256)
#define GATE_HEADER_RESERVED_BYTES 32
#define GATE_RECORD_RESERVED_BYTES 8
#define GATE_WEIGHT_NAME_CAPACITY 96
#define GATE_LINEAR_NAME_CAPACITY 64
#define GATE_KIND_NAME_CAPACITY 40
#define GATE_LINEAR_COUNT_LIMIT 65536u          /* the limits of the writer: every count is held to them first */
#define GATE_ROW_COUNT_LIMIT 1048576u
#define GATE_ROW_SIZE_LIMIT 65536u
#define VALUES_PER_BLOCK TEST_LINEAR_VALUES_PER_BLOCK
#define VALUE_BYTES ((uint64_t)LINEAR_BFLOAT16_BYTES)
#define BATCH_ROWS LINEAR_MAXIMUM_ROWS
#define SLICE_VARIANTS 5u
#define FIRST_VERSION_VARIANTS 4u               /* the counts of the first version of the test, which come first */
#define TWO_SLICES 2u                           /* the count the first version did not run */
#define MATRIX_BYTE_LIMIT (192ull << 20u)       /* the largest matrix of the model holds 157 MiB */
#define WEIGHTS_CAPACITY                         \
	(MATRIX_BYTE_LIMIT + MATRIX_BYTE_LIMIT + \
	 2ull * BACKEND_STAGING_ALIGNMENT_BYTES)
#define WORK_CAPACITY (16ull << 20u)            /* eight rows of the widest linear: values, codes, four outputs */
#define MAXIMUM_KINDS 64u
#define MATRIX_DIMENSIONS 2u                    /* outputs, then inputs */
#define DUMMY_GATE_INPUT_SIZE \
	LINEAR_VALUES_PER_LOAD /* of the gate and up no launch of this test runs */
#define ACTIVATION_LIMIT 10.0f                  /* of the chain the product of 4-bit weights is a launch of */
#define FAILURES_SHOWN_PER_CARD 16u

enum test_status { TEST_PASSED = 0, TEST_FAILED = 1, TEST_WRONG_USAGE = 2 };

/* The numbers of slices every product is run with: one (the order in which the reference adds), the three the
 * first version of the test ran, and 2: every count a kernel is stamped for. */
static const uint32_t slice_counts[SLICE_VARIANTS] = {1, 4, 8, 16, 2};

struct gate_header {                            /* little-endian, 64 bytes, nothing between its members */
    char magic[GATE_MAGIC_BYTES];
    uint32_t format_version;
    uint32_t linear_count;
    uint64_t file_byte_count;
    uint32_t record_byte_count;
    uint32_t values_per_scale;
    unsigned char reserved_zero[GATE_HEADER_RESERVED_BYTES];
};

struct gate_record {                            /* little-endian, 256 bytes, nothing between its members */
    char weight_name[GATE_WEIGHT_NAME_CAPACITY];
    char linear_name[GATE_LINEAR_NAME_CAPACITY];
    char kind[GATE_KIND_NAME_CAPACITY];
    uint32_t weight_format;
    uint32_t row_count;
    uint32_t input_size;
    uint32_t output_size;
    uint64_t input_byte_offset;
    uint64_t quantized_input_byte_offset;
    uint64_t input_scales_byte_offset;
    uint64_t output_byte_offset;
    unsigned char reserved_zero[GATE_RECORD_RESERVED_BYTES];
};

_Static_assert(sizeof(struct gate_header) == GATE_HEADER_BYTES, "the header has no padding");
_Static_assert(sizeof(struct gate_record) == GATE_RECORD_BYTES, "a record has no padding");
_Static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__, "the gate file is read as the host's own integers");

/* What the test reads from: the gate file, and the pack through its storage reader. */
struct gate_sources {
    FILE *gate_file;
    struct gate_header header;
    struct storage_reader *reader;
    struct pack_record *pack;
    uint32_t block_alignment_bytes;
};

/* One linear on one card: its ranges and its launches, the quantization first, then the product for each number of
 * slices its matrix allows, then (8-bit weights) the product of the matrix with its copy in one launch for each. */
struct linear_plan {
    struct gate_record record;
    struct matrix_shape_record shape;
    struct linear_range matrix;
    struct linear_range copy; /* of the matrix: 8-bit weights only */
    struct linear_range values;
    struct linear_range codes;
    struct linear_range exponents;
    struct linear_range operands;
    struct linear_range outputs[SLICE_VARIANTS];
    struct linear_range merged_outputs[SLICE_VARIANTS][LINEAR_MERGED_MATRICES];
    struct kernel_launch launches[1u + SLICE_VARIANTS + SLICE_VARIANTS];
    uint32_t variant_count;
    uint32_t merged_count;
    uint32_t batch_rows;
};

/* What the linears of one kind added up to on one card. */
struct kind_total {
    char name[GATE_KIND_NAME_CAPACITY];
    uint64_t linear_count;
    uint64_t output_count;
    uint64_t equal_count[SLICE_VARIANTS];
    uint64_t compared_count[SLICE_VARIANTS];
    uint32_t largest_distance[SLICE_VARIANTS];
    uint32_t failed_linear_count;
};

/* The rows of the batch at hand, as the gate file holds them. */
static uint16_t batch_values[BATCH_ROWS * GATE_ROW_SIZE_LIMIT];
static uint8_t batch_codes[BATCH_ROWS * GATE_ROW_SIZE_LIMIT];           /* in the order of the values */
static uint8_t batch_scales[BATCH_ROWS * (GATE_ROW_SIZE_LIMIT / VALUES_PER_BLOCK)];
static uint16_t batch_expected[BATCH_ROWS * GATE_ROW_SIZE_LIMIT];
static uint16_t batch_outputs[BATCH_ROWS * GATE_ROW_SIZE_LIMIT];
static uint64_t two_slices_digest;              /* of the card at hand: the digests of the outputs of 2 slices, added */

/* byte_count bytes of the gate file from byte_offset on, or false: the range was held to the file's size before. */
static bool read_gate(FILE *gate_file, uint64_t byte_offset, void *destination, size_t byte_count)
{
    return byte_offset <= (uint64_t)INT64_MAX && fseeko(gate_file, (off_t)byte_offset, SEEK_SET) == 0 &&
           fread(destination, 1, byte_count, gate_file) == byte_count;
}

/* Whether a name of a record is one: not empty, and closed by a zero inside its field. */
static bool name_is_sound(const char *name, size_t capacity)
{
    return name[0] != '\0' && memchr(name, '\0', capacity) != NULL;
}

/* Whether an array of a record, byte_count bytes at byte_offset, lies in the file behind the records. */
static bool array_is_inside(const struct gate_header *header, uint64_t byte_offset, uint64_t byte_count)
{
    const uint64_t first_array_byte = GATE_HEADER_BYTES + (uint64_t)header->linear_count * GATE_RECORD_BYTES;
    return byte_offset >= first_array_byte && byte_offset <= header->file_byte_count &&
           byte_count <= header->file_byte_count - byte_offset;
}

/* Reads record record_index and holds every count and offset of it to the limits and to the file. */
static bool read_record(const struct gate_sources *sources, uint32_t record_index, struct gate_record *record)
{
    if (!read_gate(sources->gate_file, GATE_HEADER_BYTES + (uint64_t)record_index * GATE_RECORD_BYTES, record,
                   sizeof(*record))) {
        test_linear_fail("record %" PRIu32 " of the gate file cannot be read", record_index);
        return false;
    }
    const uint64_t row_count = record->row_count;
    const bool is_bounded = record->row_count >= 1 && record->row_count <= GATE_ROW_COUNT_LIMIT &&
                            record->input_size >= 1 && record->input_size <= GATE_ROW_SIZE_LIMIT &&
                            record->output_size >= 1 && record->output_size <= GATE_ROW_SIZE_LIMIT &&
                            record->input_size % VALUES_PER_BLOCK == 0;
    /* the counts are bounded: no product below passes 2^37 */
    if (!is_bounded || !name_is_sound(record->weight_name, sizeof(record->weight_name)) ||
        !name_is_sound(record->linear_name, sizeof(record->linear_name)) ||
        !name_is_sound(record->kind, sizeof(record->kind)) ||
        !array_is_inside(&sources->header, record->input_byte_offset, row_count * record->input_size * VALUE_BYTES) ||
        !array_is_inside(&sources->header, record->quantized_input_byte_offset, row_count * record->input_size) ||
        !array_is_inside(&sources->header, record->input_scales_byte_offset,
                         row_count * (record->input_size / VALUES_PER_BLOCK)) ||
        !array_is_inside(&sources->header, record->output_byte_offset,
                         row_count * record->output_size * VALUE_BYTES)) {
        test_linear_fail("record %" PRIu32 " of the gate file is not sound: a name without its end, a count beyond "
                         "the writer's limits, or an array outside the file", record_index);
        return false;
    }
    return true;
}

/* Finds the matrix of a linear in the pack, holds it to the record, reads it through the storage reader into the
 * image's weights region and says where it lies in the arena. */
static bool matrix_load(struct test_linear_card *card, const struct gate_sources *sources, struct linear_plan *plan)
{
    const struct gate_record *const record = &plan->record;
    const uint32_t name_byte_count = (uint32_t)strnlen(record->weight_name, sizeof(record->weight_name));
    const uint32_t tensor_index = pack_find_tensor(
	    sources->pack, record->weight_name, name_byte_count);
    const uint64_t alignment = sources->block_alignment_bytes;
    if (tensor_index == PACK_NOT_FOUND || alignment == 0) {
	    test_linear_fail(
		    "the pack holds no tensor of the name %s, or its reader states no block",
		    record->weight_name);
	    return false;
    }
    const struct pack_tensor *const tensor =
	    &pack_tensors_of(sources->pack)[tensor_index];
    const struct pack_extent *const extent =
	    &pack_extents_of(sources->pack)[tensor->extent_index];
    if (tensor->weight_format != record->weight_format || tensor->dimension_count != MATRIX_DIMENSIONS ||
        tensor->dimensions[0] != record->output_size || tensor->dimensions[1] != record->input_size ||
        tensor->byte_count == 0 || tensor->byte_count > MATRIX_BYTE_LIMIT) {
        test_linear_fail("%s is, in the pack, of format %" PRIu32 " and %" PRIu32 " by %" PRIu32 " in %" PRIu64
                         " bytes: the capture has format %" PRIu32 " and %" PRIu32 " by %" PRIu32 ", and the test "
                         "reads a matrix of at most %llu bytes", record->weight_name, tensor->weight_format,
                         tensor->dimensions[0], tensor->dimensions[1], tensor->byte_count, record->weight_format,
                         record->output_size, record->input_size, MATRIX_BYTE_LIMIT);
        return false;
    }
    /* a read is whole blocks of the reader: from the block the matrix begins in to the end of the one it ends in */
    const uint64_t file_byte_offset = extent->byte_offset + tensor->byte_offset_in_extent;
    const uint64_t lead_byte_count = file_byte_offset % alignment;
    const uint64_t read_byte_count =
        (lead_byte_count + tensor->byte_count + alignment - 1) / alignment * alignment;
    const struct linear_range read_range =
	    test_linear_card_weights(card, read_byte_count);
    unsigned char *const destination = test_linear_card_writable(card, read_range);
    plan->shape = (struct matrix_shape_record){ record->output_size,
						record->input_size,
						tensor->bits_per_weight, 0 };
    plan->matrix =
	    (struct linear_range){ read_range.byte_offset + lead_byte_count,
				   tensor->byte_count };
    return destination != NULL &&
	   test_linear_expect_status(storage_direct.reader_read(
					     sources->reader, 0,
					     file_byte_offset - lead_byte_count,
					     destination, read_byte_count),
				     STATUS_OK,
				     "the matrix is read from the pack");
}

/* Of a linear of 8-bit weights: a copy of its matrix, and for each number of slices one launch of the matrix and
 * the copy, behind the launches of the matrix alone. */
static bool merged_launches_prepare(struct test_linear_card *card, struct linear_plan *plan)
{
    const uint8_t *const original = test_linear_card_stored(card, plan->matrix);
    plan->merged_count = 0;
    if (plan->record.weight_format != WEIGHT_FORMAT_E4M3_SCALED) {
	    return true;
    }
    plan->copy = test_linear_card_weights(card, plan->matrix.byte_count);
    unsigned char *const copy = test_linear_card_writable(card, plan->copy);
    if (original == NULL || copy == NULL) {
        test_linear_fail("the matrix is not copied");
        return false;
    }
    memcpy(copy, original, plan->matrix.byte_count);
    for (uint32_t variant = 0; variant < plan->variant_count; variant++) {
	    struct linear_trunk_quantized_description pair = {
		    .codes = plan->codes,
		    .exponents = plan->exponents,
		    .operands = plan->operands,
		    .matrix_count = LINEAR_MERGED_MATRICES,
		    .row_count = plan->batch_rows
	    };
	    for (uint32_t half = 0; half < LINEAR_MERGED_MATRICES; half++) {
		    pair.matrices[half] = (struct linear_matrix){
			    .stored = half == 0 ? plan->matrix : plan->copy,
			    .shape = plan->shape,
			    .weight_format = WEIGHT_FORMAT_E4M3_SCALED,
			    .slice_count = slice_counts[variant]
		    };
		    plan->merged_outputs[variant][half] = pair.outputs[half] =
			    test_linear_card_work(
				    card, plan->outputs[variant].byte_count);
	    }
	    if (!test_linear_expect_status(
			linear_trunk_quantized_prepare(
				&pair, card->arena_byte_count, &card->kernels,
				&plan->launches[1u + SLICE_VARIANTS + variant]),
			STATUS_OK,
			"the linear as two matrices of one launch is prepared")) {
		    return false;
	    }
	plan->merged_count++;
    }
    return true;
}

/* Sets the ranges of a linear aside and prepares its launches: see the top. */
static bool launches_prepare(struct test_linear_card *card, struct linear_plan *plan)
{
    const struct gate_record *const record = &plan->record;
    const bool is_e2m1 = record->weight_format == WEIGHT_FORMAT_E2M1_SCALED;
    const uint32_t rows = plan->batch_rows;
    const uint32_t load_count = record->input_size / LINEAR_VALUES_PER_LOAD;
    struct expert_job_record job = { .gate_offset = plan->matrix.byte_offset,
				     .up_offset = plan->matrix.byte_offset,
				     .down_offset = plan->matrix.byte_offset,
				     .row_count = rows };
    bool is_ready = true;
    plan->values = test_linear_card_work(card, (uint64_t)rows * record->input_size * VALUE_BYTES);
    plan->codes = test_linear_card_work(card, (uint64_t)rows * record->input_size);
    plan->exponents = test_linear_card_work(card, (uint64_t)rows * (record->input_size / VALUES_PER_BLOCK));
    /* of a linear of 4-bit weights: the one job, and the arrays of the chain's launches that are not run */
    struct linear_range jobs = { 0, 0 };
    struct linear_range row_codes = { 0, 0 };
    struct linear_range row_exponents = { 0, 0 };
    struct linear_range row_output = { 0, 0 };
    for (uint32_t column = 0; column < rows; column++) {
        job.pair_index[column] = column;
        job.route_weight[column] = 1.0f;
    }
    if (is_e2m1) {
        jobs = test_linear_card_work(card, sizeof(job));
        row_codes = test_linear_card_work(card, (uint64_t)rows * DUMMY_GATE_INPUT_SIZE);
        row_exponents = test_linear_card_work(card, (uint64_t)rows * DUMMY_GATE_INPUT_SIZE / VALUES_PER_BLOCK);
	row_output = test_linear_card_work(
		card, (uint64_t)rows * record->output_size * LINEAR_SUM_BYTES);
	is_ready = test_linear_card_put(card, jobs, 0, &job, sizeof(job));
    }
    /* the ranges of the first version, then those it did not have: the outputs of 2 slices, the operands */
    plan->variant_count = 0;
    while (plan->variant_count < SLICE_VARIANTS && slice_counts[plan->variant_count] <= load_count) {
        plan->outputs[plan->variant_count] =
            test_linear_card_work(card, (uint64_t)rows * record->output_size * VALUE_BYTES);
        card->prefix_byte_count = plan->variant_count < FIRST_VERSION_VARIANTS ? card->work_byte_count
                                                                               : card->prefix_byte_count;
        plan->variant_count++;
    }
    plan->operands = test_linear_card_work(card, (uint64_t)rows * record->input_size * VALUE_BYTES);
    const struct linear_range row_operands =
	    is_e2m1 ?
		    test_linear_card_work(card, (uint64_t)rows *
							DUMMY_GATE_INPUT_SIZE *
							VALUE_BYTES) :
		    jobs;
    const struct linear_quantize_description quantize = {
	    .values = plan->values,
	    .codes = plan->codes,
	    .exponents = plan->exponents,
	    .operands = plan->operands,
	    .width = record->input_size,
	    .row_count = rows
    };
    if (!is_e2m1) {
	    is_ready = test_linear_expect_status(
		    linear_quantize_prepare(&quantize, card->arena_byte_count,
					    &card->kernels, &plan->launches[0]),
		    STATUS_OK,
		    "the quantization of the linear's rows is prepared");
    }
    for (uint32_t variant = 0; is_ready && variant < plan->variant_count; variant++) {
	    struct kernel_launch chain[LINEAR_ROUTED_EXPERTS_LAUNCH_COUNT];
	    const struct linear_matrix matrix = {
		    .stored = plan->matrix,
		    .shape = plan->shape,
		    .weight_format = WEIGHT_FORMAT_E4M3_SCALED,
		    .slice_count = slice_counts[variant]
	    };
	    const struct linear_trunk_quantized_description trunk = {
		    .matrices = { matrix },
		    .codes = plan->codes,
		    .exponents = plan->exponents,
		    .operands = plan->operands,
		    .outputs = { plan->outputs[variant] },
		    .matrix_count = 1,
		    .row_count = rows
	    };
	    const struct linear_routed_experts_description experts = {
		    .jobs = jobs,
		    .expert_weights = plan->matrix,
		    .codes = row_codes,
		    .exponents = row_exponents,
		    .operands = row_operands,
		    .hidden = plan->values,
		    .hidden_codes = plan->codes,
		    .hidden_exponents = plan->exponents,
		    .hidden_operands = plan->operands,
		    .pair_output = plan->outputs[variant],
		    .row_output = row_output,
		    .gate_up_shape = { record->input_size,
				       DUMMY_GATE_INPUT_SIZE,
				       plan->shape.bits_per_weight, 0 },
		    .down_shape = plan->shape,
		    .weight_format = WEIGHT_FORMAT_E2M1_SCALED,
		    .job_count = 1,
		    .row_count = rows,
		    .pairs_per_row = 1,
		    .gate_up_slice_count = 1,
		    .down_slice_count = slice_counts[variant],
		    .activation_limit = ACTIVATION_LIMIT
	    };
	    if (is_e2m1) {
		    is_ready = test_linear_expect_status(
			    linear_routed_experts_prepare(
				    &experts, card->arena_byte_count,
				    &card->kernels, chain),
			    STATUS_OK,
			    "the chain of the routed experts is prepared for its product of down");
		    if (is_ready) { /* a refusal leaves the launches untouched: nothing to take from them */
			    plan->launches[0] =
				    chain[LINEAR_ROUTED_EXPERTS_QUANTIZE];
			    plan->launches[1u + variant] =
				    chain[LINEAR_ROUTED_EXPERTS_DOWN];
		    }
        } else {
		is_ready = test_linear_expect_status(
			linear_trunk_quantized_prepare(
				&trunk, card->arena_byte_count, &card->kernels,
				&plan->launches[1u + variant]),
			STATUS_OK, "the product of the linear is prepared");
	}
    }
    return is_ready && merged_launches_prepare(card, plan) &&
           test_linear_expect(plan->variant_count != 0 && test_linear_card_has_room(card),
                              "the linear has a number of slices and its ranges have room");
}

/* One batch of live_rows rows from first_row on: reads them from the gate file, runs the launches and compares.
 * False when the batch could not be run or its quantization is not the reference's. */
static bool batch_run(struct test_linear_card *card, const struct gate_sources *sources,
                      const struct linear_plan *plan, uint32_t first_row, uint32_t live_rows,
                      struct kind_total *linear_total)
{
    const struct gate_record *const record = &plan->record;
    const size_t input_size = record->input_size;
    const size_t output_size = record->output_size;
    const size_t scales_per_row = input_size / VALUES_PER_BLOCK;
    const size_t output_count = live_rows * output_size;
    bool is_quantized_as_the_reference = true;
    if (!read_gate(sources->gate_file, record->input_byte_offset + first_row * input_size * VALUE_BYTES, batch_values,
                   live_rows * input_size * VALUE_BYTES) ||
        !read_gate(sources->gate_file, record->quantized_input_byte_offset + first_row * input_size, batch_codes,
                   live_rows * input_size) ||
        !read_gate(sources->gate_file, record->input_scales_byte_offset + first_row * scales_per_row, batch_scales,
                   live_rows * scales_per_row) ||
        !read_gate(sources->gate_file, record->output_byte_offset + first_row * output_size * VALUE_BYTES,
                   batch_expected, output_count * VALUE_BYTES)) {
        test_linear_fail("rows %" PRIu32 " and on of %s cannot be read from the gate file", first_row,
                         record->linear_name);
        return false;
    }
    if (!test_linear_card_put(card, plan->values, 0, batch_values, live_rows * input_size * VALUE_BYTES) ||
        !test_linear_card_upload(card, plan->values, 0, live_rows * input_size * VALUE_BYTES) ||
        !test_linear_card_run(card, plan->launches, 1u + plan->variant_count) ||
        (plan->merged_count != 0 &&
         !test_linear_card_run(card, &plan->launches[1u + SLICE_VARIANTS], plan->merged_count)) ||
        !test_linear_card_fetch(card)) {
        return false;
    }
    for (uint32_t index = 0;
	 index < plan->merged_count * LINEAR_MERGED_MATRICES; index++) {
	    if (!test_linear_card_expect_same(
			card,
			plan->merged_outputs[index / LINEAR_MERGED_MATRICES]
					    [index % LINEAR_MERGED_MATRICES],
			0, plan->outputs[index / LINEAR_MERGED_MATRICES], 0,
			output_count * VALUE_BYTES,
			"the linear as either matrix of a launch of two has the bits it has alone")) {
		    return false;
	    }
    }
    for (uint32_t row = 0; is_quantized_as_the_reference && row < live_rows; row++) {
        is_quantized_as_the_reference =
            test_linear_card_expect_codes(card, plan->codes, row * input_size, batch_codes + row * input_size,
                                          record->input_size, "the engine's quantization of a captured row") &&
            test_linear_card_expect_bytes(card, plan->exponents, row * scales_per_row,
                                          batch_scales + row * scales_per_row, scales_per_row,
                                          "the engine's exponents of a captured row") &&
            test_linear_card_expect_operands(card, plan->operands, row * input_size * VALUE_BYTES,
                                             batch_codes + row * input_size, batch_scales + row * scales_per_row,
                                             record->input_size, "the engine's operands of a captured row");
    }
    for (uint32_t variant = 0; variant < plan->variant_count; variant++) {
        uint64_t digest = 0;
	if (!test_linear_card_get(card, plan->outputs[variant], 0,
				  batch_outputs, output_count * VALUE_BYTES) ||
	    !test_linear_expect_status(
		    pack_digest_of_host_bytes(
			    batch_outputs, output_count * VALUE_BYTES, &digest),
		    STATUS_OK, "the outputs are digested")) {
		return false;
	}
	two_slices_digest += slice_counts[variant] == TWO_SLICES ? digest : 0;
        for (size_t index = 0; index < output_count; index++) {
            const int32_t distance = test_linear_bfloat16_ordinal(batch_outputs[index]) -
                                     test_linear_bfloat16_ordinal(batch_expected[index]);
            const uint32_t last_places = (uint32_t)(distance < 0 ? -distance : distance);
            linear_total->equal_count[variant] += batch_outputs[index] == batch_expected[index] ? 1u : 0u;
            if (last_places > linear_total->largest_distance[variant]) {
                linear_total->largest_distance[variant] = last_places;
            }
        }
        linear_total->compared_count[variant] += output_count;
    }
    linear_total->output_count += output_count;
    return is_quantized_as_the_reference;
}

/* The total of a kind of linear in a table of MAXIMUM_KINDS entries, added if it is new; NULL if the table is
 * full. */
static struct kind_total *kind_of(struct kind_total *kinds, const char *name)
{
    uint32_t kind_index = 0;
    while (kind_index < MAXIMUM_KINDS && kinds[kind_index].name[0] != '\0' &&
           strncmp(kinds[kind_index].name, name, GATE_KIND_NAME_CAPACITY) != 0) {
        kind_index++;
    }
    if (kind_index == MAXIMUM_KINDS) {
        return NULL;
    }
    if (kinds[kind_index].name[0] == '\0') {
        (void)snprintf(kinds[kind_index].name, sizeof(kinds[kind_index].name), "%s", name);
    }
    return &kinds[kind_index];
}

/* One linear of the gate file on one card: false when it fails the gate. */
static bool linear_run(struct test_linear_card *card, const struct gate_sources *sources, uint32_t record_index,
                       struct kind_total *kinds)
{
    static struct linear_plan plan;
    struct kind_total linear_total;
    bool is_passed = true;
    memset(&plan, 0, sizeof(plan));
    memset(&linear_total, 0, sizeof(linear_total));
    test_linear_context("card %" PRIu32 ", linear %" PRIu32 " of the gate file", card->card_index, record_index);
    if (!read_record(sources, record_index, &plan.record)) {
        return false;
    }
    const struct gate_record *const record = &plan.record;
    struct kind_total *const kind = kind_of(kinds, record->kind);
    test_linear_context("card %" PRIu32 ", %s", card->card_index, record->linear_name);
    plan.batch_rows = record->row_count < BATCH_ROWS ? record->row_count : BATCH_ROWS;
    test_linear_card_clear(card);
    if (kind == NULL) {
        test_linear_fail("the gate file holds more kinds of linear than the test counts");
        return false;
    }
    if (!test_linear_expect(
		(record->weight_format == WEIGHT_FORMAT_E2M1_SCALED ||
		 record->weight_format == WEIGHT_FORMAT_E4M3_SCALED) &&
			record->input_size % LINEAR_VALUES_PER_LOAD == 0,
		"the linear is of a weight format and an input size the kernels take") ||
	!matrix_load(card, sources, &plan) || !launches_prepare(card, &plan) ||
	!test_linear_card_load(card) ||
	!test_linear_card_expect_weights_untouched(card)) {
	    return false;
    }
    for (uint32_t first_row = 0; first_row < record->row_count; first_row += plan.batch_rows) {
        const uint32_t rows_left = record->row_count - first_row;
        if (!batch_run(card, sources, &plan, first_row, rows_left < plan.batch_rows ? rows_left : plan.batch_rows,
                       &linear_total)) {
            return false;
        }
    }
    kind->linear_count++;
    kind->output_count += linear_total.output_count;
    for (uint32_t variant = 0; variant < plan.variant_count; variant++) {
        kind->equal_count[variant] += linear_total.equal_count[variant];
        kind->compared_count[variant] += linear_total.compared_count[variant];
        kind->largest_distance[variant] = linear_total.largest_distance[variant] > kind->largest_distance[variant]
                                              ? linear_total.largest_distance[variant]
                                              : kind->largest_distance[variant];
        if (!test_linear_share_holds(linear_total.equal_count[variant], linear_total.compared_count[variant])) {
            if (kind->failed_linear_count < FAILURES_SHOWN_PER_CARD) {
                test_linear_fail("in %" PRIu32 " slices %" PRIu64 " of the %" PRIu64 " outputs of %s are bit-equal "
                                 "to the reference's: %.4f%%, below 99.9%%", slice_counts[variant],
                                 linear_total.equal_count[variant], linear_total.compared_count[variant],
                                 record->weight_name,
                                 test_linear_percent(linear_total.equal_count[variant],
                                                     linear_total.compared_count[variant]));
            }
            is_passed = false;
        }
    }
    kind->failed_linear_count += is_passed ? 0u : 1u;
    return is_passed && test_linear_card_expect_weights_untouched(card);
}

/* Every linear of the gate file on one card, and what its kinds added up to. The linears that failed the gate. */
static uint32_t card_run(uint32_t card_index, const struct gate_sources *sources)
{
    static struct test_linear_card card;
    static struct kind_total kinds[MAXIMUM_KINDS];
    uint32_t failed_count = 0;
    memset(kinds, 0, sizeof(kinds));
    two_slices_digest = 0;
    if (!test_linear_card_open(&card, card_index, WORK_CAPACITY, WEIGHTS_CAPACITY)) {
        test_linear_card_close(&card);
        return sources->header.linear_count;
    }
    for (uint32_t record_index = 0; record_index < sources->header.linear_count; record_index++) {
        failed_count += linear_run(&card, sources, record_index, kinds) ? 0u : 1u;
    }
    for (uint32_t kind_index = 0; kind_index < MAXIMUM_KINDS && kinds[kind_index].name[0] != '\0'; kind_index++) {
        const struct kind_total *const kind = &kinds[kind_index];
        (void)printf("card %" PRIu32 ", %s: %" PRIu64 " linears, %" PRIu64 " outputs, %" PRIu32 " linears below "
                     "99.9%%\n", card_index, kind->name, kind->linear_count, kind->output_count,
                     kind->failed_linear_count);
        for (uint32_t variant = 0; variant < SLICE_VARIANTS; variant++) {
            if (kind->compared_count[variant] != 0) {
                (void)printf("  in %2" PRIu32 " slices: %.4f%% bit-equal to the reference's, the largest "
                             "difference %" PRIu32 " last places\n", slice_counts[variant],
                             test_linear_percent(kind->equal_count[variant], kind->compared_count[variant]),
                             kind->largest_distance[variant]);
            }
        }
    }
    (void)printf("card %" PRIu32 " (class %" PRIu32 "): %" PRIu32 " of %" PRIu32 " linears fail the gate; digest of "
                 "everything it computed 0x%016" PRIX64 "\n", card_index, card.arithmetic_class, failed_count,
                 sources->header.linear_count, card.digest);
    (void)printf("card %" PRIu32 " (class %" PRIu32 "): digest of the arrays of the first version 0x%016" PRIX64 "\n",
                 card_index, card.arithmetic_class, card.prefix_digest);
    (void)printf("card %" PRIu32 " (class %" PRIu32 "): digest of the outputs of 2 slices 0x%016" PRIX64 "\n",
                 card_index, card.arithmetic_class, two_slices_digest);
    test_linear_card_close(&card);
    return failed_count;
}

/* Opens the gate file and reads its header: the magic, the version, the sizes and the counts, each held to its
 * limit and the file to the size it states. */
static bool gate_open(const char *path, struct gate_sources *sources)
{
    struct gate_header *const header = &sources->header;
    off_t end_position = -1;
    sources->gate_file = fopen(path, "rb");
    if (sources->gate_file != NULL && fseeko(sources->gate_file, 0, SEEK_END) == 0) {
        end_position = ftello(sources->gate_file);
    }
    if (sources->gate_file == NULL || end_position < (off_t)GATE_HEADER_BYTES ||
        !read_gate(sources->gate_file, 0, header, sizeof(*header))) {
        test_linear_fail("the gate file cannot be opened or holds no header");
        return false;
    }
    if (memcmp(header->magic, GATE_MAGIC, GATE_MAGIC_BYTES) != 0 || header->format_version != GATE_FORMAT_VERSION ||
        header->record_byte_count != GATE_RECORD_BYTES || header->values_per_scale != VALUES_PER_BLOCK ||
        header->linear_count == 0 || header->linear_count > GATE_LINEAR_COUNT_LIMIT ||
        header->file_byte_count != (uint64_t)end_position ||
        header->file_byte_count < GATE_HEADER_BYTES + (uint64_t)header->linear_count * GATE_RECORD_BYTES) {
        test_linear_fail("the gate file is not one of make_linear_gate.py, format version %u: %" PRIu32 " linears "
                         "in records of %" PRIu32 " bytes, %" PRIu64 " bytes stated, %lld bytes long",
                         GATE_FORMAT_VERSION, header->linear_count, header->record_byte_count,
                         header->file_byte_count, (long long)end_position);
        return false;
    }
    return true;
}

int main(int argument_count, char **arguments)
{
    static struct gate_sources sources;
    struct storage_description description = { 0, 0, 0 };
    uint32_t card_count = 0;
    uint32_t failed_count = 0;

    if (argument_count == 1) {
        (void)printf("linear_gate_card_test passed (nothing compared: it needs a gate file and a pack)\n");
        return TEST_PASSED;
    }
    if (argument_count <= ARGUMENTS_BEFORE_THE_COPIES ||
	argument_count - ARGUMENTS_BEFORE_THE_COPIES > STORAGE_MAXIMUM_COPIES) {
	    (void)fprintf(
		    stderr,
		    "usage: linear_gate_card_test <gate file> <pack copy> [<pack copy> ...]\n");
	    return TEST_WRONG_USAGE;
    }
    (void)alarm(TEST_TIME_LIMIT_SECONDS);
    const char *const *const copy_paths = (const char *const *)&arguments[ARGUMENTS_BEFORE_THE_COPIES];
    const uint32_t copy_count = (uint32_t)(argument_count - ARGUMENTS_BEFORE_THE_COPIES);
    result_status status =
	    storage_direct.reader_open(copy_paths, copy_count, &sources.reader);
    status = status == STATUS_OK ?
		     storage_direct.reader_describe(sources.reader,
						    &description) :
		     status;
    status = status == STATUS_OK ?
		     pack_open_entry(&storage_direct, sources.reader,
				     &sources.pack) :
		     status;
    sources.block_alignment_bytes = description.block_alignment_bytes;
    /* the weights region of a card's image begins on a staging page: a read into it is aligned for the reader */
    if (test_linear_expect_status(status, STATUS_OK, "the pack opens") &&
	test_linear_expect(
		description.block_alignment_bytes != 0 &&
			BACKEND_STAGING_ALIGNMENT_BYTES %
					description.block_alignment_bytes ==
				0,
		"the reader's blocks divide a staging page") &&
	gate_open(arguments[1], &sources) &&
	test_linear_expect_status(backend_cuda_value.card_count(&card_count),
				  STATUS_OK, "the cards are counted") &&
	test_linear_expect(card_count > 0 &&
				   card_count <= BACKEND_MAXIMUM_CARDS,
			   "the machine has cards for the test")) {
	    for (uint32_t card_index = 0; card_index < card_count;
		 card_index++) {
		    failed_count += card_run(card_index, &sources);
	    }
    }
    if (sources.gate_file != NULL) {
        (void)fclose(sources.gate_file);
    }
    pack_close(sources.pack);
    storage_direct.reader_close(sources.reader);
    if (test_linear_failures() != 0 || failed_count != 0) {
        (void)fprintf(stderr, "linear_gate_card_test FAILED: %" PRIu32 " linears fail the gate, %d failure lines\n",
                      failed_count, test_linear_failures());
        return TEST_FAILED;
    }
    (void)printf("linear_gate_card_test passed: %" PRIu32 " linears on %" PRIu32 " cards, each with at least 99.9%% "
                 "of its outputs bit-equal to the reference's at every number of slices\n",
                 sources.header.linear_count, card_count);
    return TEST_PASSED;
}
