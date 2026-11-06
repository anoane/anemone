/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * relaxed_linear.c: the launches of the trunk products of class 80 in the relaxed order, prepared from descriptions.
 * Contract: relaxed_linear.h; the kernels, records and launches: relaxed_linear_records.h. As linear.c: the counts
 * and matrices first, then the ranges, then the launches; nothing written before every check has passed.
 */
#include "anemone/relaxed_linear.h"

#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>

#include "anemone/hooks.h"
#include "anemone/model_layout.h"
#include "anemone/step_records.h"
#include "prepare_common.h"

#define COUNT_OF(table) ((uint32_t)(sizeof(table) / sizeof((table)[0])))
#define KERNEL_COUNT (sizeof(struct relaxed_linear_kernels) / sizeof(uint32_t))
#define FLOAT32_EXPONENT_SHIFT 23u
#define SINGLE_RANGES \
	3u /* the ranges of one matrix: the operands, the matrix, its output */
#define MERGED_RANGES 5u /* and the second matrix with its output */
#define FLOAT32_LARGEST_FIELD 254
#define CARRIED_OPERAND_EXPONENT \
	100 /* a carried operand is its value times 2^100 */
/* the float32 field of 2^(reference - 227), the power that undoes the fold and the operands' 2^100 */
#define CARRIED_OUTPUT_FIELD (RELAXED_FOLD_REFERENCE - CARRIED_OPERAND_EXPONENT)

/* The kernels in the order of the members of struct relaxed_linear_kernels, with their record sizes. */
struct relaxed_kernel {
	const char *function_name;
	uint32_t argument_byte_count;
};
static const struct relaxed_kernel relaxed_kernels[] = {
	{ KERNEL_RELAXED_WIDE_QUANTIZED_SMALL,
	  TRUNK_RELAXED_QUANTIZED_ARGUMENT_BYTES },
	{ KERNEL_RELAXED_WIDE_QUANTIZED_LARGE,
	  TRUNK_RELAXED_QUANTIZED_ARGUMENT_BYTES },
	{ KERNEL_ORDERED_WIDE_QUANTIZED_SMALL,
	  TRUNK_RELAXED_QUANTIZED_ARGUMENT_BYTES },
	{ KERNEL_ORDERED_WIDE_QUANTIZED_LARGE,
	  TRUNK_RELAXED_QUANTIZED_ARGUMENT_BYTES },
	{ KERNEL_RELAXED_WIDE_BFLOAT16_SMALL,
	  TRUNK_RELAXED_BFLOAT16_ARGUMENT_BYTES },
	{ KERNEL_RELAXED_WIDE_BFLOAT16_LARGE,
	  TRUNK_RELAXED_BFLOAT16_ARGUMENT_BYTES },
	{ KERNEL_ORDERED_WIDE_BFLOAT16_SMALL,
	  TRUNK_RELAXED_BFLOAT16_ARGUMENT_BYTES },
	{ KERNEL_ORDERED_WIDE_BFLOAT16_LARGE,
	  TRUNK_RELAXED_BFLOAT16_ARGUMENT_BYTES },
	{ KERNEL_RELAXED_DECODE_QUANTIZED,
	  TRUNK_RELAXED_QUANTIZED_ARGUMENT_BYTES },
	{ KERNEL_RELAXED_DECODE_BFLOAT16,
	  TRUNK_RELAXED_BFLOAT16_ARGUMENT_BYTES },
	{ KERNEL_RELAXED_REDUCE, TRUNK_RELAXED_REDUCE_ARGUMENT_BYTES },
};
_Static_assert(COUNT_OF(relaxed_kernels) == KERNEL_COUNT &&
		       KERNEL_COUNT <= BACKEND_MAXIMUM_KERNELS,
	       "a function for each handle, in their order");

/* The slice counts the decode form takes: the stamped counts of linear_records.h. */
#define LINEAR_SLICE_STAMP(slices) (slices),
static const uint32_t decode_slice_counts[] = { LINEAR_EVERY_SLICE_COUNT };
#undef LINEAR_SLICE_STAMP

result_status relaxed_linear_kernels_prepare(
	const struct backend_operations *backend, struct device_card *card,
	uint32_t arithmetic_class, struct relaxed_linear_kernels *kernels_out)
{
	uint32_t handles[KERNEL_COUNT] = { 0 };
	if (backend == NULL || backend->kernel_prepare == NULL ||
	    card == NULL || kernels_out == NULL) {
		LOG_ERROR("the relaxed kernels: a missing argument");
		return STATUS_INVALID_ARGUMENT;
	}
	if (arithmetic_class != ARITHMETIC_CLASS_SM_80) {
		LOG_ERROR("the relaxed kernels: class %" PRIu32
			  " has its own forms; they are class 80's",
			  arithmetic_class);
		return STATUS_UNSUPPORTED;
	}
	for (uint32_t index = 0; index < KERNEL_COUNT; index++) {
		handles[index] = BACKEND_NO_KERNEL;
		const result_status status = backend->kernel_prepare(
			card, KERNEL_FILE_TRUNK_RELAXED,
			relaxed_kernels[index].function_name,
			relaxed_kernels[index].argument_byte_count,
			&handles[index]);
		if (status != STATUS_OK) {
			LOG_ERROR("the relaxed kernels: %s is not prepared: %s",
				  relaxed_kernels[index].function_name,
				  status_text(status));
			return status;
		}
	}
	memcpy(kernels_out, handles, sizeof(*kernels_out));
	return STATUS_OK;
}

result_status relaxed_fold_check(enum relaxed_rows_code rows,
				 const uint8_t *scale_codes,
				 uint64_t scale_count,
				 int32_t *fold_reference_out)
{
	uint32_t smallest = UINT8_MAX, largest = 0;
	if (scale_codes == NULL || scale_count == 0 ||
	    fold_reference_out == NULL ||
	    (rows != RELAXED_QUANTIZED_ROWS && rows != RELAXED_BFLOAT16_ROWS)) {
		LOG_ERROR(
			"the fold check: a missing argument, no scale code or rows of no known form");
		return STATUS_INVALID_ARGUMENT;
	}
	for (uint64_t index = 0; index < scale_count; index++) {
		smallest = scale_codes[index] < smallest ? scale_codes[index] :
							   smallest;
		largest = scale_codes[index] > largest ? scale_codes[index] :
							 largest;
	}
	if (rows == RELAXED_QUANTIZED_ROWS &&
	    (smallest < RELAXED_SCALE_SMALLEST ||
	     largest > RELAXED_SCALE_LARGEST)) {
		LOG_ERROR("the fold check: scale codes %" PRIu32 " to %" PRIu32
			  ": quantized rows fold %u to %u "
			  "exactly",
			  smallest, largest, RELAXED_SCALE_SMALLEST,
			  RELAXED_SCALE_LARGEST);
		return STATUS_INVALID_ARGUMENT;
	}
	/* bfloat16 rows: 16-bit float weights from 2^-24 to 448 * 2^7, and an output power 2^(reference - 127) */
	const int32_t reference = (int32_t)largest - RELAXED_HALF_TOP;
	if (rows == RELAXED_BFLOAT16_ROWS &&
	    (largest - smallest > RELAXED_HALF_SCALE_SPREAD ||
	     largest > RELAXED_HALF_SCALE_LARGEST || reference < 1 ||
	     reference > FLOAT32_LARGEST_FIELD)) {
		LOG_ERROR("the fold check: scale codes %" PRIu32 " to %" PRIu32
			  ": bfloat16 rows fold a spread "
			  "of %u at most, with codes at most %u",
			  smallest, largest, RELAXED_HALF_SCALE_SPREAD,
			  RELAXED_HALF_SCALE_LARGEST);
		return STATUS_INVALID_ARGUMENT;
	}
	*fold_reference_out = rows == RELAXED_BFLOAT16_ROWS ?
				      reference :
				      RELAXED_FOLD_REFERENCE;
	return STATUS_OK;
}

/* A matrix the relaxed kernels read: E4M3_SCALED of a shape that format stores, whole blocks of 128 outputs, whole
 * loads of inputs, at most RELAXED_MOST_INPUT_SIZE of them. Gives its stored bytes. */
static bool matrix_is_read(const char *what, const char *name,
			   const struct linear_matrix *matrix,
			   uint64_t *stored_byte_count_out)
{
	const struct weight_format_operations *const operations =
		weight_format_operations_of(matrix->weight_format);
	const struct matrix_shape_record *const shape = &matrix->shape;
	if (matrix->weight_format != WEIGHT_FORMAT_E4M3_SCALED ||
	    operations == NULL || matrix->reserved_zero != 0 ||
	    shape->reserved_must_be_zero != 0 ||
	    operations->stored_byte_count(shape, stored_byte_count_out) !=
		    STATUS_OK) {
		LOG_ERROR(
			"%s: %s is no matrix of format %u that the format stores",
			what, name, (uint32_t)WEIGHT_FORMAT_E4M3_SCALED);
		return false;
	}
	if (shape->output_size % RELAXED_WIDE_OUTPUTS != 0 ||
	    shape->input_size % LINEAR_VALUES_PER_LOAD != 0 ||
	    shape->input_size > RELAXED_MOST_INPUT_SIZE) {
		LOG_ERROR("%s: %s is %" PRIu32 " by %" PRIu32
			  ": the relaxed kernels take blocks of %u outputs "
			  "and whole loads of %u inputs, at most %u",
			  what, name, shape->output_size, shape->input_size,
			  RELAXED_WIDE_OUTPUTS,
			  (uint32_t)LINEAR_VALUES_PER_LOAD,
			  RELAXED_MOST_INPUT_SIZE);
		return false;
	}
	return true;
}

/* The counts of a launch: rows, splits, slices for the form they make. */
static bool counts_are_valid(const char *what, uint32_t row_count,
			     uint32_t k_split_count, uint32_t slice_count,
			     uint32_t input_size, uint32_t output_total,
			     bool is_ordered)
{
	const uint32_t load_count = input_size / LINEAR_VALUES_PER_LOAD;
	const bool is_decode = row_count < RELAXED_WIDE_LEAST_ROWS;
	bool is_stamped = false;
	for (uint32_t index = 0; index < COUNT_OF(decode_slice_counts);
	     index++) {
		is_stamped = is_stamped ||
			     decode_slice_counts[index] == slice_count;
	}
	if (row_count == 0 || row_count > RELAXED_MOST_ROWS ||
	    (is_ordered && is_decode)) {
		LOG_ERROR(
			"%s: %" PRIu32
			" rows: a launch takes 1 to %u (the ordered one more than %u)",
			what, row_count, RELAXED_MOST_ROWS,
			LINEAR_MAXIMUM_ROWS);
		return false;
	}
	if (k_split_count == 0 || k_split_count > RELAXED_MOST_K_SPLITS ||
	    k_split_count > load_count ||
	    (k_split_count != 1u && (is_decode || is_ordered))) {
		LOG_ERROR(
			"%s: %" PRIu32
			" splits: 1 to %u of the wide relaxed form, at most its loads",
			what, k_split_count, RELAXED_MOST_K_SPLITS);
		return false;
	}
	if (is_decode && (!is_stamped || slice_count > load_count)) {
		LOG_ERROR(
			"%s: %" PRIu32
			" slices: the decode form takes a stamped count, at most its loads",
			what, slice_count);
		return false;
	}
	const uint64_t row_tiles =
		((uint64_t)row_count + RELAXED_WIDE_ROWS_SMALL - 1u) /
		RELAXED_WIDE_ROWS_SMALL;
	const uint64_t reduce_blocks = ((uint64_t)row_count * output_total +
					RELAXED_REDUCE_THREADS - 1u) /
				       RELAXED_REDUCE_THREADS;
	if ((!is_decode &&
	     row_tiles * (output_total / RELAXED_WIDE_OUTPUTS) > UINT32_MAX) ||
	    (k_split_count > 1u && reduce_blocks > UINT32_MAX)) {
		LOG_ERROR(
			"%s: the tile grid or split reduction passes 32-bit block counts",
			what);
		return false;
	}
	return true;
}

/* The launches of a product (the record built by the caller) and, with splits, of their reduce. */
static void product_launches_fill(
	struct kernel_launch *launches, uint32_t *launch_count_out,
	const struct relaxed_linear_kernels *kernels,
	const uint32_t *wide_kernels, uint32_t decode_kernel,
	uint32_t slice_count, uint32_t output_total,
	const uint64_t *output_offsets, const uint32_t *output_sizes,
	uint64_t partials_offset, uint32_t row_count, uint32_t k_split_count,
	float output_scale, uint32_t input_size, const void *record,
	uint32_t record_byte_count)
{
	if (row_count < RELAXED_WIDE_LEAST_ROWS) {
		const uint32_t threads = slice_count * LINEAR_THREADS_PER_WARP;
		prepare_launch_fill(&launches[0], decode_kernel,
				    output_total / LINEAR_OUTPUTS_PER_TILE, 1,
				    threads, threads * RELAXED_DECODE_SUM_BYTES,
				    record, record_byte_count);
		*launch_count_out = 1;
		return;
	}
	/* tiles of 64 rows, three blocks a multiprocessor: faster than tiles of 128 at every row count measured
	 * (trunk_relaxed_card_test, 26 October); the instance of 128 rows stays for the measurements */
	const uint32_t tile_rows = RELAXED_WIDE_ROWS_SMALL;
	const uint32_t threads = tile_rows / RELAXED_WIDE_WARP_ROWS *
				 RELAXED_WIDE_WARPS_ALONG_OUTPUTS *
				 LINEAR_THREADS_PER_WARP;
	prepare_launch_fill(&launches[0], wide_kernels[RELAXED_TILE_SMALL],
			    (row_count + tile_rows - 1u) / tile_rows,
			    output_total / RELAXED_WIDE_OUTPUTS, threads,
			    RELAXED_WIDE_SHARED_BYTES(tile_rows, input_size),
			    record, record_byte_count);
	launches[0].grid[LAUNCH_DIMENSIONS - 1u] = k_split_count;
	*launch_count_out = 1;
	if (k_split_count == 1u) {
		return;
	}
	const struct trunk_relaxed_reduce_arguments reduce = {
		.partials_offset = partials_offset,
		.output_offset = { output_offsets[0], output_offsets[1] },
		.output_size = { output_sizes[0], output_sizes[1] },
		.row_count = row_count,
		.k_split_count = k_split_count,
		.output_scale = output_scale,
		.reserved_zero = 0,
	};
	const uint64_t result_count = (uint64_t)row_count * output_total;
	prepare_launch_fill(
		&launches[1], kernels->reduce,
		(uint32_t)((result_count + RELAXED_REDUCE_THREADS - 1u) /
			   RELAXED_REDUCE_THREADS),
		1, RELAXED_REDUCE_THREADS, 0, &reduce,
		TRUNK_RELAXED_REDUCE_ARGUMENT_BYTES);
	*launch_count_out = RELAXED_MOST_LAUNCHES;
}

static float power_of_two(int32_t exponent_field)
{
	const uint32_t bits = (uint32_t)exponent_field
			      << FLOAT32_EXPONENT_SHIFT;
	float power = 0.0f;
	memcpy(&power, &bits, sizeof(power));
	return power;
}

result_status relaxed_trunk_quantized_prepare(
	const struct relaxed_trunk_quantized_description *description,
	uint64_t arena_byte_count, const struct relaxed_linear_kernels *kernels,
	struct kernel_launch *launches_out, uint32_t *launch_count_out)
{
	static const char what[] = "a relaxed trunk product of quantized rows";
	static const char *const names[LINEAR_MERGED_MATRICES] = {
		"the first matrix", "the second matrix"
	};
	uint64_t matrix_byte_counts[LINEAR_MERGED_MATRICES] = { 0, 0 };
	if (launch_count_out == NULL) {
		LOG_ERROR("%s: a missing launch count", what);
		return STATUS_INVALID_ARGUMENT;
	}
	if (!prepare_arguments_are_given(what, description, kernels,
					 launches_out)) {
		return STATUS_INVALID_ARGUMENT;
	}
	const struct linear_matrix *const matrices = description->matrices;
	const uint32_t matrix_count = description->matrix_count,
		       row_count = description->row_count;
	const uint32_t input_size = matrices[0].shape.input_size,
		       k_split_count = description->k_split_count;
	const bool is_ordered = description->is_ordered == 1u,
		   is_merged = matrix_count == LINEAR_MERGED_MATRICES;
	if (matrix_count == 0 || matrix_count > LINEAR_MERGED_MATRICES ||
	    description->is_ordered > 1u) {
		LOG_ERROR("%s: %" PRIu32
			  " matrices (1 or 2), or an order that is no flag",
			  what, matrix_count);
		return STATUS_INVALID_ARGUMENT;
	}
	for (uint32_t matrix = 0; matrix < matrix_count; matrix++) {
		if (!matrix_is_read(what, names[matrix], &matrices[matrix],
				    &matrix_byte_counts[matrix])) {
			return STATUS_INVALID_ARGUMENT;
		}
	}
	if (is_merged && (matrices[1].shape.input_size != input_size ||
			  matrices[1].slice_count != matrices[0].slice_count ||
			  (uint64_t)matrices[0].shape.output_size +
					  matrices[1].shape.output_size >
				  UINT32_MAX)) {
		LOG_ERROR(
			"%s: the two matrices differ in input_size or slice_count, or have too many outputs",
			what);
		return STATUS_INVALID_ARGUMENT;
	}
	const uint32_t output_sizes[LINEAR_MERGED_MATRICES] = {
		matrices[0].shape.output_size,
		is_merged ? matrices[1].shape.output_size : 0u
	};
	const uint32_t output_total = output_sizes[0] + output_sizes[1];
	const uint32_t *const wide_kernels =
		is_ordered ? kernels->ordered_wide_quantized :
			     kernels->wide_quantized;
	const bool is_decode = row_count < RELAXED_WIDE_LEAST_ROWS;
	if (!counts_are_valid(what, row_count, k_split_count,
			      matrices[0].slice_count, input_size, output_total,
			      is_ordered) ||
	    !prepare_kernel_is_prepared(
		    what,
		    is_decode ? kernels->decode_quantized : wide_kernels[0],
		    is_decode ? KERNEL_RELAXED_DECODE_QUANTIZED :
				"a wide instance") ||
	    (k_split_count > 1u &&
	     !prepare_kernel_is_prepared(what, kernels->reduce,
					 KERNEL_RELAXED_REDUCE))) {
		return STATUS_INVALID_ARGUMENT;
	}
	const struct checked_range ranges[] = {
		{ "the operands", &description->operands,
		  linear_values_byte_count(row_count, input_size), false },
		{ "the first matrix", &matrices[0].stored,
		  matrix_byte_counts[0], true },
		{ "the first output", &description->outputs[0],
		  linear_values_byte_count(row_count, output_sizes[0]), false },
		{ "the second matrix", &matrices[1].stored,
		  matrix_byte_counts[1], true },
		{ "the second output", &description->outputs[1],
		  linear_values_byte_count(row_count, output_sizes[1]), false },
		{ "the partial sums", &description->partials,
		  linear_sums_byte_count((uint64_t)k_split_count * row_count,
					 output_total),
		  false },
	};
	const uint32_t merged_ranges = is_merged ? MERGED_RANGES :
						   SINGLE_RANGES;
	struct checked_range checked[COUNT_OF(ranges)];
	memcpy(checked, ranges, sizeof(ranges));
	checked[merged_ranges] =
		ranges[COUNT_OF(ranges) -
		       1u]; /* the partial sums, behind the ranges in use */
	if (!prepare_ranges_are_valid(what, checked,
				      merged_ranges +
					      (k_split_count > 1u ? 1u : 0u),
				      arena_byte_count)) {
		return STATUS_INVALID_ARGUMENT;
	}
	const struct trunk_relaxed_quantized_arguments record = {
		.weights_offset = { matrices[0].stored.byte_offset,
				    is_merged ? matrices[1].stored.byte_offset :
						0u },
		.output_offset = { description->outputs[0].byte_offset,
				   is_merged ?
					   description->outputs[1].byte_offset :
					   0u },
		.operands_offset = description->operands.byte_offset,
		.partials_offset = k_split_count > 1u ?
					   description->partials.byte_offset :
					   0u,
		.output_size = { output_sizes[0], output_sizes[1] },
		.input_size = input_size,
		.row_count = row_count,
		.k_split_count = k_split_count,
		.cone_layer = 0,
	};
	product_launches_fill(launches_out, launch_count_out, kernels,
			      wide_kernels, kernels->decode_quantized,
			      matrices[0].slice_count, output_total,
			      record.output_offset, output_sizes,
			      record.partials_offset, row_count, k_split_count,
			      power_of_two(CARRIED_OUTPUT_FIELD), input_size,
			      &record, TRUNK_RELAXED_QUANTIZED_ARGUMENT_BYTES);
	return STATUS_OK;
}

result_status relaxed_trunk_bfloat16_prepare(
	const struct relaxed_trunk_bfloat16_description *description,
	uint64_t arena_byte_count, const struct relaxed_linear_kernels *kernels,
	struct kernel_launch *launches_out, uint32_t *launch_count_out)
{
	static const char what[] = "a relaxed trunk product of bfloat16 rows";
	uint64_t matrix_byte_count = 0;
	if (launch_count_out == NULL) {
		LOG_ERROR("%s: a missing launch count", what);
		return STATUS_INVALID_ARGUMENT;
	}
	if (!prepare_arguments_are_given(what, description, kernels,
					 launches_out)) {
		return STATUS_INVALID_ARGUMENT;
	}
	const struct matrix_shape_record *const shape =
		&description->matrix.shape;
	const uint32_t row_count = description->row_count,
		       segment_count = description->segment_count;
	const uint32_t k_split_count = description->k_split_count;
	const int32_t reference = description->fold_reference;
	const bool is_decode = row_count < RELAXED_WIDE_LEAST_ROWS;
	const bool is_ordered = description->is_ordered == 1u;
	const uint32_t *const wide_kernels =
		is_ordered ? kernels->ordered_wide_bfloat16 :
			     kernels->wide_bfloat16;
	if (description->is_ordered > 1u) {
		LOG_ERROR("%s: an order that is no flag", what);
		return STATUS_INVALID_ARGUMENT;
	}
	if (!matrix_is_read(what, "the matrix", &description->matrix,
			    &matrix_byte_count) ||
	    !counts_are_valid(what, row_count, k_split_count,
			      description->matrix.slice_count,
			      shape->input_size, shape->output_size,
			      is_ordered) ||
	    !prepare_kernel_is_prepared(
		    what,
		    is_decode ? kernels->decode_bfloat16 : wide_kernels[0],
		    is_decode ? KERNEL_RELAXED_DECODE_BFLOAT16 :
				"a wide instance") ||
	    (k_split_count > 1u &&
	     !prepare_kernel_is_prepared(what, kernels->reduce,
					 KERNEL_RELAXED_REDUCE))) {
		return STATUS_INVALID_ARGUMENT;
	}
	if (segment_count == 0 ||
	    shape->output_size / RELAXED_WIDE_OUTPUTS % segment_count != 0 ||
	    reference < 1 || reference > FLOAT32_LARGEST_FIELD) {
		LOG_ERROR(
			"%s: %" PRIu32
			" segments (each whole blocks of %u outputs), or a fold reference of %" PRId32
			" that no float32 power holds",
			what, segment_count, RELAXED_WIDE_OUTPUTS, reference);
		return STATUS_INVALID_ARGUMENT;
	}
	const struct checked_range ranges[] = {
		{ "the matrix", &description->matrix.stored, matrix_byte_count,
		  true },
		{ "the values", &description->values,
		  linear_values_byte_count(row_count,
					   (uint64_t)segment_count *
						   shape->input_size),
		  false },
		{ "the output", &description->output,
		  linear_values_byte_count(row_count, shape->output_size),
		  false },
		{ "the partial sums", &description->partials,
		  linear_sums_byte_count((uint64_t)k_split_count * row_count,
					 shape->output_size),
		  false },
	};
	if (!prepare_ranges_are_valid(what, ranges,
				      COUNT_OF(ranges) -
					      (k_split_count > 1u ? 0u : 1u),
				      arena_byte_count)) {
		return STATUS_INVALID_ARGUMENT;
	}
	const struct trunk_relaxed_bfloat16_arguments record = {
		.weights_offset = description->matrix.stored.byte_offset,
		.values_offset = description->values.byte_offset,
		.output_offset = description->output.byte_offset,
		.partials_offset = k_split_count > 1u ?
					   description->partials.byte_offset :
					   0u,
		.output_size = shape->output_size,
		.input_size = shape->input_size,
		.row_count = row_count,
		.segment_count = segment_count,
		.k_split_count = k_split_count,
		.fold_reference = reference,
	};
	const uint64_t output_offsets[LINEAR_MERGED_MATRICES] = {
		record.output_offset, 0u
	};
	const uint32_t output_sizes[LINEAR_MERGED_MATRICES] = {
		shape->output_size, 0u
	};
	product_launches_fill(launches_out, launch_count_out, kernels,
			      wide_kernels, kernels->decode_bfloat16,
			      description->matrix.slice_count,
			      shape->output_size, output_offsets, output_sizes,
			      record.partials_offset, row_count, k_split_count,
			      power_of_two(reference), shape->input_size,
			      &record, TRUNK_RELAXED_BFLOAT16_ARGUMENT_BYTES);
	return STATUS_OK;
}

result_status
relaxed_trunk_cone_set(const struct relaxed_linear_kernels *kernels,
		       struct linear_range step, uint32_t cone_layer,
		       uint64_t arena_byte_count, struct kernel_launch *launch)
{
	const char *const what = "the relaxed trunk's cone";
	if (!prepare_arguments_are_given(what, kernels, launch, launch)) {
		return STATUS_INVALID_ARGUMENT;
	}
	bool is_wide = false;
	for (uint32_t tile = 0; tile < RELAXED_TILES; tile++) {
		is_wide = is_wide || launch->kernel == kernels->wide_quantized[tile] ||
			  launch->kernel == kernels->ordered_wide_quantized[tile] ||
			  launch->kernel == kernels->wide_bfloat16[tile] ||
			  launch->kernel == kernels->ordered_wide_bfloat16[tile];
	}
	/* the record's rows by its size (a card without the module has no handles to tell them apart) */
	const bool is_quantized = launch->argument_byte_count ==
				  TRUNK_RELAXED_QUANTIZED_ARGUMENT_BYTES;
	const bool is_bfloat16 = launch->argument_byte_count ==
				 TRUNK_RELAXED_BFLOAT16_ARGUMENT_BYTES;
	if (!is_wide || launch->kernel == BACKEND_NO_KERNEL ||
	    (!is_quantized && !is_bfloat16)) {
		return STATUS_OK; /* not the wide form: every row */
	}
	struct trunk_relaxed_quantized_arguments quantized;
	struct trunk_relaxed_bfloat16_arguments bfloat16;
	const uint32_t bytes = launch->argument_byte_count;
	memcpy(is_quantized ? (void *)&quantized : (void *)&bfloat16, launch->argument, bytes);
	const uint32_t row_count = is_quantized ? quantized.row_count : bfloat16.row_count;
	const uint64_t step_bytes = STEP_BLOCK_BYTES_VALUE +
				    (uint64_t)row_count * STEP_ROW_BYTES_VALUE;
	if (cone_layer < MODEL_DECODER_FIRST_LAYER ||
	    cone_layer >= MODEL_LAYERS || step.byte_count < step_bytes ||
	    step.byte_offset > arena_byte_count ||
	    step.byte_count > arena_byte_count - step.byte_offset) {
		LOG_ERROR("%s: layer %" PRIu32
			  " or a step area short of %" PRIu32 " rows",
			  what, cone_layer, row_count);
		return STATUS_INVALID_ARGUMENT;
	}
	if (is_quantized) {
		quantized.step_offset = step.byte_offset;
		quantized.cone_layer = cone_layer;
	} else {
		bfloat16.step_offset = step.byte_offset;
		bfloat16.cone_layer = cone_layer;
	}
	memcpy(launch->argument, is_quantized ? (const void *)&quantized : (const void *)&bfloat16, bytes);
	return STATUS_OK;
}
