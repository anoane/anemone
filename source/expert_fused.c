/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * expert_fused.c: the launches of the routed experts' phase in fewer launches. Contract: expert_fused.h; the
 * kernels: expert_fused_records.h. A prepare function first hands what it stands for to linear.c's prepare functions
 * and discards their launches; then it checks what the fusion adds and every range against every other
 * (prepare_common.h), and only then fills the launch: nothing is written before every check has passed.
 */
#include "anemone/expert_fused.h"

#include <inttypes.h>
#include <stdbool.h>
#include <string.h>

#include "anemone/hooks.h"
#include "anemone/hyper_connection_records.h"
#include "anemone/model_layout.h"
#include "prepare_common.h"

#define OK STATUS_OK
#define COUNT_OF(table) ((uint32_t)(sizeof(table) / sizeof((table)[0])))
#define INSTANCES LINEAR_SLICE_INSTANCES
#define STAMPED_FAMILIES 2u                     /* the producer and the shared join, an instance per slice count */
#define KERNEL_COUNT (STAMPED_FAMILIES * INSTANCES + 1u)   /* then the join */
/* a block of the quantization: a pair of tiles */
#define PRODUCER_VALUES (EXPERT_PRODUCER_TILES * LINEAR_OUTPUTS_PER_TILE)
#define ARRIVAL_BYTES ((uint64_t)sizeof(uint32_t))
#define ORPHAN_ROWS 1u                          /* the producer's grid row of the pairs of no job */
#define SHARED_JOIN_THREADS_PER_SLICE LINEAR_THREADS_PER_WARP

/* Each count the kernels are stamped for, with the producer's and the shared join's function. */
static const struct fused_instance {
    uint32_t slice_count;
    const char *producer_name;
    const char *shared_join_name;
} fused_instances[] = {
#define LINEAR_SLICE_STAMP(slices)                               \
	{ (slices), KERNEL_EXPERT_GATE_UP_QUANTIZE_STEM #slices, \
	  KERNEL_EXPERT_SHARED_JOIN_STEM #slices },
	LINEAR_EVERY_SLICE_COUNT
#undef LINEAR_SLICE_STAMP
};
_Static_assert(
	COUNT_OF(fused_instances) == INSTANCES &&
		sizeof(struct expert_fused_kernels) ==
			KERNEL_COUNT * sizeof(uint32_t),
	"a handle for each instance of the producer and of the shared join, then the join's");

result_status expert_fused_kernels_prepare(
	const struct backend_operations *backend, struct device_card *card,
	uint32_t arithmetic_class, struct expert_fused_kernels *kernels_out)
{
	static const uint32_t argument_byte_counts[] = {
		EXPERT_GATE_UP_QUANTIZE_ARGUMENT_BYTES,
		EXPERT_SHARED_JOIN_ARGUMENT_BYTES, EXPERT_JOIN_ARGUMENT_BYTES
	}; /* a family's, in order */
	const bool is_native = arithmetic_class ==
			       ARITHMETIC_CLASS_SM_120_NATIVE;
	uint32_t handles[KERNEL_COUNT];
	if (backend == NULL || backend->kernel_prepare == NULL ||
	    card == NULL || kernels_out == NULL) {
		LOG_ERROR(
			"the kernels of the fused experts: a missing argument");
		return STATUS_INVALID_ARGUMENT;
	}
    for (uint32_t kernel = 0; kernel < KERNEL_COUNT; kernel++) {
        const bool is_producer = kernel < INSTANCES, is_join = kernel == KERNEL_COUNT - 1u;
	const char *const file_name =
		is_join	    ? KERNEL_FILE_HYPER_CONNECTION :
		is_producer ? (is_native ? KERNEL_FILE_EXPERT_LINEAR_NATIVE :
					   KERNEL_FILE_EXPERT_LINEAR) :
			      (is_native ? KERNEL_FILE_TRUNK_LINEAR_NATIVE :
					   KERNEL_FILE_TRUNK_LINEAR);
	const char *const function_name =
		is_join ? KERNEL_EXPERT_JOIN :
		is_producer ?
			  fused_instances[kernel].producer_name :
			  fused_instances[kernel - INSTANCES].shared_join_name;
	const uint32_t argument_byte_count = argument_byte_counts[kernel / INSTANCES];
        /* a backend that answers OK and gives no handle leaves this one, which every prepare function refuses */
	handles[kernel] = BACKEND_NO_KERNEL;
	const result_status status =
		backend->kernel_prepare(card, file_name, function_name,
					argument_byte_count, &handles[kernel]);
	if (status != OK) {
		LOG_ERROR(
			"the kernels of the fused experts: %s of %s is not prepared: %s",
			function_name, file_name, status_text(status));
		return status;
        }
    }
    memcpy(kernels_out, handles, sizeof(*kernels_out));
    return OK;
}

/* The place of a slice count among the stamped counts; linear.c's prepare has checked that the count is stamped. */
static uint32_t instance_of(uint32_t slice_count)
{
    uint32_t instance = 0;
    while (instance + 1u < INSTANCES && fused_instances[instance].slice_count != slice_count) {
        instance++;
    }
    return instance;
}

static bool arguments_are_given(const char *what, const void *description, const void *linear_kernels,
                                const void *kernels, const void *launch_out)
{
	return prepare_arguments_are_given(what, description, kernels,
					   launch_out) &&
	       prepare_arguments_are_given(what, linear_kernels, kernels,
					   launch_out);
}

result_status expert_producer_prepare(
	const struct linear_routed_experts_description *description,
	struct linear_range arrivals, uint64_t arena_byte_count,
	const struct linear_kernels_record *linear_kernels,
	const struct expert_fused_kernels *kernels,
	struct kernel_launch *launch_out)
{
    static const char what[] = "the producer of the routed experts";
    struct kernel_launch separate[LINEAR_ROUTED_EXPERTS_LAUNCH_COUNT];
    if (!arguments_are_given(what, description, linear_kernels, kernels, launch_out)) {
	    return STATUS_INVALID_ARGUMENT;
    }
    const result_status status = linear_routed_experts_prepare(
	    description, arena_byte_count, linear_kernels, separate);
    if (status != OK) {
        return status;
    }
    const struct matrix_shape_record *const shape = &description->gate_up_shape;
    const uint64_t rows = description->row_count, pairs = rows * description->pairs_per_row;
    if (rows > LINEAR_MAXIMUM_ROWS || pairs > EXPERT_PRODUCER_MOST_PAIRS ||
	shape->output_size % PRODUCER_VALUES != 0 ||
	description->job_count > UINT32_MAX - ORPHAN_ROWS) {
	    LOG_ERROR(
		    "%s: %" PRIu64 " rows, %" PRIu64 " pairs, %" PRIu32
		    " jobs, %" PRIu32 " outputs: it takes "
		    "1 to %u rows, at most %u pairs, a grid row behind the jobs and whole blocks of %u outputs",
		    what, rows, pairs, description->job_count,
		    shape->output_size, LINEAR_MAXIMUM_ROWS,
		    EXPERT_PRODUCER_MOST_PAIRS, PRODUCER_VALUES);
	    return STATUS_INVALID_ARGUMENT;
    }
    const uint32_t instance = instance_of(description->gate_up_slice_count);
    const uint32_t block_count = shape->output_size / LINEAR_VALUES_PER_BLOCK;
    const struct checked_range ranges[] = {
	    { "the job table", &description->jobs,
	      linear_jobs_byte_count(description->job_count), false },
	    { "the experts' matrices", &description->expert_weights,
	      description->expert_weights.byte_count, false },
	    { "the rows' codes", &description->codes,
	      linear_codes_byte_count(rows, shape->input_size), false },
	    { "the rows' exponents", &description->exponents,
	      linear_exponents_byte_count(rows, shape->input_size), false },
	    { "the rows' operands", &description->operands,
	      linear_values_byte_count(rows, shape->input_size), false },
	    { "the hidden values", &description->hidden,
	      linear_values_byte_count(pairs, shape->output_size), false },
	    { "the hidden codes", &description->hidden_codes,
	      linear_codes_byte_count(pairs, shape->output_size), false },
	    { "the hidden exponents", &description->hidden_exponents,
	      linear_exponents_byte_count(pairs, shape->output_size), false },
	    { "the hidden operands", &description->hidden_operands,
	      linear_values_byte_count(pairs, shape->output_size), false },
	    { "the arrival counters", &arrivals,
	      (uint64_t)description->job_count * block_count * ARRIVAL_BYTES,
	      false },
    };
    if (!prepare_kernel_is_prepared(what, kernels->gate_up_quantize[instance],
				    fused_instances[instance].producer_name) ||
	!prepare_ranges_are_valid(what, ranges, COUNT_OF(ranges),
				  arena_byte_count)) {
	    return STATUS_INVALID_ARGUMENT;
    }
    const struct expert_gate_up_quantize_arguments record = {
	    .jobs_offset = description->jobs.byte_offset,
	    .codes_offset = description->codes.byte_offset,
	    .exponents_offset = description->exponents.byte_offset,
	    .operands_offset = description->operands.byte_offset,
	    .hidden_offset = description->hidden.byte_offset,
	    .hidden_codes_offset = description->hidden_codes.byte_offset,
	    .hidden_exponents_offset =
		    description->hidden_exponents.byte_offset,
	    .hidden_operands_offset = description->hidden_operands.byte_offset,
	    .arrivals_offset = arrivals.byte_offset,
	    .output_size = shape->output_size,
	    .input_size = shape->input_size,
	    .pair_count = (uint32_t)pairs,
	    .pairs_per_row = description->pairs_per_row,
	    .activation_limit = description->activation_limit,
	    .reserved_zero = 0,
    };
    /* gate with up's launch, a block of 2 S warps for each tile of each job, and the row of the pairs of no job */
    const uint32_t block_threads = description->gate_up_slice_count *
				   EXPERT_GATE_UP_MATRICES *
				   LINEAR_THREADS_PER_WARP;
    prepare_launch_fill(launch_out, kernels->gate_up_quantize[instance],
			shape->output_size / LINEAR_OUTPUTS_PER_TILE,
			description->job_count + ORPHAN_ROWS, block_threads,
			block_threads * LINEAR_SHARED_BYTES_PER_THREAD, &record,
			EXPERT_GATE_UP_QUANTIZE_ARGUMENT_BYTES);
    return OK;
}

/* The counts of a join: the post's rows, a pair or more a row. */
static bool join_counts_are_valid(const char *what,
				  const struct expert_join_description *join)
{
	if (join->row_count == 0 || join->row_count > LINEAR_MAXIMUM_ROWS ||
	    join->pairs_per_row == 0) {
		LOG_ERROR("%s: %" PRIu32 " rows and %" PRIu32
			  " pairs per row: it takes 1 to %u rows and a pair or"
			  " more",
			  what, join->row_count, join->pairs_per_row,
			  LINEAR_MAXIMUM_ROWS);
		return false;
	}
    return true;
}

/* The ranges of a join into ranges_out, from its first entry: JOIN_RANGES of them, the row sums last; a join that
 * writes no row sums checks one fewer. */
enum { JOIN_RANGES = 6, DOWN_RANGES = 4 };   /* the shared join's down, its codes, exponents and operands */
static uint32_t join_ranges_of(const struct expert_join_description *join,
			       struct checked_range *ranges_out)
{
    const uint64_t rows = join->row_count;
    const struct checked_range ranges[JOIN_RANGES] = {
	    { "the pair outputs", &join->pair_output,
	      rows * join->pairs_per_row * MODEL_HIDDEN_ROW_BYTES, false },
	    { "the shared expert's output", &join->shared_output,
	      rows * MODEL_HIDDEN_ROW_BYTES, false },
	    { "the streams", &join->streams, rows * MODEL_STREAMS_ROW_BYTES,
	      false },
	    { "the post", &join->post, rows * MODEL_SITE_ROW_BYTES, false },
	    { "the comb", &join->comb, rows * MODEL_COMB_ROW_BYTES, false },
	    { "the row sums", &join->row_sums,
	      linear_sums_byte_count(rows, MODEL_HIDDEN), false },
    };
    memcpy(ranges_out, ranges, sizeof(ranges));
    return join->row_sums.byte_count != 0 ? JOIN_RANGES : JOIN_RANGES - 1u;
}

/* The record's offset of a join's row sums: theirs, or none where it writes none. */
static uint64_t row_sums_offset_of(const struct expert_join_description *join)
{
	return join->row_sums.byte_count != 0 ? join->row_sums.byte_offset :
						EXPERT_JOIN_NO_ROW_SUMS;
}

result_status
expert_join_prepare(const struct expert_join_description *description,
		    uint64_t arena_byte_count,
		    const struct expert_fused_kernels *kernels,
		    struct kernel_launch *launch_out)
{
    static const char what[] = "the join of the FFN site";
    struct checked_range ranges[JOIN_RANGES];
    if (!prepare_arguments_are_given(what, description, kernels, launch_out) ||
	!prepare_kernel_is_prepared(what, kernels->join, KERNEL_EXPERT_JOIN) ||
	!join_counts_are_valid(what, description)) {
	    return STATUS_INVALID_ARGUMENT;
    }
    const uint32_t range_count = join_ranges_of(description, ranges);
    if (!prepare_ranges_are_valid(what, ranges, range_count,
				  arena_byte_count)) {
	    return STATUS_INVALID_ARGUMENT;
    }
    const struct expert_join_arguments record = {
	    .pair_output_offset = description->pair_output.byte_offset,
	    .shared_output_offset = description->shared_output.byte_offset,
	    .streams_offset = description->streams.byte_offset,
	    .post_offset = description->post.byte_offset,
	    .comb_offset = description->comb.byte_offset,
	    .row_sums_offset = row_sums_offset_of(description),
	    .pairs_per_row = description->pairs_per_row,
	    .reserved_zero = 0,
    };
    prepare_launch_fill(launch_out, kernels->join, HYPER_PIECES,
			description->row_count, HYPER_PIECE_THREADS, 0, &record,
			EXPERT_JOIN_ARGUMENT_BYTES);
    return OK;
}

result_status expert_shared_join_prepare(
	const struct expert_shared_join_description *description,
	uint64_t arena_byte_count,
	const struct linear_kernels_record *linear_kernels,
	const struct expert_fused_kernels *kernels,
	struct kernel_launch *launch_out)
{
    static const char what[] = "the shared join of the FFN site";
    struct checked_range ranges[DOWN_RANGES + JOIN_RANGES];
    struct kernel_launch separate;
    if (!arguments_are_given(what, description, linear_kernels, kernels, launch_out)) {
	    return STATUS_INVALID_ARGUMENT;
    }
    const struct expert_join_description *const join = &description->join;
    const struct linear_trunk_quantized_description down = {
	    .matrices = { description->down },
	    .codes = description->codes,
	    .exponents = description->exponents,
	    .operands = description->operands,
	    .outputs = { join->shared_output },
	    .matrix_count = 1,
	    .row_count = join->row_count,
    };
    const result_status status = linear_trunk_quantized_prepare(
	    &down, arena_byte_count, linear_kernels, &separate);
    if (status != OK) {
        return status;
    }
    const struct matrix_shape_record *const shape = &description->down.shape;
    const uint32_t instance = instance_of(description->down.slice_count);
    if (!join_counts_are_valid(what, join) ||
	!prepare_kernel_is_prepared(
		what, kernels->shared_join[instance],
		fused_instances[instance].shared_join_name)) {
	    return STATUS_INVALID_ARGUMENT;
    }
    if (shape->output_size != MODEL_HIDDEN) {
	    LOG_ERROR("%s: down has %" PRIu32
		      " outputs: the streams are %u wide",
		      what, shape->output_size, MODEL_HIDDEN);
	    return STATUS_INVALID_ARGUMENT;
    }
    const uint64_t rows = join->row_count;
    const struct checked_range down_ranges[DOWN_RANGES] = {
	    { "down", &description->down.stored,
	      description->down.stored.byte_count, true },
	    { "the hidden codes", &description->codes,
	      linear_codes_byte_count(rows, shape->input_size), false },
	    { "the hidden exponents", &description->exponents,
	      linear_exponents_byte_count(rows, shape->input_size), false },
	    { "the hidden operands", &description->operands,
	      linear_values_byte_count(rows, shape->input_size), false },
    };
    memcpy(ranges, down_ranges, sizeof(down_ranges));
    const uint32_t range_count = DOWN_RANGES + join_ranges_of(join, ranges + DOWN_RANGES);
    if (!prepare_ranges_are_valid(what, ranges, range_count,
				  arena_byte_count)) {
	    return STATUS_INVALID_ARGUMENT;
    }
    const struct expert_shared_join_arguments record = {
	    .weights_offset = description->down.stored.byte_offset,
	    .codes_offset = description->codes.byte_offset,
	    .exponents_offset = description->exponents.byte_offset,
	    .operands_offset = description->operands.byte_offset,
	    .output_offset = join->shared_output.byte_offset,
	    .pair_output_offset = join->pair_output.byte_offset,
	    .streams_offset = join->streams.byte_offset,
	    .post_offset = join->post.byte_offset,
	    .comb_offset = join->comb.byte_offset,
	    .row_sums_offset = row_sums_offset_of(join),
	    .input_size = shape->input_size,
	    .row_count = join->row_count,
	    .pairs_per_row = join->pairs_per_row,
	    .reserved_zero = 0,
    };
    const uint32_t block_threads = description->down.slice_count * SHARED_JOIN_THREADS_PER_SLICE;
    prepare_launch_fill(launch_out, kernels->shared_join[instance],
			MODEL_HIDDEN / LINEAR_OUTPUTS_PER_TILE, 1,
			block_threads,
			block_threads * LINEAR_SHARED_BYTES_PER_THREAD, &record,
			EXPERT_SHARED_JOIN_ARGUMENT_BYTES);
    return OK;
}
