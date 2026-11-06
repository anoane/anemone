/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * hyper_connection.c: the launches of the hyper-connection sites. Contract: hyper_connection.h; the kernels:
 * hyper_connection_records.h.
 */
#include "anemone/hyper_connection.h"

#include <inttypes.h>
#include <string.h>

#include "anemone/hooks.h"
#include "anemone/step_records.h"
#include "prepare_common.h"

#define COUNT_OF(table) ((uint32_t)(sizeof(table) / sizeof((table)[0])))
#define SCALE_BYTES ((uint64_t)HYPER_SCALES * sizeof(float))
#define SUMS_ROW_BYTES ((uint64_t)MODEL_HIDDEN * sizeof(float))
#define DRAFT_KERNELS 2u                            /* draft_inputs, final_kept, behind the table's: draft.h's */

/* The functions of kernel_hyper_connection, in the order of the members of struct hyper_kernels_record. */
static const struct hyper_function {
    const char *name;
    uint32_t argument_byte_count;
} hyper_functions[] = {
	{ KERNEL_HYPER_MIX_FINISH, HYPER_FINISH_ARGUMENT_BYTES },
	{ KERNEL_HYPER_FINAL, HYPER_FINISH_ARGUMENT_BYTES },
	{ KERNEL_HYPER_POST, HYPER_POST_ARGUMENT_BYTES },
	{ KERNEL_HYPER_EMBED, HYPER_EMBED_ARGUMENT_BYTES },
};
_Static_assert(sizeof(struct hyper_kernels_record) ==
		       sizeof(uint32_t) *
			       (COUNT_OF(hyper_functions) + DRAFT_KERNELS),
	       "a handle for each function, in their order, then the draft's");

result_status hyper_kernels_prepare(const struct backend_operations *backend,
				    struct device_card *card,
				    struct hyper_kernels_record *kernels_out)
{
    uint32_t handles[COUNT_OF(hyper_functions) + DRAFT_KERNELS];
    if (backend == NULL || backend->kernel_prepare == NULL || card == NULL || kernels_out == NULL) {
	    LOG_ERROR(
		    "the kernels of the hyper-connection sites: a missing argument");
	    return STATUS_INVALID_ARGUMENT;
    }
    for (uint32_t draft = 0; draft < DRAFT_KERNELS; draft++) {
	    handles[COUNT_OF(hyper_functions) + draft] = BACKEND_NO_KERNEL;
    }
    for (uint32_t function = 0; function < COUNT_OF(hyper_functions); function++) {
        /* a backend that answers OK and gives no handle leaves this one, which every prepare function refuses */
	handles[function] = BACKEND_NO_KERNEL;
	const result_status status = backend->kernel_prepare(
		card, KERNEL_FILE_HYPER_CONNECTION,
		hyper_functions[function].name,
		hyper_functions[function].argument_byte_count,
		&handles[function]);
	if (status != STATUS_OK) {
		LOG_ERROR(
			"the kernels of the hyper-connection sites: %s is not prepared: %s",
			hyper_functions[function].name, status_text(status));
		return status;
	}
    }
    memcpy(kernels_out, handles, sizeof(*kernels_out));
    return STATUS_OK;
}

/* The row count, 1 to most, and the reserved field of a description: true, or false behind the one error line of the
 * refusal. */
static bool rows_are_valid(const char *what, uint32_t row_count, uint32_t most, uint32_t reserved_zero)
{
    if (row_count == 0 || row_count > most) {
	    LOG_ERROR("%s: %" PRIu32 " rows: a launch takes 1 to %" PRIu32,
		      what, row_count, most);
	    return false;
    }
    return prepare_reserved_is_zero(what, reserved_zero);
}

result_status
hyper_site_prepare(const struct hyper_site_description *description,
		   uint64_t arena_byte_count,
		   const struct hyper_kernels_record *kernels,
		   struct kernel_launch *launches_out)
{
    static const char what[] = "a hyper-connection site";
    if (!prepare_arguments_are_given(what, description, kernels,
				     launches_out)) {
	    return STATUS_INVALID_ARGUMENT;
    }
    const uint64_t rows = description->row_count;
    const struct linear_range *const sums = &description->routed_sums;
    const bool is_combined = sums->byte_offset != 0 || sums->byte_count != 0;
    const struct checked_range ranges[] = {
	    { "the streams", &description->streams,
	      rows * MODEL_STREAMS_ROW_BYTES, false },
	    { "the carried pre", &description->carried_pre,
	      rows * MODEL_SITE_ROW_BYTES, false },
	    { "the mixes", &description->mixes, rows * MODEL_MIXES_ROW_BYTES,
	      false },
	    { "the scale", &description->scale, SCALE_BYTES, true },
	    { "the base", &description->base, MODEL_MIXES_ROW_BYTES, true },
	    { "the norm's weight", &description->norm_weight,
	      MODEL_HIDDEN_ROW_BYTES, true },
	    { "the post", &description->post, rows * MODEL_SITE_ROW_BYTES,
	      false },
	    { "the comb", &description->comb, rows * MODEL_COMB_ROW_BYTES,
	      false },
	    { "the sublayer's input", &description->hidden,
	      rows * MODEL_HIDDEN_ROW_BYTES, false },
	    { "the sublayer's output", &description->output,
	      rows * MODEL_HIDDEN_ROW_BYTES, false },
	    { "the routed sums", sums, rows * SUMS_ROW_BYTES,
	      false }, /* last: left out where there are none */
    };
    if (!prepare_kernel_is_prepared(what, kernels->mix_finish,
				    KERNEL_HYPER_MIX_FINISH) ||
	!prepare_kernel_is_prepared(what, kernels->post, KERNEL_HYPER_POST) ||
	!rows_are_valid(what, description->row_count, MODEL_WIDE_MOST_ROWS,
			description->reserved_zero) ||
	!prepare_ranges_are_valid(what, ranges,
				  COUNT_OF(ranges) - (is_combined ? 0u : 1u),
				  arena_byte_count)) {
	    return STATUS_INVALID_ARGUMENT;
    }
    const struct hyper_finish_arguments finish = {
	    .streams_offset = description->streams.byte_offset,
	    .pre_offset = description->carried_pre.byte_offset,
	    .mixes_offset = description->mixes.byte_offset,
	    .scale_offset = description->scale.byte_offset,
	    .base_offset = description->base.byte_offset,
	    .post_offset = description->post.byte_offset,
	    .comb_offset = description->comb.byte_offset,
	    .weight_offset = description->norm_weight.byte_offset,
	    .hidden_offset = description->hidden.byte_offset
    };
    const struct hyper_post_arguments post = {
	    .streams_offset = description->streams.byte_offset,
	    .post_offset = description->post.byte_offset,
	    .comb_offset = description->comb.byte_offset,
	    .output_offset = description->output.byte_offset,
	    .sums_offset = sums->byte_offset,
	    .is_combined = is_combined ? 1u : 0u,
	    .reserved_zero = 0
    };
    prepare_launch_fill(&launches_out[HYPER_SITE_MIX_FINISH],
			kernels->mix_finish, 1, description->row_count,
			HYPER_FINISH_THREADS, 0, &finish,
			HYPER_FINISH_ARGUMENT_BYTES);
    prepare_launch_fill(&launches_out[HYPER_SITE_POST], kernels->post,
			HYPER_PIECES, description->row_count,
			HYPER_PIECE_THREADS, 0, &post,
			HYPER_POST_ARGUMENT_BYTES);
    return STATUS_OK;
}

result_status
hyper_embed_prepare(const struct hyper_embed_description *description,
		    uint64_t arena_byte_count,
		    const struct hyper_kernels_record *kernels,
		    struct kernel_launch *launch_out)
{
    static const char what[] = "the embedding into the streams";
    if (!prepare_arguments_are_given(what, description, kernels, launch_out)) {
	    return STATUS_INVALID_ARGUMENT;
    }
    const uint64_t rows = description->row_count;
    const struct checked_range ranges[] = {
	    { "the step area", &description->step,
	      STEP_BLOCK_BYTES_VALUE + rows * STEP_ROW_BYTES_VALUE, false },
	    { "the embedding", &description->embedding,
	      (uint64_t)description->vocabulary_size * MODEL_HIDDEN_ROW_BYTES,
	      true },
	    { "the streams", &description->streams,
	      rows * MODEL_STREAMS_ROW_BYTES, false },
	    { "the carried pre", &description->carried_pre,
	      rows * MODEL_SITE_ROW_BYTES, false },
    };
    if (!prepare_kernel_is_prepared(what, kernels->embed, KERNEL_HYPER_EMBED) ||
	!rows_are_valid(what, description->row_count, MODEL_WIDE_MOST_ROWS,
			0)) {
	    return STATUS_INVALID_ARGUMENT;
    }
    if (description->vocabulary_size == 0) {
	    LOG_ERROR("%s: an embedding of no tokens", what);
	    return STATUS_INVALID_ARGUMENT;
    }
    if (!prepare_ranges_are_valid(what, ranges, COUNT_OF(ranges),
				  arena_byte_count)) {
	    return STATUS_INVALID_ARGUMENT;
    }
    const struct hyper_embed_arguments record = {
	    .step_offset = description->step.byte_offset,
	    .embedding_offset = description->embedding.byte_offset,
	    .streams_offset = description->streams.byte_offset,
	    .pre_offset = description->carried_pre.byte_offset,
	    .vocabulary_size = description->vocabulary_size,
	    .reserved_zero = 0
    };
    prepare_launch_fill(launch_out, kernels->embed, HYPER_PIECES,
			description->row_count, HYPER_PIECE_THREADS, 0, &record,
			HYPER_EMBED_ARGUMENT_BYTES);
    return STATUS_OK;
}

result_status
hyper_final_prepare(const struct hyper_final_description *description,
		    uint64_t arena_byte_count,
		    const struct hyper_kernels_record *kernels,
		    struct kernel_launch *launch_out)
{
    static const char what[] = "the final collapse";
    if (!prepare_arguments_are_given(what, description, kernels, launch_out)) {
	    return STATUS_INVALID_ARGUMENT;
    }
    const uint64_t rows = description->row_count;
    const struct linear_range *const collapse = &description->collapse;
    const bool is_kept = collapse->byte_offset != 0 || collapse->byte_count != 0;
    const struct checked_range ranges[] = {
	    { "the streams", &description->streams,
	      rows * MODEL_STREAMS_ROW_BYTES, false },
	    { "the carried pre", &description->carried_pre,
	      rows * MODEL_SITE_ROW_BYTES, false },
	    { "the norm's weight", &description->norm_weight,
	      MODEL_HIDDEN_ROW_BYTES, true },
	    { "the head's input", &description->hidden,
	      rows * MODEL_HIDDEN_ROW_BYTES, false },
	    { "the collapse", collapse, rows * MODEL_HIDDEN_ROW_BYTES,
	      false }, /* last: left out where none */
    };
    const uint32_t kernel = is_kept ? kernels->final_kept : kernels->final_collapse;
    if (!prepare_kernel_is_prepared(what, kernel,
				    is_kept ? KERNEL_HYPER_FINAL_KEPT :
					      KERNEL_HYPER_FINAL) ||
	!rows_are_valid(what, description->row_count, LINEAR_MAXIMUM_ROWS,
			description->reserved_zero) ||
	!prepare_ranges_are_valid(what, ranges,
				  COUNT_OF(ranges) - (is_kept ? 0u : 1u),
				  arena_byte_count)) {
	    return STATUS_INVALID_ARGUMENT;
    }
    if (is_kept && collapse->byte_offset == 0) {
	    LOG_ERROR(
		    "%s: the collapse at byte 0, the arena's first weight, where no state array lies",
		    what);
	    return STATUS_INVALID_ARGUMENT;
    }
    const struct hyper_finish_arguments record = {
	    .streams_offset = description->streams.byte_offset,
	    .pre_offset = description->carried_pre.byte_offset,
	    .comb_offset = collapse->byte_offset,
	    .weight_offset = description->norm_weight.byte_offset,
	    .hidden_offset = description->hidden.byte_offset
    };
    prepare_launch_fill(launch_out, kernel, 1, description->row_count,
			HYPER_ROW_THREADS, 0, &record,
			HYPER_FINISH_ARGUMENT_BYTES);
    return STATUS_OK;
}

result_status hyper_draft_inputs_prepare(
	const struct hyper_draft_inputs_description *description,
	uint64_t arena_byte_count, const struct hyper_kernels_record *kernels,
	struct kernel_launch *launch_out)
{
    static const char what[] = "the draft's inputs";
    if (!prepare_arguments_are_given(what, description, kernels, launch_out) ||
	!prepare_kernel_is_prepared(what, kernels->draft_inputs,
				    KERNEL_HYPER_DRAFT_INPUTS) ||
	!prepare_reserved_is_zero(what, description->reserved_zero)) {
	    return STATUS_INVALID_ARGUMENT;
    }
    const bool is_mean = description->mode == HYPER_STREAM_MEAN;
    const struct linear_range *const weight = &description->weight;
    const bool has_weight = weight->byte_offset != 0 || weight->byte_count != 0;
    if (description->row_count == 0 ||
	description->row_count > MODEL_WIDE_MOST_ROWS ||
	(!is_mean && description->mode != HYPER_ROW_NORM) ||
	(is_mean ? description->tap >= MODEL_DRAFT_TAPS || has_weight :
		   description->tap != 0)) {
	    LOG_ERROR("%s: %" PRIu32 " rows, mode %" PRIu32 ", tap %" PRIu32
		      "%s: 1 to %u rows; mode %u (a tap "
		      "below %u, no weight) or %u (tap 0, the norm's weight)",
		      what, description->row_count, description->mode,
		      description->tap, has_weight ? " with a weight" : "",
		      MODEL_WIDE_MOST_ROWS, (uint32_t)HYPER_STREAM_MEAN,
		      MODEL_DRAFT_TAPS, (uint32_t)HYPER_ROW_NORM);
	    return STATUS_INVALID_ARGUMENT;
    }
    const uint64_t rows = description->row_count;
    const struct checked_range ranges[] = {
	    { "the input", &description->input,
	      rows * (is_mean ? MODEL_STREAMS_ROW_BYTES :
				MODEL_HIDDEN_ROW_BYTES),
	      false },
	    { "the output", &description->output,
	      rows * (is_mean ? MODEL_DRAFT_TAP_ROW_BYTES :
				MODEL_HIDDEN_ROW_BYTES),
	      false },
	    { "the norm's weight", weight, MODEL_HIDDEN_ROW_BYTES,
	      true }, /* last: left out for the mean */
    };
    if (!prepare_ranges_are_valid(what, ranges,
				  COUNT_OF(ranges) - (is_mean ? 1u : 0u),
				  arena_byte_count)) {
	    return STATUS_INVALID_ARGUMENT;
    }
    const struct hyper_draft_inputs_arguments record = {
	    .input_offset = description->input.byte_offset,
	    .weight_offset = is_mean ? 0 : weight->byte_offset,
	    .output_offset = description->output.byte_offset,
	    .output_row_values = is_mean ? MODEL_DRAFT_FEATURES : MODEL_HIDDEN,
	    .column_offset = description->tap * MODEL_HIDDEN,
	    .mode = description->mode,
	    .reserved_zero = 0
    };
    prepare_launch_fill(launch_out, kernels->draft_inputs, 1,
			description->row_count, HYPER_ROW_THREADS, 0, &record,
			HYPER_DRAFT_INPUTS_ARGUMENT_BYTES);
    return STATUS_OK;
}
