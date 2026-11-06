/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * sampler.c: the launch of the sampler. Contract: sampler.h; the kernel: sampler_records.h.
 */
#include "anemone/sampler.h"

#include <inttypes.h>

#include "anemone/hooks.h"
#include "prepare_common.h"

#define COUNT_OF(table) ((uint32_t)(sizeof(table) / sizeof((table)[0])))

result_status
sampler_kernels_prepare(const struct backend_operations *backend,
			struct device_card *card,
			struct sampler_kernels_record *kernels_out)
{
    /* a backend that answers OK and gives no handle leaves this one, which the prepare function refuses */
    uint32_t handle = BACKEND_NO_KERNEL;
    if (backend == NULL || backend->kernel_prepare == NULL || card == NULL || kernels_out == NULL) {
	    LOG_ERROR("the kernel of the sampler: a missing argument");
	    return STATUS_INVALID_ARGUMENT;
    }
    const result_status status =
	    backend->kernel_prepare(card, KERNEL_FILE_SAMPLER, KERNEL_SAMPLER,
				    SAMPLER_ARGUMENT_BYTES, &handle);
    if (status != STATUS_OK) {
	    LOG_ERROR("the kernel of the sampler: %s is not prepared: %s",
		      KERNEL_SAMPLER, status_text(status));
	    return status;
    }
    kernels_out->sampler = handle;
    return STATUS_OK;
}

result_status
sampler_prepare_entry(const struct sampler_description *description,
		      uint64_t arena_byte_count,
		      const struct sampler_kernels_record *kernels,
		      struct kernel_launch *launch_out)
{
    static const char what[] = "the sampler";
    if (!prepare_arguments_are_given(what, description, kernels, launch_out) ||
	!prepare_kernel_is_prepared(what, kernels->sampler, KERNEL_SAMPLER) ||
	!prepare_reserved_is_zero(what, description->reserved_zero)) {
	    return STATUS_INVALID_ARGUMENT;
    }
    const uint32_t row_count = description->row_count;
    const uint32_t vocabulary_size = description->vocabulary_size;
    if (row_count == 0 || row_count > LINEAR_MAXIMUM_ROWS ||
	vocabulary_size == 0 || vocabulary_size > SAMPLER_MOST_VOCABULARY ||
	vocabulary_size % SAMPLER_TOKENS_PER_LOAD_VALUE != 0 ||
	description->session_capacity == 0) {
	    LOG_ERROR("%s: %" PRIu32 " rows, %" PRIu32 " tokens, %" PRIu32
		      " sessions: 1 to %u rows, %u to %u "
		      "tokens in %us, a session or more",
		      what, row_count, vocabulary_size,
		      description->session_capacity, LINEAR_MAXIMUM_ROWS,
		      SAMPLER_TOKENS_PER_LOAD_VALUE, SAMPLER_MOST_VOCABULARY,
		      SAMPLER_TOKENS_PER_LOAD_VALUE);
	    return STATUS_INVALID_ARGUMENT;
    }
    const uint64_t row_bytes =
	    linear_sums_byte_count(row_count, vocabulary_size);
    const struct checked_range ranges[] = {
	    { "the step area", &description->step,
	      STEP_BLOCK_BYTES_VALUE +
		      (uint64_t)row_count * STEP_ROW_BYTES_VALUE,
	      false },
	    { "the sessions", &description->sessions,
	      (uint64_t)description->session_capacity * STEP_SESSION_BYTES,
	      false },
	    { "the logits", &description->logits, row_bytes, false },
	    { "the weights", &description->weights, row_bytes, false },
	    { "the result area", &description->result,
	      STEP_RESULT_BYTES + (uint64_t)row_count * STEP_ROW_RESULT_BYTES,
	      false },
    };
    if (!prepare_ranges_are_valid(what, ranges, COUNT_OF(ranges),
				  arena_byte_count)) {
	    return STATUS_INVALID_ARGUMENT;
    }
    const struct sampler_arguments record = {
	    .step_offset = description->step.byte_offset,
	    .session_offset = description->sessions.byte_offset,
	    .logits_offset = description->logits.byte_offset,
	    .weights_offset = description->weights.byte_offset,
	    .result_offset = description->result.byte_offset,
	    .vocabulary_size = vocabulary_size,
	    .session_capacity = description->session_capacity,
    };
    prepare_launch_fill(launch_out, kernels->sampler, 1, row_count,
			SAMPLER_THREADS, 0, &record, SAMPLER_ARGUMENT_BYTES);
    return STATUS_OK;
}
