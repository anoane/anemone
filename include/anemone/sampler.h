/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * sampler.h: the launch of the sampler (sampler_records.h), the last of a step on the card that samples. Contract:
 * linear.h's for the kernels and the prepare functions (once at load, arithmetic on a description, one error line for
 * a refusal, the launch untouched). Refused with STATUS_INVALID_ARGUMENT: a NULL argument; the kernel not
 * prepared; a reserved field not zero; a row_count of 0 or above LINEAR_MAXIMUM_ROWS, a vocabulary_size of 0,
 * above SAMPLER_MOST_VOCABULARY or no multiple of SAMPLER_TOKENS_PER_LOAD_VALUE, no session; a range that
 * breaks a rule of prepare_common.h.
 */
#ifndef SAMPLER_H
#define SAMPLER_H

#include <stdint.h>

#include "anemone/backend_seam.h"
#include "anemone/linear.h"
#include "anemone/sampler_records.h"
#include "anemone/status.h"

struct sampler_kernels_record {
	uint32_t sampler;
};
/* Prepares the kernel on one card, as linear_kernels_prepare does. */
MUST_CHECK result_status sampler_kernels_prepare(
	const struct backend_operations *backend, struct device_card *card,
	struct sampler_kernels_record *kernels_out);

struct sampler_description {
	struct linear_range step; /* read: the step area, for row_count rows */
	struct linear_range
		sessions; /* read: session settings [session_capacity] */
	struct linear_range
		logits; /* read: float32 [row_count][vocabulary_size] */
	struct linear_range
		weights; /* scratch: float32 [row_count][vocabulary_size] */
	struct linear_range
		result; /* written: the result area, for row_count rows */
	uint32_t vocabulary_size;
	uint32_t session_capacity;
	uint32_t row_count;
	uint32_t reserved_zero;
};
MUST_CHECK result_status sampler_prepare_entry(
	const struct sampler_description *description,
	uint64_t arena_byte_count, const struct sampler_kernels_record *kernels,
	struct kernel_launch *launch_out);

#endif
