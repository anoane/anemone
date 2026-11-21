/* SPDX-License-Identifier: AGPL-3.0-or-later */
#include "peer_row_copy.h"

#include <stddef.h>
#include "common/hooks.h"
#include "engine/library/prepare_checks.h"

#define COUNT_OF(table) ((uint32_t)(sizeof(table) / sizeof((table)[0])))

static uint64_t row_span(uint64_t stride, uint32_t rows, uint64_t bytes)
{
	return rows > 1u && stride > (UINT64_MAX - bytes) / (rows - 1u) ?
		       UINT64_MAX :
		       (rows - 1u) * stride + bytes;
}

result_status
head_offload_prepare(const struct head_offload_description *description,
		     uint64_t source_arena_byte_count, uint32_t kernel,
		     struct kernel_launch *launch_out)
{
	const char *const what = "head offload row copy";
	if (description == NULL || launch_out == NULL) {
		LOG_ERROR("%s: a missing argument", what);
		return STATUS_INVALID_ARGUMENT;
	}
	const struct head_offload_description *const d = description;
	if (d->row_count == 0 || d->row_count > LINEAR_MAXIMUM_ROWS ||
	    d->row_bytes == 0 || d->row_bytes % HEAD_OFFLOAD_WORD_BYTES != 0 ||
	    d->source_stride % HEAD_OFFLOAD_WORD_BYTES != 0 ||
	    d->destination_stride % HEAD_OFFLOAD_WORD_BYTES != 0 ||
	    d->column_bytes % HEAD_OFFLOAD_WORD_BYTES != 0 ||
	    d->source_stride < d->row_bytes ||
	    d->destination_stride < (uint64_t)d->column_bytes + d->row_bytes) {
		LOG_ERROR("%s: invalid row bounds or alignment", what);
		return STATUS_INVALID_ARGUMENT;
	}
	const struct checked_range local[] = {
		{ "source rows", &d->source,
		  row_span(d->source_stride, d->row_count, d->row_bytes),
		  false },
		{ "peer address word", &d->peer_word, sizeof(uint64_t), false },
		{ "step area", &d->step,
		  STEP_BLOCK_BYTES_VALUE +
			  (uint64_t)d->row_count * STEP_ROW_BYTES_VALUE,
		  false }
	};
	const struct checked_range peer = {
		"destination rows", &d->destination,
		row_span(d->destination_stride, d->row_count,
			 (uint64_t)d->column_bytes + d->row_bytes),
		false
	};
	if (!prepare_reserved_is_zero(what, d->reserved_zero) ||
	    !prepare_kernel_is_prepared(what, kernel,
					KERNEL_HEAD_OFFLOAD_COPY) ||
	    !prepare_ranges_are_valid(what, local, COUNT_OF(local),
				      source_arena_byte_count) ||
	    !prepare_ranges_are_valid(what, &peer, 1u,
				      d->destination_arena_byte_count))
		return STATUS_INVALID_ARGUMENT;
	const struct head_offload_arguments arguments = {
		d->source.byte_offset,
		d->peer_word.byte_offset,
		d->destination.byte_offset + d->column_bytes,
		d->step.byte_offset,
		d->source_stride,
		d->destination_stride,
		d->row_bytes,
		d->row_count,
		{ 0, 0 }
	};
	const uint32_t words = d->row_bytes / HEAD_OFFLOAD_WORD_BYTES;
	const uint32_t blocks =
		(words + HEAD_OFFLOAD_THREADS - 1u) / HEAD_OFFLOAD_THREADS;
	prepare_launch_fill(launch_out, kernel,
			    blocks < HEAD_OFFLOAD_MOST_BLOCKS ?
				    blocks :
				    HEAD_OFFLOAD_MOST_BLOCKS,
			    d->row_count, HEAD_OFFLOAD_THREADS, 0, &arguments,
			    sizeof(arguments));
	return STATUS_OK;
}
