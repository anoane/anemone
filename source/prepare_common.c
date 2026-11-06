/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * prepare_common.c: the checks every prepare function makes. Contract: prepare_common.h. Moved out of linear.c as
 * they stood, so that every family of kernels refuses a description for one reason with one error line.
 */
#include "prepare_common.h"

#include <inttypes.h>
#include <string.h>

#include "anemone/hooks.h"

enum launch_dimension { DIMENSION_X, DIMENSION_Y, DIMENSION_Z, DIMENSION_COUNT };

_Static_assert(DIMENSION_COUNT == LAUNCH_DIMENSIONS,
	       "a launch has the three dimensions of a grid");

bool prepare_arguments_are_given(const char *what, const void *description,
				 const void *kernels, const void *launches_out)
{
    if (description == NULL || kernels == NULL || launches_out == NULL) {
	    LOG_ERROR("%s: a missing argument", what);
	    return false;
    }
    return true;
}

bool prepare_kernel_is_prepared(const char *what, uint32_t kernel,
				const char *function_name)
{
	if (kernel == BACKEND_NO_KERNEL) {
		LOG_ERROR("%s: the kernel %s is not prepared", what,
			  function_name);
		return false;
	}
    return true;
}

bool prepare_reserved_is_zero(const char *what, uint32_t reserved_zero)
{
    if (reserved_zero != 0) {
	    LOG_ERROR("%s: a reserved field of the description is not zero",
		      what);
	    return false;
    }
    return true;
}

bool prepare_ranges_are_valid(const char *what,
			      const struct checked_range *ranges,
			      uint32_t range_count, uint64_t arena_byte_count)
{
    for (uint32_t range_index = 0; range_index < range_count; range_index++) {
        const struct checked_range *const checked = &ranges[range_index];
        const uint64_t byte_offset = checked->range->byte_offset;
        const uint64_t byte_count = checked->range->byte_count;
	if (byte_offset % LINEAR_OFFSET_ALIGNMENT_BYTES != 0) {
		LOG_ERROR("%s: the range of %s begins at byte %" PRIu64
			  ": an offset is a multiple of %u",
			  what, checked->name, byte_offset,
			  LINEAR_OFFSET_ALIGNMENT_BYTES);
		return false;
	}
	if (byte_offset > arena_byte_count || byte_count > arena_byte_count - byte_offset) {
		LOG_ERROR("%s: the range of %s, %" PRIu64
			  " bytes from byte %" PRIu64
			  ", leaves the arena of %" PRIu64 " bytes",
			  what, checked->name, byte_count, byte_offset,
			  arena_byte_count);
		return false;
        }
        if (checked->is_matrix && byte_count != checked->touched_byte_count) {
		LOG_ERROR("%s: the range of %s holds %" PRIu64
			  " bytes: its format stores %" PRIu64 " for its shape",
			  what, checked->name, byte_count,
			  checked->touched_byte_count);
		return false;
        }
        if (!checked->is_matrix &&
            (checked->touched_byte_count == UINT64_MAX || byte_count < checked->touched_byte_count)) {
		LOG_ERROR("%s: the range of %s holds %" PRIu64
			  " bytes: the launches touch %" PRIu64 " there",
			  what, checked->name, byte_count,
			  checked->touched_byte_count);
		return false;
        }
    }
    for (uint32_t first_index = 0; first_index < range_count; first_index++) {
	    const struct linear_range *const first = ranges[first_index].range;
	    for (uint32_t second_index = first_index + 1;
		 second_index < range_count; second_index++) {
		    const struct linear_range *const second =
			    ranges[second_index].range;
		    if (first->byte_offset <
				second->byte_offset + second->byte_count &&
			second->byte_offset <
				first->byte_offset + first->byte_count) {
			    LOG_ERROR(
				    "%s: the ranges of %s and of %s share a byte",
				    what, ranges[first_index].name,
				    ranges[second_index].name);
			    return false;
		    }
        }
    }
    return true;
}

void prepare_launch_fill(struct kernel_launch *launch, uint32_t kernel,
			 uint32_t grid_columns, uint32_t grid_rows,
			 uint32_t block_threads,
			 uint32_t dynamic_shared_byte_count, const void *record,
			 uint32_t record_byte_count)
{
	ASSERT_TEXT(
		record_byte_count <= LAUNCH_ARGUMENT_BYTES &&
			grid_columns != 0 && grid_rows != 0,
		"record_byte_count <= ANEMONE_LAUNCH_ARGUMENT_BYTES && grid_columns != 0 && grid_rows != 0");
	memset(launch, 0, sizeof(*launch));
	launch->kernel = kernel;
	launch->grid[DIMENSION_X] = grid_columns;
	launch->grid[DIMENSION_Y] = grid_rows;
	launch->grid[DIMENSION_Z] = 1;
	launch->block[DIMENSION_X] = block_threads;
	launch->block[DIMENSION_Y] = 1;
	launch->block[DIMENSION_Z] = 1;
	launch->dynamic_shared_byte_count = dynamic_shared_byte_count;
	launch->argument_byte_count = record_byte_count;
	memcpy(launch->argument, record, record_byte_count);
}
