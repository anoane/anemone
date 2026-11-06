/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * prepare_common.h: the checks every prepare function makes, and the one way it fills a launch. Private to source/:
 * linear.c and every family of kernels after it (linear.h's contract for prepare functions holds for all of them).
 *
 * Contract:
 *   - A check is true, or false behind the one error line of the refusal, which begins with what (the launches the
 *     description is for); it writes nothing else.
 *   - A range is checked one way: it begins on a multiple of LINEAR_OFFSET_ALIGNMENT_BYTES, lies inside the
 *     arena (in 64 bits, with no wrap), holds what the launches touch there (a count of UINT64_MAX, beyond 64 bits,
 *     fits no range) and, for a matrix, exactly its stored bytes; no two ranges of one description share a byte.
 *   - prepare_launch_fill writes every byte of a launch: grid_columns by grid_rows blocks of block_threads
 *     threads in one dimension, the record, zero behind it. Its caller has checked that the record fits.
 */
#ifndef PREPARE_COMMON_H
#define PREPARE_COMMON_H

#include <stdbool.h>
#include <stdint.h>

#include "anemone/backend_seam.h"
#include "anemone/linear.h"

/* One range of a description as it is checked. */
struct checked_range {
    const char *name;                           /* what an error line calls it */
    const struct linear_range *range;
    uint64_t touched_byte_count;                /* what the launches read or write in it, from its first byte on */
    bool is_matrix;                             /* a matrix occupies exactly its bytes */
};

bool prepare_arguments_are_given(const char *what, const void *description,
				 const void *kernels, const void *launches_out);
bool prepare_kernel_is_prepared(const char *what, uint32_t kernel,
				const char *function_name);
bool prepare_reserved_is_zero(const char *what, uint32_t reserved_zero);
bool prepare_ranges_are_valid(const char *what,
			      const struct checked_range *ranges,
			      uint32_t range_count, uint64_t arena_byte_count);
void prepare_launch_fill(struct kernel_launch *launch, uint32_t kernel,
			 uint32_t grid_columns, uint32_t grid_rows,
			 uint32_t block_threads,
			 uint32_t dynamic_shared_byte_count, const void *record,
			 uint32_t record_byte_count);

#endif
