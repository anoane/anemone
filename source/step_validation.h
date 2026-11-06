/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Private immutable admission checks shared by program creation and host-expert attachment. */
#ifndef STEP_VALIDATION_H
#define STEP_VALIDATION_H

#include <stdbool.h>
#include <stdint.h>

#include "anemone/engram.h"
#include "anemone/model_seam.h"

/* offset and count units of unit_byte_count bytes lie inside byte_count bytes. */
static inline bool step_range_fits(uint64_t offset, uint64_t count,
				   uint64_t unit_byte_count,
				   uint64_t byte_count)
{
	return offset <= byte_count &&
	       (unit_byte_count == 0 ||
		count <= (byte_count - offset) / unit_byte_count);
}

/* Creation only: preserve the first diagnostic and the original unsupported classification.
 * The caller has checked arrays/output pointers and admits programs in order against programs[0]. */
const char *step_program_problem(const struct execution_program *program,
				 const struct execution_program *first,
				 struct device_card *const *cards,
				 const struct engram_record *engram,
				 bool *is_unsupported_out);

#endif
