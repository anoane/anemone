/* SPDX-License-Identifier: AGPL-3.0-or-later */
#ifndef DS41F_OFFICIAL_HC_PARTITIONED_RECORDS_H
#define DS41F_OFFICIAL_HC_PARTITIONED_RECORDS_H

#include "ds41f_official_hc_records.h"

#define DS41F_MODULE_OFFICIAL_HC_PARTITIONED "ds41f_official_hc_partitioned"
#define DS41F_KERNEL_OFFICIAL_HC_PARTIALS "official_hc_partials"
#define DS41F_KERNEL_OFFICIAL_HC_REDUCE "official_hc_reduce"
#define DS41F_OFFICIAL_HC_PARTITIONS 47u
#define DS41F_OFFICIAL_HC_PARTIAL_THREADS 128u
#define DS41F_OFFICIAL_HC_PARTITIONED_ARGUMENT_BYTES 48u
#define DS41F_OFFICIAL_HC_PARTITIONED_LAUNCHES 2u
#define DS41F_OFFICIAL_HC_ROW_GRID_DIMENSION 2u

/* Scratch F32 [actual_rows,24,47]. Pass one writes each partial exactly once;
 * pass two reads all partials after an ordered stream/graph dependency. Its
 * row RMS is always the F1 reference tree. No cross-CTA polling or atomics.
 */
struct ds41f_official_hc_partitioned_arguments {
	uint64_t weights_offset;
	uint64_t rows_offset;
	uint64_t partials_offset;
	uint64_t output_offset;
	uint32_t actual_rows;
	uint32_t reserved_zero;
	float epsilon;
	uint32_t reserved_tail;
};

RECORD_HOLDS(sizeof(struct ds41f_official_hc_partitioned_arguments) ==
	     DS41F_OFFICIAL_HC_PARTITIONED_ARGUMENT_BYTES);
#endif
