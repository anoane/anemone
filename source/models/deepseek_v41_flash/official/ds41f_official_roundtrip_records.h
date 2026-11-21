/* SPDX-License-Identifier: AGPL-3.0-or-later */
#ifndef DS41F_OFFICIAL_ROUNDTRIP_RECORDS_H
#define DS41F_OFFICIAL_ROUNDTRIP_RECORDS_H

#include <stdint.h>
#include "models/deepseek_v41_flash/ds41f_geometry.h"

#define DS41F_OFFICIAL_ROUNDTRIP_THREADS 256u
#define DS41F_OFFICIAL_ROUNDTRIP_BLOCK_VALUES 32u
#define DS41F_OFFICIAL_ROUNDTRIP_MOST_WIDTH \
	(DS41F_HEADS_VALUE * DS41F_HEAD_DIMS)
#define DS41F_OFFICIAL_ROUNDTRIP_ARGUMENT_BYTES 24u
#define DS41F_MODULE_OFFICIAL_ROUNDTRIP "ds41f_official_roundtrip"
#define DS41F_KERNEL_OFFICIAL_ROUNDTRIP "official_roundtrip_entry"

struct ds41f_official_roundtrip_arguments {
	uint64_t input_offset;
	uint64_t output_offset;
	uint32_t width;
	uint32_t row_count;
};

RECORD_HOLDS(sizeof(struct ds41f_official_roundtrip_arguments) ==
	     DS41F_OFFICIAL_ROUNDTRIP_ARGUMENT_BYTES);
#endif
