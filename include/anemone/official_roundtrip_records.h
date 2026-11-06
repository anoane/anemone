/* SPDX-License-Identifier: AGPL-3.0-or-later */
#ifndef OFFICIAL_ROUNDTRIP_RECORDS_H
#define OFFICIAL_ROUNDTRIP_RECORDS_H

#include <stdint.h>
#include "anemone/model_layout.h"

#define OFFICIAL_ROUNDTRIP_THREADS 256u
#define OFFICIAL_ROUNDTRIP_BLOCK_VALUES 32u
#define OFFICIAL_ROUNDTRIP_MOST_WIDTH (MODEL_HEADS_VALUE * MODEL_HEAD_DIMS)
#define OFFICIAL_ROUNDTRIP_ARGUMENT_BYTES 24u
#define KERNEL_FILE_OFFICIAL_ROUNDTRIP "kernel_official_roundtrip"
#define KERNEL_OFFICIAL_ROUNDTRIP "official_roundtrip_entry"

struct official_roundtrip_arguments {
	uint64_t input_offset;
	uint64_t output_offset;
	uint32_t width;
	uint32_t row_count;
};

MODEL_LAYOUT_HOLDS(sizeof(struct official_roundtrip_arguments) ==
		   OFFICIAL_ROUNDTRIP_ARGUMENT_BYTES);
#endif
