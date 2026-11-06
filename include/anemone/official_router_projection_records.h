/* SPDX-License-Identifier: AGPL-3.0-or-later */
#ifndef OFFICIAL_ROUTER_PROJECTION_RECORDS_H
#define OFFICIAL_ROUTER_PROJECTION_RECORDS_H

#include <stdint.h>
#include "anemone/model_layout.h"

#define KERNEL_FILE_OFFICIAL_ROUTER_PROJECTION \
	"kernel_official_router_projection"
#define KERNEL_OFFICIAL_ROUTER_PROJECTION "official_router_projection"
#define OFFICIAL_ROUTER_PROJECTION_ARGUMENT_BYTES 32u
#define OFFICIAL_ROUTER_PROJECTION_OWNERS 32u
#define OFFICIAL_ROUTER_PROJECTION_OUTPUTS 4u
#define OFFICIAL_ROUTER_PROJECTION_THREADS \
	(OFFICIAL_ROUTER_PROJECTION_OWNERS * OFFICIAL_ROUTER_PROJECTION_OUTPUTS)
#define OFFICIAL_ROUTER_PROJECTION_ODD_ROWS 3u

struct official_router_projection_arguments {
	uint64_t weights_offset;
	uint64_t rows_offset;
	uint64_t output_offset;
	uint32_t row_count;
	uint32_t reserved_zero;
};

MODEL_LAYOUT_HOLDS(sizeof(struct official_router_projection_arguments) ==
		   OFFICIAL_ROUTER_PROJECTION_ARGUMENT_BYTES);
MODEL_LAYOUT_HOLDS(MODEL_EXPERTS % OFFICIAL_ROUTER_PROJECTION_OUTPUTS == 0);
#endif
