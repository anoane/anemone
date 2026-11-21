/* SPDX-License-Identifier: AGPL-3.0-or-later */
#ifndef DS41F_OFFICIAL_ROUTER_PROJECTION_RECORDS_H
#define DS41F_OFFICIAL_ROUTER_PROJECTION_RECORDS_H

#include <stdint.h>
#include "models/deepseek_v41_flash/ds41f_geometry.h"

#define DS41F_MODULE_OFFICIAL_ROUTER_PROJECTION \
	"ds41f_official_router_projection"
#define DS41F_KERNEL_OFFICIAL_ROUTER_PROJECTION "official_router_projection"
#define DS41F_OFFICIAL_ROUTER_PROJECTION_ARGUMENT_BYTES 32u
#define DS41F_OFFICIAL_ROUTER_PROJECTION_OWNERS 32u
#define DS41F_OFFICIAL_ROUTER_PROJECTION_OUTPUTS 4u
#define DS41F_OFFICIAL_ROUTER_PROJECTION_THREADS   \
	(DS41F_OFFICIAL_ROUTER_PROJECTION_OWNERS * \
	 DS41F_OFFICIAL_ROUTER_PROJECTION_OUTPUTS)
#define DS41F_OFFICIAL_ROUTER_PROJECTION_ODD_ROWS 3u

struct ds41f_official_router_projection_arguments {
	uint64_t weights_offset;
	uint64_t rows_offset;
	uint64_t output_offset;
	uint32_t row_count;
	uint32_t reserved_zero;
};

RECORD_HOLDS(sizeof(struct ds41f_official_router_projection_arguments) ==
	     DS41F_OFFICIAL_ROUTER_PROJECTION_ARGUMENT_BYTES);
RECORD_HOLDS(DS41F_EXPERTS % DS41F_OFFICIAL_ROUTER_PROJECTION_OUTPUTS == 0);
#endif
