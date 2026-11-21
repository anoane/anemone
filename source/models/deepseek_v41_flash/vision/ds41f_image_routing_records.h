/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Image-aware variants keep the base numerical records intact. C11 and CUDA record layouts only.
 * Router rows flagged IMAGE select with gate.bias_vl; every other row selects with gate.bias. Affinity and route
 * weights retain the base arithmetic. The image Engram gate skips flagged rows as a whole block, before reads.
 * Step ranges include the block and all row flags used by the launch; all are immutable during the launch.
 */
#ifndef DS41F_IMAGE_ROUTING_RECORDS_H
#define DS41F_IMAGE_ROUTING_RECORDS_H

#include <stdint.h>
#include "models/deepseek_v41_flash/moe/ds41f_router_records.h"
#include "models/deepseek_v41_flash/engram/ds41f_engram_rows_records.h"

#define DS41F_MODULE_IMAGE_ROUTING "ds41f_image_routing"
#define DS41F_KERNEL_IMAGE_ROUTER_SELECT "image_router_select"
#define DS41F_KERNEL_IMAGE_ROUTER_WIDE_SELECT "image_router_wide_select"
#define DS41F_MODULE_IMAGE_ENGRAM "ds41f_image_engram"
#define DS41F_KERNEL_IMAGE_ENGRAM_GATE "image_engram_gate"
#define DS41F_IMAGE_ROUTER_ARGUMENT_BYTES 88u
#define DS41F_IMAGE_ROUTER_WIDE_ARGUMENT_BYTES 128u
#define DS41F_IMAGE_ENGRAM_ARGUMENT_BYTES 40u

struct ds41f_image_router_arguments {
	struct ds41f_router_select_arguments base;
	uint64_t vision_bias_offset;
};
struct ds41f_image_router_wide_arguments {
	struct router_wide_arguments base;
	uint64_t vision_bias_offset;
};
struct ds41f_image_engram_arguments {
	struct ds41f_engram_gate_arguments base;
	uint64_t step_offset;
};
RECORD_HAS(ds41f_image_router_arguments, DS41F_IMAGE_ROUTER_ARGUMENT_BYTES);
RECORD_HAS(ds41f_image_router_wide_arguments,
	   DS41F_IMAGE_ROUTER_WIDE_ARGUMENT_BYTES);
RECORD_HAS(ds41f_image_engram_arguments, DS41F_IMAGE_ENGRAM_ARGUMENT_BYTES);
#endif
