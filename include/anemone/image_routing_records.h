/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Image-aware variants keep the base numerical records intact. C11 and CUDA record layouts only.
 * Router rows flagged IMAGE select with gate.bias_vl; every other row selects with gate.bias. Affinity and route
 * weights retain the base arithmetic. The image Engram gate skips flagged rows as a whole block, before reads.
 * Step ranges include the block and all row flags used by the launch; all are immutable during the launch.
 */
#ifndef IMAGE_ROUTING_RECORDS_H
#define IMAGE_ROUTING_RECORDS_H

#include <stdint.h>
#include "anemone/router_records.h"
#include "anemone/engram_rows_records.h"

#define KERNEL_FILE_IMAGE_ROUTING "kernel_image_routing"
#define KERNEL_IMAGE_ROUTER_SELECT "image_router_select"
#define KERNEL_IMAGE_ROUTER_WIDE_SELECT "image_router_wide_select"
#define KERNEL_FILE_IMAGE_ENGRAM "kernel_image_engram"
#define KERNEL_IMAGE_ENGRAM_GATE "image_engram_gate"
#define IMAGE_ROUTER_ARGUMENT_BYTES 88u
#define IMAGE_ROUTER_WIDE_ARGUMENT_BYTES 128u
#define IMAGE_ENGRAM_ARGUMENT_BYTES 40u

struct image_router_arguments {
	struct router_select_arguments base;
	uint64_t vision_bias_offset;
};
struct image_router_wide_arguments {
	struct router_wide_arguments base;
	uint64_t vision_bias_offset;
};
struct image_engram_arguments {
	struct engram_gate_arguments base;
	uint64_t step_offset;
};
LINEAR_RECORD_HAS(image_router_arguments, IMAGE_ROUTER_ARGUMENT_BYTES);
LINEAR_RECORD_HAS(image_router_wide_arguments,
		  IMAGE_ROUTER_WIDE_ARGUMENT_BYTES);
LINEAR_RECORD_HAS(image_engram_arguments, IMAGE_ENGRAM_ARGUMENT_BYTES);
#endif
