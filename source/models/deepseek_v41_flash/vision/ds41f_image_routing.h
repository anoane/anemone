/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Image-aware router and Engram prepares. Host C11.
 * Each prepare first validates the complete base description using its ordinary prepare, then validates the extra
 * range against the arena and every base range. It writes no output on failure. A step range must hold its block
 * and row flags. Vision bias is float32 [384], disjoint from all base arrays. The main router alone takes this
 * bias: draft routers are refused. Old prepares and records stay unchanged for text-only packs/programs.
 */
#ifndef DS41F_IMAGE_ROUTING_H
#define DS41F_IMAGE_ROUTING_H
#include <stdint.h>
#include "engine/backend/backend_seam.h"
#include "models/deepseek_v41_flash/engram/ds41f_engram_rows.h"
#include "ds41f_image_routing_records.h"
#include "models/deepseek_v41_flash/moe/ds41f_router.h"
#include "common/status.h"

struct ds41f_image_routing_kernels {
	uint32_t select, wide_select, engram_gate;
};
MUST_CHECK result_status ds41f_image_routing_kernels_prepare(
	const struct backend_operations *backend, struct device_card *card,
	struct ds41f_image_routing_kernels *kernels_out);
MUST_CHECK result_status ds41f_image_router_select_prepare(
	const struct ds41f_router_description *description,
	struct linear_range vision_bias, uint64_t arena_byte_count,
	const struct router_kernels_record *base_kernels,
	const struct ds41f_image_routing_kernels *kernels,
	struct kernel_launch *launch_out);
MUST_CHECK result_status ds41f_image_router_wide_prepare(
	const struct router_wide_description *description,
	struct linear_range vision_bias, uint64_t arena_byte_count,
	const struct router_kernels_record *base_kernels,
	const struct ds41f_image_routing_kernels *kernels,
	struct kernel_launch *launches_out);
MUST_CHECK result_status ds41f_image_engram_rows_prepare(
	const struct ds41f_engram_rows_description *description,
	struct linear_range step, uint64_t arena_byte_count,
	const struct ds41f_engram_rows_kernels *base_kernels,
	const struct ds41f_image_routing_kernels *kernels,
	struct kernel_launch *launches_out);
#endif
