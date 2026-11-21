/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Prepared handles for the bounded, serial original-weight reference recipe. No token-path dispatch or allocation. */
#ifndef DS41F_OFFICIAL_REFERENCE_H
#define DS41F_OFFICIAL_REFERENCE_H
#include "ds41f_official_attention.h"
#include "ds41f_official_head.h"
#include "ds41f_official_index.h"
#include "ds41f_official_linear.h"
#include "ds41f_official_norm.h"
#include "ds41f_official_rope.h"
#include "ds41f_official_roundtrip.h"
#include "ds41f_official_router_order.h"
#include "ds41f_official_router_projection.h"
#include "ds41f_official_state_records.h"

#define DS41F_OFFICIAL_REFERENCE_CONTEXT_LIMIT 512u
struct ds41f_official_reference_kernels {
	struct ds41f_official_linear_kernels linear;
	uint32_t norm, rope, roundtrip, head, router_projection, router_order,
		attention, index;
	uint32_t window, compressor, key, index_queries, site, engram,
		projection_float32, projection_bfloat16;
	uint32_t expert_gate_up, expert_down, final_kept;
};
MUST_CHECK result_status ds41f_official_reference_kernels_prepare(
	const struct backend_operations *backend, struct device_card *card,
	uint32_t arithmetic_class,
	struct ds41f_official_reference_kernels *kernels_out);
#endif
