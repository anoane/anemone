/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Prepared handles for the bounded, serial original-weight reference recipe. No token-path dispatch or allocation. */
#ifndef OFFICIAL_REFERENCE_H
#define OFFICIAL_REFERENCE_H
#include "anemone/official_attention.h"
#include "anemone/official_head.h"
#include "anemone/official_index.h"
#include "anemone/official_linear.h"
#include "anemone/official_norm.h"
#include "anemone/official_rope.h"
#include "anemone/official_roundtrip.h"
#include "anemone/official_router_order.h"
#include "anemone/official_router_projection.h"
#include "anemone/official_state_records.h"

#define OFFICIAL_REFERENCE_CONTEXT_LIMIT 512u
struct official_reference_kernels {
	struct official_linear_kernels linear;
	uint32_t norm, rope, roundtrip, head, router_projection, router_order,
		attention, index;
	uint32_t window, compressor, key, index_queries, site, engram,
		projection_float32, projection_bfloat16;
	uint32_t expert_gate_up, expert_down, final_kept;
};
MUST_CHECK result_status official_reference_kernels_prepare(
	const struct backend_operations *backend, struct device_card *card,
	uint32_t arithmetic_class,
	struct official_reference_kernels *kernels_out);
#endif
