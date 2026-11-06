/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Cold route validation shared with TP's explicitly larger kernel. This accepts the full element count,
 * bounded by the caller's kernel limit (positive and no greater than UINT32_MAX minus the thread count).
 * All card, operation and directed-mask checks and the route policy are unchanged. No output is published
 * on refusal; output may alias the input masks. This pure metadata operation does not qualify a transfer,
 * allocation or kernel. The public planner retains ORDERED_COLLECTIVE_MAXIMUM_ELEMENTS.
 */
#ifndef ORDERED_COLLECTIVE_INTERNAL_H
#define ORDERED_COLLECTIVE_INTERNAL_H

#include "anemone/ordered_collective.h"

MUST_CHECK result_status ordered_collective_routes_prepare_capacity(
	const struct ordered_collective_route_description *description,
	uint32_t maximum_elements, uint32_t *host_reader_masks_out);

#endif
