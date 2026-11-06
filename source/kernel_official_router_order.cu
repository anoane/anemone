/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Canonical M1 finite-score tie order and normalization. The existing select
 * builds six jobs in ascending expert order. This independent pass preserves
 * that layout and every job field except its live route weight. */
#include "anemone/official_router_order_records.h"
#include <cuda_runtime.h>

#define ROUTER_ORDER_INFINITY_BITS 0x7f800000u
#define ROUTER_ORDER_PARTITIONS 2u
#define ROUTER_ORDER_LEFT_TAIL 4u
#define ROUTER_ORDER_LEFT_END 2u
#define ROUTER_ORDER_RIGHT_TAIL 5u
#define ROUTER_ORDER_RIGHT_END 3u
#define ROUTER_ORDER_NEGATIVE_INFINITY_BITS 0xff800000u

extern "C" __global__ void
official_router_order(struct router_select_arguments source,
		      unsigned char *arena)
{
	if (threadIdx.x != 0 || blockIdx.x != 0)
		return;
	const struct step_block *step =
		(const struct step_block *)(arena + source.step_offset);
	struct expert_job_record *jobs =
		(struct expert_job_record *)(arena + source.jobs_offset);
	const float *affinities =
		(const float *)(arena + source.affinities_offset);
	const float *bias = (const float *)(arena + source.bias_offset);
	if (source.row_count != 1 || step->row_count != 1 || step->flags != 0 ||
	    source.reference_order != 1)
		return;
	float cut = __uint_as_float(ROUTER_ORDER_INFINITY_BITS);
	for (uint32_t slot = 0; slot < MODEL_EXPERTS_PER_ROW; slot++) {
		if (jobs[slot].row_count != 1 ||
		    jobs[slot].expert >= MODEL_EXPERTS ||
		    jobs[slot].pair_index[0] != slot ||
		    (slot != 0 && jobs[slot - 1u].expert >= jobs[slot].expert))
			return;
		const float value =
			affinities[jobs[slot].expert] + bias[jobs[slot].expert];
		if (!isfinite(value) ||
		    !isfinite(affinities[jobs[slot].expert]))
			return;
		cut = value < cut ? value : cut;
	}
	__shared__ uint32_t ids[OFFICIAL_ROUTER_ORDER_SORT_WIDTH];
	uint32_t count = 0;
	__shared__ float keys[OFFICIAL_ROUTER_ORDER_SORT_WIDTH];
	for (uint32_t equal = 0; equal < ROUTER_ORDER_PARTITIONS; equal++)
		for (uint32_t slot = 0; slot < MODEL_EXPERTS_PER_ROW; slot++) {
			const uint32_t expert = jobs[slot].expert;
			const float value = affinities[expert] + bias[expert];
			if ((equal == 0 && value > cut) ||
			    (equal != 0 && value == cut)) {
				ids[count] = expert;
				keys[count++] = value;
			}
		}
	if (count != MODEL_EXPERTS_PER_ROW)
		return;
	for (uint32_t slot = MODEL_EXPERTS_PER_ROW;
	     slot < OFFICIAL_ROUTER_ORDER_SORT_WIDTH; slot++) {
		ids[slot] = MODEL_EXPERTS;
		keys[slot] =
			__uint_as_float(ROUTER_ORDER_NEGATIVE_INFINITY_BITS);
	}
	for (uint32_t size = ROUTER_ORDER_PARTITIONS;
	     size <= OFFICIAL_ROUTER_ORDER_SORT_WIDTH;
	     size *= ROUTER_ORDER_PARTITIONS)
		for (uint32_t distance = size / ROUTER_ORDER_PARTITIONS;
		     distance != 0; distance /= ROUTER_ORDER_PARTITIONS)
			for (uint32_t slot = 0;
			     slot < OFFICIAL_ROUTER_ORDER_SORT_WIDTH; slot++) {
				const uint32_t other = slot ^ distance;
				if (other <= slot)
					continue;
				const bool exchange =
					(slot & size) == 0 ?
						keys[slot] <= keys[other] :
						keys[slot] > keys[other];
				if (exchange) {
					const float key = keys[slot];
					const uint32_t id = ids[slot];
					keys[slot] = keys[other];
					ids[slot] = ids[other];
					keys[other] = key;
					ids[other] = id;
				}
			}
	const float sum = ((affinities[ids[0]] +
			    affinities[ids[ROUTER_ORDER_LEFT_TAIL]]) +
			   affinities[ids[ROUTER_ORDER_LEFT_END]]) +
			  ((affinities[ids[1]] +
			    affinities[ids[ROUTER_ORDER_RIGHT_TAIL]]) +
			   affinities[ids[ROUTER_ORDER_RIGHT_END]]);
	for (uint32_t slot = 0; slot < MODEL_EXPERTS_PER_ROW; slot++)
		jobs[slot].route_weight[0] = affinities[jobs[slot].expert] /
					     (sum + ROUTER_WEIGHT_EPSILON) *
					     MODEL_ROUTE_SCALE;
}
