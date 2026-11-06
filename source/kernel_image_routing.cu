/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* The reference VL routing bias follows each typed image row, including delimiters. */
#include "anemone/image_routing_records.h"
#include "kernel_router.cuh"

extern "C" __global__ void
image_router_select(struct image_router_arguments record, unsigned char *arena)
{
	const struct router_select_arguments arguments = record.base;
	__shared__ float affinities[LINEAR_MAXIMUM_ROWS * EXPERTS],
		pair_weights[MOST_PAIRS];
	__shared__ uint32_t pair_experts[MOST_PAIRS], pair_is_first[MOST_PAIRS];
	const struct step_block *const step =
		(const struct step_block *)(arena + arguments.step_offset);
	const uint32_t step_rows = step->row_count;
	const uint32_t live_rows = step_rows < arguments.row_count ?
					   step_rows :
					   arguments.row_count;
	const uint32_t warp = threadIdx.x / THREADS_PER_WARP,
		       pair = threadIdx.x, pair_count = live_rows * PICKS;
	LINEAR_DEVICE_CHECK(blockDim.x == ROUTER_THREADS &&
			    arguments.row_count <= LINEAR_MAXIMUM_ROWS &&
			    step_rows >= 1u &&
			    step_rows <= arguments.row_count &&
			    arguments.reference_order <= 1u);
	for (uint32_t index = threadIdx.x; index < live_rows * EXPERTS;
	     index += blockDim.x) {
		const float score = ((
			const float *)(arena + arguments.scores_offset))[index];
		affinities[index] = sqrtf(score > ROUTER_SOFTPLUS_THRESHOLD ?
						  score :
						  log1pf(expf(score)));
		((float *)(arena + arguments.affinities_offset))[index] =
			affinities[index];
	}
	__syncthreads();
	const struct step_row_record *const rows =
		(const struct step_row_record *)(step + 1);
	const uint64_t bias_offset =
		warp < live_rows && (rows[warp].flags & STEP_ROW_IMAGE) != 0 ?
			record.vision_bias_offset :
			arguments.bias_offset;
	ROUTER_SELECT_JOBS(arguments, (const float *)(arena + bias_offset),
			   EXPERTS, PICKS);
}

extern "C" __global__ void
image_router_wide_select(struct image_router_wide_arguments record,
			 unsigned char *arena)
{
	const struct router_wide_arguments arguments = record.base;
	__shared__ float affinities[LINEAR_MAXIMUM_ROWS * EXPERTS],
		weights[MOST_PAIRS];
	__shared__ uint32_t experts[MOST_PAIRS];
	const struct step_block *const step =
		(const struct step_block *)(arena + arguments.step_offset);
	const uint32_t step_rows = step->row_count;
	const uint32_t live_rows = step_rows < arguments.row_count ?
					   step_rows :
					   arguments.row_count;
	const uint32_t lane = threadIdx.x % THREADS_PER_WARP;
	const uint32_t first_row = blockIdx.y * LINEAR_MAXIMUM_ROWS,
		       warp = threadIdx.x / THREADS_PER_WARP;
	LINEAR_DEVICE_CHECK(blockDim.x == ROUTER_THREADS &&
			    step_rows <= arguments.row_count &&
			    arguments.row_count <= MODEL_WIDE_MOST_ROWS &&
			    arguments.reference_order <= 1u);
	if (first_row >= live_rows) {
		return;
	}
	const uint32_t row_count = live_rows - first_row < LINEAR_MAXIMUM_ROWS ?
					   live_rows - first_row :
					   LINEAR_MAXIMUM_ROWS;
	for (uint32_t index = threadIdx.x; index < row_count * EXPERTS;
	     index += blockDim.x) {
		const float score =
			((const float *)(arena + arguments.scores_offset))
				[first_row * EXPERTS + index];
		affinities[index] = sqrtf(score > ROUTER_SOFTPLUS_THRESHOLD ?
						  score :
						  log1pf(expf(score)));
		((float *)(arena +
			   arguments.affinities_offset))[first_row * EXPERTS +
							 index] =
			affinities[index];
	}
	__syncthreads();
	if (warp < row_count) {
		const struct step_row_record *const rows =
			(const struct step_row_record *)(step + 1);
		const uint64_t bias_offset =
			(rows[first_row + warp].flags & STEP_ROW_IMAGE) != 0 ?
				record.vision_bias_offset :
				arguments.bias_offset;
		router_row<EXPERTS, PICKS>(affinities + warp * EXPERTS,
					   (const float *)(arena + bias_offset),
					   lane, arguments.reference_order,
					   experts + warp * PICKS,
					   weights + warp * PICKS);
		if (lane < PICKS) {
			const struct router_pick pick = {
				experts[warp * PICKS + lane],
				weights[warp * PICKS + lane]
			};
			((struct router_pick *)(arena + arguments.picks_offset))
				[(first_row + warp) * PICKS + lane] = pick;
		}
	}
}
