/* SPDX-License-Identifier: AGPL-3.0-or-later */
#include "anemone/ep_filter_records.h"

extern "C" __global__ void ep_filter(struct ep_filter_arguments arguments,
				     unsigned char *arena)
{
	const uint32_t thread = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t stride = gridDim.x * blockDim.x;
	struct expert_job_record *const jobs =
		(struct expert_job_record *)(arena + arguments.jobs_offset);
	for (uint32_t job = thread; job < arguments.job_count; job += stride)
		if (jobs[job].gate_offset == arguments.unowned_gate_offset)
			jobs[job].row_count = 0;
	uint16_t *const hidden = (uint16_t *)(arena + arguments.hidden_offset);
	uint16_t *const output = (uint16_t *)(arena + arguments.output_offset);
	const uint32_t hidden_count =
		arguments.pair_count * arguments.hidden_width;
	const uint32_t output_count =
		arguments.pair_count * arguments.output_width;
	for (uint32_t element = thread; element < hidden_count;
	     element += stride)
		hidden[element] = 0;
	for (uint32_t element = thread; element < output_count;
	     element += stride)
		output[element] = 0;
}
