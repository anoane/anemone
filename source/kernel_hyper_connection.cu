/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * kernel_hyper_connection.cu: the hyper-connection sites of a block (model.py:948-966), the embedding into the four
 * streams (:1253-1260), the final collapse (:1268-1269) and the draft's inputs (:1133, 1269). Device code only, built
 * once per architecture into the module "kernel_hyper_connection".
 *
 * Contract: include/anemone/hyper_connection_records.h. Here:
 *   - Thread t of a row takes its values 8 t .. 8 t + 7: one 16-byte load of each stream, which the post writes back
 *     in place. No other thread touches them.
 *   - The mix finish is one block for a row. Its first HYPER_ROW_THREADS threads collapse and norm while its
 *     last warp runs Sinkhorn on lanes 4 j + k (lanes 16 to 31 repeat 0 to 15), with the layout and the exchanges of
 *     the reference's kernel. That warp writes the site's pre over the carried one after the block's one barrier,
 *     which every read of the carried pre precedes.
 *   - The draft's inputs take the final's collapse (the mean: pre 1/4 each) and the final's norm, operation for
 *     operation in the same order: a tap is bit for bit the final's kept collapse of streams whose carried pre is 1/4
 *     each, and a row's main_norm the final's norm of that row (hyper_connection_card_test holds both). They are
 *     written apart: shared helpers changed the mix finish's generated code. The final that keeps its collapse is a
 *     stamp of its own (is_kept), so the mix finish and the backbone's final keep their generated code (sm_80 and
 *     sm_120a, instruction for instruction).
 *   - No atomic operation; nothing depends on timing.
 *   - The join (expert_fused_records.h) takes a row's values as the post does, each thread its eight
 *     (expert_join_values, kernel_expert_join.cuh): the row sum's float32 sums (stored only where it is given their
 *     array), then the post's own code (hyper_post_values).
 */
#include "kernel_expert_join.cuh"
#include "kernel_hyper_connection.cuh"

#include "anemone/step_records.h"

#define STREAMS HYPER_STREAMS
#define ROW_THREADS HYPER_ROW_THREADS
#define ROW_WARPS (ROW_THREADS / THREADS_PER_WARP)
#define VALUES HYPER_VALUES
#define COMB_VALUES HYPER_COMB_VALUES
#define MEAN_FACTOR HYPER_MEAN_FACTOR
#define SUM_LOADS 2u                                /* 16-byte loads of a thread's eight float32 sums */

static_assert(ROW_WARPS <= THREADS_PER_WARP && ROW_THREADS % THREADS_PER_WARP == 0,
              "one warp adds the sums of a row's warps");

/* Sinkhorn for one row on the lanes of one warp: writes post and comb and returns the site's pre (lanes 0 to 3). */
__device__ static __forceinline__ float
hyper_sinkhorn(const struct hyper_finish_arguments *arguments,
	       unsigned char *arena, size_t row, uint32_t lane)
{
	const float *const mixes =
		(const float *)(arena + arguments->mixes_offset) +
		row * MODEL_MIXES;
	const float *const scale =
		(const float *)(arena + arguments->scale_offset);
	const float *const base =
		(const float *)(arena + arguments->base_offset);
	const uint32_t sub_lane = lane % COMB_VALUES, stream = lane % STREAMS;
	const float comb = hyper_sinkhorn_comb(mixes, scale, base, sub_lane);
	if (lane < COMB_VALUES) {
		((float *)(arena +
			   arguments->comb_offset))[row * COMB_VALUES + lane] =
			comb;
	}
    if (lane < STREAMS) {
        ((float *)(arena + arguments->post_offset))[row * STREAMS + lane] = hyper_post_of(mixes, scale, base, lane);
    }
    return hyper_pre_of(mixes, scale, base, stream);
}

/* A row's collapse with the carried pre and its norm: the final (is_kept: which also keeps the collapse at
 * comb_offset), and the mix finish with Sinkhorn in its last warp. */
template <bool is_mix_finish, bool is_kept = false>
__device__ __forceinline__ void
hyper_collapse(struct hyper_finish_arguments arguments, unsigned char *arena)
{
    __shared__ float warp_sums[ROW_WARPS];
    const size_t row = blockIdx.y;
    const uint32_t thread = threadIdx.x, lane = thread % THREADS_PER_WARP;
    const bool is_sinkhorn = is_mix_finish && thread >= ROW_THREADS;
    float *const carried = (float *)(arena + arguments.pre_offset) + row * STREAMS;
    float hidden[VALUES] = {}, site_pre = 0.0f, square_sum = 0.0f;
    LINEAR_DEVICE_CHECK(blockDim.x ==
			(is_mix_finish ? HYPER_FINISH_THREADS : ROW_THREADS));
    if (is_sinkhorn) {
        site_pre = hyper_sinkhorn(&arguments, arena, row, lane);
    } else {
        const uint4 *const streams =
            (const uint4 *)(arena + arguments.streams_offset) + row * STREAMS * ROW_THREADS + thread;
        for (uint32_t stream = 0; stream < STREAMS; stream++) {
            float values[VALUES];
            const float weight = carried[stream];
            hyper_unpack(streams[stream * ROW_THREADS], values);
            for (uint32_t value = 0; value < VALUES; value++) {
                hidden[value] = hidden[value] + weight * values[value];
            }
        }
        for (uint32_t value = 0; value < VALUES; value++) {
            hidden[value] = linear_bfloat16_value(linear_bfloat16_bits(hidden[value]));
            square_sum = square_sum + hidden[value] * hidden[value];
        }
        if (is_kept) {
            ((uint4 *)(arena + arguments.comb_offset))[row * ROW_THREADS + thread] = hyper_pack(hidden);
        }
        square_sum = hyper_warp_sum(square_sum);
        if (lane == 0) {
            warp_sums[thread / THREADS_PER_WARP] = square_sum;
        }
    }
    __syncthreads();
    if (is_sinkhorn) {
        if (lane < STREAMS) {
            carried[lane] = site_pre;
        }
        return;
    }
    const float factor =
	    rsqrtf(hyper_warp_sum(lane < ROW_WARPS ? warp_sums[lane] : 0.0f) *
			   MEAN_FACTOR +
		   MODEL_NORM_EPSILON);
    float weights[VALUES];
    hyper_unpack(((const uint4 *)(arena + arguments.weight_offset))[thread], weights);
    for (uint32_t value = 0; value < VALUES; value++) {
        hidden[value] = weights[value] * (hidden[value] * factor);
    }
    ((uint4 *)(arena + arguments.hidden_offset))[row * ROW_THREADS + thread] = hyper_pack(hidden);
}

/* Each kernel's bound (its block, one block) holds a thread to the registers such a block leaves it: a build that
 * would not fit spills and fails (no spill passes the build), where without it every launch would. */
extern "C" __global__ void __launch_bounds__(HYPER_FINISH_THREADS, 1)
	hyper_mix_finish(struct hyper_finish_arguments arguments,
			 unsigned char *arena)
{
    hyper_collapse<true>(arguments, arena);
}

extern "C" __global__ void __launch_bounds__(HYPER_ROW_THREADS, 1)
	hyper_final(struct hyper_finish_arguments arguments,
		    unsigned char *arena)
{
    hyper_collapse<false>(arguments, arena);
}

/* The draft's final: the backbone's, keeping its collapse for the draft's confidence. */
extern "C" __global__ void __launch_bounds__(HYPER_ROW_THREADS, 1)
	hyper_final_kept(struct hyper_finish_arguments arguments,
			 unsigned char *arena)
{
    hyper_collapse<false, true>(arguments, arena);
}

/* A tap (the mean of the row's streams, rounded to bfloat16 once) into its columns of the taps, or main_norm of a
 * bfloat16 row: the final's operations above. The mode is the launch's: every thread of a block reaches the norm's
 * barrier or none does. */
extern "C" __global__ void __launch_bounds__(HYPER_ROW_THREADS, 1)
	hyper_draft_inputs(struct hyper_draft_inputs_arguments arguments,
			   unsigned char *arena)
{
    __shared__ float warp_sums[ROW_WARPS];
    const size_t row = blockIdx.y;
    const uint32_t thread = threadIdx.x, lane = thread % THREADS_PER_WARP;
    uint4 *const output = (uint4 *)(arena + arguments.output_offset) +
                          (row * arguments.output_row_values + arguments.column_offset) / VALUES + thread;
    float hidden[VALUES] = {}, square_sum = 0.0f;
    LINEAR_DEVICE_CHECK(blockDim.x == ROW_THREADS &&
			arguments.mode <= HYPER_ROW_NORM &&
			arguments.column_offset % VALUES == 0 &&
			arguments.output_row_values % VALUES == 0 &&
			arguments.column_offset + MODEL_HIDDEN <=
				arguments.output_row_values);
    if (arguments.mode == HYPER_STREAM_MEAN) {
	    const uint4 *const streams =
		    (const uint4 *)(arena + arguments.input_offset) +
		    row * STREAMS * ROW_THREADS + thread;
	    for (uint32_t stream = 0; stream < STREAMS; stream++) {
		    float values[VALUES];
		    hyper_unpack(streams[stream * ROW_THREADS], values);
		    for (uint32_t value = 0; value < VALUES; value++) {
			    hidden[value] =
				    hidden[value] +
				    HYPER_STREAM_MEAN_PRE * values[value];
		    }
	    }
	    *output = hyper_pack(hidden);
	    return;
    }
    hyper_unpack(((const uint4 *)(arena + arguments.input_offset))[row * ROW_THREADS + thread], hidden);
    for (uint32_t value = 0; value < VALUES; value++) {
        square_sum = square_sum + hidden[value] * hidden[value];
    }
    square_sum = hyper_warp_sum(square_sum);
    if (lane == 0) {
        warp_sums[thread / THREADS_PER_WARP] = square_sum;
    }
    __syncthreads();
    const float factor =
	    rsqrtf(hyper_warp_sum(lane < ROW_WARPS ? warp_sums[lane] : 0.0f) *
			   MEAN_FACTOR +
		   MODEL_NORM_EPSILON);
    float weights[VALUES];
    hyper_unpack(((const uint4 *)(arena + arguments.weight_offset))[thread], weights);
    for (uint32_t value = 0; value < VALUES; value++) {
        hidden[value] = weights[value] * (hidden[value] * factor);
    }
    *output = hyper_pack(hidden);
}

/* A row's streams after its sublayer, in place; with the combine, the routed sums added to y first. */
extern "C" __global__ void __launch_bounds__(HYPER_PIECE_THREADS, 1)
	hyper_post_entry(struct hyper_post_arguments arguments,
			 unsigned char *arena)
{
    const size_t row = blockIdx.y, thread = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const float *const post = (const float *)(arena + arguments.post_offset) + row * STREAMS;
    const float *const comb = (const float *)(arena + arguments.comb_offset) + row * COMB_VALUES;
    uint4 *const streams = (uint4 *)(arena + arguments.streams_offset) + row * STREAMS * ROW_THREADS + thread;
    float output[VALUES];
    LINEAR_DEVICE_CHECK(blockDim.x * gridDim.x == ROW_THREADS);
    hyper_unpack(((const uint4 *)(arena + arguments.output_offset))[row * ROW_THREADS + thread], output);
    if (arguments.is_combined != 0) {
        const float4 *const sums =
            (const float4 *)(arena + arguments.sums_offset) + (row * ROW_THREADS + thread) * SUM_LOADS;
        const float4 low = sums[0], high = sums[1];
        const float routed[VALUES] = {low.x, low.y, low.z, low.w, high.x, high.y, high.z, high.w};
        for (uint32_t value = 0; value < VALUES; value++) {
            output[value] = linear_bfloat16_value(linear_bfloat16_bits(routed[value] + output[value]));
        }
    }
    hyper_post_values(output, post, comb, streams);
}

extern "C" __global__ void __launch_bounds__(HYPER_PIECE_THREADS, 1)
	hyper_embed(struct hyper_embed_arguments arguments,
		    unsigned char *arena)
{
    const size_t row = blockIdx.y, thread = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t token_id =
	    ((const struct step_row_record *)(arena + arguments.step_offset +
					      STEP_BLOCK_BYTES_VALUE))[row]
		    .token_id;
    LINEAR_DEVICE_CHECK(blockDim.x * gridDim.x == ROW_THREADS && token_id >= 0 &&
                        (uint32_t)token_id < arguments.vocabulary_size);
    const uint4 value = ((const uint4 *)(arena + arguments.embedding_offset))[(size_t)token_id * ROW_THREADS + thread];
    uint4 *const streams = (uint4 *)(arena + arguments.streams_offset) + row * STREAMS * ROW_THREADS + thread;
    for (uint32_t stream = 0; stream < STREAMS; stream++) {
        streams[stream * ROW_THREADS] = value;
    }
    if (thread < STREAMS) {
        ((float *)(arena + arguments.pre_offset))[row * STREAMS + thread] = thread == 0 ? 1.0f : 0.0f;
    }
}

/* The FFN site's join: a row's routed sums (expert_row_sum_entry), the shared expert's output and the post with the
 * combine (hyper_post_entry), in one launch: thread t of a row joins its values 8 t .. 8 t + 7. */
extern "C" __global__ void __launch_bounds__(HYPER_PIECE_THREADS, 1)
	expert_join(struct expert_join_arguments arguments,
		    unsigned char *arena)
{
    const size_t row = blockIdx.y, thread = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    float shared_output[VALUES];
    LINEAR_DEVICE_CHECK(blockDim.x * gridDim.x == ROW_THREADS);
    hyper_unpack(((const uint4 *)(arena + arguments.shared_output_offset))[row * ROW_THREADS + thread], shared_output);
    expert_join_values(
	    (const uint4 *)(arena + arguments.pair_output_offset) +
		    row * arguments.pairs_per_row * ROW_THREADS + thread,
	    arguments.pairs_per_row, shared_output,
	    (const float *)(arena + arguments.post_offset) + row * STREAMS,
	    (const float *)(arena + arguments.comb_offset) + row * COMB_VALUES,
	    (uint4 *)(arena + arguments.streams_offset) +
		    row * STREAMS * ROW_THREADS + thread,
	    arguments.row_sums_offset == EXPERT_JOIN_NO_ROW_SUMS ?
		    NULL :
		    (float4 *)(arena + arguments.row_sums_offset) +
			    (row * ROW_THREADS + thread) * SUM_LOADS);
}
