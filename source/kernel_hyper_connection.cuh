/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * kernel_hyper_connection.cuh: what the hyper-connection sites (kernel_hyper_connection.cu) share with the fused site
 * front (kernel_site_front.cu) and the FFN site's joins (kernel_expert_join.cuh): eight bfloat16 values of a 16-byte
 * load as float32 and back, the post of a thread's eight values, the warp sum of the norms, and Sinkhorn of one row
 * on 16 lanes. Device code only.
 *
 * Contract: include/anemone/hyper_connection_records.h. Here:
 *   - Lane 4 j + k of a row's 16 lanes holds comb entry (j, k); the row's four values (x_0 + x_2) + (x_1 + x_3) are
 *     added and their largest taken by the two exchanges of the reference's kernel (distance 2 then 1 for a row, 8
 *     then 4 for a column). The exchanges stay inside 16 lanes, so each half of a warp may take a row of its own; a
 *     row's bits are the same in either half and with the other half doing anything.
 *   - Each function is pure arithmetic on what it is handed: its caller turned a checked record's offsets into the
 *     pointers.
 */
#ifndef KERNEL_HYPER_CONNECTION_CUH
#define KERNEL_HYPER_CONNECTION_CUH

#include "kernel_linear_common.cuh"

#include "anemone/hyper_connection_records.h"

#define HYPER_STREAMS MODEL_STREAMS
#define HYPER_VALUES HYPER_VALUES_PER_THREAD
#define HYPER_VALUES_PER_WORD 2u
#define HYPER_COMB_VALUES (HYPER_STREAMS * HYPER_STREAMS)
#define HYPER_COMB_FIRST_MIX (MODEL_SITE_VECTORS * HYPER_STREAMS)
#define HYPER_ROW_DISTANCE 2u                       /* lane 4 j + k: k ^ 2, then k ^ 1 */
#define HYPER_COLUMN_DISTANCE 8u                    /* j ^ 2, then j ^ 1 */
#define HYPER_POST_FACTOR 2.0f
#define HYPER_MEAN_FACTOR (1.0f / (float)MODEL_HIDDEN)
#define HYPER_LOW_HALF_MASK 0xFFFFu
#define HYPER_EVERY_LANE 0xFFFFFFFFu
enum hyper_scale { HYPER_SCALE_PRE, HYPER_SCALE_POST, HYPER_SCALE_COMB };
enum hyper_word { HYPER_FIRST_WORD, HYPER_SECOND_WORD, HYPER_THIRD_WORD, HYPER_FOURTH_WORD, HYPER_WORDS };

static_assert(HYPER_WORDS *HYPER_VALUES_PER_WORD == HYPER_VALUES &&
		      HYPER_COMB_FIRST_MIX + HYPER_COMB_VALUES == MODEL_MIXES,
	      "a load holds a thread's values; comb's mixes come last");

/* The eight bfloat16 values of a load as float32, and eight float32 values rounded to bfloat16 as one. */
__device__ static __forceinline__ void hyper_unpack(uint4 load, float values[HYPER_VALUES])
{
    const uint32_t words[HYPER_WORDS] = {load.x, load.y, load.z, load.w};
    for (uint32_t value = 0; value < HYPER_VALUES; value++) {
        const uint32_t word = words[value / HYPER_VALUES_PER_WORD];
        values[value] = linear_bfloat16_value(value % HYPER_VALUES_PER_WORD == 0 ? word & HYPER_LOW_HALF_MASK
                                                                                 : word >> BITS_PER_HALF);
    }
}
__device__ static __forceinline__ uint4 hyper_pack(const float values[HYPER_VALUES])
{
    uint32_t words[HYPER_WORDS];
    for (uint32_t word = 0; word < HYPER_WORDS; word++) {
        words[word] = linear_bfloat16_bits(values[HYPER_VALUES_PER_WORD * word]) |
                      (linear_bfloat16_bits(values[HYPER_VALUES_PER_WORD * word + 1u]) << BITS_PER_HALF);
    }
    return make_uint4(words[HYPER_FIRST_WORD], words[HYPER_SECOND_WORD], words[HYPER_THIRD_WORD],
                      words[HYPER_FOURTH_WORD]);
}

/* The post of a thread's eight values of a row (hyper_post_entry's): output y, the sublayer's eight values as the
 * post takes them (with the combine, the routed sums added and rounded); post and comb the row's; streams the
 * thread's load of stream 0, the streams HYPER_ROW_THREADS loads apart, read and written in place. Stream k
 * becomes bfloat16(post_k y + ((((0 + c_0k X_0) + c_1k X_1) + c_2k X_2) + c_3k X_3)) of the streams X as they were
 * before any is written, each product and sum rounded on its own (the modules are built with --fmad=false). */
__device__ static __forceinline__ void hyper_post_values(const float output[HYPER_VALUES], const float *post,
                                                         const float *comb, uint4 *streams)
{
    float values[HYPER_STREAMS][HYPER_VALUES];
    for (uint32_t stream = 0; stream < HYPER_STREAMS; stream++) {
	    hyper_unpack(streams[stream * HYPER_ROW_THREADS], values[stream]);
    }
    for (uint32_t target = 0; target < HYPER_STREAMS; target++) {
        float mixed[HYPER_VALUES];
        for (uint32_t value = 0; value < HYPER_VALUES; value++) {
            float sum = 0.0f;
            for (uint32_t source = 0; source < HYPER_STREAMS; source++) {
                sum = sum + comb[source * HYPER_STREAMS + target] * values[source][value];
            }
            mixed[value] = post[target] * output[value] + sum;
        }
	streams[target * HYPER_ROW_THREADS] = hyper_pack(mixed);
    }
}

/* The sum of a value of every lane of a warp, by exchanges at distance 16, 8, 4, 2 and 1: the same on every lane. */
__device__ static __forceinline__ float hyper_warp_sum(float value)
{
    for (uint32_t distance = THREADS_PER_WARP >> 1u; distance != 0; distance >>= 1u) {
        value = value + __shfl_xor_sync(HYPER_EVERY_LANE, value, (int)distance);
    }
    return value;
}

/* Of the four values of a lane's row (distance 2) or column (distance 8): the sum (x_0 + x_2) + (x_1 + x_3), or the
 * largest, by the two exchanges of the reference's kernel. */
__device__ static __forceinline__ float hyper_exchange(float value, uint32_t distance, bool is_largest)
{
    const float first = __shfl_xor_sync(HYPER_EVERY_LANE, value, (int)distance);
    value = is_largest ? fmaxf(value, first) : value + first;
    const float second = __shfl_xor_sync(HYPER_EVERY_LANE, value, (int)(distance >> 1u));
    return is_largest ? fmaxf(value, second) : value + second;
}

__device__ static __forceinline__ float hyper_sigmoid(float value)
{
    return 1.0f / (1.0f + expf(-value));
}

/* Comb entry sub_lane (lane 4 j + k of the row's 16) of one row's mixes, scale and base, after its 20 iterations. */
__device__ static __forceinline__ float hyper_sinkhorn_comb(const float *mixes, const float *scale, const float *base,
                                                            uint32_t sub_lane)
{
    const uint32_t mix = HYPER_COMB_FIRST_MIX + sub_lane;
    float comb = fmaf(mixes[mix], scale[HYPER_SCALE_COMB], base[mix]);
    comb = expf(comb - hyper_exchange(comb, HYPER_ROW_DISTANCE, true));
    comb = comb / hyper_exchange(comb, HYPER_ROW_DISTANCE, false) +
	   MODEL_HYPER_EPSILON;
    comb = comb / (hyper_exchange(comb, HYPER_COLUMN_DISTANCE, false) +
		   MODEL_HYPER_EPSILON);
    for (uint32_t iteration = 1; iteration < MODEL_SINKHORN_ITERATIONS;
	 iteration++) {
	    comb = comb / (hyper_exchange(comb, HYPER_ROW_DISTANCE, false) +
			   MODEL_HYPER_EPSILON);
	    comb = comb / (hyper_exchange(comb, HYPER_COLUMN_DISTANCE, false) +
			   MODEL_HYPER_EPSILON);
    }
    return comb;
}

/* Post and pre of stream `stream` of one row's mixes, scale and base. */
__device__ static __forceinline__ float hyper_post_of(const float *mixes, const float *scale, const float *base,
                                                      uint32_t stream)
{
    return HYPER_POST_FACTOR *
           hyper_sigmoid(fmaf(mixes[HYPER_STREAMS + stream], scale[HYPER_SCALE_POST], base[HYPER_STREAMS + stream]));
}
__device__ static __forceinline__ float hyper_pre_of(const float *mixes, const float *scale, const float *base,
                                                     uint32_t stream)
{
	return hyper_sigmoid(fmaf(mixes[stream], scale[HYPER_SCALE_PRE],
				  base[stream])) +
	       MODEL_HYPER_EPSILON;
}

#endif
