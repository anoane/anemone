/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * kernel_expert_join.cuh: the join of the FFN site (expert_fused_records.h) for eight consecutive values of one row,
 * which the join (kernel_hyper_connection.cu) and the shared join (kernel_trunk_linear.cu) share. Device code only.
 *
 * Contract: the arithmetic of expert_row_sum_entry and hyper_post_entry with the combine, value by value, in
 * their order (the kernels are built with --fmad=false: a product and a sum are rounded each):
 *   - s = float32 sum of the row's pair outputs, from +0 in ascending slot order, stored where the row sum stores it
 *     when the join is given its array;
 *   - y = bfloat16(s + y) of the shared expert's output y;
 *   - the post of y (hyper_post_values of kernel_hyper_connection.cuh, the post's own code).
 * A bfloat16 is rounded to nearest with a tie to even, a not-a-number to 0x7FC0 (linear_bfloat16_bits).
 */
#ifndef KERNEL_EXPERT_JOIN_CUH
#define KERNEL_EXPERT_JOIN_CUH

#include "kernel_hyper_connection.cuh"

#include "anemone/expert_fused_records.h"

#define JOIN_STREAMS HYPER_STREAMS
#define JOIN_VALUES EXPERT_JOIN_VALUES
#define JOIN_SUM_STORES 2u                       /* 16-byte stores of a thread's eight float32 sums */

static_assert(JOIN_VALUES == HYPER_VALUES && JOIN_SUM_STORES * HYPER_WORDS == JOIN_VALUES,
              "a thread joins the post's eight values; two stores of four float32 words hold their sums");

/* The join of eight values of a row: pairs is their load of the row's first pair output, a slot's
 * HYPER_ROW_THREADS loads apart (the loads of a row of values); shared_output the shared expert's output there
 * (bfloat16 values as float32); post and comb the row's; streams their load of stream 0, read and written
 * (hyper_post_values); row_sums where their sums go (float32, the row sum's array), or NULL where the join is given
 * none. */
__device__ static __forceinline__ void expert_join_values(const uint4 *pairs, uint32_t pairs_per_row,
                                                          const float shared_output[JOIN_VALUES], const float *post,
                                                          const float *comb, uint4 *streams, float4 *row_sums)
{
    float routed[JOIN_VALUES], output[JOIN_VALUES];
    for (uint32_t value = 0; value < JOIN_VALUES; value++) {
        routed[value] = 0.0f;
    }
    for (uint32_t slot = 0; slot < pairs_per_row; slot++) {
        float pair[JOIN_VALUES];
	hyper_unpack(pairs[(size_t)slot * HYPER_ROW_THREADS], pair);
	for (uint32_t value = 0; value < JOIN_VALUES; value++) {
            routed[value] += pair[value];
        }
    }
    for (uint32_t store = 0; row_sums != NULL && store < JOIN_SUM_STORES; store++) {
        const float *const words = routed + store * HYPER_WORDS;   /* the four float32 words of a 16-byte store */
        row_sums[store] = make_float4(words[HYPER_FIRST_WORD], words[HYPER_SECOND_WORD], words[HYPER_THIRD_WORD],
                                      words[HYPER_FOURTH_WORD]);
    }
    for (uint32_t value = 0; value < JOIN_VALUES; value++) {
        output[value] = linear_bfloat16_value(linear_bfloat16_bits(routed[value] + shared_output[value]));
    }
    hyper_post_values(output, post, comb, streams);
}

#endif
