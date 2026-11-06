/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* mma_cases.cu: for each case, one m16n8k16 instruction (form 0: bf16, form 1: f16) whose output (0, 0) is
 * C + sum over k of A[0][k] B[k][0]; the 16 A and B values and C come from the host, every other operand is 0. */
#include <stdint.h>
extern "C" __global__ void mma_cases(uint32_t form, const uint16_t *a_values, const uint16_t *b_values,
                                     const float *c_values, float *results)
{
    const uint32_t lane = threadIdx.x, group = lane >> 2, place = lane & 3u, case_index = blockIdx.x;
    const uint16_t *a_row = a_values + case_index * 16u, *b_column = b_values + case_index * 16u;
    uint32_t a[4] = {0u, 0u, 0u, 0u}, b[2] = {0u, 0u};
    if (group == 0) {                           /* row 0 of A: k = 2 place + {0, 1} in a[0], + 8 in a[2] */
        a[0] = (uint32_t)a_row[2u * place] | ((uint32_t)a_row[2u * place + 1u] << 16);
        a[2] = (uint32_t)a_row[2u * place + 8u] | ((uint32_t)a_row[2u * place + 9u] << 16);
        b[0] = (uint32_t)b_column[2u * place] | ((uint32_t)b_column[2u * place + 1u] << 16);   /* column 0 */
        b[1] = (uint32_t)b_column[2u * place + 8u] | ((uint32_t)b_column[2u * place + 9u] << 16);
    }
    float d[4] = {lane == 0 ? c_values[case_index] : 0.0f, 0.0f, 0.0f, 0.0f};
    if (form == 0u) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
                     "{%8, %9}, {%0, %1, %2, %3};"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    } else {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
                     "{%8, %9}, {%0, %1, %2, %3};"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    }
    if (lane == 0) {
        results[case_index] = d[0];
    }
}
