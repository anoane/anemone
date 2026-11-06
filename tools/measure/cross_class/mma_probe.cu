/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* mma_probe.cu: one bf16 m16n8k16 instruction on a zero accumulator, operands made so that the 16 products of lane
 * 0's first output are known: the result word of each card, for the host to compare with the exact sum. */
#include <stdint.h>
extern "C" __global__ void mma_probe(const uint32_t *operands, float *results)
{
    const uint32_t lane = threadIdx.x;
    uint32_t a[4], b[2];
    for (uint32_t word = 0; word < 4u; word++) {
        a[word] = operands[lane * 6u + word];
    }
    b[0] = operands[lane * 6u + 4u];
    b[1] = operands[lane * 6u + 5u];
    float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
                 "{%8, %9}, {%0, %1, %2, %3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    for (uint32_t element = 0; element < 4u; element++) {
        results[lane * 4u + element] = d[element];
    }
}
