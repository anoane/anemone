/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * mma_probe.c: feeds the same bf16 operands to one m16n8k16 instruction on two cards and prints, for output (0, 0),
 * each card's result beside the exact sum of its 16 products (long double), for several operand patterns.
 *   mma_probe <sm_80 cubin> <sm_120a cubin> <class 80 card> <class 120 card>
 */
#include <cuda.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LANES 32u
#define WORDS_PER_LANE 6u
#define CASES 6u

static uint16_t bfloat16_of(float value)
{
    uint32_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    return (uint16_t)(bits >> 16);
}

static float float_of_bfloat16(uint16_t half)
{
    const uint32_t bits = (uint32_t)half << 16;
    float value = 0.0f;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

int main(int argument_count, char **arguments)
{
    if (argument_count != 5 || cuInit(0) != CUDA_SUCCESS) {
        return 1;
    }
    /* m16n8k16 row.col: lane l holds A row l/4 (and +8), k pairs (l%4)*2 + {0,1} (and +8); B col l/4, k the same.
     * Output (0,0) is lane 0's d[0]: row 0 of A times column 0 of B over k = 0..15, from lanes 0..3. */
    static const float magnitudes[CASES][2] = {{1.0f, 1.0f / 1024}, {1.0f, 1.0f / 65536}, {1.0f, 1.0f / 16777216},
                                                {1.0f, -1.0f / 3}, {3.0f, 1.0f / 7}, {1.0f, 1.0f / 300000}};
    for (uint32_t pattern = 0; pattern < CASES; pattern++) {
        uint32_t operands[LANES * WORDS_PER_LANE];
        float a_values[16], b_values[16];
        memset(operands, 0, sizeof(operands));
        for (uint32_t k = 0; k < 16u; k++) {   /* A row 0: alternating big and small; B column 0: ones and thirds */
            a_values[k] = float_of_bfloat16(bfloat16_of(k % 2 == 0 ? magnitudes[pattern][0] * (1.0f + (float)k / 64)
                                                                   : magnitudes[pattern][1] * (1.0f + (float)k / 32)));
            b_values[k] = float_of_bfloat16(bfloat16_of(k < 8 ? 1.0f : 1.0f + 1.0f / 128));
        }
        for (uint32_t lane = 0; lane < 4u; lane++) {
            for (uint32_t half = 0; half < 2u; half++) {     /* k (lane*2 + j) in a[0] (rows 0), k + 8 in a[2] */
                const uint32_t k_low = lane * 2u + half, k_high = k_low + 8u;
                operands[lane * WORDS_PER_LANE + 0u] |= (uint32_t)bfloat16_of(a_values[k_low]) << (16u * half);
                operands[lane * WORDS_PER_LANE + 2u] |= (uint32_t)bfloat16_of(a_values[k_high]) << (16u * half);
                operands[lane * WORDS_PER_LANE + 4u] |= (uint32_t)bfloat16_of(b_values[k_low]) << (16u * half);
                operands[lane * WORDS_PER_LANE + 5u] |= (uint32_t)bfloat16_of(b_values[k_high]) << (16u * half);
            }
        }
        long double exact = 0.0L;
        for (uint32_t k = 0; k < 16u; k++) {
            exact += (long double)a_values[k] * (long double)b_values[k];
        }
        float results[2] = {0.0f, 0.0f};
        for (uint32_t card = 0; card < 2u; card++) {
            CUdevice device = 0;
            CUcontext context = NULL;
            CUmodule module = NULL;
            CUfunction function = NULL;
            CUdeviceptr operand_buffer = 0, result_buffer = 0;
            float lane_results[LANES * 4u];
            void *launch_arguments[] = {&operand_buffer, &result_buffer};
            if (cuDeviceGet(&device, atoi(arguments[3 + card])) != CUDA_SUCCESS ||
                cuDevicePrimaryCtxRetain(&context, device) != CUDA_SUCCESS || cuCtxSetCurrent(context) != CUDA_SUCCESS ||
                cuModuleLoad(&module, arguments[1 + card]) != CUDA_SUCCESS ||
                cuModuleGetFunction(&function, module, "mma_probe") != CUDA_SUCCESS ||
                cuMemAlloc(&operand_buffer, sizeof(operands)) != CUDA_SUCCESS ||
                cuMemAlloc(&result_buffer, sizeof(lane_results)) != CUDA_SUCCESS ||
                cuMemcpyHtoD(operand_buffer, operands, sizeof(operands)) != CUDA_SUCCESS ||
                cuLaunchKernel(function, 1, 1, 1, LANES, 1, 1, 0, NULL, launch_arguments, NULL) != CUDA_SUCCESS ||
                cuMemcpyDtoH(lane_results, result_buffer, sizeof(lane_results)) != CUDA_SUCCESS) {
                (void)fprintf(stderr, "card %u failed\n", card);
                return 1;
            }
            results[card] = lane_results[0];
            (void)cuMemFree(operand_buffer);
            (void)cuMemFree(result_buffer);
            (void)cuModuleUnload(module);
        }
        const float nearest = (float)exact;
        (void)printf("pattern %u: exact %.12Lg  nearest-f32 %a  class80 %a  class120 %a\n", pattern, exact,
                     (double)nearest, (double)results[0], (double)results[1]);
    }
    return 0;
}
