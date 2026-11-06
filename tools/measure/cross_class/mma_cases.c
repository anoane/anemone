/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * mma_cases.c: runs a file of cases through one m16n8k16 instruction on two cards and prints each card's result:
 * a line per case "form c a0 b0 a1 b1 .." (floats; up to 16 pairs, the rest 0), output "c80 c120 exact".
 *   mma_cases <sm_80 cubin> <sm_120a cubin> <class 80 card> <class 120 card> < cases
 */
#include <cuda.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MOST_CASES 4096u
#define TERMS 16u

static uint16_t half_of(float value, uint32_t form)
{
    uint32_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    if (form == 0u) {
        return (uint16_t)(bits >> 16);          /* bf16: the host gives exact bf16 values */
    }
    const uint32_t sign = (bits >> 16) & 0x8000u;  /* f16 of an exact f16 value, normal or subnormal */
    const int32_t exponent = (int32_t)((bits >> 23) & 0xFFu) - 127;
    uint32_t mantissa = bits & 0x7FFFFFu;
    if (value == 0.0f) {
        return (uint16_t)sign;
    }
    if (exponent >= -14) {
        return (uint16_t)(sign | (uint32_t)(exponent + 15) << 10 | mantissa >> 13);
    }
    mantissa |= 0x800000u;
    return (uint16_t)(sign | mantissa >> (13 + (-14 - exponent)));
}

int main(int argument_count, char **arguments)
{
    static uint16_t a_values[2][MOST_CASES * TERMS], b_values[2][MOST_CASES * TERMS];
    static float c_values[2][MOST_CASES], results[2][2][MOST_CASES];
    static long double exact[2][MOST_CASES];
    static uint32_t indices[2][MOST_CASES];
    uint32_t counts[2] = {0u, 0u}, total = 0;
    char line[4096];
    if (argument_count != 5 || cuInit(0) != CUDA_SUCCESS) {
        return 1;
    }
    while (fgets(line, sizeof(line), stdin) != NULL && total < MOST_CASES) {
        char *cursor = line;
        const uint32_t form = (uint32_t)strtoul(cursor, &cursor, 10);
        const float c = strtof(cursor, &cursor);
        const uint32_t slot = counts[form & 1u]++;
        long double sum = c;
        c_values[form][slot] = c;
        for (uint32_t term = 0; term < TERMS; term++) {
            char *before = cursor;
            float a = strtof(cursor, &cursor), b = 0.0f;
            if (cursor == before) {
                a = 0.0f;
            } else {
                b = strtof(cursor, &cursor);
            }
            a_values[form][slot * TERMS + term] = half_of(a, form);
            b_values[form][slot * TERMS + term] = half_of(b, form);
            sum += (long double)a * (long double)b;
        }
        exact[form][slot] = sum;
        indices[form][slot] = total++;
    }
    for (uint32_t card = 0; card < 2u; card++) {
        CUdevice device = 0;
        CUcontext context = NULL;
        CUmodule module = NULL;
        CUfunction function = NULL;
        if (cuDeviceGet(&device, atoi(arguments[3 + card])) != CUDA_SUCCESS ||
            cuDevicePrimaryCtxRetain(&context, device) != CUDA_SUCCESS || cuCtxSetCurrent(context) != CUDA_SUCCESS ||
            cuModuleLoad(&module, arguments[1 + card]) != CUDA_SUCCESS ||
            cuModuleGetFunction(&function, module, "mma_cases") != CUDA_SUCCESS) {
            return 1;
        }
        for (uint32_t form = 0; form < 2u; form++) {
            CUdeviceptr a_buffer = 0, b_buffer = 0, c_buffer = 0, result_buffer = 0;
            uint32_t form_value = form;
            void *launch_arguments[] = {&form_value, &a_buffer, &b_buffer, &c_buffer, &result_buffer};
            if (counts[form] == 0) {
                continue;
            }
            if (cuMemAlloc(&a_buffer, sizeof(a_values[0])) != CUDA_SUCCESS ||
                cuMemAlloc(&b_buffer, sizeof(b_values[0])) != CUDA_SUCCESS ||
                cuMemAlloc(&c_buffer, sizeof(c_values[0])) != CUDA_SUCCESS ||
                cuMemAlloc(&result_buffer, sizeof(results[0][0])) != CUDA_SUCCESS ||
                cuMemcpyHtoD(a_buffer, a_values[form], sizeof(a_values[0])) != CUDA_SUCCESS ||
                cuMemcpyHtoD(b_buffer, b_values[form], sizeof(b_values[0])) != CUDA_SUCCESS ||
                cuMemcpyHtoD(c_buffer, c_values[form], sizeof(c_values[0])) != CUDA_SUCCESS ||
                cuLaunchKernel(function, counts[form], 1, 1, 32, 1, 1, 0, NULL, launch_arguments, NULL) != CUDA_SUCCESS ||
                cuMemcpyDtoH(results[card][form], result_buffer, sizeof(results[0][0])) != CUDA_SUCCESS) {
                return 1;
            }
            (void)cuMemFree(a_buffer);
            (void)cuMemFree(b_buffer);
            (void)cuMemFree(c_buffer);
            (void)cuMemFree(result_buffer);
        }
    }
    for (uint32_t index = 0; index < total; index++) {
        for (uint32_t form = 0; form < 2u; form++) {
            for (uint32_t slot = 0; slot < counts[form]; slot++) {
                if (indices[form][slot] == index) {
                    (void)printf("%u: %a %a exact %.20La%s\n", index, (double)results[0][form][slot],
                                 (double)results[1][form][slot], exact[form][slot],
                                 results[0][form][slot] != results[1][form][slot] ? "  DIFFER" : "");
                }
            }
        }
    }
    return 0;
}
