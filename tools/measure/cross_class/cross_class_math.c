/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * cross_class_math.c: runs cross_class_math.cu on two cards (a class 80 and a class 120 cubin) over every float32
 * input of each function and reports the inputs whose results differ. C11 on the CUDA driver API.
 *   cross_class_math <sm_80 cubin> <sm_120a cubin> <card of class 80> <card of class 120>
 */
#include <cuda.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FUNCTION_COUNT 8u
#define CHUNK_INPUTS (1u << 24)
#define CHUNK_COUNT 256u                      /* 2^32 inputs */
#define INPUTS_PER_THREAD 64u
#define THREADS 256u
#define BLOCKS (CHUNK_INPUTS / THREADS / INPUTS_PER_THREAD)
#define MOST_REPORTED 6u
#define CARDS 2u

static const char *const function_names[FUNCTION_COUNT] = {"expf", "logf", "log1pf", "sinf", "cosf", "rsqrtf",
                                                           "sqrtf", "1/x"};

struct card {
    CUcontext context;
    CUmodule module;
    CUfunction hash, values;
    CUdeviceptr hashes, value_buffer;
};

static int checked(CUresult result, const char *what)
{
    if (result != CUDA_SUCCESS) {
        const char *text = NULL;
        (void)cuGetErrorString(result, &text);
        (void)fprintf(stderr, "%s failed: %s\n", what, text != NULL ? text : "unknown");
        return 0;
    }
    return 1;
}

static int card_open(struct card *card, int ordinal, const char *cubin)
{
    CUdevice device = 0;
    return checked(cuDeviceGet(&device, ordinal), "cuDeviceGet") &&
           checked(cuDevicePrimaryCtxRetain(&card->context, device), "cuDevicePrimaryCtxRetain") &&
           checked(cuCtxSetCurrent(card->context), "cuCtxSetCurrent") &&
           checked(cuModuleLoad(&card->module, cubin), "cuModuleLoad") &&
           (getenv("CROSS_CLASS_MMA_ONLY") != NULL ||
            (checked(cuModuleGetFunction(&card->hash, card->module, "cross_class_hash"), "hash function") &&
             checked(cuModuleGetFunction(&card->values, card->module, "cross_class_values"), "values function"))) &&
           checked(cuMemAlloc(&card->hashes, (size_t)BLOCKS * sizeof(uint64_t)), "cuMemAlloc hashes") &&
           checked(cuMemAlloc(&card->value_buffer, (size_t)CHUNK_INPUTS * sizeof(uint32_t)), "cuMemAlloc values");
}

/* The block hashes of one chunk of one function, into hashes_out. */
static int chunk_hashed(struct card *card, uint32_t function, uint32_t first, uint64_t *hashes_out)
{
    uint32_t inputs_per_thread = INPUTS_PER_THREAD;
    void *arguments[] = {&function, &first, &inputs_per_thread, &card->hashes};
    return checked(cuCtxSetCurrent(card->context), "cuCtxSetCurrent") &&
           checked(cuLaunchKernel(card->hash, BLOCKS, 1, 1, THREADS, 1, 1, 0, NULL, arguments, NULL), "hash launch") &&
           checked(cuMemcpyDtoH(hashes_out, card->hashes, (size_t)BLOCKS * sizeof(uint64_t)), "hash copy");
}

static int chunk_values(struct card *card, uint32_t function, uint32_t first, uint32_t *values_out)
{
    void *arguments[] = {&function, &first, &card->value_buffer};
    return checked(cuCtxSetCurrent(card->context), "cuCtxSetCurrent") &&
           checked(cuLaunchKernel(card->values, CHUNK_INPUTS / THREADS, 1, 1, THREADS, 1, 1, 0, NULL, arguments, NULL),
                   "values launch") &&
           checked(cuMemcpyDtoH(values_out, card->value_buffer, (size_t)CHUNK_INPUTS * sizeof(uint32_t)), "values copy");
}

#define MMA_WARPS 65536u
#define MMA_BLOCK_THREADS 128u

/* The hashes of every warp of the tensor-core test on both cards, compared: the warps that differ. */
static int mma_compared(struct card *cards)
{
    static uint64_t hashes[CARDS][MMA_WARPS];
    CUfunction functions[CARDS];
    CUdeviceptr buffers[CARDS];
    for (uint32_t form = 0; form < 2u; form++) {
        unsigned long long seed = 12345ull + form;
        for (uint32_t card = 0; card < CARDS; card++) {
            uint32_t form_value = form;
            void *mma_arguments[] = {&form_value, &seed, &buffers[card]};
            if (!checked(cuCtxSetCurrent(cards[card].context), "cuCtxSetCurrent") ||
                !checked(cuModuleGetFunction(&functions[card], cards[card].module, getenv("CROSS_CLASS_KERNEL") != NULL ? getenv("CROSS_CLASS_KERNEL") : "cross_class_mma"), "mma function") ||
                !checked(cuMemAlloc(&buffers[card], (size_t)MMA_WARPS * sizeof(uint64_t)), "cuMemAlloc mma") ||
                !checked(cuLaunchKernel(functions[card], MMA_WARPS * 32u / MMA_BLOCK_THREADS, 1, 1, MMA_BLOCK_THREADS, 1,
                                        1, 0, NULL, mma_arguments, NULL), "mma launch") ||
                !checked(cuMemcpyDtoH(hashes[card], buffers[card], sizeof(hashes[card])), "mma copy") ||
                !checked(cuMemFree(buffers[card]), "cuMemFree mma")) {
                return 0;
            }
        }
        uint32_t differing = 0;
        for (uint32_t warp = 0; warp < MMA_WARPS; warp++) {
            differing += hashes[0][warp] != hashes[1][warp] ? 1u : 0u;
        }
        (void)printf("mma m16n8k16 %s: %" PRIu32 " of %u warps (64 chained instructions each) differ\n",
                     form == 0 ? "bf16" : "f16", differing, MMA_WARPS);
    }
    return 1;
}

int main(int argument_count, char **arguments)
{
    struct card cards[CARDS] = {{0}};
    static uint64_t hashes[CARDS][BLOCKS];
    uint32_t *values[CARDS] = {malloc((size_t)CHUNK_INPUTS * sizeof(uint32_t)),
                               malloc((size_t)CHUNK_INPUTS * sizeof(uint32_t))};
    if (argument_count != 5 || values[0] == NULL || values[1] == NULL || !checked(cuInit(0), "cuInit") ||
        !card_open(&cards[0], atoi(arguments[3]), arguments[1]) ||
        !card_open(&cards[1], atoi(arguments[4]), arguments[2])) {
        (void)fprintf(stderr, "usage: cross_class_math <sm_80 cubin> <sm_120a cubin> <class 80 card> <class 120 card>\n");
        return 1;
    }
    if (getenv("CROSS_CLASS_MMA_ONLY") != NULL) {
        return mma_compared(cards) ? 0 : 1;
    }
    for (uint32_t function = 0; function < FUNCTION_COUNT; function++) {
        uint64_t differing = 0;
        uint32_t reported = 0;
        for (uint32_t chunk = 0; chunk < CHUNK_COUNT; chunk++) {
            const uint32_t first = chunk * CHUNK_INPUTS;
            if (!chunk_hashed(&cards[0], function, first, hashes[0]) ||
                !chunk_hashed(&cards[1], function, first, hashes[1])) {
                return 1;
            }
            if (memcmp(hashes[0], hashes[1], sizeof(hashes[0])) == 0) {
                continue;
            }
            if (!chunk_values(&cards[0], function, first, values[0]) ||
                !chunk_values(&cards[1], function, first, values[1])) {
                return 1;
            }
            for (uint32_t index = 0; index < CHUNK_INPUTS; index++) {
                if (values[0][index] != values[1][index]) {
                    differing++;
                    if (reported++ < MOST_REPORTED) {
                        uint32_t input = first + index;
                        float input_value = 0.0f, value_80 = 0.0f, value_120 = 0.0f;
                        memcpy(&input_value, &input, sizeof(input_value));
                        memcpy(&value_80, &values[0][index], sizeof(value_80));
                        memcpy(&value_120, &values[1][index], sizeof(value_120));
                        (void)printf("  %s(%a = %.9g): class 80 %08" PRIx32 " (%.9g), class 120 %08" PRIx32 " (%.9g)\n",
                                     function_names[function], (double)input_value, (double)input_value,
                                     values[0][index], (double)value_80, values[1][index], (double)value_120);
                    }
                }
            }
        }
        (void)printf("%s: %" PRIu64 " of 2^32 inputs differ\n", function_names[function], differing);
        (void)fflush(stdout);
    }
    return 0;
}
