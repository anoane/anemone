/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * probe_harness_cuda.h: the card side of probe_harness.h (the CUDA runtime: the card, its arrays with guard bytes,
 * the timed region), moved out of it unchanged on 2026-10-17 to keep each file under STYLE.md's 1,000 lines. It is
 * included by probe_harness.h at the place it held there, and only there.
 */
#ifndef PROBE_HARNESS_CUDA_H
#define PROBE_HARNESS_CUDA_H
/* ---- with the CUDA runtime: the card, its arrays with guard bytes, the timed region ---- */
#if defined(__CUDACC__)
#include <cuda_runtime.h>
static_assert(PROBE_SM == PROBE_SM_80 || PROBE_SM == PROBE_SM_120A, "-DPROBE_SM=80 (sm_80) or 120 (sm_120a)");
#define PROBE_ARCHITECTURE (PROBE_SM == PROBE_SM_80 ? "sm_80" : "sm_120a")
#define PROBE_CHECK_CUDA(call) probe_cuda_check((call), #call, __FILE__, __LINE__)
PROBE_FUNCTION void probe_cuda_check(cudaError_t status, const char *call, const char *file, int line) {
    if (status == cudaSuccess) { return; }
    fprintf(stderr, "%s:%d: %s failed: %s\n", file, line, call, cudaGetErrorString(status));
    exit(EXIT_FAILURE);
}
static struct {
    cudaDeviceProp properties;
    cudaEvent_t started, stopped;
    uint8_t *arrays[PROBE_MOST_CARD_ARRAYS];     /* every card array of the probe, and its bytes before the guard */
    uint64_t array_bytes[PROBE_MOST_CARD_ARRAYS];
    uint32_t array_count;
} probe_cuda;
/* The one card this process sees, ordinal 0. 0: more or fewer than one card, or not this build's architecture. */
PROBE_FUNCTION int probe_cuda_open(struct probe_card_facts *facts) {
    const cudaDeviceProp *properties = &probe_cuda.properties;
    int card_count = 0;
    PROBE_CHECK_CUDA(cudaGetDeviceCount(&card_count));
    PROBE_CHECK_CUDA(cudaSetDevice(0));
    PROBE_CHECK_CUDA(cudaGetDeviceProperties(&probe_cuda.properties, 0));
    if (card_count != 1 || properties->major * PROBE_SM_PER_MAJOR + properties->minor != PROBE_SM) {
        fprintf(stderr, "this process sees %d cards, the first of capability %d.%d: a probe is given exactly one "
                        "(CUDA_VISIBLE_DEVICES=<index>), of its build's architecture, %s: nothing is timed through "
                        "another architecture's code\n", card_count, properties->major, properties->minor,
                PROBE_ARCHITECTURE);
        return 0;
    }
    PROBE_CHECK_CUDA(cudaEventCreate(&probe_cuda.started));
    PROBE_CHECK_CUDA(cudaEventCreate(&probe_cuda.stopped));
    facts->name = properties->name;
    facts->architecture = PROBE_ARCHITECTURE;
    facts->sm_count = properties->multiProcessorCount;
    facts->l2_bytes = (uint64_t)properties->l2CacheSize;
    facts->block_threads_most = (uint32_t)properties->maxThreadsPerBlock;
    facts->warps_per_sm = (uint32_t)(properties->maxThreadsPerMultiProcessor / properties->warpSize);
    facts->shared_bytes_most = properties->sharedMemPerBlock;
    for (int axis = 0; axis < PROBE_GRID_AXES; axis++) {
        facts->grid_most[axis] = (uint32_t)properties->maxGridSize[axis];
    }
    return 1;
}
/* A card array of `bytes` with PROBE_GUARD_BYTES of PROBE_GUARD_VALUE behind it, known by its pointer from now on. */
PROBE_FUNCTION void *probe_cuda_allocate(uint64_t bytes) {
    uint8_t *device = NULL;
    PROBE_REQUIRE(probe_cuda.array_count < PROBE_MOST_CARD_ARRAYS && bytes != 0 && bytes < SIZE_MAX - PROBE_GUARD_BYTES,
                  "a card array too many, or one of no size");
    PROBE_CHECK_CUDA(cudaMalloc((void **)&device, (size_t)bytes + PROBE_GUARD_BYTES));
    PROBE_CHECK_CUDA(cudaMemset(device + bytes, (int)PROBE_GUARD_VALUE, PROBE_GUARD_BYTES));
    probe_cuda.arrays[probe_cuda.array_count] = device;
    probe_cuda.array_bytes[probe_cuda.array_count++] = bytes;
    return device;
}
PROBE_FUNCTION uint64_t probe_cuda_array_bytes(const void *device) {
    for (uint32_t index = 0; index < probe_cuda.array_count; index++) {
        if (probe_cuda.arrays[index] == (const uint8_t *)device) { return probe_cuda.array_bytes[index]; }
    }
    return 0;
}
PROBE_FUNCTION uint64_t probe_cuda_guards_differing(void) {
    uint8_t guard[PROBE_GUARD_BYTES];
    uint64_t differing = 0;
    for (uint32_t index = 0; index < probe_cuda.array_count; index++) {
        PROBE_CHECK_CUDA(cudaMemcpy(guard, probe_cuda.arrays[index] + probe_cuda.array_bytes[index], PROBE_GUARD_BYTES,
                                    cudaMemcpyDeviceToHost));
        for (uint32_t byte = 0; byte < PROBE_GUARD_BYTES; byte++) {
            differing += guard[byte] != PROBE_GUARD_VALUE ? 1u : 0u;
        }
    }
    return differing;
}
PROBE_FUNCTION void probe_cuda_check_guards(const char *case_name) {     /* after the launches of the case */
    int broken = probe_flag_given(PROBE_FLAG_BREAK_GUARD) && probe_cuda.array_count > 0;
    if (broken) {                                /* the byte behind the first array, as a store one too far leaves it */
        PROBE_CHECK_CUDA(cudaMemset(probe_cuda.arrays[0] + probe_cuda.array_bytes[0], (int)PROBE_GUARD_VALUE + 1, 1));
    }
    probe_check_count(case_name, "all", "guard_bytes", probe_cuda_guards_differing(),
                      (uint64_t)probe_cuda.array_count * PROBE_GUARD_BYTES, broken);
}
PROBE_FUNCTION void probe_cuda_mark_start(void *) { PROBE_CHECK_CUDA(cudaEventRecord(probe_cuda.started)); }
PROBE_FUNCTION double probe_cuda_elapsed_microseconds(void *) {
    float milliseconds = 0.0f;
    PROBE_CHECK_CUDA(cudaEventRecord(probe_cuda.stopped));
    PROBE_CHECK_CUDA(cudaEventSynchronize(probe_cuda.stopped));
    PROBE_CHECK_CUDA(cudaGetLastError());
    PROBE_CHECK_CUDA(cudaEventElapsedTime(&milliseconds, probe_cuda.started, probe_cuda.stopped));
    return (double)milliseconds * PROBE_MICROSECONDS_PER_MILLISECOND;
}
PROBE_FUNCTION void probe_cuda_close(void) {
    PROBE_CHECK_CUDA(cudaEventDestroy(probe_cuda.started));
    PROBE_CHECK_CUDA(cudaEventDestroy(probe_cuda.stopped));
    while (probe_cuda.array_count > 0) { PROBE_CHECK_CUDA(cudaFree(probe_cuda.arrays[--probe_cuda.array_count])); }
}
#endif /* __CUDACC__ */
#endif /* PROBE_HARNESS_CUDA_H */
