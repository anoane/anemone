/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * host_cost_probe: two host-side costs that other engines report and that Anemone's loader and decode step depend
 * on, measured on this machine instead of taken on trust.
 *
 *   1. What it costs to make host memory page-locked for the cards (the loader's host arena and staging slots):
 *        - the driver's own allocation (cudaHostAlloc, portable);
 *        - memory the program maps and touches itself, on 4 KiB pages and on transparent 2 MiB pages, then
 *          registered with the driver (cudaHostRegister, portable).
 *      A recipe in the clones reports 3 to 38 s per 4.2 GB on 4 KiB pages against 0.2 s on 2 MiB pages (two RTX 5070
 *      Ti, DDR4). Each phase is timed apart and given in GiB/s.
 *   2. What a kernel launch costs the host while something else is busy. The same recipe reports about 250 us per
 *      launch instead of 8 while host memory or a link is saturated. Here: the time inside the launch call and the
 *      time until an empty kernel has finished, for one card at a time, (a) with nothing else running, (b) while
 *      another thread fills this card's link with page-locked copies, (c) while it fills another card's link, and
 *      (d) while a number of threads copy host memory as fast as they can.
 *
 *   host_cost_probe [--pin-gib N] [--launches N] [--memory-threads N]
 *
 * Measurement tool: not part of the engine.
 */
#include <cuda_runtime.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <pthread.h>
#include <sys/mman.h>
#include <vector>

#define BYTES_PER_GIB (1024.0 * 1024.0 * 1024.0)
#define BYTES_PER_MIB (1024ull * 1024ull)
#define SMALL_PAGE_BYTES 4096ull
#define HUGE_PAGE_BYTES (2ull * 1024ull * 1024ull)
#define DEFAULT_PIN_GIB 8
#define DEFAULT_LAUNCH_COUNT 4000
#define DEFAULT_MEMORY_THREADS 64
#define WARM_UP_LAUNCHES 200
#define COPY_BYTES (64ull * BYTES_PER_MIB)           /* one background copy: larger than any cache on the path */
#define MEMORY_HOG_BYTES (256ull * BYTES_PER_MIB)   /* per thread, source and destination each */
#define PERCENTILE_HIGH 0.99
#define MICROSECONDS_PER_SECOND 1.0e6
#define MAX_CARDS 16

#define CHECK_CUDA(call)                                                                              \
    do {                                                                                              \
        cudaError_t check_cuda_status = (call);                                                       \
        if (check_cuda_status != cudaSuccess) {                                                       \
            fprintf(stderr, "%s:%d: %s failed: %s\n", __FILE__, __LINE__, #call,                      \
                    cudaGetErrorString(check_cuda_status));                                           \
            exit(EXIT_FAILURE);                                                                       \
        }                                                                                             \
    } while (0)

__global__ void empty_kernel(unsigned int *counter) { counter[0] += 1u; }

static double seconds_now()
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

/* ---- part 1: page-locking ---- */

static void touch_every_page(unsigned char *memory, uint64_t byte_count, uint64_t page_bytes)
{
    for (uint64_t offset = 0; offset < byte_count; offset += page_bytes) {
        memory[offset] = (unsigned char)(offset / page_bytes);
    }
}

static uint64_t huge_pages_backing(void)          /* AnonHugePages of this process, in bytes */
{
    FILE *status = fopen("/proc/self/smaps_rollup", "r");
    char line[256];
    uint64_t kilobytes = 0;
    if (status == NULL) {
        return 0;
    }
    while (fgets(line, sizeof(line), status) != NULL) {
        if (sscanf(line, "AnonHugePages: %lu kB", &kilobytes) == 1) {
            break;
        }
    }
    fclose(status);
    return kilobytes * 1024ull;
}

static void measure_mapped_and_registered(uint64_t byte_count, bool huge_pages)
{
    const double map_started = seconds_now();
    unsigned char *memory = (unsigned char *)mmap(NULL, byte_count, PROT_READ | PROT_WRITE,
                                                  MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (memory == MAP_FAILED) {
        printf("  mapping %.0f GiB failed\n", (double)byte_count / BYTES_PER_GIB);
        return;
    }
    if (madvise(memory, byte_count, huge_pages ? MADV_HUGEPAGE : MADV_NOHUGEPAGE) != 0) {
        printf("  madvise refused: the page size below is whatever the kernel chose\n");
    }
    const uint64_t huge_before = huge_pages_backing();
    touch_every_page(memory, byte_count, SMALL_PAGE_BYTES);
    const double touched = seconds_now();
    const uint64_t huge_bytes = huge_pages_backing() - huge_before;
    CHECK_CUDA(cudaHostRegister(memory, byte_count, cudaHostRegisterPortable));
    const double registered = seconds_now();
    CHECK_CUDA(cudaHostUnregister(memory));
    const double unregistered = seconds_now();
    munmap(memory, byte_count);
    const double gib = (double)byte_count / BYTES_PER_GIB;
    printf("  mapped by the program, %s pages (%.0f%% of it on 2 MiB pages): touch %6.2f s (%6.2f GiB/s), register "
           "%6.2f s (%6.2f GiB/s), together %6.2f GiB/s; unregister %.2f s\n",
           huge_pages ? "2 MiB" : "4 KiB", 100.0 * (double)huge_bytes / (double)byte_count, touched - map_started,
           gib / (touched - map_started), registered - touched, gib / (registered - touched),
           gib / (registered - map_started), unregistered - registered);
}

static void measure_page_locking(uint64_t byte_count)
{
    const double gib = (double)byte_count / BYTES_PER_GIB;
    printf("page-locking %.0f GiB of host memory for the cards:\n", gib);
    void *driver_memory = NULL;
    const double started = seconds_now();
    CHECK_CUDA(cudaHostAlloc(&driver_memory, byte_count, cudaHostAllocPortable));
    const double allocated = seconds_now();
    touch_every_page((unsigned char *)driver_memory, byte_count, SMALL_PAGE_BYTES);
    const double touched = seconds_now();
    CHECK_CUDA(cudaFreeHost(driver_memory));
    const double freed = seconds_now();
    printf("  allocated by the driver:                  allocate %6.2f s (%6.2f GiB/s), first touch after it "
           "%.2f s, free %.2f s\n", allocated - started, gib / (allocated - started), touched - allocated,
           freed - touched);
    measure_mapped_and_registered(byte_count, false);
    measure_mapped_and_registered(byte_count, true);
}

/* ---- part 2: launches under load ---- */

struct background_load {
    std::atomic<bool> stop{false};
    int card = 0;                                   /* for a link load: the card whose link is filled */
    std::atomic<uint64_t> bytes_moved{0};
};

static void *link_load_thread(void *argument)
{
    background_load *load = (background_load *)argument;
    void *host_memory = NULL;
    void *card_memory = NULL;
    cudaStream_t stream;
    CHECK_CUDA(cudaSetDevice(load->card));
    CHECK_CUDA(cudaHostAlloc(&host_memory, COPY_BYTES, cudaHostAllocPortable));
    memset(host_memory, 1, COPY_BYTES);
    CHECK_CUDA(cudaMalloc(&card_memory, COPY_BYTES));
    CHECK_CUDA(cudaStreamCreate(&stream));
    while (!load->stop.load()) {
        CHECK_CUDA(cudaMemcpyAsync(card_memory, host_memory, COPY_BYTES, cudaMemcpyHostToDevice, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        load->bytes_moved.fetch_add(COPY_BYTES);
    }
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(card_memory));
    CHECK_CUDA(cudaFreeHost(host_memory));
    return NULL;
}

static void *memory_load_thread(void *argument)
{
    background_load *load = (background_load *)argument;
    unsigned char *source = (unsigned char *)malloc(MEMORY_HOG_BYTES);
    unsigned char *destination = (unsigned char *)malloc(MEMORY_HOG_BYTES);
    if (source == NULL || destination == NULL) {
        fprintf(stderr, "no memory for the memory load\n");
        exit(EXIT_FAILURE);
    }
    memset(source, 1, MEMORY_HOG_BYTES);
    while (!load->stop.load()) {
        memcpy(destination, source, MEMORY_HOG_BYTES);
        load->bytes_moved.fetch_add(MEMORY_HOG_BYTES);
    }
    free(source);
    free(destination);
    return NULL;
}

static void measure_launches(int card, int launch_count, const char *condition)
{
    unsigned int *counter = NULL;
    cudaStream_t stream;
    std::vector<double> call_microseconds((size_t)launch_count);
    std::vector<double> round_microseconds((size_t)launch_count);
    CHECK_CUDA(cudaSetDevice(card));
    CHECK_CUDA(cudaMalloc(&counter, sizeof(unsigned int)));
    CHECK_CUDA(cudaMemset(counter, 0, sizeof(unsigned int)));
    CHECK_CUDA(cudaStreamCreate(&stream));
    for (int launch = 0; launch < WARM_UP_LAUNCHES + launch_count; launch++) {
        const double started = seconds_now();
        empty_kernel<<<1, 1, 0, stream>>>(counter);
        const double returned = seconds_now();
        CHECK_CUDA(cudaStreamSynchronize(stream));
        const double finished = seconds_now();
        if (launch >= WARM_UP_LAUNCHES) {
            call_microseconds[(size_t)(launch - WARM_UP_LAUNCHES)] = (returned - started) * MICROSECONDS_PER_SECOND;
            round_microseconds[(size_t)(launch - WARM_UP_LAUNCHES)] = (finished - started) * MICROSECONDS_PER_SECOND;
        }
    }
    CHECK_CUDA(cudaGetLastError());
    unsigned int counted = 0;
    CHECK_CUDA(cudaMemcpy(&counted, counter, sizeof(counted), cudaMemcpyDeviceToHost));
    if (counted != (unsigned int)(WARM_UP_LAUNCHES + launch_count)) {
        printf("    WRONG: the kernel ran %u times, %d were launched\n", counted, WARM_UP_LAUNCHES + launch_count);
    }
    std::sort(call_microseconds.begin(), call_microseconds.end());
    std::sort(round_microseconds.begin(), round_microseconds.end());
    const size_t middle = (size_t)launch_count / 2;
    const size_t high = (size_t)((double)launch_count * PERCENTILE_HIGH);
    printf("    %-44s launch call %6.1f us median, %7.1f us at 99%%; until done %6.1f us median, %7.1f us at 99%%\n",
           condition, call_microseconds[middle], call_microseconds[high], round_microseconds[middle],
           round_microseconds[high]);
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(counter));
}

static void measure_launches_under(int card, int launch_count, const char *condition, int link_load_card,
                                   int memory_thread_count)
{
    std::vector<pthread_t> threads;
    background_load load;
    load.card = link_load_card;
    if (link_load_card >= 0) {
        pthread_t thread;
        pthread_create(&thread, NULL, link_load_thread, &load);
        threads.push_back(thread);
    }
    for (int thread_index = 0; thread_index < memory_thread_count; thread_index++) {
        pthread_t thread;
        pthread_create(&thread, NULL, memory_load_thread, &load);
        threads.push_back(thread);
    }
    if (!threads.empty()) {
        struct timespec settle = {1, 0};            /* let the load reach its rate before the launches are timed */
        nanosleep(&settle, NULL);
    }
    const uint64_t moved_before = load.bytes_moved.load();
    const double started = seconds_now();
    measure_launches(card, launch_count, condition);
    const double elapsed = seconds_now() - started;
    const uint64_t moved = load.bytes_moved.load() - moved_before;
    load.stop.store(true);
    for (pthread_t thread : threads) {
        pthread_join(thread, NULL);
    }
    if (!threads.empty()) {
        printf("      (the load moved %.1f GB/s meanwhile)\n", (double)moved / 1.0e9 / elapsed);
    }
}

int main(int argument_count, char **arguments)
{
    int pin_gib = DEFAULT_PIN_GIB;
    int launch_count = DEFAULT_LAUNCH_COUNT;
    int memory_threads = DEFAULT_MEMORY_THREADS;
    for (int argument_index = 1; argument_index < argument_count; argument_index++) {
        const bool has_value = argument_index + 1 < argument_count;
        if (strcmp(arguments[argument_index], "--pin-gib") == 0 && has_value) {
            pin_gib = atoi(arguments[++argument_index]);
        } else if (strcmp(arguments[argument_index], "--launches") == 0 && has_value) {
            launch_count = atoi(arguments[++argument_index]);
        } else if (strcmp(arguments[argument_index], "--memory-threads") == 0 && has_value) {
            memory_threads = atoi(arguments[++argument_index]);
        } else {
            fprintf(stderr, "usage: %s [--pin-gib N] [--launches N] [--memory-threads N]\n", arguments[0]);
            return EXIT_FAILURE;
        }
    }
    if (pin_gib < 1 || launch_count < 100 || memory_threads < 1) {
        fprintf(stderr, "--pin-gib at least 1, --launches at least 100, --memory-threads at least 1\n");
        return EXIT_FAILURE;
    }
    int card_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&card_count));
    if (card_count < 1 || card_count > MAX_CARDS) {
        fprintf(stderr, "this probe needs 1 to %d cards\n", MAX_CARDS);
        return EXIT_FAILURE;
    }
    for (int card = 0; card < card_count; card++) {  /* every context exists before anything is page-locked */
        CHECK_CUDA(cudaSetDevice(card));
        CHECK_CUDA(cudaFree(NULL));
    }
    CHECK_CUDA(cudaSetDevice(0));
    measure_page_locking((uint64_t)pin_gib * (uint64_t)BYTES_PER_GIB);

    printf("an empty kernel, %d launches each:\n", launch_count);
    for (int card = 0; card < card_count; card++) {
        cudaDeviceProp properties;
        CHECK_CUDA(cudaGetDeviceProperties(&properties, card));
        const int other_card = (card + 1) % card_count;
        char condition[96];
        printf("  card %d %s\n", card, properties.name);
        measure_launches_under(card, launch_count, "nothing else running:", -1, 0);
        measure_launches_under(card, launch_count, "copies fill this card's link:", card, 0);
        if (card_count > 1) {
            snprintf(condition, sizeof(condition), "copies fill the link of card %d:", other_card);
            measure_launches_under(card, launch_count, condition, other_card, 0);
        }
        snprintf(condition, sizeof(condition), "%d threads copy host memory:", memory_threads);
        measure_launches_under(card, launch_count, condition, -1, memory_threads);
    }
    return EXIT_SUCCESS;
}
