/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * ipc_holder_probe: can a restarted engine take over weights that are already on the cards, instead of uploading
 * them again?
 *
 * Several notes propose it (a small holder process keeps the cards' allocations and hands them to the engine over
 * CUDA inter-process handles), none measured it on these cards. A full load here takes 45 s, nearly all of it the
 * links. This probe plays both parts:
 *
 *   hold <handle file> [--gib N] [--corrupt]
 *        allocates N GiB on every card, fills it with a pattern that depends on the card and the position, writes
 *        one inter-process handle per card to the handle file, and waits until the file "<handle file>.stop"
 *        appears. With --corrupt it changes one word after the fill, which an attaching process must report.
 *   attach <handle file>
 *        a separate process: opens every handle, has a kernel sum the memory in place and compares the sum with
 *        the pattern's, and reports how long opening took and how long the check took. It then closes the handles
 *        and exits; the holder's memory must still be there for the next attach.
 *
 * Measurement tool: not part of the engine.
 */
#include <cuda_runtime.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

#define BYTES_PER_GIB (1024ull * 1024ull * 1024ull)
#define DEFAULT_GIB 16
#define MAX_CARDS 16
#define SUM_BLOCKS 1024
#define SUM_THREADS_PER_BLOCK 256
#define PATTERN_MULTIPLIER 0x9E3779B97F4A7C15ull
#define CORRUPTED_WORD_INDEX 12345ull
#define HANDLE_FILE_MAGIC 0x414e454d4f4e4531ull      /* identifies a handle file of this probe */
#define STOP_POLL_MICROSECONDS 200000

#define CHECK_CUDA(call)                                                                              \
    do {                                                                                              \
        cudaError_t check_cuda_status = (call);                                                       \
        if (check_cuda_status != cudaSuccess) {                                                       \
            fprintf(stderr, "%s:%d: %s failed: %s\n", __FILE__, __LINE__, #call,                      \
                    cudaGetErrorString(check_cuda_status));                                           \
            exit(EXIT_FAILURE);                                                                       \
        }                                                                                             \
    } while (0)

struct handle_record {
    uint64_t magic;
    uint64_t byte_count;
    uint64_t expected_sum;
    int card;
    int padding;
    cudaIpcMemHandle_t handle;
};

__host__ __device__ inline uint64_t pattern_word(uint64_t card, uint64_t word_index)
{
    return (word_index + 1ull) * PATTERN_MULTIPLIER + card;
}

__global__ void fill_kernel(unsigned long long *words, unsigned long long word_count, unsigned long long card)
{
    const unsigned long long thread_count = (unsigned long long)gridDim.x * blockDim.x;
    for (unsigned long long word = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; word < word_count; word += thread_count) {
        words[word] = pattern_word(card, word);
    }
}

__global__ void sum_kernel(const unsigned long long *words, unsigned long long word_count, unsigned long long *sum_out)
{
    const unsigned long long thread_count = (unsigned long long)gridDim.x * blockDim.x;
    unsigned long long sum = 0;
    for (unsigned long long word = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; word < word_count; word += thread_count) {
        sum += words[word];
    }
    atomicAdd(sum_out, sum);                     /* integer: the order of the additions does not matter */
}

static double seconds_now()
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static uint64_t sum_on_card(const unsigned long long *words, uint64_t word_count)
{
    unsigned long long *sum_on_the_card = NULL;
    unsigned long long sum = 0;
    CHECK_CUDA(cudaMalloc(&sum_on_the_card, sizeof(unsigned long long)));
    CHECK_CUDA(cudaMemset(sum_on_the_card, 0, sizeof(unsigned long long)));
    sum_kernel<<<SUM_BLOCKS, SUM_THREADS_PER_BLOCK>>>(words, word_count, sum_on_the_card);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaMemcpy(&sum, sum_on_the_card, sizeof(sum), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(sum_on_the_card));
    return sum;
}

static int hold(const char *handle_path, uint64_t byte_count, bool corrupt)
{
    int card_count = 0;
    char stop_path[4096];
    CHECK_CUDA(cudaGetDeviceCount(&card_count));
    if (card_count < 1 || card_count > MAX_CARDS || snprintf(stop_path, sizeof(stop_path), "%s.stop", handle_path) >= (int)sizeof(stop_path)) {
        return EXIT_FAILURE;
    }
    FILE *handle_file = fopen(handle_path, "wb");
    if (handle_file == NULL) {
        fprintf(stderr, "cannot write the handle file\n");
        return EXIT_FAILURE;
    }
    const uint64_t word_count = byte_count / sizeof(unsigned long long);
    for (int card = 0; card < card_count; card++) {
        unsigned long long *words = NULL;
        handle_record record;
        memset(&record, 0, sizeof(record));
        CHECK_CUDA(cudaSetDevice(card));
        const double started = seconds_now();
        CHECK_CUDA(cudaMalloc(&words, byte_count));
        fill_kernel<<<SUM_BLOCKS, SUM_THREADS_PER_BLOCK>>>(words, word_count, (unsigned long long)card);
        CHECK_CUDA(cudaDeviceSynchronize());
        record.expected_sum = sum_on_card(words, word_count);
        if (corrupt) {
            const unsigned long long wrong = 0;
            CHECK_CUDA(cudaMemcpy(words + CORRUPTED_WORD_INDEX, &wrong, sizeof(wrong), cudaMemcpyHostToDevice));
        }
        CHECK_CUDA(cudaIpcGetMemHandle(&record.handle, words));
        record.magic = HANDLE_FILE_MAGIC;
        record.byte_count = byte_count;
        record.card = card;
        if (fwrite(&record, sizeof(record), 1, handle_file) != 1) {
            return EXIT_FAILURE;
        }
        printf("holder: card %d holds %.0f GiB (allocated, filled and summed in %.2f s)\n", card,
               (double)byte_count / (double)BYTES_PER_GIB, seconds_now() - started);
    }
    if (fclose(handle_file) != 0) {
        return EXIT_FAILURE;
    }
    printf("holder: handles written; waiting for the stop file\n");
    fflush(stdout);
    while (access(stop_path, F_OK) != 0) {
        usleep(STOP_POLL_MICROSECONDS);
    }
    printf("holder: stopping\n");
    return EXIT_SUCCESS;
}

static int attach(const char *handle_path)
{
    FILE *handle_file = fopen(handle_path, "rb");
    handle_record record;
    int card_count = 0;
    bool all_right = true;
    if (handle_file == NULL) {
        fprintf(stderr, "cannot read the handle file\n");
        return EXIT_FAILURE;
    }
    const double process_started = seconds_now();
    while (fread(&record, sizeof(record), 1, handle_file) == 1) {
        void *memory = NULL;
        if (record.magic != HANDLE_FILE_MAGIC) {
            fprintf(stderr, "not a handle file of this probe\n");
            return EXIT_FAILURE;
        }
        CHECK_CUDA(cudaSetDevice(record.card));
        const double started = seconds_now();
        CHECK_CUDA(cudaIpcOpenMemHandle(&memory, record.handle, cudaIpcMemLazyEnablePeerAccess));
        const double opened = seconds_now();
        const uint64_t sum = sum_on_card((const unsigned long long *)memory, record.byte_count / sizeof(unsigned long long));
        const double checked = seconds_now();
        const bool right = sum == record.expected_sum;
        printf("attach: card %d, %.0f GiB taken over in %.4f s (first use of the card included), summed in place in %.3f s: %s\n",
               record.card, (double)record.byte_count / (double)BYTES_PER_GIB, opened - started, checked - opened,
               right ? "the bytes are the holder's" : "THE BYTES DIFFER");
        all_right = all_right && right;
        CHECK_CUDA(cudaIpcCloseMemHandle(memory));
        card_count++;
    }
    fclose(handle_file);
    printf("attach: %d cards in %.2f s in all: %s\n", card_count, seconds_now() - process_started,
           all_right && card_count > 0 ? "every card as the holder left it" : "NOT as the holder left it");
    return all_right && card_count > 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

int main(int argument_count, char **arguments)
{
    uint64_t gib = DEFAULT_GIB;
    bool corrupt = false;
    if (argument_count < 3 || (strcmp(arguments[1], "hold") != 0 && strcmp(arguments[1], "attach") != 0)) {
        fprintf(stderr, "usage: %s hold <handle file> [--gib N] [--corrupt] | attach <handle file>\n", arguments[0]);
        return EXIT_FAILURE;
    }
    for (int argument_index = 3; argument_index < argument_count; argument_index++) {
        if (strcmp(arguments[argument_index], "--corrupt") == 0) {
            corrupt = true;
        } else if (strcmp(arguments[argument_index], "--gib") == 0 && argument_index + 1 < argument_count) {
            gib = (uint64_t)atoi(arguments[++argument_index]);
        } else {
            fprintf(stderr, "unknown argument %s\n", arguments[argument_index]);
            return EXIT_FAILURE;
        }
    }
    if (strcmp(arguments[1], "hold") == 0) {
        return gib >= 1 ? hold(arguments[2], gib * BYTES_PER_GIB, corrupt) : EXIT_FAILURE;
    }
    return attach(arguments[2]);
}
