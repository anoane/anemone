/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * card_read_rate_probe: what one card can read per second, and what its link carries.
 *
 * For every visible card it reports
 *   - the facts the engine sizes its kernels from: SM count, shared memory per block, L2 size, free memory;
 *   - the rate at which a kernel reads a buffer far larger than L2 in 16-byte loads (the ceiling of any kernel
 *     that is bound by reading weights);
 *   - host-to-card and card-to-host copy rates from page-locked memory, by size of the single copy.
 *
 * The read is checked: the buffer holds a known pattern and the kernel's checksum must equal the host's.
 * --corrupt flips one word first, and the check must then fail (the proof that it can).
 *
 *   card_read_rate_probe [--buffer-gib N] [--corrupt]
 *
 * Measurement tool: not part of the engine.
 */
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BYTES_PER_KIB 1024ull
#define BYTES_PER_MIB (1024ull * BYTES_PER_KIB)
#define BYTES_PER_GIB (1024ull * BYTES_PER_MIB)
#define BYTES_PER_GB 1.0e9
#define MILLISECONDS_PER_SECOND 1000.0

#define DEFAULT_READ_BUFFER_GIB 8ull
#define WORDS_PER_LOAD 4                         /* one load is four 32-bit words = 16 bytes */
#define BYTES_PER_LOAD (WORDS_PER_LOAD * sizeof(uint32_t))
#define THREADS_PER_BLOCK 256
#define BLOCKS_PER_SM 4
#define READ_REPEATS 5
#define PATTERN_MULTIPLIER 2654435761u           /* Knuth's multiplicative hash constant: a cheap non-repeating fill */
#define COPY_TOTAL_BYTES (4ull * BYTES_PER_GIB)
#define COPY_HOST_BUFFER_BYTES (256ull * BYTES_PER_MIB)
#define MAX_CARDS 16

static const size_t copy_piece_sizes[] = {64 * BYTES_PER_KIB, 1 * BYTES_PER_MIB, 4 * BYTES_PER_MIB,
                                          16 * BYTES_PER_MIB, 64 * BYTES_PER_MIB, 256 * BYTES_PER_MIB};
#define COPY_PIECE_SIZE_COUNT (sizeof(copy_piece_sizes) / sizeof(copy_piece_sizes[0]))

#define CHECK_CUDA(call)                                                                              \
    do {                                                                                              \
        cudaError_t check_cuda_status = (call);                                                       \
        if (check_cuda_status != cudaSuccess) {                                                       \
            fprintf(stderr, "%s:%d: %s failed: %s\n", __FILE__, __LINE__, #call,                      \
                    cudaGetErrorString(check_cuda_status));                                           \
            exit(EXIT_FAILURE);                                                                       \
        }                                                                                             \
    } while (0)

__global__ void fill_pattern_kernel(uint32_t *words, uint64_t word_count)
{
    uint64_t thread_index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t thread_total = (uint64_t)gridDim.x * blockDim.x;
    for (uint64_t word_index = thread_index; word_index < word_count; word_index += thread_total) {
        words[word_index] = (uint32_t)word_index * PATTERN_MULTIPLIER;
    }
}

/* Each thread reads 16-byte loads spread evenly over the buffer and folds them into one word per block. */
__global__ void read_checksum_kernel(const uint4 *loads, uint64_t load_count, uint32_t *block_checksums)
{
    uint64_t thread_index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t thread_total = (uint64_t)gridDim.x * blockDim.x;
    uint32_t folded = 0;
    for (uint64_t load_index = thread_index; load_index < load_count; load_index += thread_total) {
        uint4 load = loads[load_index];
        folded ^= load.x ^ load.y ^ load.z ^ load.w;
    }
    __shared__ uint32_t block_fold[THREADS_PER_BLOCK];
    block_fold[threadIdx.x] = folded;
    __syncthreads();
    for (unsigned int stride = THREADS_PER_BLOCK / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            block_fold[threadIdx.x] ^= block_fold[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        block_checksums[blockIdx.x] = block_fold[0];
    }
}

static uint32_t host_pattern_checksum(uint64_t word_count)
{
    uint32_t folded = 0;
    for (uint64_t word_index = 0; word_index < word_count; word_index++) {
        folded ^= (uint32_t)word_index * PATTERN_MULTIPLIER;
    }
    return folded;
}

static double elapsed_milliseconds(cudaEvent_t started, cudaEvent_t stopped)
{
    float milliseconds = 0.0f;
    CHECK_CUDA(cudaEventSynchronize(stopped));
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, started, stopped));
    return (double)milliseconds;
}

static void measure_read_rate(int card_index, const cudaDeviceProp *properties, uint64_t buffer_bytes, int corrupt_one_word)
{
    uint64_t word_count = buffer_bytes / sizeof(uint32_t);
    uint64_t load_count = buffer_bytes / BYTES_PER_LOAD;
    unsigned int block_count = (unsigned int)properties->multiProcessorCount * BLOCKS_PER_SM;
    uint32_t *words = NULL;
    uint32_t *block_checksums = NULL;
    uint32_t *host_block_checksums = (uint32_t *)malloc(block_count * sizeof(uint32_t));
    cudaEvent_t started, stopped;
    if (host_block_checksums == NULL) {
        fprintf(stderr, "out of host memory\n");
        exit(EXIT_FAILURE);
    }
    CHECK_CUDA(cudaMalloc(&words, buffer_bytes));
    CHECK_CUDA(cudaMalloc(&block_checksums, block_count * sizeof(uint32_t)));
    CHECK_CUDA(cudaEventCreate(&started));
    CHECK_CUDA(cudaEventCreate(&stopped));
    fill_pattern_kernel<<<block_count, THREADS_PER_BLOCK>>>(words, word_count);
    CHECK_CUDA(cudaGetLastError());
    if (corrupt_one_word) {
        uint32_t wrong_word = ~0u;
        CHECK_CUDA(cudaMemcpy(words + word_count / 2, &wrong_word, sizeof(wrong_word), cudaMemcpyHostToDevice));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    double best_milliseconds = 0.0;
    for (int repeat = 0; repeat < READ_REPEATS; repeat++) {
        CHECK_CUDA(cudaEventRecord(started));
        read_checksum_kernel<<<block_count, THREADS_PER_BLOCK>>>((const uint4 *)words, load_count, block_checksums);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaEventRecord(stopped));
        double milliseconds = elapsed_milliseconds(started, stopped);
        if (repeat == 0 || milliseconds < best_milliseconds) {
            best_milliseconds = milliseconds;
        }
    }
    CHECK_CUDA(cudaMemcpy(host_block_checksums, block_checksums, block_count * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    uint32_t card_checksum = 0;
    for (unsigned int block_index = 0; block_index < block_count; block_index++) {
        card_checksum ^= host_block_checksums[block_index];
    }
    int checksum_matches = card_checksum == host_pattern_checksum(load_count * WORDS_PER_LOAD);
    printf("card %d read rate: %.0f GB/s over %.1f GiB in 16-byte loads (%u blocks of %d threads, best of %d), checksum %s\n",
           card_index, (double)(load_count * BYTES_PER_LOAD) / BYTES_PER_GB / (best_milliseconds / MILLISECONDS_PER_SECOND),
           (double)buffer_bytes / (double)BYTES_PER_GIB, block_count, THREADS_PER_BLOCK, READ_REPEATS,
           checksum_matches ? "matches" : "DIFFERS");
    CHECK_CUDA(cudaEventDestroy(started));
    CHECK_CUDA(cudaEventDestroy(stopped));
    CHECK_CUDA(cudaFree(block_checksums));
    CHECK_CUDA(cudaFree(words));
    free(host_block_checksums);
    if (checksum_matches == corrupt_one_word) {
        fprintf(stderr, "card %d: the checksum %s\n", card_index,
                corrupt_one_word ? "did not notice the corrupted word" : "does not match the pattern");
        exit(EXIT_FAILURE);
    }
}

static void measure_copy_rates(int card_index)
{
    void *host_buffer = NULL;
    void *card_buffer = NULL;
    cudaEvent_t started, stopped;
    cudaStream_t stream;
    CHECK_CUDA(cudaHostAlloc(&host_buffer, COPY_HOST_BUFFER_BYTES, cudaHostAllocDefault));
    memset(host_buffer, 1, COPY_HOST_BUFFER_BYTES);
    CHECK_CUDA(cudaMalloc(&card_buffer, COPY_HOST_BUFFER_BYTES));
    CHECK_CUDA(cudaStreamCreate(&stream));
    CHECK_CUDA(cudaEventCreate(&started));
    CHECK_CUDA(cudaEventCreate(&stopped));
    for (size_t size_index = 0; size_index < COPY_PIECE_SIZE_COUNT; size_index++) {
        size_t piece_bytes = copy_piece_sizes[size_index];
        size_t pieces_per_pass = COPY_HOST_BUFFER_BYTES / piece_bytes;
        size_t passes = COPY_TOTAL_BYTES / COPY_HOST_BUFFER_BYTES;
        double rate_gb_per_second[2];
        for (int direction = 0; direction < 2; direction++) {
            CHECK_CUDA(cudaEventRecord(started, stream));
            for (size_t pass = 0; pass < passes; pass++) {
                for (size_t piece = 0; piece < pieces_per_pass; piece++) {
                    char *host_piece = (char *)host_buffer + piece * piece_bytes;
                    char *card_piece = (char *)card_buffer + piece * piece_bytes;
                    if (direction == 0) {
                        CHECK_CUDA(cudaMemcpyAsync(card_piece, host_piece, piece_bytes, cudaMemcpyHostToDevice, stream));
                    } else {
                        CHECK_CUDA(cudaMemcpyAsync(host_piece, card_piece, piece_bytes, cudaMemcpyDeviceToHost, stream));
                    }
                }
            }
            CHECK_CUDA(cudaEventRecord(stopped, stream));
            double milliseconds = elapsed_milliseconds(started, stopped);
            rate_gb_per_second[direction] = (double)(passes * pieces_per_pass * piece_bytes) / BYTES_PER_GB /
                                            (milliseconds / MILLISECONDS_PER_SECOND);
        }
        printf("card %d copies of %8.2f MiB: host to card %6.2f GB/s, card to host %6.2f GB/s\n", card_index,
               (double)piece_bytes / (double)BYTES_PER_MIB, rate_gb_per_second[0], rate_gb_per_second[1]);
    }
    CHECK_CUDA(cudaEventDestroy(started));
    CHECK_CUDA(cudaEventDestroy(stopped));
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(card_buffer));
    CHECK_CUDA(cudaFreeHost(host_buffer));
}

int main(int argument_count, char **arguments)
{
    uint64_t buffer_bytes = DEFAULT_READ_BUFFER_GIB * BYTES_PER_GIB;
    int corrupt_one_word = 0;
    for (int argument_index = 1; argument_index < argument_count; argument_index++) {
        if (strcmp(arguments[argument_index], "--corrupt") == 0) {
            corrupt_one_word = 1;
        } else if (strcmp(arguments[argument_index], "--buffer-gib") == 0 && argument_index + 1 < argument_count) {
            long requested_gib = strtol(arguments[++argument_index], NULL, 10);
            if (requested_gib < 1) {
                fprintf(stderr, "--buffer-gib needs a positive number\n");
                return EXIT_FAILURE;
            }
            buffer_bytes = (uint64_t)requested_gib * BYTES_PER_GIB;
        } else {
            fprintf(stderr, "usage: %s [--buffer-gib N] [--corrupt]\n", arguments[0]);
            return EXIT_FAILURE;
        }
    }
    int card_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&card_count));
    if (card_count > MAX_CARDS) {
        card_count = MAX_CARDS;
    }
    for (int card_index = 0; card_index < card_count; card_index++) {
        cudaDeviceProp properties;
        size_t free_bytes = 0, total_bytes = 0;
        CHECK_CUDA(cudaSetDevice(card_index));
        CHECK_CUDA(cudaGetDeviceProperties(&properties, card_index));
        CHECK_CUDA(cudaMemGetInfo(&free_bytes, &total_bytes));
        printf("card %d %s: sm_%d%d, %d SMs, shared memory per block %zu B (opt-in %zu B), L2 %.1f MiB, memory %.2f of %.2f GiB free\n",
               card_index, properties.name, properties.major, properties.minor, properties.multiProcessorCount,
               properties.sharedMemPerBlock, properties.sharedMemPerBlockOptin,
               (double)properties.l2CacheSize / (double)BYTES_PER_MIB,
               (double)free_bytes / (double)BYTES_PER_GIB, (double)total_bytes / (double)BYTES_PER_GIB);
        uint64_t card_buffer_bytes = buffer_bytes;
        while (card_buffer_bytes + COPY_HOST_BUFFER_BYTES + BYTES_PER_GIB > free_bytes && card_buffer_bytes > BYTES_PER_GIB) {
            card_buffer_bytes /= 2;                                   /* a small card reads a smaller buffer */
        }
        measure_read_rate(card_index, &properties, card_buffer_bytes, corrupt_one_word);
        if (!corrupt_one_word) {
            measure_copy_rates(card_index);
        }
    }
    return EXIT_SUCCESS;
}
