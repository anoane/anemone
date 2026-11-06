/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * mapped_host_exchange_probe: what it costs to hand a few kilobytes from one card to another through page-locked
 * host memory that both cards map, with no host thread in the path.
 *
 * Two long-running kernels, one per card, play ping-pong over one mailbox in mapped host memory: the sender
 * stores a payload, then an epoch word; the receiver waits for the epoch, reads and checks the payload, and answers
 * with its own epoch word; the sender waits for that and starts the next round. One round is therefore one payload
 * hop plus one 4-byte acknowledgement, and thousands of rounds make the two launches negligible.
 *
 * For comparison the same payload is moved by the driver's card-to-card copy, one copy per round.
 * Every wait is bounded: a kernel that never sees its epoch gives up and reports it.
 * --corrupt makes the sender write one wrong word, and the receiver must report it (the proof that it can).
 *
 *   mapped_host_exchange_probe [--rounds N] [--corrupt]
 *
 * Measurement tool: not part of the engine.
 */
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define BYTES_PER_KIB 1024ull
#define MICROSECONDS_PER_SECOND 1.0e6
#define NANOSECONDS_PER_SECOND 1.0e9

#define MAX_CARDS 16
#define THREADS_PER_BLOCK 256
#define DEFAULT_ROUND_COUNT 2000u
#define SPIN_LIMIT 20000000u                     /* polls of the epoch word before a kernel gives up: seconds, not minutes */
#define WORDS_PER_LOAD 4
#define BYTES_PER_LOAD (WORDS_PER_LOAD * sizeof(uint32_t))
#define PATTERN_MULTIPLIER 2654435761u

/* Mailbox layout, in 32-bit words. The two epoch words sit on separate cache lines, the payload after them. */
#define MAILBOX_SENDER_EPOCH_WORD 0
#define MAILBOX_RECEIVER_EPOCH_WORD 16
#define MAILBOX_PAYLOAD_FIRST_WORD 64
#define MAILBOX_HEADER_BYTES (MAILBOX_PAYLOAD_FIRST_WORD * sizeof(uint32_t))

/* 16 B = the acknowledgement alone; 20 KiB = one tensor-parallel partial sum; 84 KiB = one pipeline hop of one
 * row; 120 and 160 KiB = hops with selection and candidate lists. */
static const size_t payload_sizes[] = {BYTES_PER_LOAD, 20 * BYTES_PER_KIB, 84 * BYTES_PER_KIB, 120 * BYTES_PER_KIB,
                                       160 * BYTES_PER_KIB};
#define PAYLOAD_SIZE_COUNT (sizeof(payload_sizes) / sizeof(payload_sizes[0]))
#define LARGEST_PAYLOAD_BYTES (160 * BYTES_PER_KIB)

struct exchange_status {
    uint32_t rounds_completed;
    uint32_t timed_out;
    uint32_t mismatched_loads;
};

#define CHECK_CUDA(call)                                                                              \
    do {                                                                                              \
        cudaError_t check_cuda_status = (call);                                                       \
        if (check_cuda_status != cudaSuccess) {                                                       \
            fprintf(stderr, "%s:%d: %s failed: %s\n", __FILE__, __LINE__, #call,                      \
                    cudaGetErrorString(check_cuda_status));                                           \
            exit(EXIT_FAILURE);                                                                       \
        }                                                                                             \
    } while (0)

__device__ __forceinline__ uint32_t payload_pattern(uint32_t round, uint32_t load_index)
{
    return (round * PATTERN_MULTIPLIER) ^ (load_index + 1u);
}

/* Polls one epoch word in mapped host memory until it holds the wanted round; returns 0 if it gave up. */
__device__ uint32_t wait_for_epoch(const volatile uint32_t *epoch_word, uint32_t wanted_round)
{
    for (uint32_t spins = 0; spins < SPIN_LIMIT; spins++) {
        if (*epoch_word == wanted_round) {
            return 1u;
        }
    }
    return 0u;
}

__global__ void sender_kernel(uint32_t *mailbox, uint32_t payload_load_count, uint32_t round_count,
                              uint32_t corrupted_round, struct exchange_status *status)
{
    __shared__ uint32_t round_succeeded;
    uint4 *payload = (uint4 *)(mailbox + MAILBOX_PAYLOAD_FIRST_WORD);
    uint32_t rounds_completed = 0;
    for (uint32_t round = 1; round <= round_count; round++) {
        for (uint32_t load_index = threadIdx.x; load_index < payload_load_count; load_index += blockDim.x) {
            uint32_t value = payload_pattern(round, load_index);
            if (round == corrupted_round && load_index == 0) {
                value = ~value;
            }
            payload[load_index] = make_uint4(value, value, value, value);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            __threadfence_system();
            mailbox[MAILBOX_SENDER_EPOCH_WORD] = round;
            __threadfence_system();
            round_succeeded = wait_for_epoch((const volatile uint32_t *)(mailbox + MAILBOX_RECEIVER_EPOCH_WORD), round);
        }
        __syncthreads();
        if (!round_succeeded) {
            break;
        }
        rounds_completed = round;
    }
    if (threadIdx.x == 0) {
        status->rounds_completed = rounds_completed;
        status->timed_out = rounds_completed != round_count;
    }
}

__global__ void receiver_kernel(uint32_t *mailbox, uint32_t payload_load_count, uint32_t round_count,
                                struct exchange_status *status)
{
    __shared__ uint32_t round_arrived;
    const uint4 *payload = (const uint4 *)(mailbox + MAILBOX_PAYLOAD_FIRST_WORD);
    uint32_t rounds_completed = 0;
    uint32_t mismatched_loads = 0;
    for (uint32_t round = 1; round <= round_count; round++) {
        if (threadIdx.x == 0) {
            round_arrived = wait_for_epoch((const volatile uint32_t *)(mailbox + MAILBOX_SENDER_EPOCH_WORD), round);
        }
        __syncthreads();
        if (!round_arrived) {
            break;
        }
        for (uint32_t load_index = threadIdx.x; load_index < payload_load_count; load_index += blockDim.x) {
            uint4 load = payload[load_index];
            uint32_t expected = payload_pattern(round, load_index);
            mismatched_loads += (load.x != expected) | (load.y != expected) | (load.z != expected) | (load.w != expected);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            __threadfence_system();
            mailbox[MAILBOX_RECEIVER_EPOCH_WORD] = round;
            __threadfence_system();
        }
        rounds_completed = round;
    }
    if (mismatched_loads != 0) {
        atomicAdd(&status->mismatched_loads, mismatched_loads);
    }
    if (threadIdx.x == 0) {
        status->rounds_completed = rounds_completed;
        status->timed_out = rounds_completed != round_count;
    }
}

static double seconds_now(void)
{
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (double)now.tv_sec + (double)now.tv_nsec / NANOSECONDS_PER_SECOND;
}

static struct exchange_status read_status(int card_index, struct exchange_status *card_status)
{
    struct exchange_status host_status;
    CHECK_CUDA(cudaSetDevice(card_index));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(&host_status, card_status, sizeof(host_status), cudaMemcpyDeviceToHost));
    return host_status;
}

/* One ping-pong run between two cards; returns microseconds per round, or a negative value on failure. */
static double run_mailbox_exchange(int sender_card, int receiver_card, uint32_t *mailbox, size_t payload_bytes,
                                   uint32_t round_count, uint32_t corrupted_round, uint32_t *mismatched_loads)
{
    struct exchange_status *sender_status = NULL;
    struct exchange_status *receiver_status = NULL;
    uint32_t payload_load_count = (uint32_t)(payload_bytes / BYTES_PER_LOAD);
    memset(mailbox, 0, MAILBOX_HEADER_BYTES + LARGEST_PAYLOAD_BYTES);
    CHECK_CUDA(cudaSetDevice(sender_card));
    CHECK_CUDA(cudaMalloc(&sender_status, sizeof(*sender_status)));
    CHECK_CUDA(cudaMemset(sender_status, 0, sizeof(*sender_status)));
    CHECK_CUDA(cudaSetDevice(receiver_card));
    CHECK_CUDA(cudaMalloc(&receiver_status, sizeof(*receiver_status)));
    CHECK_CUDA(cudaMemset(receiver_status, 0, sizeof(*receiver_status)));
    CHECK_CUDA(cudaDeviceSynchronize());

    double started = seconds_now();
    receiver_kernel<<<1, THREADS_PER_BLOCK>>>(mailbox, payload_load_count, round_count, receiver_status);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaSetDevice(sender_card));
    sender_kernel<<<1, THREADS_PER_BLOCK>>>(mailbox, payload_load_count, round_count, corrupted_round, sender_status);
    CHECK_CUDA(cudaGetLastError());
    struct exchange_status sender_result = read_status(sender_card, sender_status);
    struct exchange_status receiver_result = read_status(receiver_card, receiver_status);
    double elapsed_seconds = seconds_now() - started;

    CHECK_CUDA(cudaSetDevice(sender_card));
    CHECK_CUDA(cudaFree(sender_status));
    CHECK_CUDA(cudaSetDevice(receiver_card));
    CHECK_CUDA(cudaFree(receiver_status));
    *mismatched_loads = receiver_result.mismatched_loads;
    if (sender_result.timed_out || receiver_result.timed_out) {
        fprintf(stderr, "  timed out: sender finished %u rounds, receiver %u of %u\n", sender_result.rounds_completed,
                receiver_result.rounds_completed, round_count);
        return -1.0;
    }
    return elapsed_seconds * MICROSECONDS_PER_SECOND / (double)round_count;
}

/* The driver's own card-to-card copy of the same payload, one copy and one wait per round. */
static double run_driver_copy(int sender_card, int receiver_card, size_t payload_bytes, uint32_t round_count)
{
    void *sender_buffer = NULL;
    void *receiver_buffer = NULL;
    CHECK_CUDA(cudaSetDevice(sender_card));
    CHECK_CUDA(cudaMalloc(&sender_buffer, LARGEST_PAYLOAD_BYTES));
    CHECK_CUDA(cudaMemset(sender_buffer, 1, LARGEST_PAYLOAD_BYTES));
    CHECK_CUDA(cudaSetDevice(receiver_card));
    CHECK_CUDA(cudaMalloc(&receiver_buffer, LARGEST_PAYLOAD_BYTES));
    CHECK_CUDA(cudaDeviceSynchronize());
    double started = seconds_now();
    for (uint32_t round = 0; round < round_count; round++) {
        CHECK_CUDA(cudaMemcpyPeer(receiver_buffer, receiver_card, sender_buffer, sender_card, payload_bytes));
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    double elapsed_seconds = seconds_now() - started;
    CHECK_CUDA(cudaFree(receiver_buffer));
    CHECK_CUDA(cudaSetDevice(sender_card));
    CHECK_CUDA(cudaFree(sender_buffer));
    return elapsed_seconds * MICROSECONDS_PER_SECOND / (double)round_count;
}

int main(int argument_count, char **arguments)
{
    uint32_t round_count = DEFAULT_ROUND_COUNT;
    int corrupt_one_load = 0;
    for (int argument_index = 1; argument_index < argument_count; argument_index++) {
        if (strcmp(arguments[argument_index], "--corrupt") == 0) {
            corrupt_one_load = 1;
        } else if (strcmp(arguments[argument_index], "--rounds") == 0 && argument_index + 1 < argument_count) {
            long requested_rounds = strtol(arguments[++argument_index], NULL, 10);
            if (requested_rounds < 2 || requested_rounds > 10000000) {
                fprintf(stderr, "--rounds needs a number between 2 and 10000000\n");
                return EXIT_FAILURE;
            }
            round_count = (uint32_t)requested_rounds;
        } else {
            fprintf(stderr, "usage: %s [--rounds N] [--corrupt]\n", arguments[0]);
            return EXIT_FAILURE;
        }
    }
    int card_count = 0;
    char card_names[MAX_CARDS][256];
    /* Asked before any card is initialised, so it applies to every card this thread initialises. */
    cudaError_t flag_status = cudaSetDeviceFlags(cudaDeviceMapHost);
    if (flag_status != cudaSuccess) {
        fprintf(stderr, "mapping host memory could not be requested: %s\n", cudaGetErrorString(flag_status));
    }
    CHECK_CUDA(cudaGetDeviceCount(&card_count));
    if (card_count > MAX_CARDS) {
        card_count = MAX_CARDS;
    }
    if (card_count < 2) {
        fprintf(stderr, "needs at least two cards\n");
        return EXIT_FAILURE;
    }
    for (int card_index = 0; card_index < card_count; card_index++) {
        cudaDeviceProp properties;
        CHECK_CUDA(cudaSetDevice(card_index));
        CHECK_CUDA(cudaGetDeviceProperties(&properties, card_index));
        snprintf(card_names[card_index], sizeof(card_names[card_index]), "%s", properties.name);
    }
    uint32_t *mailbox = NULL;
    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaHostAlloc((void **)&mailbox, MAILBOX_HEADER_BYTES + LARGEST_PAYLOAD_BYTES,
                             cudaHostAllocMapped | cudaHostAllocPortable));

    int failures = 0;
    for (int sender_card = 0; sender_card < card_count; sender_card++) {
        for (int receiver_card = 0; receiver_card < card_count; receiver_card++) {
            if (sender_card == receiver_card) {
                continue;
            }
            int peer_access_offered = 0;
            CHECK_CUDA(cudaDeviceCanAccessPeer(&peer_access_offered, receiver_card, sender_card));
            printf("%s -> %s (driver offers peer access: %s)\n", card_names[sender_card], card_names[receiver_card],
                   peer_access_offered ? "yes, unverified" : "no");
            for (size_t size_index = 0; size_index < PAYLOAD_SIZE_COUNT; size_index++) {
                size_t payload_bytes = payload_sizes[size_index];
                uint32_t mismatched_loads = 0;
                uint32_t corrupted_round = corrupt_one_load ? round_count / 2 : 0;
                double mailbox_microseconds = run_mailbox_exchange(sender_card, receiver_card, mailbox, payload_bytes,
                                                                   round_count, corrupted_round, &mismatched_loads);
                double driver_microseconds = corrupt_one_load ? 0.0 : run_driver_copy(sender_card, receiver_card, payload_bytes, round_count);
                int payload_intact = mismatched_loads == 0;
                printf("  %8.2f KiB: mapped mailbox %8.2f us per round, driver copy %8.2f us per copy, payload %s (%u wrong loads)\n",
                       (double)payload_bytes / (double)BYTES_PER_KIB, mailbox_microseconds, driver_microseconds,
                       payload_intact ? "intact" : "DIFFERS", mismatched_loads);
                if (mailbox_microseconds < 0.0 || payload_intact == corrupt_one_load) {
                    failures++;
                }
            }
        }
    }
    CHECK_CUDA(cudaFreeHost(mailbox));
    if (failures != 0) {
        fprintf(stderr, "%d exchange(s) failed%s\n", failures, corrupt_one_load ? " to notice the corrupted load" : "");
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
