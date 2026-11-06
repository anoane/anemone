/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * link_hop_probe: what a prefill hop costs between every ordered pair of cards through page-locked host memory that
 * both cards map, and what each card's link carries while the other cards use theirs.
 *
 * It reports
 *   - each card with its PCI location, and for every ordered pair the driver's peer answers:
 *     cudaDeviceCanAccessPeer and the peer attributes (nothing is enabled);
 *   - for every ordered pair and hop size (58 and 160 MiB unless --hop-mib says otherwise) the time from the payload
 *     in the sender's memory to the payload in the receiver's, three ways:
 *       copies    the copy engines: a card-to-host copy into the shared pages, then a host-to-card copy out of them,
 *                 whole or cut through in pieces of --piece-mib (the receiver copies a piece in once an event says
 *                 the sender's copy of it landed: no host thread between the two cards);
 *       kernels   the same with kernels: the sender's stores each piece into the shared pages, the receiver's loads it;
 *       driver    cudaMemcpyPeerAsync: the driver's own path, which goes through the host when it offers no peer
 *                 access;
 *     every hop checked word by word on the receiver against a pattern that changes every run, so stale bytes fail;
 *   - each card's host-to-card and card-to-host rate in 64 MiB copies alone, then both directions at once, then with
 *     every card copying at once, every copy started by one word in mapped memory.
 * --corrupt flips one word of the sender's payload, and every hop must then report exactly one wrong word.
 *
 * --peer measures through peer mappings instead, and only on the ordered pairs that the gate (p2p_gate_probe) passed
 * on this boot, driver, module, options and cards: --gate-record names its record (p2p_record.h), which must be
 * trusted under this machine's key, or nothing runs. Peer access is enabled on exactly those pairs, and the first
 * check that fails through one of them ends all peer traffic. Per passed pair:
 *   pushes      payload, fence and flag stored into the receiver's memory, the receiver polling its own memory and
 *               acknowledging through the reverse mapping if the gate passed it, else through a host page: 16 B to
 *               400 KiB over 1 and 8 blocks, --rounds rounds of a pattern per round, each checked by the receiver;
 *   round trip  the 16 B push over one block acknowledged through the peer: a flag there and back, each fenced;
 *   sum         a two-rank sum of fp32 partials, 1 and 8 rows of 5,120, both ranks pushing into each other's memory,
 *               every received value and both ranks' last results checked bit for bit (pairs passed both ways);
 *   hops        58 and 160 MiB (or --hop-mib) one way and both ways at once, three ways: the copy engines
 *               (cudaMemcpyPeerAsync), the sender's kernel storing into the receiver's memory, the receiver's kernel
 *               loading from the sender's; each checked word by word on the receiver.
 * With --corrupt each of them runs damaged once (one bit of one word or value) and must report exactly that one (the
 * pushes and sums: at its place).
 * --stand-in-host runs the peer mode on every ordered pair with page-locked host memory, mapped into every card,
 * standing in for the memory a peer reaches (no record is read, nothing is enabled): the tool shown working where the
 * driver offers no peer access. Its figures are host pages', never a peer's.
 *
 *   link_hop_probe [--hop-mib N]... [--piece-mib N]... [--repeats N] [--write-combined] [--hops-only] [--corrupt]
 *   link_hop_probe --peer (--gate-record PATH | --stand-in-host) [--hop-mib N]... [--repeats N] [--rounds N]
 *                  [--corrupt]
 *
 * Measurement tool: not part of the engine.
 */
#include "p2p_common.cuh"

#define BYTES_PER_MIB (1024ull * 1024ull)
#define BYTES_PER_GB 1.0e9
#define MILLISECONDS_PER_SECOND 1000.0
#define NANOSECONDS_PER_SECOND 1.0e9

#define MAX_CARDS 16
#define MAX_SIZES 8
#define THREADS_PER_BLOCK 256
#define BLOCKS_PER_SM 4
#define LOADS_IN_FLIGHT 4                        /* loads a thread of the copy kernel issues before it stores them */
#define LOAD_BYTES sizeof(uint4)
#define PATTERN_MULTIPLIER 2654435761u           /* Knuth's multiplicative hash constant, as the other probes */
#define SEED_STEP 0x9e3779b9u                    /* the golden-ratio constant: each run's pattern differs everywhere */
#define CORRUPTED_WORD_PAST_MIDDLE 3
#define DEFAULT_REPEATS 3
#define MAX_REPEATS 100
#define MAX_HOP_MIB 1024
#define SPIN_LIMIT 20000000u                     /* polls of the gate word before a kernel gives up: seconds */
#define CARD_MEMORY_MARGIN_BYTES (512ull * BYTES_PER_MIB)
#define WAIT_DEADLINE_SECONDS 120.0              /* a hop, fill, check or shared run that takes longer: a fault */

#define LINK_PIECE_BYTES (64ull * BYTES_PER_MIB)
#define LINK_BUFFER_PIECES 4                     /* the copies cycle over 256 MiB on each side */
#define LINK_BUFFER_BYTES (LINK_BUFFER_PIECES * LINK_PIECE_BYTES)
#define LINK_ALONE_PIECES 16                     /* 1 GiB per card and direction alone */
#define LINK_SHARED_SECONDS 0.5                  /* what each copy of a shared run is sized to last */
#define LINK_FEWEST_PIECES 4
#define LINK_MOST_PIECES 256
#define TO_CARD 0
#define TO_HOST 1
#define DIRECTIONS 2

/* The prefill hop sizes of the plan: 160 MiB is 4,096 rows of four bfloat16 streams of 5,120. Pieces of 0 MiB stand
 * for the whole hop in one piece, which is always run. */
static const unsigned int default_hop_mib[] = {58, 160};
static const unsigned int default_piece_mib[] = {1, 4, 16};
static const char *const direction_names[DIRECTIONS] = {"host to card", "card to host"};

enum hop_way { HOP_COPIES, HOP_KERNELS, HOP_DRIVER, HOP_WAY_COUNT };
static const char *const hop_way_names[HOP_WAY_COUNT] = {"copies", "kernels", "driver copy"};

#define CHECK_CUDA(call)                                                                              \
    do {                                                                                              \
        cudaError_t check_cuda_status = (call);                                                       \
        if (check_cuda_status != cudaSuccess) {                                                       \
            fprintf(stderr, "%s:%d: %s failed: %s\n", __FILE__, __LINE__, #call,                      \
                    cudaGetErrorString(check_cuda_status));                                           \
            exit(EXIT_FAILURE);                                                                       \
        }                                                                                             \
    } while (0)

struct card {
    int index;
    char name[256];
    int pci_domain, pci_bus, pci_device;
    unsigned int block_count;
    cudaStream_t stream;
    uint32_t *payload;                           /* the largest hop: a sender's source, a receiver's destination */
    uint32_t *block_counts;                      /* one mismatch count per block of the check */
    uint32_t *host_block_counts;
    char *staging_view;                          /* the shared pages as this card addresses them */
    uint32_t *gate_view;
    uint32_t *gave_up_view;
    cudaStream_t link_streams[DIRECTIONS];
    char *link_host[DIRECTIONS];
    char *link_card[DIRECTIONS];
    cudaEvent_t link_started[DIRECTIONS], link_stopped[DIRECTIONS];
    double alone_gb_per_second[DIRECTIONS];
};

__host__ __device__ __forceinline__ uint32_t pattern_word(uint64_t word_index, uint32_t seed)
{
    return ((uint32_t)word_index * PATTERN_MULTIPLIER) ^ seed;
}

__global__ void fill_kernel(uint32_t *words, uint64_t word_count, uint32_t seed, uint32_t flipped_bits)
{
    uint64_t thread_index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t thread_total = (uint64_t)gridDim.x * blockDim.x;
    for (uint64_t word_index = thread_index; word_index < word_count; word_index += thread_total) {
        words[word_index] = pattern_word(word_index, seed) ^ flipped_bits;
    }
}

/* One count per block, summed by the host: no atomic operation. */
__global__ void count_mismatches_kernel(const uint32_t *words, uint64_t word_count, uint32_t seed,
                                        uint32_t *block_counts)
{
    __shared__ uint32_t block_sums[THREADS_PER_BLOCK];
    uint64_t thread_index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t thread_total = (uint64_t)gridDim.x * blockDim.x;
    uint32_t mismatches = 0;
    for (uint64_t word_index = thread_index; word_index < word_count; word_index += thread_total) {
        mismatches += words[word_index] != pattern_word(word_index, seed);
    }
    block_sums[threadIdx.x] = mismatches;
    __syncthreads();
    for (unsigned int stride = THREADS_PER_BLOCK / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            block_sums[threadIdx.x] += block_sums[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        block_counts[blockIdx.x] = block_sums[0];
    }
}

/* 16-byte loads, LOADS_IN_FLIGHT of them per thread before the stores, so that reads of host memory overlap. */
__global__ void copy_kernel(uint4 *destination, const uint4 *source, uint64_t load_count)
{
    uint64_t thread_index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t thread_total = (uint64_t)gridDim.x * blockDim.x;
    for (uint64_t first = thread_index; first < load_count; first += thread_total * LOADS_IN_FLIGHT) {
        uint4 loads[LOADS_IN_FLIGHT];
#pragma unroll
        for (int step = 0; step < LOADS_IN_FLIGHT; step++) {
            uint64_t load_index = first + (uint64_t)step * thread_total;
            loads[step] = load_index < load_count ? source[load_index] : make_uint4(0u, 0u, 0u, 0u);
        }
#pragma unroll
        for (int step = 0; step < LOADS_IN_FLIGHT; step++) {
            uint64_t load_index = first + (uint64_t)step * thread_total;
            if (load_index < load_count) {
                destination[load_index] = loads[step];
            }
        }
    }
}

/* Holds a stream until the host writes the wanted generation into the gate word; gives up after SPIN_LIMIT polls. */
__global__ void wait_at_gate_kernel(const volatile uint32_t *gate, uint32_t wanted_generation,
                                    volatile uint32_t *gave_up)
{
    for (uint32_t spins = 0; spins < SPIN_LIMIT; spins++) {
        if (*gate == wanted_generation) {
            return;
        }
    }
    *gave_up = 1u;
}

static double seconds_now(void)
{
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (double)now.tv_sec + (double)now.tv_nsec / NANOSECONDS_PER_SECOND;
}

/* A queued operation that does not end within the deadline is a fault, never a long wait. */
static void wait_stream(int card_index, cudaStream_t stream)
{
    if (!p2p_wait_stream(card_index, stream, WAIT_DEADLINE_SECONDS)) {
        fprintf(stderr, "card %d: a queued operation did not end within %.0f s\n", card_index, WAIT_DEADLINE_SECONDS);
        exit(EXIT_FAILURE);
    }
}

static double gb_per_second(double bytes, double milliseconds)
{
    return bytes / BYTES_PER_GB / (milliseconds / MILLISECONDS_PER_SECOND);
}

static void sort_ascending(double *values, unsigned int count)
{
    for (unsigned int at = 1; at < count; at++) {
        double held = values[at];
        unsigned int before = at;
        while (before > 0 && values[before - 1] > held) {
            values[before] = values[before - 1];
            before--;
        }
        values[before] = held;
    }
}

static const char *bus_text(const struct card *card)
{
    static char texts[MAX_CARDS][32];
    snprintf(texts[card->index], sizeof(texts[card->index]), "%02x:%02x.0", (unsigned int)card->pci_bus,
             (unsigned int)card->pci_device);
    return texts[card->index];
}

/* ---- the hops ---- */

static void enqueue_hop_piece(enum hop_way way, struct card *sender, struct card *receiver, char *staging,
                              size_t offset, size_t length, cudaEvent_t landed)
{
    CHECK_CUDA(cudaSetDevice(sender->index));
    if (way == HOP_COPIES) {
        CHECK_CUDA(cudaMemcpyAsync(staging + offset, (char *)sender->payload + offset, length, cudaMemcpyDeviceToHost,
                                   sender->stream));
    } else {
        copy_kernel<<<sender->block_count, THREADS_PER_BLOCK, 0, sender->stream>>>(
            (uint4 *)(sender->staging_view + offset), (const uint4 *)((char *)sender->payload + offset),
            length / LOAD_BYTES);
        CHECK_CUDA(cudaGetLastError());
    }
    CHECK_CUDA(cudaEventRecord(landed, sender->stream));
    CHECK_CUDA(cudaSetDevice(receiver->index));
    CHECK_CUDA(cudaStreamWaitEvent(receiver->stream, landed, 0));
    if (way == HOP_COPIES) {
        CHECK_CUDA(cudaMemcpyAsync((char *)receiver->payload + offset, staging + offset, length, cudaMemcpyHostToDevice,
                                   receiver->stream));
    } else {
        copy_kernel<<<receiver->block_count, THREADS_PER_BLOCK, 0, receiver->stream>>>(
            (uint4 *)((char *)receiver->payload + offset), (const uint4 *)(receiver->staging_view + offset),
            length / LOAD_BYTES);
        CHECK_CUDA(cudaGetLastError());
    }
}

/* One hop of hop_bytes from sender to receiver; returns its milliseconds and the receiver's wrong words. A piece of
 * 0 bytes is the whole hop. */
static double run_hop(enum hop_way way, struct card *sender, struct card *receiver, char *staging,
                      cudaEvent_t *piece_events, size_t hop_bytes, size_t piece_bytes, uint32_t seed,
                      int corrupt_one_word, uint64_t *wrong_words)
{
    uint64_t word_count = hop_bytes / sizeof(uint32_t);
    CHECK_CUDA(cudaSetDevice(sender->index));
    fill_kernel<<<sender->block_count, THREADS_PER_BLOCK, 0, sender->stream>>>(sender->payload, word_count, seed, 0u);
    CHECK_CUDA(cudaGetLastError());
    wait_stream(sender->index, sender->stream);
    if (corrupt_one_word) {
        uint64_t damaged_index = word_count / 2 + CORRUPTED_WORD_PAST_MIDDLE;
        uint32_t wrong_word = ~pattern_word(damaged_index, seed);
        CHECK_CUDA(cudaMemcpy(sender->payload + damaged_index, &wrong_word, sizeof(wrong_word),
                              cudaMemcpyHostToDevice));
    }
    CHECK_CUDA(cudaSetDevice(receiver->index));
    fill_kernel<<<receiver->block_count, THREADS_PER_BLOCK, 0, receiver->stream>>>(receiver->payload, word_count, seed,
                                                                                   ~0u);  /* every word must change */
    CHECK_CUDA(cudaGetLastError());
    wait_stream(receiver->index, receiver->stream);

    double started = seconds_now();
    if (way == HOP_DRIVER) {
        CHECK_CUDA(cudaSetDevice(sender->index));
        CHECK_CUDA(cudaMemcpyPeerAsync(receiver->payload, receiver->index, sender->payload, sender->index, hop_bytes,
                                       sender->stream));
    } else {
        size_t piece_length = piece_bytes == 0 ? hop_bytes : piece_bytes;
        unsigned int piece_index = 0;
        for (size_t offset = 0; offset < hop_bytes; offset += piece_length, piece_index++) {
            size_t length = hop_bytes - offset < piece_length ? hop_bytes - offset : piece_length;
            enqueue_hop_piece(way, sender, receiver, staging, offset, length, piece_events[piece_index]);
        }
    }
    wait_stream(receiver->index, receiver->stream);
    wait_stream(sender->index, sender->stream);
    double milliseconds = (seconds_now() - started) * MILLISECONDS_PER_SECOND;

    CHECK_CUDA(cudaSetDevice(receiver->index));
    count_mismatches_kernel<<<receiver->block_count, THREADS_PER_BLOCK, 0, receiver->stream>>>(
        receiver->payload, word_count, seed, receiver->block_counts);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaMemcpyAsync(receiver->host_block_counts, receiver->block_counts,
                               receiver->block_count * sizeof(uint32_t), cudaMemcpyDeviceToHost, receiver->stream));
    wait_stream(receiver->index, receiver->stream);
    *wrong_words = 0;
    for (unsigned int block_index = 0; block_index < receiver->block_count; block_index++) {
        *wrong_words += receiver->host_block_counts[block_index];
    }
    return milliseconds;
}

/* Every way and piece size for one ordered pair and one hop size; returns the number of cases whose check failed. */
static int measure_hop_size(struct card *sender, struct card *receiver, char *staging, cudaEvent_t *piece_events,
                            unsigned int hop_mib, const unsigned int *piece_mib, int piece_count, unsigned int repeats,
                            int corrupt_one_word, uint32_t *run_counter)
{
    int failed_cases = 0;
    size_t hop_bytes = (size_t)hop_mib * BYTES_PER_MIB;
    uint64_t wanted_wrong_words = corrupt_one_word ? 1u : 0u;
    for (int way = 0; way < HOP_WAY_COUNT; way++) {
        int piece_case_count = way == HOP_DRIVER ? 1 : 1 + piece_count;     /* the whole hop, then each piece size */
        for (int piece_case = 0; piece_case < piece_case_count; piece_case++) {
            unsigned int case_piece_mib = piece_case == 0 ? 0u : piece_mib[piece_case - 1];
            if (case_piece_mib >= hop_mib) {
                continue;                                                   /* one piece: the whole hop again */
            }
            double milliseconds[MAX_REPEATS];
            unsigned int failed_runs = 0;
            uint64_t worst_wrong_words = 0;
            for (unsigned int repeat = 0; repeat < repeats; repeat++) {
                uint64_t wrong_words = 0;
                uint32_t seed = ++*run_counter * SEED_STEP;
                milliseconds[repeat] = run_hop((enum hop_way)way, sender, receiver, staging, piece_events, hop_bytes,
                                               (size_t)case_piece_mib * BYTES_PER_MIB, seed, corrupt_one_word,
                                               &wrong_words);
                if (wrong_words != wanted_wrong_words) {
                    failed_runs++;
                    if (wrong_words > worst_wrong_words) {
                        worst_wrong_words = wrong_words;
                    }
                }
            }
            sort_ascending(milliseconds, repeats);
            char piece_text[32];
            if (case_piece_mib == 0) {
                snprintf(piece_text, sizeof(piece_text), "%s", way == HOP_DRIVER ? "one call" : "whole");
            } else {
                snprintf(piece_text, sizeof(piece_text), "pieces of %u MiB", case_piece_mib);
            }
            printf("  %4u MiB %-11s %-16s best %8.2f ms (%6.2f GB/s), median %8.2f ms, %u runs, ", hop_mib,
                   hop_way_names[way], piece_text, milliseconds[0], gb_per_second((double)hop_bytes, milliseconds[0]),
                   milliseconds[repeats / 2], repeats);
            if (failed_runs == 0) {
                printf("%s\n", corrupt_one_word ? "the damaged word reported in every run" : "payload intact");
            } else {
                printf("CHECK FAILED in %u runs (up to %llu wrong words, %llu wanted)\n", failed_runs,
                       (unsigned long long)worst_wrong_words, (unsigned long long)wanted_wrong_words);
                failed_cases++;
            }
        }
    }
    return failed_cases;
}

/* ---- the links, alone and shared ---- */

static uint32_t gate_generation;

/* One shared run: every (card, direction) with pieces waits at the gate, then copies its pieces of 64 MiB. Returns 0
 * if a stream gave up waiting; fills each copy's own milliseconds and the window from the gate's opening to the
 * last copy's end. */
static int run_links(struct card *cards, int card_count, unsigned int pieces[][DIRECTIONS], uint32_t *gate,
                     uint32_t *gave_up, double milliseconds[][DIRECTIONS], double *window_milliseconds)
{
    gate_generation++;
    for (int card_index = 0; card_index < card_count; card_index++) {
        ((volatile uint32_t *)gave_up)[card_index] = 0u;
    }
    for (int card_index = 0; card_index < card_count; card_index++) {
        struct card *card = &cards[card_index];
        CHECK_CUDA(cudaSetDevice(card->index));
        for (int direction = 0; direction < DIRECTIONS; direction++) {
            if (pieces[card_index][direction] == 0) {
                continue;
            }
            cudaStream_t stream = card->link_streams[direction];
            wait_at_gate_kernel<<<1, 1, 0, stream>>>(card->gate_view, gate_generation, card->gave_up_view + card_index);
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaEventRecord(card->link_started[direction], stream));
            for (unsigned int piece = 0; piece < pieces[card_index][direction]; piece++) {
                size_t offset = (size_t)(piece % LINK_BUFFER_PIECES) * LINK_PIECE_BYTES;
                if (direction == TO_CARD) {
                    CHECK_CUDA(cudaMemcpyAsync(card->link_card[TO_CARD] + offset, card->link_host[TO_CARD] + offset,
                                               LINK_PIECE_BYTES, cudaMemcpyHostToDevice, stream));
                } else {
                    CHECK_CUDA(cudaMemcpyAsync(card->link_host[TO_HOST] + offset, card->link_card[TO_HOST] + offset,
                                               LINK_PIECE_BYTES, cudaMemcpyDeviceToHost, stream));
                }
            }
            CHECK_CUDA(cudaEventRecord(card->link_stopped[direction], stream));
        }
    }
    double opened = seconds_now();
    __sync_synchronize();
    *(volatile uint32_t *)gate = gate_generation;
    __sync_synchronize();
    for (int card_index = 0; card_index < card_count; card_index++) {
        for (int direction = 0; direction < DIRECTIONS; direction++) {
            if (pieces[card_index][direction] != 0) {
                wait_stream(cards[card_index].index, cards[card_index].link_streams[direction]);
            }
        }
    }
    *window_milliseconds = (seconds_now() - opened) * MILLISECONDS_PER_SECOND;
    int every_gate_held = 1;
    for (int card_index = 0; card_index < card_count; card_index++) {
        if (((volatile uint32_t *)gave_up)[card_index] != 0u) {
            fprintf(stderr, "card %d gave up waiting at the gate\n", cards[card_index].index);
            every_gate_held = 0;
        }
        for (int direction = 0; direction < DIRECTIONS; direction++) {
            milliseconds[card_index][direction] = 0.0;
            if (pieces[card_index][direction] != 0) {
                float elapsed = 0.0f;
                CHECK_CUDA(cudaEventElapsedTime(&elapsed, cards[card_index].link_started[direction],
                                                cards[card_index].link_stopped[direction]));
                milliseconds[card_index][direction] = (double)elapsed;
            }
        }
    }
    return every_gate_held;
}

/* Prints one shared run: each copy's rate beside its rate alone, and the sum over the window. */
static void print_links(const char *title, const struct card *cards, int card_count,
                        unsigned int pieces[][DIRECTIONS], double milliseconds[][DIRECTIONS],
                        double window_milliseconds)
{
    double total_bytes = 0.0;
    for (int card_index = 0; card_index < card_count; card_index++) {
        for (int direction = 0; direction < DIRECTIONS; direction++) {
            total_bytes += (double)pieces[card_index][direction] * (double)LINK_PIECE_BYTES;
        }
    }
    printf("%s: %.2f GB/s in all over %.0f ms\n", title, gb_per_second(total_bytes, window_milliseconds),
           window_milliseconds);
    for (int card_index = 0; card_index < card_count; card_index++) {
        for (int direction = 0; direction < DIRECTIONS; direction++) {
            if (pieces[card_index][direction] == 0) {
                continue;
            }
            double bytes = (double)pieces[card_index][direction] * (double)LINK_PIECE_BYTES;
            printf("  card %d (%s) %s %6.2f GB/s (alone %6.2f), %u copies of 64 MiB\n", cards[card_index].index,
                   bus_text(&cards[card_index]), direction_names[direction],
                   gb_per_second(bytes, milliseconds[card_index][direction]),
                   cards[card_index].alone_gb_per_second[direction], pieces[card_index][direction]);
        }
    }
}

static unsigned int pieces_for_shared_run(double alone_gb_per_second)
{
    double pieces = alone_gb_per_second * BYTES_PER_GB * LINK_SHARED_SECONDS / (double)LINK_PIECE_BYTES;
    if (pieces < LINK_FEWEST_PIECES) {
        return LINK_FEWEST_PIECES;
    }
    if (pieces > LINK_MOST_PIECES) {
        return LINK_MOST_PIECES;
    }
    return (unsigned int)pieces;
}

/* Returns the number of runs in which a stream gave up waiting at the gate. */
static int measure_links(struct card *cards, int card_count, uint32_t *gate, uint32_t *gave_up)
{
    unsigned int pieces[MAX_CARDS][DIRECTIONS];
    double milliseconds[MAX_CARDS][DIRECTIONS];
    double window_milliseconds = 0.0;
    int failed_runs = 0;
    printf("\nlinks, 64 MiB copies from and to page-locked host memory\n");
    for (int card_index = 0; card_index < card_count; card_index++) {
        for (int direction = 0; direction < DIRECTIONS; direction++) {
            memset(pieces, 0, sizeof(pieces));
            pieces[card_index][direction] = LINK_ALONE_PIECES;
            failed_runs += !run_links(cards, card_count, pieces, gate, gave_up, milliseconds, &window_milliseconds);
            cards[card_index].alone_gb_per_second[direction] =
                gb_per_second((double)LINK_ALONE_PIECES * (double)LINK_PIECE_BYTES,
                              milliseconds[card_index][direction]);
        }
        printf("card %d (%s) alone: host to card %6.2f GB/s, card to host %6.2f GB/s\n", cards[card_index].index,
               bus_text(&cards[card_index]), cards[card_index].alone_gb_per_second[TO_CARD],
               cards[card_index].alone_gb_per_second[TO_HOST]);
    }
    for (int card_index = 0; card_index < card_count; card_index++) {
        char title[96];
        memset(pieces, 0, sizeof(pieces));
        for (int direction = 0; direction < DIRECTIONS; direction++) {
            pieces[card_index][direction] = pieces_for_shared_run(cards[card_index].alone_gb_per_second[direction]);
        }
        failed_runs += !run_links(cards, card_count, pieces, gate, gave_up, milliseconds, &window_milliseconds);
        snprintf(title, sizeof(title), "card %d (%s) alone, both directions at once", cards[card_index].index,
                 bus_text(&cards[card_index]));
        print_links(title, cards, card_count, pieces, milliseconds, window_milliseconds);
    }
    static const char *const shared_titles[] = {"every card at once, host to card", "every card at once, card to host",
                                                "every card at once, both directions"};
    static const int shared_directions[][DIRECTIONS] = {{1, 0}, {0, 1}, {1, 1}};
    for (int shared = 0; shared < (int)(sizeof(shared_titles) / sizeof(shared_titles[0])); shared++) {
        memset(pieces, 0, sizeof(pieces));
        for (int card_index = 0; card_index < card_count; card_index++) {
            for (int direction = 0; direction < DIRECTIONS; direction++) {
                if (shared_directions[shared][direction]) {
                    pieces[card_index][direction] =
                        pieces_for_shared_run(cards[card_index].alone_gb_per_second[direction]);
                }
            }
        }
        failed_runs += !run_links(cards, card_count, pieces, gate, gave_up, milliseconds, &window_milliseconds);
        print_links(shared_titles[shared], cards, card_count, pieces, milliseconds, window_milliseconds);
    }
    return failed_runs;
}

#include "link_hop_peer.cuh"

/* ---- setup ---- */

static int read_size_list(const char *text, unsigned int *values, int *count, unsigned int lowest, unsigned int highest)
{
    char *end = NULL;
    long value = strtol(text, &end, 10);
    if (end == text || *end != '\0' || value < (long)lowest || value > (long)highest || *count >= MAX_SIZES) {
        return 0;
    }
    values[(*count)++] = (unsigned int)value;
    return 1;
}

static void open_card(struct card *card, int card_index, size_t payload_bytes, int hops_only)
{
    cudaDeviceProp properties;
    size_t free_bytes = 0, total_bytes = 0;
    size_t wanted_bytes = payload_bytes + (hops_only ? 0 : DIRECTIONS * LINK_BUFFER_BYTES) + CARD_MEMORY_MARGIN_BYTES;
    card->index = card_index;
    CHECK_CUDA(cudaSetDevice(card_index));
    CHECK_CUDA(cudaGetDeviceProperties(&properties, card_index));
    snprintf(card->name, sizeof(card->name), "%s", properties.name);
    card->pci_domain = properties.pciDomainID;
    card->pci_bus = properties.pciBusID;
    card->pci_device = properties.pciDeviceID;
    card->block_count = (unsigned int)properties.multiProcessorCount * BLOCKS_PER_SM;
    if (!properties.unifiedAddressing || !properties.canMapHostMemory) {
        fprintf(stderr, "card %d cannot map host memory into one address space\n", card_index);
        exit(EXIT_FAILURE);
    }
    CHECK_CUDA(cudaMemGetInfo(&free_bytes, &total_bytes));
    printf("card %d %s at %04x:%s: sm_%d%d, %d SMs, %.2f of %.2f GiB free\n", card_index, card->name,
           (unsigned int)card->pci_domain,
           bus_text(card), properties.major, properties.minor, properties.multiProcessorCount,
           (double)free_bytes / (double)(1024ull * BYTES_PER_MIB),
           (double)total_bytes / (double)(1024ull * BYTES_PER_MIB));
    if (free_bytes < wanted_bytes) {
        fprintf(stderr, "card %d has %zu bytes free, the probe needs %zu\n", card_index, free_bytes, wanted_bytes);
        exit(EXIT_FAILURE);
    }
    CHECK_CUDA(cudaStreamCreateWithFlags(&card->stream, cudaStreamNonBlocking));
    CHECK_CUDA(cudaMalloc(&card->payload, payload_bytes));
    CHECK_CUDA(cudaMalloc(&card->block_counts, card->block_count * sizeof(uint32_t)));
    card->host_block_counts = (uint32_t *)malloc(card->block_count * sizeof(uint32_t));
    if (card->host_block_counts == NULL) {
        fprintf(stderr, "out of host memory\n");
        exit(EXIT_FAILURE);
    }
    if (hops_only) {
        return;
    }
    for (int direction = 0; direction < DIRECTIONS; direction++) {
        CHECK_CUDA(cudaStreamCreateWithFlags(&card->link_streams[direction], cudaStreamNonBlocking));
        CHECK_CUDA(cudaHostAlloc((void **)&card->link_host[direction], LINK_BUFFER_BYTES, cudaHostAllocPortable));
        memset(card->link_host[direction], 1, LINK_BUFFER_BYTES);
        CHECK_CUDA(cudaMalloc((void **)&card->link_card[direction], LINK_BUFFER_BYTES));
        CHECK_CUDA(cudaMemset(card->link_card[direction], 1, LINK_BUFFER_BYTES));
        CHECK_CUDA(cudaEventCreate(&card->link_started[direction]));
        CHECK_CUDA(cudaEventCreate(&card->link_stopped[direction]));
    }
}

static void close_card(struct card *card, int hops_only)
{
    CHECK_CUDA(cudaSetDevice(card->index));
    CHECK_CUDA(cudaDeviceSynchronize());
    if (!hops_only) {
        for (int direction = 0; direction < DIRECTIONS; direction++) {
            CHECK_CUDA(cudaEventDestroy(card->link_started[direction]));
            CHECK_CUDA(cudaEventDestroy(card->link_stopped[direction]));
            CHECK_CUDA(cudaFree(card->link_card[direction]));
            CHECK_CUDA(cudaFreeHost(card->link_host[direction]));
            CHECK_CUDA(cudaStreamDestroy(card->link_streams[direction]));
        }
    }
    free(card->host_block_counts);
    CHECK_CUDA(cudaFree(card->block_counts));
    CHECK_CUDA(cudaFree(card->payload));
    CHECK_CUDA(cudaStreamDestroy(card->stream));
}

/* One peer attribute, or -1 where the driver refuses the question (the probe goes on). */
static int peer_attribute(enum cudaDeviceP2PAttr attribute, int accessor, int owner)
{
    int value = 0;
    if (cudaDeviceGetP2PAttribute(&value, attribute, accessor, owner) != cudaSuccess) {
        (void)cudaGetLastError();
        return -1;
    }
    return value;
}

/* The driver's answers for every ordered pair; returns how many pairs it offers peer access on. */
static int print_peer_answers(const struct card *cards, int card_count)
{
    int offered_pairs = 0;
    printf("\npeer answers (cudaDeviceCanAccessPeer; attributes access, native atomics, performance rank); "
           "nothing enabled\n");
    for (int accessor = 0; accessor < card_count; accessor++) {
        for (int owner = 0; owner < card_count; owner++) {
            if (accessor == owner) {
                continue;
            }
            int can_access = 0;
            CHECK_CUDA(cudaDeviceCanAccessPeer(&can_access, accessor, owner));
            int access_supported = peer_attribute(cudaDevP2PAttrAccessSupported, accessor, owner);
            int native_atomics = peer_attribute(cudaDevP2PAttrNativeAtomicSupported, accessor, owner);
            int performance_rank = peer_attribute(cudaDevP2PAttrPerformanceRank, accessor, owner);
            printf("card %d (%s) reads card %d (%s): can access peer %s; access %d, native atomics %d, rank %d\n",
                   accessor, bus_text(&cards[accessor]), owner, bus_text(&cards[owner]),
                   can_access ? "offered, unverified until the gate passes it" : "no",
                   access_supported, native_atomics, performance_rank);
            offered_pairs += can_access != 0;
        }
    }
    printf("peer access offered on %d of %d ordered pairs\n", offered_pairs, card_count * (card_count - 1));
    return offered_pairs;
}

int main(int argument_count, char **arguments)
{
    unsigned int hop_mib[MAX_SIZES], piece_mib[MAX_SIZES];
    int hop_count = 0, piece_count = 0;
    unsigned int repeats = DEFAULT_REPEATS, rounds = PEER_DEFAULT_ROUNDS;
    int corrupt_one_word = 0, write_combined = 0, hops_only = 0, peer = 0, rounds_given = 0;
    const char *gate_record = NULL;
    for (int argument_index = 1; argument_index < argument_count; argument_index++) {
        const char *argument = arguments[argument_index];
        const char *value = argument_index + 1 < argument_count ? arguments[argument_index + 1] : "";
        if (strcmp(argument, "--corrupt") == 0) {
            corrupt_one_word = 1;
        } else if (strcmp(argument, "--write-combined") == 0) {
            write_combined = 1;
        } else if (strcmp(argument, "--hops-only") == 0) {
            hops_only = 1;
        } else if (strcmp(argument, "--peer") == 0) {
            peer = 1;
        } else if (strcmp(argument, "--stand-in-host") == 0) {
            peer_stand_in = 1;
        } else if (strcmp(argument, "--gate-record") == 0 && *value != '\0' && strlen(value) < P2P_PATH_BYTES) {
            gate_record = value;
            argument_index++;
        } else if (strcmp(argument, "--rounds") == 0) {
            unsigned int round_list[MAX_SIZES];
            int round_count = 0;
            if (!read_size_list(value, round_list, &round_count, 1, PEER_MOST_ROUNDS)) {
                fprintf(stderr, "--rounds needs a number from 1 to %u\n", PEER_MOST_ROUNDS);
                return EXIT_FAILURE;
            }
            rounds = round_list[0];
            rounds_given = 1;
            argument_index++;
        } else if (strcmp(argument, "--hop-mib") == 0 && read_size_list(value, hop_mib, &hop_count, 1, MAX_HOP_MIB)) {
            argument_index++;
        } else if (strcmp(argument, "--piece-mib") == 0 &&
                   read_size_list(value, piece_mib, &piece_count, 1, MAX_HOP_MIB)) {
            argument_index++;
        } else if (strcmp(argument, "--repeats") == 0) {
            unsigned int repeat_list[MAX_SIZES];
            int repeat_count = 0;
            if (!read_size_list(value, repeat_list, &repeat_count, 1, MAX_REPEATS)) {
                fprintf(stderr, "--repeats needs a number from 1 to %d\n", MAX_REPEATS);
                return EXIT_FAILURE;
            }
            repeats = repeat_list[0];
            argument_index++;
        } else {
            fprintf(stderr, "usage: %s [--hop-mib 1..%d]... (at most %d) [--piece-mib 1..%d]... [--repeats 1..%d] "
                    "[--write-combined] [--hops-only] [--corrupt]\n       %s --peer (--gate-record PATH | "
                    "--stand-in-host) [--hop-mib N]... [--repeats N] [--rounds 1..%u] [--corrupt]\n", arguments[0],
                    MAX_HOP_MIB, MAX_SIZES, MAX_HOP_MIB, MAX_REPEATS, arguments[0], PEER_MOST_ROUNDS);
            return EXIT_FAILURE;
        }
    }
    if (peer ? piece_count != 0 || write_combined || hops_only || (gate_record != NULL) == peer_stand_in
             : gate_record != NULL || peer_stand_in || rounds_given) {
        fprintf(stderr, "--peer takes exactly one of --gate-record and --stand-in-host, and neither --piece-mib, "
                        "--write-combined nor --hops-only; --gate-record, --stand-in-host and --rounds need --peer\n");
        return EXIT_FAILURE;
    }
    if (hop_count == 0) {
        for (size_t at = 0; at < sizeof(default_hop_mib) / sizeof(default_hop_mib[0]); at++) {
            hop_mib[hop_count++] = default_hop_mib[at];
        }
    }
    if (piece_count == 0) {
        for (size_t at = 0; at < sizeof(default_piece_mib) / sizeof(default_piece_mib[0]); at++) {
            piece_mib[piece_count++] = default_piece_mib[at];
        }
    }
    unsigned int largest_hop_mib = 0, smallest_piece_mib = MAX_HOP_MIB;
    for (int at = 0; at < hop_count; at++) {
        largest_hop_mib = hop_mib[at] > largest_hop_mib ? hop_mib[at] : largest_hop_mib;
    }
    for (int at = 0; at < piece_count; at++) {
        smallest_piece_mib = piece_mib[at] < smallest_piece_mib ? piece_mib[at] : smallest_piece_mib;
    }
    size_t largest_hop_bytes = (size_t)largest_hop_mib * BYTES_PER_MIB;
    unsigned int most_pieces = (largest_hop_mib + smallest_piece_mib - 1) / smallest_piece_mib;

    int card_count = 0, runtime_version = 0, driver_version = 0;
    CHECK_CUDA(cudaRuntimeGetVersion(&runtime_version));
    CHECK_CUDA(cudaDriverGetVersion(&driver_version));
    CHECK_CUDA(cudaGetDeviceCount(&card_count));
    printf("runtime %d, driver interface %d, %d cards\n", runtime_version, driver_version, card_count);
    if (card_count < 2 || card_count > MAX_CARDS) {
        fprintf(stderr, "needs 2 to %d cards, sees %d\n", MAX_CARDS, card_count);
        return EXIT_FAILURE;
    }
    static struct card cards[MAX_CARDS];
    for (int card_index = 0; card_index < card_count; card_index++) {
        open_card(&cards[card_index], card_index, largest_hop_bytes, hops_only || peer);
    }
    print_peer_answers(cards, card_count);
    if (peer) {
        int status = peer_main(cards, card_count, gate_record, hop_mib, hop_count, repeats, rounds, corrupt_one_word);
        for (int card_index = 0; card_index < card_count; card_index++) {
            close_card(&cards[card_index], 1);
        }
        return status;
    }

    /* The shared pages: page-locked, mapped into every card, portable to every context. */
    char *staging = NULL;
    uint32_t *gate = NULL, *gave_up = NULL;
    unsigned int staging_flags = cudaHostAllocMapped | cudaHostAllocPortable;
    if (write_combined) {
        staging_flags |= cudaHostAllocWriteCombined;
    }
    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaHostAlloc((void **)&staging, largest_hop_bytes, staging_flags));
    CHECK_CUDA(cudaHostAlloc((void **)&gate, sizeof(uint32_t), cudaHostAllocMapped | cudaHostAllocPortable));
    CHECK_CUDA(cudaHostAlloc((void **)&gave_up, MAX_CARDS * sizeof(uint32_t),
                             cudaHostAllocMapped | cudaHostAllocPortable));
    *(volatile uint32_t *)gate = 0u;
    for (int card_index = 0; card_index < card_count; card_index++) {
        CHECK_CUDA(cudaSetDevice(card_index));
        CHECK_CUDA(cudaHostGetDevicePointer((void **)&cards[card_index].staging_view, staging, 0));
        CHECK_CUDA(cudaHostGetDevicePointer((void **)&cards[card_index].gate_view, gate, 0));
        CHECK_CUDA(cudaHostGetDevicePointer((void **)&cards[card_index].gave_up_view, gave_up, 0));
    }

    printf("\nhops through the shared pages (%s), %u runs each%s\n",
           write_combined ? "write-combined" : "cached", repeats,
           corrupt_one_word ? ", one word of the sender's payload damaged" : "");
    int failed_cases = 0;
    uint32_t run_counter = 0;
    cudaEvent_t *piece_events = (cudaEvent_t *)calloc(most_pieces, sizeof(cudaEvent_t));
    if (piece_events == NULL) {
        fprintf(stderr, "out of host memory\n");
        return EXIT_FAILURE;
    }
    for (int sender = 0; sender < card_count; sender++) {
        for (int receiver = 0; receiver < card_count; receiver++) {
            if (sender == receiver) {
                continue;
            }
            printf("card %d (%s, %s) -> card %d (%s, %s)\n", sender, bus_text(&cards[sender]), cards[sender].name,
                   receiver, bus_text(&cards[receiver]), cards[receiver].name);
            CHECK_CUDA(cudaSetDevice(sender));
            for (unsigned int piece = 0; piece < most_pieces; piece++) {
                CHECK_CUDA(cudaEventCreateWithFlags(&piece_events[piece], cudaEventDisableTiming));
            }
            for (int hop_index = 0; hop_index < hop_count; hop_index++) {
                failed_cases += measure_hop_size(&cards[sender], &cards[receiver], staging, piece_events,
                                                 hop_mib[hop_index], piece_mib, piece_count, repeats, corrupt_one_word,
                                                 &run_counter);
            }
            CHECK_CUDA(cudaSetDevice(sender));
            for (unsigned int piece = 0; piece < most_pieces; piece++) {
                CHECK_CUDA(cudaEventDestroy(piece_events[piece]));
            }
        }
    }
    free(piece_events);

    int failed_link_runs = hops_only ? 0 : measure_links(cards, card_count, gate, gave_up);

    for (int card_index = 0; card_index < card_count; card_index++) {
        close_card(&cards[card_index], hops_only);
    }
    CHECK_CUDA(cudaFreeHost(gave_up));
    CHECK_CUDA(cudaFreeHost(gate));
    CHECK_CUDA(cudaFreeHost(staging));
    if (failed_cases != 0 || failed_link_runs != 0) {
        fprintf(stderr, "%d hop case(s) failed their check%s, %d link run(s) gave up at the gate\n", failed_cases,
                corrupt_one_word ? " (the damaged word not reported exactly once)" : "", failed_link_runs);
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
