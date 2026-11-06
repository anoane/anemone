/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * link_hop_peer.cuh: the peer mode of link_hop_probe.cu (its first comment says what it measures), kept beside it so
 * that each file stays under STYLE.md's 1,000 lines. It is included by link_hop_probe.cu at the place it holds there,
 * and only there: it uses that file's cards, kernels and helpers. Measurement tool: not part of the engine.
 */
#ifndef LINK_HOP_PEER_CUH
#define LINK_HOP_PEER_CUH

/* ---- peer mode: only the pairs the gate passed, through peer mappings ---- */

#define PEER_DEFAULT_ROUNDS 2000u
#define PEER_MOST_ROUNDS 100000u
#define PEER_SUM_WIDTH 5120u                     /* fp32 values of one row's partial sum: 20 KiB */
#define PEER_MOST_SUM_ROWS 8u
#define PEER_MAILBOX_BYTES (P2P_FLAG_AREA_WORDS * sizeof(uint32_t) + 400u * 1024u)
#define PEER_ROUND_TRIP_BYTES 16u
#define PEER_SUM_DAMAGE_PAST_MIDDLE 1u
#define PEER_DIRECTIONS 2                        /* of a hop both ways */
#define PEER_RANKS 2u                            /* of a sum */
#define PEER_RANK_STEP 0x85ebca6bu
#define PEER_PARTIAL_SHIFT 8u                    /* a partial sum's value: 24 bits of a word, scaled into [-1, 1) */
#define PEER_PARTIAL_SCALE (1.0f / 8388608.0f)   /* 2^-23 */
#define PEER_SUM_DAMAGE 0.5f                     /* added to the damaged value: see peer_partial_vector */

/* 16 B, a flag with its payload; 10 and 20 KiB, a rounded and an fp32 partial sum of a row; 44 KB, one row's
 * hand-off; 84 KiB, one row's hop with fp32 streams; 160 KiB, eight rows of fp32 partial sums; 354 KB, eight rows
 * of hand-off; 400 KiB (design/peer_to_peer.md 5). */
static const uint32_t peer_push_bytes[] = {PEER_ROUND_TRIP_BYTES, 10240u, 20480u, 44000u, 86016u, 163840u, 354000u,
                                           409600u};
static const uint32_t peer_block_counts[] = {1u, 8u};
static const uint32_t peer_sum_rows[] = {1u, PEER_MOST_SUM_ROWS};
enum peer_way { PEER_COPY, PEER_STORES, PEER_LOADS, PEER_WAY_COUNT };
static const char *const peer_way_names[PEER_WAY_COUNT] = {"peer_copy", "peer_stores", "peer_loads"};
#define PEER_COUNT(list) (sizeof(list) / sizeof((list)[0]))

struct peer_memory {                             /* reachable: the card's own memory, or a stand-in host page */
    uint32_t *inbox;                             /* reachable: a hop's destination for copies and stores */
    uint32_t *outbox;                            /* reachable: a hop's source for the receiver's loads */
    uint32_t *mailbox;                           /* reachable: the push flags, then the payload */
    uint32_t *acknowledgements;                  /* reachable: written by the receiver of this card's pushes */
    uint32_t *sum_flags;                         /* reachable */
    float *sum_inbox;                            /* reachable: two slots of the largest sum */
    float *sum_result;                           /* own memory */
    struct p2p_block_status *statuses;           /* own memory: a sender's, then a receiver's */
    float *host_sum_result;
};

static struct peer_memory peer_memories[MAX_CARDS];
static unsigned char peer_allowed[MAX_CARDS][MAX_CARDS];     /* [accessor][owner]: the gate passed it */
static int peer_stand_in;
static uint32_t *peer_host_acknowledgements;     /* where a push is acknowledged when the gate failed the reverse */
static void *peer_allocations[MAX_CARDS * 8];
static int peer_allocation_count;

/* Memory a peer reaches: the card's own, or with --stand-in-host a mapped host page whose one address every card
 * uses. Zeroed. */
static void *peer_reachable(int card_index, int card_count, size_t bytes)
{
    void *memory = NULL;
    CHECK_CUDA(cudaSetDevice(card_index));
    if (peer_stand_in) {
        CHECK_CUDA(cudaHostAlloc(&memory, bytes, cudaHostAllocMapped | cudaHostAllocPortable));
        memset(memory, 0, bytes);
        for (int other = 0; other < card_count; other++) {
            void *view = NULL;
            CHECK_CUDA(cudaSetDevice(other));
            CHECK_CUDA(cudaHostGetDevicePointer(&view, memory, 0));
            if (view != memory) {
                fprintf(stderr, "card %d maps a host page at another address\n", other);
                exit(EXIT_FAILURE);
            }
        }
    } else {
        CHECK_CUDA(cudaMalloc(&memory, bytes));
        CHECK_CUDA(cudaMemset(memory, 0, bytes));
    }
    peer_allocations[peer_allocation_count++] = memory;
    return memory;
}

static void open_peer_memory(int card_index, int card_count, size_t hop_bytes)
{
    struct peer_memory *memory = &peer_memories[card_index];
    size_t flag_bytes = P2P_FLAG_AREA_WORDS * sizeof(uint32_t);
    size_t sum_bytes = (size_t)PEER_MOST_SUM_ROWS * PEER_SUM_WIDTH * sizeof(float);
    memory->inbox = (uint32_t *)peer_reachable(card_index, card_count, hop_bytes);
    memory->outbox = (uint32_t *)peer_reachable(card_index, card_count, hop_bytes);
    memory->mailbox = (uint32_t *)peer_reachable(card_index, card_count, PEER_MAILBOX_BYTES);
    memory->acknowledgements = (uint32_t *)peer_reachable(card_index, card_count, flag_bytes);
    memory->sum_flags = (uint32_t *)peer_reachable(card_index, card_count, flag_bytes);
    memory->sum_inbox = (float *)peer_reachable(card_index, card_count, P2P_SUM_SLOTS * sum_bytes);
    CHECK_CUDA(cudaSetDevice(card_index));
    CHECK_CUDA(cudaMalloc(&memory->sum_result, sum_bytes));
    CHECK_CUDA(cudaMalloc(&memory->statuses, PEER_RANKS * P2P_MAX_BLOCKS * sizeof(struct p2p_block_status)));
    memory->host_sum_result = (float *)malloc(sum_bytes);
    if (memory->host_sum_result == NULL) {
        fprintf(stderr, "out of host memory\n");
        exit(EXIT_FAILURE);
    }
}

static void close_peer_memory(int card_count)
{
    for (int card_index = 0; card_index < card_count; card_index++) {
        if (peer_memories[card_index].sum_result != NULL) {
            CHECK_CUDA(cudaSetDevice(card_index));
            CHECK_CUDA(cudaFree(peer_memories[card_index].sum_result));
            CHECK_CUDA(cudaFree(peer_memories[card_index].statuses));
            free(peer_memories[card_index].host_sum_result);
        }
    }
    for (int at = 0; at < peer_allocation_count; at++) {
        CHECK_CUDA(peer_stand_in ? cudaFreeHost(peer_allocations[at]) : cudaFree(peer_allocations[at]));
    }
}

static const char *peer_through(void)
{
    return peer_stand_in ? "host_stand_in" : "peer";
}

static void peer_deadline_passed(const char *what)
{
    fprintf(stderr, "%s did not end within the host's deadline: the cards are in an unknown state\n", what);
    exit(EXIT_FAILURE);
}

/* A check failed through real peer mappings: the run sends no further peer traffic (a stand-in run goes on). */
static int peer_stopped(int failed)
{
    return failed != 0 && !peer_stand_in;
}

static int peer_must_stop(int failed)
{
    if (peer_stopped(failed)) {
        printf("PEER STOPPED: a check failed through a pair the gate passed; no further peer traffic is sent\n");
    }
    return peer_stopped(failed);
}

/* Every push size and block count from sender into the receiver's memory; returns the runs whose check failed. A
 * damaged run must report exactly the damaged word, at its place. */
static int peer_pushes(const struct card *cards, int sender, int receiver, uint32_t rounds, int corrupt)
{
    int failed = 0, acknowledged_by_peer = peer_allowed[receiver][sender];
    struct p2p_push_buffers buffers;
    buffers.flags = peer_memories[receiver].mailbox;
    buffers.payload = peer_memories[receiver].mailbox + P2P_FLAG_AREA_WORDS;
    buffers.acknowledgements =
        acknowledged_by_peer ? peer_memories[sender].acknowledgements : peer_host_acknowledgements;
    buffers.sender_status = peer_memories[sender].statuses;
    buffers.receiver_status = peer_memories[receiver].statuses + P2P_MAX_BLOCKS;
    for (size_t size = 0; size < PEER_COUNT(peer_push_bytes); size++) {
        for (size_t block_case = 0; block_case < PEER_COUNT(peer_block_counts); block_case++) {
            uint32_t words = peer_push_bytes[size] / (uint32_t)sizeof(uint32_t), blocks = peer_block_counts[block_case];
            uint32_t seed = (uint32_t)p2p_random_seed();
            struct p2p_damage damage = {corrupt, (rounds + 1u) / 2u, words / 2u};
            struct p2p_outcome outcome;
            if (!p2p_run_push(sender, cards[sender].stream, receiver, cards[receiver].stream, &buffers, words, blocks,
                              rounds, seed, &damage, &outcome)) {
                peer_deadline_passed("a push");
            }
            int held = outcome.timed_out == 0 && outcome.rounds_completed == rounds &&
                       outcome.wrong_words == (corrupt ? 1u : 0u) &&
                       outcome.first_wrong_word == (corrupt ? damage.index : P2P_NO_WORD);
            failed += !held;
            printf("PUSH pair=%s>%s through=%s ack=%s bytes=%u blocks=%u rounds=%u seed=%08x us_per_round=%.2f "
                   "completed=%u timed_out_blocks=%u wrong_words=%u verdict=%s\n", bus_text(&cards[sender]),
                   bus_text(&cards[receiver]), peer_through(), acknowledged_by_peer ? peer_through() : "host_page",
                   peer_push_bytes[size], blocks, rounds, seed, outcome.microseconds_per_round,
                   outcome.rounds_completed, outcome.timed_out, outcome.wrong_words,
                   !held ? "FAILED" : corrupt ? "reported" : "ok");
            if (peer_push_bytes[size] == PEER_ROUND_TRIP_BYTES && blocks == 1 && acknowledged_by_peer) {
                printf("ROUND_TRIP pair=%s<>%s through=%s us=%.2f verdict=%s: 16 B and a flag there, a flag back, "
                       "one fence each\n", bus_text(&cards[sender]), bus_text(&cards[receiver]), peer_through(),
                       outcome.microseconds_per_round, !held ? "FAILED" : corrupt ? "reported" : "ok");
            }
            if (peer_must_stop(failed)) {
                return failed;
            }
        }
    }
    return failed;
}

/* ---- the two-rank sum ---- */

/* A rank's partial sum at one step and element: a multiple of 2^-23 in [-1, 1), so any two add exactly. */
__host__ __device__ __forceinline__ float peer_partial_value(uint32_t seed, uint32_t rank, uint32_t step,
                                                             uint32_t element)
{
    uint32_t bits = p2p_round_word(seed ^ (rank * PEER_RANK_STEP), step, element);
    return (float)(bits >> PEER_PARTIAL_SHIFT) * PEER_PARTIAL_SCALE - 1.0f;
}

/* Four partial values from first_element; damaged_element, if among them, moved by exactly PEER_SUM_DAMAGE, which
 * changes its bits and, both partials being multiples of 2^-23 under 1 in magnitude, every sum it enters by about
 * as much (a change of the lowest bit can vanish in the sum's rounding). */
__device__ __forceinline__ float4 peer_partial_vector(uint32_t seed, uint32_t rank, uint32_t step,
                                                      uint32_t first_element, uint32_t damaged_element)
{
    float values[P2P_VECTOR_WORDS];
    float4 vector;
#pragma unroll
    for (uint32_t at = 0; at < P2P_VECTOR_WORDS; at++) {
        values[at] = peer_partial_value(seed, rank, step, first_element + at);
        values[at] = first_element + at == damaged_element ? __fadd_rn(values[at], PEER_SUM_DAMAGE) : values[at];
    }
    memcpy(&vector, values, sizeof(vector));
    return vector;
}

/* The values of a 16-byte unit whose bits differ from wanted's; *least is lowered to the index of the first. */
__device__ __forceinline__ uint32_t peer_float4_wrong(float4 got, float4 wanted, uint32_t first_element,
                                                      uint32_t *least)
{
    uint4 got_bits, wanted_bits;
    memcpy(&got_bits, &got, sizeof(got_bits));
    memcpy(&wanted_bits, &wanted, sizeof(wanted_bits));
    return p2p_vector_wrong(got_bits, wanted_bits, first_element, least);
}

/* One rank of a two-rank sum, in 16-byte units. Per step, each block stores its slice of the rank's partial into the
 * peer's inbox, slot step % 2; after the barrier its thread 0 fences, stores the step into the block's flag there and
 * waits for the peer's flag in its own inbox with acquire loads (one fence a block and step, as the push of
 * p2p_common.cuh); then the block loads the peer's slice past the first-level cache, P2P_LOADS_IN_FLIGHT units a
 * thread before it uses them, counts the values that are not the peer's partial, and stores its own partial plus the
 * peer's into result.
 * Two slots suffice: a rank writes a slot again only after it received the peer's next step, which the peer sends
 * after it finished reading that slot. The damaged instance moves one value of one step (peer_partial_vector). */
template <bool damaged>
__global__ void peer_sum_kernel(float *peer_inbox, uint32_t *peer_flags, const float *own_inbox,
                                const uint32_t *own_flags, float *result, uint32_t elements, uint32_t steps,
                                uint32_t seed, uint32_t rank, uint64_t timeout_ns, uint32_t damaged_step,
                                uint32_t damaged_element, struct p2p_block_status *status)
{
    __shared__ uint32_t arrived;
    uint32_t first = 0, end = 0, completed = 0, wrong = 0, least = P2P_NO_WORD, peer_rank = 1u - rank;
    p2p_block_slice(elements / P2P_VECTOR_WORDS, &first, &end);
    for (uint32_t step = 1; step <= steps; step++) {
        uint32_t slot = step % P2P_SUM_SLOTS;
        uint32_t flag_index = (slot * gridDim.x + blockIdx.x) * P2P_FLAG_STRIDE_WORDS;
        uint32_t damaged_here = damaged && step == damaged_step ? damaged_element : P2P_NO_WORD;
        float4 *outgoing = (float4 *)(peer_inbox + (uint64_t)slot * elements);
        for (uint32_t unit = first + threadIdx.x; unit < end; unit += blockDim.x) {
            outgoing[unit] = peer_partial_vector(seed, rank, step, unit * P2P_VECTOR_WORDS, damaged_here);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            __threadfence_system();
            ((volatile uint32_t *)peer_flags)[flag_index] = step;
            arrived = p2p_wait_flag((const volatile uint32_t *)own_flags + flag_index, step, timeout_ns);
        }
        __syncthreads();
        if (arrived == 0u) {
            break;
        }
        const float4 *incoming = (const float4 *)(own_inbox + (uint64_t)slot * elements);
        for (uint32_t base = first + threadIdx.x; base < end; base += blockDim.x * P2P_LOADS_IN_FLIGHT) {
            float4 loaded[P2P_LOADS_IN_FLIGHT];
#pragma unroll
            for (uint32_t at = 0; at < P2P_LOADS_IN_FLIGHT; at++) {
                uint32_t unit = base + at * blockDim.x;
                loaded[at] = unit < end ? __ldcv(incoming + unit) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
#pragma unroll
            for (uint32_t at = 0; at < P2P_LOADS_IN_FLIGHT; at++) {
                uint32_t unit = base + at * blockDim.x;
                if (unit < end) {
                    float4 own = peer_partial_vector(seed, rank, step, unit * P2P_VECTOR_WORDS, P2P_NO_WORD);
                    wrong += peer_float4_wrong(loaded[at], peer_partial_vector(seed, peer_rank, step,
                                                                               unit * P2P_VECTOR_WORDS, P2P_NO_WORD),
                                               unit * P2P_VECTOR_WORDS, &least);
                    ((float4 *)result)[unit] =
                        make_float4(__fadd_rn(own.x, loaded[at].x), __fadd_rn(own.y, loaded[at].y),
                                    __fadd_rn(own.z, loaded[at].z), __fadd_rn(own.w, loaded[at].w));
                }
            }
        }
        __syncthreads();                         /* `arrived` is written again only after every thread read it */
        completed = step;
    }
    p2p_block_reduce(&wrong, &least);
    if (threadIdx.x == 0) {
        status[blockIdx.x].rounds_completed = completed;
        status[blockIdx.x].timed_out = completed != steps;
        status[blockIdx.x].wrong_words = wrong;
        status[blockIdx.x].first_wrong_word = least;
    }
}

struct peer_sum_rank {                           /* one rank's side of a two-rank sum */
    int card;
    cudaStream_t stream;
    float *inbox;                                /* its own memory (or a host page), two slots of `elements` */
    uint32_t *flags;                             /* its own memory (or a host page): P2P_FLAG_AREA_WORDS */
    float *result;                               /* its own memory: elements */
    struct p2p_block_status *status;             /* its own memory: P2P_MAX_BLOCKS entries */
    float *host_result;                          /* host memory: elements, for the check */
};

/* One two-rank sum of `elements` values a rank (a multiple of four) over `steps` steps. Returns 0 if the host's
 * deadline passed. The last step's results of both ranks are compared with the host's sums, bit for bit. */
static int peer_run_sum(struct peer_sum_rank *ranks, uint32_t elements, uint32_t blocks, uint32_t steps,
                        uint32_t seed, const struct p2p_damage *damage, struct p2p_outcome *outcome)
{
    if (blocks == 0 || blocks > P2P_MAX_BLOCKS || steps == 0 || elements == 0 || elements % P2P_VECTOR_WORDS != 0 ||
        (damage->active && (damage->round == 0 || damage->round > steps || damage->index >= elements))) {
        fprintf(stderr, "a sum of %u blocks, %u steps, damage at step %u element %u of %u is refused\n", blocks, steps,
                damage->round, damage->index, elements);
        exit(P2P_EXIT_CRASH);
    }
    memset(outcome, 0, sizeof(*outcome));
    outcome->rounds_completed = steps;
    outcome->first_wrong_word = P2P_NO_WORD;
    for (uint32_t rank = 0; rank < PEER_RANKS; rank++) {
        p2p_zero_words(ranks[rank].card, ranks[rank].stream, ranks[rank].flags, P2P_FLAG_AREA_WORDS);
        p2p_zero_words(ranks[rank].card, ranks[rank].stream, ranks[rank].status, P2P_STATUS_WORDS(blocks));
    }
    for (uint32_t rank = 0; rank < PEER_RANKS; rank++) {
        if (!p2p_wait_stream(ranks[rank].card, ranks[rank].stream, P2P_HOST_DEADLINE_SECONDS)) {
            return 0;
        }
    }
    double started = p2p_seconds_now();
    for (uint32_t rank = 0; rank < PEER_RANKS; rank++) {
        struct peer_sum_rank *own = &ranks[rank], *peer = &ranks[1u - rank];
        CHECK_CUDA(cudaSetDevice(own->card));
        if (damage->active && rank == 0) {       /* rank 0 sends the damaged value; rank 1 must find it */
            peer_sum_kernel<true><<<blocks, P2P_THREADS, 0, own->stream>>>(
                peer->inbox, peer->flags, own->inbox, own->flags, own->result, elements, steps, seed, rank,
                P2P_WAIT_TIMEOUT_NS, damage->round, damage->index, own->status);
        } else {
            peer_sum_kernel<false><<<blocks, P2P_THREADS, 0, own->stream>>>(
                peer->inbox, peer->flags, own->inbox, own->flags, own->result, elements, steps, seed, rank,
                P2P_WAIT_TIMEOUT_NS, 0u, 0u, own->status);
        }
        CHECK_CUDA(cudaGetLastError());
    }
    for (uint32_t rank = 0; rank < PEER_RANKS; rank++) {
        if (!p2p_wait_stream(ranks[rank].card, ranks[rank].stream, P2P_HOST_DEADLINE_SECONDS)) {
            return 0;
        }
    }
    outcome->microseconds_per_round = (p2p_seconds_now() - started) * P2P_MICROSECONDS_PER_SECOND / (double)steps;
    for (uint32_t rank = 0; rank < PEER_RANKS; rank++) {
        p2p_add_statuses(ranks[rank].card, ranks[rank].status, blocks, steps, outcome);
        CHECK_CUDA(cudaMemcpy(ranks[rank].host_result, ranks[rank].result, elements * sizeof(float),
                              cudaMemcpyDeviceToHost));
        for (uint32_t element = 0; element < elements; element++) {
            float sum = peer_partial_value(seed, 0u, steps, element) + peer_partial_value(seed, 1u, steps, element);
            uint32_t expected_bits = 0, result_bits = 0;
            memcpy(&expected_bits, &sum, sizeof(sum));
            memcpy(&result_bits, &ranks[rank].host_result[element], sizeof(result_bits));
            outcome->result_differing += expected_bits != result_bits;
        }
    }
    return 1;
}

/* The two-rank sum between two cards that the gate passed both ways; returns the runs whose check failed. */
static int peer_sums(const struct card *cards, int first, int second, uint32_t steps, int corrupt)
{
    int failed = 0;
    struct peer_sum_rank ranks[PEER_RANKS];
    const int rank_cards[PEER_RANKS] = {first, second};
    for (uint32_t rank = 0; rank < PEER_RANKS; rank++) {
        struct peer_memory *memory = &peer_memories[rank_cards[rank]];
        ranks[rank].card = rank_cards[rank];
        ranks[rank].stream = cards[rank_cards[rank]].stream;
        ranks[rank].inbox = memory->sum_inbox;
        ranks[rank].flags = memory->sum_flags;
        ranks[rank].result = memory->sum_result;
        ranks[rank].status = memory->statuses;
        ranks[rank].host_result = memory->host_sum_result;
    }
    for (size_t row_case = 0; row_case < PEER_COUNT(peer_sum_rows); row_case++) {
        for (size_t block_case = 0; block_case < PEER_COUNT(peer_block_counts); block_case++) {
            uint32_t elements = peer_sum_rows[row_case] * PEER_SUM_WIDTH, seed = (uint32_t)p2p_random_seed();
            struct p2p_damage damage = {corrupt, steps, elements / 2u + PEER_SUM_DAMAGE_PAST_MIDDLE};
            struct p2p_outcome outcome;
            if (!peer_run_sum(ranks, elements, peer_block_counts[block_case], steps, seed, &damage, &outcome)) {
                peer_deadline_passed("a two-rank sum");
            }
            uint32_t wanted = corrupt ? 1u : 0u;
            int held = outcome.timed_out == 0 && outcome.rounds_completed == steps && outcome.wrong_words == wanted &&
                       outcome.result_differing == wanted &&
                       outcome.first_wrong_word == (corrupt ? damage.index : P2P_NO_WORD);
            failed += !held;
            printf("SUM pair=%s<>%s through=%s rows=%u blocks=%u steps=%u seed=%08x us_per_step=%.2f completed=%u "
                   "timed_out_blocks=%u received_wrong=%u result_differing=%u verdict=%s\n", bus_text(&cards[first]),
                   bus_text(&cards[second]), peer_through(), peer_sum_rows[row_case], peer_block_counts[block_case],
                   steps, seed, outcome.microseconds_per_round, outcome.rounds_completed, outcome.timed_out,
                   outcome.wrong_words, outcome.result_differing, !held ? "FAILED" : corrupt ? "reported" : "ok");
            if (peer_must_stop(failed)) {
                return failed;
            }
        }
    }
    return failed;
}

/* One hop of `bytes` by one way from first to second, and with both_ways from second to first at the same time.
 * Returns its milliseconds; wrong[d] is direction d's count of wrong words on its receiver. */
static double run_peer_hop(enum peer_way way, struct card *first, struct card *second, int both_ways, size_t bytes,
                           uint32_t seed, int corrupt, uint64_t wrong[PEER_DIRECTIONS])
{
    struct card *senders[PEER_DIRECTIONS] = {first, second}, *receivers[PEER_DIRECTIONS] = {second, first};
    uint32_t *sources[PEER_DIRECTIONS], *destinations[PEER_DIRECTIONS];
    uint64_t word_count = bytes / sizeof(uint32_t);
    int directions = both_ways ? PEER_DIRECTIONS : 1;
    for (int direction = 0; direction < directions; direction++) {
        struct card *sender = senders[direction], *receiver = receivers[direction];
        uint32_t direction_seed = seed + (uint32_t)direction * SEED_STEP;
        sources[direction] = way == PEER_LOADS ? peer_memories[sender->index].outbox : sender->payload;
        destinations[direction] = way == PEER_LOADS ? receiver->payload : peer_memories[receiver->index].inbox;
        CHECK_CUDA(cudaSetDevice(sender->index));
        fill_kernel<<<sender->block_count, THREADS_PER_BLOCK, 0, sender->stream>>>(sources[direction], word_count,
                                                                                   direction_seed, 0u);
        CHECK_CUDA(cudaGetLastError());
        wait_stream(sender->index, sender->stream);
        if (corrupt) {
            uint64_t damaged_index = word_count / 2 + CORRUPTED_WORD_PAST_MIDDLE;
            uint32_t wrong_word = ~pattern_word(damaged_index, direction_seed);
            CHECK_CUDA(cudaMemcpy(sources[direction] + damaged_index, &wrong_word, sizeof(wrong_word),
                                  cudaMemcpyDefault));
        }
        CHECK_CUDA(cudaSetDevice(receiver->index));
        fill_kernel<<<receiver->block_count, THREADS_PER_BLOCK, 0, receiver->stream>>>(destinations[direction],
                                                                                       word_count, direction_seed, ~0u);
        CHECK_CUDA(cudaGetLastError());
        wait_stream(receiver->index, receiver->stream);
    }
    double started = seconds_now();
    for (int direction = 0; direction < directions; direction++) {
        struct card *sender = senders[direction], *receiver = receivers[direction];
        struct card *runner = way == PEER_LOADS ? receiver : sender;
        CHECK_CUDA(cudaSetDevice(runner->index));
        if (way == PEER_COPY && !peer_stand_in) {
            CHECK_CUDA(cudaMemcpyPeerAsync(destinations[direction], receiver->index, sources[direction], sender->index,
                                           bytes, runner->stream));
        } else if (way == PEER_COPY) {
            CHECK_CUDA(cudaMemcpyAsync(destinations[direction], sources[direction], bytes, cudaMemcpyDefault,
                                       runner->stream));
        } else {
            copy_kernel<<<runner->block_count, THREADS_PER_BLOCK, 0, runner->stream>>>(
                (uint4 *)destinations[direction], (const uint4 *)sources[direction], bytes / LOAD_BYTES);
            CHECK_CUDA(cudaGetLastError());
        }
    }
    wait_stream(first->index, first->stream);
    wait_stream(second->index, second->stream);
    double milliseconds = (seconds_now() - started) * MILLISECONDS_PER_SECOND;
    for (int direction = 0; direction < directions; direction++) {
        struct card *receiver = receivers[direction];
        CHECK_CUDA(cudaSetDevice(receiver->index));
        count_mismatches_kernel<<<receiver->block_count, THREADS_PER_BLOCK, 0, receiver->stream>>>(
            destinations[direction], word_count, seed + (uint32_t)direction * SEED_STEP, receiver->block_counts);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaMemcpyAsync(receiver->host_block_counts, receiver->block_counts,
                                   receiver->block_count * sizeof(uint32_t), cudaMemcpyDeviceToHost, receiver->stream));
        wait_stream(receiver->index, receiver->stream);
        wrong[direction] = 0;
        for (unsigned int block_index = 0; block_index < receiver->block_count; block_index++) {
            wrong[direction] += receiver->host_block_counts[block_index];
        }
    }
    return milliseconds;
}

/* Every hop size by one way, one way or both ways; returns the cases whose check failed. */
static int peer_hops(struct card *cards, int first, int second, int both_ways, enum peer_way way,
                     const unsigned int *hop_mib, int hop_count, unsigned int repeats, int corrupt,
                     uint32_t *run_counter)
{
    int failed = 0;
    uint64_t wanted = corrupt ? 1u : 0u;
    for (int hop_index = 0; hop_index < hop_count; hop_index++) {
        size_t bytes = (size_t)hop_mib[hop_index] * BYTES_PER_MIB;
        double milliseconds[MAX_REPEATS];
        uint64_t worst[PEER_DIRECTIONS] = {0, 0};
        unsigned int failed_runs = 0, runs = 0;
        while (runs < repeats && !(failed_runs != 0 && !peer_stand_in)) {          /* a failed run ends the hop */
            uint64_t wrong[PEER_DIRECTIONS] = {wanted, wanted};
            milliseconds[runs++] = run_peer_hop(way, &cards[first], &cards[second], both_ways, bytes,
                                                ++*run_counter * SEED_STEP, corrupt, wrong);
            failed_runs += wrong[0] != wanted || wrong[1] != wanted;
            for (int direction = 0; direction < PEER_DIRECTIONS; direction++) {
                worst[direction] = wrong[direction] != wanted && wrong[direction] > worst[direction] ? wrong[direction]
                                                                                                    : worst[direction];
            }
        }
        sort_ascending(milliseconds, runs);
        failed += failed_runs != 0;
        /* through host pages, the two directions of a hop both ways leave over two cards' links, not over one */
        printf("HOP pair=%s%s%s through=%s way=%s hop_mib=%u directions=%s runs=%u best_ms=%.2f gbps_each=%.2f "
               "median_ms=%.2f failed_runs=%u worst_wrong=%llu,%llu verdict=%s\n", bus_text(&cards[first]),
               both_ways ? "<>" : ">", bus_text(&cards[second]), peer_through(), peer_way_names[way],
               hop_mib[hop_index], !both_ways ? "one_way" : peer_stand_in ? "both_ways_two_links" : "both_ways",
               runs, milliseconds[0], gb_per_second((double)bytes, milliseconds[0]), milliseconds[runs / 2],
               failed_runs,
               (unsigned long long)worst[0], (unsigned long long)worst[1],
               failed_runs != 0 ? "FAILED" : corrupt ? "reported" : "ok");
        if (peer_must_stop(failed)) {
            return failed;
        }
    }
    return failed;
}

/* Which ordered pairs the gate passed: all with --stand-in-host; else those of a record trusted under this machine's
 * key, each also offered by the driver now. Returns the count, or -1 if nothing may run. */
static int peer_choose_pairs(const struct card *cards, int card_count, const char *record_path)
{
    int count = 0;
    struct p2p_key key;
    struct p2p_record record;
    memset(&record, 0, sizeof(record));
    if (!peer_stand_in) {
        p2p_machine_key(&key);
        enum p2p_record_state state = p2p_record_read(record_path, &key, &record);
        printf("PEER record=%s verdict=%s key=%016" PRIx64 " key_complete=%s\n", p2p_record_state_names[state],
               state == P2P_RECORD_TRUSTED ? record.verdict : "na", key.value, key.complete ? "yes" : "no");
        if (state != P2P_RECORD_TRUSTED) {
            printf("PEER refused: no gate record trusted for this boot, driver, module, options and cards\n");
            return -1;
        }
    }
    for (int accessor = 0; accessor < card_count; accessor++) {
        for (int owner = 0; owner < card_count; owner++) {
            char accessor_bus[P2P_BUS_TEXT_BYTES], owner_bus[P2P_BUS_TEXT_BYTES];
            int can_access = 0;
            if (accessor == owner) {
                continue;
            }
            p2p_bus_text(accessor, accessor_bus);
            p2p_bus_text(owner, owner_bus);
            CHECK_CUDA(cudaDeviceCanAccessPeer(&can_access, accessor, owner));
            peer_allowed[accessor][owner] = (unsigned char)(peer_stand_in || p2p_record_allows(&record, accessor_bus,
                                                                                                 owner_bus));
            if (peer_allowed[accessor][owner] && !peer_stand_in && !can_access) {
                printf("PEER refused: the gate passed %s>%s, but the driver does not offer it now\n", accessor_bus,
                       owner_bus);
                return -1;
            }
            count += peer_allowed[accessor][owner];
            printf("PEER pair=%s>%s gate=%s\n", bus_text(&cards[accessor]), bus_text(&cards[owner]),
                   peer_stand_in ? "stand_in" : peer_allowed[accessor][owner] ? "passed" : "not_passed");
        }
    }
    return count;
}

/* The peer mode; returns the program's exit status. */
static int peer_main(struct card *cards, int card_count, const char *record_path, const unsigned int *hop_mib,
                     int hop_count, unsigned int repeats, uint32_t rounds, int corrupt)
{
    int failed = 0, pair_count = peer_choose_pairs(cards, card_count, record_path);
    uint32_t run_counter = 0;
    if (pair_count <= 0) {
        printf("PEER %s\n", pair_count == 0 ? "no pair passed the gate: nothing is measured" : "nothing is measured");
        return pair_count == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    size_t largest_hop_bytes = 0;
    for (int at = 0; at < hop_count; at++) {
        size_t hop_bytes = (size_t)hop_mib[at] * BYTES_PER_MIB;
        largest_hop_bytes = hop_bytes > largest_hop_bytes ? hop_bytes : largest_hop_bytes;
    }
    CHECK_CUDA(cudaHostAlloc((void **)&peer_host_acknowledgements, P2P_FLAG_AREA_WORDS * sizeof(uint32_t),
                             cudaHostAllocMapped | cudaHostAllocPortable));
    for (int card_index = 0; card_index < card_count; card_index++) {
        int involved = 0;
        for (int other = 0; other < card_count; other++) {
            involved |= peer_allowed[card_index][other] | peer_allowed[other][card_index];
            if (!peer_stand_in && peer_allowed[card_index][other]) {
                CHECK_CUDA(cudaSetDevice(card_index));
                cudaError_t status = cudaDeviceEnablePeerAccess(other, 0);
                if (status != cudaSuccess && status != cudaErrorPeerAccessAlreadyEnabled) {
                    CHECK_CUDA(status);
                }
                (void)cudaGetLastError();
            }
        }
        if (involved) {
            open_peer_memory(card_index, card_count, largest_hop_bytes);
        }
    }
    for (int sender = 0; sender < card_count && !peer_stopped(failed); sender++) {
        for (int receiver = 0; receiver < card_count && !peer_stopped(failed); receiver++) {
            if (sender != receiver && peer_allowed[sender][receiver]) {
                failed += peer_pushes(cards, sender, receiver, rounds, corrupt);
            }
        }
    }
    for (int first = 0; first < card_count && !peer_stopped(failed); first++) {
        for (int second = first + 1; second < card_count && !peer_stopped(failed); second++) {
            if (peer_allowed[first][second] && peer_allowed[second][first]) {
                failed += peer_sums(cards, first, second, rounds, corrupt);
            }
        }
    }
    for (int way = 0; way < PEER_WAY_COUNT && !peer_stopped(failed); way++) {
        for (int first = 0; first < card_count && !peer_stopped(failed); first++) {
            for (int second = 0; second < card_count && !peer_stopped(failed); second++) {
                /* copies and stores: the sender reaches the receiver's memory; loads: the receiver the sender's */
                int forward = way == PEER_LOADS ? peer_allowed[second][first] : peer_allowed[first][second];
                int backward = way == PEER_LOADS ? peer_allowed[first][second] : peer_allowed[second][first];
                if (first != second && forward) {
                    failed += peer_hops(cards, first, second, 0, (enum peer_way)way, hop_mib, hop_count, repeats,
                                        corrupt, &run_counter);
                }
                if (first < second && forward && backward && !peer_stopped(failed)) {
                    failed += peer_hops(cards, first, second, 1, (enum peer_way)way, hop_mib, hop_count, repeats,
                                        corrupt, &run_counter);
                }
            }
        }
    }
    close_peer_memory(card_count);
    CHECK_CUDA(cudaFreeHost(peer_host_acknowledgements));
    printf("PEER %d measurements failed their check%s\n", failed,
           corrupt ? " (the damaged one not reported exactly)" : "");
    return failed == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

#endif
