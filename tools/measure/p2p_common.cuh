/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * p2p_common.cuh: what the peer-to-peer gate (p2p_gate_probe.cu) and the peer mode of link_hop_probe.cu share
 * (design/peer_to_peer.md 6): waits with deadlines on the host and on the cards, the byte and word patterns, the
 * kernels that move bytes through a peer mapping and check them (fill, copy, count, the mailbox push with its
 * acknowledgement) with the push's host launcher; and through p2p_record.h, the gate's record, its key and the AER
 * counters.
 *
 * Measurement tool: not part of the engine.
 */
#ifndef P2P_COMMON_CUH
#define P2P_COMMON_CUH
#include <cuda_runtime.h>
#include <sys/random.h>
#include <time.h>

#include "p2p_record.h"

#if !defined(__BYTE_ORDER__) || __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "the pattern of a 64-bit word is its eight bytes in little-endian order, as the cards store them"
#endif

#define P2P_MIX_INCREMENT 0x9e3779b97f4a7c15ull  /* the splitmix64 finalizer's constants */
#define P2P_MIX_FIRST_MULTIPLIER 0xbf58476d1ce4e5b9ull
#define P2P_MIX_SECOND_MULTIPLIER 0x94d049bb133111ebull
#define P2P_MIX_FIRST_SHIFT 30
#define P2P_MIX_SECOND_SHIFT 27
#define P2P_MIX_THIRD_SHIFT 31
#define P2P_BITS_PER_BYTE 8u
#define P2P_GROUP_SHIFT 3u                       /* one mixed 64-bit word gives the pattern's next eight bytes */
#define P2P_GROUP_BYTES (1u << P2P_GROUP_SHIFT)
#define P2P_WORD_MULTIPLIER 2654435761u          /* odd: word -> word * multiplier is one to one */
#define P2P_ROUND_STEP 0x9e3779b9u               /* nonzero: every word of a round differs from the round before */

#define P2P_THREADS 256                          /* every kernel of this file runs blocks of this many threads */
#define P2P_FLAG_STRIDE_WORDS 32u                /* a flag on 128 bytes of its own */
#define P2P_MAX_BLOCKS 32u                       /* of a push or a sum: each block has its flags */
#define P2P_VECTOR_WORDS 4u                      /* a 16-byte access */
#define P2P_LOADS_IN_FLIGHT 4u                   /* 16-byte loads a receiver's thread issues before it checks them */
#define P2P_NO_WORD UINT32_MAX                   /* no damaged word; no wrong word */
#define P2P_COUNT_WORDS_PER_BLOCK 2u             /* of p2p_count_kernel: a block's count of wrong words, its first */
#define P2P_STATUS_WORDS(blocks) ((size_t)(blocks) * sizeof(struct p2p_block_status) / sizeof(uint32_t))
#define P2P_SUM_SLOTS 2u                         /* of a sum's inbox: a step's slot is its parity */
#define P2P_FLAG_AREA_WORDS (P2P_SUM_SLOTS * P2P_MAX_BLOCKS * P2P_FLAG_STRIDE_WORDS)   /* a push uses one slot */
#define P2P_WAIT_TIMEOUT_NS 2000000000ull        /* a kernel's wait for one flag */
#define P2P_HOST_DEADLINE_SECONDS 120.0          /* the host's wait for one queued run */
#define P2P_NANOSECONDS_PER_SECOND 1.0e9
#define P2P_MICROSECONDS_PER_SECOND 1.0e6

#define P2P_CHECK(call) p2p_check((call), #call, __FILE__, __LINE__)

static inline void p2p_check(cudaError_t status, const char *call, const char *file, int line)
{
    if (status == cudaSuccess) {
        return;
    }
    fprintf(stderr, "%s:%d: %s failed: %s\n", file, line, call, cudaGetErrorString(status));
    fflush(stdout);
    exit(P2P_EXIT_CRASH);
}

static inline double p2p_seconds_now(void)
{
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) {
        fprintf(stderr, "the monotonic clock cannot be read\n");
        exit(P2P_EXIT_CRASH);
    }
    return (double)now.tv_sec + (double)now.tv_nsec / P2P_NANOSECONDS_PER_SECOND;
}

/* Waits for every operation queued on a card's stream to end, at most seconds: 1 if they ended, 0 at the deadline.
 * An error of a queued operation ends the program: the card is then in a state no check may trust. */
static inline int p2p_wait_stream(int card, cudaStream_t stream, double seconds)
{
    double deadline = p2p_seconds_now() + seconds;
    P2P_CHECK(cudaSetDevice(card));
    for (;;) {
        cudaError_t status = cudaStreamQuery(stream);
        if (status == cudaSuccess) {
            return 1;
        }
        if (status != cudaErrorNotReady) {
            P2P_CHECK(status);
        }
        if (p2p_seconds_now() > deadline) {
            return 0;
        }
    }
}

/* 64 bits from the kernel's random source: every run's data is fresh. */
static inline uint64_t p2p_random_seed(void)
{
    uint64_t seed = 0;
    if (getrandom(&seed, sizeof(seed), 0) != (ssize_t)sizeof(seed)) {
        fprintf(stderr, "no random bytes from the kernel: %s\n", strerror(errno));
        exit(P2P_EXIT_CRASH);
    }
    return seed;
}

/* ---- patterns: a byte or a word is a function of a seed and its place, so stale or misplaced data fails ---- */

__host__ __device__ __forceinline__ uint64_t p2p_mix(uint64_t value)
{
    value += P2P_MIX_INCREMENT;
    value = (value ^ (value >> P2P_MIX_FIRST_SHIFT)) * P2P_MIX_FIRST_MULTIPLIER;
    value = (value ^ (value >> P2P_MIX_SECOND_SHIFT)) * P2P_MIX_SECOND_MULTIPLIER;
    return value ^ (value >> P2P_MIX_THIRD_SHIFT);
}

/* Byte `index` of the pattern of seed: byte index % 8 of the mix of seed + index / 8. */
__host__ __device__ __forceinline__ uint8_t p2p_pattern_byte(uint64_t seed, uint64_t index)
{
    return (uint8_t)(p2p_mix(seed + (index >> P2P_GROUP_SHIFT)) >>
                     ((index & (P2P_GROUP_BYTES - 1u)) * P2P_BITS_PER_BYTE));
}

/* Word `word` of round `round` of a push: one to one in the word, and every word changes from one round to the next. */
__host__ __device__ __forceinline__ uint32_t p2p_round_word(uint32_t seed, uint32_t round, uint32_t word)
{
    return (word * P2P_WORD_MULTIPLIER) ^ (seed + round * P2P_ROUND_STEP);
}

/* Bytes of got[0, count) unlike the pattern of seed from pattern index origin; the first one's position in got. */
static inline uint64_t p2p_count_wrong_bytes(const uint8_t *got, uint64_t count, uint64_t seed, uint64_t origin,
                                              uint64_t *first_wrong)
{
    uint64_t wrong = 0, group = UINT64_MAX, word = 0;
    for (uint64_t at = 0; at < count; at++) {
        uint64_t index = origin + at;
        if (index >> P2P_GROUP_SHIFT != group) {
            group = index >> P2P_GROUP_SHIFT;
            word = p2p_mix(seed + group);
        }
        if (got[at] != (uint8_t)(word >> ((index & (P2P_GROUP_BYTES - 1u)) * P2P_BITS_PER_BYTE))) {
            if (wrong == 0) {
                *first_wrong = at;
            }
            wrong++;
        }
    }
    return wrong;
}

/* ---- kernels. Grid-stride loops of P2P_THREADS threads a block; none uses an atomic operation. ---- */

__device__ __forceinline__ uint64_t p2p_globaltimer_ns(void)
{
    uint64_t now;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
    return now;
}

/* Polls a flag with acquire loads at system scope until it holds wanted: 1, or 0 once timeout_ns have passed. What the
 * writer stored before it released the flag is then seen by this thread's later loads, and after a barrier by its
 * block's: no fence on the waiting side. */
__device__ __forceinline__ uint32_t p2p_wait_flag(const volatile uint32_t *flag, uint32_t wanted, uint64_t timeout_ns)
{
    uint64_t deadline = p2p_globaltimer_ns() + timeout_ns;
    for (;;) {
        uint32_t value;
        asm volatile("ld.acquire.sys.u32 %0, [%1];" : "=r"(value) : "l"((uint64_t)(uintptr_t)flag) : "memory");
        if (value == wanted) {
            return 1u;
        }
        if (p2p_globaltimer_ns() > deadline) {
            return 0u;
        }
    }
}

/* Fills region[0, region_bytes): [test_begin, test_end) with the pattern of seed, counted from test_begin (or its
 * complement, so that every byte starts wrong), and every other byte with the pattern of guard_seed. */
__global__ void p2p_fill_kernel(uint8_t *region, uint64_t region_bytes, uint64_t test_begin, uint64_t test_end,
                                uint64_t seed, uint64_t guard_seed, uint32_t complement)
{
    uint64_t threads = (uint64_t)gridDim.x * blockDim.x;
    for (uint64_t at = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; at < region_bytes; at += threads) {
        uint8_t value;
        if (at >= test_begin && at < test_end) {
            value = p2p_pattern_byte(seed, at - test_begin);
            value = complement != 0u ? (uint8_t)~value : value;
        } else {
            value = p2p_pattern_byte(guard_seed, at);
        }
        region[at] = value;
    }
}

/* Copies bytes at any alignment: 16-byte words where source and destination share their alignment, else bytes.
 * Loads bypass the first-level cache: a peer's memory or a host page may have changed since an earlier kernel. */
__global__ void p2p_copy_kernel(uint8_t *destination, const uint8_t *source, uint64_t bytes)
{
    const uint64_t vector_bytes = sizeof(uint4);
    uint64_t threads = (uint64_t)gridDim.x * blockDim.x;
    uint64_t thread = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t misalignment = (uint64_t)(uintptr_t)destination % vector_bytes;
    uint64_t head = 0, vectors = 0;
    if (misalignment == (uint64_t)(uintptr_t)source % vector_bytes) {
        head = (vector_bytes - misalignment) % vector_bytes;
        head = head < bytes ? head : bytes;
        vectors = (bytes - head) / vector_bytes;
    }
    uint64_t body_end = head + vectors * vector_bytes;
    for (uint64_t at = thread; at < head; at += threads) {
        destination[at] = __ldcg(source + at);
    }
    uint4 *destination_vectors = (uint4 *)(destination + head);
    const uint4 *source_vectors = (const uint4 *)(source + head);
    for (uint64_t at = thread; at < vectors; at += threads) {
        destination_vectors[at] = __ldcg(source_vectors + at);
    }
    for (uint64_t at = body_end + thread; at < bytes; at += threads) {
        destination[at] = __ldcg(source + at);
    }
}

__global__ void p2p_zero_kernel(uint32_t *words, uint32_t word_count)
{
    for (uint32_t at = blockIdx.x * blockDim.x + threadIdx.x; at < word_count; at += gridDim.x * blockDim.x) {
        ((volatile uint32_t *)words)[at] = 0u;
    }
}

struct p2p_block_status {                       /* one per block of a push or a sum, in the card's own memory */
    uint32_t rounds_completed;
    uint32_t timed_out;
    uint32_t wrong_words;
    uint32_t first_wrong_word;                   /* the least index of a wrong word, P2P_NO_WORD if none */
};

/* Sums the threads' counts and takes the least of their word indices over the block, into *count and *least of
 * every thread. Every thread of the block must call it. */
__device__ __forceinline__ void p2p_block_reduce(uint32_t *count, uint32_t *least)
{
    __shared__ uint32_t counts[P2P_THREADS], leasts[P2P_THREADS];
    counts[threadIdx.x] = *count;
    leasts[threadIdx.x] = *least;
    __syncthreads();
    for (unsigned int stride = P2P_THREADS / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            counts[threadIdx.x] += counts[threadIdx.x + stride];
            leasts[threadIdx.x] = min(leasts[threadIdx.x], leasts[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    *count = counts[0];
    *least = leasts[0];
}

/* Counts the 8-byte words of words[0, word_count < P2P_NO_WORD) that differ from the pattern of seed: per block, the
 * count and the least index of a wrong word, at block_results[P2P_COUNT_WORDS_PER_BLOCK * block] and the word after. */
__global__ void p2p_count_kernel(const uint64_t *words, uint64_t word_count, uint64_t seed, uint32_t *block_results)
{
    uint64_t threads = (uint64_t)gridDim.x * blockDim.x;
    uint32_t wrong = 0, least = P2P_NO_WORD;
    for (uint64_t at = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; at < word_count; at += threads) {
        uint32_t differs = __ldcg((const unsigned long long *)words + at) != p2p_mix(seed + at);
        wrong += differs;
        least = differs != 0u && at < least ? (uint32_t)at : least;
    }
    p2p_block_reduce(&wrong, &least);
    if (threadIdx.x == 0) {
        block_results[P2P_COUNT_WORDS_PER_BLOCK * blockIdx.x] = wrong;
        block_results[P2P_COUNT_WORDS_PER_BLOCK * blockIdx.x + 1u] = least;
    }
}

/* The words of a payload that a block owns: payload_words split into gridDim.x slices, the last ones maybe empty. */
__device__ __forceinline__ void p2p_block_slice(uint32_t payload_words, uint32_t *first, uint32_t *end)
{
    uint32_t per_block = (payload_words + gridDim.x - 1u) / gridDim.x;
    uint32_t begin = blockIdx.x * per_block;
    *first = begin < payload_words ? begin : payload_words;
    *end = begin + per_block < payload_words ? begin + per_block : payload_words;
}

/* The four words of a push's 16-byte unit from first_word; damaged_word, if among them, with one bit changed. */
__device__ __forceinline__ uint4 p2p_round_vector(uint32_t seed, uint32_t round, uint32_t first_word,
                                                  uint32_t damaged_word)
{
    uint32_t words[P2P_VECTOR_WORDS];
    uint4 vector;
#pragma unroll
    for (uint32_t at = 0; at < P2P_VECTOR_WORDS; at++) {
        words[at] = p2p_round_word(seed, round, first_word + at) ^ (first_word + at == damaged_word ? 1u : 0u);
    }
    memcpy(&vector, words, sizeof(vector));
    return vector;
}

/* The words of a 16-byte unit unlike wanted's; *least is lowered to the index of the first (the unit's first word is
 * word first_word). */
__device__ __forceinline__ uint32_t p2p_vector_wrong(uint4 got, uint4 wanted, uint32_t first_word, uint32_t *least)
{
    uint32_t got_words[P2P_VECTOR_WORDS], wanted_words[P2P_VECTOR_WORDS], wrong = 0;
    memcpy(got_words, &got, sizeof(got));
    memcpy(wanted_words, &wanted, sizeof(wanted));
#pragma unroll
    for (uint32_t at = 0; at < P2P_VECTOR_WORDS; at++) {
        uint32_t differs = got_words[at] != wanted_words[at];
        wrong += differs;
        *least = differs != 0u && first_word + at < *least ? first_word + at : *least;
    }
    return wrong;
}

/* A push moves 16-byte units where its words divide into them, else words: the slices are of units. */
__device__ __forceinline__ uint32_t p2p_push_units(uint32_t payload_words)
{
    return payload_words % P2P_VECTOR_WORDS == 0u ? payload_words / P2P_VECTOR_WORDS : payload_words;
}

/* The sender of a push: per round, each block stores its slice of the round's payload into the receiver's memory;
 * after the barrier its thread 0 fences (cumulative: it orders the block's stores, which the barrier made its own,
 * before what follows) and stores the round into the block's flag there, then waits until the receiver's
 * acknowledgement holds the round. One fence a block and round on each side, as the host-page mailbox of b02b, and
 * acquire loads where a side waits (p2p_wait_flag). Weaker ordering could only show stale words, which fail: never a
 * false pass. The damaged instance changes one bit of one word in one round: the receiver must report that word. */
template <bool damaged>
__global__ void p2p_push_sender_kernel(uint32_t *payload, uint32_t *flags, const uint32_t *acknowledgements,
                                       uint32_t payload_words, uint32_t rounds, uint32_t seed, uint64_t timeout_ns,
                                       uint32_t damaged_round, uint32_t damaged_word, struct p2p_block_status *status)
{
    __shared__ uint32_t acknowledged;
    uint32_t first = 0, end = 0, completed = 0;
    int vectors = payload_words % P2P_VECTOR_WORDS == 0u;
    p2p_block_slice(p2p_push_units(payload_words), &first, &end);
    uint32_t flag_index = blockIdx.x * P2P_FLAG_STRIDE_WORDS;
    for (uint32_t round = 1; round <= rounds; round++) {
        uint32_t damaged_here = damaged && round == damaged_round ? damaged_word : P2P_NO_WORD;
        for (uint32_t unit = first + threadIdx.x; unit < end; unit += blockDim.x) {
            if (vectors) {
                ((uint4 *)payload)[unit] = p2p_round_vector(seed, round, unit * P2P_VECTOR_WORDS, damaged_here);
            } else {
                payload[unit] = p2p_round_word(seed, round, unit) ^ (unit == damaged_here ? 1u : 0u);
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            __threadfence_system();
            ((volatile uint32_t *)flags)[flag_index] = round;
            acknowledged = p2p_wait_flag((const volatile uint32_t *)acknowledgements + flag_index, round, timeout_ns);
        }
        __syncthreads();
        if (acknowledged == 0u) {
            break;
        }
        completed = round;
    }
    if (threadIdx.x == 0) {
        status[blockIdx.x].rounds_completed = completed;
        status[blockIdx.x].timed_out = completed != rounds;
        status[blockIdx.x].first_wrong_word = P2P_NO_WORD;
    }
}

/* The receiver of a push: per round, each block's thread 0 waits for the block's flag in its own memory; after the
 * barrier the block loads its slice past the first-level cache, P2P_LOADS_IN_FLIGHT units a thread before it checks
 * them, counts the words that are not the round's, and after a second barrier thread 0 fences and acknowledges. */
__global__ void p2p_push_receiver_kernel(const uint32_t *payload, const uint32_t *flags, uint32_t *acknowledgements,
                                         uint32_t payload_words, uint32_t rounds, uint32_t seed, uint64_t timeout_ns,
                                         struct p2p_block_status *status)
{
    __shared__ uint32_t arrived;
    uint32_t first = 0, end = 0, completed = 0, wrong = 0, least = P2P_NO_WORD;
    int vectors = payload_words % P2P_VECTOR_WORDS == 0u;
    p2p_block_slice(p2p_push_units(payload_words), &first, &end);
    uint32_t flag_index = blockIdx.x * P2P_FLAG_STRIDE_WORDS;
    for (uint32_t round = 1; round <= rounds; round++) {
        if (threadIdx.x == 0) {
            arrived = p2p_wait_flag((const volatile uint32_t *)flags + flag_index, round, timeout_ns);
        }
        __syncthreads();
        if (arrived == 0u) {
            break;
        }
        for (uint32_t base = first + threadIdx.x; base < end; base += blockDim.x * P2P_LOADS_IN_FLIGHT) {
            uint4 loaded[P2P_LOADS_IN_FLIGHT];
#pragma unroll
            for (uint32_t at = 0; at < P2P_LOADS_IN_FLIGHT; at++) {
                uint32_t unit = base + at * blockDim.x;
                loaded[at] = make_uint4(0u, 0u, 0u, 0u);
                if (unit < end && vectors) {
                    loaded[at] = __ldcv((const uint4 *)payload + unit);
                } else if (unit < end) {
                    loaded[at].x = __ldcv(payload + unit);
                }
            }
#pragma unroll
            for (uint32_t at = 0; at < P2P_LOADS_IN_FLIGHT; at++) {
                uint32_t unit = base + at * blockDim.x;
                if (unit < end && vectors) {
                    wrong += p2p_vector_wrong(loaded[at],
                                              p2p_round_vector(seed, round, unit * P2P_VECTOR_WORDS, P2P_NO_WORD),
                                              unit * P2P_VECTOR_WORDS, &least);
                } else if (unit < end && loaded[at].x != p2p_round_word(seed, round, unit)) {
                    wrong++;
                    least = unit < least ? unit : least;
                }
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            __threadfence_system();
            ((volatile uint32_t *)acknowledgements)[flag_index] = round;
        }
        completed = round;
    }
    p2p_block_reduce(&wrong, &least);
    if (threadIdx.x == 0) {
        status[blockIdx.x].rounds_completed = completed;
        status[blockIdx.x].timed_out = completed != rounds;
        status[blockIdx.x].wrong_words = wrong;
        status[blockIdx.x].first_wrong_word = least;
    }
}

/* ---- the host launcher of a push ---- */

struct p2p_damage {                              /* where a damaged run changes one bit; active 0: an intact run */
    int active;
    uint32_t round;
    uint32_t index;
};

struct p2p_outcome {                             /* of one push run, or a sum's (link_hop_peer.cuh) */
    double microseconds_per_round;
    uint32_t rounds_completed;                   /* the fewest any block of either side completed */
    uint32_t timed_out;                          /* blocks of either side that gave up waiting */
    uint32_t wrong_words;                        /* words the receivers found not to be what was sent */
    uint32_t first_wrong_word;                   /* the least index of one of them, P2P_NO_WORD if none */
    uint32_t result_differing;                   /* a sum's: values of the last step's result unlike the host's */
};

struct p2p_push_buffers {
    uint32_t *payload;                           /* the receiver's memory (or a host page): payload words */
    uint32_t *flags;                             /* the receiver's memory: one flag per block */
    uint32_t *acknowledgements;                  /* where the sender polls: its own memory, or a host page */
    struct p2p_block_status *sender_status;      /* the sender's own memory, P2P_MAX_BLOCKS entries */
    struct p2p_block_status *receiver_status;    /* the receiver's own memory */
};

static inline void p2p_zero_words(int card, cudaStream_t stream, void *words, size_t word_count)
{
    P2P_CHECK(cudaSetDevice(card));
    p2p_zero_kernel<<<1, P2P_THREADS, 0, stream>>>((uint32_t *)words, (uint32_t)word_count);
    P2P_CHECK(cudaGetLastError());
}

/* Adds the block statuses of one side into an outcome. */
static inline void p2p_add_statuses(int card, const struct p2p_block_status *card_status, uint32_t blocks,
                                    uint32_t rounds, struct p2p_outcome *outcome)
{
    struct p2p_block_status statuses[P2P_MAX_BLOCKS];
    P2P_CHECK(cudaSetDevice(card));
    P2P_CHECK(cudaMemcpy(statuses, card_status, blocks * sizeof(statuses[0]), cudaMemcpyDeviceToHost));
    for (uint32_t block = 0; block < blocks; block++) {
        if (statuses[block].rounds_completed > rounds) {
            fprintf(stderr, "a block reports %u rounds of %u\n", statuses[block].rounds_completed, rounds);
            exit(P2P_EXIT_CRASH);
        }
        outcome->rounds_completed = statuses[block].rounds_completed < outcome->rounds_completed
                                        ? statuses[block].rounds_completed : outcome->rounds_completed;
        outcome->timed_out += statuses[block].timed_out != 0u;
        outcome->wrong_words += statuses[block].wrong_words;
        outcome->first_wrong_word = statuses[block].first_wrong_word < outcome->first_wrong_word
                                        ? statuses[block].first_wrong_word : outcome->first_wrong_word;
    }
}

/* One push run: rounds of payload_words from sender into the receiver's memory, each acknowledged, over `blocks`
 * blocks. Returns 0 if the host's deadline passed (the cards are then in an unknown state), else fills outcome. */
static inline int p2p_run_push(int sender, cudaStream_t sender_stream, int receiver, cudaStream_t receiver_stream,
                               const struct p2p_push_buffers *buffers, uint32_t payload_words, uint32_t blocks,
                               uint32_t rounds, uint32_t seed, const struct p2p_damage *damage,
                               struct p2p_outcome *outcome)
{
    if (blocks == 0 || blocks > P2P_MAX_BLOCKS || rounds == 0 ||
        (payload_words % P2P_VECTOR_WORDS == 0 && (uintptr_t)buffers->payload % sizeof(uint4) != 0) ||
        (damage->active && (damage->round == 0 || damage->round > rounds || damage->index >= payload_words))) {
        fprintf(stderr, "a push of %u blocks, %u rounds, damage at round %u word %u of %u is refused\n", blocks,
                rounds, damage->round, damage->index, payload_words);
        exit(P2P_EXIT_CRASH);
    }
    memset(outcome, 0, sizeof(*outcome));
    outcome->rounds_completed = rounds;
    outcome->first_wrong_word = P2P_NO_WORD;
    p2p_zero_words(receiver, receiver_stream, buffers->flags, (size_t)blocks * P2P_FLAG_STRIDE_WORDS);
    p2p_zero_words(receiver, receiver_stream, buffers->receiver_status, P2P_STATUS_WORDS(blocks));
    p2p_zero_words(sender, sender_stream, buffers->acknowledgements, (size_t)blocks * P2P_FLAG_STRIDE_WORDS);
    p2p_zero_words(sender, sender_stream, buffers->sender_status, P2P_STATUS_WORDS(blocks));
    if (!p2p_wait_stream(receiver, receiver_stream, P2P_HOST_DEADLINE_SECONDS) ||
        !p2p_wait_stream(sender, sender_stream, P2P_HOST_DEADLINE_SECONDS)) {
        return 0;
    }
    double started = p2p_seconds_now();
    P2P_CHECK(cudaSetDevice(receiver));
    p2p_push_receiver_kernel<<<blocks, P2P_THREADS, 0, receiver_stream>>>(
        buffers->payload, buffers->flags, buffers->acknowledgements, payload_words, rounds, seed, P2P_WAIT_TIMEOUT_NS,
        buffers->receiver_status);
    P2P_CHECK(cudaGetLastError());
    P2P_CHECK(cudaSetDevice(sender));
    if (damage->active) {
        p2p_push_sender_kernel<true><<<blocks, P2P_THREADS, 0, sender_stream>>>(
            buffers->payload, buffers->flags, buffers->acknowledgements, payload_words, rounds, seed,
            P2P_WAIT_TIMEOUT_NS, damage->round, damage->index, buffers->sender_status);
    } else {
        p2p_push_sender_kernel<false><<<blocks, P2P_THREADS, 0, sender_stream>>>(
            buffers->payload, buffers->flags, buffers->acknowledgements, payload_words, rounds, seed,
            P2P_WAIT_TIMEOUT_NS, 0u, 0u, buffers->sender_status);
    }
    P2P_CHECK(cudaGetLastError());
    if (!p2p_wait_stream(sender, sender_stream, P2P_HOST_DEADLINE_SECONDS) ||
        !p2p_wait_stream(receiver, receiver_stream, P2P_HOST_DEADLINE_SECONDS)) {
        return 0;
    }
    outcome->microseconds_per_round = (p2p_seconds_now() - started) * P2P_MICROSECONDS_PER_SECOND / (double)rounds;
    p2p_add_statuses(sender, buffers->sender_status, blocks, rounds, outcome);
    p2p_add_statuses(receiver, buffers->receiver_status, blocks, rounds, outcome);
    return 1;
}

/* The PCI location of a card the CUDA runtime sees, as the driver's directory names it. */
static inline void p2p_bus_text(int card, char *text)
{
    cudaDeviceProp properties;
    P2P_CHECK(cudaGetDeviceProperties(&properties, card));
    snprintf(text, P2P_BUS_TEXT_BYTES, "%04x:%02x:%02x.0", (unsigned int)properties.pciDomainID,
             (unsigned int)properties.pciBusID, (unsigned int)properties.pciDeviceID);
}

#endif
