/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * p2p_gate_probe: the content gate of design/peer_to_peer.md 6, which decides per ordered pair of cards whether peer
 * access may be used. The driver's answer, its sample program and a few floats all passed the known fault (bytes past
 * 128 KiB wrong while every call succeeds), so none of them counts here.
 *
 * For every ordered pair (an accessor reaching an owner's memory) it prints the driver's answer and attributes. A pair
 * of two different models of card (a CMP 170HX and the RTX PRO 6000) stays on host pages even when offered: printed
 * as offered and unverified, never enabled (design 1 and P6). On every other pair the driver offers it enables peer
 * access, all pairs at once and before it takes the cards' memory, then checks by content, three ways:
 *   copy    cudaMemcpyPeer from the accessor's memory into the owner's;
 *   store   a kernel on the accessor storing through the peer mapping into the owner's memory;
 *   load    a kernel on the accessor loading, through the peer mapping, what the owner wrote with its own seed;
 * each at 128 KiB - 1, 128 KiB, 128 KiB + 1, 512 KiB - 1, 512 KiB, 512 KiB + 1, 1, 8 and 32 MiB, and at four places:
 * the start of the card's memory as this process holds it, its end (the end of the largest block the card gives, all
 * its free memory but a margin; the ARENA line says whether that lies within the card's last GiB), and both at odd
 * byte offsets that differ between the two sides, so that kernels store single bytes. Every check has fresh data, its
 * seed printed. A destination starts as the complement of what must arrive, and 64 KiB of guard bytes on each side
 * must stay as they were. Every byte is compared on the host after an ordinary copy from the card that holds the
 * result. Then the engine's message shapes (20 B, 20 KiB, 84 KiB, 160 KiB) as mailbox pushes: payload, fence, flag,
 * the receiver polling its own memory, a pattern per round, over 1 and 8 blocks. Then a soak: every tested pair at
 * once, both directions, 32 MiB copies and stores checked on the owner. The AER counters of every card and of the
 * port above it are read before and after (the batch reads the kernel log's Xid lines).
 *
 * VERDICT. One wrong byte, one timeout, one failed call to enable, one uncorrectable AER error on any card or port, or
 * an AER counter that cannot be read, prints GATE FAILED and refuses peer access on every pair (exit 1); otherwise a
 * PASS line per tested pair (exit 0). With no pair offered between two cards of one model: exit 0, verdict
 * `none_offered`. --record keeps the verdict in the record of p2p_record.h, which says `incomplete` from the moment
 * the checks begin, so a run that dies passes nothing; a trusted record under this machine's key is printed and not
 * run again unless --rerun. A record that cannot be written is removed (exit 3 if it cannot be removed either).
 * --record takes only the whole gate with every card the driver holds visible: no --skip-shapes, --arena-mib or
 * --seed, and no fewer seconds of soak or rounds than the defaults. A failed call: exit 4, no verdict.
 *
 * --stand-in-host runs the whole gate on every ordered pair with page-locked host memory, mapped into every card,
 * standing in for each owner's memory: nothing is enabled and no record is written. With --corrupt-byte WAY (copy,
 * store, load, push or soak) one byte of one check of that kind is damaged: the first pair; 512 KiB + 1 at the odd
 * end, byte 128 KiB + 5, for the three ways; 20 KiB over one block, round rounds / 2, word 1,000, for the push; the
 * first iteration's 32 MiB at byte 16 MiB + 3 for the soak. That run exits 0 only if exactly that byte or word was
 * reported, at its place, and nothing else failed. --arena-mib caps each card's block (a run under compute-sanitizer);
 * --skip-shapes leaves out the pushes, whose kernels wait on each other (the sanitizer may run kernels one at a time).
 *
 *   p2p_gate_probe [--record PATH [--rerun]] [--soak-seconds 0..3600] [--rounds 1..100000] [--seed HEX]
 *                  [--stand-in-host [--corrupt-byte copy|store|load|push|soak]] [--arena-mib N] [--skip-shapes]
 *
 * Measurement tool: not part of the engine.
 */
#include "p2p_common.cuh"

#define BYTES_PER_KIB 1024ull
#define BYTES_PER_MIB (1024ull * BYTES_PER_KIB)
#define EDGE_SMALL (128ull * BYTES_PER_KIB)          /* where the untrapped mailbox path breaks (design 3) */
#define EDGE_LARGE (512ull * BYTES_PER_KIB)
#define MIDDLE_CHECK (8ull * BYTES_PER_MIB)
#define LARGEST_CHECK (32ull * BYTES_PER_MIB)
#define GUARD_BYTES (64ull * BYTES_PER_KIB)          /* checked on each side of a destination */
#define ODD_SLACK 16ull                              /* room for the odd offsets */
#define REMOTE_ODD_SHIFT 1ull                        /* unequal, so a kernel moves single bytes */
#define LOCAL_ODD_SHIFT 3ull
#define PLACE_ALIGNMENT 256ull
#define REGION_ALIGNMENT (2ull * BYTES_PER_MIB)
#define CHECK_REGION_BYTES (34ull * BYTES_PER_MIB)   /* a guard, the largest check, the slack, a guard: rounded up */
#define SOAK_BYTES (32ull * BYTES_PER_MIB)
#define MAILBOX_BYTES REGION_ALIGNMENT               /* flags, then the payload of the largest shape */
#define FLAG_AREA_BYTES (P2P_FLAG_AREA_WORDS * sizeof(uint32_t))
#define SMALL_REGION_BYTES REGION_ALIGNMENT          /* the soak's counts; the pushes' block statuses */
#define BYTES_PER_GIB (1024ull * BYTES_PER_MIB)
#define ARENA_MARGIN_BYTES (128ull * BYTES_PER_MIB)  /* left to the driver on each card (its peer mappings among it) */
#define ARENA_STEP_BYTES (1024ull * BYTES_PER_MIB)
#define ARENA_TRIES 4
#define BLOCKS_PER_SM 4u
#define MOST_GRID_BLOCKS 1024u
#define DEFAULT_SOAK_SECONDS 60ull
#define MOST_SOAK_SECONDS 3600ull
#define DEFAULT_ROUNDS 2000ull
#define MOST_ROUNDS 100000ull
#define MOST_ARENA_MIB (1024ull * 1024ull)
#define CHECK_DEADLINE_SECONDS 60.0
#define DAMAGED_SIZE (EDGE_LARGE + 1ull)
#define DAMAGED_BYTE (EDGE_SMALL + 5ull)             /* in the test, past the 128 KiB edge */
#define LARGEST_SHAPE_BYTES (160u * 1024u)
#define DAMAGED_SHAPE_BYTES (20u * 1024u)
#define DAMAGED_SHAPE_BLOCKS 1u
#define DAMAGED_PUSH_WORD 1000u
#define DAMAGED_SOAK_BYTE (SOAK_BYTES / 2ull + 3ull)
#define CAPABILITY_PER_MAJOR 10
#define CAPABILITY_SM_80 80
#define CAPABILITY_SM_120 120
#define EXIT_REFUSED 1
#define EXIT_SETUP 2
#define EXIT_RECORD_STUCK 3                          /* a record that could be neither written nor removed */
#define REASON_BYTES 192
#define MOST_SOAK_LINES_PER_PAIR 10u

static_assert(CHECK_REGION_BYTES % REGION_ALIGNMENT == 0 &&
              CHECK_REGION_BYTES >= 2 * GUARD_BYTES + LARGEST_CHECK + ODD_SLACK + PLACE_ALIGNMENT,
              "a check region holds the largest check, its slack and both guards");
static_assert(FLAG_AREA_BYTES + LARGEST_SHAPE_BYTES <= MAILBOX_BYTES,
              "the mailbox holds the flags and the largest shape");
static_assert(2u * P2P_MAX_BLOCKS * sizeof(struct p2p_block_status) <= SMALL_REGION_BYTES &&
              (P2P_MAX_CARDS - 1u) * MOST_GRID_BLOCKS * P2P_COUNT_WORDS_PER_BLOCK * sizeof(uint32_t) <=
                  SMALL_REGION_BYTES && SOAK_BYTES / sizeof(uint64_t) < P2P_NO_WORD,
              "the small regions hold the statuses and the soak's counts; a soak's word has an index");
static_assert(DAMAGED_BYTE < DAMAGED_SIZE && DAMAGED_PUSH_WORD < DAMAGED_SHAPE_BYTES / sizeof(uint32_t),
              "the damaged byte lies inside its check");

static const uint64_t check_sizes[] = {EDGE_SMALL - 1, EDGE_SMALL, EDGE_SMALL + 1, EDGE_LARGE - 1, EDGE_LARGE,
                                       EDGE_LARGE + 1, BYTES_PER_MIB, MIDDLE_CHECK, LARGEST_CHECK};
#define CHECK_SIZE_COUNT (sizeof(check_sizes) / sizeof(check_sizes[0]))
static const uint32_t shape_bytes[] = {20u, DAMAGED_SHAPE_BYTES, 84u * 1024u, LARGEST_SHAPE_BYTES};
#define SHAPE_COUNT (sizeof(shape_bytes) / sizeof(shape_bytes[0]))
static const uint32_t shape_blocks[] = {1u, 8u};
#define SHAPE_BLOCK_COUNT (sizeof(shape_blocks) / sizeof(shape_blocks[0]))

enum way { WAY_COPY, WAY_STORE, WAY_LOAD, WAYS };
static const char *const way_names[WAYS] = {"copy", "store", "load"};
enum damage_kind { DAMAGE_PUSH = WAYS, DAMAGE_SOAK, DAMAGE_KINDS };      /* the ways come first */
static const char *const damage_names[DAMAGE_KINDS] = {"copy", "store", "load", "push", "soak"};
enum placement { PLACE_LOW, PLACE_HIGH, PLACE_LOW_ODD, PLACE_HIGH_ODD, PLACEMENTS };
static const char *const placement_names[PLACEMENTS] = {"low", "high", "low_odd", "high_odd"};
enum region { LOW_REMOTE, LOW_LOCAL, MAILBOX, SOAK_DESTINATION, SOAK_SOURCE, COUNTS, STATUS, LOWER_END, HIGH_LOCAL,
              HIGH_REMOTE };

struct gate_card {
    int ordinal;
    char bus[P2P_BUS_TEXT_BYTES];
    char name[P2P_NAME_BYTES];
    int capability, sms;
    unsigned int blocks;                         /* the grid of the fill, copy and count kernels */
    size_t free_bytes, total_bytes;
    uint8_t *arena;                              /* card memory: every region; the high ones at its very end */
    uint64_t arena_bytes;
    uint8_t *stand_in;                           /* --stand-in-host: the remote regions, in mapped host pages */
    cudaStream_t stream;
    int aer_readable;
    struct p2p_aer aer_before;
};

struct pair_state {
    int offered, across_models, tested, failed;
    char reason[REASON_BYTES];
    uint64_t checks, bytes_checked, soak_iterations;
};

static struct gate_card cards[P2P_MAX_CARDS];
static int card_count, stand_in, skip_shapes, damaged_kind = -1;
static uint64_t run_seed, seed_counter, arena_cap_bytes;
static uint8_t *staging;                         /* page-locked: a result's window, downloaded for the comparison */
static uint32_t *acknowledgement_page;           /* page-locked, mapped into every card: the pushes' acknowledgements */
static struct pair_state pairs[P2P_MAX_CARDS][P2P_MAX_CARDS];
static int first_accessor = -1, first_owner = -1;     /* the first tested pair: the one a damaged run damages */
static unsigned int designated_reported, other_failures;
static const char *record_path;
static struct p2p_key key;

static uint64_t next_seed(void)
{
    return p2p_mix(run_seed + ++seed_counter);
}

static uint64_t align_down(uint64_t value, uint64_t alignment)
{
    return value / alignment * alignment;
}

static int is_first_pair(int accessor, int owner)
{
    return accessor == first_accessor && owner == first_owner;
}

/* Keeps the verdict in the record. A record that cannot be written is removed, so that no older verdict stays behind
 * it; one that can be neither written nor removed ends the run before anything else (exit 3). */
static void write_record(const char *verdict, const char *pair_list)
{
    if (record_path == NULL || stand_in) {
        return;
    }
    if (key.complete && p2p_record_write(record_path, &key, verdict, pair_list)) {
        printf("RECORD written verdict=%s\n", verdict);
        return;
    }
    printf("RECORD NOT WRITTEN verdict=%s: %s\n", verdict, key.complete ? strerror(errno) : "the key is incomplete");
    if (unlink(record_path) == 0 || errno == ENOENT) {
        printf("RECORD removed: this run keeps no verdict\n");
        return;
    }
    printf("RECORD STUCK: it can be neither written nor removed (%s); no peer user may trust it\n", strerror(errno));
    fflush(stdout);
    exit(EXIT_RECORD_STUCK);
}

/* The host's deadline passed: the cards are in an unknown state; every pair is refused and nothing else runs. */
static void timed_out(int accessor, int owner, const char *what)
{
    printf("GATE FAILED pair=%s>%s reason=\"timeout: %s\"\n", cards[accessor].bus, cards[owner].bus, what);
    printf("GATE refused: peer access is refused on every pair\n");
    printf("VERDICT gate=refused reason=timeout\n");
    write_record("refused", "");
    fflush(stdout);
    exit(EXIT_REFUSED);
}

static void fail_pair(int accessor, int owner, const char *reason)
{
    struct pair_state *pair = &pairs[accessor][owner];
    if (!pair->failed) {
        snprintf(pair->reason, sizeof(pair->reason), "%s", reason);
    }
    pair->failed = 1;
}

/* Judges one check: wrong bytes or words, and other trouble (a guard changed, a timeout, rounds missing). The one
 * check a shown-failing run damaged must report exactly one, at its place, and nothing else. */
static const char *judge(int accessor, int owner, int designated, uint64_t wrong, int right_place, int other_trouble,
                         const char *what)
{
    if (designated && wrong == 1 && right_place && !other_trouble) {
        designated_reported++;
        fail_pair(accessor, owner, "the byte damaged on purpose");
        return "reported";
    }
    if (!designated && wrong == 0 && !other_trouble) {
        return "ok";
    }
    other_failures++;
    fail_pair(accessor, owner, what);
    return "FAILED";
}

static void wait_or_fail(const struct gate_card *card, int accessor, int owner, const char *what)
{
    if (!p2p_wait_stream(card->ordinal, card->stream, CHECK_DEADLINE_SECONDS)) {
        timed_out(accessor, owner, what);
    }
}

/* ---- the card's memory: one block, the regions at fixed offsets, the high ones at its very end ---- */

static uint64_t arena_offset(const struct gate_card *card, enum region kind)
{
    uint64_t others = (uint64_t)card_count - 1u;
    uint64_t soak = 2u * CHECK_REGION_BYTES + MAILBOX_BYTES;
    switch (kind) {
    case LOW_REMOTE: return 0;
    case LOW_LOCAL: return CHECK_REGION_BYTES;
    case MAILBOX: return 2u * CHECK_REGION_BYTES;
    case SOAK_DESTINATION: return soak;
    case SOAK_SOURCE: return soak + others * SOAK_BYTES;
    case COUNTS: return soak + 2u * others * SOAK_BYTES;
    case STATUS: return soak + 2u * others * SOAK_BYTES + SMALL_REGION_BYTES;
    case LOWER_END: return soak + 2u * others * SOAK_BYTES + 2u * SMALL_REGION_BYTES;
    case HIGH_LOCAL: return card->arena_bytes - 2u * CHECK_REGION_BYTES;
    case HIGH_REMOTE: return card->arena_bytes - CHECK_REGION_BYTES;
    }
    return 0;
}

/* The regions the peer reaches; --stand-in-host keeps them in host pages, in this order. */
static int stand_in_offset(enum region kind, uint64_t *offset)
{
    switch (kind) {
    case LOW_REMOTE: *offset = 0; return 1;
    case HIGH_REMOTE: *offset = CHECK_REGION_BYTES; return 1;
    case MAILBOX: *offset = 2u * CHECK_REGION_BYTES; return 1;
    case SOAK_DESTINATION: *offset = 2u * CHECK_REGION_BYTES + MAILBOX_BYTES; return 1;
    default: return 0;
    }
}

static uint8_t *region_of(const struct gate_card *card, enum region kind)
{
    uint64_t offset = 0;
    if (stand_in && stand_in_offset(kind, &offset)) {
        return card->stand_in + offset;
    }
    return card->arena + arena_offset(card, kind);
}

/* The largest block the card gives, less a margin for the driver: its end is as near the end of the card's memory as
 * this process can reach (the addresses are virtual; where the pages lie is the driver's choice). */
static void open_arena(struct gate_card *card)
{
    size_t free_bytes = 0, total_bytes = 0;
    P2P_CHECK(cudaSetDevice(card->ordinal));
    P2P_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    uint64_t smallest = arena_offset(card, LOWER_END) + 2u * CHECK_REGION_BYTES;
    uint64_t wanted =
        free_bytes > ARENA_MARGIN_BYTES ? align_down(free_bytes - ARENA_MARGIN_BYTES, REGION_ALIGNMENT) : 0;
    if (arena_cap_bytes != 0 && wanted > arena_cap_bytes) {
        wanted = arena_cap_bytes;
    }
    for (int attempt = 0; attempt < ARENA_TRIES && wanted >= smallest; attempt++) {
        void *arena = NULL;
        if (cudaMalloc(&arena, wanted) == cudaSuccess) {
            card->arena = (uint8_t *)arena;
            card->arena_bytes = wanted;
            break;
        }
        (void)cudaGetLastError();
        wanted = wanted > ARENA_STEP_BYTES ? wanted - ARENA_STEP_BYTES : 0;
    }
    if (card->arena == NULL) {
        fprintf(stderr, "card %s: no block of at least %" PRIu64 " bytes (%zu free)\n", card->bus, smallest,
                free_bytes);
        exit(EXIT_SETUP);
    }
    if (stand_in) {
        uint64_t bytes = 2u * CHECK_REGION_BYTES + MAILBOX_BYTES + ((uint64_t)card_count - 1u) * SOAK_BYTES;
        void *host = NULL;
        P2P_CHECK(cudaHostAlloc(&host, bytes, cudaHostAllocMapped | cudaHostAllocPortable));
        card->stand_in = (uint8_t *)host;
    }
    uint64_t short_of_total = (uint64_t)total_bytes - card->arena_bytes;
    printf("ARENA card=%s total_mib=%llu free_mib=%llu arena_mib=%llu arena_short_of_total_mib=%llu "
           "high_within_last_gib=%s remote=%s\n", card->bus, (unsigned long long)(total_bytes / BYTES_PER_MIB),
           (unsigned long long)(free_bytes / BYTES_PER_MIB), (unsigned long long)(card->arena_bytes / BYTES_PER_MIB),
           (unsigned long long)(short_of_total / BYTES_PER_MIB),
           short_of_total + 2u * CHECK_REGION_BYTES <= BYTES_PER_GIB ? "yes" : "no", stand_in ? "host_pages" : "card");
}

/* A pointer every card uses as it is: a host page mapped through the unified address space must have one address. */
static void check_one_address(void *host)
{
    for (int card = 0; card < card_count; card++) {
        void *device = NULL;
        P2P_CHECK(cudaSetDevice(cards[card].ordinal));
        P2P_CHECK(cudaHostGetDevicePointer(&device, host, 0));
        if (device != host) {
            fprintf(stderr, "card %s maps a host page at another address: this tool passes one pointer to all\n",
                    cards[card].bus);
            exit(EXIT_SETUP);
        }
    }
}

/* ---- the three ways, at every size and place ---- */

/* Where a test of `size` bytes begins in its region. */
static uint64_t test_begin(enum placement placement, uint64_t size, uint64_t odd_shift)
{
    switch (placement) {
    case PLACE_LOW: return GUARD_BYTES;
    case PLACE_LOW_ODD: return GUARD_BYTES + odd_shift;
    case PLACE_HIGH: return align_down(CHECK_REGION_BYTES - GUARD_BYTES - size, PLACE_ALIGNMENT);
    case PLACE_HIGH_ODD:
        return align_down(CHECK_REGION_BYTES - GUARD_BYTES - size - ODD_SLACK, PLACE_ALIGNMENT) + odd_shift;
    case PLACEMENTS: break;
    }
    return GUARD_BYTES;
}

static void fill(const struct gate_card *card, uint8_t *bytes, uint64_t count, uint64_t test_begin_at,
                 uint64_t test_end_at, uint64_t seed, uint64_t guard_seed, uint32_t complement)
{
    P2P_CHECK(cudaSetDevice(card->ordinal));
    p2p_fill_kernel<<<card->blocks, P2P_THREADS, 0, card->stream>>>(bytes, count, test_begin_at, test_end_at, seed,
                                                                      guard_seed, complement);
    P2P_CHECK(cudaGetLastError());
}

/* One check of one way, size and place on one ordered pair. */
static void run_check(int accessor_index, int owner_index, enum way way, uint64_t size, enum placement placement)
{
    struct gate_card *accessor = &cards[accessor_index], *owner = &cards[owner_index];
    int high = placement == PLACE_HIGH || placement == PLACE_HIGH_ODD;
    uint8_t *remote = region_of(owner, high ? HIGH_REMOTE : LOW_REMOTE);
    uint8_t *local = region_of(accessor, high ? HIGH_LOCAL : LOW_LOCAL);
    uint64_t remote_begin = test_begin(placement, size, REMOTE_ODD_SHIFT);
    uint64_t local_begin = test_begin(placement, size, LOCAL_ODD_SHIFT);
    uint64_t seed = next_seed(), guard_seed = next_seed();
    int loads = way == WAY_LOAD;
    struct gate_card *source_card = loads ? owner : accessor, *result_card = loads ? accessor : owner;
    uint8_t *source = (loads ? remote : local) + (loads ? remote_begin : local_begin);
    uint8_t *result_region = loads ? local : remote;
    uint64_t result_begin = loads ? local_begin : remote_begin;
    uint64_t window_begin = result_begin - GUARD_BYTES, window_bytes = size + 2u * GUARD_BYTES;
    uint64_t test_in_window = result_begin - window_begin;
    int designated = damaged_kind == (int)way && is_first_pair(accessor_index, owner_index) && size == DAMAGED_SIZE &&
                     placement == PLACE_HIGH_ODD;

    fill(source_card, source, size, 0, size, seed, guard_seed, 0u);
    fill(result_card, result_region + window_begin, window_bytes, test_in_window, test_in_window + size, seed,
         guard_seed, 1u);
    wait_or_fail(source_card, accessor_index, owner_index, "filling the source");
    wait_or_fail(result_card, accessor_index, owner_index, "filling the destination");
    if (designated) {
        uint8_t damaged = (uint8_t)~p2p_pattern_byte(seed, DAMAGED_BYTE);
        P2P_CHECK(cudaSetDevice(source_card->ordinal));
        P2P_CHECK(cudaMemcpy(source + DAMAGED_BYTE, &damaged, 1, cudaMemcpyDefault));
    }
    uint8_t *destination = result_region + result_begin;
    P2P_CHECK(cudaSetDevice(accessor->ordinal));
    if (way == WAY_COPY && !stand_in) {
        P2P_CHECK(cudaMemcpyPeerAsync(destination, owner->ordinal, source, accessor->ordinal, size, accessor->stream));
    } else if (way == WAY_COPY) {
        P2P_CHECK(cudaMemcpyAsync(destination, source, size, cudaMemcpyDefault, accessor->stream));
    } else {
        p2p_copy_kernel<<<accessor->blocks, P2P_THREADS, 0, accessor->stream>>>(destination, source, size);
        P2P_CHECK(cudaGetLastError());
    }
    wait_or_fail(accessor, accessor_index, owner_index, way_names[way]);
    P2P_CHECK(cudaSetDevice(result_card->ordinal));
    P2P_CHECK(cudaMemcpyAsync(staging, result_region + window_begin, window_bytes, cudaMemcpyDefault,
                              result_card->stream));
    wait_or_fail(result_card, accessor_index, owner_index, "downloading the result");

    uint64_t first_wrong = 0, first_guard = 0, unused = 0;
    uint64_t wrong = p2p_count_wrong_bytes(staging + test_in_window, size, seed, 0, &first_wrong);
    uint64_t guard_changed =
        p2p_count_wrong_bytes(staging, test_in_window, guard_seed, 0, &first_guard) +
        p2p_count_wrong_bytes(staging + test_in_window + size, window_bytes - test_in_window - size, guard_seed,
                              test_in_window + size, &unused);
    const char *verdict = judge(accessor_index, owner_index, designated, wrong,
                                wrong == 0 || first_wrong == DAMAGED_BYTE, guard_changed != 0, way_names[way]);
    char first_text[P2P_VERDICT_BYTES] = "na";
    if (wrong != 0) {
        snprintf(first_text, sizeof(first_text), "%" PRIu64, first_wrong);
    }
    uint8_t *remote_base = stand_in ? owner->stand_in : owner->arena;
    printf("CHECK pair=%s>%s way=%s size=%" PRIu64 " placement=%s remote_offset_mib=%.3f seed=%016" PRIx64
           " wrong_bytes=%" PRIu64 " first_wrong=%s guard_changed=%" PRIu64 " verdict=%s\n", accessor->bus, owner->bus,
           way_names[way], size, placement_names[placement],
           (double)(uint64_t)(remote + remote_begin - remote_base) / (double)BYTES_PER_MIB, seed, wrong, first_text,
           guard_changed, verdict);
    pairs[accessor_index][owner_index].checks++;
    pairs[accessor_index][owner_index].bytes_checked += size;
}

/* ---- the engine's message shapes: pushes with acknowledgements ---- */

static void run_shapes(int accessor_index, int owner_index, uint32_t rounds)
{
    struct gate_card *accessor = &cards[accessor_index], *owner = &cards[owner_index];
    struct p2p_push_buffers buffers;
    buffers.flags = (uint32_t *)region_of(owner, MAILBOX);
    buffers.payload = (uint32_t *)(region_of(owner, MAILBOX) + FLAG_AREA_BYTES);
    buffers.acknowledgements = acknowledgement_page;
    buffers.sender_status = (struct p2p_block_status *)region_of(accessor, STATUS);
    buffers.receiver_status = (struct p2p_block_status *)region_of(owner, STATUS) + P2P_MAX_BLOCKS;
    for (size_t shape = 0; shape < SHAPE_COUNT; shape++) {
        for (size_t block_case = 0; block_case < SHAPE_BLOCK_COUNT; block_case++) {
            uint32_t words = shape_bytes[shape] / (uint32_t)sizeof(uint32_t), seed = (uint32_t)next_seed();
            struct p2p_damage damage = {0, 0u, 0u};
            damage.active = damaged_kind == DAMAGE_PUSH && is_first_pair(accessor_index, owner_index) &&
                            shape_bytes[shape] == DAMAGED_SHAPE_BYTES &&
                            shape_blocks[block_case] == DAMAGED_SHAPE_BLOCKS;
            damage.round = (rounds + 1u) / 2u;
            damage.index = DAMAGED_PUSH_WORD;
            struct p2p_outcome outcome;
            if (!p2p_run_push(accessor->ordinal, accessor->stream, owner->ordinal, owner->stream, &buffers, words,
                              shape_blocks[block_case], rounds, seed, &damage, &outcome)) {
                timed_out(accessor_index, owner_index, "a push");
            }
            const char *verdict = judge(accessor_index, owner_index, damage.active, outcome.wrong_words,
                                        outcome.first_wrong_word == DAMAGED_PUSH_WORD,
                                        outcome.timed_out != 0 || outcome.rounds_completed != rounds, "push");
            printf("SHAPE pair=%s>%s bytes=%u blocks=%u rounds=%u seed=%08x us_per_round=%.2f completed=%u "
                   "timed_out_blocks=%u wrong_words=%u first_wrong_word=%d verdict=%s\n", accessor->bus, owner->bus,
                   shape_bytes[shape], shape_blocks[block_case], rounds, seed, outcome.microseconds_per_round,
                   outcome.rounds_completed, outcome.timed_out, outcome.wrong_words,
                   outcome.first_wrong_word == P2P_NO_WORD ? -1 : (int)outcome.first_wrong_word, verdict);
            pairs[accessor_index][owner_index].checks++;
        }
    }
}

/* ---- the soak: every tested pair at once, both directions ---- */

struct soak_pair {
    int accessor, owner;
    uint8_t *source, *destination;
    uint32_t *counts, *host_counts;
    cudaStream_t accessor_stream, owner_stream;
    cudaEvent_t filled, moved;
    uint64_t seed, wrong_words;
    unsigned int lines;
};

static int slot_of(int card, int other)
{
    return other < card ? other : other - 1;
}

static void run_soak(double seconds)
{
    static struct soak_pair soak[P2P_MAX_CARDS * P2P_MAX_CARDS];
    int count = 0;
    for (int accessor = 0; accessor < card_count; accessor++) {
        for (int owner = 0; owner < card_count; owner++) {
            if (accessor == owner || !pairs[accessor][owner].tested) {
                continue;
            }
            struct soak_pair *pair = &soak[count++];
            memset(pair, 0, sizeof(*pair));
            pair->accessor = accessor;
            pair->owner = owner;
            uint64_t accessor_slot = (uint64_t)slot_of(accessor, owner);
            uint64_t owner_slot = (uint64_t)slot_of(owner, accessor);
            pair->source = region_of(&cards[accessor], SOAK_SOURCE) + accessor_slot * SOAK_BYTES;
            pair->destination = region_of(&cards[owner], SOAK_DESTINATION) + owner_slot * SOAK_BYTES;
            pair->counts = (uint32_t *)region_of(&cards[owner], COUNTS) +
                           owner_slot * MOST_GRID_BLOCKS * P2P_COUNT_WORDS_PER_BLOCK;
            P2P_CHECK(cudaHostAlloc((void **)&pair->host_counts,
                                    MOST_GRID_BLOCKS * P2P_COUNT_WORDS_PER_BLOCK * sizeof(uint32_t),
                                    cudaHostAllocPortable));
            P2P_CHECK(cudaSetDevice(cards[accessor].ordinal));
            P2P_CHECK(cudaStreamCreateWithFlags(&pair->accessor_stream, cudaStreamNonBlocking));
            P2P_CHECK(cudaEventCreateWithFlags(&pair->moved, cudaEventDisableTiming));
            P2P_CHECK(cudaSetDevice(cards[owner].ordinal));
            P2P_CHECK(cudaStreamCreateWithFlags(&pair->owner_stream, cudaStreamNonBlocking));
            P2P_CHECK(cudaEventCreateWithFlags(&pair->filled, cudaEventDisableTiming));
        }
    }
    double started = p2p_seconds_now();
    uint64_t iteration = 0;
    for (; p2p_seconds_now() - started < seconds; iteration++) {
        for (int at = 0; at < count; at++) {
            struct soak_pair *pair = &soak[at];
            struct gate_card *accessor = &cards[pair->accessor], *owner = &cards[pair->owner];
            pair->seed = next_seed();
            P2P_CHECK(cudaSetDevice(accessor->ordinal));
            p2p_fill_kernel<<<accessor->blocks, P2P_THREADS, 0, pair->accessor_stream>>>(
                pair->source, SOAK_BYTES, 0, SOAK_BYTES, pair->seed, 0u, 0u);
            P2P_CHECK(cudaGetLastError());
            if (damaged_kind == DAMAGE_SOAK && at == 0 && iteration == 0) {
                uint8_t damaged = (uint8_t)~p2p_pattern_byte(pair->seed, DAMAGED_SOAK_BYTE);
                if (!p2p_wait_stream(accessor->ordinal, pair->accessor_stream, CHECK_DEADLINE_SECONDS)) {
                    timed_out(pair->accessor, pair->owner, "the soak's fill");
                }
                P2P_CHECK(cudaMemcpy(pair->source + DAMAGED_SOAK_BYTE, &damaged, 1, cudaMemcpyDefault));
            }
            P2P_CHECK(cudaSetDevice(owner->ordinal));
            p2p_fill_kernel<<<owner->blocks, P2P_THREADS, 0, pair->owner_stream>>>(
                pair->destination, SOAK_BYTES, 0, SOAK_BYTES, pair->seed, 0u, 1u);
            P2P_CHECK(cudaGetLastError());
            P2P_CHECK(cudaEventRecord(pair->filled, pair->owner_stream));
            P2P_CHECK(cudaSetDevice(accessor->ordinal));
            P2P_CHECK(cudaStreamWaitEvent(pair->accessor_stream, pair->filled, 0));
            if (iteration % 2u == 0u && !stand_in) {
                P2P_CHECK(cudaMemcpyPeerAsync(pair->destination, owner->ordinal, pair->source, accessor->ordinal,
                                              SOAK_BYTES, pair->accessor_stream));
            } else if (iteration % 2u == 0u) {
                P2P_CHECK(cudaMemcpyAsync(pair->destination, pair->source, SOAK_BYTES, cudaMemcpyDefault,
                                          pair->accessor_stream));
            } else {
                p2p_copy_kernel<<<accessor->blocks, P2P_THREADS, 0, pair->accessor_stream>>>(pair->destination,
                                                                                           pair->source, SOAK_BYTES);
                P2P_CHECK(cudaGetLastError());
            }
            P2P_CHECK(cudaEventRecord(pair->moved, pair->accessor_stream));
            P2P_CHECK(cudaSetDevice(owner->ordinal));
            P2P_CHECK(cudaStreamWaitEvent(pair->owner_stream, pair->moved, 0));
            p2p_count_kernel<<<owner->blocks, P2P_THREADS, 0, pair->owner_stream>>>(
                (const uint64_t *)pair->destination, SOAK_BYTES / sizeof(uint64_t), pair->seed, pair->counts);
            P2P_CHECK(cudaGetLastError());
            P2P_CHECK(cudaMemcpyAsync(pair->host_counts, pair->counts,
                                      owner->blocks * P2P_COUNT_WORDS_PER_BLOCK * sizeof(uint32_t),
                                      cudaMemcpyDeviceToHost, pair->owner_stream));
        }
        for (int at = 0; at < count; at++) {
            struct soak_pair *pair = &soak[at];
            if (!p2p_wait_stream(cards[pair->accessor].ordinal, pair->accessor_stream, CHECK_DEADLINE_SECONDS) ||
                !p2p_wait_stream(cards[pair->owner].ordinal, pair->owner_stream, CHECK_DEADLINE_SECONDS)) {
                timed_out(pair->accessor, pair->owner, "the soak");
            }
            uint64_t wrong = 0;
            uint32_t first_wrong = P2P_NO_WORD;
            for (unsigned int block = 0; block < cards[pair->owner].blocks; block++) {
                wrong += pair->host_counts[P2P_COUNT_WORDS_PER_BLOCK * block];
                uint32_t block_first = pair->host_counts[P2P_COUNT_WORDS_PER_BLOCK * block + 1u];
                first_wrong = block_first < first_wrong ? block_first : first_wrong;
            }
            int designated = damaged_kind == DAMAGE_SOAK && at == 0 && iteration == 0;
            pair->wrong_words += wrong;
            pairs[pair->accessor][pair->owner].soak_iterations++;
            pairs[pair->accessor][pair->owner].bytes_checked += SOAK_BYTES;
            if (designated || wrong != 0) {
                const char *verdict = judge(pair->accessor, pair->owner, designated, wrong,
                                            first_wrong == DAMAGED_SOAK_BYTE / sizeof(uint64_t), 0, "soak");
                if (pair->lines++ < MOST_SOAK_LINES_PER_PAIR) {
                    printf("SOAK_ITERATION pair=%s>%s iteration=%" PRIu64 " way=%s seed=%016" PRIx64
                           " wrong_words=%" PRIu64 " first_wrong_word=%lld verdict=%s\n", cards[pair->accessor].bus,
                           cards[pair->owner].bus, iteration, iteration % 2u == 0u ? "copy" : "store", pair->seed,
                           wrong, first_wrong == P2P_NO_WORD ? -1ll : (long long)first_wrong, verdict);
                }
            }
        }
    }
    double elapsed = p2p_seconds_now() - started;
    for (int at = 0; at < count; at++) {
        struct soak_pair *pair = &soak[at];
        printf("SOAK pair=%s>%s seconds=%.1f iterations=%" PRIu64 " gib_moved=%.1f wrong_words=%" PRIu64 "\n",
               cards[pair->accessor].bus, cards[pair->owner].bus, elapsed, iteration,
               (double)(iteration * SOAK_BYTES) / (double)(BYTES_PER_MIB * BYTES_PER_KIB), pair->wrong_words);
        P2P_CHECK(cudaSetDevice(cards[pair->accessor].ordinal));
        P2P_CHECK(cudaEventDestroy(pair->moved));
        P2P_CHECK(cudaStreamDestroy(pair->accessor_stream));
        P2P_CHECK(cudaSetDevice(cards[pair->owner].ordinal));
        P2P_CHECK(cudaEventDestroy(pair->filled));
        P2P_CHECK(cudaStreamDestroy(pair->owner_stream));
        P2P_CHECK(cudaFreeHost(pair->host_counts));
    }
}

/* ---- the AER counters of the cards and of the ports above them ---- */

/* After the checks: an uncorrectable error added on any card or port, or a counter that cannot be read, refuses every
 * pair (a misdirected peer write may land on a card outside the pair); correctable ones are printed. */
static void judge_aer(void)
{
    int refuse = 0;
    for (int card = 0; card < card_count; card++) {
        const struct p2p_aer *before = &cards[card].aer_before;
        struct p2p_aer after;
        if (!cards[card].aer_readable || !p2p_aer_read(cards[card].bus, &after) || after.has_port != before->has_port) {
            printf("AER card=%s unreadable\n", cards[card].bus);
            refuse = 1;
            continue;
        }
        unsigned long long added[P2P_AER_PLACES][P2P_AER_KINDS];
        for (int place = 0; place < P2P_AER_PLACES; place++) {
            for (int kind = 0; kind < P2P_AER_KINDS; kind++) {
                added[place][kind] = after.totals[place][kind] - before->totals[place][kind];   /* a fall wraps */
            }
            refuse |= added[place][P2P_AER_NONFATAL] != 0 || added[place][P2P_AER_FATAL] != 0;
        }
        printf("AER card=%s port=%s added correctable,nonfatal,fatal: card=%llu,%llu,%llu port=%llu,%llu,%llu\n",
               cards[card].bus, after.has_port ? after.port : "none", added[P2P_AER_CARD][P2P_AER_CORRECTABLE],
               added[P2P_AER_CARD][P2P_AER_NONFATAL], added[P2P_AER_CARD][P2P_AER_FATAL],
               added[P2P_AER_PORT][P2P_AER_CORRECTABLE], added[P2P_AER_PORT][P2P_AER_NONFATAL],
               added[P2P_AER_PORT][P2P_AER_FATAL]);
    }
    if (!refuse) {
        return;
    }
    other_failures++;
    for (int accessor = 0; accessor < card_count; accessor++) {
        for (int owner = 0; owner < card_count; owner++) {
            if (pairs[accessor][owner].tested) {
                fail_pair(accessor, owner, "an uncorrectable AER error or an unreadable counter, card or port");
            }
        }
    }
}

/* ---- the driver's answers, and peer access enabled on every offered pair at once ---- */

static int peer_attribute(enum cudaDeviceP2PAttr attribute, int accessor, int owner)
{
    int value = 0;
    if (cudaDeviceGetP2PAttribute(&value, attribute, accessor, owner) != cudaSuccess) {
        (void)cudaGetLastError();
        return -1;
    }
    return value;
}

/* The driver's answer for every ordered pair. A pair is tested if the driver offers it between two cards of one
 * model (every pair with --stand-in-host); the first tested pair is the one a damaged run damages. Returns the pairs
 * offered between two cards of one model. */
static int ask_driver(void)
{
    int offered = 0, within_models = 0;
    for (int accessor = 0; accessor < card_count; accessor++) {
        for (int owner = 0; owner < card_count; owner++) {
            struct pair_state *pair = &pairs[accessor][owner];
            int can_access = 0;
            if (accessor == owner) {
                continue;
            }
            P2P_CHECK(cudaDeviceCanAccessPeer(&can_access, cards[accessor].ordinal, cards[owner].ordinal));
            pair->offered = can_access != 0;
            pair->across_models = strcmp(cards[accessor].name, cards[owner].name) != 0 ||
                                  cards[accessor].capability != cards[owner].capability;
            pair->tested = stand_in || (pair->offered && !pair->across_models);
            offered += pair->offered;
            within_models += pair->offered && !pair->across_models;
            if (pair->tested && first_accessor < 0) {
                first_accessor = accessor;
                first_owner = owner;
            }
            printf("PAIR pair=%s>%s can_access=%d access=%d atomics=%d rank=%d offered=%s models=%s route=%s\n",
                   cards[accessor].bus, cards[owner].bus, can_access,
                   peer_attribute(cudaDevP2PAttrAccessSupported, cards[accessor].ordinal, cards[owner].ordinal),
                   peer_attribute(cudaDevP2PAttrNativeAtomicSupported, cards[accessor].ordinal, cards[owner].ordinal),
                   peer_attribute(cudaDevP2PAttrPerformanceRank, cards[accessor].ordinal, cards[owner].ordinal),
                   can_access ? "yes" : "no", pair->across_models ? "two" : "one",
                   !pair->offered ? "host_pages" : pair->across_models ? "host_pages_offered_unverified"
                                                                       : "unverified_until_the_gate_passes");
        }
    }
    printf("peer access offered on %d pairs, of %d ordered pairs; %d of them between two cards of one model, the "
           "pairs the gate tests\n", offered, card_count * (card_count - 1), within_models);
    return within_models;
}

static void enable_offered(void)
{
    for (int accessor = 0; accessor < card_count; accessor++) {
        for (int owner = 0; owner < card_count; owner++) {
            if (!pairs[accessor][owner].tested) {
                continue;
            }
            P2P_CHECK(cudaSetDevice(cards[accessor].ordinal));
            cudaError_t status = cudaDeviceEnablePeerAccess(cards[owner].ordinal, 0);
            if (status == cudaErrorPeerAccessAlreadyEnabled) {
                (void)cudaGetLastError();
                status = cudaSuccess;
            }
            printf("ENABLE pair=%s>%s %s\n", cards[accessor].bus, cards[owner].bus,
                   status == cudaSuccess ? "ok" : cudaGetErrorString(status));
            if (status != cudaSuccess) {
                (void)cudaGetLastError();
                other_failures++;
                fail_pair(accessor, owner, "peer access offered, but enabling it failed");
                pairs[accessor][owner].tested = 0;               /* nothing runs through a pair that is not enabled */
            }
        }
    }
}

/* ---- the command line, the cards, the verdict ---- */

static int read_number(const char *text, unsigned long long least, unsigned long long most, int base,
                       unsigned long long *value)
{
    char *end = NULL;
    if (text == NULL || *text == '\0' || *text == '-') {
        return 0;
    }
    errno = 0;
    *value = strtoull(text, &end, base);
    return errno == 0 && *end == '\0' && *value >= least && *value <= most;
}

static void usage(const char *program)
{
    fprintf(stderr, "usage: %s [--record PATH [--rerun]] [--soak-seconds 0..%llu] [--rounds 1..%llu] [--seed HEX] "
                    "[--stand-in-host [--corrupt-byte copy|store|load|push|soak]] [--arena-mib N] [--skip-shapes]\n",
            program, MOST_SOAK_SECONDS, MOST_ROUNDS);
    exit(EXIT_SETUP);
}

static void open_card(struct gate_card *card, int ordinal)
{
    cudaDeviceProp properties;
    memset(card, 0, sizeof(*card));
    card->ordinal = ordinal;
    P2P_CHECK(cudaSetDevice(ordinal));
    P2P_CHECK(cudaGetDeviceProperties(&properties, ordinal));
    snprintf(card->name, sizeof(card->name), "%s", properties.name);
    p2p_bus_text(ordinal, card->bus);
    card->capability = properties.major * CAPABILITY_PER_MAJOR + properties.minor;
    card->sms = properties.multiProcessorCount;
    unsigned int blocks = (unsigned int)properties.multiProcessorCount * BLOCKS_PER_SM;
    card->blocks = blocks < MOST_GRID_BLOCKS ? blocks : MOST_GRID_BLOCKS;
    if (card->capability != CAPABILITY_SM_80 && card->capability != CAPABILITY_SM_120) {
        fprintf(stderr, "card %s has capability %d.%d: this build holds code for 8.0 and 12.0 only\n", card->bus,
                properties.major, properties.minor);
        exit(EXIT_SETUP);
    }
    if (!properties.unifiedAddressing || !properties.canMapHostMemory) {
        fprintf(stderr, "card %s cannot map host memory into one address space\n", card->bus);
        exit(EXIT_SETUP);
    }
    P2P_CHECK(cudaStreamCreateWithFlags(&card->stream, cudaStreamNonBlocking));
    P2P_CHECK(cudaMemGetInfo(&card->free_bytes, &card->total_bytes));
    card->aer_readable = p2p_aer_read(card->bus, &card->aer_before);
    printf("CARD card=%s name=\"%s\" capability=%d.%d sms=%d total_mib=%llu free_mib=%llu aer=%s port=%s\n",
           card->bus, card->name, properties.major, properties.minor, card->sms,
           (unsigned long long)(card->total_bytes / BYTES_PER_MIB),
           (unsigned long long)(card->free_bytes / BYTES_PER_MIB), card->aer_readable ? "readable" : "unreadable",
           card->aer_before.has_port ? card->aer_before.port : "none");
}

/* A record is a verdict for the whole machine: every card the driver holds must be one this run sees. */
static int sees_every_card(void)
{
    char bus[P2P_BUS_TEXT_BYTES];
    int seen = 0;
    for (int card = 0; card < card_count; card++) {
        p2p_bus_text(card, bus);
        for (int held = 0; held < key.card_count; held++) {
            seen += strcmp(bus, key.buses[held]) == 0;
        }
    }
    return key.complete && card_count == key.card_count && seen == card_count;
}

/* A trusted record of this machine's key ends the run with its verdict, unless --rerun; returns otherwise. */
static void use_record_if_trusted(int rerun)
{
    struct p2p_record record;
    enum p2p_record_state state = p2p_record_read(record_path, &key, &record);
    printf("RECORD state=%s verdict=%s\n", p2p_record_state_names[state],
           state == P2P_RECORD_TRUSTED ? record.verdict : "na");
    if (state != P2P_RECORD_TRUSTED || rerun || strcmp(record.verdict, "incomplete") == 0) {
        return;
    }
    printf("CACHED verdict=%s pairs=%s: the gate ran on this boot, driver, module, options and cards\n",
           record.verdict, record.pairs[0] != '\0' ? record.pairs + 1 : "none");
    exit(strcmp(record.verdict, "refused") == 0 ? EXIT_REFUSED : EXIT_SUCCESS);
}

static int finish(void)
{
    static char passed[P2P_FILE_BYTES];
    int tested = 0, failed = 0;
    for (int accessor = 0; accessor < card_count; accessor++) {
        for (int owner = 0; owner < card_count; owner++) {
            const struct pair_state *pair = &pairs[accessor][owner];
            tested += pair->tested;
            failed += pair->failed;
            if (pair->failed) {
                printf("GATE FAILED pair=%s>%s reason=\"%s\"\n", cards[accessor].bus, cards[owner].bus, pair->reason);
            }
        }
    }
    for (int accessor = 0; accessor < card_count; accessor++) {
        for (int owner = 0; owner < card_count; owner++) {
            const struct pair_state *pair = &pairs[accessor][owner];
            if (!pair->tested || pair->failed) {
                continue;
            }
            printf("%s pair=%s>%s checks=%" PRIu64 " soak_iterations=%" PRIu64 " bytes_checked=%" PRIu64 "%s\n",
                   failed != 0 ? "REFUSED" : "PASS", cards[accessor].bus, cards[owner].bus, pair->checks,
                   pair->soak_iterations, pair->bytes_checked, failed != 0 ? ": its own checks held" : "");
            size_t length = strlen(passed);
            snprintf(passed + length, sizeof(passed) - length, " %s>%s", cards[accessor].bus, cards[owner].bus);
        }
    }
    if (failed != 0) {
        printf("GATE refused: %d pairs failed, and one failure refuses peer access on every pair\n", failed);
        printf("VERDICT gate=refused failed_pairs=%d tested_pairs=%d\n", failed, tested);
        write_record("refused", "");
    } else {
        printf("GATE passed: %d pairs\n", tested);
        printf("VERDICT gate=passed pairs=%d\n", tested);
        write_record("passed", passed);
    }
    if (damaged_kind >= 0) {
        int as_required = designated_reported == 1 && other_failures == 0;
        printf("SHOWN_FAILING kind=%s reported=%u other_failures=%u verdict=%s\n", damage_names[damaged_kind],
               designated_reported, other_failures, as_required ? "as_required" : "NOT_AS_REQUIRED");
        return as_required ? EXIT_SUCCESS : EXIT_REFUSED;
    }
    return failed != 0 ? EXIT_REFUSED : EXIT_SUCCESS;
}

int main(int argument_count, char **arguments)
{
    unsigned long long soak_seconds = DEFAULT_SOAK_SECONDS, rounds = DEFAULT_ROUNDS, number = 0;
    int rerun = 0, seed_given = 0;
    for (int at = 1; at < argument_count; at++) {
        const char *argument = arguments[at], *value = at + 1 < argument_count ? arguments[at + 1] : NULL;
        if (strcmp(argument, "--record") == 0 && value != NULL && strlen(value) < P2P_PATH_BYTES / 2) {
            record_path = value;
            at++;
        } else if (strcmp(argument, "--rerun") == 0) {
            rerun = 1;
        } else if (strcmp(argument, "--soak-seconds") == 0 && read_number(value, 0, MOST_SOAK_SECONDS, P2P_DECIMAL_BASE,
                                                                               &soak_seconds)) {
            at++;
        } else if (strcmp(argument, "--rounds") == 0 && read_number(value, 1, MOST_ROUNDS, P2P_DECIMAL_BASE, &rounds)) {
            at++;
        } else if (strcmp(argument, "--seed") == 0 && read_number(value, 0, UINT64_MAX, P2P_HEX_BASE, &number)) {
            run_seed = number;
            seed_given = 1;
            at++;
        } else if (strcmp(argument, "--arena-mib") == 0 &&
                   read_number(value, 1, MOST_ARENA_MIB, P2P_DECIMAL_BASE, &number)) {
            arena_cap_bytes = number * BYTES_PER_MIB;
            at++;
        } else if (strcmp(argument, "--stand-in-host") == 0) {
            stand_in = 1;
        } else if (strcmp(argument, "--skip-shapes") == 0) {
            skip_shapes = 1;
        } else if (strcmp(argument, "--corrupt-byte") == 0 && value != NULL) {
            for (int kind = 0; kind < DAMAGE_KINDS; kind++) {
                damaged_kind = strcmp(value, damage_names[kind]) == 0 ? kind : damaged_kind;
            }
            if (damaged_kind < 0) {
                usage(arguments[0]);
            }
            at++;
        } else {
            usage(arguments[0]);
        }
    }
    if ((rerun && record_path == NULL) || (stand_in && record_path != NULL) || (damaged_kind >= 0 && !stand_in) ||
        (damaged_kind == DAMAGE_PUSH && skip_shapes) || (damaged_kind == DAMAGE_SOAK && soak_seconds == 0) ||
        (record_path != NULL && (skip_shapes || soak_seconds < DEFAULT_SOAK_SECONDS || rounds < DEFAULT_ROUNDS ||
                                 arena_cap_bytes != 0 || seed_given))) {
        fprintf(stderr, "--rerun needs --record; a stand-in run writes no record; --corrupt-byte needs "
                        "--stand-in-host and the part it damages; --record takes the whole gate only\n");
        usage(arguments[0]);
    }
    run_seed = seed_given ? run_seed : p2p_random_seed();
    int runtime_version = 0, driver_version = 0;
    P2P_CHECK(cudaRuntimeGetVersion(&runtime_version));
    P2P_CHECK(cudaDriverGetVersion(&driver_version));
    P2P_CHECK(cudaGetDeviceCount(&card_count));
    printf("GATE version=%u mode=%s runtime=%d driver_interface=%d cards=%d seed=%016" PRIx64 " soak_seconds=%llu "
           "rounds=%llu corrupt_byte=%s\n", P2P_GATE_VERSION, stand_in ? "stand_in_host" : "peer", runtime_version,
           driver_version, card_count, run_seed, soak_seconds, rounds,
           damaged_kind >= 0 ? damage_names[damaged_kind] : "none");
    if (card_count < 2 || card_count > P2P_MAX_CARDS) {
        fprintf(stderr, "needs 2 to %d cards, sees %d\n", P2P_MAX_CARDS, card_count);
        return EXIT_SETUP;
    }
    p2p_machine_key(&key);
    p2p_key_print(&key);
    if (record_path != NULL && !sees_every_card()) {
        fprintf(stderr, "--record needs a complete key and every card the driver holds visible to this run\n");
        return EXIT_SETUP;
    }
    if (record_path != NULL) {
        use_record_if_trusted(rerun);
    }
    for (int card = 0; card < card_count; card++) {
        open_card(&cards[card], card);
    }
    int offered_within_models = ask_driver();
    if (stand_in) {
        printf("STAND_IN every ordered pair is checked through page-locked host memory standing in for the owner's "
               "memory; nothing is enabled and no record is written\n");
    } else if (offered_within_models == 0) {
        printf("VERDICT gate=none_offered: no pair is offered between two cards of one model; every pair stays on "
               "host pages\n");
        write_record("none_offered", "");
        return EXIT_SUCCESS;
    } else {
        write_record("incomplete", "");          /* a run that dies from here on passes nothing */
        enable_offered();                        /* before the arenas: their peer mappings come out of the margin */
    }
    for (int card = 0; card < card_count; card++) {
        int in_a_tested_pair = 0;
        for (int other = 0; other < card_count; other++) {
            in_a_tested_pair |= pairs[card][other].tested | pairs[other][card].tested;
        }
        if (in_a_tested_pair) {
            open_arena(&cards[card]);
        }
    }
    P2P_CHECK(cudaHostAlloc((void **)&staging, CHECK_REGION_BYTES, cudaHostAllocPortable));
    P2P_CHECK(cudaHostAlloc((void **)&acknowledgement_page, FLAG_AREA_BYTES,
                            cudaHostAllocMapped | cudaHostAllocPortable));
    check_one_address(acknowledgement_page);
    for (int card = 0; card < card_count; card++) {
        if (cards[card].stand_in != NULL) {
            check_one_address(cards[card].stand_in);
        }
    }
    for (int accessor = 0; accessor < card_count; accessor++) {
        for (int owner = 0; owner < card_count; owner++) {
            if (!pairs[accessor][owner].tested) {
                continue;
            }
            for (int way = 0; way < WAYS; way++) {
                for (int placement = 0; placement < PLACEMENTS; placement++) {
                    for (size_t size = 0; size < CHECK_SIZE_COUNT; size++) {
                        run_check(accessor, owner, (enum way)way, check_sizes[size], (enum placement)placement);
                    }
                }
            }
            if (!skip_shapes) {
                run_shapes(accessor, owner, (uint32_t)rounds);
            }
            fflush(stdout);
        }
    }
    if (soak_seconds != 0) {
        run_soak((double)soak_seconds);
    }
    for (int card = 0; card < card_count; card++) {            /* a card's late error ends the run before a verdict */
        P2P_CHECK(cudaSetDevice(cards[card].ordinal));
        P2P_CHECK(cudaDeviceSynchronize());
    }
    judge_aer();
    int status = finish();
    fflush(stdout);
    for (int card = 0; card < card_count; card++) {
        P2P_CHECK(cudaSetDevice(cards[card].ordinal));
        if (cards[card].arena != NULL) {
            P2P_CHECK(cudaFree(cards[card].arena));
        }
        if (cards[card].stand_in != NULL) {
            P2P_CHECK(cudaFreeHost(cards[card].stand_in));
        }
        P2P_CHECK(cudaStreamDestroy(cards[card].stream));
    }
    P2P_CHECK(cudaFreeHost(acknowledgement_page));
    P2P_CHECK(cudaFreeHost(staging));
    return status;
}
