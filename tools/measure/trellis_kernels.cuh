/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * trellis_kernels.cuh: the routed experts of one MoE layer with EXL3 trellis weights (and, through
 * exlr8_kernels.cuh, exlr8 v2 ones), written for the engine: one product body for every width K, codebook, operand
 * form and stage, each lever a compile-time constant, a kernel stamped per choice (TR_STAMP). Round 1, package 7 of
 * the speed campaign (research/speed/ROUND_1.md); trellis_expert_probe.cu times the stamps; trellis_reference.h
 * has the format's definition.
 *
 * A STEP is five launches (struct tr_step is what they read and write; y = (((x .* suh) Hb) T) Hb .* svh):
 *   tr_rotate_input_kernel   per pair (a row at one of its experts) and matrix (gate, up): (x .* suh) Hb, made the
 *                            operand the product's form wants (the FEED)
 *   product, gate and up     T by the fed vectors, a job per expert and matrix, the job's rows as columns
 *   tr_rotate_middle_kernel  per pair: the two sums through Hb and .* svh, the hidden step, .* suh of down, Hb, feed
 *   product, down            a job per expert
 *   tr_rotate_output_kernel  per row: its six pairs' sums through Hb and .* svh, added in slot order from +0
 * THE PRODUCT. The file's tile is already the instruction's operand A: lane 4 g + t of a warp owns stream weights
 * 8 lane .. 8 lane + 7, which are outputs g and g + 8 at inputs 2 t, 2 t + 1, 2 t + 8, 2 t + 9: ONE instruction
 * m16n8k16 per tile, the rows of a job the columns of operand B (an expert is decoded once for all its rows).
 * A card holds a matrix as [output tile][input tile][8 K words]: a warp walks one contiguous stream; a tile's
 * bytes are the file's. A lane's eight states lie in two consecutive words of the tile (the second wrapping to
 * word 0): the window. The forms (TR_FORM_*) say what the eight states become:
 *   MUL1_FP16      the format's values, two to a register: one m16n8k16 f16 per tile, fp32 sums running on
 *   MUL1_IMMA8     the four bytes of state * 0x83DCD12D as one register of the unsigned operand of m16n8k32 u8 by
 *                  s8, the row's 8-bit integer in four equal bytes: two instructions per tile, the sum of a block
 *                  of 128 inputs an exact integer P; then T = fma(P, a, T - c), a and c the fed block's two floats
 *                  (a = q * 1774 / 2^18, c = q * 905216 / 2^18 * the block's sum of integers, q its scale): the
 *                  weight is the format's before its fp16 rounding
 *   MCG_FP16, EXLR8_ROUND, EXLR8_CVT (sm_120a)   as MUL1_FP16 with the other codebook and exlr8's rounding
 *   EXLR8_TABLE    the E4M3 byte from a 64 KiB table by state in the block's shared memory, placed as fp16 times
 *                  2^-8; such blocks are few, of several teams (at most 1,024 threads: these kernels' launch bound)
 *   EXLR8_FP8 (sm_120a)   the E4M3 bytes as bytes: one m16n8k32 e4m3 per two tiles, rows as E4M3 bytes with one
 *                  power of two per block of 128: T = fma(sum of the block, a, T)
 *   NO_DECODE      the window's two words as the operand: what a step costs with no decode at all (never checked)
 * A warp is one slice of the input width (slice s of S holds blocks s B / S .. (s + 1) B / S - 1 of the B blocks
 * of 128 inputs); S is the warps of a team (blockDim.x / 32): the host chooses it, and it is part of a float form's
 * contract. The slices' sums are added in slice order from +0 by the team's warp 0 after the block's barrier.
 * Whether a row keeps its bits in any company is measured per card by the probe, not argued (unit 3a: a CMP
 * 170HX's fp16 sums follow no simple rule where they round).
 * The kernels check nothing: the host holds every job, pair, grid and array to its range before a launch (the
 * probe's trx_step_is_valid). No atomics. Device code and its launch stubs. Measurement tool: not part of the
 * engine.
 */
#ifndef TRELLIS_KERNELS_CUH
#define TRELLIS_KERNELS_CUH
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include "exlr8_kernels.cuh"

enum {
    /* the model and a step: a pair is a row at one of its experts (row * 6 + slot) */
    TR_HIDDEN_SIZE = 5120, TR_INNER_SIZE = 2304, TR_EXPERTS_PER_ROW = 6, TR_MOST_ROWS = 8, TR_EXPERT_COUNT = 128,
    TR_MOST_PAIRS = TR_MOST_ROWS * TR_EXPERTS_PER_ROW, TR_MATRIX_KINDS = 3, TR_GATE = 0, TR_UP = 1, TR_DOWN = 2,
    TR_GATE_UP_MATRICES = 2, TR_GATE_UP_STAGE = 0, TR_DOWN_STAGE = 1, TR_STAGES = 2,
    TR_MOST_GATE_UP_JOBS = TR_MOST_PAIRS * TR_GATE_UP_MATRICES, TR_MOST_JOBS = TR_MOST_GATE_UP_JOBS + TR_MOST_PAIRS,
    /* a tile: 16 by 16 weights of K bits in 8 K words; a state is 16 bits of its stream */
    TR_TILE = 16, TR_TILE_WEIGHTS = 256, TR_WORDS_PER_BIT = 8, TR_STATE_BITS = 16, TR_STATE_MASK = 0xFFFF,
    TR_WORD_BITS = 32, TR_LEAST_BITS = 2, TR_WIDTHS = 3,
    TR_MATRIX_TILES = (TR_HIDDEN_SIZE / TR_TILE) * (TR_INNER_SIZE / TR_TILE),
    /* a lane of the instruction: four registers of weights, four sums (outputs g, g + 8 by columns 2 t, 2 t + 1) */
    TR_LANES_PER_WARP = 32, TR_LANES_PER_GROUP = 4, TR_LANE_WEIGHTS = 8, TR_HALF_WEIGHTS = 4, TR_HALF_OUTPUTS = 8,
    TR_HALF_INPUTS = 8, TR_WEIGHT_REGISTERS = 4, TR_REGISTER_WEIGHTS = 2, TR_LANE_SUMS = 4, TR_LANE_COLUMNS = 2,
    TR_LANE_OUTPUTS = 2, TR_HALF_BITS = 16, TR_BYTE_MASK = 0xFF,
    /* the rotations: a block of 128 values is 8 tiles of inputs and four warps of threads */
    TR_HADAMARD = 128, TR_HALF_HADAMARD = 64, TR_BLOCK_TILES = 8, TR_BLOCK_WARPS = 4, TR_MIDDLE_ROTATIONS = 3,
    TR_BLOCK_SCALES = 2, TR_FP8_TILES = 2,       /* a block's a and c; the tiles of one m16n8k32 of bytes */
    TR_REDUCE_SCRATCH = TR_BLOCK_SCALES * TR_BLOCK_WARPS, TR_INPUT_SCRATCH = TR_HADAMARD + TR_REDUCE_SCRATCH,
    TR_MIDDLE_SCRATCH = TR_MIDDLE_ROTATIONS * TR_HADAMARD + TR_REDUCE_SCRATCH,
    TR_OUTPUT_THREADS = TR_EXPERTS_PER_ROW * TR_HADAMARD, TR_OUTPUT_SCRATCH = TR_OUTPUT_THREADS + TR_OUTPUT_THREADS,
    /* an expert's side vectors, fp16, in this order */
    TR_SUH_GATE_AT = 0, TR_SUH_UP_AT = TR_HIDDEN_SIZE, TR_SVH_GATE_AT = TR_SUH_UP_AT + TR_HIDDEN_SIZE,
    TR_SVH_UP_AT = TR_SVH_GATE_AT + TR_INNER_SIZE, TR_SUH_DOWN_AT = TR_SVH_UP_AT + TR_INNER_SIZE,
    TR_SVH_DOWN_AT = TR_SUH_DOWN_AT + TR_INNER_SIZE, TR_SIDE_VALUES = TR_SVH_DOWN_AT + TR_HIDDEN_SIZE,
    /* what a product is stamped for; what a rotation writes for it (a feed has TR_FEED_BYTES a value at most) */
    TR_FORM_MUL1_FP16 = 0, TR_FORM_MUL1_IMMA8, TR_FORM_MCG_FP16, TR_FORM_EXLR8_ROUND, TR_FORM_EXLR8_TABLE,
    TR_FORM_EXLR8_CVT, TR_FORM_EXLR8_FP8, TR_FORM_NO_DECODE, TR_FORMS, TR_BYTE_ONES = 0x01010101,
    TR_WINDOW_SHUFFLE = 0, TR_WINDOW_LOADS, TR_FEED_FP16 = 0, TR_FEED_INT8, TR_FEED_FP8, TR_FEEDS, TR_FEED_BYTES = 4,
    TR_TABLE_LOADS = EXLR8_TABLE_BYTES / sizeof(uint4), TR_TABLE_BLOCK_THREADS = 1024, TR_MUL1_FP16_OF_1024 = 0x6400,
    TR_FP32_EXPONENT_SHIFT = 23, TR_FP32_MANTISSA_MASK = 0x007FFFFF, TR_INVERSE_FIELD_BASE = 254
};
#define TR_EVERY_LANE 0xFFFFFFFFu
#define TR_MUL1_MULTIPLIER 0x83DCD12Du
#define TR_MUL1_STEP_PAIR 0x1EEE1EEEu            /* fp16 1774 / 2^18 in both halves */
#define TR_MUL1_OFFSET_PAIR 0xC931C931u          /* fp16 -1329 / 2^7 in both halves */
#define TR_MUL1_STEP 0.00676727294921875f        /* 1774 / 2^18: a mul1 value is this times its byte sum, less */
#define TR_MUL1_OFFSET 3.453125f                 /* 905216 / 2^18 */
#define TR_INT8_LARGEST 127.0f
#define TR_E4M3_LARGEST_INVERSE (1.0f / 448.0f)
#define TR_FEED_FLOOR 1.0e-4f                    /* the least a block's largest magnitude counts as, for E4M3 rows */
#define TR_HADAMARD_SCALE 0.08838834764831845f   /* float32(1 / sqrt(128)) */
#define TR_SWIGLU_LIMIT 10.0f
#define TR_STAGE_INPUT_SIZE(stage) ((stage) == TR_DOWN_STAGE ? TR_INNER_SIZE : TR_HIDDEN_SIZE)
#define TR_STAGE_OUTPUT_SIZE(stage) ((stage) == TR_DOWN_STAGE ? TR_HIDDEN_SIZE : TR_INNER_SIZE)
#define TR_STAGE_ITEMS(stage) (TR_STAGE_OUTPUT_SIZE(stage) / TR_TILE)        /* of one job: its output tiles */
#define TR_FORM_FEED(form) \
    ((form) == TR_FORM_MUL1_IMMA8 ? TR_FEED_INT8 : (form) == TR_FORM_EXLR8_FP8 ? TR_FEED_FP8 : TR_FEED_FP16)
#define TR_FEED_UNIT(feed) ((feed) == TR_FEED_INT8 ? sizeof(uint32_t) : (feed) == TR_FEED_FP8 ? 1u : sizeof(uint16_t))
#define TR_FORM_BLOCK_STEP(form) ((form) == TR_FORM_MUL1_IMMA8 || (form) == TR_FORM_EXLR8_FP8)
#define TR_FORM_SCALE(form) ((form) == TR_FORM_EXLR8_TABLE ? EXLR8_PLACED_SCALE : 1.0f)
#define TR_EXPERT_WORDS(bits) ((uint64_t)TR_MATRIX_KINDS * TR_MATRIX_TILES * TR_WORDS_PER_BIT * (bits))
#define TR_SHARED_BYTES(form, threads) \
    ((threads) * TR_LANE_SUMS * sizeof(float) + ((form) == TR_FORM_EXLR8_TABLE ? (size_t)EXLR8_TABLE_BYTES : 0u))

struct tr_pair { uint32_t row, expert; float route_weight; };
struct tr_job {                                  /* one matrix of one expert with the rows routed to the expert */
    uint32_t matrix, row_count;                  /* expert * 3 + kind; the columns in use, 1 .. TR_MOST_ROWS */
    uint32_t vector[TR_MOST_ROWS];               /* per column: the vector it is fed and whose sums it writes
                                                    (gate, up: pair * 2 + kind; down: pair); unused: column 0's */
};
struct tr_step {                                 /* what the five launches of a step read and write */
    const uint32_t *stored;                      /* [expert][matrix kind][output tile][input tile][8 K words] */
    const uint16_t *sides;                       /* [expert][TR_SIDE_VALUES] */
    const uint8_t *table;                        /* exlr8: an E4M3 byte per state */
    const float *inputs;                         /* [row][hidden size] */
    uint8_t *feed_input, *feed_hidden;           /* [pair * 2 + kind], [pair]: [value][TR_FEED_BYTES], see tr_feed */
    float *scales_input, *scales_hidden;         /* of the same vectors: [block of 128][a, c] */
    float *sums_inner;                           /* [pair * 2 + kind][inner size]: the gate and up products */
    float *hidden;                               /* [pair][inner size] */
    float *sums_outer;                           /* [pair][hidden size]: the down products */
    float *row_output;                           /* [row][hidden size] */
    const struct tr_pair *pairs;                 /* [pair] */
    const struct tr_job *jobs;                   /* gate and up: [expert of the step][kind]; down: from
                                                    TR_MOST_GATE_UP_JOBS on */
    uint32_t item_count[TR_STAGES];              /* jobs of the stage times TR_STAGE_ITEMS: whole blocks of teams */
};
struct tr_window {                               /* where a lane's eight states lie in a tile: set once a thread */
    uint32_t first_word, second_word;            /* the two words that hold them; the second wraps to word 0 */
    uint32_t early_shift, late_shift;            /* in those 64 bits: where state 0 begins; where the 32 bits that
                                                    end with state 7 begin (0 .. 32) */
};
typedef void tr_launcher(const struct tr_step *step, uint32_t blocks, dim3 block, size_t shared_bytes);
struct tr_kernels {                              /* a stamped set: [K - 2][stage], and its kernels' names per K */
    tr_launcher *launch[TR_WIDTHS][TR_STAGES];
    const char *names[TR_WIDTHS];
};

/* ---- the instructions ---- */
/* The sum of the four bytes of a word, added to `onto`. */
__device__ __forceinline__ uint32_t tr_byte_sum(uint32_t word, uint32_t onto) {
    uint32_t sum;
    asm volatile("dp4a.u32.u32 %0, %1, %2, %3;\n" : "=r"(sum) : "r"(word), "r"((uint32_t)TR_BYTE_ONES), "r"(onto));
    return sum;
}
/* Two fp16 fused multiply-adds in one instruction: first * second + third, each half rounded once. */
__device__ __forceinline__ uint32_t tr_fp16_pair_fma(uint32_t first, uint32_t second, uint32_t third) {
    uint32_t results;
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(results) : "r"(first), "r"(second), "r"(third));
    return results;
}
/* sums += A B for the lane's part of each. weights: m16n8k16: register r is output g + 8 (r % 2) at inputs 2 t +
 * 8 (r / 2) and the next; m16n8k32: output g + 8 (r % 2) at the four bytes 4 t .. 4 t + 3 of half r / 2 of the 32.
 * first_fed, second_fed: the fed column at the same inputs of the two halves. */
#define TR_MMA(name, sum_type, spelling, result, operand)                                                            \
    __device__ __forceinline__ void name(sum_type *sums, const uint32_t *weights, uint32_t first_fed,                \
                                         uint32_t second_fed) {                                                      \
        asm volatile("mma.sync.aligned." spelling " {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"     \
                     : result(sums[0]), result(sums[1]), result(sums[2]), result(sums[3])                            \
                     : "r"(weights[0]), "r"(weights[1]), "r"(weights[2]), "r"(weights[3]), "r"(first_fed),           \
                       "r"(second_fed), operand(sums[0]), operand(sums[1]), operand(sums[2]), operand(sums[3]));     \
    }
TR_MMA(tr_mma_fp16, float, "m16n8k16.row.col.f32.f16.f16.f32", "=f", "f")
TR_MMA(tr_mma_u8_s8, int32_t, "m16n8k32.row.col.s32.u8.s8.s32", "=r", "r")
#if PROBE_SM == EXLR8_SM_120A
TR_MMA(tr_mma_e4m3, float, "m16n8k32.row.col.f32.e4m3.e4m3.f32", "=f", "f")
#endif

/* ---- a lane's window of a tile, its eight states, its registers ---- */
/* State 0 of lane L begins at stream bit (8 L + 1) K - 16, which wraps for lane 0; its eight states span 7 K + 16
 * bits: two words at every K of 2 .. 4. */
template <int bits>
__device__ __forceinline__ void tr_set_window(uint32_t lane, struct tr_window *window) {
    const uint32_t stream_bits = TR_TILE_WEIGHTS * bits;
    const uint32_t begin = (lane * TR_LANE_WEIGHTS * bits + bits + stream_bits - TR_STATE_BITS) % stream_bits;
    window->first_word = begin / TR_WORD_BITS;
    window->second_word = (window->first_word + 1u) % (TR_WORDS_PER_BIT * bits);
    window->early_shift = begin % TR_WORD_BITS;
    window->late_shift = window->early_shift + (TR_LANE_WEIGHTS - 1u) * bits - TR_STATE_BITS;
}
/* What a lane keeps of one tile until it is decoded: its own two words (TR_WINDOW_LOADS), or the one word of the
 * tile that the lane's number names (TR_WINDOW_SHUFFLE: the lanes below 8 K load the tile once between them). */
template <int bits, int window_kind>
__device__ __forceinline__ void tr_fetch(const uint32_t *tile, uint32_t lane, const struct tr_window *window,
                                         uint32_t *kept) {
    if (window_kind == TR_WINDOW_LOADS) {
        kept[0] = tile[window->first_word];
        kept[1] = tile[window->second_word];
    } else {
        kept[0] = lane < TR_WORDS_PER_BIT * bits ? tile[lane] : 0u;
    }
}
/* The eight states of a lane from the two words of its window: the 32 bits from state 0's first bit hold states
 * 0 .. 3, the 32 that end with state 7 hold states 4 .. 7 (3 K + 16 is at most 32). */
template <int bits>
__device__ __forceinline__ void tr_states(uint32_t high, uint32_t low, const struct tr_window *window,
                                          uint32_t *states) {
    const uint32_t early = __funnelshift_l(low, high, window->early_shift);
    const uint32_t late = __funnelshift_lc(low, high, window->late_shift);
    for (uint32_t weight = 0; weight < TR_HALF_WEIGHTS; weight++) {
        states[weight] = (early >> (TR_STATE_BITS - weight * bits)) & TR_STATE_MASK;
        states[weight + TR_HALF_WEIGHTS] = (late >> ((TR_HALF_WEIGHTS - 1u - weight) * bits)) & TR_STATE_MASK;
    }
}
/* Two states as one register of a float form: the lower half is low_state's weight. FP8: its two E4M3 bytes. */
template <int form>
__device__ __forceinline__ uint32_t tr_weight_pair(uint32_t low_state, uint32_t high_state, const uint8_t *table) {
    uint32_t pair = 0;
    if (form == TR_FORM_MUL1_FP16) {             /* the upper byte sum rides in the start value of the lower one */
        const uint32_t upper = tr_byte_sum(high_state * TR_MUL1_MULTIPLIER, TR_MUL1_FP16_OF_1024);
        pair = tr_byte_sum(low_state * TR_MUL1_MULTIPLIER, (upper << TR_HALF_BITS) + TR_MUL1_FP16_OF_1024);
        pair = tr_fp16_pair_fma(pair, TR_MUL1_STEP_PAIR, TR_MUL1_OFFSET_PAIR);
    } else if (form == TR_FORM_EXLR8_TABLE) {
        pair = exlr8_place_pair((uint32_t)table[low_state] | ((uint32_t)table[high_state] << TR_HALF_BITS));
    } else {
        pair = exlr8_mcg_pair(low_state, high_state);
        if (form == TR_FORM_EXLR8_ROUND) { pair = exlr8_round_pair(pair); }
#if PROBE_SM == EXLR8_SM_120A
        if (form == TR_FORM_EXLR8_CVT) { pair = exlr8_pair_of_bytes(exlr8_bytes_of_pair(pair)); }
        if (form == TR_FORM_EXLR8_FP8) { pair = exlr8_bytes_of_pair(pair); }
#endif
    }
    return pair;
}
/* A lane's registers of one tile. Stream weights 4 (r % 2) + 2 (r / 2) and the next are the two halves of register
 * r of m16n8k16 (float forms: registers[0 .. 3]). IMMA8: the first of each is its register of the tile's first
 * instruction (the even inputs), the next its register of the second (registers[4 .. 7]). FP8: registers[0] is
 * output g's four bytes of the tile, registers[1] output g + 8's. NO_DECODE: the window's two words, undecoded. */
template <int bits, int form, int window_kind>
__device__ __forceinline__ void tr_tile_registers(const uint32_t *kept, const struct tr_window *window,
                                                  const uint8_t *table, uint32_t *registers) {
    uint32_t states[TR_LANE_WEIGHTS], high = kept[0], low = kept[1];
    if (window_kind == TR_WINDOW_SHUFFLE) {      /* from the lanes that loaded them */
        high = __shfl_sync(TR_EVERY_LANE, kept[0], (int)window->first_word);
        low = __shfl_sync(TR_EVERY_LANE, kept[0], (int)window->second_word);
    }
    tr_states<bits>(high, low, window, states);
    for (uint32_t pair = 0; pair < TR_WEIGHT_REGISTERS; pair++) {
        const uint32_t first = pair % TR_LANE_OUTPUTS * TR_HALF_WEIGHTS + pair / TR_LANE_OUTPUTS * TR_REGISTER_WEIGHTS;
        if (form == TR_FORM_NO_DECODE) {
            registers[pair] = pair % TR_LANE_OUTPUTS == 0u ? high : low;
        } else if (form == TR_FORM_MUL1_IMMA8) {
            registers[pair] = states[first] * TR_MUL1_MULTIPLIER;
            registers[pair + TR_WEIGHT_REGISTERS] = states[first + 1u] * TR_MUL1_MULTIPLIER;
        } else if (form == TR_FORM_EXLR8_FP8) {
            const uint32_t bytes = tr_weight_pair<form>(states[first], states[first + 1u], table);
            registers[pair % TR_LANE_OUTPUTS] = pair < TR_LANE_OUTPUTS ? bytes
                                                : registers[pair % TR_LANE_OUTPUTS] | (bytes << TR_HALF_BITS);
        } else {
            registers[pair] = tr_weight_pair<form>(states[first], states[first + 1u], table);
        }
    }
}

/* ---- the product ---- */
/* One thread of a block of teams of S warps: slice threadIdx.x / 32 of one output tile of one job, its team's item:
 * items (blockIdx.x + pass * gridDim.x) * teams + team, pass 0 on. The table's form has blockDim.y teams, its items a
 * multiple of them by the host, so every team meets every barrier; every other form one team, as a constant. shared:
 * the table's bytes (that form only), then the block's sums as [thread][4]. broken_order (the stamp of the probe's
 * --break exactness, never timed): a job of several rows adds its slices from another slice on, by their number. */
template <int bits, int form, int window_kind, int stage, int broken_order>
__device__ __forceinline__ void tr_product_block(const struct tr_step *step, uint4 *shared) {
    const uint32_t lane = threadIdx.x % TR_LANES_PER_WARP, place = lane % TR_LANES_PER_GROUP;
    const uint32_t slice = threadIdx.x / TR_LANES_PER_WARP, slices = blockDim.x / TR_LANES_PER_WARP;
    const uint32_t teams = form == TR_FORM_EXLR8_TABLE ? blockDim.y : 1u;
    const uint32_t team = form == TR_FORM_EXLR8_TABLE ? threadIdx.y : 0u, thread = team * blockDim.x + threadIdx.x;
    const uint32_t input_size = TR_STAGE_INPUT_SIZE(stage), blocks = input_size / TR_HADAMARD;
    const uint32_t first_tile = slice * blocks / slices * TR_BLOCK_TILES;
    const uint32_t end_tile = (slice + 1u) * blocks / slices * TR_BLOCK_TILES;
    const size_t tile_words = TR_WORDS_PER_BIT * bits, unit = TR_FEED_UNIT(TR_FORM_FEED(form));
    const uint8_t *feed = stage == TR_DOWN_STAGE ? step->feed_hidden : step->feed_input;
    const float *scales = stage == TR_DOWN_STAGE ? step->scales_hidden : step->scales_input;
    float *sums_out = stage == TR_DOWN_STAGE ? step->sums_outer : step->sums_inner;
    const struct tr_job *jobs = step->jobs + (stage == TR_DOWN_STAGE ? TR_MOST_GATE_UP_JOBS : 0);
    const uint8_t *table = (const uint8_t *)shared;
    float *partial_sums = (float *)(shared + (form == TR_FORM_EXLR8_TABLE ? TR_TABLE_LOADS : 0));
    struct tr_window window;
    tr_set_window<bits>(lane, &window);
    if (form == TR_FORM_EXLR8_TABLE) {
        for (uint32_t load = thread; load < TR_TABLE_LOADS; load += blockDim.x * teams) {
            shared[load] = ((const uint4 *)step->table)[load];
        }
        __syncthreads();
    }
    for (uint32_t item = blockIdx.x * teams + team; item < step->item_count[stage]; item += gridDim.x * teams) {
        const struct tr_job *job = &jobs[item / TR_STAGE_ITEMS(stage)];
        const uint32_t tile = item % TR_STAGE_ITEMS(stage);
        const uint32_t *stream = step->stored + ((size_t)job->matrix * TR_MATRIX_TILES
                                                 + (size_t)tile * (input_size / TR_TILE)) * tile_words;
        const uint8_t *lane_feed = feed + (size_t)job->vector[lane / TR_LANES_PER_GROUP] * input_size * TR_FEED_BYTES
                                   + place * TR_LANES_PER_GROUP * unit;
        const float *column_scales[TR_LANE_COLUMNS] = {           /* of the lane's two result columns */
            scales + (size_t)job->vector[place * TR_LANE_COLUMNS] * blocks * TR_BLOCK_SCALES,
            scales + (size_t)job->vector[place * TR_LANE_COLUMNS + 1u] * blocks * TR_BLOCK_SCALES};
        float totals[TR_LANE_SUMS] = {0.0f, 0.0f, 0.0f, 0.0f}, float_sums[TR_LANE_SUMS] = {0.0f, 0.0f, 0.0f, 0.0f};
        int32_t integer_sums[TR_LANE_SUMS] = {0, 0, 0, 0};
        uint32_t kept[TR_REGISTER_WEIGHTS] = {0, 0}, ahead[TR_REGISTER_WEIGHTS] = {0, 0};
#if PROBE_SM == EXLR8_SM_120A
        uint32_t held[TR_LANE_OUTPUTS + 1] = {0, 0, 0};       /* FP8: the first tile of two, and its fed word */
#endif
        tr_fetch<bits, window_kind>(stream + first_tile * tile_words, lane, &window, ahead);
        for (uint32_t input_tile = first_tile; input_tile < end_tile; input_tile++) {
            const uint8_t *fed = lane_feed + (size_t)input_tile * TR_TILE * unit;
            uint32_t registers[TR_LANE_WEIGHTS];
            kept[0] = ahead[0];
            kept[1] = ahead[1];
            if (input_tile + 1u < end_tile) {    /* the next tile is asked before this one is worked on */
                tr_fetch<bits, window_kind>(stream + (input_tile + 1u) * tile_words, lane, &window, ahead);
            }
            tr_tile_registers<bits, form, window_kind>(kept, &window, table, registers);
            if (form == TR_FORM_MUL1_IMMA8) {
                const uint4 integers = *(const uint4 *)fed;
                tr_mma_u8_s8(integer_sums, registers, integers.x, integers.y);
                tr_mma_u8_s8(integer_sums, registers + TR_WEIGHT_REGISTERS, integers.z, integers.w);
            } else if (form == TR_FORM_EXLR8_FP8) {
#if PROBE_SM == EXLR8_SM_120A
                if (input_tile % TR_FP8_TILES != 0u) {
                    const uint32_t weights[TR_WEIGHT_REGISTERS] = {held[0], held[1], registers[0], registers[1]};
                    tr_mma_e4m3(float_sums, weights, held[TR_LANE_OUTPUTS], *(const uint32_t *)fed);
                }
                held[0] = registers[0];
                held[1] = registers[1];
                held[TR_LANE_OUTPUTS] = *(const uint32_t *)fed;
#endif
            } else {
                const uint2 pairs = *(const uint2 *)fed;
                tr_mma_fp16(float_sums, registers, pairs.x, pairs.y);
            }
            if (TR_FORM_BLOCK_STEP(form) && input_tile % TR_BLOCK_TILES == TR_BLOCK_TILES - 1u) {
                for (uint32_t column = 0; column < TR_LANE_COLUMNS; column++) {
                    const float2 scale = *(const float2 *)(column_scales[column]
                                                           + input_tile / TR_BLOCK_TILES * TR_BLOCK_SCALES);
                    for (uint32_t element = column; element < TR_LANE_SUMS; element += TR_LANE_COLUMNS) {
                        const float block_sum = form == TR_FORM_MUL1_IMMA8 ? (float)integer_sums[element]
                                                                            : float_sums[element];
                        totals[element] = fmaf(block_sum, scale.x, totals[element] - scale.y);
                        integer_sums[element] = 0;
                        float_sums[element] = 0.0f;
                    }
                }
            }
        }
        for (uint32_t element = 0; element < TR_LANE_SUMS; element++) {
            partial_sums[thread * TR_LANE_SUMS + element] = TR_FORM_BLOCK_STEP(form) ? totals[element]
                                                                                      : float_sums[element];
        }
        __syncthreads();
        for (uint32_t element = 0; element < TR_LANE_SUMS && slice == 0u; element++) {
            const uint32_t column = place * TR_LANE_COLUMNS + element % TR_LANE_COLUMNS, rows = job->row_count;
            const uint32_t output = tile * TR_TILE + lane / TR_LANES_PER_GROUP
                                    + element / TR_LANE_COLUMNS * TR_HALF_OUTPUTS;
            const uint32_t from = broken_order != 0 && rows > 1u ? 1u + rows % (slices - 1u) : 0u;
            float sum = 0.0f;
            for (uint32_t turn = from; turn < from + slices; turn++) {
                const uint32_t added = turn < slices ? turn : turn - slices;      /* this lane of that slice: */
                sum += partial_sums[(thread + added * TR_LANES_PER_WARP) * TR_LANE_SUMS + element];
            }
            if (column < rows) {
                sums_out[(size_t)job->vector[column] * TR_STAGE_OUTPUT_SIZE(stage) + output] =
                    sum * TR_FORM_SCALE(form);
            }
        }
        if (form == TR_FORM_EXLR8_TABLE) { __syncthreads(); }      /* before the next item's sums */
    }
}

/* ---- the rotations: a thread per value, a block of 128 values in four warps ---- */
/* Value `index` of a block through Hb: five stages in the warp, the pair (lower, upper) of distance 1, 2, .. 16
 * becoming (lower + upper, lower - upper); then, through shared memory, the stages of distance 32 and 64 in that
 * order; then times float32(1 / sqrt(128)). The order is fixed: trellis_reference.h's tr_reference_rotate has the
 * same bits. scratch: TR_HADAMARD floats that no other call of the block uses. One barrier. */
__device__ __forceinline__ float tr_hadamard(float value, uint32_t index, float *scratch) {
    const uint32_t lane = index % TR_LANES_PER_WARP, warp = index / TR_LANES_PER_WARP;
    for (uint32_t distance = 1; distance < TR_LANES_PER_WARP; distance += distance) {
        const float other = __shfl_xor_sync(TR_EVERY_LANE, value, (int)distance);
        value = (lane & distance) != 0u ? other - value : value + other;
    }
    scratch[index] = value;
    __syncthreads();
    const float *lower = scratch + lane, *upper = lower + TR_HALF_HADAMARD;       /* of warps 0, 1 and of 2, 3 */
    const float low = (warp & 1u) != 0u ? lower[0] - lower[TR_LANES_PER_WARP] : lower[0] + lower[TR_LANES_PER_WARP];
    const float high = (warp & 1u) != 0u ? upper[0] - upper[TR_LANES_PER_WARP] : upper[0] + upper[TR_LANES_PER_WARP];
    return __fmul_rn(index >= TR_HALF_HADAMARD ? low - high : low + high, TR_HADAMARD_SCALE);
}
/* One value of each of a block's 128 threads reduced to the largest magnitude (sum 0) or to the sum (sum 1: of
 * whole numbers, exact in any order), known to every thread after. scratch: TR_BLOCK_WARPS floats of its own. */
__device__ __forceinline__ float tr_block_reduce(float value, int sum, uint32_t index, float *scratch) {
    float reduced = sum ? value : fabsf(value);
    for (uint32_t distance = 1; distance < TR_LANES_PER_WARP; distance += distance) {
        const float other = __shfl_xor_sync(TR_EVERY_LANE, reduced, (int)distance);
        reduced = sum ? reduced + other : fmaxf(reduced, other);
    }
    if (index % TR_LANES_PER_WARP == 0u) { scratch[index / TR_LANES_PER_WARP] = reduced; }
    __syncthreads();
    return sum ? (scratch[0] + scratch[1]) + (scratch[2] + scratch[3])
               : fmaxf(fmaxf(scratch[0], scratch[1]), fmaxf(scratch[2], scratch[3]));
}
/* Where value `column` of a vector lies in its feed, in the feed's units. Lane place t of a tile is fed inputs
 * 2 t, 2 t + 1, 2 t + 8, 2 t + 9: fp16 and E4M3 bytes in that order (B's two registers; the four bytes of a word);
 * the four-byte integers as 2 t, 2 t + 8, 2 t + 1, 2 t + 9 (B of the tile's first instruction, then the second's). */
__device__ __forceinline__ uint32_t tr_feed_place(uint32_t column, int integers) {
    const uint32_t in_tile = column % TR_TILE, odd = in_tile % TR_REGISTER_WEIGHTS, far = in_tile / TR_HALF_INPUTS;
    return column - in_tile + in_tile % TR_HALF_INPUTS / TR_REGISTER_WEIGHTS * TR_LANES_PER_GROUP
           + (integers ? odd * TR_REGISTER_WEIGHTS + far : far * TR_REGISTER_WEIGHTS + odd);
}
/* THE FEED: one rotated value written as the product's operand. FP16: the nearest fp16. INT8: with m the block's
 * largest magnitude the integer is the nearest to value * (127 / m), in four equal bytes, and the block's two
 * floats are a = (m / 127) * 1774 / 2^18 and c = (the block's sum of integers * 905216 / 2^18) * (m / 127); a
 * block of zeros has integers, a and c of 0. FP8 (sm_120a): the scale is the power of two at or above
 * max(m, 1e-4) / 448, the byte the E4M3 nearest to value / scale, a the scale and c 0. Every product here is one
 * fp32 rounding: the host's reference repeats them. scratch: TR_REDUCE_SCRATCH floats. */
template <int feed>
__device__ __forceinline__ void tr_feed(float value, uint32_t index, uint32_t column, uint8_t *vector_feed,
                                        float *block_scales, float *scratch) {
    if (feed == TR_FEED_FP16) {
        ((uint16_t *)vector_feed)[tr_feed_place(column, 0)] = __half_as_ushort(__float2half(value));
    } else if (feed == TR_FEED_INT8) {
        const float largest = tr_block_reduce(value, 0, index, scratch), scale = largest / TR_INT8_LARGEST;
        const int32_t integer = __float2int_rn(__fmul_rn(value, largest > 0.0f ? TR_INT8_LARGEST / largest : 0.0f));
        const float block_sum = tr_block_reduce((float)integer, 1, index, scratch + TR_BLOCK_WARPS);
        ((uint32_t *)vector_feed)[tr_feed_place(column, 1)] = ((uint32_t)integer & TR_BYTE_MASK) * TR_BYTE_ONES;
        if (index == 0u) {
            block_scales[0] = __fmul_rn(scale, TR_MUL1_STEP);
            block_scales[1] = __fmul_rn(__fmul_rn(block_sum, TR_MUL1_OFFSET), scale);
        }
    } else {
#if PROBE_SM == EXLR8_SM_120A
        const float largest = fmaxf(tr_block_reduce(value, 0, index, scratch), TR_FEED_FLOOR);
        const uint32_t ratio = __float_as_uint(__fmul_rn(largest, TR_E4M3_LARGEST_INVERSE));
        const uint32_t field = (ratio >> TR_FP32_EXPONENT_SHIFT) + ((ratio & TR_FP32_MANTISSA_MASK) != 0u ? 1u : 0u);
        const float inverse = __uint_as_float((TR_INVERSE_FIELD_BASE - field) << TR_FP32_EXPONENT_SHIFT);
        vector_feed[tr_feed_place(column, 0)] = (uint8_t)exlr8_byte_of_value(__fmul_rn(value, inverse));
        if (index == 0u) {
            block_scales[0] = __uint_as_float(field << TR_FP32_EXPONENT_SHIFT);
            block_scales[1] = 0.0f;
        }
#endif
    }
}
__device__ __forceinline__ float tr_side(const uint16_t *sides, uint32_t at) {
    return __half2float(__ushort_as_half(sides[at]));
}
/* grid (hidden size / 128, pairs * 2), TR_HADAMARD threads: vector blockIdx.y is pair blockIdx.y / 2 at gate (0)
 * or up (1): (x .* suh) Hb, fed. */
template <int feed>
__global__ void tr_rotate_input_kernel(struct tr_step step) {
    __shared__ float scratch[TR_INPUT_SCRATCH];
    const uint32_t index = threadIdx.x, vector = blockIdx.y, column = blockIdx.x * TR_HADAMARD + index;
    const struct tr_pair pair = step.pairs[vector / TR_GATE_UP_MATRICES];
    const float side = tr_side(step.sides + (size_t)pair.expert * TR_SIDE_VALUES,
                               vector % TR_GATE_UP_MATRICES * TR_HIDDEN_SIZE + column);
    const float value = step.inputs[(size_t)pair.row * TR_HIDDEN_SIZE + column];
    tr_feed<feed>(tr_hadamard(__fmul_rn(value, side), index, scratch), index, column,
                  step.feed_input + (size_t)vector * TR_HIDDEN_SIZE * TR_FEED_BYTES,
                  step.scales_input + ((size_t)vector * (TR_HIDDEN_SIZE / TR_HADAMARD) + blockIdx.x) * TR_BLOCK_SCALES,
                  scratch + TR_HADAMARD);
}
/* grid (inner size / 128, pairs), TR_HADAMARD threads: gate = (sums Hb) .* svh, up likewise, the hidden value
 * silu(min(gate, L)) * clamp(up, -L, L) * route weight (stored: the probe checks it), then (h .* suh) Hb, fed. */
template <int feed>
__global__ void tr_rotate_middle_kernel(struct tr_step step) {
    __shared__ float scratch[TR_MIDDLE_SCRATCH];
    const uint32_t index = threadIdx.x, column = blockIdx.x * TR_HADAMARD + index;
    const size_t pair_index = blockIdx.y;
    const struct tr_pair pair = step.pairs[pair_index];
    const uint16_t *sides = step.sides + (size_t)pair.expert * TR_SIDE_VALUES;
    const float *sums = step.sums_inner + pair_index * TR_GATE_UP_MATRICES * TR_INNER_SIZE + column;
    const float gate = __fmul_rn(tr_hadamard(sums[0], index, scratch), tr_side(sides, TR_SVH_GATE_AT + column));
    const float up = __fmul_rn(tr_hadamard(sums[TR_INNER_SIZE], index, scratch + TR_HADAMARD),
                               tr_side(sides, TR_SVH_UP_AT + column));
    const float bounded = fminf(gate, TR_SWIGLU_LIMIT), clamped = fminf(fmaxf(up, -TR_SWIGLU_LIMIT), TR_SWIGLU_LIMIT);
    const float gated = __fmul_rn(bounded / (1.0f + expf(-bounded)), clamped);
    const float hidden = __fmul_rn(gated, pair.route_weight);
    step.hidden[pair_index * TR_INNER_SIZE + column] = hidden;
    tr_feed<feed>(tr_hadamard(__fmul_rn(hidden, tr_side(sides, TR_SUH_DOWN_AT + column)), index,
                              scratch + (TR_MIDDLE_ROTATIONS - 1) * TR_HADAMARD),
                  index, column, step.feed_hidden + pair_index * TR_INNER_SIZE * TR_FEED_BYTES,
                  step.scales_hidden + (pair_index * (TR_INNER_SIZE / TR_HADAMARD) + blockIdx.x) * TR_BLOCK_SCALES,
                  scratch + TR_MIDDLE_ROTATIONS * TR_HADAMARD);
}
/* grid (hidden size / 128, rows), TR_OUTPUT_THREADS threads: 128 per slot of the row: (sums Hb) .* svh of the
 * slot's pair; the threads of slot 0 add the six in slot order from +0. */
__global__ void tr_rotate_output_kernel(struct tr_step step) {
    __shared__ float scratch[TR_OUTPUT_SCRATCH];
    const uint32_t slot = threadIdx.x / TR_HADAMARD, index = threadIdx.x % TR_HADAMARD;
    const uint32_t column = blockIdx.x * TR_HADAMARD + index;
    const size_t pair_index = (size_t)blockIdx.y * TR_EXPERTS_PER_ROW + slot;
    const uint16_t *sides = step.sides + (size_t)step.pairs[pair_index].expert * TR_SIDE_VALUES;
    float *rotated = scratch + TR_OUTPUT_THREADS;
    rotated[threadIdx.x] = __fmul_rn(
        tr_hadamard(step.sums_outer[pair_index * TR_HIDDEN_SIZE + column], index, scratch + slot * TR_HADAMARD),
        tr_side(sides, TR_SVH_DOWN_AT + column));
    __syncthreads();
    if (slot == 0u) {
        float sum = 0.0f;
        for (uint32_t added = 0; added < TR_EXPERTS_PER_ROW; added++) { sum += rotated[added * TR_HADAMARD + index]; }
        step.row_output[(size_t)blockIdx.y * TR_HIDDEN_SIZE + column] = sum;
    }
}

/* ---- the stamps: a kernel and its launch per form, width K (the 2, 3 and 4 below) and stage ---- */
#define TR_NO_BOUND                         /* no launch bound: every form but the table's, as the engine's kernels */
#define TR_PRODUCT(name, bits, form, window_kind, stage, broken_order, bound)                                        \
    __global__ void bound name(struct tr_step step) {                                                                \
        extern __shared__ uint4 tr_shared[];                                                                         \
        tr_product_block<bits, form, window_kind, stage, broken_order>(&step, tr_shared);                            \
    }                                                                                                                \
    inline void launch_##name(const struct tr_step *step, uint32_t blocks, dim3 block, size_t shared_bytes) {        \
        name<<<blocks, block, shared_bytes>>>(*step);                                                                \
    }                                                                                                                \
    /* more than the 48 KiB a block has unasked; then *resident: the blocks of `threads` an SM holds at once */      \
    inline cudaError_t admit_##name(size_t shared_bytes, int threads, int *resident) {                               \
        const cudaError_t status =                                                                                   \
            cudaFuncSetAttribute(name, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shared_bytes);              \
        return status != cudaSuccess ? status                                                                        \
                                     : cudaOccupancyMaxActiveBlocksPerMultiprocessor(resident, name, threads,        \
                                                                                     shared_bytes);                  \
    }
#define TR_STAMP_WIDTH(tag, bits, form, window_kind, broken_order, bound)                                            \
    TR_PRODUCT(tr_product_##tag##_k##bits##_gate_up, bits, form, window_kind, TR_GATE_UP_STAGE, broken_order, bound) \
    TR_PRODUCT(tr_product_##tag##_k##bits##_down, bits, form, window_kind, TR_DOWN_STAGE, broken_order, bound)
#define TR_STAMP(tag, form, window_kind, broken_order, bound)                                                        \
    TR_STAMP_WIDTH(tag, 2, form, window_kind, broken_order, bound)                                                   \
    TR_STAMP_WIDTH(tag, 3, form, window_kind, broken_order, bound)                                                   \
    TR_STAMP_WIDTH(tag, 4, form, window_kind, broken_order, bound)
#define TR_SET_WIDTH(prefix, tag, bits) \
    {prefix##tr_product_##tag##_k##bits##_gate_up, prefix##tr_product_##tag##_k##bits##_down}
#define TR_SET(prefix, tag) {TR_SET_WIDTH(prefix, tag, 2), TR_SET_WIDTH(prefix, tag, 3), TR_SET_WIDTH(prefix, tag, 4)}
#define TR_NAMES_WIDTH(tag, bits) "tr_product_" #tag "_k" #bits "_gate_up+tr_product_" #tag "_k" #bits "_down"
#define TR_KERNELS(tag) {TR_SET(launch_, tag), {TR_NAMES_WIDTH(tag, 2), TR_NAMES_WIDTH(tag, 3), TR_NAMES_WIDTH(tag, 4)}}
#define TR_NO_KERNELS {{{NULL, NULL}, {NULL, NULL}, {NULL, NULL}}, {NULL, NULL, NULL}}
TR_STAMP(fp16, TR_FORM_MUL1_FP16, TR_WINDOW_SHUFFLE, 0, TR_NO_BOUND)
TR_STAMP(fp16_loads, TR_FORM_MUL1_FP16, TR_WINDOW_LOADS, 0, TR_NO_BOUND)
TR_STAMP(fp16_broken_order, TR_FORM_MUL1_FP16, TR_WINDOW_SHUFFLE, 1, TR_NO_BOUND)
TR_STAMP(imma8, TR_FORM_MUL1_IMMA8, TR_WINDOW_SHUFFLE, 0, TR_NO_BOUND)
TR_STAMP(mcg, TR_FORM_MCG_FP16, TR_WINDOW_SHUFFLE, 0, TR_NO_BOUND)
TR_STAMP(exlr8_round, TR_FORM_EXLR8_ROUND, TR_WINDOW_SHUFFLE, 0, TR_NO_BOUND)
TR_STAMP(exlr8_table, TR_FORM_EXLR8_TABLE, TR_WINDOW_SHUFFLE, 0, __launch_bounds__(TR_TABLE_BLOCK_THREADS, 1))
TR_STAMP(no_decode, TR_FORM_NO_DECODE, TR_WINDOW_SHUFFLE, 0, TR_NO_BOUND)
#if PROBE_SM == EXLR8_SM_120A
TR_STAMP(exlr8_cvt, TR_FORM_EXLR8_CVT, TR_WINDOW_SHUFFLE, 0, TR_NO_BOUND)
TR_STAMP(exlr8_fp8, TR_FORM_EXLR8_FP8, TR_WINDOW_SHUFFLE, 0, TR_NO_BOUND)
#define TR_CVT_KERNELS TR_KERNELS(exlr8_cvt)
#define TR_FP8_KERNELS TR_KERNELS(exlr8_fp8)
#else
#define TR_CVT_KERNELS TR_NO_KERNELS
#define TR_FP8_KERNELS TR_NO_KERNELS
#endif
#endif /* TRELLIS_KERNELS_CUH */
