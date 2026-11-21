/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * attention_core_kernels.cuh: the attention core of one layer, all rows of a launch, as ONE kernel body with
 * compile-time choices: package 11b of round 1 of the speed campaign (research/speed/ROUND_1_attention.md,
 * "attention_core"; owner's rulings of DECISIONS_2026-10-12.md section 1). Everything down to "stamping" is written to
 * move into engine/source as it stands once a choice has won; attention_core_probe_kernels.cuh holds what exists for
 * the measurement alone and attention_core_probe.cu every check. Measurement tool: not part of the engine.
 *
 * Contract (a row's arithmetic is the same whatever the heads of a block, the ring's depth, the rows of the launch and
 * where its rows lie; a step rule of a measurement's own, attention_rescale_step, may break it on purpose):
 *   - Queries [row][64][512] fp16, rotated by their producer. Key and value of every head is ONE stored row of 512
 *     dims. indices [row][640]: 128 window rows (into window_rows) then 512 pool entries (into pool_rows), -1 for
 *     none. A row's 640 keys are 40 groups of 16 in that order; a group is one softmax step.
 *   - Stored rows, by kind: FP16: 1,024 B of fp16. TRAINED (the values the model was trained on, 8-bit codes):
 *     window rows 512 E4M3 codes then 16 E8M0 scales, a power of two per 32 dims (528 B, reference model.py:707);
 *     pool entries 512 E2M1 codes, one in the low nibble of a byte (bit 3 the sign), then 32 E4M3 scales, one per 16
 *     dims (544 B, model.py:760). FRESH (a fresh 8-bit rounding): window rows and pool entries both in the 528 B form.
 *   - Score s = (q . k) x 512^-0.5 in fp32. The 512 dims are 4 slices of 128; a slice is one fp32 chain from +0 in
 *     ascending dims (the instruction adds 16 products at a time); the slices are added in ascending order from +0.
 *     8-bit rows: a code is placed as an fp16 (exact: the code's value times 2^-8, an E2M1 value times 2^-14), the
 *     products of one scale block (32 dims, or 16 for E2M1) are one sum from +0, and the chain takes
 *     fmaf(block sum, scale x the placement's inverse, chain): window scales reach 2^-22, below fp16's range, so the
 *     scale is applied in fp32 after the block's products and never to an fp16 operand.
 *   - Softmax, per head: a running maximum m from ATTENTION_SCORE_FLOOR (the reference's finite floor); a group's
 *     largest score g; if g > m + step (ATTENTION_CONTRACT_STEP; 0 for ATTENTION_STEP_ALWAYS) then O and the running
 *     sum are multiplied by expf(m - g) and m = g. p = expf(s - m), 0 for a key of index -1. The group's 16 p are
 *     added in a fixed tree (key j with key j + 8, then lanes 1, 2, 4 apart); sum = sum x factor + group sum.
 *     P is rounded to fp16 (FP16) or bfloat16 (8-bit kinds, as the reference rounds it) for P x V.
 *   - O += P x V in fp32, the instruction adding 16 keys to the accumulator. FP16: V is the stored fp16. 8-bit kinds:
 *     the scale of V changes along the keys the product adds, so it cannot follow the product: each value is decoded
 *     to bfloat16 (code x scale has at most 6 significant bits and lies far inside bfloat16's range: exact) into a
 *     tile of shared memory, by the warps that placed the same codes for the scores.
 *   - Epilogue: the denominator is sum + expf(sink - m), O is divided by it (IEEE division), the last 64 dims are
 *     de-rotated ((x0 + i x1) times cos - i sin of the row's position, products rounded, then added), and each value
 *     is rounded once to bfloat16 and stored [row][64][512], the value order wo_a reads. A row with no valid key
 *     gives zeros.
 *   - A kernel uses no atomic operation and checks nothing (ATTENTION_CORE_CHECKS 1, the debug build, adds the checks
 *     of an index into its plane and of the block): the host has held every range a launch reads or writes, every
 *     index of a list and the geometry to their arrays and limits before the launch.
 *
 * The kernel: grid (rows, 64 / heads of a block); a block is 4 warps for 8 or 16 heads, 16 warps for 64. Warp w takes
 * slice w % 4 of the dims and the head tiles 2 (w / 4), 2 (w / 4) + 1 (8 heads a tile). m16n8k16 with the keys or
 * the dims as its 16 rows and 8 heads as its columns: scores^T = K Q^T, O^T += V^T P^T. Queries live in registers.
 * A group's rows come through a ring of `stages` stages filled by 16-byte asynchronous copies; every iteration
 * commits one copy group, an empty one where it issues nothing, so the wait before a stage is read counts the
 * copies still pending also at the ring's tail. A thread reads its key's index of the group the ring fills next one
 * group ahead. Shared memory: the ring, the decoded tile (8-bit kinds), the slices' partial scores, P, the factors,
 * the indices; the epilogue's staging reuses the first of them (attention_layout_of).
 */
#ifndef ATTENTION_CORE_KERNELS_CUH
#define ATTENTION_CORE_KERNELS_CUH
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#ifndef ATTENTION_CORE_CHECKS
#define ATTENTION_CORE_CHECKS 0                  /* the includer's debug switch: 1 adds the checks of a launch */
#endif

enum attention_kind { ATTENTION_FP16 = 0, ATTENTION_TRAINED = 1, ATTENTION_FRESH = 2 };
enum attention_step_rule { ATTENTION_STEP_CONTRACT = 0, ATTENTION_STEP_ALWAYS = 1 };
enum attention_format { ATTENTION_FORMAT_FP16 = 0, ATTENTION_FORMAT_POW2 = 1, ATTENTION_FORMAT_E2M1 = 2 };
enum {
    ATTENTION_HEADS = 64, ATTENTION_DIMS = 512, ATTENTION_ROPE_FIRST_DIM = 448, ATTENTION_ROPE_PAIRS = 32,
    ATTENTION_ROPE_VALUES = 2,                   /* cos, sin of a pair */
    ATTENTION_WINDOW_KEYS = 128, ATTENTION_POOL_KEYS = 512, ATTENTION_KEYS = 640, ATTENTION_GROUP_KEYS = 16,
    ATTENTION_GROUPS = 40, ATTENTION_WINDOW_GROUPS = 8, ATTENTION_HALF_GROUP = 8,
    ATTENTION_LANES = 32, ATTENTION_QUAD = 4,    /* lanes 4 g + t: g the instruction's row group, t its place */
    ATTENTION_TILE_HEADS = 8, ATTENTION_SLICES = 4, ATTENTION_SLICE_DIMS = 128, ATTENTION_STEP_DIMS = 16,
    ATTENTION_STEPS = 8,                         /* of 16 dims, a slice */
    ATTENTION_WARP_TILES = 2,                    /* head tiles of a warp at most */
    ATTENTION_PAIR_THREADS = 128,                /* the 4 warps of one pair of head tiles */
    ATTENTION_ELEMENTS = 4, ATTENTION_REGISTERS_A = 4, ATTENTION_REGISTERS_B = 2, ATTENTION_MATRIX_ROWS = 8,
    ATTENTION_CHUNK_BYTES = 16, ATTENTION_HALF_BYTES = 2, ATTENTION_SWIZZLE_MASK = 7, ATTENTION_CHUNK_HALVES = 8,
    ATTENTION_FP16_ROW_BYTES = 1024, ATTENTION_CODE_BYTES = 512, ATTENTION_CODE_CHUNKS = 32,
    ATTENTION_POW2_ROW_BYTES = 528, ATTENTION_E2M1_ROW_BYTES = 544, ATTENTION_STAGE_SCALE_BYTES = 32,
    ATTENTION_POW2_BLOCK_DIMS = 32, ATTENTION_E2M1_BLOCK_DIMS = 16, ATTENTION_SOFTMAX_LANES = 8,
    ATTENTION_PARTIAL_PAD = 4,
    ATTENTION_STAGED_HALVES = 136,               /* a head's 128 dims and 8 halves more: rows apart in the banks */
    ATTENTION_PROBS_SWIZZLE_SHIFT = 2,           /* P [head][key]: heads 4 .. 7 of a tile take the other chunk */
    ATTENTION_BYTE_BITS = 8, ATTENTION_HALF_BITS = 16, ATTENTION_FLOAT_EXPONENT_SHIFT = 23,
    ATTENTION_FP16_PLACEMENT = 8,                /* an E4M3 code placed as fp16 is its value times 2^-8 */
    ATTENTION_E4M3_FLOAT_SHIFT = 20,             /* an E4M3 magnitude at float bits 20 .. 26: its value times 2^-120 */
    ATTENTION_E4M3_MANTISSA_BITS = 3,
    ATTENTION_FAILED_INDEX = 1, ATTENTION_FAILED_GEOMETRY = 2,       /* ATTENTION_CORE_CHECKS: what a status says */
    ATTENTION_STATUS_SHIFT = 32,
    ATTENTION_TILE_CHUNKS_PER_STEP = 2         /* 16 dims of bfloat16: two chunks of the tile, two lanes t a chunk */
};
#define ATTENTION_SOFTMAX_SCALE 0.044194173824159216f       /* 512^-0.5, as the reference's softmax_scale in fp32 */
#define ATTENTION_CONTRACT_STEP 8.0f             /* a rise of the maximum that rescales; p stays under e^8 */
#define ATTENTION_SCORE_FLOOR (-1.0e30f)         /* the reference's finite floor: no exp(-inf - -inf) */
#define ATTENTION_E4M3_TO_FLOAT 0x1p120f         /* undoes ATTENTION_E4M3_FLOAT_SHIFT's 2^-120 */
#define ATTENTION_E2M1_INVERSE_PLACEMENT 0x1p14f
#define ATTENTION_EVERY_LANE 0xFFFFFFFFu
#define ATTENTION_E4M3_MAGNITUDE_MASK 0x7Fu
#define ATTENTION_E4M3_SIGN 0x80u
/* placement of four codes of a word as two registers of fp16 halves: bytes 0 and 2, then bytes 1 and 3 */
#define ATTENTION_E4M3_EVEN_SIGN 0x00800080u
#define ATTENTION_E4M3_EVEN_MAGNITUDE 0x007F007Fu
#define ATTENTION_E4M3_ODD_SIGN 0x80008000u
#define ATTENTION_E4M3_ODD_MAGNITUDE 0x7F007F00u
#define ATTENTION_E2M1_EVEN_SIGN 0x00080008u
#define ATTENTION_E2M1_EVEN_MAGNITUDE 0x00070007u
#define ATTENTION_E2M1_ODD_SIGN 0x08000800u
#define ATTENTION_E2M1_ODD_MAGNITUDE 0x07000700u
#define ATTENTION_E4M3_EVEN_SIGN_SHIFT 8
#define ATTENTION_E4M3_EVEN_SHIFT 7
#define ATTENTION_E4M3_ODD_SHIFT 1
#define ATTENTION_E2M1_EVEN_SIGN_SHIFT 12
#define ATTENTION_E2M1_EVEN_SHIFT 9
#define ATTENTION_E2M1_ODD_SIGN_SHIFT 4
#define ATTENTION_E2M1_ODD_SHIFT 1
#define ATTENTION_LOW_HALVES 0x5410u             /* prmt: halves 0 and 2 of two words; then halves 1 and 3 */
#define ATTENTION_HIGH_HALVES 0x7632u

struct attention_core_arguments {
    const uint16_t *queries;                     /* fp16 [row][64][512] */
    const unsigned char *window_rows;            /* the stored window rows an index of 0 .. 127 counts from */
    const unsigned char *pool_rows;              /* the stored pool entries (or a compact copy of them) */
    const int32_t *indices;                      /* [row][640] */
    const float *sinks;                          /* [64] */
    const float *rotation;                       /* [row][32][2]: cos, sin of the row's position */
    uint16_t *output;                            /* bfloat16 [row][64][512] */
};
struct attention_core_checks {                   /* ATTENTION_CORE_CHECKS only */
    unsigned long long *status_word;             /* 0, or (what << 32 | row) of a block that met a bad value */
    uint64_t window_row_count, pool_row_count;   /* rows of the two planes from the launch's pointers on */
};
#if ATTENTION_CORE_CHECKS
#define ATTENTION_CHECKS_PARAMETER , struct attention_core_checks checks
#define ATTENTION_CHECKS_ARGUMENT , checks
#define ATTENTION_CHECKS_POINTER , const struct attention_core_checks *checks
#define ATTENTION_CHECKS_ADDRESS , &checks
#else
#define ATTENTION_CHECKS_PARAMETER
#define ATTENTION_CHECKS_ARGUMENT
#define ATTENTION_CHECKS_POINTER
#define ATTENTION_CHECKS_ADDRESS
#endif

/* ---- sizes and places, the same on the host and on the card ---- */

__host__ __device__ constexpr uint32_t attention_threads(int head_tiles)
{
    return head_tiles > ATTENTION_WARP_TILES ? ATTENTION_PAIR_THREADS * (uint32_t)(head_tiles / ATTENTION_WARP_TILES)
                                             : (uint32_t)ATTENTION_PAIR_THREADS;
}

__host__ __device__ constexpr uint32_t attention_row_bytes(int format)
{
    return format == ATTENTION_FORMAT_FP16 ? ATTENTION_FP16_ROW_BYTES
           : format == ATTENTION_FORMAT_POW2 ? ATTENTION_POW2_ROW_BYTES : ATTENTION_E2M1_ROW_BYTES;
}

/* The format of a group's rows: window groups first. */
__host__ __device__ constexpr int attention_format_of(int kind, uint32_t group)
{
    return kind == ATTENTION_FP16 ? ATTENTION_FORMAT_FP16
           : kind == ATTENTION_TRAINED && group >= ATTENTION_WINDOW_GROUPS ? ATTENTION_FORMAT_E2M1
                                                                           : ATTENTION_FORMAT_POW2;
}

/* A stage: 16 rows of fp16 (16 KiB), or 16 rows of codes (8 KiB) and their scales (32 B a row). */
__host__ __device__ constexpr uint32_t attention_stage_bytes(int kind)
{
    return ATTENTION_GROUP_KEYS *
           (kind == ATTENTION_FP16 ? (uint32_t)ATTENTION_FP16_ROW_BYTES
                                   : (uint32_t)(ATTENTION_CODE_BYTES + ATTENTION_STAGE_SCALE_BYTES));
}

struct attention_layout { uint32_t ring, tile, partials, probs, factors, indices, total; };

/* Byte offsets of the dynamic shared memory; every part begins on 16 bytes. The epilogue's staging, 8 heads by 136
 * halves a warp, lies over the ring, the tile and the partials, which nothing reads by then. */
__host__ __device__ constexpr struct attention_layout attention_layout_of(int kind, int head_tiles, int stages)
{
    const uint32_t heads = (uint32_t)head_tiles * ATTENTION_TILE_HEADS;
    const uint32_t warps = attention_threads(head_tiles) / ATTENTION_LANES;
    const uint32_t tile = (uint32_t)stages * attention_stage_bytes(kind);
    const uint32_t partials = tile + (kind == ATTENTION_FP16 ? 0u : ATTENTION_GROUP_KEYS * ATTENTION_FP16_ROW_BYTES);
    const uint32_t work_end =
        partials + ATTENTION_SLICES * ATTENTION_GROUP_KEYS * (heads + ATTENTION_PARTIAL_PAD) * (uint32_t)sizeof(float);
    const uint32_t staged = warps * ATTENTION_TILE_HEADS * ATTENTION_STAGED_HALVES * ATTENTION_HALF_BYTES;
    const uint32_t probs = work_end > staged ? work_end : staged;
    const uint32_t factors = probs + heads * ATTENTION_GROUP_KEYS * ATTENTION_HALF_BYTES;
    const uint32_t indices = factors + heads * (uint32_t)sizeof(float);
    return {0u, tile, partials, probs, factors, indices,
            indices + (uint32_t)stages * ATTENTION_GROUP_KEYS * (uint32_t)sizeof(int32_t)};
}

/* The place of 16-byte chunk `chunk` of a key's row in a stage or tile: chunks of 8 rows in a row of the banks. */
__host__ __device__ constexpr uint32_t attention_swizzled(uint32_t key, uint32_t chunk)
{
    return chunk ^ (key & ATTENTION_SWIZZLE_MASK);
}

/* ---- the instructions ---- */

__device__ __forceinline__ uint32_t attention_shared_address(const void *place)
{
    return (uint32_t)__cvta_generic_to_shared(place);
}

/* 16 bytes from global to shared memory without the registers, outside the L1. */
__device__ __forceinline__ void attention_copy_chunk(void *place, const void *source)
{
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" ::"r"(attention_shared_address(place)), "l"(source),
                 "n"((int)ATTENTION_CHUNK_BYTES)
                 : "memory");
}

__device__ __forceinline__ void attention_commit_copies(void) { asm volatile("cp.async.commit_group;\n" ::: "memory"); }

template <int pending>
__device__ __forceinline__ void attention_wait_copies(void)
{
    asm volatile("cp.async.wait_group %0;\n" ::"n"(pending) : "memory");
}

/* Four 8 x 8 matrices of 16-bit elements; lanes 8 m .. 8 m + 7 give the rows of matrix m, register m receives row g,
 * elements 2 t and 2 t + 1 of it (transposed: rows 2 t and 2 t + 1, element g). */
#define ATTENTION_LOAD_MATRICES(function_name, instruction)                                                      \
    __device__ __forceinline__ void function_name(uint32_t registers[ATTENTION_REGISTERS_A], const void *place) \
    {                                                                                                            \
        asm volatile(instruction " {%0, %1, %2, %3}, [%4];\n"                                                   \
                     : "=r"(registers[0]), "=r"(registers[1]), "=r"(registers[2]), "=r"(registers[3])            \
                     : "r"(attention_shared_address(place))                                                      \
                     : "memory");                                                                                \
    }
ATTENTION_LOAD_MATRICES(attention_load_matrices, "ldmatrix.sync.aligned.m8n8.x4.shared.b16")
ATTENTION_LOAD_MATRICES(attention_load_matrices_transposed, "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16")

__device__ __forceinline__ void attention_load_two_matrices(uint32_t registers[ATTENTION_REGISTERS_B],
                                                            const void *place)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                 : "=r"(registers[0]), "=r"(registers[1])
                 : "r"(attention_shared_address(place))
                 : "memory");
}

/* sums += A B: A 16 rows by 16 (four registers), B 16 by 8 columns (two), the lane's four results. */
#define ATTENTION_TENSOR_CORE(function_name, types)                                                                  \
    __device__ __forceinline__ void function_name(float sums[ATTENTION_ELEMENTS],                                   \
                                                  const uint32_t a[ATTENTION_REGISTERS_A], uint32_t b_first,         \
                                                  uint32_t b_second)                                                 \
    {                                                                                                                \
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32." types ".f32 {%0, %1, %2, %3}, {%4, %5, %6, %7},"       \
                     " {%8, %9}, {%10, %11, %12, %13};\n"                                                            \
                     : "=f"(sums[0]), "=f"(sums[1]), "=f"(sums[2]), "=f"(sums[3])                                    \
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b_first), "r"(b_second), "f"(sums[0]),        \
                       "f"(sums[1]), "f"(sums[2]), "f"(sums[3]));                                                    \
    }
ATTENTION_TENSOR_CORE(attention_core_f16, "f16.f16")
ATTENTION_TENSOR_CORE(attention_core_bf16, "bf16.bf16")

/* ---- codes and scales ---- */

/* Bytes 0 and 2 of a word of codes as the low and high fp16 half of a register (an E4M3 code's value times 2^-8,
 * an E2M1 code's times 2^-14, subnormal codes included); then bytes 1 and 3. Placement computes nothing. */
template <int format>
__device__ __forceinline__ uint32_t attention_place_even(uint32_t word)
{
    return format == ATTENTION_FORMAT_E2M1
               ? ((word & ATTENTION_E2M1_EVEN_SIGN) << ATTENTION_E2M1_EVEN_SIGN_SHIFT) |
                     ((word & ATTENTION_E2M1_EVEN_MAGNITUDE) << ATTENTION_E2M1_EVEN_SHIFT)
               : ((word & ATTENTION_E4M3_EVEN_SIGN) << ATTENTION_E4M3_EVEN_SIGN_SHIFT) |
                     ((word & ATTENTION_E4M3_EVEN_MAGNITUDE) << ATTENTION_E4M3_EVEN_SHIFT);
}

template <int format>
__device__ __forceinline__ uint32_t attention_place_odd(uint32_t word)
{
    return format == ATTENTION_FORMAT_E2M1
               ? ((word & ATTENTION_E2M1_ODD_SIGN) << ATTENTION_E2M1_ODD_SIGN_SHIFT) |
                     ((word & ATTENTION_E2M1_ODD_MAGNITUDE) << ATTENTION_E2M1_ODD_SHIFT)
               : (word & ATTENTION_E4M3_ODD_SIGN) | ((word & ATTENTION_E4M3_ODD_MAGNITUDE) >> ATTENTION_E4M3_ODD_SHIFT);
}

/* The scale of a block times the inverse of its codes' placement, in fp32: an E8M0 code c gives 2^(c - 127 + 8)
 * (c at most 246: a scale of the reference is at most 2^119); an E4M3 code its value times 2^14, through a float of
 * its magnitude bits (value times 2^-120, denormal for a subnormal code) and two exact products. */
template <int format>
__device__ __forceinline__ float attention_scale_factor(uint32_t code)
{
    if (format == ATTENTION_FORMAT_POW2) {
        return __uint_as_float((code + ATTENTION_FP16_PLACEMENT) << ATTENTION_FLOAT_EXPONENT_SHIFT);
    }
    const uint32_t bits = ((code & ATTENTION_E4M3_SIGN) << (ATTENTION_HALF_BITS + ATTENTION_BYTE_BITS)) |
                          ((code & ATTENTION_E4M3_MAGNITUDE_MASK) << ATTENTION_E4M3_FLOAT_SHIFT);
    return __uint_as_float(bits) * ATTENTION_E4M3_TO_FLOAT * ATTENTION_E2M1_INVERSE_PLACEMENT;
}

__device__ __forceinline__ float attention_half_value(uint32_t bits)
{
    return __half2float(__ushort_as_half((unsigned short)(bits & UINT16_MAX)));
}

__device__ __forceinline__ uint32_t attention_bfloat16_pair(float low, float high)
{
    const __nv_bfloat162 pair = __floats2bfloat162_rn(low, high);
    uint32_t bits = 0;
    memcpy(&bits, &pair, sizeof(bits));
    return bits;
}

/* Four values of one key, from its two placed registers (dims 4 t, 4 t + 2 and 4 t + 1, 4 t + 3 of a 16-dim chunk),
 * decoded to bfloat16 into the tile [key][512] in value order (the stage's swizzle): exact, see the contract. */
__device__ __forceinline__ void attention_store_values(unsigned char *tile, uint32_t key, uint32_t chunk, uint32_t t,
                                                       uint32_t even, uint32_t odd, float factor)
{
    const uint32_t low = attention_bfloat16_pair(attention_half_value(even) * factor,
                                                 attention_half_value(odd) * factor);
    const uint32_t high = attention_bfloat16_pair(attention_half_value(even >> ATTENTION_HALF_BITS) * factor,
                                                  attention_half_value(odd >> ATTENTION_HALF_BITS) * factor);
    const uint32_t tile_chunk = chunk * ATTENTION_TILE_CHUNKS_PER_STEP + t / ATTENTION_TILE_CHUNKS_PER_STEP;
    *(uint2 *)(tile + key * ATTENTION_FP16_ROW_BYTES + attention_swizzled(key, tile_chunk) * ATTENTION_CHUNK_BYTES +
               t % ATTENTION_TILE_CHUNKS_PER_STEP * sizeof(uint2)) = make_uint2(low, high);
}

/* ---- the pieces of a group ---- */

/* Where 16-byte chunk `chunk` of a key's row lies in a stage: an fp16 row's chunks swizzled; a code row's 32 chunks
 * swizzled and its scales behind the codes of all 16 keys, 32 bytes a key. */
template <int kind>
__device__ __forceinline__ unsigned char *attention_stage_place(unsigned char *stage, uint32_t key, uint32_t chunk)
{
    if (kind == ATTENTION_FP16) {
        return stage + key * ATTENTION_FP16_ROW_BYTES + attention_swizzled(key, chunk) * ATTENTION_CHUNK_BYTES;
    }
    return chunk < ATTENTION_CODE_CHUNKS
               ? stage + key * ATTENTION_CODE_BYTES + attention_swizzled(key, chunk) * ATTENTION_CHUNK_BYTES
               : stage + ATTENTION_GROUP_KEYS * ATTENTION_CODE_BYTES + key * ATTENTION_STAGE_SCALE_BYTES +
                     (chunk - ATTENTION_CODE_CHUNKS) * ATTENTION_CHUNK_BYTES;
}

/* The copies of one group into its stage. A thread takes the key threadIdx.x / (threads / 16) and every (threads /
 * 16)th chunk of that key's row from threadIdx.x % (threads / 16) on; a key of index -1 gets zeros (no NaN or stale
 * byte can reach P x V). The first thread of a key puts its index among the stage's indices. With
 * ATTENTION_CORE_CHECKS an index past its plane is reported in the status word and taken as -1. */
template <int kind, uint32_t threads>
__device__ __forceinline__ void attention_issue(const struct attention_core_arguments *arguments, unsigned char *stage,
                                                int32_t *stage_indices, uint32_t group,
                                                int32_t index ATTENTION_CHECKS_POINTER)
{
    constexpr uint32_t threads_per_key = threads / ATTENTION_GROUP_KEYS;
    const uint32_t key = threadIdx.x / threads_per_key, first_chunk = threadIdx.x % threads_per_key;
    const uint32_t row_bytes = attention_row_bytes(attention_format_of(kind, group));
    const unsigned char *plane = group < ATTENTION_WINDOW_GROUPS ? arguments->window_rows : arguments->pool_rows;
#if ATTENTION_CORE_CHECKS
    const uint64_t plane_rows = group < ATTENTION_WINDOW_GROUPS ? checks->window_row_count : checks->pool_row_count;
    if (index < -1 || (index >= 0 && (uint64_t)index >= plane_rows)) {
        *checks->status_word = ((unsigned long long)ATTENTION_FAILED_INDEX << ATTENTION_STATUS_SHIFT) | blockIdx.x;
        index = -1;
    }
#endif
    const unsigned char *source = plane + (uint64_t)(index < 0 ? 0 : index) * row_bytes;
    for (uint32_t chunk = first_chunk; chunk < row_bytes / ATTENTION_CHUNK_BYTES; chunk += threads_per_key) {
        unsigned char *place = attention_stage_place<kind>(stage, key, chunk);
        if (index < 0) {
            *(uint4 *)place = make_uint4(0u, 0u, 0u, 0u);
        } else {
            attention_copy_chunk(place, source + chunk * ATTENTION_CHUNK_BYTES);
        }
    }
    if (first_chunk == 0) {
        stage_indices[key] = index;
    }
}

/* The lane's queries of its slice for head g of a tile, as operand B of each step of 16 dims: FP16 dims 2 t, 2 t + 1
 * and 2 t + 8, 2 t + 9; 8-bit kinds dims 4 t, 4 t + 2 and 4 t + 1, 4 t + 3 (the order a code word is placed in). */
template <int kind>
__device__ __forceinline__ void attention_load_queries(const uint16_t *slice_queries, uint32_t t,
                                                       uint32_t queries[ATTENTION_STEPS][ATTENTION_REGISTERS_B])
{
#pragma unroll
    for (uint32_t step = 0; step < ATTENTION_STEPS; step++) {
        const uint16_t *step_queries = slice_queries + step * ATTENTION_STEP_DIMS;
        if (kind == ATTENTION_FP16) {
            queries[step][0] = *(const uint32_t *)(step_queries + t * ATTENTION_REGISTERS_B);
            queries[step][1] = *(const uint32_t *)(step_queries + ATTENTION_CHUNK_HALVES + t * ATTENTION_REGISTERS_B);
        } else {
            const uint2 four = *(const uint2 *)(step_queries + t * ATTENTION_QUAD);
            queries[step][0] = __byte_perm(four.x, four.y, ATTENTION_LOW_HALVES);
            queries[step][1] = __byte_perm(four.x, four.y, ATTENTION_HIGH_HALVES);
        }
    }
}

/* FP16: the warp's scores of one group over its slice, per head tile one chain from +0 (keys g, g + 8 by heads
 * 2 t, 2 t + 1). Operand A, keys by dims, from the stage: matrices (keys 0-7, 8-15) x (dims of the step's two
 * chunks). */
template <int warp_tiles>
__device__ __forceinline__ void attention_scores_fp16(const unsigned char *stage, uint32_t slice, uint32_t lane,
                                                      const uint32_t queries[][ATTENTION_STEPS][ATTENTION_REGISTERS_B],
                                                      float sums[][ATTENTION_ELEMENTS])
{
    const uint32_t matrix = lane / ATTENTION_MATRIX_ROWS;
    const uint32_t key = lane % ATTENTION_MATRIX_ROWS + matrix % ATTENTION_REGISTERS_B * ATTENTION_HALF_GROUP;
#pragma unroll
    for (uint32_t step = 0; step < ATTENTION_STEPS; step++) {
        const uint32_t chunk = (slice * ATTENTION_SLICE_DIMS + step * ATTENTION_STEP_DIMS) / ATTENTION_CHUNK_HALVES +
                               matrix / ATTENTION_REGISTERS_B;
        uint32_t operand[ATTENTION_REGISTERS_A];
        attention_load_matrices(operand, stage + key * ATTENTION_FP16_ROW_BYTES +
                                             attention_swizzled(key, chunk) * ATTENTION_CHUNK_BYTES);
#pragma unroll
        for (int tile = 0; tile < warp_tiles; tile++) {
            attention_core_f16(sums[tile], operand, queries[tile][step][0], queries[tile][step][1]);
        }
    }
}

/* 8-bit rows: the same scores from codes, a scale block's products summed from +0 and then taken into the chain by
 * fmaf with the block's factor; on the way every code of the slice is decoded into the tile for P x V. One load of
 * four matrices is a pair of 16-dim chunks: (keys 0-7, 8-15) x (chunk, chunk + 1), a word of four codes a lane. */
template <int format, int warp_tiles>
__device__ __forceinline__ void attention_scores_codes(const unsigned char *stage, unsigned char *tile, uint32_t slice,
                                                       uint32_t lane,
                                                       const uint32_t queries[][ATTENTION_STEPS][ATTENTION_REGISTERS_B],
                                                       float chains[][ATTENTION_ELEMENTS])
{
    const uint32_t matrix = lane / ATTENTION_MATRIX_ROWS, g = lane / ATTENTION_QUAD, t = lane % ATTENTION_QUAD;
    const uint32_t key = lane % ATTENTION_MATRIX_ROWS + matrix % ATTENTION_REGISTERS_B * ATTENTION_HALF_GROUP;
    const unsigned char *scales = stage + ATTENTION_GROUP_KEYS * ATTENTION_CODE_BYTES;
    constexpr uint32_t block_steps =
        format == ATTENTION_FORMAT_POW2 ? ATTENTION_POW2_BLOCK_DIMS / ATTENTION_STEP_DIMS : 1u;   /* a scale's steps */
    float block_sums[warp_tiles][ATTENTION_ELEMENTS] = {};
#pragma unroll
    for (uint32_t pair = 0; pair < ATTENTION_STEPS / ATTENTION_REGISTERS_B; pair++) {
        const uint32_t first_step = pair * ATTENTION_REGISTERS_B;
        uint32_t words[ATTENTION_REGISTERS_A];   /* key g and key g + 8 of the first chunk, then of the second */
        const uint32_t chunk = slice * ATTENTION_STEPS + first_step + matrix / ATTENTION_REGISTERS_B;
        attention_load_matrices(words, stage + key * ATTENTION_CODE_BYTES +
                                           attention_swizzled(key, chunk) * ATTENTION_CHUNK_BYTES);
#pragma unroll
        for (uint32_t half = 0; half < ATTENTION_REGISTERS_B; half++) {
            const uint32_t step = first_step + half, code_chunk = slice * ATTENTION_STEPS + step;
            const uint32_t block = code_chunk / block_steps;
            const uint32_t low_word = words[half * ATTENTION_REGISTERS_B];
            const uint32_t high_word = words[half * ATTENTION_REGISTERS_B + 1u];
            const uint32_t operand[ATTENTION_REGISTERS_A] = {
                attention_place_even<format>(low_word), attention_place_even<format>(high_word),
                attention_place_odd<format>(low_word), attention_place_odd<format>(high_word)};
            const float low_factor = attention_scale_factor<format>(scales[g * ATTENTION_STAGE_SCALE_BYTES + block]);
            const float high_factor = attention_scale_factor<format>(
                scales[(g + ATTENTION_HALF_GROUP) * ATTENTION_STAGE_SCALE_BYTES + block]);
#pragma unroll
            for (int head_tile = 0; head_tile < warp_tiles; head_tile++) {
                if (step % block_steps == 0) {
#pragma unroll
                    for (uint32_t element = 0; element < ATTENTION_ELEMENTS; element++) {
                        block_sums[head_tile][element] = 0.0f;
                    }
                }
                attention_core_f16(block_sums[head_tile], operand, queries[head_tile][step][0],
                                   queries[head_tile][step][1]);
#pragma unroll
                for (uint32_t element = 0; step % block_steps == block_steps - 1u && element < ATTENTION_ELEMENTS;
                     element++) {
                    chains[head_tile][element] =
                        fmaf(block_sums[head_tile][element],
                             element < ATTENTION_REGISTERS_B ? low_factor : high_factor, chains[head_tile][element]);
                }
            }
            attention_store_values(tile, g, code_chunk, t, operand[0], operand[ATTENTION_REGISTERS_B], low_factor);
            attention_store_values(tile, g + ATTENTION_HALF_GROUP, code_chunk, t, operand[1],
                                   operand[ATTENTION_REGISTERS_B + 1u], high_factor);
        }
    }
}

/* The rise of the maximum that rescales, by the rule a kernel is stamped with. A measurement adds a rule of its own by
 * an explicit specialization, declared before its stamps. */
template <int step_rule>
__device__ __forceinline__ float attention_rescale_step(void)
{
    return step_rule == ATTENTION_STEP_ALWAYS ? 0.0f : ATTENTION_CONTRACT_STEP;
}

/* The softmax of one group for one head, by the 8 lanes of that head: lane j takes keys j and j + 8. Returns 1 where
 * the head's maximum rose by more than the step (O and the sum are then rescaled by factors[head]). */
template <int kind, int step_rule>
__device__ __forceinline__ int attention_softmax(const float *partials, uint32_t partial_row, const int32_t *indices,
                                                 uint16_t *probs, float *factors, uint32_t head, uint32_t head_lane,
                                                 float *maximum, float *total)
{
    float scores[ATTENTION_REGISTERS_B];
#pragma unroll
    for (uint32_t half = 0; half < ATTENTION_REGISTERS_B; half++) {
        const uint32_t key = head_lane + half * ATTENTION_HALF_GROUP;
        float sum = 0.0f;
#pragma unroll
        for (uint32_t slice = 0; slice < ATTENTION_SLICES; slice++) {
            sum += partials[(slice * ATTENTION_GROUP_KEYS + key) * partial_row + head];
        }
        scores[half] = indices[key] < 0 ? -INFINITY : sum * ATTENTION_SOFTMAX_SCALE;
    }
    float largest = fmaxf(scores[0], scores[1]);
#pragma unroll
    for (uint32_t distance = 1; distance < ATTENTION_SOFTMAX_LANES; distance *= ATTENTION_REGISTERS_B) {
        largest = fmaxf(largest, __shfl_xor_sync(ATTENTION_EVERY_LANE, largest, (int)distance));
    }
    const int rescaled = largest > *maximum + attention_rescale_step<step_rule>();
    const float factor = rescaled ? expf(*maximum - largest) : 1.0f;
    *maximum = rescaled ? largest : *maximum;
    float group_sum = 0.0f;
#pragma unroll
    for (uint32_t half = 0; half < ATTENTION_REGISTERS_B; half++) {
        const float probability = expf(scores[half] - *maximum);
        const uint32_t key = head_lane + half * ATTENTION_HALF_GROUP;
        const uint32_t chunk = half ^ ((head >> ATTENTION_PROBS_SWIZZLE_SHIFT) & 1u);
        probs[head * ATTENTION_GROUP_KEYS + chunk * ATTENTION_CHUNK_HALVES + key % ATTENTION_HALF_GROUP] =
            kind == ATTENTION_FP16 ? __half_as_ushort(__float2half_rn(probability))
                                   : __bfloat16_as_ushort(__float2bfloat16_rn(probability));
        group_sum = half == 0 ? probability : group_sum + probability;
    }
#pragma unroll
    for (uint32_t distance = 1; distance < ATTENTION_SOFTMAX_LANES; distance *= ATTENTION_REGISTERS_B) {
        group_sum += __shfl_xor_sync(ATTENTION_EVERY_LANE, group_sum, (int)distance);
    }
    *total = *total * factor + group_sum;
    if (head_lane == 0) {
        factors[head] = factor;
    }
    return rescaled;
}

/* O^T += V^T P^T over the warp's slice: operand A (dims by keys) from the stage or the decoded tile, transposed;
 * operand B (keys by heads) from P [head][key], one load for the warp's tiles. */
template <int kind, int warp_tiles>
__device__ __forceinline__ void attention_values(const unsigned char *rows, const uint16_t *probs, uint32_t first_tile,
                                                 uint32_t slice, uint32_t lane,
                                                 float values[ATTENTION_STEPS][warp_tiles][ATTENTION_ELEMENTS])
{
    const uint32_t matrix = lane / ATTENTION_MATRIX_ROWS, row = lane % ATTENTION_MATRIX_ROWS;
    const uint32_t tile = first_tile + (warp_tiles > 1 ? matrix / ATTENTION_REGISTERS_B : 0u);
    const uint32_t head = tile * ATTENTION_TILE_HEADS + row;
    const uint32_t chunk = (matrix % ATTENTION_REGISTERS_B) ^ ((head >> ATTENTION_PROBS_SWIZZLE_SHIFT) & 1u);
    const uint16_t *head_probs = probs + head * ATTENTION_GROUP_KEYS + chunk * ATTENTION_CHUNK_HALVES;
    uint32_t operand_b[ATTENTION_REGISTERS_A];
    if (warp_tiles > 1) {
        attention_load_matrices(operand_b, head_probs);
    } else {
        attention_load_two_matrices(operand_b, head_probs);
    }
    const uint32_t key = row + matrix / ATTENTION_REGISTERS_B * ATTENTION_HALF_GROUP;
#pragma unroll
    for (uint32_t step = 0; step < ATTENTION_STEPS; step++) {
        const uint32_t place = (slice * ATTENTION_SLICE_DIMS + step * ATTENTION_STEP_DIMS) / ATTENTION_CHUNK_HALVES +
                               matrix % ATTENTION_REGISTERS_B;
        uint32_t operand_a[ATTENTION_REGISTERS_A];
        attention_load_matrices_transposed(operand_a, rows + key * ATTENTION_FP16_ROW_BYTES +
                                                          attention_swizzled(key, place) * ATTENTION_CHUNK_BYTES);
#pragma unroll
        for (int tile = 0; tile < warp_tiles; tile++) {
            if (kind == ATTENTION_FP16) {
                attention_core_f16(values[step][tile], operand_a, operand_b[tile * ATTENTION_REGISTERS_B],
                                   operand_b[tile * ATTENTION_REGISTERS_B + 1]);
            } else {
                attention_core_bf16(values[step][tile], operand_a, operand_b[tile * ATTENTION_REGISTERS_B],
                                    operand_b[tile * ATTENTION_REGISTERS_B + 1]);
            }
        }
    }
}

/* O and the factors of the heads of a lane's results (heads 2 t and 2 t + 1 of each tile): a product each. */
template <int warp_tiles>
__device__ __forceinline__ void attention_scale_values(float values[ATTENTION_STEPS][warp_tiles][ATTENTION_ELEMENTS],
                                                       const float *factors, uint32_t first_tile, uint32_t t)
{
#pragma unroll                                   /* O stays in registers: every index a constant */
    for (int tile = 0; tile < warp_tiles; tile++) {
        const float *pair = factors + (first_tile + (uint32_t)tile) * ATTENTION_TILE_HEADS + t * ATTENTION_REGISTERS_B;
#pragma unroll
        for (uint32_t step = 0; step < ATTENTION_STEPS; step++) {
#pragma unroll
            for (uint32_t element = 0; element < ATTENTION_ELEMENTS; element++) {
                values[step][tile][element] *= pair[element % ATTENTION_REGISTERS_B];
            }
        }
    }
}

/* The epilogue of a warp: O over its denominator, the de-rotation of dims 448 .. 511 (a pair's partner is lane
 * 4 (g ^ 1) + t), one rounding to bfloat16, and through the warp's staging rows 16-byte stores of whole rows. */
template <int warp_tiles>
__device__ __forceinline__ void attention_finish(float values[ATTENTION_STEPS][warp_tiles][ATTENTION_ELEMENTS],
                                                 const float *denominators, const float *rotation, uint32_t first_tile,
                                                 uint32_t slice, uint32_t lane, uint16_t *staged, uint16_t *output)
{
    const uint32_t g = lane / ATTENTION_QUAD, t = lane % ATTENTION_QUAD;
#pragma unroll
    for (int tile = 0; tile < warp_tiles; tile++) {
        const float *pair =
            denominators + (first_tile + (uint32_t)tile) * ATTENTION_TILE_HEADS + t * ATTENTION_REGISTERS_B;
#pragma unroll
        for (uint32_t step = 0; step < ATTENTION_STEPS; step++) {
#pragma unroll
            for (uint32_t element = 0; element < ATTENTION_ELEMENTS; element++) {
                float *value = &values[step][tile][element];
                const uint32_t dim = slice * ATTENTION_SLICE_DIMS + step * ATTENTION_STEP_DIMS + g +
                                     element / ATTENTION_REGISTERS_B * ATTENTION_HALF_GROUP;
                *value = *value / pair[element % ATTENTION_REGISTERS_B];
                if (slice == ATTENTION_SLICES - 1u && step * ATTENTION_STEP_DIMS >= ATTENTION_ROPE_FIRST_DIM -
                                                                                     slice * ATTENTION_SLICE_DIMS) {
                    const float partner = __shfl_xor_sync(ATTENTION_EVERY_LANE, *value, ATTENTION_QUAD);
                    const float *angle = rotation + (dim - ATTENTION_ROPE_FIRST_DIM) / ATTENTION_REGISTERS_B *
                                                        ATTENTION_ROPE_VALUES;
                    const float own = *value * angle[0], other = partner * angle[1];
                    *value = dim % ATTENTION_REGISTERS_B == 0 ? own + other : own - other;
                }
            }
        }
    }
#pragma unroll
    for (int tile = 0; tile < warp_tiles; tile++) {
#pragma unroll
        for (uint32_t step = 0; step < ATTENTION_STEPS; step++) {
#pragma unroll
            for (uint32_t element = 0; element < ATTENTION_ELEMENTS; element++) {
                staged[(t * ATTENTION_REGISTERS_B + element % ATTENTION_REGISTERS_B) * ATTENTION_STAGED_HALVES +
                       step * ATTENTION_STEP_DIMS + g + element / ATTENTION_REGISTERS_B * ATTENTION_HALF_GROUP] =
                    __bfloat16_as_ushort(__float2bfloat16_rn(values[step][tile][element]));
            }
        }
        __syncwarp();
        constexpr uint32_t row_chunks = ATTENTION_SLICE_DIMS / ATTENTION_CHUNK_HALVES;
        for (uint32_t chunk = lane; chunk < ATTENTION_TILE_HEADS * row_chunks; chunk += ATTENTION_LANES) {
            const uint32_t head = chunk / row_chunks, piece = chunk % row_chunks;
            *(uint4 *)(output + ((first_tile + (uint32_t)tile) * ATTENTION_TILE_HEADS + head) * ATTENTION_DIMS +
                       slice * ATTENTION_SLICE_DIMS + piece * ATTENTION_CHUNK_HALVES) =
                *(const uint4 *)(staged + head * ATTENTION_STAGED_HALVES + piece * ATTENTION_CHUNK_HALVES);
        }
        __syncwarp();
    }
}

/* ---- one block: a row's heads of the block, over the 40 groups ---- */

template <int kind, int head_tiles, int stages, int step_rule>
__device__ __forceinline__ void attention_core_block(const struct attention_core_arguments &arguments,
                                                     unsigned char *shared ATTENTION_CHECKS_PARAMETER)
{
    constexpr uint32_t threads = attention_threads(head_tiles), heads = (uint32_t)head_tiles * ATTENTION_TILE_HEADS;
    constexpr int warp_tiles = head_tiles < ATTENTION_WARP_TILES ? head_tiles : ATTENTION_WARP_TILES;
    constexpr struct attention_layout layout = attention_layout_of(kind, head_tiles, stages);
    constexpr uint32_t stage_bytes = attention_stage_bytes(kind), partial_row = heads + ATTENTION_PARTIAL_PAD;
    const uint32_t warp = threadIdx.x / ATTENTION_LANES, lane = threadIdx.x % ATTENTION_LANES;
    const uint32_t g = lane / ATTENTION_QUAD, t = lane % ATTENTION_QUAD;
    const uint32_t slice = warp % ATTENTION_SLICES, first_tile = warp / ATTENTION_SLICES * ATTENTION_WARP_TILES;
    const uint32_t row = blockIdx.x, first_head = blockIdx.y * heads;
    const uint32_t softmax_head = threadIdx.x / ATTENTION_SOFTMAX_LANES;
    const uint32_t head_lane = threadIdx.x % ATTENTION_SOFTMAX_LANES;
    const uint32_t copy_key = threadIdx.x / (threads / ATTENTION_GROUP_KEYS);
    float *partials = (float *)(shared + layout.partials), *factors = (float *)(shared + layout.factors);
    uint16_t *probs = (uint16_t *)(shared + layout.probs);
    int32_t *indices = (int32_t *)(shared + layout.indices);
#if ATTENTION_CORE_CHECKS
    if (blockDim.x != threads || gridDim.y * heads != ATTENTION_HEADS) {          /* the same in every thread */
        *checks.status_word = ((unsigned long long)ATTENTION_FAILED_GEOMETRY << ATTENTION_STATUS_SHIFT) | row;
        return;
    }
#endif
    uint32_t queries[warp_tiles][ATTENTION_STEPS][ATTENTION_REGISTERS_B];
    const uint16_t *row_queries = arguments.queries + ((uint64_t)row * ATTENTION_HEADS + first_head) * ATTENTION_DIMS;
#pragma unroll
    for (int tile = 0; tile < warp_tiles; tile++) {
        attention_load_queries<kind>(row_queries + ((first_tile + (uint32_t)tile) * ATTENTION_TILE_HEADS + g) *
                                                       ATTENTION_DIMS + slice * ATTENTION_SLICE_DIMS, t, queries[tile]);
    }
    const int32_t *row_indices = arguments.indices + (uint64_t)row * ATTENTION_KEYS;
    for (uint32_t group = 0; group + 1u < (uint32_t)stages; group++) {          /* the ring's first stages */
        attention_issue<kind, threads>(&arguments, shared + layout.ring + group * stage_bytes,
                                       indices + group * ATTENTION_GROUP_KEYS, group,
                                       row_indices[group * ATTENTION_GROUP_KEYS + copy_key] ATTENTION_CHECKS_ADDRESS);
        attention_commit_copies();
    }
    int32_t index_ahead = row_indices[(stages - 1) * ATTENTION_GROUP_KEYS + copy_key];
    float values[ATTENTION_STEPS][warp_tiles][ATTENTION_ELEMENTS] = {};
    float maximum = ATTENTION_SCORE_FLOOR, total = 0.0f;
    for (uint32_t group = 0; group < ATTENTION_GROUPS; group++) {
        const uint32_t next = group + (uint32_t)stages - 1u;                     /* the group the ring takes next */
        if (stages == 1 && group > 0) {
            __syncthreads();                     /* the one stage is free once every warp has read it */
        }
        if (stages == 1) {
            attention_issue<kind, threads>(&arguments, shared + layout.ring, indices, group,
                                           index_ahead ATTENTION_CHECKS_ADDRESS);
            attention_commit_copies();
        }
        attention_wait_copies<(stages > 1 ? stages - 1 : 1) - 1>();      /* this group done; those after it may wait */
        __syncthreads();
        if (stages > 1 && next < ATTENTION_GROUPS) {
            attention_issue<kind, threads>(&arguments, shared + layout.ring + next % stages * stage_bytes,
                                           indices + next % stages * ATTENTION_GROUP_KEYS, next,
                                           index_ahead ATTENTION_CHECKS_ADDRESS);
        }
        if (stages > 1) {
            attention_commit_copies();           /* empty at the ring's tail: the wait counts it */
        }
        if (next + 1u < ATTENTION_GROUPS) {
            index_ahead = row_indices[(next + 1u) * ATTENTION_GROUP_KEYS + copy_key];
        }
        const unsigned char *stage = shared + layout.ring + group % stages * stage_bytes;
        float sums[warp_tiles][ATTENTION_ELEMENTS] = {};
        if constexpr (kind == ATTENTION_FP16) {
            attention_scores_fp16<warp_tiles>(stage, slice, lane, queries, sums);
        } else if (attention_format_of(kind, group) == ATTENTION_FORMAT_E2M1) {
            attention_scores_codes<ATTENTION_FORMAT_E2M1, warp_tiles>(stage, shared + layout.tile, slice, lane, queries,
                                                                      sums);
        } else {
            attention_scores_codes<ATTENTION_FORMAT_POW2, warp_tiles>(stage, shared + layout.tile, slice, lane, queries,
                                                                      sums);
        }
#pragma unroll
        for (int tile = 0; tile < warp_tiles; tile++) {
            float *slice_partials = partials + slice * ATTENTION_GROUP_KEYS * partial_row +
                                    (first_tile + (uint32_t)tile) * ATTENTION_TILE_HEADS + t * ATTENTION_REGISTERS_B;
            *(float2 *)(slice_partials + g * partial_row) = make_float2(sums[tile][0], sums[tile][1]);
            *(float2 *)(slice_partials + (g + ATTENTION_HALF_GROUP) * partial_row) =
                make_float2(sums[tile][ATTENTION_REGISTERS_B], sums[tile][ATTENTION_REGISTERS_B + 1]);
        }
        __syncthreads();
        int rescaled = 0;
        if (softmax_head < heads) {
            rescaled = attention_softmax<kind, step_rule>(partials, partial_row, indices + group % stages *
                                                          ATTENTION_GROUP_KEYS, probs, factors, softmax_head,
                                                          head_lane, &maximum, &total);
        }
        if (__syncthreads_or(rescaled)) {
            attention_scale_values<warp_tiles>(values, factors, first_tile, t);
        }
        attention_values<kind, warp_tiles>(kind == ATTENTION_FP16 ? stage : shared + layout.tile, probs, first_tile,
                                           slice, lane, values);
    }
    __syncthreads();                             /* every warp is past its last P x V */
    if (softmax_head < heads && head_lane == 0) {
        factors[softmax_head] = total + expf(arguments.sinks[first_head + softmax_head] - maximum);
    }
    __syncthreads();
    attention_finish<warp_tiles>(values, factors, arguments.rotation + (uint64_t)row * ATTENTION_ROPE_PAIRS *
                                                      ATTENTION_ROPE_VALUES, first_tile, slice, lane,
                                 (uint16_t *)shared + warp * ATTENTION_TILE_HEADS * ATTENTION_STAGED_HALVES,
                                 arguments.output + ((uint64_t)row * ATTENTION_HEADS + first_head) * ATTENTION_DIMS);
}

/* ---- stamping: one kernel per choice, the body the same function; `bound` is empty or a __launch_bounds__ ---- */

#define ATTENTION_CORE_KERNEL(kernel_name, bound, kind, head_tiles, stages, step_rule)                              \
    __global__ void bound kernel_name(struct attention_core_arguments arguments ATTENTION_CHECKS_PARAMETER)         \
    {                                                                                                               \
        extern __shared__ __align__(ATTENTION_CHUNK_BYTES) unsigned char attention_shared[];                       \
        attention_core_block<kind, head_tiles, stages, step_rule>(arguments,                                       \
                                                                  attention_shared ATTENTION_CHECKS_ARGUMENT);      \
    }

#endif /* ATTENTION_CORE_KERNELS_CUH */
