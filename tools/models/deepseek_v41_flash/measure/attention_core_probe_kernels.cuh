/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * attention_core_probe_kernels.cuh: what package 11b's probe needs beside the kernels of attention_core_kernels.cuh
 * and that exists for the measurement alone (research/speed/ROUND_1_attention.md, "attention_core"): the kernels the
 * round stamps, the bounds a kernel is held against (gather_only, tensor_only), and the synthetic state, lists and
 * compact copies (with their host helpers: the check's hand-made row, the rows' angles, a list's distinct pool rows).
 * attention_core_probe.cu holds every check. Measurement tool: not part of the engine.
 *
 * The planes (one row a position; row r at r times the format's row bytes):
 *   - fp16: 512 values of attention_plane_value; the window rows and pool entries of FP16, 1,024 B a row.
 *   - pow2: the same values as the reference's act_quant leaves a window row (model.py:707, kernel.py act_quant with
 *     scales rounded up to a power of two): per 32 dims m = max(largest magnitude, 1e-4), the E8M0 code c with
 *     2^(c - 127) the least power of two not under m x (1/448) in fp32, each value over 2^(c - 127), limited to 448,
 *     to the nearest E4M3 code (a tie to the even code, the sign kept also of zero); 512 codes, then 16 scales. The
 *     window rows of both 8-bit kinds and the pool entries of FRESH.
 *   - e2m1: as fp4_act_quant with E4M3 scales leaves a pool entry (model.py:760): per 16 dims m = max(largest
 *     magnitude, 6 x 2^-9), the scale s = m / 6 in fp32 to the nearest E4M3 number (limited to 448), each value
 *     over s in fp32, limited to 6, to the nearest E2M1 number (a tie to the even code, the sign kept); 512 codes in
 *     the low nibbles, then 32 scales. The pool entries of TRAINED.
 * The 8-bit planes are the reference's roundings of the fp16 plane's numbers: the kinds see the same rows. The
 * fills are the probe's own and are not checked: the host decodes what the card holds.
 *
 * A list (attention_list_kernel): row r's window rows are r .. r + 127 of the window plane from the step's window
 * pointer on (neighbouring rows share 127 of them); its pool entry e is a row of band e of the pool plane, by a hash
 * of (seed, r, e): ascending, distinct within a row, spread over the whole plane. Rows share pool rows only by chance
 * up to 256 rows (about 6% of a step's picks at 256); at 4,096 rows a step's 2 GiB of picks fall on a 1 GiB plane, two
 * on a row on average, and the rows of a launch read the same 16 bands a group at a time, so the L2 serves a share of
 * them. The probe's RATE counts each distinct row once.
 */
#ifndef ATTENTION_CORE_PROBE_KERNELS_CUH
#define ATTENTION_CORE_PROBE_KERNELS_CUH
#include "attention_core_kernels.cuh"

enum {
    ATTENTION_POW2_BLOCKS = 16, ATTENTION_E2M1_HALF_BLOCKS = 2,
    ATTENTION_VALUE_STEPS = 256, ATTENTION_VALUE_CENTRE = 128, ATTENTION_VALUE_UNIT_SHIFT = 7,
    ATTENTION_VALUE_KINDS = 5, ATTENTION_VALUE_SPREAD = 1, ATTENTION_VALUE_ZEROS = 2, ATTENTION_VALUE_TINY = 3,
    ATTENTION_SPREAD_PLACES = 18, ATTENTION_TINY_SHIFT = 15, ATTENTION_EIGHTHS_MASK = 7, ATTENTION_EIGHTHS = 8,
    ATTENTION_MIX_SHIFT_OUTER = 16, ATTENTION_MIX_SHIFT_INNER = 15,
    ATTENTION_E4M3_UNITS_EXPONENT = 130, ATTENTION_E4M3_CODE_BASE = 968,
    ATTENTION_E2M1_CODES = 8, ATTENTION_E2M1_SIGN = 8, ATTENTION_E2M1_IMPLICIT_HALVES = 2,
    ATTENTION_FLOAT_EXPONENT_FIELD = 255, ATTENTION_FLOAT_BIAS = 127,
    /* the check's poisoned rows: fp16 NaN; in the 8-bit planes the largest codes by the largest scales, which the
     * kernels place as finite numbers: E4M3 448 by 2^119 overflows a score, E2M1 6 by 448 outweighs every key */
    ATTENTION_POISON_FP16 = 0xFF, ATTENTION_POISON_E4M3 = 0x7E, ATTENTION_POISON_E8M0 = 246,
    ATTENTION_POISON_E2M1 = 0x7,
    /* the check's hand-made window row: E4M3 codes 1 .. 7 (all subnormal) of both signs by 2^(133 - 127) */
    ATTENTION_SUBNORMAL_CODES = 7, ATTENTION_SUBNORMAL_SCALE = 133,
    ATTENTION_DIMS_PER_FREQUENCY = 2,            /* a rotary frequency turns a pair of dims */
    ATTENTION_BASELINE_STAGES = 2, ATTENTION_DEEP_STAGES = 3,
    ATTENTION_ALL_TILES = ATTENTION_HEADS / ATTENTION_TILE_HEADS,       /* gather:shared: every head in one block */
    ATTENTION_STEP_BY_ROWS = ATTENTION_STEP_ALWAYS + 1,  /* break:group_by_rows: the step follows the launch's rows */
    ATTENTION_GATHER_SHARED_THREADS = 512, ATTENTION_TENSOR_GROUPS = ATTENTION_GROUPS,
    ATTENTION_TENSOR_V_SUM = 820, ATTENTION_TENSOR_TILE_SUM = 3,        /* tensor_only: 1 + .. + 40; 1 + 2 */
    ATTENTION_TENSOR_STEP_SUM = 44, ATTENTION_TENSOR_FIRST_NUMBER = 2,  /* 2 + .. + 9: never the scores' 1 */
    ATTENTION_FP16_EXPONENT_ONE = 0x0400         /* a doubling of an fp16 in its bits */
};
#define ATTENTION_MIX_FIRST 0x7feb352du
#define ATTENTION_MIX_SECOND 0x846ca68bu
#define ATTENTION_POW2_FLOOR 1.0e-4f             /* act_quant's floor of a block's largest magnitude */
#define ATTENTION_E4M3_LARGEST 448.0f
#define ATTENTION_E4M3_LARGEST_INVERSE (1.0f / 448.0f)
#define ATTENTION_E4M3_LEAST_NORMAL 0x1p-6f
#define ATTENTION_E2M1_LARGEST 6.0f
#define ATTENTION_E2M1_QUARTERS 4.0f
#define ATTENTION_E2M1_FLOOR (6.0f * 0x1p-9f)    /* fp4_act_quant's floor with E4M3 scales */
#define ATTENTION_FLOAT_MANTISSA_MASK 0x007FFFFFu
#define ATTENTION_FLOAT_SIGN_TO_BYTE 24
#define ATTENTION_ONE_PAIR 0x3C003C00u           /* two fp16 1.0: tensor_only's scores' operand A */
/* the angles of a rate-2 layer (model.py precompute_freqs_cis: YaRN over compress_rope_theta, config.json) */
#define ATTENTION_FIRST_POSITION 900000.0
#define ATTENTION_ROPE_BASE 160000.0
#define ATTENTION_ROPE_ORIGINAL 65536.0
#define ATTENTION_ROPE_FACTOR 16.0
#define ATTENTION_ROPE_BETA_FAST 32.0
#define ATTENTION_ROPE_BETA_SLOW 1.0
#define ATTENTION_ROPE_DIM 64.0
#define ATTENTION_ROPE_RAMP_LEAST 1.0e-3
#define ATTENTION_TWO_PI 6.283185307179586
#define ATTENTION_BOUND_NONE
#define ATTENTION_BOUND_GATHER_SHARED __launch_bounds__(ATTENTION_GATHER_SHARED_THREADS, 1)

/* break:group_by_rows: the contract's step at one row, 0 in a launch of more (a row then differs alone). */
template <>
__device__ __forceinline__ float attention_rescale_step<ATTENTION_STEP_BY_ROWS>(void)
{
    return gridDim.x > 1u ? 0.0f : ATTENTION_CONTRACT_STEP;
}

/* id, launch bound, kind, head tiles, ring stages, step rule. A kernel is attention_core_<id>; the first is the
 * baseline's (16 heads a block: two tiles of 8; two stages). group_by_rows exists for --break exactness alone. */
#define ATTENTION_CORE_KERNELS(STAMP)                                                                                \
    STAMP(baseline, ATTENTION_BOUND_NONE, ATTENTION_FP16, ATTENTION_WARP_TILES, ATTENTION_BASELINE_STAGES,           \
          ATTENTION_STEP_CONTRACT)                                                                                   \
    STAMP(heads_8, ATTENTION_BOUND_NONE, ATTENTION_FP16, 1, ATTENTION_BASELINE_STAGES, ATTENTION_STEP_CONTRACT)      \
    STAMP(stages_1, ATTENTION_BOUND_NONE, ATTENTION_FP16, ATTENTION_WARP_TILES, 1, ATTENTION_STEP_CONTRACT)          \
    STAMP(stages_3, ATTENTION_BOUND_NONE, ATTENTION_FP16, ATTENTION_WARP_TILES, ATTENTION_DEEP_STAGES,               \
          ATTENTION_STEP_CONTRACT)                                                                                   \
    STAMP(gather_shared, ATTENTION_BOUND_GATHER_SHARED, ATTENTION_FP16, ATTENTION_ALL_TILES, 1,                      \
          ATTENTION_STEP_CONTRACT)                                                                                   \
    STAMP(rescale_always, ATTENTION_BOUND_NONE, ATTENTION_FP16, ATTENTION_WARP_TILES, ATTENTION_BASELINE_STAGES,     \
          ATTENTION_STEP_ALWAYS)                                                                                     \
    STAMP(trained, ATTENTION_BOUND_NONE, ATTENTION_TRAINED, ATTENTION_WARP_TILES, ATTENTION_BASELINE_STAGES,         \
          ATTENTION_STEP_CONTRACT)                                                                                   \
    STAMP(fresh, ATTENTION_BOUND_NONE, ATTENTION_FRESH, ATTENTION_WARP_TILES, ATTENTION_BASELINE_STAGES,             \
          ATTENTION_STEP_CONTRACT)                                                                                   \
    STAMP(group_by_rows, ATTENTION_BOUND_NONE, ATTENTION_FP16, ATTENTION_WARP_TILES, ATTENTION_BASELINE_STAGES,      \
          ATTENTION_STEP_BY_ROWS)
#define ATTENTION_STAMP_CORE(id, bound, kind, head_tiles, stages, step_rule) \
    ATTENTION_CORE_KERNEL(attention_core_##id, bound, kind, head_tiles, stages, step_rule)
ATTENTION_CORE_KERNELS(ATTENTION_STAMP_CORE)
#define ATTENTION_GATHER_HEAD_TILES ATTENTION_WARP_TILES       /* gather_only: the baseline's grid, block and ring */
#define ATTENTION_GATHER_STAGES ATTENTION_BASELINE_STAGES

__host__ __device__ inline uint32_t attention_mix_bits(uint32_t value)
{
    value ^= value >> ATTENTION_MIX_SHIFT_OUTER;
    value *= ATTENTION_MIX_FIRST;
    value ^= value >> ATTENTION_MIX_SHIFT_INNER;
    value *= ATTENTION_MIX_SECOND;
    value ^= value >> ATTENTION_MIX_SHIFT_OUTER;
    return value;
}

/* ---- the planes ---- */

/* The number at dim `dim` of plane row `row`. The blocks of 32 of a row take turns: plain values (k - 128) / 128;
 * one large beside magnitudes spread over 2^17 (subnormal E4M3 codes); zeros of both signs between plain values;
 * a block under act_quant's floor (its scale 2^-22, the least a window row has); plain again. Every one is an fp16. */
__device__ inline float attention_plane_value(uint32_t row, uint32_t dim)
{
    const uint32_t bits = attention_mix_bits(row * ATTENTION_DIMS + dim + 1u);
    const float plain = ldexpf((float)(bits % ATTENTION_VALUE_STEPS) - (float)ATTENTION_VALUE_CENTRE,
                               -ATTENTION_VALUE_UNIT_SHIFT);
    const float sign = ((bits >> ATTENTION_HALF_BITS) & 1u) != 0 ? -1.0f : 1.0f;
    const float eighths = (float)((bits >> ATTENTION_BYTE_BITS) & ATTENTION_EIGHTHS_MASK) / (float)ATTENTION_EIGHTHS;
    switch ((row + dim / ATTENTION_POW2_BLOCK_DIMS) % ATTENTION_VALUE_KINDS) {
    case ATTENTION_VALUE_SPREAD:
        return sign * ldexpf(1.0f + eighths, (int)(dim % ATTENTION_SPREAD_PLACES) - (ATTENTION_SPREAD_PLACES - 1));
    case ATTENTION_VALUE_ZEROS: return (dim & 1u) != 0 ? plain : sign * 0.0f;
    case ATTENTION_VALUE_TINY: return ldexpf(plain, -ATTENTION_TINY_SHIFT);
    default: return plain;
    }
}

/* The nearest E4M3 code of a value (a tie to the even code; magnitude limited to 448; the sign kept): at a float
 * exponent field f the E4M3 numbers are 2^(f - 130) apart, 2^-9 under 2^-6, so the magnitude in those units,
 * rounded, added to the code of the stretch's first number is the code. */
__device__ inline uint32_t attention_e4m3_code(float value)
{
    const float magnitude = fminf(fabsf(value), ATTENTION_E4M3_LARGEST);
    const uint32_t field =
        __float_as_uint(fmaxf(magnitude, ATTENTION_E4M3_LEAST_NORMAL)) >> ATTENTION_FLOAT_EXPONENT_SHIFT;
    const uint32_t units = (uint32_t)__float2int_rn(
        magnitude * __uint_as_float((ATTENTION_E4M3_UNITS_EXPONENT + ATTENTION_FLOAT_BIAS - field)
                                    << ATTENTION_FLOAT_EXPONENT_SHIFT));
    const uint32_t sign = (__float_as_uint(value) >> ATTENTION_FLOAT_SIGN_TO_BYTE) & ATTENTION_E4M3_SIGN;
    return ((field << ATTENTION_E4M3_MANTISSA_BITS) + units - ATTENTION_E4M3_CODE_BASE) | sign;
}

__device__ inline float attention_e4m3_number(uint32_t code)       /* of a code without its sign */
{
    return __uint_as_float((code & ATTENTION_E4M3_MAGNITUDE_MASK) << ATTENTION_E4M3_FLOAT_SHIFT) *
           ATTENTION_E4M3_TO_FLOAT;
}

/* An E2M1 magnitude code's number in halves: m with exponent 0, else (2 + m) 2^(e - 1): 0, 1, 2, 3, 4, 6, 8, 12. */
__device__ inline uint32_t attention_e2m1_halves(uint32_t code)
{
    const uint32_t exponent = code >> 1, mantissa = code & 1u;
    return exponent == 0 ? mantissa : (ATTENTION_E2M1_IMPLICIT_HALVES + mantissa) << (exponent - 1u);
}

/* The nearest E2M1 code of a value already over its scale (magnitude limited to 6; a tie to the even code): a
 * midpoint between two numbers is their sum in halves, in quarters. */
__device__ inline uint32_t attention_e2m1_code(float value)
{
    const float quarters = fminf(fabsf(value), ATTENTION_E2M1_LARGEST) * ATTENTION_E2M1_QUARTERS;
    uint32_t code = 0;
    for (uint32_t upper = 1; upper < ATTENTION_E2M1_CODES; upper++) {
        const float midpoint = (float)(attention_e2m1_halves(upper - 1u) + attention_e2m1_halves(upper));
        code = quarters > midpoint || (quarters == midpoint && (upper & 1u) == 0) ? upper : code;
    }
    return code | (signbit(value) ? (uint32_t)ATTENTION_E2M1_SIGN : 0u);
}

struct attention_fill_arguments {
    unsigned char *fp16_plane, *pow2_plane, *e2m1_plane;
    const uint32_t *rows;                        /* the rows to fill; NULL: rows first_row .. first_row + count - 1 */
    uint64_t first_row, count;
};

/* A thread a block of 32 dims of a row: its fp16 values, its pow2 codes and scale, its two E2M1 halves. */
__global__ void attention_fill_kernel(struct attention_fill_arguments arguments)
{
    const uint64_t item = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= arguments.count * ATTENTION_POW2_BLOCKS) {
        return;
    }
    const uint64_t place = item / ATTENTION_POW2_BLOCKS;
    const uint32_t row = arguments.rows != NULL ? arguments.rows[place] : (uint32_t)(arguments.first_row + place);
    const uint32_t block = (uint32_t)(item % ATTENTION_POW2_BLOCKS), first_dim = block * ATTENTION_POW2_BLOCK_DIMS;
    uint16_t *halves = (uint16_t *)(arguments.fp16_plane + (uint64_t)row * ATTENTION_FP16_ROW_BYTES) + first_dim;
    unsigned char *pow2 = arguments.pow2_plane + (uint64_t)row * ATTENTION_POW2_ROW_BYTES;
    unsigned char *e2m1 = arguments.e2m1_plane + (uint64_t)row * ATTENTION_E2M1_ROW_BYTES;
    float largest = 0.0f, half_largest[ATTENTION_E2M1_HALF_BLOCKS] = {0.0f, 0.0f};
    for (uint32_t dim = 0; dim < ATTENTION_POW2_BLOCK_DIMS; dim++) {
        const float value = attention_plane_value(row, first_dim + dim);         /* an fp16 number: exact */
        halves[dim] = __half_as_ushort(__float2half_rn(value));
        largest = fmaxf(largest, fabsf(value));
        float *half_block = &half_largest[dim / ATTENTION_E2M1_BLOCK_DIMS];
        *half_block = fmaxf(*half_block, fabsf(value));
    }
    const uint32_t ratio = __float_as_uint(fmaxf(largest, ATTENTION_POW2_FLOOR) * ATTENTION_E4M3_LARGEST_INVERSE);
    const uint32_t scale_code = ((ratio >> ATTENTION_FLOAT_EXPONENT_SHIFT) & ATTENTION_FLOAT_EXPONENT_FIELD) +
                                ((ratio & ATTENTION_FLOAT_MANTISSA_MASK) != 0 ? 1u : 0u);
    pow2[ATTENTION_CODE_BYTES + block] = (unsigned char)scale_code;
    uint32_t e2m1_scales[ATTENTION_E2M1_HALF_BLOCKS];
    for (uint32_t half = 0; half < ATTENTION_E2M1_HALF_BLOCKS; half++) {
        e2m1_scales[half] =
            attention_e4m3_code(fmaxf(half_largest[half], ATTENTION_E2M1_FLOOR) / ATTENTION_E2M1_LARGEST);
        e2m1[ATTENTION_CODE_BYTES + block * ATTENTION_E2M1_HALF_BLOCKS + half] = (unsigned char)e2m1_scales[half];
    }
    for (uint32_t dim = 0; dim < ATTENTION_POW2_BLOCK_DIMS; dim++) {
        const float value = __half2float(__ushort_as_half(halves[dim]));
        const float scale = __uint_as_float(scale_code << ATTENTION_FLOAT_EXPONENT_SHIFT);   /* 2^(code - 127) */
        pow2[first_dim + dim] = (unsigned char)attention_e4m3_code(value / scale);
        e2m1[first_dim + dim] = (unsigned char)attention_e2m1_code(
            value / attention_e4m3_number(e2m1_scales[dim / ATTENTION_E2M1_BLOCK_DIMS]));
    }
}

/* On the host: the check's hand-made 528-byte window row of the 8-bit planes. Its codes are all subnormal E4M3 and
 * its scale makes them as large as the plain values: were their fp16 placement (subnormal halves) flushed, every
 * score of a row that names it would move. The fills make no such row (a block's largest code is 224 or more). */
static inline void attention_subnormal_record(unsigned char *record)
{
    for (uint32_t byte = 0; byte < ATTENTION_POW2_ROW_BYTES; byte++) {
        const uint32_t sign = ((byte / ATTENTION_SUBNORMAL_CODES) & 1u) != 0 ? (uint32_t)ATTENTION_E4M3_SIGN : 0u;
        record[byte] = (unsigned char)(byte < ATTENTION_CODE_BYTES ? (byte % ATTENTION_SUBNORMAL_CODES + 1u) | sign
                                                                   : (uint32_t)ATTENTION_SUBNORMAL_SCALE);
    }
}

/* On the host: the angles of a row's position as a rate-2 layer of the reference forms them (precompute_freqs_cis:
 * YaRN between beta_fast and beta_slow, the frequencies and the angle in fp32), its cos and sin to the nearest fp32. */
static inline void attention_rotation_row(double position, float *rotation)
{
    const double low_dim = ATTENTION_ROPE_DIM * log(ATTENTION_ROPE_ORIGINAL /
                                                    (ATTENTION_ROPE_BETA_FAST * ATTENTION_TWO_PI)) /
                           (ATTENTION_DIMS_PER_FREQUENCY * log(ATTENTION_ROPE_BASE));
    const double high_dim = ATTENTION_ROPE_DIM * log(ATTENTION_ROPE_ORIGINAL /
                                                     (ATTENTION_ROPE_BETA_SLOW * ATTENTION_TWO_PI)) /
                            (ATTENTION_DIMS_PER_FREQUENCY * log(ATTENTION_ROPE_BASE));
    const double low = fmax(floor(low_dim), 0.0), high = fmin(ceil(high_dim), ATTENTION_ROPE_DIM - 1.0);
    for (uint32_t pair = 0; pair < ATTENTION_ROPE_PAIRS; pair++) {
        const double place = (double)(pair * ATTENTION_DIMS_PER_FREQUENCY) / ATTENTION_ROPE_DIM;
        const float frequency = (float)(1.0 / pow(ATTENTION_ROPE_BASE, place));
        const float ramp =
            (float)fmin(fmax(((double)pair - low) / fmax(high - low, ATTENTION_ROPE_RAMP_LEAST), 0.0), 1.0);
        const float smooth = 1.0f - ramp;
        const float corrected = frequency / (float)ATTENTION_ROPE_FACTOR * (1.0f - smooth) + frequency * smooth;
        const float angle = (float)position * corrected;
        rotation[pair * ATTENTION_ROPE_VALUES] = (float)cos((double)angle);
        rotation[pair * ATTENTION_ROPE_VALUES + 1u] = (float)sin((double)angle);
    }
}

/* Poisoned rows for the check, where no list of it reaches, in the three planes (ATTENTION_POISON_*): a key read from
 * one turns a row's output to NaN or far from the reference in every kind. */
__global__ void attention_poison_kernel(struct attention_fill_arguments arguments)
{
    const uint64_t item = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= arguments.count * ATTENTION_FP16_ROW_BYTES) {
        return;
    }
    const uint64_t row = arguments.rows[item / ATTENTION_FP16_ROW_BYTES], byte = item % ATTENTION_FP16_ROW_BYTES;
    const int code = byte < ATTENTION_CODE_BYTES;
    arguments.fp16_plane[row * ATTENTION_FP16_ROW_BYTES + byte] = ATTENTION_POISON_FP16;
    if (byte < ATTENTION_POW2_ROW_BYTES) {
        arguments.pow2_plane[row * ATTENTION_POW2_ROW_BYTES + byte] =
            code ? ATTENTION_POISON_E4M3 : ATTENTION_POISON_E8M0;
    }
    if (byte < ATTENTION_E2M1_ROW_BYTES) {
        arguments.e2m1_plane[row * ATTENTION_E2M1_ROW_BYTES + byte] =
            code ? ATTENTION_POISON_E2M1 : ATTENTION_POISON_E4M3;
    }
}

/* CONVENTIONS 3.5 for pool:compact: after its copy (not timed), every 16 bytes of a scratch of four L2 sizes read
 * and written changed, so that no line of the region it just wrote is left in the L2 for the timed launch. */
__global__ void attention_evict_kernel(unsigned char *scratch, uint64_t chunks, uint32_t mark)
{
    const uint64_t item = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (item < chunks) {
        uint4 chunk = ((const uint4 *)scratch)[item];
        chunk.x ^= mark;
        ((uint4 *)scratch)[item] = chunk;
    }
}

/* ---- lists and compact copies ---- */

struct attention_list_arguments { int32_t *indices; uint32_t rows, band, seed, reserved_zero; };

__global__ void attention_list_kernel(struct attention_list_arguments arguments)
{
    const uint64_t item = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= (uint64_t)arguments.rows * ATTENTION_KEYS) {
        return;
    }
    const uint32_t row = (uint32_t)(item / ATTENTION_KEYS), key = (uint32_t)(item % ATTENTION_KEYS);
    const uint32_t entry = key - ATTENTION_WINDOW_KEYS;
    arguments.indices[item] = key < ATTENTION_WINDOW_KEYS
        ? (int32_t)(row + key)
        : (int32_t)(entry * arguments.band +
                    attention_mix_bits(arguments.seed + attention_mix_bits((uint32_t)item)) % arguments.band);
}

/* On the host: the distinct pool rows the lists [rows][640] name, through `seen` (a byte a plane row, cleared here);
 * UINT64_MAX where an index lies outside the plane. */
static inline uint64_t attention_distinct_pool_rows(const int32_t *lists, uint32_t rows, uint8_t *seen,
                                                    uint64_t plane_rows)
{
    uint64_t distinct = 0;
    memset(seen, 0, plane_rows);
    for (size_t place = 0; place < (size_t)rows * ATTENTION_KEYS; place++) {
        const int32_t index = lists[place];
        if (place % ATTENTION_KEYS < ATTENTION_WINDOW_KEYS || index == -1) {
            continue;
        }
        if (index < 0 || (uint64_t)index >= plane_rows) {
            return UINT64_MAX;
        }
        distinct += seen[index] != 0 ? 0u : 1u;
        seen[index] = 1;
    }
    return distinct;
}

struct attention_compact_arguments {
    const int32_t *indices;                      /* the step's lists */
    const unsigned char *pool_rows;
    unsigned char *compact;                      /* [row][512] rows of row_bytes */
    uint32_t rows, row_bytes;
};

/* Row r's pool entry e is copied to compact row 512 r + e, 16 bytes a thread; an entry of -1 leaves zeros. */
__global__ void attention_compact_kernel(struct attention_compact_arguments arguments)
{
    const uint64_t chunks = arguments.row_bytes / ATTENTION_CHUNK_BYTES;
    const uint64_t item = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= (uint64_t)arguments.rows * ATTENTION_POOL_KEYS * chunks) {
        return;
    }
    const uint64_t entry = item / chunks, chunk = item % chunks;
    const int32_t index = arguments.indices[entry / ATTENTION_POOL_KEYS * ATTENTION_KEYS + ATTENTION_WINDOW_KEYS +
                                            entry % ATTENTION_POOL_KEYS];
    ((uint4 *)(arguments.compact + entry * arguments.row_bytes))[chunk] =
        index < 0 ? make_uint4(0u, 0u, 0u, 0u)
                  : ((const uint4 *)(arguments.pool_rows + (uint64_t)index * arguments.row_bytes))[chunk];
}

/* ---- the bounds a kernel is held against ---- */

/* gather_only: the baseline's index reads and copies through the same ring, each thread folding the words it
 * copied (a key of -1: zeros) once its stage is there; the block's sum of every 32-bit word of its row's 640 stored
 * rows, whatever the order, to checksums[row][head group]. No arithmetic on them. */
template <int kind>
__device__ __forceinline__ void attention_gather_block(const struct attention_core_arguments &arguments,
                                                       uint32_t *checksums,
                                                       unsigned char *shared ATTENTION_CHECKS_PARAMETER)
{
    constexpr uint32_t threads = attention_threads(ATTENTION_GATHER_HEAD_TILES);
    constexpr uint32_t stage_bytes = attention_stage_bytes(kind);
    constexpr struct attention_layout layout =
        attention_layout_of(kind, ATTENTION_GATHER_HEAD_TILES, ATTENTION_GATHER_STAGES);
    constexpr uint32_t threads_per_key = threads / ATTENTION_GROUP_KEYS;
    const uint32_t copy_key = threadIdx.x / threads_per_key, lane = threadIdx.x % ATTENTION_LANES;
    const int32_t *row_indices = arguments.indices + (uint64_t)blockIdx.x * ATTENTION_KEYS;
    int32_t *indices = (int32_t *)(shared + layout.indices);
    uint32_t *warp_sums = (uint32_t *)(shared + layout.factors);
    uint32_t fold = 0;
    attention_issue<kind, threads>(&arguments, shared, indices, 0, row_indices[copy_key] ATTENTION_CHECKS_ADDRESS);
    attention_commit_copies();
    int32_t index_ahead = row_indices[ATTENTION_GROUP_KEYS + copy_key];
    for (uint32_t group = 0; group < ATTENTION_GROUPS; group++) {
        const uint32_t next = group + 1u, slot = group % ATTENTION_GATHER_STAGES;
        attention_wait_copies<0>();
        __syncthreads();
        if (next < ATTENTION_GROUPS) {
            attention_issue<kind, threads>(&arguments, shared + next % ATTENTION_GATHER_STAGES * stage_bytes,
                                           indices + next % ATTENTION_GATHER_STAGES * ATTENTION_GROUP_KEYS, next,
                                           index_ahead ATTENTION_CHECKS_ADDRESS);
        }
        attention_commit_copies();
        if (next + 1u < ATTENTION_GROUPS) {
            index_ahead = row_indices[(next + 1u) * ATTENTION_GROUP_KEYS + copy_key];
        }
        const uint32_t chunks = attention_row_bytes(attention_format_of(kind, group)) / ATTENTION_CHUNK_BYTES;
        for (uint32_t chunk = threadIdx.x % threads_per_key; chunk < chunks; chunk += threads_per_key) {
            const uint4 words =
                *(const uint4 *)attention_stage_place<kind>(shared + slot * stage_bytes, copy_key, chunk);
            fold += words.x + words.y + words.z + words.w;
        }
    }
    for (uint32_t distance = ATTENTION_LANES >> 1; distance != 0; distance >>= 1) {
        fold += __shfl_xor_sync(ATTENTION_EVERY_LANE, fold, (int)distance);
    }
    __syncthreads();                             /* the factors' place is free: nothing else of it is used */
    if (lane == 0) {
        warp_sums[threadIdx.x / ATTENTION_LANES] = fold;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        uint32_t block_sum = 0;
        for (uint32_t warp = 0; warp < threads / ATTENTION_LANES; warp++) {
            block_sum += warp_sums[warp];
        }
        checksums[(uint64_t)blockIdx.x * gridDim.y + blockIdx.y] = block_sum;
    }
}

#define ATTENTION_GATHER_KERNEL(kernel_name, kind)                                                                \
    __global__ void kernel_name(struct attention_core_arguments arguments,                                        \
                                uint32_t *checksums ATTENTION_CHECKS_PARAMETER)                                   \
    {                                                                                                             \
        extern __shared__ __align__(ATTENTION_CHUNK_BYTES) unsigned char attention_shared[];                     \
        attention_gather_block<kind>(arguments, checksums, attention_shared ATTENTION_CHECKS_ARGUMENT);          \
    }
ATTENTION_GATHER_KERNEL(attention_gather_only_fp16, ATTENTION_FP16)
ATTENTION_GATHER_KERNEL(attention_gather_only_trained, ATTENTION_TRAINED)
ATTENTION_GATHER_KERNEL(attention_gather_only_fresh, ATTENTION_FRESH)

/* tensor_only: the baseline's grid and its tensor instructions a group (16 for the scores, 16 for P x V a warp) on
 * exact small fp16 made in registers, no load. No two instructions have the same operands and accumulator, every
 * group's operands are its own, and every operand adds `unknown` (0 from the host), so the assembler can neither
 * merge, hoist nor fold one: group g's v = g + 1, a tile's operand B (tile + 1) v, P x V's operand A of step s the
 * number s + 2. The scores of a group from +0 go to a total. Each thread folds its 72 sums (whole numbers under 2^24:
 * exact) to sums[block][thread]: attention_tensor_fold. */
__global__ void attention_tensor_only_kernel(float *sums, uint32_t unknown)
{
    const uint32_t one = ATTENTION_ONE_PAIR + unknown;
    const uint32_t ones[ATTENTION_REGISTERS_A] = {one, one, one, one};
    float totals[ATTENTION_WARP_TILES][ATTENTION_ELEMENTS] = {};
    float values[ATTENTION_STEPS][ATTENTION_WARP_TILES][ATTENTION_ELEMENTS] = {};
    for (uint32_t group = 0; group < ATTENTION_TENSOR_GROUPS; group++) {
        const uint32_t value_bits = __half_as_ushort(__float2half_rn((float)(group + 1u))) + unknown;
        float scores[ATTENTION_WARP_TILES][ATTENTION_ELEMENTS] = {};
        uint32_t operand_b[ATTENTION_WARP_TILES];
#pragma unroll
        for (uint32_t tile = 0; tile < ATTENTION_WARP_TILES; tile++) {     /* (tile + 1) v: one more exponent */
            const uint32_t scaled = value_bits + tile * ATTENTION_FP16_EXPONENT_ONE;
            operand_b[tile] = scaled | (scaled << ATTENTION_HALF_BITS);
        }
#pragma unroll
        for (uint32_t step = 0; step < ATTENTION_STEPS; step++) {
#pragma unroll
            for (uint32_t tile = 0; tile < ATTENTION_WARP_TILES; tile++) {
                attention_core_f16(scores[tile], ones, operand_b[tile], operand_b[tile]);
            }
        }
#pragma unroll
        for (uint32_t tile = 0; tile < ATTENTION_WARP_TILES; tile++) {
#pragma unroll
            for (uint32_t element = 0; element < ATTENTION_ELEMENTS; element++) {
                totals[tile][element] += scores[tile][element];
            }
        }
#pragma unroll
        for (uint32_t step = 0; step < ATTENTION_STEPS; step++) {
            const uint32_t number = __half_as_ushort(__float2half_rn((float)(step + ATTENTION_TENSOR_FIRST_NUMBER))) +
                                    unknown;
            const uint32_t pair = number | (number << ATTENTION_HALF_BITS);
            const uint32_t operand_a[ATTENTION_REGISTERS_A] = {pair, pair, pair, pair};
#pragma unroll
            for (uint32_t tile = 0; tile < ATTENTION_WARP_TILES; tile++) {
                attention_core_f16(values[step][tile], operand_a, operand_b[tile], operand_b[tile]);
            }
        }
    }
    float fold = 0.0f;
#pragma unroll
    for (uint32_t tile = 0; tile < ATTENTION_WARP_TILES; tile++) {
#pragma unroll
        for (uint32_t element = 0; element < ATTENTION_ELEMENTS; element++) {
            fold += totals[tile][element];
#pragma unroll
            for (uint32_t step = 0; step < ATTENTION_STEPS; step++) {
                fold += values[step][tile][element];
            }
        }
    }
    sums[((uint64_t)blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x + threadIdx.x] = fold;
}

/* What every fold of tensor_only is: 16 products an instruction, v summed over the groups (1 + .. + 40 = 820), the
 * tiles' 1 + 2; the scores 8 steps of 1, the values steps of 2 + .. + 9 = 44; 4 elements each: 8,186,880. */
__host__ __device__ constexpr float attention_tensor_fold(void)
{
    return (float)(ATTENTION_ELEMENTS * ATTENTION_GROUP_KEYS * ATTENTION_TENSOR_V_SUM * ATTENTION_TENSOR_TILE_SUM *
                   (ATTENTION_STEPS + ATTENTION_TENSOR_STEP_SUM));
}

#endif /* ATTENTION_CORE_PROBE_KERNELS_CUH */
