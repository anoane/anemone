/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * index_scan_kernels.cuh: the index scan of the attention's eight selecting layers and the exact top 512 after it, 1
 * to 256 rows over up to 1,048,576 entries, FP16 keys or the owner's two 8-bit kinds (research/speed/
 * ROUND_1_attention.md 11a; DECISIONS_2026-10-12.md section 1). Meant to move into engine/source as it stands once
 * measured; what exists for the measurement alone is in index_scan_probe_kernels.cuh. Not part of the engine yet.
 *
 * THE SCORE of (row, entry): the sum over 32 heads of weight[row][head] x relu(query[row][head] . key[entry]), 128
 * dims a head, rounded once to fp16; an entry at or beyond extent[row] scores -inf (0xFC00).
 *   - A head's dot. FP16 keys (form HALF): per tile of 16 heads one m16n8k16 chain from +0 over the eight steps of 16
 *     dims in ascending order. 8-bit keys: per block of 32 dims a sum from +0 (one m16n8k32, or two m16n8k16 in
 *     PLACED), then chain = fma(sum, 2^(the block's scale codes), chain), the blocks in ascending order from +0.
 *   - Then per key in fp32: a lane's heads g, g + 8, 16 + g, 24 + g as (r w + r w) + (r w + r w), r the ReLU (+0 for a
 *     dot at or under 0) and one multiply by the head's weight; the eight lane groups by the butterfly xor 4, 8, 16,
 *     which is ((v0 + v1) + (v2 + v3)) + ((v4 + v5) + (v6 + v7)) on every lane. Nothing of it depends on the rows of
 *     a launch, the group, the warps of a block, the segment or the grid's order: class R.
 * THE SELECTION of a row: its top min(512, extent) by (order key descending, position ascending), the order key of an
 *   fp16 score its bits with the sign flipped, or all of them flipped where the sign is set (unsigned order is score
 *   order, -0 below +0); stored as positions ascending, then -1 to 512. Histograms join the lanes of a warp that hit
 *   one bin with the match instruction; the lowest of them adds their count to the warp's own bins (no atomics).
 *
 * RECORDS (a tile is 8 keys; lane 4 g + t loads key g and holds the results of keys 2 t and 2 t + 1)
 *   FP16 keys: 2,048 B a tile, key g at 256 g, its run t (64 B at 64 t) the halves of dims 16 j + 2 t + {0, 1, 8, 9}
 *     for j = 0 .. 7: registers 0 and 1 of operand B of step j; one 16-byte load holds two steps.
 *   8-bit keys: 1,056 B a tile: key g's 128 code bytes at 128 g, its run t (32 B at 32 t) a word per step j, byte i of
 *     it dim 16 j + 2 t + {0, 8, 1, 9}[i]; at 1,024 the eight keys' scale words, byte b of key g's the code c of its
 *     block b (dims 32 b .. 32 b + 31), a scale of 2^(c - 127). TRAINED (the reference's rounding of index keys and
 *     queries, model.py:546, 552): a code is a signed byte, twice an E2M1 value (-12 .. 12; its -0 is +0 here).
 *     FRESH: an E4M3 byte.
 *   Queries, placed by their producer. FP16 (forms HALF, PLACED): 8,192 B a row, [head tile h][step j][lane][register
 *     r]: the halves of head 16 h + g + 8 (r % 2) at dims 16 j + 2 t + 8 (r / 2) + {0, 1}. Codes (WHOLE, E2M1, E4M3):
 *     4,224 B a row, [h][block b][lane][register r], byte i: head 16 h + g + 8 (r % 2), dim 32 b + 16 (r / 2) + 2 t +
 *     {0, 8, 1, 9}[i], the keys' order (an instruction of bytes meets its 32 products in an order of its own, the same
 *     on both sides: measurements/i03a_mma_unit), then the scale codes, [head][block].
 *   weights fp32 [row][32]; extents [row]; scores fp16 [row][capacity]; a candidate list [row][2,048] ascending.
 * FORMS: HALF (FP16 keys and queries); WHOLE (TRAINED: m16n8k32 s8, exact whole sums, both scales outside); E2M1
 *   (TRAINED, sm_120a, behind INDEX_E2M1_SCALED: the block-scaled E2M1 instruction on a zero sum, both scales inside,
 *   the codes turned to E2M1 in registers: WHOLE's bits where the instruction sums exactly); E4M3 (FRESH, sm_120a:
 *   m16n8k32 e4m3, the queries rounded to E4M3 per 32 by their producer: no form of the card takes fp16 by 8 bits);
 *   PLACED (FRESH: the codes placed as fp16, value times 2^-8, FP16 queries, two m16n8k16 a block).
 * A kernel checks nothing and uses no atomic operation: the host holds every count, range and value a launch reads,
 * writes or loops by (check 10 of research/speed/CONVENTIONS.md).
 */
#ifndef INDEX_SCAN_KERNELS_CUH
#define INDEX_SCAN_KERNELS_CUH
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>
#if !defined(INDEX_E2M1_SCALED)
#define INDEX_E2M1_SCALED 0                      /* 1: the batch compiled the block-scaled E2M1 form alone first */
#endif

enum index_form { INDEX_FORM_HALF = 0, INDEX_FORM_WHOLE, INDEX_FORM_E2M1, INDEX_FORM_E4M3, INDEX_FORM_PLACED,
                  INDEX_FORMS };
enum index_source { INDEX_SOURCE_ROW = 0, INDEX_SOURCE_SPAN, INDEX_SOURCE_CANDIDATES };
enum {
    INDEX_HEADS = 32, INDEX_DIMS = 128, INDEX_KEYS_PER_TILE = 8, INDEX_LANES = 32, INDEX_LANE_PLACES = 4,
    INDEX_HEAD_TILES = 2, INDEX_HEADS_PER_TILE = 16, INDEX_HALF_TILE = 8, INDEX_STEPS = 8, INDEX_DIMS_PER_STEP = 16,
    INDEX_BLOCKS = 4, INDEX_DIMS_PER_BLOCK = 32, INDEX_STEPS_PER_LOAD = 2, INDEX_COLUMNS_PER_LANE = 2,
    INDEX_MOST_GROUP_ROWS = 8, INDEX_LOAD_BYTES = 16, INDEX_WORD_BYTES = 4, INDEX_BYTE_BITS = 8, INDEX_HALF_BITS = 16,
    INDEX_HALF_KEY_BYTES = 256, INDEX_HALF_RUN_BYTES = 64, INDEX_HALF_TILE_BYTES = 2048, INDEX_HALF_RUN_LOADS = 4,
    INDEX_CODE_KEY_BYTES = 128, INDEX_CODE_RUN_BYTES = 32, INDEX_CODE_RUN_LOADS = 2, INDEX_CODE_SCALES_AT = 1024,
    INDEX_CODE_TILE_BYTES = 1056, INDEX_HALF_QUERY_BYTES = 8192, INDEX_QUERY_SCALES_AT = 4096,
    INDEX_CODE_QUERY_BYTES = 4224, INDEX_CANDIDATE_BLOCKS = 2048, INDEX_CANDIDATE_ENTRIES = 16384,
    INDEX_SELECTED = 512, INDEX_SPAN_ENTRIES = 32768, INDEX_MOST_SPANS = 32, INDEX_SELECT_THREADS = 1024,
    INDEX_SELECT_WARPS = 32, INDEX_BINS = 256, INDEX_BINS_PER_LANE = 8, INDEX_KEYS_PER_LOAD = 8,
    INDEX_PAIRS_PER_LOAD = 4, INDEX_CUT_FIELDS = 2, INDEX_DIGIT_MASK = 0xFF, INDEX_NEGATIVE_INFINITY = 0xFC00,
    INDEX_FLOAT32_EXPONENT_SHIFT = 23, INDEX_FLOAT32_EXPONENT_INFINITE = 255, INDEX_HALF_SIGN_SHIFT = 15,
    INDEX_E8M0_BIAS = 127,                       /* a scale code c is 2^(c - 127) */
    /* the float32 exponent field of a block's power of two is its scale codes' sum less this: WHOLE (two codes of
     * twice the value: a quarter), E2M1 and E4M3 (two codes), PLACED adds 8 to the key's code (placed: 2^-8) */
    INDEX_WHOLE_FIELD_BIAS = 129, INDEX_E4M3_FIELD_BIAS = 127, INDEX_PLACED_FIELD_BIAS = -8,
    /* twice an E2M1 value is its code up to 4, then 6, 8, 12 are 5, 6, 7: 4 more than a quarter; in a byte of kind
     * f8f6f4 a code lies at bits 2 .. 5 */
    INDEX_E2M1_QUARTER_SHIFT = 2, INDEX_E2M1_SIGN_SHIFT = 4, INDEX_E2M1_BYTE_SHIFT = 2,
    INDEX_FP16_MAGNITUDE_SHIFT = 7               /* an E4M3 byte's magnitude at bits 7 .. 13 of an fp16: value x 2^-8 */
};
#define INDEX_EVERY_LANE 0xFFFFFFFFu
#define INDEX_LOW_BITS_OF_HALVES 0x00010001u
#define INDEX_ORDER_FLIP_NEGATIVE 0x7FFFu        /* with the sign set every bit flips, else the sign alone */
#define INDEX_ORDER_SIGNS 0x80008000u
#define INDEX_EVERY_BYTE_FOUR 0x04040404u
#define INDEX_EVERY_BYTE_SIX_BITS 0x3F3F3F3Fu
#define INDEX_EVERY_BYTE_BIT_THREE 0x08080808u
#define INDEX_EVEN_BYTE_SIGNS 0x00800080u        /* bytes 0 and 2 of a word: the halves of one register */
#define INDEX_EVEN_BYTE_MAGNITUDES 0x007F007Fu
#define INDEX_ODD_BYTE_MAGNITUDES 0x7F007F00u
#define INDEX_HALF_SIGNS 0x80008000u

struct index_scan_arguments {                    /* grid: (groups, segments), or (segments, rows) with rows slow */
    const unsigned char *keys;                   /* the plane: tile records from entry 0 */
    const unsigned char *queries;                /* placed, [row][index_query_bytes(form)] */
    const float *weights;                        /* [row][32] */
    const uint32_t *extents;                     /* [row]: the entries the row sees */
    const uint32_t *listed;                      /* [row][2,048]: candidate blocks ascending; NULL unless listed */
    uint16_t *scores;                            /* fp16 [row][capacity] */
    uint32_t row_count, capacity, segment_tiles, reserved_zero;
};
struct index_select_arguments {                  /* one_block, merge: grid (rows); spans: (spans, rows); 32 warps */
    const uint16_t *scores;                      /* fp16 [row][capacity] */
    const uint32_t *extents;
    int32_t *selected;                           /* [row][512]: positions ascending, then -1 */
    uint16_t *span_keys;                         /* [row][spans][512]: order keys of a span's top 512 */
    uint32_t *span_positions;                    /* [row][spans][512] */
    uint32_t *span_counts;                       /* [row][spans] */
    uint32_t capacity, spans;                    /* spans: of a row, each of 32,768 entries */
};
struct index_place {                             /* what a block of a scan-shaped grid works on */
    uint32_t first_row, rows, first_tile, end_tile;
};

__host__ __device__ inline uint32_t index_query_bytes(int form)
{
    return form == INDEX_FORM_HALF || form == INDEX_FORM_PLACED ? INDEX_HALF_QUERY_BYTES : INDEX_CODE_QUERY_BYTES;
}
__host__ __device__ inline uint32_t index_tile_bytes(int form)
{
    return form == INDEX_FORM_HALF ? INDEX_HALF_TILE_BYTES : INDEX_CODE_TILE_BYTES;
}
__host__ __device__ inline uint32_t index_tiles_of(uint32_t entries)       /* whole tiles that hold `entries` */
{
    return entries / INDEX_KEYS_PER_TILE + (entries % INDEX_KEYS_PER_TILE != 0 ? 1u : 0u);
}
__host__ __device__ inline uint32_t index_listed_tiles(uint32_t extent)    /* a row's candidate blocks */
{
    return index_tiles_of(extent) < INDEX_CANDIDATE_BLOCKS ? index_tiles_of(extent) : (uint32_t)INDEX_CANDIDATE_BLOCKS;
}

/* sums += A B for a lane's part: A 16 heads by 16 or 32 dims, B one key; sums x, y, z, w are (head g, key 2 t),
 * (g, 2 t + 1), (g + 8, 2 t), (g + 8, 2 t + 1). */
#define INDEX_MMA_TEXT " {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}"
#define INDEX_MMA_SUMS(written, read, sums, a, b_first, b_second)                                                    \
    written((sums)->x), written((sums)->y), written((sums)->z), written((sums)->w)                                   \
        : "r"((a).x), "r"((a).y), "r"((a).z), "r"((a).w), "r"(b_first), "r"(b_second), read((sums)->x),             \
          read((sums)->y), read((sums)->z), read((sums)->w)
#define INDEX_MMA(function_name, sum_type, written, read, instruction)                                              \
    __device__ __forceinline__ void function_name(sum_type *sums, uint4 a, uint32_t b_first, uint32_t b_second)      \
    {                                                                                                                \
        asm volatile(instruction INDEX_MMA_TEXT ";\n" : INDEX_MMA_SUMS(written, read, sums, a, b_first, b_second));   \
    }
INDEX_MMA(index_mma_half, float4, "=f", "f", "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32")
INDEX_MMA(index_mma_whole, int4, "=r", "r", "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32")
#if defined(__CUDA_ARCH_FEAT_SM120_ALL)
INDEX_MMA(index_mma_e4m3, float4, "=f", "f", "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32")
#else       /* a build without the form stamps none of its kernels; a call that reaches one anyway traps */
__device__ __forceinline__ void index_mma_e4m3(float4 *, uint4, uint32_t, uint32_t) { __trap(); }
#endif
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) && INDEX_E2M1_SCALED
/* Both scales inside: byte 0 of lane 4 g's and of lane 4 g + 1's scale word for heads g and g + 8, byte 0 of lane
 * 4 g's for key g (selectors {0, 0}: measurements/i03a_mma_unit, as block_scaled_kernels.cuh reads them). */
__device__ __forceinline__ void index_mma_e2m1(float4 *sums, uint4 a, uint32_t b_first, uint32_t b_second,
                                               uint32_t a_scales, uint32_t b_scales)
{
    asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e2m1.e2m1.f32.ue8m0"
                 INDEX_MMA_TEXT ", %14, {0, 0}, %15, {0, 0};\n"
                 : INDEX_MMA_SUMS("=f", "f", sums, a, b_first, b_second), "r"(a_scales), "r"(b_scales));
}
#else
__device__ __forceinline__ void index_mma_e2m1(float4 *, uint4, uint32_t, uint32_t, uint32_t, uint32_t) { __trap(); }
#endif

/* 2^(field - 127) as a float32, by the definition: a subnormal for a field of -22 to 0, +0 under it, +infinity
 * from 255 (no fp16 input reaches either end: a nonzero block's scale code is at least about 100). */
__device__ __forceinline__ float index_power_of_two(int32_t field)
{
    const int32_t bounded = field > INDEX_FLOAT32_EXPONENT_INFINITE ? INDEX_FLOAT32_EXPONENT_INFINITE : field;
    return bounded > 0 ? __uint_as_float((uint32_t)bounded << INDEX_FLOAT32_EXPONENT_SHIFT)
           : bounded > -INDEX_FLOAT32_EXPONENT_SHIFT
               ? __uint_as_float(1u << (bounded + INDEX_FLOAT32_EXPONENT_SHIFT - 1))
               : 0.0f;
}
__device__ __forceinline__ uint32_t index_byte(uint32_t word, uint32_t byte)
{
    return (word >> (byte * INDEX_BYTE_BITS)) & INDEX_DIGIT_MASK;
}
__device__ __forceinline__ float index_relu(float dot) { return dot > 0.0f ? dot : 0.0f; }

/* Four TRAINED codes (twice an E2M1 value) as E2M1 bytes of kind f8f6f4: the code at bits 2 .. 5 of its byte. */
__device__ __forceinline__ uint32_t index_e2m1_word(uint32_t doubled)
{
    const uint32_t magnitude = __vabs4(doubled);
    const uint32_t large = __vcmpgtu4(magnitude, INDEX_EVERY_BYTE_FOUR);
    const uint32_t quarter = (magnitude >> INDEX_E2M1_QUARTER_SHIFT) & INDEX_EVERY_BYTE_SIX_BITS;
    const uint32_t code = (large & (quarter + INDEX_EVERY_BYTE_FOUR)) | (~large & magnitude);
    return (code | ((doubled >> INDEX_E2M1_SIGN_SHIFT) & INDEX_EVERY_BYTE_BIT_THREE)) << INDEX_E2M1_BYTE_SHIFT;
}
__device__ __forceinline__ uint4 index_e2m1_load(uint4 doubled)
{
    return make_uint4(index_e2m1_word(doubled.x), index_e2m1_word(doubled.y), index_e2m1_word(doubled.z),
                      index_e2m1_word(doubled.w));
}

/* Four E4M3 bytes as two registers of fp16 halves (each its value times 2^-8, subnormal codes included): bytes 0 and
 * 2 the first, 1 and 3 the second. Placement computes nothing. */
__device__ __forceinline__ void index_place_e4m3(uint32_t word, uint32_t *first, uint32_t *second)
{
    *first = ((word & INDEX_EVEN_BYTE_SIGNS) << INDEX_BYTE_BITS) |
             ((word & INDEX_EVEN_BYTE_MAGNITUDES) << INDEX_FP16_MAGNITUDE_SHIFT);
    *second = (word & INDEX_HALF_SIGNS) | ((word & INDEX_ODD_BYTE_MAGNITUDES) >> (INDEX_BYTE_BITS -
                                                                                 INDEX_FP16_MAGNITUDE_SHIFT));
}

/* The rows, and the tiles of its segment up to its rows' largest extent, of the block of a scan-shaped grid (rows on
 * the fast axis, or rows_fast 0: on the slow one); listed: one row, its list's positions instead of tiles. */
template <int group_rows, int rows_fast, int listed>
__device__ __forceinline__ struct index_place index_block_place(const struct index_scan_arguments *arguments,
                                                                uint32_t extents[group_rows])
{
    const uint32_t group = rows_fast ? blockIdx.x : blockIdx.y, segment = rows_fast ? blockIdx.y : blockIdx.x;
    struct index_place place;
    uint32_t largest = 0;
    place.first_row = group * group_rows;
    place.rows = arguments->row_count - place.first_row < group_rows ? arguments->row_count - place.first_row
                                                                     : (uint32_t)group_rows;
#pragma unroll
    for (uint32_t row = 0; row < group_rows; row++) {
        extents[row] = row < place.rows ? arguments->extents[place.first_row + row] : 0u;
        largest = extents[row] > largest ? extents[row] : largest;
    }
    const uint32_t tile_count = listed ? index_listed_tiles(largest) : index_tiles_of(largest);
    place.first_tile = segment * arguments->segment_tiles;
    place.end_tile = place.first_tile >= tile_count ? place.first_tile
                     : tile_count - place.first_tile < arguments->segment_tiles ? tile_count
                     : place.first_tile + arguments->segment_tiles;
    return place;
}

/* FP16 keys: the lane's four loads of its key, then per step the rows' and head tiles' instructions interleaved. */
template <int group_rows>
__device__ __forceinline__ void index_dots_half(const unsigned char *tile, const uint4 *queries, uint32_t rows,
                                                uint32_t lane, float4 dots[][INDEX_HEAD_TILES])
{
    const uint4 *run = (const uint4 *)(tile + lane / INDEX_LANE_PLACES * INDEX_HALF_KEY_BYTES +
                                       lane % INDEX_LANE_PLACES * INDEX_HALF_RUN_BYTES);
    uint4 loads[INDEX_HALF_RUN_LOADS];
#pragma unroll
    for (uint32_t load = 0; load < INDEX_HALF_RUN_LOADS; load++) {
        loads[load] = run[load];
    }
#pragma unroll
    for (uint32_t step = 0; step < INDEX_STEPS; step++) {
        const uint4 pair = loads[step / INDEX_STEPS_PER_LOAD];
        const uint32_t first = (step & 1u) != 0 ? pair.z : pair.x, second = (step & 1u) != 0 ? pair.w : pair.y;
#pragma unroll
        for (uint32_t row = 0; row < group_rows; row++) {
#pragma unroll
            for (uint32_t head_tile = 0; row < rows && head_tile < INDEX_HEAD_TILES; head_tile++) {
                const size_t at = ((row * INDEX_HEAD_TILES + head_tile) * INDEX_STEPS + step) * INDEX_LANES + lane;
                index_mma_half(&dots[row][head_tile], queries[at], first, second);
            }
        }
    }
}

/* 8-bit keys: per block the instruction from a zero sum, then the scales. The scale words of the keys of the lane's
 * two result columns, and for E2M1 of its own key; the queries' scale words, per head, behind their codes. */
template <int form, int group_rows>
__device__ __forceinline__ void index_dots_codes(const unsigned char *tile, const uint4 *queries, uint32_t rows,
                                                 uint32_t lane, float4 dots[][INDEX_HEAD_TILES])
{
    const uint32_t lane_group = lane / INDEX_LANE_PLACES, lane_place = lane % INDEX_LANE_PLACES;
    const uint4 *run = (const uint4 *)(tile + lane_group * INDEX_CODE_KEY_BYTES + lane_place * INDEX_CODE_RUN_BYTES);
    const uint32_t *key_scales = (const uint32_t *)(tile + INDEX_CODE_SCALES_AT);
    const uint4 loads[INDEX_CODE_RUN_LOADS] = {run[0], run[1]};
    const uint32_t even_scales = key_scales[lane_place * INDEX_COLUMNS_PER_LANE];
    const uint32_t odd_scales = key_scales[lane_place * INDEX_COLUMNS_PER_LANE + 1u];
    const uint32_t own_scales = form == INDEX_FORM_E2M1 ? key_scales[lane_group] : 0u;
    const uint32_t row_loads = (form == INDEX_FORM_PLACED ? INDEX_HALF_QUERY_BYTES : INDEX_CODE_QUERY_BYTES) /
                               INDEX_LOAD_BYTES;
#pragma unroll
    for (uint32_t block = 0; block < INDEX_BLOCKS; block++) {
        const uint4 pair = loads[block / INDEX_STEPS_PER_LOAD];
        uint32_t first = (block & 1u) != 0 ? pair.z : pair.x, second = (block & 1u) != 0 ? pair.w : pair.y;
        uint32_t placed[INDEX_LOAD_BYTES / INDEX_WORD_BYTES];
        const int32_t even_code = (int32_t)index_byte(even_scales, block);
        const int32_t odd_code = (int32_t)index_byte(odd_scales, block);
        if (form == INDEX_FORM_E2M1) {
            first = index_e2m1_word(first);
            second = index_e2m1_word(second);
        }
        index_place_e4m3(first, &placed[0], &placed[1]);
        index_place_e4m3(second, &placed[INDEX_COLUMNS_PER_LANE], &placed[INDEX_COLUMNS_PER_LANE + 1u]);
#pragma unroll
        for (uint32_t row = 0; row < group_rows; row++) {
#pragma unroll
            for (uint32_t head_tile = 0; row < rows && head_tile < INDEX_HEAD_TILES; head_tile++) {
                const uint4 *row_queries = queries + row * row_loads;
                const uint32_t *query_scales =
                    (const uint32_t *)(row_queries + INDEX_QUERY_SCALES_AT / INDEX_LOAD_BYTES);
                const uint32_t head = head_tile * INDEX_HEADS_PER_TILE + lane_group;
                const int32_t low_code = (int32_t)index_byte(query_scales[head], block);
                const int32_t high_code = (int32_t)index_byte(query_scales[head + INDEX_HALF_TILE], block);
                float4 *chain = &dots[row][head_tile], sums = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                int32_t bias = INDEX_E4M3_FIELD_BIAS;
                if (form == INDEX_FORM_PLACED) {
                    const uint4 *steps = row_queries + ((head_tile * INDEX_STEPS + block * INDEX_STEPS_PER_LOAD) *
                                                        INDEX_LANES + lane);
                    index_mma_half(&sums, steps[0], placed[0], placed[1]);
                    index_mma_half(&sums, steps[INDEX_LANES], placed[INDEX_COLUMNS_PER_LANE],
                                   placed[INDEX_COLUMNS_PER_LANE + 1u]);
                    chain->x = fmaf(sums.x, index_power_of_two(even_code - INDEX_PLACED_FIELD_BIAS), chain->x);
                    chain->y = fmaf(sums.y, index_power_of_two(odd_code - INDEX_PLACED_FIELD_BIAS), chain->y);
                    chain->z = fmaf(sums.z, index_power_of_two(even_code - INDEX_PLACED_FIELD_BIAS), chain->z);
                    chain->w = fmaf(sums.w, index_power_of_two(odd_code - INDEX_PLACED_FIELD_BIAS), chain->w);
                    continue;
                }
                const uint4 a = row_queries[(head_tile * INDEX_BLOCKS + block) * INDEX_LANES + lane];
                if (form == INDEX_FORM_E2M1) {           /* the lanes of a group's places 0 and 1: heads g, g + 8 */
                    const uint32_t a_scales = query_scales[head + (lane_place & 1u) * INDEX_HALF_TILE];
                    index_mma_e2m1(&sums, a, first, second, a_scales >> (block * INDEX_BYTE_BITS),
                                   own_scales >> (block * INDEX_BYTE_BITS));
                    chain->x += sums.x;
                    chain->y += sums.y;
                    chain->z += sums.z;
                    chain->w += sums.w;
                    continue;
                }
                if (form == INDEX_FORM_WHOLE) {          /* under 2^13 in magnitude: exact as a float */
                    int4 whole = make_int4(0, 0, 0, 0);
                    index_mma_whole(&whole, a, first, second);
                    sums = make_float4((float)whole.x, (float)whole.y, (float)whole.z, (float)whole.w);
                    bias = INDEX_WHOLE_FIELD_BIAS;
                } else {
                    index_mma_e4m3(&sums, a, first, second);
                }
                chain->x = fmaf(sums.x, index_power_of_two(low_code + even_code - bias), chain->x);
                chain->y = fmaf(sums.y, index_power_of_two(low_code + odd_code - bias), chain->y);
                chain->z = fmaf(sums.z, index_power_of_two(high_code + even_code - bias), chain->z);
                chain->w = fmaf(sums.w, index_power_of_two(high_code + odd_code - bias), chain->w);
            }
        }
    }
}

/* A head tile's four heads of one key: (r w + r w) + (r w + r w); tree 1 (the break instance alone): another
 * association where the row's index in its group is odd. */
template <int tree>
__device__ __forceinline__ float index_heads_of_lane(float low_first, float low_second, float high_first,
                                                     float high_second, float4 weight, uint32_t row)
{
    const float first = index_relu(low_first) * weight.x, second = index_relu(low_second) * weight.y;
    const float third = index_relu(high_first) * weight.z, fourth = index_relu(high_second) * weight.w;
    return tree != 0 && (row & 1u) != 0 ? ((first + second) + third) + fourth : (first + second) + (third + fourth);
}

/* The rows' scores of one tile: the butterfly over the lane groups, one rounding, -inf at or beyond an extent; lane
 * group r stores row r's two scores of its lane place: a row's 16 bytes from four lanes. */
template <int group_rows, int tree>
__device__ __forceinline__ void index_tile_scores(const struct index_scan_arguments *arguments,
                                                  float4 dots[][INDEX_HEAD_TILES], const float4 *weights,
                                                  const uint32_t *extents, struct index_place place,
                                                  uint32_t first_entry, uint32_t slot, uint32_t lane)
{
    const uint32_t lane_group = lane / INDEX_LANE_PLACES, lane_place = lane % INDEX_LANE_PLACES;
    const uint32_t entry = first_entry + lane_place * INDEX_COLUMNS_PER_LANE;
    uint32_t mine = 0;
#pragma unroll
    for (uint32_t row = 0; row < group_rows; row++) {
        if (row < place.rows) {
            const float4 low = dots[row][0], high = dots[row][1];
            float even = index_heads_of_lane<tree>(low.x, low.z, high.x, high.z, weights[row], row);
            float odd = index_heads_of_lane<tree>(low.y, low.w, high.y, high.w, weights[row], row);
#pragma unroll
            for (uint32_t distance = INDEX_LANE_PLACES; distance < INDEX_LANES; distance <<= 1u) {
                even += __shfl_xor_sync(INDEX_EVERY_LANE, even, (int)distance);
                odd += __shfl_xor_sync(INDEX_EVERY_LANE, odd, (int)distance);
            }
            const uint32_t even_bits = entry < extents[row] ? (uint32_t)__half_as_ushort(__float2half_rn(even))
                                                            : (uint32_t)INDEX_NEGATIVE_INFINITY;
            const uint32_t odd_bits = entry + 1u < extents[row] ? (uint32_t)__half_as_ushort(__float2half_rn(odd))
                                                                : (uint32_t)INDEX_NEGATIVE_INFINITY;
            mine = lane_group == row ? even_bits | (odd_bits << INDEX_HALF_BITS) : mine;
        }
    }
    if (lane_group < place.rows) {
        *(uint32_t *)(arguments->scores + (size_t)(place.first_row + lane_group) * arguments->capacity + slot +
                      lane_place * INDEX_COLUMNS_PER_LANE) = mine;
    }
}

/* One block of a scan: the group's queries into shared memory (E2M1 turns their codes there), then each warp its
 * tiles w, w + warps, ... of the segment. A listed scan (one row) takes tile k of its range from the row's list and
 * stores it at 8 k of the row's candidate scores. */
template <int form, int group_rows, int rows_fast, int listed, int tree>
__device__ __forceinline__ void index_scan_block(const struct index_scan_arguments *arguments, uint4 *queries)
{
    const uint32_t lane = threadIdx.x % INDEX_LANES, warp = threadIdx.x / INDEX_LANES;
    const uint32_t warps = blockDim.x / INDEX_LANES, lane_group = lane / INDEX_LANE_PLACES;
    const uint32_t query_loads = index_query_bytes(form) / INDEX_LOAD_BYTES;
    const uint32_t code_loads = INDEX_QUERY_SCALES_AT / INDEX_LOAD_BYTES;
    uint32_t extents[group_rows];
    float4 weights[group_rows];
    const struct index_place place = index_block_place<group_rows, rows_fast, listed>(arguments, extents);
    const uint4 *source = (const uint4 *)arguments->queries + (size_t)place.first_row * query_loads;
#pragma unroll
    for (uint32_t row = 0; row < group_rows; row++) {
        const float *weight = arguments->weights + (size_t)(place.first_row + row) * INDEX_HEADS + lane_group;
        weights[row] = row < place.rows ? make_float4(weight[0], weight[INDEX_HALF_TILE], weight[INDEX_HEADS_PER_TILE],
                                                      weight[INDEX_HEADS_PER_TILE + INDEX_HALF_TILE])
                                        : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    for (uint32_t index = threadIdx.x; index < place.rows * query_loads; index += blockDim.x) {
        queries[index] = form == INDEX_FORM_E2M1 && index % query_loads < code_loads ? index_e2m1_load(source[index])
                                                                                    : source[index];
    }
    __syncthreads();
    for (uint32_t tile = place.first_tile + warp; tile < place.end_tile; tile += warps) {
        const uint32_t key_tile = listed ? arguments->listed[(size_t)place.first_row * INDEX_CANDIDATE_BLOCKS + tile]
                                         : tile;
        const unsigned char *keys = arguments->keys + (size_t)key_tile * index_tile_bytes(form);
        float4 dots[group_rows][INDEX_HEAD_TILES];
#pragma unroll
        for (uint32_t row = 0; row < group_rows; row++) {
            dots[row][0] = dots[row][1] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        if (form == INDEX_FORM_HALF) {
            index_dots_half<group_rows>(keys, queries, place.rows, lane, dots);
        } else {
            index_dots_codes<form, group_rows>(keys, queries, place.rows, lane, dots);
        }
        index_tile_scores<group_rows, tree>(arguments, dots, weights, extents, place, key_tile * INDEX_KEYS_PER_TILE,
                                            tile * INDEX_KEYS_PER_TILE, lane);
    }
}

/* One kernel per choice, the body the same function; no launch bound, as no kernel of the engine states one. */
#define INDEX_SCAN_KERNEL(kernel_name, form, group_rows, rows_fast, listed, tree)                                    \
    __global__ void kernel_name(struct index_scan_arguments arguments)                                               \
    {                                                                                                                \
        extern __shared__ uint4 index_shared_queries[];                                                              \
        index_scan_block<form, group_rows, rows_fast, listed, tree>(&arguments, index_shared_queries);               \
    }

/* ---- the selection ---- */

/* Two fp16 scores of a word as their order keys. */
__device__ __forceinline__ uint32_t index_order_keys(uint32_t pair)
{
    const uint32_t signs = (pair >> INDEX_HALF_SIGN_SHIFT) & INDEX_LOW_BITS_OF_HALVES;
    return pair ^ (signs * INDEX_ORDER_FLIP_NEGATIVE | INDEX_ORDER_SIGNS);
}
/* An exclusive prefix over the lanes of a warp; *total: of all of them. */
__device__ __forceinline__ uint32_t index_lanes_before(uint32_t value, uint32_t lane, uint32_t *total)
{
    uint32_t sum = value;
#pragma unroll
    for (uint32_t distance = 1; distance < INDEX_LANES; distance <<= 1u) {
        const uint32_t below = __shfl_up_sync(INDEX_EVERY_LANE, sum, distance);
        sum += lane >= distance ? below : 0u;
    }
    *total = __shfl_sync(INDEX_EVERY_LANE, sum, INDEX_LANES - 1);
    return sum - value;
}

struct index_elements {                          /* what one block selects among: elements 0 .. count - 1 */
    const uint16_t *keys;                        /* a row's scores, or (merge) its spans' order keys */
    const uint32_t *positions, *span_counts;     /* merge: of its spans' candidates */
    uint32_t count, first_position;              /* a row or span: its first element's position */
};
/* Elements base .. base + 7 (base a multiple of 8): their order keys, and a bit each if the element is there. */
template <int source>
__device__ __forceinline__ uint32_t index_load_keys(const struct index_elements *elements, uint32_t base,
                                                    uint32_t keys[INDEX_KEYS_PER_LOAD])
{
    uint32_t present = 0;
    if (base < elements->count) {
        const uint4 load = *(const uint4 *)(elements->keys + base);
        const uint32_t pairs[INDEX_PAIRS_PER_LOAD] = {load.x, load.y, load.z, load.w};
        const uint32_t held = source == INDEX_SOURCE_CANDIDATES
                                  ? elements->span_counts[base / INDEX_SELECTED] - base % INDEX_SELECTED
                                  : elements->count - base;
#pragma unroll
        for (uint32_t element = 0; element < INDEX_KEYS_PER_LOAD; element++) {
            const uint32_t stored = pairs[element / INDEX_COLUMNS_PER_LANE];
            const uint32_t pair = source == INDEX_SOURCE_CANDIDATES ? stored : index_order_keys(stored);
            keys[element] = (pair >> ((element & 1u) * INDEX_HALF_BITS)) & ((1u << INDEX_HALF_BITS) - 1u);
            present |= (int32_t)held > (int32_t)element ? 1u << element : 0u;
        }
    }
    return present;
}

/* bins[d] = the elements whose digit (key >> shift, a byte) is d, of those whose key >> 8 is `high` where shift is
 * 0. Per warp: lanes with one digit are joined by the match instruction and the lowest adds their count. */
template <int source>
__device__ __forceinline__ void index_histogram(const struct index_elements *elements, uint32_t shift, uint32_t high,
                                                uint32_t (*warp_bins)[INDEX_BINS], uint32_t *bins)
{
    const uint32_t lane = threadIdx.x % INDEX_LANES, warp = threadIdx.x / INDEX_LANES;
    const uint32_t stride = blockDim.x * INDEX_KEYS_PER_LOAD;
    for (uint32_t bin = lane; bin < INDEX_BINS; bin += INDEX_LANES) {
        warp_bins[warp][bin] = 0;
    }
    __syncwarp();
    for (uint32_t first = 0; first < elements->count; first += stride) {      /* the same turns on every lane */
        uint32_t keys[INDEX_KEYS_PER_LOAD];
        const uint32_t present = index_load_keys<source>(elements, first + threadIdx.x * INDEX_KEYS_PER_LOAD, keys);
#pragma unroll
        for (uint32_t element = 0; element < INDEX_KEYS_PER_LOAD; element++) {
            const uint32_t digit = (keys[element] >> shift) & INDEX_DIGIT_MASK;
            const bool wanted = (present >> element & 1u) != 0 &&
                                (shift != 0 || keys[element] >> INDEX_BYTE_BITS == high);
            const uint32_t peers = __match_any_sync(INDEX_EVERY_LANE, wanted ? digit : (uint32_t)INDEX_BINS);
            if (wanted && (peers & ((1u << lane) - 1u)) == 0) {
                warp_bins[warp][digit] += (uint32_t)__popc(peers);
            }
            __syncwarp();
        }
    }
    __syncthreads();
    for (uint32_t bin = threadIdx.x; bin < INDEX_BINS; bin += blockDim.x) {
        uint32_t total = 0;
        for (uint32_t other = 0; other < INDEX_SELECT_WARPS; other++) {
            total += warp_bins[other][bin];
        }
        bins[bin] = total;
    }
    __syncthreads();
}

/* Warp 0: the bin where `need` elements counted from the top are reached, and the count above it, into cut[0, 1]. */
__device__ __forceinline__ void index_cut(const uint32_t *bins, uint32_t need, uint32_t *cut)
{
    const uint32_t lane = threadIdx.x % INDEX_LANES;
    uint32_t lane_sum = 0, total = 0;
    for (uint32_t bin = 0; bin < INDEX_BINS_PER_LANE; bin++) {
        lane_sum += bins[INDEX_BINS - 1u - lane * INDEX_BINS_PER_LANE - bin];
    }
    uint32_t above = index_lanes_before(lane_sum, lane, &total);
    for (uint32_t bin = 0; above < need && need <= above + lane_sum && bin < INDEX_BINS_PER_LANE; bin++) {
        const uint32_t digit = INDEX_BINS - 1u - lane * INDEX_BINS_PER_LANE - bin;
        if (above + bins[digit] >= need) {
            cut[0] = digit;
            cut[1] = above;
            break;
        }
        above += bins[digit];
    }
}

/* One block: the top min(512, present elements) by (key descending, element ascending) in element order. Elements
 * are in position order in every source, so element order is position order. */
template <int source>
__device__ __forceinline__ void index_select_block(const struct index_select_arguments *arguments)
{
    __shared__ uint32_t warp_bins[INDEX_SELECT_WARPS][INDEX_BINS], bins[INDEX_BINS], cut[INDEX_CUT_FIELDS];
    __shared__ uint32_t warp_greater[INDEX_SELECT_WARPS], warp_ties[INDEX_SELECT_WARPS];
    __shared__ uint32_t outputs_before[INDEX_SELECT_WARPS], ties_before[INDEX_SELECT_WARPS], present_total;
    const uint32_t lane = threadIdx.x % INDEX_LANES, warp = threadIdx.x / INDEX_LANES;
    const uint32_t row = source == INDEX_SOURCE_SPAN ? blockIdx.y : blockIdx.x;
    const uint32_t span = source == INDEX_SOURCE_SPAN ? blockIdx.x : 0u, extent = arguments->extents[row];
    const size_t spans_at = (size_t)row * arguments->spans;
    struct index_elements elements = {arguments->scores + (size_t)row * arguments->capacity, NULL, NULL, extent, 0};
    if (source == INDEX_SOURCE_SPAN) {
        elements.first_position = span * INDEX_SPAN_ENTRIES;
        elements.keys += elements.first_position;
        elements.count = extent > elements.first_position ? extent - elements.first_position : 0u;
        elements.count = elements.count < INDEX_SPAN_ENTRIES ? elements.count : (uint32_t)INDEX_SPAN_ENTRIES;
    } else if (source == INDEX_SOURCE_CANDIDATES) {
        elements.keys = arguments->span_keys + spans_at * INDEX_SELECTED;
        elements.positions = arguments->span_positions + spans_at * INDEX_SELECTED;
        elements.span_counts = arguments->span_counts + spans_at;
        elements.count = arguments->spans * INDEX_SELECTED;
    }
    if (warp == 0) {
        uint32_t total = 0;
        index_lanes_before(source != INDEX_SOURCE_CANDIDATES ? (lane == 0 ? elements.count : 0u)
                           : lane < arguments->spans ? elements.span_counts[lane] : 0u, lane, &total);
        if (lane == 0) {
            present_total = total;
        }
    }
    __syncthreads();
    const uint32_t need = present_total < INDEX_SELECTED ? present_total : (uint32_t)INDEX_SELECTED;
    uint32_t threshold = 0, ties_needed = INDEX_SELECTED;       /* at most 512 present: every one of them */
    if (present_total > INDEX_SELECTED) {
        index_histogram<source>(&elements, INDEX_BYTE_BITS, 0, warp_bins, bins);
        if (warp == 0) {
            index_cut(bins, need, cut);
        }
        __syncthreads();
        const uint32_t high = cut[0], above_high = cut[1];
        index_histogram<source>(&elements, 0, high, warp_bins, bins);
        if (warp == 0) {
            index_cut(bins, need - above_high, cut);
        }
        __syncthreads();
        threshold = high << INDEX_BYTE_BITS | cut[0];
        ties_needed = need - above_high - cut[1];
    }
    /* a warp a run of whole turns (32 lanes of 8 elements) of the elements, in order: first its counts */
    const uint32_t turn = INDEX_LANES * INDEX_KEYS_PER_LOAD;
    const uint32_t run = (elements.count / INDEX_SELECT_WARPS / turn + 1u) * turn, first = warp * run;
    uint32_t greater = 0, ties = 0;
    for (uint32_t base = first + lane * INDEX_KEYS_PER_LOAD; base < first + run; base += turn) {
        uint32_t keys[INDEX_KEYS_PER_LOAD];
        const uint32_t present = index_load_keys<source>(&elements, base, keys);
#pragma unroll
        for (uint32_t element = 0; element < INDEX_KEYS_PER_LOAD; element++) {
            greater += (present >> element & 1u) != 0 && keys[element] > threshold ? 1u : 0u;
            ties += (present >> element & 1u) != 0 && keys[element] == threshold ? 1u : 0u;
        }
    }
    greater = __reduce_add_sync(INDEX_EVERY_LANE, greater);
    ties = __reduce_add_sync(INDEX_EVERY_LANE, ties);
    if (lane == 0) {
        warp_greater[warp] = greater;
        warp_ties[warp] = ties;
    }
    __syncthreads();
    if (warp == 0) {                             /* the ties a run takes are the first of the cut's value */
        uint32_t total = 0;
        const uint32_t before = index_lanes_before(warp_ties[lane], lane, &total);
        const uint32_t left = ties_needed > before ? ties_needed - before : 0u;
        const uint32_t taken = left < warp_ties[lane] ? left : warp_ties[lane];
        ties_before[lane] = before;
        outputs_before[lane] = index_lanes_before(warp_greater[lane] + taken, lane, &total);
    }
    __syncthreads();
    uint32_t outputs_done = outputs_before[warp], ties_done = ties_before[warp];
    for (uint32_t base = first + lane * INDEX_KEYS_PER_LOAD; base < first + run; base += turn) {
        uint32_t keys[INDEX_KEYS_PER_LOAD], lane_ties = 0, total = 0;
        const uint32_t present = index_load_keys<source>(&elements, base, keys);
        uint32_t chosen = 0;
#pragma unroll
        for (uint32_t element = 0; element < INDEX_KEYS_PER_LOAD; element++) {
            lane_ties += (present >> element & 1u) != 0 && keys[element] == threshold ? 1u : 0u;
        }
        uint32_t tie_rank = ties_done + index_lanes_before(lane_ties, lane, &total);
        ties_done += total;
#pragma unroll
        for (uint32_t element = 0; element < INDEX_KEYS_PER_LOAD; element++) {
            const bool there = (present >> element & 1u) != 0, tie = there && keys[element] == threshold;
            chosen |= (there && keys[element] > threshold) || (tie && tie_rank < ties_needed) ? 1u << element : 0u;
            tie_rank += tie ? 1u : 0u;
        }
        uint32_t output = outputs_done + index_lanes_before((uint32_t)__popc(chosen), lane, &total);
        outputs_done += total;
        for (uint32_t element = 0; chosen != 0 && element < INDEX_KEYS_PER_LOAD; element++) {
            if ((chosen >> element & 1u) == 0) {
                continue;
            }
            const uint32_t position = source == INDEX_SOURCE_CANDIDATES ? elements.positions[base + element]
                                                                       : elements.first_position + base + element;
            if (source == INDEX_SOURCE_SPAN) {
                arguments->span_keys[(spans_at + span) * INDEX_SELECTED + output] = (uint16_t)keys[element];
                arguments->span_positions[(spans_at + span) * INDEX_SELECTED + output] = position;
            } else {
                arguments->selected[(size_t)row * INDEX_SELECTED + output] = (int32_t)position;
            }
            output++;
        }
    }
    for (uint32_t output = need + threadIdx.x; source != INDEX_SOURCE_SPAN && output < INDEX_SELECTED;
         output += blockDim.x) {
        arguments->selected[(size_t)row * INDEX_SELECTED + output] = -1;
    }
    if (source == INDEX_SOURCE_SPAN && threadIdx.x == 0) {
        arguments->span_counts[spans_at + span] = need;
    }
}

__global__ void index_select_one_block(struct index_select_arguments arguments)
{
    index_select_block<INDEX_SOURCE_ROW>(&arguments);
}
__global__ void index_select_spans(struct index_select_arguments arguments)
{
    index_select_block<INDEX_SOURCE_SPAN>(&arguments);
}
__global__ void index_select_merge(struct index_select_arguments arguments)
{
    index_select_block<INDEX_SOURCE_CANDIDATES>(&arguments);
}

#endif /* INDEX_SCAN_KERNELS_CUH */
