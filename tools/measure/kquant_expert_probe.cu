/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * kquant_expert_probe: what the routed experts of one MoE layer cost at 1 to 8 rows when an expert's three matrices are
 * K-quant blocks of the GGUF file format, Q2_K (2.625 bits per weight) or Q3_K (3.4375), as a published quantization
 * of the model stores them. It is shared_rows_expert_probe (the checkpoint's own 4-bit weights) with another weight
 * format: same shapes, tensor-core instruction with the rows of a job as the columns of operand B (an expert is read
 * once whatever the number of its rows), activation quantization, SwiGLU, row sum, cases, timing and table, so that
 * the two logs lie side by side. There a 170HX is bound by arithmetic per weight and the PRO 6000 by bytes read:
 * whether a scale per 16 weights and, for Q2_K, a minimum per 16 cost more than the bytes save is what a run says.
 * Not as there: the way of the steps through the experts (route_expert: that probe's left some of these smaller
 * experts in a PRO 6000's cache); down's block, 3 warps of 48 instructions and not 4 of 36, a slice being whole blocks
 * of 256 inputs and down 9 wide, so that what differs in down's time is in part the warps' (gate and up: 8 warps of
 * 80 in both); no time of the quantizations alone: what Q2_K's adds to them (X of step 3) is in a layer's time only.
 * The format (canonical_fields has every bit): 256 consecutive inputs of one output are a block, 16 a group.
 *   Q2_K, 84 bytes a block: weight = d * scale * code - dmin * minimum; 2-bit codes, a 4-bit scale and minimum a group.
 *   Q3_K, 110 bytes: weight = d * (scale - 32) * code; 3-bit codes -4 .. 3, a 6-bit scale a group. d, dmin: fp16.
 * Lane order (pack_matrix writes it, tile_slice_product reads it): the 16 blocks a tile of 16 outputs has for the same
 * 256 inputs are one record, in the order the instruction's lanes take them (an instruction consumes a group). Q2_K:
 * 16 * 84 bytes, the file's bits and no others. Q3_K: 16 * 114, 3.6% more than the file (3.5625 bits per weight): a
 * 6-bit scale has a byte, so that a lane loads a half block's scales at once; packed to 6 bits they lie across words.
 *   codes         [half block][lane], 16 bytes: the lane's eight 2-bit codes of each of the half's eight instructions
 *   signs         Q3_K, [half block][lane], 8 bytes: a bit per code; a negative code is kept as its magnitude less one
 *   group scales  [half block][lane group], 16 bytes: a byte per instruction and output (Q2_K: the file's byte)
 *   block scales  [output]: d, then Q2_K's dmin, as the file's fp16
 * The arithmetic is defined here: nothing computes with these blocks on quantized activations to be equal to. a is an
 * input's E4M3 value and 2^E the scale of its block of 32, both from the reference's activation quantization.
 *   1. G = the sum over a group of code * a: one instruction from a zero accumulator. Its operands are exact fp16 (the
 *      code as a whole number in the mantissa of a subnormal, times 2^-24; the E4M3 value times 2^-8) and every
 *      partial sum is a multiple of 2^-9 below 2^15, which fp32 holds: G is exact in whatever order it is added, if
 *      the unit multiplies subnormal operands exactly and keeps 24 bits under its largest addend (assumed: see below);
 *   2. per pair of groups, which share E: R = fma(G1, s1, fma(G0, s0, 0)), then P = fma(R, 2^E, P), P being the
 *      block's and s the group's scale as a whole number (Q3_K: less 32);
 *   3. Q2_K, per group: M = fma(X, m, M), M being the block's, m the group's minimum and X = 2^E * the sum of the
 *      group's 16 a: exact in fp32, the same for every output, made once per vector by the quantization kernel;
 *   4. per block: T = fma(P, d, T), then for Q2_K T = fma(M, -dmin, T); T belongs to a slice of consecutive blocks;
 *   5. a product is its slices added in slice order; then bf16 casts, SwiGLU and the row sum as in the other probe.
 * Every step after 1 is one correctly rounded fp32 operation in a fixed order, each written fmaf: no flag, timing or
 * number of rows changes a bit. There is one decode: a code is placed by one shift and one mask (b08 measured a table
 * of placed codes slower for 4-bit ones), and multiplying the scale into the operand was weighed and left out: it
 * trades four multiply-adds and two conversions per instruction for four integer multiplies and gives up the exact G.
 * Checked on every run, for two (row, expert) pairs of every case, step by step: the card's quantized vectors and
 * Q2_K's group sums, bit for bit, against a host quantizer written from the definition; the hidden vector and the
 * output against double precision from the CANONICAL blocks decoded by the format's definition, fed with the card's
 * own quantized vector; every row launched alone against the same row in its launch: 0 words may differ. The second
 * holds the format's meaning and the packing to 2e-3 of the rms, not the bits of steps 1 to 5: a unit that rounded G
 * would pass, and what is to return a card's bits by these steps has to be compared with a card bit for bit first.
 * Each check is shown able to fail: --corrupt overwrites a megabyte of the first checked expert's gate matrix and of
 * its down matrix (hidden vector and output must differ); --break-exactness makes the order of a block's slices
 * depend on the rows of its job (rows that share an expert must differ, no others); --break-quantization moves the
 * card's scales one power of two down (quantized bytes, exponents and, a block's largest values being clamped, group
 * sums must differ; the products, made from the card's own vector, must not); --break-route asks for five launches,
 * each past one bound of what a kernel indexes by: an expert, a pair, a job's columns, the jobs, the rows (each must
 * be refused on the host, where a launch gets its jobs: no card is touched). Measurement tool: not part of the engine.
 *   kquant_expert_probe [--type q2_k|q3_k] [--overlap 0|50|100] [--steps N] [any of the four flags above]
 */
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum probe_constant {
    /* the model and a launch: a pair is a row at one of its experts (row * 6 + slot); a stage is gate with up, which
     * share the input, or down; its warps share a tile's input width in slices of whole blocks */
    HIDDEN_SIZE = 5120, EXPERT_INNER_SIZE = 2304, EXPERTS_PER_ROW = 6, MAX_ROWS = 8, EXPERT_COUNT = 128,
    MAX_PAIRS = MAX_ROWS * EXPERTS_PER_ROW, MATRIX_KINDS = 3, GATE_MATRIX = 0, UP_MATRIX = 1, DOWN_MATRIX = 2,
    GATE_UP_MATRICES = 2, DOWN_MATRICES = 1, GATE_UP_SLICES = 4, DOWN_SLICES = 3, PAIRS_PER_HIDDEN_VECTOR = 1,
    MATRIX_WEIGHTS = HIDDEN_SIZE * EXPERT_INNER_SIZE, FOUR_BIT_PROBE_DOWN_SLICES = 4,
    /* the file's blocks (canonical_fields): Q2_K is 16 bytes of scales and minimums, 64 of codes, d, dmin; Q3_K
     * 32 bytes of high code bits, 64 of low ones, 12 of scales (as words: two of low halves, one of high pairs), d */
    KQUANT_Q2_K = 0, KQUANT_Q3_K, KQUANT_TYPES, BLOCK_WEIGHTS = 256, GROUP_WEIGHTS = 16, HALVES_PER_BLOCK = 2,
    GROUPS_PER_BLOCK = BLOCK_WEIGHTS / GROUP_WEIGHTS, HALF_WEIGHTS = BLOCK_WEIGHTS / HALVES_PER_BLOCK,
    GROUPS_PER_HALF = GROUPS_PER_BLOCK / HALVES_PER_BLOCK, GROUPS_PER_PAIR = 2,
    PAIRS_PER_HALF = GROUPS_PER_HALF / GROUPS_PER_PAIR, HALF_CODE_BYTES = 32, CODE_BITS = 2, CODE_MASK = 3,
    CANONICAL_BLOCKS_PER_MATRIX = MATRIX_WEIGHTS / BLOCK_WEIGHTS, Q2_K_BLOCK_BYTES = 84, Q2_K_CODES_AT = 16,
    Q2_K_D_AT = 80, Q3_K_BLOCK_BYTES = 110, Q3_K_CODES_AT = 32, Q3_K_SCALES_AT = 96, Q3_K_SCALE_BYTES = 12,
    Q3_K_D_AT = 108, Q3_K_SCALE_BIAS = 32, Q3_K_LOW_SCALE_WORDS = 2, Q3_K_CODE_OFFSET = 4,
    /* fp16: exponent field 0 is mantissa * 2^-24, otherwise (1024 + mantissa) * 2^(field - 25). Synthetic d and dmin
     * are the bit patterns from that of 2^-10 up to that of 2^-7: 9.8e-4 .. 7.8e-3, all eleven significant bits used */
    FP16_BYTES = 2, FP16_EXPONENT_SHIFT = 10, FP16_EXPONENT_MASK = 31, FP16_MANTISSA_MASK = 1023,
    FP16_IMPLICIT_ONE = 1024, FP16_SUBNORMAL_EXPONENT = 24, FP16_NORMAL_EXPONENT = 25, FP16_SIGN_SHIFT = 15,
    D_BITS_SMALLEST = 0x1400, D_BITS_SPAN = 0x0C00, CANONICAL_SEED = 0x243F6A88,
    /* the instruction, m16n8k16: a lane holds four registers of weights, two of inputs, four sums: lane_instruction */
    MMA_ROWS = 16, MMA_HALF_ROWS = MMA_ROWS / 2, MMA_INPUTS = GROUP_WEIGHTS, MMA_HALF_INPUTS = MMA_INPUTS / 2,
    LANES_PER_WARP = 32, LANES_PER_LANE_GROUP = 4, HALVES_PER_REGISTER = 2, WEIGHT_REGISTERS = 4, INPUT_REGISTERS = 2,
    LANE_SUMS = 4, LANE_COLUMNS = 2, LANE_OUTPUTS = 2, WORDS_PER_LOAD = 4, BYTES_PER_WORD = 4, BITS_PER_BYTE = 8,
    BITS_PER_HALF = 16, BYTE_MASK = 0xFF, NIBBLE_BITS = 4, NIBBLE_MASK = 15,
    /* a record in lane order, as the header has it: a code word's halves hold 8 bits an instruction, a sign word's 4 */
    CODE_BYTES_PER_RECORD = HALVES_PER_BLOCK * LANES_PER_WARP * WORDS_PER_LOAD * BYTES_PER_WORD,
    SIGN_BYTES_PER_RECORD = CODE_BYTES_PER_RECORD / 2, SCALE_BYTES_PER_LANE_GROUP = GROUPS_PER_HALF * LANE_OUTPUTS,
    SCALE_BYTES_PER_RECORD = HALVES_PER_BLOCK * MMA_HALF_ROWS * SCALE_BYTES_PER_LANE_GROUP,
    Q2_K_GROUP_SCALES_AT = CODE_BYTES_PER_RECORD, Q3_K_GROUP_SCALES_AT = CODE_BYTES_PER_RECORD + SIGN_BYTES_PER_RECORD,
    Q2_K_BLOCK_SCALES_AT = Q2_K_GROUP_SCALES_AT + SCALE_BYTES_PER_RECORD, Q2_K_BLOCK_SCALES = 2,
    Q3_K_BLOCK_SCALES_AT = Q3_K_GROUP_SCALES_AT + SCALE_BYTES_PER_RECORD, Q3_K_BLOCK_SCALES = 1,
    Q2_K_RECORD_BYTES = Q2_K_BLOCK_SCALES_AT + MMA_ROWS * Q2_K_BLOCK_SCALES * FP16_BYTES,
    Q3_K_RECORD_BYTES = Q3_K_BLOCK_SCALES_AT + MMA_ROWS * Q3_K_BLOCK_SCALES * FP16_BYTES, BLOCK_SCALE_D = 0,
    BLOCK_SCALE_DMIN, CODE_PAIR_MASK = 0x00030003, SIGN_PAIR_MASK = 0x00010001,
    CODE_BITS_PER_MMA_STEP = WEIGHT_REGISTERS * CODE_BITS, SIGN_BITS_PER_MMA_STEP = WEIGHT_REGISTERS,
    MMA_STEPS_PER_SIGN_WORD = BITS_PER_HALF / SIGN_BITS_PER_MMA_STEP,
    SIGN_WORDS_PER_HALF = GROUPS_PER_HALF / MMA_STEPS_PER_SIGN_WORD,
    TILE_BLOCKS_PER_MATRIX = MATRIX_WEIGHTS / (MMA_ROWS * BLOCK_WEIGHTS),
    /* activations: a block of 32 under one scale, 2^(E8M0 code - 127), is a pair of groups; a 16-byte load is four
     * instructions; ((b & 0x80) << 8) | ((b & 0x7F) << 7) is the fp16 of E4M3 byte b times 2^-8: PAIR masks do two */
    ACTIVATION_BLOCK = GROUP_WEIGHTS * GROUPS_PER_PAIR, MMA_STEPS_PER_LOAD = 4,
    LOADS_PER_HALF = GROUPS_PER_HALF / MMA_STEPS_PER_LOAD, PAIRS_PER_LOAD = MMA_STEPS_PER_LOAD / GROUPS_PER_PAIR,
    WORDS_PER_ACTIVATION_BLOCK = LANES_PER_LANE_GROUP * GROUPS_PER_PAIR, E4M3_SIGN_BIT = 0x80,
    E4M3_NAN_MAGNITUDE = 0x7F, E4M3_EXPONENT_SHIFT = 3, E4M3_EXPONENT_MASK = 15, E4M3_EXPONENT_BIAS = 7,
    E4M3_MANTISSA_MASK = 7, E4M3_IMPLICIT_ONE = 8, E4M3_PLACEMENT_EXPONENT = 8, E4M3_PAIR_SIGN_MASK = 0x00800080,
    E4M3_PAIR_MAGNITUDE_MASK = 0x007F007F, E4M3_MAGNITUDE_SHIFT = 7, E4M3_SIGN_FROM_FP32_SHIFT = 24, E8M0_BIAS = 127,
    FP32_EXPONENT_SHIFT = 23, FP32_EXPONENT_BIAS = 127, FP32_MANTISSA_MASK = 0x007FFFFF, BF16_ROUNDING_BIAS = 0x7FFF,
    BF16_SIGNIFICANT_BITS = 8,
    /* quantizing a value of fp32 exponent field f: codes are 2^(f - 130) apart, the code is 8 * (f - 120) + steps - 8.
     * An instruction's sum is the true one times 2^-(24 + 8): 2^(E + 32) has E's code plus 32 for its fp32 field */
    E4M3_STEPS_EXPONENT_BASE = 2 * FP32_EXPONENT_BIAS + E4M3_EXPONENT_SHIFT,
    E4M3_STEP_FIELD_OFFSET = E4M3_STEPS_EXPONENT_BASE - 2 * FP32_EXPONENT_BIAS,
    E4M3_CODE_BASE = ((FP32_EXPONENT_BIAS - E4M3_EXPONENT_BIAS) << E4M3_EXPONENT_SHIFT) + E4M3_IMPLICIT_ONE,
    ACTIVATION_POWER_FIELD_OFFSET = FP32_EXPONENT_BIAS - E8M0_BIAS + FP16_SUBNORMAL_EXPONENT + E4M3_PLACEMENT_EXPONENT,
    /* the hash; synthetic inputs (multiples of 1/64 in -2 .. 2, divided per block of 32 by 2^0 .. 2^6, or by 2^16 for
     * one block in eight: below the reference's floor) and route weights (1/64 .. 32/64); the run: step 0 is checked;
     * every 13th output of down is compared; stale bytes fill the outputs first: an unwritten word cannot match */
    MIX_SHIFT_OUTER = 16, MIX_SHIFT_INNER = 15, MIX_MULTIPLIER = 0x7feb352d, INPUT_SEED = 0x1234567, STEP_STRIDE = 977,
    INPUT_VALUE_STEPS = 255, INPUT_BLOCK_KINDS = 8, INPUT_TINY_SHIFT = 16, ROUTE_SEED = 0xABCDEF,
    ROUTE_WEIGHT_STEPS = 32, EXPERT_HALVES = 2, OVERLAP_STEP = 50, PERCENT = 100, DEFAULT_STEP_COUNT = 200,
    MAX_STEP_COUNT = 1000000, DECIMAL_BASE = 10, WARM_UP_STEPS = 10, CHECKED_STEP = 0, CHECKED_PAIRS = 2,
    DOWN_CHECK_STRIDE = 13, CORRUPTED_BYTES = 1024 * 1024, CORRUPTED_BYTE_VALUE = 0x5A, STALE_BYTE_SHARED = 0xFF,
    STALE_BYTE_ALONE = 0xEE, CASE_NAME_BYTES = 48, SM_80_MAJOR = 8, SM_120_MAJOR = 12, FILL_THREADS_PER_BLOCK = 256,
    BREAK_EXPERT = 1, BREAK_PAIR, BREAK_COLUMNS, BREAK_JOBS, BREAK_ROWS, BREAK_ROUTE_END    /* --break-route, in turn */
};
static_assert(HIDDEN_SIZE % (BLOCK_WEIGHTS * GATE_UP_SLICES) + EXPERT_INNER_SIZE % (BLOCK_WEIGHTS * DOWN_SLICES) == 0
              && HIDDEN_SIZE % MMA_ROWS == 0 && EXPERT_INNER_SIZE % MMA_ROWS == 0 && CHECKED_STEP < EXPERT_HALVES
              && Q2_K_RECORD_BYTES == MMA_ROWS * Q2_K_BLOCK_BYTES          /* the file's size; Q3_K: a byte per scale */
              && Q3_K_RECORD_BYTES == MMA_ROWS * (Q3_K_BLOCK_BYTES + GROUPS_PER_BLOCK - Q3_K_SCALE_BYTES)
              && CORRUPTED_BYTES <= TILE_BLOCKS_PER_MATRIX * Q2_K_RECORD_BYTES,
              "shapes that do not divide, a record not of the header's size, or a checked step past a half's first");
#define STAGE_SLICES(matrix_count) ((matrix_count) == DOWN_MATRICES ? DOWN_SLICES : GATE_UP_SLICES)
#define STAGE_INPUT_SIZE(matrix_count) ((matrix_count) == DOWN_MATRICES ? EXPERT_INNER_SIZE : HIDDEN_SIZE)
#define STAGE_VECTOR_PAIRS(matrix_count) ((matrix_count) == DOWN_MATRICES ? PAIRS_PER_HIDDEN_VECTOR : EXPERTS_PER_ROW)
#define BLOCK_BYTES(kquant) ((kquant) == KQUANT_Q2_K ? Q2_K_BLOCK_BYTES : Q3_K_BLOCK_BYTES)
#define CODES_AT(kquant) ((kquant) == KQUANT_Q2_K ? Q2_K_CODES_AT : Q3_K_CODES_AT)
#define D_AT(kquant) ((kquant) == KQUANT_Q2_K ? Q2_K_D_AT : Q3_K_D_AT)
#define GROUP_SCALES_AT(kquant) ((kquant) == KQUANT_Q2_K ? Q2_K_GROUP_SCALES_AT : Q3_K_GROUP_SCALES_AT)
#define BLOCK_SCALES_AT(kquant) ((kquant) == KQUANT_Q2_K ? Q2_K_BLOCK_SCALES_AT : Q3_K_BLOCK_SCALES_AT)
#define BLOCK_SCALES(kquant) ((kquant) == KQUANT_Q2_K ? Q2_K_BLOCK_SCALES : Q3_K_BLOCK_SCALES)
#define RECORD_BYTES(kquant) ((size_t)((kquant) == KQUANT_Q2_K ? Q2_K_RECORD_BYTES : Q3_K_RECORD_BYTES))
#define MATRIX_BYTES(kquant) (TILE_BLOCKS_PER_MATRIX * RECORD_BYTES(kquant))
#define EXPERT_BYTES(kquant) (MATRIX_KINDS * MATRIX_BYTES(kquant))
#define MOST_CANONICAL_MATRIX_BYTES ((size_t)CANONICAL_BLOCKS_PER_MATRIX * Q3_K_BLOCK_BYTES)
#define QUANTIZED_BYTES(size) ((size) + (size) / ACTIVATION_BLOCK)       /* a vector's codes, then its exponents */
#define ROW_EXPERT_FLOATS ((size_t)EXPERTS_PER_ROW * HIDDEN_SIZE)
#define ROW_SUMS_AT ((size_t)MAX_PAIRS * HIDDEN_SIZE)                      /* a result: [pair][hidden size] outputs, */
#define RESULT_FLOATS (ROW_SUMS_AT + (size_t)MAX_ROWS * HIDDEN_SIZE)       /* then [row][hidden size] sums */
/* word or value `index` of a 16-byte load: a constant where the kernels use it, so no array is indexed at run time */
#define LOAD_PART(load, index) \
    ((index) == 0 ? (load).x : (index) == 1 ? (load).y : (index) == WORDS_PER_LOAD - 1 ? (load).w : (load).z)
#define FP32_SIGN_MASK 0x80000000u
#define BF16_KEPT_MASK 0xFFFF0000u               /* the upper half; the bias and its lowest bit round ties to even */
#define MIX_MULTIPLIER_SECOND 0x846ca68bu
#define SWIGLU_LIMIT 10.0f
#define E4M3_VALUE_DIVISOR 1024.0                /* 2^(bias + mantissa bits) */
#define E4M3_MAXIMUM 448.0f
#define E4M3_MAXIMUM_INVERSE (1.0f / E4M3_MAXIMUM)
#define E4M3_SMALLEST_NORMAL 0.015625f           /* 2^-6: codes below it are 2^-9 apart, as those just above it */
#define AMAX_FLOOR 1.0e-4f                       /* the reference's lower bound of a block's largest magnitude */
#define INPUT_VALUE_OFFSET 128.0f
#define INPUT_VALUE_DIVISOR 64.0f
#define ROUTE_WEIGHT_DIVISOR 64.0f
/* What the card may differ by from the host's double precision: a root of summed squares over the reference's, the
 * other probes' measure and figure. The contract's roundings to fp32, some 40 per block of 256 inputs, come to 1e-7.
 * What the figure has to hold is a cast to bf16 that falls the other way on the card: 2^-7 of one of 2,304 hidden
 * values, 2e-4 of the rms, so a dozen pass. A packing mistake misplaces a sixteenth of the codes or more: 0.1. */
#define RELATIVE_TOLERANCE 2.0e-3
#define CARD_READ_GB_PER_SECOND_SM_80 1440.0     /* card_read_rate_probe, b02: 1,429 and 1,447 on the two 170HX, */
#define CARD_READ_GB_PER_SECOND_SM_120 1647.0    /* and the PRO 6000 */
#define BYTES_PER_MB 1.0e6
#define GB_PER_SECOND_PER_MB_PER_MICROSECOND 1000.0
#define MICROSECONDS_PER_MILLISECOND 1000.0
static const uint32_t row_counts[] = {1, 2, 4, 6, 8};
#define ROW_COUNT_CASES (sizeof(row_counts) / sizeof(row_counts[0]))
static const char *const type_names[KQUANT_TYPES] = {"q2_k", "q3_k"};
struct expert_job {                              /* one expert with the rows routed to it: a block's columns */
    uint32_t expert_index, row_count;            /* row_count: columns in use, 1 .. MAX_ROWS */
    uint32_t pair_index[MAX_ROWS];               /* of each column; a column not in use repeats column 0 */
    float route_weight[MAX_ROWS];
};
struct quantized_vectors {                       /* in front of a product: rows' inputs, or pairs' hidden vectors */
    float *values;                               /* [vector][size]: inputs exact in fp16; hidden vectors cast to bf16 */
    uint32_t *words;                             /* the E4M3 bytes: [vector][load of 64][lane position]: 16 bytes */
    uint8_t *exponents;                          /* [vector][block of 32]: E8M0 codes */
    float *group_sums;                           /* Q2_K: [vector][group of 16]: 2^E times the sum of its E4M3 values */
    uint32_t size;
};
struct lane_loads {                              /* what a lane holds in one half block of its tile: 8 instructions */
    const uint4 *inputs_at;                      /* this half block of the column the lane feeds, */
    const float4 *sums_at[LANE_COLUMNS];         /* and of the group sums of its two result columns */
    uint4 codes, scales;                         /* a word per pair of instructions */
    uint2 signs;                                 /* Q3_K: a word per four instructions */
    uint32_t exponents[LANE_COLUMNS];            /* of its two result columns: an E8M0 code per pair */
    uint4 inputs;                                /* four instructions of the column it feeds: refilled per load */
    float4 sums[LANE_COLUMNS];                   /* Q2_K: the same four groups of its two result columns */
};
static void check_cuda(cudaError_t status, const char *call, int line) {
    if (status != cudaSuccess) {
        fprintf(stderr, "%s:%d: %s failed: %s\n", __FILE__, line, call, cudaGetErrorString(status));
        exit(EXIT_FAILURE);
    }
}
#define CHECK_CUDA(call) check_cuda((call), #call, __LINE__)
/* ---- synthetic inputs and the order of a quantized vector: the same functions on the host and on the card ---- */
__host__ __device__ inline uint32_t mix_bits(uint32_t value) {
    value = (value ^ (value >> MIX_SHIFT_OUTER)) * MIX_MULTIPLIER;
    value = (value ^ (value >> MIX_SHIFT_INNER)) * MIX_MULTIPLIER_SECOND;
    return value ^ (value >> MIX_SHIFT_OUTER);
}
/* One element of a row's input (see the enumeration): every value is exact in fp16, and so in bf16 and fp32. */
__host__ __device__ inline float input_value(uint32_t row, uint32_t column, uint32_t step) {
    uint32_t row_seed = INPUT_SEED + step * STEP_STRIDE + row * HIDDEN_SIZE;
    uint32_t block_kind = mix_bits(~row_seed + column / ACTIVATION_BLOCK) % INPUT_BLOCK_KINDS;
    uint32_t shift = block_kind < INPUT_BLOCK_KINDS - 1u ? block_kind : (uint32_t)INPUT_TINY_SHIFT;
    float steps = (float)(mix_bits(row_seed + column) & INPUT_VALUE_STEPS);
    return (steps - INPUT_VALUE_OFFSET) / INPUT_VALUE_DIVISOR / (float)(1u << shift);
}
/* A quantized vector's bytes. Lane position t of the column that carries the vector needs, per instruction, inputs
 * 2t, 2t + 1 (register 0 of operand B) and 2t + 8, 2t + 9 (register 1): one word, bytes 0 and 2 register 0, 1 and 3
 * register 1. activation_offset is a byte's place in its block of 32; words are [load of 64 values][t][instruction]. */
__host__ __device__ inline uint32_t activation_offset(uint32_t step_in_block, uint32_t lane_position, uint32_t byte) {
    return step_in_block * MMA_INPUTS + lane_position * HALVES_PER_REGISTER + byte / INPUT_REGISTERS
           + (byte % INPUT_REGISTERS) * MMA_HALF_INPUTS;
}
__host__ __device__ inline uint32_t activation_word_index(uint32_t block, uint32_t lane_position, uint32_t step) {
    return ((block / PAIRS_PER_LOAD) * LANES_PER_LANE_GROUP + lane_position) * MMA_STEPS_PER_LOAD
           + (block % PAIRS_PER_LOAD) * GROUPS_PER_PAIR + step;
}
/* One block per row of the launch: slot blockIdx.x of the batch holds row first_row + blockIdx.x. */
__global__ void fill_input_kernel(float *layer_input, uint32_t first_row, uint32_t step) {
    for (uint32_t column = threadIdx.x; column < HIDDEN_SIZE; column += blockDim.x) {
        layer_input[(size_t)blockIdx.x * HIDDEN_SIZE + column] = input_value(first_row + blockIdx.x, column, step);
    }
}
/* ---- the kernels under test ---- */
/* An fp32 rounded to bf16, its upper half, to nearest with ties to even: the value the cast to bf16 keeps. */
__device__ __forceinline__ float bf16_value(float value) {
    uint32_t bits = __float_as_uint(value);
    return __uint_as_float((bits + BF16_ROUNDING_BIAS + ((bits >> BITS_PER_HALF) & 1u)) & BF16_KEPT_MASK);
}
/* The E4M3 code nearest to value * inverse_scale (clamped to +-448, ties to the even code) and, exact, its value. */
__device__ __forceinline__ uint32_t quantize_to_e4m3(float value, float inverse_scale, float *quantized) {
    float scaled = value * inverse_scale;                         /* by a power of two: exact */
    float magnitude = fminf(fabsf(scaled), E4M3_MAXIMUM);
    uint32_t exponent_field = __float_as_uint(fmaxf(magnitude, E4M3_SMALLEST_NORMAL)) >> FP32_EXPONENT_SHIFT;
    float steps_per_unit = __uint_as_float((E4M3_STEPS_EXPONENT_BASE - exponent_field) << FP32_EXPONENT_SHIFT);
    float step = __uint_as_float((exponent_field - E4M3_STEP_FIELD_OFFSET) << FP32_EXPONENT_SHIFT);
    uint32_t steps = (uint32_t)__float2int_rn(magnitude * steps_per_unit);      /* 0 .. 16, ties to even */
    uint32_t sign = __float_as_uint(scaled) & FP32_SIGN_MASK;
    *quantized = __uint_as_float(__float_as_uint((float)steps * step) | sign);
    return ((exponent_field << E4M3_EXPONENT_SHIFT) + steps - E4M3_CODE_BASE) | (sign >> E4M3_SIGN_FROM_FP32_SHIFT);
}
/* The reference's act_quant for blocks of 32: grid (vectors), one thread per block. The largest magnitude (at least
 * 1e-4) times 1 / 448 in fp32, rounded up to a power of two as the reference reads it from the fp32 fields, is the
 * scale; break_quantization (0 or 1) takes the one below, which clamps the block's largest values (under the one
 * above no group sum of an input differs: the codes move, their values do not). Q2_K also gets its groups' sums, X. */
template <int kquant>
__global__ void quantize_kernel(struct quantized_vectors vectors, uint32_t break_quantization) {
    uint32_t block_count = vectors.size / ACTIVATION_BLOCK;
    uint32_t *vector_words = vectors.words + (size_t)blockIdx.x * (vectors.size / BYTES_PER_WORD);
    for (uint32_t block = threadIdx.x; block < block_count; block += blockDim.x) {
        const float *block_values = vectors.values + (size_t)blockIdx.x * vectors.size + block * ACTIVATION_BLOCK;
        size_t vector_block = (size_t)blockIdx.x * block_count + block;
        float largest = 0.0f;
        for (uint32_t offset = 0; offset < ACTIVATION_BLOCK; offset++) {
            largest = fmaxf(largest, fabsf(block_values[offset]));
        }
        uint32_t ratio_bits = __float_as_uint(fmaxf(largest, AMAX_FLOOR) * E4M3_MAXIMUM_INVERSE);
        uint32_t scale_exponent = (ratio_bits >> FP32_EXPONENT_SHIFT) - break_quantization
                                  + ((ratio_bits & FP32_MANTISSA_MASK) != 0u ? 1u : 0u);
        float inverse_scale = __uint_as_float((FP32_EXPONENT_BIAS + E8M0_BIAS - scale_exponent) << FP32_EXPONENT_SHIFT);
        for (uint32_t step_in_block = 0; step_in_block < GROUPS_PER_PAIR; step_in_block++) {
            float group_sum = 0.0f;
            for (uint32_t lane_position = 0; lane_position < LANES_PER_LANE_GROUP; lane_position++) {
                uint32_t word = 0;
                for (uint32_t byte = 0; byte < BYTES_PER_WORD; byte++) {
                    float quantized = 0.0f;
                    word |= quantize_to_e4m3(block_values[activation_offset(step_in_block, lane_position, byte)],
                                             inverse_scale, &quantized) << (byte * BITS_PER_BYTE);
                    group_sum += quantized;
                }
                vector_words[activation_word_index(block, lane_position, step_in_block)] = word;
            }
            if (kquant == KQUANT_Q2_K) {
                vectors.group_sums[vector_block * GROUPS_PER_PAIR + step_in_block] =
                    group_sum * __uint_as_float(scale_exponent << FP32_EXPONENT_SHIFT);
            }
        }
        vectors.exponents[vector_block] = (uint8_t)scale_exponent;
    }
}
/* The instruction: sums, next_sums of the lane's outputs g, g + 8; weights, next_weights of inputs 0 .. 7, 8 .. 15. */
__device__ __forceinline__ void tensor_core_m16n8k16(float *sums, const uint32_t *weights, uint32_t input_low,
                                                     uint32_t input_high) {
    float *next_sums = sums + LANE_COLUMNS;
    const uint32_t *next_weights = weights + LANE_OUTPUTS;
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(sums[0]), "=f"(sums[1]), "=f"(next_sums[0]), "=f"(next_sums[1])
        : "r"(weights[0]), "r"(weights[1]), "r"(next_weights[0]), "r"(next_weights[1]), "r"(input_low), "r"(input_high),
          "f"(sums[0]), "f"(sums[1]), "f"(next_sums[0]), "f"(next_sums[1]));
}
/* One instruction of a lane: instruction `instruction` (a constant) of the half block in loads, a group of 16 inputs.
 * Lane L (g = L / 4, t = L % 4) holds of operand A outputs g and g + 8 at inputs 2t, 2t + 1, 2t + 8, 2t + 9 (register
 * r: output g + 8 (r % 2), inputs 2t + 8 (r / 2) and the next); of operand B column g, one row's vector, at the same
 * inputs; of the result outputs g and g + 8 for columns 2t and 2t + 1. A weight is its code, 0 .. 3 (Q2_K) or -4 .. 3
 * (Q3_K: the stored code plus its sign flag is the magnitude), a whole number in the mantissa of an fp16 of exponent
 * field 0. The sums, exact, go times their output's scale into pair_sums; Q2_K's minimum term into minimum_sums. */
template <int kquant>
__device__ __forceinline__ void lane_instruction(const struct lane_loads *loads, uint32_t instruction, float *pair_sums,
                                                 float *minimum_sums) {
    float sums[LANE_SUMS] = {};
    uint32_t weights[WEIGHT_REGISTERS], pair = instruction / GROUPS_PER_PAIR, step = instruction % GROUPS_PER_PAIR;
    /* register r's two codes at bits 2r and 2r + 16, its sign flags at bits r and r + 16, a scale byte per output */
    uint32_t codes = LOAD_PART(loads->codes, pair) >> (step * CODE_BITS_PER_MMA_STEP);
    uint32_t scales = LOAD_PART(loads->scales, pair) >> (step * LANE_OUTPUTS * BITS_PER_BYTE);
    uint32_t signs = (instruction / MMA_STEPS_PER_SIGN_WORD == 0 ? loads->signs.x : loads->signs.y)
                     >> (instruction % MMA_STEPS_PER_SIGN_WORD * SIGN_BITS_PER_MMA_STEP);
    /* four E4M3 bytes become the two fp16 pairs of operand B, each the value times 2^-8 */
    uint32_t word = LOAD_PART(loads->inputs, instruction % MMA_STEPS_PER_LOAD), odd_bytes = word >> BITS_PER_BYTE;
    uint32_t input_low = ((word & E4M3_PAIR_SIGN_MASK) << BITS_PER_BYTE)
                         | ((word & E4M3_PAIR_MAGNITUDE_MASK) << E4M3_MAGNITUDE_SHIFT);
    uint32_t input_high = ((odd_bytes & E4M3_PAIR_SIGN_MASK) << BITS_PER_BYTE)
                          | ((odd_bytes & E4M3_PAIR_MAGNITUDE_MASK) << E4M3_MAGNITUDE_SHIFT);
    for (uint32_t weight_register = 0; weight_register < WEIGHT_REGISTERS; weight_register++) {
        uint32_t negative = kquant == KQUANT_Q3_K ? (signs >> weight_register) & SIGN_PAIR_MASK : 0u;
        weights[weight_register] = (((codes >> (weight_register * CODE_BITS)) & CODE_PAIR_MASK) + negative)
                                   | (negative << FP16_SIGN_SHIFT);
    }
    tensor_core_m16n8k16(sums, weights, input_low, input_high);
    for (uint32_t element = 0; element < LANE_SUMS; element++) {
        uint32_t scale_byte = (scales >> (element / LANE_COLUMNS * BITS_PER_BYTE)) & BYTE_MASK;
        int scale = kquant == KQUANT_Q2_K ? (int)(scale_byte & NIBBLE_MASK) : (int)scale_byte - Q3_K_SCALE_BIAS;
        pair_sums[element] = fmaf(sums[element], (float)scale, pair_sums[element]);
        if (kquant == KQUANT_Q2_K) {
            float group_sum = LOAD_PART(loads->sums[element % LANE_COLUMNS], instruction % MMA_STEPS_PER_LOAD);
            minimum_sums[element] = fmaf(group_sum, (float)(scale_byte >> NIBBLE_BITS), minimum_sums[element]);
        }
    }
}
/* One pair of groups of a lane: 32 inputs under one activation scale of each column. The first pair of a load brings
 * the column the lane feeds and, for Q2_K, its result columns' group sums. The pair's sums go times 2^(E + 32) (one
 * fp32 exponent field, normal for the probe's vectors) into scale_sums. `pair` is a constant where this is called. */
template <int kquant>
__device__ __forceinline__ void lane_pair(struct lane_loads *loads, uint32_t pair, float *scale_sums,
                                          float *minimum_sums) {
    float pair_sums[LANE_SUMS] = {};
    if (pair % PAIRS_PER_LOAD == 0) {
        loads->inputs = loads->inputs_at[pair / PAIRS_PER_LOAD * LANES_PER_LANE_GROUP];
        for (uint32_t column = 0; column < LANE_COLUMNS && kquant == KQUANT_Q2_K; column++) {
            loads->sums[column] = loads->sums_at[column][pair / PAIRS_PER_LOAD];
        }
    }
    lane_instruction<kquant>(loads, pair * GROUPS_PER_PAIR, pair_sums, minimum_sums);
    lane_instruction<kquant>(loads, pair * GROUPS_PER_PAIR + 1, pair_sums, minimum_sums);
    for (uint32_t element = 0; element < LANE_SUMS; element++) {
        uint32_t code = (loads->exponents[element % LANE_COLUMNS] >> (pair * BITS_PER_BYTE)) & BYTE_MASK;
        float power = __uint_as_float((code + ACTIVATION_POWER_FIELD_OFFSET) << FP32_EXPONENT_SHIFT);
        scale_sums[element] = fmaf(pair_sums[element], power, scale_sums[element]);
    }
}
/* One warp, one tile of 16 outputs of one job, block_count blocks of the input width from first_block; tile_records
 * is the tile's record of block 0. Per half block a lane reads 16 bytes of codes (8 of signs), 16 of group scales, 32
 * of its column's vector, 8 of its result columns' E8M0 codes (Q2_K: 64 of their group sums); per block d and dmin. */
template <int kquant>
__device__ __forceinline__ void tile_slice_product(const uint8_t *tile_records, struct quantized_vectors vectors,
                                                   uint32_t input_size, uint32_t pairs_per_vector,
                                                   const struct expert_job *job, uint32_t first_block,
                                                   uint32_t block_count, float *partial_sums) {
    float totals[LANE_SUMS] = {};
    uint32_t lane = threadIdx.x % LANES_PER_WARP;
    uint32_t lane_group = lane / LANES_PER_LANE_GROUP, lane_position = lane % LANES_PER_LANE_GROUP;
    size_t vector_halves = input_size / HALF_WEIGHTS;     /* of one vector: per half block two loads, four E8M0 codes */
    size_t fed_loads = job->pair_index[lane_group] / pairs_per_vector * vector_halves * LOADS_PER_HALF;
    const uint4 *lane_inputs = (const uint4 *)vectors.words + fed_loads * LANES_PER_LANE_GROUP + lane_position;
    const uint32_t *column_exponents[LANE_COLUMNS];
    const float4 *column_sums[LANE_COLUMNS];
    for (uint32_t column = 0; column < LANE_COLUMNS; column++) {
        size_t vector = job->pair_index[lane_position * LANE_COLUMNS + column] / pairs_per_vector;
        column_exponents[column] = (const uint32_t *)vectors.exponents + vector * vector_halves;
        column_sums[column] = (const float4 *)vectors.group_sums + vector * vector_halves * LOADS_PER_HALF;
    }
    for (uint32_t block = first_block; block < first_block + block_count; block++) {
        const uint8_t *record = tile_records + block * RECORD_BYTES(kquant);
        const uint16_t *block_scales = (const uint16_t *)(record + BLOCK_SCALES_AT(kquant));
        float scale_sums[LANE_SUMS] = {}, minimum_sums[LANE_SUMS] = {};
        for (uint32_t half = 0; half < HALVES_PER_BLOCK; half++) {
            struct lane_loads loads = {};
            size_t vector_half = (size_t)block * HALVES_PER_BLOCK + half;
            loads.codes = ((const uint4 *)record)[half * LANES_PER_WARP + lane];
            loads.scales = ((const uint4 *)(record + GROUP_SCALES_AT(kquant)))[half * MMA_HALF_ROWS + lane_group];
            if (kquant == KQUANT_Q3_K) {
                loads.signs = ((const uint2 *)(record + CODE_BYTES_PER_RECORD))[half * LANES_PER_WARP + lane];
            }
            for (uint32_t column = 0; column < LANE_COLUMNS; column++) {
                loads.exponents[column] = column_exponents[column][vector_half];
                loads.sums_at[column] = column_sums[column] + vector_half * LOADS_PER_HALF;
            }
            loads.inputs_at = lane_inputs + vector_half * LOADS_PER_HALF * LANES_PER_LANE_GROUP;
            lane_pair<kquant>(&loads, 0, scale_sums, minimum_sums);
            lane_pair<kquant>(&loads, 1, scale_sums, minimum_sums);
            lane_pair<kquant>(&loads, PAIRS_PER_LOAD, scale_sums, minimum_sums);
            lane_pair<kquant>(&loads, PAIRS_PER_LOAD + 1, scale_sums, minimum_sums);
        }
        for (uint32_t element = 0; element < LANE_SUMS; element++) {
            uint32_t first_scale = (lane_group + element / LANE_COLUMNS * MMA_HALF_ROWS) * BLOCK_SCALES(kquant);
            float d = __half2float(__ushort_as_half(block_scales[first_scale + BLOCK_SCALE_D]));
            totals[element] = fmaf(scale_sums[element], d, totals[element]);
            if (kquant == KQUANT_Q2_K) {
                float dmin = __half2float(__ushort_as_half(block_scales[first_scale + BLOCK_SCALE_DMIN]));
                totals[element] = fmaf(minimum_sums[element], -dmin, totals[element]);
            }
        }
    }
    for (uint32_t element = 0; element < LANE_SUMS; element++) {
        partial_sums[threadIdx.x * LANE_SUMS + element] = totals[element];
    }
}
/* One element of one matrix: the partial sums of its slices (warps first_warp ..), added in slice order. Under
 * --break-exactness the sum of a job of several rows starts at another slice, which depends on their number. */
__device__ __forceinline__ float sum_slices_in_order(const float *partial_sums, uint32_t first_warp, uint32_t slices,
                                                     uint32_t lane, uint32_t element, uint32_t rows, uint32_t broken) {
    uint32_t first_slice = broken != 0u && rows > 1u ? 1u + rows % (slices - 1u) : 0u;
    float sum = 0.0f;
    for (uint32_t position = 0; position < slices; position++) {
        uint32_t warp = first_warp + (first_slice + position) % slices;
        sum += partial_sums[(warp * LANES_PER_WARP + lane) * LANE_SUMS + element];
    }
    return sum;
}
/* One product stage of every job: grid (tiles of 16 outputs, jobs); per block as many warps for each matrix of the
 * stage as its input width has slices; the element's own lane of warp 0 adds the slices. Two matrices: gate (the first
 * warps) and up from the rows' quantized inputs, each rounded to bf16 as the reference's product returns it;
 * h = silu(min(gate, L)) * clamp(up, -L, L) * route weight, cast to bf16, goes to hidden_values, [pair][inner size].
 * One matrix: down from the pairs' quantized hidden vectors, as fp32 to pair_output, [pair][hidden size]. */
template <int kquant, int matrix_count>
__global__ void expert_kernel(const uint8_t *stored, struct quantized_vectors vectors, const struct expert_job *jobs,
                              float *hidden_values, float *pair_output, uint32_t break_exactness) {
    __shared__ float partial_sums[matrix_count * STAGE_SLICES(matrix_count) * LANES_PER_WARP * LANE_SUMS];
    const struct expert_job *job = &jobs[blockIdx.y];
    const uint32_t slices = STAGE_SLICES(matrix_count), blocks = STAGE_INPUT_SIZE(matrix_count) / BLOCK_WEIGHTS;
    uint32_t warp = threadIdx.x / LANES_PER_WARP, lane = threadIdx.x % LANES_PER_WARP;
    size_t matrix = (size_t)job->expert_index * MATRIX_KINDS
                    + (matrix_count == DOWN_MATRICES ? (uint32_t)DOWN_MATRIX : warp / slices);
    size_t tile_block = matrix * TILE_BLOCKS_PER_MATRIX + (size_t)blockIdx.x * blocks;
    tile_slice_product<kquant>(stored + tile_block * RECORD_BYTES(kquant), vectors, STAGE_INPUT_SIZE(matrix_count),
                               STAGE_VECTOR_PAIRS(matrix_count), job, (warp % slices) * (blocks / slices),
                               blocks / slices, partial_sums);
    __syncthreads();
    for (uint32_t element = 0; element < LANE_SUMS && warp == 0; element++) {
        uint32_t column = (lane % LANES_PER_LANE_GROUP) * LANE_COLUMNS + element % LANE_COLUMNS;
        uint32_t output = blockIdx.x * MMA_ROWS + lane / LANES_PER_LANE_GROUP + element / LANE_COLUMNS * MMA_HALF_ROWS;
        float value = sum_slices_in_order(partial_sums, 0, slices, lane, element, job->row_count, break_exactness);
        if (column < job->row_count && matrix_count == DOWN_MATRICES) {
            pair_output[(size_t)job->pair_index[column] * HIDDEN_SIZE + output] = value;
        } else if (column < job->row_count) {
            float gate_value = fminf(bf16_value(value), SWIGLU_LIMIT);
            float up_value = bf16_value(sum_slices_in_order(partial_sums, UP_MATRIX * slices, slices, lane, element,
                                                            job->row_count, break_exactness));
            float silu = gate_value / (1.0f + expf(-gate_value));
            up_value = fminf(fmaxf(up_value, -SWIGLU_LIMIT), SWIGLU_LIMIT);
            hidden_values[(size_t)job->pair_index[column] * EXPERT_INNER_SIZE + output] =
                bf16_value(silu * up_value * job->route_weight[column]);
        }
    }
}
/* A row's six experts, each rounded to bf16 as the reference's product returns it, added in slot order: grid (rows). */
__global__ void row_sum_kernel(const float *pair_output, float *row_output) {
    for (uint32_t column = threadIdx.x; column < HIDDEN_SIZE; column += blockDim.x) {
        float sum = 0.0f;
        for (uint32_t slot = 0; slot < EXPERTS_PER_ROW; slot++) {
            sum += bf16_value(pair_output[((size_t)blockIdx.x * EXPERTS_PER_ROW + slot) * HIDDEN_SIZE + column]);
        }
        row_output[(size_t)blockIdx.x * HIDDEN_SIZE + column] = sum;
    }
}
/* ---- host side: the file's blocks, their lane order, and the references ---- */
static uint8_t canonical[MATRIX_KINDS][MOST_CANONICAL_MATRIX_BYTES];    /* the matrices being packed or checked */
struct weight_fields { int code, scale, minimum, scale_byte; };     /* scale_byte: what lane order keeps per group */
/* One matrix's canonical blocks, [output][block]: every byte a hash of its place (any is valid), then d (and dmin). */
static void make_canonical_matrix(int kquant, uint32_t expert, uint32_t matrix_kind, uint8_t *blocks) {
    uint32_t matrix = (expert * MATRIX_KINDS + matrix_kind) * KQUANT_TYPES + (uint32_t)kquant;
    uint32_t seed = mix_bits(CANONICAL_SEED + matrix), scales = BLOCK_SCALES(kquant);
    size_t block_bytes = BLOCK_BYTES(kquant);
    for (size_t word = 0; word < CANONICAL_BLOCKS_PER_MATRIX * block_bytes / BYTES_PER_WORD; word++) {
        uint32_t bits = mix_bits(seed + (uint32_t)word);
        memcpy(blocks + word * BYTES_PER_WORD, &bits, BYTES_PER_WORD);
    }
    for (uint32_t scale = 0; scale < CANONICAL_BLOCKS_PER_MATRIX * scales; scale++) {
        uint16_t bits = (uint16_t)(D_BITS_SMALLEST + mix_bits(~seed + scale) % D_BITS_SPAN);
        memcpy(blocks + scale / scales * block_bytes + D_AT(kquant) + scale % scales * FP16_BYTES, &bits, FP16_BYTES);
    }
}
/* An fp16 from its two bytes, by its definition (an exponent field of 31 is not a scale and does not occur here). */
static double half_value(const uint8_t *bytes) {
    uint32_t bits = bytes[0] | ((uint32_t)bytes[1] << BITS_PER_BYTE);
    uint32_t exponent = (bits >> FP16_EXPONENT_SHIFT) & FP16_EXPONENT_MASK, mantissa = bits & FP16_MANTISSA_MASK;
    double magnitude = exponent == 0 ? ldexp((double)mantissa, -FP16_SUBNORMAL_EXPONENT)
                       : ldexp((double)(mantissa + FP16_IMPLICIT_ONE), (int)exponent - FP16_NORMAL_EXPONENT);
    return (bits >> FP16_SIGN_SHIFT) != 0u ? -magnitude : magnitude;
}
/* Weight w of group k of a canonical block, by the format's definition and by nothing of the lane order: the one
 * reader of the file's bytes, for the packer and for the reference. Its two low code bits are at bits 2 (k % 8 / 2)
 * of code byte 32 (k / 8) + 16 (k % 2) + w. Q2_K: byte k holds the scale (low half) and the minimum. Q3_K: the code's
 * high bit is bit 4 (k / 8) + k % 8 / 2 of high-bit byte 16 (k % 2) + w, and 4 is subtracted where it is clear; scale
 * k has its low four bits in scale byte k % 4 + 4 (k / 4 % 2), the low half for k < 8 and the high half above, its two
 * high bits at bits 2 (k / 4) of scale byte 8 + k % 4, and 32 is subtracted; there is no minimum. */
static struct weight_fields canonical_fields(int kquant, const uint8_t *block, uint32_t group, uint32_t weight) {
    uint32_t half = group / GROUPS_PER_HALF, side = group % GROUPS_PER_PAIR, quarter = group / BYTES_PER_WORD;
    uint32_t pair = group % GROUPS_PER_HALF / GROUPS_PER_PAIR, byte = block[group];
    uint32_t code_byte = block[CODES_AT(kquant) + half * HALF_CODE_BYTES + side * GROUP_WEIGHTS + weight];
    struct weight_fields fields = {(int)((code_byte >> (pair * CODE_BITS)) & CODE_MASK), (int)(byte & NIBBLE_MASK),
                                   (int)(byte >> NIBBLE_BITS), (int)byte};
    if (kquant == KQUANT_Q3_K) {
        const uint8_t *scales = block + Q3_K_SCALES_AT;
        uint32_t high_bit = (block[side * GROUP_WEIGHTS + weight] >> (half * PAIRS_PER_HALF + pair)) & 1u;
        uint32_t low_byte = scales[group % BYTES_PER_WORD + quarter % Q3_K_LOW_SCALE_WORDS * BYTES_PER_WORD];
        uint32_t low = group < GROUPS_PER_HALF ? low_byte & NIBBLE_MASK : low_byte >> NIBBLE_BITS;
        uint32_t high = scales[Q3_K_LOW_SCALE_WORDS * BYTES_PER_WORD + group % BYTES_PER_WORD] >> (quarter * CODE_BITS);
        fields.code -= (int)(1u - high_bit) * Q3_K_CODE_OFFSET;
        fields.scale_byte = (int)(low | ((high & CODE_MASK) << NIBBLE_BITS));
        fields.scale = fields.scale_byte - Q3_K_SCALE_BIAS;
        fields.minimum = 0;
    }
    return fields;
}
/* One matrix from its canonical blocks into lane order: records, zeroed, gets its [tile][block] records. Weight w of
 * output o is in lane 4 (o % 8) + w % 8 / 2, register o % 16 / 8 + 2 (w / 8), half w % 2: in the code word of its pair
 * of groups at bit 2 * register + 8 (group % 2) + 16 * half, in the sign word of its four groups at bit register +
 * 4 (group % 4) + 16 * half. The engine's converter has to refuse a d or dmin that is not finite; here none is. */
static void pack_matrix(int kquant, const uint8_t *blocks, uint32_t outputs, uint32_t inputs, uint8_t *records) {
    uint32_t blocks_per_tile = inputs / BLOCK_WEIGHTS;
    for (size_t row_block = 0; row_block < (size_t)outputs * blocks_per_tile; row_block++) {
        const uint8_t *source = blocks + row_block * BLOCK_BYTES(kquant);
        uint32_t output = (uint32_t)(row_block / blocks_per_tile), output_half = output % MMA_ROWS / MMA_HALF_ROWS;
        size_t tile_block = (size_t)(output / MMA_ROWS) * blocks_per_tile + row_block % blocks_per_tile;
        uint8_t *record = records + tile_block * RECORD_BYTES(kquant), *group_scales = record + GROUP_SCALES_AT(kquant);
        uint32_t *code_words = (uint32_t *)record, *sign_words = (uint32_t *)(record + CODE_BYTES_PER_RECORD);
        memcpy(record + BLOCK_SCALES_AT(kquant) + output % MMA_ROWS * BLOCK_SCALES(kquant) * FP16_BYTES,
               source + D_AT(kquant), (size_t)BLOCK_SCALES(kquant) * FP16_BYTES);
        for (uint32_t index = 0; index < BLOCK_WEIGHTS; index++) {
            uint32_t group = index / GROUP_WEIGHTS, weight = index % GROUP_WEIGHTS, step = group % GROUPS_PER_HALF;
            uint32_t half_group = group / GROUPS_PER_HALF * MMA_HALF_ROWS + output % MMA_HALF_ROWS;
            uint32_t lane_half = half_group * LANES_PER_LANE_GROUP + weight % MMA_HALF_INPUTS / HALVES_PER_REGISTER;
            uint32_t weight_register = output_half + weight / MMA_HALF_INPUTS * LANE_OUTPUTS;
            uint32_t half_bit = weight % HALVES_PER_REGISTER * BITS_PER_HALF;
            struct weight_fields fields = canonical_fields(kquant, source, group, weight);
            uint32_t stored_code = (uint32_t)(fields.code < 0 ? -fields.code - 1 : fields.code);
            code_words[lane_half * PAIRS_PER_HALF + step / GROUPS_PER_PAIR] |=
                stored_code << ((weight_register + group % GROUPS_PER_PAIR * WEIGHT_REGISTERS) * CODE_BITS + half_bit);
            if (kquant == KQUANT_Q3_K && fields.code < 0) {
                sign_words[lane_half * SIGN_WORDS_PER_HALF + step / MMA_STEPS_PER_SIGN_WORD] |=
                    1u << (weight_register + step % MMA_STEPS_PER_SIGN_WORD * SIGN_BITS_PER_MMA_STEP + half_bit);
            }
            group_scales[half_group * SCALE_BYTES_PER_LANE_GROUP + step * LANE_OUTPUTS + output_half] =
                (uint8_t)fields.scale_byte;
        }
    }
}
/* E4M3 from its fields: (8 + mantissa) * 2^(exponent - 10), or mantissa * 2^-9 where the exponent is 0. */
static double e4m3_value(uint32_t code) {
    uint32_t exponent = (code >> E4M3_EXPONENT_SHIFT) & E4M3_EXPONENT_MASK, mantissa = code & E4M3_MANTISSA_MASK;
    uint32_t scaled = exponent == 0 ? mantissa << 1 : (E4M3_IMPLICIT_ONE + mantissa) << exponent;
    return ((code & E4M3_SIGN_BIT) != 0u ? -1.0 : 1.0) * (double)scaled / E4M3_VALUE_DIVISOR;
}
/* A double rounded to bf16's eight significant bits, to nearest with ties to even (rint in the default mode). */
static double host_round_to_bf16(double value) {
    int binary_exponent = 0;
    double mantissa = frexp(value, &binary_exponent);     /* value = mantissa * 2^exponent, 0.5 <= |mantissa| < 1 */
    return ldexp(rint(ldexp(mantissa, BF16_SIGNIFICANT_BITS)), binary_exponent - BF16_SIGNIFICANT_BITS);
}
/* The reference's act_quant for one vector, from its definition and not from the card's handling of bits: per block
 * of 32 the scale is 2^ceil(log2(largest magnitude, at least 1e-4, times 1 / 448 in fp32)); a value over the scale,
 * clamped to +-448, becomes the nearest E4M3 value, ties to the even code. quantized: codes, then E8M0 codes. */
static void host_quantize(const float *values, uint32_t vector_size, uint8_t *quantized, float *group_sums) {
    for (uint32_t block = 0; block < vector_size / ACTIVATION_BLOCK; block++) {
        const float *block_values = values + block * ACTIVATION_BLOCK;
        float largest = AMAX_FLOOR;
        for (uint32_t offset = 0; offset < ACTIVATION_BLOCK; offset++) {
            largest = fmaxf(largest, fabsf(block_values[offset]));
        }
        float ratio = largest * E4M3_MAXIMUM_INVERSE;
        int log2_ceiling = (int)ceil(log2((double)ratio));    /* exact: a float's log2 is not within 1e-8 of a whole */
        double group_sum = 0.0;
        quantized[vector_size + block] = (uint8_t)(log2_ceiling + E8M0_BIAS);
        for (uint32_t offset = 0; offset < ACTIVATION_BLOCK; offset++) {
            double scaled = ldexp((double)block_values[offset], -log2_ceiling);
            double magnitude = fmin(fabs(scaled), (double)E4M3_MAXIMUM);
            uint32_t nearest = 0;
            for (uint32_t code = 1; code < E4M3_NAN_MAGNITUDE; code++) {
                double distance = fabs(e4m3_value(code) - magnitude), least = fabs(e4m3_value(nearest) - magnitude);
                nearest = distance < least || (distance == least && (code & 1u) == 0u) ? code : nearest;
            }
            nearest |= (signbit(scaled) ? 1u : 0u) * E4M3_SIGN_BIT;
            quantized[block * ACTIVATION_BLOCK + offset] = (uint8_t)nearest;
            group_sum = (offset % GROUP_WEIGHTS == 0 ? 0.0 : group_sum) + e4m3_value(nearest);
            group_sums[(block * ACTIVATION_BLOCK + offset) / GROUP_WEIGHTS] = (float)ldexp(group_sum, log2_ceiling);
        }
    }
}
/* One output of one canonical matrix in double, by the format's definition: its row times the quantized vector. */
static double host_row_product(int kquant, uint32_t matrix_kind, uint32_t output, const uint8_t *quantized) {
    uint32_t input_size = matrix_kind == DOWN_MATRIX ? EXPERT_INNER_SIZE : HIDDEN_SIZE;
    const uint8_t *row = canonical[matrix_kind] + (size_t)output * (input_size / BLOCK_WEIGHTS) * BLOCK_BYTES(kquant);
    double sum = 0.0;
    for (uint32_t first_input = 0; first_input < input_size; first_input += ACTIVATION_BLOCK) {
        const uint8_t *block = row + (size_t)(first_input / BLOCK_WEIGHTS) * BLOCK_BYTES(kquant);
        double d = half_value(block + D_AT(kquant)), block_sum = 0.0;
        double dmin = kquant == KQUANT_Q2_K ? half_value(block + D_AT(kquant) + FP16_BYTES) : 0.0;
        for (uint32_t offset = 0; offset < ACTIVATION_BLOCK; offset++) {
            uint32_t index = first_input % BLOCK_WEIGHTS + offset;
            struct weight_fields fields = canonical_fields(kquant, block, index / GROUP_WEIGHTS, index % GROUP_WEIGHTS);
            block_sum += (d * fields.scale * fields.code - dmin * fields.minimum)
                          * e4m3_value(quantized[first_input + offset]);
        }
        sum += ldexp(block_sum, (int)quantized[input_size + first_input / ACTIVATION_BLOCK] - E8M0_BIAS);
    }
    return sum;
}
static uint64_t count_differing(const void *first, const void *second, size_t element_count, size_t element_bytes) {
    uint64_t differing = 0;
    for (size_t at = 0; at < element_count * element_bytes; at += element_bytes) {
        differing += memcmp((const char *)first + at, (const char *)second + at, element_bytes) != 0 ? 1u : 0u;
    }
    return differing;
}
/* ---- host side: options, routes and jobs ---- */
static struct {
    uint32_t step_count, corrupt_weights, break_exactness, break_quantization, break_route;
    uint32_t first_overlap, last_overlap;        /* the overlaps and the types of this run: all, or the one named */
    int first_type, last_type;
} options = {DEFAULT_STEP_COUNT, 0, 0, 0, 0, 0, PERCENT, KQUANT_Q2_K, KQUANT_Q3_K};
static uint8_t *packed[KQUANT_TYPES];            /* a type's experts in lane order: packed once, sent to each card */
/* The synthetic routing. Row r takes six consecutive experts from the 6r-th (overlap 0), the 3r-th (50) or the first
 * (100) of its step's span, which is what span_rows rows take. Spans are in one half of the experts for even steps
 * and in the other for odd ones, and each starts where the one before it in its half ended, wrapped to the places a
 * span has there: between two reads of an expert a card reads 24 others or more, 279 MB of Q2_K, twice a PRO 6000's
 * cache (six further each time, the other probe's way, left 104 MB at 2 rows and overlap 50). The checked step is a
 * half's first: there every span starts at the half's first expert, and a row launched alone keeps its experts. */
static uint32_t route_expert(uint32_t step, uint32_t overlap_percent, uint32_t span_rows, uint32_t row, uint32_t slot) {
    uint32_t half = EXPERT_COUNT / EXPERT_HALVES, row_stride = EXPERTS_PER_ROW * (PERCENT - overlap_percent) / PERCENT;
    uint32_t span = EXPERTS_PER_ROW + (span_rows - 1u) * row_stride;
    uint32_t first = (step % EXPERT_HALVES) * half + (step / EXPERT_HALVES) * span % (half - span + 1u);
    return first + row * row_stride + slot + (uint32_t)(options.break_route == BREAK_EXPERT) * EXPERT_COUNT;
}
static float route_weight(uint32_t step, uint32_t row, uint32_t slot) {
    uint32_t code = mix_bits(~(uint32_t)ROUTE_SEED + step * STEP_STRIDE + row * EXPERTS_PER_ROW + slot);
    return (float)(1u + code % ROUTE_WEIGHT_STEPS) / ROUTE_WEIGHT_DIVISOR;
}
/* The jobs of row_count rows from first_row, which fill the batch from its slot 0: one per expert, columns by row. */
static uint32_t build_jobs(uint32_t step, uint32_t overlap_percent, uint32_t first_row, uint32_t row_count,
                           struct expert_job *jobs) {
    uint32_t job_count = 0, pair_count = row_count * EXPERTS_PER_ROW, broken = options.break_route;
    for (uint32_t pair = 0; pair < pair_count; pair++) {
        uint32_t row = first_row + pair / EXPERTS_PER_ROW, slot = pair % EXPERTS_PER_ROW;
        uint32_t expert = route_expert(step, overlap_percent, first_row + row_count, row, slot), job_index = 0;
        while (job_index < job_count && jobs[job_index].expert_index != expert) {
            job_index++;
        }
        struct expert_job *job = &jobs[job_index];
        if (job_index == job_count) {
            job_count++;
            job->expert_index = expert;
            job->row_count = 0;
            for (uint32_t column = 0; column < MAX_ROWS; column++) {
                job->pair_index[column] = pair;              /* a column not in use reads what column 0 reads */
                job->route_weight[column] = 0.0f;
            }
        }
        job->pair_index[job->row_count] = pair;
        job->route_weight[job->row_count++] = route_weight(step, row, slot);
    }
    jobs[0].pair_index[MAX_ROWS - 1] += broken == BREAK_PAIR ? pair_count : 0u;      /* --break-route: one of what */
    jobs[0].row_count += broken == BREAK_COLUMNS ? row_count : 0u;                   /* a kernel indexes by is put */
    for (; broken == BREAK_JOBS && job_count <= pair_count && job_count < MAX_PAIRS; job_count++) {
        jobs[job_count] = jobs[0];                  /* past its bound, the rest left right (an expert: route_expert) */
    }
    return job_count;
}
/* The jobs of a launch, the one way to them: built, then all that a kernel indexes by them held to what the card holds
 * (n - 1u < most: 1 .. most, as 0 wraps): rows, jobs, a job's expert, its columns, their pairs. 0: launch nothing. */
static uint32_t checked_jobs(uint32_t step, uint32_t overlap_percent, uint32_t first_row, uint32_t row_count,
                             struct expert_job *jobs) {
    uint32_t pair_count = row_count * EXPERTS_PER_ROW;
    uint32_t job_count = row_count - 1u < MAX_ROWS ? build_jobs(step, overlap_percent, first_row, row_count, jobs) : 0u;
    int in_bounds = job_count - 1u < pair_count;
    for (uint32_t job_index = 0; in_bounds && job_index < job_count; job_index++) {
        const struct expert_job *job = &jobs[job_index];
        in_bounds = job->expert_index < EXPERT_COUNT && job->row_count - 1u < row_count;
        for (uint32_t column = 0; column < MAX_ROWS; column++) {
            in_bounds = in_bounds && job->pair_index[column] < pair_count;
        }
    }
    return in_bounds ? job_count : 0u;
}
/* ---- host side: one card ---- */
static struct {
    uint8_t *stored;                             /* the experts of the type being probed */
    struct expert_job *jobs;
    struct quantized_vectors input, hidden;      /* of the rows of the batch, of the pairs */
    float *result;                               /* [pair][hidden size] of an expert for a row, then the rows' sums */
    cudaEvent_t started, stopped;
} card;                                          /* the card being probed: one at a time */
static void allocate_vectors(struct quantized_vectors *vectors, size_t vector_count, uint32_t size) {
    CHECK_CUDA(cudaMalloc((void **)&vectors->values, vector_count * size * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void **)&vectors->words, vector_count * size));
    CHECK_CUDA(cudaMalloc((void **)&vectors->exponents, vector_count * (size / ACTIVATION_BLOCK)));
    CHECK_CUDA(cudaMalloc((void **)&vectors->group_sums, vector_count * (size / GROUP_WEIGHTS) * sizeof(float)));
    vectors->size = size;
}
/* What a launch needs that is not timed: the jobs of these rows, checked and on the card, and the rows' inputs. */
static uint32_t prepare_layer(uint32_t step, uint32_t overlap_percent, uint32_t first_row, uint32_t row_count) {
    struct expert_job jobs[MAX_PAIRS] = {};
    uint32_t job_count = checked_jobs(step, overlap_percent, first_row, row_count, jobs);
    if (job_count == 0) {
        fprintf(stderr, "a job reaches past what the card holds: nothing is launched\n");
        exit(EXIT_FAILURE);
    }
    CHECK_CUDA(cudaMemcpy(card.jobs, jobs, job_count * sizeof(struct expert_job), cudaMemcpyHostToDevice));
    fill_input_kernel<<<row_count, FILL_THREADS_PER_BLOCK>>>(card.input.values, first_row, step);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    return job_count;
}
/* The routed experts of one layer: both quantizations, gate with up, down, and the rows' sums. */
template <int kquant>
static void launch_layer_as(uint32_t job_count, uint32_t row_count) {
    dim3 gate_up_grid(EXPERT_INNER_SIZE / MMA_ROWS, job_count), down_grid(HIDDEN_SIZE / MMA_ROWS, job_count);
    unsigned int gate_up_threads = GATE_UP_MATRICES * GATE_UP_SLICES * LANES_PER_WARP;
    quantize_kernel<kquant><<<row_count, HIDDEN_SIZE / ACTIVATION_BLOCK>>>(card.input, options.break_quantization);
    expert_kernel<kquant, GATE_UP_MATRICES><<<gate_up_grid, gate_up_threads>>>(
        card.stored, card.input, card.jobs, card.hidden.values, NULL, options.break_exactness);
    quantize_kernel<kquant><<<row_count * EXPERTS_PER_ROW, EXPERT_INNER_SIZE / ACTIVATION_BLOCK>>>(
        card.hidden, options.break_quantization);
    expert_kernel<kquant, DOWN_MATRICES><<<down_grid, DOWN_SLICES * LANES_PER_WARP>>>(
        card.stored, card.hidden, card.jobs, NULL, card.result, options.break_exactness);
    row_sum_kernel<<<row_count, FILL_THREADS_PER_BLOCK>>>(card.result, card.result + ROW_SUMS_AT);
}
static void (*const launch_layer[KQUANT_TYPES])(uint32_t job_count, uint32_t row_count) = {
    launch_layer_as<KQUANT_Q2_K>, launch_layer_as<KQUANT_Q3_K>};
/* One launch of the checked step outside the timing, its outputs made stale first. result gets RESULT_FLOATS. */
static void run_checked_layer(int kquant, uint32_t overlap_percent, uint32_t first_row, uint32_t row_count,
                              int stale_byte, float *result) {
    CHECK_CUDA(cudaMemset(card.result, stale_byte, RESULT_FLOATS * sizeof(float)));
    uint32_t job_count = prepare_layer(CHECKED_STEP, overlap_percent, first_row, row_count);
    launch_layer[kquant](job_count, row_count);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(result, card.result, RESULT_FLOATS * sizeof(float), cudaMemcpyDeviceToHost));
}
/* One vector of the launch that just ran against the host's quantization of the same values: how many quantized bytes
 * and exponents differ, how many of Q2_K's group sums. card_quantized: the card's codes by value, then E8M0 codes. */
struct differing_counts { unsigned long long bytes, sums; };
static struct differing_counts check_quantization(int kquant, const struct quantized_vectors *vectors, size_t vector,
                                                  const float *values, uint8_t *card_quantized) {
    static uint8_t host_quantized[QUANTIZED_BYTES(HIDDEN_SIZE)];
    static float host_sums[HIDDEN_SIZE / GROUP_WEIGHTS], card_sums[HIDDEN_SIZE / GROUP_WEIGHTS];
    static uint32_t words[HIDDEN_SIZE / BYTES_PER_WORD];
    uint32_t size = vectors->size, block_count = size / ACTIVATION_BLOCK, group_count = size / GROUP_WEIGHTS;
    host_quantize(values, size, host_quantized, host_sums);
    CHECK_CUDA(cudaMemcpy(words, vectors->words + vector * (size / BYTES_PER_WORD), size, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(card_quantized + size, vectors->exponents + vector * block_count, block_count,
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(card_sums, vectors->group_sums + vector * group_count, group_count * sizeof(float),
                          cudaMemcpyDeviceToHost));
    for (uint32_t word_index = 0; word_index < block_count * WORDS_PER_ACTIVATION_BLOCK; word_index++) {
        uint32_t block = word_index / WORDS_PER_ACTIVATION_BLOCK, step_in_block = word_index % GROUPS_PER_PAIR;
        uint32_t lane_position = word_index % WORDS_PER_ACTIVATION_BLOCK / GROUPS_PER_PAIR;
        uint32_t word = words[activation_word_index(block, lane_position, step_in_block)];
        for (uint32_t byte = 0; byte < BYTES_PER_WORD; byte++) {
            card_quantized[block * ACTIVATION_BLOCK + activation_offset(step_in_block, lane_position, byte)] =
                (uint8_t)(word >> (byte * BITS_PER_BYTE));
        }
    }
    struct differing_counts counts = {
        count_differing(card_quantized, host_quantized, QUANTIZED_BYTES(size), 1),
        count_differing(card_sums, host_sums, kquant == KQUANT_Q2_K ? group_count : 0u, sizeof(float))};
    return counts;
}
/* One case's checks. The launch against the host for two pairs (the first row at its last expert, the last row at
 * its first: column 0 and the last column of a job where rows share experts): the row's quantized input, the hidden
 * vector from it, that vector's quantization, the output from that. Then each row launched alone against the same
 * row in the launch. Returns how many did not come out as required (the header says what each flag requires). */
static int check_case(int kquant, const char *case_name, uint32_t overlap_percent, uint32_t row_count) {
    static float result[RESULT_FLOATS], alone_result[RESULT_FLOATS], values[HIDDEN_SIZE];
    static uint8_t card_quantized[QUANTIZED_BYTES(HIDDEN_SIZE)];
    size_t checked_expert = route_expert(CHECKED_STEP, overlap_percent, row_count, 0, EXPERTS_PER_ROW - 1u);
    int rows_must_differ = options.break_exactness != 0u && overlap_percent > 0u && row_count > 1u, failures = 0;
    int scales_moved = options.break_quantization != 0u, sums_moved = scales_moved && kquant == KQUANT_Q2_K;
    uint64_t differing = 0;
    for (size_t matrix = GATE_MATRIX; matrix <= DOWN_MATRIX && options.corrupt_weights != 0u; matrix += DOWN_MATRIX) {
        size_t first_byte = (checked_expert * MATRIX_KINDS + matrix) * MATRIX_BYTES(kquant);    /* of gate, of down */
        CHECK_CUDA(cudaMemset(card.stored + first_byte, CORRUPTED_BYTE_VALUE, CORRUPTED_BYTES));
    }
    run_checked_layer(kquant, overlap_percent, 0, row_count, STALE_BYTE_SHARED, result);
    for (uint32_t checked = 0; checked < CHECKED_PAIRS; checked++) {
        uint32_t pair = checked == 0 ? EXPERTS_PER_ROW - 1u : (row_count - 1u) * EXPERTS_PER_ROW;
        uint32_t row = pair / EXPERTS_PER_ROW, slot = pair % EXPERTS_PER_ROW;
        uint32_t expert = route_expert(CHECKED_STEP, overlap_percent, row_count, row, slot);
        double hidden_error = 0.0, hidden_square = 0.0, output_error = 0.0, output_square = 0.0;
        int corrupted = options.corrupt_weights != 0u && checked == 0;
        for (uint32_t matrix_kind = 0; matrix_kind < MATRIX_KINDS; matrix_kind++) {
            make_canonical_matrix(kquant, expert, matrix_kind, canonical[matrix_kind]);
        }
        for (uint32_t column = 0; column < HIDDEN_SIZE; column++) {
            values[column] = input_value(row, column, CHECKED_STEP);
        }
        struct differing_counts input_counts = check_quantization(kquant, &card.input, row, values, card_quantized);
        CHECK_CUDA(cudaMemcpy(values, card.hidden.values + (size_t)pair * EXPERT_INNER_SIZE,
                              EXPERT_INNER_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
        for (uint32_t inner = 0; inner < EXPERT_INNER_SIZE; inner++) {
            double gate_value = host_round_to_bf16(host_row_product(kquant, GATE_MATRIX, inner, card_quantized));
            double up_value = host_round_to_bf16(host_row_product(kquant, UP_MATRIX, inner, card_quantized));
            gate_value = fmin(gate_value, (double)SWIGLU_LIMIT);
            up_value = fmin(fmax(up_value, -(double)SWIGLU_LIMIT), (double)SWIGLU_LIMIT);
            double reference = host_round_to_bf16(gate_value / (1.0 + exp(-gate_value)) * up_value
                                                  * (double)route_weight(CHECKED_STEP, row, slot));
            hidden_error += ((double)values[inner] - reference) * ((double)values[inner] - reference);
            hidden_square += reference * reference;
        }
        struct differing_counts hidden_counts = check_quantization(kquant, &card.hidden, pair, values, card_quantized);
        for (uint32_t output = 0; output < HIDDEN_SIZE; output += DOWN_CHECK_STRIDE) {
            double reference = host_row_product(kquant, DOWN_MATRIX, output, card_quantized);
            double card_value = (double)result[(size_t)pair * HIDDEN_SIZE + output];
            output_error += (card_value - reference) * (card_value - reference);
            output_square += reference * reference;
        }
        /* written so that a NaN from the card is a difference */
        int hidden_matches = sqrt(hidden_error) <= RELATIVE_TOLERANCE * sqrt(hidden_square);
        int output_matches = sqrt(output_error) <= RELATIVE_TOLERANCE * sqrt(output_square);
        printf("  %s, row %u at expert %u: %llu quantized bytes and exponents and %llu group sums of the input differ, "
               "%llu and %llu of the hidden vector; hidden vector off by %.3e of its rms: %s; output off by %.3e of "
               "its rms: %s\n", case_name, row, expert, input_counts.bytes, input_counts.sums,
               hidden_counts.bytes, hidden_counts.sums, sqrt(hidden_error / hidden_square),
               hidden_matches ? "matches the reference" : "DIFFERS", sqrt(output_error / output_square),
               output_matches ? "matches the reference" : "DIFFERS");
        failures += (hidden_matches == corrupted) + (output_matches == corrupted)
                    + ((input_counts.bytes != 0) != scales_moved) + ((input_counts.sums != 0) != sums_moved)
                    + ((hidden_counts.bytes != 0) != scales_moved) + ((hidden_counts.sums != 0) != sums_moved);
    }
    for (uint32_t row = 0; row < row_count; row++) {
        run_checked_layer(kquant, overlap_percent, row, 1, STALE_BYTE_ALONE, alone_result);
        differing += count_differing(result + row * ROW_EXPERT_FLOATS, alone_result, ROW_EXPERT_FLOATS, sizeof(float))
                     + count_differing(result + ROW_SUMS_AT + (size_t)row * HIDDEN_SIZE, alone_result + ROW_SUMS_AT,
                                       HIDDEN_SIZE, sizeof(float));
    }
    printf("  %s: %llu of %zu words differ between a row in this launch and the same row launched alone: %s%s\n",
           case_name, (unsigned long long)differing, row_count * (ROW_EXPERT_FLOATS + HIDDEN_SIZE),
           differing == 0 ? "rows are exact" : "ROWS DIFFER",
           rows_must_differ ? " (required: the order of additions was made to depend on the rows of a job)" : "");
    return failures + ((differing != 0) != rows_must_differ);
}
/* One type on the card that is set: its experts sent, every case checked and timed (the mean time of a layer's routed
 * experts over the steps' routes and inputs, each step started on an idle card), then the type's own line. */
static int probe_type(int kquant, int card_index, const cudaDeviceProp *properties) {
    size_t case_count = ((options.last_overlap - options.first_overlap) / OVERLAP_STEP + 1) * ROW_COUNT_CASES;
    double expert_megabytes = (double)EXPERT_BYTES(kquant) / BYTES_PER_MB, peak_rate = 0.0, megabytes_at_peak = 0.0;
    double card_rate = properties->minor != 0 ? 0.0 : properties->major == SM_80_MAJOR ? CARD_READ_GB_PER_SECOND_SM_80
                       : properties->major == SM_120_MAJOR ? CARD_READ_GB_PER_SECOND_SM_120 : 0.0;
    char case_name[CASE_NAME_BYTES] = "";
    int failures = 0;
    CHECK_CUDA(cudaMemcpy(card.stored, packed[kquant], EXPERT_COUNT * EXPERT_BYTES(kquant), cudaMemcpyHostToDevice));
    printf("card %d %s (sm_%d%d, %d SMs): %s weights, e4m3 activations, %d experts of %.2f MB in memory, %.4f bits per "
           "weight\n", card_index, properties->name, properties->major, properties->minor,
           properties->multiProcessorCount, type_names[kquant], (int)EXPERT_COUNT, expert_megabytes,
           expert_megabytes * BYTES_PER_MB * (double)BITS_PER_BYTE / ((double)MATRIX_KINDS * (double)MATRIX_WEIGHTS));
    for (size_t case_index = 0; case_index < case_count; case_index++) {
        uint32_t overlap = options.first_overlap + (uint32_t)(case_index / ROW_COUNT_CASES) * OVERLAP_STEP;
        uint32_t row_count = row_counts[case_index % ROW_COUNT_CASES], job_count = 0;
        double total_milliseconds = 0.0;
        snprintf(case_name, sizeof(case_name), "overlap %3u, %u row(s), %s", overlap, row_count, type_names[kquant]);
        failures += check_case(kquant, case_name, overlap, row_count);
        if (options.corrupt_weights != 0u) {
            return failures;                             /* the megabytes stay overwritten: one case shows it */
        }
        for (uint32_t step = 0; step < WARM_UP_STEPS + options.step_count; step++) {
            float milliseconds = 0.0f;
            job_count = prepare_layer(step, overlap, 0, row_count);
            CHECK_CUDA(cudaEventRecord(card.started));
            launch_layer[kquant](job_count, row_count);
            CHECK_CUDA(cudaEventRecord(card.stopped));
            CHECK_CUDA(cudaEventSynchronize(card.stopped));
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaEventElapsedTime(&milliseconds, card.started, card.stopped));
            total_milliseconds += step >= WARM_UP_STEPS ? (double)milliseconds : 0.0;
        }
        double microseconds = total_milliseconds * MICROSECONDS_PER_MILLISECOND / options.step_count;
        double megabytes = job_count * expert_megabytes;
        double rate = megabytes / microseconds * GB_PER_SECOND_PER_MB_PER_MICROSECOND;
        megabytes_at_peak = rate > peak_rate ? megabytes : megabytes_at_peak;
        peak_rate = fmax(rate, peak_rate);
        printf("  %s: %8.1f us per layer for the routed experts (both quantizations in it), %2u expert(s) read once, "
               "%6.1f MB read, %6.0f GB/s\n", case_name, microseconds, job_count, megabytes, rate);
    }
    printf("  %s: %.2f MB of stored weights per expert; down in %d warps of %d instructions a block (the 4-bit probe: "
           "%d of %d); read fastest where a layer step reads %.1f MB: %.0f GB/s", type_names[kquant], expert_megabytes,
           (int)DOWN_SLICES, EXPERT_INNER_SIZE / MMA_INPUTS / DOWN_SLICES, (int)FOUR_BIT_PROBE_DOWN_SLICES,
           EXPERT_INNER_SIZE / MMA_INPUTS / FOUR_BIT_PROBE_DOWN_SLICES, megabytes_at_peak, peak_rate);
    if (card_rate > 0.0) {                                /* what the card can read: see the constants */
        printf(", %.0f%% of the card's %.0f GB/s (b02)", peak_rate / card_rate * (double)PERCENT, card_rate);
    }
    printf("\n");
    return failures;
}
static int probe_card(int card_index) {
    cudaDeviceProp properties;
    int failures = 0;
    CHECK_CUDA(cudaSetDevice(card_index));
    CHECK_CUDA(cudaGetDeviceProperties(&properties, card_index));
    CHECK_CUDA(cudaMalloc((void **)&card.stored, EXPERT_COUNT * EXPERT_BYTES(KQUANT_Q3_K)));       /* the larger type */
    CHECK_CUDA(cudaMalloc((void **)&card.jobs, MAX_PAIRS * sizeof(struct expert_job)));
    CHECK_CUDA(cudaMalloc((void **)&card.result, RESULT_FLOATS * sizeof(float)));
    allocate_vectors(&card.input, MAX_ROWS, HIDDEN_SIZE);
    allocate_vectors(&card.hidden, MAX_PAIRS, EXPERT_INNER_SIZE);
    CHECK_CUDA(cudaEventCreate(&card.started));
    CHECK_CUDA(cudaEventCreate(&card.stopped));
    for (int kquant = options.first_type; kquant <= options.last_type; kquant++) {
        failures += probe_type(kquant, card_index, &properties);
    }
    CHECK_CUDA(cudaDeviceReset());           /* frees what this process holds on the card: its memory and its events */
    return failures;
}
/* Sets *number to the whole number in minimum .. maximum that text is; returns 0 if it is none. */
static int parse_number(const char *text, long minimum, long maximum, long *number) {
    char *number_end = NULL;
    *number = strtol(text, &number_end, DECIMAL_BASE);
    return number_end != text && *number_end == '\0' && *number >= minimum && *number <= maximum;
}
int main(int argument_count, char **arguments) {
    long step_count = DEFAULT_STEP_COUNT, overlap = 0;
    int usage_is_wrong = 0, card_count = 0, failures = 0;
    for (int argument_index = 1; argument_index < argument_count; argument_index++) {
        const char *argument = arguments[argument_index];
        const char *value = argument_index + 1 < argument_count ? arguments[argument_index + 1] : NULL;
        if (strcmp(argument, "--corrupt") == 0) {
            options.corrupt_weights = 1;
        } else if (strcmp(argument, "--break-exactness") == 0) {
            options.break_exactness = 1;
        } else if (strcmp(argument, "--break-quantization") == 0) {
            options.break_quantization = 1;
        } else if (strcmp(argument, "--break-route") == 0) {
            options.break_route = 1;
        } else if (value == NULL || ++argument_index >= argument_count) {
            usage_is_wrong = 1;                                    /* every other option has a value */
        } else if (strcmp(argument, "--overlap") == 0) {
            usage_is_wrong |= !parse_number(value, 0, PERCENT, &overlap) || overlap % OVERLAP_STEP != 0;
            options.first_overlap = options.last_overlap = (uint32_t)overlap;
        } else if (strcmp(argument, "--type") == 0) {
            options.first_type = strcmp(value, type_names[KQUANT_Q3_K]) == 0 ? KQUANT_Q3_K : KQUANT_Q2_K;
            options.last_type = options.first_type;
            usage_is_wrong |= strcmp(value, type_names[options.first_type]) != 0;
        } else {
            usage_is_wrong |= strcmp(argument, "--steps") != 0 || !parse_number(value, 1, MAX_STEP_COUNT, &step_count);
        }
    }
    if (usage_is_wrong) {
        fprintf(stderr, "usage: %s [--type q2_k|q3_k] [--overlap 0|50|100] [--steps N] [--corrupt] [--break-exactness] "
                        "[--break-quantization] [--break-route]\n", arguments[0]);
        return EXIT_FAILURE;
    }
    options.step_count = (uint32_t)step_count;
    if (options.break_route != 0u) {         /* no card is touched: every one of these launches must be refused */
        static const char *const broken_bounds[BREAK_ROUTE_END] = {"", "experts past the stored ones",
            "a pair past its pairs", "a job of more columns than rows", "more jobs than pairs", "a row too many"};
        struct expert_job jobs[MAX_PAIRS + EXPERTS_PER_ROW] = {};      /* room for what is asked: a row too many, */
        for (options.break_route = BREAK_EXPERT; options.break_route < BREAK_ROUTE_END; options.break_route++) {
            uint32_t row_count = options.break_route == BREAK_ROWS ? MAX_ROWS + 1u : MAX_ROWS - 1u;   /* or one less */
            int refused = checked_jobs(CHECKED_STEP, 0, 0, row_count, jobs) == 0;       /* overlap 0: a job a pair */
            printf("a launch with %s: %s\n", broken_bounds[options.break_route],
                   refused ? "refused before any launch, as required" : "NOT REFUSED");
            failures += !refused;
        }
        return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    CHECK_CUDA(cudaGetDeviceCount(&card_count));
    for (int kquant = options.first_type; kquant <= options.last_type; kquant++) {
        packed[kquant] = (uint8_t *)calloc(EXPERT_COUNT, EXPERT_BYTES(kquant));       /* packed here, once */
        if (packed[kquant] == NULL) {
            fprintf(stderr, "out of host memory for the packed experts\n");
            return EXIT_FAILURE;
        }
        for (uint32_t matrix = 0; matrix < EXPERT_COUNT * MATRIX_KINDS; matrix++) {
            uint32_t matrix_kind = matrix % MATRIX_KINDS;
            make_canonical_matrix(kquant, matrix / MATRIX_KINDS, matrix_kind, canonical[0]);
            pack_matrix(kquant, canonical[0], matrix_kind == DOWN_MATRIX ? HIDDEN_SIZE : EXPERT_INNER_SIZE,
                        matrix_kind == DOWN_MATRIX ? EXPERT_INNER_SIZE : HIDDEN_SIZE,
                        packed[kquant] + matrix * MATRIX_BYTES(kquant));
        }
    }
    for (int card_index = 0; card_index < card_count; card_index++) {
        failures += probe_card(card_index);
    }
    printf("%d check(s) did not come out as required; %d card(s) probed\n", failures, card_count);
    return failures != 0 || card_count < 1 ? EXIT_FAILURE : EXIT_SUCCESS;
}
