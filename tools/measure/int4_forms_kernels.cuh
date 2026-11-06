/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * int4_forms_kernels.cuh: the routed experts of one MoE layer with W4A16 weights (AutoRound's int4 pack: a weight is
 * (q - z) * s, q an unsigned 4-bit code, s one fp16 scale per output and group of 32 inputs, z one zero point a
 * layer), written for the engine: ONE body for both stages (gate with up, down) and every operand form, the form a
 * compile-time constant, a kernel stamped per choice (I4_STAMP). Round 1, package 6 of the speed campaign
 * (research/speed/ROUND_1.md, "int4_forms"; ledger W1, W2, W3); int4_forms_probe.cu times the stamps.
 *
 * THE STORED ORDER is the operand's own (no pack layout of int4 is committed). Lane 4 g + t of a warp (quad g, place
 * t) has one 16-byte load of codes per tile of 16 outputs and load of 64 inputs:
 *   k16 order (sum24, hsub, bytes, e2m3), the checkpoint's: word i is instruction i of m16n8k16; the code at bit
 *     4 r + 16 h is output g + 8 (r % 2) at input 16 i + 2 t + 8 (r / 2) + h
 *   planes order: word 2 G + o holds the eight consecutive inputs 32 G + 8 t .. + 7 of ONE output, g + 8 o: a
 *     register of m16n8k64 as it is stored
 * and the scales [tile][load][quad], 8 bytes: half 2 G + o is the fp16 of group G of the load and output g + 8 o.
 *
 * THE FORMS, and what i4_prepare_kernel makes of a vector (bfloat16 values) in front of a product:
 *   sum24   the earlier prototype's (four_bit_expert_probe --format int4): 16 + q placed as fp16, two instructions
 *           m16n8k16 a group from a zero accumulator, then (16 + z) * X subtracted, X the fp32 sum of the group's
 *           inputs in ascending order.
 *   hsub    q - z as an exact fp16: a shift, one three-input logic instruction and one fp16 subtraction a register;
 *           no X, and gate with up stores its hidden vector as the fp16 down takes: no launch between the two.
 *   both    the inputs as fp16 in the instruction's order, by THE RULE: a magnitude above 65,504 becomes 65,504,
 *           anything else the nearest fp16 (2^-24 apart under 2^-14), ties to even. T = fma(sum, s, T) per group.
 *           How the unit rounds a sum of 16 products is the card's own (unit 3a: on a CMP 170HX no rule fits).
 *   bytes   class F: the inputs as 8-bit integers a = min(127, nearest((-127 / m) * value)) with m the group's value
 *           of largest magnitude (of equal ones the first) and the scale x = 1 / (-127 / m), 0 for a group of
 *           zeros, all in fp32 (the rule unit 5c priced on logits); the codes as bytes in ONE instruction m16n8k32
 *           u8 by s8 a group, its accumulator started at -z * (the sum of a): the exact integer I of (q - z) * a.
 *           T = fma((float)I, s * x, T).
 *   planes  class F, sm_80, the integers and the bits of bytes: a as two nibble planes, a >> 4 and a & 15, and the
 *           stored words untouched as operand A of m16n8k64 u4 by s4 and u4 by u4: I = 16 * high + low. The two
 *           groups of a load are two COLUMNS of the instruction (a column's operand is zero in the other group's
 *           half), so one pair of instructions serves a load for four rows of a job, a second pair rows 5 to 8.
 *   e2m3    class F, sm_120a: the code as the E2M3 byte of q / 8, the inputs as E4M3 bytes under a power of two per
 *           group (the reference's act_quant, by the card's own conversion), one instruction m16n8k32 e2m3 by e4m3
 *           a group, its accumulator started at -(z / 8) * (the sum of the bytes' values).
 * A slice of the input width runs through its groups in ascending order from +0; the slices are added in slice
 * order from +0 (I4_SLICES: the kernel's, never the rows').
 *
 * A LAUNCH. Grid (tiles of 16 outputs, jobs); a block is I4_STAGE_THREADS threads: per matrix of the stage a warp
 * per slice. i4_prepare_kernel: grid (vectors), exactly size / 32 threads. The kernels check nothing: the host holds
 * every job, grid and array to its range before a launch. No atomics. Measurement tool: not part of the engine.
 */
#ifndef INT4_FORMS_KERNELS_CUH
#define INT4_FORMS_KERNELS_CUH
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

#define I4_SM_80 80                              /* -DPROBE_SM of the two builds, for the preprocessor */
#define I4_SM_120A 120
enum {
    /* the model and a launch: a pair is a row at one of its experts (row * 6 + slot); a stage is named by its
     * number of matrices: gate with up, which share the input, or down */
    I4_HIDDEN_SIZE = 5120, I4_INNER_SIZE = 2304, I4_EXPERTS_PER_ROW = 6, I4_MOST_ROWS = 8, I4_EXPERT_COUNT = 128,
    I4_MOST_PAIRS = I4_MOST_ROWS * I4_EXPERTS_PER_ROW, I4_MATRIX_KINDS = 3, I4_UP_MATRIX = 1, I4_DOWN_MATRIX = 2,
    I4_GATE_UP = 2, I4_DOWN = 1, I4_STAGES = 2, I4_SLICES = 4, I4_ROW_SUM_THREADS = 256,
    /* the instruction: lane 4 g + t holds four registers of weights, two of inputs and four sums */
    I4_LANES = 32, I4_QUADS = 8, I4_QUAD_LANES = 4, I4_TILE_OUTPUTS = 16, I4_HALF_OUTPUTS = 8, I4_LANE_OUTPUTS = 2,
    I4_WEIGHT_REGISTERS = 4, I4_LANE_SUMS = 4, I4_LANE_COLUMNS = 2,
    /* the stored order and a vector's: a load is 64 inputs, two groups, four instructions of 16 */
    I4_GROUP_INPUTS = 32, I4_LOAD_INPUTS = 64, I4_LOAD_GROUPS = 2, I4_LOAD_WORDS = 4, I4_STEP_INPUTS = 16,
    I4_GROUP_STEPS = 2, I4_LOAD_STEPS = 4, I4_HALF_INPUTS = 8, I4_WORD_HALVES = 2, I4_WORD_NIBBLES = 8,
    I4_NIBBLE_BITS = 4, I4_NIBBLE_MASK = 15, I4_BYTE_BITS = 8, I4_BYTE_MASK = 0xFF, I4_HALF_BITS = 16,
    I4_WORD_VALUES = 2,                          /* a vector's operand has two bytes a value, whatever the form */
    I4_RECORD_WORDS = 2,                         /* struct i4_group as words: start, then scale */
    I4_MATRIX_LOADS = I4_INNER_SIZE / I4_TILE_OUTPUTS * (I4_HIDDEN_SIZE / I4_LOAD_INPUTS),    /* of any of the three */
    /* the forms; planes: a pass is four rows of a job */
    I4_SUM24 = 0, I4_HSUB, I4_BYTES, I4_PLANES, I4_E2M3, I4_PASS_ROWS = 4, I4_PASSES = 2,
    /* a register of fp16 weights: 16 + q is the code in the mantissa of 16.0, 1024 + q in that of 1024.0 */
    I4_SUM24_ADDEND = 16, I4_SUM24_BASE = 0x4C004C00, I4_SUM24_MASK = 0x03C003C0, I4_SUM24_SHIFT = 6,
    I4_HSUB_BASE = 0x64006400, I4_HSUB_MASK = 0x000F000F, I4_HALF_REPEAT = 0x00010001,
    /* whole numbers: a word's codes of one output as four bytes; an input's two nibbles */
    I4_CODE_BYTES_MASK = 0x0F0F0F0F, I4_INT8_MOST = 127, I4_HIGH_PLANE_WEIGHT = 16,
    /* E4M3: ((b & 0x80) << 8) | ((b & 0x7F) << 7) is the fp16 of byte b times 2^-8; a code of E2M3 is q / 8 */
    I4_E4M3_SIGN = 0x80, I4_E4M3_MAGNITUDE = 0x7F, I4_E4M3_MAGNITUDE_SHIFT = 7, I4_E2M3_DENOMINATOR_BITS = 3,
    I4_FP32_EXPONENT_SHIFT = 23, I4_FP32_EXPONENT_BIAS = 127, I4_FP32_MANTISSA_MASK = 0x007FFFFF,
    I4_BF16_ROUNDING_BIAS = 0x7FFF
};
#define I4_BF16_KEPT_MASK 0xFFFF0000u            /* the upper half; the bias and its lowest bit round ties to even */
#define I4_SWIGLU_LIMIT 10.0f
#define I4_HALF_LARGEST 65504.0f                 /* THE RULE: the largest fp16 */
#define I4_INT8_LARGEST_TO -127.0f               /* the value of largest magnitude becomes -127 whatever its sign */
#define I4_AMAX_FLOOR 1.0e-4f                    /* the reference's lower bound of a group's largest magnitude */
#define I4_E4M3_MAXIMUM_INVERSE (1.0f / 448.0f)
#define I4_E2M3_START_FACTOR 32.0f               /* 2^8 of the bytes' placement over the 8 of q / 8 */
#define I4_EXPERT_CODE_BYTES ((uint64_t)I4_MATRIX_KINDS * I4_MATRIX_LOADS * I4_LANES * sizeof(uint4))
#define I4_EXPERT_SCALE_BYTES ((uint64_t)I4_MATRIX_KINDS * I4_MATRIX_LOADS * I4_QUADS * sizeof(uint2))
#define I4_STAGE_INPUT_SIZE(matrices) ((matrices) == I4_DOWN ? I4_INNER_SIZE : I4_HIDDEN_SIZE)
#define I4_STAGE_TILES(matrices) (((matrices) == I4_DOWN ? I4_HIDDEN_SIZE : I4_INNER_SIZE) / I4_TILE_OUTPUTS)
#define I4_STAGE_LOADS(matrices) (I4_STAGE_INPUT_SIZE(matrices) / I4_LOAD_INPUTS)
#define I4_STAGE_THREADS(matrices) ((matrices) * I4_SLICES * I4_LANES)
#define I4_IS_FP16(form) ((form) == I4_SUM24 || (form) == I4_HSUB)
/* the words a lane feeds the instruction with per load: of a place, and from one load of a vector to the next */
#define I4_FED_PLACE_WORDS(form) (I4_IS_FP16(form) ? I4_LOAD_STEPS * I4_WORD_HALVES : I4_LOAD_WORDS)
#define I4_FED_LOAD_WORDS(form) \
    ((form) == I4_PLANES ? I4_LOAD_GROUPS * I4_QUAD_LANES * I4_LOAD_WORDS : I4_QUAD_LANES * I4_FED_PLACE_WORDS(form))
static_assert(I4_STAGE_TILES(I4_DOWN) * I4_STAGE_LOADS(I4_DOWN) == I4_MATRIX_LOADS
              && I4_STAGE_LOADS(I4_GATE_UP) % I4_SLICES + I4_STAGE_LOADS(I4_DOWN) % I4_SLICES == 0
              && I4_PASSES == I4_LANE_COLUMNS && I4_PASSES * I4_PASS_ROWS == I4_MOST_ROWS,
              "a matrix of another size than the others, a width that is not whole loads for every slice, or passes "
              "that are not the lane's two columns");

struct i4_job {                                  /* one expert with the rows routed to it: a block's columns */
    uint32_t expert, row_count;                  /* row_count: the columns in use, 1 .. I4_MOST_ROWS */
    uint32_t pair[I4_MOST_ROWS];                 /* of each column; a column not in use repeats column 0 */
    float route_weight[I4_MOST_ROWS];
};
struct i4_group { uint32_t start; float scale; };       /* of a vector's group: an accumulator's start, the scale x */
struct i4_vectors {                              /* the vectors in front of a product: i4_prepare_kernel's work */
    const float *values;                         /* [vector][size]: what is made ready */
    uint32_t *operand;                           /* [vector][size / 2] words, in the form's order (i4_prepare_kernel) */
    struct i4_group *groups;                     /* [vector][size / 32] */
    uint32_t size, pairs_per_vector;             /* 5,120 and 6: a row's input; 2,304 and 1: a pair's hidden vector */
};
struct i4_launch {
    const uint4 *codes;                          /* [expert][matrix kind][tile][load][lane], in the form's order */
    const uint2 *scales;                         /* [expert][matrix kind][tile][load][quad] */
    struct i4_vectors vectors;
    const struct i4_job *jobs;
    float *hidden_values;                        /* gate with up: [pair][inner size]; hsub stores hidden_halves */
    uint16_t *hidden_halves;                     /* instead: [pair][inner size] fp16, the operand of down */
    float *pair_output;                          /* down: [pair][hidden size] */
    uint32_t zero_point;
};
struct i4_lane {                                 /* what a lane reads in its slice: set once, from its job */
    const uint4 *codes;                          /* its tile's loads at its lane: a load on is I4_LANES further */
    const uint2 *scales;                         /* its tile's scales at its quad: a load on is I4_QUADS further */
    const uint32_t *fed[I4_PASSES];              /* the column it feeds, at its place of load 0 (planes: per pass) */
    const struct i4_group *groups[I4_LANE_COLUMNS];     /* of its two result columns (planes: of its row of a pass) */
};
typedef void i4_stage_launcher(const struct i4_launch *launch, uint32_t job_count);
typedef void i4_prepare_launcher(const struct i4_vectors *vectors, uint32_t vector_count, uint32_t zero_point);

/* ---- numbers ---- */
/* An fp32 rounded to bf16, its upper half, to nearest with ties to even: the value the cast to bf16 keeps. */
__device__ __forceinline__ float i4_bf16_value(float value) {
    const uint32_t bits = __float_as_uint(value);
    return __uint_as_float((bits + I4_BF16_ROUNDING_BIAS + ((bits >> I4_HALF_BITS) & 1u)) & I4_BF16_KEPT_MASK);
}
__device__ __forceinline__ uint32_t i4_half_bits(float value) {         /* THE RULE of a 16-bit input */
    return __half_as_ushort(__float2half_rn(fminf(fmaxf(value, -I4_HALF_LARGEST), I4_HALF_LARGEST)));
}
__device__ __forceinline__ float i4_half_value(uint32_t bits) {         /* of the lower 16 bits */
    return __half2float(__ushort_as_half((unsigned short)bits));
}
__device__ __forceinline__ float i4_power_of_two(uint32_t field) {      /* 2^(field - 127), field 1 .. 254 */
    return __uint_as_float(field << I4_FP32_EXPONENT_SHIFT);
}
/* Where value `index` of a vector lies among its fp16 halves: [load][place t][instruction i][register u], the
 * halves of a register being inputs 16 i + 2 t + 8 u and the next. */
__device__ __forceinline__ uint32_t i4_half_place(uint32_t index) {
    const uint32_t step = index / I4_STEP_INPUTS, in_step = index % I4_STEP_INPUTS;
    const uint32_t place = in_step % I4_HALF_INPUTS / I4_WORD_HALVES, load = step / I4_LOAD_STEPS;
    return (((load * I4_QUAD_LANES + place) * I4_LOAD_STEPS + step % I4_LOAD_STEPS) * I4_WORD_HALVES
            + in_step / I4_HALF_INPUTS) * I4_WORD_HALVES + in_step % I4_WORD_HALVES;
}
#if PROBE_SM == I4_SM_120A
/* Two fp32 as two E4M3 bytes, the first in the upper one, by the card's conversion: to nearest, ties to even,
 * beyond the largest code the largest code (unit 3a: the format's rounding on all 2^32 patterns). */
__device__ __forceinline__ uint32_t i4_e4m3_pair(float first, float second) {
    uint32_t converted = 0;
    asm volatile("{ .reg .b16 packed;\ncvt.rn.satfinite.e4m3x2.f32 packed, %1, %2;\ncvt.u32.u16 %0, packed; }\n"
                 : "=r"(converted)
                 : "f"(first), "f"(second));
    return converted;
}
#endif

/* ---- a vector made ready: grid (vectors), a thread per group of 32 (the header says what each form gets) ---- */
template <int form>
__global__ void i4_prepare_kernel(struct i4_vectors vectors, uint32_t zero_point) {
    const uint32_t group = threadIdx.x, load = group / I4_LOAD_GROUPS, in_load = group % I4_LOAD_GROUPS;
    const float *values = vectors.values + (size_t)blockIdx.x * vectors.size + group * I4_GROUP_INPUTS;
    uint32_t *words = vectors.operand + (size_t)blockIdx.x * (vectors.size / I4_WORD_VALUES);
    struct i4_group *record = vectors.groups + (size_t)blockIdx.x * (vectors.size / I4_GROUP_INPUTS) + group;
    uint32_t codes[I4_GROUP_INPUTS], start = 0;  /* a value's fp16 bits, its 8-bit integer or its E4M3 byte */
    float largest = 0.0f, sum = 0.0f, scale = 1.0f, inverse_scale = 0.0f;
    int32_t integers = 0;
    for (uint32_t index = 0; index < I4_GROUP_INPUTS; index++) {
        largest = fabsf(values[index]) > fabsf(largest) ? values[index] : largest;
    }
    if (form == I4_BYTES || form == I4_PLANES) {
        inverse_scale = largest == 0.0f ? 0.0f : I4_INT8_LARGEST_TO / largest;
        scale = largest == 0.0f ? 0.0f : 1.0f / inverse_scale;
    }
#if PROBE_SM == I4_SM_120A
    if (form == I4_E2M3) {                       /* the largest magnitude over 448, rounded up to a power of two */
        const uint32_t ratio = __float_as_uint(fmaxf(fabsf(largest), I4_AMAX_FLOOR) * I4_E4M3_MAXIMUM_INVERSE);
        const uint32_t field = (ratio >> I4_FP32_EXPONENT_SHIFT) + ((ratio & I4_FP32_MANTISSA_MASK) != 0u ? 1u : 0u);
        inverse_scale = i4_power_of_two(I4_FP32_EXPONENT_BIAS + I4_FP32_EXPONENT_BIAS - field);
        scale = i4_power_of_two(field + I4_E2M3_DENOMINATOR_BITS);
        for (uint32_t index = 0; index < I4_GROUP_INPUTS; index += I4_WORD_HALVES) {
            const uint32_t pair = i4_e4m3_pair(values[index] * inverse_scale, values[index + 1u] * inverse_scale);
            codes[index] = pair >> I4_BYTE_BITS;
            codes[index + 1u] = pair & I4_BYTE_MASK;
        }
    }
#endif
    for (uint32_t index = 0; index < I4_GROUP_INPUTS; index++) {
        if (I4_IS_FP16(form)) {
            codes[index] = i4_half_bits(values[index]);
            sum += i4_half_value(codes[index]);
        } else if (form == I4_E2M3) {            /* a byte's value times 2^-8: their sum is exact */
            sum += i4_half_value(((codes[index] & I4_E4M3_SIGN) << I4_BYTE_BITS)
                                 | ((codes[index] & I4_E4M3_MAGNITUDE) << I4_E4M3_MAGNITUDE_SHIFT));
        } else {
            const int32_t nearest = __float2int_rn(inverse_scale * values[index]);
            const int32_t integer = nearest > I4_INT8_MOST ? (int32_t)I4_INT8_MOST : nearest;
            codes[index] = (uint32_t)integer & I4_BYTE_MASK;
            integers += integer;
        }
    }
    for (uint32_t place = 0; place < I4_QUAD_LANES; place++) {
        if (form == I4_PLANES) {                 /* [group][place]: high plane, low plane, each in its group's half */
            uint32_t *registers = words + (size_t)(group * I4_QUAD_LANES + place) * I4_LOAD_WORDS;
            uint32_t high = 0, low = 0;
            for (uint32_t nibble = 0; nibble < I4_WORD_NIBBLES; nibble++) {
                const uint32_t code = codes[place * I4_WORD_NIBBLES + nibble];
                high |= ((code >> I4_NIBBLE_BITS) & I4_NIBBLE_MASK) << (nibble * I4_NIBBLE_BITS);
                low |= (code & I4_NIBBLE_MASK) << (nibble * I4_NIBBLE_BITS);
            }
            registers[in_load] = high;
            registers[1u - in_load] = 0u;
            registers[I4_LOAD_GROUPS + in_load] = low;
            registers[I4_LOAD_GROUPS + 1u - in_load] = 0u;
        }
        for (uint32_t step = 0; form != I4_PLANES && step < I4_GROUP_STEPS; step++) {   /* [load][place][step] */
            const uint32_t *first = codes + step * I4_STEP_INPUTS + place * I4_WORD_HALVES;
            const size_t at = (size_t)(load * I4_QUAD_LANES + place) * I4_LOAD_STEPS + in_load * I4_GROUP_STEPS + step;
            if (I4_IS_FP16(form)) {              /* two registers: inputs 2 t, 2 t + 1, then 2 t + 8, 2 t + 9 */
                words[at * I4_WORD_HALVES] = first[0] | (first[1] << I4_HALF_BITS);
                words[at * I4_WORD_HALVES + 1u] = first[I4_HALF_INPUTS] | (first[I4_HALF_INPUTS + 1] << I4_HALF_BITS);
            } else {                             /* one word: bytes 0 and 2 the first register, 1 and 3 the second */
                words[at] = first[0] | (first[I4_HALF_INPUTS] << I4_BYTE_BITS) | (first[1] << I4_HALF_BITS)
                            | (first[I4_HALF_INPUTS + 1] << (I4_HALF_BITS + I4_BYTE_BITS));
            }
        }
    }
    if (form == I4_SUM24) { start = __float_as_uint((float)(I4_SUM24_ADDEND + zero_point) * sum); }
    if (form == I4_E2M3) { start = __float_as_uint(-((float)zero_point * sum) * I4_E2M3_START_FACTOR); }
    if (form == I4_BYTES || form == I4_PLANES) { start = (uint32_t)(-(int32_t)zero_point * integers); }
    if (form != I4_HSUB) {
        record->start = start;
        record->scale = scale;
    }
}
template <int form>
inline void i4_launch_prepare(const struct i4_vectors *vectors, uint32_t vector_count, uint32_t zero_point) {
    i4_prepare_kernel<form><<<vector_count, vectors->size / I4_GROUP_INPUTS>>>(*vectors, zero_point);
}

/* ---- the instructions. sums: outputs g (elements 0, 1) and g + 8 for the lane's columns 2 t and 2 t + 1, the
 * accumulators on entry; weights: the lane's four registers of operand A; first, second: its two of operand B ---- */
#define I4_TENSOR_CORE(name, sum_type, result, input, spelling)                                                      \
    __device__ __forceinline__ void name(sum_type *sums, const uint32_t *weights, uint32_t first, uint32_t second) { \
        asm volatile("mma.sync.aligned." spelling " {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"    \
                     : result(sums[0]), result(sums[1]), result(sums[2]), result(sums[3])                            \
                     : "r"(weights[0]), "r"(weights[1]), "r"(weights[2]), "r"(weights[3]), "r"(first), "r"(second),  \
                       input(sums[0]), input(sums[1]), input(sums[2]), input(sums[3]));                              \
    }
I4_TENSOR_CORE(i4_core_f16, float, "=f", "f", "m16n8k16.row.col.f32.f16.f16.f32")
I4_TENSOR_CORE(i4_core_u8_s8, uint32_t, "=r", "r", "m16n8k32.row.col.s32.u8.s8.s32")
I4_TENSOR_CORE(i4_core_u4_s4, uint32_t, "=r", "r", "m16n8k64.row.col.s32.u4.s4.s32")
I4_TENSOR_CORE(i4_core_u4_u4, uint32_t, "=r", "r", "m16n8k64.row.col.s32.u4.u4.s32")
#if PROBE_SM == I4_SM_120A
I4_TENSOR_CORE(i4_core_e2m3, float, "=f", "f", "m16n8k32.row.col.kind::f8f6f4.f32.e2m3.e4m3.f32")
#endif
/* Register `weight_register` of an m16n8k16 instruction from its word of codes (the code of half h at bit 4 r +
 * 16 h): sum24 the fp16 pair of 16 + q; hsub that of q - z: 1024 + q less `offsets`, which is 1024 + z twice. */
template <int form>
__device__ __forceinline__ uint32_t i4_fp16_weights(uint32_t word, uint32_t weight_register, uint32_t offsets) {
    const uint32_t codes = word >> (weight_register * I4_NIBBLE_BITS);
    uint32_t placed = ((codes << I4_SUM24_SHIFT) & I4_SUM24_MASK) | I4_SUM24_BASE;
    if (form == I4_HSUB) {
        placed = (codes & I4_HSUB_MASK) | I4_HSUB_BASE;
        asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(placed) : "r"(placed), "r"(offsets));
    }
    return placed;
}
__device__ __forceinline__ void i4_load_words(const void *address, uint32_t *words) {      /* one 16-byte load */
    const uint4 loaded = *(const uint4 *)address;
    const uint32_t loaded_words[I4_LOAD_WORDS] = {loaded.x, loaded.y, loaded.z, loaded.w};
    for (uint32_t index = 0; index < I4_LOAD_WORDS; index++) { words[index] = loaded_words[index]; }
}

/* ---- the slice ---- */
/* The places a lane reads in its slice, from its job: its matrix is the stage's own (down) or its warp's (gate in
 * the first warps, up in the others). Quad g feeds column g and place t holds the sums of columns 2 t, 2 t + 1;
 * planes: in pass p quad g feeds group g % 2 of column 4 p + g / 2 and place t holds column 4 p + t, both groups. */
template <int form, int matrices>
__device__ __forceinline__ void i4_set_lane(const struct i4_launch *launch, const struct i4_job *job,
                                            struct i4_lane *lane) {
    const struct i4_vectors *vectors = &launch->vectors;
    const uint32_t in_warp = threadIdx.x % I4_LANES, quad = in_warp / I4_QUAD_LANES, place = in_warp % I4_QUAD_LANES;
    const uint32_t kind = matrices == I4_DOWN ? (uint32_t)I4_DOWN_MATRIX : threadIdx.x / I4_LANES / I4_SLICES;
    const size_t matrix = (size_t)job->expert * I4_MATRIX_KINDS + kind;
    const size_t first_load = (matrix * I4_STAGE_TILES(matrices) + blockIdx.x) * I4_STAGE_LOADS(matrices);
    const uint32_t fed_place = form == I4_PLANES ? (quad % I4_LOAD_GROUPS * I4_QUAD_LANES + place) * I4_LOAD_WORDS
                                                 : place * I4_FED_PLACE_WORDS(form);
    lane->codes = launch->codes + first_load * I4_LANES + in_warp;
    lane->scales = launch->scales + first_load * I4_QUADS + quad;
    for (uint32_t index = 0; index < I4_PASSES; index++) {       /* a pass of planes, else a result column */
        const uint32_t fed = form == I4_PLANES ? index * I4_PASS_ROWS + quad / I4_LOAD_GROUPS : quad;
        const uint32_t held = form == I4_PLANES ? index * I4_PASS_ROWS + place : place * I4_LANE_COLUMNS + index;
        const size_t fed_vector = job->pair[fed] / vectors->pairs_per_vector;
        const size_t held_vector = job->pair[held] / vectors->pairs_per_vector;
        lane->fed[index] = vectors->operand + fed_vector * (vectors->size / I4_WORD_VALUES) + fed_place;
        lane->groups[index] = vectors->groups + held_vector * (vectors->size / I4_GROUP_INPUTS);
    }
}
/* One warp, one tile of 16 outputs of one job: loads first_load .. first_load + load_count - 1 of the input width,
 * each lane's four sums to partial_sums [thread][element]: element e is output g + 8 (e / 2) of column 2 t + e % 2;
 * planes: output g + 8 (e % 2) of column 4 (e / 2) + t. passes: 2 where a job of planes has more than four rows. */
template <int form>
__device__ __forceinline__ void i4_slice(const struct i4_lane *lane, uint32_t zero_point, uint32_t passes,
                                         uint32_t first_load, uint32_t load_count, float *partial_sums) {
    const uint32_t offsets = I4_HSUB_BASE + zero_point * I4_HALF_REPEAT;
    float totals[I4_LANE_SUMS] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t load = first_load; load < first_load + load_count; load++) {
        const uint2 stored_scales = lane->scales[(size_t)load * I4_QUADS];
        const float scales[I4_LOAD_GROUPS][I4_LANE_OUTPUTS] = {
            {i4_half_value(stored_scales.x), i4_half_value(stored_scales.x >> I4_HALF_BITS)},
            {i4_half_value(stored_scales.y), i4_half_value(stored_scales.y >> I4_HALF_BITS)}};
        uint32_t words[I4_LOAD_WORDS], fed[I4_LOAD_STEPS * I4_WORD_HALVES], records[I4_LANE_COLUMNS][I4_LOAD_WORDS];
        i4_load_words(lane->codes + (size_t)load * I4_LANES, words);
        for (uint32_t pass = 0; form == I4_PLANES && pass < passes; pass++) {
            uint32_t sums[I4_LANE_SUMS] = {0, 0, 0, 0};          /* outputs g, g + 8 by groups 0, 1 of the pass's row */
            i4_load_words(lane->fed[pass] + (size_t)load * I4_FED_LOAD_WORDS(form), fed);
            i4_load_words(lane->groups[pass] + (size_t)load * I4_LOAD_GROUPS, records[0]);
            i4_core_u4_s4(sums, words, fed[0], fed[1]);
            for (uint32_t element = 0; element < I4_LANE_SUMS; element++) {
                sums[element] = sums[element] * I4_HIGH_PLANE_WEIGHT
                                + records[0][element % I4_LOAD_GROUPS * I4_RECORD_WORDS];
            }
            i4_core_u4_u4(sums, words, fed[I4_LOAD_GROUPS], fed[I4_LOAD_GROUPS + 1]);
            for (uint32_t element = 0; element < I4_LANE_SUMS; element++) {
                const uint32_t group = element % I4_LOAD_GROUPS, output = element / I4_LOAD_GROUPS;
                const float input_scale = __uint_as_float(records[0][group * I4_RECORD_WORDS + 1u]);
                totals[pass * I4_LANE_OUTPUTS + output] = fmaf((float)(int32_t)sums[element],
                                                               scales[group][output] * input_scale,
                                                               totals[pass * I4_LANE_OUTPUTS + output]);
            }
        }
        if (form != I4_PLANES) { i4_load_words(lane->fed[0] + (size_t)load * I4_FED_LOAD_WORDS(form), fed); }
        if (I4_IS_FP16(form)) {
            i4_load_words(lane->fed[0] + (size_t)load * I4_FED_LOAD_WORDS(form) + I4_LOAD_WORDS, fed + I4_LOAD_WORDS);
        }
        for (uint32_t column = 0; form != I4_PLANES && form != I4_HSUB && column < I4_LANE_COLUMNS; column++) {
            i4_load_words(lane->groups[column] + (size_t)load * I4_LOAD_GROUPS, records[column]);
        }
        for (uint32_t group = 0; form != I4_PLANES && group < I4_LOAD_GROUPS; group++) {
            const uint32_t first = words[group * I4_GROUP_STEPS], second = words[group * I4_GROUP_STEPS + 1u];
            float sums[I4_LANE_SUMS] = {0.0f, 0.0f, 0.0f, 0.0f};
            uint32_t whole[I4_LANE_SUMS], weights[I4_WEIGHT_REGISTERS] = {
                first & I4_CODE_BYTES_MASK, (first >> I4_NIBBLE_BITS) & I4_CODE_BYTES_MASK,
                second & I4_CODE_BYTES_MASK, (second >> I4_NIBBLE_BITS) & I4_CODE_BYTES_MASK};
            for (uint32_t step = 0; I4_IS_FP16(form) && step < I4_GROUP_STEPS; step++) {
                const uint32_t instruction = group * I4_GROUP_STEPS + step;
                for (uint32_t weight_register = 0; weight_register < I4_WEIGHT_REGISTERS; weight_register++) {
                    weights[weight_register] = i4_fp16_weights<form>(words[instruction], weight_register, offsets);
                }
                i4_core_f16(sums, weights, fed[instruction * I4_WORD_HALVES], fed[instruction * I4_WORD_HALVES + 1u]);
            }
            for (uint32_t element = 0; !I4_IS_FP16(form) && element < I4_LANE_SUMS; element++) {
                whole[element] = records[element % I4_LANE_COLUMNS][group * I4_RECORD_WORDS];
                sums[element] = __uint_as_float(whole[element]);
            }
            if (form == I4_BYTES) {
                i4_core_u8_s8(whole, weights, fed[group * I4_GROUP_STEPS], fed[group * I4_GROUP_STEPS + 1u]);
                for (uint32_t element = 0; element < I4_LANE_SUMS; element++) {
                    sums[element] = (float)(int32_t)whole[element];
                }
            }
#if PROBE_SM == I4_SM_120A
            if (form == I4_E2M3) {
                i4_core_e2m3(sums, weights, fed[group * I4_GROUP_STEPS], fed[group * I4_GROUP_STEPS + 1u]);
            }
#endif
            for (uint32_t element = 0; element < I4_LANE_SUMS; element++) {
                const uint32_t *record = records[element % I4_LANE_COLUMNS] + group * I4_RECORD_WORDS;
                const float scale = scales[group][element / I4_LANE_COLUMNS];
                if (form == I4_SUM24) {
                    totals[element] = fmaf(sums[element] - __uint_as_float(record[0]), scale, totals[element]);
                } else if (form == I4_HSUB) {
                    totals[element] = fmaf(sums[element], scale, totals[element]);
                } else {
                    totals[element] = fmaf(sums[element], scale * __uint_as_float(record[1]), totals[element]);
                }
            }
        }
    }
    for (uint32_t element = 0; element < I4_LANE_SUMS; element++) {
        partial_sums[threadIdx.x * I4_LANE_SUMS + element] = totals[element];
    }
}

/* ---- a block of threads: its slices, and what it stores ---- */
/* One element of one matrix: the partial sums of its slices (warps first_warp ..), added in slice order from +0.
 * broken_order (the stamp of --break exactness, never timed): the sum of a job of several rows starts at another
 * slice, which depends on their number. */
template <int broken_order>
__device__ __forceinline__ float i4_sum_slices(const float *partial_sums, uint32_t first_warp, uint32_t element,
                                               uint32_t rows) {
    const uint32_t first_slice = broken_order != 0 && rows > 1u ? 1u + rows % (I4_SLICES - 1u) : 0u;
    float sum = 0.0f;
    for (uint32_t position = 0; position < I4_SLICES; position++) {
        const uint32_t warp = first_warp + (first_slice + position) % I4_SLICES;
        sum += partial_sums[(warp * I4_LANES + threadIdx.x) * I4_LANE_SUMS + element];
    }
    return sum;
}
/* After the block's barrier the lanes of warp 0 add the slices. Two matrices: gate and up, each rounded to bf16 as
 * the reference's product returns it; h = silu(min(gate, L)) * clamp(up, -L, L) * route weight, cast to bf16, goes
 * to hidden_values, or (hsub) by THE RULE to hidden_halves. One matrix: down, as fp32, to pair_output. */
template <int form, int matrices, int broken_order>
__device__ __forceinline__ void i4_close_block(const struct i4_launch *launch, const struct i4_job *job,
                                               const float *partial_sums) {
    const uint32_t quad = threadIdx.x / I4_QUAD_LANES, place = threadIdx.x % I4_QUAD_LANES, rows = job->row_count;
    for (uint32_t element = 0; element < I4_LANE_SUMS && threadIdx.x < I4_LANES; element++) {
        const uint32_t half = form == I4_PLANES ? element % I4_LANE_OUTPUTS : element / I4_LANE_COLUMNS;
        const uint32_t column = form == I4_PLANES ? element / I4_LANE_OUTPUTS * I4_PASS_ROWS + place
                                                  : place * I4_LANE_COLUMNS + element % I4_LANE_COLUMNS;
        const uint32_t output = blockIdx.x * I4_TILE_OUTPUTS + quad + half * I4_HALF_OUTPUTS;
        const float value = i4_sum_slices<broken_order>(partial_sums, 0, element, rows);
        if (column < rows && matrices == I4_DOWN) {
            launch->pair_output[(size_t)job->pair[column] * I4_HIDDEN_SIZE + output] = value;
        } else if (column < rows) {
            const float gate_value = fminf(i4_bf16_value(value), I4_SWIGLU_LIMIT);
            const float up_value = i4_bf16_value(
                i4_sum_slices<broken_order>(partial_sums, I4_UP_MATRIX * I4_SLICES, element, rows));
            const float clamped = fminf(fmaxf(up_value, -I4_SWIGLU_LIMIT), I4_SWIGLU_LIMIT);
            const float hidden = i4_bf16_value(gate_value / (1.0f + expf(-gate_value)) * clamped
                                               * job->route_weight[column]);
            const size_t first = (size_t)job->pair[column] * I4_INNER_SIZE;
            if (form == I4_HSUB) {
                launch->hidden_halves[first + i4_half_place(output)] = (uint16_t)i4_half_bits(hidden);
            } else {
                launch->hidden_values[first + output] = hidden;
            }
        }
    }
}
/* One product stage of every job, for one thread. */
template <int form, int matrices, int broken_order>
__device__ __forceinline__ void i4_block(const struct i4_launch *launch, float *partial_sums) {
    const struct i4_job *job = &launch->jobs[blockIdx.y];
    const uint32_t passes = form == I4_PLANES && job->row_count > I4_PASS_ROWS ? (uint32_t)I4_PASSES : 1u;
    const uint32_t slice_loads = I4_STAGE_LOADS(matrices) / I4_SLICES;
    struct i4_lane lane;
    i4_set_lane<form, matrices>(launch, job, &lane);
    i4_slice<form>(&lane, launch->zero_point, passes, threadIdx.x / I4_LANES % I4_SLICES * slice_loads, slice_loads,
                   partial_sums);
    __syncthreads();
    i4_close_block<form, matrices, broken_order>(launch, job, partial_sums);
}
/* A row's six experts, each rounded to bf16 as the reference's product returns it, added in slot order: grid (rows). */
__global__ void i4_row_sum_kernel(const float *pair_output, float *row_output) {
    const float *row_pairs = pair_output + (size_t)blockIdx.x * I4_EXPERTS_PER_ROW * I4_HIDDEN_SIZE;
    for (uint32_t column = threadIdx.x; column < I4_HIDDEN_SIZE; column += blockDim.x) {
        float sum = 0.0f;
        for (uint32_t slot = 0; slot < I4_EXPERTS_PER_ROW; slot++) {
            sum += i4_bf16_value(row_pairs[(size_t)slot * I4_HIDDEN_SIZE + column]);
        }
        row_output[(size_t)blockIdx.x * I4_HIDDEN_SIZE + column] = sum;
    }
}

/* ---- the stamps: the two kernels of a form and their launches. No launch bound: the baseline is the earlier
 * probe's kernel, which states none, and the forms are compared with it; a bound is a variant of a later batch. ---- */
#define I4_STAMP_STAGE(name, form, matrices, order)                                                   \
    __global__ void name(struct i4_launch launch) {                                                   \
        __shared__ float partial_sums[I4_STAGE_THREADS(matrices) * I4_LANE_SUMS];                     \
        i4_block<form, matrices, order>(&launch, partial_sums);                                       \
    }                                                                                                 \
    inline void launch_##name(const struct i4_launch *launch, uint32_t job_count) {                   \
        name<<<dim3(I4_STAGE_TILES(matrices), job_count), I4_STAGE_THREADS(matrices)>>>(*launch);     \
    }
#define I4_STAMP(tag, form, order)                              \
    I4_STAMP_STAGE(i4_gate_up_##tag, form, I4_GATE_UP, order)   \
    I4_STAMP_STAGE(i4_down_##tag, form, I4_DOWN, order)
I4_STAMP(sum24, I4_SUM24, 0)
I4_STAMP(sum24_broken_order, I4_SUM24, 1)
I4_STAMP(hsub, I4_HSUB, 0)
I4_STAMP(bytes, I4_BYTES, 0)
#if PROBE_SM == I4_SM_80
I4_STAMP(planes, I4_PLANES, 0)
#else
I4_STAMP(e2m3, I4_E2M3, 0)
#endif
#endif /* INT4_FORMS_KERNELS_CUH */
