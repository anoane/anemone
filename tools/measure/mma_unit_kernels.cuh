/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * mma_unit_kernels.cuh: the kernels of mma_unit_probe.cu (speed campaign, round 1, unit 3a): one per tensor-core
 * form. A kernel computes nothing of its own: a block is ONE warp, every lane takes its registers of the two
 * multiplicands, its four accumulators and, for a block-scaled form, its two scale words from its record, the warp
 * executes the form's instruction ONCE, and every lane stores the four sums it is returned. The spellings are those
 * measurements/h01_mma_forms/forms.log shows compiling, each as one instruction, but the three 4-bit forms on
 * sm_120a, which the compiler lowers to two 8-bit instructions and integer code:
 *   f16, bf16          HMMA.16816.F32 (every exact kernel's instruction today: the reference point) and .BF16
 *   s8, u8_s8          IMMA.16832.S8.S8 and .U8.S8 (A is the unsigned side); s8_saturating: .S8.S8.SAT
 *   u4_s4, u4, s4      IMMA.16864.U4.S4, .U4.U4, .S4.S4
 *   sm_120a only: e2m1_e4m3, e2m3_e4m3 (QMMA.16832.F32.E2M1.E4M3 and .E2M3.E4M3: kind f8f6f4, an element a byte),
 *   e4m3, e4m3_k16     QMMA.16832.F32.E4M3.E4M3 and QMMA.16816.F32.E4M3.E4M3 (the one form of two and one registers)
 *   e2m1_e4m3_scaled, e4m3_scaled      QMMA.SF.16832.F32.<A>.E4M3.E8: a kernel per pair of selectors, which are
 *                      written into the instruction (in registers they cost a branch: forms.log line 112)
 *   convert_e4m3, convert_ue8m0        cvt.rn.satfinite.e4m3x2.f32 and cvt.rp.satfinite.ue8m0x2.f32: no record; a
 *                      thread converts the two fp32 patterns of its own index and stores the 16 bits it gets
 * What a launch must satisfy (a kernel checks nothing: the host validates every launch, check 10 of CONVENTIONS 5):
 * a form's kernel: blocks of exactly UNIT_LANES threads, a record for every thread of the grid and
 * UNIT_SUMS_PER_LANE result words for each; a conversion's: a 16-bit result for every thread of the grid. A thread
 * reads its own record and writes its own results: no shared memory, no atomics, no loop. Not part of the engine.
 */
#ifndef MMA_UNIT_KERNELS_CUH
#define MMA_UNIT_KERNELS_CUH
#include <cuda_runtime.h>
#include <stdint.h>

enum {
    UNIT_LANES = 32,                             /* of a warp: the block of every form's kernel */
    UNIT_A_REGISTERS = 4, UNIT_B_REGISTERS = 2,  /* the most a form takes of a lane */
    UNIT_SUMS_PER_LANE = 4,                      /* two outputs by two columns */
    UNIT_SELECTOR_COUNT = 23,                    /* the kernels of a block-scaled form: UNIT_EVERY_SELECTOR */
    UNIT_CASE_SUMS = UNIT_LANES * UNIT_SUMS_PER_LANE
};
#define UNIT_SM_120A 120                         /* -DPROBE_SM of the sm_120a build, for the preprocessor */
#define UNIT_PATTERN_FIRST 0x3F800000u           /* pattern 0 of a conversion: 1, so that its pair is 1 and 2 */
#define UNIT_PATTERN_STRIDE 0x9E3779B1u          /* odd: the patterns of 2^32 indices are every fp32 pattern once */
#define UNIT_PARTNER_STEP 0x00800000u            /* the second pattern of a conversion: the first's value doubled */
#define UNIT_FP32_PATTERN(index) (UNIT_PATTERN_FIRST + (index) * UNIT_PATTERN_STRIDE)     /* index: a uint32_t */

struct unit_lane_record {                        /* what one lane gives the instruction */
    uint32_t a[UNIT_A_REGISTERS];                /* its registers of multiplicand A, as many as the form takes */
    uint32_t b[UNIT_B_REGISTERS];                /* of multiplicand B */
    uint32_t c[UNIT_SUMS_PER_LANE];              /* its accumulators: the bits of four fp32, or four whole numbers */
    uint32_t scale_a, scale_b;                   /* its scale words: a block-scaled form reads them, no other */
};

/* One kernel. Its PTX text is `spelling`, the operand lists of `shape` (D in %0 to %3, then the registers of A, of
 * B and the four of C) and `tail`; `shape` and `scales` are the ends of the names of the lists below: a list has
 * commas, so it is named where it is used and never passed through a macro. */
#define UNIT_MMA_KERNEL(kernel_name, sum_type, result_letter, sum_letter, word_to_sum, sum_to_word, spelling,         \
                        shape, tail, scales)                                                                          \
    __global__ void kernel_name(const struct unit_lane_record *records, uint32_t *results)                            \
    {                                                                                                                 \
        uint64_t thread = (uint64_t)blockIdx.x * UNIT_LANES + threadIdx.x;                                            \
        const struct unit_lane_record *lane = &records[thread];                                                       \
        uint32_t *sums = &results[thread * UNIT_SUMS_PER_LANE];                                                       \
        sum_type c0 = word_to_sum(lane->c[0]), c1 = word_to_sum(lane->c[1]), c2 = word_to_sum(lane->c[2]);            \
        sum_type c3 = word_to_sum(lane->c[3]), d0, d1, d2, d3;                                                        \
        asm volatile(spelling UNIT_D4_##shape tail ";\n"                                                              \
                     : result_letter(d0), result_letter(d1), result_letter(d2), result_letter(d3)                     \
                     : UNIT_INPUTS_##shape, sum_letter(c0), sum_letter(c1), sum_letter(c2), sum_letter(c3)            \
                       UNIT_##scales);                                                                                \
        sums[0] = sum_to_word(d0);                                                                                    \
        sums[1] = sum_to_word(d1);                                                                                    \
        sums[2] = sum_to_word(d2);                                                                                    \
        sums[3] = sum_to_word(d3);                                                                                    \
    }
#define UNIT_D4_A4_B2 " {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}"
#define UNIT_D4_A2_B1 " {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10}"
#define UNIT_INPUTS_A2_B1 "r"(lane->a[0]), "r"(lane->a[1]), "r"(lane->b[0])
#define UNIT_INPUTS_A4_B2                                                                                             \
    "r"(lane->a[0]), "r"(lane->a[1]), "r"(lane->a[2]), "r"(lane->a[3]), "r"(lane->b[0]), "r"(lane->b[1])
#define UNIT_NO_SCALES
#define UNIT_SCALES , "r"(lane->scale_a), "r"(lane->scale_b)
#define UNIT_SAME_WORD(word) (word)
#define UNIT_FLOAT_KERNEL(kernel_name, spelling, shape, tail, scales)                                                 \
    UNIT_MMA_KERNEL(kernel_name, float, "=f", "f", __uint_as_float, __float_as_uint, spelling, shape, tail, scales)
#define UNIT_FLOAT_FORM(kernel_name, spelling) UNIT_FLOAT_KERNEL(kernel_name, spelling, A4_B2, "", NO_SCALES)
#define UNIT_WHOLE_FORM(kernel_name, spelling)                                                                        \
    UNIT_MMA_KERNEL(kernel_name, uint32_t, "=r", "r", UNIT_SAME_WORD, UNIT_SAME_WORD, spelling, A4_B2, "", NO_SCALES)
/* A block-scaled form with its selectors: after C the scale word of A and {byte, thread pair of the quad}, then
 * the scale word of B and {byte, thread of the quad}. */
#define UNIT_SCALED_FORM(form, spelling, a_byte, a_pair, b_byte, b_thread)                                            \
    UNIT_FLOAT_KERNEL(unit_kernel_##form##_a##a_byte##a_pair##_b##b_byte##b_thread, spelling, A4_B2,                  \
                      ", %14, {" #a_byte ", " #a_pair "}, %15, {" #b_byte ", " #b_thread "}", SCALES)
/* Every selector of one side beside {0, 0} of the other: 8 of A's, then the 15 others of B's. The first is the
 * form's own kernel. stamp(form, spelling, byte of A, thread pair of A, byte of B, thread of B). */
#define UNIT_EVERY_SELECTOR(stamp, form, spelling)                                                                    \
    stamp(form, spelling, 0, 0, 0, 0) stamp(form, spelling, 1, 0, 0, 0) stamp(form, spelling, 2, 0, 0, 0)             \
    stamp(form, spelling, 3, 0, 0, 0) stamp(form, spelling, 0, 1, 0, 0) stamp(form, spelling, 1, 1, 0, 0)             \
    stamp(form, spelling, 2, 1, 0, 0) stamp(form, spelling, 3, 1, 0, 0) stamp(form, spelling, 0, 0, 1, 0)             \
    stamp(form, spelling, 0, 0, 2, 0) stamp(form, spelling, 0, 0, 3, 0) stamp(form, spelling, 0, 0, 0, 1)             \
    stamp(form, spelling, 0, 0, 1, 1) stamp(form, spelling, 0, 0, 2, 1) stamp(form, spelling, 0, 0, 3, 1)             \
    stamp(form, spelling, 0, 0, 0, 2) stamp(form, spelling, 0, 0, 1, 2) stamp(form, spelling, 0, 0, 2, 2)             \
    stamp(form, spelling, 0, 0, 3, 2) stamp(form, spelling, 0, 0, 0, 3) stamp(form, spelling, 0, 0, 1, 3)             \
    stamp(form, spelling, 0, 0, 2, 3) stamp(form, spelling, 0, 0, 3, 3)
/* A conversion of two fp32 to two 8-bit codes in one 16-bit result: thread i of a launch converts the patterns
 * UNIT_FP32_PATTERN(first_index + i) and that plus UNIT_PARTNER_STEP, so that 2^32 threads convert every
 * pattern as the first operand and as the second. */
#define UNIT_CONVERT_KERNEL(kernel_name, instruction)                                                                 \
    __global__ void kernel_name(uint16_t *results, uint32_t first_index)                                              \
    {                                                                                                                 \
        uint32_t index = blockIdx.x * blockDim.x + threadIdx.x, converted = 0;                                        \
        float first = __uint_as_float(UNIT_FP32_PATTERN(first_index + index));                                        \
        float second = __uint_as_float(UNIT_FP32_PATTERN(first_index + index) + UNIT_PARTNER_STEP);                   \
        asm volatile("{ .reg .b16 packed;\n" instruction " packed, %1, %2;\n" "cvt.u32.u16 %0, packed; }\n"           \
                     : "=r"(converted)                                                                                \
                     : "f"(first), "f"(second));                                                                      \
        results[index] = (uint16_t)converted;                                                                         \
    }

UNIT_FLOAT_FORM(unit_kernel_f16, "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32")
UNIT_FLOAT_FORM(unit_kernel_bf16, "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32")
UNIT_WHOLE_FORM(unit_kernel_s8, "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32")
UNIT_WHOLE_FORM(unit_kernel_u8_s8, "mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32")
UNIT_WHOLE_FORM(unit_kernel_s8_saturating, "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32")
UNIT_WHOLE_FORM(unit_kernel_u4_s4, "mma.sync.aligned.m16n8k64.row.col.s32.u4.s4.s32")
UNIT_WHOLE_FORM(unit_kernel_u4, "mma.sync.aligned.m16n8k64.row.col.s32.u4.u4.s32")
UNIT_WHOLE_FORM(unit_kernel_s4, "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32")
#if PROBE_SM == UNIT_SM_120A
#define UNIT_SCALED_KIND "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32"
UNIT_FLOAT_FORM(unit_kernel_e2m1_e4m3, "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e2m1.e4m3.f32")
UNIT_FLOAT_FORM(unit_kernel_e2m3_e4m3, "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e2m3.e4m3.f32")
UNIT_FLOAT_FORM(unit_kernel_e4m3, "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32")
UNIT_FLOAT_KERNEL(unit_kernel_e4m3_k16, "mma.sync.aligned.m16n8k16.row.col.f32.e4m3.e4m3.f32", A2_B1, "", NO_SCALES)
UNIT_EVERY_SELECTOR(UNIT_SCALED_FORM, e2m1_e4m3_scaled, UNIT_SCALED_KIND ".e2m1.e4m3.f32.ue8m0")
UNIT_EVERY_SELECTOR(UNIT_SCALED_FORM, e4m3_scaled, UNIT_SCALED_KIND ".e4m3.e4m3.f32.ue8m0")
UNIT_CONVERT_KERNEL(unit_convert_e4m3, "cvt.rn.satfinite.e4m3x2.f32")
UNIT_CONVERT_KERNEL(unit_convert_ue8m0, "cvt.rp.satfinite.ue8m0x2.f32")
#endif

#endif /* MMA_UNIT_KERNELS_CUH */
