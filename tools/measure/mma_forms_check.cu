/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * mma_forms_check: which tensor-core instruction forms, and which of the conversions, loads and small helpers that
 * feed them, the compiler accepts for each card architecture, and which machine instruction each one becomes.
 *
 * The routed experts are products of small-format weights and quantized inputs. The probes so far compute them with
 * one form, m16n8k16 on 16-bit floats (16 inputs an instruction), after turning both operands into 16-bit floats with
 * integer instructions. research/speed/instruction_forms.md lists every other form the two architectures document
 * (PTX ISA 9.4, the toolkit's own) and which weight format each could serve. This file is the compile-only check of
 * that list: one kernel per form, each under its own preprocessor symbol, each holding ONE instruction under test in
 * inline PTX (the check of the feature symbol holds none). mma_forms_check.sh compiles every form on its own for
 * every architecture and reports, per form and architecture, "compiles" or the compiler's message, and for those that
 * compile the mnemonics the disassembler shows: a form that becomes one tensor-core instruction is told from one the
 * compiler expands or refuses.
 *
 * What a pass proves: the spelling is accepted for that target and lowers to the instructions listed.
 * What it does not prove: that the instruction computes what the documentation says for the operands a kernel will
 * give it (which lane holds which element, how a scale byte is selected, what a subnormal operand does, in what
 * order and precision the products are added), nor anything about its speed. Nothing here runs: no kernel is
 * launched and no card is touched. Those questions belong to the probes (the Experiments table of the note).
 *
 * A kernel reads one record per lane, executes the instruction once and stores what it returns in the lane's result
 * record, so that nothing is eliminated. The script defines exactly one CHECK_ symbol per compilation (with two,
 * the first in this file would be compiled and nothing would say so). The comment beside
 * each says which targets the documentation names for the form, in words the script reads:
 *   all                   every architecture from sm_80 on
 *   from_N                architectures numbered N or higher (from_89: sm_120, sm_120f and sm_120a, not sm_80)
 *   sm_120a sm_120f ...   these targets by name
 *   sm_120a?              the documentation names the family target sm_120f and is silent on sm_120a
 *   unstated              the documentation's grammar neither lists nor excludes this spelling
 *   none                  no architecture of this machine
 * The spellings: integer, half-precision and FP8 forms as the documentation's grammar has them; the block-scaled
 * kinds with the kind before the shape, as the owner compiled two of them in August, and twice more with the kind
 * after the layout, as the documentation prints it; the kind without scales the documentation's way, and once with
 * the kind first and once with no kind at all.
 *
 *   nvcc -arch=sm_80|sm_120a -cubin -DCHECK_<FORM> -o <form>.cubin mma_forms_check.cu      (the script does this)
 *
 * Measurement tool: not part of the engine.
 */
#include <cuda_runtime.h>
#include <stdint.h>

enum check_constant {
    /* a lane's record: its words are multiplicand A (up to four), multiplicand B (up to two), the two scale words of
     * a block-scaled kind, then an accumulator of whole numbers or of packed halves (up to four), then two words a
     * kernel may take as run-time selectors; its fp32 and fp64 values are accumulators and, for fp64, operands; a
     * pair is two fp32 values in one 64-bit word */
    RECORD_WORDS = 16, RECORD_SINGLES = 8, RECORD_DOUBLES = 8, RECORD_PAIRS = 4,
    A_AT = 0, B_AT = 4, SCALE_A_AT = 6, SCALE_B_AT = 7, C_AT = 8, BYTE_SELECTOR_AT = 12, THREAD_SELECTOR_AT = 13,
    A_DOUBLE_AT = 0, B_DOUBLE_AT = 2, C_DOUBLE_AT = 4,
    /* the matrix loads read shared memory: 16 bytes a lane, which is one row of a 16 by 16 matrix of bytes */
    SHARED_ROW_WORDS = 4, LANES_PER_WARP = 32, SHARED_TILE_WORDS = LANES_PER_WARP * SHARED_ROW_WORDS
};

struct lane_record {
    uint32_t words[RECORD_WORDS];
    float singles[RECORD_SINGLES];
    double doubles[RECORD_DOUBLES];
    uint64_t pairs[RECORD_PAIRS];
};

/* One kernel: a lane's operands from its record, the PTX text once, what it returns into the lane's result record. */
#define FORM_KERNEL(ptx_text, destinations, sources)                                                                  \
    __global__ void form_kernel(const struct lane_record *lanes, struct lane_record *results)                         \
    {                                                                                                                 \
        const struct lane_record *lane = &lanes[threadIdx.x];                                                         \
        struct lane_record *result = &results[threadIdx.x];                                                           \
        asm volatile(ptx_text : destinations : sources);                                                              \
    }

/* The operand lists of an instruction, by what a lane holds of D, A, B and C. */
#define A_WORDS_1 "r"(lane->words[A_AT])
#define A_WORDS_2 A_WORDS_1, "r"(lane->words[A_AT + 1])
#define A_WORDS_4 A_WORDS_2, "r"(lane->words[A_AT + 2]), "r"(lane->words[A_AT + 3])
#define B_WORDS_1 "r"(lane->words[B_AT])
#define B_WORDS_2 B_WORDS_1, "r"(lane->words[B_AT + 1])
#define C_WORDS_2 "r"(lane->words[C_AT]), "r"(lane->words[C_AT + 1])
#define C_WORDS_4 C_WORDS_2, "r"(lane->words[C_AT + 2]), "r"(lane->words[C_AT + 3])
#define D_WORDS_2 "=r"(result->words[0]), "=r"(result->words[1])
#define D_WORDS_4 D_WORDS_2, "=r"(result->words[2]), "=r"(result->words[3])
#define C_SINGLES_4 "f"(lane->singles[0]), "f"(lane->singles[1]), "f"(lane->singles[2]), "f"(lane->singles[3])
#define C_SINGLES_8                                                                                                   \
    C_SINGLES_4, "f"(lane->singles[4]), "f"(lane->singles[5]), "f"(lane->singles[6]), "f"(lane->singles[7])
#define D_SINGLES_4                                                                                                   \
    "=f"(result->singles[0]), "=f"(result->singles[1]), "=f"(result->singles[2]), "=f"(result->singles[3])
#define D_SINGLES_8                                                                                                   \
    D_SINGLES_4, "=f"(result->singles[4]), "=f"(result->singles[5]), "=f"(result->singles[6]), "=f"(result->singles[7])
#define A_DOUBLES_1 "d"(lane->doubles[A_DOUBLE_AT])
#define A_DOUBLES_2 A_DOUBLES_1, "d"(lane->doubles[A_DOUBLE_AT + 1])
#define B_DOUBLES_1 "d"(lane->doubles[B_DOUBLE_AT])
#define C_DOUBLES_2 "d"(lane->doubles[C_DOUBLE_AT]), "d"(lane->doubles[C_DOUBLE_AT + 1])
#define C_DOUBLES_4 C_DOUBLES_2, "d"(lane->doubles[C_DOUBLE_AT + 2]), "d"(lane->doubles[C_DOUBLE_AT + 3])
#define D_DOUBLES_2 "=d"(result->doubles[0]), "=d"(result->doubles[1])
#define D_DOUBLES_4 D_DOUBLES_2, "=d"(result->doubles[2]), "=d"(result->doubles[3])
#define SCALE_WORDS "r"(lane->words[SCALE_A_AT]), "r"(lane->words[SCALE_B_AT])
#define SELECTOR_WORDS "r"(lane->words[BYTE_SELECTOR_AT]), "r"(lane->words[THREAD_SELECTOR_AT])

/* The operand text of an instruction: D, A, B, C in braces, numbered as the lists above follow one another. */
#define TEXT_D4_A4_B2 " {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}"
#define TEXT_D4_A2_B1 " {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10}"
#define TEXT_D2_A4_B2 " {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%9}"
#define TEXT_D2_A2_B1 " {%0,%1}, {%2,%3}, {%4}, {%5,%6}"
#define TEXT_D2_A1_B1 " {%0,%1}, {%2}, {%3}, {%4,%5}"
#define TEXT_D8_A2_B2 " {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%12,%13,%14,%15,%16,%17,%18,%19}"

/* fp32 sums: sixteen outputs by eight columns, two outputs by two columns a lane */
#define SINGLE_SUMS_A4_B2(instruction)                                                                                \
    FORM_KERNEL(instruction TEXT_D4_A4_B2 ";\n", D_SINGLES_4, A_WORDS_4 COMMA B_WORDS_2 COMMA C_SINGLES_4)
#define SINGLE_SUMS_A2_B1(instruction)                                                                                \
    FORM_KERNEL(instruction TEXT_D4_A2_B1 ";\n", D_SINGLES_4, A_WORDS_2 COMMA B_WORDS_1 COMMA C_SINGLES_4)
/* the shape of the first tensor cores: four products of eight by eight in a warp, eight fp32 sums a lane */
#define SINGLE_SUMS_EIGHT_A2_B2(instruction)                                                                          \
    FORM_KERNEL(instruction TEXT_D8_A2_B2 ";\n", D_SINGLES_8, A_WORDS_2 COMMA B_WORDS_2 COMMA C_SINGLES_8)
/* sums in words: four whole numbers a lane, or two words of two packed fp16 sums each */
#define WORD_SUMS_4_A4_B2(instruction)                                                                                \
    FORM_KERNEL(instruction TEXT_D4_A4_B2 ";\n", D_WORDS_4, A_WORDS_4 COMMA B_WORDS_2 COMMA C_WORDS_4)
#define WORD_SUMS_4_A2_B1(instruction)                                                                                \
    FORM_KERNEL(instruction TEXT_D4_A2_B1 ";\n", D_WORDS_4, A_WORDS_2 COMMA B_WORDS_1 COMMA C_WORDS_4)
#define WORD_SUMS_2_A4_B2(instruction)                                                                                \
    FORM_KERNEL(instruction TEXT_D2_A4_B2 ";\n", D_WORDS_2, A_WORDS_4 COMMA B_WORDS_2 COMMA C_WORDS_2)
#define WORD_SUMS_2_A2_B1(instruction)                                                                                \
    FORM_KERNEL(instruction TEXT_D2_A2_B1 ";\n", D_WORDS_2, A_WORDS_2 COMMA B_WORDS_1 COMMA C_WORDS_2)
#define WORD_SUMS_2_A1_B1(instruction)                                                                                \
    FORM_KERNEL(instruction TEXT_D2_A1_B1 ";\n", D_WORDS_2, A_WORDS_1 COMMA B_WORDS_1 COMMA C_WORDS_2)
/* fp64 sums: the one kind whose rounding the documentation specifies */
#define DOUBLE_SUMS_2_A1_B1(instruction)                                                                              \
    FORM_KERNEL(instruction TEXT_D2_A1_B1 ";\n", D_DOUBLES_2, A_DOUBLES_1 COMMA B_DOUBLES_1 COMMA C_DOUBLES_2)
#define DOUBLE_SUMS_4_A2_B1(instruction)                                                                              \
    FORM_KERNEL(instruction TEXT_D4_A2_B1 ";\n", D_DOUBLES_4, A_DOUBLES_2 COMMA B_DOUBLES_1 COMMA C_DOUBLES_4)
/* a block-scaled kind: after C the scale word of A with its {byte, thread pair} selector, then B's with its
 * {byte, thread} selector; the selectors are written into the instruction */
#define BLOCK_SCALED(instruction, selector_a, selector_b)                                                             \
    FORM_KERNEL(instruction TEXT_D4_A4_B2 ", %14, " selector_a ", %15, " selector_b ";\n", D_SINGLES_4,               \
                A_WORDS_4 COMMA B_WORDS_2 COMMA C_SINGLES_4 COMMA SCALE_WORDS)
/* the same with the selectors in two 16-bit registers, each used for both operands, that the lane fills from its
 * record: are run-time selectors accepted, and if they are, as one instruction? */
#define BLOCK_SCALED_REGISTER_SELECTORS(instruction)                                                                  \
    FORM_KERNEL("{ .reg .u16 byte_selector, thread_selector;\n"                                                       \
                "cvt.u16.u32 byte_selector, %16;\n"                                                                   \
                "cvt.u16.u32 thread_selector, %17;\n"                                                                 \
                instruction TEXT_D4_A4_B2                                                                             \
                ", %14, {byte_selector, thread_selector}, %15, {byte_selector, thread_selector}; }\n",                \
                D_SINGLES_4, A_WORDS_4 COMMA B_WORDS_2 COMMA C_SINGLES_4 COMMA SCALE_WORDS COMMA SELECTOR_WORDS)
#define COMMA ,

/* A matrix load: the lane gives the address of its row of the block's shared tile. */
#define MATRIX_LOAD(instruction, destination_text, destinations)                                                      \
    __global__ void form_kernel(const struct lane_record *lanes, struct lane_record *results)                         \
    {                                                                                                                 \
        __shared__ uint32_t shared_tile[SHARED_TILE_WORDS];                                                           \
        struct lane_record *result = &results[threadIdx.x];                                                           \
        uint64_t row_address = (uint64_t)&shared_tile[(threadIdx.x % LANES_PER_WARP) * SHARED_ROW_WORDS];             \
        shared_tile[threadIdx.x % SHARED_TILE_WORDS] = lanes[threadIdx.x].words[A_AT];                                \
        __syncthreads();                                                                                              \
        asm volatile("{ .reg .u64 shared_address;\n"                                                                  \
                     "cvta.to.shared.u64 shared_address, %" destination_text ";\n"                                    \
                     instruction ", [shared_address]; }\n"                                                            \
                     : destinations                                                                                   \
                     : "l"(row_address));                                                                             \
    }

/* ---- half precision, bfloat16 and tf32 operands: sm_80 has them all ---- */
#if defined(CHECK_F16_M16N8K16_SINGLE_SUMS)                  /* documented for: all (the form the probes use) */
SINGLE_SUMS_A4_B2("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32")
#elif defined(CHECK_BF16_M16N8K16_SINGLE_SUMS)               /* documented for: all */
SINGLE_SUMS_A4_B2("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32")
#elif defined(CHECK_F16_M16N8K8_SINGLE_SUMS)                 /* documented for: all */
SINGLE_SUMS_A2_B1("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32")
#elif defined(CHECK_BF16_M16N8K8_SINGLE_SUMS)                /* documented for: all */
SINGLE_SUMS_A2_B1("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32")
#elif defined(CHECK_TF32_M16N8K8_SINGLE_SUMS)                /* documented for: all */
SINGLE_SUMS_A4_B2("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32")
#elif defined(CHECK_TF32_M16N8K4_SINGLE_SUMS)                /* documented for: all */
SINGLE_SUMS_A2_B1("mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32")
#elif defined(CHECK_F16_M16N8K16_HALF_SUMS)                  /* documented for: all */
WORD_SUMS_2_A4_B2("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16")
#elif defined(CHECK_F16_M16N8K8_HALF_SUMS)                   /* documented for: all */
WORD_SUMS_2_A2_B1("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16")
#elif defined(CHECK_F16_M8N8K4_SINGLE_SUMS)                  /* documented for: all (made for sm_70, slow elsewhere) */
SINGLE_SUMS_EIGHT_A2_B2("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32")

/* ---- whole numbers of 8 bits, of 4 bits, and single bits: sm_80 has them all ---- */
#elif defined(CHECK_S8_M16N8K32)                             /* documented for: all */
WORD_SUMS_4_A4_B2("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32")
#elif defined(CHECK_U8_S8_M16N8K32)                          /* documented for: all */
WORD_SUMS_4_A4_B2("mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32")
#elif defined(CHECK_S8_M16N8K32_SATURATING)                  /* documented for: all */
WORD_SUMS_4_A4_B2("mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32")
#elif defined(CHECK_S8_M16N8K16)                             /* documented for: all */
WORD_SUMS_4_A2_B1("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32")
#elif defined(CHECK_S8_M8N8K16)                              /* documented for: all */
WORD_SUMS_2_A1_B1("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32")
#elif defined(CHECK_S4_M16N8K64)                             /* documented for: all */
WORD_SUMS_4_A4_B2("mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32")
#elif defined(CHECK_U4_M16N8K64)                             /* documented for: all */
WORD_SUMS_4_A4_B2("mma.sync.aligned.m16n8k64.row.col.s32.u4.u4.s32")
#elif defined(CHECK_U4_S4_M16N8K64)                          /* documented for: all (its type table says both alike) */
WORD_SUMS_4_A4_B2("mma.sync.aligned.m16n8k64.row.col.s32.u4.s4.s32")
#elif defined(CHECK_S4_M16N8K32)                             /* documented for: all */
WORD_SUMS_4_A2_B1("mma.sync.aligned.m16n8k32.row.col.s32.s4.s4.s32")
#elif defined(CHECK_U4_S4_M16N8K32)                          /* documented for: all (its type table says both alike) */
WORD_SUMS_4_A2_B1("mma.sync.aligned.m16n8k32.row.col.s32.u4.s4.s32")
#elif defined(CHECK_S4_M8N8K32)                              /* documented for: all */
WORD_SUMS_2_A1_B1("mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32")
#elif defined(CHECK_B1_M16N8K256_AND_POPCOUNT)               /* documented for: all */
WORD_SUMS_4_A4_B2("mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc")
#elif defined(CHECK_B1_M16N8K256_XOR_POPCOUNT)               /* documented for: all */
WORD_SUMS_4_A4_B2("mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.xor.popc")
#elif defined(CHECK_B1_M16N8K128_AND_POPCOUNT)               /* documented for: all */
WORD_SUMS_4_A2_B1("mma.sync.aligned.m16n8k128.row.col.s32.b1.b1.s32.and.popc")
#elif defined(CHECK_B1_M8N8K128_XOR_POPCOUNT)                /* documented for: all */
WORD_SUMS_2_A1_B1("mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.xor.popc")

/* ---- double precision ---- */
#elif defined(CHECK_F64_M8N8K4)                              /* documented for: all */
DOUBLE_SUMS_2_A1_B1("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64")
#elif defined(CHECK_F64_M16N8K4)                             /* documented for: from_90 */
DOUBLE_SUMS_4_A2_B1("mma.sync.aligned.m16n8k4.row.col.f64.f64.f64.f64.rn")

/* ---- 8-bit floats without scales: from sm_89 ---- */
#elif defined(CHECK_E4M3_M16N8K32_SINGLE_SUMS)               /* documented for: from_89 */
SINGLE_SUMS_A4_B2("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32")
#elif defined(CHECK_E4M3_M16N8K16_SINGLE_SUMS)               /* documented for: from_89 */
SINGLE_SUMS_A2_B1("mma.sync.aligned.m16n8k16.row.col.f32.e4m3.e4m3.f32")
#elif defined(CHECK_E4M3_E5M2_M16N8K32_SINGLE_SUMS)          /* documented for: from_89 */
SINGLE_SUMS_A4_B2("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e5m2.f32")
#elif defined(CHECK_E4M3_M16N8K32_HALF_SUMS)                 /* documented for: from_89 */
WORD_SUMS_2_A4_B2("mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16")

/* ---- 8-, 6- and 4-bit floats mixed, each element in a byte, without scales: kind f8f6f4 ---- */
#elif defined(CHECK_E2M1_E4M3_M16N8K32_SINGLE_SUMS)          /* documented for: sm_120a sm_120f */
SINGLE_SUMS_A4_B2("mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e2m1.e4m3.f32")
#elif defined(CHECK_E2M1_E4M3_M16N8K32_KIND_FIRST)           /* documented for: sm_120a sm_120f */
SINGLE_SUMS_A4_B2("mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e2m1.e4m3.f32")
#elif defined(CHECK_E2M1_E4M3_M16N8K32_NO_KIND)              /* documented for: unstated */
SINGLE_SUMS_A4_B2("mma.sync.aligned.m16n8k32.row.col.f32.e2m1.e4m3.f32")
#elif defined(CHECK_E2M3_E4M3_M16N8K32_SINGLE_SUMS)          /* documented for: sm_120a sm_120f */
SINGLE_SUMS_A4_B2("mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e2m3.e4m3.f32")
#elif defined(CHECK_E3M2_E4M3_M16N8K32_SINGLE_SUMS)          /* documented for: sm_120a sm_120f */
SINGLE_SUMS_A4_B2("mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e3m2.e4m3.f32")
#elif defined(CHECK_E2M1_E4M3_M16N8K32_HALF_SUMS)            /* documented for: sm_120a sm_120f */
WORD_SUMS_2_A4_B2("mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f16.e2m1.e4m3.f16")

/* ---- the same with a power-of-two scale per output and per column inside the instruction: kind mxf8f6f4 ---- */
#elif defined(CHECK_SCALED_E2M1_E4M3_M16N8K32)               /* documented for: sm_120a sm_120f (the native product) */
BLOCK_SCALED("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e2m1.e4m3.f32.ue8m0",
             "{0, 0}", "{0, 0}")
#elif defined(CHECK_SCALED_E2M1_E4M3_M16N8K32_SELECTORS)     /* documented for: sm_120a sm_120f */
BLOCK_SCALED("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e2m1.e4m3.f32.ue8m0",
             "{2, 1}", "{1, 3}")
#elif defined(CHECK_SCALED_E2M1_E4M3_M16N8K32_KIND_AFTER)    /* documented for: sm_120a sm_120f */
BLOCK_SCALED("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e2m1.e4m3.f32.ue8m0",
             "{0, 0}", "{0, 0}")
#elif defined(CHECK_SCALED_E2M1_E4M3_M16N8K32_REGISTERS)     /* documented for: unstated */
BLOCK_SCALED_REGISTER_SELECTORS(
    "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e2m1.e4m3.f32.ue8m0")
#elif defined(CHECK_SCALED_E4M3_E2M1_M16N8K32)               /* documented for: sm_120a sm_120f */
BLOCK_SCALED("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e2m1.f32.ue8m0",
             "{0, 0}", "{0, 0}")
#elif defined(CHECK_SCALED_E4M3_E4M3_M16N8K32)               /* documented for: sm_120a sm_120f (the 8-bit trunk) */
BLOCK_SCALED("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0",
             "{0, 0}", "{0, 0}")
#elif defined(CHECK_SCALED_E2M3_E4M3_M16N8K32)               /* documented for: sm_120a sm_120f */
BLOCK_SCALED("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e2m3.e4m3.f32.ue8m0",
             "{0, 0}", "{0, 0}")

/* ---- 4-bit floats packed as nibbles, scales inside: kinds mxf4 (one per 32) and mxf4nvf4 (one per 32 or 16) ---- */
#elif defined(CHECK_SCALED_E2M1_M16N8K64_MXF4)               /* documented for: sm_120a sm_120f (compiled in August) */
BLOCK_SCALED("mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0",
             "{0, 0}", "{0, 0}")
/* differs from the form above in three things at once (where the kind stands, no scale_vec, the selectors): a
 * refusal of it does not say which */
#elif defined(CHECK_SCALED_E2M1_M16N8K64_MXF4_KIND_AFTER)    /* documented for: sm_120a sm_120f */
BLOCK_SCALED("mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.f32.e2m1.e2m1.f32.ue8m0",
             "{2, 1}", "{2, 3}")
#elif defined(CHECK_SCALED_E2M1_M16N8K64_NVF4_UE4M3)         /* documented for: sm_120a sm_120f (compiled in August) */
BLOCK_SCALED("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3",
             "{0, 0}", "{0, 0}")
#elif defined(CHECK_SCALED_E2M1_M16N8K64_NVF4_UE8M0_PER_32)  /* documented for: sm_120a sm_120f */
BLOCK_SCALED("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0",
             "{0, 0}", "{0, 0}")
#elif defined(CHECK_SCALED_E2M1_M16N8K64_NVF4_UE8M0_PER_16)  /* documented for: sm_120a sm_120f (since PTX ISA 9.1) */
BLOCK_SCALED("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0",
             "{0, 0}", "{0, 0}")

/* ---- conversions: the activation quantization and the decodes as single instructions ---- */
#elif defined(CHECK_CONVERT_SINGLES_TO_E4M3)                 /* documented for: from_89 */
FORM_KERNEL("{ .reg .b16 packed;\n"
            "cvt.rn.satfinite.e4m3x2.f32 packed, %1, %2;\n"
            "cvt.u32.u16 %0, packed; }\n",
            "=r"(result->words[0]), "f"(lane->singles[0]) COMMA "f"(lane->singles[1]))
#elif defined(CHECK_CONVERT_E4M3_TO_F16)                     /* documented for: from_89 */
FORM_KERNEL("{ .reg .b16 low_pair, high_pair;\n"
            "mov.b32 {low_pair, high_pair}, %1;\n"
            "cvt.rn.f16x2.e4m3x2 %0, low_pair; }\n",
            "=r"(result->words[0]), A_WORDS_1)
#elif defined(CHECK_CONVERT_E4M3_TO_BF16_SCALED)             /* documented for: sm_100f sm_110f sm_120f sm_120a? */
FORM_KERNEL("{ .reg .b16 low_pair, high_pair, low_scales, high_scales;\n"
            "mov.b32 {low_pair, high_pair}, %1;\n"
            "mov.b32 {low_scales, high_scales}, %2;\n"
            "cvt.rn.scaled::n2::ue8m0.bf16x2.e4m3x2 %0, low_pair, low_scales; }\n",
            "=r"(result->words[0]), A_WORDS_1 COMMA "r"(lane->words[SCALE_A_AT]))
#elif defined(CHECK_CONVERT_SINGLES_TO_E2M1)                 /* documented for: sm_100a sm_100f sm_120a sm_120f */
FORM_KERNEL("{ .reg .b8 packed;\n"
            "cvt.rn.satfinite.e2m1x2.f32 packed, %1, %2;\n"
            "cvt.u32.u8 %0, packed; }\n",
            "=r"(result->words[0]), "f"(lane->singles[0]) COMMA "f"(lane->singles[1]))
#elif defined(CHECK_CONVERT_E2M1_TO_F16)                     /* documented for: sm_100a sm_100f sm_120a sm_120f */
FORM_KERNEL("{ .reg .b8 byte_0, byte_1, byte_2, byte_3;\n"
            "mov.b32 {byte_0, byte_1, byte_2, byte_3}, %1;\n"
            "cvt.rn.f16x2.e2m1x2 %0, byte_0; }\n",
            "=r"(result->words[0]), A_WORDS_1)
#elif defined(CHECK_CONVERT_SINGLES_TO_UE8M0_UPWARD)         /* documented for: sm_100a sm_100f sm_120a sm_120f */
FORM_KERNEL("{ .reg .b16 packed;\n"
            "cvt.rp.satfinite.ue8m0x2.f32 packed, %1, %2;\n"
            "cvt.u32.u16 %0, packed; }\n",
            "=r"(result->words[0]), "f"(lane->singles[0]) COMMA "f"(lane->singles[1]))
#elif defined(CHECK_CONVERT_UE8M0_TO_BF16)                   /* documented for: sm_100a sm_100f sm_120a sm_120f */
FORM_KERNEL("{ .reg .b16 low_pair, high_pair;\n"
            "mov.b32 {low_pair, high_pair}, %1;\n"
            "cvt.rn.bf16x2.ue8m0x2 %0, low_pair; }\n",
            "=r"(result->words[0]), A_WORDS_1)

/* ---- small helpers of a decode or of the sums around an instruction ---- */
#elif defined(CHECK_PERMUTE_BYTES)                           /* documented for: all (a table of 8 bytes, 4 lookups) */
FORM_KERNEL("prmt.b32 %0, %1, %2, %3;\n", "=r"(result->words[0]), A_WORDS_2 COMMA B_WORDS_1)
#elif defined(CHECK_DOT_PRODUCT_OF_FOUR_BYTES)               /* documented for: all */
FORM_KERNEL("dp4a.s32.s32 %0, %1, %2, %3;\n", "=r"(result->words[0]),
            A_WORDS_1 COMMA B_WORDS_1 COMMA "r"(lane->words[C_AT]))
#elif defined(CHECK_LOGIC_OF_THREE_INPUTS)                   /* documented for: all (0xE4: A where C is set, else B) */
FORM_KERNEL("lop3.b32 %0, %1, %2, %3, 0xE4;\n", "=r"(result->words[0]),
            A_WORDS_1 COMMA B_WORDS_1 COMMA "r"(lane->words[C_AT]))
#elif defined(CHECK_FUSED_MULTIPLY_ADD_OF_TWO_SINGLES)       /* documented for: from_100 */
FORM_KERNEL("fma.rn.f32x2 %0, %1, %2, %3;\n", "=l"(result->pairs[0]),
            "l"(lane->pairs[0]) COMMA "l"(lane->pairs[1]) COMMA "l"(lane->pairs[RECORD_PAIRS - 1]))

/* ---- matrix loads from shared memory ---- */
#elif defined(CHECK_LOAD_MATRIX_OF_HALVES)                   /* documented for: all */
MATRIX_LOAD("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}", "4", D_WORDS_4)
#elif defined(CHECK_LOAD_MATRIX_OF_BYTES)                    /* documented for: sm_100a sm_100f sm_120a sm_120f */
MATRIX_LOAD("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0,%1,%2,%3}", "4", D_WORDS_4)
#elif defined(CHECK_LOAD_MATRIX_NIBBLES_TO_BYTES)            /* documented for: sm_100a sm_100f sm_120a sm_120f */
MATRIX_LOAD("ldmatrix.sync.aligned.m16n16.x1.trans.shared.b8x16.b4x16_p64 {%0,%1}", "2", D_WORDS_2)
#elif defined(CHECK_LOAD_MATRIX_SIGNED_NIBBLES_TO_BYTES)     /* documented for: sm_90a sm_100f sm_120f sm_120a? */
MATRIX_LOAD("ldmatrix.sync.aligned.m8n16.x1.shared.s8.s4 {%0}", "1", "=r"(result->words[0]))

/* ---- the two other tensor-core families: neither card should have them ---- */
#elif defined(CHECK_WARPGROUP_FAMILY)                        /* documented for: sm_90a */
__global__ void form_kernel(const struct lane_record *lanes, struct lane_record *results)
{
    asm volatile("wgmma.fence.sync.aligned;\n");
    results[threadIdx.x].words[0] = lanes[threadIdx.x].words[A_AT];
}
#elif defined(CHECK_FIFTH_GENERATION_FAMILY)                 /* documented for: sm_100a sm_100f sm_103a sm_103f */
__global__ void form_kernel(const struct lane_record *lanes, struct lane_record *results)
{
    asm volatile("tcgen05.fence::before_thread_sync;\n");
    results[threadIdx.x].words[0] = lanes[threadIdx.x].words[A_AT];
}

/* ---- the symbol that tells a build for sm_120a from one for the generic sm_120: kernel sources guard with it ---- */
#elif defined(CHECK_FEATURE_SYMBOL_OF_SM_120A)               /* documented for: unstated */
#if defined(__CUDA_ARCH__) && !defined(__CUDA_ARCH_FEAT_SM120_ALL)
#error "__CUDA_ARCH_FEAT_SM120_ALL is not defined for this target"
#endif
__global__ void form_kernel(const struct lane_record *lanes, struct lane_record *results)
{
    results[threadIdx.x].words[0] = lanes[threadIdx.x].words[A_AT];
}

#else
#error "define exactly one CHECK_ symbol that this file knows (mma_forms_check.sh reads their names from this file)"
#endif
