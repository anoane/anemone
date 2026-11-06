/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * attention_forms_check: which of the forms the attention units may want the compiler accepts for each card
 * architecture, and what each becomes: package 11a of round 1 of the speed campaign
 * (research/speed/ROUND_1_attention.md, "The cards": none of these is in a row of h01). In the layout of
 * mma_forms_check.cu: one kernel per form, each under its own preprocessor symbol and holding the one instruction
 * under test in inline PTX; batch_i11a_index_scan.sh compiles every form alone for every architecture (the symbols
 * and the words beside them are read from this file) and prints "compiles" with the disassembly's mnemonics, or the
 * compiler's message. Its result for SCALED_E2M1_E2M1_M16N8K32 on sm_120a sets INDEX_E2M1_SCALED for
 * index_scan_probe: the block-scaled E2M1 form of the TRAINED keys.
 *
 * What a pass proves: the spelling is accepted for that target and lowers to the instructions listed. Not what an
 * instruction computes, nor its speed; nothing here runs, no kernel is launched and no card is touched. The comment
 * beside a symbol says which targets the documentation (PTX ISA 9.4) names for the form, in the words of
 * mma_forms_check.cu: all, from_N, target names, sm_120a? (named for the family target only), unstated. The
 * dependent launch is its two device instructions and, as CHECK_HOST_ form, its launch attribute: host code, which a
 * module does not hold, so a CHECK_HOST_ form is compiled as an object of a whole program (-c), never linked or run.
 *
 *   nvcc -arch=sm_80|sm_120a -cubin -DCHECK_<FORM> -o <form>.cubin attention_forms_check.cu      (the batch does this)
 *   nvcc -arch=sm_80 (or -gencode arch=compute_120a,code=sm_120a) -c -DCHECK_HOST_<FORM> -o <form>.o ...
 *
 * Measurement tool: not part of the engine.
 */
#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

enum check_constant {
    /* a lane's record: multiplicand A (four words), B (two), the two scale words, an accumulator of four floats; the
     * shared tile a copy fills: 16 bytes a lane; the transaction barrier's 8 bytes behind it */
    RECORD_WORDS = 8, RECORD_SINGLES = 4, A_AT = 0, B_AT = 4, SCALE_A_AT = 6, SCALE_B_AT = 7, LANES_PER_WARP = 32,
    COPY_BYTES = 16, COPY_WORDS = 4, SHARED_WORDS = LANES_PER_WARP * COPY_WORDS
};
struct lane_record {
    uint32_t words[RECORD_WORDS];
    float singles[RECORD_SINGLES];
};

/* One kernel: the PTX text once, with the lane's record in and its result record out. */
#define FORM_KERNEL(body)                                                                                            \
    __global__ void form_kernel(const struct lane_record *lanes, struct lane_record *results)                        \
    {                                                                                                                \
        const struct lane_record *lane = &lanes[threadIdx.x];                                                        \
        struct lane_record *result = &results[threadIdx.x];                                                          \
        body                                                                                                         \
    }
/* A form that fills shared memory from the lane's record, then hands the copied words back. */
#define COPY_KERNEL(ptx_text, ...)                                                                                   \
    FORM_KERNEL(__shared__ __align__(COPY_BYTES) uint32_t tile[SHARED_WORDS];                                        \
                __shared__ __align__(sizeof(uint64_t)) uint64_t barrier;                                             \
                const uint32_t tile_at = (uint32_t)__cvta_generic_to_shared(&tile[threadIdx.x * COPY_WORDS]);        \
                const uint32_t barrier_at = (uint32_t)__cvta_generic_to_shared(&barrier);                            \
                asm volatile(ptx_text :: "r"(tile_at), "l"(lane->words), "r"(barrier_at) : "memory");               \
                __syncthreads();                                                                                     \
                result->words[0] = tile[threadIdx.x * COPY_WORDS] + (uint32_t)barrier;)

#if defined(CHECK_ASYNC_COPY_16)                             /* documented for: all (cp.async, 16 bytes a lane) */
COPY_KERNEL("cp.async.cg.shared.global [%0], [%1], 16;\n"
            "cp.async.commit_group;\n"
            "cp.async.wait_group 0;\n")
#elif defined(CHECK_BULK_COPY_TRANSACTION)                   /* documented for: from_90 (the transaction barrier) */
COPY_KERNEL("{ .reg .pred done;\n"
            "mbarrier.init.shared::cta.b64 [%2], 1;\n"
            "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%2], 16;\n"
            "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], 16, [%2];\n"
            "waiting:\n"
            "mbarrier.try_wait.parity.shared::cta.b64 done, [%2], 0;\n"
            "@!done bra waiting; }\n")
#elif defined(CHECK_SET_MAX_REGISTERS)                       /* documented for: sm_90a sm_100a sm_120a? */
FORM_KERNEL(asm volatile("setmaxnreg.inc.sync.aligned.u32 240;\n");
            result->words[0] = lane->words[A_AT];)
#elif defined(CHECK_MATCH_ANY)                               /* documented for: all (from sm_70) */
FORM_KERNEL(asm volatile("match.any.sync.b32 %0, %1, 0xffffffff;\n" : "=r"(result->words[0]) : "r"(lane->words[A_AT]));)
#elif defined(CHECK_LOAD_256_BITS)                           /* documented for: from_100 (.v8 of 32-bit words) */
FORM_KERNEL(asm volatile("ld.global.v8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];\n"
                         : "=r"(result->words[0]), "=r"(result->words[1]), "=r"(result->words[2]),
                           "=r"(result->words[3]), "=r"(result->words[4]), "=r"(result->words[5]),
                           "=r"(result->words[6]), "=r"(result->words[7])
                         : "l"(lane->words));)
#elif defined(CHECK_DEPENDENT_LAUNCH)                        /* documented for: from_90 (griddepcontrol) */
FORM_KERNEL(asm volatile("griddepcontrol.wait;\n" ::: "memory");
            result->words[0] = lane->words[A_AT];
            asm volatile("griddepcontrol.launch_dependents;\n");)
#elif defined(CHECK_SCALED_E2M1_E2M1_M16N8K32)               /* documented for: sm_120a sm_120f (the TRAINED scan) */
FORM_KERNEL(asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e2m1.e2m1"
                         ".f32.ue8m0 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13},"
                         " %14, {0, 0}, %15, {0, 0};\n"
                         : "=f"(result->singles[0]), "=f"(result->singles[1]), "=f"(result->singles[2]),
                           "=f"(result->singles[3])
                         : "r"(lane->words[A_AT]), "r"(lane->words[A_AT + 1]), "r"(lane->words[A_AT + 2]),
                           "r"(lane->words[A_AT + 3]), "r"(lane->words[B_AT]), "r"(lane->words[B_AT + 1]),
                           "f"(lane->singles[0]), "f"(lane->singles[1]), "f"(lane->singles[2]), "f"(lane->singles[3]),
                           "r"(lane->words[SCALE_A_AT]), "r"(lane->words[SCALE_B_AT]));)
#elif defined(CHECK_HOST_DEPENDENT_LAUNCH_ATTRIBUTE)     /* documented for: from_90 (the attribute is host code) */
FORM_KERNEL(result->words[0] = lane->words[A_AT];)
/* The launch a step would make with the attribute: built here, called by nothing (no main: nothing is linked). */
cudaError_t form_launch_dependent(const struct lane_record *lanes, struct lane_record *results, cudaStream_t stream)
{
    cudaLaunchAttribute attribute;
    cudaLaunchConfig_t configuration;
    memset(&attribute, 0, sizeof(attribute));
    memset(&configuration, 0, sizeof(configuration));
    attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute.val.programmaticStreamSerializationAllowed = 1;
    configuration.gridDim = dim3(1);
    configuration.blockDim = dim3(LANES_PER_WARP);
    configuration.stream = stream;
    configuration.attrs = &attribute;
    configuration.numAttrs = 1;
    return cudaLaunchKernelEx(&configuration, form_kernel, lanes, results);
}
#elif defined(CHECK_FEATURE_SYMBOL_OF_SM_120A)               /* documented for: unstated (what the probe guards with) */
#if defined(__CUDA_ARCH__) && !defined(__CUDA_ARCH_FEAT_SM120_ALL)
#error "__CUDA_ARCH_FEAT_SM120_ALL is not defined for this target"
#endif
FORM_KERNEL(result->words[0] = lane->words[A_AT];)
#else
#error "define exactly one CHECK_ symbol that this file knows (batch_i11a_index_scan.sh reads their names from it)"
#endif
