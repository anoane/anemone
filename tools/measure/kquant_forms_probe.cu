/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * kquant_forms_probe: round 1, package 5 of the speed campaign (research/speed/ROUND_1.md, "kquant_forms"; ledger
 * K1, K2, K3, K4, K8, K10). What the routed experts of one MoE layer cost with Q2_K or Q3_K weights when the
 * kernel of kquant_expert_probe.cu is given the occupancy and the operands its profile says it lacks, and what
 * the format's own integer contract costs on the same cards. One type a run (--type), every variant a named
 * instance of one binary, interleaved by probe_harness.h; a step is the five launches of a layer: the input made
 * ready, gate with up, the hidden vectors made ready, down, the rows' sums.
 *
 * INSTANCES. baseline (twice): the earlier probe's structure, no launch bound. Contract C1, same bits required:
 * bounds:256x3 .. 256x6, hold:fetch, hold:shared, ahead:1, operand:placed16. Contract C1 where the unit computes
 * exactly, which is measured (unit_bits_against_baseline: a variant that differs prints `reported` and class F,
 * and is then held to the host reference as a contract of its own): operand:carried_bf16, operand:fp8 (sm_120a),
 * and the combinations with carried_bf16. The combinations are {registers, fetch} x {ahead 0, 1} x {codes,
 * carried_bf16} at one bound (256x5; kquant_exact_kernels.cuh, KQ_COMBINED_BLOCKS). Own contracts, class F (their
 * values are not C1's): inputs:fp16 (ledger K8: the carried kernel on unquantized inputs) and inputs:int8
 * (kquant_integer_kernels.cuh).
 *
 * CHECKED per case before anything is timed, on step 0, for two pairs (the first row at its last expert, the last
 * row at its first: column 0 and the last column of a job where rows share experts):
 *   prepared_vectors        what a prepare kernel wrote for the row's input and for the pair's hidden vector: codes,
 *                           scales and X bit for bit against the host's quantizer; every operand pair, decoded by
 *                           its type, against the value it stands for
 *   host_reference_*        the hidden vector and the output against double precision, fed with the card's own
 *                           previous stage: 2e-3 of the rms, the baseline's fp32 output 1e-4. The stored records
 *                           are synthetic; the host reads an output's blocks out of its records by the record's
 *                           layout, written here a second time
 *   bits_against_baseline   every output word of a same-bits variant against the baseline's: 0 may differ
 *   integer_contract        inputs:int8: down's outputs for the pair, bit for bit, against the host's integers and
 *                           the contract's fp32 steps (gate and up, the same body, are held by the reference)
 *   rows_exact              every row launched alone against the same row in its launch: 0 words may differ
 *   guard_bytes, and the DIGEST of all outputs, which the batch compares between the two CMP 170HX
 * EACH CHECK SHOWN FAILING, by values and never by anything a kernel indexes by (kqf_check_case): --corrupt (the
 * first pair's gate and down matrices: its two references, not the other pair's), --break exactness (with --rows 4
 * --overlap 50: the baseline's twin, whose order of slices follows a job's rows: rows_exact, and the next variant's
 * bits_against_baseline), --break quantization (codes of the first row's input, after its prepare launch:
 * prepared_vectors), --break integers (one integer of the host's copy: integer_contract), --break-route (seven
 * launches past one bound each), --break-guard.
 *
 * VERDICTS (ROUND_1.md, package 5): c1_at_one_row (the best variant that kept C1's bits against the baseline) and
 * integer_further_expert (the 27-expert case less the 21-expert case, against the baseline's same difference).
 *
 *   kquant_forms_probe [--type q2_k|q3_k] [--rows 1,2,4,6,8] [--overlap 0,50,100] [--break NAME,...]
 *                      and the flags of probe_harness.h
 * Measurement tool: not part of the engine.
 */
#include "probe_harness.h"
#include "kquant_integer_kernels.cuh"
#include "kquant_forms_reference.h"

enum {
    KQF_BITS_REQUIRED = 0, KQF_BITS_MEASURED, KQF_BITS_OWN,     /* a variant's outputs against the baseline's */
    KQF_PREPARE_INPUT = 0, KQF_GATE_UP, KQF_PREPARE_HIDDEN, KQF_DOWN, KQF_ROW_SUM, KQF_STEP_LAUNCHES,
    KQF_VALUES = 0, KQF_WORDS, KQF_PAIRS, KQF_EXPONENTS, KQF_GROUP_SUMS, KQF_SCALES, KQF_LIMBS, KQF_VECTOR_PARTS,
    KQF_CHECKED_STEP = 0, KQF_CHECKED_PAIRS = 2, KQF_OVERLAP_STEP = 50, KQF_PERCENT = 100,
    KQF_FIRST_VARIANT = PROBE_BASELINE_AGAIN + 1, KQF_FEWER_EXPERTS = 21, KQF_MORE_EXPERTS = 27,
    KQF_ROW_SUMS_AT = KQ_MOST_PAIRS * KQ_HIDDEN_SIZE,            /* a result: the pairs' outputs, then the rows' sums */
    KQF_RESULT_FLOATS = KQF_ROW_SUMS_AT + KQ_MOST_ROWS * KQ_HIDDEN_SIZE,
    KQF_CORRUPTED_BYTES = 1024 * 1024, KQF_CORRUPTED_BYTE = 0x5A, KQF_BROKEN_BYTES = 64, KQF_BROKEN_CODE = 0x7E,
    KQF_STALE_BYTE = 0xFF, KQF_COLD_L2_SIZES = 4, KQF_FILL_THREADS = 256, KQF_FILL_BLOCKS_PER_SM = 8,
    KQF_BROKEN_EXPERT = 0, KQF_BROKEN_PAIR, KQF_BROKEN_COLUMNS, KQF_BROKEN_JOBS, KQF_BROKEN_ROWS,   /* --break-route */
    KQF_BROKEN_FIRST_ROW, KQF_BROKEN_SLOT, KQF_BROKEN_ROUTES,
    /* the stored words: any bits are a record, but a Q3_K scale has six bits, and d, dmin are fp16 of 2^-10 .. 2^-7 */
    KQF_Q3_K_SCALES_MASK = 0x3F3F3F3F, KQF_D_BITS_SMALLEST = 0x1400, KQF_D_BITS_SPAN = 0x0C00, KQF_HALF_MASK = 0xFFFF,
    KQF_STORED_SEED = 0x243F6A88, KQF_MIX_SHIFT_OUTER = 16, KQF_MIX_SHIFT_INNER = 15, KQF_WORD_BITS = 32,
    KQF_OUTPUT_CODE_WORDS = KQ_BLOCK_HALVES * KQ_LANES_PER_GROUP * KQ_HALF_PAIRS, KQF_WORD_OUTPUT_CODES = 8,
    KQF_CODE_MASK = 3,
    /* synthetic inputs, exact in bfloat16: a block of 32 is of one of eight kinds (kqf_input_value) */
    KQF_INPUT_SEED = 0x1234567, KQF_INPUT_KINDS = 8, KQF_INPUT_PLAIN_KINDS = 4, KQF_INPUT_SPREAD = 4,
    KQF_INPUT_ZEROS = 5, KQF_INPUT_TIES = 6, KQF_INPUT_TINY_SHIFT = 16, KQF_INPUT_STEPS = 255, KQF_SPREAD_POWERS = 18,
    KQF_TIE_NUMERATORS = 8, KQF_TIE_FIRST_NUMERATOR = 17, KQF_TIE_NUMERATOR_STEP = 2, KQF_TIE_POWERS = 14,
    KQF_TIE_LOWEST_POWER = -20, KQF_TIE_LARGEST_POWER = -10,
    KQF_ROUTE_SEED = 0xABCDEF, KQF_ROUTE_WEIGHT_STEPS = 32, KQF_STEP_STRIDE = 977
};
#define KQF_MIX_MULTIPLIER_FIRST 0x7feb352du
#define KQF_MIX_MULTIPLIER_SECOND 0x846ca68bu
#define KQF_FLAG_TYPE "--type"
#define KQF_FLAG_ROWS "--rows"
#define KQF_FLAG_OVERLAP "--overlap"
#define KQF_FLAG_BREAK "--break"
#define KQF_RELATIVE_TOLERANCE 2.0e-3            /* a bf16 cast that falls the other way is 2e-4 of the rms (f08) */
#define KQF_BASELINE_OUTPUT_TOLERANCE 1.0e-4     /* fp32: f08's 180 outputs of this contract were off by 4e-7 .. 7e-6 */
#define KQF_INPUT_OFFSET 128.0f
#define KQF_INPUT_DIVISOR 64.0f
#define KQF_ROUTE_WEIGHT_DIVISOR 64.0f
#define KQF_LARGEST_CODE_VALUE 448.0f
#define KQF_BYTES_PER_MB 1.0e6
#define KQF_ROW_FLOATS ((size_t)KQ_EXPERTS_PER_ROW * KQ_HIDDEN_SIZE)
#define KQF_ROW_WORDS(rows) ((uint64_t)(rows) * (KQ_EXPERTS_PER_ROW + 1u) * KQ_HIDDEN_SIZE)
#define KQF_TEXT_OF(value) #value
#define KQF_TEXT(value) KQF_TEXT_OF(value)

struct kqf_variant {
    const char *name;
    int inputs, bits;                            /* KQ_INPUTS_* of its prepare kernel; KQF_BITS_* */
    uint32_t bound_blocks, shared_hold;          /* its bound in blocks of gate with up (0: none); its hold is shared */
    struct kq_kernels kernels;
};
#define KQF_COMBINED(name, inputs, bits, tag) \
    {name ",bounds:256x" KQF_TEXT(KQ_COMBINED_BLOCKS), inputs, bits, KQ_COMBINED_BLOCKS, 0, KQ_BOTH_TYPES(tag)}
static const struct kqf_variant kqf_variants[] = {
    {"baseline", KQ_INPUTS_E4M3, KQF_BITS_REQUIRED, 0, 0, KQ_BOTH_TYPES(baseline)},
    {"baseline_again", KQ_INPUTS_E4M3, KQF_BITS_REQUIRED, 0, 0, KQ_BOTH_TYPES(baseline)},
    {"bounds:256x3", KQ_INPUTS_E4M3, KQF_BITS_REQUIRED, 3, 0, KQ_BOTH_TYPES(bound_3)},       /* KQF_FIRST_VARIANT */
    {"bounds:256x4", KQ_INPUTS_E4M3, KQF_BITS_REQUIRED, 4, 0, KQ_BOTH_TYPES(bound_4)},
    {"bounds:256x5", KQ_INPUTS_E4M3, KQF_BITS_REQUIRED, 5, 0, KQ_BOTH_TYPES(bound_5)},
    {"bounds:256x6", KQ_INPUTS_E4M3, KQF_BITS_REQUIRED, 6, 0, KQ_BOTH_TYPES(bound_6)},
    {"hold:fetch", KQ_INPUTS_E4M3, KQF_BITS_REQUIRED, 0, 0, KQ_BOTH_TYPES(fetch)},
    {"hold:shared", KQ_INPUTS_E4M3, KQF_BITS_REQUIRED, 0, 1, KQ_SHARED_KERNELS},
    {"ahead:1", KQ_INPUTS_E4M3, KQF_BITS_REQUIRED, 0, 0, KQ_BOTH_TYPES(ahead)},
    {"operand:placed16", KQ_INPUTS_PLACED16, KQF_BITS_REQUIRED, 0, 0, KQ_BOTH_TYPES(placed16)},
    {"operand:carried_bf16", KQ_INPUTS_CARRIED, KQF_BITS_MEASURED, 0, 0, KQ_BOTH_TYPES(carried)},
    {"operand:fp8", KQ_INPUTS_E4M3, KQF_BITS_MEASURED, 0, 0, KQ_FP8_KERNELS},
    KQF_COMBINED("ahead:1", KQ_INPUTS_E4M3, KQF_BITS_REQUIRED, held_ahead),
    KQF_COMBINED("operand:carried_bf16", KQ_INPUTS_CARRIED, KQF_BITS_MEASURED, held_carried),
    KQF_COMBINED("ahead:1,operand:carried_bf16", KQ_INPUTS_CARRIED, KQF_BITS_MEASURED, held_ahead_carried),
    KQF_COMBINED("hold:fetch", KQ_INPUTS_E4M3, KQF_BITS_REQUIRED, fetched),
    KQF_COMBINED("hold:fetch,ahead:1", KQ_INPUTS_E4M3, KQF_BITS_REQUIRED, fetched_ahead),
    KQF_COMBINED("hold:fetch,operand:carried_bf16", KQ_INPUTS_CARRIED, KQF_BITS_MEASURED, fetched_carried),
    KQF_COMBINED("hold:fetch,ahead:1,operand:carried_bf16", KQ_INPUTS_CARRIED, KQF_BITS_MEASURED,
                 fetched_ahead_carried),
    {"inputs:fp16", KQ_INPUTS_RAW, KQF_BITS_OWN, 0, 0, KQ_BOTH_TYPES(carried)},
    {"inputs:int8", KQ_INPUTS_INT8, KQF_BITS_OWN, 0, 0, KQ_BOTH_TYPES(int8)},
};
#define KQF_VARIANTS PROBE_COUNT_OF(kqf_variants)
/* the twin of --break exactness, which no instance times */
static const struct kqf_variant kqf_broken_order = {"baseline", KQ_INPUTS_E4M3, KQF_BITS_REQUIRED, 0, 0,
                                                    KQ_BOTH_TYPES(broken_order)};
static struct probe_flag kqf_flags[] = {
    PROBE_COMMON_FLAGS,
    PROBE_WORD_LIST(KQF_FLAG_TYPE, "q2_k,q3_k", "q2_k", "the block type of this run: one"),
    PROBE_CASE_NUMBERS(KQF_FLAG_ROWS, 1, KQ_MOST_ROWS, "1,2,4,6,8", "rows of a step"),
    PROBE_CASE_WORDS(KQF_FLAG_OVERLAP, "0,50,100", "0,50,100", "percent of its experts a row shares with the next"),
    PROBE_WORD_LIST(KQF_FLAG_BREAK, "exactness,quantization,integers", "", "show that check failing"),
};
static struct probe_instance kqf_instances[KQF_VARIANTS];
struct kqf_step {                                /* one step, as its launches take it */
    struct kq_launch gate_up, down;
    uint32_t job_count, row_count;
};
static struct {
    int kquant;                                  /* --type */
    uint8_t *stored;
    struct kq_job *jobs;                         /* [slot of a stream][job] */
    struct kq_vectors input, hidden;             /* of the rows of the batch, of the pairs */
    float *result;                               /* [pair][hidden size] of an expert for a row, then the rows' sums */
    struct kqf_step steps[PROBE_STREAM_STEPS];
} kqf;
static float kqf_result[KQF_RESULT_FLOATS], kqf_baseline_result[KQF_RESULT_FLOATS], kqf_alone_result[KQF_RESULT_FLOATS];

/* ---- synthetic experts and inputs ---- */
__host__ __device__ inline uint32_t kqf_mix(uint32_t value) {
    value = (value ^ (value >> KQF_MIX_SHIFT_OUTER)) * KQF_MIX_MULTIPLIER_FIRST;
    value = (value ^ (value >> KQF_MIX_SHIFT_INNER)) * KQF_MIX_MULTIPLIER_SECOND;
    return value ^ (value >> KQF_MIX_SHIFT_OUTER);
}
/* Word `word_index` of the stored experts: a hash of its place, held to what a record may hold. */
__host__ __device__ inline uint32_t kqf_stored_word(int kquant, uint64_t word_index) {
    const uint32_t word = (uint32_t)(word_index % (KQ_RECORD_BYTES(kquant) / KQ_WORD_BYTES));
    const uint32_t bits = kqf_mix(KQF_STORED_SEED + (uint32_t)kquant + (uint32_t)word_index
                                  + kqf_mix((uint32_t)(word_index >> KQF_WORD_BITS)));
    if (word >= (uint32_t)KQ_BLOCK_SCALES_AT(kquant) / KQ_WORD_BYTES) {          /* two fp16 block scales */
        return (KQF_D_BITS_SMALLEST + (bits & KQF_HALF_MASK) % KQF_D_BITS_SPAN)
               | ((KQF_D_BITS_SMALLEST + (bits >> KQ_HALF_BITS) % KQF_D_BITS_SPAN) << KQ_HALF_BITS);
    }
    return kquant == KQ_Q3_K && word >= KQ_Q3_K_GROUP_SCALES_AT / KQ_WORD_BYTES ? bits & KQF_Q3_K_SCALES_MASK : bits;
}
/* Any grid, any block: words 0 .. word_count - 1 of the stored experts (1.5 to 2 GB would be seconds over the
 * link, and minutes of packing on the host). Its name keeps it out of a profile of the step's kernels. */
template <int kquant>
__global__ void setup_fill_kernel(uint32_t *stored_words, uint64_t word_count) {
    const uint64_t thread_total = (uint64_t)gridDim.x * blockDim.x;
    for (uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; index < word_count; index += thread_total) {
        stored_words[index] = kqf_stored_word(kquant, index);
    }
}
/* One output's row of one matrix out of the stored records, block by block, as the reference takes it: the record's
 * layout (the header of kquant_exact_kernels.cuh) written a second time, for the host. Of output o a code word of
 * lane 4 (o % 8) + t holds, per step, inputs 2t and 2t + 1 (register o % 16 / 8) and 2t + 8, 2t + 9 (two registers
 * on), the second of each in the word's upper half. */
static const struct kq_reference_block *kqf_row(uint32_t expert, uint32_t matrix_kind, uint32_t output) {
    static struct kq_reference_block row[KQ_HIDDEN_SIZE / KQ_BLOCK_WEIGHTS];
    const int kquant = kqf.kquant, is_q3_k = kquant == KQ_Q3_K;
    const uint32_t blocks = (matrix_kind == KQ_DOWN_MATRIX ? KQ_INNER_SIZE : KQ_HIDDEN_SIZE) / KQ_BLOCK_WEIGHTS;
    const uint32_t lane_group = output % KQ_HALF_OUTPUTS, output_half = output % KQ_TILE_OUTPUTS / KQ_HALF_OUTPUTS;
    const uint32_t scale_word = output % KQ_TILE_OUTPUTS * KQ_BLOCK_SCALES(kquant) / KQ_FP16_BYTES;
    for (uint32_t block = 0; block < blocks; block++) {
        const uint64_t record = ((uint64_t)expert * KQ_MATRIX_KINDS + matrix_kind) * KQ_MATRIX_RECORDS
                                + (uint64_t)(output / KQ_TILE_OUTPUTS) * blocks + block;
        const uint64_t first_word = record * (KQ_RECORD_BYTES(kquant) / KQ_WORD_BYTES);
        const uint64_t signs_at = first_word + KQ_SIGNS_AT / KQ_WORD_BYTES;
        const uint64_t scales_at = first_word + KQ_BLOCK_SCALES_AT(kquant) / KQ_WORD_BYTES;
        const uint32_t d_bits = kqf_stored_word(kquant, scales_at + scale_word);
        struct kq_reference_block *decoded = &row[block];
        for (uint32_t place = 0; place < KQF_OUTPUT_CODE_WORDS; place++) {             /* [half][lane position][pair] */
            const uint32_t half = place / (KQ_LANES_PER_GROUP * KQ_HALF_PAIRS), pair = place % KQ_HALF_PAIRS;
            const uint32_t position = place / KQ_HALF_PAIRS % KQ_LANES_PER_GROUP;
            const uint64_t half_lane = (half * KQ_HALF_OUTPUTS + lane_group) * KQ_LANES_PER_GROUP + position;
            const uint32_t codes = kqf_stored_word(kquant, first_word + half_lane * KQ_HALF_PAIRS + pair);
            const uint64_t sign_word = signs_at + half_lane * KQ_SIGN_WORDS_PER_HALF + pair / KQ_LOAD_PAIRS;
            const uint32_t signs = !is_q3_k ? 0u
                                   : kqf_stored_word(kquant, sign_word)
                                     >> (pair % KQ_LOAD_PAIRS * KQ_PAIR_GROUPS * KQ_STEP_SIGN_BITS);
            for (uint32_t field = 0; field < KQF_WORD_OUTPUT_CODES; field++) {        /* step, register, its half */
                const uint32_t step = field / (KQ_PAIR_GROUPS * KQ_REGISTER_HALVES);
                const uint32_t in_half = field % KQ_REGISTER_HALVES;
                const uint32_t upper = field / KQ_REGISTER_HALVES % KQ_PAIR_GROUPS, half_bit = in_half * KQ_HALF_BITS;
                const uint32_t weight_register = output_half + upper * KQ_LANE_OUTPUTS;
                const uint32_t input = position * KQ_REGISTER_HALVES + upper * KQ_HALF_INPUTS + in_half;
                const uint32_t code_at = step * KQ_STEP_CODE_BITS + weight_register * KQ_CODE_BITS + half_bit;
                const int stored = (int)((codes >> code_at) & KQF_CODE_MASK);
                const int negative = (int)((signs >> (step * KQ_STEP_SIGN_BITS + weight_register + half_bit)) & 1u);
                decoded->code[((half * KQ_HALF_PAIRS + pair) * KQ_PAIR_GROUPS + step) * KQ_GROUP_WEIGHTS + input] =
                    negative ? -stored - 1 : stored;
            }
        }
        for (uint32_t group = 0; group < KQ_REFERENCE_GROUPS; group++) {               /* scales: [half][lane group] */
            const uint32_t half_group = group / KQ_HALF_GROUPS * KQ_HALF_OUTPUTS + lane_group;
            const uint32_t place = KQ_GROUP_SCALES_AT(kquant) + output_half
                                   + (half_group * KQ_HALF_GROUPS + group % KQ_HALF_GROUPS) * KQ_LANE_OUTPUTS;
            const uint32_t byte = (kqf_stored_word(kquant, first_word + place / KQ_WORD_BYTES)
                                   >> (place % KQ_WORD_BYTES * KQ_BYTE_BITS)) & KQ_BYTE_MASK;
            decoded->scale[group] = is_q3_k ? (int)byte - KQ_Q3_K_SCALE_BIAS : (int)(byte & KQ_NIBBLE_MASK);
            decoded->minimum[group] = is_q3_k ? 0 : (int)(byte >> KQ_NIBBLE_BITS);
        }
        decoded->d = kq_reference_half(is_q3_k ? d_bits >> (output % KQ_FP16_BYTES * KQ_HALF_BITS) : d_bits);
        decoded->dmin = is_q3_k ? 0.0 : kq_reference_half(d_bits >> KQ_HALF_BITS);
    }
    return row;
}
/* One element of a row's input, exact in bfloat16. Its block of 32 is plain (multiples of 1/64 in -2 .. 2, over
 * 2^0 .. 2^3), spread (powers of two over 2^17: subnormal codes), zeros of both signs, ties (its first value the
 * largest code at its scale, 448 * 2^-10, the others half way between two neighbouring codes), or tiny (plain over
 * 2^16: under the reference's floor of 1e-4). */
static float kqf_input_value(uint32_t row, uint32_t column) {
    const uint32_t seed = KQF_INPUT_SEED + row * KQ_HIDDEN_SIZE, bits = kqf_mix(seed + column);
    const uint32_t kind = kqf_mix(~seed + column / KQ_ACTIVATION_BLOCK) % KQF_INPUT_KINDS;
    const float sign = ((bits >> KQ_BYTE_BITS) & 1u) != 0u ? -1.0f : 1.0f;
    if (kind == KQF_INPUT_SPREAD) { return ldexpf(sign, -(int)(bits % KQF_SPREAD_POWERS)); }
    if (kind == KQF_INPUT_ZEROS) { return sign * 0.0f; }
    if (kind == KQF_INPUT_TIES && column % KQ_ACTIVATION_BLOCK == 0) {
        return ldexpf(sign * KQF_LARGEST_CODE_VALUE, KQF_TIE_LARGEST_POWER);
    }
    if (kind == KQF_INPUT_TIES) {
        return ldexpf(sign * (float)(KQF_TIE_FIRST_NUMERATOR + bits % KQF_TIE_NUMERATORS * KQF_TIE_NUMERATOR_STEP),
                      KQF_TIE_LOWEST_POWER + (int)(bits / KQF_TIE_NUMERATORS % KQF_TIE_POWERS));
    }
    return ((float)(bits & KQF_INPUT_STEPS) - KQF_INPUT_OFFSET) / KQF_INPUT_DIVISOR
           / (float)(1u << (kind < KQF_INPUT_PLAIN_KINDS ? kind : (uint32_t)KQF_INPUT_TINY_SHIFT));
}

/* ---- routes, jobs and the validation of a step (check 10) ---- */
static uint32_t kqf_rows(const struct probe_case *probe_case) {
    return (uint32_t)probe_case_value(probe_case, KQF_FLAG_ROWS);
}
static uint32_t kqf_overlap(const struct probe_case *probe_case) {
    return (uint32_t)probe_case_value(probe_case, KQF_FLAG_OVERLAP) * KQF_OVERLAP_STEP;
}
static uint32_t kqf_span(uint32_t overlap, uint32_t rows) {      /* the experts a step of `rows` rows reads */
    return KQ_EXPERTS_PER_ROW + (rows - 1u) * (KQ_EXPERTS_PER_ROW * (KQF_PERCENT - overlap) / KQF_PERCENT);
}
/* The synthetic routing. Row r takes six consecutive experts from the 6r-th (overlap 0), the 3r-th (50) or the
 * first (100) of its step's span, and the spans of consecutive steps follow one another around the ring of
 * experts: between two reads of an expert a card reads 80 others or more (CONVENTIONS 3.5; kqf_open_card holds
 * that to four L2 sizes). span_rows is the case's rows also where one row is launched alone: it keeps its experts. */
static uint32_t kqf_expert(uint64_t step, uint32_t overlap, uint32_t span_rows, uint32_t row, uint32_t slot) {
    const uint32_t stride = KQ_EXPERTS_PER_ROW * (KQF_PERCENT - overlap) / KQF_PERCENT;
    return (uint32_t)((step * kqf_span(overlap, span_rows) + row * stride + slot) % KQ_EXPERT_COUNT);
}
static float kqf_route_weight(uint64_t step, uint32_t row, uint32_t slot) {
    const uint32_t place = (uint32_t)step * KQF_STEP_STRIDE + row * KQ_EXPERTS_PER_ROW + slot;
    return (float)(1u + kqf_mix(~(uint32_t)KQF_ROUTE_SEED + place) % KQF_ROUTE_WEIGHT_STEPS) / KQF_ROUTE_WEIGHT_DIVISOR;
}
/* The jobs of rows first_row .. first_row + row_count - 1 (at most KQ_MOST_ROWS), which fill the batch from its
 * slot 0: one per expert, its columns in row order. */
static uint32_t kqf_build_jobs(uint64_t step, const struct probe_case *probe_case, uint32_t first_row,
                               uint32_t row_count, struct kq_job *jobs) {
    uint32_t job_count = 0;
    for (uint32_t pair = 0; pair < row_count * KQ_EXPERTS_PER_ROW; pair++) {
        const uint32_t row = first_row + pair / KQ_EXPERTS_PER_ROW, slot = pair % KQ_EXPERTS_PER_ROW;
        const uint32_t expert = kqf_expert(step, kqf_overlap(probe_case), kqf_rows(probe_case), row, slot);
        uint32_t job_index = 0;
        while (job_index < job_count && jobs[job_index].expert != expert) { job_index++; }
        struct kq_job *job = &jobs[job_index];
        if (job_index == job_count) {
            job_count++;
            job->expert = expert;
            job->row_count = 0;
            for (uint32_t column = 0; column < KQ_MOST_ROWS; column++) {
                job->pair[column] = pair;                     /* a column not in use reads what column 0 reads */
                job->route_weight[column] = 0.0f;
            }
        }
        job->pair[job->row_count] = pair;
        job->route_weight[job->row_count++] = kqf_route_weight(step, row, slot);
    }
    return job_count;
}
/* The bytes one vector has in array `part` of a set of vectors of `size` values, in struct kq_vectors' order. */
static uint64_t kqf_part_bytes(uint32_t part, uint64_t size) {
    static const uint32_t bytes_per[KQF_VECTOR_PARTS][2] = {{4, 1},  {1, 1},   {2, 1},   {1, 32},
                                                            {4, 16}, {4, 256}, {32, 256}};
    return size * bytes_per[part][0] / bytes_per[part][1];
}
/* Check 10: a step of these jobs with one variant's kernels, prepared in a stream's slot, stays inside every array
 * as allocated, inside the card's limits and the variant's launch bound. 1: it does; 0: it printed why not, and no
 * job was read past the first value out of its range. */
static int kqf_step_is_valid(uint32_t variant_index, const struct kq_job *jobs, uint64_t job_count, uint64_t slot,
                             uint64_t first_row, uint64_t row_count) {
    const struct kqf_variant *variant = &kqf_variants[variant_index];
    const char *kernel = variant->name;
    const uint64_t pair_count = row_count * KQ_EXPERTS_PER_ROW;
    const struct kq_vectors *sets[KQ_STAGES] = {&kqf.input, &kqf.hidden};
    int valid = 1;
    probe_launch_value(&valid, kernel, "rows", row_count, 1, KQ_MOST_ROWS);
    probe_launch_value(&valid, kernel, "the first row with the rows", first_row + row_count, 1, KQ_MOST_ROWS);
    probe_launch_value(&valid, kernel, "the slot of a stream", slot, 0, PROBE_STREAM_STEPS - 1u);
    probe_launch_value(&valid, kernel, "jobs", job_count, 1, valid ? pair_count : 1);
    for (uint64_t job = 0; valid && job < job_count; job++) {
        probe_launch_value(&valid, kernel, "a job's expert", jobs[job].expert, 0, KQ_EXPERT_COUNT - 1u);
        probe_launch_value(&valid, kernel, "a job's columns", jobs[job].row_count, 1, row_count);
        for (uint32_t column = 0; column < KQ_MOST_ROWS; column++) {
            probe_launch_value(&valid, kernel, "a column's pair", jobs[job].pair[column], 0, pair_count - 1u);
        }
    }
    for (uint32_t stage = 0; valid && stage < KQ_STAGES; stage++) {
        const struct kq_vectors *set = sets[stage];
        const struct probe_instance *instance = stage == 0 ? &kqf_instances[variant_index] : NULL;
        const uint32_t matrices = stage == 0 ? KQ_GATE_UP : KQ_DOWN, threads = KQ_STAGE_THREADS(matrices);
        const uint64_t vectors = stage == 0 ? row_count : pair_count;
        const uint64_t slot_words = variant->shared_hold ? KQ_SLOT_STAGES * threads * KQ_SLOT_WORDS
                                                         : (uint32_t)KQ_LOAD_WORDS;
        const void *parts[KQF_VECTOR_PARTS] = {set->values,     set->words,  set->pairs, set->exponents,
                                               set->group_sums, set->scales, set->minimum_operand};
        probe_launch_geometry(&valid, kernel, instance, KQ_STAGE_TILES(matrices), job_count, 1, threads,
                              (KQ_SHARED_SUMS(matrices) + slot_words) * sizeof(uint32_t));
        probe_launch_geometry(&valid, "kq_prepare_kernel", NULL, vectors, 1, 1, set->size / KQ_ACTIVATION_BLOCK, 0);
        for (uint32_t part = 0; part < KQF_VECTOR_PARTS; part++) {
            const uint64_t first = stage == 0 && part == KQF_VALUES ? first_row : 0;
            probe_launch_range(&valid, kernel, "an array of vectors, in vectors", first, vectors,
                               kqf_part_bytes(part, set->size), probe_cuda_array_bytes(parts[part]));
        }
    }
    probe_launch_geometry(&valid, "kq_row_sum_kernel", NULL, row_count, 1, 1, KQ_ROW_SUM_THREADS, 0);
    probe_launch_range(&valid, kernel, "stored, in experts", 0, KQ_EXPERT_COUNT, KQ_EXPERT_BYTES(kqf.kquant),
                       probe_cuda_array_bytes(kqf.stored));
    probe_launch_range(&valid, kernel, "jobs, in jobs", slot * KQ_MOST_PAIRS, job_count, sizeof(struct kq_job),
                       probe_cuda_array_bytes(kqf.jobs));
    probe_launch_range(&valid, kernel, "result, in floats", 0, KQF_RESULT_FLOATS, sizeof(float),
                       probe_cuda_array_bytes(kqf.result));
    return valid;
}
/* A validated step into a stream's slot: its jobs to the card, its two launches' arguments. */
static void kqf_set_step(uint32_t slot, const struct kq_job *jobs, uint32_t job_count, uint32_t first_row,
                         uint32_t row_count) {
    struct kqf_step *step = &kqf.steps[slot];
    struct kq_job *slot_jobs = kqf.jobs + (size_t)slot * KQ_MOST_PAIRS;
    const struct kq_launch launch = {kqf.stored, kqf.input, slot_jobs, (float *)kqf.hidden.values, kqf.result};
    PROBE_CHECK_CUDA(cudaMemcpy(slot_jobs, jobs, job_count * sizeof(struct kq_job), cudaMemcpyHostToDevice));
    PROBE_CHECK_CUDA(cudaDeviceSynchronize());   /* a step starts on an idle card, as the earlier probe's */
    step->gate_up = step->down = launch;
    step->gate_up.vectors.values = kqf.input.values + (size_t)first_row * KQ_HIDDEN_SIZE;
    step->down.vectors = kqf.hidden;
    step->job_count = job_count;
    step->row_count = row_count;
}

/* ---- launches ---- */
template <int kquant, int inputs>
static void kqf_prepare_floats(const struct kq_vectors *vectors, uint32_t vector_count) {
    kq_prepare_kernel<kquant, inputs><<<vector_count, vectors->size / KQ_ACTIVATION_BLOCK>>>(*vectors);
}
template <int kquant>
static void kqf_prepare_integers(const struct kq_vectors *vectors, uint32_t vector_count) {
    kq_prepare_integers_kernel<kquant><<<vector_count, vectors->size / KQ_ACTIVATION_BLOCK>>>(*vectors);
}
#define KQF_PREPARE_KERNELS(kquant)                                                              \
    {kqf_prepare_floats<kquant, KQ_INPUTS_E4M3>, kqf_prepare_floats<kquant, KQ_INPUTS_PLACED16>, \
     kqf_prepare_floats<kquant, KQ_INPUTS_CARRIED>, kqf_prepare_floats<kquant, KQ_INPUTS_RAW>,   \
     kqf_prepare_integers<kquant>}
static void (*const kqf_prepare[KQ_TYPES][KQ_INPUT_KINDS])(const struct kq_vectors *, uint32_t) = {
    KQF_PREPARE_KERNELS(KQ_Q2_K), KQF_PREPARE_KERNELS(KQ_Q3_K)};
/* One of the five launches of a step that kqf_set_step prepared. */
static void kqf_launch(const struct kqf_variant *variant, const struct kqf_step *step, uint32_t launch) {
    if (launch == KQF_PREPARE_INPUT) {
        kqf_prepare[kqf.kquant][variant->inputs](&step->gate_up.vectors, step->row_count);
    } else if (launch == KQF_GATE_UP) {
        variant->kernels.launch[kqf.kquant][0](&step->gate_up, step->job_count);
    } else if (launch == KQF_PREPARE_HIDDEN) {
        kqf_prepare[kqf.kquant][variant->inputs](&step->down.vectors, step->row_count * KQ_EXPERTS_PER_ROW);
    } else if (launch == KQF_DOWN) {
        variant->kernels.launch[kqf.kquant][1](&step->down, step->job_count);
    } else {
        kq_row_sum_kernel<<<step->row_count, KQ_ROW_SUM_THREADS>>>(kqf.result, kqf.result + KQF_ROW_SUMS_AT);
    }
    PROBE_AFTER_LAUNCH();
}
static int kqf_prepare_step(void *, const struct probe_case *probe_case, uint32_t instance, uint64_t step,
                            uint32_t slot) {
    struct kq_job jobs[KQ_MOST_PAIRS];
    const uint32_t rows = kqf_rows(probe_case), job_count = kqf_build_jobs(step, probe_case, 0, rows, jobs);
    if (!kqf_step_is_valid(instance, jobs, job_count, slot, 0, rows)) { return 0; }
    kqf_set_step(slot, jobs, job_count, 0, rows);
    return 1;
}
static uint32_t kqf_launch_step(void *, const struct probe_case *, uint32_t instance, uint32_t slot) {
    for (uint32_t launch = 0; launch < KQF_STEP_LAUNCHES; launch++) {
        kqf_launch(&kqf_variants[instance], &kqf.steps[slot], launch);
    }
    return KQF_STEP_LAUNCHES;
}

/* ---- the card ---- */
static void *kqf_allocate(uint64_t bytes) {      /* zeroed: no kernel reads a byte nothing wrote */
    void *array = probe_cuda_allocate(bytes);
    PROBE_CHECK_CUDA(cudaMemset(array, 0, bytes));
    return array;
}
static void kqf_allocate_vectors(struct kq_vectors *vectors, uint64_t count, uint32_t size, uint32_t pairs_per_vector) {
    vectors->values = (float *)kqf_allocate(count * kqf_part_bytes(KQF_VALUES, size));
    vectors->words = (uint32_t *)kqf_allocate(count * kqf_part_bytes(KQF_WORDS, size));
    vectors->pairs = (uint32_t *)kqf_allocate(count * kqf_part_bytes(KQF_PAIRS, size));
    vectors->exponents = (uint8_t *)kqf_allocate(count * kqf_part_bytes(KQF_EXPONENTS, size));
    vectors->group_sums = (float *)kqf_allocate(count * kqf_part_bytes(KQF_GROUP_SUMS, size));
    vectors->scales = (float *)kqf_allocate(count * kqf_part_bytes(KQF_SCALES, size));
    vectors->minimum_operand = (uint32_t *)kqf_allocate(count * kqf_part_bytes(KQF_LIMBS, size));
    vectors->size = size;
    vectors->pairs_per_vector = pairs_per_vector;
}
static int kqf_open_card(void *, struct probe_card_facts *facts) {
    static float inputs[KQ_MOST_ROWS * KQ_HIDDEN_SIZE];
    const uint64_t expert_bytes = KQ_EXPERT_BYTES(kqf.kquant);
    const uint64_t stored_words = KQ_EXPERT_COUNT * expert_bytes / KQ_WORD_BYTES;
    int valid = 1;
    if (!probe_cuda_open(facts)) { return 0; }
    const uint32_t fill_blocks = (uint32_t)facts->sm_count * KQF_FILL_BLOCKS_PER_SM;
    PROBE_REQUIRE((KQ_EXPERT_COUNT - KQ_MOST_PAIRS) * expert_bytes >= KQF_COLD_L2_SIZES * facts->l2_bytes,
                  "between two reads of an expert a card must read four times its L2 (CONVENTIONS 3.5)");
    kqf.stored = (uint8_t *)probe_cuda_allocate(stored_words * KQ_WORD_BYTES);
    kqf.jobs = (struct kq_job *)kqf_allocate(PROBE_STREAM_STEPS * KQ_MOST_PAIRS * sizeof(struct kq_job));
    kqf.result = (float *)kqf_allocate(KQF_RESULT_FLOATS * sizeof(float));
    kqf_allocate_vectors(&kqf.input, KQ_MOST_ROWS, KQ_HIDDEN_SIZE, KQ_EXPERTS_PER_ROW);
    kqf_allocate_vectors(&kqf.hidden, KQ_MOST_PAIRS, KQ_INNER_SIZE, 1);
    probe_launch_geometry(&valid, "setup_fill_kernel", NULL, fill_blocks, 1, 1, KQF_FILL_THREADS, 0);
    probe_launch_range(&valid, "setup_fill_kernel", "stored, in words", 0, stored_words, KQ_WORD_BYTES,
                       probe_cuda_array_bytes(kqf.stored));
    PROBE_REQUIRE(valid, "the launch that fills the stored experts was refused");
    if (kqf.kquant == KQ_Q2_K) {
        setup_fill_kernel<KQ_Q2_K><<<fill_blocks, KQF_FILL_THREADS>>>((uint32_t *)kqf.stored, stored_words);
    } else {
        setup_fill_kernel<KQ_Q3_K><<<fill_blocks, KQF_FILL_THREADS>>>((uint32_t *)kqf.stored, stored_words);
    }
    for (uint32_t index = 0; index < KQ_MOST_ROWS * KQ_HIDDEN_SIZE; index++) {
        inputs[index] = kqf_input_value(index / KQ_HIDDEN_SIZE, index % KQ_HIDDEN_SIZE);
    }
    PROBE_CHECK_CUDA(cudaMemcpy((void *)kqf.input.values, inputs, sizeof(inputs), cudaMemcpyHostToDevice));
    PROBE_CHECK_CUDA(cudaGetLastError());
    PROBE_CHECK_CUDA(cudaDeviceSynchronize());
    facts->ring_bytes = stored_words * KQ_WORD_BYTES;
    return 1;
}
static void kqf_close_card(void *) { probe_cuda_close(); }
static void kqf_describe_case(void *, const struct probe_case *probe_case, struct probe_case_facts *facts) {
    facts->format = kqf.kquant == KQ_Q3_K ? "q3_k" : "q2_k";
    facts->part = "experts";
    facts->rows = kqf_rows(probe_case);
    facts->overlap = kqf_overlap(probe_case);
    facts->experts = kqf_span(kqf_overlap(probe_case), kqf_rows(probe_case));
    facts->megabytes = (double)facts->experts * (double)KQ_EXPERT_BYTES(kqf.kquant) / KQF_BYTES_PER_MB;
}
static int kqf_configure(void *) {
    if (probe_flag(KQF_FLAG_TYPE)->value_count != 1) {
        fprintf(stderr, KQF_FLAG_TYPE " takes one of q2_k and q3_k: a run times one type, against its own baseline\n");
        return 0;
    }
    kqf.kquant = (int)probe_flag_number(KQF_FLAG_TYPE, 0);
    probe_run.breaking |= (uint32_t)probe_flag_given(KQF_FLAG_BREAK);     /* as a --break-* switch of the harness */
    for (uint32_t index = 0; index < KQF_VARIANTS; index++) {
        const struct kqf_variant *variant = &kqf_variants[index];
        const struct probe_instance instance = {           /* an own contract's values are not C1's: class F */
            variant->name, variant->kernels.names[kqf.kquant], variant->bits == KQF_BITS_OWN ? 'F' : 'C',
            variant->kernels.launch[kqf.kquant][0] != NULL, KQ_STAGE_THREADS(KQ_GATE_UP),
            variant->bound_blocks != 0 ? (uint32_t)KQ_STAGE_THREADS(KQ_GATE_UP) : 0u, variant->bound_blocks, 0, 0};
        kqf_instances[index] = instance;
    }
    return 1;
}

/* ---- checks ---- */
static uint64_t kqf_differing(const void *first, const void *second, size_t count, size_t bytes) {
    uint64_t differing = 0;
    for (size_t at = 0; at < count * bytes; at += bytes) {
        differing += memcmp((const char *)first + at, (const char *)second + at, bytes) != 0 ? 1u : 0u;
    }
    return differing;
}
/* Rows first_row .. of one result against rows second_row .. of another: their pairs' outputs and their sums. */
static uint64_t kqf_rows_differing(const float *first, uint32_t first_row, const float *second, uint32_t second_row,
                                   uint32_t row_count) {
    return kqf_differing(first + first_row * KQF_ROW_FLOATS, second + second_row * KQF_ROW_FLOATS,
                         row_count * KQF_ROW_FLOATS, sizeof(float))
           + kqf_differing(first + KQF_ROW_SUMS_AT + (size_t)first_row * KQ_HIDDEN_SIZE,
                           second + KQF_ROW_SUMS_AT + (size_t)second_row * KQ_HIDDEN_SIZE,
                           (size_t)row_count * KQ_HIDDEN_SIZE, sizeof(float));
}
/* The checked step of rows first_row .. with one variant's kernels, outside the timing: validated, what it stores made
 * stale first (a word a launch does not write cannot pass for a result), the result read back. overwrite (--break
 * quantization): after the input's prepare launch KQF_BROKEN_BYTES codes of the launch's first row are overwritten
 * (of row 0 also where it is launched alone: it stays the row it was in its launch). */
static void kqf_run_checked(const struct probe_case *probe_case, uint32_t variant_index,
                            const struct kqf_variant *launched, uint32_t first_row, uint32_t row_count, int overwrite,
                            float *result) {
    struct kq_job jobs[KQ_MOST_PAIRS];
    const uint32_t job_count = kqf_build_jobs(KQF_CHECKED_STEP, probe_case, first_row, row_count, jobs);
    PROBE_REQUIRE(kqf_step_is_valid(variant_index, jobs, job_count, 0, first_row, row_count),
                  "the checked step was refused");
    kqf_set_step(0, jobs, job_count, first_row, row_count);
    PROBE_CHECK_CUDA(cudaMemset(kqf.result, KQF_STALE_BYTE, KQF_RESULT_FLOATS * sizeof(float)));
    PROBE_CHECK_CUDA(cudaMemset((void *)kqf.hidden.values, KQF_STALE_BYTE, probe_cuda_array_bytes(kqf.hidden.values)));
    for (uint32_t launch = 0; launch < KQF_STEP_LAUNCHES; launch++) {
        kqf_launch(launched, &kqf.steps[0], launch);
        if (overwrite && launch == KQF_PREPARE_INPUT) {
            PROBE_CHECK_CUDA(cudaMemset(kqf.input.words, KQF_BROKEN_CODE, KQF_BROKEN_BYTES));
        }
    }
    PROBE_CHECK_CUDA(cudaGetLastError());
    PROBE_CHECK_CUDA(cudaDeviceSynchronize());
    PROBE_CHECK_CUDA(cudaMemcpy(result, kqf.result, KQF_RESULT_FLOATS * sizeof(float), cudaMemcpyDeviceToHost));
}
struct kqf_prepared {                            /* one vector as the card made it ready, by value index */
    double values[KQ_HIDDEN_SIZE];               /* what a product multiplies: dequantized, raw, or integer by scale */
    double group_sums[KQ_HIDDEN_SIZE / KQ_GROUP_WEIGHTS];       /* Q2_K: X, or the integer contract's b times x */
    int8_t integers[KQ_HIDDEN_SIZE];
    float scales[KQ_HIDDEN_SIZE / KQ_BLOCK_WEIGHTS];
};
/* Check 5 for vector `vector` of a set, made from `values`: what the card wrote against the host's quantizer and
 * what the definitions make of its codes (the order of a vector's words is written out here a second time: value
 * k of a group is at lane position k % 8 / 2, byte 2 (k % 2) + k / 8). Returns how many of the things this kind of
 * inputs writes differ, adds their number to *compared, and fills what the card holds. */
static uint64_t kqf_check_prepared(int inputs, const struct kq_vectors *vectors, size_t vector, const float *values,
                                   struct kqf_prepared *prepared, uint64_t *compared) {
    static uint32_t words[KQ_HIDDEN_SIZE / KQ_WORD_BYTES], pairs[KQ_HIDDEN_SIZE / KQ_REGISTER_HALVES];
    static uint8_t exponents[KQ_HIDDEN_SIZE / KQ_ACTIVATION_BLOCK];
    static uint8_t limbs[KQ_HIDDEN_SIZE / KQ_GROUP_WEIGHTS / KQ_LANE_MINIMUMS * KQ_LIMB_BYTES];
    static float group_sums[KQ_HIDDEN_SIZE / KQ_GROUP_WEIGHTS], scales[KQ_HIDDEN_SIZE / KQ_BLOCK_WEIGHTS];
    static uint8_t host_codes[KQ_HIDDEN_SIZE], host_exponents[KQ_HIDDEN_SIZE / KQ_ACTIVATION_BLOCK];
    static float host_scales[KQ_HIDDEN_SIZE / KQ_BLOCK_WEIGHTS];
    const void *on_card[KQF_VECTOR_PARTS] = {NULL,                vectors->words,  vectors->pairs, vectors->exponents,
                                             vectors->group_sums, vectors->scales, vectors->minimum_operand};
    void *read[KQF_VECTOR_PARTS] = {NULL, words, pairs, exponents, group_sums, scales, limbs};
    const uint32_t size = vectors->size;
    const int integer = inputs == KQ_INPUTS_INT8, raw = inputs == KQ_INPUTS_RAW, placed = inputs == KQ_INPUTS_PLACED16;
    const int with_pairs = placed || raw || inputs == KQ_INPUTS_CARRIED, q2_k = kqf.kquant == KQ_Q2_K;
    uint64_t differing = 0;
    for (uint32_t part = KQF_WORDS; part < KQF_VECTOR_PARTS; part++) {
        PROBE_CHECK_CUDA(cudaMemcpy(read[part], (const char *)on_card[part] + vector * kqf_part_bytes(part, size),
                                    kqf_part_bytes(part, size), cudaMemcpyDeviceToHost));
    }
    if (integer) {
        kq_reference_quantize_int8(values, size, (int8_t *)host_codes, host_scales);
    } else if (!raw) {
        kq_reference_quantize_e4m3(values, size, host_codes, host_exponents);
    }
    for (uint32_t index = 0; index < size; index++) {
        const uint32_t block = index / KQ_ACTIVATION_BLOCK, in_group = index % KQ_GROUP_WEIGHTS;
        const uint32_t word = kq_word_index(block, in_group % KQ_HALF_INPUTS / KQ_REGISTER_HALVES,
                                            index / KQ_GROUP_WEIGHTS % KQ_PAIR_GROUPS);
        const uint32_t byte = in_group % KQ_REGISTER_HALVES * KQ_REGISTER_HALVES + in_group / KQ_HALF_INPUTS;
        const uint32_t code = (words[word] >> (byte * KQ_BYTE_BITS)) & KQ_BYTE_MASK;
        const uint32_t half = (pairs[word * KQ_REGISTER_HALVES + byte % KQ_REGISTER_HALVES]
                               >> (byte / KQ_REGISTER_HALVES * KQ_HALF_BITS)) & KQF_HALF_MASK;
        const uint32_t carried_bits = half << KQ_HALF_BITS;
        const int power = (int)exponents[block] - KQ_REFERENCE_E8M0_BIAS;
        const int host_power = placed ? 0 : (int)host_exponents[block] - KQ_REFERENCE_E8M0_BIAS;
        const double expected = raw ? (double)values[index] : ldexp(kq_reference_e4m3(host_codes[index]), host_power);
        float carried = 0.0f;                    /* an operand pair, decoded by its type, is `expected` */
        memcpy(&carried, &carried_bits, sizeof(carried));
        const double operand = placed ? ldexp(kq_reference_half(half), KQ_BYTE_BITS)
                               : ldexp((double)carried, -KQ_CARRIED_EXPONENT);
        differing += (!raw && code != host_codes[index] ? 1u : 0u)
                     + (with_pairs && (operand != expected || signbit(operand) != signbit(expected)) ? 1u : 0u);
        prepared->integers[index] = (int8_t)code;
        prepared->values[index] = integer ? (double)(int8_t)code * (double)scales[index / KQ_BLOCK_WEIGHTS]
                                  : raw ? expected : ldexp(kq_reference_e4m3(code), power);
    }
    for (uint32_t group = 0; group < size / KQ_GROUP_WEIGHTS; group++) {
        const uint32_t block = group / KQ_PAIR_GROUPS, span = group / KQ_REFERENCE_GROUPS;
        const uint32_t low_limb = group / KQ_LANE_MINIMUMS * KQ_LIMB_BYTES + group % KQ_LANE_MINIMUMS;
        const int32_t integers = kq_reference_group_sum((const int8_t *)host_codes, group);
        const int32_t card_integers = (int8_t)limbs[low_limb + KQ_WORD_BYTES] * (KQ_LIMB_MASK + 1) + limbs[low_limb];
        const float *card_sum = &group_sums[group];
        double exact = 0.0;
        float in_order = 0.0f;
        for (uint32_t place = 0; place < KQ_GROUP_WEIGHTS; place++) {         /* raw: the order of the vector's words */
            const uint32_t offset = kq_value_offset(0, place / KQ_WORD_BYTES, place % KQ_WORD_BYTES);
            in_order += values[group * KQ_GROUP_WEIGHTS + offset];
            exact += kq_reference_e4m3(host_codes[group * KQ_GROUP_WEIGHTS + place]);
        }
        const float sum = raw ? in_order : (float)ldexp(exact, (int)host_exponents[block] - KQ_REFERENCE_E8M0_BIAS);
        differing += q2_k && (integer ? card_integers != integers : memcmp(&sum, card_sum, sizeof(float)) != 0);
        differing += !integer && !raw && group % KQ_PAIR_GROUPS == 0 && exponents[block] != host_exponents[block];
        differing += integer && group % KQ_REFERENCE_GROUPS == 0
                     && memcmp(&scales[span], &host_scales[span], sizeof(float)) != 0;
        prepared->group_sums[group] = integer ? (double)card_integers * (double)scales[span] : (double)*card_sum;
    }
    memcpy(prepared->scales, scales, sizeof(scales));
    *compared += (raw ? 0u : size) + (with_pairs ? size : 0u) + (q2_k ? size / KQ_GROUP_WEIGHTS : 0u)
                 + (integer ? size / KQ_BLOCK_WEIGHTS : raw ? 0u : size / KQ_ACTIVATION_BLOCK);
    return differing;
}
/* Checks 1, 4 and 5 for the launch of a variant that just ran (its result read back): the two checked pairs. */
static void kqf_check_against_host(const struct probe_case *probe_case, uint32_t variant_index, const float *result) {
    static const char *const names[KQF_CHECKED_PAIRS][KQ_STAGES] = {
        {"host_reference_hidden_first_row", "host_reference_output_first_row"},
        {"host_reference_hidden_last_row", "host_reference_output_last_row"}};
    static struct kqf_prepared input, hidden;
    static float values[KQ_HIDDEN_SIZE];
    const struct kqf_variant *variant = &kqf_variants[variant_index];
    const char *case_name = probe_case->name;
    const uint32_t rows = kqf_rows(probe_case);
    const int corrupt = probe_flag_given(PROBE_FLAG_CORRUPT), integer = variant->inputs == KQ_INPUTS_INT8;
    uint64_t prepared_differing = 0, prepared_compared = 0;
    for (uint32_t checked = 0; checked < KQF_CHECKED_PAIRS; checked++) {
        const uint32_t pair = checked == 0 ? KQ_EXPERTS_PER_ROW - 1u : (rows - 1u) * KQ_EXPERTS_PER_ROW;
        const uint32_t row = pair / KQ_EXPERTS_PER_ROW, slot = pair % KQ_EXPERTS_PER_ROW;
        const uint32_t expert = kqf_expert(KQF_CHECKED_STEP, kqf_overlap(probe_case), rows, row, slot);
        const double weight = (double)kqf_route_weight(KQF_CHECKED_STEP, row, slot);
        const int changed = checked == 0 && probe_flag_has_word(KQF_FLAG_BREAK, "integers");
        const float *outputs = result + (size_t)pair * KQ_HIDDEN_SIZE, *hidden_values = kqf.hidden.values;
        double error[KQ_STAGES] = {0.0, 0.0}, square[KQ_STAGES] = {0.0, 0.0};
        uint64_t contract_differing = 0;
        for (uint32_t column = 0; column < KQ_HIDDEN_SIZE; column++) { values[column] = kqf_input_value(row, column); }
        prepared_differing += kqf_check_prepared(variant->inputs, &kqf.input, row, values, &input, &prepared_compared);
        PROBE_CHECK_CUDA(cudaMemcpy(values, hidden_values + (size_t)pair * KQ_INNER_SIZE, KQ_INNER_SIZE * sizeof(float),
                                    cudaMemcpyDeviceToHost));
        for (uint32_t inner = 0; inner < KQ_INNER_SIZE; inner++) {
            const double *sums = input.group_sums;
            double gate = kq_reference_product(kqf_row(expert, 0, inner), KQ_HIDDEN_SIZE, input.values, sums);
            double up = kq_reference_product(kqf_row(expert, KQ_UP_MATRIX, inner), KQ_HIDDEN_SIZE, input.values, sums);
            gate = fmin(kq_reference_bf16(gate), (double)KQ_SWIGLU_LIMIT);
            up = fmin(fmax(kq_reference_bf16(up), -(double)KQ_SWIGLU_LIMIT), (double)KQ_SWIGLU_LIMIT);
            const double reference = kq_reference_bf16(gate / (1.0 + exp(-gate)) * up * weight);
            error[0] += ((double)values[inner] - reference) * ((double)values[inner] - reference);
            square[0] += reference * reference;
        }
        prepared_differing += kqf_check_prepared(variant->inputs, &kqf.hidden, pair, values, &hidden,
                                                 &prepared_compared);
        hidden.integers[0] = (int8_t)(hidden.integers[0] ^ (changed ? 1 : 0));        /* --break integers */
        for (uint32_t output = 0; output < KQ_HIDDEN_SIZE; output++) {
            const struct kq_reference_block *blocks = kqf_row(expert, KQ_DOWN_MATRIX, output);
            const double reference = kq_reference_product(blocks, KQ_INNER_SIZE, hidden.values, hidden.group_sums);
            const float contract = !integer ? 0.0f
                                   : kq_reference_integer_output(blocks, KQ_INNER_SIZE, KQ_DOWN_SLICES, hidden.integers,
                                                                 hidden.scales, kqf.kquant == KQ_Q2_K);
            error[1] += ((double)outputs[output] - reference) * ((double)outputs[output] - reference);
            square[1] += reference * reference;
            contract_differing += integer && memcmp(&contract, &outputs[output], sizeof(float)) != 0;
        }
        for (uint32_t stage = 0; stage < KQ_STAGES; stage++) {
            probe_check_error(case_name, variant->name, names[checked][stage], sqrt(error[stage] / square[stage]),
                              stage != 0 && variant_index == PROBE_BASELINE ? KQF_BASELINE_OUTPUT_TOLERANCE
                              : KQF_RELATIVE_TOLERANCE, corrupt && checked == 0);
        }
        if (integer) {
            probe_check_count(case_name, variant->name, "integer_contract", contract_differing, KQ_HIDDEN_SIZE,
                              changed);
        }
    }
    probe_check_count(case_name, variant->name, "prepared_vectors", prepared_differing, prepared_compared,
                      probe_flag_has_word(KQF_FLAG_BREAK, "quantization"));
}
/* --break-route: seven launches, each past one bound of what a kernel indexes by or of what the host prepares: the
 * validation every real step passes must refuse each (1 of 1); none is launched. */
static void kqf_break_route(const struct probe_case *probe_case) {
    static const char *const names[KQF_BROKEN_ROUTES] = {
        "refused_expert_past_the_stored", "refused_pair_past_the_pairs", "refused_job_of_more_columns_than_rows",
        "refused_more_jobs_than_pairs",   "refused_row_too_many",        "refused_first_row_past_the_rows",
        "refused_slot_past_the_stream"};
    for (uint32_t broken = 0; broken < KQF_BROKEN_ROUTES; broken++) {
        struct kq_job jobs[KQ_MOST_PAIRS];
        const uint32_t rows = kqf_rows(probe_case);
        uint64_t job_count = kqf_build_jobs(KQF_CHECKED_STEP, probe_case, 0, rows, jobs);
        jobs[0].expert += broken == KQF_BROKEN_EXPERT ? (uint32_t)KQ_EXPERT_COUNT : 0u;
        jobs[0].pair[KQ_MOST_ROWS - 1] += broken == KQF_BROKEN_PAIR ? rows * KQ_EXPERTS_PER_ROW : 0u;
        jobs[0].row_count += broken == KQF_BROKEN_COLUMNS ? rows : 0u;
        job_count = broken == KQF_BROKEN_JOBS ? (uint64_t)rows * KQ_EXPERTS_PER_ROW + 1u : job_count;
        const int refused = !kqf_step_is_valid(PROBE_BASELINE, jobs, job_count,
                                               broken == KQF_BROKEN_SLOT ? PROBE_STREAM_STEPS : 0u,
                                               broken == KQF_BROKEN_FIRST_ROW ? (uint64_t)KQ_MOST_ROWS : 0u,
                                               broken == KQF_BROKEN_ROWS ? KQ_MOST_ROWS + 1u : rows);
        probe_check_count(probe_case->name, kqf_variants[PROBE_BASELINE].name, names[broken], refused ? 1u : 0u, 1, 1);
    }
}
static void kqf_check_case(void *, const struct probe_case *probe_case) {
    const char *case_name = probe_case->name;
    const uint32_t rows = kqf_rows(probe_case), overlap = kqf_overlap(probe_case);
    const int break_exactness = probe_flag_has_word(KQF_FLAG_BREAK, "exactness");
    const int break_integers = probe_flag_has_word(KQF_FLAG_BREAK, "integers");
    const int overwrite = probe_flag_has_word(KQF_FLAG_BREAK, "quantization");
    const int reordered = break_exactness && overlap > 0 && rows > 1u;       /* the twin has a job of several rows */
    const size_t corrupted_expert = kqf_expert(KQF_CHECKED_STEP, overlap, rows, 0, KQ_EXPERTS_PER_ROW - 1u);
    const size_t matrix_bytes = (size_t)KQ_MATRIX_RECORDS * KQ_RECORD_BYTES(kqf.kquant);
    if (probe_flag_given(PROBE_FLAG_BREAK_ROUTE)) {
        kqf_break_route(probe_case);
        return;
    }
    for (size_t kind = 0; kind <= KQ_DOWN_MATRIX && probe_flag_given(PROBE_FLAG_CORRUPT); kind += KQ_DOWN_MATRIX) {
        PROBE_CHECK_CUDA(cudaMemset(kqf.stored + (corrupted_expert * KQ_MATRIX_KINDS + kind) * matrix_bytes,
                                    KQF_CORRUPTED_BYTE, KQF_CORRUPTED_BYTES));
    }
    for (uint32_t index = 0; index < KQF_VARIANTS; index++) {
        const struct kqf_variant *variant = &kqf_variants[index];
        const struct kqf_variant *launched = break_exactness && index == PROBE_BASELINE ? &kqf_broken_order : variant;
        /* a break run: integers the integer variant, else the baseline; exactness its twin and a variant against it */
        const int concerned = break_integers ? variant->inputs == KQ_INPUTS_INT8
                              : index == PROBE_BASELINE || (break_exactness && index == KQF_FIRST_VARIANT);
        const int measured = variant->bits == KQF_BITS_MEASURED;     /* the unit's answer: its count fails no run */
        uint64_t differing = 0;
        if (index == PROBE_BASELINE_AGAIN || !kqf_instances[index].available || (probe_breaking() && !concerned)) {
            continue;
        }
        kqf_run_checked(probe_case, index, launched, 0, rows, overwrite, kqf_result);
        if (index == PROBE_BASELINE) {
            memcpy(kqf_baseline_result, kqf_result, sizeof(kqf_result));
        } else if (variant->bits != KQF_BITS_OWN) {
            differing = kqf_rows_differing(kqf_result, 0, kqf_baseline_result, 0, rows);
            kqf_instances[index].contract_class = differing != 0 ? 'F' : kqf_instances[index].contract_class;
            probe_check_count(case_name, variant->name,
                              measured ? "unit_bits_against_baseline" : "bits_against_baseline", differing,
                              KQF_ROW_WORDS(rows), measured ? differing != 0 : reordered);
        }
        if (index == PROBE_BASELINE || variant->bits == KQF_BITS_OWN || differing != 0) {
            kqf_check_against_host(probe_case, index, kqf_result);        /* its bits are not the baseline's */
        }
        probe_digest_add(kqf_result, sizeof(kqf_result));
        differing = 0;
        for (uint32_t row = 0; row < rows && rows > 1u; row++) {
            kqf_run_checked(probe_case, index, launched, row, 1, overwrite && row == 0, kqf_alone_result);
            differing += kqf_rows_differing(kqf_result, row, kqf_alone_result, 0, 1);
        }
        if (rows > 1u) {
            probe_check_count(case_name, variant->name, "rows_exact", differing, KQF_ROW_WORDS(rows),
                              reordered && index == PROBE_BASELINE);
        }
    }
    probe_cuda_check_guards(case_name);
}

/* The figures of ROUND_1.md, package 5. c1_at_one_row: the best variant that kept contract C1's bits and does not
 * spill, against the baseline (bars for a CMP 170HX and for the PRO; us at f08's 187, 226, 71.2 and 94.5). A further
 * expert under the integer contract: the 27-expert case less the 21-expert case over the baseline's same difference
 * (a bar for a CMP 170HX only; us at f08's 27.9 and 34.1), judged by the fewer agreeing rounds of the two cases. */
static void kqf_verdicts(void *, const struct probe_case *probe_case, const struct probe_result *results,
                         double noise_ratio) {
    static const double c1_bar[KQ_TYPES][2][2] = {{{0.80, -37.0}, {0.95, -3.2}}, {{0.82, -41.0}, {0.92, -7.5}}};
    static const double further_bar[KQ_TYPES + 1][2] = {{0.65, -9.9}, {0.67, -11.1}, {PROBE_NAN, PROBE_NAN}};
    static double fewer_us[KQF_VARIANTS];
    static uint32_t fewer_signs[KQF_VARIANTS];
    const uint32_t experts = kqf_span(kqf_overlap(probe_case), kqf_rows(probe_case));
    const int on_the_pro = PROBE_SM == PROBE_SM_120A;
    const double *bar = c1_bar[kqf.kquant][on_the_pro], *further = further_bar[on_the_pro ? KQ_TYPES : kqf.kquant];
    const double baseline_more = results[PROBE_BASELINE].step.median - fewer_us[PROBE_BASELINE];
    uint32_t best = PROBE_BASELINE;
    for (uint32_t index = KQF_FIRST_VARIANT; index < KQF_VARIANTS; index++) {
        const struct probe_result *result = &results[index];
        const int kept_c1 = kqf_variants[index].bits != KQF_BITS_OWN && kqf_instances[index].contract_class == 'C';
        const uint32_t signs = result->same_sign < fewer_signs[index] ? result->same_sign : fewer_signs[index];
        if (result->mode != NULL && kept_c1 && result->spill_bytes <= 0
            && (best == PROBE_BASELINE || result->ratio < results[best].ratio)) {
            best = index;
        }
        if (result->mode != NULL && kqf_variants[index].inputs == KQ_INPUTS_INT8 && experts == KQF_MORE_EXPERTS
            && fewer_us[PROBE_BASELINE] > 0.0 && fewer_us[index] > 0.0) {
            probe_verdict("integer_further_expert", further[0], further[1],
                          (result->step.median - fewer_us[index]) / baseline_more, noise_ratio, signs, result->rounds);
        }
    }
    for (uint32_t index = 0; index < KQF_VARIANTS && experts == KQF_FEWER_EXPERTS; index++) {
        fewer_us[index] = results[index].mode != NULL ? results[index].step.median : 0.0;
        fewer_signs[index] = results[index].same_sign;
    }
    if (kqf_rows(probe_case) == 1u) {
        probe_verdict("c1_at_one_row", bar[0], bar[1], best == PROBE_BASELINE ? PROBE_NAN : results[best].ratio,
                      noise_ratio, results[best].same_sign, results[best].rounds);
    }
}

static struct probe_program kqf_program = {
    "kquant_forms",     kqf_flags,         PROBE_COUNT_OF(kqf_flags), kqf_instances,   KQF_VARIANTS,
    NULL,               kqf_configure,     kqf_open_card,             kqf_describe_case, kqf_check_case,
    kqf_prepare_step,   kqf_launch_step,   probe_cuda_mark_start,     probe_cuda_elapsed_microseconds,
    kqf_verdicts,       kqf_close_card,    NULL};

int main(int argument_count, char **arguments) { return probe_main(argument_count, arguments, &kqf_program); }
