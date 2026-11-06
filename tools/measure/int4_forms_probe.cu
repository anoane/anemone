/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * int4_forms_probe: round 1, package 6 of the speed campaign (research/speed/ROUND_1.md, "int4_forms"; ledger W1,
 * W2, W3): what the routed experts of one MoE layer cost with W4A16 weights in each operand form of
 * int4_forms_kernels.cuh, every form a named instance of one binary, interleaved by probe_harness.h. A step is every
 * launch from the bfloat16 input to the rows' sums: the input made ready, gate with up, the hidden vectors made
 * ready (hsub has no such launch), down, the rows' sums. Codes and scales are synthetic: hashes of their places.
 * INSTANCES. baseline, baseline_again: sum24, the earlier prototype's form in this round's job structure (an expert
 * once whatever its rows; the prototype had a job a row and made its input ready outside what it timed);
 * operand:hsub (class C on one card generation); operand:bytes, operand:planes (sm_80), operand:e2m3 (sm_120a): F.
 * CHECKED per case before anything is timed, on step 0, for two pairs (the first row at its last expert, the last
 * row at its first):
 *   prepared_vectors    every value's field and every group's record that a prepare launch wrote for the row's
 *                       input and the pair's hidden vector against the host's conversion or quantizer, bit for bit;
 *                       input_rule (sum24, hsub): the same for a vector of the fp16 rule's ends (above 65,504,
 *                       under 2^-24, ties), which feeds no product
 *   host_reference_*    the hidden vector (2e-3 of the rms) and the output (1e-4) against double precision from the
 *                       format's definition, fed with what the card's previous stage left
 *   integer_contract    bytes, planes: every output of down for the pair, bit for bit, against the host's integers
 *                       and the contract's fp32 steps
 *   row_sums            every row's sum, bit for bit, against the host's sum of the card's six outputs for the row
 *   rows_exact          every row launched alone against the same row in its launch: 0 words may differ. Only a
 *                       float form on a CMP 170HX may (how its instruction rounds is no rule: unit 3a): the count
 *                       is the card's answer, printed `reported`, and the class is F from that case's lines on
 *   guard_bytes, and the DIGEST of all outputs, which the batch compares between the two CMP 170HX
 * EACH SHOWN FAILING, by values and never by what a kernel indexes by: --corrupt (the first pair's gate and down
 * matrices: its two references), --break exactness (the baseline's twin, whose order of slices follows a job's
 * rows: rows_exact where rows share an expert), --break zero_point (the launches get the zero point plus one: both
 * pairs' references and prepared_vectors), --break conversion (a word of the host's copy: input_rule and
 * prepared_vectors), --break integers (an integer of the host's copy: integer_contract), --break sums (a sum of
 * the host's: row_sums), --break-route (two launches refused: past the experts, past the pairs), --break-guard.
 * VERDICTS: hsub_at_one_row, planes_at_one_row (sm_80). Measurement tool: not part of the engine.
 *   int4_forms_probe [--rows 1,2,4,6,8] [--overlap 0,50,100] [--zero-point 8|7] [--break NAME,...] and the harness's
 */
#include "probe_harness.h"
#include "int4_forms_kernels.cuh"

enum {
    I4F_STEP_LAUNCHES = 5,                       /* of every form but hsub, which has one fewer (i4f_launch) */
    I4F_K16_ORDER = 0, I4F_PLANES_ORDER, I4F_ORDERS, I4F_SCALES = I4F_ORDERS, I4F_PARTS,   /* of setup_fill_kernel */
    I4F_CHECKED_STEP = 0, I4F_CHECKED_PAIRS = 2, I4F_OVERLAP_STEP = 50, I4F_PERCENT = 100,
    I4F_RULE_VECTOR = I4_MOST_ROWS, I4F_INPUT_VECTORS,           /* input_rule's vector lies behind the rows' */
    I4F_ROW_SUMS_AT = I4_MOST_PAIRS * I4_HIDDEN_SIZE,            /* a result: the pairs' outputs, then the rows' sums */
    I4F_RESULT_FLOATS = I4F_ROW_SUMS_AT + I4_MOST_ROWS * I4_HIDDEN_SIZE,
    I4F_CORRUPTED_BYTES = 1024 * 1024, I4F_CORRUPTED_BYTE = 0x5A, I4F_STALE_BYTE = 0xFF, I4F_COLD_L2_SIZES = 4,
    I4F_FILL_THREADS = 256, I4F_FILL_BLOCKS_PER_SM = 8,
    I4F_BROKEN_EXPERT = 0, I4F_BROKEN_PAIR,      /* --break-route: what a kernel indexes by, in the order of names */
    /* the synthetic format: any code; a scale's fp16 bits are 2^-10 to under 2^-7 */
    I4F_CODE_SEED = 0x243F6A88, I4F_SCALE_SEED = 0x13198A2E, I4F_MIX_SHIFT_OUTER = 16, I4F_MIX_SHIFT_INNER = 15,
    I4F_SCALE_BITS_SMALLEST = 0x1400, I4F_SCALE_BITS_SPAN = 0x0C00, I4F_ZERO_POINT_LEAST = 7, I4F_ZERO_POINT_MOST = 8,
    /* synthetic inputs, exact in bfloat16: a group of 32 is of one of eight kinds (i4f_input_value) */
    I4F_INPUT_SEED = 0x1234567, I4F_INPUT_KINDS = 8, I4F_INPUT_PLAIN_KINDS = 3, I4F_INPUT_SPREAD = 4,
    I4F_INPUT_ZEROS = 5, I4F_INPUT_E4M3_TIES = 6, I4F_INPUT_INTEGER_TIES = 7, I4F_INPUT_TINY_SHIFT = 16,
    I4F_INPUT_STEPS = 255, I4F_SPREAD_POWERS = 18, I4F_TIE_NUMERATORS = 8, I4F_TIE_FIRST_NUMERATOR = 17,
    I4F_TIE_NUMERATOR_STEP = 2, I4F_TIE_POWERS = 14, I4F_TIE_LOWEST_POWER = -20, I4F_TIE_LARGEST_POWER = -10,
    I4F_INTEGER_TIES = 127, I4F_INTEGER_TIE_POWER = -8,
    I4F_ROUTE_SEED = 0xABCDEF, I4F_ROUTE_WEIGHT_STEPS = 32, I4F_STEP_STRIDE = 977,
    /* a small float as the host reads and makes it: its significant bits and the exponent e of its smallest normal
     * number m * 2^e, 0.5 <= m < 1: fp16 and E4M3; bf16's significant bits */
    I4F_HALF_BITS = 11, I4F_HALF_LEAST_EXPONENT = -13, I4F_HALF_SIGN = 0x8000, I4F_HALF_MASK = 0xFFFF,
    I4F_E4M3_BITS = 4, I4F_E4M3_LEAST_EXPONENT = -5, I4F_BF16_BITS = 8, I4F_ARCHITECTURES = 2, I4F_BAR_FIGURES = 2
};
#define I4F_MIX_MULTIPLIER_FIRST 0x7feb352du
#define I4F_MIX_MULTIPLIER_SECOND 0x846ca68bu
#define I4F_FLAG_ROWS "--rows"
#define I4F_FLAG_OVERLAP "--overlap"
#define I4F_FLAG_ZERO_POINT "--zero-point"
#define I4F_FLAG_BREAK "--break"
#define I4F_HIDDEN_TOLERANCE 2.0e-3              /* a bf16 cast that falls the other way is 2e-4 of the rms (f08) */
#define I4F_OUTPUT_TOLERANCE 1.0e-4              /* fp32: what the instructions' own roundings leave is far below */
#define I4F_INPUT_OFFSET 128.0f
#define I4F_INPUT_DIVISOR 64.0f
#define I4F_ROUTE_WEIGHT_DIVISOR 64.0f
#define I4F_E4M3_LARGEST 448.0
#define I4F_INTEGER_LARGEST 254.0f               /* times 2^-8: 127 at the scale of the integer ties */
#define I4F_POWER_OF_TWO_MANTISSA 0.5f           /* what frexp returns of a power of two */
#define I4F_BYTES_PER_MB 1.0e6
#define I4F_EXPERT_BYTES (I4_EXPERT_CODE_BYTES + I4_EXPERT_SCALE_BYTES)
#define I4F_ROW_FLOATS ((size_t)I4_EXPERTS_PER_ROW * I4_HIDDEN_SIZE)
#define I4F_ROW_WORDS(rows) ((uint64_t)(rows) * (I4_EXPERTS_PER_ROW + 1u) * I4_HIDDEN_SIZE)
#define I4F_BREAKS(name) probe_flag_has_word(I4F_FLAG_BREAK, name)
#define I4F_IS_WHOLE(form) ((form) == I4_BYTES || (form) == I4_PLANES)

struct i4f_variant {
    const char *name, *kernels;                  /* the instance; its two product kernels, for the registers file */
    int form;                                    /* I4_* */
    char contract_class;                         /* as claimed */
    i4_stage_launcher *gate_up, *down;           /* NULL: the form is not in this build */
    i4_prepare_launcher *prepare;
};
#define I4F_FORM(name, tag, form, contract_class)                                                 \
    {name, "i4_gate_up_" #tag "+i4_down_" #tag, form, contract_class, launch_i4_gate_up_##tag,   \
     launch_i4_down_##tag, i4_launch_prepare<form>}
#define I4F_ABSENT(name, form) {name, NULL, form, 'F', NULL, NULL, NULL}
static const struct i4f_variant i4f_variants[] = {
    I4F_FORM("baseline", sum24, I4_SUM24, 'C'), I4F_FORM("baseline_again", sum24, I4_SUM24, 'C'),
    I4F_FORM("operand:hsub", hsub, I4_HSUB, 'C'), I4F_FORM("operand:bytes", bytes, I4_BYTES, 'F'),
#if PROBE_SM == I4_SM_80
    I4F_FORM("operand:planes", planes, I4_PLANES, 'F'), I4F_ABSENT("operand:e2m3", I4_E2M3),
#else
    I4F_ABSENT("operand:planes", I4_PLANES), I4F_FORM("operand:e2m3", e2m3, I4_E2M3, 'F'),
#endif
};
#define I4F_VARIANTS PROBE_COUNT_OF(i4f_variants)
/* the twin of --break exactness, which no instance times */
static const struct i4f_variant i4f_broken_order = I4F_FORM("baseline", sum24_broken_order, I4_SUM24, 'C');
static struct probe_flag i4f_flags[] = {
    PROBE_COMMON_FLAGS,
    PROBE_CASE_NUMBERS(I4F_FLAG_ROWS, 1, I4_MOST_ROWS, "1,2,4,6,8", "rows of a step"),
    PROBE_CASE_WORDS(I4F_FLAG_OVERLAP, "0,50,100", "0,50,100", "percent of its experts a row shares with the next"),
    PROBE_NUMBER(I4F_FLAG_ZERO_POINT, I4F_ZERO_POINT_LEAST, I4F_ZERO_POINT_MOST, "8", "the layer's zero point"),
    PROBE_WORD_LIST(I4F_FLAG_BREAK, "exactness,zero_point,conversion,integers,sums", "", "show that check failing"),
};
static struct probe_instance i4f_instances[I4F_VARIANTS];
struct i4f_step {                                /* one step, as its launches take it */
    struct i4_launch gate_up, down;
    uint32_t job_count, row_count;
};
static struct {
    void *stored[I4F_PARTS];                     /* the codes in the k16 order and (sm_80) in planes', the scales */
    struct i4_job *jobs;                         /* [slot of a stream][job] */
    struct i4_vectors input, hidden;             /* of the rows of the batch and input_rule's; of the pairs */
    float *result;                               /* [pair][hidden size] of an expert for a row, then the rows' sums */
    uint32_t zero_point;                         /* the format's; a launch's is one more under --break zero_point */
    struct i4f_step steps[PROBE_STREAM_STEPS];
} i4f;
static float i4f_result[I4F_RESULT_FLOATS], i4f_alone_result[I4F_RESULT_FLOATS];
struct i4f_prepared {                            /* one vector as a product takes it, by value index */
    double values[I4_HIDDEN_SIZE];               /* what the product multiplies */
    int32_t integers[I4_HIDDEN_SIZE];            /* whole-number forms: a, and its group's scale x */
    float scales[I4_HIDDEN_SIZE / I4_GROUP_INPUTS];
};

/* ---- the synthetic format and inputs ---- */
__host__ __device__ inline uint32_t i4f_mix(uint32_t value) {
    value = (value ^ (value >> I4F_MIX_SHIFT_OUTER)) * I4F_MIX_MULTIPLIER_FIRST;
    value = (value ^ (value >> I4F_MIX_SHIFT_INNER)) * I4F_MIX_MULTIPLIER_SECOND;
    return value ^ (value >> I4F_MIX_SHIFT_OUTER);
}
/* The format's content, by definition: the code of (matrix, output, input) and the fp16 bits of the scale of
 * (matrix, output, group of 32 inputs), matrix being expert * 3 + kind: hashes of their places. */
__host__ __device__ inline uint32_t i4f_code(uint32_t matrix, uint32_t output, uint32_t input) {
    return i4f_mix(i4f_mix(I4F_CODE_SEED + matrix) + output * I4_HIDDEN_SIZE + input) & I4_NIBBLE_MASK;
}
__host__ __device__ inline uint32_t i4f_scale_bits(uint32_t matrix, uint32_t output, uint32_t group) {
    return I4F_SCALE_BITS_SMALLEST
           + i4f_mix(i4f_mix(I4F_SCALE_SEED + matrix) + output * I4_HIDDEN_SIZE + group) % I4F_SCALE_BITS_SPAN;
}
/* Any grid, any block: words 0 .. word_count - 1 of stored array `part` of every expert, packed from the
 * definition: the codes in the k16 order or in planes' (four words a lane) or the scales (two a quad), as the header
 * of int4_forms_kernels.cuh has the three. Its name keeps it out of a profile of the step's kernels. */
__global__ void setup_fill_kernel(uint32_t *stored_words, uint64_t word_count, uint32_t part) {
    const uint64_t thread_total = (uint64_t)gridDim.x * blockDim.x;
    const uint32_t lane_words = part == I4F_SCALES ? I4_LOAD_GROUPS : I4_LOAD_WORDS;
    const uint32_t load_words = (part == I4F_SCALES ? I4_QUADS : I4_LANES) * lane_words;
    for (uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; index < word_count; index += thread_total) {
        const uint32_t matrix = (uint32_t)(index / load_words / I4_MATRIX_LOADS);
        const uint32_t in_matrix = (uint32_t)(index / load_words % I4_MATRIX_LOADS);
        const uint32_t loads = matrix % I4_MATRIX_KINDS == I4_DOWN_MATRIX ? I4_STAGE_LOADS(I4_DOWN)
                                                                          : I4_STAGE_LOADS(I4_GATE_UP);
        const uint32_t word = (uint32_t)(index % lane_words), lane = (uint32_t)(index % load_words) / lane_words;
        const uint32_t quad = part == I4F_SCALES ? lane : lane / I4_QUAD_LANES, place = lane % I4_QUAD_LANES;
        const uint32_t output = in_matrix / loads * I4_TILE_OUTPUTS + quad, group = in_matrix % loads * I4_LOAD_GROUPS;
        uint32_t packed = 0;
        for (uint32_t nibble = 0; nibble < I4_WORD_NIBBLES && part != I4F_SCALES; nibble++) {
            const uint32_t half = (part == I4F_PLANES_ORDER ? word : nibble) % I4_LANE_OUTPUTS;
            const uint32_t input = part == I4F_PLANES_ORDER
                                       ? word / I4_LANE_OUTPUTS * I4_GROUP_INPUTS + place * I4_WORD_NIBBLES + nibble
                                       : word * I4_STEP_INPUTS + place * I4_WORD_HALVES + nibble / I4_WEIGHT_REGISTERS
                                             + nibble % I4_WEIGHT_REGISTERS / I4_LANE_OUTPUTS * I4_HALF_INPUTS;
            packed |= i4f_code(matrix, output + half * I4_HALF_OUTPUTS, group * I4_GROUP_INPUTS + input)
                      << (nibble * I4_NIBBLE_BITS);
        }
        if (part == I4F_SCALES) {                /* word G: group G of the load, outputs g and g + 8 */
            packed = i4f_scale_bits(matrix, output, group + word)
                     | (i4f_scale_bits(matrix, output + I4_HALF_OUTPUTS, group + word) << I4_HALF_BITS);
        }
        stored_words[index] = packed;
    }
}
/* One element of a row's input, exact in bfloat16. Its group of 32 is plain (multiples of 1/64 in -2 .. 2, over
 * 2^0 .. 2^2), tiny (plain over 2^16: under the reference's floor of 1e-4), spread (powers of two over 2^17), zeros
 * of both signs, E4M3 ties (its first value the largest code at its scale, 448 * 2^-10, the others half way between
 * two neighbouring codes) or integer ties (its first value 127 * 2^-7, the others a whole number and a half of
 * 2^-7). input_rule's vector: the ends of THE RULE, each with either sign. */
static float i4f_input_value(uint32_t row, uint32_t column) {
    static const float rule_ends[] = {65504.0f,   65519.0f,   65520.0f,   1.0e30f,    0x1.0p-24f,   0x1.0p-25f,
                                      0x1.8p-25f, 0x1.0p-26f, 0x1.8p-24f, 0x1.4p-23f, 0x1.002p0f,   0x1.006p0f,
                                      0x1.ffep-15f, 0x1.ffcp-15f, 0x1.8p-20f, 3.0f};
    const uint32_t seed = I4F_INPUT_SEED + row * I4_HIDDEN_SIZE, bits = i4f_mix(seed + column);
    const uint32_t kind = i4f_mix(~seed + column / I4_GROUP_INPUTS) % I4F_INPUT_KINDS;
    const int first = column % I4_GROUP_INPUTS == 0;
    const float sign = ((bits >> I4_BYTE_BITS) & 1u) != 0u ? -1.0f : 1.0f;
    if (row == I4F_RULE_VECTOR) { return sign * rule_ends[column % PROBE_COUNT_OF(rule_ends)]; }
    if (kind == I4F_INPUT_SPREAD) { return ldexpf(sign, -(int)(bits % I4F_SPREAD_POWERS)); }
    if (kind == I4F_INPUT_ZEROS) { return sign * 0.0f; }
    if (kind == I4F_INPUT_E4M3_TIES && first) { return ldexpf(sign * (float)I4F_E4M3_LARGEST, I4F_TIE_LARGEST_POWER); }
    if (kind == I4F_INPUT_E4M3_TIES) {
        return ldexpf(sign * (float)(I4F_TIE_FIRST_NUMERATOR + bits % I4F_TIE_NUMERATORS * I4F_TIE_NUMERATOR_STEP),
                      I4F_TIE_LOWEST_POWER + (int)(bits / I4F_TIE_NUMERATORS % I4F_TIE_POWERS));
    }
    if (kind == I4F_INPUT_INTEGER_TIES) {
        return ldexpf(sign * (first ? I4F_INTEGER_LARGEST : (float)(bits % I4F_INTEGER_TIES * I4_WORD_HALVES + 1u)),
                      I4F_INTEGER_TIE_POWER);
    }
    return ((float)(bits & I4F_INPUT_STEPS) - I4F_INPUT_OFFSET) / I4F_INPUT_DIVISOR
           / (float)(1u << (kind < I4F_INPUT_PLAIN_KINDS ? kind : (uint32_t)I4F_INPUT_TINY_SHIFT));
}

/* ---- routes, jobs and the validation of a step (check 10) ---- */
static uint32_t i4f_rows(const struct probe_case *probe_case) {
    return (uint32_t)probe_case_value(probe_case, I4F_FLAG_ROWS);
}
static uint32_t i4f_overlap(const struct probe_case *probe_case) {
    return (uint32_t)probe_case_value(probe_case, I4F_FLAG_OVERLAP) * I4F_OVERLAP_STEP;
}
static uint32_t i4f_span(uint32_t overlap, uint32_t rows) {      /* the experts a step of `rows` rows reads */
    return I4_EXPERTS_PER_ROW + (rows - 1u) * (I4_EXPERTS_PER_ROW * (I4F_PERCENT - overlap) / I4F_PERCENT);
}
/* The synthetic routing. Row r takes six consecutive experts from the 6r-th (overlap 0), the 3r-th (50) or the
 * first (100) of its step's span, and the spans of consecutive steps follow one another around the ring of
 * experts: between two reads of an expert a card reads 80 others or more (CONVENTIONS 3.5; i4f_open_card holds
 * that to four L2 sizes). span_rows is the case's rows also where one row is launched alone: it keeps its experts. */
static uint32_t i4f_expert(uint64_t step, uint32_t overlap, uint32_t span_rows, uint32_t row, uint32_t slot) {
    const uint32_t stride = I4_EXPERTS_PER_ROW * (I4F_PERCENT - overlap) / I4F_PERCENT;
    return (uint32_t)((step * i4f_span(overlap, span_rows) + row * stride + slot) % I4_EXPERT_COUNT);
}
static float i4f_route_weight(uint64_t step, uint32_t row, uint32_t slot) {
    const uint32_t place = (uint32_t)step * I4F_STEP_STRIDE + row * I4_EXPERTS_PER_ROW + slot;
    return (float)(1u + i4f_mix(~(uint32_t)I4F_ROUTE_SEED + place) % I4F_ROUTE_WEIGHT_STEPS) / I4F_ROUTE_WEIGHT_DIVISOR;
}
/* The jobs of rows first_row .. first_row + row_count - 1 (at most I4_MOST_ROWS), which fill the batch from its
 * slot 0: one per expert, its columns in row order. */
static uint32_t i4f_build_jobs(uint64_t step, const struct probe_case *probe_case, uint32_t first_row,
                               uint32_t row_count, struct i4_job *jobs) {
    uint32_t job_count = 0;
    for (uint32_t pair = 0; pair < row_count * I4_EXPERTS_PER_ROW; pair++) {
        const uint32_t row = first_row + pair / I4_EXPERTS_PER_ROW, slot = pair % I4_EXPERTS_PER_ROW;
        const uint32_t expert = i4f_expert(step, i4f_overlap(probe_case), i4f_rows(probe_case), row, slot);
        uint32_t job_index = 0;
        while (job_index < job_count && jobs[job_index].expert != expert) { job_index++; }
        struct i4_job *job = &jobs[job_index];
        for (uint32_t column = 0; job_index == job_count && column < I4_MOST_ROWS; column++) {
            job->expert = expert;
            job->row_count = 0;
            job->pair[column] = pair;            /* a column not in use reads what column 0 reads */
            job->route_weight[column] = 0.0f;
        }
        job_count += job_index == job_count ? 1u : 0u;
        job->pair[job->row_count] = pair;
        job->route_weight[job->row_count++] = i4f_route_weight(step, row, slot);
    }
    return job_count;
}
/* Check 10 for one prepare launch: vectors first .. first + count - 1 of a set are read, vectors 0 .. count - 1 of
 * its operand and its records written (the hidden values are what gate with up stores: the same range). */
static void i4f_validate_prepare(int *valid, const struct i4_vectors *set, uint64_t first, uint64_t count) {
    const char *kernel = "i4_prepare_kernel";
    probe_launch_geometry(valid, kernel, NULL, count, 1, 1, set->size / I4_GROUP_INPUTS, 0);
    probe_launch_range(valid, kernel, "values, in vectors", first, count, set->size * sizeof(float),
                       probe_cuda_array_bytes(set->values));
    probe_launch_range(valid, kernel, "operand words, in vectors", 0, count,
                       set->size / I4_WORD_VALUES * sizeof(uint32_t), probe_cuda_array_bytes(set->operand));
    probe_launch_range(valid, kernel, "group records, in vectors", 0, count,
                       set->size / I4_GROUP_INPUTS * sizeof(struct i4_group), probe_cuda_array_bytes(set->groups));
}
/* Check 10: a step of these jobs with one variant's kernels, prepared in a stream's slot, stays inside every array
 * as allocated, inside the card's limits and the variant's launch bound. 1: it does; 0: it printed why not, and no
 * job was read past the first value out of its range. */
static int i4f_step_is_valid(uint32_t variant_index, const struct i4_job *jobs, uint64_t job_count, uint64_t slot,
                             uint64_t first_row, uint64_t row_count) {
    const int planes = i4f_variants[variant_index].form == I4_PLANES;
    const char *kernel = i4f_variants[variant_index].name;
    const uint64_t pair_count = row_count * I4_EXPERTS_PER_ROW, shared = I4_LANE_SUMS * sizeof(float);
    int valid = 1;
    probe_launch_value(&valid, kernel, "rows", row_count, 1, I4_MOST_ROWS);
    probe_launch_value(&valid, kernel, "the first row with the rows", first_row + row_count, 1, I4_MOST_ROWS);
    probe_launch_value(&valid, kernel, "the slot of a stream", slot, 0, PROBE_STREAM_STEPS - 1u);
    probe_launch_value(&valid, kernel, "jobs", job_count, 1, valid ? pair_count : 1);
    for (uint64_t job = 0; valid && job < job_count; job++) {
        probe_launch_value(&valid, kernel, "a job's expert", jobs[job].expert, 0, I4_EXPERT_COUNT - 1u);
        probe_launch_value(&valid, kernel, "a job's columns", jobs[job].row_count, 1, row_count);
        for (uint32_t column = 0; column < I4_MOST_ROWS; column++) {
            probe_launch_value(&valid, kernel, "a column's pair", jobs[job].pair[column], 0, pair_count - 1u);
        }
    }
    if (!valid) { return 0; }
    probe_launch_geometry(&valid, kernel, &i4f_instances[variant_index], I4_STAGE_TILES(I4_GATE_UP), job_count, 1,
                          I4_STAGE_THREADS(I4_GATE_UP), I4_STAGE_THREADS(I4_GATE_UP) * shared);
    probe_launch_geometry(&valid, kernel, NULL, I4_STAGE_TILES(I4_DOWN), job_count, 1, I4_STAGE_THREADS(I4_DOWN),
                          I4_STAGE_THREADS(I4_DOWN) * shared);
    probe_launch_geometry(&valid, "i4_row_sum_kernel", NULL, row_count, 1, 1, I4_ROW_SUM_THREADS, 0);
    i4f_validate_prepare(&valid, &i4f.input, first_row, row_count);
    i4f_validate_prepare(&valid, &i4f.hidden, 0, pair_count);
    probe_launch_range(&valid, kernel, "codes, in experts", 0, I4_EXPERT_COUNT, I4_EXPERT_CODE_BYTES,
                       probe_cuda_array_bytes(i4f.stored[planes ? I4F_PLANES_ORDER : I4F_K16_ORDER]));
    probe_launch_range(&valid, kernel, "scales, in experts", 0, I4_EXPERT_COUNT, I4_EXPERT_SCALE_BYTES,
                       probe_cuda_array_bytes(i4f.stored[I4F_SCALES]));
    probe_launch_range(&valid, kernel, "jobs, in jobs", slot * I4_MOST_PAIRS, job_count, sizeof(struct i4_job),
                       probe_cuda_array_bytes(i4f.jobs));
    probe_launch_range(&valid, kernel, "result, in floats", 0, I4F_RESULT_FLOATS, sizeof(float),
                       probe_cuda_array_bytes(i4f.result));
    return valid;
}
/* A validated step into a stream's slot: its jobs to the card, its two launches' arguments. */
static void i4f_set_step(int form, uint32_t slot, const struct i4_job *jobs, uint32_t job_count, uint32_t first_row,
                         uint32_t row_count) {
    struct i4f_step *step = &i4f.steps[slot];
    struct i4_job *slot_jobs = i4f.jobs + (size_t)slot * I4_MOST_PAIRS;
    const struct i4_launch launch = {
        (const uint4 *)i4f.stored[form == I4_PLANES ? I4F_PLANES_ORDER : I4F_K16_ORDER],
        (const uint2 *)i4f.stored[I4F_SCALES], i4f.input, slot_jobs, (float *)i4f.hidden.values,
        (uint16_t *)i4f.hidden.operand, i4f.result, i4f.zero_point + (I4F_BREAKS("zero_point") ? 1u : 0u)};
    PROBE_INDEX_CHECK(slot, PROBE_STREAM_STEPS);
    PROBE_INDEX_CHECK(first_row + row_count - 1u, I4_MOST_ROWS);
    PROBE_CHECK_CUDA(cudaMemcpy(slot_jobs, jobs, job_count * sizeof(struct i4_job), cudaMemcpyHostToDevice));
    PROBE_CHECK_CUDA(cudaDeviceSynchronize());   /* a step starts on an idle card, as the earlier probe's */
    step->gate_up = step->down = launch;
    step->gate_up.vectors.values = i4f.input.values + (size_t)first_row * I4_HIDDEN_SIZE;
    step->down.vectors = i4f.hidden;
    step->job_count = job_count;
    step->row_count = row_count;
}

/* ---- launches ---- */
/* The launches of a step that i4f_set_step prepared: the input made ready, gate with up, the hidden vectors made
 * ready (hsub's come ready from gate with up), down, the rows' sums; returns their number. */
static uint32_t i4f_launch(const struct i4f_variant *variant, const struct i4f_step *step) {
    const uint32_t zero_point = step->gate_up.zero_point, hidden_ready = variant->form == I4_HSUB ? 1u : 0u;
    variant->prepare(&step->gate_up.vectors, step->row_count, zero_point);
    PROBE_AFTER_LAUNCH();
    variant->gate_up(&step->gate_up, step->job_count);
    PROBE_AFTER_LAUNCH();
    if (!hidden_ready) {
        variant->prepare(&step->down.vectors, step->row_count * I4_EXPERTS_PER_ROW, zero_point);
        PROBE_AFTER_LAUNCH();
    }
    variant->down(&step->down, step->job_count);
    PROBE_AFTER_LAUNCH();
    i4_row_sum_kernel<<<step->row_count, I4_ROW_SUM_THREADS>>>(i4f.result, i4f.result + I4F_ROW_SUMS_AT);
    PROBE_AFTER_LAUNCH();
    return I4F_STEP_LAUNCHES - hidden_ready;
}
static int i4f_prepare_step(void *, const struct probe_case *probe_case, uint32_t instance, uint64_t step,
                            uint32_t slot) {
    struct i4_job jobs[I4_MOST_PAIRS];
    const uint32_t rows = i4f_rows(probe_case), job_count = i4f_build_jobs(step, probe_case, 0, rows, jobs);
    if (!i4f_step_is_valid(instance, jobs, job_count, slot, 0, rows)) { return 0; }
    i4f_set_step(i4f_variants[instance].form, slot, jobs, job_count, 0, rows);
    return 1;
}
static uint32_t i4f_launch_step(void *, const struct probe_case *, uint32_t instance, uint32_t slot) {
    return i4f_launch(&i4f_variants[instance], &i4f.steps[slot]);
}

/* ---- the card ---- */
static void *i4f_allocate(uint64_t bytes) {      /* zeroed: no kernel reads a byte nothing wrote */
    void *array = probe_cuda_allocate(bytes);
    PROBE_CHECK_CUDA(cudaMemset(array, 0, bytes));
    return array;
}
static void i4f_allocate_vectors(struct i4_vectors *vectors, uint64_t count, uint32_t size, uint32_t pairs_per_vector) {
    vectors->values = (float *)i4f_allocate(count * size * sizeof(float));
    vectors->operand = (uint32_t *)i4f_allocate(count * size / I4_WORD_VALUES * sizeof(uint32_t));
    vectors->groups = (struct i4_group *)i4f_allocate(count * size / I4_GROUP_INPUTS * sizeof(struct i4_group));
    vectors->size = size;
    vectors->pairs_per_vector = pairs_per_vector;
}
static int i4f_open_card(void *, struct probe_card_facts *facts) {
    static float inputs[I4F_INPUT_VECTORS * I4_HIDDEN_SIZE];
    if (!probe_cuda_open(facts)) { return 0; }
    const uint32_t fill_blocks = (uint32_t)facts->sm_count * I4F_FILL_BLOCKS_PER_SM;
    PROBE_REQUIRE((I4_EXPERT_COUNT - I4_MOST_PAIRS) * I4F_EXPERT_BYTES >= I4F_COLD_L2_SIZES * facts->l2_bytes,
                  "between two reads of an expert a card must read four times its L2 (CONVENTIONS 3.5)");
    for (uint32_t part = 0; part < I4F_PARTS; part++) {
        const uint64_t bytes = I4_EXPERT_COUNT * (part == I4F_SCALES ? I4_EXPERT_SCALE_BYTES : I4_EXPERT_CODE_BYTES);
        int valid = 1;
        if (part == I4F_PLANES_ORDER && !i4f_instances[probe_instance_index("operand:planes")].available) { continue; }
        i4f.stored[part] = probe_cuda_allocate(bytes);
        probe_launch_geometry(&valid, "setup_fill_kernel", NULL, fill_blocks, 1, 1, I4F_FILL_THREADS, 0);
        probe_launch_range(&valid, "setup_fill_kernel", "stored, in words", 0, bytes / sizeof(uint32_t),
                           sizeof(uint32_t), probe_cuda_array_bytes(i4f.stored[part]));
        PROBE_REQUIRE(valid, "the launch that fills a stored array was refused");
        setup_fill_kernel<<<fill_blocks, I4F_FILL_THREADS>>>((uint32_t *)i4f.stored[part], bytes / sizeof(uint32_t),
                                                             part);
    }
    i4f.jobs = (struct i4_job *)i4f_allocate(PROBE_STREAM_STEPS * I4_MOST_PAIRS * sizeof(struct i4_job));
    i4f.result = (float *)i4f_allocate(I4F_RESULT_FLOATS * sizeof(float));
    i4f_allocate_vectors(&i4f.input, I4F_INPUT_VECTORS, I4_HIDDEN_SIZE, I4_EXPERTS_PER_ROW);
    i4f_allocate_vectors(&i4f.hidden, I4_MOST_PAIRS, I4_INNER_SIZE, 1);
    for (uint32_t index = 0; index < I4F_INPUT_VECTORS * I4_HIDDEN_SIZE; index++) {
        inputs[index] = i4f_input_value(index / I4_HIDDEN_SIZE, index % I4_HIDDEN_SIZE);
    }
    PROBE_CHECK_CUDA(cudaMemcpy((void *)i4f.input.values, inputs, sizeof(inputs), cudaMemcpyHostToDevice));
    PROBE_CHECK_CUDA(cudaGetLastError());
    PROBE_CHECK_CUDA(cudaDeviceSynchronize());
    facts->ring_bytes = I4_EXPERT_COUNT * I4F_EXPERT_BYTES;
    return 1;
}
static void i4f_close_card(void *) { probe_cuda_close(); }
static void i4f_describe_case(void *, const struct probe_case *probe_case, struct probe_case_facts *facts) {
    facts->format = "int4";
    facts->part = "experts";
    facts->rows = i4f_rows(probe_case);
    facts->overlap = i4f_overlap(probe_case);
    facts->experts = i4f_span(i4f_overlap(probe_case), i4f_rows(probe_case));
    facts->megabytes = (double)facts->experts * (double)I4F_EXPERT_BYTES / I4F_BYTES_PER_MB;
}
static int i4f_configure(void *) {
    i4f.zero_point = (uint32_t)probe_flag_number(I4F_FLAG_ZERO_POINT, 0);
    probe_run.breaking |= (uint32_t)probe_flag_given(I4F_FLAG_BREAK);     /* as a --break-* switch of the harness */
    for (uint32_t index = 0; index < I4F_VARIANTS; index++) {
        const struct i4f_variant *variant = &i4f_variants[index];
        const struct probe_instance instance = {variant->name, variant->kernels, variant->contract_class,
                                                variant->gate_up != NULL, I4_STAGE_THREADS(I4_GATE_UP), 0, 0, 0, 0};
        i4f_instances[index] = instance;
    }
    return 1;
}

/* ---- the host's side of a prepared vector: THE RULE and the two quantizers, from their definitions ---- */
/* A magnitude as the code of a small float of `bits` significant bits whose smallest normal number is 2^(least - 1):
 * the nearest whole number of units, ties to even (the default rounding mode), 2^(bits - 1) codes a binade, the
 * numbers under the smallest normal one in its units: fp16 (11, -13), E4M3 (4, -5). And what a code stands for. */
static uint32_t i4f_float_code(double magnitude, int bits, int least) {
    int exponent = 0;
    (void)frexp(magnitude, &exponent);
    exponent = magnitude == 0.0 || exponent < least ? least : exponent;
    return (uint32_t)((exponent - least) * (1 << (bits - 1)) + (int)nearbyint(ldexp(magnitude, bits - exponent)));
}
static double i4f_float_value(uint32_t code, int bits, int least) {
    const int binade = (int)(code >> (bits - 1)), steps = (int)(code & ((1u << (bits - 1)) - 1u));
    return ldexp((double)(binade == 0 ? steps : steps + (1 << (bits - 1))),
                 (binade == 0 ? 1 : binade) + least - 1 - bits);
}
static uint32_t i4f_half_bits(float value) {     /* THE RULE */
    return i4f_float_code(fmin(fabs((double)value), (double)I4_HALF_LARGEST), I4F_HALF_BITS, I4F_HALF_LEAST_EXPONENT)
           | (signbit(value) ? (uint32_t)I4F_HALF_SIGN : 0u);
}
static double i4f_half_value(uint32_t bits) {
    const double magnitude = i4f_float_value(bits & (I4F_HALF_SIGN - 1u), I4F_HALF_BITS, I4F_HALF_LEAST_EXPONENT);
    return (bits & I4F_HALF_SIGN) != 0 ? -magnitude : magnitude;
}
static double i4f_bf16(double value) {           /* to bf16's eight significant bits, to nearest, ties to even */
    int exponent = 0;
    const double mantissa = frexp(value, &exponent);
    return ldexp(nearbyint(ldexp(mantissa, I4F_BF16_BITS)), exponent - I4F_BF16_BITS);
}
/* Where value `index` of a group lies in a vector's operand words, by form: the word and the shift of its field
 * (planes: of its high nibble; the low one is I4_LOAD_GROUPS words on). The orders of int4_forms_kernels.cuh,
 * written a second time. */
static uint32_t i4f_operand_place(int form, uint32_t group, uint32_t index, uint32_t *shift) {
    const uint32_t step = group % I4_LOAD_GROUPS * I4_GROUP_STEPS + index / I4_STEP_INPUTS;
    const uint32_t in_step = index % I4_STEP_INPUTS, place = in_step % I4_HALF_INPUTS / I4_WORD_HALVES;
    const uint32_t upper = in_step / I4_HALF_INPUTS, half = in_step % I4_WORD_HALVES;
    const uint32_t word = (group / I4_LOAD_GROUPS * I4_QUAD_LANES + place) * I4_LOAD_STEPS + step;
    if (form == I4_PLANES) {
        *shift = index % I4_WORD_NIBBLES * I4_NIBBLE_BITS;
        return (group * I4_QUAD_LANES + index / I4_WORD_NIBBLES) * I4_LOAD_WORDS + group % I4_LOAD_GROUPS;
    }
    *shift = I4_IS_FP16(form) ? half * I4_HALF_BITS : (half * I4_WORD_HALVES + upper) * I4_BYTE_BITS;
    return I4_IS_FP16(form) ? word * I4_WORD_HALVES + upper : word;
}
/* Checks 5 and 1 for vector `vector` of a set, made ready from `values` by a launch of `form`: every value's field
 * in the card's operand words and every group's record against the host's, made from the definitions (the zero
 * halves of planes' registers are held by integer_contract); returns how many differ, adds their number to *compared
 * and fills what a product multiplies. flipped (--break conversion): the host's first field is changed. values NULL
 * (hsub's hidden vector, which gate with up stored as fp16): the card's fields are taken as they are. */
static uint64_t i4f_check_prepared(int form, const struct i4_vectors *set, size_t vector, const float *values,
                                   int flipped, struct i4f_prepared *prepared, uint64_t *compared) {
    static uint32_t words[I4_HIDDEN_SIZE / I4_WORD_VALUES];
    static struct i4_group records[I4_HIDDEN_SIZE / I4_GROUP_INPUTS];
    const uint32_t size = set->size, word_count = size / I4_WORD_VALUES, group_count = size / I4_GROUP_INPUTS;
    const uint32_t mask = I4_IS_FP16(form) ? (uint32_t)I4F_HALF_MASK
                          : form == I4_PLANES ? (uint32_t)I4_NIBBLE_MASK : (uint32_t)I4_BYTE_MASK;
    uint64_t differing = 0;
    PROBE_INDEX_CHECK(vector, set == &i4f.input ? (size_t)I4F_INPUT_VECTORS : (size_t)I4_MOST_PAIRS);
    PROBE_CHECK_CUDA(cudaMemcpy(words, set->operand + vector * word_count, word_count * sizeof(uint32_t),
                                cudaMemcpyDeviceToHost));
    PROBE_CHECK_CUDA(cudaMemcpy(records, set->groups + vector * group_count, group_count * sizeof(struct i4_group),
                                cudaMemcpyDeviceToHost));
    for (uint32_t first = 0; first < size; first += I4_GROUP_INPUTS) {
        struct i4_group record = {0, 1.0f};
        float largest = 0.0f, sum = 0.0f, inverse = 0.0f;
        int32_t integers = 0;
        int exponent = 0;
        for (uint32_t index = first; values != NULL && index < first + I4_GROUP_INPUTS; index++) {
            largest = fabsf(values[index]) > fabsf(largest) ? values[index] : largest;
        }
        if (I4F_IS_WHOLE(form)) {                /* the rule unit 5c priced: the largest magnitude becomes -127 */
            inverse = largest == 0.0f ? 0.0f : I4_INT8_LARGEST_TO / largest;
            record.scale = largest == 0.0f ? 0.0f : 1.0f / inverse;
        } else if (form == I4_E2M3) {            /* the reference's act_quant: a power of two at or above amax / 448 */
            const float mantissa = frexpf(fmaxf(fabsf(largest), I4_AMAX_FLOOR) * I4_E4M3_MAXIMUM_INVERSE, &exponent);
            exponent -= mantissa == I4F_POWER_OF_TWO_MANTISSA ? 1 : 0;
            record.scale = ldexpf(1.0f, exponent + I4_E2M3_DENOMINATOR_BITS);
        }
        for (uint32_t index = first; index < first + I4_GROUP_INPUTS; index++) {
            uint32_t shift = 0;
            const uint32_t at = i4f_operand_place(form, first / I4_GROUP_INPUTS, index - first, &shift);
            const uint32_t low = form == I4_PLANES ? (words[at + I4_LOAD_GROUPS] >> shift) & mask : 0u;
            const uint32_t card = (((words[at] >> shift) & mask) << (form == I4_PLANES ? I4_NIBBLE_BITS : 0)) | low;
            uint32_t field = card;
            if (I4_IS_FP16(form)) {
                field = values == NULL ? card : i4f_half_bits(values[index]);
                prepared->values[index] = i4f_half_value(field);
                sum += (float)prepared->values[index];
            } else if (form == I4_E2M3) {        /* the nearest E4M3 code of the value over the scale, at most 448 */
                const double scaled = ldexp((double)values[index], -exponent);
                field = i4f_float_code(fmin(fabs(scaled), I4F_E4M3_LARGEST), I4F_E4M3_BITS, I4F_E4M3_LEAST_EXPONENT);
                const double code_value = i4f_float_value(field, I4F_E4M3_BITS, I4F_E4M3_LEAST_EXPONENT);
                field |= signbit(scaled) ? (uint32_t)I4_E4M3_SIGN : 0u;
                prepared->values[index] = ldexp(copysign(code_value, scaled), exponent);
                sum += (float)ldexp(copysign(code_value, scaled), -I4_BYTE_BITS);
            } else {
                const float nearest = nearbyintf(inverse * values[index]);
                prepared->integers[index] = nearest > (float)I4_INT8_MOST ? (int32_t)I4_INT8_MOST : (int32_t)nearest;
                field = (uint32_t)prepared->integers[index] & I4_BYTE_MASK;
                integers += prepared->integers[index];
                prepared->values[index] = (double)prepared->integers[index] * (double)record.scale;
            }
            differing += (field ^ (flipped && index == 0 ? 1u : 0u)) != card ? 1u : 0u;
        }
        if (form == I4_SUM24) { sum = (float)(I4_SUM24_ADDEND + i4f.zero_point) * sum; }
        if (form == I4_E2M3) { sum = -((float)i4f.zero_point * sum) * I4_E2M3_START_FACTOR; }
        if (!I4F_IS_WHOLE(form)) { memcpy(&record.start, &sum, sizeof(sum)); }       /* an fp32's bits */
        if (I4F_IS_WHOLE(form)) { record.start = (uint32_t)(-(int32_t)i4f.zero_point * integers); }
        prepared->scales[first / I4_GROUP_INPUTS] = record.scale;
        differing += form != I4_HSUB && memcmp(&record, &records[first / I4_GROUP_INPUTS], sizeof(record)) != 0;
    }
    *compared += values == NULL ? 0u : size + (form != I4_HSUB ? group_count : 0u);
    return differing;
}

/* ---- checks ---- */
/* Output `output` of matrix `matrix` (expert * 3 + kind) of `inputs` inputs times a vector, from the format's
 * definition, in double: a weight is (code - zero point) * its group's fp16 scale. The stored bytes are not read: a
 * pack, an order or a kernel's zero point that is not the definition's shows here. *contract: the whole-number
 * contract's fp32 output, bit for bit: per slice from +0, per group in ascending order one fused multiply-add of
 * the group's exact integer by (its scale times the input's scale, one fp32 product); the slices added in order. */
static double i4f_reference_product(uint32_t matrix, uint32_t output, uint32_t inputs,
                                    const struct i4f_prepared *vector, float *contract) {
    const uint32_t slice_groups = inputs / I4_GROUP_INPUTS / I4_SLICES;
    double sum = 0.0;
    float total = 0.0f;
    *contract = 0.0f;
    for (uint32_t group = 0; group < inputs / I4_GROUP_INPUTS; group++) {
        const double scale = i4f_half_value(i4f_scale_bits(matrix, output, group));
        double group_sum = 0.0;
        int32_t integer = 0;
        for (uint32_t input = group * I4_GROUP_INPUTS; input < (group + 1u) * I4_GROUP_INPUTS; input++) {
            const int32_t weight = (int32_t)i4f_code(matrix, output, input) - (int32_t)i4f.zero_point;
            group_sum += (double)weight * vector->values[input];
            integer += weight * vector->integers[input];
        }
        sum += group_sum * scale;
        total = fmaf((float)integer, (float)scale * vector->scales[group], total);
        if ((group + 1u) % slice_groups == 0) {
            *contract += total;
            total = 0.0f;
        }
    }
    return sum;
}
/* How many of `count` words of two results differ, bit for bit. */
static uint64_t i4f_words_differing(const float *first, const float *second, size_t count) {
    uint64_t differing = 0;
    for (size_t at = 0; at < count; at++) { differing += memcmp(first + at, second + at, sizeof(float)) != 0; }
    return differing;
}
/* The checked step of rows first_row .. with one variant's kernels, outside the timing: validated, what it stores
 * made stale first (a word a launch does not write cannot pass for a result), the result read back. */
static void i4f_run_checked(const struct probe_case *probe_case, uint32_t variant_index,
                            const struct i4f_variant *launched, uint32_t first_row, uint32_t row_count, float *result) {
    void *const written[] = {i4f.result,       (void *)i4f.hidden.values, i4f.hidden.operand,
                             i4f.hidden.groups, i4f.input.operand,         i4f.input.groups};
    struct i4_job jobs[I4_MOST_PAIRS];
    const uint32_t job_count = i4f_build_jobs(I4F_CHECKED_STEP, probe_case, first_row, row_count, jobs);
    PROBE_REQUIRE(i4f_step_is_valid(variant_index, jobs, job_count, 0, first_row, row_count),
                  "the checked step was refused");
    i4f_set_step(launched->form, 0, jobs, job_count, first_row, row_count);
    for (uint32_t array = 0; array < PROBE_COUNT_OF(written); array++) {
        PROBE_CHECK_CUDA(cudaMemset(written[array], I4F_STALE_BYTE, probe_cuda_array_bytes(written[array])));
    }
    (void)i4f_launch(launched, &i4f.steps[0]);
    PROBE_CHECK_CUDA(cudaGetLastError());
    PROBE_CHECK_CUDA(cudaDeviceSynchronize());
    PROBE_CHECK_CUDA(cudaMemcpy(result, i4f.result, I4F_RESULT_FLOATS * sizeof(float), cudaMemcpyDeviceToHost));
}
/* Checks 1, 4 and 5 for the launch of a variant that just ran (its result read back): the two checked pairs; then,
 * with_rule, input_rule: the variant's prepare kernel alone on the vector of the rule's ends; then row_sums. */
static void i4f_check_against_host(const struct probe_case *probe_case, uint32_t variant_index, const float *result,
                                   int with_rule) {
    static const char *const names[I4F_CHECKED_PAIRS][I4_STAGES] = {
        {"host_reference_hidden_first_row", "host_reference_output_first_row"},
        {"host_reference_hidden_last_row", "host_reference_output_last_row"}};
    static const double limits[I4_STAGES] = {I4F_HIDDEN_TOLERANCE, I4F_OUTPUT_TOLERANCE};
    static struct i4f_prepared input, hidden;
    static float values[I4_HIDDEN_SIZE];
    const struct i4f_variant *variant = &i4f_variants[variant_index];
    const char *case_name = probe_case->name;
    const uint32_t rows = i4f_rows(probe_case);
    const int form = variant->form, flipped = I4F_BREAKS("conversion"), off_by_one = I4F_BREAKS("zero_point");
    const int negated = I4F_BREAKS("sums");      /* --break sums: the host's first sum gets the other sign */
    uint64_t differing = 0, compared = 0, sums_differing = 0;
    for (uint32_t checked = 0; checked < I4F_CHECKED_PAIRS + (with_rule ? 1u : 0u); checked++) {
        const uint32_t pair = checked == 0 ? I4_EXPERTS_PER_ROW - 1u : (rows - 1u) * I4_EXPERTS_PER_ROW;
        const uint32_t row = checked == I4F_CHECKED_PAIRS ? (uint32_t)I4F_RULE_VECTOR : pair / I4_EXPERTS_PER_ROW;
        const uint32_t slot = pair % I4_EXPERTS_PER_ROW;
        const uint32_t expert = i4f_expert(I4F_CHECKED_STEP, i4f_overlap(probe_case), rows, row, slot);
        const uint32_t matrix = expert * I4_MATRIX_KINDS;
        const double weight = (double)i4f_route_weight(I4F_CHECKED_STEP, row, slot);
        const int changed = checked == 0 && I4F_BREAKS("integers");
        const float *outputs = result + (size_t)pair * I4_HIDDEN_SIZE, *hidden_values = i4f.hidden.values;
        double error[I4_STAGES] = {0.0, 0.0}, square[I4_STAGES] = {0.0, 0.0};
        uint64_t contract_differing = 0;
        float contract = 0.0f;
        for (uint32_t column = 0; column < I4_HIDDEN_SIZE; column++) { values[column] = i4f_input_value(row, column); }
        if (checked == I4F_CHECKED_PAIRS) {      /* input_rule: after the pairs, whose vectors it overwrites */
            struct i4_vectors rule = i4f.input;
            int valid = 1;
            i4f_validate_prepare(&valid, &i4f.input, I4F_RULE_VECTOR, 1);
            PROBE_REQUIRE(valid, "the launch of input_rule was refused");
            rule.values += (size_t)I4F_RULE_VECTOR * I4_HIDDEN_SIZE;
            variant->prepare(&rule, 1, i4f.zero_point);
            PROBE_CHECK_CUDA(cudaGetLastError());
            PROBE_AFTER_LAUNCH();
            compared = 0;
            differing = i4f_check_prepared(form, &i4f.input, 0, values, flipped, &input, &compared);
            probe_check_count(case_name, variant->name, "input_rule", differing, compared, flipped);
            break;
        }
        differing += i4f_check_prepared(form, &i4f.input, row, values, flipped, &input, &compared);
        PROBE_CHECK_CUDA(cudaMemcpy(values, hidden_values + (size_t)pair * I4_INNER_SIZE, I4_INNER_SIZE * sizeof(float),
                                    cudaMemcpyDeviceToHost));
        differing += i4f_check_prepared(form, &i4f.hidden, pair, form == I4_HSUB ? NULL : values, flipped, &hidden,
                                        &compared);
        for (uint32_t inner = 0; inner < I4_INNER_SIZE; inner++) {
            const double card = form == I4_HSUB ? hidden.values[inner] : (double)values[inner];
            double gate = i4f_reference_product(matrix, inner, I4_HIDDEN_SIZE, &input, &contract);
            double up = i4f_reference_product(matrix + I4_UP_MATRIX, inner, I4_HIDDEN_SIZE, &input, &contract);
            gate = fmin(i4f_bf16(gate), (double)I4_SWIGLU_LIMIT);
            up = fmin(fmax(i4f_bf16(up), -(double)I4_SWIGLU_LIMIT), (double)I4_SWIGLU_LIMIT);
            const double reference = i4f_bf16(gate / (1.0 + exp(-gate)) * up * weight);
            error[0] += (card - reference) * (card - reference);
            square[0] += reference * reference;
        }
        hidden.integers[0] ^= changed ? 1 : 0;   /* --break integers */
        for (uint32_t output = 0; output < I4_HIDDEN_SIZE; output++) {
            const double reference = i4f_reference_product(matrix + I4_DOWN_MATRIX, output, I4_INNER_SIZE, &hidden,
                                                           &contract);
            error[1] += ((double)outputs[output] - reference) * ((double)outputs[output] - reference);
            square[1] += reference * reference;
            contract_differing += memcmp(&contract, &outputs[output], sizeof(float)) != 0 ? 1u : 0u;
        }
        for (uint32_t stage = 0; stage < I4_STAGES; stage++) {
            probe_check_error(case_name, variant->name, names[checked][stage], sqrt(error[stage] / square[stage]),
                              limits[stage], off_by_one || (probe_flag_given(PROBE_FLAG_CORRUPT) && checked == 0));
        }
        if (I4F_IS_WHOLE(form)) {
            probe_check_count(case_name, variant->name, "integer_contract", contract_differing, I4_HIDDEN_SIZE,
                              changed);
        }
        if (checked + 1u == I4F_CHECKED_PAIRS) {
            probe_check_count(case_name, variant->name, "prepared_vectors", differing, compared, flipped || off_by_one);
        }
    }
    for (size_t word = 0; word < (size_t)rows * I4_HIDDEN_SIZE; word++) {
        const float *outputs = result + word / I4_HIDDEN_SIZE * I4F_ROW_FLOATS + word % I4_HIDDEN_SIZE;
        float sum = 0.0f;                        /* by definition: the card's six outputs as bf16, in slot order */
        for (size_t slot = 0; slot < I4_EXPERTS_PER_ROW; slot++) {
            sum += (float)i4f_bf16((double)outputs[slot * I4_HIDDEN_SIZE]);
        }
        sum = negated && word == 0 ? -sum : sum;
        sums_differing += memcmp(&sum, &result[I4F_ROW_SUMS_AT + word], sizeof(sum)) != 0 ? 1u : 0u;
    }
    probe_check_count(case_name, variant->name, "row_sums", sums_differing, (uint64_t)rows * I4_HIDDEN_SIZE, negated);
}
/* --break-route: two launches, each past one bound of what a kernel indexes by, a job's expert and a column's pair:
 * the validation every real step passes must refuse each (1 of 1); none is launched. */
static void i4f_break_route(const struct probe_case *probe_case) {
    static const char *const names[] = {"refused_expert_past_the_stored", "refused_pair_past_the_pairs"};
    for (uint32_t broken = 0; broken < PROBE_COUNT_OF(names); broken++) {
        struct i4_job jobs[I4_MOST_PAIRS];
        const uint32_t rows = i4f_rows(probe_case);
        const uint32_t job_count = i4f_build_jobs(I4F_CHECKED_STEP, probe_case, 0, rows, jobs);
        jobs[0].expert += broken == I4F_BROKEN_EXPERT ? (uint32_t)I4_EXPERT_COUNT : 0u;
        jobs[0].pair[I4_MOST_ROWS - 1] += broken == I4F_BROKEN_PAIR ? rows * I4_EXPERTS_PER_ROW : 0u;
        const int refused = !i4f_step_is_valid(PROBE_BASELINE, jobs, job_count, 0, 0, rows);
        probe_check_count(probe_case->name, i4f_variants[PROBE_BASELINE].name, names[broken], refused ? 1u : 0u, 1, 1);
    }
}
static void i4f_check_case(void *, const struct probe_case *probe_case) {
    const uint32_t rows = i4f_rows(probe_case), overlap = i4f_overlap(probe_case);
    const int break_exactness = I4F_BREAKS("exactness"), break_integers = I4F_BREAKS("integers");
    const int reordered = break_exactness && overlap > 0 && rows > 1u;       /* the twin has a job of several rows */
    const size_t corrupted = (size_t)i4f_expert(I4F_CHECKED_STEP, overlap, rows, 0, I4_EXPERTS_PER_ROW - 1u)
                             * I4_MATRIX_KINDS;
    if (probe_flag_given(PROBE_FLAG_BREAK_ROUTE)) {
        i4f_break_route(probe_case);
        return;
    }
    for (size_t kind = 0; kind <= I4_DOWN_MATRIX && probe_flag_given(PROBE_FLAG_CORRUPT); kind += I4_DOWN_MATRIX) {
        PROBE_CHECK_CUDA(cudaMemset((char *)i4f.stored[I4F_K16_ORDER]
                                        + (corrupted + kind) * (I4_EXPERT_CODE_BYTES / I4_MATRIX_KINDS),
                                    I4F_CORRUPTED_BYTE, I4F_CORRUPTED_BYTES));
    }
    for (uint32_t index = 0; index < I4F_VARIANTS; index++) {
        const struct i4f_variant *variant = &i4f_variants[index];
        const struct i4f_variant *launched = break_exactness && index == PROBE_BASELINE ? &i4f_broken_order : variant;
        /* a break run: integers the whole-number forms, any other the baseline (exactness: its twin) */
        const int whole = I4F_IS_WHOLE(variant->form), concerned = break_integers ? whole : index == PROBE_BASELINE;
        uint64_t differing = 0;
        if (index == PROBE_BASELINE_AGAIN || !i4f_instances[index].available || (probe_breaking() && !concerned)) {
            continue;
        }
        i4f_run_checked(probe_case, index, launched, 0, rows, i4f_result);
        i4f_check_against_host(probe_case, index, i4f_result, I4_IS_FP16(variant->form));
        probe_digest_add(i4f_result, sizeof(i4f_result));
        for (uint32_t row = 0; row < rows && rows > 1u; row++) {
            i4f_run_checked(probe_case, index, launched, row, 1, i4f_alone_result);
            differing += i4f_words_differing(i4f_result + row * I4F_ROW_FLOATS, i4f_alone_result, I4F_ROW_FLOATS)
                         + i4f_words_differing(i4f_result + I4F_ROW_SUMS_AT + (size_t)row * I4_HIDDEN_SIZE,
                                               i4f_alone_result + I4F_ROW_SUMS_AT, I4_HIDDEN_SIZE);
        }
        if (rows > 1u) {                         /* a float form's count on a CMP 170HX is the card's answer */
            probe_check_count(probe_case->name, variant->name, "rows_exact", differing, I4F_ROW_WORDS(rows),
                              reordered || (!whole && PROBE_SM == PROBE_SM_80 && differing != 0));
            i4f_instances[index].contract_class = differing != 0 ? 'F' : i4f_instances[index].contract_class;
        }
    }
    i4f_instances[PROBE_BASELINE_AGAIN].contract_class = i4f_instances[PROBE_BASELINE].contract_class;
    probe_cuda_check_guards(probe_case->name);
}

/* The figures of ROUND_1.md, package 6, at one row (once, at overlap 0: one row is the same step at any overlap), as
 * ratios to this run's sum24 (us at a04's and b02's 188 and 92): hsub 0.80, on the PRO 0.93; planes 0.64 (sm_80). */
static void i4f_verdicts(void *, const struct probe_case *probe_case, const struct probe_result *results,
                         double noise_ratio) {
    static const double hsub_bar[I4F_ARCHITECTURES][I4F_BAR_FIGURES] = {{0.80, -38.0}, {0.93, -6.0}};
    static const double planes_bar[I4F_BAR_FIGURES] = {0.64, -68.0};
    const struct probe_result *hsub = &results[probe_instance_index("operand:hsub")];
    const struct probe_result *planes = &results[probe_instance_index("operand:planes")];
    const int on_the_pro = PROBE_SM == PROBE_SM_120A;
    const double *bar = hsub_bar[on_the_pro];
    if (i4f_rows(probe_case) != 1u || i4f_overlap(probe_case) != 0u) { return; }
    probe_verdict("hsub_at_one_row", bar[0], bar[1], hsub->ratio, noise_ratio, hsub->same_sign, hsub->rounds);
    if (!on_the_pro) {
        probe_verdict("planes_at_one_row", planes_bar[0], planes_bar[1], planes->ratio, noise_ratio, planes->same_sign,
                      planes->rounds);
    }
}

static struct probe_program i4f_program = {
    "int4_forms",     i4f_flags,       PROBE_COUNT_OF(i4f_flags), i4f_instances,     I4F_VARIANTS,
    NULL,             i4f_configure,   i4f_open_card,             i4f_describe_case, i4f_check_case,
    i4f_prepare_step, i4f_launch_step, probe_cuda_mark_start,     probe_cuda_elapsed_microseconds,
    i4f_verdicts,     i4f_close_card,  NULL};

int main(int argument_count, char **arguments) { return probe_main(argument_count, arguments, &i4f_program); }
