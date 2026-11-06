/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * block_scaled_probe: round 1, unit 3b of the speed campaign (research/speed/ROUND_1.md; ledger N5, N6, P4): the
 * checkpoint's routed experts on the RTX PRO 6000 through the instruction that is their product (kernels and forms:
 * block_scaled_kernels.cuh). A step is one layer's routed experts for the rows of a case, five launches; each step
 * reads experts a span further on in a ring of 128: nothing the L2 still holds. The rows' inputs are the same in
 * every step and hold the corners (scaled_input_value). Instances: SCALED_OTHERS, interleaved in one run by
 * probe_harness.h; sm_80 has the fp16 form alone. Cases: --rows by --overlap. A job is an expert with the rows that
 * share it: up to 8 are the columns of one instruction; more (--rows 64 --overlap 100: the 64 rows an expert has of
 * a 4,096-row chunk) are eight column groups, and a run with such a case takes those kernels for all its cases.
 * Checked per case and instance on step 0, before anything is timed, of the first and the last row against the host:
 *   quantized         every code and exponent of the input and of the six hidden vectors: the reference's quantizer
 *                     written from its definition, which both of the card's quantizers are held to
 *   host_reference_*  gate and up (each also launched alone, for its fp32 sums), down, the hidden vectors and the
 *                     row's sum against double precision from the stored words, each stage fed with the card's last
 *   MEASURE           no check. float64_differing: the sampled sums whose bf16 is not the float64 sum's (EVIDENCE.md,
 *                     "The ground truth"); baseline_differing: all its sums that are not the baseline's words. An
 *                     instance with both, the first over 0.1%, is class C, not E
 * and rows_exact (every row again alone: no word of its six outputs and of its sum differs), guard_bytes, DIGEST.
 * Every launch, checked or timed, is validated on the host first (scaled_plan_is_ready). Shown failing, on the first
 * case and with nothing timed: --corrupt (after the baseline a megabyte of every matrix the step reads: every later
 * host_reference of gate, up and down reports), --break-route (an expert past the ring: refused), --break-guard,
 * --break exactness (the slices of a job of several rows added from the second on: rows_exact reports where rows
 * share an expert), quantization (the host's floor doubled), scale_lanes (sm_120a: the lanes that scale outputs g and
 * g + 8 exchanged: the host_reference of gate, up and down reports). VERDICT: product:scaled_fresh against the
 * baseline at one row (bar 1.00: undecided by the harness's rule; the ratio and its pairs are printed) and at 64 rows
 * a job (bar 0.60). Not shown: how the instruction rounds (unit 3a's log); a form's bits against the reference's
 * captures (round 2).
 *   block_scaled_probe [--rows 1,6,8] [--overlap 0,50,100] [--break NAME] and the flags of probe_harness.h
 * Measurement tool: not part of the engine.
 */
#include "probe_harness.h"
#include "block_scaled_kernels.cuh"
enum {
    SCALED_EXPERTS = 128,                        /* the ring: 2.2 GiB, 17 times the L2 of the PRO */
    SCALED_MOST_ROWS = SCALED_MOST_COLUMNS, SCALED_MOST_PAIRS = SCALED_MOST_ROWS * SCALED_EXPERTS_PER_ROW,
    SCALED_MOST_JOBS = SCALED_EXPERTS,           /* a step's jobs are experts of the ring, each once */
    SCALED_INPUT_BLOCKS = SCALED_HIDDEN_SIZE / SCALED_VALUES_PER_BLOCK,  /* the quantizer's block: a thread each */
    SCALED_HIDDEN_BLOCKS = SCALED_INNER_SIZE / SCALED_VALUES_PER_BLOCK,
    SCALED_EXPERT_BYTES = SCALED_MATRIX_KINDS * SCALED_PACKED_WORDS_PER_MATRIX * SCALED_WORD_BYTES,
    SCALED_EXPERT_SCALE_BYTES = SCALED_MATRIX_KINDS * SCALED_SCALE_WORDS_PER_MATRIX * SCALED_WORD_BYTES,
    SCALED_STEP_LAUNCHES = 5, SCALED_FILL_BLOCKS_PER_SM = 8, SCALED_WIDTHS = 2, SCALED_NAME_BYTES = 256,
    SCALED_OVERLAP_STEP = 50, SCALED_PERCENT = 100, SCALED_ROUTE_WEIGHT_STEPS = 32, SCALED_CHECKED_STEP = 0,
    SCALED_CHECK_STRIDE = 5,                     /* every 5th output is compared: all lanes, over the tiles */
    SCALED_PER_MILLE = 1000,                     /* class E admits one sampled sum in a thousand off the measure */
    SCALED_CORRUPTED_BYTES = 1024 * 1024, SCALED_CORRUPTED_BYTE = 0x5A,
    SCALED_STALE_SHARED = 0xFF, SCALED_STALE_ALONE = 0xEE,     /* outputs before a launch: no unwritten word matches */
    /* a launch's results on the host: the pairs' down sums, the rows' sums, the pairs' gate and up sums */
    SCALED_DOWN_AT = 0, SCALED_ROWS_AT = SCALED_MOST_PAIRS * SCALED_HIDDEN_SIZE,
    SCALED_GATE_AT = SCALED_ROWS_AT + SCALED_MOST_ROWS * SCALED_HIDDEN_SIZE,
    SCALED_UP_AT = SCALED_GATE_AT + SCALED_MOST_PAIRS * SCALED_INNER_SIZE,
    SCALED_RESULT_FLOATS = SCALED_UP_AT + SCALED_MOST_PAIRS * SCALED_INNER_SIZE,
    SCALED_REFERENCE = 0, SCALED_CURRENT, SCALED_ALONE, SCALED_RESULT_SETS,
    SCALED_STAGE_HIDDEN = SCALED_MATRIX_KINDS, SCALED_STAGE_ROWS, SCALED_STAGES,     /* after the three matrices */
    SCALED_QUANTIZED_BYTES = SCALED_HIDDEN_SIZE + SCALED_INPUT_BLOCKS,   /* a vector's codes, then its exponents */
    SCALED_SLICE_ORDER = 0, SCALED_SCALE_LANES,  /* scaled_faults */
    /* synthetic inputs: multiples of 1/64 in -2 .. 2 over 2^0 .. 2^3, and the kinds of block (scaled_input_value) */
    SCALED_INPUT_SEED = 0x1234567, SCALED_ROUTE_SEED = 0xABCDEF,
    SCALED_INPUT_VALUE_STEPS = 255, SCALED_SIGN_SHIFT = 31, SCALED_CORNER_SUBNORMAL_CODES = 1, SCALED_CORNER_ZEROS,
    SCALED_CORNER_FLOOR, SCALED_CORNER_LARGEST_CODE, SCALED_CORNER_KINDS, SCALED_CORNER_STEPS = 15,
    SCALED_CORNER_TIE_BASE = 17, SCALED_CORNER_TIE_STEPS = 7, SCALED_CORNER_SHIFTS = 3,
    SCALED_E4M3_NOT_A_NUMBER = 0x7F, SCALED_E4M3_EXPONENT_MASK = 15, SCALED_E4M3_MANTISSA_MASK = 7,
    SCALED_BF16_SIGNIFICANT_BITS = 8, SCALED_NIBBLE_MASK = 15
};
#define SCALED_FLAG_ROWS "--rows"
#define SCALED_FLAG_OVERLAP "--overlap"
#define SCALED_FLAG_BREAK "--break"
#define SCALED_FRESH "product:scaled_fresh"      /* the instance the acceptance figures judge */
#define SCALED_SIXTY_FOURTHS 64.0f               /* inputs and route weights are multiples of 1/64 */
#define SCALED_INPUT_VALUE_OFFSET 128.0f
#define SCALED_CORNER_SUBNORMAL_DIVISOR 262144.0f       /* 2^18: beside a 1.0, steps of half a subnormal code */
#define SCALED_CORNER_FLOOR_DIVISOR 131072.0f    /* 2^17: magnitudes whose own scale would be under the floor's */
#define SCALED_CORNER_LARGEST 1.6875f            /* 432 * 2^-8: a tie between the two largest codes at its scale */
#define SCALED_CORNER_TIE_DIVISOR 4096.0f        /* odd multiples of 2^-12: ties between codes at that scale */
#define SCALED_E4M3_VALUE_DIVISOR 1024.0         /* 2^(bias + mantissa bits) */
#define SCALED_SUM_TOLERANCE 1.0e-6              /* CONVENTIONS 5.1: an fp32 intermediate; a bfloat16 output */
#define SCALED_BF16_TOLERANCE 2.0e-3
#define SCALED_BYTES_PER_MB 1.0e6
/* ROUND_1.md's bars, 1 row and 64 a job; -168 us: 0.4 of a 64th of the thin form's 26.8 ms a layer and chunk (b08) */
static const struct { double ratio, microseconds; } scaled_bars[] = {{1.00, 0.0}, {0.60, -168.0}};
/* A range of a card array against what the array was allocated with: a REFUSED line names the array. */
#define SCALED_FITS(valid, array, first, count, bytes) \
    probe_launch_range(valid, "step", #array, first, count, bytes, probe_cuda_array_bytes(array))
#if PROBE_SM == SCALED_SM_120A
#define SCALED_SM_120A_ONLY(code) code
#define SCALED_BREAKS "exactness,quantization,scale_lanes"
#else
#define SCALED_SM_120A_ONLY(code)
#define SCALED_BREAKS "exactness,quantization"
#endif
struct scaled_plan {                             /* one step as numbers: validated, then launched */
    uint32_t slot, first_row, row_count, job_count;
    uint32_t alone;                              /* SCALED_DOWN_MATRIX: the whole step; a check's gate or up: it */
    const struct probe_instance *instance;       /* whose block and bound the launches are held to; NULL: none's */
    struct scaled_launch gate_up, matrix;        /* of a plan that is ready; matrix: down, or the one alone */
    struct scaled_job jobs[SCALED_MOST_JOBS];
};
typedef void scaled_product_kernel(struct scaled_launch launch);
struct scaled_variant {                          /* the kernels of an instance: per width, eight rows a job and 64 */
    scaled_product_kernel *gate_up[SCALED_WIDTHS], *matrix[SCALED_WIDTHS];
    void (*quantize)(struct scaled_vectors vectors, uint32_t first_vector);
    const char *kernels[SCALED_WIDTHS];          /* as the registers file names them */
};
static struct {
    struct scaled_launch gate_up, down;          /* a step's products on the card's arrays, jobs at slot 0 */
    float *matrix_sums, *row_output;             /* [pair][inner size], of gate or up alone; [row][hidden size] */
    int wide;                                    /* a job of the run has over eight rows: eight column groups */
    struct scaled_plan plans[PROBE_STREAM_STEPS], checked;      /* of each slot's timed step; of a checked launch */
    float input[SCALED_MOST_ROWS][SCALED_HIDDEN_SIZE];          /* the rows' inputs, as uploaded */
    float results[SCALED_RESULT_SETS][SCALED_RESULT_FLOATS];
} scaled;
#define SCALED_STAGE(at, size, name) {at, size, name, "host_reference_" name}
static const struct { uint32_t at, size; const char *name, *check; } scaled_stages[SCALED_STAGES] = {
    SCALED_STAGE(SCALED_GATE_AT, SCALED_INNER_SIZE, "gate"), SCALED_STAGE(SCALED_UP_AT, SCALED_INNER_SIZE, "up"),
    SCALED_STAGE(SCALED_DOWN_AT, SCALED_HIDDEN_SIZE, "down"), SCALED_STAGE(0, 0, "hidden"),
    SCALED_STAGE(SCALED_ROWS_AT, SCALED_HIDDEN_SIZE, "rows")};
static struct probe_flag scaled_flags[] = {
    PROBE_COMMON_FLAGS, PROBE_CASE_NUMBERS(SCALED_FLAG_ROWS, 1, SCALED_MOST_ROWS, "1,6,8", "rows of a step"),
    PROBE_CASE_WORDS(SCALED_FLAG_OVERLAP, "0,50,100", "0", "percent of a row's experts the row before has too"),
    PROBE_WORD_LIST(SCALED_FLAG_BREAK, SCALED_BREAKS, "", "show that check failing")};
#define SCALED_STAMP(id, name, product, quantizer, bound, blocks, claimed)                                          \
    SCALED_PRODUCT_KERNELS(id, SCALED_PRODUCT_##product, bound, blocks, SCALED_SOUND)
#define SCALED_INSTANCE(id, name, product, quantizer, bound, blocks, claimed)                                       \
    {name, NULL, claimed, 1, SCALED_BLOCK_THREADS(SCALED_GATE_UP_MATRICES),                                         \
     (blocks) != 0 ? SCALED_BLOCK_THREADS(SCALED_GATE_UP_MATRICES) : 0, blocks, 0, 0},
#define SCALED_VARIANT(id, name, product, quantizer, bound, blocks, claimed)                                        \
    {{scaled_gate_up_##id, scaled_wide_gate_up_##id}, {scaled_matrix_##id, scaled_wide_matrix_##id},                \
     scaled_quantize_##quantizer,                                                                                   \
     {"scaled_gate_up_" #id "+scaled_matrix_" #id, "scaled_wide_gate_up_" #id "+scaled_wide_matrix_" #id}},
/* id, name, product, quantizer, launch bound and its blocks of gate with up to an SM, the class it claims. The
 * bound is the one of the pilot's two that was faster on the PRO, and one that caps no kernel's registers. */
#define SCALED_BASELINE(X, name) X(baseline, name, F16, software, OPEN, 0, 'E')
#define SCALED_OTHERS(X)                                                                                            \
    X(bound, "bounds:256x1", F16, software, STATED, 1, 'E')                                                         \
    SCALED_SM_120A_ONLY(                                                                                            \
        X(mixed, "product:mixed", MIXED, software, OPEN, 0, 'E')                                                    \
        X(fresh, SCALED_FRESH, FRESH, software, OPEN, 0, 'E')                                                       \
        X(chained, "product:scaled_chained", CHAINED, software, OPEN, 0, 'F')                                       \
        X(conversion, "quantizer:cvt", F16, conversion, OPEN, 0, 'E')                                               \
        X(fresh_bound, SCALED_FRESH ",bounds:256x1", FRESH, software, STATED, 1, 'E')                               \
        X(fresh_conversion_bound, SCALED_FRESH ",quantizer:cvt,bounds:256x1", FRESH, conversion, STATED, 1, 'E'))
SCALED_BASELINE(SCALED_STAMP, "baseline") SCALED_OTHERS(SCALED_STAMP)
/* never timed: what --break exactness and --break scale_lanes launch */
SCALED_PRODUCT_KERNELS(slice_order, SCALED_PRODUCT_F16, OPEN, 0, SCALED_FAULT_SLICE_ORDER)
SCALED_SM_120A_ONLY(SCALED_PRODUCT_KERNELS(scale_lanes, SCALED_PRODUCT_FRESH, OPEN, 0, SCALED_FAULT_SCALE_LANES))
static struct probe_instance scaled_instances[] = {
    SCALED_BASELINE(SCALED_INSTANCE, "baseline") SCALED_BASELINE(SCALED_INSTANCE, "baseline_again")
    SCALED_OTHERS(SCALED_INSTANCE)};
static const struct scaled_variant scaled_variants[] = {
    SCALED_BASELINE(SCALED_VARIANT, "baseline") SCALED_BASELINE(SCALED_VARIANT, "baseline_again")
    SCALED_OTHERS(SCALED_VARIANT)};
static const struct scaled_variant scaled_faults[] = {SCALED_VARIANT(slice_order, , , software, , , )
    SCALED_SM_120A_ONLY(SCALED_VARIANT(scale_lanes, , , software, , , ))};
#define SCALED_INSTANCE_COUNT PROBE_COUNT_OF(scaled_instances)
/* The launches of a plan, with the blocks and grids it was validated with, by the kernels of the run's width. */
static void scaled_queue_plan(const struct scaled_variant *variant, const struct scaled_plan *plan)
{
    scaled_product_kernel *const gate_up = variant->gate_up[scaled.wide];
    scaled_product_kernel *const matrix = variant->matrix[scaled.wide];
    void (*const quantize)(struct scaled_vectors, uint32_t) = variant->quantize;
    const dim3 gate_up_grid(SCALED_INNER_SIZE / SCALED_OUTPUTS_PER_TILE, plan->job_count);
    const dim3 matrix_grid(plan->matrix.output_size / SCALED_OUTPUTS_PER_TILE, plan->job_count);
    const uint32_t pairs = plan->row_count * SCALED_EXPERTS_PER_ROW;
    if (plan->alone == SCALED_DOWN_MATRIX) {
        quantize<<<plan->row_count, SCALED_INPUT_BLOCKS>>>(plan->gate_up.fed, plan->first_row);
        gate_up<<<gate_up_grid, SCALED_BLOCK_THREADS(SCALED_GATE_UP_MATRICES)>>>(plan->gate_up);
        quantize<<<pairs, SCALED_HIDDEN_BLOCKS>>>(plan->matrix.fed, plan->first_row * SCALED_EXPERTS_PER_ROW);
    }
    matrix<<<matrix_grid, SCALED_BLOCK_THREADS(SCALED_ONE_MATRIX)>>>(plan->matrix);
    if (plan->alone == SCALED_DOWN_MATRIX) {
        scaled_row_sum_kernel<<<plan->row_count, SCALED_ELEMENT_THREADS>>>(plan->matrix.output, scaled.row_output,
                                                                           plan->first_row);
    }
    PROBE_AFTER_LAUNCH();
}
/* ---- routes, inputs, plans, check 10 ---- */
static uint32_t scaled_case_rows(const struct probe_case *probe_case)
{
    return (uint32_t)probe_case_value(probe_case, SCALED_FLAG_ROWS);
}
/* The experts a further row brings (6, 3 or 0); those a step reads, consecutive in the ring; its largest job's rows. */
static uint32_t scaled_stride(uint64_t overlap_word)
{
    return SCALED_EXPERTS_PER_ROW * (SCALED_PERCENT - (uint32_t)overlap_word * SCALED_OVERLAP_STEP) / SCALED_PERCENT;
}
static uint32_t scaled_span(uint32_t rows, uint32_t stride) { return (rows - 1u) * stride + SCALED_EXPERTS_PER_ROW; }
static uint32_t scaled_job_rows(uint32_t rows, uint32_t stride)
{
    return stride == 0 || rows < SCALED_EXPERTS_PER_ROW / stride ? rows : SCALED_EXPERTS_PER_ROW / stride;
}
/* A step's experts lie a span further on in the ring each step; row r takes six from the (r * stride)-th on. */
static uint32_t scaled_route_expert(const struct probe_case *probe_case, uint64_t step, uint32_t row, uint32_t slot)
{
    const uint32_t stride = scaled_stride(probe_case_value(probe_case, SCALED_FLAG_OVERLAP));
    const uint32_t span = scaled_span(scaled_case_rows(probe_case), stride);
    return (uint32_t)(step * span % (SCALED_EXPERTS - span + 1u)) + row * stride + slot;
}
static float scaled_route_weight(uint32_t row, uint32_t slot)
{
    const uint32_t code = scaled_mix_bits((uint32_t)SCALED_ROUTE_SEED + row * SCALED_EXPERTS_PER_ROW + slot);
    return (float)(1u + code % SCALED_ROUTE_WEIGHT_STEPS) / SCALED_SIXTY_FOURTHS;
}
/* Value `column` of row `vector`'s input, exact in fp32. Block b is of kind (b + vector) % SCALED_CORNER_KINDS, so
 * that every kind lies beside every other in a vector and in the next column: plain values; subnormal codes and ties
 * beside a 1.0 (magnitudes over 2^18); zeros of both signs; under the floor of 1e-4; a tie up to the largest code. */
static float scaled_input_value(uint32_t vector, uint32_t column)
{
    const uint32_t bits = scaled_mix_bits((uint32_t)SCALED_INPUT_SEED + vector * SCALED_HIDDEN_SIZE + column);
    const uint32_t kind = (column / SCALED_VALUES_PER_BLOCK + vector) % SCALED_CORNER_KINDS;
    const uint32_t steps = (bits >> SCALED_BYTE_BITS) & SCALED_CORNER_STEPS;
    const uint32_t tie = SCALED_CORNER_TIE_BASE + ((steps & SCALED_CORNER_TIE_STEPS) << 1);         /* odd */
    const int is_first = column % SCALED_VALUES_PER_BLOCK == 0;
    const float value = ((float)(bits & SCALED_INPUT_VALUE_STEPS) - SCALED_INPUT_VALUE_OFFSET) / SCALED_SIXTY_FOURTHS;
    const float sign = (bits >> SCALED_SIGN_SHIFT) != 0u ? -1.0f : 1.0f;
    if (kind == SCALED_CORNER_SUBNORMAL_CODES || kind == SCALED_CORNER_LARGEST_CODE) {
        const float beside = kind == SCALED_CORNER_SUBNORMAL_CODES ? (float)steps / SCALED_CORNER_SUBNORMAL_DIVISOR
                                                                    : (float)tie / SCALED_CORNER_TIE_DIVISOR;
        return !is_first ? sign * beside : kind == SCALED_CORNER_SUBNORMAL_CODES ? 1.0f : SCALED_CORNER_LARGEST;
    }
    if (kind == SCALED_CORNER_ZEROS) { return (steps & 1u) != 0u ? value : sign * 0.0f; }
    if (kind == SCALED_CORNER_FLOOR) { return value / SCALED_CORNER_FLOOR_DIVISOR; }
    return value / (float)(1u << (steps & SCALED_CORNER_SHIFTS));
}
/* Step `step` of a case for rows first_row .. first_row + row_count - 1 (a row keeps its pairs, vectors and output
 * places whatever the launch): one job per expert, its columns in the order of the rows. */
static void scaled_make_plan(struct scaled_plan *plan, const struct probe_case *probe_case, uint32_t instance,
                             uint64_t step, uint32_t slot, uint32_t first_row, uint32_t row_count)
{
    plan->slot = slot;
    plan->first_row = first_row;
    plan->row_count = row_count;
    plan->job_count = 0;
    plan->alone = SCALED_DOWN_MATRIX;
    plan->instance = instance < SCALED_INSTANCE_COUNT ? &scaled_instances[instance] : NULL;
    for (uint32_t pair = first_row * SCALED_EXPERTS_PER_ROW; pair < (first_row + row_count) * SCALED_EXPERTS_PER_ROW;
         pair++) {
        const uint32_t row = pair / SCALED_EXPERTS_PER_ROW, row_slot = pair % SCALED_EXPERTS_PER_ROW;
        const uint32_t expert = scaled_route_expert(probe_case, step, row, row_slot);
        uint32_t job_index = 0;
        while (job_index < plan->job_count && plan->jobs[job_index].expert_index != expert) { job_index++; }
        PROBE_REQUIRE(job_index < SCALED_MOST_JOBS, "a step of more jobs than the ring has experts");
        struct scaled_job *job = &plan->jobs[job_index];
        if (job_index == plan->job_count) {
            plan->job_count++;
            job->expert_index = expert;
            job->row_count = 0;
            for (uint32_t column = 0; column < SCALED_MOST_COLUMNS; column++) {
                job->pair_index[column] = pair;  /* a column not in use reads what column 0 reads */
                job->route_weight[column] = 0.0f;
            }
        }
        PROBE_REQUIRE(job->row_count < SCALED_MOST_COLUMNS, "a job of more rows than a block has columns");
        job->pair_index[job->row_count] = pair;
        job->route_weight[job->row_count++] = scaled_route_weight(row, row_slot);
    }
}
/* Check 10, then the plan's records and its jobs at its slot on the card. 1: every launch of the plan stays inside
 * the card's arrays as allocated (a launch's sizes are those they were allocated by), the card's limits and the
 * instance's block and bound; what a kernel indexes by is held to its range first. 0: it printed why not. */
static int scaled_plan_is_ready(struct scaled_plan *plan)
{
    const uint64_t rows_end = (uint64_t)plan->first_row + plan->row_count, pairs = rows_end * SCALED_EXPERTS_PER_ROW;
    const uint64_t first_pair = (uint64_t)plan->first_row * SCALED_EXPERTS_PER_ROW;
    const uint64_t job_rows = scaled.wide ? SCALED_MOST_COLUMNS : SCALED_COLUMNS_PER_GROUP;
    const uint64_t lane_bytes = (scaled.wide ? SCALED_WIDE : SCALED_NARROW) * SCALED_ELEMENTS_PER_LANE * sizeof(float);
    const struct scaled_vectors *rows = &scaled.gate_up.fed, *hidden = &scaled.down.fed;
    int valid = 1;
    probe_launch_value(&valid, "step", "slot", plan->slot, 0, PROBE_STREAM_STEPS - 1u);
    probe_launch_value(&valid, "step", "rows", plan->row_count, 1, SCALED_MOST_ROWS);
    probe_launch_value(&valid, "step", "row after the last", rows_end, 1, SCALED_MOST_ROWS);
    probe_launch_value(&valid, "step", "jobs", plan->job_count, 1, SCALED_MOST_JOBS);
    probe_launch_value(&valid, "step", "matrix alone", plan->alone, SCALED_GATE_MATRIX, SCALED_DOWN_MATRIX);
    for (uint32_t job = 0; valid && job < plan->job_count; job++) {
        probe_launch_value(&valid, "step", "expert of a job", plan->jobs[job].expert_index, 0, SCALED_EXPERTS - 1u);
        probe_launch_value(&valid, "step", "rows of a job", plan->jobs[job].row_count, 1, job_rows);
        for (uint32_t column = 0; column < SCALED_MOST_COLUMNS; column++) {
            probe_launch_value(&valid, "step", "pair of a column", plan->jobs[job].pair_index[column], first_pair,
                               pairs - 1u);
        }
    }
    if (!valid) { return 0; }
    probe_launch_geometry(&valid, "scaled_gate_up", plan->instance, SCALED_INNER_SIZE / SCALED_OUTPUTS_PER_TILE,
                          plan->job_count, 1, SCALED_BLOCK_THREADS(SCALED_GATE_UP_MATRICES),
                          SCALED_BLOCK_THREADS(SCALED_GATE_UP_MATRICES) * lane_bytes);
    probe_launch_geometry(&valid, "scaled_matrix", NULL, SCALED_HIDDEN_SIZE / SCALED_OUTPUTS_PER_TILE, plan->job_count,
                          1, SCALED_BLOCK_THREADS(SCALED_ONE_MATRIX),
                          SCALED_BLOCK_THREADS(SCALED_ONE_MATRIX) * lane_bytes);
    probe_launch_geometry(&valid, "scaled_quantize", NULL, pairs, 1, 1, SCALED_INPUT_BLOCKS, 0);
    probe_launch_geometry(&valid, "scaled_row_sum_kernel", NULL, rows_end, 1, 1, SCALED_ELEMENT_THREADS, 0);
    SCALED_FITS(&valid, scaled.down.packed, 0, SCALED_EXPERTS, SCALED_EXPERT_BYTES);
    SCALED_FITS(&valid, scaled.down.scales, 0, SCALED_EXPERTS, SCALED_EXPERT_SCALE_BYTES);
    SCALED_FITS(&valid, scaled.down.jobs, (uint64_t)plan->slot * SCALED_MOST_JOBS, plan->job_count,
                sizeof(struct scaled_job));
    SCALED_FITS(&valid, rows->values, 0, rows_end, SCALED_HIDDEN_SIZE * sizeof(float));
    SCALED_FITS(&valid, rows->codes, 0, rows_end, SCALED_HIDDEN_SIZE);
    SCALED_FITS(&valid, rows->exponents, 0, rows_end, SCALED_INPUT_BLOCKS);
    SCALED_FITS(&valid, hidden->values, 0, pairs, SCALED_INNER_SIZE * sizeof(float));
    SCALED_FITS(&valid, hidden->codes, 0, pairs, SCALED_INNER_SIZE);
    SCALED_FITS(&valid, hidden->exponents, 0, pairs, SCALED_HIDDEN_BLOCKS);
    SCALED_FITS(&valid, scaled.down.output, 0, pairs, SCALED_HIDDEN_SIZE * sizeof(float));
    SCALED_FITS(&valid, scaled.matrix_sums, 0, pairs, SCALED_INNER_SIZE * sizeof(float));
    SCALED_FITS(&valid, scaled.row_output, 0, rows_end, SCALED_HIDDEN_SIZE * sizeof(float));
    if (!valid) { return 0; }
    plan->gate_up = scaled.gate_up;
    plan->matrix = plan->alone == SCALED_DOWN_MATRIX ? scaled.down : scaled.gate_up;
    plan->gate_up.jobs = plan->matrix.jobs = scaled.down.jobs + (size_t)plan->slot * SCALED_MOST_JOBS;
    if (plan->alone != SCALED_DOWN_MATRIX) {     /* gate or up by the kernel of one matrix: its fp32 sums */
        plan->matrix.matrix_kind = plan->alone;
        plan->matrix.output = scaled.matrix_sums;
    }
    PROBE_CHECK_CUDA(cudaMemcpy((struct scaled_job *)plan->matrix.jobs, plan->jobs,
                                plan->job_count * sizeof(struct scaled_job), cudaMemcpyHostToDevice));
    return 1;
}
/* ---- the card ---- */
static struct scaled_vectors scaled_new_vectors(uint64_t count, uint32_t size, uint32_t pairs_per_vector)
{
    return {(float *)probe_cuda_allocate(count * size * sizeof(float)), (uint32_t *)probe_cuda_allocate(count * size),
            (uint8_t *)probe_cuda_allocate(count * (size / SCALED_VALUES_PER_BLOCK)), size, pairs_per_vector};
}
static int scaled_open_card(void *, struct probe_card_facts *facts)
{
    if (!probe_cuda_open(facts)) { return 0; }
    uint32_t *const stored[] = {(uint32_t *)probe_cuda_allocate((uint64_t)SCALED_EXPERTS * SCALED_EXPERT_BYTES),
                                (uint32_t *)probe_cuda_allocate((uint64_t)SCALED_EXPERTS * SCALED_EXPERT_SCALE_BYTES)};
    const struct scaled_job *jobs = (const struct scaled_job *)probe_cuda_allocate(
        PROBE_STREAM_STEPS * SCALED_MOST_JOBS * sizeof(struct scaled_job));
    const struct scaled_vectors rows = scaled_new_vectors(SCALED_MOST_ROWS, SCALED_HIDDEN_SIZE, SCALED_EXPERTS_PER_ROW);
    const struct scaled_vectors pairs = scaled_new_vectors(SCALED_MOST_PAIRS, SCALED_INNER_SIZE, 1);
    float *pair_output = (float *)probe_cuda_allocate((uint64_t)SCALED_ROWS_AT * sizeof(float));
    const uint4 *packed = (const uint4 *)stored[0];
    scaled.gate_up = {packed, stored[1], jobs, rows, pairs.values, SCALED_INNER_SIZE, SCALED_GATE_MATRIX};
    scaled.down = {packed, stored[1], jobs, pairs, pair_output, SCALED_HIDDEN_SIZE, SCALED_DOWN_MATRIX};
    scaled.matrix_sums = (float *)probe_cuda_allocate((uint64_t)SCALED_MOST_PAIRS * SCALED_INNER_SIZE * sizeof(float));
    scaled.row_output = (float *)probe_cuda_allocate((uint64_t)SCALED_MOST_ROWS * SCALED_HIDDEN_SIZE * sizeof(float));
    for (uint32_t value = 0; value < SCALED_MOST_ROWS * SCALED_HIDDEN_SIZE; value++) {
        scaled.input[value / SCALED_HIDDEN_SIZE][value % SCALED_HIDDEN_SIZE] =
            scaled_input_value(value / SCALED_HIDDEN_SIZE, value % SCALED_HIDDEN_SIZE);
    }
    PROBE_CHECK_CUDA(cudaMemcpy(rows.values, scaled.input, sizeof(scaled.input), cudaMemcpyHostToDevice));
    for (uint32_t is_scale = 0; is_scale <= 1u; is_scale++) {    /* by a kernel: 2.4 GB are seconds over the link */
        const uint32_t blocks = (uint32_t)facts->sm_count * SCALED_FILL_BLOCKS_PER_SM;
        int valid = 1;
        probe_launch_geometry(&valid, "scaled_fill_stored_kernel", NULL, blocks, 1, 1, SCALED_ELEMENT_THREADS, 0);
        PROBE_REQUIRE(valid, "the launch that fills the stored experts was refused");
        scaled_fill_stored_kernel<<<blocks, SCALED_ELEMENT_THREADS>>>(
            stored[is_scale], is_scale, probe_cuda_array_bytes(stored[is_scale]) / sizeof(uint32_t));
    }
    PROBE_CHECK_CUDA(cudaGetLastError());
    PROBE_CHECK_CUDA(cudaDeviceSynchronize());
    facts->ring_bytes = (uint64_t)SCALED_EXPERTS * (SCALED_EXPERT_BYTES + SCALED_EXPERT_SCALE_BYTES);
    return 1;
}
static void scaled_close_card(void *) { probe_cuda_close(); }
static void scaled_describe_case(void *, const struct probe_case *probe_case, struct probe_case_facts *facts)
{
    const uint64_t overlap_word = probe_case_value(probe_case, SCALED_FLAG_OVERLAP);
    const uint32_t rows = scaled_case_rows(probe_case), experts = scaled_span(rows, scaled_stride(overlap_word));
    *facts = {"native", "experts", rows, experts, (int64_t)overlap_word * SCALED_OVERLAP_STEP,
              (double)experts * (SCALED_EXPERT_BYTES + SCALED_EXPERT_SCALE_BYTES) / SCALED_BYTES_PER_MB};
}
/* The flags as a whole: every case's experts inside the ring; the width of the run's kernels, whose names the
 * instances then report; and --break counts as a break flag of the harness. 0: refused, it said why. */
static int scaled_configure(void *)
{
    const struct probe_flag *rows = probe_flag(SCALED_FLAG_ROWS), *overlaps = probe_flag(SCALED_FLAG_OVERLAP);
    for (uint32_t index = 0; index < rows->value_count * overlaps->value_count; index++) {
        const uint32_t row_count = (uint32_t)rows->values[index / overlaps->value_count];
        const uint32_t stride = scaled_stride(overlaps->values[index % overlaps->value_count]);
        if (scaled_span(row_count, stride) > SCALED_EXPERTS) {
            fprintf(stderr, "%u rows that each bring %u experts read more than the ring's %d: fewer rows, or more "
                            "--overlap\n", row_count, stride, (int)SCALED_EXPERTS);
            return 0;
        }
        scaled.wide |= scaled_job_rows(row_count, stride) > SCALED_COLUMNS_PER_GROUP;
    }
    for (uint32_t instance = 0; instance < SCALED_INSTANCE_COUNT; instance++) {
        scaled_instances[instance].kernel = scaled_variants[instance].kernels[scaled.wide];
    }
    probe_run.breaking |= (uint32_t)probe_flag_given(SCALED_FLAG_BREAK);     /* as a --break-* switch of the harness */
    return 1;
}
/* ---- host side: the formats decoded from their fields, the references in double precision; they share with the
 * kernels the stored words (scaled_stored_word), the word of a vector's value and the constants of the formats ---- */
static const double scaled_e2m1_values[] = {0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
                                            -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0};
/* E4M3 from its fields: (8 + mantissa) * 2^(exponent - 10), or mantissa * 2^-9 where the exponent is 0. */
static double scaled_host_e4m3(uint32_t code)
{
    const uint32_t exponent = (code >> SCALED_E4M3_EXPONENT_SHIFT) & SCALED_E4M3_EXPONENT_MASK;
    const uint32_t mantissa = code & SCALED_E4M3_MANTISSA_MASK;
    const uint32_t scaled_up = exponent == 0 ? mantissa << 1 : (SCALED_E4M3_IMPLICIT_ONE + mantissa) << exponent;
    return ((code & SCALED_E4M3_SIGN_BIT) != 0u ? -1.0 : 1.0) * scaled_up / SCALED_E4M3_VALUE_DIVISOR;
}
/* A double rounded to bf16's eight significant bits, to nearest with ties to even (rint in the default mode). */
static double scaled_host_bf16(double value)
{
    int binary_exponent = 0;
    const double mantissa = frexp(value, &binary_exponent);
    return ldexp(rint(ldexp(mantissa, SCALED_BF16_SIGNIFICANT_BITS)), binary_exponent - SCALED_BF16_SIGNIFICANT_BITS);
}
/* One output of one matrix of one expert in double, from the stored words: per block of 32 inputs the sum of weight
 * times activation, times both scales. Where the kernels find them: packed words are [expert][matrix][tile of 16
 * outputs][load of 64 inputs][lane][step of the load], eight nibbles each, that of register r and half h of operand
 * A of an m16n8k16 at bit 4 r + 16 h; scale words are [expert][matrix][tile][load][lane group g], byte 2 q + r the
 * E8M0 code of block q of the load for output g + 8 r. quantized: a vector's codes by value, then its exponents. */
static double scaled_host_product(uint32_t matrix_kind, uint32_t expert, uint32_t output_row, const uint8_t *quantized)
{
    const uint32_t input_size = matrix_kind == SCALED_DOWN_MATRIX ? SCALED_INNER_SIZE : SCALED_HIDDEN_SIZE;
    const uint32_t lane_group = output_row % SCALED_HALF_TILE, tile = output_row / SCALED_OUTPUTS_PER_TILE;
    const uint32_t output_half = output_row % SCALED_OUTPUTS_PER_TILE / SCALED_HALF_TILE;
    const uint64_t matrix = (uint64_t)expert * SCALED_MATRIX_KINDS + matrix_kind;
    const uint64_t first_tile_load =
        matrix * SCALED_TILE_LOADS_PER_MATRIX + (uint64_t)tile * (input_size / SCALED_INPUTS_PER_LOAD);
    double sum = 0.0;
    for (uint32_t block = 0; block < input_size / SCALED_VALUES_PER_BLOCK; block++) {
        const uint64_t tile_load = first_tile_load + block / SCALED_BLOCKS_PER_LOAD;
        const uint32_t first_step = block % SCALED_BLOCKS_PER_LOAD * SCALED_STEPS_PER_BLOCK;
        const uint32_t scale_byte = block % SCALED_BLOCKS_PER_LOAD * SCALED_BLOCKS_PER_LOAD + output_half;
        const uint32_t scale_word = scaled_stored_word(tile_load * SCALED_LANE_GROUPS + lane_group, 1);
        double block_sum = 0.0;
        for (uint32_t offset = 0; offset < SCALED_VALUES_PER_BLOCK; offset++) {
            const uint32_t input = offset % SCALED_INPUTS_PER_STEP, upper_inputs = input / SCALED_HALF_INPUTS;
            const uint64_t lane = tile_load * SCALED_LANES_PER_WARP + lane_group * SCALED_LANES_PER_LANE_GROUP
                                  + input % SCALED_HALF_INPUTS / SCALED_COLUMNS_PER_LANE;
            const uint32_t nibble = output_half + upper_inputs * SCALED_COLUMNS_PER_LANE
                                    + input % SCALED_COLUMNS_PER_LANE * SCALED_REGISTERS_PER_OPERAND_A;
            const uint32_t word =
                scaled_stored_word(lane * SCALED_STEPS_PER_LOAD + first_step + offset / SCALED_INPUTS_PER_STEP, 0);
            block_sum += scaled_e2m1_values[(word >> (nibble * SCALED_NIBBLE_BITS)) & SCALED_NIBBLE_MASK]
                         * scaled_host_e4m3(quantized[block * SCALED_VALUES_PER_BLOCK + offset]);
        }
        sum += ldexp(block_sum, (int)((scale_word >> (scale_byte * SCALED_BYTE_BITS)) & SCALED_BYTE_MASK)
                                    + (int)quantized[input_size + block] - SCALED_E8M0_BIAS - SCALED_E8M0_BIAS);
    }
    return sum;
}
/* ---- checks ---- */
struct scaled_tally {                            /* what the checked rows of one launch add up to */
    double error[SCALED_STAGES], square[SCALED_STAGES];          /* against double precision, per stage */
    uint64_t off_float64[SCALED_MATRIX_KINDS], sampled[SCALED_MATRIX_KINDS], quantized_differing, quantized_bytes;
};
static void scaled_tally_add(struct scaled_tally *tally, uint32_t stage, double card_value, double reference)
{
    tally->error[stage] += (card_value - reference) * (card_value - reference);
    tally->square[stage] += reference * reference;
    if (stage < SCALED_MATRIX_KINDS) {           /* the float64 sum rounded once, against the card's sum rounded */
        tally->off_float64[stage] += scaled_host_bf16(card_value) != scaled_host_bf16(reference) ? 1u : 0u;
        tally->sampled[stage]++;
    }
}
/* Vector `vector` of a set as the card quantized it (card_quantized: its codes by value, then its exponents) against
 * the reference's act_quant from its definition: per block of 32 the scale is 2^ceil(log2(ratio)), ratio being the
 * largest magnitude (at least 1e-4; --break quantization: twice that) times 1 / 448 in fp32; each value over the
 * scale, clamped to +-448, becomes the nearest E4M3 value, ties to the even code, the sign kept. Value 16 s + i of a
 * block is byte (i % 2) 2 + i / 8 of the word of lane place i % 8 / 2 at step s. Counted: the bytes that differ. */
static void scaled_compare_quantized(const struct scaled_vectors *vectors, size_t vector, const float *values,
                                     uint8_t *card_quantized, struct scaled_tally *tally)
{
    static uint32_t codes[SCALED_HIDDEN_SIZE / SCALED_WORD_BYTES];
    const uint32_t size = vectors->size, block_count = size / SCALED_VALUES_PER_BLOCK;
    const float broken_floor = probe_flag_has_word(SCALED_FLAG_BREAK, "quantization") ? SCALED_AMAX_FLOOR : 0.0f;
    PROBE_INDEX_CHECK(vector, vectors->pairs_per_vector == 1 ? SCALED_MOST_PAIRS : SCALED_MOST_ROWS);
    PROBE_CHECK_CUDA(cudaMemcpy(codes, vectors->codes + vector * (size / SCALED_WORD_BYTES), size,
                                cudaMemcpyDeviceToHost));
    PROBE_CHECK_CUDA(cudaMemcpy(card_quantized + size, vectors->exponents + vector * block_count, block_count,
                                cudaMemcpyDeviceToHost));
    for (uint32_t block = 0; block < block_count; block++) {
        const float *block_values = values + block * SCALED_VALUES_PER_BLOCK;
        uint8_t *block_codes = card_quantized + block * SCALED_VALUES_PER_BLOCK;
        float largest = SCALED_AMAX_FLOOR + broken_floor;
        for (uint32_t offset = 0; offset < SCALED_VALUES_PER_BLOCK; offset++) {
            largest = fmaxf(largest, fabsf(block_values[offset]));
        }
        const int log2_ceiling = (int)ceil(log2((double)(largest * SCALED_E4M3_MAXIMUM_INVERSE)));
        tally->quantized_differing += card_quantized[size + block] != (uint8_t)(log2_ceiling + SCALED_E8M0_BIAS);
        for (uint32_t offset = 0; offset < SCALED_VALUES_PER_BLOCK; offset++) {
            const uint32_t input = offset % SCALED_INPUTS_PER_STEP, pair = SCALED_REGISTERS_PER_OPERAND_B;
            const uint32_t lane_place = input % SCALED_HALF_INPUTS / pair;
            const uint32_t byte = input % pair * pair + input / SCALED_HALF_INPUTS;
            const uint32_t word = codes[scaled_activation_word(block, lane_place, offset / SCALED_INPUTS_PER_STEP)];
            const double scaled_value = ldexp((double)block_values[offset], -log2_ceiling);
            const double magnitude = fmin(fabs(scaled_value), (double)SCALED_E4M3_MAXIMUM);
            uint32_t nearest = 0;
            for (uint32_t code = 1; code < SCALED_E4M3_NOT_A_NUMBER; code++) {
                const double distance = fabs(scaled_host_e4m3(code) - magnitude);
                const double least = fabs(scaled_host_e4m3(nearest) - magnitude);
                nearest = distance < least || (distance == least && (code & 1u) == 0u) ? code : nearest;
            }
            nearest |= signbit(scaled_value) ? (uint32_t)SCALED_E4M3_SIGN_BIT : 0u;
            block_codes[offset] = (uint8_t)(word >> (byte * SCALED_BYTE_BITS));
            tally->quantized_differing += block_codes[offset] != nearest ? 1u : 0u;
        }
    }
    tally->quantized_bytes += size + block_count;
}
/* One row of the checked launch that just ran, against the host, every SCALED_CHECK_STRIDE-th output of its six
 * pairs: each stage from the card's own previous stage. A not-a-number of the card is an error no limit admits. */
static void scaled_check_row(const struct probe_case *probe_case, uint32_t row, struct scaled_tally *tally)
{
    static uint8_t input_quantized[SCALED_QUANTIZED_BYTES], hidden_quantized[SCALED_QUANTIZED_BYTES];
    static float hidden[SCALED_INNER_SIZE];
    static double row_sum[SCALED_HIDDEN_SIZE];
    const float *results = scaled.results[SCALED_CURRENT];
    const double limit = SCALED_SWIGLU_LIMIT;
    memset(row_sum, 0, sizeof(row_sum));
    scaled_compare_quantized(&scaled.gate_up.fed, row, scaled.input[row], input_quantized, tally);
    for (uint32_t slot = 0; slot < SCALED_EXPERTS_PER_ROW; slot++) {
        const size_t pair = (size_t)row * SCALED_EXPERTS_PER_ROW + slot;
        const uint32_t expert = scaled_route_expert(probe_case, SCALED_CHECKED_STEP, row, slot);
        const float *inner_sums = results + pair * SCALED_INNER_SIZE;
        const float *down_sums = results + SCALED_DOWN_AT + pair * SCALED_HIDDEN_SIZE;
        PROBE_CHECK_CUDA(cudaMemcpy(hidden, scaled.down.fed.values + pair * SCALED_INNER_SIZE, sizeof(hidden),
                                    cudaMemcpyDeviceToHost));
        for (uint32_t output = 0; output < SCALED_INNER_SIZE; output += SCALED_CHECK_STRIDE) {
            const double gate = scaled_host_product(SCALED_GATE_MATRIX, expert, output, input_quantized);
            const double up = scaled_host_product(SCALED_UP_MATRIX, expert, output, input_quantized);
            const double bounded_gate = fmin(scaled_host_bf16(inner_sums[SCALED_GATE_AT + output]), limit);
            const double bounded_up = fmin(fmax(scaled_host_bf16(inner_sums[SCALED_UP_AT + output]), -limit), limit);
            scaled_tally_add(tally, SCALED_GATE_MATRIX, inner_sums[SCALED_GATE_AT + output], gate);
            scaled_tally_add(tally, SCALED_UP_MATRIX, inner_sums[SCALED_UP_AT + output], up);
            scaled_tally_add(tally, SCALED_STAGE_HIDDEN, hidden[output],
                             scaled_host_bf16(bounded_gate / (1.0 + exp(-bounded_gate)) * bounded_up
                                              * (double)scaled_route_weight(row, slot)));
        }
        scaled_compare_quantized(&scaled.down.fed, pair, hidden, hidden_quantized, tally);
        for (uint32_t output = 0; output < SCALED_HIDDEN_SIZE; output += SCALED_CHECK_STRIDE) {
            const double down = scaled_host_product(SCALED_DOWN_MATRIX, expert, output, hidden_quantized);
            scaled_tally_add(tally, SCALED_DOWN_MATRIX, down_sums[output], down);
            row_sum[output] += scaled_host_bf16(down_sums[output]);
        }
    }
    for (uint32_t output = 0; output < SCALED_HIDDEN_SIZE; output += SCALED_CHECK_STRIDE) {
        scaled_tally_add(tally, SCALED_STAGE_ROWS, results[SCALED_ROWS_AT + (size_t)row * SCALED_HIDDEN_SIZE + output],
                         row_sum[output]);
    }
}
/* A plan outside the timing: validated, its outputs made stale, launched, then read whole into a set of results. */
static void scaled_run(const struct scaled_variant *variant, struct scaled_plan *plan, uint32_t set, int stale_byte)
{
    const size_t row_bytes = probe_cuda_array_bytes(scaled.row_output);
    PROBE_REQUIRE(scaled_plan_is_ready(plan), "a checked launch was refused");
    PROBE_CHECK_CUDA(cudaMemset(plan->matrix.output, stale_byte, probe_cuda_array_bytes(plan->matrix.output)));
    PROBE_CHECK_CUDA(cudaMemset(scaled.row_output, stale_byte, row_bytes));
    scaled_queue_plan(variant, plan);
    PROBE_CHECK_CUDA(cudaGetLastError());
    PROBE_CHECK_CUDA(cudaDeviceSynchronize());
    PROBE_CHECK_CUDA(cudaMemcpy(scaled.results[set] + scaled_stages[plan->alone].at, plan->matrix.output,
                                probe_cuda_array_bytes(plan->matrix.output), cudaMemcpyDeviceToHost));
    if (plan->alone == SCALED_DOWN_MATRIX) {
        PROBE_CHECK_CUDA(cudaMemcpy(scaled.results[set] + SCALED_ROWS_AT, scaled.row_output, row_bytes,
                                    cudaMemcpyDeviceToHost));
    }
}
static uint64_t scaled_differing(const float *first, const float *second, size_t float_count)
{
    uint64_t differing = 0;
    for (size_t index = 0; index < float_count; index++) {
        differing += memcmp(first + index, second + index, sizeof(float)) != 0 ? 1u : 0u;
    }
    return differing;
}
/* Every check of one variant on the case: `instance` is whose it is, or SCALED_INSTANCE_COUNT for kernels that are
 * no instance's. products_broken: the host references of its three products must report (the stages after them are
 * fed with the card's own); rows_differ: its rows must not be exact. */
static void scaled_check_variant(const struct probe_case *probe_case, const struct scaled_variant *variant,
                                 const char *name, uint32_t instance, int products_broken, int rows_differ)
{
    const uint32_t rows = scaled_case_rows(probe_case), pairs = rows * SCALED_EXPERTS_PER_ROW;
    const char claimed = instance < SCALED_INSTANCE_COUNT ? scaled_instances[instance].contract_class : 'E';
    struct scaled_plan *plan = &scaled.checked;
    float *current = scaled.results[SCALED_CURRENT], *alone = scaled.results[SCALED_ALONE];
    struct scaled_tally tally;
    uint64_t rows_differing = 0;
    memset(&tally, 0, sizeof(tally));
    scaled_make_plan(plan, probe_case, instance, SCALED_CHECKED_STEP, 0, 0, rows);
    scaled_run(variant, plan, SCALED_CURRENT, SCALED_STALE_SHARED);
    for (plan->alone = SCALED_GATE_MATRIX; plan->alone <= SCALED_UP_MATRIX; plan->alone++) {     /* gate, then up */
        scaled_run(variant, plan, SCALED_CURRENT, SCALED_STALE_SHARED);
    }
    if (instance == PROBE_BASELINE) { memcpy(scaled.results[SCALED_REFERENCE], current, sizeof(scaled.results[0])); }
    scaled_check_row(probe_case, 0, &tally);
    if (rows > 1u) { scaled_check_row(probe_case, rows - 1u, &tally); }
    probe_check_count(probe_case->name, name, "quantized", tally.quantized_differing, tally.quantized_bytes,
                      probe_flag_has_word(SCALED_FLAG_BREAK, "quantization"));
    for (uint32_t stage = 0; stage < SCALED_STAGES; stage++) {
        const int is_sum = stage < SCALED_MATRIX_KINDS && claimed != 'F';
        probe_check_error(probe_case->name, name, scaled_stages[stage].check,
                          sqrt(tally.error[stage] / tally.square[stage]),
                          is_sum ? SCALED_SUM_TOLERANCE : SCALED_BF16_TOLERANCE,
                          products_broken && stage < SCALED_MATRIX_KINDS);
    }
    for (uint32_t kind = 0; kind < SCALED_MATRIX_KINDS; kind++) {
        const size_t at = scaled_stages[kind].at, floats = (size_t)pairs * scaled_stages[kind].size;
        const uint64_t off_baseline = scaled_differing(current + at, scaled.results[SCALED_REFERENCE] + at, floats);
        printf("MEASURE card=%s case=%s variant=%s matrix=%s float64_differing=%" PRIu64 "/%" PRIu64
               " baseline_differing=%" PRIu64 "/%zu\n", probe_run.card_label, probe_case->name, name,
               scaled_stages[kind].name, tally.off_float64[kind], tally.sampled[kind], off_baseline, floats);
        if (off_baseline != 0 && instance < SCALED_INSTANCE_COUNT && claimed == 'E'
            && tally.off_float64[kind] * SCALED_PER_MILLE > tally.sampled[kind]) {
            scaled_instances[instance].contract_class = 'C';     /* not the baseline's sums, and under 99.9% */
        }
    }
    probe_digest_add(current + SCALED_DOWN_AT, (uint64_t)pairs * SCALED_HIDDEN_SIZE * sizeof(float));
    probe_digest_add(current + SCALED_ROWS_AT, (uint64_t)rows * SCALED_HIDDEN_SIZE * sizeof(float));
    for (uint32_t row = 0; row < rows; row++) {  /* each row again alone: its six outputs and its sum */
        const size_t pair_at = (size_t)row * SCALED_EXPERTS_PER_ROW * SCALED_HIDDEN_SIZE;
        const size_t row_at = SCALED_ROWS_AT + (size_t)row * SCALED_HIDDEN_SIZE;
        scaled_make_plan(plan, probe_case, instance, SCALED_CHECKED_STEP, 0, row, 1);
        scaled_run(variant, plan, SCALED_ALONE, SCALED_STALE_ALONE);
        rows_differing +=
            scaled_differing(current + pair_at, alone + pair_at, SCALED_EXPERTS_PER_ROW * SCALED_HIDDEN_SIZE)
            + scaled_differing(current + row_at, alone + row_at, SCALED_HIDDEN_SIZE);
    }
    probe_check_count(probe_case->name, name, "rows_exact", rows_differing,
                      (uint64_t)(pairs + rows) * SCALED_HIDDEN_SIZE, rows_differ);
}
static void scaled_check_case(void *, const struct probe_case *probe_case)
{
    const uint32_t rows = scaled_case_rows(probe_case);
    const uint32_t stride = scaled_stride(probe_case_value(probe_case, SCALED_FLAG_OVERLAP));
    const size_t matrix_bytes = SCALED_EXPERT_BYTES / SCALED_MATRIX_KINDS;
    const size_t first_byte = (size_t)scaled_route_expert(probe_case, SCALED_CHECKED_STEP, 0, 0) * SCALED_EXPERT_BYTES;
    const int corrupt = probe_flag_given(PROBE_FLAG_CORRUPT);
    if (probe_flag_given(PROBE_FLAG_BREAK_ROUTE)) {      /* an expert past the ring: check 10 must refuse the plan */
        scaled_make_plan(&scaled.checked, probe_case, PROBE_BASELINE, SCALED_CHECKED_STEP, 0, 0, rows);
        scaled.checked.jobs[0].expert_index += SCALED_EXPERTS;
        probe_check_count(probe_case->name, "baseline", "refused_expert_past_the_ring",
                          scaled_plan_is_ready(&scaled.checked) ? 0u : 1u, 1, 1);
        return;
    }
    for (uint32_t instance = 0; instance < SCALED_INSTANCE_COUNT; instance++) {
        if (instance == PROBE_BASELINE_AGAIN) { continue; }       /* the baseline's kernels a second time */
        scaled_check_variant(probe_case, &scaled_variants[instance], scaled_instances[instance].name, instance,
                             corrupt && instance != PROBE_BASELINE, 0);
        for (uint32_t matrix = 0; corrupt && instance == PROBE_BASELINE
                                  && matrix < scaled_span(rows, stride) * SCALED_MATRIX_KINDS; matrix++) {
            /* after the baseline: the first megabyte of every matrix the checked step reads */
            PROBE_CHECK_CUDA(cudaMemset((uint8_t *)scaled.gate_up.packed + first_byte + matrix * matrix_bytes,
                                        SCALED_CORRUPTED_BYTE, SCALED_CORRUPTED_BYTES));
        }
    }
    if (probe_flag_has_word(SCALED_FLAG_BREAK, "exactness")) {
        scaled_check_variant(probe_case, &scaled_faults[SCALED_SLICE_ORDER], "baseline_slices_by_rows",
                             SCALED_INSTANCE_COUNT, 0, scaled_job_rows(rows, stride) > 1u);
    }
    SCALED_SM_120A_ONLY(if (probe_flag_has_word(SCALED_FLAG_BREAK, "scale_lanes")) {
        scaled_check_variant(probe_case, &scaled_faults[SCALED_SCALE_LANES], SCALED_FRESH "_lanes_exchanged",
                             SCALED_INSTANCE_COUNT, 1, 0);
    })
    probe_cuda_check_guards(probe_case->name);
}
/* ---- timing callbacks and verdicts ---- */
static int scaled_prepare_step(void *, const struct probe_case *probe_case, uint32_t instance, uint64_t step,
                               uint32_t slot)
{
    struct scaled_plan *plan = &scaled.plans[slot < PROBE_STREAM_STEPS ? slot : 0];      /* a slot past them: refused */
    scaled_make_plan(plan, probe_case, instance, step, slot, 0, scaled_case_rows(probe_case));
    return scaled_plan_is_ready(plan);
}
static uint32_t scaled_launch_step(void *, const struct probe_case *, uint32_t instance, uint32_t slot)
{
    scaled_queue_plan(&scaled_variants[instance], &scaled.plans[slot]);
    return SCALED_STEP_LAUNCHES;
}
/* ROUND_1.md, unit 3b: the class E candidate against today's form at one row and at 64 rows a job. */
static void scaled_verdicts(void *, const struct probe_case *probe_case, const struct probe_result *results,
                            double noise_ratio)
{
    const uint32_t rows = scaled_case_rows(probe_case), fresh = probe_instance_index(SCALED_FRESH);
    const int is_wide = rows == SCALED_MOST_ROWS && scaled.wide;
    char figure[SCALED_NAME_BYTES];
    if (fresh >= SCALED_INSTANCE_COUNT || (rows != 1u && !is_wide)) { return; }      /* sm_80; a case without a bar */
    PROBE_REQUIRE(snprintf(figure, sizeof(figure), "scaled_fresh_at_%s", probe_case->name) > 0, "a figure's name");
    probe_verdict(figure, scaled_bars[is_wide].ratio, scaled_bars[is_wide].microseconds, results[fresh].ratio,
                  noise_ratio, results[fresh].same_sign, results[fresh].rounds);
}
static struct probe_program scaled_program = {      /* no context: one card; no time_case: the harness's rounds */
    "block_scaled", scaled_flags, PROBE_COUNT_OF(scaled_flags), scaled_instances, SCALED_INSTANCE_COUNT, NULL,
    scaled_configure, scaled_open_card, scaled_describe_case, scaled_check_case, scaled_prepare_step,
    scaled_launch_step, probe_cuda_mark_start, probe_cuda_elapsed_microseconds, scaled_verdicts, scaled_close_card,
    NULL};
int main(int argument_count, char **arguments) { return probe_main(argument_count, arguments, &scaled_program); }
