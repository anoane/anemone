/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * native_expert_forms_probe: round 1, package 2 of the speed campaign (research/speed/ROUND_1.md; ledger N1, N2, N9,
 * N13): which exact form of the checkpoint's routed experts is fastest on a CMP 170HX, by fewer instructions per
 * load (the operand forms) and by fewer idle issue slots (loads ahead, launch bounds, persistent blocks, slices),
 * alone and together; the RTX PRO 6000 is the control. Kernels and levers: native_expert_forms_kernels.cuh. A step
 * is one layer's routed experts for the rows of a case, five launches; step s reads the experts from (s * span) on
 * in a ring of 128, span being the experts a step of the case reads: nothing a card's L2 still holds.
 *
 * Instances: the table at FORMS_BASELINE and FORMS_OTHERS, interleaved in one run by probe_harness.h; one whose block
 * no SM holds by its registers (the assembler's report) is not launched: its lines say n=0. live_sums:1 launches the
 * baseline's kernels in a case whose rows share experts: its lines there are the baseline's. Cases: --rows by
 * --overlap.
 *
 * Checked per case before anything is timed, on step 0, whose vectors hold the corners (forms_input_value), every
 * instance twice: the whole step, and down alone on corner vectors (the hidden step rounds gate and up to 8 bits;
 * down's fp32 sums are where a form's arithmetic shows whole):
 *   quantized_*, operand_*  of the vector a launch begins with (a row's input; a pair's corner vector): codes and
 *                           exponents against a host quantizer written from the definition, and the operand words
 *                           (with CARRIED16's unfit word) against what the formats' fields say: bit for bit
 *   host_reference_*        the product made from that vector (the hidden vector; down's output) for two (row,
 *                           expert) pairs against double precision from the stored words, fed with the card's own
 *                           codes: the hidden vector to 2e-3 of the rms, down's fp32 output to 1e-6 (CONVENTIONS
 *                           5.1; the instances without a structural lever; a form whose bits differ)
 *   bits_against_codes*     every output word against the codes instance of the same slice count: 0 may differ. For
 *                           a carried operand a difference is a finding, not a fault: its line says verdict=reported,
 *                           the instance's class becomes C and the host reference judges it
 *   rows_exact*             every row launched again alone: 0 words of its outputs and of its sum may differ
 *   guard_bytes; DIGEST     nothing wrote behind a card array; all outputs, for the two CMP 170HX to agree on
 * Every launch, checked or timed, is validated on the host first (forms_plan_is_valid). Each check is shown failing:
 * --corrupt (after the baseline's launches a megabyte of the first checked gate matrix and of its down matrix are
 * overwritten on the card: every later instance's host_reference_* and, against the baseline, bits_* report),
 * --break-route (three launches past one bound each: all refused), --break-guard, and --break exactness (a kernel
 * that adds the slices of a job of several rows from the second on: rows_exact reports where rows share an expert),
 * operand (one exponent moved in what the host expects of carried words) or quantization (the host's floor
 * doubled). --break is a break flag of the harness: its first case is checked, nothing is timed.
 * VERDICT figures, of the instances that kept their bits (the bars are a CMP 170HX's; the PRO only measures: its bar
 * is the batch's CONTROL line): carried_operand_alone (the better of the two), ahead_or_bound_alone, best (one row;
 * eight unshared), one_slice_fed (no bar: ledger N13). Not shown here: how an instruction rounds its own sum (unit
 * 3a). A carried form with 0 differing words is exact on these vectors and this card, no further.
 *
 *   native_expert_forms_probe [--rows 1,6,8] [--overlap 0,50,100] [--break NAME] and the flags of probe_harness.h
 *
 * Measurement tool: not part of the engine.
 */
#include "probe_harness.h"
#include "native_expert_forms_kernels.cuh"

enum {
    FORMS_EXPERTS = 128,                         /* the ring: 2.2 GiB, 17 times the L2 of the PRO */
    FORMS_INPUT_BLOCKS = FORMS_HIDDEN_SIZE / FORMS_VALUES_PER_BLOCK,     /* the quantizer's block: a thread each */
    FORMS_HIDDEN_BLOCKS = FORMS_INNER_SIZE / FORMS_VALUES_PER_BLOCK,
    FORMS_GATE_UP_TILES = FORMS_INNER_SIZE / FORMS_OUTPUTS_PER_TILE,
    FORMS_DOWN_TILES = FORMS_HIDDEN_SIZE / FORMS_OUTPUTS_PER_TILE,
    FORMS_EXPERT_BYTES = FORMS_MATRIX_KINDS * FORMS_PACKED_WORDS_PER_MATRIX * FORMS_WORD_BYTES,  /* of its codes */
    FORMS_EXPERT_SCALE_BYTES = FORMS_MATRIX_KINDS * FORMS_SCALE_WORDS_PER_MATRIX * FORMS_WORD_BYTES,
    FORMS_STEP_LAUNCHES = 5, FORMS_FILL_BLOCKS_PER_SM = 8,
    FORMS_GRID_AXES = 2,                         /* of a product's grid: tiles by jobs, or persistent blocks by 1 */
    FORMS_REGISTER_FILES = 4, FORMS_FILE_REGISTERS = 16384, FORMS_REGISTER_STEP = 8,     /* of an SM: forms_sm_holds */
    FORMS_BLOCKS_PER_SM_80 = 32, FORMS_BLOCKS_PER_SM_120A = 24,         /* "Block Limit SM" of batches h02 and i00b */
    FORMS_OVERLAP_STEP = 50, FORMS_PERCENT = 100,
    FORMS_ROUTE_WEIGHT_STEPS = 32,               /* synthetic route weights 1/64 .. 32/64 */
    FORMS_CHECKED_STEP = 0, FORMS_CHECKED_PAIRS = 2,
    FORMS_CHECK_STRIDE = 13,                     /* every 13th output is compared: all lanes, over the tiles */
    FORMS_WHOLE_STEP = 0, FORMS_DOWN_ALONE = 1, FORMS_PHASES = 2,       /* the two launches of a check */
    FORMS_CORRUPTED_BYTES = 1024 * 1024, FORMS_CORRUPTED_BYTE = 0x5A,
    FORMS_STALE_SHARED = 0xFF, FORMS_STALE_ALONE = 0xEE,   /* outputs before a launch: no unwritten word can match */
    FORMS_ROW_FLOATS = FORMS_EXPERTS_PER_ROW * FORMS_HIDDEN_SIZE,
    FORMS_ROW_SUMS_AT = FORMS_MOST_PAIRS * FORMS_HIDDEN_SIZE,           /* a result: [pair][hidden size], then */
    FORMS_RESULT_FLOATS = FORMS_ROW_SUMS_AT + FORMS_MOST_ROWS * FORMS_HIDDEN_SIZE,       /* [row][hidden size] */
    FORMS_QUANTIZED_BYTES = FORMS_HIDDEN_SIZE + FORMS_INPUT_BLOCKS,     /* a vector's codes, then its exponents */
    FORMS_HALF_BYTES = 2,                        /* of an operand half: every value has one */
    FORMS_FIGURE_BYTES = 256,
    FORMS_BAR_CARRIED = 0, FORMS_BAR_LEVER, FORMS_BAR_BEST, FORMS_BAR_BEST_EIGHT_ROWS,     /* the rows of forms_bars */
    FORMS_BROKEN_EXPERT = 0, FORMS_BROKEN_PAIR, FORMS_BROKEN_GRID, FORMS_BROKEN_END        /* --break-route, in turn */
};
#define FORMS_FLAG_ROWS "--rows"
#define FORMS_FLAG_OVERLAP "--overlap"
#define FORMS_FLAG_BREAK "--break"
#define FORMS_ROUTE_SEED 0xABCDEFu
#define FORMS_ROUTE_WEIGHT_DIVISOR 64.0f
#define FORMS_TOLERANCE 2.0e-3                   /* CONVENTIONS 5.1: a bfloat16 vector against its rms */
#define FORMS_OUTPUT_TOLERANCE 1.0e-6            /* and an fp32 intermediate: down's sums */
#define FORMS_BYTES_PER_MB 1.0e6
#define FORMS_SM_80 80                           /* -DPROBE_SM of the sm_80 build, for the preprocessor */
#define FORMS_PHASE_NAME(name, phase) ((phase) == FORMS_DOWN_ALONE ? name "_down_alone" : name)
/* A range of a card array against what the array was allocated with: a REFUSED line names the array. */
#define FORMS_FITS(valid, array, first, count, bytes) \
    probe_launch_range(valid, "step", #array, first, count, bytes, probe_cuda_array_bytes(array))

/* acceptance (ROUND_1.md, package 2): a ratio to the run's baseline; its microseconds at 176 (one row), 1,250 (8) */
static const struct {
    double ratio, microseconds;
} forms_bars[] = {{0.92, -14.0}, {0.915, -15.0}, {0.77, -41.0}, {0.80, -250.0}};
static const char *const forms_broken_names[FORMS_BROKEN_END] = {
    "refused_expert_past_the_ring", "refused_pair_past_the_step", "refused_grid_past_the_tiles"};
static struct probe_flag forms_flags[] = {
    PROBE_COMMON_FLAGS,
    PROBE_CASE_NUMBERS(FORMS_FLAG_ROWS, 1, FORMS_MOST_ROWS, "1,6,8", "rows of a step"),
    PROBE_CASE_WORDS(FORMS_FLAG_OVERLAP, "0,50,100", "0", "percent of a row's experts the row before has too"),
    PROBE_WORD_LIST(FORMS_FLAG_BREAK, "exactness,operand,quantization", "", "show that check failing"),
};

struct forms_plan {                              /* one step of one instance as numbers: validated, then launched */
    uint32_t instance, slot, row_count, job_count;
    uint32_t one_row_jobs, hidden_given;         /* no job has two rows; down alone, on hidden vectors as filled */
    uint32_t gate_up_grid[FORMS_GRID_AXES], down_grid[FORMS_GRID_AXES], gate_up_threads, down_threads;
    struct forms_job jobs[FORMS_MOST_PAIRS];
    struct forms_step arguments;                 /* pointers: forms_prepare makes them of the validated numbers */
};
typedef void forms_launcher(const struct forms_plan *plan);
struct forms_variant {                           /* what the instance of the same index is made of */
    forms_launcher *launch;
    uint32_t operand, ahead, persistent, slices, columns;
};

/* The launches of a step, with the blocks and grids its plan was validated with. */
#define FORMS_LAUNCHER(id, operand, columns)                                                                        \
    static void forms_launch_##id(const struct forms_plan *plan)                                                    \
    {                                                                                                               \
        const struct forms_step *step = &plan->arguments;                                                           \
        const uint32_t pairs = plan->row_count * FORMS_EXPERTS_PER_ROW;                                             \
        if ((columns) != FORMS_COLUMNS_PER_LANE && !plan->one_row_jobs) {                                           \
            forms_launch_baseline(plan);         /* a job of several rows needs every column scaled */              \
            return;                                                                                                 \
        }                                                                                                           \
        if (!plan->hidden_given) {                                                                                  \
            forms_quantize_kernel<operand><<<plan->row_count, FORMS_INPUT_BLOCKS>>>(step->input);                   \
            forms_gate_up_##id<<<dim3(plan->gate_up_grid[0], plan->gate_up_grid[1]), plan->gate_up_threads>>>(*step); \
        }                                                                                                           \
        forms_quantize_kernel<operand><<<pairs, FORMS_HIDDEN_BLOCKS>>>(step->hidden);                               \
        forms_down_##id<<<dim3(plan->down_grid[0], plan->down_grid[1]), plan->down_threads>>>(*step);               \
        forms_row_sum_kernel<<<plan->row_count, FORMS_ELEMENT_THREADS>>>(step->pair_output, step->row_output);      \
        PROBE_AFTER_LAUNCH();                                                                                       \
    }
#define FORMS_STAMP(id, name, operand, ahead, bound, blocks, persistent, slices, columns)                           \
    FORMS_PRODUCT_KERNELS(id, FORMS_OPERAND_##operand, ahead, bound, blocks, persistent, slices, columns, 0)        \
    FORMS_LAUNCHER(id, FORMS_OPERAND_##operand, columns)
#define FORMS_INSTANCE(id, name, operand, ahead, bound, blocks, persistent, slices, columns)                        \
    {name, "forms_gate_up_" #id "+forms_down_" #id, 'E', 1, FORMS_BLOCK_THREADS(FORMS_GATE_UP_MATRICES, slices),    \
     (blocks) != 0 ? FORMS_BLOCK_THREADS(FORMS_GATE_UP_MATRICES, slices) : 0, blocks, 0, 0},
#define FORMS_VARIANT(id, name, operand, ahead, bound, blocks, persistent, slices, columns)                         \
    {forms_launch_##id, FORMS_OPERAND_##operand, ahead, persistent, slices, columns},
#if PROBE_SM == FORMS_SM_80                      /* sm_80. 2,048 threads to an SM: more than the PRO's holds */
#define FORMS_SM_80_ONLY(row) row
#else
#define FORMS_SM_80_ONLY(row)
#endif
/* id, name, operand, loads ahead, launch bound and its blocks, persistent, slices, scaled columns. The first row
 * of a slice count is its codes instance, which the others of that count are compared with. */
#define FORMS_BASELINE(X, name) X(baseline, name, CODES, 0, OPEN, 0, 0, 4, 2)
#define FORMS_OTHERS(X)                                                                                             \
    X(placed16, "operand:placed16", PLACED16, 0, OPEN, 0, 0, 4, 2)                                                  \
    X(carried16, "operand:carried16", CARRIED16, 0, OPEN, 0, 0, 4, 2)                                               \
    X(carried_bf16, "operand:carried_bf16", CARRIED_BF16, 0, OPEN, 0, 0, 4, 2)                                      \
    X(ahead_1, "ahead:1", CODES, 1, OPEN, 0, 0, 4, 2)                                                               \
    X(ahead_2, "ahead:2", CODES, 2, OPEN, 0, 0, 4, 2)                                                               \
    X(bound_4, "bounds:256x4", CODES, 0, STATED, 4, 0, 4, 2)                                                        \
    X(bound_5, "bounds:256x5", CODES, 0, STATED, 5, 0, 4, 2)                                                        \
    X(bound_6, "bounds:256x6", CODES, 0, STATED, 6, 0, 4, 2)                                                        \
    FORMS_SM_80_ONLY(X(bound_8, "bounds:256x8", CODES, 0, STATED, 8, 0, 4, 2))                                      \
    X(persistent, "persistent:1", CODES, 0, OPEN, 0, 1, 4, 2)                                                       \
    X(live_sums, "live_sums:1", CODES, 0, OPEN, 0, 0, 4, 1)                                                         \
    X(ahead_bound, "ahead:1,bounds:256x6", CODES, 1, STATED, 6, 0, 4, 2)                                            \
    X(ahead_persistent, "ahead:1,persistent:1", CODES, 1, OPEN, 0, 1, 4, 2)                                         \
    X(bound_persistent, "bounds:256x6,persistent:1", CODES, 0, STATED, 6, 1, 4, 2)                                  \
    X(ahead_bound_persistent, "ahead:1,bounds:256x6,persistent:1", CODES, 1, STATED, 6, 1, 4, 2)                    \
    X(bf16_ahead, "operand:carried_bf16,ahead:1", CARRIED_BF16, 1, OPEN, 0, 0, 4, 2)                                \
    X(bf16_bound, "operand:carried_bf16,bounds:256x6", CARRIED_BF16, 0, STATED, 6, 0, 4, 2)                         \
    X(bf16_persistent, "operand:carried_bf16,persistent:1", CARRIED_BF16, 0, OPEN, 0, 1, 4, 2)                      \
    X(bf16_ahead_bound, "operand:carried_bf16,ahead:1,bounds:256x6", CARRIED_BF16, 1, STATED, 6, 0, 4, 2)           \
    X(bf16_ahead_persistent, "operand:carried_bf16,ahead:1,persistent:1", CARRIED_BF16, 1, OPEN, 0, 1, 4, 2)        \
    X(bf16_bound_persistent, "operand:carried_bf16,bounds:256x6,persistent:1", CARRIED_BF16, 0, STATED, 6, 1, 4, 2) \
    X(bf16_all, "operand:carried_bf16,ahead:1,bounds:256x6,persistent:1", CARRIED_BF16, 1, STATED, 6, 1, 4, 2)      \
    X(slices_1, "slices:1", CODES, 0, OPEN, 0, 0, 1, 2)                                                             \
    X(slices_1_fed, "slices:1,ahead:1,persistent:1", CODES, 1, OPEN, 0, 1, 1, 2)                                    \
    X(slices_2, "slices:2", CODES, 0, OPEN, 0, 0, 2, 2)                                                             \
    X(slices_2_fed, "slices:2,ahead:1,persistent:1", CODES, 1, OPEN, 0, 1, 2, 2)                                    \
    X(slices_8, "slices:8", CODES, 0, OPEN, 0, 0, 8, 2)                                                             \
    X(slices_8_fed, "slices:8,ahead:1,persistent:1", CODES, 1, OPEN, 0, 1, 8, 2)                                    \
    X(slices_16, "slices:16", CODES, 0, OPEN, 0, 0, 16, 2)                                                          \
    X(slices_16_fed, "slices:16,ahead:1,persistent:1", CODES, 1, OPEN, 0, 1, 16, 2)
FORMS_BASELINE(FORMS_STAMP, "baseline")
FORMS_OTHERS(FORMS_STAMP)
/* never timed: the baseline's kernel with its slices added from one that follows the rows of a job */
FORMS_PRODUCT_KERNELS(slices_by_rows, FORMS_OPERAND_CODES, 0, OPEN, 0, 0, 4, FORMS_COLUMNS_PER_LANE, 1)
FORMS_LAUNCHER(slices_by_rows, FORMS_OPERAND_CODES, FORMS_COLUMNS_PER_LANE)
static struct probe_instance forms_instances[] = {
    FORMS_BASELINE(FORMS_INSTANCE, "baseline") FORMS_BASELINE(FORMS_INSTANCE, "baseline_again")
    FORMS_OTHERS(FORMS_INSTANCE)};
static const struct forms_variant forms_variants[] = {
    FORMS_BASELINE(FORMS_VARIANT, "baseline") FORMS_BASELINE(FORMS_VARIANT, "baseline_again")
    FORMS_OTHERS(FORMS_VARIANT)};
#define FORMS_INSTANCE_COUNT PROBE_COUNT_OF(forms_instances)

static struct {
    struct forms_step card;                      /* the card's arrays, each at its first element */
    uint32_t sm_count, warps_per_sm;
    struct forms_plan plans[PROBE_STREAM_STEPS], checked;    /* of each slot's timed step; of the checked launch */
} forms;

/* ---- routes, plans, check 10 ---- */

static uint32_t forms_case_rows(const struct probe_case *probe_case)
{
    return (uint32_t)probe_case_value(probe_case, FORMS_FLAG_ROWS);
}

/* The experts a further row of the case brings (6, 3 or 0), and those a step reads: consecutive ones of the ring. */
static uint32_t forms_case_stride(const struct probe_case *probe_case)
{
    const uint32_t overlap = (uint32_t)probe_case_value(probe_case, FORMS_FLAG_OVERLAP) * FORMS_OVERLAP_STEP;
    return FORMS_EXPERTS_PER_ROW * (FORMS_PERCENT - overlap) / FORMS_PERCENT;
}

static uint32_t forms_case_span(const struct probe_case *probe_case)
{
    return (forms_case_rows(probe_case) - 1u) * forms_case_stride(probe_case) + FORMS_EXPERTS_PER_ROW;
}

/* A step's experts lie a span further on in the ring each step; row r takes six from the (r * stride)-th on. */
static uint32_t forms_route_expert(const struct probe_case *probe_case, uint64_t step, uint32_t row, uint32_t slot)
{
    const uint32_t span = forms_case_span(probe_case);
    return (uint32_t)(step * span % (FORMS_EXPERTS - span + 1u)) + row * forms_case_stride(probe_case) + slot;
}

static float forms_route_weight(uint64_t step, uint32_t row, uint32_t slot)
{
    const uint32_t code =
        forms_mix_bits(~FORMS_ROUTE_SEED + (uint32_t)step * FORMS_STEP_STRIDE + row * FORMS_EXPERTS_PER_ROW + slot);
    return (float)(1u + code % FORMS_ROUTE_WEIGHT_STEPS) / FORMS_ROUTE_WEIGHT_DIVISOR;
}

/* Blocks of `threads` threads of an instance that an SM holds by the registers of the assembler's report (the most
 * of its two kernels; no report: the fewest). An SM has four files of 16,384 registers, each for whole warps, a thread
 * taking a multiple of 8: all 44 "Block Limit Registers" of batches h02 and i00b. 0: the driver refuses the launch. */
static uint32_t forms_sm_holds(uint32_t instance, uint32_t threads)
{
    const int32_t reported = forms_instances[instance].registers;
    const uint32_t steps = ((reported > 0 ? (uint32_t)reported : 1u) + FORMS_REGISTER_STEP - 1u) / FORMS_REGISTER_STEP;
    const uint32_t of_a_warp = steps * FORMS_REGISTER_STEP * FORMS_LANES_PER_WARP;
    return FORMS_REGISTER_FILES * (FORMS_FILE_REGISTERS / of_a_warp) * FORMS_LANES_PER_WARP / threads;
}

/* The grid of a product over `jobs` jobs. Persistent: one row of as many blocks as the card holds at once, by an
 * SM's warps, its blocks and the instance's registers, and no more than there are units (a tile of a job). */
static void forms_set_grid(uint32_t *grid, uint32_t instance, uint32_t threads, uint32_t tiles, uint32_t jobs)
{
    const uint32_t by_blocks = PROBE_SM == PROBE_SM_80 ? FORMS_BLOCKS_PER_SM_80 : FORMS_BLOCKS_PER_SM_120A;
    const uint32_t by_registers = forms_sm_holds(instance, threads);
    uint32_t blocks = forms.warps_per_sm * FORMS_LANES_PER_WARP / threads;
    blocks = blocks < by_registers ? blocks : by_registers;
    blocks = (blocks == 0 ? 1u : blocks < by_blocks ? blocks : by_blocks) * forms.sm_count;
    grid[0] = !forms_variants[instance].persistent ? tiles : blocks < tiles * jobs ? blocks : tiles * jobs;
    grid[1] = forms_variants[instance].persistent ? 1u : jobs;
}

/* The plan of step `step` of a case for rows first_row .. first_row + row_count - 1, which fill the batch from its
 * slot 0: one job per expert, its columns in the order of the rows. */
static void forms_make_plan(struct forms_plan *plan, const struct probe_case *probe_case, uint32_t instance,
                            uint64_t step, uint32_t slot, uint32_t first_row, uint32_t row_count)
{
    plan->instance = instance;
    plan->slot = slot;
    plan->row_count = row_count;
    plan->job_count = 0;
    plan->one_row_jobs = 1;
    plan->hidden_given = 0;
    for (uint32_t pair = 0; pair < row_count * FORMS_EXPERTS_PER_ROW; pair++) {
        const uint32_t row = first_row + pair / FORMS_EXPERTS_PER_ROW, row_slot = pair % FORMS_EXPERTS_PER_ROW;
        const uint32_t expert = forms_route_expert(probe_case, step, row, row_slot);
        uint32_t job_index = 0;
        while (job_index < plan->job_count && plan->jobs[job_index].expert_index != expert) {
            job_index++;
        }
        struct forms_job *job = &plan->jobs[job_index];
        if (job_index == plan->job_count) {
            plan->job_count++;
            job->expert_index = expert;
            job->row_count = 0;
            for (uint32_t column = 0; column < FORMS_MOST_ROWS; column++) {
                job->pair_index[column] = pair;  /* a column not in use reads what column 0 reads */
                job->route_weight[column] = 0.0f;
            }
        }
        job->pair_index[job->row_count] = pair;
        job->route_weight[job->row_count++] = forms_route_weight(step, row, row_slot);
        plan->one_row_jobs &= job->row_count == 1u ? 1u : 0u;
    }
    plan->gate_up_threads = FORMS_BLOCK_THREADS(FORMS_GATE_UP_MATRICES, forms_variants[instance].slices);
    plan->down_threads = FORMS_BLOCK_THREADS(FORMS_DOWN_MATRICES, forms_variants[instance].slices);
    forms_set_grid(plan->gate_up_grid, instance, plan->gate_up_threads, FORMS_GATE_UP_TILES, plan->job_count);
    forms_set_grid(plan->down_grid, instance, plan->down_threads, FORMS_DOWN_TILES, plan->job_count);
}

/* Check 10: 1 if every launch of the plan stays inside the card's arrays as they were allocated, the card's limits
 * and the instance's block and bound; 0: it printed why not. What a kernel indexes by is held to its range first;
 * the sums and products after that are of such values, in 64 bits. */
static int forms_plan_is_valid(const struct forms_plan *plan)
{
    const struct forms_step *card = &forms.card;
    const int persistent = forms_variants[plan->instance].persistent != 0;
    const uint64_t pairs = (uint64_t)plan->row_count * FORMS_EXPERTS_PER_ROW, jobs = plan->job_count;
    const uint64_t chain_bytes = (persistent ? FORMS_CHAIN_TURNS : 1) * FORMS_ELEMENTS_PER_LANE * sizeof(float);
    const char *product = "forms_product";
    int valid = 1;
    probe_launch_value(&valid, "step", "slot", plan->slot, 0, PROBE_STREAM_STEPS - 1u);
    probe_launch_value(&valid, "step", "rows", plan->row_count, 1, FORMS_MOST_ROWS);
    probe_launch_value(&valid, "step", "jobs", jobs, 1, pairs);
    for (uint32_t job = 0; valid && job < jobs; job++) {
        probe_launch_value(&valid, product, "expert of a job", plan->jobs[job].expert_index, 0, FORMS_EXPERTS - 1u);
        probe_launch_value(&valid, product, "rows of a job", plan->jobs[job].row_count, 1, FORMS_MOST_ROWS);
        for (uint32_t column = 0; column < FORMS_MOST_ROWS; column++) {
            probe_launch_value(&valid, product, "pair of a column", plan->jobs[job].pair_index[column], 0, pairs - 1u);
        }
    }
    if (!valid) {
        return 0;
    }
    /* the quantizer and the kernels of a thread a value, at the largest grid and block the launchers give them */
    probe_launch_geometry(&valid, "forms_quantize_kernel", NULL, pairs, 1, 1, FORMS_INPUT_BLOCKS,
                          FORMS_INPUT_BLOCKS * sizeof(uint32_t));
    probe_launch_geometry(&valid, "forms_element_kernels", NULL, pairs, 1, 1, FORMS_ELEMENT_THREADS, 0);
    for (uint32_t matrices = FORMS_DOWN_MATRICES; matrices <= FORMS_GATE_UP_MATRICES; matrices++) {
        const int is_down = matrices == FORMS_DOWN_MATRICES;
        const char *kernel = is_down ? "forms_down" : "forms_gate_up";
        const uint32_t *grid = is_down ? plan->down_grid : plan->gate_up_grid;
        const uint64_t threads = is_down ? plan->down_threads : plan->gate_up_threads;
        const uint64_t tiles = is_down ? FORMS_DOWN_TILES : FORMS_GATE_UP_TILES;
        const uint64_t own_threads = FORMS_BLOCK_THREADS(matrices, forms_variants[plan->instance].slices);
        probe_launch_geometry(&valid, kernel, is_down ? NULL : &forms_instances[plan->instance], grid[0], grid[1], 1,
                              threads, threads * chain_bytes);
        probe_launch_value(&valid, kernel, "threads of its block", threads, own_threads, own_threads);
        probe_launch_value(&valid, kernel, "grid x: the tiles, or persistent blocks with a unit each", grid[0],
                           persistent ? 1 : tiles, persistent ? tiles * jobs : tiles);
        probe_launch_value(&valid, kernel, "grid y: the jobs, or 1 where blocks persist", grid[1],
                           persistent ? 1 : jobs, persistent ? 1 : jobs);
    }
    FORMS_FITS(&valid, card->packed, 0, FORMS_EXPERTS, FORMS_EXPERT_BYTES);
    FORMS_FITS(&valid, card->scales, 0, FORMS_EXPERTS, FORMS_EXPERT_SCALE_BYTES);
    FORMS_FITS(&valid, card->jobs, (uint64_t)plan->slot * FORMS_MOST_PAIRS, jobs, sizeof(struct forms_job));
    FORMS_FITS(&valid, card->pair_output, 0, pairs, FORMS_HIDDEN_SIZE * sizeof(float));
    FORMS_FITS(&valid, card->row_output, 0, plan->row_count, FORMS_HIDDEN_SIZE * sizeof(float));
    for (uint32_t set = 0; set <= 1u; set++) {           /* the rows' vectors, their values a set a slot; the pairs' */
        const struct forms_vectors *vectors = set == 0 ? &card->input : &card->hidden;
        const uint64_t count = set == 0 ? plan->row_count : pairs, size = vectors->size;
        const uint64_t first_values = set == 0 ? (uint64_t)plan->slot * FORMS_MOST_ROWS : 0;
        FORMS_FITS(&valid, vectors->values, first_values, count, size * sizeof(float));
        FORMS_FITS(&valid, vectors->codes, 0, count, size);
        FORMS_FITS(&valid, vectors->exponents, 0, count, size / FORMS_VALUES_PER_BLOCK);
        FORMS_FITS(&valid, vectors->operands, 0, count, size * FORMS_HALF_BYTES);
        FORMS_FITS(&valid, vectors->unfit, 0, count, sizeof(uint32_t));
    }
    return valid;
}

/* What a validated plan needs on the card that is not timed: its pointers, its jobs, and the values of the step's
 * vectors from first_row on: the rows' inputs, or for down alone the pairs' hidden vectors. */
static void forms_prepare(struct forms_plan *plan, uint64_t step, uint32_t first_row, uint32_t corners)
{
    struct forms_step *arguments = &plan->arguments;
    const uint32_t per_row = plan->hidden_given ? FORMS_EXPERTS_PER_ROW : 1;         /* vectors to fill */
    *arguments = forms.card;
    arguments->jobs += (size_t)plan->slot * FORMS_MOST_PAIRS;
    arguments->input.values += (size_t)plan->slot * FORMS_MOST_ROWS * FORMS_HIDDEN_SIZE;
    arguments->job_count = plan->job_count;
    PROBE_CHECK_CUDA(cudaMemcpy((struct forms_job *)arguments->jobs, plan->jobs,
                                plan->job_count * sizeof(struct forms_job), cudaMemcpyHostToDevice));
    forms_fill_values_kernel<<<plan->row_count * per_row, FORMS_ELEMENT_THREADS>>>(
        plan->hidden_given ? arguments->hidden.values : arguments->input.values,
        plan->hidden_given ? FORMS_INNER_SIZE : FORMS_HIDDEN_SIZE, first_row * per_row, (uint32_t)step, corners);
    PROBE_CHECK_CUDA(cudaGetLastError());
    PROBE_CHECK_CUDA(cudaDeviceSynchronize());
}

/* ---- the card ---- */

/* vector_count vectors of a size and what the quantizer writes of them; value_vectors: a set of values a slot. */
static void forms_allocate_vectors(struct forms_vectors *vectors, uint64_t value_vectors, uint64_t vector_count,
                                   uint32_t size, uint32_t pairs_per_vector)
{
    vectors->values = (float *)probe_cuda_allocate(value_vectors * size * sizeof(float));
    vectors->codes = (uint32_t *)probe_cuda_allocate(vector_count * size);
    vectors->exponents = (uint8_t *)probe_cuda_allocate(vector_count * (size / FORMS_VALUES_PER_BLOCK));
    vectors->operands = (uint32_t *)probe_cuda_allocate(vector_count * size * FORMS_HALF_BYTES);
    vectors->unfit = (uint32_t *)probe_cuda_allocate(vector_count * sizeof(uint32_t));
    /* what only some operand forms write, and every check reads */
    PROBE_CHECK_CUDA(cudaMemset(vectors->operands, 0, vector_count * size * FORMS_HALF_BYTES));
    PROBE_CHECK_CUDA(cudaMemset(vectors->unfit, 0, vector_count * sizeof(uint32_t)));
    vectors->size = size;
    vectors->pairs_per_vector = pairs_per_vector;
}

static int forms_open_card(void *, struct probe_card_facts *facts)
{
    uint32_t *stored[] = {NULL, NULL};           /* the experts' codes, their scales */
    if (!probe_cuda_open(facts)) {
        return 0;
    }
    probe_run.breaking |= (uint32_t)probe_flag_given(FORMS_FLAG_BREAK);     /* a list is no break flag to the harness */
    forms.sm_count = (uint32_t)facts->sm_count;
    forms.warps_per_sm = facts->warps_per_sm;
    for (uint32_t instance = 0; instance < FORMS_INSTANCE_COUNT; instance++) {   /* a block no SM holds: n=0 */
        forms_instances[instance].available = forms_sm_holds(instance, forms_instances[instance].block_threads) != 0;
    }
    stored[0] = (uint32_t *)probe_cuda_allocate((uint64_t)FORMS_EXPERTS * FORMS_EXPERT_BYTES);
    stored[1] = (uint32_t *)probe_cuda_allocate((uint64_t)FORMS_EXPERTS * FORMS_EXPERT_SCALE_BYTES);
    forms.card.packed = (const uint4 *)stored[0];
    forms.card.scales = stored[1];
    forms.card.jobs = (const struct forms_job *)probe_cuda_allocate(PROBE_STREAM_STEPS * FORMS_MOST_PAIRS
                                                                    * sizeof(struct forms_job));
    forms.card.pair_output = (float *)probe_cuda_allocate(FORMS_ROW_SUMS_AT * sizeof(float));
    forms.card.row_output = (float *)probe_cuda_allocate(FORMS_MOST_ROWS * FORMS_HIDDEN_SIZE * sizeof(float));
    forms_allocate_vectors(&forms.card.input, PROBE_STREAM_STEPS * FORMS_MOST_ROWS, FORMS_MOST_ROWS,
                           FORMS_HIDDEN_SIZE, FORMS_EXPERTS_PER_ROW);
    forms_allocate_vectors(&forms.card.hidden, FORMS_MOST_PAIRS, FORMS_MOST_PAIRS, FORMS_INNER_SIZE, 1);
    for (uint32_t is_scale = 0; is_scale <= 1u; is_scale++) {    /* by a kernel: 2.4 GB are seconds over the link */
        const uint32_t blocks = forms.sm_count * FORMS_FILL_BLOCKS_PER_SM;
        int valid = 1;
        probe_launch_geometry(&valid, "forms_fill_stored_kernel", NULL, blocks, 1, 1, FORMS_ELEMENT_THREADS, 0);
        PROBE_REQUIRE(valid, "the launch that fills the stored experts was refused");
        forms_fill_stored_kernel<<<blocks, FORMS_ELEMENT_THREADS>>>(
            stored[is_scale], is_scale, probe_cuda_array_bytes(stored[is_scale]) / sizeof(uint32_t));
    }
    PROBE_CHECK_CUDA(cudaGetLastError());
    PROBE_CHECK_CUDA(cudaDeviceSynchronize());
    facts->ring_bytes = (uint64_t)FORMS_EXPERTS * (FORMS_EXPERT_BYTES + FORMS_EXPERT_SCALE_BYTES);
    return 1;
}

static void forms_close_card(void *) { probe_cuda_close(); }

static void forms_describe_case(void *, const struct probe_case *probe_case, struct probe_case_facts *facts)
{
    facts->format = "native";
    facts->part = "experts";
    facts->rows = forms_case_rows(probe_case);
    facts->experts = forms_case_span(probe_case);
    facts->overlap = (int64_t)probe_case_value(probe_case, FORMS_FLAG_OVERLAP) * FORMS_OVERLAP_STEP;
    facts->megabytes = (double)facts->experts * (FORMS_EXPERT_BYTES + FORMS_EXPERT_SCALE_BYTES) / FORMS_BYTES_PER_MB;
}

static uint64_t forms_differing(const void *first, const void *second, size_t element_count, size_t element_bytes)
{
    uint64_t differing = 0;
    for (size_t element = 0; element < element_count; element++) {
        const char *first_element = (const char *)first + element * element_bytes;
        differing += memcmp(first_element, (const char *)second + element * element_bytes, element_bytes) != 0;
    }
    return differing;
}

/* ---- checks ---- */

/* One launch of the checked step outside the timing, for rows first_row .. , whole or down alone, by an instance's
 * launcher or one that is no instance's; its outputs made stale first. result gets FORMS_RESULT_FLOATS: the expert
 * outputs as down wrote them, then the rows' sums. */
static void forms_run_checked(forms_launcher *launch, const struct probe_case *probe_case, uint32_t instance,
                              uint32_t first_row, uint32_t row_count, uint32_t phase, int stale_byte, float *result)
{
    struct forms_plan *plan = &forms.checked;
    const size_t sum_bytes = (FORMS_RESULT_FLOATS - FORMS_ROW_SUMS_AT) * sizeof(float);
    forms_make_plan(plan, probe_case, instance, FORMS_CHECKED_STEP, 0, first_row, row_count);
    plan->hidden_given = phase == FORMS_DOWN_ALONE;
    PROBE_REQUIRE(forms_plan_is_valid(plan), "the checked step was refused");
    forms_prepare(plan, FORMS_CHECKED_STEP, first_row, 1);
    PROBE_CHECK_CUDA(cudaMemset(plan->arguments.pair_output, stale_byte, FORMS_ROW_SUMS_AT * sizeof(float)));
    PROBE_CHECK_CUDA(cudaMemset(plan->arguments.row_output, stale_byte, sum_bytes));
    launch(plan);
    PROBE_CHECK_CUDA(cudaGetLastError());
    PROBE_CHECK_CUDA(cudaDeviceSynchronize());
    PROBE_CHECK_CUDA(cudaMemcpy(result, plan->arguments.pair_output, FORMS_ROW_SUMS_AT * sizeof(float),
                                cudaMemcpyDeviceToHost));
    PROBE_CHECK_CUDA(cudaMemcpy(result + FORMS_ROW_SUMS_AT, plan->arguments.row_output, sum_bytes,
                                cudaMemcpyDeviceToHost));
}

/* The checked launch that just ran, against the host, for two pairs (the first row at its last expert, the last row
 * at its first: where rows share experts, column 0 and the last column of a job). Of the vector the launch began
 * with (the whole step: the row's input; down alone: the pair's corner vector): the card's codes and exponents
 * against the host quantizer's; and what an operand form's words hold (and its unfit word) against the formats'
 * fields applied to the card's own codes and exponents: the code's value times 2^-8 (placed), or the dequantized
 * value times 2^7 where the block's scale is at most 1, or times 2^100 (--break operand: one exponent moved). Then
 * the product made from that vector (the pair's hidden vector; its output) against double precision, fed with the
 * card's own codes and exponents. A not-a-number from the card makes an error that no limit admits. */
static void forms_check_against_host(const struct probe_case *probe_case, uint32_t instance, uint32_t phase,
                                     const float *result)
{
    static float values[FORMS_HIDDEN_SIZE], hidden[FORMS_INNER_SIZE];
    static uint32_t codes[FORMS_HIDDEN_SIZE / FORMS_WORD_BYTES];
    static uint32_t operands[FORMS_HIDDEN_SIZE / FORMS_HALVES_PER_REGISTER];
    static uint8_t host_quantized[FORMS_QUANTIZED_BYTES], card_quantized[FORMS_QUANTIZED_BYTES];
    const struct forms_step *step = &forms.checked.arguments;
    const struct forms_vectors *vectors = phase == FORMS_WHOLE_STEP ? &step->input : &step->hidden;
    const uint32_t size = vectors->size, block_count = size / FORMS_VALUES_PER_BLOCK;
    const uint32_t output_size = phase == FORMS_WHOLE_STEP ? FORMS_INNER_SIZE : FORMS_HIDDEN_SIZE;
    const uint32_t operand = forms_variants[instance].operand;
    const int carried_by = operand == FORMS_OPERAND_CARRIED16 ? FORMS_CARRIED16_EXPONENT : FORMS_CARRIED_BF16_EXPONENT;
    const int operand_broken = probe_flag_has_word(FORMS_FLAG_BREAK, "operand");
    const int quantizer_broken = probe_flag_has_word(FORMS_FLAG_BREAK, "quantization");
    const float host_floor = quantizer_broken ? FORMS_AMAX_FLOOR + FORMS_AMAX_FLOOR : FORMS_AMAX_FLOOR;
    const char *name = forms_instances[instance].name;
    for (uint32_t checked = 0; checked < FORMS_CHECKED_PAIRS; checked++) {
        const uint32_t pair = checked == 0 ? FORMS_EXPERTS_PER_ROW - 1u
                                           : (forms_case_rows(probe_case) - 1u) * FORMS_EXPERTS_PER_ROW;
        const uint32_t row = pair / FORMS_EXPERTS_PER_ROW, slot = pair % FORMS_EXPERTS_PER_ROW;
        const uint32_t expert = forms_route_expert(probe_case, FORMS_CHECKED_STEP, row, slot);
        const size_t vector = phase == FORMS_WHOLE_STEP ? row : pair;
        uint32_t unfit = 0, host_unfit = 0, moved_block = PROBE_NONE;
        uint64_t operand_differing = 0;
        double error = 0.0, square = 0.0;
        for (uint32_t column = 0; column < size; column++) {
            values[column] = forms_input_value((uint32_t)vector, column, FORMS_CHECKED_STEP, 1);
        }
        forms_host_quantize(values, size, host_floor, host_quantized);
        PROBE_INDEX_CHECK(vector, phase == FORMS_WHOLE_STEP ? FORMS_MOST_ROWS : FORMS_MOST_PAIRS);
        PROBE_CHECK_CUDA(cudaMemcpy(codes, vectors->codes + vector * (size / FORMS_WORD_BYTES), size,
                                    cudaMemcpyDeviceToHost));
        PROBE_CHECK_CUDA(cudaMemcpy(card_quantized + size, vectors->exponents + vector * block_count, block_count,
                                    cudaMemcpyDeviceToHost));
        PROBE_CHECK_CUDA(cudaMemcpy(operands, vectors->operands + vector * (size / FORMS_HALVES_PER_REGISTER),
                                    size * FORMS_HALF_BYTES, cudaMemcpyDeviceToHost));
        PROBE_CHECK_CUDA(cudaMemcpy(&unfit, vectors->unfit + vector, sizeof(unfit), cudaMemcpyDeviceToHost));
        PROBE_CHECK_CUDA(cudaMemcpy(hidden, step->hidden.values + (size_t)pair * FORMS_INNER_SIZE, sizeof(hidden),
                                    cudaMemcpyDeviceToHost));
        for (uint32_t block = 0; operand_broken && moved_block == PROBE_NONE && block < block_count; block++) {
            moved_block = card_quantized[size + block] <= FORMS_E8M0_BIAS ? block : moved_block;  /* a carried one */
        }
        for (uint32_t place = 0; place < size; place++) {        /* every byte of every word of every block */
            const uint32_t block = place / FORMS_VALUES_PER_BLOCK, byte = place % FORMS_WORD_BYTES;
            const uint32_t lane_place = place % FORMS_VALUES_PER_BLOCK / (FORMS_WORD_BYTES * FORMS_STEPS_PER_BLOCK);
            const uint32_t step_in_block = place / FORMS_WORD_BYTES % FORMS_STEPS_PER_BLOCK;
            const uint32_t word = forms_activation_word(block, lane_place, step_in_block);
            const uint32_t value_at =
                block * FORMS_VALUES_PER_BLOCK + forms_activation_offset(step_in_block, lane_place, byte);
            const uint32_t held = operands[word * FORMS_REGISTERS_PER_OPERAND_B + byte % FORMS_REGISTERS_PER_OPERAND_B]
                                  >> (byte / FORMS_REGISTERS_PER_OPERAND_B * FORMS_HALF_BITS);
            const uint32_t code = (codes[word] >> (byte * FORMS_BYTE_BITS)) & FORMS_BYTE_MASK;
            const int exponent = card_quantized[size + block] + (block == moved_block ? 1 : 0);
            const int carried = operand == FORMS_OPERAND_CARRIED_BF16
                                || (operand == FORMS_OPERAND_CARRIED16 && exponent <= FORMS_E8M0_BIAS);
            const double expected = ldexp(forms_host_e4m3(code), carried ? exponent - FORMS_E8M0_BIAS + carried_by
                                                                         : -FORMS_CODES_EXPONENT);
            const double held_value = forms_host_half(operand, held & FORMS_HALF_MASK);
            card_quantized[value_at] = (uint8_t)code;
            host_unfit |= card_quantized[size + block] > FORMS_E8M0_BIAS ? 1u : 0u;
            operand_differing += held_value != expected || signbit(held_value) != signbit(expected);
        }
        probe_check_count(probe_case->name, name, phase == FORMS_WHOLE_STEP ? "quantized_input" : "quantized_hidden",
                          forms_differing(card_quantized, host_quantized, size + block_count, 1), size + block_count,
                          quantizer_broken);
        if (operand != FORMS_OPERAND_CODES) {
            operand_differing += operand == FORMS_OPERAND_CARRIED16 && unfit != host_unfit;
            probe_check_count(probe_case->name, name, phase == FORMS_WHOLE_STEP ? "operand_input" : "operand_hidden",
                              operand_differing, size + 1u, operand_broken && operand >= FORMS_OPERAND_CARRIED16);
        }
        for (uint32_t output = 0; output < output_size; output += FORMS_CHECK_STRIDE) {
            const double reference =
                phase == FORMS_WHOLE_STEP
                    ? forms_host_hidden(expert, output, card_quantized,
                                        (double)forms_route_weight(FORMS_CHECKED_STEP, row, slot))
                    : forms_host_product(FORMS_DOWN_MATRIX, expert, output, card_quantized);
            const double card_value =
                phase == FORMS_WHOLE_STEP ? hidden[output] : result[(size_t)pair * FORMS_HIDDEN_SIZE + output];
            error += (card_value - reference) * (card_value - reference);
            square += reference * reference;
        }
        probe_check_error(probe_case->name, name,
                          phase == FORMS_WHOLE_STEP ? "host_reference_hidden" : "host_reference_output",
                          sqrt(error / square), phase == FORMS_WHOLE_STEP ? FORMS_TOLERANCE : FORMS_OUTPUT_TOLERANCE,
                          probe_flag_given(PROBE_FLAG_CORRUPT) && instance != PROBE_BASELINE && checked == 0);
    }
}

/* Words that differ between each row of a checked launch (result) and the same row launched alone. */
static uint64_t forms_rows_differing(forms_launcher *launch, const struct probe_case *probe_case, uint32_t instance,
                                     uint32_t phase, const float *result)
{
    static float alone[FORMS_RESULT_FLOATS];
    uint64_t differing = 0;
    for (uint32_t row = 0; row < forms_case_rows(probe_case); row++) {
        forms_run_checked(launch, probe_case, instance, row, 1, phase, FORMS_STALE_ALONE, alone);
        differing += forms_differing(result + (size_t)row * FORMS_ROW_FLOATS, alone, FORMS_ROW_FLOATS, sizeof(float))
                     + forms_differing(result + FORMS_ROW_SUMS_AT + (size_t)row * FORMS_HIDDEN_SIZE,
                                       alone + FORMS_ROW_SUMS_AT, FORMS_HIDDEN_SIZE, sizeof(float));
    }
    return differing;
}

/* --break-route: launches past one bound each. The validation that every real launch passes must refuse each; none
 * is launched. A refusal is the failure the flag requires: 1 of 1. */
static void forms_break_route(const struct probe_case *probe_case)
{
    const uint32_t instance = probe_instance_index("bounds:256x6");
    for (uint32_t broken = 0; broken < FORMS_BROKEN_END; broken++) {
        struct forms_plan *plan = &forms.checked;
        forms_make_plan(plan, probe_case, instance, FORMS_CHECKED_STEP, 0, 0, forms_case_rows(probe_case));
        plan->jobs[0].expert_index += broken == FORMS_BROKEN_EXPERT ? FORMS_EXPERTS : 0;
        plan->jobs[0].pair_index[FORMS_MOST_ROWS - 1u] += broken == FORMS_BROKEN_PAIR ? FORMS_MOST_PAIRS : 0;
        plan->down_grid[0] += broken == FORMS_BROKEN_GRID ? 1u : 0u;
        probe_check_count(probe_case->name, forms_instances[instance].name, forms_broken_names[broken],
                          forms_plan_is_valid(plan) ? 0u : 1u, 1, 1);
    }
}

static void forms_check_case(void *, const struct probe_case *probe_case)
{
    static float references[FORMS_INSTANCE_COUNT][FORMS_PHASES][FORMS_RESULT_FLOATS], result[FORMS_RESULT_FLOATS];
    const uint32_t rows = forms_case_rows(probe_case);
    const uint64_t compared = (uint64_t)rows * (FORMS_ROW_FLOATS + FORMS_HIDDEN_SIZE);
    if (probe_flag_given(PROBE_FLAG_BREAK_ROUTE)) {
        forms_break_route(probe_case);
        return;
    }
    for (uint32_t index = 0; index < FORMS_INSTANCE_COUNT * FORMS_PHASES; index++) {
        const uint32_t instance = index / FORMS_PHASES, phase = index % FORMS_PHASES;
        const struct forms_variant *variant = &forms_variants[instance];
        const char *name = forms_instances[instance].name;
        const int corrupted = probe_flag_given(PROBE_FLAG_CORRUPT) && instance != PROBE_BASELINE;
        const int structural = variant->ahead != 0 || variant->persistent || forms_instances[instance].bound_blocks != 0
                               || variant->columns != FORMS_COLUMNS_PER_LANE;
        uint32_t reference = 0;                  /* the first instance of its slice count: that count's codes */
        uint64_t differing = 0;
        if (instance == PROBE_BASELINE_AGAIN || !forms_instances[instance].available) {
            continue;                            /* the baseline's kernels a second time; a block no SM holds */
        }
        while (forms_variants[reference].slices != variant->slices) {
            reference++;
        }
        forms_run_checked(variant->launch, probe_case, instance, 0, rows, phase, FORMS_STALE_SHARED, result);
        if (reference == instance) {
            memcpy(references[instance][phase], result, sizeof(result));
        } else {                                 /* what a launch did not write is stale in both */
            differing = forms_differing(result, references[reference][phase], FORMS_RESULT_FLOATS, sizeof(float));
            probe_check_count(probe_case->name, name, FORMS_PHASE_NAME("bits_against_codes", phase), differing,
                              compared, (corrupted && reference == PROBE_BASELINE)
                                            || (differing != 0 && variant->operand >= FORMS_OPERAND_CARRIED16));
            forms_instances[instance].contract_class = differing != 0 ? 'C' : forms_instances[instance].contract_class;
        }
        if (!structural || differing != 0) {
            forms_check_against_host(probe_case, instance, phase, result);
        }
        probe_check_count(probe_case->name, name, FORMS_PHASE_NAME("rows_exact", phase),
                          forms_rows_differing(variant->launch, probe_case, instance, phase, result), compared, 0);
        probe_digest_add(result, sizeof(result));
        if (probe_flag_given(PROBE_FLAG_CORRUPT) && index + 1u == FORMS_PHASES) {
            /* after the baseline's launches: the start of the gate and of the down matrix of the first checked pair */
            uint8_t *expert = (uint8_t *)forms.card.packed + (size_t)FORMS_EXPERT_BYTES
                              * forms_route_expert(probe_case, FORMS_CHECKED_STEP, 0, FORMS_EXPERTS_PER_ROW - 1u);
            PROBE_CHECK_CUDA(cudaMemset(expert, FORMS_CORRUPTED_BYTE, FORMS_CORRUPTED_BYTES));
            PROBE_CHECK_CUDA(cudaMemset(expert + FORMS_EXPERT_BYTES / FORMS_MATRIX_KINDS * FORMS_DOWN_MATRIX,
                                        FORMS_CORRUPTED_BYTE, FORMS_CORRUPTED_BYTES));
        }
    }
    if (probe_flag_has_word(FORMS_FLAG_BREAK, "exactness")) {
        forms_run_checked(forms_launch_slices_by_rows, probe_case, PROBE_BASELINE, 0, rows, FORMS_WHOLE_STEP,
                          FORMS_STALE_SHARED, result);
        probe_check_count(probe_case->name, "baseline_slices_by_rows", "rows_exact",
                          forms_rows_differing(forms_launch_slices_by_rows, probe_case, PROBE_BASELINE,
                                               FORMS_WHOLE_STEP, result),
                          compared, forms_case_stride(probe_case) < FORMS_EXPERTS_PER_ROW && rows > 1u);
    }
    probe_cuda_check_guards(probe_case->name);
}

/* ---- timing callbacks and verdicts ---- */

static int forms_prepare_step(void *, const struct probe_case *probe_case, uint32_t instance, uint64_t step,
                              uint32_t slot)
{
    PROBE_REQUIRE(slot < PROBE_STREAM_STEPS, "a slot past the slots of a stream");
    forms_make_plan(&forms.plans[slot], probe_case, instance, step, slot, 0, forms_case_rows(probe_case));
    if (!forms_plan_is_valid(&forms.plans[slot])) {
        return 0;
    }
    forms_prepare(&forms.plans[slot], step, 0, 0);
    return 1;
}

static uint32_t forms_launch_step(void *, const struct probe_case *, uint32_t instance, uint32_t slot)
{
    forms_variants[instance].launch(&forms.plans[slot]);
    return FORMS_STEP_LAUNCHES;
}

/* One acceptance figure of a case, against row `bar` of forms_bars; a negative bar only measures. */
static void forms_judge(const char *figure_name, const struct probe_case *probe_case, int bar,
                        const struct probe_result *judged, double noise_ratio)
{
    char figure[FORMS_FIGURE_BYTES];
    PROBE_REQUIRE(snprintf(figure, sizeof(figure), "%s_at_%s", figure_name, probe_case->name) > 0, "a figure's name");
    probe_verdict(figure, bar < 0 ? PROBE_NAN : forms_bars[bar].ratio,
                  bar < 0 ? PROBE_NAN : forms_bars[bar].microseconds, judged != NULL ? judged->ratio : PROBE_NAN,
                  noise_ratio, judged != NULL ? judged->same_sign : 0u, judged != NULL ? judged->rounds : 0u);
}

/* The bars are a CMP 170HX's. The PRO, the control, only measures (its bar takes two cards' logs: the batch's CONTROL
 * line), as does one slice fed ahead on both (ledger N13: its price beside the best). A form that lost bits: none. */
static void forms_verdicts(void *, const struct probe_case *probe_case, const struct probe_result *results,
                           double noise_ratio)
{
    const int on_sm_80 = PROBE_SM == PROBE_SM_80, one_row = forms_case_rows(probe_case) == 1u;
    const int eight_unshared = forms_case_span(probe_case) == FORMS_MOST_PAIRS;
    const struct probe_result *best = NULL, *lever = NULL, *carried = NULL;
    for (uint32_t instance = 0; instance < FORMS_INSTANCE_COUNT; instance++) {
        const struct forms_variant *variant = &forms_variants[instance];
        const struct probe_result *result = &results[instance];
        const int bounded = forms_instances[instance].bound_blocks != 0;
        const int plain = !variant->persistent && variant->columns == FORMS_COLUMNS_PER_LANE
                          && variant->slices == forms_variants[PROBE_BASELINE].slices;
        const int one_lever = plain && variant->operand == FORMS_OPERAND_CODES && (variant->ahead != 0) != bounded;
        const int one_carried = plain && variant->operand >= FORMS_OPERAND_CARRIED16 && variant->ahead == 0 && !bounded;
        if (instance == PROBE_BASELINE_AGAIN || result->mode == NULL || result->spill_bytes > 0 || isnan(result->ratio)
            || forms_instances[instance].contract_class != 'E') {
            continue;                            /* as the harness chooses BEST (timed, never a spill), and exact */
        }
        best = best == NULL || result->ratio < best->ratio ? result : best;
        lever = one_lever && (lever == NULL || result->ratio < lever->ratio) ? result : lever;
        carried = one_carried && (carried == NULL || result->ratio < carried->ratio) ? result : carried;
    }
    if (one_row) {
        forms_judge("carried_operand_alone", probe_case, on_sm_80 ? FORMS_BAR_CARRIED : -1, carried, noise_ratio);
        forms_judge("ahead_or_bound_alone", probe_case, on_sm_80 ? FORMS_BAR_LEVER : -1, lever, noise_ratio);
        forms_judge("one_slice_fed", probe_case, -1, &results[probe_instance_index("slices:1,ahead:1,persistent:1")],
                    noise_ratio);
    }
    if (one_row || eight_unshared) {
        forms_judge("best", probe_case, !on_sm_80 ? -1 : one_row ? FORMS_BAR_BEST : FORMS_BAR_BEST_EIGHT_ROWS, best,
                    noise_ratio);
    }
}

static struct probe_program forms_program = {
    "native_expert_forms", forms_flags, PROBE_COUNT_OF(forms_flags), forms_instances, FORMS_INSTANCE_COUNT,
    NULL,                                        /* no context: one card, one state */
    NULL,                                        /* nothing to configure: the instances are a fixed table */
    forms_open_card, forms_describe_case, forms_check_case, forms_prepare_step, forms_launch_step,
    probe_cuda_mark_start, probe_cuda_elapsed_microseconds, forms_verdicts, forms_close_card,
    NULL};                                       /* the harness's rounds are the protocol */

int main(int argument_count, char **arguments) { return probe_main(argument_count, arguments, &forms_program); }
