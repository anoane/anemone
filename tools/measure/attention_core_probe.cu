/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * attention_core_probe: package 11b of round 1 of the speed campaign (research/speed/ROUND_1_attention.md,
 * "attention_core"; ledger A3 heads a block, A4 stage ring, A5 compact rows; the owner's 8-bit state of
 * DECISIONS_2026-10-12.md 1): one layer's attention core, one row's latency against the blocks that serve it, the
 * rate at 4,096 rows against the card's tensor rate, what one gather for all heads, the ring's depth and a compact
 * pool are worth, that none of them changes a bit, and the two 8-bit kinds beside FP16. Kernels:
 * attention_core_kernels.cuh (the engine's), attention_core_probe_kernels.cuh (the round's, the planes, the bounds).
 *
 * A step is one launch for a case's rows. Its lists are made on the card while it is prepared (not timed): window
 * rows r .. r + 127 from a pointer that walks the plane, pool entries one a band by a hash of the step; a plane is
 * 1,048,576 rows (1 GiB fp16, 528 and 544 MiB in 8 bits), four L2 sizes of a PRO or more. pool:compact first copies
 * the pool entries into a compact region (a ring of four L2 sizes and a region, or one step's: a stream's slots may
 * then share it), then empties the L2 of it: its timed launch reads the region cold, as a later layer would.
 * Instances (CORE_INSTANCES), interleaved by probe_harness.h: baseline (16 heads a block, 2 stages); heads:8;
 * stages:1,pad, stages:2,pad (the baseline's kernel), stages:3, all at the 3-stage shared memory; gather:shared (64
 * heads, 16 warps, 1 stage); rescale:always (step 0: another contract, held to the reference); pool:compact;
 * gather_only; tensor_only; state:trained, state:fresh and gather_only of each; break:group_by_rows with --break
 * exactness only. Cases: --rows by --blocks-per-sm (shared memory padded so that at most so many blocks fit an SM).
 *
 * Checked per case before anything is timed, on the host's lists with the corners (row 0 whole, its head 5's maximum in
 * the last group and the other p of that head far under 2^-24 of it; row 1 no valid key; row 2 five window rows; row 3
 * 300 pool entries; in the 8-bit planes row 0's newest window row hand-made of subnormal E4M3 codes only, by 2^6, so
 * that their placement as fp16 subnormals is seen) and the plane rows beside the listed ones poisoned (NaN in fp16, the
 * largest codes by the largest scales in 8 bits): host_reference (rows 0 .. 3, the middle and the last against
 * attention_core_reference.h, 2e-3 of the rms, the worst row); bits_against_baseline (the instances that only move
 * work: 0 words may differ; a difference is a finding, verdict=reported and class C) and bits_blocks_per_sm (per kind,
 * the padded launch against an unpadded one); rows_exact (every row of 8, 37 of more, alone); gather_checksum and
 * tensor_sums (bit for bit); guard_bytes; DIGEST. Every launch is validated on the host first (core_step_is_valid); the
 * debug build's kernels also check each index and their geometry into a status word read after each launch. Shown
 * failing: --corrupt (a megabyte of each plane at the checked window rows, after the host's copy), --break exactness,
 * --break bits (a word of the host's baseline), --break index (row 0's index 130 moved on the card only), --break-route
 * (four launches, all refused), --break-guard. Its own lines: RATE card= case= variant= gbps= tflops= (the instance's
 * bytes, each distinct stored row once, and FLOP), SOFTMAX card= case= kind= rescales_per_row= flushed_per_row= (the
 * reference's sampled rows). RESULT: format= the baseline's, mb= and gbps= na (RATE has each kind's bytes).
 *
 *   attention_core_probe [--rows 1,8,256] [--blocks-per-sm 0] [--break exactness,bits,index] [harness flags]
 *
 * Measurement tool: not part of the engine.
 */
#include "probe_harness.h"
#define ATTENTION_CORE_CHECKS PROBE_DIAGNOSTICS
#include "attention_core_probe_kernels.cuh"
#include "attention_core_reference.h"

enum {
    CORE_PLANE_ROWS = 1048576, CORE_MOST_ROWS = 4096,
    CORE_ROW_HALVES = ATTENTION_HEADS * ATTENTION_DIMS,
    CORE_ROTATION_FLOATS = ATTENTION_ROPE_PAIRS * ATTENTION_ROPE_VALUES,
    CORE_WINDOW_SPAN_EXTRA = ATTENTION_WINDOW_KEYS - 1,     /* window rows a launch reads beyond one a row */
    CORE_PLANE_FP16 = 0, CORE_PLANE_POW2 = 1, CORE_PLANE_E2M1 = 2, CORE_PLANES = 3,
    CORE_ROLE_CORE = 0, CORE_ROLE_GATHER = 1, CORE_ROLE_TENSOR = 2,
    CORE_ELEMENT_THREADS = 256, CORE_MOST_BLOCKS_PER_SM = 32, CORE_COLD_L2_SIZES = 4,
    CORE_SAMPLED_ROWS = 6, CORE_EXACT_ROWS = 37, CORE_ALL_ROWS_EXACT = 8,
    CORE_CORNER_ROW_EMPTY = 1, CORE_CORNER_ROW_FEW = 2, CORE_CORNER_ROW_SHORT = 3, CORE_CORNER_HEAD = 5,
    CORE_FEW_KEYS = 5, CORE_SHORT_POOL = 300, CORE_BROKEN_KEY = ATTENTION_WINDOW_KEYS + 2,
    CORE_CHECK_WINDOW_SHARE = 2,                 /* the check's window rows begin in the middle of the plane */
    CORE_CHECK_SEED = 0x11B, CORE_SHORT_RUN_PLANE_ROWS = 131072,     /* after --steps: no timing, smaller planes */
    CORE_CORRUPTED_BYTES = 1024 * 1024, CORE_CORRUPTED_BYTE = 0x5A, CORE_STALE_BYTE = 0xEE,
    CORE_VALUE_STEPS = 256, CORE_VALUE_CENTRE = 128, CORE_QUERY_SHIFT = 5, CORE_SINK_SHIFT = 6,
    CORE_REFERENCE_FP16 = 0, CORE_REFERENCE_STEP_0, CORE_REFERENCE_TRAINED, CORE_REFERENCE_FRESH, CORE_REFERENCES,
    CORE_BROKEN_LISTS = 0, CORE_BROKEN_WINDOW, CORE_BROKEN_BLOCK, CORE_BROKEN_SHARED, CORE_BROKEN_ROUTES,
    CORE_FIGURE_BYTES = 256, CORE_PADDED_STAGES = ATTENTION_DEEP_STAGES, CORE_MOST_POISONED = ATTENTION_POOL_KEYS + 2,
    CORE_GROUPS_PER_ROW = ATTENTION_HEADS / (ATTENTION_TILE_HEADS * ATTENTION_GATHER_HEAD_TILES),   /* the bounds' */
    CORE_TENSOR_FLOATS_PER_ROW = CORE_GROUPS_PER_ROW * attention_threads(ATTENTION_GATHER_HEAD_TILES),
    CORE_SUBNORMAL_KEY = ATTENTION_WINDOW_KEYS - 1  /* the hand-made window row: row 0's newest, row 2's too */
};
#define CORE_FLAG_ROWS "--rows"
#define CORE_FLAG_BLOCKS "--blocks-per-sm"
#define CORE_FLAG_BREAK "--break"
#define CORE_TOLERANCE 2.0e-3                    /* CONVENTIONS 5.1: a bfloat16 vector against its rms */
#define CORE_CORNER_SCORE 40.0                   /* head 5 of row 0 on its last key; every other p under e^-25 */
#define CORE_MEGA 1.0e6                          /* FLOP over microseconds: MFLOP/s; over 10^6 more: TFLOP/s */

static struct probe_flag core_flags[] = {
    PROBE_COMMON_FLAGS,
    PROBE_CASE_NUMBERS(CORE_FLAG_ROWS, 1, CORE_MOST_ROWS, "1,8,256", "rows of a step"),
    PROBE_CASE_NUMBERS(CORE_FLAG_BLOCKS, 0, CORE_MOST_BLOCKS_PER_SM, "0",
                       "blocks an SM may hold by its shared memory: the launch is padded (0: not padded)"),
    PROBE_WORD_LIST(CORE_FLAG_BREAK, "exactness,bits,index", "", "show that check failing"),
};

/* ---- the kernels and the instances ---- */

struct core_step {                               /* one launch: built, validated, then made */
    uint32_t instance, kernel, rows, head_groups, threads, shared_bytes;
    uint64_t window_rows, pool_rows;             /* rows of the planes from the launch's pointers on */
    uint64_t window_named, pool_named;           /* the rows of each its lists can name: at most those */
    struct attention_core_arguments arguments;
    uint32_t *checksums;
    float *tensor_sums;
};
typedef void core_launcher(const struct core_step *step);
#if PROBE_DIAGNOSTICS
#define CORE_CHECKS_ARGUMENT , core_checks_of(step)
static struct attention_core_checks core_checks_of(const struct core_step *step);
#else
#define CORE_CHECKS_ARGUMENT
#endif
#define CORE_LAUNCH_GRID dim3(step->rows, step->head_groups), step->threads, step->shared_bytes
#define CORE_LAUNCH_CORE(function) function<<<CORE_LAUNCH_GRID>>>(step->arguments CORE_CHECKS_ARGUMENT)
#define CORE_LAUNCH_GATHER(function) \
    function<<<CORE_LAUNCH_GRID>>>(step->arguments, step->checksums CORE_CHECKS_ARGUMENT)
#define CORE_LAUNCH_TENSOR(function) function<<<CORE_LAUNCH_GRID>>>(step->tensor_sums, 0u)
#define CORE_STAMP_CORE(id, bound, kind, head_tiles, stages, step_rule) \
    static void core_launch_##id(const struct core_step *step) { CORE_LAUNCH_CORE(attention_core_##id); }
#define CORE_STAMP_OTHER(id, function, role, kind) \
    static void core_launch_##id(const struct core_step *step) { CORE_LAUNCH_##role(function); }
/* the bounds' kernels: id, function, role, kind */
#define CORE_OTHER_KERNELS(STAMP)                                                  \
    STAMP(gather_fp16, attention_gather_only_fp16, GATHER, ATTENTION_FP16)        \
    STAMP(gather_trained, attention_gather_only_trained, GATHER, ATTENTION_TRAINED) \
    STAMP(gather_fresh, attention_gather_only_fresh, GATHER, ATTENTION_FRESH)     \
    STAMP(tensor, attention_tensor_only_kernel, TENSOR, ATTENTION_FP16)
ATTENTION_CORE_KERNELS(CORE_STAMP_CORE)
CORE_OTHER_KERNELS(CORE_STAMP_OTHER)
#define CORE_KERNEL_ID(id, ...) CORE_KERNEL_##id,
enum { ATTENTION_CORE_KERNELS(CORE_KERNEL_ID) CORE_OTHER_KERNELS(CORE_KERNEL_ID) CORE_KERNEL_COUNT };
struct core_kernel {
    core_launcher *launch;
    const void *function;
    const char *name;
    int role, kind, head_tiles, stages, bounded;
};
#define CORE_CORE_ROW(id, bound, kind, head_tiles, stages, step_rule)                                  \
    {core_launch_##id, (const void *)attention_core_##id, "attention_core_" #id, CORE_ROLE_CORE, kind, \
     head_tiles, stages, ATTENTION_THREADS_OF_##bound},
#define ATTENTION_THREADS_OF_ATTENTION_BOUND_NONE 0
#define ATTENTION_THREADS_OF_ATTENTION_BOUND_GATHER_SHARED ATTENTION_GATHER_SHARED_THREADS
#define CORE_OTHER_ROW(id, function, role, kind) \
    {core_launch_##id, (const void *)function, #function, CORE_ROLE_##role, kind, ATTENTION_GATHER_HEAD_TILES, \
     ATTENTION_GATHER_STAGES, 0},
static const struct core_kernel core_kernels[CORE_KERNEL_COUNT] = {
    ATTENTION_CORE_KERNELS(CORE_CORE_ROW) CORE_OTHER_KERNELS(CORE_OTHER_ROW)};

/* id, variant, kernel, the class it claims, then 0 or 1 each: its bits must be the baseline's; its shared memory is
 * the 3-stage size; it reads a compact pool. The last joins a run with --break exactness alone. */
#define CORE_INSTANCES(ROW)                                                            \
    ROW(baseline, "baseline", baseline, 'R', 0, 0, 0)                                  \
    ROW(baseline_again, "baseline_again", baseline, 'R', 1, 0, 0)                      \
    ROW(heads_8, "heads:8", heads_8, 'R', 1, 0, 0)                                     \
    ROW(stages_1_pad, "stages:1,pad", stages_1, 'R', 1, 1, 0)                          \
    ROW(stages_2_pad, "stages:2,pad", baseline, 'R', 1, 1, 0)                          \
    ROW(stages_3, "stages:3", stages_3, 'R', 1, 0, 0)                                  \
    ROW(gather_shared, "gather:shared", gather_shared, 'R', 1, 0, 0)                   \
    ROW(rescale_always, "rescale:always", rescale_always, 'C', 0, 0, 0)                \
    ROW(pool_compact, "pool:compact", baseline, 'R', 1, 0, 1)                          \
    ROW(gather_only, "gather_only", gather_fp16, 'C', 0, 0, 0)                         \
    ROW(tensor_only, "tensor_only", tensor, 'C', 0, 0, 0)                              \
    ROW(trained, "state:trained", trained, 'R', 0, 0, 0)                               \
    ROW(fresh, "state:fresh", fresh, 'R', 0, 0, 0)                                     \
    ROW(gather_trained, "gather_only,state:trained", gather_trained, 'C', 0, 0, 0)     \
    ROW(gather_fresh, "gather_only,state:fresh", gather_fresh, 'C', 0, 0, 0)           \
    ROW(group_by_rows, "break:group_by_rows", group_by_rows, 'C', 0, 0, 0)
#define CORE_INSTANCE_ID(id, ...) CORE_INSTANCE_##id,
enum { CORE_INSTANCES(CORE_INSTANCE_ID) CORE_INSTANCE_COUNT };
#define CORE_INSTANCE_ROW(id, variant, kernel, claimed, bits, padded, compact) \
    {variant, NULL, claimed, 1, 0, 0, 0, 0, 0},
static struct probe_instance core_instances[CORE_INSTANCE_COUNT] = {CORE_INSTANCES(CORE_INSTANCE_ROW)};
struct core_variant { uint32_t kernel; int bits_required, padded, compact; };
#define CORE_VARIANT_ROW(id, variant, kernel, claimed, bits, padded, compact) \
    {CORE_KERNEL_##kernel, bits, padded, compact},
static const struct core_variant core_variants[CORE_INSTANCE_COUNT] = {CORE_INSTANCES(CORE_VARIANT_ROW)};
static uint32_t core_instances_in_run = CORE_INSTANCE_COUNT - 1u;        /* all but the one --break exactness adds */

static struct {                                  /* the card's arrays and what a run keeps of them */
    uint64_t sm_shared_bytes, reserved_shared_bytes, most_shared_bytes, compact_ring_bytes;
    uint32_t most_rows;                          /* the largest --rows value: what the arrays are sized for */
    uint64_t plane_rows, band, evicted_bytes;    /* rows of a plane (fewer in a short run), of a pool band; scratch */
    unsigned char *planes[CORE_PLANES], *compact, *evicted;
    uint16_t *queries, *outputs;
    int32_t *lists, *compact_list;               /* lists [slot][most rows][640]; compact_list [most rows][640] */
    float *sinks, *rotation, *tensor_sums;
    uint32_t *checksums, *row_list;
    unsigned long long *status_word;
    struct core_step steps[PROBE_STREAM_STEPS];
} core;
static struct {                                  /* the host's copies */
    unsigned char *planes[CORE_PLANES];          /* the planes as filled, before any check changed them */
    uint16_t *queries, *outputs, *baseline, *alone, *unpadded;
    int32_t *list, *compact_list;
    uint32_t *checksums, *poisoned;
    float *rotation, *tensor_sums, sinks[ATTENTION_HEADS];
    uint8_t *referenced;                         /* [plane row]: a list of the check names it */
    double *keys, *references;                   /* [640][512]; [CORE_REFERENCES][sampled row][64][512] */
    uint32_t poisoned_count, sampled[CORE_SAMPLED_ROWS], sampled_count;
    uint64_t pool_distinct;                      /* the distinct pool rows a timed step of the case names */
} core_host;

static const uint32_t core_row_bytes[CORE_PLANES] = {ATTENTION_FP16_ROW_BYTES, ATTENTION_POW2_ROW_BYTES,
                                                     ATTENTION_E2M1_ROW_BYTES};
static const int core_reference_formats[CORE_PLANES] = {REFERENCE_FORMAT_FP16, REFERENCE_FORMAT_POW2,
                                                        REFERENCE_FORMAT_E2M1};

static uint32_t core_window_plane(int kind) { return kind == ATTENTION_FP16 ? CORE_PLANE_FP16 : CORE_PLANE_POW2; }
static uint32_t core_pool_plane(int kind)
{
    return kind == ATTENTION_FP16 ? CORE_PLANE_FP16 : kind == ATTENTION_TRAINED ? CORE_PLANE_E2M1 : CORE_PLANE_POW2;
}
static uint32_t core_rows_of(const struct probe_case *probe_case)
{
    return (uint32_t)probe_case_value(probe_case, CORE_FLAG_ROWS);
}
static uint64_t core_blocks(uint64_t items) { return (items + CORE_ELEMENT_THREADS - 1u) / CORE_ELEMENT_THREADS; }
static int32_t *core_slot_list(uint32_t slot) { return core.lists + (size_t)slot * core.most_rows * ATTENTION_KEYS; }
static uint16_t core_half_bits(float value) { return __half_as_ushort(__float2half_rn(value)); }

/* byte_count bytes from `pointer` on must lie inside the card array that begins at `array`. */
static void core_range(int *valid, const char *kernel, const char *what, const void *array, const void *pointer,
                       uint64_t byte_count)
{
    probe_launch_range(valid, kernel, what, (uint64_t)((uintptr_t)pointer - (uintptr_t)array), byte_count, 1,
                       probe_cuda_array_bytes(array));
}

/* The window row a step's lists count from: the plane walked in spans of the case's rows, never wrapped. */
static uint64_t core_window_base(uint64_t ring_step, uint32_t rows)
{
    const uint64_t span = (uint64_t)rows + CORE_WINDOW_SPAN_EXTRA;
    return ring_step == 0 ? core.plane_rows / CORE_CHECK_WINDOW_SHARE
                          : ring_step * span % (core.plane_rows - span + 1u);
}

/* The compact region of a step: the ring holds as many regions of the case's rows as fit, at least one. */
static unsigned char *core_compact_region(uint64_t ring_step, uint32_t rows)
{
    const uint64_t region = (uint64_t)rows * ATTENTION_POOL_KEYS * ATTENTION_FP16_ROW_BYTES;
    return core.compact + ring_step % (core.compact_ring_bytes / region) * region;
}

/* Of two card arrays the one `pointer` lies in; the first where it lies in neither (its range is then refused). */
static const void *core_array_of(const void *pointer, const void *first, const void *second)
{
    const uintptr_t at = (uintptr_t)pointer, second_at = (uintptr_t)second;
    return second != NULL && at >= second_at && at - second_at < probe_cuda_array_bytes(second) ? second : first;
}

/* Check 10: 1 if the launch reads and writes inside its arrays and the card's and its instance's limits; else it has
 * printed why not. The lists it reads were validated where they were made: the host's index by index, the card's by
 * the arguments of the kernel that made them (core_list_launch); a step says how many rows of each plane its lists
 * can name (window_named, pool_named) and the planes must hold them from the launch's pointers on. */
static int core_step_is_valid(const struct core_step *step, uint32_t instance)
{
    const struct core_kernel *kernel = &core_kernels[step->kernel];
    const struct attention_core_arguments *arguments = &step->arguments;
    const uint64_t rows = step->rows, blocks = rows * step->head_groups;
    const uint32_t window_plane = core_window_plane(kernel->kind), pool_plane = core_pool_plane(kernel->kind);
    const char *name = kernel->name;
    int valid = 1;
    probe_launch_value(&valid, name, "rows of a launch", rows, 1, core.most_rows);
    probe_launch_value(&valid, name, "heads of the blocks of a row",
                       (uint64_t)step->head_groups * kernel->head_tiles * ATTENTION_TILE_HEADS, ATTENTION_HEADS,
                       ATTENTION_HEADS);
    probe_launch_value(&valid, name, "shared bytes: at least the kernel's layout", step->shared_bytes,
                       attention_layout_of(kernel->kind, kernel->head_tiles, kernel->stages).total,
                       core.most_shared_bytes);
    probe_launch_geometry(&valid, name, &core_instances[instance], rows, step->head_groups, 1, step->threads,
                          step->shared_bytes);
    if (kernel->role == CORE_ROLE_TENSOR) {
        core_range(&valid, name, "tensor sums", core.tensor_sums, step->tensor_sums,
                   blocks * step->threads * sizeof(float));
        return valid;
    }
    core_range(&valid, name, "indices", core_array_of(arguments->indices, core.lists, core.compact_list),
               arguments->indices, rows * ATTENTION_KEYS * sizeof(int32_t));
    core_range(&valid, name, "window rows", core.planes[window_plane], arguments->window_rows,
               step->window_rows * core_row_bytes[window_plane]);
    probe_launch_value(&valid, name, "window rows the lists name", step->window_named, 1, step->window_rows);
    core_range(&valid, name, "pool rows", core_array_of(arguments->pool_rows, core.planes[pool_plane], core.compact),
               arguments->pool_rows, step->pool_rows * core_row_bytes[pool_plane]);
    probe_launch_value(&valid, name, "pool rows the lists name", step->pool_named, 1, step->pool_rows);
    if (kernel->role == CORE_ROLE_GATHER) {
        core_range(&valid, name, "checksums", core.checksums, step->checksums, blocks * sizeof(uint32_t));
        return valid;
    }
    core_range(&valid, name, "queries", core.queries, arguments->queries, rows * CORE_ROW_HALVES * sizeof(uint16_t));
    core_range(&valid, name, "sinks", core.sinks, arguments->sinks, ATTENTION_HEADS * sizeof(float));
    core_range(&valid, name, "rotation", core.rotation, arguments->rotation,
               rows * CORE_ROTATION_FLOATS * sizeof(float));
    core_range(&valid, name, "outputs", core.outputs, arguments->output, rows * CORE_ROW_HALVES * sizeof(uint16_t));
    return valid;
}

/* ---- a step: built, launched; the lists and compact copies of a timed step ---- */

/* The step of an instance on rows first_row .. first_row + rows - 1 of a list on the card (the compact list where
 * the instance reads a compact pool), its window rows at ring step ring_step (0: the check's); padded: the case's
 * --blocks-per-sm applies. 0: refused, as the REFUSED lines say. */
static int core_build_step(const struct probe_case *probe_case, uint32_t instance, uint64_t ring_step,
                           const int32_t *list, uint32_t first_row, uint32_t rows, int padded, struct core_step *step)
{
    const struct core_variant *variant = &core_variants[instance];
    const struct core_kernel *kernel = &core_kernels[variant->kernel];
    const uint32_t case_rows = core_rows_of(probe_case);
    const uint64_t blocks_per_sm = probe_case_value(probe_case, CORE_FLAG_BLOCKS);
    const uint32_t window_plane = core_window_plane(kernel->kind), pool_plane = core_pool_plane(kernel->kind);
    const uint64_t window_base = core_window_base(ring_step, case_rows);
    uint64_t shared = attention_layout_of(kernel->kind, kernel->head_tiles, kernel->stages).total;
    const uint64_t padded_size = attention_layout_of(ATTENTION_FP16, ATTENTION_WARP_TILES, CORE_PADDED_STAGES).total;
    memset(step, 0, sizeof(*step));
    step->instance = instance;
    step->kernel = variant->kernel;
    step->rows = rows;
    step->head_groups = ATTENTION_HEADS / ((uint32_t)kernel->head_tiles * ATTENTION_TILE_HEADS);
    step->threads = attention_threads(kernel->head_tiles);
    shared = variant->padded && padded_size > shared ? padded_size : shared;
    if (padded && blocks_per_sm != 0 && core.sm_shared_bytes / blocks_per_sm > core.reserved_shared_bytes + shared) {
        shared = core.sm_shared_bytes / blocks_per_sm - core.reserved_shared_bytes;      /* at most that many fit */
    }
    step->shared_bytes = (uint32_t)(shared < UINT32_MAX ? shared : UINT32_MAX);
    step->arguments.queries = core.queries + (size_t)first_row * CORE_ROW_HALVES;
    step->arguments.window_rows = core.planes[window_plane] + window_base * core_row_bytes[window_plane];
    step->arguments.pool_rows = variant->compact ? core_compact_region(ring_step, case_rows) : core.planes[pool_plane];
    step->arguments.indices = list + (size_t)first_row * ATTENTION_KEYS;
    step->arguments.sinks = core.sinks;
    step->arguments.rotation = core.rotation + (size_t)first_row * CORE_ROTATION_FLOATS;
    step->arguments.output = core.outputs + (size_t)first_row * CORE_ROW_HALVES;
    step->window_rows = core.plane_rows - window_base;
    step->window_named = (uint64_t)first_row + rows + CORE_WINDOW_SPAN_EXTRA;
    step->pool_rows = variant->compact ? (uint64_t)case_rows * ATTENTION_POOL_KEYS : core.plane_rows;
    step->pool_named = variant->compact ? ((uint64_t)first_row + rows) * ATTENTION_POOL_KEYS : core.plane_rows;
    step->checksums = core.checksums + (size_t)first_row * CORE_GROUPS_PER_ROW;      /* the bounds' arrays */
    step->tensor_sums = core.tensor_sums + (size_t)first_row * CORE_TENSOR_FLOATS_PER_ROW;
    return core_step_is_valid(step, instance);
}

#if PROBE_DIAGNOSTICS
static struct attention_core_checks core_checks_of(const struct core_step *step)
{
    struct attention_core_checks checks = {core.status_word, step->window_rows, step->pool_rows};
    return checks;
}
#endif

/* Queues a validated step; returns its launches. The debug build validates it again, waits, and reads the status
 * word the kernels' own checks store into. */
static uint32_t core_launch(const struct core_step *step)
{
#if PROBE_DIAGNOSTICS
    unsigned long long status = 0;
    PROBE_REQUIRE(core_step_is_valid(step, step->instance), "a step that its preparation did not validate");
#endif
    core_kernels[step->kernel].launch(step);
#if PROBE_DIAGNOSTICS
    PROBE_AFTER_LAUNCH();
    PROBE_CHECK_CUDA(cudaMemcpy(&status, core.status_word, sizeof(status), cudaMemcpyDeviceToHost));
    PROBE_REQUIRE(status == 0, "a kernel's own check failed: an index past its plane, or a block not its own");
#endif
    return 1;
}

/* A timed step's lists into a slot, made on the card: held to the slot by the launch's range and to the planes by
 * the kernel's arguments (window rows r + j of at most rows + 127; pool rows of 512 bands of the plane). */
static int core_list_launch(uint32_t slot, uint32_t rows, uint64_t seed)
{
    const char *name = "attention_list_kernel";
    struct attention_list_arguments arguments = {core_slot_list(slot), rows, (uint32_t)core.band, (uint32_t)seed, 0};
    int valid = 1;
    probe_launch_value(&valid, name, "slot", slot, 0, PROBE_STREAM_STEPS - 1u);
    probe_launch_value(&valid, name, "rows", rows, 1, core.most_rows);
    probe_launch_value(&valid, name, "rows of the pool's bands", core.band * ATTENTION_POOL_KEYS, ATTENTION_POOL_KEYS,
                       core.plane_rows);
    if (valid) {
        probe_launch_geometry(&valid, name, NULL, core_blocks((uint64_t)rows * ATTENTION_KEYS), 1, 1,
                              CORE_ELEMENT_THREADS, 0);
        core_range(&valid, name, "a slot's lists", core.lists, arguments.indices,
                   (uint64_t)rows * ATTENTION_KEYS * sizeof(int32_t));
    }
    if (valid) {
        attention_list_kernel<<<(unsigned int)core_blocks((uint64_t)rows * ATTENTION_KEYS), CORE_ELEMENT_THREADS>>>(
            arguments);
        PROBE_CHECK_CUDA(cudaGetLastError());
    }
    return valid;
}

/* The pool entries a list names, copied into a compact region (fp16 rows: pool:compact is of the FP16 kind); then
 * the scratch of four L2 sizes read and written (CONVENTIONS 3.5): no line of the copy is left in the L2. */
static int core_compact_launch(const int32_t *list, unsigned char *region, uint32_t rows)
{
    const char *name = "attention_compact_kernel", *evicting = "attention_evict_kernel";
    const uint64_t chunks = (uint64_t)rows * ATTENTION_POOL_KEYS * (ATTENTION_FP16_ROW_BYTES / ATTENTION_CHUNK_BYTES);
    const uint64_t sweep = core.evicted_bytes / sizeof(uint4);          /* the scratch's 16-byte chunks */
    struct attention_compact_arguments arguments = {list, core.planes[CORE_PLANE_FP16], region, rows,
                                                    ATTENTION_FP16_ROW_BYTES};
    int valid = 1;
    probe_launch_geometry(&valid, name, NULL, core_blocks(chunks), 1, 1, CORE_ELEMENT_THREADS, 0);
    core_range(&valid, name, "the lists", core.lists, list, (uint64_t)rows * ATTENTION_KEYS * sizeof(int32_t));
    core_range(&valid, name, "a compact region", core.compact, region,
               (uint64_t)rows * ATTENTION_POOL_KEYS * ATTENTION_FP16_ROW_BYTES);
    probe_launch_geometry(&valid, evicting, NULL, core_blocks(sweep), 1, 1, CORE_ELEMENT_THREADS, 0);
    core_range(&valid, evicting, "the scratch", core.evicted, core.evicted, sweep * sizeof(uint4));
    if (valid) {
        attention_compact_kernel<<<(unsigned int)core_blocks(chunks), CORE_ELEMENT_THREADS>>>(arguments);
        PROBE_CHECK_CUDA(cudaGetLastError());
        attention_evict_kernel<<<(unsigned int)core_blocks(sweep), CORE_ELEMENT_THREADS>>>(core.evicted, sweep, rows);
        PROBE_CHECK_CUDA(cudaGetLastError());
    }
    return valid;
}

/* Not timed: the step's lists (a hash of its number), its compact copy and the L2 emptied of it, its launch
 * validated; the card idle after. */
static int core_prepare_step(void *, const struct probe_case *probe_case, uint32_t instance, uint64_t step,
                             uint32_t slot)
{
    const struct core_variant *variant = &core_variants[instance];
    const uint32_t rows = core_rows_of(probe_case);
    const uint64_t ring_step = step + 1u;        /* ring step 0 is the check's */
    PROBE_REQUIRE(slot < PROBE_STREAM_STEPS, "a slot past the slots of a stream");
    int valid = core_kernels[variant->kernel].role == CORE_ROLE_TENSOR || core_list_launch(slot, rows, ring_step);
    valid = valid && (!variant->compact ||
                      core_compact_launch(core_slot_list(slot), core_compact_region(ring_step, rows), rows));
    valid = valid && core_build_step(probe_case, instance, ring_step,
                                     variant->compact ? core.compact_list : core_slot_list(slot), 0, rows, 1,
                                     &core.steps[slot]);
    PROBE_CHECK_CUDA(cudaDeviceSynchronize());
    return valid;
}

static uint32_t core_launch_step(void *, const struct probe_case *, uint32_t, uint32_t slot)
{
    return core_launch(&core.steps[slot]);
}

/* Bytes a step of an instance moves (each distinct stored row, query and output once: of the pool the distinct rows
 * the case's timed lists name, or a compact region's copies) and its FLOP (scores and P x V). */
static void core_step_work(uint32_t instance, uint32_t rows, double *bytes, double *flop)
{
    const struct core_kernel *kernel = &core_kernels[core_variants[instance].kernel];
    const double row_bytes = (double)CORE_ROW_HALVES * sizeof(uint16_t);
    const double pool_rows = core_variants[instance].compact ? (double)rows * ATTENTION_POOL_KEYS
                                                             : (double)core_host.pool_distinct;
    *bytes = kernel->role == CORE_ROLE_TENSOR ? 0.0
             : (double)(rows + CORE_WINDOW_SPAN_EXTRA) * core_row_bytes[core_window_plane(kernel->kind)] +
                   pool_rows * core_row_bytes[core_pool_plane(kernel->kind)] +
                   (kernel->role == CORE_ROLE_CORE ? (double)rows * (row_bytes + row_bytes) : 0.0);
    *flop = kernel->role == CORE_ROLE_GATHER ? 0.0
            : (double)rows * ATTENTION_HEADS * ATTENTION_KEYS * ATTENTION_DIMS * PROBE_HALF * PROBE_HALF;
}

/* ---- the card, and a case's inputs ---- */

static void *core_host_allocate(uint64_t bytes)
{
    void *pointer = calloc(1, (size_t)bytes);
    return PROBE_REQUIRE(pointer != NULL, "out of host memory"), pointer;
}

/* The planes' rows `rows` (count of them; NULL: rows 0 .. count - 1) filled, or poisoned (attention_poison_kernel). */
static void core_fill_rows(const uint32_t *rows, uint64_t count, int poison)
{
    const char *name = poison ? "attention_poison_kernel" : "attention_fill_kernel";
    const uint64_t items = count * (poison ? (uint64_t)ATTENTION_FP16_ROW_BYTES : (uint64_t)ATTENTION_POW2_BLOCKS);
    struct attention_fill_arguments arguments = {core.planes[CORE_PLANE_FP16], core.planes[CORE_PLANE_POW2],
                                                 core.planes[CORE_PLANE_E2M1], rows, 0, count};
    int valid = 1;
    probe_launch_value(&valid, name, "rows", count, 1, rows == NULL ? core.plane_rows : (uint64_t)CORE_MOST_POISONED);
    probe_launch_geometry(&valid, name, NULL, core_blocks(items), 1, 1, CORE_ELEMENT_THREADS, 0);
    PROBE_REQUIRE(valid && (rows != NULL || !poison), "a fill of the planes was refused");
    if (poison) {
        attention_poison_kernel<<<(unsigned int)core_blocks(items), CORE_ELEMENT_THREADS>>>(arguments);
    } else {
        attention_fill_kernel<<<(unsigned int)core_blocks(items), CORE_ELEMENT_THREADS>>>(arguments);
    }
    PROBE_CHECK_CUDA(cudaDeviceSynchronize());
}

static int core_open_card(void *, struct probe_card_facts *facts)
{
    if (!probe_cuda_open(facts)) {
        return 0;
    }
    const cudaDeviceProp *properties = &probe_cuda.properties;
    const int checks_only = probe_flag_given(PROBE_FLAG_STEPS) && probe_flag_text(PROBE_FLAG_PROFILE_CASE)[0] == '\0';
    const uint64_t most = core.most_rows, region = most * ATTENTION_POOL_KEYS * ATTENTION_FP16_ROW_BYTES;
    const uint64_t cold = CORE_COLD_L2_SIZES * facts->l2_bytes;                       /* CONVENTIONS 3.5 */
    core.sm_shared_bytes = properties->sharedMemPerMultiprocessor;
    core.reserved_shared_bytes = properties->reservedSharedMemPerBlock;
    core.most_shared_bytes = properties->sharedMemPerBlockOptin;
    facts->shared_bytes_most = core.most_shared_bytes;   /* the kernels ask for more than the default 48 KiB */
    core.plane_rows = checks_only ? CORE_SHORT_RUN_PLANE_ROWS : CORE_PLANE_ROWS;
    core.band = core.plane_rows / ATTENTION_POOL_KEYS;
    for (uint32_t plane = 0; plane < CORE_PLANES; plane++) {
        core.planes[plane] = (unsigned char *)probe_cuda_allocate(core.plane_rows * core_row_bytes[plane]);
        core_host.planes[plane] = (unsigned char *)core_host_allocate(core.plane_rows * core_row_bytes[plane]);
    }
    core_fill_rows(NULL, core.plane_rows, 0);
    for (uint32_t plane = 0; plane < CORE_PLANES; plane++) {
        PROBE_CHECK_CUDA(cudaMemcpy(core_host.planes[plane], core.planes[plane],
                                    core.plane_rows * core_row_bytes[plane], cudaMemcpyDeviceToHost));
    }
    const uint64_t subnormal_at = (core_window_base(0, 1) + CORE_SUBNORMAL_KEY) * ATTENTION_POW2_ROW_BYTES;
    attention_subnormal_record(core_host.planes[CORE_PLANE_POW2] + subnormal_at);      /* the host's copy, the card's */
    PROBE_CHECK_CUDA(cudaMemcpy(core.planes[CORE_PLANE_POW2] + subnormal_at, core_host.planes[CORE_PLANE_POW2] +
                                subnormal_at, ATTENTION_POW2_ROW_BYTES, cudaMemcpyHostToDevice));
    facts->ring_bytes = core.plane_rows * ATTENTION_POW2_ROW_BYTES;                  /* the least of the planes */
    /* a stream reads a region again after four L2 sizes of the others' (or a region of four L2 sizes at least) */
    core.compact_ring_bytes = checks_only || cold <= region ? region : ((cold + region - 1u) / region + 1u) * region;
    core.compact = (unsigned char *)probe_cuda_allocate(core.compact_ring_bytes);
    core.evicted_bytes = checks_only ? facts->l2_bytes : cold;                       /* untimed after --steps */
    core.evicted = (unsigned char *)probe_cuda_allocate(core.evicted_bytes);
    PROBE_CHECK_CUDA(cudaMemset(core.evicted, 0, core.evicted_bytes));            /* written before it is read */
    core.queries = (uint16_t *)probe_cuda_allocate(most * CORE_ROW_HALVES * sizeof(uint16_t));
    core.outputs = (uint16_t *)probe_cuda_allocate(most * CORE_ROW_HALVES * sizeof(uint16_t));
    core.lists = (int32_t *)probe_cuda_allocate(PROBE_STREAM_STEPS * most * ATTENTION_KEYS * sizeof(int32_t));
    core.compact_list = (int32_t *)probe_cuda_allocate(most * ATTENTION_KEYS * sizeof(int32_t));
    core.sinks = (float *)probe_cuda_allocate(ATTENTION_HEADS * sizeof(float));
    core.rotation = (float *)probe_cuda_allocate(most * CORE_ROTATION_FLOATS * sizeof(float));
    core.checksums = (uint32_t *)probe_cuda_allocate(most * CORE_GROUPS_PER_ROW * sizeof(uint32_t));
    core.tensor_sums = (float *)probe_cuda_allocate(most * CORE_TENSOR_FLOATS_PER_ROW * sizeof(float));
    core.row_list = (uint32_t *)probe_cuda_allocate(CORE_MOST_POISONED * sizeof(uint32_t));
    core.status_word = (unsigned long long *)probe_cuda_allocate(sizeof(unsigned long long));
    PROBE_CHECK_CUDA(cudaMemset(core.status_word, 0, sizeof(unsigned long long)));
    for (uint32_t kernel = 0; kernel < CORE_KERNEL_COUNT; kernel++) {   /* every kernel may take all a block can */
        const cudaFuncAttribute shared = cudaFuncAttributeMaxDynamicSharedMemorySize;
        PROBE_CHECK_CUDA(cudaFuncSetAttribute(core_kernels[kernel].function, shared, (int)core.most_shared_bytes));
    }
    core_host.queries = (uint16_t *)core_host_allocate(most * CORE_ROW_HALVES * sizeof(uint16_t));
    core_host.outputs = (uint16_t *)core_host_allocate(most * CORE_ROW_HALVES * sizeof(uint16_t));
    core_host.baseline = (uint16_t *)core_host_allocate(most * CORE_ROW_HALVES * sizeof(uint16_t));
    core_host.unpadded = (uint16_t *)core_host_allocate(most * CORE_ROW_HALVES * sizeof(uint16_t));
    core_host.alone = (uint16_t *)core_host_allocate(CORE_ROW_HALVES * sizeof(uint16_t));
    core_host.list = (int32_t *)core_host_allocate(most * ATTENTION_KEYS * sizeof(int32_t));
    core_host.compact_list = (int32_t *)core_host_allocate(most * ATTENTION_KEYS * sizeof(int32_t));
    core_host.checksums = (uint32_t *)core_host_allocate(most * CORE_GROUPS_PER_ROW * sizeof(uint32_t));
    core_host.poisoned = (uint32_t *)core_host_allocate(CORE_MOST_POISONED * sizeof(uint32_t));
    core_host.rotation = (float *)core_host_allocate(most * CORE_ROTATION_FLOATS * sizeof(float));
    core_host.tensor_sums = (float *)core_host_allocate(most * CORE_TENSOR_FLOATS_PER_ROW * sizeof(float));
    core_host.referenced = (uint8_t *)core_host_allocate(core.plane_rows);
    core_host.keys = (double *)core_host_allocate((uint64_t)ATTENTION_KEYS * ATTENTION_DIMS * sizeof(double));
    core_host.references =
        (double *)core_host_allocate((uint64_t)CORE_REFERENCES * CORE_SAMPLED_ROWS * CORE_ROW_HALVES * sizeof(double));
    return 1;
}

static void core_close_card(void *) { probe_cuda_close(); }
/* A value of the case's inputs: (k - 128) 2^-shift, from a hash of its place. */
static float core_input_value(uint64_t place, int shift)
{
    return ldexpf((float)(attention_mix_bits((uint32_t)place + CORE_CHECK_SEED) % CORE_VALUE_STEPS) - CORE_VALUE_CENTRE,
                  -shift);
}

/* The compact list of a timed step: row r's window rows as in its lists, its pool entry e at compact row 512 r + e. */
static void core_timed_compact_list(uint32_t rows)
{
    for (size_t place = 0; place < (size_t)rows * ATTENTION_KEYS; place++) {
        const uint32_t row = (uint32_t)(place / ATTENTION_KEYS), key = (uint32_t)(place % ATTENTION_KEYS);
        core_host.compact_list[place] = key < ATTENTION_WINDOW_KEYS
                                            ? (int32_t)(row + key)
                                            : (int32_t)(row * ATTENTION_POOL_KEYS + key - ATTENTION_WINDOW_KEYS);
    }
    PROBE_CHECK_CUDA(cudaMemcpy(core.compact_list, core_host.compact_list,
                                (size_t)rows * ATTENTION_KEYS * sizeof(int32_t), cudaMemcpyHostToDevice));
}

/* A case's inputs on the card: queries, sinks, the rows' angles, the compact list of a timed step. */
static void core_describe_case(void *, const struct probe_case *probe_case, struct probe_case_facts *facts)
{
    const uint32_t rows = core_rows_of(probe_case);
    facts->format = "fp16";                      /* the baseline's */
    facts->part = "core";
    facts->rows = rows;
    facts->experts = facts->overlap = -1;
    facts->megabytes = -1.0;                     /* mb=na: the kinds read different bytes; RATE gives each's */
    for (uint64_t place = 0; place < (uint64_t)rows * CORE_ROW_HALVES; place++) {
        core_host.queries[place] = core_half_bits(core_input_value(place, CORE_QUERY_SHIFT));
    }
    for (uint32_t head = 0; head < ATTENTION_HEADS; head++) {
        core_host.sinks[head] = core_input_value((uint64_t)CORE_PLANE_ROWS * CORE_ROW_HALVES + head, CORE_SINK_SHIFT);
    }
    for (uint32_t row = 0; row < rows; row++) {
        attention_rotation_row(ATTENTION_FIRST_POSITION + row, core_host.rotation + (size_t)row * CORE_ROTATION_FLOATS);
    }
    PROBE_CHECK_CUDA(cudaMemcpy(core.queries, core_host.queries, (size_t)rows * CORE_ROW_HALVES * sizeof(uint16_t),
                                cudaMemcpyHostToDevice));
    PROBE_CHECK_CUDA(cudaMemcpy(core.sinks, core_host.sinks, sizeof(core_host.sinks), cudaMemcpyHostToDevice));
    PROBE_CHECK_CUDA(cudaMemcpy(core.rotation, core_host.rotation, (size_t)rows * CORE_ROTATION_FLOATS * sizeof(float),
                                cudaMemcpyHostToDevice));
    core_timed_compact_list(rows);
    /* the distinct pool rows a timed step names, from the card's lists of the first (slot 0 until the check's) */
    PROBE_REQUIRE(core_list_launch(0, rows, 1), "the lists of a timed step were refused");
    PROBE_CHECK_CUDA(cudaMemcpy(core_host.list, core_slot_list(0), (size_t)rows * ATTENTION_KEYS * sizeof(int32_t),
                                cudaMemcpyDeviceToHost));
    core_host.pool_distinct = attention_distinct_pool_rows(core_host.list, rows, core_host.referenced, core.plane_rows);
    PROBE_REQUIRE(core_host.pool_distinct != UINT64_MAX, "a timed list past its plane");
}

/* ---- checks ---- */

/* The check's lists on the host and on the card (slot 0, and the compact list): row 0 whole, row 1 no valid key, row
 * 2 its five newest window rows, row 3 300 pool entries, the others whole; every index held to its plane here. Then
 * the rows next to the window span and to row 0's pool entries that no list names: poisoned for the check. */
static void core_make_check_lists(uint32_t rows)
{
    const uint64_t base = core_window_base(0, rows);
    uint64_t candidates[CORE_MOST_POISONED];
    memset(core_host.referenced, 0, core.plane_rows);
    for (uint32_t row = 0; row < rows; row++) {
        for (uint32_t key = 0; key < ATTENTION_KEYS; key++) {
            const size_t place = (size_t)row * ATTENTION_KEYS + key;
            const uint32_t entry = key - ATTENTION_WINDOW_KEYS;
            const int window = key < ATTENTION_WINDOW_KEYS;
            const int few = row == CORE_CORNER_ROW_FEW && (!window || key < ATTENTION_WINDOW_KEYS - CORE_FEW_KEYS);
            const int left_out = row == CORE_CORNER_ROW_EMPTY || few ||
                                 (row == CORE_CORNER_ROW_SHORT && !window && entry >= CORE_SHORT_POOL);
            const uint32_t hash = attention_mix_bits(CORE_CHECK_SEED + attention_mix_bits((uint32_t)place));
            const uint64_t index = window ? (uint64_t)row + key : entry * core.band + hash % core.band;
            PROBE_REQUIRE(left_out || (window ? base + index : index) < core.plane_rows, "a check list past its plane");
            core_host.list[place] = left_out ? -1 : (int32_t)index;
            core_host.compact_list[place] = left_out ? -1 : window ? (int32_t)index
                                                                   : (int32_t)(row * ATTENTION_POOL_KEYS + entry);
            if (!left_out) {
                core_host.referenced[window ? base + index : index] = 1;
            }
        }
    }
    candidates[0] = base - 1u;
    candidates[1] = base + rows + CORE_WINDOW_SPAN_EXTRA;
    for (uint32_t entry = 0; entry < ATTENTION_POOL_KEYS; entry++) {
        candidates[CORE_MOST_POISONED - ATTENTION_POOL_KEYS + entry] =
            (uint64_t)core_host.list[ATTENTION_WINDOW_KEYS + entry] + 1u;
    }
    core_host.poisoned_count = 0;
    for (uint32_t candidate = 0; candidate < CORE_MOST_POISONED; candidate++) {
        if (candidates[candidate] < core.plane_rows && !core_host.referenced[candidates[candidate]]) {
            core_host.referenced[candidates[candidate]] = 1;
            core_host.poisoned[core_host.poisoned_count++] = (uint32_t)candidates[candidate];
        }
    }
    PROBE_CHECK_CUDA(cudaMemcpy(core_slot_list(0), core_host.list, (size_t)rows * ATTENTION_KEYS * sizeof(int32_t),
                                cudaMemcpyHostToDevice));
    PROBE_CHECK_CUDA(cudaMemcpy(core.compact_list, core_host.compact_list, (size_t)rows * ATTENTION_KEYS *
                                sizeof(int32_t), cudaMemcpyHostToDevice));
    PROBE_CHECK_CUDA(cudaMemcpy(core.row_list, core_host.poisoned, core_host.poisoned_count * sizeof(uint32_t),
                                cudaMemcpyHostToDevice));
}

/* Head 5 of row 0 made a multiple of its last key (an fp16 row): its score there about 40, its maximum in the last
 * group, the other keys' p of that head far under 2^-24 of it. */
static void core_corner_query(void)
{
    double key[ATTENTION_DIMS], squares = 0.0;
    uint16_t *head = core_host.queries + (size_t)CORE_CORNER_HEAD * ATTENTION_DIMS;
    reference_row_values(REFERENCE_FORMAT_FP16, core_host.planes[CORE_PLANE_FP16] +
                         (uint64_t)core_host.list[ATTENTION_KEYS - 1] * ATTENTION_FP16_ROW_BYTES, key);
    for (uint32_t dim = 0; dim < ATTENTION_DIMS; dim++) {
        squares += key[dim] * key[dim];
    }
    PROBE_REQUIRE(squares > 0.0, "the corner's key is all zeros");
    for (uint32_t dim = 0; dim < ATTENTION_DIMS; dim++) {
        head[dim] = core_half_bits((float)(CORE_CORNER_SCORE / (squares * ATTENTION_SOFTMAX_SCALE) * key[dim]));
    }
    PROBE_CHECK_CUDA(cudaMemcpy(core.queries + (size_t)CORE_CORNER_HEAD * ATTENTION_DIMS, head,
                                ATTENTION_DIMS * sizeof(uint16_t), cudaMemcpyHostToDevice));
}

/* The reference of each kind for the sampled rows (rows 0 .. 3, the middle and the last), from the host's planes and
 * list, and a SOFTMAX line each. */
static const struct { const char *name; int kind; double step; } core_reference_kinds[CORE_REFERENCES] = {
    {"fp16", ATTENTION_FP16, ATTENTION_CONTRACT_STEP}, {"fp16_step_0", ATTENTION_FP16, 0.0},
    {"trained", ATTENTION_TRAINED, ATTENTION_CONTRACT_STEP}, {"fresh", ATTENTION_FRESH, ATTENTION_CONTRACT_STEP}};

static uint32_t core_reference_of(uint32_t instance)
{
    const int kind = core_kernels[core_variants[instance].kernel].kind;
    return kind == ATTENTION_TRAINED ? (uint32_t)CORE_REFERENCE_TRAINED
           : kind == ATTENTION_FRESH ? (uint32_t)CORE_REFERENCE_FRESH
           : instance == CORE_INSTANCE_rescale_always ? (uint32_t)CORE_REFERENCE_STEP_0 : (uint32_t)CORE_REFERENCE_FP16;
}

static void core_references(const struct probe_case *probe_case, uint32_t rows)
{
    const uint32_t wanted[CORE_SAMPLED_ROWS] = {0, CORE_CORNER_ROW_EMPTY, CORE_CORNER_ROW_FEW, CORE_CORNER_ROW_SHORT,
                                                rows / PROBE_HALF, rows - 1u};
    const uint64_t base = core_window_base(0, rows);
    core_host.sampled_count = 0;
    for (uint32_t index = 0; index < CORE_SAMPLED_ROWS; index++) {
        int seen = wanted[index] >= rows;
        for (uint32_t earlier = 0; earlier < core_host.sampled_count; earlier++) {
            seen |= core_host.sampled[earlier] == wanted[index];
        }
        core_host.sampled[core_host.sampled_count] = wanted[index];
        core_host.sampled_count += seen ? 0u : 1u;
    }
    for (uint32_t reference = 0; reference < CORE_REFERENCES; reference++) {
        const int kind = core_reference_kinds[reference].kind;
        const uint32_t planes[] = {core_window_plane(kind), core_pool_plane(kind)};
        struct reference_counts counts = {0, 0};
        for (uint32_t sample = 0; sample < core_host.sampled_count; sample++) {
            const uint32_t row = core_host.sampled[sample];
            struct reference_row inputs;
            memset(&inputs, 0, sizeof(inputs));
            for (uint32_t key = 0; key < ATTENTION_KEYS; key++) {
                const int32_t index = core_host.list[(size_t)row * ATTENTION_KEYS + key];
                const uint32_t plane = planes[key < ATTENTION_WINDOW_KEYS ? 0 : 1];
                const uint64_t stored_row = key < ATTENTION_WINDOW_KEYS ? base + (uint64_t)index : (uint64_t)index;
                double *values = core_host.keys + (size_t)key * ATTENTION_DIMS;
                if (index >= 0) {
                    reference_row_values(core_reference_formats[plane],
                                         core_host.planes[plane] + stored_row * core_row_bytes[plane], values);
                }
                inputs.keys[key] = index >= 0 ? values : NULL;
            }
            inputs.queries = core_host.queries + (size_t)row * CORE_ROW_HALVES;
            inputs.sinks = core_host.sinks;
            inputs.rotation = core_host.rotation + (size_t)row * CORE_ROTATION_FLOATS;
            inputs.step = core_reference_kinds[reference].step;
            inputs.probability_format =
                kind == ATTENTION_FP16 ? REFERENCE_PROBABILITY_FP16 : REFERENCE_PROBABILITY_BFLOAT16;
            reference_attention_row(&inputs, core_host.references + ((size_t)reference * CORE_SAMPLED_ROWS + sample) *
                                                                        CORE_ROW_HALVES, &counts);
        }
        printf("SOFTMAX card=%s case=%s kind=%s rows=%u rescales_per_row=%s flushed_per_row=%s\n", probe_run.card_label,
               probe_case->name, core_reference_kinds[reference].name, core_host.sampled_count,
               probe_number((double)counts.rescales / core_host.sampled_count, 1),
               probe_number((double)counts.flushed / core_host.sampled_count, 1));
    }
}

/* The checked launch of an instance on rows first_row .. first_row + rows - 1 of the check's lists, every word it
 * writes made stale first; a whole case's launch of pool:compact copies its pool entries first. */
static void core_run_checked(const struct probe_case *probe_case, uint32_t instance, uint32_t first_row, uint32_t rows,
                             int padded)
{
    const struct core_variant *variant = &core_variants[instance];
    const uint32_t case_rows = core_rows_of(probe_case);
    struct core_step step;
    PROBE_REQUIRE(!variant->compact || first_row != 0 || rows != case_rows ||
                  core_compact_launch(core_slot_list(0), core_compact_region(0, case_rows), case_rows),
                  "the check's compact copy was refused");
    PROBE_REQUIRE(core_build_step(probe_case, instance, 0, variant->compact ? core.compact_list : core_slot_list(0),
                                  first_row, rows, padded, &step), "a checked launch was refused");
    const int role = core_kernels[variant->kernel].role;    /* what the role writes: the range its step was held to */
    void *written = role == CORE_ROLE_GATHER ? (void *)step.checksums
                    : role == CORE_ROLE_TENSOR ? (void *)step.tensor_sums : (void *)step.arguments.output;
    PROBE_CHECK_CUDA(cudaMemset(written, CORE_STALE_BYTE,
                                role == CORE_ROLE_GATHER ? (size_t)rows * CORE_GROUPS_PER_ROW * sizeof(uint32_t)
                                : role == CORE_ROLE_TENSOR ? (size_t)rows * CORE_TENSOR_FLOATS_PER_ROW * sizeof(float)
                                                           : (size_t)rows * CORE_ROW_HALVES * sizeof(uint16_t)));
    core_launch(&step);
    PROBE_CHECK_CUDA(cudaDeviceSynchronize());
}

static uint64_t core_differing(const uint16_t *first, const uint16_t *second, size_t words)
{
    uint64_t differing = 0;
    for (size_t word = 0; word < words; word++) {
        differing += first[word] != second[word];
    }
    return differing;
}

/* gather_only's checksums against the host: of every row, the 32-bit words of its listed stored rows, wrapping. */
static uint64_t core_gather_differing(int kind, uint32_t rows)
{
    const uint64_t base = core_window_base(0, rows);
    uint64_t differing = 0;
    for (uint32_t row = 0; row < rows; row++) {
        uint32_t sum = 0;
        for (uint32_t key = 0; key < ATTENTION_KEYS; key++) {
            const int32_t index = core_host.list[(size_t)row * ATTENTION_KEYS + key];
            const int window = key < ATTENTION_WINDOW_KEYS;
            const uint32_t plane = window ? core_window_plane(kind) : core_pool_plane(kind);
            if (index < 0) {
                continue;                        /* no key: no record, nor a pointer to one */
            }
            const uint8_t *record = core_host.planes[plane] + (window ? base + (uint64_t)index : (uint64_t)index) *
                                                                  core_row_bytes[plane];
            for (uint32_t word = 0; word < core_row_bytes[plane] / sizeof(uint32_t); word++) {
                uint32_t value;
                memcpy(&value, record + word * sizeof(uint32_t), sizeof(value));
                sum += value;
            }
        }
        for (uint32_t group = 0; group < CORE_GROUPS_PER_ROW; group++) {
            differing += core_host.checksums[(size_t)row * CORE_GROUPS_PER_ROW + group] != sum ? 1u : 0u;
        }
    }
    return differing;
}

/* An instance's checks on the case's checked launch (the baseline first: it is what the others are held to). */
static void core_check_instance(const struct probe_case *probe_case, uint32_t instance)
{
    const struct core_variant *variant = &core_variants[instance];
    const struct core_kernel *kernel = &core_kernels[variant->kernel];
    const uint32_t rows = core_rows_of(probe_case);
    const char *name = core_instances[instance].name;
    const int must_report = probe_flag_given(PROBE_FLAG_CORRUPT) || probe_flag_has_word(CORE_FLAG_BREAK, "index");
    const size_t words = (size_t)rows * CORE_ROW_HALVES;
    uint64_t differing = 0, compared = 0;
    core_run_checked(probe_case, instance, 0, rows, 1);
    if (kernel->role == CORE_ROLE_TENSOR) {
        PROBE_CHECK_CUDA(cudaMemcpy(core_host.tensor_sums, core.tensor_sums, (size_t)rows * CORE_TENSOR_FLOATS_PER_ROW *
                                    sizeof(float), cudaMemcpyDeviceToHost));
        for (size_t sum = 0; sum < (size_t)rows * CORE_TENSOR_FLOATS_PER_ROW; sum++) {
            differing += core_host.tensor_sums[sum] != attention_tensor_fold() ? 1u : 0u;
        }
        probe_check_count(probe_case->name, name, "tensor_sums", differing,
                          (uint64_t)rows * CORE_TENSOR_FLOATS_PER_ROW, 0);
        probe_digest_add(core_host.tensor_sums, (uint64_t)rows * CORE_TENSOR_FLOATS_PER_ROW * sizeof(float));
        return;
    }
    if (kernel->role == CORE_ROLE_GATHER) {
        PROBE_CHECK_CUDA(cudaMemcpy(core_host.checksums, core.checksums, (size_t)rows * CORE_GROUPS_PER_ROW *
                                    sizeof(uint32_t), cudaMemcpyDeviceToHost));
        probe_check_count(probe_case->name, name, "gather_checksum", core_gather_differing(kernel->kind, rows),
                          (uint64_t)rows * CORE_GROUPS_PER_ROW, must_report);
        probe_digest_add(core_host.checksums, (uint64_t)rows * CORE_GROUPS_PER_ROW * sizeof(uint32_t));
        return;
    }
    PROBE_CHECK_CUDA(cudaMemcpy(core_host.outputs, core.outputs, words * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    double worst = 0.0;
    for (uint32_t sample = 0; sample < core_host.sampled_count; sample++) {
        const double error = reference_error_over_rms(
            core_host.references + ((size_t)core_reference_of(instance) * CORE_SAMPLED_ROWS + sample) * CORE_ROW_HALVES,
            core_host.outputs + (size_t)core_host.sampled[sample] * CORE_ROW_HALVES, CORE_ROW_HALVES);
        worst = !(error <= worst) ? error : worst;                 /* a NaN stays */
    }
    probe_check_error(probe_case->name, name, "host_reference", worst, CORE_TOLERANCE, must_report);
    if (instance == PROBE_BASELINE) {
        memcpy(core_host.baseline, core_host.outputs, words * sizeof(uint16_t));
        core_host.baseline[words / PROBE_HALF] ^= probe_flag_has_word(CORE_FLAG_BREAK, "bits") ? 1u : 0u;
    } else if (variant->bits_required) {
        differing = core_differing(core_host.outputs, core_host.baseline, words);       /* a finding: reported */
        probe_check_count(probe_case->name, name, "bits_against_baseline", differing, words,
                          probe_flag_has_word(CORE_FLAG_BREAK, "bits") || differing != 0);
        core_instances[instance].contract_class = differing != 0 ? 'C' : core_instances[instance].contract_class;
    }
    differing = 0;
    const uint32_t exact_rows = rows <= CORE_ALL_ROWS_EXACT ? rows : (uint32_t)CORE_EXACT_ROWS;
    for (uint32_t index = 0; rows > 1 && index < exact_rows; index++) {
        const uint32_t row = (uint32_t)((uint64_t)index * (rows - 1u) / (exact_rows - 1u));
        core_run_checked(probe_case, instance, row, 1, 1);
        PROBE_CHECK_CUDA(cudaMemcpy(core_host.alone, core.outputs + (size_t)row * CORE_ROW_HALVES,
                                    CORE_ROW_HALVES * sizeof(uint16_t), cudaMemcpyDeviceToHost));
        differing +=
            core_differing(core_host.alone, core_host.outputs + (size_t)row * CORE_ROW_HALVES, CORE_ROW_HALVES);
        compared += CORE_ROW_HALVES;
    }
    if (rows > 1) {
        probe_check_count(probe_case->name, name, "rows_exact", differing, compared,
                          instance == CORE_INSTANCE_group_by_rows);
    }
    if (probe_case_value(probe_case, CORE_FLAG_BLOCKS) != 0 &&
        (instance == PROBE_BASELINE || instance == CORE_INSTANCE_trained || instance == CORE_INSTANCE_fresh)) {
        core_run_checked(probe_case, instance, 0, rows, 0);           /* per kind: the padding alone changes no bit */
        PROBE_CHECK_CUDA(cudaMemcpy(core_host.unpadded, core.outputs, words * sizeof(uint16_t),
                                    cudaMemcpyDeviceToHost));
        differing = core_differing(core_host.unpadded, instance == PROBE_BASELINE ? core_host.baseline
                                                                                  : core_host.outputs, words);
        probe_check_count(probe_case->name, name, "bits_blocks_per_sm", differing, words,
                          (instance == PROBE_BASELINE && probe_flag_has_word(CORE_FLAG_BREAK, "bits")) ||
                              differing != 0);
    }
    probe_digest_add(core_host.outputs, words * sizeof(uint16_t));
}

/* --break-route: the baseline's checked step past one bound at a time (its lists past their array, its window rows
 * past the plane, a block of one warp, shared memory past the card's): the validation every step passes must refuse
 * each; none is launched. */
static void core_break_route(const struct probe_case *probe_case)
{
    static const char *const names[CORE_BROKEN_ROUTES] = {"refused_lists_past_their_array",
                                                          "refused_window_past_the_plane", "refused_block_of_one_warp",
                                                          "refused_shared_past_the_card"};
    for (uint32_t broken = 0; broken < CORE_BROKEN_ROUTES; broken++) {
        struct core_step step;
        PROBE_REQUIRE(core_build_step(probe_case, PROBE_BASELINE, 0, core_slot_list(0), 0, core_rows_of(probe_case), 1,
                                      &step), "--break-route needs a case whose baseline step is valid");
        step.arguments.indices +=
            broken == CORE_BROKEN_LISTS ? probe_cuda_array_bytes(core.lists) / sizeof(int32_t) : 0u;
        step.arguments.window_rows +=
            broken == CORE_BROKEN_WINDOW ? probe_cuda_array_bytes(core.planes[CORE_PLANE_FP16]) : 0u;
        step.threads = broken == CORE_BROKEN_BLOCK ? (uint32_t)ATTENTION_LANES : step.threads;
        step.shared_bytes = broken == CORE_BROKEN_SHARED ? (uint32_t)core.most_shared_bytes + 1u : step.shared_bytes;
        probe_check_count(probe_case->name, "baseline", names[broken],
                          core_step_is_valid(&step, PROBE_BASELINE) ? 0u : 1u, 1, 1);
    }
}

static void core_check_case(void *, const struct probe_case *probe_case)
{
    const uint32_t rows = core_rows_of(probe_case);
    core_make_check_lists(rows);
    if (probe_flag_given(PROBE_FLAG_BREAK_ROUTE)) {
        core_break_route(probe_case);
        return;
    }
    core_corner_query();
    core_fill_rows(core.row_list, core_host.poisoned_count, 1);
    core_references(probe_case, rows);
    if (probe_flag_has_word(CORE_FLAG_BREAK, "index")) {     /* row 0 reads the row before there, the host not */
        const int32_t moved = core_host.list[CORE_BROKEN_KEY] - 1;
        PROBE_REQUIRE(moved >= 0, "the moved index lies before the plane");
        PROBE_CHECK_CUDA(cudaMemcpy(core_slot_list(0) + CORE_BROKEN_KEY, &moved, sizeof(moved),
                                    cudaMemcpyHostToDevice));
    }
    for (uint32_t plane = 0; probe_flag_given(PROBE_FLAG_CORRUPT) && plane < CORE_PLANES; plane++) {
        PROBE_CHECK_CUDA(cudaMemset(core.planes[plane] + core_window_base(0, rows) * core_row_bytes[plane],
                                    CORE_CORRUPTED_BYTE, CORE_CORRUPTED_BYTES));
    }
    for (uint32_t instance = 0; instance < core_instances_in_run; instance++) {
        if (instance != PROBE_BASELINE_AGAIN) {
            core_check_instance(probe_case, instance);
        }
    }
    probe_cuda_check_guards(probe_case->name);
    core_fill_rows(core.row_list, core_host.poisoned_count, 0);              /* the poisoned rows as they were */
    core_timed_compact_list(rows);
}

/* ---- what a case's results say ---- */

/* The acceptance figures (ROUND_1_attention.md 11b): the judged instance's median over the other's; a bar is
 * "at most" (NAN: measured); rows 0: every case. The harness pairs an instance with the baseline alone: a figure of
 * two other instances gets no pairs, so its VERDICT is never met (undecided where the ratio reached the bar) and the
 * batch judges it by the ratio alone (BAR_JUDGED). */
static const struct core_figure {
    const char *name;
    uint32_t judged, against, rows;
    double bar;
} core_figures[] = {     /* CORE_MEASURED: an instance over the baseline at every row count, measured */
#define CORE_MEASURED(id) {#id "_over_baseline", CORE_INSTANCE_##id, PROBE_BASELINE, 0, PROBE_NAN}
    {"heads_8_over_baseline", CORE_INSTANCE_heads_8, PROBE_BASELINE, 1, 0.75},
    {"stages_1_pad_over_gather_shared", CORE_INSTANCE_stages_1_pad, CORE_INSTANCE_gather_shared, 1, 0.50},
    {"gather_only_over_baseline", CORE_INSTANCE_gather_only, PROBE_BASELINE, 1, PROBE_NAN},
    {"trained_gather_over_fp16_gather", CORE_INSTANCE_gather_trained, CORE_INSTANCE_gather_only, 1, 0.60},
    {"fresh_gather_over_fp16_gather", CORE_INSTANCE_gather_fresh, CORE_INSTANCE_gather_only, 1, 0.60},
    {"baseline_over_tensor_only", PROBE_BASELINE, CORE_INSTANCE_tensor_only, CORE_MOST_ROWS, 1.50},
    CORE_MEASURED(gather_shared), CORE_MEASURED(stages_1_pad), CORE_MEASURED(stages_2_pad), CORE_MEASURED(stages_3),
    CORE_MEASURED(pool_compact), CORE_MEASURED(rescale_always), CORE_MEASURED(trained), CORE_MEASURED(fresh)};

static void core_verdicts(void *, const struct probe_case *probe_case, const struct probe_result *results,
                          double noise_ratio)
{
    const uint32_t rows = core_rows_of(probe_case);
    char figure[CORE_FIGURE_BYTES];
    for (uint32_t instance = 0; instance < core_instances_in_run; instance++) {
        double bytes = 0.0, flop = 0.0;
        if (results[instance].mode == NULL) {
            continue;
        }
        core_step_work(instance, rows, &bytes, &flop);
        const double microseconds = results[instance].step.median;
        printf("RATE card=%s case=%s variant=%s gbps=%s tflops=%s\n", probe_run.card_label, probe_case->name,
               core_instances[instance].name,
               probe_number(bytes > 0.0 ? bytes / microseconds / PROBE_GBPS_PER_MB_PER_US : PROBE_NAN, 0),
               probe_number(flop > 0.0 ? flop / microseconds / CORE_MEGA : PROBE_NAN, 1));
    }
    for (uint32_t index = 0; index < PROBE_COUNT_OF(core_figures); index++) {
        const struct core_figure *judged = &core_figures[index];
        const uint32_t paired = judged->judged == PROBE_BASELINE ? judged->against
                                : judged->against == PROBE_BASELINE ? judged->judged : PROBE_NONE;
        const double ratio = results[judged->judged].step.median / results[judged->against].step.median;
        if (judged->rows != 0 && judged->rows != rows) {
            continue;
        }
        PROBE_REQUIRE(snprintf(figure, sizeof(figure), "%s_at_%s", judged->name, probe_case->name) > 0, "a figure");
        probe_verdict(figure, judged->bar, judged->bar * results[judged->against].step.median,
                      results[judged->judged].mode != NULL && results[judged->against].mode != NULL ? ratio : PROBE_NAN,
                      noise_ratio, paired != PROBE_NONE ? results[paired].same_sign : 0u,
                      paired != PROBE_NONE ? results[paired].rounds : 0u);
    }
}

static int core_configure(void *);
static struct probe_program core_program = {
    "attention_core", core_flags, PROBE_COUNT_OF(core_flags), core_instances, CORE_INSTANCE_COUNT - 1u, NULL,
    core_configure, core_open_card, core_describe_case, core_check_case, core_prepare_step, core_launch_step,
    probe_cuda_mark_start, probe_cuda_elapsed_microseconds, core_verdicts, core_close_card, NULL};

/* The harness knows break switches only: --break NAME is made one here, and the instance that breaks row
 * exactness joins the run with its flag. Each instance takes its kernel's name, block and bound. */
static int core_configure(void *)
{
    const struct probe_flag *rows = probe_flag(CORE_FLAG_ROWS);
    probe_run.breaking |= probe_flag_given(CORE_FLAG_BREAK) ? 1u : 0u;
    core_instances_in_run += probe_flag_has_word(CORE_FLAG_BREAK, "exactness") ? 1u : 0u;
    core_program.instance_count = core_instances_in_run;
    for (uint32_t value = 0; value < rows->value_count; value++) {
        core.most_rows = rows->values[value] > core.most_rows ? (uint32_t)rows->values[value] : core.most_rows;
    }
    if (probe_flag_has_word(CORE_FLAG_BREAK, "exactness") && rows->values[0] < PROBE_HALF) {
        fprintf(stderr, "--break exactness needs a first case of 2 rows or more: a break checks the first case only\n");
        return 0;
    }
    for (uint32_t instance = 0; instance < CORE_INSTANCE_COUNT; instance++) {
        const struct core_kernel *kernel = &core_kernels[core_variants[instance].kernel];
        core_instances[instance].kernel = kernel->name;
        core_instances[instance].block_threads = attention_threads(kernel->head_tiles);
        core_instances[instance].bound_threads = (uint32_t)kernel->bounded;
        core_instances[instance].bound_blocks = kernel->bounded != 0 ? 1u : 0u;
    }
    return 1;
}

int main(int argument_count, char **arguments) { return probe_main(argument_count, arguments, &core_program); }
