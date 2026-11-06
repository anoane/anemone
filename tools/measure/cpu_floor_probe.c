/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * cpu_floor_probe: where the time of a routed expert goes when the processor computes it and no kernel changes it
 * (speed campaign, round 1, unit 8a; ledger rows C1, C2, C15).
 *
 * b04 found one cold expert at 262 us on 96 threads: 31 us are the one-thread arithmetic shared out, 30 us the
 * activation, 200 us neither, and no case of 36 was under 209 us. This probe takes that probe's kernel and ring
 * (cpu_probe_common.h) and measures, each as a list on the command line and all of one run interleaved:
 *   --kernel fp32,read      fp32: b04's kernel as it is (two parallel regions, gate with up and then down, the
 *                           activation between them on the calling thread). read: the same threads sum the bytes
 *                           of the same rows and do nothing else: what the memory delivers to any kernel.
 *   --threads 1,..,96       threads of the team; --placement die,packed,smt: the processors they are pinned to;
 *   --pool openmp,own       an OpenMP team as in b04, or the minimal spinning pool of the common header; docked
 *                           (one thread only): the OpenMP team beside the runtime's whole-machine team, docked
 *                           and spinning, which is how b04 ran its one-thread case (bystanders= in RESULT);
 *   --experts 1,2,3,6,9     cold experts computed at once, the threads shared evenly among them (C15);
 *   --timeline              two stamps per thread and region: per step the wake-up, the work and the wait at the
 *                           barrier of the thread that finished last, the activation, and the same over all threads.
 * A run is --rounds rounds; a round gives every instance one slot: its team is started and pinned, led in for
 * --lead-in-us (cores at their clock, pages mapped), timed for --steps-per-round steps and ended, so that no
 * thread of one instance spins beside another. The first instance is the baseline and has a second slot in the
 * middle of every round (with three instances or more neither then follows the other); which of the two is "again"
 * changes from round to round, and their difference is the run's noise. Every step takes its experts in turn from
 * the ring, 2.24 GiB that no cache holds. --perf-control <fifo>,<fifo> has `perf stat --control` count the rounds
 * only (every slot's thread start, lead-in and timed steps; not the filling, the references, the checks) and ends
 * the run where perf does not answer. A run under any tracer is no timing: --profiled names it mode=profile.
 *
 * Checked before anything is timed, for every instance (CHECK lines), all but the first on a team of its own:
 *   shares_inside    when the instance is planned: every thread's rows lie inside its matrix and its expert among
 *                    the step's. No kernel is handed a share that fails; the debug build asks again at every use;
 *   host_reference   fp32: every output of every expert of a step against double precision from the format's
 *                    definition, rms error within 1e-6 of the rms (b04: its worst of six outputs was 6.5e-7). The
 *                    reference's down product takes the reference's hidden vector: no stage here is quantized, so
 *                    no rounding upstream can fall the other way and fail a product downstream;
 *   same_bits        fp32: the outputs are bit for bit those of the first instance that computed that expert;
 *   byte_sum         read: the sum of an expert's bytes against the sum plain host code took when it was written;
 *                    this one is also compared after every timed step (wrong_sums in RESULT);
 *   pinning          every thread says, without being pinned again, the processor it runs on: at the check, after
 *                    every slot, and with --timeline at the start of every share.
 * Each is shown able to fail by a flag; such a run times nothing and exits 0 only when its check reported and no
 * other did: --corrupt overwrites a megabyte of the first checked expert's gate matrix after its reference and its
 * byte sum were taken (host_reference and byte_sum must report); --break-bits doubles the route weight of every
 * instance but the first, an input and not a kernel switch (same_bits must report, host_reference must not);
 * --break-pinning pins every thread one place further in the placement's order than the run announces (pinning);
 * --break-share plans one share a row past its matrix (shares_inside must report, and no step is run).
 *
 * What it does not show: the bits of a card (this order of additions is no card's: that is unit 8b), the cost of a
 * pool that sleeps (ledger C4), and nothing about real weights: bytes are hashes of their position. A kernel here
 * indexes only by the rows of a share that passed shares_inside and by ring positions taken modulo the ring; the
 * debug build also counts the bytes of every share (DIAG lines).
 *
 * Lines (CONVENTIONS.md 6, card=host): PLACEMENT (an instance's processors), HOST, CHECK, LOAD (the steps a round
 * has after --load-seconds was applied: the modules heat), ROUND (wall clock and the modules' temperatures), NOISE,
 * RESULT (busy_threads: the process's processor time over the length of the timed steps), TIMELINE, BEST and
 * VERDICT. A debug build's timings are no results: its lines begin with DEBUG_.
 *
 *   cpu_floor_probe [--kernel L] [--threads L] [--placement L] [--pool L] [--experts L] [--timeline] [--rounds N]
 *                   [--steps-per-round N] [--lead-in-us N] [--load-seconds N] [--ring-experts N] [--batch NAME]
 *                   [--perf-control FIFO,FIFO] [--profiled] [--corrupt | --break-bits | --break-pinning |
 *                   --break-share]
 *
 * Measurement tool: not part of the engine.
 */
#include "cpu_probe_common.h"
#include <fcntl.h>
#include <float.h>
#include <poll.h>
#include <unistd.h>

enum probe_constant {
    KERNEL_FP32 = 0, KERNEL_READ, KERNEL_KINDS, REGION_GATE_UP = 0, REGION_DOWN, REGIONS_PER_STEP,
    GATE_UP_ROWS = 2 * CPU_EXPERT_INNER_SIZE, MAX_EXPERTS_AT_ONCE = 9, MAX_THREADS = 256, MAX_INSTANCES = 64,
    MAX_LIST = 16, ROUTE_WEIGHTS = 2,
    /* the ring: its stride visits every expert before one returns as long as the two share no factor */
    DEFAULT_RING_EXPERTS = 128, RING_EXPERTS_MIN = 12, RING_EXPERTS_MAX = 4096, RING_STRIDE = 37,
    FIRST_CHECKED_EXPERT = 1, CACHES_OF_RING_MIN = 4,
    /* the protocol of CONVENTIONS.md 3.3 and 8 */
    DEFAULT_ROUNDS = 20, ROUNDS_MAX = CPU_ROUNDS_MAX, DEFAULT_STEPS_PER_ROUND = 100, STEPS_PER_ROUND_MAX = 2000,
    STEPS_PER_ROUND_MIN = 5, LEAD_IN_STEPS_MIN = 4, WARM_UP_STEPS = 8, DEFAULT_LEAD_IN_MICROSECONDS = 20000,
    LEAD_IN_MICROSECONDS_MAX = 1000000, DEFAULT_LOAD_SECONDS = 60, LOAD_SECONDS_MAX = 600, SLOT_HEAD_SHARE = 10,
    HOT_MODULE_C = 70, PERCENT = 100, SAME_SIGN_PERCENT_MIN = 85, BASELINE_SLOTS = 2, COMPARED_INSTANCES_MIN = 2,
    INPUT_SEED = 7, INPUT_VALUE_STEPS = 255, CORRUPTED_BYTES = 1024 * 1024, CORRUPTED_BYTE_VALUE = 0x5A,
    UNWRITTEN_BYTE = 0xFF, VARIANT_BYTES = 96, DETAIL_BYTES = 200, ACKNOWLEDGE_MILLISECONDS = 5000
};
_Static_assert(FIRST_CHECKED_EXPERT + MAX_EXPERTS_AT_ONCE <= RING_EXPERTS_MIN, "the checked experts lie in any ring");
#define SWIGLU_LIMIT 10.0f
#define ROUTE_WEIGHT 0.25f
#define INPUT_VALUE_OFFSET 127.0f                /* synthetic inputs -2 .. 2 in steps of 1/64, as b04's */
#define INPUT_VALUE_DIVISOR 64.0f
#define FP32_RMS_ERROR_LIMIT 1.0e-6              /* CONVENTIONS.md 5, check 1: an fp32 result against its rms */
#define WARM_UP_SECONDS 1.0
#define BYTES_PER_GB 1.0e9
#if PROBE_DIAGNOSTICS                            /* a debug timing is never printed as a result: its lines are */
#define LINE_PREFIX "DEBUG_"                     /* DEBUG_RESULT and so on, the code under the sanitizers the same */
#define BUILD_NAME "debug"
#else
#define LINE_PREFIX ""
#define BUILD_NAME "release"
#endif

/* What a step's stamps come to, in microseconds, the two regions added. The first three follow the thread that
 * left its region last: they and the activation are the step's time but for the calls between them. */
enum timeline_term {
    TERM_WAKE, TERM_WORK, TERM_TAIL, TERM_ACTIVATION, TERM_OTHER, TERM_WAKE_MEAN, TERM_WAKE_LATEST, TERM_WORK_MEAN,
    TERM_WORK_SHORTEST, TERM_WORK_LONGEST, TERM_IDLE_MEAN, TERM_COUNT
};
static const char *const term_names[TERM_COUNT] = {
    "wake_us", "work_us", "tail_us", "activation_us", "other_us", "wake_mean_us", "wake_latest_us", "work_mean_us",
    "work_shortest_us", "work_longest_us", "idle_mean_us"};
static const char *const kernel_names[KERNEL_KINDS] = {"fp32", "read"};

struct row_share {                               /* what one thread does in one region */
    int expert_slot, first_row, row_end;
};

struct thread_result {                           /* a thread's own line: the read kernel's sum, the debug counts */
    _Alignas(CPU_CACHE_LINE_BYTES) uint64_t byte_sum;
    uint64_t bytes_read, refused;
};

struct step_stamps {
    uint64_t started, activation_started, activation_ended, ended;
};

struct instance {
    int kernel, thread_count, placement, pool, experts_at_once;
    char variant[VARIANT_BYTES];
    int announced_cpus[MAX_THREADS], pinned_cpus[MAX_THREADS];
    float route_weight;
    struct row_share share[REGIONS_PER_STEP][MAX_THREADS];
    int expert_index[MAX_EXPERTS_AT_ONCE];       /* the ring positions of the step in hand */
    double *step_microseconds, *again_microseconds, *terms[TERM_COUNT];
    double estimated_microseconds, expert_microseconds, busy_seconds, timed_seconds;
    uint64_t switches, bytes_read, refused_shares;
    long misplaced, backwards, wrong_sums;
    int checks_held, bystanders;
};

static struct {
    /* the command line */
    int kernels[MAX_LIST], kernel_count, threads[MAX_LIST], thread_cases, placements[MAX_LIST], placement_count;
    int pools[MAX_LIST], pool_count, experts[MAX_LIST], experts_cases;
    int timeline, profiled, rounds, steps_per_round, lead_in_microseconds, load_seconds, ring_experts;
    int corrupt, break_bits, break_pinning, break_share;
    const char *batch_name, *perf_control;
    /* the machine, the ring, the instances */
    double ticks_per_microsecond;
    struct cpu_topology topology;
    struct cpu_ring ring;
    struct instance instances[MAX_INSTANCES];
    int instance_count, checks_as_required, checks_reported, hottest_module_c;
    uint64_t ring_walk;
    /* what the threads of the running team read and write */
    float input[CPU_HIDDEN_SIZE];
    float gate_up[MAX_EXPERTS_AT_ONCE][GATE_UP_ROWS], hidden[MAX_EXPERTS_AT_ONCE][CPU_EXPERT_INNER_SIZE];
    float output[MAX_EXPERTS_AT_ONCE][CPU_HIDDEN_SIZE];
    struct thread_result results[REGIONS_PER_STEP][MAX_THREADS];
    struct cpu_thread_stamps stamps[REGIONS_PER_STEP][MAX_THREADS];
    struct cpu_region_record records[REGIONS_PER_STEP];
    /* the checks' references */
    double reference[ROUTE_WEIGHTS][MAX_EXPERTS_AT_ONCE][CPU_HIDDEN_SIZE];
    float first_bits[MAX_EXPERTS_AT_ONCE][CPU_HIDDEN_SIZE];
    int first_bits_known[MAX_EXPERTS_AT_ONCE];
} run;

static _Noreturn void fail(const char *what)
{
    (void)fflush(stdout);                        /* a step's log is one file: what was printed stands before this */
    fprintf(stderr, "cpu_floor_probe: %s\n", what);
    exit(EXIT_FAILURE);
}

static double microseconds_between(uint64_t from_ticks, uint64_t to_ticks)
{
    return (double)(int64_t)(to_ticks - from_ticks) / run.ticks_per_microsecond;     /* negative: to is earlier */
}

/* ---- a step: gate with up, the activation, down ---- */

/* Whether a thread's share of a region lies inside what its kernel indexes by it: its rows inside the region's
 * rows, its expert among the step's. Asked of every share when its instance is planned (CHECK shares_inside). */
static int share_is_inside(const struct instance *instance, int region, int thread_index)
{
    const int rows = region == REGION_GATE_UP ? (int)GATE_UP_ROWS : (int)CPU_HIDDEN_SIZE;
    const struct row_share share = instance->share[region][thread_index];
    return share.expert_slot >= 0 && share.expert_slot < instance->experts_at_once && share.first_row >= 0 &&
           share.first_row <= share.row_end && share.row_end <= rows;
}

#if PROBE_DIAGNOSTICS
/* The debug build's second look at a share, where it is used: the same question, and its expert inside the ring.
 * It counts the bytes the share reads; a share that fails is not worked and is counted. */
static int diagnose_share(const struct instance *instance, int region, int thread_index)
{
    const int row_weights = region == REGION_GATE_UP ? CPU_HIDDEN_SIZE : CPU_EXPERT_INNER_SIZE;
    const uint64_t row_bytes = (uint64_t)(row_weights / CPU_NIBBLES_PER_BYTE +
                                          row_weights / CPU_WEIGHTS_PER_SCALE_GROUP);
    if (thread_index < 0 || thread_index >= instance->thread_count) {
        return 0;                                /* no thread of this team: it has no line to count on either */
    }
    const struct row_share share = instance->share[region][thread_index];
    struct thread_result *result = &run.results[region][thread_index];
    const int inside = share_is_inside(instance, region, thread_index) &&
                       instance->expert_index[share.expert_slot] >= 0 &&
                       instance->expert_index[share.expert_slot] < run.ring.expert_count;
    result->bytes_read += inside ? (uint64_t)(share.row_end - share.first_row) * row_bytes : 0;
    result->refused += (uint64_t)!inside;
    return inside;
}
#define DIAGNOSE_SHARE(instance, region, thread_index)              \
    if (!diagnose_share(instance, region, thread_index)) {          \
        return;                                                     \
    }
#else
#define DIAGNOSE_SHARE(instance, region, thread_index)
#endif

static void work_gate_up_fp32(void *context, int thread_index)
{
    const struct instance *instance = context;
    const struct row_share share = instance->share[REGION_GATE_UP][thread_index];
    DIAGNOSE_SHARE(instance, REGION_GATE_UP, thread_index)
    const struct cpu_expert *expert = &run.ring.experts[instance->expert_index[share.expert_slot]];
    float *gate_up = run.gate_up[share.expert_slot];
    for (int row = share.first_row; row < share.row_end; row++) {
        gate_up[row] = row < CPU_EXPERT_INNER_SIZE
                           ? cpu_row_dot_fp32(&expert->matrix[CPU_GATE], row, run.input, run.ring.scale_times_half)
                           : cpu_row_dot_fp32(&expert->matrix[CPU_UP], row - CPU_EXPERT_INNER_SIZE, run.input,
                                              run.ring.scale_times_half);
    }
}

static void work_down_fp32(void *context, int thread_index)
{
    const struct instance *instance = context;
    const struct row_share share = instance->share[REGION_DOWN][thread_index];
    DIAGNOSE_SHARE(instance, REGION_DOWN, thread_index)
    const struct cpu_matrix *down = &run.ring.experts[instance->expert_index[share.expert_slot]].matrix[CPU_DOWN];
    for (int row = share.first_row; row < share.row_end; row++) {
        run.output[share.expert_slot][row] = cpu_row_dot_fp32(down, row, run.hidden[share.expert_slot],
                                                              run.ring.scale_times_half);
    }
}

static void work_gate_up_read(void *context, int thread_index)
{
    const struct instance *instance = context;
    const struct row_share share = instance->share[REGION_GATE_UP][thread_index];
    DIAGNOSE_SHARE(instance, REGION_GATE_UP, thread_index)
    const struct cpu_expert *expert = &run.ring.experts[instance->expert_index[share.expert_slot]];
    const int gate_end = share.row_end < CPU_EXPERT_INNER_SIZE ? share.row_end : CPU_EXPERT_INNER_SIZE;
    const int up_first = share.first_row > CPU_EXPERT_INNER_SIZE ? share.first_row : CPU_EXPERT_INNER_SIZE;
    run.results[REGION_GATE_UP][thread_index].byte_sum =
        cpu_sum_matrix_rows(&expert->matrix[CPU_GATE], share.first_row, gate_end) +
        cpu_sum_matrix_rows(&expert->matrix[CPU_UP], up_first - CPU_EXPERT_INNER_SIZE,
                            share.row_end - CPU_EXPERT_INNER_SIZE);
}

static void work_down_read(void *context, int thread_index)
{
    const struct instance *instance = context;
    const struct row_share share = instance->share[REGION_DOWN][thread_index];
    DIAGNOSE_SHARE(instance, REGION_DOWN, thread_index)
    const struct cpu_matrix *down = &run.ring.experts[instance->expert_index[share.expert_slot]].matrix[CPU_DOWN];
    run.results[REGION_DOWN][thread_index].byte_sum = cpu_sum_matrix_rows(down, share.first_row, share.row_end);
}

/* b04's activation, on the calling thread: h = silu(min(gate, L)) * clamp(up, -L, L) * route weight. */
static void activate(const struct instance *instance, int expert_slot)
{
    const float *gate = run.gate_up[expert_slot], *up = gate + CPU_EXPERT_INNER_SIZE;
    float *hidden = run.hidden[expert_slot];
    for (int row = 0; row < CPU_EXPERT_INNER_SIZE; row++) {
        const float limited_gate = gate[row] < SWIGLU_LIMIT ? gate[row] : SWIGLU_LIMIT;
        const float limited_up = up[row] < -SWIGLU_LIMIT ? -SWIGLU_LIMIT
                                                         : (up[row] > SWIGLU_LIMIT ? SWIGLU_LIMIT : up[row]);
        hidden[row] = limited_gate / (1.0f + expf(-limited_gate)) * limited_up * instance->route_weight;
    }
}

/* One step of an instance on its team: the timed region is from stamps->started to stamps->ended. Returns 0 when
 * the team did not do a whole region; nothing of such a step is used. */
static int run_step(struct instance *instance, struct cpu_team *team, struct step_stamps *stamps)
{
    const int is_fp32 = instance->kernel == KERNEL_FP32;
    struct cpu_region_record *records = run.timeline ? run.records : NULL;
    stamps->started = __rdtsc();
    int whole = cpu_team_run(team, is_fp32 ? work_gate_up_fp32 : work_gate_up_read, instance,
                             records != NULL ? &records[REGION_GATE_UP] : NULL);
    stamps->activation_started = __rdtsc();
    for (int slot = 0; is_fp32 && slot < instance->experts_at_once; slot++) {
        activate(instance, slot);
    }
    stamps->activation_ended = __rdtsc();
    whole &= cpu_team_run(team, is_fp32 ? work_down_fp32 : work_down_read, instance,
                          records != NULL ? &records[REGION_DOWN] : NULL);
    stamps->ended = __rdtsc();
    return whole;
}

/* The next experts of the ring for a step: a position returns only after every other was read once. */
static void take_experts(struct instance *instance)
{
    for (int slot = 0; slot < instance->experts_at_once; slot++) {
        instance->expert_index[slot] = (int)(run.ring_walk++ * RING_STRIDE % (uint64_t)run.ring.expert_count);
    }
}

/* The read kernel's result of the step in hand, per expert: the threads' sums against the host's. Returns how
 * many experts differ. */
static int wrong_byte_sums(const struct instance *instance)
{
    uint64_t found[MAX_EXPERTS_AT_ONCE] = {0};
    int wrong = 0;
    for (int thread = 0; thread < instance->thread_count; thread++) {
        found[instance->share[REGION_GATE_UP][thread].expert_slot] +=
            run.results[REGION_GATE_UP][thread].byte_sum + run.results[REGION_DOWN][thread].byte_sum;
    }
    for (int slot = 0; slot < instance->experts_at_once; slot++) {
        uint64_t expected = 0;
        (void)cpu_expert_bytes(&run.ring.experts[instance->expert_index[slot]], &expected);
        wrong += found[slot] != expected;
    }
    return wrong;
}

/* ---- the checks ---- */

/* An expert's outputs in double precision from the stored bytes. The hidden vector is rounded to fp32 where the
 * kernel hands it to the down matrix as fp32. */
static void reference_outputs(int expert_index, float route_weight, double *outputs)
{
    static double input[CPU_HIDDEN_SIZE], hidden[CPU_EXPERT_INNER_SIZE];
    const struct cpu_expert *expert = &run.ring.experts[expert_index];
    const double limit = (double)SWIGLU_LIMIT;
    for (int column = 0; column < CPU_HIDDEN_SIZE; column++) {
        input[column] = (double)run.input[column];
    }
    #pragma omp parallel for if(CPU_OPENMP_HELPS) schedule(static)
    for (int row = 0; row < CPU_EXPERT_INNER_SIZE; row++) {
        double gate = cpu_reference_row(&run.ring, &expert->matrix[CPU_GATE], row, input);
        double up = cpu_reference_row(&run.ring, &expert->matrix[CPU_UP], row, input);
        gate = gate < limit ? gate : limit;
        up = up < -limit ? -limit : (up > limit ? limit : up);
        hidden[row] = (double)(float)(gate / (1.0 + exp(-gate)) * up * (double)route_weight);
    }
    #pragma omp parallel for if(CPU_OPENMP_HELPS) schedule(static)
    for (int row = 0; row < CPU_HIDDEN_SIZE; row++) {
        outputs[row] = cpu_reference_row(&run.ring, &expert->matrix[CPU_DOWN], row, hidden);
    }
}

/* One CHECK line. `failed` is what the check found, `required` whether this run's flag requires it to fail. */
static void report_check(struct instance *instance, const char *name, const char *detail, int failed, int required)
{
    printf("CHECK  card=host case=experts:%d variant=%s name=%s %s verdict=%s\n", instance->experts_at_once,
           instance->variant, name, detail, failed ? (required ? "reported" : "FAILED") : (required ? "FAILED" : "ok"));
    instance->checks_held &= !failed;
    run.checks_as_required &= failed == required;
    run.checks_reported += failed && required;
}

/* How a run with a break flag ends: well only when a check reported, and none came out otherwise than required. */
static int end_of_broken_run(void)
{
    const int as_required = run.checks_as_required && run.checks_reported > 0;
    printf("%s\n", as_required ? "the broken run was reported, and nothing else was"
                               : "FAILED: a check did not come out as its flag requires");
    return as_required ? EXIT_SUCCESS : EXIT_FAILURE;
}

/* The checks of one instance on a team of its own: one step on the first experts_at_once checked experts. What
 * the step is to write holds, before it, values no kernel produces: a share that does not run is then seen. */
static void check_instance(struct instance *instance, int is_first_of_kernel)
{
    struct cpu_team team;
    struct step_stamps stamps;
    char detail[DETAIL_BYTES];
    const int weight = instance->route_weight == ROUTE_WEIGHT ? 0 : 1;
    if (!cpu_team_open(&team, (enum cpu_pool_kind)instance->pool, instance->thread_count, instance->pinned_cpus,
                       run.ticks_per_microsecond)) {
        fail("a team could not be started or pinned (is every listed processor this process's to use?)");
    }
    instance->bystanders = team.bystanders;
    memset(run.gate_up, UNWRITTEN_BYTE, sizeof(run.gate_up));                /* four of these bytes are a NaN */
    memset(run.hidden, UNWRITTEN_BYTE, sizeof(run.hidden));
    memset(run.output, UNWRITTEN_BYTE, sizeof(run.output));
    memset(run.results, 0, sizeof(run.results));
    for (int slot = 0; slot < instance->experts_at_once; slot++) {
        instance->expert_index[slot] = FIRST_CHECKED_EXPERT + slot;
    }
    if (!run_step(instance, &team, &stamps)) {
        fail("a team did not finish the checked step");
    }
    const int misplaced = cpu_team_misplaced(&team, instance->announced_cpus);
    if (!cpu_team_close(&team)) {
        fail("a team could not be ended");
    }
    if (instance->kernel == KERNEL_FP32) {
        double worst_share = 0.0, worst_single = 0.0;
        long differing = 0, compared = 0;
        for (int slot = 0; slot < instance->experts_at_once; slot++) {
            const double *reference = run.reference[weight][slot];
            double error_squares = 0.0, squares = 0.0, largest = 0.0;
            for (int row = 0; row < CPU_HIDDEN_SIZE; row++) {
                const double error = fabs((double)run.output[slot][row] - reference[row]);
                error_squares += error * error;
                squares += reference[row] * reference[row];
                largest = error > largest || isnan(error) ? error : largest;
                differing += run.first_bits_known[slot] &&
                             memcmp(&run.output[slot][row], &run.first_bits[slot][row], sizeof(float)) != 0;
            }
            const double slot_error = sqrt(error_squares / squares);         /* a NaN, once seen, stays the worst */
            worst_share = slot_error > worst_share || isnan(slot_error) ? slot_error : worst_share;
            const double single = largest / sqrt(squares / (double)CPU_HIDDEN_SIZE);
            worst_single = single > worst_single || isnan(single) ? single : worst_single;
            compared += run.first_bits_known[slot] ? CPU_HIDDEN_SIZE : 0;
            if (!run.first_bits_known[slot]) {
                memcpy(run.first_bits[slot], run.output[slot], sizeof(run.first_bits[slot]));
                run.first_bits_known[slot] = 1;
            }
        }
        (void)snprintf(detail, sizeof(detail), "error_over_rms=%.1e limit=%.1e worst_output_over_rms=%.1e",
                       worst_share, FP32_RMS_ERROR_LIMIT, worst_single);
        report_check(instance, "host_reference", detail, !(worst_share <= FP32_RMS_ERROR_LIMIT), run.corrupt);
        (void)snprintf(detail, sizeof(detail), "differing=%ld of=%ld", differing, compared);
        report_check(instance, "same_bits", detail, differing != 0, run.break_bits && !is_first_of_kernel);
    } else {
        const int wrong = wrong_byte_sums(instance);
        (void)snprintf(detail, sizeof(detail), "differing=%d of=%d", wrong, instance->experts_at_once);
        report_check(instance, "byte_sum", detail, wrong != 0, run.corrupt);
    }
    (void)snprintf(detail, sizeof(detail), "misplaced=%d of=%d", misplaced, instance->thread_count);
    report_check(instance, "pinning", detail, misplaced != 0, run.break_pinning);
}

/* ---- the timed protocol ---- */

/* The stamps of the step in hand as the terms of the timeline, and what the stamps themselves show to be wrong:
 * a share that began on another processor than announced, a thread's stamp before the caller's. */
static void keep_timeline(struct instance *instance, int step_index, const struct step_stamps *stamps)
{
    double term[TERM_COUNT] = {0.0};
    const double threads = (double)instance->thread_count;
    for (int region = 0; region < REGIONS_PER_STEP; region++) {
        const struct cpu_region_record *record = &run.records[region];
        double wake_latest = -DBL_MAX, work_shortest = DBL_MAX, work_longest = 0.0;
        int last = 0;
        for (int thread = 0; thread < instance->thread_count; thread++) {
            const struct cpu_thread_stamps *stamp = &record->threads[thread];
            const double wake = microseconds_between(record->forked_ticks, stamp->entered_ticks);
            const double work = microseconds_between(stamp->entered_ticks, stamp->left_ticks);
            last = stamp->left_ticks > record->threads[last].left_ticks ? thread : last;
            wake_latest = wake > wake_latest ? wake : wake_latest;
            work_shortest = work < work_shortest ? work : work_shortest;
            work_longest = work > work_longest ? work : work_longest;
            term[TERM_WAKE_MEAN] += wake / threads;
            term[TERM_WORK_MEAN] += work / threads;
            term[TERM_IDLE_MEAN] += microseconds_between(stamp->left_ticks, record->joined_ticks) / threads;
            instance->misplaced += stamp->entered_cpu != instance->announced_cpus[thread];
            instance->backwards += wake < 0.0;
        }
        term[TERM_WAKE] += microseconds_between(record->forked_ticks, record->threads[last].entered_ticks);
        term[TERM_WORK] += microseconds_between(record->threads[last].entered_ticks, record->threads[last].left_ticks);
        term[TERM_TAIL] += microseconds_between(record->threads[last].left_ticks, record->joined_ticks);
        term[TERM_WAKE_LATEST] += wake_latest;
        term[TERM_WORK_SHORTEST] += work_shortest;
        term[TERM_WORK_LONGEST] += work_longest;
    }
    term[TERM_ACTIVATION] = microseconds_between(stamps->activation_started, stamps->activation_ended);
    term[TERM_OTHER] = microseconds_between(stamps->started, stamps->ended) - term[TERM_WAKE] - term[TERM_WORK] -
                       term[TERM_TAIL] - term[TERM_ACTIVATION];
    for (int kind = 0; kind < TERM_COUNT; kind++) {
        instance->terms[kind][step_index] = term[kind];
    }
}

/* One slot: the instance's team is started and pinned, led in, timed for `steps` steps into microseconds[], and
 * ended. first_step is the step's place in the instance's timeline, or negative for a slot that keeps none (the
 * warm-up, the baseline's second slot). Returns 0 when the team failed; the run then ends. */
static int run_slot(struct instance *instance, int steps, double *microseconds, int first_step)
{
    struct cpu_team team;
    struct step_stamps stamps;
    uint64_t switches_before = 0, switches_after = 0;
    if (!cpu_team_open(&team, (enum cpu_pool_kind)instance->pool, instance->thread_count, instance->pinned_cpus,
                       run.ticks_per_microsecond)) {
        return 0;
    }
    const uint64_t led_in_from = __rdtsc();
    for (int step = 0; step < LEAD_IN_STEPS_MIN ||
                       microseconds_between(led_in_from, __rdtsc()) < (double)run.lead_in_microseconds; step++) {
        take_experts(instance);
        if (!run_step(instance, &team, &stamps)) {
            return 0;
        }
    }
    memset(run.results, 0, sizeof(run.results));
    if (!cpu_team_context_switches(&team, &switches_before)) {
        return 0;
    }
    const double busy_before = cpu_process_seconds();
    const uint64_t timed_from = __rdtsc();
    for (int step = 0; step < steps; step++) {
        take_experts(instance);
        if (!run_step(instance, &team, &stamps)) {
            return 0;
        }
        microseconds[step] = microseconds_between(stamps.started, stamps.ended);
        instance->wrong_sums += instance->kernel == KERNEL_READ ? wrong_byte_sums(instance) : 0;
        if (run.timeline && first_step >= 0) {
            keep_timeline(instance, first_step + step, &stamps);
        }
    }
    const double timed_seconds = microseconds_between(timed_from, __rdtsc()) / CPU_MICROSECONDS_PER_SECOND;
    const double busy_seconds = cpu_process_seconds() - busy_before;
    if (!cpu_team_context_switches(&team, &switches_after)) {
        return 0;
    }
    for (int thread = 0; first_step >= 0 && thread < instance->thread_count; thread++) {
        instance->bytes_read += run.results[REGION_GATE_UP][thread].bytes_read +
                                run.results[REGION_DOWN][thread].bytes_read;
        instance->refused_shares += run.results[REGION_GATE_UP][thread].refused +
                                    run.results[REGION_DOWN][thread].refused;
    }
    instance->switches += first_step >= 0 ? switches_after - switches_before : 0;
    instance->busy_seconds += first_step >= 0 ? busy_seconds : 0.0;
    instance->timed_seconds += first_step >= 0 ? timed_seconds : 0.0;
    instance->misplaced += cpu_team_misplaced(&team, instance->announced_cpus);
    return cpu_team_close(&team);
}

/* `perf stat --control fifo:<first>,<second>` counts between "enable" and "disable" written to the first fifo
 * and answers on the second; without --perf-control this does nothing. What happened is printed, and a command
 * that perf did not take or did not answer ends the run: its counters would not be the rounds'. */
static void perf_control(const char *command)
{
    char control_path[PATH_MAX], answer[CPU_STAMP_TEXT_BYTES];
    if (run.perf_control == NULL) {
        return;
    }
    const char *comma = strchr(run.perf_control, ',');
    const size_t length = comma != NULL ? (size_t)(comma - run.perf_control) : sizeof(control_path);
    if (length >= sizeof(control_path)) {
        fail("--perf-control needs two fifos, and the first's path no longer than a path may be");
    }
    memcpy(control_path, run.perf_control, length);
    control_path[length] = '\0';
    const int control = open(control_path, O_WRONLY | O_NONBLOCK | O_CLOEXEC);
    const int answers = open(comma + 1, O_RDONLY | O_NONBLOCK | O_CLOEXEC);
    struct pollfd waiting = {.fd = answers, .events = POLLIN, .revents = 0};
    const int sent = control >= 0 && answers >= 0 && dprintf(control, "%s\n", command) > 0;
    const int answered = sent && poll(&waiting, 1, ACKNOWLEDGE_MILLISECONDS) == 1 &&
                         read(answers, answer, sizeof(answer)) > 0;
    printf("PERF   command=%s sent=%d acknowledged=%d\n", command, sent, answered);
    if (!answered) {                             /* the run ends here, and its descriptors with it */
        fail("perf took or answered no control command: this run's counters are not its rounds'");
    }
    (void)close(control);                        /* both are open: the command was sent and answered */
    (void)close(answers);
}

/* The modules' temperatures between two rounds, into the ROUND line and into the run's hottest: the instances are
 * interleaved, so every one of them saw every round. */
static void print_round(int round)
{
    struct cpu_module_temperatures modules;
    char stamp[CPU_STAMP_TEXT_BYTES];
    cpu_wall_clock_stamp(stamp, sizeof(stamp));
    cpu_read_module_temperatures(&modules);
    printf("ROUND  card=host round=%d stamp=%s", round, stamp);
    if (modules.module_count > 0) {
        printf(" dimm_c=%d/%d\n", modules.lowest_c, modules.highest_c);
    } else {
        printf(" dimm_c=na\n");
    }
    run.hottest_module_c = modules.module_count > 0 && modules.highest_c > run.hottest_module_c
                               ? modules.highest_c : run.hottest_module_c;
}

/* The warm-up (the whole sequence untimed until a second has passed), the step count that keeps the memory's
 * load inside its budget, then the rounds. */
static void run_rounds(void)
{
    double warm_up_microseconds[WARM_UP_STEPS], sequence_microseconds = 0.0, lead_in_microseconds = 0.0;
    const double started = cpu_seconds_now();
    do {
        for (int index = 0; index < run.instance_count; index++) {
            struct instance *instance = &run.instances[index];
            if (!run_slot(instance, WARM_UP_STEPS, warm_up_microseconds, -1)) {
                fail("a team failed in the warm-up");
            }
            instance->estimated_microseconds = cpu_quantile(warm_up_microseconds, WARM_UP_STEPS, CPU_MEDIAN);
        }
    } while (cpu_seconds_now() - started < WARM_UP_SECONDS);
    for (int index = 0; index <= run.instance_count; index++) {              /* the baseline twice */
        const double step = run.instances[index % run.instance_count].estimated_microseconds;
        const double fewest_steps = step * (double)LEAD_IN_STEPS_MIN, asked_time = (double)run.lead_in_microseconds;
        sequence_microseconds += step;
        lead_in_microseconds += fewest_steps > asked_time ? fewest_steps : asked_time;
    }
    const double budget = (double)run.load_seconds * CPU_MICROSECONDS_PER_SECOND / (double)run.rounds;
    const int asked = run.steps_per_round;
    if (lead_in_microseconds + sequence_microseconds * asked > budget) {     /* fewer steps than asked, never more */
        const double fitting = (budget - lead_in_microseconds) / sequence_microseconds;
        const int fewest = asked < STEPS_PER_ROUND_MIN ? asked : STEPS_PER_ROUND_MIN;
        run.steps_per_round = fitting > (double)fewest ? (int)fitting : fewest;
    }
    printf("LOAD   card=host budget_s=%d rounds=%d steps_per_round=%d asked=%d projected_s=%.1f\n", run.load_seconds,
           run.rounds, run.steps_per_round, asked, (lead_in_microseconds + sequence_microseconds *
           run.steps_per_round) * (double)run.rounds / CPU_MICROSECONDS_PER_SECOND);
    /* The baseline's slots are the round's first and one mid-round: each then follows another instance's team, as
     * every other slot does, and not its own. Which is "again" alternates: where two do follow each other (one or
     * two instances), neither name keeps the place behind its own team, whose cores were busy a moment before. */
    const int middle_slot = run.instance_count / BASELINE_SLOTS + 1;
    perf_control("enable");
    for (int round = 0; round < run.rounds; round++) {
        const int first_step = round * run.steps_per_round;
        print_round(round);
        for (int slot = 0; slot <= run.instance_count; slot++) {
            const int is_baseline = slot == 0 || slot == middle_slot;
            struct instance *instance = &run.instances[is_baseline ? 0 : slot - (slot > middle_slot)];
            const int is_again = is_baseline && (slot == middle_slot) != (round % BASELINE_SLOTS == 1);
            double *microseconds = (is_again ? instance->again_microseconds : instance->step_microseconds) +
                                   first_step;
            if (!run_slot(instance, run.steps_per_round, microseconds, is_again ? -1 : first_step)) {
                fail("a team failed in a timed slot");
            }
        }
    }
    perf_control("disable");
    print_round(run.rounds);
}

/* ---- results ---- */

static void print_timeline(const struct instance *instance, const struct cpu_summary *summary, double *scratch)
{
    const int steps = run.rounds * run.steps_per_round, head_steps = run.steps_per_round / SLOT_HEAD_SHARE + 1;
    const struct instance *one_thread = NULL;
    int head_count = 0, rest_count = 0;
    for (int index = 0; index < run.instance_count; index++) {      /* the same kernel on one thread, and alone */
        const struct instance *other = &run.instances[index];
        if (other->kernel == instance->kernel && other->thread_count == 1 && other->experts_at_once == 1 &&
            other->pool != CPU_POOL_DOCKED) {
            one_thread = one_thread == NULL ? other : one_thread;
        }
    }
    printf(LINE_PREFIX "TIMELINE card=host case=experts:%d variant=%s n=%d step_us=%.1f", instance->experts_at_once,
           instance->variant, steps, summary->median);
    if (one_thread != NULL) {
        memcpy(scratch, one_thread->terms[TERM_WORK], (size_t)steps * sizeof(*scratch));
        const double shared_out = cpu_quantile(scratch, steps, CPU_MEDIAN) * instance->experts_at_once /
                                  instance->thread_count;
        memcpy(scratch, instance->terms[TERM_ACTIVATION], (size_t)steps * sizeof(*scratch));
        printf(" shared_out_us=%.1f neither_us=%.1f", shared_out,
               summary->median - shared_out - cpu_quantile(scratch, steps, CPU_MEDIAN));
    } else {
        printf(" shared_out_us=na neither_us=na");
    }
    for (int kind = 0; kind < TERM_COUNT; kind++) {
        memcpy(scratch, instance->terms[kind], (size_t)steps * sizeof(*scratch));
        printf(" %s=%.2f", term_names[kind], cpu_quantile(scratch, steps, CPU_MEDIAN));
    }
    for (int step = 0; step < steps; step++) {       /* a slot's first steps against its others: the lead-in's proof */
        if (step % run.steps_per_round < head_steps) {
            scratch[head_count++] = instance->step_microseconds[step];
        } else {
            scratch[steps - ++rest_count] = instance->step_microseconds[step];
        }
    }
    const double head = cpu_quantile(scratch, head_count, CPU_MEDIAN);
    printf(" slot_head_us=%.1f", head);
    if (rest_count > 0) {
        printf(" slot_rest_us=%.1f", cpu_quantile(scratch + head_count, rest_count, CPU_MEDIAN));
    } else {
        printf(" slot_rest_us=na");
    }
    printf(" switches_per_step=%.3f misplaced=%ld backwards=%ld\n", (double)instance->switches / steps,
           instance->misplaced, instance->backwards);
}

static void print_results(void)
{
    static struct cpu_summary baseline, again, summary;
    const int steps = run.rounds * run.steps_per_round;
    const int same_sign_needed = (run.rounds * SAME_SIGN_PERCENT_MIN + PERCENT - 1) / PERCENT;
    double *scratch = malloc((size_t)steps * sizeof(*scratch)), differences[ROUNDS_MAX];
    struct instance *first = &run.instances[0];
    if (scratch == NULL) {
        fail("no memory for the statistics");
    }
    cpu_summarize(first->step_microseconds, run.rounds, run.steps_per_round, scratch, &baseline);
    cpu_summarize(first->again_microseconds, run.rounds, run.steps_per_round, scratch, &again);
    const double noise = again.median > baseline.median ? again.median / baseline.median
                                                        : baseline.median / again.median;
    printf(LINE_PREFIX "NOISE  card=host case=experts:%d baseline_us=%.1f baseline_again_us=%.1f difference_us=%.1f "
           "ratio=%.3f\n", first->experts_at_once, baseline.median, again.median, again.median - baseline.median,
           noise);
    for (int index = 0; index < run.instance_count; index++) {
        struct instance *instance = &run.instances[index];
        const double megabytes = (double)cpu_expert_bytes(&run.ring.experts[0], NULL) * instance->experts_at_once /
                                 CPU_BYTES_PER_MB;
        int same_sign = 0;
        cpu_summarize(instance->step_microseconds, run.rounds, run.steps_per_round, scratch, &summary);
        instance->expert_microseconds = summary.median / instance->experts_at_once;
        for (int round = 0; round < run.rounds; round++) {
            differences[round] = summary.round_median[round] / instance->experts_at_once -
                                 baseline.round_median[round] / first->experts_at_once;
            scratch[round] = differences[round];
        }
        const double paired = cpu_quantile(scratch, run.rounds, CPU_MEDIAN);
        for (int round = 0; round < run.rounds; round++) {
            same_sign += (differences[round] > 0.0) == (paired > 0.0) && differences[round] != 0.0;
        }
        const double ratio = instance->expert_microseconds / (baseline.median / first->experts_at_once);
        const int differs = same_sign >= same_sign_needed && (ratio > noise || ratio < 1.0 / noise);
        instance->checks_held &= instance->wrong_sums == 0 && instance->misplaced == 0 &&
                                 instance->refused_shares == 0;
        run.checks_as_required &= instance->checks_held;
        printf(LINE_PREFIX "RESULT batch=%s probe=cpu_floor build=" BUILD_NAME " flags=exact arch=znver4 card=host "
               "format=native part=expert rows=1 experts=%d overlap=0 variant=%s class=none mode=%s n=%d "
               "median_us=%.1f p99_us=%.1f min_us=%.1f mean_us=%.1f expert_us=%.1f ratio=%.3f paired_us=%.1f "
               "same_sign=%d/%d differs=%s stream_us=na mb=%.1f gbps=%.1f registers=na spill_bytes=na kcyc=na "
               "clock_mhz=na power_w=na uj=na wrong_sums=%ld misplaced=%ld switches_per_step=%.3f bystanders=%d "
               "busy_threads=%.1f checks=%s dimm=%s\n", run.batch_name, instance->experts_at_once, instance->variant,
               run.profiled || run.perf_control != NULL ? "profile" : "step", steps, summary.median, summary.high,
               summary.lowest, summary.mean, instance->expert_microseconds, ratio, paired, same_sign, run.rounds,
               index == 0 ? "baseline" : (differs ? "yes" : "undecided"), megabytes,
               megabytes * CPU_BYTES_PER_MB / BYTES_PER_GB / (summary.median / CPU_MICROSECONDS_PER_SECOND),
               instance->wrong_sums, instance->misplaced, (double)instance->switches / steps, instance->bystanders,
               instance->busy_seconds / instance->timed_seconds, instance->checks_held ? "ok" : "FAILED",
               run.hottest_module_c == 0 ? "na" : (run.hottest_module_c >= HOT_MODULE_C ? "hot" : "ok"));
        if (run.timeline) {
            print_timeline(instance, &summary, scratch);
        }
    }
    for (int kernel = 0; kernel < KERNEL_KINDS; kernel++) {      /* per kernel: the least time per expert */
        const struct instance *best = NULL;
        for (int index = 0; index < run.instance_count; index++) {
            const struct instance *instance = &run.instances[index];
            if (instance->kernel == kernel && instance->checks_held &&
                (best == NULL || instance->expert_microseconds < best->expert_microseconds)) {
                best = instance;
            }
        }
        if (best != NULL) {
            printf(LINE_PREFIX "BEST   card=host case=kernel:%s,experts:%d variant=%s\n", kernel_names[kernel],
                   best->experts_at_once, best->variant);
            printf(LINE_PREFIX "VERDICT card=host figure=%s bar_ratio=na bar_us=na expert_us=%.1f gbps=%.1f "
                   "noise_ratio=%.3f verdict=measured\n", kernel == KERNEL_FP32 ? "fp32_expert_floor"
                   : "read_ceiling", best->expert_microseconds, (double)cpu_expert_bytes(&run.ring.experts[0], NULL) /
                   BYTES_PER_GB / (best->expert_microseconds / CPU_MICROSECONDS_PER_SECOND), noise);
        }
    }
    printf(LINE_PREFIX "VERDICT card=host figure=floor_attribution timeline_lines=%d verdict=%s\n",
           run.timeline ? run.instance_count : 0, run.timeline ? "measured" : "not_asked");
    printf(LINE_PREFIX "VERDICT card=host figure=experts_at_once counts_of_experts=%d verdict=%s\n", run.experts_cases,
           run.experts_cases > 1 ? "measured" : "not_asked");
    free(scratch);
}

/* ---- the run ---- */

/* One instance of the lists' cross product: its name, its processors, each thread's rows. The threads of expert
 * slot s are the next thread_count / experts (one more for the first few); each takes its rows as b04's loops gave
 * them. Returns 0 when the instance cannot be: more experts than threads, more threads than processors, or a share
 * outside its matrix (CHECK shares_inside; a run that breaks one on purpose goes on to its end, which is at once). */
static int plan_instance(struct instance *instance)
{
    int order[CPU_MAX_HARDWARE_THREADS], refused = 0;
    char detail[DETAIL_BYTES];
    const int threads = instance->thread_count, experts = instance->experts_at_once;
    if (experts > threads || threads + run.break_pinning > run.topology.thread_count) {
        return 0;
    }
    (void)snprintf(instance->variant, sizeof(instance->variant), "kernel:%s,threads:%d,placement:%s,pool:%s",
                   kernel_names[instance->kernel], threads, cpu_placement_names[instance->placement],
                   cpu_pool_names[instance->pool]);
    cpu_placement_order(&run.topology, (enum cpu_placement)instance->placement, order);
    printf("PLACEMENT card=host variant=%s cpus=", instance->variant);
    for (int thread = 0; thread < threads; thread++) {
        instance->announced_cpus[thread] = order[thread];
        instance->pinned_cpus[thread] = order[thread + run.break_pinning];
        printf("%d%s", order[thread], thread + 1 < threads ? "," : "\n");
    }
    for (int slot = 0; slot < experts; slot++) {
        const int first_thread = slot * threads / experts, workers = (slot + 1) * threads / experts - first_thread;
        for (int worker = 0; worker < workers; worker++) {
            struct row_share *gate_up = &instance->share[REGION_GATE_UP][first_thread + worker];
            struct row_share *down = &instance->share[REGION_DOWN][first_thread + worker];
            gate_up->expert_slot = slot;
            down->expert_slot = slot;
            cpu_static_share(GATE_UP_ROWS, workers, worker, &gate_up->first_row, &gate_up->row_end);
            cpu_static_share(CPU_HIDDEN_SIZE, workers, worker, &down->first_row, &down->row_end);
        }
    }
    instance->checks_held = 1;
    instance->share[REGION_DOWN][threads - 1].row_end += run.break_share;    /* a row past the down matrix */
    for (int thread = 0; thread < threads; thread++) {
        refused += !share_is_inside(instance, REGION_GATE_UP, thread) + !share_is_inside(instance, REGION_DOWN, thread);
    }
    (void)snprintf(detail, sizeof(detail), "refused=%d of=%d", refused, REGIONS_PER_STEP * threads);
    report_check(instance, "shares_inside", detail, refused != 0, run.break_share);
    return refused == 0 || run.break_share;
}

/* The command line into `run`. Returns 0 for anything it does not understand or will not run. */
static int parse_arguments(int argument_count, char **arguments)
{
    int right = 1;
    for (int index = 1; index < argument_count && right; index++) {
        const char *argument = arguments[index], *value = index + 1 < argument_count ? arguments[index + 1] : "";
        int *flag = strcmp(argument, "--timeline") == 0 ? &run.timeline
                    : strcmp(argument, "--profiled") == 0 ? &run.profiled
                    : strcmp(argument, "--corrupt") == 0 ? &run.corrupt
                    : strcmp(argument, "--break-bits") == 0 ? &run.break_bits
                    : strcmp(argument, "--break-share") == 0 ? &run.break_share
                    : strcmp(argument, "--break-pinning") == 0 ? &run.break_pinning : NULL;
        if (flag != NULL) {
            *flag = 1;
            continue;
        }
        index++;                                                             /* every other option has a value */
        if (strcmp(argument, "--kernel") == 0) {
            run.kernel_count = cpu_parse_name_list(value, kernel_names, KERNEL_KINDS, run.kernels, MAX_LIST);
        } else if (strcmp(argument, "--threads") == 0) {
            run.thread_cases = cpu_parse_number_list(value, 1, MAX_THREADS, run.threads, MAX_LIST);
        } else if (strcmp(argument, "--placement") == 0) {
            run.placement_count = cpu_parse_name_list(value, cpu_placement_names, CPU_PLACEMENT_KINDS,
                                                      run.placements, MAX_LIST);
        } else if (strcmp(argument, "--pool") == 0) {
            run.pool_count = cpu_parse_name_list(value, cpu_pool_names, CPU_POOL_KINDS, run.pools, MAX_LIST);
        } else if (strcmp(argument, "--experts") == 0) {
            run.experts_cases = cpu_parse_number_list(value, 1, MAX_EXPERTS_AT_ONCE, run.experts, MAX_LIST);
        } else if (strcmp(argument, "--rounds") == 0) {
            right = cpu_parse_number(value, 1, ROUNDS_MAX, &run.rounds);
        } else if (strcmp(argument, "--steps-per-round") == 0) {
            right = cpu_parse_number(value, 1, STEPS_PER_ROUND_MAX, &run.steps_per_round);
        } else if (strcmp(argument, "--lead-in-us") == 0) {
            right = cpu_parse_number(value, 0, LEAD_IN_MICROSECONDS_MAX, &run.lead_in_microseconds);
        } else if (strcmp(argument, "--load-seconds") == 0) {
            right = cpu_parse_number(value, 1, LOAD_SECONDS_MAX, &run.load_seconds);
        } else if (strcmp(argument, "--ring-experts") == 0) {
            right = cpu_parse_number(value, RING_EXPERTS_MIN, RING_EXPERTS_MAX, &run.ring_experts);
        } else if (strcmp(argument, "--batch") == 0 && value[0] != '\0') {
            run.batch_name = value;
        } else if (strcmp(argument, "--perf-control") == 0 && strchr(value, ',') != NULL) {
            run.perf_control = value;
        } else {
            right = 0;
        }
    }
    return right && run.kernel_count > 0 && run.thread_cases > 0 && run.placement_count > 0 && run.pool_count > 0 &&
           run.experts_cases > 0 && run.ring_experts % RING_STRIDE != 0 &&
           run.corrupt + run.break_bits + run.break_pinning + run.break_share <= 1;
}

int main(int argument_count, char **arguments)
{
    int first_of_kernel[KERNEL_KINDS] = {-1, -1}, most_experts = 0, fp32_instances = 0;
    run.kernels[0] = KERNEL_FP32;
    run.threads[0] = 1;
    run.placements[0] = CPU_PLACEMENT_DIE;
    run.pools[0] = CPU_POOL_OPENMP;
    run.experts[0] = 1;
    run.kernel_count = run.thread_cases = run.placement_count = run.pool_count = run.experts_cases = 1;
    run.rounds = DEFAULT_ROUNDS;
    run.steps_per_round = DEFAULT_STEPS_PER_ROUND;
    run.lead_in_microseconds = DEFAULT_LEAD_IN_MICROSECONDS;
    run.load_seconds = DEFAULT_LOAD_SECONDS;
    run.ring_experts = DEFAULT_RING_EXPERTS;
    run.batch_name = "none";
    run.checks_as_required = 1;
    if (!parse_arguments(argument_count, arguments)) {
        fprintf(stderr, "usage: %s [--kernel fp32,read] [--threads 1,..] [--placement die,packed,smt] "
                        "[--pool openmp,own,docked] [--experts 1,..,%d] [--timeline] [--rounds N] "
                        "[--steps-per-round N] [--lead-in-us N] [--load-seconds N] [--ring-experts N, no multiple "
                        "of %d] [--batch NAME] [--perf-control FIFO,FIFO] [--profiled] [one of --corrupt "
                        "--break-bits --break-pinning --break-share]\n",
                arguments[0], MAX_EXPERTS_AT_ONCE, RING_STRIDE);
        return EXIT_FAILURE;
    }
    if (!__builtin_cpu_supports("avx512bw") || !__builtin_cpu_supports("avx512vl")) {
        fail("this probe needs AVX-512 with BW and VL");
    }
    omp_set_dynamic(0);                              /* a team has the threads it was asked for, or the run ends */
    run.ticks_per_microsecond = cpu_ticks_per_microsecond();
    if (!cpu_topology_read(&run.topology)) {
        fail("the processors' topology could not be read from sysfs");
    }
    const int cases = run.kernel_count * run.experts_cases * run.placement_count * run.pool_count * run.thread_cases;
    for (int index = 0; index < cases; index++) {    /* the lists' cross product: the thread counts turn fastest */
        struct instance *instance = &run.instances[run.instance_count];
        int digits = index;
        if (run.instance_count == MAX_INSTANCES) {
            fail("more instances than one run holds: shorten a list");
        }
        instance->thread_count = run.threads[digits % run.thread_cases];
        digits /= run.thread_cases;
        instance->pool = run.pools[digits % run.pool_count];
        digits /= run.pool_count;
        instance->placement = run.placements[digits % run.placement_count];
        digits /= run.placement_count;
        instance->experts_at_once = run.experts[digits % run.experts_cases];
        instance->kernel = run.kernels[digits / run.experts_cases];
        const int is_first = first_of_kernel[instance->kernel] < 0;
        most_experts = instance->experts_at_once > most_experts ? instance->experts_at_once : most_experts;
        instance->route_weight = run.break_bits && !is_first ? ROUTE_WEIGHT + ROUTE_WEIGHT : ROUTE_WEIGHT;
        fp32_instances += instance->kernel == KERNEL_FP32;
        if (instance->pool != CPU_POOL_OWN && !CPU_OPENMP_HELPS) {
            fail("the thread sanitizer cannot see the OpenMP runtime's barriers: --pool own only");
        }
        if (instance->pool == CPU_POOL_DOCKED && (instance->thread_count != 1 || omp_get_num_places() == 0)) {
            fail("--pool docked needs --threads 1 and OMP_PLACES, which binds the docked threads");
        }
        if (!plan_instance(instance)) {
            fail("more experts than threads, more threads than processors, or a share refused");
        }
        first_of_kernel[instance->kernel] = is_first ? run.instance_count : first_of_kernel[instance->kernel];
        run.instance_count++;
    }
    if (run.break_share) {                       /* reported when planned: no step runs on a share that was refused */
        return end_of_broken_run();
    }
    if (run.break_bits && (fp32_instances < COMPARED_INSTANCES_MIN || run.experts_cases != 1)) {
        fail("--break-bits needs two instances of the fp32 kernel and one --experts value");
    }
    if (!cpu_ring_create(&run.ring, run.ring_experts)) {
        fail("no memory for the ring, or the kernel refused the page-size advice");
    }
    const double expert_bytes = (double)cpu_expert_bytes(&run.ring.experts[0], NULL);
    const double ring_mib = expert_bytes * run.ring_experts / CPU_BYTES_PER_MIB;
    const double cache_mib = (double)run.topology.slice_kib * run.topology.slice_count / CPU_KIB_PER_MIB;
    char cache_text[DETAIL_BYTES] = "l3_mib=na ring_l3=na";                  /* where sysfs does not give the size */
    if (cache_mib > 0.0) {
        (void)snprintf(cache_text, sizeof(cache_text), "l3_mib=%.0f ring_l3=%.1f", cache_mib, ring_mib / cache_mib);
    }
    printf("HOST   card=host build=" BUILD_NAME " cpus=%d cores=%d l3_slices=%d %s ring_experts=%d ring_mib=%.0f "
           "expert_mb=%.2f tsc_mhz=%.1f\n", run.topology.thread_count, run.topology.core_count,
           run.topology.slice_count, cache_text, run.ring_experts, ring_mib, expert_bytes / CPU_BYTES_PER_MB,
           run.ticks_per_microsecond);
    for (int column = 0; column < CPU_HIDDEN_SIZE; column++) {
        run.input[column] = ((float)(cpu_mix_bits((uint64_t)column + INPUT_SEED) % INPUT_VALUE_STEPS) -
                             INPUT_VALUE_OFFSET) / INPUT_VALUE_DIVISOR;
    }
    for (int region = 0; region < REGIONS_PER_STEP; region++) {
        run.records[region].threads = run.stamps[region];
    }
    for (int slot = 0; first_of_kernel[KERNEL_FP32] >= 0 && slot < most_experts; slot++) {
        reference_outputs(FIRST_CHECKED_EXPERT + slot, ROUTE_WEIGHT, run.reference[0][slot]);
        if (run.break_bits) {
            reference_outputs(FIRST_CHECKED_EXPERT + slot, ROUTE_WEIGHT + ROUTE_WEIGHT, run.reference[1][slot]);
        }
    }
    if (first_of_kernel[KERNEL_FP32] >= 0 && !cpu_openmp_release_threads()) {
        fail("the OpenMP runtime would not end its threads");
    }
    if (run.corrupt) {                           /* after the reference and the byte sum of this expert were taken */
        memset(run.ring.experts[FIRST_CHECKED_EXPERT].matrix[CPU_GATE].nibbles, CORRUPTED_BYTE_VALUE, CORRUPTED_BYTES);
    }
    for (int index = 0; index < run.instance_count; index++) {
        check_instance(&run.instances[index], index == first_of_kernel[run.instances[index].kernel]);
    }
    if (run.corrupt || run.break_bits || run.break_pinning) {
        return end_of_broken_run();
    }
    if (cache_mib > 0.0 && ring_mib < cache_mib * (double)CACHES_OF_RING_MIN && !PROBE_DIAGNOSTICS) {
        fail("the ring is under four times the third-level cache: its experts would not be cold");
    }
    const int timed_steps = run.rounds * run.steps_per_round;                /* the round's steps can only shrink */
    for (int index = 0; index < run.instance_count; index++) {
        struct instance *instance = &run.instances[index];
        instance->step_microseconds = calloc((size_t)timed_steps, sizeof(double));
        instance->again_microseconds = index == 0 ? calloc((size_t)timed_steps, sizeof(double)) : NULL;
        int missing = instance->step_microseconds == NULL || (index == 0 && instance->again_microseconds == NULL);
        for (int kind = 0; kind < TERM_COUNT; kind++) {
            instance->terms[kind] = run.timeline ? calloc((size_t)timed_steps, sizeof(double)) : NULL;
            missing |= run.timeline && instance->terms[kind] == NULL;
        }
        if (missing) {
            fail("no memory for the step times or the timeline");
        }
    }
    run_rounds();
    print_results();
#if PROBE_DIAGNOSTICS
    for (int index = 0; index < run.instance_count; index++) {      /* the counters: bytes_read is to equal expected */
        const struct instance *instance = &run.instances[index];
        printf("DIAG   card=host variant=%s bytes_read=%llu expected=%llu refused_shares=%llu\n", instance->variant,
               (unsigned long long)instance->bytes_read, (unsigned long long)(cpu_expert_bytes(&run.ring.experts[0],
               NULL) * (uint64_t)(instance->experts_at_once * run.rounds * run.steps_per_round)),
               (unsigned long long)instance->refused_shares);
    }
#endif
    printf("%s\n", run.checks_as_required ? "every check held" : "FAILED: a check did not hold");
    return run.checks_as_required ? EXIT_SUCCESS : EXIT_FAILURE;
}
