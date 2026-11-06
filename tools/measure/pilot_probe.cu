/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * pilot_probe: package 0 of round 1 of the speed campaign (research/speed/ROUND_1.md). It times nothing of the
 * model: it runs the `work` kernel of card_work_kernels.cuh, one launch with the instruction mix of the 4-bit
 * expert kernel reading 112.8 MB of a cold ring, through probe_harness.h, so that the noise figures the round's
 * conventions rest on are numbers and the two shared files have run once before eight packages include them.
 *
 * Instances, interleaved in one run: baseline (work_kernel, no launch bound), baseline_again (the same kernel: the
 * A against A noise floor), bounds:256x1 and bounds:256x5 (the same body under __launch_bounds__(256, 1) and
 * (256, 5): what stating a bound does to a kernel that had none). Cases: --rows 1 .. 8, the columns of operand B
 * in use (a column past them repeats row 0). A step is one launch of 1,296 blocks; step s reads ring step
 * s % --ring-steps, so that a launch never reads what the launches before it left in the card's L2.
 *
 * What a run prints: the lines of probe_harness.h. Of the pilot's own: VERDICT figure=noise_floor_at_<case> (the
 * ratio of baseline_again to baseline) and figure=launch_bound_<bound>_at_<case> (each bound against no bound), all
 * verdict=measured: the pilot has no bar. The scatter of the round medians and the fit of step time against the
 * reported clock are the batch's (round_join_clocks), from this run's ROUND lines.
 *
 * Checked before anything is timed, per case and kernel, on step 0, each launch validated as its own instance:
 *   host_reference_checksums  every checksum of every block (what each lane read of the ring, folded) against the
 *                             host
 *   host_reference_sums       the eight sums of every lane of every 37th block against the host, bit for bit: the
 *                             sums are exact by construction (card_work_kernels.cuh says why)
 *   bits_against_baseline     every output word of a bounded kernel against the baseline's: 0 may differ
 *   kernel_diagnostics        debug build: the kernels' own bound checks stored nothing, and every thread counted
 *                             the loads of a warp
 *   guard_bytes               nothing wrote behind a card array
 * and the DIGEST of all outputs, which the batch compares between the two CMP 170HX.
 *
 * Each check is shown failing by a flag of its own; in such a run every other check must still hold:
 *   --corrupt             a megabyte of the checked step's weights is overwritten on the card before the launches:
 *                         host_reference_checksums must report (and the sums, whose block 0 lies in it)
 *   --break-inputs        the activation codes are overwritten on the card: host_reference_sums must report and the
 *                         checksums, which fold no activation code, must not
 *   --break-bits          the same megabyte is overwritten after the baseline's checked launch: bits_against_baseline
 *                         must report for both bounded kernels, the host reference must not
 *   --break-guard         probe_harness.h: guard_bytes must report
 *   --break-diagnostics   debug build only (the flag does not exist elsewhere): one more launch of the baseline's
 *                         kernel with a record that says the launch has no block; the kernel must store its word
 *                         and return before it reads anything: kernel_diagnostics must report
 *   --break-route         seven launches, each past one bound of what the kernel indexes by or of what a card
 *                         allows: each must be refused on the host (nothing is launched)
 * What a break flag overwrites is weights and activation codes: values, never anything a kernel indexes by.
 *
 *   pilot_probe [--rows 1,8] [--ring-steps N] and the flags of probe_harness.h
 *
 * Measurement tool: not part of the engine.
 */
#include "probe_harness.h"
#include "card_work_kernels.cuh"

enum {
    PILOT_BOUND_ONE_BLOCK = 2, PILOT_BOUND_FIVE_BLOCKS = 3,      /* the instances after the baseline pair */
    PILOT_CHECKED_STEP = 0,
    PILOT_CHECKED_BLOCK_STRIDE = 37,             /* the blocks whose sums the host forms: 36 of 1,296, block 0 first */
    PILOT_CORRUPTED_BYTES = 1024 * 1024, PILOT_CORRUPTED_BYTE_VALUE = 0x5A,
    PILOT_STALE_BYTE = 0xFF,                     /* the outputs before a checked launch: a NaN, and no checksum */
    PILOT_FILL_THREADS = 256, PILOT_FILL_BLOCKS_PER_SM = 8,
    PILOT_LEAST_RING_STEPS = 2, PILOT_MOST_RING_STEPS = 64,
    PILOT_COLD_L2_SIZES = 4,                     /* CONVENTIONS 3.5: bytes read between two reads of the same bytes */
    PILOT_STEP_SUMS = WORK_STEP_BLOCKS * WORK_SUMS_PER_BLOCK,     /* a checked launch's outputs, in words */
    PILOT_STEP_CHECKSUMS = WORK_STEP_BLOCKS * WORK_LANES_PER_WARP,
    PILOT_STEP_OUTPUT_WORDS = PILOT_STEP_SUMS + PILOT_STEP_CHECKSUMS,
    PILOT_STEP_THREADS = WORK_STEP_BLOCKS * WORK_BLOCK_THREADS,
    PILOT_FIGURE_BYTES = 256,                    /* a figure's name: a few words and the case's name */
    /* --break-route, in turn: what is put past its bound */
    PILOT_BROKEN_RING_END = 1, PILOT_BROKEN_RING_WRAP, PILOT_BROKEN_ROWS, PILOT_BROKEN_LOADS, PILOT_BROKEN_BLOCKS,
    PILOT_BROKEN_BLOCK_THREADS, PILOT_BROKEN_ARRAY, PILOT_BROKEN_END
};
#define PILOT_FLAG_ROWS "--rows"
#define PILOT_FLAG_RING_STEPS "--ring-steps"
#define PILOT_FLAG_BREAK_INPUTS "--break-inputs"
#define PILOT_FLAG_BREAK_BITS "--break-bits"
#define PILOT_FLAG_BREAK_DIAGNOSTICS "--break-diagnostics"
#define PILOT_REFUSED_BY_THE_KERNEL "baseline_given_no_blocks"     /* variant= of the launch --break-diagnostics adds */
#define PILOT_STEP_BYTES ((uint64_t)WORK_STEP_WARP_LOADS * WORK_BYTES_PER_WARP_LOAD)
#define PILOT_BYTES_PER_MB 1.0e6

static struct probe_flag pilot_flags[] = {
    PROBE_COMMON_FLAGS,
    PROBE_CASE_NUMBERS(PILOT_FLAG_ROWS, 1, WORK_MOST_ROWS, "1", "rows of a step: the columns of operand B in use"),
    PROBE_NUMBER(PILOT_FLAG_RING_STEPS, PILOT_LEAST_RING_STEPS, PILOT_MOST_RING_STEPS, "16",
                 "steps of weights the cold ring holds"),
    PROBE_SWITCH(PILOT_FLAG_BREAK_INPUTS, "overwrite the activation codes on the card: only the sums must report"),
    PROBE_SWITCH(PILOT_FLAG_BREAK_BITS, "overwrite weights after the baseline's launch: the bounded kernels differ"),
#if PROBE_DIAGNOSTICS
    PROBE_SWITCH(PILOT_FLAG_BREAK_DIAGNOSTICS, "launch with a record of no blocks: the kernel's own check must report"),
#endif
};
static struct probe_instance pilot_instances[] = {
    {"baseline", "work_kernel", 'C', 1, WORK_BLOCK_THREADS, 0, 0, 0, 0},
    {"baseline_again", "work_kernel", 'C', 1, WORK_BLOCK_THREADS, 0, 0, 0, 0},
    {"bounds:256x1", "work_kernel_bound_256x1", 'C', 1, WORK_BLOCK_THREADS, WORK_BLOCK_THREADS, 1, 0, 0},
    {"bounds:256x5", "work_kernel_bound_256x5", 'C', 1, WORK_BLOCK_THREADS, WORK_BLOCK_THREADS, WORK_BOUND_FIVE_BLOCKS,
     0, 0},
};
static const char *const pilot_broken_names[PILOT_BROKEN_END] = {
    "", "refused_ring_past_its_end", "refused_ring_place_that_wraps", "refused_row_too_many", "refused_load_too_many",
    "refused_block_too_many", "refused_block_of_other_threads", "refused_output_that_is_no_card_array"};

static struct {
    uint4 *ring_packed;                          /* the card's arrays: struct work_arguments says what each holds */
    uint32_t *ring_scales;
    uint4 *input_words;
    uint16_t *input_exponents;
    float *sums;
    uint32_t *checksums;
    uint64_t ring_steps;
    struct work_arguments arguments[PROBE_STREAM_STEPS];        /* of the step prepared in each slot */
    uint32_t host_input_words[WORK_INPUT_WORDS];                /* the host's copies of what was uploaded */
    uint16_t host_input_exponents[WORK_INPUT_EXPONENTS];
#if PROBE_DIAGNOSTICS
    struct work_diagnostics diagnostics;
#endif
} pilot;
/* a checked launch's outputs on the host: every block's sums, then every block's checksums */
static uint32_t pilot_outputs[PILOT_STEP_OUTPUT_WORDS], pilot_baseline_outputs[PILOT_STEP_OUTPUT_WORDS];

/* ---- launches: built, validated, launched ---- */

/* The arguments of step `step` of a case: its place in the ring, the model's loads, the case's rows. */
static void pilot_step_arguments(const struct probe_case *probe_case, uint64_t step, struct work_arguments *arguments)
{
    arguments->ring_packed = pilot.ring_packed;
    arguments->ring_scales = pilot.ring_scales;
    arguments->input_words = pilot.input_words;
    arguments->input_exponents = pilot.input_exponents;
    arguments->sums = pilot.sums;
    arguments->checksums = pilot.checksums;
    arguments->first_warp_load = step % pilot.ring_steps * WORK_STEP_WARP_LOADS;
    arguments->loads_per_warp = WORK_MOST_LOADS_PER_WARP;
    arguments->row_count = (uint32_t)probe_case_value(probe_case, PILOT_FLAG_ROWS);
}

/* Check 10: a launch against what its pointers were allocated with (0 bytes for a pointer that is no card array). */
static int pilot_launch_is_valid(uint32_t instance, const struct work_arguments *arguments, uint64_t block_count,
                                 uint64_t block_threads)
{
    struct work_array_bytes bytes;
    bytes.ring_packed = probe_cuda_array_bytes(arguments->ring_packed);
    bytes.ring_scales = probe_cuda_array_bytes(arguments->ring_scales);
    bytes.input_words = probe_cuda_array_bytes(arguments->input_words);
    bytes.input_exponents = probe_cuda_array_bytes(arguments->input_exponents);
    bytes.sums = probe_cuda_array_bytes(arguments->sums);
    bytes.checksums = probe_cuda_array_bytes(arguments->checksums);
    return work_launch_is_valid(&pilot_instances[instance], arguments, &bytes, block_count, block_threads);
}

/* One step of an instance: one launch of its kernel, with the block and grid that were validated. */
static void pilot_launch(uint32_t instance, const struct work_arguments *arguments)
{
#if PROBE_DIAGNOSTICS
    struct work_diagnostics diagnostics = pilot.diagnostics;
    PROBE_REQUIRE(pilot_launch_is_valid(instance, arguments, WORK_STEP_BLOCKS, WORK_BLOCK_THREADS)
                  && work_diagnostics_are_valid(&diagnostics, probe_cuda_array_bytes(diagnostics.status_word),
                                                probe_cuda_array_bytes(diagnostics.thread_loads)),
                  "a launch that its preparation did not validate");
#endif
    if (instance == PILOT_BOUND_ONE_BLOCK) {
        work_kernel_bound_256x1<<<WORK_STEP_BLOCKS, WORK_BLOCK_THREADS>>>(*arguments WORK_DIAGNOSTICS_ARGUMENT);
    } else if (instance == PILOT_BOUND_FIVE_BLOCKS) {
        work_kernel_bound_256x5<<<WORK_STEP_BLOCKS, WORK_BLOCK_THREADS>>>(*arguments WORK_DIAGNOSTICS_ARGUMENT);
    } else {
        work_kernel<<<WORK_STEP_BLOCKS, WORK_BLOCK_THREADS>>>(*arguments WORK_DIAGNOSTICS_ARGUMENT);
    }
    PROBE_AFTER_LAUNCH();
}

/* ---- the card ---- */

/* A ring array gets its stored words from a kernel: 1.8 GB would be seconds over the link. */
static void pilot_fill(void *words, uint64_t word_count, uint32_t kind, int sm_count)
{
    unsigned int block_count = (unsigned int)sm_count * PILOT_FILL_BLOCKS_PER_SM;
    int valid = 1;
    probe_launch_geometry(&valid, "work_fill_kernel", NULL, block_count, 1, 1, PILOT_FILL_THREADS, 0);
    probe_launch_range(&valid, "work_fill_kernel", "ring words", 0, word_count, sizeof(uint32_t),
                       probe_cuda_array_bytes(words));
    PROBE_REQUIRE(valid, "the launch that fills the ring was refused");
    work_fill_kernel<<<block_count, PILOT_FILL_THREADS>>>((uint32_t *)words, word_count, kind);
    PROBE_CHECK_CUDA(cudaGetLastError());
    PROBE_CHECK_CUDA(cudaDeviceSynchronize());
}

static int pilot_open_card(void *, struct probe_card_facts *facts)
{
    uint64_t ring_warp_loads = probe_flag_number(PILOT_FLAG_RING_STEPS, 0) * WORK_STEP_WARP_LOADS;
    uint64_t packed_bytes = ring_warp_loads * WORK_PACKED_BYTES_PER_WARP_LOAD;
    uint64_t scale_bytes = ring_warp_loads * WORK_SCALE_BYTES_PER_WARP_LOAD;
    if (!probe_cuda_open(facts)) {
        return 0;
    }
    pilot.ring_steps = probe_flag_number(PILOT_FLAG_RING_STEPS, 0);
    if ((pilot.ring_steps - 1u) * PILOT_STEP_BYTES < PILOT_COLD_L2_SIZES * facts->l2_bytes) {
        fprintf(stderr, "%s %" PRIu64 ": between two reads of the same bytes a card must read %d times its L2 "
                        "(CONVENTIONS 3.5), and this card's L2 is %.0f MiB\n", PILOT_FLAG_RING_STEPS, pilot.ring_steps,
                (int)PILOT_COLD_L2_SIZES, (double)facts->l2_bytes / PROBE_BYTES_PER_MIB);
        return 0;
    }
    pilot.ring_packed = (uint4 *)probe_cuda_allocate(packed_bytes);
    pilot.ring_scales = (uint32_t *)probe_cuda_allocate(scale_bytes);
    pilot.input_words = (uint4 *)probe_cuda_allocate(sizeof(pilot.host_input_words));
    pilot.input_exponents = (uint16_t *)probe_cuda_allocate(sizeof(pilot.host_input_exponents));
    pilot.sums = (float *)probe_cuda_allocate(PILOT_STEP_SUMS * sizeof(float));
    pilot.checksums = (uint32_t *)probe_cuda_allocate(PILOT_STEP_CHECKSUMS * sizeof(uint32_t));
#if PROBE_DIAGNOSTICS
    pilot.diagnostics.status_word = (uint64_t *)probe_cuda_allocate(sizeof(uint64_t));
    pilot.diagnostics.thread_loads = (uint32_t *)probe_cuda_allocate(PILOT_STEP_THREADS * sizeof(uint32_t));
    pilot.diagnostics.ring_warp_loads = ring_warp_loads;
    pilot.diagnostics.block_count = WORK_STEP_BLOCKS;
#endif
    pilot_fill(pilot.ring_packed, packed_bytes / sizeof(uint32_t), WORK_PACKED, facts->sm_count);
    pilot_fill(pilot.ring_scales, scale_bytes / sizeof(uint32_t), WORK_SCALES, facts->sm_count);
    for (uint32_t word = 0; word < WORK_INPUT_WORDS; word++) {
        pilot.host_input_words[word] = work_stored_word(word, WORK_INPUTS);
    }
    for (uint32_t pair = 0; pair < WORK_INPUT_EXPONENTS; pair++) {
        pilot.host_input_exponents[pair] = (uint16_t)work_stored_word(pair, WORK_EXPONENTS);
    }
    PROBE_CHECK_CUDA(cudaMemcpy(pilot.input_words, pilot.host_input_words, sizeof(pilot.host_input_words),
                                cudaMemcpyHostToDevice));
    PROBE_CHECK_CUDA(cudaMemcpy(pilot.input_exponents, pilot.host_input_exponents, sizeof(pilot.host_input_exponents),
                                cudaMemcpyHostToDevice));
    facts->ring_bytes = packed_bytes + scale_bytes;
    return 1;
}

static void pilot_close_card(void *) { probe_cuda_close(); }

static void pilot_describe_case(void *, const struct probe_case *probe_case, struct probe_case_facts *facts)
{
    facts->format = "synthetic";
    facts->part = "work";
    facts->rows = (int64_t)probe_case_value(probe_case, PILOT_FLAG_ROWS);
    facts->experts = -1;                         /* the bytes of six experts, and no expert */
    facts->overlap = -1;
    facts->megabytes = (double)PILOT_STEP_BYTES / PILOT_BYTES_PER_MB;
}

/* ---- checks ---- */

static uint64_t pilot_differing_words(const uint32_t *first, const uint32_t *second, uint64_t word_count)
{
    uint64_t differing = 0;
    for (uint64_t word = 0; word < word_count; word++) {
        differing += first[word] != second[word] ? 1u : 0u;
    }
    return differing;
}

/* The checked step with one instance's kernel: validated as that instance (its block against its own launch
 * bound), its outputs made stale first (a word the launch does not write cannot pass for a result), read back as
 * words: every block's sums, then every block's checksums. */
static void pilot_run_checked(uint32_t instance, const struct work_arguments *arguments, uint32_t *outputs)
{
    size_t sum_bytes = PILOT_STEP_SUMS * sizeof(float), checksum_bytes = PILOT_STEP_CHECKSUMS * sizeof(uint32_t);
    PROBE_REQUIRE(pilot_launch_is_valid(instance, arguments, WORK_STEP_BLOCKS, WORK_BLOCK_THREADS),
                  "the checked step was refused");
    PROBE_CHECK_CUDA(cudaMemset(pilot.sums, PILOT_STALE_BYTE, sum_bytes));
    PROBE_CHECK_CUDA(cudaMemset(pilot.checksums, PILOT_STALE_BYTE, checksum_bytes));
#if PROBE_DIAGNOSTICS
    PROBE_CHECK_CUDA(cudaMemset(pilot.diagnostics.status_word, 0, sizeof(uint64_t)));
    PROBE_CHECK_CUDA(cudaMemset(pilot.diagnostics.thread_loads, 0, PILOT_STEP_THREADS * sizeof(uint32_t)));
#endif
    pilot_launch(instance, arguments);
    PROBE_CHECK_CUDA(cudaGetLastError());
    PROBE_CHECK_CUDA(cudaDeviceSynchronize());
    PROBE_CHECK_CUDA(cudaMemcpy(outputs, pilot.sums, sum_bytes, cudaMemcpyDeviceToHost));
    PROBE_CHECK_CUDA(cudaMemcpy(outputs + PILOT_STEP_SUMS, pilot.checksums, checksum_bytes, cudaMemcpyDeviceToHost));
}

/* The baseline's outputs against the host: every checksum, and the sums of every PILOT_CHECKED_BLOCK_STRIDE-th
 * block, word for word. The two must_differ: a break flag overwrote on the card what the host still takes from its
 * definition (the weights: both checks see it) or from its own copy (the activation codes: only the sums do). */
static void pilot_check_against_host(const char *case_name, const struct work_arguments *arguments,
                                     int checksums_must_differ, int sums_must_differ)
{
    static float host_sums[WORK_SUMS_PER_BLOCK];
    uint32_t host_checksums[WORK_LANES_PER_WARP], host_sum_words[WORK_SUMS_PER_BLOCK];
    const uint32_t *card_checksums = pilot_outputs + PILOT_STEP_SUMS;
    uint64_t checksums_differing = 0, sums_differing = 0, sums_compared = 0;
    for (uint64_t block = 0; block < WORK_STEP_BLOCKS; block++) {
        int with_sums = block % PILOT_CHECKED_BLOCK_STRIDE == 0;
        work_host_block(pilot.host_input_words, pilot.host_input_exponents, arguments, block,
                        with_sums ? host_sums : NULL, host_checksums);
        checksums_differing += pilot_differing_words(host_checksums, card_checksums + block * WORK_LANES_PER_WARP,
                                                     WORK_LANES_PER_WARP);
        if (with_sums) {
            memcpy(host_sum_words, host_sums, sizeof(host_sums));
            sums_differing += pilot_differing_words(host_sum_words, pilot_outputs + block * WORK_SUMS_PER_BLOCK,
                                                    WORK_SUMS_PER_BLOCK);
            sums_compared += WORK_SUMS_PER_BLOCK;
        }
    }
    probe_check_count(case_name, pilot_instances[PROBE_BASELINE].name, "host_reference_checksums", checksums_differing,
                      PILOT_STEP_CHECKSUMS, checksums_must_differ);
    probe_check_count(case_name, pilot_instances[PROBE_BASELINE].name, "host_reference_sums", sums_differing,
                      sums_compared, sums_must_differ);
}

#if PROBE_DIAGNOSTICS
/* Debug build: what the kernel's own checks stored, and the loads every thread counted. must_report: the launch
 * was given a record its kernel must refuse (--break-diagnostics). */
static void pilot_check_diagnostics(const char *case_name, const char *variant, const struct work_arguments *arguments,
                                    int must_report)
{
    static uint32_t thread_loads[PILOT_STEP_THREADS];
    uint64_t status_word = 0, wrong = 0;
    PROBE_CHECK_CUDA(cudaMemcpy(&status_word, pilot.diagnostics.status_word, sizeof(status_word),
                                cudaMemcpyDeviceToHost));
    PROBE_CHECK_CUDA(cudaMemcpy(thread_loads, pilot.diagnostics.thread_loads, sizeof(thread_loads),
                                cudaMemcpyDeviceToHost));
    for (uint32_t thread = 0; thread < PILOT_STEP_THREADS; thread++) {
        wrong += thread_loads[thread] != arguments->loads_per_warp ? 1u : 0u;
    }
    probe_check_count(case_name, variant, "kernel_diagnostics", wrong + (status_word != 0 ? 1u : 0u),
                      PILOT_STEP_THREADS + 1u, must_report);
}
#endif

/* A megabyte of the checked step's weights overwritten on the card, block 0 in it: the host still computes from the
 * weights' definition, and a kernel launched before sees other weights than one launched after. */
static void pilot_overwrite_weights(const struct work_arguments *arguments)
{
    uint64_t first_byte = arguments->first_warp_load * WORK_PACKED_BYTES_PER_WARP_LOAD;
    PROBE_REQUIRE(first_byte + PILOT_CORRUPTED_BYTES <= probe_cuda_array_bytes(pilot.ring_packed),
                  "the megabyte to overwrite lies outside the ring");
    PROBE_CHECK_CUDA(cudaMemset((uint8_t *)pilot.ring_packed + first_byte, PILOT_CORRUPTED_BYTE_VALUE,
                                PILOT_CORRUPTED_BYTES));
}

/* --break-route: launches past one bound each. The validation that every real launch passes must refuse each; none
 * is launched. A refusal is the failure the flag requires: 1 of 1. */
static void pilot_break_route(const struct probe_case *probe_case)
{
    for (uint32_t broken = PILOT_BROKEN_RING_END; broken < PILOT_BROKEN_END; broken++) {
        struct work_arguments arguments;
        uint64_t block_count = WORK_STEP_BLOCKS, block_threads = WORK_BLOCK_THREADS;
        pilot_step_arguments(probe_case, PILOT_CHECKED_STEP, &arguments);
        if (broken == PILOT_BROKEN_RING_END) {
            arguments.first_warp_load = (pilot.ring_steps - 1u) * WORK_STEP_WARP_LOADS + 1u;
        } else if (broken == PILOT_BROKEN_RING_WRAP) {
            arguments.first_warp_load = UINT64_MAX - 1u;         /* a sum that wraps must not pass for a small one */
        } else if (broken == PILOT_BROKEN_ROWS) {
            arguments.row_count = WORK_MOST_ROWS + 1u;
        } else if (broken == PILOT_BROKEN_LOADS) {
            arguments.loads_per_warp = WORK_MOST_LOADS_PER_WARP + 1u;
        } else if (broken == PILOT_BROKEN_BLOCKS) {
            block_count = WORK_STEP_BLOCKS + 1u;
        } else if (broken == PILOT_BROKEN_BLOCK_THREADS) {
            block_threads = WORK_BLOCK_THREADS + WORK_BLOCK_THREADS;
        } else {
            arguments.sums = (float *)pilot_outputs;             /* host memory: the card has no array there */
        }
        int refused = !pilot_launch_is_valid(PILOT_BOUND_FIVE_BLOCKS, &arguments, block_count, block_threads);
        probe_check_count(probe_case->name, pilot_instances[PILOT_BOUND_FIVE_BLOCKS].name, pilot_broken_names[broken],
                          refused ? 1u : 0u, 1, 1);
    }
}

static void pilot_check_case(void *, const struct probe_case *probe_case)
{
    struct work_arguments arguments;
    int corrupt = probe_flag_given(PROBE_FLAG_CORRUPT), break_inputs = probe_flag_given(PILOT_FLAG_BREAK_INPUTS);
    int break_bits = probe_flag_given(PILOT_FLAG_BREAK_BITS);
    if (probe_flag_given(PROBE_FLAG_BREAK_ROUTE)) {
        pilot_break_route(probe_case);
        return;
    }
    pilot_step_arguments(probe_case, PILOT_CHECKED_STEP, &arguments);
    if (corrupt) {
        pilot_overwrite_weights(&arguments);
    }
    if (break_inputs) {                          /* every row's codes: the host keeps the copy it uploaded */
        PROBE_CHECK_CUDA(cudaMemset(pilot.input_words, PILOT_CORRUPTED_BYTE_VALUE, sizeof(pilot.host_input_words)));
    }
    for (uint32_t instance = PROBE_BASELINE; instance < PROBE_COUNT_OF(pilot_instances); instance++) {
        const char *variant = pilot_instances[instance].name;
        if (instance == PROBE_BASELINE_AGAIN) {
            continue;                            /* the baseline's kernel a second time */
        }
        pilot_run_checked(instance, &arguments, pilot_outputs);
        if (instance == PROBE_BASELINE) {
            pilot_check_against_host(probe_case->name, &arguments, corrupt, corrupt || break_inputs);
            memcpy(pilot_baseline_outputs, pilot_outputs, sizeof(pilot_outputs));
            if (break_bits) {                    /* from here on a kernel reads other weights than the baseline did */
                pilot_overwrite_weights(&arguments);
            }
        } else {
            probe_check_count(probe_case->name, variant, "bits_against_baseline",
                              pilot_differing_words(pilot_outputs, pilot_baseline_outputs, PILOT_STEP_OUTPUT_WORDS),
                              PILOT_STEP_OUTPUT_WORDS, break_bits);
        }
#if PROBE_DIAGNOSTICS
        pilot_check_diagnostics(probe_case->name, variant, &arguments, 0);
#endif
        probe_digest_add(pilot_outputs, sizeof(pilot_outputs));
    }
#if PROBE_DIAGNOSTICS
    if (probe_flag_given(PILOT_FLAG_BREAK_DIAGNOSTICS)) {        /* its outputs stay stale and enter no digest */
        pilot.diagnostics.block_count = 0;
        pilot_run_checked(PROBE_BASELINE, &arguments, pilot_outputs);
        pilot.diagnostics.block_count = WORK_STEP_BLOCKS;
        pilot_check_diagnostics(probe_case->name, PILOT_REFUSED_BY_THE_KERNEL, &arguments, 1);
    }
#endif
    probe_cuda_check_guards(probe_case->name);
}

/* ---- timing callbacks ---- */

static int pilot_prepare_step(void *, const struct probe_case *probe_case, uint32_t instance, uint64_t step,
                              uint32_t slot)
{
    PROBE_REQUIRE(slot < PROBE_STREAM_STEPS, "a slot past the slots of a stream");
    pilot_step_arguments(probe_case, step, &pilot.arguments[slot]);
    return pilot_launch_is_valid(instance, &pilot.arguments[slot], WORK_STEP_BLOCKS, WORK_BLOCK_THREADS);
}

static uint32_t pilot_launch_step(void *, const struct probe_case *, uint32_t instance, uint32_t slot)
{
    pilot_launch(instance, &pilot.arguments[slot]);
    return 1;
}

/* The pilot has no bar: its figures are the noise floor and what a stated bound does, per case. */
static void pilot_verdicts(void *, const struct probe_case *probe_case, const struct probe_result *results,
                           double noise_ratio)
{
    char figure[PILOT_FIGURE_BYTES];
    PROBE_REQUIRE(snprintf(figure, sizeof(figure), "noise_floor_at_%s", probe_case->name) > 0, "a figure's name");
    probe_verdict(figure, PROBE_NAN, PROBE_NAN, noise_ratio, noise_ratio, results[PROBE_BASELINE_AGAIN].same_sign,
                  results[PROBE_BASELINE_AGAIN].rounds);
    for (uint32_t instance = PILOT_BOUND_ONE_BLOCK; instance <= PILOT_BOUND_FIVE_BLOCKS; instance++) {
        const struct probe_result *result = &results[instance];         /* not timed: its ratio is no number */
        PROBE_REQUIRE(snprintf(figure, sizeof(figure), "launch_bound_256x%u_at_%s",
                               pilot_instances[instance].bound_blocks, probe_case->name) > 0, "a figure's name");
        probe_verdict(figure, PROBE_NAN, PROBE_NAN, result->ratio, noise_ratio, result->same_sign, result->rounds);
    }
}

static struct probe_program pilot_program = {
    "pilot", pilot_flags, PROBE_COUNT_OF(pilot_flags), pilot_instances, PROBE_COUNT_OF(pilot_instances),
    NULL,                                        /* no context: one card, one state */
    NULL,                                        /* nothing to configure: the instances are a fixed table */
    pilot_open_card, pilot_describe_case, pilot_check_case, pilot_prepare_step, pilot_launch_step,
    probe_cuda_mark_start, probe_cuda_elapsed_microseconds, pilot_verdicts, pilot_close_card,
    NULL};                                       /* the harness's rounds are the pilot's protocol */

int main(int argument_count, char **arguments) { return probe_main(argument_count, arguments, &pilot_program); }
