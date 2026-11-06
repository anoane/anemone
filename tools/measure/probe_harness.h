/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * probe_harness.h: what every probe of the speed campaign's rounds shares, written once: the command line, the
 * table of instances, the timing protocol (rounds, `step`, `stream`, `soak`), the statistics, the validation of a
 * launch, guard bytes, the digest and the output lines of research/speed/CONVENTIONS.md 6. A probe includes this
 * file and changes nothing in it. Header only: C11 (gcc, the engine's warnings, -D_GNU_SOURCE) and C++17 (nvcc).
 * It proves nothing about a kernel: every check is the probe's; the harness prints and counts them. It is over the
 * 1,000-line limit, pending the owner's word; hence its dense layout. Measurement tool: not part of the engine.
 *
 * BUILD (batch_round_common.sh passes all of it). Required, 0 or 1 each: -DPROBE_DIAGNOSTICS (1: the debug build:
 * PROBE_INDEX_CHECK and PROBE_AFTER_LAUNCH exist, COUNTERS is printed), -DPROBE_PROFILE (1: the profile build, for
 * the profiler and compute-sanitizer), -DPROBE_FMAD (1: the fast-mode build of CONVENTIONS 1.2: flags=fmad). Only
 * the release build (the first two 0) prints RESULT, BEST and a verdict; the others print UNTIMED with build=debug
 * or build=profile. In a .cu also -DPROBE_SM=80|120 (sm_80, sm_120a): probe_cuda_open refuses any other card.
 * A PROBE WRITES a table of struct probe_flag that begins with PROBE_COMMON_FLAGS, a table of struct probe_instance,
 * a struct probe_program (every field is described there) and `return probe_main(count, arguments, &program);`
 * as its main: pilot_probe.cu is the model. Instance 0 is named "baseline" and instance 1 "baseline_again" (the
 * same kernel a second time: their difference is the run's noise floor). A probe that times nothing has no
 * instance and no timing callback. One with a timing protocol of its own gives time_case and prints there, per
 * case, a ROUND line per round or window (probe_print_round: round_join_clocks takes the case's span of time, and
 * so its clock, from them; RESULT lines after no ROUND line reach the joined log unchanged), then its RESULT lines.
 *
 * FLAGS OF EVERY PROBE (PROBE_COMMON_FLAGS)
 *   --card-index N       the nvidia-smi index of the ONE card this process sees (CUDA_VISIBLE_DEVICES=<index>): what
 *                        card= says in every line (default 0). --batch ID: what batch= says (default none).
 *   --rounds N           rounds of a timed run (default 20); --steps-per-round N (default 100); --mode step,stream:
 *                        which of the two timings a run takes (default both).
 *   --steps N            a short run, for the debug build and the sanitizer's tools: one round of N steps, streams
 *                        of at most N steps, no warm-up. Its lines are UNTIMED, never RESULT.
 *   --instance NAME      repeatable (a variant's name holds commas: no list): only these are timed, and the
 *                        baseline pair with them.
 *   --soak-seconds S     instead of rounds: the ONE instance --instance names (the baseline without it) alone, in
 *                        streams for S seconds after its warm-up: one RESULT line, mode=soak (CONVENTIONS 3.6).
 *   --profile-case NAME  for the profiler: only that variant's step, --steps times (default 8), in the first case of
 *                        the case flags, no check, no warm-up; prints LAUNCHES.
 *   --list-instances     no card is touched: an INSTANCE line each; exit 1 if the baseline spills.
 *   --corrupt            the probe overwrites stored weights on the card after the host decoded them.
 *   --break-route        the probe asks for launches outside their arrays or limits: each must be refused.
 *   --break-guard        a .cu: probe_cuda_check_guards first overwrites a guard byte: guard_bytes must report.
 * A probe's own: PROBE_SWITCH(name, what); PROBE_NUMBER(name, least, most, default, what); PROBE_NUMBER_LIST (same
 * arguments); PROBE_WORD_LIST(name, words, default, what); PROBE_TEXT(name, default, what); PROBE_CASE_NUMBERS and
 * PROBE_CASE_WORDS: lists of which every value is a case. The cases of a run are the cross product of the case
 * flags in the order declared, named "rows:6,overlap:50" (the flag without its dashes, '-' as '_'; no case flag:
 * one case, "all"). The value of a word is its index among the flag's words. A switch named --corrupt or --break-*
 * is a break flag: the run then checks its first case only, times nothing, and exits 0 only if at least one check
 * reported what it was required to and none came out otherwise. Readers: probe_flag_given(name), probe_flag_text,
 * probe_flag_number(name, index), probe_flag_has_word(name, word), probe_flag(name) (the record: value_count,
 * values), probe_case_value(case, flag name), probe_breaking().
 * LIMITS. A list flag: at most 16 values (more is refused, with the usage text). A run: at most 256 cases of at most
 * 8 case flags, a case's name under 128 bytes, 128 instances, 32 card arrays (beyond any of them the run stops and
 * says which). A kernel's name in the registers file: under 128 bytes (a longer one is not found: registers=na).
 *
 * A RUN (probe_main): flags; configure(); the registers file (<path of the executable>.registers, lines "<kernel>
 * <registers> <spill bytes>" from round_build_card_probe; absent: registers=na); --list-instances ends here;
 * open_card() and CARD; then per case describe_case(), check_case() and, unless a break flag is given, the timing:
 * 3 s of the interleaved sequence untimed (CONVENTIONS 3.2), then per round every selected instance for
 * --steps-per-round steps (step mode: one step alone on an idle card) and one stream each (PROBE_STREAM_STEPS steps
 * prepared into slots 0 .. 63, then queued back to back), the result lines, verdicts(); at the end DIGEST, COUNTERS,
 * CHECKS, close_card(). The step number given to prepare_step counts up over the whole run and no two timed steps
 * have the same: a probe that maps it into its ring never reads what the launch before it read.
 * OUTPUT LINES (one fact a line, fixed keys, never wrapped; card= is the nvidia-smi index, or "host")
 *   CARD     card= name="" arch= sms= l2_mib= ring_mib= ring_l2=
 *   CHECK    card= case= variant= name= differing= of= verdict=             probe_check_count
 *   CHECK    card= case= variant= name= error_over_rms= limit= verdict=     probe_check_error
 *            verdict: ok, FAILED, or reported where the caller said the check must fail (the line then ends with
 *            required=report). A launch that --break-route broke and the host refused is a count: 1 of 1.
 *   REFUSED  card= kernel= what="" value= least= most=      a launch check that did not hold
 *   ROUND    card= case= round= stamp= end= baseline_us=    local wall-clock time of the round's begin and end
 *            (YYYY-MM-DDTHH:MM:SS.mmm), the baseline's median in it; a soak prints one for its whole window
 *   NOISE    card= case= baseline_us= baseline_again_us= difference_us= ratio=
 *   RESULT   batch= probe= build=release flags= arch= card= format= part= rows= experts= overlap= variant= class=
 *            [role=bound] mode= n= [seconds=] median_us= p99_us= min_us= mean_us= ratio= paired_us= same_sign=
 *            stream_us= mb= gbps= registers= spill_bytes= kcyc=na clock_mhz=na power_w=na uj=na checks=
 *            kcyc, clock_mhz, power_w, uj: ALWAYS na here (no probe reads a card's clock or power): the batch fills
 *            them where CONVENTIONS allows (round_join_clocks: <step>.joined.log). seconds=: a soak's only, what
 *            its window lasted. An instance refused at compile time (available 0): n=0, every figure na.
 *   UNTIMED  the keys of RESULT, from a debug or profile build or after --steps: never a result
 *   BEST     card= case= variant=      the lowest step median of the timed instances that neither spill nor are bounds
 *   VERDICT  card= figure= bar_ratio= bar_us= ratio= noise_ratio= same_sign= verdict=     probe_verdict: met,
 *            not_met, undecided, measured, or untimed (a debug or profile build, or after --steps)
 *   LAUNCHES card= case= variant= per_step= steps=                              after --profile-case
 *   INSTANCE probe= variant= class= [role=bound] kernel= block_threads= bound= available= registers= spill_bytes=
 *            bound=: the launch bound; role=bound, here and in RESULT: an instance probe_mark_bound marked
 *   DIGEST   card= arch= probe= bytes= value=      of everything given to probe_digest_add, in order
 *   COUNTERS card= launches= steps=                debug build only
 *   CHECKS   card= failed= reported= verdict=      the last line; exit 0 exactly when failed=0
 *
 * FUNCTIONS A PROBE CALLS: probe_check_count and probe_check_error (a CHECK line, counted); probe_digest_add(bytes,
 * count) (what the card computed in the check cases); probe_launch_value, probe_launch_range, probe_launch_geometry
 * (check 10 of CONVENTIONS 5: see them below); probe_instance_index(variant) (a variant that lost its bits gets C
 * or F as the contract_class of its instance); in verdicts(), per acceptance figure, probe_verdict(figure, bar
 * ratio, bar us, ratio, noise ratio, same-sign rounds, rounds) with the ratio, same_sign and rounds of the judged
 * instance's result (0 and 0: a figure without pairs): its comment says how CONVENTIONS 3.3 is applied; for a bar
 * between two instances, either one the baseline or neither, probe_verdict_between(figure, bar ratio, judged result,
 * against result, noise ratio) pairs them itself; in configure(), probe_mark_bound(instance) for a bound;
 * PROBE_REQUIRE(condition, text) (ends the run on a state that must not occur); PROBE_NAN (a figure that cannot be
 * had); PROBE_COUNT_OF(array). For a timing protocol of a probe's own (a duty cycle): probe_timed_steps(case,
 * instance, steps) (microseconds a step, through the program's four timing callbacks), probe_fill_result,
 * probe_statistics_of, probe_median, probe_print_round, probe_print_result, probe_stamp, probe_monotonic_seconds,
 * probe_number, probe_whole, probe_ratio, probe_pairs; a stamp needs PROBE_STAMP_BYTES, a stream has 64 slots.
 * WITH THE CUDA RUNTIME (a .cu): PROBE_CHECK_CUDA(call); probe_cuda_open(facts) (one visible card of this build's
 * capability; fills name, architecture, SMs, L2 and the limits); probe_cuda_allocate(bytes) (a card array with
 * PROBE_GUARD_BYTES of PROBE_GUARD_VALUE behind it); probe_cuda_array_bytes(pointer) (what was allocated: a launch
 * is validated against it, not against a planned size; 0 for a pointer that is no array);
 * probe_cuda_check_guards(case name) (a CHECK line: call it at the end of check_case); probe_cuda_mark_start and
 * probe_cuda_elapsed_microseconds (the two timing callbacks: the driver's events, as the earlier probes timed);
 * probe_cuda_close(); PROBE_AFTER_LAUNCH() (debug: waits for the card, verifies every guard; release: nothing).
 */
#ifndef PROBE_HARNESS_H
#define PROBE_HARNESS_H
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#if !defined(PROBE_DIAGNOSTICS) || !defined(PROBE_PROFILE) || !defined(PROBE_FMAD)
#error "PROBE_DIAGNOSTICS, PROBE_PROFILE and PROBE_FMAD must each be defined 0 or 1: batch_round_common.sh does"
#endif
#define PROBE_BUILD (PROBE_DIAGNOSTICS ? "debug" : PROBE_PROFILE ? "profile" : "release")       /* build= of a line */
#if defined(__cplusplus)
#define PROBE_FUNCTION inline                    /* plain inline: no front end reports one a probe leaves unused */
#else
#define PROBE_FUNCTION static inline
#endif
enum {                                           /* PROBE_NUMBER_TEXTS: the numbers one printf may hold at once */
    PROBE_MOST_LIST_VALUES = 16, PROBE_MOST_INSTANCES = 128, PROBE_MOST_CASE_AXES = 8, PROBE_MOST_CASES = 256,
    PROBE_NAME_BYTES = 128, PROBE_PATH_BYTES = 1024, PROBE_STAMP_BYTES = 32, PROBE_NUMBER_BYTES = 32,
    PROBE_NUMBER_TEXTS = 32, PROBE_REGISTERS_FIELDS = 3, PROBE_DECIMAL_BASE = 10, PROBE_GRID_AXES = 3,
    PROBE_MOST_CARD_ARRAYS = 32, PROBE_GUARD_BYTES = 4096, PROBE_GUARD_VALUE = 0xA5, PROBE_LANES_PER_WARP = 32,
    PROBE_BASELINE = 0, PROBE_BASELINE_AGAIN = 1, PROBE_PERCENTILE = 99, PROBE_PERCENT = 100, PROBE_HALF = 2,
    PROBE_RATIO_DECIMALS = 3, PROBE_MOST_CARD_INDEX = 255, PROBE_PAIRS_NEEDED = 17, PROBE_PAIRS_OF = 20,
    PROBE_SM_80 = 80, PROBE_SM_120A = 120, PROBE_SM_PER_MAJOR = 10       /* -DPROBE_SM: ten times major, plus minor */
};
#define PROBE_REGISTERS_SCAN "%127s %d %lld"         /* a line of the registers file; the width: the bytes less 1 */
#define PROBE_MOST_ROUNDS 100000u
#define PROBE_MOST_STEPS 1000000u                    /* also the most seconds and the most streams of a soak */
#define PROBE_STREAM_STEPS 64u                       /* CONVENTIONS 3.1: at least 64 steps back to back */
#define PROBE_PROFILE_STEPS 8u
#define PROBE_WARM_UP_SECONDS 3.0                    /* CONVENTIONS 3.2: a CMP 170HX needs about 3 s to its clock */
#define PROBE_NONE UINT32_MAX
#define PROBE_COUNT_OF(array) ((uint32_t)(sizeof(array) / sizeof((array)[0])))
#define PROBE_NAN ((double)NAN)                      /* a figure that cannot be had; NAN itself is a float */
#define PROBE_NANOSECONDS_PER_SECOND 1.0e9
#define PROBE_NANOSECONDS_PER_MILLISECOND 1000000L
#define PROBE_MICROSECONDS_PER_MILLISECOND 1000.0
#define PROBE_GBPS_PER_MB_PER_US 1000.0     /* 1 MB in 1 us is 1,000 GB/s */
#define PROBE_BYTES_PER_MIB (1024.0 * 1024.0)
#define PROBE_DIGEST_OFFSET 0xcbf29ce484222325ull    /* FNV-1a over 64 bits */
#define PROBE_DIGEST_PRIME 0x100000001b3ull
/* ---- flags ---- */
enum probe_flag_kind { PROBE_FLAG_SWITCH, PROBE_FLAG_NUMBERS, PROBE_FLAG_WORDS, PROBE_FLAG_TEXT };
struct probe_flag {
    const char *name;                            /* "--rows" */
    int kind, is_case_axis;                      /* enum probe_flag_kind; 1: every value of the list is a case */
    uint32_t most_values;                        /* 1: one value, no list */
    const char *words;                           /* PROBE_FLAG_WORDS: the accepted words, comma separated */
    uint64_t minimum, maximum;                   /* PROBE_FLAG_NUMBERS: of every value */
    const char *default_text, *what;             /* as on the command line, "" for none; a line of the usage text */
    uint64_t values[PROBE_MOST_LIST_VALUES];     /* read: the numbers, or the indices of the words */
    const char *tokens[PROBE_MOST_LIST_VALUES];  /* read: each value as it was written, and its length */
    size_t token_bytes[PROBE_MOST_LIST_VALUES];
    uint32_t value_count;
    const char *text;                            /* read: the whole value */
    int given;                                   /* read: it was on the command line */
};
#define PROBE_FLAG_READ {0}, {NULL}, {0}, 0, NULL, 0
#define PROBE_FLAG_LIST(name, kind, is_case, words, least, most, preset, what) \
    {name, kind, is_case, PROBE_MOST_LIST_VALUES, words, least, most, preset, what, PROBE_FLAG_READ}
#define PROBE_SWITCH(name, what) {name, PROBE_FLAG_SWITCH, 0, 0, NULL, 0, 0, "", what, PROBE_FLAG_READ}
#define PROBE_NUMBER(name, least, most, preset, what) \
    {name, PROBE_FLAG_NUMBERS, 0, 1, NULL, least, most, preset, what, PROBE_FLAG_READ}
#define PROBE_TEXT(name, preset, what) {name, PROBE_FLAG_TEXT, 0, 1, NULL, 0, 0, preset, what, PROBE_FLAG_READ}
#define PROBE_NUMBER_LIST(name, least, most, preset, what) \
    PROBE_FLAG_LIST(name, PROBE_FLAG_NUMBERS, 0, NULL, least, most, preset, what)
#define PROBE_CASE_NUMBERS(name, least, most, preset, what) \
    PROBE_FLAG_LIST(name, PROBE_FLAG_NUMBERS, 1, NULL, least, most, preset, what)
#define PROBE_WORD_LIST(name, list, preset, what) PROBE_FLAG_LIST(name, PROBE_FLAG_WORDS, 0, list, 0, 0, preset, what)
#define PROBE_CASE_WORDS(name, list, preset, what) PROBE_FLAG_LIST(name, PROBE_FLAG_WORDS, 1, list, 0, 0, preset, what)
#define PROBE_FLAG_STEPS "--steps"
#define PROBE_FLAG_MODE "--mode"
#define PROBE_FLAG_SOAK_SECONDS "--soak-seconds"
#define PROBE_FLAG_INSTANCE "--instance"
#define PROBE_FLAG_PROFILE_CASE "--profile-case"
#define PROBE_FLAG_CORRUPT "--corrupt"
#define PROBE_FLAG_BREAK_ROUTE "--break-route"
#define PROBE_FLAG_BREAK_GUARD "--break-guard"
#define PROBE_MODE_STEP "step"
#define PROBE_MODE_STREAM "stream"
#define PROBE_MODES PROBE_MODE_STEP "," PROBE_MODE_STREAM
#define PROBE_COMMON_FLAGS                                                                                        \
    PROBE_NUMBER("--card-index", 0, PROBE_MOST_CARD_INDEX, "0", "the nvidia-smi index of the one visible card"),  \
    PROBE_TEXT("--batch", "none", "the batch's id, for the result lines"),                                        \
    PROBE_NUMBER("--rounds", 1, PROBE_MOST_ROUNDS, "20", "rounds of a timed run"),                                \
    PROBE_NUMBER("--steps-per-round", 1, PROBE_MOST_STEPS, "100", "steps of each instance in a round"),           \
    PROBE_NUMBER(PROBE_FLAG_STEPS, 1, PROBE_MOST_STEPS, "", "a short run: one round of N steps, never a result"), \
    PROBE_WORD_LIST(PROBE_FLAG_MODE, PROBE_MODES, PROBE_MODES, "the timings of a run"),                           \
    PROBE_NUMBER(PROBE_FLAG_SOAK_SECONDS, 1, PROBE_MOST_STEPS, "", "one instance alone, in streams, this long"),  \
    PROBE_TEXT(PROBE_FLAG_INSTANCE, "", "repeatable: time only this variant (and the baseline pair)"),            \
    PROBE_TEXT(PROBE_FLAG_PROFILE_CASE, "", "only this variant's step, for the profiler"),                        \
    PROBE_SWITCH("--list-instances", "print the instances with their registers; touch no card"),                  \
    PROBE_SWITCH(PROBE_FLAG_CORRUPT, "overwrite stored weights on the card: the reference must report"),          \
    PROBE_SWITCH(PROBE_FLAG_BREAK_ROUTE, "ask for launches outside their arrays: each must be refused"),          \
    PROBE_SWITCH(PROBE_FLAG_BREAK_GUARD, "overwrite a guard byte behind a card array: guard_bytes must report")
/* ---- what a probe declares ---- */
struct probe_instance {                          /* one named instance of the program: a variant */
    const char *name;                            /* "baseline", "bounds:256x5", "operand:carried_bf16,ahead:1" */
    const char *kernel;                          /* the kernels whose registers it reports, as c++filt names them less
                                                    blanks, result and parameters: "a_kernel<0,2>+b_kernel"; or NULL */
    char contract_class;                         /* 'E', 'R', 'C' or 'F', as claimed */
    int available;                               /* 0: the form was refused at compile time */
    uint32_t block_threads;                      /* the block every launch of it has; 0: not checked */
    uint32_t bound_threads, bound_blocks;        /* its launch bound; 0 0: it states none */
    int32_t registers;                           /* read from the registers file, the most of its kernels; < 0: none */
    int64_t spill_bytes;                         /* spill stores plus spill loads of the report, over its kernels */
};
struct probe_card_facts {                        /* open_card fills it */
    const char *name, *architecture;             /* "NVIDIA CMP 170HX", "sm_80" */
    int sm_count;
    uint64_t l2_bytes, ring_bytes, shared_bytes_most;       /* ring: the cold weights the steps walk through; 0: none */
    uint32_t block_threads_most, warps_per_sm, grid_most[PROBE_GRID_AXES];      /* with the shared bytes: the limits */
};
struct probe_case {                              /* one combination of the case flags */
    uint32_t index, axis_count;
    const struct probe_flag *axes[PROBE_MOST_CASE_AXES];
    uint64_t values[PROBE_MOST_CASE_AXES];       /* a number, or the index of a word */
    char name[PROBE_NAME_BYTES];                 /* "rows:6,overlap:50" */
};
struct probe_case_facts {                        /* describe_case fills it; a number that cannot be had: negative */
    const char *format, *part;                   /* RESULT format= part= */
    int64_t rows, experts, overlap;
    double megabytes;                            /* the weight bytes one step reads, in MB */
};
struct probe_statistics { uint64_t count; double median, percentile_99, minimum, mean; };
struct probe_result {                            /* one RESULT line */
    const char *variant, *mode;                  /* mode NULL: the instance was not timed */
    char contract_class;
    int short_run;                               /* after --steps: printed as UNTIMED */
    struct probe_case_facts facts;
    struct probe_statistics step;                /* count 0: no figure */
    double ratio, paired_microseconds, stream_microseconds, soak_seconds;       /* PROBE_NAN: none */
    uint32_t same_sign, rounds;                  /* rounds 0: no pairs */
    int32_t registers;
    int64_t spill_bytes;
    int is_bound;                                /* its instance is a bound (probe_mark_bound) */
    const double *round_medians;                 /* in verdicts(): its step median in each of round_count rounds */
    uint32_t round_count;                        /* 0, round_medians NULL: not timed in step mode */
};
struct probe_program {
    const char *probe_name;                      /* RESULT probe= */
    struct probe_flag *flags;                    /* PROBE_COMMON_FLAGS, then the probe's own */
    uint32_t flag_count;
    struct probe_instance *instances;            /* configure may fill them */
    uint32_t instance_count;
    void *context;                               /* given back to every callback */
    int (*configure)(void *context);             /* after the flags; 0 refuses the run (it has said why); may be NULL */
    int (*open_card)(void *context, struct probe_card_facts *facts);     /* 0 refuses; NULL: a host probe, card=host */
    void (*describe_case)(void *context, const struct probe_case *probe_case, struct probe_case_facts *facts);
    void (*check_case)(void *context, const struct probe_case *probe_case);      /* its checks, untimed; may be NULL */
    /* not timed: routes, tables and inputs of step `step` into slot `slot`, and the validation of every launch the
     * step will make (check 10). 0: a launch was refused; nothing more of the case is timed and the run fails. */
    int (*prepare_step)(void *context, const struct probe_case *probe_case, uint32_t instance, uint64_t step,
                        uint32_t slot);
    /* timed: queues the launches of the step prepared in the slot and nothing else; returns their number. */
    uint32_t (*launch_step)(void *context, const struct probe_case *probe_case, uint32_t instance, uint32_t slot);
    void (*mark_start)(void *context);           /* the first event, on an idle card */
    double (*elapsed_microseconds)(void *context);       /* the second event, the wait for it, their distance */
    /* after a case's result lines (not after a soak): probe_verdict per acceptance figure. results[instance]: of
     * one not timed mode is NULL and ratio PROBE_NAN; noise_ratio: baseline_again over baseline. May be NULL. */
    void (*verdicts)(void *context, const struct probe_case *probe_case, const struct probe_result *results,
                     double noise_ratio);
    void (*close_card)(void *context);           /* may be NULL */
    /* a timing protocol of the probe's own (a duty cycle) in place of the rounds, after check_case. May be NULL. */
    void (*time_case)(void *context, const struct probe_case *probe_case, const struct probe_case_facts *facts);
};
static struct {
    struct probe_program *program;
    char card_label[PROBE_NUMBER_BYTES];
    struct probe_card_facts card;                /* architecture "host" and no limits without a card */
    uint32_t failures, reports, breaking;        /* checks not as required; required failures reported; a break flag */
    uint64_t digest, digest_bytes, launches, steps;
    const char *selected_names[PROBE_MOST_INSTANCES];
    uint32_t selected_count;
    uint8_t is_bound[PROBE_MOST_INSTANCES];      /* 1: probe_mark_bound marked the instance */
} probe_run;
/* A state that must not occur ends the run. A macro over a function that does not return: the compiler then sees,
 * without inlining anything, that nothing after a failed requirement is reached. */
PROBE_FUNCTION __attribute__((noreturn)) void probe_stop(const char *what) {
    fprintf(stderr, "probe harness: %s\n", what);
    exit(EXIT_FAILURE);
}
#define PROBE_REQUIRE(holds, what) ((holds) ? (void)0 : probe_stop(what))
/* Adds a piece to a text that must hold it whole: a name that does not fit is a mistake of the probe. */
PROBE_FUNCTION void probe_append(char *text, size_t text_bytes, const char *piece, size_t piece_bytes) {
    size_t used = strlen(text);
    PROBE_REQUIRE(piece_bytes < text_bytes - used, "a name is longer than PROBE_NAME_BYTES");
    memcpy(text + used, piece, piece_bytes);
    text[used + piece_bytes] = '\0';
}
/* A figure as text, "na" for one that cannot be had (PROBE_NAN). The last PROBE_NUMBER_TEXTS texts stay valid. */
PROBE_FUNCTION char *probe_number(double value, int decimals) {
    static char texts[PROBE_NUMBER_TEXTS][PROBE_NUMBER_BYTES];
    static uint32_t next_text;
    char *text = texts[next_text++ % PROBE_NUMBER_TEXTS];
    int written = isnan(value) ? -1 : snprintf(text, PROBE_NUMBER_BYTES, "%.*f", decimals, value);
    if (written < 0 || written >= PROBE_NUMBER_BYTES) { memcpy(text, "na", sizeof("na")); }
    return text;
}
PROBE_FUNCTION const char *probe_whole(int64_t value) { return probe_number(value < 0 ? PROBE_NAN : (double)value, 0); }
PROBE_FUNCTION const char *probe_ratio(double value) { return probe_number(value, PROBE_RATIO_DECIMALS); }
PROBE_FUNCTION const char *probe_pairs(uint32_t same_sign, uint32_t rounds) {    /* "17/20"; "na" of no pair */
    char *text = probe_number(PROBE_NAN, 0);
    PROBE_REQUIRE(rounds == 0 || snprintf(text, PROBE_NUMBER_BYTES, "%u/%u", same_sign, rounds) > 0, "pairs as text");
    return text;
}
/* ---- reading the command line ---- */
PROBE_FUNCTION struct probe_flag *probe_flag_find(const char *name) {    /* NULL: no flag of that name */
    struct probe_program *program = probe_run.program;
    for (uint32_t index = 0; index < program->flag_count; index++) {
        if (strcmp(program->flags[index].name, name) == 0) { return &program->flags[index]; }
    }
    return NULL;
}
PROBE_FUNCTION struct probe_flag *probe_flag(const char *name) {
    struct probe_flag *flag = probe_flag_find(name);
    PROBE_REQUIRE(flag != NULL, "a flag is read that the table lacks: it begins with PROBE_COMMON_FLAGS");
    return flag;
}
PROBE_FUNCTION int probe_flag_given(const char *name) { return probe_flag(name)->given; }
PROBE_FUNCTION const char *probe_flag_text(const char *name) { return probe_flag(name)->text; }
PROBE_FUNCTION int probe_breaking(void) { return probe_run.breaking != 0; }
PROBE_FUNCTION uint64_t probe_flag_number(const char *name, uint32_t index) {
    const struct probe_flag *flag = probe_flag(name);
    PROBE_REQUIRE(flag->kind != PROBE_FLAG_TEXT && index < flag->value_count, "a flag is read past its values");
    return flag->values[index];
}
PROBE_FUNCTION int probe_flag_has_word(const char *name, const char *word) {     /* among the values of a word list */
    const struct probe_flag *flag = probe_flag(name);
    int has_word = 0;
    for (uint32_t index = 0; flag->kind == PROBE_FLAG_WORDS && index < flag->value_count; index++) {
        has_word |= strlen(word) == flag->token_bytes[index] && strncmp(flag->tokens[index], word, strlen(word)) == 0;
    }
    return has_word;
}
/* A flag's value: a text, or a list of whole numbers inside its range, or of its words. 0: not of its kind. */
PROBE_FUNCTION int probe_flag_set(struct probe_flag *flag, const char *text) {
    flag->text = text;
    flag->value_count = flag->kind == PROBE_FLAG_TEXT ? 1u : 0u;
    for (const char *token = text; flag->kind != PROBE_FLAG_TEXT; token += strcspn(token, ",") + 1u) {
        size_t token_bytes = strcspn(token, ","), word_bytes = 0;
        uint32_t at = flag->value_count;
        const char *word = flag->kind == PROBE_FLAG_WORDS ? flag->words : "";
        char *number_end = NULL;
        if (at >= flag->most_values || at >= PROBE_MOST_LIST_VALUES || token_bytes == 0) { return 0; }
        errno = 0;
        flag->values[at] = flag->kind == PROBE_FLAG_WORDS ? 0 : strtoull(token, &number_end, PROBE_DECIMAL_BASE);
        for (; *word != '\0'; flag->values[at]++) {                  /* the index of the word the token is */
            word_bytes = strcspn(word, ",");
            if (word_bytes == token_bytes && strncmp(word, token, token_bytes) == 0) { break; }
            word += word_bytes + (word[word_bytes] == ',' ? 1u : 0u);
        }
        if (flag->kind == PROBE_FLAG_WORDS ? *word == '\0'           /* strtoull would take a sign or a blank */
                : token[0] < '0' || token[0] > '9' || errno != 0 || number_end != token + token_bytes
                  || flag->values[at] < flag->minimum || flag->values[at] > flag->maximum) {
            return 0;
        }
        flag->tokens[at] = token;
        flag->token_bytes[at] = token_bytes;
        flag->value_count++;
        if (token[token_bytes] == '\0') { break; }
    }
    return 1;
}
/* Defaults first, then the command line. 0, after the usage text: no such flag, or no value of the flag's kind. */
PROBE_FUNCTION int probe_read_flags(int argument_count, char **arguments) {
    struct probe_program *program = probe_run.program;
    const char *refused = NULL;
    for (uint32_t index = 0; index < program->flag_count; index++) {
        struct probe_flag *flag = &program->flags[index];
        flag->text = flag->default_text;
        PROBE_REQUIRE(flag->kind == PROBE_FLAG_SWITCH || flag->default_text[0] == '\0'
                      || probe_flag_set(flag, flag->default_text), "a flag's default is not one of its values");
        PROBE_REQUIRE(probe_flag_find(flag->name) == flag, "a flag is declared twice");
    }
    for (int argument_index = 1; refused == NULL && argument_index < argument_count; argument_index++) {
        struct probe_flag *flag = probe_flag_find(arguments[argument_index]);
        int is_switch = flag != NULL && flag->kind == PROBE_FLAG_SWITCH;
        refused = arguments[argument_index];
        if (flag == NULL || (!is_switch && (++argument_index >= argument_count
                                            || !probe_flag_set(flag, arguments[argument_index])))) {
            break;                               /* no such flag, or no value of its kind */
        }
        flag->given = 1;
        flag->value_count = is_switch ? 1u : flag->value_count;
        probe_run.breaking |= (uint32_t)(is_switch && (strcmp(flag->name, PROBE_FLAG_CORRUPT) == 0
                                                       || strstr(flag->name, "--break-") == flag->name));
        if (strcmp(flag->name, PROBE_FLAG_INSTANCE) == 0 && probe_run.selected_count < PROBE_MOST_INSTANCES) {
            probe_run.selected_names[probe_run.selected_count++] = arguments[argument_index];
        }
        refused = NULL;
    }
    for (uint32_t index = 0; refused != NULL && index < program->flag_count; index++) {
        const struct probe_flag *flag = &program->flags[index];
        int numbers = flag->kind == PROBE_FLAG_NUMBERS, valued = numbers || flag->kind == PROBE_FLAG_WORDS;
        fprintf(stderr, "%s%s  %s%s: %s (%s%s%s%sdefault: %s)\n", index == 0 ? refused : "",
                index == 0 ? ": no such flag, or no value of its kind (a list: commas, no blank). The flags:\n" : "",
                flag->name, flag->kind == PROBE_FLAG_SWITCH ? "" : flag->most_values > 1 ? " <list>" : " <value>",
                flag->what, numbers ? probe_number((double)flag->minimum, 0) : valued ? flag->words : "",
                numbers ? " to " : "", numbers ? probe_number((double)flag->maximum, 0) : "", valued ? "; " : "",
                flag->default_text);
    }
    return refused == NULL;
}
/* ---- cases ---- */
PROBE_FUNCTION uint64_t probe_case_value(const struct probe_case *probe_case, const char *flag_name) {
    uint32_t axis = 0;
    while (axis < probe_case->axis_count && strcmp(probe_case->axes[axis]->name, flag_name) != 0) { axis++; }
    PROBE_REQUIRE(axis < probe_case->axis_count, "a case value is read of a flag that is no case flag");
    return probe_case->values[axis];
}
/* Case case_index of the cross product of the case flags, the first flag declared varying slowest; returns the
 * number of cases, which is the product of the case flags' value counts. */
PROBE_FUNCTION uint32_t probe_make_case(uint32_t case_index, struct probe_case *made) {
    struct probe_program *program = probe_run.program;
    uint64_t case_count = 1;
    for (uint32_t index = 0; index < program->flag_count; index++) {
        case_count *= program->flags[index].is_case_axis ? program->flags[index].value_count : 1u;
        PROBE_REQUIRE(case_count > 0 && case_count <= PROBE_MOST_CASES, "no value of a case flag, or too many cases");
    }
    uint32_t rest = case_index % (uint32_t)case_count, divisor = (uint32_t)case_count;
    memset(made, 0, sizeof(*made));
    made->index = case_index;
    for (uint32_t index = 0; index < program->flag_count; index++) {
        const struct probe_flag *flag = &program->flags[index];
        if (!flag->is_case_axis) { continue; }
        PROBE_REQUIRE(made->axis_count < PROBE_MOST_CASE_AXES, "more case flags than PROBE_MOST_CASE_AXES");
        divisor /= flag->value_count;
        uint32_t chosen = rest / divisor;
        rest %= divisor;
        made->axes[made->axis_count] = flag;
        made->values[made->axis_count++] = flag->values[chosen];
        probe_append(made->name, sizeof(made->name), ",", made->name[0] != '\0' ? 1u : 0u);
        for (const char *letter = flag->name + strspn(flag->name, "-"); *letter != '\0'; letter++) {
            probe_append(made->name, sizeof(made->name), *letter == '-' ? "_" : letter, 1);
        }
        probe_append(made->name, sizeof(made->name), ":", 1);
        probe_append(made->name, sizeof(made->name), flag->tokens[chosen], flag->token_bytes[chosen]);
    }
    probe_append(made->name, sizeof(made->name), "all", made->axis_count == 0 ? strlen("all") : 0u);
    return (uint32_t)case_count;
}
/* ---- time, numbers, statistics ---- */
PROBE_FUNCTION double probe_monotonic_seconds(void) {
    struct timespec now;
    PROBE_REQUIRE(clock_gettime(CLOCK_MONOTONIC, &now) == 0, "the monotonic clock cannot be read");
    return (double)now.tv_sec + (double)now.tv_nsec / PROBE_NANOSECONDS_PER_SECOND;
}
/* Local wall-clock time as YYYY-MM-DDTHH:MM:SS.mmm: what the batch joins with its clock samples. */
PROBE_FUNCTION void probe_stamp(char *text, size_t text_bytes) {
    struct timespec now;
    struct tm parts;
    PROBE_REQUIRE(clock_gettime(CLOCK_REALTIME, &now) == 0 && localtime_r(&now.tv_sec, &parts) != NULL,
                  "the wall clock cannot be read");
    size_t used = strftime(text, text_bytes, "%Y-%m-%dT%H:%M:%S", &parts);
    int added = used == 0 ? -1 : snprintf(text + used, text_bytes - used, ".%03ld",
                                          (long)(now.tv_nsec / PROBE_NANOSECONDS_PER_MILLISECOND));
    PROBE_REQUIRE(added >= 0 && (size_t)added < text_bytes - used, "a time stamp does not fit PROBE_STAMP_BYTES");
}
PROBE_FUNCTION int probe_compare_doubles(const void *first_pointer, const void *second_pointer) {
    double first = *(const double *)first_pointer, second = *(const double *)second_pointer;
    return (first > second) - (first < second);
}
/* The median of values, which it sorts in place; PROBE_NAN of none. */
PROBE_FUNCTION double probe_median(double *values, uint64_t count) {
    if (count == 0) { return PROBE_NAN; }
    qsort(values, (size_t)count, sizeof(double), probe_compare_doubles);
    return (values[(count - 1u) / PROBE_HALF] + values[count / PROBE_HALF]) / (double)PROBE_HALF;
}
/* Sorts the values in place. The 99th percentile is by nearest rank. Every figure of no value is PROBE_NAN. */
PROBE_FUNCTION void probe_statistics_of(double *values, uint64_t count, struct probe_statistics *statistics) {
    uint64_t rank = (count * PROBE_PERCENTILE + PROBE_PERCENT - 1u) / PROBE_PERCENT;
    double sum = 0.0;
    statistics->count = count;
    statistics->median = probe_median(values, count);
    for (uint64_t index = 0; index < count; index++) { sum += values[index]; }
    statistics->percentile_99 = count == 0 ? PROBE_NAN : values[rank - 1u];
    statistics->minimum = count == 0 ? PROBE_NAN : values[0];
    statistics->mean = count == 0 ? PROBE_NAN : sum / (double)count;
}
/* ---- output lines and checks ---- */
PROBE_FUNCTION void probe_print_result(const struct probe_result *result) {
    const struct probe_case_facts *facts = &result->facts;
    double megabytes = facts->megabytes > 0.0 ? facts->megabytes : PROBE_NAN;
    printf("%s batch=%s probe=%s build=%s flags=%s arch=%s card=%s format=%s part=%s rows=%s experts=%s overlap=%s"
           " variant=%s class=%c%s mode=%s n=%" PRIu64,
           PROBE_DIAGNOSTICS || PROBE_PROFILE || result->short_run ? "UNTIMED" : "RESULT", probe_flag_text("--batch"),
           probe_run.program->probe_name, PROBE_BUILD, PROBE_FMAD ? "fmad" : "exact", probe_run.card.architecture,
           probe_run.card_label, facts->format, facts->part, probe_whole(facts->rows), probe_whole(facts->experts),
           probe_whole(facts->overlap), result->variant, result->contract_class, result->is_bound ? " role=bound" : "",
           result->mode, result->step.count);
    if (!isnan(result->soak_seconds)) { printf(" seconds=%s", probe_number(result->soak_seconds, 0)); }
    printf(" median_us=%s p99_us=%s min_us=%s mean_us=%s ratio=%s paired_us=%s same_sign=%s stream_us=%s mb=%s"
           " gbps=%s registers=%s spill_bytes=%s kcyc=na clock_mhz=na power_w=na uj=na checks=%s\n",
           probe_number(result->step.median, 1), probe_number(result->step.percentile_99, 1),
           probe_number(result->step.minimum, 1), probe_number(result->step.mean, 1), probe_ratio(result->ratio),
           probe_number(result->paired_microseconds, 1), probe_pairs(result->same_sign, result->rounds),
           probe_number(result->stream_microseconds, 1), probe_number(megabytes, 1),
           probe_number(megabytes * PROBE_GBPS_PER_MB_PER_US / result->step.median, 0), probe_whole(result->registers),
           probe_whole(result->spill_bytes), probe_run.failures == 0 ? "ok" : "FAILED");
}
/* The end of every CHECK line: whether it failed, against whether it had to. */
PROBE_FUNCTION void probe_check_close(int failed, int must_fail) {
    printf(" verdict=%s%s\n", failed ? (must_fail ? "reported" : "FAILED") : (must_fail ? "FAILED" : "ok"),
           must_fail ? " required=report" : "");
    probe_run.failures += (failed != 0) != (must_fail != 0) ? 1u : 0u;
    probe_run.reports += failed && must_fail ? 1u : 0u;
}
/* A check that counts: `differing` of `of` words, bytes or values are not what they must be. must_differ: a break
 * flag requires this check to report (differing above 0): for a launch that must be refused, 1 of 1 when it was. */
PROBE_FUNCTION void probe_check_count(const char *case_name, const char *variant, const char *check_name,
                                      uint64_t differing, uint64_t of, int must_differ) {
    printf("CHECK card=%s case=%s variant=%s name=%s differing=%" PRIu64 " of=%" PRIu64, probe_run.card_label,
           case_name, variant, check_name, differing, of);
    probe_check_close(differing != 0, must_differ);
}
/* A check with a tolerance: the error over the rms of the reference; a NaN is a failure. */
PROBE_FUNCTION void probe_check_error(const char *case_name, const char *variant, const char *check_name,
                                      double error_over_rms, double limit, int must_exceed) {
    printf("CHECK card=%s case=%s variant=%s name=%s error_over_rms=%.1e limit=%.1e", probe_run.card_label,
           case_name, variant, check_name, error_over_rms, limit);
    probe_check_close(!(error_over_rms <= limit), must_exceed);
}
PROBE_FUNCTION void probe_digest_add(const void *bytes, uint64_t byte_count) {
    const uint8_t *byte = (const uint8_t *)bytes;
    probe_run.digest = probe_run.digest_bytes == 0 ? PROBE_DIGEST_OFFSET : probe_run.digest;
    for (uint64_t index = 0; index < byte_count; index++) {
        probe_run.digest = (probe_run.digest ^ byte[index]) * PROBE_DIGEST_PRIME;
    }
    probe_run.digest_bytes += byte_count;
}
PROBE_FUNCTION uint32_t probe_instance_index(const char *variant) {     /* instance_count: no such instance */
    struct probe_program *program = probe_run.program;
    uint32_t index = 0;
    while (index < program->instance_count && strcmp(program->instances[index].name, variant) != 0) { index++; }
    return index;
}
/* In configure(): the instance is a bound, a part of the work alone (only the reads, only the products) that shows
 * what the whole could reach. It is timed and paired as any other, is never BEST, and its INSTANCE and RESULT lines
 * say role=bound, which round_fastest_variants leaves out. Neither instance of the baseline pair can be one. */
PROBE_FUNCTION void probe_mark_bound(uint32_t instance) {
    PROBE_REQUIRE(probe_run.program != NULL && instance > PROBE_BASELINE_AGAIN && instance < PROBE_MOST_INSTANCES
                  && instance < probe_run.program->instance_count, "a bound of the baseline pair or past the table");
    probe_run.is_bound[instance] = 1;
}
/* One acceptance figure, by CONVENTIONS 3.3. A bar is "ratio at most bar_ratio" to the same run's baseline and
 * bar_microseconds what it stands for; PROBE_NAN as the bar: the package only measures. Undecided: a ratio that is
 * no number (not timed); a bar no farther from 1 than the noise floor; and a verdict that puts the ratio beyond the
 * bar as seen from 1 (met under a bar below 1, not_met over one above 1: it says the variant differs from the
 * baseline) unless same_sign of rounds, as its RESULT line has them, are 17 of 20 or that share of more rounds. */
PROBE_FUNCTION void probe_verdict(const char *figure, double bar_ratio, double bar_microseconds, double ratio,
                                  double noise_ratio, uint32_t same_sign, uint32_t rounds) {
    const char *verdict = "untimed";             /* a debug or profile build, or after --steps: never a verdict */
    uint64_t signs = (uint64_t)same_sign * PROBE_PAIRS_OF, needed = (uint64_t)rounds * PROBE_PAIRS_NEEDED;
    int reached = ratio <= bar_ratio, differs = rounds >= PROBE_PAIRS_OF && signs >= needed;
    if (!PROBE_DIAGNOSTICS && !PROBE_PROFILE && !probe_flag_given(PROBE_FLAG_STEPS)) {
        verdict = isnan(bar_ratio) ? "measured"
                  : isnan(ratio) || isnan(noise_ratio) || fabs(bar_ratio - 1.0) <= fabs(noise_ratio - 1.0)
                    || (reached == (bar_ratio < 1.0) && !differs) ? "undecided"
                  : reached ? "met" : "not_met";
    }
    printf("VERDICT card=%s figure=%s bar_ratio=%s bar_us=%s ratio=%s noise_ratio=%s same_sign=%s verdict=%s\n",
           probe_run.card_label, figure, probe_ratio(bar_ratio), probe_number(bar_microseconds, 1),
           probe_ratio(ratio), probe_ratio(noise_ratio), probe_pairs(same_sign, rounds), verdict);
}
/* The pairs of CONVENTIONS 3.3: per round, judged's step median less other's; their median (paired_microseconds) and
 * the rounds whose difference has its sign (same_sign of rounds), into the result. scratch: `rounds` values. */
PROBE_FUNCTION void probe_pair_rounds(struct probe_result *result, const double *judged, const double *other,
                                      uint32_t rounds, double *scratch) {
    result->same_sign = 0;
    result->rounds = rounds;
    for (uint32_t round_index = 0; round_index < rounds; round_index++) {
        scratch[round_index] = judged[round_index] - other[round_index];
    }
    result->paired_microseconds = probe_median(scratch, rounds);
    for (uint32_t round_index = 0; round_index < rounds; round_index++) {
        double difference = scratch[round_index], paired = result->paired_microseconds;
        result->same_sign += (difference > 0.0) == (paired > 0.0) && (difference < 0.0) == (paired < 0.0) ? 1u : 0u;
    }
}
/* probe_verdict of a bar between two instances, either one the baseline or neither: "judged's step median at most
 * bar_ratio times against's" (bar_us: at against's median), with the pairs of the two as a RESULT line has an
 * instance's against the baseline. One not timed in step mode, or one instance twice: ratio na, no pairs. */
PROBE_FUNCTION void probe_verdict_between(const char *figure, double bar_ratio, const struct probe_result *judged,
                                          const struct probe_result *against, double noise_ratio) {
    struct probe_result pair;
    uint32_t rounds = judged != against && judged->round_count == against->round_count ? judged->round_count : 0u;
    double *scratch = (double *)calloc((size_t)rounds + 1u, sizeof(double));         /* + 1: never calloc of 0 */
    PROBE_REQUIRE(scratch != NULL && (rounds == 0 || (judged->round_medians != NULL && against->round_medians != NULL)),
                  "out of host memory for the pairs, or a result's round count without its medians");
    probe_pair_rounds(&pair, judged->round_medians, against->round_medians, rounds, scratch);
    free(scratch);
    probe_verdict(figure, bar_ratio, bar_ratio * against->step.median,
                  rounds > 0 ? judged->step.median / against->step.median : PROBE_NAN, noise_ratio, pair.same_sign,
                  pair.rounds);
}
/* ---- check 10: every launch validated on the host, in every build ----
 * A probe starts with `int valid = 1;`, passes &valid to each check of the launch, and launches only if it is still
 * 1. A check that does not hold prints a REFUSED line and clears it. First: a value a kernel indexes or loops by. */
PROBE_FUNCTION void probe_launch_value(int *valid, const char *kernel, const char *what, uint64_t value,
                                       uint64_t least, uint64_t most) {
    if (value < least || value > most) {
        printf("REFUSED card=%s kernel=%s what=\"%s\" value=%" PRIu64 " least=%" PRIu64 " most=%" PRIu64 "\n",
               probe_run.card_label, kernel, what, value, least, most);
        *valid = 0;
    }
}
/* Elements first_element .. first_element + element_count - 1, of element_bytes each, must lie inside an array of
 * array_bytes: no product or sum that can wrap. An empty range and an array of 0 bytes are refused. The line says
 * how many elements were asked from first_element on and how many the array has from there. */
PROBE_FUNCTION void probe_launch_range(int *valid, const char *kernel, const char *array_name, uint64_t first_element,
                                       uint64_t element_count, uint64_t element_bytes, uint64_t array_bytes) {
    uint64_t held = element_bytes == 0 ? 0 : array_bytes / element_bytes;
    probe_launch_value(valid, kernel, array_name, element_count, 1, first_element > held ? 0 : held - first_element);
}
/* Grid and block inside the card's limits; the block the instance's own; its bound not above what an SM holds.
 * instance NULL: a kernel of the step that is no instance's own (a quantization, a row sum). */
PROBE_FUNCTION void probe_launch_geometry(int *valid, const char *kernel, const struct probe_instance *instance,
                                          uint64_t grid_x, uint64_t grid_y, uint64_t grid_z, uint64_t block_threads,
                                          uint64_t shared_bytes) {
    const struct probe_card_facts *card = &probe_run.card;
    uint64_t own_threads = instance != NULL && instance->block_threads != 0 ? instance->block_threads : block_threads;
    uint64_t bound_threads = instance != NULL && instance->bound_threads != 0 ? instance->bound_threads : block_threads;
    uint64_t bound_asks = instance != NULL ? (uint64_t)instance->bound_threads * instance->bound_blocks : 0;
    probe_launch_value(valid, kernel, "grid x", grid_x, 1, card->grid_most[0]);
    probe_launch_value(valid, kernel, "grid y", grid_y, 1, card->grid_most[1]);
    probe_launch_value(valid, kernel, "grid z", grid_z, 1, card->grid_most[PROBE_GRID_AXES - 1]);
    probe_launch_value(valid, kernel, "threads of a block", block_threads, 1, card->block_threads_most);
    probe_launch_value(valid, kernel, "shared bytes of a block", shared_bytes, 0, card->shared_bytes_most);
    probe_launch_value(valid, kernel, "threads of the instance's block", block_threads, own_threads, own_threads);
    probe_launch_value(valid, kernel, "threads against the launch bound", block_threads, 1, bound_threads);
    probe_launch_value(valid, kernel, "threads the launch bound asks of an SM", bound_asks, 0,
                       (uint64_t)card->warps_per_sm * PROBE_LANES_PER_WARP);
}
/* Debug only. An index formed for a card array, checked where it is used (the arguments have no side effect); and,
 * in a .cu, after a launch: the wait for the card, then every guard verified. */
#if PROBE_DIAGNOSTICS
#define PROBE_INDEX_CHECK(index, count) \
    PROBE_REQUIRE((uint64_t)(index) < (uint64_t)(count), "an index outside its array: " #index " of " #count)
#define PROBE_AFTER_LAUNCH() \
    (PROBE_CHECK_CUDA(cudaGetLastError()), PROBE_CHECK_CUDA(cudaDeviceSynchronize()), \
     PROBE_REQUIRE(probe_cuda_guards_differing() == 0, "a launch wrote behind a card array"))
#else
#define PROBE_INDEX_CHECK(index, count) ((void)0)
#define PROBE_AFTER_LAUNCH() ((void)0)
#endif
/* ---- the assembler's report: registers and spill bytes of an instance's kernels, from <executable>.registers ---- */
PROBE_FUNCTION void probe_read_registers(const char *executable) {       /* without the file they stay unknown */
    struct probe_program *program = probe_run.program;
    char path[PROBE_PATH_BYTES], line[PROBE_PATH_BYTES], kernel[PROBE_NAME_BYTES];
    int written = snprintf(path, sizeof(path), "%s.registers", executable), registers = 0;
    long long spill_bytes = 0;
    FILE *report = written < 0 || (size_t)written >= sizeof(path) ? NULL : fopen(path, "r");
    for (uint32_t index = 0; index < program->instance_count; index++) {
        program->instances[index].registers = -1;
        program->instances[index].spill_bytes = -1;
    }
    while (report != NULL && fgets(line, sizeof(line), report) != NULL) {
        int fields = sscanf(line, PROBE_REGISTERS_SCAN, kernel, &registers, &spill_bytes);
        for (uint32_t index = 0; fields == PROBE_REGISTERS_FIELDS && index < program->instance_count; index++) {
            struct probe_instance *instance = &program->instances[index];
            const char *list = instance->kernel, *named = list == NULL ? NULL : strstr(list, kernel);
            for (size_t length = strlen(kernel); named != NULL; named = strstr(named + 1, kernel)) {
                if ((named == list || named[-1] == '+') && (named[length] == '\0' || named[length] == '+')) {
                    instance->registers = registers > instance->registers ? registers : instance->registers;
                    instance->spill_bytes = (instance->spill_bytes < 0 ? 0 : instance->spill_bytes) + spill_bytes;
                }
            }
        }
    }
    PROBE_REQUIRE(report == NULL || fclose(report) == 0, "the registers file cannot be closed");
}
PROBE_FUNCTION int probe_list_instances(void) {
    struct probe_program *program = probe_run.program;
    for (uint32_t index = 0; index < program->instance_count; index++) {
        const struct probe_instance *instance = &program->instances[index];
        printf("INSTANCE probe=%s variant=%s class=%c%s kernel=%s block_threads=%u bound=%ux%u available=%d"
               " registers=%s spill_bytes=%s\n", program->probe_name, instance->name, instance->contract_class,
               probe_run.is_bound[index] ? " role=bound" : "", instance->kernel != NULL ? instance->kernel : "na",
               instance->block_threads, instance->bound_threads, instance->bound_blocks, instance->available,
               probe_whole(instance->registers), probe_whole(instance->spill_bytes));
    }
    int spills = program->instance_count > 0 && program->instances[PROBE_BASELINE].spill_bytes > 0;
    fprintf(stderr, "%s", spills ? "the baseline instance spills: CONVENTIONS 1.1 fails this build\n" : "");
    return spills ? EXIT_FAILURE : EXIT_SUCCESS;
}
/* ---- the timing protocol; first the ROUND line: a case's span of time, for the join with the clock samples ---- */
PROBE_FUNCTION void probe_print_round(const struct probe_case *probe_case, uint32_t round_index, const char *begin,
                                      const char *end, double baseline_microseconds) {
    printf("ROUND card=%s case=%s round=%u stamp=%s end=%s baseline_us=%s\n", probe_run.card_label, probe_case->name,
           round_index, begin, end, probe_number(baseline_microseconds, 1));
    PROBE_REQUIRE(fflush(stdout) == 0, "the output cannot be written");
}
/* step_count steps prepared (not timed), then queued back to back between the two marks: microseconds per step.
 * One step is `step` mode, PROBE_STREAM_STEPS are a stream. PROBE_NAN: a launch was refused, which fails the run. */
PROBE_FUNCTION double probe_timed_steps(const struct probe_case *probe_case, uint32_t instance, uint32_t step_count) {
    struct probe_program *program = probe_run.program;
    for (uint32_t slot = 0; slot < step_count; slot++) {
        if (!program->prepare_step(program->context, probe_case, instance, probe_run.steps + slot, slot)) {
            printf("REFUSED card=%s kernel=na what=\"a timed step of %s\" value=0 least=1 most=1\n",
                   probe_run.card_label, program->instances[instance].name);
            probe_run.failures++;
            return PROBE_NAN;
        }
    }
    program->mark_start(program->context);
    for (uint32_t slot = 0; slot < step_count; slot++) {
        probe_run.launches += program->launch_step(program->context, probe_case, instance, slot);
    }
    double microseconds = program->elapsed_microseconds(program->context);
    probe_run.steps += step_count;
    return microseconds / (double)step_count;
}
/* The instances a run times: all that are available, or the baseline pair with those --instance names; a soak
 * takes the named one alone, the baseline when none is named. */
PROBE_FUNCTION uint32_t probe_select_instances(uint32_t *selected, int soak) {
    struct probe_program *program = probe_run.program;
    uint32_t count = 0;
    for (uint32_t index = 0; index < program->instance_count; index++) {
        int wanted = probe_run.selected_count == 0 ? !soak || index == PROBE_BASELINE
                                                   : !soak && index <= PROBE_BASELINE_AGAIN;
        for (uint32_t name = 0; name < probe_run.selected_count; name++) {
            wanted |= strcmp(probe_run.selected_names[name], program->instances[index].name) == 0;
        }
        if (wanted && program->instances[index].available) { selected[count++] = index; }
    }
    return count;
}
PROBE_FUNCTION void probe_fill_result(struct probe_result *result, const struct probe_case_facts *facts,
                                      uint32_t instance_index, const char *mode) {
    const struct probe_instance *instance = &probe_run.program->instances[instance_index];
    memset(result, 0, sizeof(*result));
    result->variant = instance->name;
    result->mode = mode;
    result->contract_class = instance->contract_class;
    result->short_run = probe_flag_given(PROBE_FLAG_STEPS);
    result->facts = *facts;
    result->registers = instance->registers;
    result->spill_bytes = instance->spill_bytes;
    result->is_bound = probe_run.is_bound[instance_index];
    result->ratio = result->paired_microseconds = result->stream_microseconds = result->soak_seconds = PROBE_NAN;
    probe_statistics_of(NULL, 0, &result->step);
}
/* One case. In rounds (CONVENTIONS 3.3): every figure of a variant is from this run, beside its baseline. As a soak
 * (CONVENTIONS 3.6): one instance alone, a round is one stream, and the rounds end when the seconds are over. */
PROBE_FUNCTION void probe_time_case(const struct probe_case *probe_case, const struct probe_case_facts *facts) {
    struct probe_program *program = probe_run.program;
    int short_run = probe_flag_given(PROBE_FLAG_STEPS), soak = probe_flag_given(PROBE_FLAG_SOAK_SECONDS);
    int step_mode = !soak && probe_flag_has_word(PROBE_FLAG_MODE, PROBE_MODE_STEP);
    int stream_mode = soak || probe_flag_has_word(PROBE_FLAG_MODE, PROBE_MODE_STREAM);
    uint32_t selected[PROBE_MOST_INSTANCES], selected_count = probe_select_instances(selected, soak), best = PROBE_NONE;
    uint32_t rounds = soak ? PROBE_MOST_STEPS : short_run ? 1u : (uint32_t)probe_flag_number("--rounds", 0);
    uint32_t steps = soak ? 1u : (uint32_t)probe_flag_number(short_run ? PROBE_FLAG_STEPS : "--steps-per-round", 0);
    uint32_t stream_steps = short_run && steps < PROBE_STREAM_STEPS ? steps : PROBE_STREAM_STEPS;
    size_t stride = rounds, per_instance = stride * steps, per_round = selected_count * stride;
    /* one block: [selected][round][step] step times, [selected][round] medians and stream times, a scratch row */
    double *step_times =
        (double *)calloc(selected_count * per_instance + per_round + per_round + steps + stride, sizeof(double));
    double *round_medians = step_times + selected_count * per_instance, *stream_times = round_medians + per_round;
    double *scratch = stream_times + per_round, noise_ratio = PROBE_NAN, begun = probe_monotonic_seconds();
    uint64_t soak_whole_seconds = soak ? probe_flag_number(PROBE_FLAG_SOAK_SECONDS, 0) : 0;
    double soak_seconds = soak ? (double)soak_whole_seconds : PROBE_NAN;
    struct probe_result *results = (struct probe_result *)calloc(program->instance_count, sizeof(struct probe_result));
    char begin[PROBE_STAMP_BYTES] = "", end[PROBE_STAMP_BYTES] = "";
    int held = selected_count > 0;
    PROBE_REQUIRE(step_times != NULL && results != NULL, "out of host memory for the times");
    /* CONVENTIONS 3.2: the card at its clock before the first figure: the sequence that follows, not timed */
    while (held && !short_run && probe_monotonic_seconds() - begun < PROBE_WARM_UP_SECONDS) {
        for (uint32_t index = 0; held && index < selected_count; index++) {
            held = !isnan(probe_timed_steps(probe_case, selected[index], soak ? stream_steps : 1u));
        }
    }
    begun = probe_monotonic_seconds();
    /* The instance that opens a round starts one place later each round, so that over the rounds every instance
     * holds every place: on a CMP 170HX at its cap the first instance of a round ran 0.4 to 4.7% slower than the one
     * after it (i05, every case, baseline against baseline_again), which a fixed order gives to the baseline alone. */
    for (uint32_t round_index = 0; held && round_index < rounds; round_index++) {
        if (!soak || round_index == 0) { probe_stamp(begin, sizeof(begin)); }
        for (uint32_t place = 0; held && step_mode && place < selected_count; place++) {
            uint32_t index = (place + round_index) % selected_count;
            double *times = step_times + index * per_instance + (size_t)round_index * steps;
            for (uint32_t step = 0; held && step < steps; step++) {
                times[step] = probe_timed_steps(probe_case, selected[index], 1);
                held = !isnan(times[step]);
            }
            memcpy(scratch, times, steps * sizeof(double));
            round_medians[index * stride + round_index] = probe_median(scratch, steps);
        }
        for (uint32_t place = 0; held && stream_mode && place < selected_count; place++) {
            uint32_t index = (place + round_index) % selected_count;
            stream_times[index * stride + round_index] = probe_timed_steps(probe_case, selected[index], stream_steps);
            held = !isnan(stream_times[index * stride + round_index]);
        }
        probe_stamp(end, sizeof(end));
        if (soak && probe_monotonic_seconds() - begun >= soak_seconds) { rounds = round_index + 1u; }
        if (soak && round_index + 1u < rounds) { continue; }       /* a soak: one line, for its whole window */
        probe_print_round(probe_case, round_index, begin, end, step_mode ? round_medians[round_index] : PROBE_NAN);
    }
    soak_seconds = soak ? probe_monotonic_seconds() - begun : soak_seconds;     /* seconds=: what the window lasted */
    for (uint32_t index = 0; held && index < selected_count; index++) {
        struct probe_result *result = &results[selected[index]];
        const double *medians = round_medians + index * stride;
        probe_fill_result(result, facts, selected[index],
                          soak ? "soak" : step_mode ? PROBE_MODE_STEP : PROBE_MODE_STREAM);
        if (step_mode) {                         /* paired with the baseline (its own paired_us 0, its pairs none) */
            probe_statistics_of(step_times + index * per_instance, per_instance, &result->step);
            result->ratio = result->step.median / results[PROBE_BASELINE].step.median;
            probe_pair_rounds(result, medians, round_medians, rounds, scratch);
            result->same_sign = index != 0 ? result->same_sign : 0u;
            result->rounds = index != 0 ? rounds : 0u;
            result->round_medians = medians;     /* for probe_verdict_between, until the case's times are freed */
            result->round_count = rounds;
        }
        if (stream_mode) {
            memcpy(scratch, stream_times + index * stride, rounds * sizeof(double));
            result->stream_microseconds = probe_median(scratch, rounds);
        }
        if (soak) {                              /* the figures of a soak are its streams', per step */
            probe_statistics_of(scratch, rounds, &result->step);
            result->step.count = (uint64_t)rounds * stream_steps;
            result->stream_microseconds = result->step.mean;
            result->soak_seconds = soak_seconds;
        }
        if (step_mode && selected[index] == PROBE_BASELINE_AGAIN) {
            double baseline_median = results[PROBE_BASELINE].step.median;
            noise_ratio = result->ratio;
            printf("NOISE card=%s case=%s baseline_us=%s baseline_again_us=%s difference_us=%s ratio=%s\n",
                   probe_run.card_label, probe_case->name, probe_number(baseline_median, 1),
                   probe_number(result->step.median, 1), probe_number(result->step.median - baseline_median, 1),
                   probe_ratio(noise_ratio));
        }
    }
    for (uint32_t index = 0; held && index < program->instance_count; index++) {
        struct probe_result *result = &results[index];
        int timed = result->mode != NULL, refused = !timed && !soak && !program->instances[index].available;
        if (!timed) { probe_fill_result(result, facts, index, refused ? PROBE_MODE_STEP : NULL); }      /* figures na */
        if (result->mode != NULL) { probe_print_result(result); }
        result->mode = timed ? result->mode : NULL;                  /* what verdicts() reads: not timed, ratio na */
        if (timed && index != PROBE_BASELINE_AGAIN && result->spill_bytes <= 0 && !result->is_bound
            && !isnan(result->step.median) && (best == PROBE_NONE || result->step.median < results[best].step.median)) {
            best = index;                        /* never the baseline a second time, never a spill, never a bound */
        }
    }
    if (best != PROBE_NONE && !soak && !short_run && !PROBE_DIAGNOSTICS && !PROBE_PROFILE) {
        printf("BEST card=%s case=%s variant=%s\n", probe_run.card_label, probe_case->name, results[best].variant);
    }
    if (held && !soak && program->verdicts != NULL) {
        program->verdicts(program->context, probe_case, results, noise_ratio);
    }
    free(step_times);
    free(results);
}
/* For the profiler: only this variant's step, no check; the launches of one step are printed for its skip count. */
PROBE_FUNCTION void probe_profile_case(const struct probe_case *probe_case) {
    uint32_t instance = probe_instance_index(probe_flag_text(PROBE_FLAG_PROFILE_CASE));
    uint32_t steps = probe_flag_given(PROBE_FLAG_STEPS) ? (uint32_t)probe_flag_number(PROBE_FLAG_STEPS, 0)
                                                        : PROBE_PROFILE_STEPS;
    uint64_t launches_before = probe_run.launches;
    for (uint32_t step = 0; step < steps; step++) {
        if (isnan(probe_timed_steps(probe_case, instance, 1))) { return; }
    }
    printf("LAUNCHES card=%s case=%s variant=%s per_step=%" PRIu64 " steps=%u\n", probe_run.card_label,
           probe_case->name, probe_run.program->instances[instance].name,
           (probe_run.launches - launches_before) / steps, steps);
}
/* What the instance table and the instance flags must satisfy before anything runs. 0: refused, with the reason. */
PROBE_FUNCTION int probe_program_is_sound(void) {
    struct probe_program *program = probe_run.program;
    const struct probe_instance *instances = program->instances;
    const char *profiled = probe_flag_text(PROBE_FLAG_PROFILE_CASE);
    PROBE_REQUIRE(program->describe_case != NULL && program->instance_count <= PROBE_MOST_INSTANCES
                  && program->instance_count != PROBE_BASELINE_AGAIN && (program->instance_count == 0
                      || (strcmp(instances[PROBE_BASELINE].name, "baseline") == 0 && instances[PROBE_BASELINE].available
                          && strcmp(instances[PROBE_BASELINE_AGAIN].name, "baseline_again") == 0
                          && instances[PROBE_BASELINE_AGAIN].available && program->prepare_step != NULL
                          && program->launch_step != NULL && program->mark_start != NULL
                          && program->elapsed_microseconds != NULL)),
                  "the program's table: instance 0 baseline, instance 1 baseline_again, the timing callbacks");
    for (uint32_t name = 0; name <= probe_run.selected_count; name++) {
        const char *wanted = name < probe_run.selected_count ? probe_run.selected_names[name] : profiled;
        uint32_t index = probe_instance_index(wanted);
        int alone = name == 0 || !probe_flag_given(PROBE_FLAG_SOAK_SECONDS) || wanted == profiled;
        if (wanted[0] != '\0' && (!alone || index >= program->instance_count || !instances[index].available)) {
            fprintf(stderr, "%s: no such instance to time or to profile (--list-instances names them), or a second one "
                            "for a soak, where an instance is alone\n", wanted);
            return 0;
        }
    }
    return 1;
}
PROBE_FUNCTION int probe_main(int argument_count, char **arguments, struct probe_program *program) {
    struct probe_card_facts *card = &probe_run.card;
    struct probe_case probe_case;
    probe_run.program = program;
    if (!probe_read_flags(argument_count, arguments)
        || (program->configure != NULL && !program->configure(program->context)) || !probe_program_is_sound()) {
        return EXIT_FAILURE;
    }
    probe_read_registers(arguments[0]);
    if (probe_flag_given("--list-instances")) { return probe_list_instances(); }
    memcpy(probe_run.card_label, "host", sizeof("host"));
    card->architecture = "host";
    if (program->open_card != NULL) {
        if (!program->open_card(program->context, card)) { return EXIT_FAILURE; }
        double l2_mib = (double)card->l2_bytes / PROBE_BYTES_PER_MIB;
        double ring_mib = card->ring_bytes == 0 ? PROBE_NAN : (double)card->ring_bytes / PROBE_BYTES_PER_MIB;
        PROBE_REQUIRE(snprintf(probe_run.card_label, sizeof(probe_run.card_label), "%" PRIu64,
                               probe_flag_number("--card-index", 0)) > 0, "the card's index cannot be printed");
        printf("CARD card=%s name=\"%s\" arch=%s sms=%d l2_mib=%s ring_mib=%s ring_l2=%s\n", probe_run.card_label,
               card->name, card->architecture, card->sm_count, probe_number(l2_mib, 0), probe_number(ring_mib, 0),
               probe_number(ring_mib / l2_mib, 1));
    }
    for (uint32_t case_index = 0; case_index < probe_make_case(case_index, &probe_case); case_index++) {
        struct probe_case_facts facts;
        memset(&facts, 0, sizeof(facts));
        program->describe_case(program->context, &probe_case, &facts);
        if (probe_flag_text(PROBE_FLAG_PROFILE_CASE)[0] != '\0') {
            probe_profile_case(&probe_case);
            break;
        }
        if (program->check_case != NULL) { program->check_case(program->context, &probe_case); }
        if (probe_run.breaking) { break; }       /* what a break flag changed stays changed: one case shows it */
        if (program->time_case != NULL) { program->time_case(program->context, &probe_case, &facts); }
        if (program->time_case == NULL && program->instance_count > 0) { probe_time_case(&probe_case, &facts); }
    }
    if (probe_run.digest_bytes > 0) {
        printf("DIGEST card=%s arch=%s probe=%s bytes=%" PRIu64 " value=%016" PRIx64 "\n", probe_run.card_label,
               card->architecture, program->probe_name, probe_run.digest_bytes, probe_run.digest);
    }
#if PROBE_DIAGNOSTICS
    printf("COUNTERS card=%s launches=%" PRIu64 " steps=%" PRIu64 "\n", probe_run.card_label, probe_run.launches,
           probe_run.steps);
#endif
    if (program->close_card != NULL) { program->close_card(program->context); }
    probe_run.failures += probe_run.breaking && probe_run.reports == 0 ? 1u : 0u;       /* a break flag broke nothing */
    printf("CHECKS card=%s failed=%u reported=%u verdict=%s\n", probe_run.card_label, probe_run.failures,
           probe_run.reports, probe_run.failures == 0 ? "ok" : "FAILED");
    return probe_run.failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
#include "probe_harness_cuda.h"      /* the card side, with the CUDA runtime only */
#endif /* PROBE_HARNESS_H */
