/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * cpu_probe_common.h: what the processor probes of the speed campaign share: cpu_floor_probe (round 1, unit 8a)
 * and, read only, cpu_expert_forms_probe (unit 8b). A probe is one translation unit, so everything here is static
 * inline; the build gives -D_GNU_SOURCE, -march=znver4 and -fopenmp (CONVENTIONS.md 1.3).
 *
 *   the cold ring   Synthetic experts in the checkpoint's 4-bit format, byte for byte those of cpu_expert_probe
 *                   (b04: same seeds): sign and 3-bit magnitude per weight, two weights to a byte, low nibble first,
 *                   one power-of-two scale per 32 weights of a row. 128 of them are 2.24 GiB, six times this
 *                   machine's third-level cache: an expert taken in turn is read from memory. Each matrix keeps
 *                   the sum of its bytes, taken by plain code when it was written: the reference of a read kernel.
 *   row kernels     b04's fp32 row product, and a row kernel that only reads. The static split of rows to threads.
 *   the reference   A weight decoded from the format's definition; a row product in double precision.
 *   placements      The hardware threads by third-level cache slice, core and sibling (sysfs), and three orders
 *                   of them (cpu_placement_order). A team of n threads sits on the first n of an order.
 *   a team          n pinned threads for one slot of a run: an OpenMP team, or the minimal pool of this file,
 *                   whose workers spin on a mailbox of their own. Both run the same share function between the
 *                   same two stamps per thread and region. A third kind is the OpenMP team of ONE thread with
 *                   the runtime's whole-machine team docked beside it: what cpu_expert_probe timed at one thread.
 *   and             the memory modules' temperatures, lists from a command line, quantiles, a run's statistics.
 *
 * What the minimal pool is not: it never sleeps (a team lives for one slot and is then joined), it has one way to
 * wait (a spin with PAUSE) and a deadline on the caller's wait only. Other ways to wait are ledger row C4. A mailbox
 * is two words under the library's spin lock, one cache line per worker; no result of a kernel passes through one:
 * a thread writes only rows and sums that are its own.
 *
 * Measurement tool: not part of the engine.
 */
#ifndef CPU_PROBE_COMMON_H
#define CPU_PROBE_COMMON_H

#ifndef _GNU_SOURCE
#error "build with -D_GNU_SOURCE (CONVENTIONS.md 1.3): thread affinity and per-thread usage are behind it"
#endif
#include <errno.h>
#include <immintrin.h>
#include <limits.h>
#include <math.h>
#include <omp.h>
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <time.h>

#ifndef PROBE_DIAGNOSTICS
#define PROBE_DIAGNOSTICS 0                      /* 1 in the debug and the thread-sanitizer build */
#endif
#if defined(__SANITIZE_THREAD__)                 /* that sanitizer cannot see the OpenMP runtime's barriers and calls */
#define CPU_OPENMP_HELPS 0                       /* every later read of what its threads wrote a race: a loop that */
#else                                            /* only prepares runs on the calling thread there, and a team is */
#define CPU_OPENMP_HELPS 1                       /* the pool of this file */
#endif

enum cpu_common_constant {
    /* the model's expert and the checkpoint's 4-bit format */
    CPU_HIDDEN_SIZE = 5120, CPU_EXPERT_INNER_SIZE = 2304, CPU_GATE = 0, CPU_UP, CPU_DOWN, CPU_MATRIX_KINDS,
    CPU_NIBBLES_PER_BYTE = 2, CPU_NIBBLE_BITS = 4, CPU_NIBBLE_MASK = 0x0F, CPU_SIGN_BIT = 0x08,
    CPU_MAGNITUDE_CODES = 8, CPU_NIBBLE_CODES = 16, CPU_WEIGHTS_PER_SCALE_GROUP = 32,
    CPU_BYTES_PER_SCALE_GROUP = CPU_WEIGHTS_PER_SCALE_GROUP / CPU_NIBBLES_PER_BYTE, CPU_E8M0_BIAS = 127,
    CPU_SCALE_CODES = 256, CPU_FLOATS_PER_VECTOR = 16, CPU_BYTES_PER_VECTOR = 64,
    /* the synthetic ring as cpu_expert_probe fills it: scales 2^-9 .. 2^-6 */
    CPU_SCALE_CODE_BASE = 118, CPU_SCALE_CODE_SPREAD = 4, CPU_EXPERT_SEED_SHIFT = 40, CPU_MATRIX_SEED_SHIFT = 36,
    CPU_MIX_FIRST_SHIFT = 30, CPU_MIX_SECOND_SHIFT = 27, CPU_MIX_THIRD_SHIFT = 31,
    /* the machine and a team */
    CPU_CACHE_LINE_BYTES = 64, CPU_STAMP_LINE_BYTES = 64, CPU_MAX_HARDWARE_THREADS = 1024,
    CPU_CORE_KEY_PACKAGE_STRIDE = 65536, CPU_PLACEMENT_KEY_SHIFT = 20, CPU_THIRD_LEVEL_CACHE = 3, CPU_NOWHERE = -1,
    CPU_TREE_ARITY = 2, CPU_TEAM_WAIT_SECONDS_MAX = 60, CPU_ROUNDS_MAX = 100,
    /* text and files */
    CPU_DECIMAL = 10, CPU_PATH_BYTES = 128, CPU_TEXT_BYTES = 256, CPU_STAMP_TEXT_BYTES = 40,
    CPU_SECONDS_TEXT_BYTES = 20,                 /* "2026-10-12T09:14:03" and its end */
    CPU_HWMON_NODES_SCANNED = 128, CPU_MILLIDEGREES_PER_DEGREE = 1000, CPU_NANOSECONDS_PER_MILLISECOND = 1000000
};
#define CPU_HUGE_PAGE_BYTES ((size_t)2 * 1024 * 1024)
#define CPU_MIX_FIRST_MULTIPLIER 0xbf58476d1ce4e5b9ull
#define CPU_MIX_SECOND_MULTIPLIER 0x94d049bb133111ebull
#define CPU_BYTES_PER_MB 1.0e6
#define CPU_BYTES_PER_MIB (1024.0 * 1024.0)
#define CPU_KIB_PER_MIB 1024.0
#define CPU_NANOSECONDS_PER_SECOND 1.0e9
#define CPU_MICROSECONDS_PER_SECOND 1.0e6
#define CPU_CALIBRATION_SECONDS 0.05             /* the time counter against the system's clock: good to 1e-6 */
#define CPU_MEDIAN 0.5
#define CPU_HIGH_QUANTILE 0.99
#define CPU_TEAM_WORKER_NAME "cpu_team_worker"   /* what a team's threads but its caller are called while it lives */
#define CPU_MODULE_SENSOR_NAME "spd5118"         /* the temperature sensor of a DDR5 module's hub, as hwmon names it */
#define CPU_SYSTEM_DIRECTORY "/sys/devices/system/cpu"

/* ---- small things ---- */

static inline uint64_t cpu_mix_bits(uint64_t value)
{
    value ^= value >> CPU_MIX_FIRST_SHIFT;
    value *= CPU_MIX_FIRST_MULTIPLIER;
    value ^= value >> CPU_MIX_SECOND_SHIFT;
    value *= CPU_MIX_SECOND_MULTIPLIER;
    value ^= value >> CPU_MIX_THIRD_SHIFT;
    return value;
}

static inline double cpu_seconds_now(void)
{
    struct timespec now = {0, 0};
    (void)clock_gettime(CLOCK_MONOTONIC_RAW, &now);          /* cannot fail for this clock and a valid address */
    return (double)now.tv_sec + (double)now.tv_nsec / CPU_NANOSECONDS_PER_SECOND;
}

/* The processor seconds of every thread this process has had: over an interval and divided by its length, the
 * number of threads that were busy in it, spinning ones included. */
static inline double cpu_process_seconds(void)
{
    struct timespec used = {0, 0};
    (void)clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &used);    /* cannot fail for this clock and a valid address */
    return (double)used.tv_sec + (double)used.tv_nsec / CPU_NANOSECONDS_PER_SECOND;
}

/* Ticks of the time-stamp counter per microsecond. Every core reads the same counter, so stamps taken on
 * different cores can be subtracted. */
static inline double cpu_ticks_per_microsecond(void)
{
    const double started = cpu_seconds_now();
    const uint64_t ticks_started = __rdtsc();
    double now = started;
    while (now - started < CPU_CALIBRATION_SECONDS) {
        now = cpu_seconds_now();
    }
    return (double)(__rdtsc() - ticks_started) / ((now - started) * CPU_MICROSECONDS_PER_SECOND);
}

/* The wall clock to a millisecond: what joins a round with whatever a batch sampled beside it. */
static inline void cpu_wall_clock_stamp(char *text, size_t text_bytes)
{
    struct timespec now = {0, 0};
    struct tm calendar;
    char seconds[CPU_SECONDS_TEXT_BYTES];
    text[0] = '\0';
    if (clock_gettime(CLOCK_REALTIME, &now) == 0 && localtime_r(&now.tv_sec, &calendar) != NULL &&
        strftime(seconds, sizeof(seconds), "%Y-%m-%dT%H:%M:%S", &calendar) != 0) {
        (void)snprintf(text, text_bytes, "%s.%03d", seconds, (int)(now.tv_nsec / CPU_NANOSECONDS_PER_MILLISECOND));
    }
}

static inline int cpu_compare_doubles(const void *left, const void *right)
{
    const double difference = *(const double *)left - *(const double *)right;
    return (difference > 0.0) - (difference < 0.0);
}

/* The value at a quantile of count values, which are sorted in place (0.5: the upper median). */
static inline double cpu_quantile(double *values, int count, double quantile)
{
    const int index = (int)(quantile * (double)count);
    if (count < 1) {
        return 0.0;
    }
    qsort(values, (size_t)count, sizeof(*values), cpu_compare_doubles);
    return values[index < count ? index : count - 1];
}

struct cpu_summary {                             /* the statistics of CONVENTIONS.md 3.3 over one variant's steps */
    double median, high, lowest, mean, round_median[CPU_ROUNDS_MAX];
};

/* microseconds[] holds `rounds` rounds (at most CPU_ROUNDS_MAX) of steps_per_round step times each; scratch holds
 * as many values. */
static inline void cpu_summarize(const double *microseconds, int rounds, int steps_per_round, double *scratch,
                                 struct cpu_summary *summary)
{
    const int steps = rounds * steps_per_round;
    summary->mean = 0.0;
    for (int round = 0; round < rounds; round++) {
        memcpy(scratch, microseconds + round * steps_per_round, (size_t)steps_per_round * sizeof(*scratch));
        summary->round_median[round] = cpu_quantile(scratch, steps_per_round, CPU_MEDIAN);
    }
    memcpy(scratch, microseconds, (size_t)steps * sizeof(*scratch));
    summary->median = cpu_quantile(scratch, steps, CPU_MEDIAN);
    summary->high = cpu_quantile(scratch, steps, CPU_HIGH_QUANTILE);
    summary->lowest = scratch[0];
    for (int step = 0; step < steps; step++) {
        summary->mean += microseconds[step] / (double)steps;
    }
}

/* "1,2,4" into numbers. Returns how many, or 0 when the text is not such a list within the bounds. */
static inline int cpu_parse_number_list(const char *text, long minimum, long maximum, int *numbers, int capacity)
{
    const char *cursor = text;
    int count = 0;
    while (*cursor != '\0') {
        char *end = NULL;
        errno = 0;
        const long number = strtol(cursor, &end, CPU_DECIMAL);
        if (end == cursor || errno != 0 || number < minimum || number > maximum || count == capacity ||
            (*end != ',' && *end != '\0') || (*end == ',' && end[1] == '\0')) {
            return 0;
        }
        numbers[count++] = (int)number;
        cursor = *end == ',' ? end + 1 : end;
    }
    return count;
}

/* One number within the bounds. Returns 0, and leaves *number alone, when the text is not that. */
static inline int cpu_parse_number(const char *text, long minimum, long maximum, int *number)
{
    int parsed = 0;
    const int right = cpu_parse_number_list(text, minimum, maximum, &parsed, 1) == 1;
    *number = right ? parsed : *number;
    return right;
}

/* "die,smt" into the indices of those names. Returns how many, or 0 when a word is not one of the names. */
static inline int cpu_parse_name_list(const char *text, const char *const *names, int name_count, int *chosen,
                                      int capacity)
{
    const char *cursor = text;
    int count = 0;
    while (*cursor != '\0') {
        const size_t length = strcspn(cursor, ",");
        int found = name_count;
        for (int name = 0; name < name_count; name++) {
            if (strlen(names[name]) == length && strncmp(cursor, names[name], length) == 0) {
                found = name;
            }
        }
        if (found == name_count || count == capacity || (cursor[length] == ',' && cursor[length + 1] == '\0')) {
            return 0;
        }
        chosen[count++] = found;
        cursor += length + (cursor[length] == ',' ? 1u : 0u);
    }
    return count;
}

/* A file that begins with one number (sysfs). Returns 0 when there is no such file or no number in it. */
static inline int cpu_read_number_file(const char *path, long *number)
{
    char text[CPU_TEXT_BYTES];
    char *end = text;
    FILE *file = fopen(path, "r");
    if (file == NULL) {
        return 0;
    }
    errno = 0;
    const long value = fgets(text, sizeof(text), file) != NULL ? strtol(text, &end, CPU_DECIMAL) : 0;
    (void)fclose(file);
    if (end == text || errno != 0) {
        return 0;
    }
    *number = value;
    return 1;
}

/* ---- the cold ring ---- */

struct cpu_matrix {                              /* output_size rows of input_size weights */
    uint8_t *nibbles;                            /* input_size / 2 bytes a row */
    uint8_t *scale_codes;                        /* input_size / 32 bytes a row */
    int output_size, input_size;
    uint64_t byte_sum;                           /* of both arrays, by plain code, when they were written */
};

struct cpu_expert {
    struct cpu_matrix matrix[CPU_MATRIX_KINDS];  /* gate and up of 2304 x 5120, down of 5120 x 2304 */
    int filled;                                  /* 0 where one of them could not be had */
};

struct cpu_ring {
    struct cpu_expert *experts;
    int expert_count;
    float scale_times_half[CPU_SCALE_CODES];     /* the fp32 kernel's: 2^(code - 127) / 2, its table holds 2 x value */
    double scale_of_code[CPU_SCALE_CODES];       /* the reference's: 2^(code - 127) */
};

static inline size_t cpu_row_nibble_bytes(const struct cpu_matrix *matrix)
{
    return (size_t)(matrix->input_size / CPU_NIBBLES_PER_BYTE);
}

static inline size_t cpu_row_scale_bytes(const struct cpu_matrix *matrix)
{
    return (size_t)(matrix->input_size / CPU_WEIGHTS_PER_SCALE_GROUP);
}

/* The bytes of an expert's three matrices and, with a non-null address, the sum of them all. */
static inline uint64_t cpu_expert_bytes(const struct cpu_expert *expert, uint64_t *byte_sum)
{
    uint64_t bytes = 0, sum = 0;
    for (int kind = 0; kind < CPU_MATRIX_KINDS; kind++) {
        const struct cpu_matrix *matrix = &expert->matrix[kind];
        bytes += (uint64_t)matrix->output_size * (cpu_row_nibble_bytes(matrix) + cpu_row_scale_bytes(matrix));
        sum += matrix->byte_sum;
    }
    if (byte_sum != NULL) {
        *byte_sum = sum;
    }
    return bytes;
}

/* Ends the OpenMP threads this program started: with OMP_WAIT_POLICY=active they would spin on their cores beside
 * whatever runs next. Returns 0 when the runtime refuses. gcc's runtime ends them and answers 0 each time, also
 * twice in a row; another runtime refused a call that no parallel region came before, so this is called after a
 * region. */
static inline int cpu_openmp_release_threads(void)
{
    return omp_pause_resource_all(omp_pause_hard) == 0;
}

/* Runs the OpenMP runtime's default team once (every hardware thread, bound as OMP_PLACES says) and returns its
 * size. gcc's runtime keeps those threads docked afterwards, spinning under OMP_WAIT_POLICY=active, beside every
 * later region of ONE thread: a team of one starts no thread and releases none, where a team of two or more ends
 * the surplus. That is the state cpu_expert_probe times its one-thread case in, having filled and checked with
 * the whole machine. Whether they spin is not assumed: a probe prints the busy threads of its timed steps. */
static inline int cpu_openmp_dock_default_team(void)
{
    int team_size = 0;
    #pragma omp parallel
    {
        if (omp_get_thread_num() == 0) {
            team_size = omp_get_num_threads();
        }
    }
    return team_size;
}

/* One matrix as cpu_expert_probe writes it, on 2 MiB pages where the kernel gives them: a weight is then one
 * page-table step away, as in the engine's host arena. Returns 0 without memory or without the page advice. */
static inline int cpu_matrix_fill(struct cpu_matrix *matrix, int output_size, int input_size, uint64_t seed)
{
    const size_t nibble_bytes = (size_t)output_size * (size_t)(input_size / CPU_NIBBLES_PER_BYTE);
    const size_t scale_bytes = (size_t)output_size * (size_t)(input_size / CPU_WEIGHTS_PER_SCALE_GROUP);
    const size_t nibble_room = (nibble_bytes + CPU_HUGE_PAGE_BYTES - 1) / CPU_HUGE_PAGE_BYTES * CPU_HUGE_PAGE_BYTES;
    const size_t scale_room = (scale_bytes + CPU_BYTES_PER_VECTOR - 1) / CPU_BYTES_PER_VECTOR * CPU_BYTES_PER_VECTOR;
    uint64_t byte_sum = 0;
    matrix->output_size = output_size;
    matrix->input_size = input_size;
    matrix->nibbles = aligned_alloc(CPU_HUGE_PAGE_BYTES, nibble_room);
    matrix->scale_codes = aligned_alloc(CPU_BYTES_PER_VECTOR, scale_room);
    if (matrix->nibbles == NULL || matrix->scale_codes == NULL || nibble_bytes % sizeof(uint64_t) != 0 ||
        madvise(matrix->nibbles, nibble_room, MADV_HUGEPAGE) != 0) {
        return 0;
    }
    for (size_t word = 0; word < nibble_bytes / sizeof(uint64_t); word++) {
        const uint64_t bits = cpu_mix_bits(seed + word);
        memcpy(matrix->nibbles + word * sizeof(bits), &bits, sizeof(bits));
    }
    for (size_t index = 0; index < scale_bytes; index++) {
        matrix->scale_codes[index] =
            (uint8_t)(CPU_SCALE_CODE_BASE + cpu_mix_bits(~seed + index) % CPU_SCALE_CODE_SPREAD);
        byte_sum += matrix->scale_codes[index];
    }
    for (size_t index = 0; index < nibble_bytes; index++) {
        byte_sum += matrix->nibbles[index];
    }
    matrix->byte_sum = byte_sum;
    return 1;
}

/* The ring, filled by an OpenMP loop. Returns 0 on any failure; a failed ring is not used. Every expert has a
 * flag of its own: a reduction over the loop would be an atomic instruction in the program. */
static inline int cpu_ring_create(struct cpu_ring *ring, int expert_count)
{
    int filled = 1;
    ring->expert_count = expert_count;
    ring->experts = calloc((size_t)expert_count, sizeof(*ring->experts));
    if (ring->experts == NULL) {
        return 0;
    }
    for (int code = 0; code < CPU_SCALE_CODES; code++) {
        ring->scale_of_code[code] = ldexp(1.0, code - CPU_E8M0_BIAS);
        ring->scale_times_half[code] = (float)ldexp(1.0, code - CPU_E8M0_BIAS - 1);
    }
    #pragma omp parallel for if(CPU_OPENMP_HELPS) schedule(dynamic)
    for (int expert_index = 0; expert_index < expert_count; expert_index++) {
        const uint64_t seed = (uint64_t)expert_index << CPU_EXPERT_SEED_SHIFT;
        struct cpu_expert *expert = &ring->experts[expert_index];
        expert->filled =
            cpu_matrix_fill(&expert->matrix[CPU_GATE], CPU_EXPERT_INNER_SIZE, CPU_HIDDEN_SIZE, seed + 1) &&
            cpu_matrix_fill(&expert->matrix[CPU_UP], CPU_EXPERT_INNER_SIZE, CPU_HIDDEN_SIZE,
                            seed + ((uint64_t)CPU_UP << CPU_MATRIX_SEED_SHIFT)) &&
            cpu_matrix_fill(&expert->matrix[CPU_DOWN], CPU_HIDDEN_SIZE, CPU_EXPERT_INNER_SIZE,
                            seed + ((uint64_t)CPU_DOWN << CPU_MATRIX_SEED_SHIFT));
    }
    for (int expert_index = 0; expert_index < expert_count; expert_index++) {
        filled &= ring->experts[expert_index].filled;
    }
    return cpu_openmp_release_threads() && filled;
}

/* ---- the reference: the format decoded from its definition, in double precision ---- */

static inline double cpu_reference_weight(const struct cpu_ring *ring, const struct cpu_matrix *matrix, int row,
                                          int column)
{
    static const double magnitude_of_code[CPU_MAGNITUDE_CODES] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};
    const uint8_t byte = matrix->nibbles[(size_t)row * cpu_row_nibble_bytes(matrix) +
                                         (size_t)(column / CPU_NIBBLES_PER_BYTE)];
    const unsigned nibble = column % CPU_NIBBLES_PER_BYTE == 0 ? (byte & (unsigned)CPU_NIBBLE_MASK)
                                                               : ((unsigned)byte >> CPU_NIBBLE_BITS);
    const uint8_t scale_code = matrix->scale_codes[(size_t)row * cpu_row_scale_bytes(matrix) +
                                                   (size_t)(column / CPU_WEIGHTS_PER_SCALE_GROUP)];
    const double magnitude = magnitude_of_code[nibble & (unsigned)(CPU_MAGNITUDE_CODES - 1)];
    return ((nibble & (unsigned)CPU_SIGN_BIT) != 0u ? -magnitude : magnitude) * ring->scale_of_code[scale_code];
}

static inline double cpu_reference_row(const struct cpu_ring *ring, const struct cpu_matrix *matrix, int row,
                                       const double *input)
{
    double sum = 0.0;
    for (int column = 0; column < matrix->input_size; column++) {
        sum += cpu_reference_weight(ring, matrix, row, column) * input[column];
    }
    return sum;
}

/* ---- row kernels ---- */

/* The 32 nibbles of one scale group, in weight order, one to a byte. */
static inline __m256i cpu_group_nibbles(const uint8_t *bytes)
{
    const __m128i packed = _mm_loadu_si128((const void *)bytes);
    const __m128i mask = _mm_set1_epi8(CPU_NIBBLE_MASK);
    const __m128i low = _mm_and_si128(packed, mask);
    const __m128i high = _mm_and_si128(_mm_srli_epi16(packed, CPU_NIBBLE_BITS), mask);
    return _mm256_set_m128i(_mm_unpackhi_epi8(low, high), _mm_unpacklo_epi8(low, high));
}

/* b04's exact kernel: the weights of a scale group become sixteen-wide fp32 vectors (nibble, table, integer to
 * float, times the group's scale) and meet the fp32 input in fused multiply-adds, along the INPUT, two sums of
 * sixteen lanes a row. Nothing is rounded but the sums; no card adds in this order. */
static inline float cpu_row_dot_fp32(const struct cpu_matrix *matrix, int row, const float *input,
                                     const float *scale_times_half)
{
    /* twice the value of each nibble code, as a signed byte: exact, so the halving goes into the scale */
    static const int8_t twice_value[CPU_NIBBLE_CODES] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    const __m256i twice_values = _mm256_broadcastsi128_si256(_mm_loadu_si128((const void *)twice_value));
    const int group_count = matrix->input_size / CPU_WEIGHTS_PER_SCALE_GROUP;
    const uint8_t *nibbles = matrix->nibbles + (size_t)row * cpu_row_nibble_bytes(matrix);
    const uint8_t *scale_codes = matrix->scale_codes + (size_t)row * cpu_row_scale_bytes(matrix);
    __m512 sum_first = _mm512_setzero_ps(), sum_second = _mm512_setzero_ps();
    for (int group = 0; group < group_count; group++) {
        const __m256i values = _mm256_shuffle_epi8(twice_values,
                                                   cpu_group_nibbles(nibbles + group * CPU_BYTES_PER_SCALE_GROUP));
        const __m512 scale = _mm512_set1_ps(scale_times_half[scale_codes[group]]);
        const __m512 first = _mm512_mul_ps(
            _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm256_castsi256_si128(values))), scale);
        const __m512 second = _mm512_mul_ps(
            _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm256_extracti128_si256(values, 1))), scale);
        const float *group_input = input + group * CPU_WEIGHTS_PER_SCALE_GROUP;
        sum_first = _mm512_fmadd_ps(first, _mm512_loadu_ps(group_input), sum_first);
        sum_second = _mm512_fmadd_ps(second, _mm512_loadu_ps(group_input + CPU_FLOATS_PER_VECTOR), sum_second);
    }
    return _mm512_reduce_add_ps(_mm512_add_ps(sum_first, sum_second));
}

/* Adds the bytes at an address to eight running sums and does nothing else with them: what a kernel would have
 * to read, read. A tail shorter than a register is loaded under a mask, which touches no byte past it. */
static inline __m512i cpu_sum_bytes_into(__m512i sums, const uint8_t *bytes, size_t byte_count)
{
    const __m512i zero = _mm512_setzero_si512();
    size_t offset = 0;
    for (; offset + CPU_BYTES_PER_VECTOR <= byte_count; offset += CPU_BYTES_PER_VECTOR) {
        sums = _mm512_add_epi64(sums, _mm512_sad_epu8(_mm512_loadu_si512(bytes + offset), zero));
    }
    if (offset < byte_count) {
        const __mmask64 tail = ((__mmask64)1 << (byte_count - offset)) - 1;
        sums = _mm512_add_epi64(sums, _mm512_sad_epu8(_mm512_maskz_loadu_epi8(tail, bytes + offset), zero));
    }
    return sums;
}

/* Rows first_row up to row_end of a matrix, both streams of a row, in the order the fp32 kernel walks them. */
static inline uint64_t cpu_sum_matrix_rows(const struct cpu_matrix *matrix, int first_row, int row_end)
{
    const size_t nibble_bytes = cpu_row_nibble_bytes(matrix), scale_bytes = cpu_row_scale_bytes(matrix);
    __m512i sums = _mm512_setzero_si512();
    for (int row = first_row; row < row_end; row++) {
        sums = cpu_sum_bytes_into(sums, matrix->nibbles + (size_t)row * nibble_bytes, nibble_bytes);
        sums = cpu_sum_bytes_into(sums, matrix->scale_codes + (size_t)row * scale_bytes, scale_bytes);
    }
    return (uint64_t)_mm512_reduce_add_epi64(sums);
}

/* Worker `worker` of `workers` takes these rows of `rows`: the split of a static OpenMP loop, so that a team of
 * either kind gives a thread the rows b04's loops gave it. */
static inline void cpu_static_share(int rows, int workers, int worker, int *first_row, int *row_end)
{
    const int quotient = rows / workers, remainder = rows % workers;
    *first_row = worker * quotient + (worker < remainder ? worker : remainder);
    *row_end = *first_row + quotient + (worker < remainder ? 1 : 0);
}

/* ---- where threads sit ---- */

struct cpu_hardware_thread {
    int cpu;
    long slice_id, core_key;                     /* the third-level cache it shares; its core among all cores */
    int slice_rank, core_rank, sibling_rank;     /* its slice among slices, its core in the slice, itself in the core */
};

struct cpu_topology {
    int thread_count, core_count, slice_count;
    long slice_kib;                              /* one slice of third-level cache; 0 where sysfs does not say */
    struct cpu_hardware_thread threads[CPU_MAX_HARDWARE_THREADS];
};

enum cpu_placement { CPU_PLACEMENT_DIE, CPU_PLACEMENT_PACKED, CPU_PLACEMENT_SMT, CPU_PLACEMENT_KINDS };
static const char *const cpu_placement_names[CPU_PLACEMENT_KINDS] = {"die", "packed", "smt"};

struct cpu_placement_key {
    uint64_t key;
    int cpu;
};

static inline int cpu_compare_hardware_threads(const void *left, const void *right)
{
    const struct cpu_hardware_thread *first = left, *second = right;
    if (first->slice_id != second->slice_id) {
        return first->slice_id < second->slice_id ? -1 : 1;
    }
    if (first->core_key != second->core_key) {
        return first->core_key < second->core_key ? -1 : 1;
    }
    return (first->cpu > second->cpu) - (first->cpu < second->cpu);
}

static inline int cpu_compare_placement_keys(const void *left, const void *right)
{
    const struct cpu_placement_key *first = left, *second = right;
    return (first->key > second->key) - (first->key < second->key);
}

/* One online hardware thread with its core and its slice of third-level cache. Returns 0 when sysfs lacks it. */
static inline int cpu_topology_add(struct cpu_topology *topology, long cpu)
{
    char path[CPU_PATH_BYTES];
    struct cpu_hardware_thread *thread = &topology->threads[topology->thread_count];
    long core = 0, package = 0;
    if (cpu < 0 || cpu >= CPU_SETSIZE || topology->thread_count == CPU_MAX_HARDWARE_THREADS) {
        return 0;
    }
    (void)snprintf(path, sizeof(path), CPU_SYSTEM_DIRECTORY "/cpu%ld/topology/core_id", cpu);
    const int has_core = cpu_read_number_file(path, &core);
    (void)snprintf(path, sizeof(path), CPU_SYSTEM_DIRECTORY "/cpu%ld/topology/physical_package_id", cpu);
    if (!has_core || !cpu_read_number_file(path, &package)) {
        return 0;
    }
    (void)snprintf(path, sizeof(path), CPU_SYSTEM_DIRECTORY "/cpu%ld/cache/index%d/id", cpu, CPU_THIRD_LEVEL_CACHE);
    if (!cpu_read_number_file(path, &thread->slice_id)) {
        thread->slice_id = package;                          /* no third-level cache listed: a package is a slice */
    }
    (void)snprintf(path, sizeof(path), CPU_SYSTEM_DIRECTORY "/cpu%ld/cache/index%d/size", cpu, CPU_THIRD_LEVEL_CACHE);
    (void)cpu_read_number_file(path, &topology->slice_kib);  /* "32768K": the number is KiB; absent: stays 0 */
    thread->cpu = (int)cpu;
    thread->core_key = package * CPU_CORE_KEY_PACKAGE_STRIDE + core;
    topology->thread_count++;
    return 1;
}

/* The online hardware threads, sorted by slice, core and number, each with its ranks. The online list is read and
 * not this thread's affinity: under OMP_PROC_BIND the OpenMP runtime can have bound the first thread to one core
 * before main runs. A thread this process may not use fails where a team is pinned to it. Returns 0 without sysfs. */
static inline int cpu_topology_read(struct cpu_topology *topology)
{
    char text[CPU_TEXT_BYTES];
    const char *cursor = text;
    FILE *file = fopen(CPU_SYSTEM_DIRECTORY "/online", "r");
    memset(topology, 0, sizeof(*topology));
    if (file == NULL) {
        return 0;
    }
    if (fgets(text, sizeof(text), file) == NULL) {           /* its result must be used: the C library says so where */
        text[0] = '\0';                                      /* the compiler fortifies, and a cast does not count */
    }
    (void)fclose(file);
    if (strchr(text, '\n') == NULL) {                        /* nothing read, or a list longer than this reads */
        return 0;
    }
    while (*cursor != '\0' && *cursor != '\n') {             /* "0-95,100,104-191" */
        char *end = NULL;
        const long first = strtol(cursor, &end, CPU_DECIMAL);
        long last = first;
        if (end != cursor && *end == '-') {
            cursor = end + 1;
            last = strtol(cursor, &end, CPU_DECIMAL);
        }
        if (end == cursor) {
            return 0;
        }
        for (long cpu = first; cpu <= last; cpu++) {
            if (!cpu_topology_add(topology, cpu)) {
                return 0;
            }
        }
        cursor = *end == ',' ? end + 1 : end;
    }
    if (topology->thread_count == 0) {
        return 0;
    }
    qsort(topology->threads, (size_t)topology->thread_count, sizeof(topology->threads[0]),
          cpu_compare_hardware_threads);
    topology->slice_count = 1;
    topology->core_count = 1;
    for (int index = 1; index < topology->thread_count; index++) {
        struct cpu_hardware_thread *thread = &topology->threads[index];
        const struct cpu_hardware_thread before = topology->threads[index - 1];
        const int new_slice = before.slice_id != thread->slice_id;
        const int new_core = new_slice || before.core_key != thread->core_key;
        topology->slice_count += new_slice;
        topology->core_count += new_core;
        thread->slice_rank = topology->slice_count - 1;
        thread->core_rank = new_slice ? 0 : before.core_rank + new_core;
        thread->sibling_rank = new_core ? 0 : before.sibling_rank + 1;
    }
    return 1;
}

/* The hardware threads in the order a placement takes them; a team of n threads sits on the first n.
 *   die     the first thread of one core of every slice in turn: n <= slices is one thread a slice
 *   packed  the first thread of every core of a slice before the next slice
 *   smt     both threads of a core before the next core, slice by slice
 * die and packed come to the sibling threads only after every core has one thread. */
static inline void cpu_placement_order(const struct cpu_topology *topology, enum cpu_placement placement, int *cpus)
{
    struct cpu_placement_key keys[CPU_MAX_HARDWARE_THREADS];
    for (int index = 0; index < topology->thread_count; index++) {
        const struct cpu_hardware_thread *thread = &topology->threads[index];
        const uint64_t slice = (uint64_t)thread->slice_rank, core = (uint64_t)thread->core_rank;
        const uint64_t sibling = (uint64_t)thread->sibling_rank;
        const uint64_t first = placement == CPU_PLACEMENT_SMT ? slice : sibling;
        const uint64_t second = placement == CPU_PLACEMENT_PACKED ? slice : core;
        const uint64_t third = placement == CPU_PLACEMENT_DIE ? slice
                                                              : (placement == CPU_PLACEMENT_SMT ? sibling : core);
        keys[index].key = (((first << CPU_PLACEMENT_KEY_SHIFT) | second) << CPU_PLACEMENT_KEY_SHIFT) | third;
        keys[index].cpu = thread->cpu;
    }
    qsort(keys, (size_t)topology->thread_count, sizeof(keys[0]), cpu_compare_placement_keys);
    for (int index = 0; index < topology->thread_count; index++) {
        cpus[index] = keys[index].cpu;
    }
}

static inline int cpu_pin_this_thread(int cpu)
{
    cpu_set_t set;
    if (cpu < 0 || cpu >= CPU_SETSIZE) {
        return 0;
    }
    CPU_ZERO(&set);
    CPU_SET((size_t)cpu, &set);
    return sched_setaffinity(0, sizeof(set), &set) == 0;
}

/* ---- a team: n pinned threads, an OpenMP team or the minimal pool ---- */

typedef void cpu_work_function(void *context, int thread_index);

enum cpu_pool_kind { CPU_POOL_OPENMP, CPU_POOL_OWN, CPU_POOL_DOCKED, CPU_POOL_KINDS };
static const char *const cpu_pool_names[CPU_POOL_KINDS] = {"openmp", "own", "docked"};

struct cpu_thread_stamps {                       /* one thread's stamps of one region, on a line of its own */
    _Alignas(CPU_STAMP_LINE_BYTES) uint64_t entered_ticks;
    uint64_t left_ticks;
    int entered_cpu;
};

struct cpu_region_record {                       /* one parallel region: the caller's two stamps, each thread's two */
    uint64_t forked_ticks, joined_ticks;
    struct cpu_thread_stamps *threads;           /* the caller's array, one element per thread of the team */
};

struct cpu_team;

struct cpu_team_node {                           /* a worker's mailbox: one line, read and written under its lock */
    _Alignas(CPU_CACHE_LINE_BYTES) pthread_spinlock_t lock;
    uint64_t job_number;                         /* the job its parent in the tree released it for */
    uint64_t done_number;                        /* the job it and the workers below it have finished */
    pthread_t thread;
    struct cpu_team *team;
    int thread_index;
};

struct cpu_team {
    enum cpu_pool_kind kind;
    int thread_count;
    int bystanders;                              /* docked: the runtime's threads left spinning beside the team */
    const int *pinned_cpus;                      /* thread i is pinned to pinned_cpus[i] */
    int *reported_cpus;                          /* where thread i found itself at the last settle or where job */
    uint64_t *switches;                          /* thread i's context switches, as the last usage job read them */
    uint64_t wait_ticks_max;                     /* a join that lasts longer has failed */
    struct cpu_team_node *nodes;                 /* the minimal pool only */
    /* the job: written by the caller before it is released, read by a worker after its mailbox showed its number */
    cpu_work_function *function;
    void *context;
    struct cpu_region_record *record;
    uint64_t job_number;
};

/* A mailbox's lock is held for three instructions, by its worker or by the worker's parent in the tree and by no
 * third thread; the library documents no failure for an initialised lock the caller does not hold. */
static inline void cpu_team_write(struct cpu_team_node *node, int done, uint64_t number)
{
    (void)pthread_spin_lock(&node->lock);
    if (done) {
        node->done_number = number;
    } else {
        node->job_number = number;
    }
    (void)pthread_spin_unlock(&node->lock);
}

static inline uint64_t cpu_team_read(struct cpu_team_node *node, int done)
{
    (void)pthread_spin_lock(&node->lock);
    const uint64_t number = done ? node->done_number : node->job_number;
    (void)pthread_spin_unlock(&node->lock);
    return number;
}

/* Thread i releases threads 2 i + 1 and 2 i + 2 before it works: the last of 96 is six hops from the caller,
 * where one thread releasing all would fetch every worker's line in turn. */
static inline void cpu_team_release_below(struct cpu_team *team, int thread_index, uint64_t job_number)
{
    const int first_below = CPU_TREE_ARITY * thread_index + 1;
    for (int below = first_below; below < first_below + CPU_TREE_ARITY && below < team->thread_count; below++) {
        cpu_team_write(&team->nodes[below], 0, job_number);
    }
}

/* Waits until the threads below have finished the job, and those below them. deadline_ticks 0: none (a worker;
 * the caller's deadline ends the run). Returns 0 when the deadline passed. */
static inline int cpu_team_collect_below(struct cpu_team *team, int thread_index, uint64_t job_number,
                                         uint64_t deadline_ticks)
{
    const int first_below = CPU_TREE_ARITY * thread_index + 1;
    for (int below = first_below; below < first_below + CPU_TREE_ARITY && below < team->thread_count; below++) {
        while (cpu_team_read(&team->nodes[below], 1) != job_number) {
            _mm_pause();
            if (deadline_ticks != 0 && __rdtsc() > deadline_ticks) {
                return 0;
            }
        }
    }
    return 1;
}

/* One thread's share of the job, between its two stamps when the region is recorded: the work begins at the
 * instruction after the first stamp and ends at the second. */
static inline void cpu_team_do_share(struct cpu_team *team, int thread_index)
{
    struct cpu_thread_stamps *stamps = team->record != NULL ? &team->record->threads[thread_index] : NULL;
    if (stamps != NULL) {
        stamps->entered_cpu = sched_getcpu();
        stamps->entered_ticks = __rdtsc();
    }
    team->function(team->context, thread_index);
    if (stamps != NULL) {
        stamps->left_ticks = __rdtsc();
    }
}

static inline void *cpu_team_worker(void *argument)
{
    struct cpu_team_node *node = argument;
    struct cpu_team *team = node->team;
    uint64_t seen_number = 0;
    for (;;) {
        const uint64_t job_number = cpu_team_read(node, 0);
        if (job_number == seen_number) {
            _mm_pause();
            continue;
        }
        seen_number = job_number;
        cpu_team_release_below(team, node->thread_index, job_number);
        const int is_last_job = team->function == NULL;
        if (!is_last_job) {
            cpu_team_do_share(team, node->thread_index);
        }
        (void)cpu_team_collect_below(team, node->thread_index, job_number, 0);
        cpu_team_write(node, 1, job_number);
        if (is_last_job) {
            return NULL;
        }
    }
}

/* Runs function(context, i) once on every thread i of the team and returns when all have returned. With a record
 * the caller's stamps bracket the region and every thread stamps its own share. Returns 0 when the team did not
 * do the whole job: the OpenMP runtime gave fewer threads than asked for, or the pool's join passed its deadline. */
static inline int cpu_team_run(struct cpu_team *team, cpu_work_function *function, void *context,
                               struct cpu_region_record *record)
{
    int delivered = 0;
    team->function = function;
    team->context = context;
    team->record = record;
    if (record != NULL) {
        record->forked_ticks = __rdtsc();
    }
    if (team->kind == CPU_POOL_OWN) {
        const uint64_t job_number = ++team->job_number;
        cpu_team_release_below(team, 0, job_number);
        cpu_team_do_share(team, 0);
        delivered = cpu_team_collect_below(team, 0, job_number, __rdtsc() + team->wait_ticks_max)
                        ? team->thread_count : 0;
    } else {
        #pragma omp parallel num_threads(team->thread_count)
        {
            const int thread_index = omp_get_thread_num();
            if (thread_index == 0) {
                delivered = omp_get_num_threads();
            }
            if (thread_index < team->thread_count) {
                cpu_team_do_share(team, thread_index);
            }
        }
    }
    if (record != NULL) {
        record->joined_ticks = __rdtsc();
    }
    return delivered == team->thread_count;
}

/* A job: the thread pins itself, then says where it is. An OpenMP thread has no other moment to be pinned; a
 * worker of the pool was created pinned and pins itself to the same place again. Every thread but the caller
 * takes the workers' name, by which a tracer tells a team's threads from all others of the process. */
static inline void cpu_team_settle_share(void *context, int thread_index)
{
    struct cpu_team *team = context;
    if (thread_index > 0) {
        (void)pthread_setname_np(pthread_self(), CPU_TEAM_WORKER_NAME);      /* a name: nothing depends on it */
    }
    team->reported_cpus[thread_index] = cpu_pin_this_thread(team->pinned_cpus[thread_index]) ? sched_getcpu()
                                                                                             : CPU_NOWHERE;
}

/* A job: the thread says where it runs and changes nothing, so that a thread the runtime moved or replaced after
 * it settled is seen. */
static inline void cpu_team_where_share(void *context, int thread_index)
{
    struct cpu_team *team = context;
    team->reported_cpus[thread_index] = sched_getcpu();
}

static inline void cpu_team_usage_share(void *context, int thread_index)
{
    struct cpu_team *team = context;
    struct rusage usage;
    team->switches[thread_index] = getrusage(RUSAGE_THREAD, &usage) == 0
                                       ? (uint64_t)usage.ru_nvcsw + (uint64_t)usage.ru_nivcsw : 0;
}

/* Starts a team of thread_count threads, the caller being thread 0 and thread i pinned to pinned_cpus[i], and
 * has every thread settle; a docked team (one thread, no other count) then docks the runtime's default team beside
 * itself. Returns 0 when a thread could not be created or pinned: such a team is not usable and not closed, the
 * run has to end (threads that were created spin until it does). */
static inline int cpu_team_open(struct cpu_team *team, enum cpu_pool_kind kind, int thread_count,
                                const int *pinned_cpus, double ticks_per_microsecond)
{
    const size_t node_bytes = (size_t)thread_count * sizeof(*team->nodes);
    memset(team, 0, sizeof(*team));
    team->kind = kind;
    team->thread_count = thread_count;
    team->pinned_cpus = pinned_cpus;
    team->wait_ticks_max = (uint64_t)(ticks_per_microsecond * CPU_MICROSECONDS_PER_SECOND) * CPU_TEAM_WAIT_SECONDS_MAX;
    team->reported_cpus = calloc((size_t)thread_count, sizeof(*team->reported_cpus));
    team->switches = calloc((size_t)thread_count, sizeof(*team->switches));
    team->nodes = kind == CPU_POOL_OWN ? aligned_alloc(CPU_CACHE_LINE_BYTES, node_bytes) : NULL;
    if (team->reported_cpus == NULL || team->switches == NULL || (kind == CPU_POOL_OWN && team->nodes == NULL) ||
        (kind == CPU_POOL_DOCKED && thread_count != 1) || !cpu_pin_this_thread(pinned_cpus[0])) {
        return 0;
    }
    for (int index = 0; kind == CPU_POOL_OWN && index < thread_count; index++) {
        struct cpu_team_node *node = &team->nodes[index];
        pthread_attr_t attributes;
        cpu_set_t set;
        memset(node, 0, sizeof(*node));
        node->team = team;
        node->thread_index = index;
        CPU_ZERO(&set);
        CPU_SET((size_t)pinned_cpus[index], &set);
        if (pthread_spin_init(&node->lock, PTHREAD_PROCESS_PRIVATE) != 0 || pthread_attr_init(&attributes) != 0) {
            return 0;
        }
        const int created = index == 0 || (pthread_attr_setaffinity_np(&attributes, sizeof(set), &set) == 0 &&
                                           pthread_create(&node->thread, &attributes, cpu_team_worker, node) == 0);
        (void)pthread_attr_destroy(&attributes);
        if (!created) {
            return 0;
        }
    }
    int settled = cpu_team_run(team, cpu_team_settle_share, team, NULL);
    for (int index = 0; index < thread_count; index++) {
        settled &= team->reported_cpus[index] != CPU_NOWHERE;
    }
    if (settled && kind == CPU_POOL_DOCKED) {
        team->bystanders = cpu_openmp_dock_default_team() - thread_count;
    }
    return settled;
}

/* Ends a team: the pool's workers return and are joined, the OpenMP runtime's threads are ended, so that no
 * thread of this slot spins beside the next. Returns 0 when a worker could not be collected (it is then left). */
static inline int cpu_team_close(struct cpu_team *team)
{
    int closed = 1;
    if (team->kind == CPU_POOL_OWN) {
        const uint64_t job_number = ++team->job_number;
        team->function = NULL;
        cpu_team_release_below(team, 0, job_number);
        closed = cpu_team_collect_below(team, 0, job_number, __rdtsc() + team->wait_ticks_max);
        for (int index = 1; closed && index < team->thread_count; index++) {
            closed = pthread_join(team->nodes[index].thread, NULL) == 0;
        }
        for (int index = 0; closed && index < team->thread_count; index++) {
            (void)pthread_spin_destroy(&team->nodes[index].lock);
        }
        if (closed) {
            free(team->nodes);
        }
    } else {
        closed = cpu_openmp_release_threads();
    }
    free(team->reported_cpus);
    free(team->switches);
    memset(team, 0, sizeof(*team));
    return closed;
}

/* Asks every thread where it runs, without pinning it again, and returns how many are not on the processor the
 * run announced for them. */
static inline int cpu_team_misplaced(struct cpu_team *team, const int *announced_cpus)
{
    int misplaced = 0;
    if (!cpu_team_run(team, cpu_team_where_share, team, NULL)) {
        return team->thread_count;
    }
    for (int index = 0; index < team->thread_count; index++) {
        misplaced += team->reported_cpus[index] != announced_cpus[index];
    }
    return misplaced;
}

/* The context switches of all the team's threads since each began, voluntary and not. Returns 0 on a failed job. */
static inline int cpu_team_context_switches(struct cpu_team *team, uint64_t *switches)
{
    const int whole = cpu_team_run(team, cpu_team_usage_share, team, NULL);
    *switches = 0;
    for (int index = 0; index < team->thread_count; index++) {
        *switches += team->switches[index];
    }
    return whole;
}

/* ---- the memory modules ---- */

struct cpu_module_temperatures {
    int module_count, lowest_c, highest_c;       /* module_count 0: no sensor of that name on this machine */
};

static inline void cpu_read_module_temperatures(struct cpu_module_temperatures *temperatures)
{
    char path[CPU_PATH_BYTES], name[CPU_TEXT_BYTES];
    temperatures->module_count = 0;
    temperatures->lowest_c = INT_MAX;
    temperatures->highest_c = INT_MIN;
    for (int node = 0; node < CPU_HWMON_NODES_SCANNED; node++) {
        long millidegrees = 0;
        (void)snprintf(path, sizeof(path), "/sys/class/hwmon/hwmon%d/name", node);
        FILE *file = fopen(path, "r");
        if (file == NULL) {
            continue;
        }
        const int is_module = fgets(name, sizeof(name), file) != NULL &&
                              strncmp(name, CPU_MODULE_SENSOR_NAME, strlen(CPU_MODULE_SENSOR_NAME)) == 0;
        (void)fclose(file);
        (void)snprintf(path, sizeof(path), "/sys/class/hwmon/hwmon%d/temp1_input", node);
        if (!is_module || !cpu_read_number_file(path, &millidegrees)) {
            continue;
        }
        const int degrees = (int)(millidegrees / CPU_MILLIDEGREES_PER_DEGREE);
        temperatures->module_count++;
        temperatures->lowest_c = degrees < temperatures->lowest_c ? degrees : temperatures->lowest_c;
        temperatures->highest_c = degrees > temperatures->highest_c ? degrees : temperatures->highest_c;
    }
}

#endif
