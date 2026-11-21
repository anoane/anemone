/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * cpu_expert_probe: what one routed expert of DeepSeek-V4.1-Flash costs when the HOST computes it, in the
 * checkpoint's own 4-bit format, on this machine's processor.
 *
 * The question it settles. An expert that does not fit on a card lives in host memory. Anemone's plan uploads it
 * when a token routes to it: 17.3 MiB over the card's link (10.7 ms at the 1.69 GB/s of a CMP 170HX on four lanes,
 * 2.8 ms at 6.5 GB/s on sixteen), then 28 us of card time. KTransformers computes such experts on the processor
 * instead. An earlier measurement on a 16-core desktop found the processor no better than the link; this machine
 * has 96 cores and eight memory channels. So: how many microseconds does the processor need for one cold expert,
 * with how many threads, and what does a whole layer (six experts) cost if all of it ran here?
 *
 * Format: sign and 3-bit magnitude per weight (0, 0.5, 1, 1.5, 2, 3, 4, 6), two weights to a byte, low nibble
 * first, and one power-of-two scale (2^(e - 127)) per 32 weights of a row. An expert is gate = W1 x, up = W3 x,
 * h = silu(min(gate, L)) * clamp(up, -L, L) * route weight, y = W2 h, with W1 and W3 of 2304 x 5120 and W2 of
 * 5120 x 2304. Weights are synthetic; every run checks sampled outputs against a double-precision reference that
 * decodes the format from first principles, and --corrupt overwrites part of a matrix first, where the check must
 * fail.
 *
 * Two kernels, both with AVX-512:
 *   fp32   the weights of a scale group become sixteen-wide fp32 vectors (nibble, table, integer to float, times
 *          the group's scale) and meet the fp32 input in fused multiply-adds: nothing is rounded but the sums.
 *   bf16   the weights become bf16 by a table, the group's scale is an integer added to their exponent, and one
 *          dot-product instruction (AVX512_BF16) takes all 32 products of a group; the input is rounded to bf16
 *          (8 bits of mantissa instead of fp16's 11), so this one changes values.
 * Experts are taken in turn from a ring larger than the processor's caches, so every one is read from memory.
 *
 *   cpu_expert_probe [--experts N] [--steps N] [--small-pages] [--corrupt]
 *
 * Measurement tool: not part of the engine.
 */
#define _GNU_SOURCE
#include <immintrin.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>

#define HIDDEN_SIZE 5120
#define EXPERT_INNER_SIZE 2304
#define EXPERTS_PER_ROW 6
#define SWIGLU_LIMIT 10.0f
#define WEIGHTS_PER_SCALE_GROUP 32
#define BYTES_PER_SCALE_GROUP (WEIGHTS_PER_SCALE_GROUP / 2)
#define FLOATS_PER_VECTOR 16
#define E8M0_BIAS 127
#define MAGNITUDE_CODES 8
#define NIBBLE_MASK 0x0F
#define SIGN_BIT 0x08
#define BF16_EXPONENT_SHIFT 7                    /* the exponent field of a bf16 begins at bit 7 */
#define DEFAULT_EXPERT_COUNT 128                 /* 2.2 GiB: far above the 384 MiB of third-level cache */
#define DEFAULT_STEP_COUNT 48
#define HUGE_PAGE_BYTES (2u * 1024u * 1024u)
#define WARM_UP_STEPS 4
#define SCALE_CODE_BASE 118                      /* synthetic scales 2^-9 .. 2^-6, as in the card probes */
#define SCALE_CODE_SPREAD 4u
#define INPUT_VALUE_STEPS 255u                   /* synthetic inputs -2 .. 2 in steps of 1/64: exact in fp16 */
#define INPUT_VALUE_OFFSET 128.0f
#define INPUT_VALUE_DIVISOR 64.0f
#define ROUTE_WEIGHT 0.25f
#define OUTPUTS_CHECKED 6
#define CORRUPTED_BYTES (1024u * 1024u)
#define CORRUPTED_BYTE_VALUE 0x5A
#define FP32_RELATIVE_TOLERANCE 1.0e-4
#define BF16_RELATIVE_TOLERANCE 2.0e-2
#define MICROSECONDS_PER_SECOND 1.0e6
#define BYTES_PER_GB 1.0e9
#define BYTES_PER_MIB (1024.0 * 1024.0)
#define LINK_FOUR_LANES_GB_PER_SECOND 1.69       /* measured: a CMP 170HX on PCIe generation 2, four lanes */
#define LINK_SIXTEEN_LANES_GB_PER_SECOND 6.5     /* expected after the cards are modified */
#define CARD_MICROSECONDS_PER_EXPERT 28.0        /* measured: 166 us for six experts on a CMP 170HX */

static const double magnitude_of_code[MAGNITUDE_CODES] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};
static const int thread_counts[] = {1, 2, 4, 8, 16, 24, 32, 48, 96};
#define THREAD_COUNT_CASES (sizeof(thread_counts) / sizeof(thread_counts[0]))

struct matrix {                                  /* output_size rows of input_size weights */
    uint8_t *nibbles;                            /* input_size / 2 bytes per row */
    uint8_t *scale_codes;                        /* input_size / 32 bytes per row */
    int output_size;
    int input_size;
};

struct expert {
    struct matrix gate, up, down;
};

static float scale_times_half[256];              /* fp32 kernel: 2^(code - 127) / 2, because its table holds 2 x value */
static float input_fp32[HIDDEN_SIZE];
static int use_huge_pages = 1;
static uint16_t input_bf16[HIDDEN_SIZE];

static double seconds_now(void)
{
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (double)now.tv_sec + (double)now.tv_nsec / 1.0e9;
}

static uint64_t mix_bits(uint64_t value)
{
    value ^= value >> 30;
    value *= 0xbf58476d1ce4e5b9ull;
    value ^= value >> 27;
    value *= 0x94d049bb133111ebull;
    value ^= value >> 31;
    return value;
}

static uint16_t bf16_of(float value)             /* round to nearest even, as the processor's conversion does */
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

static float float_of_bf16(uint16_t pattern)
{
    const uint32_t bits = (uint32_t)pattern << 16;
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static void fill_matrix(struct matrix *matrix, int output_size, int input_size, uint64_t seed)
{
    const size_t nibble_bytes = (size_t)output_size * (size_t)(input_size / 2);
    const size_t scale_bytes = (size_t)output_size * (size_t)(input_size / WEIGHTS_PER_SCALE_GROUP);
    matrix->output_size = output_size;
    matrix->input_size = input_size;
    /* on 2 MiB pages when asked for, as the engine's host arena will be: a weight is then one page-table step away */
    const size_t nibble_room = (nibble_bytes + HUGE_PAGE_BYTES - 1) / HUGE_PAGE_BYTES * HUGE_PAGE_BYTES;
    matrix->nibbles = aligned_alloc(HUGE_PAGE_BYTES, nibble_room);
    matrix->scale_codes = aligned_alloc(64, (scale_bytes + 63) / 64 * 64);
    if (matrix->nibbles == NULL || matrix->scale_codes == NULL) {
        fprintf(stderr, "no memory for a matrix\n");
        exit(EXIT_FAILURE);
    }
    if (madvise(matrix->nibbles, nibble_room, use_huge_pages ? MADV_HUGEPAGE : MADV_NOHUGEPAGE) != 0) {
        fprintf(stderr, "the kernel refused the page-size advice\n");
        exit(EXIT_FAILURE);
    }
    for (size_t word = 0; word < nibble_bytes / 8; word++) {
        const uint64_t bits = mix_bits(seed + word);
        memcpy(matrix->nibbles + word * 8, &bits, sizeof(bits));
    }
    for (size_t index = 0; index < scale_bytes; index++) {
        matrix->scale_codes[index] = (uint8_t)(SCALE_CODE_BASE + (mix_bits(~seed + index) % SCALE_CODE_SPREAD));
    }
}

/* ---- the reference: the format decoded from first principles, in double precision ---- */

static double reference_weight(const struct matrix *matrix, int row, int column)
{
    const uint8_t byte = matrix->nibbles[(size_t)row * (size_t)(matrix->input_size / 2) + (size_t)(column / 2)];
    const unsigned nibble = (column % 2 == 0) ? (byte & NIBBLE_MASK) : (byte >> 4);
    const int code = matrix->scale_codes[(size_t)row * (size_t)(matrix->input_size / WEIGHTS_PER_SCALE_GROUP) +
                                         (size_t)(column / WEIGHTS_PER_SCALE_GROUP)];
    const double magnitude = magnitude_of_code[nibble & (MAGNITUDE_CODES - 1)];
    return ((nibble & SIGN_BIT) ? -magnitude : magnitude) * ldexp(1.0, code - E8M0_BIAS);
}

static double reference_row(const struct matrix *matrix, int row, const double *input)
{
    double sum = 0.0;
    for (int column = 0; column < matrix->input_size; column++) {
        sum += reference_weight(matrix, row, column) * input[column];
    }
    return sum;
}

/* hidden_as_float: the kernels hand the hidden vector to the down matrix as fp32, so the reference rounds there
 * too (the bf16 kernel's further rounding is its own error and is what its tolerance allows). */
static void reference_outputs(const struct expert *expert, const int *output_rows, int output_count, double *outputs)
{
    static double input[HIDDEN_SIZE], hidden[EXPERT_INNER_SIZE];
    for (int column = 0; column < HIDDEN_SIZE; column++) {
        input[column] = (double)input_fp32[column];
    }
    #pragma omp parallel for schedule(static)
    for (int row = 0; row < EXPERT_INNER_SIZE; row++) {
        double gate = reference_row(&expert->gate, row, input);
        double up = reference_row(&expert->up, row, input);
        gate = gate < (double)SWIGLU_LIMIT ? gate : (double)SWIGLU_LIMIT;
        up = up < -(double)SWIGLU_LIMIT ? -(double)SWIGLU_LIMIT : (up > (double)SWIGLU_LIMIT ? (double)SWIGLU_LIMIT : up);
        hidden[row] = (double)(float)(gate / (1.0 + exp(-gate)) * up * (double)ROUTE_WEIGHT);
    }
    for (int index = 0; index < output_count; index++) {
        outputs[index] = reference_row(&expert->down, output_rows[index], hidden);
    }
}

/* ---- the kernels ---- */

/* The 32 nibbles of one scale group, in weight order, one to a byte. */
static inline __m256i group_nibbles(const uint8_t *bytes)
{
    const __m128i packed = _mm_loadu_si128((const __m128i *)bytes);
    const __m128i mask = _mm_set1_epi8(NIBBLE_MASK);
    const __m128i low = _mm_and_si128(packed, mask);
    const __m128i high = _mm_and_si128(_mm_srli_epi16(packed, 4), mask);
    return _mm256_set_m128i(_mm_unpackhi_epi8(low, high), _mm_unpacklo_epi8(low, high));
}

static float row_dot_fp32(const struct matrix *matrix, int row, const float *input)
{
    /* twice the value of each nibble code, as a signed byte: exact, so the halving goes into the scale */
    const __m256i twice_value = _mm256_broadcastsi128_si256(
        _mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12));
    const int group_count = matrix->input_size / WEIGHTS_PER_SCALE_GROUP;
    const uint8_t *nibbles = matrix->nibbles + (size_t)row * (size_t)(matrix->input_size / 2);
    const uint8_t *scale_codes = matrix->scale_codes + (size_t)row * (size_t)group_count;
    __m512 sum_first = _mm512_setzero_ps(), sum_second = _mm512_setzero_ps();
    for (int group = 0; group < group_count; group++) {
        const __m256i values = _mm256_shuffle_epi8(twice_value, group_nibbles(nibbles + group * BYTES_PER_SCALE_GROUP));
        const __m512 scale = _mm512_set1_ps(scale_times_half[scale_codes[group]]);
        const __m512 first = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm256_castsi256_si128(values))), scale);
        const __m512 second = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm256_extracti128_si256(values, 1))), scale);
        sum_first = _mm512_fmadd_ps(first, _mm512_loadu_ps(input + group * WEIGHTS_PER_SCALE_GROUP), sum_first);
        sum_second = _mm512_fmadd_ps(second, _mm512_loadu_ps(input + group * WEIGHTS_PER_SCALE_GROUP + FLOATS_PER_VECTOR), sum_second);
    }
    return _mm512_reduce_add_ps(_mm512_add_ps(sum_first, sum_second));
}

static uint16_t bf16_of_code[2 * MAGNITUDE_CODES * 2];   /* the sixteen nibble codes as bf16, twice over for the permute */

static float row_dot_bf16(const struct matrix *matrix, int row, const uint16_t *input)
{
    const __m512i table = _mm512_loadu_si512(bf16_of_code);
    const __m256i magnitude_mask = _mm256_set1_epi8(MAGNITUDE_CODES - 1);
    const int group_count = matrix->input_size / WEIGHTS_PER_SCALE_GROUP;
    const uint8_t *nibbles = matrix->nibbles + (size_t)row * (size_t)(matrix->input_size / 2);
    const uint8_t *scale_codes = matrix->scale_codes + (size_t)row * (size_t)group_count;
    __m512 sum = _mm512_setzero_ps();
    for (int group = 0; group < group_count; group++) {
        const __m256i codes = group_nibbles(nibbles + group * BYTES_PER_SCALE_GROUP);
        /* a weight of zero keeps its pattern of zero: the scale is added to the exponent of the others only */
        const __mmask32 not_zero = _mm256_test_epi8_mask(codes, magnitude_mask);
        const __m512i exponent_step = _mm512_set1_epi16((short)((scale_codes[group] - E8M0_BIAS) * (1 << BF16_EXPONENT_SHIFT)));
        __m512i weights = _mm512_permutexvar_epi16(_mm512_cvtepu8_epi16(codes), table);
        weights = _mm512_mask_add_epi16(weights, not_zero, weights, exponent_step);
        sum = _mm512_dpbf16_ps(sum, (__m512bh)weights,
                               (__m512bh)_mm512_loadu_si512(input + group * WEIGHTS_PER_SCALE_GROUP));
    }
    return _mm512_reduce_add_ps(sum);
}

/* One expert on thread_count threads. Returns the seconds the activation between the two halves took. */
static double run_expert(const struct expert *expert, int use_bf16, int thread_count, float *output)
{
    static float gate[EXPERT_INNER_SIZE], up[EXPERT_INNER_SIZE], hidden[EXPERT_INNER_SIZE];
    static uint16_t hidden_bf16[EXPERT_INNER_SIZE];
    #pragma omp parallel for schedule(static) num_threads(thread_count)
    for (int row = 0; row < 2 * EXPERT_INNER_SIZE; row++) {
        if (row < EXPERT_INNER_SIZE) {
            gate[row] = use_bf16 ? row_dot_bf16(&expert->gate, row, input_bf16) : row_dot_fp32(&expert->gate, row, input_fp32);
        } else {
            const int up_row = row - EXPERT_INNER_SIZE;
            up[up_row] = use_bf16 ? row_dot_bf16(&expert->up, up_row, input_bf16) : row_dot_fp32(&expert->up, up_row, input_fp32);
        }
    }
    const double activation_started = seconds_now();
    for (int row = 0; row < EXPERT_INNER_SIZE; row++) {
        const float limited_gate = gate[row] < SWIGLU_LIMIT ? gate[row] : SWIGLU_LIMIT;
        const float limited_up = up[row] < -SWIGLU_LIMIT ? -SWIGLU_LIMIT : (up[row] > SWIGLU_LIMIT ? SWIGLU_LIMIT : up[row]);
        hidden[row] = limited_gate / (1.0f + expf(-limited_gate)) * limited_up * ROUTE_WEIGHT;
        hidden_bf16[row] = bf16_of(hidden[row]);
    }
    const double activation_seconds = seconds_now() - activation_started;
    #pragma omp parallel for schedule(static) num_threads(thread_count)
    for (int row = 0; row < HIDDEN_SIZE; row++) {
        output[row] = use_bf16 ? row_dot_bf16(&expert->down, row, hidden_bf16) : row_dot_fp32(&expert->down, row, hidden);
    }
    return activation_seconds;
}

static int check_expert(const struct expert *expert, int expert_index, int use_bf16, int must_differ)
{
    static float output[HIDDEN_SIZE];
    int output_rows[OUTPUTS_CHECKED];
    double reference[OUTPUTS_CHECKED];
    double worst_error = 0.0, squares = 0.0;
    for (int index = 0; index < OUTPUTS_CHECKED; index++) {
        output_rows[index] = (int)(mix_bits((uint64_t)expert_index * 131u + (uint64_t)index) % HIDDEN_SIZE);
    }
    reference_outputs(expert, output_rows, OUTPUTS_CHECKED, reference);
    if (must_differ) {
        memset(expert->gate.nibbles, CORRUPTED_BYTE_VALUE, CORRUPTED_BYTES);
    }
    (void)run_expert(expert, use_bf16, omp_get_max_threads(), output);
    for (int index = 0; index < OUTPUTS_CHECKED; index++) {
        const double error = fabs((double)output[output_rows[index]] - reference[index]);
        worst_error = error > worst_error ? error : worst_error;
        squares += reference[index] * reference[index];
    }
    const double rms = sqrt(squares / OUTPUTS_CHECKED);
    const double tolerance = (use_bf16 ? BF16_RELATIVE_TOLERANCE : FP32_RELATIVE_TOLERANCE) * rms;
    const int matches = worst_error <= tolerance;
    printf("  %s kernel, expert %d: worst error %.3e against rms %.3e (%.1e of it): %s\n", use_bf16 ? "bf16" : "fp32",
           expert_index, worst_error, rms, worst_error / rms, matches ? "matches the reference" : "DIFFERS");
    return must_differ ? !matches : matches;
}

int main(int argument_count, char **arguments)
{
    int expert_count = DEFAULT_EXPERT_COUNT, step_count = DEFAULT_STEP_COUNT, corrupt = 0;
    for (int argument_index = 1; argument_index < argument_count; argument_index++) {
        const int has_value = argument_index + 1 < argument_count;
        if (strcmp(arguments[argument_index], "--corrupt") == 0) {
            corrupt = 1;
        } else if (strcmp(arguments[argument_index], "--small-pages") == 0) {
            use_huge_pages = 0;
        } else if (strcmp(arguments[argument_index], "--experts") == 0 && has_value) {
            expert_count = atoi(arguments[++argument_index]);
        } else if (strcmp(arguments[argument_index], "--steps") == 0 && has_value) {
            step_count = atoi(arguments[++argument_index]);
        } else {
            fprintf(stderr, "usage: %s [--experts N] [--steps N] [--small-pages] [--corrupt]\n", arguments[0]);
            return EXIT_FAILURE;
        }
    }
    if (expert_count < EXPERTS_PER_ROW || step_count < 1) {
        fprintf(stderr, "--experts at least %d, --steps at least 1\n", EXPERTS_PER_ROW);
        return EXIT_FAILURE;
    }
    if (!__builtin_cpu_supports("avx512bf16") || !__builtin_cpu_supports("avx512bw")) {
        fprintf(stderr, "this probe needs AVX-512 with BW and BF16\n");
        return EXIT_FAILURE;
    }
    for (int code = 0; code < 256; code++) {
        scale_times_half[code] = (float)ldexp(0.5, code - E8M0_BIAS);
    }
    for (int code = 0; code < 2 * MAGNITUDE_CODES; code++) {
        const double magnitude = magnitude_of_code[code & (MAGNITUDE_CODES - 1)];
        bf16_of_code[code] = bf16_of((float)((code & SIGN_BIT) ? -magnitude : magnitude));
        bf16_of_code[code + 2 * MAGNITUDE_CODES] = bf16_of_code[code];
    }
    for (int column = 0; column < HIDDEN_SIZE; column++) {
        input_fp32[column] = ((float)(mix_bits((uint64_t)column + 7u) % INPUT_VALUE_STEPS) - INPUT_VALUE_OFFSET + 1.0f) / INPUT_VALUE_DIVISOR;
        input_bf16[column] = bf16_of(input_fp32[column]);
        if (float_of_bf16(input_bf16[column]) != input_fp32[column]) {
            fprintf(stderr, "a synthetic input is not exact in bf16\n");
            return EXIT_FAILURE;
        }
    }
    const double expert_bytes = 3.0 * EXPERT_INNER_SIZE * HIDDEN_SIZE / 2.0 + (2.0 * EXPERT_INNER_SIZE * HIDDEN_SIZE + (double)HIDDEN_SIZE * EXPERT_INNER_SIZE) / WEIGHTS_PER_SCALE_GROUP;
    struct expert *experts = calloc((size_t)expert_count, sizeof(*experts));
    if (experts == NULL) {
        return EXIT_FAILURE;
    }
    #pragma omp parallel for schedule(dynamic)
    for (int expert_index = 0; expert_index < expert_count; expert_index++) {
        const uint64_t seed = (uint64_t)expert_index << 40;
        fill_matrix(&experts[expert_index].gate, EXPERT_INNER_SIZE, HIDDEN_SIZE, seed + 1);
        fill_matrix(&experts[expert_index].up, EXPERT_INNER_SIZE, HIDDEN_SIZE, seed + (1ull << 36));
        fill_matrix(&experts[expert_index].down, HIDDEN_SIZE, EXPERT_INNER_SIZE, seed + (2ull << 36));
    }
    printf("weights on %s pages; ", use_huge_pages ? "2 MiB" : "4 KiB");
    printf("%d experts of %.2f MiB (%.2f GiB in all), %d hardware threads; one expert over a link: %.1f ms on four "
           "lanes, %.1f ms on sixteen, then %.0f us of card time\n", expert_count, expert_bytes / BYTES_PER_MIB,
           expert_bytes * expert_count / BYTES_PER_MIB / 1024.0, omp_get_max_threads(),
           expert_bytes / BYTES_PER_GB / LINK_FOUR_LANES_GB_PER_SECOND * 1000.0,
           expert_bytes / BYTES_PER_GB / LINK_SIXTEEN_LANES_GB_PER_SECOND * 1000.0, CARD_MICROSECONDS_PER_EXPERT);

    int all_as_required = 1;
    for (int use_bf16 = 0; use_bf16 <= 1; use_bf16++) {
        all_as_required &= check_expert(&experts[1], 1, use_bf16, 0);
        all_as_required &= check_expert(&experts[expert_count - 1], expert_count - 1, use_bf16, 0);
    }
    if (corrupt) {
        all_as_required &= check_expert(&experts[2], 2, 0, 1);
        all_as_required &= check_expert(&experts[3], 3, 1, 1);
        printf("%s\n", all_as_required ? "the damaged matrices were noticed" : "FAILED: a check did not come out as required");
        return all_as_required ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    static float output[HIDDEN_SIZE];
    for (int use_bf16 = 0; use_bf16 <= 1; use_bf16++) {
        printf("%s kernel, one cold expert at a time:\n", use_bf16 ? "bf16" : "fp32");
        for (size_t count_case = 0; count_case < THREAD_COUNT_CASES; count_case++) {
            const int thread_count = thread_counts[count_case];
            if (thread_count > omp_get_max_threads()) {
                continue;
            }
            double activation_seconds = 0.0, started = 0.0;
            for (int step = 0; step < WARM_UP_STEPS + step_count; step++) {
                if (step == WARM_UP_STEPS) {
                    started = seconds_now();
                    activation_seconds = 0.0;
                }
                activation_seconds += run_expert(&experts[(step * 37 + (int)count_case * 11) % expert_count], use_bf16, thread_count, output);
            }
            const double seconds_per_expert = (seconds_now() - started) / step_count;
            printf("  %2d threads: %8.1f us per expert (%5.1f us of it the activation on one thread), %6.1f GB/s, "
                   "%5.1f us for six = one layer of one row\n", thread_count, seconds_per_expert * MICROSECONDS_PER_SECOND,
                   activation_seconds / step_count * MICROSECONDS_PER_SECOND, expert_bytes / BYTES_PER_GB / seconds_per_expert,
                   seconds_per_expert * MICROSECONDS_PER_SECOND * EXPERTS_PER_ROW);
        }
    }
    printf("%s\n", all_as_required ? "every check matched the reference" : "FAILED: a check did not match the reference");
    return all_as_required ? EXIT_SUCCESS : EXIT_FAILURE;
}
