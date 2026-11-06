/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * cpu_expert_forms_probe: a routed expert computed by the processor in whole numbers, in the forms that could make
 * that cheaper, and what a cheaper kernel is worth where the memory bounds the step (speed campaign, round 1, unit
 * 8b; ledger rows C3 and C14; B5 is the batch's). Unit 8a found one cold expert on 96 threads at 107 to 121 us =
 * wake 5 + work 85 + activation 28 + tail 3 and the bytes alone at 72 to 85 us: there no kernel shortens the work,
 * a cheaper one pays as fewer cores for the same time, and the serial activation is what a kernel can still remove.
 * Instances (--instance NAME, repeatable; all of them without it; a ratio is to `baseline` of the same case):
 *   baseline, baseline_again  8a's fp32 kernel (b04's), its activation on the calling thread;
 *   read                      8a's: the same threads sum the bytes of the same rows and do nothing else;
 *   vnni                      strips of 16 output rows; weights as unsigned bytes (the value doubled plus 12, by a
 *                             16-entry byte shuffle), the E4M3 input in units of 2^-9 as three signed byte digits,
 *                             one VPDPBUSD per digit and 64 weights; per group of 32 inputs the exact sum S, then
 *                             the cards' chain (linear_records.h): chain = fma(S, 2^e, chain) per slice of the input
 *                             width, four slices added in order; gate, up and the hidden vector are cast to bfloat16
 *                             (the last is quantized again); down is left as the fp32 sum, BEFORE the cards' cast;
 *   vnni,<lever>:<value>      spread:shift (nibbles to bytes by shift and mask, not VPMULTISHIFTQB); accumulators:2;
 *                             unroll:1 (a group's eight steps as a loop); pass:joined (gate and up of a block of 32
 *                             rows in one pass); activation:inline (with joined: the thread that holds gate and up
 *                             of a block applies the activation and quantizes it: the caller's loop is gone); and
 *                             accumulators:2 with each of the last two.
 * A case is one value each of --experts (cold experts at once, the threads shared evenly), --pool, --placement and
 * --threads, with ONE pinned team that lives as long as the case (8a's teams of one slot timed cores below their
 * clock). Rounds, the baseline twice, statistics and lines are probe_harness.h's, whose dense layout this file has;
 * a step is timed by the time-stamp counter around its two regions and what lies between, a stream is 64 steps back
 * to back. Every step takes its experts in turn from a ring that no cache holds (8a's, and the same as strips).
 *
 * Checked in every case before anything is timed, and shown failing by --break NAME (nothing is timed; exit 0 only
 * if what the name requires was reported and nothing else):
 *   shares_inside   every thread's units lie inside its matrix; no kernel gets a refused share  (--break share)
 *   integer_sums    every group sum of the first expert, kept by a twin of the instance that differs only in
 *                   storing them, against whole numbers in plain code from the definitions: it holds only if every
 *                   digit, bias and exponent the block quantizer wrote is right; the input has zeros of both signs,
 *                   blocks under the floor, the largest code, ties and subnormal codes  (--break plane: one digit
 *                   plane of the input negated; --break corrupt: a megabyte of the first checked expert's gate
 *                   matrix overwritten in ring and strips after the references were taken)
 *   same_bits       hidden vector and output, bit for bit, against the plain code's chain: every form, thread
 *                   count, placement and pool                                                    (corrupt, plane)
 *   host_reference  the output against double precision from the format's definition (vnni: from the plain chain's
 *                   own quantized hidden vector): 1e-6 of the rms                (corrupt; for vnni also plane)
 *   byte_sum (read; corrupt); pinning, and pinning_after once a case is timed (where a thread runs; --break pinning).
 * A kernel indexes by the units of a share that passed shares_inside, by ring positions taken modulo the ring and
 * by a stream slot the harness bounds; weights and scales reach arithmetic only.
 * Lines beyond the harness's: HOST, TIMELINE (--timeline: 8a's terms as MEANS over all a case's single steps, its
 * warm-up's too: wake, work, tail of the thread that left its region last, the caller between the regions), VERDICT
 * figure=fewest_threads (per instance over a list of thread counts: the fewest threads within 5% of the most threads'
 * median, and read's time there), PERF (--profile-case, as root: cycles and instructions per weight of the profiled
 * steps), DIAG (debug: the group sums at or above 2^24 units, which fp32 need not hold, from the last check step on).
 * RESULT's part= is the case; BEST may name read, which computes nothing.
 * Not shown: a card's bits (no card is asked, and expf is the C library's); the cards' lane order read as it is;
 * software prefetch (ledger C7); several rows on one expert; a pool that sleeps; real weights (bytes are hashes);
 * the memory modules' temperatures (the batch gates every step on them and prints them).
 *   cpu_expert_forms_probe [--experts L] [--pool openmp|own,..] [--placement die|packed|smt,..] [--threads L]
 *       [--timeline] [--ring-experts N] [--break corrupt|plane|share|pinning] and the flags of probe_harness.h
 *
 * Measurement tool: not part of the engine.
 */
#include "cpu_probe_common.h"
#include "probe_harness.h"
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>

enum probe_constant {
    KERNEL_FP32 = 0, KERNEL_READ, KERNEL_VNNI, REGIONS_PER_STEP = 2, MAX_EXPERTS = 9, MAX_THREADS = 256, PAIR = 2,
    HIDDEN_SIZE = CPU_HIDDEN_SIZE, INNER_SIZE = CPU_EXPERT_INNER_SIZE, GROUP_INPUTS = CPU_WEIGHTS_PER_SCALE_GROUP,
    GATE_UP_ROWS = PAIR * INNER_SIZE, RING_EXPERTS_MIN = 12, RING_EXPERTS_MAX = 4096, RING_STRIDE = 37,
    FIRST_CHECKED_EXPERT = 1, CACHES_OF_RING_MIN = 4, COUNTERS = 2, PER_WEIGHT_DECIMALS = 4, STALE_BYTE = 0xFF,
    /* the integer kernel: a strip is 16 output rows, a step four inputs of them (a dword a row), a group 8 steps */
    ROWS_PER_STRIP = 16, INPUTS_PER_STEP = 4, STEPS_PER_GROUP = GROUP_INPUTS / INPUTS_PER_STEP,
    STEP_BYTES = ROWS_PER_STRIP * INPUTS_PER_STEP / CPU_NIBBLES_PER_BYTE, CODE_BYTES = STEPS_PER_GROUP * STEP_BYTES,
    GROUP_BYTES = CODE_BYTES + ROWS_PER_STRIP, INPUT_SLICES = 4, DIGIT_PLANES = 3, LAST_PLANE = DIGIT_PLANES - 1,
    DIGIT_BITS = 8, DIGIT_BIASES = 0x8080, WEIGHT_BIAS = 12, MATRICES_MOST = 2, SETS_MOST = 2, WIDE_SUM_LEAST = 1 << 24,
    INNER_STRIPS = INNER_SIZE / ROWS_PER_STRIP, STRIPS_PER_BLOCK = 2, DOWN_STRIPS = HIDDEN_SIZE / ROWS_PER_STRIP,
    HIDDEN_BLOCKS = INNER_SIZE / GROUP_INPUTS, GATE_GROUPS = HIDDEN_SIZE / GROUP_INPUTS, DOWN_GROUPS = HIDDEN_BLOCKS,
    MATRIX_GROUP_SUMS = INNER_SIZE * GATE_GROUPS,                    /* of each of the three matrices */
    /* E4M3 (sign, four exponent bits, three of mantissa), fp32's fields, the cast to bfloat16 */
    E4M3_SIGN_BIT = 0x80, E4M3_NAN_MAGNITUDE = 0x7F, E4M3_EXPONENT_SHIFT = 3, E4M3_EXPONENT_MASK = 15,
    E4M3_MANTISSA_MASK = 7, E4M3_IMPLICIT_ONE = 8, E4M3_UNIT_EXPONENT = 9, FP32_EXPONENT_SHIFT = 23,
    FP32_EXPONENT_BIAS = 127, FP32_MANTISSA_MASK = 0x7FFFFF, E4M3_SMALLEST_NORMAL_FIELD = FP32_EXPONENT_BIAS - 6,
    E4M3_STEPS_FIELD_BASE = 2 * FP32_EXPONENT_BIAS + E4M3_EXPONENT_SHIFT, ACTIVATION_EXPONENT_SMALLEST = 105,
    BF16_DROPPED_BITS = 16, BF16_DROPPED_MASK = 0xFFFF, BF16_ROUNDING_BIAS = 0x7FFF,
    /* a group's sum counts a weight doubled and an input in units of 2^-9: S 2^(w + a - 264) is the true sum, and
     * the fp32 exponent field of that power is w + a - 137 */
    PRODUCT_SCALE_BIAS = 2 * CPU_E8M0_BIAS + E4M3_UNIT_EXPONENT + 1,
    SCALE_FIELD_BIAS = PRODUCT_SCALE_BIAS - FP32_EXPONENT_BIAS,
    /* the synthetic input: 8a's values, per block divided by 2^0 .. 2^5 or one of four corners of the quantizer; the
     * tie block is divided by 2^8 or 2^9 (same codes): at 448 it drove the gates to their limit and hid up's errors */
    INPUT_SEED = 7, INPUT_VALUE_STEPS = 255, INPUT_BLOCK_KINDS = 10, INPUT_PLAIN_KINDS = 6, INPUT_ZEROS = 6, INPUT_TINY,
    INPUT_TIES, INPUT_TIE_SHIFT = 8, INPUT_SPREAD_BITS = 18, CORRUPTED_BYTES = 1024 * 1024, CORRUPTED_BYTE_VALUE = 0x5A
};
_Static_assert(FIRST_CHECKED_EXPERT + MAX_EXPERTS <= RING_EXPERTS_MIN, "the checked experts lie in any ring");
_Static_assert(GATE_GROUPS % INPUT_SLICES == 0 && DOWN_GROUPS % INPUT_SLICES == 0 && INNER_SIZE % GROUP_INPUTS == 0 &&
               HIDDEN_SIZE % ROWS_PER_STRIP == 0, "whole slices, blocks and strips: no kernel has a tail");
_Static_assert(CPU_SCALE_CODE_BASE - CORRUPTED_BYTE_VALUE >= 0 && CORRUPTED_BYTE_VALUE + ACTIVATION_EXPONENT_SMALLEST >
               SCALE_FIELD_BIAS, "a group's power of two is a normal fp32, also where a scale was overwritten");
#define SWIGLU_LIMIT 10.0f
#define ROUTE_WEIGHT 0.25f
#define AMAX_FLOOR 1.0e-4f                       /* the reference's lower bound of a block's largest magnitude */
#define E4M3_MAXIMUM 448.0f
#define E4M3_MAXIMUM_INVERSE (1.0f / E4M3_MAXIMUM)
#define E4M3_SMALLEST_NORMAL 0.015625f           /* 2^-6: codes below it are 2^-9 apart, as those just above it */
#define E4M3_UNITS_PER_ONE 512.0
#define INPUT_VALUE_OFFSET 127.0f                /* synthetic inputs -2 .. 2 in steps of 1/64, as b04's */
#define INPUT_VALUE_DIVISOR 64.0f
#define INPUT_TINY_VALUE 3.0e-6f                 /* 31 of them stay under the quantizer's floor */
#define INPUT_TIE_GAP 16.0f                      /* 448 less an odd multiple of it lies midway between two codes */
#define INPUT_SPREAD_VALUE 1.5f                  /* halved up to 17 times: the small ones become subnormal codes */
#define FP32_RMS_ERROR_LIMIT 1.0e-6              /* CONVENTIONS.md 5, check 1 */
#define VNNI_BAR_RATIO 0.40                      /* ROUND_1.md: vnni on one thread against fp32 */
#define FEWEST_THREADS_RATIO 1.05
#define MULTISHIFT_NIBBLE_STARTS 0x2C2824200C080400ll    /* bits 0, 4, 8, 12 of each of a qword's two dwords */
#define OR_THEN_AND 0xA8                         /* VPTERNLOG's table of (a | b) & c */
#define KERNEL_INLINE static inline __attribute__((always_inline))

enum spread { SPREAD_MULTISHIFT, SPREAD_SHIFT };
enum unit_kind { UNIT_ROWS_GATE_UP, UNIT_ROWS_DOWN, UNIT_STRIPS_GATE_UP, UNIT_BLOCKS, UNIT_STRIPS_DOWN, UNIT_KINDS };
enum timeline_term { TERM_WAKE, TERM_WORK, TERM_TAIL, TERM_SERIAL, TERM_STEPS, TERM_COUNT };
static const int units_of_kind[] = {GATE_UP_ROWS, HIDDEN_SIZE, PAIR * INNER_STRIPS, HIDDEN_BLOCKS, DOWN_STRIPS};
static const int twice_weight_of_code[CPU_NIBBLE_CODES] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
static const uint8_t unsigned_weight[CPU_NIBBLE_CODES] = {12, 13, 14, 15, 16, 18, 20, 24, 12, 11, 10, 9, 8, 6, 4, 0};

/* An instance of the integer kernel (constants where it is stamped). What a block of 32 inputs gives every output's
 * group: 12 x the sum of its units, its E8M0 code less SCALE_FIELD_BIAS. A quantized vector as the kernel reads it. */
struct kernel_form { int spread, accumulator_sets, unrolled, joined, inline_activation; };
struct group_terms { int32_t bias, exponent_offset; };
struct operand { int8_t digits[DIGIT_PLANES][HIDDEN_SIZE]; struct group_terms terms[GATE_GROUPS]; };
struct share { int expert_slot, first_unit, unit_end; };         /* what a thread does in a region */
struct thread_line { _Alignas(CPU_CACHE_LINE_BYTES) uint64_t byte_sum, wide_sums; };
struct reference {                               /* of one checked expert, by plain code */
    double fp32_output[HIDDEN_SIZE];             /* b04's contract in double precision */
    float hidden[INNER_SIZE], output[HIDDEN_SIZE];       /* the integer contract: its chain, */
    double output_double[HIDDEN_SIZE];           /* and its output in double precision from its hidden vector */
};
static struct {
    double ticks_per_microsecond;                /* the machine, the ring and its strips, the case and its team */
    struct cpu_topology topology;
    struct cpu_ring ring;
    uint8_t *(*strips)[CPU_MATRIX_KINDS];        /* [expert][matrix]: see pack_matrix */
    struct cpu_team team;
    int experts, threads, refused_shares, timeline, announced_cpus[MAX_THREADS], pinned_cpus[MAX_THREADS];
    struct share shares[UNIT_KINDS][MAX_THREADS];
    int slot_experts[PROBE_STREAM_STEPS][MAX_EXPERTS];
    const int *step_experts;                     /* the ring positions of the step in hand */
    float input[HIDDEN_SIZE];                    /* what the threads of the team read and write */
    struct operand input_operand, hidden_operand[MAX_EXPERTS];
    float gate_up[MAX_EXPERTS][GATE_UP_ROWS], hidden[MAX_EXPERTS][INNER_SIZE], output[MAX_EXPERTS][HIDDEN_SIZE];
    struct thread_line lines[REGIONS_PER_STEP][MAX_THREADS];
    struct cpu_thread_stamps stamps[REGIONS_PER_STEP][MAX_THREADS];
    struct cpu_region_record records[REGIONS_PER_STEP];
    uint64_t started_ticks, steps_since_mark, counted_steps;     /* a timed step, the timeline, the counters, */
    uint32_t step_instance;                                      /* the medians of a list of thread counts */
    double terms[PROBE_MOST_INSTANCES][TERM_COUNT], medians[PROBE_MOST_LIST_VALUES][PROBE_MOST_INSTANCES];
    int case_threads[PROBE_MOST_LIST_VALUES], counters[COUNTERS], counting;
} run;
static struct reference references[MAX_EXPERTS];
static int32_t reference_sums[CPU_MATRIX_KINDS][MATRIX_GROUP_SUMS], twin_sums[CPU_MATRIX_KINDS][MATRIX_GROUP_SUMS];

union float_word { float value; uint32_t bits; };
static int broken(const char *name) { return probe_flag_has_word("--break", name); }
static uint32_t float_bits(float value) { return ((union float_word){.value = value}).bits; }
/* An fp32 as the cast to bfloat16 keeps it: its upper half, to nearest, a tie to even. */
static float bf16_value(float value) {
    const uint32_t bits = float_bits(value), odd = (bits >> BF16_DROPPED_BITS) & 1u;
    return ((union float_word){.bits = (bits + BF16_ROUNDING_BIAS + odd) & ~(uint32_t)BF16_DROPPED_MASK}).value;
}
KERNEL_INLINE __m512 bf16_values(__m512 values) {
    const __m512i bits = _mm512_castps_si512(values);
    const __m512i odd = _mm512_and_si512(_mm512_srli_epi32(bits, BF16_DROPPED_BITS), _mm512_set1_epi32(1));
    const __m512i biased = _mm512_add_epi32(_mm512_add_epi32(bits, _mm512_set1_epi32(BF16_ROUNDING_BIAS)), odd);
    return _mm512_castsi512_ps(_mm512_and_si512(biased, _mm512_set1_epi32(~BF16_DROPPED_MASK)));
}
/* The hidden step as b04 writes it: h = silu(min(gate, L)) * clamp(up, -L, L) * route weight, in fp32. */
static float hidden_value(float gate, float up) {
    const float limited_gate = gate < SWIGLU_LIMIT ? gate : SWIGLU_LIMIT;
    const float limited_up = up < -SWIGLU_LIMIT ? -SWIGLU_LIMIT : (up > SWIGLU_LIMIT ? SWIGLU_LIMIT : up);
    return limited_gate / (1.0f + expf(-limited_gate)) * limited_up * ROUTE_WEIGHT;
}
/* Rows first_row .. row_end - 1 of a slot's hidden vector from its gate and up; casts: to bfloat16, as the integer
 * kernel's down product quantizes it. The caller's loop, or (activation:inline) each thread's for its own rows. */
static void activate_rows(int slot, int first_row, int row_end, int casts) {
    const float *gate = run.gate_up[slot], *up = gate + INNER_SIZE;
    for (int row = first_row; row < row_end; row++) {
        const float hidden = hidden_value(gate[row], up[row]);
        run.hidden[slot][row] = casts ? bf16_value(hidden) : hidden;
    }
}
/* linear_records.h, "Quantization", for one block of 32 values: the largest magnitude (at least 1e-4) times 1 / 448
 * in fp32, rounded up to a power of two read from its bits, is the scale; a value over it, limited to 448, becomes
 * the nearest E4M3 number, a tie to the even code. Of fp32 exponent field f the codes are 2^(f - 121) units of 2^-9
 * apart, so a code's units are round(magnitude 2^(130 - f)) << (f - 121), with the value's sign. A + 0x8080 has in
 * its bytes the digits of A = d0 + 256 d1 + 65536 d2 (d0, d1 in -128 .. 127), the first two plus 128: the sign is
 * in before the split, since -128 has no negative in a byte. Finite values only. */
KERNEL_INLINE void quantize_block(const float *values, struct operand *operand, int block) {
    const __m512 halves[PAIR] = {_mm512_loadu_ps(values), _mm512_loadu_ps(values + CPU_FLOATS_PER_VECTOR)};
    const float largest = _mm512_reduce_max_ps(_mm512_max_ps(_mm512_abs_ps(halves[0]), _mm512_abs_ps(halves[1])));
    const uint32_t ratio_bits = float_bits((largest > AMAX_FLOOR ? largest : AMAX_FLOOR) * E4M3_MAXIMUM_INVERSE);
    const int exponent = (int)(ratio_bits >> FP32_EXPONENT_SHIFT) + ((ratio_bits & FP32_MANTISSA_MASK) != 0u);
    const __m512 inverse_scale = _mm512_castsi512_ps(
        _mm512_set1_epi32((FP32_EXPONENT_BIAS + CPU_E8M0_BIAS - exponent) << FP32_EXPONENT_SHIFT));
    const __m512i digit_biases = _mm512_set1_epi32(DIGIT_BIASES);
    int units_sum = 0;
    for (int half = 0; half < PAIR; half++) {
        const __m512 scaled = _mm512_mul_ps(halves[half], inverse_scale);        /* by a power of two: exact */
        const __m512 magnitude = _mm512_min_ps(_mm512_abs_ps(scaled), _mm512_set1_ps(E4M3_MAXIMUM));
        const __m512i field = _mm512_srli_epi32(
            _mm512_castps_si512(_mm512_max_ps(magnitude, _mm512_set1_ps(E4M3_SMALLEST_NORMAL))), FP32_EXPONENT_SHIFT);
        const __m512 steps_per_unit = _mm512_castsi512_ps(_mm512_slli_epi32(
            _mm512_sub_epi32(_mm512_set1_epi32(E4M3_STEPS_FIELD_BASE), field), FP32_EXPONENT_SHIFT));
        const __m512i steps = _mm512_cvtps_epi32(_mm512_mul_ps(magnitude, steps_per_unit));     /* ties to even */
        const __m512i units = _mm512_sllv_epi32(
            steps, _mm512_sub_epi32(field, _mm512_set1_epi32(E4M3_SMALLEST_NORMAL_FIELD)));
        const __m512i signed_units = _mm512_mask_sub_epi32(
            units, _mm512_movepi32_mask(_mm512_castps_si512(scaled)), _mm512_setzero_si512(), units);
        const __m512i digits = _mm512_xor_si512(_mm512_add_epi32(signed_units, digit_biases), digit_biases);
        const int place = block * GROUP_INPUTS + half * CPU_FLOATS_PER_VECTOR;
        for (int plane = 0; plane < DIGIT_PLANES; plane++) {
            _mm_storeu_si128((void *)(operand->digits[plane] + place),
                             _mm512_cvtepi32_epi8(_mm512_srli_epi32(digits, (unsigned)(plane * DIGIT_BITS))));
        }
        units_sum += _mm512_reduce_add_epi32(signed_units);
    }
    operand->terms[block] = (struct group_terms){WEIGHT_BIAS * units_sum, exponent - SCALE_FIELD_BIAS};
}

/* ---- the integer kernel ---- */

/* One step of a group: 16 rows by 4 inputs of each matrix against the three digits of those inputs. The 32 bytes
 * of a step hold in byte 2 r + b row r's inputs 2 b (low nibble) and 2 b + 1; they become a dword a row and a byte
 * an input. sums[2 matrix] are the matrix's three accumulators of the set the caller points at. */
KERNEL_INLINE void group_step(const struct kernel_form form, int matrices, const uint8_t *const *strips,
                              const struct operand *operand, int group, int step, __m512i (*sums)[DIGIT_PLANES]) {
    const __m512i nibble_mask = _mm512_set1_epi8(CPU_NIBBLE_MASK);
    const __m512i weight_table = _mm512_broadcast_i32x4(_mm_loadu_si128((const void *)unsigned_weight));
    const int place = group * GROUP_INPUTS + step * INPUTS_PER_STEP;
    __m512i digits[DIGIT_PLANES];
    for (int plane = 0; plane < DIGIT_PLANES; plane++) {
        int32_t four_inputs = 0;
        memcpy(&four_inputs, operand->digits[plane] + place, sizeof(four_inputs));
        digits[plane] = _mm512_set1_epi32(four_inputs);
    }
    for (int matrix = 0; matrix < matrices; matrix++) {
        const __m256i packed = _mm256_loadu_si256(
            (const void *)(strips[matrix] + (size_t)(group * GROUP_BYTES + step * STEP_BYTES)));
        const __m512i words = _mm512_cvtepu8_epi16(packed);
        const __m512i codes = form.spread == SPREAD_MULTISHIFT
            ? _mm512_and_si512(_mm512_multishift_epi64_epi8(_mm512_set1_epi64(MULTISHIFT_NIBBLE_STARTS),
                                                            _mm512_cvtepu16_epi32(packed)), nibble_mask)
            : _mm512_ternarylogic_epi64(words, _mm512_slli_epi16(words, CPU_NIBBLE_BITS), nibble_mask, OR_THEN_AND);
        const __m512i weights = _mm512_shuffle_epi8(weight_table, codes);
        __m512i *matrix_sums = sums[matrix * SETS_MOST];
        for (int plane = 0; plane < DIGIT_PLANES; plane++) {
            matrix_sums[plane] = _mm512_dpbusd_epi32(matrix_sums[plane], weights, digits[plane]);
        }
    }
}
/* 16 output rows of one matrix, or of two that share the input (gate and up in one pass), against a quantized
 * vector of `groups` blocks: per group the exact sums S, chain = fma(S, 2^e, chain) in each of four slices of the
 * input width, the slices added from +0 in order. results gets a register of fp32 sums per matrix. sums_out, not
 * null in a twin instance only, gets every S: [matrix] -> [group][row]. Nothing here depends on a weight's value.
 * sums[2 matrix + set] are a group's accumulators; with two sets a step's odd half adds to the second. */
KERNEL_INLINE void strip_products(const struct kernel_form form, int matrices, const uint8_t *const *strips,
                                  const struct operand *operand, int groups, __m512 *results, int32_t *const *sums_out,
                                  uint64_t *wide_sums) {
    const int groups_per_slice = groups / INPUT_SLICES, odd_set = form.accumulator_sets - 1;
    (void)wide_sums;
    for (int matrix = 0; matrix < matrices; matrix++) { results[matrix] = _mm512_setzero_ps(); }
    for (int slice = 0; slice < INPUT_SLICES; slice++) {
        __m512 chain[MATRICES_MOST] = {_mm512_setzero_ps(), _mm512_setzero_ps()};
        for (int group = slice * groups_per_slice; group < (slice + 1) * groups_per_slice; group++) {
            const __m512i bias = _mm512_set1_epi32(operand->terms[group].bias);
            const __m512i exponent_offset = _mm512_set1_epi32(operand->terms[group].exponent_offset);
            __m512i sums[MATRICES_MOST * SETS_MOST][DIGIT_PLANES];
            for (int place = 0; place < MATRICES_MOST * SETS_MOST * DIGIT_PLANES; place++) {
                sums[place / DIGIT_PLANES][place % DIGIT_PLANES] = _mm512_setzero_si512();
            }
            if (form.unrolled) {                 /* two steps a turn: each names its set of accumulators */
                #pragma GCC unroll STEPS_PER_GROUP / PAIR
                for (int step = 0; step < STEPS_PER_GROUP; step += PAIR) {
                    group_step(form, matrices, strips, operand, group, step, sums);
                    group_step(form, matrices, strips, operand, group, step + 1, sums + odd_set);
                }
            } else {
                #pragma GCC unroll 1
                for (int step = 0; step < STEPS_PER_GROUP; step++) {
                    group_step(form, matrices, strips, operand, group, step, sums + step % form.accumulator_sets);
                }
            }
            for (int matrix = 0; matrix < matrices; matrix++) {
                __m512i (*const sets)[DIGIT_PLANES] = sums + matrix * SETS_MOST, digit_sums[DIGIT_PLANES];
                for (int plane = 0; plane < DIGIT_PLANES; plane++) {     /* an unused second set stayed zero */
                    digit_sums[plane] = _mm512_add_epi32(sets[0][plane], sets[1][plane]);
                }
                const __m512i sum = _mm512_sub_epi32(
                    _mm512_add_epi32(_mm512_add_epi32(digit_sums[0], _mm512_slli_epi32(digit_sums[1], DIGIT_BITS)),
                                     _mm512_slli_epi32(digit_sums[LAST_PLANE], LAST_PLANE * DIGIT_BITS)), bias);
                const __m512i scale_codes = _mm512_cvtepu8_epi32(
                    _mm_loadu_si128((const void *)(strips[matrix] + (size_t)(group * GROUP_BYTES + CODE_BYTES))));
                const __m512 scale = _mm512_castsi512_ps(
                    _mm512_slli_epi32(_mm512_add_epi32(scale_codes, exponent_offset), FP32_EXPONENT_SHIFT));
                chain[matrix] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(sum), scale, chain[matrix]);
                if (sums_out != NULL) { _mm512_storeu_si512(sums_out[matrix] + (size_t)group * ROWS_PER_STRIP, sum); }
#if PROBE_DIAGNOSTICS
                *wide_sums += (uint64_t)__builtin_popcount((unsigned)_mm512_cmpgt_epi32_mask(
                    _mm512_abs_epi32(sum), _mm512_set1_epi32(WIDE_SUM_LEAST - 1)));
#endif
            }
        }
        for (int matrix = 0; matrix < matrices; matrix++) {
            results[matrix] = _mm512_add_ps(results[matrix], chain[matrix]);
        }
    }
}
/* A thread's share of a region. The first: strips of gate, then of up (units 0 .. 287), or blocks of 32 rows of
 * both (joined), whose hidden values the thread then makes and quantizes itself (activation:inline). The second:
 * strips of down, left as fp32 sums (the cards cast theirs). `region` and `keeps_sums` are constants where stamped. */
KERNEL_INLINE void vnni_work(const struct kernel_form form, int region, int thread_index, int keeps_sums) {
    const int joined = region == 0 && form.joined, matrices = joined ? MATRICES_MOST : 1;
    const int unit_kind = region != 0 ? UNIT_STRIPS_DOWN : (joined ? UNIT_BLOCKS : UNIT_STRIPS_GATE_UP);
    const int groups = region != 0 ? DOWN_GROUPS : GATE_GROUPS;
    const struct share share = run.shares[unit_kind][thread_index];
    const int slot = share.expert_slot;
    uint8_t *const *expert = run.strips[run.step_experts[slot]];
    for (int unit = share.first_unit; unit < share.unit_end; unit++) {
        PROBE_INDEX_CHECK(unit, units_of_kind[unit_kind]);
        for (int part = 0; part < (joined ? STRIPS_PER_BLOCK : 1); part++) {
            const int strip = region != 0 ? unit : (joined ? unit * STRIPS_PER_BLOCK + part : unit % INNER_STRIPS);
            const int first_kind = region != 0 ? CPU_DOWN : (joined ? CPU_GATE : unit / INNER_STRIPS);
            const uint8_t *strips[MATRICES_MOST];
            int32_t *sums[MATRICES_MOST];
            __m512 results[MATRICES_MOST];
            for (int matrix = 0; matrix < matrices; matrix++) {
                strips[matrix] = expert[first_kind + matrix] + (size_t)(strip * groups) * GROUP_BYTES;
                sums[matrix] = twin_sums[first_kind + matrix] + (size_t)(strip * groups * ROWS_PER_STRIP);
            }
            strip_products(form, matrices, strips, region != 0 ? &run.hidden_operand[slot] : &run.input_operand, groups,
                           results, keeps_sums && slot == 0 ? sums : NULL, &run.lines[region][thread_index].wide_sums);
            for (int matrix = 0; matrix < matrices; matrix++) {
                float *values = region != 0 ? run.output[slot] : run.gate_up[slot] + (first_kind + matrix) * INNER_SIZE;
                _mm512_storeu_ps(values + strip * ROWS_PER_STRIP,
                                 region != 0 ? results[matrix] : bf16_values(results[matrix]));
            }
        }
        if (joined && form.inline_activation) {
            activate_rows(slot, unit * GROUP_INPUTS, (unit + 1) * GROUP_INPUTS, 1);
            quantize_block(run.hidden[slot] + unit * GROUP_INPUTS, &run.hidden_operand[slot], unit);
        }
    }
}
/* The instances of the integer kernel: id, name, spread, sets of accumulators, a group's steps unrolled, gate and
 * up in one pass, the activation in the threads. Each is stamped as it is timed, and as the twin that also stores
 * its group sums; `down`, not null, is the second region. */
#define VNNI_FORMS(FORM)                                                                                          \
    FORM(plain, "vnni", SPREAD_MULTISHIFT, 1, 1, 0, 0)                                                            \
    FORM(shift, "vnni,spread:shift", SPREAD_SHIFT, 1, 1, 0, 0)                                                    \
    FORM(two_sets, "vnni,accumulators:2", SPREAD_MULTISHIFT, SETS_MOST, 1, 0, 0)                                  \
    FORM(looped, "vnni,unroll:1", SPREAD_MULTISHIFT, 1, 0, 0, 0)                                                  \
    FORM(joined, "vnni,pass:joined", SPREAD_MULTISHIFT, 1, 1, 1, 0)                                               \
    FORM(in_threads, "vnni,pass:joined,activation:inline", SPREAD_MULTISHIFT, 1, 1, 1, 1)                         \
    FORM(two_sets_joined, "vnni,accumulators:2,pass:joined", SPREAD_MULTISHIFT, SETS_MOST, 1, 1, 0)               \
    FORM(all, "vnni,accumulators:2,pass:joined,activation:inline", SPREAD_MULTISHIFT, SETS_MOST, 1, 1, 1)
#define FORM_FUNCTION(function_name, keeps_sums, ...)                                                             \
    static void function_name(void *down, int thread_index) {                                                    \
        if (down == NULL) { vnni_work((struct kernel_form){__VA_ARGS__}, 0, thread_index, keeps_sums); }          \
        if (down != NULL) { vnni_work((struct kernel_form){__VA_ARGS__}, 1, thread_index, keeps_sums); }          \
    }
#define FORM_FUNCTIONS(id, name, ...) FORM_FUNCTION(work_##id, 0, __VA_ARGS__) FORM_FUNCTION(twin_##id, 1, __VA_ARGS__)
VNNI_FORMS(FORM_FUNCTIONS)

/* 8a's kernels, a thread's rows as b04's loops gave them: gate's rows and then up's, or (down not null) down's. */
static void work_fp32(void *down, int thread_index) {
    const struct share share = run.shares[down != NULL ? UNIT_ROWS_DOWN : UNIT_ROWS_GATE_UP][thread_index];
    const struct cpu_matrix *matrix = run.ring.experts[run.step_experts[share.expert_slot]].matrix;
    const float *scales = run.ring.scale_times_half, *hidden = run.hidden[share.expert_slot];
    for (int row = share.first_unit; row < share.unit_end; row++) {
        if (down != NULL) {
            run.output[share.expert_slot][row] = cpu_row_dot_fp32(&matrix[CPU_DOWN], row, hidden, scales);
        } else {
            run.gate_up[share.expert_slot][row] =
                cpu_row_dot_fp32(&matrix[row / INNER_SIZE], row % INNER_SIZE, run.input, scales);
        }
    }
}
static void work_read(void *down, int thread_index) {
    const struct share share = run.shares[down != NULL ? UNIT_ROWS_DOWN : UNIT_ROWS_GATE_UP][thread_index];
    const struct cpu_matrix *matrix = run.ring.experts[run.step_experts[share.expert_slot]].matrix;
    const int gate_end = share.unit_end < INNER_SIZE ? share.unit_end : INNER_SIZE;
    const int up_first = share.first_unit > INNER_SIZE ? share.first_unit : INNER_SIZE;
    run.lines[down != NULL][thread_index].byte_sum =
        down != NULL ? cpu_sum_matrix_rows(&matrix[CPU_DOWN], share.first_unit, share.unit_end)
                     : cpu_sum_matrix_rows(&matrix[CPU_GATE], share.first_unit, gate_end) +
                           cpu_sum_matrix_rows(&matrix[CPU_UP], up_first - INNER_SIZE, share.unit_end - INNER_SIZE);
}

/* ---- the instances and a step ---- */

/* What the harness's instance of the same index runs. */
struct instance_kind { int kernel; struct kernel_form form; cpu_work_function *work, *twin; };
#define FORM_KIND(id, name, ...) {KERNEL_VNNI, {__VA_ARGS__}, work_##id, twin_##id},
#define FORM_INSTANCE(id, name, ...) {name, "work_" #id, 'R', 1, 0, 0, 0, 0, 0},
enum { INSTANCE_READ = 2, INSTANCE_VNNI = 3 };
static const struct instance_kind kinds[] = {{KERNEL_FP32, {0}, work_fp32, NULL}, {KERNEL_FP32, {0}, work_fp32, NULL},
                                             {KERNEL_READ, {0}, work_read, NULL}, VNNI_FORMS(FORM_KIND)};
static struct probe_instance instances[] = {     /* fp32's order of additions is no card's: F; vnni: what is shown */
    {"baseline", "work_fp32", 'F', 1, 0, 0, 0, 0, 0}, {"baseline_again", "work_fp32", 'F', 1, 0, 0, 0, 0, 0},
    {"read", "work_read", '-', 1, 0, 0, 0, 0, 0}, VNNI_FORMS(FORM_INSTANCE)};

/* One step of an instance on the case's team: the first region, the caller's loop unless the threads did it
 * (the activation, and for the integer kernel every block of the hidden vector quantized), the second region. */
static void run_step(const struct instance_kind *kind, cpu_work_function *work) {
    struct cpu_region_record *records = run.timeline ? run.records : NULL;
    int whole = cpu_team_run(&run.team, work, NULL, records);
    for (int slot = 0; kind->kernel != KERNEL_READ && !kind->form.inline_activation && slot < run.experts; slot++) {
        activate_rows(slot, 0, INNER_SIZE, kind->kernel == KERNEL_VNNI);
        for (int block = 0; kind->kernel == KERNEL_VNNI && block < HIDDEN_BLOCKS; block++) {
            quantize_block(run.hidden[slot] + block * GROUP_INPUTS, &run.hidden_operand[slot], block);
        }
    }
    whole &= cpu_team_run(&run.team, work, &run, records != NULL ? records + 1 : NULL);
    PROBE_REQUIRE(whole, "a team did not do a whole region: nothing of this run can be used");
}

/* ---- the references: the definitions in plain code ---- */

/* An E4M3 byte from its fields, in units of 2^-9: (8 + mantissa) 2^(exponent - 1), the mantissa alone where the
 * exponent is 0, negative with the sign bit. */
static int e4m3_units(int code) {
    const int exponent = (code >> E4M3_EXPONENT_SHIFT) & E4M3_EXPONENT_MASK, mantissa = code & E4M3_MANTISSA_MASK;
    const int units = exponent == 0 ? mantissa : (E4M3_IMPLICIT_ONE + mantissa) << (exponent - 1);
    return (code & E4M3_SIGN_BIT) != 0 ? -units : units;
}
/* linear_records.h, "Quantization", from its words: codes in the order of the values, an E8M0 code a block. */
static void reference_quantize(const float *values, int size, uint8_t *codes, uint8_t *exponents) {
    for (int block = 0; block < size / GROUP_INPUTS; block++) {
        const float *block_values = values + block * GROUP_INPUTS;
        float largest = AMAX_FLOOR;
        int binary_exponent = 0;
        for (int offset = 0; offset < GROUP_INPUTS; offset++) { largest = fmaxf(largest, fabsf(block_values[offset])); }
        const int is_power = frexpf(largest * E4M3_MAXIMUM_INVERSE, &binary_exponent) == 1.0f / PAIR;
        const int scale_exponent = binary_exponent - is_power;           /* the least power of two not under it */
        exponents[block] = (uint8_t)(scale_exponent + CPU_E8M0_BIAS);
        for (int offset = 0; offset < GROUP_INPUTS; offset++) {
            const double scaled = ldexp((double)block_values[offset], -scale_exponent);
            const double magnitude = fmin(fabs(scaled), (double)E4M3_MAXIMUM);
            int nearest = 0;
            for (int code = 1; code < E4M3_NAN_MAGNITUDE; code++) {
                const double distance = fabs((double)e4m3_units(code) / E4M3_UNITS_PER_ONE - magnitude);
                const double least = fabs((double)e4m3_units(nearest) / E4M3_UNITS_PER_ONE - magnitude);
                nearest = distance < least || (distance == least && code % PAIR == 0) ? code : nearest;
            }
            codes[block * GROUP_INPUTS + offset] = (uint8_t)(nearest | (signbit(scaled) ? E4M3_SIGN_BIT : 0));
        }
    }
}
/* One matrix of the ring against a quantized vector by the integer contract: per output and group the sum in
 * whole numbers, then the cards' chain in fp32. sums, where not null, gets them as [strip][group][row of strip]. */
static void reference_products(const struct cpu_matrix *matrix, const uint8_t *codes, const uint8_t *exponents,
                               int32_t *sums, float *outputs) {
    const int groups = matrix->input_size / GROUP_INPUTS, groups_per_slice = groups / INPUT_SLICES;
    #pragma omp parallel for if(CPU_OPENMP_HELPS) schedule(static)
    for (int row = 0; row < matrix->output_size; row++) {
        const uint8_t *nibbles = matrix->nibbles + (size_t)row * cpu_row_nibble_bytes(matrix);
        const uint8_t *scale_codes = matrix->scale_codes + (size_t)row * cpu_row_scale_bytes(matrix);
        float total = 0.0f;
        for (int slice = 0; slice < INPUT_SLICES; slice++) {
            float chain = 0.0f;
            for (int group = slice * groups_per_slice; group < (slice + 1) * groups_per_slice; group++) {
                int sum = 0;
                for (int column = group * GROUP_INPUTS; column < (group + 1) * GROUP_INPUTS; column++) {
                    const int shift = column % CPU_NIBBLES_PER_BYTE * CPU_NIBBLE_BITS;
                    sum += twice_weight_of_code[(nibbles[column / CPU_NIBBLES_PER_BYTE] >> shift) & CPU_NIBBLE_MASK] *
                           e4m3_units(codes[column]);
                }
                if (sums != NULL) {
                    sums[(row / ROWS_PER_STRIP * groups + group) * ROWS_PER_STRIP + row % ROWS_PER_STRIP] = sum;
                }
                chain = fmaf((float)sum, ldexpf(1.0f, scale_codes[group] + exponents[group] - PRODUCT_SCALE_BIAS),
                             chain);
            }
            total += chain;
        }
        outputs[row] = total;
    }
}
static void double_products(const struct cpu_matrix *matrix, const double *input, double *outputs) {
    #pragma omp parallel for if(CPU_OPENMP_HELPS) schedule(static)
    for (int row = 0; row < matrix->output_size; row++) {
        outputs[row] = cpu_reference_row(&run.ring, matrix, row, input);
    }
}
/* The rms error of fp32 values over the rms of their reference; a not-a-number where a value is one. */
static double error_over_rms(const float *values, const double *reference, int count) {
    double error_squares = 0.0, squares = 0.0;
    for (int index = 0; index < count; index++) {
        error_squares += ((double)values[index] - reference[index]) * ((double)values[index] - reference[index]);
        squares += reference[index] * reference[index];
    }
    return sqrt(error_squares / squares);
}
/* The references of the checked expert of a slot. b04's contract in double precision, its hidden vector rounded to
 * fp32 where the kernel hands it on. The integer contract in whole numbers and fp32, the group sums of the first
 * slot kept; then its output once more in double precision from its own quantized hidden vector. */
static void take_reference(int slot) {
    static double input[HIDDEN_SIZE], gate[INNER_SIZE], up[INNER_SIZE], hidden[INNER_SIZE];
    static float gate_chain[INNER_SIZE], up_chain[INNER_SIZE];
    static uint8_t codes[HIDDEN_SIZE], exponents[GATE_GROUPS];
    const struct cpu_matrix *matrix = run.ring.experts[FIRST_CHECKED_EXPERT + slot].matrix;
    const double limit = (double)SWIGLU_LIMIT;
    struct reference *reference = &references[slot];
    for (int column = 0; column < HIDDEN_SIZE; column++) { input[column] = (double)run.input[column]; }
    double_products(&matrix[CPU_GATE], input, gate);
    double_products(&matrix[CPU_UP], input, up);
    for (int row = 0; row < INNER_SIZE; row++) {
        const double limited_gate = gate[row] < limit ? gate[row] : limit;
        const double limited_up = up[row] < -limit ? -limit : (up[row] > limit ? limit : up[row]);
        hidden[row] = (double)(float)(limited_gate / (1.0 + exp(-limited_gate)) * limited_up * (double)ROUTE_WEIGHT);
    }
    double_products(&matrix[CPU_DOWN], hidden, reference->fp32_output);
    reference_quantize(run.input, HIDDEN_SIZE, codes, exponents);
    reference_products(&matrix[CPU_GATE], codes, exponents, slot == 0 ? reference_sums[CPU_GATE] : NULL, gate_chain);
    reference_products(&matrix[CPU_UP], codes, exponents, slot == 0 ? reference_sums[CPU_UP] : NULL, up_chain);
    for (int row = 0; row < INNER_SIZE; row++) {
        reference->hidden[row] = bf16_value(hidden_value(bf16_value(gate_chain[row]), bf16_value(up_chain[row])));
    }
    reference_quantize(reference->hidden, INNER_SIZE, codes, exponents);
    reference_products(&matrix[CPU_DOWN], codes, exponents, slot == 0 ? reference_sums[CPU_DOWN] : NULL,
                       reference->output);
    for (int row = 0; row < INNER_SIZE; row++) {
        hidden[row] = ldexp((double)e4m3_units(codes[row]),
                            exponents[row / GROUP_INPUTS] - CPU_E8M0_BIAS - E4M3_UNIT_EXPONENT);
    }
    double_products(&matrix[CPU_DOWN], hidden, reference->output_double);
}
/* One matrix of the ring as strips of 16 rows, on 2 MiB pages as the ring's own: per strip and group of 32 inputs
 * 256 bytes of codes (step s at 32 s; byte 2 r + b holds row r's inputs 4 s + 2 b, the low nibble, and 4 s + 2 b
 * + 1), then the 16 scales of the rows: one stream a thread. Returns NULL without memory or the page advice. */
static uint8_t *pack_matrix(const struct cpu_matrix *matrix) {
    const size_t groups = cpu_row_scale_bytes(matrix), group_bytes = GROUP_BYTES;
    const size_t bytes = (size_t)(matrix->output_size / ROWS_PER_STRIP) * groups * group_bytes;
    const size_t room = (bytes + CPU_HUGE_PAGE_BYTES - 1) / CPU_HUGE_PAGE_BYTES * CPU_HUGE_PAGE_BYTES;
    uint8_t *packed = aligned_alloc(CPU_HUGE_PAGE_BYTES, room);
    if (packed == NULL || madvise(packed, room, MADV_HUGEPAGE) != 0) { return NULL; }
    memset(packed, 0, room);
    for (int row = 0; row < matrix->output_size; row++) {
        const uint8_t *nibbles = matrix->nibbles + (size_t)row * cpu_row_nibble_bytes(matrix);
        for (int column = 0; column < matrix->input_size; column++) {    /* a nibble keeps its half of the byte */
            const int offset = column % GROUP_INPUTS, strip_row = row % ROWS_PER_STRIP;
            const int byte = offset / INPUTS_PER_STEP * STEP_BYTES + strip_row * PAIR + offset % INPUTS_PER_STEP / PAIR;
            const size_t group = (size_t)(column / GROUP_INPUTS);
            const uint8_t half_mask = (uint8_t)(CPU_NIBBLE_MASK << (column % CPU_NIBBLES_PER_BYTE * CPU_NIBBLE_BITS));
            uint8_t *group_at = packed + ((size_t)(row / ROWS_PER_STRIP) * groups + group) * group_bytes;
            group_at[byte] = (uint8_t)(group_at[byte] | (nibbles[column / CPU_NIBBLES_PER_BYTE] & half_mask));
            group_at[CODE_BYTES + strip_row] = matrix->scale_codes[(size_t)row * groups + group];
        }
    }
    return packed;
}

/* ---- the checks ---- */

static uint64_t differing_words(const void *first, const void *second, size_t count) {     /* of 32 bits each */
    uint64_t differing = 0;
    for (size_t at = 0; at < count * sizeof(int32_t); at += sizeof(int32_t)) {
        differing += memcmp((const char *)first + at, (const char *)second + at, sizeof(int32_t)) != 0;
    }
    return differing;
}
/* The checks of one instance on the case's team: a step on the checked experts. What the step is to write holds,
 * before it, bytes no kernel produces (a not-a-number), so that a share that does not run is seen. */
static void check_instance(const char *case_name, uint32_t index) {
    const struct instance_kind *kind = &kinds[index];
    const char *variant = instances[index].name;
    const int is_vnni = kind->kernel == KERNEL_VNNI, is_read = kind->kernel == KERNEL_READ;
    const int must_report = broken("corrupt") || (is_vnni && broken("plane"));
    const uint64_t all_sums = (uint64_t)CPU_MATRIX_KINDS * MATRIX_GROUP_SUMS;
    double worst = 0.0;
    uint64_t differing = 0, sums[MAX_EXPERTS] = {0};
    for (int slot = 0; slot < run.experts; slot++) { run.slot_experts[0][slot] = FIRST_CHECKED_EXPERT + slot; }
    run.step_experts = run.slot_experts[0];
    for (int pass = is_vnni ? 0 : 1; pass < PAIR; pass++) {              /* the twin first, then as it is timed */
        memset(run.gate_up, STALE_BYTE, sizeof(run.gate_up));
        memset(run.hidden, STALE_BYTE, sizeof(run.hidden));
        memset(run.output, STALE_BYTE, sizeof(run.output));
        memset(twin_sums, STALE_BYTE, sizeof(twin_sums));
        memset(run.lines, 0, sizeof(run.lines));
        run_step(kind, pass == 0 ? kind->twin : kind->work);
        if (pass == 0) {
            probe_check_count(case_name, variant, "integer_sums", differing_words(twin_sums, reference_sums, all_sums),
                              all_sums, must_report);
        }
    }
    for (int thread = 0; thread < run.threads; thread++) {
        sums[run.shares[UNIT_ROWS_DOWN][thread].expert_slot] +=
            run.lines[0][thread].byte_sum + run.lines[1][thread].byte_sum;
    }
    for (int slot = 0; slot < run.experts; slot++) {
        const struct reference *reference = &references[slot];
        const double *in_double = is_vnni ? reference->output_double : reference->fp32_output;
        const double error = error_over_rms(run.output[slot], in_double, HIDDEN_SIZE);
        uint64_t expected_sum = 0;
        (void)cpu_expert_bytes(&run.ring.experts[FIRST_CHECKED_EXPERT + slot], &expected_sum);
        worst = error > worst || isnan(error) ? error : worst;
        differing += is_read ? sums[slot] != expected_sum
                             : differing_words(run.hidden[slot], reference->hidden, INNER_SIZE) +
                                   differing_words(run.output[slot], reference->output, HIDDEN_SIZE);
    }
    if (!is_read) { probe_check_error(case_name, variant, "host_reference", worst, FP32_RMS_ERROR_LIMIT, must_report); }
    if (is_read || is_vnni) {
        probe_check_count(case_name, variant, is_read ? "byte_sum" : "same_bits", differing,
                          (uint64_t)run.experts * (is_read ? 1 : INNER_SIZE + HIDDEN_SIZE), must_report);
    }
}
static void check_case(void *context, const struct probe_case *probe_case) {
    uint32_t selected[PROBE_MOST_INSTANCES];
    const uint32_t selected_count = probe_select_instances(selected, 0);
    (void)context;
    probe_check_count(probe_case->name, "all", "shares_inside", (uint64_t)run.refused_shares,
                      (uint64_t)(UNIT_KINDS * run.threads), broken("share"));
    if (run.refused_shares != 0) { return; }     /* no step runs on a share that was refused */
    for (uint32_t index = 0; index < selected_count; index++) { check_instance(probe_case->name, selected[index]); }
    probe_check_count(probe_case->name, "all", "pinning", (uint64_t)cpu_team_misplaced(&run.team, run.announced_cpus),
                      (uint64_t)run.threads, broken("pinning"));
}

/* ---- the harness's callbacks ---- */

/* A case: its team (the one before it is ended), every thread's units of every kind of unit, each validated. */
static void describe_case(void *context, const struct probe_case *probe_case, struct probe_case_facts *facts) {
    static int order[CPU_MAX_HARDWARE_THREADS];
    const int threads = (int)probe_case_value(probe_case, "--threads"), shifted = broken("pinning");
    const int experts = (int)probe_case_value(probe_case, "--experts");
    const enum cpu_pool_kind pool = (enum cpu_pool_kind)probe_case_value(probe_case, "--pool");
    (void)context;
    PROBE_REQUIRE(run.team.thread_count == 0 || cpu_team_close(&run.team), "a team could not be ended");
    PROBE_REQUIRE(experts <= threads && threads + shifted <= run.topology.thread_count,
                  "more experts than threads, or more threads than processors");
    cpu_placement_order(&run.topology, (enum cpu_placement)probe_case_value(probe_case, "--placement"), order);
    for (int thread = 0; thread < threads; thread++) {
        run.announced_cpus[thread] = order[thread];
        run.pinned_cpus[thread] = order[thread + shifted];
    }
    run.experts = experts;
    run.threads = threads;
    run.refused_shares = 0;
    memset(run.terms, 0, sizeof(run.terms));
    for (int index = 0; index < UNIT_KINDS * experts; index++) {         /* the threads of slot s, shared evenly */
        const int kind = index / experts, slot = index % experts, first_thread = slot * threads / experts;
        const int workers = (slot + 1) * threads / experts - first_thread;
        for (int worker = 0; worker < workers; worker++) {
            struct share *share = &run.shares[kind][first_thread + worker];
            int inside = 1;
            share->expert_slot = slot;
            cpu_static_share(units_of_kind[kind], workers, worker, &share->first_unit, &share->unit_end);
            share->unit_end += kind == UNIT_STRIPS_DOWN && first_thread + worker + 1 == threads && broken("share");
            probe_launch_value(&inside, "share", "the end of a thread's units", (uint64_t)share->unit_end,
                               (uint64_t)share->first_unit, (uint64_t)units_of_kind[kind]);
            run.refused_shares += !inside;
        }
    }
    PROBE_REQUIRE(cpu_team_open(&run.team, pool, threads, run.pinned_cpus, run.ticks_per_microsecond),
                  "a team could not be started or pinned (is every listed processor this process's to use?)");
    facts->format = "native";
    facts->part = probe_case->name;
    facts->rows = 1;
    facts->experts = experts;
    facts->megabytes = (double)cpu_expert_bytes(&run.ring.experts[0], NULL) * experts / CPU_BYTES_PER_MB;
}
/* The experts of a step: the next of the ring, so that a position returns only after every other was read. */
static int prepare_step(void *context, const struct probe_case *probe_case, uint32_t instance, uint64_t step,
                        uint32_t slot) {
    const uint64_t ring_experts = (uint64_t)run.ring.expert_count, experts = (uint64_t)run.experts;
    (void)context, (void)probe_case, (void)instance;
    PROBE_REQUIRE(slot < PROBE_STREAM_STEPS, "a stream slot beyond the harness's streams");
    for (uint64_t expert = 0; expert < experts; expert++) {
        run.slot_experts[slot][expert] = (int)((step * experts + expert) * RING_STRIDE % ring_experts);
    }
    return run.refused_shares == 0;
}
static uint32_t launch_step(void *context, const struct probe_case *probe_case, uint32_t instance, uint32_t slot) {
    (void)context, (void)probe_case;
    run.step_experts = run.slot_experts[slot];
    run.step_instance = instance;
    run.steps_since_mark++;
    run_step(&kinds[instance], kinds[instance].work);
    return REGIONS_PER_STEP;
}
static void mark_start(void *context) {
    (void)context;
    run.steps_since_mark = 0;
    for (int counter = 0; run.counting && counter < COUNTERS; counter++) {
        (void)ioctl(run.counters[counter], PERF_EVENT_IOC_ENABLE, 0);
    }
    run.started_ticks = __rdtsc();
}
/* The end of the timed region; then, outside it, the counters are stopped and a single step's stamps become 8a's
 * terms: wake, work and tail follow the thread that left its region last, serial is the caller between regions. */
static double elapsed_microseconds(void *context) {
    const uint64_t ended_ticks = __rdtsc();
    const double ticks = run.ticks_per_microsecond;
    double *terms = run.terms[run.step_instance];
    (void)context;
    for (int counter = 0; run.counting && counter < COUNTERS; counter++) {
        (void)ioctl(run.counters[counter], PERF_EVENT_IOC_DISABLE, 0);
    }
    run.counted_steps += run.steps_since_mark;
    for (int region = 0; run.timeline && run.steps_since_mark == 1 && region < REGIONS_PER_STEP; region++) {
        const struct cpu_region_record *record = &run.records[region];
        const struct cpu_thread_stamps *last = &record->threads[0];
        for (int thread = 1; thread < run.threads; thread++) {
            last = record->threads[thread].left_ticks > last->left_ticks ? &record->threads[thread] : last;
        }
        terms[TERM_WAKE] += (double)(int64_t)(last->entered_ticks - record->forked_ticks) / ticks;
        terms[TERM_WORK] += (double)(int64_t)(last->left_ticks - last->entered_ticks) / ticks;
        terms[TERM_TAIL] += (double)(int64_t)(record->joined_ticks - last->left_ticks) / ticks;
        terms[TERM_SERIAL] += (double)(int64_t)(record->forked_ticks - run.records[0].joined_ticks) * region / ticks;
        terms[TERM_STEPS] += region;
    }
    return (double)(ended_ticks - run.started_ticks) / ticks;
}
/* After a case's results: its terms, the bar of ROUND_1.md where the case is one thread, and, after the last of
 * a list of thread counts, per instance the fewest threads that are within 5% of the most threads' median. */
static void verdicts(void *context, const struct probe_case *probe_case, const struct probe_result *results,
                     double noise_ratio) {
    const uint32_t thread_cases = probe_flag("--threads")->value_count, place = probe_case->index % thread_cases;
    const uint32_t compared = place + 1 == thread_cases && thread_cases > 1 ? PROBE_COUNT_OF(instances) : 0;
    const struct probe_result *vnni = &results[INSTANCE_VNNI];
    const char *standing = PROBE_DIAGNOSTICS || probe_flag_given(PROBE_FLAG_STEPS) ? "untimed" : "measured";
    uint32_t most = 0;
    (void)context;
    probe_check_count(probe_case->name, "all", "pinning_after", (uint64_t)cpu_team_misplaced(&run.team,
                      run.announced_cpus), (uint64_t)run.threads, 0);       /* an OpenMP runtime may bind again */
    run.case_threads[place] = run.threads;
    for (uint32_t index = 0; index < PROBE_COUNT_OF(instances); index++) {
        const double *terms = run.terms[index], steps = terms[TERM_STEPS];
        run.medians[place][index] = results[index].mode != NULL ? results[index].step.median : PROBE_NAN;
        if (steps > 0.0) {
            printf("TIMELINE card=host case=%s variant=%s steps=%.0f wake_us=%.2f work_us=%.2f tail_us=%.2f "
                   "serial_us=%.2f\n", probe_case->name, instances[index].name, steps, terms[TERM_WAKE] / steps,
                   terms[TERM_WORK] / steps, terms[TERM_TAIL] / steps, terms[TERM_SERIAL] / steps);
        }
    }
    if (run.threads == 1 && vnni->mode != NULL) {
        probe_verdict("vnni_one_thread", VNNI_BAR_RATIO, VNNI_BAR_RATIO * results[PROBE_BASELINE].step.median,
                      vnni->ratio, noise_ratio, vnni->same_sign, vnni->rounds);
    }
    for (uint32_t other = 1; other < thread_cases; other++) {
        most = run.case_threads[other] > run.case_threads[most] ? other : most;
    }
    for (uint32_t index = 0; index < compared; index++) {
        uint32_t fewest = most;
        for (uint32_t other = 0; other < thread_cases; other++) {
            fewest = run.medians[other][index] <= FEWEST_THREADS_RATIO * run.medians[most][index] &&
                     run.case_threads[other] < run.case_threads[fewest] ? other : fewest;
        }
        if (!isnan(run.medians[most][index])) {
            printf("VERDICT card=host figure=fewest_threads case=%s variant=%s threads=%d us=%s fewest_threads=%d "
                   "fewest_us=%s read_us_there=%s verdict=%s\n", probe_case->name, instances[index].name,
                   run.case_threads[most], probe_number(run.medians[most][index], 1), run.case_threads[fewest],
                   probe_number(run.medians[fewest][index], 1), probe_number(run.medians[fewest][INSTANCE_READ], 1),
                   standing);
        }
    }
}
/* The end of a run: the team, the profiled steps' counters, the debug build's count. */
static void close_run(void *context) {
    const char *profiled = probe_flag_text(PROBE_FLAG_PROFILE_CASE);
    const double weights = (double)run.counted_steps * run.experts * CPU_MATRIX_KINDS * HIDDEN_SIZE * INNER_SIZE;
    uint64_t counts[COUNTERS] = {0, 0};
    (void)context;
    PROBE_REQUIRE(run.team.thread_count == 0 || cpu_team_close(&run.team), "a team could not be ended");
    for (int counter = 0; run.counting && counter < COUNTERS; counter++) {
        run.counting = read(run.counters[counter], &counts[counter], sizeof(counts[0])) == (ssize_t)sizeof(counts[0]);
    }
    if (profiled[0] != '\0') {
        printf("PERF card=host variant=%s steps=%llu cycles_per_weight=%s instructions_per_weight=%s\n", profiled,
               (unsigned long long)run.counted_steps,
               probe_number(run.counting ? (double)counts[0] / weights : PROBE_NAN, PER_WEIGHT_DECIMALS),
               probe_number(run.counting ? (double)counts[1] / weights : PROBE_NAN, PER_WEIGHT_DECIMALS));
    }
#if PROBE_DIAGNOSTICS
    uint64_t wide_sums = 0;
    for (int thread = 0; thread < MAX_THREADS; thread++) {
        wide_sums += run.lines[0][thread].wide_sums + run.lines[1][thread].wide_sums;
    }
    printf("DIAG card=host group_sums_at_or_above_2^24=%llu\n", (unsigned long long)wide_sums);
#endif
}
/* Before the first case: the machine, the ring and its strips, the input as the kernel reads it, the references,
 * and then what a --break name changes behind them. Returns 0 after saying why. */
static int configure(void *context) {
    static const uint64_t events[COUNTERS] = {PERF_COUNT_HW_CPU_CYCLES, PERF_COUNT_HW_INSTRUCTIONS};
    const struct probe_flag *experts = probe_flag("--experts");
    const int ring_experts = (int)probe_flag_number("--ring-experts", 0);
    const int profiles = probe_flag_text(PROBE_FLAG_PROFILE_CASE)[0] != '\0';
    int most_experts = 0;
    (void)context;
    if (probe_flag_given("--list-instances")) { return 1; }
    if (ring_experts % RING_STRIDE == 0 || !cpu_topology_read(&run.topology)) {
        fprintf(stderr, "refused: --ring-experts is a multiple of %d, or sysfs lacks the topology\n", RING_STRIDE);
        return 0;
    }
    probe_run.breaking |= (uint32_t)probe_flag_given("--break");         /* the harness knows switches only */
    run.timeline = probe_flag_given("--timeline");
    run.records[0].threads = run.stamps[0], run.records[1].threads = run.stamps[1];
    omp_set_dynamic(0);                              /* a team has the threads it was asked for, or the run ends */
    for (int counter = 0; profiles && counter < COUNTERS; counter++) {   /* before any thread: later ones count too */
        struct perf_event_attr attributes = {.type = PERF_TYPE_HARDWARE, .size = sizeof(attributes),
                                             .config = events[counter], .disabled = 1, .inherit = 1,
                                             .exclude_kernel = 1, .exclude_hv = 1};
        run.counters[counter] = (int)syscall(SYS_perf_event_open, &attributes, 0, -1, -1, 0ul);
        run.counting = run.counters[counter] >= 0 && (counter == 0 || run.counting);
    }
    run.ticks_per_microsecond = cpu_ticks_per_microsecond();
    PROBE_REQUIRE(cpu_ring_create(&run.ring, ring_experts), "no memory for the ring, or no advice for its pages");
    run.strips = calloc((size_t)ring_experts, sizeof(*run.strips));
    PROBE_REQUIRE(run.strips != NULL, "no memory for the table of strips");
    #pragma omp parallel for if(CPU_OPENMP_HELPS) schedule(dynamic)
    for (int index = 0; index < ring_experts * CPU_MATRIX_KINDS; index++) {
        uint8_t **packed = &run.strips[index / CPU_MATRIX_KINDS][index % CPU_MATRIX_KINDS];
        *packed = pack_matrix(&run.ring.experts[index / CPU_MATRIX_KINDS].matrix[index % CPU_MATRIX_KINDS]);
        PROBE_REQUIRE(*packed != NULL, "no memory for the strips of the ring, or no advice for their pages");
    }
    for (int column = 0; column < HIDDEN_SIZE; column++) {
        const int block = column / GROUP_INPUTS, offset = column % GROUP_INPUTS;
        const int kind = (int)(cpu_mix_bits(~((uint64_t)block + INPUT_SEED)) % INPUT_BLOCK_KINDS);
        const float plain = (float)(cpu_mix_bits((uint64_t)column + INPUT_SEED) % INPUT_VALUE_STEPS);
        const float value = kind < INPUT_PLAIN_KINDS ? ldexpf((plain - INPUT_VALUE_OFFSET) / INPUT_VALUE_DIVISOR, -kind)
                            : kind == INPUT_ZEROS ? 0.0f : kind == INPUT_TINY ? INPUT_TINY_VALUE * (float)offset
                            : kind != INPUT_TIES ? ldexpf(INPUT_SPREAD_VALUE, -(offset % INPUT_SPREAD_BITS))
                            : ldexpf(E4M3_MAXIMUM - INPUT_TIE_GAP * (float)offset, -(INPUT_TIE_SHIFT + block % PAIR));
        run.input[column] = kind >= INPUT_PLAIN_KINDS && offset % PAIR != 0 ? -value : value;
    }
    for (int block = 0; block < GATE_GROUPS; block++) {
        quantize_block(run.input + block * GROUP_INPUTS, &run.input_operand, block);
    }
    for (uint32_t index = 0; index < experts->value_count; index++) {
        most_experts = (int)experts->values[index] > most_experts ? (int)experts->values[index] : most_experts;
    }
    for (int slot = 0; slot < most_experts && !profiles; slot++) { take_reference(slot); }
    PROBE_REQUIRE(cpu_openmp_release_threads(), "the OpenMP runtime would not end its threads");
    for (int column = 0; broken("plane") && column < HIDDEN_SIZE; column++) {    /* the third digit's sign rule */
        run.input_operand.digits[LAST_PLANE][column] = (int8_t)-run.input_operand.digits[LAST_PLANE][column];
    }
    if (broken("corrupt")) {                         /* after the references of this expert were taken */
        memset(run.ring.experts[FIRST_CHECKED_EXPERT].matrix[CPU_GATE].nibbles, CORRUPTED_BYTE_VALUE, CORRUPTED_BYTES);
        memset(run.strips[FIRST_CHECKED_EXPERT][CPU_GATE], CORRUPTED_BYTE_VALUE, CORRUPTED_BYTES);
    }
    const double ring_mib = (double)cpu_expert_bytes(&run.ring.experts[0], NULL) * ring_experts / CPU_BYTES_PER_MIB;
    const double cache_mib = (double)run.topology.slice_kib * run.topology.slice_count / CPU_KIB_PER_MIB;
    printf("HOST card=host build=%s cpus=%d cores=%d l3_slices=%d l3_mib=%.0f ring_experts=%d ring_mib=%.0f "
           "tsc_mhz=%.1f\n", PROBE_BUILD, run.topology.thread_count, run.topology.core_count,
           run.topology.slice_count, cache_mib, ring_experts, ring_mib, run.ticks_per_microsecond);
    if (ring_mib < cache_mib * CACHES_OF_RING_MIN && !PROBE_DIAGNOSTICS && !probe_breaking()) {
        fprintf(stderr, "refused: the ring is under four times the third-level cache: its experts would not be cold\n");
        return 0;
    }
    return 1;
}

static struct probe_flag flags[] = {
    PROBE_COMMON_FLAGS,
    PROBE_CASE_NUMBERS("--experts", 1, MAX_EXPERTS, "1", "cold experts at once, the threads shared evenly"),
    PROBE_CASE_WORDS("--pool", "openmp,own", "own", "the team: OpenMP's, or the spinning pool of cpu_probe_common.h"),
    PROBE_CASE_WORDS("--placement", "die,packed,smt", "die", "the order in which a team takes the processors"),
    PROBE_CASE_NUMBERS("--threads", 1, MAX_THREADS, "1", "threads of the team"),
    PROBE_SWITCH("--timeline", "two stamps per thread and region: TIMELINE lines"),
    PROBE_NUMBER("--ring-experts", RING_EXPERTS_MIN, RING_EXPERTS_MAX, "128", "experts of the cold ring"),
    PROBE_WORD_LIST("--break", "corrupt,plane,share,pinning", "", "show a check failing; nothing is timed")};
static struct probe_program program = {
    "cpu_expert_forms", flags, PROBE_COUNT_OF(flags), instances, PROBE_COUNT_OF(instances), NULL, configure, NULL,
    describe_case, check_case, prepare_step, launch_step, mark_start, elapsed_microseconds, verdicts, close_run, NULL};

int main(int argument_count, char **arguments) { return probe_main(argument_count, arguments, &program); }
