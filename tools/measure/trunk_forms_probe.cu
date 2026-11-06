/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * trunk_forms_probe: package 4 of round 1 of the speed campaign (research/speed/ROUND_1.md, "trunk_forms"; ledger
 * rows TR1, TR2, TR3, TR5, M15, S13): the 8-bit trunk with the reference's quantized inputs, the loads its narrow
 * matrices wait for, and a first figure for the head, the router, a mixing projection and two passes over a vector.
 * Weights and inputs are synthetic; trunk_forms_kernels.cuh holds the kernels and says what each form is.
 *
 * Cases: --matrix by --slices by --rows. A matrix word is one step: the quantization of the rows and the product
 * (wq_a, wq_b, wkv, wo_b, down, engram_wkv, head, router, mixing), two products that share their input (narrow:
 * wq_a and wkv; gate_up), or the quantization alone (pass_5120, pass_129280: a launch that reads every value,
 * reduces per 32 and writes a byte a value stands in for a norm and for the sampler's pass over the logits; timed
 * with --instance of the instances that quantize, any other is refused there; inputs:placed16 and
 * inputs:carried_bf16 store three bytes a value where their products read two: the codes are the host check's).
 * Every matrix of a run has a ring of copies of four times the card's L2, in E4M3 and in bfloat16; of one L2 size
 * after --steps without --profile-case (a debug or sanitizer run: nothing of it is timed or profiled).
 * Instances, interleaved in one run (probe_harness.h; the table TRUNK_INSTANCES): baseline is the engine's form
 * (codes decoded in the loop, the weights one load ahead); inputs:fp16 the bridge to eight_bit_trunk_probe;
 * ring_multiple:0 reads the same copy every step; touch_before:1 times its touch launches with its products;
 * instruction:imma_s8 is class F (stored bytes and rows as whole numbers; its input quantizer is the power-of-two
 * sibling of unit 5c's rule and no format of the model stores such weights with these scales: a timing of the
 * form); weights:bf16 (bfloat16 weights and rows) is the figure of head, router and mixing. An instance whose
 * registers leave a block fewer threads than --slices asks is out of that case (a REFUSED line; n=0).
 *
 * Checked per case and instance before anything is timed: quantizer (every code, exponent and operand word the
 * card wrote, against the definition); host_reference_<m> (256 outputs of the first and of the last row against
 * double-precision sums rounded once, the weights read by the formats' layouts, fed with what the card holds:
 * 2e-3 of the rms); integer_contract_<m> (instruction:imma_s8: the same outputs bit for bit from whole-number
 * sums); bits_against_baseline (required of the instances that only move loads or operands: 0 words may differ;
 * for every instance a BITS line has the count, the sampled outputs equal to the float64 sum rounded once, and the
 * class: E where 0 words differ, else C; F stays F); rows_exact (every row launched again alone: 0 words may
 * differ); guard_bytes; and the DIGEST of all outputs, which the batch compares between the two CMP 170HX.
 * Shown failing, each by a run that exits 0 only if its check reported and no other did: --corrupt (a megabyte of
 * the first matrix in both rings: host_reference, integer_contract); --break bits (that megabyte after the
 * baseline's launches: bits_against_baseline); --break exactness (an instance that rounds after every slice where
 * a launch has rows: rows_exact; --slices 2 or more, --rows 2 or more); --break quantization (the host's floor
 * doubled: quantizer); --break-route (a step past a range, one past a count: each refused); --break-guard.
 *
 *   trunk_forms_probe [--matrix wkv,narrow] [--slices 1,8] [--rows 1,8] [--break NAME] [harness flags]
 *
 * RESULT: part=<matrix>,slices:<S>; mb= is the case's E4M3 bytes (weights:bf16 reads twice that). Its own lines:
 * BITS; at one row JUDGED card= case= slices= variant= median_us= fp16_us= (the instance the case is judged by:
 * the fastest of class E that neither spills nor reads a warm ring; of head, router and mixing weights:bf16) and
 * VERDICT judged_over_fp16_at_<case> or first_figure_at_<case>, measured: the package's bars are of sums over
 * fp16 at its own best slice count, which one case does not see: the batch prints them from the JUDGED lines.
 * Measurement tool: not part of the engine.
 */
#include "probe_harness.h"
#include "trunk_forms_kernels.cuh"

enum {
    TRUNK_STORE_E4M3 = TRUNK_FILL_E4M3, TRUNK_STORE_BF16 = TRUNK_FILL_BF16, TRUNK_STORES = 2,
    TRUNK_MOST_WIDTH = 129280, TRUNK_MOST_OUTPUTS = 129280,
    TRUNK_ROW_VALUES = TRUNK_MOST_ROWS * TRUNK_MOST_WIDTH,       /* of a case's rows in an array of [row][width] */
    TRUNK_SLOT_VALUES = TRUNK_MOST_ROWS * TRUNK_MOST_OUTPUTS,    /* of one matrix's outputs, [row][output_size] */
    TRUNK_ELEMENT_THREADS = 256, TRUNK_FILL_BLOCKS_PER_SM = 8, TRUNK_TOUCH_BLOCKS_PER_SM = 4,
    TRUNK_SAMPLES = 256, TRUNK_LEADING_SAMPLES = 16, TRUNK_LEADING_STRIDE = 3,   /* the first lie in the megabyte */
    TRUNK_CORRUPTED_BYTES = 1024 * 1024, TRUNK_CORRUPTED_BYTE = 0x5A, TRUNK_STALE_BYTE = 0xFF,
    TRUNK_MATRIX_ALIGNMENT_BYTES = 64, TRUNK_COLD_L2_SIZES = 4, TRUNK_E4M3_MAGNITUDES = 127,
    TRUNK_E4M3_LEAST_EXPONENT = 10,              /* an E4M3 number is its units times 2^(exponent field - 10) */
    /* trunk_input_value, the kinds of a block of 32: (k - 128) / 64; that times 2^-15 beside 448 2^-8; beside zeros;
     * times 2^-16; (17 + 2 m) 2^(e - 19) beside 448 2^-8: halfway between two codes at the scale 2^-8; 18 powers */
    TRUNK_INPUT_PLAIN = 0, TRUNK_INPUT_SUBNORMAL, TRUNK_INPUT_ZEROS, TRUNK_INPUT_UNDER_FLOOR, TRUNK_INPUT_TIES,
    TRUNK_INPUT_SPREAD, TRUNK_INPUT_KINDS, TRUNK_INPUT_STEPS = 256, TRUNK_INPUT_CENTRE = 128,
    TRUNK_SUBNORMAL_SHIFT = 15, TRUNK_UNDER_FLOOR_SHIFT = 16, TRUNK_LARGEST_SHIFT = 8, TRUNK_TIE_UNITS = 17,
    TRUNK_TIE_EXPONENT = 19, TRUNK_TIE_EXPONENTS = 14, TRUNK_SPREAD_PLACES = 18, TRUNK_FIGURE_BYTES = 256,
    TRUNK_BROKEN_COPY = 0, TRUNK_BROKEN_SLICES, TRUNK_BROKEN_COUNT,
    WQ_A = 0, WQ_B, WKV, WO_B, GATE, UP, DOWN, ENGRAM_WKV, HEAD, ROUTER, MIXING, PASS_5120, PASS_129280,
    TRUNK_MATRICES, NONE = TRUNK_MATRICES
};
#define TRUNK_NO_LAUNCH UINT32_MAX               /* the quantizer of a step that has no quantization */
#define TRUNK_FLAG_MATRIX "--matrix"
#define TRUNK_FLAG_SLICES "--slices"
#define TRUNK_FLAG_ROWS "--rows"
#define TRUNK_FLAG_BREAK "--break"
#define TRUNK_TOLERANCE 2.0e-3                   /* CONVENTIONS 5.1: a bfloat16 vector against its rms */
#define TRUNK_BYTES_PER_MB 1.0e6

struct trunk_shape { const char *name; uint32_t output_size, input_size; };
static const struct trunk_shape trunk_shapes[TRUNK_MATRICES] = {
    {"wq_a", 1280, 5120}, {"wq_b", 32768, 1280}, {"wkv", 512, 5120}, {"wo_b", 5120, 8192}, {"gate", 2304, 5120},
    {"up", 2304, 5120}, {"down", 5120, 2304}, {"engram_wkv", 25600, 6144}, {"head", 129280, 5120},
    {"router", 384, 5120}, {"mixing", 96, 20480}, {"pass_5120", 0, 5120}, {"pass_129280", 0, 129280}};
/* The matrices of a case, in the order of --matrix: the first, and one that shares its input. A matrix of no
 * output is a width alone: its step is the quantization of its rows. */
static const uint32_t trunk_cases[][TRUNK_MERGED_MATRICES] = {
    {WQ_A, NONE}, {WQ_B, NONE}, {WKV, NONE}, {WO_B, NONE}, {WQ_A, WKV}, {GATE, UP}, {DOWN, NONE},
    {ENGRAM_WKV, NONE}, {HEAD, NONE}, {ROUTER, NONE}, {MIXING, NONE}, {PASS_5120, NONE}, {PASS_129280, NONE}};
#define TRUNK_CASE_WORDS "wq_a,wq_b,wkv,wo_b,narrow,gate_up,down,engram_wkv,head,router,mixing,pass_5120,pass_129280"

static struct probe_flag trunk_flags[] = {
    PROBE_COMMON_FLAGS,
    PROBE_CASE_WORDS(TRUNK_FLAG_MATRIX, TRUNK_CASE_WORDS, "wq_a,wq_b,wkv,wo_b,narrow,gate_up,down,engram_wkv",
                     "the matrices of a step, or a pass over a width"),
    PROBE_CASE_NUMBERS(TRUNK_FLAG_SLICES, 1, TRUNK_MOST_SLICES, "8", "warps that share a tile's input width"),
    PROBE_CASE_NUMBERS(TRUNK_FLAG_ROWS, 1, TRUNK_MOST_ROWS, "1", "rows of a step"),
    PROBE_WORD_LIST(TRUNK_FLAG_BREAK, "bits,exactness,quantization", "", "one check shown failing"),
};

/* ---- the stamped kernels and the instances that launch them ---- */

typedef void (*trunk_launcher)(unsigned int tiles, unsigned int threads, const struct trunk_product_arguments *);
#define TRUNK_STAMP_LAUNCHER(id, form, ahead_weights, ahead_inputs, inputs_first, merged, row_dependent)              \
    static void trunk_launch_##id(unsigned int tiles, unsigned int threads,                                        \
                                  const struct trunk_product_arguments *arguments)                                 \
    {                                                                                                              \
        trunk_product_##id<<<tiles, threads, threads * TRUNK_SHARED_BYTES_PER_THREAD>>>(*arguments);               \
    }
TRUNK_PRODUCT_KERNELS(TRUNK_STAMP_LAUNCHER)
#define TRUNK_STAMP_ID(id, ...) TRUNK_KERNEL_##id,
enum { TRUNK_PRODUCT_KERNELS(TRUNK_STAMP_ID) TRUNK_KERNEL_COUNT };
struct trunk_kernel { trunk_launcher launch; const void *function; int form, merged; };
#define TRUNK_STAMP_ROW(id, form, ahead_weights, ahead_inputs, inputs_first, merged, row_dependent) \
    {trunk_launch_##id, (const void *)trunk_product_##id, form, merged},
static const struct trunk_kernel trunk_kernels[TRUNK_KERNEL_COUNT] = {TRUNK_PRODUCT_KERNELS(TRUNK_STAMP_ROW)};

/* id, variant, kernel, the class it claims, then 0 or 1 each: its bits must be the baseline's; sm_120a only; a
 * launch touches each matrix before its product; every step reads the same copy. The last: --break exactness. */
#define TRUNK_INSTANCES(ROW)                                                                                \
    ROW(baseline, "baseline", codes, 'E', 0, 0, 0, 0)                                                       \
    ROW(baseline_again, "baseline_again", codes, 'E', 1, 0, 0, 0)                                           \
    ROW(fp16, "inputs:fp16", fp16, 'F', 0, 0, 0, 0)                                                         \
    ROW(placed16, "inputs:placed16", placed16, 'E', 1, 0, 0, 0)                                             \
    ROW(carried_bf16, "inputs:carried_bf16", carried_bf16, 'E', 0, 0, 0, 0)                                 \
    ROW(weights_2, "ahead_weights:2", weights_2, 'E', 1, 0, 0, 0)                                           \
    ROW(weights_4, "ahead_weights:4", weights_4, 'E', 1, 0, 0, 0)                                           \
    ROW(inputs_1, "ahead_inputs:1", inputs_1, 'E', 1, 0, 0, 0)                                              \
    ROW(inputs_2, "ahead_inputs:2", inputs_2, 'E', 1, 0, 0, 0)                                              \
    ROW(inputs_first, "request_order:inputs_first", inputs_first, 'E', 1, 0, 0, 0)                          \
    ROW(weights_2_inputs_1, "ahead_weights:2,ahead_inputs:1", weights_2_inputs_1, 'E', 1, 0, 0, 0)          \
    ROW(weights_2_first, "ahead_weights:2,request_order:inputs_first", weights_2_first, 'E', 1, 0, 0, 0)    \
    ROW(inputs_1_first, "ahead_inputs:1,request_order:inputs_first", inputs_1_first, 'E', 1, 0, 0, 0)       \
    ROW(all_ahead, "ahead_weights:2,ahead_inputs:1,request_order:inputs_first", all_ahead, 'E', 1, 0, 0, 0) \
    ROW(merged, "merge_narrow:1", merged, 'E', 1, 0, 0, 0)                                                  \
    ROW(same_copy, "ring_multiple:0", codes, 'E', 1, 0, 0, 1)                                               \
    ROW(touched, "touch_before:1", codes, 'E', 1, 0, 1, 0)                                                  \
    ROW(fp8, "instruction:fp8", fp8, 'E', 0, 1, 0, 0)                                                       \
    ROW(fp8_scaled, "instruction:fp8_scaled", fp8_scaled, 'E', 0, 1, 0, 0)                                  \
    ROW(whole_s8, "instruction:imma_s8", whole_s8, 'F', 0, 0, 0, 0)                                         \
    ROW(bf16_weights, "weights:bf16", bf16_weights, 'C', 0, 0, 0, 0)                                        \
    ROW(row_dependent, "break:row_dependent", row_dependent, 'C', 0, 0, 0, 0)
#define TRUNK_INSTANCE_ID(id, ...) TRUNK_INSTANCE_##id,
enum { TRUNK_INSTANCES(TRUNK_INSTANCE_ID) TRUNK_INSTANCE_COUNT };
#define TRUNK_INSTANCE_ROW(id, variant, kernel, claimed, bits_required, sm_120a_only, touches, same_copy) \
    {variant, "trunk_product_" #kernel, claimed, !(sm_120a_only) || PROBE_SM == PROBE_SM_120A, 0, 0, 0, 0, 0},
static struct probe_instance trunk_instances[TRUNK_INSTANCE_COUNT] = {TRUNK_INSTANCES(TRUNK_INSTANCE_ROW)};
struct trunk_variant { uint32_t kernel; char claimed_class; int bits_required, touches, same_copy, compiled; };
#define TRUNK_VARIANT_ROW(id, variant, kernel, claimed, bits_required, sm_120a_only, touches, same_copy) \
    {TRUNK_KERNEL_##kernel, claimed, bits_required, touches, same_copy, !(sm_120a_only) || PROBE_SM == PROBE_SM_120A},
static const struct trunk_variant trunk_variants[TRUNK_INSTANCE_COUNT] = {TRUNK_INSTANCES(TRUNK_VARIANT_ROW)};
static uint32_t trunk_instances_in_run = TRUNK_INSTANCE_COUNT - 1u;      /* all but the one --break exactness adds */

struct trunk_step {                              /* the launches of one step, in the order they are made */
    uint32_t kernel, block_threads;              /* of its products */
    uint32_t touch_count, quantizer, product_count;      /* quantizer: a form, or TRUNK_NO_LAUNCH */
    struct { const uint32_t *words; uint64_t word_count; } touches[TRUNK_MERGED_MATRICES];
    struct trunk_quantize_arguments quantize;    /* its vectors: the rows of products[0] */
    struct trunk_product_arguments products[TRUNK_MERGED_MATRICES];
};
static struct {
    int sm_count;
    uint32_t threads_most[TRUNK_KERNEL_COUNT];   /* of a block of each product kernel on this card: its registers */
    uint8_t *stored[TRUNK_STORES];               /* the rings: every matrix's copies in E4M3, and in bfloat16 */
    uint64_t ring_at[TRUNK_STORES][TRUNK_MATRICES], copy_bytes[TRUNK_STORES][TRUNK_MATRICES];
    uint64_t copies[TRUNK_STORES][TRUNK_MATRICES];          /* 0: no case of the run reads the matrix */
    uint16_t *values, *values_fp16, *operands;   /* [row][width] of the case */
    uint8_t *codes, *exponents;                  /* [row][width], [row][width / 32] */
    uint16_t *outputs;                           /* [matrix of the case][row][output_size] */
    uint32_t *touched;                           /* a word a thread of a touch launch */
    struct trunk_step steps[PROBE_STREAM_STEPS]; /* of the step prepared in each slot */
} trunk;
static struct {                                  /* the host's copies of what a checked step left on the card */
    uint16_t outputs[TRUNK_MERGED_MATRICES * TRUNK_SLOT_VALUES], baseline[TRUNK_MERGED_MATRICES * TRUNK_SLOT_VALUES];
    uint16_t alone[TRUNK_MERGED_MATRICES * TRUNK_SLOT_VALUES];
    uint16_t values[TRUNK_ROW_VALUES], values_fp16[TRUNK_ROW_VALUES], operands[TRUNK_ROW_VALUES];
    uint8_t codes[TRUNK_ROW_VALUES], codes_by_value[TRUNK_ROW_VALUES];
    uint8_t exponents[TRUNK_ROW_VALUES / TRUNK_VALUES_PER_BLOCK];
    double fed[TRUNK_ROW_VALUES];                /* what the products are fed, as numbers: [row][width] */
} trunk_host;

/* ---- a step: built, validated, launched ---- */

static uint32_t trunk_seed(uint32_t matrix) { return trunk_mix_bits(matrix + 1u); }
static float trunk_float_of(uint32_t bits) { float value; memcpy(&value, &bits, sizeof(value)); return value; }
static uint32_t trunk_bits_of(float value) { uint32_t bits; memcpy(&bits, &value, sizeof(bits)); return bits; }
/* value / unit for a value that is whole units, else 0: a size that must divide is held to a range of units. */
static uint64_t trunk_units(uint64_t value, uint64_t unit) { return value % unit == 0 ? value / unit : 0; }

static uint32_t trunk_matrix_of(const struct probe_case *probe_case, uint32_t index)    /* NONE: it has no pair */
{
    return trunk_cases[probe_case_value(probe_case, TRUNK_FLAG_MATRIX)][index];
}

static uint64_t trunk_stored_bytes(int store, uint32_t matrix)       /* of one stored matrix, as a kernel reads it */
{
    const uint64_t weights = (uint64_t)trunk_shapes[matrix].output_size * trunk_shapes[matrix].input_size;
    return store == TRUNK_STORE_BF16 ? weights * TRUNK_HALF_BYTES
                                     : weights + weights / (TRUNK_OUTPUTS_PER_SCALE * TRUNK_VALUES_PER_BLOCK);
}

static uint8_t *trunk_stored(int store, uint32_t matrix, uint64_t ring_step)        /* the copy a step reads */
{
    const uint64_t copy = trunk.copies[store][matrix] == 0 ? 0 : ring_step % trunk.copies[store][matrix];
    return trunk.stored[store] + trunk.ring_at[store][matrix] + copy * trunk.copy_bytes[store][matrix];
}

/* The card array a form's products are fed from, and the bytes of one value in it. */
static const uint8_t *trunk_fed_array(int form, uint64_t *value_bytes)
{
    *value_bytes = form == TRUNK_FP16 || form == TRUNK_BF16_WEIGHTS || form == TRUNK_PLACED16 ||
                   form == TRUNK_CARRIED_BF16 ? TRUNK_HALF_BYTES : 1;
    return form == TRUNK_FP16 ? (const uint8_t *)trunk.values_fp16
           : form == TRUNK_BF16_WEIGHTS ? (const uint8_t *)trunk.values
           : *value_bytes == 1 ? trunk.codes : (const uint8_t *)trunk.operands;
}

/* The quantization a form's step begins with: a form of the quantizer, or TRUNK_NO_LAUNCH. */
static uint32_t trunk_quantizer_of(int form)
{
    return form == TRUNK_FP16 || form == TRUNK_BF16_WEIGHTS ? TRUNK_NO_LAUNCH
           : form == TRUNK_FP8 || form == TRUNK_FP8_SCALED ? (uint32_t)TRUNK_CODES : (uint32_t)form;
}

/* byte_count bytes from `pointer` on must lie inside the card array that begins at `array`. */
static void trunk_range(int *valid, const char *kernel, const char *what, const void *array, const void *pointer,
                        uint64_t byte_count)
{
    probe_launch_range(valid, kernel, what, (uint64_t)((uintptr_t)pointer - (uintptr_t)array), byte_count, 1,
                       probe_cuda_array_bytes(array));
}

/* Check 10: 1 if every launch of the step reads and writes inside its arrays and inside what its kernel loops by;
 * else it has printed why not. Grids are derived from the same fields when the step is launched. Every count is
 * formed in 64 bits from values already held to their ranges. */
static int trunk_step_is_valid(const struct trunk_step *step)
{
    const struct trunk_kernel *kernel = &trunk_kernels[step->kernel];
    const int form = kernel->form, store = form == TRUNK_BF16_WEIGHTS ? TRUNK_STORE_BF16 : TRUNK_STORE_E4M3;
    const uint64_t touch_blocks = (uint64_t)trunk.sm_count * TRUNK_TOUCH_BLOCKS_PER_SM;
    uint64_t value_bytes = 0;
    const uint8_t *fed_array = trunk_fed_array(form, &value_bytes);
    int valid = 1;
    for (uint32_t index = 0; index < step->touch_count; index++) {
        probe_launch_geometry(&valid, "trunk_touch_kernel", NULL, touch_blocks, 1, 1, TRUNK_ELEMENT_THREADS, 0);
        trunk_range(&valid, "trunk_touch_kernel", "a stored matrix", trunk.stored[store], step->touches[index].words,
                    step->touches[index].word_count * sizeof(uint32_t));
        trunk_range(&valid, "trunk_touch_kernel", "a word a thread", trunk.touched, trunk.touched,
                    touch_blocks * TRUNK_ELEMENT_THREADS * sizeof(uint32_t));
    }
    if (step->quantizer != TRUNK_NO_LAUNCH) {
        const uint64_t width = step->quantize.width, rows = step->products[0].row_count;
        const char *name = "trunk_quantize";
        probe_launch_value(&valid, name, "loads of 64 values", trunk_units(width, TRUNK_VALUES_PER_LOAD), 1,
                           TRUNK_MOST_WIDTH / TRUNK_VALUES_PER_LOAD);
        probe_launch_value(&valid, name, "vectors", rows, 1, TRUNK_MOST_ROWS);
        if (valid) {
            probe_launch_geometry(&valid, name, NULL, (width + TRUNK_ELEMENT_THREADS - 1) / TRUNK_ELEMENT_THREADS, rows,
                                  1, TRUNK_ELEMENT_THREADS, 0);
            trunk_range(&valid, name, "values", trunk.values, step->quantize.values, rows * width * TRUNK_HALF_BYTES);
            trunk_range(&valid, name, "codes", trunk.codes, step->quantize.codes, rows * width);
            trunk_range(&valid, name, "exponents", trunk.exponents, step->quantize.exponents,
                        rows * width / TRUNK_VALUES_PER_BLOCK);
            trunk_range(&valid, name, "operands", trunk.operands, step->quantize.operands,
                        rows * width * TRUNK_HALF_BYTES);
        }
    }
    for (uint32_t index = 0; index < step->product_count; index++) {
        const struct trunk_product_arguments *product = &step->products[index];
        const uint64_t width = product->input_size, loads = trunk_units(width, TRUNK_VALUES_PER_LOAD);
        const uint64_t rows = product->row_count;
        const char *name = "trunk_product";
        uint64_t tiles = 0;
        probe_launch_value(&valid, name, "loads of 64 inputs", loads, 1, TRUNK_MOST_WIDTH / TRUNK_VALUES_PER_LOAD);
        probe_launch_value(&valid, name, "rows", rows, 1, TRUNK_MOST_ROWS);
        probe_launch_value(&valid, name, "slices: the block's whole warps, at most the loads",
                           trunk_units(step->block_threads, TRUNK_THREADS_PER_WARP), 1,
                           loads < TRUNK_MOST_SLICES ? loads : (uint64_t)TRUNK_MOST_SLICES);
        for (uint32_t merged = 0; valid && merged < TRUNK_MERGED_MATRICES; merged++) {
            const struct trunk_matrix *matrix = &product->matrices[merged];
            const uint64_t outputs = matrix->output_size, weights = outputs * width;
            if (merged > 0 && outputs == 0) {
                break;                           /* one matrix: the grid, which is of both, ends with its tiles */
            }
            probe_launch_value(&valid, name, "a second matrix: of a merged kernel", merged, 0, kernel->merged != 0);
            probe_launch_value(&valid, name, "blocks of 32 outputs", trunk_units(outputs, TRUNK_OUTPUTS_PER_SCALE), 1,
                               TRUNK_MOST_OUTPUTS / TRUNK_OUTPUTS_PER_SCALE);
            trunk_range(&valid, name, "a stored matrix", trunk.stored[store], matrix->stored,
                        store == TRUNK_STORE_BF16 ? weights * TRUNK_HALF_BYTES
                            : weights + weights / (TRUNK_OUTPUTS_PER_SCALE * TRUNK_VALUES_PER_BLOCK));
            trunk_range(&valid, name, "outputs", trunk.outputs, matrix->output, rows * outputs * TRUNK_HALF_BYTES);
            tiles += outputs / TRUNK_OUTPUTS_PER_TILE;
        }
        if (valid) {
            probe_launch_geometry(&valid, name, NULL, tiles, 1, 1, step->block_threads,
                                  (uint64_t)step->block_threads * TRUNK_SHARED_BYTES_PER_THREAD);
            probe_launch_value(&valid, name, "threads its registers leave a block", step->block_threads, 1,
                               trunk.threads_most[step->kernel]);
            trunk_range(&valid, name, "what the rows feed", fed_array, product->fed, rows * width * value_bytes);
            trunk_range(&valid, name, "exponents", trunk.exponents, product->exponents,
                        rows * width / TRUNK_VALUES_PER_BLOCK);
        }
    }
    return valid;
}

/* The step of an instance on rows first_row .. first_row + row_count - 1 (a row alone is the same launches on its
 * own vectors), reading the copies of ring step ring_step. 0: refused, as the REFUSED lines say. */
static int trunk_build_step(const struct probe_case *probe_case, uint32_t instance, uint64_t ring_step,
                            uint32_t first_row, uint32_t row_count, struct trunk_step *step)
{
    const struct trunk_variant *variant = &trunk_variants[instance];
    const struct trunk_kernel *kernel = &trunk_kernels[variant->kernel];
    const int form = kernel->form, store = form == TRUNK_BF16_WEIGHTS ? TRUNK_STORE_BF16 : TRUNK_STORE_E4M3;
    const uint32_t width = trunk_shapes[trunk_matrix_of(probe_case, 0)].input_size;
    const size_t vectors = (size_t)first_row * width;
    uint64_t value_bytes = 0;
    const uint8_t *fed_array = trunk_fed_array(form, &value_bytes);
    int valid = 1;
    PROBE_INDEX_CHECK(first_row + row_count - 1u, TRUNK_MOST_ROWS);
    memset(step, 0, sizeof(*step));
    step->kernel = variant->kernel;
    step->block_threads = (uint32_t)probe_case_value(probe_case, TRUNK_FLAG_SLICES) * TRUNK_THREADS_PER_WARP;
    step->quantizer = trunk_quantizer_of(form);
    step->products[0].row_count = row_count;     /* also of a step that is its quantization alone */
    step->quantize.values = trunk.values + vectors;
    step->quantize.codes = trunk.codes + vectors;
    step->quantize.exponents = trunk.exponents + vectors / TRUNK_VALUES_PER_BLOCK;
    step->quantize.operands = trunk.operands + vectors;
    step->quantize.width = width;
    for (uint32_t index = 0; index < TRUNK_MERGED_MATRICES && trunk_matrix_of(probe_case, index) != NONE &&
                             trunk_shapes[trunk_matrix_of(probe_case, index)].output_size != 0; index++) {
        const uint32_t matrix = trunk_matrix_of(probe_case, index);
        struct trunk_product_arguments *product = &step->products[kernel->merged ? 0 : index];
        struct trunk_matrix *target = &product->matrices[kernel->merged ? index : 0];
        target->stored = trunk_stored(store, matrix, variant->same_copy ? 0 : ring_step);
        target->output = trunk.outputs + (size_t)index * TRUNK_SLOT_VALUES +
                         (size_t)first_row * trunk_shapes[matrix].output_size;
        target->output_size = trunk_shapes[matrix].output_size;
        product->fed = fed_array + vectors * value_bytes;
        product->exponents = step->quantize.exponents;
        product->input_size = width;
        product->row_count = row_count;
        step->product_count = kernel->merged ? 1 : index + 1u;
        if (variant->touches) {
            step->touches[step->touch_count].words = (const uint32_t *)target->stored;
            step->touches[step->touch_count++].word_count = trunk_stored_bytes(store, matrix) / sizeof(uint32_t);
        }
    }
    probe_launch_value(&valid, "trunk_step", "launches: a pass has only instances that quantize", step->product_count +
                       (step->quantizer != TRUNK_NO_LAUNCH ? 1u : 0u), 1, TRUNK_MERGED_MATRICES + 1);
    return valid && trunk_step_is_valid(step);
}

/* Queues the launches of a validated step; returns their number. */
static uint32_t trunk_launch(const struct trunk_step *step)
{
    const unsigned int touch_blocks = (unsigned int)trunk.sm_count * TRUNK_TOUCH_BLOCKS_PER_SM;
    const dim3 vectors((step->quantize.width + TRUNK_ELEMENT_THREADS - 1) / TRUNK_ELEMENT_THREADS,
                       step->products[0].row_count);
#if PROBE_DIAGNOSTICS
    PROBE_REQUIRE(trunk_step_is_valid(step), "a step that its preparation did not validate");
#endif
    for (uint32_t index = 0; index < step->touch_count; index++) {
        trunk_touch_kernel<<<touch_blocks, TRUNK_ELEMENT_THREADS>>>(step->touches[index].words,
                                                                    step->touches[index].word_count, trunk.touched);
    }
    if (step->quantizer == TRUNK_PLACED16) {
        trunk_quantize_placed16<<<vectors, TRUNK_ELEMENT_THREADS>>>(step->quantize);
    } else if (step->quantizer == TRUNK_CARRIED_BF16) {
        trunk_quantize_carried_bf16<<<vectors, TRUNK_ELEMENT_THREADS>>>(step->quantize);
    } else if (step->quantizer == TRUNK_WHOLE_S8) {
        trunk_quantize_whole_s8<<<vectors, TRUNK_ELEMENT_THREADS>>>(step->quantize);
    } else if (step->quantizer != TRUNK_NO_LAUNCH) {
        trunk_quantize_codes<<<vectors, TRUNK_ELEMENT_THREADS>>>(step->quantize);
    }
    for (uint32_t index = 0; index < step->product_count; index++) {
        const struct trunk_matrix *matrices = step->products[index].matrices;
        trunk_kernels[step->kernel].launch((matrices[0].output_size + matrices[1].output_size) / TRUNK_OUTPUTS_PER_TILE,
                                           step->block_threads, &step->products[index]);
    }
    PROBE_AFTER_LAUNCH();
    return step->touch_count + step->product_count + (step->quantizer != TRUNK_NO_LAUNCH ? 1u : 0u);
}

static int trunk_open_card(void *, struct probe_card_facts *facts)
{
    const struct probe_flag *chosen = probe_flag(TRUNK_FLAG_MATRIX);
    uint64_t store_bytes[TRUNK_STORES] = {0, 0};
    int used[TRUNK_MATRICES + 1] = {0};          /* the last: of a case that has no second matrix */
    if (!probe_cuda_open(facts)) {
        return 0;
    }
    trunk.sm_count = facts->sm_count;
    for (uint32_t kernel = 0; kernel < TRUNK_KERNEL_COUNT; kernel++) {
        struct cudaFuncAttributes attributes;
        PROBE_CHECK_CUDA(cudaFuncGetAttributes(&attributes, trunk_kernels[kernel].function));
        trunk.threads_most[kernel] = attributes.maxThreadsPerBlock > 0 ? (uint32_t)attributes.maxThreadsPerBlock : 0u;
    }
    for (uint32_t value = 0; value < chosen->value_count; value++) {         /* the matrices this run's cases read */
        for (uint32_t index = 0; index < TRUNK_MERGED_MATRICES; index++) {
            used[trunk_cases[chosen->values[value]][index]] = 1;
        }
    }
    for (uint32_t pass = 0; pass < TRUNK_STORES * TRUNK_MATRICES; pass++) {  /* a ring a store and matrix, in a row */
        const int store = (int)(pass / TRUNK_MATRICES);
        const uint32_t matrix = pass % TRUNK_MATRICES;
        const uint64_t unit = TRUNK_MATRIX_ALIGNMENT_BYTES;
        const uint64_t bytes = (trunk_stored_bytes(store, matrix) + unit - 1) / unit * unit;
        const int checks_only = probe_flag_given(PROBE_FLAG_STEPS) && !probe_flag_text(PROBE_FLAG_PROFILE_CASE)[0];
        const uint64_t ring = (checks_only ? 1 : (uint64_t)TRUNK_COLD_L2_SIZES) * facts->l2_bytes; /* CONVENTIONS 3.5 */
        const uint64_t copies = used[matrix] && bytes != 0 ? (ring + bytes - 1) / bytes : 0;
        trunk.copy_bytes[store][matrix] = bytes;
        trunk.copies[store][matrix] = copies;
        trunk.ring_at[store][matrix] = store_bytes[store];
        store_bytes[store] += copies * bytes;
        facts->ring_bytes = copies != 0 && (facts->ring_bytes == 0 || copies * bytes < facts->ring_bytes)
                                ? copies * bytes : facts->ring_bytes;       /* CARD shows the smallest ring */
    }
    for (uint32_t pass = 0; pass < TRUNK_STORES * TRUNK_MATRICES; pass++) {
        const int store = (int)(pass / TRUNK_MATRICES);
        const uint32_t matrix = pass % TRUNK_MATRICES;
        const unsigned int fill_blocks = (unsigned int)facts->sm_count * TRUNK_FILL_BLOCKS_PER_SM;
        const uint64_t weights = (uint64_t)trunk_shapes[matrix].output_size * trunk_shapes[matrix].input_size;
        struct trunk_fill_arguments fill;
        int valid = 1;
        if (matrix == 0 && store_bytes[store] != 0) {
            trunk.stored[store] = (uint8_t *)probe_cuda_allocate(store_bytes[store]);
        }
        if (trunk.copies[store][matrix] == 0) {
            continue;
        }
        fill.words = (uint32_t *)trunk_stored(store, matrix, 0);
        fill.words_per_copy = trunk.copy_bytes[store][matrix] / sizeof(uint32_t);
        fill.word_count = trunk.copies[store][matrix] * fill.words_per_copy;
        fill.scale_word = store == TRUNK_STORE_E4M3 ? weights / TRUNK_WORD_BYTES : UINT64_MAX;
        fill.kind = (uint32_t)store;
        fill.seed = trunk_seed(matrix);
        probe_launch_geometry(&valid, "trunk_fill_kernel", NULL, fill_blocks, 1, 1, TRUNK_ELEMENT_THREADS, 0);
        trunk_range(&valid, "trunk_fill_kernel", "a ring", trunk.stored[store], fill.words,
                    fill.word_count * sizeof(uint32_t));
        PROBE_REQUIRE(valid, "the launch that fills a ring was refused");
        trunk_fill_kernel<<<fill_blocks, TRUNK_ELEMENT_THREADS>>>(fill);
        PROBE_CHECK_CUDA(cudaGetLastError());
    }
    trunk.values = (uint16_t *)probe_cuda_allocate(sizeof(trunk_host.values));
    trunk.values_fp16 = (uint16_t *)probe_cuda_allocate(sizeof(trunk_host.values_fp16));
    trunk.operands = (uint16_t *)probe_cuda_allocate(sizeof(trunk_host.operands));
    PROBE_CHECK_CUDA(cudaMemset(trunk.operands, 0, sizeof(trunk_host.operands)));   /* read back before any is placed */
    trunk.codes = (uint8_t *)probe_cuda_allocate(sizeof(trunk_host.codes));
    trunk.exponents = (uint8_t *)probe_cuda_allocate(sizeof(trunk_host.exponents));
    trunk.outputs = (uint16_t *)probe_cuda_allocate(sizeof(trunk_host.outputs));
    trunk.touched = (uint32_t *)probe_cuda_allocate(sizeof(uint32_t) * (size_t)facts->sm_count *
                                                    TRUNK_TOUCH_BLOCKS_PER_SM * TRUNK_ELEMENT_THREADS);
    PROBE_CHECK_CUDA(cudaDeviceSynchronize());
    return 1;
}

static void trunk_close_card(void *) { probe_cuda_close(); }

/* An input value. The blocks of 32 of a row take turns at what CONVENTIONS 5.7 asks for: plain values in steps of
 * 1/64; one large value beside values that become subnormal codes; zeros of both signs; a block under the floor of
 * 1e-4; the largest code at its scale beside ties between neighbouring codes; magnitudes spread over 2^17. All but
 * the fourth reach about 2 and so weigh on an output; every value is a bfloat16 and a 16-bit float exactly. */
static float trunk_input_value(uint32_t row, uint32_t column)
{
    const uint32_t block = column / TRUNK_VALUES_PER_BLOCK, place = column % TRUNK_VALUES_PER_BLOCK;
    const uint32_t bits = trunk_mix_bits(row * TRUNK_MOST_WIDTH + column + 1u);
    const float plain = ((float)(bits % TRUNK_INPUT_STEPS) - (float)TRUNK_INPUT_CENTRE) / (float)TRUNK_VALUES_PER_LOAD;
    const float sign = ((bits >> TRUNK_HALF_BITS) & 1u) != 0 ? -1.0f : 1.0f;
    const float eighths = (float)((bits >> TRUNK_BYTE_BITS) & TRUNK_E4M3_MANTISSA_MASK);
    const float largest = ldexpf(TRUNK_E4M3_LARGEST, -TRUNK_LARGEST_SHIFT);
    const int tie_exponent = 1 + (int)((bits >> TRUNK_E4M3_MANTISSA_BITS) % TRUNK_TIE_EXPONENTS) - TRUNK_TIE_EXPONENT;
    switch ((block + row) % TRUNK_INPUT_KINDS) {
    case TRUNK_INPUT_SUBNORMAL: return place == 0 ? largest : ldexpf(plain, -TRUNK_SUBNORMAL_SHIFT);
    case TRUNK_INPUT_ZEROS: return (place & 1u) != 0 ? plain : sign * 0.0f;
    case TRUNK_INPUT_UNDER_FLOOR: return ldexpf(plain, -TRUNK_UNDER_FLOOR_SHIFT);
    case TRUNK_INPUT_TIES:
        return place == 0 ? largest : sign * ldexpf((float)TRUNK_TIE_UNITS + eighths + eighths, tie_exponent);
    case TRUNK_INPUT_SPREAD:
        return sign * ldexpf(1.0f + eighths / (float)TRUNK_E4M3_IMPLICIT_ONE,
                             (int)(place % TRUNK_SPREAD_PLACES) - (TRUNK_SPREAD_PLACES - 1));
    default: return plain;
    }
}

static void trunk_describe_case(void *, const struct probe_case *probe_case, struct probe_case_facts *facts)
{
    static char part[PROBE_NAME_BYTES];
    const uint32_t first = trunk_matrix_of(probe_case, 0), pair = trunk_matrix_of(probe_case, 1);
    const uint32_t width = trunk_shapes[first].input_size;
    const char *word = probe_case->name + strlen("matrix:");                 /* "<matrix>,slices:<S>,rows:<R>" */
    PROBE_REQUIRE(snprintf(part, sizeof(part), "%.*s", (int)(strrchr(word, ',') - word), word) > 0, "a case's part");
    facts->format = "e4m3";
    facts->part = part;
    facts->rows = (int64_t)probe_case_value(probe_case, TRUNK_FLAG_ROWS);
    facts->experts = facts->overlap = -1;
    facts->megabytes = (double)(trunk_stored_bytes(TRUNK_STORE_E4M3, first) +
                                (pair == NONE ? 0 : trunk_stored_bytes(TRUNK_STORE_E4M3, pair))) / TRUNK_BYTES_PER_MB;
    for (uint32_t index = 0; index < TRUNK_MOST_ROWS * width; index++) {     /* the case's rows, as both floats */
        const float value = trunk_input_value(index / width, index % width);
        trunk_host.values[index] = (uint16_t)(trunk_bits_of(value) >> TRUNK_HALF_BITS);
        trunk_host.values_fp16[index] = __half_as_ushort(__float2half(value));
    }
    PROBE_CHECK_CUDA(cudaMemcpy(trunk.values, trunk_host.values, sizeof(trunk_host.values), cudaMemcpyHostToDevice));
    PROBE_CHECK_CUDA(cudaMemcpy(trunk.values_fp16, trunk_host.values_fp16, sizeof(trunk_host.values_fp16),
                                cudaMemcpyHostToDevice));
}

/* ---- the formats on the host, from their definitions ---- */

static double trunk_e4m3_value(uint32_t code)   /* sign, four exponent bits of bias 7, three mantissa bits */
{
    const int exponent = (int)((code >> TRUNK_E4M3_MANTISSA_BITS) & TRUNK_E4M3_EXPONENT_MASK);
    const int units = (int)(code & TRUNK_E4M3_MANTISSA_MASK) + (exponent == 0 ? 0 : TRUNK_E4M3_IMPLICIT_ONE);
    const double magnitude = ldexp(units, (exponent == 0 ? 1 : exponent) - TRUNK_E4M3_LEAST_EXPONENT);
    return (code & TRUNK_E4M3_SIGN_BIT) != 0 ? -magnitude : magnitude;
}

/* A byte of a matrix as its E4M3 ring stores it: `at` among its weights' bytes, or behind them among its scales. */
static uint32_t trunk_stored_byte(uint32_t matrix, uint64_t at)
{
    const uint64_t weights = (uint64_t)trunk_shapes[matrix].output_size * trunk_shapes[matrix].input_size;
    return (trunk_stored_word(trunk_seed(matrix), at / TRUNK_WORD_BYTES, weights / TRUNK_WORD_BYTES) >>
            (at % TRUNK_WORD_BYTES * TRUNK_BYTE_BITS)) & TRUNK_BYTE_MASK;
}

/* Where the formats put the weight of an output at an input (engine/include/anemone/weight_format_seam.h, and for
 * bfloat16 the contract of trunk_forms_kernels.cuh): its byte among the E4M3 bytes, or its half among the halves. */
static uint64_t trunk_weight_place(uint32_t matrix, uint32_t output, uint32_t input, int bfloat16)
{
    const uint64_t part_inputs = bfloat16 ? TRUNK_VALUES_PER_STEP : TRUNK_VALUES_PER_BLOCK;     /* of 16 bytes a lane */
    const uint64_t lane = output % TRUNK_OUTPUTS_PER_HALF_TILE * TRUNK_LANES_PER_LANE_GROUP +
                          input % TRUNK_VALUES_PER_HALF_STEP / TRUNK_VALUES_PER_LANE_PLACE;
    const uint64_t part = (output / TRUNK_OUTPUTS_PER_TILE * (trunk_shapes[matrix].input_size / part_inputs) +
                           input / part_inputs) * TRUNK_THREADS_PER_WARP + lane;
    const uint64_t upper_output = output % TRUNK_OUTPUTS_PER_TILE / TRUNK_OUTPUTS_PER_HALF_TILE;
    const uint64_t eighth = input % part_inputs / TRUNK_VALUES_PER_HALF_STEP, odd = input % TRUNK_VALUES_PER_LANE_PLACE;
    return bfloat16 ? (part * TRUNK_REGISTERS_PER_OPERAND_A + upper_output + eighth * TRUNK_OUTPUT_HALVES) *
                          TRUNK_HALVES_PER_REGISTER + odd
                    : part * TRUNK_PART_BYTES + eighth * TRUNK_WORD_BYTES + odd * TRUNK_OUTPUT_HALVES + upper_output;
}

/* A weight as a form reads it: an E4M3 number, or the same byte as a signed whole number, times its block's
 * scale; or the bfloat16 that the second ring holds. */
static double trunk_weight(uint32_t matrix, uint32_t output, uint32_t input, int form)
{
    const uint64_t width = trunk_shapes[matrix].input_size, blocks = width / TRUNK_VALUES_PER_BLOCK;
    const uint64_t half = trunk_weight_place(matrix, output, input, 1);
    const uint32_t byte = trunk_stored_byte(matrix, trunk_weight_place(matrix, output, input, 0));
    const uint32_t scale = trunk_stored_byte(matrix, trunk_shapes[matrix].output_size * width +
                                                         output / TRUNK_OUTPUTS_PER_SCALE * blocks +
                                                         input / TRUNK_VALUES_PER_BLOCK);
    const uint32_t word = trunk_stored_word(trunk_seed(matrix), half / TRUNK_HALVES_PER_REGISTER, UINT64_MAX);
    const double coded = form == TRUNK_WHOLE_S8 ? (double)(int8_t)byte : trunk_e4m3_value(byte);
    return form != TRUNK_BF16_WEIGHTS ? ldexp(coded, (int)scale - TRUNK_E8M0_BIAS)
               : trunk_bfloat16_weight(word >> (half % TRUNK_HALVES_PER_REGISTER * TRUNK_BYTE_BITS));
}

static double trunk_bfloat16_number(uint16_t bits) { return trunk_float_of((uint32_t)bits << TRUNK_HALF_BITS); }

static uint16_t trunk_bfloat16_of(double value)  /* rounded once to eight significant bits: nearest, a tie to even */
{
    int exponent = 0;
    const double fraction = frexp(value, &exponent);
    const double rounded = ldexp(nearbyint(ldexp(fraction, TRUNK_BYTE_BITS)), exponent - TRUNK_BYTE_BITS);
    return (uint16_t)(trunk_bits_of((float)rounded) >> TRUNK_HALF_BITS);
}

/* The value whose code is byte `place` of a vector's codes (lane order: linear_records.h), and where operand B
 * has its 16-bit half: the same instruction of the same lane place, the two bits within it exchanged. */
static uint32_t trunk_value_of_place(uint32_t place)
{
    const uint32_t byte = place % TRUNK_WORD_BYTES, step = place / TRUNK_WORD_BYTES % TRUNK_STEPS_PER_LOAD;
    const uint32_t lane_place = place / TRUNK_PART_BYTES % TRUNK_LANES_PER_LANE_GROUP;
    return place / TRUNK_VALUES_PER_LOAD * TRUNK_VALUES_PER_LOAD + step * TRUNK_VALUES_PER_STEP +
           lane_place * TRUNK_VALUES_PER_LANE_PLACE + byte / TRUNK_HALVES_PER_REGISTER +
           byte % TRUNK_HALVES_PER_REGISTER * TRUNK_VALUES_PER_HALF_STEP;
}

static uint32_t trunk_operand_place(uint32_t place)
{
    const uint32_t byte = place % TRUNK_WORD_BYTES, halves = TRUNK_HALVES_PER_REGISTER;
    return place - byte + byte % halves * halves + byte / halves;
}

/* The E4M3 code nearest to a value already divided by its scale: its magnitude limited to 448, a tie to the even
 * code, the sign kept also where it is zero. */
static uint32_t trunk_host_code(float scaled)
{
    static double numbers[TRUNK_E4M3_MAGNITUDES];           /* of the codes 0 .. 0x7E, ascending: filled once */
    const double magnitude = fmin(fabs((double)scaled), (double)TRUNK_E4M3_LARGEST);
    uint32_t best = 0;
    for (uint32_t code = 1; code < TRUNK_E4M3_MAGNITUDES; code++) {
        numbers[code] = numbers[code] != 0.0 ? numbers[code] : trunk_e4m3_value(code);
        const double distance = fabs(numbers[code] - magnitude), best_distance = fabs(numbers[best] - magnitude);
        if (distance < best_distance || (distance == best_distance && (code & 1u) == 0)) {
            best = code;
        }
    }
    return best | (signbit(scaled) ? (uint32_t)TRUNK_E4M3_SIGN_BIT : 0u);
}

/* ---- checks ---- */

/* The checked step of an instance on a window of rows: validated, every output made stale first (a word a launch
 * does not write cannot pass for a result), launched, the outputs read into the host's [matrix][row][output_size].
 * 0: refused. */
static int trunk_run_checked(const struct probe_case *probe_case, uint32_t instance, uint32_t first_row,
                             uint32_t row_count, uint16_t *outputs)
{
    struct trunk_step step;
    if (!trunk_build_step(probe_case, instance, 0, first_row, row_count, &step)) {
        return 0;
    }
    PROBE_CHECK_CUDA(cudaMemset(trunk.outputs, TRUNK_STALE_BYTE, sizeof(trunk_host.outputs)));
    trunk_launch(&step);
    PROBE_CHECK_CUDA(cudaGetLastError());
    PROBE_CHECK_CUDA(cudaMemcpy(outputs, trunk.outputs, sizeof(trunk_host.outputs), cudaMemcpyDeviceToHost));
    return 1;
}

/* The words of a window of rows that differ between two host copies of the outputs; *of gains the words compared. */
static uint64_t trunk_differing(const struct probe_case *probe_case, uint32_t first_row, uint32_t row_count,
                                const uint16_t *first, const uint16_t *second, uint64_t *of)
{
    uint64_t differing = 0;
    for (uint32_t index = 0; index < TRUNK_MERGED_MATRICES && trunk_matrix_of(probe_case, index) != NONE; index++) {
        const uint32_t output_size = trunk_shapes[trunk_matrix_of(probe_case, index)].output_size;
        const size_t at = (size_t)index * TRUNK_SLOT_VALUES + (size_t)first_row * output_size;
        for (size_t word = 0; word < (size_t)row_count * output_size; word++) {
            differing += first[at + word] != second[at + word] ? 1u : 0u;
        }
        *of += (uint64_t)row_count * output_size;
    }
    return differing;
}

/* The outputs of one matrix that trunk_against_host samples: 256 of the first row and, of several, of the last. */
static uint32_t trunk_sampled(uint32_t rows) { return rows > 1 ? TRUNK_SAMPLES + TRUNK_SAMPLES : TRUNK_SAMPLES; }

/* One product against the host for 256 outputs (the first few, then spread to the last) of the first and of the
 * last row: the rms of what the card stored less the double-precision sum rounded once to bfloat16, over the rms
 * of those (against the sums themselves a right vector is 1.7e-3 off by its rounding alone), and into *equal the
 * outputs that are equal. whole_slices above 0: instead the outputs that are not, bit for bit, the whole-number
 * contract of that many slices: exact sums of 32 products, a float32 chain a slice, slices in order, one rounding. */
static double trunk_against_host(uint32_t matrix, int form, uint32_t row_count, const uint16_t *card,
                                 uint32_t whole_slices, uint32_t *equal)
{
    const uint32_t width = trunk_shapes[matrix].input_size, output_size = trunk_shapes[matrix].output_size;
    const uint32_t blocks = width / TRUNK_VALUES_PER_BLOCK, loads = width / TRUNK_VALUES_PER_LOAD;
    double error_squares = 0.0, squares = 0.0, not_the_contract = 0.0;
    for (uint32_t row = 0; row < row_count; row += row_count > 1 ? row_count - 1u : 1u) {
        for (uint32_t sample = 0; sample < TRUNK_SAMPLES; sample++) {
            const uint32_t output = sample < TRUNK_LEADING_SAMPLES ? sample * TRUNK_LEADING_STRIDE % output_size
                                    : (uint32_t)((uint64_t)(output_size - 1u) * (sample - TRUNK_LEADING_SAMPLES + 1u) /
                                                 (TRUNK_SAMPLES - TRUNK_LEADING_SAMPLES));
            const uint16_t stored = card[(size_t)row * output_size + output];
            const uint64_t scales_at = (uint64_t)output_size * width + output / TRUNK_OUTPUTS_PER_SCALE * blocks;
            double sum = 0.0;
            float total = 0.0f;
            for (uint32_t input = 0; whole_slices == 0 && input < width; input++) {
                sum += trunk_weight(matrix, output, input, form) * trunk_host.fed[(size_t)row * width + input];
            }
            for (uint32_t slice = 0; slice < whole_slices; slice++) {
                float chain = 0.0f;
                for (uint32_t block = slice * loads / whole_slices * TRUNK_BLOCKS_PER_LOAD;
                     block < (slice + 1u) * loads / whole_slices * TRUNK_BLOCKS_PER_LOAD; block++) {
                    const int power = (int)trunk_stored_byte(matrix, scales_at + block) +
                                      trunk_host.exponents[(size_t)row * blocks + block] + TRUNK_WHOLE_CODE_EXPONENT -
                                      TRUNK_E8M0_BIAS - TRUNK_E8M0_BIAS;
                    int64_t whole = 0;
                    for (uint32_t input = block * TRUNK_VALUES_PER_BLOCK; input < (block + 1u) * TRUNK_VALUES_PER_BLOCK;
                         input++) {
                        const uint32_t byte = trunk_stored_byte(matrix, trunk_weight_place(matrix, output, input, 0));
                        whole += (int64_t)(int8_t)byte * (int8_t)trunk_host.codes_by_value[(size_t)row * width + input];
                    }
                    chain = fmaf((float)whole, ldexpf(1.0f, power), chain);
                }
                total += chain;
            }
            const double rounded = trunk_bfloat16_number(trunk_bfloat16_of(sum));
            not_the_contract += trunk_bfloat16_of(total) != stored ? 1.0 : 0.0;
            error_squares += (trunk_bfloat16_number(stored) - rounded) * (trunk_bfloat16_number(stored) - rounded);
            squares += rounded * rounded;
            *equal += whole_slices == 0 && trunk_bfloat16_of(sum) == stored ? 1u : 0u;
        }
    }
    return whole_slices > 0 ? not_the_contract : sqrt(error_squares) / sqrt(squares);
}

/* A checked step against the host: what its quantizer wrote, then each product fed with what the card holds.
 * must_report: the megabyte of the case's first matrix was overwritten before this launch. */
static void trunk_check_against_host(const struct probe_case *probe_case, uint32_t instance, int must_report,
                                     uint32_t *equal)
{
    const uint32_t rows = (uint32_t)probe_case_value(probe_case, TRUNK_FLAG_ROWS);
    const uint32_t slices = (uint32_t)probe_case_value(probe_case, TRUNK_FLAG_SLICES);
    const uint32_t width = trunk_shapes[trunk_matrix_of(probe_case, 0)].input_size;
    const int form = trunk_kernels[trunk_variants[instance].kernel].form;
    const int has_codes = form != TRUNK_FP16 && form != TRUNK_BF16_WEIGHTS;
    const int has_operands = form == TRUNK_PLACED16 || form == TRUNK_CARRIED_BF16;
    const int floor_doubled = probe_flag_has_word(TRUNK_FLAG_BREAK, "quantization");
    const float floor = floor_doubled ? TRUNK_MAGNITUDE_FLOOR + TRUNK_MAGNITUDE_FLOOR : TRUNK_MAGNITUDE_FLOOR;
    const char *variant = trunk_instances[instance].name;
    const size_t count = (size_t)rows * width, block_count = count / TRUNK_VALUES_PER_BLOCK;
    char check[TRUNK_FIGURE_BYTES];
    uint64_t differing = 0;
    PROBE_CHECK_CUDA(cudaMemcpy(trunk_host.operands, trunk.operands, count * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    PROBE_CHECK_CUDA(cudaMemcpy(trunk_host.codes, trunk.codes, count, cudaMemcpyDeviceToHost));
    PROBE_CHECK_CUDA(cudaMemcpy(trunk_host.exponents, trunk.exponents, block_count, cudaMemcpyDeviceToHost));
    for (size_t index = 0; index < count; index++) {      /* a row's byte `place`, the value it codes, its block */
        const uint32_t place = (uint32_t)(index % width), value_at = trunk_value_of_place(place);
        const size_t row_at = index - place, block_at = (row_at + value_at) / TRUNK_VALUES_PER_BLOCK;
        const float value = trunk_float_of((uint32_t)trunk_host.values[row_at + value_at] << TRUNK_HALF_BITS);
        float largest = 0.0f;
        int exponent = 0;
        for (size_t at = block_at * TRUNK_VALUES_PER_BLOCK; at < (block_at + 1u) * TRUNK_VALUES_PER_BLOCK; at++) {
            largest = fmaxf(largest, fabsf(trunk_float_of((uint32_t)trunk_host.values[at] << TRUNK_HALF_BITS)));
        }
        /* 2^(code - 127): the least power of two not under max(largest, floor) / 448, in float32; frexp of one: 1/2 */
        const float fraction = frexpf(fmaxf(largest, floor) * TRUNK_E4M3_LARGEST_INVERSE, &exponent);
        const int code = exponent - (fraction + fraction == 1.0f ? 1 : 0) + TRUNK_E8M0_BIAS;
        const float scaled = ldexpf(value, TRUNK_E8M0_BIAS - code);
        const uint32_t byte = form == TRUNK_WHOLE_S8 ? (uint32_t)(uint8_t)lrintf(scaled * TRUNK_WHOLE_CODE_FACTOR)
                                                     : trunk_host_code(scaled);
        const float placed = (float)ldexp(trunk_e4m3_value(byte), form == TRUNK_PLACED16 ? -TRUNK_E4M3_FP16_EXPONENT
                                              : code - TRUNK_E8M0_BIAS + TRUNK_CARRIED_EXPONENT);
        const uint16_t operand = form == TRUNK_PLACED16 ? __half_as_ushort(__float2half(placed))
                                                        : (uint16_t)(trunk_bits_of(placed) >> TRUNK_HALF_BITS);
        const double card_scale = ldexp(1.0, trunk_host.exponents[block_at] - TRUNK_E8M0_BIAS);
        const int first_of_block = value_at % TRUNK_VALUES_PER_BLOCK == 0;       /* its exponent: compared once */
        differing += has_codes && trunk_host.codes[index] != byte ? 1u : 0u;
        differing += has_codes && first_of_block && trunk_host.exponents[block_at] != code ? 1u : 0u;
        differing += has_operands && trunk_host.operands[row_at + trunk_operand_place(place)] != operand ? 1u : 0u;
        PROBE_REQUIRE(form != TRUNK_CARRIED_BF16 || code <= TRUNK_CARRIED_EXPONENT_CODE_MOST,
                      "a block too large for the carried operand: its exponent code is above 145");
        trunk_host.codes_by_value[row_at + value_at] = trunk_host.codes[index];
        trunk_host.fed[row_at + value_at] =
            form == TRUNK_FP16 ? (double)__half2float(__ushort_as_half(trunk_host.values_fp16[row_at + value_at]))
            : !has_codes ? (double)value
            : form == TRUNK_WHOLE_S8 ? (int8_t)trunk_host.codes[index] * card_scale / TRUNK_WHOLE_CODE_FACTOR
            : trunk_e4m3_value(trunk_host.codes[index]) * card_scale;
    }
    if (has_codes) {
        probe_check_count(probe_case->name, variant, "quantizer", differing,
                          count + block_count + (has_operands ? count : 0), floor_doubled);
    }
    for (uint32_t index = 0; index < TRUNK_MERGED_MATRICES && trunk_matrix_of(probe_case, index) != NONE &&
                             trunk_shapes[trunk_matrix_of(probe_case, index)].output_size != 0; index++) {
        const uint32_t matrix = trunk_matrix_of(probe_case, index);
        const uint16_t *card = trunk_host.outputs + (size_t)index * TRUNK_SLOT_VALUES;
        const char *name = trunk_shapes[matrix].name;
        PROBE_REQUIRE(snprintf(check, sizeof(check), "host_reference_%s", name) > 0, "a name");
        probe_check_error(probe_case->name, variant, check, trunk_against_host(matrix, form, rows, card, 0, equal),
                          instance == TRUNK_INSTANCE_row_dependent ? (double)INFINITY : TRUNK_TOLERANCE,
                          must_report && index == 0);       /* the instance that rounds by its rows: no limit */
        if (form == TRUNK_WHOLE_S8) {
            PROBE_REQUIRE(snprintf(check, sizeof(check), "integer_contract_%s", name) > 0, "a name");
            probe_check_count(probe_case->name, variant, check,
                              (uint64_t)trunk_against_host(matrix, form, rows, card, slices, equal),
                              trunk_sampled(rows), must_report && index == 0);
        }
    }
}

/* A megabyte from the start of the copy that the checked step reads of the case's first matrix, in both rings:
 * the host still takes the weights from their definition. */
static void trunk_overwrite(uint32_t matrix)
{
    for (int store = 0; store < TRUNK_STORES; store++) {
        const uint64_t bytes = trunk_stored_bytes(store, matrix);
        PROBE_CHECK_CUDA(cudaMemset(trunk_stored(store, matrix, 0), TRUNK_CORRUPTED_BYTE,
                                    bytes < TRUNK_CORRUPTED_BYTES ? bytes : (uint64_t)TRUNK_CORRUPTED_BYTES));
    }
}

/* --break-route: the baseline's step past a range (its copy, a whole ring on) and past a count (slices above its
 * loads). The validation every real step passes must refuse each; neither is launched. A refusal is 1 of 1. */
static void trunk_break_route(const struct probe_case *probe_case)
{
    static const char *const names[TRUNK_BROKEN_COUNT] = {"refused_copy_past_the_ring",
                                                          "refused_slices_above_the_loads"};
    for (uint32_t broken = 0; broken < TRUNK_BROKEN_COUNT; broken++) {
        struct trunk_step step;
        PROBE_REQUIRE(trunk_build_step(probe_case, PROBE_BASELINE, 0, 0, 1, &step) && step.product_count > 0,
                      "--break-route needs a case of matrices whose baseline step is valid");
        if (broken == TRUNK_BROKEN_COPY) {
            step.products[0].matrices[0].stored += probe_cuda_array_bytes(trunk.stored[TRUNK_STORE_E4M3]);
        } else {
            step.block_threads = (step.products[0].input_size / TRUNK_VALUES_PER_LOAD + 1u) * TRUNK_THREADS_PER_WARP;
        }
        probe_check_count(probe_case->name, "baseline", names[broken], trunk_step_is_valid(&step) ? 0u : 1u, 1, 1);
    }
}

/* An instance whose kernel's registers leave a block fewer threads than the case's slices ask is out of the case,
 * neither checked nor timed (its RESULT line has n=0), with a REFUSED line that says so: a block of 32 warps holds
 * no kernel of more than 64 registers. The baseline pair is never left out: check 10 refuses its launches, and the
 * run fails. */
static void trunk_fit_instances(const struct probe_case *probe_case)
{
    const uint64_t block_threads = probe_case_value(probe_case, TRUNK_FLAG_SLICES) * TRUNK_THREADS_PER_WARP;
    for (uint32_t instance = PROBE_BASELINE_AGAIN + 1u; instance < trunk_instances_in_run; instance++) {
        int fits = 1;
        probe_launch_value(&fits, trunk_instances[instance].kernel, "threads its registers leave a block",
                           block_threads, 1, trunk.threads_most[trunk_variants[instance].kernel]);
        trunk_instances[instance].available = trunk_variants[instance].compiled && fits;
    }
}

static void trunk_check_case(void *, const struct probe_case *probe_case)
{
    const uint32_t first = trunk_matrix_of(probe_case, 0), pair = trunk_matrix_of(probe_case, 1);
    const uint32_t rows = (uint32_t)probe_case_value(probe_case, TRUNK_FLAG_ROWS);
    const uint32_t sampled = (pair == NONE ? 1 : TRUNK_MERGED_MATRICES) * trunk_sampled(rows);
    const int has_products = trunk_shapes[first].output_size != 0, corrupt = probe_flag_given(PROBE_FLAG_CORRUPT);
    const int break_bits = probe_flag_has_word(TRUNK_FLAG_BREAK, "bits");
    trunk_fit_instances(probe_case);
    if (probe_flag_given(PROBE_FLAG_BREAK_ROUTE)) {
        trunk_break_route(probe_case);
        return;
    }
    if (corrupt) {
        trunk_overwrite(first);
    }
    for (uint32_t instance = 0; instance < trunk_instances_in_run; instance++) {
        const struct trunk_variant *variant = &trunk_variants[instance];
        const uint32_t quantizer = trunk_quantizer_of(trunk_kernels[variant->kernel].form);
        const char *name = trunk_instances[instance].name;
        uint64_t differing = 0, alone_differing = 0, words = 0, alone_words = 0;
        uint32_t equal = 0;
        const int again = quantizer == TRUNK_NO_LAUNCH || (quantizer == TRUNK_CODES && instance != PROBE_BASELINE);
        if (instance == PROBE_BASELINE_AGAIN || !trunk_instances[instance].available || (!has_products && again)) {
            continue;        /* the baseline's kernel again; a form this card lacks; of a pass, a quantizer again */
        }
        if (!trunk_run_checked(probe_case, instance, 0, rows, trunk_host.outputs)) {
            probe_check_count(probe_case->name, name, "launches_valid", 1, 1, 0);
            continue;
        }
        trunk_check_against_host(probe_case, instance, corrupt || (break_bits && instance != PROBE_BASELINE), &equal);
        differing = trunk_differing(probe_case, 0, rows, trunk_host.outputs, trunk_host.baseline, &words);
        if (instance == PROBE_BASELINE) {
            memcpy(trunk_host.baseline, trunk_host.outputs, sizeof(trunk_host.outputs));
            differing = 0;
        } else if (variant->bits_required && has_products) {
            probe_check_count(probe_case->name, name, "bits_against_baseline", differing, words, break_bits);
        }
        trunk_instances[instance].contract_class =
            variant->claimed_class == 'E' && differing != 0 ? 'C' : variant->claimed_class;
        if (has_products) {
            printf("BITS card=%" PRIu64 " case=%s variant=%s differing=%" PRIu64 " of=%" PRIu64 " float64_equal=%u"
                   " of=%u class=%c\n", probe_flag_number("--card-index", 0), probe_case->name, name, differing, words,
                   equal, sampled, trunk_instances[instance].contract_class);
        }
        for (uint32_t row = 0; has_products && rows > 1 && row < rows; row++) {
            PROBE_REQUIRE(trunk_run_checked(probe_case, instance, row, 1, trunk_host.alone), "a row alone was refused");
            alone_differing += trunk_differing(probe_case, row, 1, trunk_host.alone, trunk_host.outputs, &alone_words);
        }
        if (has_products && rows > 1) {
            probe_check_count(probe_case->name, name, "rows_exact", alone_differing, alone_words,
                              instance == TRUNK_INSTANCE_row_dependent);
        }
        probe_digest_add(trunk_host.outputs, sizeof(trunk_host.outputs));   /* what no launch wrote is stale in it */
        if (instance == PROBE_BASELINE && break_bits) {      /* from here on a kernel reads other weights than it did */
            trunk_overwrite(first);
        }
    }
    probe_cuda_check_guards(probe_case->name);
}

static int trunk_prepare_step(void *, const struct probe_case *probe_case, uint32_t instance, uint64_t step,
                              uint32_t slot)
{
    PROBE_REQUIRE(slot < PROBE_STREAM_STEPS, "a slot past the slots of a stream");
    return trunk_build_step(probe_case, instance, step, 0, (uint32_t)probe_case_value(probe_case, TRUNK_FLAG_ROWS),
                            &trunk.steps[slot]);
}

static uint32_t trunk_launch_step(void *, const struct probe_case *, uint32_t, uint32_t slot)
{
    return trunk_launch(&trunk.steps[slot]);
}

/* At one row, one figure a case, of the instance JUDGED names with its median and the fp16 variant's (the batch
 * takes a matrix's slice count and what it profiles from that line). A trunk matrix: the fastest instance of class E
 * that neither spills, nor reads a warm ring, nor merges where the case has one matrix, over fp16 (the pairs are
 * the harness's, against the baseline). A first figure: weights:bf16 of head, router and mixing, a pass's baseline. */
static void trunk_verdicts(void *, const struct probe_case *probe_case, const struct probe_result *results,
                           double noise_ratio)
{
    const uint32_t matrix = trunk_matrix_of(probe_case, 0), pair = trunk_matrix_of(probe_case, 1);
    const int is_first_figure = matrix >= HEAD;
    const double fp16_us = results[TRUNK_INSTANCE_fp16].step.median;
    uint32_t judged = is_first_figure && matrix < PASS_5120 ? (uint32_t)TRUNK_INSTANCE_bf16_weights
                                                            : (uint32_t)PROBE_BASELINE;
    char figure[TRUNK_FIGURE_BYTES];
    if (probe_case_value(probe_case, TRUNK_FLAG_ROWS) != 1 || results[PROBE_BASELINE].mode == NULL) {
        return;
    }
    for (uint32_t instance = 0; !is_first_figure && instance < trunk_instances_in_run; instance++) {
        const struct trunk_variant *variant = &trunk_variants[instance];
        const int cold_and_apt = instance != PROBE_BASELINE_AGAIN && !variant->same_copy &&
                                 (pair != NONE || !trunk_kernels[variant->kernel].merged);
        if (cold_and_apt && results[instance].contract_class == 'E' && results[instance].spill_bytes <= 0 &&
            results[instance].step.median < results[judged].step.median) {       /* not timed: no number, never less */
            judged = instance;
        }
    }
    printf("JUDGED card=%" PRIu64 " case=%s slices=%" PRIu64 " variant=%s median_us=%s fp16_us=%s\n",
           probe_flag_number("--card-index", 0), probe_case->name, probe_case_value(probe_case, TRUNK_FLAG_SLICES),
           results[judged].variant, probe_number(results[judged].step.median, 1), probe_number(fp16_us, 1));
    PROBE_REQUIRE(snprintf(figure, sizeof(figure), "%s_at_%s", is_first_figure ? "first_figure" : "judged_over_fp16",
                           probe_case->name) > 0, "a figure's name");
    probe_verdict(figure, PROBE_NAN, PROBE_NAN,          /* measured: a bar is across slice counts, the batch's */
                  is_first_figure ? results[judged].ratio : results[judged].step.median / fp16_us, noise_ratio,
                  results[judged].same_sign, results[judged].rounds);
}

static int trunk_configure(void *);
static struct probe_program trunk_program = {
    "trunk_forms", trunk_flags, PROBE_COUNT_OF(trunk_flags), trunk_instances, TRUNK_INSTANCE_COUNT - 1u, NULL,
    trunk_configure, trunk_open_card, trunk_describe_case, trunk_check_case, trunk_prepare_step, trunk_launch_step,
    probe_cuda_mark_start, probe_cuda_elapsed_microseconds, trunk_verdicts, trunk_close_card, NULL};

/* The harness knows break switches only: --break NAME is made one here. The instance that breaks row exactness
 * is the table's last and joins the run with its flag alone. */
static int trunk_configure(void *)
{
    probe_run.breaking |= probe_flag_given(TRUNK_FLAG_BREAK) ? 1u : 0u;
    trunk_instances_in_run += probe_flag_has_word(TRUNK_FLAG_BREAK, "exactness") ? 1u : 0u;
    trunk_program.instance_count = trunk_instances_in_run;
    return 1;
}

int main(int argument_count, char **arguments) { return probe_main(argument_count, arguments, &trunk_program); }
