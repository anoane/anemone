/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * engine_linear_timing_probe: what the ENGINE's linear kernels cost at the model's shapes on the one card it is given
 * (the speed campaign: research/speed/ROUND_1.md, package 1, and its round 2, RESEARCH.md 11.2 C item 6): a kernel
 * instance per slice count, the carried and block-scaled forms, two matrices in one trunk launch, the operands of
 * the quantizer (round 1 timed its engine with this probe as it stood in round 1's commit). The engine's own wait
 * looks at an event every 100 us (source/backend_cuda.c), so nothing can be timed through it: this program loads the
 * engine's kernel modules itself through the driver interface, takes every launch from the engine's own preparation
 * (source/linear.c, compiled in unchanged, with every check it makes), queues it as it was prepared and times with
 * the driver's events, by the protocol and with the lines of probe_harness.h.
 *
 * A step (--part) is a part of a layer, its launches in the engine's order:
 *   experts        quantize_rows (5,120 wide), gate_up, quantize_hidden (pairs, 2,304 wide), down, row_sum
 *   shared_expert  gate_up (one launch of two matrices), hidden, quantize_hidden, down; quantize_rows before them
 *                  is the layer's (the routed experts read the same arrays): it runs for the checks and is no part
 *                  of "all"
 *   attn.wq_a attn.wq_b attn.wkv attn.wo_b engram.wkv attn.indexer.wq_b      quantize_rows, linear
 *   attn.wq_a+attn.wkv  quantize_rows, linear: one launch of the two matrices
 *   attn.wo_a      linear, of rows that are not quantized: 8,192 by 4,096 in one segment (the bridge probe's shape);
 *                  attn.wo_a_groups: as the model has it, eight groups, a row of 32,768 values
 *   mtp.0.main_proj  quantize_rows, linear: the draft's main_proj, 5,120 by 15,360 (dspark_v1/DESIGN.md K8)
 * --native: the PRO in its own forms (class 120 native, backend_seam.h): the linear's modules <file>_native from
 * --modules (the engine's build/kernels/sm_120a) and the default counts of that class.
 * --only all|small|<launch> times the whole step, its launches that are no product, or one launch alone: a fixed
 * part is its kernels timed alone, never a subtraction. A case is one value each of --part, --only, --rows,
 * --overlap (the percent of its six experts a row shares with the row before) and --gap-us (the host sleeps and the
 * card idles that long before every step). Variants: baseline and baseline_again (4 slices), default (the counts
 * linear_default_slice_count gives the speed profile on this card's class: a DEFAULT line per case says
 * which), and slices:<A>x<B> for every pair of --slices (of gate with up, or of the part's matrices) and
 * --slices-down, each a count a kernel is stamped for. A slice count is a contract of its own (linear_records.h):
 * variants are compared in time only. Weights and rows are a generator's
 * bytes (scale codes 115 to 123 over 4-bit codes, 107 to 115 over E4M3 bytes: every sum with a vector's exponent is
 * inside the range linear_records.h states); a step reads units of a ring that the card has not read for four times
 * its L2 or more (RING line).
 *
 * Checked per case before anything is timed, on step 0 with the baseline's slices, every launch validated on the
 * host first (linear.c's checks, the job-table checks linear.h:40-46 leaves to a table's writer, the card's
 * limits). The checks, and the switch that shows each failing:
 *   quantized_rows, quantized_hidden: every code, exponent and operand against tests/linear_reference.c, 0 may
 *       differ (--break-quantizer: the host doubles the last block of every vector it quantizes);
 *   hidden_reference, output_reference: 32 windows of 16 outputs, from the first to the last, of two vectors against
 *       the reference's products of the stored bytes, each stage fed with the card's own previous one, within 2e-3
 *       of the rms (--corrupt: a megabyte of the first checked matrix, the gate where there is one, is overwritten);
 *   row_sums: the routed experts' sums against the reference's, bit for bit (--break-sums: without the last pair);
 *   rows_exact: every row launched again alone, 0 bytes of its outputs may differ (--break-rows: with one slice);
 *   launch_refused (--break-route): weights behind the arena are refused, and nothing is launched.
 * HOST, per case: the host's half of a step, which the card's events do not show: submit_us (until the last launch
 * call returned) and host_us (the step's wall time less the card's time), medians over the steps timed alone, the
 * warm-up's among them. Not shown: what the engine's sleeping wait adds (the driver waits here, and after a gap the
 * host prepares the step before its first event); rows at the corners of the quantizer, which the card tests feed.
 *
 *   engine_linear_timing_probe --modules <engine/build/kernels/sm_80 | sm_120a> [--part ..] [--only ..] [--rows ..]
 *       [--overlap ..] [--gap-us ..] [--slices ..] [--slices-down ..] and the flags of probe_harness.h
 * Measurement tool: not part of the engine.
 */
#include <cuda.h>
#include <stddef.h>

#include "probe_harness.h"
#include "anemone/linear.h"
#include "../../tests/linear_reference.h"

enum {
	HIDDEN_SIZE = 5120,
	INNER_SIZE = 2304,
	EXPERTS_PER_ROW = 6,
	MOST_ROWS = 8,
	MOST_PAIRS = MOST_ROWS * EXPERTS_PER_ROW,
	WIDEST =
		32768, /* the most values of a vector: the outputs of attn.wq_b */
	BLOCK_VALUES = LINEAR_VALUES_PER_BLOCK,
	RING_EXPERTS =
		136, /* 128 leave the down kernel alone, at eight rows, 3.7 L2 of the PRO */
	COLD_L2_SIZES = 4,
	RING_SPARE_UNITS =
		2, /* CONVENTIONS 3.5; the unit being read, and the division's rest */
	MOST_MATRICES = 3,
	DOWN_MATRIX = 2,
	MOST_LAUNCHES = 5,
	TRUNK_LAUNCHES = 2,
	MOST_KEPT = 4,
	MODULE_COUNT = 3,
	INSTANCES = LINEAR_SLICE_INSTANCES,
	SECOND_MATRIX = 1,
	DEFAULT_VARIANT = PROBE_BASELINE_AGAIN + 1,
	BASELINE_SLICES = 4,
	BROKEN_SLICES = 1,
	CHECKED_STEP = 0,
	CHECKED_OUTPUTS = 16,
	CHECKED_WINDOWS = 32,
	CHECKED_VECTORS = 2,
	CHECKED_PER_VALUE = 2, /* a value's code and operand */
	CORRUPTED_BYTES = 1024 * 1024,
	CORRUPTED_BYTE = 0x5A,
	TOGETHER = 0,
	ALONE = 1,
	FETCHES = 2,
	STALE_BYTE =
		0xA5, /* of every array before a checked launch: a number in each of them */
	E2M1_BITS = 4,
	E2M1_WEIGHTS_PER_BYTE = 2,
	E4M3_BITS = 8,
	E4M3_MAGNITUDE = 0x7F,
	E2M1_SCALE_LEAST = 115,
	E4M3_SCALE_LEAST = 107,
	SCALE_SPAN = 9,
	ROUTE_WEIGHT_STEPS = 16,
	PERCENT = 100,
	BFLOAT16_BYTES = 2,
	SUM_BYTES = 4,
	/* a bfloat16: sign, 8 exponent bits of bias 127, 7 fraction bits; a row's values: 2^-6 to 8, both signs */
	BFLOAT16_SIGN = 0x8000,
	BFLOAT16_EXPONENT_SHIFT = 7,
	BFLOAT16_FRACTION_MASK = 0x7F,
	BFLOAT16_DOUBLED = 0x80,
	LEAST_EXPONENT = 121,
	EXPONENT_SPAN = 9,
	MOST_HOST_SAMPLES = 65536,
	HOST_KINDS = 2,
	HOST_HALF = 0,
	SUBMIT = 1,
	MOST_GAP_MICROSECONDS = 999999,
	NANOSECONDS_PER_MICROSECOND = 1000,
	CAPABILITY_SM_80 = 8,
	CAPABILITY_SM_120 = 12,
	MIX_FIRST_SHIFT = 12,
	MIX_SECOND_SHIFT = 25,
	MIX_THIRD_SHIFT = 27
};
#define MIX_MULTIPLIER 0x2545F4914F6CDD1Dull
#define SEED_MULTIPLIER 0x9E3779B97F4A7C15ull    /* odd: no seed is 0, the one state the generator never leaves */
#define MICROSECONDS_PER_SECOND 1.0e6
#define BYTES_PER_MB 1.0e6
#define ACTIVATION_LIMIT 10.0f                   /* the model's swiglu_limit */
#define REFERENCE_LIMIT 2.0e-3                   /* of a bfloat16 vector against its reference, over the rms */
#define ARRAY_BYTES ((uint64_t)MOST_ROWS * WIDEST * BFLOAT16_BYTES)     /* every array: the bytes of the largest */
#define ARRAYS_BYTES (ARRAY_COUNT * ARRAY_BYTES)
#define JOB_TABLE_BYTES \
	((uint64_t)MOST_PAIRS * EXPERT_JOB_BYTES) /* of one slot of a stream */
#define FLAG_ROWS "--rows"
#define FLAG_OVERLAP "--overlap"
#define FLAG_MODULES "--modules"
#define PART_WORDS "experts,shared_expert,attn.wq_a,attn.wq_b,attn.wkv,attn.wo_b,engram.wkv,attn.indexer.wq_b," \
                   "attn.wo_a,attn.wo_a_groups,attn.wq_a+attn.wkv,mtp.0.main_proj"
#define FLAG_NATIVE "--native"
#define DRAFT_FEATURES 15360                     /* main_proj's inputs: three taps of the hidden size */
#define ONLY_WORDS "all,small,quantize_rows,gate_up,quantize_hidden,down,row_sum,linear,hidden"
#define DRIVER(call) driver_require((call), #call, __LINE__)
enum {                                           /* the values of --only, in the order of ONLY_WORDS */
    ONLY_ALL, ONLY_SMALL, ONLY_QUANTIZE_ROWS, ONLY_GATE_UP, ONLY_QUANTIZE_HIDDEN, ONLY_DOWN, ONLY_ROW_SUM, ONLY_LINEAR,
    ONLY_HIDDEN
};
enum { KIND_EXPERTS, KIND_SHARED, KIND_TRUNK, KIND_BFLOAT16, KIND_MERGED, KIND_COUNT };
enum array {                                     /* the arrays of linear_records.h that a part's launches touch */
    ARRAY_VALUES, ARRAY_CODES, ARRAY_EXPONENTS, ARRAY_OPERANDS, ARRAY_GATE, ARRAY_UP, ARRAY_HIDDEN, ARRAY_HIDDEN_CODES,
    ARRAY_HIDDEN_EXPONENTS, ARRAY_HIDDEN_OPERANDS, ARRAY_PAIR_OUTPUT, ARRAY_OUTPUT, ARRAY_COUNT
};
/* A launch's kernel handle is its place in the order of the members of struct linear_kernels_record: the
 * quantizer, gate with up and down of each slice count, the row sum, the trunk's two of each slice count, the
 * hidden step. The wide trunk instances behind them (more than 8 rows, which no case has) stay
 * BACKEND_NO_KERNEL: linear.c refuses a launch of them. */
enum {
    KERNEL_QUANTIZE = 0, KERNEL_GATE_UP = 1, KERNEL_DOWN = KERNEL_GATE_UP + INSTANCES,
    KERNEL_ROW_SUM = KERNEL_DOWN + INSTANCES, KERNEL_TRUNK_QUANTIZED = KERNEL_ROW_SUM + 1,
    KERNEL_TRUNK_BFLOAT16 = KERNEL_TRUNK_QUANTIZED + INSTANCES, KERNEL_TRUNK_HIDDEN = KERNEL_TRUNK_BFLOAT16 + INSTANCES,
    KERNEL_COUNT
};
static struct linear_kernels_record
	kernels; /* member k holds KERNEL_* k: open_card fills it */
#define HANDLE_COUNT (sizeof(struct linear_kernels_record) / sizeof(uint32_t))
_Static_assert(
	offsetof(struct linear_kernels_record, trunk_hidden) ==
			(KERNEL_COUNT - 1) * sizeof(uint32_t) &&
		sizeof(struct linear_kernels_record) % sizeof(uint32_t) == 0,
	"the first handles are the kernels of linear_records.h in their order, one number each");
/* The function of a handle: of a stamped kernel its stem and slice count, a row of stamped_names per count. */
enum { GATE_UP_COLUMN, DOWN_COLUMN, TRUNK_QUANTIZED_COLUMN, TRUNK_BFLOAT16_COLUMN, STAMPED_COLUMNS };
#define LINEAR_SLICE_STAMP(slices)                                             \
	{ KERNEL_EXPERT_GATE_UP_STEM #slices, KERNEL_EXPERT_DOWN_STEM #slices, \
	  KERNEL_TRUNK_LINEAR_QUANTIZED_STEM #slices,                          \
	  KERNEL_TRUNK_LINEAR_BFLOAT16_STEM #slices },
static const char *const stamped_names[INSTANCES][STAMPED_COLUMNS] = {
	LINEAR_EVERY_SLICE_COUNT
};
#undef LINEAR_SLICE_STAMP
static const char *function_name(uint32_t kernel)
{
	return kernel == KERNEL_QUANTIZE ?
		       KERNEL_ACTIVATION_QUANTIZE :
	       kernel < KERNEL_DOWN ?
		       stamped_names[kernel - KERNEL_GATE_UP][GATE_UP_COLUMN] :
	       kernel < KERNEL_ROW_SUM ?
		       stamped_names[kernel - KERNEL_DOWN][DOWN_COLUMN] :
	       kernel == KERNEL_ROW_SUM ?
		       KERNEL_EXPERT_ROW_SUM :
	       kernel < KERNEL_TRUNK_BFLOAT16 ?
		       stamped_names[kernel - KERNEL_TRUNK_QUANTIZED]
				    [TRUNK_QUANTIZED_COLUMN] :
	       kernel < KERNEL_TRUNK_HIDDEN ?
		       stamped_names[kernel - KERNEL_TRUNK_BFLOAT16]
				    [TRUNK_BFLOAT16_COLUMN] :
		       KERNEL_TRUNK_HIDDEN_VALUE;
}
/* The file and the record of a handle (with --native the products' native builds). */
static const char *kernel_file(uint32_t kernel)
{
    const int is_native = probe_flag_given(FLAG_NATIVE);
    return kernel == KERNEL_QUANTIZE ?
		   KERNEL_FILE_ACTIVATION_QUANTIZE :
	   kernel < KERNEL_TRUNK_QUANTIZED ?
		   (is_native ? KERNEL_FILE_EXPERT_LINEAR_NATIVE :
				KERNEL_FILE_EXPERT_LINEAR) :
	   is_native ? KERNEL_FILE_TRUNK_LINEAR_NATIVE :
		       KERNEL_FILE_TRUNK_LINEAR;
}
static size_t record_bytes_of(uint32_t kernel)
{
	return kernel == KERNEL_QUANTIZE ? ACTIVATION_QUANTIZE_ARGUMENT_BYTES :
	       kernel < KERNEL_DOWN	 ? EXPERT_GATE_UP_ARGUMENT_BYTES :
	       kernel < KERNEL_ROW_SUM	 ? EXPERT_DOWN_ARGUMENT_BYTES :
	       kernel == KERNEL_ROW_SUM ?
					 EXPERT_ROW_SUM_ARGUMENT_BYTES :
	       kernel < KERNEL_TRUNK_BFLOAT16 ?
					 TRUNK_LINEAR_QUANTIZED_ARGUMENT_BYTES :
	       kernel < KERNEL_TRUNK_HIDDEN ?
					 TRUNK_LINEAR_BFLOAT16_ARGUMENT_BYTES :
					 TRUNK_HIDDEN_ARGUMENT_BYTES;
}
static int is_gate_up(uint32_t kernel) { return kernel >= KERNEL_GATE_UP && kernel < KERNEL_DOWN; }
/* What the parts of one kind share: the launches of the chain, the first that "all" times, the matrices of a unit of
 * weights, their format, and the value of --only that names each launch. */
static const struct kind {
    uint32_t launch_count, first_own, matrix_count;
    enum weight_format_code format;
    uint32_t launches[MOST_LAUNCHES];
} kinds[KIND_COUNT] = { { 1 + LINEAR_ROUTED_EXPERTS_LAUNCH_COUNT,
			  0,
			  MOST_MATRICES,
			  WEIGHT_FORMAT_E2M1_SCALED,
			  { ONLY_QUANTIZE_ROWS, ONLY_GATE_UP,
			    ONLY_QUANTIZE_HIDDEN, ONLY_DOWN, ONLY_ROW_SUM } },
			{ 1 + LINEAR_SHARED_EXPERT_LAUNCH_COUNT,
			  1,
			  MOST_MATRICES,
			  WEIGHT_FORMAT_E4M3_SCALED,
			  { ONLY_QUANTIZE_ROWS, ONLY_GATE_UP, ONLY_HIDDEN,
			    ONLY_QUANTIZE_HIDDEN, ONLY_DOWN } },
			{ TRUNK_LAUNCHES,
			  0,
			  1,
			  WEIGHT_FORMAT_E4M3_SCALED,
			  { ONLY_QUANTIZE_ROWS, ONLY_LINEAR } },
			{ 1, 0, 1, WEIGHT_FORMAT_E4M3_SCALED, { ONLY_LINEAR } },
			{ TRUNK_LAUNCHES,
			  0,
			  LINEAR_MERGED_MATRICES,
			  WEIGHT_FORMAT_E4M3_SCALED,
			  { ONLY_QUANTIZE_ROWS, ONLY_LINEAR } } };
/* The parts, in the order of PART_WORDS: outputs by inputs of the matrix (of gate and up; down is the other way
 * round; of two matrices in one launch, the first and the second's outputs), and the segments of a row that is not
 * quantized. */
static const struct part { uint32_t kind, output_size, input_size, segment_count, second_output_size; } parts[] = {
    {KIND_EXPERTS, INNER_SIZE, HIDDEN_SIZE, 1, 0}, {KIND_SHARED, INNER_SIZE, HIDDEN_SIZE, 1, 0},
    {KIND_TRUNK, 1280, HIDDEN_SIZE, 1, 0}, {KIND_TRUNK, 32768, 1280, 1, 0}, {KIND_TRUNK, 512, HIDDEN_SIZE, 1, 0},
    {KIND_TRUNK, HIDDEN_SIZE, 8192, 1, 0}, {KIND_TRUNK, 25600, 6144, 1, 0}, {KIND_TRUNK, 4096, 1280, 1, 0},
    {KIND_BFLOAT16, 8192, 4096, 1, 0}, {KIND_BFLOAT16, 8192, 4096, 8, 0}, {KIND_MERGED, 1280, HIDDEN_SIZE, 1, 512},
    {KIND_TRUNK, HIDDEN_SIZE, DRAFT_FEATURES, 1, 0}};
/* What a checked launch leaves and a row launched alone must leave again, per kind: the array, its vectors to a row,
 * the values of a vector (0: the outputs of the part's matrix), the bytes of a value. */
struct kept { uint32_t array, vectors_per_row, values, value_bytes; };
static const struct kept kept_arrays[KIND_COUNT][MOST_KEPT] = {
    {{ARRAY_HIDDEN, EXPERTS_PER_ROW, INNER_SIZE, BFLOAT16_BYTES},
     {ARRAY_PAIR_OUTPUT, EXPERTS_PER_ROW, HIDDEN_SIZE, BFLOAT16_BYTES}, {ARRAY_OUTPUT, 1, HIDDEN_SIZE, SUM_BYTES}},
    {{ARRAY_GATE, 1, INNER_SIZE, BFLOAT16_BYTES}, {ARRAY_UP, 1, INNER_SIZE, BFLOAT16_BYTES},
     {ARRAY_HIDDEN, 1, INNER_SIZE, BFLOAT16_BYTES}, {ARRAY_OUTPUT, 1, HIDDEN_SIZE, BFLOAT16_BYTES}},
    {{ARRAY_OUTPUT, 1, 0, BFLOAT16_BYTES}}, {{ARRAY_OUTPUT, 1, 0, BFLOAT16_BYTES}},
    {{ARRAY_OUTPUT, 1, 0, BFLOAT16_BYTES}, {ARRAY_UP, 1, 512, BFLOAT16_BYTES}}};
static struct probe_flag flags[] = {
	PROBE_COMMON_FLAGS,
	PROBE_CASE_WORDS("--part", PART_WORDS, "experts",
			 "what a step is: a part of a layer"),
	PROBE_CASE_WORDS("--only", ONLY_WORDS, "all",
			 "the launches of the step that are timed"),
	PROBE_CASE_NUMBERS(FLAG_ROWS, 1, MOST_ROWS, "1", "rows of a step"),
	PROBE_CASE_NUMBERS(
		FLAG_OVERLAP, 0, PERCENT, "0",
		"percent of its six experts a row shares with the row before"),
	PROBE_CASE_NUMBERS("--gap-us", 0, MOST_GAP_MICROSECONDS, "0",
			   "the card idles this long before every step"),
	PROBE_NUMBER_LIST("--slices", 1, LINEAR_MAXIMUM_SLICES, "1,2,4,8,16",
			  "of gate with up, or a launch's"),
	PROBE_NUMBER_LIST("--slices-down", 1, LINEAR_MAXIMUM_SLICES,
			  "1,2,4,8,16", "of the down matrix"),
	PROBE_TEXT(
		FLAG_MODULES, "",
		"the folder of the engine's kernel modules for this card: .../sm_80 or .../sm_120a"),
	PROBE_SWITCH(
		"--break-rows",
		"launch a row alone with one slice: rows_exact must report"),
	PROBE_SWITCH(
		"--break-quantizer",
		"the host doubles the last block it quantizes: the quantized checks must report"),
	PROBE_SWITCH(
		"--break-sums",
		"the host leaves a row's last pair out of its sum: row_sums must report"),
	PROBE_SWITCH(
		FLAG_NATIVE,
		"class 120 native: the products' _native modules and that class's default counts")
};
struct variant { uint32_t slices, slices_down; int is_default; char name[PROBE_NUMBER_BYTES]; };
static struct variant variants[PROBE_MOST_INSTANCES];
static struct probe_instance instances[PROBE_MOST_INSTANCES];
static uint16_t host_values[MOST_ROWS * WIDEST];        /* the rows as they were uploaded: [row][values of a row] */
static double host_samples[HOST_KINDS][MOST_HOST_SAMPLES];      /* of the case's steps timed alone: the last of them */
static struct {
    CUdevice device;
    CUcontext context;
    CUstream stream;                             /* as the engine's: every launch and both events of a timed step */
    CUevent started, stopped;
    CUmodule modules[MODULE_COUNT];              /* the engine's three kernel files of the linear */
    CUfunction functions[KERNEL_COUNT];
    CUdeviceptr arena;                           /* one, as the engine has: the ring, the job tables, the arrays */
    uint64_t arena_bytes, l2_bytes, unit_bytes, unit_count, jobs_offset, arrays_offset, sample_count;
    uint32_t arithmetic_class;                   /* of backend_seam.h: what the default slice counts are read for */
    const struct part *part;                     /* what the arena is laid out for; NULL: nothing yet */
    struct matrix_shape_record
	    shapes[MOST_MATRICES]; /* of a unit of weights: its matrices, */
    uint64_t matrix_bytes[MOST_MATRICES], matrix_offsets[MOST_MATRICES];    /* their bytes and where they begin */
    uint8_t *host_unit, *fetched[FETCHES];       /* a unit as it was uploaded; the arrays after a checked launch */
    struct kernel_launch
	    launches[PROBE_STREAM_STEPS]
		    [MOST_LAUNCHES]; /* of the step prepared in each slot */
    uint32_t timed_launches, queued_steps;       /* of the case: bit l is launch l of the chain; steps since the mark */
    double begun_seconds;
} timing;

static void driver_require(CUresult result, const char *call, int line)
{
    const char *error_name = NULL;
    if (result == CUDA_SUCCESS) { return; }
    if (cuGetErrorName(result, &error_name) != CUDA_SUCCESS || error_name == NULL) { error_name = "an unnamed error"; }
    fprintf(stderr, "engine_linear_timing_probe.c:%d: %s failed: %s\n", line, call, error_name);
    exit(EXIT_FAILURE);
}

static uint64_t mix_next(uint64_t *state)       /* xorshift64*: the same bytes on every machine */
{
    *state ^= *state >> MIX_FIRST_SHIFT;
    *state ^= *state << MIX_SECOND_SHIFT;
    *state ^= *state >> MIX_THIRD_SHIFT;
    return *state * MIX_MULTIPLIER;
}

static struct linear_range
array_range(enum array array) /* where an array lies in the arena */
{
	return (struct linear_range){
		timing.arrays_offset + array * ARRAY_BYTES, ARRAY_BYTES
	};
}

static const void *array_of(const uint8_t *fetched, uint32_t array) { return fetched + array * ARRAY_BYTES; }

/* Unit `unit` of the part into host_unit: for each matrix its codes, any bytes but an E4M3 byte that is no number,
 * and behind them its scale codes, the nine from the format's least on. */
static void unit_generate(uint64_t unit)
{
    const struct kind *kind = &kinds[timing.part->kind];
    const int is_e2m1 = kind->format == WEIGHT_FORMAT_E2M1_SCALED;
    for (uint32_t matrix = 0; matrix < kind->matrix_count; matrix++) {
        const uint64_t weights = (uint64_t)timing.shapes[matrix].output_size * timing.shapes[matrix].input_size;
        const uint64_t code_bytes = is_e2m1 ? weights / E2M1_WEIGHTS_PER_BYTE : weights;
        uint64_t state = (unit * MOST_MATRICES + matrix + 1u) * SEED_MULTIPLIER;
        uint8_t *stored = timing.host_unit + timing.matrix_offsets[matrix];
        for (uint64_t byte = 0; byte < timing.matrix_bytes[matrix]; byte += sizeof(uint64_t)) {
            const uint64_t word = mix_next(&state);
            memcpy(stored + byte, &word, sizeof(word));
        }
        for (uint64_t byte = is_e2m1 ? code_bytes : 0; byte < timing.matrix_bytes[matrix]; byte++) {
            if (byte >= code_bytes) {
                stored[byte] = (uint8_t)((is_e2m1 ? E2M1_SCALE_LEAST : E4M3_SCALE_LEAST) + stored[byte] % SCALE_SPAN);
            } else if ((stored[byte] & E4M3_MAGNITUDE) == E4M3_MAGNITUDE) {
                stored[byte] = (uint8_t)(stored[byte] - 1u);
            }
        }
    }
}

static uint32_t route_stride(uint32_t overlap) { return EXPERTS_PER_ROW * (PERCENT - overlap) / PERCENT; }

static float route_weight(uint64_t expert, uint32_t row)
{
    return (float)(1u + (expert + row) % ROUTE_WEIGHT_STEPS) / (float)ROUTE_WEIGHT_STEPS;
}

/* The job table of rows first_row .. first_row + row_count - 1, which fill the launch from its pair 0: row r takes
 * six experts of the ring from the one at first_expert + r * stride on (its slots in that order); a job is an
 * expert with every row routed to it. unit_shift moves every matrix (0, or what --break-route asks). Returns the
 * jobs. */
static uint32_t jobs_build(uint64_t first_expert, uint32_t overlap,
			   uint32_t first_row, uint32_t row_count,
			   uint64_t unit_shift,
			   struct expert_job_record jobs[MOST_PAIRS])
{
    uint32_t job_count = 0;
    memset(jobs, 0, MOST_PAIRS * sizeof(jobs[0]));
    for (uint32_t pair = 0; pair < row_count * EXPERTS_PER_ROW; pair++) {
        const uint32_t row = first_row + pair / EXPERTS_PER_ROW;
        const uint64_t expert = (first_expert + row * route_stride(overlap) + pair % EXPERTS_PER_ROW) % RING_EXPERTS;
        const uint64_t unit_offset = expert * timing.unit_bytes + unit_shift;
        uint32_t job_index = 0;
        while (job_index < job_count && jobs[job_index].gate_offset != unit_offset) { job_index++; }
	struct expert_job_record *job = &jobs[job_index];
	job_count += job_index == job_count ? 1u : 0u;
        job->gate_offset = unit_offset;
        job->up_offset = unit_offset + timing.matrix_offsets[1];
        job->down_offset = unit_offset + timing.matrix_offsets[DOWN_MATRIX];
        job->pair_index[job->row_count] = pair;
        job->route_weight[job->row_count++] = route_weight(expert, row);
    }
    return job_count;
}

/* What linear.h leaves to whoever writes a job table (linear.h:40-46), for every entry: its rows; of an entry in
 * use each matrix on its alignment and wholly inside the ring; every pair below the launch's pairs, in one column. */
static void jobs_validate(int *valid, const struct expert_job_record *jobs,
			  uint32_t job_count, uint32_t pair_count)
{
    const char *kernel = function_name(KERNEL_GATE_UP);
    uint64_t pairs_seen = 0;
    for (uint32_t job_index = 0; job_index < job_count; job_index++) {
	    const struct expert_job_record *job = &jobs[job_index];
	    const uint64_t offsets[MOST_MATRICES] = { job->gate_offset,
						      job->up_offset,
						      job->down_offset };
	    probe_launch_value(valid, kernel, "rows of a job", job->row_count,
			       0, LINEAR_MAXIMUM_ROWS);
	    for (uint32_t matrix = 0;
		 job->row_count != 0 && matrix < MOST_MATRICES; matrix++) {
		    probe_launch_value(valid, kernel,
				       "bytes of a matrix off its alignment",
				       offsets[matrix] %
					       LINEAR_OFFSET_ALIGNMENT_BYTES,
				       0, 0);
		    probe_launch_range(valid, kernel,
				       "bytes of a job's matrix in the ring",
				       offsets[matrix],
				       timing.matrix_bytes[matrix], 1,
				       timing.jobs_offset);
	    }
	    for (uint32_t column = 0;
		 column < job->row_count && column < LINEAR_MAXIMUM_ROWS;
		 column++) {
		    const uint32_t pair = job->pair_index[column];
		    const uint64_t pair_bit = pair < pair_count ? 1ull << pair :
								  0;
		    probe_launch_value(valid, kernel, "a pair of a job", pair,
				       0, pair_count - 1u);
		    probe_launch_value(
			    valid, kernel,
			    "columns a pair stands in before this one",
			    pairs_seen & pair_bit, 0, 0);
		    pairs_seen |= pair_bit;
	    }
    }
}

/* The slices of a variant for matrix `matrix` of the part (gate and up, or the matrices of a trunk launch: the
 * first count; down: the second): its own, or of default those the engine's table gives the speed profile on this
 * card's class. */
static uint32_t slices_of(const struct variant *variant, uint32_t matrix)
{
	const enum linear_family_code family =
		timing.part->kind == KIND_EXPERTS ?
			(matrix == DOWN_MATRIX ? LINEAR_FAMILY_EXPERT_DOWN :
						 LINEAR_FAMILY_EXPERT_GATE_UP) :
		timing.part->kind == KIND_BFLOAT16 ?
			LINEAR_FAMILY_TRUNK_BFLOAT16 :
			LINEAR_FAMILY_TRUNK_QUANTIZED;
	uint32_t slices = matrix == DOWN_MATRIX ? variant->slices_down :
						  variant->slices;
	if (variant->is_default) {
		PROBE_REQUIRE(
			linear_default_slice_count(
				LINEAR_PROFILE_SPEED, LINEAR_ORDER_FIXED,
				family, &timing.shapes[matrix],
				timing.arithmetic_class, &slices) == STATUS_OK,
			"the engine's table has no default slice count for a matrix of the part");
	}
    return slices;
}

static struct linear_matrix unit_matrix(uint64_t unit_offset, uint32_t matrix,
					uint32_t slice_count)
{
	return (struct linear_matrix){ { unit_offset +
						 timing.matrix_offsets[matrix],
					 timing.matrix_bytes[matrix] },
				       timing.shapes[matrix],
				       kinds[timing.part->kind].format,
				       slice_count,
				       LINEAR_WIDE_GROUP_BLOCKS,
				       0 };
}

/* The launches of step `step` of the part for rows first_row .. first_row + row_count - 1, into launches; the job
 * table of the routed experts into the slot's place in the arena. Every launch is the engine's (linear.c refuses,
 * with one error line, what breaks a rule of linear_records.h or leaves the arena) and is held against the card's
 * limits here; the job table is checked here. Returns the launches, or 0: refused, nothing may be queued. */
static uint32_t step_prepare(uint64_t step, uint32_t slot, uint32_t first_row,
			     uint32_t row_count, uint32_t overlap,
			     const struct variant *variant, uint64_t unit_shift,
			     struct kernel_launch *launches)
{
    const struct part *part = timing.part;
    const struct kind *kind = &kinds[part->kind];
    const uint64_t unit_offset = step % timing.unit_count * timing.unit_bytes + unit_shift;
    const uint64_t rows_before = (uint64_t)first_row * part->input_size * part->segment_count * BFLOAT16_BYTES;
    const struct linear_range values = { array_range(ARRAY_VALUES).byte_offset +
						 rows_before,
					 ARRAY_BYTES - rows_before };
    const struct linear_range codes = array_range(ARRAY_CODES),
			      exponents = array_range(ARRAY_EXPONENTS);
    const struct linear_range operands = array_range(ARRAY_OPERANDS);
    const struct linear_range hidden = array_range(ARRAY_HIDDEN),
			      output = array_range(ARRAY_OUTPUT);
    const struct linear_range hidden_codes = array_range(ARRAY_HIDDEN_CODES);
    const struct linear_range hidden_exponents =
	    array_range(ARRAY_HIDDEN_EXPONENTS);
    const struct linear_range hidden_operands =
	    array_range(ARRAY_HIDDEN_OPERANDS);
    const struct linear_matrix gate =
	    unit_matrix(unit_offset, 0, slices_of(variant, 0));
    const uint64_t arena_bytes = timing.arena_bytes;
    struct kernel_launch *own =
	    launches; /* behind the rows' quantization, where the part has one */
    result_status status = STATUS_OK;
    int valid = 1;
    if (part->kind != KIND_BFLOAT16) {
	    const struct linear_quantize_description rows = {
		    values,	      codes,	exponents, operands,
		    part->input_size, row_count
	    };
	    status = linear_quantize_prepare(&rows, arena_bytes, &kernels,
					     own++);
    }
    if (status != STATUS_OK) {
	    valid = 0;
    } else if (part->kind == KIND_EXPERTS) {
	    struct expert_job_record jobs[MOST_PAIRS];
	    const uint64_t first_expert =
		    step * (EXPERTS_PER_ROW +
			    (row_count - 1u) * route_stride(overlap));
	    const uint32_t job_count = jobs_build(first_expert, overlap,
						  first_row, row_count,
						  unit_shift, jobs);
	    const struct linear_routed_experts_description experts = {
		    .jobs = { timing.jobs_offset + slot * JOB_TABLE_BYTES,
			      linear_jobs_byte_count(job_count) },
		    .expert_weights = { 0, timing.jobs_offset },
		    .codes = codes,
		    .exponents = exponents,
		    .operands = operands,
		    .hidden = hidden,
		    .hidden_codes = hidden_codes,
		    .hidden_exponents = hidden_exponents,
		    .hidden_operands = hidden_operands,
		    .pair_output = array_range(ARRAY_PAIR_OUTPUT),
		    .row_output = output,
		    .gate_up_shape = timing.shapes[0],
		    .down_shape = timing.shapes[DOWN_MATRIX],
		    .weight_format = kind->format,
		    .job_count = job_count,
		    .row_count = row_count,
		    .pairs_per_row = EXPERTS_PER_ROW,
		    .gate_up_slice_count = slices_of(variant, 0),
		    .down_slice_count = slices_of(variant, DOWN_MATRIX),
		    .activation_limit = ACTIVATION_LIMIT
	    };
	    jobs_validate(&valid, jobs, job_count, row_count * EXPERTS_PER_ROW);
	    status = linear_routed_experts_prepare(&experts, arena_bytes,
						   &kernels, own);
	    if (valid && status == STATUS_OK) {
		    DRIVER(cuMemcpyHtoD(timing.arena + experts.jobs.byte_offset,
					jobs, experts.jobs.byte_count));
	    }
    } else if (part->kind == KIND_SHARED) {
	    const struct linear_shared_expert_description shared = {
		    .gate = gate,
		    .up = unit_matrix(unit_offset, 1, slices_of(variant, 1)),
		    .down = unit_matrix(unit_offset, DOWN_MATRIX,
					slices_of(variant, DOWN_MATRIX)),
		    .codes = codes,
		    .exponents = exponents,
		    .operands = operands,
		    .gate_output = array_range(ARRAY_GATE),
		    .up_output = array_range(ARRAY_UP),
		    .hidden = hidden,
		    .hidden_codes = hidden_codes,
		    .hidden_exponents = hidden_exponents,
		    .hidden_operands = hidden_operands,
		    .output = output,
		    .row_count = row_count,
		    .activation_limit = ACTIVATION_LIMIT
	    };
	    status = linear_shared_expert_prepare(&shared, arena_bytes,
						  &kernels, own);
    } else if (part->kind !=
	       KIND_BFLOAT16) { /* one matrix, or two: the second's outputs in ARRAY_UP */
	    const struct linear_trunk_quantized_description trunk = {
		    .matrices = { gate,
				  kind->matrix_count > 1 ?
					  unit_matrix(
						  unit_offset, SECOND_MATRIX,
						  slices_of(variant,
							    SECOND_MATRIX)) :
					  gate },
		    .codes = codes,
		    .exponents = exponents,
		    .operands = operands,
		    .outputs = { output, array_range(ARRAY_UP) },
		    .matrix_count = kind->matrix_count,
		    .row_count = row_count
	    };
	    status = linear_trunk_quantized_prepare(&trunk, arena_bytes,
						    &kernels, own);
    } else {
	    const struct linear_trunk_bfloat16_description rows = {
		    gate, values, output, row_count, part->segment_count
	    };
	    status = linear_trunk_bfloat16_prepare(&rows, arena_bytes, &kernels,
						   own);
    }
    valid = valid && status == STATUS_OK;
    for (uint32_t launch = 0; valid && launch < kind->launch_count; launch++) {
	    const struct kernel_launch *prepared = &launches[launch];
	    probe_launch_geometry(&valid, function_name(prepared->kernel), NULL,
				  prepared->grid[0], prepared->grid[1],
				  prepared->grid[LAUNCH_DIMENSIONS - 1],
				  prepared->block[0],
				  prepared->dynamic_shared_byte_count);
    }
    return valid ? kind->launch_count : 0;
}

/* Queues the launches of a prepared step that the bits of `wanted` name, in their order; returns their number. */
static uint32_t launches_queue(struct kernel_launch *launches, uint32_t wanted)
{
    uint32_t queued = 0;
    for (uint32_t launch = 0; launch < MOST_LAUNCHES; launch++) {
	    struct kernel_launch *prepared = &launches[launch];
	    void *parameters[] = {
		    prepared->argument, &timing.arena
	    }; /* every kernel: its record, the arena */
	    if (((wanted >> launch) & 1u) == 0) {
		    continue;
	    }
	    DRIVER(cuLaunchKernel(timing.functions[prepared->kernel],
				  prepared->grid[0], prepared->grid[1],
				  prepared->grid[LAUNCH_DIMENSIONS - 1],
				  prepared->block[0], prepared->block[1],
				  prepared->block[LAUNCH_DIMENSIONS - 1],
				  prepared->dynamic_shared_byte_count,
				  timing.stream, parameters, NULL));
	    queued++;
    }
    return queued;
}

static uint64_t card_attribute(CUdevice_attribute attribute)
{
    int value = 0;
    DRIVER(cuDeviceGetAttribute(&value, attribute, timing.device));
    PROBE_REQUIRE(value > 0, "the card states an attribute that is not above zero");
    return (uint64_t)value;
}

/* The engine's build of one kernel file, as make -C engine kernels left it: read whole and loaded. */
static CUmodule module_load(const char *file)
{
    char path[PROBE_PATH_BYTES];
    CUmodule module = NULL;
    const int written = snprintf(path, sizeof(path), "%s/%s.cubin", probe_flag_text(FLAG_MODULES), file);
    FILE *stream = written < 0 || (size_t)written >= sizeof(path) ? NULL : fopen(path, "rb");
    const long byte_count = stream != NULL && fseek(stream, 0, SEEK_END) == 0 ? ftell(stream) : -1;
    char *image = byte_count > 0 ? calloc((size_t)byte_count + 1u, 1) : NULL;
    if (image == NULL || fseek(stream, 0, SEEK_SET) != 0
        || fread(image, 1, (size_t)byte_count, stream) != (size_t)byte_count || fclose(stream) != 0) {
        fprintf(stderr, "%s cannot be read: %s is the folder make -C engine kernels fills\n", path, FLAG_MODULES);
        exit(EXIT_FAILURE);
    }
    DRIVER(cuModuleLoadData(&module, image));
    printf("MODULE card=%s file=%s bytes=%ld\n", probe_flag_text("--card-index"), path, byte_count);
    free(image);
    return module;
}

/* The one card this process sees, of the architecture the modules are for; context and stream as the engine's;
 * the rows of every part: values of both signs from 2^-6 to 8, the same in every run. */
static int open_card(void *context, struct probe_card_facts *facts)
{
    static char name[PROBE_NAME_BYTES];
    int card_count = 0, major = 0, minor = 0;
    (void)context;
    DRIVER(cuInit(0));
    DRIVER(cuDeviceGetCount(&card_count));
    if (card_count == 1) {
        DRIVER(cuDeviceGet(&timing.device, 0));
        DRIVER(cuDeviceGetName(name, PROBE_NAME_BYTES - 1, timing.device));
        DRIVER(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, timing.device));
        DRIVER(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, timing.device));
    }
    facts->name = name;
    facts->architecture = major == CAPABILITY_SM_120 ? "sm_120a" : "sm_80";
    timing.arithmetic_class =
	    major != CAPABILITY_SM_120	  ? ARITHMETIC_CLASS_SM_80 :
	    probe_flag_given(FLAG_NATIVE) ? ARITHMETIC_CLASS_SM_120_NATIVE :
					    ARITHMETIC_CLASS_SM_120;
    if (card_count != 1 || minor != 0 || (major != CAPABILITY_SM_80 && major != CAPABILITY_SM_120)
        || (probe_flag_given(FLAG_NATIVE) && major != CAPABILITY_SM_120)
        || strstr(probe_flag_text(FLAG_MODULES), facts->architecture) == NULL) {
        fprintf(stderr, "this process sees %d cards, the first of capability %d.%d, with the modules of \"%s\": a "
                        "probe is given one card (CUDA_VISIBLE_DEVICES=<index>) of capability 8.0 or 12.0, and %s "
                        "<the folder of that architecture: sm_80 or sm_120a>; %s on a card of 12.0 alone\n",
                card_count, major, minor, probe_flag_text(FLAG_MODULES), FLAG_MODULES, FLAG_NATIVE);
        return 0;
    }
    facts->sm_count = (int)card_attribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT);
    facts->l2_bytes = timing.l2_bytes = card_attribute(CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE);
    facts->block_threads_most = (uint32_t)card_attribute(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK);
    facts->shared_bytes_most = card_attribute(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK);
    facts->grid_most[0] = (uint32_t)card_attribute(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X);
    facts->grid_most[1] = (uint32_t)card_attribute(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y);
    facts->grid_most[PROBE_GRID_AXES - 1] = (uint32_t)card_attribute(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z);
    DRIVER(cuDevicePrimaryCtxRetain(&timing.context, timing.device));
    DRIVER(cuCtxSetCurrent(timing.context));
    DRIVER(cuStreamCreate(&timing.stream, CU_STREAM_NON_BLOCKING));
    DRIVER(cuEventCreate(&timing.started, CU_EVENT_DEFAULT));
    DRIVER(cuEventCreate(&timing.stopped, CU_EVENT_DEFAULT));
    for (uint32_t kernel = 0; kernel < HANDLE_COUNT; kernel++) {   /* the handles: the kernels in their order */
	    ((uint32_t *)(void *)&kernels)[kernel] =
		    kernel < KERNEL_COUNT ? kernel : BACKEND_NO_KERNEL;
    }
    for (uint32_t kernel = 0, module = 0; kernel < KERNEL_COUNT; kernel++) {
        size_t record_offset = 0, record_bytes = 0;
        if (kernel != 0 && strcmp(kernel_file(kernel), kernel_file(kernel - 1u)) != 0) { module++; }
        if (timing.modules[module] == NULL) { timing.modules[module] = module_load(kernel_file(kernel)); }
        DRIVER(cuModuleGetFunction(&timing.functions[kernel], timing.modules[module], function_name(kernel)));
        DRIVER(cuFuncLoad(timing.functions[kernel]));       /* as the engine: loaded before its record is asked */
        DRIVER(cuFuncGetParamInfo(timing.functions[kernel], 0, &record_offset, &record_bytes));
        PROBE_REQUIRE(record_offset == 0 && record_bytes == record_bytes_of(kernel),
                      "a module's kernel takes another record than linear_records.h: not this engine's build");
    }
    for (uint64_t value = 0, state = SEED_MULTIPLIER; value < PROBE_COUNT_OF(host_values); value++) {
        const uint32_t word = (uint32_t)mix_next(&state), exponent = (word >> BFLOAT16_EXPONENT_SHIFT) % EXPONENT_SPAN;
        host_values[value] = (uint16_t)((word & (BFLOAT16_SIGN | BFLOAT16_FRACTION_MASK)) |
                                        ((LEAST_EXPONENT + exponent) << BFLOAT16_EXPONENT_SHIFT));
    }
    timing.fetched[TOGETHER] = malloc(ARRAYS_BYTES);
    timing.fetched[ALONE] = malloc(ARRAYS_BYTES);
    PROBE_REQUIRE(timing.fetched[TOGETHER] != NULL && timing.fetched[ALONE] != NULL, "out of host memory");
    return 1;
}

/* The arena for one part: its ring of units of weights (the experts each of its own bytes, a trunk ring of copies
 * of one unit), a job table for every slot of a stream, the arrays; the rows uploaded, every other array stale. */
static void part_set_up(const struct part *part, const char *case_name)
{
    const struct kind *kind = &kinds[part->kind];
    const struct weight_format_operations *format =
	    weight_format_operations_of(kind->format);
    const uint32_t bits =
	    kind->format == WEIGHT_FORMAT_E2M1_SCALED ? E2M1_BITS : E4M3_BITS;
    const uint64_t units_of_a_step = part->kind == KIND_EXPERTS ? MOST_PAIRS : 1;
    timing.part = part;
    timing.unit_bytes = 0;
    for (uint32_t matrix = 0; matrix < kind->matrix_count; matrix++) {
        const int is_down = matrix == DOWN_MATRIX;
        const uint32_t output_size = part->kind == KIND_MERGED && matrix == SECOND_MATRIX ? part->second_output_size
                                                                                            : part->output_size;
	timing.shapes[matrix] = (struct matrix_shape_record){
		is_down ? part->input_size : output_size,
		is_down ? part->output_size : part->input_size, bits, 0
	};
	const result_status status = format->stored_byte_count(
		&timing.shapes[matrix], &timing.matrix_bytes[matrix]);
	PROBE_REQUIRE(
		status == STATUS_OK &&
			timing.matrix_bytes[matrix] %
					LINEAR_OFFSET_ALIGNMENT_BYTES ==
				0,
		"a matrix its format does not store, or whose bytes are no multiple of the alignment");
	timing.matrix_offsets[matrix] = timing.unit_bytes;
        timing.unit_bytes += timing.matrix_bytes[matrix];
    }
    timing.unit_count = part->kind == KIND_EXPERTS ? RING_EXPERTS
                        : COLD_L2_SIZES * timing.l2_bytes / timing.matrix_bytes[0] + RING_SPARE_UNITS;
    const uint64_t cold_bytes = (timing.unit_count - units_of_a_step) * timing.matrix_bytes[0];
    const double cold_l2 = (double)cold_bytes / (double)timing.l2_bytes;
    PROBE_REQUIRE(cold_l2 >= (double)COLD_L2_SIZES, "the ring is too small for this card's L2 (CONVENTIONS 3.5)");
    timing.jobs_offset = timing.unit_count * timing.unit_bytes;
    timing.arrays_offset = timing.jobs_offset + PROBE_STREAM_STEPS * JOB_TABLE_BYTES;
    timing.arena_bytes = timing.arrays_offset + ARRAYS_BYTES;
    if (timing.arena != 0) { DRIVER(cuMemFree(timing.arena)); }
    free(timing.host_unit);
    timing.host_unit = malloc(timing.unit_bytes);
    PROBE_REQUIRE(timing.host_unit != NULL, "out of host memory for a unit of weights");
    DRIVER(cuMemAlloc(&timing.arena, timing.arena_bytes));
    for (uint64_t unit = 0; unit < timing.unit_count; unit++) {
        if (unit == 0 || part->kind == KIND_EXPERTS) { unit_generate(unit); }
        DRIVER(cuMemcpyHtoD(timing.arena + unit * timing.unit_bytes, timing.host_unit, timing.unit_bytes));
    }
    DRIVER(cuMemsetD8(timing.arena + timing.jobs_offset, STALE_BYTE, timing.arena_bytes - timing.jobs_offset));
    DRIVER(cuMemcpyHtoD(timing.arena + timing.arrays_offset, host_values, sizeof(host_values)));
    printf("RING card=%s case=%s units=%" PRIu64 " unit_mb=%.1f ring_mib=%.0f cold_l2=%.1f\n",
           probe_flag_text("--card-index"), case_name, timing.unit_count, (double)timing.unit_bytes / BYTES_PER_MB,
           (double)timing.jobs_offset / PROBE_BYTES_PER_MIB, cold_l2);
}

/* A case: its part's arena where another part's stands, the launches --only names, and what its RESULT lines
 * say: part= is the case, mb= the weight bytes its timed launches read. */
static void describe_case(void *context, const struct probe_case *probe_case, struct probe_case_facts *facts)
{
    const struct part *part = &parts[probe_case_value(probe_case, "--part")];       /* a word's value: its index */
    const uint64_t only = probe_case_value(probe_case, "--only");
    const struct kind *kind = &kinds[part->kind];
    const uint32_t rows = (uint32_t)probe_case_value(probe_case, FLAG_ROWS);
    const uint32_t overlap = (uint32_t)probe_case_value(probe_case, FLAG_OVERLAP);
    const int is_experts = part->kind == KIND_EXPERTS;
    uint64_t weight_bytes = 0;
    (void)context;
    if (part != timing.part) { part_set_up(part, probe_case->name); }
    facts->format = "native";
    facts->part = probe_case->name;
    facts->rows = rows;
    facts->experts = is_experts ? (int64_t)(EXPERTS_PER_ROW + (rows - 1u) * route_stride(overlap)) : -1;
    facts->overlap = is_experts ? (int64_t)overlap : -1;
    timing.sample_count = timing.timed_launches = 0;
    if (step_prepare(CHECKED_STEP, 0, 0, rows, overlap, &variants[PROBE_BASELINE], 0, timing.launches[0]) == 0) {
        return;                                  /* refused, and said why: prepare_step fails the run */
    }
    for (uint32_t launch = 0; launch < kind->launch_count; launch++) {
	    const struct kernel_launch *prepared = &timing.launches[0][launch];
	    const int is_small = prepared->dynamic_shared_byte_count == 0,
		      is_own = launch >= kind->first_own;
	    const uint64_t matrices =
		    is_gate_up(prepared->kernel) ||
				    (part->kind == KIND_SHARED &&
				     launch == 1u) ?
			    EXPERT_GATE_UP_MATRICES :
			    1;
	    if (only == ONLY_ALL   ? !is_own :
		only == ONLY_SMALL ? !(is_own && is_small) :
				     only != kind->launches[launch]) {
		    continue;
	    }
        timing.timed_launches |= 1u << launch;
        weight_bytes += is_small ? 0
                        : part->kind == KIND_MERGED ? timing.matrix_bytes[0] + timing.matrix_bytes[SECOND_MATRIX]
                                                    : timing.matrix_bytes[0] * prepared->grid[1] * matrices;
    }
    facts->megabytes = (double)weight_bytes / BYTES_PER_MB;
    printf("DEFAULT card=%s case=%s class=%u slices=%u slices_down=%u\n", probe_flag_text("--card-index"),
           probe_case->name, timing.arithmetic_class, slices_of(&variants[DEFAULT_VARIANT], 0),
           kind->matrix_count == MOST_MATRICES ? slices_of(&variants[DEFAULT_VARIANT], DOWN_MATRIX) : 0);
}

/* The codes, exponents and operands the card wrote for vector_count vectors of `width` values against the reference
 * quantizer of the same values (a code where the reference puts its value, operands in their order): the values that
 * differ. doubled: the host doubles the last block of every vector first (--break-quantizer). */
static uint64_t quantizer_differing(const uint16_t *values, const uint8_t *codes, const uint8_t *exponents,
                                    const uint16_t *operands, uint32_t vector_count, uint32_t width, int doubled)
{
    static uint16_t fed[WIDEST], wanted_operands[WIDEST];
    static uint8_t wanted[WIDEST], wanted_exponents[WIDEST / BLOCK_VALUES];
    const uint32_t blocks = width / BLOCK_VALUES;
    uint64_t differing = 0;
    for (uint32_t vector = 0; vector < vector_count; vector++) {
        memcpy(fed, values + (size_t)vector * width, width * sizeof(fed[0]));
        for (uint32_t value = width - BLOCK_VALUES; doubled && value < width; value++) {
            fed[value] = (uint16_t)(fed[value] + BFLOAT16_DOUBLED);
        }
	PROBE_REQUIRE(test_linear_quantize(fed, width, wanted,
					   wanted_exponents) == STATUS_OK &&
			      test_linear_operands_lay_out(
				      wanted, wanted_exponents, width,
				      wanted_operands) == STATUS_OK,
		      "the reference quantizer refused a vector");
	for (uint32_t value = 0; value < width; value++) {
            differing += wanted[value] != codes[(size_t)vector * width + test_linear_code_place(value)] ? 1u : 0u;
            differing += wanted_operands[value] != operands[(size_t)vector * width + value] ? 1u : 0u;
        }
        for (uint32_t block = 0; block < blocks; block++) {
            differing += wanted_exponents[block] != exponents[(size_t)vector * blocks + block] ? 1u : 0u;
        }
    }
    return differing;
}

/* The reference's products of outputs first_output .. first_output + CHECKED_OUTPUTS - 1 of matrix `matrix` of the
 * unit in host_unit, for one vector: its codes as the card holds them, in lane order, with its exponents; or, codes
 * NULL, the bfloat16 values of a row that is not quantized. */
static void reference_products(uint32_t matrix, uint32_t first_output, const uint8_t *codes, const uint8_t *exponents,
                               const uint16_t *values, struct test_linear_product products[CHECKED_OUTPUTS])
{
    static double fed[WIDEST];
    static uint8_t value_codes[WIDEST];
    const struct matrix_shape_record *shape = &timing.shapes[matrix];
    const uint32_t segments = codes != NULL ? 1 : timing.part->segment_count, width = shape->input_size * segments;
    for (uint32_t value = 0; codes != NULL && value < width; value++) {
        value_codes[value] = codes[test_linear_code_place(value)];
    }
    result_status status =
	    codes != NULL ?
		    test_linear_fed_values_of_codes(value_codes, width, fed) :
		    test_linear_fed_values_of_bfloat16(values, width, fed);
    if (status == STATUS_OK) {
	    status = test_linear_products(
		    kinds[timing.part->kind].format, shape,
		    timing.host_unit + timing.matrix_offsets[matrix],
		    timing.matrix_bytes[matrix], first_output, CHECKED_OUTPUTS,
		    fed, codes != NULL ? exponents : NULL, 1, segments,
		    products);
    }
    PROBE_REQUIRE(status == STATUS_OK, "the reference refused a product");
}

static double rounded_once(double value) { return test_linear_bfloat16_value(test_linear_bfloat16_nearest(value)); }

struct rms_error { double error, square; };          /* of the card's values against the reference's, both summed */
static void error_add(struct rms_error *sum, uint16_t card, double wanted)
{
    const double difference = test_linear_bfloat16_value(card) - rounded_once(wanted);
    sum->error += difference * difference;
    sum->square += rounded_once(wanted) * rounded_once(wanted);
}

/* What a checked launch of `rows` rows left (card: the arrays as fetched) against the host reference, every stage
 * fed with the card's own previous stage. The products: windows of outputs spread from the first to the last, of two
 * vectors (with experts the first row at its last expert and the last row at its first; else the first and last). */
static void check_reference(const char *name, uint32_t rows, uint32_t overlap, const uint8_t *card)
{
    static struct test_linear_product gate[CHECKED_OUTPUTS], up[CHECKED_OUTPUTS], product[CHECKED_OUTPUTS];
    static struct test_linear_product second[CHECKED_OUTPUTS];
    const struct part *part = timing.part;
    const int is_experts = part->kind == KIND_EXPERTS, has_hidden = is_experts || part->kind == KIND_SHARED;
    const int corrupt = probe_flag_given(PROBE_FLAG_CORRUPT), doubled = probe_flag_given("--break-quantizer");
    const int sums_broken = probe_flag_given("--break-sums");
    const uint32_t input = part->input_size, blocks = input / BLOCK_VALUES, inner_blocks = INNER_SIZE / BLOCK_VALUES;
    const uint32_t vectors = rows * (is_experts ? EXPERTS_PER_ROW : 1);
    const uint32_t final_size = has_hidden ? HIDDEN_SIZE : part->output_size;      /* of the last product */
    const uint8_t *codes = array_of(card, ARRAY_CODES), *exponents = array_of(card, ARRAY_EXPONENTS);
    const uint8_t *hidden_codes = array_of(card, ARRAY_HIDDEN_CODES);
    const uint8_t *hidden_exponents = array_of(card, ARRAY_HIDDEN_EXPONENTS);
    const uint16_t *hidden = array_of(card, ARRAY_HIDDEN), *pair_outputs = array_of(card, ARRAY_PAIR_OUTPUT);
    const uint16_t *final = is_experts ? pair_outputs : array_of(card, ARRAY_OUTPUT);
    const uint16_t *second_outputs = array_of(card, ARRAY_UP);      /* of two matrices in one launch: the second's */
    const float *sums = array_of(card, ARRAY_OUTPUT);
    const char *variant = variants[PROBE_BASELINE].name;
    struct rms_error hidden_error = {0.0, 0.0}, output_error = {0.0, 0.0};
    uint64_t sums_differing = 0;
    for (uint32_t checked = 0; checked < CHECKED_VECTORS * CHECKED_WINDOWS; checked++) {
        const uint32_t window = checked % CHECKED_WINDOWS, last_row = checked / CHECKED_WINDOWS;
        const uint32_t pair = last_row != 0 ? (rows - 1u) * EXPERTS_PER_ROW : EXPERTS_PER_ROW - 1u;
        const uint32_t row = is_experts ? pair / EXPERTS_PER_ROW : last_row * (rows - 1u);
        const uint32_t vector = is_experts ? pair : row;
        const uint32_t final_first = window * (final_size - CHECKED_OUTPUTS) / (CHECKED_WINDOWS - 1u);
        const uint32_t first = window * (part->output_size - CHECKED_OUTPUTS) / (CHECKED_WINDOWS - 1u);
        const uint32_t second_first = part->kind != KIND_MERGED ? 0
                                      : window * (part->second_output_size - CHECKED_OUTPUTS) / (CHECKED_WINDOWS - 1u);
        const uint64_t expert = (row * route_stride(overlap) + pair % EXPERTS_PER_ROW) % RING_EXPERTS;
        const double weight = is_experts ? (double)route_weight(expert, row) : 1.0;
        const uint8_t *row_codes = codes + (size_t)row * input, *row_exponents = exponents + (size_t)row * blocks;
        if (is_experts && window == 0) { unit_generate(expert); }
        if (has_hidden) {
            reference_products(0, first, row_codes, row_exponents, NULL, gate);
            reference_products(1, first, row_codes, row_exponents, NULL, up);
            reference_products(DOWN_MATRIX, final_first, hidden_codes + (size_t)vector * INNER_SIZE,
                               hidden_exponents + (size_t)vector * inner_blocks, NULL, product);
        } else {
            reference_products(0, first, part->kind != KIND_BFLOAT16 ? row_codes : NULL, row_exponents,
                               host_values + (size_t)row * input * part->segment_count, product);
        }
        if (part->kind == KIND_MERGED) {
            reference_products(SECOND_MATRIX, second_first, row_codes, row_exponents, NULL, second);
        }
        for (uint32_t output = 0; output < CHECKED_OUTPUTS; output++) {
            if (has_hidden) {
                error_add(&hidden_error, hidden[(size_t)vector * INNER_SIZE + first + output],
                          test_linear_hidden_value(rounded_once(gate[output].sum), rounded_once(up[output].sum),
                                                   (double)ACTIVATION_LIMIT, weight));
            }
            error_add(&output_error, final[(size_t)vector * final_size + final_first + output], product[output].sum);
            if (part->kind == KIND_MERGED) {
                error_add(&output_error, second_outputs[(size_t)row * part->second_output_size + second_first + output],
                          second[output].sum);
            }
        }
    }
    if (has_hidden) {                            /* --corrupt overwrote the head of the first checked gate matrix */
        probe_check_error(name, variant, "hidden_reference", sqrt(hidden_error.error / hidden_error.square),
                          REFERENCE_LIMIT, corrupt);
        probe_check_count(name, variant, "quantized_hidden",
                          quantizer_differing(hidden, hidden_codes, hidden_exponents,
                                              array_of(card, ARRAY_HIDDEN_OPERANDS), vectors, INNER_SIZE, doubled),
                          (uint64_t)vectors * (CHECKED_PER_VALUE * INNER_SIZE + inner_blocks), doubled);
    }
    probe_check_error(name, variant, "output_reference", sqrt(output_error.error / output_error.square),
                      REFERENCE_LIMIT, corrupt && !has_hidden);
    if (part->kind != KIND_BFLOAT16) {
        probe_check_count(name, variant, "quantized_rows",
                          quantizer_differing(host_values, codes, exponents, array_of(card, ARRAY_OPERANDS), rows,
                                              input, doubled),
                          (uint64_t)rows * (CHECKED_PER_VALUE * input + blocks), doubled);
    }
    for (uint32_t value = 0; is_experts && value < rows * HIDDEN_SIZE; value++) {
        const uint16_t *row_pairs = pair_outputs + (size_t)(value / HIDDEN_SIZE) * EXPERTS_PER_ROW * HIDDEN_SIZE;
        const float wanted = test_linear_row_sum(row_pairs, EXPERTS_PER_ROW - (sums_broken ? 1u : 0u), HIDDEN_SIZE,
                                                 value % HIDDEN_SIZE);
        sums_differing += memcmp(&wanted, &sums[value], sizeof(wanted)) != 0 ? 1u : 0u;
    }
    if (is_experts) {
        probe_check_count(name, variant, "row_sums", sums_differing, (uint64_t)rows * HIDDEN_SIZE, sums_broken);
    }
}

/* Step 0 for rows first_row .. first_row + row_count - 1 with every launch of the chain, the arrays it writes made
 * stale first (a byte no launch wrote cannot pass for a result); then all arrays fetched. */
static void checked_run(uint32_t first_row, uint32_t row_count, uint32_t overlap, const struct variant *variant,
                        uint8_t *fetched)
{
	struct kernel_launch launches[MOST_LAUNCHES];
	const uint32_t launch_count = step_prepare(CHECKED_STEP, 0, first_row,
						   row_count, overlap, variant,
						   0, launches);
	PROBE_REQUIRE(launch_count != 0, "the checked step was refused");
	DRIVER(cuMemsetD8(timing.arena + timing.arrays_offset + ARRAY_BYTES,
			  STALE_BYTE, ARRAYS_BYTES - ARRAY_BYTES));
	DRIVER(cuCtxSynchronize()); /* fills and uploads are not the stream's: a launch waits for none */
	(void)launches_queue(launches, (1u << launch_count) - 1u);
	DRIVER(cuStreamSynchronize(timing.stream));
	DRIVER(cuMemcpyDtoH(fetched, timing.arena + timing.arrays_offset,
			    ARRAYS_BYTES));
}

static void check_case(void *context, const struct probe_case *probe_case)
{
    const struct part *part = timing.part;
    const uint32_t rows = (uint32_t)probe_case_value(probe_case, FLAG_ROWS);
    const uint32_t overlap = (uint32_t)probe_case_value(probe_case, FLAG_OVERLAP);
    const int break_rows = probe_flag_given("--break-rows");
    struct variant alone = variants[PROBE_BASELINE];        /* how a row alone is launched: with the same slices */
    struct kernel_launch launches[MOST_LAUNCHES];
    uint64_t differing = 0, compared = 0;
    (void)context;
    if (probe_flag_given(PROBE_FLAG_BREAK_ROUTE)) {     /* every matrix behind the arena: refused, nothing launched */
        const uint32_t prepared = step_prepare(CHECKED_STEP, 0, 0, rows, overlap, &alone, timing.arena_bytes, launches);
        probe_check_count(probe_case->name, alone.name, "launch_refused", prepared == 0 ? 1u : 0u, 1, 1);
        return;
    }
    if (probe_flag_given(PROBE_FLAG_CORRUPT)) {         /* the host keeps the unit's own bytes for its reference */
        DRIVER(cuMemsetD8(timing.arena + (part->kind == KIND_EXPERTS ? EXPERTS_PER_ROW - 1u : 0u) * timing.unit_bytes,
                          CORRUPTED_BYTE, CORRUPTED_BYTES));
    }
    checked_run(0, rows, overlap, &variants[PROBE_BASELINE], timing.fetched[TOGETHER]);
    check_reference(probe_case->name, rows, overlap, timing.fetched[TOGETHER]);
    alone.slices = alone.slices_down = break_rows ? BROKEN_SLICES : BASELINE_SLICES;
    for (uint32_t row = 0; row < rows; row++) {
        checked_run(row, 1, overlap, &alone, timing.fetched[ALONE]);
        const struct kept *kept = kept_arrays[part->kind];
        for (; kept < kept_arrays[part->kind] + MOST_KEPT && kept->vectors_per_row != 0; kept++) {
            const uint64_t row_bytes = (uint64_t)kept->vectors_per_row * kept->value_bytes *
                                       (kept->values != 0 ? kept->values : part->output_size);
            const uint8_t *together = array_of(timing.fetched[TOGETHER], kept->array);
            const uint8_t *single = array_of(timing.fetched[ALONE], kept->array);
            for (uint64_t byte = 0; byte < row_bytes; byte++) {
                differing += together[row * row_bytes + byte] != single[byte] ? 1u : 0u;
            }
            compared += row_bytes;
            probe_digest_add(together + row * row_bytes, row_bytes);
        }
    }
    probe_check_count(probe_case->name, alone.name, "rows_exact", differing, compared, break_rows);
}

static int prepare_step(void *context, const struct probe_case *probe_case, uint32_t instance, uint64_t step,
                        uint32_t slot)
{
    const struct timespec gap = {0, (long)probe_case_value(probe_case, "--gap-us") * NANOSECONDS_PER_MICROSECOND};
    const uint32_t rows = (uint32_t)probe_case_value(probe_case, FLAG_ROWS);
    const uint32_t overlap = (uint32_t)probe_case_value(probe_case, FLAG_OVERLAP);
    (void)context;
    if (timing.timed_launches == 0) {
        fprintf(stderr, "--only names no launch in the chain of this part, or the case's step was refused\n");
        return 0;
    }
    if (gap.tv_nsec != 0) { (void)nanosleep(&gap, NULL); }      /* the card idles; the host wakes before it prepares */
    return step_prepare(step, slot, 0, rows, overlap, &variants[instance], 0, timing.launches[slot]) != 0;
}

static uint32_t launch_step(void *context, const struct probe_case *probe_case, uint32_t instance, uint32_t slot)
{
    (void)context, (void)probe_case, (void)instance;
    timing.queued_steps++;
    return launches_queue(timing.launches[slot], timing.timed_launches);
}

static void mark_start(void *context)
{
    (void)context;
    DRIVER(cuCtxSynchronize());                  /* an idle card: the upload of a job table is not the stream's */
    timing.queued_steps = 0;
    timing.begun_seconds = probe_monotonic_seconds();
    DRIVER(cuEventRecord(timing.started, timing.stream));
}

/* The card's time between the two events. Of a step alone also the host's half, which the events do not show: until
 * the last launch call returned, and the wall time of the step less the card's time. */
static double elapsed_microseconds(void *context)
{
    float milliseconds = 0.0f;
    (void)context;
    DRIVER(cuEventRecord(timing.stopped, timing.stream));
    const double submitted_seconds = probe_monotonic_seconds();
    DRIVER(cuEventSynchronize(timing.stopped));
    const double waited_seconds = probe_monotonic_seconds();
    DRIVER(cuEventElapsedTime(&milliseconds, timing.started, timing.stopped));
    const double card_microseconds = (double)milliseconds * PROBE_MICROSECONDS_PER_MILLISECOND;
    if (timing.queued_steps == 1) {
        const uint64_t at = timing.sample_count++ % MOST_HOST_SAMPLES;
        host_samples[SUBMIT][at] = (submitted_seconds - timing.begun_seconds) * MICROSECONDS_PER_SECOND;
        host_samples[HOST_HALF][at] =
            (waited_seconds - timing.begun_seconds) * MICROSECONDS_PER_SECOND - card_microseconds;
    }
    return card_microseconds;
}

/* After a case's result lines: the host's half of its steps, and its one figure, which is measured and held
 * against no bar: the baseline against itself. */
static void verdicts(void *context, const struct probe_case *probe_case, const struct probe_result *results,
                     double noise_ratio)
{
    const uint64_t kept = timing.sample_count < MOST_HOST_SAMPLES ? timing.sample_count : MOST_HOST_SAMPLES;
    struct probe_statistics host, submit;
    (void)context;
    probe_statistics_of(host_samples[HOST_HALF], kept, &host);
    probe_statistics_of(host_samples[SUBMIT], kept, &submit);
    printf("HOST card=%s case=%s n=%" PRIu64 " submit_us=%s host_us=%s host_p99_us=%s\n",
           probe_flag_text("--card-index"), probe_case->name, kept, probe_number(submit.median, 1),
           probe_number(host.median, 1), probe_number(host.percentile_99, 1));
    probe_verdict(probe_case->name, PROBE_NAN, PROBE_NAN, noise_ratio, noise_ratio,
                  results[PROBE_BASELINE_AGAIN].same_sign, results[PROBE_BASELINE_AGAIN].rounds);
}

/* One more variant. 0: the table is full. */
static int variant_add(struct probe_program *program, const char *name, uint32_t slices, uint32_t slices_down)
{
    struct variant *variant = &variants[program->instance_count % PROBE_MOST_INSTANCES];
    if (program->instance_count >= PROBE_MOST_INSTANCES) { return 0; }
    *variant = (struct variant){slices, slices_down, 0, ""};
    (void)snprintf(variant->name, sizeof(variant->name), "%s", name);
    program->instances[program->instance_count++] =
        (struct probe_instance){variant->name, NULL, 'E', 1, 0, 0, 0, 0, 0};
    return 1;
}

/* The variants: the baseline twice, the default counts, then every pair of the two slice lists but the
 * baseline's. 0: too many. */
static int configure(void *context)
{
    struct probe_program *program = context;
    const struct probe_flag *slices = probe_flag("--slices"), *slices_down = probe_flag("--slices-down");
    char name[PROBE_NUMBER_BYTES];
    int fits = variant_add(program, "baseline", BASELINE_SLICES, BASELINE_SLICES)
               && variant_add(program, "baseline_again", BASELINE_SLICES, BASELINE_SLICES)
               && variant_add(program, "default", BASELINE_SLICES, BASELINE_SLICES);
    variants[DEFAULT_VARIANT].is_default = 1;
    for (uint32_t pair = 0; pair < slices->value_count * slices_down->value_count; pair++) {
        const uint32_t first = (uint32_t)slices->values[pair / slices_down->value_count];
        const uint32_t second = (uint32_t)slices_down->values[pair % slices_down->value_count];
        (void)snprintf(name, sizeof(name), "slices:%ux%u", first, second);
        fits = fits && ((first == BASELINE_SLICES && second == BASELINE_SLICES)
                        || variant_add(program, name, first, second));
    }
    if (!fits) {
        fprintf(stderr, "more variants than %d: fewer values of --slices, --slices-down\n", (int)PROBE_MOST_INSTANCES);
    }
    return fits;
}

static struct probe_program program = {
    .probe_name = "engine_linear_timing", .flags = flags, .flag_count = PROBE_COUNT_OF(flags), .instances = instances,
    .context = &program, .configure = configure, .open_card = open_card, .describe_case = describe_case,
    .check_case = check_case, .prepare_step = prepare_step, .launch_step = launch_step, .mark_start = mark_start,
    .elapsed_microseconds = elapsed_microseconds, .verdicts = verdicts};

int main(int argument_count, char **arguments) { return probe_main(argument_count, arguments, &program); }
