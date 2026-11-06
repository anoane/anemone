/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * engine_attention_core_timing_probe: what the ENGINE's attention core costs a layer in each order (unit
 * core-reference): the fixed order's kernel (the default), the reference order's, and the fixed order's kernel of an
 * earlier build of the module (--before-modules), on the one card it is given. As engine_linear_timing_probe does,
 * it loads the engine's module itself through the driver interface, takes every launch from the engine's own
 * preparation (source/attention_core.c, compiled in unchanged, with every check it makes), queues it as prepared and
 * times it with the driver's events, by the protocol and with the lines of probe_harness.h.
 *
 * A step is one launch of the core for --rows rows of a layer of one --kind (window: ratio 0, the window alone;
 * pooled: ratio 2; direct: ratio 1), each row at position 2,047 of a context of 2,048 tokens: its 128 window rows
 * and, with a pool, 512 of the entries it sees, one in each band of entries / 512 by a hash of the step. Each row of
 * a step reads a session of its own, the sessions in turn: between two reads of one session the card reads at
 * least 4 L2 sizes of the others (RING line), as a later layer's core finds its rows cold. Queries of magnitude up
 * to 1/4 and keys up to 2, sinks from -2 to 2: softmaxes neither flat nor of one key. Variants: baseline and
 * baseline_again (the fixed order), reference (the reference order), before (the fixed order's kernel of
 * --before-modules: only with it).
 * The reference order is its two launches (scores, then values) into a scores area of the arena.
 * Checked per case before anything is timed, on step 0: bits_before (before's output is the baseline's, every
 * byte); orders_identical (the reference order's output is not the baseline's in every byte: the order picks
 * another kernel); reference_near_fixed (within 4e-3 of the baseline's rms: the orders differ by P's rounding,
 * about 1e-3 of the core's [DRIFT 1]); --break-route: the step's output behind the arena, which the engine's
 * prepare must refuse (launch_refused). DIGEST: the outputs of the checked steps.
 *
 *   engine_attention_core_timing_probe --modules <engine/build/kernels/sm_80 | sm_120a> [--before-modules <dir>]
 *       [--rows 1,8] [--kind window,pooled,direct] and the flags of probe_harness.h
 * Measurement tool: not part of the engine.
 */
#include <cuda.h>

#include "probe_harness.h"
#include "anemone/attention_core.h"

enum {
	MOST_ROWS = LINEAR_MAXIMUM_ROWS,
	CONTEXT = 2048,
	POSITION = CONTEXT - 1,
	COLD_L2_SIZES = 4,
	KERNELS = 3,
	KERNEL_FIXED = 0,
	KERNEL_SCORES = 1,
	KERNEL_VALUES = 2,
	ORDERS = 2,
	ORDER_FIXED = 0,
	ORDER_REFERENCE = 1,
	VARIANT_REFERENCE = PROBE_BASELINE_AGAIN + 1,
	VARIANT_BEFORE = VARIANT_REFERENCE + 1,
	KIND_WINDOW = 0,
	KIND_POOLED = 1,
	KIND_DIRECT = 2,
	PATTERN_ROWS = 1024,
	HALF_BYTES = 2,
	CHECKED_STEP = 0,
	CAPABILITY_SM_80 = 8,
	CAPABILITY_SM_120 = 12,
	VALUE_STEPS = 2048,
	VALUE_CENTRE = 1024,
	ALIGNMENT = LINEAR_OFFSET_ALIGNMENT_BYTES,
	MIX_FIRST_SHIFT = 12,
	MIX_SECOND_SHIFT = 25,
	MIX_THIRD_SHIFT = 27,
	BYTES_PER_KIB = 1024,
	KEY_SHIFT = 9,
	QUERY_SHIFT = 12,
	SINK_SHIFT =
		9, /* keys (k - 1024) 2^-9: -2 .. 2; queries 2^-12: +-1/4 */
	HALF_FRACTION_BITS = 10,
	HALF_EXPONENT_BIAS = 15,
	HALF_SIGN = 0x8000,
	HALF_FRACTION_MASK = 0x3FF,
	BFLOAT16_SHIFT = 16,
	VARIANTS = VARIANT_BEFORE + 1
};
#define MIX_MULTIPLIER 0x2545F4914F6CDD1Dull
#define SEED_MULTIPLIER 0x9E3779B97F4A7C15ull    /* odd: no seed is 0, the one state the generator never leaves */
#define ORDERS_APART_LIMIT 4.0e-3                /* of the baseline's rms */
#define BYTES_PER_MB 1.0e6
#define FLAG_MODULES "--modules"
#define FLAG_BEFORE "--before-modules"
#define FLAG_ROWS "--rows"
#define FLAG_KIND "--kind"
#define DRIVER(call) driver_require((call), #call, __LINE__)

static const uint32_t ratio_of_kind[] = { 0, MODEL_POOLED_RATIO,
					  MODEL_DIRECT_RATIO };
static struct probe_flag flags[] = {           /* the kind first: its arena is laid out once for its row counts */
    PROBE_COMMON_FLAGS,
    PROBE_CASE_WORDS(FLAG_KIND, "window,pooled,direct", "window,pooled,direct", "the layer: its pool's ratio"),
    PROBE_CASE_NUMBERS(FLAG_ROWS, 1, MOST_ROWS, "1,8", "rows of a step"),
    PROBE_TEXT(FLAG_MODULES, "", "the folder of the engine's kernel modules for this card: .../sm_80 or .../sm_120a"),
    PROBE_TEXT(FLAG_BEFORE, "", "the same of an earlier build: its fixed order's kernel is the variant before")};
static struct probe_instance instances[PROBE_MOST_INSTANCES];
static struct {
    CUdevice device;
    CUcontext context;
    CUstream stream;
    CUevent started, stopped;
    CUmodule module, before_module;
    CUfunction functions[KERNELS], before_function;
    CUdeviceptr arena;
    uint64_t arena_bytes, l2_bytes, sessions, session_ring_bytes, session_pool_bytes, slot_bytes;
    uint64_t ring_offset, pool_offset, queries_offset, rotation_offset, output_offset, sinks_offset, slots_offset;
    uint64_t scores_offset;
    uint32_t kind, rows, case_ready;
    uint32_t arithmetic_class;                   /* of the card: the launch of the exact profile */
    struct attention_core_kernels kernels; /* the handles: the KERNEL_ indices */
    struct kernel_launch launches[PROBE_STREAM_STEPS][ORDERS]
				 [ATTENTION_CORE_MOST_LAUNCHES];
    uint32_t launch_counts[PROBE_STREAM_STEPS][ORDERS];  /* of the step prepared in each slot, per order */
    uint16_t pattern[PATTERN_ROWS *
		     MODEL_HEAD_DIMS]; /* the rows every ring and pool repeats */
    uint16_t *outputs[VARIANTS];                 /* of the checked step, per variant */
} timing;

static void driver_require(CUresult result, const char *call, int line)
{
    const char *error_name = NULL;
    if (result == CUDA_SUCCESS) { return; }
    if (cuGetErrorName(result, &error_name) != CUDA_SUCCESS || error_name == NULL) { error_name = "an unnamed error"; }
    fprintf(stderr, "engine_attention_core_timing_probe.c:%d: %s failed: %s\n", line, call, error_name);
    exit(EXIT_FAILURE);
}

static uint64_t mix_next(uint64_t *state)       /* xorshift64*: the same bytes on every machine */
{
    *state ^= *state >> MIX_FIRST_SHIFT;
    *state ^= *state << MIX_SECOND_SHIFT;
    *state ^= *state >> MIX_THIRD_SHIFT;
    return *state * MIX_MULTIPLIER;
}

/* The FP16 bits of (k - 1024) 2^-shift, k the hash modulo 2,048: a whole number below 2^11 times a power of two that
 * keeps it normal (shift at most 14), so FP16 holds it exactly. */
static uint16_t half_of(uint64_t hash, uint32_t shift)
{
    const int32_t whole = (int32_t)(hash % VALUE_STEPS) - VALUE_CENTRE;
    const uint32_t magnitude = (uint32_t)(whole < 0 ? -whole : whole);
    uint32_t top = 0;
    while (magnitude >> (top + 1u) != 0) { top++; }
    if (magnitude == 0) { return 0; }
    return (uint16_t)((whole < 0 ? (uint32_t)HALF_SIGN : 0u) |
                      ((top + (uint32_t)HALF_EXPONENT_BIAS - shift) << (uint32_t)HALF_FRACTION_BITS) |
                      ((magnitude << ((uint32_t)HALF_FRACTION_BITS - top)) & (uint32_t)HALF_FRACTION_MASK));
}

static uint64_t aligned(uint64_t bytes) { return (bytes + ALIGNMENT - 1u) / ALIGNMENT * ALIGNMENT; }

/* The engine's build of the kernel file in folder, read whole and loaded. */
static CUmodule module_load(const char *folder)
{
    char path[PROBE_PATH_BYTES];
    CUmodule module = NULL;
    const int written = snprintf(path, sizeof(path), "%s/%s.cubin", folder,
				 KERNEL_FILE_ATTENTION_CORE);
    FILE *stream = written < 0 || (size_t)written >= sizeof(path) ? NULL : fopen(path, "rb");
    const long byte_count = stream != NULL && fseek(stream, 0, SEEK_END) == 0 ? ftell(stream) : -1;
    char *image = byte_count > 0 ? calloc((size_t)byte_count + 1u, 1) : NULL;
    if (image == NULL || fseek(stream, 0, SEEK_SET) != 0
        || fread(image, 1, (size_t)byte_count, stream) != (size_t)byte_count || fclose(stream) != 0) {
        fprintf(stderr, "%s cannot be read: the folder make -C engine kernels fills\n", path);
        exit(EXIT_FAILURE);
    }
    DRIVER(cuModuleLoadData(&module, image));
    printf("MODULE card=%s file=%s bytes=%ld\n", probe_flag_text("--card-index"), path, byte_count);
    free(image);
    return module;
}

/* A kernel of a module, loaded, its one record of the engine's size. */
static CUfunction function_load(CUmodule module, const char *name)
{
    CUfunction function = NULL;
    size_t record_offset = 0, record_bytes = 0;
    DRIVER(cuModuleGetFunction(&function, module, name));
    DRIVER(cuFuncLoad(function));
    int card_most_shared = 0, static_shared = 0;     /* as the engine's backend: the card's most for any function */
    DRIVER(cuDeviceGetAttribute(&card_most_shared, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
                                timing.device));
    DRIVER(cuFuncGetAttribute(&static_shared, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, function));
    DRIVER(cuFuncSetAttribute(function, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                              card_most_shared - static_shared));
    DRIVER(cuFuncGetParamInfo(function, 0, &record_offset, &record_bytes));
    PROBE_REQUIRE(
	    record_offset == 0 && record_bytes == ATTENTION_CORE_ARGUMENT_BYTES,
	    "a module's kernel takes another record than attention_core_records.h: not this engine's build");
    return function;
}

static uint64_t card_attribute(CUdevice_attribute attribute)
{
    int value = 0;
    DRIVER(cuDeviceGetAttribute(&value, attribute, timing.device));
    PROBE_REQUIRE(value > 0, "the card states an attribute that is not above zero");
    return (uint64_t)value;
}

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
    if (card_count != 1 || minor != 0 || (major != CAPABILITY_SM_80 && major != CAPABILITY_SM_120)
        || strstr(probe_flag_text(FLAG_MODULES), facts->architecture) == NULL
        || (probe_flag_given(FLAG_BEFORE) && strstr(probe_flag_text(FLAG_BEFORE), facts->architecture) == NULL)) {
        fprintf(stderr, "this process sees %d cards, the first of capability %d.%d, with the modules of \"%s\": a "
                        "probe is given one card (CUDA_VISIBLE_DEVICES=<index>) of capability 8.0 or 12.0, and %s "
                        "(and %s) of that architecture: sm_80 or sm_120a\n", card_count, major, minor,
                probe_flag_text(FLAG_MODULES), FLAG_MODULES, FLAG_BEFORE);
        return 0;
    }
    facts->sm_count = (int)card_attribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT);
    facts->l2_bytes = timing.l2_bytes = card_attribute(CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE);
    facts->block_threads_most = (uint32_t)card_attribute(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK);
    facts->shared_bytes_most = card_attribute(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN);
    timing.arithmetic_class = major == CAPABILITY_SM_120 ?
				      ARITHMETIC_CLASS_SM_120 :
				      ARITHMETIC_CLASS_SM_80;
    facts->grid_most[0] = (uint32_t)card_attribute(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X);
    facts->grid_most[1] = (uint32_t)card_attribute(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y);
    facts->grid_most[PROBE_GRID_AXES - 1] = (uint32_t)card_attribute(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z);
    DRIVER(cuDevicePrimaryCtxRetain(&timing.context, timing.device));
    DRIVER(cuCtxSetCurrent(timing.context));
    DRIVER(cuStreamCreate(&timing.stream, CU_STREAM_NON_BLOCKING));
    DRIVER(cuEventCreate(&timing.started, CU_EVENT_DEFAULT));
    DRIVER(cuEventCreate(&timing.stopped, CU_EVENT_DEFAULT));
    timing.module = module_load(probe_flag_text(FLAG_MODULES));
    timing.functions[KERNEL_FIXED] =
	    function_load(timing.module, KERNEL_ATTENTION_CORE);
    timing.functions[KERNEL_SCORES] =
	    function_load(timing.module, KERNEL_ATTENTION_CORE_SCORES);
    timing.functions[KERNEL_VALUES] =
	    function_load(timing.module, KERNEL_ATTENTION_CORE_VALUES);
    if (probe_flag_given(FLAG_BEFORE)) {
        timing.before_module = module_load(probe_flag_text(FLAG_BEFORE));
	timing.before_function =
		function_load(timing.before_module, KERNEL_ATTENTION_CORE);
    }
    timing.kernels = (struct attention_core_kernels){ KERNEL_FIXED,
						      KERNEL_SCORES,
						      KERNEL_VALUES };
    for (uint64_t value = 0, state = SEED_MULTIPLIER; value < PROBE_COUNT_OF(timing.pattern); value++) {
        timing.pattern[value] = half_of(mix_next(&state), KEY_SHIFT);
    }
    for (uint32_t instance = 0; instance < VARIANTS; instance++) {
	    timing.outputs[instance] =
		    malloc((size_t)MOST_ROWS * MODEL_HEADS_ROW_BYTES);
	    PROBE_REQUIRE(timing.outputs[instance] != NULL,
			  "out of host memory");
    }
    return 1;
}

/* The bytes one row of the kind reads: its window rows and its selected entries. */
static uint64_t row_read_bytes(uint32_t kind)
{
	return (uint64_t)(MODEL_WINDOW +
			  (kind == KIND_WINDOW ? 0u : MODEL_SELECTED)) *
	       MODEL_STATE_ROW_BYTES;
}

/* The arena of a kind: sessions enough that a session's rows come back cold, their rings and pools (the rows of the
 * pattern again and again), then the queries, rotation, sinks and output every step shares, then per slot of a
 * stream its step area and its selection. */
static void kind_set_up(uint32_t kind, const char *case_name)
{
    const uint32_t ratio = ratio_of_kind[kind];
    timing.kind = kind;
    timing.sessions = COLD_L2_SIZES * timing.l2_bytes / row_read_bytes(kind) + 2u * MOST_ROWS;
    timing.session_ring_bytes =
	    (uint64_t)MODEL_RING_SLOTS * MODEL_STATE_ROW_BYTES;
    timing.session_pool_bytes =
	    ratio == 0 ? 0 : (uint64_t)CONTEXT / ratio * MODEL_STATE_ROW_BYTES;
    timing.slot_bytes = aligned(STEP_BLOCK_BYTES_VALUE +
				(uint64_t)MOST_ROWS * STEP_ROW_BYTES_VALUE) +
			(uint64_t)MOST_ROWS * MODEL_SELECTION_ROW_BYTES;
    timing.ring_offset = 0;
    timing.pool_offset = timing.sessions * timing.session_ring_bytes;
    timing.queries_offset = timing.pool_offset + timing.sessions * timing.session_pool_bytes;
    timing.rotation_offset =
	    timing.queries_offset + (uint64_t)MOST_ROWS * MODEL_HEADS_ROW_BYTES;
    timing.output_offset = timing.rotation_offset +
			   (uint64_t)MOST_ROWS * MODEL_ROTATION_ROW_BYTES;
    timing.sinks_offset =
	    timing.output_offset + (uint64_t)MOST_ROWS * MODEL_HEADS_ROW_BYTES;
    timing.scores_offset =
	    timing.sinks_offset + aligned(MODEL_HEADS_VALUE * sizeof(float));
    timing.slots_offset = timing.scores_offset +
			  (uint64_t)MOST_ROWS * ATTENTION_CORE_SCORES_ROW_BYTES;
    timing.arena_bytes = timing.slots_offset + PROBE_STREAM_STEPS * timing.slot_bytes;
    if (timing.arena != 0) { DRIVER(cuMemFree(timing.arena)); }
    DRIVER(cuMemAlloc(&timing.arena, timing.arena_bytes));
    const uint64_t states_bytes = timing.queries_offset;
    for (uint64_t byte = 0; byte < states_bytes; byte += sizeof(timing.pattern)) {
        const uint64_t left = states_bytes - byte;
        DRIVER(cuMemcpyHtoD(timing.arena + byte, timing.pattern, left < sizeof(timing.pattern) ? left
                                                                                           : sizeof(timing.pattern)));
    }
    uint64_t state = SEED_MULTIPLIER + kind;
    static uint16_t queries[MOST_ROWS * MODEL_HEADS_VALUE * MODEL_HEAD_DIMS];
    static float rotation[MOST_ROWS * MODEL_ROTATION_ROW_BYTES / sizeof(float)];
    static float sinks[MODEL_HEADS_VALUE];
    for (size_t value = 0; value < PROBE_COUNT_OF(queries); value++) {
        queries[value] = half_of(mix_next(&state), QUERY_SHIFT);
    }
    for (size_t value = 0; value < PROBE_COUNT_OF(rotation); value++) {
        rotation[value] = (float)(((double)(mix_next(&state) % VALUE_STEPS) - VALUE_CENTRE) / VALUE_CENTRE);
    }
    for (size_t head = 0; head < PROBE_COUNT_OF(sinks); head++) {
        sinks[head] = (float)ldexp((double)(mix_next(&state) % VALUE_STEPS) - VALUE_CENTRE, -SINK_SHIFT);
    }
    DRIVER(cuMemcpyHtoD(timing.arena + timing.queries_offset, queries, sizeof(queries)));
    DRIVER(cuMemcpyHtoD(timing.arena + timing.rotation_offset, rotation, sizeof(rotation)));
    DRIVER(cuMemcpyHtoD(timing.arena + timing.sinks_offset, sinks, sizeof(sinks)));
    const double cold_l2 = (double)((timing.sessions - MOST_ROWS) * row_read_bytes(kind)) / (double)timing.l2_bytes;
    PROBE_REQUIRE(cold_l2 >= (double)COLD_L2_SIZES, "the sessions are too few for this card's L2 (CONVENTIONS 3.5)");
    printf("RING card=%s case=%s sessions=%" PRIu64 " row_read_kib=%" PRIu64 " arena_mib=%.0f cold_l2=%.1f\n",
           probe_flag_text("--card-index"), case_name, timing.sessions, row_read_bytes(kind) / BYTES_PER_KIB,
           (double)timing.arena_bytes / PROBE_BYTES_PER_MIB, cold_l2);
}

/* The launches of step `step` in slot `slot`, both orders: its rows' sessions and selections uploaded, each launch
 * prepared by the engine (shift moves the output behind the arena: --break-route). 0: the engine refused. */
static int step_prepare(uint64_t step, uint32_t slot, uint64_t shift)
{
    static struct {
	    struct step_block block;
	    struct step_row_record rows[MOST_ROWS];
    } area;
    static int32_t selection[MOST_ROWS * MODEL_SELECTED];
    const uint32_t ratio = ratio_of_kind[timing.kind], visible = ratio == 0 ? 0 : CONTEXT / ratio;
    const uint32_t band = visible / MODEL_SELECTED;
    const uint64_t slot_offset = timing.slots_offset + slot * timing.slot_bytes;
    const uint64_t step_bytes = STEP_BLOCK_BYTES_VALUE +
				(uint64_t)timing.rows * STEP_ROW_BYTES_VALUE;
    const uint64_t selection_offset =
	    slot_offset + aligned(STEP_BLOCK_BYTES_VALUE +
				  (uint64_t)MOST_ROWS * STEP_ROW_BYTES_VALUE);
    int valid = 1;
    area.block = (struct step_block){ step + 1u, timing.rows, 0 };
    for (uint32_t row = 0; row < timing.rows; row++) {
        const uint64_t session = (step * timing.rows + row) % timing.sessions;
	area.rows[row] = (struct step_row_record){ POSITION, 0,
						   (uint32_t)session,
						   STEP_ROW_LOGITS };
	uint64_t state = (step * MOST_ROWS + row + 1u) * SEED_MULTIPLIER;
	for (uint32_t place = 0; place < MODEL_SELECTED; place++) {
		selection[row * MODEL_SELECTED + place] =
			ratio == 0 ?
				-1 :
				(int32_t)(place * band +
					  (band > 1u ? mix_next(&state) % band :
						       0u));
	}
	probe_launch_value(&valid, KERNEL_ATTENTION_CORE, "a row's session",
			   session, 0, timing.sessions - 1u);
    }
    DRIVER(cuMemcpyHtoD(timing.arena + slot_offset, &area, step_bytes));
    DRIVER(cuMemcpyHtoD(timing.arena + selection_offset, selection,
			(size_t)timing.rows * MODEL_SELECTION_ROW_BYTES));
    for (uint32_t order = 0; order < ORDERS && valid; order++) {
        const uint64_t rows = timing.rows;
	const struct attention_core_description description = {
		.step = { slot_offset, step_bytes },
		.queries = { timing.queries_offset,
			     rows * MODEL_HEADS_ROW_BYTES },
		.ring = { timing.ring_offset,
			  timing.sessions * timing.session_ring_bytes },
		.pool = { timing.pool_offset,
			  timing.sessions * timing.session_pool_bytes },
		.selection = { selection_offset,
			       rows * MODEL_SELECTION_ROW_BYTES },
		.sinks = { timing.sinks_offset,
			   MODEL_HEADS_VALUE * sizeof(float) },
		.rotation = { timing.rotation_offset,
			      rows * MODEL_ROTATION_ROW_BYTES },
		.output = { timing.output_offset + shift,
			    rows * MODEL_HEADS_ROW_BYTES },
		.scores = { timing.scores_offset,
			    rows * ATTENTION_CORE_SCORES_ROW_BYTES },
		.row_count = timing.rows,
		.session_capacity = (uint32_t)timing.sessions,
		.context_token_limit = CONTEXT,
		.pool_ratio = ratio,
		.first_head = 0,
		.head_count = MODEL_HEADS_VALUE,
		.order = order == ORDER_REFERENCE ? LINEAR_ORDER_REFERENCE :
						    LINEAR_ORDER_FIXED,
		.arithmetic_class = timing.arithmetic_class
	};
	valid = attention_core_prepare(&description, timing.arena_bytes,
				       &timing.kernels,
				       timing.launches[slot][order],
				       &timing.launch_counts[slot][order]) ==
		STATUS_OK;
	for (uint32_t index = 0; valid && index < timing.launch_counts[slot][order]; index++) {
		const struct kernel_launch *const launch =
			&timing.launches[slot][order][index];
		probe_launch_geometry(&valid, KERNEL_ATTENTION_CORE, NULL,
				      launch->grid[0], launch->grid[1],
				      launch->grid[LAUNCH_DIMENSIONS - 1],
				      launch->block[0],
				      launch->dynamic_shared_byte_count);
	}
    }
    return valid;
}

/* The launches of a variant prepared in a slot: their functions, the engine's records, the arena. */
static uint32_t launch_queue(uint32_t instance, uint32_t slot)
{
    const uint32_t order = instance == VARIANT_REFERENCE ? ORDER_REFERENCE : ORDER_FIXED;
    for (uint32_t index = 0; index < timing.launch_counts[slot][order]; index++) {
	    struct kernel_launch *launch = &timing.launches[slot][order][index];
	    void *parameters[] = { launch->argument, &timing.arena };
	    DRIVER(cuLaunchKernel(instance == VARIANT_BEFORE ?
					  timing.before_function :
					  timing.functions[launch->kernel],
				  launch->grid[0], launch->grid[1],
				  launch->grid[LAUNCH_DIMENSIONS - 1],
				  launch->block[0], launch->block[1],
				  launch->block[LAUNCH_DIMENSIONS - 1],
				  launch->dynamic_shared_byte_count,
				  timing.stream, parameters, NULL));
    }
    return 1;
}

static void describe_case(void *context, const struct probe_case *probe_case, struct probe_case_facts *facts)
{
    const uint32_t kind = (uint32_t)probe_case_value(probe_case, FLAG_KIND);
    (void)context;
    if (!timing.case_ready || kind != timing.kind) {
        kind_set_up(kind, probe_case->name);
        timing.case_ready = 1;
    }
    timing.rows = (uint32_t)probe_case_value(probe_case, FLAG_ROWS);
    facts->format = "fp16_state";
    facts->part = probe_case->name;
    facts->rows = timing.rows;
    facts->experts = -1;
    facts->overlap = -1;
    facts->megabytes = (double)(timing.rows * row_read_bytes(kind)) / BYTES_PER_MB;
}

/* Step 0 of a variant, its output fetched into timing.outputs[instance]. */
static void checked_run(uint32_t instance)
{
	const uint64_t bytes = (uint64_t)timing.rows * MODEL_HEADS_ROW_BYTES;
	PROBE_REQUIRE(step_prepare(CHECKED_STEP, 0, 0),
		      "the checked step was refused");
	DRIVER(cuMemsetD8(timing.arena + timing.output_offset, 0, bytes));
	DRIVER(cuCtxSynchronize());
	(void)launch_queue(instance, 0);
	DRIVER(cuStreamSynchronize(timing.stream));
	DRIVER(cuMemcpyDtoH(timing.outputs[instance],
			    timing.arena + timing.output_offset, bytes));
	probe_digest_add(timing.outputs[instance], bytes);
}

static double bfloat16_value(uint16_t bits)
{
    const uint32_t word = (uint32_t)bits << BFLOAT16_SHIFT;
    float value = 0.0f;
    memcpy(&value, &word, sizeof(value));
    return (double)value;
}

static void check_case(void *context, const struct probe_case *probe_case)
{
	const uint64_t values =
		(uint64_t)timing.rows * MODEL_HEADS_ROW_BYTES / HALF_BYTES;
	double error = 0.0, square = 0.0;
	uint64_t differing = 0;
	(void)context;
	if (probe_flag_given(
		    PROBE_FLAG_BREAK_ROUTE)) { /* the output behind the arena: refused, nothing run */
		probe_check_count(
			probe_case->name, "baseline", "launch_refused",
			step_prepare(CHECKED_STEP, 0, timing.arena_bytes) == 0 ?
				1u :
				0u,
			1, 1);
		return;
	}
    checked_run(PROBE_BASELINE);
    checked_run(VARIANT_REFERENCE);
    for (uint64_t value = 0; value < values; value++) {
        const double fixed = bfloat16_value(timing.outputs[PROBE_BASELINE][value]);
        const double reference = bfloat16_value(timing.outputs[VARIANT_REFERENCE][value]);
        differing += timing.outputs[PROBE_BASELINE][value] != timing.outputs[VARIANT_REFERENCE][value] ? 1u : 0u;
        error += (reference - fixed) * (reference - fixed);
        square += fixed * fixed;
    }
    probe_check_count(probe_case->name, "reference", "orders_identical", differing == 0 ? 1u : 0u, 1, 0);
    probe_check_error(probe_case->name, "reference", "reference_near_fixed", square > 0.0 ? sqrt(error / square) : 0.0,
                      ORDERS_APART_LIMIT, 0);
    if (probe_flag_given(FLAG_BEFORE)) {
        checked_run(VARIANT_BEFORE);
        probe_check_count(probe_case->name, "before", "bits_before",
                          (uint64_t)(memcmp(timing.outputs[PROBE_BASELINE], timing.outputs[VARIANT_BEFORE],
                                            values * HALF_BYTES) != 0), 1, 0);
    }
}

static int prepare_step(void *context, const struct probe_case *probe_case, uint32_t instance, uint64_t step,
                        uint32_t slot)
{
    (void)context, (void)probe_case, (void)instance;
    return step_prepare(step, slot, 0);
}

static uint32_t launch_step(void *context, const struct probe_case *probe_case, uint32_t instance, uint32_t slot)
{
    (void)context, (void)probe_case;
    return launch_queue(instance, slot);
}

static void mark_start(void *context)
{
    (void)context;
    DRIVER(cuCtxSynchronize());                  /* an idle card: the uploads of a step are not the stream's */
    DRIVER(cuEventRecord(timing.started, timing.stream));
}

static double elapsed_microseconds(void *context)
{
    float milliseconds = 0.0f;
    (void)context;
    DRIVER(cuEventRecord(timing.stopped, timing.stream));
    DRIVER(cuEventSynchronize(timing.stopped));
    DRIVER(cuEventElapsedTime(&milliseconds, timing.started, timing.stopped));
    return (double)milliseconds * PROBE_MICROSECONDS_PER_MILLISECOND;
}

/* Measured, held against no bar: the reference order's ratio, and before's (1 where the default is unchanged). */
static void verdicts(void *context, const struct probe_case *probe_case, const struct probe_result *results,
                     double noise_ratio)
{
    (void)context, (void)probe_case;
    probe_verdict("reference_order", PROBE_NAN, PROBE_NAN, results[VARIANT_REFERENCE].ratio, noise_ratio,
                  results[VARIANT_REFERENCE].same_sign, results[VARIANT_REFERENCE].rounds);
    if (probe_flag_given(FLAG_BEFORE)) {
        probe_verdict("default_before", PROBE_NAN, PROBE_NAN, results[VARIANT_BEFORE].ratio, noise_ratio,
                      results[VARIANT_BEFORE].same_sign, results[VARIANT_BEFORE].rounds);
    }
}

static void close_card(void *context)
{
    (void)context;
    if (timing.arena != 0) { DRIVER(cuMemFree(timing.arena)); }
    for (uint32_t instance = 0; instance < VARIANTS; instance++) { free(timing.outputs[instance]); }
}

/* baseline, baseline_again, reference; before with --before-modules. */
static int configure(void *context)
{
    struct probe_program *program = context;
    static const char *const names[] = {"baseline", "baseline_again", "reference", "before"};
    program->instance_count = probe_flag_given(FLAG_BEFORE) ? VARIANT_BEFORE + 1u : VARIANT_REFERENCE + 1u;
    for (uint32_t instance = 0; instance < program->instance_count; instance++) {
	    program->instances[instance] = (struct probe_instance){
		    names[instance],
		    instance == VARIANT_REFERENCE ?
			    KERNEL_ATTENTION_CORE_VALUES :
			    KERNEL_ATTENTION_CORE,
		    'E',
		    1,
		    ATTENTION_CORE_THREADS,
		    0,
		    0,
		    0,
		    0
	    };
    }
    if (!probe_flag_given(FLAG_MODULES)) {
        fprintf(stderr, "%s is required\n", FLAG_MODULES);
        return 0;
    }
    return 1;
}

static struct probe_program program = {
    .probe_name = "engine_attention_core_timing", .flags = flags, .flag_count = PROBE_COUNT_OF(flags),
    .instances = instances, .context = &program, .configure = configure, .open_card = open_card,
    .describe_case = describe_case, .check_case = check_case, .prepare_step = prepare_step,
    .launch_step = launch_step, .mark_start = mark_start, .elapsed_microseconds = elapsed_microseconds,
    .verdicts = verdicts, .close_card = close_card};

int main(int argument_count, char **arguments) { return probe_main(argument_count, arguments, &program); }
