/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * trellis_expert_probe: round 1, package 7 of the speed campaign (research/speed/ROUND_1.md, "trellis_expert";
 * ledger T1, T2, T9, T11, part of T8). What the routed experts of one MoE layer cost with EXL3 trellis weights in
 * kernels of our own (trellis_kernels.cuh), as the format's fp16 values and as mul1's integer operand, and what
 * exlr8 v2's decode costs by each way of its rounding (exlr8_kernels.cuh). One codebook and one width K a run
 * (--codebook, --bits), every variant a named instance of one binary (trx_variants), interleaved by probe_harness.h;
 * a step is the five launches of a layer (--part experts), or its three rotations or its two products alone.
 * CHECKED per case before anything is timed, on step 0, for two pairs (the first row at its last expert, the last
 * row at its first):
 *   codebook_states         once a run: all 65,536 states through each form's decode on the card against the
 *                           definition, bit for bit (the two halves of a register: a state and its complement)
 *   rotated_vectors         what the rotations wrote for the pair (the fed input of gate and of up, the fed hidden
 *                           vector, its row's output from the card's six down products) against the host in the
 *                           kernels' order, bit for bit
 *   host_reference_*        the hidden vector and the down product against double precision from the file-order
 *                           tiles decoded by trellis_reference.h, fed what the card fed: 2e-3 of the rms
 *   integer_contract        operand:imma8: its three products against the host's integers and fp32 steps, bit for bit
 *   bits_against_baseline   every word of a variant that must have the baseline's bits (window:loads, round:cvt)
 *   rows_exact              every row launched alone against the row in its launch: of a float form on a CMP 170HX the
 *                           count is the card's answer (class C becomes R at 0 where rows share an expert, --overlap
 *                           50 or 100; a difference prints `reported` and makes it F; unit 3a: its fp16 sums follow
 *                           no simple rule), of the integer form and on the PRO 0 is required
 *   guard_bytes, and the DIGEST of all outputs, which the batch compares between the two CMP 170HX
 * EACH CHECK SHOWN FAILING, by the host's side or by values, never by what a kernel indexes by: --corrupt (the first
 * pair's gate and down matrices on the card), --break codebook (the host reads the codebook one state off), --break
 * wrap (the host's first states of a tile do not wrap), --break placement (the host exchanges the two outputs of a
 * lane): the first pair's references and integer_contract must report, the other pair's hold; --break rotation (fed
 * bytes overwritten after the first launch: rotated_vectors); --break exactness (--rows 4 --overlap 50, mul1: the
 * baseline's twin, whose order of slices follows a job's rows: rows_exact, and the next variant's
 * bits_against_baseline); --break-route (eight launches past one bound each); --break-guard.
 *
 *   trellis_expert_probe [--codebook mul1|mcg|exlr8] [--bits 2|3|4] [--rows 1,2,4,6,8] [--overlap 0,50,100]
 *                        [--part experts|rotations|products] [--break NAME] and the flags of probe_harness.h
 * Measurement tool: not part of the engine.
 */
#include "probe_harness.h"
#include "trellis_kernels.cuh"
#include "trellis_reference.h"

enum {
    TRX_BITS_REQUIRED = 0, TRX_BITS_OWN,         /* a variant's outputs: the baseline's, or a contract of its own */
    TRX_ROTATE_INPUT = 0, TRX_GATE_UP, TRX_ROTATE_MIDDLE, TRX_DOWN, TRX_ROTATE_OUTPUT, TRX_STEP_LAUNCHES,
    TRX_PART_EXPERTS = 0, TRX_PART_ROTATIONS, TRX_PART_PRODUCTS, TRX_STAGE_LAUNCHES = 2,  /* a rotation, its product */
    TRX_FAULT_NONE = 0, TRX_FAULT_CODEBOOK, TRX_FAULT_WRAP, TRX_FAULT_PLACEMENT, TRX_FAULT_ROTATION,   /* --break */
    TRX_FAULT_EXACTNESS, TRX_MUL1 = TR_REFERENCE_MUL1, TRX_MCG = TR_REFERENCE_MCG, TRX_EXLR8 = TR_REFERENCE_EXLR8,
    TRX_ANY_CODEBOOK = -1, TRX_MOST_BITS = 4, TRX_GATE_UP_SLICES = 4, TRX_DOWN_SLICES = 3, TRX_FEWER_SLICES = 2,
    TRX_MORE_SLICES = 8, TRX_CHECKED_STEP = 0, TRX_CHECKED_PAIRS = 2, TRX_OVERLAP_STEP = 50, TRX_PERCENT = 100,
    TRX_FIRST_VARIANT = PROBE_BASELINE_AGAIN + 1, TRX_BASELINE_COPIES = 2, TRX_FORMAT_BYTES = 32,
    TRX_CORRUPTED_BYTES = 1024 * 1024, TRX_CORRUPTED_BYTE = 0x5A, TRX_BROKEN_BYTES = 64, TRX_BROKEN_BYTE = 0x3C,
    TRX_STALE_BYTE = 0xFF, TRX_COLD_L2_SIZES = 4, TRX_FILL_THREADS = 256, TRX_FILL_BLOCKS_PER_SM = 8,
    TRX_CODEBOOK_THREADS = 256, TRX_PLACED_EXPONENT = 8,
    TRX_BROKEN_EXPERT = 0, TRX_BROKEN_MATRIX, TRX_BROKEN_VECTOR, TRX_BROKEN_COLUMNS, TRX_BROKEN_JOBS,
    TRX_BROKEN_ROWS, TRX_BROKEN_FIRST_ROW, TRX_BROKEN_SLOT, TRX_BROKEN_ROUTES,         /* --break-route, in turn */
    TRX_STORED_SEED = 0x243F6A88, TRX_MIX_SHIFT_OUTER = 16, TRX_MIX_SHIFT_INNER = 15,
    /* a side value: a hashed sign and mantissa, a magnitude of 2^-1 .. 2^2, of gate's and up's svh 2^-9 .. 2^-6 (so
     * that a hidden value is of order 1) */
    TRX_SIDE_SEED = 0x13198A2E, TRX_SIDE_FIELD = 14, TRX_SMALL_SIDE_FIELD = 6, TRX_SIDE_FIELDS = 3,
    /* synthetic inputs, exact in bfloat16: a block of 128 is of one of eight kinds (trx_input_value) */
    TRX_INPUT_SEED = 0x1234567, TRX_INPUT_KINDS = 8, TRX_INPUT_SPREAD = 4, TRX_INPUT_ZEROS = 5, TRX_INPUT_SINGLE = 6,
    TRX_INPUT_TINY = 7, TRX_INPUT_TINY_SHIFT = 16, TRX_INPUT_STEPS = 255, TRX_SPREAD_POWERS = 18, TRX_SIGN_BIT = 8,
    TRX_ROUTE_SEED = 0xABCDEF, TRX_ROUTE_WEIGHT_STEPS = 32, TRX_STEP_STRIDE = 977
};
#define TRX_MIX_MULTIPLIER_FIRST 0x7feb352du
#define TRX_MIX_MULTIPLIER_SECOND 0x846ca68bu
#define TRX_FLAG_CODEBOOK "--codebook"
#define TRX_FLAG_BITS "--bits"
#define TRX_FLAG_ROWS "--rows"
#define TRX_FLAG_OVERLAP "--overlap"
#define TRX_FLAG_PART "--part"
#define TRX_FLAG_BREAK "--break"
#define TRX_PARTS "experts,rotations,products"
#define TRX_RELATIVE_TOLERANCE 2.0e-3
#define TRX_INPUT_OFFSET 128.0f
#define TRX_INPUT_DIVISOR 64.0f
#define TRX_ROUTE_WEIGHT_DIVISOR 64.0f
#define TRX_BYTES_PER_MB 1.0e6
#define TRX_POWER_OF_TWO_MANTISSA 0.5f           /* what frexpf returns of a power of two */
#define TRX_IMMA8_BAR 0.85                       /* ROUND_1.md: operand:imma8 over fp16 on a CMP 170HX, and what */
#define TRX_IMMA8_BAR_US -30.0                   /* it stands for at the 205 us the fp16 form is expected at */
#define TRX_SLICES {TRX_GATE_UP_SLICES, TRX_DOWN_SLICES}
#define TRX_ROW_WORDS(rows) ((uint64_t)(rows) * (TR_HIDDEN_SIZE + TR_EXPERTS_PER_ROW * TR_INNER_SIZE))

struct trx_variant {
    const char *name;
    int codebook, form, bits_kind;               /* the codebook of the runs that have it; TR_FORM_*; TRX_BITS_* */
    uint32_t slices[TR_STAGES];
    struct tr_kernels kernels;
};
static const struct trx_variant trx_variants[] = {          /* a run's baseline: the first of its codebook */
    {"baseline", TRX_MUL1, TR_FORM_MUL1_FP16, TRX_BITS_REQUIRED, TRX_SLICES, TR_KERNELS(fp16)},
    {"window:loads", TRX_MUL1, TR_FORM_MUL1_FP16, TRX_BITS_REQUIRED, TRX_SLICES, TR_KERNELS(fp16_loads)},
    {"slices:1", TRX_MUL1, TR_FORM_MUL1_FP16, TRX_BITS_OWN, {1, 1}, TR_KERNELS(fp16)},
    {"slices:2", TRX_MUL1, TR_FORM_MUL1_FP16, TRX_BITS_OWN, {TRX_FEWER_SLICES, TRX_FEWER_SLICES}, TR_KERNELS(fp16)},
    {"slices:8", TRX_MUL1, TR_FORM_MUL1_FP16, TRX_BITS_OWN, {TRX_MORE_SLICES, TRX_MORE_SLICES}, TR_KERNELS(fp16)},
    {"operand:imma8", TRX_MUL1, TR_FORM_MUL1_IMMA8, TRX_BITS_OWN, TRX_SLICES, TR_KERNELS(imma8)},
    {"baseline", TRX_MCG, TR_FORM_MCG_FP16, TRX_BITS_REQUIRED, TRX_SLICES, TR_KERNELS(mcg)},
    {"baseline", TRX_EXLR8, TR_FORM_EXLR8_ROUND, TRX_BITS_REQUIRED, TRX_SLICES, TR_KERNELS(exlr8_round)},
    {"round:table", TRX_EXLR8, TR_FORM_EXLR8_TABLE, TRX_BITS_OWN, TRX_SLICES, TR_KERNELS(exlr8_table)},
    {"round:cvt", TRX_EXLR8, TR_FORM_EXLR8_CVT, TRX_BITS_REQUIRED, TRX_SLICES, TR_CVT_KERNELS},
    {"operand:fp8", TRX_EXLR8, TR_FORM_EXLR8_FP8, TRX_BITS_OWN, TRX_SLICES, TR_FP8_KERNELS},
    {"decode:none", TRX_ANY_CODEBOOK, TR_FORM_NO_DECODE, TRX_BITS_OWN, TRX_SLICES, TR_KERNELS(no_decode)},
};
/* the twin of --break exactness, which no instance times; what admits the table's blocks their shared memory and
 * says how many an SM holds */
static const struct trx_variant trx_broken_order = {"baseline",  TRX_MUL1, TR_FORM_MUL1_FP16, TRX_BITS_REQUIRED,
                                                    TRX_SLICES, TR_KERNELS(fp16_broken_order)};
static cudaError_t (*const trx_admit_table[TR_WIDTHS][TR_STAGES])(size_t, int, int *) = TR_SET(admit_, exlr8_table);
static struct probe_flag trx_flags[] = {
    PROBE_COMMON_FLAGS,
    PROBE_WORD_LIST(TRX_FLAG_CODEBOOK, "mul1,mcg,exlr8", "mul1", "the codebook of this run: one"),
    PROBE_NUMBER(TRX_FLAG_BITS, TR_LEAST_BITS, TRX_MOST_BITS, "3", "K: the bits of a weight"),
    PROBE_CASE_NUMBERS(TRX_FLAG_ROWS, 1, TR_MOST_ROWS, "1,2,4,6,8", "rows of a step"),
    PROBE_CASE_WORDS(TRX_FLAG_OVERLAP, "0,50,100", "0,50,100", "percent of its experts a row shares with the next"),
    PROBE_WORD_LIST(TRX_FLAG_PART, TRX_PARTS, "experts", "a step's launches: all five, or three or two alone: one"),
    PROBE_WORD_LIST(TRX_FLAG_BREAK, "codebook,wrap,placement,rotation,exactness", "", "show that check failing: one"),
};
struct trx_table {                               /* a step's pairs and jobs, as a slot of a stream holds them */
    struct tr_pair pairs[TR_MOST_PAIRS];
    struct tr_job jobs[TR_MOST_JOBS];
};
struct trx_step {                                /* one step, as its launches take it */
    struct tr_step launch;
    uint32_t row_count, blocks[TR_STAGES];
};
struct trx_result { float rows[TR_MOST_ROWS * TR_HIDDEN_SIZE], hidden[TR_MOST_PAIRS * TR_INNER_SIZE]; };
struct trx_fed {                                 /* one vector as the card feeds it, by value index */
    double values[TR_HIDDEN_SIZE];               /* what a product multiplies */
    int32_t integers[TR_HIDDEN_SIZE];
    float scales[TR_HIDDEN_SIZE / TR_HADAMARD * TR_BLOCK_SCALES];        /* the card's a and c of every block */
};
static struct {
    int codebook, part, fault;                   /* --codebook, --part, --break */
    uint32_t bits, width;                        /* K, and K - 2 */
    struct tr_step card;                         /* the arrays as allocated */
    struct trx_table *tables;                    /* [slot of a stream] */
    uint32_t *codebook_registers;                /* check 1: a register per state */
    uint64_t table_blocks, shared_most;          /* the blocks round:table is launched with; a block's most bytes */
    struct trx_step steps[PROBE_STREAM_STEPS];
    double values[TR_REFERENCE_STATES];          /* the run's codebook by the definition: a state's value, */
    uint16_t byte_sums[TR_REFERENCE_STATES];     /* mul1's byte sum, */
    uint8_t codes[TR_REFERENCE_STATES];          /* exlr8's E4M3 code */
} trx;
static struct probe_instance trx_instances[PROBE_COUNT_OF(trx_variants) + 1];
static const struct trx_variant *trx_used[PROBE_COUNT_OF(trx_variants) + 1];        /* of each instance */
static uint16_t trx_sides[TR_EXPERT_COUNT * TR_SIDE_VALUES];
static uint16_t trx_states[TRX_CHECKED_PAIRS][TR_MATRIX_KINDS][TR_HIDDEN_SIZE * TR_INNER_SIZE];   /* [input][output] */
static char trx_format[TRX_FORMAT_BYTES];
static uint32_t trx_instance_count;

/* ---- synthetic experts, side vectors and inputs ---- */
__host__ __device__ inline uint32_t trx_mix(uint32_t value) {
    value = (value ^ (value >> TRX_MIX_SHIFT_OUTER)) * TRX_MIX_MULTIPLIER_FIRST;
    value = (value ^ (value >> TRX_MIX_SHIFT_INNER)) * TRX_MIX_MULTIPLIER_SECOND;
    return value ^ (value >> TRX_MIX_SHIFT_OUTER);
}
/* Word `index` of the experts in the FILE's order ([expert][matrix kind][input tile][output tile][8 K words]): a
 * hash of its place. Every bit pattern is a tile, and hashed bits have the codebook's own distribution of values. */
__host__ __device__ inline uint32_t trx_file_word(uint64_t index) {
    return trx_mix(TRX_STORED_SEED + (uint32_t)index + trx_mix((uint32_t)(index >> TR_WORD_BITS)));
}
/* The packer: the card's order of tiles ([output tile][input tile]) from the file's, a tile's words untouched. Any
 * grid, any block (1.1 to 2.3 GB would be seconds over the link). Its name keeps it out of a profile of a step. */
__global__ void setup_fill_kernel(uint32_t *stored, uint64_t word_count, uint32_t bits) {
    const uint64_t tile_words = TR_WORDS_PER_BIT * bits, thread_total = (uint64_t)gridDim.x * blockDim.x;
    for (uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; index < word_count; index += thread_total) {
        const uint64_t tile = index / tile_words, matrix = tile / TR_MATRIX_TILES, in_matrix = tile % TR_MATRIX_TILES;
        const uint64_t input_tiles = (matrix % TR_MATRIX_KINDS == TR_DOWN ? TR_INNER_SIZE : TR_HIDDEN_SIZE) / TR_TILE;
        const uint64_t file_tile = in_matrix % input_tiles * (TR_MATRIX_TILES / input_tiles) + in_matrix / input_tiles;
        stored[index] = trx_file_word((matrix * TR_MATRIX_TILES + file_tile) * tile_words + index % tile_words);
    }
}
/* Check 1's kernel: grid (65,536 / block), any block: the register a form makes of a state and its complement. */
template <int form>
__global__ void setup_codebook_kernel(uint32_t *registers, const uint8_t *table) {
    const uint32_t state = blockIdx.x * blockDim.x + threadIdx.x;
    registers[state] = tr_weight_pair<form>(state, state ^ TR_STATE_MASK, table);
}
static uint16_t trx_side_bits(uint32_t index) { /* value `index` of the experts' side vectors, as fp16 */
    const uint32_t bits = trx_mix(TRX_SIDE_SEED + index), at = index % TR_SIDE_VALUES;
    const uint32_t field = (at >= TR_SVH_GATE_AT && at < TR_SUH_DOWN_AT ? TRX_SMALL_SIDE_FIELD : TRX_SIDE_FIELD)
                           + bits % TRX_SIDE_FIELDS;
    return (uint16_t)((bits & TR_REFERENCE_FP16_SIGN) | (field << TR_REFERENCE_FP16_EXPONENT_SHIFT)
                      | ((bits >> TR_HALF_BITS) & TR_REFERENCE_FP16_MANTISSA_MASK));
}
/* One element of a row's input, exact in bfloat16. Its block of 128 is plain (multiples of 1/64 in -2 .. 2), spread
 * (powers of two over 2^17), zeros of both signs (a fed block of no scale), single (one value: every rotated value
 * is the block's largest) or tiny (plain over 2^16: under the floor of an E4M3 row). */
static float trx_input_value(uint32_t row, uint32_t column) {
    const uint32_t seed = TRX_INPUT_SEED + row * TR_HIDDEN_SIZE, bits = trx_mix(seed + column);
    const uint32_t kind = trx_mix(~seed + column / TR_HADAMARD) % TRX_INPUT_KINDS;
    const float sign = ((bits >> TRX_SIGN_BIT) & 1u) != 0u ? -1.0f : 1.0f;
    if (kind == TRX_INPUT_SPREAD) { return ldexpf(sign, -(int)(bits % TRX_SPREAD_POWERS)); }
    if (kind == TRX_INPUT_ZEROS || (kind == TRX_INPUT_SINGLE && column % TR_HADAMARD != 0u)) { return sign * 0.0f; }
    return ((float)(bits & TRX_INPUT_STEPS) - TRX_INPUT_OFFSET) / TRX_INPUT_DIVISOR
           / (float)(1u << (kind == TRX_INPUT_TINY ? (uint32_t)TRX_INPUT_TINY_SHIFT : 0u));
}

/* ---- routes, a step's pairs and jobs, and its validation (check 10) ---- */
static uint32_t trx_rows(const struct probe_case *probe_case) {
    return (uint32_t)probe_case_value(probe_case, TRX_FLAG_ROWS);
}
static uint32_t trx_overlap(const struct probe_case *probe_case) {
    return (uint32_t)probe_case_value(probe_case, TRX_FLAG_OVERLAP) * TRX_OVERLAP_STEP;
}
static uint32_t trx_span(uint32_t overlap, uint32_t rows) {    /* the experts a step of `rows` rows reads */
    return TR_EXPERTS_PER_ROW + (rows - 1u) * (TR_EXPERTS_PER_ROW * (TRX_PERCENT - overlap) / TRX_PERCENT);
}
/* The synthetic routing. Row r takes six consecutive experts from the 6r-th (overlap 0), the 3r-th (50) or the first
 * (100) of its step's span, and the spans of consecutive steps follow one another around the ring of experts:
 * between two reads of an expert a card reads 80 others or more (CONVENTIONS 3.5). span_rows is the case's rows
 * also where one row is launched alone: it keeps its experts. */
static uint32_t trx_expert(uint64_t step, uint32_t overlap, uint32_t span_rows, uint32_t row, uint32_t slot) {
    const uint32_t stride = TR_EXPERTS_PER_ROW * (TRX_PERCENT - overlap) / TRX_PERCENT;
    return (uint32_t)((step * trx_span(overlap, span_rows) + row * stride + slot) % TR_EXPERT_COUNT);
}
static float trx_route_weight(uint64_t step, uint32_t row, uint32_t slot) {
    const uint32_t place = (uint32_t)step * TRX_STEP_STRIDE + row * TR_EXPERTS_PER_ROW + slot;
    return (float)(1u + trx_mix(~(uint32_t)TRX_ROUTE_SEED + place) % TRX_ROUTE_WEIGHT_STEPS) / TRX_ROUTE_WEIGHT_DIVISOR;
}
/* The pairs and jobs of rows first_row .. first_row + row_count - 1, which fill the step from its pair 0: a job per
 * expert and matrix, its columns in row order. Returns the experts: the jobs of down, half those of gate and up. */
static uint32_t trx_build(uint64_t step, const struct probe_case *probe_case, uint32_t first_row, uint32_t row_count,
                          struct trx_table *table) {
    uint32_t experts = 0;
    memset(table, 0, sizeof(*table));
    for (uint32_t pair = 0; pair < row_count * TR_EXPERTS_PER_ROW; pair++) {
        const uint32_t row = first_row + pair / TR_EXPERTS_PER_ROW, slot = pair % TR_EXPERTS_PER_ROW;
        const uint32_t expert = trx_expert(step, trx_overlap(probe_case), trx_rows(probe_case), row, slot);
        const struct tr_pair made = {row, expert, trx_route_weight(step, row, slot)};
        uint32_t job = 0;
        while (job < experts && table->jobs[TR_MOST_GATE_UP_JOBS + job].matrix != expert * TR_MATRIX_KINDS + TR_DOWN) {
            job++;
        }
        experts += job == experts ? 1u : 0u;
        table->pairs[pair] = made;
        for (uint32_t kind = 0; kind < TR_MATRIX_KINDS; kind++) {
            struct tr_job *matrix = kind == TR_DOWN ? &table->jobs[TR_MOST_GATE_UP_JOBS + job]
                                                    : &table->jobs[job * TR_GATE_UP_MATRICES + kind];
            const uint32_t vector = kind == TR_DOWN ? pair : pair * TR_GATE_UP_MATRICES + kind;
            matrix->matrix = expert * TR_MATRIX_KINDS + kind;
            for (uint32_t column = 0; column < TR_MOST_ROWS && matrix->row_count == 0; column++) {
                matrix->vector[column] = vector;             /* a column not in use is fed what column 0 is fed */
            }
            matrix->vector[matrix->row_count++] = vector;
        }
    }
    return experts;
}
/* A product's block is one team on one item; round:table's, which are few and stay, hold the teams their bound lets. */
static uint32_t trx_teams(const struct trx_variant *variant, uint32_t stage) {
    const uint32_t team_threads = variant->slices[stage] * TR_LANES_PER_WARP;
    return variant->form == TR_FORM_EXLR8_TABLE ? TR_TABLE_BLOCK_THREADS / team_threads : 1u;
}
static uint64_t trx_blocks(const struct trx_variant *variant, uint32_t stage, uint64_t items) {
    const uint64_t blocks = items / trx_teams(variant, stage);
    return variant->form == TR_FORM_EXLR8_TABLE && trx.table_blocks < blocks ? trx.table_blocks : blocks;
}
/* Check 10: a step of this table with one instance's kernels, prepared in a stream's slot, stays inside every array
 * as allocated, inside the card's limits and its block's shared memory. 1: it does; 0: it printed why not, and no
 * job or pair was read past the first value out of its range. */
static int trx_step_is_valid(uint32_t instance, const struct trx_table *table, uint64_t experts, uint64_t slot,
                             uint64_t first_row, uint64_t row_count) {
    const struct trx_variant *variant = trx_used[instance];
    const struct tr_step *card = &trx.card;
    const char *kernel = variant->name;
    const uint64_t pairs = row_count * TR_EXPERTS_PER_ROW, vectors = pairs * TR_GATE_UP_MATRICES;
    const uint64_t hidden_bytes = TR_HIDDEN_SIZE * sizeof(float), inner_bytes = TR_INNER_SIZE * sizeof(float);
    const uint64_t input_scales = TR_HIDDEN_SIZE / TR_HADAMARD * TR_BLOCK_SCALES * sizeof(float);
    const uint64_t hidden_scales = TR_INNER_SIZE / TR_HADAMARD * TR_BLOCK_SCALES * sizeof(float);
    const struct { const void *array; const char *name; uint64_t first, count, bytes; } ranges[] = {
        {card->stored, "stored, in experts", 0, TR_EXPERT_COUNT, TR_EXPERT_WORDS(trx.bits) * sizeof(uint32_t)},
        {card->sides, "side vectors, in experts", 0, TR_EXPERT_COUNT, TR_SIDE_VALUES * sizeof(uint16_t)},
        {card->table, "the table, in states", 0, EXLR8_TABLE_BYTES, 1},
        {card->inputs, "inputs, in rows", first_row, row_count, hidden_bytes},
        {card->feed_input, "the fed inputs, in vectors", 0, vectors, hidden_bytes},
        {card->scales_input, "their blocks' scales, in vectors", 0, vectors, input_scales},
        {card->sums_inner, "the gate and up sums, in vectors", 0, vectors, inner_bytes},
        {card->hidden, "hidden, in pairs", 0, pairs, inner_bytes},
        {card->feed_hidden, "the fed hidden vectors, in pairs", 0, pairs, inner_bytes},
        {card->scales_hidden, "their blocks' scales, in pairs", 0, pairs, hidden_scales},
        {card->sums_outer, "the down sums, in pairs", 0, pairs, hidden_bytes},
        {card->row_output, "the rows' outputs, in rows", 0, row_count, hidden_bytes},
        {trx.tables, "the steps' tables, in slots", slot, 1, sizeof(struct trx_table)}};
    int valid = 1;
    probe_launch_value(&valid, kernel, "rows", row_count, 1, TR_MOST_ROWS);
    probe_launch_value(&valid, kernel, "the first row with the rows", first_row + row_count, 1, TR_MOST_ROWS);
    probe_launch_value(&valid, kernel, "the slot of a stream", slot, 0, PROBE_STREAM_STEPS - 1u);
    probe_launch_value(&valid, kernel, "experts", experts, 1, valid ? pairs : 1);
    for (uint64_t pair = 0; valid && pair < pairs; pair++) {
        const struct tr_pair *checked = &table->pairs[pair];
        probe_launch_value(&valid, kernel, "a pair's row", checked->row, first_row, first_row + row_count - 1u);
        probe_launch_value(&valid, kernel, "a pair's expert", checked->expert, 0, TR_EXPERT_COUNT - 1u);
    }
    for (uint64_t job = 0; valid && job < experts * TR_MATRIX_KINDS; job++) {       /* gate and up, then down */
        const int down = job >= experts * TR_GATE_UP_MATRICES;
        const struct tr_job *checked = &table->jobs[down ? TR_MOST_GATE_UP_JOBS + job % experts : job];
        probe_launch_value(&valid, kernel, "a job's matrix", checked->matrix, 0, TR_EXPERT_COUNT * TR_MATRIX_KINDS - 1);
        probe_launch_value(&valid, kernel, "a job's columns", checked->row_count, 1, row_count);
        for (uint32_t column = 0; column < TR_MOST_ROWS; column++) {
            probe_launch_value(&valid, kernel, "a column's vector", checked->vector[column], 0,
                               (down ? pairs : vectors) - 1u);
        }
    }
    for (uint32_t stage = 0; valid && stage < TR_STAGES; stage++) {
        const uint64_t items = experts * (stage == TR_DOWN_STAGE ? 1u : (uint32_t)TR_GATE_UP_MATRICES)
                               * TR_STAGE_ITEMS(stage);
        const uint64_t slices = variant->slices[stage], teams = trx_teams(variant, stage);
        const uint64_t threads = teams * slices * TR_LANES_PER_WARP, blocks = trx_blocks(variant, stage, items);
        probe_launch_value(&valid, kernel, "slices", slices, 1, TR_STAGE_INPUT_SIZE(stage) / TR_HADAMARD);
        probe_launch_value(&valid, kernel, "items beyond whole blocks of teams", items % teams, 0, 0);
        probe_launch_geometry(&valid, kernel, stage == 0 ? &trx_instances[instance] : NULL, blocks, 1, 1, threads,
                              TR_SHARED_BYTES(TR_FORM_MUL1_FP16, threads));
        probe_launch_value(&valid, kernel, "shared bytes of a block, its table with them",
                           TR_SHARED_BYTES(variant->form, threads), 0, trx.shared_most);
    }
    probe_launch_geometry(&valid, "tr_rotate_input_kernel", NULL, TR_HIDDEN_SIZE / TR_HADAMARD, vectors, 1,
                          TR_HADAMARD, TR_INPUT_SCRATCH * sizeof(float));
    probe_launch_geometry(&valid, "tr_rotate_middle_kernel", NULL, TR_INNER_SIZE / TR_HADAMARD, pairs, 1, TR_HADAMARD,
                          TR_MIDDLE_SCRATCH * sizeof(float));
    probe_launch_geometry(&valid, "tr_rotate_output_kernel", NULL, TR_HIDDEN_SIZE / TR_HADAMARD, row_count, 1,
                          TR_OUTPUT_THREADS, TR_OUTPUT_SCRATCH * sizeof(float));
    for (uint32_t range = 0; range < PROBE_COUNT_OF(ranges); range++) {
        probe_launch_range(&valid, kernel, ranges[range].name, ranges[range].first, ranges[range].count,
                           ranges[range].bytes, probe_cuda_array_bytes(ranges[range].array));
    }
    return valid;
}
/* A validated step into a stream's slot: its table to the card, its launches' arguments. */
static void trx_set_step(uint32_t instance, uint32_t slot, const struct trx_table *table, uint32_t experts,
                         uint32_t row_count) {
    struct trx_step *step = &trx.steps[slot];
    struct trx_table *slot_table = trx.tables + slot;
    PROBE_CHECK_CUDA(cudaMemcpy(slot_table, table, sizeof(*table), cudaMemcpyHostToDevice));
    PROBE_CHECK_CUDA(cudaDeviceSynchronize());   /* a step starts on an idle card, as the earlier probes' */
    step->launch = trx.card;
    step->launch.pairs = slot_table->pairs;
    step->launch.jobs = slot_table->jobs;
    step->row_count = row_count;
    for (uint32_t stage = 0; stage < TR_STAGES; stage++) {
        const uint32_t jobs = experts * (stage == TR_DOWN_STAGE ? 1u : (uint32_t)TR_GATE_UP_MATRICES);
        step->launch.item_count[stage] = jobs * TR_STAGE_ITEMS(stage);
        step->blocks[stage] = (uint32_t)trx_blocks(trx_used[instance], stage, step->launch.item_count[stage]);
    }
}

/* ---- launches ---- */
template <int feed>
static void trx_rotate_input(const struct tr_step *step, uint32_t vectors) {
    tr_rotate_input_kernel<feed><<<dim3(TR_HIDDEN_SIZE / TR_HADAMARD, vectors), TR_HADAMARD>>>(*step);
}
template <int feed>
static void trx_rotate_middle(const struct tr_step *step, uint32_t pairs) {
    tr_rotate_middle_kernel<feed><<<dim3(TR_INNER_SIZE / TR_HADAMARD, pairs), TR_HADAMARD>>>(*step);
}
static void (*const trx_rotate[TR_FEEDS][TR_STAGES])(const struct tr_step *, uint32_t) = {
    {trx_rotate_input<TR_FEED_FP16>, trx_rotate_middle<TR_FEED_FP16>},
    {trx_rotate_input<TR_FEED_INT8>, trx_rotate_middle<TR_FEED_INT8>},
    {trx_rotate_input<TR_FEED_FP8>, trx_rotate_middle<TR_FEED_FP8>}};
/* One of the five launches of a step that trx_set_step prepared. */
static void trx_launch(const struct trx_variant *variant, const struct trx_step *step, uint32_t launch) {
    const uint32_t pairs = step->row_count * TR_EXPERTS_PER_ROW, stage = launch / TRX_STAGE_LAUNCHES;
    if (launch == TRX_GATE_UP || launch == TRX_DOWN) {
        const dim3 block(variant->slices[stage] * TR_LANES_PER_WARP, trx_teams(variant, stage));    /* a team; teams */
        variant->kernels.launch[trx.width][stage](&step->launch, step->blocks[stage], block,
                                                  TR_SHARED_BYTES(variant->form, block.x * block.y));
    } else if (launch == TRX_ROTATE_OUTPUT) {
        const dim3 grid(TR_HIDDEN_SIZE / TR_HADAMARD, step->row_count);
        tr_rotate_output_kernel<<<grid, TR_OUTPUT_THREADS>>>(step->launch);
    } else {
        trx_rotate[TR_FORM_FEED(variant->form)][stage](&step->launch, stage == 0 ? pairs * TR_GATE_UP_MATRICES : pairs);
    }
    PROBE_AFTER_LAUNCH();
}
static int trx_prepare_step(void *, const struct probe_case *probe_case, uint32_t instance, uint64_t step,
                            uint32_t slot) {
    static struct trx_table table;
    const uint32_t rows = trx_rows(probe_case), experts = trx_build(step, probe_case, 0, rows, &table);
    if (!trx_step_is_valid(instance, &table, experts, slot, 0, rows)) { return 0; }
    trx_set_step(instance, slot, &table, experts, rows);
    return 1;
}
static uint32_t trx_launch_step(void *, const struct probe_case *, uint32_t instance, uint32_t slot) {
    uint32_t launched = 0;
    for (uint32_t launch = 0; launch < TRX_STEP_LAUNCHES; launch++) {      /* a stage's product: its last launch */
        if (trx.part == TRX_PART_EXPERTS || (launch % TRX_STAGE_LAUNCHES != 0u) == (trx.part == TRX_PART_PRODUCTS)) {
            trx_launch(trx_used[instance], &trx.steps[slot], launch);
            launched++;
        }
    }
    return launched;
}

/* ---- the card ---- */
static void *trx_allocate(uint64_t bytes) {      /* zeroed: no kernel reads a byte nothing wrote */
    void *array = probe_cuda_allocate(bytes);
    PROBE_CHECK_CUDA(cudaMemset(array, 0, bytes));
    return array;
}
static int trx_open_card(void *, struct probe_card_facts *facts) {
    static float inputs[TR_MOST_ROWS * TR_HIDDEN_SIZE];
    const uint64_t expert_bytes = TR_EXPERT_WORDS(trx.bits) * sizeof(uint32_t);
    const uint64_t stored_bytes = TR_EXPERT_COUNT * expert_bytes;
    const uint64_t table_bytes = TR_SHARED_BYTES(TR_FORM_EXLR8_TABLE, TR_TABLE_BLOCK_THREADS);
    const uint64_t input_bytes = TR_MOST_PAIRS * TR_GATE_UP_MATRICES * TR_HIDDEN_SIZE * sizeof(float);
    const uint64_t pair_bytes = (uint64_t)TR_MOST_PAIRS * TR_INNER_SIZE * sizeof(float);
    int valid = 1;
    if (!probe_cuda_open(facts)) { return 0; }
    const uint32_t fill_blocks = (uint32_t)facts->sm_count * TRX_FILL_BLOCKS_PER_SM;
    PROBE_REQUIRE((TR_EXPERT_COUNT - TR_MOST_PAIRS) * expert_bytes >= TRX_COLD_L2_SIZES * facts->l2_bytes,
                  "between two reads of an expert a card must read four times its L2 (CONVENTIONS 3.5)");
    trx.shared_most = probe_cuda.properties.sharedMemPerBlockOptin;
    trx.card.stored = (uint32_t *)probe_cuda_allocate(stored_bytes);
    trx.card.sides = (uint16_t *)trx_allocate(sizeof(trx_sides));
    trx.card.table = (uint8_t *)trx_allocate(sizeof(trx.codes));
    trx.card.inputs = (float *)trx_allocate(sizeof(inputs));
    trx.card.feed_input = (uint8_t *)trx_allocate(input_bytes);
    trx.card.scales_input = (float *)trx_allocate(input_bytes / TR_HADAMARD * TR_BLOCK_SCALES);
    trx.card.sums_inner = (float *)trx_allocate(pair_bytes * TR_GATE_UP_MATRICES);
    trx.card.hidden = (float *)trx_allocate(pair_bytes);
    trx.card.feed_hidden = (uint8_t *)trx_allocate(pair_bytes);
    trx.card.scales_hidden = (float *)trx_allocate(pair_bytes / TR_HADAMARD * TR_BLOCK_SCALES);
    trx.card.sums_outer = (float *)trx_allocate(input_bytes / TR_GATE_UP_MATRICES);
    trx.card.row_output = (float *)trx_allocate(sizeof(inputs));
    trx.tables = (struct trx_table *)trx_allocate(PROBE_STREAM_STEPS * sizeof(struct trx_table));
    trx.codebook_registers = (uint32_t *)trx_allocate(TR_REFERENCE_STATES * sizeof(uint32_t));
    probe_launch_geometry(&valid, "setup_fill_kernel", NULL, fill_blocks, 1, 1, TRX_FILL_THREADS, 0);
    probe_launch_range(&valid, "setup_fill_kernel", "stored, in words", 0, stored_bytes / sizeof(uint32_t),
                       sizeof(uint32_t), probe_cuda_array_bytes(trx.card.stored));
    PROBE_REQUIRE(valid, "the launch that fills the stored experts was refused");
    setup_fill_kernel<<<fill_blocks, TRX_FILL_THREADS>>>((uint32_t *)trx.card.stored, stored_bytes / sizeof(uint32_t),
                                                         trx.bits);
    for (uint32_t index = 0; index < TR_MOST_ROWS * TR_HIDDEN_SIZE; index++) {
        inputs[index] = trx_input_value(index / TR_HIDDEN_SIZE, index % TR_HIDDEN_SIZE);
    }
    PROBE_CHECK_CUDA(cudaMemcpy((void *)trx.card.inputs, inputs, sizeof(inputs), cudaMemcpyHostToDevice));
    PROBE_CHECK_CUDA(cudaMemcpy((void *)trx.card.sides, trx_sides, sizeof(trx_sides), cudaMemcpyHostToDevice));
    PROBE_CHECK_CUDA(cudaMemcpy((void *)trx.card.table, trx.codes, sizeof(trx.codes), cudaMemcpyHostToDevice));
    /* round:table's blocks stay: as many as the SMs hold at once by registers, threads and shared memory, for the
     * larger block of the two stages */
    for (uint32_t stage = 0; stage < TR_STAGES && trx.codebook == TRX_EXLR8; stage++) {
        int resident = 0;
        PROBE_CHECK_CUDA(trx_admit_table[trx.width][stage](table_bytes, TR_TABLE_BLOCK_THREADS, &resident));
        const uint64_t blocks = (uint64_t)facts->sm_count * (uint64_t)(resident > 0 ? resident : 0);
        trx.table_blocks = stage == 0 || blocks < trx.table_blocks ? blocks : trx.table_blocks;
    }
    PROBE_CHECK_CUDA(cudaGetLastError());
    PROBE_CHECK_CUDA(cudaDeviceSynchronize());
    facts->ring_bytes = stored_bytes;
    return 1;
}
static void trx_close_card(void *) { probe_cuda_close(); }
static void trx_describe_case(void *, const struct probe_case *probe_case, struct probe_case_facts *facts) {
    static const char *const parts[] = {"experts", "rotations", "products"};
    facts->format = trx_format;
    facts->part = parts[trx.part];
    facts->rows = trx_rows(probe_case);
    facts->overlap = trx_overlap(probe_case);
    facts->experts = trx_span(trx_overlap(probe_case), trx_rows(probe_case));
    facts->megabytes = (double)(trx.part == TRX_PART_ROTATIONS ? 0 : facts->experts)       /* the weights it reads */
                       * (double)(TR_EXPERT_WORDS(trx.bits) * sizeof(uint32_t)) / TRX_BYTES_PER_MB;
}
static int trx_configure(void *) {
    static const char *const formats[] = {"exl3_mul1", "exl3_mcg", "exlr8"};
    uint32_t count = 0;
    if (probe_flag(TRX_FLAG_CODEBOOK)->value_count != 1 || probe_flag(TRX_FLAG_PART)->value_count != 1
        || probe_flag(TRX_FLAG_BREAK)->value_count > 1) {
        fprintf(stderr, TRX_FLAG_CODEBOOK ", " TRX_FLAG_PART " and " TRX_FLAG_BREAK " take one value each: a run "
                        "times one format against its own baseline and breaks one check\n");
        return 0;
    }
    trx.codebook = (int)probe_flag_number(TRX_FLAG_CODEBOOK, 0);
    trx.bits = (uint32_t)probe_flag_number(TRX_FLAG_BITS, 0);
    trx.width = trx.bits - TR_LEAST_BITS;
    trx.part = (int)probe_flag_number(TRX_FLAG_PART, 0);
    trx.fault = probe_flag_given(TRX_FLAG_BREAK) ? (int)probe_flag_number(TRX_FLAG_BREAK, 0) + 1 : (int)TRX_FAULT_NONE;
    if (trx.fault == TRX_FAULT_EXACTNESS && trx.codebook != TRX_MUL1) {
        fprintf(stderr, TRX_FLAG_BREAK " exactness: the twin kernel exists for " TRX_FLAG_CODEBOOK " mul1 only\n");
        return 0;
    }
    probe_run.breaking |= (uint32_t)probe_flag_given(TRX_FLAG_BREAK);     /* as a --break-* switch of the harness */
    PROBE_REQUIRE(snprintf(trx_format, sizeof(trx_format), "%s_k%u", formats[trx.codebook], trx.bits) > 0, "format");
    for (uint32_t state = 0; state < TR_REFERENCE_STATES; state++) {
        uint32_t code = 0;
        trx.values[state] = tr_reference_weight(trx.codebook, state, &code);
        trx.codes[state] = (uint8_t)code;
        trx.byte_sums[state] = (uint16_t)tr_reference_byte_sum(state);
    }
    for (uint32_t index = 0; index < PROBE_COUNT_OF(trx_sides); index++) { trx_sides[index] = trx_side_bits(index); }
    for (uint32_t index = 0; index < PROBE_COUNT_OF(trx_variants); index++) {
        const struct trx_variant *variant = &trx_variants[index];
        const int float_form = TR_FORM_FEED(variant->form) == TR_FEED_FP16 && variant->form != TR_FORM_NO_DECODE;
        const uint32_t copies = variant->codebook == trx.codebook ? (count == 0 ? (uint32_t)TRX_BASELINE_COPIES : 1u)
                                : variant->form == TR_FORM_NO_DECODE && trx.part == TRX_PART_PRODUCTS ? 1u : 0u;
        for (uint32_t copy = 0; copy < copies; copy++) {
            const uint32_t bound = variant->form == TR_FORM_EXLR8_TABLE ? (uint32_t)TR_TABLE_BLOCK_THREADS : 0u;
            const struct probe_instance instance = {
                count == PROBE_BASELINE_AGAIN ? "baseline_again" : variant->name, variant->kernels.names[trx.width],
                float_form ? 'C' : 'F', variant->kernels.launch[trx.width][0] != NULL,
                trx_teams(variant, 0) * variant->slices[0] * TR_LANES_PER_WARP, bound, bound != 0u ? 1u : 0u, 0, 0};
            trx_used[count] = variant;
            trx_instances[count++] = instance;
        }
    }
    probe_run.program->instance_count = trx_instance_count = count;
    return 1;
}

/* ---- checks ---- */
static uint64_t trx_differing(const void *first, const void *second, size_t count, size_t bytes) {
    uint64_t differing = 0;
    for (size_t at = 0; at < count * bytes; at += bytes) {
        differing += memcmp((const char *)first + at, (const char *)second + at, bytes) != 0 ? 1u : 0u;
    }
    return differing;
}
/* Rows first_row .. of one result against rows second_row .. of another: their outputs and their hidden vectors. */
static uint64_t trx_rows_differing(const struct trx_result *first, uint32_t first_row, const struct trx_result *second,
                                   uint32_t second_row, uint32_t row_count) {
    const size_t row_hidden = (size_t)TR_EXPERTS_PER_ROW * TR_INNER_SIZE, row_floats = TR_HIDDEN_SIZE;
    return trx_differing(first->rows + first_row * row_floats, second->rows + second_row * row_floats,
                         row_count * row_floats, sizeof(float))
           + trx_differing(first->hidden + first_row * row_hidden, second->hidden + second_row * row_hidden,
                           row_count * row_hidden, sizeof(float));
}
static void trx_read(void *host, const void *card, size_t first_byte, size_t bytes) {
    PROBE_CHECK_CUDA(cudaMemcpy(host, (const char *)card + first_byte, bytes, cudaMemcpyDeviceToHost));
}
/* What half a register must hold of a state, by the definition: the fp16 of the format's value (round:table: of
 * 2^-8 times it); operand:fp8: its E4M3 code. */
static uint32_t trx_expected_half(int form, uint32_t state) {
    const double value = trx.values[state];
    if (form == TR_FORM_EXLR8_FP8) { return trx.codes[state]; }
    return tr_reference_half_bits(form == TR_FORM_EXLR8_TABLE ? ldexp(value, -TRX_PLACED_EXPONENT) : value);
}
template <int form>
static void trx_codebook(uint32_t *registers, const uint8_t *table) {
    setup_codebook_kernel<form><<<TR_REFERENCE_STATES / TRX_CODEBOOK_THREADS, TRX_CODEBOOK_THREADS>>>(registers, table);
}
/* Check 1, once a run: every form the run's instances decode by, all 65,536 states. --break codebook: the host
 * reads its codebook one state off. */
static void trx_check_codebook(const char *case_name) {
    static void (*const launch[TR_FORMS])(uint32_t *, const uint8_t *) = {
        trx_codebook<TR_FORM_MUL1_FP16>,   NULL,                             trx_codebook<TR_FORM_MCG_FP16>,
        trx_codebook<TR_FORM_EXLR8_ROUND>, trx_codebook<TR_FORM_EXLR8_TABLE>, trx_codebook<TR_FORM_EXLR8_CVT>,
        trx_codebook<TR_FORM_EXLR8_FP8>,   NULL};
    static uint32_t on_card[TR_REFERENCE_STATES];
    const char *kernel = "setup_codebook_kernel";
    const uint32_t off = trx.fault == TRX_FAULT_CODEBOOK ? 1u : 0u, blocks = TR_REFERENCE_STATES / TRX_CODEBOOK_THREADS;
    uint32_t checked_forms = 0;
    int valid = 1;
    probe_launch_geometry(&valid, kernel, NULL, blocks, 1, 1, TRX_CODEBOOK_THREADS, 0);
    probe_launch_range(&valid, kernel, "registers, in states", 0, TR_REFERENCE_STATES, sizeof(uint32_t),
                       probe_cuda_array_bytes(trx.codebook_registers));
    probe_launch_range(&valid, kernel, "the table, in states", 0, TR_REFERENCE_STATES, 1,
                       probe_cuda_array_bytes(trx.card.table));
    PROBE_REQUIRE(valid, "the codebook's launch was refused");
    for (uint32_t index = 0; index < trx_instance_count; index++) {
        const int form = trx_used[index]->form;
        const uint32_t shift = form == TR_FORM_EXLR8_FP8 ? (uint32_t)TR_REFERENCE_BYTE_BITS : (uint32_t)TR_HALF_BITS;
        uint64_t differing = 0;
        if (!trx_instances[index].available || launch[form] == NULL || ((checked_forms >> form) & 1u) != 0u) {
            continue;
        }
        checked_forms |= 1u << form;
        PROBE_CHECK_CUDA(cudaMemset(trx.codebook_registers, TRX_STALE_BYTE, sizeof(on_card)));     /* no form's words */
        launch[form](trx.codebook_registers, trx.card.table);
        PROBE_CHECK_CUDA(cudaGetLastError());
        PROBE_AFTER_LAUNCH();
        trx_read(on_card, trx.codebook_registers, 0, sizeof(on_card));
        for (uint32_t state = 0; state < TR_REFERENCE_STATES; state++) {
            const uint32_t low = (state ^ off) & TR_STATE_MASK;
            const uint32_t upper = trx_expected_half(form, low ^ TR_STATE_MASK);
            differing += on_card[state] != (trx_expected_half(form, low) | (upper << shift)) ? 1u : 0u;
        }
        probe_check_count(case_name, trx_used[index]->name, "codebook_states", differing, TR_REFERENCE_STATES, off);
    }
}
/* The states of one matrix as [input][output], from the file's tiles by the definition. A fault (a --break of the
 * host's side) is put in here: the states one off, the first states of a tile without the tile's end, or the two
 * outputs of a lane exchanged. */
static void trx_decode_matrix(uint32_t expert, uint32_t kind, int fault, uint16_t *states) {
    const uint32_t output_size = kind == TR_DOWN ? TR_HIDDEN_SIZE : TR_INNER_SIZE, output_tiles = output_size / TR_TILE;
    const uint32_t tile_words = TR_WORDS_PER_BIT * trx.bits;
    for (uint32_t tile = 0; tile < TR_MATRIX_TILES; tile++) {              /* the file's order: input tiles first */
        const uint64_t first = ((uint64_t)(expert * TR_MATRIX_KINDS + kind) * TR_MATRIX_TILES + tile) * tile_words;
        uint16_t *tile_at = states + (size_t)(tile / output_tiles) * TR_TILE * output_size
                            + tile % output_tiles * TR_TILE;
        uint32_t words[TR_WORDS_PER_BIT * TRX_MOST_BITS] = {0};
        uint16_t tile_states[TR_TILE_WEIGHTS];
        for (uint32_t word = 0; word < tile_words; word++) { words[word] = trx_file_word(first + word); }
        tr_reference_tile_states(trx.bits, words, tile_states);
        for (uint32_t position = 0; position < TR_TILE_WEIGHTS; position++) {
            const uint32_t own_bits = (position + 1u) * trx.bits;          /* of its state, before the tile's end */
            uint32_t input = 0, output = 0, state = tile_states[position] ^ (fault == TRX_FAULT_CODEBOOK ? 1u : 0u);
            tr_reference_place(fault == TRX_FAULT_PLACEMENT ? position ^ TR_HALF_WEIGHTS : position, &input, &output);
            if (fault == TRX_FAULT_WRAP && own_bits < TR_STATE_BITS) { state &= (1u << own_bits) - 1u; }
            tile_at[(size_t)input * output_size + output] = (uint16_t)state;
        }
    }
}
/* Check 3 for one vector. values: the vector before its rotation (already times its side vector), `size` of them;
 * rotated here in the kernels' order, then fed by the feed's rule (tr_feed), every product one fp32 rounding as on
 * the card. The feed's order is written here a second time: input i of a tile lies at lane place i % 8 / 2, as
 * unit 2 (i / 8) + i % 2 of the place, an integer as word 2 (i % 2) + i / 8. Returns how many fed values and block
 * scales differ from the card's, adds their number to *compared, and fills what the card feeds. */
static uint64_t trx_check_feed(int feed, float *values, uint32_t size, const uint8_t *feeds, const float *scales,
                               size_t vector, struct trx_fed *fed, uint64_t *compared) {
    static uint8_t on_card[TR_HIDDEN_SIZE * TR_FEED_BYTES];
    const size_t unit = TR_FEED_UNIT(feed), scale_bytes = size / TR_HADAMARD * TR_BLOCK_SCALES * sizeof(float);
    uint64_t differing = 0;
    trx_read(on_card, feeds, vector * size * TR_FEED_BYTES, (size_t)size * TR_FEED_BYTES);
    trx_read(fed->scales, scales, vector * scale_bytes, scale_bytes);
    for (uint32_t block = 0; block < size / TR_HADAMARD; block++) {
        float *rotated = values + (size_t)block * TR_HADAMARD, largest = 0.0f, block_sum = 0.0f;
        float expected[TR_BLOCK_SCALES] = {0.0f, 0.0f};
        int power = 0;
        tr_reference_rotate(rotated);
        for (uint32_t index = 0; index < TR_HADAMARD; index++) { largest = fmaxf(largest, fabsf(rotated[index])); }
        const float scale = largest / TR_INT8_LARGEST, per_one = largest > 0.0f ? TR_INT8_LARGEST / largest : 0.0f;
        const float ratio = frexpf(fmaxf(largest, TR_FEED_FLOOR) * TR_E4M3_LARGEST_INVERSE, &power);
        power -= ratio == TRX_POWER_OF_TWO_MANTISSA ? 1 : 0;              /* the power of two at or above the ratio */
        for (uint32_t index = 0; index < TR_HADAMARD; index++) {
            const uint32_t column = block * TR_HADAMARD + index, in_tile = column % TR_TILE;
            const uint32_t odd = in_tile % TR_REGISTER_WEIGHTS, far = in_tile / TR_HALF_INPUTS;
            const uint32_t unit_of_place = feed == TR_FEED_INT8 ? odd * TR_REGISTER_WEIGHTS + far
                                                                : far * TR_REGISTER_WEIGHTS + odd;
            const uint32_t place = in_tile % TR_HALF_INPUTS / TR_REGISTER_WEIGHTS * TR_LANES_PER_GROUP + unit_of_place;
            const uint8_t *at = on_card + (column - in_tile + place) * unit;
            uint32_t word = 0;
            memcpy(&word, at, unit);
            if (feed == TR_FEED_FP16) {
                differing += word != tr_reference_half_bits((double)rotated[index]) ? 1u : 0u;
                fed->values[column] = tr_reference_half(word);
            } else if (feed == TR_FEED_INT8) {
                const float nearest = rintf(rotated[index] * per_one);
                differing += word != ((uint32_t)(int32_t)nearest & TR_BYTE_MASK) * TR_BYTE_ONES ? 1u : 0u;
                fed->integers[column] = (int8_t)at[0];
                fed->values[column] = (double)fed->integers[column] * (double)scale;
                block_sum += nearest;
            } else {
                differing += word != tr_reference_e4m3_code(ldexp((double)rotated[index], -power)) ? 1u : 0u;
                fed->values[column] = ldexp(tr_reference_e4m3(word), power);
            }
        }
        expected[0] = feed == TR_FEED_INT8 ? scale * TR_MUL1_STEP : feed == TR_FEED_FP8 ? ldexpf(1.0f, power) : 0.0f;
        expected[1] = feed == TR_FEED_INT8 ? block_sum * TR_MUL1_OFFSET * scale : 0.0f;
        differing += feed != TR_FEED_FP16
                     && memcmp(expected, &fed->scales[block * TR_BLOCK_SCALES], sizeof(expected)) != 0 ? 1u : 0u;
    }
    *compared += size + (feed != TR_FEED_FP16 ? size / TR_HADAMARD : 0u);
    return differing;
}
/* operand:imma8's contract for one matrix, every step in its order (the header of trellis_kernels.cuh): per block
 * of 128 inputs the integer P, then T = fma(P, a, T - c) with the card's a and c of the block; T from +0 a slice,
 * the slices added in order from +0. Returns the outputs whose bits are not the card's. */
static uint64_t trx_integer_differing(const uint16_t *states, uint32_t input_size, uint32_t output_size,
                                      uint32_t slices, const struct trx_fed *fed, const float *card_sums) {
    static int32_t block_sums[TR_HIDDEN_SIZE];
    static float totals[TR_HIDDEN_SIZE], sums[TR_HIDDEN_SIZE];
    const uint32_t blocks = input_size / TR_HADAMARD;
    memset(sums, 0, sizeof(sums));
    for (uint32_t slice = 0; slice < slices; slice++) {
        memset(totals, 0, sizeof(totals));
        for (uint32_t block = slice * blocks / slices; block < (slice + 1u) * blocks / slices; block++) {
            const float *scale = &fed->scales[block * TR_BLOCK_SCALES];
            memset(block_sums, 0, sizeof(block_sums));
            for (uint32_t input = block * TR_HADAMARD; input < (block + 1u) * TR_HADAMARD; input++) {
                const uint16_t *weights = states + (size_t)input * output_size;
                for (uint32_t output = 0; output < output_size; output++) {
                    block_sums[output] += (int32_t)trx.byte_sums[weights[output]] * fed->integers[input];
                }
            }
            for (uint32_t output = 0; output < output_size; output++) {
                totals[output] = fmaf((float)block_sums[output], scale[0], totals[output] - scale[1]);
            }
        }
        for (uint32_t output = 0; output < output_size; output++) { sums[output] += totals[output]; }
    }
    return trx_differing(sums, card_sums, output_size, sizeof(float));
}
/* Checks 3 and 4 for the step of an instance that just ran (its result read back, its arrays still on the card):
 * the two checked pairs, each stage fed with the card's own previous stage. */
static void trx_check_against_host(const struct probe_case *probe_case, uint32_t instance,
                                   const struct trx_result *result) {
    static const char *const names[TRX_CHECKED_PAIRS][TR_STAGES] = {
        {"host_reference_hidden_first_row", "host_reference_down_first_row"},
        {"host_reference_hidden_last_row", "host_reference_down_last_row"}};
    static struct trx_fed fed;
    static float values[TR_HIDDEN_SIZE], card_sums[TR_HIDDEN_SIZE], row_sums[TR_HIDDEN_SIZE];
    static double sums[TR_MATRIX_KINDS][TR_HIDDEN_SIZE];
    const struct trx_variant *variant = trx_used[instance];
    const struct tr_step *card = &trx.card;
    const char *case_name = probe_case->name;
    const uint32_t rows = trx_rows(probe_case), overlap = trx_overlap(probe_case);
    const int feed = TR_FORM_FEED(variant->form), integer = variant->form == TR_FORM_MUL1_IMMA8;
    const int host_fault = trx.fault >= TRX_FAULT_CODEBOOK && trx.fault <= TRX_FAULT_PLACEMENT;
    uint64_t rotated_differing = 0, rotated_compared = 0;
    for (uint32_t checked = 0; checked < TRX_CHECKED_PAIRS; checked++) {
        const size_t pair = checked == 0 ? TR_EXPERTS_PER_ROW - 1u : (rows - 1u) * TR_EXPERTS_PER_ROW;
        const uint32_t row = (uint32_t)pair / TR_EXPERTS_PER_ROW, slot = (uint32_t)pair % TR_EXPERTS_PER_ROW;
        const uint16_t *sides = trx_sides
                                + (size_t)trx_expert(TRX_CHECKED_STEP, overlap, rows, row, slot) * TR_SIDE_VALUES;
        const float *hidden = result->hidden + pair * TR_INNER_SIZE;
        const double weight = (double)trx_route_weight(TRX_CHECKED_STEP, row, slot), limit = (double)TR_SWIGLU_LIMIT;
        const int must_report = checked == 0 && (host_fault || probe_flag_given(PROBE_FLAG_CORRUPT));
        double error[TR_STAGES] = {0.0, 0.0}, square[TR_STAGES] = {0.0, 0.0};
        uint64_t integer_differing = 0;
        for (uint32_t kind = 0; kind < TR_MATRIX_KINDS; kind++) {          /* gate, up and the hidden step, then down */
            const int down = kind == TR_DOWN;
            const uint32_t input_size = down ? TR_INNER_SIZE : TR_HIDDEN_SIZE, output_size = TR_STAGE_OUTPUT_SIZE(down);
            const size_t vector = down ? pair : pair * TR_GATE_UP_MATRICES + kind;
            const uint16_t *states = trx_states[checked][kind];
            for (uint32_t column = 0; column < input_size; column++) {     /* what the rotation is given */
                const uint32_t side_at = (down ? (uint32_t)TR_SUH_DOWN_AT : kind * TR_HIDDEN_SIZE) + column;
                values[column] = (down ? hidden[column] : trx_input_value(row, column))
                                 * (float)tr_reference_half(sides[side_at]);
            }
            rotated_differing += trx_check_feed(feed, values, input_size, down ? card->feed_hidden : card->feed_input,
                                                down ? card->scales_hidden : card->scales_input, vector, &fed,
                                                &rotated_compared);
            memset(sums[kind], 0, sizeof(sums[kind]));                     /* the product in double precision */
            for (uint32_t input = 0; input < input_size; input++) {
                for (uint32_t output = 0; output < output_size; output++) {
                    sums[kind][output] += trx.values[states[(size_t)input * output_size + output]] * fed.values[input];
                }
            }
            trx_read(card_sums, down ? card->sums_outer : card->sums_inner, vector * output_size * sizeof(float),
                     output_size * sizeof(float));
            if (integer) {
                integer_differing += trx_integer_differing(states, input_size, output_size, variant->slices[down], &fed,
                                                           card_sums);
            }
            for (uint32_t block = 0; kind == TR_UP && block < TR_INNER_SIZE / TR_HADAMARD; block++) {
                double rotated[TR_GATE_UP_MATRICES][TR_HADAMARD];          /* the hidden vector, by the definition */
                tr_reference_hadamard(sums[TR_GATE] + (size_t)block * TR_HADAMARD, rotated[TR_GATE]);
                tr_reference_hadamard(sums[TR_UP] + (size_t)block * TR_HADAMARD, rotated[TR_UP]);
                for (uint32_t index = 0; index < TR_HADAMARD; index++) {
                    const uint32_t column = block * TR_HADAMARD + index;
                    const double gate_side = tr_reference_half(sides[TR_SVH_GATE_AT + column]);
                    const double up = rotated[TR_UP][index] * tr_reference_half(sides[TR_SVH_UP_AT + column]);
                    const double gate = fmin(rotated[TR_GATE][index] * gate_side, limit);
                    const double reference = gate / (1.0 + exp(-gate)) * fmin(fmax(up, -limit), limit) * weight;
                    error[0] += ((double)hidden[column] - reference) * ((double)hidden[column] - reference);
                    square[0] += reference * reference;
                }
            }
            for (uint32_t output = 0; down && output < TR_HIDDEN_SIZE; output++) {
                const double difference = (double)card_sums[output] - sums[kind][output];
                error[1] += difference * difference;
                square[1] += sums[kind][output] * sums[kind][output];
            }
        }
        memset(row_sums, 0, sizeof(row_sums));   /* the row's output from the card's six down products */
        for (uint32_t added = 0; added < TR_EXPERTS_PER_ROW; added++) {
            const uint32_t expert = trx_expert(TRX_CHECKED_STEP, overlap, rows, row, added);
            const uint16_t *svh = trx_sides + (size_t)expert * TR_SIDE_VALUES + TR_SVH_DOWN_AT;
            trx_read(values, card->sums_outer, ((size_t)row * TR_EXPERTS_PER_ROW + added) * sizeof(values),
                     sizeof(values));
            for (uint32_t column = 0; column < TR_HIDDEN_SIZE; column++) {
                if (column % TR_HADAMARD == 0u) { tr_reference_rotate(values + column); }
                row_sums[column] += values[column] * (float)tr_reference_half(svh[column]);
            }
        }
        rotated_differing += trx_differing(row_sums, result->rows + (size_t)row * TR_HIDDEN_SIZE, TR_HIDDEN_SIZE,
                                           sizeof(float));
        rotated_compared += TR_HIDDEN_SIZE;
        for (uint32_t stage = 0; stage < TR_STAGES; stage++) {
            probe_check_error(case_name, variant->name, names[checked][stage], sqrt(error[stage] / square[stage]),
                              TRX_RELATIVE_TOLERANCE, must_report);
        }
        if (integer) {
            probe_check_count(case_name, variant->name, "integer_contract", integer_differing,
                              TR_GATE_UP_MATRICES * TR_INNER_SIZE + TR_HIDDEN_SIZE, must_report);
        }
    }
    probe_check_count(case_name, variant->name, "rotated_vectors", rotated_differing, rotated_compared,
                      trx.fault == TRX_FAULT_ROTATION);
}
/* The checked step of rows first_row .. with one instance's kernels (launched: the instance's, or its twin's),
 * outside the timing: validated, what it stores made stale first (a word a launch does not write cannot pass for a
 * result), the result read back. overwrite (--break rotation): after the first launch TRX_BROKEN_BYTES of the fed
 * gate input of the launch's sixth pair are overwritten (of row 0 also where it is launched alone). */
static void trx_run_checked(const struct probe_case *probe_case, uint32_t instance, const struct trx_variant *launched,
                            uint32_t first_row, uint32_t row_count, int overwrite, struct trx_result *result) {
    static struct trx_table table;
    void *const written[] = {trx.card.feed_input,  trx.card.scales_input, trx.card.sums_inner, trx.card.hidden,
                             trx.card.feed_hidden, trx.card.scales_hidden, trx.card.sums_outer, trx.card.row_output};
    const uint32_t experts = trx_build(TRX_CHECKED_STEP, probe_case, first_row, row_count, &table);
    const size_t broken_vector = (size_t)(TR_EXPERTS_PER_ROW - 1u) * TR_GATE_UP_MATRICES;
    PROBE_REQUIRE(trx_step_is_valid(instance, &table, experts, 0, first_row, row_count), "the checked step: refused");
    trx_set_step(instance, 0, &table, experts, row_count);
    for (uint32_t array = 0; array < PROBE_COUNT_OF(written); array++) {
        PROBE_CHECK_CUDA(cudaMemset(written[array], TRX_STALE_BYTE, probe_cuda_array_bytes(written[array])));
    }
    for (uint32_t launch = 0; launch < TRX_STEP_LAUNCHES; launch++) {
        trx_launch(launched, &trx.steps[0], launch);
        if (overwrite && launch == TRX_ROTATE_INPUT) {
            PROBE_CHECK_CUDA(cudaMemset(trx.card.feed_input + broken_vector * TR_HIDDEN_SIZE * TR_FEED_BYTES,
                                        TRX_BROKEN_BYTE, TRX_BROKEN_BYTES));
        }
    }
    PROBE_CHECK_CUDA(cudaGetLastError());
    PROBE_CHECK_CUDA(cudaDeviceSynchronize());
    trx_read(result->rows, trx.card.row_output, 0, sizeof(result->rows));
    trx_read(result->hidden, trx.card.hidden, 0, sizeof(result->hidden));
}
/* --break-route: eight launches, each past one bound of what a kernel indexes by or of what the host prepares: the
 * validation every real step passes must refuse each (1 of 1); none is launched. */
static void trx_break_route(const struct probe_case *probe_case) {
    static const char *const names[TRX_BROKEN_ROUTES] = {
        "refused_expert_past_the_stored",        "refused_matrix_past_the_stored", "refused_vector_past_the_vectors",
        "refused_job_of_more_columns_than_rows", "refused_more_experts_than_pairs", "refused_row_too_many",
        "refused_first_row_past_the_rows",       "refused_slot_past_the_stream"};
    static struct trx_table table;
    for (uint32_t broken = 0; broken < TRX_BROKEN_ROUTES; broken++) {
        const uint32_t rows = trx_rows(probe_case);
        uint64_t experts = trx_build(TRX_CHECKED_STEP, probe_case, 0, rows, &table);
        table.pairs[0].expert += broken == TRX_BROKEN_EXPERT ? (uint32_t)TR_EXPERT_COUNT : 0u;
        table.jobs[0].matrix += broken == TRX_BROKEN_MATRIX ? (uint32_t)TR_EXPERT_COUNT * TR_MATRIX_KINDS : 0u;
        table.jobs[0].vector[TR_MOST_ROWS - 1] += broken == TRX_BROKEN_VECTOR ? rows * TR_MOST_GATE_UP_JOBS : 0u;
        table.jobs[0].row_count += broken == TRX_BROKEN_COLUMNS ? rows : 0u;
        experts = broken == TRX_BROKEN_JOBS ? (uint64_t)rows * TR_EXPERTS_PER_ROW + 1u : experts;
        const int refused = !trx_step_is_valid(PROBE_BASELINE, &table, experts,
                                               broken == TRX_BROKEN_SLOT ? PROBE_STREAM_STEPS : 0u,
                                               broken == TRX_BROKEN_FIRST_ROW ? (uint64_t)TR_MOST_ROWS : 0u,
                                               broken == TRX_BROKEN_ROWS ? TR_MOST_ROWS + 1u : rows);
        probe_check_count(probe_case->name, trx_used[PROBE_BASELINE]->name, names[broken], refused ? 1u : 0u, 1, 1);
    }
}
static void trx_check_case(void *, const struct probe_case *probe_case) {
    static struct trx_result result, baseline_result, alone;
    const char *case_name = probe_case->name;
    const uint32_t rows = trx_rows(probe_case), overlap = trx_overlap(probe_case);
    const int corrupt = probe_flag_given(PROBE_FLAG_CORRUPT), break_exactness = trx.fault == TRX_FAULT_EXACTNESS;
    const int host_fault = trx.fault >= TRX_FAULT_CODEBOOK && trx.fault <= TRX_FAULT_PLACEMENT;
    const int overwrite = trx.fault == TRX_FAULT_ROTATION;
    const int reordered = break_exactness && overlap > 0 && rows > 1u;      /* the twin has a job of several rows */
    const size_t matrix_bytes = TR_EXPERT_WORDS(trx.bits) / TR_MATRIX_KINDS * sizeof(uint32_t);
    if (probe_flag_given(PROBE_FLAG_BREAK_ROUTE)) {
        trx_break_route(probe_case);
        return;
    }
    if (probe_case->index == 0) { trx_check_codebook(case_name); }
    for (uint32_t checked = 0; checked < TRX_CHECKED_PAIRS; checked++) {    /* the host's tiles of the checked pairs */
        const uint32_t expert = checked == 0 ? trx_expert(TRX_CHECKED_STEP, overlap, rows, 0, TR_EXPERTS_PER_ROW - 1u)
                                             : trx_expert(TRX_CHECKED_STEP, overlap, rows, rows - 1u, 0);
        uint8_t *stored = (uint8_t *)trx.card.stored + (size_t)expert * TR_MATRIX_KINDS * matrix_bytes;
        for (uint32_t kind = 0; kind < TR_MATRIX_KINDS; kind++) {
            trx_decode_matrix(expert, kind, checked == 0 && host_fault ? trx.fault : (int)TRX_FAULT_NONE,
                              trx_states[checked][kind]);
            if (corrupt && checked == 0 && kind != TR_UP) {
                PROBE_CHECK_CUDA(cudaMemset(stored + kind * matrix_bytes, TRX_CORRUPTED_BYTE, TRX_CORRUPTED_BYTES));
            }
        }
    }
    for (uint32_t index = 0; index < trx_instance_count; index++) {
        const struct trx_variant *variant = trx_used[index];
        const struct trx_variant *launched = break_exactness && index == PROBE_BASELINE ? &trx_broken_order : variant;
        const int integer = variant->form == TR_FORM_MUL1_IMMA8;
        /* a break run: the baseline; the integer form where its contract must report; exactness: a variant too */
        const int concerned = index == PROBE_BASELINE || (integer && (corrupt || host_fault))
                              || (break_exactness && index == TRX_FIRST_VARIANT);
        uint64_t differing = 0;
        if (index == PROBE_BASELINE_AGAIN || !trx_instances[index].available || variant->form == TR_FORM_NO_DECODE
            || (probe_breaking() && !concerned)) {
            continue;
        }
        trx_run_checked(probe_case, index, launched, 0, rows, overwrite, &result);
        if (index == PROBE_BASELINE) {
            memcpy(&baseline_result, &result, sizeof(result));
        } else if (variant->bits_kind != TRX_BITS_OWN) {
            differing = trx_rows_differing(&result, 0, &baseline_result, 0, rows);
            probe_check_count(case_name, variant->name, "bits_against_baseline", differing, TRX_ROW_WORDS(rows),
                              reordered);
        }
        if (index == PROBE_BASELINE || variant->bits_kind == TRX_BITS_OWN || differing != 0) {
            trx_check_against_host(probe_case, index, &result);     /* its bits are not the baseline's */
        }
        probe_digest_add(&result, sizeof(result));
        differing = 0;
        for (uint32_t row = 0; row < rows && rows > 1u; row++) {
            trx_run_checked(probe_case, index, launched, row, 1, overwrite && row == 0, &alone);
            differing += trx_rows_differing(&result, row, &alone, 0, 1);
        }
        if (rows > 1u) {                 /* a float form's count on a 170HX is the card's answer; a twin's must show */
            char *held = &trx_instances[index].contract_class;       /* R: seen where rows share a job, or not at all */
            *held = differing != 0 || *held == 'F' ? 'F' : overlap > 0 ? 'R' : *held;
            probe_check_count(case_name, variant->name, "rows_exact", differing, TRX_ROW_WORDS(rows),
                              !integer && ((PROBE_SM == PROBE_SM_80 && differing != 0)
                                           || (reordered && index == PROBE_BASELINE)));
        }
    }
    trx_instances[PROBE_BASELINE_AGAIN].contract_class = trx_instances[PROBE_BASELINE].contract_class;  /* one kernel */
    probe_cuda_check_guards(case_name);
}

/* The figures of ROUND_1.md, package 7, in a run's first case (one row): operand:imma8 over fp16 (a bar for a
 * CMP 170HX's whole step; measured elsewhere); operand:fp8 over round:cvt; with --part products the share of the
 * decode in the products (1 less decode:none over the baseline). */
static void trx_verdicts(void *, const struct probe_case *probe_case, const struct probe_result *results,
                         double noise_ratio) {
    const int barred = PROBE_SM == PROBE_SM_80 && trx.part == TRX_PART_EXPERTS;
    double cvt_us = PROBE_NAN;
    if (probe_case->index != 0) { return; }
    for (uint32_t index = TRX_FIRST_VARIANT; index < trx_instance_count; index++) {
        const struct probe_result *result = &results[index];
        const int form = trx_used[index]->form;
        if (result->mode == NULL) { continue; }
        cvt_us = form == TR_FORM_EXLR8_CVT ? result->step.median : cvt_us;
        if (form == TR_FORM_MUL1_IMMA8) {
            probe_verdict("imma8_against_fp16", barred ? TRX_IMMA8_BAR : PROBE_NAN,
                          barred ? TRX_IMMA8_BAR_US : PROBE_NAN, result->ratio, noise_ratio, result->same_sign,
                          result->rounds);
        } else if (form == TR_FORM_EXLR8_FP8) {
            probe_verdict("fp8_against_cvt", PROBE_NAN, PROBE_NAN, result->step.median / cvt_us, noise_ratio, 0, 0);
        } else if (form == TR_FORM_NO_DECODE) {
            probe_verdict("decode_share_of_the_products", PROBE_NAN, PROBE_NAN, 1.0 - result->ratio, noise_ratio, 0, 0);
        }
    }
}

static struct probe_program trx_program = {
    "trellis_expert",  trx_flags,        PROBE_COUNT_OF(trx_flags), trx_instances,     0,
    NULL,              trx_configure,    trx_open_card,             trx_describe_case, trx_check_case,
    trx_prepare_step,  trx_launch_step,  probe_cuda_mark_start,     probe_cuda_elapsed_microseconds,
    trx_verdicts,      trx_close_card,   NULL};

int main(int argument_count, char **arguments) { return probe_main(argument_count, arguments, &trx_program); }
