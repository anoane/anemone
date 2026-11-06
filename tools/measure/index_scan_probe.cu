/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * index_scan_probe: package 11a of round 1 of the speed campaign (research/speed/ROUND_1_attention.md, "index_scan";
 * ledger A1 scan, A2 select, A6 candidate scan) with the owner's 8-bit keys (DECISIONS_2026-10-12.md section 1). The
 * kernels and their contract: index_scan_kernels.cuh; the synthetic state, the bounds, the stamps, the instances and
 * figures: index_scan_probe_kernels.cuh; the host's definitions: index_scan_reference.h.
 *
 * Cases: --state (fp16, trained, fresh: fp16 HALF, trained WHOLE, fresh PLACED on sm_80 and E4M3 on sm_120a, whose
 * queries are E4M3 too) by --rows by --entries (row r of R sees entries - (R - 1 - r) / --rate) by --segment-keys by
 * --blocks-per-sm (shared memory padded so that this many blocks fit an SM; 0: none); --capacity entries a slab row. A
 * step reads the next plane of its state's ring (four L2 sizes and a plane; one L2 size after --steps without
 * --profile-case); the checks read the first. Instances, interleaved: baseline (up to 8 rows a block of 4 warps; a
 * row a block in a one-row step), warps:16, the per-row forms (rows on the slow or fast grid axis; checked, and so
 * available, in a run of 8 rows or fewer), scan:candidates (2,048 listed blocks a row, scores to [row][16,384]), the
 * bounds (the baseline's grid; unpadded its shared bytes and so its residency, padded the padding's), the two selects
 * (on the slab the baseline wrote when the step was prepared: warm), step:scan_select (spans at 16 rows or fewer and
 * 32,768 entries or more), scan:fp16 (the 8-bit bars' measure), scan:e2m1_scaled and scan:fp16_queries (sm_120a),
 * break:head_order.
 *
 * Checked per case before anything is timed (rows: all up to 8, else every 7th, 37 of 256): host_reference (sampled
 * scores against the reference in the instance's state: 2e-3 of the row's rms); extents (the baseline's -inf at or
 * beyond an extent in the written tiles, the sentinel beyond them; first case: 128, 129 and 513 against a slab row of
 * that size); integer_contract (trained, bit for bit from the card's codes; EXACT: the pairs whose chains stayed the
 * exact whole-and-scale sums); quantizer_queries (first case also quantizer_keys: --capacity keys of both 8-bit
 * rings); bits_* (0 differing words against the baseline: warps:16, the per-row forms up to 8 rows, scan:candidates,
 * scan:e2m1_scaled, the case at segment 2048 unpadded, bits_select_forms; a form that differs loses class R);
 * rows_exact; bound_sums_and_folds; selection_* (against the host's counting sort, tie rows apart; first case also 8
 * rows of one launch: 513, 16,385, 524,288, 1,048,576 and 128 visible, all zero, 600 ties at the cut, 1,000);
 * guard_bytes; DIGEST of the baseline's slab and the selections (every CMP 170HX must print the same).
 * Shown failing (a run exits 0 only if its check reported and no other did): --corrupt (a megabyte of the case's ring:
 * host_reference, the read_only bound, that ring's quantizer_keys); --break-exactness (8 rows: rows_exact of
 * break:head_order); --break-ties (the host takes ties from the top: selection_tie_rows); --break-extent (row 0's card
 * extent one higher, 2 rows or more, row 0 not its group's largest: extents; the 513 launch's: extents_exact_size);
 * --break-bits (every bits_, integer_contract and quantizer_ check); --break-route (a plane past its ring, an extent
 * past the plane, no tile a segment); --break-guard.
 * RESULT format=<state> part=<the case less state and rows> mb=<the baseline's key bytes>; RATE card= case= variant=
 * gbps= tflops= shared= resident= reads= (its bytes and products over its median; a block's shared bytes and the
 * blocks an SM holds, na for a select alone; cold_keys, warm_slab or none); VERDICT: 11a's acceptance and the 8-bit
 * bars (0.60 of the FP16 scan; on sm_120a also the fresh keys with FP16 queries), with the judged instance's pairs;
 * UNAVAILABLE card= variant= state= threads= most_threads=: an instance left out of the cases of a state whose kernel's
 * registers refuse its block (once a state; its RESULT then says n=0).
 * Measurement tool: not part of the engine.
 */
#include "probe_harness.h"
#include "index_scan_probe_kernels.cuh"
#include "index_scan_reference.h"

enum {
    INDEX_MOST_ROWS = 256, INDEX_MOST_CAPACITY = 1048576, INDEX_MOST_PADDING = 32, INDEX_RATE_MOST = 2,
    INDEX_FILL_BLOCKS_PER_SM = 8, INDEX_COLD_L2_SIZES = 4, INDEX_CHECKED_STRIDE = 7,
    INDEX_CANONICAL_SEGMENT_KEYS = 2048, INDEX_SENTINEL_BYTE = 0x7E, INDEX_SENTINEL = 0x7E7E,
    INDEX_CORRUPTED_BYTES = 1048576, INDEX_CORRUPTED_BYTE = 0x5A, INDEX_MOST_FOLDS = 4194304,
    INDEX_SPANS_MOST_ROWS = 16, INDEX_SPANS_LEAST_ENTRIES = 32768, INDEX_SPECIAL_ROWS = 8, INDEX_EXACT_SIZES = 3,
    INDEX_ZERO_ROW = 5, INDEX_TIES_ROW = 6, INDEX_TIES_ABOVE = 300, INDEX_TIES_EQUAL = 600,
    INDEX_HALF_TWO = 0x4000, INDEX_HALF_ONE = 0x3C00, INDEX_HALF_HALF = 0x3800, INDEX_NAME_BYTES = 128,
    INDEX_SHARED_ALIGNMENT = 16, INDEX_SPAN_LAUNCHES = 2,
    INDEX_QUANTIZER_PARTS = 2,                   /* a key's codes and its scale codes */
    INDEX_SELECT_NONE = 0, INDEX_SELECT_ONE, INDEX_SELECT_SPANS
};
static const uint32_t index_special_extents[INDEX_SPECIAL_ROWS] = {513, 16385, 524288, 1048576, 128, 4096, 4096, 1000};
static const uint32_t index_exact_extents[INDEX_EXACT_SIZES] = {128, 129, 513};
#define INDEX_FLAG_STATE "--state"
#define INDEX_FLAG_ROWS "--rows"
#define INDEX_FLAG_ENTRIES "--entries"
#define INDEX_FLAG_SEGMENT "--segment-keys"
#define INDEX_FLAG_PADDING "--blocks-per-sm"
#define INDEX_FLAG_CAPACITY "--capacity"
#define INDEX_FLAG_RATE "--rate"
#define INDEX_FLAG_BREAK_EXACTNESS "--break-exactness"
#define INDEX_FLAG_BREAK_TIES "--break-ties"
#define INDEX_FLAG_BREAK_EXTENT "--break-extent"
#define INDEX_FLAG_BREAK_BITS "--break-bits"
#define INDEX_TOLERANCE 2.0e-3                   /* CONVENTIONS 5.1, of a row's sampled scores against its rms */
#define INDEX_BYTES_PER_MB 1.0e6
#define INDEX_GB_PER_BYTE_MICROSECOND 1.0e-3
#define INDEX_TFLOPS_PER_FLOP_MICROSECOND 1.0e-6

static struct probe_flag index_flags[] = {
    PROBE_COMMON_FLAGS,
    PROBE_CASE_WORDS(INDEX_FLAG_STATE, "fp16,trained,fresh", "fp16", "the keys' state kind"),
    PROBE_CASE_NUMBERS(INDEX_FLAG_ROWS, 1, INDEX_MOST_ROWS, "1,8,256", "rows of a step"),
    PROBE_CASE_NUMBERS(INDEX_FLAG_ENTRIES, INDEX_KEYS_PER_TILE, INDEX_MOST_CAPACITY, "128,4096,65536,524288,1048576",
                       "entries the last row sees (whole tiles of 8)"),
    PROBE_CASE_NUMBERS(INDEX_FLAG_SEGMENT, INDEX_KEYS_PER_TILE, INDEX_MOST_CAPACITY, "2048", "keys a block's segment"),
    PROBE_CASE_NUMBERS(INDEX_FLAG_PADDING, 0, INDEX_MOST_PADDING, "0", "blocks an SM's shared memory is padded to"),
    PROBE_NUMBER(INDEX_FLAG_CAPACITY, INDEX_KEYS_PER_TILE, INDEX_MOST_CAPACITY, "1048576", "entries of a row's slab"),
    PROBE_NUMBER(INDEX_FLAG_RATE, 1, INDEX_RATE_MOST, "2", "tokens an entry, for the rows' extents"),
    PROBE_SWITCH(INDEX_FLAG_BREAK_EXACTNESS, "break:head_order joins: its rows_exact must report (8 rows)"),
    PROBE_SWITCH(INDEX_FLAG_BREAK_TIES, "the host takes the cut's ties from the top: selection_tie_rows must report"),
    PROBE_SWITCH(INDEX_FLAG_BREAK_EXTENT, "the card's extent of row 0 and of an exact-size launch one higher: extents "
                 "and extents_exact_size must report"),
    PROBE_SWITCH(INDEX_FLAG_BREAK_BITS, "a word or byte of each copy compared bit for bit changed: every bits_, "
                 "integer_contract and quantizer_ check must report"),
};

/* ---- the instances (INDEX_INSTANCES and the figures: index_scan_probe_kernels.cuh) ---- */

#define INDEX_INSTANCE_ROW(id, variant, family, keys, claimed, bits, architecture, e2m1)                             \
    {variant, NULL, claimed, (architecture == 0 || architecture == PROBE_SM) && (!(e2m1) || INDEX_E2M1_SCALED),      \
     0, 0, 0, 0, 0},
static struct probe_instance index_instances[INDEX_INSTANCE_COUNT] = {INDEX_INSTANCES(INDEX_INSTANCE_ROW)};
struct index_variant { uint32_t family, keys; int bits; };
#define INDEX_VARIANT_ROW(id, variant, family, keys, claimed, bits, ...) {family, keys, bits},
static const struct index_variant index_variants[INDEX_INSTANCE_COUNT] = {INDEX_INSTANCES(INDEX_VARIANT_ROW)};
static uint32_t index_instances_in_run = INDEX_INSTANCE_COUNT - 1u;      /* all but the one the break adds */
static char index_kernel_lists[INDEX_INSTANCE_COUNT][INDEX_NAME_BYTES * INDEX_FORMS * INDEX_BOUND_SIZES];

static struct {
    uint32_t shared_per_sm, shared_reserved, capacity, most_rows;
    uint32_t most_threads[INDEX_KERNEL_SLOTS];   /* a block's threads as each kernel's registers allow */
    uint32_t told[INDEX_INSTANCE_COUNT];         /* a bit a state whose UNAVAILABLE line was printed */
    int configured[INDEX_INSTANCE_COUNT];        /* available as index_configure left it */
    uint64_t cold_bytes;
    unsigned char *rings[INDEX_STATES], *placed[INDEX_STATES];   /* placed: FP16 (HALF, PLACED), TRAINED, E4M3 */
    uint16_t *natural, *slab, *listed_scores, *span_keys, *exact[INDEX_EXACT_SIZES];
    float *weights;
    uint32_t *extents, *lists, *folds, *span_positions, *span_counts;
    int32_t *selected;
    struct index_step steps[PROBE_STREAM_STEPS];
} index_card;
static struct {                                  /* the host's copies; extents: the host's, card_extents: the card's */
    uint32_t extents[INDEX_MOST_ROWS], card_extents[INDEX_MOST_ROWS], lists[INDEX_MOST_ROWS * INDEX_CANDIDATE_BLOCKS];
    float weights[INDEX_MOST_ROWS * INDEX_HEADS];
    unsigned char placed[INDEX_MOST_ROWS * INDEX_CODE_QUERY_BYTES];
    int32_t selected[INDEX_MOST_ROWS * INDEX_SELECTED], selected_other[INDEX_MOST_ROWS * INDEX_SELECTED];
    uint32_t folds[INDEX_MOST_FOLDS];
    uint16_t *slab, *other, *listed, *key_folds[INDEX_STATES];      /* [row][capacity], [row][16,384], [key] */
    unsigned char *plane;                        /* an 8-bit ring's first --capacity keys, read back */
    double queries[INDEX_HEADS * INDEX_DIMS], key[INDEX_DIMS], values[INDEX_DIMS];
} index_host;

/* ---- the case, the forms, the kernels and work of a step ---- */

static uint32_t index_case(const struct probe_case *probe_case, const char *flag)
{
    return (uint32_t)probe_case_value(probe_case, flag);
}
static int index_form_of(uint32_t instance, uint32_t state)
{
    return index_form_of_keys(index_variants[instance].keys, state, PROBE_SM == PROBE_SM_120A);
}
static int index_checked_row(uint32_t row, uint32_t rows)
{
    return rows <= INDEX_MOST_GROUP_ROWS || row % INDEX_CHECKED_STRIDE == 0;
}

/* ---- a step: built, validated, launched ---- */

static void index_range(int *valid, const char *kernel, const char *what, const void *array, const void *pointer,
                        uint64_t byte_count)
{
    probe_launch_range(valid, kernel, what, (uint64_t)((uintptr_t)pointer - (uintptr_t)array), byte_count, 1,
                       probe_cuda_array_bytes(array));
}
/* Check 10: 1 if every launch of the step reads and writes inside its arrays and loops by held counts. */
static int index_step_is_valid(const struct index_step *step)
{
    const struct index_scan_arguments *scan = &step->scan;
    const struct index_select_arguments *choose = &step->choose;
    const char *name = step->scan_kernel != INDEX_NO_KERNEL ? index_scans[step->scan_kernel].name
                       : step->bound_kernel != INDEX_NO_KERNEL ? index_bounds[step->bound_kernel].name : "index_select";
    const uint64_t rows = scan->row_count, first_row = (uint64_t)(scan->extents - index_card.extents);
    const uint64_t listed = step->family == INDEX_LISTED, spans_of_rows = rows * INDEX_MOST_SPANS * INDEX_SELECTED;
    const uint32_t *card_extents = index_host.card_extents + (first_row < INDEX_MOST_ROWS ? first_row : 0u);
    int valid = 1;
    probe_launch_value(&valid, name, "the extents' first row", first_row, 0, INDEX_MOST_ROWS - 1u);
    probe_launch_value(&valid, name, "rows", rows, 1, INDEX_MOST_ROWS - (first_row < INDEX_MOST_ROWS ? first_row : 0u));
    probe_launch_value(&valid, name, "plane entries: whole tiles within the capacity",
                       step->entries % INDEX_KEYS_PER_TILE == 0 ? step->entries : 0, 1, index_card.capacity);
    probe_launch_value(&valid, name, "tiles a segment", scan->segment_tiles, 1, index_tiles_of(INDEX_MOST_CAPACITY));
    for (uint64_t row = 0; valid && row < rows; row++) {                        /* what the kernels read and loop by */
        probe_launch_value(&valid, name, "a row's extent", card_extents[row], 0, step->entries);
    }
    if (valid && (step->scan_kernel != INDEX_NO_KERNEL || step->bound_kernel != INDEX_NO_KERNEL)) {
        probe_launch_value(&valid, name, "a kernel of a form this build has", index_function_of(step) != NULL, 1, 1);
        probe_launch_geometry(&valid, name, &index_instances[step->instance], step->grid.x, step->grid.y, step->grid.z,
                              step->threads, step->shared_bytes);
        probe_launch_value(&valid, name, "threads of a block its registers allow", step->threads, 1,
                           index_card.most_threads[index_slot_of(step->scan_kernel, step->bound_kernel)]);
        index_range(&valid, name, "plane", step->ring, scan->keys, index_plane_bytes((int)step->form, step->entries));
        index_range(&valid, name, "queries", step->queries, scan->queries, rows * index_query_bytes((int)step->form));
        index_range(&valid, name, "weights", index_card.weights, scan->weights, rows * INDEX_HEADS * sizeof(float));
        index_range(&valid, name, "extents", index_card.extents, scan->extents, rows * sizeof(uint32_t));
        index_range(&valid, name, "scores", step->scores, scan->scores, rows * scan->capacity * sizeof(uint16_t));
        probe_launch_value(&valid, name, "a slab row", scan->capacity, listed ? (uint32_t)INDEX_CANDIDATE_ENTRIES
                           : step->entries, listed ? (uint32_t)INDEX_CANDIDATE_ENTRIES : index_card.capacity);
        index_range(&valid, name, "lists", index_card.lists, scan->listed, listed ? rows * INDEX_CANDIDATE_BLOCKS *
                    sizeof(uint32_t) : 1u);           /* a byte: the arrays a launch does not touch are not held */
        index_range(&valid, name, "folds", index_card.folds, index_card.folds, step->bound_kernel != INDEX_NO_KERNEL
                    ? (uint64_t)step->grid.x * step->grid.y * sizeof(uint32_t) : 1u);
        for (uint64_t row = 0; valid && listed && row < rows; row++) {      /* a listed block: a tile, ascending */
            const uint32_t *list = index_host.lists + (scan->listed - index_card.lists) + row * INDEX_CANDIDATE_BLOCKS;
            for (uint32_t at = 0; valid && at < index_listed_tiles(card_extents[row]); at++) {
                probe_launch_value(&valid, name, "a listed block", list[at], at == 0 ? 0u : list[at - 1u] + 1u,
                                   index_tiles_of(step->entries) - 1u);
            }
        }
    }
    if (valid && step->select != INDEX_SELECT_NONE) {
        name = "index_select";
        probe_launch_geometry(&valid, name, NULL, step->select_grid.x, step->select_grid.y, 1, INDEX_SELECT_THREADS, 0);
        probe_launch_value(&valid, name, "spans over every extent", (uint64_t)choose->spans * INDEX_SPAN_ENTRIES,
                           index_largest_extent(0, (uint32_t)rows, card_extents),
                           (uint64_t)INDEX_MOST_SPANS * INDEX_SPAN_ENTRIES);
        probe_launch_value(&valid, name, "a slab row", choose->capacity, step->entries, index_card.capacity);
        index_range(&valid, name, "scores", index_card.slab, choose->scores,
                    rows * choose->capacity * sizeof(uint16_t));
        index_range(&valid, name, "extents", index_card.extents, choose->extents, rows * sizeof(uint32_t));
        index_range(&valid, name, "selected", index_card.selected, choose->selected,
                    rows * INDEX_SELECTED * sizeof(int32_t));
        index_range(&valid, name, "span keys", index_card.span_keys, choose->span_keys,
                    spans_of_rows * sizeof(uint16_t));
        index_range(&valid, name, "span positions", index_card.span_positions, choose->span_positions,
                    spans_of_rows * sizeof(uint32_t));
        index_range(&valid, name, "span counts", index_card.span_counts, choose->span_counts,
                    rows * INDEX_MOST_SPANS * sizeof(uint32_t));
    }
    return valid;
}

/* The step of an instance on rows first_row .. first_row + row_count - 1 of the case (its slab rows too), reading the
 * plane of ring step ring_step. canonical: at --segment-keys 2048 and no padding. Every field derives from the host's
 * copy of what the card holds; index_step_is_valid has the last word. 0: refused. */
static int index_build_step(const struct probe_case *probe_case, uint32_t instance, uint64_t ring_step,
                            uint32_t first_row, uint32_t row_count, int canonical, struct index_step *step)
{
    const uint32_t family = index_variants[instance].family, state = index_case(probe_case, INDEX_FLAG_STATE);
    const int form = index_form_of(instance, state);
    const uint32_t group = index_group_of(family, row_count), entries = index_case(probe_case, INDEX_FLAG_ENTRIES);
    const uint32_t padding = canonical ? 0u : index_case(probe_case, INDEX_FLAG_PADDING);
    const uint32_t segment = canonical ? (uint32_t)INDEX_CANONICAL_SEGMENT_KEYS
                                       : index_case(probe_case, INDEX_FLAG_SEGMENT);
    const uint64_t plane = index_plane_bytes(form, entries);
    const uint64_t copies = plane >= index_card.cold_bytes ? 1u : (index_card.cold_bytes + plane - 1u) / plane;
    const uint32_t largest = index_largest_extent(first_row, first_row < INDEX_MOST_ROWS &&
                                                  row_count <= INDEX_MOST_ROWS - first_row ? row_count : 0u,
                                                  index_host.card_extents);
    const uint32_t tiles = family == INDEX_LISTED ? index_listed_tiles(largest) : index_tiles_of(largest);
    const uint32_t segment_tiles = segment / INDEX_KEYS_PER_TILE;
    const uint32_t spans = (largest + INDEX_SPAN_ENTRIES - 1u) / INDEX_SPAN_ENTRIES + (largest == 0 ? 1u : 0u);
    const uint32_t segments = tiles > 0 && segment_tiles > 0 ? (tiles + segment_tiles - 1u) / segment_tiles : 1u;
    const uint64_t padded = padding == 0 ? 0 : (index_card.shared_per_sm / padding - index_card.shared_reserved) /
                                                   INDEX_SHARED_ALIGNMENT * INDEX_SHARED_ALIGNMENT;
    const uint64_t queries = (uint64_t)(row_count < group ? row_count : group) * index_query_bytes(form);
    *step = index_step();
    step->instance = instance;
    step->form = (uint32_t)form;
    step->family = family;
    step->entries = entries;
    step->ring = index_card.rings[index_key_state(form)];
    step->queries = index_card.placed[index_query_state(form)];
    step->scores = family == INDEX_LISTED ? (const void *)index_card.listed_scores : (const void *)index_card.slab;
    step->scan_kernel = index_scan_kernel(family, form, group);
    step->bound_kernel = index_bound_kernel(family, form, group);
    step->threads = index_threads_of(family);
    /* a scan holds its group's queries; a bound the baseline's bytes unpadded (its residency), else the padding's */
    const uint64_t held = step->scan_kernel != INDEX_NO_KERNEL || padding == 0 ? (queries > padded ? queries : padded)
                                                                                : padded;
    step->shared_bytes = held > INDEX_BOUND_SHARED_BYTES ? held : (uint64_t)INDEX_BOUND_SHARED_BYTES;
    step->grid = family == INDEX_ROW_MAJOR ? dim3(segments, row_count)
                                           : dim3((row_count + group - 1u) / group, segments);
    step->scan.keys = (const unsigned char *)step->ring + ring_step % copies * plane;
    step->scan.queries = (const unsigned char *)step->queries + (size_t)first_row * index_query_bytes(form);
    step->scan.weights = index_card.weights + (size_t)first_row * INDEX_HEADS;
    step->scan.extents = index_card.extents + first_row;
    step->scan.listed = index_card.lists + (size_t)first_row * INDEX_CANDIDATE_BLOCKS;
    step->scan.row_count = row_count;
    step->scan.capacity = family == INDEX_LISTED ? (uint32_t)INDEX_CANDIDATE_ENTRIES : index_card.capacity;
    step->scan.scores = (uint16_t *)step->scores + (size_t)first_row * step->scan.capacity;
    step->scan.segment_tiles = segment_tiles;
    if (family == INDEX_ONE_BLOCK || family == INDEX_SPANS || family == INDEX_SCAN_SELECT) {
        step->select = family == INDEX_SPANS || (family == INDEX_SCAN_SELECT && row_count <= INDEX_SPANS_MOST_ROWS &&
                                                 entries >= INDEX_SPANS_LEAST_ENTRIES) ? INDEX_SELECT_SPANS
                                                                                       : INDEX_SELECT_ONE;
        step->choose.scores = index_card.slab + (size_t)first_row * index_card.capacity;
        step->choose.extents = step->scan.extents;
        step->choose.selected = index_card.selected + (size_t)first_row * INDEX_SELECTED;
        step->choose.span_keys = index_card.span_keys + (size_t)first_row * INDEX_MOST_SPANS * INDEX_SELECTED;
        step->choose.span_positions = index_card.span_positions + (size_t)first_row * INDEX_MOST_SPANS * INDEX_SELECTED;
        step->choose.span_counts = index_card.span_counts + (size_t)first_row * INDEX_MOST_SPANS;
        step->choose.capacity = index_card.capacity;
        step->choose.spans = spans;
        step->select_grid = step->select == INDEX_SELECT_SPANS ? dim3(spans, row_count) : dim3(row_count);
    }
    return index_step_is_valid(step);
}

/* Queues the launches of a validated step; returns their number. */
static uint32_t index_launch(const struct index_step *step)
{
    struct index_scan_arguments scan = step->scan;
    uint32_t *folds = index_card.folds, launches = 0;
    void *arguments[] = {&scan, &folds};         /* a scan takes the first, a bound both */
#if PROBE_DIAGNOSTICS
    PROBE_REQUIRE(index_step_is_valid(step), "a step that its preparation did not validate");
#endif
    if (index_function_of(step) != NULL) {
        PROBE_CHECK_CUDA(cudaLaunchKernel(index_function_of(step), step->grid, dim3(step->threads), arguments,
                                          step->shared_bytes, 0));
        launches++;
    }
    if (step->select == INDEX_SELECT_ONE) {
        index_select_one_block<<<step->select_grid, INDEX_SELECT_THREADS>>>(step->choose);
        launches++;
    } else if (step->select == INDEX_SELECT_SPANS) {
        index_select_spans<<<step->select_grid, INDEX_SELECT_THREADS>>>(step->choose);
        index_select_merge<<<step->select_grid.y, INDEX_SELECT_THREADS>>>(step->choose);
        launches += INDEX_SPAN_LAUNCHES;
    }
    PROBE_AFTER_LAUNCH();
    return launches;
}

/* ---- the card, the case's inputs ---- */

static void *index_allocate(uint64_t bytes, int filled)
{
    void *array = probe_cuda_allocate(bytes);
    PROBE_CHECK_CUDA(cudaMemset(array, filled ? INDEX_SENTINEL_BYTE : 0, bytes));
    return array;
}
static int index_open_card(void *, struct probe_card_facts *facts)
{
    const struct probe_flag *rows = probe_flag(INDEX_FLAG_ROWS);
    const int checks_only = probe_flag_given(PROBE_FLAG_STEPS) && !probe_flag_text(PROBE_FLAG_PROFILE_CASE)[0];
    if (!probe_cuda_open(facts)) {
        return 0;
    }
    const unsigned int fill_blocks = (unsigned int)facts->sm_count * INDEX_FILL_BLOCKS_PER_SM;
    facts->shared_bytes_most = probe_cuda.properties.sharedMemPerBlockOptin;
    index_card.shared_per_sm = (uint32_t)probe_cuda.properties.sharedMemPerMultiprocessor;
    index_card.shared_reserved = (uint32_t)probe_cuda.properties.reservedSharedMemPerBlock;
    index_card.capacity = (uint32_t)probe_flag_number(INDEX_FLAG_CAPACITY, 0);
    index_card.cold_bytes = (checks_only ? 1u : (uint64_t)INDEX_COLD_L2_SIZES) * facts->l2_bytes;  /* CONVENTIONS 3.5 */
    index_card.most_rows = INDEX_SPECIAL_ROWS;                  /* the special rows' launch has 8 */
    for (uint32_t value = 0; value < rows->value_count; value++) {
        index_card.most_rows = (uint32_t)fmax(index_card.most_rows, (double)rows->values[value]);
    }
    for (uint32_t state = 0; state < INDEX_STATES; state++) {    /* the cold bytes and one whole plane, in tiles */
        const int form = state == INDEX_STATE_FP16 ? INDEX_FORM_HALF : INDEX_FORM_WHOLE;
        const uint64_t tiles = (index_card.cold_bytes + index_plane_bytes(form, index_card.capacity)) /
                               index_tile_bytes(form) + 1u;
        index_card.rings[state] = (unsigned char *)probe_cuda_allocate(tiles * index_tile_bytes(form));
        facts->ring_bytes = state == INDEX_STATE_FP16 ? tiles * index_tile_bytes(form) : facts->ring_bytes;
        index_fill_ring(state, index_card.rings[state], tiles, fill_blocks);
        PROBE_CHECK_CUDA(cudaGetLastError());
        index_card.placed[state] =                   /* a row of FP16 queries is the largest placed row */
            (unsigned char *)index_allocate((uint64_t)INDEX_MOST_ROWS * INDEX_HALF_QUERY_BYTES, 0);
    }
    const uint64_t rows_most = index_card.most_rows, capacity = index_card.capacity;
    const uint64_t spans_of_rows = rows_most * INDEX_MOST_SPANS * INDEX_SELECTED;
    index_card.natural = (uint16_t *)index_allocate(INDEX_MOST_ROWS * INDEX_HEADS * INDEX_DIMS * sizeof(uint16_t), 0);
    index_card.weights = (float *)index_allocate(sizeof(index_host.weights), 0);
    index_card.extents = (uint32_t *)index_allocate(sizeof(index_host.extents), 0);
    index_card.lists = (uint32_t *)index_allocate(sizeof(index_host.lists), 0);
    index_card.slab = (uint16_t *)index_allocate(rows_most * capacity * sizeof(uint16_t), 1);
    index_card.listed_scores = (uint16_t *)index_allocate(rows_most * INDEX_CANDIDATE_ENTRIES * sizeof(uint16_t), 1);
    index_card.folds = (uint32_t *)index_allocate(sizeof(index_host.folds), 0);
    index_card.selected = (int32_t *)index_allocate(INDEX_MOST_ROWS * INDEX_SELECTED * sizeof(int32_t), 0);
    index_card.span_keys = (uint16_t *)index_allocate(spans_of_rows * sizeof(uint16_t), 0);
    index_card.span_positions = (uint32_t *)index_allocate(spans_of_rows * sizeof(uint32_t), 0);
    index_card.span_counts = (uint32_t *)index_allocate(rows_most * INDEX_MOST_SPANS * sizeof(uint32_t), 0);
    for (uint32_t size = 0; size < INDEX_EXACT_SIZES; size++) {      /* the extents check's exact-size slab rows */
        index_card.exact[size] = (uint16_t *)index_allocate(
            index_tiles_of(index_exact_extents[size]) * INDEX_KEYS_PER_TILE * sizeof(uint16_t), 1);
    }
    index_host.slab = (uint16_t *)calloc(rows_most * capacity, sizeof(uint16_t));
    index_host.other = (uint16_t *)calloc(rows_most * capacity, sizeof(uint16_t));
    index_host.listed = (uint16_t *)calloc(rows_most * INDEX_CANDIDATE_ENTRIES, sizeof(uint16_t));
    index_host.plane = (unsigned char *)calloc(index_plane_bytes(INDEX_FORM_WHOLE, index_card.capacity), 1);
    PROBE_REQUIRE(index_host.slab != NULL && index_host.other != NULL && index_host.listed != NULL &&
                  index_host.plane != NULL, "out of host memory for the slabs");
    for (uint32_t kernel = 0; kernel < INDEX_KERNEL_SLOTS; kernel++) {        /* in the order of index_slot_of */
        const uint32_t scans = INDEX_FORMS * INDEX_SHAPES;
        const void *function = kernel < scans ? index_scans[kernel].function : index_bounds[kernel - scans].function;
        if (function != NULL) {                  /* NULL: a form this build lacks */
            struct cudaFuncAttributes attributes;
            PROBE_CHECK_CUDA(cudaFuncSetAttribute(function, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                  (int)facts->shared_bytes_most));
            PROBE_CHECK_CUDA(cudaFuncGetAttributes(&attributes, function));
            index_card.most_threads[kernel] = (uint32_t)attributes.maxThreadsPerBlock;
        }
    }
    PROBE_CHECK_CUDA(cudaDeviceSynchronize());
    return 1;
}
static void index_close_card(void *)
{
    void *owned[] = {index_host.slab, index_host.other, index_host.listed, index_host.plane, index_host.key_folds[0],
                     index_host.key_folds[INDEX_STATE_TRAINED], index_host.key_folds[INDEX_STATE_FRESH]};
    for (uint32_t index = 0; index < PROBE_COUNT_OF(owned); index++) {
        free(owned[index]);
    }
    probe_cuda_close();
}
static void index_upload(void *card, const void *host, size_t bytes)
{
    PROBE_CHECK_CUDA(cudaMemcpy(card, host, bytes, cudaMemcpyHostToDevice));
}
static void index_describe_case(void *, const struct probe_case *probe_case, struct probe_case_facts *facts)
{
    static const char *const states[INDEX_STATES] = {"fp16", "trained", "fresh"};
    static char part[PROBE_NAME_BYTES];          /* the case less its state and rows */
    const uint32_t state = index_case(probe_case, INDEX_FLAG_STATE), rows = index_case(probe_case, INDEX_FLAG_ROWS);
    const uint32_t entries = index_case(probe_case, INDEX_FLAG_ENTRIES);
    const uint32_t rate = (uint32_t)probe_flag_number(INDEX_FLAG_RATE, 0);
    const uint32_t placed = rows > INDEX_SPECIAL_ROWS ? rows : (uint32_t)INDEX_SPECIAL_ROWS;
    double bytes = 0.0, flops = 0.0;
    for (uint32_t row = 0; row < placed; row++) {
        const uint32_t behind = row < rows ? (rows - 1u - row) / rate : 0u;
        index_host.extents[row] = index_host.card_extents[row] = behind < entries ? entries - behind : 0u;
        for (uint32_t head = 0; head < INDEX_HEADS; head++) {
            index_host.weights[row * INDEX_HEADS + head] = index_weight_value(row, head);
        }
        index_make_list(row, index_host.extents[row], index_host.lists + (size_t)row * INDEX_CANDIDATE_BLOCKS);
    }
    if (probe_flag_given(INDEX_FLAG_BREAK_EXTENT)) {     /* row 0 below its group's largest: the same tiles written */
        const uint32_t group = rows < INDEX_MOST_GROUP_ROWS ? rows : (uint32_t)INDEX_MOST_GROUP_ROWS;
        PROBE_REQUIRE(rows > 1 && index_host.extents[0] < index_largest_extent(0, group, index_host.extents),
                      INDEX_FLAG_BREAK_EXTENT " needs 2 rows or more, row 0 not its group's largest");
        index_host.card_extents[0]++;
    }
    for (uint32_t instance = 0; instance < INDEX_INSTANCE_COUNT; instance++) {     /* left out where it cannot run */
        uint32_t most = 0, family = index_variants[instance].family;
        const int launchable = index_launchable(index_card.most_threads, family, index_form_of(instance, state), &most);
        if (index_card.configured[instance] && !launchable && (index_card.told[instance] >> state & 1u) == 0) {
            printf("UNAVAILABLE card=%s variant=%s state=%s threads=%u most_threads=%u\n", probe_run.card_label,
                   index_instances[instance].name, states[state], index_threads_of(family), most);
            index_card.told[instance] |= 1u << state;
        }
        index_instances[instance].available = index_card.configured[instance] && launchable;
    }
    PROBE_REQUIRE(index_instances[PROBE_BASELINE].available, "the baseline's kernels refuse its block in this state");
    index_upload(index_card.weights, index_host.weights, sizeof(index_host.weights));
    index_upload(index_card.extents, index_host.card_extents, sizeof(index_host.card_extents));
    index_upload(index_card.lists, index_host.lists, sizeof(index_host.lists));
    index_place_queries(index_card.natural, index_card.placed, placed);
    PROBE_CHECK_CUDA(cudaDeviceSynchronize());
    index_step_work(INDEX_SCAN, index_form_of(PROBE_BASELINE, state), rows, index_host.extents, &bytes, &flops);
    facts->format = states[state];
    const char *after_state = strchr(probe_case->name, ',');
    const char *after_rows = after_state != NULL ? strchr(after_state + 1, ',') : NULL;
    PROBE_REQUIRE(after_rows != NULL && snprintf(part, sizeof(part), "%s", after_rows + 1) > 0, "a case's part");
    facts->part = part;
    facts->rows = rows;
    facts->experts = facts->overlap = -1;
    facts->megabytes = bytes / INDEX_BYTES_PER_MB;
}

/* ---- checks ---- */

/* A checked launch: the region it writes made the sentinel, the step, its rows read back into `into`. */
static void index_run(const struct index_step *step, uint16_t *into)
{
    const size_t words = (size_t)step->scan.row_count * step->scan.capacity;
    PROBE_CHECK_CUDA(cudaMemset(step->scan.scores, INDEX_SENTINEL_BYTE, words * sizeof(uint16_t)));
    index_launch(step);
    PROBE_CHECK_CUDA(cudaMemcpy(into, step->scan.scores, words * sizeof(uint16_t), cudaMemcpyDeviceToHost));
}
static int index_build_checked(const struct probe_case *probe_case, uint32_t instance, uint32_t first_row,
                               uint32_t rows, int canonical, struct index_step *step)
{
    const int valid = index_build_step(probe_case, instance, 0, first_row, rows, canonical, step);
    if (!valid) {
        probe_check_count(probe_case->name, index_instances[instance].name, "launches_valid", 1, 1, 0);
    }
    return valid;
}
/* Words of rows' written tiles that differ between two copies [row][capacity]; a bits_ check under --break-bits
 * changes one of the second first. */
static uint64_t index_differing(const uint16_t *first, uint16_t *second, uint32_t first_row, uint32_t rows, int bits,
                                uint64_t *of)
{
    uint64_t differing = 0;
    second[(size_t)first_row * index_card.capacity] ^= bits && probe_flag_given(INDEX_FLAG_BREAK_BITS) ? 1u : 0u;
    for (uint32_t row = first_row; row < first_row + rows; row++) {
        const size_t at = (size_t)row * index_card.capacity;
        for (uint32_t entry = 0; entry < index_tiles_of(index_host.extents[row]) * INDEX_KEYS_PER_TILE; entry++) {
            differing += first[at + entry] != second[at + entry] ? 1u : 0u;
            (*of)++;
        }
    }
    return differing;
}
/* host_reference: the sampled scores of the checked rows in the instance's state; the worst row's error. */
static void index_check_reference(const struct probe_case *probe_case, uint32_t instance, const uint16_t *slab,
                                  int must_report)
{
    const int form = index_form_of(instance, index_case(probe_case, INDEX_FLAG_STATE));
    const uint32_t rows = index_case(probe_case, INDEX_FLAG_ROWS);
    double worst = 0.0;
    for (uint32_t row = 0; row < rows; row++) {
        const uint32_t extent = index_host.extents[row];
        double errors = 0.0, squares = 0.0;
        for (uint32_t head = 0; index_checked_row(row, rows) && head < INDEX_HEADS; head++) {
            index_natural(row, head, 1, index_host.values);
            reference_state_values((int)index_query_state(form), index_host.values,
                                   index_host.queries + head * INDEX_DIMS);
        }
        for (uint32_t sample = 0; index_checked_row(row, rows) && sample < extent && sample < INDEX_SAMPLES; sample++) {
            const uint32_t entry = index_sampled_entry(extent, sample);
            index_natural(entry, 0, 0, index_host.values);
            reference_state_values((int)index_key_state(form), index_host.values, index_host.key);
            const double expected =
                reference_score(index_host.queries, index_host.key, index_host.weights + row * INDEX_HEADS);
            const double card = reference_half(slab[(size_t)row * index_card.capacity + entry]);
            errors += (card - expected) * (card - expected);
            squares += expected * expected;
        }
        worst = fmax(worst, squares > 0.0 ? sqrt(errors / squares) : errors > 0.0 ? (double)INFINITY : 0.0);
    }
    probe_check_error(probe_case->name, index_instances[instance].name, "host_reference", worst, INDEX_TOLERANCE,
                      must_report);
}
/* extents: the baseline's scores, -inf and untouched sentinel where each must be, every row. */
static void index_check_extents(const struct probe_case *probe_case, const uint16_t *slab)
{
    const uint32_t rows = index_case(probe_case, INDEX_FLAG_ROWS), group = index_group_of(INDEX_SCAN, rows);
    uint64_t differing = 0;
    for (uint32_t row = 0; row < rows; row++) {
        const uint32_t extent = index_host.extents[row];
        const uint32_t written =
            index_group_tiles(row / group * group, rows, group, index_host.extents) * INDEX_KEYS_PER_TILE;
        for (uint32_t entry = 0; entry < index_card.capacity; entry++) {
            const uint16_t bits = slab[(size_t)row * index_card.capacity + entry];
            differing += entry < extent ? (bits == INDEX_SENTINEL || bits == INDEX_NEGATIVE_INFINITY)
                         : entry < written ? bits != INDEX_NEGATIVE_INFINITY : bits != INDEX_SENTINEL;
        }
    }
    probe_check_count(probe_case->name, "baseline", "extents", differing, (uint64_t)rows * index_card.capacity,
                      probe_flag_given(INDEX_FLAG_BREAK_EXTENT));
}
/* integer_contract of a TRAINED instance: its sampled scores against the contract from the card's own codes (under
 * --break-bits the first one changed). */
static void index_check_contract(const struct probe_case *probe_case, uint32_t instance, const uint16_t *slab)
{
    const uint32_t rows = index_case(probe_case, INDEX_FLAG_ROWS), entries = index_case(probe_case, INDEX_FLAG_ENTRIES);
    const int breaks_bits = probe_flag_given(INDEX_FLAG_BREAK_BITS);
    static int8_t query[INDEX_HEADS * INDEX_DIMS], key[INDEX_DIMS];
    static uint8_t query_scales[INDEX_HEADS * INDEX_BLOCKS], key_scales[INDEX_BLOCKS];
    uint64_t differing = 0, pairs = 0, exact_pairs = 0;
    PROBE_CHECK_CUDA(cudaMemcpy(index_host.plane, index_card.rings[INDEX_STATE_TRAINED],
                                index_plane_bytes(INDEX_FORM_WHOLE, entries), cudaMemcpyDeviceToHost));
    PROBE_CHECK_CUDA(cudaMemcpy(index_host.placed, index_card.placed[INDEX_STATE_TRAINED],
                                (size_t)rows * INDEX_CODE_QUERY_BYTES, cudaMemcpyDeviceToHost));
    for (uint32_t row = 0; row < rows; row++) {
        const uint32_t extent = index_host.extents[row];
        reference_read_queries(index_host.placed + (size_t)row * INDEX_CODE_QUERY_BYTES, query, query_scales);
        for (uint32_t entry = 0; index_checked_row(row, rows) && entry < extent;
             entry += extent <= INDEX_SAMPLES ? 1u : extent / INDEX_SAMPLES) {
            int exact = 0;
            reference_read_key(index_host.plane, entry, key, key_scales);
            const uint32_t bits = reference_whole_score(query, query_scales, key, key_scales,
                                                        index_host.weights + row * INDEX_HEADS, &exact);
            differing += (bits ^ (breaks_bits && pairs == 0 ? 1u : 0u)) != slab[(size_t)row * index_card.capacity +
                                                                                   entry];
            exact_pairs += exact ? 1u : 0u;
            pairs++;
        }
    }
    probe_check_count(probe_case->name, index_instances[instance].name, "integer_contract", differing, pairs,
                      breaks_bits);
    printf("EXACT card=%s case=%s variant=%s pairs=%" PRIu64 " exact=%" PRIu64 " inexact=%" PRIu64 "\n",
           probe_run.card_label, probe_case->name, index_instances[instance].name, pairs, exact_pairs,
           pairs - exact_pairs);
}
/* quantizer_queries (every placed row of both 8-bit forms) and, once a run, quantizer_keys (the first --capacity keys
 * of both 8-bit rings) against the host's rounding; corrupted_state: the ring --corrupt overwrote. --break-bits
 * changes the first byte of each read-back copy. */
static void index_check_quantizers(const struct probe_case *probe_case, int keys_too, uint32_t corrupted_state)
{
    static int8_t codes[INDEX_HEADS * INDEX_DIMS];
    static uint8_t scales[INDEX_HEADS * INDEX_BLOCKS], host_codes[INDEX_DIMS], host_scales[INDEX_BLOCKS];
    const uint32_t rows = index_case(probe_case, INDEX_FLAG_ROWS);
    const unsigned char flip = probe_flag_given(INDEX_FLAG_BREAK_BITS) ? 1u : 0u;
    for (uint32_t state = INDEX_STATE_TRAINED; state < INDEX_STATES; state++) {
        const char *variant = state == INDEX_STATE_TRAINED ? "trained" : "fresh";
        uint64_t differing = 0;
        PROBE_CHECK_CUDA(cudaMemcpy(index_host.placed, index_card.placed[state], (size_t)rows * INDEX_CODE_QUERY_BYTES,
                                    cudaMemcpyDeviceToHost));
        index_host.placed[0] ^= flip;
        for (uint32_t row = 0; row < rows; row++) {
            reference_read_queries(index_host.placed + (size_t)row * INDEX_CODE_QUERY_BYTES, codes, scales);
            for (uint32_t head = 0; head < INDEX_HEADS; head++) {
                index_natural(row, head, 1, index_host.values);
                reference_codes((int)state, index_host.values, host_codes, host_scales);
                differing += (uint64_t)(memcmp(codes + head * INDEX_DIMS, host_codes, INDEX_DIMS) != 0) +
                             (uint64_t)(memcmp(scales + head * INDEX_BLOCKS, host_scales, INDEX_BLOCKS) != 0);
            }
        }
        probe_check_count(probe_case->name, variant, "quantizer_queries", differing,
                          (uint64_t)rows * INDEX_HEADS * INDEX_QUANTIZER_PARTS, flip);
        PROBE_CHECK_CUDA(cudaMemcpy(index_host.plane, index_card.rings[state], keys_too ? index_plane_bytes(
                                        INDEX_FORM_WHOLE, index_card.capacity) : 0u, cudaMemcpyDeviceToHost));
        index_host.plane[0] ^= flip;
        differing = 0;
        for (uint32_t key = 0; keys_too && key < index_card.capacity; key++) {
            index_natural(key, 0, 0, index_host.values);
            reference_codes((int)state, index_host.values, host_codes, host_scales);
            reference_read_key(index_host.plane, key, codes, scales);
            differing += (uint64_t)(memcmp(codes, host_codes, INDEX_DIMS) != 0) +
                         (uint64_t)(memcmp(scales, host_scales, INDEX_BLOCKS) != 0);
        }
        if (keys_too) {
            probe_check_count(probe_case->name, variant, "quantizer_keys", differing,
                              (uint64_t)index_card.capacity * INDEX_QUANTIZER_PARTS, state == corrupted_state || flip);
        }
    }
}
/* The fold of every key, which the host forms from the values alone: its halves, or its codes and scale codes. */
static const uint16_t *index_key_folds(uint32_t state)
{
    if (index_host.key_folds[state] != NULL) {
        return index_host.key_folds[state];
    }
    index_host.key_folds[state] = (uint16_t *)calloc(index_card.capacity, sizeof(uint16_t));
    PROBE_REQUIRE(index_host.key_folds[state] != NULL, "out of host memory for the folds");
    for (uint32_t key = 0; key < index_card.capacity; key++) {
        index_natural(key, 0, 0, index_host.values);
        index_host.key_folds[state][key] = reference_key_fold((int)state, index_host.values);
    }
    return index_host.key_folds[state];
}
/* bound_sums_and_folds: every block's fold (read_only) or sum (tensor_only) against the host's. */
static void index_check_bound(const struct probe_case *probe_case, uint32_t instance, int must_report)
{
    const uint32_t rows = index_case(probe_case, INDEX_FLAG_ROWS), family = index_variants[instance].family;
    const uint32_t size = index_group_of(family, rows);       /* rows a block */
    const int form = index_form_of(instance, index_case(probe_case, INDEX_FLAG_STATE));
    const uint16_t *folds = family == INDEX_READ ? index_key_folds(index_key_state(form)) : NULL;
    struct index_step step;
    uint64_t differing = 0;
    if (!index_build_checked(probe_case, instance, 0, rows, 0, &step)) {
        return;
    }
    const uint64_t blocks = (uint64_t)step.grid.x * step.grid.y;
    PROBE_REQUIRE(blocks <= INDEX_MOST_FOLDS, "more blocks than folds: a larger --segment-keys is needed here");
    index_launch(&step);
    PROBE_CHECK_CUDA(cudaMemcpy(index_host.folds, index_card.folds, blocks * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    for (uint64_t block = 0; block < blocks; block++) {
        differing += index_host.folds[block] != index_bound_expected(family, (uint32_t)(block % step.grid.x) * size,
                     (uint32_t)(block / step.grid.x), step.scan.segment_tiles, rows, index_host.extents, folds);
    }
    probe_check_count(probe_case->name, index_instances[instance].name, "bound_sums_and_folds", differing, blocks,
                      must_report);
}
/* Both select forms on the card's slab of rows 0 .. rows - 1 against the host's counting sort of `slab` and against
 * each other (the checked rows alone where only_checked). */
static void index_check_selection(const struct probe_case *probe_case, const char *variant, const uint16_t *slab,
                                  uint32_t rows, int only_checked)
{
    static int32_t expected[INDEX_SELECTED];
    const int ties_from_top = probe_flag_given(INDEX_FLAG_BREAK_TIES);
    const int breaks_bits = probe_flag_given(INDEX_FLAG_BREAK_BITS);
    uint64_t checked[INDEX_COLUMNS_PER_LANE] = {0, 0}, wrong[INDEX_COLUMNS_PER_LANE] = {0, 0}, agree_of = 0;
    uint64_t agree_differing = 0;                /* checked and wrong: [0] rows whose cut takes all its ties, [1] not */
    struct index_step one, spans;
    if (!index_build_checked(probe_case, INDEX_INSTANCE_one_block, 0, rows, 0, &one) ||
        !index_build_checked(probe_case, INDEX_INSTANCE_spans, 0, rows, 0, &spans)) {
        return;
    }
    index_launch(&one);
    PROBE_CHECK_CUDA(cudaMemcpy(index_host.selected, index_card.selected,
                                (size_t)rows * INDEX_SELECTED * sizeof(int32_t), cudaMemcpyDeviceToHost));
    index_launch(&spans);
    PROBE_CHECK_CUDA(cudaMemcpy(index_host.selected_other, index_card.selected,
                                (size_t)rows * INDEX_SELECTED * sizeof(int32_t), cudaMemcpyDeviceToHost));
    index_host.selected_other[0] ^= breaks_bits ? 1 : 0;
    for (uint32_t row = 0; row < rows; row++) {
        const int32_t *card = index_host.selected + (size_t)row * INDEX_SELECTED;
        if (only_checked && !index_checked_row(row, rows)) {
            continue;
        }
        const int ties_left = reference_select(slab + (size_t)row * index_card.capacity, index_host.extents[row],
                                               ties_from_top, expected);
        checked[ties_left]++;
        wrong[ties_left] += memcmp(card, expected, sizeof(expected)) != 0 ? 1u : 0u;
        for (uint32_t at = 0; at < INDEX_SELECTED; at++, agree_of++) {
            agree_differing += card[at] != index_host.selected_other[(size_t)row * INDEX_SELECTED + at] ? 1u : 0u;
        }
    }
    probe_digest_add(index_host.selected, (size_t)rows * INDEX_SELECTED * sizeof(int32_t));
    probe_check_count(probe_case->name, variant, "selection_tie_rows", wrong[1], checked[1],
                      ties_from_top && checked[1] > 0);
    probe_check_count(probe_case->name, variant, "selection_other_rows", wrong[0], checked[0], 0);
    probe_check_count(probe_case->name, variant, "bits_select_forms", agree_differing, agree_of, breaks_bits);
}
/* The case with other rows and entries: a launch of the checks that is no step of the case. */
static void index_case_with(const struct probe_case *probe_case, uint32_t rows, uint32_t entries,
                            struct probe_case *made)
{
    *made = *probe_case;
    for (uint32_t axis = 0; axis < made->axis_count; axis++) {
        made->values[axis] = strcmp(made->axes[axis]->name, INDEX_FLAG_ENTRIES) == 0 ? entries
                             : strcmp(made->axes[axis]->name, INDEX_FLAG_ROWS) == 0 ? rows : made->values[axis];
    }
}
/* Once a run: the exact-size launches (--break-extent: the last one's card extent one higher), then 8 special rows of
 * one launch selected. The case's extents are put back. */
static void index_check_once(const struct probe_case *probe_case)
{
    static uint32_t saved[INDEX_MOST_ROWS];
    static uint16_t exact[INDEX_SAMPLES];
    const int breaks_extent = probe_flag_given(INDEX_FLAG_BREAK_EXTENT);
    struct index_step step;
    struct probe_case special;
    uint64_t differing = 0, of = 0;
    memcpy(saved, index_host.card_extents, sizeof(saved));
    for (uint32_t size = 0; size < INDEX_EXACT_SIZES && index_exact_extents[size] <= index_card.capacity; size++) {
        const uint32_t extent = index_exact_extents[size], words = index_tiles_of(extent) * INDEX_KEYS_PER_TILE;
        index_host.extents[0] = index_host.card_extents[0] = extent;
        index_upload(index_card.extents, index_host.card_extents, sizeof(uint32_t));
        index_case_with(probe_case, 1, words, &special);      /* the plane of copy 0 begins where the ring does */
        if (!index_build_checked(&special, PROBE_BASELINE, 0, 1, 1, &step)) {
            break;
        }
        index_run(&step, index_host.other);
        step.scan.capacity = words;
        step.scan.scores = index_card.exact[size];
        step.scores = index_card.exact[size];
        index_host.card_extents[0] += breaks_extent && size + 1u == INDEX_EXACT_SIZES ? 1u : 0u;   /* 514 of 520 */
        index_upload(index_card.extents, index_host.card_extents, sizeof(uint32_t));
        PROBE_REQUIRE(index_step_is_valid(&step), "the exact-size launch was refused");
        index_run(&step, exact);
        for (uint32_t entry = 0; entry < words; entry++, of++) {
            differing += exact[entry] != index_host.other[entry] ? 1u : 0u;
        }
    }
    probe_check_count(probe_case->name, "baseline", "extents_exact_size", differing, of, breaks_extent);
    for (uint32_t row = 0; row < INDEX_SPECIAL_ROWS; row++) {
        const uint32_t extent = index_special_extents[row];
        index_host.extents[row] = index_host.card_extents[row] = extent < index_card.capacity ? extent
                                                                                              : index_card.capacity;
    }
    index_upload(index_card.extents, index_host.card_extents, sizeof(index_host.card_extents));
    index_case_with(probe_case, INDEX_SPECIAL_ROWS, index_card.capacity, &special);
    if (index_build_checked(&special, PROBE_BASELINE, 0, INDEX_SPECIAL_ROWS, 1, &step)) {
        index_run(&step, index_host.other);
        for (uint32_t entry = 0; entry < index_host.extents[INDEX_TIES_ROW]; entry++) {    /* all +0; ties at the cut */
            index_host.other[(size_t)INDEX_ZERO_ROW * index_card.capacity + entry] = 0;
            index_host.other[(size_t)INDEX_TIES_ROW * index_card.capacity + entry] = entry < INDEX_TIES_ABOVE
                ? INDEX_HALF_TWO : entry < INDEX_TIES_ABOVE + INDEX_TIES_EQUAL ? INDEX_HALF_ONE : INDEX_HALF_HALF;
        }
        index_upload(index_card.slab, index_host.other,
                     (size_t)INDEX_SPECIAL_ROWS * index_card.capacity * sizeof(uint16_t));
        index_check_selection(&special, "special_rows", index_host.other, INDEX_SPECIAL_ROWS, 0);
    }
    memcpy(index_host.card_extents, saved, sizeof(saved));
    memcpy(index_host.extents, saved, sizeof(saved));
    index_host.extents[0] -= breaks_extent ? 1u : 0u;
    index_upload(index_card.extents, index_host.card_extents, sizeof(index_host.card_extents));
}
/* --break-route: the baseline's step with its plane past the ring, with an extent past its plane, with no tile a
 * segment: the validation every real step passes must refuse each; none is launched. A refusal is 1 of 1. */
static void index_break_route(const struct probe_case *probe_case)
{
    static const char *const names[] = {"refused_plane_past_the_ring", "refused_extent_past_the_plane",
                                        "refused_no_tile_a_segment"};
    for (uint32_t broken = 0; broken < PROBE_COUNT_OF(names); broken++) {
        struct index_step step;
        PROBE_REQUIRE(index_build_step(probe_case, PROBE_BASELINE, 0, 0, 1, 0, &step) && index_host.card_extents[0] > 0,
                      "--break-route needs a case whose baseline step is valid and whose row 0 sees an entry");
        step.scan.keys += broken == 0 ? probe_cuda_array_bytes(step.ring) : 0u;
        step.entries = broken == 1 ? (index_host.card_extents[0] - 1u) / INDEX_KEYS_PER_TILE * INDEX_KEYS_PER_TILE
                                   : step.entries;
        step.scan.segment_tiles = broken + 1u == PROBE_COUNT_OF(names) ? 0u : step.scan.segment_tiles;
        probe_check_count(probe_case->name, "baseline", names[broken], index_step_is_valid(&step) ? 0u : 1u, 1, 1);
    }
}
/* Each instance against the baseline's words, or a check of its own: bits, the reference, its bound, its contract. */
static void index_check_instances(const struct probe_case *probe_case, uint32_t ring_state)
{
    const uint32_t rows = index_case(probe_case, INDEX_FLAG_ROWS), state = index_case(probe_case, INDEX_FLAG_STATE);
    const int corrupt = probe_flag_given(PROBE_FLAG_CORRUPT), breaks_bits = probe_flag_given(INDEX_FLAG_BREAK_BITS);
    for (uint32_t instance = PROBE_BASELINE_AGAIN + 1u; instance < index_instances_in_run; instance++) {
        const struct index_variant *variant = &index_variants[instance];
        const int form = index_form_of(instance, state);
        /* the E2M1 form claims WHOLE's bits on trained codes; bytes --corrupt wrote are none, and it reads them so */
        const int scaled = form == INDEX_FORM_E2M1 && state == INDEX_STATE_TRAINED && !corrupt;
        const int same = form == index_form_of(PROBE_BASELINE, state) || scaled;
        const int selects = variant->family >= INDEX_ONE_BLOCK && variant->family <= INDEX_SCAN_SELECT;
        const char *name = index_instances[instance].name;
        struct index_step step;
        uint64_t differing = 0, of = 0;
        if (!index_instances[instance].available || (index_is_per_row(variant->family) && rows > INDEX_MOST_GROUP_ROWS)
            || selects || (variant->family == INDEX_KEY_MAJOR && rows == 1)) {
            continue;          /* the selections: index_check_selection; key_major at one row: the baseline's launch */
        }
        if (index_is_bound(variant->family)) {
            index_check_bound(probe_case, instance, corrupt && variant->family == INDEX_READ);
            continue;
        }
        if (!index_build_checked(probe_case, instance, 0, rows, 0, &step)) {
            continue;
        }
        index_run(&step, variant->family == INDEX_LISTED ? index_host.listed : index_host.other);
        if (variant->family == INDEX_LISTED) {   /* its words against the baseline's at the listed entries */
            index_host.listed[0] ^= breaks_bits ? 1u : 0u;
            differing = index_listed_differing(index_host.listed, index_host.slab, index_card.capacity,
                                               index_host.lists, index_host.extents, rows, &of);
        } else {
            index_check_reference(probe_case, instance, index_host.other,
                                  corrupt && index_key_state(form) == ring_state);
            if (scaled) {
                index_check_contract(probe_case, instance, index_host.other);
            }
            differing = same ? index_differing(index_host.slab, index_host.other, 0, rows, variant->bits, &of) : 0u;
        }
        if (variant->bits && same) {             /* a form that differs is reported and loses class R for the run */
            probe_check_count(probe_case->name, name, "bits_against_baseline", differing, of, breaks_bits);
            index_instances[instance].contract_class = differing != 0 ? 'C' : index_instances[instance].contract_class;
        }
    }
}
static void index_check_case(void *, const struct probe_case *probe_case)
{
    const uint32_t rows = index_case(probe_case, INDEX_FLAG_ROWS), state = index_case(probe_case, INDEX_FLAG_STATE);
    const int corrupt = probe_flag_given(PROBE_FLAG_CORRUPT), breaks_bits = probe_flag_given(INDEX_FLAG_BREAK_BITS);
    const uint32_t ring_state = index_key_state(index_form_of(PROBE_BASELINE, state));
    struct index_step step;
    uint64_t of = 0;
    if (probe_flag_given(PROBE_FLAG_BREAK_ROUTE)) {
        index_break_route(probe_case);
        return;
    }
    if (corrupt) {                                /* the host still takes every key from its definition */
        PROBE_CHECK_CUDA(cudaMemset(index_card.rings[ring_state], INDEX_CORRUPTED_BYTE, INDEX_CORRUPTED_BYTES));
    }
    index_check_quantizers(probe_case, probe_case->index == 0, corrupt ? ring_state : (uint32_t)INDEX_STATES);
    if (probe_case->index == 0) {
        index_check_once(probe_case);
    }
    if (!index_build_checked(probe_case, PROBE_BASELINE, 0, rows, 0, &step)) {
        return;
    }
    index_run(&step, index_host.slab);
    index_check_reference(probe_case, PROBE_BASELINE, index_host.slab, corrupt);
    index_check_extents(probe_case, index_host.slab);
    if (index_form_of(PROBE_BASELINE, state) == INDEX_FORM_WHOLE) {
        index_check_contract(probe_case, PROBE_BASELINE, index_host.slab);
    }
    probe_digest_add(index_host.slab, (size_t)rows * index_card.capacity * sizeof(uint16_t));
    if (!probe_flag_given(INDEX_FLAG_BREAK_EXTENT)) {      /* on the card's slab, which is the baseline's */
        index_check_selection(probe_case, "selections", index_host.slab, rows, 1);
    }
    index_check_instances(probe_case, ring_state);
    if ((index_case(probe_case, INDEX_FLAG_SEGMENT) != INDEX_CANONICAL_SEGMENT_KEYS ||
         index_case(probe_case, INDEX_FLAG_PADDING) != 0) && index_build_checked(probe_case, PROBE_BASELINE, 0, rows, 1,
                                                                                 &step)) {
        index_run(&step, index_host.other);                       /* the same scores at the canonical shape */
        probe_check_count(probe_case->name, "baseline", "bits_at_segment_2048_unpadded",
                          index_differing(index_host.slab, index_host.other, 0, rows, 1, &of), of, breaks_bits);
    }
    for (uint32_t instance = PROBE_BASELINE; rows > 1 && instance < index_instances_in_run;
         instance += instance == PROBE_BASELINE ? (uint32_t)INDEX_INSTANCE_head_order : 1u) {
        uint64_t differing = 0, compared = 0;
        if (index_build_checked(probe_case, instance, 0, rows, 0, &step)) {
            index_run(&step, index_host.other);   /* all rows of this instance, then each checked row alone */
        }
        for (uint32_t row = 0; row < rows; row++) {
            if (index_checked_row(row, rows) && index_build_checked(probe_case, instance, row, 1, 0, &step)) {
                index_run(&step, index_host.slab + (size_t)row * index_card.capacity);
                differing += index_differing(index_host.other, index_host.slab, row, 1, 0, &compared);
            }
        }
        probe_check_count(probe_case->name, index_instances[instance].name, "rows_exact", differing, compared,
                          instance == INDEX_INSTANCE_head_order);
    }
    probe_cuda_check_guards(probe_case->name);
}

/* ---- timing ---- */

static int index_prepare_step(void *, const struct probe_case *probe_case, uint32_t instance, uint64_t step,
                              uint32_t slot)
{
    const uint32_t family = index_variants[instance].family, rows = index_case(probe_case, INDEX_FLAG_ROWS);
    PROBE_REQUIRE(slot < PROBE_STREAM_STEPS, "a slot past the slots of a stream");
    if ((family == INDEX_ONE_BLOCK || family == INDEX_SPANS) && slot == 0) {     /* the slab a selection reads */
        struct index_step scan;
        if (!index_build_step(probe_case, PROBE_BASELINE, step, 0, rows, 0, &scan)) {
            return 0;
        }
        index_launch(&scan);
        PROBE_CHECK_CUDA(cudaDeviceSynchronize());
    }
    return index_build_step(probe_case, instance, step, 0, rows, 0, &index_card.steps[slot]);
}
static uint32_t index_launch_step(void *, const struct probe_case *, uint32_t, uint32_t slot)
{
    return index_launch(&index_card.steps[slot]);
}
/* The RATE line of each timed instance, then the figures of index_figures. */
static void index_verdicts(void *, const struct probe_case *probe_case, const struct probe_result *results,
                           double noise_ratio)
{
    const uint32_t rows = index_case(probe_case, INDEX_FLAG_ROWS), entries = index_case(probe_case, INDEX_FLAG_ENTRIES);
    const uint32_t state = index_case(probe_case, INDEX_FLAG_STATE);
    const double scan_us = results[PROBE_BASELINE].step.median;
    char figure[INDEX_NAME_BYTES];
    for (uint32_t instance = 0; instance < index_instances_in_run; instance++) {
        struct index_step step;
        double bytes = 0.0, flops = 0.0, median = results[instance].step.median;
        int resident = 0;
        if (results[instance].mode == NULL || !index_build_step(probe_case, instance, 0, 0, rows, 0, &step)) {
            continue;
        }
        index_step_work(step.family, (int)step.form, rows, index_host.extents, &bytes, &flops);
        const void *function = index_function_of(&step);    /* none: a select alone, whose figures are na here */
        if (function != NULL) {
            PROBE_CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, function, (int)step.threads,
                                                                           step.shared_bytes));
        }
        printf("RATE card=%s case=%s variant=%s gbps=%s tflops=%s shared=%s resident=%s reads=%s\n",
               probe_run.card_label, probe_case->name, index_instances[instance].name,
               probe_number(bytes * INDEX_GB_PER_BYTE_MICROSECOND / median, 0),
               probe_number(flops * INDEX_TFLOPS_PER_FLOP_MICROSECOND / median, 1),
               probe_whole(function != NULL ? (int64_t)step.shared_bytes : -1),
               probe_whole(function != NULL ? resident : -1), index_reads_of(step.family));
    }
    for (uint32_t index = 0; index < PROBE_COUNT_OF(index_figures); index++) {
        const struct index_figure *judged = &index_figures[index];
        const struct probe_result *first = &results[judged->judged], *second = &results[judged->other];
        const uint32_t instance = first->mode == NULL || second->step.median < first->step.median
                                      ? judged->other : judged->judged;     /* the faster of two, where two are named */
        const double bar = judged->scaled_by < INDEX_INSTANCE_COUNT ? judged->bar * results[judged->scaled_by].ratio
                                                                     : judged->bar;
        if (judged->rows != rows || judged->entries != entries || ((judged->states >> state) & 1u) == 0 ||
            (first->mode == NULL && second->mode == NULL)) {
            continue;                            /* not this case, or nothing it judges was timed in this run */
        }
        PROBE_REQUIRE(snprintf(figure, sizeof(figure), "%s_at_%s", judged->name, probe_case->name) > 0, "a figure");
        probe_verdict(figure, bar, bar * (judged->inverted ? results[instance].step.median : scan_us),
                      judged->inverted ? 1.0 / results[instance].ratio : results[instance].ratio, noise_ratio,
                      results[instance].same_sign, results[instance].rounds);
    }
}

static int index_configure(void *);
static struct probe_program index_program = {
    "index_scan", index_flags, PROBE_COUNT_OF(index_flags), index_instances, INDEX_INSTANCE_COUNT - 1u, NULL,
    index_configure, index_open_card, index_describe_case, index_check_case, index_prepare_step, index_launch_step,
    probe_cuda_mark_start, probe_cuda_elapsed_microseconds, index_verdicts, index_close_card, NULL};

/* What the flags must hold beyond their ranges; the break instance joins with its flag; and each instance's kernels
 * (of the group and of a row alone) for the states of the run, for its registers. */
static int index_configure(void *)
{
    const struct probe_flag *entries = probe_flag(INDEX_FLAG_ENTRIES), *segments = probe_flag(INDEX_FLAG_SEGMENT);
    const struct probe_flag *states = probe_flag(INDEX_FLAG_STATE), *rows = probe_flag(INDEX_FLAG_ROWS);
    const uint64_t capacity = probe_flag_number(INDEX_FLAG_CAPACITY, 0);
    int sound = capacity % INDEX_KEYS_PER_TILE == 0;
    for (uint32_t value = 0; value < entries->value_count; value++) {
        sound &= entries->values[value] % INDEX_KEYS_PER_TILE == 0 && entries->values[value] <= capacity;
    }
    for (uint32_t value = 0; value < segments->value_count; value++) {
        sound &= segments->values[value] % INDEX_KEYS_PER_TILE == 0;
    }
    if (!sound) {
        fprintf(stderr, "--capacity, every --entries and every --segment-keys are whole tiles of 8 entries, and no "
                        "--entries is above --capacity\n");
        return 0;
    }
    index_instances_in_run += probe_flag_given(INDEX_FLAG_BREAK_EXACTNESS) ? 1u : 0u;
    index_program.instance_count = index_instances_in_run;
    for (uint32_t instance = 0; instance < INDEX_INSTANCE_COUNT; instance++) {
        const uint32_t family = index_variants[instance].family;
        char *list = index_kernel_lists[instance];
        for (uint32_t value = 0; index_is_per_row(family) && value < rows->value_count; value++) {
            index_instances[instance].available &= rows->values[value] <= INDEX_MOST_GROUP_ROWS;   /* never unchecked */
        }
        for (uint32_t value = 0; value < states->value_count; value++) {
            const int form = index_form_of(instance, (uint32_t)states->values[value]);
            for (uint32_t group = 1; group <= INDEX_MOST_GROUP_ROWS; group += INDEX_MOST_GROUP_ROWS - 1u) {
                const uint32_t scan = index_scan_kernel(family, form, group);
                const uint32_t bound = index_bound_kernel(family, form, group);
                const int selects = family >= INDEX_ONE_BLOCK && family <= INDEX_SCAN_SELECT;
                const struct index_kernel *kernel = scan != INDEX_NO_KERNEL ? &index_scans[scan]
                                                    : bound != INDEX_NO_KERNEL ? &index_bounds[bound] : NULL;
                const char *names[] = {kernel != NULL && kernel->function != NULL ? kernel->name : "",
                                       selects ? INDEX_SELECT_KERNELS : ""};
                for (uint32_t index = 0; index < PROBE_COUNT_OF(names); index++) {
                    PROBE_REQUIRE(strlen(list) + strlen(names[index]) + 1u < sizeof(index_kernel_lists[instance]),
                                  "an instance's kernels");
                    if (names[index][0] != '\0' && strstr(list, names[index]) == NULL) {
                        strcat(strcat(list, list[0] != '\0' ? "+" : ""), names[index]);
                    }
                }
            }
        }
        index_instances[instance].kernel = list;
        index_card.configured[instance] = index_instances[instance].available;
    }
    if (!index_instances[INDEX_INSTANCE_row_major].available) {
        fprintf(stderr, "the per-row forms are checked up to 8 rows: none is timed or profiled in a run of more\n");
    }
    return 1;
}

int main(int argument_count, char **arguments) { return probe_main(argument_count, arguments, &index_program); }
