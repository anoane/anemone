/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * model_v41_test: DeepSeek-V4.1-Flash behind the model seam (source/model_v41.h): description, tensors, plan, arena
 * and frequency words, on packs of the model's names and shapes held as tables only, on a backend in host memory,
 * and, given its path, on the real pack's tables.
 *
 *   - The model pack is written here from the reference (REF and CFG as model_layout.h), not from model_v41.c: the
 *     embedding, then each layer as a stage unit of its tensors followed by 384 on-demand units of one routed expert,
 *     then the head; one extent a unit. Only block 0 and the tables exist: memory_storage.h holds them, and the pack
 *     opens at its whole size (about 278 GiB) without one extent byte.
 *   - Its plan on the five cards of first_token_plan.md 1.3 (layers 0-7, 8-15, 16-23, 24-31, 32-39; the embedding
 *     on the first, the head on the last): each tensor where the layout puts it, the expert tables and the frequency
 *     words of each image, every array inside its card's state, aligned and apart from every other, and the pools,
 *     key tiles, selections, rings and compressors of the table of 1.3, with their sizes. state_byte_count of the
 *     same placement gives each card's state, and 0 for host memory.
 *   - Each refusal once, with its status (and, where a name is the point, its one error line): missing arguments,
 *     another model, every limit out of range and beyond this build, a layout of another pack, weights that end off
 *     an array's boundary, a placement the loader refuses, a tensor missing or of another type, shape or format, an
 *     expert's matrix on another card, a layer on two cards, the norm apart from the head, and no memory (with the
 *     checking hooks). model_v41_host_experts_test holds plans with experts in host memory.
 *   - arena_initialise on a plan written here, on the backend in host memory with a stand-in of the fill kernel:
 *     on two cards with state and one without, each image then zeros over garbage, and the work in its order (the
 *     last word's digest, one fill shaped as backend_records.h says, the image's upload, the state's digest).
 *     Refused before any work: no card, plan or digest function, too many cards, an image longer than the state or
 *     off whole words. Then one fault at a time, with the staging memory left where the contract says: no staging,
 *     a byte a link changed and a zero a card did not keep (both caught by the digest), an upload that times out or
 *     fails, either digest failing, a fill that faults, and with no work begun an arena shorter than the plan's and
 *     a card with no build of the fill.
 *   - The draft (dspark_v1/DESIGN.md 5.2), on the pack with the draft head behind the head (model_v41_pack.h):
 *     without it the plan is byte for byte the one of the pack without a draft head (its units on a sixth card) and
 *     accepts a draft head that is broken or apart; with it, on the head's card, every other card's plan is byte for
 *     byte the plan without it, the result card holds the stages' tensors, expert tables (zeros from the 128th) and
 *     rings, the draft's globals and arrays (the round right behind the result area), apart and aligned, its state
 *     grown by exactly that, counted by state_byte_count; the thin program's session ranges gain exactly the three
 *     rings, identities 160, 164 and 168. Each missing or misshaped draft tensor, a draft tensor, stage, expert or tap
 *     layer off the head's card, a draft expert in host memory, draft limits the model has none of and a pack
 *     without a draft head are refused with one line. arena_initialise copies embed.weight into DRAFT_EMBEDDING piece
 *     by piece behind the states, and answers a changed byte, a failed download, upload or digest, and a plan whose
 *     copy is not inside its cards or a backend without downloads.
 *   - The frequency words bit for bit against the 64 written below, and within four places of the reference's
 *     expression in double precision.
 *   - With an argument, a pack's path: its tables opened through memory_storage.h, placed in order on five cards of
 *     the 1.3 budgets, planned and checked as above; the cards' layers, state and the words are printed.
 * A failure is one line on the standard output: the case, and what was expected of it.
 */
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "anemone/backend_records.h"
#include "anemone/expert_relaxed.h"
#include "log_expectation.h"
#include "anemone/hooks.h"
#include "anemone/index_scan.h"
#include "anemone/loader.h"
#include "anemone/pack.h"
#include "anemone/pack_digest.h"
#include "anemone/placement.h"
#include "anemone/relaxed_linear.h"
#include "anemone/storage_seam.h"
#include "anemone/weight_format_seam.h"
#include "host_memory_backend.h"
#include "memory_storage.h"
#include "model_v41.h"
#include "anemone/official_hc.h"
#include "model_v41_pack.h"

#define BLOCK PACK_BLOCK_BYTES
#define NAME_CAPACITY 64
#define CAPTURE_CAPACITY 4096u
#define LAYERS MODEL_LAYERS
#define EXPERTS MODEL_EXPERTS
#define CARDS TEST_MODEL_V41_CARDS
#define LAYERS_PER_CARD TEST_MODEL_V41_LAYERS_PER_CARD
#define UNIT_COUNT TEST_MODEL_V41_UNIT_COUNT
#define CONTEXT 16384u
#define ROWS 8u
#define WIDE_ROWS 1024u
#define SESSIONS 3u
#define MEBIBYTE 1048576.0
#define REGIONS_PER_LAYER 7u                         /* table, selection, ring, checkpoint, slots, pool, keys */
#define SHORTENED_BLOCKS 2u                                      /* below any card's last tensor */
#define POOLED_RATIO 2u
#define SMALL_CARDS 3u                                           /* with a large state, a small one, none */
#define SOURCE_CARDS 3u                                          /* the cards of layers 2, 8, 14 and 20 */
#define WRONG_OUTPUTS 5u
#define IMAGE_SHIFT 13u
#define SMALL_STATE_BYTES (4ull << 20u)                          /* zeros past the widest grid of a fill */
#define SMALL_WEIGHTS_BYTES 65536ull
#define SMALL_IMAGE_BYTES 102400ull
#define TINY_IMAGE_BYTES 256ull
#define TINY_ZERO_BYTES                           \
	((uint64_t)FILL_THREADS_PER_BLOCK_VALUE * \
	 FILL_WORD_BYTES) /* one block, exactly */
#define SHORT_ARENA_BYTES 4096ull
#define SKIPPED_WORD 54321u
#define GARBAGE_BYTE 0xABu
#define IMAGE_SEED 0x9E3779B9u
#define WAIT_LIMIT_NANOSECONDS 10000000000ull
#define CLOSE_LIMIT_NANOSECONDS 10000000000ull
#define POLLS_BEFORE_FINISHING 3u
#define FAILING_TICKET 2u                                        /* the image's: the garbage went up first */
#define FAILING_DIGEST 2u                                        /* the state's, behind the last word's */
#define FLIPPED_BYTE_OFFSET (SMALL_WEIGHTS_BYTES + 12345ull)     /* a byte of the image */
#define OFF_WORD_BYTES 4u                                        /* an image that is no whole number of words */
#define MISALIGNED_BYTES 16u                                     /* weights that end off an array's boundary */
#define FOLD_OUTPUTS 128u
#define FOLD_INPUTS 64u
#define FOLD_BITS_PER_WEIGHT 8u
#define FOLD_WEIGHT_BYTES ((uint64_t)FOLD_OUTPUTS * FOLD_INPUTS)
#define FOLD_SCALE_BYTES 8u                                      /* four output blocks, two input groups */
#define FOLD_GROUPS_OFFSET 16384ull
#define FOLD_CODE 122u
#define FOLD_GROUPED_REFERENCE 115                              /* largest code less the fp16 fold's seven */
#define EXPERT_FOLD_VALUES ((uint64_t)MODEL_HIDDEN * V41_DATA_EXPERT_INNER)
#define EXPERT_FOLD_WEIGHT_BYTES (EXPERT_FOLD_VALUES / 2u)
#define EXPERT_FOLD_SCALE_BYTES (EXPERT_FOLD_VALUES / LINEAR_VALUES_PER_BLOCK)
#define EXPERT_FOLD_MATRIX_BYTES (EXPERT_FOLD_WEIGHT_BYTES + EXPERT_FOLD_SCALE_BYTES)
#define EXPERT_FOLD_PHYSICAL_MATRICES 2u
#define WORD_PLACES 4.0                     /* a word within four places of the double reference: the band's ramp */
#define LOWEST_BAND_INDEX 15u                                    /* REF:383 low for the compressed theta */
#define HIGHEST_BAND_INDEX 25u                                   /* REF:384 high */
#define ROPE_FACTOR 16.0f
#define GIB (1ull << 30u)
#define CARD_170HX_BUDGET_MIB 63180ull                           /* 61.7 GiB: first_token_plan.md 1.3 */
#define CARD_PRO_BUDGET_MIB 95437ull                             /* 93.2 GiB */
#define BYTES_PER_MIB (1ull << 20u)
#define ARGUMENT_COUNT_WITH_A_PACK 2
#define DRAFT_UNIT_COUNT TEST_MODEL_V41_DRAFT_UNIT_COUNT
#define RESULT_CARD (CARDS - 1u)                                 /* of the five: the head's, and the draft's */
#define TAP_CARD (CARDS - 2u)                                    /* of the five: one that holds no tap layer */
#define DRAFT_CARDS (CARDS + 1u)                                 /* the five, and one for the draft head alone */
#define DRAFT_HOST_EXPERT 5u                                     /* a draft expert put in host memory */
#define SESSION_KINDS 4u                    /* a layer's session ranges (model_v41_program.c): identity 4 l + kind */
#define HOST_CARD_BYTES 4096u                                    /* an arena no launch reads: programs name offsets */
#define COPY_CARDS 2u                                            /* the embedding card, the result card */
#define COPY_FULL_PIECES 2u
#define COPY_PIECES (COPY_FULL_PIECES + 1u)                      /* the last of one block */
#define COPY_BYTES \
	(COPY_FULL_PIECES * V41_DATA_EMBEDDING_COPY_PIECE_BYTES + BLOCK)
#define COPY_SOURCE_OFFSET BLOCK                                 /* embed.weight in the embedding card's weights */
#define COPY_WEIGHTS_BYTES (COPY_SOURCE_OFFSET + COPY_BYTES + BLOCK)
#define COPY_PATTERN_SEED 0x2545F491u
#define COPY_FLIPPED_BYTE                      \
	(V41_DATA_EMBEDDING_COPY_PIECE_BYTES + \
	 777u) /* a byte of the second piece */
#define COPY_FIRST_DOWNLOAD 3u              /* card 0's tickets: embed.weight's upload, the image's, the copy's */
#define COPY_SECOND_UPLOAD 3u               /* card 1's: the image's, the first piece's, the second's */
#define COPY_SOURCE_DIGEST 3u               /* card 0's digests: the last word's, the state's, the source's */

static int failures = 0;
static const char *context_text = "";
static char capture_path[] = "/tmp/anemone_model_v41_test_XXXXXX";
static int capture_descriptor = -1;
static int error_stream_descriptor = -1;

static void expect(bool condition, const char *what)
{
    if (!condition) {
        (void)printf("FAILED: %s: %s\n", context_text, what);
        failures++;
    }
}

static void expect_status(result_status status, result_status expected,
			  const char *what)
{
    if (status != expected) {
	    (void)printf("FAILED: %s: %s: %s, expected %s\n", context_text,
			 what, status_text(status), status_text(expected));
	    failures++;
    }
}

static void capture_begin(void)
{
    expect(capture_descriptor >= 0 && fflush(stderr) == 0 && ftruncate(capture_descriptor, 0) == 0 &&
               lseek(capture_descriptor, 0, SEEK_SET) == 0 && dup2(capture_descriptor, STDERR_FILENO) == STDERR_FILENO,
           "the error stream is pointed at the temporary file");
}

/* What was logged since capture_begin is one error line that holds words. */
static void capture_end_expecting(const char *words, const char *what)
{
    char text[CAPTURE_CAPACITY];
    const bool is_open = capture_descriptor >= 0 && error_stream_descriptor >= 0 && fflush(stderr) == 0;
    const ssize_t byte_count = is_open ? pread(capture_descriptor, text, sizeof(text) - 1, 0) : -1;
    expect(is_open && dup2(error_stream_descriptor, STDERR_FILENO) == STDERR_FILENO && byte_count >= 0,
           "the temporary file is read back and the error stream restored");
    text[byte_count >= 0 ? (size_t)byte_count : 0] = '\0';
    if (!test_log_is_expected(text, words, true)) {
        (void)printf("FAILED: %s: %s: expected one error line with \"%s\", got \"%s\"\n", context_text, what, words,
                     text);
        failures++;
    }
}

/* ---- placements and plans ---- */

enum { SOURCE = TEST_MODEL_V41_SOURCE, POOLED = TEST_MODEL_V41_POOLED };

static const struct model_limits limits_of_m1 = { CONTEXT,
						  ROWS,
						  SESSIONS,
						  STATE_WIDTH_16,
						  LINEAR_PROFILE_PRECISION,
						  LINEAR_ORDER_FIXED,
						  CMP_ORDER_REFERENCE,
						  0,
						  0,
						  0 };
/* Chunks of wide steps: two slabs, the last of one row; the first qualification's; the most. */
static const uint32_t chunk_capacities[] = { ROWS + 1u, WIDE_ROWS,
					     MODEL_WIDE_MOST_ROWS };

/* plan_create on a placement of the pack's units on card_count cards; the plan, or NULL with the status in
 * status_out. */
static struct model_plan_record *
plan_of(const struct pack_record *pack, const uint32_t *card_of_unit,
	uint32_t card_count, const struct model_limits *limits,
	result_status *status_out, struct load_layout **layout_out)
{
	struct model_plan_record *plan = NULL;
	result_status status = load_layout_create(
		pack, card_of_unit, pack_header_of(pack)->unit_count,
		card_count, layout_out);
	status = status == STATUS_OK ?
			 v41_plan_create(pack, *layout_out, limits, &plan) :
			 status;
	*status_out = status;
	return plan;
}

/* ---- what a plan must hold ---- */

/* Where the layout puts the pack's tensor of that name: its arena offset, or UINT64_MAX. */
static uint64_t arena_offset_of(const struct pack_record *pack,
				const struct load_layout *layout,
				const char *name, uint64_t *byte_count_out)
{
	const uint32_t index =
		pack_find_tensor(pack, name, (uint32_t)strlen(name));
	if (index == PACK_NOT_FOUND) {
		return UINT64_MAX;
	}
	const struct pack_tensor *const tensor = &pack_tensors_of(pack)[index];
	*byte_count_out = tensor->byte_count;
	return layout->byte_offset_of_extent[tensor->extent_index] +
	       tensor->byte_offset_in_extent;
}

static void expect_place(const struct pack_record *pack,
			 const struct load_layout *layout,
			 const struct v41_tensor *place, const char *name,
			 const struct test_model_v41_tensor *spec)
{
    uint64_t byte_count = 0;
    const uint64_t offset = arena_offset_of(pack, layout, name, &byte_count);
    const bool is_right =
	    place->stored.byte_offset == offset &&
	    place->stored.byte_count == byte_count &&
	    place->card ==
		    layout->card_of_extent
			    [pack_tensors_of(pack)
				     [pack_find_tensor(pack, name,
						       (uint32_t)strlen(name))]
					     .extent_index] &&
	    place->shape.output_size == spec->dimensions[0] &&
	    place->shape.input_size ==
		    (spec->dimensions[1] == 0 ? 1u : spec->dimensions[1]) &&
	    place->shape.bits_per_weight == spec->bits_per_weight &&
	    (uint32_t)place->weight_format == spec->weight_format;
    if (!is_right) {
        (void)printf("FAILED: %s: the plan does not place %s where the layout puts it\n", context_text, name);
        failures++;
    }
}

/* One region of a card's state. */
struct region {
    uint64_t offset;
    uint64_t byte_count;
};

static int region_order(const void *first, const void *second)
{
    const uint64_t a = ((const struct region *)first)->offset;
    const uint64_t b = ((const struct region *)second)->offset;
    return a < b ? -1 : a > b ? 1 : 0;
}

#define ARRAY_SIZE_OF(array, row_bytes, fixed_bytes, holder, rows)                                                \
    {(uint64_t)(row_bytes), (uint64_t)(fixed_bytes), rows, holder},
static const struct array_size {
    uint64_t row_bytes;
    uint64_t fixed_bytes;
    uint32_t rows;
    uint32_t holder;
} array_sizes[V41_DATA_ARRAY_COUNT] = { V41_DATA_ARRAYS(ARRAY_SIZE_OF) };
#undef ARRAY_SIZE_OF

/* Whether an array serves the wide steps alone: the wide select's and the views. */
static bool is_wide_only(uint32_t array)
{
	return array == V41_DATA_ARRAY_VIEWS ||
	       array == V41_DATA_ARRAY_ROUTER_PICKS ||
	       array == V41_DATA_ARRAY_ROUTER_COUNTS ||
	       array == V41_DATA_ARRAY_ROUTER_PAIRS ||
	       array == V41_DATA_ARRAY_ROUTER_RUNS ||
	       array == V41_DATA_ARRAY_ROUTER_TILES;
}

/* Every region of a card inside its state, aligned, each where the one before it ends (so none is larger than its
 * size); the sizes of what model_v41.h fixes: a chunk's rows, a span's (at most 64) or a slab's (at most 8), the wide
 * select's arrays and the views only where a chunk holds more than one slab, the compressor's kv product a bfloat16
 * row on a card without a 2-to-1 source. */
static void expect_card_regions(const struct model_plan_record *plan,
				uint32_t card_index, uint64_t index_entries,
				uint32_t engram_tables)
{
	const struct v41_card *const card = &plan->cards[card_index];
	const uint64_t rows = plan->limits.row_capacity;
	const uint64_t slab_rows =
		rows < LINEAR_MAXIMUM_ROWS ? rows : LINEAR_MAXIMUM_ROWS;
	const uint64_t span_rows =
		rows < V41_DATA_SPAN_ROWS ? rows : V41_DATA_SPAN_ROWS;
	const bool is_pooled =
		(card->layers & test_model_v41_layers_of_kind[POOLED]) != 0;
	const uint64_t slabs = (rows + slab_rows - 1u) / slab_rows;
	const uint64_t wide_jobs =
		(uint64_t)ROUTER_WIDE_JOBS(plan->limits.row_capacity);
	const uint64_t thin_jobs = slab_rows * MODEL_EXPERTS_PER_ROW;
	const uint64_t jobs = thin_jobs > wide_jobs ? thin_jobs : wide_jobs;
	const uint64_t sessions = plan->limits.session_capacity;
	static struct region regions[V41_DATA_ARRAY_COUNT +
				     V41_DATA_WALK_LAYERS * REGIONS_PER_LAYER +
				     1u];
	uint32_t count = 0, misplaced = 0;
	for (uint32_t array = 0; array < V41_DATA_ARRAY_COUNT; array++) {
		const uint64_t array_rows =
			array_sizes[array].rows == V41_DRAFT_ROWS ?
				plan->limits.draft_positions :
			array_sizes[array].rows == V41_SLAB ? slab_rows :
			array_sizes[array].rows == V41_SPAN ? span_rows :
							      rows;
		uint64_t bytes = array_sizes[array].fixed_bytes +
				 array_rows * array_sizes[array].row_bytes;
		bytes = array == V41_DATA_ARRAY_SESSIONS ?
				sessions * STEP_SESSION_BYTES :
			array == V41_DATA_ARRAY_COMPRESSOR_KV && !is_pooled ?
				rows * MODEL_HEAD_DIMS * sizeof(uint16_t) :
			array == V41_DATA_ARRAY_INDEX_SCORES ?
				slab_rows *
					(index_entries * sizeof(uint16_t) +
					 (index_entries >=
							  INDEX_SPLIT_MIN_ENTRIES ?
						  INDEX_SPLIT_ROW_BYTES :
						  0u)) :
			array == V41_DATA_ARRAY_ENGRAM_ROWS ?
				engram_tables * rows *
					V41_DATA_ENGRAM_ROWS_PER_TOKEN *
					ENGRAM_ROW_BYTES :
			array == V41_DATA_ARRAY_JOBS ?
				jobs * EXPERT_JOB_BYTES :
			array == V41_DATA_ARRAY_ROUTER_COUNTS ?
				(rows + ROUTER_SORT_ROWS - 1u) /
					ROUTER_SORT_ROWS * EXPERTS *
					sizeof(uint32_t) :
			array == V41_DATA_ARRAY_ROUTER_TILES ?
				ROUTER_WIDE_TILES(rows) *
					sizeof(struct wide_expert_tile) :
			array == V41_DATA_ARRAY_VIEWS /* the slabs', then split_overlap's two halves' (7a6a332) */
				?
				slabs * (STEP_BLOCK_BYTES_VALUE +
					 slab_rows * STEP_ROW_BYTES_VALUE) +
					V41_DATA_HALF_VIEWS *
						STEP_BLOCK_BYTES_VALUE +
					rows * STEP_ROW_BYTES_VALUE :
				bytes;
		misplaced += is_wide_only(array) && slabs == 1u &&
					     card->array_offset[array] !=
						     PROGRAM_NOWHERE ?
				     1u :
				     0u;
		if (card->array_offset[array] != PROGRAM_NOWHERE) {
			regions[count++] =
				(struct region){ card->array_offset[array],
						 bytes };
		}
	}
    expect(misplaced == 0, "a chunk of one slab gives no array of the wide select and no views");
    for (uint32_t layer = 0; layer < V41_DATA_WALK_LAYERS; layer++) {
	    const struct region per_layer[REGIONS_PER_LAYER] = {
		    { card->expert_table_offset[layer],
		      MODEL_EXPERT_OFFSETS_BYTES },
		    { card->selection_offset[layer],
		      rows * MODEL_SELECTION_ROW_BYTES },
		    { card->ring_offset[layer],
		      sessions * MODEL_RING_SLOTS * MODEL_STATE_ROW_BYTES },
		    { card->checkpoint_offset[layer],
		      sessions * (MODEL_WINDOW - 1u) * MODEL_STATE_ROW_BYTES },
		    { card->compressor_offset[layer],
		      sessions * V41_DATA_COMPRESSOR_SESSION_BYTES },
		    { card->pool_offset[layer],
		      sessions * plan->pool_session_bytes[layer] },
		    { card->key_tiles_offset[layer],
		      sessions * plan->key_session_bytes[layer] }
	    };
	    for (uint32_t index = 0; index < REGIONS_PER_LAYER; index++) {
		    if (per_layer[index].offset != PROGRAM_NOWHERE) {
			    regions[count++] = per_layer[index];
		    }
	    }
    }
    if (card->layers != 0 || (card->holders & (1u << V41_DRAFT)) != 0) {
	    expect(card->producer_arrivals_offset ==
				   card->front_scratch_offset +
					   SITE_FRONT_SCRATCH_BYTES &&
			   card->front_scratch_offset <=
				   card->arena_byte_count &&
			   V41_DATA_FUSED_SCRATCH_BYTES <=
				   card->arena_byte_count -
					   card->front_scratch_offset,
		   "thin front scratch and producer counters are disjoint and inside the arena");
	    if (rows < MODEL_WIDE_LEAST_ROWS) {
		    regions[count++] =
			    (struct region){ card->front_scratch_offset,
					     V41_DATA_FUSED_SCRATCH_BYTES };
	    } else {
		    expect(card->front_scratch_offset ==
					   card->array_offset
						   [V41_DATA_ARRAY_ROUTER_PICKS] &&
				   card->front_scratch_offset +
						   V41_DATA_FUSED_SCRATCH_BYTES <=
					   card->array_offset
						   [V41_DATA_ARRAY_JOBS],
			   "wide-capacity plans reuse only wide router scratch, ending before the thin job table");
	    }
    } else {
	    expect(card->front_scratch_offset == PROGRAM_NOWHERE &&
			   card->producer_arrivals_offset == PROGRAM_NOWHERE,
		   "a card without target or draft layers has no fusion scratch");
    }
    qsort(regions, count, sizeof(regions[0]), region_order);
    const uint64_t alignment = V41_DATA_ARRAY_ALIGNMENT_BYTES;
    bool packed = count != 0 && regions[0].offset == card->state_offset;
    for (uint32_t index = 0; index < count; index++) {
        const uint64_t aligned_end = (regions[index].offset + regions[index].byte_count + alignment - 1u) / alignment *
                                     alignment;
        packed = packed && regions[index].offset % alignment == 0 && regions[index].byte_count != 0 &&
                 aligned_end == (index + 1u == count ? card->arena_byte_count : regions[index + 1u].offset);
    }
    expect(packed || card->layers == 0,
           "every array of a card lies in its state, aligned, each where the one before it ends: of its size alone");
    expect(card->image_byte_count ==
		   (card->layers == 0 ?
			    0 :
			    card->array_offset[V41_DATA_ARRAY_FREQUENCIES] +
				    sizeof(float[MODEL_THETAS]
						[MODEL_FREQUENCIES]) -
				    card->state_offset),
	   "the image is the expert tables and the frequency words");
}

/* The plan of the 1.3 placement against the layout, the reference and that table. */
static void expect_plan_of_five_cards(const struct pack_record *pack,
				      const struct load_layout *layout,
				      const struct model_plan_record *plan)
{
    /* first_token_plan.md 1.3: bit l of each card's pools, key tiles and selections */
    static const uint64_t pools[CARDS] = {0x4ull, 0x4100ull, 0x104000ull, 0x100000ull, 0x100000ull};
    static const uint64_t key_tiles[CARDS] = {0x4ull, 0x4100ull, 0x100000ull, 0x100000ull, 0x100000ull};
    static const uint64_t selections[CARDS] = {0x4ull, 0x4100ull, 0x104000ull, 0x11000000ull, 0x1100000000ull};
    static const uint64_t index_entries[CARDS] = {CONTEXT / POOLED_RATIO, CONTEXT / POOLED_RATIO, CONTEXT, CONTEXT,
                                                  CONTEXT};
    static const uint32_t engram_tables[CARDS] = {1u, 2u, 0, 0, 0};
    float words[MODEL_THETAS][MODEL_FREQUENCIES];
    char name[NAME_CAPACITY];
    v41_frequency_words(words);
    expect(plan->card_count == CARDS && plan->embedding_card == 0 && plan->result_card == CARDS - 1u,
           "the cards of the embedding and of the head");
    for (uint32_t index = 0; index < V41_DATA_GLOBALS; index++) {
	    const struct test_model_v41_tensor *const spec =
		    &test_model_v41_global_tensors[index];
	    expect_place(pack, layout, &plan->globals[spec->place], spec->name,
			 spec);
    }
    for (uint32_t layer = 0; layer < LAYERS; layer++) {
	    const struct v41_card *const card =
		    &plan->cards[layer / LAYERS_PER_CARD];
	    expect(plan->layer_card[layer] == layer / LAYERS_PER_CARD &&
			   (card->layers >> layer & 1u) != 0,
		   "a layer runs on its card");
	    for (uint32_t index = 0; index < TEST_MODEL_V41_LAYER_TENSORS;
		 index++) {
		    const struct test_model_v41_tensor *const spec =
			    &test_model_v41_layer_tensors[index];
		    if (spec->place == MODEL_LAYERS) {
			    continue; /* in the pack, read by no program */
		    }
		    const struct v41_tensor *const place =
			    &plan->layer_tensors[layer][spec->place];
		    (void)snprintf(name, sizeof(name), "layers.%" PRIu32 ".%s",
				   layer, spec->name);
		    if ((test_model_v41_layers_of_kind[spec->layers] >> layer &
			 1u) != 0) {
			    expect_place(pack, layout, place, name, spec);
		    } else {
			    expect(place->stored.byte_count == 0 &&
					   place->card == 0,
				   "a tensor a layer lacks is all zero");
		    }
        }
        uint64_t first = UINT64_MAX, end = 0;
        bool table_is_right = card->image != NULL;
        for (uint32_t expert = 0; expert < EXPERTS && table_is_right; expert++) {
		for (uint32_t index = 0; index < V41_DATA_EXPERT_TENSOR_COUNT;
		     index++) {
			const struct test_model_v41_tensor *const spec =
				&test_model_v41_expert_tensors[index];
			uint64_t byte_count = 0, entry = 0;
			(void)snprintf(name, sizeof(name),
				       "layers.%" PRIu32 ".ffn.experts.%" PRIu32
				       ".%s",
				       layer, expert, spec->name);
			const uint64_t offset = arena_offset_of(
				pack, layout, name, &byte_count);
			memcpy(&entry,
			       card->image +
				       (card->expert_table_offset[layer] -
					card->state_offset) +
				       (expert * MODEL_EXPERT_MATRICES +
					spec->place) *
					       sizeof(uint64_t),
			       sizeof(entry));
			table_is_right = table_is_right && entry == offset;
			first = offset < first ? offset : first;
			end = offset + byte_count > end ? offset + byte_count :
							  end;
		}
	}
        expect(table_is_right, "the expert table holds the offsets of w1, w3 and w2 of each expert");
        expect(plan->experts[layer].byte_offset == first && plan->experts[layer].byte_count == end - first,
               "the experts' range holds every expert matrix of the layer and no more");
        const bool is_source = (test_model_v41_layers_of_kind[SOURCE] >> layer & 1u) != 0;
        const uint64_t entries =
            (test_model_v41_layers_of_kind[POOLED] >> layer & 1u) != 0 ? CONTEXT / POOLED_RATIO : CONTEXT;
	expect(plan->pool_session_bytes[layer] ==
			       (is_source ? entries * MODEL_STATE_ROW_BYTES :
					    0) &&
		       plan->key_session_bytes[layer] ==
			       (is_source ? entries * MODEL_KEY_BYTES : 0),
	       "a source's pool and key tiles hold its entries of the context");
    }
    for (uint32_t card_index = 0; card_index < CARDS; card_index++) {
	    const struct v41_card *const card = &plan->cards[card_index];
	    uint64_t pools_held = 0, keys_held = 0, selections_held = 0,
		     rings = 0, compressors = 0;
	    for (uint32_t layer = 0; layer < LAYERS; layer++) {
		    pools_held |= card->pool_offset[layer] != PROGRAM_NOWHERE ?
					  1ull << layer :
					  0;
		    keys_held |= card->key_tiles_offset[layer] !=
						 PROGRAM_NOWHERE ?
					 1ull << layer :
					 0;
		    selections_held |= card->selection_offset[layer] !=
						       PROGRAM_NOWHERE ?
					       1ull << layer :
					       0;
		    rings |= card->ring_offset[layer] != PROGRAM_NOWHERE ?
				     1ull << layer :
				     0;
		    compressors |= card->compressor_offset[layer] !=
						   PROGRAM_NOWHERE ?
					   1ull << layer :
					   0;
	    }
        expect(pools_held == pools[card_index] && keys_held == key_tiles[card_index] &&
                   selections_held == selections[card_index],
               "the pools, key tiles and selections of the table of 1.3");
        expect(rings == card->layers &&
                   compressors == (card->layers & test_model_v41_layers_of_kind[POOLED]) &&
                   card->engram_tables == engram_tables[card_index],
               "a ring for each layer, the slots of each 2-to-1 source, the n-gram tables of its layers");
        expect(card->state_offset == layout->card_byte_count[card_index], "the state begins behind the weights");
        const uint64_t *const arrays = card->array_offset;
	expect((arrays[V41_DATA_ARRAY_LOGITS] != PROGRAM_NOWHERE) ==
			       (card_index == CARDS - 1u) &&
		       (arrays[V41_DATA_ARRAY_ENGRAM_ROWS] !=
			PROGRAM_NOWHERE) == (engram_tables[card_index] != 0) &&
		       (arrays[V41_DATA_ARRAY_LATENT] != PROGRAM_NOWHERE) ==
			       (card_index < SOURCE_CARDS) &&
		       (arrays[V41_DATA_ARRAY_COMPRESSOR_SCORE] !=
			PROGRAM_NOWHERE) ==
			       ((card->layers &
				 test_model_v41_layers_of_kind[POOLED]) != 0),
	       "the head's, the n-gram layers', the sources' and the 2-to-1 sources' arrays on their cards alone");
	unsigned char word_bytes[sizeof(words)];
        memcpy(word_bytes, words, sizeof(words));
	expect(card->image != NULL &&
		       memcmp(card->image +
				      (card->array_offset
					       [V41_DATA_ARRAY_FREQUENCIES] -
				       card->state_offset),
			      word_bytes, sizeof(word_bytes)) == 0,
	       "the image holds the frequency words");
	expect_card_regions(plan, card_index, index_entries[card_index], engram_tables[card_index] != 0 ? 1u : 0u);
    }
}

/* ---- the cases ---- */

static void test_five_cards(const struct pack_record *pack)
{
    static uint32_t card_of_unit[UNIT_COUNT];
    struct load_layout *layout = NULL;
    result_status status = STATUS_OK;
    struct model_description_record description = { 0 };
    context_text = "the model's pack on the five cards of 1.3";
    expect_status(v41_describe(pack, &description), STATUS_OK,
		  "the pack is described");
    expect(description.layer_count == LAYERS &&
		   description.hidden_size == MODEL_HIDDEN &&
		   description.vocabulary_size == MODEL_VOCABULARY_VALUE &&
		   description.context_token_limit == V41_DATA_CONTEXT_LIMIT &&
		   description.end_token_id == 1u,
	   "40 layers, 5,120, 129,280, 1,048,576 tokens, the end token 1");
    test_model_v41_five_cards(card_of_unit, UNIT_COUNT, 0);
    struct model_plan_record *const plan =
	    plan_of(pack, card_of_unit, CARDS, &limits_of_m1, &status, &layout);
    expect_status(status, STATUS_OK, "the plan is made");
    if (plan != NULL) {
        expect_plan_of_five_cards(pack, layout, plan);
	const struct placement_record placement = { card_of_unit, UNIT_COUNT,
						    CARDS };
	for (uint32_t card = 0; card < CARDS; card++) {
            uint64_t state = 0, arena = 0;
	    expect_status(v41_state_byte_count(pack, &placement, &limits_of_m1,
					       card, &state),
			  STATUS_OK, "a card's state is counted");
	    expect_status(v41_arena_byte_count(plan, card, &arena), STATUS_OK,
			  "an arena is counted");
	    expect(state == plan->cards[card].arena_byte_count - plan->cards[card].state_offset &&
                       arena == plan->cards[card].arena_byte_count,
                   "state_byte_count gives the plan's state behind the weights");
        }
        uint64_t host_state = 1;
	expect_status(v41_state_byte_count(pack, &placement, &limits_of_m1,
					   PLACEMENT_IN_HOST_MEMORY,
					   &host_state),
		      STATUS_OK, "host memory is asked for");
	expect(host_state == 0, "the state takes no host memory");
    }
    v41_plan_destroy(plan);
    load_layout_destroy(layout);
}

/* Plans whose row capacity is a wide step's chunk, on the same placement: every array where model_v41.h puts it and
 * of its size for a chunk's rows or a slab's; the state of each card printed against the plan of 8 rows. */
static void test_chunk_plans(const struct pack_record *pack)
{
    static uint32_t card_of_unit[UNIT_COUNT];
    uint64_t thin_state[CARDS] = {0};
    context_text = "plans for the chunks of wide steps";
    test_model_v41_five_cards(card_of_unit, UNIT_COUNT, 0);
    for (size_t index = 0; index <= sizeof(chunk_capacities) / sizeof(chunk_capacities[0]); index++) {
	    struct model_limits limits = limits_of_m1;
	    struct load_layout *layout = NULL;
	    result_status status = STATUS_OK;
	    limits.row_capacity = index == 0 ? ROWS :
					       chunk_capacities[index - 1u];
	    struct model_plan_record *const plan = plan_of(
		    pack, card_of_unit, CARDS, &limits, &status, &layout);
	    expect_status(status, STATUS_OK,
			  "a plan is made for a chunk's rows");
	    for (uint32_t card = 0; card < CARDS && plan != NULL; card++) {
		    const uint64_t state = plan->cards[card].arena_byte_count -
					   plan->cards[card].state_offset;
		    thin_state[card] = index == 0 ? state : thin_state[card];
		    if (index != 0) {
			    (void)printf(
				    "  a chunk of %" PRIu32
				    " rows: card %" PRIu32
				    " holds %.1f MiB of state, %.1f MiB more "
				    "than at %u rows\n",
				    limits.row_capacity, card,
				    (double)state / MEBIBYTE,
				    (double)(state - thin_state[card]) /
					    MEBIBYTE,
				    ROWS);
		    }
        }
        if (plan != NULL) {
            expect_plan_of_five_cards(pack, layout, plan);
        }
	v41_plan_destroy(plan);
	load_layout_destroy(layout);
    }
}

/* A plan that must be refused with that status, and the words of its error line where they are given. */
static void expect_refused_plan(const struct pack_record *pack,
				const uint32_t *card_of_unit,
				const struct model_limits *limits,
				result_status expected, const char *words,
				const char *what)
{
	struct load_layout *layout = NULL;
	struct model_plan_record *plan = NULL;
	result_status status = load_layout_create(
		pack, card_of_unit, pack_header_of(pack)->unit_count, CARDS,
		&layout);
	expect_status(status, STATUS_OK, "the loader lays the placement out");
	capture_begin();
	status = layout == NULL ? status :
				  v41_plan_create(pack, layout, limits, &plan);
	capture_end_expecting(words, what);
	expect_status(status, expected, what);
	expect(plan == NULL,
	       "a refused plan_create leaves its output untouched");
	v41_plan_destroy(plan);
	load_layout_destroy(layout);
}

#define REMOVE TEST_MODEL_V41_CHANGE_REMOVE
#define ELEMENT TEST_MODEL_V41_CHANGE_ELEMENT
#define OUTPUTS TEST_MODEL_V41_CHANGE_OUTPUTS
#define FORMAT TEST_MODEL_V41_CHANGE_FORMAT
#define MOVE TEST_MODEL_V41_CHANGE_UNIT
struct refused_pack {
    struct test_model_v41_change change;
    result_status expected;
    const char *words;
};

static void test_tensor_refusals(void)
{
    static uint32_t card_of_unit[UNIT_COUNT];
    static const struct refused_pack cases[] = {
	    { { REMOVE, "layers.17.attn.wo_b.weight", 0 },
	      STATUS_CORRUPT_PACK,
	      "layers.17.attn.wo_b.weight" },
	    { { REMOVE, "layers.39.ffn.experts.383.w2.weight", 0 },
	      STATUS_CORRUPT_PACK,
	      "layers.39.ffn.experts.383.w2.weight" },
	    { { REMOVE, "head.weight", 0 }, STATUS_CORRUPT_PACK, "head.weight" },
	    { { REMOVE, "layers.8.attn.compressor.wgate.weight", 0 },
	      STATUS_CORRUPT_PACK,
	      "wgate" },
	    { { ELEMENT, "layers.20.attn.attn_sink", PACK_ELEMENT_BFLOAT16 },
	      STATUS_CORRUPT_PACK,
	      "layers.20.attn.attn_sink" },
	    { { OUTPUTS, "layers.14.engram.q_weight", WRONG_OUTPUTS },
	      STATUS_CORRUPT_PACK,
	      "engram.q_weight" },
	    { { FORMAT, "layers.36.attn.indexer.wq_b.weight",
		WEIGHT_FORMAT_NONE },
	      STATUS_CORRUPT_PACK,
	      "indexer.wq_b" },
	    { { TEST_MODEL_V41_CHANGE_MODEL, NULL, 0 },
	      STATUS_UNSUPPORTED,
	      "another model" },
	    { { MOVE, "norm.weight", 0 },
	      STATUS_UNSUPPORTED,
	      "the head with the norm" },
	    { { MOVE, "layers.3.ffn_norm.weight",
		TEST_MODEL_V41_LAYER_UNIT(LAYERS_PER_CARD) },
	      STATUS_UNSUPPORTED,
	      "layer 3 is not" },
	    { { MOVE, "layers.5.ffn.experts.0.w2.weight",
		TEST_MODEL_V41_LAYER_UNIT(LAYERS_PER_CARD) },
	      STATUS_UNSUPPORTED,
	      "expert 0 of layer 5 is neither" },
    };
    test_model_v41_five_cards(card_of_unit, UNIT_COUNT, 0);
    for (size_t index = 0; index < sizeof(cases) / sizeof(cases[0]); index++) {
        context_text = cases[index].words;
	struct pack_record *pack = NULL;
	expect_status(test_model_v41_pack_open(cases[index].change, &pack),
		      STATUS_OK, "the changed pack opens from its tables");
	if (pack != NULL) {
            expect_refused_plan(pack, card_of_unit, &limits_of_m1, cases[index].expected, cases[index].words,
                                "a pack the model cannot run is refused with the tensor's name");
        }
	pack_close(pack);
    }
}

/* ---- the draft (dspark_v1/DESIGN.md 5.2) ---- */

static struct model_limits draft_limits(uint32_t row_capacity,
					uint32_t draft_positions)
{
	struct model_limits limits = limits_of_m1;
	limits.row_capacity = row_capacity;
	limits.draft_positions = draft_positions;
	return limits;
}

/* Two plans' cards hold the same bytes: every member, and the image's bytes in place of its address. */
static bool cards_equal(const struct v41_card *first,
			const struct v41_card *second)
{
	struct v41_card copy = *first;
	const bool images_are_equal =
		first->image_byte_count == second->image_byte_count &&
		(first->image_byte_count == 0 ||
		 memcmp(first->image, second->image,
			(size_t)first->image_byte_count) == 0);
	copy.image = second->image;
	return images_are_equal && memcmp(&copy, second, sizeof(copy)) == 0;
}

/* Tensor records gained a scale-fold word after their last naturally aligned member. Compare values, not padding. */
static bool tensors_equal(const struct v41_tensor *first,
			  const struct v41_tensor *second)
{
    return first->stored.byte_offset == second->stored.byte_offset &&
           first->stored.byte_count == second->stored.byte_count &&
           memcmp(&first->shape, &second->shape, sizeof(first->shape)) == 0 &&
           first->weight_format == second->weight_format && first->card == second->card &&
           first->fold_reference == second->fold_reference;
}

/* The plan of the pack with the draft head, the draft off and its units on a sixth card, is byte for byte the plan
 * of the pack without one on the five cards: every member (the struct has padding, so member by member), no draft
 * member set, no card of the five changed. */
static bool plan_is_plain(const struct model_plan_record *draft_off,
			  const struct model_plan_record *plain)
{
#define SAME(member) (memcmp(&draft_off->member, &plain->member, sizeof(plain->member)) == 0)
    bool is_plain = draft_off->card_count == DRAFT_CARDS && plain->card_count == CARDS && SAME(limits) &&
                    SAME(experts) && SAME(expert_tables) && SAME(host_expert_offsets) && SAME(expert_matrix_offsets) &&
                    SAME(expert_byte_count) && SAME(slot_byte_count) && SAME(host_expert_count) &&
                    SAME(pool_session_bytes) && SAME(key_session_bytes) && SAME(layer_card) && SAME(embedding_card) &&
                    SAME(result_card) && SAME(reserved_zero);
#undef SAME
    for (uint32_t layer = 0; layer < V41_DATA_WALK_LAYERS; layer++) {
	    for (uint32_t tensor = 0; tensor < V41_DATA_LAYER_TENSOR_COUNT;
		 tensor++) {
		    is_plain = is_plain &&
			       tensors_equal(
				       &draft_off->layer_tensors[layer][tensor],
				       &plain->layer_tensors[layer][tensor]);
	    }
    }
    for (uint32_t tensor = 0; tensor < V41_DATA_GLOBALS; tensor++) {
	    is_plain = is_plain && tensors_equal(&draft_off->globals[tensor],
						 &plain->globals[tensor]);
    }
    for (uint32_t tensor = 0; tensor < V41_DATA_DRAFT_GLOBALS; tensor++) {
	    is_plain = is_plain &&
		       tensors_equal(&draft_off->draft_globals[tensor],
				     &plain->draft_globals[tensor]);
    }
    for (uint32_t card = 0; card < CARDS; card++) {
        is_plain = is_plain && cards_equal(&draft_off->cards[card], &plain->cards[card]);
    }
    return is_plain;
}

static void test_draft_off(const struct pack_record *plain)
{
    static uint32_t plain_units[UNIT_COUNT], draft_units[DRAFT_UNIT_COUNT];
    static const uint32_t row_capacities[] = {ROWS, WIDE_ROWS};
    struct pack_record *pack = NULL;
    context_text = "the pack with a draft head, the draft off";
    expect_status(test_model_v41_draft_pack_open(
			  (struct test_model_v41_change){
				  TEST_MODEL_V41_CHANGE_NONE, NULL, 0 },
			  &pack),
		  STATUS_OK, "the pack opens from its tables");
    test_model_v41_five_cards(plain_units, UNIT_COUNT, 0);
    test_model_v41_five_cards(draft_units, UNIT_COUNT, 0);
    test_model_v41_draft_place(draft_units, CARDS);
    for (size_t index = 0; pack != NULL && index < sizeof(row_capacities) / sizeof(row_capacities[0]); index++) {
	    const struct model_limits limits =
		    draft_limits(row_capacities[index], 0);
	    struct load_layout *plain_layout = NULL, *layout = NULL;
	    result_status plain_status = STATUS_OK, status = STATUS_OK;
	    struct model_plan_record *const plain_plan =
		    plan_of(plain, plain_units, CARDS, &limits, &plain_status,
			    &plain_layout);
	    struct model_plan_record *const plan = plan_of(
		    pack, draft_units, DRAFT_CARDS, &limits, &status, &layout);
	    expect(plain_status == STATUS_OK && status == STATUS_OK,
		   "both plans are made");
	    expect(plan != NULL && plain_plan != NULL &&
			   plan_is_plain(plan, plain_plan) &&
			   plan->cards[CARDS].arena_byte_count ==
				   plan->cards[CARDS].state_offset &&
			   plan->cards[CARDS].holders == 0,
		   "without the draft the draft head is not read: the plan is byte for byte the one without it");
	    v41_plan_destroy(plain_plan);
	    v41_plan_destroy(plan);
	    load_layout_destroy(plain_layout);
	    load_layout_destroy(layout);
    }
    pack_close(pack);
}

/* Bytes an array or a region takes in the state: whole aligned blocks. */
static uint64_t aligned_bytes(uint64_t byte_count)
{
	return (byte_count + V41_DATA_ARRAY_ALIGNMENT_BYTES - 1u) /
	       V41_DATA_ARRAY_ALIGNMENT_BYTES * V41_DATA_ARRAY_ALIGNMENT_BYTES;
}

/* The draft's stage of a walk layer, against the layout and the pack: its tensors, its experts' table in the result
 * card's image (entries from the 128th zero) and their range, its ring and table on the result card alone. */
static void expect_draft_stage(const struct pack_record *pack,
			       const struct load_layout *layout,
			       const struct model_plan_record *plan,
			       uint32_t stage)
{
	const uint32_t layer = MODEL_DRAFT_FIRST_LAYER + stage;
	const struct v41_card *const result = &plan->cards[RESULT_CARD];
	char name[NAME_CAPACITY];
	for (uint32_t index = 0; index < TEST_MODEL_V41_DRAFT_STAGE_TENSORS;
	     index++) {
		const struct test_model_v41_tensor *const spec =
			&test_model_v41_draft_stage_tensors[index];
		(void)snprintf(name, sizeof(name), "mtp.%" PRIu32 ".%s", stage,
			       spec->name);
		if (spec->place != MODEL_LAYERS) {
			expect_place(pack, layout,
				     &plan->layer_tensors[layer][spec->place],
				     name, spec);
		}
    }
    bool has_none_else = true;
    for (uint32_t tensor = V41_DATA_COMPRESSOR;
	 tensor < V41_DATA_LAYER_TENSOR_COUNT; tensor++) {
	    has_none_else =
		    has_none_else &&
		    plan->layer_tensors[layer][tensor].stored.byte_count == 0;
    }
    expect(has_none_else && plan->layer_card[layer] == RESULT_CARD && plan->host_expert_count[layer] == 0 &&
               plan->pool_session_bytes[layer] == 0 && plan->key_session_bytes[layer] == 0,
           "a draft stage runs on the result card with no compressor, indexer, n-gram rows, pool or host experts");
    uint64_t first = UINT64_MAX, end = 0;
    bool table_is_right = result->image != NULL;
    for (uint32_t expert = 0; expert < EXPERTS && table_is_right; expert++) {
	    for (uint32_t index = 0; index < V41_DATA_EXPERT_TENSOR_COUNT;
		 index++) {
		    const struct test_model_v41_tensor *const spec =
			    &test_model_v41_expert_tensors[index];
		    uint64_t byte_count = 0, entry = 0, offset = 0;
		    (void)snprintf(name, sizeof(name),
				   "mtp.%" PRIu32 ".ffn.experts.%" PRIu32 ".%s",
				   stage, expert, spec->name);
		    offset = expert < MODEL_DRAFT_EXPERTS ?
				     arena_offset_of(pack, layout, name,
						     &byte_count) :
				     0;
		    memcpy(&entry,
			   result->image +
				   (result->expert_table_offset[layer] -
				    result->state_offset) +
				   (expert * MODEL_EXPERT_MATRICES +
				    spec->place) *
					   sizeof(uint64_t),
			   sizeof(entry));
		    table_is_right = table_is_right && entry == offset;
		    first = expert < MODEL_DRAFT_EXPERTS && offset < first ?
				    offset :
				    first;
		    end = offset + byte_count > end ? offset + byte_count : end;
	    }
    }
    expect(table_is_right, "a stage's expert table holds its 128 experts' offsets, then zeros");
    expect(plan->experts[layer].byte_offset == first && plan->experts[layer].byte_count == end - first,
           "a stage's experts' range holds every matrix of its experts and no more");
    bool is_placed = true;
    for (uint32_t card = 0; card < CARDS; card++) {
	    const struct v41_card *const planned = &plan->cards[card];
	    const bool is_result = card == RESULT_CARD;
	    is_placed = is_placed &&
			(planned->ring_offset[layer] != PROGRAM_NOWHERE) ==
				is_result &&
			(planned->expert_table_offset[layer] !=
			 PROGRAM_NOWHERE) == is_result &&
			planned->selection_offset[layer] == PROGRAM_NOWHERE &&
			planned->compressor_offset[layer] == PROGRAM_NOWHERE &&
			planned->pool_offset[layer] == PROGRAM_NOWHERE &&
			planned->key_tiles_offset[layer] == PROGRAM_NOWHERE;
    }
    expect(is_placed, "a stage has a ring and an expert table on the result card, nothing else anywhere");
}

/* The result card adds the draft's arrays and stages, including prompt-end window checkpoint storage. */
static void expect_draft_plan(const struct pack_record *pack,
			      const struct load_layout *layout,
			      const struct model_plan_record *off,
			      const struct model_plan_record *on)
{
	const struct v41_card *const result = &on->cards[RESULT_CARD],
				     *const plain = &off->cards[RESULT_CARD];
	const uint64_t sessions = on->limits.session_capacity,
		       rows = on->limits.row_capacity;
	uint64_t growth =
		MODEL_DRAFT_STAGES *
		(aligned_bytes(MODEL_EXPERT_OFFSETS_BYTES) +
		 aligned_bytes(sessions * V41_DATA_RING_SESSION_BYTES) +
		 aligned_bytes(sessions * V41_DATA_CHECKPOINT_SESSION_BYTES));
	for (uint32_t card = 0; card < RESULT_CARD; card++) {
		expect(cards_equal(&on->cards[card], &off->cards[card]),
		       "a card without the draft keeps its plan");
	}
    /* the draft and its copy of the embedding (1afdd88: V41_DRAFT_COPY, the result card's with the draft) */
    const uint32_t draft_holders = (1u << V41_DRAFT) | (1u << V41_DRAFT_COPY);
    expect(result->layers == plain->layers &&
		   result->holders == (plain->holders | draft_holders) &&
		   v41_walk_layers(result) ==
			   (plain->layers | V41_DATA_DRAFT_LAYERS),
	   "the result card holds the draft: its card keeps the 40 layers' bits, the holder marks the stages");
    for (uint32_t stage = 0; stage < MODEL_DRAFT_STAGES; stage++) {
	    expect_draft_stage(pack, layout, on, stage);
    }
    for (uint32_t index = 0; index < TEST_MODEL_V41_DRAFT_GLOBAL_TENSORS; index++) {
        const struct test_model_v41_tensor *const spec = &test_model_v41_draft_global_tensors[index];
        expect_place(pack, layout, &on->draft_globals[spec->place], spec->name, spec);
    }
    for (uint32_t array = 0; array < V41_DATA_ARRAY_COUNT; array++) {
	    const bool is_draft =
		    ((1u << array_sizes[array].holder) & draft_holders) != 0;
	    const uint64_t array_rows = array_sizes[array].rows ==
							V41_DRAFT_ROWS ?
						on->limits.draft_positions :
						rows;
	    const uint64_t bytes = array_sizes[array].fixed_bytes +
				   array_rows * array_sizes[array].row_bytes;
	    growth += is_draft ? aligned_bytes(bytes) : 0;
	    expect((result->array_offset[array] != PROGRAM_NOWHERE) ==
			   (is_draft ||
			    plain->array_offset[array] != PROGRAM_NOWHERE),
		   "the result card holds the draft's arrays and the ones it held");
    }
    expect(result->array_offset[V41_DATA_ARRAY_ROUND] ==
		   result->array_offset[V41_DATA_ARRAY_RESULT] +
			   V41_DATA_ARRAY_ALIGNMENT_BYTES,
	   "the round lies right behind the result area, one download for both");
    expect(result->arena_byte_count - result->state_offset == plain->arena_byte_count - plain->state_offset + growth,
           "the draft adds its arrays, rings and tables to the result card's state, and nothing else");
    expect_card_regions(on, RESULT_CARD, CONTEXT, 0);
}

/* The session ranges of the thin programs of one row of both plans: those of the plan without the draft, then the
 * three stages' rings on the result card, identities 160, 164 and 168. */
enum { PLAN_OFF, PLAN_ON, PLAN_PAIR };
static void expect_draft_ranges(const struct model_plan_record *off,
				const struct model_plan_record *on)
{
	const struct model_operations *const model =
		model_operations_of(V41_DATA_MODEL_NAME);
	const struct model_plan_record *const plans[PLAN_PAIR] = { off, on };
	struct execution_program *programs[PLAN_PAIR] = { NULL, NULL };
	struct device_card *cards[CARDS] = { NULL };
	struct card_description_record descriptions[CARDS];
	test_host_memory_backend_reset(CARDS, HOST_CARD_BYTES);
	for (uint32_t card = 0; card < CARDS; card++) {
		test_host_memory_backend_set_arithmetic_class(
			card, card == RESULT_CARD ? ARITHMETIC_CLASS_SM_120 :
						    ARITHMETIC_CLASS_SM_80);
		expect(test_host_memory_backend.card_describe(
			       card, &descriptions[card]) == STATUS_OK &&
			       test_host_memory_backend.card_open(
				       card, HOST_CARD_BYTES, &cards[card]) ==
				       STATUS_OK,
		       "a card of the backend in host memory opens");
	}
    for (uint32_t index = 0; model != NULL && index < PLAN_PAIR; index++) {
	    expect_status(model->program_prepare(
				  plans[index], &test_host_memory_backend,
				  cards, descriptions, PROGRAM_THIN, 1,
				  &programs[index]),
			  STATUS_OK, "the thin program of one row is made");
    }
    const struct execution_program *const plain = programs[PLAN_OFF],
					  *const drafted = programs[PLAN_ON];
    bool is_kept = plain != NULL && drafted != NULL &&
		   drafted->session_range_count ==
			   plain->session_range_count + MODEL_DRAFT_STAGES;
    for (uint32_t index = 0; is_kept && index < plain->session_range_count; index++) {
	    const struct session_range *const was =
		    &plain->session_ranges[index];
	    const struct session_range *const is =
		    &drafted->session_ranges[index];
	    is_kept = was->identity == is->identity && was->card == is->card &&
		      was->unit_count == is->unit_count &&
		      was->unit_bytes == is->unit_bytes &&
		      was->unit_tokens == is->unit_tokens &&
		      was->session_stride == is->session_stride &&
		      (was->card == RESULT_CARD || was->offset == is->offset);
    }
    for (uint32_t stage = 0; is_kept && stage < MODEL_DRAFT_STAGES; stage++) {
	    const uint32_t layer = MODEL_DRAFT_FIRST_LAYER + stage;
	    const struct session_range expected = {
		    on->cards[RESULT_CARD].ring_offset[layer],
		    V41_DATA_RING_SESSION_BYTES,
		    MODEL_RING_SLOTS,
		    MODEL_STATE_ROW_BYTES,
		    1u,
		    RESULT_CARD,
		    SESSION_KINDS * layer
	    };
	    is_kept =
		    memcmp(&drafted->session_ranges[plain->session_range_count +
						    stage],
			   &expected, sizeof(expected)) == 0;
    }
    expect(is_kept, "the draft adds exactly the three stages' rings to what a session keeps (160, 164, 168)");
    for (uint32_t index = 0; model != NULL && index < PLAN_PAIR; index++) {
        model->program_destroy(programs[index]);
    }
    for (uint32_t card = 0; card < CARDS; card++) {
        test_host_memory_backend.card_close(cards[card], 0);
    }
    test_host_memory_backend_reset(0, 0);
}

static void test_draft_plans(void)
{
    static uint32_t card_of_unit[DRAFT_UNIT_COUNT];
    static const uint32_t row_capacities[] = {ROWS, WIDE_ROWS};
    struct pack_record *pack = NULL;
    context_text = "the draft on the result card";
    expect_status(test_model_v41_draft_pack_open(
			  (struct test_model_v41_change){
				  TEST_MODEL_V41_CHANGE_NONE, NULL, 0 },
			  &pack),
		  STATUS_OK, "the pack opens from its tables");
    test_model_v41_five_cards(card_of_unit, UNIT_COUNT, 0);
    test_model_v41_draft_place(card_of_unit, RESULT_CARD);
    const struct placement_record placement = { card_of_unit, DRAFT_UNIT_COUNT,
						CARDS };
    for (size_t index = 0; pack != NULL && index < sizeof(row_capacities) / sizeof(row_capacities[0]); index++) {
	    for (uint32_t block = MODEL_DRAFT_BLOCK; block <= STEP_DRAFT_MOST;
		 block++) {
		    const struct model_limits off_limits =
			    draft_limits(row_capacities[index], 0);
		    const struct model_limits on_limits =
			    draft_limits(row_capacities[index], block);
		    struct load_layout *off_layout = NULL, *layout = NULL;
		    result_status off_status = STATUS_OK, status = STATUS_OK;
		    struct model_plan_record *const off =
			    plan_of(pack, card_of_unit, CARDS, &off_limits,
				    &off_status, &off_layout);
		    struct model_plan_record *const on =
			    plan_of(pack, card_of_unit, CARDS, &on_limits,
				    &status, &layout);
		    expect(off_status == STATUS_OK && status == STATUS_OK,
			   "the plans with and without the "
			   "draft are made");
		    if (off != NULL && on != NULL) {
			    expect_draft_plan(pack, layout, off, on);
			    for (uint32_t card = 0; card < CARDS; card++) {
				    uint64_t state = 0;
				    expect(v41_state_byte_count(
						   pack, &placement, &on_limits,
						   card, &state) == STATUS_OK &&
						   state ==
							   on->cards[card].arena_byte_count -
								   on->cards[card]
									   .state_offset,
					   "state_byte_count counts the draft's state before anything is loaded");
			    }
		    }
		    if (off != NULL && on != NULL && index == 0) {
			    expect_draft_ranges(off, on);
		    }
		    v41_plan_destroy(off);
		    v41_plan_destroy(on);
		    load_layout_destroy(off_layout);
		    load_layout_destroy(layout);
	    }
    }
    pack_close(pack);
}

/* A draft unit range placed elsewhere, and the words of the refusal's line. */
struct moved_units {
    uint32_t first_unit, unit_count, card;
    const char *words;
};

static void test_draft_refusals(void)
{
    static uint32_t card_of_unit[DRAFT_UNIT_COUNT];
    static const struct refused_pack cases[] = {
	    { { REMOVE, "mtp.2.markov_head.head.weight", 0 },
	      STATUS_CORRUPT_PACK,
	      "mtp.2.markov_head.head.weight" },
	    { { REMOVE, "mtp.0.main_norm.weight", 0 },
	      STATUS_CORRUPT_PACK,
	      "mtp.0.main_norm.weight" },
	    { { REMOVE, "mtp.1.attn.wkv.weight", 0 },
	      STATUS_CORRUPT_PACK,
	      "mtp.1.attn.wkv.weight" },
	    { { REMOVE, "mtp.2.ffn.experts.127.w3.weight", 0 },
	      STATUS_CORRUPT_PACK,
	      "mtp.2.ffn.experts.127.w3.weight" },
	    { { OUTPUTS, "mtp.1.ffn.gate.weight", EXPERTS },
	      STATUS_CORRUPT_PACK,
	      "mtp.1.ffn.gate.weight" },
	    { { OUTPUTS, "mtp.0.ffn.gate.bias", EXPERTS },
	      STATUS_CORRUPT_PACK,
	      "mtp.0.ffn.gate.bias" },
	    { { FORMAT, "mtp.0.main_proj.weight", WEIGHT_FORMAT_NONE },
	      STATUS_CORRUPT_PACK,
	      "mtp.0.main_proj.weight" },
	    { { ELEMENT, "mtp.2.confidence_head.proj.weight",
		PACK_ELEMENT_FLOAT32 },
	      STATUS_CORRUPT_PACK,
	      "mtp.2.confidence_head.proj.weight" },
	    { { MOVE, "mtp.2.norm.weight", TEST_MODEL_V41_LAYER_UNIT(0) },
	      STATUS_UNSUPPORTED,
	      "mtp.2.norm.weight lies on card 0 and the head on card 4: place them together (--layout, or --card_order" },
	    { { MOVE, "mtp.1.hc_ffn_fn", TEST_MODEL_V41_LAYER_UNIT(0) },
	      STATUS_UNSUPPORTED,
	      "mtp.1.hc_ffn_fn lies on card 0" },
	    { { MOVE, "mtp.0.ffn.experts.3.w1.weight",
		TEST_MODEL_V41_LAYER_UNIT(0) },
	      STATUS_UNSUPPORTED,
	      "expert 3 of draft stage 0 is not on the draft's card" },
    };
    static const struct moved_units moves[] = {
	    { TEST_MODEL_V41_DRAFT_STAGE_UNIT(1),
	      TEST_MODEL_V41_DRAFT_UNITS_PER_STAGE, TAP_CARD,
	      "mtp.1.attn_norm.weight lies on card 3" },
	    { TEST_MODEL_V41_LAYER_UNIT(MODEL_DRAFT_FIRST_TAP_LAYER),
	      TEST_MODEL_V41_UNITS_PER_LAYER, TAP_CARD,
	      "layers.37.attn_norm.weight lies on card 3" },
	    { TEST_MODEL_V41_DRAFT_EXPERT_UNIT(1, DRAFT_HOST_EXPERT), 1,
	      PLACEMENT_IN_HOST_MEMORY,
	      "expert 5 of draft stage 1 is not on the draft's card" },
	    { UNIT_COUNT, DRAFT_UNIT_COUNT - UNIT_COUNT, 0,
	      "mtp.0.main_proj.weight lies on card 0" },
    };
    const struct model_limits off_limits = draft_limits(ROWS, 0);
    const struct model_limits on_limits = draft_limits(ROWS, MODEL_DRAFT_BLOCK);
    for (size_t index = 0; index < sizeof(cases) / sizeof(cases[0]); index++) {
	    struct pack_record *pack = NULL;
	    context_text = cases[index].words;
	    test_model_v41_five_cards(card_of_unit, UNIT_COUNT, 0);
	    test_model_v41_draft_place(card_of_unit, RESULT_CARD);
	    expect_status(test_model_v41_draft_pack_open(cases[index].change,
							 &pack),
			  STATUS_OK, "the changed pack opens from its tables");
	    if (pack != NULL) {
		    expect_refused_plan(
			    pack, card_of_unit, &on_limits,
			    cases[index].expected, cases[index].words,
			    "a draft the model cannot run is refused with one line");
		    struct load_layout *layout = NULL;
		    result_status status = STATUS_OK;
		    struct model_plan_record *const plan =
			    plan_of(pack, card_of_unit, CARDS, &off_limits,
				    &status, &layout);
		    expect(status == STATUS_OK,
			   "without the draft the same pack is planned: its draft is not read");
		    v41_plan_destroy(plan);
		    load_layout_destroy(layout);
	    }
	    pack_close(pack);
    }
    struct pack_record *pack = NULL;
    expect_status(test_model_v41_draft_pack_open(
			  (struct test_model_v41_change){
				  TEST_MODEL_V41_CHANGE_NONE, NULL, 0 },
			  &pack),
		  STATUS_OK, "the pack opens from its tables");
    for (size_t index = 0; pack != NULL && index < sizeof(moves) / sizeof(moves[0]); index++) {
        context_text = moves[index].words;
        test_model_v41_five_cards(card_of_unit, UNIT_COUNT, 0);
        test_model_v41_draft_place(card_of_unit, RESULT_CARD);
        for (uint32_t unit = 0; unit < moves[index].unit_count; unit++) {
            card_of_unit[moves[index].first_unit + unit] = moves[index].card;
        }
	expect_refused_plan(
		pack, card_of_unit, &on_limits, STATUS_UNSUPPORTED,
		moves[index].words,
		"a placement that cannot run the draft is refused with one line");
    }
    context_text = "draft limits";
    test_model_v41_five_cards(card_of_unit, UNIT_COUNT, 0);
    test_model_v41_draft_place(card_of_unit, RESULT_CARD);
    const struct placement_record placement = { card_of_unit, DRAFT_UNIT_COUNT,
						CARDS };
    const struct model_limits limits[] = {
	    draft_limits(ROWS, MODEL_DRAFT_BLOCK - 1u),
	    draft_limits(ROWS, STEP_DRAFT_MOST + 1u),
	    draft_limits(MODEL_DRAFT_BLOCK - 1u, MODEL_DRAFT_BLOCK)
    };
    for (size_t index = 0; pack != NULL && index < sizeof(limits) / sizeof(limits[0]); index++) {
        uint64_t state = UINT64_MAX;
	expect_refused_plan(
		pack, card_of_unit, &limits[index], STATUS_INVALID_ARGUMENT,
		"the model has none of",
		"draft positions other than its block's, or above a slab's rows");
	expect(v41_state_byte_count(pack, &placement, &limits[index], 0,
				    &state) == STATUS_INVALID_ARGUMENT &&
		       state == UINT64_MAX,
	       "state_byte_count refuses the same limits");
    }
    pack_close(pack);
    context_text = "the draft asked of a pack without a draft head";
    pack = NULL;
    expect_status(test_model_v41_pack_open(
			  (struct test_model_v41_change){
				  TEST_MODEL_V41_CHANGE_NONE, NULL, 0 },
			  &pack),
		  STATUS_OK, "the pack opens from its tables");
    if (pack != NULL) {
	    expect_refused_plan(pack, card_of_unit, &on_limits,
				STATUS_CORRUPT_PACK, "mtp.0.main_proj.weight",
				"a pack without the draft head cannot draft");
    }
    pack_close(pack);
}

/* Limits the model has none of, then limits beyond this build. */
#define INVALID_LIMIT_CASES 9u
static const struct model_limits limit_cases[] = {
	{ 0, ROWS, SESSIONS, STATE_WIDTH_16, LINEAR_PROFILE_PRECISION,
	  LINEAR_ORDER_FIXED, CMP_ORDER_REFERENCE, 0, 0, 0 },
	{ CONTEXT, MODEL_WIDE_MOST_ROWS + 1u, SESSIONS, STATE_WIDTH_16,
	  LINEAR_PROFILE_PRECISION, LINEAR_ORDER_FIXED, CMP_ORDER_REFERENCE, 0,
	  0, 0 },
	{ CONTEXT + 1u, ROWS, SESSIONS, STATE_WIDTH_16,
	  LINEAR_PROFILE_PRECISION, LINEAR_ORDER_FIXED, CMP_ORDER_REFERENCE, 0,
	  0, 0 },
	{ 2097152, ROWS, SESSIONS, STATE_WIDTH_16, LINEAR_PROFILE_PRECISION,
	  LINEAR_ORDER_FIXED, CMP_ORDER_REFERENCE, 0, 0, 0 },
	{ CONTEXT, 0, SESSIONS, STATE_WIDTH_16, LINEAR_PROFILE_PRECISION,
	  LINEAR_ORDER_FIXED, CMP_ORDER_REFERENCE, 0, 0, 0 },
	{ CONTEXT, ROWS, 0, STATE_WIDTH_16, LINEAR_PROFILE_PRECISION,
	  LINEAR_ORDER_FIXED, CMP_ORDER_REFERENCE, 0, 0, 0 },
	{ CONTEXT, ROWS, SESSIONS, (enum state_width_code)17,
	  LINEAR_PROFILE_PRECISION, LINEAR_ORDER_FIXED, CMP_ORDER_REFERENCE, 0,
	  0, 0 },
	{ CONTEXT, ROWS, SESSIONS, STATE_WIDTH_16, (enum linear_profile_code)2,
	  LINEAR_ORDER_FIXED, CMP_ORDER_REFERENCE, 0, 0, 0 },
	{ CONTEXT, ROWS, SESSIONS, STATE_WIDTH_16, LINEAR_PROFILE_PRECISION,
	  LINEAR_ORDER_FIXED, (enum cmp_order_code)2, 0, 0, 0 },
	{ CONTEXT, ROWS, SESSIONS, STATE_WIDTH_8, LINEAR_PROFILE_PRECISION,
	  LINEAR_ORDER_FIXED, CMP_ORDER_REFERENCE, 0, 0, 0 },
};

static void test_refusals(const struct pack_record *pack)
{
    static uint32_t card_of_unit[UNIT_COUNT];
    struct load_layout *layout = NULL;
    struct model_plan_record *plan = NULL;
    struct model_description_record description = { 0 };
    uint64_t byte_count = UINT64_MAX;
    context_text = "refusals";
    test_model_v41_five_cards(card_of_unit, UNIT_COUNT, 0);
    const struct placement_record placement = { card_of_unit, UNIT_COUNT,
						CARDS };
    for (size_t index = 0; index < sizeof(limit_cases) / sizeof(limit_cases[0]); index++) {
	    const struct model_limits *const limits = &limit_cases[index];
	    const bool is_valid = index >= INVALID_LIMIT_CASES;
	    expect_refused_plan(
		    pack, card_of_unit, limits,
		    is_valid ? STATUS_UNSUPPORTED : STATUS_INVALID_ARGUMENT,
		    is_valid ? "beyond this build" : "the model has none of",
		    "limits are refused");
	    expect_status(v41_state_byte_count(pack, &placement, limits, 0,
					       &byte_count),
			  is_valid ? STATUS_UNSUPPORTED :
				     STATUS_INVALID_ARGUMENT,
			  "state_byte_count refuses the same limits");
    }
    const struct placement_record short_placement = { card_of_unit,
						      UNIT_COUNT - 1u, CARDS };
    expect_status(v41_state_byte_count(pack, &placement, &limits_of_m1, CARDS,
				       &byte_count),
		  STATUS_INVALID_ARGUMENT, "a card beyond the placement");
    expect_status(v41_state_byte_count(pack, &short_placement, &limits_of_m1, 0,
				       &byte_count),
		  STATUS_INVALID_ARGUMENT, "a placement of another pack");
    expect_status(v41_state_byte_count(NULL, &placement, &limits_of_m1, 0,
				       &byte_count),
		  STATUS_INVALID_ARGUMENT, "no pack");
    expect_status(v41_state_byte_count(pack, NULL, &limits_of_m1, 0,
				       &byte_count),
		  STATUS_INVALID_ARGUMENT, "no placement");
    expect_status(v41_state_byte_count(pack, &placement, NULL, 0, &byte_count),
		  STATUS_INVALID_ARGUMENT, "no limits");
    expect_status(v41_state_byte_count(pack, &placement, &limits_of_m1, 0,
				       NULL),
		  STATUS_INVALID_ARGUMENT, "no output");
    expect(byte_count == UINT64_MAX, "a refused state_byte_count leaves its output untouched");
    expect_status(v41_describe(NULL, &description), STATUS_INVALID_ARGUMENT,
		  "describe: no pack");
    expect_status(v41_describe(pack, NULL), STATUS_INVALID_ARGUMENT,
		  "describe: no output");
    expect_status(load_layout_create(pack, card_of_unit, UNIT_COUNT, CARDS,
				     &layout),
		  STATUS_OK, "the five cards are laid out");
    if (layout != NULL) {
	    expect_status(v41_plan_create(NULL, layout, &limits_of_m1, &plan),
			  STATUS_INVALID_ARGUMENT, "plan_create: no pack");
	    expect_status(v41_plan_create(pack, NULL, &limits_of_m1, &plan),
			  STATUS_INVALID_ARGUMENT, "plan_create: no layout");
	    expect_status(v41_plan_create(pack, layout, NULL, &plan),
			  STATUS_INVALID_ARGUMENT, "plan_create: no limits");
	    expect_status(v41_plan_create(pack, layout, &limits_of_m1, NULL),
			  STATUS_INVALID_ARGUMENT, "plan_create: no output");
	    layout->extent_count--;
	    expect_status(v41_plan_create(pack, layout, &limits_of_m1, &plan),
			  STATUS_INVALID_ARGUMENT, "a layout of another pack");
	    layout->extent_count++;
	    layout->card_byte_count[1] -= (uint64_t)SHORTENED_BLOCKS * BLOCK;
	    expect_status(v41_plan_create(pack, layout, &limits_of_m1, &plan),
			  STATUS_INVALID_ARGUMENT,
			  "a layout whose weights do not hold its tensors");
	    layout->card_byte_count[1] +=
		    (uint64_t)SHORTENED_BLOCKS * BLOCK + MISALIGNED_BYTES;
	    capture_begin();
	    expect_status(v41_plan_create(pack, layout, &limits_of_m1, &plan),
			  STATUS_INVALID_ARGUMENT,
			  "weights that end off an array's boundary");
	    capture_end_expecting(
		    "not on a multiple of",
		    "the card whose weights end off the boundary is named");
	    layout->card_byte_count[1] -= MISALIGNED_BYTES;
#if CHECKING_HOOKS_ENABLED
	    expect(fault_arm("model_v41_plan_memory", 1),
		   "the fault of the plan's memory is armed");
	    expect_status(v41_plan_create(pack, layout, &limits_of_m1, &plan),
			  STATUS_OUT_OF_MEMORY, "no memory for the plan");
	    expect(fault_disarm() == 1,
		   "the fault point of the plan's memory was reached");
	    expect(fault_arm("model_v41_image_memory", CARDS),
		   "the fault of an image's memory is armed");
	    expect_status(v41_plan_create(pack, layout, &limits_of_m1, &plan),
			  STATUS_OUT_OF_MEMORY, "no memory for the last image");
	    expect(fault_disarm() == CARDS,
		   "the fault point of the images was reached for each card");
#endif
        expect(plan == NULL, "a refused plan_create leaves its output untouched");
    }
    expect_status(v41_arena_byte_count(NULL, 0, &byte_count),
		  STATUS_INVALID_ARGUMENT, "arena_byte_count: no plan");
    load_layout_destroy(layout);
    test_memory_storage_reset();
}

/* ---- arena_initialise on the backend in host memory ---- */

static void *kept_staging = NULL;               /* the block arena_initialise last obtained */

MUST_CHECK static result_status watched_staging_allocate(uint64_t byte_count,
							 void **memory_out)
{
	const result_status status = test_host_memory_backend.staging_allocate(
		byte_count, memory_out);
	kept_staging = status == STATUS_OK ? *memory_out : kept_staging;
	return status;
}

static struct backend_operations watched_backend;

/* What the fill's stand-in was asked: the launches it ran, and a word of its range it leaves as it was (0 for none),
 * as card memory that did not keep a write. */
static struct fill_calls {
    uint32_t count;
    uint64_t skipped_word;
} fill_calls;

/* The fill kernel in host memory: a launch outside the arena is a fault, as backend_records.h has its preparer check
 * before it. */
static bool fill_stand_in(const struct kernel_launch *launch,
			  uint32_t card_index, unsigned char *arena,
			  uint64_t arena_byte_count, void *context)
{
    struct fill_calls *const calls = context;
    struct fill_record record;
    memcpy(&record, launch->argument, sizeof(record));
    (void)card_index;
    calls->count++;
    if (record.arena_byte_offset % FILL_WORD_BYTES != 0 ||
	record.arena_byte_offset > arena_byte_count ||
	record.word_count > (arena_byte_count - record.arena_byte_offset) /
				    FILL_WORD_BYTES) {
	    return false;
    }
    for (uint64_t word = 0; word < record.word_count; word++) {
        if (calls->skipped_word == 0 || word != calls->skipped_word) {
		memcpy(arena + record.arena_byte_offset +
			       word * FILL_WORD_BYTES,
		       &record.value, sizeof(record.value));
	}
    }
    return true;
}

/* A plan of three cards: card 0 with SMALL_STATE_BYTES of state, card 1 with an image and the zeros of exactly one
 * block of a fill, card 2 with no state. */
static void small_plan(struct model_plan_record *plan, unsigned char *image)
{
    memset(plan, 0, sizeof(*plan));
    plan->card_count = SMALL_CARDS;
    plan->cards[0] =
	    (struct v41_card){ .state_offset = SMALL_WEIGHTS_BYTES,
			       .arena_byte_count =
				       SMALL_WEIGHTS_BYTES + SMALL_STATE_BYTES,
			       .image_byte_count = SMALL_IMAGE_BYTES,
			       .image = image };
    plan->cards[1] =
	    (struct v41_card){ .state_offset = SMALL_WEIGHTS_BYTES,
			       .arena_byte_count = SMALL_WEIGHTS_BYTES +
						   TINY_IMAGE_BYTES +
						   TINY_ZERO_BYTES,
			       .image_byte_count = TINY_IMAGE_BYTES,
			       .image = image };
    plan->cards[SMALL_CARDS - 1u] =
	    (struct v41_card){ .state_offset = BLOCK,
			       .arena_byte_count = BLOCK };
}

/* The cards of the plan with state, of an arithmetic class (0: no build of any kernel) with the fill's stand-in,
 * card 0's arena short_bytes shorter than the plan's; their states hold garbage, as a card's memory before it is
 * set. */
static void cards_open(const struct model_plan_record *plan,
		       struct device_card **cards, uint64_t short_bytes,
		       uint32_t arithmetic_class)
{
    void *garbage = NULL;
    uint64_t ticket = 0;
    test_host_memory_backend_reset(SMALL_CARDS - 1u, SMALL_WEIGHTS_BYTES + SMALL_STATE_BYTES);
    fill_calls = (struct fill_calls){0, 0};
    expect_status(test_host_memory_backend_set_stand_in(
			  KERNEL_FILE_FILL, KERNEL_FILL_WORDS, fill_stand_in,
			  &fill_calls),
		  STATUS_OK, "the fill has its stand-in");
    expect_status(test_host_memory_backend.staging_allocate(SMALL_STATE_BYTES,
							    &garbage),
		  STATUS_OK, "staging for the garbage");
    for (uint32_t card = 0; garbage != NULL && card + 1u < SMALL_CARDS; card++) {
        const uint64_t arena_bytes = plan->cards[card].arena_byte_count - (card == 0 ? short_bytes : 0);
        test_host_memory_backend_set_arithmetic_class(card, arithmetic_class);
        memset(garbage, GARBAGE_BYTE, (size_t)SMALL_STATE_BYTES);
	expect(test_host_memory_backend.card_open(card, arena_bytes,
						  &cards[card]) == STATUS_OK &&
		       test_host_memory_backend.upload_begin(
			       cards[card], SMALL_WEIGHTS_BYTES, garbage,
			       arena_bytes - SMALL_WEIGHTS_BYTES,
			       &ticket) == STATUS_OK &&
		       test_host_memory_backend.upload_wait(
			       cards[card], ticket, WAIT_LIMIT_NANOSECONDS) ==
			       STATUS_OK,
	       "a card opens and its state holds garbage");
    }
    test_host_memory_backend.staging_free(garbage);
}

/* Closes the cards, the staging memory where the contract leaves it: kept while a card may still read it. */
static void cards_close(struct device_card **cards, bool staging_is_kept)
{
    expect(test_host_memory_backend_staging_block_count() == (staging_is_kept ? 1u : 0u),
           "the staging memory is kept only while a card may read it");
    for (uint32_t card = 0; card < SMALL_CARDS; card++) {
        test_host_memory_backend.card_close(cards[card], CLOSE_LIMIT_NANOSECONDS);
        cards[card] = NULL;
    }
    if (staging_is_kept) {
        test_host_memory_backend.staging_free(kept_staging);
    }
    kept_staging = NULL;
}

/* Cards 0 and 1 hold their image, then zeros, over the garbage. */
static void expect_state(const struct model_plan_record *plan,
			 struct device_card *const *cards)
{
    for (uint32_t card = 0; card + 1u < SMALL_CARDS; card++) {
	    const struct v41_card *const planned = &plan->cards[card];
	    const unsigned char *const arena =
		    test_host_memory_backend_arena_of(cards[card]);
	    bool is_right = arena != NULL &&
			    memcmp(arena + planned->state_offset,
				   planned->image,
				   (size_t)planned->image_byte_count) == 0;
	    for (uint64_t index =
			 planned->state_offset + planned->image_byte_count;
		 is_right && index < planned->arena_byte_count; index++) {
		    is_right = arena[index] == 0;
	    }
        expect(is_right, "a card's state is its image, then zeros, over the garbage");
    }
}

/* The work of arena_initialise on cards 0 and 1, in order: the digest of the plan's last word, one fill of the zeros
 * shaped as backend_records.h says, the image's upload, the digest of the whole state. */
enum { WORK_LAST_WORD, WORK_FILL, WORK_IMAGE, WORK_STATE, WORK_PER_CARD };
enum { DIMENSION_X, DIMENSION_Y, DIMENSION_Z };
static void expect_work(const struct model_plan_record *plan,
			uint64_t work_index)
{
	static const uint32_t grids[SMALL_CARDS - 1u] = { FILL_MOST_BLOCKS,
							  1u };
	for (uint32_t card = 0; card + 1u < SMALL_CARDS; card++) {
		const struct v41_card *const planned = &plan->cards[card];
		const uint64_t state_bytes =
			planned->arena_byte_count - planned->state_offset;
		const uint64_t last_word =
			planned->arena_byte_count - PACK_DIGEST_WORD_BYTES;
		struct test_host_memory_backend_work work[WORK_PER_CARD];
		struct fill_record record;
		bool is_right = true;
		memset(work, 0, sizeof(work));
		for (uint32_t index = 0; index < WORK_PER_CARD; index++) {
			is_right = is_right &&
				   test_host_memory_backend_work_at(
					   work_index++, &work[index]) &&
				   work[index].card_index == card;
		}
		const struct kernel_launch *const fill =
			&work[WORK_FILL].launch;
		memcpy(&record, fill->argument, sizeof(record));
		expect(is_right &&
			       work[WORK_LAST_WORD].kind ==
				       TEST_HOST_MEMORY_BACKEND_DIGEST &&
			       work[WORK_LAST_WORD].arena_byte_offset ==
				       last_word &&
			       work[WORK_LAST_WORD].byte_count ==
				       PACK_DIGEST_WORD_BYTES &&
			       work[WORK_IMAGE].kind ==
				       TEST_HOST_MEMORY_BACKEND_UPLOAD &&
			       work[WORK_IMAGE].arena_byte_offset ==
				       planned->state_offset &&
			       work[WORK_IMAGE].byte_count ==
				       planned->image_byte_count &&
			       work[WORK_STATE].kind ==
				       TEST_HOST_MEMORY_BACKEND_DIGEST &&
			       work[WORK_STATE].arena_byte_offset ==
				       planned->state_offset &&
			       work[WORK_STATE].byte_count == state_bytes,
		       "the last word's digest, the image's upload and the state's digest, in their order");
		expect(work[WORK_FILL].kind ==
				       TEST_HOST_MEMORY_BACKEND_LAUNCH &&
			       fill->grid[DIMENSION_X] == grids[card] &&
			       fill->grid[DIMENSION_Y] == 1u &&
			       fill->grid[DIMENSION_Z] == 1u &&
			       fill->block[DIMENSION_X] ==
				       FILL_THREADS_PER_BLOCK_VALUE &&
			       fill->block[DIMENSION_Y] == 1u &&
			       fill->block[DIMENSION_Z] == 1u &&
			       fill->dynamic_shared_byte_count == 0 &&
			       fill->argument_byte_count == FILL_RECORD_BYTES &&
			       record.arena_byte_offset ==
				       planned->state_offset +
					       planned->image_byte_count &&
			       record.word_count ==
				       (state_bytes -
					planned->image_byte_count) /
					       FILL_WORD_BYTES &&
			       record.value == 0 && record.reserved_zero == 0,
		       "one fill of the zeros, a block for every 256 words up to the most a fill takes");
	}
    expect(test_host_memory_backend_work_count() == work_index && fill_calls.count == SMALL_CARDS - 1u,
           "no other work, and the fill ran once a card with state");
}

/* A fault of card 0 or its link, armed after the cards opened (a field of 0 arms nothing; polls go with a wait limit
 * of 0), what arena_initialise must answer, whether the staging stays (an upload began that was not seen to finish),
 * and the words of its one error line. A card refused before its work: its arena short of the plan's by short_bytes,
 * or no build of the fill. */
struct fault_case {
    const char *what;
    result_status expected;
    bool staging_is_kept;
    const char *words;
    uint64_t staging_number, flipped_offset, skipped_word, upload_ticket, digest_number, launch_number, short_bytes;
    uint32_t polls;
    bool has_no_build;
};
#define DIGEST_DIFFERS "the card's digest of it differs from the host's"
static const struct fault_case fault_cases[] = {
	{ .what = "no staging memory",
	  .expected = STATUS_OUT_OF_MEMORY,
	  .staging_number = 1 },
	{ .what = "a byte the link changed",
	  .expected = STATUS_DEVICE_ERROR,
	  .words = DIGEST_DIFFERS,
	  .flipped_offset = FLIPPED_BYTE_OFFSET },
	{ .what = "a zero the card did not keep",
	  .expected = STATUS_DEVICE_ERROR,
	  .words = DIGEST_DIFFERS,
	  .skipped_word = SKIPPED_WORD },
	{ .what = "an upload not seen to finish",
	  .expected = STATUS_TIMED_OUT,
	  .staging_is_kept = true,
	  .polls = POLLS_BEFORE_FINISHING },
	{ .what = "a failed upload",
	  .expected = STATUS_DEVICE_ERROR,
	  .staging_is_kept = true,
	  .upload_ticket = FAILING_TICKET },
	{ .what = "a failed digest of the last word",
	  .expected = STATUS_DEVICE_ERROR,
	  .digest_number = 1 },
	{ .what = "a failed digest of the state",
	  .expected = STATUS_DEVICE_ERROR,
	  .digest_number = FAILING_DIGEST },
	{ .what = "a fill that faults",
	  .expected = STATUS_DEVICE_ERROR,
	  .launch_number = 1 },
	{ .what = "an arena shorter than the plan's",
	  .expected = STATUS_INVALID_ARGUMENT,
	  .words = "arena bytes",
	  .short_bytes = SHORT_ARENA_BYTES },
	{ .what = "a card with no build of the fill",
	  .expected = STATUS_UNSUPPORTED,
	  .has_no_build = true },
};

static void test_fault(struct model_plan_record *plan,
		       struct device_card **cards,
		       const struct fault_case *fault)
{
    context_text = fault->what;
    cards_open(plan, cards, fault->short_bytes,
	       fault->has_no_build ? 0 : ARITHMETIC_CLASS_SM_80);
    const uint64_t work_before = test_host_memory_backend_work_count();
    fill_calls.skipped_word = fault->skipped_word;
    if (fault->staging_number != 0) { test_host_memory_backend_fail_staging_allocation(fault->staging_number); }
    if (fault->flipped_offset != 0) { test_host_memory_backend_flip_bit_in_next_upload(0, fault->flipped_offset); }
    if (fault->polls != 0) { test_host_memory_backend_set_polls_before_an_upload_finishes(0, fault->polls); }
    if (fault->upload_ticket != 0) { test_host_memory_backend_fail_upload(0, fault->upload_ticket); }
    if (fault->digest_number != 0) { test_host_memory_backend_fail_digest(0, fault->digest_number); }
    if (fault->launch_number != 0) { test_host_memory_backend_fail_launch(0, fault->launch_number); }
    if (fault->words != NULL) { capture_begin(); }
    const result_status status = v41_arena_initialise(
	    plan, &watched_backend, cards,
	    fault->polls != 0 ? 0 : WAIT_LIMIT_NANOSECONDS);
    if (fault->words != NULL) { capture_end_expecting(fault->words, "the one error line says why"); }
    expect_status(status, fault->expected, "arena_initialise answers the fault");
    expect((fault->short_bytes == 0 && !fault->has_no_build) ||
               (test_host_memory_backend_work_count() == work_before && fill_calls.count == 0),
           "a card refused before its work is given none");
    cards_close(cards, fault->staging_is_kept);
}

static void test_arena_initialise(void)
{
    static unsigned char image[SMALL_IMAGE_BYTES];
    struct model_plan_record *const plan = calloc(1, sizeof(*plan));
    struct device_card *cards[SMALL_CARDS] = { NULL, NULL, NULL };
    context_text = "arena_initialise";
    watched_backend = test_host_memory_backend;
    watched_backend.staging_allocate = watched_staging_allocate;
    for (uint64_t index = 0; index < SMALL_IMAGE_BYTES; index++) {
        image[index] = (unsigned char)((index * IMAGE_SEED) >> IMAGE_SHIFT);
    }
    if (plan == NULL) {
        expect(false, "memory for a plan");
        return;
    }
    small_plan(plan, image);
    cards_open(plan, cards, 0, ARITHMETIC_CLASS_SM_80);
    const uint64_t work_before = test_host_memory_backend_work_count();
    expect_status(v41_arena_initialise(plan, &watched_backend, cards,
				       WAIT_LIMIT_NANOSECONDS),
		  STATUS_OK, "the states are written");
    expect_state(plan, cards);
    expect_work(plan, work_before);
    /* refusals: nothing obtained, nothing begun */
    const uint64_t work_after = test_host_memory_backend_work_count();
    struct device_card *const no_cards[SMALL_CARDS] = { NULL, NULL, NULL };
    struct backend_operations without_digest = watched_backend;
    without_digest.digest = NULL;
    expect_status(v41_arena_initialise(plan, &watched_backend, no_cards,
				       WAIT_LIMIT_NANOSECONDS),
		  STATUS_INVALID_ARGUMENT, "a card with state must be given");
    expect_status(v41_arena_initialise(NULL, &watched_backend, cards,
				       WAIT_LIMIT_NANOSECONDS),
		  STATUS_INVALID_ARGUMENT, "no plan");
    expect_status(v41_arena_initialise(plan, &without_digest, cards,
				       WAIT_LIMIT_NANOSECONDS),
		  STATUS_INVALID_ARGUMENT, "a backend without a digest");
    plan->card_count = BACKEND_MAXIMUM_CARDS + 1u;
    expect_status(v41_arena_initialise(plan, &watched_backend, cards,
				       WAIT_LIMIT_NANOSECONDS),
		  STATUS_INVALID_ARGUMENT,
		  "a plan of more cards than there can be");
    plan->card_count = SMALL_CARDS;
    plan->cards[0].image_byte_count = SMALL_STATE_BYTES + BLOCK;
    expect_status(v41_arena_initialise(plan, &watched_backend, cards,
				       WAIT_LIMIT_NANOSECONDS),
		  STATUS_INVALID_ARGUMENT, "an image longer than the state");
    plan->cards[0].image_byte_count = SMALL_IMAGE_BYTES + OFF_WORD_BYTES;
    expect_status(v41_arena_initialise(plan, &watched_backend, cards,
				       WAIT_LIMIT_NANOSECONDS),
		  STATUS_INVALID_ARGUMENT,
		  "an image that is no whole number of words");
    plan->cards[0].image_byte_count = SMALL_IMAGE_BYTES;
    expect(test_host_memory_backend_work_count() == work_after, "a refusal begins no work");
    cards_close(cards, false);
    for (size_t index = 0; index < sizeof(fault_cases) / sizeof(fault_cases[0]); index++) {
        test_fault(plan, cards, &fault_cases[index]);
    }
    context_text = "arena_initialise";
    expect(test_host_memory_backend_open_card_count() == 0 && test_host_memory_backend_staging_block_count() == 0,
           "the case ends with no card open and no staging alive");
    test_host_memory_backend_reset(0, 0);
    free(plan);
}

/* Two synthetic trunk matrices live in the small plan's weights. Only their scale tails are read at load. */
static void fold_scales_put(struct device_card *card, uint64_t matrix_offset,
			    uint8_t smallest)
{
    void *staging = NULL;
    uint64_t ticket = 0;
    result_status status = test_host_memory_backend.staging_allocate(
	    FOLD_SCALE_BYTES, &staging);
    if (status == STATUS_OK) {
	    memset(staging, FOLD_CODE, FOLD_SCALE_BYTES);
	    ((uint8_t *)staging)[0] = smallest;
	    status = test_host_memory_backend.upload_begin(
		    card, matrix_offset + FOLD_WEIGHT_BYTES, staging,
		    FOLD_SCALE_BYTES, &ticket);
    }
    status = status == STATUS_OK ?
		     test_host_memory_backend.upload_wait(
			     card, ticket, WAIT_LIMIT_NANOSECONDS) :
		     status;
    expect_status(status, STATUS_OK,
		  "the synthetic matrix's scales are uploaded");
    if (staging != NULL) {
        test_host_memory_backend.staging_free(staging);
    }
}

MUST_CHECK static result_status failed_fold_download(struct device_card *card,
						     uint64_t offset,
						     void *destination,
						     uint64_t byte_count,
						     uint64_t *ticket_out)
{
    (void)card;
    (void)offset;
    (void)destination;
    (void)byte_count;
    (void)ticket_out;
    return STATUS_DEVICE_ERROR;
}

/* Begin succeeds, but its ticket never reports success: load must retain the destination until card close. */
MUST_CHECK static result_status
failed_fold_download_wait(struct device_card *card, uint64_t offset,
			  void *destination, uint64_t byte_count,
			  uint64_t *ticket_out)
{
	const result_status status = test_host_memory_backend.download_begin(
		card, offset, destination, byte_count, ticket_out);
	if (status == STATUS_OK) {
		test_host_memory_backend_fail_upload(0, *ticket_out);
	}
    return status;
}

static void test_fold_load(void)
{
    static unsigned char image[SMALL_IMAGE_BYTES];
    struct model_plan_record *const plan = calloc(1, sizeof(*plan));
    struct device_card *cards[SMALL_CARDS] = { NULL, NULL, NULL };
    context_text = "the load's CMP scale folds";
    if (plan == NULL) {
        expect(false, "memory for the fold plan");
        return;
    }
    watched_backend = test_host_memory_backend;
    watched_backend.staging_allocate = watched_staging_allocate;
    small_plan(plan, image);
    plan->limits.cmp_order = CMP_ORDER_RELAXED;
    const struct v41_tensor quantized = {
	    .stored = { 0, FOLD_WEIGHT_BYTES + FOLD_SCALE_BYTES },
	    .shape = { FOLD_OUTPUTS, FOLD_INPUTS, FOLD_BITS_PER_WEIGHT, 0 },
	    .weight_format = WEIGHT_FORMAT_E4M3_SCALED,
    };
    plan->layer_tensors[0][V41_DATA_QUERY_DOWN] = quantized;
    plan->layer_tensors[0][V41_DATA_OUTPUT_GROUPS] = quantized;
    plan->layer_tensors[0][V41_DATA_OUTPUT_GROUPS].stored.byte_offset =
	    FOLD_GROUPS_OFFSET;
    cards_open(plan, cards, 0, ARITHMETIC_CLASS_SM_80);
    fold_scales_put(cards[0], 0, RELAXED_SCALE_SMALLEST);
    fold_scales_put(cards[0], FOLD_GROUPS_OFFSET,
		    FOLD_CODE - RELAXED_HALF_SCALE_SPREAD);
    expect_status(v41_arena_initialise(plan, &watched_backend, cards,
				       WAIT_LIMIT_NANOSECONDS),
		  STATUS_OK, "valid scales admit both row forms");
    expect(plan->layer_tensors[0][V41_DATA_QUERY_DOWN].fold_reference ==
			   RELAXED_FOLD_REFERENCE &&
		   plan->layer_tensors[0][V41_DATA_OUTPUT_GROUPS]
				   .fold_reference == FOLD_GROUPED_REFERENCE,
	   "load pins each row form's checked fold reference");
    fold_scales_put(cards[0], 0, RELAXED_SCALE_SMALLEST - 1u);
    fold_scales_put(cards[0], FOLD_GROUPS_OFFSET,
		    FOLD_CODE - RELAXED_HALF_SCALE_SPREAD - 1u);
    expect_status(v41_arena_initialise(plan, &watched_backend, cards,
				       WAIT_LIMIT_NANOSECONDS),
		  STATUS_OK,
		  "out-of-range scales fall back instead of failing load");
    expect(plan->layer_tensors[0][V41_DATA_QUERY_DOWN].fold_reference == 0 &&
		   plan->layer_tensors[0][V41_DATA_OUTPUT_GROUPS]
				   .fold_reference == 0,
	   "the quantized bound and grouped fp16 spread each force reference order");
    fold_scales_put(cards[0], 0, RELAXED_SCALE_SMALLEST);
    struct backend_operations failed_backend = watched_backend;
    failed_backend.download_begin = NULL;
    const uint64_t work_before = test_host_memory_backend_work_count();
    expect_status(v41_arena_initialise(plan, &failed_backend, cards,
				       WAIT_LIMIT_NANOSECONDS),
		  STATUS_INVALID_ARGUMENT,
		  "relaxed load requires a scale download function");
    expect(test_host_memory_backend_work_count() == work_before,
           "a missing scale download function is refused before state or weight work");
    failed_backend.download_begin = failed_fold_download;
    expect_status(v41_arena_initialise(plan, &failed_backend, cards,
				       WAIT_LIMIT_NANOSECONDS),
		  STATUS_DEVICE_ERROR,
		  "a scale download failure still fails load");
    failed_backend.download_begin = failed_fold_download_wait;
    expect_status(v41_arena_initialise(plan, &failed_backend, cards,
				       WAIT_LIMIT_NANOSECONDS),
		  STATUS_DEVICE_ERROR,
		  "a failed scale download ticket still fails load");
    cards_close(cards, true);
    expect(test_host_memory_backend_open_card_count() == 0 && test_host_memory_backend_staging_block_count() == 0,
           "fold admission and failed download retain no card or staging block");
    test_host_memory_backend_reset(0, 0);
    free(plan);
}

/* ---- the frequency words ---- */

/* A plan of two cards with the draft: card 0 the embedding card, its embed.weight COPY_BYTES at COPY_SOURCE_OFFSET of
 * its weights; card 1 the result card, its state the tiny image, a block of a fill's zeros and DRAFT_EMBEDDING. */
static void copy_plan(struct model_plan_record *plan, unsigned char *image)
{
    memset(plan, 0, sizeof(*plan));
    plan->card_count = COPY_CARDS;
    plan->limits.draft_positions = MODEL_DRAFT_BLOCK;
    plan->result_card = COPY_CARDS - 1u;
    plan->globals[V41_DATA_EMBEDDING].stored =
	    (struct linear_range){ COPY_SOURCE_OFFSET, COPY_BYTES };
    plan->cards[0] = (struct v41_card){ .state_offset = COPY_WEIGHTS_BYTES,
					.arena_byte_count = COPY_WEIGHTS_BYTES +
							    TINY_IMAGE_BYTES +
							    TINY_ZERO_BYTES,
					.image_byte_count = TINY_IMAGE_BYTES,
					.image = image };
    plan->cards[1] =
	    (struct v41_card){ .state_offset = SMALL_WEIGHTS_BYTES,
			       .arena_byte_count = SMALL_WEIGHTS_BYTES +
						   TINY_IMAGE_BYTES +
						   TINY_ZERO_BYTES + COPY_BYTES,
			       .image_byte_count = TINY_IMAGE_BYTES,
			       .image = image };
    plan->cards[1].array_offset[V41_DATA_ARRAY_DRAFT_EMBEDDING] =
	    SMALL_WEIGHTS_BYTES + TINY_IMAGE_BYTES + TINY_ZERO_BYTES;
}

/* The two cards of the copy's plan, card 0's embed.weight the pattern. */
static void copy_cards_open(const struct model_plan_record *plan,
			    struct device_card **cards,
			    const unsigned char *pattern)
{
    void *staging = NULL;
    uint64_t ticket = 0;
    test_host_memory_backend_reset(COPY_CARDS, plan->cards[1].arena_byte_count > plan->cards[0].arena_byte_count
                                                   ? plan->cards[1].arena_byte_count
                                                   : plan->cards[0].arena_byte_count);
    fill_calls = (struct fill_calls){0, 0};
    expect_status(test_host_memory_backend_set_stand_in(
			  KERNEL_FILE_FILL, KERNEL_FILL_WORDS, fill_stand_in,
			  &fill_calls),
		  STATUS_OK, "the fill has its stand-in");
    for (uint32_t card = 0; card < COPY_CARDS; card++) {
	    test_host_memory_backend_set_arithmetic_class(
		    card, ARITHMETIC_CLASS_SM_80);
	    expect_status(test_host_memory_backend.card_open(
				  card, plan->cards[card].arena_byte_count,
				  &cards[card]),
			  STATUS_OK, "a card opens");
    }
    result_status status =
	    test_host_memory_backend.staging_allocate(COPY_BYTES, &staging);
    if (status == STATUS_OK) {
	    memcpy(staging, pattern, COPY_BYTES);
	    status = test_host_memory_backend.upload_begin(
		    cards[0], COPY_SOURCE_OFFSET, staging, COPY_BYTES, &ticket);
	    status = status == STATUS_OK ?
			     test_host_memory_backend.upload_wait(
				     cards[0], ticket, WAIT_LIMIT_NANOSECONDS) :
			     status;
    }
    expect_status(status, STATUS_OK, "the embedding card holds embed.weight");
    test_host_memory_backend.staging_free(staging);
}

/* A fault of the copy, armed after the cards opened, what arena_initialise must answer, whether the staging stays (a
 * copy began that was not seen to finish: then a second line says so, and the first is not captured) and the words of
 * its one error line; a refusal before any work (a backend without downloads, an embedding past its card's weights, a
 * copy past the result card's arena). */
struct copy_case {
    const char *what;
    const char *words;
    uint64_t flipped_offset, download_ticket, upload_ticket, digest_number, source_offset, copy_offset;
    result_status expected;
    bool staging_is_kept;
    bool has_no_download;
};
static const struct copy_case copy_cases[] = {
	{ .what = "a byte the copy's link changed",
	  .expected = STATUS_DEVICE_ERROR,
	  .words = "the cards' digests of the two ranges differ",
	  .flipped_offset = COPY_FLIPPED_BYTE },
	{ .what = "a download that fails",
	  .expected = STATUS_DEVICE_ERROR,
	  .staging_is_kept = true,
	  .download_ticket = COPY_FIRST_DOWNLOAD },
	{ .what = "an upload of the copy that fails",
	  .expected = STATUS_DEVICE_ERROR,
	  .staging_is_kept = true,
	  .upload_ticket = COPY_SECOND_UPLOAD },
	{ .what = "a digest of the source that fails",
	  .expected = STATUS_DEVICE_ERROR,
	  .words = "the draft's copy of embed.weight",
	  .digest_number = COPY_SOURCE_DIGEST },
	{ .what = "a backend without downloads",
	  .expected = STATUS_INVALID_ARGUMENT,
	  .words = "not a plan",
	  .has_no_download = true },
	{ .what = "an embedding past its card's weights",
	  .expected = STATUS_INVALID_ARGUMENT,
	  .words = "not a plan",
	  .source_offset = COPY_WEIGHTS_BYTES },
	{ .what = "a copy past the result card's arena",
	  .expected = STATUS_INVALID_ARGUMENT,
	  .words = "not a plan",
	  .copy_offset = SMALL_WEIGHTS_BYTES + TINY_IMAGE_BYTES +
			 TINY_ZERO_BYTES + BLOCK },
};

/* The draft's copy of embed.weight (model_v41.h): behind both cards' state, three pieces, each downloaded from the
 * embedding card and uploaded into DRAFT_EMBEDDING, then the digests of both ranges; the copy holds the bytes. Then
 * each fault and refusal, the staging left where the contract says. */
static void test_embedding_copy(void)
{
    static unsigned char image[TINY_IMAGE_BYTES];
    struct model_plan_record *const plan = calloc(1, sizeof(*plan));
    unsigned char *const pattern = malloc(COPY_BYTES);
    struct device_card *cards[SMALL_CARDS] = { NULL, NULL, NULL };
    context_text = "the draft's copy of embed.weight";
    if (plan == NULL || pattern == NULL) {
        expect(false, "memory for a plan and a pattern");
        free(plan);
        free(pattern);
        return;
    }
    for (uint64_t index = 0; index < COPY_BYTES; index++) {
        pattern[index] = (unsigned char)((index * COPY_PATTERN_SEED) >> IMAGE_SHIFT);
    }
    memset(image, GARBAGE_BYTE, sizeof(image));
    copy_plan(plan, image);
    const uint64_t copy =
	    plan->cards[1].array_offset[V41_DATA_ARRAY_DRAFT_EMBEDDING];
    copy_cards_open(plan, cards, pattern);
    const uint64_t work_before = test_host_memory_backend_work_count();
    expect_status(v41_arena_initialise(plan, &watched_backend, cards,
				       WAIT_LIMIT_NANOSECONDS),
		  STATUS_OK, "the states are written and embed.weight copied");
    const unsigned char *const arena = test_host_memory_backend_arena_of(cards[1]);
    expect(arena != NULL && memcmp(arena + copy, pattern, COPY_BYTES) == 0 &&
               memcmp(arena + plan->cards[1].state_offset, image, sizeof(image)) == 0,
           "DRAFT_EMBEDDING holds embed.weight's bytes, behind the result card's image and zeros");
    const uint64_t copy_work = work_before + (uint64_t)COPY_CARDS * WORK_PER_CARD;  /* behind both cards' states */
    bool is_in_order = test_host_memory_backend_work_count() == copy_work + (uint64_t)COPY_PIECES * COPY_CARDS +
                                                                   COPY_CARDS;
    for (uint32_t piece = 0; piece < COPY_PIECES; piece++) {
	    const uint64_t done =
		    (uint64_t)piece * V41_DATA_EMBEDDING_COPY_PIECE_BYTES;
	    const uint64_t bytes =
		    COPY_BYTES - done < V41_DATA_EMBEDDING_COPY_PIECE_BYTES ?
			    COPY_BYTES - done :
			    V41_DATA_EMBEDDING_COPY_PIECE_BYTES;
	    struct test_host_memory_backend_work down, up;
	    const uint64_t at = copy_work + (uint64_t)COPY_CARDS * piece;
	    is_in_order = is_in_order &&
			  test_host_memory_backend_work_at(at, &down) &&
			  test_host_memory_backend_work_at(at + 1u, &up) &&
			  down.kind == TEST_HOST_MEMORY_BACKEND_DOWNLOAD &&
			  down.card_index == 0 &&
			  down.arena_byte_offset == COPY_SOURCE_OFFSET + done &&
			  down.byte_count == bytes &&
			  up.kind == TEST_HOST_MEMORY_BACKEND_UPLOAD &&
			  up.card_index == 1u &&
			  up.arena_byte_offset == copy + done &&
			  up.byte_count == bytes;
    }
    for (uint32_t card = 0; card < COPY_CARDS; card++) {
        struct test_host_memory_backend_work digest;
        is_in_order = is_in_order &&
                      test_host_memory_backend_work_at(copy_work + (uint64_t)COPY_CARDS * COPY_PIECES + card,
                                                       &digest) &&
                      digest.kind == TEST_HOST_MEMORY_BACKEND_DIGEST && digest.card_index == card &&
                      digest.arena_byte_offset == (card == 0 ? COPY_SOURCE_OFFSET : copy) &&
                      digest.byte_count == COPY_BYTES;
    }
    expect(is_in_order, "behind both states, each piece down then up, then both ranges' digests");
    cards_close(cards, false);
    for (size_t index = 0; index < sizeof(copy_cases) / sizeof(copy_cases[0]); index++) {
        const struct copy_case *const fault = &copy_cases[index];
	struct backend_operations backend = watched_backend;
	context_text = fault->what;
        copy_plan(plan, image);
	plan->globals[V41_DATA_EMBEDDING].stored.byte_offset +=
		fault->source_offset;
	plan->cards[1].array_offset[V41_DATA_ARRAY_DRAFT_EMBEDDING] +=
		fault->copy_offset;
	backend.download_begin = fault->has_no_download ? NULL : backend.download_begin;
        copy_cards_open(plan, cards, pattern);
        const uint64_t work_count = test_host_memory_backend_work_count();
        if (fault->flipped_offset != 0) {
            test_host_memory_backend_flip_bit_in_next_upload(1, copy + fault->flipped_offset);
        }
        if (fault->download_ticket != 0) {
            test_host_memory_backend_fail_upload(0, fault->download_ticket);
        }
        if (fault->upload_ticket != 0) {
            test_host_memory_backend_fail_upload(1, fault->upload_ticket);
        }
        if (fault->digest_number != 0) {
            test_host_memory_backend_fail_digest(0, fault->digest_number);
        }
        if (fault->words != NULL) {
            capture_begin();
        }
	const result_status status = v41_arena_initialise(
		plan, &backend, cards, WAIT_LIMIT_NANOSECONDS);
	if (fault->words != NULL) {
            capture_end_expecting(fault->words, "the one error line says why");
        }
        expect_status(status, fault->expected, "arena_initialise answers the fault");
	expect(fault->expected != STATUS_INVALID_ARGUMENT ||
		       test_host_memory_backend_work_count() == work_count,
	       "a plan refused before its work is given none");
	cards_close(cards, fault->staging_is_kept);
    }
    context_text = "the draft's copy of embed.weight";
    expect(test_host_memory_backend_open_card_count() == 0 && test_host_memory_backend_staging_block_count() == 0,
           "the case ends with no card open and no staging alive");
    test_host_memory_backend_reset(0, 0);
    free(plan);
    free(pattern);
}

/* ---- the frequency words ---- */

/* Every logical matrix aliases the first physical one except the last down matrix. Its scales must be read too. */
static void test_expert_fold_load(void)
{
	struct model_plan_record *const plan = calloc(1, sizeof(*plan));
	struct device_card *cards[SMALL_CARDS] = { NULL, NULL, NULL };
	void *scales = NULL;
	context_text = "the load's routed-expert scale folds";
	if (plan == NULL) {
		expect(false, "memory for an expert-fold plan");
		return;
	}
    const uint64_t arena_bytes = EXPERT_FOLD_PHYSICAL_MATRICES * EXPERT_FOLD_MATRIX_BYTES;
    plan->card_count = 1;
    plan->limits.cmp_order = CMP_ORDER_RELAXED;
    plan->cards[0].arena_byte_count = arena_bytes;
    plan->cards[0].state_offset = arena_bytes;
    plan->experts[0] = (struct linear_range){ 0, arena_bytes };
    plan->expert_tables[0][MODEL_EXPERTS - 1u][MODEL_EXPERT_MATRICES - 1u] =
	    EXPERT_FOLD_MATRIX_BYTES;
    watched_backend = test_host_memory_backend;
    watched_backend.staging_allocate = watched_staging_allocate;
    test_host_memory_backend_reset(1, arena_bytes);
    test_host_memory_backend_set_arithmetic_class(0, ARITHMETIC_CLASS_SM_80);
    result_status status =
	    test_host_memory_backend.card_open(0, arena_bytes, &cards[0]);
    status = status == STATUS_OK ? test_host_memory_backend.staging_allocate(
					   EXPERT_FOLD_SCALE_BYTES, &scales) :
				   status;
    expect_status(status, STATUS_OK, "the resident expert fixture opens");
    if (status != STATUS_OK) {
	    test_host_memory_backend.card_close(cards[0],
						CLOSE_LIMIT_NANOSECONDS);
	    free(plan);
	    return;
    }
    memset(scales, EXPERT_RELAXED_SCALE_SMALLEST,
	   (size_t)EXPERT_FOLD_SCALE_BYTES);
    for (uint32_t matrix = 0; matrix < EXPERT_FOLD_PHYSICAL_MATRICES; matrix++) {
        uint64_t ticket = 0;
        status = test_host_memory_backend.upload_begin(cards[0], matrix * EXPERT_FOLD_MATRIX_BYTES +
                                                       EXPERT_FOLD_WEIGHT_BYTES, scales,
                                                       EXPERT_FOLD_SCALE_BYTES, &ticket);
	status = status == STATUS_OK ?
			 test_host_memory_backend.upload_wait(
				 cards[0], ticket, WAIT_LIMIT_NANOSECONDS) :
			 status;
	expect_status(status, STATUS_OK,
		      "a physical expert matrix's scales are uploaded");
    }
    expect_status(v41_arena_initialise(plan, &watched_backend, cards,
				       WAIT_LIMIT_NANOSECONDS),
		  STATUS_OK,
		  "all resident expert matrices admit the relaxed fold");
    expect(plan->expert_folds_admitted[0], "the layer pins admitted expert folds after the final down matrix");
    ((uint8_t *)scales)[EXPERT_FOLD_SCALE_BYTES - 1u] =
	    EXPERT_RELAXED_SCALE_LARGEST + 1u;
    uint64_t ticket = 0;
    status = test_host_memory_backend.upload_begin(cards[0], EXPERT_FOLD_MATRIX_BYTES + EXPERT_FOLD_WEIGHT_BYTES,
                                                  scales, EXPERT_FOLD_SCALE_BYTES, &ticket);
    status = status == STATUS_OK ?
		     test_host_memory_backend.upload_wait(
			     cards[0], ticket, WAIT_LIMIT_NANOSECONDS) :
		     status;
    expect_status(status, STATUS_OK,
		  "the final expert matrix gets an unadmitted scale");
    expect_status(v41_arena_initialise(plan, &watched_backend, cards,
				       WAIT_LIMIT_NANOSECONDS),
		  STATUS_OK,
		  "a refused expert fold falls back without failing load");
    expect(!plan->expert_folds_admitted[0], "one refused code of the last expert forces the layer's reference order");
    const uint64_t work_before = test_host_memory_backend_work_count();
    test_host_memory_backend_set_arithmetic_class(0, ARITHMETIC_CLASS_SM_120);
    struct backend_operations failed_backend = watched_backend;
    failed_backend.card_describe = NULL;
    expect_status(
	    v41_arena_initialise(plan, &failed_backend, cards,
				 WAIT_LIMIT_NANOSECONDS),
	    STATUS_OK,
	    "fold admission follows opened card handles without physical-ordinal queries");
    expect(test_host_memory_backend_work_count() > work_before && !plan->expert_folds_admitted[0],
           "all resident layers check scales; only CMP selection consumes their admission");
    plan->limits.cmp_order = CMP_ORDER_REFERENCE;
    const uint64_t reference_work_before = test_host_memory_backend_work_count();
    failed_backend.download_begin = NULL;
    failed_backend.card_describe = NULL;
    expect_status(
	    v41_arena_initialise(plan, &failed_backend, cards,
				 WAIT_LIMIT_NANOSECONDS),
	    STATUS_OK,
	    "reference order requires no expert download or card-description operation");
    expect(test_host_memory_backend_work_count() == reference_work_before, "reference load performs no fold work");
    plan->limits.cmp_order = CMP_ORDER_RELAXED;
    test_host_memory_backend_set_arithmetic_class(0, ARITHMETIC_CLASS_SM_80);
    failed_backend = watched_backend;
    failed_backend.download_begin = failed_fold_download;
    plan->expert_folds_admitted[0] = true;
    expect_status(v41_arena_initialise(plan, &failed_backend, cards,
				       WAIT_LIMIT_NANOSECONDS),
		  STATUS_DEVICE_ERROR,
		  "a failed expert-scale download fails load");
    expect(!plan->expert_folds_admitted[0], "a failed reload clears stale expert-fold admission");
    failed_backend.download_begin = failed_fold_download_wait;
    expect_status(
	    v41_arena_initialise(plan, &failed_backend, cards,
				 WAIT_LIMIT_NANOSECONDS),
	    STATUS_DEVICE_ERROR,
	    "an unfinished expert-scale ticket fails load and retains staging");
    test_host_memory_backend.staging_free(scales);
    cards_close(cards, true);
    expect(test_host_memory_backend_staging_block_count() == 0, "expert folds retain no completed staging block");
    test_host_memory_backend_reset(0, 0);
    free(plan);
}

/* Both thetas' words as the reference forms them on the card (its stepwise capture's frequencies.theta_<T>.words,
 * which the table gate of model_v41_card_test holds the engine's to): the engine's table, written again here. */
static const uint32_t written_words[MODEL_THETAS][MODEL_FREQUENCIES] = {
	{ 0x3f800000u, 0x3f3ff911u, 0x3f0ff59au, 0x3ed7e89bu, 0x3ea1e89bu,
	  0x3e72d423u, 0x3e361887u, 0x3e088d77u, 0x3dcccccdu, 0x3d99940du,
	  0x3d6655c2u, 0x3d2cba16u, 0x3d0186e3u, 0x3cc2434fu, 0x3c91ad39u,
	  0x3c5a7bf2u, 0x3c23d70au, 0x3bf5b9b0u, 0x3bb8449cu, 0x3b8a2e77u,
	  0x3b4f3e38u, 0x3b1b690du, 0x3ae91528u, 0x3aaec98eu, 0x3a83126fu,
	  0x3a44948du, 0x3a136a16u, 0x39dd1725u, 0x39a5cb60u, 0x3978a815u,
	  0x393a7753u, 0x390bd472u },
	{ 0x3f800000u, 0x3f300a3au, 0x3ef21c1fu, 0x3ea67d01u, 0x3e64f92eu,
	  0x3e1d7475u, 0x3dd88cb4u, 0x3d94e963u, 0x3d4ccccdu, 0x3d0cd4fbu,
	  0x3cc1b019u, 0x3c8530ceu, 0x3c372dbfu, 0x3bfbed88u, 0x3bad3d5eu,
	  0x3b6e4237u, 0x3b147ae1u, 0x3ab714e0u, 0x3a5ebdb6u, 0x3a0530ceu,
	  0x399bb3afu, 0x39305978u, 0x38be904du, 0x383e9b5fu, 0x37a3d708u,
	  0x36b443d0u, 0x3677eba6u, 0x362a7be8u, 0x35ea77ffu, 0x35a13bdcu,
	  0x355dbf30u, 0x35187c4du },
};

static void test_frequency_words(void)
{
	float words[MODEL_THETAS][MODEL_FREQUENCIES];
	static const double bases[MODEL_THETAS] = {
		10000.0, 160000.0
	}; /* CFG:30, CFG:51 */
	context_text = "frequency words";
	v41_frequency_words(words);
	for (uint32_t theta = 0; theta < MODEL_THETAS; theta++) {
		for (uint32_t index = 0; index < MODEL_FREQUENCIES; index++) {
			uint32_t bits = 0;
			memcpy(&bits, &words[theta][index], sizeof(bits));
			if (bits != written_words[theta][index]) {
				(void)printf("FAILED: %s: theta %" PRIu32
					     " word %" PRIu32 " is %08" PRIx32
					     ", not %08" PRIx32 "\n",
					     context_text, theta, index, bits,
					     written_words[theta][index]);
				failures++;
			}
			const double exponent =
				(double)(MODEL_PAIR_VALUES * index) /
				MODEL_ROTATED_DIMS;
			const double plain = 1.0 / pow(bases[theta], exponent);
			/* REF:383-386: the ramp from low to high, the word divided by the factor beyond it */
			const double ramp =
				fmin(fmax(((double)index - LOWEST_BAND_INDEX) /
						  (HIGHEST_BAND_INDEX -
						   LOWEST_BAND_INDEX),
					  0.0),
				     1.0);
			const double expected =
				theta == 0 ?
					plain :
					plain / (double)ROPE_FACTOR * ramp +
						plain * (1.0 - ramp);
			const double place =
				(double)nextafterf(words[theta][index],
						   INFINITY) -
				(double)words[theta][index];
			expect(fabs((double)words[theta][index] - expected) <=
				       WORD_PLACES * place,
			       "a word within four places of the reference's expression in double precision");
		}
	}
}

/* ---- the real pack ---- */

static void test_real_pack(const char *path)
{
    static const uint64_t budgets_mib[CARDS] = {CARD_170HX_BUDGET_MIB, CARD_170HX_BUDGET_MIB, CARD_170HX_BUDGET_MIB,
                                                CARD_170HX_BUDGET_MIB, CARD_PRO_BUDGET_MIB};
    struct storage_reader *direct = NULL, *reader = NULL;
    struct pack_record *pack = NULL;
    struct load_layout *layout = NULL;
    struct model_plan_record *plan = NULL;
    uint64_t budgets[CARDS];
    uint32_t on_demand = 0;
    float words[MODEL_THETAS][MODEL_FREQUENCIES];
    context_text = "the real pack's tables";
    test_memory_storage_reset();
    result_status status = storage_direct.reader_open(&path, 1, &direct);
    status = status == STATUS_OK ?
		     test_memory_storage_hold_tables_of(&storage_direct,
							direct) :
		     status;
    storage_direct.reader_close(direct);
    status = status == STATUS_OK ?
		     test_memory_storage.reader_open(&path, 1, &reader) :
		     status;
    status = status == STATUS_OK ?
		     pack_open_entry(&test_memory_storage, reader, &pack) :
		     status;
    test_memory_storage.reader_close(reader);
    expect_status(status, STATUS_OK, "the real pack opens from its tables");
    const uint32_t unit_count =
	    pack != NULL ? pack_header_of(pack)->unit_count : 0;
    uint32_t *const card_of_unit = calloc(unit_count + 1u, sizeof(uint32_t));
    for (uint32_t card = 0; card < CARDS; card++) {
        budgets[card] = budgets_mib[card] * BYTES_PER_MIB;
    }
    status = status == STATUS_OK && card_of_unit != NULL ?
		     placement_fill_in_order(pack, budgets, CARDS, unit_count,
					     card_of_unit, &on_demand) :
		     status;
    expect(status != STATUS_OK || on_demand == EXPERTS,
	   "every expert of a layer sits on its card");
    status = status == STATUS_OK ?
		     load_layout_create(pack, card_of_unit, unit_count, CARDS,
					&layout) :
		     status;
    status = status == STATUS_OK ?
		     v41_plan_create(pack, layout, &limits_of_m1, &plan) :
		     status;
    expect_status(status, STATUS_OK,
		  "the real pack is placed on five cards and planned");
    if (plan != NULL) {
	    const struct placement_record placement = { card_of_unit,
							unit_count, CARDS };
	    for (uint32_t card = 0; card < CARDS; card++) {
		    const struct v41_card *const planned = &plan->cards[card];
		    uint64_t state = 0;
		    expect_status(v41_state_byte_count(pack, &placement,
						       &limits_of_m1, card,
						       &state),
				  STATUS_OK, "a card's state is counted");
		    expect(state == planned->arena_byte_count -
					    planned->state_offset,
			   "the state is the plan's");
		    (void)printf(
			    "real pack: card %" PRIu32 ": layers %#" PRIx64
			    ", weights %.2f GiB, state %.2f MiB "
			    "(%" PRIu32 " sessions, %u rows, %u tokens)%s%s\n",
			    card, planned->layers,
			    (double)planned->state_offset / (double)GIB,
			    (double)state / (double)BYTES_PER_MIB, SESSIONS,
			    ROWS, CONTEXT,
			    card == plan->embedding_card ? ", the embedding" :
							   "",
			    card == plan->result_card ? ", the head" : "");
	    }
	    v41_frequency_words(words);
	    for (uint32_t theta = 0; theta < MODEL_THETAS; theta++) {
		    (void)printf("real pack: frequency words of theta %" PRIu32
				 ":",
				 theta);
		    for (uint32_t index = 0; index < MODEL_FREQUENCIES;
			 index++) {
			    uint32_t bits = 0;
			    memcpy(&bits, &words[theta][index], sizeof(bits));
			    (void)printf(" %08" PRIx32, bits);
		    }
		    (void)printf("\n");
	    }
    }
    v41_plan_destroy(plan);
    load_layout_destroy(layout);
    free(card_of_unit);
    pack_close(pack);
    test_memory_storage_reset();
}

/* The opt-in's shape and recipe admission must agree for planning and state sizing. */
static void test_official_hc_limits(const struct pack_record *pack)
{
	static const uint32_t reference_rows[] = { 0, 1, 3, 4, 8,	  2,
						   5, 6, 7, 9, UINT32_MAX };
	static uint32_t card_of_unit[UNIT_COUNT];
	test_model_v41_five_cards(card_of_unit, UNIT_COUNT, 0);
	const struct placement_record placement = { card_of_unit, UNIT_COUNT,
						    CARDS };
	context_text = "official HC limits";
	for (uint32_t index = 0;
	     index < sizeof(reference_rows) / sizeof(reference_rows[0]);
	     index++) {
		struct model_limits limits = limits_of_m1;
		limits.linear_order = LINEAR_ORDER_REFERENCE;
		limits.official_hc_rows = reference_rows[index];
		uint64_t bytes = UINT64_MAX;
		const bool admitted =
			limits.official_hc_rows == 0 ||
			limits.official_hc_rows == 1 ||
			limits.official_hc_rows ==
				OFFICIAL_HC_ODD_PROMPT_ROWS ||
			limits.official_hc_rows == MODEL_STREAMS ||
			limits.official_hc_rows == LINEAR_MAXIMUM_ROWS;
		expect_status(v41_state_byte_count(pack, &placement, &limits, 0,
						   &bytes),
			      admitted ? STATUS_OK : STATUS_INVALID_ARGUMENT,
			      "only measured reference shapes admitted");
		expect(admitted || bytes == UINT64_MAX,
		       "refused shape leaves byte count untouched");
	}
	struct model_limits cases[] = { limits_of_m1, limits_of_m1,
					limits_of_m1, limits_of_m1,
					limits_of_m1 };
	enum { WIDE_CASE, SPEED_CASE, CMP_CASE, DRAFT_CASE };
	cases[WIDE_CASE].row_capacity = MODEL_WIDE_LEAST_ROWS;
	cases[SPEED_CASE].linear_profile = LINEAR_PROFILE_SPEED;
	cases[CMP_CASE].cmp_order = CMP_ORDER_RELAXED;
	cases[DRAFT_CASE].draft_positions = MODEL_DRAFT_BLOCK;
	for (uint32_t index = 0; index < sizeof(cases) / sizeof(cases[0]);
	     index++) {
		struct model_limits *const limits = &cases[index];
		limits->official_hc_rows = LINEAR_MAXIMUM_ROWS;
		if (index + 1u < sizeof(cases) / sizeof(cases[0]))
			limits->linear_order = LINEAR_ORDER_REFERENCE;
		uint64_t bytes = UINT64_MAX;
		expect_status(v41_state_byte_count(pack, &placement, limits, 0,
						   &bytes),
			      STATUS_UNSUPPORTED,
			      "unqualified HC recipes refused");
		expect(bytes == UINT64_MAX,
		       "unsupported recipe leaves byte count untouched");
	}
}

int main(int argument_count, char **arguments)
{
    (void)setvbuf(stdout, NULL, _IOLBF, 0);             /* a failure's line survives a sanitizer's stop */
    if (argument_count > ARGUMENT_COUNT_WITH_A_PACK) {
        (void)printf("usage: model_v41_test [the path of a pack]\n");
        return EXIT_FAILURE;
    }
    error_stream_descriptor = dup(STDERR_FILENO);
    capture_descriptor = mkstemp(capture_path);
    if (error_stream_descriptor < 0 || capture_descriptor < 0) {
        (void)printf("FAILED: the temporary file or the memory of the test could not be made\n");
        return EXIT_FAILURE;
    }
    test_frequency_words();
    test_arena_initialise();
    test_fold_load();
    test_embedding_copy();
    test_expert_fold_load();
    struct pack_record *pack = NULL;
    context_text = "the model's pack";
    expect_status(test_model_v41_pack_open(
			  (struct test_model_v41_change){
				  TEST_MODEL_V41_CHANGE_NONE, NULL, 0 },
			  &pack),
		  STATUS_OK, "the model's pack opens from its tables");
    if (pack != NULL) {
        test_five_cards(pack);
        test_chunk_plans(pack);
        test_refusals(pack);
	test_official_hc_limits(pack);
	test_draft_off(pack);
    }
    pack_close(pack);
    test_tensor_refusals();
    test_draft_plans();
    test_draft_refusals();
    test_memory_storage_reset();
    if (argument_count == ARGUMENT_COUNT_WITH_A_PACK) {
        test_real_pack(arguments[1]);
    } else {
        (void)printf("model_v41_test: no pack was given, no real pack's tables were planned\n");
    }
    (void)close(capture_descriptor);
    (void)unlink(capture_path);
    if (failures != 0) {
        (void)printf("model_v41_test FAILED: %d failures\n", failures);
        return EXIT_FAILURE;
    }
    (void)printf("model_v41_test passed\n");
    return EXIT_SUCCESS;
}
