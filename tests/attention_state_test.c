/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * attention_state_test: source/attention_state.c against attention_state.h and attention_state_records.h, and the
 * host reference of the attention state itself. No card: a prepare function is arithmetic on a description.
 *
 *   - The four kinds of layer (window only; an indexer; a pool of ratio 1; of ratio 2) for 1 and 8 rows and 1 and 3
 *     sessions, and a pool of ratio 2 for half the heads, each range exactly the bytes counted here by hand, the arena
 *     ending with the last: accepted without a word, every launch of the kernel, grid, block and record
 *     attention_state_records.h states; one byte less in any range, or of the arena, is refused.
 *   - Every refusal of attention_state.h, alone in a description otherwise accepted: the status, one error line that
 *     names the value refused (the line of a group of rules names them all), the launches left as they were; of the
 *     layer and of the rotation step.
 *   - qr alone over a chunk's rows (MODEL_WIDE_MOST_ROWS): accepted, the prepare kernel over rows by 1 with
 *     a record of its three offsets alone; each of its refusals, alone, as the rotation step's.
 *   - The queries alone: of a slab's rows (1 and 8) of a layer of each kind (the window alone in theta 0 and theta
 *     1, the indexer, a pool of ratio 2), the layer's QUERIES launch byte for byte; over a chunk's rows with the
 *     indexer, the queries kernel over its heads and the index heads by the rows, its record by hand; each of its
 *     refusals, alone.
 *   - The kernels on a card of the host-memory backend: none on a card of no class; the five in order, with their
 *     record sizes, on a card of class 80, the draft's window left unprepared.
 *   - The window alone (dspark_v1/DESIGN.md C2) over a chunk's rows: accepted, the window kernel over rows by 1 with
 *     the prepare's record of the window part; each of its refusals, alone.
 *   - The reference of the window row: of a row of ones and weights of ones, the row is ones (the factor 1).
 *   - The reference: the square root, the exponential, the cosine and sine at known values (2^20 - 1 radians among
 *     them) and their identities; FP16 rounding at its edges; both placements a bijection that puts the values
 *     model_layout.h names where it says. The card tests' rule of a rounded value: the nearest, either at a tie.
 * A failure is one line on the standard output: the case, and what was expected of it.
 */
#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "anemone/attention_state.h"
#include "log_expectation.h"
#include "anemone/attention_state_records.h"
#include "anemone/backend_seam.h"
#include "anemone/status.h"
#include "attention_state_card_fixture.h"
#include "attention_state_reference.h"
#include "host_memory_backend.h"

#define COUNT_OF(table) (sizeof(table) / sizeof((table)[0]))
#define INVALID STATUS_INVALID_ARGUMENT
#define CAPTURE_CAPACITY 4096
#define UNTOUCHED_BYTE 0xA5
#define ALIGNMENT 16u /* of every range: LINEAR_OFFSET_ALIGNMENT_BYTES */
#define RANGES 21u
#define ROWS_OF_A_FULL_STEP 8u
#define SESSIONS 3u
#define CONTEXT 64u
#define HEADS 64u
#define CARD_BYTES (1ull << 30u)
#define CARD_ARENA_BYTES (1ull << 20u)
#define CLASS_80 ARITHMETIC_CLASS_SM_80
/* The bytes of the arrays, by hand from model_layout.h: per row, per session, or of a weight. */
#define STEP_BLOCK 16ull
#define STEP_ROW 16ull
#define ROTATION_ROW 768ull
#define QUERY_RANK_ROW 2560ull
#define STATE_ROW 1024ull
#define RING_SESSION 4194304ull                 /* 4,096 slots of 1,024 bytes */
#define INDEX_ROW 8192ull
#define FLOAT_STATE_ROW 2048ull
#define KEY 256ull
#define SLOTS_SESSION 65536ull
/* Known values of the reference. */
#define SQUARE_ROOT_OF_TWO 1.41421356237309504880
#define EULER 2.71828182845904523536
#define HALF_PI 1.57079632679489661923
#define DOUBLE_TOLERANCE 1e-14
#define IDENTITY_ANGLES 1000u
#define IDENTITY_STEP 1048.57
#define FOUR 4.0
#define LARGEST_ANGLE 1048575.0                 /* 2^20 - 1: the last position by the word 1 */
#define COSINE_OF_IT 0.78804223952892746867     /* to 20 places, from a reduction in 80 digits */
#define SINE_OF_IT (-0.61562117305875088409)
#define HALF_PLACE_ABOVE_ONE 1.00048828125      /* 1 + 2^-11: the tie between FP16's 1 and the next */
#define HALF_NEXT_AFTER_ONE 1.0009765625        /* 1 + 2^-10 */
#define NEARER_THE_NEXT 1.0008                  /* rounds to it, cut toward zero to 1 */
#define FREQUENCIES_BYTES 256u                  /* float32 [2][32] */
#define QUERY_NORM_BYTES 2560ull                /* bfloat16 [1280] */
#define MISALIGNMENT 8u
#define INDEX_HALVES 4096u                      /* 32 heads of 128 */
#define HALF_ONE 0x3C00u
#define HALF_MINUS_ZERO 0x8000u
#define HALF_MINUS_SMALLEST 0x8001u
#define KEY_HALVES 128u
#define DRAFT_HANDLE_BYTES sizeof(uint32_t)      /* the window's kernel, which draft.h prepares */
#define WINDOW_DIMS 512u
#define WINDOW_ROTATION_FLOATS 64u              /* of a row's use: 32 (cos, sin) */
#define BFLOAT16_ONE 0x3F80u
#define HALF_OF_ONE 0x3C00u

struct half_case {
    double value;
    uint16_t bits;
};
static const struct half_case half_cases[] = {
    {1.0, 0x3C00u}, {65504.0, 0x7BFFu}, {65519.99, 0x7BFFu}, {65520.0, 0x7C00u}, {-65520.0, 0xFC00u},
    {5.9604644775390625e-8, 0x0001u},           /* 2^-24 */
    {2.98023223876953125e-8, 0x0000u},          /* 2^-25: a tie to the even zero */
    {8.94069671630859375e-8, 0x0002u},          /* 3 * 2^-25: a tie to the even 2 */
    {1.00048828125, 0x3C00u},                   /* 1 + 2^-11: a tie to the even 1 */
    {1.00146484375, 0x3C02u},                   /* 1 + 3 * 2^-11: a tie to the even 1 + 2^-9 */
    {6.103515625e-05, 0x0400u},                 /* 2^-14 */
    {6.0975551605224609375e-05, 0x03FFu},       /* the largest subnormal */
};

/* Known places: the halves of (head, dim) and of a key's dim, model_layout.h's order by hand. */
struct place_case {
    uint32_t head;
    uint32_t dim;
    uint32_t half;
};
static const struct place_case query_places[] = {{0, 0, 0}, {9, 0, 34}, {19, 93, 3445}, {31, 127, 4095}};
static const struct place_case key_places[] = {{0, 0, 0}, {0, 9, 3}, {0, 2, 32}, {0, 27, 39}, {0, 127, 127}};

static int failures = 0;
static const char *context_text = "";
static char capture_path[] = "/tmp/anemone_attention_state_test_XXXXXX";
static int capture_descriptor = -1;
static int error_stream_descriptor = -1;
static const struct attention_state_kernels made_up_kernels = { 40, 41, 42,
								43, 44, 45 };

static void expect(bool condition, const char *what)
{
    if (!condition) {
        (void)printf("FAILED: %s: %s\n", context_text, what);
        failures++;
    }
}

static void capture_begin(void)
{
    expect(capture_descriptor >= 0 && fflush(stderr) == 0 && ftruncate(capture_descriptor, 0) == 0 &&
               lseek(capture_descriptor, 0, SEEK_SET) == 0 && dup2(capture_descriptor, STDERR_FILENO) == STDERR_FILENO,
           "the error stream is pointed at the temporary file");
}

static void capture_end(char *text, size_t text_capacity)
{
    const bool is_open = capture_descriptor >= 0 && error_stream_descriptor >= 0 && fflush(stderr) == 0;
    const ssize_t byte_count = is_open ? pread(capture_descriptor, text, text_capacity - 1, 0) : -1;
    expect(is_open && dup2(error_stream_descriptor, STDERR_FILENO) == STDERR_FILENO && byte_count >= 0,
           "the temporary file is read back and the error stream restored");
    text[byte_count >= 0 ? (size_t)byte_count : 0] = '\0';
}

static void expect_one_error_line_with(const char *text, const char *words, const char *what)
{
    if (!test_log_is_expected(text, words, true)) {
        (void)printf("FAILED: %s: %s: no single error line with \"%s\": \"%s\"\n", context_text, what, words, text);
        failures++;
    }
}

/* A layer of a kind, its ranges laid out one after another in the order of the description. */
struct layer_case {
	struct attention_state_description description;
	uint64_t arena_byte_count;
	uint32_t range_count; /* the ranges the kind has: a prefix of the description's */
};

/* The ranges of a description in their order, which is the order prepare checks them in. */
static struct linear_range *
range_at(struct attention_state_description *description, uint32_t index)
{
	struct linear_range *const ranges[RANGES] = {
		&description->step,	    &description->rotation,
		&description->query_rank,   &description->query_norm,
		&description->query_normed, &description->window_input,
		&description->window_norm,  &description->ring,
		&description->heads,	    &description->queries,
		&description->index_heads,  &description->index_queries,
		&description->pool_norm,    &description->latent,
		&description->pool,	    &description->key_input,
		&description->key_norm,	    &description->keys,
		&description->pool_input,   &description->pool_score,
		&description->slots
	};
	return ranges[index];
}

/* A copy of the kernels with one of the layer's four not prepared. */
static struct attention_state_kernels without_kernel(uint32_t launch)
{
	struct attention_state_kernels kernels = made_up_kernels;
	uint32_t *const handles[ATTENTION_STATE_LAUNCH_COUNT] = {
		&kernels.prepare, &kernels.queries, &kernels.compressor,
		&kernels.key
	};
	*handles[launch] = BACKEND_NO_KERNEL;
	return kernels;
}

/* The bytes of range index of a layer, counted by hand; 0 for a range the layer has not. */
static uint64_t bytes_of(const struct attention_state_description *layer,
			 uint32_t index)
{
    const uint64_t rows = layer->row_count, sessions = layer->session_capacity;
    const uint64_t entries = layer->pool_ratio == 0 ? 0 : layer->context_token_limit / layer->pool_ratio;
    const uint64_t counts[RANGES] = {
        STEP_BLOCK + STEP_ROW * rows, ROTATION_ROW * rows, QUERY_RANK_ROW * rows, QUERY_RANK_ROW,
        QUERY_RANK_ROW * rows, STATE_ROW * rows, STATE_ROW, RING_SESSION * sessions,
        STATE_ROW * HEADS * rows, STATE_ROW * HEADS * rows, INDEX_ROW * rows,
        INDEX_ROW * rows, STATE_ROW, STATE_ROW * rows, STATE_ROW * entries * sessions, KEY * rows, KEY,
        KEY * entries * sessions, (layer->pool_ratio == 1 ? STATE_ROW : FLOAT_STATE_ROW) * rows,
        FLOAT_STATE_ROW * rows, SLOTS_SESSION * sessions};
    return counts[index];
}

static struct layer_case layer_of(uint32_t theta, uint32_t has_index, uint32_t ratio, uint32_t rows, uint32_t sessions)
{
    static const uint32_t prefixes[] = {10, 12, 19, 21};       /* window only, indexer, ratio 1, ratio 2 */
    struct layer_case layer;
    memset(&layer, 0, sizeof(layer));
    layer.description =
	    (struct attention_state_description){ .row_count = rows,
						  .session_capacity = sessions,
						  .context_token_limit =
							  CONTEXT,
						  .theta = theta,
						  .first_head = 0,
						  .head_count = HEADS,
						  .has_index = has_index,
						  .pool_ratio = ratio };
    layer.range_count = prefixes[ratio != 0 ? 1u + ratio : has_index];
    for (uint32_t index = 0; index < layer.range_count; index++) {
        const uint64_t byte_count = bytes_of(&layer.description, index);
	*range_at(&layer.description, index) =
		(struct linear_range){ layer.arena_byte_count, byte_count };
	layer.arena_byte_count += (byte_count + ALIGNMENT - 1u) / ALIGNMENT * ALIGNMENT;
    }
    layer.arena_byte_count = range_at(&layer.description, layer.range_count - 1u)->byte_offset +
                             range_at(&layer.description, layer.range_count - 1u)->byte_count;
    return layer;
}

static void expect_launch(const struct kernel_launch *launch, uint32_t kernel,
			  uint32_t grid_columns, uint32_t grid_rows,
			  uint32_t block_threads, const void *record,
			  size_t record_byte_count, const char *what)
{
	unsigned char zeros[LAUNCH_ARGUMENT_BYTES];
	memset(zeros, 0, sizeof(zeros));
	expect(launch->kernel == kernel && launch->grid[0] == grid_columns &&
		       launch->grid[1] == grid_rows &&
		       launch->grid[LAUNCH_DIMENSIONS - 1] == 1 &&
		       launch->block[0] == block_threads &&
		       launch->block[1] == 1 &&
		       launch->block[LAUNCH_DIMENSIONS - 1] == 1 &&
		       launch->dynamic_shared_byte_count == 0 &&
		       launch->argument_byte_count == record_byte_count &&
		       launch->branch == LAUNCH_CHAIN &&
		       memcmp(launch->argument, record, record_byte_count) ==
			       0 &&
		       memcmp(launch->argument + record_byte_count, zeros,
			      sizeof(zeros) - record_byte_count) == 0,
	       what);
}

/* The four launches of an accepted layer, against records built here from its ranges. */
static void expect_layer_launches(const struct layer_case *layer,
				  const struct kernel_launch *launches)
{
	const struct attention_state_description *const d =
		&layer->description; /* short: records below */
	const uint32_t ratio = d->pool_ratio;
	const uint64_t entries = ratio == 0 ? 0 : CONTEXT / ratio;
	const bool is_two_to_one = ratio == MODEL_POOLED_RATIO;
	const uint32_t use = d->theta == 0 ? ATTENTION_ROTATION_WINDOW_ONLY :
					     ATTENTION_ROTATION_POOLED;
	const struct attention_prepare_arguments prepare = {
		d->step.byte_offset,
		d->rotation.byte_offset,
		d->query_rank.byte_offset,
		d->query_norm.byte_offset,
		d->query_normed.byte_offset,
		d->window_input.byte_offset,
		d->window_norm.byte_offset,
		d->ring.byte_offset,
		RING_SESSION,
		use,
		d->session_capacity,
		CONTEXT,
		0
	};
	const struct attention_queries_arguments queries = {
		d->rotation.byte_offset,
		d->heads.byte_offset,
		d->queries.byte_offset,
		d->has_index != 0 ? d->index_heads.byte_offset : 0,
		d->has_index != 0 ? d->index_queries.byte_offset : 0,
		d->first_head,
		d->head_count,
		d->has_index != 0 ? MODEL_INDEX_HEADS : 0,
		use,
		HEADS,
		d->first_head
	};
	const struct attention_compressor_arguments compressor = {
		d->step.byte_offset,
		d->rotation.byte_offset,
		d->pool_input.byte_offset,
		is_two_to_one ? d->pool_score.byte_offset : 0,
		is_two_to_one ? d->slots.byte_offset : 0,
		SLOTS_SESSION,
		d->pool_norm.byte_offset,
		d->latent.byte_offset,
		d->pool.byte_offset,
		STATE_ROW * entries,
		ratio,
		d->session_capacity,
		CONTEXT,
		0
	};
	const struct attention_key_arguments key = { d->step.byte_offset,
						     d->rotation.byte_offset,
						     d->key_input.byte_offset,
						     d->key_norm.byte_offset,
						     d->keys.byte_offset,
						     KEY * entries,
						     ratio,
						     d->session_capacity,
						     CONTEXT,
						     0 };
	const uint32_t index_heads = d->has_index != 0 ? MODEL_INDEX_HEADS : 0;
	struct kernel_launch zero;
	memset(&zero, 0, sizeof(zero));
	expect_launch(&launches[ATTENTION_STATE_PREPARE],
		      made_up_kernels.prepare, d->row_count,
		      ATTENTION_PREPARE_PARTS, ATTENTION_STATE_THREADS,
		      &prepare, sizeof(prepare), "the prepare's launch");
	expect_launch(&launches[ATTENTION_STATE_QUERIES],
		      made_up_kernels.queries, d->head_count + index_heads,
		      d->row_count, ATTENTION_STATE_THREADS, &queries,
		      sizeof(queries), "the queries' launch");
	if (ratio == 0) {
		expect(memcmp(&launches[ATTENTION_STATE_COMPRESSOR], &zero,
			      sizeof(zero)) == 0 &&
			       memcmp(&launches[ATTENTION_STATE_KEY], &zero,
				      sizeof(zero)) == 0,
		       "a layer without a pool has two launches of no block");
		return;
	}
	expect_launch(&launches[ATTENTION_STATE_COMPRESSOR],
		      made_up_kernels.compressor, d->row_count, 1,
		      ATTENTION_STATE_THREADS, &compressor, sizeof(compressor),
		      "the compressor's launch");
	expect_launch(&launches[ATTENTION_STATE_KEY], made_up_kernels.key,
		      d->row_count, 1, ATTENTION_STATE_THREADS, &key,
		      sizeof(key), "the key's launch");
}

static void layer_refused(const struct layer_case *layer,
			  const struct attention_state_kernels *kernels,
			  const char *rule, const char *what)
{
	struct kernel_launch launches[ATTENTION_STATE_LAUNCH_COUNT];
	unsigned char untouched[sizeof(launches)];
	char logged[CAPTURE_CAPACITY];
	memset(launches, UNTOUCHED_BYTE, sizeof(launches));
	memset(untouched, UNTOUCHED_BYTE, sizeof(untouched));
	capture_begin();
	const result_status status = attention_state_prepare(
		&layer->description, layer->arena_byte_count, kernels,
		launches);
	capture_end(logged, sizeof(logged));
	expect(status == INVALID, what);
	expect_one_error_line_with(logged, rule, what);
	expect(memcmp(launches, untouched, sizeof(untouched)) == 0,
	       "a refusal leaves the launches untouched");
}

static void test_layers(void)
{
    static const struct {
        uint32_t theta, has_index, pool_ratio;
    } kinds[] = {
	    { 0, 0, 0 }, { 1, 1, 0 }, { 1, 1, 1 }, { 1, 1, MODEL_POOLED_RATIO }
    };
    static const uint32_t row_counts[] = {1, ROWS_OF_A_FULL_STEP};
    static const uint32_t session_counts[] = {1, SESSIONS};
    for (size_t kind = 0; kind < COUNT_OF(kinds); kind++) {
        for (size_t rows = 0; rows < COUNT_OF(row_counts); rows++) {
            for (size_t sessions = 0; sessions < COUNT_OF(session_counts); sessions++) {
                struct layer_case layer = layer_of(kinds[kind].theta, kinds[kind].has_index, kinds[kind].pool_ratio,
                                                   row_counts[rows], session_counts[sessions]);
		struct kernel_launch launches[ATTENTION_STATE_LAUNCH_COUNT];
		char logged[CAPTURE_CAPACITY];
                context_text = "a layer of each kind";
                capture_begin();
		const result_status status = attention_state_prepare(
			&layer.description, layer.arena_byte_count,
			&made_up_kernels, launches);
		capture_end(logged, sizeof(logged));
		expect(status == STATUS_OK && logged[0] == '\0',
		       "accepted without a word");
		if (status == STATUS_OK) {
			expect_layer_launches(&layer, launches);
		}
		for (uint32_t index = 0; index < layer.range_count; index++) {
                    struct layer_case short_range = layer;
                    range_at(&short_range.description, index)->byte_count -= 1u;
                    context_text = "a range one byte short";
                    layer_refused(&short_range, &made_up_kernels, "holds", "a range one byte short is refused");
                }
                struct layer_case short_arena = layer;
                short_arena.arena_byte_count -= 1u;
                context_text = "an arena one byte short";
                layer_refused(&short_arena, &made_up_kernels, "leaves the arena", "the last range leaves the arena");
            }
        }
    }
    struct layer_case half =
	    layer_of(1, 1, MODEL_POOLED_RATIO, ROWS_OF_A_FULL_STEP, SESSIONS);
    struct kernel_launch launches[ATTENTION_STATE_LAUNCH_COUNT];
    half.description.first_head = HEADS / MODEL_PAIR_VALUES;
    half.description.head_count = HEADS / MODEL_PAIR_VALUES;
    context_text = "the second half of the heads";
    expect(attention_state_prepare(&half.description, half.arena_byte_count,
				   &made_up_kernels, launches) == STATUS_OK,
	   "accepted");
    expect_layer_launches(&half, launches);
}

/* A count of a description set to a value its rule refuses, and where the error line names the value. */
struct member_refusal {
    size_t member_offset;
    uint32_t value;
    const char *before;                         /* the line holds these words, the value, then after */
    const char *after;
};
#define MEMBER(member) offsetof(struct attention_state_description, member)
static const struct member_refusal member_refusals[] = {
	{ MEMBER(row_count), 0, ": ", " rows:" },
	{ MEMBER(row_count), ROWS_OF_A_FULL_STEP + 1u, ": ", " rows:" },
	{ MEMBER(session_capacity), 0, ": ", " sessions of" },
	{ MEMBER(context_token_limit), 0, " of ", " tokens," },
	{ MEMBER(context_token_limit),
	  CONTEXT + MODEL_CONTEXT_GRANULE / MODEL_POOLED_RATIO, " of ",
	  " tokens," },
	{ MEMBER(head_count), 0, ", ", " heads from head" },
	{ MEMBER(head_count), HEADS + 1u, ", ", " heads from head" },
	{ MEMBER(first_head), 1, "from head ", ", theta" },
	{ MEMBER(theta), MODEL_THETAS, ", theta ", ", indexer" },
	{ MEMBER(has_index), MODEL_POOLED_RATIO, ", indexer ", ", pool ratio" },
	{ MEMBER(pool_ratio), MODEL_POOLED_RATIO + 1u, ", pool ratio ", ":" },
	{ MEMBER(has_index), 0, ", indexer ", ", pool ratio" },
	{ MEMBER(theta), 0, ", theta ", ", indexer" },
};
#undef MEMBER

/* Each rule alone, in a pool of ratio 2 (or the kind the rule needs) that is otherwise accepted. */
static void test_layer_refusals(void)
{
	const struct layer_case base = layer_of(1, 1, MODEL_POOLED_RATIO,
						ROWS_OF_A_FULL_STEP, SESSIONS);
	struct layer_case layer = base;
	struct kernel_launch launches[ATTENTION_STATE_LAUNCH_COUNT];
	context_text = "a refused layer";
	capture_begin();
	expect(attention_state_prepare(NULL, base.arena_byte_count,
				       &made_up_kernels, launches) == INVALID &&
		       attention_state_prepare(&base.description,
					       base.arena_byte_count, NULL,
					       launches) == INVALID &&
		       attention_state_prepare(
			       &base.description, base.arena_byte_count,
			       &made_up_kernels, NULL) == INVALID,
	       "a missing argument is refused");
	char logged[CAPTURE_CAPACITY];
	capture_end(logged, sizeof(logged));
	for (uint32_t launch = 0; launch < ATTENTION_STATE_LAUNCH_COUNT;
	     launch++) {
		const struct attention_state_kernels kernels =
			without_kernel(launch);
		layer_refused(&base, &kernels, "is not prepared",
			      "a kernel of the layer not prepared");
	}
    layer = base;
    layer.description.reserved_zero = 1;
    layer_refused(&layer, &made_up_kernels, "a reserved field", "a reserved field not zero");
    for (size_t index = 0; index < COUNT_OF(member_refusals); index++) {
        const struct member_refusal *const refusal = &member_refusals[index];
        char words[CAPTURE_CAPACITY];
        layer = base;
        memcpy((unsigned char *)&layer.description + refusal->member_offset, &refusal->value, sizeof(uint32_t));
        (void)snprintf(words, sizeof(words), "%s%" PRIu32 "%s", refusal->before, refusal->value, refusal->after);
        layer_refused(&layer, &made_up_kernels, words, words);
    }
    layer = layer_of(1, 1, 0, 1, 1);
    layer.description.theta = 0;
    layer_refused(&layer, &made_up_kernels, "theta 0, indexer 1, pool ratio 0", "an indexer of theta 0");
    layer = base;
    layer.description.ring.byte_offset += MISALIGNMENT;
    layer_refused(&layer, &made_up_kernels, "an offset is a multiple", "a range that is not aligned");
    layer = base;
    layer.description.queries = layer.description.heads;
    layer_refused(&layer, &made_up_kernels, "share a byte", "two ranges that share a byte");
    layer = base;
    layer.description.key_norm.byte_count += ALIGNMENT;
    layer_refused(&layer, &made_up_kernels, "its format stores", "a weight longer than its bytes");
}

static void test_rotation(void)
{
	struct attention_rotation_description description = {
		{ 0, STEP_BLOCK + STEP_ROW * ROWS_OF_A_FULL_STEP },
		{ STEP_BLOCK + STEP_ROW * ROWS_OF_A_FULL_STEP,
		  FREQUENCIES_BYTES },
		{ 0, 0 },
		ROWS_OF_A_FULL_STEP,
		0
	};
	description.rotation = (struct linear_range){
		description.frequencies.byte_offset +
			description.frequencies.byte_count,
		ROTATION_ROW * ROWS_OF_A_FULL_STEP
	};
	const uint64_t arena = description.rotation.byte_offset +
			       description.rotation.byte_count;
	struct kernel_launch launch, untouched;
	char logged[CAPTURE_CAPACITY];
	context_text = "the rotation step";
	const struct attention_rotation_arguments record = {
		description.step.byte_offset,
		description.frequencies.byte_offset,
		description.rotation.byte_offset
	};
	expect(attention_rotation_prepare(&description, arena, &made_up_kernels,
					  &launch) == STATUS_OK,
	       "accepted");
	expect_launch(&launch, made_up_kernels.rotation, ROWS_OF_A_FULL_STEP, 1,
		      ATTENTION_ROTATION_THREADS, &record, sizeof(record),
		      "the rotation step's launch");
	enum {
		NO_DESCRIPTION,
		NO_KERNELS,
		NO_LAUNCH,
		NOT_PREPARED,
		NO_ROWS,
		MORE_ROWS,
		RESERVED,
		SHORT_STEP,
		SHORT_FREQUENCIES,
		SHORT_ROTATION,
		MISALIGNED,
		REFUSALS
	};
	static const char *const words[REFUSALS] = {
		"step: a missing argument",
		"step: a missing argument",
		"step: a missing argument",
		"step: the kernel attention_rotation is not prepared",
		"step: 0 rows: a launch takes 1 to",
		"step: 9 rows: a launch takes 1 to",
		"step: a reserved field",
		"the range of the step area holds",
		"the range of the frequencies holds",
		"the range of the rotation holds",
		"an offset is a multiple of"
	};
	struct attention_rotation_description refused[REFUSALS];
	struct attention_state_kernels unprepared = made_up_kernels;
	unprepared.rotation = BACKEND_NO_KERNEL;
	for (uint32_t refusal = 0; refusal < REFUSALS; refusal++) {
		refused[refusal] = description;
	}
    refused[NO_ROWS].row_count = 0;
    refused[MORE_ROWS].row_count = ROWS_OF_A_FULL_STEP + 1u;
    refused[RESERVED].reserved_zero = 1;
    refused[SHORT_STEP].step.byte_count -= 1u;
    refused[SHORT_FREQUENCIES].frequencies.byte_count -= 1u;
    refused[SHORT_ROTATION].rotation.byte_count -= 1u;
    refused[MISALIGNED].frequencies.byte_offset += MISALIGNMENT;
    for (size_t index = 0; index < COUNT_OF(refused); index++) {
        memset(&launch, UNTOUCHED_BYTE, sizeof(launch));
        memset(&untouched, UNTOUCHED_BYTE, sizeof(untouched));
        capture_begin();
	const result_status status = attention_rotation_prepare(
		index == NO_DESCRIPTION ? NULL : &refused[index], arena,
		index == NO_KERNELS   ? NULL :
		index == NOT_PREPARED ? &unprepared :
					&made_up_kernels,
		index == NO_LAUNCH ? NULL : &launch);
	capture_end(logged, sizeof(logged));
        expect(status == INVALID && memcmp(&launch, &untouched, sizeof(launch)) == 0, "refused, launch untouched");
        expect_one_error_line_with(logged, words[index], "one error line, the rule's");
    }
}

static void test_query_norm(void)
{
	const uint32_t rows = MODEL_WIDE_MOST_ROWS;
	struct attention_query_norm_description description = {
		{ 0, QUERY_RANK_ROW * rows },
		{ QUERY_RANK_ROW * rows, QUERY_NORM_BYTES },
		{ QUERY_RANK_ROW * rows + QUERY_NORM_BYTES,
		  QUERY_RANK_ROW * rows },
		rows,
		0
	};
	const uint64_t arena = description.query_normed.byte_offset +
			       description.query_normed.byte_count;
	struct kernel_launch launch, untouched;
	char logged[CAPTURE_CAPACITY];
	context_text = "qr alone";
	const struct attention_prepare_arguments record = {
		.query_rank_offset = description.query_rank.byte_offset,
		.query_norm_offset = description.query_norm.byte_offset,
		.query_normed_offset = description.query_normed.byte_offset
	};
	expect(attention_query_norm_prepare(&description, arena,
					    &made_up_kernels,
					    &launch) == STATUS_OK,
	       "accepted");
	expect_launch(
		&launch, made_up_kernels.prepare, rows, 1,
		ATTENTION_STATE_THREADS, &record, sizeof(record),
		"the prepare over the rows by 1: qr alone, its three offsets");
	enum {
		NO_DESCRIPTION,
		NO_KERNELS,
		NO_LAUNCH,
		NOT_PREPARED,
		NO_ROWS,
		MORE_ROWS,
		RESERVED,
		SHORT_RANK,
		LONG_NORM,
		SHORT_NORMED,
		MISALIGNED,
		SHARED,
		REFUSALS
	};
	static const char *const words[REFUSALS] = {
		"qr alone: a missing argument",
		"qr alone: a missing argument",
		"qr alone: a missing argument",
		"qr alone: the kernel attention_prepare_entry is not prepared",
		"qr alone: 0 rows: a launch takes 1 to",
		"qr alone: 4097 rows: a launch takes 1 to",
		"qr alone: a reserved field",
		"the range of wq_a's output holds",
		"its format stores",
		"the range of qr holds",
		"an offset is a multiple of",
		"share a byte"
	};
	struct attention_query_norm_description refused[REFUSALS];
	struct attention_state_kernels unprepared = made_up_kernels;
	unprepared.prepare = BACKEND_NO_KERNEL;
	for (uint32_t refusal = 0; refusal < REFUSALS; refusal++) {
		refused[refusal] = description;
	}
    refused[NO_ROWS].row_count = 0;
    refused[MORE_ROWS].row_count = rows + 1u;
    refused[RESERVED].reserved_zero = 1;
    refused[SHORT_RANK].query_rank.byte_count -= 1u;
    refused[LONG_NORM].query_norm.byte_count += ALIGNMENT;
    refused[SHORT_NORMED].query_normed.byte_count -= 1u;
    refused[MISALIGNED].query_norm.byte_offset += MISALIGNMENT;
    refused[SHARED].query_normed = description.query_rank;
    for (size_t index = 0; index < COUNT_OF(refused); index++) {
        memset(&launch, UNTOUCHED_BYTE, sizeof(launch));
        memset(&untouched, UNTOUCHED_BYTE, sizeof(untouched));
        capture_begin();
	const result_status status = attention_query_norm_prepare(
		index == NO_DESCRIPTION ? NULL : &refused[index], arena,
		index == NO_KERNELS   ? NULL :
		index == NOT_PREPARED ? &unprepared :
					&made_up_kernels,
		index == NO_LAUNCH ? NULL : &launch);
	capture_end(logged, sizeof(logged));
        expect(status == INVALID && memcmp(&launch, &untouched, sizeof(launch)) == 0, "refused, launch untouched");
        expect_one_error_line_with(logged, words[index], "one error line, the rule's");
    }
}

/* A description of the queries alone over rows, with the indexer, its ranges one behind another; its arena's end. */
static uint64_t
queries_of(uint32_t rows, struct attention_queries_description *description_out)
{
    const uint64_t byte_counts[] = {ROTATION_ROW * rows, STATE_ROW * HEADS * rows, STATE_ROW * HEADS * rows,
                                    INDEX_ROW * rows, INDEX_ROW * rows};
    struct linear_range *const ranges[] = { &description_out->rotation,
					    &description_out->heads,
					    &description_out->queries,
					    &description_out->index_heads,
					    &description_out->index_queries };
    uint64_t arena = 0;
    *description_out =
	    (struct attention_queries_description){ .row_count = rows,
						    .theta = 1,
						    .first_head = 0,
						    .head_count = HEADS,
						    .has_index = 1 };
    for (size_t index = 0; index < COUNT_OF(ranges); index++) {
	    *ranges[index] = (struct linear_range){ arena, byte_counts[index] };
	    arena += byte_counts[index];
    }
    return arena;
}

static void test_queries(void)
{
    static const struct {
        uint32_t theta, has_index, pool_ratio;
    } kinds[] = {
	    { 0, 0, 0 }, { 1, 0, 0 }, { 1, 1, 0 }, { 1, 1, MODEL_POOLED_RATIO }
    };
    static const uint32_t row_counts[] = {1, ROWS_OF_A_FULL_STEP};
    struct kernel_launch launch, untouched,
	    layer_launches[ATTENTION_STATE_LAUNCH_COUNT];
    char logged[CAPTURE_CAPACITY];
    context_text = "the queries alone of a slab's rows";
    for (size_t index = 0; index < COUNT_OF(kinds) * COUNT_OF(row_counts); index++) {
        const struct layer_case layer = layer_of(kinds[index / COUNT_OF(row_counts)].theta,
                                                 kinds[index / COUNT_OF(row_counts)].has_index,
                                                 kinds[index / COUNT_OF(row_counts)].pool_ratio,
                                                 row_counts[index % COUNT_OF(row_counts)], SESSIONS);
	const struct attention_state_description *const d = &layer.description;
	const struct attention_queries_description queries = {
		d->rotation,	  d->heads,	d->queries, d->index_heads,
		d->index_queries, d->row_count, d->theta,   d->first_head,
		d->head_count,	  d->has_index, 0
	};
	memset(&launch, UNTOUCHED_BYTE, sizeof(launch));
	expect(attention_state_prepare(d, layer.arena_byte_count,
				       &made_up_kernels,
				       layer_launches) == STATUS_OK &&
		       attention_queries_prepare(
			       &queries, layer.arena_byte_count,
			       &made_up_kernels, &launch) == STATUS_OK &&
		       memcmp(&launch, &layer_launches[ATTENTION_STATE_QUERIES],
			      sizeof(launch)) == 0,
	       "the layer's QUERIES launch, byte for byte");
    }
    const uint32_t rows = MODEL_WIDE_MOST_ROWS;
    struct attention_queries_description description;
    const uint64_t arena = queries_of(rows, &description);
    const struct attention_queries_arguments record = {
	    description.rotation.byte_offset,
	    description.heads.byte_offset,
	    description.queries.byte_offset,
	    description.index_heads.byte_offset,
	    description.index_queries.byte_offset,
	    0,
	    HEADS,
	    MODEL_INDEX_HEADS,
	    ATTENTION_ROTATION_POOLED,
	    HEADS,
	    0
    };
    context_text = "the queries alone of a chunk's rows";
    expect(attention_queries_prepare(&description, arena, &made_up_kernels,
				     &launch) == STATUS_OK,
	   "accepted");
    expect_launch(&launch, made_up_kernels.queries, HEADS + MODEL_INDEX_HEADS,
		  rows, ATTENTION_STATE_THREADS, &record, sizeof(record),
		  "the queries over their heads by the rows");
    enum { NO_DESCRIPTION, NO_KERNELS, NO_LAUNCH, NOT_PREPARED, NO_ROWS, MORE_ROWS, COMPACT, NO_HEADS, MORE_HEADS,
           PAST_THE_HEADS, THETA, INDEXER, INDEXER_OF_THETA_0, SHORT_ROTATION, SHORT_HEADS, SHORT_QUERIES,
           SHORT_INDEX_HEADS, SHORT_INDEX_QUERIES, MISALIGNED, SHARED, REFUSALS };
    static const char *const words[REFUSALS] = {
	    "queries alone: a missing argument",
	    "queries alone: a missing argument",
	    "queries alone: a missing argument",
	    "queries alone: the kernel attention_queries_entry is not prepared",
	    "queries alone: 0 rows: a launch takes 1 to",
	    "queries alone: 4097 rows: a launch takes 1 to",
	    ": 64 heads from head 0, theta 1, indexer 1, compact 2",
	    ": 0 heads from head 0, theta 1, indexer 1",
	    ": 65 heads from head 0, theta 1, indexer 1",
	    ": 64 heads from head 1, theta 1, indexer 1",
	    ": 64 heads from head 0, theta 2, indexer 1",
	    ": 64 heads from head 0, theta 1, indexer 2",
	    ": 64 heads from head 0, theta 0, indexer 1",
	    "the range of the rotation holds",
	    "the range of wq_b's output holds",
	    "the range of the queries holds",
	    "the range of the indexer's wq_b output holds",
	    "the range of the index queries holds",
	    "an offset is a multiple of",
	    "share a byte"
    };
    struct attention_queries_description refused[REFUSALS];
    struct attention_state_kernels unprepared = made_up_kernels;
    unprepared.queries = BACKEND_NO_KERNEL;
    for (uint32_t refusal = 0; refusal < REFUSALS; refusal++) {
        refused[refusal] = description;
    }
    refused[NO_ROWS].row_count = 0;
    refused[MORE_ROWS].row_count = rows + 1u;
    refused[COMPACT].heads_compact = 2;
    refused[NO_HEADS].head_count = 0;
    refused[MORE_HEADS].head_count = HEADS + 1u;
    refused[PAST_THE_HEADS].first_head = 1;
    refused[THETA].theta = MODEL_THETAS;
    refused[INDEXER].has_index = MODEL_POOLED_RATIO;
    refused[INDEXER_OF_THETA_0].theta = 0;
    refused[SHORT_ROTATION].rotation.byte_count -= 1u;
    refused[SHORT_HEADS].heads.byte_count -= 1u;
    refused[SHORT_QUERIES].queries.byte_count -= 1u;
    refused[SHORT_INDEX_HEADS].index_heads.byte_count -= 1u;
    refused[SHORT_INDEX_QUERIES].index_queries.byte_count -= 1u;
    refused[MISALIGNED].heads.byte_offset += MISALIGNMENT;
    refused[SHARED].queries = description.heads;
    for (size_t index = 0; index < COUNT_OF(refused); index++) {
        memset(&launch, UNTOUCHED_BYTE, sizeof(launch));
        memset(&untouched, UNTOUCHED_BYTE, sizeof(untouched));
        capture_begin();
	const result_status status = attention_queries_prepare(
		index == NO_DESCRIPTION ? NULL : &refused[index], arena,
		index == NO_KERNELS   ? NULL :
		index == NOT_PREPARED ? &unprepared :
					&made_up_kernels,
		index == NO_LAUNCH ? NULL : &launch);
	capture_end(logged, sizeof(logged));
        expect(status == INVALID && memcmp(&launch, &untouched, sizeof(launch)) == 0, "refused, launch untouched");
        expect_one_error_line_with(logged, words[index], "one error line, the rule's");
    }
    /* heads_compact: a rank's heads read from rows of its own heads alone (a TP rank's wq_b output where it wrote
     * it), the queries written at the heads' place in the full rows; one byte short refused */
    struct attention_queries_description compact = description;
    const uint32_t rank_heads = HEADS / 4u, rank_first = 2u * rank_heads;
    compact.first_head = rank_first;
    compact.head_count = rank_heads;
    compact.heads_compact = 1;
    compact.heads.byte_count = STATE_ROW * rank_heads * rows;
    const struct attention_queries_arguments compact_record = {
	    description.rotation.byte_offset,
	    description.heads.byte_offset,
	    description.queries.byte_offset,
	    description.index_heads.byte_offset,
	    description.index_queries.byte_offset,
	    rank_first,
	    rank_heads,
	    MODEL_INDEX_HEADS,
	    ATTENTION_ROTATION_POOLED,
	    rank_heads,
	    0
    };
    context_text = "the queries alone of a rank's compact heads";
    expect(attention_queries_prepare(&compact, arena, &made_up_kernels,
				     &launch) == STATUS_OK,
	   "accepted");
    expect_launch(&launch, made_up_kernels.queries,
		  rank_heads + MODEL_INDEX_HEADS, rows, ATTENTION_STATE_THREADS,
		  &compact_record, sizeof(compact_record),
		  "the rank's heads read from its compact rows");
    compact.heads.byte_count -= 1u;
    capture_begin();
    const result_status short_compact = attention_queries_prepare(
	    &compact, arena, &made_up_kernels, &launch);
    capture_end(logged, sizeof(logged));
    expect(short_compact == INVALID, "compact rows one byte short refused");
    expect_one_error_line_with(logged, "the range of wq_b's output holds", "one error line, the rule's");
    /* without the indexer the index ranges are not looked at */
    description.has_index = 0;
    description.index_heads = description.heads;
    description.index_queries = (struct linear_range){ 0, 0 };
    expect(attention_queries_prepare(&description, arena, &made_up_kernels,
				     &launch) == STATUS_OK &&
		   launch.grid[0] == HEADS,
	   "without the indexer: accepted, the index ranges not checked, the heads alone");
}

static void test_kernels(void)
{
	struct device_card *card = NULL;
	struct attention_state_kernels kernels;
	struct test_host_memory_backend_kernel kernel;
	char logged[CAPTURE_CAPACITY];
	static const char *const names[] = { KERNEL_ATTENTION_ROTATION,
					     KERNEL_ATTENTION_PREPARE,
					     KERNEL_ATTENTION_QUERIES,
					     KERNEL_ATTENTION_COMPRESSOR,
					     KERNEL_ATTENTION_KEY };
	static const uint32_t record_sizes[] = {
		ATTENTION_ROTATION_ARGUMENT_BYTES,
		ATTENTION_PREPARE_ARGUMENT_BYTES,
		ATTENTION_QUERIES_ARGUMENT_BYTES,
		ATTENTION_COMPRESSOR_ARGUMENT_BYTES,
		ATTENTION_KEY_ARGUMENT_BYTES
	};
	context_text = "the kernels on a card in host memory";
	test_host_memory_backend_reset(1, CARD_BYTES);
	expect(test_host_memory_backend.card_open(0, CARD_ARENA_BYTES, &card) ==
		       STATUS_OK,
	       "a card opens");
	capture_begin();
	expect(attention_state_kernels_prepare(NULL, card, &kernels) ==
			       INVALID &&
		       attention_state_kernels_prepare(
			       &test_host_memory_backend, NULL, &kernels) ==
			       INVALID &&
		       attention_state_kernels_prepare(
			       &test_host_memory_backend, card, NULL) ==
			       INVALID,
	       "a missing argument is refused");
	expect(attention_state_kernels_prepare(&test_host_memory_backend, card,
					       &kernels) == STATUS_UNSUPPORTED,
	       "a card of no class has no kernel");
	capture_end(logged, sizeof(logged));
	test_host_memory_backend.card_close(card, 0);
	test_host_memory_backend_reset(1, CARD_BYTES);
	test_host_memory_backend_set_arithmetic_class(0, CLASS_80);
	card = NULL;
	expect(test_host_memory_backend.card_open(0, CARD_ARENA_BYTES, &card) ==
			       STATUS_OK &&
		       attention_state_kernels_prepare(
			       &test_host_memory_backend, card, &kernels) ==
			       STATUS_OK,
	       "the kernels are prepared on a card of class 80");
	uint32_t handles[COUNT_OF(names)];
	_Static_assert(sizeof(handles) + DRAFT_HANDLE_BYTES == sizeof(kernels),
		       "a handle per kernel, the draft's");
	memcpy(handles, &kernels, sizeof(handles));
	expect(kernels.window == BACKEND_NO_KERNEL,
	       "the draft's kernel is left unprepared");
	for (uint32_t index = 0; index < COUNT_OF(names); index++) {
		const uint32_t handle = handles[index];
		expect(handle == index &&
			       test_host_memory_backend_kernel_of(0, handle,
								  &kernel) &&
			       strcmp(kernel.kernel_file_name,
				      KERNEL_FILE_ATTENTION_STATE) == 0 &&
			       strcmp(kernel.function_name, names[index]) ==
				       0 &&
			       kernel.argument_byte_count ==
				       record_sizes[index],
		       "a kernel of the file, in the order of the members, of its record's size");
	}
    test_host_memory_backend.card_close(card, 0);
}

static void test_reference(void)
{
    double cosine = 0.0, sine = 0.0;
    uint16_t halves[INDEX_HALVES], placed[INDEX_HALVES], key[KEY_HALVES], placed_key[KEY_HALVES];
    bool seen[INDEX_HALVES];
    context_text = "the reference";
    expect(test_attention_square_root(FOUR) == FOUR / MODEL_PAIR_VALUES &&
		   test_attention_square_root((double)MODEL_PAIR_VALUES) -
				   SQUARE_ROOT_OF_TWO <
			   DOUBLE_TOLERANCE &&
		   SQUARE_ROOT_OF_TWO - test_attention_square_root(
						(double)MODEL_PAIR_VALUES) <
			   DOUBLE_TOLERANCE,
	   "square roots");
    expect(test_attention_exponential(0.0) == 1.0 && test_attention_exponential(1.0) / EULER - 1.0 < DOUBLE_TOLERANCE &&
               1.0 - test_attention_exponential(1.0) / EULER < DOUBLE_TOLERANCE &&
               test_attention_exponential(-(double)CARD_ARENA_BYTES) == 0.0,
           "the exponential");
    test_attention_cosine_sine(0.0, &cosine, &sine);
    expect(cosine == 1.0 && sine == 0.0, "cosine and sine of 0");
    test_attention_cosine_sine(HALF_PI, &cosine, &sine);
    expect(cosine < DOUBLE_TOLERANCE && cosine > -DOUBLE_TOLERANCE && 1.0 - sine < DOUBLE_TOLERANCE,
           "cosine and sine of pi / 2");
    for (uint32_t index = 0; index < IDENTITY_ANGLES; index++) {
        test_attention_cosine_sine(IDENTITY_STEP * index, &cosine, &sine);
        const double one = cosine * cosine + sine * sine;
        expect(one - 1.0 < DOUBLE_TOLERANCE && 1.0 - one < DOUBLE_TOLERANCE, "cosine squared and sine squared");
    }
    test_attention_cosine_sine(LARGEST_ANGLE, &cosine, &sine);
    expect(cosine - COSINE_OF_IT < DOUBLE_TOLERANCE && COSINE_OF_IT - cosine < DOUBLE_TOLERANCE &&
               sine - SINE_OF_IT < DOUBLE_TOLERANCE && SINE_OF_IT - sine < DOUBLE_TOLERANCE,
           "cosine and sine of 2^20 - 1 radians");
    expect(test_attention_within(HALF_NEXT_AFTER_ONE, NEARER_THE_NEXT, NEARER_THE_NEXT, TEST_ATTENTION_HALF_BITS,
                                 TEST_ATTENTION_HALF_LOWEST) &&
               !test_attention_within(1.0, NEARER_THE_NEXT, NEARER_THE_NEXT, TEST_ATTENTION_HALF_BITS,
                                      TEST_ATTENTION_HALF_LOWEST) &&
               test_attention_within(1.0, HALF_PLACE_ABOVE_ONE, HALF_PLACE_ABOVE_ONE, TEST_ATTENTION_HALF_BITS,
                                     TEST_ATTENTION_HALF_LOWEST) &&
               test_attention_within(HALF_NEXT_AFTER_ONE, HALF_PLACE_ABOVE_ONE, HALF_PLACE_ABOVE_ONE,
                                     TEST_ATTENTION_HALF_BITS, TEST_ATTENTION_HALF_LOWEST),
           "the card tests take the nearest FP16 value, either at a tie, and refuse one cut toward zero");
    for (size_t index = 0; index < COUNT_OF(half_cases); index++) {
        expect(test_attention_half_nearest(half_cases[index].value) == half_cases[index].bits, "FP16 rounding");
    }
    expect(test_attention_half_value(HALF_ONE) == 1.0 && test_attention_half_nearest(-0.0) == HALF_MINUS_ZERO &&
               test_attention_half_ordinal(HALF_MINUS_SMALLEST) == -1 &&
               test_attention_half_ordinal(HALF_MINUS_ZERO) == 0,
           "FP16 values and ordinals");
    for (uint32_t index = 0; index < INDEX_HALVES; index++) {
        halves[index] = (uint16_t)index;
        seen[index] = false;
    }
    test_attention_layout_index_queries(halves, placed);
    for (uint32_t index = 0; index < INDEX_HALVES; index++) {
        expect(!seen[placed[index]], "the index queries' placement is a bijection");
        seen[placed[index]] = true;
    }
    for (size_t index = 0; index < COUNT_OF(query_places); index++) {
        expect(placed[query_places[index].half] == query_places[index].head * KEY_HALVES + query_places[index].dim,
               "a value of the index queries where model_layout.h places it");
    }
    for (uint32_t index = 0; index < KEY_HALVES; index++) {
        key[index] = (uint16_t)index;
        seen[index] = false;
    }
    test_attention_layout_key(key, placed_key);
    for (uint32_t index = 0; index < KEY_HALVES; index++) {
        expect(!seen[placed_key[index]], "a key's placement is a bijection");
        seen[placed_key[index]] = true;
    }
    for (size_t index = 0; index < COUNT_OF(key_places); index++) {
        expect(placed_key[key_places[index].half] == key_places[index].dim, "a key's dim where model_layout.h says");
    }
}

static void test_window(void)
{
	const uint32_t rows = MODEL_WIDE_MOST_ROWS;
	const uint64_t step = STEP_BLOCK + STEP_ROW * rows,
		       rotation = ROTATION_ROW * rows, input = STATE_ROW * rows;
	struct attention_window_description description = {
		{ 0, step },
		{ step, rotation },
		{ step + rotation, input },
		{ step + rotation + input, STATE_ROW },
		{ step + rotation + input + STATE_ROW,
		  RING_SESSION * SESSIONS },
		rows,
		SESSIONS,
		CONTEXT,
		0,
		0
	};
	const uint64_t arena =
		description.ring.byte_offset + description.ring.byte_count;
	struct kernel_launch launch, untouched;
	char logged[CAPTURE_CAPACITY];
	context_text = "the window alone";
	const struct attention_prepare_arguments record = {
		.step_offset = description.step.byte_offset,
		.rotation_offset = description.rotation.byte_offset,
		.window_input_offset = description.window_input.byte_offset,
		.window_norm_offset = description.window_norm.byte_offset,
		.ring_offset = description.ring.byte_offset,
		.ring_session_stride = RING_SESSION,
		.rotation_use = 0,
		.session_capacity = SESSIONS,
		.context_token_limit = CONTEXT
	};
	expect(attention_window_prepare(&description, arena, &made_up_kernels,
					&launch) == STATUS_OK,
	       "accepted");
	expect_launch(
		&launch, made_up_kernels.window, rows, 1,
		ATTENTION_STATE_THREADS, &record, sizeof(record),
		"the window kernel over the rows by 1: the prepare's record of the window part");
	enum {
		NO_DESCRIPTION,
		NO_KERNELS,
		NO_LAUNCH,
		NOT_PREPARED,
		NO_ROWS,
		MORE_ROWS,
		RESERVED,
		NO_SESSIONS,
		NO_GRANULE,
		USE,
		SHORT_STEP,
		SHORT_ROTATION,
		SHORT_INPUT,
		LONG_NORM,
		SHORT_RING,
		MISALIGNED,
		SHARED,
		REFUSALS
	};
	static const char *const words[REFUSALS] = {
		"the window alone: a missing argument",
		"the window alone: a missing argument",
		"the window alone: a missing argument",
		"the kernel attention_window is not prepared",
		"the window alone: 0 rows: a launch takes 1 to",
		"the window alone: 4097 rows: a launch takes 1 to",
		"the window alone: a reserved field",
		"0 sessions of",
		"tokens, rotation use",
		"rotation use 3",
		"the range of the step area holds",
		"the range of the rotation holds",
		"the range of wkv's output holds",
		"its format stores",
		"the range of the ring holds",
		"an offset is a multiple of",
		"share a byte"
	};
	struct attention_window_description refused[REFUSALS];
	struct attention_state_kernels unprepared = made_up_kernels;
	unprepared.window = BACKEND_NO_KERNEL;
	for (uint32_t refusal = 0; refusal < REFUSALS; refusal++) {
		refused[refusal] = description;
	}
    refused[NO_ROWS].row_count = 0;
    refused[MORE_ROWS].row_count = rows + 1u;
    refused[RESERVED].reserved_zero = 1;
    refused[NO_SESSIONS].session_capacity = 0;
    refused[NO_GRANULE].context_token_limit = CONTEXT + 1u;
    refused[USE].rotation_use = ATTENTION_ROTATION_GROUP_START + 1u;
    refused[SHORT_STEP].step.byte_count -= 1u;
    refused[SHORT_ROTATION].rotation.byte_count -= 1u;
    refused[SHORT_INPUT].window_input.byte_count -= 1u;
    refused[LONG_NORM].window_norm.byte_count += ALIGNMENT;
    refused[SHORT_RING].ring.byte_count -= 1u;
    refused[MISALIGNED].window_norm.byte_offset += MISALIGNMENT;
    refused[SHARED].ring.byte_offset = description.window_input.byte_offset;
    for (size_t index = 0; index < COUNT_OF(refused); index++) {
        const bool is_unprepared = index == NOT_PREPARED;
        memset(&launch, UNTOUCHED_BYTE, sizeof(launch));
        memset(&untouched, UNTOUCHED_BYTE, sizeof(untouched));
        capture_begin();
	const result_status status = attention_window_prepare(
		index == NO_DESCRIPTION ? NULL : &refused[index], arena,
		index == NO_KERNELS ? NULL :
		is_unprepared	    ? &unprepared :
				      &made_up_kernels,
		index == NO_LAUNCH ? NULL : &launch);
	capture_end(logged, sizeof(logged));
        expect(status == INVALID && memcmp(&launch, &untouched, sizeof(launch)) == 0, "refused, launch untouched");
        expect_one_error_line_with(logged, words[index], "one error line, the rule's");
    }

    context_text = "the reference of the window row";
    static uint16_t ones[WINDOW_DIMS], ring_row[WINDOW_DIMS];
    static float angles[WINDOW_ROTATION_FLOATS];
    for (uint32_t dim = 0; dim < WINDOW_DIMS; dim++) {
        ones[dim] = BFLOAT16_ONE;
    }
    for (uint32_t word = 0; word < WINDOW_ROTATION_FLOATS;
	 word += MODEL_PAIR_VALUES) {
	    angles[word] = 1.0f; /* (cos, sin) = (1, 0): no turn */
	    angles[word + 1u] = 0.0f;
    }
    const float operand = test_attention_window_operand(ones);
    test_attention_window_row(ones, ones, 1.0f, angles, ring_row);
    bool holds = operand == 1.0f + (float)MODEL_NORM_EPSILON;
    for (uint32_t dim = 0; dim < WINDOW_DIMS; dim++) {
        holds = holds && ring_row[dim] == HALF_OF_ONE;
    }
    expect(holds, "a row of ones: the operand 1 + 1e-20, the row ones");
}

int main(void)
{
    capture_descriptor = mkstemp(capture_path);
    error_stream_descriptor = dup(STDERR_FILENO);
    if (capture_descriptor < 0 || error_stream_descriptor < 0 || unlink(capture_path) != 0) {
        (void)printf("FAILED: the temporary file for the error stream\n");
        return EXIT_FAILURE;
    }
    test_layers();
    test_layer_refusals();
    test_rotation();
    test_query_norm();
    test_queries();
    test_window();
    test_kernels();
    test_reference();
    (void)close(capture_descriptor);
    (void)close(error_stream_descriptor);
    if (failures != 0) {
        (void)printf("attention_state_test FAILED: %d failures\n", failures);
        return EXIT_FAILURE;
    }
    (void)printf("attention_state_test passed\n");
    return EXIT_SUCCESS;
}
